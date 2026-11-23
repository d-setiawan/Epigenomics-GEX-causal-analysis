#!/usr/bin/env python3
"""Train/development benchmark for thresholded CA-BOSS v4 candidates.

Expensive nuisance fits, residual permutations, and bootstraps are computed once
per nested matrix.  Candidate configurations only re-threshold cached evidence,
which makes empirical-null calibration practical without reopening truth during
discovery.  Truth enters only after each six-run world consensus is frozen.
Formal DEV/LOCKED validation uses the exported two-phase API to freeze every
registered world's discovery result before any scorer domain is opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import hashlib
import io
import json
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from channel_aware_boss_v4 import (
    CABossV4Config,
    apply_evidence_thresholds,
    build_typed_graph,
    channel_modality,
    discover_v4,
    fit_cell_loading_group_evidence,
    fit_hierarchical_cell_evidence,
    node_channel,
)
from consensus_ca_boss_v4 import (
    attach_group_projection_once,
    consensus_typed_graph,
    direct_collider_children,
    replace_pairwise_state_layer,
)
from crossfit_hierarchical_collider_v4 import (
    CrossfitColliderWorldEvidence,
    CrossfitHierarchicalColliderConfig,
    CrossfitHierarchicalColliderResult,
    fit_crossfit_collider_world_evidence,
    resolve_crossfit_hierarchical_colliders_v4,
)
from loading_group_projection_v4 import (
    LoadingProjectionConfig,
    estimate_loading_group_projection,
    make_evidence_view,
)
from observed_collider_structure import (
    ObservedColliderStructureConfig,
    ObservedColliderStructureResult,
    apply_observed_collider_refinements,
    select_observed_collider_refinements,
)
from latent_axis_clustering_v4 import (
    LatentAxisClusteringConfig,
    LatentAxisClusteringResult,
    estimate_latent_axis_clustering,
)
from prepare_ca_boss_v4_inputs import MODALITIES, PrepareConfig, prepare_inputs
from score_latent_loading_groups_v4 import score_latent_loading_groups
from score_multimodal_v4_graph import score_v4
from stratified_beta_binomial_edges import (
    StratifiedBetaBinomialConfig,
    estimate_from_observed_tables as estimate_beta_binomial_from_observed_tables,
    frame_sha256,
)
from structure_aware_collider_motif import (
    ColliderMotifConfig,
    ColliderMotifResult,
    select_unique_collider_motifs,
)
from structure_anchored_loading_groups_v4 import (
    StructureAnchoredGroupConfig,
    StructureAnchoredGroupResult,
    estimate_structure_anchored_groups,
)


COLLIDER_MOTIF_VOTES = 6
COLLIDER_MOTIF_SEED_STRIDE = 10_007
FROZEN_BETA_BINOMIAL_CONFIG = StratifiedBetaBinomialConfig()
FROZEN_COLLIDER_MOTIF_CONFIG = ColliderMotifConfig()
FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG = ObservedColliderStructureConfig()
FROZEN_STRUCTURE_ANCHORED_GROUP_CONFIG = StructureAnchoredGroupConfig()
FROZEN_CROSSFIT_HIERARCHICAL_COLLIDER_CONFIG = (
    CrossfitHierarchicalColliderConfig()
)
_DIRECT_EDGE_TYPE = "within_assay_direct_candidate"
_FORBIDDEN_DISCOVERY_PATH_TOKENS = {
    "scorer",
    "truth",
    "oracle",
    "counterfactual",
    "role_assignments",
    "generative_dag",
}
BENCHMARK_MANIFEST_SCHEMA_VERSION = "ca_boss_v4.benchmark.v5"
WORLD_INPUT_BINDING_SCHEMA_VERSION = "ca_boss_v4.world_input_binding.v1"
GLOBAL_PRETRUTH_FREEZE_SCHEMA_VERSION = "ca_boss_v4.global_pretruth_freeze.v1"
RUN_SUMMARY_SCHEMA_VERSION = "multimodal_simulation_v4.run.v2"
DISCOVERY_MANIFEST_SCHEMA_VERSION = "multimodal_simulation_v4.discovery.v2"
SCORER_MANIFEST_SCHEMA_VERSION = "multimodal_simulation_v4.scorer.v2"
_WORLD_DIRECTORY_PATTERN = re.compile(r"^world_seed(\d{4,8})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenWorldPretruth:
    """Serializable handle to one discovery-only frozen world result."""

    world_dir: str
    world_name: str
    world_out: str
    candidate_names: tuple[str, ...]
    pretruth_manifest: str
    pretruth_manifest_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes_once(path: Path, *, label: str) -> bytes:
    """Open one non-symlink regular file once for both hashing and parsing."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Could not open {label} safely: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular, non-symlink file: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_regular_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular, non-symlink file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _expected_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return normalized


def _validate_content_addressed_domain(
    root: Path,
    manifest: dict[str, Any],
    *,
    label: str,
) -> None:
    """Verify an exact, non-symlink payload set against manifest hashes."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} must be a regular, non-symlink directory")
    files = manifest.get("files")
    hashes = manifest.get("file_sha256")
    if not isinstance(files, list) or not isinstance(hashes, dict):
        raise ValueError(f"{label} lacks its content-addressed payload contract")
    listed = [str(value) for value in files]
    if len(listed) != len(set(listed)) or set(listed) != set(map(str, hashes)):
        raise ValueError(f"{label} file list and hash keys do not match exactly")
    for relative in listed:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative == "manifest.json"
        ):
            raise ValueError(f"{label} contains an unsafe path: {relative!r}")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symlink: {path}")
        if path.is_file():
            relative = str(path.relative_to(root))
            if relative != "manifest.json":
                actual.add(relative)
    if actual != set(listed):
        raise ValueError(
            f"{label} payload set differs from its manifest: "
            f"missing={sorted(set(listed) - actual)}, "
            f"extra={sorted(actual - set(listed))}"
        )
    for relative in listed:
        expected = _expected_sha256(
            hashes[relative], label=f"{label} hash for {relative}"
        )
        if _file_sha256(root / relative) != expected:
            raise ValueError(f"{label} payload hash mismatch: {relative}")


def validate_world_input_binding(
    world_dir: Path | str,
    *,
    verify_scorer_payload: bool,
) -> dict[str, Any]:
    """Bind a world label to current content-addressed simulator payloads.

    Discovery is always verified.  The scorer domain is deliberately unopened
    when ``verify_scorer_payload`` is false so the benchmark can establish its
    input identity before fitting without crossing the truth boundary.  A full
    binding is required immediately after the pretruth consensus freeze.
    """

    world = Path(world_dir).absolute()
    if world.is_symlink() or not world.is_dir():
        raise ValueError(f"World must be a regular, non-symlink directory: {world}")
    match = _WORLD_DIRECTORY_PATTERN.fullmatch(world.name)
    if match is None:
        raise ValueError(
            f"World directory must be named world_seed<seed>: {world.name!r}"
        )
    seed = int(match.group(1))
    run_path = world / "run_summary.json"
    run_summary = _read_regular_json(run_path, label=f"world {seed} run summary")
    if run_summary.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"World {seed} has a stale or unsupported run-summary schema")
    if int(run_summary.get("seed", -1)) != seed:
        raise ValueError(f"World {seed} run-summary seed does not match its directory")
    if (
        run_summary.get("discovery_directory") != "discovery"
        or run_summary.get("scorer_directory") != "scorer"
        or run_summary.get("information_domains_separated_by_directory") is not True
        or run_summary.get("raw_only_preparer_required_for_discovery") is not True
        or run_summary.get("domain_payloads_content_addressed") is not True
    ):
        raise ValueError(f"World {seed} lacks the v2 information-boundary contract")
    if int(run_summary.get("n_modalities", -1)) != len(MODALITIES):
        raise ValueError(f"World {seed} has the wrong modality count")

    discovery_root = world / "discovery"
    discovery_path = discovery_root / "manifest.json"
    discovery = _read_regular_json(
        discovery_path, label=f"world {seed} discovery manifest"
    )
    if discovery.get("schema_version") != DISCOVERY_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"World {seed} has a stale or unsupported discovery schema")
    modalities = discovery.get("modalities")
    if (
        not isinstance(modalities, list)
        or len(modalities) != len(MODALITIES)
        or set(map(str, modalities)) != set(MODALITIES)
    ):
        raise ValueError(f"World {seed} discovery modalities do not match v4")
    discovery_hash = _file_sha256(discovery_path)
    expected_discovery_hash = _expected_sha256(
        run_summary.get("discovery_manifest_sha256"),
        label=f"world {seed} discovery-manifest hash",
    )
    if discovery_hash != expected_discovery_hash:
        raise ValueError(
            f"World {seed} discovery manifest is not bound to its run summary"
        )
    _validate_content_addressed_domain(
        discovery_root,
        discovery,
        label=f"world {seed} discovery domain",
    )

    scorer_hash = _expected_sha256(
        run_summary.get("scorer_manifest_sha256"),
        label=f"world {seed} scorer-manifest hash",
    )
    scorer_schema: str | None = None
    scorer_verified = False
    if verify_scorer_payload:
        scorer_root = world / "scorer"
        scorer_path = scorer_root / "manifest.json"
        scorer = _read_regular_json(
            scorer_path, label=f"world {seed} scorer manifest"
        )
        if scorer.get("schema_version") != SCORER_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"World {seed} has a stale or unsupported scorer schema")
        if (
            scorer.get("scorer_only") is not True
            or scorer.get("must_not_be_read_by_discovery") is not True
        ):
            raise ValueError(f"World {seed} scorer information boundary is malformed")
        observed_scorer_hash = _file_sha256(scorer_path)
        if observed_scorer_hash != scorer_hash:
            raise ValueError(
                f"World {seed} scorer manifest is not bound to its run summary"
            )
        _validate_content_addressed_domain(
            scorer_root,
            scorer,
            label=f"world {seed} scorer domain",
        )
        scorer_schema = str(scorer["schema_version"])
        scorer_verified = True

    binding: dict[str, Any] = {
        "schema_version": WORLD_INPUT_BINDING_SCHEMA_VERSION,
        "seed": seed,
        "world": str(world),
        "run_summary": {
            "path": str(run_path),
            "schema_version": str(run_summary["schema_version"]),
            "sha256": _file_sha256(run_path),
        },
        "discovery_manifest": {
            "path": str(discovery_path),
            "schema_version": str(discovery["schema_version"]),
            "sha256": discovery_hash,
            "payloads_content_addressed_verified": True,
        },
        "scorer_manifest": {
            "path": str(world / "scorer" / "manifest.json"),
            "schema_version": scorer_schema,
            "sha256": scorer_hash,
            "payloads_content_addressed_verified": scorer_verified,
        },
        "directory_seed_matches_run_summary": True,
        "information_domains_separated": True,
        "scorer_payload_verification_performed": bool(verify_scorer_payload),
    }
    binding["binding_payload_sha256"] = _canonical_sha256(binding)
    return binding


def _assert_discovery_artifact(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    forbidden = sorted(lowered & _FORBIDDEN_DISCOVERY_PATH_TOKENS)
    forbidden.extend(
        token
        for token in _FORBIDDEN_DISCOVERY_PATH_TOKENS
        if token in path.name.lower() and token not in forbidden
    )
    if forbidden:
        raise ValueError(
            f"Collider-motif evidence accepts discovery artifacts only: {path} "
            f"matched {sorted(set(forbidden))}"
        )


def load_frozen_state_membership(path: Path) -> tuple[pd.DataFrame, str]:
    """Load one content-addressed discovery-only state-membership table."""

    lexical = Path(path).absolute()
    _assert_discovery_artifact(lexical)
    payload = _read_regular_bytes_once(lexical, label="frozen state membership")
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse frozen state membership: {exc}") from exc
    forbidden_tokens = ("truth", "role", "oracle", "scorer", "generative")
    forbidden_columns = sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in forbidden_tokens)
    )
    if forbidden_columns:
        raise ValueError(
            "Frozen state membership contains forbidden semantic columns: "
            f"{forbidden_columns}"
        )
    required = {
        "feature",
        "modality",
        "selection_status",
        "feature_global_maxT_confirmation_gate",
        "assay_global_maxT_admission_gate",
        "conditional_wald_view_support",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen state membership lacks columns: {missing}")
    if frame.empty or frame["feature"].astype(str).duplicated().any():
        raise ValueError("Frozen state membership must have unique nonempty features")
    return frame, _bytes_sha256(payload)


def _chrom_feature_columns(path: Path) -> list[str]:
    _assert_discovery_artifact(path)
    return [
        str(column)
        for column in pd.read_csv(path, sep="\t", nrows=0).columns
        if str(column).startswith("chr")
    ]


def fit_collider_motif_vote_evidence(
    chrom_cell_tables: list[Path],
    embedding: Path,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, list[dict[str, Any]]]]:
    """Fit six discovery-only beta-binomial votes from one prepared embedding.

    The frozen statistical configuration is changed only by the deterministic
    ``vote + 10_007 * (assay_index + 1)`` random seed.  Evidence is fitted once
    per world and reused across every threshold candidate.
    """

    _assert_discovery_artifact(embedding)
    tables = sorted(Path(path) for path in chrom_cell_tables)
    if len(tables) != 6 or any("observed_focal__chrom_" not in path.name for path in tables):
        raise ValueError("Collider-motif fitting requires exactly six chromatin focal tables")
    for path in tables:
        _assert_discovery_artifact(path)
    vote_frames: dict[int, pd.DataFrame] = {}
    vote_metadata: dict[int, list[dict[str, Any]]] = {}
    for vote in range(COLLIDER_MOTIF_VOTES):
        assay_frames: list[pd.DataFrame] = []
        metadata: list[dict[str, Any]] = []
        for assay_index, table in enumerate(tables):
            source_assay = table.stem.removeprefix("observed_focal__")
            random_state = vote + COLLIDER_MOTIF_SEED_STRIDE * (assay_index + 1)
            result = estimate_beta_binomial_from_observed_tables(
                table,
                embedding,
                feature_columns=_chrom_feature_columns(table),
                config=replace(
                    FROZEN_BETA_BINOMIAL_CONFIG,
                    random_state=random_state,
                ),
            )
            evidence = result.edge_evidence.copy()
            evidence["source_assay"] = source_assay
            evidence["vote"] = vote
            assay_frames.append(evidence)
            metadata.append(
                {
                    "source_assay": source_assay,
                    "vote": vote,
                    "random_state": random_state,
                    **result.metadata,
                }
            )
        vote_frames[vote] = pd.concat(assay_frames, ignore_index=True).sort_values(
            ["source_assay", "node1", "node2"]
        ).reset_index(drop=True)
        vote_metadata[vote] = metadata
    combined = pd.concat(
        [vote_frames[vote] for vote in range(COLLIDER_MOTIF_VOTES)],
        ignore_index=True,
    )
    return combined, vote_frames, vote_metadata


def write_collider_motif_vote_evidence(
    world_out: Path,
    vote_frames: dict[int, pd.DataFrame],
    vote_metadata: dict[int, list[dict[str, Any]]],
) -> dict[str, str]:
    """Persist discovery evidence and return byte hashes for pretruth freezing."""

    evidence_out = world_out / "collider_motif_evidence"
    evidence_out.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for vote in range(COLLIDER_MOTIF_VOTES):
        evidence_path = evidence_out / f"beta_binomial_vote{vote}.tsv"
        metadata_path = evidence_out / f"beta_binomial_vote{vote}_metadata.json"
        vote_frames[vote].to_csv(evidence_path, sep="\t", index=False)
        metadata_path.write_text(
            json.dumps(vote_metadata[vote], indent=2, sort_keys=True, default=str)
            + "\n"
        )
        hashes[str(evidence_path.relative_to(world_out))] = _file_sha256(evidence_path)
        hashes[str(metadata_path.relative_to(world_out))] = _file_sha256(metadata_path)
    return hashes


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _direct_consensus_for_motif(preliminary_consensus: pd.DataFrame) -> pd.DataFrame:
    direct = preliminary_consensus.loc[
        preliminary_consensus.get("edge_type", pd.Series(dtype=str))
        .astype(str)
        .eq(_DIRECT_EDGE_TYPE)
    ].copy()
    if direct.empty:
        return pd.DataFrame(columns=["source_assay", "node1", "node2"])
    assays: list[str] = []
    for record in direct.to_dict(orient="records"):
        left_channel = node_channel(str(record["node1"]))
        right_channel = node_channel(str(record["node2"]))
        if left_channel != right_channel or left_channel == "RNA":
            raise ValueError("A within-assay direct candidate must have one chromatin channel")
        assays.append(channel_modality(left_channel))
    direct["source_assay"] = assays
    direct = direct[["source_assay", *[column for column in direct if column != "source_assay"]]]
    return direct


def replace_direct_layer_with_collider_motifs(
    preliminary_consensus: pd.DataFrame,
    motif: ColliderMotifResult,
) -> pd.DataFrame:
    """Replace the preliminary direct layer with selected/unresolved motifs.

    Candidate pairs classified by the motif layer take precedence over the
    preliminary pairwise row.  This avoids duplicate adjacencies; selected
    arms become oriented direct candidates.  A rejected edge is retained in
    the ordinary graph as ``unresolved`` only when it was already selected by
    the preliminary nested consensus.  Rescue-only rejections remain in motif
    diagnostics and never become skeleton adjacencies.  Unrelated state and
    prior rows are unchanged and can subsequently be replaced by loading-group
    projections.
    """

    original = preliminary_consensus.copy()
    original_map = {
        _canonical_pair(str(record["node1"]), str(record["node2"])): record
        for record in original.to_dict(orient="records")
    }
    selected_records = (
        motif.direct_candidates.to_dict(orient="records")
        if not motif.direct_candidates.empty
        else []
    )
    diagnostic_unresolved_records = (
        motif.demoted_or_unresolved.to_dict(orient="records")
        if not motif.demoted_or_unresolved.empty
        else []
    )
    unresolved_records = [
        record
        for record in diagnostic_unresolved_records
        if bool(record.get("was_base_selected", False))
    ]
    motif_pairs = {
        _canonical_pair(str(record["node1"]), str(record["node2"]))
        for record in [*selected_records, *unresolved_records]
    }
    retained = original.loc[
        ~original.apply(
            lambda row: _canonical_pair(str(row["node1"]), str(row["node2"]))
            in motif_pairs
            or str(row.get("edge_type", "")) == _DIRECT_EDGE_TYPE,
            axis=1,
        )
    ].copy()

    additions: list[dict[str, Any]] = []
    for status, records in (
        ("selected_unique_wedge", selected_records),
        ("demoted_unresolved", unresolved_records),
    ):
        for record in records:
            key = _canonical_pair(str(record["node1"]), str(record["node2"]))
            base = dict(original_map.get(key, {}))
            base.update(
                {
                    "node1": key[0],
                    "node2": key[1],
                    "endpoint1": str(record.get("endpoint1", "CIRCLE")),
                    "endpoint2": str(record.get("endpoint2", "CIRCLE")),
                    "edge_type": (
                        _DIRECT_EDGE_TYPE
                        if status == "selected_unique_wedge"
                        else "unresolved"
                    ),
                    "interpretation": str(
                        record.get(
                            "interpretation",
                            "collider motif candidate unresolved after post-consensus selection",
                        )
                    ),
                    "orientation_source": str(
                        record.get(
                            "orientation_source",
                            "collider_motif_unresolved",
                        )
                    ),
                    "selection_count": int(base.get("selection_count", 0)),
                    "selection_frequency": base.get("selection_frequency", float("nan")),
                    "n_nested_runs": int(base.get("n_nested_runs", COLLIDER_MOTIF_VOTES)),
                    "edge_type_vote_count": int(base.get("edge_type_vote_count", 0)),
                    "endpoint1_vote_count": int(base.get("endpoint1_vote_count", 0)),
                    "endpoint2_vote_count": int(base.get("endpoint2_vote_count", 0)),
                    "endpoint_vote_count": int(base.get("endpoint_vote_count", 0)),
                    "collider_motif_status": status,
                    "collider_motif_component_id": str(record.get("component_id", "")),
                    "collider_motif_center": str(record.get("collider_center", "")),
                    "collider_motif_wedge_score": record.get("wedge_score", float("nan")),
                    "collider_motif_wedge_margin": record.get("wedge_margin", float("nan")),
                    "collider_motif_vote_fraction": record.get(
                        "selected_vote_fraction", float("nan")
                    ),
                    "collider_motif_empirical_p_positive": record.get(
                        "mean_empirical_p_positive", float("nan")
                    ),
                    "collider_motif_pooled_log_odds": record.get(
                        "mean_pooled_log_odds", float("nan")
                    ),
                    "collider_motif_unresolved_reason": str(record.get("reason", "")),
                }
            )
            if status == "demoted_unresolved":
                base["endpoint1"] = "CIRCLE"
                base["endpoint2"] = "CIRCLE"
                base["orientation_source"] = "collider_motif_unresolved"
            additions.append(base)
    if additions:
        retained = pd.concat(
            [retained, pd.DataFrame(additions)], ignore_index=True, sort=False
        )
    if retained.empty:
        return retained
    if retained.apply(
        lambda row: _canonical_pair(str(row["node1"]), str(row["node2"])), axis=1
    ).duplicated().any():
        raise ValueError("Collider-motif replacement produced duplicate adjacencies")
    return retained.sort_values(["edge_type", "node1", "node2"]).reset_index(drop=True)


def resolve_postconsensus_candidate(
    preliminary_consensus: pd.DataFrame,
    beta_vote_evidence: pd.DataFrame,
    loading_views: list[Any],
    cfg: CABossV4Config,
    projection_cache: dict[tuple[Any, ...], Any],
    axis_cache: dict[tuple[Any, ...], LatentAxisClusteringResult]
    | None = None,
    axis_eligible_modalities: tuple[str, ...] | None = None,
    axis_loading_views: list[Any] | None = None,
    loading_coordinates_cross_assay_aligned: bool = True,
    base_edge_views: Mapping[str, pd.DataFrame] | None = None,
    residual_views: Mapping[str, pd.DataFrame] | None = None,
    feature_model_views: Mapping[str, pd.DataFrame] | None = None,
    state_membership: pd.DataFrame | None = None,
    crossfit_world_evidence: CrossfitColliderWorldEvidence | None = None,
) -> tuple[
    pd.DataFrame,
    ColliderMotifResult,
    Any,
    LatentAxisClusteringResult,
    ObservedColliderStructureResult | None,
    StructureAnchoredGroupResult | None,
    CrossfitHierarchicalColliderResult | None,
]:
    """Resolve direct motifs, local groups, then assay-spanning latent axes.

    Cross-assay group similarities are represented by n-ary axis hyperedges,
    not ordinary pairwise skeleton edges.  Only selected within-assay group
    projections are attached to the pair graph.
    """

    motif = select_unique_collider_motifs(
        _direct_consensus_for_motif(preliminary_consensus),
        beta_vote_evidence,
        FROZEN_COLLIDER_MOTIF_CONFIG,
    )
    refinement_inputs = (
        base_edge_views,
        residual_views,
        feature_model_views,
    )
    if any(value is not None for value in refinement_inputs) and not all(
        value is not None for value in refinement_inputs
    ):
        raise ValueError(
            "Observed collider refinement requires all three view mappings"
        )
    refinement: ObservedColliderStructureResult | None = None
    if all(value is not None for value in refinement_inputs):
        assert base_edge_views is not None
        assert residual_views is not None
        assert feature_model_views is not None
        refinement = select_observed_collider_refinements(
            motif.component_diagnostics,
            motif.wedge_diagnostics,
            motif.demoted_or_unresolved,
            beta_vote_evidence,
            base_edge_views,
            residual_views,
            feature_model_views,
            FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG,
            loading_pair_views=(
                {
                    str(view.view_id): view.pair_evidence
                    for view in loading_views
                }
                if state_membership is not None
                else None
            ),
            state_membership=state_membership,
        )
        motif = apply_observed_collider_refinements(motif, refinement)
    crossfit_result: CrossfitHierarchicalColliderResult | None = None
    if crossfit_world_evidence is not None:
        if not all(value is not None for value in refinement_inputs):
            raise ValueError(
                "Crossfit hierarchical collider selection requires all observed "
                "evidence view mappings"
            )
        if state_membership is None:
            raise ValueError(
                "Crossfit hierarchical collider selection requires frozen state "
                "membership"
            )
        assert base_edge_views is not None
        assert residual_views is not None
        assert feature_model_views is not None
        crossfit_result = resolve_crossfit_hierarchical_colliders_v4(
            world_evidence=crossfit_world_evidence,
            beta_vote_evidence=beta_vote_evidence,
            base_edge_views=base_edge_views,
            residual_views=residual_views,
            feature_model_views=feature_model_views,
            loading_pair_views={
                str(view.view_id): view.pair_evidence for view in loading_views
            },
            state_membership=state_membership,
            current_direct_candidates=motif.direct_candidates,
            current_demoted_or_unresolved=motif.demoted_or_unresolved,
            config=FROZEN_CROSSFIT_HIERARCHICAL_COLLIDER_CONFIG,
        )
        motif = ColliderMotifResult(
            direct_candidates=crossfit_result.direct_candidates,
            demoted_or_unresolved=crossfit_result.demoted_or_unresolved,
            component_diagnostics=motif.component_diagnostics,
            wedge_diagnostics=motif.wedge_diagnostics,
            aggregated_evidence=motif.aggregated_evidence,
            metadata={
                **motif.metadata,
                "crossfit_hierarchical_collider_applied": True,
                "crossfit_hierarchical_collider_schema": (
                    crossfit_result.provenance.get("schema_version")
                ),
                "crossfit_hierarchical_selected_replacements": int(
                    crossfit_result.provenance.get("n_selected_replacements", 0)
                ),
            },
        )
    direct_resolved = replace_direct_layer_with_collider_motifs(
        preliminary_consensus,
        motif,
    )
    excluded = (
        state_group_exclusions(direct_resolved, motif)
        if cfg.exclude_direct_collider_children_from_state_groups
        else ()
    )
    forbidden_parent_pairs = direct_collider_parent_pairs(motif)
    evidence_projection_key = (
        "ungated_loading_evidence",
        excluded,
        forbidden_parent_pairs,
        float(cfg.group_projection_empirical_q),
        float(cfg.loading_min_abs_cosine),
        float(cfg.group_projection_min_relative_margin),
    )
    if evidence_projection_key not in projection_cache:
        projection_cache[evidence_projection_key] = estimate_loading_group_projection(
            loading_views,
            LoadingProjectionConfig(
                min_within_absolute_cosine=cfg.loading_min_abs_cosine,
                min_group_relative_margin=(
                    cfg.group_projection_min_relative_margin
                ),
                cross_projection_empirical_q=cfg.group_projection_empirical_q,
                allow_cross_assay_group_matching=(
                    loading_coordinates_cross_assay_aligned
                ),
            ),
            excluded_features=excluded,
            forbidden_within_pairs=forbidden_parent_pairs,
        )
    evidence_projection = projection_cache[evidence_projection_key]
    structure_anchor: StructureAnchoredGroupResult | None = None
    permitted_pairs: tuple[tuple[str, str], ...] | None = None
    if state_membership is not None:
        pair_view_frames: list[pd.DataFrame] = []
        for view in loading_views:
            frame = view.pair_evidence.copy()
            frame["view_id"] = str(view.view_id)
            pair_view_frames.append(frame)
        structure_anchor = estimate_structure_anchored_groups(
            motif.direct_candidates,
            evidence_projection.within_group_candidates,
            pd.concat(pair_view_frames, ignore_index=True),
            state_membership,
            FROZEN_STRUCTURE_ANCHORED_GROUP_CONFIG,
        )
        permitted_pairs = tuple(
            sorted(
                tuple(sorted(str(row.members).split(";")))
                for row in structure_anchor.typed_groups.itertuples(index=False)
            )
        )
        projection_key = (
            "layered_state_sibling_structure_anchor",
            excluded,
            forbidden_parent_pairs,
            permitted_pairs,
            float(cfg.group_projection_empirical_q),
            float(cfg.loading_min_abs_cosine),
            float(cfg.group_projection_min_relative_margin),
        )
        if projection_key not in projection_cache:
            projection_cache[projection_key] = estimate_loading_group_projection(
                loading_views,
                LoadingProjectionConfig(
                    min_within_absolute_cosine=cfg.loading_min_abs_cosine,
                    min_group_relative_margin=(
                        cfg.group_projection_min_relative_margin
                    ),
                    cross_projection_empirical_q=cfg.group_projection_empirical_q,
                    allow_cross_assay_group_matching=(
                        loading_coordinates_cross_assay_aligned
                    ),
                ),
                excluded_features=excluded,
                forbidden_within_pairs=forbidden_parent_pairs,
                permitted_within_pairs=permitted_pairs,
            )
        group_projection = projection_cache[projection_key]
    else:
        projection_key = evidence_projection_key
        group_projection = evidence_projection
    eligible_modalities = tuple(
        sorted(axis_eligible_modalities or tuple(MODALITIES))
    )
    axis_key = (*projection_key, eligible_modalities)
    axis_store = axis_cache if axis_cache is not None else {}
    if axis_key not in axis_store:
        axis_groups = group_projection.groups.loc[
            group_projection.groups["modality"].astype(str).isin(
                eligible_modalities
            )
        ].copy()
        eligible_group_ids = set(axis_groups["loading_group"].astype(str))
        axis_membership = group_projection.group_membership.loc[
            group_projection.group_membership["loading_group"]
            .astype(str)
            .isin(eligible_group_ids)
        ].copy()
        axis_store[axis_key] = estimate_latent_axis_clustering(
            axis_groups,
            axis_membership,
            axis_loading_views or loading_views,
            LatentAxisClusteringConfig(),
        )
        axis_store[axis_key].provenance["alignment_eligibility"] = {
            "eligible_modalities": list(eligible_modalities),
            "abstained_modalities": sorted(set(MODALITIES) - set(eligible_modalities)),
            "selection_source": "prepared_observed_background_alignment_gate",
        }
    axis_projection = axis_store[axis_key]
    local_sparse_projection = group_projection.sparse_pair_projection.loc[
        group_projection.sparse_pair_projection.get(
            "edge_type", pd.Series(dtype=str)
        ).astype(str).eq("within_assay_shared_state")
    ].copy()
    if cfg.replace_pairwise_state_layer_with_groups:
        consensus = replace_pairwise_state_layer(
            direct_resolved,
            local_sparse_projection,
        )
    else:
        consensus = attach_group_projection_once(
            direct_resolved,
            local_sparse_projection,
        )
    return (
        consensus,
        motif,
        group_projection,
        axis_projection,
        refinement,
        structure_anchor,
        crossfit_result,
    )


def state_group_exclusions(
    direct_resolved: pd.DataFrame,
    motif: ColliderMotifResult,
) -> tuple[str, ...]:
    """Exclude only observed collider children from pure-state grouping.

    Broad exclusion of every endpoint in an unresolved direct component was a
    major source of false negatives in the first TRAIN block.  The revised
    rule is deliberately narrow: a selected collider child mixes its parents'
    responses and is not a pure state reporter, while unresolved endpoints
    remain eligible for an independently stable loading-group decision.  The
    parent--parent pair is constrained separately by
    :func:`direct_collider_parent_pairs`.
    """

    _ = motif  # retained in the public signature for call-site compatibility
    return tuple(sorted(set(direct_collider_children(direct_resolved))))


def direct_collider_parent_pairs(
    motif: ColliderMotifResult,
) -> tuple[tuple[str, str], ...]:
    """Return observable parent pairs from accepted two-parent colliders."""

    direct = motif.direct_candidates
    if direct.empty:
        return ()
    required = {"source_assay", "node1", "node2", "collider_center"}
    missing = required - set(direct.columns)
    if missing:
        raise ValueError(
            "Direct motif lacks fields needed for parent-pair constraints: "
            f"{sorted(missing)}"
        )
    pairs: set[tuple[str, str]] = set()
    oriented_rows: list[dict[str, Any]] = []
    for record in direct.to_dict(orient="records"):
        endpoint1 = str(record.get("endpoint1", "")).upper()
        endpoint2 = str(record.get("endpoint2", "")).upper()
        center = str(record.get("collider_center", "")).strip()
        if endpoint1 == endpoint2 == "CIRCLE" and center.lower() in {
            "",
            "nan",
            "none",
        }:
            continue
        oriented_rows.append(record)
    if not oriented_rows:
        return ()
    oriented = pd.DataFrame(oriented_rows)
    for (_, center), frame in oriented.groupby(
        ["source_assay", "collider_center"], sort=True
    ):
        center = str(center)
        parents: set[str] = set()
        for row in frame.itertuples(index=False):
            left, right = str(row.node1), str(row.node2)
            if center == left and center != right:
                parents.add(right)
            elif center == right and center != left:
                parents.add(left)
            else:
                raise ValueError(
                    "Direct motif collider_center is not an endpoint of its arm"
                )
        if len(parents) == 2:
            pairs.add(tuple(sorted(parents)))
        elif len(parents) > 2:
            raise ValueError(
                "Accepted collider has more than two parents after refinement"
            )
    return tuple(sorted(pairs))


def _motif_result_sha256(motif: ColliderMotifResult) -> str:
    digest = hashlib.sha256()
    for label, frame in (
        ("direct", motif.direct_candidates),
        ("unresolved", motif.demoted_or_unresolved),
        ("components", motif.component_diagnostics),
        ("wedges", motif.wedge_diagnostics),
        ("aggregated", motif.aggregated_evidence),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(frame_sha256(frame).encode("ascii"))
    digest.update(json.dumps(motif.metadata, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _observed_collider_refinement_sha256(
    refinement: ObservedColliderStructureResult,
) -> str:
    """Hash every decision/evidence table and its observed-only provenance."""

    digest = hashlib.sha256()
    for label, frame in (
        ("decisions", refinement.decisions),
        ("wedge_views", refinement.wedge_view_evidence),
        ("wedges", refinement.wedge_evidence),
        ("completion_views", refinement.completion_view_evidence),
        ("completions", refinement.completion_evidence),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(frame_sha256(frame).encode("ascii"))
    digest.update(
        json.dumps(
            refinement.metadata, sort_keys=True, default=str
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _crossfit_collider_world_evidence_sha256(
    evidence: CrossfitColliderWorldEvidence,
) -> str:
    """Hash the once-per-world observed sparse-count fit and provenance."""

    digest = hashlib.sha256()
    digest.update(b"generative_candidates")
    digest.update(frame_sha256(evidence.generative_candidates).encode("ascii"))
    digest.update(
        json.dumps(
            evidence.provenance, sort_keys=True, default=str
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _crossfit_collider_result_sha256(
    result: CrossfitHierarchicalColliderResult,
) -> str:
    """Hash every candidate-level replacement table and its provenance."""

    digest = hashlib.sha256()
    for label, frame in (
        ("candidates", result.candidates),
        ("decisions", result.decisions),
        ("direct", result.direct_candidates),
        ("unresolved", result.demoted_or_unresolved),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(frame_sha256(frame).encode("ascii"))
    digest.update(
        json.dumps(result.provenance, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def _structure_anchored_group_sha256(
    result: StructureAnchoredGroupResult,
) -> str:
    """Hash the complete state-gated assay-local group decision."""

    digest = hashlib.sha256()
    for label, frame in (
        ("pair_evidence", result.pair_evidence),
        ("typed_groups", result.typed_groups),
        ("assay_decisions", result.assay_decisions),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(frame_sha256(frame).encode("ascii"))
    digest.update(
        json.dumps(result.provenance, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def _axis_result_sha256(result: LatentAxisClusteringResult) -> str:
    """Hash the complete re-thresholdable latent-axis result."""

    digest = hashlib.sha256()
    for label, frame in (
        ("group_pair_views", result.group_pair_view_evidence),
        ("group_pairs", result.group_pair_evidence),
        ("axis_candidates", result.axis_candidates),
        ("group_axis_views", result.group_axis_view_evidence),
        ("group_axes", result.group_axis_evidence),
        ("typed_axes", result.typed_axis_hyperedges),
        ("representatives", result.representative_evidence),
        ("sparse_projection", result.sparse_pair_projection),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(frame_sha256(frame).encode("ascii"))
    digest.update(
        json.dumps(result.provenance, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def _validate_crossfit_world_evidence_freeze(
    world_out: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Verify the world-level collider fit before any scorer-domain read."""

    if manifest.get("crossfit_hierarchical_collider_applied") is not True:
        return
    if (
        manifest.get(
            "crossfit_hierarchical_collider_randomization_key_bound_to_world_input_seed"
        )
        is not True
    ):
        raise RuntimeError(
            "Frozen crossfit collider lacks its randomization-key binding"
        )
    world_result_hash = _expected_sha256(
        manifest.get("crossfit_hierarchical_collider_world_result_sha256"),
        label="crossfit hierarchical collider world-result hash",
    )
    expected_files = manifest.get(
        "crossfit_hierarchical_collider_world_file_sha256"
    )
    required_files = {
        "world_generative_candidates.tsv",
        "world_evidence_provenance.json",
    }
    if not isinstance(expected_files, dict) or set(expected_files) != required_files:
        raise RuntimeError("Frozen crossfit collider world hashes are incomplete")
    crossfit_world_out = world_out / "crossfit_hierarchical_collider"
    if crossfit_world_out.is_symlink() or not crossfit_world_out.is_dir():
        raise RuntimeError("Frozen crossfit collider world evidence is missing")
    entries = list(crossfit_world_out.iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in entries)
        or {path.name for path in entries} != required_files
    ):
        raise RuntimeError(
            "Frozen crossfit collider world evidence has an unsafe entry set"
        )
    observed_files = {
        path.name: _file_sha256(path)
        for path in sorted(entries)
        if path.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError("Frozen crossfit collider world evidence changed")
    world_provenance = _read_regular_json(
        crossfit_world_out / "world_evidence_provenance.json",
        label="crossfit collider world provenance",
    )
    input_binding = manifest.get("world_input_pretruth_binding")
    if not isinstance(input_binding, dict):
        raise RuntimeError("Frozen crossfit collider lacks its world binding")
    def strict_seed(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("boolean is not a world seed")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{3,7}", value):
            parsed = int(value)
        else:
            raise ValueError("world seed must be an integer or decimal string")
        if not re.fullmatch(r"[0-9]{4,8}", str(parsed)):
            raise ValueError("world seed is outside the registered lexical contract")
        return parsed

    try:
        bound_seed = strict_seed(input_binding["seed"])
        randomization_key = strict_seed(
            manifest["crossfit_hierarchical_collider_randomization_key"]
        )
        provenance_key = strict_seed(world_provenance["randomization_key"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Frozen crossfit collider randomization binding is malformed"
        ) from exc
    if (
        world_provenance.get("world_evidence_result_sha256")
        != world_result_hash
        or randomization_key != bound_seed
        or provenance_key != bound_seed
    ):
        raise RuntimeError(
            "Frozen crossfit collider randomization key differs from its world"
        )


def load_scorer_artifacts_after_pretruth_freeze(
    world_dir: Path,
    world_out: Path,
    *,
    expected_pretruth_manifest_sha256: str | None = None,
    return_binding: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame] | tuple[
    pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]
]:
    """Open scorer artifacts only after verifying the pretruth freeze marker."""

    manifest_path = world_out / "pretruth_consensus_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Refusing scorer access before pretruth manifest exists")
    try:
        manifest_bytes = _read_regular_bytes_once(
            manifest_path, label="pretruth consensus manifest"
        )
        manifest = _json_from_bytes(
            manifest_bytes, label="pretruth consensus manifest"
        )
    except ValueError as exc:
        raise RuntimeError(f"Refusing unsafe pretruth manifest: {exc}") from exc
    if expected_pretruth_manifest_sha256 is not None and _bytes_sha256(
        manifest_bytes
    ) != _expected_sha256(
        expected_pretruth_manifest_sha256,
        label="expected pretruth consensus-manifest hash",
    ):
        raise RuntimeError("Pretruth consensus manifest changed before scorer access")
    required_attestations = (
        "all_consensuses_built_before_truth_open",
        "all_collider_motif_outputs_built_before_truth_open",
        "collider_motif_discovery_inputs_only",
        "collider_motif_applied_after_preliminary_nested_consensus",
        "collider_motif_applied_before_direct_collider_children",
        "collider_motif_applied_before_loading_group_projection",
        "collider_motif_base_demotions_retained_as_unresolved",
        "observed_collider_rescue_only_completion_requires_state_loading_stability",
        "latent_axis_hyperedges_built_before_truth_open",
        "latent_axis_hyperedges_attached_once_post_consensus",
        "ordinary_state_skeleton_contains_within_assay_groups_only",
        "pairwise_cross_group_matches_are_diagnostics_only",
        "loading_views_reuse_same_observed_cells",
        "loading_views_are_algorithmic_stability_repeats",
        "loading_view_support_not_world_sample_size",
        "world_input_discovery_payloads_content_addressed",
    )
    failed = [name for name in required_attestations if manifest.get(name) is not True]
    if manifest.get("loading_views_are_independent_worlds") is not False:
        failed.append("loading_views_are_independent_worlds")
    for name in (
        "crossfit_hierarchical_collider_discovery_inputs_only",
    ):
        if manifest.get(name) is not True:
            failed.append(name)
    for name in (
        "crossfit_hierarchical_collider_truth_pairing_external_inputs",
        "crossfit_hierarchical_collider_paired_cells_used",
        "crossfit_hierarchical_collider_cross_assay_coordinate_registration_required",
    ):
        if manifest.get(name) is not False:
            failed.append(name)
    if manifest.get("structure_anchored_loading_applied_before_group_projection") is True:
        for name in (
            "crossfit_hierarchical_collider_applied",
            "crossfit_hierarchical_collider_fit_once_per_world",
            "crossfit_hierarchical_collider_randomization_key_bound_to_world_input_seed",
            "crossfit_hierarchical_collider_applied_after_observed_refinement",
            "crossfit_hierarchical_collider_applied_before_direct_collider_children",
            "crossfit_hierarchical_collider_applied_before_loading_group_projection",
        ):
            if manifest.get(name) is not True:
                failed.append(name)
    for hash_key in (
        "consensus_sha256",
        "collider_motif_beta_vote_sha256",
        "collider_motif_result_sha256",
        "latent_axis_projection_sha256",
    ):
        if not manifest.get(hash_key):
            failed.append(hash_key)
    if manifest.get("crossfit_hierarchical_collider_applied") is True:
        for hash_key in (
            "crossfit_hierarchical_collider_world_result_sha256",
            "crossfit_hierarchical_collider_world_file_sha256",
            "crossfit_hierarchical_collider_result_sha256",
            "crossfit_hierarchical_collider_file_sha256",
            "crossfit_hierarchical_collider_source_sha256",
        ):
            if not manifest.get(hash_key):
                failed.append(hash_key)
    if failed:
        raise RuntimeError(
            f"Refusing scorer access before complete pretruth freeze: {sorted(failed)}"
        )
    _validate_crossfit_world_evidence_freeze(world_out, manifest)
    try:
        full_binding = validate_world_input_binding(
            world_dir,
            verify_scorer_payload=True,
        )
    except ValueError as error:
        raise RuntimeError(
            f"Refusing scorer access for an unbound or tampered world: {error}"
        ) from error
    expected_binding = manifest.get("world_input_pretruth_binding")
    if not isinstance(expected_binding, dict):
        raise RuntimeError("Pretruth manifest lacks its world-input binding")
    for domain in ("run_summary", "discovery_manifest", "scorer_manifest"):
        expected_record = expected_binding.get(domain)
        observed_record = full_binding.get(domain)
        if not isinstance(expected_record, dict) or not isinstance(observed_record, dict):
            raise RuntimeError(f"Malformed pretruth world-input binding for {domain}")
        if expected_record.get("sha256") != observed_record.get("sha256"):
            raise RuntimeError(
                f"World input changed between discovery and scorer access: {domain}"
            )
    scorer_dir = world_dir / "scorer"
    scorer_manifest_path = scorer_dir / "manifest.json"
    truth = scorer_dir / "truth_typed_edges.tsv"
    parameters = scorer_dir / "world_parameters.json"
    roles = scorer_dir / "role_assignments.tsv"
    if not scorer_dir.is_dir() or not all(
        path.exists() for path in (scorer_manifest_path, truth, parameters, roles)
    ):
        raise FileNotFoundError(f"World lacks split v4 scorer artifacts: {world_dir}")
    try:
        scorer_snapshot = {
            "manifest.json": _read_regular_bytes_once(
                scorer_manifest_path, label="scorer manifest"
            ),
            truth.name: _read_regular_bytes_once(truth, label="truth typed edges"),
            parameters.name: _read_regular_bytes_once(
                parameters, label="world parameters"
            ),
            roles.name: _read_regular_bytes_once(roles, label="role assignments"),
        }
        scorer_manifest = _json_from_bytes(
            scorer_snapshot["manifest.json"], label="scorer manifest"
        )
    except ValueError as exc:
        raise RuntimeError(f"Refusing unsafe scorer artifact: {exc}") from exc
    if _bytes_sha256(scorer_snapshot["manifest.json"]) != full_binding[
        "scorer_manifest"
    ]["sha256"]:
        raise RuntimeError("Held scorer manifest differs from the verified binding")
    scorer_file_hashes = scorer_manifest.get("file_sha256")
    if not isinstance(scorer_file_hashes, dict):
        raise RuntimeError("Scorer manifest lacks content-addressed payload hashes")
    for filename in (truth.name, parameters.name, roles.name):
        if scorer_file_hashes.get(filename) != _bytes_sha256(
            scorer_snapshot[filename]
        ):
            raise RuntimeError(
                f"Held scorer artifact differs from its manifest: {filename}"
            )
    try:
        parameter_payload = _json_from_bytes(
            scorer_snapshot[parameters.name], label="world parameters"
        )
        truth_frame = pd.read_csv(
            io.BytesIO(scorer_snapshot[truth.name]),
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        role_frame = pd.read_csv(
            io.BytesIO(scorer_snapshot[roles.name]),
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not parse held scorer artifacts: {exc}") from exc
    result = (truth_frame, parameter_payload, role_frame)
    if return_binding:
        return (*result, full_binding)
    return result


def combine_attainable_metrics(
    pair_metrics: dict[str, Any],
    latent_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Make latent-axis recovery, rather than an arbitrary exact pair, primary.

    Exact P/S and mark-pair metrics remain in the result as diagnostics.  The
    pass/fail state class instead requires recovery of the assay-local groups,
    coverage of every simulated latent axis across assays, slow-axis gene
    support, and no selected axis-mismatch or wrong-layer claims.
    """

    state_success = int(
        float(latent_metrics["combined_within_assay_ps_group_recall"]) >= 0.80
        and float(latent_metrics["combined_cross_assay_latent_axis_coverage"]) >= 1.0
        and int(latent_metrics["combined_slow_gene_axis_support"]) == 1
        and int(latent_metrics["n_selected_axis_mismatch_claims"]) == 0
        and int(latent_metrics["n_selected_wrong_layer_claims"]) == 0
    )
    combined = {
        **pair_metrics,
        **latent_metrics,
        "exact_pair_shared_class_success": int(pair_metrics["shared_class_success"]),
        "exact_pair_slow_class_success": int(pair_metrics["slow_class_success"]),
        "latent_state_class_success": state_success,
        "shared_class_success": state_success,
        "slow_class_success": int(
            int(latent_metrics["combined_slow_gene_axis_support"]) == 1
        ),
        "world_success_contract": "typed_direct_plus_native_latent_axes_v3",
    }
    combined["world_success"] = int(
        int(combined["direct_class_success"]) == 1
        and int(combined["collider_class_success"]) == 1
        and state_success == 1
        and int(combined["null_class_success"]) == 1
        and int(combined["density_class_success"]) == 1
        and int(combined["precision_class_success"]) == 1
    )
    return combined


def candidate_grid(base: CABossV4Config) -> list[tuple[str, CABossV4Config]]:
    candidates: list[tuple[str, CABossV4Config]] = []
    for loading_score in (0.005, 0.010, 0.020):
        for loading_cosine in (0.65, 0.75, 0.85):
            for hierarchical_probability in (0.85, 0.88, 0.90, 0.95):
                for direct_q in (0.02, 0.05, 0.10):
                    for mutual_neighbors in (1, 2):
                        name = (
                            f"v4_ls{loading_score:.3f}_lc{loading_cosine:.2f}_"
                            f"hp{hierarchical_probability:.3f}_q{direct_q:.2f}_"
                            f"k{mutual_neighbors}"
                        )
                        candidates.append(
                            (
                                name,
                                replace(
                                    base,
                                    loading_min_pair_state_score=loading_score,
                                    loading_min_abs_cosine=loading_cosine,
                                    hierarchical_probability_threshold=hierarchical_probability,
                                    direct_residual_q=direct_q,
                                    loading_cross_neighbors=mutual_neighbors,
                                ),
                            )
                        )
    return candidates


def nested_inputs(discovery_dir: Path) -> list[tuple[Path, Path]]:
    matrices = sorted(discovery_dir.glob("nested/*/observed_metacell_matrix.tsv"))
    inputs: list[tuple[Path, Path]] = []
    for matrix in matrices:
        membership = matrix.with_name("analysis_membership.tsv")
        if not membership.exists():
            raise FileNotFoundError(f"Missing safe membership beside {matrix}")
        inputs.append((matrix, membership))
    return inputs


def axis_eligible_modalities_from_prepared(prepared_dir: Path) -> tuple[str, ...]:
    """Read the observable synchronization gate used for cross-assay axes."""

    provenance_path = prepared_dir / "alignment_provenance.json"
    _assert_discovery_artifact(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    selection = provenance.get("alignment_selection", {})
    if not isinstance(selection, dict):
        raise ValueError("Prepared alignment selection provenance is malformed")
    selected_mode = str(selection.get("selected_mode", "hard"))
    eligible = selection.get("cross_assay_axis_eligible_modalities")
    if selected_mode.startswith("soft"):
        if not isinstance(eligible, list):
            raise ValueError("Soft alignment lacks axis-eligibility provenance")
        modalities = tuple(sorted(str(value) for value in eligible))
    else:
        # A deliberately requested hard baseline retains its historical
        # all-assay behavior.  The production v4 default is partial soft.
        modalities = tuple(sorted(MODALITIES))
    if (
        not modalities
        or "rna" not in modalities
        or not set(modalities).issubset(MODALITIES)
    ):
        raise ValueError("Prepared axis-eligible modality set is invalid")
    return modalities


def _evaluate_world_internal(
    world_dir: Path,
    candidates: list[tuple[str, CABossV4Config]],
    *,
    out_dir: Path,
    save_candidate_names: set[str],
    defer_scorer: bool,
    loading_embedding_path: Path | None = None,
    state_membership_path: Path | None = None,
) -> list[dict[str, Any]] | FrozenWorldPretruth:
    pretruth_input_binding = validate_world_input_binding(
        world_dir,
        verify_scorer_payload=False,
    )
    discovery_dir = world_dir / "discovery"
    if not discovery_dir.is_dir():
        raise FileNotFoundError(f"V4 world lacks discovery domain: {world_dir}")
    world_out = out_dir / world_dir.name
    world_out.mkdir(parents=True, exist_ok=True)
    state_membership: pd.DataFrame | None = None
    state_membership_sha256: str | None = None
    if state_membership_path is not None:
        expected_membership = (
            world_out
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        )
        supplied_membership = Path(state_membership_path).resolve(strict=True)
        if supplied_membership != expected_membership.resolve(strict=True):
            raise ValueError(
                "State membership must be the same world's frozen subspace output"
            )
        state_membership, state_membership_sha256 = load_frozen_state_membership(
            supplied_membership
        )
    prepared_dir = world_out / "prepared_discovery"
    prepare_inputs(
        discovery_dir,
        prepared_dir,
        PrepareConfig(
            alignment_seed=candidates[0][1].seed,
            alignment_mode=candidates[0][1].alignment_mode,
            soft_n_microstates=candidates[0][1].soft_n_microstates,
            soft_max_rotation_dispersion=(
                candidates[0][1].soft_max_rotation_dispersion
            ),
            soft_low_confidence_action=(
                candidates[0][1].soft_low_confidence_action
            ),
        ),
    )
    embedding = prepared_dir / "analysis_embeddings.tsv"
    group_loading_embedding = (
        Path(loading_embedding_path).resolve(strict=True)
        if loading_embedding_path is not None
        else embedding.resolve(strict=True)
    )
    loading_coordinates_cross_assay_aligned = (
        group_loading_embedding == embedding.resolve(strict=True)
    )
    axis_eligible_modalities = axis_eligible_modalities_from_prepared(prepared_dir)
    cell_tables = sorted(discovery_dir.glob("observed_focal__*.tsv"))
    chrom_cell_tables = sorted(discovery_dir.glob("observed_focal__chrom_*.tsv"))
    (
        beta_vote_evidence,
        beta_vote_frames,
        beta_vote_metadata,
    ) = fit_collider_motif_vote_evidence(chrom_cell_tables, embedding)
    beta_vote_hashes = write_collider_motif_vote_evidence(
        world_out,
        beta_vote_frames,
        beta_vote_metadata,
    )
    beta_input_hashes = {
        str(embedding.relative_to(world_out)): _file_sha256(embedding),
        **{
            str(path.relative_to(world_dir)): _file_sha256(path)
            for path in chrom_cell_tables
        },
    }
    beta_source_hashes = {
        "stratified_beta_binomial_edges.py": _file_sha256(
            Path(__file__).with_name("stratified_beta_binomial_edges.py")
        ),
        "structure_aware_collider_motif.py": _file_sha256(
            Path(__file__).with_name("structure_aware_collider_motif.py")
        ),
        "structure_aware_collider_rescue.py": _file_sha256(
            Path(__file__).with_name("structure_aware_collider_rescue.py")
        ),
        "observed_collider_structure.py": _file_sha256(
            Path(__file__).with_name("observed_collider_structure.py")
        ),
    }
    crossfit_world_evidence: CrossfitColliderWorldEvidence | None = None
    crossfit_world_result_sha256: str | None = None
    crossfit_world_file_hashes: dict[str, str] = {}
    crossfit_source_hashes = {
        "crossfit_hierarchical_collider_v4.py": _file_sha256(
            Path(__file__).with_name("crossfit_hierarchical_collider_v4.py")
        ),
    }
    if state_membership is not None:
        expected_assay_local_embedding = (
            world_out
            / "assay_local_state_embedding_v4"
            / "analysis_embeddings.tsv"
        )
        if group_loading_embedding != expected_assay_local_embedding.resolve(
            strict=True
        ):
            raise ValueError(
                "Formal crossfit collider evidence requires the same world's "
                "frozen assay-local background embedding"
            )
        if loading_coordinates_cross_assay_aligned:
            raise ValueError(
                "Crossfit collider evidence must not use cross-assay registered "
                "coordinates"
            )
        assay_cell_frames: dict[str, pd.DataFrame] = {}
        for path in chrom_cell_tables:
            _assert_discovery_artifact(path)
            frame = pd.read_csv(path, sep="\t", keep_default_na=False)
            if frame.empty or "modality" not in frame:
                raise ValueError(f"Malformed chromatin focal table: {path}")
            modalities = set(frame["modality"].astype(str))
            if len(modalities) != 1:
                raise ValueError(f"Chromatin focal table mixes modalities: {path}")
            modality = next(iter(modalities))
            if modality in assay_cell_frames:
                raise ValueError(f"Duplicate chromatin focal modality: {modality}")
            assay_cell_frames[modality] = frame
        assay_local_embedding = pd.read_csv(
            group_loading_embedding, sep="\t", keep_default_na=False
        )
        crossfit_world_evidence = fit_crossfit_collider_world_evidence(
            assay_cell_tables=assay_cell_frames,
            assay_local_embedding=assay_local_embedding,
            randomization_key=int(pretruth_input_binding["seed"]),
            config=FROZEN_CROSSFIT_HIERARCHICAL_COLLIDER_CONFIG,
        )
        crossfit_world_result_sha256 = (
            _crossfit_collider_world_evidence_sha256(crossfit_world_evidence)
        )
        crossfit_world_out = world_out / "crossfit_hierarchical_collider"
        crossfit_world_out.mkdir(parents=True, exist_ok=True)
        crossfit_world_evidence.generative_candidates.to_csv(
            crossfit_world_out / "world_generative_candidates.tsv",
            sep="\t",
            index=False,
        )
        (crossfit_world_out / "world_evidence_provenance.json").write_text(
            json.dumps(
                {
                    **crossfit_world_evidence.provenance,
                    "world_evidence_result_sha256": (
                        crossfit_world_result_sha256
                    ),
                    "applied_before_candidate_thresholding": True,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        crossfit_world_file_hashes = {
            path.name: _file_sha256(path)
            for path in sorted(crossfit_world_out.iterdir())
            if path.is_file()
        }
    structure_anchor_source_hashes = {
        "structure_anchored_loading_groups_v4.py": _file_sha256(
            Path(__file__).with_name("structure_anchored_loading_groups_v4.py")
        ),
        "loading_group_projection_v4.py": _file_sha256(
            Path(__file__).with_name("loading_group_projection_v4.py")
        ),
    }
    nested = nested_inputs(prepared_dir)
    if len(nested) != 6:
        raise ValueError(f"Expected six nested metacell matrices in {world_dir}, found {len(nested)}")
    base = candidates[0][1]
    raw_evidence: list[pd.DataFrame] = []
    hierarchical_evidence: list[pd.DataFrame] = []
    loading_views = []
    axis_loading_views = []
    for nested_i, (matrix, membership) in enumerate(nested):
        nested_base = replace(base, seed=base.seed + nested_i)
        # Refit the sparse cell layer under six deterministic seed/fold views.
        # These fits reuse the same observed cells: they measure correlated
        # algorithmic stability, never six independent data replicates or
        # six worlds.  The projection provenance records that distinction.
        hierarchy, hierarchy_details, hierarchy_metadata = fit_hierarchical_cell_evidence(
            cell_tables,
            embedding,
            nested_base,
        )
        loading, loading_details, loading_metadata = fit_cell_loading_group_evidence(
            discovery_dir,
            group_loading_embedding,
            nested_base,
        )
        if loading_coordinates_cross_assay_aligned:
            axis_loading = loading
            axis_loading_details = loading_details
            axis_loading_metadata = loading_metadata
        else:
            (
                axis_loading,
                axis_loading_details,
                axis_loading_metadata,
            ) = fit_cell_loading_group_evidence(
                discovery_dir,
                embedding,
                nested_base,
            )
        _, _, evidence, _, _, _, summary = discover_v4(
            matrix,
            embedding,
            membership,
            nested_base,
            hierarchical_evidence=hierarchy,
            cell_loading_evidence=axis_loading,
        )
        raw_evidence.append(evidence)
        hierarchical_evidence.append(hierarchy)
        loading_views.append(
            make_evidence_view(
                view_id=matrix.parent.name,
                feature_models=loading_details["feature_models"],
                loadings=loading_details["loadings"],
                pair_evidence=loading,
            )
        )
        axis_loading_views.append(
            make_evidence_view(
                view_id=matrix.parent.name,
                feature_models=axis_loading_details["feature_models"],
                loadings=axis_loading_details["loadings"],
                pair_evidence=axis_loading,
            )
        )
        nested_out = world_out / matrix.parent.name
        nested_out.mkdir(parents=True, exist_ok=True)
        evidence.to_csv(nested_out / "raw_pair_evidence.tsv", sep="\t", index=False)
        hierarchy.to_csv(
            nested_out / "hierarchical_sparse_edge_evidence.tsv", sep="\t", index=False
        )
        (nested_out / "hierarchical_sparse_metadata.json").write_text(
            json.dumps(hierarchy_metadata, indent=2)
        )
        loading.to_csv(nested_out / "cell_loading_pair_evidence.tsv", sep="\t", index=False)
        (nested_out / "cell_loading_provenance.json").write_text(
            json.dumps(loading_metadata, indent=2)
        )
        for name, frame in loading_details.items():
            frame.to_csv(nested_out / f"cell_loading_{name}.tsv", sep="\t", index=False)
        if not loading_coordinates_cross_assay_aligned:
            axis_loading.to_csv(
                nested_out / "axis_loading_pair_evidence.tsv",
                sep="\t",
                index=False,
            )
            (nested_out / "axis_loading_provenance.json").write_text(
                json.dumps(axis_loading_metadata, indent=2)
            )
            for name, frame in axis_loading_details.items():
                frame.to_csv(
                    nested_out / f"axis_loading_{name}.tsv",
                    sep="\t",
                    index=False,
                )
        for name, frame in hierarchy_details.items():
            frame.to_csv(
                nested_out / f"hierarchical_sparse_{name}.tsv", sep="\t", index=False
            )
        (nested_out / "discovery_summary.json").write_text(json.dumps(summary, indent=2))

    # Freeze every candidate consensus before opening the scorer domain.
    consensuses: dict[str, pd.DataFrame] = {}
    motif_results: dict[str, ColliderMotifResult] = {}
    observed_collider_refinements: dict[
        str, ObservedColliderStructureResult
    ] = {}
    crossfit_collider_results: dict[
        str, CrossfitHierarchicalColliderResult
    ] = {}
    structure_anchored_groups: dict[str, StructureAnchoredGroupResult] = {}
    group_projections: dict[str, Any] = {}
    axis_projections: dict[str, LatentAxisClusteringResult] = {}
    consensus_hashes: dict[str, str] = {}
    consensus_file_hashes: dict[str, str] = {}
    motif_result_hashes: dict[str, str] = {}
    observed_collider_refinement_hashes: dict[str, str] = {}
    observed_collider_refinement_file_hashes: dict[str, dict[str, str]] = {}
    crossfit_collider_result_hashes: dict[str, str] = {}
    crossfit_collider_result_file_hashes: dict[str, dict[str, str]] = {}
    structure_anchored_group_hashes: dict[str, str] = {}
    structure_anchored_group_file_hashes: dict[str, dict[str, str]] = {}
    group_projection_hashes: dict[str, str] = {}
    group_projection_file_hashes: dict[str, dict[str, str]] = {}
    axis_projection_hashes: dict[str, str] = {}
    axis_projection_file_hashes: dict[str, dict[str, str]] = {}
    projection_cache: dict[tuple[Any, ...], Any] = {}
    axis_cache: dict[tuple[Any, ...], LatentAxisClusteringResult] = {}
    for candidate_name, cfg in candidates:
        graphs = [
            build_typed_graph(
                apply_evidence_thresholds(evidence, cfg),
                cfg,
                hierarchical_evidence=hierarchy,
            )
            for evidence, hierarchy in zip(
                raw_evidence, hierarchical_evidence, strict=True
            )
        ]
        preliminary_consensus = consensus_typed_graph(graphs, minimum_count=4)
        (
            consensus,
            motif_result,
            group_projection,
            axis_projection,
            observed_refinement,
            structure_anchor,
            crossfit_collider_result,
        ) = resolve_postconsensus_candidate(
            preliminary_consensus,
            beta_vote_evidence,
            loading_views,
            cfg,
            projection_cache,
            axis_cache,
            axis_eligible_modalities,
            axis_loading_views,
            loading_coordinates_cross_assay_aligned,
            base_edge_views={
                view.view_id: graph
                for view, graph in zip(loading_views, graphs, strict=True)
            },
            residual_views={
                view.view_id: evidence
                for view, evidence in zip(
                    loading_views, raw_evidence, strict=True
                )
            },
            feature_model_views={
                view.view_id: view.feature_models for view in loading_views
            },
            state_membership=state_membership,
            crossfit_world_evidence=crossfit_world_evidence,
        )
        if observed_refinement is None:
            raise RuntimeError(
                "Formal candidate omitted observed collider refinement evidence"
            )
        if state_membership is not None and structure_anchor is None:
            raise RuntimeError(
                "Formal candidate omitted state-gated structure anchor evidence"
            )
        if state_membership is not None and crossfit_collider_result is None:
            raise RuntimeError(
                "Formal candidate omitted crossfit hierarchical collider evidence"
            )
        consensuses[candidate_name] = consensus
        motif_results[candidate_name] = motif_result
        observed_collider_refinements[candidate_name] = observed_refinement
        if crossfit_collider_result is not None:
            crossfit_collider_results[candidate_name] = crossfit_collider_result
            crossfit_collider_result_hashes[candidate_name] = (
                _crossfit_collider_result_sha256(crossfit_collider_result)
            )
        if structure_anchor is not None:
            structure_anchored_groups[candidate_name] = structure_anchor
            structure_anchored_group_hashes[candidate_name] = (
                _structure_anchored_group_sha256(structure_anchor)
            )
        group_projections[candidate_name] = group_projection
        axis_projections[candidate_name] = axis_projection
        consensus_hashes[candidate_name] = hashlib.sha256(
            consensus.to_csv(sep="\t", index=False).encode("utf-8")
        ).hexdigest()
        motif_result_hashes[candidate_name] = _motif_result_sha256(motif_result)
        observed_collider_refinement_hashes[candidate_name] = (
            _observed_collider_refinement_sha256(observed_refinement)
        )
        group_projection_hashes[candidate_name] = hashlib.sha256(
            (
                group_projection.typed_hyperedges.to_csv(sep="\t", index=False)
                + group_projection.sparse_pair_projection.to_csv(sep="\t", index=False)
            ).encode("utf-8")
        ).hexdigest()
        axis_projection_hashes[candidate_name] = _axis_result_sha256(
            axis_projection
        )
        if candidate_name in save_candidate_names:
            candidate_out = world_out / candidate_name
            candidate_out.mkdir(parents=True, exist_ok=True)
            for nested_i, graph in enumerate(graphs):
                graph.to_csv(candidate_out / f"nested_{nested_i}_edges.tsv", sep="\t", index=False)
            consensus.to_csv(candidate_out / "consensus_edges.tsv", sep="\t", index=False)
            motif_out = candidate_out / "collider_motif"
            motif_out.mkdir(parents=True, exist_ok=True)
            motif_tables = {
                "direct_candidates.tsv": motif_result.direct_candidates,
                "demoted_or_unresolved.tsv": motif_result.demoted_or_unresolved,
                "component_diagnostics.tsv": motif_result.component_diagnostics,
                "wedge_diagnostics.tsv": motif_result.wedge_diagnostics,
                "aggregated_vote_evidence.tsv": motif_result.aggregated_evidence,
                "observed_structure_decisions.tsv": (
                    observed_refinement.decisions
                ),
                "observed_structure_wedge_view_evidence.tsv": (
                    observed_refinement.wedge_view_evidence
                ),
                "observed_structure_wedge_evidence.tsv": (
                    observed_refinement.wedge_evidence
                ),
                "observed_structure_completion_view_evidence.tsv": (
                    observed_refinement.completion_view_evidence
                ),
                "observed_structure_completion_evidence.tsv": (
                    observed_refinement.completion_evidence
                ),
            }
            for filename, frame in motif_tables.items():
                frame.to_csv(motif_out / filename, sep="\t", index=False)
            (motif_out / "motif_provenance.json").write_text(
                json.dumps(
                    {
                        **motif_result.metadata,
                        "applied_after_preliminary_nested_consensus": True,
                        "applied_before_direct_collider_child_exclusion": True,
                        "applied_before_loading_group_projection": True,
                        "beta_vote_evidence_sha256": beta_vote_hashes,
                        "motif_result_sha256": motif_result_hashes[candidate_name],
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            (motif_out / "observed_structure_provenance.json").write_text(
                json.dumps(
                    {
                        **observed_refinement.metadata,
                        "config": asdict(
                            FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG
                        ),
                        "refinement_result_sha256": (
                            observed_collider_refinement_hashes[candidate_name]
                        ),
                        "applied_to_collider_motif_before_loading_groups": True,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            observed_collider_refinement_file_hashes[candidate_name] = {
                path.name: _file_sha256(path)
                for path in sorted(motif_out.iterdir())
                if path.is_file()
                and path.name.startswith("observed_structure_")
            }
            if crossfit_collider_result is not None:
                crossfit_out = (
                    candidate_out / "crossfit_hierarchical_collider"
                )
                crossfit_out.mkdir(parents=True, exist_ok=True)
                crossfit_tables = {
                    "candidate_evidence.tsv": (
                        crossfit_collider_result.candidates
                    ),
                    "assay_decisions.tsv": crossfit_collider_result.decisions,
                    "direct_candidates.tsv": (
                        crossfit_collider_result.direct_candidates
                    ),
                    "demoted_or_unresolved.tsv": (
                        crossfit_collider_result.demoted_or_unresolved
                    ),
                }
                for filename, frame in crossfit_tables.items():
                    frame.to_csv(
                        crossfit_out / filename, sep="\t", index=False
                    )
                (crossfit_out / "provenance.json").write_text(
                    json.dumps(
                        {
                            **crossfit_collider_result.provenance,
                            "result_sha256": crossfit_collider_result_hashes[
                                candidate_name
                            ],
                            "world_evidence_result_sha256": (
                                crossfit_world_result_sha256
                            ),
                            "applied_after_observed_collider_refinement": True,
                            "applied_before_direct_collider_child_exclusion": True,
                            "applied_before_loading_group_projection": True,
                        },
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )
                crossfit_collider_result_file_hashes[candidate_name] = {
                    path.name: _file_sha256(path)
                    for path in sorted(crossfit_out.iterdir())
                    if path.is_file()
                }
            if structure_anchor is not None:
                anchor_out = candidate_out / "structure_anchored_loading_groups"
                anchor_out.mkdir(parents=True, exist_ok=True)
                anchor_tables = {
                    "structure_anchor_pair_evidence.tsv": (
                        structure_anchor.pair_evidence
                    ),
                    "typed_structure_anchored_groups.tsv": (
                        structure_anchor.typed_groups
                    ),
                    "assay_decisions.tsv": structure_anchor.assay_decisions,
                }
                for filename, frame in anchor_tables.items():
                    frame.to_csv(anchor_out / filename, sep="\t", index=False)
                (anchor_out / "structure_anchor_provenance.json").write_text(
                    json.dumps(
                        {
                            **structure_anchor.provenance,
                            "config": asdict(
                                FROZEN_STRUCTURE_ANCHORED_GROUP_CONFIG
                            ),
                            "state_membership_sha256": state_membership_sha256,
                            "result_sha256": structure_anchored_group_hashes[
                                candidate_name
                            ],
                            "applied_before_loading_group_projection": True,
                        },
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )
                structure_anchored_group_file_hashes[candidate_name] = {
                    path.name: _file_sha256(path)
                    for path in sorted(anchor_out.iterdir())
                    if path.is_file()
                }
            group_out = candidate_out / "loading_group_projection"
            group_out.mkdir(parents=True, exist_ok=True)
            group_tables = {
                "feature_consensus.tsv": group_projection.feature_consensus,
                "within_group_candidates.tsv": group_projection.within_group_candidates,
                "loading_groups.tsv": group_projection.groups,
                "loading_group_membership.tsv": group_projection.group_membership,
                "group_match_evidence.tsv": group_projection.group_match_evidence,
                "typed_group_hyperedges.tsv": group_projection.typed_hyperedges,
                "sparse_pair_projection.tsv": group_projection.sparse_pair_projection,
            }
            for filename, frame in group_tables.items():
                frame.to_csv(group_out / filename, sep="\t", index=False)
            (group_out / "projection_provenance.json").write_text(
                json.dumps(group_projection.provenance, indent=2, default=str)
            )
            axis_out = candidate_out / "latent_axis_projection"
            axis_out.mkdir(parents=True, exist_ok=True)
            axis_tables = {
                "group_pair_view_evidence.tsv": axis_projection.group_pair_view_evidence,
                "group_pair_evidence.tsv": axis_projection.group_pair_evidence,
                "axis_candidates.tsv": axis_projection.axis_candidates,
                "group_axis_view_evidence.tsv": axis_projection.group_axis_view_evidence,
                "group_axis_evidence.tsv": axis_projection.group_axis_evidence,
                "typed_axis_hyperedges.tsv": axis_projection.typed_axis_hyperedges,
                "representative_evidence.tsv": axis_projection.representative_evidence,
                "sparse_pair_projection.tsv": axis_projection.sparse_pair_projection,
            }
            for filename, frame in axis_tables.items():
                frame.to_csv(axis_out / filename, sep="\t", index=False)
            (axis_out / "axis_provenance.json").write_text(
                json.dumps(
                    {
                        **axis_projection.provenance,
                        "attached_once_post_consensus": True,
                        "expanded_into_ordinary_skeleton": False,
                        "axis_result_sha256": axis_projection_hashes[candidate_name],
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            axis_projection_file_hashes[candidate_name] = {
                path.name: _file_sha256(path)
                for path in sorted(axis_out.iterdir())
                if path.is_file()
            }
            consensus_file_hashes[candidate_name] = _file_sha256(
                candidate_out / "consensus_edges.tsv"
            )
            group_projection_file_hashes[candidate_name] = {
                path.name: _file_sha256(path)
                for path in sorted(group_out.iterdir())
                if path.is_file()
            }

    pretruth_manifest_path = world_out / "pretruth_consensus_manifest.json"
    pretruth_manifest_path.write_text(
        json.dumps(
            {
                "candidate_count": len(consensuses),
                "all_consensuses_built_before_truth_open": True,
                "consensus_sha256": consensus_hashes,
                "consensus_file_sha256": consensus_file_hashes,
                "collider_motif_beta_vote_sha256": beta_vote_hashes,
                "collider_motif_discovery_input_sha256": beta_input_hashes,
                "collider_motif_source_sha256": beta_source_hashes,
                "collider_motif_result_sha256": motif_result_hashes,
                "observed_collider_structure_result_sha256": (
                    observed_collider_refinement_hashes
                ),
                "observed_collider_structure_file_sha256": (
                    observed_collider_refinement_file_hashes
                ),
                "crossfit_hierarchical_collider_world_result_sha256": (
                    crossfit_world_result_sha256
                ),
                "crossfit_hierarchical_collider_world_file_sha256": (
                    crossfit_world_file_hashes
                ),
                "crossfit_hierarchical_collider_result_sha256": (
                    crossfit_collider_result_hashes
                ),
                "crossfit_hierarchical_collider_file_sha256": (
                    crossfit_collider_result_file_hashes
                ),
                "crossfit_hierarchical_collider_source_sha256": (
                    crossfit_source_hashes
                ),
                "structure_anchored_loading_result_sha256": (
                    structure_anchored_group_hashes
                ),
                "structure_anchored_loading_file_sha256": (
                    structure_anchored_group_file_hashes
                ),
                "structure_anchored_loading_source_sha256": (
                    structure_anchor_source_hashes
                ),
                "structure_anchored_loading_state_membership_sha256": (
                    state_membership_sha256
                ),
                "collider_motif_frozen_beta_config": asdict(
                    FROZEN_BETA_BINOMIAL_CONFIG
                ),
                "collider_motif_frozen_selector_config": asdict(
                    FROZEN_COLLIDER_MOTIF_CONFIG
                ),
                "observed_collider_structure_frozen_config": asdict(
                    FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG
                ),
                "crossfit_hierarchical_collider_frozen_config": asdict(
                    FROZEN_CROSSFIT_HIERARCHICAL_COLLIDER_CONFIG
                ),
                "structure_anchored_loading_frozen_config": asdict(
                    FROZEN_STRUCTURE_ANCHORED_GROUP_CONFIG
                ),
                "collider_motif_vote_count": COLLIDER_MOTIF_VOTES,
                "collider_motif_seed_contract": (
                    "vote + 10007 * (zero_based_sorted_assay_index + 1)"
                ),
                "collider_motif_discovery_inputs_only": True,
                "collider_motif_truth_pairing_external_inputs": False,
                "collider_motif_evidence_fitted_once_per_world": True,
                "collider_motif_applied_after_preliminary_nested_consensus": True,
                "collider_motif_applied_before_direct_collider_children": True,
                "collider_motif_applied_before_loading_group_projection": True,
                "observed_collider_structure_applied_before_loading_group_projection": True,
                "observed_collider_structure_discovery_inputs_only": True,
                "observed_collider_structure_truth_pairing_external_inputs": False,
                "crossfit_hierarchical_collider_applied": (
                    state_membership is not None
                ),
                "crossfit_hierarchical_collider_fit_once_per_world": (
                    state_membership is not None
                ),
                "crossfit_hierarchical_collider_randomization_key": (
                    int(pretruth_input_binding["seed"])
                    if state_membership is not None
                    else None
                ),
                "crossfit_hierarchical_collider_randomization_key_bound_to_world_input_seed": (
                    state_membership is not None
                ),
                "crossfit_hierarchical_collider_applied_after_observed_refinement": (
                    state_membership is not None
                ),
                "crossfit_hierarchical_collider_applied_before_direct_collider_children": (
                    state_membership is not None
                ),
                "crossfit_hierarchical_collider_applied_before_loading_group_projection": (
                    state_membership is not None
                ),
                "crossfit_hierarchical_collider_discovery_inputs_only": True,
                "crossfit_hierarchical_collider_truth_pairing_external_inputs": False,
                "crossfit_hierarchical_collider_paired_cells_used": False,
                "crossfit_hierarchical_collider_cross_assay_coordinate_registration_required": False,
                "structure_anchored_loading_applied_before_group_projection": (
                    state_membership is not None
                ),
                "structure_anchored_loading_baseline_and_legacy_rescue_use_"
                "formal_state_sibling_only": (
                    state_membership is not None
                ),
                "structure_anchored_loading_tiered_rescue_separates_assay_"
                "local_from_cross_assay_state": (state_membership is not None),
                "structure_anchored_loading_parent_membership_used_as_gate": False,
                "structure_anchored_loading_truth_pairing_external_inputs": False,
                "collider_motif_base_demotions_retained_as_unresolved": True,
                "observed_collider_rescue_only_completion_requires_state_loading_stability": True,
                "all_collider_motif_outputs_built_before_truth_open": True,
                "loading_group_projection_sha256": group_projection_hashes,
                "loading_group_projection_file_sha256": (
                    group_projection_file_hashes
                ),
                "latent_axis_projection_sha256": axis_projection_hashes,
                "latent_axis_projection_file_sha256": axis_projection_file_hashes,
                "loading_group_projection_attached_once_post_consensus": True,
                "loading_group_embedding_sha256": _file_sha256(
                    group_loading_embedding
                ),
                "loading_group_coordinates_cross_assay_aligned": (
                    loading_coordinates_cross_assay_aligned
                ),
                "loading_group_coordinates_used_for_cross_assay_matching": (
                    loading_coordinates_cross_assay_aligned
                ),
                "latent_axis_embedding_sha256": _file_sha256(embedding),
                "latent_axis_coordinates_cross_assay_aligned": True,
                "assay_local_group_and_cross_assay_axis_estimands_separated": (
                    not loading_coordinates_cross_assay_aligned
                ),
                "loading_views_reuse_same_observed_cells": True,
                "loading_views_are_algorithmic_stability_repeats": True,
                "loading_views_are_independent_worlds": False,
                "loading_view_support_not_world_sample_size": True,
                "latent_axis_hyperedges_built_before_truth_open": True,
                "latent_axis_hyperedges_attached_once_post_consensus": True,
                "latent_axis_hyperedges_expanded_into_pairwise_skeleton": False,
                "ordinary_state_skeleton_contains_within_assay_groups_only": True,
                "pairwise_cross_group_matches_are_diagnostics_only": True,
                "world_input_pretruth_binding": pretruth_input_binding,
                "world_input_discovery_payloads_content_addressed": True,
                "world_input_scorer_manifest_unopened_before_pretruth": True,
                "latent_axis_eligible_modalities": list(axis_eligible_modalities),
                "latent_axis_abstained_modalities": sorted(
                    set(MODALITIES) - set(axis_eligible_modalities)
                ),
                "preliminary_pairwise_state_layer_replaced": all(
                    cfg.replace_pairwise_state_layer_with_groups
                    for _, cfg in candidates
                ),
            },
            indent=2,
        )
    )
    frozen = FrozenWorldPretruth(
        world_dir=str(world_dir.resolve()),
        world_name=world_dir.name,
        world_out=str(world_out.resolve()),
        candidate_names=tuple(name for name, _ in candidates),
        pretruth_manifest=str(pretruth_manifest_path.resolve()),
        pretruth_manifest_sha256=_file_sha256(pretruth_manifest_path),
    )
    if defer_scorer:
        return frozen
    # The scorer domain is first inspected only after every candidate graph,
    # motif decision, group projection, and associated hash has been frozen.
    truth_frame, world_parameters, role_assignments = (
        load_scorer_artifacts_after_pretruth_freeze(world_dir, world_out)
    )
    result_rows: list[dict[str, Any]] = []
    for candidate_name, _ in candidates:
        consensus = consensuses[candidate_name]
        metrics, pair_scores = score_v4(consensus, truth_frame)
        latent_score = score_latent_loading_groups(
            world_parameters=world_parameters,
            role_assignments=role_assignments,
            typed_hyperedges=group_projections[
                candidate_name
            ].typed_hyperedges.loc[
                group_projections[candidate_name].typed_hyperedges.get(
                    "group2", pd.Series(dtype=str)
                ).fillna("").astype(str).eq("")
            ].copy(),
            typed_axis_hyperedges=(
                axis_projections[candidate_name].typed_axis_hyperedges
            ),
            sparse_edges=consensus,
        )
        combined_metrics = combine_attainable_metrics(metrics, latent_score.metrics)
        row = {
            "world": world_dir.name,
            "candidate": candidate_name,
            **combined_metrics,
        }
        for record in pair_scores.to_dict(orient="records"):
            row[f"present__{record['label']}"] = int(record["present"])
        result_rows.append(row)
        if candidate_name in save_candidate_names:
            candidate_out = world_out / candidate_name
            pair_scores.to_csv(candidate_out / "typed_pair_scores.tsv", sep="\t", index=False)
            (candidate_out / "metrics.json").write_text(
                json.dumps(combined_metrics, indent=2)
            )
            latent_score.hyperedge_scores.to_csv(
                candidate_out / "latent_group_hyperedge_scores.tsv",
                sep="\t",
                index=False,
            )
            latent_score.sparse_edge_scores.to_csv(
                candidate_out / "latent_group_edge_scores.tsv",
                sep="\t",
                index=False,
            )
            latent_score.axis_hyperedge_scores.to_csv(
                candidate_out / "latent_axis_hyperedge_scores.tsv",
                sep="\t",
                index=False,
            )
            (candidate_out / "latent_group_metrics.json").write_text(
                json.dumps(latent_score.metrics, indent=2)
            )
    return result_rows


def discover_world_pretruth(
    world_dir: Path,
    candidates: list[tuple[str, CABossV4Config]],
    *,
    out_dir: Path,
    save_candidate_names: set[str],
    loading_embedding_path: Path | None = None,
    state_membership_path: Path | None = None,
) -> FrozenWorldPretruth:
    """Run one world through artifact freeze without opening its scorer domain.

    Every candidate must be persisted because the later scoring process is
    intentionally independent of the discovery process and reloads only
    content-addressed files.
    """

    candidate_names = {name for name, _ in candidates}
    if not candidates:
        raise ValueError("Pretruth discovery requires at least one candidate")
    if set(save_candidate_names) != candidate_names:
        raise ValueError(
            "Two-phase discovery must persist every candidate before scorer access"
        )
    result = _evaluate_world_internal(
        world_dir,
        candidates,
        out_dir=out_dir,
        save_candidate_names=save_candidate_names,
        defer_scorer=True,
        loading_embedding_path=loading_embedding_path,
        state_membership_path=state_membership_path,
    )
    if not isinstance(result, FrozenWorldPretruth):
        raise RuntimeError("Pretruth discovery returned an unexpected result")
    return result


def evaluate_world(
    world_dir: Path,
    candidates: list[tuple[str, CABossV4Config]],
    *,
    out_dir: Path,
    save_candidate_names: set[str],
    loading_embedding_path: Path | None = None,
    state_membership_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible single-world TRAIN/exploration evaluation."""

    result = _evaluate_world_internal(
        world_dir,
        candidates,
        out_dir=out_dir,
        save_candidate_names=save_candidate_names,
        defer_scorer=False,
        loading_embedding_path=loading_embedding_path,
        state_membership_path=state_membership_path,
    )
    if isinstance(result, FrozenWorldPretruth):
        raise RuntimeError("World evaluation unexpectedly deferred its scorer")
    return result


def _validate_global_pretruth_freeze(
    frozen: FrozenWorldPretruth,
    manifest_path: Path,
) -> dict[str, Any]:
    """Bind one frozen world to a completed all-world discovery barrier."""

    manifest = _read_regular_json(
        manifest_path, label="global pretruth freeze manifest"
    )
    if manifest.get("schema_version") != GLOBAL_PRETRUTH_FREEZE_SCHEMA_VERSION:
        raise RuntimeError("Global pretruth freeze manifest has an unsupported schema")
    if manifest.get("all_registered_discovery_attempts_finished") is not True:
        raise RuntimeError("Global pretruth freeze does not cover all discovery attempts")
    expected_payload_hash = str(manifest.get("manifest_payload_sha256", ""))
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    if not _SHA256_PATTERN.fullmatch(expected_payload_hash) or (
        _canonical_sha256(payload) != expected_payload_hash
    ):
        raise RuntimeError("Global pretruth freeze manifest self-hash mismatch")
    registered = manifest.get("registered_world_names")
    records = manifest.get("world_records")
    if not isinstance(registered, list) or not isinstance(records, list):
        raise RuntimeError("Global pretruth freeze lacks its registered world records")
    registered_names = [str(value) for value in registered]
    if len(set(registered_names)) != len(registered_names):
        raise RuntimeError("Global pretruth freeze repeats a registered world")
    record_map: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("Global pretruth freeze has a malformed world record")
        name = str(record.get("world_name", ""))
        if not name or name in record_map:
            raise RuntimeError("Global pretruth freeze has duplicate world records")
        record_map[name] = record
    if set(record_map) != set(registered_names):
        raise RuntimeError("Global pretruth world records do not match registration")
    record = record_map.get(frozen.world_name)
    if record is None or record.get("status") != "ok":
        raise RuntimeError("Frozen world is not a successful global discovery attempt")
    if str(record.get("world_dir", "")) != frozen.world_dir:
        raise RuntimeError("Global pretruth freeze binds a different world directory")
    if str(record.get("pretruth_manifest_sha256", "")) != (
        frozen.pretruth_manifest_sha256
    ):
        raise RuntimeError("Global pretruth freeze binds a different world manifest")
    return manifest


def _load_frozen_scoring_inputs(
    frozen: FrozenWorldPretruth,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
]:
    """Load and verify all candidate bytes before any scorer-domain access."""

    world_out = Path(frozen.world_out)
    manifest_path = Path(frozen.pretruth_manifest)
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.parent.resolve() != world_out.resolve()
        or _file_sha256(manifest_path) != frozen.pretruth_manifest_sha256
    ):
        raise RuntimeError("Frozen world pretruth manifest changed before scoring")
    manifest = _read_regular_json(manifest_path, label="world pretruth manifest")
    if manifest.get("all_consensuses_built_before_truth_open") is not True:
        raise RuntimeError("World pretruth manifest lacks its consensus barrier")
    candidate_names = tuple(str(value) for value in frozen.candidate_names)
    if (
        not candidate_names
        or len(set(candidate_names)) != len(candidate_names)
        or any(Path(name).name != name for name in candidate_names)
    ):
        raise RuntimeError("Frozen candidate names are empty, repeated, or unsafe")
    if int(manifest.get("candidate_count", -1)) != len(candidate_names):
        raise RuntimeError("World pretruth candidate count differs from its handle")
    consensus_hashes = manifest.get("consensus_sha256")
    consensus_file_hashes = manifest.get("consensus_file_sha256")
    group_hashes = manifest.get("loading_group_projection_sha256")
    group_file_hashes = manifest.get("loading_group_projection_file_sha256")
    axis_file_hashes = manifest.get("latent_axis_projection_file_sha256")
    structure_anchor_applied = bool(
        manifest.get("structure_anchored_loading_applied_before_group_projection")
    )
    structure_anchor_file_hashes = manifest.get(
        "structure_anchored_loading_file_sha256"
    )
    crossfit_applied = bool(
        manifest.get("crossfit_hierarchical_collider_applied")
    )
    crossfit_result_hashes = manifest.get(
        "crossfit_hierarchical_collider_result_sha256"
    )
    crossfit_result_file_hashes = manifest.get(
        "crossfit_hierarchical_collider_file_sha256"
    )
    world_result_hash: str | None = None
    if not all(isinstance(value, dict) for value in (
        consensus_hashes,
        consensus_file_hashes,
        group_hashes,
        group_file_hashes,
        axis_file_hashes,
    )):
        raise RuntimeError("World pretruth manifest lacks candidate artifact hashes")
    if not all(set(value) == set(candidate_names) for value in (
        consensus_hashes,
        consensus_file_hashes,
        group_hashes,
        group_file_hashes,
        axis_file_hashes,
    )):
        raise RuntimeError("World pretruth candidate hashes do not match the frozen handle")
    if structure_anchor_applied:
        if not isinstance(structure_anchor_file_hashes, dict) or set(
            structure_anchor_file_hashes
        ) != set(candidate_names):
            raise RuntimeError(
                "World pretruth structure-anchor hashes do not match the frozen handle"
            )
        membership_hash = _expected_sha256(
            manifest.get("structure_anchored_loading_state_membership_sha256"),
            label="frozen state-membership hash",
        )
        membership_path = (
            world_out
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        )
        if (
            membership_path.is_symlink()
            or not membership_path.is_file()
            or _file_sha256(membership_path) != membership_hash
        ):
            raise RuntimeError(
                "Frozen state membership changed after structure-anchored selection"
            )
    if structure_anchor_applied and not crossfit_applied:
        raise RuntimeError(
            "Frozen formal state run lacks crossfit hierarchical collider evidence"
        )
    if crossfit_applied:
        world_result_hash = _expected_sha256(
            manifest.get(
                "crossfit_hierarchical_collider_world_result_sha256"
            ),
            label="crossfit hierarchical collider world-result hash",
        )
        if (
            not isinstance(crossfit_result_hashes, dict)
            or set(crossfit_result_hashes) != set(candidate_names)
            or not isinstance(crossfit_result_file_hashes, dict)
            or set(crossfit_result_file_hashes) != set(candidate_names)
        ):
            raise RuntimeError(
                "World pretruth crossfit collider hashes do not match the frozen handle"
            )
        _validate_crossfit_world_evidence_freeze(world_out, manifest)

    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for candidate_name in candidate_names:
        candidate_out = world_out / candidate_name
        if (
            candidate_out.is_symlink()
            or not candidate_out.is_dir()
            or candidate_out.resolve().parent != world_out.resolve()
        ):
            raise RuntimeError(f"Frozen candidate directory is unsafe: {candidate_name}")
        consensus_path = candidate_out / "consensus_edges.tsv"
        group_out = candidate_out / "loading_group_projection"
        group_hyperedges_path = group_out / "typed_group_hyperedges.tsv"
        group_sparse_path = group_out / "sparse_pair_projection.tsv"
        axis_out = candidate_out / "latent_axis_projection"
        axis_hyperedges_path = axis_out / "typed_axis_hyperedges.tsv"
        structure_anchor_out = candidate_out / "structure_anchored_loading_groups"
        crossfit_out = candidate_out / "crossfit_hierarchical_collider"
        required_paths = (
            consensus_path,
            group_hyperedges_path,
            group_sparse_path,
            axis_hyperedges_path,
        )
        if any(path.is_symlink() or not path.is_file() for path in required_paths):
            raise RuntimeError(f"Frozen candidate artifacts are incomplete: {candidate_name}")
        score_artifacts = (
            candidate_out / "typed_pair_scores.tsv",
            candidate_out / "metrics.json",
            candidate_out / "latent_group_hyperedge_scores.tsv",
            candidate_out / "latent_group_edge_scores.tsv",
            candidate_out / "latent_axis_hyperedge_scores.tsv",
            candidate_out / "latent_group_metrics.json",
        )
        if any(path.exists() for path in score_artifacts):
            raise RuntimeError(
                f"Frozen candidate was already scored: {candidate_name}"
            )
        consensus = pd.read_csv(consensus_path, sep="\t", keep_default_na=False)
        group_hyperedges = pd.read_csv(
            group_hyperedges_path, sep="\t", keep_default_na=False
        )
        group_sparse = pd.read_csv(
            group_sparse_path, sep="\t", keep_default_na=False
        )
        axis_hyperedges = pd.read_csv(
            axis_hyperedges_path, sep="\t", keep_default_na=False
        )
        if _file_sha256(consensus_path) != str(
            consensus_file_hashes[candidate_name]
        ):
            raise RuntimeError(f"Frozen consensus changed: {candidate_name}")
        expected_group_files = group_file_hashes[candidate_name]
        if not isinstance(expected_group_files, dict) or not expected_group_files:
            raise RuntimeError(
                f"Frozen loading-group file hashes are missing: {candidate_name}"
            )
        observed_group_files = {
            path.name: _file_sha256(path)
            for path in sorted(group_out.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        if observed_group_files != expected_group_files:
            raise RuntimeError(f"Frozen loading-group projection changed: {candidate_name}")
        expected_axis_files = axis_file_hashes[candidate_name]
        if not isinstance(expected_axis_files, dict) or not expected_axis_files:
            raise RuntimeError(f"Frozen latent-axis file hashes are missing: {candidate_name}")
        observed_axis_files = {
            path.name: _file_sha256(path)
            for path in sorted(axis_out.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        if observed_axis_files != expected_axis_files:
            raise RuntimeError(f"Frozen latent-axis projection changed: {candidate_name}")
        if structure_anchor_applied:
            expected_anchor_files = structure_anchor_file_hashes[candidate_name]
            if (
                not isinstance(expected_anchor_files, dict)
                or not expected_anchor_files
                or structure_anchor_out.is_symlink()
                or not structure_anchor_out.is_dir()
            ):
                raise RuntimeError(
                    f"Frozen structure-anchor hashes are missing: {candidate_name}"
                )
            observed_anchor_files = {
                path.name: _file_sha256(path)
                for path in sorted(structure_anchor_out.iterdir())
                if path.is_file() and not path.is_symlink()
            }
            if observed_anchor_files != expected_anchor_files:
                raise RuntimeError(
                    f"Frozen structure-anchored loading evidence changed: {candidate_name}"
                )
        if crossfit_applied:
            expected_crossfit_files = crossfit_result_file_hashes[candidate_name]
            expected_crossfit_result = _expected_sha256(
                crossfit_result_hashes[candidate_name],
                label=f"crossfit collider result hash for {candidate_name}",
            )
            if (
                not isinstance(expected_crossfit_files, dict)
                or not expected_crossfit_files
                or crossfit_out.is_symlink()
                or not crossfit_out.is_dir()
            ):
                raise RuntimeError(
                    f"Frozen crossfit collider hashes are missing: {candidate_name}"
                )
            required_crossfit_files = {
                "candidate_evidence.tsv",
                "assay_decisions.tsv",
                "direct_candidates.tsv",
                "demoted_or_unresolved.tsv",
                "provenance.json",
            }
            crossfit_entries = list(crossfit_out.iterdir())
            if (
                any(
                    path.is_symlink() or not path.is_file()
                    for path in crossfit_entries
                )
                or {path.name for path in crossfit_entries}
                != required_crossfit_files
            ):
                raise RuntimeError(
                    f"Frozen crossfit collider evidence has an unsafe entry set: {candidate_name}"
                )
            observed_crossfit_files = {
                path.name: _file_sha256(path)
                for path in sorted(crossfit_entries)
            }
            if (
                observed_crossfit_files != expected_crossfit_files
            ):
                raise RuntimeError(
                    f"Frozen crossfit collider evidence changed: {candidate_name}"
                )
            crossfit_provenance = _read_regular_json(
                crossfit_out / "provenance.json",
                label=f"crossfit collider provenance for {candidate_name}",
            )
            if (
                crossfit_provenance.get("result_sha256")
                != expected_crossfit_result
                or crossfit_provenance.get("world_evidence_result_sha256")
                != world_result_hash
            ):
                raise RuntimeError(
                    f"Frozen crossfit collider provenance changed: {candidate_name}"
                )
        loaded[candidate_name] = (consensus, group_hyperedges, axis_hyperedges)
    return world_out, manifest, loaded


def score_frozen_world(
    frozen: FrozenWorldPretruth,
    *,
    global_freeze_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Score a persisted world only after an optional all-world freeze barrier."""

    if global_freeze_manifest is not None:
        _validate_global_pretruth_freeze(frozen, global_freeze_manifest)
    world_out, _, loaded = _load_frozen_scoring_inputs(frozen)
    world_dir = Path(frozen.world_dir)
    if world_dir.name != frozen.world_name:
        raise RuntimeError("Frozen world name does not match its directory")
    # This is deliberately the first scorer-domain access in this function.
    truth_frame, world_parameters, role_assignments = (
        load_scorer_artifacts_after_pretruth_freeze(world_dir, world_out)
    )
    rows: list[dict[str, Any]] = []
    for candidate_name in frozen.candidate_names:
        consensus, group_hyperedges, axis_hyperedges = loaded[candidate_name]
        metrics, pair_scores = score_v4(consensus, truth_frame)
        group2 = group_hyperedges.get(
            "group2", pd.Series("", index=group_hyperedges.index, dtype=str)
        )
        latent_score = score_latent_loading_groups(
            world_parameters=world_parameters,
            role_assignments=role_assignments,
            typed_hyperedges=group_hyperedges.loc[
                group2.fillna("").astype(str).eq("")
            ].copy(),
            typed_axis_hyperedges=axis_hyperedges,
            sparse_edges=consensus,
        )
        combined_metrics = combine_attainable_metrics(metrics, latent_score.metrics)
        row = {
            "world": frozen.world_name,
            "candidate": candidate_name,
            **combined_metrics,
        }
        for record in pair_scores.to_dict(orient="records"):
            row[f"present__{record['label']}"] = int(record["present"])
        rows.append(row)
        candidate_out = world_out / candidate_name
        scored_paths = (
            candidate_out / "typed_pair_scores.tsv",
            candidate_out / "metrics.json",
            candidate_out / "latent_group_hyperedge_scores.tsv",
            candidate_out / "latent_group_edge_scores.tsv",
            candidate_out / "latent_axis_hyperedge_scores.tsv",
            candidate_out / "latent_group_metrics.json",
        )
        pair_scores.to_csv(scored_paths[0], sep="\t", index=False)
        scored_paths[1].write_text(json.dumps(combined_metrics, indent=2))
        latent_score.hyperedge_scores.to_csv(
            scored_paths[2], sep="\t", index=False
        )
        latent_score.sparse_edge_scores.to_csv(
            scored_paths[3], sep="\t", index=False
        )
        latent_score.axis_hyperedge_scores.to_csv(
            scored_paths[4], sep="\t", index=False
        )
        scored_paths[5].write_text(json.dumps(latent_score.metrics, indent=2))
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "world_success",
        "direct_raw_adjacency_macro_recall",
        "direct_typed_candidate_macro_recall",
        "direct_adjacency_macro_recall",
        "collider_arrowhead_recall",
        "shared_state_adjacency_recall",
        "slow_peak_gene_present",
        "matched_null_false_positive_rate",
        "primary_skeleton_precision",
        "primary_skeleton_recall",
        "primary_skeleton_f1",
        "n_estimated_focal_edges",
        "strict_reverse_compelled_count",
        "shared_state_false_one_way_count",
        "fast_challenge_present",
        "selected_group_definition_axis_precision",
        "selected_hyperedge_axis_precision",
        "selected_hyperedge_typed_precision",
        "selected_latent_axis_purity_precision",
        "selected_latent_axis_axis_precision",
        "selected_latent_axis_typed_precision",
        "native_latent_axis_dimension_coverage",
        "native_slow_rna_axis_support",
        "selected_loading_edge_axis_precision",
        "selected_loading_edge_typed_precision",
        "combined_within_assay_ps_group_recall",
        "combined_cross_assay_latent_axis_coverage",
        "combined_slow_gene_axis_support",
        "n_selected_axis_mismatch_claims",
        "n_selected_wrong_layer_claims",
    ]
    summary = results.groupby("candidate")[metrics].mean().reset_index()
    summary["successes"] = (
        results.groupby("candidate")["world_success"].sum().astype(int).to_numpy()
    )
    summary["n_worlds"] = results.groupby("candidate").size().to_numpy()
    return summary.sort_values(
        [
            "world_success",
            "matched_null_false_positive_rate",
            "primary_skeleton_f1",
            "n_estimated_focal_edges",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds-glob", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--residual-permutations", type=int, default=199)
    parser.add_argument("--stability-bootstraps", type=int, default=200)
    parser.add_argument("--alignment-mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--soft-n-microstates", type=int, default=24)
    parser.add_argument("--soft-max-rotation-dispersion", type=float, default=0.35)
    parser.add_argument(
        "--soft-low-confidence-action",
        choices=("partial_soft", "fallback_hard", "abstain"),
        default="partial_soft",
    )
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument(
        "--save-candidate",
        action="append",
        default=[],
        help="Candidate name whose nested and consensus graphs should be retained.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    world_dirs = [Path(path) for path in sorted(glob.glob(args.worlds_glob))]
    if not world_dirs:
        raise SystemExit(f"No worlds matched {args.worlds_glob}")
    base = CABossV4Config(
        seed=args.seed,
        residual_permutations=args.residual_permutations,
        stability_bootstraps=args.stability_bootstraps,
        alignment_mode=args.alignment_mode,
        soft_n_microstates=args.soft_n_microstates,
        soft_max_rotation_dispersion=args.soft_max_rotation_dispersion,
        soft_low_confidence_action=args.soft_low_confidence_action,
    )
    candidates = candidate_grid(base)
    if args.candidate:
        wanted = set(args.candidate)
        candidates = [item for item in candidates if item[0] in wanted]
        missing = wanted - {item[0] for item in candidates}
        if missing:
            raise SystemExit(f"Unknown candidates: {sorted(missing)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for world_i, world_dir in enumerate(world_dirs):
            print(f"[{world_i + 1}/{len(world_dirs)}] {world_dir.name}", flush=True)
            rows.extend(
                evaluate_world(
                    world_dir,
                    candidates,
                    out_dir=args.out_dir,
                    save_candidate_names=set(args.save_candidate),
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(
                    evaluate_world,
                    world_dir,
                    candidates,
                    out_dir=args.out_dir,
                    save_candidate_names=set(args.save_candidate),
                ): world_dir
                for world_dir in world_dirs
            }
            completed = 0
            for future in as_completed(pending):
                world_dir = pending[future]
                rows.extend(future.result())
                completed += 1
                print(f"[{completed}/{len(world_dirs)}] {world_dir.name}", flush=True)
    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(args.out_dir / "world_results.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "candidate_summary.tsv", sep="\t", index=False)
    world_input_bindings = [
        validate_world_input_binding(path, verify_scorer_payload=True)
        for path in world_dirs
    ]
    manifest = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "worlds_glob": args.worlds_glob,
        "worlds": [record["world"] for record in world_input_bindings],
        "world_input_binding_schema_version": WORLD_INPUT_BINDING_SCHEMA_VERSION,
        "world_input_bindings": world_input_bindings,
        "world_input_bindings_canonical_sha256": _canonical_sha256(
            world_input_bindings
        ),
        "all_world_inputs_content_addressed_and_verified": True,
        "base_engine": asdict(base),
        "workers": int(args.workers),
        "candidates": [
            {"name": name, "config": asdict(config)} for name, config in candidates
        ],
        "truth_opened_after_consensus_only": True,
        "truth_opened_after_all_candidate_consensuses": True,
        "truth_barrier_scope": "per_world_training_benchmark",
        "formal_all_world_barrier_available_via_two_phase_api": True,
        "information_domains": {"discovery": "discovery/", "scorer": "scorer/"},
        "hierarchical_evidence_refit_per_nested_vote": True,
        "cell_loading_evidence_refit_per_nested_vote": True,
        "collider_motif_beta_binomial_evidence_refit_per_vote": True,
        "collider_motif_beta_binomial_votes": COLLIDER_MOTIF_VOTES,
        "collider_motif_evidence_fitted_once_per_world": True,
        "collider_motif_applied_post_nested_consensus": True,
        "collider_motif_applied_before_loading_group_projection": True,
        "collider_motif_base_demotions_retained_as_unresolved": True,
        "observed_collider_rescue_only_completion_requires_state_loading_stability": True,
        "collider_motif_frozen_beta_config": asdict(FROZEN_BETA_BINOMIAL_CONFIG),
        "collider_motif_frozen_selector_config": asdict(
            FROZEN_COLLIDER_MOTIF_CONFIG
        ),
        "observed_collider_structure_frozen_config": asdict(
            FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG
        ),
        "loading_group_projection_attached_once_post_consensus": True,
        "loading_views_reuse_same_observed_cells": True,
        "loading_views_are_algorithmic_stability_repeats": True,
        "loading_views_are_independent_worlds": False,
        "loading_view_support_not_world_sample_size": True,
        "latent_axis_hyperedges_attached_once_post_consensus": True,
        "latent_axis_hyperedges_expanded_into_pairwise_skeleton": False,
        "ordinary_state_skeleton_contains_within_assay_groups_only": True,
        "pairwise_cross_group_matches_are_diagnostics_only": True,
        "latent_axis_clustering_config": asdict(LatentAxisClusteringConfig()),
        "preliminary_pairwise_state_layer_replaced": bool(
            base.replace_pairwise_state_layer_with_groups
        ),
        "direct_collider_children_excluded_from_state_groups": bool(
            base.exclude_direct_collider_children_from_state_groups
        ),
        "supplied_alignment_ignored_and_recomputed_from_raw_background": True,
    }
    (args.out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
