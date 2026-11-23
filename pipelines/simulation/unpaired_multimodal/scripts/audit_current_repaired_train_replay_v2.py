#!/usr/bin/env python3
"""Two-phase, development-only replay of the current repaired CA-BOSS v4.

This audit is deliberately restricted to the already opened TRAIN330,
TRAIN360, TRAIN370, TRAIN380, and TRAIN400 blocks.  Phase A reads observable
artifacts only, reruns the current strict-state/assay-participation, direct,
and loading decisions, writes every prediction, and content-addresses the
complete observed input/output closure.  Phase B must be invoked in a new
process; it verifies that freeze before opening the five TRAIN scorer
snapshots.

The result is posthoc development evidence, never a certification result.
The script rejects DEV340, LOCKED450, TRAIN410, and all unregistered paths.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_retired_combined_replay_v1 import (  # noqa: E402
    _apply_state_v7_filter,
    _loading_projection_config,
    _pair_members,
)
from crossfit_hierarchical_collider_v4 import (  # noqa: E402
    SCHEMA_VERSION as DIRECT_SCHEMA_VERSION,
    UNORIENTED_RESCUE_OVERLAP_ACTION,
    _annotate_unique_rescue_rank_overlap_adjacencies,
    select_crossfit_hierarchical_collider_replacements,
)
from loading_group_projection_v4 import (  # noqa: E402
    estimate_loading_group_projection,
    load_evidence_view,
)
from score_layered_state_evidence_v4 import (  # noqa: E402
    MAX_BLOCK_MEAN_FOCAL_DENSITY,
    MAX_WORLD_FOCAL_DENSITY,
    MIN_BLOCK_MEAN_DIRECT_RECALL,
    MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION,
    MIN_BLOCK_MEAN_LOADING_PAIR_RECALL,
    MIN_CONFIRMED_SUBSET_PRECISION,
    MIN_DIRECT_PRECISION,
    MIN_LOADING_PAIR_PRECISION,
    MIN_WORLD_DIRECT_RECALL,
    evaluate_layered_world_reliability,
)
from shared_state_subspace_v4 import (  # noqa: E402
    SharedStateSubspaceConfig,
    _select_assay_state_participation,
)
from structure_anchored_loading_groups_v4 import (  # noqa: E402
    StructureAnchoredGroupConfig,
    estimate_structure_anchored_groups,
)


SCHEMA = "ca_boss_v4.current_repaired_train_replay.v4"
STATE_ROLES = frozenset({"P", "Q", "S", "C"})
NULL_ROLES = frozenset({"N1", "N2"})
EXPECTED_MODALITIES = 7
MIN_RELIABLE_WORLDS = 22
PHASE_A_OUTPUT_NAMES = frozenset(
    {
        "phase_a_state_membership.tsv",
        "phase_a_assay_participation.tsv",
        "phase_a_subspace_hyperedges.tsv",
        "phase_a_direct_decisions.tsv",
        "phase_a_direct_edges.tsv",
        "phase_a_prior_direct_edges.tsv",
        "phase_a_loading_groups.tsv",
        "phase_a_loading_decisions.tsv",
        "phase_a_prior_loading_groups.tsv",
        "phase_a_changed_decisions.tsv",
    }
)

RETIRED_DIRECT_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "retired_crossfit_hierarchical_collider_audit_v1"
)
PRIOR_RETIRED_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/retired_combined_replay_v1"
)


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    end: int
    run_root: Path
    world_root: Path

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1))


BLOCKS = (
    Block(
        "TRAIN330",
        33_001,
        33_024,
        Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_v1"),
        Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_worlds_v1"),
    ),
    Block(
        "TRAIN360",
        36_001,
        36_024,
        Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_36001_36024"),
        Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_36001_36024"),
    ),
    Block(
        "TRAIN370",
        37_001,
        37_024,
        Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_37001_37024"),
        Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_37001_37024"),
    ),
    Block(
        "TRAIN380",
        38_001,
        38_024,
        Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_38001_38024"),
        Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_38001_38024"),
    ),
    Block(
        "TRAIN400",
        40_001,
        40_024,
        Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_40001_40024"),
        Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_40001_40024"),
    ),
)
ALLOWED_SEEDS = frozenset(seed for block in BLOCKS for seed in block.seeds)
FORBIDDEN_PATTERN = re.compile(
    r"(?:^|[/_])(?:dev(?:elopment)?|locked|train[_-]?340|train[_-]?410|"
    r"train[_-]?450|fresh[_-]?410)(?:$|[/_0-9-])",
    flags=re.IGNORECASE,
)
TRUTH_PARTS = frozenset({"scorer", "truth", "role_assignments", "truth_typed_edges"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Not a strict Boolean: {value!r}")


def _assert_path(path: Path, *, allow_truth: bool = False) -> Path:
    lexical = str(path.resolve(strict=False))
    if FORBIDDEN_PATTERN.search(lexical):
        raise ValueError(f"Refusing protected path: {path}")
    folded = {part.casefold() for part in Path(lexical).parts}
    if not allow_truth and folded & TRUTH_PARTS:
        raise ValueError(f"Phase A refuses scorer/truth path: {path}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _snapshot_regular_bytes(
    path: Path, *, allow_truth: bool = False
) -> tuple[Path, bytes]:
    safe = _assert_path(path, allow_truth=allow_truth)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(safe, flags)
    except OSError as exc:
        raise ValueError(f"Could not open regular file safely: {safe}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Expected a regular non-symlink file: {safe}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return safe, handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_tsv_bytes(payload: bytes, *, allow_empty: bool = False) -> pd.DataFrame:
    if not payload.strip():
        if allow_empty:
            return pd.DataFrame()
        raise ValueError("Observed TSV is empty")
    return pd.read_csv(
        io.BytesIO(payload),
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )


def _read_tsv(
    path: Path,
    hashes: dict[str, str],
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    safe, payload = _snapshot_regular_bytes(path)
    hashes[str(safe)] = _bytes_sha256(payload)
    return _parse_tsv_bytes(payload, allow_empty=allow_empty)


def _read_json(path: Path, hashes: dict[str, str]) -> dict[str, Any]:
    safe, payload = _snapshot_regular_bytes(path)
    hashes[str(safe)] = _bytes_sha256(payload)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _candidate_dir(run_root: Path, seed: int) -> Path:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Unregistered seed: {seed}")
    world = run_root / "artifacts" / f"world_seed{seed}"
    candidates = sorted(
        path
        for path in world.glob("v4_*")
        if path.is_dir()
        and not path.is_symlink()
        and (path / "loading_group_projection").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one frozen candidate for seed {seed}; found {len(candidates)}"
        )
    return candidates[0]


def _current_state_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize older observed state tables to the current strict rule."""

    if "pre_redundancy_selection_status" in frame:
        output = frame.copy()
    else:
        output = _apply_state_v7_filter(frame)
    participation = _select_assay_state_participation(
        output, SharedStateSubspaceConfig()
    ).set_index("modality")
    modalities = output["modality"].astype(str)
    output["assay_participation_partial_conjunction_p"] = modalities.map(
        participation["partial_conjunction_empirical_p"]
    ).astype(float)
    output["assay_participation_holm_p"] = modalities.map(
        participation["assay_participation_holm_p"]
    ).astype(float)
    output["assay_participation_single_feature_order_statistic_p"] = (
        modalities.map(
            participation["single_feature_order_statistic_p"]
        ).astype(float)
    )
    output["assay_participation_single_feature_p"] = modalities.map(
        participation["single_feature_bonferroni_p"]
    ).astype(float)
    output["assay_participation_single_feature_witness_view_count"] = (
        modalities.map(
            participation["single_feature_witness_view_count"]
        ).astype(int)
    )
    output[
        "assay_participation_single_feature_multiview_replication_gate"
    ] = modalities.map(
        participation["single_feature_multiview_replication_gate"]
    ).astype(bool)
    output[
        "assay_participation_partial_conjunction_multiview_replication_gate"
    ] = modalities.map(
        participation["partial_conjunction_multiview_replication_gate"]
    ).astype(bool)
    output["assay_participation_omnibus_route_count"] = modalities.map(
        participation["assay_participation_omnibus_route_count"]
    ).astype(int)
    output["assay_participation_omnibus_p"] = modalities.map(
        participation["assay_participation_omnibus_p"]
    ).astype(float)
    output["assay_participation_omnibus_holm_p"] = modalities.map(
        participation["assay_participation_omnibus_holm_p"]
    ).astype(float)
    output["assay_participation_winning_route"] = modalities.map(
        participation["assay_participation_winning_route"]
    ).astype(str)
    output["assay_participation_selection_route"] = modalities.map(
        participation["assay_participation_selection_route"]
    ).astype(str)
    output["assay_participation_selected"] = modalities.map(
        participation["selection_status"].eq("selected")
    ).astype(bool)
    output["assay_participation_order"] = modalities.map(
        participation["partial_conjunction_order"]
    ).astype(int)
    output["assay_participation_family_size"] = modalities.map(
        participation["n_formally_eligible_features"]
    ).astype(int)
    output["assay_participation_contributing_feature_count"] = modalities.map(
        participation["contributing_feature_count"]
    ).astype(int)
    output["assay_participation_minimum_witness_view_count"] = modalities.map(
        participation["minimum_witness_view_count"]
    ).astype(int)
    output["assay_participation_multiview_replication_gate"] = modalities.map(
        participation["multiview_replication_gate"]
    ).astype(bool)
    reporters = modalities.map(participation["selected_reporter"]).astype(str)
    output["assay_participation_reporter"] = (
        output["assay_participation_selected"]
        & output["feature"].astype(str).eq(reporters)
    )
    return output


def _participation_table(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality, group in state.groupby("modality", sort=True):
        first = group.iloc[0]
        reporter = group.loc[
            group["assay_participation_reporter"].map(_strict_bool), "feature"
        ].astype(str)
        rows.append(
            {
                "modality": str(modality),
                "partial_conjunction_p": float(
                    first["assay_participation_partial_conjunction_p"]
                ),
                "holm_p": float(first["assay_participation_holm_p"]),
                "single_feature_order_statistic_p": float(
                    first[
                        "assay_participation_single_feature_order_statistic_p"
                    ]
                ),
                "single_feature_p": float(
                    first["assay_participation_single_feature_p"]
                ),
                "single_feature_witness_view_count": int(
                    first[
                        "assay_participation_single_feature_witness_view_count"
                    ]
                ),
                "single_feature_multiview_replication_gate": _strict_bool(
                    first[
                        "assay_participation_single_feature_multiview_"
                        "replication_gate"
                    ]
                ),
                "partial_conjunction_multiview_replication_gate": _strict_bool(
                    first[
                        "assay_participation_partial_conjunction_multiview_"
                        "replication_gate"
                    ]
                ),
                "omnibus_route_count": int(
                    first["assay_participation_omnibus_route_count"]
                ),
                "omnibus_p": float(
                    first["assay_participation_omnibus_p"]
                ),
                "omnibus_holm_p": float(
                    first["assay_participation_omnibus_holm_p"]
                ),
                "winning_route": str(
                    first["assay_participation_winning_route"]
                ),
                "selection_route": str(
                    first["assay_participation_selection_route"]
                ),
                "selected": _strict_bool(first["assay_participation_selected"]),
                "order": int(first["assay_participation_order"]),
                "family_size": int(first["assay_participation_family_size"]),
                "contributing_feature_count": int(
                    first["assay_participation_contributing_feature_count"]
                ),
                "minimum_witness_view_count": int(
                    first["assay_participation_minimum_witness_view_count"]
                ),
                "multiview_replication_gate": _strict_bool(
                    first["assay_participation_multiview_replication_gate"]
                ),
                "reporter": reporter.iloc[0] if len(reporter) else "",
                "strict_feature_present": bool(
                    group["selection_status"].astype(str).eq("selected").any()
                ),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != EXPECTED_MODALITIES:
        raise ValueError("Every world must contain exactly seven modalities")
    return output


def _production_direct_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in frame.itertuples(index=False):
        rows.append(
            {
                "source_assay": str(record.source_assay),
                "node1": str(record.parent),
                "endpoint1": "TAIL",
                "node2": str(record.child),
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "collider_center": str(record.child),
            }
        )
    return pd.DataFrame(rows)


def _train400_production(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in decisions.itertuples(index=False):
        if not _strict_bool(record.production_exact_two_arm_star):
            continue
        for parent in (record.production_outer1, record.production_outer2):
            rows.append(
                {
                    "source_assay": str(record.source_assay),
                    "node1": str(parent),
                    "endpoint1": "TAIL",
                    "node2": str(record.production_center),
                    "endpoint2": "ARROW",
                    "edge_type": "within_assay_direct_candidate",
                    "collider_center": str(record.production_center),
                }
            )
    return pd.DataFrame(rows)


def _edge_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    has_complete_parent_child = bool(
        {"parent", "child"} <= set(frame)
        and not frame.empty
        and frame[["parent", "child"]].notna().all().all()
        and frame["parent"].astype(str).str.strip().ne("").all()
        and frame["child"].astype(str).str.strip().ne("").all()
    )
    if has_complete_parent_child:
        return {
            (str(row.source_assay), str(row.parent), str(row.child))
            for row in frame.itertuples(index=False)
        }
    output: set[tuple[str, str, str]] = set()
    for row in frame.itertuples(index=False):
        if str(row.endpoint1).upper() == "TAIL" and str(row.endpoint2).upper() == "ARROW":
            output.add((str(row.source_assay), str(row.node1), str(row.node2)))
        elif str(row.endpoint2).upper() == "TAIL" and str(row.endpoint1).upper() == "ARROW":
            output.add((str(row.source_assay), str(row.node2), str(row.node1)))
    return output


def _observed_action(
    candidate: Path,
    assay: str,
    parent1: str,
    parent2: str,
    center: str,
    hashes: dict[str, str],
) -> tuple[str, str]:
    path = candidate / "collider_motif" / "direct_candidates.tsv"
    frame = _read_tsv(path, hashes)
    assay_frame = frame.loc[frame["source_assay"].astype(str).eq(assay)].copy()
    if _edge_set(assay_frame) != {
        (assay, parent1, center),
        (assay, parent2, center),
    }:
        return "", ""
    component_ids = {
        str(value).strip()
        for value in assay_frame.get("component_id", pd.Series(dtype=str))
        if str(value).strip()
    }
    component_id = next(iter(component_ids)) if len(component_ids) == 1 else ""
    if "observed_refinement_action" in assay_frame:
        values = {
            str(value).strip()
            for value in assay_frame["observed_refinement_action"]
            if str(value).strip()
        }
        if len(values) <= 1:
            return next(iter(values), ""), component_id
    values: set[str] = set()
    for interpretation in assay_frame.get("interpretation", pd.Series(dtype=str)):
        match = re.search(r"\(([^()]*)\)\s*$", str(interpretation))
        if match:
            values.add(match.group(1))
    return (next(iter(values)) if len(values) == 1 else ""), component_id


def _resolved_direct_frame(
    decisions: pd.DataFrame,
    candidate: Path,
    hashes: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in decisions.itertuples(index=False):
        selected = _strict_bool(record.selected_replacement)
        demoted = _strict_bool(record.demote_production_motif)
        unoriented = _strict_bool(record.selected_unoriented_adjacency)
        if int(selected) + int(demoted) + int(unoriented) > 1:
            raise ValueError(
                "A decision cannot replace, demote, and select an unoriented "
                "adjacency at the same time"
            )
        if unoriented:
            if str(record.action) != UNORIENTED_RESCUE_OVERLAP_ACTION:
                raise ValueError("Unoriented adjacency lacks its explicit action")
            rows.append(
                {
                    "source_assay": str(record.source_assay),
                    "node1": str(record.unoriented_adjacency_node1),
                    "endpoint1": "CIRCLE",
                    "node2": str(record.unoriented_adjacency_node2),
                    "endpoint2": "CIRCLE",
                    "edge_type": "within_assay_direct_candidate",
                    "collider_center": "",
                    "observed_refinement_action": "",
                    "component_id": "",
                    "crossfit_hierarchical_action": str(record.action),
                    "unoriented_adjacency_only": True,
                    "direction_claimed": False,
                    "collider_center_claimed": False,
                }
            )
            continue
        if demoted:
            if not _strict_bool(record.production_exact_two_arm_star):
                raise ValueError("Demotion requires an exact production motif")
            if str(record.action) != (
                "demote_retained_orientation_ambiguous_production_motif"
            ):
                raise ValueError("Demotion lacks its explicit action")
            # The production resolver preserves these two adjacencies only in
            # its CIRCLE--CIRCLE unresolved table.  The replay's strict direct
            # table must not silently reconstruct them as TAIL--ARROW edges.
            continue
        if selected:
            center = str(record.selected_center)
            parents = (str(record.selected_outer1), str(record.selected_outer2))
            observed_action = ""
            component_id = ""
        elif _strict_bool(record.production_exact_two_arm_star):
            center = str(record.production_center)
            parents = (str(record.production_outer1), str(record.production_outer2))
            observed_action, component_id = _observed_action(
                candidate,
                str(record.source_assay),
                parents[0],
                parents[1],
                center,
                hashes,
            )
        else:
            continue
        for parent in parents:
            rows.append(
                {
                    "source_assay": str(record.source_assay),
                    "node1": parent,
                    "endpoint1": "TAIL",
                    "node2": center,
                    "endpoint2": "ARROW",
                    "edge_type": "within_assay_direct_candidate",
                    "collider_center": center,
                    "observed_refinement_action": observed_action,
                    "component_id": component_id,
                    "crossfit_hierarchical_action": str(record.action),
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty and output.duplicated(
        ["source_assay", "node1", "node2"]
    ).any():
        raise ValueError("Resolved direct graph contains duplicate edges")
    return output


def _parent_constraints(direct: pd.DataFrame) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    centers: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for (_, center), group in direct.groupby(
        ["source_assay", "collider_center"], sort=True
    ):
        parents = sorted(
            node
            for row in group.itertuples(index=False)
            for node in (str(row.node1), str(row.node2))
            if node != str(center)
        )
        parents = sorted(set(parents))
        if len(group) == 2 and len(parents) == 2:
            centers.add(str(center))
            pairs.add((parents[0], parents[1]))
    return tuple(sorted(centers)), tuple(sorted(pairs))


def _selected_loading(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    output: set[tuple[str, str, str]] = set()
    if frame.empty:
        return output
    selected = frame.loc[frame["selection_status"].astype(str).eq("selected")]
    for row in selected.itertuples(index=False):
        members = sorted(
            value.strip() for value in str(row.members).split(";") if value.strip()
        )
        if len(members) != 2:
            raise ValueError("Loading relation is not a pair")
        output.add((str(row.modality), members[0], members[1]))
    return output


def _tag(frame: pd.DataFrame, block: str, seed: int) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "seed", seed)
    output.insert(0, "block", block)
    return output


def _phase_a(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_hashes: dict[str, str] = {}

    retired_manifest = _read_json(
        RETIRED_DIRECT_ROOT / "phase_a_manifest.json", input_hashes
    )
    if retired_manifest.get("truth_opened") is True or retired_manifest.get(
        "dev_or_locked_opened"
    ) is True:
        raise ValueError("Retired direct universe lacks an observed-only barrier")
    retired_candidates = _read_tsv(
        RETIRED_DIRECT_ROOT / "phase_a_crossfit_star_candidates.tsv", input_hashes
    )
    retired_production = _read_tsv(
        RETIRED_DIRECT_ROOT / "phase_a_production_direct_edges.tsv", input_hashes
    )
    prior_retired_direct = _read_tsv(
        PRIOR_RETIRED_ROOT / "phase_a_direct_edges.tsv", input_hashes
    )
    prior_retired_loading = _read_tsv(
        PRIOR_RETIRED_ROOT / "phase_a_loading_groups.tsv", input_hashes
    )

    state_frames: list[pd.DataFrame] = []
    participation_frames: list[pd.DataFrame] = []
    subspace_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    direct_frames: list[pd.DataFrame] = []
    prior_direct_frames: list[pd.DataFrame] = []
    loading_frames: list[pd.DataFrame] = []
    loading_decision_frames: list[pd.DataFrame] = []
    prior_loading_frames: list[pd.DataFrame] = []
    change_rows: list[dict[str, Any]] = []
    projection_config = _loading_projection_config()
    anchor_config = StructureAnchoredGroupConfig()

    for block in BLOCKS:
        _read_json(block.run_root / "global_pretruth_freeze.json", input_hashes)
        for seed in block.seeds:
            world = block.run_root / "artifacts" / f"world_seed{seed}"
            candidate = _candidate_dir(block.run_root, seed)
            state_path = (
                world / "shared_state_subspace_v4" / "subspace_feature_membership.tsv"
            )
            original_state = _read_tsv(state_path, input_hashes)
            state = _current_state_membership(original_state)
            participation = _participation_table(state)
            subspace = _read_tsv(
                world
                / "shared_state_subspace_v4"
                / "typed_subspace_hyperedges.tsv",
                input_hashes,
            )
            required_subspace = {"edge_type", "selection_status"}
            if not required_subspace.issubset(subspace.columns):
                raise ValueError(
                    f"Subspace hyperedge is missing columns for {block.name}/{seed}"
                )
            selected_subspace = subspace.loc[
                subspace["edge_type"].astype(str).eq(
                    "cross_assay_shared_state_subspace"
                )
                & subspace["selection_status"].astype(str).eq("selected")
            ]
            if len(selected_subspace) > 1:
                raise ValueError(
                    f"Multiple selected state hyperedges for {block.name}/{seed}"
                )
            state_frames.append(_tag(state, block.name, seed))
            participation_frames.append(_tag(participation, block.name, seed))
            subspace_frames.append(_tag(subspace, block.name, seed))
            for row in participation.itertuples(index=False):
                if bool(row.selected) != bool(row.strict_feature_present):
                    change_rows.append(
                        {
                            "block": block.name,
                            "seed": seed,
                            "layer": "assay_participation",
                            "modality": row.modality,
                            "change": (
                                "added_relative_to_strict_feature_proxy"
                                if row.selected
                                else "removed_relative_to_strict_feature_proxy"
                            ),
                            "node1": "",
                            "node2": "",
                            "parent": "",
                            "child": "",
                            "details": json.dumps(
                                {
                                    "partial_conjunction_p": row.partial_conjunction_p,
                                    "holm_p": row.holm_p,
                                    "single_feature_p": row.single_feature_p,
                                    "single_feature_witness_view_count": (
                                        row.single_feature_witness_view_count
                                    ),
                                    "omnibus_p": row.omnibus_p,
                                    "omnibus_holm_p": row.omnibus_holm_p,
                                    "winning_route": row.winning_route,
                                    "selection_route": row.selection_route,
                                    "minimum_witness_view_count": (
                                        row.minimum_witness_view_count
                                    ),
                                },
                                sort_keys=True,
                            ),
                        }
                    )

            if block.name == "TRAIN400":
                candidates = _read_tsv(
                    candidate
                    / "crossfit_hierarchical_collider"
                    / "candidate_evidence.tsv",
                    input_hashes,
                )
                old_decisions = _read_tsv(
                    candidate
                    / "crossfit_hierarchical_collider"
                    / "assay_decisions.tsv",
                    input_hashes,
                )
                production_direct = _train400_production(old_decisions)
                prior_direct = _read_tsv(
                    candidate
                    / "crossfit_hierarchical_collider"
                    / "direct_candidates.tsv",
                    input_hashes,
                )
                prior_loading = _read_tsv(
                    candidate
                    / "structure_anchored_loading_groups"
                    / "typed_structure_anchored_groups.tsv",
                    input_hashes,
                )
            else:
                candidates = retired_candidates.loc[
                    retired_candidates["block"].astype(str).eq(block.name)
                    & retired_candidates["seed"].astype(int).eq(seed)
                ].drop(columns=["block", "seed"])
                production_rows = retired_production.loc[
                    retired_production["block"].astype(str).eq(block.name)
                    & retired_production["seed"].astype(int).eq(seed)
                ]
                production_direct = _production_direct_frame(production_rows)
                prior_direct = prior_retired_direct.loc[
                    prior_retired_direct["block"].astype(str).eq(block.name)
                    & prior_retired_direct["seed"].astype(int).eq(seed)
                ].drop(columns=["block", "seed"])
                prior_loading = prior_retired_loading.loc[
                    prior_retired_loading["block"].astype(str).eq(block.name)
                    & prior_retired_loading["seed"].astype(int).eq(seed)
                ].drop(columns=["block", "seed"])

            decisions = select_crossfit_hierarchical_collider_replacements(
                candidates=candidates,
                state_membership=state,
                current_direct_candidates=production_direct,
            )
            unresolved = _read_tsv(
                candidate / "collider_motif" / "demoted_or_unresolved.tsv",
                input_hashes,
                allow_empty=True,
            )
            decisions = _annotate_unique_rescue_rank_overlap_adjacencies(
                decisions=decisions,
                current_direct_candidates=production_direct,
                current_demoted_or_unresolved=unresolved,
            )
            direct = _resolved_direct_frame(decisions, candidate, input_hashes)
            decision_frames.append(_tag(decisions, block.name, seed))
            direct_frames.append(_tag(direct, block.name, seed))
            prior_direct_frames.append(_tag(prior_direct, block.name, seed))
            current_edges = _edge_set(direct)
            prior_edges = _edge_set(prior_direct)
            for assay, parent, child in sorted(current_edges - prior_edges):
                change_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "layer": "within_assay_direct_candidate",
                        "modality": assay,
                        "change": "added_directed_edge",
                        "node1": parent,
                        "node2": child,
                        "parent": parent,
                        "child": child,
                        "details": "",
                    }
                )
            for assay, parent, child in sorted(prior_edges - current_edges):
                change_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "layer": "within_assay_direct_candidate",
                        "modality": assay,
                        "change": "removed_directed_edge",
                        "node1": parent,
                        "node2": child,
                        "parent": parent,
                        "child": child,
                        "details": "",
                    }
                )

            view_dirs = sorted(
                path
                for path in world.glob("cells*_seed*")
                if path.is_dir() and not path.is_symlink()
            )
            if len(view_dirs) != 6:
                raise ValueError(f"Expected six loading views for {block.name}/{seed}")
            views = []
            empirical_frames = []
            for view_dir in view_dirs:
                for name in (
                    "cell_loading_feature_models.tsv",
                    "cell_loading_loadings.tsv",
                    "cell_loading_pair_evidence.tsv",
                ):
                    path = _assert_path(view_dir / name)
                    input_hashes[str(path)] = _sha256(path)
                view = load_evidence_view(view_dir)
                views.append(view)
                empirical = view.pair_evidence.copy()
                empirical["view_id"] = str(view.view_id)
                empirical_frames.append(empirical)
            centers, parent_pairs = _parent_constraints(direct)
            projection = estimate_loading_group_projection(
                views,
                projection_config,
                excluded_features=centers,
                forbidden_within_pairs=parent_pairs,
            )
            anchor = estimate_structure_anchored_groups(
                direct,
                projection.within_group_candidates,
                pd.concat(empirical_frames, ignore_index=True, sort=False),
                state,
                anchor_config,
            )
            loading_frames.append(_tag(anchor.typed_groups, block.name, seed))
            loading_decision_frames.append(
                _tag(anchor.assay_decisions, block.name, seed)
            )
            prior_loading_frames.append(_tag(prior_loading, block.name, seed))
            current_pairs = _selected_loading(anchor.typed_groups)
            prior_pairs = _selected_loading(prior_loading)
            for modality, left, right in sorted(current_pairs - prior_pairs):
                change_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "layer": "within_assay_shared_state_support",
                        "modality": modality,
                        "change": "added_loading_pair",
                        "node1": left,
                        "node2": right,
                        "parent": "",
                        "child": "",
                        "details": "",
                    }
                )
            for modality, left, right in sorted(prior_pairs - current_pairs):
                change_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "layer": "within_assay_shared_state_support",
                        "modality": modality,
                        "change": "removed_loading_pair",
                        "node1": left,
                        "node2": right,
                        "parent": "",
                        "child": "",
                        "details": "",
                    }
                )

    outputs = {
        "phase_a_state_membership.tsv": pd.concat(
            state_frames, ignore_index=True, sort=False
        ),
        "phase_a_assay_participation.tsv": pd.concat(
            participation_frames, ignore_index=True, sort=False
        ),
        "phase_a_subspace_hyperedges.tsv": pd.concat(
            subspace_frames, ignore_index=True, sort=False
        ),
        "phase_a_direct_decisions.tsv": pd.concat(
            decision_frames, ignore_index=True, sort=False
        ),
        "phase_a_direct_edges.tsv": pd.concat(
            direct_frames, ignore_index=True, sort=False
        ),
        "phase_a_prior_direct_edges.tsv": pd.concat(
            prior_direct_frames, ignore_index=True, sort=False
        ),
        "phase_a_loading_groups.tsv": pd.concat(
            loading_frames, ignore_index=True, sort=False
        ),
        "phase_a_loading_decisions.tsv": pd.concat(
            loading_decision_frames, ignore_index=True, sort=False
        ),
        "phase_a_prior_loading_groups.tsv": pd.concat(
            prior_loading_frames, ignore_index=True, sort=False
        ),
        "phase_a_changed_decisions.tsv": pd.DataFrame(change_rows),
    }
    if set(outputs) != set(PHASE_A_OUTPUT_NAMES):
        raise RuntimeError("Phase-A output set differs from the frozen replay contract")
    output_hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, sep="\t", index=False)
        output_hashes[name] = _sha256(path)

    source_paths = {
        Path(__file__),
        SCRIPT_DIR / "audit_retired_combined_replay_v1.py",
        SCRIPT_DIR / "crossfit_hierarchical_collider_v4.py",
        SCRIPT_DIR / "shared_state_subspace_v4.py",
        SCRIPT_DIR / "loading_group_projection_v4.py",
        SCRIPT_DIR / "structure_anchored_loading_groups_v4.py",
        SCRIPT_DIR / "score_layered_state_evidence_v4.py",
    }
    manifest = {
        "schema_version": f"{SCHEMA}.phase_a",
        "phase": "observed_only_before_truth",
        "status": "posthoc_development_only_not_certification",
        "registered_blocks": {
            block.name: [block.start, block.end] for block in BLOCKS
        },
        "n_worlds": len(ALLOWED_SEEDS),
        "truth_or_scorer_opened": False,
        "dev340_locked450_train410_or_new_world_opened": False,
        "all_predictions_frozen_before_truth": True,
        "direct_schema_version": DIRECT_SCHEMA_VERSION,
        "state_participation_config": asdict(SharedStateSubspaceConfig()),
        "loading_projection_config": asdict(projection_config),
        "structure_anchored_loading_config": asdict(anchor_config),
        "input_sha256": dict(sorted(input_hashes.items())),
        "source_sha256": {
            str(path): _sha256(path) for path in sorted(source_paths)
        },
        "output_sha256": dict(sorted(output_hashes.items())),
        "phase_b_must_run_as_separate_process": True,
    }
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _verify_phase_a(out_dir: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    _, manifest_payload = _snapshot_regular_bytes(
        out_dir / "phase_a_manifest.json"
    )
    manifest = json.loads(manifest_payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Phase-A manifest must be a JSON object")
    if manifest.get("schema_version") != f"{SCHEMA}.phase_a":
        raise ValueError("Phase-A schema version is not current")
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Not an observed-only Phase-A freeze")
    if manifest.get("n_worlds") != 120:
        raise ValueError("Phase A does not contain exactly 120 registered worlds")
    expected_blocks = {block.name: [block.start, block.end] for block in BLOCKS}
    if manifest.get("registered_blocks") != expected_blocks:
        raise ValueError("Phase A does not bind the exact opened TRAIN blocks")
    if manifest.get("direct_schema_version") != DIRECT_SCHEMA_VERSION:
        raise ValueError("Phase-A direct schema version is not current")
    if manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Phase A reports truth access")
    if manifest.get("dev340_locked450_train410_or_new_world_opened") is not False:
        raise ValueError("Phase A reports protected split access")
    if manifest.get("all_predictions_frozen_before_truth") is not True:
        raise ValueError("Phase A does not freeze all predictions before truth")
    if manifest.get("phase_b_must_run_as_separate_process") is not True:
        raise ValueError("Phase A does not require a separate Phase-B process")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, Mapping) or set(output_hashes) != set(
        PHASE_A_OUTPUT_NAMES
    ):
        raise ValueError("Phase-A output file set is missing or extra")
    output_payloads: dict[str, bytes] = {}
    for name, expected in output_hashes.items():
        _, payload = _snapshot_regular_bytes(out_dir / str(name))
        if _bytes_sha256(payload) != expected:
            raise ValueError(f"Phase-A output drift: {name}")
        output_payloads[str(name)] = payload
    source_hashes = manifest.get("source_sha256")
    input_hashes = manifest.get("input_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("Phase A lacks its source hash closure")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ValueError("Phase A lacks its observed-input hash closure")
    for text, expected in source_hashes.items():
        path, payload = _snapshot_regular_bytes(Path(str(text)))
        if _bytes_sha256(payload) != expected:
            raise ValueError(f"Source drift after Phase A: {path}")
    for text, expected in input_hashes.items():
        path, payload = _snapshot_regular_bytes(Path(str(text)))
        if _bytes_sha256(payload) != expected:
            raise ValueError(f"Observed input drift after Phase A: {path}")
    return manifest, output_payloads


def _direct_metrics(
    predicted: set[tuple[str, str]], true: set[tuple[str, str]]
) -> dict[str, Any]:
    predicted_skeleton = {frozenset(edge) for edge in predicted}
    true_skeleton = {frozenset(edge) for edge in true}
    tp = predicted_skeleton & true_skeleton
    reverse = {(parent, child) for parent, child in predicted if (child, parent) in true}
    return {
        "tp": len(tp),
        "fp": len(predicted_skeleton - true_skeleton),
        "fn": len(true_skeleton - predicted_skeleton),
        "reversals": len(reverse),
        "precision": _rate(len(tp), len(predicted_skeleton)),
        "recall": _rate(len(tp), len(true_skeleton)),
        "n_predicted": len(predicted_skeleton),
        "n_correctly_oriented": len(predicted & true),
    }


def _loading_metrics(
    predicted: set[frozenset[str]], true: set[frozenset[str]], sentinels: set[str]
) -> dict[str, Any]:
    tp = predicted & true
    members = set().union(*predicted) if predicted else set()
    return {
        "tp": len(tp),
        "fp": len(predicted - true),
        "fn": len(true - predicted),
        "precision": _rate(len(tp), len(predicted)),
        "recall": _rate(len(tp), len(true)),
        "n_predicted": len(predicted),
        "sentinel_members": len(members & sentinels),
    }


def _gate_payload(
    *,
    state_score: Mapping[str, Any],
    participating: set[str],
    direct: Mapping[str, Any],
    loading: Mapping[str, Any],
    density: int,
    state_hyperedge_selected: bool,
) -> dict[str, Any]:
    return {
        "set_scores": {"confirmed_state_response": dict(state_score)},
        "assay_participation_modalities": sorted(participating),
        "assay_participation_coverage": _rate(
            len(participating), EXPECTED_MODALITIES
        ),
        "confirmed_state_recovery_success": bool(state_score["recall"] == 1.0),
        "direct_score": {
            "skeleton_precision": direct["precision"],
            "skeleton_recall": direct["recall"],
            "n_strict_reverse": direct["reversals"],
        },
        "assay_local_loading_pair_score": {
            "pair_precision": loading["precision"],
            "pair_recall": loading["recall"],
            "n_sentinel_members": loading["sentinel_members"],
            "n_non_pair_hyperedges": 0,
            "n_feature_membership_mismatches": 0,
        },
        "cross_assay_state_hyperedge": {
            "n_selected": int(state_hyperedge_selected),
            "members_equal_strict_confirmed_participating_subset": bool(
                state_hyperedge_selected
            ),
            "modalities_equal_assay_participation": bool(
                state_hyperedge_selected
            ),
            "safe_hypothesis_semantics": bool(state_hyperedge_selected),
        },
        "typed_focal_graph": {"n_expanded_focal_relations": density},
        "semantics": {
            "layer_union_is_state_membership": False,
            "assay_participation_implies_feature_membership": False,
            "cross_assay_pair_claimed": False,
            "direct_cross_assay_effect_claimed": False,
        },
    }


def _world_subset(frame: pd.DataFrame, block: str, seed: int) -> pd.DataFrame:
    return frame.loc[
        frame["block"].astype(str).eq(block)
        & frame["seed"].astype(int).eq(seed)
    ].copy()


def _phase_b(out_dir: Path) -> None:
    phase_a, phase_a_payloads = _verify_phase_a(out_dir)
    state = _parse_tsv_bytes(phase_a_payloads["phase_a_state_membership.tsv"])
    participation = _parse_tsv_bytes(
        phase_a_payloads["phase_a_assay_participation.tsv"]
    )
    subspace = _parse_tsv_bytes(
        phase_a_payloads["phase_a_subspace_hyperedges.tsv"]
    )
    direct = _parse_tsv_bytes(phase_a_payloads["phase_a_direct_edges.tsv"])
    prior_direct = _parse_tsv_bytes(
        phase_a_payloads["phase_a_prior_direct_edges.tsv"]
    )
    loading = _parse_tsv_bytes(phase_a_payloads["phase_a_loading_groups.tsv"])
    prior_loading = _parse_tsv_bytes(
        phase_a_payloads["phase_a_prior_loading_groups.tsv"]
    )
    changes = _parse_tsv_bytes(
        phase_a_payloads["phase_a_changed_decisions.tsv"]
    )

    world_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    labeled_change_rows: list[dict[str, Any]] = []
    truth_hashes: dict[str, str] = {}

    for block in BLOCKS:
        for seed in block.seeds:
            scorer = block.world_root / f"world_seed{seed}" / "scorer"
            roles_path, roles_payload = _snapshot_regular_bytes(
                scorer / "role_assignments.tsv", allow_truth=True
            )
            truth_path, truth_payload = _snapshot_regular_bytes(
                scorer / "truth_typed_edges.tsv", allow_truth=True
            )
            truth_hashes[str(roles_path)] = _bytes_sha256(roles_payload)
            truth_hashes[str(truth_path)] = _bytes_sha256(truth_payload)
            roles = _parse_tsv_bytes(roles_payload)
            truth = _parse_tsv_bytes(truth_payload)
            role_map = dict(zip(roles["node"].astype(str), roles["role"].astype(str)))

            state_world = _world_subset(state, block.name, seed)
            selected_state = set(
                state_world.loc[
                    state_world["selection_status"].astype(str).eq("selected"),
                    "feature",
                ].astype(str)
            )
            genes = set(
                state_world.loc[
                    state_world["modality"].astype(str).eq("rna"), "feature"
                ].astype(str)
            )
            if len(genes) != 1:
                raise ValueError(f"Expected one RNA feature for {block.name}/{seed}")
            gene = next(iter(genes))
            positives = {
                feature for feature, role in role_map.items() if role in STATE_ROLES
            } | {gene}
            sentinels = {
                feature for feature, role in role_map.items() if role in NULL_ROLES
            }
            state_tp = selected_state & positives
            state_fp = selected_state - positives
            state_score = {
                "n_selected": len(selected_state),
                "n_true_selected": len(state_tp),
                "n_sentinel_selected": len(selected_state & sentinels),
                "n_unknown_selected": len(state_fp - sentinels),
                "precision": _rate(len(state_tp), len(selected_state)),
                "recall": _rate(len(state_tp), len(positives)),
            }
            participation_world = _world_subset(participation, block.name, seed)
            participating = set(
                participation_world.loc[
                    participation_world["selected"].map(_strict_bool), "modality"
                ].astype(str)
            )
            strict_modalities = set(
                state_world.loc[
                    state_world["feature"].astype(str).isin(selected_state), "modality"
                ].astype(str)
            )
            for row in state_world.itertuples(index=False):
                feature = str(row.feature)
                state_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "feature": feature,
                        "modality": str(row.modality),
                        "role": role_map.get(feature, "GENE" if feature == gene else ""),
                        "strict_selected": feature in selected_state,
                        "assay_participation_selected": str(row.modality)
                        in participating,
                        "assay_participation_reporter": _strict_bool(
                            row.assay_participation_reporter
                        ),
                        "truth_state_member": feature in positives,
                        "truth_sentinel": feature in sentinels,
                    }
                )

            true_direct = {
                (str(row.parent), str(row.child))
                for row in truth.loc[
                    truth["truth_type"].astype(str).eq("within_assay_direct")
                ].itertuples(index=False)
            }
            direct_world = _world_subset(direct, block.name, seed)
            prior_direct_world = _world_subset(prior_direct, block.name, seed)
            predicted_direct = {
                (parent, child)
                for _, parent, child in _edge_set(direct_world)
            }
            prior_predicted_direct = {
                (parent, child)
                for _, parent, child in _edge_set(prior_direct_world)
            }
            direct_score = _direct_metrics(predicted_direct, true_direct)
            prior_direct_score = _direct_metrics(prior_predicted_direct, true_direct)
            all_direct_pairs = sorted(predicted_direct | true_direct)
            for parent, child in all_direct_pairs:
                skeleton = frozenset((parent, child))
                direct_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "parent": parent,
                        "child": child,
                        "predicted_oriented": (parent, child) in predicted_direct,
                        "truth_oriented": (parent, child) in true_direct,
                        "predicted_skeleton": skeleton
                        in {frozenset(edge) for edge in predicted_direct},
                        "truth_skeleton": skeleton
                        in {frozenset(edge) for edge in true_direct},
                        "strict_reverse": (child, parent) in true_direct
                        and (parent, child) in predicted_direct,
                    }
                )

            true_loading = {
                frozenset((str(row.node1), str(row.node2)))
                for row in truth.loc[
                    truth["truth_type"].astype(str).eq("within_assay_shared_state")
                ].itertuples(index=False)
            }
            loading_world = _world_subset(loading, block.name, seed)
            prior_loading_world = _world_subset(prior_loading, block.name, seed)
            predicted_loading = {
                frozenset((left, right))
                for _, left, right in _selected_loading(loading_world)
            }
            prior_predicted_loading = {
                frozenset((left, right))
                for _, left, right in _selected_loading(prior_loading_world)
            }
            loading_score = _loading_metrics(
                predicted_loading, true_loading, sentinels
            )
            prior_loading_score = _loading_metrics(
                prior_predicted_loading, true_loading, sentinels
            )
            for pair in sorted(
                predicted_loading | true_loading, key=lambda value: tuple(sorted(value))
            ):
                left, right = sorted(pair)
                loading_rows.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "node1": left,
                        "node2": right,
                        "predicted": pair in predicted_loading,
                        "truth": pair in true_loading,
                        "classification": (
                            "true_positive"
                            if pair in predicted_loading & true_loading
                            else "false_positive"
                            if pair in predicted_loading
                            else "false_negative"
                        ),
                    }
                )

            subspace_world = _world_subset(subspace, block.name, seed)
            selected_upstream_subspace = subspace_world.loc[
                subspace_world["edge_type"].astype(str).eq(
                    "cross_assay_shared_state_subspace"
                )
                & subspace_world["selection_status"].astype(str).eq("selected")
            ]
            if len(selected_upstream_subspace) > 1:
                raise ValueError(
                    f"Multiple selected upstream hyperedges for {block.name}/{seed}"
                )
            upstream_subspace_selected = len(selected_upstream_subspace) == 1
            state_hyperedge_selected = bool(
                upstream_subspace_selected and len(participating) >= 2
            )
            prior_state_hyperedge_selected = bool(
                upstream_subspace_selected and len(strict_modalities) >= 2
            )
            density = (
                direct_score["n_predicted"]
                + loading_score["n_predicted"]
                + int(state_hyperedge_selected)
            )
            prior_density = (
                prior_direct_score["n_predicted"]
                + prior_loading_score["n_predicted"]
                + int(prior_state_hyperedge_selected)
            )
            current_gate = evaluate_layered_world_reliability(
                _gate_payload(
                    state_score=state_score,
                    participating=participating,
                    direct=direct_score,
                    loading=loading_score,
                    density=density,
                    state_hyperedge_selected=state_hyperedge_selected,
                )
            )
            prior_gate = evaluate_layered_world_reliability(
                _gate_payload(
                    state_score=state_score,
                    participating=strict_modalities,
                    direct=prior_direct_score,
                    loading=prior_loading_score,
                    density=prior_density,
                    state_hyperedge_selected=prior_state_hyperedge_selected,
                )
            )
            failures = sorted(
                key for key, value in current_gate["gate_checks"].items() if not value
            )
            prior_failures = sorted(
                key for key, value in prior_gate["gate_checks"].items() if not value
            )
            world_rows.append(
                {
                    "block": block.name,
                    "seed": seed,
                    "strict_state_selected": len(selected_state),
                    "strict_state_tp": len(state_tp),
                    "strict_state_fp": len(state_fp),
                    "strict_state_precision": state_score["precision"],
                    "strict_state_recall": state_score["recall"],
                    "strict_state_modality_count": len(strict_modalities),
                    "assay_participation_count": len(participating),
                    "assay_participation_full_7": len(participating) == 7,
                    "upstream_subspace_selected": upstream_subspace_selected,
                    "state_hyperedge_selected": state_hyperedge_selected,
                    "direct_tp": direct_score["tp"],
                    "direct_fp": direct_score["fp"],
                    "direct_fn": direct_score["fn"],
                    "direct_reversals": direct_score["reversals"],
                    "direct_precision": direct_score["precision"],
                    "direct_recall": direct_score["recall"],
                    "loading_tp": loading_score["tp"],
                    "loading_fp": loading_score["fp"],
                    "loading_fn": loading_score["fn"],
                    "loading_precision": loading_score["precision"],
                    "loading_recall": loading_score["recall"],
                    "loading_sentinel_members": loading_score["sentinel_members"],
                    "expanded_focal_density": density,
                    "prior_expanded_focal_density": prior_density,
                    "prior_world_reliability_success": bool(
                        prior_gate["world_reliability_success"]
                    ),
                    "world_reliability_success": bool(
                        current_gate["world_reliability_success"]
                    ),
                    "reliability_changed": bool(
                        prior_gate["world_reliability_success"]
                        != current_gate["world_reliability_success"]
                    ),
                    "failed_gate_checks": ";".join(failures),
                    "prior_failed_gate_checks": ";".join(prior_failures),
                }
            )

            world_changes = _world_subset(changes, block.name, seed)
            for record in world_changes.to_dict(orient="records"):
                layer = str(record["layer"])
                change = str(record["change"])
                truth_classification = "not_applicable"
                if layer == "within_assay_direct_candidate":
                    oriented = (str(record["parent"]), str(record["child"]))
                    skeleton = frozenset(oriented)
                    truth_skeleton = {frozenset(edge) for edge in true_direct}
                    if change.startswith("added"):
                        truth_classification = (
                            "added_true_oriented_edge"
                            if oriented in true_direct
                            else "added_reverse_edge"
                            if (oriented[1], oriented[0]) in true_direct
                            else "added_false_edge"
                        )
                    else:
                        truth_classification = (
                            "removed_true_oriented_edge"
                            if oriented in true_direct
                            else "removed_reverse_edge"
                            if (oriented[1], oriented[0]) in true_direct
                            else "removed_false_edge"
                        )
                    if skeleton not in truth_skeleton and "false" not in truth_classification:
                        raise ValueError("Direct change truth classification is inconsistent")
                elif layer == "within_assay_shared_state_support":
                    pair = frozenset((str(record["node1"]), str(record["node2"])))
                    truth_classification = (
                        ("added_true_loading_pair" if pair in true_loading else "added_false_loading_pair")
                        if change.startswith("added")
                        else ("removed_true_loading_pair" if pair in true_loading else "removed_false_loading_pair")
                    )
                elif layer == "assay_participation":
                    truth_classification = (
                        "true_assay_participation_added"
                        if change.startswith("added")
                        else "true_assay_participation_removed"
                    )
                labeled_change_rows.append(
                    {**record, "truth_classification": truth_classification}
                )

    worlds = pd.DataFrame(world_rows).sort_values(["block", "seed"])
    block_rows = []
    for block_name, group in worlds.groupby("block", sort=True):
        direct_tp = int(group["direct_tp"].sum())
        direct_fp = int(group["direct_fp"].sum())
        direct_fn = int(group["direct_fn"].sum())
        loading_tp = int(group["loading_tp"].sum())
        loading_fp = int(group["loading_fp"].sum())
        loading_fn = int(group["loading_fn"].sum())
        reliable = int(group["world_reliability_success"].sum())
        mean_direct_recall = float(group["direct_recall"].mean())
        mean_loading_precision = float(group["loading_precision"].mean())
        mean_loading_recall = float(group["loading_recall"].mean())
        mean_density = float(group["expanded_focal_density"].mean())
        gates = {
            "world_count_gate": reliable >= MIN_RELIABLE_WORLDS,
            "direct_mean_gate": mean_direct_recall >= MIN_BLOCK_MEAN_DIRECT_RECALL,
            "loading_precision_gate": mean_loading_precision
            >= MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION,
            "loading_recall_gate": mean_loading_recall
            >= MIN_BLOCK_MEAN_LOADING_PAIR_RECALL,
            "density_gate": mean_density <= MAX_BLOCK_MEAN_FOCAL_DENSITY,
        }
        block_rows.append(
            {
                "block": block_name,
                "n_worlds": len(group),
                "full_assay_participation_worlds": int(
                    group["assay_participation_full_7"].sum()
                ),
                "prior_reliable_worlds": int(
                    group["prior_world_reliability_success"].sum()
                ),
                "reliable_worlds": reliable,
                "meets_22_of_24": gates["world_count_gate"],
                "block_mean_direct_recall_gate": gates["direct_mean_gate"],
                "block_mean_loading_precision_gate": gates[
                    "loading_precision_gate"
                ],
                "block_mean_loading_recall_gate": gates["loading_recall_gate"],
                "block_mean_density_gate": gates["density_gate"],
                "all_block_gates_pass": all(gates.values()),
                "direct_tp": direct_tp,
                "direct_fp": direct_fp,
                "direct_fn": direct_fn,
                "direct_reversals": int(group["direct_reversals"].sum()),
                "direct_micro_precision": _rate(direct_tp, direct_tp + direct_fp),
                "direct_micro_recall": _rate(direct_tp, direct_tp + direct_fn),
                "mean_direct_recall": mean_direct_recall,
                "loading_tp": loading_tp,
                "loading_fp": loading_fp,
                "loading_fn": loading_fn,
                "loading_micro_precision": _rate(
                    loading_tp, loading_tp + loading_fp
                ),
                "loading_micro_recall": _rate(
                    loading_tp, loading_tp + loading_fn
                ),
                "mean_loading_precision": mean_loading_precision,
                "mean_loading_recall": mean_loading_recall,
                "mean_expanded_focal_density": mean_density,
            }
        )
    blocks = pd.DataFrame(block_rows)
    outputs = {
        "phase_b_world_metrics.tsv": worlds,
        "phase_b_block_summary.tsv": blocks,
        "phase_b_state_feature_outcomes.tsv": pd.DataFrame(state_rows),
        "phase_b_direct_edge_outcomes.tsv": pd.DataFrame(direct_rows),
        "phase_b_loading_pair_outcomes.tsv": pd.DataFrame(loading_rows),
        "phase_b_changed_decisions_labeled.tsv": pd.DataFrame(
            labeled_change_rows
        ),
    }
    output_hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, sep="\t", index=False)
        output_hashes[name] = _sha256(path)

    failures = worlds.loc[~worlds["world_reliability_success"]]
    summary = {
        "schema_version": f"{SCHEMA}.summary",
        "status": "posthoc_development_only_not_certification",
        "truth_opened_only_after_phase_a_hash_verification": True,
        "dev340_locked450_train410_or_new_world_opened": False,
        "n_worlds": len(worlds),
        "n_reliable_worlds": int(worlds["world_reliability_success"].sum()),
        "unresolved_failures": failures[
            ["block", "seed", "failed_gate_checks"]
        ].to_dict(orient="records"),
        "block_summary": blocks.to_dict(orient="records"),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "phase_a_source_sha256": phase_a["source_sha256"],
        "truth_input_sha256": dict(sorted(truth_hashes.items())),
        "output_sha256": dict(sorted(output_hashes.items())),
    }
    _write_json(out_dir / "summary.json", summary)
    output_hashes["summary.json"] = _sha256(out_dir / "summary.json")
    _write_json(
        out_dir / "phase_b_manifest.json",
        {
            "schema_version": f"{SCHEMA}.phase_b",
            "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
            "truth_opened_only_after_phase_a_hash_verification": True,
            "dev340_locked450_train410_or_new_world_opened": False,
            "truth_input_sha256": dict(sorted(truth_hashes.items())),
            "output_sha256": dict(sorted(output_hashes.items())),
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.phase == "a":
        _phase_a(args.out_dir)
    else:
        _phase_b(args.out_dir)


if __name__ == "__main__":
    main()
