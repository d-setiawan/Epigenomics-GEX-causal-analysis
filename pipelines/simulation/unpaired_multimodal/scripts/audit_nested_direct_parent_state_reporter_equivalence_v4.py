#!/usr/bin/env python3
"""Two-phase cached audit of CA-BOSS v4.33 partial identification.

Phase A binds the already frozen v4.22 direct-collider candidates, the v4.27
truth-free observable feature table, a fixed v4.33 rule, and balanced random
subsets of the existing cached worlds.  It never opens simulator truth or the
v4.27 outer prediction table.  Phase B verifies every binding first, opens the
existing leave-one-source-block-out v4.27 predictions, applies v4.33 to a
truth-free projection, and only then opens truth for scoring.

One selected row is one set-valued hypothesis.  Alternative pairs inside that
hypothesis are never expanded into edges.  Exact cross-assay pair and direction
metrics are reported only as identifiability stress tests and are not success
criteria.  The script has no world-generation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_nested_loading_group_axis_consensus_v4 as v429_audit  # noqa: E402
import audit_nested_topology_consensus_state_v4 as v427_audit  # noqa: E402
import cross_assay_benchmark_metrics_v4 as benchmark_metrics  # noqa: E402
import direct_parent_state_reporter_equivalence_v4 as equivalence  # noqa: E402


SCHEMA = "ca_boss_v4.nested_direct_parent_state_reporter_equivalence.v1"
WORLD_KEY = ["block", "seed"]
ASSAY_KEY = ["block", "seed", "modality"]
PHASE_A_DIRECT = "phase_a_direct_candidates.tsv"
PHASE_A_STATE = "phase_a_observable_state_evidence.tsv"
PHASE_A_PANEL = "phase_a_random_panels.tsv"
PHASE_A_TABLES = (PHASE_A_DIRECT, PHASE_A_STATE, PHASE_A_PANEL)
PHASE_B_HYPOTHESES = "phase_b_outer_equivalence_hypotheses.tsv"
PHASE_B_WORLD = "phase_b_world_metrics.tsv"
PHASE_B_BLOCK = "phase_b_source_block_metrics.tsv"
PHASE_B_PANEL = "phase_b_random_panel_metrics.tsv"
PHASE_B_MODALITY = "phase_b_modality_metrics.tsv"
PHASE_B_TABLES = (
    PHASE_B_HYPOTHESES,
    PHASE_B_WORLD,
    PHASE_B_BLOCK,
    PHASE_B_PANEL,
    PHASE_B_MODALITY,
)
SOURCE_FILES = (
    "audit_nested_direct_parent_state_reporter_equivalence_v4.py",
    "cross_assay_benchmark_metrics_v4.py",
    "direct_parent_state_reporter_equivalence_v4.py",
)
DEFAULT_DIRECT_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_layered_contract_v422_full_pool_v2"
)
DEFAULT_V427_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_topology_consensus_v427_full_pool"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    observed = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} payload hash mismatch")
    return observed


def _safe_file(path: Path, *, phase_a: bool) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if phase_a and {"scorer", "truth", "ground_truth", "oracle"} & {
        part.casefold() for part in resolved.parts
    }:
        raise ValueError(f"Phase A refuses scorer/truth path: {resolved}")
    return resolved


def _load_json(path: Path, *, phase_a: bool) -> tuple[Path, dict[str, Any]]:
    safe = _safe_file(path, phase_a=phase_a)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {safe}")
    return safe, value


def _balanced_panels(
    worlds: pd.DataFrame,
    *,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> pd.DataFrame:
    unique = worlds.loc[:, WORLD_KEY].drop_duplicates().copy()
    blocks = sorted(set(unique["block"].astype(str)))
    if not blocks or panel_size < len(blocks):
        raise ValueError("panel-size must include every source block")
    minimum, remainder = divmod(panel_size, len(blocks))
    rows: list[dict[str, Any]] = []
    for panel_index in range(panels):
        seed = int(selection_seed + panel_index)
        rng = np.random.default_rng(seed)
        extra = set(rng.choice(blocks, size=remainder, replace=False).tolist())
        selected: list[tuple[str, int]] = []
        for block in blocks:
            candidates = (
                unique.loc[unique["block"].astype(str).eq(block), "seed"]
                .astype(int)
                .sort_values()
                .to_numpy()
            )
            count = minimum + int(block in extra)
            if len(candidates) < count:
                raise ValueError(f"Source block is too small: {block}")
            selected.extend(
                (block, int(value))
                for value in rng.choice(candidates, size=count, replace=False)
            )
        if len(selected) != panel_size or len(set(selected)) != panel_size:
            raise RuntimeError("Balanced panel selection was not exact")
        for block, world_seed in sorted(selected):
            rows.append(
                {
                    "panel_id": f"panel_{panel_index + 1:02d}",
                    "selection_seed": seed,
                    "block": block,
                    "seed": world_seed,
                }
            )
    return pd.DataFrame(rows)


def _validate_direct_table(direct: pd.DataFrame) -> pd.DataFrame:
    required = {*WORLD_KEY, *equivalence.DIRECT_COLUMNS}
    missing = sorted(required.difference(direct.columns))
    if missing:
        raise ValueError(f"Direct table is missing columns: {missing}")
    output = direct.loc[:, [*WORLD_KEY, "source_assay", "parent", "child"]].copy()
    output["seed"] = pd.to_numeric(output["seed"], errors="raise").astype(int)
    if output.duplicated([*WORLD_KEY, "source_assay", "parent", "child"]).any():
        raise ValueError("Direct table contains duplicate arms")
    for key, arms in output.groupby([*WORLD_KEY, "source_assay"], sort=True):
        if (
            len(arms) != 2
            or arms["parent"].astype(str).nunique() != 2
            or arms["child"].astype(str).nunique() != 1
        ):
            raise ValueError(f"Direct table does not encode a two-parent collider: {key}")
    return output


def run_phase_a(
    *,
    out_dir: Path,
    direct_dir: Path,
    v427_dir: Path,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-A output: {out_dir}")

    direct_root = direct_dir.expanduser().absolute().resolve(strict=True)
    direct_manifest_path, direct_manifest = _load_json(
        direct_root / "phase_a_manifest.json", phase_a=True
    )
    direct_payload_hash = _validate_self_hash(
        direct_manifest,
        field="manifest_payload_sha256",
        label="v4.22 Phase-A manifest",
    )
    direct_path = _safe_file(
        direct_root / "phase_a_direct_predictions.tsv", phase_a=True
    )
    if (
        direct_manifest.get("schema_version")
        != "ca_boss_v4.cached_layered_contract_audit.v1"
        or direct_manifest.get("phase") != "a"
        or direct_manifest.get("scorer_or_truth_opened") is not False
        or direct_manifest.get("new_worlds_or_datasets_created") is not False
        or direct_manifest.get("output_sha256", {}).get(direct_path.name)
        != _sha256(direct_path)
    ):
        raise ValueError("The v4.22 direct cache is not a valid truth-closed input")
    direct = _validate_direct_table(pd.read_csv(direct_path, sep="\t"))

    v427_root = v427_dir.expanduser().absolute().resolve(strict=True)
    v427_manifest, observable, _ = v427_audit._load_phase_a(v427_root)
    v427_manifest_path = _safe_file(
        v427_root / "phase_a_manifest.json", phase_a=True
    )
    observable = observable.copy()
    observable["seed"] = pd.to_numeric(
        observable["seed"], errors="raise"
    ).astype(int)
    direct_worlds = set(map(tuple, direct.loc[:, WORLD_KEY].drop_duplicates().to_numpy()))
    state_worlds = set(
        map(tuple, observable.loc[:, WORLD_KEY].drop_duplicates().to_numpy())
    )
    if direct_worlds != state_worlds:
        raise ValueError("v4.22 direct and v4.27 observable world pools differ")

    # Exercise the public application boundary using only a false placeholder
    # membership vector.  Phase B supplies the frozen outer state decisions.
    validation_state = observable.loc[:, ["feature", "modality"]].copy()
    validation_state["state_v427_supported_member"] = False
    assays_per_world: list[int] = []
    for block, seed in sorted(direct_worlds):
        direct_world = direct.loc[
            direct["block"].astype(str).eq(str(block))
            & direct["seed"].astype(int).eq(int(seed)),
            ["source_assay", "parent", "child"],
        ]
        state_world = validation_state.loc[
            observable["block"].astype(str).eq(str(block))
            & observable["seed"].astype(int).eq(int(seed))
        ]
        decisions = equivalence.build_direct_parent_state_reporter_equivalence(
            direct_world, state_world
        ).decisions
        assays_per_world.append(len(decisions))
    if len(set(assays_per_world)) != 1 or assays_per_world[0] <= 1:
        raise ValueError("Direct/state inputs do not expose a stable assay count")

    panel_table = _balanced_panels(
        direct.loc[:, WORLD_KEY],
        selection_seed=selection_seed,
        panels=panels,
        panel_size=panel_size,
    )
    out_dir.mkdir(parents=True)
    direct.to_csv(out_dir / PHASE_A_DIRECT, sep="\t", index=False)
    observable.to_csv(out_dir / PHASE_A_STATE, sep="\t", index=False)
    panel_table.to_csv(out_dir / PHASE_A_PANEL, sep="\t", index=False)

    artifact = {
        "schema_version": equivalence.SCHEMA,
        "algorithm": equivalence.ALGORITHM,
        "fixed_rule": equivalence.FIXED_RULE,
        "artifact_role": "future_application_rule_binding",
        "application_requires_truth": False,
        "numeric_threshold_fitted": False,
        "direct_candidates_changed": False,
        "state_v427_membership_changed": False,
        "candidate_pair_expansion_permitted": False,
        "exact_parent_reporter_pair_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "new_simulation_worlds_or_datasets_created": False,
        "source_sha256": _source_hashes(),
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    artifact_path = out_dir / "phase_a_fixed_equivalence_rule.json"
    _write_json(artifact_path, artifact)

    manifest = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": equivalence.SCHEMA,
        "phase": "a",
        "cached_inputs_only": True,
        "all_observable_inputs_loaded_before_truth": True,
        "upstream_v427_phase_b_opened": False,
        "scorer_or_truth_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "fresh_locked_claim": False,
        "retrospective_development_only": True,
        "worlds": len(direct_worlds),
        "source_blocks": int(direct["block"].nunique()),
        "assays_per_world": assays_per_world[0],
        "random_panels": panels,
        "panel_size": panel_size,
        "selection_seed": selection_seed,
        "v422_direct_manifest": str(direct_manifest_path),
        "v422_direct_manifest_sha256": _sha256(direct_manifest_path),
        "v422_direct_manifest_payload_sha256": direct_payload_hash,
        "v422_direct_input": str(direct_path),
        "v422_direct_input_sha256": _sha256(direct_path),
        "v427_phase_a_manifest": str(v427_manifest_path),
        "v427_phase_a_manifest_sha256": _sha256(v427_manifest_path),
        "v427_phase_a_manifest_payload_sha256": v427_manifest[
            "manifest_payload_sha256"
        ],
        "fixed_equivalence_artifact_sha256": _sha256(artifact_path),
        "source_sha256": _source_hashes(),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_A_TABLES
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _, manifest = _load_json(out_dir / "phase_a_manifest.json", phase_a=True)
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="v4.33 Phase-A manifest"
    )
    required = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": equivalence.SCHEMA,
        "phase": "a",
        "all_observable_inputs_loaded_before_truth": True,
        "upstream_v427_phase_b_opened": False,
        "scorer_or_truth_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    bad = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if bad:
        raise ValueError(f"Unsupported or unsafe v4.33 Phase-A manifest: {bad}")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("v4.33 sources changed after Phase A")

    artifact_path = _safe_file(
        out_dir / "phase_a_fixed_equivalence_rule.json", phase_a=True
    )
    if manifest.get("fixed_equivalence_artifact_sha256") != _sha256(
        artifact_path
    ):
        raise ValueError("Fixed v4.33 rule artifact changed")
    _, artifact = _load_json(artifact_path, phase_a=True)
    _validate_self_hash(
        artifact, field="artifact_payload_sha256", label="fixed v4.33 rule"
    )
    if (
        artifact.get("schema_version") != equivalence.SCHEMA
        or artifact.get("fixed_rule") != equivalence.FIXED_RULE
        or artifact.get("numeric_threshold_fitted") is not False
        or artifact.get("application_requires_truth") is not False
        or artifact.get("candidate_pair_expansion_permitted") is not False
    ):
        raise ValueError("Fixed v4.33 rule binding changed")

    for field in ("v422_direct_manifest", "v422_direct_input"):
        source = _safe_file(Path(str(manifest[field])), phase_a=True)
        expected = manifest[f"{field}_sha256"]
        if _sha256(source) != expected:
            raise ValueError(f"Bound input changed after Phase A: {field}")
    direct_manifest = json.loads(
        Path(str(manifest["v422_direct_manifest"])).read_text(encoding="utf-8")
    )
    if (
        _validate_self_hash(
            direct_manifest,
            field="manifest_payload_sha256",
            label="bound v4.22 manifest",
        )
        != manifest["v422_direct_manifest_payload_sha256"]
    ):
        raise ValueError("Bound v4.22 manifest payload changed")

    v427_manifest_path = _safe_file(
        Path(str(manifest["v427_phase_a_manifest"])), phase_a=True
    )
    if _sha256(v427_manifest_path) != manifest["v427_phase_a_manifest_sha256"]:
        raise ValueError("Bound v4.27 Phase-A manifest changed")
    v427_root = v427_manifest_path.parent
    v427_manifest, _, _ = v427_audit._load_phase_a(v427_root)
    if (
        v427_manifest.get("manifest_payload_sha256")
        != manifest["v427_phase_a_manifest_payload_sha256"]
    ):
        raise ValueError("Bound v4.27 Phase-A payload changed")

    tables: dict[str, pd.DataFrame] = {}
    for name in PHASE_A_TABLES:
        path = _safe_file(out_dir / name, phase_a=True)
        if _sha256(path) != manifest.get("output_sha256", {}).get(name):
            raise ValueError(f"Phase-A table changed after freeze: {name}")
        tables[name] = pd.read_csv(path, sep="\t", keep_default_na=False)
    tables[PHASE_A_DIRECT] = _validate_direct_table(tables[PHASE_A_DIRECT])
    return manifest, tables


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def maximum_truth_matching(
    candidates: Sequence[set[frozenset[str]]],
    truth_pairs: Iterable[frozenset[str]],
) -> dict[int, frozenset[str]]:
    """Return a deterministic maximum one-to-one hypothesis/truth matching."""

    truths = sorted(set(truth_pairs), key=lambda pair: tuple(sorted(pair)))
    adjacency = [
        [index for index, truth in enumerate(truths) if truth in candidate]
        for candidate in candidates
    ]
    truth_owner: dict[int, int] = {}

    def augment(hypothesis: int, seen: set[int]) -> bool:
        for truth_index in adjacency[hypothesis]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            owner = truth_owner.get(truth_index)
            if owner is None or augment(owner, seen):
                truth_owner[truth_index] = hypothesis
                return True
        return False

    for hypothesis in range(len(candidates)):
        augment(hypothesis, set())
    return {
        hypothesis: truths[truth_index]
        for truth_index, hypothesis in truth_owner.items()
    }


def _truth_tables(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().absolute().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(path)
    truth = pd.read_csv(resolved, sep="\t", keep_default_na=False)
    required = {"node1", "node2", "truth_type", "orientation_credit"}
    if missing := sorted(required.difference(truth.columns)):
        raise ValueError(f"Truth table is missing columns: {missing}")
    local = {
        _pair(row.node1, row.node2)
        for row in truth.loc[
            truth["truth_type"].astype(str).eq("within_assay_shared_state")
        ].itertuples(index=False)
    }
    cross = truth.loc[
        truth["truth_type"].astype(str).eq("cross_assay_shared_state")
    ].copy()
    cross_pairs = {_pair(row.node1, row.node2) for row in cross.itertuples(index=False)}
    directed_cross = int(
        pd.Series(cross["orientation_credit"]).astype(str).str.casefold().isin(
            {"true", "1"}
        ).sum()
    )
    return {
        "within_assay_shared_state_pairs": local,
        "cross_assay_exact_pairs": cross_pairs,
        "cross_assay_direction_identifiable_truths": directed_cross,
    }, _sha256(resolved)


def _coverage_metrics(frame: pd.DataFrame, *, total_truth: int) -> dict[str, Any]:
    return benchmark_metrics.coverage_metrics(frame, total_truth=total_truth)


def _exact_cross_assay_stress(
    *, cross_truth_pairs: int, direction_truths: int
) -> dict[str, Any]:
    return benchmark_metrics.exact_cross_assay_stress(
        cross_truth_pairs=cross_truth_pairs,
        direction_truths=direction_truths,
        interpretation=(
            "v4.33_recovers_an_assay_local_partial_identification_set_and_"
            "does_not_claim_exact_cross_assay_feature_identity_or_direction"
        ),
    )


def run_phase_b(
    *, out_dir: Path, worlds_root: Path, v427_dir: Path
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")

    manifest, tables = _load_phase_a(out_dir)
    direct = tables[PHASE_A_DIRECT]
    observable = tables[PHASE_A_STATE]
    panels = tables[PHASE_A_PANEL]
    features, upstream_provenance = v429_audit._verified_v427_predictions(
        v427_dir
    )
    feature_keys = features.loc[:, [*WORLD_KEY, "feature", "modality"]].copy()
    observable_keys = observable.loc[
        :, [*WORLD_KEY, "feature", "modality"]
    ].copy()
    for frame in (feature_keys, observable_keys):
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if not feature_keys.sort_values([*WORLD_KEY, "feature"]).reset_index(
        drop=True
    ).equals(
        observable_keys.sort_values([*WORLD_KEY, "feature"]).reset_index(drop=True)
    ):
        raise ValueError("v4.27 outer predictions do not match bound Phase-A keys")

    prediction_frames: list[pd.DataFrame] = []
    local_truth_totals: dict[tuple[str, int], int] = {}
    cross_truth_totals: dict[tuple[str, int], int] = {}
    direction_truth_totals: dict[tuple[str, int], int] = {}
    truth_hashes: dict[str, str] = {}
    worlds = direct.loc[:, WORLD_KEY].drop_duplicates().sort_values(WORLD_KEY)
    for world in worlds.itertuples(index=False):
        block, seed = str(world.block), int(world.seed)

        def mask(frame: pd.DataFrame) -> pd.Series:
            return frame["block"].astype(str).eq(block) & pd.to_numeric(
                frame["seed"], errors="raise"
            ).eq(seed)

        direct_world = direct.loc[
            mask(direct), ["source_assay", "parent", "child"]
        ].copy()
        feature_world = features.loc[mask(features)].copy()
        state_world = feature_world.loc[
            :, ["feature", "modality", "state_v427_supported_member"]
        ].copy()
        result = equivalence.build_direct_parent_state_reporter_equivalence(
            direct_world, state_world
        ).decisions

        truth_path = (
            worlds_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        truth, truth_hash = _truth_tables(truth_path)
        local_pairs = truth["within_assay_shared_state_pairs"]
        cross_pairs = truth["cross_assay_exact_pairs"]
        local_truth_totals[(block, seed)] = len(local_pairs)
        cross_truth_totals[(block, seed)] = len(cross_pairs)
        direction_truth_totals[(block, seed)] = int(
            truth["cross_assay_direction_identifiable_truths"]
        )
        truth_hashes[f"{block}:{seed}"] = truth_hash

        selected_indices = result.index[
            result["hypothesis_selected"].astype(bool)
        ].tolist()
        candidate_sets = [
            {
                _pair(left, right)
                for left, right in equivalence.parse_candidate_pairs(
                    result.loc[index, "candidate_pairs_json"]
                )
            }
            for index in selected_indices
        ]
        local_matching = maximum_truth_matching(candidate_sets, local_pairs)
        matched_by_index = {
            selected_indices[position]: pair
            for position, pair in local_matching.items()
        }
        roles = feature_world.set_index("feature")["truth_role"].astype(str)
        sentinel_opportunities_by_modality = (
            feature_world.loc[
                feature_world["truth_role"].astype(str).isin({"N1", "N2"})
            ]
            .groupby("modality", sort=False)["feature"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        candidate_truth_counts: list[int] = []
        sentinel_flags: list[bool] = []
        sentinel_alternative_counts: list[int] = []
        sentinel_feature_counts: list[int] = []
        sentinel_opportunity_counts: list[int] = []
        for row in result.itertuples(index=False):
            pairs = {
                _pair(left, right)
                for left, right in equivalence.parse_candidate_pairs(
                    row.candidate_pairs_json
                )
            }
            candidate_truth_counts.append(len(pairs & local_pairs))
            sentinel_alternatives = sum(
                any(roles.get(member, "") in {"N1", "N2"} for member in pair)
                for pair in pairs
            )
            sentinel_members = {
                member
                for pair in pairs
                for member in pair
                if roles.get(member, "") in {"N1", "N2"}
            }
            sentinel_alternative_counts.append(int(sentinel_alternatives))
            sentinel_feature_counts.append(len(sentinel_members))
            sentinel_flags.append(bool(sentinel_members))
            sentinel_opportunity_counts.append(
                int(sentinel_opportunities_by_modality.get(row.modality, 0))
            )
        result["candidate_truth_pair_count"] = candidate_truth_counts
        result["contains_any_true_pair"] = result[
            "candidate_truth_pair_count"
        ].gt(0)
        result["matched_to_unique_truth_pair"] = result.index.isin(
            matched_by_index
        )
        result["matched_truth_pair_key"] = [
            (
                "|||".join(sorted(matched_by_index[index]))
                if index in matched_by_index
                else ""
            )
            for index in result.index
        ]
        result["contains_sentinel_candidate"] = sentinel_flags
        result["sentinel_candidate_pair_count"] = sentinel_alternative_counts
        result["sentinel_candidate_feature_count"] = sentinel_feature_counts
        result["sentinel_null_feature_opportunities"] = (
            sentinel_opportunity_counts
        )
        result["truth_used_for_equivalence_application"] = False
        result["truth_used_for_phase_b_scoring"] = True
        result["outer_held_block"] = block
        result["outer_truth_used_for_state_fit_or_threshold"] = False
        result.insert(0, "seed", seed)
        result.insert(0, "block", block)
        prediction_frames.append(result)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.duplicated(ASSAY_KEY).any():
        raise RuntimeError("v4.33 predictions contain duplicate assay hypotheses")

    def truth_total(
        frame: pd.DataFrame, totals: Mapping[tuple[str, int], int]
    ) -> int:
        return sum(
            totals[(str(row.block), int(row.seed))]
            for row in frame.loc[:, WORLD_KEY]
            .drop_duplicates()
            .itertuples(index=False)
        )

    world_rows: list[dict[str, Any]] = []
    for (block, seed), frame in predictions.groupby(WORLD_KEY, sort=True):
        local_total = local_truth_totals[(str(block), int(seed))]
        metrics = _coverage_metrics(frame, total_truth=local_total)
        stress = _exact_cross_assay_stress(
            cross_truth_pairs=cross_truth_totals[(str(block), int(seed))],
            direction_truths=direction_truth_totals[(str(block), int(seed))],
        )
        world_rows.append(
            {
                "block": str(block),
                "seed": int(seed),
                **metrics,
                "exact_cross_assay_pair_recall_stress": stress[
                    "exact_cross_assay_pair_recall"
                ],
                "exact_cross_assay_pair_abstention_rate": stress[
                    "exact_cross_assay_pair_abstention_rate"
                ],
                "cross_assay_direction_recovery_stress": np.nan,
                "identifiability_stress_is_reliability_gate": False,
                "whole_layer_reliability_success": bool(
                    metrics["shared_state_hyperedge_precision"] == 1.0
                    and metrics["sentinel_false_positive_hypotheses"] == 0
                    and not metrics[
                        "candidate_pair_alternatives_expanded_as_edges"
                    ]
                ),
            }
        )
    world_metrics = pd.DataFrame(world_rows)

    def aggregate_rows(
        grouped: Iterable[tuple[Any, pd.DataFrame]],
        *,
        label_name: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for label, frame in grouped:
            metrics = _coverage_metrics(
                frame, total_truth=truth_total(frame, local_truth_totals)
            )
            rows.append({label_name: str(label), **metrics})
        return pd.DataFrame(rows)

    block_metrics = aggregate_rows(
        predictions.groupby("block", sort=True), label_name="source_block"
    )
    panel_rows: list[dict[str, Any]] = []
    for panel_id, selected_worlds in panels.groupby("panel_id", sort=True):
        frame = predictions.merge(
            selected_worlds.loc[:, WORLD_KEY],
            on=WORLD_KEY,
            validate="many_to_one",
        )
        metrics = _coverage_metrics(
            frame, total_truth=truth_total(frame, local_truth_totals)
        )
        panel_rows.append(
            {
                "panel_id": str(panel_id),
                "selection_seed": int(selected_worlds["selection_seed"].iloc[0]),
                "worlds": len(selected_worlds),
                **metrics,
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)
    modality_rows: list[dict[str, Any]] = []
    for modality, frame in predictions.groupby("modality", sort=True):
        # The simulator truth contains exactly one assay-local shared-state
        # loading group per chromatin assay and world.
        modality_rows.append(
            {
                "modality": str(modality),
                **_coverage_metrics(frame, total_truth=len(frame)),
            }
        )
    modality_metrics = pd.DataFrame(modality_rows)

    for name, frame in zip(
        PHASE_B_TABLES,
        (
            predictions,
            world_metrics,
            block_metrics,
            panel_metrics,
            modality_metrics,
        ),
        strict=True,
    ):
        frame.to_csv(out_dir / name, sep="\t", index=False)

    overall = _coverage_metrics(
        predictions, total_truth=sum(local_truth_totals.values())
    )
    cross_stress = _exact_cross_assay_stress(
        cross_truth_pairs=sum(cross_truth_totals.values()),
        direction_truths=sum(direction_truth_totals.values()),
    )
    reliability_gates = {
        "shared_state_hyperedge_precision_is_one": (
            overall["shared_state_hyperedge_precision"] == 1.0
        ),
        "sentinel_false_positive_rate_is_zero": (
            overall["sentinel_false_positive_rate"] == 0.0
        ),
        "all_source_blocks_have_hyperedge_precision_one": bool(
            block_metrics["shared_state_hyperedge_precision"].eq(1.0).all()
        ),
        "all_random_panels_have_hyperedge_precision_one": bool(
            panel_metrics["shared_state_hyperedge_precision"].eq(1.0).all()
        ),
        "candidate_alternatives_never_expanded": True,
        "application_does_not_require_truth": True,
        "exact_cross_assay_stress_is_not_a_gate": True,
    }
    summary = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": equivalence.SCHEMA,
        "algorithm": equivalence.ALGORITHM,
        "phase": "b",
        "retrospective_development_only": True,
        "fresh_locked_claim": False,
        "cached_inputs_only": True,
        "new_simulation_worlds_or_datasets_created": False,
        "worlds": len(worlds),
        "source_blocks": int(predictions["block"].nunique()),
        "assays": int(len(predictions)),
        "benchmark_metric_schema_version": benchmark_metrics.SCHEMA,
        "benchmark_metrics": benchmark_metrics.metric_families(
            overall, cross_stress
        ),
        "metric_scope": {
            "shared_state_hyperedges": (
                "assay_local_parent_reporter_partial_identification_hypotheses"
            ),
            "cross_assay_exact_pair_and_direction": (
                "separate_stress_test_because_v433_does_not_align_assay_axes"
            ),
            "one_hypothesis_is_one_prediction": True,
            "ambiguity_alternatives_are_not_edges": True,
        },
        "reliability_gates": reliability_gates,
        "all_reliability_gates_pass": all(reliability_gates.values()),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "upstream_v427": upstream_provenance,
        "truth_input_sha256": truth_hashes,
        "source_sha256": _source_hashes(),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_B_TABLES
        },
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    _write_json(out_dir / "phase_b_summary.json", summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--worlds-root", type=Path, default=SCRIPT_DIR.parent / "formal_worlds"
    )
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--v427-dir", type=Path, default=DEFAULT_V427_DIR)
    parser.add_argument("--selection-seed", type=int, default=20261101)
    parser.add_argument("--panels", type=int, default=5)
    parser.add_argument("--panel-size", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.panels <= 0 or args.panel_size <= 0:
        raise ValueError("panels and panel-size must be positive")
    if args.phase == "a":
        run_phase_a(
            out_dir=args.out_dir,
            direct_dir=args.direct_dir,
            v427_dir=args.v427_dir,
            selection_seed=args.selection_seed,
            panels=args.panels,
            panel_size=args.panel_size,
        )
    else:
        run_phase_b(
            out_dir=args.out_dir,
            worlds_root=args.worlds_root,
            v427_dir=args.v427_dir,
        )


if __name__ == "__main__":
    main()
