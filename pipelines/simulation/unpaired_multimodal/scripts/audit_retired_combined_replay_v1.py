#!/usr/bin/env python3
"""Two-phase retired-only replay of the current layered CA-BOSS components.

Phase A is strictly observable-only.  It combines the landed subtractive
state-v7 redundancy rule, the frozen raw-evidence three-tier direct
replacement rule, and the current structure-anchored loading selector.  All
96 retired TRAIN worlds are loaded and every prediction is serialized and
hashed before Phase B may be run in a separate process.

Phase B verifies the Phase-A freeze, opens only the retired scorer snapshots,
and evaluates the current formal layered world-reliability predicate.  This
script rejects DEVELOPMENT, LOCKED, and the reserved fresh TRAIN390 block.
It is a posthoc development audit, never a certification result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_retired_raw_evidence_collider_gate_v1 import (  # noqa: E402
    KEYS as DIRECT_KEYS,
    STAR_KEYS,
    THRESHOLDS as DIRECT_THRESHOLDS,
    _apply_replacements,
    _select as select_direct_replacements,
    _verify_phase_a as verify_raw_direct_phase_a,
)
from loading_group_projection_v4 import (  # noqa: E402
    LoadingProjectionConfig,
    estimate_loading_group_projection,
    load_evidence_view,
)
from score_layered_state_evidence_v4 import (  # noqa: E402
    MAX_BLOCK_MEAN_FOCAL_DENSITY,
    MAX_WORLD_FOCAL_DENSITY,
    MIN_CONFIRMED_SUBSET_PRECISION,
    MIN_BLOCK_MEAN_DIRECT_RECALL,
    MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION,
    MIN_BLOCK_MEAN_LOADING_PAIR_RECALL,
    MIN_DIRECT_PRECISION,
    MIN_LOADING_PAIR_PRECISION,
    MIN_WORLD_DIRECT_RECALL,
    evaluate_layered_world_reliability,
)
from structure_anchored_loading_groups_v4 import (  # noqa: E402
    StructureAnchoredGroupConfig,
    estimate_structure_anchored_groups,
)


SCHEMA = "ca_boss_v4.retired_combined_replay.v1"
RAW_DIRECT_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "retired_crossfit_hierarchical_collider_audit_v1"
)
FROZEN_DIRECT_OBSERVED_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "retired_raw_evidence_collider_gate_audit_v1_complete"
)
FROZEN_STATE_AUDIT_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "retired_train_360_370_380_redundancy_state_confirmation_v1"
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
)
BLOCK_BY_NAME = {block.name: block for block in BLOCKS}
STATE_ROLES = frozenset({"P", "Q", "S", "C"})
NULL_ROLES = frozenset({"N1", "N2"})
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "dev",
        "development",
        "locked",
        "train_34001_34016",
        "train_39001_39024",
        "train_45001_45040",
        "fresh390",
    }
)
TRUTH_PATH_PARTS = frozenset(
    {"scorer", "truth", "role_assignments", "truth_typed_edges"}
)


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


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Not a strict Boolean: {value!r}")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _assert_safe_path(path: Path, *, allow_truth: bool = False) -> Path:
    lexical = path.resolve(strict=False)
    folded_parts = {part.casefold() for part in lexical.parts}
    if folded_parts & FORBIDDEN_PATH_PARTS:
        raise ValueError(f"Refusing protected path: {path}")
    if not allow_truth and folded_parts & TRUTH_PATH_PARTS:
        raise ValueError(f"Phase A refuses scorer/truth path: {path}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _candidate_dir(run_root: Path, seed: int) -> Path:
    world = run_root / "artifacts" / f"world_seed{seed}"
    candidates = sorted(
        path
        for path in world.glob("v4_*")
        if path.is_dir() and not path.is_symlink()
        and (path / "loading_group_projection").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one frozen candidate for seed {seed}; found {len(candidates)}"
        )
    return candidates[0]


def _apply_state_v7_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the landed strictly subtractive redundancy confirmation."""

    required = {
        "feature",
        "modality",
        "selection_status",
        "feature_global_maxT_p",
        "reference_conditional_wald",
        "direction_stability_diagnostic",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"State membership lacks columns: {missing}")
    if frame["feature"].astype(str).duplicated().any():
        raise ValueError("State membership contains duplicate features")
    output = frame.copy().reset_index(drop=True)
    preliminary = output["selection_status"].astype(str).eq("selected")
    retained: set[int] = set()
    fallback: set[int] = set()
    for _, group in output.loc[preliminary].groupby("modality", sort=True):
        stable = group.loc[
            group["direction_stability_diagnostic"].map(_strict_bool)
        ]
        if not stable.empty:
            retained.update(map(int, stable.index))
            continue
        winner = group.sort_values(
            ["feature_global_maxT_p", "reference_conditional_wald", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        ).index[0]
        retained.add(int(winner))
        fallback.add(int(winner))
    output["pre_v7_selection_status"] = output["selection_status"].astype(str)
    output["v7_redundancy_confirmed"] = output.index.isin(retained)
    output["v7_redundancy_fallback_selected"] = output.index.isin(fallback)
    output["v7_redundancy_demoted"] = preliminary & ~output.index.isin(retained)
    output.loc[output["v7_redundancy_demoted"], "selection_status"] = "unresolved"
    if "abstention_reason" in output:
        output.loc[
            output["v7_redundancy_demoted"], "abstention_reason"
        ] = "within_assay_redundancy_direction_instability"
    if "subspace_id" in output:
        output.loc[output["selection_status"].ne("selected"), "subspace_id"] = ""
    return output


def _selected_state_signature(frame: pd.DataFrame) -> set[str]:
    return set(
        frame.loc[
            frame["selection_status"].astype(str).eq("selected"), "feature"
        ].astype(str)
    )


def _direct_frame(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.duplicated(["source_assay", "parent", "child"]).any():
        raise ValueError("Direct prediction contains duplicate directed edges")
    return pd.DataFrame(
        {
            "source_assay": edges["source_assay"].astype(str),
            "node1": edges["parent"].astype(str),
            "endpoint1": "TAIL",
            "node2": edges["child"].astype(str),
            "endpoint2": "ARROW",
            "edge_type": "within_assay_direct_candidate",
            "collider_motif_center": edges["child"].astype(str),
        }
    )


def _motif_constraints(direct: pd.DataFrame) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    centers: set[str] = set()
    parent_pairs: set[tuple[str, str]] = set()
    for (_, child), group in direct.groupby(
        ["source_assay", "child"], sort=True
    ):
        parents = sorted(set(group["parent"].astype(str)))
        if len(group) == 2 and len(parents) == 2:
            centers.add(str(child))
            parent_pairs.add((parents[0], parents[1]))
    return tuple(sorted(centers)), tuple(sorted(parent_pairs))


def _loading_projection_config() -> LoadingProjectionConfig:
    return LoadingProjectionConfig(
        min_within_absolute_cosine=0.85,
        min_group_relative_margin=0.10,
        cross_projection_empirical_q=0.40,
        allow_cross_assay_group_matching=False,
    )


def _selected_loading_signature(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    signature: set[tuple[str, str, str]] = set()
    if frame.empty:
        return signature
    for row in frame.loc[
        frame["selection_status"].astype(str).eq("selected")
    ].itertuples(index=False):
        members = sorted(
            member.strip()
            for member in str(row.members).split(";")
            if member.strip()
        )
        if len(members) != 2:
            raise ValueError("Structure-anchored group is not a pair")
        signature.add((str(row.modality), members[0], members[1]))
    return signature


def _state_hyperedge(
    source: pd.DataFrame, state: pd.DataFrame
) -> pd.DataFrame:
    selected = state.loc[
        state["selection_status"].astype(str).eq("selected")
    ]
    members = sorted(selected["feature"].astype(str))
    modalities = sorted(set(selected["modality"].astype(str)))
    if len(source) != 1:
        raise ValueError("Expected one frozen state hyperedge row")
    row = source.iloc[0].copy()
    row["member_features"] = ";".join(members)
    row["modalities"] = ";".join(modalities)
    row["n_features"] = len(members)
    row["n_modalities"] = len(modalities)
    return pd.DataFrame([row])


def _verify_prior_state_phase_a() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = _assert_safe_path(FROZEN_STATE_AUDIT_ROOT / "phase_a_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_path = _assert_safe_path(FROZEN_STATE_AUDIT_ROOT / "phase_a_fixed_state_filter.tsv")
    if _sha256(table_path) != manifest.get("phase_a_table_sha256"):
        raise ValueError("Frozen state-v7 Phase-A table drifted")
    if manifest.get("all_worlds_loaded_before_any_truth_open") is not True:
        raise ValueError("Frozen state audit lacks a global pretruth barrier")
    if manifest.get("dev_or_locked_opened") is not False:
        raise ValueError("Frozen state audit reports protected data access")
    return pd.read_csv(table_path, sep="\t"), {
        "phase_a_manifest_sha256": _sha256(manifest_path),
        "phase_a_table_sha256": _sha256(table_path),
    }


def _verify_prior_direct_replay(
    selected: pd.DataFrame,
) -> dict[str, Any]:
    path = _assert_safe_path(
        FROZEN_DIRECT_OBSERVED_ROOT / "observed_only_selected_replacements.tsv"
    )
    # Deliberately do not open this audit directory's post-label manifest in
    # Phase A.  The table itself contains only observed candidate fields and
    # is checked against a fresh selection from the independently frozen raw
    # Phase-A universe.
    prior = pd.read_csv(path, sep="\t")
    columns = list(STAR_KEYS) + ["selection_tier"]
    left = {
        tuple(row)
        for row in selected[columns].itertuples(index=False, name=None)
    }
    right = {
        tuple(row) for row in prior[columns].itertuples(index=False, name=None)
    }
    if left != right:
        raise ValueError("Recomputed raw three-tier replacements differ from freeze")
    return {
        "replacement_table_sha256": _sha256(path),
        "n_exact_observed_replacements": len(prior),
        "postlabel_manifest_opened": False,
    }


def _phase_a(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_identity = verify_raw_direct_phase_a(RAW_DIRECT_ROOT)
    raw_candidates_path = _assert_safe_path(
        RAW_DIRECT_ROOT / "phase_a_crossfit_star_candidates.tsv"
    )
    production_path = _assert_safe_path(
        RAW_DIRECT_ROOT / "phase_a_production_direct_edges.tsv"
    )
    production_status_path = _assert_safe_path(
        RAW_DIRECT_ROOT / "phase_a_production_assay_status.tsv"
    )
    raw_candidates = pd.read_csv(raw_candidates_path, sep="\t")
    production = pd.read_csv(production_path, sep="\t")
    production_status = pd.read_csv(production_status_path, sep="\t")
    replacements, direct_decisions = select_direct_replacements(
        raw_candidates,
        production,
        production_status,
        DIRECT_THRESHOLDS,
    )
    replacements = replacements.sort_values(list(STAR_KEYS)).reset_index(drop=True)
    direct_decisions = direct_decisions.sort_values(list(DIRECT_KEYS)).reset_index(drop=True)
    prior_direct_identity = _verify_prior_direct_replay(replacements)
    final_direct = _apply_replacements(production, replacements).sort_values(
        [*DIRECT_KEYS, "parent", "child"]
    ).reset_index(drop=True)

    prior_state, prior_state_identity = _verify_prior_state_phase_a()
    prior_state_selected = {
        (str(block), int(seed)): set(
            group.loc[group["redundancy_confirmed"].map(_strict_bool), "feature"].astype(str)
        )
        for (block, seed), group in prior_state.groupby(["block", "seed"], sort=True)
    }

    input_hashes: dict[str, str] = {
        str(raw_candidates_path): _sha256(raw_candidates_path),
        str(production_path): _sha256(production_path),
        str(production_status_path): _sha256(production_status_path),
    }
    state_frames: list[pd.DataFrame] = []
    hyperedge_frames: list[pd.DataFrame] = []
    loading_group_frames: list[pd.DataFrame] = []
    loading_decision_frames: list[pd.DataFrame] = []
    loading_pair_frames: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    projection_cfg = _loading_projection_config()
    anchor_cfg = StructureAnchoredGroupConfig()

    # This loop completes all 96 observed calculations before any output is
    # labeled and before Phase B is allowed to open a scorer directory.
    for block in BLOCKS:
        freeze = _assert_safe_path(block.run_root / "global_pretruth_freeze.json")
        input_hashes[str(freeze)] = _sha256(freeze)
        for seed in block.seeds:
            world_artifact = block.run_root / "artifacts" / f"world_seed{seed}"
            state_path = _assert_safe_path(
                world_artifact
                / "shared_state_subspace_v4"
                / "subspace_feature_membership.tsv"
            )
            hyperedge_path = _assert_safe_path(
                world_artifact
                / "shared_state_subspace_v4"
                / "typed_subspace_hyperedges.tsv"
            )
            input_hashes[str(state_path)] = _sha256(state_path)
            input_hashes[str(hyperedge_path)] = _sha256(hyperedge_path)
            original_state = pd.read_csv(state_path, sep="\t")
            filtered_state = _apply_state_v7_filter(original_state)
            tagged_state = filtered_state.copy()
            tagged_state.insert(0, "seed", seed)
            tagged_state.insert(0, "block", block.name)
            state_frames.append(tagged_state)

            reconstructed_hyperedge = _state_hyperedge(
                pd.read_csv(hyperedge_path, sep="\t"), filtered_state
            )
            reconstructed_hyperedge.insert(0, "seed", seed)
            reconstructed_hyperedge.insert(0, "block", block.name)
            hyperedge_frames.append(reconstructed_hyperedge)

            state_prior_available = block.name != "TRAIN330"
            state_prior_match: bool | None = None
            if state_prior_available:
                state_prior_match = (
                    _selected_state_signature(filtered_state)
                    == prior_state_selected[(block.name, seed)]
                )
                if not state_prior_match:
                    raise ValueError(
                        f"State-v7 replay differs from frozen audit for {block.name}/{seed}"
                    )

            candidate = _candidate_dir(block.run_root, seed)
            world_final_direct = final_direct.loc[
                final_direct["block"].astype(str).eq(block.name)
                & final_direct["seed"].astype(int).eq(seed)
            ].copy()
            world_production_direct = production.loc[
                production["block"].astype(str).eq(block.name)
                & production["seed"].astype(int).eq(seed)
            ].copy()
            direct_table = _direct_frame(world_final_direct)
            centers, parent_pairs = _motif_constraints(world_final_direct)

            view_dirs = sorted(
                path
                for path in world_artifact.glob("cells*_seed*")
                if path.is_dir() and not path.is_symlink()
            )
            if len(view_dirs) != 6:
                raise ValueError(f"Expected six loading views for seed {seed}")
            views = []
            empirical_frames: list[pd.DataFrame] = []
            for view_dir in view_dirs:
                for name in (
                    "cell_loading_feature_models.tsv",
                    "cell_loading_loadings.tsv",
                    "cell_loading_pair_evidence.tsv",
                ):
                    path = _assert_safe_path(view_dir / name)
                    input_hashes[str(path)] = _sha256(path)
                view = load_evidence_view(view_dir)
                views.append(view)
                empirical = view.pair_evidence.copy()
                empirical["view_id"] = str(view.view_id)
                empirical_frames.append(empirical)

            projection = estimate_loading_group_projection(
                views,
                projection_cfg,
                excluded_features=centers,
                forbidden_within_pairs=parent_pairs,
            )
            empirical = pd.concat(empirical_frames, ignore_index=True, sort=False)
            anchor = estimate_structure_anchored_groups(
                direct_table,
                projection.within_group_candidates,
                empirical,
                filtered_state,
                anchor_cfg,
            )
            for frame, sink in (
                (anchor.typed_groups, loading_group_frames),
                (anchor.assay_decisions, loading_decision_frames),
                (anchor.pair_evidence, loading_pair_frames),
            ):
                tagged = frame.copy()
                tagged.insert(0, "seed", seed)
                tagged.insert(0, "block", block.name)
                sink.append(tagged)

            # TRAIN380 is the only block with a persisted current-v4 loading
            # artifact.  Re-run its original inputs and require exact selected
            # pair and decision-stage agreement before trusting the replay.
            loading_prior_available = block.name == "TRAIN380"
            loading_prior_match: bool | None = None
            if loading_prior_available:
                stored_path = _assert_safe_path(
                    candidate
                    / "structure_anchored_loading_groups"
                    / "typed_structure_anchored_groups.tsv"
                )
                stored_decision_path = _assert_safe_path(
                    candidate
                    / "structure_anchored_loading_groups"
                    / "assay_decisions.tsv"
                )
                provenance_path = _assert_safe_path(
                    candidate
                    / "structure_anchored_loading_groups"
                    / "structure_anchor_provenance.json"
                )
                for path in (stored_path, stored_decision_path, provenance_path):
                    input_hashes[str(path)] = _sha256(path)
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance.get("schema_version") != "structure_anchored_loading_groups_v4.v4":
                    raise ValueError(f"TRAIN380 loading artifact is not current v4: {seed}")
                prod_centers, prod_parent_pairs = _motif_constraints(
                    world_production_direct
                )
                prod_projection = estimate_loading_group_projection(
                    views,
                    projection_cfg,
                    excluded_features=prod_centers,
                    forbidden_within_pairs=prod_parent_pairs,
                )
                prod_anchor = estimate_structure_anchored_groups(
                    _direct_frame(world_production_direct),
                    prod_projection.within_group_candidates,
                    empirical,
                    original_state,
                    anchor_cfg,
                )
                stored = pd.read_csv(stored_path, sep="\t")
                stored_decisions = pd.read_csv(stored_decision_path, sep="\t")
                selected_match = (
                    _selected_loading_signature(prod_anchor.typed_groups)
                    == _selected_loading_signature(stored)
                )
                decision_columns = [
                    "modality",
                    "selection_status",
                    "selection_stage",
                    "selected_pair_key",
                ]
                replay_decisions = {
                    tuple(row)
                    for row in prod_anchor.assay_decisions[
                        decision_columns
                    ].fillna("").itertuples(index=False, name=None)
                }
                stored_decision_signature = {
                    tuple(row)
                    for row in stored_decisions[decision_columns]
                    .fillna("")
                    .itertuples(index=False, name=None)
                }
                loading_prior_match = bool(
                    selected_match
                    and replay_decisions == stored_decision_signature
                )
                if not loading_prior_match:
                    raise ValueError(
                        f"Current loading-v4 replay differs from stored TRAIN380 seed {seed}"
                    )

            validation_rows.append(
                {
                    "block": block.name,
                    "seed": seed,
                    "state_v7_prior_exact_artifact_available": state_prior_available,
                    "state_v7_prior_selected_set_match": state_prior_match,
                    "direct_three_tier_prior_exact_artifact_available": True,
                    "direct_three_tier_prior_selected_set_match": True,
                    "loading_v4_prior_exact_artifact_available": loading_prior_available,
                    "loading_v4_original_selected_and_stage_match": loading_prior_match,
                    "combined_prediction_status": (
                        "exact_replay_components_available"
                        if state_prior_available and loading_prior_available
                        else "recomputed_from_frozen_observed_inputs_where_prior_artifact_unavailable"
                    ),
                }
            )

    outputs: dict[str, pd.DataFrame] = {
        "phase_a_state_membership.tsv": pd.concat(state_frames, ignore_index=True, sort=False),
        "phase_a_state_hyperedges.tsv": pd.concat(hyperedge_frames, ignore_index=True, sort=False),
        "phase_a_direct_replacements.tsv": replacements,
        "phase_a_direct_assay_decisions.tsv": direct_decisions,
        "phase_a_direct_edges.tsv": final_direct,
        "phase_a_loading_groups.tsv": pd.concat(loading_group_frames, ignore_index=True, sort=False),
        "phase_a_loading_assay_decisions.tsv": pd.concat(loading_decision_frames, ignore_index=True, sort=False),
        "phase_a_loading_pair_evidence.tsv": pd.concat(loading_pair_frames, ignore_index=True, sort=False),
        "phase_a_replay_validation.tsv": pd.DataFrame(validation_rows),
    }
    output_hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, sep="\t", index=False)
        output_hashes[name] = _sha256(path)

    availability_rows = []
    validation = outputs["phase_a_replay_validation.tsv"]
    for block in BLOCKS:
        group = validation.loc[validation["block"].eq(block.name)]
        availability_rows.extend(
            [
                {
                    "block": block.name,
                    "component": "state_v7_redundancy",
                    "prior_exact_artifact_available": bool(block.name != "TRAIN330"),
                    "replay_basis": (
                        "matched_retired_state_v7_phase_a_artifact"
                        if block.name != "TRAIN330"
                        else "recomputed_identical_rule_from_frozen_membership"
                    ),
                    "n_worlds": len(group),
                    "n_prior_matches": int(
                        group["state_v7_prior_selected_set_match"].fillna(False).sum()
                    ),
                },
                {
                    "block": block.name,
                    "component": "raw_three_tier_direct",
                    "prior_exact_artifact_available": True,
                    "replay_basis": "matched_frozen_observed_only_replacement_artifact",
                    "n_worlds": len(group),
                    "n_prior_matches": len(group),
                },
                {
                    "block": block.name,
                    "component": "current_structure_anchored_loading_v4",
                    "prior_exact_artifact_available": bool(block.name == "TRAIN380"),
                    "replay_basis": (
                        "matched_stored_current_v4_original_decisions_then_replayed_combination"
                        if block.name == "TRAIN380"
                        else "recomputed_current_v4_from_six_frozen_observed_views"
                    ),
                    "n_worlds": len(group),
                    "n_prior_matches": int(
                        group["loading_v4_original_selected_and_stage_match"].fillna(False).sum()
                    ),
                },
            ]
        )
    availability_path = out_dir / "phase_a_component_availability.tsv"
    pd.DataFrame(availability_rows).to_csv(availability_path, sep="\t", index=False)
    output_hashes[availability_path.name] = _sha256(availability_path)

    source_paths = {
        Path(__file__).resolve(),
        SCRIPT_DIR / "audit_retired_raw_evidence_collider_gate_v1.py",
        SCRIPT_DIR / "loading_group_projection_v4.py",
        SCRIPT_DIR / "structure_anchored_loading_groups_v4.py",
        SCRIPT_DIR / "score_layered_state_evidence_v4.py",
    }
    manifest = {
        "schema_version": f"{SCHEMA}.phase_a",
        "phase": "observed_only_before_truth",
        "status": "retired_posthoc_development_not_certification",
        "blocks": {
            block.name: [block.start, block.end] for block in BLOCKS
        },
        "n_worlds": sum(len(block.seeds) for block in BLOCKS),
        "all_observed_worlds_loaded_and_calculated_before_manifest": True,
        "truth_or_scorer_opened": False,
        "dev_locked_or_fresh390_opened": False,
        "raw_direct_phase_a_identity": raw_identity,
        "prior_direct_replay_identity": prior_direct_identity,
        "prior_state_phase_a_identity": prior_state_identity,
        "direct_thresholds": DIRECT_THRESHOLDS,
        "loading_projection_config": asdict(projection_cfg),
        "structure_anchored_loading_config": asdict(anchor_cfg),
        "actual_formal_world_gate_constants": {
            "minimum_confirmed_subset_precision": MIN_CONFIRMED_SUBSET_PRECISION,
            "minimum_direct_precision": MIN_DIRECT_PRECISION,
            "minimum_direct_recall": MIN_WORLD_DIRECT_RECALL,
            "minimum_loading_pair_precision": MIN_LOADING_PAIR_PRECISION,
            "maximum_focal_density": MAX_WORLD_FOCAL_DENSITY,
        },
        "input_sha256": dict(sorted(input_hashes.items())),
        "source_sha256": {
            str(path): _sha256(path) for path in sorted(source_paths)
        },
        "output_sha256": dict(sorted(output_hashes.items())),
        "phase_b_must_run_as_separate_process": True,
    }
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _verify_phase_a_freeze(out_dir: Path) -> dict[str, Any]:
    path = _assert_safe_path(out_dir / "phase_a_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Phase-A manifest is not an observed-only freeze")
    if manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Phase A reports scorer/truth access")
    if manifest.get("dev_locked_or_fresh390_opened") is not False:
        raise ValueError("Phase A reports protected block access")
    for name, expected in manifest.get("output_sha256", {}).items():
        candidate = _assert_safe_path(out_dir / name)
        if _sha256(candidate) != expected:
            raise ValueError(f"Phase-A output drift: {name}")
    for source_text, expected in manifest.get("source_sha256", {}).items():
        source = Path(source_text)
        if not source.is_file() or source.is_symlink() or _sha256(source) != expected:
            raise ValueError(f"Executable source changed after Phase A: {source}")
    return manifest


def _pair_members(row: Any) -> tuple[str, str]:
    members = sorted(
        member.strip()
        for member in str(row.members).split(";")
        if member.strip()
    )
    if len(members) != 2:
        raise ValueError("Selected loading group does not have two members")
    return members[0], members[1]


def _phase_b(out_dir: Path) -> None:
    phase_a_manifest = _verify_phase_a_freeze(out_dir)
    state = pd.read_csv(out_dir / "phase_a_state_membership.tsv", sep="\t")
    hyperedges = pd.read_csv(out_dir / "phase_a_state_hyperedges.tsv", sep="\t")
    direct = pd.read_csv(out_dir / "phase_a_direct_edges.tsv", sep="\t")
    loading = pd.read_csv(out_dir / "phase_a_loading_groups.tsv", sep="\t")

    world_rows: list[dict[str, Any]] = []
    state_outcomes: list[dict[str, Any]] = []
    direct_outcomes: list[dict[str, Any]] = []
    loading_outcomes: list[dict[str, Any]] = []
    truth_hashes: dict[str, str] = {}

    for block in BLOCKS:
        for seed in block.seeds:
            scorer = block.world_root / f"world_seed{seed}" / "scorer"
            roles_path = _assert_safe_path(
                scorer / "role_assignments.tsv", allow_truth=True
            )
            truth_path = _assert_safe_path(
                scorer / "truth_typed_edges.tsv", allow_truth=True
            )
            truth_hashes[str(roles_path)] = _sha256(roles_path)
            truth_hashes[str(truth_path)] = _sha256(truth_path)
            roles = pd.read_csv(roles_path, sep="\t")
            truth = pd.read_csv(truth_path, sep="\t")
            role_map = {
                str(row.node): str(row.role)
                for row in roles[["node", "role"]].itertuples(index=False)
            }

            state_world = state.loc[
                state["block"].astype(str).eq(block.name)
                & state["seed"].astype(int).eq(seed)
            ].copy()
            selected_state = set(
                state_world.loc[
                    state_world["selection_status"].astype(str).eq("selected"),
                    "feature",
                ].astype(str)
            )
            rna_features = set(
                state_world.loc[
                    state_world["modality"].astype(str).eq("rna"), "feature"
                ].astype(str)
            )
            if len(rna_features) != 1:
                raise ValueError(f"Expected one RNA feature in seed {seed}")
            gene = next(iter(rna_features))
            positives = {
                feature
                for feature, role in role_map.items()
                if role in STATE_ROLES
            } | {gene}
            sentinels = {
                feature
                for feature, role in role_map.items()
                if role in NULL_ROLES
            }
            true_state = selected_state & positives
            false_state = selected_state & sentinels
            state_precision = _rate(len(true_state), len(selected_state))
            selected_modalities = sorted(
                set(
                    state_world.loc[
                        state_world["feature"].astype(str).isin(selected_state),
                        "modality",
                    ].astype(str)
                )
            )
            state_subset_reliable = bool(
                state_precision >= MIN_CONFIRMED_SUBSET_PRECISION
                and not false_state
                and len(selected_modalities) == 7
                and gene in selected_state
            )
            for row in state_world.itertuples(index=False):
                feature = str(row.feature)
                state_outcomes.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "feature": feature,
                        "modality": str(row.modality),
                        "role": role_map.get(feature, "GENE" if feature == gene else ""),
                        "selected": feature in selected_state,
                        "truth_state_member": feature in positives,
                        "truth_sentinel": feature in sentinels,
                    }
                )

            direct_world = direct.loc[
                direct["block"].astype(str).eq(block.name)
                & direct["seed"].astype(int).eq(seed)
            ].copy()
            if direct_world.duplicated(["parent", "child"]).any():
                raise ValueError(f"Duplicate direct edge in seed {seed}")
            predicted_direct = {
                (str(row.parent), str(row.child))
                for row in direct_world.itertuples(index=False)
            }
            true_direct_frame = truth.loc[
                truth["truth_type"].astype(str).eq("within_assay_direct")
            ]
            true_direct = {
                (str(row.parent), str(row.child))
                for row in true_direct_frame.itertuples(index=False)
            }
            predicted_skeleton = {
                frozenset(edge) for edge in predicted_direct
            }
            true_skeleton = {frozenset(edge) for edge in true_direct}
            true_predicted = predicted_skeleton & true_skeleton
            direct_fp = predicted_skeleton - true_skeleton
            direct_fn = true_skeleton - predicted_skeleton
            strict_reverse = {
                (parent, child)
                for parent, child in predicted_direct
                if (child, parent) in true_direct
            }
            direct_precision = _rate(len(true_predicted), len(predicted_skeleton))
            direct_recall = _rate(len(true_predicted), len(true_skeleton))
            direct_formal_reliable = bool(
                direct_precision >= MIN_DIRECT_PRECISION
                and direct_recall >= MIN_WORLD_DIRECT_RECALL
                and not strict_reverse
            )
            direct_precision99_reliable = bool(
                direct_precision >= 0.99
                and direct_recall >= 0.90
                and not strict_reverse
            )
            for parent, child in sorted(predicted_direct | true_direct):
                skeleton = frozenset((parent, child))
                direct_outcomes.append(
                    {
                        "block": block.name,
                        "seed": seed,
                        "parent": parent,
                        "child": child,
                        "predicted_oriented": (parent, child) in predicted_direct,
                        "truth_oriented": (parent, child) in true_direct,
                        "predicted_skeleton": skeleton in predicted_skeleton,
                        "truth_skeleton": skeleton in true_skeleton,
                        "strict_reverse": (parent, child) in strict_reverse,
                    }
                )

            loading_world = loading.loc[
                loading["block"].astype(str).eq(block.name)
                & loading["seed"].astype(int).eq(seed)
                & loading["selection_status"].astype(str).eq("selected")
            ].copy()
            predicted_loading: set[frozenset[str]] = set()
            loading_member_mismatches = 0
            selected_loading_members: set[str] = set()
            for row in loading_world.itertuples(index=False):
                left, right = _pair_members(row)
                pair = frozenset((left, right))
                if pair in predicted_loading:
                    raise ValueError(f"Duplicate loading pair in seed {seed}")
                predicted_loading.add(pair)
                selected_loading_members.update(pair)
                expected_suffix = f"__{str(row.modality).removeprefix('chrom_')}"
                loading_member_mismatches += int(
                    not left.endswith(expected_suffix) or not right.endswith(expected_suffix)
                )
            true_loading_frame = truth.loc[
                truth["truth_type"].astype(str).eq("within_assay_shared_state")
            ]
            true_loading = {
                frozenset((str(row.node1), str(row.node2)))
                for row in true_loading_frame.itertuples(index=False)
            }
            true_loading_predicted = predicted_loading & true_loading
            loading_fp = predicted_loading - true_loading
            loading_fn = true_loading - predicted_loading
            loading_precision = _rate(
                len(true_loading_predicted), len(predicted_loading)
            )
            loading_recall = _rate(
                len(true_loading_predicted), len(true_loading)
            )
            sentinel_loading_members = selected_loading_members & sentinels
            loading_formal_reliable = bool(
                loading_precision >= MIN_LOADING_PAIR_PRECISION
                and not sentinel_loading_members
                and loading_member_mismatches == 0
            )
            for pair in sorted(
                predicted_loading | true_loading, key=lambda value: tuple(sorted(value))
            ):
                left, right = sorted(pair)
                loading_outcomes.append(
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

            hyperedge_world = hyperedges.loc[
                hyperedges["block"].astype(str).eq(block.name)
                & hyperedges["seed"].astype(int).eq(seed)
            ]
            selected_hyperedges = hyperedge_world.loc[
                hyperedge_world["selection_status"].astype(str).eq("selected")
            ]
            if len(hyperedge_world) != 1:
                raise ValueError(f"Expected one replayed hyperedge for seed {seed}")
            hyperedge = hyperedge_world.iloc[0]
            hyperedge_members = {
                member.strip()
                for member in str(hyperedge.member_features).split(";")
                if member.strip()
            }
            hyperedge_modalities = {
                modality.strip()
                for modality in str(hyperedge.modalities).split(";")
                if modality.strip()
            }
            hyperedge_safe = bool(
                _strict_bool(hyperedge.conditional_on_common_state_model)
                and str(hyperedge.causal_interpretation) == "hypothesis_only"
                and not _strict_bool(hyperedge.direction_claimed)
                and not _strict_bool(hyperedge.exact_axis_pairing_claimed)
                and not _strict_bool(hyperedge.direct_causal_claimed)
            )
            focal_density = len(predicted_skeleton) + len(predicted_loading) + len(
                selected_hyperedges
            )
            direct_score = {
                "n_predicted": len(predicted_skeleton),
                "n_true_predicted": len(true_predicted),
                "n_false_predicted": len(direct_fp),
                "skeleton_precision": direct_precision,
                "skeleton_recall": direct_recall,
                "n_correctly_oriented": len(predicted_direct & true_direct),
                "n_strict_reverse": len(strict_reverse),
            }
            loading_score = {
                "n_selected_hyperedges": len(loading_world),
                "n_expanded_pairs": len(predicted_loading),
                "n_true_expanded_pairs": len(true_loading_predicted),
                "n_false_expanded_pairs": len(loading_fp),
                "pair_precision": loading_precision,
                "pair_recall": loading_recall,
                "n_sentinel_members": len(sentinel_loading_members),
                "n_non_pair_hyperedges": 0,
                "n_feature_membership_mismatches": loading_member_mismatches,
            }
            gate_payload = {
                "set_scores": {
                    "confirmed_state_response": {
                        "n_selected": len(selected_state),
                        "n_true_selected": len(true_state),
                        "n_sentinel_selected": len(false_state),
                        "n_unknown_selected": len(selected_state - positives - sentinels),
                        "precision": state_precision,
                        "recall": _rate(len(true_state), len(positives)),
                    }
                },
                "confirmed_modalities": selected_modalities,
                "confirmed_assay_coverage": _rate(len(selected_modalities), 7),
                "confirmed_gene_present": gene in selected_state,
                "confirmed_state_recovery_success": state_subset_reliable,
                "direct_score": direct_score,
                "assay_local_loading_pair_score": loading_score,
                "cross_assay_state_hyperedge": {
                    "n_selected": len(selected_hyperedges),
                    "members_equal_confirmed_subset": hyperedge_members == selected_state,
                    "modalities_equal_confirmed_modalities": (
                        hyperedge_modalities == set(selected_modalities)
                    ),
                    "safe_hypothesis_semantics": hyperedge_safe,
                },
                "typed_focal_graph": {
                    "n_expanded_focal_relations": focal_density,
                },
                "semantics": {
                    "layer_union_is_state_membership": False,
                    "cross_assay_pair_claimed": False,
                    "direct_cross_assay_effect_claimed": False,
                },
            }
            formal = evaluate_layered_world_reliability(gate_payload)
            failed_checks = sorted(
                name for name, passed in formal["gate_checks"].items() if not passed
            )
            world_rows.append(
                {
                    "block": block.name,
                    "seed": seed,
                    "state_selected": len(selected_state),
                    "state_true_selected": len(true_state),
                    "state_sentinel_selected": len(false_state),
                    "state_precision": state_precision,
                    "state_assay_count": len(selected_modalities),
                    "state_subset_reliable": state_subset_reliable,
                    "direct_tp": len(true_predicted),
                    "direct_fp": len(direct_fp),
                    "direct_fn": len(direct_fn),
                    "direct_strict_reverse": len(strict_reverse),
                    "direct_precision": direct_precision,
                    "direct_recall": direct_recall,
                    "direct_formal_reliable": direct_formal_reliable,
                    "direct_precision99_reliable": direct_precision99_reliable,
                    "loading_tp": len(true_loading_predicted),
                    "loading_fp": len(loading_fp),
                    "loading_fn": len(loading_fn),
                    "loading_precision": loading_precision,
                    "loading_recall": loading_recall,
                    "loading_sentinel_members": len(sentinel_loading_members),
                    "loading_member_mismatches": loading_member_mismatches,
                    "loading_formal_reliable": loading_formal_reliable,
                    "expanded_focal_density": focal_density,
                    "formal_world_reliability_success": bool(
                        formal["world_reliability_success"]
                    ),
                    "failed_formal_gate_checks": ";".join(failed_checks),
                }
            )

    worlds = pd.DataFrame(world_rows).sort_values(["block", "seed"])
    block_rows: list[dict[str, Any]] = []
    for block_name, group in worlds.groupby("block", sort=True):
        direct_tp = int(group["direct_tp"].sum())
        direct_fp = int(group["direct_fp"].sum())
        direct_fn = int(group["direct_fn"].sum())
        loading_tp = int(group["loading_tp"].sum())
        loading_fp = int(group["loading_fp"].sum())
        loading_fn = int(group["loading_fn"].sum())
        combined_reliable = int(group["formal_world_reliability_success"].sum())
        mean_direct_recall = float(group["direct_recall"].mean())
        mean_loading_precision = float(group["loading_precision"].mean())
        mean_loading_recall = float(group["loading_recall"].mean())
        mean_density = float(group["expanded_focal_density"].mean())
        world_count_gate = combined_reliable >= 22
        direct_mean_gate = mean_direct_recall >= MIN_BLOCK_MEAN_DIRECT_RECALL
        loading_precision_gate = (
            mean_loading_precision >= MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION
        )
        loading_recall_gate = (
            mean_loading_recall >= MIN_BLOCK_MEAN_LOADING_PAIR_RECALL
        )
        density_gate = mean_density <= MAX_BLOCK_MEAN_FOCAL_DENSITY
        block_rows.append(
            {
                "block": block_name,
                "n_worlds": len(group),
                "state_reliable_worlds": int(group["state_subset_reliable"].sum()),
                "direct_formal_reliable_worlds": int(group["direct_formal_reliable"].sum()),
                "direct_precision99_reliable_worlds": int(
                    group["direct_precision99_reliable"].sum()
                ),
                "loading_formal_reliable_worlds": int(
                    group["loading_formal_reliable"].sum()
                ),
                "combined_formal_reliable_worlds": combined_reliable,
                "combined_meets_22_of_24": world_count_gate,
                "block_mean_direct_recall_gate": direct_mean_gate,
                "block_mean_loading_precision_gate": loading_precision_gate,
                "block_mean_loading_recall_gate": loading_recall_gate,
                "block_mean_density_gate": density_gate,
                "formal_replay_block_success": bool(
                    world_count_gate
                    and direct_mean_gate
                    and loading_precision_gate
                    and loading_recall_gate
                    and density_gate
                ),
                "direct_tp": direct_tp,
                "direct_fp": direct_fp,
                "direct_fn": direct_fn,
                "direct_strict_reverse": int(group["direct_strict_reverse"].sum()),
                "direct_micro_precision": _rate(direct_tp, direct_tp + direct_fp),
                "direct_micro_recall": _rate(direct_tp, direct_tp + direct_fn),
                "mean_direct_recall": mean_direct_recall,
                "loading_tp": loading_tp,
                "loading_fp": loading_fp,
                "loading_fn": loading_fn,
                "loading_micro_precision": _rate(loading_tp, loading_tp + loading_fp),
                "loading_micro_recall": _rate(loading_tp, loading_tp + loading_fn),
                "mean_loading_precision": mean_loading_precision,
                "mean_loading_recall": mean_loading_recall,
                "mean_expanded_focal_density": mean_density,
            }
        )
    block_summary = pd.DataFrame(block_rows)

    output_frames = {
        "phase_b_component_world_metrics.tsv": worlds,
        "phase_b_block_summary.tsv": block_summary,
        "phase_b_state_feature_outcomes.tsv": pd.DataFrame(state_outcomes),
        "phase_b_direct_edge_outcomes.tsv": pd.DataFrame(direct_outcomes),
        "phase_b_loading_pair_outcomes.tsv": pd.DataFrame(loading_outcomes),
    }
    output_hashes: dict[str, str] = {}
    for name, frame in output_frames.items():
        path = out_dir / name
        frame.to_csv(path, sep="\t", index=False)
        output_hashes[name] = _sha256(path)

    summary = {
        "schema_version": f"{SCHEMA}.summary",
        "status": "posthoc_retired_development_not_certification",
        "truth_opened_only_after_verified_phase_a_freeze": True,
        "dev_locked_or_fresh390_opened": False,
        "formal_world_predicate_source": "evaluate_layered_world_reliability",
        "component_and_combined_block_counts": block_summary.to_dict(orient="records"),
        "prior_exact_artifact_availability": {
            "state_v7": {
                "TRAIN330": "unavailable_recomputed_from_frozen_observed_membership",
                "TRAIN360": "available_and_matched",
                "TRAIN370": "available_and_matched",
                "TRAIN380": "available_and_matched",
            },
            "raw_three_tier_direct": {
                block.name: "available_and_matched" for block in BLOCKS
            },
            "current_structure_anchored_loading_v4": {
                "TRAIN330": "unavailable_recomputed_from_six_frozen_observed_views",
                "TRAIN360": "unavailable_prior_artifact_is_v2_recomputed_current_v4",
                "TRAIN370": "unavailable_prior_artifact_is_v3_recomputed_current_v4",
                "TRAIN380": "available_original_v4_and_matched_24_of_24_before_combined_replay",
            },
        },
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "phase_a_source_sha256": phase_a_manifest["source_sha256"],
        "truth_input_sha256": dict(sorted(truth_hashes.items())),
        "output_sha256": dict(sorted(output_hashes.items())),
    }
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    output_hashes[summary_path.name] = _sha256(summary_path)
    _write_json(
        out_dir / "phase_b_manifest.json",
        {
            "schema_version": f"{SCHEMA}.phase_b",
            "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
            "truth_opened_only_after_phase_a_hash_verification": True,
            "dev_locked_or_fresh390_opened": False,
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
