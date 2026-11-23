#!/usr/bin/env python3
"""Fast truth-separated replay of layered CA-BOSS v4 on cached worlds only.

Phase A randomly selects a block-balanced panel from already materialized
worlds, consumes observable artifacts only, and rebuilds the production
layered output.  By default it reuses the frozen full-pool v4.22 direct-edge
table; ``--recompute-direct`` replays the same selector from cached evidence
without generating a world.  Phase B is a separate explicit command that
opens existing scorer files only after every Phase-A hash has been verified.

This command has no code path for simulation or missing-world generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import layered_state_evidence_v4 as layered  # noqa: E402
import score_layered_state_evidence_v4 as layered_scorer  # noqa: E402
from replay_existing_world_panel_v4 import (  # noqa: E402
    ExistingWorld,
    _assert_observable,
    _canonical_sha256,
    _replay_world,
    _sha256,
    inventory_existing_worlds,
    load_excluded_world_keys,
    select_balanced_random_panel,
)


SCHEMA = "ca_boss_v4.cached_layered_random_replay.v9"
LOCAL_CANDIDATE_NAME = "v4_ls0.010_lc0.85_hp0.880_q0.10_k1"
DEFAULT_DIRECT_CACHE_ROOT = (
    SCRIPT_DIR.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_layered_contract_v423_full_pool"
)
DEFAULT_EMPIRICAL_NULL_CALIBRATOR = (
    SCRIPT_DIR.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_nested_empirical_null_v425_full_pool"
    / "phase_b_full_calibrator.json"
)
DEFAULT_MULTIEVIDENCE_CALIBRATOR = (
    SCRIPT_DIR.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_nested_multievidence_v426_full_pool"
    / "phase_b_full_calibrator.json"
)
DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR = (
    SCRIPT_DIR.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_nested_topology_consensus_v427_full_pool"
    / "phase_b_full_calibrator.json"
)
PHASE_A_TABLES = (
    "phase_a_feature_evidence.tsv",
    "phase_a_typed_hyperedges.tsv",
    "phase_a_direct_candidates.tsv",
    "phase_a_world_diagnostics.tsv",
)
SOURCE_FILES = (
    "replay_cached_layered_state_v4.py",
    "layered_state_evidence_v4.py",
    "state_equivalence_membership_v4.py",
    "shared_state_subspace_v4.py",
    "score_layered_state_evidence_v4.py",
    "score_shared_state_subspace_v4.py",
    "replay_existing_world_panel_v4.py",
    "crossfit_hierarchical_collider_v4.py",
    "empirical_null_state_calibration_v4.py",
    "loading_group_state_guard_v4.py",
    "loading_group_structure_uncertainty_guard_v4.py",
    "loading_group_competing_parent_observability_guard_v4.py",
    "loading_group_structure_equivalence_v4.py",
    "multievidence_empirical_null_state_calibration_v4.py",
    "topology_consensus_state_calibration_v4.py",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def _load_empirical_null_calibrator(
    path: Path,
) -> tuple[
    layered.empirical_null_state.FrozenEmpiricalNullStateCalibrator,
    dict[str, Any],
]:
    safe = layered._safe_observable_path(path, "empirical_null_calibrator")
    payload = layered._read_regular_bytes_once(
        safe, "empirical_null_calibrator"
    )
    calibrator, metadata = layered.load_empirical_null_state_calibrator(payload)
    binding = {
        **metadata,
        "path": str(safe),
        "artifact_file_sha256": layered._bytes_sha256(payload),
    }
    return calibrator, binding


def _load_multievidence_calibrator(
    path: Path,
) -> tuple[
    layered.multievidence_state.FrozenMultievidenceStateCalibrator,
    dict[str, Any],
]:
    safe = layered._safe_observable_path(path, "multievidence_calibrator")
    payload = layered._read_regular_bytes_once(
        safe, "multievidence_calibrator"
    )
    calibrator, metadata = layered.load_multievidence_state_calibrator(payload)
    binding = {
        **metadata,
        "path": str(safe),
        "artifact_file_sha256": layered._bytes_sha256(payload),
    }
    return calibrator, binding


def _load_topology_consensus_calibrator(
    path: Path,
) -> tuple[
    layered.topology_consensus_state.FrozenTopologyConsensusStateCalibrator,
    dict[str, Any],
]:
    safe = layered._safe_observable_path(path, "topology_consensus_calibrator")
    payload = layered._read_regular_bytes_once(
        safe, "topology_consensus_calibrator"
    )
    calibrator, metadata = layered.load_topology_consensus_state_calibrator(
        payload
    )
    binding = {
        **metadata,
        "path": str(safe),
        "artifact_file_sha256": layered._bytes_sha256(payload),
    }
    return calibrator, binding


def _layered_input_paths(world: ExistingWorld) -> dict[str, Path]:
    local = world.artifact_root / LOCAL_CANDIDATE_NAME
    paths = {
        "membership": (
            world.artifact_root
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        ),
        "subspace_hyperedges": (
            world.artifact_root
            / "shared_state_subspace_v4"
            / "typed_subspace_hyperedges.tsv"
        ),
        "loading_groups": (
            local / "loading_group_projection" / "loading_groups.tsv"
        ),
        "loading_group_membership": (
            local
            / "loading_group_projection"
            / "loading_group_membership.tsv"
        ),
        "group_axis_evidence": (
            local
            / "latent_axis_projection"
            / "group_axis_evidence.tsv"
        ),
        "axis_candidates": (
            local / "latent_axis_projection" / "axis_candidates.tsv"
        ),
        "structure_anchor_pair_evidence": (
            local
            / "structure_anchored_loading_groups"
            / "structure_anchor_pair_evidence.tsv"
        ),
    }
    return {
        label: _assert_observable(path) for label, path in paths.items()
    }


def _eligible_layered_worlds(
    runs_root: Path, worlds_root: Path
) -> tuple[ExistingWorld, ...]:
    eligible: list[ExistingWorld] = []
    for world in inventory_existing_worlds(runs_root, worlds_root):
        try:
            paths = _layered_input_paths(world)
            columns = set(
                pd.read_csv(
                    paths["membership"], sep="\t", nrows=0
                ).columns
            )
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if layered._MEMBERSHIP_REQUIRED.issubset(columns):
            eligible.append(world)
    return tuple(sorted(eligible, key=lambda item: (item.block, item.seed)))


def _selected_from_frozen_pretruth_file(
    selection_path: Path,
    eligible: Sequence[ExistingWorld],
) -> tuple[tuple[ExistingWorld, ...], dict[str, Any]]:
    """Resolve an exact self-hashed selection against the current cache."""

    safe = _assert_observable(selection_path)
    payload = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen selection must be a JSON object")
    observed = str(payload.get("selection_payload_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("selection_payload_sha256", None)
    if not observed or observed != _canonical_sha256(unsigned):
        raise ValueError("Frozen selection payload hash mismatch")
    if payload.get("scorer_or_truth_opened") is not False:
        raise ValueError("Frozen selection does not certify the pre-truth boundary")
    if payload.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Frozen selection unexpectedly created data")
    records = payload.get("selected_worlds")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen selection contains no selected worlds")
    keys = [(str(row["block"]), int(row["seed"])) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen selection contains duplicate worlds")
    lookup = {(world.block, world.seed): world for world in eligible}
    if missing := sorted(set(keys).difference(lookup)):
        raise ValueError(f"Frozen selection is not layered-cache eligible: {missing}")
    selected = tuple(lookup[key] for key in keys)
    binding = {
        "path": str(safe),
        "artifact_file_sha256": _sha256(safe),
        "selection_payload_sha256": observed,
        "schema_version": str(payload.get("schema_version", "")),
        "selection_seed": int(payload["selection_seed"]),
    }
    return selected, binding


def _upgrade_cached_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """Replay the current subtractive strong-core tier on older cache schemas."""

    output = membership.copy()
    required = set(layered_scorer.subspace_scorer.MEMBERSHIP_COLUMNS).difference(
        {
            "within_assay_strong_core_member",
            "within_assay_strong_core_count",
            "within_assay_weak_maxT_tail_demoted",
        }
    )
    missing = sorted(required.difference(output.columns))
    if missing:
        raise ValueError(f"Cached membership cannot be upgraded; missing {missing}")
    status = output["selection_status"].fillna("").astype(str)
    reason = output["abstention_reason"].fillna("").astype(str)
    pre_guard_selected = status.eq("selected")
    if "within_assay_weak_maxT_tail_demoted" in output:
        prior_weak_tail = output[
            "within_assay_weak_maxT_tail_demoted"
        ].map(lambda value: str(value).strip().casefold() in {"true", "1"})
        pre_guard_selected |= prior_weak_tail & reason.eq(
            "within_assay_strong_maxT_core_weak_tail"
        )
    p_values = pd.to_numeric(output["feature_global_maxT_p"], errors="raise")
    config = layered.state_subspace.SharedStateSubspaceConfig()
    strong = pre_guard_selected & p_values.le(
        config.within_assay_strong_core_maxT_p
    )
    counts = (
        pd.DataFrame(
            {
                "modality": output["modality"].astype(str),
                "strong": strong.astype(int),
            }
        )
        .groupby("modality", sort=True)["strong"]
        .transform("sum")
        .astype(int)
    )
    weak_tail = (
        pre_guard_selected
        & counts.ge(config.within_assay_strong_core_minimum_members)
        & p_values.gt(config.within_assay_strong_core_maxT_p)
    )
    output.loc[pre_guard_selected, "selection_status"] = "selected"
    output.loc[pre_guard_selected, "abstention_reason"] = ""
    output.loc[weak_tail, "selection_status"] = "unresolved"
    output.loc[
        weak_tail, "abstention_reason"
    ] = "within_assay_strong_maxT_core_weak_tail"
    output["within_assay_strong_core_member"] = strong
    output["within_assay_strong_core_count"] = counts
    output["within_assay_weak_maxT_tail_demoted"] = weak_tail
    output["subspace_id"] = output["selection_status"].astype(str).map(
        lambda status: "shared_latent_state_subspace_001"
        if status == "selected"
        else ""
    )
    canonical = list(layered_scorer.subspace_scorer.MEMBERSHIP_COLUMNS)
    return output.loc[:, canonical]


def _load_direct_cache(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_root = root.expanduser().resolve(strict=True)
    table_path = cache_root / "phase_a_direct_predictions.tsv"
    manifest_path = cache_root / "phase_a_manifest.json"
    if (
        table_path.is_symlink()
        or manifest_path.is_symlink()
        or not table_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ValueError("Direct replay cache is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("output_sha256", {}).get(table_path.name)
    if expected_hash != _sha256(table_path):
        raise ValueError("Direct replay cache table differs from its manifest")
    if (
        manifest.get("scorer_or_truth_opened") is not False
        or manifest.get("new_worlds_or_datasets_created") is not False
        or manifest.get("all_observable_inputs_loaded_before_truth") is not True
    ):
        raise ValueError("Direct replay cache is not a truth-closed cached run")
    table = pd.read_csv(table_path, sep="\t", keep_default_na=False)
    required = {"block", "seed", "source_assay", "parent", "child"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Direct replay cache is missing columns {missing}")
    if table.duplicated(["block", "seed", "parent", "child"]).any():
        raise ValueError("Direct replay cache contains duplicate edges")
    provenance = {
        "root": str(cache_root),
        "table_sha256": _sha256(table_path),
        "manifest_sha256": _sha256(manifest_path),
        "algorithm_schema_version": manifest.get("algorithm_schema_version"),
        "direct_selector_schema_version": manifest.get(
            "direct_selector_schema_version"
        ),
    }
    return table, provenance


def _cached_direct_consensus(
    cache: pd.DataFrame, world: ExistingWorld
) -> pd.DataFrame:
    rows = cache.loc[
        cache["block"].astype(str).eq(world.block)
        & pd.to_numeric(cache["seed"], errors="raise").eq(world.seed)
    ].copy()
    if rows.empty:
        raise ValueError(
            f"Direct replay cache has no rows for {world.block}:{world.seed}"
        )
    return pd.DataFrame(
        {
            "node1": rows["parent"].astype(str),
            "endpoint1": "TAIL",
            "node2": rows["child"].astype(str),
            "endpoint2": "ARROW",
            "edge_type": "within_assay_direct_candidate",
            "source_assay": rows["source_assay"].astype(str),
            "orientation_source": "cached_v4.22_crossfit_hierarchical_collider",
        }
    ).reset_index(drop=True)


def _recomputed_direct_consensus(world: ExistingWorld) -> tuple[pd.DataFrame, dict[str, str]]:
    tables, hashes, _ = _replay_world(world)
    direct = tables["direct"].drop(columns=["block", "seed"])
    return direct, hashes


def _tag(frame: pd.DataFrame, world: ExistingWorld) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "seed", world.seed)
    tagged.insert(0, "block", world.block)
    return tagged


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    values = list(frames)
    if not values:
        return pd.DataFrame(columns=["block", "seed"])
    return pd.concat(values, ignore_index=True, sort=False)


def _direct_tuples(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    return {
        (
            str(row.node1),
            str(row.endpoint1).upper(),
            str(row.node2),
            str(row.endpoint2).upper(),
        )
        for row in frame.itertuples(index=False)
        if str(row.edge_type) == "within_assay_direct_candidate"
    }


def _world_diagnostics(
    world: ExistingWorld,
    result: layered.LayeredStateEvidenceResult,
    direct_input: pd.DataFrame,
) -> dict[str, Any]:
    features = result.feature_evidence
    equivalence_groups = result.typed_hyperedges.loc[
        result.typed_hyperedges["edge_type"].astype(str).eq(
            "within_assay_shared_state_equivalence_class"
        )
    ]
    if _direct_tuples(result.within_assay_direct_candidates) != _direct_tuples(
        direct_input
    ):
        raise ValueError("Layering changed the direct candidate set")
    if not features["strict_state_member"].equals(
        features["confirmed_depth_orthogonal_state_response"]
    ):
        raise ValueError("Layering changed strict state membership")
    if not features["empirical_null_calibration_active"].astype(bool).all():
        raise ValueError("Cached v4.25 replay did not activate its calibrator")
    expected_calibrated = (
        features["state_supported_member"].astype(bool)
        | features["empirical_null_state_member"].astype(bool)
    )
    if not features["state_calibrated_supported_member"].astype(bool).equals(
        expected_calibrated
    ):
        raise ValueError("Empirical-null tier is not an additive state union")
    if (
        features["empirical_null_state_member"].astype(bool)
        & features["state_supported_member"].astype(bool)
    ).any():
        raise ValueError("Empirical-null tier overlaps existing state membership")
    if not features["multievidence_calibration_active"].astype(bool).all():
        raise ValueError("Cached v4.26 replay did not activate its calibrator")
    multievidence_addition = features[
        "multievidence_empirical_null_state_member"
    ].astype(bool)
    expected_v426 = expected_calibrated | multievidence_addition
    if not features["state_v426_supported_member"].astype(bool).equals(
        expected_v426
    ):
        raise ValueError("v4.26 tier is not the exact stacked state union")
    if (
        multievidence_addition
        & ~features["multievidence_state_specific_gate"].astype(bool)
    ).any():
        raise ValueError("v4.26 tier bypassed its state-specific gate")
    if (
        multievidence_addition
        & features["state_supported_member"].astype(bool)
    ).any():
        raise ValueError("v4.26 tier overlaps existing v4.24 membership")
    if not features["topology_consensus_calibration_active"].astype(bool).all():
        raise ValueError("Cached v4.27 replay did not activate its calibrator")
    topology_addition = features["topology_consensus_state_member"].astype(bool)
    expected_v427 = expected_v426 | topology_addition
    if not features["state_v427_supported_member"].astype(bool).equals(
        expected_v427
    ):
        raise ValueError("v4.27 tier is not the exact stacked state union")
    if (
        topology_addition
        & ~features["topology_consensus_state_specific_gate"].astype(bool)
    ).any():
        raise ValueError("v4.27 tier bypassed its state-specific gate")
    if (
        topology_addition
        & ~features[
            "topology_consensus_residual_structure_support"
        ].astype(bool)
    ).any():
        raise ValueError("v4.27 tier bypassed residual-topology support")
    if (
        topology_addition
        & ~features["within_assay_direct_candidate_overlap"].astype(bool)
    ).any():
        raise ValueError("v4.27 residual support is not a direct endpoint")
    if (topology_addition & expected_v426).any():
        raise ValueError("v4.27 tier overlaps upstream v4.26 membership")
    state_v427_members = set(
        features.loc[
            features["state_v427_supported_member"].astype(bool), "feature"
        ].astype(str)
    )
    loading_rows = result.typed_hyperedges.loc[
        result.typed_hyperedges["relation_id"].astype(str).str.startswith(
            "assay_local__"
        )
    ]
    retained_loading_rows = loading_rows.loc[
        loading_rows["selection_status"].astype(str).eq("selected")
    ]
    rejected_loading_rows = loading_rows.loc[
        loading_rows["selection_status"].astype(str).eq("unresolved")
    ]
    v429_state_route_groups = 0
    v429_axis_route_groups = 0
    v430_structure_downgrades = 0
    v430_uncertainty_downgrades = 0
    v431_competitor_observability_downgrades = 0
    v431_comparison_unavailable_downgrades = 0
    v432_structure_equivalence_hyperedges = 0
    v432_exact_loading_groups = 0
    v432_candidate_alternatives = 0
    feature_lookup = features.set_index("feature", drop=False)
    loading_guard_safe = True
    for row in loading_rows.itertuples(index=False):
        hyperedge_members = {
            member.strip()
            for member in str(row.members).split(";")
            if member.strip()
        }
        members = {
            member.strip()
            for member in str(row.nominal_members).split(";")
            if member.strip()
        }
        if len(members) != 2:
            raise ValueError("v4.32 nominal loading group must have two members")
        state_subset = members.issubset(state_v427_members)
        has_state_member = bool(members & state_v427_members)
        route = str(row.loading_group_guard_route)
        upstream_route = str(row.loading_group_v429_guard_route)
        upstream_retained = bool(row.loading_group_v429_retained)
        v430_route = str(row.loading_group_v430_guard_route)
        v430_retained = bool(row.loading_group_v430_retained)
        selected = str(row.selection_status) == "selected"
        if upstream_route == layered.loading_group_state_guard.STATE_ROUTE:
            v429_state_route_groups += 1
            upstream_safe = bool(
                upstream_retained
                and state_subset
                and not bool(row.cross_assay_axis_id)
                and not bool(row.cross_assay_axis_modalities)
            )
        elif upstream_route == layered.loading_group_state_guard.AXIS_ROUTE:
            v429_axis_route_groups += 1
            axis_modalities = {
                value.strip()
                for value in str(row.cross_assay_axis_modalities).split(";")
                if value.strip()
            }
            upstream_safe = bool(
                upstream_retained
                and not state_subset
                and has_state_member
                and str(row.cross_assay_axis_id)
                and int(row.cross_assay_axis_n_groups) >= 2
                and int(row.cross_assay_axis_n_modalities) >= 2
                and len(axis_modalities) >= 2
                and str(row.modalities) in axis_modalities
                and float(row.group_axis_view_support) > 0.0
                and float(row.group_axis_lower_cosine) > 0.0
                and float(row.group_axis_relative_margin) > 0.0
            )
        else:
            upstream_safe = bool(
                not upstream_retained
                and upstream_route == "unresolved"
                and not state_subset
            )
        passes_floor = bool(row.passes_loading_uncertainty_floor)
        direct_overlap = bool(row.complete_direct_collider_overlap)
        expected_v430_selected = bool(
            upstream_retained
            and not direct_overlap
            and (
                upstream_route == layered.loading_group_state_guard.AXIS_ROUTE
                or (
                    upstream_route == layered.loading_group_state_guard.STATE_ROUTE
                    and passes_floor
                )
            )
        )
        if upstream_retained and not expected_v430_selected:
            v430_structure_downgrades += int(direct_overlap)
            v430_uncertainty_downgrades += int(
                not direct_overlap
                and upstream_route == layered.loading_group_state_guard.STATE_ROUTE
                and not passes_floor
            )
        comparison_required = bool(
            v430_retained
            and v430_route == layered.loading_group_state_guard.STATE_ROUTE
        )
        comparison_available = bool(
            row.complete_two_parent_comparison_available
        )
        selected_parent = str(row.selected_parent_candidate)
        sibling = str(row.noncollider_sibling_candidate)
        competitor_name = str(row.competing_parent_candidate)
        competitor_fixed = False
        comparison_safe = not comparison_available
        if comparison_available and competitor_name in feature_lookup.index:
            competitor = feature_lookup.loc[competitor_name]
            competitor_fixed = bool(
                competitor["topology_consensus_fixed_rule_eligible"]
            )
            comparison_safe = bool(
                {selected_parent, sibling} == members
                and str(competitor["modality"]) == str(row.modalities)
                and bool(row.competing_parent_fixed_rule_eligible)
                == competitor_fixed
            )
        passes_competitor_guard = bool(
            not comparison_required
            or (comparison_available and competitor_fixed)
        )
        expected_selected = bool(
            v430_retained and passes_competitor_guard
        )
        if v430_retained and not expected_selected:
            v431_comparison_unavailable_downgrades += int(
                not comparison_available
            )
            v431_competitor_observability_downgrades += int(
                comparison_available and not competitor_fixed
            )
        if expected_selected and v430_route == layered.loading_group_state_guard.STATE_ROUTE:
            expected_guard_status = (
                layered.loading_group_competitor_guard.STATE_RETAINED_STATUS
            )
        elif expected_selected and v430_route == layered.loading_group_state_guard.AXIS_ROUTE:
            expected_guard_status = (
                layered.loading_group_competitor_guard.AXIS_RETAINED_STATUS
            )
        elif not v430_retained:
            expected_guard_status = (
                layered.loading_group_competitor_guard.UPSTREAM_REJECTED_STATUS
            )
        elif not comparison_available:
            expected_guard_status = (
                layered.loading_group_competitor_guard.COMPARISON_UNAVAILABLE_STATUS
            )
        else:
            expected_guard_status = (
                layered.loading_group_competitor_guard.COMPETITOR_UNOBSERVABLE_STATUS
            )
        expected_route = v430_route if expected_selected else "unresolved"
        expected_equivalence = (
            layered.loading_group_structure_equivalence
            .build_loading_group_structure_equivalence(
                pd.DataFrame(
                    [
                        {
                            "loading_group": str(row.relation_id).removeprefix(
                                "assay_local__"
                            ),
                            "modality": str(row.modalities),
                            "members": ";".join(sorted(members)),
                            "retained": expected_selected,
                            "guard_route": expected_route,
                            "guard_status": expected_guard_status,
                            "complete_two_parent_comparison_available": (
                                comparison_available
                            ),
                            "selected_parent_candidate": selected_parent,
                            "competing_parent_candidate": competitor_name,
                            "noncollider_sibling_candidate": sibling,
                            "collider_child_candidate": str(
                                row.collider_child_candidate
                            ),
                            "passes_competing_parent_observability_guard": (
                                passes_competitor_guard
                            ),
                        }
                    ]
                )
            ).decisions.iloc[0]
        )
        candidate_pairs = (
            layered.loading_group_structure_equivalence.parse_candidate_pairs(
                str(row.candidate_pairs_json)
            )
        )
        candidate_nodes = {
            node for pair in candidate_pairs for node in pair
        }
        expected_structure_equivalence = bool(
            expected_equivalence["structure_equivalence_selected"]
        )
        expected_exact_group = bool(
            expected_equivalence["exact_loading_group_selected"]
        )
        expected_hyperedge_members = (
            {
                selected_parent,
                competitor_name,
                sibling,
                str(row.collider_child_candidate),
            }
            if expected_structure_equivalence
            else members
        )
        expected_edge_type = (
            layered.loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE
            if expected_structure_equivalence
            else "within_assay_shared_state_support"
        )
        structure_equivalence_safe = bool(
            str(row.edge_type) == expected_edge_type
            and hyperedge_members == expected_hyperedge_members
            and (
                (expected_selected and candidate_nodes == expected_hyperedge_members)
                or (not expected_selected and not candidate_pairs)
            )
        )
        structure_equivalence_safe = bool(
            structure_equivalence_safe
            and str(row.loading_group_v431_guard_route) == expected_route
            and str(row.loading_group_v431_guard_status)
            == expected_guard_status
            and bool(row.loading_group_v431_retained) == expected_selected
            and str(row.structure_equivalence_schema)
            == layered.loading_group_structure_equivalence.SCHEMA
            and str(row.structure_equivalence_fixed_rule)
            == layered.loading_group_structure_equivalence.FIXED_RULE
            and str(row.structure_equivalence_hypothesis_type)
            == str(expected_equivalence["hypothesis_type"])
            and str(row.structure_equivalence_hypothesis_status)
            == str(expected_equivalence["hypothesis_status"])
            and str(row.nominal_members) == ";".join(sorted(members))
            and str(row.parent_equivalence_members)
            == str(expected_equivalence["parent_equivalence_members"])
            and str(row.counterpart_equivalence_members)
            == str(expected_equivalence["counterpart_equivalence_members"])
            and str(row.candidate_pairs_json)
            == str(expected_equivalence["candidate_pairs_json"])
            and int(row.candidate_pair_count)
            == int(expected_equivalence["candidate_pair_count"])
            and bool(row.exact_loading_group_selected) == expected_exact_group
            and bool(row.structure_equivalence_selected)
            == expected_structure_equivalence
            and bool(row.exact_pair_identified)
            == bool(expected_equivalence["exact_pair_identified"])
            and bool(row.shared_state_hypothesis_supported)
            == expected_selected
            and not bool(row.candidate_pairs_may_be_expanded_as_edges)
        )
        v432_structure_equivalence_hyperedges += int(
            expected_structure_equivalence
        )
        v432_exact_loading_groups += int(expected_exact_group)
        v432_candidate_alternatives += len(candidate_pairs)
        route_safe = bool(
            upstream_safe
            and v430_retained == expected_v430_selected
            and v430_route
            == (upstream_route if expected_v430_selected else "unresolved")
            and bool(row.competing_parent_observability_required)
            == comparison_required
            and comparison_safe
            and bool(row.passes_competing_parent_observability_guard)
            == passes_competitor_guard
            and selected == expected_selected
            and route == (v430_route if selected else "unresolved")
            and np.isclose(
                float(row.minimum_state_route_member_centroid_cosine),
                layered.loading_group_structure_guard
                .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE,
            )
            and structure_equivalence_safe
        )
        loading_guard_safe &= route_safe
    if not loading_guard_safe:
        raise ValueError(
            "v4.32 loading/structure output violates its frozen routes"
        )
    v430_retained_loading_rows = loading_rows.loc[
        loading_rows["loading_group_v430_retained"].astype(bool)
    ]
    retained_loading_members = {
        member.strip()
        for members in retained_loading_rows["nominal_members"].astype(str)
        for member in members.split(";")
        if member.strip()
    }
    axis_only_loading_members = retained_loading_members.difference(
        state_v427_members
    )
    if (
        features["strict_membership_changed"].astype(bool).any()
        or features["exact_cross_assay_pair_claimed"].astype(bool).any()
        or features["direct_causal_claimed"].astype(bool).any()
        or equivalence_groups["pairwise_expansion_permitted"].astype(bool).any()
        or equivalence_groups["exact_cross_assay_pair_claimed"].astype(bool).any()
        or equivalence_groups["direct_causal_claimed"].astype(bool).any()
    ):
        raise ValueError("State-equivalence layer emitted an unsupported claim")
    return {
        "block": world.block,
        "seed": world.seed,
        "status": "ok",
        "n_features": len(features),
        "n_strict_state_members": int(features["strict_state_member"].sum()),
        "n_equivalence_only_members": int(
            features["state_equivalence_only_member"].sum()
        ),
        "n_state_supported_members": int(features["state_supported_member"].sum()),
        "n_empirical_null_state_members": int(
            features["empirical_null_state_member"].sum()
        ),
        "n_state_calibrated_supported_members": int(
            features["state_calibrated_supported_member"].sum()
        ),
        "n_multievidence_state_members": int(
            features["multievidence_empirical_null_state_member"].sum()
        ),
        "n_multievidence_incremental_beyond_v425": int(
            (
                features["multievidence_empirical_null_state_member"].astype(bool)
                & ~features["state_calibrated_supported_member"].astype(bool)
            ).sum()
        ),
        "n_state_v426_supported_members": int(
            features["state_v426_supported_member"].sum()
        ),
        "n_topology_consensus_state_members": int(topology_addition.sum()),
        "n_topology_consensus_incremental_beyond_v426": int(
            (topology_addition & ~expected_v426).sum()
        ),
        "n_state_v427_supported_members": int(
            features["state_v427_supported_member"].sum()
        ),
        "n_pre_v429_selected_loading_groups": len(loading_rows),
        "n_v429_retained_loading_groups": int(
            loading_rows["loading_group_v429_retained"].astype(bool).sum()
        ),
        "n_v429_state_route_loading_groups": v429_state_route_groups,
        "n_v429_axis_route_loading_groups": v429_axis_route_groups,
        "n_v430_retained_loading_groups": len(v430_retained_loading_rows),
        "n_v430_rejected_loading_groups": (
            len(loading_rows) - len(v430_retained_loading_rows)
        ),
        "n_v430_structure_overlap_downgrades": v430_structure_downgrades,
        "n_v430_uncertainty_floor_downgrades": v430_uncertainty_downgrades,
        "n_v431_retained_loading_groups": len(retained_loading_rows),
        "n_v431_rejected_loading_groups": len(rejected_loading_rows),
        "n_v431_competing_parent_observability_downgrades": (
            v431_competitor_observability_downgrades
        ),
        "n_v431_comparison_unavailable_downgrades": (
            v431_comparison_unavailable_downgrades
        ),
        "n_v431_retained_loading_members": len(retained_loading_members),
        "n_v431_axis_only_loading_members": len(axis_only_loading_members),
        "n_v432_structure_equivalence_hyperedges": (
            v432_structure_equivalence_hyperedges
        ),
        "n_v432_exact_assay_local_loading_groups": v432_exact_loading_groups,
        "n_v432_candidate_pair_alternatives": v432_candidate_alternatives,
        "loading_group_state_guard_matches_fixed_rule": True,
        "loading_group_structure_uncertainty_guard_matches_fixed_rule": True,
        "loading_group_competing_parent_observability_guard_matches_fixed_rule": True,
        "loading_group_structure_equivalence_matches_fixed_rule": True,
        "loading_group_state_route_exact_pair_identified": False,
        "loading_group_candidate_pairs_are_joint_alternatives": True,
        "loading_group_candidate_pairs_expanded_as_edges": False,
        "loading_group_structure_equivalence_changes_v431_selection": False,
        "loading_group_axis_route_changes_state_membership": False,
        "loading_group_direct_structure_alone_is_insufficient": True,
        "n_state_equivalence_sets": len(equivalence_groups),
        "n_state_equivalence_pairwise_relations": 0,
        "n_direct_candidates": len(result.within_assay_direct_candidates),
        "direct_candidates_unchanged_by_layering": True,
        "strict_membership_unchanged_by_equivalence": True,
        "unsupported_equivalence_claims": 0,
        "empirical_null_pairwise_relations": 0,
        "empirical_null_direct_relations_created": 0,
        "multievidence_pairwise_relations": 0,
        "multievidence_direct_relations_created": 0,
        "topology_consensus_pairwise_relations": 0,
        "topology_consensus_direct_relations_created": 0,
        "topology_consensus_direct_structure_alone_is_insufficient": True,
    }


def run_phase_a(
    *,
    out_dir: Path,
    runs_root: Path,
    worlds_root: Path,
    panel_size: int,
    selection_seed: int,
    direct_cache_root: Path | None,
    recompute_direct: bool,
    empirical_null_calibrator_path: Path,
    multievidence_calibrator_path: Path,
    topology_consensus_calibrator_path: Path,
    selection_file: Path | None = None,
    exclude_selections: Sequence[Path] = (),
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Phase-A output already exists: {out_dir}")
    inventory = _eligible_layered_worlds(runs_root, worlds_root)
    if selection_file is not None and exclude_selections:
        raise ValueError(
            "Use either --selection-file or --exclude-selection, not both"
        )
    excluded, exclusion_hashes = load_excluded_world_keys(
        exclude_selections
    )
    pool = tuple(
        world
        for world in inventory
        if (world.block, world.seed) not in excluded
    )
    selection_binding: dict[str, Any] | None = None
    if selection_file is None:
        selected = select_balanced_random_panel(
            pool, panel_size, selection_seed
        )
        effective_selection_seed = int(selection_seed)
        sampling_method = (
            "random_block_balanced_without_replacement_after_pretruth_"
            "selection_exclusion"
            if excluded
            else "random_block_balanced_without_replacement"
        )
    else:
        selected, selection_binding = _selected_from_frozen_pretruth_file(
            selection_file, pool
        )
        if len(selected) != panel_size:
            raise ValueError(
                "--panel-size must match the frozen selection world count"
            )
        effective_selection_seed = int(selection_binding["selection_seed"])
        sampling_method = "exact_self_hashed_pretruth_selection_replay"
    empirical_null_calibrator, empirical_null_binding = (
        _load_empirical_null_calibrator(empirical_null_calibrator_path)
    )
    multievidence_calibrator, multievidence_binding = (
        _load_multievidence_calibrator(multievidence_calibrator_path)
    )
    topology_consensus_calibrator, topology_consensus_binding = (
        _load_topology_consensus_calibrator(
            topology_consensus_calibrator_path
        )
    )
    direct_cache: pd.DataFrame | None = None
    direct_cache_provenance: dict[str, Any] | None = None
    if not recompute_direct:
        if direct_cache_root is None:
            raise ValueError("A direct cache is required unless --recompute-direct is set")
        direct_cache, direct_cache_provenance = _load_direct_cache(
            direct_cache_root
        )
    out_dir.mkdir(parents=True)
    selection = {
        "schema_version": SCHEMA,
        "selection_seed": effective_selection_seed,
        "sampling_method": sampling_method,
        "source_pretruth_selection": selection_binding,
        "inventory_pool_size": len(inventory),
        "eligible_pool_size": len(pool),
        "excluded_world_count": len(inventory) - len(pool),
        "excluded_world_keys": [
            {"block": block, "seed": seed}
            for block, seed in sorted(excluded)
        ],
        "exclusion_selection_sha256": exclusion_hashes,
        "eligible_block_counts": {
            block: sum(world.block == block for world in pool)
            for block in sorted({world.block for world in pool})
        },
        "panel_size": len(selected),
        "selected_worlds": [world.payload() for world in selected],
        "direct_source": "recomputed_cached_evidence"
        if recompute_direct
        else "frozen_v4.22_full_pool_table",
        "empirical_null_calibrator_artifact_sha256": (
            empirical_null_binding["artifact_file_sha256"]
        ),
        "empirical_null_calibrator_payload_sha256": (
            empirical_null_binding["artifact_payload_sha256"]
        ),
        "multievidence_calibrator_artifact_sha256": (
            multievidence_binding["artifact_file_sha256"]
        ),
        "multievidence_calibrator_payload_sha256": (
            multievidence_binding["artifact_payload_sha256"]
        ),
        "topology_consensus_calibrator_artifact_sha256": (
            topology_consensus_binding["artifact_file_sha256"]
        ),
        "topology_consensus_calibrator_payload_sha256": (
            topology_consensus_binding["artifact_payload_sha256"]
        ),
        "new_worlds_or_datasets_created": False,
        "scorer_or_truth_opened": False,
        "cached_inputs_only": True,
        "retrospective_development_only": True,
    }
    selection["selection_payload_sha256"] = _canonical_sha256(selection)
    _write_json(out_dir / "phase_a_selection.json", selection)

    feature_frames: list[pd.DataFrame] = []
    hyperedge_frames: list[pd.DataFrame] = []
    direct_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for world in selected:
        key = f"{world.block}:{world.seed}"
        try:
            paths = _layered_input_paths(world)
            hashes = {label: _sha256(path) for label, path in paths.items()}
            if recompute_direct:
                direct_input, direct_hashes = _recomputed_direct_consensus(world)
                hashes.update(
                    {f"direct::{name}": digest for name, digest in direct_hashes.items()}
                )
            else:
                assert direct_cache is not None
                direct_input = _cached_direct_consensus(direct_cache, world)
            result = layered.build_layered_state_evidence(
                _upgrade_cached_membership(
                    pd.read_csv(
                        paths["membership"], sep="\t", keep_default_na=False
                    )
                ),
                pd.read_csv(
                    paths["subspace_hyperedges"], sep="\t", keep_default_na=False
                ),
                direct_input,
                pd.read_csv(paths["loading_groups"], sep="\t", keep_default_na=False),
                pd.read_csv(
                    paths["loading_group_membership"],
                    sep="\t",
                    keep_default_na=False,
                ),
                pd.read_csv(
                    paths["structure_anchor_pair_evidence"],
                    sep="\t",
                    keep_default_na=False,
                ),
                empirical_null_calibrator=empirical_null_calibrator,
                multievidence_calibrator=multievidence_calibrator,
                topology_consensus_calibrator=(
                    topology_consensus_calibrator
                ),
                group_axis_evidence=pd.read_csv(
                    paths["group_axis_evidence"],
                    sep="\t",
                    keep_default_na=False,
                ),
                axis_candidates=pd.read_csv(
                    paths["axis_candidates"],
                    sep="\t",
                    keep_default_na=False,
                ),
            )
            diagnostics = _world_diagnostics(world, result, direct_input)
        except Exception as exc:
            attempts.append(
                {
                    "block": world.block,
                    "seed": world.seed,
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            diagnostic_rows.append(
                {
                    "block": world.block,
                    "seed": world.seed,
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        input_hashes[key] = hashes
        feature_frames.append(_tag(result.feature_evidence, world))
        hyperedge_frames.append(_tag(result.typed_hyperedges, world))
        direct_frames.append(_tag(result.within_assay_direct_candidates, world))
        diagnostic_rows.append(diagnostics)
        attempts.append({"block": world.block, "seed": world.seed, "status": "ok"})

    outputs = {
        "phase_a_feature_evidence.tsv": _concat(feature_frames),
        "phase_a_typed_hyperedges.tsv": _concat(hyperedge_frames),
        "phase_a_direct_candidates.tsv": _concat(direct_frames),
        "phase_a_world_diagnostics.tsv": pd.DataFrame(diagnostic_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    _write_json(out_dir / "phase_a_attempts.json", attempts)
    manifest = {
        "schema_version": SCHEMA,
        "phase": "a",
        "algorithm_schema_version": "ca_boss_v4.layered_contract.v4.32",
        "layered_provenance_schema_version": layered.PROVENANCE_SCHEMA_VERSION,
        "state_equivalence_schema_version": layered.state_equivalence.SCHEMA,
        "empirical_null_state_schema_version": (
            layered.empirical_null_state.SCHEMA
        ),
        "empirical_null_calibrator": empirical_null_binding,
        "multievidence_state_schema_version": (
            layered.multievidence_state.SCHEMA
        ),
        "multievidence_calibrator": multievidence_binding,
        "topology_consensus_state_schema_version": (
            layered.topology_consensus_state.SCHEMA
        ),
        "loading_group_state_guard_schema_version": (
            layered.loading_group_state_guard.SCHEMA
        ),
        "loading_group_structure_uncertainty_guard_schema_version": (
            layered.loading_group_structure_guard.SCHEMA
        ),
        "loading_group_competing_parent_observability_guard_schema_version": (
            layered.loading_group_competitor_guard.SCHEMA
        ),
        "loading_group_structure_equivalence_schema_version": (
            layered.loading_group_structure_equivalence.SCHEMA
        ),
        "loading_group_structure_equivalence_fixed_rule": (
            layered.loading_group_structure_equivalence.FIXED_RULE
        ),
        "loading_group_state_route_candidate_pair_count": 4,
        "loading_group_state_route_exact_pair_identified": False,
        "loading_group_axis_route_candidate_pair_count": 1,
        "loading_group_candidate_pairs_may_be_expanded_as_edges": False,
        "loading_group_structure_equivalence_changes_v431_selection": False,
        "loading_group_structure_equivalence_application_requires_truth": False,
        "topology_consensus_calibrator": topology_consensus_binding,
        "selection_sha256": _sha256(out_dir / "phase_a_selection.json"),
        "attempts_sha256": _sha256(out_dir / "phase_a_attempts.json"),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_A_TABLES
        },
        "source_sha256": _source_hashes(),
        "observable_input_sha256": input_hashes,
        "direct_cache_provenance": direct_cache_provenance,
        "selected_world_count": len(selected),
        "successful_world_count": sum(row["status"] == "ok" for row in attempts),
        "all_selected_worlds_attempted": len(attempts) == len(selected),
        "all_observable_inputs_loaded_before_truth": True,
        "scorer_or_truth_opened": False,
        "new_worlds_or_datasets_created": False,
        "cached_inputs_only": True,
        "retrospective_development_only": True,
        "calibrator_application_on_training_pool_is_not_independent_validation": True,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, pd.DataFrame]]:
    manifest_path = out_dir / "phase_a_manifest.json"
    selection_path = out_dir / "phase_a_selection.json"
    attempts_path = out_dir / "phase_a_attempts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    observed_payload_hash = unsigned.pop("manifest_payload_sha256", None)
    if observed_payload_hash != _canonical_sha256(unsigned):
        raise ValueError("Phase-A manifest payload changed")
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("algorithm_schema_version")
        != "ca_boss_v4.layered_contract.v4.32"
        or manifest.get("layered_provenance_schema_version")
        != layered.PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("Phase A uses another layered v4.32 contract")
    if (
        manifest.get("selection_sha256") != _sha256(selection_path)
        or manifest.get("attempts_sha256") != _sha256(attempts_path)
        or manifest.get("source_sha256") != _source_hashes()
    ):
        raise ValueError("Phase-A selection, attempts, or source changed")
    calibrator_binding = manifest.get("empirical_null_calibrator")
    if not isinstance(calibrator_binding, Mapping):
        raise ValueError("Phase A lacks an empirical-null calibrator binding")
    _, current_calibrator_binding = _load_empirical_null_calibrator(
        Path(str(calibrator_binding.get("path", "")))
    )
    if current_calibrator_binding != dict(calibrator_binding):
        raise ValueError("Phase-A empirical-null calibrator changed")
    multievidence_binding = manifest.get("multievidence_calibrator")
    if not isinstance(multievidence_binding, Mapping):
        raise ValueError("Phase A lacks a multievidence calibrator binding")
    _, current_multievidence_binding = _load_multievidence_calibrator(
        Path(str(multievidence_binding.get("path", "")))
    )
    if current_multievidence_binding != dict(multievidence_binding):
        raise ValueError("Phase-A multievidence calibrator changed")
    topology_consensus_binding = manifest.get(
        "topology_consensus_calibrator"
    )
    if not isinstance(topology_consensus_binding, Mapping):
        raise ValueError("Phase A lacks a topology-consensus calibrator binding")
    _, current_topology_consensus_binding = (
        _load_topology_consensus_calibrator(
            Path(str(topology_consensus_binding.get("path", "")))
        )
    )
    if current_topology_consensus_binding != dict(
        topology_consensus_binding
    ):
        raise ValueError("Phase-A topology-consensus calibrator changed")
    if manifest.get("loading_group_state_guard_schema_version") != (
        layered.loading_group_state_guard.SCHEMA
    ):
        raise ValueError("Phase A uses another loading-group state guard")
    if manifest.get(
        "loading_group_structure_uncertainty_guard_schema_version"
    ) != layered.loading_group_structure_guard.SCHEMA:
        raise ValueError(
            "Phase A uses another loading-group structure/uncertainty guard"
        )
    if manifest.get(
        "loading_group_competing_parent_observability_guard_schema_version"
    ) != layered.loading_group_competitor_guard.SCHEMA:
        raise ValueError(
            "Phase A uses another loading-group competing-parent guard"
        )
    if manifest.get(
        "loading_group_structure_equivalence_schema_version"
    ) != layered.loading_group_structure_equivalence.SCHEMA:
        raise ValueError(
            "Phase A uses another loading-group structure-equivalence contract"
        )
    if (
        manifest.get("loading_group_structure_equivalence_fixed_rule")
        != layered.loading_group_structure_equivalence.FIXED_RULE
        or manifest.get("loading_group_state_route_candidate_pair_count") != 4
        or manifest.get("loading_group_state_route_exact_pair_identified")
        is not False
        or manifest.get("loading_group_axis_route_candidate_pair_count") != 1
        or manifest.get(
            "loading_group_candidate_pairs_may_be_expanded_as_edges"
        )
        is not False
        or manifest.get(
            "loading_group_structure_equivalence_changes_v431_selection"
        )
        is not False
        or manifest.get(
            "loading_group_structure_equivalence_application_requires_truth"
        )
        is not False
    ):
        raise ValueError("Phase A violates v4.32 set-valued semantics")
    if (
        manifest.get("scorer_or_truth_opened") is not False
        or manifest.get("new_worlds_or_datasets_created") is not False
        or manifest.get("cached_inputs_only") is not True
    ):
        raise ValueError("Phase A does not certify cached truth-closed replay")
    tables: dict[str, pd.DataFrame] = {}
    for name in PHASE_A_TABLES:
        path = out_dir / name
        if manifest.get("output_sha256", {}).get(name) != _sha256(path):
            raise ValueError(f"Phase-A output changed: {name}")
        tables[name] = pd.read_csv(path, sep="\t", keep_default_na=False)
    return manifest, selection, attempts, tables


def run_phase_b(*, out_dir: Path) -> None:
    phase_b_outputs = (
        out_dir / "phase_b_world_metrics.tsv",
        out_dir / "phase_b_metrics.json",
        out_dir / "phase_b_summary.json",
    )
    if any(path.exists() for path in phase_b_outputs):
        raise FileExistsError("Phase-B output already exists; refusing overwrite")
    manifest, selection, attempts, tables = _load_phase_a(out_dir)
    feature_table = tables["phase_a_feature_evidence.tsv"]
    hyperedge_table = tables["phase_a_typed_hyperedges.tsv"]
    direct_table = tables["phase_a_direct_candidates.tsv"]
    attempt_map = {
        (str(row["block"]), int(row["seed"])): row for row in attempts
    }
    metrics_by_world: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for payload in selection["selected_worlds"]:
        block = str(payload["block"])
        seed = int(payload["seed"])
        key = (block, seed)
        attempt = attempt_map[key]
        if attempt["status"] != "ok":
            rows.append({"block": block, "seed": seed, "status": "invalid_phase_a"})
            continue
        block_world_root = Path(payload["world_root"]).expanduser().resolve(
            strict=True
        )
        world_root = (
            block_world_root / f"world_seed{seed}"
        ).resolve(strict=True)
        scorer_root = world_root / "scorer"
        role_path = scorer_root / "role_assignments.tsv"
        truth_path = scorer_root / "truth_typed_edges.tsv"
        parameters_path = scorer_root / "world_parameters.json"
        try:
            metrics = layered_scorer.score_layered_state_evidence(
                feature_table.loc[
                    feature_table["block"].astype(str).eq(block)
                    & pd.to_numeric(feature_table["seed"], errors="raise").eq(seed)
                ].drop(columns=["block", "seed"]),
                hyperedge_table.loc[
                    hyperedge_table["block"].astype(str).eq(block)
                    & pd.to_numeric(hyperedge_table["seed"], errors="raise").eq(seed)
                ].drop(columns=["block", "seed"]),
                direct_table.loc[
                    direct_table["block"].astype(str).eq(block)
                    & pd.to_numeric(direct_table["seed"], errors="raise").eq(seed)
                ].drop(columns=["block", "seed"]),
                pd.read_csv(role_path, sep="\t", keep_default_na=False),
                pd.read_csv(truth_path, sep="\t", keep_default_na=False),
                json.loads(parameters_path.read_text(encoding="utf-8")),
            )
        except Exception as exc:
            rows.append(
                {
                    "block": block,
                    "seed": seed,
                    "status": "invalid_phase_b",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        metrics_by_world[f"{block}:{seed}"] = metrics
        supported = metrics["set_scores"]["state_equivalence_supported"]
        calibrated = metrics["set_scores"]["state_calibrated_supported"]
        v426_supported = metrics["set_scores"]["state_v426_supported"]
        v427_supported = metrics["set_scores"]["state_v427_supported"]
        direct = metrics["direct_score"]
        equivalence = metrics["state_equivalence_layer"]
        empirical = metrics["empirical_null_state_layer"]
        multievidence = metrics["multievidence_state_layer"]
        topology_consensus = metrics["topology_consensus_state_layer"]
        loading_pair = metrics["assay_local_loading_pair_score"]
        loading_guard = metrics["assay_local_loading_group_state_guard"]
        rows.append(
            {
                "block": block,
                "seed": seed,
                "status": "ok",
                "state_supported_precision": supported["precision"],
                "state_supported_recall": supported["recall"],
                "state_supported_sentinels": supported["n_sentinel_selected"],
                "state_calibrated_precision": calibrated["precision"],
                "state_calibrated_recall": calibrated["recall"],
                "state_calibrated_sentinels": calibrated[
                    "n_sentinel_selected"
                ],
                "state_v426_precision": v426_supported["precision"],
                "state_v426_recall": v426_supported["recall"],
                "state_v426_sentinels": v426_supported[
                    "n_sentinel_selected"
                ],
                "state_v427_precision": v427_supported["precision"],
                "state_v427_recall": v427_supported["recall"],
                "state_v427_sentinels": v427_supported[
                    "n_sentinel_selected"
                ],
                "empirical_null_added_members": empirical["n_added_members"],
                "empirical_null_added_sentinels": empirical[
                    "n_sentinel_added_members"
                ],
                "empirical_null_pairwise_relations": empirical[
                    "pairwise_relations_emitted"
                ],
                "empirical_null_direct_relations_created": empirical[
                    "direct_relations_created"
                ],
                "multievidence_added_members": multievidence["n_added_members"],
                "multievidence_incremental_members_beyond_v425": (
                    multievidence["n_incremental_members_beyond_v425"]
                ),
                "multievidence_added_sentinels": multievidence[
                    "n_sentinel_added_members"
                ],
                "multievidence_pairwise_relations": multievidence[
                    "pairwise_relations_emitted"
                ],
                "multievidence_direct_relations_created": multievidence[
                    "direct_relations_created"
                ],
                "topology_consensus_added_members": topology_consensus[
                    "n_added_members"
                ],
                "topology_consensus_incremental_members_beyond_v426": (
                    topology_consensus[
                        "n_incremental_members_beyond_v426"
                    ]
                ),
                "topology_consensus_added_sentinels": topology_consensus[
                    "n_sentinel_added_members"
                ],
                "topology_consensus_pairwise_relations": (
                    topology_consensus["pairwise_relations_emitted"]
                ),
                "topology_consensus_direct_relations_created": (
                    topology_consensus["direct_relations_created"]
                ),
                "topology_consensus_direct_structure_alone_is_insufficient": (
                    topology_consensus[
                        "direct_structure_alone_is_insufficient"
                    ]
                ),
                "loading_pair_precision": loading_pair["pair_precision"],
                "loading_pair_recall": loading_pair["pair_recall"],
                "loading_hypothesis_coverage_precision": loading_pair[
                    "coverage_precision"
                ],
                "loading_hypothesis_coverage_recall": loading_pair[
                    "coverage_recall"
                ],
                "loading_nominal_v431_pair_precision": loading_pair[
                    "nominal_v431_pair_precision"
                ],
                "loading_nominal_v431_pair_recall": loading_pair[
                    "nominal_v431_pair_recall"
                ],
                "loading_v432_structure_equivalence_hyperedges": loading_pair[
                    "n_structure_equivalence_hyperedges"
                ],
                "loading_v432_exact_loading_groups": loading_pair[
                    "n_exact_loading_groups"
                ],
                "loading_v432_candidate_pair_alternatives": loading_pair[
                    "n_candidate_pairs"
                ],
                "loading_candidate_pairs_expanded_as_edges": loading_pair[
                    "candidate_pairs_expanded_as_edges"
                ],
                "loading_sentinel_members": loading_pair[
                    "n_sentinel_members"
                ],
                "loading_pre_v429_groups": loading_guard[
                    "n_pre_v429_selected_groups"
                ],
                "loading_v429_retained_groups": loading_guard[
                    "n_v429_retained_groups"
                ],
                "loading_v429_state_route_groups": loading_guard[
                    "n_v429_state_route_groups"
                ],
                "loading_v429_axis_route_groups": loading_guard[
                    "n_v429_axis_route_groups"
                ],
                "loading_v430_retained_groups": loading_guard[
                    "n_v430_retained_groups"
                ],
                "loading_v430_state_route_groups": loading_guard[
                    "n_v430_state_route_groups"
                ],
                "loading_v430_axis_route_groups": loading_guard[
                    "n_v430_axis_route_groups"
                ],
                "loading_v430_rejected_groups": (
                    loading_guard["n_pre_v429_selected_groups"]
                    - loading_guard["n_v430_retained_groups"]
                ),
                "loading_v431_retained_groups": loading_guard[
                    "n_retained_groups"
                ],
                "loading_v431_state_route_groups": loading_guard[
                    "n_state_route_groups"
                ],
                "loading_v431_axis_route_groups": loading_guard[
                    "n_axis_route_groups"
                ],
                "loading_v431_rejected_groups": loading_guard[
                    "n_rejected_groups"
                ],
                "loading_v430_structure_overlap_downgrades": loading_guard[
                    "n_structure_overlap_downgrades"
                ],
                "loading_v430_uncertainty_floor_downgrades": loading_guard[
                    "n_uncertainty_floor_downgrades"
                ],
                "loading_v431_competitor_observability_downgrades": loading_guard[
                    "n_competing_parent_observability_downgrades"
                ],
                "loading_v431_comparison_unavailable_downgrades": loading_guard[
                    "n_competing_parent_comparison_unavailable_downgrades"
                ],
                "loading_v431_guard_safe": loading_guard[
                    "safe_layered_semantics"
                ],
                "loading_v432_structure_equivalence_safe": bool(
                    loading_guard[
                        "state_route_is_one_unexpanded_two_by_two_equivalence_hypothesis"
                    ]
                    and not loading_guard["state_route_exact_pair_identified"]
                    and loading_guard[
                        "axis_route_remains_one_exact_assay_local_group"
                    ]
                    and not loading_guard[
                        "candidate_pairs_expanded_as_edges"
                    ]
                ),
                "loading_v430_direct_structure_alone_is_insufficient": (
                    loading_guard["direct_structure_alone_is_insufficient"]
                ),
                "equivalence_only_members": equivalence[
                    "n_equivalence_only_members"
                ],
                "equivalence_groups_match": equivalence[
                    "groups_equal_feature_membership"
                ],
                "equivalence_pairwise_relations": equivalence[
                    "pairwise_relations_emitted"
                ],
                "direct_precision": direct["skeleton_precision"],
                "direct_recall": direct["skeleton_recall"],
                "strict_reverse_edges": direct["n_strict_reverse"],
                "world_reliability_success": metrics[
                    "preregistered_validation"
                ]["world_reliability_success"],
            }
        )
    world_metrics = pd.DataFrame(rows)
    world_metrics.to_csv(
        out_dir / "phase_b_world_metrics.tsv", sep="\t", index=False
    )
    _write_json(out_dir / "phase_b_metrics.json", metrics_by_world)
    valid = world_metrics.loc[world_metrics["status"].eq("ok")]
    summary = {
        "schema_version": SCHEMA,
        "phase": "b",
        "algorithm_schema_version": manifest["algorithm_schema_version"],
        "selection_seed": int(selection["selection_seed"]),
        "eligible_pool_size": int(selection["eligible_pool_size"]),
        "panel_size": int(selection["panel_size"]),
        "successful_phase_a_worlds": int(
            sum(row["status"] == "ok" for row in attempts)
        ),
        "successful_phase_b_worlds": len(valid),
        "mean_state_supported_precision": float(
            valid["state_supported_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_state_supported_recall": float(
            valid["state_supported_recall"].mean()
        ) if len(valid) else 0.0,
        "total_state_supported_sentinels": int(
            valid["state_supported_sentinels"].sum()
        ) if len(valid) else 0,
        "mean_state_calibrated_precision": float(
            valid["state_calibrated_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_state_calibrated_recall": float(
            valid["state_calibrated_recall"].mean()
        ) if len(valid) else 0.0,
        "total_state_calibrated_sentinels": int(
            valid["state_calibrated_sentinels"].sum()
        ) if len(valid) else 0,
        "mean_state_v426_precision": float(
            valid["state_v426_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_state_v426_recall": float(
            valid["state_v426_recall"].mean()
        ) if len(valid) else 0.0,
        "total_state_v426_sentinels": int(
            valid["state_v426_sentinels"].sum()
        ) if len(valid) else 0,
        "mean_state_v427_precision": float(
            valid["state_v427_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_state_v427_recall": float(
            valid["state_v427_recall"].mean()
        ) if len(valid) else 0.0,
        "minimum_state_v427_recall": float(
            valid["state_v427_recall"].min()
        ) if len(valid) else 0.0,
        "total_state_v427_sentinels": int(
            valid["state_v427_sentinels"].sum()
        ) if len(valid) else 0,
        "total_empirical_null_added_members": int(
            valid["empirical_null_added_members"].sum()
        ) if len(valid) else 0,
        "total_empirical_null_added_sentinels": int(
            valid["empirical_null_added_sentinels"].sum()
        ) if len(valid) else 0,
        "total_empirical_null_pairwise_relations": int(
            valid["empirical_null_pairwise_relations"].sum()
        ) if len(valid) else 0,
        "total_empirical_null_direct_relations_created": int(
            valid["empirical_null_direct_relations_created"].sum()
        ) if len(valid) else 0,
        "total_multievidence_added_members": int(
            valid["multievidence_added_members"].sum()
        ) if len(valid) else 0,
        "total_multievidence_incremental_members_beyond_v425": int(
            valid["multievidence_incremental_members_beyond_v425"].sum()
        ) if len(valid) else 0,
        "total_multievidence_added_sentinels": int(
            valid["multievidence_added_sentinels"].sum()
        ) if len(valid) else 0,
        "total_multievidence_pairwise_relations": int(
            valid["multievidence_pairwise_relations"].sum()
        ) if len(valid) else 0,
        "total_multievidence_direct_relations_created": int(
            valid["multievidence_direct_relations_created"].sum()
        ) if len(valid) else 0,
        "total_topology_consensus_added_members": int(
            valid["topology_consensus_added_members"].sum()
        ) if len(valid) else 0,
        "total_topology_consensus_incremental_members_beyond_v426": int(
            valid[
                "topology_consensus_incremental_members_beyond_v426"
            ].sum()
        ) if len(valid) else 0,
        "total_topology_consensus_added_sentinels": int(
            valid["topology_consensus_added_sentinels"].sum()
        ) if len(valid) else 0,
        "total_topology_consensus_pairwise_relations": int(
            valid["topology_consensus_pairwise_relations"].sum()
        ) if len(valid) else 0,
        "total_topology_consensus_direct_relations_created": int(
            valid[
                "topology_consensus_direct_relations_created"
            ].sum()
        ) if len(valid) else 0,
        "all_worlds_direct_structure_alone_is_insufficient": bool(
            len(valid)
            and valid[
                "topology_consensus_direct_structure_alone_is_insufficient"
            ].astype(bool).all()
        ),
        "mean_loading_pair_precision": float(
            valid["loading_pair_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_loading_pair_recall": float(
            valid["loading_pair_recall"].mean()
        ) if len(valid) else 0.0,
        "mean_loading_hypothesis_coverage_precision": float(
            valid["loading_hypothesis_coverage_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_loading_hypothesis_coverage_recall": float(
            valid["loading_hypothesis_coverage_recall"].mean()
        ) if len(valid) else 0.0,
        "mean_loading_nominal_v431_pair_precision": float(
            valid["loading_nominal_v431_pair_precision"].mean()
        ) if len(valid) else 0.0,
        "mean_loading_nominal_v431_pair_recall": float(
            valid["loading_nominal_v431_pair_recall"].mean()
        ) if len(valid) else 0.0,
        "total_v432_structure_equivalence_hyperedges": int(
            valid["loading_v432_structure_equivalence_hyperedges"].sum()
        ) if len(valid) else 0,
        "total_v432_exact_assay_local_loading_groups": int(
            valid["loading_v432_exact_loading_groups"].sum()
        ) if len(valid) else 0,
        "total_v432_candidate_pair_alternatives": int(
            valid["loading_v432_candidate_pair_alternatives"].sum()
        ) if len(valid) else 0,
        "candidate_pair_alternatives_expanded_as_edges": bool(
            len(valid)
            and valid[
                "loading_candidate_pairs_expanded_as_edges"
            ].astype(bool).any()
        ),
        "total_loading_sentinel_members": int(
            valid["loading_sentinel_members"].sum()
        ) if len(valid) else 0,
        "total_pre_v429_loading_groups": int(
            valid["loading_pre_v429_groups"].sum()
        ) if len(valid) else 0,
        "total_v429_retained_loading_groups": int(
            valid["loading_v429_retained_groups"].sum()
        ) if len(valid) else 0,
        "total_v429_state_route_loading_groups": int(
            valid["loading_v429_state_route_groups"].sum()
        ) if len(valid) else 0,
        "total_v429_axis_route_loading_groups": int(
            valid["loading_v429_axis_route_groups"].sum()
        ) if len(valid) else 0,
        "total_v430_retained_loading_groups": int(
            valid["loading_v430_retained_groups"].sum()
        ) if len(valid) else 0,
        "total_v430_state_route_loading_groups": int(
            valid["loading_v430_state_route_groups"].sum()
        ) if len(valid) else 0,
        "total_v430_axis_route_loading_groups": int(
            valid["loading_v430_axis_route_groups"].sum()
        ) if len(valid) else 0,
        "total_v430_rejected_loading_groups": int(
            valid["loading_v430_rejected_groups"].sum()
        ) if len(valid) else 0,
        "total_v430_structure_overlap_downgrades": int(
            valid["loading_v430_structure_overlap_downgrades"].sum()
        ) if len(valid) else 0,
        "total_v430_uncertainty_floor_downgrades": int(
            valid["loading_v430_uncertainty_floor_downgrades"].sum()
        ) if len(valid) else 0,
        "total_v431_retained_loading_groups": int(
            valid["loading_v431_retained_groups"].sum()
        ) if len(valid) else 0,
        "total_v431_state_route_loading_groups": int(
            valid["loading_v431_state_route_groups"].sum()
        ) if len(valid) else 0,
        "total_v431_axis_route_loading_groups": int(
            valid["loading_v431_axis_route_groups"].sum()
        ) if len(valid) else 0,
        "total_v431_rejected_loading_groups": int(
            valid["loading_v431_rejected_groups"].sum()
        ) if len(valid) else 0,
        "total_v431_competitor_observability_downgrades": int(
            valid[
                "loading_v431_competitor_observability_downgrades"
            ].sum()
        ) if len(valid) else 0,
        "total_v431_comparison_unavailable_downgrades": int(
            valid[
                "loading_v431_comparison_unavailable_downgrades"
            ].sum()
        ) if len(valid) else 0,
        "all_worlds_loading_v431_guard_safe": bool(
            len(valid) and valid["loading_v431_guard_safe"].astype(bool).all()
        ),
        "all_worlds_loading_v432_structure_equivalence_safe": bool(
            len(valid)
            and valid[
                "loading_v432_structure_equivalence_safe"
            ].astype(bool).all()
        ),
        "all_worlds_loading_direct_structure_alone_is_insufficient": bool(
            len(valid)
            and valid[
                "loading_v430_direct_structure_alone_is_insufficient"
            ].astype(bool).all()
        ),
        "total_equivalence_pairwise_relations": int(
            valid["equivalence_pairwise_relations"].sum()
        ) if len(valid) else 0,
        "mean_direct_precision": float(valid["direct_precision"].mean())
        if len(valid)
        else 0.0,
        "mean_direct_recall": float(valid["direct_recall"].mean())
        if len(valid)
        else 0.0,
        "total_strict_reverse_edges": int(valid["strict_reverse_edges"].sum())
        if len(valid)
        else 0,
        "world_reliability_successes": int(
            valid["world_reliability_success"].astype(bool).sum()
        ) if len(valid) else 0,
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "phase_b_world_metrics_sha256": _sha256(
            out_dir / "phase_b_world_metrics.tsv"
        ),
        "phase_b_metrics_sha256": _sha256(out_dir / "phase_b_metrics.json"),
        "new_worlds_or_datasets_created": False,
        "cached_inputs_only": True,
        "retrospective_development_only": True,
        "calibrator_application_on_training_pool_is_not_independent_validation": True,
        "scorer_or_truth_opened_only_after_phase_a_verification": True,
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    _write_json(out_dir / "phase_b_summary.json", summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=SCRIPT_DIR.parent / "formal_runs")
    parser.add_argument("--worlds-root", type=Path, default=SCRIPT_DIR.parent / "formal_worlds")
    parser.add_argument("--panel-size", type=int, default=12)
    parser.add_argument("--selection-seed", type=int, default=20260901)
    parser.add_argument(
        "--selection-file",
        type=Path,
        help=(
            "Replay an exact self-hashed pre-truth phase_a_selection.json "
            "instead of drawing another panel"
        ),
    )
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help=(
            "Self-hashed phase_a_selection.json whose worlds are excluded; "
            "repeat for multiple prior panels"
        ),
    )
    parser.add_argument("--direct-cache-root", type=Path, default=DEFAULT_DIRECT_CACHE_ROOT)
    parser.add_argument(
        "--empirical-null-calibrator",
        type=Path,
        default=DEFAULT_EMPIRICAL_NULL_CALIBRATOR,
    )
    parser.add_argument(
        "--multievidence-calibrator",
        type=Path,
        default=DEFAULT_MULTIEVIDENCE_CALIBRATOR,
    )
    parser.add_argument(
        "--topology-consensus-calibrator",
        type=Path,
        default=DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR,
    )
    parser.add_argument("--recompute-direct", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.phase == "a":
        run_phase_a(
            out_dir=args.out_dir,
            runs_root=args.runs_root,
            worlds_root=args.worlds_root,
            panel_size=args.panel_size,
            selection_seed=args.selection_seed,
            direct_cache_root=args.direct_cache_root,
            recompute_direct=args.recompute_direct,
            empirical_null_calibrator_path=args.empirical_null_calibrator,
            multievidence_calibrator_path=args.multievidence_calibrator,
            topology_consensus_calibrator_path=(
                args.topology_consensus_calibrator
            ),
            selection_file=args.selection_file,
            exclude_selections=args.exclude_selection,
        )
    else:
        run_phase_b(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
