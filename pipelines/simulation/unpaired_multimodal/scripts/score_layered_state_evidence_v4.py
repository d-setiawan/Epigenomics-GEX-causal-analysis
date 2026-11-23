#!/usr/bin/env python3
"""Score layered CA-BOSS v4 evidence without collapsing its estimands."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import empirical_null_state_calibration_v4 as empirical_null_state
import loading_group_competing_parent_observability_guard_v4 as loading_group_competitor_guard
import loading_group_state_guard_v4 as loading_group_state_guard
import loading_group_structure_equivalence_v4 as loading_group_structure_equivalence
import loading_group_structure_uncertainty_guard_v4 as loading_group_structure_guard
import multievidence_empirical_null_state_calibration_v4 as multievidence_state
import score_shared_state_subspace_v4 as subspace_scorer
import topology_consensus_state_calibration_v4 as topology_consensus_state


STATE_ROLES = {"P", "Q", "S", "C"}
NULL_ROLES = {"N1", "N2"}

SCORER_SCHEMA_VERSION = "layered_state_evidence_v4.scorer.v16"
VALIDATION_PROTOCOL_SCHEMA_VERSION = (
    "layered_state_evidence_v4.validation_protocol.v20"
)
BLOCK_VALIDATION_SCHEMA_VERSION = "layered_state_evidence_v4.block_validation.v14"

LAYERED_FEATURE_COLUMNS = (
    *subspace_scorer.MEMBERSHIP_COLUMNS,
    "upstream_conditional_state_candidate",
    "confirmed_depth_orthogonal_state_response",
    "strict_state_member",
    "state_equivalence_only_member",
    "state_supported_member",
    "state_membership_tier",
    "state_equivalence_maxT_threshold",
    "state_equivalence_basis",
    "exact_cross_assay_pair_claimed",
    "direct_causal_claimed",
    "strict_membership_changed",
    "empirical_null_calibration_active",
    "empirical_null_calibration_candidate",
    "empirical_null_marginal_state_z",
    "empirical_null_conditional_excess_z",
    "empirical_null_composite_score",
    "empirical_null_selection_threshold",
    "empirical_null_state_member",
    "state_calibrated_supported_member",
    "state_calibrated_membership_tier",
    "empirical_null_calibration_schema",
    "empirical_null_threshold_method",
    "multievidence_calibration_active",
    "multievidence_calibration_candidate",
    "multievidence_state_specific_gate",
    "multievidence_component_z__oof_marginal_state_fit",
    "multievidence_component_z__oof_joint_fit",
    "multievidence_component_z__conditional_wald_above_null_95pct",
    "multievidence_component_z__median_conditional_score_norm",
    "multievidence_route_z__joint_conditional_magnitude",
    "multievidence_route_z__joint_conditional_excess",
    "multievidence_route_z__marginal_conditional_magnitude",
    "multievidence_winning_route",
    "multievidence_family_score",
    "multievidence_selection_threshold",
    "multievidence_empirical_null_state_member",
    "state_multievidence_supported_member",
    "state_multievidence_membership_tier",
    "multievidence_calibration_schema",
    "multievidence_threshold_method",
    "state_v426_supported_member",
    "state_v426_membership_tier",
    "state_response_status",
    "assay_local_loading_group_pre_v428_ids",
    "assay_local_loading_group_pre_v428_support",
    "assay_local_singleton_candidate_ids",
    "assay_local_singleton_candidate",
    "within_assay_direct_candidate_degree",
    "within_assay_direct_candidate_overlap",
    "within_assay_direct_role",
    "topology_consensus_calibration_active",
    "topology_consensus_calibration_candidate",
    "topology_consensus_state_specific_gate",
    "topology_consensus_residual_structure_support",
    "topology_consensus_fixed_rule_eligible",
    "topology_consensus_empirical_null_certified",
    "topology_consensus_state_member",
    "state_v427_supported_member",
    "state_v427_membership_tier",
    "topology_consensus_calibration_schema",
    "topology_consensus_null_control_method",
    "assay_local_loading_group_ids",
    "assay_local_loading_group_rejected_ids",
    "assay_local_loading_group_support",
    "assay_local_loading_group_guard_status",
    "assay_local_state_support_status",
    "support_layers",
    "layer_union_is_state_membership",
    "assay_participation_implies_feature_membership",
    "cross_assay_pair_claimed",
    "direct_cross_assay_effect_claimed",
)

FRESH_TRAIN_SEEDS = tuple(range(49_001, 49_025))
FRESH_DEVELOPMENT_SEEDS = tuple(range(50_001, 50_017))
FRESH_LOCKED_SEEDS = tuple(range(51_001, 51_041))
VALIDATION_SPLITS = {
    "TRAIN": FRESH_TRAIN_SEEDS,
    "DEVELOPMENT": FRESH_DEVELOPMENT_SEEDS,
    "LOCKED": FRESH_LOCKED_SEEDS,
}
MINIMUM_RELIABLE_WORLDS = {
    "TRAIN": 22,
    "DEVELOPMENT": 16,
    "LOCKED": 36,
}

MIN_CONFIRMED_SUBSET_PRECISION = 0.95
EXPECTED_ASSAY_COUNT = 7
MIN_DIRECT_PRECISION = 0.80
MIN_WORLD_DIRECT_RECALL = 0.90
MIN_LOADING_PAIR_PRECISION = 0.80
MIN_TOPOLOGY_STATE_RECALL = 0.80
MAX_WORLD_FOCAL_DENSITY = 20

MIN_BLOCK_MEAN_DIRECT_RECALL = 0.95
MIN_BLOCK_MEAN_LOADING_PAIR_RECALL = 0.80
MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION = 0.90
MAX_BLOCK_MEAN_FOCAL_DENSITY = 20.0
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_BINDING_HASH_FIELDS = frozenset(
    {
        "layered_algorithm_source_sha256",
        "layered_algorithm_config_sha256",
        "layered_validation_protocol_payload_sha256",
        "empirical_null_calibrator_artifact_sha256",
        "empirical_null_calibrator_payload_sha256",
        "multievidence_calibrator_artifact_sha256",
        "multievidence_calibrator_payload_sha256",
        "topology_consensus_calibrator_artifact_sha256",
        "topology_consensus_calibrator_payload_sha256",
        "subspace_algorithm_source_sha256",
        "subspace_algorithm_config_sha256",
        "subspace_embedding_config_sha256",
        "subspace_embedding_source_sha256",
        "local_config_sha256",
        "joint_candidate_identity_sha256",
        "layered_candidate_identity_sha256",
        "global_pretruth_manifest_payload_sha256",
    }
)
_FORMAL_BINDING_TEXT_FIELDS = frozenset(
    {"subspace_embedding_method", "local_candidate_name"}
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().upper()
    if normalized == "DEV":
        normalized = "DEVELOPMENT"
    if normalized not in VALIDATION_SPLITS:
        raise ValueError(
            f"Unknown validation split {split!r}; expected TRAIN, DEVELOPMENT/DEV, or LOCKED"
        )
    return normalized


def _doe_cell_for_seed(seed: int) -> int:
    """Return the fixed eight-cell half-fraction assignment used by v4."""

    return int((int(seed) - 1) % 8 + 1)


def _validate_formal_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    required = _FORMAL_BINDING_HASH_FIELDS | _FORMAL_BINDING_TEXT_FIELDS | {
        "joint_executable_source_sha256"
    }
    if set(binding) != set(required):
        raise ValueError(
            "Formal layered binding fields are missing or extra: "
            f"missing={sorted(set(required).difference(binding))}, "
            f"extra={sorted(set(binding).difference(required))}"
        )
    normalized: dict[str, Any] = {}
    for field in sorted(_FORMAL_BINDING_HASH_FIELDS):
        value = str(binding[field]).strip().lower()
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"Formal binding {field} is not a SHA-256")
        normalized[field] = value
    for field in sorted(_FORMAL_BINDING_TEXT_FIELDS):
        value = str(binding[field]).strip()
        if not value:
            raise ValueError(f"Formal binding {field} may not be empty")
        normalized[field] = value
    source_map = binding["joint_executable_source_sha256"]
    if not isinstance(source_map, Mapping) or not source_map:
        raise ValueError("Formal binding joint source map must be non-empty")
    normalized_sources: dict[str, str] = {}
    for name, digest in source_map.items():
        basename = str(name).strip()
        value = str(digest).strip().lower()
        if Path(basename).name != basename or basename in normalized_sources:
            raise ValueError("Formal binding source names must be unique basenames")
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"Formal binding source hash is invalid: {basename}")
        normalized_sources[basename] = value
    if normalized_sources.get("layered_state_evidence_v4.py") != normalized[
        "layered_algorithm_source_sha256"
    ] or not {
        "score_layered_state_evidence_v4.py",
        "loading_group_state_guard_v4.py",
        "loading_group_structure_uncertainty_guard_v4.py",
        "loading_group_competing_parent_observability_guard_v4.py",
        "loading_group_structure_equivalence_v4.py",
        "multievidence_empirical_null_state_calibration_v4.py",
        "topology_consensus_state_calibration_v4.py",
    }.issubset(normalized_sources):
        raise ValueError("Formal binding does not include every layered v4.32 source")
    if normalized["layered_validation_protocol_payload_sha256"] != (
        layered_validation_protocol()["protocol_payload_sha256"]
    ):
        raise ValueError("Formal binding uses another layered validation protocol")
    normalized["joint_executable_source_sha256"] = dict(
        sorted(normalized_sources.items())
    )
    return normalized


def layered_validation_protocol() -> dict[str, Any]:
    """Return the preregistrable fresh-block protocol without opening data."""

    protocol: dict[str, Any] = {
        "schema_version": VALIDATION_PROTOCOL_SCHEMA_VERSION,
        "fresh_blocks": {
            split: {
                "world_seeds": list(seeds),
                "n_registered_worlds": len(seeds),
                "minimum_reliable_worlds": MINIMUM_RELIABLE_WORLDS[split],
                "expected_worlds_per_DOE_cell": len(seeds) // 8,
            }
            for split, seeds in VALIDATION_SPLITS.items()
        },
        "world_reliability_gate": {
            "confirmed_maxT_subset_precision_minimum": MIN_CONFIRMED_SUBSET_PRECISION,
            "confirmed_sentinel_members_maximum": 0,
            "assay_participation_coverage": f"all_{EXPECTED_ASSAY_COUNT}_assays",
            "rna_assay_participation_required": True,
            "direct_pair_precision_minimum": MIN_DIRECT_PRECISION,
            "direct_pair_recall_minimum": MIN_WORLD_DIRECT_RECALL,
            "strict_reverse_edges_maximum": 0,
            "loading_group_pair_precision_minimum": MIN_LOADING_PAIR_PRECISION,
            "loading_group_sentinel_members_maximum": 0,
            "loading_group_member_conflations_maximum": 0,
            "loading_group_state_guard_required": True,
            "loading_group_state_guard_schema": loading_group_state_guard.SCHEMA,
            "loading_group_state_guard_fixed_rule": loading_group_state_guard.FIXED_RULE,
            "loading_group_structure_uncertainty_guard_required": True,
            "loading_group_structure_uncertainty_guard_schema": (
                loading_group_structure_guard.SCHEMA
            ),
            "loading_group_structure_uncertainty_guard_fixed_rule": (
                loading_group_structure_guard.FIXED_RULE
            ),
            "loading_group_competing_parent_observability_guard_required": True,
            "loading_group_competing_parent_observability_guard_schema": (
                loading_group_competitor_guard.SCHEMA
            ),
            "loading_group_competing_parent_observability_guard_fixed_rule": (
                loading_group_competitor_guard.FIXED_RULE
            ),
            "loading_group_state_route_requires_exact_structure_anchor_competition": True,
            "loading_group_competing_parent_uses_frozen_v427_observability_rule": True,
            "loading_group_axis_route_bypasses_competing_parent_gate": True,
            "loading_group_competing_parent_downgrade_means_absence": False,
            "loading_group_structure_equivalence_required": True,
            "loading_group_structure_equivalence_schema": (
                loading_group_structure_equivalence.SCHEMA
            ),
            "loading_group_structure_equivalence_fixed_rule": (
                loading_group_structure_equivalence.FIXED_RULE
            ),
            "loading_group_state_route_candidate_pair_count": 4,
            "loading_group_state_route_exact_pair_identified": False,
            "loading_group_state_route_candidate_pair_expansion_allowed": False,
            "loading_group_axis_route_candidate_pair_count": 1,
            "loading_group_state_route_minimum_member_centroid_cosine": (
                loading_group_structure_guard
                .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
            ),
            "loading_group_complete_direct_collider_overlap_is_unresolved": True,
            "loading_group_direct_overlap_downgrade_means_absence": False,
            "loading_group_state_route_members_must_be_state_v427_supported": True,
            "loading_group_axis_route_requires_leave_one_group_out_cross_assay_consensus": True,
            "loading_group_axis_route_requires_at_least_one_state_v427_member": True,
            "loading_group_axis_route_changes_state_v427_membership": False,
            "loading_group_axis_route_claims_exact_cross_assay_feature_pair": False,
            "loading_group_direct_structure_alone_is_sufficient": False,
            "selected_cross_assay_state_hyperedges_required": 1,
            "state_hyperedge_members_equal_strict_confirmed_participating_subset": True,
            "state_hyperedge_modalities_equal_assay_participation": True,
            "state_hyperedge_is_hypothesis_only_common_model_without_pair_direction_or_direct_claim": True,
            "layer_union_claim_allowed": False,
            "assay_participation_feature_membership_claim_allowed": False,
            "cross_assay_pair_claim_allowed": False,
            "direct_cross_assay_effect_claim_allowed": False,
            "state_equivalence_pairwise_expansion_allowed": False,
            "state_equivalence_recall_is_required": False,
            "empirical_null_state_calibrator_required": True,
            "empirical_null_state_supported_precision_minimum": (
                MIN_CONFIRMED_SUBSET_PRECISION
            ),
            "empirical_null_state_sentinel_members_maximum": 0,
            "empirical_null_state_pairwise_or_direct_claim_allowed": False,
            "empirical_null_state_recall_is_required": False,
            "multievidence_state_calibrator_required": True,
            "multievidence_state_supported_precision_minimum": (
                MIN_CONFIRMED_SUBSET_PRECISION
            ),
            "multievidence_state_sentinel_members_maximum": 0,
            "multievidence_state_pairwise_or_direct_claim_allowed": False,
            "multievidence_state_recall_is_required": False,
            "topology_consensus_state_calibrator_required": True,
            "topology_consensus_state_supported_precision_minimum": (
                MIN_CONFIRMED_SUBSET_PRECISION
            ),
            "topology_consensus_state_supported_recall_minimum": (
                MIN_TOPOLOGY_STATE_RECALL
            ),
            "topology_consensus_state_sentinel_members_maximum": 0,
            "topology_consensus_state_pairwise_or_direct_claim_allowed": False,
            "topology_consensus_direct_structure_alone_is_sufficient": False,
            "expanded_focal_density_maximum": MAX_WORLD_FOCAL_DENSITY,
            "strict_confirmed_state_feature_recall_is_required": False,
            "topology_consensus_v427_state_feature_recall_is_required": True,
            "loading_group_pair_recall_is_required": False,
        },
        "reported_per_world_completeness_diagnostics": {
            "loading_group_pair_recall_at_least_0_80": True,
            "full_state_feature_recovery": True,
            "state_equivalence_supported_recall": True,
            "empirical_null_calibrated_supported_recall": True,
            "multievidence_v426_supported_recall": True,
            "topology_consensus_v427_supported_recall": True,
            "only_topology_consensus_v427_supported_recall_is_part_of_"
            "world_reliability_gate": True,
        },
        "block_typed_layer_quality_guardrails": {
            "mean_direct_pair_recall_minimum": MIN_BLOCK_MEAN_DIRECT_RECALL,
            "mean_loading_group_pair_recall_minimum": (
                MIN_BLOCK_MEAN_LOADING_PAIR_RECALL
            ),
            "mean_loading_group_pair_precision_minimum": (
                MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION
            ),
            "mean_expanded_focal_density_maximum": MAX_BLOCK_MEAN_FOCAL_DENSITY,
        },
        "failure_policy": {
            "every_registered_world_remains_in_world_and_mean_denominators": True,
            "unscored_recall_and_precision_contribution": 0.0,
            "unscored_density_contribution": MAX_BLOCK_MEAN_FOCAL_DENSITY,
            "missing_record_is_failure_and_breaks_DOE_cell_coverage": True,
        },
        "DOE_cell_coverage": {
            "n_cells": 8,
            "assignment": "((world_seed - 1) mod 8) + 1",
            "requires_exact_registered_record_count_per_cell": True,
            "DOE_cell_success_counts_are_reported_without_an_extra_unregistered_threshold": True,
        },
        "history_boundary": {
            "seeds_30001_30024": "diagnostic_and_algorithm_development_only",
            "seeds_33001_33024": "diagnostic_and_algorithm_development_only",
            "seeds_36001_36024": "diagnostic_and_algorithm_development_only",
            "seeds_37001_37024": "diagnostic_and_algorithm_development_only",
            "seeds_38001_38024": "diagnostic_and_algorithm_development_only",
            "seeds_39001_39024": (
                "aborted_pretruth_infrastructure_schema_mismatch_no_scorer_opened"
            ),
            "seeds_40001_40024": (
                "failed_scientific_confirmation_then_algorithm_development_only"
            ),
            "seeds_41001_41024": (
                "passed_confirmation_then_algorithm_changed_and_development_only"
            ),
            "seeds_34001_34016": (
                "opened_development_failed_v43_then_posthoc_passed_v44_"
                "development_only"
            ),
            "seeds_45001_45040": (
                "unopened_reserved_superseded_before_v44_fresh_validation"
            ),
            "seeds_46001_46024": (
                "passed_confirmation_then_DEV470_informed_v45_and_"
                "development_only"
            ),
            "seeds_47001_47016": (
                "opened_development_failed_v44_then_posthoc_passed_v45_"
                "development_only"
            ),
            "seeds_48001_48040": (
                "unopened_reserved_superseded_before_v45_fresh_validation"
            ),
            "seeds_31001_31016": (
                "unopened_reserved_superseded_exact_axis_DEV"
            ),
            "seeds_42001_42040": (
                "unopened_reserved_superseded_exact_axis_LOCKED"
            ),
        },
    }
    protocol["protocol_payload_sha256"] = _canonical_sha256(protocol)
    return protocol


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _maximum_truth_matching(
    candidate_sets: Sequence[set[frozenset[str]]],
    truth_pairs: set[frozenset[str]],
) -> dict[int, frozenset[str]]:
    """Match set-valued hypotheses to distinct truth pairs for scoring only."""

    truth_to_hypothesis: dict[frozenset[str], int] = {}

    def augment(hypothesis: int, seen: set[frozenset[str]]) -> bool:
        for pair in sorted(
            candidate_sets[hypothesis], key=lambda value: tuple(sorted(value))
        ):
            if pair not in truth_pairs or pair in seen:
                continue
            seen.add(pair)
            incumbent = truth_to_hypothesis.get(pair)
            if incumbent is None or augment(incumbent, seen):
                truth_to_hypothesis[pair] = hypothesis
                return True
        return False

    for hypothesis in range(len(candidate_sets)):
        augment(hypothesis, set())
    return {
        hypothesis: pair for pair, hypothesis in truth_to_hypothesis.items()
    }


def _holm(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=np.float64)
    if p_values.ndim != 1 or not len(p_values) or not np.isfinite(p_values).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("Holm inputs must lie in [0, 1]")
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order] * np.arange(len(p_values), 0, -1)
    ranked = np.maximum.accumulate(ranked)
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _strict_bool(value: Any, *, label: str) -> bool:
    """Parse serialized Booleans without treating non-empty ``"False"`` as true."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{label} must be a strict Boolean")


def _optional_text(value: Any) -> str:
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none"} else text


def _optional_float(value: Any, *, label: str) -> float:
    text = _optional_text(value)
    if not text:
        return float("nan")
    try:
        result = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite when present")
    return result


def _set_metrics(selected: set[str], positive: set[str], sentinel: set[str]) -> dict[str, Any]:
    true_selected = selected & positive
    sentinel_selected = selected & sentinel
    unknown = selected.difference(positive | sentinel)
    return {
        "n_selected": len(selected),
        "n_true_selected": len(true_selected),
        "n_sentinel_selected": len(sentinel_selected),
        "n_unknown_selected": len(unknown),
        "precision": _rate(len(true_selected), len(selected)),
        "recall": _rate(len(true_selected), len(positive)),
        "sentinel_inclusion_rate": _rate(len(sentinel_selected), len(sentinel)),
    }


def evaluate_layered_world_reliability(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen per-world reliability gate to one score payload.

    Full state-feature recall and assay-local P/S-pair recall are deliberately
    returned as diagnostics rather than reliability requirements. Their
    completeness is enforced across a registered block instead.
    """

    confirmed = metrics["set_scores"]["confirmed_state_response"]
    direct = metrics["direct_score"]
    loading = metrics["assay_local_loading_pair_score"]
    loading_guard = metrics.get("assay_local_loading_group_state_guard", {})
    focal = metrics["typed_focal_graph"]
    semantics = metrics["semantics"]
    state_hyperedge = metrics.get("cross_assay_state_hyperedge", {})
    state_equivalence_layer = metrics.get("state_equivalence_layer", {})
    empirical_null_layer = metrics.get("empirical_null_state_layer", {})
    multievidence_layer = metrics.get("multievidence_state_layer", {})
    topology_consensus_layer = metrics.get(
        "topology_consensus_state_layer", {}
    )
    participating_modalities = set(
        metrics.get("assay_participation_modalities", [])
    )

    checks = {
        "confirmed_maxT_subset_precision_at_least_0_95": float(
            confirmed["precision"]
        )
        >= MIN_CONFIRMED_SUBSET_PRECISION,
        "confirmed_zero_sentinel_members": int(
            confirmed["n_sentinel_selected"]
        )
        == 0,
        "state_equivalence_supported_precision_at_least_0_95": float(
            state_equivalence_layer.get("supported_precision", 0.0)
        )
        >= MIN_CONFIRMED_SUBSET_PRECISION,
        "state_equivalence_zero_sentinel_members": int(
            state_equivalence_layer.get("n_sentinel_supported", -1)
        )
        == 0,
        "state_equivalence_groups_match_feature_tiers": bool(
            state_equivalence_layer.get(
                "groups_equal_feature_membership", False
            )
        ),
        "state_equivalence_has_safe_set_semantics": bool(
            state_equivalence_layer.get("safe_set_semantics", False)
            and int(
                state_equivalence_layer.get("pairwise_relations_emitted", -1)
            )
            == 0
        ),
        "empirical_null_state_calibrator_active": bool(
            empirical_null_layer.get("application_active", False)
        ),
        "empirical_null_state_supported_precision_at_least_0_95": float(
            empirical_null_layer.get("calibrated_supported_precision", 0.0)
        )
        >= MIN_CONFIRMED_SUBSET_PRECISION,
        "empirical_null_state_zero_sentinel_members": int(
            empirical_null_layer.get("n_sentinel_calibrated_supported", -1)
        )
        == 0,
        "empirical_null_state_has_safe_membership_semantics": bool(
            empirical_null_layer.get("safe_membership_semantics", False)
            and int(empirical_null_layer.get("pairwise_relations_emitted", -1))
            == 0
            and int(empirical_null_layer.get("direct_relations_created", -1))
            == 0
        ),
        "multievidence_state_calibrator_active": bool(
            multievidence_layer.get("application_active", False)
        ),
        "multievidence_state_supported_precision_at_least_0_95": float(
            multievidence_layer.get("v426_supported_precision", 0.0)
        )
        >= MIN_CONFIRMED_SUBSET_PRECISION,
        "multievidence_state_zero_sentinel_members": int(
            multievidence_layer.get("n_sentinel_v426_supported", -1)
        )
        == 0,
        "multievidence_state_has_safe_membership_semantics": bool(
            multievidence_layer.get("safe_membership_semantics", False)
            and int(multievidence_layer.get("pairwise_relations_emitted", -1))
            == 0
            and int(multievidence_layer.get("direct_relations_created", -1))
            == 0
        ),
        "topology_consensus_state_calibrator_active": bool(
            topology_consensus_layer.get("application_active", False)
        ),
        "topology_consensus_state_supported_precision_at_least_0_95": float(
            topology_consensus_layer.get("v427_supported_precision", 0.0)
        )
        >= MIN_CONFIRMED_SUBSET_PRECISION,
        "topology_consensus_state_supported_recall_at_least_0_80": float(
            topology_consensus_layer.get("v427_supported_recall", 0.0)
        )
        >= MIN_TOPOLOGY_STATE_RECALL,
        "topology_consensus_state_zero_sentinel_members": int(
            topology_consensus_layer.get("n_sentinel_v427_supported", -1)
        )
        == 0,
        "topology_consensus_state_has_safe_membership_semantics": bool(
            topology_consensus_layer.get("safe_membership_semantics", False)
            and topology_consensus_layer.get(
                "direct_structure_alone_is_insufficient", False
            )
            and int(
                topology_consensus_layer.get("pairwise_relations_emitted", -1)
            )
            == 0
            and int(
                topology_consensus_layer.get("direct_relations_created", -1)
            )
            == 0
        ),
        "assay_participation_all_7_assays": bool(
            len(participating_modalities) == EXPECTED_ASSAY_COUNT
            and float(metrics.get("assay_participation_coverage", 0.0)) == 1.0
        ),
        "assay_participation_rna_present": "rna" in participating_modalities,
        "direct_pair_precision_at_least_0_80": float(
            direct["skeleton_precision"]
        )
        >= MIN_DIRECT_PRECISION,
        "direct_pair_recall_at_least_0_90": float(direct["skeleton_recall"])
        >= MIN_WORLD_DIRECT_RECALL,
        "direct_zero_strict_reverse": int(direct["n_strict_reverse"]) == 0,
        "loading_pair_precision_at_least_0_80": float(
            loading["pair_precision"]
        )
        >= MIN_LOADING_PAIR_PRECISION,
        "loading_zero_sentinel_or_member_conflation": bool(
            int(loading["n_sentinel_members"]) == 0
            and int(loading["n_non_pair_hyperedges"]) == 0
            and int(loading["n_feature_membership_mismatches"]) == 0
        ),
        "loading_group_state_guard_active": bool(
            loading_guard.get("application_active", False)
            and loading_guard.get("guard_schema")
            == loading_group_state_guard.SCHEMA
            and loading_guard.get("fixed_rule")
            == loading_group_state_guard.FIXED_RULE
            and loading_guard.get("structure_uncertainty_guard_schema")
            == loading_group_structure_guard.SCHEMA
            and loading_guard.get("structure_uncertainty_fixed_rule")
            == loading_group_structure_guard.FIXED_RULE
            and loading_guard.get(
                "competing_parent_observability_guard_schema"
            )
            == loading_group_competitor_guard.SCHEMA
            and loading_guard.get(
                "competing_parent_observability_fixed_rule"
            )
            == loading_group_competitor_guard.FIXED_RULE
            and float(
                loading_guard.get(
                    "minimum_state_route_member_centroid_cosine", -1.0
                )
            )
            == loading_group_structure_guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        ),
        "loading_group_state_guard_safe": bool(
            loading_guard.get("safe_layered_semantics", False)
            and loading_guard.get(
                "state_route_members_are_state_v427_supported", False
            )
            and loading_guard.get(
                "axis_route_has_state_anchor_and_leave_one_group_out_consensus",
                False,
            )
            and loading_guard.get("selection_matches_fixed_rule", False)
            and loading_guard.get(
                "direct_structure_alone_is_insufficient", False
            )
            and loading_guard.get(
                "direct_overlap_is_unresolved_not_refuted", False
            )
            and loading_guard.get(
                "state_route_requires_exact_structure_anchor_competition",
                False,
            )
            and loading_guard.get(
                "competing_parent_uses_frozen_v427_observability_rule",
                False,
            )
            and loading_guard.get(
                "axis_route_bypasses_competing_parent_gate", False
            )
            and loading_guard.get(
                "competing_parent_downgrade_is_unresolved_not_refuted",
                False,
            )
            and not loading_guard.get("direct_candidates_changed", True)
            and not loading_guard.get("changes_state_v427_membership", True)
            and not loading_guard.get("truth_required_for_application", True)
        ),
        "loading_group_structure_equivalence_active": bool(
            loading_guard.get("structure_equivalence_schema")
            == loading_group_structure_equivalence.SCHEMA
            and loading_guard.get("structure_equivalence_fixed_rule")
            == loading_group_structure_equivalence.FIXED_RULE
        ),
        "loading_group_structure_equivalence_safe": bool(
            loading_guard.get(
                "state_route_is_one_unexpanded_two_by_two_equivalence_hypothesis",
                False,
            )
            and loading_guard.get("state_route_exact_pair_identified", True)
            is False
            and loading_guard.get(
                "axis_route_remains_one_exact_assay_local_group", False
            )
            and loading_guard.get("candidate_pairs_expanded_as_edges", True)
            is False
            and int(loading.get("n_non_pair_hyperedges", -1)) == 0
            and int(loading.get("n_structure_equivalence_hyperedges", -1))
            + int(loading.get("n_exact_loading_groups", -1))
            == int(loading.get("n_selected_hyperedges", -2))
        ),
        "exactly_one_selected_cross_assay_state_hyperedge": int(
            state_hyperedge.get("n_selected", -1)
        )
        == 1,
        "state_hyperedge_members_equal_strict_confirmed_participating_subset": bool(
            state_hyperedge.get(
                "members_equal_strict_confirmed_participating_subset", False
            )
        ),
        "state_hyperedge_modalities_equal_assay_participation": bool(
            state_hyperedge.get(
                "modalities_equal_assay_participation", False
            )
        ),
        "state_hyperedge_has_only_common_model_hypothesis_semantics": bool(
            state_hyperedge.get("safe_hypothesis_semantics", False)
        ),
        "semantic_no_union_or_cross_assay_pair_claim": bool(
            not semantics["layer_union_is_state_membership"]
            and not semantics["cross_assay_pair_claimed"]
            and not semantics["direct_cross_assay_effect_claimed"]
            and not bool(
                semantics.get("state_equivalence_is_pairwise_structure", True)
            )
        ),
        "semantic_assay_participation_is_not_feature_membership": bool(
            not semantics["assay_participation_implies_feature_membership"]
        ),
        "expanded_focal_density_at_most_20": int(
            focal["n_expanded_focal_relations"]
        )
        <= MAX_WORLD_FOCAL_DENSITY,
    }
    diagnostics = {
        "confirmed_state_feature_recall": float(confirmed["recall"]),
        "full_state_feature_recovery_success": bool(
            metrics.get("confirmed_state_recovery_success", False)
        ),
        "loading_pair_recall": float(loading["pair_recall"]),
        "state_equivalence_supported_recall": float(
            state_equivalence_layer.get("supported_recall", 0.0)
        ),
        "empirical_null_calibrated_supported_recall": float(
            empirical_null_layer.get("calibrated_supported_recall", 0.0)
        ),
        "multievidence_v426_supported_recall": float(
            multievidence_layer.get("v426_supported_recall", 0.0)
        ),
        "topology_consensus_v427_supported_recall": float(
            topology_consensus_layer.get("v427_supported_recall", 0.0)
        ),
        "loading_pair_recall_at_least_0_80": float(loading["pair_recall"])
        >= MIN_BLOCK_MEAN_LOADING_PAIR_RECALL,
        "full_state_feature_recall_is_world_gate": False,
        "loading_pair_recall_is_world_gate": False,
        "state_equivalence_supported_recall_is_world_gate": False,
        "empirical_null_calibrated_supported_recall_is_world_gate": False,
        "multievidence_v426_supported_recall_is_world_gate": False,
        "topology_consensus_v427_supported_recall_is_world_gate": True,
    }
    return {
        "schema_version": "layered_state_evidence_v4.world_reliability.v10",
        "gate_checks": checks,
        "world_reliability_success": bool(all(checks.values())),
        "reported_completeness_diagnostics": diagnostics,
        "estimand": (
            "precision_and_safety_of_separate_typed_layers_with_completeness_"
            "guarded_at_block_level"
        ),
    }


def validate_layered_evidence_block(
    split: str,
    attempts: Sequence[Mapping[str, Any]],
    *,
    formal_binding: Mapping[str, Any] | None = None,
    require_formal_binding: bool = False,
) -> dict[str, Any]:
    """Validate one registered fresh block, retaining every failure.

    Each attempt record has ``world_seed``, ``execution_status``, and, for a
    successful score, ``metrics``. Missing records, crashes, failed discovery,
    and invalid scores remain in the registered denominator with zero recall
    and precision. Their density contribution is the maximum allowed density,
    so a failed world can never improve the density mean.
    """

    normalized_split = _normalize_split(split)
    if require_formal_binding and formal_binding is None:
        raise ValueError("Formal block certification requires a frozen binding")
    validated_binding = (
        _validate_formal_binding(formal_binding)
        if formal_binding is not None
        else None
    )
    expected_seeds = VALIDATION_SPLITS[normalized_split]
    expected_set = set(expected_seeds)
    by_seed: dict[int, Mapping[str, Any]] = {}
    for attempt in attempts:
        seed = int(attempt["world_seed"])
        if seed not in expected_set:
            raise ValueError(
                f"Attempt seed {seed} is outside registered {normalized_split} block"
            )
        if seed in by_seed:
            raise ValueError(f"Duplicate attempt record for seed {seed}")
        by_seed[seed] = attempt

    normalized_records: list[dict[str, Any]] = []
    scored_metrics: dict[int, Mapping[str, Any]] = {}
    for seed in expected_seeds:
        attempt = by_seed.get(seed)
        if attempt is None:
            normalized_records.append(
                {
                    "world_seed": seed,
                    "DOE_cell": _doe_cell_for_seed(seed),
                    "execution_status": "missing_record",
                    "scored": False,
                    "world_reliability_success": False,
                }
            )
            continue
        status = str(attempt.get("execution_status", "missing_status"))
        metrics = attempt.get("metrics")
        scored = bool(status == "ok" and isinstance(metrics, Mapping))
        reliability: dict[str, Any] | None = None
        if scored:
            assert isinstance(metrics, Mapping)
            if int(metrics.get("world_seed", -1)) != seed:
                raise ValueError(f"Metrics/world seed mismatch for seed {seed}")
            if metrics.get("schema_version") != SCORER_SCHEMA_VERSION:
                scored = False
                status = "invalid_score_schema"
        if scored:
            assert isinstance(metrics, Mapping)
            try:
                reliability = evaluate_layered_world_reliability(metrics)
            except (KeyError, TypeError, ValueError):
                scored = False
                status = "invalid_score"
            else:
                embedded = metrics.get("preregistered_validation")
                if isinstance(embedded, Mapping) and bool(
                    embedded.get("world_reliability_success", False)
                ) != bool(reliability["world_reliability_success"]):
                    raise ValueError(
                        f"Embedded/recomputed reliability mismatch for seed {seed}"
                    )
        if scored:
            assert isinstance(metrics, Mapping)
            assert reliability is not None
            scored_metrics[seed] = metrics
        normalized_records.append(
            {
                "world_seed": seed,
                "DOE_cell": _doe_cell_for_seed(seed),
                "execution_status": status,
                "scored": scored,
                "world_reliability_success": bool(
                    scored and reliability and reliability["world_reliability_success"]
                ),
            }
        )

    denominator = len(expected_seeds)

    def mean_with_failures(path: tuple[str, ...], failed_value: float = 0.0) -> float:
        total = 0.0
        for seed in expected_seeds:
            metrics = scored_metrics.get(seed)
            if metrics is None:
                total += failed_value
                continue
            value: Any = metrics
            for key in path:
                value = value[key]
            total += float(value)
        return float(total / denominator)

    reliability_successes = sum(
        int(record["world_reliability_success"]) for record in normalized_records
    )
    mean_direct_recall = mean_with_failures(("direct_score", "skeleton_recall"))
    mean_loading_recall = mean_with_failures(
        ("assay_local_loading_pair_score", "pair_recall")
    )
    mean_loading_precision = mean_with_failures(
        ("assay_local_loading_pair_score", "pair_precision")
    )
    mean_density = mean_with_failures(
        ("typed_focal_graph", "n_expanded_focal_relations"),
        failed_value=MAX_BLOCK_MEAN_FOCAL_DENSITY,
    )

    expected_per_cell = denominator // 8
    doe_cells: dict[str, dict[str, Any]] = {}
    for cell in range(1, 9):
        recorded = [
            record
            for record in normalized_records
            if record["DOE_cell"] == cell
            and record["execution_status"] != "missing_record"
        ]
        doe_cells[str(cell)] = {
            "recorded_worlds": len(recorded),
            "expected_worlds": expected_per_cell,
            "reliable_worlds": sum(
                int(record["world_reliability_success"]) for record in recorded
            ),
            "coverage_pass": len(recorded) == expected_per_cell,
        }
    doe_coverage_pass = all(row["coverage_pass"] for row in doe_cells.values())

    world_gate = {
        "reliable_worlds": reliability_successes,
        "registered_worlds": denominator,
        "minimum_required": MINIMUM_RELIABLE_WORLDS[normalized_split],
        "pass": reliability_successes
        >= MINIMUM_RELIABLE_WORLDS[normalized_split],
    }
    typed_layer_quality_guardrails = {
        "mean_direct_pair_recall": {
            "observed": mean_direct_recall,
            "minimum": MIN_BLOCK_MEAN_DIRECT_RECALL,
            "pass": mean_direct_recall >= MIN_BLOCK_MEAN_DIRECT_RECALL,
        },
        "mean_loading_group_pair_recall": {
            "observed": mean_loading_recall,
            "minimum": MIN_BLOCK_MEAN_LOADING_PAIR_RECALL,
            "pass": mean_loading_recall >= MIN_BLOCK_MEAN_LOADING_PAIR_RECALL,
        },
        "mean_loading_group_pair_precision": {
            "observed": mean_loading_precision,
            "minimum": MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION,
            "pass": mean_loading_precision >= MIN_BLOCK_MEAN_LOADING_PAIR_PRECISION,
        },
        "mean_expanded_focal_density": {
            "observed": mean_density,
            "maximum": MAX_BLOCK_MEAN_FOCAL_DENSITY,
            "pass": mean_density <= MAX_BLOCK_MEAN_FOCAL_DENSITY,
        },
    }
    diagnostics = {
        "worlds_with_loading_pair_recall_at_least_0_80": sum(
            int(
                float(metrics["assay_local_loading_pair_score"]["pair_recall"])
                >= MIN_BLOCK_MEAN_LOADING_PAIR_RECALL
            )
            for metrics in scored_metrics.values()
        ),
        "worlds_with_full_state_feature_recovery": sum(
            int(bool(metrics.get("confirmed_state_recovery_success", False)))
            for metrics in scored_metrics.values()
        ),
        "these_diagnostics_are_not_world_reliability_gates": True,
    }
    gate_checks = {
        "minimum_reliable_worlds": bool(world_gate["pass"]),
        "block_mean_direct_recall": bool(
            typed_layer_quality_guardrails["mean_direct_pair_recall"]["pass"]
        ),
        "block_mean_loading_pair_recall": bool(
            typed_layer_quality_guardrails["mean_loading_group_pair_recall"]["pass"]
        ),
        "block_mean_loading_pair_precision": bool(
            typed_layer_quality_guardrails["mean_loading_group_pair_precision"]["pass"]
        ),
        "block_mean_density": bool(
            typed_layer_quality_guardrails["mean_expanded_focal_density"]["pass"]
        ),
        "complete_DOE_cell_record_coverage": doe_coverage_pass,
    }
    if require_formal_binding:
        gate_checks["formal_candidate_and_global_freeze_binding"] = bool(
            validated_binding
        )
    output: dict[str, Any] = {
        "schema_version": BLOCK_VALIDATION_SCHEMA_VERSION,
        "protocol_payload_sha256": layered_validation_protocol()[
            "protocol_payload_sha256"
        ],
        "split": normalized_split,
        "world_seeds": list(expected_seeds),
        "all_failures_and_crashes_retained_in_denominator": True,
        "n_scored_worlds": len(scored_metrics),
        "execution_status_counts": {
            status: sum(
                record["execution_status"] == status for record in normalized_records
            )
            for status in sorted(
                {record["execution_status"] for record in normalized_records}
            )
        },
        "world_reliability_gate": world_gate,
        "typed_layer_quality_guardrails": typed_layer_quality_guardrails,
        "DOE_cell_coverage": {
            "cells": doe_cells,
            "pass": doe_coverage_pass,
        },
        "reported_completeness_diagnostics": diagnostics,
        "result_records": normalized_records,
        "gate_checks": gate_checks,
        "block_certified": bool(all(gate_checks.values())),
    }
    if validated_binding is not None:
        output.update(validated_binding)
        output["formal_candidate_and_global_freeze_binding_validated"] = True
    output["block_validation_payload_sha256"] = _canonical_sha256(output)
    return output


def score_layered_state_evidence(
    feature_evidence: pd.DataFrame,
    typed_hyperedges: pd.DataFrame,
    direct_candidates: pd.DataFrame,
    role_assignments: pd.DataFrame,
    truth_edges: pd.DataFrame,
    world_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    required_feature = {
        "feature",
        "modality",
        "n_views",
        "formal_test_eligible",
        "empirical_null_p",
        "reference_conditional_wald",
        "conditional_wald_view_support",
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p",
        "assay_participation_single_feature_p",
        "assay_participation_single_feature_witness_view_count",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_omnibus_route_count",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "assay_participation_winning_route",
        "assay_participation_selection_route",
        "assay_participation_selected",
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_multiview_replication_gate",
        "assay_participation_reporter",
        "confirmed_depth_orthogonal_state_response",
        "upstream_conditional_state_candidate",
        "strict_state_member",
        "state_equivalence_only_member",
        "state_supported_member",
        "state_membership_tier",
        "state_equivalence_maxT_threshold",
        "state_equivalence_basis",
        "exact_cross_assay_pair_claimed",
        "direct_causal_claimed",
        "strict_membership_changed",
        "empirical_null_calibration_active",
        "empirical_null_calibration_candidate",
        "empirical_null_marginal_state_z",
        "empirical_null_conditional_excess_z",
        "empirical_null_composite_score",
        "empirical_null_selection_threshold",
        "empirical_null_state_member",
        "state_calibrated_supported_member",
        "state_calibrated_membership_tier",
        "empirical_null_calibration_schema",
        "empirical_null_threshold_method",
        "multievidence_calibration_active",
        "multievidence_calibration_candidate",
        "multievidence_state_specific_gate",
        "multievidence_component_z__oof_marginal_state_fit",
        "multievidence_component_z__oof_joint_fit",
        "multievidence_component_z__conditional_wald_above_null_95pct",
        "multievidence_component_z__median_conditional_score_norm",
        "multievidence_route_z__joint_conditional_magnitude",
        "multievidence_route_z__joint_conditional_excess",
        "multievidence_route_z__marginal_conditional_magnitude",
        "multievidence_winning_route",
        "multievidence_family_score",
        "multievidence_selection_threshold",
        "multievidence_empirical_null_state_member",
        "state_multievidence_supported_member",
        "state_multievidence_membership_tier",
        "multievidence_calibration_schema",
        "multievidence_threshold_method",
        "state_v426_supported_member",
        "state_v426_membership_tier",
        "assay_local_loading_group_support",
        "within_assay_direct_candidate_degree",
        "within_assay_direct_candidate_overlap",
        "topology_consensus_calibration_active",
        "topology_consensus_calibration_candidate",
        "topology_consensus_state_specific_gate",
        "topology_consensus_residual_structure_support",
        "topology_consensus_fixed_rule_eligible",
        "topology_consensus_empirical_null_certified",
        "topology_consensus_state_member",
        "state_v427_supported_member",
        "state_v427_membership_tier",
        "topology_consensus_calibration_schema",
        "topology_consensus_null_control_method",
        "layer_union_is_state_membership",
        "assay_participation_implies_feature_membership",
        "cross_assay_pair_claimed",
        "direct_cross_assay_effect_claimed",
    }
    missing = sorted(required_feature.difference(feature_evidence.columns))
    if missing:
        raise ValueError(f"Layered feature evidence is missing columns: {missing}")
    extra = sorted(set(feature_evidence.columns).difference(LAYERED_FEATURE_COLUMNS))
    if extra:
        raise ValueError(
            f"Layered feature evidence has unregistered columns: {extra}"
        )
    if tuple(feature_evidence.columns) != LAYERED_FEATURE_COLUMNS:
        raise ValueError(
            "Layered feature evidence columns are not in the frozen v4 order"
        )
    required_roles = {"node", "mark", "role"}
    if missing := sorted(required_roles.difference(role_assignments.columns)):
        raise ValueError(f"Role assignments are missing columns: {missing}")
    required_edges = {"node1", "node2", "truth_type", "parent", "child"}
    if missing := sorted(required_edges.difference(truth_edges.columns)):
        raise ValueError(f"Truth edges are missing columns: {missing}")
    required_direct = {
        "node1", "endpoint1", "node2", "endpoint2", "edge_type", "source_assay"
    }
    if missing := sorted(required_direct.difference(direct_candidates.columns)):
        raise ValueError(f"Direct candidates are missing columns: {missing}")
    required_hyperedges = {
        "relation_id",
        "edge_type",
        "members",
        "modalities",
        "n_modalities",
        "n_members",
        "selection_status",
        "conditional_on_common_state_model",
        "hypothesis_only",
        "direct_causal_claimed",
        "direction_claimed",
        "exact_axis_pairing_claimed",
        "strict_members",
        "equivalence_only_members",
        "pairwise_expansion_permitted",
        "exact_cross_assay_pair_claimed",
        "loading_group_v431_guard_route",
        "loading_group_v431_guard_status",
        "loading_group_v431_retained",
        "structure_equivalence_schema",
        "structure_equivalence_fixed_rule",
        "structure_equivalence_hypothesis_type",
        "structure_equivalence_hypothesis_status",
        "nominal_members",
        "parent_equivalence_members",
        "counterpart_equivalence_members",
        "candidate_pairs_json",
        "candidate_pair_count",
        "exact_loading_group_selected",
        "structure_equivalence_selected",
        "exact_pair_identified",
        "shared_state_hypothesis_supported",
        "candidate_pairs_may_be_expanded_as_edges",
        "loading_group_guard_route",
        "loading_group_guard_status",
        "loading_group_v429_guard_route",
        "loading_group_v429_guard_status",
        "loading_group_v429_retained",
        "loading_group_v430_guard_route",
        "loading_group_v430_guard_status",
        "loading_group_v430_retained",
        "minimum_member_centroid_cosine",
        "minimum_state_route_member_centroid_cosine",
        "passes_loading_uncertainty_floor",
        "complete_direct_collider_overlap",
        "direct_collider_child_overlap",
        "shared_direct_parent_children",
        "cross_assay_axis_id",
        "cross_assay_axis_modalities",
        "cross_assay_axis_n_groups",
        "cross_assay_axis_n_modalities",
        "cross_assay_axis_confidence",
        "group_axis_lower_cosine",
        "group_axis_view_support",
        "group_axis_relative_margin",
        "competing_parent_observability_required",
        "complete_two_parent_comparison_available",
        "selected_parent_candidate",
        "competing_parent_candidate",
        "noncollider_sibling_candidate",
        "collider_child_candidate",
        "competition_selection_stage",
        "competing_parent_formal_test_eligible",
        "competing_parent_oof_marginal_state_fit",
        "competing_parent_oof_partial_state_fit",
        "competing_parent_direct_candidate_degree",
        "competing_parent_state_specific_gate",
        "competing_parent_residual_structure_support",
        "competing_parent_fixed_rule_eligible",
        "passes_competing_parent_observability_guard",
    }
    if missing := sorted(required_hyperedges.difference(typed_hyperedges.columns)):
        raise ValueError(f"Typed hyperedges are missing columns: {missing}")

    features = feature_evidence.copy()
    features["feature"] = features["feature"].astype(str)
    roles = role_assignments.copy()
    roles["node"] = roles["node"].astype(str)
    roles["role"] = roles["role"].astype(str)
    if features["feature"].duplicated().any() or roles["node"].duplicated().any():
        raise ValueError("Feature and role identifiers must be unique")

    feature_boolean_columns = (
        "formal_test_eligible",
        "assay_participation_selected",
        "assay_participation_multiview_replication_gate",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_reporter",
        "confirmed_depth_orthogonal_state_response",
        "upstream_conditional_state_candidate",
        "strict_state_member",
        "state_equivalence_only_member",
        "state_supported_member",
        "exact_cross_assay_pair_claimed",
        "direct_causal_claimed",
        "strict_membership_changed",
        "empirical_null_calibration_active",
        "empirical_null_calibration_candidate",
        "empirical_null_state_member",
        "state_calibrated_supported_member",
        "multievidence_calibration_active",
        "multievidence_calibration_candidate",
        "multievidence_state_specific_gate",
        "multievidence_empirical_null_state_member",
        "state_multievidence_supported_member",
        "state_v426_supported_member",
        "assay_local_loading_group_support",
        "within_assay_direct_candidate_overlap",
        "topology_consensus_calibration_active",
        "topology_consensus_calibration_candidate",
        "topology_consensus_state_specific_gate",
        "topology_consensus_residual_structure_support",
        "topology_consensus_fixed_rule_eligible",
        "topology_consensus_empirical_null_certified",
        "topology_consensus_state_member",
        "state_v427_supported_member",
        "layer_union_is_state_membership",
        "assay_participation_implies_feature_membership",
        "cross_assay_pair_claimed",
        "direct_cross_assay_effect_claimed",
    )
    for column in feature_boolean_columns:
        features[column] = features[column].map(
            lambda value, label=column: _strict_bool(value, label=label)
        )
    for column in (
        "empirical_null_p",
        "conditional_wald_view_support",
        "state_equivalence_maxT_threshold",
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p",
        "assay_participation_single_feature_p",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
    ):
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not features[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must lie in [0, 1]")
    if not np.allclose(
        features["state_equivalence_maxT_threshold"].to_numpy(dtype=float),
        0.01,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("State-equivalence maxT threshold differs from 0.01")

    confirmed_membership = features[
        "confirmed_depth_orthogonal_state_response"
    ].astype(bool)
    if not features["strict_state_member"].astype(bool).equals(
        confirmed_membership
    ):
        raise ValueError("State-equivalence tier changed strict membership")
    redundancy_demoted = features["redundancy_demoted"].map(
        lambda value: _strict_bool(value, label="redundancy_demoted")
    )
    expected_equivalence = (
        ~confirmed_membership
        & features["pre_redundancy_selection_status"].astype(str).eq("selected")
        & redundancy_demoted
        & pd.to_numeric(features["feature_global_maxT_p"], errors="raise").le(
            0.01
        )
    )
    observed_equivalence = features[
        "state_equivalence_only_member"
    ].astype(bool)
    if not observed_equivalence.equals(expected_equivalence):
        raise ValueError("State-equivalence membership is not reproducible")
    expected_supported = confirmed_membership | expected_equivalence
    if not features["state_supported_member"].astype(bool).equals(
        expected_supported
    ):
        raise ValueError("State-supported membership is not the tier union")
    expected_tier = pd.Series("unresolved", index=features.index, dtype=object)
    expected_tier.loc[expected_equivalence] = (
        "within_assay_state_equivalence_member"
    )
    expected_tier.loc[confirmed_membership] = "strict_state_member"
    if not features["state_membership_tier"].astype(str).equals(expected_tier):
        raise ValueError("State-membership tier labels are inconsistent")
    expected_basis = pd.Series("", index=features.index, dtype=object)
    expected_basis.loc[expected_equivalence] = (
        "pre_redundancy_selected;redundancy_demoted;world_maxT_strong_core"
    )
    if not features["state_equivalence_basis"].astype(str).equals(
        expected_basis
    ):
        raise ValueError("State-equivalence evidence basis is inconsistent")
    for column in (
        "exact_cross_assay_pair_claimed",
        "direct_causal_claimed",
        "strict_membership_changed",
    ):
        if features[column].astype(bool).any():
            raise ValueError(f"Unsupported state-equivalence claim: {column}")

    calibration_active_values = set(
        features["empirical_null_calibration_active"].astype(bool)
    )
    if len(calibration_active_values) != 1:
        raise ValueError("Empirical-null calibration activity must be world-constant")
    empirical_null_active = bool(next(iter(calibration_active_values)))
    expected_calibration_candidate = (
        features["formal_test_eligible"].astype(bool) & ~expected_supported
    )
    if not features["empirical_null_calibration_candidate"].astype(bool).equals(
        expected_calibration_candidate
    ):
        raise ValueError("Empirical-null calibration candidate set is inconsistent")
    empirical_numeric_columns = (
        "empirical_null_marginal_state_z",
        "empirical_null_conditional_excess_z",
        "empirical_null_composite_score",
        "empirical_null_selection_threshold",
    )
    for column in empirical_numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
    empirical_addition = features["empirical_null_state_member"].astype(bool)
    calibrated_supported = features[
        "state_calibrated_supported_member"
    ].astype(bool)
    if (empirical_addition & ~expected_calibration_candidate).any():
        raise ValueError("Empirical-null tier promoted a non-candidate feature")
    if not calibrated_supported.equals(expected_supported | empirical_addition):
        raise ValueError("Calibrated membership is not the exact additive union")
    expected_calibrated_tier = expected_tier.copy()
    expected_calibrated_tier.loc[empirical_addition] = empirical_null_state.TIER
    if not features["state_calibrated_membership_tier"].astype(str).equals(
        expected_calibrated_tier
    ):
        raise ValueError("Calibrated state-membership tier labels are inconsistent")
    if empirical_null_active:
        if set(features["empirical_null_calibration_schema"].astype(str)) != {
            empirical_null_state.SCHEMA
        }:
            raise ValueError("Active empirical-null layer has another schema")
        if set(features["empirical_null_threshold_method"].astype(str)) != {
            "one_source_block_maximum_gap_beyond_null_maximum"
        }:
            raise ValueError("Active empirical-null layer has another threshold method")
        threshold_values = features[
            "empirical_null_selection_threshold"
        ].to_numpy(dtype=float)
        if not np.allclose(
            threshold_values,
            threshold_values[0],
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("Empirical-null selection threshold must be world-constant")
        expected_addition = expected_calibration_candidate & features[
            "empirical_null_composite_score"
        ].ge(float(threshold_values[0]))
        if not empirical_addition.equals(expected_addition):
            raise ValueError("Empirical-null membership is not threshold-reproducible")
    else:
        if empirical_addition.any():
            raise ValueError("Inactive empirical-null calibration promoted a feature")
        if (
            features.loc[:, list(empirical_numeric_columns)]
            .to_numpy(dtype=float)
            .any()
        ):
            raise ValueError("Inactive empirical-null calibration has nonzero scores")
        if features["empirical_null_calibration_schema"].astype(str).ne("").any():
            raise ValueError("Inactive empirical-null calibration has a schema")
        if features["empirical_null_threshold_method"].astype(str).ne("").any():
            raise ValueError("Inactive empirical-null calibration has a threshold method")

    multievidence_active_values = set(
        features["multievidence_calibration_active"].astype(bool)
    )
    if len(multievidence_active_values) != 1:
        raise ValueError("Multievidence calibration activity must be world-constant")
    multievidence_active = bool(next(iter(multievidence_active_values)))
    if not features["multievidence_calibration_candidate"].astype(bool).equals(
        expected_calibration_candidate
    ):
        raise ValueError("Multievidence calibration candidate set is inconsistent")
    expected_state_specific_gate = (
        pd.to_numeric(features["oof_marginal_state_fit"], errors="raise").gt(0.0)
        & pd.to_numeric(features["oof_partial_state_fit"], errors="raise").gt(0.0)
    )
    if not features["multievidence_state_specific_gate"].astype(bool).equals(
        expected_state_specific_gate
    ):
        raise ValueError("Multievidence state-specific gate is not reproducible")
    multievidence_component_columns = tuple(
        f"multievidence_component_z__{column}"
        for column in multievidence_state.SCORE_COMPONENT_COLUMNS
    )
    multievidence_route_columns = tuple(
        f"multievidence_route_z__{route}"
        for route in multievidence_state.ROUTE_NAMES
    )
    multievidence_numeric_columns = (
        *multievidence_component_columns,
        *multievidence_route_columns,
        "multievidence_family_score",
        "multievidence_selection_threshold",
    )
    for column in multievidence_numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
    multievidence_route_values = features.loc[
        :, list(multievidence_route_columns)
    ].to_numpy(dtype=float)
    expected_family_score = np.max(multievidence_route_values, axis=1)
    if not np.allclose(
        features["multievidence_family_score"].to_numpy(dtype=float),
        expected_family_score,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("Multievidence family score is not the route maximum")
    expected_winning_route = pd.Series(
        [
            multievidence_state.ROUTE_NAMES[int(index)]
            for index in np.argmax(multievidence_route_values, axis=1)
        ],
        index=features.index,
        dtype=object,
    )
    multievidence_addition = features[
        "multievidence_empirical_null_state_member"
    ].astype(bool)
    state_multievidence_supported = features[
        "state_multievidence_supported_member"
    ].astype(bool)
    state_v426_supported = features["state_v426_supported_member"].astype(bool)
    if (
        multievidence_addition
        & ~(expected_calibration_candidate & expected_state_specific_gate)
    ).any():
        raise ValueError("Multievidence tier promoted a non-candidate or failed gate")
    if not state_multievidence_supported.equals(
        expected_supported | multievidence_addition
    ):
        raise ValueError("Multievidence base membership is not its additive union")
    expected_multievidence_tier = expected_tier.copy()
    expected_multievidence_tier.loc[multievidence_addition] = (
        multievidence_state.TIER
    )
    if not features["state_multievidence_membership_tier"].astype(str).equals(
        expected_multievidence_tier
    ):
        raise ValueError("Multievidence base tier labels are inconsistent")
    if not state_v426_supported.equals(calibrated_supported | multievidence_addition):
        raise ValueError("v4.26 membership is not the exact stacked union")
    expected_v426_tier = expected_calibrated_tier.copy()
    expected_v426_tier.loc[
        multievidence_addition & ~calibrated_supported
    ] = multievidence_state.TIER
    if not features["state_v426_membership_tier"].astype(str).equals(
        expected_v426_tier
    ):
        raise ValueError("v4.26 stacked tier labels are inconsistent")
    if multievidence_active:
        if set(features["multievidence_calibration_schema"].astype(str)) != {
            multievidence_state.SCHEMA
        }:
            raise ValueError("Active multievidence layer has another schema")
        if set(features["multievidence_threshold_method"].astype(str)) != {
            "max_over_fixed_routes_then_one_source_block_maximum_gap_beyond_null_maximum"
        }:
            raise ValueError("Active multievidence layer has another threshold method")
        if not features["multievidence_winning_route"].astype(str).equals(
            expected_winning_route
        ):
            raise ValueError("Multievidence winning route is not reproducible")
        threshold_values = features[
            "multievidence_selection_threshold"
        ].to_numpy(dtype=float)
        if not np.allclose(
            threshold_values,
            threshold_values[0],
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("Multievidence selection threshold must be world-constant")
        expected_addition = (
            expected_calibration_candidate
            & expected_state_specific_gate
            & features["multievidence_family_score"].ge(
                float(threshold_values[0])
            )
        )
        if not multievidence_addition.equals(expected_addition):
            raise ValueError("Multievidence membership is not threshold-reproducible")
    else:
        if multievidence_addition.any():
            raise ValueError("Inactive multievidence calibration promoted a feature")
        if features.loc[:, list(multievidence_numeric_columns)].to_numpy(
            dtype=float
        ).any():
            raise ValueError("Inactive multievidence calibration has nonzero scores")
        if features["multievidence_winning_route"].astype(str).ne("").any():
            raise ValueError("Inactive multievidence calibration has a winning route")
        if features["multievidence_calibration_schema"].astype(str).ne("").any():
            raise ValueError("Inactive multievidence calibration has a schema")
        if features["multievidence_threshold_method"].astype(str).ne("").any():
            raise ValueError("Inactive multievidence calibration has a threshold method")

    feature_to_modality = dict(
        zip(features["feature"], features["modality"].astype(str), strict=True)
    )
    direct_rows = direct_candidates.loc[
        direct_candidates["edge_type"].astype(str).eq(
            "within_assay_direct_candidate"
        )
    ]
    if len(direct_rows) != len(direct_candidates):
        raise ValueError("Direct-candidate table contains an unregistered edge type")
    expected_direct_degree = {feature: 0 for feature in feature_to_modality}
    for row in direct_rows.itertuples(index=False):
        left, right = str(row.node1), str(row.node2)
        unknown = sorted({left, right}.difference(feature_to_modality))
        if unknown:
            raise ValueError(
                f"Direct candidates contain unknown feature endpoints: {unknown}"
            )
        if left == right:
            raise ValueError("Within-assay direct candidates cannot be self edges")
        if feature_to_modality[left] != feature_to_modality[right]:
            raise ValueError(
                "Within-assay direct candidate joins different modalities"
            )
        if hasattr(row, "source_assay") and str(row.source_assay) != (
            feature_to_modality[left]
        ):
            raise ValueError(
                "Direct-candidate source_assay disagrees with feature modality"
            )
        expected_direct_degree[left] += 1
        expected_direct_degree[right] += 1
    observed_direct_degree = pd.to_numeric(
        features["within_assay_direct_candidate_degree"], errors="raise"
    )
    if (
        not np.isfinite(observed_direct_degree.to_numpy(dtype=float)).all()
        or (observed_direct_degree < 0).any()
        or not observed_direct_degree.eq(observed_direct_degree.round()).all()
    ):
        raise ValueError(
            "within_assay_direct_candidate_degree must contain nonnegative integers"
        )
    observed_direct_degree = observed_direct_degree.astype(int)
    reconstructed_direct_degree = features["feature"].map(
        expected_direct_degree
    ).astype(int)
    if not observed_direct_degree.equals(reconstructed_direct_degree):
        raise ValueError(
            "Within-assay direct-candidate degree does not match the edge table"
        )
    expected_direct_overlap = reconstructed_direct_degree.gt(0)
    if not features["within_assay_direct_candidate_overlap"].astype(bool).equals(
        expected_direct_overlap
    ):
        raise ValueError(
            "Within-assay direct-candidate overlap does not match endpoint degree"
        )
    features["within_assay_direct_candidate_degree"] = observed_direct_degree

    topology_active_values = set(
        features["topology_consensus_calibration_active"].astype(bool)
    )
    if len(topology_active_values) != 1:
        raise ValueError(
            "Topology-consensus calibration activity must be world-constant"
        )
    topology_active = bool(next(iter(topology_active_values)))
    expected_topology_candidate = (
        features["formal_test_eligible"].astype(bool) & ~state_v426_supported
    )
    if not features["topology_consensus_calibration_candidate"].astype(
        bool
    ).equals(expected_topology_candidate):
        raise ValueError(
            "Topology-consensus calibration candidate set is inconsistent"
        )
    expected_topology_state_gate = expected_state_specific_gate
    if not features["topology_consensus_state_specific_gate"].astype(
        bool
    ).equals(expected_topology_state_gate):
        raise ValueError("Topology-consensus state gate is not reproducible")
    expected_topology_residual_support = reconstructed_direct_degree.ge(
        topology_consensus_state.MINIMUM_DIRECT_CANDIDATE_DEGREE
    )
    if not features[
        "topology_consensus_residual_structure_support"
    ].astype(bool).equals(expected_topology_residual_support):
        raise ValueError(
            "Topology-consensus residual support is not reproducible"
        )
    expected_fixed_rule_eligible = (
        features["formal_test_eligible"].astype(bool)
        & expected_topology_state_gate
        & expected_topology_residual_support
    )
    if not features["topology_consensus_fixed_rule_eligible"].astype(
        bool
    ).equals(expected_fixed_rule_eligible):
        raise ValueError("Topology-consensus fixed rule is not reproducible")
    certified_values = set(
        features["topology_consensus_empirical_null_certified"].astype(bool)
    )
    if certified_values != {topology_active}:
        raise ValueError(
            "Topology-consensus certification must be world-constant and match activity"
        )
    topology_addition = features["topology_consensus_state_member"].astype(bool)
    expected_topology_addition = (
        topology_active
        & expected_topology_candidate
        & expected_topology_state_gate
        & expected_topology_residual_support
    )
    if not topology_addition.equals(expected_topology_addition):
        raise ValueError(
            "Topology-consensus membership is not the exact fixed conjunction"
        )
    state_v427_supported = features["state_v427_supported_member"].astype(bool)
    if not state_v427_supported.equals(state_v426_supported | topology_addition):
        raise ValueError("v4.27 membership is not the exact stacked union")
    expected_v427_tier = expected_v426_tier.copy()
    expected_v427_tier.loc[topology_addition] = topology_consensus_state.TIER
    if not features["state_v427_membership_tier"].astype(str).equals(
        expected_v427_tier
    ):
        raise ValueError("v4.27 stacked tier labels are inconsistent")
    if topology_active:
        if set(
            features["topology_consensus_calibration_schema"].astype(str)
        ) != {topology_consensus_state.SCHEMA}:
            raise ValueError("Active topology-consensus layer has another schema")
        if set(
            features["topology_consensus_null_control_method"].astype(str)
        ) != {topology_consensus_state.NULL_CONTROL_METHOD}:
            raise ValueError(
                "Active topology-consensus layer has another null-control method"
            )
    else:
        if topology_addition.any():
            raise ValueError(
                "Inactive topology-consensus calibration promoted a feature"
            )
        if features["topology_consensus_calibration_schema"].astype(
            str
        ).ne("").any():
            raise ValueError("Inactive topology-consensus calibration has a schema")
        if features["topology_consensus_null_control_method"].astype(
            str
        ).ne("").any():
            raise ValueError(
                "Inactive topology-consensus calibration has a null-control method"
            )
    features["reference_conditional_wald"] = pd.to_numeric(
        features["reference_conditional_wald"], errors="raise"
    )
    if (
        not np.isfinite(
            features["reference_conditional_wald"].to_numpy(dtype=float)
        ).all()
        or (features["reference_conditional_wald"] < 0.0).any()
    ):
        raise ValueError(
            "reference_conditional_wald must contain finite nonnegative values"
        )
    for column in (
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_single_feature_witness_view_count",
        "assay_participation_omnibus_route_count",
    ):
        values = pd.to_numeric(features[column], errors="raise")
        if (values < 0).any() or not (values == values.round()).all():
            raise ValueError(f"{column} must contain nonnegative integers")
        features[column] = values.astype(int)

    participation_constant_columns = (
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_selected",
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_multiview_replication_gate",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_omnibus_route_count",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "assay_participation_winning_route",
        "assay_participation_selection_route",
    )
    participation_expectations: dict[str, dict[str, Any]] = {}
    for modality, group in features.groupby("modality", sort=True):
        inconsistent = [
            column
            for column in participation_constant_columns
            if group[column].nunique(dropna=False) != 1
        ]
        if inconsistent:
            raise ValueError(
                "Assay participation evidence is inconsistent within modality "
                f"{modality}: {inconsistent}"
            )
        first = group.iloc[0]
        selected_assay = bool(first["assay_participation_selected"])
        n_views_values = pd.to_numeric(group["n_views"], errors="raise")
        if (
            n_views_values.nunique(dropna=False) != 1
            or int(n_views_values.iloc[0])
            != int(
                subspace_scorer.FROZEN_DISCOVERY_CONFIG[
                    "observed_model_views"
                ]
            )
        ):
            raise ValueError(f"n_views is invalid for {modality}")
        n_views = int(n_views_values.iloc[0])
        expected_order = 1 if len(group) == 1 else 2
        eligible = group.loc[group["formal_test_eligible"]].sort_values(
            ["empirical_null_p", "reference_conditional_wald", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        expected_family = len(eligible)
        expected_contributing = (
            expected_order if expected_family >= expected_order else 0
        )
        leading_reporter = (
            str(eligible.iloc[0]["feature"]) if expected_family else ""
        )
        if expected_family:
            single_order_p = float(eligible.iloc[0]["empirical_null_p"])
            single_p = min(1.0, expected_family * single_order_p)
            single_views = int(
                np.rint(
                    float(eligible.iloc[0]["conditional_wald_view_support"])
                    * n_views
                )
            )
        else:
            single_order_p = 1.0
            single_p = 1.0
            single_views = 0
        single_replication = bool(
            expected_family >= 1 and single_views == n_views
        )
        enough_partial = expected_family >= expected_order > 0
        witnesses = (
            eligible.head(expected_order)
            if enough_partial
            else eligible.iloc[0:0]
        )
        if enough_partial:
            partial_p = min(
                1.0,
                (expected_family - expected_order + 1)
                * float(witnesses.iloc[-1]["empirical_null_p"]),
            )
            minimum_views = int(
                np.rint(
                    witnesses["conditional_wald_view_support"].to_numpy(
                        dtype=float
                    )
                    * n_views
                ).min()
            )
        else:
            partial_p = 1.0
            minimum_views = 0
        partial_replication = bool(
            enough_partial and minimum_views == n_views
        )
        route_count = 1 if len(group) == 1 else 2
        live_routes = {
            "single_feature": single_p if single_replication else 1.0
        }
        if route_count == 2:
            live_routes["partial_conjunction"] = (
                partial_p if partial_replication else 1.0
            )
        if any(value < 1.0 for value in live_routes.values()):
            winning_route = min(
                live_routes,
                key=lambda route: (
                    live_routes[route],
                    0 if route == "partial_conjunction" else 1,
                ),
            )
        else:
            winning_route = "none"
        omnibus_p = min(1.0, route_count * min(live_routes.values()))
        if (
            int(first["assay_participation_order"]) != expected_order
            or int(first["assay_participation_family_size"]) != expected_family
            or int(first["assay_participation_contributing_feature_count"])
            != expected_contributing
            or int(
                first["assay_participation_minimum_witness_view_count"]
            )
            != minimum_views
            or int(
                first[
                    "assay_participation_single_feature_witness_view_count"
                ]
            )
            != single_views
            or int(first["assay_participation_omnibus_route_count"])
            != route_count
            or bool(first["assay_participation_multiview_replication_gate"])
            != partial_replication
            or bool(
                first[
                    "assay_participation_partial_conjunction_multiview_"
                    "replication_gate"
                ]
            )
            != partial_replication
            or bool(
                first[
                    "assay_participation_single_feature_multiview_"
                    "replication_gate"
                ]
            )
            != single_replication
        ):
            raise ValueError(
                f"Assay participation count/replication contract is invalid for {modality}"
            )
        for column, expected_value in (
            ("assay_participation_partial_conjunction_p", partial_p),
            (
                "assay_participation_single_feature_order_statistic_p",
                single_order_p,
            ),
            ("assay_participation_single_feature_p", single_p),
            ("assay_participation_omnibus_p", omnibus_p),
        ):
            if not np.isclose(
                float(first[column]),
                expected_value,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"Assay participation p-value is invalid for {modality}: {column}"
                )
        if str(first["assay_participation_winning_route"]) != winning_route:
            raise ValueError(
                f"Assay participation winning route is invalid for {modality}"
            )
        participation_expectations[str(modality)] = {
            "group": group,
            "partial_p": partial_p,
            "single_order_p": single_order_p,
            "single_p": single_p,
            "single_views": single_views,
            "single_replication": single_replication,
            "partial_views": minimum_views,
            "partial_replication": partial_replication,
            "route_count": route_count,
            "omnibus_p": omnibus_p,
            "winning_route": winning_route,
            "leading_reporter": leading_reporter,
            "order": expected_order,
        }

    modalities = sorted(participation_expectations)
    partial_holm = _holm(
        [participation_expectations[m]["partial_p"] for m in modalities]
    )
    omnibus_holm = _holm(
        [participation_expectations[m]["omnibus_p"] for m in modalities]
    )
    participation_records: list[dict[str, Any]] = []
    for modality, expected_partial_holm, expected_omnibus_holm in zip(
        modalities, partial_holm, omnibus_holm, strict=True
    ):
        expected = participation_expectations[modality]
        group = expected["group"]
        first = group.iloc[0]
        if not np.isclose(
            float(first["assay_participation_holm_p"]),
            expected_partial_holm,
            atol=1e-12,
            rtol=1e-10,
        ) or not np.isclose(
            float(first["assay_participation_omnibus_holm_p"]),
            expected_omnibus_holm,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"Assay participation Holm adjustment is invalid for {modality}"
            )
        expected_selected = bool(
            expected_omnibus_holm <= 0.10
            and expected["winning_route"] != "none"
        )
        selected_assay = bool(first["assay_participation_selected"])
        expected_selection_route = (
            expected["winning_route"] if expected_selected else ""
        )
        if (
            selected_assay != expected_selected
            or str(first["assay_participation_selection_route"])
            != expected_selection_route
        ):
            raise ValueError(
                f"Assay participation selection is invalid for {modality}"
            )
        reporter_rows = group.loc[group["assay_participation_reporter"]]
        expected_reporter = (
            expected["leading_reporter"] if expected_selected else ""
        )
        observed_reporters = list(reporter_rows["feature"].astype(str))
        if observed_reporters != ([expected_reporter] if expected_reporter else []):
            raise ValueError(
                f"Assay participation reporter is invalid for {modality}"
            )
        participation_records.append(
            {
                "modality": str(modality),
                "selected": selected_assay,
                "partial_conjunction_p": float(
                    first["assay_participation_partial_conjunction_p"]
                ),
                "holm_p": float(first["assay_participation_holm_p"]),
                "single_feature_order_statistic_p": expected[
                    "single_order_p"
                ],
                "single_feature_p": expected["single_p"],
                "single_feature_witness_view_count": expected["single_views"],
                "single_feature_multiview_replication_gate": expected[
                    "single_replication"
                ],
                "partial_conjunction_multiview_replication_gate": expected[
                    "partial_replication"
                ],
                "omnibus_route_count": expected["route_count"],
                "omnibus_p": expected["omnibus_p"],
                "omnibus_holm_p": float(expected_omnibus_holm),
                "winning_route": expected["winning_route"],
                "selection_route": expected_selection_route,
                "order": expected["order"],
                "minimum_witness_view_count": expected["partial_views"],
                "reporter": expected_reporter,
            }
        )
    participation = pd.DataFrame(participation_records)

    gene = str(world_parameters.get("gene", "")).strip()
    if not gene:
        raise ValueError("world_parameters must identify the simulated RNA gene")
    role_map = dict(zip(roles["node"], roles["role"]))
    role_map[gene] = "GENE"
    expected_feature_set = set(role_map)
    observed_feature_set = set(features["feature"])
    if observed_feature_set != expected_feature_set:
        raise ValueError(
            "Layered feature universe disagrees with scorer truth; "
            f"missing={sorted(expected_feature_set-observed_feature_set)}, "
            f"extra={sorted(observed_feature_set-expected_feature_set)}"
        )

    positives = {
        feature
        for feature, role in role_map.items()
        if role in STATE_ROLES or role == "GENE"
    }
    sentinels = {
        feature for feature, role in role_map.items() if role in NULL_ROLES
    }

    def selected(column: str) -> set[str]:
        return set(features.loc[features[column].astype(bool), "feature"])

    confirmed = selected("confirmed_depth_orthogonal_state_response")
    conditional = selected("upstream_conditional_state_candidate")
    equivalence_only = selected("state_equivalence_only_member")
    state_supported = selected("state_supported_member")
    empirical_null_addition = selected("empirical_null_state_member")
    state_calibrated_supported = selected("state_calibrated_supported_member")
    multievidence_addition_set = selected(
        "multievidence_empirical_null_state_member"
    )
    state_v426_supported_set = selected("state_v426_supported_member")
    topology_consensus_addition_set = selected(
        "topology_consensus_state_member"
    )
    state_v427_supported_set = selected("state_v427_supported_member")
    local_group = selected("assay_local_loading_group_support")
    direct_nodes = selected("within_assay_direct_candidate_overlap")
    any_layer = state_v427_supported_set | local_group | direct_nodes

    set_scores = {
        "confirmed_state_response": _set_metrics(confirmed, positives, sentinels),
        "upstream_conditional_candidate": _set_metrics(
            conditional, positives, sentinels
        ),
        "state_equivalence_only": _set_metrics(
            equivalence_only, positives, sentinels
        ),
        "state_equivalence_supported": _set_metrics(
            state_supported, positives, sentinels
        ),
        "empirical_null_state_only": _set_metrics(
            empirical_null_addition, positives, sentinels
        ),
        "state_calibrated_supported": _set_metrics(
            state_calibrated_supported, positives, sentinels
        ),
        "multievidence_state_only": _set_metrics(
            multievidence_addition_set, positives, sentinels
        ),
        "state_v426_supported": _set_metrics(
            state_v426_supported_set, positives, sentinels
        ),
        "topology_consensus_state_only": _set_metrics(
            topology_consensus_addition_set, positives, sentinels
        ),
        "state_v427_supported": _set_metrics(
            state_v427_supported_set, positives, sentinels
        ),
        "assay_local_loading_support": _set_metrics(
            local_group, positives, sentinels
        ),
        "within_assay_direct_endpoint": _set_metrics(
            direct_nodes, positives, sentinels
        ),
        "descriptive_any_layer_feature_coverage": _set_metrics(
            any_layer, positives, sentinels
        ),
    }

    confirmed_frame = features.loc[features["feature"].isin(confirmed)]
    confirmed_modalities = sorted(set(confirmed_frame["modality"].astype(str)))
    participating_modalities = sorted(
        participation.loc[participation["selected"], "modality"].astype(str)
    )
    chrom_assays = sorted(
        modality
        for modality in set(features["modality"].astype(str))
        if modality.startswith("chrom_")
    )
    per_assay_recall: dict[str, float] = {}
    for assay in chrom_assays:
        assay_features = set(
            features.loc[features["modality"].astype(str).eq(assay), "feature"]
        )
        expected = assay_features & positives
        per_assay_recall[assay] = _rate(len(confirmed & expected), len(expected))
    per_assay_recall["rna"] = float(gene in confirmed)
    role_recall = {
        role: _rate(
            len(confirmed & {f for f, r in role_map.items() if r == role}),
            len({f for f, r in role_map.items() if r == role}),
        )
        for role in sorted(STATE_ROLES)
    }

    predicted_pairs = {
        frozenset((str(row.node1), str(row.node2)))
        for row in direct_candidates.itertuples(index=False)
        if str(row.edge_type) == "within_assay_direct_candidate"
    }
    true_direct = truth_edges.loc[
        truth_edges["truth_type"].astype(str).eq("within_assay_direct")
    ]
    true_pairs = {
        frozenset((str(row.node1), str(row.node2)))
        for row in true_direct.itertuples(index=False)
    }
    true_oriented = {
        (str(row.parent), str(row.child)) for row in true_direct.itertuples(index=False)
    }
    predicted_oriented: set[tuple[str, str]] = set()
    for row in direct_candidates.itertuples(index=False):
        if str(row.edge_type) != "within_assay_direct_candidate":
            continue
        if str(row.endpoint1).upper() == "TAIL" and str(row.endpoint2).upper() == "ARROW":
            predicted_oriented.add((str(row.node1), str(row.node2)))
        elif str(row.endpoint2).upper() == "TAIL" and str(row.endpoint1).upper() == "ARROW":
            predicted_oriented.add((str(row.node2), str(row.node1)))
    true_predicted_pairs = predicted_pairs & true_pairs
    direct_score = {
        "n_predicted": len(predicted_pairs),
        "n_true_predicted": len(true_predicted_pairs),
        "n_false_predicted": len(predicted_pairs - true_pairs),
        "skeleton_precision": _rate(len(true_predicted_pairs), len(predicted_pairs)),
        "skeleton_recall": _rate(len(true_predicted_pairs), len(true_pairs)),
        "n_correctly_oriented": len(predicted_oriented & true_oriented),
        "n_strict_reverse": len(
            {(child, parent) for parent, child in predicted_oriented} & true_oriented
        ),
    }
    collider_parents: dict[tuple[str, str], set[str]] = {}
    for row in direct_candidates.itertuples(index=False):
        if str(row.edge_type) != "within_assay_direct_candidate":
            continue
        endpoint1 = str(row.endpoint1).upper()
        endpoint2 = str(row.endpoint2).upper()
        if endpoint1 == "TAIL" and endpoint2 == "ARROW":
            parent, child = str(row.node1), str(row.node2)
        elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
            parent, child = str(row.node2), str(row.node1)
        else:
            raise ValueError(
                "Within-assay direct candidates must have one TAIL and one ARROW"
            )
        collider_parents.setdefault(
            (str(row.source_assay), child), set()
        ).add(parent)
    complete_colliders = {
        key: parents
        for key, parents in collider_parents.items()
        if len(parents) >= 2
    }

    all_local_hyperedges = typed_hyperedges.loc[
        typed_hyperedges["relation_id"].astype(str).str.startswith(
            "assay_local__"
        )
    ].copy()
    if all_local_hyperedges["relation_id"].astype(str).duplicated().any():
        raise ValueError("Assay-local loading hyperedges have duplicate IDs")
    feature_modalities = features.set_index("feature")["modality"].astype(str)
    feature_lookup = features.set_index("feature", drop=False)
    pre_group_map: dict[str, set[str]] = {}
    selected_group_map: dict[str, set[str]] = {}
    selected_state_group_map: dict[str, set[str]] = {}
    selected_axis_group_map: dict[str, set[str]] = {}
    rejected_group_map: dict[str, set[str]] = {}
    loading_guard_safe = True
    state_route_group_ids: set[str] = set()
    axis_route_group_ids: set[str] = set()
    v429_state_route_group_ids: set[str] = set()
    v429_axis_route_group_ids: set[str] = set()
    v430_state_route_group_ids: set[str] = set()
    v430_axis_route_group_ids: set[str] = set()
    structure_overlap_downgrade_ids: set[str] = set()
    uncertainty_floor_downgrade_ids: set[str] = set()
    competitor_observability_downgrade_ids: set[str] = set()
    competitor_comparison_unavailable_ids: set[str] = set()
    for row in all_local_hyperedges.itertuples(index=False):
        relation_id = str(row.relation_id)
        if not relation_id.startswith("assay_local__"):
            raise ValueError("Assay-local loading relation ID is malformed")
        group_id = relation_id.removeprefix("assay_local__")
        hyperedge_members = sorted(
            member.strip()
            for member in set(str(row.members).split(";"))
            if member.strip()
        )
        members = sorted(
            member.strip()
            for member in set(str(row.nominal_members).split(";"))
            if member.strip()
        )
        if len(hyperedge_members) < 2 or len(members) != 2:
            raise ValueError("Guarded assay-local hyperedge has fewer than two members")
        if int(row.n_members) != len(hyperedge_members) or int(row.n_modalities) != 1:
            raise ValueError("Guarded assay-local hyperedge cardinality is invalid")
        modalities = {
            str(feature_modalities.loc[member]) for member in hyperedge_members
        }
        if len(modalities) != 1 or str(row.modalities) not in modalities:
            raise ValueError("Guarded assay-local hyperedge crosses assays")
        for member in members:
            pre_group_map.setdefault(member, set()).add(group_id)
        member_set = set(members)
        state_subset = member_set.issubset(state_v427_supported_set)
        has_state_member = bool(member_set & state_v427_supported_set)
        route = str(row.loading_group_guard_route)
        guard_status = str(row.loading_group_guard_status)
        v429_route = str(row.loading_group_v429_guard_route)
        v429_retained = _strict_bool(
            row.loading_group_v429_retained,
            label="loading-group v4.29 retained",
        )
        v430_route = str(row.loading_group_v430_guard_route)
        v430_guard_status = str(row.loading_group_v430_guard_status)
        v430_retained = _strict_bool(
            row.loading_group_v430_retained,
            label="loading-group v4.30 retained",
        )
        axis_modalities = {
            value.strip()
            for value in str(row.cross_assay_axis_modalities).split(";")
            if value.strip() and value.strip().lower() != "nan"
        }
        v429_state_route = v429_route == loading_group_state_guard.STATE_ROUTE
        v429_axis_route = v429_route == loading_group_state_guard.AXIS_ROUTE
        v429_unresolved_route = v429_route == "unresolved"
        state_route = route == loading_group_state_guard.STATE_ROUTE
        axis_route = route == loading_group_state_guard.AXIS_ROUTE
        unresolved_route = route == "unresolved"
        if v429_state_route:
            v429_state_route_group_ids.add(group_id)
        if v429_axis_route:
            v429_axis_route_group_ids.add(group_id)
        axis_evidence_safe = bool(
            str(row.cross_assay_axis_id).strip()
            and str(row.cross_assay_axis_id).strip().lower() != "nan"
            and int(float(row.cross_assay_axis_n_groups)) >= 2
            and int(float(row.cross_assay_axis_n_modalities)) >= 2
            and len(axis_modalities) >= 2
            and str(row.modalities) in axis_modalities
            and 0.0 <= float(row.cross_assay_axis_confidence) <= 1.0
            and 0.0 < float(row.group_axis_lower_cosine) <= 1.0
            and 0.0 < float(row.group_axis_view_support) <= 1.0
            and float(row.group_axis_relative_margin) > 0.0
        )
        v429_state_route_safe = bool(
            v429_state_route
            and state_subset
            and not axis_modalities
            and (
                not str(row.cross_assay_axis_id).strip()
                or str(row.cross_assay_axis_id).strip().lower() == "nan"
            )
        )
        v429_axis_route_safe = bool(
            v429_axis_route
            and not state_subset
            and has_state_member
            and axis_evidence_safe
        )
        v429_unresolved_safe = bool(
            v429_unresolved_route
            and not state_subset
            and not axis_evidence_safe
        )
        v429_should_select = v429_state_route_safe or v429_axis_route_safe

        minimum_cosine = float(row.minimum_member_centroid_cosine)
        frozen_floor = float(
            loading_group_structure_guard
            .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        )
        passes_floor = minimum_cosine >= frozen_floor
        observed_passes_floor = _strict_bool(
            row.passes_loading_uncertainty_floor,
            label="loading-group uncertainty floor",
        )
        observed_floor = float(row.minimum_state_route_member_centroid_cosine)

        direct_child_overlap: set[str] = set()
        shared_parent_children: set[str] = set()
        for (assay, child), parents in complete_colliders.items():
            if assay != str(row.modalities):
                continue
            if child in member_set:
                direct_child_overlap.add(child)
            if len(member_set.intersection(parents)) >= 2:
                shared_parent_children.add(child)
        direct_overlap = bool(
            direct_child_overlap or shared_parent_children
        )
        observed_direct_overlap = _strict_bool(
            row.complete_direct_collider_overlap,
            label="loading-group direct collider overlap",
        )

        v430_should_select = bool(
            v429_retained
            and not direct_overlap
            and (
                v429_axis_route
                or (v429_state_route and passes_floor)
            )
        )
        if v430_should_select and v429_state_route:
            expected_v430_guard_status = (
                loading_group_structure_guard.STATE_RETAINED_STATUS
            )
        elif v430_should_select and v429_axis_route:
            expected_v430_guard_status = (
                loading_group_structure_guard.AXIS_RETAINED_STATUS
            )
        elif not v429_retained:
            expected_v430_guard_status = (
                loading_group_structure_guard.UPSTREAM_REJECTED_STATUS
            )
        elif direct_overlap:
            expected_v430_guard_status = (
                loading_group_structure_guard.DIRECT_OVERLAP_STATUS
            )
            structure_overlap_downgrade_ids.add(group_id)
        else:
            expected_v430_guard_status = (
                loading_group_structure_guard.LOW_CONFIDENCE_STATUS
            )
            uncertainty_floor_downgrade_ids.add(group_id)
        expected_v430_route = (
            v429_route if v430_should_select else "unresolved"
        )
        if v430_should_select and v429_state_route:
            v430_state_route_group_ids.add(group_id)
        if v430_should_select and v429_axis_route:
            v430_axis_route_group_ids.add(group_id)

        comparison_required = bool(
            v430_retained
            and v430_route == loading_group_state_guard.STATE_ROUTE
        )
        observed_comparison_required = _strict_bool(
            row.competing_parent_observability_required,
            label="loading-group competing-parent requirement",
        )
        comparison_available = _strict_bool(
            row.complete_two_parent_comparison_available,
            label="loading-group two-parent comparison availability",
        )
        selected_parent = _optional_text(row.selected_parent_candidate)
        competing_parent = _optional_text(row.competing_parent_candidate)
        noncollider_sibling = _optional_text(
            row.noncollider_sibling_candidate
        )
        collider_child = _optional_text(row.collider_child_candidate)
        competition_stage = _optional_text(row.competition_selection_stage)
        observed_competitor_formal = _strict_bool(
            row.competing_parent_formal_test_eligible,
            label="competing-parent formal eligibility",
        )
        observed_competitor_state_gate = _strict_bool(
            row.competing_parent_state_specific_gate,
            label="competing-parent state gate",
        )
        observed_competitor_residual_gate = _strict_bool(
            row.competing_parent_residual_structure_support,
            label="competing-parent residual-structure gate",
        )
        observed_competitor_fixed = _strict_bool(
            row.competing_parent_fixed_rule_eligible,
            label="competing-parent fixed-rule eligibility",
        )
        observed_passes_competitor_guard = _strict_bool(
            row.passes_competing_parent_observability_guard,
            label="competing-parent guard result",
        )
        observed_competitor_marginal = _optional_float(
            row.competing_parent_oof_marginal_state_fit,
            label="competing-parent marginal state fit",
        )
        observed_competitor_partial = _optional_float(
            row.competing_parent_oof_partial_state_fit,
            label="competing-parent partial state fit",
        )
        observed_competitor_degree = int(
            float(row.competing_parent_direct_candidate_degree)
        )

        competitor_fixed = False
        comparison_record_safe = False
        if comparison_available and competing_parent in feature_lookup.index:
            competitor = feature_lookup.loc[competing_parent]
            competitor_formal = bool(competitor["formal_test_eligible"])
            competitor_state_gate = bool(
                competitor["topology_consensus_state_specific_gate"]
            )
            competitor_residual_gate = bool(
                competitor[
                    "topology_consensus_residual_structure_support"
                ]
            )
            competitor_fixed = bool(
                competitor["topology_consensus_fixed_rule_eligible"]
            )
            competitor_marginal = float(
                competitor["oof_marginal_state_fit"]
            )
            competitor_partial = float(
                competitor["oof_partial_state_fit"]
            )
            competitor_degree = int(
                competitor["within_assay_direct_candidate_degree"]
            )
            comparison_record_safe = bool(
                {selected_parent, noncollider_sibling} == member_set
                and len(
                    {
                        selected_parent,
                        noncollider_sibling,
                        competing_parent,
                        collider_child,
                    }
                )
                == 4
                and bool(competition_stage)
                and str(competitor["modality"]) == str(row.modalities)
                and observed_competitor_formal == competitor_formal
                and observed_competitor_state_gate == competitor_state_gate
                and observed_competitor_residual_gate
                == competitor_residual_gate
                and observed_competitor_fixed == competitor_fixed
                and np.isclose(
                    observed_competitor_marginal, competitor_marginal
                )
                and np.isclose(
                    observed_competitor_partial, competitor_partial
                )
                and observed_competitor_degree == competitor_degree
            )
        elif not comparison_available:
            comparison_record_safe = bool(
                not any(
                    (
                        selected_parent,
                        competing_parent,
                        noncollider_sibling,
                        collider_child,
                        competition_stage,
                    )
                )
                and not observed_competitor_formal
                and not observed_competitor_state_gate
                and not observed_competitor_residual_gate
                and not observed_competitor_fixed
                and np.isnan(observed_competitor_marginal)
                and np.isnan(observed_competitor_partial)
                and observed_competitor_degree == 0
            )

        passes_competitor_guard = bool(
            not comparison_required
            or (comparison_available and competitor_fixed)
        )
        should_select = bool(v430_retained and passes_competitor_guard)
        if should_select and v430_route == loading_group_state_guard.STATE_ROUTE:
            expected_guard_status = (
                loading_group_competitor_guard.STATE_RETAINED_STATUS
            )
            state_route_group_ids.add(group_id)
        elif should_select and v430_route == loading_group_state_guard.AXIS_ROUTE:
            expected_guard_status = (
                loading_group_competitor_guard.AXIS_RETAINED_STATUS
            )
            axis_route_group_ids.add(group_id)
        elif not v430_retained:
            expected_guard_status = (
                loading_group_competitor_guard.UPSTREAM_REJECTED_STATUS
            )
        elif not comparison_available:
            expected_guard_status = (
                loading_group_competitor_guard.COMPARISON_UNAVAILABLE_STATUS
            )
            competitor_comparison_unavailable_ids.add(group_id)
        else:
            expected_guard_status = (
                loading_group_competitor_guard.COMPETITOR_UNOBSERVABLE_STATUS
            )
            competitor_observability_downgrade_ids.add(group_id)
        expected_route = v430_route if should_select else "unresolved"
        expected_equivalence = (
            loading_group_structure_equivalence
            .build_loading_group_structure_equivalence(
                pd.DataFrame(
                    [
                        {
                            "loading_group": group_id,
                            "modality": str(row.modalities),
                            "members": ";".join(members),
                            "retained": should_select,
                            "guard_route": expected_route,
                            "guard_status": expected_guard_status,
                            "complete_two_parent_comparison_available": (
                                comparison_available
                            ),
                            "selected_parent_candidate": selected_parent,
                            "competing_parent_candidate": competing_parent,
                            "noncollider_sibling_candidate": noncollider_sibling,
                            "collider_child_candidate": collider_child,
                            "passes_competing_parent_observability_guard": (
                                passes_competitor_guard
                            ),
                        }
                    ]
                )
            ).decisions.iloc[0]
        )
        expected_structure_equivalence = bool(
            expected_equivalence["structure_equivalence_selected"]
        )
        expected_exact_group = bool(
            expected_equivalence["exact_loading_group_selected"]
        )
        expected_hyperedge_members = (
            {
                selected_parent,
                competing_parent,
                noncollider_sibling,
                collider_child,
            }
            if expected_structure_equivalence
            else set(members)
        )
        expected_edge_type = (
            loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE
            if expected_structure_equivalence
            else "within_assay_shared_state_support"
        )
        observed_status = str(row.selection_status)
        expected_status = "selected" if should_select else "unresolved"
        expected_layer = (
            "assay_local_state_structure_equivalence"
            if expected_structure_equivalence
            else (
                "assay_local_loading_group_support"
                if should_select
                else "unresolved"
            )
        )
        loading_guard_safe &= bool(
            observed_status == expected_status
            and str(row.edge_type) == expected_edge_type
            and str(row.layer) == expected_layer
            and set(hyperedge_members) == expected_hyperedge_members
            and route == expected_route
            and guard_status == expected_guard_status
            and str(row.loading_group_v431_guard_route) == expected_route
            and str(row.loading_group_v431_guard_status)
            == expected_guard_status
            and _strict_bool(
                row.loading_group_v431_retained,
                label="loading-group v4.31 retained",
            )
            == should_select
            and str(row.structure_equivalence_schema)
            == loading_group_structure_equivalence.SCHEMA
            and str(row.structure_equivalence_fixed_rule)
            == loading_group_structure_equivalence.FIXED_RULE
            and str(row.structure_equivalence_hypothesis_type)
            == str(expected_equivalence["hypothesis_type"])
            and str(row.structure_equivalence_hypothesis_status)
            == str(expected_equivalence["hypothesis_status"])
            and str(row.nominal_members) == ";".join(members)
            and str(row.parent_equivalence_members)
            == str(expected_equivalence["parent_equivalence_members"])
            and str(row.counterpart_equivalence_members)
            == str(expected_equivalence["counterpart_equivalence_members"])
            and str(row.candidate_pairs_json)
            == str(expected_equivalence["candidate_pairs_json"])
            and int(float(row.candidate_pair_count))
            == int(expected_equivalence["candidate_pair_count"])
            and _strict_bool(
                row.exact_loading_group_selected,
                label="loading-group exact selection",
            )
            == expected_exact_group
            and _strict_bool(
                row.structure_equivalence_selected,
                label="loading-group structure-equivalence selection",
            )
            == expected_structure_equivalence
            and _strict_bool(
                row.exact_pair_identified,
                label="loading-group exact-pair identification",
            )
            == bool(expected_equivalence["exact_pair_identified"])
            and _strict_bool(
                row.shared_state_hypothesis_supported,
                label="loading-group shared-state hypothesis support",
            )
            == should_select
            and not _strict_bool(
                row.candidate_pairs_may_be_expanded_as_edges,
                label="loading-group candidate-pair expansion",
            )
            and v430_retained == v430_should_select
            and v430_route == expected_v430_route
            and v430_guard_status == expected_v430_guard_status
            and v429_retained == v429_should_select
            and (v429_should_select or v429_unresolved_safe)
            and np.isclose(observed_floor, frozen_floor)
            and observed_passes_floor == passes_floor
            and observed_direct_overlap == direct_overlap
            and str(row.direct_collider_child_overlap)
            == ";".join(sorted(direct_child_overlap))
            and str(row.shared_direct_parent_children)
            == ";".join(sorted(shared_parent_children))
            and observed_comparison_required == comparison_required
            and comparison_record_safe
            and observed_passes_competitor_guard
            == passes_competitor_guard
            and _strict_bool(
                row.conditional_on_common_state_model,
                label="loading-group conditional_on_common_state_model",
            )
            == should_select
            and _strict_bool(
                row.hypothesis_only,
                label="loading-group hypothesis_only",
            )
            and not _strict_bool(
                row.direct_causal_claimed,
                label="loading-group direct_causal_claimed",
            )
            and not _strict_bool(
                row.direction_claimed,
                label="loading-group direction_claimed",
            )
            and not _strict_bool(
                row.exact_cross_assay_pair_claimed,
                label="loading-group exact_cross_assay_pair_claimed",
            )
            and not _strict_bool(
                row.pairwise_expansion_permitted,
                label="loading-group pairwise_expansion_permitted",
            )
            and (should_select or unresolved_route)
        )
        target = selected_group_map if should_select else rejected_group_map
        for member in members:
            target.setdefault(member, set()).add(group_id)
            if should_select and v429_state_route_safe:
                selected_state_group_map.setdefault(member, set()).add(group_id)
            if should_select and v429_axis_route_safe:
                selected_axis_group_map.setdefault(member, set()).add(group_id)

    def encoded_ids(values: set[str]) -> str:
        return ";".join(sorted(values))

    for row in features.itertuples(index=False):
        feature = str(row.feature)
        pre_ids = pre_group_map.get(feature, set())
        selected_ids = selected_group_map.get(feature, set())
        selected_state_ids = selected_state_group_map.get(feature, set())
        selected_axis_ids = selected_axis_group_map.get(feature, set())
        rejected_ids = rejected_group_map.get(feature, set())
        expected_guard_status = "not_pre_v431_loading_group_member"
        if rejected_ids:
            expected_guard_status = "unresolved_by_v431_loading_group_guard"
        if selected_state_ids:
            expected_guard_status = (
                loading_group_competitor_guard.STATE_RETAINED_STATUS
            )
        if selected_axis_ids:
            expected_guard_status = (
                loading_group_competitor_guard.AXIS_RETAINED_STATUS
            )
        if selected_ids and rejected_ids:
            expected_guard_status = "retained_in_one_group_and_rejected_in_another"
        if (
            str(row.assay_local_loading_group_pre_v428_ids)
            != encoded_ids(pre_ids)
            or bool(row.assay_local_loading_group_pre_v428_support) != bool(pre_ids)
            or str(row.assay_local_loading_group_ids) != encoded_ids(selected_ids)
            or str(row.assay_local_loading_group_rejected_ids)
            != encoded_ids(rejected_ids)
            or bool(row.assay_local_loading_group_support) != bool(selected_ids)
            or str(row.assay_local_loading_group_guard_status)
            != expected_guard_status
        ):
            raise ValueError(
                f"Loading-group v4.31 feature annotation is invalid for {feature}"
            )

    local_hyperedges = all_local_hyperedges.loc[
        all_local_hyperedges["selection_status"].astype(str).eq("selected")
    ].copy()
    selected_local_members: set[str] = set()
    candidate_local_members: set[str] = set()
    candidate_sets: list[set[frozenset[str]]] = []
    nominal_v431_pairs: set[frozenset[str]] = set()
    n_invalid_hyperedges = 0
    n_structure_equivalence_hyperedges = 0
    n_exact_loading_groups = 0
    for row in local_hyperedges.itertuples(index=False):
        members = sorted(
            member.strip()
            for member in set(str(row.nominal_members).split(";"))
            if member.strip()
        )
        hyperedge_members = sorted(
            member.strip()
            for member in set(str(row.members).split(";"))
            if member.strip()
        )
        if len(members) != 2:
            raise ValueError("Selected assay-local nominal group must have two members")
        pairs = {
            frozenset(pair)
            for pair in loading_group_structure_equivalence.parse_candidate_pairs(
                str(row.candidate_pairs_json)
            )
        }
        pair_nodes = set().union(*(set(pair) for pair in pairs)) if pairs else set()
        is_equivalence = _strict_bool(
            row.structure_equivalence_selected,
            label="selected loading-group structure equivalence",
        )
        is_exact = _strict_bool(
            row.exact_loading_group_selected,
            label="selected loading-group exact pair",
        )
        valid_equivalence = bool(
            is_equivalence
            and not is_exact
            and str(row.edge_type)
            == loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE
            and len(pairs) == 4
            and len(pair_nodes) == 4
            and set(hyperedge_members) == pair_nodes
            and not _strict_bool(
                row.exact_pair_identified,
                label="state-equivalence exact-pair identification",
            )
            and int(float(row.candidate_pair_count)) == len(pairs)
            and _strict_bool(
                row.shared_state_hypothesis_supported,
                label="state-equivalence shared-state support",
            )
            and not _strict_bool(
                row.candidate_pairs_may_be_expanded_as_edges,
                label="state-equivalence candidate-pair expansion",
            )
        )
        valid_exact = bool(
            is_exact
            and not is_equivalence
            and str(row.edge_type) == "within_assay_shared_state_support"
            and len(pairs) == 1
            and pairs == {frozenset(members)}
            and set(hyperedge_members) == set(members)
            and _strict_bool(
                row.exact_pair_identified,
                label="axis exact-pair identification",
            )
            and int(float(row.candidate_pair_count)) == len(pairs)
            and _strict_bool(
                row.shared_state_hypothesis_supported,
                label="axis shared-state support",
            )
            and not _strict_bool(
                row.candidate_pairs_may_be_expanded_as_edges,
                label="axis candidate-pair expansion",
            )
        )
        n_structure_equivalence_hyperedges += int(valid_equivalence)
        n_exact_loading_groups += int(valid_exact)
        n_invalid_hyperedges += int(not (valid_equivalence or valid_exact))
        selected_local_members.update(members)
        candidate_local_members.update(pair_nodes)
        candidate_sets.append(pairs)
        nominal_v431_pairs.add(frozenset(members))
        modalities = {
            str(feature_modalities.loc[member]) for member in pair_nodes
        }
        if len(modalities) != 1 or str(row.modalities) not in modalities:
            raise ValueError("Assay-local hyperedge contains cross-assay members")
    true_shared = truth_edges.loc[
        truth_edges["truth_type"].astype(str).eq("within_assay_shared_state")
    ]
    true_local_pairs = {
        frozenset((str(row.node1), str(row.node2)))
        for row in true_shared.itertuples(index=False)
    }
    matching = _maximum_truth_matching(candidate_sets, true_local_pairs)
    matched_count = len(matching)
    nominal_true = nominal_v431_pairs & true_local_pairs
    n_candidate_pairs = sum(len(pairs) for pairs in candidate_sets)
    local_pair_score = {
        "n_selected_hyperedges": len(local_hyperedges),
        # Compatibility names now count one set-valued hypothesis as one
        # prediction; the four candidate pairs are explicitly not expanded.
        "n_expanded_pairs": len(local_hyperedges),
        "n_true_expanded_pairs": matched_count,
        "n_false_expanded_pairs": len(local_hyperedges) - matched_count,
        "pair_precision": _rate(matched_count, len(local_hyperedges)),
        "pair_recall": _rate(matched_count, len(true_local_pairs)),
        "coverage_precision": _rate(matched_count, len(local_hyperedges)),
        "coverage_recall": _rate(matched_count, len(true_local_pairs)),
        "n_matched_unique_truth_pairs": matched_count,
        "n_candidate_pairs": n_candidate_pairs,
        "candidate_pairs_are_joint_alternatives": True,
        "candidate_pairs_expanded_as_edges": False,
        "n_structure_equivalence_hyperedges": n_structure_equivalence_hyperedges,
        "n_exact_loading_groups": n_exact_loading_groups,
        "n_nominal_v431_true_pairs": len(nominal_true),
        "n_nominal_v431_false_pairs": len(
            nominal_v431_pairs - true_local_pairs
        ),
        "nominal_v431_pair_precision": _rate(
            len(nominal_true), len(nominal_v431_pairs)
        ),
        "nominal_v431_pair_recall": _rate(
            len(nominal_true), len(true_local_pairs)
        ),
        "n_sentinel_members": len(candidate_local_members & sentinels),
        "n_non_pair_hyperedges": n_invalid_hyperedges,
        "n_feature_membership_mismatches": len(
            selected_local_members.symmetric_difference(local_group)
        ),
        "zero_sentinel_or_member_conflation": bool(
            not (candidate_local_members & sentinels)
            and n_invalid_hyperedges == 0
            and selected_local_members == local_group
        ),
        "chromatin_assay_coverage": _rate(
            len(set(local_hyperedges["modalities"].astype(str)) & set(chrom_assays)),
            len(chrom_assays),
        ),
    }
    loading_group_state_guard_layer = {
        "application_active": True,
        "guard_schema": loading_group_state_guard.SCHEMA,
        "fixed_rule": loading_group_state_guard.FIXED_RULE,
        "structure_uncertainty_guard_schema": (
            loading_group_structure_guard.SCHEMA
        ),
        "structure_uncertainty_fixed_rule": (
            loading_group_structure_guard.FIXED_RULE
        ),
        "competing_parent_observability_guard_schema": (
            loading_group_competitor_guard.SCHEMA
        ),
        "competing_parent_observability_fixed_rule": (
            loading_group_competitor_guard.FIXED_RULE
        ),
        "structure_equivalence_schema": (
            loading_group_structure_equivalence.SCHEMA
        ),
        "structure_equivalence_fixed_rule": (
            loading_group_structure_equivalence.FIXED_RULE
        ),
        "minimum_state_route_member_centroid_cosine": (
            loading_group_structure_guard
            .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        ),
        "state_route": loading_group_state_guard.STATE_ROUTE,
        "axis_route": loading_group_state_guard.AXIS_ROUTE,
        "n_pre_v429_selected_groups": len(all_local_hyperedges),
        "n_v429_retained_groups": int(
            all_local_hyperedges[
                "loading_group_v429_retained"
            ].map(
                lambda value: _strict_bool(
                    value, label="loading-group v4.29 retained count"
                )
            ).sum()
        ),
        "n_v430_retained_groups": int(
            all_local_hyperedges[
                "loading_group_v430_retained"
            ].map(
                lambda value: _strict_bool(
                    value, label="loading-group v4.30 retained count"
                )
            ).sum()
        ),
        "n_retained_groups": len(local_hyperedges),
        "n_v432_structure_equivalence_hyperedges": (
            n_structure_equivalence_hyperedges
        ),
        "n_v432_exact_assay_local_loading_groups": n_exact_loading_groups,
        "n_state_route_groups": len(state_route_group_ids),
        "n_axis_route_groups": len(axis_route_group_ids),
        "n_rejected_groups": len(all_local_hyperedges) - len(local_hyperedges),
        "n_v429_state_route_groups": len(v429_state_route_group_ids),
        "n_v429_axis_route_groups": len(v429_axis_route_group_ids),
        "n_v430_state_route_groups": len(v430_state_route_group_ids),
        "n_v430_axis_route_groups": len(v430_axis_route_group_ids),
        "n_structure_overlap_downgrades": len(
            structure_overlap_downgrade_ids
        ),
        "n_uncertainty_floor_downgrades": len(
            uncertainty_floor_downgrade_ids
        ),
        "n_competing_parent_observability_downgrades": len(
            competitor_observability_downgrade_ids
        ),
        "n_competing_parent_comparison_unavailable_downgrades": len(
            competitor_comparison_unavailable_ids
        ),
        "n_pre_v429_members": len(pre_group_map),
        "n_retained_members": len(selected_local_members),
        "n_rejected_members": len(rejected_group_map),
        "n_axis_only_members": len(
            selected_local_members.difference(state_v427_supported_set)
        ),
        "state_route_members_are_state_v427_supported": bool(
            set(selected_state_group_map).issubset(state_v427_supported_set)
        ),
        "axis_route_has_state_anchor_and_leave_one_group_out_consensus": bool(
            all(
                str(row.loading_group_guard_route)
                != loading_group_state_guard.AXIS_ROUTE
                or (
                    set(str(row.members).split(";")) & state_v427_supported_set
                    and str(row.cross_assay_axis_id).strip()
                    and int(float(row.cross_assay_axis_n_modalities)) >= 2
                    and int(float(row.cross_assay_axis_n_groups)) >= 2
                )
                for row in all_local_hyperedges.itertuples(index=False)
            )
        ),
        "selection_matches_fixed_rule": bool(loading_guard_safe),
        "state_route_requires_exact_structure_anchor_competition": True,
        "competing_parent_uses_frozen_v427_observability_rule": True,
        "axis_route_bypasses_competing_parent_gate": True,
        "competing_parent_downgrade_is_unresolved_not_refuted": True,
        "direct_structure_alone_is_insufficient": True,
        "direct_overlap_is_unresolved_not_refuted": True,
        "direct_candidates_changed": False,
        "changes_state_v427_membership": False,
        "axis_route_changes_state_v427_membership": False,
        "state_route_is_one_unexpanded_two_by_two_equivalence_hypothesis": bool(
            n_structure_equivalence_hyperedges == len(state_route_group_ids)
        ),
        "state_route_exact_pair_identified": False,
        "axis_route_remains_one_exact_assay_local_group": bool(
            n_exact_loading_groups == len(axis_route_group_ids)
        ),
        "candidate_pairs_are_joint_alternatives": True,
        "candidate_pairs_expanded_as_edges": False,
        "changes_v431_selection": False,
        "exact_cross_assay_feature_pair_claimed": False,
        "truth_required_for_application": False,
        "safe_layered_semantics": bool(
            loading_guard_safe
            and not (
                structure_overlap_downgrade_ids
                & uncertainty_floor_downgrade_ids
            )
            and not (
                competitor_observability_downgrade_ids
                & competitor_comparison_unavailable_ids
            )
            and set(selected_state_group_map).issubset(
                state_v427_supported_set
            )
            and n_invalid_hyperedges == 0
            and n_structure_equivalence_hyperedges
            == len(state_route_group_ids)
            and n_exact_loading_groups == len(axis_route_group_ids)
        ),
    }

    allowed_hyperedge_types = {
        "cross_assay_shared_state_subspace",
        "within_assay_shared_state_support",
        "within_assay_shared_state_equivalence_class",
        loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE,
    }
    unknown_hyperedge_types = sorted(
        set(typed_hyperedges["edge_type"].astype(str)).difference(
            allowed_hyperedge_types
        )
    )
    if unknown_hyperedge_types:
        raise ValueError(
            f"Typed hyperedges contain unregistered edge types: "
            f"{unknown_hyperedge_types}"
        )
    selected_equivalence_hyperedges = typed_hyperedges.loc[
        typed_hyperedges["edge_type"].astype(str).eq(
            "within_assay_shared_state_equivalence_class"
        )
        & typed_hyperedges["selection_status"].astype(str).eq("selected")
    ].copy()
    expected_equivalence_groups = {
        str(modality): set(group.loc[group["state_supported_member"], "feature"])
        for modality, group in features.groupby("modality", sort=True)
        if bool(group["state_supported_member"].any())
    }
    observed_equivalence_groups: dict[str, set[str]] = {}
    equivalence_group_semantics_safe = True
    for row in selected_equivalence_hyperedges.itertuples(index=False):
        modality = str(row.modalities)
        if modality in observed_equivalence_groups:
            raise ValueError(
                f"Duplicate state-equivalence group for modality {modality}"
            )
        members = {
            member.strip()
            for member in str(row.members).split(";")
            if member.strip()
        }
        strict_members = {
            member.strip()
            for member in str(row.strict_members).split(";")
            if member.strip()
        }
        equivalence_members = {
            member.strip()
            for member in str(row.equivalence_only_members).split(";")
            if member.strip()
        }
        if (
            int(row.n_modalities) != 1
            or int(row.n_members) != len(members)
            or members != strict_members | equivalence_members
            or str(row.relation_id) != f"state_equivalence__{modality}"
        ):
            raise ValueError(
                f"State-equivalence group structure is invalid for {modality}"
            )
        expected_strict = set(
            features.loc[
                features["modality"].astype(str).eq(modality)
                & features["strict_state_member"],
                "feature",
            ].astype(str)
        )
        expected_only = set(
            features.loc[
                features["modality"].astype(str).eq(modality)
                & features["state_equivalence_only_member"],
                "feature",
            ].astype(str)
        )
        if strict_members != expected_strict or equivalence_members != expected_only:
            raise ValueError(
                f"State-equivalence group tiers differ from feature evidence: "
                f"{modality}"
            )
        safe = bool(
            _strict_bool(
                row.conditional_on_common_state_model,
                label="state-equivalence conditional_on_common_state_model",
            )
            and _strict_bool(
                row.hypothesis_only,
                label="state-equivalence hypothesis_only",
            )
            and not _strict_bool(
                row.direct_causal_claimed,
                label="state-equivalence direct_causal_claimed",
            )
            and not _strict_bool(
                row.direction_claimed,
                label="state-equivalence direction_claimed",
            )
            and not _strict_bool(
                row.exact_axis_pairing_claimed,
                label="state-equivalence exact_axis_pairing_claimed",
            )
            and not _strict_bool(
                row.pairwise_expansion_permitted,
                label="state-equivalence pairwise_expansion_permitted",
            )
            and not _strict_bool(
                row.exact_cross_assay_pair_claimed,
                label="state-equivalence exact_cross_assay_pair_claimed",
            )
        )
        equivalence_group_semantics_safe &= safe
        observed_equivalence_groups[modality] = members
    equivalence_group_members_match = (
        observed_equivalence_groups == expected_equivalence_groups
    )
    equivalence_supported_score = set_scores["state_equivalence_supported"]
    equivalence_layer_reliable = bool(
        equivalence_supported_score["precision"] >= 0.95
        and equivalence_supported_score["n_sentinel_selected"] == 0
        and equivalence_group_members_match
        and equivalence_group_semantics_safe
    )
    empirical_addition_score = set_scores["empirical_null_state_only"]
    calibrated_supported_score = set_scores["state_calibrated_supported"]
    empirical_null_semantics_safe = bool(
        state_supported.issubset(state_calibrated_supported)
        and state_calibrated_supported
        == state_supported | empirical_null_addition
        and not features["exact_cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_causal_claimed"].astype(bool).any()
        and not features["cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_cross_assay_effect_claimed"].astype(bool).any()
    )
    empirical_null_layer_reliable = bool(
        empirical_null_active
        and calibrated_supported_score["precision"] >= 0.95
        and calibrated_supported_score["n_sentinel_selected"] == 0
        and empirical_null_semantics_safe
    )
    calibrated_role_recall = {
        role: _rate(
            len(
                state_calibrated_supported
                & {feature for feature, value in role_map.items() if value == role}
            ),
            len({feature for feature, value in role_map.items() if value == role}),
        )
        for role in sorted(STATE_ROLES)
    }
    multievidence_addition_score = set_scores["multievidence_state_only"]
    state_v426_supported_score = set_scores["state_v426_supported"]
    multievidence_semantics_safe = bool(
        state_calibrated_supported.issubset(state_v426_supported_set)
        and state_v426_supported_set
        == state_calibrated_supported | multievidence_addition_set
        and not features["exact_cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_causal_claimed"].astype(bool).any()
        and not features["cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_cross_assay_effect_claimed"].astype(bool).any()
    )
    multievidence_layer_reliable = bool(
        multievidence_active
        and state_v426_supported_score["precision"] >= 0.95
        and state_v426_supported_score["n_sentinel_selected"] == 0
        and multievidence_semantics_safe
    )
    v426_role_recall = {
        role: _rate(
            len(
                state_v426_supported_set
                & {feature for feature, value in role_map.items() if value == role}
            ),
            len({feature for feature, value in role_map.items() if value == role}),
        )
        for role in sorted(STATE_ROLES)
    }
    topology_consensus_addition_score = set_scores[
        "topology_consensus_state_only"
    ]
    state_v427_supported_score = set_scores["state_v427_supported"]
    topology_state_gate_set = selected(
        "topology_consensus_state_specific_gate"
    )
    topology_residual_support_set = selected(
        "topology_consensus_residual_structure_support"
    )
    topology_consensus_semantics_safe = bool(
        state_v426_supported_set.issubset(state_v427_supported_set)
        and state_v427_supported_set
        == state_v426_supported_set | topology_consensus_addition_set
        and topology_consensus_addition_set.issubset(topology_state_gate_set)
        and topology_consensus_addition_set.issubset(
            topology_residual_support_set
        )
        and not features["exact_cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_causal_claimed"].astype(bool).any()
        and not features["cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_cross_assay_effect_claimed"].astype(bool).any()
    )
    topology_consensus_layer_reliable = bool(
        topology_active
        and state_v427_supported_score["precision"]
        >= MIN_CONFIRMED_SUBSET_PRECISION
        and state_v427_supported_score["recall"] >= MIN_TOPOLOGY_STATE_RECALL
        and state_v427_supported_score["n_sentinel_selected"] == 0
        and topology_consensus_semantics_safe
    )
    v427_role_recall = {
        role: _rate(
            len(
                state_v427_supported_set
                & {
                    feature
                    for feature, value in role_map.items()
                    if value == role
                }
            ),
            len({feature for feature, value in role_map.items() if value == role}),
        )
        for role in sorted(STATE_ROLES)
    }

    selected_state_hyperedges = typed_hyperedges.loc[
        typed_hyperedges["edge_type"].astype(str).eq(
            "cross_assay_shared_state_subspace"
        )
        & typed_hyperedges["selection_status"].astype(str).eq("selected")
    ].copy()
    state_members_match = False
    state_modalities_match = False
    state_semantics_safe = False
    if len(selected_state_hyperedges) == 1:
        state_row = selected_state_hyperedges.iloc[0]
        state_members = {
            member.strip()
            for member in str(state_row["members"]).split(";")
            if member.strip()
        }
        state_modalities = {
            modality.strip()
            for modality in str(state_row["modalities"]).split(";")
            if modality.strip()
        }
        expected_state_members = set(
            features.loc[
                features["feature"].isin(confirmed)
                & features["modality"].astype(str).isin(
                    participating_modalities
                ),
                "feature",
            ].astype(str)
        )
        state_members_match = state_members == expected_state_members
        state_modalities_match = state_modalities == set(participating_modalities)
        state_semantics_safe = bool(
            _strict_bool(
                state_row["conditional_on_common_state_model"],
                label="state hyperedge conditional_on_common_state_model",
            )
            and _strict_bool(
                state_row["hypothesis_only"],
                label="state hyperedge hypothesis_only",
            )
            and not _strict_bool(
                state_row["direct_causal_claimed"],
                label="state hyperedge direct_causal_claimed",
            )
            and not _strict_bool(
                state_row["direction_claimed"],
                label="state hyperedge direction_claimed",
            )
            and not _strict_bool(
                state_row["exact_axis_pairing_claimed"],
                label="state hyperedge exact_axis_pairing_claimed",
            )
        )
    confirmed_score = set_scores["confirmed_state_response"]
    focal_relation_count = (
        len(predicted_pairs)
        + len(local_hyperedges)
        + len(selected_state_hyperedges)
    )
    unsupported_typed_relations = (
        len(predicted_pairs - true_pairs)
        + int(local_pair_score["n_false_expanded_pairs"])
        + state_v427_supported_score["n_sentinel_selected"]
    )

    no_union_claim = not features["layer_union_is_state_membership"].astype(bool).any()
    no_participation_membership_claim = not features[
        "assay_participation_implies_feature_membership"
    ].astype(bool).any()
    no_pair_claim = not features["cross_assay_pair_claimed"].astype(bool).any()
    no_direct_cross_assay = not features[
        "direct_cross_assay_effect_claimed"
    ].astype(bool).any()
    no_equivalence_pairwise_claim = bool(
        not features["exact_cross_assay_pair_claimed"].astype(bool).any()
        and not features["direct_causal_claimed"].astype(bool).any()
        and not features["strict_membership_changed"].astype(bool).any()
        and equivalence_group_semantics_safe
    )
    local_group_score = set_scores["assay_local_loading_support"]
    state_recall_gate = confirmed_score["recall"] >= 0.80
    per_assay_gate = all(
        per_assay_recall[assay] >= 0.75 for assay in chrom_assays
    )
    role_gate = all(role_recall[role] >= 0.80 for role in STATE_ROLES)
    semantic_gate = bool(
        no_union_claim
        and no_participation_membership_claim
        and no_pair_claim
        and no_direct_cross_assay
        and no_equivalence_pairwise_claim
        and empirical_null_semantics_safe
        and multievidence_semantics_safe
        and topology_consensus_semantics_safe
        and loading_group_state_guard_layer["safe_layered_semantics"]
    )
    confirmed_subset_reliable = bool(
        confirmed_score["precision"] >= 0.95
        and confirmed_score["n_sentinel_selected"] == 0
    )
    assay_participation_reliable = bool(
        len(participating_modalities) == len(chrom_assays) + 1
    )
    loading_support_reliable = bool(
        local_group_score["n_selected"] > 0
        and local_group_score["precision"] >= 0.95
        and local_group_score["n_sentinel_selected"] == 0
        and loading_group_state_guard_layer["safe_layered_semantics"]
    )
    loading_support_complete = bool(
        local_pair_score["pair_precision"] >= 0.80
        and local_pair_score["pair_recall"] >= 0.80
    )
    direct_layer_reliable = bool(
        direct_score["skeleton_precision"] >= 0.80
        and direct_score["skeleton_recall"] >= 0.90
        and direct_score["n_strict_reverse"] == 0
    )
    metrics = {
        "schema_version": SCORER_SCHEMA_VERSION,
        "world_seed": int(world_parameters["seed"]),
        "set_scores": set_scores,
        "confirmed_modalities": confirmed_modalities,
        "confirmed_gene_present": gene in confirmed,
        "confirmed_assay_coverage": _rate(
            len(confirmed_modalities), len(chrom_assays) + 1
        ),
        "confirmed_per_assay_recall": per_assay_recall,
        "confirmed_minimum_chromatin_assay_recall": min(
            per_assay_recall[assay] for assay in chrom_assays
        ),
        "confirmed_role_recall": role_recall,
        "assay_participation_modalities": participating_modalities,
        "assay_participation_coverage": _rate(
            len(participating_modalities), len(chrom_assays) + 1
        ),
        "assay_participation_records": participation_records,
        "direct_score": direct_score,
        "assay_local_loading_pair_score": local_pair_score,
        "assay_local_loading_group_state_guard": (
            loading_group_state_guard_layer
        ),
        "cross_assay_state_hyperedge": {
            "n_selected": len(selected_state_hyperedges),
            "members_equal_strict_confirmed_participating_subset": (
                state_members_match
            ),
            "modalities_equal_assay_participation": state_modalities_match,
            "safe_hypothesis_semantics": state_semantics_safe,
            "reliability_success": bool(
                len(selected_state_hyperedges) == 1
                and state_members_match
                and state_modalities_match
                and state_semantics_safe
            ),
            "estimand": (
                "one_hypothesis_only_common_state_hyperedge_over_replicated_"
                "assay_participation_with_strict_maxT_feature_annotations_and_"
                "without_feature_pair_direction_or_direct_claim"
            ),
        },
        "state_equivalence_layer": {
            "n_selected_groups": len(selected_equivalence_hyperedges),
            "n_equivalence_only_members": len(equivalence_only),
            "n_supported_members": len(state_supported),
            "supported_precision": equivalence_supported_score["precision"],
            "supported_recall": equivalence_supported_score["recall"],
            "n_sentinel_supported": equivalence_supported_score[
                "n_sentinel_selected"
            ],
            "groups_equal_feature_membership": equivalence_group_members_match,
            "safe_set_semantics": equivalence_group_semantics_safe,
            "pairwise_relations_emitted": 0,
            "reliability_success": equivalence_layer_reliable,
            "estimand": (
                "within_assay_set_valued_uncertainty_over_strict_and_strong_"
                "redundancy_demoted_shared_state_reporters"
            ),
        },
        "empirical_null_state_layer": {
            "application_active": empirical_null_active,
            "calibration_schema": (
                empirical_null_state.SCHEMA if empirical_null_active else ""
            ),
            "threshold_method": (
                "one_source_block_maximum_gap_beyond_null_maximum"
                if empirical_null_active
                else ""
            ),
            "selection_threshold": (
                float(features["empirical_null_selection_threshold"].iloc[0])
                if empirical_null_active
                else None
            ),
            "n_candidates": int(expected_calibration_candidate.sum()),
            "n_added_members": len(empirical_null_addition),
            "n_true_added_members": empirical_addition_score["n_true_selected"],
            "n_sentinel_added_members": empirical_addition_score[
                "n_sentinel_selected"
            ],
            "addition_precision": empirical_addition_score["precision"],
            "n_calibrated_supported_members": len(state_calibrated_supported),
            "calibrated_supported_precision": calibrated_supported_score[
                "precision"
            ],
            "calibrated_supported_recall": calibrated_supported_score["recall"],
            "n_sentinel_calibrated_supported": calibrated_supported_score[
                "n_sentinel_selected"
            ],
            "calibrated_role_recall": calibrated_role_recall,
            "existing_state_members_removed": len(
                state_supported.difference(state_calibrated_supported)
            ),
            "pairwise_relations_emitted": 0,
            "direct_relations_created": 0,
            "safe_membership_semantics": empirical_null_semantics_safe,
            "reliability_success": empirical_null_layer_reliable,
            "estimand": (
                "within_assay_feature_participation_in_shared_latent_state_"
                "without_exact_cross_assay_pairing_direction_or_direct_claim"
            ),
        },
        "multievidence_state_layer": {
            "application_active": multievidence_active,
            "calibration_schema": (
                multievidence_state.SCHEMA if multievidence_active else ""
            ),
            "threshold_method": (
                "max_over_fixed_routes_then_one_source_block_maximum_gap_beyond_null_maximum"
                if multievidence_active
                else ""
            ),
            "selection_threshold": (
                float(features["multievidence_selection_threshold"].iloc[0])
                if multievidence_active
                else None
            ),
            "fixed_route_family": list(multievidence_state.ROUTE_NAMES),
            "state_specific_gate": "positive_marginal_and_partial_state_fit",
            "n_candidates": int(expected_calibration_candidate.sum()),
            "n_gate_passing_candidates": int(
                (expected_calibration_candidate & expected_state_specific_gate).sum()
            ),
            "n_added_members": len(multievidence_addition_set),
            "n_true_added_members": multievidence_addition_score[
                "n_true_selected"
            ],
            "n_sentinel_added_members": multievidence_addition_score[
                "n_sentinel_selected"
            ],
            "addition_precision": multievidence_addition_score["precision"],
            "n_incremental_members_beyond_v425": len(
                multievidence_addition_set.difference(state_calibrated_supported)
            ),
            "n_v426_supported_members": len(state_v426_supported_set),
            "v426_supported_precision": state_v426_supported_score["precision"],
            "v426_supported_recall": state_v426_supported_score["recall"],
            "n_sentinel_v426_supported": state_v426_supported_score[
                "n_sentinel_selected"
            ],
            "v426_role_recall": v426_role_recall,
            "existing_v425_members_removed": len(
                state_calibrated_supported.difference(state_v426_supported_set)
            ),
            "pairwise_relations_emitted": 0,
            "direct_relations_created": 0,
            "safe_membership_semantics": multievidence_semantics_safe,
            "reliability_success": multievidence_layer_reliable,
            "estimand": (
                "familywise_multiroute_within_assay_feature_participation_in_"
                "shared_latent_state_without_exact_cross_assay_pairing_"
                "direction_or_direct_claim"
            ),
        },
        "topology_consensus_state_layer": {
            "application_active": topology_active,
            "calibration_schema": (
                topology_consensus_state.SCHEMA if topology_active else ""
            ),
            "null_control_method": (
                topology_consensus_state.NULL_CONTROL_METHOD
                if topology_active
                else ""
            ),
            "sentinel_certified": topology_active,
            "state_specific_gate": (
                topology_consensus_state.STATE_SPECIFIC_GATE
            ),
            "minimum_direct_candidate_degree": (
                topology_consensus_state.MINIMUM_DIRECT_CANDIDATE_DEGREE
            ),
            "n_candidates": int(expected_topology_candidate.sum()),
            "n_gate_passing_candidates": int(
                (
                    expected_topology_candidate
                    & expected_topology_state_gate
                ).sum()
            ),
            "n_residual_structure_supported_candidates": int(
                (
                    expected_topology_candidate
                    & expected_topology_residual_support
                ).sum()
            ),
            "n_fixed_rule_eligible_features": int(
                expected_fixed_rule_eligible.sum()
            ),
            "n_added_members": len(topology_consensus_addition_set),
            "n_true_added_members": topology_consensus_addition_score[
                "n_true_selected"
            ],
            "n_sentinel_added_members": topology_consensus_addition_score[
                "n_sentinel_selected"
            ],
            "addition_precision": topology_consensus_addition_score[
                "precision"
            ],
            "n_incremental_members_beyond_v426": len(
                topology_consensus_addition_set.difference(
                    state_v426_supported_set
                )
            ),
            "n_v427_supported_members": len(state_v427_supported_set),
            "v427_supported_precision": state_v427_supported_score[
                "precision"
            ],
            "v427_supported_recall": state_v427_supported_score["recall"],
            "n_sentinel_v427_supported": state_v427_supported_score[
                "n_sentinel_selected"
            ],
            "v427_role_recall": v427_role_recall,
            "existing_v426_members_removed": len(
                state_v426_supported_set.difference(state_v427_supported_set)
            ),
            "n_direct_structure_only_promotions": len(
                topology_consensus_addition_set.difference(
                    topology_state_gate_set
                )
            ),
            "direct_structure_alone_is_insufficient": bool(
                topology_consensus_addition_set.issubset(
                    topology_state_gate_set
                )
            ),
            "pairwise_relations_emitted": 0,
            "direct_relations_created": 0,
            "safe_membership_semantics": (
                topology_consensus_semantics_safe
            ),
            "reliability_success": topology_consensus_layer_reliable,
            "estimand": (
                "sentinel_certified_conjunction_of_positive_shared_state_"
                "evidence_and_separately_typed_within_assay_residual_"
                "topology_for_feature_participation_without_pairing_"
                "direction_or_direct_causal_claim"
            ),
        },
        "typed_focal_graph": {
            "n_expanded_focal_relations": focal_relation_count,
            "n_typed_focal_hypotheses": focal_relation_count,
            "loading_structure_candidate_pairs_expanded_as_edges": False,
            "n_loading_structure_candidate_alternatives": n_candidate_pairs,
            "n_selected_loading_structure_hypotheses": len(local_hyperedges),
            "n_selected_state_hyperedges": len(selected_state_hyperedges),
            "n_selected_state_equivalence_sets": len(
                selected_equivalence_hyperedges
            ),
            "n_state_equivalence_pairwise_relations": 0,
            "n_empirical_null_state_pairwise_relations": 0,
            "n_multievidence_state_pairwise_relations": 0,
            "n_topology_consensus_state_pairwise_relations": 0,
            "n_unsupported_expanded_typed_relations": unsupported_typed_relations,
            "unsupported_definition": (
                "false direct pair plus unmatched assay-local loading-structure "
                "hypothesis plus sentinel v4.27 state-supported member"
            ),
        },
        "semantics": {
            "layer_union_is_state_membership": not no_union_claim,
            "assay_participation_implies_feature_membership": (
                not no_participation_membership_claim
            ),
            "cross_assay_pair_claimed": not no_pair_claim,
            "direct_cross_assay_effect_claimed": not no_direct_cross_assay,
            "state_equivalence_is_pairwise_structure": (
                not no_equivalence_pairwise_claim
            ),
            "state_equivalence_recall_is_a_gate": False,
            "empirical_null_state_membership_is_pairwise_structure": False,
            "empirical_null_state_membership_implies_direct_causation": False,
            "empirical_null_state_recall_is_a_gate": False,
            "multievidence_state_membership_is_pairwise_structure": False,
            "multievidence_state_membership_implies_direct_causation": False,
            "multievidence_state_recall_is_a_gate": False,
            "topology_consensus_state_membership_is_pairwise_structure": False,
            "topology_consensus_state_membership_implies_direct_causation": False,
            "topology_consensus_direct_structure_alone_implies_state_membership": False,
            "topology_consensus_state_recall_is_a_gate": True,
            "loading_group_state_route_members_are_subset_of_state_v427_members": bool(
                loading_group_state_guard_layer[
                    "state_route_members_are_state_v427_supported"
                ]
            ),
            "loading_group_axis_route_may_include_one_unresolved_state_member": True,
            "loading_group_axis_route_changes_state_v427_membership": False,
            "loading_group_axis_route_claims_exact_cross_assay_feature_pair": False,
            "loading_group_direct_structure_alone_implies_membership": False,
            "loading_group_complete_direct_collider_overlap_is_unresolved": True,
            "loading_group_direct_overlap_downgrade_means_absence": False,
            "loading_group_uncertainty_floor_changes_feature_state_membership": False,
            "loading_group_competing_parent_downgrade_means_absence": False,
            "loading_group_axis_route_bypasses_competing_parent_gate": True,
            "loading_group_guard_changes_state_v427_membership": False,
            "loading_group_state_route_is_exact_pair": False,
            "loading_group_state_route_is_one_set_valued_hypothesis": True,
            "loading_group_candidate_pairs_are_joint_alternatives": True,
            "loading_group_candidate_pairs_expanded_as_edges": False,
            "loading_group_axis_route_is_exact_assay_local_group": True,
            "loading_group_v432_changes_v431_selection": False,
            "descriptive_any_layer_feature_coverage_is_a_gate": False,
        },
        "gates": {
            "confirmed_precision_at_least_0_95": confirmed_score["precision"] >= 0.95,
            "confirmed_zero_sentinels": confirmed_score["n_sentinel_selected"] == 0,
            "assay_participation_all_assays": assay_participation_reliable,
            "confirmed_state_recall_at_least_0_80": state_recall_gate,
            "confirmed_each_chromatin_assay_recall_at_least_0_75": per_assay_gate,
            "confirmed_each_P_Q_S_C_role_recall_at_least_0_80": role_gate,
            "direct_skeleton_precision_at_least_0_80": direct_score[
                "skeleton_precision"
            ]
            >= 0.80,
            "direct_skeleton_recall_at_least_0_90": direct_score["skeleton_recall"]
            >= 0.90,
            "zero_strict_reverse": direct_score["n_strict_reverse"] == 0,
            "no_layer_conflation_or_unsupported_cross_assay_claim": (
                semantic_gate
            ),
            "loading_group_state_guard_active": True,
            "loading_group_state_guard_matches_fixed_rule": bool(
                loading_group_state_guard_layer["selection_matches_fixed_rule"]
            ),
            "loading_group_structure_uncertainty_guard_active": bool(
                loading_group_state_guard_layer[
                    "structure_uncertainty_guard_schema"
                ]
                == loading_group_structure_guard.SCHEMA
            ),
            "loading_group_structure_uncertainty_guard_matches_fixed_rule": bool(
                loading_group_state_guard_layer["selection_matches_fixed_rule"]
            ),
            "loading_group_competing_parent_observability_guard_active": bool(
                loading_group_state_guard_layer[
                    "competing_parent_observability_guard_schema"
                ]
                == loading_group_competitor_guard.SCHEMA
            ),
            "loading_group_competing_parent_observability_guard_matches_fixed_rule": bool(
                loading_group_state_guard_layer["selection_matches_fixed_rule"]
            ),
            "loading_group_structure_equivalence_active": bool(
                loading_group_state_guard_layer[
                    "structure_equivalence_schema"
                ]
                == loading_group_structure_equivalence.SCHEMA
            ),
            "loading_group_structure_equivalence_matches_fixed_rule": bool(
                loading_group_state_guard_layer[
                    "structure_equivalence_fixed_rule"
                ]
                == loading_group_structure_equivalence.FIXED_RULE
                and loading_group_state_guard_layer[
                    "state_route_is_one_unexpanded_two_by_two_equivalence_hypothesis"
                ]
                and not loading_group_state_guard_layer[
                    "state_route_exact_pair_identified"
                ]
                and loading_group_state_guard_layer[
                    "axis_route_remains_one_exact_assay_local_group"
                ]
                and not loading_group_state_guard_layer[
                    "candidate_pairs_expanded_as_edges"
                ]
            ),
            "loading_group_state_route_members_are_state_v427_supported": bool(
                loading_group_state_guard_layer[
                    "state_route_members_are_state_v427_supported"
                ]
            ),
            "loading_group_axis_route_has_state_anchor_and_cross_assay_consensus": bool(
                loading_group_state_guard_layer[
                    "axis_route_has_state_anchor_and_leave_one_group_out_consensus"
                ]
            ),
            "state_equivalence_supported_precision_at_least_0_95": (
                equivalence_supported_score["precision"] >= 0.95
            ),
            "state_equivalence_zero_sentinels": (
                equivalence_supported_score["n_sentinel_selected"] == 0
            ),
            "state_equivalence_groups_match_feature_tiers": (
                equivalence_group_members_match
            ),
            "empirical_null_calibrator_active": empirical_null_active,
            "empirical_null_supported_precision_at_least_0_95": (
                calibrated_supported_score["precision"] >= 0.95
            ),
            "empirical_null_zero_sentinels": (
                calibrated_supported_score["n_sentinel_selected"] == 0
            ),
            "empirical_null_safe_membership_semantics": (
                empirical_null_semantics_safe
            ),
            "multievidence_calibrator_active": multievidence_active,
            "multievidence_supported_precision_at_least_0_95": (
                state_v426_supported_score["precision"] >= 0.95
            ),
            "multievidence_zero_sentinels": (
                state_v426_supported_score["n_sentinel_selected"] == 0
            ),
            "multievidence_safe_membership_semantics": (
                multievidence_semantics_safe
            ),
            "topology_consensus_calibrator_active": topology_active,
            "topology_consensus_supported_precision_at_least_0_95": (
                state_v427_supported_score["precision"]
                >= MIN_CONFIRMED_SUBSET_PRECISION
            ),
            "topology_consensus_supported_recall_at_least_0_80": (
                state_v427_supported_score["recall"]
                >= MIN_TOPOLOGY_STATE_RECALL
            ),
            "topology_consensus_zero_sentinels": (
                state_v427_supported_score["n_sentinel_selected"] == 0
            ),
            "topology_consensus_safe_membership_semantics": (
                topology_consensus_semantics_safe
            ),
        },
        "confirmed_state_recovery_success": bool(
            confirmed_score["precision"] >= 0.95
            and confirmed_score["n_sentinel_selected"] == 0
            and len(confirmed_modalities) == len(chrom_assays) + 1
            and state_recall_gate
            and per_assay_gate
            and role_gate
        ),
        "layer_reliability": {
            "confirmed_state_subset_success": confirmed_subset_reliable,
            "state_equivalence_subset_success": equivalence_layer_reliable,
            "empirical_null_state_subset_success": (
                empirical_null_layer_reliable
            ),
            "multievidence_state_subset_success": (
                multievidence_layer_reliable
            ),
            "topology_consensus_state_subset_success": (
                topology_consensus_layer_reliable
            ),
            "assay_state_participation_success": assay_participation_reliable,
            "assay_local_loading_state_guard_success": bool(
                loading_group_state_guard_layer["safe_layered_semantics"]
            ),
            "assay_local_loading_support_success": loading_support_reliable,
            "assay_local_loading_structure_completeness_success": (
                loading_support_complete
            ),
            "within_assay_direct_layer_success": direct_layer_reliable,
            "honest_layered_reliability_success": bool(
                confirmed_subset_reliable
                and equivalence_layer_reliable
                and empirical_null_layer_reliable
                and multievidence_layer_reliable
                and topology_consensus_layer_reliable
                and assay_participation_reliable
                and loading_support_reliable
                and direct_layer_reliable
                and semantic_gate
            ),
            "honest_layered_structure_recovery_success": bool(
                confirmed_subset_reliable
                and equivalence_layer_reliable
                and empirical_null_layer_reliable
                and multievidence_layer_reliable
                and topology_consensus_layer_reliable
                and assay_participation_reliable
                and loading_support_reliable
                and loading_support_complete
                and direct_layer_reliable
                and semantic_gate
            ),
            "completeness_or_full_state_recovery_claimed": False,
            "estimand": (
                "reliability_of_separate_typed_evidence_layers_with_set_"
                "valued_state_membership_not_exact_cross_assay_pairing"
            ),
        },
        "assay_participation_is_not_feature_membership_rescue": True,
    }
    metrics["preregistered_validation"] = evaluate_layered_world_reliability(
        metrics
    )
    metrics["layer_reliability"][
        "preprotocol_honest_layered_reliability_diagnostic"
    ] = metrics["layer_reliability"]["honest_layered_reliability_success"]
    metrics["layer_reliability"]["honest_layered_reliability_success"] = metrics[
        "preregistered_validation"
    ]["world_reliability_success"]
    metrics["layer_reliability"][
        "per_world_loading_recall_or_strict_confirmed_state_recall_is_a_"
        "reliability_gate"
    ] = False
    metrics["layer_reliability"][
        "per_world_topology_consensus_v427_state_recall_is_a_reliability_gate"
    ] = True
    return metrics


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-evidence", type=Path, required=True)
    parser.add_argument("--typed-hyperedges", type=Path, required=True)
    parser.add_argument("--direct-candidates", type=Path, required=True)
    parser.add_argument("--role-assignments", type=Path, required=True)
    parser.add_argument("--truth-edges", type=Path, required=True)
    parser.add_argument("--world-parameters", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    tsv_options = {
        "sep": "\t",
        "keep_default_na": False,
        "float_precision": "round_trip",
    }
    metrics = score_layered_state_evidence(
        pd.read_csv(args.feature_evidence, **tsv_options),
        pd.read_csv(args.typed_hyperedges, **tsv_options),
        pd.read_csv(args.direct_candidates, **tsv_options),
        pd.read_csv(args.role_assignments, **tsv_options),
        pd.read_csv(args.truth_edges, **tsv_options),
        json.loads(args.world_parameters.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite score output: {args.out}")
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate_layered_world_reliability",
    "layered_validation_protocol",
    "score_layered_state_evidence",
    "validate_layered_evidence_block",
]
