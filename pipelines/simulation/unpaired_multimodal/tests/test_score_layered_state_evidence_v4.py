from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import layered_state_evidence_v4 as layered  # noqa: E402
import score_layered_state_evidence_v4 as scorer  # noqa: E402
from test_layered_state_evidence_v4 import (  # noqa: E402
    _consensus,
    _groups,
    _membership,
    _structure_evidence,
    _subspace,
)


def _precise_partial_metrics(
    seed: int,
    *,
    loading_pair_recall: float = 0.50,
    direct_recall: float = 0.96,
) -> dict[str, object]:
    """A safe seven-assay output with deliberately incomplete membership."""

    metrics: dict[str, object] = {
        "schema_version": scorer.SCORER_SCHEMA_VERSION,
        "world_seed": seed,
        "set_scores": {
            "confirmed_state_response": {
                "n_selected": 7,
                "n_true_selected": 7,
                "n_sentinel_selected": 0,
                "n_unknown_selected": 0,
                "precision": 1.0,
                "recall": 0.28,
                "sentinel_inclusion_rate": 0.0,
            }
        },
        "confirmed_modalities": [
            "chrom_H3K27ac",
            "chrom_H3K27me3",
            "chrom_H3K4me1",
            "chrom_H3K4me2",
            "chrom_H3K4me3",
            "rna",
        ],
        "confirmed_assay_coverage": 6.0 / 7.0,
        "assay_participation_modalities": [
            "chrom_H3K27ac",
            "chrom_H3K27me3",
            "chrom_H3K4me1",
            "chrom_H3K4me2",
            "chrom_H3K4me3",
            "chrom_H3K9me3",
            "rna",
        ],
        "assay_participation_coverage": 1.0,
        "confirmed_gene_present": True,
        "confirmed_state_recovery_success": False,
        "direct_score": {
            "skeleton_precision": 0.95,
            "skeleton_recall": direct_recall,
            "n_strict_reverse": 0,
        },
        "assay_local_loading_pair_score": {
            "n_selected_hyperedges": 6,
            "pair_precision": 0.95,
            "pair_recall": loading_pair_recall,
            "n_structure_equivalence_hyperedges": 6,
            "n_exact_loading_groups": 0,
            "n_sentinel_members": 0,
            "n_non_pair_hyperedges": 0,
            "n_feature_membership_mismatches": 0,
        },
        "assay_local_loading_group_state_guard": {
            "application_active": True,
            "guard_schema": scorer.loading_group_state_guard.SCHEMA,
            "fixed_rule": scorer.loading_group_state_guard.FIXED_RULE,
            "structure_uncertainty_guard_schema": (
                scorer.loading_group_structure_guard.SCHEMA
            ),
            "structure_uncertainty_fixed_rule": (
                scorer.loading_group_structure_guard.FIXED_RULE
            ),
            "competing_parent_observability_guard_schema": (
                scorer.loading_group_competitor_guard.SCHEMA
            ),
            "competing_parent_observability_fixed_rule": (
                scorer.loading_group_competitor_guard.FIXED_RULE
            ),
            "structure_equivalence_schema": (
                scorer.loading_group_structure_equivalence.SCHEMA
            ),
            "structure_equivalence_fixed_rule": (
                scorer.loading_group_structure_equivalence.FIXED_RULE
            ),
            "minimum_state_route_member_centroid_cosine": (
                scorer.loading_group_structure_guard
                .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
            ),
            "safe_layered_semantics": True,
            "state_route_members_are_state_v427_supported": True,
            "axis_route_has_state_anchor_and_leave_one_group_out_consensus": True,
            "selection_matches_fixed_rule": True,
            "direct_structure_alone_is_insufficient": True,
            "direct_overlap_is_unresolved_not_refuted": True,
            "state_route_requires_exact_structure_anchor_competition": True,
            "competing_parent_uses_frozen_v427_observability_rule": True,
            "axis_route_bypasses_competing_parent_gate": True,
            "competing_parent_downgrade_is_unresolved_not_refuted": True,
            "state_route_is_one_unexpanded_two_by_two_equivalence_hypothesis": True,
            "state_route_exact_pair_identified": False,
            "axis_route_remains_one_exact_assay_local_group": True,
            "candidate_pairs_expanded_as_edges": False,
            "direct_candidates_changed": False,
            "changes_state_v427_membership": False,
            "truth_required_for_application": False,
        },
        "cross_assay_state_hyperedge": {
            "n_selected": 1,
            "members_equal_strict_confirmed_participating_subset": True,
            "modalities_equal_assay_participation": True,
            "safe_hypothesis_semantics": True,
        },
        "state_equivalence_layer": {
            "supported_precision": 1.0,
            "supported_recall": 0.84,
            "n_sentinel_supported": 0,
            "groups_equal_feature_membership": True,
            "safe_set_semantics": True,
            "pairwise_relations_emitted": 0,
        },
        "empirical_null_state_layer": {
            "application_active": True,
            "calibrated_supported_precision": 1.0,
            "calibrated_supported_recall": 0.895,
            "n_sentinel_calibrated_supported": 0,
            "safe_membership_semantics": True,
            "pairwise_relations_emitted": 0,
            "direct_relations_created": 0,
        },
        "multievidence_state_layer": {
            "application_active": True,
            "v426_supported_precision": 1.0,
            "v426_supported_recall": 0.918,
            "n_sentinel_v426_supported": 0,
            "safe_membership_semantics": True,
            "pairwise_relations_emitted": 0,
            "direct_relations_created": 0,
        },
        "topology_consensus_state_layer": {
            "application_active": True,
            "v427_supported_precision": 1.0,
            "v427_supported_recall": 0.96,
            "n_sentinel_v427_supported": 0,
            "safe_membership_semantics": True,
            "direct_structure_alone_is_insufficient": True,
            "pairwise_relations_emitted": 0,
            "direct_relations_created": 0,
        },
        "typed_focal_graph": {"n_expanded_focal_relations": 18},
        "semantics": {
            "layer_union_is_state_membership": False,
            "assay_participation_implies_feature_membership": False,
            "cross_assay_pair_claimed": False,
            "direct_cross_assay_effect_claimed": False,
            "state_equivalence_is_pairwise_structure": False,
        },
    }
    return metrics


def test_layered_scorer_does_not_use_the_descriptive_union_as_a_gate() -> None:
    groups, group_membership = _groups()
    result = layered.build_layered_state_evidence(
        _membership(),
        _subspace(),
        _consensus(),
        groups,
        group_membership,
        _structure_evidence(),
    )
    roles = pd.DataFrame(
        [
            {"node": "peak_A", "mark": "A", "role": "P"},
            {"node": "peak_B", "mark": "A", "role": "S"},
            {"node": "peak_C", "mark": "A", "role": "C"},
            {"node": "peak_Z", "mark": "A", "role": "OTHER"},
            {"node": "peak_N", "mark": "B", "role": "N1"},
        ]
    )
    truth_edges = pd.DataFrame(
        [
            {
                "node1": "peak_A",
                "node2": "peak_C",
                "truth_type": "within_assay_direct",
                "parent": "peak_A",
                "child": "peak_C",
            },
            {
                "node1": "peak_B",
                "node2": "peak_C",
                "truth_type": "within_assay_shared_state",
                "parent": "",
                "child": "",
            },
        ]
    )
    metrics = scorer.score_layered_state_evidence(
        result.feature_evidence,
        result.typed_hyperedges,
        result.within_assay_direct_candidates,
        roles,
        truth_edges,
        {"seed": 7, "gene": "gene_G"},
    )
    assert metrics["set_scores"]["confirmed_state_response"]["precision"] == 1.0
    assert metrics["set_scores"]["confirmed_state_response"]["recall"] == 0.75
    assert metrics["set_scores"]["state_equivalence_supported"]["precision"] == 1.0
    assert metrics["set_scores"]["state_equivalence_supported"]["recall"] == 1.0
    assert metrics["state_equivalence_layer"]["n_equivalence_only_members"] == 1
    assert metrics["state_equivalence_layer"]["n_selected_groups"] == 2
    assert metrics["state_equivalence_layer"]["pairwise_relations_emitted"] == 0
    assert metrics["state_equivalence_layer"]["reliability_success"]
    assert not metrics["empirical_null_state_layer"]["application_active"]
    assert not metrics["preregistered_validation"]["gate_checks"][
        "empirical_null_state_calibrator_active"
    ]
    assert not metrics["preregistered_validation"]["gate_checks"][
        "multievidence_state_calibrator_active"
    ]
    assert not metrics["confirmed_state_recovery_success"]
    assert not metrics["gates"][
        "confirmed_each_chromatin_assay_recall_at_least_0_75"
    ]
    assert not metrics["semantics"][
        "descriptive_any_layer_feature_coverage_is_a_gate"
    ]
    assert metrics["direct_score"]["skeleton_precision"] == 1.0
    assert metrics["direct_score"]["n_strict_reverse"] == 0
    assert metrics["layer_reliability"]["confirmed_state_subset_success"]
    assert metrics["layer_reliability"]["state_equivalence_subset_success"]
    assert not metrics["layer_reliability"]["assay_state_participation_success"]
    assert metrics["layer_reliability"]["assay_local_loading_support_success"]
    assert metrics["layer_reliability"]["within_assay_direct_layer_success"]
    assert not metrics["layer_reliability"]["honest_layered_reliability_success"]
    assert metrics["layer_reliability"][
        "assay_local_loading_structure_completeness_success"
    ]
    assert not metrics["layer_reliability"][
        "honest_layered_structure_recovery_success"
    ]
    assert metrics["assay_local_loading_pair_score"]["pair_recall"] == 1.0
    assert metrics["typed_focal_graph"]["n_expanded_focal_relations"] == 3
    assert metrics["typed_focal_graph"][
        "n_unsupported_expanded_typed_relations"
    ] == 0
    assert not metrics["layer_reliability"][
        "completeness_or_full_state_recovery_claimed"
    ]

    serialized_false = result.feature_evidence.copy()
    serialized_false["layer_union_is_state_membership"] = serialized_false[
        "layer_union_is_state_membership"
    ].astype(object)
    serialized_false["layer_union_is_state_membership"] = "False"
    reparsed = scorer.score_layered_state_evidence(
        serialized_false,
        result.typed_hyperedges,
        result.within_assay_direct_candidates,
        roles,
        truth_edges,
        {"seed": 7, "gene": "gene_G"},
    )
    assert not reparsed["semantics"]["layer_union_is_state_membership"]

    invalid_boolean = result.feature_evidence.copy()
    invalid_boolean["formal_test_eligible"] = invalid_boolean[
        "formal_test_eligible"
    ].astype(object)
    invalid_boolean.loc[invalid_boolean.index[0], "formal_test_eligible"] = "unknown"
    with pytest.raises(ValueError, match="formal_test_eligible must be a strict Boolean"):
        scorer.score_layered_state_evidence(
            invalid_boolean,
            result.typed_hyperedges,
            result.within_assay_direct_candidates,
            roles,
            truth_edges,
            {"seed": 7, "gene": "gene_G"},
        )


def test_preregistered_world_gate_accepts_a_precise_partial_state_subset() -> None:
    metrics = _precise_partial_metrics(40_001, loading_pair_recall=0.50)
    validation = scorer.evaluate_layered_world_reliability(metrics)

    assert validation["world_reliability_success"]
    assert all(validation["gate_checks"].values())
    diagnostics = validation["reported_completeness_diagnostics"]
    assert diagnostics["confirmed_state_feature_recall"] == 0.28
    assert not diagnostics["full_state_feature_recovery_success"]
    assert not diagnostics["loading_pair_recall_at_least_0_80"]
    assert not diagnostics["full_state_feature_recall_is_world_gate"]
    assert not diagnostics["loading_pair_recall_is_world_gate"]
    assert diagnostics["topology_consensus_v427_supported_recall"] == 0.96
    assert diagnostics[
        "topology_consensus_v427_supported_recall_is_world_gate"
    ]


def test_world_gate_rejects_assay_participation_as_feature_membership() -> None:
    metrics = _precise_partial_metrics(40_001)
    semantics = metrics["semantics"]
    assert isinstance(semantics, dict)
    semantics["assay_participation_implies_feature_membership"] = True

    validation = scorer.evaluate_layered_world_reliability(metrics)

    assert not validation["gate_checks"][
        "semantic_assay_participation_is_not_feature_membership"
    ]
    assert not validation["world_reliability_success"]


@pytest.mark.parametrize(
    ("hyperedge_override", "failed_gate"),
    [
        (
            {"n_selected": 0},
            "exactly_one_selected_cross_assay_state_hyperedge",
        ),
        (
            {"members_equal_strict_confirmed_participating_subset": False},
            "state_hyperedge_members_equal_strict_confirmed_participating_subset",
        ),
        (
            {"modalities_equal_assay_participation": False},
            "state_hyperedge_modalities_equal_assay_participation",
        ),
    ],
)
def test_world_gate_rejects_missing_or_mismatched_state_hyperedge(
    hyperedge_override: dict[str, object], failed_gate: str
) -> None:
    metrics = _precise_partial_metrics(40_001)
    state_hyperedge = metrics["cross_assay_state_hyperedge"]
    assert isinstance(state_hyperedge, dict)
    state_hyperedge.update(hyperedge_override)

    validation = scorer.evaluate_layered_world_reliability(metrics)

    assert not validation["gate_checks"][failed_gate]
    assert not validation["world_reliability_success"]


def test_block_with_reliable_worlds_but_poor_loading_recall_cannot_certify() -> None:
    attempts = [
        {
            "world_seed": seed,
            "execution_status": "ok",
            "metrics": _precise_partial_metrics(seed, loading_pair_recall=0.50),
        }
        for seed in scorer.FRESH_TRAIN_SEEDS
    ]
    result = scorer.validate_layered_evidence_block("TRAIN", attempts)

    assert result["world_reliability_gate"]["reliable_worlds"] == 24
    assert result["world_reliability_gate"]["pass"]
    guardrails = result["typed_layer_quality_guardrails"]
    assert guardrails["mean_direct_pair_recall"]["pass"]
    assert guardrails["mean_loading_group_pair_precision"]["pass"]
    assert guardrails["mean_expanded_focal_density"]["pass"]
    assert guardrails["mean_loading_group_pair_recall"]["observed"] == 0.50
    assert not guardrails["mean_loading_group_pair_recall"]["pass"]
    assert result["DOE_cell_coverage"]["pass"]
    assert not result["block_certified"]


def test_protocol_registers_exact_blocks_and_missing_worlds_break_coverage() -> None:
    protocol = scorer.layered_validation_protocol()
    assert protocol["schema_version"] == (
        "layered_state_evidence_v4.validation_protocol.v20"
    )
    assert protocol["fresh_blocks"]["TRAIN"]["world_seeds"] == list(
        range(49_001, 49_025)
    )
    assert protocol["fresh_blocks"]["DEVELOPMENT"]["world_seeds"] == list(
        range(50_001, 50_017)
    )
    assert protocol["fresh_blocks"]["LOCKED"]["world_seeds"] == list(
        range(51_001, 51_041)
    )
    assert protocol["fresh_blocks"]["DEVELOPMENT"]["minimum_reliable_worlds"] == 16
    assert protocol["fresh_blocks"]["LOCKED"]["minimum_reliable_worlds"] == 36
    assert not protocol["world_reliability_gate"][
        "strict_confirmed_state_feature_recall_is_required"
    ]
    assert not protocol["world_reliability_gate"][
        "loading_group_pair_recall_is_required"
    ]
    assert not protocol["world_reliability_gate"][
        "assay_participation_feature_membership_claim_allowed"
    ]
    assert not protocol["world_reliability_gate"][
        "state_equivalence_pairwise_expansion_allowed"
    ]
    assert not protocol["world_reliability_gate"][
        "state_equivalence_recall_is_required"
    ]
    assert protocol["world_reliability_gate"][
        "empirical_null_state_calibrator_required"
    ]
    assert not protocol["world_reliability_gate"][
        "empirical_null_state_recall_is_required"
    ]
    assert protocol["world_reliability_gate"][
        "multievidence_state_calibrator_required"
    ]
    assert not protocol["world_reliability_gate"][
        "multievidence_state_recall_is_required"
    ]
    assert protocol["world_reliability_gate"][
        "topology_consensus_state_calibrator_required"
    ]
    assert protocol["world_reliability_gate"][
        "topology_consensus_v427_state_feature_recall_is_required"
    ]
    assert not protocol["world_reliability_gate"][
        "topology_consensus_direct_structure_alone_is_sufficient"
    ]
    assert protocol["history_boundary"]["seeds_33001_33024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_36001_36024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_37001_37024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_38001_38024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_39001_39024"] == (
        "aborted_pretruth_infrastructure_schema_mismatch_no_scorer_opened"
    )
    assert protocol["history_boundary"]["seeds_40001_40024"] == (
        "failed_scientific_confirmation_then_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_46001_46024"] == (
        "passed_confirmation_then_DEV470_informed_v45_and_development_only"
    )
    assert protocol["history_boundary"]["seeds_47001_47016"] == (
        "opened_development_failed_v44_then_posthoc_passed_v45_development_only"
    )
    assert protocol["history_boundary"]["seeds_48001_48040"] == (
        "unopened_reserved_superseded_before_v45_fresh_validation"
    )
    assert set(protocol["fresh_blocks"]["TRAIN"]["world_seeds"]).isdisjoint(
        range(33_001, 33_025)
    )
    assert set(protocol["fresh_blocks"]["TRAIN"]["world_seeds"]).isdisjoint(
        range(36_001, 36_025)
    )
    assert set(protocol["fresh_blocks"]["TRAIN"]["world_seeds"]).isdisjoint(
        range(37_001, 37_025)
    )
    assert set(protocol["fresh_blocks"]["TRAIN"]["world_seeds"]).isdisjoint(
        range(38_001, 38_025)
    )
    assert set(protocol["fresh_blocks"]["TRAIN"]["world_seeds"]).isdisjoint(
        range(39_001, 39_025)
    )
    assert set(protocol["fresh_blocks"]["TRAIN"]["world_seeds"]).isdisjoint(
        range(40_001, 40_025)
    )

    attempts = [
        {
            "world_seed": seed,
            "execution_status": "ok",
            "metrics": _precise_partial_metrics(seed, loading_pair_recall=1.0),
        }
        for seed in scorer.FRESH_TRAIN_SEEDS[:-1]
    ]
    result = scorer.validate_layered_evidence_block("TRAIN", attempts)
    assert len(result["result_records"]) == 24
    assert result["execution_status_counts"]["missing_record"] == 1
    assert not result["DOE_cell_coverage"]["pass"]
    assert not result["block_certified"]


def test_failed_worlds_remain_in_block_completeness_denominator() -> None:
    attempts = [
        {
            "world_seed": seed,
            "execution_status": "ok",
            "metrics": _precise_partial_metrics(seed, loading_pair_recall=1.0),
        }
        for seed in scorer.FRESH_TRAIN_SEEDS[:22]
    ]
    attempts.extend(
        [
            {
                "world_seed": scorer.FRESH_TRAIN_SEEDS[22],
                "execution_status": "discovery_failed",
            },
            {
                "world_seed": scorer.FRESH_TRAIN_SEEDS[23],
                "execution_status": "score_crashed",
            },
        ]
    )

    result = scorer.validate_layered_evidence_block("TRAIN", attempts)

    assert result["world_reliability_gate"]["reliable_worlds"] == 22
    assert result["world_reliability_gate"]["pass"]
    assert result["DOE_cell_coverage"]["pass"]
    assert result["execution_status_counts"]["discovery_failed"] == 1
    assert result["execution_status_counts"]["score_crashed"] == 1
    assert result["typed_layer_quality_guardrails"]["mean_direct_pair_recall"][
        "observed"
    ] == pytest.approx(22 * 0.96 / 24)
    assert not result["typed_layer_quality_guardrails"]["mean_direct_pair_recall"][
        "pass"
    ]
    assert not result["block_certified"]


def test_scorer_independently_detects_forged_v430_direct_overlap_flag() -> None:
    groups, group_membership = _groups()
    consensus = pd.concat(
        [
            _consensus(),
            pd.DataFrame(
                [
                    {
                        "node1": "peak_B",
                        "endpoint1": "TAIL",
                        "node2": "peak_C",
                        "endpoint2": "ARROW",
                        "edge_type": "within_assay_direct_candidate",
                        "collider_motif_center": "peak_C",
                        "orientation_source": "unique_positive_residual_wedge",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    result = layered.build_layered_state_evidence(
        _membership(),
        _subspace(),
        consensus,
        groups,
        group_membership,
        _structure_evidence(),
    )
    forged = result.typed_hyperedges.copy()
    mask = forged["relation_id"].eq("assay_local__group_1")
    forged.loc[mask, "complete_direct_collider_overlap"] = False
    roles = pd.DataFrame(
        [
            {"node": "peak_A", "mark": "A", "role": "P"},
            {"node": "peak_B", "mark": "A", "role": "S"},
            {"node": "peak_C", "mark": "A", "role": "C"},
            {"node": "peak_Z", "mark": "A", "role": "OTHER"},
            {"node": "peak_N", "mark": "B", "role": "N1"},
        ]
    )
    truth_edges = pd.DataFrame(
        [
            {
                "node1": "peak_A",
                "node2": "peak_C",
                "truth_type": "within_assay_direct",
                "parent": "peak_A",
                "child": "peak_C",
            },
            {
                "node1": "peak_B",
                "node2": "peak_C",
                "truth_type": "within_assay_direct",
                "parent": "peak_B",
                "child": "peak_C",
            },
        ]
    )

    metrics = scorer.score_layered_state_evidence(
        result.feature_evidence,
        forged,
        result.within_assay_direct_candidates,
        roles,
        truth_edges,
        {"seed": 8, "gene": "gene_G"},
    )

    guard_metrics = metrics["assay_local_loading_group_state_guard"]
    assert not guard_metrics["selection_matches_fixed_rule"]
    assert not guard_metrics["safe_layered_semantics"]
    assert not metrics["preregistered_validation"]["gate_checks"][
        "loading_group_state_guard_safe"
    ]


def test_scorer_independently_detects_forged_v431_competitor_evidence() -> None:
    groups, group_membership = _groups()
    result = layered.build_layered_state_evidence(
        _membership(),
        _subspace(),
        _consensus(),
        groups,
        group_membership,
        _structure_evidence(),
    )
    forged = result.typed_hyperedges.copy()
    mask = forged["relation_id"].eq("assay_local__group_1")
    assert forged.loc[mask, "selection_status"].iloc[0] == "selected"
    forged.loc[mask, "competing_parent_fixed_rule_eligible"] = False
    roles = pd.DataFrame(
        [
            {"node": "peak_A", "mark": "A", "role": "P"},
            {"node": "peak_B", "mark": "A", "role": "S"},
            {"node": "peak_C", "mark": "A", "role": "C"},
            {"node": "peak_Z", "mark": "A", "role": "OTHER"},
            {"node": "peak_N", "mark": "B", "role": "N1"},
        ]
    )
    truth_edges = pd.DataFrame(
        [
            {
                "node1": "peak_A",
                "node2": "peak_C",
                "truth_type": "within_assay_direct",
                "parent": "peak_A",
                "child": "peak_C",
            },
            {
                "node1": "peak_B",
                "node2": "peak_C",
                "truth_type": "within_assay_shared_state",
                "parent": "",
                "child": "",
            },
        ]
    )

    metrics = scorer.score_layered_state_evidence(
        result.feature_evidence,
        forged,
        result.within_assay_direct_candidates,
        roles,
        truth_edges,
        {"seed": 9, "gene": "gene_G"},
    )

    guard_metrics = metrics["assay_local_loading_group_state_guard"]
    assert not guard_metrics["selection_matches_fixed_rule"]
    assert not guard_metrics["safe_layered_semantics"]
    assert not metrics["preregistered_validation"]["gate_checks"][
        "loading_group_state_guard_safe"
    ]


def test_formal_block_certification_requires_frozen_binding() -> None:
    attempts = [
        {
            "world_seed": seed,
            "execution_status": "ok",
            "metrics": _precise_partial_metrics(seed, loading_pair_recall=1.0),
        }
        for seed in scorer.FRESH_TRAIN_SEEDS
    ]

    with pytest.raises(
        ValueError, match="Formal block certification requires a frozen binding"
    ):
        scorer.validate_layered_evidence_block(
            "TRAIN", attempts, require_formal_binding=True
        )
