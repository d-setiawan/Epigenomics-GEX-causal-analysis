from __future__ import annotations

import inspect
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crossfit_hierarchical_collider_v4 as collider  # noqa: E402


ASSAY = "chrom_test"


def _state(
    features: tuple[str, ...] = ("C", "P", "Q", "N", "R", "S"),
    *,
    selected: tuple[str, ...] = ("P", "Q", "R", "S"),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "modality": ASSAY,
                "selection_status": "selected" if feature in selected else "not_selected",
                "feature_global_maxT_p": 0.01 + 0.001 * index,
                "oof_partial_state_fit": 0.02 if feature in selected else -0.01,
                "conditional_wald_view_support": 1.0 if feature in selected else 0.0,
                "conditional_wald_above_null_95pct": 0.01 if feature in selected else -0.01,
                "reference_conditional_wald": 0.02 if feature in selected else -0.01,
                "feature_global_maxT_confirmation_gate": feature in selected,
                "direction_stability_diagnostic": feature in selected,
            }
            for index, feature in enumerate(features)
        ]
    )


def _candidate(
    center: str,
    outer1: str,
    outer2: str,
    *,
    joint: float = 0.001,
    drop: float = -0.001,
    posterior: float = 0.60,
    fold_support: float = 0.0,
    reverse: float = -0.001,
    arm_p: float = 0.50,
    outer_p: float = 0.50,
    residual: float = 0.0,
    collider_separation: float = -0.10,
    median_collider_separation: float | None = None,
    state_wald: bool = False,
    fold_joint: float = -0.01,
    mean_parent_coefficient: float = 0.0,
    fold_posterior: float = 0.50,
    practical_fold_support: float = 0.0,
    center_detection_rate: float = 0.10,
    minimum_parent_detection_rate: float = 0.10,
    posterior_reverse_margin: float = 0.0,
    residual_reverse_margin: float = 0.0,
    fold_outer_absolute_correlation: float = 0.10,
    outer_wald_view_support: float = 0.0,
) -> dict[str, object]:
    return {
        "source_assay": ASSAY,
        "center": center,
        "outer1": min(outer1, outer2),
        "outer2": max(outer1, outer2),
        "crossfit_joint_gain_over_baseline": joint,
        "crossfit_min_drop_one_gain": drop,
        "crossfit_joint_gain_fold_min": fold_joint,
        "laplace_mean_parent_coefficient_min": mean_parent_coefficient,
        "laplace_min_parent_positive_posterior": posterior,
        "laplace_fold_min_parent_positive_posterior": fold_posterior,
        "laplace_both_positive_fold_support": fold_support,
        "laplace_both_practical_fold_support": practical_fold_support,
        "center_detection_rate": center_detection_rate,
        "minimum_parent_detection_rate": minimum_parent_detection_rate,
        "crossfit_joint_gain_over_baseline_reverse_center_margin": reverse,
        "laplace_min_parent_positive_posterior_reverse_center_margin": (
            posterior_reverse_margin
        ),
        "residual_star_separation_reverse_center_margin": residual_reverse_margin,
        "maximum_arm_empirical_positive_p": arm_p,
        "outer_empirical_null_compatibility_p": outer_p,
        "fold_maximum_outer_absolute_correlation": fold_outer_absolute_correlation,
        "residual_star_separation": residual,
        "minimum_collider_separation": collider_separation,
        "median_collider_separation": (
            collider_separation
            if median_collider_separation is None
            else median_collider_separation
        ),
        "minimum_outer_conditional_wald_view_support": outer_wald_view_support,
        "two_parent_positive_state_wald_consistency": state_wald,
        "state_center_sibling_rotation_structural_candidate": False,
        "state_center_sibling_rotation_displaced_node": "",
        "state_center_sibling_rotation_new_node": "",
        "state_center_sibling_rotation_minimum_loading_cosine": np.nan,
        "state_center_sibling_rotation_maximum_loading_empirical_p": np.nan,
        "state_center_sibling_rotation_log_odds_gain": np.nan,
        "state_center_sibling_rotation_empirical_p_improvement": np.nan,
    }


def _strong(center: str = "C", outer1: str = "P", outer2: str = "Q") -> dict[str, object]:
    return _candidate(
        center,
        outer1,
        outer2,
        joint=0.012,
        drop=0.004,
        posterior=0.995,
        fold_support=1.0,
        reverse=0.003,
        arm_p=0.0625,
        outer_p=0.25,
        residual=0.12,
        collider_separation=0.05,
        state_wald=True,
        fold_joint=0.006,
        mean_parent_coefficient=0.5,
        fold_posterior=0.995,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def _direct(
    parents: tuple[str, ...],
    center: str = "C",
    *,
    observed_action: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "node1": min(parent, center),
                "endpoint1": "ARROW" if min(parent, center) == center else "TAIL",
                "node2": max(parent, center),
                "endpoint2": "ARROW" if max(parent, center) == center else "TAIL",
                "edge_type": collider.DIRECT_EDGE_TYPE,
                "collider_center": center,
                "observed_refinement_action": observed_action,
            }
            for parent in parents
        ],
        columns=[
            "source_assay",
            "node1",
            "endpoint1",
            "node2",
            "endpoint2",
            "edge_type",
            "collider_center",
            "observed_refinement_action",
        ],
    )


def _select(
    candidates: list[dict[str, object]],
    direct: pd.DataFrame,
    *,
    state_membership: pd.DataFrame | None = None,
) -> pd.Series:
    decisions = collider.select_crossfit_hierarchical_collider_replacements(
        candidates=pd.DataFrame(candidates),
        state_membership=(
            _state() if state_membership is None else state_membership
        ),
        current_direct_candidates=direct,
    )
    assert len(decisions) == 1
    return decisions.iloc[0]


def _centrality_direct(
    *,
    minimum_contrast: float = 1.80,
) -> pd.DataFrame:
    return _direct(
        ("C", "N"),
        center="P",
        observed_action=collider.OBSERVED_CENTRALITY_REORIENTATION_ACTION,
    ).assign(
        precentrality_center="C",
        precentrality_outer1="P",
        precentrality_outer2="Q",
        centrality_selected_new_parent="N",
        centrality_mean_center_vs_outer_lor_contrast=minimum_contrast + 0.20,
        centrality_minimum_center_vs_outer_lor_contrast=minimum_contrast,
    )


def test_tier1_replaces_exact_production_only_with_unique_state_consistent_gain() -> None:
    production = _candidate(
        "C", "N", "R", joint=0.006, drop=0.001, posterior=0.99,
        fold_support=1.0, reverse=0.001, arm_p=0.10, residual=0.01,
    )
    decision = _select(
        [_strong(), production, _candidate("R", "P", "Q")],
        _direct(("N", "R")),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == "tier1_exact_production_disagreement"
    assert (decision.selected_center, decision.selected_outer1, decision.selected_outer2) == (
        "C", "P", "Q"
    )


def _tier1u_candidate() -> dict[str, object]:
    return _candidate(
        "C",
        "P",
        "Q",
        joint=0.020,
        drop=0.004,
        posterior=0.995,
        fold_support=1.0,
        reverse=0.003,
        arm_p=1.0 / 32.0,
        outer_p=0.75,
        residual=0.12,
        collider_separation=0.05,
        state_wald=False,
        fold_joint=0.006,
        mean_parent_coefficient=0.5,
        fold_posterior=0.995,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def _tier1u_production() -> dict[str, object]:
    return _candidate(
        "C",
        "N",
        "R",
        joint=0.006,
        drop=0.001,
        posterior=0.99,
        fold_support=1.0,
        reverse=0.001,
        arm_p=0.10,
        residual=0.01,
    )


def test_tier1u_replaces_only_the_fold_stable_state_wald_uncertain_top() -> None:
    decision = _select(
        [_tier1u_candidate(), _tier1u_production(), _candidate("R", "P", "Q")],
        _direct(("N", "R")),
    )
    assert bool(decision.selected_replacement)
    assert bool(decision.tier1u_selected)
    assert decision.action == "tier1u_fold_stable_exact_production_disagreement"
    assert decision.tier1u_abstention_reason == ""
    selected_identity = (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    )
    assert selected_identity == (
        "C",
        "P",
        "Q",
    )


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        (
            "crossfit_joint_gain_fold_min",
            0.0,
            "crossfit_joint_gain_fold_min_not_positive",
        ),
        (
            "laplace_fold_min_parent_positive_posterior",
            0.98,
            "laplace_fold_parent_posterior_below_threshold",
        ),
        (
            "laplace_both_practical_fold_support",
            0.8,
            "both_practical_fold_support_below_threshold",
        ),
        (
            "maximum_arm_empirical_positive_p",
            0.0625,
            "arm_empirical_positive_p_above_threshold",
        ),
        (
            "outer_empirical_null_compatibility_p",
            0.25,
            "outer_empirical_null_compatibility_below_threshold",
        ),
    ],
)
def test_tier1u_abstains_when_any_fold_uncertainty_guard_fails(
    column: str,
    value: float,
    reason: str,
) -> None:
    candidate = _tier1u_candidate()
    candidate[column] = value
    decision = _select(
        [candidate, _tier1u_production(), _candidate("R", "P", "Q")],
        _direct(("N", "R")),
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.tier1u_considered)
    assert reason in decision.tier1u_abstention_reason


def _tier1r_candidate() -> dict[str, object]:
    return _candidate(
        "C",
        "P",
        "Q",
        joint=0.010,
        drop=0.001,
        posterior=0.91,
        fold_support=1.0,
        reverse=0.001,
        arm_p=1.0 / 16.0,
        outer_p=0.91,
        residual=0.04,
        state_wald=False,
        fold_joint=0.001,
        mean_parent_coefficient=0.4,
        fold_posterior=0.72,
        practical_fold_support=1.0,
    )


def _tier1r_production() -> dict[str, object]:
    return _candidate(
        "P",
        "C",
        "S",
        joint=0.007,
        drop=0.0005,
        posterior=0.95,
        fold_support=1.0,
        reverse=0.20,
        arm_p=0.10,
        residual=0.01,
        state_wald=False,
        fold_posterior=0.92,
    )


def test_tier1r_replaces_redundancy_demoted_production_with_residual_global_top() -> None:
    decision = _select(
        [_tier1r_candidate(), _tier1r_production(), _candidate("R", "P", "Q")],
        _direct(("C", "S"), center="P"),
    )
    assert bool(decision.selected_replacement)
    assert bool(decision.tier1r_considered)
    assert bool(decision.tier1r_selected)
    assert decision.action == collider.RESIDUAL_FIRST_REPLACEMENT_ACTION
    assert decision.tier1r_abstention_reason == ""
    assert decision.tier1r_laplace_parent_posterior_improvement < 0.0
    assert decision.tier1r_laplace_fold_parent_posterior_improvement < 0.0
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        (
            "crossfit_joint_gain_fold_min",
            0.0,
            "crossfit_joint_gain_fold_min_not_positive",
        ),
        (
            "crossfit_min_drop_one_gain",
            0.0,
            "crossfit_min_drop_one_gain_not_positive",
        ),
        (
            "laplace_min_parent_positive_posterior",
            0.899,
            "laplace_mean_parent_posterior_below_threshold",
        ),
        (
            "laplace_fold_min_parent_positive_posterior",
            0.699,
            "laplace_fold_parent_posterior_below_threshold",
        ),
        (
            "laplace_both_positive_fold_support",
            0.8,
            "both_positive_fold_support_below_threshold",
        ),
        (
            "laplace_both_practical_fold_support",
            0.8,
            "both_practical_fold_support_below_threshold",
        ),
        (
            "maximum_arm_empirical_positive_p",
            0.063,
            "arm_empirical_positive_p_above_threshold",
        ),
        (
            "outer_empirical_null_compatibility_p",
            0.899,
            "outer_empirical_null_compatibility_below_threshold",
        ),
    ],
)
def test_tier1r_abstains_when_any_residual_guard_fails(
    column: str,
    value: float,
    reason: str,
) -> None:
    candidate = _tier1r_candidate()
    candidate[column] = value
    decision = _select(
        [candidate, _tier1r_production(), _candidate("R", "P", "Q")],
        _direct(("C", "S"), center="P"),
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.tier1r_considered)
    assert reason in decision.tier1r_abstention_reason


def test_tier1r_requires_global_margin_and_exact_production_improvement() -> None:
    candidate = pd.Series(_tier1r_candidate())
    production = pd.Series(_tier1r_production())
    config = collider.CrossfitHierarchicalColliderConfig()
    assert "global_runner_margin_below_threshold" in collider._tier1r_gate_failures(
        candidate=candidate,
        production_candidate=production,
        global_runner_margin=0.0029,
        score_improvement=0.01,
        config=config,
    )
    assert (
        "production_score_improvement_below_threshold"
        in collider._tier1r_gate_failures(
            candidate=candidate,
            production_candidate=production,
            global_runner_margin=0.01,
            score_improvement=0.0029,
            config=config,
        )
    )


def test_tier1r_abstains_from_same_center_parent_swap() -> None:
    production = _candidate(
        "C",
        "N",
        "S",
        joint=0.007,
        drop=0.0005,
        posterior=0.10,
        fold_support=0.0,
        reverse=0.20,
        arm_p=0.10,
        residual=0.01,
        state_wald=False,
        fold_posterior=0.04,
    )
    decision = _select(
        [_tier1r_candidate(), production, _candidate("R", "P", "Q")],
        _direct(("N", "S")),
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.tier1r_considered)
    assert "replacement_does_not_change_center" in decision.tier1r_abstention_reason


@pytest.mark.parametrize("invalid", [np.nan, "False", 1])
def test_state_wald_gate_rejects_non_boolean_stand_ins(invalid: object) -> None:
    candidate = _strong()
    candidate["two_parent_positive_state_wald_consistency"] = invalid
    with pytest.raises(ValueError, match="must contain only Boolean values"):
        _select([candidate, _candidate("R", "P", "Q")], _direct(()))


def test_selector_rejects_infinite_numeric_evidence_but_allows_missing_values() -> None:
    infinite = _strong()
    infinite["crossfit_min_drop_one_gain"] = np.inf
    with pytest.raises(ValueError, match="contains infinite evidence"):
        _select([infinite, _candidate("R", "P", "Q")], _direct(()))

    missing = _strong()
    missing["crossfit_min_drop_one_gain"] = np.nan
    decision = _select([missing, _candidate("R", "P", "Q")], _direct(()))
    assert not bool(decision.selected_replacement)


@pytest.mark.parametrize(
    "column",
    ["feature_global_maxT_confirmation_gate", "direction_stability_diagnostic"],
)
def test_state_membership_gate_rejects_non_boolean_stand_ins(column: str) -> None:
    state = _state()
    state[column] = state[column].astype(object)
    state.loc[state.index[0], column] = "False"
    with pytest.raises(ValueError, match="must contain only Boolean values"):
        collider.select_crossfit_hierarchical_collider_replacements(
            candidates=pd.DataFrame([_strong(), _candidate("R", "P", "Q")]),
            state_membership=state,
            current_direct_candidates=_direct(()),
        )


def test_tier1_completes_nonexact_production() -> None:
    decision = _select(
        [_strong(), _candidate("R", "P", "Q")],
        _direct(()),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == "tier1_nonexact_production_completion"


def test_tier2_rescues_positive_collider_when_tier1_posterior_fails() -> None:
    tier2 = _candidate(
        "C",
        "P",
        "Q",
        joint=0.010,
        drop=0.001,
        posterior=0.93,
        fold_support=1.0,
        reverse=0.001,
        arm_p=0.0625,
        outer_p=0.25,
        residual=0.06,
        collider_separation=0.05,
        state_wald=False,
    )
    decision = _select([tier2, _candidate("R", "P", "Q")], _direct(()))
    assert bool(decision.selected_replacement)
    assert decision.action == "tier2_positive_collider_incomplete_rescue"


def _complete_zero_edge_candidates(*, state_wald: bool) -> list[dict[str, object]]:
    features = ("C", "P", "Q", "N", "R", "S")
    rows = [
        _candidate(center, outer1, outer2)
        for center in features
        for outer1, outer2 in itertools.combinations(
            tuple(feature for feature in features if feature != center), 2
        )
    ]
    selected = next(
        row
        for row in rows
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "Q"}
    )
    selected.update(
        {
            "crossfit_joint_gain_over_baseline": 0.05,
            "crossfit_min_drop_one_gain": 0.02,
            "crossfit_joint_gain_fold_min": 0.01,
            "laplace_mean_parent_coefficient_min": 0.5,
            "laplace_both_practical_fold_support": 1.0,
            "minimum_outer_conditional_wald_view_support": 1.0,
            "two_parent_positive_state_wald_consistency": state_wald,
        }
    )
    return rows


def test_zero_edge_rank_fusion_selects_only_after_the_independent_state_gate() -> None:
    selected = _select(_complete_zero_edge_candidates(state_wald=True), _direct(()))
    assert bool(selected.selected_replacement)
    assert selected.action == "zero_edge_state_gated_rank_fusion_completion"
    assert selected.zero_edge_rank_fusion_status == "selected"
    assert not bool(selected.zero_edge_rank_fusion_runtime_refit)
    assert selected.zero_edge_rank_fusion_model_sha256 == (
        "a8b42323b49dac5e1c40485c236169eb3156f2fe1bb0d4ab0817925c9037cd6b"
    )
    selected_identity = (
        selected.selected_center,
        selected.selected_outer1,
        selected.selected_outer2,
    )
    assert selected_identity == (
        "C",
        "P",
        "Q",
    )
    assert selected.zero_edge_rank_fusion_center_score_margin > 0.0
    assert selected.zero_edge_rank_fusion_selected_practical_fold_support == 1.0
    assert bool(selected.zero_edge_rank_fusion_practical_fold_support_gate)
    assert selected.zero_edge_rank_fusion_arm1_evidence_json
    assert selected.zero_edge_rank_fusion_all_edge_scores_json
    assert selected.zero_edge_rank_fusion_all_center_scores_json

    abstained = _select(_complete_zero_edge_candidates(state_wald=False), _direct(()))
    assert not bool(abstained.selected_replacement)
    assert abstained.zero_edge_rank_fusion_status == "abstained"
    assert (
        "selected_star_failed_state_wald_consistency"
        in abstained.zero_edge_rank_fusion_abstention_reason
    )

    missing_support_candidates = _complete_zero_edge_candidates(state_wald=True)
    selected_candidate = next(
        row
        for row in missing_support_candidates
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "Q"}
    )
    selected_candidate["minimum_outer_conditional_wald_view_support"] = np.nan
    missing_support = _select(missing_support_candidates, _direct(()))
    assert not bool(missing_support.selected_replacement)
    assert (
        "selected_star_lacked_outer_wald_view_support"
        in missing_support.zero_edge_rank_fusion_abstention_reason
    )

    incomplete_fold_candidates = _complete_zero_edge_candidates(state_wald=True)
    incomplete_fold_selected = next(
        row
        for row in incomplete_fold_candidates
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "Q"}
    )
    incomplete_fold_selected["laplace_both_practical_fold_support"] = 0.8
    incomplete_fold = _select(incomplete_fold_candidates, _direct(()))
    assert not bool(incomplete_fold.selected_replacement)
    assert incomplete_fold.zero_edge_rank_fusion_status == "abstained"
    assert incomplete_fold.zero_edge_rank_fusion_selected_practical_fold_support == 0.8
    assert not bool(incomplete_fold.zero_edge_rank_fusion_practical_fold_support_gate)
    assert (
        "selected_star_lacked_all_fold_practical_support"
        in incomplete_fold.zero_edge_rank_fusion_abstention_reason
    )


def _weak_multicenter_production() -> dict[str, object]:
    return _candidate(
        "C",
        "N",
        "R",
        joint=0.004,
        drop=-0.0001,
        posterior=0.92,
        fold_support=1.0,
        reverse=0.002,
        arm_p=0.0625,
        outer_p=0.50,
        residual=0.04,
        state_wald=True,
        fold_joint=-0.001,
        mean_parent_coefficient=0.4,
        fold_posterior=0.72,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def _multicenter_validated_alternative() -> dict[str, object]:
    return _candidate(
        "R",
        "C",
        "S",
        joint=0.008,
        drop=0.001,
        posterior=0.95,
        fold_support=1.0,
        reverse=0.002,
        arm_p=0.25,
        outer_p=0.95,
        residual=0.12,
        collider_separation=0.05,
        state_wald=True,
        fold_joint=0.001,
        mean_parent_coefficient=0.5,
        fold_posterior=0.90,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def test_weak_multicenter_uses_one_unique_state_consistent_alternative() -> None:
    decision = _select(
        [
            _weak_multicenter_production(),
            _multicenter_validated_alternative(),
            _candidate("Q", "P", "S"),
        ],
        _direct(
            ("N", "R"),
            observed_action="resolve_competing_multicenter_path",
        ),
    )
    assert bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == collider.WEAK_MULTICENTER_REPLACEMENT_ACTION
    assert bool(decision.multicenter_validation_selected_replacement)
    assert decision.multicenter_validation_eligible_alternatives == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("R", "C", "S")


def test_weak_multicenter_demotes_without_one_validated_alternative() -> None:
    decision = _select(
        [_weak_multicenter_production(), _candidate("Q", "P", "S")],
        _direct(
            ("N", "R"),
            observed_action="resolve_competing_multicenter_path",
        ),
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.demote_production_motif)
    assert decision.action == collider.WEAK_MULTICENTER_DEMOTION_ACTION
    assert bool(decision.multicenter_validation_selected_demotion)
    assert decision.multicenter_validation_eligible_alternatives == 0


def test_strong_multicenter_production_is_retained() -> None:
    production = _weak_multicenter_production()
    production["crossfit_min_drop_one_gain"] = 0.001
    production["laplace_fold_min_parent_positive_posterior"] = 0.90
    decision = _select(
        [production, _candidate("Q", "P", "S")],
        _direct(
            ("N", "R"),
            observed_action="resolve_competing_multicenter_path",
        ),
    )
    assert not bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == "retain_current_direct_motif"
    assert decision.multicenter_validation_status == "retained"


def _centrality_production(*, strong: bool) -> dict[str, object]:
    return _candidate(
        "P",
        "C",
        "N",
        joint=0.008,
        drop=0.001 if strong else -0.0001,
        posterior=0.95 if strong else 0.78,
        fold_support=1.0,
        reverse=0.001,
        arm_p=0.0625 if strong else 0.25,
        outer_p=0.50,
        residual=0.04,
        state_wald=False,
        fold_joint=0.001,
        mean_parent_coefficient=0.4,
        fold_posterior=0.90 if strong else 0.60,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def _centrality_constrained_candidate(*, supported: bool) -> dict[str, object]:
    return _candidate(
        "P",
        "C",
        "Q",
        joint=0.006 if supported else 0.001,
        drop=0.0005 if supported else -0.001,
        posterior=0.88 if supported else 0.60,
        fold_support=1.0 if supported else 0.0,
        reverse=0.001,
        arm_p=0.25 if supported else 0.50,
        outer_p=0.50,
        residual=0.02 if supported else 0.0,
        state_wald=supported,
        fold_joint=0.001 if supported else -0.01,
        mean_parent_coefficient=0.3 if supported else 0.0,
        fold_posterior=0.70 if supported else 0.50,
        practical_fold_support=0.80 if supported else 0.0,
        outer_wald_view_support=1.0 if supported else 0.0,
    )


def _all_selected_state() -> pd.DataFrame:
    features = ("C", "P", "Q", "N", "R", "S")
    return _state(features, selected=features)


def test_strong_beta_centrality_completion_is_retained() -> None:
    decision = _select(
        [
            _centrality_production(strong=True),
            _centrality_constrained_candidate(supported=False),
            _candidate("C", "P", "Q"),
        ],
        _centrality_direct(),
        state_membership=_all_selected_state(),
    )
    assert not bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == "retain_current_direct_motif"
    assert decision.centrality_validation_status == "retained"
    assert bool(decision.centrality_validation_full_is_strong)


def test_weak_beta_centrality_uses_supported_center_on_original_skeleton() -> None:
    decision = _select(
        [
            _centrality_production(strong=False),
            _centrality_constrained_candidate(supported=True),
            _candidate("C", "P", "Q"),
        ],
        _centrality_direct(),
        state_membership=_all_selected_state(),
    )
    assert bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == collider.CENTRALITY_SAME_SKELETON_REPLACEMENT_ACTION
    assert decision.centrality_validation_status == (
        "constrained_to_original_skeleton"
    )
    assert bool(decision.centrality_validation_constrained_passed)
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("P", "C", "Q")


def test_weak_beta_centrality_reverts_to_stable_precompletion() -> None:
    decision = _select(
        [
            _centrality_production(strong=False),
            _centrality_constrained_candidate(supported=False),
            _candidate("C", "P", "Q"),
        ],
        _centrality_direct(),
        state_membership=_all_selected_state(),
    )
    assert bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == collider.CENTRALITY_PRECOMPLETION_REVERSION_ACTION
    assert decision.centrality_validation_status == "reverted_to_precompletion"
    assert not bool(decision.centrality_validation_constrained_passed)
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


def test_weak_beta_centrality_demotes_when_no_completion_is_recoverable() -> None:
    decision = _select(
        [
            _centrality_production(strong=False),
            _candidate("R", "Q", "S"),
        ],
        _centrality_direct(),
        state_membership=_all_selected_state(),
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.demote_production_motif)
    assert decision.action == collider.CENTRALITY_UNRECOVERABLE_DEMOTION_ACTION
    assert decision.centrality_validation_status == "demoted"


def test_higher_priority_replacement_preempts_centrality_validation() -> None:
    decision = _select(
        [
            _centrality_production(strong=False),
            _strong("C", "P", "Q"),
            _candidate("R", "Q", "S"),
        ],
        _centrality_direct(),
        state_membership=_all_selected_state(),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == "tier1_exact_production_disagreement"
    assert decision.centrality_validation_status == (
        "not_needed_higher_priority_selected"
    )
    assert not bool(decision.centrality_validation_selected_demotion)


def _dual_anchor_direct() -> pd.DataFrame:
    return _direct(
        ("P", "N"),
        center="C",
        observed_action=collider.OBSERVED_DUAL_ANCHOR_COMPLETION_ACTION,
    ).assign(
        completion_anchor_node1="C",
        completion_anchor_node2="P",
        completion_winner_view_count=6,
        completion_n_views=6,
    )


def _dual_anchor_production() -> dict[str, object]:
    return _candidate(
        "C",
        "P",
        "N",
        joint=0.005,
        drop=0.0005,
        posterior=0.92,
        fold_support=1.0,
        arm_p=0.10,
        residual=0.02,
        collider_separation=-0.30,
        state_wald=True,
        fold_posterior=0.84,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def _dual_anchor_validated_alternative() -> dict[str, object]:
    return _candidate(
        "C",
        "P",
        "Q",
        joint=0.0045,
        drop=-0.0002,
        posterior=0.80,
        fold_support=1.0,
        arm_p=0.25,
        residual=0.02,
        collider_separation=-0.02,
        state_wald=True,
        fold_posterior=0.70,
        practical_fold_support=1.0,
        outer_wald_view_support=1.0,
    )


def test_residually_inconsistent_dual_anchor_uses_one_validated_alternative() -> None:
    decision = _select(
        [
            _dual_anchor_production(),
            _dual_anchor_validated_alternative(),
            _candidate("R", "Q", "S"),
        ],
        _dual_anchor_direct(),
        state_membership=_all_selected_state(),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.DUAL_ANCHOR_RESIDUAL_REPLACEMENT_ACTION
    assert decision.dual_anchor_validation_status == "replaced"
    assert decision.dual_anchor_validation_eligible_alternatives == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


def test_dual_anchor_retains_when_no_unique_residual_alternative_passes() -> None:
    alternative = _dual_anchor_validated_alternative()
    alternative["laplace_both_practical_fold_support"] = 0.8
    decision = _select(
        [_dual_anchor_production(), alternative, _candidate("R", "Q", "S")],
        _dual_anchor_direct(),
        state_membership=_all_selected_state(),
    )
    assert not bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == "retain_current_direct_motif"
    assert decision.dual_anchor_validation_status == "retained"
    assert decision.dual_anchor_validation_eligible_alternatives == 0


def test_dual_anchor_uses_unique_positive_conditional_separation_signal() -> None:
    production = _dual_anchor_production()
    production["minimum_collider_separation"] = -0.08
    alternative = _dual_anchor_validated_alternative()
    alternative.update(
        {
            "crossfit_joint_gain_over_baseline": 0.002,
            "laplace_min_parent_positive_posterior": 0.72,
            "laplace_fold_min_parent_positive_posterior": 0.61,
            "laplace_both_practical_fold_support": 0.8,
            "maximum_arm_empirical_positive_p": 0.45,
            "minimum_collider_separation": 0.01,
            "minimum_outer_conditional_wald_view_support": 0.0,
            "two_parent_positive_state_wald_consistency": False,
            "residual_star_separation": 0.0,
        }
    )
    decision = _select(
        [production, alternative, _candidate("R", "Q", "S")],
        _dual_anchor_direct(),
        state_membership=_all_selected_state(),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.DUAL_ANCHOR_RESIDUAL_REPLACEMENT_ACTION
    assert decision.dual_anchor_validation_selected_mode == (
        "unique_positive_conditional_collider_separation"
    )
    assert decision.dual_anchor_validation_structural_eligible_alternatives == 1


def _dual_anchor_state_completion_alternative() -> dict[str, object]:
    return _candidate(
        "C",
        "P",
        "Q",
        joint=0.0105,
        drop=-0.0005,
        posterior=0.78,
        fold_support=1.0,
        practical_fold_support=0.8,
        arm_p=0.25,
        collider_separation=-0.20,
        state_wald=False,
        fold_joint=0.0016,
        fold_posterior=0.56,
        outer_wald_view_support=0.0,
    )


def test_dual_anchor_uses_unique_state_member_with_positive_fold_gain() -> None:
    production = _dual_anchor_production()
    production["minimum_collider_separation"] = -0.24
    decision = _select(
        [
            production,
            _dual_anchor_state_completion_alternative(),
            _candidate("R", "Q", "S"),
        ],
        _dual_anchor_direct(),
        state_membership=_state(selected=("Q", "R", "S")),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.DUAL_ANCHOR_RESIDUAL_REPLACEMENT_ACTION
    assert decision.dual_anchor_validation_selected_mode == (
        "unique_state_member_fold_residual_completion"
    )
    assert decision.dual_anchor_validation_state_completion_eligible_alternatives == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


def test_dual_anchor_state_completion_requires_new_parent_state_membership() -> None:
    production = _dual_anchor_production()
    production["minimum_collider_separation"] = -0.24
    decision = _select(
        [
            production,
            _dual_anchor_state_completion_alternative(),
            _candidate("R", "Q", "S"),
        ],
        _dual_anchor_direct(),
        state_membership=_state(selected=("R", "S")),
    )
    assert not bool(decision.selected_replacement)
    assert decision.dual_anchor_validation_state_completion_eligible_alternatives == 0


def _state_inconsistent_neighbor_production(
    *, state_wald: bool = False,
) -> dict[str, object]:
    return _candidate(
        "C",
        "N",
        "Q",
        joint=0.006,
        drop=0.001,
        posterior=0.97,
        fold_support=1.0,
        arm_p=0.0625,
        residual=0.02,
        state_wald=state_wald,
        fold_joint=0.001,
        fold_posterior=0.95,
        practical_fold_support=1.0,
        outer_wald_view_support=0.0,
    )


def _supported_neighbor(
    *, center: str = "C", outer1: str = "P", outer2: str = "Q"
) -> dict[str, object]:
    return _candidate(
        center,
        outer1,
        outer2,
        joint=0.005,
        drop=-0.0002,
        posterior=0.93,
        fold_support=1.0,
        arm_p=0.40,
        state_wald=True,
        fold_joint=-0.004,
        fold_posterior=0.80,
        practical_fold_support=1.0,
        posterior_reverse_margin=0.40,
        outer_wald_view_support=1.0,
    )


@pytest.mark.parametrize(
    ("neighbor", "expected"),
    [
        (_supported_neighbor(), ("C", "P", "Q")),
        (_supported_neighbor(center="Q", outer1="C", outer2="P"), ("Q", "C", "P")),
    ],
)
def test_state_inconsistent_production_uses_one_supported_adjacent_star(
    neighbor: dict[str, object], expected: tuple[str, str, str]
) -> None:
    decision = _select(
        [
            _state_inconsistent_neighbor_production(),
            neighbor,
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.STATE_INCONSISTENT_NEIGHBOR_REPLACEMENT_ACTION
    assert decision.neighbor_star_validation_status == "replaced"
    assert decision.neighbor_star_validation_eligible_candidates == 1
    assert decision.neighbor_star_validation_selected_shared_adjacencies == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == expected


def test_neighbor_star_validator_retains_supported_or_nonunique_production() -> None:
    supported = _select(
        [
            _state_inconsistent_neighbor_production(state_wald=True),
            _supported_neighbor(),
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
    )
    assert not bool(supported.neighbor_star_validation_selected)
    assert supported.neighbor_star_validation_reason == (
        "production_already_has_positive_state_wald_consistency"
    )

    nonunique = _select(
        [
            _state_inconsistent_neighbor_production(),
            _supported_neighbor(),
            _supported_neighbor(center="Q", outer1="C", outer2="P"),
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
    )
    assert not bool(nonunique.selected_replacement)
    assert nonunique.neighbor_star_validation_eligible_candidates == 2


def test_neighbor_star_validator_preserves_strong_residual_production() -> None:
    production = _state_inconsistent_neighbor_production()
    production.update(
        {
            "crossfit_joint_gain_over_baseline": 0.015,
            "crossfit_min_drop_one_gain": 0.005,
            "residual_star_separation": 0.12,
        }
    )
    decision = _select(
        [production, _supported_neighbor(), _candidate("R", "P", "S")],
        _direct(("N", "Q"), center="C"),
    )
    assert not bool(decision.selected_replacement)
    assert decision.neighbor_star_validation_reason == (
        "production_residual_evidence_is_too_strong_for_state_based_replacement"
    )


def _state_center_sibling_rotation_candidate(
    *, log_odds_gain: float = 0.32
) -> dict[str, object]:
    candidate = _candidate(
        "C",
        "P",
        "Q",
        joint=0.0111,
        drop=-0.00135,
        posterior=0.73,
        fold_support=0.8,
        practical_fold_support=0.6,
        arm_p=0.50,
        outer_p=0.09375,
        residual=-0.045,
        collider_separation=-0.17,
        median_collider_separation=0.045,
        state_wald=True,
        fold_joint=0.0066,
        fold_posterior=0.46,
        posterior_reverse_margin=0.60,
        outer_wald_view_support=1.0,
    )
    candidate.update(
        {
            "state_center_sibling_rotation_structural_candidate": True,
            "state_center_sibling_rotation_displaced_node": "S",
            "state_center_sibling_rotation_new_node": "P",
            "state_center_sibling_rotation_minimum_loading_cosine": 0.96,
            "state_center_sibling_rotation_maximum_loading_empirical_p": 0.102,
            "state_center_sibling_rotation_log_odds_gain": log_odds_gain,
            "state_center_sibling_rotation_empirical_p_improvement": 0.295,
        }
    )
    return candidate


def _state_center_sibling_rotation_production() -> dict[str, object]:
    return _candidate(
        "Q",
        "C",
        "S",
        joint=0.0163,
        drop=0.0027,
        posterior=0.97,
        fold_support=1.0,
        practical_fold_support=1.0,
        arm_p=0.03125,
        residual=0.115,
        collider_separation=-0.29,
        median_collider_separation=-0.13,
        state_wald=True,
        fold_joint=-0.01,
        fold_posterior=0.94,
        outer_wald_view_support=1.0,
    )


def test_state_loaded_center_uses_one_loading_equivalent_sibling_rotation() -> None:
    decision = _select(
        [
            _state_center_sibling_rotation_production(),
            _state_center_sibling_rotation_candidate(),
            _candidate("R", "P", "S"),
        ],
        _direct(("C", "S"), center="Q"),
        state_membership=_state(selected=("Q",)),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.STATE_CENTER_SIBLING_ROTATION_ACTION
    assert bool(decision.state_center_sibling_rotation_selected)
    assert decision.state_center_sibling_rotation_eligible_candidates == 1
    assert decision.state_center_sibling_rotation_selected_displaced_node == "S"
    assert decision.state_center_sibling_rotation_selected_new_node == "P"
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


def test_state_center_sibling_rotation_requires_sparse_count_improvement() -> None:
    decision = _select(
        [
            _state_center_sibling_rotation_production(),
            _state_center_sibling_rotation_candidate(log_odds_gain=0.24),
            _candidate("R", "P", "S"),
        ],
        _direct(("C", "S"), center="Q"),
        state_membership=_state(selected=("Q",)),
    )
    assert not bool(decision.selected_replacement)
    assert decision.state_center_sibling_rotation_eligible_candidates == 0


def _same_center_substitution_production() -> dict[str, object]:
    return _candidate(
        "C",
        "N",
        "Q",
        joint=0.00636,
        drop=-0.00048,
        posterior=0.892,
        fold_support=1.0,
        practical_fold_support=0.8,
        arm_p=0.09375,
        outer_p=0.96875,
        residual=0.081,
        collider_separation=-0.134,
        median_collider_separation=-0.050,
        state_wald=False,
        fold_joint=-0.00058,
        fold_posterior=0.58,
        outer_wald_view_support=0.0,
    )


def _same_center_substitution_candidate(
    *, minimum_separation: float = -0.031
) -> dict[str, object]:
    return _candidate(
        "C",
        "P",
        "Q",
        joint=0.00553,
        drop=-0.00131,
        posterior=0.54,
        fold_support=0.6,
        practical_fold_support=0.2,
        arm_p=17.0 / 32.0,
        outer_p=17.0 / 32.0,
        residual=-0.0345,
        collider_separation=minimum_separation,
        median_collider_separation=0.083,
        state_wald=True,
        fold_joint=-0.00138,
        fold_posterior=0.353,
        outer_wald_view_support=1.0,
    )


def _same_center_substitution_runner() -> dict[str, object]:
    return _candidate(
        "C",
        "Q",
        "S",
        joint=0.0061,
        drop=-0.0007,
        posterior=0.66,
        fold_support=0.8,
        practical_fold_support=0.6,
        arm_p=0.3125,
        outer_p=0.72,
        residual=0.02,
        collider_separation=-0.165,
        median_collider_separation=0.149,
        state_wald=True,
        fold_joint=-0.0012,
        fold_posterior=0.44,
        outer_wald_view_support=1.0,
    )


def test_state_inconsistent_parent_uses_stable_same_center_substitution() -> None:
    decision = _select(
        [
            _same_center_substitution_production(),
            _same_center_substitution_candidate(),
            _same_center_substitution_runner(),
        ],
        _direct(("N", "Q"), center="C"),
        state_membership=_state(selected=("Q",)),
    )
    assert bool(decision.selected_replacement)
    assert (
        decision.action
        == collider.STATE_INCONSISTENT_SAME_CENTER_SUBSTITUTION_ACTION
    )
    assert bool(decision.same_center_substitution_selected)
    assert decision.same_center_substitution_eligible_candidates == 1
    assert decision.same_center_substitution_selected_collider_separation_improvement >= 0.10
    assert decision.same_center_substitution_selected_collider_separation_runner_margin >= 0.10
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


def test_same_center_substitution_requires_cross_resolution_runner_margin() -> None:
    decision = _select(
        [
            _same_center_substitution_production(),
            _same_center_substitution_candidate(minimum_separation=-0.07),
            _same_center_substitution_runner(),
        ],
        _direct(("N", "Q"), center="C"),
        state_membership=_state(selected=("Q",)),
    )
    assert not bool(decision.selected_replacement)
    assert decision.same_center_substitution_eligible_candidates == 0


def _state_parent_rotation_candidate() -> dict[str, object]:
    return _candidate(
        "Q",
        "C",
        "P",
        joint=0.0055,
        drop=0.0006,
        posterior=0.94,
        fold_support=1.0,
        practical_fold_support=1.0,
        arm_p=11.0 / 32.0,
        state_wald=False,
        fold_joint=-0.004,
        fold_posterior=0.82,
        reverse=0.0016,
        posterior_reverse_margin=0.30,
    )


def test_state_loaded_parent_rotates_only_to_one_residual_supported_center() -> None:
    decision = _select(
        [
            _state_inconsistent_neighbor_production(),
            _state_parent_rotation_candidate(),
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
        state_membership=_state(selected=("Q", "R", "S")),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.STATE_LOADED_PARENT_ROTATION_ACTION
    assert bool(decision.state_parent_rotation_selected)
    assert decision.state_parent_rotation_eligible_candidates == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("Q", "C", "P")


def test_state_parent_rotation_rejects_a_state_loaded_candidate_parent() -> None:
    decision = _select(
        [
            _state_inconsistent_neighbor_production(),
            _state_parent_rotation_candidate(),
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
        state_membership=_state(selected=("P", "Q", "R", "S")),
    )
    assert not bool(decision.selected_replacement)
    assert not bool(decision.state_parent_rotation_selected)
    assert decision.state_parent_rotation_eligible_candidates == 0


def _state_parent_topology_rotation_candidate(
    *, arm_p: float = 0.0625
) -> dict[str, object]:
    return _candidate(
        "Q",
        "C",
        "P",
        joint=0.0052,
        drop=-0.0022,
        posterior=0.47,
        fold_support=0.4,
        practical_fold_support=0.2,
        arm_p=arm_p,
        outer_p=0.9375,
        residual=0.07,
        collider_separation=-0.065,
        median_collider_separation=0.147,
        state_wald=True,
        fold_joint=0.0015,
        fold_posterior=0.16,
        reverse=-0.001,
        posterior_reverse_margin=-0.10,
        outer_wald_view_support=1.0,
    )


def _state_parent_topology_rotation_production() -> dict[str, object]:
    production = _state_inconsistent_neighbor_production()
    production.update(
        {
            "minimum_collider_separation": -0.135,
            "median_collider_separation": -0.032,
        }
    )
    return production


def test_state_parent_rotation_accepts_unique_cross_resolution_topology() -> None:
    decision = _select(
        [
            _state_parent_topology_rotation_production(),
            _state_parent_topology_rotation_candidate(),
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
        state_membership=_state(selected=("Q", "R", "S")),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == collider.STATE_LOADED_PARENT_ROTATION_ACTION
    assert decision.state_parent_rotation_selected_mode == (
        "unique_cross_resolution_topology_rotation"
    )
    assert decision.state_parent_rotation_topology_eligible_candidates == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("Q", "C", "P")


def test_state_parent_topology_rotation_requires_empirical_null_stability() -> None:
    decision = _select(
        [
            _state_parent_topology_rotation_production(),
            _state_parent_topology_rotation_candidate(arm_p=0.09375),
            _candidate("R", "P", "S"),
        ],
        _direct(("N", "Q"), center="C"),
        state_membership=_state(selected=("Q", "R", "S")),
    )
    assert not bool(decision.selected_replacement)
    assert decision.state_parent_rotation_topology_eligible_candidates == 0


def _same_skeleton_orientation_candidates(
    *,
    runner_separation: float = -0.30,
) -> list[dict[str, object]]:
    production = _orientation_ambiguous_production()
    production["minimum_collider_separation"] = -0.20
    selected = _candidate(
        "C",
        "P",
        "Q",
        joint=0.006,
        drop=-0.001,
        posterior=0.78,
        fold_support=0.8,
        practical_fold_support=0.8,
        arm_p=0.40,
        residual=0.01,
        collider_separation=-0.08,
        state_wald=True,
        outer_wald_view_support=1.0,
    )
    runner = _candidate(
        "P",
        "C",
        "Q",
        joint=-0.001,
        drop=-0.002,
        posterior=0.70,
        fold_support=0.8,
        practical_fold_support=0.8,
        arm_p=0.40,
        residual=-0.05,
        collider_separation=runner_separation,
        state_wald=True,
        outer_wald_view_support=1.0,
    )
    return [production, selected, runner]


def test_ambiguous_orientation_uses_unique_residual_same_skeleton_winner() -> None:
    decision = _select(
        _same_skeleton_orientation_candidates(),
        _direct(("P", "C"), center="Q"),
    )
    assert bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == collider.AMBIGUOUS_SKELETON_REORIENTATION_ACTION
    assert bool(decision.ambiguous_skeleton_reorientation_selected)
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")


def test_ambiguous_orientation_demotes_without_residual_runner_margin() -> None:
    decision = _select(
        _same_skeleton_orientation_candidates(runner_separation=-0.10),
        _direct(("P", "C"), center="Q"),
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.demote_production_motif)
    assert decision.action == collider.RETAINED_ORIENTATION_DEMOTION_ACTION
    assert decision.ambiguous_skeleton_reorientation_status == "abstained"


def test_multi_component_production_collapses_only_to_a_strong_existing_star() -> None:
    winner = _strong("C", "P", "Q")
    winner.update(
        {
            "crossfit_joint_gain_over_baseline": 0.017,
            "laplace_fold_min_parent_positive_posterior": 0.91,
            "maximum_arm_empirical_positive_p": 1.0 / 32.0,
            "outer_empirical_null_compatibility_p": 0.50,
        }
    )
    winner["current_arm_support"] = 2
    runner = _candidate("N", "R", "S")
    runner["current_arm_support"] = 2
    distractor = _candidate("R", "P", "S")
    distractor["current_arm_support"] = 0
    production = pd.concat(
        [_direct(("P", "Q"), center="C"), _direct(("R", "S"), center="N")],
        ignore_index=True,
    )

    decision = _select([winner, runner, distractor], production)

    assert bool(decision.selected_replacement)
    assert decision.action == collider.MULTI_COMPONENT_COLLAPSE_ACTION
    assert bool(decision.multi_component_collapse_selected)
    assert decision.multi_component_collapse_status == "selected"
    assert decision.multi_component_collapse_production_edge_count == 4
    assert bool(decision.multi_component_collapse_star_is_production_subset)
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")

    weak_winner = dict(winner)
    weak_winner["crossfit_joint_gain_over_baseline"] = 0.0149
    abstained = _select([weak_winner, runner, distractor], production)
    assert not bool(abstained.multi_component_collapse_selected)
    assert (
        "joint_gain_below_threshold"
        in abstained.multi_component_collapse_abstention_reason
    )


def test_zero_edge_state_consistent_runner_requires_every_uncertainty_gate() -> None:
    candidates = _complete_zero_edge_candidates(state_wald=False)
    runner = next(
        row
        for row in candidates
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "R"}
    )
    runner.update(
        {
            "crossfit_joint_gain_over_baseline": 0.0065,
            "crossfit_min_drop_one_gain": 0.0031,
            "laplace_min_parent_positive_posterior": 0.998,
            "laplace_fold_min_parent_positive_posterior": 0.992,
            "laplace_both_positive_fold_support": 1.0,
            "laplace_both_practical_fold_support": 1.0,
            "maximum_arm_empirical_positive_p": 1.0 / 32.0,
            "residual_star_separation": 0.052,
            "minimum_outer_conditional_wald_view_support": 1.0,
            "two_parent_positive_state_wald_consistency": True,
        }
    )

    selected = _select(candidates, _direct(()))
    assert bool(selected.selected_replacement)
    assert selected.action == collider.ZERO_EDGE_STATE_CONSISTENT_RUNNER_ACTION
    assert selected.zero_edge_runner_status == "selected"
    assert bool(selected.zero_edge_runner_selected)
    assert selected.zero_edge_runner_eligible_candidates == 1
    assert (
        selected.selected_center,
        selected.selected_outer1,
        selected.selected_outer2,
    ) == ("C", "P", "R")

    runner["crossfit_min_drop_one_gain"] = 0.0029
    abstained = _select(candidates, _direct(()))
    assert not bool(abstained.selected_replacement)
    assert abstained.zero_edge_runner_status == "abstained"
    assert (
        abstained.zero_edge_runner_abstention_reason
        == "no_same_center_candidate_passed_all_uncertainty_gates"
    )


def _orientation_ambiguous_production(
    *,
    posterior: float = 0.844657,
    arm_p: float = 0.21875,
    posterior_reverse_margin: float = 0.075058,
) -> dict[str, object]:
    return _candidate(
        "Q",
        "P",
        "C",
        joint=0.0089,
        drop=0.00035,
        posterior=posterior,
        fold_support=1.0,
        arm_p=arm_p,
        outer_p=0.71875,
        state_wald=True,
        fold_joint=-0.006,
        fold_posterior=0.786,
        practical_fold_support=1.0,
        posterior_reverse_margin=posterior_reverse_margin,
        outer_wald_view_support=1.0,
    )


def _select_orientation_ambiguous_production(
    production: dict[str, object],
    *,
    selected_state: tuple[str, ...] = ("Q", "C", "R", "S"),
) -> pd.Series:
    decisions = collider.select_crossfit_hierarchical_collider_replacements(
        candidates=pd.DataFrame(
            [production, _candidate("C", "P", "Q"), _candidate("R", "P", "Q")]
        ),
        state_membership=_state(selected=selected_state),
        current_direct_candidates=_direct(("P", "C"), center="Q"),
    )
    assert len(decisions) == 1
    return decisions.iloc[0]


def test_uncertain_retained_orientation_is_demoted_without_recentering() -> None:
    decision = _select_orientation_ambiguous_production(
        _orientation_ambiguous_production()
    )
    assert not bool(decision.selected_replacement)
    assert bool(decision.demote_production_motif)
    assert decision.action == (
        "demote_retained_orientation_ambiguous_production_motif"
    )
    assert bool(decision.retained_orientation_demotion_considered)
    assert bool(decision.retained_orientation_demotion_selected)
    assert bool(decision.retained_orientation_demotion_evidence_complete)
    assert decision.retained_orientation_demotion_status == "selected"
    assert decision.retained_orientation_demotion_reason == (
        "redundancy_demoted_parent_with_unsupported_orientation"
    )
    assert decision.retained_orientation_demotion_parent_posterior == pytest.approx(
        0.844657
    )
    assert (
        decision.retained_orientation_demotion_maximum_arm_empirical_positive_p
        == pytest.approx(0.21875)
    )
    assert (
        decision.retained_orientation_demotion_reverse_center_posterior_margin
        == pytest.approx(0.075058)
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("laplace_min_parent_positive_posterior", 0.90),
        ("maximum_arm_empirical_positive_p", 0.125),
        ("laplace_min_parent_positive_posterior_reverse_center_margin", 0.10),
    ],
)
def test_retained_orientation_demotion_requires_every_strict_condition(
    column: str,
    value: float,
) -> None:
    production = _orientation_ambiguous_production()
    production[column] = value
    decision = _select_orientation_ambiguous_production(production)
    assert not bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)
    assert decision.action == "retain_current_direct_motif"
    assert decision.retained_orientation_demotion_status == "not_selected"

    supported_parents = _select_orientation_ambiguous_production(
        _orientation_ambiguous_production(),
        selected_state=("P", "Q", "C", "R", "S"),
    )
    assert not bool(supported_parents.demote_production_motif)
    assert not bool(supported_parents.retained_orientation_demotion_considered)


def test_retained_orientation_demotion_abstains_on_missing_consumed_evidence() -> None:
    production = _orientation_ambiguous_production(
        posterior_reverse_margin=np.nan
    )
    decision = _select_orientation_ambiguous_production(production)
    assert not bool(decision.demote_production_motif)
    assert decision.retained_orientation_demotion_status == "abstained"
    assert "missing_or_nonfinite_observed_evidence" in (
        decision.retained_orientation_demotion_reason
    )


def test_tier3_swaps_exactly_one_redundant_parent_near_tie() -> None:
    production = _candidate(
        "C",
        "N",
        "P",
        joint=0.006,
        drop=-0.00005,
        posterior=0.95,
        fold_support=1.0,
        reverse=0.001,
        arm_p=0.25,
        outer_p=0.25,
        residual=0.020,
        collider_separation=-0.01,
        state_wald=False,
    )
    alternative = _candidate(
        "C",
        "P",
        "Q",
        joint=0.006,
        drop=-0.00005,
        posterior=0.95,
        fold_support=1.0,
        reverse=0.001,
        arm_p=0.0625,
        outer_p=0.25,
        residual=0.018,
        collider_separation=0.01,
        state_wald=True,
    )
    decision = _select(
        [production, alternative, _candidate("C", "P", "R")],
        _direct(("N", "P")),
    )
    assert bool(decision.selected_replacement)
    assert decision.action == "tier3_state_consistent_one_parent_swap"
    assert {decision.selected_outer1, decision.selected_outer2} == {"P", "Q"}


@pytest.mark.parametrize("same_as_production", [False, True])
def test_counterexamples_abstain_without_state_collider_consistency(
    same_as_production: bool,
) -> None:
    false_top = _strong("C", "N", "P")
    false_top["two_parent_positive_state_wald_consistency"] = False
    false_top["minimum_collider_separation"] = -0.181
    direct = _direct(("N", "P")) if same_as_production else _direct(())
    decision = _select([false_top, _candidate("R", "P", "Q")], direct)
    assert not bool(decision.selected_replacement)
    assert decision.action == "retain_current_direct_motif"


def _votes(features: tuple[str, ...], n_views: int) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(features, 2):
        for vote in range(n_views):
            rows.append(
                {
                    "source_assay": ASSAY,
                    "node1": left,
                    "node2": right,
                    "vote": vote,
                    "selected": True,
                    "empirical_p_positive": 0.01,
                    "pooled_log_odds": 1.5,
                }
            )
    return pd.DataFrame(rows)


def _base_views(
    features: tuple[str, ...], n_views: int, selected_pair: tuple[str, str] | None = None
) -> dict[str, pd.DataFrame]:
    rows = [] if selected_pair is None else [
        {
            "source_assay": ASSAY,
            "node1": selected_pair[0],
            "node2": selected_pair[1],
            "selected": True,
            "edge_type": collider.DIRECT_EDGE_TYPE,
        }
    ]
    columns = ["source_assay", "node1", "node2", "selected", "edge_type"]
    return {
        f"v{view}": pd.DataFrame(rows, columns=columns) for view in range(n_views)
    }


def _zero_edge_abstained_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": "retain_current_direct_motif",
                "selected_replacement": False,
                "demote_production_motif": False,
                "selected_unoriented_adjacency": False,
                "zero_edge_rank_fusion_status": "abstained",
                "zero_edge_rank_fusion_abstention_reason": (
                    "selected_star_lacked_all_fold_practical_support"
                ),
                "zero_edge_rank_fusion_selected_center": "C",
                "zero_edge_rank_fusion_selected_outer1": "P",
                "zero_edge_rank_fusion_selected_outer2": "Q",
                "zero_edge_rank_fusion_selected_center_score": 1.20,
                "zero_edge_rank_fusion_runner_center_score": 1.10,
                "zero_edge_rank_fusion_center_score_margin": 0.10,
            }
        ]
    )


def _all_state_topology_candidates(*, ambiguous: bool = False) -> pd.DataFrame:
    rows = _complete_zero_edge_candidates(state_wald=False)
    for row in rows:
        row["min_arm_lor"] = 0.0
        row["max_arm_empirical_p"] = 1.0
        row["raw_generative_score"] = -1.0
    selected = next(
        row
        for row in rows
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "Q"}
    )
    selected.update(
        {
            "minimum_collider_separation": -0.011,
            "median_collider_separation": 0.069,
            "raw_generative_score": 0.0002,
            "crossfit_joint_gain_over_baseline": -0.0025,
            "crossfit_min_drop_one_gain": -0.0019,
            "crossfit_joint_gain_fold_min": -0.0052,
            "laplace_min_parent_positive_posterior": 0.65,
            "laplace_fold_min_parent_positive_posterior": 0.28,
            "laplace_both_positive_fold_support": 0.80,
            "laplace_both_practical_fold_support": 0.60,
            "maximum_arm_empirical_positive_p": 13.0 / 32.0,
            "outer_empirical_null_compatibility_p": 0.28125,
            "minimum_outer_conditional_wald_view_support": 1.0,
            "two_parent_positive_state_wald_consistency": True,
            "min_arm_lor": 0.92,
            "max_arm_empirical_p": 0.42,
        }
    )
    if ambiguous:
        runner = next(
            row
            for row in rows
            if row["center"] == "P"
            and {str(row["outer1"]), str(row["outer2"])} == {"C", "Q"}
        )
        for key, value in selected.items():
            if key not in {"center", "outer1", "outer2"}:
                runner[key] = value
    return pd.DataFrame(rows)


def test_zero_edge_all_state_topology_selects_one_complete_unique_star() -> None:
    decisions = collider._annotate_zero_edge_all_state_topology_completion(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=pd.DataFrame(),
        candidates=_all_state_topology_candidates(),
        state_membership=_state(selected=("C", "P", "Q", "R", "S")),
    )
    decision = decisions.iloc[0]
    assert bool(decision.zero_edge_all_state_topology_considered)
    assert bool(decision.zero_edge_all_state_topology_candidate_universe_complete)
    assert decision.zero_edge_all_state_topology_eligible_candidates == 1
    assert bool(decision.zero_edge_all_state_topology_selected)
    assert bool(decision.selected_replacement)
    assert decision.action == collider.ZERO_EDGE_ALL_STATE_TOPOLOGY_COMPLETION_ACTION
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")
    assert decision.zero_edge_all_state_topology_selected_beta_arm_log_odds == 0.92
    assert decision.zero_edge_all_state_topology_status == "selected"


def test_zero_edge_all_state_topology_requires_uniqueness_and_empty_layers() -> None:
    ambiguous = collider._annotate_zero_edge_all_state_topology_completion(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=pd.DataFrame(),
        candidates=_all_state_topology_candidates(ambiguous=True),
        state_membership=_state(selected=("C", "P", "Q", "R", "S")),
    ).iloc[0]
    assert not bool(ambiguous.selected_replacement)
    assert ambiguous.zero_edge_all_state_topology_eligible_candidates == 2
    assert ambiguous.zero_edge_all_state_topology_status == "abstained"

    unresolved = collider._annotate_zero_edge_all_state_topology_completion(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        candidates=_all_state_topology_candidates(),
        state_membership=_state(selected=("C", "P", "Q", "R", "S")),
    ).iloc[0]
    assert not bool(unresolved.zero_edge_all_state_topology_considered)
    assert unresolved.zero_edge_all_state_topology_reason == (
        "assay_has_existing_unresolved_adjacency"
    )


def _rescue_only_unresolved(
    pair: tuple[str, str] = ("C", "P"),
    *,
    rescue: bool = True,
    base: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "node1": min(pair),
                "endpoint1": "CIRCLE",
                "node2": max(pair),
                "endpoint2": "CIRCLE",
                "edge_type": collider.UNRESOLVED_EDGE_TYPE,
                "was_base_selected": base,
                "was_rescue_selected": rescue,
                "mean_empirical_p_positive": 1.0 / 300.0,
                "mean_pooled_log_odds": 2.20,
                "reason": "component_not_identifiable",
            }
        ]
    )


def _rescue_anchored_completion_candidates(
    *,
    ambiguous: bool = False,
) -> pd.DataFrame:
    rows = _complete_zero_edge_candidates(state_wald=False)
    identities = [("C", {"P", "R"})]
    if ambiguous:
        identities.append(("C", {"P", "S"}))
    for center, parents in identities:
        selected = next(
            row
            for row in rows
            if row["center"] == center
            and {str(row["outer1"]), str(row["outer2"])} == parents
        )
        selected.update(
            {
                "crossfit_joint_gain_over_baseline": 0.0045,
                "crossfit_min_drop_one_gain": 0.001,
                "laplace_min_parent_positive_posterior": 0.92,
                "laplace_fold_min_parent_positive_posterior": 0.91,
                "laplace_both_positive_fold_support": 1.0,
                "laplace_both_practical_fold_support": 1.0,
                "maximum_arm_empirical_positive_p": 5.0 / 32.0,
                "outer_empirical_null_compatibility_p": 0.25,
                "minimum_outer_conditional_wald_view_support": 1.0,
                "two_parent_positive_state_wald_consistency": True,
                "residual_star_separation": 0.005,
            }
        )
    frame = pd.DataFrame(rows)
    frame["raw_generative_score"] = collider._raw_score(frame)
    return frame


def _unanimous_rescue_orientation_candidates(
    *,
    unanimous: bool = True,
) -> pd.DataFrame:
    rows = _complete_zero_edge_candidates(state_wald=False)
    prior_rank_winner = next(
        row
        for row in rows
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "Q"}
    )
    prior_rank_winner.update(_candidate("C", "P", "Q"))
    third_features = ("N", "Q", "R", "S")
    for third in third_features:
        centered_c = next(
            row
            for row in rows
            if row["center"] == "C"
            and {str(row["outer1"]), str(row["outer2"])} == {"P", third}
        )
        centered_p = next(
            row
            for row in rows
            if row["center"] == "P"
            and {str(row["outer1"]), str(row["outer2"])} == {"C", third}
        )
        centered_c[
            "crossfit_joint_gain_over_baseline_reverse_center_margin"
        ] = 0.0006
        centered_p[
            "crossfit_joint_gain_over_baseline_reverse_center_margin"
        ] = -0.0006
    if not unanimous:
        failing = next(
            row
            for row in rows
            if row["center"] == "C"
            and {str(row["outer1"]), str(row["outer2"])} == {"P", "S"}
        )
        failing[
            "crossfit_joint_gain_over_baseline_reverse_center_margin"
        ] = 0.0004
    selected = next(
        row
        for row in rows
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "R"}
    )
    selected.update(
        {
            "crossfit_joint_gain_over_baseline": 0.0034,
            "crossfit_min_drop_one_gain": -0.0006,
            "laplace_min_parent_positive_posterior": 0.64,
            "laplace_fold_min_parent_positive_posterior": 0.43,
            "laplace_both_positive_fold_support": 0.6,
            "laplace_both_practical_fold_support": 0.6,
            "maximum_arm_empirical_positive_p": 5.0 / 32.0,
            "outer_empirical_null_compatibility_p": 0.10,
            "minimum_outer_conditional_wald_view_support": 1.0 / 3.0,
            "two_parent_positive_state_wald_consistency": False,
            "residual_star_separation": 0.0,
        }
    )
    frame = pd.DataFrame(rows)
    frame["raw_generative_score"] = collider._raw_score(frame)
    return frame


def _dominant_rescue_unresolved(
    *, runner_log_odds: float = 1.80,
) -> pd.DataFrame:
    return pd.concat(
        [
            _rescue_only_unresolved(("C", "P")).assign(
                mean_pooled_log_odds=2.70,
            ),
            _rescue_only_unresolved(("R", "S")).assign(
                mean_empirical_p_positive=0.007,
                mean_pooled_log_odds=runner_log_odds,
            ),
        ],
        ignore_index=True,
    )


def _dominant_rescue_decisions(
    *, forward_probability: float = 0.20, reverse_probability: float = 0.05
) -> pd.DataFrame:
    return _zero_edge_abstained_decisions().assign(
        zero_edge_rank_fusion_all_edge_scores_json=json.dumps(
            [
                {
                    "parent": "P",
                    "child": "C",
                    "probability": forward_probability,
                },
                {
                    "parent": "C",
                    "child": "P",
                    "probability": reverse_probability,
                },
            ],
            sort_keys=True,
        )
    )


def _dominant_rescue_candidates() -> pd.DataFrame:
    rows = _complete_zero_edge_candidates(state_wald=False)
    selected = next(
        row
        for row in rows
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "Q"}
    )
    selected.update(
        {
            "crossfit_joint_gain_over_baseline": 0.0045,
            "crossfit_min_drop_one_gain": -0.003,
            "laplace_min_parent_positive_posterior": 0.73,
            "laplace_fold_min_parent_positive_posterior": 0.31,
            "laplace_both_positive_fold_support": 0.8,
            "laplace_both_practical_fold_support": 0.6,
            "maximum_arm_empirical_positive_p": 10.0 / 32.0,
            "outer_empirical_null_compatibility_p": 0.75,
            "residual_star_separation": 0.01,
        }
    )
    frame = pd.DataFrame(rows)
    frame["raw_generative_score"] = collider._raw_score(frame)
    return frame


def _three_spoke_unresolved(*, third_log_odds: float = 1.65) -> pd.DataFrame:
    rows = [
        ("C", "P", True, 0.0020, 2.50),
        ("C", "R", False, 0.0025, 1.80),
        ("C", "Q", True, 0.0040, third_log_odds),
    ]
    return pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "node1": min(left, right),
                "endpoint1": "CIRCLE",
                "node2": max(left, right),
                "endpoint2": "CIRCLE",
                "edge_type": collider.UNRESOLVED_EDGE_TYPE,
                "component_id": f"{ASSAY}__component0",
                "was_base_selected": base,
                "was_rescue_selected": True,
                "mean_empirical_p_positive": empirical_p,
                "mean_pooled_log_odds": log_odds,
                "reason": "component_not_identifiable",
            }
            for left, right, base, empirical_p, log_odds in rows
        ]
    )


def _spoke_consensus_candidates() -> pd.DataFrame:
    rows = _complete_zero_edge_candidates(state_wald=False)
    selected = next(
        row
        for row in rows
        if row["center"] == "C"
        and {str(row["outer1"]), str(row["outer2"])} == {"P", "R"}
    )
    selected.update(
        {
            "crossfit_joint_gain_over_baseline": 0.009,
            "crossfit_min_drop_one_gain": 0.0021,
            "laplace_min_parent_positive_posterior": 0.996,
            "laplace_fold_min_parent_positive_posterior": 0.985,
            "laplace_both_positive_fold_support": 1.0,
            "laplace_both_practical_fold_support": 1.0,
            "maximum_arm_empirical_positive_p": 1.0 / 32.0,
            "minimum_outer_conditional_wald_view_support": 1.0,
            "two_parent_positive_state_wald_consistency": True,
            "residual_star_separation": 0.05,
        }
    )
    frame = pd.DataFrame(rows)
    frame["raw_generative_score"] = collider._raw_score(frame)
    return frame


def _wrong_zero_edge_completion_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": "tier1_nonexact_production_completion",
                "selected_replacement": True,
                "demote_production_motif": False,
                "production_exact_two_arm_star": False,
                "selected_center": "C",
                "selected_outer1": "P",
                "selected_outer2": "Q",
                "selected_raw_generative_score": 0.06,
                "selection_runner_margin": 0.01,
                "selected_score_minus_production_score": np.nan,
                "zero_edge_rank_fusion_status": "not_needed_higher_tier_selected",
                "zero_edge_rank_fusion_selected_center": "C",
                "zero_edge_rank_fusion_selected_outer1": "P",
                "zero_edge_rank_fusion_selected_outer2": "Q",
            }
        ]
    )


def test_three_spoke_beta_consensus_overrides_wrong_zero_edge_parent() -> None:
    candidates = _spoke_consensus_candidates()
    unresolved_input = _three_spoke_unresolved()
    decisions = collider._annotate_zero_edge_spoke_consensus(
        decisions=_wrong_zero_edge_completion_decision(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
        candidates=candidates,
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    decision = decisions.iloc[0]
    assert bool(decision.selected_replacement)
    assert decision.action == collider.ZERO_EDGE_SPOKE_CONSENSUS_ACTION
    assert bool(decision.zero_edge_spoke_consensus_selected)
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "R")

    annotated = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=decisions,
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
        candidates=candidates,
    )
    direct, unresolved = collider._apply_decisions(
        decisions=annotated,
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
        beta_vote_evidence=_votes(("C", "N", "P", "Q", "R", "S"), 6),
        base_edge_views=_base_views(("C", "N", "P", "Q", "R", "S"), 6),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    assert {
        (row.node1, row.node2) for row in direct.itertuples(index=False)
    } == {("C", "P"), ("C", "R")}
    assert set(direct["collider_center"]) == {"C"}
    assert {(row.node1, row.node2) for row in unresolved.itertuples(index=False)} == {
        ("C", "Q")
    }


def test_three_spoke_consensus_abstains_without_beta_arm_margin() -> None:
    decisions = collider._annotate_zero_edge_spoke_consensus(
        decisions=_wrong_zero_edge_completion_decision(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_three_spoke_unresolved(
            third_log_odds=1.75
        ),
        candidates=_spoke_consensus_candidates(),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    decision = decisions.iloc[0]
    assert not bool(decision.zero_edge_spoke_consensus_selected)
    assert decision.action == "tier1_nonexact_production_completion"
    assert decision.zero_edge_spoke_consensus_status == "abstained"
    assert decision.zero_edge_spoke_consensus_reason == (
        "second_beta_spoke_lacks_margin_over_third"
    )


def test_unique_rescue_anchored_star_is_oriented_only_after_all_gates() -> None:
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        candidates=_rescue_anchored_completion_candidates(),
    )
    decision = decisions.iloc[0]
    assert bool(decision.selected_replacement)
    assert not bool(decision.selected_unoriented_adjacency)
    assert decision.action == collider.ZERO_EDGE_RESCUE_ANCHORED_COMPLETION_ACTION
    assert bool(decision.rescue_anchored_completion_selected)
    assert decision.rescue_anchored_completion_eligible_candidates == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "R")

    direct, unresolved = collider._apply_decisions(
        decisions=decisions,
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        beta_vote_evidence=_votes(("C", "N", "P", "Q", "R", "S"), 6),
        base_edge_views=_base_views(("C", "N", "P", "Q", "R", "S"), 6),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    assert len(direct) == 2
    assert set(direct["collider_center"]) == {"C"}
    assert set(direct["orientation_source"]) == {
        "crossfit_hierarchical_zero_edge_rescue_anchored_completion"
    }
    assert unresolved.empty


def test_dominant_competing_rescue_selects_one_directional_sparse_completion() -> None:
    candidates = _dominant_rescue_candidates()
    unresolved_input = _dominant_rescue_unresolved()
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_dominant_rescue_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
        candidates=candidates,
    )
    decision = decisions.iloc[0]
    assert bool(decision.selected_replacement)
    assert not bool(decision.selected_unoriented_adjacency)
    assert decision.action == collider.ZERO_EDGE_DOMINANT_RESCUE_COMPLETION_ACTION
    assert bool(decision.dominant_rescue_completion_selected)
    assert decision.dominant_rescue_completion_eligible_candidates == 1
    assert decision.dominant_rescue_completion_log_odds_margin == pytest.approx(0.9)
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "Q")

    direct, _ = collider._apply_decisions(
        decisions=decisions,
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
        beta_vote_evidence=_votes(("C", "N", "P", "Q", "R", "S"), 6),
        base_edge_views=_base_views(("C", "N", "P", "Q", "R", "S"), 6),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    assert len(direct) == 2
    assert set(direct["collider_center"]) == {"C"}
    assert set(direct["orientation_source"]) == {
        "crossfit_hierarchical_zero_edge_dominant_rescue_completion"
    }


@pytest.mark.parametrize(
    ("runner_log_odds", "forward_probability", "reverse_probability"),
    [
        (2.30, 0.20, 0.05),
        (1.80, 0.15, 0.12),
    ],
)
def test_dominant_competing_rescue_abstains_without_both_margins(
    runner_log_odds: float,
    forward_probability: float,
    reverse_probability: float,
) -> None:
    annotated = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_dominant_rescue_decisions(
            forward_probability=forward_probability,
            reverse_probability=reverse_probability,
        ),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_dominant_rescue_unresolved(
            runner_log_odds=runner_log_odds
        ),
        candidates=_dominant_rescue_candidates(),
    )
    decision = annotated.iloc[0]
    assert not bool(decision.selected_replacement)
    assert not bool(decision.dominant_rescue_completion_selected)


def test_unanimous_rescue_orientation_recovers_one_sparse_candidate() -> None:
    candidates = _unanimous_rescue_orientation_candidates()
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        candidates=candidates,
    )
    decision = decisions.iloc[0]
    assert bool(decision.selected_replacement)
    assert not bool(decision.selected_unoriented_adjacency)
    assert decision.action == (
        collider.ZERO_EDGE_RESCUE_ORIENTATION_COMPLETION_ACTION
    )
    assert bool(decision.rescue_orientation_completion_selected)
    assert decision.rescue_orientation_completion_passing_endpoints == 1
    assert (
        decision.selected_center,
        decision.selected_outer1,
        decision.selected_outer2,
    ) == ("C", "P", "R")


def test_sparse_rescue_orientation_abstains_without_unanimous_center() -> None:
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        candidates=_unanimous_rescue_orientation_candidates(unanimous=False),
    )
    decision = decisions.iloc[0]
    assert not bool(decision.selected_replacement)
    assert bool(decision.selected_unoriented_adjacency)
    assert not bool(decision.rescue_orientation_completion_selected)
    assert decision.rescue_orientation_completion_status == "abstained"


def test_ambiguous_rescue_anchored_stars_remain_one_unoriented_adjacency() -> None:
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        candidates=_rescue_anchored_completion_candidates(ambiguous=True),
    )
    decision = decisions.iloc[0]
    assert not bool(decision.selected_replacement)
    assert bool(decision.selected_unoriented_adjacency)
    assert decision.action == collider.UNORIENTED_RESCUE_OVERLAP_ACTION
    assert decision.rescue_anchored_completion_status == "abstained"
    assert decision.rescue_anchored_completion_eligible_candidates == 2


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("laplace_fold_min_parent_positive_posterior", 0.899),
        ("crossfit_min_drop_one_gain", 0.0),
        ("maximum_arm_empirical_positive_p", 0.156251),
        ("outer_empirical_null_compatibility_p", 0.2499),
        ("two_parent_positive_state_wald_consistency", False),
    ],
)
def test_rescue_anchored_completion_requires_every_uncertainty_gate(
    column: str,
    value: object,
) -> None:
    candidates = _rescue_anchored_completion_candidates()
    mask = (
        candidates["center"].eq("C")
        & candidates["outer1"].eq("P")
        & candidates["outer2"].eq("R")
    )
    candidates.loc[mask, column] = value
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
        candidates=candidates,
    )
    decision = decisions.iloc[0]
    assert not bool(decision.selected_replacement)
    assert bool(decision.selected_unoriented_adjacency)
    assert decision.rescue_anchored_completion_eligible_candidates == 0


def test_unique_rescue_rank_overlap_selects_only_an_unoriented_adjacency() -> None:
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=_rescue_only_unresolved(),
    )
    decision = decisions.iloc[0]
    assert bool(decision.selected_unoriented_adjacency)
    assert decision.action == collider.UNORIENTED_RESCUE_OVERLAP_ACTION
    assert (decision.unoriented_adjacency_node1, decision.unoriented_adjacency_node2) == (
        "C",
        "P",
    )
    assert bool(decision.unoriented_adjacency_rank_winner_arm_overlap)
    assert not bool(decision.selected_replacement)
    assert not bool(decision.demote_production_motif)


@pytest.mark.parametrize("failure", ["none", "two", "base", "nonoverlap", "direct"])
def test_unoriented_adjacency_abstains_without_every_structural_guard(
    failure: str,
) -> None:
    unresolved = _rescue_only_unresolved()
    direct = _direct(())
    if failure == "none":
        unresolved = unresolved.assign(was_rescue_selected=False)
    elif failure == "two":
        unresolved = pd.concat(
            [unresolved, _rescue_only_unresolved(("C", "Q"))],
            ignore_index=True,
        )
    elif failure == "base":
        unresolved = unresolved.assign(was_base_selected=True)
    elif failure == "nonoverlap":
        unresolved = _rescue_only_unresolved(("N", "R"))
    elif failure == "direct":
        direct = _direct(("P",))
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=direct,
        current_demoted_or_unresolved=unresolved,
    )
    assert not bool(decisions.iloc[0].selected_unoriented_adjacency)


@pytest.mark.parametrize("column", ["was_base_selected", "was_rescue_selected"])
def test_unoriented_adjacency_requires_strict_upstream_boolean_flags(
    column: str,
) -> None:
    unresolved = _rescue_only_unresolved()
    unresolved[column] = unresolved[column].astype(object)
    unresolved.loc[0, column] = "False"
    with pytest.raises(ValueError, match="must contain only Boolean values"):
        collider._annotate_unique_rescue_rank_overlap_adjacencies(
            decisions=_zero_edge_abstained_decisions(),
            current_direct_candidates=_direct(()),
            current_demoted_or_unresolved=unresolved,
        )


@pytest.mark.parametrize(
    "column", ["mean_empirical_p_positive", "mean_pooled_log_odds"]
)
def test_unoriented_adjacency_rejects_missing_consumed_evidence(column: str) -> None:
    unresolved = _rescue_only_unresolved()
    unresolved.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="requires finite empirical"):
        collider._annotate_unique_rescue_rank_overlap_adjacencies(
            decisions=_zero_edge_abstained_decisions(),
            current_direct_candidates=_direct(()),
            current_demoted_or_unresolved=unresolved,
        )


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        (
            "mean_empirical_p_positive",
            0.0200001,
            "unique_rescue_empirical_p_exceeds_frozen_limit",
        ),
        (
            "mean_pooled_log_odds",
            0.999999,
            "unique_rescue_log_odds_below_frozen_minimum",
        ),
    ],
)
def test_unoriented_adjacency_rechecks_frozen_rescue_strength(
    column: str,
    value: float,
    reason: str,
) -> None:
    unresolved = _rescue_only_unresolved()
    unresolved.loc[0, column] = value
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved,
    )
    decision = decisions.iloc[0]
    assert not bool(decision.selected_unoriented_adjacency)
    assert decision.unoriented_adjacency_reason == reason


def test_unoriented_adjacency_is_emitted_as_one_circle_pair_and_is_idempotent() -> None:
    unresolved_input = _rescue_only_unresolved()
    decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
    )
    direct, unresolved = collider._apply_decisions(
        decisions=decisions,
        current_direct_candidates=_direct(()),
        current_demoted_or_unresolved=unresolved_input,
        beta_vote_evidence=_votes(("C", "N", "P", "Q", "R", "S"), 6),
        base_edge_views=_base_views(("C", "N", "P", "Q", "R", "S"), 6),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    assert len(direct) == 1
    assert set(direct["endpoint1"]) == {"CIRCLE"}
    assert set(direct["endpoint2"]) == {"CIRCLE"}
    assert set(direct["collider_center"]) == {""}
    assert set(direct["orientation_source"]) == {
        "crossfit_hierarchical_unique_rescue_rank_overlap"
    }
    assert unresolved.empty
    assert collider._current_directed_edges(direct).empty

    rerun_decisions = collider._annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=_zero_edge_abstained_decisions(),
        current_direct_candidates=direct,
        current_demoted_or_unresolved=unresolved,
    )
    rerun_direct, rerun_unresolved = collider._apply_decisions(
        decisions=rerun_decisions,
        current_direct_candidates=direct,
        current_demoted_or_unresolved=unresolved,
        beta_vote_evidence=_votes(("C", "N", "P", "Q", "R", "S"), 6),
        base_edge_views=_base_views(("C", "N", "P", "Q", "R", "S"), 6),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    pd.testing.assert_frame_equal(rerun_direct, direct)
    pd.testing.assert_frame_equal(rerun_unresolved, unresolved)


def test_displaced_edges_keep_honest_base_selected_unresolved_semantics() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": "tier1_exact_production_disagreement",
                "selected_replacement": True,
                "demote_production_motif": False,
                "selected_unoriented_adjacency": False,
                "selected_center": "C",
                "selected_outer1": "P",
                "selected_outer2": "Q",
                "selected_raw_generative_score": 0.06,
                "selection_runner_margin": 0.01,
            }
        ]
    )
    features = ("C", "N", "P", "Q", "R", "S")
    direct, unresolved = collider._apply_decisions(
        decisions=decisions,
        current_direct_candidates=_direct(("N", "R")),
        current_demoted_or_unresolved=pd.DataFrame(
            columns=["source_assay", "node1", "node2", "was_base_selected"]
        ),
        beta_vote_evidence=_votes(features, 6),
        base_edge_views=_base_views(features, 6, ("C", "N")),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    assert set(direct["collider_center"]) == {"C"}
    assert set(direct["observed_refinement_action"]) == {""}
    assert set(direct["crossfit_hierarchical_action"]) == {
        "tier1_exact_production_disagreement"
    }
    flags = {
        frozenset((row.node1, row.node2)): bool(row.was_base_selected)
        for row in unresolved.itertuples(index=False)
    }
    assert flags[frozenset(("C", "N"))]
    assert not flags[frozenset(("C", "R"))]
    assert "preserve_in_consensus" not in unresolved.columns


def test_orientation_demotion_emits_only_auditable_unresolved_arms() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": collider.RETAINED_ORIENTATION_DEMOTION_ACTION,
                "selected_replacement": False,
                "demote_production_motif": True,
                "selected_unoriented_adjacency": False,
                "production_center": "Q",
                "production_outer1": "C",
                "production_outer2": "P",
                "selected_center": "",
                "selected_outer1": "",
                "selected_outer2": "",
            }
        ]
    )
    features = ("C", "N", "P", "Q", "R", "S")
    current = _direct(("P", "C"), center="Q").assign(
        observed_refinement_action="reselect_overcomplete_same_center_star",
        prior_metadata="preserved_in_displacement_audit",
    )
    direct, unresolved = collider._apply_decisions(
        decisions=decisions,
        current_direct_candidates=current,
        current_demoted_or_unresolved=pd.DataFrame(
            columns=["source_assay", "node1", "node2", "was_base_selected"]
        ),
        beta_vote_evidence=_votes(features, 6),
        base_edge_views=_base_views(features, 6, ("C", "Q")),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    assert direct.empty
    assert len(unresolved) == 2
    assert set(unresolved["endpoint1"]) == {"CIRCLE"}
    assert set(unresolved["endpoint2"]) == {"CIRCLE"}
    assert set(unresolved["edge_type"]) == {collider.UNRESOLVED_EDGE_TYPE}
    assert set(unresolved["reason"]) == {
        "retained_orientation_ambiguous_production_motif"
    }
    assert set(unresolved["observed_refinement_action"]) == {""}
    assert set(unresolved["displaced_observed_refinement_action"]) == {
        "reselect_overcomplete_same_center_star"
    }
    assert set(unresolved["crossfit_hierarchical_action"]) == {
        collider.RETAINED_ORIENTATION_DEMOTION_ACTION
    }
    flags = {
        frozenset((row.node1, row.node2)): bool(row.was_base_selected)
        for row in unresolved.itertuples(index=False)
    }
    assert flags[frozenset(("C", "Q"))]
    assert not flags[frozenset(("P", "Q"))]


@pytest.mark.parametrize("invalid", ["True", 1, np.nan])
def test_apply_decisions_rejects_non_boolean_demotion_flags(invalid: object) -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": "retain_current_direct_motif",
                "selected_replacement": False,
                "demote_production_motif": invalid,
                "selected_unoriented_adjacency": False,
            }
        ]
    )
    with pytest.raises(ValueError, match="must contain only Boolean values"):
        collider._apply_decisions(
            decisions=decisions,
            current_direct_candidates=_direct(("P", "C"), center="Q"),
            current_demoted_or_unresolved=pd.DataFrame(),
            beta_vote_evidence=_votes(("C", "N", "P", "Q", "R", "S"), 6),
            base_edge_views=_base_views(("C", "N", "P", "Q", "R", "S"), 6),
            config=collider.CrossfitHierarchicalColliderConfig(),
        )


def test_apply_decisions_rejects_conflicting_or_unlabelled_actions() -> None:
    base = {
        "source_assay": ASSAY,
        "action": collider.RETAINED_ORIENTATION_DEMOTION_ACTION,
        "selected_replacement": False,
        "demote_production_motif": True,
        "selected_unoriented_adjacency": False,
        "production_center": "Q",
        "production_outer1": "C",
        "production_outer2": "P",
    }
    common = {
        "current_direct_candidates": _direct(("P", "C"), center="Q"),
        "current_demoted_or_unresolved": pd.DataFrame(),
        "beta_vote_evidence": _votes(("C", "N", "P", "Q", "R", "S"), 6),
        "base_edge_views": _base_views(("C", "N", "P", "Q", "R", "S"), 6),
        "config": collider.CrossfitHierarchicalColliderConfig(),
    }
    conflicting = pd.DataFrame([{**base, "selected_replacement": True}])
    with pytest.raises(ValueError, match="cannot replace and demote"):
        collider._apply_decisions(decisions=conflicting, **common)

    unlabelled = pd.DataFrame([{**base, "action": "retain_current_direct_motif"}])
    with pytest.raises(ValueError, match="explicit demotion action"):
        collider._apply_decisions(decisions=unlabelled, **common)


def test_base_and_vote_selection_reject_non_boolean_stand_ins() -> None:
    features = ("C", "N", "P", "Q", "R", "S")
    invalid_base = _base_views(features, 6, ("C", "N"))
    invalid_base["v0"]["selected"] = invalid_base["v0"]["selected"].astype(object)
    invalid_base["v0"].loc[:, "selected"] = "False"
    with pytest.raises(ValueError, match="must contain only Boolean values"):
        collider._base_support(invalid_base)

    invalid_votes = _votes(features, 6)
    invalid_votes["selected"] = invalid_votes["selected"].astype(object)
    invalid_votes.loc[invalid_votes.index[0], "selected"] = np.nan
    decisions = pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": "tier1_exact_production_disagreement",
                "selected_replacement": True,
                "demote_production_motif": False,
                "selected_unoriented_adjacency": False,
                "selected_center": "C",
                "selected_outer1": "P",
                "selected_outer2": "Q",
                "selected_raw_generative_score": 0.06,
                "selection_runner_margin": 0.01,
            }
        ]
    )
    with pytest.raises(ValueError, match="must contain only Boolean values"):
        collider._apply_decisions(
            decisions=decisions,
            current_direct_candidates=_direct(("N", "R")),
            current_demoted_or_unresolved=pd.DataFrame(
                columns=["source_assay", "node1", "node2", "was_base_selected"]
            ),
            beta_vote_evidence=invalid_votes,
            base_edge_views=_base_views(features, 6),
            config=collider.CrossfitHierarchicalColliderConfig(),
        )


def test_replacement_preserves_other_assay_metadata_without_action_copy() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": ASSAY,
                "action": "zero_edge_state_gated_rank_fusion_completion",
                "selected_replacement": True,
                "demote_production_motif": False,
                "selected_unoriented_adjacency": False,
                "selected_center": "C",
                "selected_outer1": "P",
                "selected_outer2": "Q",
                "selected_raw_generative_score": 0.02,
                "selection_runner_margin": 0.04,
                "zero_edge_rank_fusion_selected_center_score": 1.20,
                "zero_edge_rank_fusion_runner_center_score": 1.16,
                "zero_edge_rank_fusion_center_score_margin": 0.04,
            }
        ]
    )
    retained = _direct(("X",), center="Y").assign(
        source_assay="chrom_other",
        observed_refinement_action="reselect_overcomplete_same_center_star",
        downstream_guard_marker="preserve_me",
    )
    current = pd.concat([_direct(("N", "R")), retained], ignore_index=True)
    features = ("C", "N", "P", "Q", "R", "S")
    direct, _ = collider._apply_decisions(
        decisions=decisions,
        current_direct_candidates=current,
        current_demoted_or_unresolved=pd.DataFrame(
            columns=["source_assay", "node1", "node2", "was_base_selected"]
        ),
        beta_vote_evidence=_votes(features, 6),
        base_edge_views=_base_views(features, 6),
        config=collider.CrossfitHierarchicalColliderConfig(),
    )
    other = direct.loc[direct["source_assay"].eq("chrom_other")].iloc[0]
    assert other.downstream_guard_marker == "preserve_me"
    assert other.observed_refinement_action == (
        "reselect_overcomplete_same_center_star"
    )
    replacement = direct.loc[direct["source_assay"].eq(ASSAY)]
    assert set(replacement["observed_refinement_action"]) == {""}
    assert set(replacement["crossfit_hierarchical_action"]) == {
        "zero_edge_state_gated_rank_fusion_completion"
    }
    assert set(replacement["wedge_score"]) == {1.20}


def test_public_interfaces_have_no_privileged_parameter_and_reject_label_column() -> None:
    for function in (
        collider.fit_crossfit_collider_world_evidence,
        collider.resolve_crossfit_hierarchical_colliders_v4,
        collider.infer_crossfit_hierarchical_colliders_v4,
        collider.select_crossfit_hierarchical_collider_replacements,
    ):
        parameters = " ".join(inspect.signature(function).parameters).lower()
        assert "truth" not in parameters
        assert "role" not in parameters
        assert "oracle" not in parameters
    labeled = pd.DataFrame([_strong()]).assign(truth_exact_star=True)
    with pytest.raises(ValueError, match="privileged"):
        collider.select_crossfit_hierarchical_collider_replacements(
            candidates=labeled,
            state_membership=_state(),
            current_direct_candidates=_direct(()),
        )
    source = Path(collider.__file__).read_text(encoding="utf-8")
    assert "import audit_" not in source
    assert "from audit_" not in source
    assert "import diagnose_" not in source
    assert "from diagnose_" not in source


def _small_world_inputs() -> tuple[
    collider.CrossfitHierarchicalColliderConfig,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    features = ("A", "B", "C")
    cells = []
    for index in range(36):
        a = int(index % 2 == 0)
        b = int(index % 3 == 0)
        c = int((a and b) or index % 11 == 0)
        cells.append(
            {
                "cell": f"cell{index:03d}",
                "modality": ASSAY,
                "observed_background_library_size": 20 + index,
                "observed_library_size": 25 + index,
                "A": a,
                "B": b,
                "C": c,
            }
        )
    tables = {ASSAY: pd.DataFrame(cells)}
    embedding = pd.DataFrame(
        [
            {
                "cell": f"cell{index:03d}",
                "modality": ASSAY,
                "integration__dim0": (index - 18) / 18.0,
                "integration__dim1": np.sin(index / 5.0),
                "integration_uncertainty": 0.05,
            }
            for index in range(36)
        ]
    )
    config = collider.CrossfitHierarchicalColliderConfig(
        n_folds=2,
        microstate_resolutions=(2,),
        null_permutations=3,
        required_assays=1,
        required_features_per_assay=3,
        required_evidence_views=2,
        minimum_base_view_count=1,
    )
    votes = _votes(features, 2)
    base = _base_views(features, 2, ("A", "C"))
    residual = {
        f"v{view}": pd.DataFrame(
            [
                {"node1": left, "node2": right, "residual_r": value}
                for (left, right), value in {
                    ("A", "B"): 0.05,
                    ("A", "C"): 0.30,
                    ("B", "C"): 0.25,
                }.items()
            ]
        )
        for view in range(2)
    }
    feature_models = {
        f"v{view}": pd.DataFrame(
            [
                {
                    "feature": feature,
                    "oof_partial_state_fit": 0.02,
                    "loading_direction_stability": 0.9,
                }
                for feature in features
            ]
        )
        for view in range(2)
    }
    loading = {
        f"v{view}": pd.DataFrame(
            [
                {
                    "node1": left,
                    "node2": right,
                    "loading_empirical_null_p": 0.10,
                    "absolute_loading_cosine": 0.8,
                }
                for left, right in itertools.combinations(features, 2)
            ]
        )
        for view in range(2)
    }
    state = _state(features, selected=features)
    return config, tables, embedding, votes, base, residual, feature_models, loading, state


def test_world_evidence_is_deterministic_and_reused_across_candidate_resolutions() -> None:
    (
        config,
        tables,
        embedding,
        votes,
        base,
        residual,
        feature_models,
        loading,
        state,
    ) = _small_world_inputs()
    evidence = collider.fit_crossfit_collider_world_evidence(
        assay_cell_tables=tables,
        assay_local_embedding=embedding,
        randomization_key=38001,
        config=config,
    )
    shuffled = collider.fit_crossfit_collider_world_evidence(
        assay_cell_tables={ASSAY: tables[ASSAY].sample(frac=1, random_state=9)},
        assay_local_embedding=embedding.sample(frac=1, random_state=7),
        randomization_key=38001,
        config=config,
    )
    assert evidence.provenance["generative_candidates_sha256"] == shuffled.provenance[
        "generative_candidates_sha256"
    ]
    assert len(evidence.generative_candidates) == 3
    empty_direct = _direct(())
    empty_unresolved = pd.DataFrame(
        columns=["source_assay", "node1", "node2", "was_base_selected"]
    )
    first = collider.resolve_crossfit_hierarchical_colliders_v4(
        world_evidence=evidence,
        beta_vote_evidence=votes,
        base_edge_views=base,
        residual_views=residual,
        feature_model_views=feature_models,
        loading_pair_views=loading,
        state_membership=state,
        current_direct_candidates=empty_direct,
        current_demoted_or_unresolved=empty_unresolved,
        config=config,
    )
    second = collider.resolve_crossfit_hierarchical_colliders_v4(
        world_evidence=evidence,
        beta_vote_evidence=votes,
        base_edge_views=base,
        residual_views=residual,
        feature_model_views=feature_models,
        loading_pair_views=loading,
        state_membership=state,
        current_direct_candidates=_direct(("A", "B"), center="C"),
        current_demoted_or_unresolved=empty_unresolved,
        config=config,
    )
    first_again = collider.resolve_crossfit_hierarchical_colliders_v4(
        world_evidence=evidence,
        beta_vote_evidence=votes.sample(frac=1, random_state=13),
        base_edge_views={key: value.sample(frac=1, random_state=3) for key, value in base.items()},
        residual_views={key: value.sample(frac=1, random_state=5) for key, value in residual.items()},
        feature_model_views={key: value.sample(frac=1, random_state=7) for key, value in feature_models.items()},
        loading_pair_views={key: value.sample(frac=1, random_state=11) for key, value in loading.items()},
        state_membership=state.sample(frac=1, random_state=17),
        current_direct_candidates=empty_direct,
        current_demoted_or_unresolved=empty_unresolved,
        config=config,
    )
    expected_hash = evidence.provenance["generative_candidates_sha256"]
    assert first.provenance["world_evidence_sha256"] == expected_hash
    assert second.provenance["world_evidence_sha256"] == expected_hash
    assert first.provenance["world_evidence_reused_without_refit"]
    assert second.provenance["world_evidence_reused_without_refit"]
    assert first.provenance["output_sha256"] == first_again.provenance["output_sha256"]


def test_config_and_empty_constant_assay_abstain_deterministically() -> None:
    config, tables, embedding, *_ = _small_world_inputs()
    constant = tables[ASSAY].copy()
    constant[["A", "B", "C"]] = 0
    evidence = collider.fit_crossfit_collider_world_evidence(
        assay_cell_tables={ASSAY: constant},
        assay_local_embedding=embedding,
        randomization_key=38001,
        config=config,
    )
    assert set(evidence.generative_candidates["crossfit_status"]) == {
        "insufficient_detection_for_crossfit"
    }
    assert evidence.generative_candidates[
        "crossfit_joint_gain_over_baseline"
    ].isna().all()


def test_schema_validation_rejects_missing_assay_and_duplicate_beta_vote() -> None:
    config, tables, embedding, votes, base, residual, feature_models, loading, state = (
        _small_world_inputs()
    )
    with pytest.raises(ValueError, match="Expected 1 assay"):
        collider.fit_crossfit_collider_world_evidence(
            assay_cell_tables={},
            assay_local_embedding=embedding,
            randomization_key=38001,
            config=config,
        )
    evidence = collider.fit_crossfit_collider_world_evidence(
        assay_cell_tables=tables,
        assay_local_embedding=embedding,
        randomization_key=38001,
        config=config,
    )
    duplicate_votes = pd.concat([votes, votes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate assay/pair/vote"):
        collider.resolve_crossfit_hierarchical_colliders_v4(
            world_evidence=evidence,
            beta_vote_evidence=duplicate_votes,
            base_edge_views=base,
            residual_views=residual,
            feature_model_views=feature_models,
            loading_pair_views=loading,
            state_membership=state,
            current_direct_candidates=_direct(()),
            current_demoted_or_unresolved=pd.DataFrame(),
            config=config,
        )
