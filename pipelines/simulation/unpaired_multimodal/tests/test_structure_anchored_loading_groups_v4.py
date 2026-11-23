from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import structure_anchored_loading_groups_v4 as anchored  # noqa: E402


def _direct(
    two_edges: bool = True,
    observed_refinement_action: str = "",
) -> pd.DataFrame:
    rows = [
        {
            "node1": "P",
            "endpoint1": "TAIL",
            "node2": "C",
            "endpoint2": "ARROW",
            "edge_type": "within_assay_direct_candidate",
            "collider_motif_center": "C",
            "component_id": "component_a",
            "observed_refinement_action": observed_refinement_action,
        }
    ]
    if two_edges:
        rows.append(
            {
                "node1": "Q",
                "endpoint1": "TAIL",
                "node2": "C",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "collider_motif_center": "C",
                "component_id": "component_a",
                "observed_refinement_action": observed_refinement_action,
            }
        )
    return pd.DataFrame(rows)


def _candidate(
    node1: str,
    node2: str,
    cosine: float,
    fit: float = 0.02,
    stability: float = 0.95,
) -> dict[str, object]:
    return {
        "pair_key": "|||".join(sorted([node1, node2])),
        "node1": node1,
        "node2": node2,
        "modality": "chrom_A",
        "n_views": 3,
        "median_absolute_loading_cosine": cosine,
        "lower_absolute_loading_cosine": cosine - 0.01,
        "median_minimum_partial_state_fit": fit,
        "median_minimum_loading_stability": stability,
        "eligible_view_support": 1.0 if fit > 0 else 0.0,
    }


def _candidates(qs_cosine: float = 0.75) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _candidate("P", "S", 0.95),
            _candidate("Q", "S", qs_cosine),
            _candidate("P", "N", 0.98, fit=0.0, stability=0.4),
            _candidate("Q", "N", 0.80, fit=0.0, stability=0.4),
            _candidate("P", "C", 0.99),
        ]
    )


def _null_views(candidates: pd.DataFrame, n_views: int = 3) -> pd.DataFrame:
    rows = []
    for view_index in range(n_views):
        view = f"v{view_index}"
        for candidate in candidates.itertuples(index=False):
            rows.append(
                {
                    "pair_key": candidate.pair_key,
                    "absolute_loading_cosine": candidate.median_absolute_loading_cosine,
                    "loading_empirical_null_p": 0.10,
                    "view_id": view,
                }
            )
    return pd.DataFrame(rows)


def _six_view_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    result["n_views"] = 6
    return result


def _state(selected: tuple[str, ...] = ("S",)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "modality": "chrom_A",
                "selection_status": "selected" if feature in selected else "unresolved",
                "feature_global_maxT_confirmation_gate": feature in selected,
                "assay_global_maxT_admission_gate": True,
                "conditional_wald_view_support": 1.0 if feature in selected else 0.0,
            }
            for feature in ("P", "Q", "C", "S", "N")
        ]
    )


def test_unique_parent_sibling_is_selected_without_layer_conflation() -> None:
    candidates = _candidates()
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates), _state()
    )
    assert len(result.typed_groups) == 1
    group = result.typed_groups.iloc[0]
    assert set(group["members"].split(";")) == {"P", "S"}
    assert group["collider_parent_anchor"] == "P"
    assert group["collider_center_excluded"] == "C"
    assert not group["direct_causal_claimed"]
    assert not group["cross_assay_membership_claimed"]
    assert result.assay_decisions.iloc[0]["selection_status"] == "selected"
    assert result.pair_evidence["selected"].sum() == 1
    assert group["selection_stage"] == "formal_baseline"
    assert not group["empirical_null_rescue_used"]


def test_near_tied_parent_alignments_force_abstention() -> None:
    candidates = _candidates(qs_cosine=0.93)
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates), _state()
    )
    assert result.typed_groups.empty
    decision = result.assay_decisions.iloc[0]
    assert decision["selection_status"] == "unresolved"
    assert decision["abstention_reason"] == "state_sibling_failed_loading_geometry"


def test_incomplete_direct_motif_forces_abstention() -> None:
    candidates = _candidates()
    result = anchored.estimate_structure_anchored_groups(
        _direct(two_edges=False), candidates, _null_views(candidates), _state()
    )
    assert result.typed_groups.empty
    assert result.assay_decisions.iloc[0]["abstention_reason"] == (
        "direct_motif_not_exactly_two_edges"
    )


def test_nonmember_sibling_is_rejected_even_with_strong_loading() -> None:
    candidates = _candidates()
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates), _state(selected=())
    )
    assert result.typed_groups.empty
    assert result.assay_decisions.iloc[0]["abstention_reason"] == (
        "no_formal_state_selected_noncollider_sibling"
    )


def test_row_and_view_order_do_not_change_selection() -> None:
    candidates = _candidates()
    nulls = _null_views(candidates)
    baseline = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, _state()
    )
    shuffled = anchored.estimate_structure_anchored_groups(
        _direct().sample(frac=1.0, random_state=1).reset_index(drop=True),
        candidates.sample(frac=1.0, random_state=2).reset_index(drop=True),
        nulls.sample(frac=1.0, random_state=3).reset_index(drop=True),
        _state().sample(frac=1.0, random_state=4).reset_index(drop=True),
    )
    assert baseline.typed_groups.to_dict("records") == shuffled.typed_groups.to_dict(
        "records"
    )
    assert baseline.assay_decisions.to_dict("records") == (
        shuffled.assay_decisions.to_dict("records")
    )


def test_low_cosine_candidate_is_rescued_only_by_six_view_unanimity() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    nulls = _null_views(candidates, n_views=6)
    nulls.loc[
        nulls["pair_key"].eq("P|||S"), "loading_empirical_null_p"
    ] = 0.125
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, _state()
    )

    assert len(result.typed_groups) == 1
    group = result.typed_groups.iloc[0]
    assert set(group["members"].split(";")) == {"P", "S"}
    assert group["selection_stage"] == "empirical_null_unanimity_rescue"
    assert group["empirical_null_rescue_used"]
    assert group["rescue_state_tier"] == "legacy_formal_state"
    assert group["legacy_empirical_null_rescue_used"]
    assert not group["tiered_empirical_null_rescue_used"]
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert not evidence["passes_cosine_gate"]
    assert evidence["passes_rescue_required_view_count_gate"]
    assert evidence["passes_rescue_parent_win_gate"]
    assert evidence["passes_rescue_empirical_null_gate"]
    assert evidence["passes_all_rescue_gates"]
    assert evidence["selected"]
    assert result.provenance["schema_version"] == (
        "structure_anchored_loading_groups_v4.v5"
    )
    rule = result.provenance["selection_rule"]
    assert rule["baseline_selections_preserved_before_rescue"]
    assert rule["rescue_only_after_baseline_abstention"]
    assert rule["legacy_rescue_required_view_count"] == 6
    assert rule["legacy_and_tiered_rescues_have_absolute_cosine_cutoff"] is False
    assert rule["current_production_selections_preserved_before_tiered_rescue"]
    assert rule["tiered_empirical_null_p_le_0_50_means_not_null_dominated"]
    assert rule["six_views_are_not_claimed_as_independent_replicates"]


def test_baseline_selection_is_not_replaced_by_higher_scoring_rescue() -> None:
    candidates = _six_view_candidates(_candidates())
    candidates.loc[candidates["pair_key"].eq("N|||P"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.99, 0.84]
    candidates.loc[candidates["pair_key"].eq("N|||Q"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.30, 0.29]
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates, n_views=6), _state(("S", "N"))
    )

    group = result.typed_groups.iloc[0]
    assert set(group["members"].split(";")) == {"P", "S"}
    assert group["selection_stage"] == "formal_baseline"
    tempting_rescue = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("N|||P")
    ].iloc[0]
    assert tempting_rescue["passes_all_rescue_gates"]
    assert not tempting_rescue["selected"]
    decision = result.assay_decisions.iloc[0]
    assert decision["n_baseline_eligible_candidates"] == 1
    assert decision["n_rescue_eligible_candidates"] == 0


def test_multiple_rescue_candidates_force_abstention() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.30))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    candidates.loc[candidates["pair_key"].eq("N|||P"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.83, 0.82]
    candidates.loc[candidates["pair_key"].eq("N|||Q"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.20, 0.19]
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates, n_views=6), _state(("S", "N"))
    )

    assert result.typed_groups.empty
    decision = result.assay_decisions.iloc[0]
    assert decision["abstention_reason"] == "empirical_null_rescue_ambiguous"
    assert decision["n_rescue_eligible_candidates"] == 2
    assert result.pair_evidence["selected"].sum() == 0


def test_assay_local_state_tier_can_rescue_feature_global_abstention() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    state = _state(selected=())
    state.loc[state["feature"].eq("S"), "conditional_wald_view_support"] = 1.0
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates, n_views=6), state
    )

    assert len(result.typed_groups) == 1
    group = result.typed_groups.iloc[0]
    assert set(group["members"].split(";")) == {"P", "S"}
    assert group["rescue_state_tier"] == "assay_local_state"
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert not evidence["passes_state_membership_gate"]
    assert evidence["passes_tiered_rescue_assay_local_state_gate"]
    assert evidence["passes_tiered_rescue_assay_local_state_all_gates"]
    assert not evidence["passes_tiered_rescue_strict_state_all_gates"]


def test_same_center_reselection_blocks_only_relaxed_assay_local_rescue() -> None:
    action = "reselect_overcomplete_same_center_star"
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    state = _state(selected=())
    state.loc[state["feature"].eq("S"), "conditional_wald_view_support"] = 1.0

    result = anchored.estimate_structure_anchored_groups(
        _direct(observed_refinement_action=action),
        candidates,
        _null_views(candidates, n_views=6),
        state,
    )

    assert result.typed_groups.empty
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert evidence["active_observed_refinement_action"] == action
    assert evidence["passes_tiered_rescue_assay_local_state_gate"]
    assert not evidence["passes_tiered_rescue_assay_local_action_gate"]
    assert not evidence["passes_tiered_rescue_assay_local_state_all_gates"]
    assert evidence["blocked_relaxed_tier_by_observed_refinement_action"]
    decision = result.assay_decisions.iloc[0]
    assert decision["active_observed_refinement_action"] == action
    assert decision["abstention_reason"] == (
        "tiered_assay_local_rescue_blocked_by_observed_refinement_action"
    )
    assert result.provenance["counts"][
        "n_relaxed_tier_candidates_blocked_by_observed_refinement_action"
    ] == 1


def test_same_center_reselection_guard_preserves_stronger_stages_and_other_actions() -> None:
    blocked_action = "reselect_overcomplete_same_center_star"

    baseline_candidates = _candidates()
    baseline = anchored.estimate_structure_anchored_groups(
        _direct(observed_refinement_action=blocked_action),
        baseline_candidates,
        _null_views(baseline_candidates),
        _state(),
    )
    assert baseline.typed_groups.iloc[0]["selection_stage"] == "formal_baseline"

    rescue_candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    rescue_candidates.loc[rescue_candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    legacy_nulls = _null_views(rescue_candidates, n_views=6)
    legacy_nulls.loc[
        legacy_nulls["pair_key"].eq("P|||S"), "loading_empirical_null_p"
    ] = 0.125
    legacy = anchored.estimate_structure_anchored_groups(
        _direct(observed_refinement_action=blocked_action),
        rescue_candidates,
        legacy_nulls,
        _state(),
    )
    assert legacy.typed_groups.iloc[0]["selection_stage"] == (
        "empirical_null_unanimity_rescue"
    )

    strict_nulls = _null_views(rescue_candidates, n_views=6)
    strict_nulls["loading_empirical_null_p"] = 0.50
    strict = anchored.estimate_structure_anchored_groups(
        _direct(observed_refinement_action=blocked_action),
        rescue_candidates,
        strict_nulls,
        _state(),
    )
    assert strict.typed_groups.iloc[0]["selection_stage"] == (
        "tiered_empirical_null_non_domination_rescue"
    )
    assert strict.typed_groups.iloc[0]["rescue_state_tier"] == (
        "formal_cross_assay_state"
    )

    relaxed_state = _state(selected=())
    relaxed_state.loc[
        relaxed_state["feature"].eq("S"), "conditional_wald_view_support"
    ] = 1.0
    other_action = anchored.estimate_structure_anchored_groups(
        _direct(observed_refinement_action="resolve_competing_multicenter_path"),
        rescue_candidates,
        strict_nulls,
        relaxed_state,
    )
    assert other_action.typed_groups.iloc[0]["rescue_state_tier"] == (
        "assay_local_state"
    )


def test_action_bound_active_motif_requires_matching_arm_provenance() -> None:
    direct = _direct(
        observed_refinement_action="reselect_overcomplete_same_center_star"
    )
    direct.loc[1, "observed_refinement_action"] = ""
    candidates = _candidates()
    with pytest.raises(ValueError, match="inconsistent observed refinement action"):
        anchored.estimate_structure_anchored_groups(
            direct, candidates, _null_views(candidates), _state()
        )


def test_formal_state_rescue_tier_precedes_relaxed_state_tier() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.30))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    candidates.loc[candidates["pair_key"].eq("N|||P"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.83, 0.82]
    candidates.loc[candidates["pair_key"].eq("N|||Q"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.20, 0.19]
    state = _state(("S",))
    state.loc[state["feature"].eq("N"), "conditional_wald_view_support"] = 1.0
    nulls = _null_views(candidates, n_views=6)
    nulls["loading_empirical_null_p"] = 0.50
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, state
    )

    assert len(result.typed_groups) == 1
    group = result.typed_groups.iloc[0]
    assert set(group["members"].split(";")) == {"P", "S"}
    assert group["rescue_state_tier"] == "formal_cross_assay_state"
    assert group["selection_stage"] == (
        "tiered_empirical_null_non_domination_rescue"
    )


def test_strict_tiered_rescue_keeps_formal_four_of_six_wald_gate() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    state = _state(("S",))
    state.loc[
        state["feature"].eq("S"), "conditional_wald_view_support"
    ] = 2.0 / 3.0
    nulls = _null_views(candidates, n_views=6)
    nulls["loading_empirical_null_p"] = 0.50
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, state
    )

    group = result.typed_groups.iloc[0]
    assert group["selection_stage"] == (
        "tiered_empirical_null_non_domination_rescue"
    )
    assert group["rescue_state_tier"] == "formal_cross_assay_state"
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert evidence["passes_state_membership_gate"]
    assert not evidence["passes_tiered_rescue_conditional_state_gate"]
    assert evidence["passes_tiered_rescue_strict_state_all_gates"]


def test_strict_tier_ambiguity_never_falls_through_to_unique_relaxed() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.30))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    candidates.loc[candidates["pair_key"].eq("N|||P"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.83, 0.82]
    candidates.loc[candidates["pair_key"].eq("N|||Q"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.20, 0.19]
    extra = pd.DataFrame(
        [_candidate("P", "R", 0.82), _candidate("Q", "R", 0.20)]
    )
    candidates = pd.concat(
        [candidates, _six_view_candidates(extra)], ignore_index=True
    )
    state = _state(("S", "N"))
    state = pd.concat(
        [
            state,
            pd.DataFrame(
                [
                    {
                        "feature": "R",
                        "modality": "chrom_A",
                        "selection_status": "unresolved",
                        "feature_global_maxT_confirmation_gate": False,
                        "assay_global_maxT_admission_gate": True,
                        "conditional_wald_view_support": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    nulls = _null_views(candidates, n_views=6)
    nulls["loading_empirical_null_p"] = 0.50
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, state
    )

    assert result.typed_groups.empty
    decision = result.assay_decisions.iloc[0]
    assert decision["abstention_reason"] == (
        "tiered_empirical_null_rescue_ambiguous"
    )
    assert decision["n_tiered_rescue_eligible_candidates"] == 2
    relaxed = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||R")
    ].iloc[0]
    assert relaxed["passes_tiered_rescue_assay_local_state_all_gates"]
    assert not relaxed["selected"]


def test_legacy_p125_rescue_is_preserved_before_tiered_p50_pool() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.30))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    candidates.loc[candidates["pair_key"].eq("N|||P"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.83, 0.82]
    candidates.loc[candidates["pair_key"].eq("N|||Q"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.20, 0.19]
    nulls = _null_views(candidates, n_views=6)
    nulls.loc[nulls["pair_key"].eq("N|||P"), "loading_empirical_null_p"] = 0.50
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, _state(("S", "N"))
    )

    assert len(result.typed_groups) == 1
    group = result.typed_groups.iloc[0]
    assert set(group["members"].split(";")) == {"P", "S"}
    assert group["selection_stage"] == "empirical_null_unanimity_rescue"
    assert group["legacy_empirical_null_rescue_used"]
    assert not group["tiered_empirical_null_rescue_used"]
    tempting = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("N|||P")
    ].iloc[0]
    assert tempting["passes_tiered_rescue_strict_state_all_gates"]
    assert not tempting["selected"]
    assert result.assay_decisions.iloc[0][
        "n_tiered_rescue_eligible_candidates"
    ] == 0


def test_relaxed_state_rescue_requires_six_of_six_conditional_support() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    state = _state(selected=())
    state.loc[
        state["feature"].eq("S"), "conditional_wald_view_support"
    ] = 5.0 / 6.0
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates, n_views=6), state
    )

    assert result.typed_groups.empty
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert not evidence["passes_tiered_rescue_conditional_state_gate"]


def test_rescue_rejects_one_nonconfirming_empirical_null_view() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    nulls = _null_views(candidates, n_views=6)
    nulls.loc[
        nulls["pair_key"].eq("P|||S") & nulls["view_id"].eq("v5"),
        "loading_empirical_null_p",
    ] = 0.75
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, _state()
    )

    assert result.typed_groups.empty
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert evidence["rescue_empirical_null_view_support"] == pytest.approx(5 / 6)
    assert not evidence["passes_rescue_empirical_null_gate"]


def test_rescue_rejects_one_parent_loss_view() -> None:
    candidates = _six_view_candidates(_candidates(qs_cosine=0.40))
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.80, 0.79]
    nulls = _null_views(candidates, n_views=6)
    nulls.loc[
        nulls["pair_key"].eq("P|||S") & nulls["view_id"].eq("v5"),
        "absolute_loading_cosine",
    ] = 0.30
    nulls.loc[
        nulls["pair_key"].eq("Q|||S") & nulls["view_id"].eq("v5"),
        "absolute_loading_cosine",
    ] = 0.50
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, nulls, _state()
    )

    assert result.typed_groups.empty
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert evidence["parent_win_view_support"] == pytest.approx(5 / 6)
    assert evidence["passes_rescue_relative_parent_margin_gate"]
    assert not evidence["passes_rescue_parent_win_gate"]


def test_rescue_does_not_activate_with_fewer_than_six_views() -> None:
    candidates = _candidates(qs_cosine=0.40)
    candidates.loc[candidates["pair_key"].eq("P|||S"), [
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
    ]] = [0.82, 0.81]
    result = anchored.estimate_structure_anchored_groups(
        _direct(), candidates, _null_views(candidates), _state()
    )

    assert result.typed_groups.empty
    evidence = result.pair_evidence.loc[
        result.pair_evidence["pair_key"].eq("P|||S")
    ].iloc[0]
    assert not evidence["passes_rescue_required_view_count_gate"]


def test_rescue_config_validates_frozen_fields() -> None:
    assert asdict(anchored.StructureAnchoredGroupConfig()) == {
        "minimum_lower_absolute_loading_cosine": 0.85,
        "minimum_relative_parent_margin": 0.10,
        "minimum_sibling_state_view_support": 2.0 / 3.0,
        "enable_abstention_only_empirical_null_rescue": True,
        "rescue_empirical_null_p_threshold": 0.125,
        "rescue_minimum_empirical_null_view_support": 1.0,
        "rescue_minimum_parent_win_view_support": 1.0,
        "rescue_required_view_count": 6,
        "rescue_requires_unique_candidate": True,
        "enable_layer_separated_tiered_rescue": True,
        "tiered_rescue_empirical_null_p_threshold": 0.50,
        "tiered_rescue_minimum_empirical_null_view_support": 1.0,
        "tiered_rescue_minimum_parent_win_view_support": 1.0,
        "tiered_rescue_requires_sibling_assay_global_maxT_admission": True,
        "tiered_rescue_minimum_sibling_conditional_wald_view_support": 1.0,
        "tiered_rescue_required_view_count": 6,
        "tiered_rescue_requires_unique_candidate": True,
        "tiered_rescue_assay_local_blocked_observed_refinement_actions": (
            "reselect_overcomplete_same_center_star",
        ),
    }
    with pytest.raises(ValueError, match="positive integer"):
        anchored.StructureAnchoredGroupConfig(rescue_required_view_count=0)
    with pytest.raises(ValueError, match="Boolean"):
        anchored.StructureAnchoredGroupConfig(
            enable_abstention_only_empirical_null_rescue="yes"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        anchored.StructureAnchoredGroupConfig(
            rescue_empirical_null_p_threshold=1.01
        )
    with pytest.raises(ValueError, match="Boolean"):
        anchored.StructureAnchoredGroupConfig(
            tiered_rescue_requires_sibling_assay_global_maxT_admission="yes"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty tuple"):
        anchored.StructureAnchoredGroupConfig(
            tiered_rescue_assay_local_blocked_observed_refinement_actions=()
        )


def test_runner_content_addresses_outputs(tmp_path: Path) -> None:
    candidates = _candidates()
    consensus = tmp_path / "consensus.tsv"
    candidate_path = tmp_path / "candidates.tsv"
    membership_path = tmp_path / "state.tsv"
    _direct().to_csv(consensus, sep="\t", index=False)
    candidates.to_csv(candidate_path, sep="\t", index=False)
    _state().to_csv(membership_path, sep="\t", index=False)
    null_paths = []
    for view in ("v0", "v1", "v2"):
        view_dir = tmp_path / view
        view_dir.mkdir()
        path = view_dir / "pairs.tsv"
        pd.DataFrame(
            {
                "pair_key": candidates["pair_key"],
                "absolute_loading_cosine": candidates[
                    "median_absolute_loading_cosine"
                ],
                "loading_empirical_null_p": 0.10,
            }
        ).to_csv(path, sep="\t", index=False)
        null_paths.append(path)
    out = tmp_path / "out"
    manifest = anchored.run_structure_anchored_groups(
        consensus, candidate_path, null_paths, membership_path, out
    )
    assert manifest["candidate_frozen"]
    assert not manifest["truth_or_scorer_consumed_by_algorithm"]
    assert manifest["schema_version"] == (
        "structure_anchored_loading_groups_v4.frozen_candidate.v5"
    )
    assert manifest["config"]["rescue_required_view_count"] == 6
    assert manifest["config"]["rescue_empirical_null_p_threshold"] == 0.125
    assert manifest["config"]["tiered_rescue_empirical_null_p_threshold"] == 0.50
    assert (out / "frozen_structure_anchor_manifest.json").exists()
