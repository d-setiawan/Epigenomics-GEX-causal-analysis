from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from observed_collider_structure import (  # noqa: E402
    ObservedColliderStructureConfig,
    _abstain_overlapping_refinement_arms,
    apply_observed_collider_refinements,
    conditional_outer_correlation,
    select_observed_collider_refinements,
)
from diagnose_observed_collider_structure_train_v4 import (  # noqa: E402
    _validated_train_seeds,
)
from structure_aware_collider_motif import select_unique_collider_motifs  # noqa: E402


def _components(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_assay": "assay_a", "component_id": component, "status": status}
            for component, status in rows
        ]
    )


def _wedges(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": "assay_a",
                "component_id": component,
                "center": center,
                "outer1": outer1,
                "outer2": outer2,
                "wedge_score": score,
            }
            for component, center, outer1, outer2, score in rows
        ],
        columns=[
            "source_assay",
            "component_id",
            "center",
            "outer1",
            "outer2",
            "wedge_score",
        ],
    )


def _unresolved(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": "assay_a",
                "component_id": component,
                "node1": left,
                "node2": right,
            }
            for component, left, right in rows
        ],
        columns=["source_assay", "component_id", "node1", "node2"],
    )


def _votes(
    pairs: dict[tuple[str, str], tuple[int, float, float]],
) -> pd.DataFrame:
    rows = []
    for (left, right), (selected_count, p_value, log_odds) in pairs.items():
        for vote in range(6):
            rows.append(
                {
                    "source_assay": "assay_a",
                    "node1": left,
                    "node2": right,
                    "vote": vote,
                    "selected": vote < selected_count,
                    "empirical_p_positive": p_value,
                    "pooled_log_odds": log_odds,
                }
            )
    return pd.DataFrame(rows)


def _base_views(edges_by_count: dict[tuple[str, str], int]) -> dict[str, pd.DataFrame]:
    views = {}
    for view in range(6):
        views[f"view{view}"] = pd.DataFrame(
            [
                {
                    "node1": left,
                    "node2": right,
                    "edge_type": "within_assay_direct_candidate",
                }
                for (left, right), count in edges_by_count.items()
                if view < count
            ],
            columns=["node1", "node2", "edge_type"],
        )
    return views


def _base_frame(edges: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_assay": "assay_a",
                "node1": left,
                "node2": right,
                "selected": True,
            }
            for left, right in edges
        ],
        columns=["source_assay", "node1", "node2", "selected"],
    )


def _pairs(frame: pd.DataFrame) -> set[frozenset[str]]:
    return {
        frozenset((str(row.node1), str(row.node2))) for row in frame.itertuples()
    }


def _residual_views(
    per_view: list[dict[tuple[str, str], float]],
) -> dict[str, pd.DataFrame]:
    return {
        f"view{view}": pd.DataFrame(
            [
                {"node1": left, "node2": right, "residual_r": value}
                for (left, right), value in pairs.items()
            ]
        )
        for view, pairs in enumerate(per_view)
    }


def _feature_views(
    scores: dict[str, tuple[float, float]],
) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        [
            {
                "feature": feature,
                "modality": "assay_a",
                "oof_partial_state_fit": partial_fit,
                "loading_direction_stability": stability,
            }
            for feature, (partial_fit, stability) in scores.items()
        ]
    )
    return {f"view{view}": frame.copy() for view in range(6)}


def _loading_pair_views(
    p_values: dict[tuple[str, str], float],
) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        [
            {
                "node1": left,
                "node2": right,
                "loading_empirical_null_p": p_value,
                "absolute_loading_cosine": 1.0 - p_value / 10.0,
            }
            for (left, right), p_value in p_values.items()
        ]
    )
    return {f"view{view}": frame.copy() for view in range(6)}


def _state_membership(selected: set[str], features: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "selection_status": (
                    "selected" if feature in selected else "not_selected"
                ),
            }
            for feature in features
        ]
    )


def test_conditional_outer_correlation_matches_closed_form() -> None:
    observed = conditional_outer_correlation(0.10, 0.40, 0.30)
    expected = (0.10 - 0.40 * 0.30) / ((1 - 0.40**2) * (1 - 0.30**2)) ** 0.5
    assert observed == pytest.approx(expected)


def test_overlapping_refinement_arms_abstain_all_incident_components() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": "assay_a",
                "component_id": "component0",
                "action": "complete_dual_supported_unshielded_anchor",
                "center": "c",
                "outer1": "p",
                "outer2": "q",
            },
            {
                "source_assay": "assay_a",
                "component_id": "component1",
                "action": "complete_dual_supported_unshielded_anchor",
                "center": "q",
                "outer1": "r",
                "outer2": "c",
            },
            {
                "source_assay": "assay_a",
                "component_id": "component2",
                "action": "complete_dual_supported_unshielded_anchor",
                "center": "x",
                "outer1": "y",
                "outer2": "z",
            },
        ]
    )

    retained, abstentions = _abstain_overlapping_refinement_arms(decisions)

    assert retained["component_id"].tolist() == ["component2"]
    assert {row["component_id"] for row in abstentions} == {
        "component0",
        "component1",
    }
    assert {
        tuple(row["conflicting_selected_arms"]) for row in abstentions
    } == {("assay_a:c|q",)}
    assert {row["reason"] for row in abstentions} == {
        "overlapping_refinement_arm_requires_global_abstention"
    }


def test_production_collider_repair_config_is_fully_frozen() -> None:
    assert asdict(ObservedColliderStructureConfig()) == {
        "minimum_base_view_count": 4,
        "minimum_winner_view_count": 4,
        "same_center_minimum_pairwise_wedge_margin_exclusive": 0.0,
        "minimum_rescue_vote_fraction": 5.0 / 6.0,
        "maximum_rescue_mean_empirical_p_positive": 0.020,
        "minimum_rescue_mean_pooled_log_odds": 1.00,
        "minimum_completion_mean_pooled_log_odds": 0.0,
        "minimum_completion_center_contrast_view_count": 4,
        "minimum_unique_dual_base_winner_view_fraction": 0.50,
        "maximum_rescue_only_loading_empirical_null_p": 0.125,
        "required_rescue_only_base_graph_view_count": 6,
        "required_rescue_only_loading_view_count": 6,
        "required_rescue_only_beta_vote_count": 6,
        "require_unique_rescue_only_candidate": True,
        "correlation_clip": 1.0 - 1e-9,
    }


def test_stable_same_center_separation_abstains_when_pairwise_rank_conflicts() -> None:
    components = _components([("star", "accepted_unique_wedge")])
    wedges = _wedges(
        [
            ("star", "c", "p", "s", 2.0),
            ("star", "c", "p", "q", 1.8),
            ("star", "c", "q", "s", 1.7),
        ]
    )
    # In four views p and q become negatively associated only after c is held.
    correlations = []
    for view in range(6):
        correlations.append(
            {
                ("c", "p"): 0.30,
                ("c", "q"): 0.25,
                ("c", "s"): 0.12,
                ("p", "q"): 0.00 if view < 4 else 0.12,
                ("p", "s"): 0.08,
                ("q", "s"): 0.08,
            }
        )
    result = select_observed_collider_refinements(
        components,
        wedges,
        _unresolved([]),
        _votes(
            {
                ("c", "p"): (6, 0.001, 2.1),
                ("c", "q"): (6, 0.001, 1.8),
                ("c", "s"): (6, 0.001, 2.0),
                ("p", "q"): (0, 0.8, 0.1),
                ("p", "s"): (0, 0.8, 0.1),
                ("q", "s"): (0, 0.8, 0.1),
            }
        ),
        _base_views({("c", "p"): 6, ("c", "q"): 6, ("c", "s"): 6}),
        _residual_views(correlations),
        _feature_views({node: (0.10, 1.0) for node in "cpqs"}),
    )
    assert result.decisions.empty
    stable = result.wedge_evidence.loc[
        result.wedge_evidence["stable_view_winner"].astype(bool)
    ].iloc[0]
    assert {stable["outer1"], stable["outer2"]} == {"p", "q"}
    assert stable["winner_view_count"] == 4
    assert stable["stable_view_winner_pairwise_margin"] == pytest.approx(-0.2)
    assert not bool(stable["stable_view_winner_pairwise_compatible"])
    assert bool(stable["stable_view_winner_pairwise_conflict_abstained"])
    assert result.metadata["n_same_center_pairwise_conflict_abstentions"] == 1


def test_stable_same_center_separation_reselects_pairwise_compatible_winner() -> None:
    components = _components([("star", "accepted_unique_wedge")])
    wedges = _wedges(
        [
            ("star", "c", "p", "q", 2.0),
            ("star", "c", "p", "s", 1.8),
            ("star", "c", "q", "s", 1.7),
        ]
    )
    correlations = [
        {
            ("c", "p"): 0.30,
            ("c", "q"): 0.25,
            ("c", "s"): 0.12,
            ("p", "q"): 0.00 if view < 4 else 0.12,
            ("p", "s"): 0.08,
            ("q", "s"): 0.08,
        }
        for view in range(6)
    ]
    result = select_observed_collider_refinements(
        components,
        wedges,
        _unresolved([]),
        _votes(
            {
                ("c", "p"): (6, 0.001, 2.1),
                ("c", "q"): (6, 0.001, 2.0),
                ("c", "s"): (6, 0.001, 1.8),
                ("p", "q"): (0, 0.8, 0.1),
                ("p", "s"): (0, 0.8, 0.1),
                ("q", "s"): (0, 0.8, 0.1),
            }
        ),
        _base_views({("c", "p"): 6, ("c", "q"): 6, ("c", "s"): 6}),
        _residual_views(correlations),
        _feature_views({node: (0.10, 1.0) for node in "cpqs"}),
    )
    decision = result.decisions.iloc[0]
    assert decision["action"] == "reselect_overcomplete_same_center_star"
    assert {decision["outer1"], decision["outer2"]} == {"p", "q"}
    assert decision["winner_view_count"] == 4
    stable = result.wedge_evidence.loc[
        result.wedge_evidence["stable_view_winner"].astype(bool)
    ].iloc[0]
    assert stable["stable_view_winner_pairwise_margin"] == pytest.approx(0.2)
    assert bool(stable["stable_view_winner_pairwise_compatible"])


def test_competing_path_uses_base_support_then_outer_state_support() -> None:
    components = _components([("path", "unresolved_competing_wedges")])
    wedges = _wedges(
        [
            ("path", "c", "p", "q", 1.70),
            ("path", "p", "c", "noise", 1.90),
        ]
    )
    pairs = {
        ("c", "p"): 0.30,
        ("c", "q"): 0.25,
        ("p", "noise"): 0.30,
        ("p", "q"): 0.00,
        ("c", "noise"): 0.00,
        ("q", "noise"): 0.00,
    }
    result = select_observed_collider_refinements(
        components,
        wedges,
        _unresolved([]),
        _votes(
            {
                ("c", "p"): (6, 0.001, 2.0),
                ("c", "q"): (6, 0.001, 1.7),
                ("p", "noise"): (6, 0.001, 1.9),
                ("p", "q"): (0, 0.8, 0.1),
                ("c", "noise"): (0, 0.8, 0.1),
                ("q", "noise"): (0, 0.8, 0.1),
            }
        ),
        _base_views({("c", "p"): 6, ("c", "q"): 6, ("p", "noise"): 0}),
        _residual_views([pairs] * 6),
        _feature_views(
            {
                "c": (0.10, 1.0),
                "p": (0.10, 1.0),
                "q": (0.08, 1.0),
                "noise": (0.0, 0.0),
            }
        ),
    )
    decision = result.decisions.iloc[0]
    assert decision["action"] == "resolve_competing_multicenter_path"
    assert decision["center"] == "c"
    assert {decision["outer1"], decision["outer2"]} == {"p", "q"}
    assert decision["winner_view_count"] == 6


def test_dual_supported_anchor_completes_but_rescue_only_anchor_abstains() -> None:
    components = _components(
        [
            ("dual", "unresolved_no_complete_wedge"),
            ("rescue_only", "unresolved_no_complete_wedge"),
        ]
    )
    votes = _votes(
        {
            ("c", "p"): (6, 0.001, 2.0),
            ("c", "q"): (5, 0.05, 1.3),
            ("p", "q"): (0, 0.8, 0.1),
            ("x", "y"): (6, 0.001, 1.8),
            ("x", "z"): (5, 0.05, 1.2),
            ("y", "z"): (0, 0.8, 0.1),
            ("c", "x"): (0, 0.8, 0.1),
            ("c", "y"): (0, 0.8, 0.1),
            ("c", "z"): (0, 0.8, 0.1),
            ("p", "x"): (0, 0.8, 0.1),
            ("p", "y"): (0, 0.8, 0.1),
            ("p", "z"): (0, 0.8, 0.1),
            ("q", "x"): (0, 0.8, 0.1),
            ("q", "y"): (0, 0.8, 0.1),
            ("q", "z"): (0, 0.8, 0.1),
        }
    )
    empty_pairs = {
        (left, right): 0.0
        for left in "cpqxyz"
        for right in "cpqxyz"
        if left < right
    }
    result = select_observed_collider_refinements(
        components,
        _wedges([]),
        _unresolved([("dual", "c", "p"), ("rescue_only", "x", "y")]),
        votes,
        _base_views({("c", "p"): 6, ("x", "y"): 0}),
        _residual_views([empty_pairs] * 6),
        _feature_views({node: (0.10, 1.0) for node in "cpqxyz"}),
    )
    assert len(result.decisions) == 1
    decision = result.decisions.iloc[0]
    assert decision["action"] == "complete_dual_supported_unshielded_anchor"
    assert decision["center"] == "c"
    assert decision["new_arm_node1"] == "c"
    assert decision["new_arm_node2"] == "q"
    assert decision["winner_view_count"] >= 4


def test_completion_center_is_reoriented_by_assay_beta_centrality() -> None:
    components = _components([("completion", "unresolved_no_complete_wedge")])
    votes = _votes(
        {
            ("c", "q"): (6, 0.001, 2.0),
            ("q", "s"): (6, 0.001, 1.5),
            ("c", "p"): (5, 0.05, 1.4),
            ("c", "s"): (5, 0.05, 1.3),
            ("p", "q"): (0, 0.8, 0.1),
            ("p", "s"): (0, 0.8, 0.1),
        }
    )
    residual_pairs = {
        (left, right): 0.0
        for left in "cpqs"
        for right in "cpqs"
        if left < right
    }
    result = select_observed_collider_refinements(
        components,
        _wedges([]),
        _unresolved([("completion", "c", "q")]),
        votes,
        _base_views({("c", "q"): 6}),
        _residual_views([residual_pairs] * 6),
        _feature_views({node: (0.10, 1.0) for node in "cpqs"}),
    )
    assert len(result.decisions) == 1
    decision = result.decisions.iloc[0]
    assert decision["action"] == "reorient_completion_by_assay_beta_centrality"
    assert decision["center"] == "c"
    assert {decision["outer1"], decision["outer2"]} == {"p", "q"}
    centrality = result.completion_evidence.loc[
        result.completion_evidence["proposal_family"].eq(
            "completion_beta_centrality"
        )
    ]
    selected = centrality.loc[centrality["selected_new_parent"].astype(bool)].iloc[0]
    assert selected["candidate_new_parent"] == "p"
    assert int(selected["contrast_winner_view_count"]) == 6


def test_rejected_unique_dual_base_wedge_accepts_half_view_dominance() -> None:
    components = _components([("rejected", "unresolved_multiple_wedges")])
    wedges = _wedges(
        [
            ("rejected", "c", "p", "q", 1.8),
            ("rejected", "c", "p", "s", 1.7),
        ]
    )
    correlations = []
    for view in range(6):
        correlations.append(
            {
                ("c", "p"): 0.30,
                ("c", "q"): 0.25,
                ("c", "s"): 0.20,
                ("p", "q"): 0.00 if view < 3 else 0.20,
                ("p", "s"): 0.20 if view < 3 else 0.00,
                ("q", "s"): 0.00,
            }
        )
    result = select_observed_collider_refinements(
        components,
        wedges,
        _unresolved([]),
        _votes(
            {
                ("c", "p"): (6, 0.001, 2.0),
                ("c", "q"): (6, 0.001, 1.8),
                ("c", "s"): (6, 0.001, 1.7),
                ("p", "q"): (0, 0.8, 0.1),
                ("p", "s"): (0, 0.8, 0.1),
                ("q", "s"): (0, 0.8, 0.1),
            }
        ),
        _base_views({("c", "p"): 6, ("c", "q"): 6}),
        _residual_views(correlations),
        _feature_views({node: (0.10, 1.0) for node in "cpqs"}),
    )
    assert len(result.decisions) == 1
    decision = result.decisions.iloc[0]
    assert decision["action"] == "accept_rejected_unique_dual_base_half_view_wedge"
    assert decision["center"] == "c"
    assert {decision["outer1"], decision["outer2"]} == {"p", "q"}
    assert int(decision["winner_view_count"]) == 3
    selected = result.wedge_evidence.loc[
        result.wedge_evidence[
            "selected_by_half_view_dual_base_rule"
        ].astype(bool)
    ]
    assert len(selected) == 1


def test_rescue_only_completion_requires_unique_state_loading_candidate() -> None:
    components = _components([("rescue", "unresolved_no_complete_wedge")])
    votes = _votes(
        {
            ("x", "y"): (6, 0.001, 1.8),
            ("x", "z"): (5, 0.05, 1.2),
            ("y", "z"): (0, 0.8, 0.9),
        }
    )
    residual_pairs = {("x", "y"): 0.0, ("x", "z"): 0.0, ("y", "z"): 0.0}
    result = select_observed_collider_refinements(
        components,
        _wedges([]),
        _unresolved([("rescue", "x", "y")]),
        votes,
        _base_views({("x", "y"): 0}),
        _residual_views([residual_pairs] * 6),
        _feature_views({node: (0.10, 1.0) for node in "xyz"}),
        loading_pair_views=_loading_pair_views(
            {("x", "y"): 0.20, ("x", "z"): 0.10, ("y", "z"): 0.20}
        ),
        state_membership=_state_membership({"z"}, "xyz"),
    )
    assert len(result.decisions) == 1
    decision = result.decisions.iloc[0]
    assert (
        decision["action"]
        == "complete_unique_rescue_only_state_loading_contrast"
    )
    assert decision["center"] == "x"
    assert {decision["outer1"], decision["outer2"]} == {"y", "z"}
    evidence = result.completion_evidence.loc[
        result.completion_evidence["proposal_family"].eq(
            "rescue_only_state_loading_contrast"
        )
    ]
    passing = evidence.loc[evidence["passes_all_candidate_gates"].astype(bool)]
    assert len(passing) == 1
    assert bool(passing.iloc[0]["unique_component_candidate"])
    views = result.completion_view_evidence.loc[
        result.completion_view_evidence["proposal_family"].eq(
            "rescue_only_state_loading_contrast"
        )
    ]
    candidate_sizes = views.groupby(
        ["center", "anchor_outer", "candidate_new_parent"], sort=True
    ).size()
    assert set(candidate_sizes) == {12}
    winner_views = views.loc[
        views["center"].eq("x")
        & views["candidate_new_parent"].eq("z")
    ]
    assert int(
        winner_views.loc[
            winner_views["evidence_view_type"].eq("loading"),
            "view_gate_passes",
        ].astype(bool).sum()
    ) == 6
    assert int(
        winner_views.loc[
            winner_views["evidence_view_type"].eq("beta_contrast"),
            "view_gate_passes",
        ].astype(bool).sum()
    ) == 6
    for name, expected in {
        "loading_views_reuse_same_observed_cells": True,
        "loading_views_are_algorithmic_stability_repeats": True,
        "loading_views_are_independent_worlds": False,
        "loading_view_support_not_world_sample_size": True,
        "beta_votes_reuse_same_observed_cells": True,
        "beta_votes_are_algorithmic_stability_repeats": True,
        "beta_votes_are_independent_experiments": False,
        "beta_vote_support_not_world_sample_size": True,
    }.items():
        assert result.metadata[name] is expected


def test_rescue_only_completion_fails_closed_on_incomplete_or_duplicate_views() -> None:
    components = _components([("rescue", "unresolved_no_complete_wedge")])
    votes = _votes(
        {
            ("x", "y"): (6, 0.001, 1.8),
            ("x", "z"): (5, 0.05, 1.2),
            ("y", "z"): (0, 0.8, 0.9),
        }
    )
    base = _base_views({("x", "y"): 0})
    residual_pairs = {
        ("x", "y"): 0.0,
        ("x", "z"): 0.0,
        ("y", "z"): 0.0,
    }
    residual = _residual_views([residual_pairs] * 6)
    features = _feature_views({node: (0.10, 1.0) for node in "xyz"})
    loading = _loading_pair_views(
        {("x", "y"): 0.20, ("x", "z"): 0.10, ("y", "z"): 0.20}
    )
    membership = _state_membership({"z"}, "xyz")
    positional = (
        components,
        _wedges([]),
        _unresolved([("rescue", "x", "y")]),
    )
    with pytest.raises(ValueError, match="exactly 6 loading views"):
        select_observed_collider_refinements(
            *positional,
            votes,
            base,
            {key: value for key, value in residual.items() if key != "view5"},
            {key: value for key, value in features.items() if key != "view5"},
            loading_pair_views={
                key: value for key, value in loading.items() if key != "view5"
            },
            state_membership=membership,
        )
    with pytest.raises(ValueError, match="exactly 6 beta votes"):
        select_observed_collider_refinements(
            *positional,
            votes.loc[votes["vote"].eq(0)].copy(),
            base,
            residual,
            features,
            loading_pair_views=loading,
            state_membership=membership,
        )
    with pytest.raises(ValueError, match="exactly 6 base graph views"):
        select_observed_collider_refinements(
            *positional,
            votes,
            {key: value for key, value in base.items() if key != "view5"},
            residual,
            features,
            loading_pair_views=loading,
            state_membership=membership,
        )
    duplicated_votes = pd.concat([votes, votes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate assay/vote/pair"):
        select_observed_collider_refinements(
            *positional,
            duplicated_votes,
            base,
            residual,
            features,
            loading_pair_views=loading,
            state_membership=membership,
        )


def test_rescue_only_completion_abstains_when_two_candidates_pass() -> None:
    components = _components([("rescue", "unresolved_no_complete_wedge")])
    votes = _votes(
        {
            ("w", "x"): (5, 0.05, 1.1),
            ("w", "y"): (0, 0.8, 0.7),
            ("w", "z"): (0, 0.8, 0.7),
            ("x", "y"): (6, 0.001, 1.8),
            ("x", "z"): (5, 0.05, 1.3),
            ("y", "z"): (0, 0.8, 0.7),
        }
    )
    residual_pairs = {
        (left, right): 0.0
        for left in "wxyz"
        for right in "wxyz"
        if left < right
    }
    result = select_observed_collider_refinements(
        components,
        _wedges([]),
        _unresolved([("rescue", "x", "y")]),
        votes,
        _base_views({("x", "y"): 0}),
        _residual_views([residual_pairs] * 6),
        _feature_views({node: (0.10, 1.0) for node in "wxyz"}),
        loading_pair_views=_loading_pair_views(
            {
                ("w", "x"): 0.10,
                ("w", "y"): 0.20,
                ("w", "z"): 0.20,
                ("x", "y"): 0.20,
                ("x", "z"): 0.10,
                ("y", "z"): 0.20,
            }
        ),
        state_membership=_state_membership({"w", "z"}, "wxyz"),
    )
    assert result.decisions.empty
    evidence = result.completion_evidence.loc[
        result.completion_evidence["proposal_family"].eq(
            "rescue_only_state_loading_contrast"
        )
    ]
    passing = evidence.loc[evidence["passes_all_candidate_gates"].astype(bool)]
    assert len(passing) == 2
    assert not passing["unique_component_candidate"].astype(bool).any()


def test_rejects_privileged_columns() -> None:
    with pytest.raises(ValueError, match="privileged"):
        select_observed_collider_refinements(
            _components([("x", "accepted_unique_wedge")]),
            _wedges([("x", "c", "p", "q", 1.5)]).assign(truth_role="hidden"),
            _unresolved([]),
            _votes(
                {
                    ("c", "p"): (6, 0.001, 2.0),
                    ("c", "q"): (6, 0.001, 1.8),
                    ("p", "q"): (0, 0.8, 0.1),
                }
            ),
            _base_views({("c", "p"): 6, ("c", "q"): 6}),
            _residual_views(
                [
                    {("c", "p"): 0.3, ("c", "q"): 0.3, ("p", "q"): 0.0}
                ]
                * 6
            ),
            _feature_views({node: (0.10, 1.0) for node in "cpq"}),
        )


def test_apply_preserves_pairwise_compatible_same_center_star_and_demotes_excess(
    tmp_path: Path,
) -> None:
    votes = _votes(
        {
            ("c", "p"): (6, 0.001, 2.1),
            ("c", "q"): (6, 0.001, 2.2),
            ("c", "s"): (6, 0.001, 1.8),
            ("p", "q"): (0, 0.8, 0.1),
            ("p", "s"): (0, 0.8, 0.1),
            ("q", "s"): (0, 0.8, 0.1),
        }
    )
    motif = select_unique_collider_motifs(
        _base_frame([("c", "p"), ("c", "q"), ("c", "s")]), votes
    )
    assert _pairs(motif.direct_candidates) == {
        frozenset(("c", "p")),
        frozenset(("c", "q")),
    }
    correlations = [
        {
            ("c", "p"): 0.30,
            ("c", "q"): 0.25,
            ("c", "s"): 0.12,
            ("p", "q"): 0.00 if view < 4 else 0.12,
            ("p", "s"): 0.08,
            ("q", "s"): 0.08,
        }
        for view in range(6)
    ]
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        votes,
        _base_views({("c", "p"): 6, ("c", "q"): 6, ("c", "s"): 6}),
        _residual_views(correlations),
        _feature_views({node: (0.10, 1.0) for node in "cpqs"}),
    )
    applied = apply_observed_collider_refinements(motif, refinement)
    assert _pairs(applied.direct_candidates) == {
        frozenset(("c", "p")),
        frozenset(("c", "q")),
    }
    assert _pairs(applied.demoted_or_unresolved) == {frozenset(("c", "s"))}
    assert set(applied.direct_candidates["collider_center"]) == {"c"}
    assert set(applied.direct_candidates["observed_refinement_action"]) == {
        "reselect_overcomplete_same_center_star"
    }
    assert applied.metadata["schema_version"] == "collider_motif_selector.v4"
    assert applied.metadata[
        "active_direct_observed_refinement_action_field"
    ] == "observed_refinement_action"
    assert applied.metadata["observed_structural_refinement"]["decisions"][0][
        "action"
    ] == "reselect_overcomplete_same_center_star"
    serialized = tmp_path / "active_direct_candidates.tsv"
    applied.direct_candidates.to_csv(serialized, sep="\t", index=False)
    restored = pd.read_csv(serialized, sep="\t", keep_default_na=False)
    assert set(restored["observed_refinement_action"]) == {
        "reselect_overcomplete_same_center_star"
    }
    component = applied.component_diagnostics.iloc[0]
    assert component["nodes"] == "c;p;q;s"
    assert int(component["n_nodes"]) == 4
    retained_component_edges = pd.concat(
        [applied.direct_candidates, applied.demoted_or_unresolved],
        ignore_index=True,
    )
    assert set(component["nodes"].split(";")) == {
        node
        for row in retained_component_edges.itertuples()
        for node in (row.node1, row.node2)
    }
    for row in applied.direct_candidates.itertuples():
        center_endpoint = row.endpoint1 if row.node1 == "c" else row.endpoint2
        assert center_endpoint == "ARROW"


def test_apply_resolves_competing_path_and_keeps_excess_edge_unresolved() -> None:
    votes = _votes(
        {
            ("c", "p"): (6, 0.001, 2.0),
            ("c", "q"): (6, 0.001, 1.70),
            ("p", "noise"): (6, 0.001, 1.75),
            ("p", "q"): (0, 0.8, 0.1),
            ("c", "noise"): (0, 0.8, 0.1),
            ("q", "noise"): (0, 0.8, 0.1),
        }
    )
    motif = select_unique_collider_motifs(
        _base_frame([("c", "p"), ("c", "q")]), votes
    )
    assert motif.direct_candidates.empty
    assert set(motif.component_diagnostics["status"]) == {
        "unresolved_competing_wedges"
    }
    correlations = {
        ("c", "p"): 0.30,
        ("c", "q"): 0.25,
        ("p", "noise"): 0.30,
        ("p", "q"): 0.00,
        ("c", "noise"): 0.00,
        ("q", "noise"): 0.00,
    }
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        votes,
        _base_views({("c", "p"): 6, ("c", "q"): 6, ("p", "noise"): 0}),
        _residual_views([correlations] * 6),
        _feature_views(
            {
                "c": (0.10, 1.0),
                "p": (0.10, 1.0),
                "q": (0.08, 1.0),
                "noise": (0.0, 0.0),
            }
        ),
    )
    applied = apply_observed_collider_refinements(motif, refinement)
    assert _pairs(applied.direct_candidates) == {
        frozenset(("c", "p")),
        frozenset(("c", "q")),
    }
    assert _pairs(applied.demoted_or_unresolved) == {
        frozenset(("p", "noise"))
    }
    assert set(applied.direct_candidates["observed_refinement_action"]) == {
        "resolve_competing_multicenter_path"
    }
    assert bool(applied.component_diagnostics.iloc[0]["accepted"])


def test_apply_completes_dual_supported_anchor_and_adds_wedge_diagnostic() -> None:
    votes = _votes(
        {
            ("c", "p"): (6, 0.001, 2.0),
            ("c", "q"): (5, 0.05, 1.3),
            ("p", "q"): (0, 0.8, 0.1),
        }
    )
    motif = select_unique_collider_motifs(_base_frame([("c", "p")]), votes)
    assert motif.direct_candidates.empty
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        votes,
        _base_views({("c", "p"): 6}),
        _residual_views(
            [
                {("c", "p"): 0.3, ("c", "q"): 0.2, ("p", "q"): 0.0}
            ]
            * 6
        ),
        _feature_views({node: (0.10, 1.0) for node in "cpq"}),
    )
    applied = apply_observed_collider_refinements(motif, refinement)
    assert _pairs(applied.direct_candidates) == {
        frozenset(("c", "p")),
        frozenset(("c", "q")),
    }
    assert applied.demoted_or_unresolved.empty
    assert set(applied.direct_candidates["observed_refinement_action"]) == {
        "complete_dual_supported_unshielded_anchor"
    }
    assert set(applied.direct_candidates["completion_anchor_node1"]) == {"c"}
    assert set(applied.direct_candidates["completion_anchor_node2"]) == {"p"}
    assert set(applied.direct_candidates["completion_winner_view_count"]) == {6}
    assert set(applied.direct_candidates["completion_n_views"]) == {6}
    assert len(applied.wedge_diagnostics) == 1
    assert bool(applied.wedge_diagnostics.iloc[0]["accepted"])
    assert applied.wedge_diagnostics.iloc[0]["center"] == "c"


def test_apply_noop_preserves_frames_and_rejects_second_application() -> None:
    votes = _votes(
        {
            ("c", "p"): (6, 0.001, 2.0),
            ("c", "q"): (6, 0.001, 1.8),
            ("p", "q"): (0, 0.8, 0.1),
        }
    )
    motif = select_unique_collider_motifs(
        _base_frame([("c", "p"), ("c", "q")]), votes
    )
    correlations = {
        ("c", "p"): 0.3,
        ("c", "q"): 0.2,
        ("p", "q"): 0.0,
    }
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        votes,
        _base_views({("c", "p"): 6, ("c", "q"): 6}),
        _residual_views([correlations] * 6),
        _feature_views({node: (0.10, 1.0) for node in "cpq"}),
    )
    assert refinement.decisions.empty
    applied = apply_observed_collider_refinements(motif, refinement)
    expected_direct = motif.direct_candidates.copy()
    expected_direct["observed_refinement_action"] = ""
    pd.testing.assert_frame_equal(applied.direct_candidates, expected_direct)
    assert applied.metadata["schema_version"] == "collider_motif_selector.v4"
    pd.testing.assert_frame_equal(
        applied.demoted_or_unresolved, motif.demoted_or_unresolved
    )
    pd.testing.assert_frame_equal(
        applied.component_diagnostics, motif.component_diagnostics
    )
    pd.testing.assert_frame_equal(applied.wedge_diagnostics, motif.wedge_diagnostics)
    with pytest.raises(ValueError, match="already been applied"):
        apply_observed_collider_refinements(applied, refinement)


def test_train_diagnostic_rejects_nontrain_seed_block() -> None:
    assert _validated_train_seeds(33004, 33011) == list(range(33004, 33012))
    with pytest.raises(ValueError, match="restricted to fresh TRAIN"):
        _validated_train_seeds(36001, 36024)
