from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import component_conditioned_cross_assay_equivalence_v4 as bridge  # noqa: E402


def test_depth_adjusted_association_finds_focal_component() -> None:
    rng = np.random.default_rng(3)
    cells = [f"c{index}" for index in range(500)]
    components = rng.normal(size=(500, 3))
    depth = np.exp(4.0 + 0.25 * components[:, 0] + rng.normal(scale=0.2, size=500))
    focal_signal = rng.poisson(
        np.exp(-1.0 + 0.9 * components[:, 2] + 0.2 * np.log(depth))
    )
    focal = pd.DataFrame(
        {
            "cell": cells,
            "modality": "chrom_A",
            "observed_background_library_size": depth,
            "observed_library_size": depth + focal_signal,
            "signal": focal_signal,
        }
    )
    scores = pd.DataFrame(
        {
            "cell_id": cells,
            "modality": "chrom_A",
            "component_0": components[:, 0],
            "component_1": components[:, 1],
            "component_2": components[:, 2],
        }
    )

    result = bridge.focal_component_associations(focal, scores)
    stability = bridge.bootstrap_focal_component_assignments(
        focal, scores, ["signal"], replicates=64
    )

    assert int(result.loc[0, "dominant_component"]) == 2
    assert result.loc[0, "dominant_absolute_partial_correlation"] > 0.4
    assert int(stability.loc[0, "selected_component"]) == 2
    assert stability.loc[0, "component_assignment_bootstrap_supported"]
    assert stability.loc[0, "selected_component_assignment_fraction"] > 0.9


def test_full_and_fold_signature_builds_one_set_valued_hyperedge() -> None:
    hypotheses = pd.DataFrame(
        [
            {
                "block": "b",
                "seed": 1,
                "hypothesis_id": "h_a",
                "modality": "a",
                "direct_parent_members": "a_p;a_q",
                "external_state_reporter_members": "a_s",
                "candidate_pairs_json": json.dumps(
                    [["a_p", "a_s"], ["a_q", "a_s"]]
                ),
                "candidate_pair_count": 2,
                "hypothesis_selected": True,
            },
            {
                "block": "b",
                "seed": 1,
                "hypothesis_id": "h_b",
                "modality": "b",
                "direct_parent_members": "b_p;b_q",
                "external_state_reporter_members": "b_s",
                "candidate_pairs_json": json.dumps(
                    [["b_p", "b_s"], ["b_q", "b_s"]]
                ),
                "candidate_pair_count": 2,
                "hypothesis_selected": True,
            },
        ]
    )
    associations = pd.DataFrame(
        [
            {"modality": "a", "feature": "a_p", "dominant_component": 1},
            {"modality": "a", "feature": "a_q", "dominant_component": 0},
            {"modality": "a", "feature": "a_s", "dominant_component": 1},
            {"modality": "b", "feature": "b_p", "dominant_component": 2},
            {"modality": "b", "feature": "b_q", "dominant_component": 0},
            {"modality": "b", "feature": "b_s", "dominant_component": 2},
        ]
    )
    associations["dominant_absolute_partial_correlation"] = 0.3
    associations["component_relative_margin"] = 0.5
    fold_associations = pd.concat(
        [
            associations.assign(feature_fold=0),
            associations.assign(feature_fold=1),
        ],
        ignore_index=True,
    )
    full = pd.DataFrame(
        {
            "component_assignment_json": [
                json.dumps({"a": 0, "b": 0}),
                json.dumps({"a": 1, "b": 2}),
                json.dumps({"a": 2, "b": 1}),
            ],
            "pairwise_match_support": [1.0, 1.0, 1.0],
            "maximum_match_clique_modalities": ["a;b", "a;b", "a;b"],
            "maximum_match_clique_fraction": [1.0, 1.0, 1.0],
        }
    )
    folds = pd.concat(
        [full.assign(feature_fold=0), full.assign(feature_fold=1)],
        ignore_index=True,
    )
    pairwise = pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "left_component": left,
                "right_component": right,
                "matched": (left, right) == (1, 2),
                "scaled_ks_statistic": (
                    0.1
                    if (left, right) == (1, 2)
                    else 1.0 + 0.1 * left + 0.01 * right
                ),
                "asymptotic_critical_value": 2.0,
            }
            for left in range(3)
            for right in range(3)
        ]
    )
    fold_pairwise = pd.concat(
        [pairwise.assign(feature_fold=0), pairwise.assign(feature_fold=1)],
        ignore_index=True,
    )
    bootstrap = pd.DataFrame(
        [
            {
                "modality": "a",
                "feature": "a_s",
                "selected_component": 1,
                "selected_component_assignment_fraction": 0.9,
                "bonferroni_component_assignment_p": 0.001,
                "component_assignment_bootstrap_supported": True,
            },
            {
                "modality": "b",
                "feature": "b_s",
                "selected_component": 2,
                "selected_component_assignment_fraction": 0.9,
                "bonferroni_component_assignment_p": 0.001,
                "component_assignment_bootstrap_supported": True,
            },
        ]
    )
    fold_bootstrap = pd.concat(
        [
            bootstrap.assign(feature_fold=0),
            bootstrap.assign(feature_fold=1),
        ],
        ignore_index=True,
    )

    assay, hyperedges, pair_decisions = (
        bridge.build_component_conditioned_equivalence(
            hypotheses,
            associations,
            full,
            folds,
            pairwise_matches=pairwise,
            fold_pairwise_matches=fold_pairwise,
            fold_associations=fold_associations,
            reporter_bootstrap_stability=bootstrap,
            fold_reporter_bootstrap_stability=fold_bootstrap,
        )
    )

    assert assay["within_assay_pair_resolved_by_component"].all()
    assert len(hyperedges) == 1
    assert hyperedges.loc[0, "modalities"] == "a;b"
    assert hyperedges.loc[0, "candidate_cross_assay_feature_pair_count"] == 1
    assert hyperedges.loc[0, "candidate_assay_pair_count"] == 1
    assert hyperedges.loc[0, "exact_assay_participation_claimed"]
    assert not hyperedges.loc[0, "assay_participation_is_set_valued"]
    assert not hyperedges.loc[0, "exact_cross_assay_pair_claimed"]
    assert not hyperedges.loc[0, "candidate_pairs_may_be_expanded_as_edges"]
    assert hyperedges.loc[0, "maximum_clique_stability_guard_passed"]
    assert hyperedges.loc[0, "full_and_all_fold_direct_component_pairs_matched"]
    assert hyperedges.loc[
        0, "all_reporter_components_stable_across_full_and_feature_folds"
    ]
    assert hyperedges.loc[0, "half_assay_clique_support_guard_passed"]
    assert hyperedges.loc[
        0, "all_within_assay_pairs_component_resolved"
    ]
    assert hyperedges.loc[
        0, "local_structure_or_all_context_clique_guard_passed"
    ]
    assert hyperedges.loc[0, "hypothesis_selected"]
    assert hyperedges.loc[
        0, "all_reporter_bootstrap_component_assignments_supported"
    ]
    assert hyperedges.loc[
        0,
        "all_reporter_feature_fold_bootstrap_component_assignments_supported",
    ]
    assert hyperedges.loc[
        0, "all_reporter_context_bootstrap_component_assignments_supported"
    ]
    assert len(pair_decisions) == 1
    assert pair_decisions.loc[0, "retained_in_output_hyperedge"]
    assert pair_decisions.loc[0, "statistical_context_support_pattern"] == "111"
    assert (
        pair_decisions.loc[0, "reporter_bootstrap_context_support_pattern"]
        == "111"
    )

    unstable = fold_associations.copy()
    unstable.loc[
        unstable["feature_fold"].eq(1)
        & unstable["modality"].eq("b")
        & unstable["feature"].eq("b_s"),
        "dominant_component",
    ] = 0
    unstable_fold_bootstrap = fold_bootstrap.copy()
    unstable_fold_bootstrap.loc[
        unstable_fold_bootstrap["feature_fold"].eq(1)
        & unstable_fold_bootstrap["modality"].eq("b"),
        "selected_component",
    ] = 0
    unstable_assay, unstable_hyperedges, unstable_decisions = (
        bridge.build_component_conditioned_equivalence(
            hypotheses,
            associations,
            full,
            folds,
            pairwise_matches=pairwise,
            fold_pairwise_matches=fold_pairwise,
            fold_associations=unstable,
            reporter_bootstrap_stability=bootstrap,
            fold_reporter_bootstrap_stability=unstable_fold_bootstrap,
        )
    )

    assert not bool(
        unstable_assay.loc[
            unstable_assay["modality"].eq("b"),
            "reporter_components_stable_across_full_and_feature_folds",
        ].iloc[0]
    )
    assert unstable_hyperedges.empty
    assert (
        unstable_decisions.loc[0, "selection_status"]
        == "unresolved_not_all_contexts_supported"
    )

    minority_only = unstable.copy()
    minority_only.loc[
        minority_only["feature_fold"].eq(0)
        & minority_only["modality"].eq("b")
        & minority_only["feature"].eq("b_s"),
        "dominant_component",
    ] = 0
    minority_fold_bootstrap = unstable_fold_bootstrap.copy()
    minority_fold_bootstrap.loc[
        minority_fold_bootstrap["feature_fold"].eq(0)
        & minority_fold_bootstrap["modality"].eq("b"),
        "selected_component",
    ] = 0
    _, minority_hyperedges, minority_decisions = (
        bridge.build_component_conditioned_equivalence(
        hypotheses,
        associations,
        full,
        folds,
        pairwise_matches=pairwise,
            fold_pairwise_matches=fold_pairwise,
            fold_associations=minority_only,
            reporter_bootstrap_stability=bootstrap,
            fold_reporter_bootstrap_stability=minority_fold_bootstrap,
        )
    )

    assert minority_hyperedges.empty
    assert not minority_decisions.loc[0, "strict_context_majority_passed"]

    unsupported_bootstrap = bootstrap.copy()
    unsupported_bootstrap.loc[
        unsupported_bootstrap["modality"].eq("b"),
        "component_assignment_bootstrap_supported",
    ] = False
    _, unsupported_hyperedges, unsupported_decisions = (
        bridge.build_component_conditioned_equivalence(
            hypotheses,
            associations,
            full,
            folds,
            pairwise_matches=pairwise,
            fold_pairwise_matches=fold_pairwise,
            fold_associations=fold_associations,
            reporter_bootstrap_stability=unsupported_bootstrap,
            fold_reporter_bootstrap_stability=fold_bootstrap,
        )
    )

    assert unsupported_hyperedges.empty
    assert (
        unsupported_decisions.loc[0, "selection_status"]
        == "unresolved_bootstrap_component_assignment"
    )

    unsupported_fold_bootstrap = fold_bootstrap.copy()
    unsupported_fold_bootstrap.loc[
        unsupported_fold_bootstrap["feature_fold"].eq(1)
        & unsupported_fold_bootstrap["modality"].eq("b"),
        "component_assignment_bootstrap_supported",
    ] = False
    _, unsupported_fold_hyperedges, unsupported_fold_decisions = (
        bridge.build_component_conditioned_equivalence(
            hypotheses,
            associations,
            full,
            folds,
            pairwise_matches=pairwise,
            fold_pairwise_matches=fold_pairwise,
            fold_associations=fold_associations,
            reporter_bootstrap_stability=bootstrap,
            fold_reporter_bootstrap_stability=unsupported_fold_bootstrap,
        )
    )

    assert unsupported_fold_hyperedges.empty
    assert (
        unsupported_fold_decisions.loc[0, "statistical_context_support_pattern"]
        == "111"
    )
    assert (
        unsupported_fold_decisions.loc[
            0, "reporter_bootstrap_context_support_pattern"
        ]
        == "110"
    )
    assert unsupported_fold_decisions.loc[0, "context_support_pattern"] == "110"
    assert (
        unsupported_fold_decisions.loc[0, "selection_status"]
        == "unresolved_feature_fold_bootstrap_component_assignment"
    )


def _three_assay_cross_context_ambiguity_inputs() -> dict[str, pd.DataFrame]:
    modalities = ["a", "b", "c"]
    hypotheses = pd.DataFrame(
        [
            {
                "block": "b",
                "seed": 1,
                "hypothesis_id": f"h_{modality}",
                "modality": modality,
                "direct_parent_members": f"{modality}_p;{modality}_q",
                "external_state_reporter_members": f"{modality}_s",
                "candidate_pairs_json": json.dumps(
                    [
                        [f"{modality}_p", f"{modality}_s"],
                        [f"{modality}_q", f"{modality}_s"],
                    ]
                ),
                "candidate_pair_count": 2,
                "hypothesis_selected": True,
            }
            for modality in modalities
        ]
    )
    associations = pd.DataFrame(
        [
            {
                "modality": modality,
                "feature": f"{modality}_{suffix}",
                "dominant_component": component,
                "dominant_absolute_partial_correlation": 0.3,
                "component_relative_margin": 0.5,
            }
            for modality in modalities
            for suffix, component in (("p", 1), ("q", 0), ("s", 1))
        ]
    )
    fold_associations = pd.concat(
        [
            associations.assign(feature_fold=0),
            associations.assign(feature_fold=1),
        ],
        ignore_index=True,
    )

    ab_context = [
        {"a": 0, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 2},
        {"a": 2, "b": 2, "c": 1},
    ]
    ac_context = [
        {"a": 0, "b": 0, "c": 0},
        {"a": 1, "b": 2, "c": 1},
        {"a": 2, "b": 1, "c": 2},
    ]

    def consensus_frame(assignments: list[dict[str, int]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component_assignment_json": [
                    json.dumps(value) for value in assignments
                ],
                "pairwise_match_support": [1.0, 1.0, 1.0],
                "maximum_match_clique_modalities": [
                    "a;b;c",
                    "a;b" if assignments is ab_context else "a;c",
                    "c" if assignments is ab_context else "b",
                ],
                "maximum_match_clique_fraction": [1.0, 2 / 3, 1 / 3],
            }
        )

    def pairwise_frame(assignments: list[dict[str, int]]) -> pd.DataFrame:
        axis_by_component = {
            modality: {
                int(row[modality]): axis
                for axis, row in enumerate(assignments)
            }
            for modality in modalities
        }
        rows = []
        for left_index, left in enumerate(modalities):
            for right in modalities[left_index + 1 :]:
                for left_component in range(3):
                    for right_component in range(3):
                        matched = bool(
                            axis_by_component[left][left_component]
                            == axis_by_component[right][right_component]
                        )
                        rows.append(
                            {
                                "left_modality": left,
                                "right_modality": right,
                                "left_component": left_component,
                                "right_component": right_component,
                                "matched": matched,
                                "scaled_ks_statistic": (
                                    0.1
                                    + 0.001 * left_component
                                    + 0.0001 * right_component
                                    if matched
                                    else 1.0
                                    + 0.1 * left_component
                                    + 0.01 * right_component
                                ),
                                "asymptotic_critical_value": 0.5,
                            }
                        )
        return pd.DataFrame(rows)

    full = consensus_frame(ab_context)
    folds = pd.concat(
        [
            consensus_frame(ab_context).assign(feature_fold=0),
            consensus_frame(ac_context).assign(feature_fold=1),
        ],
        ignore_index=True,
    )
    pairwise = pairwise_frame(ab_context)
    fold_pairwise = pd.concat(
        [
            pairwise_frame(ab_context).assign(feature_fold=0),
            pairwise_frame(ac_context).assign(feature_fold=1),
        ],
        ignore_index=True,
    )
    bootstrap = pd.DataFrame(
        [
            {
                "modality": modality,
                "feature": f"{modality}_s",
                "selected_component": 1,
                "selected_component_assignment_fraction": 0.9,
                "bonferroni_component_assignment_p": 0.001,
                "component_assignment_bootstrap_supported": True,
            }
            for modality in modalities
        ]
    )
    fold_bootstrap = pd.concat(
        [
            bootstrap.assign(feature_fold=0),
            bootstrap.assign(feature_fold=1),
        ],
        ignore_index=True,
    )
    return {
        "hypotheses": hypotheses,
        "associations": associations,
        "consensus": full,
        "fold_consensus": folds,
        "pairwise_matches": pairwise,
        "fold_pairwise_matches": fold_pairwise,
        "fold_associations": fold_associations,
        "reporter_bootstrap_stability": bootstrap,
        "fold_reporter_bootstrap_stability": fold_bootstrap,
    }


def test_cross_context_alternatives_form_one_unexpanded_ambiguity_set() -> None:
    _, hyperedges, decisions = bridge.build_component_conditioned_equivalence(
        **_three_assay_cross_context_ambiguity_inputs()
    )

    assert len(hyperedges) == 1
    hyperedge = hyperedges.iloc[0]
    assert hyperedge["hypothesis_type"] == "cross_assay_shared_state_ambiguity"
    assert json.loads(hyperedge["ambiguity_source_assay_pair_json"]) == [
        "a",
        "b",
    ]
    assert json.loads(hyperedge["ambiguity_context_indices_json"]) == [2]
    assert json.loads(hyperedge["candidate_assay_pairs_json"]) == [
        ["a", "b"],
        ["a", "c"],
    ]
    assert json.loads(hyperedge["candidate_cross_assay_feature_pairs_json"]) == [
        ["a_s", "b_s"],
        ["a_s", "c_s"],
    ]
    assert hyperedge["assay_participation_is_set_valued"]
    assert not hyperedge["exact_assay_participation_claimed"]
    assert not hyperedge["exact_cross_assay_pair_claimed"]
    assert not hyperedge["cross_assay_direction_claimed"]
    assert not hyperedge["candidate_pairs_may_be_expanded_as_edges"]

    focal = decisions.loc[
        decisions["left_modality"].eq("a")
        & decisions["right_modality"].eq("b")
    ].iloc[0]
    alternative = decisions.loc[
        decisions["left_modality"].eq("a")
        & decisions["right_modality"].eq("c")
    ].iloc[0]
    assert focal["statistical_context_support_pattern"] == "110"
    assert focal["reporter_bootstrap_context_support_pattern"] == "111"
    assert focal["selection_status"] == (
        "retained_in_cross_context_assay_pair_ambiguity"
    )
    assert focal["retained_in_output_hyperedge"]
    assert alternative["statistical_context_support_pattern"] == "001"
    assert alternative["included_in_output_ambiguity_set"]
    assert not alternative["retained_in_output_hyperedge"]
    assert alternative["selection_status"] == (
        "included_only_as_unexpanded_cross_context_ambiguity_alternative"
    )
    assert not decisions["candidate_pair_may_be_expanded_as_edge"].any()


def test_unresolved_local_assay_can_only_enter_as_ambiguity_alternative() -> None:
    inputs = _three_assay_cross_context_ambiguity_inputs()
    associations = inputs["associations"].copy()
    associations.loc[
        associations["modality"].eq("c")
        & associations["feature"].eq("c_q"),
        "dominant_component",
    ] = 1
    inputs["associations"] = associations
    fold_associations = inputs["fold_associations"].copy()
    fold_associations.loc[
        fold_associations["modality"].eq("c")
        & fold_associations["feature"].eq("c_q"),
        "dominant_component",
    ] = 1
    inputs["fold_associations"] = fold_associations

    assay, hyperedges, decisions = (
        bridge.build_component_conditioned_equivalence(**inputs)
    )

    c_assignment = assay.loc[assay["modality"].eq("c")].iloc[0]
    assert not c_assignment["within_assay_pair_resolved_by_component"]
    assert len(hyperedges) == 1
    hyperedge = hyperedges.iloc[0]
    assert json.loads(
        hyperedge["ambiguity_unresolved_local_alternative_modalities_json"]
    ) == ["c"]
    assert hyperedge[
        "ambiguity_all_focal_assays_within_assay_pair_resolved"
    ]
    assert not hyperedge[
        "ambiguity_all_candidate_modalities_within_assay_pair_resolved"
    ]
    assert hyperedge["assay_participation_is_set_valued"]
    assert not hyperedge["exact_assay_participation_claimed"]

    alternative = decisions.loc[
        decisions["left_modality"].eq("a")
        & decisions["right_modality"].eq("c")
    ].iloc[0]
    assert alternative["eligible_as_cross_context_ambiguity_alternative"]
    assert not alternative["eligible_for_primary_stable_pair_graph"]
    assert alternative["included_in_output_ambiguity_set"]
    assert not alternative["majority_graph_edge_selected"]
    assert not alternative["candidate_pair_may_be_expanded_as_edge"]


def test_cross_context_rescue_abstains_without_an_alternative_pair() -> None:
    inputs = _three_assay_cross_context_ambiguity_inputs()
    fold_pairwise = inputs["fold_pairwise_matches"].copy()
    reporter_ac = (
        fold_pairwise["feature_fold"].eq(1)
        & fold_pairwise["left_modality"].eq("a")
        & fold_pairwise["right_modality"].eq("c")
        & fold_pairwise["left_component"].eq(1)
        & fold_pairwise["right_component"].eq(1)
    )
    fold_pairwise.loc[reporter_ac, "matched"] = False
    inputs["fold_pairwise_matches"] = fold_pairwise

    _, hyperedges, decisions = bridge.build_component_conditioned_equivalence(
        **inputs
    )

    assert hyperedges.empty
    focal = decisions.loc[
        decisions["left_modality"].eq("a")
        & decisions["right_modality"].eq("b")
    ].iloc[0]
    assert focal["selection_status"] == "unresolved_not_all_contexts_supported"
    assert not decisions["included_in_output_ambiguity_set"].any()


def test_local_structure_or_all_context_clique_guard_is_boolean_not_tuned() -> None:
    guard = bridge.local_structure_or_all_context_clique_guard

    assert guard(
        half_assay_clique_support=True,
        all_within_assay_pairs_component_resolved=True,
        all_modalities_in_every_context_maximum_clique=False,
    )
    assert guard(
        half_assay_clique_support=True,
        all_within_assay_pairs_component_resolved=False,
        all_modalities_in_every_context_maximum_clique=True,
    )
    assert not guard(
        half_assay_clique_support=True,
        all_within_assay_pairs_component_resolved=False,
        all_modalities_in_every_context_maximum_clique=False,
    )
    assert not guard(
        half_assay_clique_support=False,
        all_within_assay_pairs_component_resolved=True,
        all_modalities_in_every_context_maximum_clique=True,
    )


def test_stable_pair_graph_collapses_transitive_links_into_one_hyperedge() -> None:
    components = bridge.stable_pair_graph_components(
        ["assay_c", "assay_a", "assay_b", "isolated"],
        {
            frozenset(("assay_a", "assay_b")),
            frozenset(("assay_b", "assay_c")),
        },
    )

    assert components == [["assay_a", "assay_b", "assay_c"]]


def test_direct_component_assignment_margin_rejects_tied_match() -> None:
    table = pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "left_component": left,
                "right_component": right,
                "scaled_ks_statistic": statistic,
            }
            for left, right, statistic in [
                (0, 0, 1.0),
                (0, 1, 1.0),
                (0, 2, 3.0),
                (1, 0, 2.0),
                (1, 1, 4.0),
                (1, 2, 5.0),
                (2, 0, 6.0),
                (2, 1, 7.0),
                (2, 2, 8.0),
            ]
        ]
    )

    margin = bridge.direct_component_assignment_margin(table, table.iloc[0])

    assert margin == 0.0


def test_minimum_cost_component_bijection_resolves_a_local_collision() -> None:
    costs = [
        [0.01, 0.40, 0.90],
        [0.09, 0.10, 0.90],
        [0.90, 0.90, 0.10],
    ]
    table = pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "left_component": left,
                "right_component": right,
                "scaled_ks_statistic": costs[left][right],
            }
            for left in range(3)
            for right in range(3)
        ]
    )

    mapping, best_cost, relative_margin, maximum_assigned = (
        bridge.minimum_cost_component_bijection(table, "a", "b")
    )
    reverse_mapping, _, reverse_margin, _ = (
        bridge.minimum_cost_component_bijection(table, "b", "a")
    )

    assert mapping == {0: 0, 1: 1, 2: 2}
    assert reverse_mapping == {0: 0, 1: 1, 2: 2}
    assert best_cost == pytest.approx(0.21)
    assert relative_margin > 0.0
    assert reverse_margin == pytest.approx(relative_margin)
    assert maximum_assigned == pytest.approx(0.10)


def test_minimum_cost_component_bijection_marks_a_tied_optimum() -> None:
    table = pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "left_component": left,
                "right_component": right,
                "scaled_ks_statistic": 1.0,
            }
            for left in range(3)
            for right in range(3)
        ]
    )

    mapping, best_cost, relative_margin, maximum_assigned = (
        bridge.minimum_cost_component_bijection(table, "a", "b")
    )

    assert mapping == {0: 0, 1: 1, 2: 2}
    assert best_cost == 3.0
    assert relative_margin == 0.0
    assert maximum_assigned == 1.0


def test_global_bijection_completes_only_after_ambiguity_abstains() -> None:
    inputs = _three_assay_cross_context_ambiguity_inputs()
    fold_pairwise = inputs["fold_pairwise_matches"].copy()
    missing_fold = fold_pairwise["feature_fold"].eq(1)
    reporter_ac = (
        missing_fold
        & fold_pairwise["left_modality"].eq("a")
        & fold_pairwise["right_modality"].eq("c")
        & fold_pairwise["left_component"].eq(1)
        & fold_pairwise["right_component"].eq(1)
    )
    fold_pairwise.loc[reporter_ac, "matched"] = False
    costs = np.asarray(
        [
            [0.01, 0.40, 0.90],
            [0.09, 0.10, 0.90],
            [0.90, 0.90, 0.10],
        ]
    )
    assay_ab = (
        missing_fold
        & fold_pairwise["left_modality"].eq("a")
        & fold_pairwise["right_modality"].eq("b")
    )
    for row_index in fold_pairwise.index[assay_ab]:
        left_component = int(fold_pairwise.at[row_index, "left_component"])
        right_component = int(
            fold_pairwise.at[row_index, "right_component"]
        )
        fold_pairwise.at[row_index, "scaled_ks_statistic"] = costs[
            left_component, right_component
        ]
        fold_pairwise.at[row_index, "matched"] = bool(
            (left_component, right_component) in {(0, 0), (2, 2)}
        )
    inputs["fold_pairwise_matches"] = fold_pairwise

    _, hyperedges, decisions = bridge.build_component_conditioned_equivalence(
        **inputs
    )

    assert len(hyperedges) == 1
    hyperedge = hyperedges.iloc[0]
    assert hyperedge["selection_route"] == "global_bijection_completion"
    assert hyperedge["global_bijection_context_support_pattern"] == "111"
    assert hyperedge["all_context_global_bijection_supported"]
    assert hyperedge["exact_assay_participation_claimed"]
    assert not hyperedge["assay_participation_is_set_valued"]
    assert json.loads(hyperedge["candidate_assay_pairs_json"]) == [["a", "b"]]
    focal = decisions.loc[
        decisions["left_modality"].eq("a")
        & decisions["right_modality"].eq("b")
    ].iloc[0]
    assert focal["statistical_context_support_pattern"] == "110"
    assert focal["retained_by_global_bijection_completion"]
    assert focal["selection_status"] == (
        "retained_as_global_bijection_completed_assay_participation_"
        "equivalence"
    )
    assert not decisions["included_in_output_ambiguity_set"].any()


def test_cross_context_ambiguity_precedes_global_bijection_completion() -> None:
    inputs = _three_assay_cross_context_ambiguity_inputs()
    fold_pairwise = inputs["fold_pairwise_matches"].copy()
    missing_fold_ab = (
        fold_pairwise["feature_fold"].eq(1)
        & fold_pairwise["left_modality"].eq("a")
        & fold_pairwise["right_modality"].eq("b")
    )
    costs = np.asarray(
        [
            [0.01, 0.40, 0.90],
            [0.09, 0.10, 0.90],
            [0.90, 0.90, 0.10],
        ]
    )
    for row_index in fold_pairwise.index[missing_fold_ab]:
        left_component = int(fold_pairwise.at[row_index, "left_component"])
        right_component = int(
            fold_pairwise.at[row_index, "right_component"]
        )
        fold_pairwise.at[row_index, "scaled_ks_statistic"] = costs[
            left_component, right_component
        ]
        fold_pairwise.at[row_index, "matched"] = bool(
            (left_component, right_component) in {(0, 0), (2, 2)}
        )
    inputs["fold_pairwise_matches"] = fold_pairwise

    _, hyperedges, decisions = bridge.build_component_conditioned_equivalence(
        **inputs
    )

    assert len(hyperedges) == 1
    hyperedge = hyperedges.iloc[0]
    assert hyperedge["selection_route"] == "cross_context_alternative_set"
    assert hyperedge["all_context_global_bijection_supported"]
    assert hyperedge["assay_participation_is_set_valued"]
    focal = decisions.loc[
        decisions["left_modality"].eq("a")
        & decisions["right_modality"].eq("b")
    ].iloc[0]
    assert not focal["retained_by_global_bijection_completion"]
    assert focal["included_in_output_ambiguity_set"]


def test_no_cross_assay_group_keeps_a_header_only_hyperedge_table() -> None:
    hypotheses = pd.DataFrame(columns=bridge.HYPOTHESIS_COLUMNS)
    associations = pd.DataFrame(
        columns=[
            "modality",
            "feature",
            "dominant_component",
            "dominant_absolute_partial_correlation",
            "component_relative_margin",
        ]
    )
    consensus = pd.DataFrame(
        {
            "component_assignment_json": [json.dumps({"a": 0})],
            "pairwise_match_support": [1.0],
            "maximum_match_clique_modalities": ["a"],
            "maximum_match_clique_fraction": [1.0],
        }
    )
    folds = pd.concat(
        [consensus.assign(feature_fold=0), consensus.assign(feature_fold=1)]
    )

    _, hyperedges, pair_decisions = bridge.build_component_conditioned_equivalence(
        hypotheses, associations, consensus, folds
    )

    assert hyperedges.empty
    assert hyperedges.columns.tolist() == bridge.HYPEREDGE_COLUMNS
    assert pair_decisions.empty
    assert pair_decisions.columns.tolist() == bridge.PAIR_DECISION_COLUMNS
