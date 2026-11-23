from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402


def _pairwise_costs(
    orders: dict[str, list[int]], *, selected_cost: float = 0.1
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair_index, (left, right) in enumerate(
        combinations(sorted(orders), 2)
    ):
        selected = set(zip(orders[left], orders[right], strict=True))
        reverse_orientation = pair_index == 1
        for left_component in range(3):
            for right_component in range(3):
                row_left = right if reverse_orientation else left
                row_right = left if reverse_orientation else right
                row_left_component = (
                    right_component if reverse_orientation else left_component
                )
                row_right_component = (
                    left_component if reverse_orientation else right_component
                )
                rows.append(
                    {
                        "left_modality": row_left,
                        "right_modality": row_right,
                        "left_component": row_left_component,
                        "right_component": row_right_component,
                        "scaled_ks_statistic": (
                            selected_cost
                            if (left_component, right_component) in selected
                            else 3.0
                        ),
                        "asymptotic_critical_value": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_pair_cost_matrix_normalizes_reverse_orientation() -> None:
    rows = []
    for left_component in range(3):
        for right_component in range(3):
            rows.append(
                {
                    "left_modality": "b",
                    "right_modality": "a",
                    "left_component": right_component,
                    "right_component": left_component,
                    "scaled_ks_statistic": 10 * left_component + right_component,
                    "asymptotic_critical_value": 2.0,
                }
            )

    matrix, critical = synchronization.pair_cost_matrix(
        pd.DataFrame(rows), "a", "b"
    )

    assert critical == 2.0
    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        ),
    )


def test_exact_multiway_solver_recovers_cycle_consistent_permutations() -> None:
    expected = {
        "a": [0, 1, 2],
        "b": [1, 2, 0],
        "c": [2, 0, 1],
    }
    pairwise = _pairwise_costs(expected)

    solution = synchronization.solve_multiway_permutation_synchronization(
        pairwise, ["c", "a", "b"]
    )

    observed = {
        modality: solution.best_orders[index].tolist()
        for index, modality in enumerate(solution.modalities)
    }
    assert observed == expected
    assert solution.global_relative_margin > 0.0
    assert len(solution.objectives) == 36


def test_reporter_pair_exclusion_margin_abstains_on_tied_optimum() -> None:
    orders = {"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]}
    informative = _pairwise_costs(orders)
    informative_solution = (
        synchronization.solve_multiway_permutation_synchronization(
            informative, list(orders)
        )
    )

    supported = synchronization.reporter_pair_diagnostics(
        informative_solution, informative, "a", "b", 0, 0
    )

    tied = informative.copy()
    tied["scaled_ks_statistic"] = 1.0
    tied_solution = synchronization.solve_multiway_permutation_synchronization(
        tied, list(orders)
    )
    ambiguous = synchronization.reporter_pair_diagnostics(
        tied_solution, tied, "a", "b", 0, 0
    )

    assert supported["coassigned_in_best_solution"]
    assert supported["pair_exclusion_relative_margin"] > 0.0
    assert supported["reporter_pair_scaled_ks_ratio"] == 0.1
    assert ambiguous["coassigned_in_best_solution"]
    assert ambiguous["pair_exclusion_relative_margin"] == 0.0


def test_candidate_builder_selects_repeated_three_assay_group() -> None:
    orders = {
        "a": [0, 1, 2],
        "b": [1, 2, 0],
        "c": [2, 0, 1],
    }
    reporters = {"a": "a_s", "b": "b_s", "c": "c_s"}
    reporter_components = {"a": 0, "b": 1, "c": 2}
    assay_rows = []
    for modality in sorted(orders):
        reporter = reporters[modality]
        component = reporter_components[modality]
        assay_rows.append(
            {
                "modality": modality,
                "within_assay_pair_resolved_by_component": True,
                "reporter_component_identified": True,
                "reporter_context_components_identified": True,
                "reporter_all_context_bootstrap_component_assignments_supported": True,
                "external_state_reporter_members": reporter,
                "reporter_local_component": component,
                "reporter_feature_fold_components_json": json.dumps(
                    {reporter: {"0": component, "1": component}}
                ),
            }
        )
    pair_decisions = pd.DataFrame(
        [
            {
                "left_modality": left,
                "right_modality": right,
                # An integer mirrors pandas' TSV inference for a raw `001`.
                "statistical_context_support_pattern": 1,
            }
            for left, right in combinations(sorted(orders), 2)
        ]
    )
    context = _pairwise_costs(orders)

    candidates, contexts = synchronization.build_synchronized_candidates(
        pd.DataFrame(assay_rows),
        pd.DataFrame(columns=["candidate_assay_pairs_json"]),
        pair_decisions,
        [context.copy(), context.copy(), context.copy()],
        ["full_fit", "feature_fold_0", "feature_fold_1"],
        [0, 1],
    )

    assert len(contexts) == 3
    assert len(candidates) == 1
    candidate = candidates.iloc[0]
    assert candidate["modalities"] == "a;b;c"
    assert candidate["candidate_assay_pair_count"] == 3
    assert candidate["candidate_cross_assay_feature_pair_count"] == 3
    assert candidate["hypothesis_selected"]
    assert candidate["minimum_pair_statistical_context_support_count"] == 1
    assert candidate["minimum_context_pair_exclusion_relative_margin"] > 0.0
    assert not candidate["exact_assay_participation_claimed"]
    assert not candidate["cross_assay_direction_claimed"]
    assert not candidate["candidate_pairs_may_be_expanded_as_edges"]
