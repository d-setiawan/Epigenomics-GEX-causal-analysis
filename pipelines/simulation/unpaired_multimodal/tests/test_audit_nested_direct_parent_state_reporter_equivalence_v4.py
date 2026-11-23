from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_nested_direct_parent_state_reporter_equivalence_v4 as audit  # noqa: E402


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def test_maximum_matching_uses_distinct_truth_pairs() -> None:
    truth = {_pair("a", "b"), _pair("c", "d")}
    candidates = [
        {_pair("a", "b"), _pair("c", "d")},
        {_pair("a", "b")},
    ]

    matching = audit.maximum_truth_matching(candidates, truth)

    assert len(matching) == 2
    assert set(matching.values()) == truth


def test_benchmark_counts_one_equivalence_set_as_one_prediction() -> None:
    frame = pd.DataFrame(
        [
            {
                "hypothesis_selected": True,
                "matched_to_unique_truth_pair": True,
                "contains_sentinel_candidate": False,
                "candidate_pair_count": 2,
                "sentinel_candidate_feature_count": 0,
                "sentinel_null_feature_opportunities": 2,
            },
            {
                "hypothesis_selected": True,
                "matched_to_unique_truth_pair": False,
                "contains_sentinel_candidate": True,
                "candidate_pair_count": 4,
                "sentinel_candidate_feature_count": 1,
                "sentinel_null_feature_opportunities": 2,
            },
            {
                "hypothesis_selected": False,
                "matched_to_unique_truth_pair": False,
                "contains_sentinel_candidate": False,
                "candidate_pair_count": 0,
                "sentinel_candidate_feature_count": 0,
                "sentinel_null_feature_opportunities": 2,
            },
        ]
    )

    metrics = audit._coverage_metrics(frame, total_truth=3)

    assert metrics["selected_hyperedges"] == 2
    assert metrics["true_positive_hyperedges"] == 1
    assert metrics["shared_state_hyperedge_precision"] == 0.5
    assert metrics["shared_state_hyperedge_recall"] == 1.0 / 3.0
    assert metrics["sentinel_false_positive_rate"] == 1.0 / 6.0
    assert metrics["sentinel_false_discovery_proportion"] == 0.5
    assert metrics["average_ambiguity_set_size"] == 3.0
    assert metrics["candidate_pair_alternatives"] == 6
    assert not metrics["candidate_pair_alternatives_expanded_as_edges"]


def test_exact_cross_assay_metrics_are_labeled_as_stress_only() -> None:
    stress = audit._exact_cross_assay_stress(
        cross_truth_pairs=9, direction_truths=0
    )

    assert stress["exact_cross_assay_pair_recall"] == 0.0
    assert stress["exact_cross_assay_pair_abstention_rate"] == 1.0
    assert stress["direction_is_not_defined_for_shared_state_truth"]
    assert not stress["included_in_reliability_gate"]
    assert "stress_test" in stress["label"]
