from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
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
                "contains_sentinel_candidate": True,
                "candidate_pair_count": 3,
                "sentinel_candidate_feature_count": 1,
                "sentinel_null_feature_opportunities": 2,
            },
        ]
    )


def test_all_requested_metric_families_are_reported() -> None:
    coverage = benchmark.coverage_metrics(_frame(), total_truth=3)
    stress = benchmark.exact_cross_assay_stress(
        cross_truth_pairs=9,
        direction_truths=0,
        interpretation="set_valued_method_abstains",
    )
    families = benchmark.metric_families(coverage, stress)

    assert set(families) == {
        "shared_state_hyperedge",
        "assay_participation_and_sentinel_false_positive",
        "equivalence_class",
        "identifiability_stress_test",
    }
    assert families["shared_state_hyperedge"][
        "shared_state_hyperedge_precision"
    ] == 0.5
    assert families["shared_state_hyperedge"][
        "shared_state_hyperedge_recall"
    ] == 1.0 / 3.0
    assert families["assay_participation_and_sentinel_false_positive"][
        "assay_participation_precision"
    ] == 0.5
    assert families["assay_participation_and_sentinel_false_positive"][
        "assay_participation_recall"
    ] == 1.0 / 3.0
    assert families["equivalence_class"]["equivalence_class_coverage"] == (
        1.0 / 3.0
    )


def test_sentinel_fpr_uses_null_opportunities_not_selected_predictions() -> None:
    metrics = benchmark.coverage_metrics(_frame(), total_truth=3)

    assert metrics["sentinel_false_positive_features"] == 1
    assert metrics["sentinel_null_feature_opportunities"] == 6
    assert metrics["sentinel_false_positive_rate"] == 1.0 / 6.0
    assert metrics["sentinel_false_discovery_proportion"] == 0.5


def test_multi_assay_hyperedges_use_explicit_participation_counts() -> None:
    metrics = benchmark.coverage_metrics(
        _frame(),
        total_truth=3,
        truth_assay_participations=6,
        predicted_assay_participations=5,
        true_positive_assay_participations=4,
        ambiguous_assay_participation_hypotheses=2,
        candidate_assay_pair_alternatives=6,
        assay_participation_point_abstentions=2,
        sentinel_null_feature_opportunities=12,
        sentinel_false_positive_features=1,
    )

    assert metrics["assay_participation_precision"] == 0.8
    assert metrics["assay_participation_recall"] == 4.0 / 6.0
    assert metrics["assay_participation_false_positives"] == 1
    assert metrics["assay_participation_false_negatives"] == 2
    assert metrics["sentinel_false_positive_rate"] == 1.0 / 12.0
    assert metrics["ambiguous_assay_participation_hypotheses"] == 2
    assert metrics["candidate_assay_pair_alternatives"] == 6
    assert metrics["assay_participation_point_abstentions"] == 2


def test_ambiguity_alternatives_are_not_scored_as_edges() -> None:
    metrics = benchmark.coverage_metrics(_frame(), total_truth=3)

    assert metrics["selected_hyperedges"] == 2
    assert metrics["candidate_pair_alternatives"] == 6
    assert metrics["average_ambiguity_set_size"] == 3.0
    assert not metrics["candidate_pair_alternatives_expanded_as_edges"]


def test_exact_pair_and_direction_metrics_are_stress_only() -> None:
    stress = benchmark.exact_cross_assay_stress(
        cross_truth_pairs=9,
        pair_predictions=2,
        pair_true_positives=1,
        direction_truths=4,
        direction_predictions=1,
        direction_true_positives=1,
        interpretation="diagnostic_only",
    )

    assert stress["exact_cross_assay_pair_precision"] == 0.5
    assert stress["exact_cross_assay_pair_recall"] == 1.0 / 9.0
    assert stress["cross_assay_direction_precision"] == 1.0
    assert stress["cross_assay_direction_recovery"] == 0.25
    assert not stress["included_in_reliability_gate"]
    assert "stress_test" in stress["label"]


def test_invalid_matching_and_sentinel_counts_are_rejected() -> None:
    invalid_match = _frame()
    invalid_match.loc[2, "matched_to_unique_truth_pair"] = True
    with pytest.raises(ValueError, match="must also be selected"):
        benchmark.coverage_metrics(invalid_match, total_truth=3)

    invalid_sentinel = _frame()
    invalid_sentinel.loc[0, "sentinel_candidate_feature_count"] = 3
    with pytest.raises(ValueError, match="cannot exceed"):
        benchmark.coverage_metrics(invalid_sentinel, total_truth=3)
