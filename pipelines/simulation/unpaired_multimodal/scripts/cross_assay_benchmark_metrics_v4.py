#!/usr/bin/env python3
"""Reusable benchmark metrics for set-valued cross-assay structure.

The benchmark treats one selected row as one hypothesis.  Candidate pairs
inside an ambiguity set are never expanded and scored as separate edges.  The
caller must perform a one-to-one hypothesis/truth matching before invoking
``coverage_metrics`` and record the result in
``matched_to_unique_truth_pair``.

Assay-participation counts use the same rows as assay-world participation
units.  Algorithms that return multi-assay hyperedges should first explode
their participation decisions to one row per hyperedge/assay unit before
using the participation fields.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


SCHEMA = "ca_boss_v4.cross_assay_benchmark_metrics.v1"
STRESS_LABEL = "identifiability_stress_test_not_expected_success_criterion"

REQUIRED_COLUMNS = {
    "hypothesis_selected",
    "matched_to_unique_truth_pair",
    "contains_sentinel_candidate",
    "candidate_pair_count",
    "sentinel_candidate_feature_count",
    "sentinel_null_feature_opportunities",
}


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a deterministic zero when a metric has no positive denominator."""

    return float(numerator / denominator) if denominator else 0.0


def _nonnegative_integer_series(
    frame: pd.DataFrame, column: str
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"{column} must contain nonnegative integers")
    return values.astype(int)


def coverage_metrics(
    frame: pd.DataFrame,
    *,
    total_truth: int,
    truth_assay_participations: int | None = None,
    predicted_assay_participations: int | None = None,
    true_positive_assay_participations: int | None = None,
    ambiguous_assay_participation_hypotheses: int = 0,
    candidate_assay_pair_alternatives: int = 0,
    assay_participation_point_abstentions: int = 0,
    sentinel_null_feature_opportunities: int | None = None,
    sentinel_false_positive_features: int | None = None,
) -> dict[str, Any]:
    """Score hyperedges, participation, sentinels, and ambiguity sets.

    ``sentinel_false_positive_rate`` uses sentinel features as its denominator:
    selected sentinel-null features divided by all sentinel-null feature
    opportunities.  ``sentinel_false_discovery_proportion`` separately reports
    the fraction of selected hypotheses containing at least one sentinel.
    """

    if missing := sorted(REQUIRED_COLUMNS.difference(frame.columns)):
        raise ValueError(f"Benchmark frame is missing columns: {missing}")
    if int(total_truth) != total_truth or total_truth < 0:
        raise ValueError("total_truth must be a nonnegative integer")
    total_truth = int(total_truth)

    selected = frame["hypothesis_selected"].astype(bool)
    matched = frame["matched_to_unique_truth_pair"].astype(bool)
    if (matched & ~selected).any():
        raise ValueError("A matched hypothesis must also be selected")

    ambiguity_sizes = _nonnegative_integer_series(frame, "candidate_pair_count")
    sentinel_selected_counts = _nonnegative_integer_series(
        frame, "sentinel_candidate_feature_count"
    )
    sentinel_opportunities = _nonnegative_integer_series(
        frame, "sentinel_null_feature_opportunities"
    )
    if (sentinel_selected_counts > sentinel_opportunities).any():
        raise ValueError(
            "Selected sentinel counts cannot exceed sentinel opportunities"
        )

    true_positive = int((selected & matched).sum())
    false_positive = int((selected & ~matched).sum())
    if true_positive > total_truth:
        raise ValueError("Matched hypotheses cannot exceed total_truth")
    false_negative = int(total_truth - true_positive)
    selected_count = int(selected.sum())

    sentinel_hypotheses = int(
        (
            selected
            & frame["contains_sentinel_candidate"].astype(bool)
        ).sum()
    )
    sentinel_false_positives = int(sentinel_selected_counts.loc[selected].sum())
    sentinel_nulls = int(sentinel_opportunities.sum())
    selected_ambiguity = ambiguity_sizes.loc[selected]

    truth_participation = (
        total_truth
        if truth_assay_participations is None
        else int(truth_assay_participations)
    )
    predicted_participation = (
        selected_count
        if predicted_assay_participations is None
        else int(predicted_assay_participations)
    )
    true_positive_participation = (
        true_positive
        if true_positive_assay_participations is None
        else int(true_positive_assay_participations)
    )
    if min(
        truth_participation,
        predicted_participation,
        true_positive_participation,
        int(ambiguous_assay_participation_hypotheses),
        int(candidate_assay_pair_alternatives),
        int(assay_participation_point_abstentions),
    ) < 0:
        raise ValueError("Assay-participation counts must be nonnegative")
    if true_positive_participation > min(
        truth_participation, predicted_participation
    ):
        raise ValueError(
            "Correct assay participations exceed truth or prediction counts"
        )
    if sentinel_null_feature_opportunities is not None:
        sentinel_nulls = int(sentinel_null_feature_opportunities)
    if sentinel_false_positive_features is not None:
        sentinel_false_positives = int(sentinel_false_positive_features)
    if sentinel_nulls < 0 or sentinel_false_positives < 0:
        raise ValueError("Sentinel counts must be nonnegative")
    if sentinel_false_positives > sentinel_nulls:
        raise ValueError(
            "Sentinel false positives cannot exceed sentinel opportunities"
        )

    return {
        "metric_schema_version": SCHEMA,
        "truth_hyperedges": total_truth,
        "selected_hyperedges": selected_count,
        "true_positive_hyperedges": true_positive,
        "false_positive_hyperedges": false_positive,
        "false_negative_hyperedges": false_negative,
        "shared_state_hyperedge_precision": _safe_ratio(
            true_positive, selected_count
        ),
        "shared_state_hyperedge_recall": _safe_ratio(
            true_positive, total_truth
        ),
        "truth_participating_assays": truth_participation,
        "predicted_participating_assays": predicted_participation,
        "correctly_recovered_assay_participations": (
            true_positive_participation
        ),
        "assay_participation_false_positives": (
            predicted_participation - true_positive_participation
        ),
        "assay_participation_false_negatives": (
            truth_participation - true_positive_participation
        ),
        "assay_participation_precision": _safe_ratio(
            true_positive_participation, predicted_participation
        ),
        "assay_participation_recall": _safe_ratio(
            true_positive_participation, truth_participation
        ),
        "ambiguous_assay_participation_hypotheses": int(
            ambiguous_assay_participation_hypotheses
        ),
        "candidate_assay_pair_alternatives": int(
            candidate_assay_pair_alternatives
        ),
        "assay_participation_point_abstentions": int(
            assay_participation_point_abstentions
        ),
        # Backward-compatible count name used in earlier v4.33 reports.
        "participating_assays": predicted_participation,
        "sentinel_null_feature_opportunities": sentinel_nulls,
        "sentinel_false_positive_features": sentinel_false_positives,
        "sentinel_false_positive_hypotheses": sentinel_hypotheses,
        "sentinel_false_positive_rate": _safe_ratio(
            sentinel_false_positives, sentinel_nulls
        ),
        "sentinel_false_discovery_proportion": _safe_ratio(
            sentinel_hypotheses, selected_count
        ),
        "equivalence_class_coverage": _safe_ratio(
            true_positive, total_truth
        ),
        "equivalence_class_coverage_precision": _safe_ratio(
            true_positive, selected_count
        ),
        "average_ambiguity_set_size": (
            float(selected_ambiguity.mean())
            if len(selected_ambiguity)
            else 0.0
        ),
        "maximum_ambiguity_set_size": (
            int(selected_ambiguity.max()) if len(selected_ambiguity) else 0
        ),
        "candidate_pair_alternatives": int(selected_ambiguity.sum()),
        "candidate_pair_alternatives_expanded_as_edges": False,
    }


def exact_cross_assay_stress(
    *,
    cross_truth_pairs: int,
    direction_truths: int,
    pair_predictions: int = 0,
    pair_true_positives: int = 0,
    direction_predictions: int = 0,
    direction_true_positives: int = 0,
    interpretation: str,
) -> dict[str, Any]:
    """Score exact pair/direction claims without making them a success gate."""

    counts = {
        "cross_truth_pairs": cross_truth_pairs,
        "direction_truths": direction_truths,
        "pair_predictions": pair_predictions,
        "pair_true_positives": pair_true_positives,
        "direction_predictions": direction_predictions,
        "direction_true_positives": direction_true_positives,
    }
    if any(int(value) != value or value < 0 for value in counts.values()):
        raise ValueError("Stress-test counts must be nonnegative integers")
    counts = {key: int(value) for key, value in counts.items()}
    if counts["pair_true_positives"] > min(
        counts["cross_truth_pairs"], counts["pair_predictions"]
    ):
        raise ValueError("Exact-pair true positives exceed available pairs")
    if counts["direction_true_positives"] > min(
        counts["direction_truths"], counts["direction_predictions"]
    ):
        raise ValueError("Direction true positives exceed available directions")

    pair_false_positives = (
        counts["pair_predictions"] - counts["pair_true_positives"]
    )
    direction_false_positives = (
        counts["direction_predictions"]
        - counts["direction_true_positives"]
    )
    # This is a claim-count coverage proxy.  A future paired opportunity table
    # can replace it with opportunity-level abstention without changing the
    # precision/recall definitions.
    pair_claim_coverage = min(
        _safe_ratio(counts["pair_predictions"], counts["cross_truth_pairs"]),
        1.0,
    )
    return {
        "metric_schema_version": SCHEMA,
        "label": STRESS_LABEL,
        "included_in_reliability_gate": False,
        "exact_cross_assay_truth_pairs": counts["cross_truth_pairs"],
        "exact_cross_assay_pair_predictions": counts["pair_predictions"],
        "exact_cross_assay_pair_true_positives": counts[
            "pair_true_positives"
        ],
        "exact_cross_assay_pair_false_positives": pair_false_positives,
        "exact_cross_assay_pair_precision": _safe_ratio(
            counts["pair_true_positives"], counts["pair_predictions"]
        ),
        "exact_cross_assay_pair_recall": _safe_ratio(
            counts["pair_true_positives"], counts["cross_truth_pairs"]
        ),
        "exact_cross_assay_pair_abstention_rate": float(
            1.0 - pair_claim_coverage
        ),
        "exact_cross_assay_pair_abstention_definition": (
            "one_minus_capped_prediction_count_over_truth_pair_count"
        ),
        "cross_assay_direction_identifiable_truths": counts[
            "direction_truths"
        ],
        "cross_assay_direction_predictions": counts[
            "direction_predictions"
        ],
        "cross_assay_direction_true_positives": counts[
            "direction_true_positives"
        ],
        "cross_assay_direction_false_positives": direction_false_positives,
        "cross_assay_direction_precision": _safe_ratio(
            counts["direction_true_positives"],
            counts["direction_predictions"],
        ),
        "cross_assay_direction_recovery": (
            _safe_ratio(
                counts["direction_true_positives"], counts["direction_truths"]
            )
            if counts["direction_truths"]
            else None
        ),
        "direction_is_not_defined_for_shared_state_truth": (
            counts["direction_truths"] == 0
        ),
        "interpretation": str(interpretation),
    }


def metric_families(
    coverage: Mapping[str, Any], stress: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return the four explicitly named benchmark metric families."""

    shared_keys = (
        "truth_hyperedges",
        "selected_hyperedges",
        "true_positive_hyperedges",
        "false_positive_hyperedges",
        "false_negative_hyperedges",
        "shared_state_hyperedge_precision",
        "shared_state_hyperedge_recall",
    )
    participation_keys = (
        "truth_participating_assays",
        "predicted_participating_assays",
        "correctly_recovered_assay_participations",
        "assay_participation_false_positives",
        "assay_participation_false_negatives",
        "assay_participation_precision",
        "assay_participation_recall",
        "ambiguous_assay_participation_hypotheses",
        "candidate_assay_pair_alternatives",
        "assay_participation_point_abstentions",
        "sentinel_null_feature_opportunities",
        "sentinel_false_positive_features",
        "sentinel_false_positive_hypotheses",
        "sentinel_false_positive_rate",
        "sentinel_false_discovery_proportion",
    )
    equivalence_keys = (
        "equivalence_class_coverage",
        "equivalence_class_coverage_precision",
        "average_ambiguity_set_size",
        "maximum_ambiguity_set_size",
        "candidate_pair_alternatives",
        "candidate_pair_alternatives_expanded_as_edges",
    )
    return {
        "shared_state_hyperedge": {key: coverage[key] for key in shared_keys},
        "assay_participation_and_sentinel_false_positive": {
            key: coverage[key] for key in participation_keys
        },
        "equivalence_class": {
            key: coverage[key] for key in equivalence_keys
        },
        "identifiability_stress_test": dict(stress),
    }
