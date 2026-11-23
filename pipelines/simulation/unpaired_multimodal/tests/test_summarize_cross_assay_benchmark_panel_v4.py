from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import summarize_cross_assay_benchmark_panel_v4 as panel  # noqa: E402


def _score_frame(*, matched: bool, ambiguity: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis_selected": True,
                "matched_to_unique_truth_pair": matched,
                "contains_sentinel_candidate": False,
                "candidate_pair_count": ambiguity,
                "sentinel_candidate_feature_count": 0,
                "sentinel_null_feature_opportunities": 12,
            }
        ]
    )


def _summary(*, matched: bool, ambiguity: int) -> dict:
    frame = _score_frame(matched=matched, ambiguity=ambiguity)
    coverage = benchmark.coverage_metrics(
        frame,
        total_truth=3,
        truth_assay_participations=6,
        predicted_assay_participations=2,
        true_positive_assay_participations=2 if matched else 0,
        sentinel_null_feature_opportunities=12,
        sentinel_false_positive_features=0,
    )
    stress = benchmark.exact_cross_assay_stress(
        cross_truth_pairs=3,
        pair_predictions=int(ambiguity == 1),
        pair_true_positives=int(matched and ambiguity == 1),
        direction_truths=0,
        interpretation="unit_test",
    )
    return {"benchmark_metrics": benchmark.metric_families(coverage, stress)}


def test_panel_micro_aggregation_reports_all_requested_families() -> None:
    summaries = [
        _summary(matched=True, ambiguity=1),
        _summary(matched=False, ambiguity=3),
    ]
    frames = [
        _score_frame(matched=True, ambiguity=1),
        _score_frame(matched=False, ambiguity=3),
    ]

    metrics = panel.aggregate_panel_metrics(summaries, frames)

    assert set(metrics) == {
        "shared_state_hyperedge",
        "assay_participation_and_sentinel_false_positive",
        "equivalence_class",
        "identifiability_stress_test",
    }
    shared = metrics["shared_state_hyperedge"]
    assert shared["truth_hyperedges"] == 6
    assert shared["selected_hyperedges"] == 2
    assert shared["shared_state_hyperedge_precision"] == 0.5
    assert shared["shared_state_hyperedge_recall"] == 1.0 / 6.0

    participation = metrics[
        "assay_participation_and_sentinel_false_positive"
    ]
    assert participation["assay_participation_precision"] == 0.5
    assert participation["assay_participation_recall"] == 1.0 / 6.0
    assert participation["sentinel_false_positive_rate"] == 0.0

    equivalence = metrics["equivalence_class"]
    assert equivalence["equivalence_class_coverage"] == 1.0 / 6.0
    assert equivalence["average_ambiguity_set_size"] == 2.0

    stress = metrics["identifiability_stress_test"]
    assert stress["exact_cross_assay_truth_pairs"] == 6
    assert stress["exact_cross_assay_pair_predictions"] == 1
    assert stress["exact_cross_assay_pair_true_positives"] == 1
    assert stress["included_in_reliability_gate"] is False
    assert stress["cross_assay_direction_predictions"] == 0


def test_panel_aggregation_requires_one_table_per_summary() -> None:
    with pytest.raises(ValueError, match="one score table"):
        panel.aggregate_panel_metrics(
            [_summary(matched=True, ambiguity=1)], []
        )


def test_verified_json_rejects_payload_tampering(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    original = {"value": 1}
    original["summary_payload_sha256"] = panel._canonical_sha256(original)
    path.write_text(json.dumps(original), encoding="utf-8")

    verified = panel._verified_json(
        path, payload_hash_field="summary_payload_sha256"
    )
    assert verified["value"] == 1

    original["value"] = 2
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="Payload hash mismatch"):
        panel._verified_json(
            path, payload_hash_field="summary_payload_sha256"
        )
