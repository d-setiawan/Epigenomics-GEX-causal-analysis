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
import summarize_cross_assay_benchmark_campaign_v4 as campaign  # noqa: E402
import summarize_cross_assay_benchmark_panel_v4 as panel  # noqa: E402


def _metrics(*, matched: bool, ambiguity: int) -> dict:
    frame = pd.DataFrame(
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
    return benchmark.metric_families(coverage, stress)


def _write_panel(
    root: Path,
    *,
    name: str,
    world_key: str,
    block: str,
    seed: int,
    metrics: dict,
) -> Path:
    directory = root / name
    directory.mkdir()
    shared = metrics["shared_state_hyperedge"]
    world_table = directory / "panel_world_metrics.tsv"
    pd.DataFrame(
        [
            {
                "world_key": world_key,
                "block": block,
                "seed": seed,
                "truth_hyperedges": shared["truth_hyperedges"],
                "selected_hyperedges": shared["selected_hyperedges"],
                "true_positive_hyperedges": shared[
                    "true_positive_hyperedges"
                ],
                "false_positive_hyperedges": shared[
                    "false_positive_hyperedges"
                ],
                "false_negative_hyperedges": shared[
                    "false_negative_hyperedges"
                ],
            }
        ]
    ).to_csv(world_table, sep="\t", index=False)
    summary = {
        "schema_version": panel.SCHEMA,
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "score_schema_version": (
            "ca_boss_v4.component_conditioned_cross_assay_equivalence."
            "scorer.v13"
        ),
        "panel_world_count": 1,
        "panel_selection_seed": seed,
        "cached_inputs_only": True,
        "new_worlds_or_datasets_created": False,
        "fresh_locked_certification": False,
        "truth_opened_only_after_all_panel_phase_a_artifacts_frozen": True,
        "benchmark_metrics": metrics,
        "output_sha256": {
            "panel_world_metrics.tsv": campaign._sha256(world_table)
        },
    }
    summary["summary_payload_sha256"] = campaign._canonical_sha256(summary)
    summary_path = directory / "panel_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def test_campaign_verifies_panels_and_micro_aggregates_metrics(
    tmp_path: Path,
) -> None:
    first = _write_panel(
        tmp_path,
        name="panel_a",
        world_key="block_a__world_seed1",
        block="block_a",
        seed=1,
        metrics=_metrics(matched=True, ambiguity=1),
    )
    second = _write_panel(
        tmp_path,
        name="panel_b",
        world_key="block_b__world_seed2",
        block="block_b",
        seed=2,
        metrics=_metrics(matched=False, ambiguity=3),
    )

    result = campaign.summarize_campaign(
        panel_summaries=[first, second],
        out_dir=tmp_path / "campaign",
        campaign_label="unit_test",
    )

    assert result["campaign_world_count"] == 2
    assert result["cross_panel_worlds_are_disjoint"]
    shared = result["benchmark_metrics"]["shared_state_hyperedge"]
    assert shared["shared_state_hyperedge_precision"] == 0.5
    assert shared["shared_state_hyperedge_recall"] == 1.0 / 6.0
    equivalence = result["benchmark_metrics"]["equivalence_class"]
    assert equivalence["average_ambiguity_set_size"] == 2.0
    assert equivalence["maximum_ambiguity_set_size"] == 3
    stress = result["benchmark_metrics"]["identifiability_stress_test"]
    assert stress["exact_cross_assay_pair_true_positives"] == 1
    assert stress["included_in_reliability_gate"] is False


def test_campaign_rejects_overlapping_world_panels(tmp_path: Path) -> None:
    first = _write_panel(
        tmp_path,
        name="panel_a",
        world_key="block_a__world_seed1",
        block="block_a",
        seed=1,
        metrics=_metrics(matched=True, ambiguity=1),
    )
    second = _write_panel(
        tmp_path,
        name="panel_b",
        world_key="block_a__world_seed1",
        block="block_a",
        seed=1,
        metrics=_metrics(matched=False, ambiguity=3),
    )

    with pytest.raises(ValueError, match="overlap on worlds"):
        campaign.summarize_campaign(
            panel_summaries=[first, second],
            out_dir=tmp_path / "campaign",
            campaign_label="unit_test",
        )
