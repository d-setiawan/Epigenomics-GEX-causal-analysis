#!/usr/bin/env python3
"""Verify and combine disjoint cached cross-assay benchmark panels.

This campaign layer consumes only hash-verified panel summaries and their
world-level metric tables.  It never opens scorer partitions and never
expands ambiguity-set alternatives into separate edge predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import summarize_cross_assay_benchmark_panel_v4 as panel  # noqa: E402


SCHEMA = "ca_boss_v4.cross_assay_benchmark_campaign_summary.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate_campaign_metrics(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Micro-aggregate the four metric families from verified panels."""

    if not summaries:
        raise ValueError("A campaign must contain at least one panel")
    metric_sets = [summary["benchmark_metrics"] for summary in summaries]
    hyperedges = [metrics["shared_state_hyperedge"] for metrics in metric_sets]
    participation = [
        metrics["assay_participation_and_sentinel_false_positive"]
        for metrics in metric_sets
    ]
    equivalence = [metrics["equivalence_class"] for metrics in metric_sets]
    stress_sets = [
        metrics["identifiability_stress_test"] for metrics in metric_sets
    ]

    def total(rows: Sequence[Mapping[str, Any]], key: str) -> int:
        return sum(int(row[key]) for row in rows)

    truth_hyperedges = total(hyperedges, "truth_hyperedges")
    selected_hyperedges = total(hyperedges, "selected_hyperedges")
    true_positive_hyperedges = total(hyperedges, "true_positive_hyperedges")
    false_positive_hyperedges = total(hyperedges, "false_positive_hyperedges")
    false_negative_hyperedges = total(hyperedges, "false_negative_hyperedges")
    if selected_hyperedges != true_positive_hyperedges + false_positive_hyperedges:
        raise ValueError("Panel hyperedge counts are internally inconsistent")
    if truth_hyperedges != true_positive_hyperedges + false_negative_hyperedges:
        raise ValueError("Panel truth counts are internally inconsistent")

    truth_participation = total(participation, "truth_participating_assays")
    predicted_participation = total(
        participation, "predicted_participating_assays"
    )
    correct_participation = total(
        participation, "correctly_recovered_assay_participations"
    )
    sentinel_opportunities = total(
        participation, "sentinel_null_feature_opportunities"
    )
    sentinel_features = total(
        participation, "sentinel_false_positive_features"
    )
    sentinel_hypotheses = total(
        participation, "sentinel_false_positive_hypotheses"
    )
    candidate_pair_alternatives = total(
        equivalence, "candidate_pair_alternatives"
    )

    coverage = {
        "metric_schema_version": benchmark.SCHEMA,
        "truth_hyperedges": truth_hyperedges,
        "selected_hyperedges": selected_hyperedges,
        "true_positive_hyperedges": true_positive_hyperedges,
        "false_positive_hyperedges": false_positive_hyperedges,
        "false_negative_hyperedges": false_negative_hyperedges,
        "shared_state_hyperedge_precision": _safe_ratio(
            true_positive_hyperedges, selected_hyperedges
        ),
        "shared_state_hyperedge_recall": _safe_ratio(
            true_positive_hyperedges, truth_hyperedges
        ),
        "truth_participating_assays": truth_participation,
        "predicted_participating_assays": predicted_participation,
        "correctly_recovered_assay_participations": correct_participation,
        "assay_participation_false_positives": (
            predicted_participation - correct_participation
        ),
        "assay_participation_false_negatives": (
            truth_participation - correct_participation
        ),
        "assay_participation_precision": _safe_ratio(
            correct_participation, predicted_participation
        ),
        "assay_participation_recall": _safe_ratio(
            correct_participation, truth_participation
        ),
        "ambiguous_assay_participation_hypotheses": total(
            participation, "ambiguous_assay_participation_hypotheses"
        ),
        "candidate_assay_pair_alternatives": total(
            participation, "candidate_assay_pair_alternatives"
        ),
        "assay_participation_point_abstentions": total(
            participation, "assay_participation_point_abstentions"
        ),
        "participating_assays": predicted_participation,
        "sentinel_null_feature_opportunities": sentinel_opportunities,
        "sentinel_false_positive_features": sentinel_features,
        "sentinel_false_positive_hypotheses": sentinel_hypotheses,
        "sentinel_false_positive_rate": _safe_ratio(
            sentinel_features, sentinel_opportunities
        ),
        "sentinel_false_discovery_proportion": _safe_ratio(
            sentinel_hypotheses, selected_hyperedges
        ),
        "equivalence_class_coverage": _safe_ratio(
            true_positive_hyperedges, truth_hyperedges
        ),
        "equivalence_class_coverage_precision": _safe_ratio(
            true_positive_hyperedges, selected_hyperedges
        ),
        "average_ambiguity_set_size": _safe_ratio(
            candidate_pair_alternatives, selected_hyperedges
        ),
        "maximum_ambiguity_set_size": max(
            int(row["maximum_ambiguity_set_size"])
            for row in equivalence
        ),
        "candidate_pair_alternatives": candidate_pair_alternatives,
        "candidate_pair_alternatives_expanded_as_edges": False,
    }
    stress = benchmark.exact_cross_assay_stress(
        cross_truth_pairs=total(
            stress_sets, "exact_cross_assay_truth_pairs"
        ),
        pair_predictions=total(
            stress_sets, "exact_cross_assay_pair_predictions"
        ),
        pair_true_positives=total(
            stress_sets, "exact_cross_assay_pair_true_positives"
        ),
        direction_truths=total(
            stress_sets, "cross_assay_direction_identifiable_truths"
        ),
        direction_predictions=total(
            stress_sets, "cross_assay_direction_predictions"
        ),
        direction_true_positives=total(
            stress_sets, "cross_assay_direction_true_positives"
        ),
        interpretation=(
            "campaign_micro_average_identifiability_stress_test; exact pair "
            "and direction recovery are diagnostic and not reliability gates"
        ),
    )
    return benchmark.metric_families(coverage, stress)


def _summary_path(path: Path) -> Path:
    resolved = path.expanduser().absolute().resolve(strict=True)
    return resolved / "panel_summary.json" if resolved.is_dir() else resolved


def summarize_campaign(
    *, panel_summaries: Sequence[Path], out_dir: Path, campaign_label: str
) -> dict[str, Any]:
    """Verify disjoint panel outputs and freeze one campaign summary."""

    if not panel_summaries:
        raise ValueError("At least one panel summary is required")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")

    summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    panel_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    score_schemas: set[str] = set()
    other_block_truth_flags: set[bool] = set()
    other_block_supervised_truth_flags: set[bool] = set()
    seen_paths: set[Path] = set()
    for supplied in panel_summaries:
        summary_path = _summary_path(supplied)
        if summary_path in seen_paths:
            raise ValueError(f"Duplicate panel summary: {summary_path}")
        seen_paths.add(summary_path)
        summary = panel._verified_json(
            summary_path, payload_hash_field="summary_payload_sha256"
        )
        if summary.get("schema_version") != panel.SCHEMA:
            raise ValueError(f"Unexpected panel schema: {summary_path}")
        if summary.get("benchmark_metric_schema_version") != benchmark.SCHEMA:
            raise ValueError(f"Unexpected metric schema: {summary_path}")
        if summary.get("cached_inputs_only") is not True:
            raise ValueError(f"Panel is not cached-only: {summary_path}")
        if summary.get("new_worlds_or_datasets_created") is not False:
            raise ValueError(f"Panel created new data: {summary_path}")
        if summary.get("fresh_locked_certification") is not False:
            raise ValueError(f"Campaign accepts retrospective panels only: {summary_path}")
        if (
            summary.get(
                "truth_opened_only_after_all_panel_phase_a_artifacts_frozen"
            )
            is not True
        ):
            raise ValueError(f"Panel truth boundary is not certified: {summary_path}")

        world_path = summary_path.parent / "panel_world_metrics.tsv"
        expected_world_hash = summary.get("output_sha256", {}).get(
            "panel_world_metrics.tsv"
        )
        if world_path.is_symlink() or not world_path.is_file():
            raise FileNotFoundError(world_path)
        if _sha256(world_path) != str(expected_world_hash):
            raise ValueError(f"Panel world table hash mismatch: {world_path}")
        frame = pd.read_csv(world_path, sep="\t", keep_default_na=False)
        if len(frame) != int(summary["panel_world_count"]):
            raise ValueError(f"Panel world count mismatch: {summary_path}")
        hyper = summary["benchmark_metrics"]["shared_state_hyperedge"]
        for column, metric in (
            ("truth_hyperedges", "truth_hyperedges"),
            ("selected_hyperedges", "selected_hyperedges"),
            ("true_positive_hyperedges", "true_positive_hyperedges"),
            ("false_positive_hyperedges", "false_positive_hyperedges"),
            ("false_negative_hyperedges", "false_negative_hyperedges"),
        ):
            if int(pd.to_numeric(frame[column], errors="raise").sum()) != int(
                hyper[metric]
            ):
                raise ValueError(
                    f"Panel metric/table mismatch for {column}: {summary_path}"
                )

        score_schemas.add(str(summary["score_schema_version"]))
        other_block_truth_flags.add(
            bool(
                summary.get(
                    "other_block_simulation_truth_used_for_null_calibration",
                    False,
                )
            )
        )
        other_block_supervised_truth_flags.add(
            bool(
                summary.get(
                    "other_block_simulation_truth_used_for_supervised_training",
                    False,
                )
            )
        )
        if (
            summary.get(
                "target_block_truth_used_for_fit_or_selection", False
            )
            is not False
        ):
            raise ValueError(
                f"Panel used target-block truth for selection: {summary_path}"
            )
        summaries.append(summary)
        frames.append(frame.assign(source_panel=str(summary_path)))
        panel_rows.append(
            {
                "panel_summary": str(summary_path),
                "panel_world_count": int(summary["panel_world_count"]),
                "selection_seed": int(summary["panel_selection_seed"]),
                "summary_payload_sha256": str(
                    summary["summary_payload_sha256"]
                ),
            }
        )
        input_hashes[str(summary_path)] = {
            "panel_summary.json": _sha256(summary_path),
            "panel_world_metrics.tsv": _sha256(world_path),
        }

    if len(score_schemas) != 1:
        raise ValueError(
            f"A campaign cannot mix score schemas: {sorted(score_schemas)}"
        )
    if len(other_block_truth_flags) != 1:
        raise ValueError("Panels disagree about cross-fit calibration truth use")
    if len(other_block_supervised_truth_flags) != 1:
        raise ValueError("Panels disagree about supervised cross-fit truth use")
    combined = pd.concat(frames, ignore_index=True)
    if combined["world_key"].astype(str).duplicated().any():
        duplicates = sorted(
            combined.loc[
                combined["world_key"].astype(str).duplicated(keep=False),
                "world_key",
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Campaign panels overlap on worlds: {duplicates}")

    metrics = aggregate_campaign_metrics(summaries)
    shared = metrics["shared_state_hyperedge"]
    sentinel = metrics[
        "assay_participation_and_sentinel_false_positive"
    ]
    stress = metrics["identifiability_stress_test"]
    reliability_metrics = {
        "shared_state_hyperedge_precision_is_one": (
            shared["shared_state_hyperedge_precision"] == 1.0
        ),
        "sentinel_false_positive_rate_is_zero": (
            sentinel["sentinel_false_positive_rate"] == 0.0
        ),
        "candidate_alternatives_never_expanded": True,
        "exact_pair_and_direction_stress_not_a_gate": (
            stress["included_in_reliability_gate"] is False
        ),
    }

    out_dir.mkdir(parents=True)
    world_output = out_dir / "campaign_world_metrics.tsv"
    panel_output = out_dir / "campaign_panels.tsv"
    combined.sort_values(["block", "seed"]).to_csv(
        world_output, sep="\t", index=False
    )
    pd.DataFrame(panel_rows).to_csv(panel_output, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA,
        "campaign_label": str(campaign_label),
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "score_schema_version": next(iter(score_schemas)),
        "panel_count": len(summaries),
        "campaign_world_count": len(combined),
        "cross_panel_worlds_are_disjoint": True,
        "cached_inputs_only": True,
        "new_worlds_or_datasets_created": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "other_block_simulation_truth_used_for_null_calibration": next(
            iter(other_block_truth_flags)
        ),
        "other_block_simulation_truth_used_for_supervised_training": next(
            iter(other_block_supervised_truth_flags)
        ),
        "target_block_truth_used_for_fit_or_selection": False,
        "candidate_alternatives_expanded_as_edges": False,
        "benchmark_metrics": metrics,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "exact_pair_and_direction_interpretation": (
            "identifiability stress test, not an expected success criterion"
        ),
        "input_sha256": input_hashes,
        "source_sha256": {
            Path(__file__).name: _sha256(Path(__file__).resolve()),
            "summarize_cross_assay_benchmark_panel_v4.py": _sha256(
                SCRIPT_DIR / "summarize_cross_assay_benchmark_panel_v4.py"
            ),
            "cross_assay_benchmark_metrics_v4.py": _sha256(
                SCRIPT_DIR / "cross_assay_benchmark_metrics_v4.py"
            ),
        },
        "output_sha256": {
            "campaign_world_metrics.tsv": _sha256(world_output),
            "campaign_panels.tsv": _sha256(panel_output),
        },
    }
    result["summary_payload_sha256"] = _canonical_sha256(result)
    (out_dir / "campaign_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--campaign-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = summarize_campaign(
        panel_summaries=args.panel_summary,
        out_dir=args.out_dir,
        campaign_label=args.campaign_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
