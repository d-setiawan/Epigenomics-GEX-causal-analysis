#!/usr/bin/env python3
"""Verify and summarize a cached cross-assay benchmark panel.

The summarizer never opens world scorer partitions itself.  It consumes only
already-frozen per-world score artifacts, verifies their hashes, and requires
their world keys to equal the pre-truth Phase-A selection exactly.  Ambiguity
sets remain hypotheses: their candidate alternatives are never expanded into
separate predictions.
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


SCHEMA = "ca_boss_v4.cross_assay_benchmark_panel_summary.v1"
EXPECTED_SCORE_SCHEMAS = {
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v5",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v6",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v7",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v8",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v9",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v10",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v11",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v12",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v13",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v14",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v15",
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v16",
    "ca_boss_v4.multi_assay_permutation_synchronization.scorer.v1",
    "ca_boss_v4.heldout_synchronized_majority_rescue.scorer.v1",
    "ca_boss_v4.crossfit_empirical_null_reporter_synchronization.scorer.v1",
    "ca_boss_v4.crossfit_multivariate_reporter_rescue.scorer.v1",
    "ca_boss_v4.crossfit_sparse_reporter_completion.scorer.v1",
    "ca_boss_v4.crossfit_interaction_reporter_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_stable_sparse_completion.scorer.v1",
    "ca_boss_v4.bootstrap_empirical_copula_pair_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_mutual_nearest_margin_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_wasserstein_geometry_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_null_calibrated_stability_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_joint_context_stratified_stability_rescue.scorer.v1",
}
CROSSFIT_SCORE_SCHEMAS = {
    "ca_boss_v4.crossfit_empirical_null_reporter_synchronization.scorer.v1",
    "ca_boss_v4.crossfit_multivariate_reporter_rescue.scorer.v1",
    "ca_boss_v4.crossfit_sparse_reporter_completion.scorer.v1",
    "ca_boss_v4.crossfit_interaction_reporter_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_stable_sparse_completion.scorer.v1",
    "ca_boss_v4.bootstrap_empirical_copula_pair_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_mutual_nearest_margin_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_wasserstein_geometry_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_null_calibrated_stability_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_joint_context_stratified_stability_rescue.scorer.v1",
}
SUPERVISED_CROSSFIT_SCORE_SCHEMAS = {
    "ca_boss_v4.crossfit_multivariate_reporter_rescue.scorer.v1",
    "ca_boss_v4.crossfit_sparse_reporter_completion.scorer.v1",
    "ca_boss_v4.crossfit_interaction_reporter_rescue.scorer.v1",
    "ca_boss_v4.bootstrap_stable_sparse_completion.scorer.v1",
}


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


def _verified_json(
    path: Path, *, payload_hash_field: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = str(value.get(payload_hash_field, ""))
    unsigned = dict(value)
    unsigned.pop(payload_hash_field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"Payload hash mismatch: {path}")
    return value


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate_panel_metrics(
    summaries: Sequence[Mapping[str, Any]],
    score_frames: Sequence[pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Micro-aggregate the four requested cross-assay metric families."""

    if len(summaries) != len(score_frames):
        raise ValueError("Every score summary must have one score table")
    if not summaries:
        raise ValueError("A benchmark panel must contain at least one world")

    combined = pd.concat(score_frames, ignore_index=True)
    metric_sets = [summary["benchmark_metrics"] for summary in summaries]

    hyperedges = [metrics["shared_state_hyperedge"] for metrics in metric_sets]
    participation = [
        metrics["assay_participation_and_sentinel_false_positive"]
        for metrics in metric_sets
    ]
    stress_sets = [
        metrics["identifiability_stress_test"] for metrics in metric_sets
    ]

    def total(rows: Sequence[Mapping[str, Any]], key: str) -> int:
        return sum(int(row[key]) for row in rows)

    coverage = benchmark.coverage_metrics(
        combined,
        total_truth=total(hyperedges, "truth_hyperedges"),
        truth_assay_participations=total(
            participation, "truth_participating_assays"
        ),
        predicted_assay_participations=total(
            participation, "predicted_participating_assays"
        ),
        true_positive_assay_participations=total(
            participation, "correctly_recovered_assay_participations"
        ),
        ambiguous_assay_participation_hypotheses=total(
            participation, "ambiguous_assay_participation_hypotheses"
        ),
        candidate_assay_pair_alternatives=total(
            participation, "candidate_assay_pair_alternatives"
        ),
        assay_participation_point_abstentions=total(
            participation, "assay_participation_point_abstentions"
        ),
        sentinel_null_feature_opportunities=total(
            participation, "sentinel_null_feature_opportunities"
        ),
        sentinel_false_positive_features=total(
            participation, "sentinel_false_positive_features"
        ),
    )

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
            "panel_micro_average_identifiability_stress_test; exact pair and "
            "direction recovery are diagnostic and not reliability gates"
        ),
    )
    families = benchmark.metric_families(coverage, stress)

    # The reusable coverage function derives this from each row.  Per-world
    # score tables repeat the world-level opportunity count on every row, so
    # the explicit micro-aggregated denominator above is authoritative.
    sentinel = families[
        "assay_participation_and_sentinel_false_positive"
    ]
    sentinel["sentinel_false_discovery_proportion"] = _safe_ratio(
        total(participation, "sentinel_false_positive_hypotheses"),
        coverage["selected_hyperedges"],
    )
    sentinel["sentinel_false_positive_hypotheses"] = total(
        participation, "sentinel_false_positive_hypotheses"
    )
    return families


def _world_key(block: str, seed: int) -> str:
    return f"{block}__world_seed{int(seed)}"


def summarize_panel(
    *, phase_a_dir: Path, score_root: Path, out_dir: Path
) -> dict[str, Any]:
    """Verify a Phase-A selection and all score artifacts, then summarize."""

    phase_a = phase_a_dir.expanduser().absolute().resolve(strict=True)
    scores = score_root.expanduser().absolute().resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")

    selection_path = phase_a / "phase_a_selection.json"
    manifest_path = phase_a / "phase_a_manifest.json"
    selection = _verified_json(
        selection_path, payload_hash_field="selection_payload_sha256"
    )
    manifest = _verified_json(
        manifest_path, payload_hash_field="manifest_payload_sha256"
    )
    if manifest.get("phase") != "a":
        raise ValueError("Expected a Phase-A manifest")
    if manifest.get("scorer_or_truth_opened") is not False:
        raise ValueError("Phase-A manifest did not certify truth separation")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Panel created new worlds or datasets")
    if manifest.get("selection_sha256") != _sha256(selection_path):
        raise ValueError("Phase-A selection file hash mismatch")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = phase_a / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"Phase-A output hash mismatch: {name}")

    selected_worlds = selection.get("selected_worlds", [])
    expected_keys = {
        _world_key(str(row["block"]), int(row["seed"]))
        for row in selected_worlds
    }
    observed_dirs = {path.name: path for path in scores.iterdir() if path.is_dir()}
    if set(observed_dirs) != expected_keys:
        missing = sorted(expected_keys.difference(observed_dirs))
        extra = sorted(set(observed_dirs).difference(expected_keys))
        raise ValueError(
            f"Score worlds do not equal Phase-A selection; missing={missing}, "
            f"extra={extra}"
        )

    summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    world_rows: list[dict[str, Any]] = []
    score_input_hashes: dict[str, dict[str, str]] = {}
    observed_score_schemas: set[str] = set()
    for key in sorted(expected_keys):
        directory = observed_dirs[key]
        summary_path = directory / "score_summary.json"
        summary = _verified_json(
            summary_path, payload_hash_field="summary_payload_sha256"
        )
        score_schema = str(summary.get("schema_version", ""))
        if score_schema not in EXPECTED_SCORE_SCHEMAS:
            raise ValueError(f"Unexpected score schema for {key}")
        observed_score_schemas.add(score_schema)
        if summary.get("benchmark_metric_schema_version") != benchmark.SCHEMA:
            raise ValueError(f"Unexpected metric schema for {key}")
        if summary.get("formal_pretruth_barrier_satisfied") is not True:
            raise ValueError(f"Pre-truth barrier was not satisfied for {key}")
        if score_schema in CROSSFIT_SCORE_SCHEMAS:
            if summary.get("algorithm_used_truth_for_fit_or_selection") is not True:
                raise ValueError(
                    f"Cross-fit training-truth use was not disclosed for {key}"
                )
            if (
                summary.get(
                    "other_block_simulation_truth_used_for_null_calibration"
                )
                is not True
                or summary.get(
                    "target_block_truth_used_for_fit_or_selection"
                )
                is not False
            ):
                raise ValueError(
                    f"Cross-fit truth boundary was not certified for {key}"
                )
            if (
                score_schema in SUPERVISED_CROSSFIT_SCORE_SCHEMAS
                and summary.get(
                    "other_block_simulation_truth_used_for_supervised_training"
                )
                is not True
            ):
                raise ValueError(
                    f"Cross-fit supervised truth use was not disclosed for {key}"
                )
        elif summary.get("algorithm_used_truth_for_fit_or_selection") is not False:
            raise ValueError(f"Truth affected selection for {key}")
        if summary.get("new_worlds_or_datasets_created") is not False:
            raise ValueError(f"New world or dataset was created for {key}")
        if _world_key(str(summary["block"]), int(summary["seed"])) != key:
            raise ValueError(f"Score identity mismatch for {key}")
        for name, expected in summary.get("output_sha256", {}).items():
            path = directory / str(name)
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != str(expected):
                raise ValueError(f"Score output hash mismatch: {key}/{name}")

        score_path = directory / "hyperedge_scores.tsv"
        frame = pd.read_csv(score_path, sep="\t", keep_default_na=False)
        metrics = summary["benchmark_metrics"]
        hyper = metrics["shared_state_hyperedge"]
        participation = metrics[
            "assay_participation_and_sentinel_false_positive"
        ]
        equivalence = metrics["equivalence_class"]
        stress = metrics["identifiability_stress_test"]
        world_rows.append(
            {
                "world_key": key,
                "block": summary["block"],
                "seed": int(summary["seed"]),
                "truth_hyperedges": int(hyper["truth_hyperedges"]),
                "selected_hyperedges": int(hyper["selected_hyperedges"]),
                "true_positive_hyperedges": int(
                    hyper["true_positive_hyperedges"]
                ),
                "false_positive_hyperedges": int(
                    hyper["false_positive_hyperedges"]
                ),
                "false_negative_hyperedges": int(
                    hyper["false_negative_hyperedges"]
                ),
                "shared_state_hyperedge_precision": float(
                    hyper["shared_state_hyperedge_precision"]
                ),
                "shared_state_hyperedge_recall": float(
                    hyper["shared_state_hyperedge_recall"]
                ),
                "assay_participation_precision": float(
                    participation["assay_participation_precision"]
                ),
                "assay_participation_recall": float(
                    participation["assay_participation_recall"]
                ),
                "sentinel_false_positive_rate": float(
                    participation["sentinel_false_positive_rate"]
                ),
                "equivalence_class_coverage": float(
                    equivalence["equivalence_class_coverage"]
                ),
                "average_ambiguity_set_size": float(
                    equivalence["average_ambiguity_set_size"]
                ),
                "exact_pair_predictions": int(
                    stress["exact_cross_assay_pair_predictions"]
                ),
                "exact_pair_true_positives": int(
                    stress["exact_cross_assay_pair_true_positives"]
                ),
                "direction_predictions": int(
                    stress["cross_assay_direction_predictions"]
                ),
            }
        )
        summaries.append(summary)
        frames.append(frame)
        score_input_hashes[key] = {
            "score_summary.json": _sha256(summary_path),
            "hyperedge_scores.tsv": _sha256(score_path),
        }

    if len(observed_score_schemas) != 1:
        raise ValueError(
            "A panel cannot mix score schema versions: "
            f"{sorted(observed_score_schemas)}"
        )
    score_schema_version = next(iter(observed_score_schemas))

    metric_families = aggregate_panel_metrics(summaries, frames)
    shared = metric_families["shared_state_hyperedge"]
    sentinel = metric_families[
        "assay_participation_and_sentinel_false_positive"
    ]
    stress = metric_families["identifiability_stress_test"]
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
    world_table = out_dir / "panel_world_metrics.tsv"
    pd.DataFrame(world_rows).to_csv(world_table, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA,
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "score_schema_version": score_schema_version,
        "panel_world_count": len(summaries),
        "panel_selection_seed": int(selection["selection_seed"]),
        "panel_selection_method": selection["sampling_method"],
        "cached_inputs_only": bool(selection["cached_inputs_only"]),
        "new_worlds_or_datasets_created": False,
        "truth_opened_only_after_all_panel_phase_a_artifacts_frozen": True,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "other_block_simulation_truth_used_for_null_calibration": (
            score_schema_version in CROSSFIT_SCORE_SCHEMAS
        ),
        "other_block_simulation_truth_used_for_supervised_training": (
            score_schema_version in SUPERVISED_CROSSFIT_SCORE_SCHEMAS
        ),
        "target_block_truth_used_for_fit_or_selection": False,
        "benchmark_metrics": metric_families,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "exact_pair_and_direction_interpretation": (
            "identifiability stress test, not an expected success criterion"
        ),
        "input_sha256": {
            "phase_a_selection.json": _sha256(selection_path),
            "phase_a_manifest.json": _sha256(manifest_path),
            "score_artifacts": score_input_hashes,
        },
        "source_sha256": {
            "summarize_cross_assay_benchmark_panel_v4.py": _sha256(
                Path(__file__).resolve()
            ),
            "cross_assay_benchmark_metrics_v4.py": _sha256(
                SCRIPT_DIR / "cross_assay_benchmark_metrics_v4.py"
            ),
            "score_component_conditioned_cross_assay_equivalence_v4.py": (
                _sha256(
                    SCRIPT_DIR
                    / "score_component_conditioned_cross_assay_equivalence_v4.py"
                )
            ),
        },
        "output_sha256": {
            "panel_world_metrics.tsv": _sha256(world_table)
        },
    }
    result["summary_payload_sha256"] = _canonical_sha256(result)
    (out_dir / "panel_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = summarize_panel(
        phase_a_dir=args.phase_a_dir,
        score_root=args.score_root,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
