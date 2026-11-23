#!/usr/bin/env python3
"""Truth-separated scorer for CA-BOSS v4.49 sparse completion.

The scorer verifies the full v4.44--v4.48 chain, the truth-blind v4.49
completion evidence, and the target-safe nested v4.49 calibration before it
opens the target scorer partition.  Other-block S/N1/N2 roles are disclosed
supervised training and empirical-null inputs; target-block truth never
affects fitting or selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import crossfit_sparse_reporter_completion_v4 as calibration  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_crossfit_empirical_null_reporter_synchronization_v4 as v447_scorer  # noqa: E402
import score_crossfit_multivariate_reporter_rescue_v4 as v448_scorer  # noqa: E402
import score_heldout_synchronized_majority_rescue_v4 as rescue_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as combined_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.crossfit_sparse_reporter_completion.scorer.v1"
ALGORITHM = (
    "CA-BOSS v4.49 combined v4.44 plus nested sparse reporter completion"
)


def _verify_completion_evidence(
    evidence_dir: Path,
    *,
    bridge: Path,
    bridge_provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    directory, manifest = calibration._verified_completion_world(evidence_dir)
    if str(manifest.get("bridge_provenance_payload_sha256")) != str(
        bridge_provenance["provenance_payload_sha256"]
    ):
        raise ValueError("v4.49 evidence and bridge payload hashes disagree")
    expected_bridge_hash = calibration.synchronization._sha256(
        bridge / "provenance.json"
    )
    if str(manifest.get("input_sha256", {}).get("bridge_provenance.json")) != (
        expected_bridge_hash
    ):
        raise ValueError("v4.49 evidence and bridge file hashes disagree")
    if (
        str(manifest.get("block")) != str(bridge_provenance["block"])
        or int(manifest.get("seed")) != int(bridge_provenance["seed"])
    ):
        raise ValueError("v4.49 evidence and bridge identities disagree")
    return directory, manifest


def _verify_calibration(
    calibration_dir: Path,
    *,
    bridge_provenance: Mapping[str, Any],
    completion_evidence: Path,
    v448_calibration: Path,
    v448_manifest: Mapping[str, Any],
) -> tuple[
    Path,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    directory = calibration_dir.expanduser().absolute().resolve(strict=True)
    manifest_path = directory / "calibration_manifest.json"
    manifest = calibration._verified_payload(
        manifest_path, "manifest_payload_sha256"
    )
    if manifest.get("schema_version") != calibration.SCHEMA:
        raise ValueError("Unexpected v4.49 calibration schema")
    boundary = manifest.get("information_boundary", {})
    required_boundary = {
        "other_block_simulation_roles_used_for_supervised_training": True,
        "other_block_sentinel_null_roles_used_for_null_calibration": True,
        "target_block_scorer_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "target_safe_v448_used_only_as_target_baseline_union": True,
        "non_target_v448_selections_used_for_training": False,
        "known_sentinel_null_peaks_used_for_empirical_null": True,
        "modality_identity_used_as_model_feature": False,
        "candidate_feature_identity_used_as_model_feature": False,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "simulator_latent_state_used": False,
        "external_biological_features_or_priors_used": False,
    }
    for field, expected in required_boundary.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.49 information boundary mismatch: {field}")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.49 lacks a target-specific pre-truth barrier")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.49 lacks a formal pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.49 created a new world or dataset")
    if manifest.get("candidate_pairs_may_be_expanded_as_edges") is not False:
        raise ValueError("v4.49 allowed candidate-pair expansion")
    if manifest.get("within_assay_direct_structure_claimed") is not False:
        raise ValueError("v4.49 made a direct-structure claim")
    if manifest.get("exact_cross_assay_pair_claimed") is not False:
        raise ValueError("v4.49 made an exact cross-assay pair claim")
    if manifest.get("cross_assay_direction_claimed") is not False:
        raise ValueError("v4.49 made a direction claim")
    if list(manifest.get("model_feature_columns", [])) != list(
        calibration.FEATURE_COLUMNS
    ):
        raise ValueError("v4.49 model feature set changed")
    if [float(value) for value in manifest.get("model_c_values", [])] != list(
        calibration.MODEL_C_VALUES
    ):
        raise ValueError("v4.49 regularization stability guard changed")
    if int(manifest.get("polynomial_degree", -1)) != (
        calibration.POLYNOMIAL_DEGREE
    ):
        raise ValueError("v4.49 polynomial degree changed")
    if manifest.get("all_nested_models_converged") is not True:
        raise ValueError("v4.49 nested model did not converge")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"v4.49 calibration output hash mismatch: {name}")

    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = calibration._world_key(block, seed)
    if str(manifest.get("target_block")) != block:
        raise ValueError("v4.49 target block does not match bridge")
    if world_key not in set(map(str, manifest.get("target_world_keys", []))):
        raise ValueError("v4.49 calibration lacks the target world")

    evidence_inputs = manifest.get("input_sha256", {}).get(
        "completion_evidence_worlds", {}
    )
    expected_evidence = {
        "reporter_completion_evidence_manifest.json": combined_scorer._sha256(
            completion_evidence / "reporter_completion_evidence_manifest.json"
        ),
        "external_reporter_candidates.tsv": combined_scorer._sha256(
            completion_evidence / "external_reporter_candidates.tsv"
        ),
        "unresolved_reporter_pair_evidence.tsv": combined_scorer._sha256(
            completion_evidence / "unresolved_reporter_pair_evidence.tsv"
        ),
        "all_assay_synchronization_contexts.tsv": combined_scorer._sha256(
            completion_evidence / "all_assay_synchronization_contexts.tsv"
        ),
    }
    if evidence_inputs.get(world_key) != expected_evidence:
        raise ValueError("v4.49 completion-evidence binding mismatch")

    expected_v448 = {
        "calibration_manifest.json": combined_scorer._sha256(
            v448_calibration / "calibration_manifest.json"
        ),
        **{
            name: combined_scorer._sha256(v448_calibration / str(name))
            for name in v448_manifest.get("output_sha256", {})
        },
    }
    if manifest.get("input_sha256", {}).get(
        "target_safe_v448_calibration"
    ) != expected_v448:
        raise ValueError("v4.49 target-safe v4.48 binding mismatch")

    candidates = pd.read_csv(
        directory / "crossfit_sparse_completion_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    selected = candidates.loc[
        candidates["world_key"].astype(str).eq(world_key)
    ].copy()
    if len(selected) and (
        ~selected["block"].astype(str).eq(block)
        | ~pd.to_numeric(selected["seed"], errors="raise").eq(seed)
    ).any():
        raise ValueError("v4.49 candidate world identity mismatch")
    if len(selected) and not selected["hypothesis_selected"].map(
        combined_scorer._as_bool
    ).all():
        raise ValueError("v4.49 candidate table contains an unselected hypothesis")

    pair_audit = pd.read_csv(
        directory / "reporter_partner_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    world_pair_audit = pair_audit.loc[
        pair_audit["world_key"].astype(str).eq(world_key)
    ].copy()
    reporter_audit = pd.read_csv(
        directory / "nested_reporter_candidate_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    world_reporter_audit = reporter_audit.loc[
        reporter_audit["world_key"].astype(str).eq(world_key)
    ].copy()
    if world_reporter_audit.empty:
        raise ValueError("v4.49 reporter audit lacks the target world")
    return (
        directory,
        manifest,
        selected,
        world_pair_audit,
        world_reporter_audit,
    )


def normalize_predictions(
    core: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Adapt the v4.49 union to the unexpanded-hyperedge scorer."""

    incremental_ids = set(candidates["completion_hyperedge_id"].astype(str))
    incremental = candidates.rename(
        columns={"completion_hyperedge_id": "synchronized_hyperedge_id"}
    ).copy()
    predictions = combined_scorer.normalize_predictions(core, incremental)
    predictions.loc[
        predictions["hyperedge_id"].astype(str).isin(incremental_ids),
        "prediction_source",
    ] = "v449_crossfit_sparse_reporter_completion"
    return predictions


def score_combined(
    *,
    bridge_dir: Path,
    synchronization_dir: Path,
    rescue_dir: Path,
    relaxed_evidence_dir: Path,
    v447_calibration_dir: Path,
    v448_calibration_dir: Path,
    completion_evidence_dir: Path,
    calibration_dir: Path,
    scorer_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Verify every pre-truth stage, then score one target world."""

    bridge, bridge_provenance = core_scorer._verify_bridge(bridge_dir)
    synchronized, synchronization_manifest = (
        rescue_scorer.rescue.verify_synchronization(
            synchronization_dir, bridge, bridge_provenance
        )
    )
    rescue, rescue_manifest = rescue_scorer._verify_rescue(
        rescue_dir,
        bridge,
        bridge_provenance,
        synchronized,
        synchronization_manifest,
    )
    relaxed_evidence, relaxed_manifest = v447_scorer._verify_relaxed_evidence(
        relaxed_evidence_dir, bridge, bridge_provenance
    )
    v447_calibrated, v447_manifest, _ = v447_scorer._verify_calibration(
        v447_calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence=relaxed_evidence,
        rescue=rescue,
    )
    (
        v448_calibrated,
        v448_manifest,
        _,
        _,
    ) = v448_scorer._verify_calibration(
        v448_calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence=relaxed_evidence,
        rescue=rescue,
        v447_calibration=v447_calibrated,
        v447_manifest=v447_manifest,
    )
    completion_evidence, completion_manifest = _verify_completion_evidence(
        completion_evidence_dir,
        bridge=bridge,
        bridge_provenance=bridge_provenance,
    )
    (
        calibrated,
        calibration_manifest,
        candidates,
        pair_audit,
        reporter_audit,
    ) = _verify_calibration(
        calibration_dir,
        bridge_provenance=bridge_provenance,
        completion_evidence=completion_evidence,
        v448_calibration=v448_calibrated,
        v448_manifest=v448_manifest,
    )

    scorer = scorer_dir.expanduser().absolute().resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    truth_path = scorer / "truth_typed_edges.tsv"
    roles_path = scorer / "role_assignments.tsv"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (truth_path, roles_path)
    ):
        raise FileNotFoundError("Scorer truth files are incomplete")

    core = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    predictions = normalize_predictions(core, candidates)

    # Target truth is deliberately opened only after every upstream check.
    truths = core_scorer._truth_hyperedges(truth_path)
    roles = pd.read_csv(roles_path, sep="\t", keep_default_na=False)
    sentinel_features = set(
        roles.loc[roles["role"].astype(str).isin({"N1", "N2"}), "node"]
        .astype(str)
        .tolist()
    )
    scored, metrics = combined_scorer.score_predictions(
        predictions, truths, sentinel_features
    )
    shared = metrics["shared_state_hyperedge"]
    sentinel = metrics["assay_participation_and_sentinel_false_positive"]
    stress = metrics["identifiability_stress_test"]
    stress["interpretation"] = (
        "singleton_reporter_pairs_are_scored_diagnostically_but_v449_"
        "emits_crossfit_calibrated_shared_state_equivalence_hypotheses_not_"
        "direct_or_directed_edges"
    )
    reliability_metrics = {
        "shared_state_hyperedge_precision_is_one": (
            shared["shared_state_hyperedge_precision"] == 1.0
        ),
        "sentinel_false_positive_rate_is_zero": (
            sentinel["sentinel_false_positive_rate"] == 0.0
        ),
        "candidate_alternatives_never_expanded": bool(
            not predictions["candidate_pairs_may_be_expanded_as_edges"]
            .astype(bool)
            .any()
        ),
        "exact_pair_and_direction_stress_not_a_gate": (
            stress["included_in_reliability_gate"] is False
        ),
    }

    out_dir.mkdir(parents=True)
    score_path = out_dir / "hyperedge_scores.tsv"
    scored.to_csv(score_path, sep="\t", index=False)
    newly_selected = int(
        pair_audit["newly_selected_by_v449_sparse_completion"]
        .map(combined_scorer._as_bool)
        .sum()
    ) if len(pair_audit) else 0
    uniquely_selected_reporters = int(
        reporter_audit["uniquely_selected_reporter_within_assay"]
        .map(combined_scorer._as_bool)
        .sum()
    )
    summary = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "bridge_schema_version": bridge_provenance["schema_version"],
        "synchronization_schema_version": synchronization_manifest[
            "schema_version"
        ],
        "rescue_schema_version": rescue_manifest["schema_version"],
        "relaxed_evidence_schema_version": relaxed_manifest["schema_version"],
        "v447_calibration_schema_version": v447_manifest["schema_version"],
        "v448_calibration_schema_version": v448_manifest["schema_version"],
        "completion_evidence_schema_version": completion_manifest[
            "schema_version"
        ],
        "calibration_schema_version": calibration_manifest["schema_version"],
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "synchronization_manifest_payload_sha256": synchronization_manifest[
            "manifest_payload_sha256"
        ],
        "rescue_manifest_payload_sha256": rescue_manifest[
            "manifest_payload_sha256"
        ],
        "relaxed_evidence_manifest_payload_sha256": relaxed_manifest[
            "manifest_payload_sha256"
        ],
        "v447_calibration_manifest_payload_sha256": v447_manifest[
            "manifest_payload_sha256"
        ],
        "v448_calibration_manifest_payload_sha256": v448_manifest[
            "manifest_payload_sha256"
        ],
        "completion_evidence_manifest_payload_sha256": completion_manifest[
            "manifest_payload_sha256"
        ],
        "calibration_manifest_payload_sha256": calibration_manifest[
            "manifest_payload_sha256"
        ],
        "calibration_manifest_file_sha256": combined_scorer._sha256(
            calibrated / "calibration_manifest.json"
        ),
        "calibration_output_sha256": calibration_manifest["output_sha256"],
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "block": bridge_provenance["block"],
        "seed": bridge_provenance["seed"],
        "truth_opened_only_after_all_pretruth_hash_verification": True,
        "other_block_simulation_truth_used_for_supervised_training": True,
        "other_block_simulation_truth_used_for_null_calibration": True,
        "target_block_truth_used_for_fit_or_selection": False,
        "algorithm_used_truth_for_fit_or_selection": True,
        "algorithm_truth_use_scope": (
            "other_seed_blocks_only_for_nested_reporter_role_training_and_"
            "out_of_fold_sentinel_null_calibration"
        ),
        "formal_pretruth_barrier_satisfied": True,
        "eligible_for_target_block_crossfit_evaluation": True,
        "eligible_for_fresh_locked_certification": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "selected_v449_union_hypotheses": len(candidates),
        "newly_selected_v449_assay_pairs": newly_selected,
        "uniquely_selected_sparse_reporters": uniquely_selected_reporters,
        "null_probability_upper_envelope_by_c": calibration_manifest[
            "null_probability_upper_envelope_by_c"
        ],
        "benchmark_metrics": metrics,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "source_sha256": {
            Path(__file__).name: combined_scorer._sha256(
                Path(__file__).resolve()
            )
        },
        "output_sha256": {
            "hyperedge_scores.tsv": combined_scorer._sha256(score_path)
        },
        "truth_input_sha256": {
            "truth_typed_edges.tsv": combined_scorer._sha256(truth_path),
            "role_assignments.tsv": combined_scorer._sha256(roles_path),
        },
    }
    summary["summary_payload_sha256"] = combined_scorer._canonical_sha256(
        summary
    )
    (out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--synchronization-dir", type=Path, required=True)
    parser.add_argument("--rescue-dir", type=Path, required=True)
    parser.add_argument("--relaxed-evidence-dir", type=Path, required=True)
    parser.add_argument("--v447-calibration-dir", type=Path, required=True)
    parser.add_argument("--v448-calibration-dir", type=Path, required=True)
    parser.add_argument("--completion-evidence-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = score_combined(
        bridge_dir=args.bridge_dir,
        synchronization_dir=args.synchronization_dir,
        rescue_dir=args.rescue_dir,
        relaxed_evidence_dir=args.relaxed_evidence_dir,
        v447_calibration_dir=args.v447_calibration_dir,
        v448_calibration_dir=args.v448_calibration_dir,
        completion_evidence_dir=args.completion_evidence_dir,
        calibration_dir=args.calibration_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
