#!/usr/bin/env python3
"""Truth-separated scorer for CA-BOSS v4.51 stable sparse completion.

The scorer verifies the complete v4.44--v4.49 provenance chain and the
target-safe v4.51 whole-world bootstrap calibration before opening target
truth.  V4.51 is scored as v4.44 plus the target-safe v4.47 union and only
those v4.49 proposals that clear the predeclared 80% stability threshold.
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

import bootstrap_stable_sparse_completion_v4 as calibration  # noqa: E402
import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_crossfit_empirical_null_reporter_synchronization_v4 as v447_scorer  # noqa: E402
import score_crossfit_multivariate_reporter_rescue_v4 as v448_scorer  # noqa: E402
import score_crossfit_sparse_reporter_completion_v4 as v449_scorer  # noqa: E402
import score_heldout_synchronized_majority_rescue_v4 as rescue_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as combined_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_stable_sparse_completion.scorer.v1"
ALGORITHM = (
    "CA-BOSS v4.51 combined v4.44 plus world-bootstrap-stable sparse completion"
)


def _verify_calibration(
    calibration_dir: Path,
    *,
    bridge_provenance: Mapping[str, Any],
    completion_evidence: Path,
    v447_calibration: Path,
    v447_manifest: Mapping[str, Any],
    v449_calibration: Path,
    v449_manifest: Mapping[str, Any],
) -> tuple[
    Path,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Verify one frozen v4.51 target-block calibration before truth."""

    directory = calibration_dir.expanduser().absolute().resolve(strict=True)
    manifest = calibration.v449._verified_payload(
        directory / "calibration_manifest.json", "manifest_payload_sha256"
    )
    if manifest.get("schema_version") != calibration.SCHEMA:
        raise ValueError("Unexpected v4.51 calibration schema")
    boundary = manifest.get("information_boundary", {})
    required_boundary = {
        "other_block_simulation_roles_used_for_supervised_training": True,
        "other_block_sentinel_null_roles_used_for_null_calibration": True,
        "target_block_scorer_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "target_safe_v447_used_only_as_target_baseline_union": True,
        "target_safe_v449_used_only_to_nominate_proposals": True,
        "non_target_v449_selections_used_for_training": False,
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
            raise ValueError(f"v4.51 information boundary mismatch: {field}")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.51 lacks a target-specific pre-truth barrier")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.51 lacks a formal pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.51 created a new world or dataset")
    for field in (
        "candidate_pairs_may_be_expanded_as_edges",
        "within_assay_direct_structure_claimed",
        "exact_cross_assay_pair_claimed",
        "cross_assay_direction_claimed",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"v4.51 invalid structural claim: {field}")
    if list(manifest.get("model_feature_columns", [])) != list(
        calibration.FEATURE_COLUMNS
    ):
        raise ValueError("v4.51 model feature set changed")
    if [float(value) for value in manifest.get("model_c_values", [])] != list(
        calibration.MODEL_C_VALUES
    ):
        raise ValueError("v4.51 regularization guard changed")
    if int(manifest.get("bootstrap_replicates_configured", -1)) != (
        calibration.BOOTSTRAP_REPLICATES
    ):
        raise ValueError("v4.51 bootstrap replicate count changed")
    if float(manifest.get("bootstrap_frequency_threshold", -1.0)) != (
        calibration.BOOTSTRAP_FREQUENCY_THRESHOLD
    ):
        raise ValueError("v4.51 bootstrap stability threshold changed")
    if int(manifest.get("bootstrap_base_seed", -1)) != (
        calibration.BOOTSTRAP_BASE_SEED
    ):
        raise ValueError("v4.51 bootstrap seed changed")
    rollback = manifest.get("rollback_policy", {})
    if rollback.get("v448_supervised_multivariate_additions_carried_forward") is not False:
        raise ValueError("v4.51 carried an unstable v4.48 addition")
    if rollback.get("v450_interaction_additions_carried_forward") is not False:
        raise ValueError("v4.51 carried an unstable v4.50 addition")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"v4.51 calibration output hash mismatch: {name}")

    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = calibration.v449._world_key(block, seed)
    if str(manifest.get("target_block")) != block:
        raise ValueError("v4.51 target block does not match bridge")
    if world_key not in set(map(str, manifest.get("target_world_keys", []))):
        raise ValueError("v4.51 calibration lacks the target world")

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
    evidence_inputs = manifest.get("input_sha256", {}).get(
        "completion_evidence_worlds", {}
    )
    if evidence_inputs.get(world_key) != expected_evidence:
        raise ValueError("v4.51 completion-evidence binding mismatch")
    expected_v447 = {
        "calibration_manifest.json": combined_scorer._sha256(
            v447_calibration / "calibration_manifest.json"
        ),
        **{
            name: combined_scorer._sha256(v447_calibration / str(name))
            for name in v447_manifest.get("output_sha256", {})
        },
    }
    if manifest.get("input_sha256", {}).get(
        "target_safe_v447_calibration"
    ) != expected_v447:
        raise ValueError("v4.51 target-safe v4.47 binding mismatch")
    expected_v449 = {
        "calibration_manifest.json": combined_scorer._sha256(
            v449_calibration / "calibration_manifest.json"
        ),
        **{
            name: combined_scorer._sha256(v449_calibration / str(name))
            for name in v449_manifest.get("output_sha256", {})
        },
    }
    if manifest.get("input_sha256", {}).get(
        "target_safe_v449_calibration"
    ) != expected_v449:
        raise ValueError("v4.51 target-safe v4.49 binding mismatch")

    candidates = pd.read_csv(
        directory / "bootstrap_stable_candidates.tsv",
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
        raise ValueError("v4.51 candidate world identity mismatch")
    if len(selected) and not selected["hypothesis_selected"].map(
        combined_scorer._as_bool
    ).all():
        raise ValueError("v4.51 contains an unselected candidate")
    if len(selected) and selected["candidate_pairs_may_be_expanded_as_edges"].map(
        combined_scorer._as_bool
    ).any():
        raise ValueError("v4.51 expanded an ambiguity candidate")

    pair_audit = pd.read_csv(
        directory / "bootstrap_stable_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    world_pair_audit = pair_audit.loc[
        pair_audit["world_key"].astype(str).eq(world_key)
    ].copy()
    stability_audit = pd.read_csv(
        directory / "bootstrap_reporter_stability_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    world_stability_audit = stability_audit.loc[
        stability_audit["world_key"].astype(str).eq(world_key)
    ].copy()
    if world_stability_audit.empty:
        raise ValueError("v4.51 stability audit lacks the target world")
    return (
        directory,
        manifest,
        selected,
        world_pair_audit,
        world_stability_audit,
    )


def normalize_predictions(
    core: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Adapt the v4.51 union to the unexpanded-hyperedge scorer."""

    incremental_ids = set(candidates["stable_hyperedge_id"].astype(str))
    incremental = candidates.rename(
        columns={"stable_hyperedge_id": "synchronized_hyperedge_id"}
    ).copy()
    predictions = combined_scorer.normalize_predictions(core, incremental)
    predictions.loc[
        predictions["hyperedge_id"].astype(str).isin(incremental_ids),
        "prediction_source",
    ] = "v451_bootstrap_stable_sparse_completion"
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
    v449_calibration_dir: Path,
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
    v448_calibrated, v448_manifest, _, _ = v448_scorer._verify_calibration(
        v448_calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence=relaxed_evidence,
        rescue=rescue,
        v447_calibration=v447_calibrated,
        v447_manifest=v447_manifest,
    )
    completion_evidence, completion_manifest = (
        v449_scorer._verify_completion_evidence(
            completion_evidence_dir,
            bridge=bridge,
            bridge_provenance=bridge_provenance,
        )
    )
    v449_calibrated, v449_manifest, _, _, _ = v449_scorer._verify_calibration(
        v449_calibration_dir,
        bridge_provenance=bridge_provenance,
        completion_evidence=completion_evidence,
        v448_calibration=v448_calibrated,
        v448_manifest=v448_manifest,
    )
    calibrated, calibration_manifest, candidates, pair_audit, stability_audit = (
        _verify_calibration(
            calibration_dir,
            bridge_provenance=bridge_provenance,
            completion_evidence=completion_evidence,
            v447_calibration=v447_calibrated,
            v447_manifest=v447_manifest,
            v449_calibration=v449_calibrated,
            v449_manifest=v449_manifest,
        )
    )

    scorer = scorer_dir.expanduser().absolute().resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    truth_path = scorer / "truth_typed_edges.tsv"
    roles_path = scorer / "role_assignments.tsv"
    if any(path.is_symlink() or not path.is_file() for path in (truth_path, roles_path)):
        raise FileNotFoundError("Scorer truth files are incomplete")

    core = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    predictions = normalize_predictions(core, candidates)

    # Target truth is deliberately opened only after all upstream checks.
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
        "singleton_reporter_pairs_are_scored_diagnostically_but_v451_"
        "emits_bootstrap_stable_shared_state_equivalence_hypotheses_not_"
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
        pair_audit["newly_selected_by_v451"].map(combined_scorer._as_bool).sum()
    ) if len(pair_audit) else 0
    stable_reporters = int(
        stability_audit["retained_by_bootstrap_stability_guard"]
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
        "v449_calibration_schema_version": v449_manifest["schema_version"],
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
        "v449_calibration_manifest_payload_sha256": v449_manifest[
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
            "other_seed_blocks_only_for_reporter_role_training_null_"
            "calibration_and_whole_world_bootstrap_stability"
        ),
        "formal_pretruth_barrier_satisfied": True,
        "eligible_for_target_block_crossfit_evaluation": True,
        "eligible_for_fresh_locked_certification": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "selected_v451_union_hypotheses": len(candidates),
        "newly_selected_v451_assay_pairs": newly_selected,
        "bootstrap_stable_sparse_reporters": stable_reporters,
        "bootstrap_replicates_run": calibration_manifest[
            "bootstrap_replicates_run"
        ],
        "bootstrap_frequency_threshold": calibration_manifest[
            "bootstrap_frequency_threshold"
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
    parser.add_argument("--v449-calibration-dir", type=Path, required=True)
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
        v449_calibration_dir=args.v449_calibration_dir,
        calibration_dir=args.calibration_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
