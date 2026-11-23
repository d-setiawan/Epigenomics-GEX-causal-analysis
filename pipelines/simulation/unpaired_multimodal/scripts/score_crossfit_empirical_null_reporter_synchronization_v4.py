#!/usr/bin/env python3
"""Truth-separated scorer for CA-BOSS v4.47 cross-fit null selection.

The scorer verifies the frozen v4.44 bridge, the v4.45 synchronization stage,
the v4.46 guarded rescue, the v4.47 relaxed evidence, and the target-block
cross-fit calibration manifest before opening the target world's scorer
partition.  Other-block simulation truth is an explicit calibration input;
target-block truth is never used for fitting or selection.
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
import crossfit_empirical_null_reporter_synchronization_v4 as calibration  # noqa: E402
import relaxed_reporter_synchronization_evidence_v4 as relaxed  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_heldout_synchronized_majority_rescue_v4 as rescue_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as combined_scorer  # noqa: E402


SCHEMA = (
    "ca_boss_v4.crossfit_empirical_null_reporter_synchronization.scorer.v1"
)
ALGORITHM = (
    "CA-BOSS v4.47 combined v4.44 plus cross-fit empirical-null "
    "reporter synchronization"
)


def _verify_relaxed_evidence(
    evidence_dir: Path,
    bridge: Path,
    bridge_provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Verify target relaxed evidence and its exact bridge binding."""

    directory, manifest = calibration._verified_manifest(
        evidence_dir,
        manifest_name="evidence_manifest.json",
        schema=relaxed.SCHEMA,
    )
    bindings = {
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": combined_scorer._sha256(
            bridge / "provenance.json"
        ),
    }
    for field, expected in bindings.items():
        if str(manifest.get(field)) != str(expected):
            raise ValueError(f"Relaxed-evidence bridge mismatch: {field}")
    if (
        str(manifest.get("block")) != str(bridge_provenance.get("block"))
        or int(manifest.get("seed")) != int(bridge_provenance.get("seed"))
    ):
        raise ValueError("Relaxed evidence and bridge world identities disagree")
    return directory, manifest


def _verify_calibration(
    calibration_dir: Path,
    *,
    bridge_provenance: Mapping[str, Any],
    evidence: Path,
    rescue: Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    """Verify the target-specific calibration and upstream file bindings."""

    directory = calibration_dir.expanduser().absolute().resolve(strict=True)
    manifest_path = directory / "calibration_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != calibration.SCHEMA:
        raise ValueError("Unexpected cross-fit calibration schema")
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != combined_scorer._canonical_sha256(unsigned):
        raise ValueError("Cross-fit calibration payload hash mismatch")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("target_block_scorer_truth_opened") is not False:
        raise ValueError("Calibration opened target-block scorer truth")
    if boundary.get("target_world_truth_used_for_fit_or_selection") is not False:
        raise ValueError("Calibration used target-world truth")
    if (
        boundary.get("other_block_simulation_truth_used_for_null_calibration")
        is not True
    ):
        raise ValueError("Calibration did not disclose training-truth use")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("Calibration lacks a target-specific pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Calibration created a new world or dataset")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"Calibration output hash mismatch: {name}")

    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = f"{block}__world_seed{seed}"
    if str(manifest.get("target_block")) != block:
        raise ValueError("Calibration target block does not match bridge")
    if world_key not in set(manifest.get("target_world_keys", [])):
        raise ValueError("Calibration does not contain the target world")
    inputs = manifest.get("input_sha256", {})
    evidence_hashes = inputs.get("relaxed_evidence_worlds", {}).get(world_key)
    rescue_hashes = inputs.get("v446_guarded_rescue_worlds", {}).get(world_key)
    expected_evidence = {
        "evidence_manifest.json": combined_scorer._sha256(
            evidence / "evidence_manifest.json"
        ),
        "relaxed_reporter_synchronization_pairs.tsv": (
            combined_scorer._sha256(
                evidence / "relaxed_reporter_synchronization_pairs.tsv"
            )
        ),
    }
    expected_rescue = {
        "rescue_manifest.json": combined_scorer._sha256(
            rescue / "rescue_manifest.json"
        ),
        "heldout_synchronized_majority_candidates.tsv": (
            combined_scorer._sha256(
                rescue / "heldout_synchronized_majority_candidates.tsv"
            )
        ),
    }
    if evidence_hashes != expected_evidence:
        raise ValueError("Calibration relaxed-evidence binding mismatch")
    if rescue_hashes != expected_rescue:
        raise ValueError("Calibration v4.46 binding mismatch")

    candidates = pd.read_csv(
        directory / "crossfit_empirical_null_candidates.tsv",
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
        raise ValueError("Calibration candidate world identity mismatch")
    return directory, manifest, selected


def normalize_crossfit_predictions(
    core: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Adapt v4.47 hyperedges to the shared unexpanded-hyperedge scorer."""

    incremental_ids = set(
        candidates["empirical_null_hyperedge_id"].astype(str)
    )
    incremental = candidates.rename(
        columns={
            "empirical_null_hyperedge_id": "synchronized_hyperedge_id"
        }
    ).copy()
    predictions = combined_scorer.normalize_predictions(core, incremental)
    predictions.loc[
        predictions["hyperedge_id"].astype(str).isin(incremental_ids),
        "prediction_source",
    ] = "v447_crossfit_empirical_null"
    return predictions


def score_combined(
    *,
    bridge_dir: Path,
    synchronization_dir: Path,
    rescue_dir: Path,
    relaxed_evidence_dir: Path,
    calibration_dir: Path,
    scorer_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Verify every pre-truth input, then score one target world."""

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
    evidence, evidence_manifest = _verify_relaxed_evidence(
        relaxed_evidence_dir, bridge, bridge_provenance
    )
    calibrated, calibration_manifest, candidates = _verify_calibration(
        calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence=evidence,
        rescue=rescue,
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
    predictions = normalize_crossfit_predictions(core, candidates)
    expected_incremental = len(candidates)
    if not candidates.empty and not candidates["hypothesis_selected"].map(
        combined_scorer._as_bool
    ).all():
        raise ValueError("Calibration table contains an unselected hypothesis")

    # Truth is deliberately opened only after every upstream verification.
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
        "singleton_reporter_pairs_are_scored_diagnostically_but_v447_"
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
    summary = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "bridge_schema_version": bridge_provenance["schema_version"],
        "synchronization_schema_version": synchronization_manifest[
            "schema_version"
        ],
        "rescue_schema_version": rescue_manifest["schema_version"],
        "relaxed_evidence_schema_version": evidence_manifest["schema_version"],
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
        "relaxed_evidence_manifest_payload_sha256": evidence_manifest[
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
        "other_block_simulation_truth_used_for_null_calibration": True,
        "target_block_truth_used_for_fit_or_selection": False,
        "algorithm_used_truth_for_fit_or_selection": True,
        "algorithm_truth_use_scope": (
            "other_seed_blocks_only_for_scalar_empirical_null_threshold"
        ),
        "formal_pretruth_barrier_satisfied": True,
        "eligible_for_target_block_crossfit_evaluation": True,
        "eligible_for_fresh_locked_certification": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "selected_v447_incremental_hypotheses": expected_incremental,
        "calibration_null_upper_envelope": calibration_manifest[
            "calibration_null_upper_envelope"
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
        calibration_dir=args.calibration_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
