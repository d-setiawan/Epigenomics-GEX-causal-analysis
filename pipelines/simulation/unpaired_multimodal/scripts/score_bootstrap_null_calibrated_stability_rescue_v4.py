#!/usr/bin/env python3
"""Truth-separated scorer for CA-BOSS v4.55 null-calibrated stability."""

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

import bootstrap_null_calibrated_stability_rescue_v4 as calibration  # noqa: E402
import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as combined_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_null_calibrated_stability_rescue.scorer.v1"
ALGORITHM = (
    "CA-BOSS v4.55 combined v4.44 plus empirical-null-calibrated "
    "bootstrap stability rescue"
)


def _verify_calibration(
    calibration_dir: Path,
    *,
    bridge_provenance: Mapping[str, Any],
    evidence_dir: Path,
    evidence_manifest: Mapping[str, Any],
    v453_dir: Path,
    v453_manifest: Mapping[str, Any],
    v454_dir: Path,
    v454_manifest: Mapping[str, Any],
    diagnostic_dir: Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    directory = calibration_dir.expanduser().absolute().resolve(strict=True)
    manifest = calibration.v454.diagnostic._verified_manifest(
        directory / "calibration_manifest.json", calibration.SCHEMA
    )
    boundary = manifest.get("information_boundary", {})
    required_boundary = {
        "target_block_scorer_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "other_block_truth_used_only_to_identify_null_pairs": True,
        "other_block_positive_labels_used_for_score_fit": False,
        "target_safe_v454_used_only_as_target_baseline_union": True,
        "existing_cached_worlds_only": True,
        "new_worlds_or_datasets_created": False,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "cell_level_transport_coupling_constructed": False,
        "simulator_latent_state_used": False,
        "external_biological_features_or_priors_used": False,
        "modality_identity_used_as_model_feature": False,
        "candidate_feature_identity_used_as_model_feature": False,
    }
    for field, expected in required_boundary.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.55 information boundary mismatch: {field}")
    for field in (
        "target_specific_pretruth_barrier_satisfied",
        "formal_pretruth_barrier_satisfied",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"v4.55 lacks required barrier: {field}")
    for field in (
        "new_worlds_or_datasets_created",
        "perfect_matching_assumed",
        "fixed_shared_relationship_count_assumed",
        "candidate_pairs_may_be_expanded_as_edges",
        "within_assay_direct_structure_claimed",
        "exact_cross_assay_pair_claimed",
        "cross_assay_direction_claimed",
        "supervised_positive_labels_used_for_score_fit",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"v4.55 invalid claim or boundary: {field}")
    if manifest.get("fixed_selection_rule") != calibration.FIXED_SELECTION_RULE:
        raise ValueError("v4.55 selection rule changed")
    if int(manifest.get("bootstrap_replicates", -1)) != (
        calibration.diagnostic.BOOTSTRAP_REPLICATES
    ):
        raise ValueError("v4.55 bootstrap count changed")
    if float(manifest.get("minimum_selection_frequency", -1.0)) != (
        calibration.diagnostic.MINIMUM_SELECTION_FREQUENCY
    ):
        raise ValueError("v4.55 majority floor changed")
    if manifest.get("minimum_selection_frequency_comparison") != (
        "strictly_greater_than"
    ):
        raise ValueError("v4.55 majority comparison changed")
    if float(manifest.get("posterior_dominance_threshold", -1.0)) != (
        calibration.diagnostic.POSTERIOR_DOMINANCE_THRESHOLD
    ):
        raise ValueError("v4.55 posterior guard changed")
    if manifest.get("posterior_interpretation") != (
        "working_beta_comparison_not_an_independent_trial_posterior"
    ):
        raise ValueError("v4.55 bootstrap uncertainty interpretation changed")

    source_hashes = manifest.get("source_sha256", {})
    for module in (
        calibration,
        calibration.diagnostic,
        calibration.v454,
        calibration.v454_scorer,
    ):
        path = Path(module.__file__).resolve()
        if source_hashes.get(path.name) != combined_scorer._sha256(path):
            raise ValueError(f"v4.55 calibration source changed: {path.name}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"v4.55 output hash mismatch: {name}")

    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = calibration.v454._world_key(block, seed)
    if str(manifest.get("target_block")) != block:
        raise ValueError("v4.55 target block does not match bridge")
    if world_key not in set(map(str, manifest.get("target_world_keys", []))):
        raise ValueError("v4.55 calibration lacks target world")

    expected_evidence = {
        "evidence_manifest.json": combined_scorer._sha256(
            evidence_dir / "evidence_manifest.json"
        ),
        "quantile_geometry_pair_evidence.tsv": combined_scorer._sha256(
            evidence_dir / "quantile_geometry_pair_evidence.tsv"
        ),
    }
    inputs = manifest.get("input_sha256", {})
    if inputs.get("geometry_evidence") != expected_evidence:
        raise ValueError("v4.55 geometry-evidence binding mismatch")
    if str(evidence_manifest.get("schema_version")) != (
        calibration.v454.geometry.SCHEMA
    ):
        raise ValueError("v4.55 geometry evidence schema changed")
    if inputs.get("target_safe_v453_calibration_manifest.json") != (
        combined_scorer._sha256(v453_dir / "calibration_manifest.json")
    ):
        raise ValueError("v4.55 v4.53 binding mismatch")
    if inputs.get("target_safe_v454_calibration_manifest.json") != (
        combined_scorer._sha256(v454_dir / "calibration_manifest.json")
    ):
        raise ValueError("v4.55 v4.54 binding mismatch")
    diagnostic_path, diagnostic_manifest, _ = calibration._verify_diagnostic(
        diagnostic_dir,
        target_block=block,
        v454_calibration_dir=v454_dir,
    )
    if inputs.get("v455_diagnostic_manifest.json") != combined_scorer._sha256(
        diagnostic_path / "diagnostic_manifest.json"
    ):
        raise ValueError("v4.55 diagnostic binding mismatch")
    parents = manifest.get("parent_manifest_payload_sha256", {})
    if parents.get("v453") != v453_manifest.get("manifest_payload_sha256"):
        raise ValueError("v4.55 v4.53 parent payload mismatch")
    if parents.get("v454") != v454_manifest.get("manifest_payload_sha256"):
        raise ValueError("v4.55 v4.54 parent payload mismatch")
    if parents.get("v455_diagnostic") != diagnostic_manifest.get(
        "manifest_payload_sha256"
    ):
        raise ValueError("v4.55 diagnostic parent payload mismatch")

    candidates = pd.read_csv(
        directory / "bootstrap_null_calibrated_candidates.tsv",
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
        raise ValueError("v4.55 candidate identity mismatch")
    if len(selected) and not selected["hypothesis_selected"].map(
        combined_scorer._as_bool
    ).all():
        raise ValueError("v4.55 contains an unselected candidate")
    if len(selected) and selected["candidate_pairs_may_be_expanded_as_edges"].map(
        combined_scorer._as_bool
    ).any():
        raise ValueError("v4.55 expands ambiguity alternatives")
    pair_audit = pd.read_csv(
        directory / "bootstrap_null_calibrated_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    world_pair_audit = pair_audit.loc[
        pair_audit["world_key"].astype(str).eq(world_key)
    ].copy()
    if world_pair_audit.empty:
        raise ValueError("v4.55 pair audit lacks target world")
    return directory, manifest, selected, world_pair_audit


def normalize_predictions(core: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    incremental_ids = set(candidates["null_calibrated_hyperedge_id"].astype(str))
    incremental = candidates.rename(
        columns={"null_calibrated_hyperedge_id": "synchronized_hyperedge_id"}
    ).copy()
    predictions = combined_scorer.normalize_predictions(core, incremental)
    predictions.loc[
        predictions["hyperedge_id"].astype(str).isin(incremental_ids),
        "prediction_source",
    ] = "v455_bootstrap_null_calibrated_stability_rescue"
    return predictions


def score_combined(
    *,
    bridge_dir: Path,
    evidence_dir: Path,
    v453_calibration_dir: Path,
    v454_calibration_dir: Path,
    diagnostic_dir: Path,
    calibration_dir: Path,
    scorer_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    bridge, bridge_provenance = core_scorer._verify_bridge(bridge_dir)
    evidence, evidence_manifest, evidence_table = (
        calibration.v454._verified_evidence(evidence_dir)
    )
    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = calibration.v454._world_key(block, seed)
    world_evidence = evidence_table.loc[
        evidence_table["world_key"].astype(str).eq(world_key)
    ]
    if world_evidence.empty:
        raise ValueError("v4.55 geometry evidence lacks bridge world")
    if set(world_evidence["block"].astype(str)) != {block} or set(
        pd.to_numeric(world_evidence["seed"], errors="raise").astype(int)
    ) != {seed}:
        raise ValueError("v4.55 geometry evidence identity mismatch")
    v453, v453_manifest, _ = calibration.v454._verified_v453(
        v453_calibration_dir,
        target_block=block,
        evidence_manifest=evidence_manifest,
    )
    v454, v454_manifest, _, _ = calibration.v454_scorer._verify_calibration(
        v454_calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence_dir=evidence,
        evidence_manifest=evidence_manifest,
        v453_dir=v453,
        v453_manifest=v453_manifest,
    )
    calibrated, calibration_manifest, candidates, pair_audit = _verify_calibration(
        calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence_dir=evidence,
        evidence_manifest=evidence_manifest,
        v453_dir=v453,
        v453_manifest=v453_manifest,
        v454_dir=v454,
        v454_manifest=v454_manifest,
        diagnostic_dir=diagnostic_dir,
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

    # No target truth is opened before this line.
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
        "singleton_reporter_pairs_are_diagnostic_but_v455_emits_"
        "empirical_null_calibrated_shared_state_equivalence_hypotheses_"
        "not_directed_edges"
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
        pair_audit["newly_selected_by_v455"].map(combined_scorer._as_bool).sum()
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "bridge_schema_version": bridge_provenance["schema_version"],
        "evidence_schema_version": evidence_manifest["schema_version"],
        "v453_calibration_schema_version": v453_manifest["schema_version"],
        "v454_calibration_schema_version": v454_manifest["schema_version"],
        "calibration_schema_version": calibration_manifest["schema_version"],
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "evidence_manifest_payload_sha256": evidence_manifest[
            "manifest_payload_sha256"
        ],
        "v453_calibration_manifest_payload_sha256": v453_manifest[
            "manifest_payload_sha256"
        ],
        "v454_calibration_manifest_payload_sha256": v454_manifest[
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
        "block": block,
        "seed": seed,
        "truth_opened_only_after_all_pretruth_hash_verification": True,
        "other_block_simulation_truth_used_for_null_calibration": True,
        "other_block_simulation_truth_used_for_supervised_training": False,
        "other_block_positive_labels_used_for_score_fit": False,
        "target_block_truth_used_for_fit_or_selection": False,
        "algorithm_used_truth_for_fit_or_selection": True,
        "algorithm_truth_use_scope": (
            "other_seed_blocks_only_to_identify_empirical_null_pairs"
        ),
        "formal_pretruth_barrier_satisfied": True,
        "eligible_for_target_block_crossfit_evaluation": True,
        "eligible_for_fresh_locked_certification": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "perfect_matching_assumed": False,
        "fixed_shared_relationship_count_assumed": False,
        "selected_v455_union_hypotheses": len(candidates),
        "newly_selected_v455_assay_pairs": newly_selected,
        "bootstrap_replicates": calibration_manifest["bootstrap_replicates"],
        "minimum_selection_frequency": calibration_manifest[
            "minimum_selection_frequency"
        ],
        "minimum_selection_frequency_comparison": calibration_manifest[
            "minimum_selection_frequency_comparison"
        ],
        "posterior_dominance_threshold": calibration_manifest[
            "posterior_dominance_threshold"
        ],
        "posterior_interpretation": calibration_manifest[
            "posterior_interpretation"
        ],
        "benchmark_metrics": metrics,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "source_sha256": {
            Path(__file__).name: combined_scorer._sha256(Path(__file__).resolve())
        },
        "output_sha256": {score_path.name: combined_scorer._sha256(score_path)},
        "truth_input_sha256": {
            truth_path.name: combined_scorer._sha256(truth_path),
            roles_path.name: combined_scorer._sha256(roles_path),
        },
    }
    summary["summary_payload_sha256"] = combined_scorer._canonical_sha256(summary)
    (out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v453-calibration-dir", type=Path, required=True)
    parser.add_argument("--v454-calibration-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = score_combined(
        bridge_dir=args.bridge_dir,
        evidence_dir=args.evidence_dir,
        v453_calibration_dir=args.v453_calibration_dir,
        v454_calibration_dir=args.v454_calibration_dir,
        diagnostic_dir=args.diagnostic_dir,
        calibration_dir=args.calibration_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
