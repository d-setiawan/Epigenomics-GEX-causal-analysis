#!/usr/bin/env python3
"""Truth-separated scorer for CA-BOSS v4.58 context-stratified stability."""

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

import bootstrap_joint_context_stratified_stability_rescue_v4 as calibration  # noqa: E402
import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as combined_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_joint_context_stratified_stability_rescue.scorer.v1"
ALGORITHM = (
    "CA-BOSS v4.58 combined v4.44 plus v4.55 plus joint local-context-"
    "stratified bootstrap stability rescue"
)


def _verify_calibration(
    directory: Path,
    *,
    bridge_provenance: Mapping[str, Any],
    evidence_path: Path,
    evidence_manifest: Mapping[str, Any],
    v455_path: Path,
    v455_manifest: Mapping[str, Any],
    v457_path: Path,
    v457_manifest: Mapping[str, Any],
    diagnostic_path: Path,
    diagnostic_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    directory = directory.expanduser().absolute().resolve(strict=True)
    manifest = calibration.diagnostic.v457.joint.geometry._verified_manifest(
        directory / "calibration_manifest.json", calibration.SCHEMA
    )
    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = calibration.diagnostic.v457.v455.v454._world_key(block, seed)
    if str(manifest.get("target_block")) != block:
        raise ValueError("v4.58 calibration target does not match bridge")
    if world_key not in set(map(str, manifest.get("target_world_keys", []))):
        raise ValueError("v4.58 calibration lacks bridge world")
    if manifest.get("fixed_selection_rule") != calibration.FIXED_SELECTION_RULE:
        raise ValueError("v4.58 calibration selection rule changed")
    if int(manifest.get("required_local_context_count", -1)) != (
        calibration.diagnostic.REQUIRED_LOCAL_CONTEXT_COUNT
    ):
        raise ValueError("v4.58 calibration context requirement changed")
    if int(manifest.get("bootstrap_replicates", -1)) != (
        calibration.diagnostic.v457.BOOTSTRAP_REPLICATES
    ):
        raise ValueError("v4.58 calibration bootstrap count changed")
    if manifest.get("posterior_interpretation") != (
        "working_beta_comparison_not_an_independent_trial_posterior"
    ):
        raise ValueError("v4.58 calibration posterior interpretation changed")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
        "target_block_scorer_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "other_block_truth_used_only_to_identify_null_pairs": True,
        "other_block_positive_labels_used_for_score_fit": False,
        "target_safe_v455_used_only_as_target_baseline_union": True,
        "existing_cached_worlds_only": True,
        "new_worlds_or_datasets_created": False,
        "heldout_background_component_scores_only_for_new_features": True,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "transport_plan_constructed": False,
        "simulator_latent_state_used": False,
        "external_biological_features_or_priors_used": False,
        "modality_identity_used_as_model_feature": False,
        "candidate_feature_identity_used_as_model_feature": False,
    }.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.58 calibration boundary mismatch: {field}")
    for field in (
        "target_specific_pretruth_barrier_satisfied",
        "formal_pretruth_barrier_satisfied",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"v4.58 calibration lacks required barrier: {field}")
    for field in (
        "new_worlds_or_datasets_created",
        "fresh_locked_certification",
        "perfect_matching_assumed",
        "fixed_shared_relationship_count_assumed",
        "supervised_positive_labels_used_for_score_fit",
        "candidate_pairs_may_be_expanded_as_edges",
        "within_assay_direct_structure_claimed",
        "exact_cross_assay_pair_claimed",
        "cross_assay_direction_claimed",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"v4.58 calibration invalid claim: {field}")
    for name, expected in manifest.get("source_sha256", {}).items():
        path = SCRIPT_DIR / str(name)
        if not path.is_file() or combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"v4.58 calibration source changed: {name}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"v4.58 calibration output hash mismatch: {name}")
    expected_inputs = {
        "joint_evidence_manifest.json": combined_scorer._sha256(
            evidence_path / "evidence_manifest.json"
        ),
        "v455_calibration_manifest.json": combined_scorer._sha256(
            v455_path / "calibration_manifest.json"
        ),
        "v457_diagnostic_manifest.json": combined_scorer._sha256(
            v457_path / "diagnostic_manifest.json"
        ),
        "v458_diagnostic_manifest.json": combined_scorer._sha256(
            diagnostic_path / "diagnostic_manifest.json"
        ),
    }
    if manifest.get("input_sha256") != expected_inputs:
        raise ValueError("v4.58 calibration input bindings changed")
    parents = manifest.get("parent_manifest_payload_sha256", {})
    expected_parents = {
        "joint_evidence": evidence_manifest["manifest_payload_sha256"],
        "v455": v455_manifest["manifest_payload_sha256"],
        "v457_diagnostic": v457_manifest["manifest_payload_sha256"],
        "v458_diagnostic": diagnostic_manifest["manifest_payload_sha256"],
    }
    if parents != expected_parents:
        raise ValueError("v4.58 calibration parent payloads changed")

    candidates = pd.read_csv(
        directory / "bootstrap_joint_context_stability_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if list(candidates.columns) != calibration.CANDIDATE_COLUMNS:
        raise ValueError("v4.58 candidate columns changed")
    selected = candidates[candidates["world_key"].astype(str).eq(world_key)].copy()
    if len(selected) and (
        ~selected["block"].astype(str).eq(block)
        | ~pd.to_numeric(selected["seed"], errors="raise").eq(seed)
    ).any():
        raise ValueError("v4.58 candidate identity mismatch")
    if len(selected) and not selected["hypothesis_selected"].map(
        combined_scorer._as_bool
    ).all():
        raise ValueError("v4.58 contains an unselected candidate")
    if len(selected) and selected["candidate_pairs_may_be_expanded_as_edges"].map(
        combined_scorer._as_bool
    ).any():
        raise ValueError("v4.58 expands ambiguity alternatives")
    pair_audit = pd.read_csv(
        directory / "bootstrap_joint_context_stability_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    world_pair_audit = pair_audit[
        pair_audit["world_key"].astype(str).eq(world_key)
    ].copy()
    if world_pair_audit.empty:
        raise ValueError("v4.58 pair audit lacks bridge world")
    return directory, manifest, selected, world_pair_audit


def normalize_predictions(core: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    incremental_ids = set(candidates["joint_context_stability_hyperedge_id"].astype(str))
    incremental = candidates.rename(
        columns={
            "joint_context_stability_hyperedge_id": "synchronized_hyperedge_id"
        }
    ).copy()
    predictions = combined_scorer.normalize_predictions(core, incremental)
    predictions.loc[
        predictions["hyperedge_id"].astype(str).isin(incremental_ids),
        "prediction_source",
    ] = "v458_joint_context_stratified_stability_rescue"
    return predictions


def score_combined(
    *,
    bridge_dir: Path,
    evidence_dir: Path,
    v455_calibration_dir: Path,
    v457_diagnostic_dir: Path,
    diagnostic_dir: Path,
    calibration_dir: Path,
    scorer_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    bridge, bridge_provenance = core_scorer._verify_bridge(bridge_dir)
    evidence_path, evidence_manifest, evidence = (
        calibration.diagnostic.v457._verified_joint_evidence(evidence_dir)
    )
    block = str(bridge_provenance["block"])
    seed = int(bridge_provenance["seed"])
    world_key = calibration.diagnostic.v457.v455.v454._world_key(block, seed)
    target_keys = sorted(
        evidence.loc[
            evidence["block"].astype(str).eq(block), "world_key"
        ].astype(str).unique()
    )
    if world_key not in target_keys:
        raise ValueError("v4.58 joint evidence lacks bridge world")
    v455_path, v455_manifest, _ = (
        calibration.diagnostic.v457._verified_v455_baseline(
            v455_calibration_dir,
            target_block=block,
            target_world_keys=target_keys,
        )
    )
    v457_path, v457_manifest, _, _ = (
        calibration.diagnostic._verify_v457_diagnostic(
            v457_diagnostic_dir,
            target_block=block,
            evidence_path=evidence_path,
            evidence_manifest=evidence_manifest,
        )
    )
    diagnostic_path, diagnostic_manifest, _ = calibration._verify_diagnostic(
        diagnostic_dir,
        target_block=block,
        evidence_path=evidence_path,
        evidence_manifest=evidence_manifest,
        v457_path=v457_path,
        v457_manifest=v457_manifest,
    )
    calibrated, calibration_manifest, candidates, pair_audit = _verify_calibration(
        calibration_dir,
        bridge_provenance=bridge_provenance,
        evidence_path=evidence_path,
        evidence_manifest=evidence_manifest,
        v455_path=v455_path,
        v455_manifest=v455_manifest,
        v457_path=v457_path,
        v457_manifest=v457_manifest,
        diagnostic_path=diagnostic_path,
        diagnostic_manifest=diagnostic_manifest,
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
        "singleton_reporter_pairs_are_identifiability_diagnostics_but_v458_"
        "emits_joint_context_stratified_shared_state_equivalence_hypotheses_"
        "not_direct_or_directed_edges"
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
        pair_audit["newly_selected_by_v458"].map(combined_scorer._as_bool).sum()
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "bridge_schema_version": bridge_provenance["schema_version"],
        "evidence_schema_version": evidence_manifest["schema_version"],
        "v455_calibration_schema_version": v455_manifest["schema_version"],
        "v457_diagnostic_schema_version": v457_manifest["schema_version"],
        "diagnostic_schema_version": diagnostic_manifest["schema_version"],
        "calibration_schema_version": calibration_manifest["schema_version"],
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "evidence_manifest_payload_sha256": evidence_manifest[
            "manifest_payload_sha256"
        ],
        "v455_calibration_manifest_payload_sha256": v455_manifest[
            "manifest_payload_sha256"
        ],
        "v457_diagnostic_manifest_payload_sha256": v457_manifest[
            "manifest_payload_sha256"
        ],
        "diagnostic_manifest_payload_sha256": diagnostic_manifest[
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
            "other_seed_blocks_only_to_identify_context_stratified_empirical_null_pairs"
        ),
        "formal_pretruth_barrier_satisfied": True,
        "eligible_for_target_block_crossfit_evaluation": True,
        "eligible_for_fresh_locked_certification": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "perfect_matching_assumed": False,
        "fixed_shared_relationship_count_assumed": False,
        "selected_v458_union_hypotheses": len(candidates),
        "newly_selected_v458_assay_pairs": newly_selected,
        "bootstrap_replicates": calibration_manifest["bootstrap_replicates"],
        "minimum_selection_frequency": calibration_manifest[
            "minimum_selection_frequency"
        ],
        "minimum_selection_frequency_comparison": calibration_manifest[
            "minimum_selection_frequency_comparison"
        ],
        "required_local_context_count": calibration_manifest[
            "required_local_context_count"
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
    parser.add_argument("--v455-calibration-dir", type=Path, required=True)
    parser.add_argument("--v457-diagnostic-dir", type=Path, required=True)
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
        v455_calibration_dir=args.v455_calibration_dir,
        v457_diagnostic_dir=args.v457_diagnostic_dir,
        diagnostic_dir=args.diagnostic_dir,
        calibration_dir=args.calibration_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
