#!/usr/bin/env python3
"""Stratify v4.57 stability by strict joint local-context consensus.

The global worst-null stability reference in v4.57 is dominated by pairs that
are not locally mutual nearest neighbours in every component-score context.
This cached-only v4.58 diagnostic therefore defines one structurally fixed
stratum: a candidate must be locally mutual-nearest under the full fit and
both disjoint held-out-background feature folds (3/3 contexts).  Target and
null candidates are compared only within that same maximum-consensus stratum.

The stage consumes a hash-bound target-pretruth v4.57 diagnostic and does not
open target truth.  It retains the strict-majority, strict empirical-null, and
working Beta(1, 1) dominance guards.  It produces shared-state-equivalence
evidence only, never a direct, exact-pair, or directional claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_joint_sliced_null_stability_v4 as v457  # noqa: E402


SCHEMA = "ca_boss_v4.joint_context_stratified_stability_diagnostic.v1"
ALGORITHM = "CA-BOSS v4.58 joint local-context-stratified stability diagnostic"
LOCAL_CONTEXT_COLUMN = "joint_sliced_w2_local_mutual_context_count"
REQUIRED_LOCAL_CONTEXT_COUNT = 3
REQUIRED_HELDOUT_LOCAL_CONTEXT_COUNT = 2
FIXED_SELECTION_RULE = (
    "start_only_from_v457_nominal_joint_geometry_pairs_new_relative_to_target_"
    "safe_v455_require_strict_local_mutual_nearest_agreement_in_all_three_"
    "component_score_contexts_comprising_full_fit_and_two_disjoint_heldout_"
    "background_feature_folds_calibrate_against_every_non_target_null_pair_"
    "in_the_same_three_of_three_local_consensus_stratum_retain_v457_100_"
    "whole_world_refit_counts_require_strictly_more_than_50_refits_strictly_"
    "exceed_the_stratum_specific_worst_null_count_and_at_least_0_975_working_"
    "beta_1_1_dominance_probability_never_claim_direct_structure_exact_pair_"
    "or_direction"
)


def _verify_v457_diagnostic(
    directory: Path,
    *,
    target_block: str,
    evidence_path: Path,
    evidence_manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = v457.joint.geometry._verified_manifest(
        resolved / "diagnostic_manifest.json", v457.SCHEMA
    )
    if str(manifest.get("target_block")) != target_block:
        raise ValueError("v4.57 diagnostic targets another block")
    if manifest.get("fixed_selection_rule") != v457.FIXED_SELECTION_RULE:
        raise ValueError("v4.57 diagnostic selection rule changed")
    if int(manifest.get("bootstrap_replicates", -1)) != v457.BOOTSTRAP_REPLICATES:
        raise ValueError("v4.57 diagnostic bootstrap count changed")
    if float(manifest.get("minimum_selection_frequency", -1.0)) != (
        v457.MINIMUM_SELECTION_FREQUENCY
    ):
        raise ValueError("v4.57 diagnostic majority floor changed")
    if float(manifest.get("posterior_dominance_threshold", -1.0)) != (
        v457.POSTERIOR_DOMINANCE_THRESHOLD
    ):
        raise ValueError("v4.57 diagnostic posterior guard changed")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
        "target_block_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "other_block_truth_used_only_to_identify_null_pairs": True,
        "other_block_positive_labels_used_for_score_fit": False,
        "target_evidence_used_as_unlabeled_empirical_rank_training": True,
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
            raise ValueError(f"v4.57 diagnostic boundary mismatch: {field}")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.57 diagnostic lacks a pretruth barrier")
    for name, expected in manifest.get("source_sha256", {}).items():
        path = SCRIPT_DIR / str(name)
        if not path.is_file() or v457.joint.geometry._sha256(path) != str(expected):
            raise ValueError(f"v4.57 diagnostic source changed: {name}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if v457.joint.geometry._sha256(path) != str(expected):
            raise ValueError(f"v4.57 diagnostic output hash mismatch: {name}")
    inputs = manifest.get("input_sha256", {})
    if inputs.get("joint_evidence_manifest.json") != v457.joint.geometry._sha256(
        evidence_path / "evidence_manifest.json"
    ):
        raise ValueError("v4.57 diagnostic and joint evidence manifest disagree")
    if inputs.get("joint_sliced_geometry_pair_evidence.tsv") != (
        v457.joint.geometry._sha256(
            evidence_path / "joint_sliced_geometry_pair_evidence.tsv"
        )
    ):
        raise ValueError("v4.57 diagnostic and joint evidence table disagree")
    if manifest.get("parent_manifest_payload_sha256", {}).get(
        "joint_evidence"
    ) != evidence_manifest.get("manifest_payload_sha256"):
        raise ValueError("v4.57 diagnostic joint-evidence parent changed")
    pair = pd.read_csv(
        resolved / "joint_null_calibrated_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    null = pd.read_csv(
        resolved / "joint_calibration_null_frequency_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if set(pair["block"].astype(str)) != {target_block}:
        raise ValueError("v4.57 pair audit crosses target blocks")
    if len(null) != int(manifest.get("empirical_null_pair_count", -1)):
        raise ValueError("v4.57 null audit row count changed")
    if not null["empirical_null"].map(
        v457.v455.v454.synchronization._as_bool
    ).all():
        raise ValueError("v4.57 null audit contains a non-null row")
    return resolved, manifest, pair, null


def _is_maximum_local_consensus(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or (~numeric.between(0, REQUIRED_LOCAL_CONTEXT_COUNT)).any():
        raise ValueError("Joint local-context counts are outside 0..3")
    if (numeric % 1 != 0).any():
        raise ValueError("Joint local-context counts must be integers")
    return numeric.astype(int).eq(REQUIRED_LOCAL_CONTEXT_COUNT)


def _selection_decision(
    *,
    nominal_new_proposal: bool,
    maximum_local_consensus: bool,
    candidate_count: int,
    null_count: int,
) -> tuple[bool, float]:
    probability = v457.v455_diagnostic._posterior_dominance_probability(
        candidate_count, null_count, v457.BOOTSTRAP_REPLICATES
    )
    selected = bool(
        nominal_new_proposal
        and maximum_local_consensus
        and candidate_count
        > int(v457.MINIMUM_SELECTION_FREQUENCY * v457.BOOTSTRAP_REPLICATES)
        and candidate_count > null_count
        and probability >= v457.POSTERIOR_DOMINANCE_THRESHOLD
    )
    return selected, probability


def diagnose_target_block(
    *,
    evidence_dir: Path,
    v457_diagnostic_dir: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    evidence_path, evidence_manifest, evidence = v457._verified_joint_evidence(
        evidence_dir
    )
    blocks = sorted(evidence["block"].astype(str).unique())
    if target_block not in blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    diagnostic_path, v457_manifest, pair, null = _verify_v457_diagnostic(
        v457_diagnostic_dir,
        target_block=target_block,
        evidence_path=evidence_path,
        evidence_manifest=evidence_manifest,
    )
    identity = ["world_key", "left_modality", "right_modality"]
    context = evidence.loc[:, identity + [LOCAL_CONTEXT_COLUMN]]
    if context.duplicated(identity).any():
        raise ValueError("Joint evidence contains duplicate context identities")
    pair = pair.merge(context, on=identity, how="left", validate="one_to_one")
    null = null.merge(context, on=identity, how="left", validate="many_to_one")
    if pair[LOCAL_CONTEXT_COLUMN].isna().any() or null[
        LOCAL_CONTEXT_COLUMN
    ].isna().any():
        raise ValueError("A v4.57 audit row lacks joint local-context evidence")

    pair["maximum_joint_local_context_consensus"] = _is_maximum_local_consensus(
        pair[LOCAL_CONTEXT_COLUMN]
    )
    null["in_maximum_joint_local_context_consensus_stratum"] = (
        _is_maximum_local_consensus(null[LOCAL_CONTEXT_COLUMN])
    )
    comparable_null = null[
        null["in_maximum_joint_local_context_consensus_stratum"]
    ].copy()
    if comparable_null.empty:
        raise ValueError("No empirical null lies in the maximum-consensus stratum")
    worst_index = comparable_null["bootstrap_selection_count"].astype(int).idxmax()
    worst_null_count = int(null.loc[worst_index, "bootstrap_selection_count"])
    null["is_worst_stratum_empirical_null"] = False
    null.loc[worst_index, "is_worst_stratum_empirical_null"] = True
    worst_null_identity = {
        key: (
            int(null.loc[worst_index, key])
            if key in {"seed", "bootstrap_selection_count", "bootstrap_replicates"}
            else str(null.loc[worst_index, key])
        )
        for key in (
            "calibration_candidate_block",
            "world_key",
            "seed",
            "left_modality",
            "right_modality",
            "bootstrap_selection_count",
            "bootstrap_replicates",
        )
    }

    selected_values: list[bool] = []
    probabilities: list[float] = []
    for row in pair.itertuples(index=False):
        selected, probability = _selection_decision(
            nominal_new_proposal=v457.v455.v454.synchronization._as_bool(
                row.nominal_joint_proposal_new_relative_to_v455
            ),
            maximum_local_consensus=v457.v455.v454.synchronization._as_bool(
                row.maximum_joint_local_context_consensus
            ),
            candidate_count=int(row.bootstrap_selection_count),
            null_count=worst_null_count,
        )
        selected_values.append(selected)
        probabilities.append(probability)
    pair["stratum_empirical_null_maximum_selection_count"] = worst_null_count
    pair["stratum_empirical_null_maximum_selection_frequency"] = (
        worst_null_count / v457.BOOTSTRAP_REPLICATES
    )
    pair["strictly_exceeds_stratum_empirical_null"] = pair[
        "bootstrap_selection_count"
    ].astype(int).gt(worst_null_count)
    pair["posterior_probability_exceeds_stratum_worst_null"] = probabilities
    pair["selected_by_joint_context_stratified_stability"] = selected_values

    out_dir.mkdir(parents=True)
    pair_path = out_dir / "joint_context_stratified_pair_audit.tsv"
    null_path = out_dir / "joint_context_stratified_null_audit.tsv"
    pair.to_csv(pair_path, sep="\t", index=False)
    null.to_csv(null_path, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "phase": "target_pretruth_diagnostic",
        "target_block": target_block,
        "target_world_keys": sorted(pair["world_key"].astype(str).unique()),
        "local_context_column": LOCAL_CONTEXT_COLUMN,
        "available_local_context_count": REQUIRED_LOCAL_CONTEXT_COUNT,
        "required_local_context_count": REQUIRED_LOCAL_CONTEXT_COUNT,
        "required_heldout_local_context_count": (
            REQUIRED_HELDOUT_LOCAL_CONTEXT_COUNT
        ),
        "local_context_definition": (
            "full_component_fit_plus_two_disjoint_heldout_background_feature_folds"
        ),
        "bootstrap_replicates": v457.BOOTSTRAP_REPLICATES,
        "minimum_selection_frequency": v457.MINIMUM_SELECTION_FREQUENCY,
        "minimum_selection_frequency_comparison": "strictly_greater_than",
        "posterior_prior": "working_independent_beta_1_1_for_candidate_and_stratum_worst_null",
        "bootstrap_replicates_treated_as_independent_trials": False,
        "posterior_dominance_threshold": v457.POSTERIOR_DOMINANCE_THRESHOLD,
        "stratum_empirical_null_pair_count": len(comparable_null),
        "stratum_empirical_null_maximum_selection_count": worst_null_count,
        "stratum_empirical_null_maximum_selection_frequency": (
            worst_null_count / v457.BOOTSTRAP_REPLICATES
        ),
        "stratum_empirical_null_maximum_identity": worst_null_identity,
        "eligible_target_proposal_count": int(
            (
                pair["nominal_joint_proposal_new_relative_to_v455"].map(
                    v457.v455.v454.synchronization._as_bool
                )
                & pair["maximum_joint_local_context_consensus"].astype(bool)
            ).sum()
        ),
        "selected_target_proposal_count": int(
            pair["selected_by_joint_context_stratified_stability"].astype(bool).sum()
        ),
        "baseline": "target-safe CA-BOSS v4.55 shared-state union",
        "information_boundary": {
            "target_block_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "other_block_truth_used_only_to_identify_null_pairs": True,
            "other_block_positive_labels_used_for_score_fit": False,
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
        },
        "candidate_pairs_may_be_expanded_as_edges": False,
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "source_sha256": {
            path.name: v457.joint.geometry._sha256(path)
            for path in (Path(__file__).resolve(), Path(v457.__file__).resolve())
        },
        "input_sha256": {
            "joint_evidence_manifest.json": v457.joint.geometry._sha256(
                evidence_path / "evidence_manifest.json"
            ),
            "joint_sliced_geometry_pair_evidence.tsv": v457.joint.geometry._sha256(
                evidence_path / "joint_sliced_geometry_pair_evidence.tsv"
            ),
            "v457_diagnostic_manifest.json": v457.joint.geometry._sha256(
                diagnostic_path / "diagnostic_manifest.json"
            ),
            "v457_joint_null_calibrated_pair_audit.tsv": (
                v457.joint.geometry._sha256(
                    diagnostic_path / "joint_null_calibrated_pair_audit.tsv"
                )
            ),
            "v457_joint_calibration_null_frequency_audit.tsv": (
                v457.joint.geometry._sha256(
                    diagnostic_path / "joint_calibration_null_frequency_audit.tsv"
                )
            ),
        },
        "parent_manifest_payload_sha256": {
            "joint_evidence": evidence_manifest["manifest_payload_sha256"],
            "v457_diagnostic": v457_manifest["manifest_payload_sha256"],
        },
        "output_sha256": {
            pair_path.name: v457.joint.geometry._sha256(pair_path),
            null_path.name: v457.joint.geometry._sha256(null_path),
        },
        "formal_pretruth_barrier_satisfied": True,
        "retrospective_rule_development_on_cached_worlds": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "promoted_to_checkpoint": False,
    }
    manifest["manifest_payload_sha256"] = v457.joint.geometry._canonical_sha256(
        manifest
    )
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v457-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = diagnose_target_block(
        evidence_dir=args.evidence_dir,
        v457_diagnostic_dir=args.v457_diagnostic_dir,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
