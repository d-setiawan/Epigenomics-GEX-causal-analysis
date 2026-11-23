#!/usr/bin/env python3
"""Audit a dual-null rescue for one-context joint-geometry disagreement.

CA-BOSS v4.58 deliberately admits only reporter pairs that are strict local
mutual nearest neighbours in the full fit and both disjoint background-feature
folds.  This cached-only v4.59 diagnostic considers the adjacent, exactly
two-of-three context stratum without weakening v4.58.  A candidate must beat
both (1) the most stable null in the same two-context stratum and (2) the
unstratified global worst-null count from v4.57.  Working Beta(1, 1)
uncertainty is evaluated against the exchangeable same-stratum null; the
global null remains a hard safety ceiling.

The stage consumes hash-bound pretruth evidence and diagnostics, never opens
target truth, and emits shared-state-equivalence evidence only.  It does not
assume a perfect matching, a fixed relationship count, a cell correspondence,
an exact feature pair, or a cross-assay direction.
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

import bootstrap_joint_context_stratified_stability_rescue_v4 as v458  # noqa: E402


SCHEMA = "ca_boss_v4.dual_null_disagreement_stability_diagnostic.v1"
ALGORITHM = "CA-BOSS v4.59 dual-null one-context-disagreement diagnostic"
REQUIRED_LOCAL_CONTEXT_COUNT = 2
AVAILABLE_LOCAL_CONTEXT_COUNT = 3
FIXED_SELECTION_RULE = (
    "start_only_from_v457_nominal_joint_geometry_pairs_new_relative_to_target_"
    "safe_v455_require_exactly_two_of_three_strict_local_mutual_nearest_"
    "component_score_contexts_require_strictly_more_than_50_of_100_whole_"
    "world_refits_require_the_candidate_count_to_strictly_exceed_both_the_"
    "same_two_context_stratum_worst_null_count_and_the_unstratified_v457_"
    "worst_null_count_require_at_least_0_975_working_beta_1_1_dominance_"
    "probability_against_the_same_stratum_null_never_claim_direct_structure_"
    "exact_pair_or_direction"
)


def _selection_decision(
    *,
    nominal_new_proposal: bool,
    local_context_count: int,
    candidate_count: int,
    stratum_null_count: int,
    global_null_count: int,
) -> tuple[bool, float]:
    """Apply the fixed dual-reference decision without target labels."""

    probability = (
        v458.diagnostic.v457.v455_diagnostic._posterior_dominance_probability(
            candidate_count,
            stratum_null_count,
            v458.diagnostic.v457.BOOTSTRAP_REPLICATES,
        )
    )
    selected = bool(
        nominal_new_proposal
        and local_context_count == REQUIRED_LOCAL_CONTEXT_COUNT
        and candidate_count
        > int(
            v458.diagnostic.v457.MINIMUM_SELECTION_FREQUENCY
            * v458.diagnostic.v457.BOOTSTRAP_REPLICATES
        )
        and candidate_count > stratum_null_count
        and candidate_count > global_null_count
        and probability
        >= v458.diagnostic.v457.POSTERIOR_DOMINANCE_THRESHOLD
    )
    return selected, probability


def diagnose_target_block(
    *,
    evidence_dir: Path,
    v457_diagnostic_dir: Path,
    v458_diagnostic_dir: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Create one target-separated, truth-blind v4.59 diagnostic."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    evidence_path, evidence_manifest, evidence = (
        v458.diagnostic.v457._verified_joint_evidence(evidence_dir)
    )
    target_keys = sorted(
        evidence.loc[
            evidence["block"].astype(str).eq(target_block), "world_key"
        ].astype(str).unique()
    )
    if not target_keys:
        raise ValueError(f"Unknown target block: {target_block}")
    v457_path, v457_manifest, _, _ = (
        v458.diagnostic._verify_v457_diagnostic(
            v457_diagnostic_dir,
            target_block=target_block,
            evidence_path=evidence_path,
            evidence_manifest=evidence_manifest,
        )
    )
    v458_path, v458_manifest, pair = v458._verify_diagnostic(
        v458_diagnostic_dir,
        target_block=target_block,
        evidence_path=evidence_path,
        evidence_manifest=evidence_manifest,
        v457_path=v457_path,
        v457_manifest=v457_manifest,
    )
    if set(pair["world_key"].astype(str).unique()) != set(target_keys):
        raise ValueError("v4.58 diagnostic and target evidence worlds differ")

    null_path = v458_path / "joint_context_stratified_null_audit.tsv"
    null = pd.read_csv(null_path, sep="\t", keep_default_na=False)
    context_column = v458.diagnostic.LOCAL_CONTEXT_COLUMN
    context = pd.to_numeric(null[context_column], errors="raise")
    if context.isna().any() or (~context.between(0, AVAILABLE_LOCAL_CONTEXT_COUNT)).any():
        raise ValueError("Null local-context counts are outside 0..3")
    if (context % 1 != 0).any():
        raise ValueError("Null local-context counts must be integers")
    null["in_two_context_disagreement_stratum"] = context.astype(int).eq(
        REQUIRED_LOCAL_CONTEXT_COUNT
    )
    comparable = null[null["in_two_context_disagreement_stratum"]].copy()
    if comparable.empty:
        raise ValueError("No empirical null lies in the two-context stratum")
    worst_index = comparable["bootstrap_selection_count"].astype(int).idxmax()
    stratum_null_count = int(null.loc[worst_index, "bootstrap_selection_count"])
    null["is_worst_two_context_stratum_null"] = False
    null.loc[worst_index, "is_worst_two_context_stratum_null"] = True

    global_counts = pd.to_numeric(
        pair["empirical_null_maximum_selection_count"], errors="raise"
    ).astype(int)
    if global_counts.nunique() != 1:
        raise ValueError("v4.57 global null count is not target-block constant")
    global_null_count = int(global_counts.iloc[0])
    if global_null_count != int(
        v457_manifest["empirical_null_maximum_selection_count"]
    ):
        raise ValueError("v4.57 manifest and pair audit global null counts differ")

    selected_values: list[bool] = []
    probabilities: list[float] = []
    for row in pair.itertuples(index=False):
        selected, probability = _selection_decision(
            nominal_new_proposal=(
                v458.diagnostic.v457.v455.v454.synchronization._as_bool(
                    row.nominal_joint_proposal_new_relative_to_v455
                )
            ),
            local_context_count=int(getattr(row, context_column)),
            candidate_count=int(row.bootstrap_selection_count),
            stratum_null_count=stratum_null_count,
            global_null_count=global_null_count,
        )
        selected_values.append(selected)
        probabilities.append(probability)
    pair["two_context_stratum_null_maximum_selection_count"] = stratum_null_count
    pair["two_context_stratum_null_maximum_selection_frequency"] = (
        stratum_null_count / v458.diagnostic.v457.BOOTSTRAP_REPLICATES
    )
    pair["unstratified_global_null_maximum_selection_count"] = global_null_count
    pair["strictly_exceeds_two_context_stratum_null"] = pair[
        "bootstrap_selection_count"
    ].astype(int).gt(stratum_null_count)
    pair["strictly_exceeds_unstratified_global_null"] = pair[
        "bootstrap_selection_count"
    ].astype(int).gt(global_null_count)
    pair["posterior_probability_exceeds_two_context_stratum_null"] = probabilities
    pair["selected_by_dual_null_disagreement_stability"] = selected_values

    worst_identity = {
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

    out_dir.mkdir(parents=True)
    pair_path = out_dir / "dual_null_disagreement_pair_audit.tsv"
    null_out = out_dir / "dual_null_disagreement_null_audit.tsv"
    pair.to_csv(pair_path, sep="\t", index=False)
    null.to_csv(null_out, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "phase": "target_pretruth_diagnostic",
        "target_block": target_block,
        "target_world_keys": target_keys,
        "available_local_context_count": AVAILABLE_LOCAL_CONTEXT_COUNT,
        "required_local_context_count": REQUIRED_LOCAL_CONTEXT_COUNT,
        "required_heldout_local_context_count": 1,
        "local_context_definition": (
            "exactly_two_of_full_fit_and_two_disjoint_heldout_background_feature_folds"
        ),
        "bootstrap_replicates": v458.diagnostic.v457.BOOTSTRAP_REPLICATES,
        "minimum_selection_frequency": (
            v458.diagnostic.v457.MINIMUM_SELECTION_FREQUENCY
        ),
        "minimum_selection_frequency_comparison": "strictly_greater_than",
        "posterior_prior": (
            "working_independent_beta_1_1_for_candidate_and_same_stratum_worst_null"
        ),
        "bootstrap_replicates_treated_as_independent_trials": False,
        "posterior_dominance_threshold": (
            v458.diagnostic.v457.POSTERIOR_DOMINANCE_THRESHOLD
        ),
        "two_context_stratum_empirical_null_pair_count": len(comparable),
        "two_context_stratum_null_maximum_selection_count": stratum_null_count,
        "two_context_stratum_null_maximum_selection_frequency": (
            stratum_null_count / v458.diagnostic.v457.BOOTSTRAP_REPLICATES
        ),
        "two_context_stratum_null_maximum_identity": worst_identity,
        "unstratified_global_null_maximum_selection_count": global_null_count,
        "unstratified_global_null_maximum_selection_frequency": (
            global_null_count / v458.diagnostic.v457.BOOTSTRAP_REPLICATES
        ),
        "eligible_target_proposal_count": int(
            (
                pair["nominal_joint_proposal_new_relative_to_v455"].map(
                    v458.diagnostic.v457.v455.v454.synchronization._as_bool
                )
                & pd.to_numeric(pair[context_column], errors="raise")
                .astype(int)
                .eq(REQUIRED_LOCAL_CONTEXT_COUNT)
            ).sum()
        ),
        "selected_target_proposal_count": int(
            pair["selected_by_dual_null_disagreement_stability"]
            .astype(bool)
            .sum()
        ),
        "baseline": "target-safe CA-BOSS v4.58 shared-state union",
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
        "perfect_matching_assumed": False,
        "fixed_shared_relationship_count_assumed": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "source_sha256": {
            Path(__file__).name: v458.diagnostic.v457.joint.geometry._sha256(
                Path(__file__).resolve()
            )
        },
        "input_sha256": {
            "joint_evidence_manifest.json": (
                v458.diagnostic.v457.joint.geometry._sha256(
                    evidence_path / "evidence_manifest.json"
                )
            ),
            "v457_diagnostic_manifest.json": (
                v458.diagnostic.v457.joint.geometry._sha256(
                    v457_path / "diagnostic_manifest.json"
                )
            ),
            "v458_diagnostic_manifest.json": (
                v458.diagnostic.v457.joint.geometry._sha256(
                    v458_path / "diagnostic_manifest.json"
                )
            ),
            "v458_pair_audit.tsv": (
                v458.diagnostic.v457.joint.geometry._sha256(
                    v458_path / "joint_context_stratified_pair_audit.tsv"
                )
            ),
            "v458_null_audit.tsv": (
                v458.diagnostic.v457.joint.geometry._sha256(null_path)
            ),
        },
        "parent_manifest_payload_sha256": {
            "joint_evidence": evidence_manifest["manifest_payload_sha256"],
            "v457_diagnostic": v457_manifest["manifest_payload_sha256"],
            "v458_diagnostic": v458_manifest["manifest_payload_sha256"],
        },
        "output_sha256": {
            pair_path.name: v458.diagnostic.v457.joint.geometry._sha256(pair_path),
            null_out.name: v458.diagnostic.v457.joint.geometry._sha256(null_out),
        },
        "formal_pretruth_barrier_satisfied": True,
        "retrospective_rule_development_on_cached_worlds": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "promoted_to_checkpoint": False,
    }
    manifest["manifest_payload_sha256"] = (
        v458.diagnostic.v457.joint.geometry._canonical_sha256(manifest)
    )
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v457-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--v458-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = diagnose_target_block(
        evidence_dir=args.evidence_dir,
        v457_diagnostic_dir=args.v457_diagnostic_dir,
        v458_diagnostic_dir=args.v458_diagnostic_dir,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
