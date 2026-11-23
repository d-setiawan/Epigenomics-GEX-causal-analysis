#!/usr/bin/env python3
"""Promote v4.58 joint local-context-stratified stability evidence.

This truth-blind production stage verifies the frozen joint evidence, the
target-safe v4.55 baseline, and both target-pretruth v4.57/v4.58 diagnostics.
It unions v4.58-selected assay pairs with v4.55 and merges connected conflicts
as unexpanded shared-state ambiguity hyperedges.  It never opens target truth
or emits direct, exact-pair, or directional causal claims.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_joint_context_stratified_stability_v4 as diagnostic  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_joint_context_stratified_stability_rescue.v1"
ALGORITHM = "CA-BOSS v4.58 joint local-context-stratified stability rescue"
FIXED_SELECTION_RULE = (
    diagnostic.FIXED_SELECTION_RULE
    + "_union_only_new_pairs_with_target_safe_v455_merge_connected_pairs_as_"
    "unexpanded_shared_state_ambiguity_hyperedges"
)
CANDIDATE_COLUMNS = [
    "joint_context_stability_hyperedge_id",
    "world_key",
    "block",
    "seed",
    "modalities",
    "n_modalities",
    "candidate_assay_pairs_json",
    "candidate_assay_pair_count",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "candidate_pair_selection_audit_json",
    "selection_sources",
    "bootstrap_replicates",
    "minimum_selection_frequency",
    "required_local_context_count",
    "posterior_dominance_threshold",
    "assay_participation_is_set_valued",
    "exact_assay_participation_claimed",
    "hypothesis_selected",
    "selection_status",
    "hypothesis_type",
    "within_assay_direct_structure_claimed",
    "exact_cross_assay_pair_claimed",
    "cross_assay_direction_claimed",
    "candidate_pairs_may_be_expanded_as_edges",
    "fixed_selection_rule",
]


def _verify_diagnostic(
    directory: Path,
    *,
    target_block: str,
    evidence_path: Path,
    evidence_manifest: Mapping[str, Any],
    v457_path: Path,
    v457_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = diagnostic.v457.joint.geometry._verified_manifest(
        resolved / "diagnostic_manifest.json", diagnostic.SCHEMA
    )
    if str(manifest.get("target_block")) != target_block:
        raise ValueError("v4.58 diagnostic targets another block")
    if manifest.get("fixed_selection_rule") != diagnostic.FIXED_SELECTION_RULE:
        raise ValueError("v4.58 diagnostic selection rule changed")
    if int(manifest.get("required_local_context_count", -1)) != (
        diagnostic.REQUIRED_LOCAL_CONTEXT_COUNT
    ):
        raise ValueError("v4.58 local-context requirement changed")
    if int(manifest.get("required_heldout_local_context_count", -1)) != (
        diagnostic.REQUIRED_HELDOUT_LOCAL_CONTEXT_COUNT
    ):
        raise ValueError("v4.58 held-out context requirement changed")
    if int(manifest.get("bootstrap_replicates", -1)) != (
        diagnostic.v457.BOOTSTRAP_REPLICATES
    ):
        raise ValueError("v4.58 bootstrap count changed")
    if float(manifest.get("minimum_selection_frequency", -1.0)) != (
        diagnostic.v457.MINIMUM_SELECTION_FREQUENCY
    ):
        raise ValueError("v4.58 majority floor changed")
    if manifest.get("minimum_selection_frequency_comparison") != (
        "strictly_greater_than"
    ):
        raise ValueError("v4.58 majority comparison changed")
    if float(manifest.get("posterior_dominance_threshold", -1.0)) != (
        diagnostic.v457.POSTERIOR_DOMINANCE_THRESHOLD
    ):
        raise ValueError("v4.58 posterior guard changed")
    if manifest.get("bootstrap_replicates_treated_as_independent_trials") is not False:
        raise ValueError("v4.58 overstates bootstrap independence")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
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
    }.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.58 diagnostic boundary mismatch: {field}")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.58 diagnostic lacks a pretruth barrier")
    if manifest.get("fresh_locked_certification") is not False:
        raise ValueError("v4.58 diagnostic incorrectly claims fresh certification")
    for name, expected in manifest.get("source_sha256", {}).items():
        path = SCRIPT_DIR / str(name)
        if not path.is_file() or diagnostic.v457.joint.geometry._sha256(path) != str(
            expected
        ):
            raise ValueError(f"v4.58 diagnostic source changed: {name}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if diagnostic.v457.joint.geometry._sha256(path) != str(expected):
            raise ValueError(f"v4.58 diagnostic output hash mismatch: {name}")
    inputs = manifest.get("input_sha256", {})
    expected_inputs = {
        "joint_evidence_manifest.json": diagnostic.v457.joint.geometry._sha256(
            evidence_path / "evidence_manifest.json"
        ),
        "joint_sliced_geometry_pair_evidence.tsv": (
            diagnostic.v457.joint.geometry._sha256(
                evidence_path / "joint_sliced_geometry_pair_evidence.tsv"
            )
        ),
        "v457_diagnostic_manifest.json": diagnostic.v457.joint.geometry._sha256(
            v457_path / "diagnostic_manifest.json"
        ),
        "v457_joint_null_calibrated_pair_audit.tsv": (
            diagnostic.v457.joint.geometry._sha256(
                v457_path / "joint_null_calibrated_pair_audit.tsv"
            )
        ),
        "v457_joint_calibration_null_frequency_audit.tsv": (
            diagnostic.v457.joint.geometry._sha256(
                v457_path / "joint_calibration_null_frequency_audit.tsv"
            )
        ),
    }
    if inputs != expected_inputs:
        raise ValueError("v4.58 diagnostic input bindings changed")
    parents = manifest.get("parent_manifest_payload_sha256", {})
    if parents.get("joint_evidence") != evidence_manifest.get(
        "manifest_payload_sha256"
    ):
        raise ValueError("v4.58 joint-evidence parent changed")
    if parents.get("v457_diagnostic") != v457_manifest.get(
        "manifest_payload_sha256"
    ):
        raise ValueError("v4.58 v4.57 parent changed")
    pair = pd.read_csv(
        resolved / "joint_context_stratified_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if set(pair["block"].astype(str)) != {target_block}:
        raise ValueError("v4.58 pair audit crosses target blocks")
    selected = pair["selected_by_joint_context_stratified_stability"].map(
        diagnostic.v457.v455.v454.synchronization._as_bool
    )
    if int(selected.sum()) != int(manifest.get("selected_target_proposal_count", -1)):
        raise ValueError("v4.58 selected proposal count mismatch")
    if (
        selected
        & ~pair["maximum_joint_local_context_consensus"].map(
            diagnostic.v457.v455.v454.synchronization._as_bool
        )
    ).any():
        raise ValueError("v4.58 selected outside its context stratum")
    return resolved, manifest, pair


def _baseline_components(
    candidates: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parser = diagnostic.v457.v455.v454.v453.v449
    for row in candidates.itertuples(index=False):
        pairs = parser._parse_pair_payload(row.candidate_assay_pairs_json)
        features = parser._parse_feature_pair_payload(
            row.candidate_cross_assay_feature_pairs_json
        )
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.55 candidate assay-pair count mismatch")
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.55 candidate feature-pair count mismatch")
        result[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(str(row.candidate_pair_selection_audit_json)),
                "source": "target_safe_v455",
            }
        )
    return result


def _union_candidates(
    *,
    target_keys: Sequence[str],
    world_frames: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[pd.DataFrame, dict[tuple[str, frozenset[str]], str]]:
    v454 = diagnostic.v457.v455.v454
    parser = v454.v453.v449
    output: list[dict[str, Any]] = []
    ids: dict[tuple[str, frozenset[str]], str] = {}
    for world_key in sorted(target_keys):
        world = world_frames[world_key]
        block = str(world["block"].iloc[0])
        seed = int(world["seed"].iloc[0])
        sources = [dict(value) for value in baseline.get(world_key, [])]
        for selected in selected_rows.get(world_key, []):
            pair = v454._pair(
                selected["left_modality"], selected["right_modality"]
            )
            sources.append(
                {
                    "pairs": {pair},
                    "features": parser._parse_feature_pair_payload(
                        selected["candidate_cross_assay_feature_pairs_json"]
                    ),
                    "audit": [
                        {
                            "assay_pair": list(v454._pair_tuple(pair)),
                            "bootstrap_selection_frequency": float(
                                selected["bootstrap_selection_frequency"]
                            ),
                            "joint_local_mutual_context_count": int(
                                selected[diagnostic.LOCAL_CONTEXT_COLUMN]
                            ),
                            "stratum_empirical_null_maximum_selection_count": int(
                                selected[
                                    "stratum_empirical_null_maximum_selection_count"
                                ]
                            ),
                            "posterior_probability_exceeds_stratum_worst_null": float(
                                selected[
                                    "posterior_probability_exceeds_stratum_worst_null"
                                ]
                            ),
                            "source": "v458_joint_context_stratified_stability",
                        }
                    ],
                    "source": "v458_joint_context_stratified_stability",
                }
            )
        all_pairs = (
            set().union(*(source["pairs"] for source in sources))
            if sources
            else set()
        )
        components = v454.v448.v447._connected_pair_components(all_pairs)
        for index, component in enumerate(components, start=1):
            hyperedge_id = f"joint_context_shared_state_{index:03d}"
            modalities = sorted(set().union(*(set(pair) for pair in component)))
            component_sources = [
                source for source in sources if set(source["pairs"]) & component
            ]
            features = set().union(
                *(source["features"] for source in component_sources)
            )
            audits: list[dict[str, Any]] = []
            source_names: set[str] = set()
            for source in component_sources:
                audit = source["audit"]
                audits.extend(audit if isinstance(audit, list) else [audit])
                source_names.add(str(source["source"]))
            pairs = [
                list(v454._pair_tuple(pair))
                for pair in sorted(component, key=v454._pair_tuple)
            ]
            for pair in component:
                ids[(world_key, pair)] = hyperedge_id
            output.append(
                {
                    "joint_context_stability_hyperedge_id": hyperedge_id,
                    "world_key": world_key,
                    "block": block,
                    "seed": seed,
                    "modalities": ";".join(modalities),
                    "n_modalities": len(modalities),
                    "candidate_assay_pairs_json": json.dumps(
                        pairs, separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": len(component),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        sorted(features), separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(features),
                    "candidate_pair_selection_audit_json": json.dumps(
                        audits, sort_keys=True, separators=(",", ":")
                    ),
                    "selection_sources": ";".join(sorted(source_names)),
                    "bootstrap_replicates": diagnostic.v457.BOOTSTRAP_REPLICATES,
                    "minimum_selection_frequency": (
                        diagnostic.v457.MINIMUM_SELECTION_FREQUENCY
                    ),
                    "required_local_context_count": (
                        diagnostic.REQUIRED_LOCAL_CONTEXT_COUNT
                    ),
                    "posterior_dominance_threshold": (
                        diagnostic.v457.POSTERIOR_DOMINANCE_THRESHOLD
                    ),
                    "assay_participation_is_set_valued": len(component) != 1,
                    "exact_assay_participation_claimed": len(component) == 1,
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_joint_context_stability_ambiguity"
                        if len(component) != 1
                        else "selected_as_joint_context_stability_point_hypothesis"
                    ),
                    "hypothesis_type": "cross_assay_shared_state_equivalence",
                    "within_assay_direct_structure_claimed": False,
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                }
            )
    return pd.DataFrame(output, columns=CANDIDATE_COLUMNS), ids


def calibrate_target_block(
    *,
    evidence_dir: Path,
    v455_calibration_dir: Path,
    v457_diagnostic_dir: Path,
    diagnostic_dir: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    evidence_path, evidence_manifest, evidence = (
        diagnostic.v457._verified_joint_evidence(evidence_dir)
    )
    target_evidence = evidence[
        evidence["block"].astype(str).eq(target_block)
    ].copy()
    target_keys = sorted(target_evidence["world_key"].astype(str).unique())
    if not target_keys:
        raise ValueError(f"Unknown target block: {target_block}")
    baseline_path, baseline_manifest, baseline_candidates = (
        diagnostic.v457._verified_v455_baseline(
            v455_calibration_dir,
            target_block=target_block,
            target_world_keys=target_keys,
        )
    )
    v457_path, v457_manifest, _, _ = diagnostic._verify_v457_diagnostic(
        v457_diagnostic_dir,
        target_block=target_block,
        evidence_path=evidence_path,
        evidence_manifest=evidence_manifest,
    )
    diagnostic_path, diagnostic_manifest, pair_audit = _verify_diagnostic(
        diagnostic_dir,
        target_block=target_block,
        evidence_path=evidence_path,
        evidence_manifest=evidence_manifest,
        v457_path=v457_path,
        v457_manifest=v457_manifest,
    )
    if set(pair_audit["world_key"].astype(str).unique()) != set(target_keys):
        raise ValueError("v4.58 diagnostic and target evidence worlds differ")

    baseline = _baseline_components(baseline_candidates)
    baseline_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs[key].update(component["pairs"])
    computed_baseline = pd.Series(
        [
            diagnostic.v457.v455.v454._pair(
                row.left_modality, row.right_modality
            )
            in baseline_pairs.get(str(row.world_key), set())
            for row in pair_audit.itertuples(index=False)
        ],
        index=pair_audit.index,
        dtype=bool,
    )
    recorded_baseline = pair_audit["selected_by_v455_baseline"].map(
        diagnostic.v457.v455.v454.synchronization._as_bool
    )
    if not computed_baseline.equals(recorded_baseline):
        raise ValueError("v4.58 pair audit and v4.55 baseline disagree")
    selected = pair_audit[
        "selected_by_joint_context_stratified_stability"
    ].map(diagnostic.v457.v455.v454.synchronization._as_bool)
    if (selected & computed_baseline).any():
        raise ValueError("v4.58 attempted to reselect a v4.55 baseline pair")
    pair_audit["newly_selected_by_v458"] = selected
    pair_audit["retained_in_union"] = computed_baseline | selected
    pair_audit["merged_hyperedge_id"] = ""
    selected_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_audit.loc[selected].to_dict(orient="records"):
        selected_rows[str(row["world_key"])].append(row)
    world_frames = {
        str(key): group.reset_index(drop=True)
        for key, group in target_evidence.groupby("world_key", sort=False)
    }
    candidates, merged = _union_candidates(
        target_keys=target_keys,
        world_frames=world_frames,
        baseline=baseline,
        selected_rows=selected_rows,
    )
    for index, row in pair_audit.iterrows():
        pair_audit.at[index, "merged_hyperedge_id"] = merged.get(
            (
                str(row["world_key"]),
                diagnostic.v457.v455.v454._pair(
                    row["left_modality"], row["right_modality"]
                ),
            ),
            "",
        )

    out_dir.mkdir(parents=True)
    outputs = {
        "bootstrap_joint_context_stability_candidates.tsv": candidates,
        "bootstrap_joint_context_stability_pair_audit.tsv": pair_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "target_block": target_block,
        "target_world_count": len(target_keys),
        "target_world_keys": target_keys,
        "baseline_algorithm": "CA-BOSS v4.55 target-safe shared-state union",
        "bootstrap_replicates": diagnostic.v457.BOOTSTRAP_REPLICATES,
        "minimum_selection_frequency": diagnostic.v457.MINIMUM_SELECTION_FREQUENCY,
        "minimum_selection_frequency_comparison": "strictly_greater_than",
        "required_local_context_count": diagnostic.REQUIRED_LOCAL_CONTEXT_COUNT,
        "required_heldout_local_context_count": (
            diagnostic.REQUIRED_HELDOUT_LOCAL_CONTEXT_COUNT
        ),
        "posterior_dominance_threshold": (
            diagnostic.v457.POSTERIOR_DOMINANCE_THRESHOLD
        ),
        "posterior_interpretation": (
            "working_beta_comparison_not_an_independent_trial_posterior"
        ),
        "newly_selected_by_v458_assay_pair_count": int(selected.sum()),
        "selected_union_hyperedge_count": len(candidates),
        "selected_union_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "perfect_matching_assumed": False,
        "fixed_shared_relationship_count_assumed": False,
        "supervised_positive_labels_used_for_score_fit": False,
        "other_block_truth_used_only_to_identify_calibration_nulls": True,
        "information_boundary": {
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
        },
        "target_specific_pretruth_barrier_satisfied": True,
        "formal_pretruth_barrier_satisfied": True,
        "retrospective_rule_development_on_cached_worlds": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "source_sha256": {
            path.name: diagnostic.v457.joint.geometry._sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(diagnostic.__file__).resolve(),
                Path(diagnostic.v457.__file__).resolve(),
            )
        },
        "input_sha256": {
            "joint_evidence_manifest.json": diagnostic.v457.joint.geometry._sha256(
                evidence_path / "evidence_manifest.json"
            ),
            "v455_calibration_manifest.json": (
                diagnostic.v457.joint.geometry._sha256(
                    baseline_path / "calibration_manifest.json"
                )
            ),
            "v457_diagnostic_manifest.json": (
                diagnostic.v457.joint.geometry._sha256(
                    v457_path / "diagnostic_manifest.json"
                )
            ),
            "v458_diagnostic_manifest.json": (
                diagnostic.v457.joint.geometry._sha256(
                    diagnostic_path / "diagnostic_manifest.json"
                )
            ),
        },
        "parent_manifest_payload_sha256": {
            "joint_evidence": evidence_manifest["manifest_payload_sha256"],
            "v455": baseline_manifest["manifest_payload_sha256"],
            "v457_diagnostic": v457_manifest["manifest_payload_sha256"],
            "v458_diagnostic": diagnostic_manifest["manifest_payload_sha256"],
        },
        "output_sha256": {
            name: diagnostic.v457.joint.geometry._sha256(out_dir / name)
            for name in outputs
        },
    }
    manifest["manifest_payload_sha256"] = (
        diagnostic.v457.joint.geometry._canonical_sha256(manifest)
    )
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v455-calibration-dir", type=Path, required=True)
    parser.add_argument("--v457-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = calibrate_target_block(
        evidence_dir=args.evidence_dir,
        v455_calibration_dir=args.v455_calibration_dir,
        v457_diagnostic_dir=args.v457_diagnostic_dir,
        diagnostic_dir=args.diagnostic_dir,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
