#!/usr/bin/env python3
"""Promote target-safe null-calibrated stability evidence as CA-BOSS v4.55.

The stage verifies the v4.54 evidence/calibration chain and one target-sealed
v4.55 diagnostic.  It admits only nominal W2 proposals selected by the fixed
strict-majority, empirical-null, and working-Beta dominance rule, then unions
them with target-safe v4.54 hypotheses.  Connected conflicts remain one
unexpanded shared-state ambiguity hyperedge.  This stage never opens truth.
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

import bootstrap_wasserstein_geometry_rescue_v4 as v454  # noqa: E402
import diagnose_null_calibrated_stability_v4 as diagnostic  # noqa: E402
import score_bootstrap_wasserstein_geometry_rescue_v4 as v454_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_null_calibrated_stability_rescue.v1"
ALGORITHM = "CA-BOSS v4.55 empirical-null-calibrated stability rescue"
FIXED_SELECTION_RULE = (
    diagnostic.FIXED_SELECTION_RULE
    + "_union_only_new_pairs_with_target_safe_v454_merge_connected_pairs_as_"
    "unexpanded_shared_state_ambiguity_hyperedges_never_claim_direct_"
    "structure_or_direction"
)
CANDIDATE_COLUMNS = [
    "null_calibrated_hyperedge_id",
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
    v454_calibration_dir: Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = v454.diagnostic._verified_manifest(
        resolved / "diagnostic_manifest.json", diagnostic.SCHEMA
    )
    if str(manifest.get("target_block")) != target_block:
        raise ValueError("v4.55 diagnostic targets another block")
    if manifest.get("fixed_selection_rule") != diagnostic.FIXED_SELECTION_RULE:
        raise ValueError("v4.55 diagnostic selection rule changed")
    if int(manifest.get("bootstrap_replicates", -1)) != (
        diagnostic.BOOTSTRAP_REPLICATES
    ):
        raise ValueError("v4.55 diagnostic bootstrap count changed")
    if float(manifest.get("minimum_selection_frequency", -1.0)) != (
        diagnostic.MINIMUM_SELECTION_FREQUENCY
    ):
        raise ValueError("v4.55 diagnostic majority floor changed")
    if manifest.get("minimum_selection_frequency_comparison") != (
        "strictly_greater_than"
    ):
        raise ValueError("v4.55 diagnostic majority comparison changed")
    if float(manifest.get("posterior_dominance_threshold", -1.0)) != (
        diagnostic.POSTERIOR_DOMINANCE_THRESHOLD
    ):
        raise ValueError("v4.55 diagnostic posterior guard changed")
    if manifest.get("bootstrap_replicates_treated_as_independent_trials") is not False:
        raise ValueError("v4.55 diagnostic overstates bootstrap independence")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
        "target_block_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "other_block_truth_used_only_to_identify_null_pairs": True,
        "other_block_positive_labels_used_for_score_fit": False,
        "existing_cached_worlds_only": True,
        "new_worlds_or_datasets_created": False,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "simulator_latent_state_used": False,
        "external_features_or_priors_used": False,
        "modality_identity_used_as_model_feature": False,
        "candidate_feature_identity_used_as_model_feature": False,
    }.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.55 diagnostic boundary mismatch: {field}")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.55 diagnostic lacks the pretruth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.55 diagnostic created new data")
    source_hashes = manifest.get("source_sha256", {})
    for module in (diagnostic, v454):
        path = Path(module.__file__).resolve()
        if source_hashes.get(path.name) != v454.diagnostic._sha256(path):
            raise ValueError(f"v4.55 diagnostic source changed: {path.name}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if v454.diagnostic._sha256(path) != str(expected):
            raise ValueError(f"v4.55 diagnostic output hash mismatch: {name}")
    expected_v454_hash = v454.diagnostic._sha256(
        v454_calibration_dir / "calibration_manifest.json"
    )
    if manifest.get("input_sha256", {}).get(
        "v454_calibration_manifest.json"
    ) != expected_v454_hash:
        raise ValueError("v4.55 diagnostic and v4.54 calibration disagree")
    table = pd.read_csv(
        resolved / "null_calibrated_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if set(table["block"].astype(str)) != {target_block}:
        raise ValueError("v4.55 diagnostic pair audit crosses target blocks")
    selected = table["selected_by_null_calibrated_stability"].map(
        v454.synchronization._as_bool
    )
    if int(selected.sum()) != int(manifest["selected_target_proposal_count"]):
        raise ValueError("v4.55 selected proposal count mismatch")
    if (
        selected
        & ~table["nominal_v454_new_proposal"].map(
            v454.synchronization._as_bool
        )
    ).any():
        raise ValueError("v4.55 selected a non-nominal proposal")
    return resolved, manifest, table


def _baseline_components(
    candidates: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        pairs = v454.v453.v449._parse_pair_payload(
            row.candidate_assay_pairs_json
        )
        features = v454.v453.v449._parse_feature_pair_payload(
            row.candidate_cross_assay_feature_pairs_json
        )
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.54 candidate assay-pair count mismatch")
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.54 candidate feature-pair count mismatch")
        result[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(
                    str(row.candidate_pair_selection_audit_json)
                ),
                "source": "target_safe_v454",
            }
        )
    return result


def _union_candidates(
    *,
    target_keys: Sequence[str],
    world_frames: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    stable_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[pd.DataFrame, dict[tuple[str, frozenset[str]], str]]:
    output: list[dict[str, Any]] = []
    ids: dict[tuple[str, frozenset[str]], str] = {}
    for world_key in sorted(target_keys):
        world = world_frames[world_key]
        block = str(world["block"].iloc[0])
        seed = int(world["seed"].iloc[0])
        sources = [dict(value) for value in baseline.get(world_key, [])]
        for selected in stable_rows.get(world_key, []):
            pair = v454._pair(
                selected["left_modality"], selected["right_modality"]
            )
            sources.append(
                {
                    "pairs": {pair},
                    "features": v454.v453.v449._parse_feature_pair_payload(
                        selected["candidate_cross_assay_feature_pairs_json"]
                    ),
                    "audit": [
                        {
                            "assay_pair": list(v454._pair_tuple(pair)),
                            "bootstrap_selection_frequency": float(
                                selected["bootstrap_selection_frequency"]
                            ),
                            "empirical_null_maximum_selection_count": int(
                                selected[
                                    "empirical_null_maximum_selection_count"
                                ]
                            ),
                            "posterior_probability_exceeds_worst_null": float(
                                selected[
                                    "posterior_probability_exceeds_worst_null"
                                ]
                            ),
                            "source": "v455_null_calibrated_stability",
                        }
                    ],
                    "source": "v455_null_calibrated_stability",
                }
            )
        all_pairs = (
            set().union(*(source["pairs"] for source in sources))
            if sources
            else set()
        )
        for index, component in enumerate(
            v454.v448.v447._connected_pair_components(all_pairs), start=1
        ):
            hyperedge_id = f"null_calibrated_shared_state_{index:03d}"
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
                    "null_calibrated_hyperedge_id": hyperedge_id,
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
                    "bootstrap_replicates": diagnostic.BOOTSTRAP_REPLICATES,
                    "minimum_selection_frequency": (
                        diagnostic.MINIMUM_SELECTION_FREQUENCY
                    ),
                    "posterior_dominance_threshold": (
                        diagnostic.POSTERIOR_DOMINANCE_THRESHOLD
                    ),
                    "assay_participation_is_set_valued": len(component) != 1,
                    "exact_assay_participation_claimed": len(component) == 1,
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_null_calibrated_stability_ambiguity"
                        if len(component) != 1
                        else "selected_as_null_calibrated_stability_point_hypothesis"
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
    v453_calibration_dir: Path,
    v454_calibration_dir: Path,
    diagnostic_dir: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    evidence_path, evidence_manifest, evidence = v454._verified_evidence(
        evidence_dir
    )
    blocks = sorted(evidence["block"].astype(str).unique())
    if target_block not in blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    target_keys = sorted(
        evidence.loc[
            evidence["block"].astype(str).eq(target_block), "world_key"
        ].astype(str).unique()
    )
    if not target_keys:
        raise ValueError("Target block has no evidence worlds")
    v453_path, v453_manifest, _ = v454._verified_v453(
        v453_calibration_dir,
        target_block=target_block,
        evidence_manifest=evidence_manifest,
    )
    first_seed = int(target_keys[0].rsplit("world_seed", 1)[1])
    v454_path, v454_manifest, _, _ = v454_scorer._verify_calibration(
        v454_calibration_dir,
        bridge_provenance={"block": target_block, "seed": first_seed},
        evidence_dir=evidence_path,
        evidence_manifest=evidence_manifest,
        v453_dir=v453_path,
        v453_manifest=v453_manifest,
    )
    v454_candidates = pd.read_csv(
        v454_path / "bootstrap_wasserstein_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if len(v454_candidates) and not v454_candidates["world_key"].astype(
        str
    ).isin(target_keys).all():
        raise ValueError("v4.54 baseline candidates cross target blocks")
    diagnostic_path, diagnostic_manifest, pair_audit = _verify_diagnostic(
        diagnostic_dir,
        target_block=target_block,
        v454_calibration_dir=v454_path,
    )
    if set(map(str, diagnostic_manifest.get("target_world_keys", []))) != set(
        target_keys
    ):
        raise ValueError("v4.55 diagnostic and evidence target worlds differ")

    identities = ["world_key", "left_modality", "right_modality"]
    feature_payload = evidence.loc[
        evidence["block"].astype(str).eq(target_block),
        identities + ["candidate_cross_assay_feature_pairs_json"],
    ]
    if feature_payload.duplicated(identities).any():
        raise ValueError("Geometry evidence contains duplicate target pairs")
    pair_audit = pair_audit.merge(
        feature_payload,
        on=identities,
        how="left",
        validate="one_to_one",
    )
    if pair_audit["candidate_cross_assay_feature_pairs_json"].isna().any():
        raise ValueError("v4.55 diagnostic pair lacks feature-pair evidence")

    baseline = _baseline_components(v454_candidates)
    baseline_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs[key].update(component["pairs"])
    selected_mask = pair_audit[
        "selected_by_null_calibrated_stability"
    ].map(v454.synchronization._as_bool)
    baseline_mask = pd.Series(
        [
            v454._pair(row.left_modality, row.right_modality)
            in baseline_pairs.get(str(row.world_key), set())
            for row in pair_audit.itertuples(index=False)
        ],
        index=pair_audit.index,
        dtype=bool,
    )
    pair_audit["selected_by_target_safe_v454"] = baseline_mask
    pair_audit["newly_selected_by_v455"] = selected_mask & ~baseline_mask
    pair_audit["retained_in_union"] = baseline_mask | selected_mask
    pair_audit["merged_hyperedge_id"] = ""
    stable_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_audit.loc[pair_audit["newly_selected_by_v455"]].to_dict(
        orient="records"
    ):
        stable_rows[str(row["world_key"])].append(row)
    world_frames = {
        str(key): group.reset_index(drop=True)
        for key, group in evidence.loc[
            evidence["block"].astype(str).eq(target_block)
        ].groupby("world_key", sort=False)
    }
    candidates, merged = _union_candidates(
        target_keys=target_keys,
        world_frames=world_frames,
        baseline=baseline,
        stable_rows=stable_rows,
    )
    for index, row in pair_audit.iterrows():
        pair_audit.at[index, "merged_hyperedge_id"] = merged.get(
            (
                str(row["world_key"]),
                v454._pair(row["left_modality"], row["right_modality"]),
            ),
            "",
        )

    out_dir.mkdir(parents=True)
    outputs = {
        "bootstrap_null_calibrated_candidates.tsv": candidates,
        "bootstrap_null_calibrated_pair_audit.tsv": pair_audit,
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
        "baseline_algorithm": "CA-BOSS v4.54 target-safe stable union",
        "bootstrap_replicates": diagnostic.BOOTSTRAP_REPLICATES,
        "minimum_selection_frequency": diagnostic.MINIMUM_SELECTION_FREQUENCY,
        "minimum_selection_frequency_comparison": "strictly_greater_than",
        "posterior_dominance_threshold": (
            diagnostic.POSTERIOR_DOMINANCE_THRESHOLD
        ),
        "posterior_interpretation": (
            "working_beta_comparison_not_an_independent_trial_posterior"
        ),
        "newly_selected_by_v455_assay_pair_count": int(
            pair_audit["newly_selected_by_v455"].sum()
        ),
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
            path.name: v454.diagnostic._sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(diagnostic.__file__).resolve(),
                Path(v454.__file__).resolve(),
                Path(v454_scorer.__file__).resolve(),
            )
        },
        "input_sha256": {
            "geometry_evidence": {
                "evidence_manifest.json": v454.diagnostic._sha256(
                    evidence_path / "evidence_manifest.json"
                ),
                "quantile_geometry_pair_evidence.tsv": v454.diagnostic._sha256(
                    evidence_path / "quantile_geometry_pair_evidence.tsv"
                ),
            },
            "target_safe_v453_calibration_manifest.json": (
                v454.diagnostic._sha256(
                    v453_path / "calibration_manifest.json"
                )
            ),
            "target_safe_v454_calibration_manifest.json": (
                v454.diagnostic._sha256(
                    v454_path / "calibration_manifest.json"
                )
            ),
            "v455_diagnostic_manifest.json": v454.diagnostic._sha256(
                diagnostic_path / "diagnostic_manifest.json"
            ),
        },
        "parent_manifest_payload_sha256": {
            "v453": v453_manifest["manifest_payload_sha256"],
            "v454": v454_manifest["manifest_payload_sha256"],
            "v455_diagnostic": diagnostic_manifest["manifest_payload_sha256"],
        },
        "output_sha256": {
            name: v454.diagnostic._sha256(out_dir / name) for name in outputs
        },
    }
    manifest["manifest_payload_sha256"] = v454._canonical_sha256(manifest)
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v453-calibration-dir", type=Path, required=True)
    parser.add_argument("--v454-calibration-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = calibrate_target_block(
        evidence_dir=args.evidence_dir,
        v453_calibration_dir=args.v453_calibration_dir,
        v454_calibration_dir=args.v454_calibration_dir,
        diagnostic_dir=args.diagnostic_dir,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
