#!/usr/bin/env python3
"""Cross-block empirical-null selection for CA-BOSS v4.47.

For one target simulation block, this stage verifies reporter-only relaxed
synchronization evidence from all cached worlds, labels eligible null assay
pairs using *other blocks only*, and sets the edge-specific exclusion-margin
threshold to the largest calibration-null value.  It then applies that frozen
threshold to the target block, unions the result with the truth-blind v4.46
guarded rescue, and merges overlapping pairs into unexpanded ambiguity
hyperedges.

The target block's scorer partition is never opened.  Training-world truth is
an explicit simulation-calibration input, not a target-world feature.  This is
retrospective cross-fitting over existing worlds and is not fresh validation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import heldout_synchronized_majority_rescue_v4 as guarded_rescue  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import relaxed_reporter_synchronization_evidence_v4 as relaxed  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as truth_reader  # noqa: E402


SCHEMA = "ca_boss_v4.crossfit_empirical_null_reporter_synchronization.v1"
ALGORITHM = "CA-BOSS v4.47 cross-block empirical-null reporter synchronization"
SCORE_COLUMN = (
    "minimum_heldout_coassigned_pair_exclusion_relative_margin"
)
FIXED_SELECTION_RULE = (
    "for_each_target_seed_block_label_only_other_block_eligible_pairs_using_"
    "simulation_truth_set_the_margin_threshold_to_the_maximum_other_block_"
    "null_minimum_heldout_coassigned_global_assignment_exclusion_margin_"
    "select_target_pairs_strictly_above_that_threshold_union_with_v446_"
    "truth_blind_guarded_pairs_deduplicate_then_merge_pairs_sharing_an_assay_"
    "into_one_unexpanded_ambiguity_hyperedge_and_never_claim_direct_"
    "structure_or_direction"
)
CANDIDATE_COLUMNS = [
    "empirical_null_hyperedge_id",
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
    "calibration_score_name",
    "calibration_null_upper_envelope",
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
PAIR_AUDIT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "left_modality",
    "right_modality",
    "eligible_for_cross_block_empirical_null_calibration",
    "minimum_heldout_coassigned_pair_exclusion_relative_margin",
    "calibration_null_upper_envelope",
    "strictly_exceeds_calibration_null_upper_envelope",
    "selected_by_v446_guarded_rescue",
    "retained_in_union",
    "merged_hyperedge_id",
]


def _as_bool(value: Any) -> bool:
    return synchronization._as_bool(value)


def _verified_manifest(
    directory: Path,
    *,
    manifest_name: str,
    schema: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    path = resolved / manifest_name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != schema:
        raise ValueError(f"Unexpected schema in {path}")
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != synchronization._canonical_sha256(unsigned):
        raise ValueError(f"Manifest payload hash mismatch: {path}")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError(f"Manifest created new data: {path}")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError(f"Manifest lacks a pre-truth barrier: {path}")
    if manifest.get("information_boundary", {}).get("scorer_truth_used") is not False:
        raise ValueError(f"Pre-truth evidence used scorer truth: {path}")
    for name, expected in manifest.get("output_sha256", {}).items():
        output = resolved / str(name)
        if output.is_symlink() or not output.is_file():
            raise FileNotFoundError(output)
        if synchronization._sha256(output) != str(expected):
            raise ValueError(f"Output hash mismatch: {output}")
    return resolved, manifest


def _discover_worlds(
    roots: Sequence[Path],
    *,
    manifest_name: str,
    schema: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    worlds: dict[str, tuple[Path, dict[str, Any]]] = {}
    for root in roots:
        resolved_root = root.expanduser().absolute().resolve(strict=True)
        for manifest_path in sorted(resolved_root.glob(f"*/{manifest_name}")):
            directory, manifest = _verified_manifest(
                manifest_path.parent,
                manifest_name=manifest_name,
                schema=schema,
            )
            key = directory.name
            expected = (
                f"{manifest['block']}__world_seed{int(manifest['seed'])}"
            )
            if key != expected:
                raise ValueError(f"World identity mismatch: {directory}")
            if key in worlds:
                raise ValueError(f"Duplicate world across roots: {key}")
            worlds[key] = (directory, manifest)
    if not worlds:
        raise ValueError("No verified worlds were discovered")
    return worlds


def _truth_assay_pairs(truth_path: Path) -> set[frozenset[str]]:
    return {
        frozenset(str(value) for value in row["modalities"])
        for row in truth_reader._truth_hyperedges(truth_path)
    }


def _pair_key(left: str, right: str) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def _pair_sort_key(pair: frozenset[str]) -> tuple[str, str]:
    values = sorted(pair)
    if len(values) != 2:
        raise ValueError("An assay-pair key must contain exactly two assays")
    return values[0], values[1]


def _connected_pair_components(
    pairs: set[frozenset[str]],
) -> list[set[frozenset[str]]]:
    remaining = set(pairs)
    components: list[set[frozenset[str]]] = []
    while remaining:
        first = min(remaining, key=_pair_sort_key)
        remaining.remove(first)
        component = {first}
        modalities = set(first)
        changed = True
        while changed:
            changed = False
            for pair in sorted(remaining, key=_pair_sort_key):
                if modalities & set(pair):
                    remaining.remove(pair)
                    component.add(pair)
                    modalities.update(pair)
                    changed = True
        components.append(component)
    return sorted(
        components,
        key=lambda component: tuple(
            _pair_sort_key(pair)
            for pair in sorted(component, key=_pair_sort_key)
        ),
    )


def calibrate_target_block(
    *,
    relaxed_roots: Sequence[Path],
    rescue_roots: Sequence[Path],
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Fit an other-block null envelope and select target-block hypotheses."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    relaxed_worlds = _discover_worlds(
        relaxed_roots,
        manifest_name="evidence_manifest.json",
        schema=relaxed.SCHEMA,
    )
    rescue_worlds = _discover_worlds(
        rescue_roots,
        manifest_name="rescue_manifest.json",
        schema=guarded_rescue.SCHEMA,
    )
    if set(relaxed_worlds) != set(rescue_worlds):
        raise ValueError("Relaxed evidence and guarded-rescue world sets differ")
    available_blocks = sorted(
        {str(manifest["block"]) for _, manifest in relaxed_worlds.values()}
    )
    target_block = str(target_block)
    if target_block not in available_blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    calibration_blocks = [
        block for block in available_blocks if block != target_block
    ]
    if len(calibration_blocks) < 2:
        raise ValueError("At least two non-target calibration blocks are required")

    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)
    evidence_frames: dict[str, pd.DataFrame] = {}
    evidence_input_hashes: dict[str, dict[str, str]] = {}
    rescue_input_hashes: dict[str, dict[str, str]] = {}
    training_truth_hashes: dict[str, str] = {}
    calibration_rows: list[pd.DataFrame] = []
    target_keys: list[str] = []
    for key in sorted(relaxed_worlds):
        evidence_dir, evidence_manifest = relaxed_worlds[key]
        rescue_dir, rescue_manifest = rescue_worlds[key]
        if (
            str(evidence_manifest["block"]) != str(rescue_manifest["block"])
            or int(evidence_manifest["seed"]) != int(rescue_manifest["seed"])
        ):
            raise ValueError(f"Upstream world identities disagree: {key}")
        evidence_path = evidence_dir / "relaxed_reporter_synchronization_pairs.tsv"
        frame = pd.read_csv(evidence_path, sep="\t", keep_default_na=False)
        frame[SCORE_COLUMN] = pd.to_numeric(
            frame[SCORE_COLUMN], errors="coerce"
        )
        frame["world_key"] = key
        frame["block"] = str(evidence_manifest["block"])
        frame["seed"] = int(evidence_manifest["seed"])
        evidence_frames[key] = frame
        evidence_input_hashes[key] = {
            "evidence_manifest.json": synchronization._sha256(
                evidence_dir / "evidence_manifest.json"
            ),
            "relaxed_reporter_synchronization_pairs.tsv": (
                synchronization._sha256(evidence_path)
            ),
        }
        rescue_input_hashes[key] = {
            "rescue_manifest.json": synchronization._sha256(
                rescue_dir / "rescue_manifest.json"
            ),
            "heldout_synchronized_majority_candidates.tsv": (
                synchronization._sha256(
                    rescue_dir
                    / "heldout_synchronized_majority_candidates.tsv"
                )
            ),
        }
        block = str(evidence_manifest["block"])
        seed = int(evidence_manifest["seed"])
        if block == target_block:
            target_keys.append(key)
            continue
        truth_path = formal_root / block / f"world_seed{seed}" / "scorer" / "truth_typed_edges.tsv"
        if target_block in truth_path.parts:
            raise RuntimeError("Attempted to open target-block truth")
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        truth_pairs = _truth_assay_pairs(truth_path)
        eligible = frame.loc[
            frame["eligible_for_cross_block_empirical_null_calibration"].map(
                _as_bool
            )
        ].copy()
        eligible["is_null"] = [
            _pair_key(left, right) not in truth_pairs
            for left, right in zip(
                eligible["left_modality"], eligible["right_modality"]
            )
        ]
        calibration_rows.append(eligible)
        training_truth_hashes[key] = synchronization._sha256(truth_path)
    if not target_keys:
        raise ValueError("Target block has no cached worlds")
    calibration = pd.concat(calibration_rows, ignore_index=True)
    nulls = calibration.loc[
        calibration["is_null"].astype(bool)
        & calibration[SCORE_COLUMN].notna()
    ].copy()
    if nulls.empty:
        raise ValueError("No calibration-null margins were available")
    null_summary_rows: list[dict[str, Any]] = []
    for block in calibration_blocks:
        values = nulls.loc[nulls["block"].astype(str).eq(block), SCORE_COLUMN]
        if values.empty:
            raise ValueError(f"Calibration block lacks an eligible null: {block}")
        null_summary_rows.append(
            {
                "calibration_block": block,
                "eligible_null_pair_count": len(values),
                "maximum_null_margin": float(values.max()),
                "mean_null_margin": float(values.mean()),
            }
        )
    threshold = float(nulls[SCORE_COLUMN].max())

    v446_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key in target_keys:
        rescue_dir, _ = rescue_worlds[key]
        guarded = pd.read_csv(
            rescue_dir / "heldout_synchronized_majority_candidates.tsv",
            sep="\t",
            keep_default_na=False,
        )
        for row in guarded.itertuples(index=False):
            if not _as_bool(row.hypothesis_selected):
                continue
            modalities = [
                value for value in str(row.modalities).split(";") if value
            ]
            if len(modalities) != 2:
                raise ValueError("v4.46 guarded candidate is not an assay pair")
            v446_pairs[key].add(frozenset(modalities))

    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for key in sorted(target_keys):
        frame = evidence_frames[key]
        eligible_mask = frame[
            "eligible_for_cross_block_empirical_null_calibration"
        ].map(_as_bool)
        calibrated_mask = eligible_mask & frame[SCORE_COLUMN].gt(threshold)
        calibrated_pairs = {
            _pair_key(row.left_modality, row.right_modality)
            for row in frame.loc[calibrated_mask].itertuples(index=False)
        }
        union_pairs = calibrated_pairs | v446_pairs.get(key, set())
        row_by_pair = {
            _pair_key(row.left_modality, row.right_modality): row
            for row in frame.itertuples(index=False)
        }
        component_id_by_pair: dict[frozenset[str], str] = {}
        for component_index, component in enumerate(
            _connected_pair_components(union_pairs), start=1
        ):
            hyperedge_id = (
                "crossfit_empirical_null_shared_state_"
                f"{component_index:03d}"
            )
            modalities = sorted(
                set().union(*(set(pair) for pair in component))
            )
            pair_payload = [
                list(_pair_sort_key(pair))
                for pair in sorted(component, key=_pair_sort_key)
            ]
            feature_pairs: set[tuple[str, str]] = set()
            selection_audit: list[dict[str, Any]] = []
            sources: set[str] = set()
            for pair in sorted(component, key=_pair_sort_key):
                component_id_by_pair[pair] = hyperedge_id
                row = row_by_pair.get(pair)
                if row is None:
                    raise ValueError("Selected pair lacks relaxed evidence")
                parsed = json.loads(
                    str(row.candidate_cross_assay_feature_pairs_json)
                )
                feature_pairs.update(
                    tuple(sorted((str(left), str(right))))
                    for left, right in parsed
                )
                score = float(getattr(row, SCORE_COLUMN))
                pair_sources: list[str] = []
                if pair in calibrated_pairs:
                    pair_sources.append("cross_block_empirical_null")
                    sources.add("cross_block_empirical_null")
                if pair in v446_pairs.get(key, set()):
                    pair_sources.append("v446_guarded_rescue")
                    sources.add("v446_guarded_rescue")
                selection_audit.append(
                    {
                        "assay_pair": list(_pair_sort_key(pair)),
                        "margin": score,
                        "threshold": threshold,
                        "sources": pair_sources,
                    }
                )
            candidate_rows.append(
                {
                    "empirical_null_hyperedge_id": hyperedge_id,
                    "world_key": key,
                    "block": str(frame["block"].iloc[0]),
                    "seed": int(frame["seed"].iloc[0]),
                    "modalities": ";".join(modalities),
                    "n_modalities": len(modalities),
                    "candidate_assay_pairs_json": json.dumps(
                        pair_payload, separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": len(component),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        sorted(feature_pairs), separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(
                        feature_pairs
                    ),
                    "candidate_pair_selection_audit_json": json.dumps(
                        selection_audit,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "selection_sources": ";".join(sorted(sources)),
                    "calibration_score_name": SCORE_COLUMN,
                    "calibration_null_upper_envelope": threshold,
                    "assay_participation_is_set_valued": len(component) != 1,
                    "exact_assay_participation_claimed": len(component) == 1,
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_crossfit_empirical_null_ambiguity"
                        if len(component) != 1
                        else "selected_as_crossfit_empirical_null_point_hypothesis"
                    ),
                    "hypothesis_type": "cross_assay_shared_state_equivalence",
                    "within_assay_direct_structure_claimed": False,
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                }
            )
        for row in frame.itertuples(index=False):
            pair = _pair_key(row.left_modality, row.right_modality)
            eligible = _as_bool(
                row.eligible_for_cross_block_empirical_null_calibration
            )
            score = float(getattr(row, SCORE_COLUMN))
            exceeds = bool(eligible and np.isfinite(score) and score > threshold)
            guarded = pair in v446_pairs.get(key, set())
            audit_rows.append(
                {
                    "world_key": key,
                    "block": str(row.block),
                    "seed": int(row.seed),
                    "left_modality": str(row.left_modality),
                    "right_modality": str(row.right_modality),
                    "eligible_for_cross_block_empirical_null_calibration": eligible,
                    "minimum_heldout_coassigned_pair_exclusion_relative_margin": score,
                    "calibration_null_upper_envelope": threshold,
                    "strictly_exceeds_calibration_null_upper_envelope": exceeds,
                    "selected_by_v446_guarded_rescue": guarded,
                    "retained_in_union": pair in union_pairs,
                    "merged_hyperedge_id": component_id_by_pair.get(pair, ""),
                }
            )

    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    audits = pd.DataFrame(audit_rows, columns=PAIR_AUDIT_COLUMNS)
    null_summary = pd.DataFrame(null_summary_rows)
    out_dir.mkdir(parents=True)
    outputs = {
        "crossfit_empirical_null_candidates.tsv": candidates,
        "crossfit_empirical_null_pair_audit.tsv": audits,
        "calibration_null_summary.tsv": null_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    manifest = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "target_block": target_block,
        "target_world_count": len(target_keys),
        "target_world_keys": sorted(target_keys),
        "calibration_blocks": calibration_blocks,
        "calibration_world_count": len(training_truth_hashes),
        "calibration_score_name": SCORE_COLUMN,
        "calibration_null_pair_count": len(nulls),
        "calibration_null_upper_envelope": threshold,
        "selected_hyperedge_count": len(candidates),
        "selected_candidate_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "selected_exact_assay_hypothesis_count": int(
            candidates["exact_assay_participation_claimed"].astype(bool).sum()
        ),
        "selected_ambiguity_hypothesis_count": int(
            candidates["assay_participation_is_set_valued"].astype(bool).sum()
        ),
        "maximum_candidate_assay_pair_count": (
            int(candidates["candidate_assay_pair_count"].max())
            if len(candidates)
            else 0
        ),
        "information_boundary": {
            "other_block_simulation_truth_used_for_null_calibration": True,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "modality_identity_used_as_model_feature": False,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
            "only_scalar_edge_specific_margin_calibrated": True,
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
            Path(__file__).name: synchronization._sha256(
                Path(__file__).resolve()
            )
        },
        "input_sha256": {
            "relaxed_evidence_worlds": evidence_input_hashes,
            "v446_guarded_rescue_worlds": rescue_input_hashes,
            "calibration_truth_typed_edges": training_truth_hashes,
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "permutation_synchronization": (
                "Pachauri, Kondor, and Singh, NeurIPS 2013"
            ),
            "multiway_matching": "Tang and Jebara, AISTATS 2017",
            "unpaired_multi_domain_causal_representation": (
                "Sturma, Squires, Drton, and Uhler, NeurIPS 2023"
            ),
        },
    }
    manifest["manifest_payload_sha256"] = synchronization._canonical_sha256(
        manifest
    )
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relaxed-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--rescue-root", type=Path, action="append", required=True
    )
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        relaxed_roots=args.relaxed_root,
        rescue_roots=args.rescue_root,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
