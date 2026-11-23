#!/usr/bin/env python3
"""Diagnose empirical-null calibration of v4.54 bootstrap frequencies.

This cached-world-only CA-BOSS diagnostic replaces the fixed 0.80 stability
cutoff with a nested comparison against stability frequencies of null pairs.
For a sealed target block T and each calibration-candidate block C, C is scored
in three folds whose null labels come only from blocks outside {T, C}; T may
contribute unlabeled feature ranks but its truth is never opened.  The largest
null selection count across C is the target's empirical-null reference.

A target proposal must have strict-majority bootstrap support, exceed that
reference, and have at least 0.975 working-posterior probability that its Bernoulli
selection probability exceeds the worst null under independent Beta(1, 1)
priors.  This is a diagnostic only; it does not alter the v4.54 checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_wasserstein_geometry_rescue_v4 as v454  # noqa: E402
import diagnose_quantile_geometry_rescue_v4 as diagnostic  # noqa: E402


SCHEMA = "ca_boss_v4.null_calibrated_stability_diagnostic.v1"
ALGORITHM = "CA-BOSS v4.55 nested empirical-null bootstrap-frequency diagnostic"
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_BASE_SEED = 455000
MINIMUM_SELECTION_FREQUENCY = 0.50
POSTERIOR_DOMINANCE_THRESHOLD = 0.975
FIXED_SELECTION_RULE = (
    "start_from_a_nominal_v454_w2_proposal_compare_its_100_refit_selection_"
    "count_with_the_maximum_count_of_every_null_pair_in_four_non_target_"
    "candidate_blocks_estimate_each_null_count_without_opening_target_truth_"
    "allow_target_evidence_only_as_unlabeled_empirical_rank_training_data_"
    "require_strictly_more_than_50_of_100_refits_strictly_exceed_the_worst_"
    "null_count_and_have_at_least_0_975_working_beta_1_1_posterior_"
    "probability_of_a_larger_selection_probability"
)


def _posterior_dominance_probability(
    candidate_count: int,
    null_count: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    if not 0 <= candidate_count <= replicates:
        raise ValueError("Candidate count is outside the binomial range")
    if not 0 <= null_count <= replicates:
        raise ValueError("Null count is outside the binomial range")
    candidate_a = candidate_count + 1
    candidate_b = replicates - candidate_count + 1
    null_a = null_count + 1
    null_b = replicates - null_count + 1
    probability, _ = quad(
        lambda value: beta.pdf(value, candidate_a, candidate_b)
        * beta.cdf(value, null_a, null_b),
        0.0,
        1.0,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )
    return float(np.clip(probability, 0.0, 1.0))


def _nested_selection(
    *,
    target: pd.DataFrame,
    calibration_blocks: Sequence[str],
    training_pool_blocks: Sequence[str],
    block_keys: Mapping[str, Sequence[str]],
    world_frames: Mapping[str, pd.DataFrame],
    truth_pairs: Mapping[str, set[frozenset[str]]],
    sampled_training_keys: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    margins: list[np.ndarray] = []
    mutual: list[np.ndarray] = []
    null_maxima: dict[str, float] = {}
    for calibration_block in calibration_blocks:
        training_blocks = [
            block for block in training_pool_blocks if block != calibration_block
        ]
        training_keys: list[str] = []
        for training_block in training_blocks:
            keys = (
                block_keys[training_block]
                if sampled_training_keys is None
                else sampled_training_keys[(calibration_block, training_block)]
            )
            training_keys.extend(keys)
        training = pd.concat(
            [world_frames[key] for key in training_keys], ignore_index=True
        )
        calibration = pd.concat(
            [world_frames[key] for key in block_keys[calibration_block]],
            ignore_index=True,
        )
        target_score = v454._copula_score(training, target)
        target_margin, target_mutual = v454.v453._mutual_nearest_absolute_margin(
            target, target_score
        )
        calibration_score = v454._copula_score(training, calibration)
        calibration_margin, calibration_mutual = (
            v454.v453._mutual_nearest_absolute_margin(
                calibration, calibration_score
            )
        )
        null = (~v454._truth_labels(calibration, truth_pairs)) & calibration_mutual
        if not null.any():
            raise ValueError(
                f"Calibration block {calibration_block} has no mutual-nearest null"
            )
        margins.append(target_margin)
        mutual.append(target_mutual)
        null_maxima[calibration_block] = float(calibration_margin[null].max())
    stacked_margins = np.vstack(margins).T
    stacked_mutual = np.vstack(mutual).T
    selected = stacked_mutual.all(axis=1) & (
        stacked_margins.min(axis=1) > max(null_maxima.values())
    )
    return selected, stacked_margins, stacked_mutual, null_maxima


def _bootstrap_counts(
    *,
    target: pd.DataFrame,
    calibration_blocks: Sequence[str],
    training_pool_blocks: Sequence[str],
    block_keys: Mapping[str, Sequence[str]],
    world_frames: Mapping[str, pd.DataFrame],
    truth_pairs: Mapping[str, set[frozenset[str]]],
    seed_prefix: Sequence[int],
) -> np.ndarray:
    counts = np.zeros(len(target), dtype=np.int64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled: dict[tuple[str, str], list[str]] = {}
        for fold_index, calibration_block in enumerate(calibration_blocks):
            training_blocks = [
                block
                for block in training_pool_blocks
                if block != calibration_block
            ]
            for training_index, training_block in enumerate(training_blocks):
                keys = list(block_keys[training_block])
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            BOOTSTRAP_BASE_SEED,
                            *seed_prefix,
                            replicate,
                            fold_index,
                            training_index,
                        ]
                    )
                )
                sampled[(calibration_block, training_block)] = [
                    keys[int(index)]
                    for index in rng.integers(0, len(keys), size=len(keys))
                ]
        selected, _, _, _ = _nested_selection(
            target=target,
            calibration_blocks=calibration_blocks,
            training_pool_blocks=training_pool_blocks,
            block_keys=block_keys,
            world_frames=world_frames,
            truth_pairs=truth_pairs,
            sampled_training_keys=sampled,
        )
        counts += selected.astype(np.int64)
    return counts


def _load_truth(
    *,
    formal_root: Path,
    blocks: Sequence[str],
    block_keys: Mapping[str, Sequence[str]],
    world_frames: Mapping[str, pd.DataFrame],
    forbidden_block: str,
) -> tuple[dict[str, set[frozenset[str]]], dict[str, str]]:
    truth: dict[str, set[frozenset[str]]] = {}
    hashes: dict[str, str] = {}
    for block in blocks:
        if block == forbidden_block:
            raise RuntimeError("Attempted to open sealed target-block truth")
        for key in block_keys[block]:
            seed = int(world_frames[key]["seed"].iloc[0])
            path = (
                formal_root
                / block
                / f"world_seed{seed}"
                / "scorer"
                / "truth_typed_edges.tsv"
            )
            if forbidden_block in path.parts:
                raise RuntimeError("Attempted to open sealed target-block truth")
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
            truth[key] = v454.v448.v447._truth_assay_pairs(path)
            hashes[key] = diagnostic._sha256(path)
    return truth, hashes


def diagnose_target_block(
    *,
    evidence_dir: Path,
    v454_calibration_dir: Path,
    formal_worlds_root: Path,
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
    if len(blocks) != 5:
        raise ValueError("The diagnostic requires the five existing seed blocks")
    target_index = blocks.index(target_block)
    world_frames = {
        str(key): group.reset_index(drop=True)
        for key, group in evidence.groupby("world_key", sort=False)
    }
    block_keys = {
        block: sorted(
            key
            for key, frame in world_frames.items()
            if str(frame["block"].iloc[0]) == block
        )
        for block in blocks
    }
    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)
    non_target = [block for block in blocks if block != target_block]
    truth_pairs, truth_hashes = _load_truth(
        formal_root=formal_root,
        blocks=non_target,
        block_keys=block_keys,
        world_frames=world_frames,
        forbidden_block=target_block,
    )

    calibration_path = v454_calibration_dir.expanduser().absolute().resolve(
        strict=True
    )
    calibration_manifest = diagnostic._verified_manifest(
        calibration_path / "calibration_manifest.json", v454.SCHEMA
    )
    if str(calibration_manifest.get("target_block")) != target_block:
        raise ValueError("v4.54 calibration targets another block")
    for name, expected in calibration_manifest.get("output_sha256", {}).items():
        path = calibration_path / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if diagnostic._sha256(path) != str(expected):
            raise ValueError(f"v4.54 calibration output hash mismatch: {name}")
    target_audit = pd.read_csv(
        calibration_path / "bootstrap_wasserstein_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if set(target_audit["block"].astype(str)) != {target_block}:
        raise ValueError("v4.54 pair audit crosses target blocks")

    null_rows: list[dict[str, Any]] = []
    worst_null_count = -1
    worst_null_identity: dict[str, Any] | None = None
    for candidate_index, candidate_block in enumerate(non_target):
        candidate = pd.concat(
            [world_frames[key] for key in block_keys[candidate_block]],
            ignore_index=True,
        )
        null_calibration_blocks = [
            block
            for block in blocks
            if block not in {target_block, candidate_block}
        ]
        training_pool_blocks = [
            block for block in blocks if block != candidate_block
        ]
        nominal, _, _, _ = _nested_selection(
            target=candidate,
            calibration_blocks=null_calibration_blocks,
            training_pool_blocks=training_pool_blocks,
            block_keys=block_keys,
            world_frames=world_frames,
            truth_pairs=truth_pairs,
        )
        counts = _bootstrap_counts(
            target=candidate,
            calibration_blocks=null_calibration_blocks,
            training_pool_blocks=training_pool_blocks,
            block_keys=block_keys,
            world_frames=world_frames,
            truth_pairs=truth_pairs,
            seed_prefix=(target_index, candidate_index),
        )
        labels = v454._truth_labels(candidate, truth_pairs)
        null_positions = np.flatnonzero(~labels)
        if not len(null_positions):
            raise ValueError(f"No null pairs in calibration block {candidate_block}")
        position = int(null_positions[np.argmax(counts[null_positions])])
        count = int(counts[position])
        row = candidate.iloc[position]
        record = {
            "sealed_target_block": target_block,
            "calibration_candidate_block": candidate_block,
            "world_key": str(row["world_key"]),
            "left_modality": str(row["left_modality"]),
            "right_modality": str(row["right_modality"]),
            "nominally_selected": bool(nominal[position]),
            "bootstrap_selection_count": count,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_selection_frequency": count / BOOTSTRAP_REPLICATES,
        }
        null_rows.append(record)
        if count > worst_null_count:
            worst_null_count = count
            worst_null_identity = record
    if worst_null_identity is None:
        raise RuntimeError("Empirical-null calibration produced no reference")

    pair_rows: list[dict[str, Any]] = []
    for row in target_audit.itertuples(index=False):
        proposal = v454.synchronization._as_bool(
            row.proposed_by_nominal_wasserstein_geometry
        )
        count = int(row.bootstrap_selection_count)
        probability = _posterior_dominance_probability(
            count, worst_null_count, BOOTSTRAP_REPLICATES
        )
        majority = count > int(
            MINIMUM_SELECTION_FREQUENCY * BOOTSTRAP_REPLICATES
        )
        selected = bool(
            proposal
            and majority
            and count > worst_null_count
            and probability >= POSTERIOR_DOMINANCE_THRESHOLD
        )
        pair_rows.append(
            {
                "world_key": str(row.world_key),
                "block": str(row.block),
                "seed": int(row.seed),
                "left_modality": str(row.left_modality),
                "right_modality": str(row.right_modality),
                "nominal_v454_new_proposal": proposal,
                "bootstrap_selection_count": count,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_selection_frequency": count / BOOTSTRAP_REPLICATES,
                "minimum_selection_frequency": MINIMUM_SELECTION_FREQUENCY,
                "empirical_null_maximum_selection_count": worst_null_count,
                "strictly_exceeds_empirical_null": count > worst_null_count,
                "posterior_probability_exceeds_worst_null": probability,
                "posterior_dominance_threshold": POSTERIOR_DOMINANCE_THRESHOLD,
                "selected_by_null_calibrated_stability": selected,
            }
        )
    pair_table = pd.DataFrame(pair_rows)
    null_table = pd.DataFrame(null_rows)

    out_dir.mkdir(parents=True)
    pair_path = out_dir / "null_calibrated_pair_audit.tsv"
    null_path = out_dir / "calibration_null_frequency_audit.tsv"
    pair_table.to_csv(pair_path, sep="\t", index=False)
    null_table.to_csv(null_path, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "phase": "target_pretruth_diagnostic",
        "target_block": target_block,
        "target_world_keys": block_keys[target_block],
        "calibration_candidate_blocks": non_target,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "minimum_selection_frequency": MINIMUM_SELECTION_FREQUENCY,
        "minimum_selection_frequency_comparison": "strictly_greater_than",
        "posterior_prior": "working_independent_beta_1_1_for_candidate_and_worst_null",
        "bootstrap_replicates_treated_as_independent_trials": False,
        "posterior_dominance_threshold": POSTERIOR_DOMINANCE_THRESHOLD,
        "empirical_null_maximum_selection_count": worst_null_count,
        "empirical_null_maximum_selection_frequency": (
            worst_null_count / BOOTSTRAP_REPLICATES
        ),
        "empirical_null_maximum_identity": worst_null_identity,
        "nominal_target_proposal_count": int(
            pair_table["nominal_v454_new_proposal"].sum()
        ),
        "selected_target_proposal_count": int(
            pair_table["selected_by_null_calibrated_stability"].sum()
        ),
        "information_boundary": {
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
        },
        "source_sha256": {
            Path(__file__).name: diagnostic._sha256(Path(__file__).resolve()),
            Path(v454.__file__).name: diagnostic._sha256(
                Path(v454.__file__).resolve()
            ),
        },
        "input_sha256": {
            "geometry_evidence_manifest.json": diagnostic._sha256(
                evidence_path / "evidence_manifest.json"
            ),
            "quantile_geometry_pair_evidence.tsv": diagnostic._sha256(
                evidence_path / "quantile_geometry_pair_evidence.tsv"
            ),
            "v454_calibration_manifest.json": diagnostic._sha256(
                calibration_path / "calibration_manifest.json"
            ),
            "other_block_truth_typed_edges": truth_hashes,
        },
        "output_sha256": {
            pair_path.name: diagnostic._sha256(pair_path),
            null_path.name: diagnostic._sha256(null_path),
        },
        "formal_pretruth_barrier_satisfied": True,
        "new_worlds_or_datasets_created": False,
        "retrospective_cached_world_diagnostic": True,
        "promoted_to_checkpoint": False,
    }
    manifest["manifest_payload_sha256"] = diagnostic._canonical_sha256(manifest)
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v454-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = diagnose_target_block(
        evidence_dir=args.evidence_dir,
        v454_calibration_dir=args.v454_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
