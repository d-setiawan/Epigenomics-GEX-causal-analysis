#!/usr/bin/env python3
"""Diagnose empirical-null stability of v4.57 joint sliced geometry.

For one sealed target block, this cached-world-only stage scores the frozen
joint sliced-Wasserstein evidence with the same four-signal rank product used
by the successful v4.54 geometry rescue.  It starts only from strict nominal
mutual-nearest proposals that are new relative to the target-safe v4.55 union.

The selection-frequency null is estimated in four non-target candidate
blocks.  For candidate block C and sealed target block T, score calibration
uses labels only from the three blocks outside {T, C}; target-block evidence
may enter only as unlabeled empirical-rank training data.  Every null pair is
audited.  A target proposal must have strict-majority bootstrap support,
strictly exceed the worst null count, and reach a 0.975 working Beta(1, 1)
dominance probability.  Bootstrap refits are dependent, so that probability
is explicitly an uncertainty score rather than a literal posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_null_calibrated_stability_rescue_v4 as v455  # noqa: E402
import diagnose_joint_sliced_geometry_rescue_v4 as joint  # noqa: E402
import diagnose_null_calibrated_stability_v4 as v455_diagnostic  # noqa: E402
import joint_sliced_geometry_synchronization_evidence_v4 as evidence  # noqa: E402


SCHEMA = "ca_boss_v4.joint_sliced_null_stability_diagnostic.v1"
ALGORITHM = "CA-BOSS v4.57 target-sealed joint sliced empirical-null diagnostic"
FEATURE_COLUMNS = (
    "statistical_context_support_count",
    "joint_sliced_w2_global_coassignment_context_count",
    "joint_sliced_w2_maximum_reporter_cost_ratio",
    "joint_sliced_w2_mean_coassigned_exclusion_margin",
)
FEATURE_DIRECTIONS = (1.0, 1.0, -1.0, 1.0)
FEATURE_QUANTIZATION_DECIMALS = 12
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_BASE_SEED = 457000
MINIMUM_SELECTION_FREQUENCY = 0.50
POSTERIOR_DOMINANCE_THRESHOLD = 0.975
FIXED_SELECTION_RULE = (
    "use_truth_blind_full_fit_plus_two_disjoint_feature_fold_joint_sliced_"
    "wasserstein_evidence_quantize_four_features_to_12_decimals_rank_"
    "statistical_context_support_global_joint_coassignment_negative_maximum_"
    "joint_reporter_cost_and_joint_exclusion_margin_by_empirical_percentiles_"
    "multiply_percentiles_require_strict_unique_mutual_nearest_pair_in_every_"
    "nested_fold_and_margin_strictly_above_each_other_block_null_envelope_"
    "start_only_from_nominal_pairs_new_relative_to_target_safe_v455_compare_"
    "100_whole_world_refit_selection_counts_to_every_null_pair_in_four_non_"
    "target_candidate_blocks_estimate_each_null_without_target_truth_and_with_"
    "target_evidence_only_as_unlabeled_rank_training_require_strictly_more_"
    "than_50_refits_strictly_exceed_worst_null_count_and_at_least_0_975_"
    "working_beta_1_1_dominance_probability_never_claim_direct_structure_"
    "exact_pair_or_direction"
)


def _verified_joint_evidence(
    directory: Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = joint.geometry._verified_manifest(
        resolved / "evidence_manifest.json", evidence.SCHEMA
    )
    boundary = manifest.get("information_boundary", {})
    required_boundary = {
        "truth_or_scorer_opened": False,
        "hypothesis_selected": False,
        "existing_cached_worlds_only": True,
        "new_worlds_or_datasets_created": False,
        "heldout_background_component_scores_only": True,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "transport_plan_constructed": False,
        "simulator_latent_state_used": False,
        "external_biological_features_or_priors_used": False,
        "modality_identity_used_as_model_feature": False,
        "candidate_feature_identity_used_as_model_feature": False,
    }
    for field, expected in required_boundary.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"Joint evidence boundary mismatch: {field}")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("Joint evidence lacks a formal pretruth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Joint evidence created new data")
    source_hashes = manifest.get("source_sha256", {})
    for module in (evidence, joint):
        path = Path(module.__file__).resolve()
        if source_hashes.get(path.name) != joint.geometry._sha256(path):
            raise ValueError(f"Joint evidence source changed: {path.name}")
    table_path = resolved / "joint_sliced_geometry_pair_evidence.tsv"
    if joint.geometry._sha256(table_path) != str(
        manifest.get("output_sha256", {}).get(table_path.name)
    ):
        raise ValueError("Joint evidence output hash mismatch")
    table = pd.read_csv(table_path, sep="\t", keep_default_na=False)
    if list(table.columns) != evidence.EVIDENCE_COLUMNS:
        raise ValueError("Joint evidence columns do not match the frozen schema")
    if len(table) != int(manifest.get("row_count", -1)):
        raise ValueError("Joint evidence row count disagrees with its manifest")
    if sorted(table["world_key"].astype(str).unique()) != sorted(
        map(str, manifest.get("world_keys", []))
    ):
        raise ValueError("Joint evidence world keys disagree with its manifest")
    identity = ["world_key", "left_modality", "right_modality"]
    if table.duplicated(identity).any():
        raise ValueError("Joint evidence contains duplicate assay-pair identities")
    if {"truth_pair", "truth_type", "role", "label"}.intersection(table.columns):
        raise ValueError("Truth-bearing columns entered joint evidence")
    return resolved, manifest, table


def _verified_v455_baseline(
    directory: Path,
    *,
    target_block: str,
    target_world_keys: Sequence[str],
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = joint.geometry._verified_manifest(
        resolved / "calibration_manifest.json", v455.SCHEMA
    )
    if str(manifest.get("target_block")) != target_block:
        raise ValueError("v4.55 baseline targets another block")
    if set(map(str, manifest.get("target_world_keys", []))) != set(
        map(str, target_world_keys)
    ):
        raise ValueError("v4.55 baseline and joint evidence target worlds differ")
    if manifest.get("fixed_selection_rule") != v455.FIXED_SELECTION_RULE:
        raise ValueError("v4.55 baseline selection rule changed")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
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
    }.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.55 baseline boundary mismatch: {field}")
    for field in (
        "target_specific_pretruth_barrier_satisfied",
        "formal_pretruth_barrier_satisfied",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"v4.55 baseline lacks required barrier: {field}")
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
            raise ValueError(f"v4.55 baseline invalid claim or boundary: {field}")
    source_hashes = manifest.get("source_sha256", {})
    for module in (v455, v455.diagnostic, v455.v454, v455.v454_scorer):
        path = Path(module.__file__).resolve()
        if source_hashes.get(path.name) != joint.geometry._sha256(path):
            raise ValueError(f"v4.55 baseline source changed: {path.name}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if joint.geometry._sha256(path) != str(expected):
            raise ValueError(f"v4.55 baseline output hash mismatch: {name}")
    candidates = pd.read_csv(
        resolved / "bootstrap_null_calibrated_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if list(candidates.columns) != v455.CANDIDATE_COLUMNS:
        raise ValueError("v4.55 candidate columns changed")
    if len(candidates) and set(candidates["block"].astype(str)) != {target_block}:
        raise ValueError("v4.55 baseline candidates cross target blocks")
    if len(candidates) and not candidates["hypothesis_selected"].map(
        v455.v454.synchronization._as_bool
    ).all():
        raise ValueError("v4.55 baseline contains an unselected candidate")
    return resolved, manifest, candidates


def _baseline_pairs(candidates: pd.DataFrame) -> dict[str, set[frozenset[str]]]:
    result: dict[str, set[frozenset[str]]] = {}
    for row in candidates.itertuples(index=False):
        pairs = v455.v454.v453.v449._parse_pair_payload(
            row.candidate_assay_pairs_json
        )
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.55 candidate assay-pair count mismatch")
        result.setdefault(str(row.world_key), set()).update(pairs)
    return result


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Joint stability features are missing: {missing}")
    values = (
        frame.loc[:, list(FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    return np.round(values, decimals=FEATURE_QUANTIZATION_DECIMALS)


def _copula_score(training: pd.DataFrame, values: pd.DataFrame) -> np.ndarray:
    train = _model_matrix(training) * np.asarray(FEATURE_DIRECTIONS)
    target = _model_matrix(values) * np.asarray(FEATURE_DIRECTIONS)
    denominator = float(len(train) + 1)
    percentiles = [
        np.searchsorted(np.sort(train[:, index]), target[:, index], side="right")
        .astype(float)
        / denominator
        for index in range(train.shape[1])
    ]
    return np.prod(np.column_stack(percentiles), axis=1)


def _truth_labels(
    frame: pd.DataFrame,
    truth_pairs: Mapping[str, set[frozenset[str]]],
) -> np.ndarray:
    return np.asarray(
        [
            v455.v454._pair(row.left_modality, row.right_modality)
            in truth_pairs[str(row.world_key)]
            for row in frame.itertuples(index=False)
        ],
        dtype=bool,
    )


def _nested_selection(
    *,
    target: pd.DataFrame,
    calibration_blocks: Sequence[str],
    training_pool_blocks: Sequence[str],
    block_keys: Mapping[str, Sequence[str]],
    world_frames: Mapping[str, pd.DataFrame],
    truth_pairs: Mapping[str, set[frozenset[str]]],
    sampled_training_keys: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    scores: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    mutual: list[np.ndarray] = []
    null_maxima: dict[str, float] = {}
    for calibration_block in calibration_blocks:
        training_blocks = [
            block for block in training_pool_blocks if block != calibration_block
        ]
        if not training_blocks:
            raise ValueError("A nested fold has no empirical-rank training blocks")
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
        target_score = _copula_score(training, target)
        target_margin, target_mutual = (
            v455.v454.v453._mutual_nearest_absolute_margin(target, target_score)
        )
        calibration_score = _copula_score(training, calibration)
        calibration_margin, calibration_mutual = (
            v455.v454.v453._mutual_nearest_absolute_margin(
                calibration, calibration_score
            )
        )
        null = (~_truth_labels(calibration, truth_pairs)) & calibration_mutual
        if not null.any():
            raise ValueError(
                f"Calibration block {calibration_block} has no mutual-nearest null"
            )
        scores.append(target_score)
        margins.append(target_margin)
        mutual.append(target_mutual)
        null_maxima[calibration_block] = float(calibration_margin[null].max())
    stacked_scores = np.vstack(scores).T
    stacked_margins = np.vstack(margins).T
    stacked_mutual = np.vstack(mutual).T
    selected = stacked_mutual.all(axis=1) & (
        stacked_margins.min(axis=1) > max(null_maxima.values())
    )
    return selected, stacked_scores, stacked_margins, stacked_mutual, null_maxima


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
        selected, _, _, _, _ = _nested_selection(
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
            truth[key] = v455.v454.v448.v447._truth_assay_pairs(path)
            hashes[key] = joint.geometry._sha256(path)
    return truth, hashes


def _selection_decision(
    *, nominal_new_proposal: bool, candidate_count: int, null_count: int
) -> tuple[bool, float]:
    probability = v455_diagnostic._posterior_dominance_probability(
        candidate_count, null_count, BOOTSTRAP_REPLICATES
    )
    selected = bool(
        nominal_new_proposal
        and candidate_count > int(
            MINIMUM_SELECTION_FREQUENCY * BOOTSTRAP_REPLICATES
        )
        and candidate_count > null_count
        and probability >= POSTERIOR_DOMINANCE_THRESHOLD
    )
    return selected, probability


def diagnose_target_block(
    *,
    evidence_dir: Path,
    v455_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    evidence_path, evidence_manifest, table = _verified_joint_evidence(evidence_dir)
    blocks = sorted(table["block"].astype(str).unique())
    if target_block not in blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    if len(blocks) != 5:
        raise ValueError("The diagnostic requires the five existing seed blocks")
    target_index = blocks.index(target_block)
    world_frames = {
        str(key): group.reset_index(drop=True)
        for key, group in table.groupby("world_key", sort=False)
    }
    block_keys = {
        block: sorted(
            key
            for key, frame in world_frames.items()
            if str(frame["block"].iloc[0]) == block
        )
        for block in blocks
    }
    target_keys = block_keys[target_block]
    non_target = [block for block in blocks if block != target_block]
    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)
    truth_pairs, truth_hashes = _load_truth(
        formal_root=formal_root,
        blocks=non_target,
        block_keys=block_keys,
        world_frames=world_frames,
        forbidden_block=target_block,
    )
    baseline_path, baseline_manifest, baseline_candidates = (
        _verified_v455_baseline(
            v455_calibration_dir,
            target_block=target_block,
            target_world_keys=target_keys,
        )
    )
    baseline_pairs = _baseline_pairs(baseline_candidates)

    target = pd.concat(
        [world_frames[key] for key in target_keys], ignore_index=True
    )
    nominal, scores, margins, mutual, target_null_maxima = _nested_selection(
        target=target,
        calibration_blocks=non_target,
        training_pool_blocks=non_target,
        block_keys=block_keys,
        world_frames=world_frames,
        truth_pairs=truth_pairs,
    )
    target_counts = _bootstrap_counts(
        target=target,
        calibration_blocks=non_target,
        training_pool_blocks=non_target,
        block_keys=block_keys,
        world_frames=world_frames,
        truth_pairs=truth_pairs,
        seed_prefix=(target_index, len(blocks)),
    )

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
        candidate_nominal, _, _, _, _ = _nested_selection(
            target=candidate,
            calibration_blocks=null_calibration_blocks,
            training_pool_blocks=training_pool_blocks,
            block_keys=block_keys,
            world_frames=world_frames,
            truth_pairs=truth_pairs,
        )
        candidate_counts = _bootstrap_counts(
            target=candidate,
            calibration_blocks=null_calibration_blocks,
            training_pool_blocks=training_pool_blocks,
            block_keys=block_keys,
            world_frames=world_frames,
            truth_pairs=truth_pairs,
            seed_prefix=(target_index, candidate_index),
        )
        labels = _truth_labels(candidate, truth_pairs)
        null_positions = np.flatnonzero(~labels)
        if not len(null_positions):
            raise ValueError(f"No null pairs in candidate block {candidate_block}")
        for position in null_positions:
            row = candidate.iloc[int(position)]
            count = int(candidate_counts[int(position)])
            record = {
                "sealed_target_block": target_block,
                "calibration_candidate_block": candidate_block,
                "world_key": str(row["world_key"]),
                "block": str(row["block"]),
                "seed": int(row["seed"]),
                "left_modality": str(row["left_modality"]),
                "right_modality": str(row["right_modality"]),
                "empirical_null": True,
                "nominally_selected": bool(candidate_nominal[int(position)]),
                "bootstrap_selection_count": count,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_selection_frequency": count / BOOTSTRAP_REPLICATES,
            }
            null_rows.append(record)
            if count > worst_null_count:
                worst_null_count = count
                worst_null_identity = dict(record)
    if worst_null_identity is None:
        raise RuntimeError("Joint empirical-null calibration produced no reference")

    baseline_mask = np.asarray(
        [
            v455.v454._pair(row.left_modality, row.right_modality)
            in baseline_pairs.get(str(row.world_key), set())
            for row in target.itertuples(index=False)
        ],
        dtype=bool,
    )
    pair_rows: list[dict[str, Any]] = []
    for position, row in enumerate(target.itertuples(index=False)):
        nominal_new = bool(nominal[position] and not baseline_mask[position])
        count = int(target_counts[position])
        selected, probability = _selection_decision(
            nominal_new_proposal=nominal_new,
            candidate_count=count,
            null_count=worst_null_count,
        )
        pair_rows.append(
            {
                "world_key": str(row.world_key),
                "block": str(row.block),
                "seed": int(row.seed),
                "left_modality": str(row.left_modality),
                "right_modality": str(row.right_modality),
                "candidate_cross_assay_feature_pairs_json": str(
                    row.candidate_cross_assay_feature_pairs_json
                ),
                **{
                    column: float(getattr(row, column))
                    for column in FEATURE_COLUMNS
                },
                "nested_empirical_rank_scores_json": json.dumps(
                    scores[position].tolist(), separators=(",", ":")
                ),
                "nested_absolute_margins_json": json.dumps(
                    margins[position].tolist(), separators=(",", ":")
                ),
                "nested_mutual_nearest_flags_json": json.dumps(
                    mutual[position].tolist(), separators=(",", ":")
                ),
                "minimum_nested_absolute_margin": float(margins[position].min()),
                "calibration_null_margin_upper_envelope": max(
                    target_null_maxima.values()
                ),
                "selected_by_v455_baseline": bool(baseline_mask[position]),
                "proposed_by_nominal_joint_geometry": bool(nominal[position]),
                "nominal_joint_proposal_new_relative_to_v455": nominal_new,
                "bootstrap_selection_count": count,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_selection_frequency": count / BOOTSTRAP_REPLICATES,
                "minimum_selection_frequency": MINIMUM_SELECTION_FREQUENCY,
                "empirical_null_maximum_selection_count": worst_null_count,
                "empirical_null_maximum_selection_frequency": (
                    worst_null_count / BOOTSTRAP_REPLICATES
                ),
                "strictly_exceeds_empirical_null": count > worst_null_count,
                "posterior_probability_exceeds_worst_null": probability,
                "posterior_dominance_threshold": POSTERIOR_DOMINANCE_THRESHOLD,
                "selected_by_joint_null_calibrated_stability": selected,
            }
        )
    pair_table = pd.DataFrame(pair_rows)
    null_table = pd.DataFrame(null_rows)

    out_dir.mkdir(parents=True)
    pair_path = out_dir / "joint_null_calibrated_pair_audit.tsv"
    null_path = out_dir / "joint_calibration_null_frequency_audit.tsv"
    pair_table.to_csv(pair_path, sep="\t", index=False)
    null_table.to_csv(null_path, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "phase": "target_pretruth_diagnostic",
        "target_block": target_block,
        "target_world_keys": target_keys,
        "calibration_candidate_blocks": non_target,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_directions": list(FEATURE_DIRECTIONS),
        "feature_quantization_decimals": FEATURE_QUANTIZATION_DECIMALS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "minimum_selection_frequency": MINIMUM_SELECTION_FREQUENCY,
        "minimum_selection_frequency_comparison": "strictly_greater_than",
        "posterior_prior": "working_independent_beta_1_1_for_candidate_and_worst_null",
        "bootstrap_replicates_treated_as_independent_trials": False,
        "posterior_dominance_threshold": POSTERIOR_DOMINANCE_THRESHOLD,
        "empirical_null_pair_count": len(null_table),
        "empirical_null_maximum_selection_count": worst_null_count,
        "empirical_null_maximum_selection_frequency": (
            worst_null_count / BOOTSTRAP_REPLICATES
        ),
        "empirical_null_maximum_identity": worst_null_identity,
        "nominal_target_proposal_count": int(nominal.sum()),
        "nominal_new_target_proposal_count": int(
            pair_table["nominal_joint_proposal_new_relative_to_v455"].sum()
        ),
        "selected_target_proposal_count": int(
            pair_table["selected_by_joint_null_calibrated_stability"].sum()
        ),
        "baseline": "target-safe CA-BOSS v4.55 shared-state union",
        "information_boundary": {
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
        },
        "candidate_pairs_may_be_expanded_as_edges": False,
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "source_sha256": {
            path.name: joint.geometry._sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(evidence.__file__).resolve(),
                Path(joint.__file__).resolve(),
                Path(v455.__file__).resolve(),
                Path(v455_diagnostic.__file__).resolve(),
            )
        },
        "input_sha256": {
            "joint_evidence_manifest.json": joint.geometry._sha256(
                evidence_path / "evidence_manifest.json"
            ),
            "joint_sliced_geometry_pair_evidence.tsv": joint.geometry._sha256(
                evidence_path / "joint_sliced_geometry_pair_evidence.tsv"
            ),
            "v455_calibration_manifest.json": joint.geometry._sha256(
                baseline_path / "calibration_manifest.json"
            ),
            "other_block_truth_typed_edges": truth_hashes,
        },
        "parent_manifest_payload_sha256": {
            "joint_evidence": evidence_manifest["manifest_payload_sha256"],
            "v455": baseline_manifest["manifest_payload_sha256"],
        },
        "output_sha256": {
            pair_path.name: joint.geometry._sha256(pair_path),
            null_path.name: joint.geometry._sha256(null_path),
        },
        "formal_pretruth_barrier_satisfied": True,
        "new_worlds_or_datasets_created": False,
        "retrospective_cached_world_diagnostic": True,
        "promoted_to_checkpoint": False,
    }
    manifest["manifest_payload_sha256"] = joint.geometry._canonical_sha256(manifest)
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v455-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = diagnose_target_block(
        evidence_dir=args.evidence_dir,
        v455_calibration_dir=args.v455_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
