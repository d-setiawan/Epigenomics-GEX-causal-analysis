#!/usr/bin/env python3
"""Bootstrap-stable empirical-copula pair rescue for CA-BOSS v4.52.

The stage converts four directional, truth-blind reporter-pair signals to
training-fold empirical percentiles and multiplies them into a conjunctive
score.  For each target block, four nested folds exclude both the target and
one calibration block.  A target pair must strictly exceed the largest
corresponding calibration-null score.  Nominal additions are then required to
survive at least 80 of 100 whole-training-world bootstrap refits.

The stable additions are unioned with target-safe v4.51 hypotheses.  No new
world, paired cell, modality identity, feature identity, simulator state,
external biology, direct cross-assay edge, or direction is used.
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

import bootstrap_stable_sparse_completion_v4 as v451  # noqa: E402
import crossfit_multivariate_reporter_rescue_v4 as v448  # noqa: E402
import crossfit_sparse_reporter_completion_v4 as v449  # noqa: E402
import heldout_synchronized_majority_rescue_v4 as guarded_rescue  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import relaxed_reporter_synchronization_evidence_v4 as relaxed  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_empirical_copula_pair_rescue.v1"
ALGORITHM = "CA-BOSS v4.52 bootstrap-stable empirical-copula pair rescue"
FEATURE_COLUMNS = v448.FEATURE_COLUMNS
FEATURE_DIRECTIONS = (1.0, 1.0, -1.0, 1.0)
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_FREQUENCY_THRESHOLD = 0.80
BOOTSTRAP_BASE_SEED = 452000
FIXED_SELECTION_RULE = (
    "use_only_reporter_pair_rows_remaining_after_truth_blind_v444_and_v446_"
    "exclusions_for_each_target_and_calibration_fold_transform_the_four_"
    "fixed_directional_signals_to_empirical_percentiles_using_all_rows_in_"
    "the_other_three_training_blocks_multiply_the_percentiles_score_the_"
    "target_by_its_minimum_across_folds_and_set_the_threshold_to_the_"
    "maximum_corresponding_other_block_null_score_require_strict_exceedance_"
    "treat_only_nominal_pairs_not_in_target_safe_v451_as_proposals_resample_"
    "whole_training_worlds_with_replacement_within_blocks_100_times_require_"
    "proposal_selection_frequency_at_least_80_percent_union_with_target_"
    "safe_v451_merge_connected_pairs_as_unexpanded_ambiguity_hyperedges_"
    "and_never_claim_direct_structure_or_direction"
)

FOLD_AUDIT_COLUMNS = [
    "target_block",
    "calibration_block",
    "training_blocks",
    "training_world_count",
    "training_pair_count",
    "calibration_world_count",
    "calibration_pair_count",
    "calibration_null_pair_count",
    "maximum_calibration_null_copula_score",
    "feature_columns_json",
    "feature_directions_json",
]

PAIR_AUDIT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "left_modality",
    "right_modality",
    *FEATURE_COLUMNS,
    "nested_copula_scores_json",
    "minimum_nested_copula_score",
    "calibration_null_upper_envelope",
    "strictly_exceeds_calibration_null_upper_envelope",
    "selected_by_target_safe_v451",
    "proposed_by_nominal_empirical_copula",
    "bootstrap_selection_count",
    "bootstrap_replicates",
    "bootstrap_selection_frequency",
    "bootstrap_frequency_threshold",
    "passes_bootstrap_stability_guard",
    "newly_selected_by_v452",
    "retained_in_union",
    "merged_hyperedge_id",
]

REPLICATE_COLUMNS = [
    "target_block",
    "bootstrap_seed",
    "bootstrap_replicate",
    "calibration_blocks",
    "calibration_null_maxima_json",
    "null_copula_score_upper_envelope",
    "raw_selected_pair_count",
]

CANDIDATE_COLUMNS = [
    "copula_hyperedge_id",
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
    "bootstrap_frequency_threshold",
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


def _as_bool(value: Any) -> bool:
    return synchronization._as_bool(value)


def _pair_key(left: str, right: str) -> frozenset[str]:
    return v448._pair_key(left, right)


def _pair_sort_key(pair: frozenset[str]) -> tuple[str, str]:
    return v448._pair_sort_key(pair)


def _world_key(block: str, seed: int) -> str:
    return v449._world_key(block, seed)


def _verified_v451_calibration(
    directory: Path, *, target_block: str
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = v449._verified_payload(
        resolved / "calibration_manifest.json", "manifest_payload_sha256"
    )
    if manifest.get("schema_version") != v451.SCHEMA:
        raise ValueError("Unexpected v4.51 calibration schema")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("target_block_scorer_truth_opened") is not False:
        raise ValueError("v4.51 opened target truth")
    if boundary.get("target_world_truth_used_for_fit_or_selection") is not False:
        raise ValueError("v4.51 used target truth")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.51 lacks a target-specific pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.51 created new data")
    if str(manifest.get("target_block")) != str(target_block):
        raise ValueError("v4.51 calibration targets a different block")
    if int(manifest.get("bootstrap_replicates_configured", -1)) != (
        v451.BOOTSTRAP_REPLICATES
    ):
        raise ValueError("v4.51 bootstrap contract changed")
    if float(manifest.get("bootstrap_frequency_threshold", -1.0)) != (
        v451.BOOTSTRAP_FREQUENCY_THRESHOLD
    ):
        raise ValueError("v4.51 stability threshold changed")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"v4.51 output hash mismatch: {name}")
    candidates = pd.read_csv(
        resolved / "bootstrap_stable_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if len(candidates) and not candidates["hypothesis_selected"].map(
        _as_bool
    ).all():
        raise ValueError("v4.51 contains an unselected candidate")
    return resolved, manifest, candidates


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Pair evidence lacks copula features: {missing}")
    numeric = frame.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    return (
        numeric.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _empirical_copula_product(
    training: pd.DataFrame, values: pd.DataFrame
) -> np.ndarray:
    """Return a deterministic product of favorable empirical percentiles."""

    x_training = _model_matrix(training) * np.asarray(FEATURE_DIRECTIONS)
    x_values = _model_matrix(values) * np.asarray(FEATURE_DIRECTIONS)
    if not len(x_training):
        raise ValueError("Empirical-copula training fold is empty")
    percentiles = []
    denominator = float(len(x_training) + 1)
    for index in range(x_training.shape[1]):
        ordered = np.sort(x_training[:, index])
        percentiles.append(
            np.searchsorted(
                ordered, x_values[:, index], side="right"
            ).astype(float)
            / denominator
        )
    return np.prod(np.column_stack(percentiles), axis=1)


def _bootstrap_frequency_passes(count: int, replicates: int) -> bool:
    if replicates <= 0:
        return False
    return (float(count) / float(replicates)) >= BOOTSTRAP_FREQUENCY_THRESHOLD


def _baseline_components(
    candidates: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        pairs = v449._parse_pair_payload(row.candidate_assay_pairs_json)
        features = v449._parse_feature_pair_payload(
            row.candidate_cross_assay_feature_pairs_json
        )
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.51 candidate assay-pair count mismatch")
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.51 candidate feature-pair count mismatch")
        components[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(str(row.candidate_pair_selection_audit_json)),
                "source": "target_safe_v451",
            }
        )
    return components


def _union_candidates(
    *,
    target_keys: Sequence[str],
    evidence_frames: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    stable_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[pd.DataFrame, dict[tuple[str, frozenset[str]], str]]:
    rows: list[dict[str, Any]] = []
    merged_id_by_pair: dict[tuple[str, frozenset[str]], str] = {}
    for world_key in sorted(target_keys):
        frame = evidence_frames[world_key]
        block = str(frame["block"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        sources = [dict(value) for value in baseline.get(world_key, [])]
        for selected in stable_rows.get(world_key, []):
            pair = _pair_key(
                selected["left_modality"], selected["right_modality"]
            )
            features = v449._parse_feature_pair_payload(
                selected["candidate_cross_assay_feature_pairs_json"]
            )
            sources.append(
                {
                    "pairs": {pair},
                    "features": features,
                    "audit": [
                        {
                            "assay_pair": list(_pair_sort_key(pair)),
                            "minimum_nested_copula_score": float(
                                selected["minimum_nested_copula_score"]
                            ),
                            "bootstrap_selection_frequency": float(
                                selected["bootstrap_selection_frequency"]
                            ),
                            "source": "v452_bootstrap_empirical_copula",
                        }
                    ],
                    "source": "v452_bootstrap_empirical_copula",
                }
            )
        all_pairs = (
            set().union(*(source["pairs"] for source in sources))
            if sources
            else set()
        )
        for component_index, component_pairs in enumerate(
            v448.v447._connected_pair_components(all_pairs), start=1
        ):
            hyperedge_id = f"copula_stable_shared_state_{component_index:03d}"
            modalities = sorted(
                set().union(*(set(pair) for pair in component_pairs))
            )
            component_sources = [
                source
                for source in sources
                if set(source["pairs"]) & component_pairs
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
            pair_payload = [
                list(_pair_sort_key(pair))
                for pair in sorted(component_pairs, key=_pair_sort_key)
            ]
            for pair in component_pairs:
                merged_id_by_pair[(world_key, pair)] = hyperedge_id
            rows.append(
                {
                    "copula_hyperedge_id": hyperedge_id,
                    "world_key": world_key,
                    "block": block,
                    "seed": seed,
                    "modalities": ";".join(modalities),
                    "n_modalities": len(modalities),
                    "candidate_assay_pairs_json": json.dumps(
                        pair_payload, separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": len(component_pairs),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        sorted(features), separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(features),
                    "candidate_pair_selection_audit_json": json.dumps(
                        audits, sort_keys=True, separators=(",", ":")
                    ),
                    "selection_sources": ";".join(sorted(source_names)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_frequency_threshold": (
                        BOOTSTRAP_FREQUENCY_THRESHOLD
                    ),
                    "assay_participation_is_set_valued": (
                        len(component_pairs) != 1
                    ),
                    "exact_assay_participation_claimed": (
                        len(component_pairs) == 1
                    ),
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_copula_stable_ambiguity"
                        if len(component_pairs) != 1
                        else "selected_as_copula_stable_point_hypothesis"
                    ),
                    "hypothesis_type": "cross_assay_shared_state_equivalence",
                    "within_assay_direct_structure_claimed": False,
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), merged_id_by_pair


def _nested_copula_selection(
    *,
    target_model: pd.DataFrame,
    calibration_blocks: Sequence[str],
    model_frames: Mapping[str, pd.DataFrame],
    block_keys: Mapping[str, Sequence[str]],
    truth_pairs_by_world: Mapping[str, set[frozenset[str]]],
    sampled_training_keys: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    fold_scores: list[np.ndarray] = []
    null_maxima: dict[str, float] = {}
    for calibration_block in calibration_blocks:
        training_blocks = [
            block
            for block in calibration_blocks
            if block != calibration_block
        ]
        training_keys: list[str] = []
        for training_block in training_blocks:
            if sampled_training_keys is None:
                training_keys.extend(block_keys[training_block])
            else:
                training_keys.extend(
                    sampled_training_keys[(calibration_block, training_block)]
                )
        training = pd.concat(
            [model_frames[key] for key in training_keys], ignore_index=True
        )
        calibration_frame = pd.concat(
            [model_frames[key] for key in block_keys[calibration_block]],
            ignore_index=True,
        )
        calibration_labels = v448._truth_labels(
            calibration_frame, truth_pairs_by_world
        )
        if not (~calibration_labels).any():
            raise ValueError("Calibration fold contains no null assay pair")
        fold_scores.append(
            _empirical_copula_product(training, target_model)
        )
        calibration_scores = _empirical_copula_product(
            training, calibration_frame
        )
        null_maxima[calibration_block] = float(
            calibration_scores[~calibration_labels].max()
        )
    stacked = np.vstack(fold_scores).T
    minimum_score = stacked.min(axis=1)
    selected = minimum_score > max(null_maxima.values())
    return selected, stacked, null_maxima


def calibrate_target_block(
    *,
    relaxed_roots: Sequence[Path],
    rescue_roots: Sequence[Path],
    v447_calibration_dir: Path,
    v451_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Select target pairs with nested copula scores and world stability."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    relaxed_worlds = v448.v447._discover_worlds(
        relaxed_roots,
        manifest_name="evidence_manifest.json",
        schema=relaxed.SCHEMA,
    )
    rescue_worlds = v448.v447._discover_worlds(
        rescue_roots,
        manifest_name="rescue_manifest.json",
        schema=guarded_rescue.SCHEMA,
    )
    if set(relaxed_worlds) != set(rescue_worlds):
        raise ValueError("Relaxed evidence and v4.46 world sets differ")
    available_blocks = sorted(
        {str(manifest["block"]) for _, manifest in relaxed_worlds.values()}
    )
    target_block = str(target_block)
    if target_block not in available_blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    calibration_blocks = [
        block for block in available_blocks if block != target_block
    ]
    if len(calibration_blocks) < 3:
        raise ValueError("At least three non-target blocks are required")

    v447_dir, v447_manifest, _ = v448._verified_v447_calibration(
        v447_calibration_dir, target_block=target_block
    )
    v451_dir, v451_manifest, v451_candidates = _verified_v451_calibration(
        v451_calibration_dir, target_block=target_block
    )
    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)

    evidence_frames: dict[str, pd.DataFrame] = {}
    model_frames: dict[str, pd.DataFrame] = {}
    truth_blind_v446_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    truth_pairs_by_world: dict[str, set[frozenset[str]]] = {}
    evidence_input_hashes: dict[str, dict[str, str]] = {}
    rescue_input_hashes: dict[str, dict[str, str]] = {}
    training_truth_hashes: dict[str, str] = {}
    world_block: dict[str, str] = {}
    target_keys: list[str] = []

    for key in sorted(relaxed_worlds):
        evidence_dir, evidence_manifest = relaxed_worlds[key]
        rescue_dir, rescue_manifest = rescue_worlds[key]
        block = str(evidence_manifest["block"])
        seed = int(evidence_manifest["seed"])
        if (
            block != str(rescue_manifest["block"])
            or seed != int(rescue_manifest["seed"])
        ):
            raise ValueError(f"Upstream world identities disagree: {key}")
        evidence_path = evidence_dir / "relaxed_reporter_synchronization_pairs.tsv"
        frame = pd.read_csv(evidence_path, sep="\t", keep_default_na=False)
        frame["world_key"] = key
        frame["block"] = block
        frame["seed"] = seed
        pair_keys = [
            _pair_key(left, right)
            for left, right in zip(
                frame["left_modality"], frame["right_modality"]
            )
        ]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError(f"Duplicate assay pair in reporter evidence: {key}")
        evidence_frames[key] = frame
        world_block[key] = block

        rescue_path = rescue_dir / "heldout_synchronized_majority_candidates.tsv"
        rescue_candidates = pd.read_csv(
            rescue_path, sep="\t", keep_default_na=False
        )
        evidence_pair_set = set(pair_keys)
        for row in rescue_candidates.itertuples(index=False):
            if not _as_bool(row.hypothesis_selected):
                continue
            modalities = [
                value for value in str(row.modalities).split(";") if value
            ]
            if len(modalities) != 2:
                raise ValueError("v4.46 candidate is not an assay pair")
            pair = frozenset(modalities)
            if pair not in evidence_pair_set:
                raise ValueError("v4.46 pair lacks relaxed reporter evidence")
            truth_blind_v446_pairs[key].add(pair)

        v444_mask = frame["v444_assay_pair_already_represented"].map(_as_bool)
        v446_mask = np.asarray(
            [pair in truth_blind_v446_pairs[key] for pair in pair_keys],
            dtype=bool,
        )
        model_frames[key] = frame.loc[
            ~(v444_mask.to_numpy(dtype=bool) | v446_mask)
        ].copy()

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
                synchronization._sha256(rescue_path)
            ),
        }
        if block == target_block:
            target_keys.append(key)
            continue
        truth_path = (
            formal_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        truth_pairs_by_world[key] = v448.v447._truth_assay_pairs(truth_path)
        training_truth_hashes[key] = synchronization._sha256(truth_path)

    if not target_keys:
        raise ValueError("Target block has no cached worlds")
    target_key_set = set(target_keys)
    for label, manifest in (("v4.47", v447_manifest), ("v4.51", v451_manifest)):
        if set(map(str, manifest.get("target_world_keys", []))) != target_key_set:
            raise ValueError(f"{label} and v4.52 target worlds differ")
    if len(v451_candidates) and not v451_candidates["world_key"].astype(
        str
    ).isin(target_key_set).all():
        raise ValueError("v4.51 contains a non-target candidate")

    v447_inputs = v447_manifest.get("input_sha256", {})
    for key in sorted(relaxed_worlds):
        if (
            v447_inputs.get("relaxed_evidence_worlds", {}).get(key)
            != evidence_input_hashes[key]
        ):
            raise ValueError(f"v4.47 evidence binding mismatch: {key}")
        if (
            v447_inputs.get("v446_guarded_rescue_worlds", {}).get(key)
            != rescue_input_hashes[key]
        ):
            raise ValueError(f"v4.47 rescue binding mismatch: {key}")

    block_keys = {
        block: sorted(
            key for key, value in world_block.items() if value == block
        )
        for block in available_blocks
    }
    target_model = pd.concat(
        [model_frames[key] for key in sorted(target_keys)], ignore_index=True
    )
    if target_model.empty:
        raise ValueError("Target block has no copula-eligible assay pairs")
    identities = [
        (str(row.world_key), _pair_key(row.left_modality, row.right_modality))
        for row in target_model.itertuples(index=False)
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Target copula pair identities are not unique")

    nominal_selected, nominal_scores, null_maxima = _nested_copula_selection(
        target_model=target_model,
        calibration_blocks=calibration_blocks,
        model_frames=model_frames,
        block_keys=block_keys,
        truth_pairs_by_world=truth_pairs_by_world,
    )
    null_envelope = max(null_maxima.values())
    fold_rows = []
    for calibration_block in calibration_blocks:
        training_blocks = [
            block for block in calibration_blocks if block != calibration_block
        ]
        fold_rows.append(
            {
                "target_block": target_block,
                "calibration_block": calibration_block,
                "training_blocks": ";".join(training_blocks),
                "training_world_count": sum(
                    len(block_keys[block]) for block in training_blocks
                ),
                "training_pair_count": sum(
                    len(model_frames[key])
                    for block in training_blocks
                    for key in block_keys[block]
                ),
                "calibration_world_count": len(block_keys[calibration_block]),
                "calibration_pair_count": sum(
                    len(model_frames[key])
                    for key in block_keys[calibration_block]
                ),
                "calibration_null_pair_count": sum(
                    1
                    for key in block_keys[calibration_block]
                    for row in model_frames[key].itertuples(index=False)
                    if _pair_key(row.left_modality, row.right_modality)
                    not in truth_pairs_by_world[key]
                ),
                "maximum_calibration_null_copula_score": null_maxima[
                    calibration_block
                ],
                "feature_columns_json": json.dumps(
                    list(FEATURE_COLUMNS), separators=(",", ":")
                ),
                "feature_directions_json": json.dumps(
                    list(FEATURE_DIRECTIONS), separators=(",", ":")
                ),
            }
        )
    fold_audit = pd.DataFrame(fold_rows, columns=FOLD_AUDIT_COLUMNS)

    baseline = _baseline_components(v451_candidates)
    baseline_pairs_by_world: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs_by_world[key].update(component["pairs"])
    baseline_mask = np.asarray(
        [pair in baseline_pairs_by_world.get(key, set()) for key, pair in identities],
        dtype=bool,
    )
    proposal_mask = nominal_selected & ~baseline_mask

    bootstrap_seed = BOOTSTRAP_BASE_SEED + available_blocks.index(target_block)
    selection_counts = np.zeros(len(target_model), dtype=int)
    replicate_rows: list[dict[str, Any]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled: dict[tuple[str, str], list[str]] = {}
        for calibration_index, calibration_block in enumerate(calibration_blocks):
            training_blocks = [
                block
                for block in calibration_blocks
                if block != calibration_block
            ]
            for training_index, training_block in enumerate(training_blocks):
                keys = block_keys[training_block]
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            bootstrap_seed,
                            replicate,
                            calibration_index,
                            training_index,
                        ]
                    )
                )
                sampled[(calibration_block, training_block)] = [
                    keys[int(index)]
                    for index in rng.integers(0, len(keys), size=len(keys))
                ]
        selected, _, replicate_null_maxima = _nested_copula_selection(
            target_model=target_model,
            calibration_blocks=calibration_blocks,
            model_frames=model_frames,
            block_keys=block_keys,
            truth_pairs_by_world=truth_pairs_by_world,
            sampled_training_keys=sampled,
        )
        selection_counts += selected.astype(int)
        replicate_rows.append(
            {
                "target_block": target_block,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_replicate": replicate + 1,
                "calibration_blocks": ";".join(calibration_blocks),
                "calibration_null_maxima_json": json.dumps(
                    replicate_null_maxima,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "null_copula_score_upper_envelope": max(
                    replicate_null_maxima.values()
                ),
                "raw_selected_pair_count": int(selected.sum()),
            }
        )
    replicate_audit = pd.DataFrame(replicate_rows, columns=REPLICATE_COLUMNS)
    frequencies = selection_counts.astype(float) / float(BOOTSTRAP_REPLICATES)
    stable_mask = np.asarray(
        [
            bool(proposal_mask[index])
            and _bootstrap_frequency_passes(
                int(selection_counts[index]), BOOTSTRAP_REPLICATES
            )
            for index in range(len(target_model))
        ],
        dtype=bool,
    )

    pair_rows: list[dict[str, Any]] = []
    stable_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(target_model.itertuples(index=False)):
        values: dict[str, Any] = {
            "world_key": str(row.world_key),
            "block": str(row.block),
            "seed": int(row.seed),
            "left_modality": str(row.left_modality),
            "right_modality": str(row.right_modality),
        }
        for feature in FEATURE_COLUMNS:
            values[feature] = float(getattr(row, feature) or 0.0)
        values.update(
            {
                "nested_copula_scores_json": json.dumps(
                    [float(value) for value in nominal_scores[index]],
                    separators=(",", ":"),
                ),
                "minimum_nested_copula_score": float(
                    nominal_scores[index].min()
                ),
                "calibration_null_upper_envelope": null_envelope,
                "strictly_exceeds_calibration_null_upper_envelope": bool(
                    nominal_selected[index]
                ),
                "selected_by_target_safe_v451": bool(baseline_mask[index]),
                "proposed_by_nominal_empirical_copula": bool(
                    proposal_mask[index]
                ),
                "bootstrap_selection_count": int(selection_counts[index]),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_selection_frequency": float(frequencies[index]),
                "bootstrap_frequency_threshold": (
                    BOOTSTRAP_FREQUENCY_THRESHOLD
                ),
                "passes_bootstrap_stability_guard": bool(stable_mask[index]),
                "newly_selected_by_v452": bool(stable_mask[index]),
                "retained_in_union": bool(
                    baseline_mask[index] or stable_mask[index]
                ),
                "merged_hyperedge_id": "",
                "candidate_cross_assay_feature_pairs_json": str(
                    row.candidate_cross_assay_feature_pairs_json
                ),
            }
        )
        pair_rows.append(values)
        if stable_mask[index]:
            stable_rows[str(row.world_key)].append(values)

    candidates, merged_id_by_pair = _union_candidates(
        target_keys=target_keys,
        evidence_frames=evidence_frames,
        baseline=baseline,
        stable_rows=stable_rows,
    )
    for row in pair_rows:
        pair = _pair_key(row["left_modality"], row["right_modality"])
        row["merged_hyperedge_id"] = merged_id_by_pair.get(
            (str(row["world_key"]), pair), ""
        )
    pair_audit = pd.DataFrame(pair_rows, columns=PAIR_AUDIT_COLUMNS)

    out_dir.mkdir(parents=True)
    outputs = {
        "bootstrap_copula_candidates.tsv": candidates,
        "bootstrap_copula_pair_audit.tsv": pair_audit,
        "bootstrap_copula_fold_audit.tsv": fold_audit,
        "bootstrap_copula_replicate_audit.tsv": replicate_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    v447_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v447_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v447_dir / str(name))
            for name in v447_manifest.get("output_sha256", {})
        },
    }
    v451_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v451_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v451_dir / str(name))
            for name in v451_manifest.get("output_sha256", {})
        },
    }
    manifest = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "target_block": target_block,
        "target_world_count": len(target_keys),
        "target_world_keys": sorted(target_keys),
        "available_blocks": available_blocks,
        "calibration_blocks": calibration_blocks,
        "baseline_algorithm": "CA-BOSS v4.51 target-safe stable union",
        "model_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_directions": list(FEATURE_DIRECTIONS),
        "model_feature_imputation": "nonfinite_or_missing_to_zero",
        "copula_score": "product_of_directional_training_fold_ecdf_percentiles",
        "copula_rank_tie_rule": "right_continuous_searchsorted",
        "copula_rank_denominator": "training_pair_count_plus_one",
        "supervised_positive_labels_used_for_score_fit": False,
        "other_block_truth_used_only_to_identify_calibration_nulls": True,
        "nominal_null_copula_score_upper_envelope": null_envelope,
        "nominal_proposal_pair_count": int(proposal_mask.sum()),
        "bootstrap_unit": "whole_training_world_within_seed_block",
        "bootstrap_sampling": "with_replacement_same_block_world_count",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "bootstrap_target_block_seed": bootstrap_seed,
        "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
        "bootstrap_threshold_comparison": "inclusive_greater_than_or_equal",
        "newly_selected_by_v452_assay_pair_count": int(stable_mask.sum()),
        "rejected_unstable_nominal_proposal_count": int(
            proposal_mask.sum() - stable_mask.sum()
        ),
        "selected_union_hyperedge_count": len(candidates),
        "selected_union_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "selected_exact_assay_hypothesis_count": int(
            candidates["exact_assay_participation_claimed"].map(_as_bool).sum()
        ),
        "selected_ambiguity_hypothesis_count": int(
            candidates["assay_participation_is_set_valued"].map(_as_bool).sum()
        ),
        "maximum_candidate_assay_pair_count": (
            int(candidates["candidate_assay_pair_count"].max())
            if len(candidates)
            else 0
        ),
        "information_boundary": {
            "other_block_simulation_truth_used_to_identify_nulls": True,
            "other_block_positive_labels_used_for_score_fit": False,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "target_safe_v451_used_only_as_target_baseline_union": True,
            "known_sentinel_or_false_pairs_used_for_empirical_null": True,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
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
            "target_safe_v447_calibration": v447_input_hashes,
            "target_safe_v451_calibration": v451_input_hashes,
            "calibration_truth_typed_edges": training_truth_hashes,
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "empirical_copula_tail_scoring": {
                "citation": "Li, Zhao, Botta, Ionescu, and Hu, ICDM 2020",
                "title": "COPOD: Copula-Based Outlier Detection",
                "doi": "10.1109/ICDM50108.2020.00135",
                "relationship": (
                    "inspiration_for_parameter_free_directional_empirical_"
                    "rank_scoring_not_a_direct_application"
                ),
            },
            "stability_selection": {
                "citation": "Meinshausen and Buehlmann, JRSS B 2010",
                "doi": "10.1111/j.1467-9868.2010.00740.x",
            },
            "permutation_synchronization": (
                "Pachauri, Kondor, and Singh, NeurIPS 2013"
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
    parser.add_argument("--v447-calibration-dir", type=Path, required=True)
    parser.add_argument("--v451-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        relaxed_roots=args.relaxed_root,
        rescue_roots=args.rescue_root,
        v447_calibration_dir=args.v447_calibration_dir,
        v451_calibration_dir=args.v451_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
