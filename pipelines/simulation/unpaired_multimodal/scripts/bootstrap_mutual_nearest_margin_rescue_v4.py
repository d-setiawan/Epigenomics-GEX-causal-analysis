#!/usr/bin/env python3
"""Bootstrap-stable mutual-nearest-margin rescue for CA-BOSS v4.53.

This stage reuses the four directional empirical-rank signals from v4.52, but
compares each assay pair only with pairs incident to either of its endpoints.
A proposal must be the unique mutual nearest neighbour in every nested fold,
and its minimum absolute copula-score margin must strictly exceed the largest
corresponding mutual-neighbour null margin in the other blocks.  The complete
rule is rerun in 100 whole-training-world bootstrap refits and requires at
least 80% selection frequency.

The stable additions are unioned with target-safe v4.52 hypotheses.  The rule
does not require a perfect matching or a fixed number of latent relationships;
it may abstain for any assay.  It uses no paired cells, modality or feature
identity, simulator latent state, external biology, direct cross-assay edge,
or direction.
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

import bootstrap_empirical_copula_pair_rescue_v4 as v452  # noqa: E402
import crossfit_multivariate_reporter_rescue_v4 as v448  # noqa: E402
import crossfit_sparse_reporter_completion_v4 as v449  # noqa: E402
import heldout_synchronized_majority_rescue_v4 as guarded_rescue  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import relaxed_reporter_synchronization_evidence_v4 as relaxed  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_mutual_nearest_margin_rescue.v1"
ALGORITHM = "CA-BOSS v4.53 bootstrap-stable mutual-nearest-margin rescue"
FEATURE_COLUMNS = v452.FEATURE_COLUMNS
FEATURE_DIRECTIONS = v452.FEATURE_DIRECTIONS
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_FREQUENCY_THRESHOLD = 0.80
BOOTSTRAP_BASE_SEED = 453500
FIXED_SELECTION_RULE = (
    "reuse_the_v452_four_signal_directional_empirical_copula_score_for_each_"
    "nested_fold_compare_each_pair_with_every_other_pair_incident_to_each_"
    "endpoint_require_strict_unique_mutual_nearest_neighbour_status_in_all_"
    "four_folds_take_the_minimum_absolute_score_margin_over_folds_set_the_"
    "threshold_to_the_maximum_corresponding_other_block_mutual_neighbour_"
    "null_margin_require_strict_exceedance_treat_only_pairs_not_in_target_"
    "safe_v452_as_proposals_resample_whole_training_worlds_with_replacement_"
    "within_blocks_100_times_require_proposal_selection_frequency_at_least_"
    "80_percent_union_with_target_safe_v452_merge_connected_pairs_as_"
    "unexpanded_ambiguity_hyperedges_never_require_a_perfect_matching_or_"
    "fixed_relationship_count_and_never_claim_direct_structure_or_direction"
)

FOLD_AUDIT_COLUMNS = [
    "target_block",
    "calibration_block",
    "training_blocks",
    "training_world_count",
    "training_pair_count",
    "calibration_world_count",
    "calibration_pair_count",
    "calibration_mutual_nearest_null_pair_count",
    "maximum_calibration_null_absolute_margin",
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
    "nested_mutual_nearest_flags_json",
    "nested_absolute_margins_json",
    "minimum_nested_absolute_margin",
    "calibration_null_margin_upper_envelope",
    "mutual_nearest_in_every_fold",
    "strictly_exceeds_calibration_null_margin_upper_envelope",
    "selected_by_target_safe_v452",
    "proposed_by_nominal_mutual_nearest_margin",
    "bootstrap_selection_count",
    "bootstrap_replicates",
    "bootstrap_selection_frequency",
    "bootstrap_frequency_threshold",
    "passes_bootstrap_stability_guard",
    "newly_selected_by_v453",
    "retained_in_union",
    "merged_hyperedge_id",
]

REPLICATE_COLUMNS = [
    "target_block",
    "bootstrap_seed",
    "bootstrap_replicate",
    "calibration_blocks",
    "calibration_null_margin_maxima_json",
    "null_margin_upper_envelope",
    "raw_selected_pair_count",
]

CANDIDATE_COLUMNS = [
    "mutual_nearest_hyperedge_id",
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


def _verified_v452_calibration(
    directory: Path, *, target_block: str
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = v449._verified_payload(
        resolved / "calibration_manifest.json", "manifest_payload_sha256"
    )
    if manifest.get("schema_version") != v452.SCHEMA:
        raise ValueError("Unexpected v4.52 calibration schema")
    boundary = manifest.get("information_boundary", {})
    required = {
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
    }
    for field, expected in required.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.52 information boundary mismatch: {field}")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.52 lacks a target-specific pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.52 created new data")
    if str(manifest.get("target_block")) != str(target_block):
        raise ValueError("v4.52 calibration targets a different block")
    if list(manifest.get("model_feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("v4.52 feature set changed")
    if [float(value) for value in manifest.get("model_feature_directions", [])] != list(
        FEATURE_DIRECTIONS
    ):
        raise ValueError("v4.52 feature directions changed")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"v4.52 output hash mismatch: {name}")
    candidates = pd.read_csv(
        resolved / "bootstrap_copula_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if len(candidates) and not candidates["hypothesis_selected"].map(_as_bool).all():
        raise ValueError("v4.52 contains an unselected candidate")
    pair_audit = pd.read_csv(
        resolved / "bootstrap_copula_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    return resolved, manifest, candidates, pair_audit


def _mutual_nearest_absolute_margin(
    frame: pd.DataFrame, scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint-local absolute margins and strict mutual-neighbour flags."""

    values = np.asarray(scores, dtype=float)
    if len(frame) != len(values):
        raise ValueError("Pair frame and copula scores differ in length")
    if "world_key" not in frame:
        raise ValueError("Pair frame lacks world_key")
    margins = np.zeros(len(frame), dtype=float)
    mutual = np.zeros(len(frame), dtype=bool)
    grouped_positions = frame.groupby("world_key", sort=False).indices
    for positions in grouped_positions.values():
        positions = np.asarray(positions, dtype=int)
        pairs = [
            _pair_key(frame.iloc[index]["left_modality"], frame.iloc[index]["right_modality"])
            for index in positions
        ]
        for local_index, position in enumerate(positions):
            pair = pairs[local_index]
            endpoint_maxima: list[float] = []
            is_mutual = True
            for endpoint in pair:
                competitors = [
                    values[positions[other_index]]
                    for other_index, other_pair in enumerate(pairs)
                    if other_index != local_index and endpoint in other_pair
                ]
                if not competitors:
                    is_mutual = False
                    endpoint_maxima.append(values[position])
                    continue
                maximum = float(max(competitors))
                endpoint_maxima.append(maximum)
                is_mutual = is_mutual and bool(values[position] > maximum)
            margins[position] = float(values[position] - max(endpoint_maxima))
            mutual[position] = is_mutual
    return margins, mutual


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
            raise ValueError("v4.52 candidate assay-pair count mismatch")
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.52 candidate feature-pair count mismatch")
        components[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(str(row.candidate_pair_selection_audit_json)),
                "source": "target_safe_v452",
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
            pair = _pair_key(selected["left_modality"], selected["right_modality"])
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
                            "minimum_nested_absolute_margin": float(
                                selected["minimum_nested_absolute_margin"]
                            ),
                            "bootstrap_selection_frequency": float(
                                selected["bootstrap_selection_frequency"]
                            ),
                            "source": "v453_bootstrap_mutual_nearest_margin",
                        }
                    ],
                    "source": "v453_bootstrap_mutual_nearest_margin",
                }
            )
        all_pairs = (
            set().union(*(source["pairs"] for source in sources)) if sources else set()
        )
        for component_index, component_pairs in enumerate(
            v448.v447._connected_pair_components(all_pairs), start=1
        ):
            hyperedge_id = f"mutual_nearest_shared_state_{component_index:03d}"
            modalities = sorted(set().union(*(set(pair) for pair in component_pairs)))
            component_sources = [
                source for source in sources if set(source["pairs"]) & component_pairs
            ]
            features = set().union(*(source["features"] for source in component_sources))
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
                    "mutual_nearest_hyperedge_id": hyperedge_id,
                    "world_key": world_key,
                    "block": block,
                    "seed": seed,
                    "modalities": ";".join(modalities),
                    "n_modalities": len(modalities),
                    "candidate_assay_pairs_json": json.dumps(pair_payload, separators=(",", ":")),
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
                    "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
                    "assay_participation_is_set_valued": len(component_pairs) != 1,
                    "exact_assay_participation_claimed": len(component_pairs) == 1,
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_mutual_nearest_stable_ambiguity"
                        if len(component_pairs) != 1
                        else "selected_as_mutual_nearest_stable_point_hypothesis"
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


def _nested_mutual_nearest_selection(
    *,
    target_model: pd.DataFrame,
    calibration_blocks: Sequence[str],
    model_frames: Mapping[str, pd.DataFrame],
    block_keys: Mapping[str, Sequence[str]],
    truth_pairs_by_world: Mapping[str, set[frozenset[str]]],
    sampled_training_keys: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, int]]:
    fold_scores: list[np.ndarray] = []
    fold_margins: list[np.ndarray] = []
    fold_mutual: list[np.ndarray] = []
    null_maxima: dict[str, float] = {}
    null_counts: dict[str, int] = {}
    for calibration_block in calibration_blocks:
        training_blocks = [
            block for block in calibration_blocks if block != calibration_block
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
        target_scores = v452._empirical_copula_product(training, target_model)
        target_margins, target_mutual = _mutual_nearest_absolute_margin(
            target_model, target_scores
        )
        calibration_scores = v452._empirical_copula_product(
            training, calibration_frame
        )
        calibration_margins, calibration_mutual = _mutual_nearest_absolute_margin(
            calibration_frame, calibration_scores
        )
        calibration_null = (~calibration_labels) & calibration_mutual
        if not calibration_null.any():
            raise ValueError("Calibration fold contains no mutual-nearest null pair")
        fold_scores.append(target_scores)
        fold_margins.append(target_margins)
        fold_mutual.append(target_mutual)
        null_counts[calibration_block] = int(calibration_null.sum())
        null_maxima[calibration_block] = float(
            calibration_margins[calibration_null].max()
        )
    stacked_scores = np.vstack(fold_scores).T
    stacked_margins = np.vstack(fold_margins).T
    stacked_mutual = np.vstack(fold_mutual).T
    minimum_margin = stacked_margins.min(axis=1)
    all_fold_mutual = stacked_mutual.all(axis=1)
    selected = all_fold_mutual & (minimum_margin > max(null_maxima.values()))
    return (
        selected,
        stacked_scores,
        stacked_margins,
        stacked_mutual,
        null_maxima,
        null_counts,
    )


def calibrate_target_block(
    *,
    relaxed_roots: Sequence[Path],
    rescue_roots: Sequence[Path],
    v452_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Select target pairs with nested local margins and world stability."""

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
    calibration_blocks = [block for block in available_blocks if block != target_block]
    if len(calibration_blocks) < 3:
        raise ValueError("At least three non-target blocks are required")

    v452_dir, v452_manifest, v452_candidates, _ = _verified_v452_calibration(
        v452_calibration_dir, target_block=target_block
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
        if block != str(rescue_manifest["block"]) or seed != int(rescue_manifest["seed"]):
            raise ValueError(f"Upstream world identities disagree: {key}")
        evidence_path = evidence_dir / "relaxed_reporter_synchronization_pairs.tsv"
        frame = pd.read_csv(evidence_path, sep="\t", keep_default_na=False)
        frame["world_key"] = key
        frame["block"] = block
        frame["seed"] = seed
        pair_keys = [
            _pair_key(left, right)
            for left, right in zip(frame["left_modality"], frame["right_modality"])
        ]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError(f"Duplicate assay pair in reporter evidence: {key}")
        evidence_frames[key] = frame
        world_block[key] = block

        rescue_path = rescue_dir / "heldout_synchronized_majority_candidates.tsv"
        rescue_candidates = pd.read_csv(rescue_path, sep="\t", keep_default_na=False)
        evidence_pair_set = set(pair_keys)
        for row in rescue_candidates.itertuples(index=False):
            if not _as_bool(row.hypothesis_selected):
                continue
            modalities = [value for value in str(row.modalities).split(";") if value]
            if len(modalities) != 2:
                raise ValueError("v4.46 candidate is not an assay pair")
            pair = frozenset(modalities)
            if pair not in evidence_pair_set:
                raise ValueError("v4.46 pair lacks relaxed reporter evidence")
            truth_blind_v446_pairs[key].add(pair)

        v444_mask = frame["v444_assay_pair_already_represented"].map(_as_bool)
        v446_mask = np.asarray(
            [pair in truth_blind_v446_pairs[key] for pair in pair_keys], dtype=bool
        )
        model_frames[key] = frame.loc[
            ~(v444_mask.to_numpy(dtype=bool) | v446_mask)
        ].copy()

        evidence_input_hashes[key] = {
            "evidence_manifest.json": synchronization._sha256(
                evidence_dir / "evidence_manifest.json"
            ),
            "relaxed_reporter_synchronization_pairs.tsv": synchronization._sha256(
                evidence_path
            ),
        }
        rescue_input_hashes[key] = {
            "rescue_manifest.json": synchronization._sha256(
                rescue_dir / "rescue_manifest.json"
            ),
            "heldout_synchronized_majority_candidates.tsv": synchronization._sha256(
                rescue_path
            ),
        }
        if block == target_block:
            target_keys.append(key)
            continue
        truth_path = formal_root / block / f"world_seed{seed}" / "scorer" / "truth_typed_edges.tsv"
        if target_block in truth_path.parts:
            raise RuntimeError("Attempted to open target-block truth")
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        truth_pairs_by_world[key] = v448.v447._truth_assay_pairs(truth_path)
        training_truth_hashes[key] = synchronization._sha256(truth_path)

    if not target_keys:
        raise ValueError("Target block has no cached worlds")
    target_key_set = set(target_keys)
    if set(map(str, v452_manifest.get("target_world_keys", []))) != target_key_set:
        raise ValueError("v4.52 and v4.53 target worlds differ")
    if len(v452_candidates) and not v452_candidates["world_key"].astype(str).isin(
        target_key_set
    ).all():
        raise ValueError("v4.52 contains a non-target candidate")

    v452_inputs = v452_manifest.get("input_sha256", {})
    if v452_inputs.get("relaxed_evidence_worlds") != evidence_input_hashes:
        raise ValueError("v4.52 relaxed-evidence binding mismatch")
    if v452_inputs.get("v446_guarded_rescue_worlds") != rescue_input_hashes:
        raise ValueError("v4.52 v4.46 binding mismatch")
    if v452_inputs.get("calibration_truth_typed_edges") != training_truth_hashes:
        raise ValueError("v4.52 calibration-truth binding mismatch")

    block_keys = {
        block: sorted(key for key, value in world_block.items() if value == block)
        for block in available_blocks
    }
    target_model = pd.concat(
        [model_frames[key] for key in sorted(target_keys)], ignore_index=True
    )
    if target_model.empty:
        raise ValueError("Target block has no mutual-neighbour-eligible assay pairs")
    identities = [
        (str(row.world_key), _pair_key(row.left_modality, row.right_modality))
        for row in target_model.itertuples(index=False)
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Target pair identities are not unique")

    (
        nominal_selected,
        nominal_scores,
        nominal_margins,
        nominal_mutual,
        null_maxima,
        null_counts,
    ) = _nested_mutual_nearest_selection(
        target_model=target_model,
        calibration_blocks=calibration_blocks,
        model_frames=model_frames,
        block_keys=block_keys,
        truth_pairs_by_world=truth_pairs_by_world,
    )
    null_envelope = max(null_maxima.values())
    fold_rows = []
    for calibration_block in calibration_blocks:
        training_blocks = [block for block in calibration_blocks if block != calibration_block]
        fold_rows.append(
            {
                "target_block": target_block,
                "calibration_block": calibration_block,
                "training_blocks": ";".join(training_blocks),
                "training_world_count": sum(len(block_keys[block]) for block in training_blocks),
                "training_pair_count": sum(
                    len(model_frames[key])
                    for block in training_blocks
                    for key in block_keys[block]
                ),
                "calibration_world_count": len(block_keys[calibration_block]),
                "calibration_pair_count": sum(
                    len(model_frames[key]) for key in block_keys[calibration_block]
                ),
                "calibration_mutual_nearest_null_pair_count": null_counts[calibration_block],
                "maximum_calibration_null_absolute_margin": null_maxima[calibration_block],
                "feature_columns_json": json.dumps(list(FEATURE_COLUMNS), separators=(",", ":")),
                "feature_directions_json": json.dumps(list(FEATURE_DIRECTIONS), separators=(",", ":")),
            }
        )
    fold_audit = pd.DataFrame(fold_rows, columns=FOLD_AUDIT_COLUMNS)

    baseline = _baseline_components(v452_candidates)
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
                block for block in calibration_blocks if block != calibration_block
            ]
            for training_index, training_block in enumerate(training_blocks):
                keys = block_keys[training_block]
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [bootstrap_seed, replicate, calibration_index, training_index]
                    )
                )
                sampled[(calibration_block, training_block)] = [
                    keys[int(index)]
                    for index in rng.integers(0, len(keys), size=len(keys))
                ]
        selected, _, _, _, replicate_null_maxima, _ = _nested_mutual_nearest_selection(
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
                "calibration_null_margin_maxima_json": json.dumps(
                    replicate_null_maxima, sort_keys=True, separators=(",", ":")
                ),
                "null_margin_upper_envelope": max(replicate_null_maxima.values()),
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
                    [float(value) for value in nominal_scores[index]], separators=(",", ":")
                ),
                "nested_mutual_nearest_flags_json": json.dumps(
                    [bool(value) for value in nominal_mutual[index]], separators=(",", ":")
                ),
                "nested_absolute_margins_json": json.dumps(
                    [float(value) for value in nominal_margins[index]], separators=(",", ":")
                ),
                "minimum_nested_absolute_margin": float(nominal_margins[index].min()),
                "calibration_null_margin_upper_envelope": null_envelope,
                "mutual_nearest_in_every_fold": bool(nominal_mutual[index].all()),
                "strictly_exceeds_calibration_null_margin_upper_envelope": bool(
                    nominal_margins[index].min() > null_envelope
                ),
                "selected_by_target_safe_v452": bool(baseline_mask[index]),
                "proposed_by_nominal_mutual_nearest_margin": bool(proposal_mask[index]),
                "bootstrap_selection_count": int(selection_counts[index]),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_selection_frequency": float(frequencies[index]),
                "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
                "passes_bootstrap_stability_guard": bool(stable_mask[index]),
                "newly_selected_by_v453": bool(stable_mask[index]),
                "retained_in_union": bool(baseline_mask[index] or stable_mask[index]),
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
        "bootstrap_mnn_candidates.tsv": candidates,
        "bootstrap_mnn_pair_audit.tsv": pair_audit,
        "bootstrap_mnn_fold_audit.tsv": fold_audit,
        "bootstrap_mnn_replicate_audit.tsv": replicate_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    v452_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v452_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v452_dir / str(name))
            for name in v452_manifest.get("output_sha256", {})
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
        "baseline_algorithm": "CA-BOSS v4.52 target-safe stable union",
        "model_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_directions": list(FEATURE_DIRECTIONS),
        "model_feature_imputation": "nonfinite_or_missing_to_zero",
        "pair_score": "v452_product_of_directional_training_fold_ecdf_percentiles",
        "local_competitor_set": "all_other_eligible_pairs_incident_to_either_endpoint_within_world",
        "local_pair_rule": "strict_unique_mutual_nearest_neighbour_in_every_nested_fold",
        "margin": "candidate_copula_score_minus_larger_endpoint_specific_competitor_maximum",
        "nested_margin_score": "minimum_absolute_margin_over_four_folds",
        "supervised_positive_labels_used_for_score_fit": False,
        "other_block_truth_used_only_to_identify_calibration_nulls": True,
        "perfect_matching_assumed": False,
        "fixed_shared_relationship_count_assumed": False,
        "nominal_null_margin_upper_envelope": null_envelope,
        "nominal_proposal_pair_count": int(proposal_mask.sum()),
        "bootstrap_unit": "whole_training_world_within_seed_block",
        "bootstrap_sampling": "with_replacement_same_block_world_count",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "bootstrap_target_block_seed": bootstrap_seed,
        "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
        "bootstrap_threshold_comparison": "inclusive_greater_than_or_equal",
        "newly_selected_by_v453_assay_pair_count": int(stable_mask.sum()),
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
            int(candidates["candidate_assay_pair_count"].max()) if len(candidates) else 0
        ),
        "information_boundary": {
            "other_block_simulation_truth_used_to_identify_nulls": True,
            "other_block_positive_labels_used_for_score_fit": False,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "target_safe_v452_used_only_as_target_baseline_union": True,
            "known_sentinel_or_false_pairs_used_for_empirical_null": True,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
            "simulator_specific_perfect_matching_used": False,
            "fixed_shared_relationship_count_used": False,
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
            Path(__file__).name: synchronization._sha256(Path(__file__).resolve())
        },
        "input_sha256": {
            "relaxed_evidence_worlds": evidence_input_hashes,
            "v446_guarded_rescue_worlds": rescue_input_hashes,
            "target_safe_v452_calibration": v452_input_hashes,
            "calibration_truth_typed_edges": training_truth_hashes,
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "mutual_nearest_neighbours": {
                "citation": "Haghverdi, Lun, Morgan, and Marioni, Nature Biotechnology 2018",
                "title": "Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors",
                "doi": "10.1038/nbt.4091",
                "relationship": "inspiration_for_partial_local_reciprocal_matching_not_batch_correction_or_a_direct_application",
            },
            "empirical_copula_tail_scoring": {
                "citation": "Li, Zhao, Botta, Ionescu, and Hu, ICDM 2020",
                "doi": "10.1109/ICDM50108.2020.00135",
            },
            "stability_selection": {
                "citation": "Meinshausen and Buehlmann, JRSS B 2010",
                "doi": "10.1111/j.1467-9868.2010.00740.x",
            },
        },
    }
    manifest["manifest_payload_sha256"] = synchronization._canonical_sha256(manifest)
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxed-root", type=Path, action="append", required=True)
    parser.add_argument("--rescue-root", type=Path, action="append", required=True)
    parser.add_argument("--v452-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        relaxed_roots=args.relaxed_root,
        rescue_roots=args.rescue_root,
        v452_calibration_dir=args.v452_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
