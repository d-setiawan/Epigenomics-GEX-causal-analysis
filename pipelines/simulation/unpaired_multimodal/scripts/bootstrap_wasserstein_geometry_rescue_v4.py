#!/usr/bin/env python3
"""Calibrate the CA-BOSS v4.54 Wasserstein geometry rescue.

For one target block, the stage uses four fixed signals: ordinary statistical
context support, W2 global reporter co-assignment count, maximum normalized
W2 reporter cost, and mean W2 global exclusion margin.  Directional empirical
percentiles are multiplied in four nested folds.  A proposal must be a strict
unique mutual nearest neighbour in every fold, exceed every other-block null
margin, and survive at least 80 of 100 whole-world bootstrap refits.

The selected proposals are unioned with the target-safe v4.53 hypotheses and
returned as unexpanded shared-state equivalence hyperedges.  Other-block truth
is used only to identify calibration nulls.  Target truth, paired cells,
simulator state, modality identity, and feature identity are never used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_mutual_nearest_margin_rescue_v4 as v453  # noqa: E402
import crossfit_multivariate_reporter_rescue_v4 as v448  # noqa: E402
import diagnose_quantile_geometry_rescue_v4 as diagnostic  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import quantile_geometry_synchronization_evidence_v4 as geometry  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_wasserstein_geometry_rescue.v1"
ALGORITHM = "CA-BOSS v4.54 bootstrap-stable W2 global geometry rescue"
FEATURE_COLUMNS = (
    "statistical_context_support_count",
    "wasserstein2_global_coassignment_context_count",
    "wasserstein2_maximum_reporter_cost_ratio",
    "wasserstein2_mean_coassigned_exclusion_margin",
)
FEATURE_DIRECTIONS = (1.0, 1.0, -1.0, 1.0)
FEATURE_QUANTIZATION_DECIMALS = 12
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_FREQUENCY_THRESHOLD = 0.80
# The all-103-world diagnostic followed the two random panels (454000 and
# 454001), so its predeclared full-pool bootstrap seed was 454002.
BOOTSTRAP_BASE_SEED = 454002
DIAGNOSTIC_VARIANT_INDEX = list(diagnostic.VARIANTS).index("w2_global_mirror")
FIXED_SELECTION_RULE = (
    "use_the_truth_blind_full_fit_plus_two_disjoint_feature_fold_w2_"
    "quantile_geometry_evidence_for_each_target_and_calibration_fold_"
    "quantize_the_four_features_to_12_decimal_places_before_ranking_"
    "transform_statistical_context_support_global_w2_reporter_"
    "coassignment_negative_maximum_median_normalized_w2_reporter_cost_"
    "and_global_w2_exclusion_margin_to_other_three_training_block_"
    "empirical_percentiles_multiply_the_four_percentiles_require_a_"
    "strict_unique_mutual_nearest_assay_pair_in_every_nested_fold_and_"
    "a_minimum_absolute_margin_strictly_above_every_corresponding_"
    "other_block_null_margin_resample_whole_training_worlds_within_"
    "blocks_100_times_require_at_least_80_percent_selection_frequency_"
    "union_with_target_safe_v453_merge_connected_pairs_as_unexpanded_"
    "shared_state_ambiguity_hyperedges_and_never_claim_direct_structure_"
    "or_direction"
)

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
    "selected_by_target_safe_v453",
    "proposed_by_nominal_wasserstein_geometry",
    "bootstrap_selection_count",
    "bootstrap_replicates",
    "bootstrap_selection_frequency",
    "bootstrap_frequency_threshold",
    "passes_bootstrap_stability_guard",
    "newly_selected_by_v454",
    "retained_in_union",
    "merged_hyperedge_id",
]
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
    "wasserstein_geometry_hyperedge_id",
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


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    stripped = {
        key: value for key, value in payload.items() if key != "manifest_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def _pair_tuple(pair: frozenset[str]) -> tuple[str, str]:
    ordered = sorted(pair)
    if len(ordered) != 2:
        raise ValueError("Assay-pair key must contain exactly two modalities")
    return ordered[0], ordered[1]


def _world_key(block: str, seed: int) -> str:
    return f"{block}__world_seed{int(seed)}"


def _verified_evidence(
    directory: Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = diagnostic._verified_manifest(
        resolved / "evidence_manifest.json", geometry.SCHEMA
    )
    boundary = manifest.get("information_boundary", {})
    required = {
        "truth_or_scorer_opened": False,
        "hypothesis_selected": False,
        "existing_cached_worlds_only": True,
        "new_worlds_or_datasets_created": False,
        "heldout_background_features_only": True,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "cell_level_transport_coupling_constructed": False,
        "simulator_latent_state_used": False,
        "external_biological_features_or_priors_used": False,
        "modality_identity_used_as_model_feature": False,
        "candidate_feature_identity_used_as_model_feature": False,
    }
    for field, expected in required.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"Geometry evidence boundary mismatch: {field}")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("Geometry evidence lacks a formal pretruth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Geometry evidence created new data")
    source_hashes = manifest.get("source_sha256", {})
    for module in (
        geometry,
        diagnostic,
        diagnostic.synchronization,
        diagnostic.relaxed,
    ):
        path = Path(module.__file__).resolve()
        if source_hashes.get(path.name) != diagnostic._sha256(path):
            raise ValueError(f"Geometry evidence source changed: {path.name}")
    evidence_path = resolved / "quantile_geometry_pair_evidence.tsv"
    if diagnostic._sha256(evidence_path) != str(
        manifest.get("output_sha256", {}).get(evidence_path.name)
    ):
        raise ValueError("Geometry evidence output hash mismatch")
    table = pd.read_csv(evidence_path, sep="\t", keep_default_na=False)
    if list(table.columns) != geometry.EVIDENCE_COLUMNS:
        raise ValueError("Geometry evidence columns do not match the frozen schema")
    if len(table) != int(manifest.get("row_count", -1)):
        raise ValueError("Geometry evidence row count disagrees with its manifest")
    if sorted(table["world_key"].astype(str).unique()) != sorted(
        map(str, manifest.get("world_keys", []))
    ):
        raise ValueError("Geometry evidence world keys disagree with its manifest")
    return resolved, manifest, table


def _verified_v453(
    directory: Path,
    *,
    target_block: str,
    evidence_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = v453.v449._verified_payload(
        resolved / "calibration_manifest.json", "manifest_payload_sha256"
    )
    if manifest.get("schema_version") != v453.SCHEMA:
        raise ValueError("Unexpected v4.53 calibration schema")
    if str(manifest.get("target_block")) != target_block:
        raise ValueError("v4.53 calibration targets another block")
    if evidence_manifest.get("v453_manifest_payload_sha256_by_block", {}).get(
        target_block
    ) != str(manifest["manifest_payload_sha256"]):
        raise ValueError("Geometry evidence and v4.53 target manifest disagree")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
        "target_block_scorer_truth_opened": False,
        "target_world_truth_used_for_fit_or_selection": False,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "simulator_latent_state_used": False,
        "external_biological_features_or_priors_used": False,
        "simulator_specific_perfect_matching_used": False,
        "fixed_shared_relationship_count_used": False,
    }.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"v4.53 boundary mismatch: {field}")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if diagnostic._sha256(path) != str(expected):
            raise ValueError(f"v4.53 output hash mismatch: {name}")
    candidates = pd.read_csv(
        resolved / "bootstrap_mnn_candidates.tsv", sep="\t", keep_default_na=False
    )
    if len(candidates) and not candidates["hypothesis_selected"].map(
        synchronization._as_bool
    ).all():
        raise ValueError("v4.53 contains an unselected candidate")
    return resolved, manifest, candidates


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = (
        frame.loc[:, FEATURE_COLUMNS]
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
            _pair(row.left_modality, row.right_modality)
            in truth_pairs[str(row.world_key)]
            for row in frame.itertuples(index=False)
        ],
        dtype=bool,
    )


def _nested_selection(
    *,
    target: pd.DataFrame,
    calibration_blocks: Sequence[str],
    block_keys: Mapping[str, Sequence[str]],
    world_frames: Mapping[str, pd.DataFrame],
    truth_pairs: Mapping[str, set[frozenset[str]]],
    sampled_training_keys: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    dict[str, int],
]:
    scores: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    mutual: list[np.ndarray] = []
    null_maxima: dict[str, float] = {}
    null_counts: dict[str, int] = {}
    for calibration_block in calibration_blocks:
        training_keys: list[str] = []
        for training_block in calibration_blocks:
            if training_block == calibration_block:
                continue
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
        target_margin, target_mutual = v453._mutual_nearest_absolute_margin(
            target, target_score
        )
        calibration_score = _copula_score(training, calibration)
        calibration_margin, calibration_mutual = (
            v453._mutual_nearest_absolute_margin(calibration, calibration_score)
        )
        null = (~_truth_labels(calibration, truth_pairs)) & calibration_mutual
        if not null.any():
            raise ValueError(
                f"Calibration block {calibration_block} has no mutual-nearest null"
            )
        scores.append(target_score)
        margins.append(target_margin)
        mutual.append(target_mutual)
        null_counts[calibration_block] = int(null.sum())
        null_maxima[calibration_block] = float(calibration_margin[null].max())
    stacked_scores = np.vstack(scores).T
    stacked_margins = np.vstack(margins).T
    stacked_mutual = np.vstack(mutual).T
    selected = stacked_mutual.all(axis=1) & (
        stacked_margins.min(axis=1) > max(null_maxima.values())
    )
    return (
        selected,
        stacked_scores,
        stacked_margins,
        stacked_mutual,
        null_maxima,
        null_counts,
    )


def _baseline_components(
    candidates: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        pairs = v453.v449._parse_pair_payload(row.candidate_assay_pairs_json)
        features = v453.v449._parse_feature_pair_payload(
            row.candidate_cross_assay_feature_pairs_json
        )
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.53 candidate assay-pair count mismatch")
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.53 candidate feature-pair count mismatch")
        result[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(str(row.candidate_pair_selection_audit_json)),
                "source": "target_safe_v453",
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
            pair = _pair(selected["left_modality"], selected["right_modality"])
            sources.append(
                {
                    "pairs": {pair},
                    "features": v453.v449._parse_feature_pair_payload(
                        selected["candidate_cross_assay_feature_pairs_json"]
                    ),
                    "audit": [
                        {
                            "assay_pair": list(_pair_tuple(pair)),
                            "minimum_nested_absolute_margin": float(
                                selected["minimum_nested_absolute_margin"]
                            ),
                            "bootstrap_selection_frequency": float(
                                selected["bootstrap_selection_frequency"]
                            ),
                            "source": "v454_bootstrap_w2_geometry",
                        }
                    ],
                    "source": "v454_bootstrap_w2_geometry",
                }
            )
        all_pairs = (
            set().union(*(source["pairs"] for source in sources)) if sources else set()
        )
        for index, component in enumerate(
            v448.v447._connected_pair_components(all_pairs), start=1
        ):
            hyperedge_id = f"wasserstein_geometry_shared_state_{index:03d}"
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
                list(_pair_tuple(pair))
                for pair in sorted(component, key=_pair_tuple)
            ]
            for pair in component:
                ids[(world_key, pair)] = hyperedge_id
            output.append(
                {
                    "wasserstein_geometry_hyperedge_id": hyperedge_id,
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
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
                    "assay_participation_is_set_valued": len(component) != 1,
                    "exact_assay_participation_claimed": len(component) == 1,
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_wasserstein_geometry_stable_ambiguity"
                        if len(component) != 1
                        else "selected_as_wasserstein_geometry_stable_point_hypothesis"
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
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    evidence_path, evidence_manifest, evidence = _verified_evidence(evidence_dir)
    available_blocks = sorted(evidence["block"].astype(str).unique())
    target_block = str(target_block)
    if target_block not in available_blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    calibration_blocks = [block for block in available_blocks if block != target_block]
    if len(calibration_blocks) != 4:
        raise ValueError("v4.54 requires exactly four non-target blocks")
    v453_path, v453_manifest, v453_candidates = _verified_v453(
        v453_calibration_dir,
        target_block=target_block,
        evidence_manifest=evidence_manifest,
    )
    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)
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
        for block in available_blocks
    }
    target_keys = block_keys[target_block]
    if set(map(str, v453_manifest.get("target_world_keys", []))) != set(target_keys):
        raise ValueError("v4.53 and geometry target worlds differ")
    target = pd.concat([world_frames[key] for key in target_keys], ignore_index=True)
    truth_pairs: dict[str, set[frozenset[str]]] = {}
    truth_hashes: dict[str, str] = {}
    for block in calibration_blocks:
        for key in block_keys[block]:
            seed = int(world_frames[key]["seed"].iloc[0])
            truth_path = (
                formal_root
                / block
                / f"world_seed{seed}"
                / "scorer"
                / "truth_typed_edges.tsv"
            )
            if target_block in truth_path.parts:
                raise RuntimeError("Attempted to open target-block truth")
            if truth_path.is_symlink() or not truth_path.is_file():
                raise FileNotFoundError(truth_path)
            truth_pairs[key] = v448.v447._truth_assay_pairs(truth_path)
            truth_hashes[key] = diagnostic._sha256(truth_path)

    (
        nominal,
        nominal_scores,
        nominal_margins,
        nominal_mutual,
        null_maxima,
        null_counts,
    ) = _nested_selection(
        target=target,
        calibration_blocks=calibration_blocks,
        block_keys=block_keys,
        world_frames=world_frames,
        truth_pairs=truth_pairs,
    )
    null_envelope = max(null_maxima.values())
    fold_rows = [
        {
            "target_block": target_block,
            "calibration_block": calibration_block,
            "training_blocks": ";".join(
                block for block in calibration_blocks if block != calibration_block
            ),
            "training_world_count": sum(
                len(block_keys[block])
                for block in calibration_blocks
                if block != calibration_block
            ),
            "training_pair_count": sum(
                len(world_frames[key])
                for block in calibration_blocks
                if block != calibration_block
                for key in block_keys[block]
            ),
            "calibration_world_count": len(block_keys[calibration_block]),
            "calibration_pair_count": sum(
                len(world_frames[key]) for key in block_keys[calibration_block]
            ),
            "calibration_mutual_nearest_null_pair_count": null_counts[
                calibration_block
            ],
            "maximum_calibration_null_absolute_margin": null_maxima[
                calibration_block
            ],
            "feature_columns_json": json.dumps(
                list(FEATURE_COLUMNS), separators=(",", ":")
            ),
            "feature_directions_json": json.dumps(
                list(FEATURE_DIRECTIONS), separators=(",", ":")
            ),
        }
        for calibration_block in calibration_blocks
    ]

    baseline = _baseline_components(v453_candidates)
    baseline_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs[key].update(component["pairs"])
    identities = [
        (str(row.world_key), _pair(row.left_modality, row.right_modality))
        for row in target.itertuples(index=False)
    ]
    baseline_mask = np.asarray(
        [pair in baseline_pairs.get(key, set()) for key, pair in identities],
        dtype=bool,
    )
    proposal = nominal & ~baseline_mask
    selection_count = np.zeros(len(target), dtype=np.int64)
    replicate_rows: list[dict[str, Any]] = []
    target_index = available_blocks.index(target_block)
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
                        [
                            BOOTSTRAP_BASE_SEED,
                            DIAGNOSTIC_VARIANT_INDEX,
                            target_index,
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
        selected, _, _, _, replicate_null, _ = _nested_selection(
            target=target,
            calibration_blocks=calibration_blocks,
            block_keys=block_keys,
            world_frames=world_frames,
            truth_pairs=truth_pairs,
            sampled_training_keys=sampled,
        )
        selection_count += selected.astype(np.int64)
        replicate_rows.append(
            {
                "target_block": target_block,
                "bootstrap_seed": BOOTSTRAP_BASE_SEED,
                "bootstrap_replicate": replicate + 1,
                "calibration_blocks": ";".join(calibration_blocks),
                "calibration_null_margin_maxima_json": json.dumps(
                    replicate_null, sort_keys=True, separators=(",", ":")
                ),
                "null_margin_upper_envelope": max(replicate_null.values()),
                "raw_selected_pair_count": int(selected.sum()),
            }
        )
    frequencies = selection_count.astype(float) / float(BOOTSTRAP_REPLICATES)
    stable = proposal & (frequencies >= BOOTSTRAP_FREQUENCY_THRESHOLD)

    pair_rows: list[dict[str, Any]] = []
    stable_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(target.itertuples(index=False)):
        values: dict[str, Any] = {
            "world_key": str(row.world_key),
            "block": str(row.block),
            "seed": int(row.seed),
            "left_modality": str(row.left_modality),
            "right_modality": str(row.right_modality),
            **{column: float(getattr(row, column)) for column in FEATURE_COLUMNS},
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
            "selected_by_target_safe_v453": bool(baseline_mask[index]),
            "proposed_by_nominal_wasserstein_geometry": bool(proposal[index]),
            "bootstrap_selection_count": int(selection_count[index]),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_selection_frequency": float(frequencies[index]),
            "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
            "passes_bootstrap_stability_guard": bool(stable[index]),
            "newly_selected_by_v454": bool(stable[index]),
            "retained_in_union": bool(baseline_mask[index] or stable[index]),
            "merged_hyperedge_id": "",
            "candidate_cross_assay_feature_pairs_json": str(
                row.candidate_cross_assay_feature_pairs_json
            ),
        }
        pair_rows.append(values)
        if stable[index]:
            stable_rows[str(row.world_key)].append(values)
    candidates, merged_ids = _union_candidates(
        target_keys=target_keys,
        world_frames=world_frames,
        baseline=baseline,
        stable_rows=stable_rows,
    )
    for row in pair_rows:
        row["merged_hyperedge_id"] = merged_ids.get(
            (
                str(row["world_key"]),
                _pair(row["left_modality"], row["right_modality"]),
            ),
            "",
        )
    pair_audit = pd.DataFrame(pair_rows, columns=PAIR_AUDIT_COLUMNS)
    fold_audit = pd.DataFrame(fold_rows, columns=FOLD_AUDIT_COLUMNS)
    replicate_audit = pd.DataFrame(replicate_rows, columns=REPLICATE_COLUMNS)

    out_dir.mkdir(parents=True)
    outputs = {
        "bootstrap_wasserstein_candidates.tsv": candidates,
        "bootstrap_wasserstein_pair_audit.tsv": pair_audit,
        "bootstrap_wasserstein_fold_audit.tsv": fold_audit,
        "bootstrap_wasserstein_replicate_audit.tsv": replicate_audit,
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
        "available_blocks": available_blocks,
        "calibration_blocks": calibration_blocks,
        "baseline_algorithm": "CA-BOSS v4.53 target-safe stable union",
        "model_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_directions": list(FEATURE_DIRECTIONS),
        "model_feature_imputation": "nonfinite_or_missing_to_zero",
        "model_feature_quantization_decimals": FEATURE_QUANTIZATION_DECIMALS,
        "pair_score": "product_of_four_directional_training_fold_empirical_percentiles",
        "local_pair_rule": "strict_unique_mutual_nearest_neighbour_in_every_nested_fold",
        "margin": "candidate_score_minus_larger_endpoint_specific_competitor_maximum",
        "nominal_null_margin_upper_envelope": null_envelope,
        "nominal_proposal_pair_count": int(proposal.sum()),
        "bootstrap_unit": "whole_training_world_within_seed_block",
        "bootstrap_sampling": "with_replacement_same_block_world_count",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "bootstrap_diagnostic_variant_index": DIAGNOSTIC_VARIANT_INDEX,
        "bootstrap_target_block_index": target_index,
        "bootstrap_seed_sequence_fields": [
            "base_seed",
            "diagnostic_variant_index",
            "target_block_index",
            "bootstrap_replicate_zero_based",
            "calibration_block_index",
            "training_block_index",
        ],
        "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
        "newly_selected_by_v454_assay_pair_count": int(stable.sum()),
        "rejected_unstable_nominal_proposal_count": int(proposal.sum() - stable.sum()),
        "selected_union_hyperedge_count": len(candidates),
        "selected_union_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "perfect_matching_assumed": False,
        "fixed_shared_relationship_count_assumed": False,
        "shared_component_rank_three_assumed": True,
        "supervised_positive_labels_used_for_score_fit": False,
        "other_block_truth_used_only_to_identify_calibration_nulls": True,
        "information_boundary": {
            "other_block_simulation_truth_used_to_identify_nulls": True,
            "other_block_positive_labels_used_for_score_fit": False,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "target_safe_v453_used_only_as_target_baseline_union": True,
            "known_sentinel_or_false_pairs_used_for_empirical_null": True,
            "heldout_background_features_only_for_geometry": True,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "cell_level_transport_coupling_constructed": False,
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
        "published_method_inspiration": {
            "unpaired_multi_domain_causal_representation": {
                "citation": "Sturma, Squires, Drton, and Uhler, NeurIPS 2023",
                "doi": "10.52202/075280-1496",
            },
            "unpaired_multiomics_optimal_transport": {
                "citation": "Demetci et al., Journal of Computational Biology 2022",
                "doi": "10.1089/cmb.2021.0446",
                "relationship": "Wasserstein_geometry_inspiration_without_cell_coupling",
            },
            "stability_selection": {
                "citation": "Meinshausen and Buehlmann, JRSS B 2010",
                "doi": "10.1111/j.1467-9868.2010.00740.x",
            },
        },
        "source_sha256": {
            module_path.name: diagnostic._sha256(module_path)
            for module_path in (
                Path(__file__).resolve(),
                Path(geometry.__file__).resolve(),
                Path(diagnostic.__file__).resolve(),
                Path(v453.__file__).resolve(),
                Path(v448.__file__).resolve(),
                Path(synchronization.__file__).resolve(),
            )
        },
        "input_sha256": {
            "geometry_evidence": {
                "evidence_manifest.json": diagnostic._sha256(
                    evidence_path / "evidence_manifest.json"
                ),
                "quantile_geometry_pair_evidence.tsv": diagnostic._sha256(
                    evidence_path / "quantile_geometry_pair_evidence.tsv"
                ),
            },
            "target_safe_v453_calibration": {
                "calibration_manifest.json": diagnostic._sha256(
                    v453_path / "calibration_manifest.json"
                ),
                **{
                    name: diagnostic._sha256(v453_path / str(name))
                    for name in v453_manifest.get("output_sha256", {})
                },
            },
            "calibration_truth_typed_edges": truth_hashes,
        },
        "output_sha256": {
            name: diagnostic._sha256(out_dir / name) for name in outputs
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--v453-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = calibrate_target_block(
        evidence_dir=args.evidence_dir,
        v453_calibration_dir=args.v453_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
