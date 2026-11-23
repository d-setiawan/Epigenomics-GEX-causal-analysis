#!/usr/bin/env python3
"""Diagnose joint sliced-Wasserstein reporter registration for CA-BOSS v4.

The v4.54 bridge compares one ICA component marginal at a time.  This cached-
world diagnostic instead compares the complete rank-three component
distribution under all 48 signed permutations.  Wasserstein-2 costs are
averaged over a deterministic, signed-permutation-closed set of 49 projection
directions.  A second view rank-normalizes each component first and therefore
compares copula geometry only.  For every possible reporter-component match,
the remaining coordinates and signs are optimized out; the resulting 3x3 cost
matrices enter the existing exact multi-assay permutation synchronizer.

No row correspondence or transport plan is constructed.  Only already-frozen
held-out-background component scores and reporter assignments are read.  The
diagnostic randomly samples existing worlds, uses truth only for other-block
empirical nulls during selection, and opens target truth only to score frozen
diagnostic predictions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from itertools import combinations, permutations, product
import json
from math import gcd
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_null_calibrated_stability_rescue_v4 as v455  # noqa: E402
import diagnose_quantile_geometry_rescue_v4 as geometry  # noqa: E402


SCHEMA = "ca_boss_v4.joint_sliced_geometry_rescue_diagnostic.v1"
ALGORITHM = "CA-BOSS v4 joint sliced-Wasserstein geometry diagnostic"
DEFAULT_RANDOM_SEED = 456000
DEFAULT_WORLDS_PER_BLOCK = 8
DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_BOOTSTRAP_FREQUENCY = 0.80
QUANTILE_GRID = geometry.QUANTILE_GRID
METRICS = ("joint_sliced_w2", "copula_sliced_w2")
FEATURE_SUFFIXES = (
    "local_mutual_context_count",
    "heldout_local_mutual_context_count",
    "mean_local_relative_margin",
    "minimum_local_relative_margin",
    "maximum_reporter_cost_ratio",
    "mean_reporter_cost_ratio",
    "global_coassignment_context_count",
    "heldout_global_coassignment_context_count",
    "mean_coassigned_exclusion_margin",
)
VARIANTS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "joint_global_mirror": (
        (
            "statistical_context_support_count",
            "joint_sliced_w2_global_coassignment_context_count",
            "joint_sliced_w2_maximum_reporter_cost_ratio",
            "joint_sliced_w2_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "copula_global_mirror": (
        (
            "statistical_context_support_count",
            "copula_sliced_w2_global_coassignment_context_count",
            "copula_sliced_w2_maximum_reporter_cost_ratio",
            "copula_sliced_w2_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
}


def _canonical_direction(values: Sequence[int]) -> tuple[tuple[int, int, int], int]:
    vector = tuple(int(value) for value in values)
    if len(vector) != 3 or not any(vector):
        raise ValueError("A projection direction must be a nonzero 3-vector")
    common = reduce(gcd, (abs(value) for value in vector))
    primitive = tuple(value // common for value in vector)
    first = next(value for value in primitive if value)
    if first < 0:
        return tuple(-value for value in primitive), -1  # type: ignore[return-value]
    return primitive, 1  # type: ignore[return-value]


def _integer_projection_directions() -> tuple[tuple[int, int, int], ...]:
    directions = {
        _canonical_direction(values)[0]
        for values in product(range(-2, 3), repeat=3)
        if any(values)
    }
    result = tuple(sorted(directions))
    if len(result) != 49:
        raise RuntimeError("The symmetry-closed rank-three direction set changed")
    return result


INTEGER_DIRECTIONS = _integer_projection_directions()
DIRECTION_INDEX = {value: index for index, value in enumerate(INTEGER_DIRECTIONS)}
UNIT_DIRECTIONS = np.asarray(INTEGER_DIRECTIONS, dtype=np.float64)
UNIT_DIRECTIONS /= np.linalg.norm(UNIT_DIRECTIONS, axis=1, keepdims=True)
SIGNED_PERMUTATIONS = tuple(
    (tuple(order), tuple(float(sign) for sign in signs))
    for order in permutations(range(3))
    for signs in product((-1.0, 1.0), repeat=3)
)


def _transformed_direction_map(
    order: Sequence[int], signs: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    indexes: list[int] = []
    orientations: list[int] = []
    for direction in INTEGER_DIRECTIONS:
        coefficients = [0, 0, 0]
        for left_index, right_index in enumerate(order):
            coefficients[int(right_index)] = int(signs[left_index]) * int(
                direction[left_index]
            )
        canonical, orientation = _canonical_direction(coefficients)
        indexes.append(DIRECTION_INDEX[canonical])
        orientations.append(orientation)
    return np.asarray(indexes, dtype=np.int64), np.asarray(orientations, dtype=np.int8)


TRANSFORM_DIRECTION_MAPS = tuple(
    _transformed_direction_map(order, signs)
    for order, signs in SIGNED_PERMUTATIONS
)


def _component_coordinates(scores: np.ndarray, *, copula_only: bool) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 8:
        raise ValueError("Joint geometry requires at least eight rank-three scores")
    if not np.isfinite(values).all():
        raise ValueError("Component scores must be finite")
    if copula_only:
        transformed = np.column_stack(
            [
                norm.ppf((rankdata(values[:, index], method="average") - 0.5) / len(values))
                for index in range(3)
            ]
        )
    else:
        scale = values.std(axis=0)
        if np.any(scale <= 1e-12):
            raise ValueError("Component score variance is degenerate")
        transformed = (values - values.mean(axis=0)) / scale
    return np.asarray(transformed, dtype=np.float64)


def _projection_quantiles(scores: np.ndarray, *, copula_only: bool) -> np.ndarray:
    coordinates = _component_coordinates(scores, copula_only=copula_only)
    projections = coordinates @ UNIT_DIRECTIONS.T
    return np.asarray(
        np.quantile(projections, QUANTILE_GRID, axis=0), dtype=np.float64
    )


def _restricted_signed_permutation_cost_matrix(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    *,
    copula_only: bool = False,
) -> np.ndarray:
    """Return best joint sliced-W2 cost conditional on each component match."""

    left = _projection_quantiles(left_scores, copula_only=copula_only)
    right = _projection_quantiles(right_scores, copula_only=copula_only)
    matrix = np.full((3, 3), np.inf, dtype=np.float64)
    for (order, _), (indexes, orientations) in zip(
        SIGNED_PERMUTATIONS, TRANSFORM_DIRECTION_MAPS, strict=True
    ):
        mapped = right[:, indexes].copy()
        negative = orientations < 0
        if np.any(negative):
            mapped[:, negative] = -right[::-1, indexes[negative]]
        cost = float(np.sqrt(np.mean(np.square(left - mapped))))
        for left_component, right_component in enumerate(order):
            matrix[left_component, int(right_component)] = min(
                matrix[left_component, int(right_component)], cost
            )
    if not np.isfinite(matrix).all():
        raise RuntimeError("A conditional signed-permutation cost was not populated")
    return matrix


def _precomputed_cost_matrix(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    matrix = np.full((3, 3), np.inf, dtype=np.float64)
    for (order, _), (indexes, orientations) in zip(
        SIGNED_PERMUTATIONS, TRANSFORM_DIRECTION_MAPS, strict=True
    ):
        mapped = right[:, indexes].copy()
        negative = orientations < 0
        if np.any(negative):
            mapped[:, negative] = -right[::-1, indexes[negative]]
        cost = float(np.sqrt(np.mean(np.square(left - mapped))))
        for left_component, right_component in enumerate(order):
            matrix[left_component, int(right_component)] = min(
                matrix[left_component, int(right_component)], cost
            )
    return matrix


def _world_features(
    pair_frame: pd.DataFrame,
    artifacts: geometry._WorldArtifacts,
) -> pd.DataFrame:
    context_labels, contexts = geometry._score_contexts(artifacts.mdcr_dir)
    components, supports = geometry._reporter_contexts(
        artifacts.assignment_path, context_labels
    )
    matrices: dict[str, list[dict[tuple[str, str], np.ndarray]]] = {
        metric: [] for metric in METRICS
    }
    solutions: dict[str, list[Any | None]] = {metric: [] for metric in METRICS}
    tables: dict[str, list[pd.DataFrame | None]] = {metric: [] for metric in METRICS}
    for context_index, label in enumerate(context_labels):
        eligible = sorted(
            modality
            for modality, flags in supports.items()
            if flags[context_index] and modality in contexts[label]
        )
        for metric, copula_only in (
            ("joint_sliced_w2", False),
            ("copula_sliced_w2", True),
        ):
            quantiles = {
                modality: _projection_quantiles(scores, copula_only=copula_only)
                for modality, scores in contexts[label].items()
            }
            pair_matrices = {
                (left, right): _precomputed_cost_matrix(
                    quantiles[left], quantiles[right]
                )
                for left, right in combinations(sorted(quantiles), 2)
            }
            matrices[metric].append(pair_matrices)
            if len(eligible) >= 2:
                table = geometry._normalized_cost_table(
                    {modality: contexts[label][modality] for modality in eligible},
                    pair_matrices,
                )
                solution = geometry.synchronization.solve_multiway_permutation_synchronization(
                    table, eligible
                )
            else:
                table = None
                solution = None
            tables[metric].append(table)
            solutions[metric].append(solution)

    enriched = pair_frame.copy()
    for metric in METRICS:
        aggregates: dict[str, list[Any]] = {
            f"{metric}_{suffix}": [] for suffix in FEATURE_SUFFIXES
        }
        for row in pair_frame.itertuples(index=False):
            left = str(row.left_modality)
            right = str(row.right_modality)
            local_mutual: list[bool] = []
            local_margins: list[float] = []
            reporter_costs: list[float] = []
            global_coassigned: list[bool] = []
            global_margins: list[float] = []
            for context_index, _ in enumerate(context_labels):
                left_component = components.get(left, [None] * len(context_labels))[
                    context_index
                ]
                right_component = components.get(right, [None] * len(context_labels))[
                    context_index
                ]
                if left_component is None or right_component is None:
                    local_mutual.append(False)
                    local_margins.append(0.0)
                    reporter_costs.append(float("nan"))
                    global_coassigned.append(False)
                    continue
                matrix = geometry._oriented_matrix(
                    matrices[metric][context_index], left, right
                )
                cost, margin, mutual = geometry._local_diagnostics(
                    matrix, int(left_component), int(right_component)
                )
                reporter_costs.append(cost)
                local_margins.append(margin)
                local_mutual.append(mutual)
                solution = solutions[metric][context_index]
                table = tables[metric][context_index]
                if (
                    solution is None
                    or table is None
                    or left not in solution.modalities
                    or right not in solution.modalities
                ):
                    global_coassigned.append(False)
                    continue
                diagnostics = geometry.synchronization.reporter_pair_diagnostics(
                    solution,
                    table,
                    left,
                    right,
                    int(left_component),
                    int(right_component),
                )
                coassigned = bool(diagnostics["coassigned_in_best_solution"])
                global_coassigned.append(coassigned)
                if coassigned:
                    global_margins.append(
                        float(diagnostics["pair_exclusion_relative_margin"])
                    )
            finite_costs = [value for value in reporter_costs if np.isfinite(value)]
            aggregates[f"{metric}_local_mutual_context_count"].append(
                int(sum(local_mutual))
            )
            aggregates[f"{metric}_heldout_local_mutual_context_count"].append(
                int(sum(local_mutual[1:]))
            )
            aggregates[f"{metric}_mean_local_relative_margin"].append(
                float(np.mean(local_margins)) if local_margins else 0.0
            )
            aggregates[f"{metric}_minimum_local_relative_margin"].append(
                float(min(local_margins)) if local_margins else 0.0
            )
            aggregates[f"{metric}_maximum_reporter_cost_ratio"].append(
                float(max(finite_costs)) if finite_costs else float("nan")
            )
            aggregates[f"{metric}_mean_reporter_cost_ratio"].append(
                float(np.mean(finite_costs)) if finite_costs else float("nan")
            )
            aggregates[f"{metric}_global_coassignment_context_count"].append(
                int(sum(global_coassigned))
            )
            aggregates[f"{metric}_heldout_global_coassignment_context_count"].append(
                int(sum(global_coassigned[1:]))
            )
            aggregates[f"{metric}_mean_coassigned_exclusion_margin"].append(
                float(np.mean(global_margins)) if global_margins else 0.0
            )
        for column, values in aggregates.items():
            enriched[column] = values
    return enriched


def _compute_world(
    key: str,
    frame: pd.DataFrame,
    artifacts: geometry._WorldArtifacts,
) -> tuple[str, pd.DataFrame]:
    return key, _world_features(frame.reset_index(drop=True), artifacts)


def _discover_v455_baseline(
    output_root: Path,
) -> tuple[dict[str, set[frozenset[str]]], dict[str, dict[str, Any]]]:
    baseline: dict[str, set[frozenset[str]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    pattern = "bootstrap_null_calibrated_stability_rescue_v455_source_current_target_*"
    for directory in sorted(output_root.glob(pattern)):
        if not directory.is_dir():
            continue
        manifest = geometry._verified_manifest(
            directory / "calibration_manifest.json", v455.SCHEMA
        )
        block = str(manifest["target_block"])
        if block in manifests:
            raise ValueError(f"Duplicate v4.55 target block: {block}")
        candidate_path = directory / "bootstrap_null_calibrated_candidates.tsv"
        if geometry._sha256(candidate_path) != str(
            manifest.get("output_sha256", {}).get(candidate_path.name)
        ):
            raise ValueError(f"V4.55 candidate hash mismatch: {block}")
        candidates = pd.read_csv(candidate_path, sep="\t", keep_default_na=False)
        for row in candidates.itertuples(index=False):
            baseline.setdefault(str(row.world_key), set()).update(
                geometry._parse_pairs(row.candidate_assay_pairs_json)
            )
        manifests[block] = manifest
    if len(manifests) != 5:
        raise ValueError(f"Expected five v4.55 target blocks, found {len(manifests)}")
    return baseline, manifests


def _evaluate_variants(
    frames: Mapping[str, pd.DataFrame],
    block_keys: Mapping[str, Sequence[str]],
    truth: Mapping[str, set[frozenset[str]]],
    baseline: Mapping[str, set[frozenset[str]]],
    *,
    bootstrap_replicates: int,
    bootstrap_frequency: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    world_frames = geometry.frames_by_world(frames)
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant_index, (variant, (columns, directions)) in enumerate(VARIANTS.items()):
        nominal_tp = nominal_fp = stable_tp = stable_fp = 0
        for target_index, target_block in enumerate(sorted(frames)):
            target = frames[target_block].reset_index(drop=True)
            nominal, margins, null_maxima = geometry._nested_selection(
                target=target,
                target_block=target_block,
                frames=frames,
                block_keys=block_keys,
                truth=truth,
                columns=columns,
                directions=directions,
                world_frames=world_frames,
            )
            baseline_mask = np.asarray(
                [
                    geometry._pair(row.left_modality, row.right_modality)
                    in baseline.get(str(row.world_key), set())
                    for row in target.itertuples(index=False)
                ],
                dtype=bool,
            )
            proposal = nominal & ~baseline_mask
            counts = np.zeros(len(target), dtype=np.int64)
            calibration_blocks = [
                block for block in sorted(frames) if block != target_block
            ]
            for replicate in range(bootstrap_replicates):
                sampled_training: dict[tuple[str, str], list[str]] = {}
                for calibration_index, calibration_block in enumerate(
                    calibration_blocks
                ):
                    training_blocks = [
                        block
                        for block in calibration_blocks
                        if block != calibration_block
                    ]
                    for training_index, training_block in enumerate(training_blocks):
                        keys = list(block_keys[training_block])
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    random_seed,
                                    variant_index,
                                    target_index,
                                    replicate,
                                    calibration_index,
                                    training_index,
                                ]
                            )
                        )
                        sampled_training[(calibration_block, training_block)] = [
                            keys[int(index)]
                            for index in rng.integers(0, len(keys), size=len(keys))
                        ]
                selected, _, _ = geometry._nested_selection(
                    target=target,
                    target_block=target_block,
                    frames=frames,
                    block_keys=block_keys,
                    truth=truth,
                    columns=columns,
                    directions=directions,
                    world_frames=world_frames,
                    sampled_training_keys=sampled_training,
                )
                counts += selected.astype(np.int64)
            frequencies = counts / float(bootstrap_replicates)
            stable = proposal & (frequencies >= bootstrap_frequency)
            labels = geometry._truth_labels(target, truth)
            nominal_tp += int(np.sum(proposal & labels))
            nominal_fp += int(np.sum(proposal & ~labels))
            stable_tp += int(np.sum(stable & labels))
            stable_fp += int(np.sum(stable & ~labels))
            for index, row in enumerate(target.itertuples(index=False)):
                if not proposal[index] and not stable[index]:
                    continue
                pair_rows.append(
                    {
                        "variant": variant,
                        "target_block": target_block,
                        "world_key": str(row.world_key),
                        "left_modality": str(row.left_modality),
                        "right_modality": str(row.right_modality),
                        "truth_pair": bool(labels[index]),
                        "selected_by_v455_baseline": bool(baseline_mask[index]),
                        "nominal_new_proposal": bool(proposal[index]),
                        "minimum_nested_absolute_margin": float(
                            margins[index].min()
                        ),
                        "null_margin_upper_envelope": float(
                            max(null_maxima.values())
                        ),
                        "bootstrap_selection_count": int(counts[index]),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_selection_frequency": float(frequencies[index]),
                        "stable_new_proposal": bool(stable[index]),
                    }
                )
        summary_rows.append(
            {
                "variant": variant,
                "feature_columns_json": json.dumps(list(columns), separators=(",", ":")),
                "feature_directions_json": json.dumps(
                    list(directions), separators=(",", ":")
                ),
                "nominal_new_true_pairs": nominal_tp,
                "nominal_new_false_pairs": nominal_fp,
                "stable_new_true_pairs": stable_tp,
                "stable_new_false_pairs": stable_fp,
                "stable_new_precision": (
                    float(stable_tp / (stable_tp + stable_fp))
                    if stable_tp + stable_fp
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def run_diagnostic(
    *,
    output_root: Path,
    formal_worlds_root: Path,
    out_dir: Path,
    worlds_per_block: int = DEFAULT_WORLDS_PER_BLOCK,
    random_seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_frequency: float = DEFAULT_BOOTSTRAP_FREQUENCY,
    exclude_sample_manifest: Path | None = None,
    max_workers: int = 10,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    if bootstrap_replicates < 1 or not 0.0 < bootstrap_frequency <= 1.0:
        raise ValueError("Bootstrap settings are invalid")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    output_root = output_root.expanduser().absolute().resolve(strict=True)
    formal_worlds_root = formal_worlds_root.expanduser().absolute().resolve(
        strict=True
    )
    raw_frames, _, v453_manifests = geometry._discover_v453(output_root)
    baseline, v455_manifests = _discover_v455_baseline(output_root)
    excluded: set[str] = set()
    excluded_hash: str | None = None
    if exclude_sample_manifest is not None:
        excluded_path = exclude_sample_manifest.expanduser().absolute().resolve(
            strict=True
        )
        excluded_manifest = geometry._read_json(excluded_path)
        excluded_hash = geometry._sha256(excluded_path)
        excluded = {
            str(key)
            for keys in excluded_manifest.get("sampled_world_keys_by_block", {}).values()
            for key in keys
        }
        if not excluded:
            raise ValueError("Exclusion manifest contains no sampled worlds")
    sampled = geometry._sample_worlds(
        raw_frames,
        worlds_per_block=worlds_per_block,
        random_seed=random_seed,
        excluded_world_keys=excluded,
    )
    sampled_keys = sorted(key for keys in sampled.values() for key in keys)
    artifacts = geometry._resolve_artifacts(output_root, sampled_keys)
    source_frames = geometry.frames_by_world(raw_frames)
    completed: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_world, key, source_frames[key], artifacts[key]
            ): key
            for key in sampled_keys
        }
        for future in as_completed(futures):
            key, frame = future.result()
            completed[key] = frame
    if set(completed) != set(sampled_keys):
        raise RuntimeError("Joint geometry did not return every sampled world")
    frames = {
        block: pd.concat([completed[key] for key in sampled[block]], ignore_index=True)
        for block in sorted(sampled)
    }
    block_keys = {block: list(sampled[block]) for block in sorted(sampled)}
    truth = geometry._load_truth(formal_worlds_root, sampled)
    feature_table = pd.concat(
        [completed[key] for key in sampled_keys], ignore_index=True
    )
    feature_table["truth_pair"] = geometry._truth_labels(feature_table, truth)
    feature_table["selected_by_v455_baseline"] = [
        geometry._pair(row.left_modality, row.right_modality)
        in baseline.get(str(row.world_key), set())
        for row in feature_table.itertuples(index=False)
    ]
    summary, proposals = _evaluate_variants(
        frames,
        block_keys,
        truth,
        baseline,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_frequency=bootstrap_frequency,
        random_seed=random_seed + 1,
    )

    out_dir.mkdir(parents=True)
    outputs = {
        "joint_sliced_geometry_pair_features.tsv": feature_table,
        "joint_sliced_geometry_variant_summary.tsv": summary,
        "joint_sliced_geometry_proposals.tsv": proposals,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "retrospective_truth_separated_diagnostic",
        "random_seed": random_seed,
        "worlds_per_block_requested": worlds_per_block,
        "sampled_world_keys_by_block": sampled,
        "sampled_world_count": len(sampled_keys),
        "excluded_prior_sample_world_count": len(excluded),
        "excluded_sample_manifest_sha256": excluded_hash,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_frequency_threshold": bootstrap_frequency,
        "component_rank": 3,
        "signed_permutation_count": len(SIGNED_PERMUTATIONS),
        "projection_direction_count": len(INTEGER_DIRECTIONS),
        "projection_direction_construction": (
            "primitive_integer_projective_directions_with_coordinates_in_minus2_to2"
        ),
        "quantile_grid": {
            "minimum": float(QUANTILE_GRID.min()),
            "maximum": float(QUANTILE_GRID.max()),
            "count": len(QUANTILE_GRID),
        },
        "variants_declared_before_scoring": {
            name: {
                "feature_columns": list(columns),
                "feature_directions": list(directions),
            }
            for name, (columns, directions) in VARIANTS.items()
        },
        "baseline": "target-safe CA-BOSS v4.55 union",
        "information_boundary": {
            "existing_cached_worlds_only": True,
            "new_worlds_or_datasets_created": False,
            "heldout_background_component_scores_only_for_new_features": True,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "transport_plan_constructed": False,
            "simulator_latent_state_used": False,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
            "other_block_truth_used_only_to_identify_nulls": True,
            "positive_labels_used_for_score_or_threshold_fit": False,
            "target_truth_opened_only_after_prediction_for_diagnostic_scoring": True,
        },
        "published_method_inspiration": {
            "mmd_ma": {
                "citation": "Liu et al., WABI 2019",
                "doi": "10.4230/LIPIcs.WABI.2019.10",
            },
            "scot": {
                "citation": "Demetci et al., Journal of Computational Biology 2022",
                "doi": "10.1089/cmb.2021.0446",
            },
            "sliced_wasserstein": {
                "citation": "Yi and Liu, PMLR 2023, Sliced Wasserstein variational inference",
                "url": "https://proceedings.mlr.press/v189/yi23a.html",
                "relationship": "joint_distribution_projection_geometry_without_cell_coupling",
            },
        },
        "v453_manifest_payload_sha256_by_block": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v453_manifests.items()
        },
        "v455_manifest_payload_sha256_by_block": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v455_manifests.items()
        },
        "source_sha256": {
            Path(__file__).name: geometry._sha256(Path(__file__).resolve())
        },
        "input_sha256": {
            key: {
                "component_scores.tsv": geometry._sha256(
                    artifacts[key].mdcr_dir / "component_scores.tsv"
                ),
                "fold_component_scores.tsv": geometry._sha256(
                    artifacts[key].mdcr_dir / "fold_component_scores.tsv"
                ),
                "assay_axis_assignments.tsv": geometry._sha256(
                    artifacts[key].assignment_path
                ),
            }
            for key in sampled_keys
        },
        "output_sha256": {
            name: geometry._sha256(out_dir / name) for name in outputs
        },
        "new_worlds_or_datasets_created": False,
        "retrospective_cached_world_diagnostic": True,
        "promoted_to_checkpoint": False,
    }
    manifest["manifest_payload_sha256"] = geometry._canonical_sha256(manifest)
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worlds-per-block", type=int, default=DEFAULT_WORLDS_PER_BLOCK)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-frequency", type=float, default=DEFAULT_BOOTSTRAP_FREQUENCY)
    parser.add_argument("--exclude-sample-manifest", type=Path)
    parser.add_argument("--max-workers", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = run_diagnostic(
        output_root=args.output_root,
        formal_worlds_root=args.formal_worlds_root,
        out_dir=args.out_dir,
        worlds_per_block=args.worlds_per_block,
        random_seed=args.random_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_frequency=args.bootstrap_frequency,
        exclude_sample_manifest=args.exclude_sample_manifest,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
