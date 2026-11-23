#!/usr/bin/env python3
"""Global unlabeled coarse-mixture registration for disjoint assays.

The estimator uses only held-out background matrices.  Every assay is reduced
to six coarse state centroids, with centroid covariance standardized under
*uniform* mode weights so population shifts do not define the coordinate
frame.  A robust generalized-Procrustes model then estimates one shared
asymmetric centroid template, one orthogonal map per assay, and an unlabeled
Hungarian mode assignment per assay.  Assays are retained only when the map is
stable across disjoint background-feature folds and multi-start global fits,
and when its held-out shape match beats an empirical scrambled-geometry null.

No focal variable, paired row, cross-assay feature identity, scorer, simulator
role, or true latent state enters discovery.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import soft_distributional_alignment as soft_alignment
import unpaired_state_alignment as hard_alignment


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GlobalCoarseMixtureConfig:
    """Frozen observable-only registration and abstention settings."""

    n_components: int = 3
    n_modes: int = 6
    n_feature_folds: int = 2
    random_seed: int = 0
    kmeans_n_init: int = 30
    random_rotation_starts: int = 24
    shape_alignment_iterations: int = 40
    template_iterations: int = 30
    convergence_tolerance: float = 1e-7
    near_optimal_relative_tolerance: float = 0.01
    empirical_null_repetitions: int = 199
    maximum_empirical_shape_null_q: float = 0.10
    maximum_feature_fold_residual: float = 0.75
    maximum_fold_rotation_distance: float = 0.35
    maximum_global_rotation_dispersion: float = 0.30
    maximum_leave_one_out_rotation_distance: float = 0.30
    minimum_centroid_covariance_ratio: float = 0.02
    abstention_uncertainty_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least two")
        if self.n_modes <= self.n_components:
            raise ValueError("n_modes must exceed n_components")
        if self.n_feature_folds < 2:
            raise ValueError("n_feature_folds must be at least two")
        if min(self.kmeans_n_init, self.shape_alignment_iterations) < 1:
            raise ValueError("K-means and shape iteration counts must be positive")
        if min(self.template_iterations, self.empirical_null_repetitions) < 1:
            raise ValueError("Template/null iteration counts must be positive")
        if self.random_rotation_starts < 0 or self.random_seed < 0:
            raise ValueError("Random-start count and seed may not be negative")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")
        probabilities = (
            self.near_optimal_relative_tolerance,
            self.maximum_empirical_shape_null_q,
            self.minimum_centroid_covariance_ratio,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("Relative/null/covariance thresholds must lie in (0, 1)")
        if min(
            self.maximum_feature_fold_residual,
            self.maximum_fold_rotation_distance,
            self.maximum_global_rotation_dispersion,
            self.maximum_leave_one_out_rotation_distance,
            self.abstention_uncertainty_penalty,
        ) <= 0.0:
            raise ValueError("Residual, distance, and uncertainty gates must be positive")


@dataclass
class GlobalCoarseMixtureResult:
    embeddings: dict[str, FloatArray]
    embedding_uncertainty: dict[str, FloatArray]
    assay_eligible: dict[str, bool]
    assay_alignment_confidence: dict[str, float]
    provenance: dict[str, Any]


@dataclass
class _LocalGeometry:
    coordinates: FloatArray
    centroids: FloatArray
    mass: FloatArray
    fold_centroids: tuple[FloatArray, ...]
    uncertainty: FloatArray
    diagnostics: dict[str, Any]


@dataclass
class _ShapeAlignment:
    rotation: FloatArray
    assignment: IntArray
    objective: float
    near_rotation_dispersion: float
    near_solution_count: int


@dataclass
class _GlobalFit:
    rotations: dict[str, FloatArray]
    assignments: dict[str, IntArray]
    template: FloatArray
    objective: float
    assay_objectives: dict[str, float]
    iterations: int
    anchor: str


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, *labels: str) -> int:
    joined = "\0".join((str(base_seed), *labels))
    digest = hashlib.sha256(f"global-coarse-mixture-v4\0{joined}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _project_orthogonal(matrix: FloatArray) -> FloatArray:
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return np.asarray(left @ right_t, dtype=np.float64)


def _orthogonal_procrustes(source: FloatArray, target: FloatArray) -> FloatArray:
    return _project_orthogonal(source.T @ target)


def _rotation_distance(left: FloatArray, right: FloatArray) -> float:
    return float(np.linalg.norm(left - right, ord="fro")) / np.sqrt(
        2.0 * left.shape[0]
    )


def _random_orthogonal(
    dimensions: int, rng: np.random.Generator
) -> FloatArray:
    return _project_orthogonal(rng.normal(size=(dimensions, dimensions)))


def _signed_permutation_starts(dimensions: int) -> list[FloatArray]:
    return soft_alignment._signed_permutation_starts(dimensions)


def _center_and_scale_centroids(values: FloatArray) -> FloatArray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
    scale = float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1)) / centered.shape[1]))
    if scale <= 1e-12:
        raise ValueError("Centroid geometry is degenerate")
    return np.asarray(centered / scale, dtype=np.float64)


def _uniform_centroid_whiten(
    coordinates: FloatArray,
    *,
    n_modes: int,
    seed: int,
    kmeans_n_init: int,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    model = KMeans(n_clusters=n_modes, random_state=seed, n_init=kmeans_n_init)
    labels = np.asarray(model.fit_predict(coordinates), dtype=np.int64)
    return _uniform_centroid_transform(coordinates, labels, n_modes=n_modes)


def _uniform_centroid_transform(
    coordinates: FloatArray,
    labels: IntArray,
    *,
    n_modes: int,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    counts = np.bincount(labels, minlength=n_modes)
    if len(counts) != n_modes or np.any(counts == 0):
        raise ValueError("Every coarse mode must contain at least one cell")
    centroids = np.stack(
        [np.mean(coordinates[labels == mode], axis=0) for mode in range(n_modes)]
    )
    mass = np.bincount(labels, minlength=n_modes).astype(np.float64)
    mass /= mass.sum()
    center = centroids.mean(axis=0)
    centered_centroids = centroids - center
    covariance = centered_centroids.T @ centered_centroids / n_modes
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(np.max(eigenvalues))
    ratio = float(np.min(eigenvalues) / max(largest, 1e-12))
    floor = max(largest * 1e-10, 1e-12)
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    transformed = np.asarray((coordinates - center) @ inverse_root, dtype=np.float64)
    transformed_centroids = np.asarray(
        centered_centroids @ inverse_root, dtype=np.float64
    )
    return transformed, transformed_centroids, mass, {
        "uniform_centroid_covariance_eigenvalues": eigenvalues.tolist(),
        "minimum_to_maximum_centroid_covariance_ratio": ratio,
        "mode_mass": mass.tolist(),
        "minimum_mode_cells": int(np.min(counts)),
    }


def _fit_sample_pca_coordinates(
    matrix: FloatArray,
    depth: FloatArray,
    feature_indices: IntArray,
    *,
    config: GlobalCoarseMixtureConfig,
) -> FloatArray:
    normalized = hard_alignment._normalize(matrix[:, feature_indices], depth, "log_depth")
    raw_scores = PCA(
        n_components=config.n_components,
        whiten=False,
        svd_solver="full",
    ).fit_transform(normalized)
    return soft_alignment._whiten(raw_scores)[0]


def _fit_local_view(
    matrix: FloatArray,
    depth: FloatArray,
    feature_indices: IntArray,
    *,
    seed: int,
    config: GlobalCoarseMixtureConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any], IntArray]:
    sample_whitened = _fit_sample_pca_coordinates(
        matrix, depth, feature_indices, config=config
    )
    model = KMeans(
        n_clusters=config.n_modes,
        random_state=seed,
        n_init=config.kmeans_n_init,
    )
    labels = np.asarray(model.fit_predict(sample_whitened), dtype=np.int64)
    coordinates, centroids, mass, diagnostics = _uniform_centroid_transform(
        sample_whitened,
        n_modes=config.n_modes,
        labels=labels,
    )
    return coordinates, centroids, mass, diagnostics, labels


def _fit_local_geometry(
    matrix: FloatArray,
    depth: FloatArray,
    *,
    modality: str,
    config: GlobalCoarseMixtureConfig,
) -> _LocalGeometry:
    if matrix.ndim != 2 or depth.shape != (len(matrix),):
        raise ValueError(f"Invalid background matrix/depth for {modality}")
    if matrix.shape[1] < config.n_feature_folds * config.n_components:
        raise ValueError(f"Too few background features for {modality}")
    full, centroids, mass, full_diagnostics, _ = _fit_local_view(
        matrix,
        depth,
        np.arange(matrix.shape[1], dtype=np.int64),
        seed=_stable_seed(config.random_seed, modality, "full"),
        config=config,
    )
    rng = np.random.default_rng(_stable_seed(config.random_seed, modality, "folds"))
    feature_folds = np.array_split(
        rng.permutation(matrix.shape[1]), config.n_feature_folds
    )
    raw_fold_scores = [
        _fit_sample_pca_coordinates(
            matrix,
            depth,
            np.asarray(indices, dtype=np.int64),
            config=config,
        )
        for indices in feature_folds
    ]
    fold_partition_labels = [
        np.asarray(
            KMeans(
                n_clusters=config.n_modes,
                random_state=_stable_seed(
                    config.random_seed, modality, f"fold_partition{fold_index}"
                ),
                n_init=config.kmeans_n_init,
            ).fit_predict(scores),
            dtype=np.int64,
        )
        for fold_index, scores in enumerate(raw_fold_scores)
    ]
    aligned_scores: list[FloatArray] = []
    fold_centroids: list[FloatArray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, indices in enumerate(feature_folds):
        label_source_fold = (fold_index + 1) % config.n_feature_folds
        # The validation coordinates and their mode partition come from
        # disjoint background-feature folds.  Repeated rows are used only
        # within this assay, never between assays.
        scores, centers, fold_mass, fold_diagnostics = _uniform_centroid_transform(
            raw_fold_scores[fold_index],
            fold_partition_labels[label_source_fold],
            n_modes=config.n_modes,
        )
        rotation = _orthogonal_procrustes(scores, full)
        aligned = np.asarray(scores @ rotation, dtype=np.float64)
        residual = float(
            np.linalg.norm(aligned - full) / max(np.linalg.norm(full), 1e-12)
        )
        aligned_scores.append(aligned)
        fold_centroids.append(np.asarray(centers @ rotation, dtype=np.float64))
        fold_rows.append(
            {
                "fold": fold_index,
                "mode_partition_source_fold": label_source_fold,
                "n_features": len(indices),
                "within_assay_rowwise_procrustes_residual": residual,
                "rotation_determinant": float(np.linalg.det(rotation)),
                "mode_mass": fold_mass.tolist(),
                **fold_diagnostics,
            }
        )
    stack = np.stack(aligned_scores, axis=0)
    uncertainty = np.sqrt(
        np.mean(np.sum(np.square(stack - full[None, :, :]), axis=2), axis=0)
    )
    return _LocalGeometry(
        coordinates=full,
        centroids=centroids,
        mass=mass,
        fold_centroids=tuple(fold_centroids),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        diagnostics={
            "n_cells": len(matrix),
            "n_background_features": matrix.shape[1],
            **full_diagnostics,
            "feature_fold_diagnostics": fold_rows,
            "maximum_feature_fold_residual": max(
                row["within_assay_rowwise_procrustes_residual"] for row in fold_rows
            ),
        },
    )


def _assignment_for_rotation(
    source: FloatArray, target: FloatArray, rotation: FloatArray
) -> tuple[IntArray, float]:
    transformed = source @ rotation
    cost = np.sum(
        np.square(transformed[:, None, :] - target[None, :, :]), axis=2
    )
    rows, columns = linear_sum_assignment(cost)
    assignment = np.empty(len(target), dtype=np.int64)
    assignment[columns] = rows
    objective = float(
        np.mean(np.sum(np.square(transformed[assignment] - target), axis=1))
        / target.shape[1]
    )
    return assignment, objective


def _refine_shape_alignment(
    source: FloatArray,
    target: FloatArray,
    initial: FloatArray,
    config: GlobalCoarseMixtureConfig,
) -> tuple[FloatArray, IntArray, float]:
    rotation = np.asarray(initial, dtype=np.float64).copy()
    previous_assignment: IntArray | None = None
    previous_objective = float("inf")
    for _ in range(config.shape_alignment_iterations):
        assignment, objective = _assignment_for_rotation(source, target, rotation)
        updated = _orthogonal_procrustes(source[assignment], target)
        delta = _rotation_distance(rotation, updated)
        stable_assignment = (
            previous_assignment is not None
            and np.array_equal(assignment, previous_assignment)
        )
        rotation = updated
        if stable_assignment and delta <= np.sqrt(config.convergence_tolerance):
            break
        if abs(previous_objective - objective) <= config.convergence_tolerance:
            if delta <= np.sqrt(config.convergence_tolerance):
                break
        previous_assignment = assignment
        previous_objective = objective
    assignment, objective = _assignment_for_rotation(source, target, rotation)
    return rotation, assignment, objective


def _align_shape(
    source: FloatArray,
    target: FloatArray,
    *,
    seed: int,
    config: GlobalCoarseMixtureConfig,
) -> _ShapeAlignment:
    if source.shape != target.shape:
        raise ValueError("Source and target centroid shapes must match")
    rng = np.random.default_rng(seed)
    starts = _signed_permutation_starts(source.shape[1])
    starts.extend(
        _random_orthogonal(source.shape[1], rng)
        for _ in range(config.random_rotation_starts)
    )
    solutions = [
        _refine_shape_alignment(source, target, start, config) for start in starts
    ]
    objectives = np.asarray([solution[2] for solution in solutions])
    best_index = int(np.argmin(objectives))
    best_rotation, best_assignment, best_objective = solutions[best_index]
    tolerance = max(
        config.near_optimal_relative_tolerance * max(abs(best_objective), 1e-12),
        1e-10,
    )
    near = np.flatnonzero(objectives <= best_objective + tolerance)
    dispersion = float(
        np.mean(
            [
                _rotation_distance(solutions[int(index)][0], best_rotation)
                for index in near
            ]
        )
    )
    return _ShapeAlignment(
        rotation=best_rotation,
        assignment=best_assignment,
        objective=float(best_objective),
        near_rotation_dispersion=dispersion,
        near_solution_count=len(near),
    )


def _geometric_median(points: FloatArray) -> FloatArray:
    estimate = np.mean(points, axis=0)
    for _ in range(50):
        distances = np.linalg.norm(points - estimate[None, :], axis=1)
        if np.any(distances <= 1e-10):
            return np.asarray(points[int(np.argmin(distances))], dtype=np.float64)
        weights = 1.0 / np.maximum(distances, 1e-10)
        updated = np.average(points, axis=0, weights=weights)
        if float(np.linalg.norm(updated - estimate)) <= 1e-9:
            estimate = updated
            break
        estimate = updated
    return np.asarray(estimate, dtype=np.float64)


def _update_template(
    centroids: Mapping[str, FloatArray],
    rotations: Mapping[str, FloatArray],
    assignments: Mapping[str, IntArray],
) -> FloatArray:
    modalities = sorted(centroids)
    rows = []
    for mode in range(next(iter(centroids.values())).shape[0]):
        points = np.stack(
            [
                centroids[modality][assignments[modality][mode]]
                @ rotations[modality]
                for modality in modalities
            ]
        )
        rows.append(_geometric_median(points))
    return _center_and_scale_centroids(np.stack(rows))


def _fit_one_global_start(
    centroids: Mapping[str, FloatArray],
    *,
    reference_modality: str,
    anchor: str,
    config: GlobalCoarseMixtureConfig,
) -> _GlobalFit:
    modalities = sorted(centroids)
    template = np.asarray(centroids[anchor], dtype=np.float64).copy()
    rotations: dict[str, FloatArray] = {}
    assignments: dict[str, IntArray] = {}
    for iteration in range(config.template_iterations):
        for modality in modalities:
            aligned = _align_shape(
                centroids[modality],
                template,
                seed=_stable_seed(
                    config.random_seed, "global", anchor, modality, str(iteration)
                ),
                config=config,
            )
            rotations[modality] = aligned.rotation
            assignments[modality] = aligned.assignment
        updated = _update_template(centroids, rotations, assignments)
        change_alignment = _align_shape(
            updated,
            template,
            seed=_stable_seed(config.random_seed, "template_change", anchor, str(iteration)),
            config=config,
        )
        change = float(np.sqrt(change_alignment.objective))
        template = updated
        if change <= np.sqrt(config.convergence_tolerance):
            break
    assay_objectives: dict[str, float] = {}
    for modality in modalities:
        aligned = _align_shape(
            centroids[modality],
            template,
            seed=_stable_seed(config.random_seed, "global_final", anchor, modality),
            config=config,
        )
        rotations[modality] = aligned.rotation
        assignments[modality] = aligned.assignment
        assay_objectives[modality] = aligned.objective
    # Gauge-fix to the observed RNA coordinate frame.  This changes no fit.
    gauge = rotations[reference_modality].T
    rotations = {
        modality: np.asarray(rotation @ gauge, dtype=np.float64)
        for modality, rotation in rotations.items()
    }
    rotations[reference_modality] = np.eye(template.shape[1], dtype=np.float64)
    template = np.asarray(template @ gauge, dtype=np.float64)
    return _GlobalFit(
        rotations=rotations,
        assignments=assignments,
        template=template,
        objective=float(np.mean(list(assay_objectives.values()))),
        assay_objectives=assay_objectives,
        iterations=iteration + 1,
        anchor=anchor,
    )


def _fit_global_registration(
    centroids: Mapping[str, FloatArray],
    *,
    reference_modality: str,
    config: GlobalCoarseMixtureConfig,
) -> tuple[_GlobalFit, dict[str, Any]]:
    modalities = sorted(centroids)
    if reference_modality not in modalities or len(modalities) < 3:
        raise ValueError("Global registration needs RNA and at least two other assays")
    expected = (config.n_modes, config.n_components)
    if any(np.asarray(values).shape != expected for values in centroids.values()):
        raise ValueError(f"Every centroid array must have shape {expected}")
    fits = [
        _fit_one_global_start(
            centroids,
            reference_modality=reference_modality,
            anchor=anchor,
            config=config,
        )
        for anchor in modalities
    ]
    objectives = np.asarray([fit.objective for fit in fits])
    best_index = int(np.argmin(objectives))
    best = fits[best_index]
    tolerance = max(
        config.near_optimal_relative_tolerance * max(abs(best.objective), 1e-12),
        1e-10,
    )
    near = [fit for fit in fits if fit.objective <= best.objective + tolerance]
    assay_dispersion = {
        modality: float(
            np.mean(
                [
                    _rotation_distance(fit.rotations[modality], best.rotations[modality])
                    for fit in near
                ]
            )
        )
        for modality in modalities
    }
    ordered = np.sort(objectives)
    return best, {
        "anchor_objectives": {
            fit.anchor: fit.objective for fit in sorted(fits, key=lambda item: item.anchor)
        },
        "selected_anchor": best.anchor,
        "selected_objective": best.objective,
        "second_best_objective_gap": float(ordered[1] - ordered[0]),
        "near_optimal_global_fit_count": len(near),
        "assay_near_optimal_rotation_dispersion": assay_dispersion,
        "iterations": best.iterations,
    }


def _scrambled_shape(values: FloatArray, rng: np.random.Generator) -> FloatArray:
    scrambled = np.column_stack(
        [rng.permutation(values[:, dimension]) for dimension in range(values.shape[1])]
    )
    centered = scrambled - scrambled.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-10)
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    return np.asarray(centered @ inverse_root, dtype=np.float64)


def _leave_one_out_template(
    fit: _GlobalFit,
    centroids: Mapping[str, FloatArray],
    held_out: str,
) -> FloatArray:
    retained = [modality for modality in sorted(centroids) if modality != held_out]
    rows = []
    for mode in range(fit.template.shape[0]):
        points = np.stack(
            [
                centroids[modality][fit.assignments[modality][mode]]
                @ fit.rotations[modality]
                for modality in retained
            ]
        )
        rows.append(_geometric_median(points))
    return _center_and_scale_centroids(np.stack(rows))


def register_global_coarse_mixtures(
    centroids: Mapping[str, FloatArray],
    fold_centroids: Mapping[str, tuple[FloatArray, ...]],
    *,
    reference_modality: str = "rna",
    config: GlobalCoarseMixtureConfig | None = None,
) -> tuple[dict[str, FloatArray], dict[str, bool], dict[str, float], dict[str, Any]]:
    """Register already-local centroid geometries with observable abstention."""

    cfg = config or GlobalCoarseMixtureConfig()
    modalities = sorted(centroids)
    if set(modalities) != set(fold_centroids):
        raise ValueError("Full and fold centroid assay keys must agree")
    if any(len(values) != cfg.n_feature_folds for values in fold_centroids.values()):
        raise ValueError("Every assay must provide the configured feature folds")
    best, global_diagnostics = _fit_global_registration(
        centroids, reference_modality=reference_modality, config=cfg
    )
    fold_fits = []
    for fold_index in range(cfg.n_feature_folds):
        fold_fit, _ = _fit_global_registration(
            {
                modality: fold_centroids[modality][fold_index]
                for modality in modalities
            },
            reference_modality=reference_modality,
            config=cfg,
        )
        fold_fits.append(fold_fit)

    eligible: dict[str, bool] = {}
    confidence: dict[str, float] = {}
    assay_rows: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        # An assay-specific stream makes its evidence invariant to unrelated
        # assay additions or removals.
        rng = np.random.default_rng(
            _stable_seed(cfg.random_seed, "empirical_null", modality)
        )
        template = _leave_one_out_template(best, centroids, modality)
        held_out = _align_shape(
            centroids[modality],
            template,
            seed=_stable_seed(cfg.random_seed, "held_out", modality),
            config=cfg,
        )
        held_out_rotation = np.asarray(
            held_out.rotation @ best.rotations[reference_modality].T,
            dtype=np.float64,
        )
        leave_one_out_distance = _rotation_distance(
            held_out_rotation, best.rotations[modality]
        )
        null_objectives = []
        for repetition in range(cfg.empirical_null_repetitions):
            null_shape = _scrambled_shape(centroids[modality], rng)
            null_fit = _align_shape(
                null_shape,
                template,
                seed=_stable_seed(
                    cfg.random_seed, "null", modality, str(repetition)
                ),
                config=cfg,
            )
            null_objectives.append(null_fit.objective)
        null_q = float(
            (1.0 + np.sum(np.asarray(null_objectives) <= held_out.objective))
            / (cfg.empirical_null_repetitions + 1.0)
        )
        fold_distances = [
            _rotation_distance(best.rotations[modality], fit.rotations[modality])
            for fit in fold_fits
        ]
        global_dispersion = float(
            global_diagnostics["assay_near_optimal_rotation_dispersion"][modality]
        )
        reasons: list[str] = []
        if null_q > cfg.maximum_empirical_shape_null_q:
            reasons.append("shape_not_separated_from_scrambled_null")
        if max(fold_distances) > cfg.maximum_fold_rotation_distance:
            reasons.append("feature_fold_rotation_instability")
        if global_dispersion > cfg.maximum_global_rotation_dispersion:
            reasons.append("near_optimal_global_rotation_ambiguity")
        if leave_one_out_distance > cfg.maximum_leave_one_out_rotation_distance:
            reasons.append("leave_one_assay_out_rotation_instability")
        passed = not reasons
        eligible[modality] = passed
        confidence[modality] = (
            float(
                (1.0 - null_q)
                * np.exp(-max(fold_distances) / cfg.maximum_fold_rotation_distance)
                * np.exp(
                    -leave_one_out_distance
                    / cfg.maximum_leave_one_out_rotation_distance
                )
            )
            if passed
            else 0.0
        )
        assay_rows[modality] = {
            "held_out_shape_objective": held_out.objective,
            "empirical_shape_null_q": null_q,
            "feature_fold_rotation_distances": fold_distances,
            "maximum_feature_fold_rotation_distance": max(fold_distances),
            "near_optimal_global_rotation_dispersion": global_dispersion,
            "leave_one_assay_out_rotation_distance": leave_one_out_distance,
            "eligible": passed,
            "abstention_reasons": reasons,
            "alignment_confidence": confidence[modality],
        }
    return best.rotations, eligible, confidence, {
        "global_fit": global_diagnostics,
        "assay_registration_diagnostics": assay_rows,
        "template": best.template.tolist(),
        "assignments": {
            modality: best.assignments[modality].tolist() for modality in modalities
        },
    }


def align_global_coarse_mixture_background_states(
    background_matrices: Mapping[str, FloatArray],
    background_depths: Mapping[str, FloatArray],
    *,
    reference_modality: str = "rna",
    config: GlobalCoarseMixtureConfig | None = None,
) -> GlobalCoarseMixtureResult:
    """Fit local state geometry, global mixture registration, and abstention."""

    cfg = config or GlobalCoarseMixtureConfig()
    modalities = sorted(str(key) for key in background_matrices)
    if len(modalities) < 3 or set(modalities) != set(background_depths):
        raise ValueError("At least three assays with matching matrix/depth keys are required")
    if reference_modality not in modalities:
        raise ValueError("reference_modality is absent")
    local = {
        modality: _fit_local_geometry(
            np.asarray(background_matrices[modality], dtype=np.float64),
            np.asarray(background_depths[modality], dtype=np.float64),
            modality=modality,
            config=cfg,
        )
        for modality in modalities
    }
    rotations, registration_eligible, confidence, registration = (
        register_global_coarse_mixtures(
            {modality: local[modality].centroids for modality in modalities},
            {modality: local[modality].fold_centroids for modality in modalities},
            reference_modality=reference_modality,
            config=cfg,
        )
    )
    eligible: dict[str, bool] = {}
    embeddings: dict[str, FloatArray] = {}
    uncertainty: dict[str, FloatArray] = {}
    local_diagnostics: dict[str, Any] = {}
    for modality in modalities:
        local_pass = bool(
            local[modality].diagnostics[
                "minimum_to_maximum_centroid_covariance_ratio"
            ]
            >= cfg.minimum_centroid_covariance_ratio
            and local[modality].diagnostics["maximum_feature_fold_residual"]
            <= cfg.maximum_feature_fold_residual
        )
        passed = bool(local_pass and registration_eligible[modality])
        eligible[modality] = passed
        if not passed:
            confidence[modality] = 0.0
        embeddings[modality] = np.asarray(
            local[modality].coordinates @ rotations[modality], dtype=np.float64
        )
        uncertainty[modality] = np.sqrt(
            np.square(local[modality].uncertainty)
            + (0.0 if passed else cfg.abstention_uncertainty_penalty**2)
        )
        local_reasons = []
        if (
            local[modality].diagnostics[
                "minimum_to_maximum_centroid_covariance_ratio"
            ]
            < cfg.minimum_centroid_covariance_ratio
        ):
            local_reasons.append("rank_deficient_coarse_mode_geometry")
        if (
            local[modality].diagnostics["maximum_feature_fold_residual"]
            > cfg.maximum_feature_fold_residual
        ):
            local_reasons.append("unstable_within_assay_feature_fold_geometry")
        local_diagnostics[modality] = {
            **local[modality].diagnostics,
            "local_geometry_eligible": local_pass,
            "local_abstention_reasons": local_reasons,
        }

    provenance = {
        "schema_version": "global_coarse_mixture_alignment_v4.v1",
        "method": (
            "assay_local_pca_uniform_mode_covariance_whitening_global_robust_"
            "unlabeled_generalized_procrustes_empirical_null_abstention"
        ),
        "reference_modality": reference_modality,
        "config": asdict(cfg),
        "accepted_assays": sorted(modality for modality in modalities if eligible[modality]),
        "abstained_assays": sorted(modality for modality in modalities if not eligible[modality]),
        "assay_eligible": eligible,
        "assay_alignment_confidence": confidence,
        "local_geometry_diagnostics": local_diagnostics,
        **registration,
        "identifiability": {
            "coarse_mode_count": cfg.n_modes,
            "coarse_modes_labeled": False,
            "population_mass_used_to_define_geometry": False,
            "assay_specific_population_mass_retained_as_diagnostic": True,
            "shared_template_is_robust_geometric_median": True,
            "cross_assay_maps_are_orthogonal": True,
            "reference_gauge_fixed_to_rna": True,
        },
        "information_boundary": {
            "consumed": [
                "observed_background_matrix_per_assay",
                "observed_leave_focal_out_depth_per_assay",
                "assay_key",
                "declared_background_state_dimension",
                "declared_coarse_quantization_resolution",
            ],
            "not_consumed": [
                "focal_features",
                "true_latent_state",
                "simulator_roles",
                "truth_graph",
                "scorer",
                "paired_cells",
                "cross_assay_feature_identity",
                "external_data",
            ],
            "same_assay_rows_used_for_feature_fold_reconciliation": True,
            "cross_assay_rows_paired": False,
            "cross_assay_feature_columns_compared": False,
            "coarse_mode_assignment_returned_as_cell_pairing": False,
            "truth_or_scorer_used": False,
        },
    }
    return GlobalCoarseMixtureResult(
        embeddings=embeddings,
        embedding_uncertainty=uncertainty,
        assay_eligible=eligible,
        assay_alignment_confidence=confidence,
        provenance=provenance,
    )


def freeze_global_coarse_mixture_evidence(
    discovery_dir: Path | str,
    out_dir: Path | str,
    *,
    config: GlobalCoarseMixtureConfig | None = None,
) -> dict[str, Any]:
    """Write content-addressed background-only evidence before scorer access."""

    cfg = config or GlobalCoarseMixtureConfig()
    output = Path(out_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    root, matrices, depths, cell_ids, input_hashes = soft_alignment._load_background_archives(
        discovery_dir
    )
    if output.resolve(strict=False) == root or root in output.resolve(strict=False).parents:
        raise ValueError("Output must be outside the discovery input")
    output.mkdir(parents=True, exist_ok=True)
    result = align_global_coarse_mixture_background_states(
        matrices, depths, reference_modality="rna", config=cfg
    )
    rows = []
    for modality in sorted(result.embeddings):
        coordinates = result.embeddings[modality]
        rows.append(
            pd.DataFrame(
                {
                    "modality": modality,
                    "cell": cell_ids[modality],
                    **{
                        f"integration__dim{dimension}": coordinates[:, dimension]
                        for dimension in range(coordinates.shape[1])
                    },
                    "integration_uncertainty": result.embedding_uncertainty[modality],
                }
            )
        )
    embedding_path = output / "analysis_embeddings.tsv"
    pd.concat(rows, ignore_index=True).to_csv(embedding_path, sep="\t", index=False)
    provenance_path = output / "alignment_provenance.json"
    provenance_path.write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    source = Path(__file__).resolve(strict=True)
    manifest = {
        "schema_version": "global_coarse_mixture_alignment_v4.frozen_evidence.v1",
        "algorithm_source_sha256": _sha256(source),
        "config": asdict(cfg),
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            embedding_path.name: _sha256(embedding_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "frozen_before_scorer_read": True,
        "information_boundary": result.provenance["information_boundary"],
    }
    (output / "frozen_evidence_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--empirical-null-repetitions", type=int, default=199)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = freeze_global_coarse_mixture_evidence(
        args.discovery,
        args.out,
        config=GlobalCoarseMixtureConfig(
            random_seed=args.random_seed,
            empirical_null_repetitions=args.empirical_null_repetitions,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GlobalCoarseMixtureConfig",
    "GlobalCoarseMixtureResult",
    "align_global_coarse_mixture_background_states",
    "freeze_global_coarse_mixture_evidence",
    "register_global_coarse_mixtures",
]
