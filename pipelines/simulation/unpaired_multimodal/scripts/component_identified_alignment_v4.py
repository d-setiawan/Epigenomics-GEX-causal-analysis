#!/usr/bin/env python3
"""Observed-only component-identified alignment for unpaired assays.

The existing distributional alignment deliberately whitens independently fitted
PCA scores and therefore leaves a full orthogonal ambiguity.  This standalone
prototype asks whether the ambiguity can be reduced to signed permutations
*before* any cross-assay comparison:

* ``pca`` retains unwhitened PCA scores and requires separated observed
  covariance eigenvalues;
* ``fastica`` identifies non-Gaussian components after PCA covariance ordering
  proves unstable;
* disjoint background-feature folds are reconciled with repeated rows only
  within the same assay;
* every assay embedding is quantized independently, and entropic transport is
  evaluated over exactly the 48 three-dimensional signed permutations;
* high-margin pair transforms are cycle-filtered and globally synchronized in
  that discrete group;
* weak fold assignments, covariance gaps, global margins, or assay-specific
  margins cause explicit assay abstention.

There is no cross-assay row or feature correspondence.  Focal measurements,
simulator labels, and scorer files are not accepted by the API.  Componentwise
standardization occurs only after the signed-permutation coordinate system is
frozen; no cross-assay rotation is fitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_BACKGROUND_PREFIX = "background_matrix__"
_FORBIDDEN_PATH_COMPONENTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
    "counterfactual",
}


@dataclass(frozen=True)
class ComponentAlignmentConfig:
    """Frozen settings for signed-permutation component identification."""

    method: str = "fastica"
    n_components: int = 3
    n_feature_folds: int = 2
    n_transport_microstates: int = 6
    random_seed: int = 0
    min_fold_component_correlation: float = 0.40
    min_fold_assignment_margin: float = 0.015
    min_pca_relative_variance_gap: float = 0.12
    kmeans_n_init: int = 20
    population_uniform_shrinkage: float = 0.75
    uncertainty_floor: float = 0.10
    entropy_fraction: float = 0.08
    sinkhorn_iterations: int = 250
    sinkhorn_tolerance: float = 1e-7
    min_pair_transport_margin: float = 0.01
    min_pair_transport_separation: float = 0.10
    min_synchronization_neighbors: int = 2
    min_cycle_support: int = 1
    min_assay_synchronization_margin: float = 0.005
    synchronization_iterations: int = 20
    fastica_max_iter: int = 3_000
    fastica_tolerance: float = 1e-5
    abstention_uncertainty_penalty: float = 2.0
    max_dense_entries: int = 50_000_000

    def __post_init__(self) -> None:
        if self.method not in {"pca", "fastica"}:
            raise ValueError("method must be 'pca' or 'fastica'")
        # Exhaustive global synchronization is intentionally fixed to the
        # simulator's declared three-dimensional integration state.
        if self.n_components != 3:
            raise ValueError("component synchronization is fixed to three components")
        if self.n_feature_folds < 2:
            raise ValueError("n_feature_folds must be at least two")
        if self.n_transport_microstates <= self.n_components:
            raise ValueError("n_transport_microstates must exceed n_components")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        bounded = (
            self.min_fold_component_correlation,
            self.min_fold_assignment_margin,
            self.min_pca_relative_variance_gap,
            self.population_uniform_shrinkage,
            self.uncertainty_floor,
            self.min_pair_transport_margin,
            self.min_pair_transport_separation,
            self.min_assay_synchronization_margin,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("correlation, gap, and margin thresholds must lie in [0, 1]")
        if self.kmeans_n_init < 1 or self.sinkhorn_iterations < 1:
            raise ValueError("quantization and Sinkhorn iteration counts must be positive")
        if self.entropy_fraction <= 0.0 or self.sinkhorn_tolerance <= 0.0:
            raise ValueError("transport regularization and tolerance must be positive")
        if self.min_synchronization_neighbors < 1 or self.min_cycle_support < 1:
            raise ValueError("synchronization support settings must be positive")
        if self.synchronization_iterations < 1:
            raise ValueError("synchronization_iterations must be positive")
        if self.fastica_max_iter < 100 or self.fastica_tolerance <= 0.0:
            raise ValueError("FastICA iteration settings are invalid")
        if self.abstention_uncertainty_penalty <= 0.0:
            raise ValueError("abstention_uncertainty_penalty must be positive")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


@dataclass
class ComponentAlignmentResult:
    """Component coordinates, uncertainty, abstention, and audit evidence."""

    embeddings: dict[str, FloatArray]
    embedding_uncertainty: dict[str, FloatArray]
    assay_eligible: dict[str, bool]
    assay_alignment_confidence: dict[str, float]
    component_evidence: pd.DataFrame
    provenance: dict[str, Any]


@dataclass
class _ComponentFit:
    scores: FloatArray
    strength: FloatArray
    converged: bool
    n_iterations: int


@dataclass
class _AssayFit:
    scores: FloatArray
    uncertainty: FloatArray
    strength: FloatArray
    fold_stable: bool
    minimum_fold_correlation: float
    minimum_fold_margin: float
    minimum_variance_gap: float
    diagnostics: dict[str, Any]


@dataclass
class _QuantizedDistribution:
    centroids: FloatArray
    mass: FloatArray
    effective_cells: float


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: FloatArray) -> str:
    values = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(values.dtype.str.encode("utf-8"))
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _stable_seed(base: int, *tokens: object) -> int:
    payload = "::".join([str(base), *(str(token) for token in tokens)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4],
        byteorder="little",
        signed=False,
    )


def _as_matrix(matrix: ArrayLike, *, max_entries: int) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 8 or values.shape[1] < 6:
        raise ValueError("Every assay needs at least eight rows and six background features")
    if int(np.prod(values.shape)) > max_entries:
        raise ValueError("Observed background matrix exceeds max_dense_entries")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Observed background matrices must be finite and non-negative")
    return values


def _as_depth(depth: ArrayLike, *, matrix: FloatArray) -> FloatArray:
    values = np.asarray(depth, dtype=np.float64).reshape(-1)
    if values.shape != (len(matrix),):
        raise ValueError("Each background depth vector must match its matrix rows")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Background depth must be finite and strictly positive")
    if not np.allclose(matrix.sum(axis=1), values, rtol=1e-8, atol=1e-8):
        raise ValueError("Background depth must equal the observed background row sum")
    return values


def _normalize(matrix: FloatArray, depth: FloatArray) -> FloatArray:
    positive = depth[depth > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    normalized = np.log1p(matrix * (scale / depth)[:, None])
    return np.asarray(StandardScaler().fit_transform(normalized), dtype=np.float64)


def _standardize_components(scores: FloatArray) -> FloatArray:
    centered = np.asarray(scores, dtype=np.float64) - np.mean(scores, axis=0)
    scale = np.std(centered, axis=0, ddof=0)
    if np.any(scale <= 1e-12):
        raise ValueError("A fitted component has zero observed variance")
    return np.asarray(centered / scale, dtype=np.float64)


def _fit_components(
    normalized: FloatArray,
    *,
    config: ComponentAlignmentConfig,
    seed: int,
) -> _ComponentFit:
    if config.method == "pca":
        model = PCA(n_components=config.n_components, whiten=False, svd_solver="full")
        raw_scores = np.asarray(model.fit_transform(normalized), dtype=np.float64)
        variance = np.asarray(model.explained_variance_, dtype=np.float64)
        strength = variance / max(float(variance.sum()), 1e-12)
        return _ComponentFit(
            scores=_standardize_components(raw_scores),
            strength=strength,
            converged=True,
            n_iterations=1,
        )

    model = FastICA(
        n_components=config.n_components,
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=config.fastica_max_iter,
        tol=config.fastica_tolerance,
        random_state=seed,
        whiten_solver="svd",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        raw_scores = np.asarray(model.fit_transform(normalized), dtype=np.float64)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    mixing = np.asarray(model.mixing_, dtype=np.float64)
    strength = np.sum(np.square(mixing), axis=0)
    strength /= max(float(strength.sum()), 1e-12)
    n_iterations = int(getattr(model, "n_iter_", config.fastica_max_iter))
    return _ComponentFit(
        scores=_standardize_components(raw_scores),
        strength=np.asarray(strength, dtype=np.float64),
        converged=not convergence_warning and n_iterations < config.fastica_max_iter,
        n_iterations=n_iterations,
    )


def _assignment_margin(absolute_correlation: FloatArray) -> tuple[IntArray, float, float]:
    dimensions = absolute_correlation.shape[0]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for order in permutations(range(dimensions)):
        score = float(
            np.mean([absolute_correlation[index, order[index]] for index in range(dimensions)])
        )
        candidates.append((score, tuple(int(value) for value in order)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best = candidates[0]
    second_score = candidates[1][0]
    margin = (best_score - second_score) / max(best_score, 1e-12)
    return np.asarray(best, dtype=np.int64), float(best_score), float(margin)


def _align_fold_to_full(
    fold: _ComponentFit,
    full: _ComponentFit,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    dimensions = full.scores.shape[1]
    correlations = np.corrcoef(full.scores.T, fold.scores.T)[:dimensions, dimensions:]
    permutation, _, margin = _assignment_margin(np.abs(correlations))
    aligned = np.empty_like(full.scores)
    matched = np.empty(dimensions, dtype=np.float64)
    aligned_strength = np.empty(dimensions, dtype=np.float64)
    for component in range(dimensions):
        local = int(permutation[component])
        correlation = float(correlations[component, local])
        aligned[:, component] = fold.scores[:, local] * (1.0 if correlation >= 0.0 else -1.0)
        matched[component] = abs(correlation)
        aligned_strength[component] = fold.strength[local]
    return aligned, matched, aligned_strength, margin


def _relative_variance_gap(strength: FloatArray) -> float:
    ordered = np.asarray(strength, dtype=np.float64)
    return float(
        np.min(
            (ordered[:-1] - ordered[1:])
            / np.maximum(ordered[:-1], 1e-12)
        )
    )


def _fit_assay(
    matrix: FloatArray,
    depth: FloatArray,
    *,
    modality: str,
    config: ComponentAlignmentConfig,
) -> _AssayFit:
    normalized = _normalize(matrix, depth)
    full = _fit_components(
        normalized,
        config=config,
        seed=_stable_seed(config.random_seed, modality, "full", config.method),
    )
    rng = np.random.default_rng(
        _stable_seed(config.random_seed, modality, "feature_folds")
    )
    feature_folds = np.array_split(
        rng.permutation(matrix.shape[1]), config.n_feature_folds
    )
    aligned_folds: list[FloatArray] = []
    correlations: list[float] = []
    margins: list[float] = []
    fold_converged: list[bool] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, feature_indices in enumerate(feature_folds):
        fit = _fit_components(
            normalized[:, np.asarray(feature_indices, dtype=np.int64)],
            config=config,
            seed=_stable_seed(
                config.random_seed, modality, "fold", fold_index, config.method
            ),
        )
        aligned, matched, aligned_strength, margin = _align_fold_to_full(fit, full)
        aligned_folds.append(aligned)
        correlations.extend(float(value) for value in matched)
        margins.append(float(margin))
        fold_converged.append(bool(fit.converged))
        fold_rows.append(
            {
                "fold": fold_index,
                "n_features": int(len(feature_indices)),
                "matched_absolute_correlations": matched.tolist(),
                "assignment_relative_margin": float(margin),
                "aligned_component_strength": aligned_strength.tolist(),
                "converged": bool(fit.converged),
                "n_iterations": int(fit.n_iterations),
            }
        )
    stack = np.stack(aligned_folds, axis=0)
    uncertainty = np.sqrt(
        np.mean(np.sum(np.square(stack - full.scores[None, :, :]), axis=2), axis=0)
    )
    minimum_correlation = float(np.min(correlations))
    minimum_margin = float(np.min(margins))
    variance_gap = (
        _relative_variance_gap(full.strength)
        if config.method == "pca"
        else float("nan")
    )
    stable = bool(
        full.converged
        and all(fold_converged)
        and minimum_correlation >= config.min_fold_component_correlation
        and minimum_margin >= config.min_fold_assignment_margin
        and (
            config.method != "pca"
            or variance_gap >= config.min_pca_relative_variance_gap
        )
    )
    return _AssayFit(
        scores=full.scores,
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        strength=full.strength,
        fold_stable=stable,
        minimum_fold_correlation=minimum_correlation,
        minimum_fold_margin=minimum_margin,
        minimum_variance_gap=variance_gap,
        diagnostics={
            "n_cells": int(len(matrix)),
            "n_background_features": int(matrix.shape[1]),
            "matrix_sha256": _array_sha256(matrix),
            "depth_sha256": _array_sha256(depth),
            "full_component_strength": full.strength.tolist(),
            "full_converged": bool(full.converged),
            "full_n_iterations": int(full.n_iterations),
            "minimum_fold_component_correlation": minimum_correlation,
            "minimum_fold_assignment_margin": minimum_margin,
            "minimum_relative_variance_gap": (
                variance_gap if np.isfinite(variance_gap) else None
            ),
            "feature_fold_diagnostics": fold_rows,
            "within_assay_component_stability_passed": stable,
        },
    )


def _cell_reliability(uncertainty: FloatArray, floor: float) -> FloatArray:
    positive = uncertainty[uncertainty > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    values = 1.0 / (1.0 + np.square(uncertainty / max(scale, 1e-12)))
    return np.asarray(np.clip(values, floor, 1.0), dtype=np.float64)


def _quantize_distribution(
    fit: _AssayFit,
    *,
    modality: str,
    config: ComponentAlignmentConfig,
) -> _QuantizedDistribution:
    reliability = _cell_reliability(fit.uncertainty, config.uncertainty_floor)
    model = KMeans(
        n_clusters=config.n_transport_microstates,
        random_state=_stable_seed(config.random_seed, modality, "transport_kmeans"),
        n_init=config.kmeans_n_init,
    )
    model.fit(fit.scores, sample_weight=reliability)
    labels = np.asarray(model.labels_, dtype=np.int64)
    centroids: list[FloatArray] = []
    raw_mass: list[float] = []
    for cluster in range(config.n_transport_microstates):
        selected = labels == cluster
        if not np.any(selected):
            raise RuntimeError("Component quantization produced an empty microstate")
        weights = reliability[selected]
        centroids.append(
            np.asarray(
                np.average(fit.scores[selected], axis=0, weights=weights),
                dtype=np.float64,
            )
        )
        raw_mass.append(float(weights.sum()))
    mass = np.asarray(raw_mass, dtype=np.float64)
    mass /= mass.sum()
    uniform = np.full(len(mass), 1.0 / len(mass), dtype=np.float64)
    shrinkage = config.population_uniform_shrinkage
    mass = (1.0 - shrinkage) * mass + shrinkage * uniform
    mass /= mass.sum()
    effective_cells = float(
        np.square(reliability.sum())
        / max(float(np.square(reliability).sum()), 1e-12)
    )
    return _QuantizedDistribution(
        centroids=np.vstack(centroids),
        mass=mass,
        effective_cells=effective_cells,
    )


def _signed_permutation_matrices(dimensions: int) -> tuple[FloatArray, ...]:
    identity = np.eye(dimensions, dtype=np.float64)
    matrices = tuple(
        np.asarray(
            identity[:, list(order)] * np.asarray(signs, dtype=np.float64)[None, :],
            dtype=np.float64,
        )
        for order in permutations(range(dimensions))
        for signs in product((-1.0, 1.0), repeat=dimensions)
    )
    if len(matrices) != 48 or len({matrix.tobytes() for matrix in matrices}) != 48:
        raise RuntimeError("Three-dimensional signed-permutation group is malformed")
    return matrices


def _squared_distances(left: FloatArray, right: FloatArray) -> FloatArray:
    left_norm = np.sum(np.square(left), axis=1)[:, None]
    right_norm = np.sum(np.square(right), axis=1)[None, :]
    return np.asarray(
        np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0),
        dtype=np.float64,
    )


def _sinkhorn_coupling(
    cost: FloatArray,
    left_mass: FloatArray,
    right_mass: FloatArray,
    *,
    regularization: float,
    iterations: int,
    tolerance: float,
) -> FloatArray:
    log_kernel = -cost / max(regularization, 1e-12)
    log_left = np.log(np.maximum(left_mass, 1e-300))
    log_right = np.log(np.maximum(right_mass, 1e-300))
    log_u = np.zeros_like(log_left)
    log_v = np.zeros_like(log_right)
    for iteration in range(iterations):
        previous = log_u.copy() if iteration % 5 == 0 else None
        log_u = log_left - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_right - logsumexp(log_kernel.T + log_u[None, :], axis=1)
        if previous is not None and float(np.max(np.abs(log_u - previous))) < tolerance:
            break
    plan = np.exp(
        np.clip(log_u[:, None] + log_kernel + log_v[None, :], -745.0, 700.0)
    )
    plan *= (left_mass / np.maximum(plan.sum(axis=1), 1e-300))[:, None]
    plan *= (right_mass / np.maximum(plan.sum(axis=0), 1e-300))[None, :]
    return np.asarray(plan, dtype=np.float64)


def _transport_regularization(
    target: _QuantizedDistribution,
    reference: _QuantizedDistribution,
    config: ComponentAlignmentConfig,
) -> float:
    distances: list[float] = []
    for centroids in (target.centroids, reference.centroids):
        matrix = _squared_distances(centroids, centroids)
        distances.extend(float(value) for value in matrix[np.triu_indices(len(matrix), 1)])
    positive = np.asarray([value for value in distances if value > 1e-12])
    scale = float(np.median(positive)) if positive.size else 1.0
    return max(config.entropy_fraction * scale, 1e-8)


def _transport_objective(
    target: _QuantizedDistribution,
    reference: _QuantizedDistribution,
    transform: FloatArray,
    *,
    regularization: float,
    config: ComponentAlignmentConfig,
) -> tuple[float, float]:
    cost = _squared_distances(target.centroids @ transform, reference.centroids)
    coupling = _sinkhorn_coupling(
        cost,
        target.mass,
        reference.mass,
        regularization=regularization,
        iterations=config.sinkhorn_iterations,
        tolerance=config.sinkhorn_tolerance,
    )
    objective = float(np.sum(coupling * cost))
    independent = float(np.sum(np.outer(target.mass, reference.mass) * cost))
    return objective, independent


def _fit_pair_transform(
    target: _QuantizedDistribution,
    reference: _QuantizedDistribution,
    *,
    transforms: Sequence[FloatArray],
    config: ComponentAlignmentConfig,
) -> tuple[FloatArray, dict[str, Any]]:
    regularization = _transport_regularization(target, reference, config)
    rows: list[tuple[float, int, float]] = []
    for index, transform in enumerate(transforms):
        objective, independent = _transport_objective(
            target,
            reference,
            transform,
            regularization=regularization,
            config=config,
        )
        rows.append((objective, index, independent))
    rows.sort(key=lambda item: (item[0], item[1]))
    best, best_index, independent = rows[0]
    second = rows[1][0]
    margin = float((second - best) / max(second, 1e-12))
    separation = float(
        np.clip((independent - best) / max(independent, 1e-12), 0.0, 1.0)
    )
    near_tolerance = max(0.01 * max(abs(best), 1e-12), 1e-8)
    return np.asarray(transforms[best_index], dtype=np.float64), {
        "transport_objective": float(best),
        "second_best_transport_objective": float(second),
        "relative_signed_permutation_margin": margin,
        "independent_coupling_objective": float(independent),
        "transport_separation": separation,
        "near_optimal_signed_permutation_count": int(
            sum(objective <= best + near_tolerance for objective, _, _ in rows)
        ),
        "signed_permutation_candidates_evaluated": len(rows),
        "sinkhorn_regularization": regularization,
        "target_effective_cells": target.effective_cells,
        "reference_effective_cells": reference.effective_cells,
    }


def _rotation_distance(left: FloatArray, right: FloatArray) -> float:
    return float(np.linalg.norm(left - right, ord="fro")) / np.sqrt(
        2.0 * left.shape[0]
    )


def _edge(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _cycle_filter(
    modalities: Sequence[str],
    reference: str,
    rotations: Mapping[tuple[str, str], FloatArray],
    preliminary: set[tuple[str, str]],
    config: ComponentAlignmentConfig,
) -> tuple[set[tuple[str, str]], dict[str, dict[str, Any]]]:
    accepted: set[tuple[str, str]] = set()
    evidence: dict[str, dict[str, Any]] = {}
    for left, right in sorted(preliminary):
        errors: list[float] = []
        for bridge in modalities:
            if bridge in {left, right}:
                continue
            if _edge(left, bridge) not in preliminary or _edge(bridge, right) not in preliminary:
                continue
            indirect = rotations[(left, bridge)] @ rotations[(bridge, right)]
            errors.append(_rotation_distance(rotations[(left, right)], indirect))
        consistent = int(np.sum(np.asarray(errors) <= 1e-10))
        required_available = 2 if reference in {left, right} else 1
        gate_applied = len(errors) >= required_available
        passed = not gate_applied or consistent >= config.min_cycle_support
        if passed:
            accepted.add((left, right))
        evidence[f"{left}||{right}"] = {
            "cycle_support": len(errors),
            "exact_cycle_support": consistent,
            "minimum_cycle_inconsistency": float(min(errors)) if errors else None,
            "median_cycle_inconsistency": float(np.median(errors)) if errors else None,
            "cycle_gate_applied": gate_applied,
            "cycle_gate_accepted": passed,
        }
    return accepted, evidence


def _reference_core(
    modalities: Sequence[str],
    reference: str,
    edges: set[tuple[str, str]],
    config: ComponentAlignmentConfig,
) -> tuple[set[str], set[tuple[str, str]]]:
    active = set(modalities)
    required = min(config.min_synchronization_neighbors, max(1, len(active) - 2))
    if len(active) == 2:
        required = 1
    changed = True
    while changed:
        changed = False
        remove = {
            modality
            for modality in active
            if modality != reference
            and sum(
                1
                for left, right in edges
                if modality in {left, right} and left in active and right in active
            )
            < required
        }
        if remove:
            active.difference_update(remove)
            changed = True
    adjacency = {modality: set() for modality in active}
    active_edges: set[tuple[str, str]] = set()
    for left, right in edges:
        if left in active and right in active:
            adjacency[left].add(right)
            adjacency[right].add(left)
            active_edges.add((left, right))
    connected = {reference}
    frontier = [reference]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in connected:
                connected.add(neighbor)
                frontier.append(neighbor)
    return connected, {
        edge for edge in active_edges if edge[0] in connected and edge[1] in connected
    }


def _synchronize_discrete_transforms(
    reference: str,
    rotations: Mapping[tuple[str, str], FloatArray],
    confidences: Mapping[tuple[str, str], float],
    assays: set[str],
    edges: set[tuple[str, str]],
    transforms: Sequence[FloatArray],
    config: ComponentAlignmentConfig,
) -> tuple[dict[str, FloatArray], dict[str, float], dict[str, Any]]:
    synchronized = {reference: np.eye(config.n_components, dtype=np.float64)}
    while set(synchronized) != assays:
        choices: list[tuple[float, str, str]] = []
        for left, right in edges:
            if left in synchronized and right not in synchronized:
                choices.append((confidences[(left, right)], right, left))
            elif right in synchronized and left not in synchronized:
                choices.append((confidences[(left, right)], left, right))
        if not choices:
            raise RuntimeError("Accepted signed-permutation graph is disconnected")
        _, target, anchor = max(choices, key=lambda item: (item[0], item[1], item[2]))
        synchronized[target] = rotations[(target, anchor)] @ synchronized[anchor]

    margins = {reference: 1.0}
    used_iterations = 0
    for iteration in range(config.synchronization_iterations):
        changed = False
        updated = {reference: synchronized[reference]}
        for modality in sorted(assays - {reference}):
            candidates: list[tuple[float, int]] = []
            for index, candidate in enumerate(transforms):
                objective = 0.0
                total_weight = 0.0
                for neighbor in assays - {modality}:
                    edge = _edge(modality, neighbor)
                    if edge not in edges:
                        continue
                    weight = confidences[edge]
                    target = rotations[(modality, neighbor)] @ synchronized[neighbor]
                    objective += weight * np.square(_rotation_distance(candidate, target))
                    total_weight += weight
                candidates.append((objective / max(total_weight, 1e-12), index))
            candidates.sort(key=lambda item: (item[0], item[1]))
            best, best_index = candidates[0]
            second = candidates[1][0]
            margins[modality] = float((second - best) / max(second, 1e-12))
            updated[modality] = np.asarray(transforms[best_index], dtype=np.float64)
            changed = changed or not np.array_equal(updated[modality], synchronized[modality])
        synchronized.update(updated)
        used_iterations = iteration + 1
        if not changed:
            break
    residuals = {
        f"{left}||{right}": _rotation_distance(
            synchronized[left], rotations[(left, right)] @ synchronized[right]
        )
        for left, right in sorted(edges)
    }
    return synchronized, margins, {
        "iterations": used_iterations,
        "accepted_edge_residuals": residuals,
        "exact_consensus_edges": int(sum(value <= 1e-10 for value in residuals.values())),
        "weighted_mean_edge_residual": float(
            np.average(
                list(residuals.values()),
                weights=[confidences[edge] for edge in sorted(edges)],
            )
        )
        if residuals
        else 0.0,
    }


def _transport_synchronization(
    fits: Mapping[str, _AssayFit],
    *,
    reference: str,
    config: ComponentAlignmentConfig,
) -> tuple[dict[str, FloatArray], set[str], dict[str, float], dict[str, Any]]:
    modalities = sorted(fits)
    transforms = _signed_permutation_matrices(config.n_components)
    distributions = {
        modality: _quantize_distribution(
            fits[modality], modality=modality, config=config
        )
        for modality in modalities
    }
    rotations: dict[tuple[str, str], FloatArray] = {}
    confidences: dict[tuple[str, str], float] = {}
    preliminary: set[tuple[str, str]] = set()
    pairwise: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(modalities):
        for right in modalities[left_index + 1 :]:
            rotation, diagnostics = _fit_pair_transform(
                distributions[left],
                distributions[right],
                transforms=transforms,
                config=config,
            )
            rotations[(left, right)] = rotation
            rotations[(right, left)] = rotation.T
            edge = (left, right)
            confidence = float(
                diagnostics["relative_signed_permutation_margin"]
                * diagnostics["transport_separation"]
            )
            confidences[edge] = confidence
            passed = bool(
                fits[left].fold_stable
                and fits[right].fold_stable
                and diagnostics["relative_signed_permutation_margin"]
                >= config.min_pair_transport_margin
                and diagnostics["transport_separation"]
                >= config.min_pair_transport_separation
            )
            if passed:
                preliminary.add(edge)
            pairwise[f"{left}||{right}"] = {
                **diagnostics,
                "confidence": confidence,
                "preliminary_gate_accepted": passed,
            }
    cycle_edges, cycle_evidence = _cycle_filter(
        modalities, reference, rotations, preliminary, config
    )
    accepted_assays, accepted_edges = _reference_core(
        modalities, reference, cycle_edges, config
    )
    if len(accepted_assays) > 1:
        synchronized, margins, sync_fit = _synchronize_discrete_transforms(
            reference,
            rotations,
            confidences,
            accepted_assays,
            accepted_edges,
            transforms,
            config,
        )
        exact_edges = {
            tuple(key.split("||"))
            for key, residual in sync_fit["accepted_edge_residuals"].items()
            if float(residual) <= 1e-10
        }
        if exact_edges != accepted_edges:
            accepted_assays, accepted_edges = _reference_core(
                modalities,
                reference,
                exact_edges,  # type: ignore[arg-type]
                config,
            )
            if len(accepted_assays) > 1:
                synchronized, margins, sync_fit = _synchronize_discrete_transforms(
                    reference,
                    rotations,
                    confidences,
                    accepted_assays,
                    accepted_edges,
                    transforms,
                    config,
                )
            else:
                synchronized = {reference: np.eye(config.n_components)}
                margins = {reference: 1.0}
                sync_fit = {
                    "iterations": 0,
                    "accepted_edge_residuals": {},
                    "exact_consensus_edges": 0,
                    "weighted_mean_edge_residual": 0.0,
                }
        weak = {
            modality
            for modality in accepted_assays - {reference}
            if margins[modality] < config.min_assay_synchronization_margin
        }
        if weak:
            reduced_edges = {
                edge for edge in accepted_edges if not set(edge).intersection(weak)
            }
            accepted_assays, accepted_edges = _reference_core(
                sorted(set(modalities) - weak), reference, reduced_edges, config
            )
            if len(accepted_assays) > 1:
                synchronized, margins, sync_fit = _synchronize_discrete_transforms(
                    reference,
                    rotations,
                    confidences,
                    accepted_assays,
                    accepted_edges,
                    transforms,
                    config,
                )
            else:
                synchronized = {reference: np.eye(config.n_components)}
                margins = {reference: 1.0}
                sync_fit = {
                    "iterations": 0,
                    "accepted_edge_residuals": {},
                    "exact_consensus_edges": 0,
                    "weighted_mean_edge_residual": 0.0,
                }
    else:
        synchronized = {reference: np.eye(config.n_components)}
        margins = {reference: 1.0}
        sync_fit = {
            "iterations": 0,
            "accepted_edge_residuals": {},
            "exact_consensus_edges": 0,
            "weighted_mean_edge_residual": 0.0,
        }
    for key, evidence in cycle_evidence.items():
        pairwise[key].update(evidence)
        left, right = key.split("||")
        pairwise[key]["consensus_edge_accepted"] = (left, right) in accepted_edges
    return synchronized, accepted_assays, margins, {
        "signed_permutation_candidates_per_pair": len(transforms),
        "pairwise_transport_diagnostics": pairwise,
        "preliminary_accepted_edges": [f"{a}||{b}" for a, b in sorted(preliminary)],
        "cycle_accepted_edges": [f"{a}||{b}" for a, b in sorted(cycle_edges)],
        "consensus_accepted_edges": [f"{a}||{b}" for a, b in sorted(accepted_edges)],
        "accepted_assays": sorted(accepted_assays),
        "abstained_assays": sorted(set(modalities) - accepted_assays),
        "assay_synchronization_margin": margins,
        **sync_fit,
    }


def align_component_identified_background_states(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    *,
    reference_modality: str = "rna",
    config: ComponentAlignmentConfig | None = None,
) -> ComponentAlignmentResult:
    """Estimate a common signed-permutation component frame from backgrounds.

    The signature intentionally has no argument for focal measurements, truth,
    feature IDs, cell IDs, or pairings.  Matrix rows are used together only
    across feature folds of the *same* assay.
    """

    cfg = config or ComponentAlignmentConfig()
    modalities = sorted(str(key) for key in background_matrices)
    if len(modalities) < 2:
        raise ValueError("At least two assays are required")
    if set(modalities) != {str(key) for key in background_depths}:
        raise ValueError("Matrix and depth mappings must have identical assay keys")
    if reference_modality not in modalities:
        raise ValueError("reference_modality is absent")
    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    fits: dict[str, _AssayFit] = {}
    for modality in modalities:
        matrix = _as_matrix(
            background_matrices[modality], max_entries=cfg.max_dense_entries
        )
        depth = _as_depth(background_depths[modality], matrix=matrix)
        if matrix.shape[1] < cfg.n_components * cfg.n_feature_folds:
            raise ValueError(f"Assay {modality!r} has too few features for fold fitting")
        matrices[modality] = matrix
        depths[modality] = depth
        fits[modality] = _fit_assay(
            matrix, depth, modality=modality, config=cfg
        )

    transforms, synchronized_assays, synchronization_margin, synchronization = (
        _transport_synchronization(
            fits, reference=reference_modality, config=cfg
        )
    )
    embeddings: dict[str, FloatArray] = {}
    uncertainty: dict[str, FloatArray] = {}
    eligible: dict[str, bool] = {}
    confidence: dict[str, float] = {}
    evidence_rows: list[dict[str, Any]] = []
    for modality in modalities:
        transform = transforms.get(
            modality, np.eye(cfg.n_components, dtype=np.float64)
        )
        coordinates = fits[modality].scores @ transform
        # The axes have already been identified.  Componentwise standardization
        # removes assay measurement scale without introducing a rotation.
        coordinates = _standardize_components(coordinates)
        passed = bool(
            fits[modality].fold_stable
            and modality in synchronized_assays
            and synchronization_margin.get(modality, 0.0)
            >= cfg.min_assay_synchronization_margin
        )
        eligible[modality] = passed
        observed_confidence = min(
            fits[modality].minimum_fold_correlation,
            fits[modality].minimum_fold_margin
            / max(cfg.min_fold_assignment_margin, 1e-12),
            synchronization_margin.get(modality, 0.0)
            / max(cfg.min_assay_synchronization_margin, 1e-12),
            1.0,
        )
        confidence[modality] = float(max(observed_confidence, 0.0)) if passed else 0.0
        embeddings[modality] = np.asarray(coordinates, dtype=np.float64)
        uncertainty[modality] = np.sqrt(
            np.square(fits[modality].uncertainty)
            + (0.0 if passed else cfg.abstention_uncertainty_penalty**2)
        )
        for axis in range(cfg.n_components):
            local_component = int(np.argmax(np.abs(transform[:, axis])))
            selected_sign = int(np.sign(transform[local_component, axis]))
            evidence_rows.append(
                {
                    "modality": modality,
                    "global_component": axis,
                    "local_component": local_component,
                    "selected_sign": selected_sign,
                    "component_strength": float(
                        fits[modality].strength[local_component]
                    ),
                    "minimum_fold_component_correlation": fits[
                        modality
                    ].minimum_fold_correlation,
                    "minimum_fold_assignment_margin": fits[
                        modality
                    ].minimum_fold_margin,
                    "assay_synchronization_margin": synchronization_margin.get(
                        modality, 0.0
                    ),
                    "transport_synchronization_accepted": (
                        modality in synchronized_assays
                    ),
                    "assay_eligible": passed,
                }
            )

    provenance = {
        "schema_version": "component_identified_alignment_v4.v1",
        "method": (
            "unwhitened_pca_signed_permutation_synchronization"
            if cfg.method == "pca"
            else "two_fold_fastica_signed_permutation_synchronization"
        ),
        "reference_modality": reference_modality,
        "config": asdict(cfg),
        "assay_diagnostics": {
            modality: fits[modality].diagnostics for modality in modalities
        },
        "selected_signed_permutation_transforms": {
            modality: transforms.get(
                modality, np.eye(cfg.n_components, dtype=np.float64)
            ).tolist()
            for modality in modalities
        },
        "transport_synchronization": synchronization,
        "assay_eligible": eligible,
        "assay_alignment_confidence": confidence,
        "accepted_assays": sorted(
            modality for modality in modalities if eligible[modality]
        ),
        "abstained_assays": sorted(
            modality for modality in modalities if not eligible[modality]
        ),
        "identifiability": {
            "preidentification_pca_whitening_used": False,
            "fastica_internal_whitening_used": cfg.method == "fastica",
            "postidentification_componentwise_standardization_used": True,
            "cross_assay_rotation_fitted": False,
            "cross_assay_transform_family": "signed_permutation_only",
            "component_identity_source": (
                "K_microstate_multivariate_entropic_transport_over_exactly_48_signed_permutations"
            ),
            "component_sign_source": "cycle_gated_discrete_signed_permutation_synchronization",
            "weak_identification_causes_abstention": True,
        },
        "information_boundary": {
            "consumed": [
                "observed_background_matrix_per_assay",
                "observed_leave_focal_out_depth_per_assay",
                "assay_key",
            ],
            "not_consumed": [
                "focal_features",
                "true_latent_state",
                "simulator_roles",
                "truth_graph",
                "scorer",
                "paired_cells",
                "cross_assay_cell_identity",
                "cross_assay_feature_identity",
                "external_data",
            ],
            "same_assay_rows_used_for_feature_fold_reconciliation": True,
            "cross_assay_rows_compared": False,
            "cross_assay_feature_columns_compared": False,
            "cross_assay_quantized_distributions_compared": True,
            "transport_plan_returned_as_pairing": False,
            "truth_or_scorer_used": False,
        },
    }
    return ComponentAlignmentResult(
        embeddings=embeddings,
        embedding_uncertainty=uncertainty,
        assay_eligible=eligible,
        assay_alignment_confidence=confidence,
        component_evidence=pd.DataFrame(evidence_rows),
        provenance=provenance,
    )


def _validated_discovery_root(path: Path | str) -> Path:
    lexical = Path(path).absolute()
    if lexical.name.casefold() != "discovery":
        raise ValueError("Input must be a simulator/world discovery/ directory")
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise ValueError("Discovery input must be a direct existing directory")
    root = lexical.resolve(strict=True)
    if any(part.casefold() in _FORBIDDEN_PATH_COMPONENTS for part in root.parts):
        raise ValueError("Refusing scorer/truth discovery input")
    return root


def _load_background_archives(
    discovery_dir: Path | str,
) -> tuple[
    Path,
    dict[str, FloatArray],
    dict[str, FloatArray],
    dict[str, NDArray[np.str_]],
    dict[str, str],
]:
    root = _validated_discovery_root(discovery_dir)
    paths = sorted(root.glob(f"{_BACKGROUND_PREFIX}*.npz"))
    if len(paths) < 2:
        raise ValueError("Discovery input needs at least two background archives")
    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    cell_ids: dict[str, NDArray[np.str_]] = {}
    hashes: dict[str, str] = {}
    seen_cells: set[str] = set()
    for path in paths:
        if path.is_symlink() or path.resolve(strict=True).parent != root:
            raise ValueError("Background archives must be direct regular files")
        modality = path.name.removeprefix(_BACKGROUND_PREFIX).removesuffix(".npz")
        if not modality or modality in matrices:
            raise ValueError("Background archive modality names must be unique")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"matrix", "cell_ids", "background_depth"}:
                    raise ValueError("Background archives contain an unexpected field")
                matrix = np.asarray(archive["matrix"], dtype=np.float64)
                depth = np.asarray(archive["background_depth"], dtype=np.float64)
                ids = np.asarray(archive["cell_ids"], dtype=str)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load safe background archive {path.name}") from exc
        matrix = _as_matrix(matrix, max_entries=50_000_000)
        depth = _as_depth(depth, matrix=matrix)
        if ids.shape != (len(matrix),) or len(set(ids.tolist())) != len(ids):
            raise ValueError("Background cell IDs must be aligned and unique")
        if not seen_cells.isdisjoint(ids.tolist()):
            raise ValueError("Cell IDs must be disjoint across unpaired assays")
        seen_cells.update(ids.tolist())
        matrices[modality] = matrix
        depths[modality] = depth
        cell_ids[modality] = ids
        hashes[path.name] = _sha256(path)
    if "rna" not in matrices:
        raise ValueError("Frozen v4 alignment requires an RNA reference assay")
    return root, matrices, depths, cell_ids, hashes


def _validated_output(path: Path | str, *, input_root: Path) -> Path:
    lexical = Path(path).absolute()
    resolved = lexical.resolve(strict=False)
    if resolved == input_root or input_root in resolved.parents:
        raise ValueError("Output must be outside the discovery input tree")
    if any(part.casefold() in _FORBIDDEN_PATH_COMPONENTS for part in resolved.parts):
        raise ValueError("Observed-only output may not enter scorer/truth paths")
    if lexical.exists() and (not lexical.is_dir() or any(lexical.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {lexical}")
    lexical.mkdir(parents=True, exist_ok=True)
    return lexical.resolve(strict=True)


def freeze_component_alignment_evidence(
    discovery_dir: Path | str,
    out_dir: Path | str,
    *,
    config: ComponentAlignmentConfig | None = None,
) -> dict[str, Any]:
    """Fit and content-address observed-only alignment evidence before scoring."""

    cfg = config or ComponentAlignmentConfig()
    root, matrices, depths, cell_ids, input_hashes = _load_background_archives(
        discovery_dir
    )
    output = _validated_output(out_dir, input_root=root)
    result = align_component_identified_background_states(
        matrices,
        depths,
        reference_modality="rna",
        config=cfg,
    )
    rows: list[pd.DataFrame] = []
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
    evidence_path = output / "component_evidence.tsv"
    provenance_path = output / "alignment_provenance.json"
    pd.concat(rows, ignore_index=True).to_csv(embedding_path, sep="\t", index=False)
    result.component_evidence.to_csv(evidence_path, sep="\t", index=False)
    provenance_path.write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    source = Path(__file__).resolve(strict=True)
    manifest = {
        "schema_version": "component_identified_alignment_v4.frozen_evidence.v1",
        "algorithm_source_sha256": _sha256(source),
        "config": asdict(cfg),
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            embedding_path.name: _sha256(embedding_path),
            evidence_path.name: _sha256(evidence_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "accepted_assays": result.provenance["accepted_assays"],
        "abstained_assays": result.provenance["abstained_assays"],
        "frozen_before_scorer_read": True,
        "information_boundary": result.provenance["information_boundary"],
    }
    manifest_path = output / "frozen_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    (output / "frozen_evidence_manifest.sha256").write_text(
        _sha256(manifest_path) + "\n"
    )
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def freeze_observed_axis_directory(
    axis_discovery_dir: Path | str,
    *,
    source_paths: Sequence[Path | str],
) -> dict[str, Any]:
    """Hash-lock a completed observed-only axis prototype before scorer access."""

    lexical = Path(axis_discovery_dir).absolute()
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise ValueError("Axis discovery directory must be a direct existing directory")
    root = lexical.resolve(strict=True)
    if any(part.casefold() in _FORBIDDEN_PATH_COMPONENTS for part in root.parts):
        raise ValueError("Refusing scorer/truth axis discovery directory")
    manifest_path = root / "axis_pretruth_freeze_manifest.json"
    digest_path = root / "axis_pretruth_freeze_manifest.sha256"
    if manifest_path.exists() or digest_path.exists():
        raise FileExistsError("Axis discovery directory is already frozen")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Axis discovery evidence may not contain symlinks")
        if path.is_file():
            files[str(path.relative_to(root))] = {
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
    sources: dict[str, str] = {}
    for raw in source_paths:
        path = Path(raw).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Axis prototype sources must be direct regular files")
        resolved = path.resolve(strict=True)
        if any(part.casefold() in _FORBIDDEN_PATH_COMPONENTS for part in resolved.parts):
            raise ValueError("Refusing scorer/truth source path")
        sources[str(resolved)] = _sha256(resolved)
    manifest = {
        "schema_version": "component_identified_axis_pretruth_freeze.v1",
        "frozen_before_scorer_read": True,
        "files": files,
        "source_sha256": sources,
        "information_boundary": {
            "scorer_or_truth_read": False,
            "paired_cells_used": False,
            "cross_assay_feature_identity_used": False,
            "external_data_used": False,
        },
    }
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    digest = _sha256(manifest_path)
    digest_path.write_text(digest + "\n")
    return {**manifest, "manifest_sha256": digest}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("pca", "fastica"), default="fastica")
    parser.add_argument(
        "--transport-microstates", type=int, choices=(6, 24), default=6
    )
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = freeze_component_alignment_evidence(
        args.discovery,
        args.out,
        config=ComponentAlignmentConfig(
            method=args.method,
            n_transport_microstates=args.transport_microstates,
            random_seed=args.random_seed,
        ),
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ComponentAlignmentConfig",
    "ComponentAlignmentResult",
    "align_component_identified_background_states",
    "freeze_component_alignment_evidence",
    "freeze_observed_axis_directory",
]
