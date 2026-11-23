#!/usr/bin/env python3
"""Soft distributional alignment for independently sampled assay states.

This module resolves the remaining orthogonal ambiguity of independently
estimated background-state coordinates.  It does **not** estimate cell
correspondence.  Each assay is independently quantized, and only the two
weighted empirical distributions of microstate centroids enter an entropic
optimal-transport/Procrustes refinement.

The estimator deliberately exposes a narrow observed-only API: one private
background matrix and its leave-focal-out depth vector per assay.  It has no
arguments for focal variables, feature links, paired cells, truth, simulator
labels, or privileged latent states.  Population prevalence is strongly
shrunk toward uniform before transport, so moderate assay-specific population
shifts do not determine the coordinate system.  Cross-fitted background
feature folds provide per-cell uncertainty and a stability diagnostic.

Pair rotations are confidence-gated from observable transport separation and
near-optimal ambiguity, checked against multi-assay rotation cycles, and then
synchronized with the RNA coordinate system fixed.  An assay outside the
confidence core explicitly abstains and cannot alter accepted rotations.  The
result is suitable for comparing *state response loadings*.  A transport plan
is an optimization device between empirical distributions; it is neither
returned nor interpretable as a cell or microstate pairing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans

try:
    import unpaired_state_alignment as hard_alignment
except ImportError as exc:  # pragma: no cover - partial checkout only
    raise ImportError(
        "soft_distributional_alignment requires unpaired_state_alignment.py"
    ) from exc


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SoftAlignmentConfig:
    """Frozen settings for observed-only soft distributional alignment."""

    n_components: int = 3
    n_microstates: int = 24
    min_cells_per_microstate: int = 20
    n_feature_folds: int = 2
    random_seed: int = 0
    kmeans_n_init: int = 20
    population_uniform_shrinkage: float = 0.75
    uncertainty_floor: float = 0.10
    entropy_fraction: float = 0.08
    n_random_starts: int = 12
    max_refinement_iterations: int = 40
    sinkhorn_iterations: int = 250
    convergence_tolerance: float = 1e-7
    synchronize_assays: bool = True
    max_pair_rotation_dispersion: float = 0.30
    min_pair_transport_separation: float = 0.10
    min_synchronization_neighbors: int = 2
    min_cycle_support: int = 1
    max_cycle_inconsistency: float = 0.35
    synchronization_iterations: int = 80
    abstention_uncertainty_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least two")
        if self.n_microstates <= self.n_components:
            raise ValueError("n_microstates must exceed n_components")
        if self.min_cells_per_microstate < 2:
            raise ValueError("min_cells_per_microstate must be at least two")
        if self.n_feature_folds < 2:
            raise ValueError("n_feature_folds must be at least two")
        if self.random_seed < 0 or self.kmeans_n_init < 1:
            raise ValueError("random_seed and kmeans_n_init are invalid")
        if not 0.0 <= self.population_uniform_shrinkage <= 1.0:
            raise ValueError("population_uniform_shrinkage must lie in [0, 1]")
        if not 0.0 < self.uncertainty_floor <= 1.0:
            raise ValueError("uncertainty_floor must lie in (0, 1]")
        if self.entropy_fraction <= 0:
            raise ValueError("entropy_fraction must be positive")
        if self.n_random_starts < 0:
            raise ValueError("n_random_starts may not be negative")
        if min(self.max_refinement_iterations, self.sinkhorn_iterations) < 1:
            raise ValueError("iteration counts must be positive")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive")
        if self.max_pair_rotation_dispersion <= 0:
            raise ValueError("max_pair_rotation_dispersion must be positive")
        if not 0.0 <= self.min_pair_transport_separation < 1.0:
            raise ValueError("min_pair_transport_separation must lie in [0, 1)")
        if self.min_synchronization_neighbors < 1:
            raise ValueError("min_synchronization_neighbors must be positive")
        if self.min_cycle_support < 1:
            raise ValueError("min_cycle_support must be positive")
        if self.max_cycle_inconsistency <= 0:
            raise ValueError("max_cycle_inconsistency must be positive")
        if self.synchronization_iterations < 1:
            raise ValueError("synchronization_iterations must be positive")
        if self.abstention_uncertainty_penalty <= 0:
            raise ValueError("abstention_uncertainty_penalty must be positive")


@dataclass(frozen=True)
class SoftDistributionalAlignmentResult:
    """Common background coordinates and observable-only diagnostics."""

    embeddings: dict[str, FloatArray]
    embedding_uncertainty: dict[str, FloatArray]
    provenance: dict[str, Any]
    assay_synchronized: dict[str, bool]
    assay_alignment_confidence: dict[str, float]


@dataclass(frozen=True)
class _QuantizedDistribution:
    centroids: FloatArray
    mass: FloatArray
    effective_cells: float


_BACKGROUND_PREFIX = "background_matrix__"
_FORBIDDEN_PATH_COMPONENTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
    "counterfactual",
}


def _whiten(values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ZCA-whiten an assay without defining axes from another assay."""

    array = np.asarray(values, dtype=np.float64)
    center = array.mean(axis=0)
    centered = array - center
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-10)
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    whitened = centered @ inverse_root
    return (
        np.asarray(whitened, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.asarray(inverse_root, dtype=np.float64),
    )


def _cell_reliability(
    uncertainty: FloatArray,
    *,
    floor: float,
) -> FloatArray:
    values = np.asarray(uncertainty, dtype=np.float64)
    positive = values[values > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    reliability = 1.0 / (1.0 + np.square(values / max(scale, 1e-12)))
    return np.asarray(np.clip(reliability, floor, 1.0), dtype=np.float64)


def _quantize(
    coordinates: FloatArray,
    reliability: FloatArray,
    *,
    n_microstates: int,
    seed: int,
    config: SoftAlignmentConfig,
) -> _QuantizedDistribution:
    model = KMeans(
        n_clusters=n_microstates,
        random_state=seed,
        n_init=config.kmeans_n_init,
    )
    model.fit(coordinates, sample_weight=reliability)
    labels = np.asarray(model.labels_, dtype=np.int64)
    centroids: list[FloatArray] = []
    raw_mass: list[float] = []
    for cluster in range(n_microstates):
        selected = labels == cluster
        weights = reliability[selected]
        if not np.any(selected) or float(weights.sum()) <= 0:
            raise RuntimeError("Weighted quantization produced an empty microstate")
        centroids.append(
            np.asarray(
                np.average(coordinates[selected], axis=0, weights=weights),
                dtype=np.float64,
            )
        )
        raw_mass.append(float(weights.sum()))
    mass = np.asarray(raw_mass, dtype=np.float64)
    mass /= mass.sum()
    uniform = np.full(n_microstates, 1.0 / n_microstates, dtype=np.float64)
    shrinkage = config.population_uniform_shrinkage
    mass = (1.0 - shrinkage) * mass + shrinkage * uniform
    mass /= mass.sum()
    effective_cells = float(
        np.square(reliability.sum()) / max(float(np.square(reliability).sum()), 1e-12)
    )
    return _QuantizedDistribution(
        centroids=np.vstack(centroids),
        mass=np.asarray(mass, dtype=np.float64),
        effective_cells=effective_cells,
    )


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
    """Balanced entropic transport in the log domain for numerical stability."""

    log_kernel = -np.asarray(cost, dtype=np.float64) / max(regularization, 1e-12)
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
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = np.exp(np.clip(log_plan, -745.0, 700.0))
    # The final two matrix-scaling steps remove accumulated floating-point drift.
    plan *= (left_mass / np.maximum(plan.sum(axis=1), 1e-300))[:, None]
    plan *= (right_mass / np.maximum(plan.sum(axis=0), 1e-300))[None, :]
    return np.asarray(plan, dtype=np.float64)


def _orthogonal_update(
    target: FloatArray,
    reference: FloatArray,
    coupling: FloatArray,
) -> FloatArray:
    cross = target.T @ coupling @ reference
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    return np.asarray(left @ right_t, dtype=np.float64)


def _signed_permutation_starts(dimensions: int) -> list[FloatArray]:
    starts: list[FloatArray] = []
    identity = np.eye(dimensions, dtype=np.float64)
    for order in permutations(range(dimensions)):
        permuted = identity[:, list(order)]
        for signs in product((-1.0, 1.0), repeat=dimensions):
            starts.append(permuted * np.asarray(signs, dtype=np.float64)[None, :])
    return starts


def _random_orthogonal_starts(
    dimensions: int,
    count: int,
    rng: np.random.Generator,
) -> list[FloatArray]:
    starts: list[FloatArray] = []
    for _ in range(count):
        left, _, right_t = np.linalg.svd(
            rng.normal(size=(dimensions, dimensions)), full_matrices=False
        )
        starts.append(np.asarray(left @ right_t, dtype=np.float64))
    return starts


def _transport_objective(
    target: _QuantizedDistribution,
    reference: _QuantizedDistribution,
    rotation: FloatArray,
    config: SoftAlignmentConfig,
) -> tuple[float, FloatArray, float]:
    cost = _squared_distances(target.centroids @ rotation, reference.centroids)
    positive = cost[cost > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    regularization = max(config.entropy_fraction * scale, 1e-8)
    coupling = _sinkhorn_coupling(
        cost,
        target.mass,
        reference.mass,
        regularization=regularization,
        iterations=config.sinkhorn_iterations,
        tolerance=config.convergence_tolerance,
    )
    return float(np.sum(coupling * cost)), coupling, regularization


def _refine_rotation(
    target: _QuantizedDistribution,
    reference: _QuantizedDistribution,
    initial: FloatArray,
    config: SoftAlignmentConfig,
) -> tuple[FloatArray, float, int, float]:
    rotation = np.asarray(initial, dtype=np.float64).copy()
    previous = float("inf")
    regularization = float("nan")
    for iteration in range(config.max_refinement_iterations):
        objective, coupling, regularization = _transport_objective(
            target, reference, rotation, config
        )
        updated = _orthogonal_update(target.centroids, reference.centroids, coupling)
        delta = float(np.linalg.norm(updated - rotation, ord="fro"))
        rotation = updated
        if abs(previous - objective) <= config.convergence_tolerance * max(
            1.0, abs(previous)
        ) and delta <= np.sqrt(config.convergence_tolerance):
            break
        previous = objective
    objective, _, regularization = _transport_objective(
        target, reference, rotation, config
    )
    return rotation, objective, iteration + 1, regularization


def _rotation_dispersion(
    rotations: list[FloatArray],
    objectives: FloatArray,
) -> tuple[float, float, int]:
    best = float(objectives.min())
    tolerance = max(0.01 * max(abs(best), 1e-12), 1e-8)
    near = np.flatnonzero(objectives <= best + tolerance)
    reference = rotations[int(np.argmin(objectives))]
    distances = [
        float(np.linalg.norm(rotations[index] - reference, ord="fro"))
        / np.sqrt(2.0 * reference.shape[0])
        for index in near
    ]
    ordered = np.sort(objectives)
    gap = float(ordered[1] - ordered[0]) if len(ordered) > 1 else float("inf")
    return float(np.mean(distances)), gap, int(len(near))


def _align_one_distribution(
    target: _QuantizedDistribution,
    reference: _QuantizedDistribution,
    *,
    seed: int,
    config: SoftAlignmentConfig,
) -> tuple[FloatArray, dict[str, Any]]:
    dimensions = target.centroids.shape[1]
    rng = np.random.default_rng(seed)
    starts = _signed_permutation_starts(dimensions)
    starts.extend(
        _random_orthogonal_starts(dimensions, config.n_random_starts, rng)
    )
    rotations: list[FloatArray] = []
    objectives: list[float] = []
    iterations: list[int] = []
    regularizations: list[float] = []
    for start in starts:
        rotation, objective, used, regularization = _refine_rotation(
            target, reference, start, config
        )
        rotations.append(rotation)
        objectives.append(objective)
        iterations.append(used)
        regularizations.append(regularization)
    objective_array = np.asarray(objectives, dtype=np.float64)
    best_index = int(np.argmin(objective_array))
    dispersion, gap, near_count = _rotation_dispersion(rotations, objective_array)
    best_rotation = rotations[best_index]
    best_cost = _squared_distances(
        target.centroids @ best_rotation, reference.centroids
    )
    independent_objective = float(
        np.sum(np.outer(target.mass, reference.mass) * best_cost)
    )
    transport_separation = float(
        np.clip(
            (independent_objective - float(objective_array[best_index]))
            / max(independent_objective, 1e-12),
            0.0,
            1.0,
        )
    )
    median_objective = float(np.median(objective_array))
    relative_start_objective_gain = float(
        np.clip(
            (median_objective - float(objective_array[best_index]))
            / max(abs(median_objective), 1e-12),
            0.0,
            1.0,
        )
    )
    return best_rotation, {
        "transport_objective": float(objective_array[best_index]),
        "independent_coupling_objective": independent_objective,
        "transport_separation": transport_separation,
        "median_start_objective": median_objective,
        "relative_start_objective_gain": relative_start_objective_gain,
        "second_best_objective_gap": gap,
        "near_optimal_rotation_count": near_count,
        "near_optimal_rotation_dispersion": dispersion,
        "n_starts": len(starts),
        "refinement_iterations": int(iterations[best_index]),
        "sinkhorn_regularization": float(regularizations[best_index]),
        "rotation_determinant": float(np.linalg.det(rotations[best_index])),
        "target_effective_cells": target.effective_cells,
        "reference_effective_cells": reference.effective_cells,
    }


def _orthogonal_procrustes(source: FloatArray, target: FloatArray) -> FloatArray:
    cross = source.T @ target
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    return np.asarray(left @ right_t, dtype=np.float64)


def _stable_pair_seed(base_seed: int, left: str, right: str) -> int:
    """Derive a pair seed that is invariant to unrelated assay additions."""

    digest = hashlib.sha256(
        f"soft-sync-v1\0{base_seed}\0{left}\0{right}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _rotation_distance(left: FloatArray, right: FloatArray) -> float:
    dimensions = left.shape[0]
    return float(np.linalg.norm(left - right, ord="fro")) / np.sqrt(
        2.0 * dimensions
    )


def _project_orthogonal(matrix: FloatArray) -> FloatArray:
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return np.asarray(left @ right_t, dtype=np.float64)


def _pair_key(left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{first}||{second}"


def _pair_confidence(
    diagnostics: Mapping[str, Any],
    config: SoftAlignmentConfig,
) -> tuple[float, bool, list[str]]:
    dispersion = float(diagnostics["near_optimal_rotation_dispersion"])
    separation = float(diagnostics["transport_separation"])
    reasons: list[str] = []
    if dispersion > config.max_pair_rotation_dispersion:
        reasons.append("rotation_ambiguity")
    if separation < config.min_pair_transport_separation:
        reasons.append("weak_transport_separation")
    # Separation measures how much observed joint transport improves over the
    # product coupling.  Dispersion measures whether nearly equivalent fits
    # imply different rotations.  Neither quantity uses focal variables.
    confidence = float(
        np.clip(separation, 0.0, 1.0)
        * np.exp(-dispersion / config.max_pair_rotation_dispersion)
    )
    return confidence, not reasons, reasons


def _cycle_consistency(
    modalities: list[str],
    reference: str,
    rotations: Mapping[tuple[str, str], FloatArray],
    preliminary_edges: set[tuple[str, str]],
    config: SoftAlignmentConfig,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    """Reject pair rotations contradicted by several observed assay triangles."""

    accepted: set[tuple[str, str]] = set()
    diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
    for left, right in sorted(preliminary_edges):
        errors: list[float] = []
        for bridge in modalities:
            if bridge in {left, right}:
                continue
            left_bridge = tuple(sorted((left, bridge)))
            bridge_right = tuple(sorted((bridge, right)))
            if (
                left_bridge not in preliminary_edges
                or bridge_right not in preliminary_edges
            ):
                continue
            indirect = rotations[(left, bridge)] @ rotations[(bridge, right)]
            errors.append(_rotation_distance(rotations[(left, right)], indirect))
        median = float(np.median(errors)) if errors else float("nan")
        minimum = float(np.min(errors)) if errors else float("nan")
        consistent_count = int(
            np.sum(np.asarray(errors) <= config.max_cycle_inconsistency)
        )
        # Direct-to-reference fits remain anchors when a single triangle cannot
        # identify which of its three edges is wrong.  With two or more
        # independent triangles they must receive ordinary cycle support.
        reference_edge = reference in {left, right}
        required_available = 2 if reference_edge else 1
        gate_applied = len(errors) >= required_available
        edge_accepted = (
            not gate_applied or consistent_count >= config.min_cycle_support
        )
        if edge_accepted:
            accepted.add((left, right))
        diagnostics[(left, right)] = {
            "cycle_support": len(errors),
            "consistent_cycle_support": consistent_count,
            "median_cycle_inconsistency": median,
            "minimum_cycle_inconsistency": minimum,
            "cycle_gate_applied": gate_applied,
            "cycle_gate_accepted": edge_accepted,
            "reference_anchor_edge": reference_edge,
        }
    return accepted, diagnostics


def _reference_consensus_component(
    modalities: list[str],
    reference: str,
    edges: set[tuple[str, str]],
    config: SoftAlignmentConfig,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Return a confidence graph core connected to the fixed reference assay."""

    active = set(modalities)
    required = min(
        config.min_synchronization_neighbors,
        max(1, len(modalities) - 2),
    )
    if len(modalities) == 2:
        required = 1
    changed = True
    while changed:
        changed = False
        degrees = {
            modality: sum(
                1
                for left, right in edges
                if modality in {left, right}
                and left in active
                and right in active
            )
            for modality in active
        }
        remove = {
            modality
            for modality in active
            if modality != reference and degrees[modality] < required
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
        for neighbor in adjacency.get(current, ()):
            if neighbor not in connected:
                connected.add(neighbor)
                frontier.append(neighbor)
    return connected, {
        edge for edge in active_edges if edge[0] in connected and edge[1] in connected
    }


def _synchronize_rotations(
    modalities: list[str],
    reference: str,
    rotations: Mapping[tuple[str, str], FloatArray],
    confidences: Mapping[tuple[str, str], float],
    accepted_assays: set[str],
    accepted_edges: set[tuple[str, str]],
    config: SoftAlignmentConfig,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Solve weighted orthogonal synchronization with the reference fixed."""

    dimensions = next(iter(rotations.values())).shape[0]
    synchronized: dict[str, FloatArray] = {
        reference: np.eye(dimensions, dtype=np.float64)
    }
    # A maximum-confidence growth pass provides a deterministic, gauge-fixed
    # initialization even when no assay has a direct accepted edge to RNA.
    while set(synchronized) != accepted_assays:
        choices: list[tuple[float, str, str]] = []
        for left, right in accepted_edges:
            if left in synchronized and right not in synchronized:
                choices.append((confidences[(left, right)], right, left))
            elif right in synchronized and left not in synchronized:
                choices.append((confidences[(left, right)], left, right))
        if not choices:  # pragma: no cover - component construction guarantees this
            raise RuntimeError("Accepted synchronization graph is disconnected")
        _, target, anchor = max(choices, key=lambda item: (item[0], item[1], item[2]))
        synchronized[target] = np.asarray(
            rotations[(target, anchor)] @ synchronized[anchor], dtype=np.float64
        )

    used_iterations = 0
    final_delta = float("inf")
    for iteration in range(config.synchronization_iterations):
        updated = {reference: synchronized[reference]}
        deltas: list[float] = []
        for modality in sorted(accepted_assays - {reference}):
            aggregate = np.zeros((dimensions, dimensions), dtype=np.float64)
            total_weight = 0.0
            for neighbor in sorted(accepted_assays - {modality}):
                edge = tuple(sorted((modality, neighbor)))
                if edge not in accepted_edges:
                    continue
                weight = float(confidences[edge])
                aggregate += weight * (
                    rotations[(modality, neighbor)] @ synchronized[neighbor]
                )
                total_weight += weight
            if total_weight <= 0:  # pragma: no cover - graph gate prevents this
                updated[modality] = synchronized[modality]
                continue
            updated[modality] = _project_orthogonal(aggregate / total_weight)
            deltas.append(
                _rotation_distance(updated[modality], synchronized[modality])
            )
        synchronized.update(updated)
        used_iterations = iteration + 1
        final_delta = max(deltas, default=0.0)
        if final_delta <= config.convergence_tolerance:
            break

    residuals = {
        _pair_key(left, right): _rotation_distance(
            synchronized[left], rotations[(left, right)] @ synchronized[right]
        )
        for left, right in sorted(accepted_edges)
    }
    return synchronized, {
        "iterations": used_iterations,
        "final_max_rotation_delta": final_delta,
        "accepted_edge_residuals": residuals,
        "weighted_mean_edge_residual": float(
            np.average(
                list(residuals.values()),
                weights=[confidences[edge] for edge in sorted(accepted_edges)],
            )
        )
        if residuals
        else 0.0,
    }


def align_soft_distributional_background_states(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    *,
    reference_modality: str | None = None,
    config: SoftAlignmentConfig | None = None,
) -> SoftDistributionalAlignmentResult:
    """Align unpaired assays through uncertainty-weighted state distributions.

    Feature folds are estimated independently within every assay.  Within each
    fold, independently whitened microstate distributions are aligned by soft
    transport and multi-start orthogonal Procrustes.  Feature folds are
    reconciled only through repeated cells *inside the same assay*.  No rows or
    columns are compared across assays.
    """

    cfg = config or SoftAlignmentConfig()
    modalities = sorted(str(key) for key in background_matrices)
    if len(modalities) < 2:
        raise ValueError("At least two assays are required")
    if set(modalities) != {str(key) for key in background_depths}:
        raise ValueError("background matrices and depths need identical assay keys")

    base_config = hard_alignment.AlignmentConfig(
        n_components=cfg.n_components,
        n_microstates=cfg.n_microstates,
        min_cells_per_microstate=cfg.min_cells_per_microstate,
        derange=False,
        random_seed=cfg.random_seed,
        kmeans_n_init=cfg.kmeans_n_init,
    )
    base = hard_alignment.align_cross_fitted_background_states(
        background_matrices,
        background_depths,
        reference_modality=reference_modality,
        n_feature_folds=cfg.n_feature_folds,
        config=base_config,
    )
    reference = str(base.provenance["reference_modality"])
    # Reconcile feature folds independently inside every assay before any
    # cross-assay distribution is compared.  Repeated rows belong to the same
    # assay and are legitimate for cross-fitting; no assay-to-assay rows enter.
    local_consensus: dict[str, FloatArray] = {}
    local_uncertainty: dict[str, FloatArray] = {}
    fold_reconciliation: dict[str, list[dict[str, Any]]] = {}
    for modality in modalities:
        whitened_folds = [
            _whiten(fold.embeddings[modality])[0] for fold in base.fold_results
        ]
        target = whitened_folds[0]
        reconciled = [target]
        diagnostics: list[dict[str, Any]] = [
            {"fold": 0, "within_assay_rowwise_procrustes_residual": 0.0}
        ]
        for fold_index, source in enumerate(whitened_folds[1:], start=1):
            rotation = _orthogonal_procrustes(source, target)
            transformed = np.asarray(source @ rotation, dtype=np.float64)
            residual = float(
                np.linalg.norm(transformed - target)
                / max(np.linalg.norm(target), 1e-12)
            )
            reconciled.append(transformed)
            diagnostics.append(
                {
                    "fold": fold_index,
                    "within_assay_rowwise_procrustes_residual": residual,
                    "rotation_determinant": float(np.linalg.det(rotation)),
                }
            )
        stack = np.stack(reconciled, axis=0)
        mean = np.asarray(stack.mean(axis=0), dtype=np.float64)
        local_consensus[modality] = _whiten(mean)[0]
        local_uncertainty[modality] = np.asarray(
            np.sqrt(
                np.mean(
                    np.sum(np.square(stack - mean[None, :, :]), axis=2), axis=0
                )
            ),
            dtype=np.float64,
        )
        fold_reconciliation[modality] = diagnostics

    distributions: dict[str, _QuantizedDistribution] = {}
    for modality in modalities:
        reliability = _cell_reliability(
            local_uncertainty[modality], floor=cfg.uncertainty_floor
        )
        seed = _stable_pair_seed(cfg.random_seed, "__quantize__", modality)
        distributions[modality] = _quantize(
            local_consensus[modality],
            reliability,
            n_microstates=cfg.n_microstates,
            seed=seed,
            config=cfg,
        )

    rotations: dict[tuple[str, str], FloatArray] = {}
    pair_confidences: dict[tuple[str, str], float] = {}
    preliminary_edges: set[tuple[str, str]] = set()
    pairwise_diagnostics: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(modalities):
        for right in modalities[left_index + 1 :]:
            seed = _stable_pair_seed(cfg.random_seed, left, right)
            rotation, diagnostics = _align_one_distribution(
                distributions[left],
                distributions[right],
                seed=seed,
                config=cfg,
            )
            rotations[(left, right)] = rotation
            rotations[(right, left)] = rotation.T
            edge = (left, right)
            confidence, pair_accepted, rejection_reasons = _pair_confidence(
                diagnostics, cfg
            )
            pair_confidences[edge] = confidence
            if pair_accepted:
                preliminary_edges.add(edge)
            pairwise_diagnostics[_pair_key(left, right)] = {
                **diagnostics,
                "confidence": confidence,
                "preliminary_gate_accepted": pair_accepted,
                "preliminary_rejection_reasons": rejection_reasons,
            }

    if cfg.synchronize_assays:
        cycle_edges, cycle_diagnostics = _cycle_consistency(
            modalities, reference, rotations, preliminary_edges, cfg
        )
        accepted_assays, accepted_edges = _reference_consensus_component(
            modalities, reference, cycle_edges, cfg
        )
        synchronized_rotations, sync_fit = _synchronize_rotations(
            modalities,
            reference,
            rotations,
            pair_confidences,
            accepted_assays,
            accepted_edges,
            cfg,
        )
    else:
        cycle_edges = preliminary_edges.copy()
        cycle_diagnostics = {
            edge: {
                "cycle_support": 0,
                "consistent_cycle_support": 0,
                "median_cycle_inconsistency": float("nan"),
                "minimum_cycle_inconsistency": float("nan"),
                "cycle_gate_applied": False,
                "cycle_gate_accepted": edge in cycle_edges,
                "reference_anchor_edge": reference in set(edge),
            }
            for edge in preliminary_edges
        }
        accepted_assays = {reference}
        accepted_edges: set[tuple[str, str]] = set()
        synchronized_rotations = {
            reference: np.eye(
                distributions[reference].centroids.shape[1], dtype=np.float64
            )
        }
        for modality in modalities:
            if modality == reference:
                continue
            edge = tuple(sorted((modality, reference)))
            if edge in preliminary_edges:
                accepted_assays.add(modality)
                accepted_edges.add(edge)
                synchronized_rotations[modality] = rotations[(modality, reference)]
        sync_fit = {
            "iterations": 0,
            "final_max_rotation_delta": 0.0,
            "accepted_edge_residuals": {},
            "weighted_mean_edge_residual": 0.0,
        }

    for edge, diagnostics in cycle_diagnostics.items():
        pairwise_diagnostics[_pair_key(*edge)].update(diagnostics)
        pairwise_diagnostics[_pair_key(*edge)]["consensus_edge_accepted"] = (
            edge in accepted_edges
        )
    for edge in set(pair_confidences) - set(cycle_diagnostics):
        pairwise_diagnostics[_pair_key(*edge)].update(
            {
                "cycle_support": 0,
                "consistent_cycle_support": 0,
                "median_cycle_inconsistency": float("nan"),
                "minimum_cycle_inconsistency": float("nan"),
                "cycle_gate_applied": False,
                "cycle_gate_accepted": False,
                "reference_anchor_edge": reference in set(edge),
                "consensus_edge_accepted": False,
            }
        )

    assay_synchronized = {
        modality: modality in accepted_assays for modality in modalities
    }
    assay_confidence: dict[str, float] = {}
    for modality in modalities:
        if modality == reference:
            assay_confidence[modality] = 1.0
            continue
        incident = [
            pair_confidences[edge]
            for edge in accepted_edges
            if modality in edge
        ]
        assay_confidence[modality] = float(np.mean(incident)) if incident else 0.0

    embeddings: dict[str, FloatArray] = {}
    uncertainty: dict[str, FloatArray] = {}
    distribution_diagnostics: dict[str, Any] = {
        reference: {
            "transport_objective": 0.0,
            "near_optimal_rotation_dispersion": 0.0,
            "n_starts": 0,
            "reference_coordinate_system": True,
            "synchronization_accepted": True,
        }
    }
    for modality in modalities:
        if modality == reference:
            embeddings[modality] = local_consensus[modality].copy()
            uncertainty[modality] = local_uncertainty[modality].copy()
            continue
        direct = pairwise_diagnostics[_pair_key(modality, reference)]
        if assay_synchronized[modality]:
            embeddings[modality] = np.asarray(
                local_consensus[modality] @ synchronized_rotations[modality],
                dtype=np.float64,
            )
            alignment_ambiguity = float(
                max(
                    (
                        pairwise_diagnostics[_pair_key(*edge)][
                            "near_optimal_rotation_dispersion"
                        ]
                        for edge in accepted_edges
                        if modality in edge
                    ),
                    default=0.0,
                )
            )
            abstention_penalty = 0.0
        else:
            # Preserve a finite within-assay state estimate, but do not rotate it
            # into the shared frame.  The explicit status bit and large added
            # uncertainty make these coordinates ineligible for cross-assay use.
            embeddings[modality] = local_consensus[modality].copy()
            alignment_ambiguity = float(
                direct["near_optimal_rotation_dispersion"]
            )
            abstention_penalty = cfg.abstention_uncertainty_penalty
        uncertainty[modality] = np.asarray(
            np.sqrt(
                np.square(local_uncertainty[modality])
                + alignment_ambiguity**2
                + abstention_penalty**2
            ),
            dtype=np.float64,
        )
        distribution_diagnostics[modality] = {
            **direct,
            "synchronization_accepted": assay_synchronized[modality],
            "assay_alignment_confidence": assay_confidence[modality],
        }

    reference_center = embeddings[reference].mean(axis=0)
    reference_scale = embeddings[reference].std(axis=0, ddof=0)
    reference_scale[reference_scale <= 1e-12] = 1.0
    embeddings = {
        modality: np.asarray(
            (values - reference_center[None, :]) / reference_scale[None, :],
            dtype=np.float64,
        )
        for modality, values in embeddings.items()
    }
    uncertainty = {
        modality: np.asarray(
            values / float(np.sqrt(np.mean(np.square(reference_scale)))),
            dtype=np.float64,
        )
        for modality, values in uncertainty.items()
    }

    provenance = {
        "schema_version": "soft_distributional_alignment.v2",
        "method": (
            "crossfit_uncertainty_weighted_entropic_ot_multistart_procrustes_"
            "confidence_gated_orthogonal_synchronization"
        ),
        "reference_modality": reference,
        "config": asdict(cfg),
        "distribution_diagnostics": distribution_diagnostics,
        "pairwise_distribution_diagnostics": pairwise_diagnostics,
        "synchronization": {
            "enabled": cfg.synchronize_assays,
            "accepted_assays": sorted(accepted_assays),
            "abstained_assays": sorted(set(modalities) - accepted_assays),
            "assay_synchronized": assay_synchronized,
            "assay_alignment_confidence": assay_confidence,
            "preliminary_accepted_edges": [
                _pair_key(*edge) for edge in sorted(preliminary_edges)
            ],
            "cycle_accepted_edges": [
                _pair_key(*edge) for edge in sorted(cycle_edges)
            ],
            "consensus_accepted_edges": [
                _pair_key(*edge) for edge in sorted(accepted_edges)
            ],
            "abstained_embedding_policy": (
                "finite_within_assay_local_coordinates_not_cross_assay_comparable"
            ),
            **sync_fit,
        },
        "fold_reconciliation": fold_reconciliation,
        "base_alignment_schema": base.provenance.get("schema_version", ""),
        "information_boundary": {
            "consumed": [
                "observed_background_matrix_per_assay",
                "observed_leave_focal_out_depth_per_assay",
                "assay_key",
            ],
            "not_consumed": [
                "focal_features",
                "true_labels",
                "true_latent_state",
                "truth_graph",
                "feature_identity_correspondence",
                "paired_cells",
                "cell_identity_correspondence",
                "external_data",
            ],
            "feature_columns_compared_across_assays": False,
            "cell_rows_compared_across_assays": False,
            "transport_plan_returned_as_pairing": False,
            "same_assay_rows_used_only_to_reconcile_background_feature_folds": True,
            "confidence_gates_use_background_distributions_only": True,
            "synchronization_uses_pairwise_rotations_only": True,
        },
    }
    return SoftDistributionalAlignmentResult(
        embeddings=embeddings,
        embedding_uncertainty=uncertainty,
        provenance=provenance,
        assay_synchronized=assay_synchronized,
        assay_alignment_confidence=assay_confidence,
    )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_discovery_root(path: Path | str) -> Path:
    lexical = Path(path).absolute()
    if lexical.name.casefold() != "discovery":
        raise ValueError("Input must be a world discovery/ directory")
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError("Discovery input must be a direct existing directory")
    resolved = lexical.resolve(strict=True)
    forbidden = {
        component.casefold()
        for component in resolved.parts
        if component.casefold() in _FORBIDDEN_PATH_COMPONENTS
    }
    if forbidden:
        raise ValueError(f"Refusing scorer/truth path components: {sorted(forbidden)}")
    return resolved


def _load_background_archives(
    discovery_dir: Path | str,
) -> tuple[
    Path,
    dict[str, FloatArray],
    dict[str, FloatArray],
    dict[str, NDArray[np.str_]],
    dict[str, str],
]:
    root = _safe_discovery_root(discovery_dir)
    paths: dict[str, Path] = {}
    for path in sorted(root.glob(f"{_BACKGROUND_PREFIX}*.npz")):
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("Background archives must be direct regular files")
        modality = path.name.removeprefix(_BACKGROUND_PREFIX).removesuffix(".npz")
        if not modality or modality in paths:
            raise ValueError(f"Invalid or duplicate background archive {path.name}")
        paths[modality] = path
    if "rna" not in paths or len(paths) < 2:
        raise ValueError("Discovery input needs RNA and at least one other assay")

    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    cell_ids: dict[str, NDArray[np.str_]] = {}
    hashes: dict[str, str] = {}
    seen_cells: set[str] = set()
    for modality, path in sorted(paths.items()):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"matrix", "cell_ids", "background_depth"}:
                    raise ValueError(
                        f"{path.name} may contain only matrix, cell_ids, background_depth"
                    )
                matrix = np.asarray(archive["matrix"], dtype=np.float64)
                ids = np.asarray(archive["cell_ids"], dtype=str)
                depth = np.asarray(archive["background_depth"], dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load safe background archive {path.name}") from exc
        if matrix.ndim != 2 or ids.ndim != 1 or depth.ndim != 1:
            raise ValueError(f"{path.name} has invalid array dimensions")
        if matrix.shape[0] != len(ids) or len(ids) != len(depth):
            raise ValueError(f"{path.name} row counts disagree")
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(f"{path.name} matrix must be finite and non-negative")
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"{path.name} depth must be finite and positive")
        if not np.allclose(matrix.sum(axis=1), depth, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{path.name} depth must equal the background row sum")
        if len(set(ids.tolist())) != len(ids) or not seen_cells.isdisjoint(ids.tolist()):
            raise ValueError("Cell IDs must be unique and disjoint across assays")
        seen_cells.update(ids.tolist())
        matrices[modality] = matrix
        depths[modality] = depth
        cell_ids[modality] = ids
        hashes[path.name] = _file_sha256(path)
    return root, matrices, depths, cell_ids, hashes


def freeze_soft_alignment_evidence(
    discovery_dir: Path | str,
    out_dir: Path | str,
    *,
    config: SoftAlignmentConfig | None = None,
) -> dict[str, Any]:
    """Write hash-locked discovery evidence without opening focal or scorer files."""

    cfg = config or SoftAlignmentConfig()
    root, matrices, depths, cell_ids, input_hashes = _load_background_archives(
        discovery_dir
    )
    output = Path(out_dir).absolute()
    resolved_output = output.resolve(strict=False)
    if resolved_output == root or root in resolved_output.parents:
        raise ValueError("Frozen evidence output must be outside input discovery/")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    output.mkdir(parents=True, exist_ok=True)

    result = align_soft_distributional_background_states(
        matrices,
        depths,
        reference_modality="rna",
        config=cfg,
    )
    embedding_rows: list[pd.DataFrame] = []
    for modality in sorted(result.embeddings):
        coordinates = result.embeddings[modality]
        embedding_rows.append(
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
    pd.concat(embedding_rows, ignore_index=True).to_csv(
        embedding_path, sep="\t", index=False
    )
    provenance_path = output / "alignment_provenance.json"
    provenance_path.write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    source_path = Path(__file__).resolve(strict=True)
    manifest: dict[str, Any] = {
        "schema_version": "soft_distributional_alignment.frozen_evidence.v2",
        "algorithm_source": source_path.name,
        "algorithm_source_sha256": _file_sha256(source_path),
        "config": asdict(cfg),
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            embedding_path.name: _file_sha256(embedding_path),
            provenance_path.name: _file_sha256(provenance_path),
        },
        "information_boundary": {
            "files_read": sorted(input_hashes),
            "allowed_input_pattern": "background_matrix__*.npz",
            "focal_files_read": False,
            "supplied_embeddings_read": False,
            "manifest_read": False,
            "scorer_or_truth_read": False,
            "external_data_read": False,
        },
        "frozen_before_scorer_read": True,
    }
    manifest_path = output / "frozen_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze observed-only soft distributional alignment evidence."
    )
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    freeze_soft_alignment_evidence(
        args.discovery,
        args.out,
        config=SoftAlignmentConfig(random_seed=args.random_seed),
    )


__all__ = [
    "SoftAlignmentConfig",
    "SoftDistributionalAlignmentResult",
    "align_soft_distributional_background_states",
    "freeze_soft_alignment_evidence",
]


if __name__ == "__main__":
    main()
