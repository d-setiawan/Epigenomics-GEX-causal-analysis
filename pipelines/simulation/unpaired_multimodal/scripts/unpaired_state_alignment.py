#!/usr/bin/env python3
"""Align background-only assay states without paired cells or feature links.

The public API intentionally accepts only one observed background matrix and one
observed background-depth vector per assay.  Every assay is normalized and
embedded independently.  Microstates are matched from rotation-invariant
within-assay geometry, prevalence, and depth summaries; no columns or rows are
matched across assays.

The resulting coordinates are suitable as *background-derived state estimates*.
They are not evidence of cell correspondence and must not be interpreted as
recovering a privileged simulator state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for :func:`align_unpaired_background_states`.

    ``derange`` is a negative-control mode.  It replaces every inferred
    target-to-reference microstate assignment by a different reference state,
    choosing the minimum-cost derangement in prevalence/depth-rank space.
    """

    n_components: int = 3
    n_microstates: int = 24
    min_cells_per_microstate: int = 20
    normalization: str = "log_depth"
    geometry_weight: float = 1.0
    prevalence_weight: float = 0.20
    depth_weight: float = 0.20
    derange: bool = False
    random_seed: int = 0
    kmeans_n_init: int = 20
    max_dense_entries: int = 50_000_000


@dataclass(frozen=True)
class AlignmentResult:
    """Background-derived aligned coordinates and their audit trail."""

    embeddings: dict[str, FloatArray]
    microstate_labels: dict[str, IntArray]
    microstate_assignments: dict[str, IntArray]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CrossFittedAlignmentResult:
    """Consensus coordinates and fold disagreement from background feature splits."""

    embeddings: dict[str, FloatArray]
    embedding_uncertainty: dict[str, FloatArray]
    fold_results: tuple[AlignmentResult, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _AssayState:
    scores: FloatArray
    labels: IntArray
    centroids: FloatArray
    prevalence: FloatArray
    mean_log_depth: FloatArray
    signature: FloatArray
    pca_explained_variance_ratio: FloatArray
    matrix_sha256: str
    depth_sha256: str
    n_background_features: int
    input_was_sparse: bool
    zero_depth_cells: int


def _as_dense_float(matrix: ArrayLike, *, max_entries: int) -> tuple[FloatArray, bool]:
    input_was_sparse = sparse.issparse(matrix)
    if input_was_sparse:
        shape = matrix.shape
        if len(shape) != 2:
            raise ValueError("Every background matrix must be two-dimensional")
        if int(np.prod(shape)) > max_entries:
            raise ValueError(
                f"Sparse background matrix {shape} exceeds max_dense_entries={max_entries}"
            )
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    if dense.ndim != 2:
        raise ValueError("Every background matrix must be two-dimensional")
    output = np.asarray(dense, dtype=np.float64)
    if output.shape[0] < 4 or output.shape[1] < 2:
        raise ValueError("Each assay needs at least four cells and two background features")
    if not np.isfinite(output).all():
        raise ValueError("Background matrices may not contain NaN or infinite values")
    return output, input_was_sparse


def _as_depth(depth: ArrayLike, *, n_cells: int) -> FloatArray:
    output = np.asarray(depth, dtype=np.float64).reshape(-1)
    if output.shape != (n_cells,):
        raise ValueError(
            f"Background depth has shape {output.shape}; expected one value for {n_cells} cells"
        )
    if not np.isfinite(output).all() or np.any(output < 0):
        raise ValueError("Background depth must contain finite, non-negative values")
    return output


def _array_sha256(array: FloatArray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _normalize(matrix: FloatArray, depth: FloatArray, mode: str) -> FloatArray:
    if mode == "log_depth":
        if np.any(matrix < 0):
            raise ValueError("log_depth normalization requires non-negative observed values")
        positive = depth[depth > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        safe_depth = np.where(depth > 0, depth, 1.0)
        normalized = np.log1p(matrix * (scale / safe_depth)[:, None])
    elif mode == "standardize":
        normalized = matrix.copy()
    else:
        raise ValueError("normalization must be 'log_depth' or 'standardize'")
    # This scaler is fitted independently within each assay.  No pooled feature
    # matrix can be constructed because feature identities are deliberately absent.
    return np.asarray(StandardScaler().fit_transform(normalized), dtype=np.float64)


def _rank01(values: FloatArray) -> FloatArray:
    if values.size <= 1:
        return np.zeros_like(values)
    return np.asarray((rankdata(values, method="average") - 1.0) / (values.size - 1.0))


def _microstate_signature(
    centroids: FloatArray,
    prevalence: FloatArray,
    mean_log_depth: FloatArray,
) -> FloatArray:
    distances = cdist(centroids, centroids, metric="euclidean")
    nonzero = pdist(centroids, metric="euclidean")
    scale = float(np.median(nonzero[nonzero > 0])) if np.any(nonzero > 0) else 1.0
    geometry = np.sort(distances / max(scale, 1e-12), axis=1)
    return np.column_stack(
        [
            geometry,
            _rank01(prevalence),
            _rank01(mean_log_depth),
        ]
    )


def _fit_assay(
    matrix: FloatArray,
    depth: FloatArray,
    *,
    n_components: int,
    n_microstates: int,
    seed: int,
    config: AlignmentConfig,
    input_was_sparse: bool,
) -> _AssayState:
    normalized = _normalize(matrix, depth, config.normalization)
    # Whitening makes independently fitted PCA coordinates comparable up to an
    # orthogonal transform, exactly the ambiguity handled below by Procrustes.
    pca = PCA(n_components=n_components, whiten=True, svd_solver="full")
    scores = np.asarray(pca.fit_transform(normalized), dtype=np.float64)
    scores = np.asarray(StandardScaler().fit_transform(scores), dtype=np.float64)
    model = KMeans(
        n_clusters=n_microstates,
        random_state=seed,
        n_init=config.kmeans_n_init,
    )
    labels = np.asarray(model.fit_predict(scores), dtype=np.int64)
    centroids = np.asarray(model.cluster_centers_, dtype=np.float64)
    counts = np.bincount(labels, minlength=n_microstates).astype(np.float64)
    prevalence = counts / counts.sum()
    log_depth = np.log1p(depth)
    mean_log_depth = np.asarray(
        [float(log_depth[labels == cluster].mean()) for cluster in range(n_microstates)],
        dtype=np.float64,
    )
    return _AssayState(
        scores=scores,
        labels=labels,
        centroids=centroids,
        prevalence=prevalence,
        mean_log_depth=mean_log_depth,
        signature=_microstate_signature(centroids, prevalence, mean_log_depth),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        matrix_sha256=_array_sha256(matrix),
        depth_sha256=_array_sha256(depth),
        n_background_features=int(matrix.shape[1]),
        input_was_sparse=input_was_sparse,
        zero_depth_cells=int(np.sum(depth <= 0)),
    )


def _matching_cost(
    target: _AssayState,
    reference: _AssayState,
    config: AlignmentConfig,
) -> FloatArray:
    n_geometry = target.centroids.shape[0]
    target_geometry = target.signature[:, :n_geometry]
    reference_geometry = reference.signature[:, :n_geometry]
    geometry = cdist(target_geometry, reference_geometry, metric="sqeuclidean") / max(
        n_geometry, 1
    )
    prevalence = cdist(
        target.signature[:, [-2]], reference.signature[:, [-2]], metric="sqeuclidean"
    )
    depth = cdist(
        target.signature[:, [-1]], reference.signature[:, [-1]], metric="sqeuclidean"
    )
    return (
        config.geometry_weight * geometry
        + config.prevalence_weight * prevalence
        + config.depth_weight * depth
    )


def _property_preserving_derangement(
    reference: _AssayState,
    *,
    rng: np.random.Generator,
) -> tuple[IntArray, dict[str, float]]:
    """Return a no-fixed-point permutation nearest in prevalence/depth ranks."""
    properties = np.column_stack(
        [_rank01(reference.prevalence), _rank01(reference.mean_log_depth)]
    )
    cost = cdist(properties, properties, metric="sqeuclidean")
    # Seeded jitter makes tied solutions reproducible without making property
    # preservation depend on arbitrary cluster numbering.
    cost += rng.uniform(0.0, 1e-9, size=cost.shape)
    np.fill_diagonal(cost, float(cost.max() + 1e6))
    rows, columns = linear_sum_assignment(cost)
    permutation = np.empty(len(rows), dtype=np.int64)
    permutation[rows] = columns
    if np.any(permutation == np.arange(permutation.size)):
        raise RuntimeError("Could not construct a complete microstate derangement")
    differences = np.abs(properties - properties[permutation])
    return permutation, {
        "mean_prevalence_rank_change": float(differences[:, 0].mean()),
        "max_prevalence_rank_change": float(differences[:, 0].max()),
        "mean_depth_rank_change": float(differences[:, 1].mean()),
        "max_depth_rank_change": float(differences[:, 1].max()),
    }


def _weighted_similarity_transform(
    target_centroids: FloatArray,
    reference_centroids: FloatArray,
    target_prevalence: FloatArray,
    reference_prevalence: FloatArray,
) -> tuple[FloatArray, float, FloatArray, FloatArray, float]:
    weights = np.sqrt(target_prevalence * reference_prevalence)
    weights = weights / max(float(weights.sum()), 1e-12)
    target_mean = np.sum(target_centroids * weights[:, None], axis=0)
    reference_mean = np.sum(reference_centroids * weights[:, None], axis=0)
    target_centered = target_centroids - target_mean
    reference_centered = reference_centroids - reference_mean
    cross = (target_centered * weights[:, None]).T @ reference_centered
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_t
    rotated = target_centered @ rotation
    denominator = float(np.sum(weights[:, None] * rotated**2))
    scale = float(
        np.sum(weights[:, None] * rotated * reference_centered)
        / max(denominator, 1e-12)
    )
    fitted = scale * rotated
    residual = float(
        np.sqrt(
            np.sum(weights[:, None] * (fitted - reference_centered) ** 2)
            / max(np.sum(weights[:, None] * reference_centered**2), 1e-12)
        )
    )
    return rotation, scale, target_mean, reference_mean, residual


def align_unpaired_background_states(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    *,
    reference_modality: str | None = None,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Estimate a common state space from unpaired background observations.

    Parameters
    ----------
    background_matrices
        One cells-by-background-features matrix per assay.  Feature counts and
        identities may differ across assays; only numeric values are consumed.
    background_depths
        One observed leave-focal-out library-depth vector per assay.
    reference_modality
        Coordinate system to retain.  Defaults to the assay with the most cells,
        with the assay name used only as a deterministic tie-breaker.
    config
        Alignment and negative-control settings.

    Notes
    -----
    This signature deliberately has no parameters for labels, focal variables,
    truth, feature links, cell identifiers, or paired observations.
    """
    cfg = config or AlignmentConfig()
    modalities = sorted(str(key) for key in background_matrices)
    if len(modalities) < 2:
        raise ValueError("At least two assays are required for unpaired alignment")
    if set(modalities) != {str(key) for key in background_depths}:
        raise ValueError("background_matrices and background_depths must have identical assay keys")
    if cfg.n_components < 2:
        raise ValueError("n_components must be at least two")
    if cfg.n_microstates < 2:
        raise ValueError("n_microstates must be at least two")
    if cfg.min_cells_per_microstate < 2:
        raise ValueError("min_cells_per_microstate must be at least two")
    if min(cfg.geometry_weight, cfg.prevalence_weight, cfg.depth_weight) < 0:
        raise ValueError("Signature weights must be non-negative")
    if cfg.geometry_weight + cfg.prevalence_weight + cfg.depth_weight <= 0:
        raise ValueError("At least one signature weight must be positive")

    dense: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    sparse_flags: dict[str, bool] = {}
    for modality in modalities:
        matrix, was_sparse = _as_dense_float(
            background_matrices[modality], max_entries=cfg.max_dense_entries
        )
        dense[modality] = matrix
        depths[modality] = _as_depth(background_depths[modality], n_cells=matrix.shape[0])
        sparse_flags[modality] = was_sparse

    n_components = min(
        cfg.n_components,
        *(min(matrix.shape[0] - 1, matrix.shape[1]) for matrix in dense.values()),
    )
    if n_components < 2:
        raise ValueError("The common independently estimable PCA rank is below two")
    n_microstates = min(
        cfg.n_microstates,
        *(matrix.shape[0] // cfg.min_cells_per_microstate for matrix in dense.values()),
    )
    n_microstates = max(2, int(n_microstates))
    if n_microstates <= n_components:
        raise ValueError(
            "Need more microstates than PCA dimensions; increase cells or reduce n_components"
        )

    if reference_modality is None:
        reference = sorted(modalities, key=lambda key: (-dense[key].shape[0], key))[0]
    else:
        reference = str(reference_modality)
        if reference not in dense:
            raise ValueError(f"Unknown reference_modality: {reference}")

    seed_sequence = np.random.SeedSequence(cfg.random_seed)
    child_sequences = seed_sequence.spawn(len(modalities) + 1)
    states: dict[str, _AssayState] = {}
    for modality, child in zip(modalities, child_sequences[:-1], strict=True):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        states[modality] = _fit_assay(
            dense[modality],
            depths[modality],
            n_components=n_components,
            n_microstates=n_microstates,
            seed=child_seed,
            config=cfg,
            input_was_sparse=sparse_flags[modality],
        )

    reference_state = states[reference]
    embeddings: dict[str, FloatArray] = {reference: reference_state.scores.copy()}
    labels = {modality: state.labels.copy() for modality, state in states.items()}
    assignments: dict[str, IntArray] = {
        reference: np.arange(n_microstates, dtype=np.int64)
    }
    alignment_provenance: dict[str, Any] = {}
    derangement_rng = np.random.default_rng(child_sequences[-1])
    for modality in modalities:
        if modality == reference:
            continue
        state = states[modality]
        cost = _matching_cost(state, reference_state, cfg)
        rows, columns = linear_sum_assignment(cost)
        inferred = np.empty(n_microstates, dtype=np.int64)
        inferred[rows] = columns
        derangement: IntArray | None = None
        derangement_summary: dict[str, float] | None = None
        if cfg.derange:
            # Draw an independent property-preserving negative-control coupling
            # for every non-reference assay.  Reusing one derangement would keep
            # all chromatin assays mutually aligned and would be a weak control.
            derangement, derangement_summary = _property_preserving_derangement(
                reference_state, rng=derangement_rng
            )
        assignment = derangement[inferred] if derangement is not None else inferred
        assignments[modality] = np.asarray(assignment, dtype=np.int64)

        matched_reference = reference_state.centroids[assignment]
        matched_reference_prevalence = reference_state.prevalence[assignment]
        rotation, scale, target_mean, reference_mean, residual = (
            _weighted_similarity_transform(
                state.centroids,
                matched_reference,
                state.prevalence,
                matched_reference_prevalence,
            )
        )
        embeddings[modality] = np.asarray(
            scale * ((state.scores - target_mean) @ rotation) + reference_mean,
            dtype=np.float64,
        )
        alignment_provenance[modality] = {
            "inferred_assignment": inferred.tolist(),
            "applied_assignment": assignment.tolist(),
            "mean_hungarian_cost": float(cost[rows, columns].mean()),
            "procrustes_relative_residual": residual,
            "procrustes_scale": scale,
            "procrustes_rotation_determinant": float(np.linalg.det(rotation)),
            "deranged": bool(cfg.derange),
            "derangement_property_summary": derangement_summary,
        }

    per_assay = {
        modality: {
            "n_cells": int(states[modality].scores.shape[0]),
            "n_background_features": states[modality].n_background_features,
            "input_was_sparse": states[modality].input_was_sparse,
            "zero_depth_cells": states[modality].zero_depth_cells,
            "matrix_sha256": states[modality].matrix_sha256,
            "depth_sha256": states[modality].depth_sha256,
            "pca_explained_variance_ratio": states[
                modality
            ].pca_explained_variance_ratio.tolist(),
            "microstate_prevalence": states[modality].prevalence.tolist(),
            "microstate_mean_log_depth": states[modality].mean_log_depth.tolist(),
        }
        for modality in modalities
    }
    provenance: dict[str, Any] = {
        "schema_version": "unpaired_state_alignment.v1",
        "method": "independent_pca_invariant_microstate_hungarian_procrustes",
        "reference_modality": reference,
        "effective_n_components": int(n_components),
        "effective_n_microstates": int(n_microstates),
        "config": asdict(cfg),
        "information_boundary": {
            "consumed": [
                "observed_background_matrix_per_assay",
                "observed_leave_focal_out_depth_per_assay",
                "assay_key",
            ],
            "not_consumed": [
                "true_labels",
                "true_latent_state",
                "truth_graph",
                "focal_features",
                "feature_identity_correspondence",
                "paired_cells",
                "cell_identity_correspondence",
                "external_data",
            ],
            "feature_columns_compared_across_assays": False,
            "cell_rows_compared_across_assays": False,
        },
        "per_assay": per_assay,
        "alignments": alignment_provenance,
    }
    return AlignmentResult(
        embeddings=embeddings,
        microstate_labels=labels,
        microstate_assignments=assignments,
        provenance=provenance,
    )


def align_cross_fitted_background_states(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    *,
    reference_modality: str | None = None,
    n_feature_folds: int = 2,
    config: AlignmentConfig | None = None,
) -> CrossFittedAlignmentResult:
    """Align disjoint background feature folds and average their state estimates.

    Feature folds are formed independently inside each assay, so this routine
    never assumes corresponding columns.  Fold coordinate systems are aligned
    using repeated cells *within the reference assay only*; no row identity is
    compared between assays.
    """

    cfg = config or AlignmentConfig()
    if n_feature_folds < 2:
        raise ValueError("n_feature_folds must be at least two")
    modalities = sorted(str(key) for key in background_matrices)
    if set(modalities) != {str(key) for key in background_depths}:
        raise ValueError("background_matrices and background_depths must have identical assay keys")
    matrices: dict[str, ArrayLike] = {}
    fold_indices: dict[str, list[NDArray[np.int64]]] = {}
    split_rng = np.random.default_rng(cfg.random_seed + 8_675_309)
    for modality in modalities:
        matrix = background_matrices[modality]
        if len(matrix.shape) != 2 or matrix.shape[1] < 2 * n_feature_folds:
            raise ValueError(
                f"Assay {modality!r} needs at least {2 * n_feature_folds} background features"
            )
        matrices[modality] = matrix
        order = split_rng.permutation(matrix.shape[1])
        fold_indices[modality] = [
            np.asarray(indices, dtype=np.int64)
            for indices in np.array_split(order, n_feature_folds)
        ]
    fold_results: list[AlignmentResult] = []
    for fold in range(n_feature_folds):
        fold_matrices = {
            modality: matrices[modality][:, fold_indices[modality][fold]]
            for modality in modalities
        }
        fold_results.append(
            align_unpaired_background_states(
                fold_matrices,
                background_depths,
                reference_modality=reference_modality,
                config=AlignmentConfig(
                    **{
                        **asdict(cfg),
                        "random_seed": cfg.random_seed + 104_729 * fold,
                    }
                ),
            )
        )
    reference = str(fold_results[0].provenance["reference_modality"])
    aligned_folds: list[dict[str, FloatArray]] = [
        {key: value.copy() for key, value in fold_results[0].embeddings.items()}
    ]
    reference_target = fold_results[0].embeddings[reference]
    uniform = np.full(len(reference_target), 1.0 / len(reference_target), dtype=float)
    fold_alignment: list[dict[str, Any]] = [
        {"fold": 0, "reference_relative_residual": 0.0, "scale": 1.0}
    ]
    for fold, result in enumerate(fold_results[1:], start=1):
        reference_source = result.embeddings[reference]
        rotation, scale, source_mean, target_mean, residual = _weighted_similarity_transform(
            reference_source,
            reference_target,
            uniform,
            uniform,
        )
        aligned_folds.append(
            {
                modality: np.asarray(
                    scale * ((embedding - source_mean) @ rotation) + target_mean,
                    dtype=np.float64,
                )
                for modality, embedding in result.embeddings.items()
            }
        )
        fold_alignment.append(
            {
                "fold": fold,
                "reference_relative_residual": residual,
                "scale": scale,
                "rotation_determinant": float(np.linalg.det(rotation)),
            }
        )
    embeddings: dict[str, FloatArray] = {}
    uncertainty: dict[str, FloatArray] = {}
    for modality in modalities:
        stack = np.stack([fold[modality] for fold in aligned_folds], axis=0)
        embeddings[modality] = np.asarray(stack.mean(axis=0), dtype=np.float64)
        uncertainty[modality] = np.asarray(
            np.sqrt(np.mean(np.sum((stack - embeddings[modality][None, :, :]) ** 2, axis=2), axis=0)),
            dtype=np.float64,
        )
    # Apply one reference-derived scale to every assay.  This retains the
    # distributional alignment while producing numerically stable coordinates.
    reference_mean = embeddings[reference].mean(axis=0)
    reference_sd = embeddings[reference].std(axis=0, ddof=0)
    reference_sd[reference_sd <= 1e-12] = 1.0
    embeddings = {
        modality: np.asarray((value - reference_mean) / reference_sd, dtype=np.float64)
        for modality, value in embeddings.items()
    }
    provenance = {
        "schema_version": "unpaired_state_alignment.crossfit.v1",
        "method": "independent_feature_fold_consensus",
        "reference_modality": reference,
        "n_feature_folds": int(n_feature_folds),
        "fold_feature_counts": {
            modality: [int(len(indices)) for indices in fold_indices[modality]]
            for modality in modalities
        },
        "fold_alignment": fold_alignment,
        "fold_provenance": [result.provenance for result in fold_results],
        "information_boundary": {
            "feature_splits_are_within_assay": True,
            "feature_columns_compared_across_assays": False,
            "reference_cells_align_folds_within_assay": True,
            "cell_rows_compared_across_assays": False,
            "truth_or_pairing_inputs": False,
        },
    }
    return CrossFittedAlignmentResult(
        embeddings=embeddings,
        embedding_uncertainty=uncertainty,
        fold_results=tuple(fold_results),
        provenance=provenance,
    )


__all__ = [
    "AlignmentConfig",
    "AlignmentResult",
    "CrossFittedAlignmentResult",
    "align_unpaired_background_states",
    "align_cross_fitted_background_states",
]
