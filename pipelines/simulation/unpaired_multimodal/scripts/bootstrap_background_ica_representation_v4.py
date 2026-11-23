"""Refit assay-local background ICA under cell and initialization uncertainty.

The caller supplies a single assay's background counts and same-cell reference
scores.  References orient fitted axes only; they never enter normalization or
ICA estimation.  A shared within-assay bootstrap sample is used across full and
feature-fold contexts.  Transforming all original rows is a diagnostic convenience,
not an independent validation split: the cached reference fit has seen those rows.

Inspired by Himberg, Hyvarinen and Esposito (2004), doi:10.1016/j.neuroimage.2004.03.027.
No cross-assay correspondence or focal feature is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Mapping
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import unpaired_mdcr_ks_baseline_v4 as mdcr


FloatArray = NDArray[np.float64]
SCHEMA = "ca_boss_v4.bootstrap_background_ica_representation.v1"
MODES = ("initialization_only", "cell_resample_only", "both")


class RepresentationFitError(ValueError):
    """A replicate cannot safely contribute representation evidence."""


@dataclass(frozen=True)
class ContextFitDiagnostics:
    permutation: tuple[int, ...]
    signs: tuple[int, ...]
    matched_abs_correlations: tuple[float, ...]
    relative_assignment_gap: float
    alignment_tied: bool
    convergence: bool
    n_iter: int
    fit_seed: int
    alignment_cells: int
    alignment_scope: str


@dataclass(frozen=True)
class AssayBootstrapResult:
    scores: dict[str, FloatArray]
    inbag_indices: NDArray[np.int64]
    oob_mask: NDArray[np.bool_]
    contexts: dict[str, ContextFitDiagnostics]
    normalization_mean: FloatArray
    normalization_scale: FloatArray
    library_scale: float
    feature_indices: dict[str, NDArray[np.int64]]
    provenance: dict[str, Any]


def _resample_indices(
    n_cells: int, *, modality: str, resample_seed: int, mode: str
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    if mode == "initialization_only":
        return np.arange(n_cells, dtype=np.int64), np.zeros(n_cells, dtype=bool)
    rng = np.random.default_rng(mdcr._stable_seed(resample_seed, modality))
    indices = np.asarray(rng.integers(0, n_cells, size=n_cells), dtype=np.int64)
    oob = np.ones(n_cells, dtype=bool)
    oob[np.unique(indices)] = False
    return indices, oob


def _alignment(
    scores: FloatArray, reference: FloatArray, alignment_mask: NDArray[np.bool_]
) -> tuple[FloatArray, tuple[int, ...], tuple[int, ...], tuple[float, ...], float]:
    fitted_subset = scores[alignment_mask]
    reference_subset = reference[alignment_mask]
    if np.any(fitted_subset.std(axis=0) <= 1e-12) or np.any(
        reference_subset.std(axis=0) <= 1e-12
    ):
        raise RepresentationFitError("Degenerate axis on alignment cells")
    rank = scores.shape[1]
    correlations = np.corrcoef(reference_subset.T, fitted_subset.T)[:rank, rank:]
    if not np.isfinite(correlations).all():
        raise RepresentationFitError("Nonfinite component-alignment correlation")
    ranked = sorted(
        (
            (float(np.mean(np.abs(correlations[np.arange(rank), order]))), order)
            for order in permutations(range(rank))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best, order = ranked[0]
    matched = correlations[np.arange(rank), order]
    signs = tuple(1 if correlation >= 0 else -1 for correlation in matched)
    gap = float((best - ranked[1][0]) / max(best, 1e-12))
    return (
        scores[:, order] * np.asarray(signs)[None, :],
        tuple(order),
        signs,
        tuple(float(abs(value)) for value in matched),
        gap,
    )


def bootstrap_fit_assay_contexts(
    counts: ArrayLike,
    depth: ArrayLike,
    reference_scores: Mapping[str, ArrayLike],
    *,
    modality: str,
    reference_seed: int,
    resample_seed: int,
    initialization_seed: int,
    mode: str = "both",
    config: mdcr.UnpairedMDCRConfig | None = None,
) -> AssayBootstrapResult:
    """Return aligned refitted scores, preserving the input assay's row order.

    Context names are ``full`` and ``fold0`` through ``fold{n_feature_folds-1}``.
    ``reference_seed`` reproduces the original background-feature split.  The
    cell-resampling-only mode also reproduces original ICA initialization seeds.
    All normalization parameters, including output score centering/scaling, are
    estimated on the in-bag sample.  Out-of-bag rows only align fitted components
    to the caller's references; initialization-only mode aligns on all rows.

    The caller must keep counts and every reference in identical cell order and
    must restrict counts to predeclared held-out background features.  The API
    cannot infer feature provenance from an unlabeled matrix.

    Nonconvergence or degeneracy rejects the whole assay replicate.  Weak or
    tied alignment is returned explicitly so it can remain unresolved downstream.
    """
    config = config or mdcr.UnpairedMDCRConfig()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not isinstance(modality, str) or not modality:
        raise ValueError("modality must be a nonempty assay identifier")
    for name, seed in (
        ("reference_seed", reference_seed),
        ("resample_seed", resample_seed),
        ("initialization_seed", initialization_seed),
    ):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    matrix = mdcr._as_matrix(counts, max_entries=config.max_dense_entries)
    observed_depth = mdcr._as_depth(depth, matrix=matrix)
    context_names = ("full",) + tuple(
        f"fold{index}" for index in range(config.n_feature_folds)
    )
    if set(reference_scores) != set(context_names):
        raise ValueError(f"reference_scores must contain exactly {context_names}")
    references = {name: np.asarray(reference_scores[name], dtype=float) for name in context_names}
    for name, reference in references.items():
        if reference.shape != (len(matrix), config.n_components) or not np.isfinite(reference).all():
            raise ValueError(f"Invalid same-assay reference scores for {name}")

    inbag, oob = _resample_indices(
        len(matrix), modality=modality, resample_seed=resample_seed, mode=mode
    )
    alignment_mask = np.ones(len(matrix), dtype=bool) if mode == "initialization_only" else oob
    if int(alignment_mask.sum()) < 8:
        raise RepresentationFitError("At least eight out-of-bag alignment cells are required")
    if np.unique(inbag).size <= config.n_components:
        raise RepresentationFitError("Too few distinct in-bag cells for ICA")

    library_scale = float(np.median(observed_depth[inbag]))
    logged = np.log1p(matrix * (library_scale / observed_depth)[:, None])
    normalizer = StandardScaler().fit(logged[inbag])
    normalized = np.asarray(normalizer.transform(logged), dtype=float)
    fold_rng = np.random.default_rng(mdcr._stable_seed(reference_seed, f"{modality}::feature_folds"))
    feature_folds = np.array_split(fold_rng.permutation(matrix.shape[1]), config.n_feature_folds)
    feature_indices = {"full": np.arange(matrix.shape[1], dtype=np.int64)}
    feature_indices.update({f"fold{index}": np.asarray(indices, dtype=np.int64) for index, indices in enumerate(feature_folds)})
    if any(len(indices) < config.n_components for indices in feature_indices.values()):
        raise ValueError("Each background-feature context needs at least rank-three features")

    scores: dict[str, FloatArray] = {}
    diagnostics: dict[str, ContextFitDiagnostics] = {}
    with threadpool_limits(limits=1):
        for name in context_names:
            token = modality if name == "full" else f"{modality}__feature_{name}"
            base_seed = reference_seed if mode == "cell_resample_only" else initialization_seed
            fit_seed = mdcr._stable_seed(base_seed, token)
            model = FastICA(
                n_components=config.n_components,
                whiten="unit-variance",
                algorithm="parallel",
                fun="logcosh",
                max_iter=config.fastica_max_iter,
                tol=config.fastica_tolerance,
                random_state=fit_seed,
                whiten_solver="svd",
            )
            context = normalized[:, feature_indices[name]]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    train_scores = np.asarray(model.fit_transform(context[inbag]), dtype=float)
                n_iter = int(getattr(model, "n_iter_", config.fastica_max_iter))
                if any(issubclass(item.category, ConvergenceWarning) for item in caught) or n_iter >= config.fastica_max_iter:
                    raise RepresentationFitError(f"FastICA did not converge in {name} ({n_iter} iterations)")
                center = train_scores.mean(axis=0)
                scale = train_scores.std(axis=0, ddof=0)
                if not np.isfinite(train_scores).all() or np.any(scale <= 1e-12):
                    raise RepresentationFitError(f"Degenerate fitted ICA scores in {name}")
                all_scores = (np.asarray(model.transform(context), dtype=float) - center) / scale
            except RepresentationFitError:
                raise
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                raise RepresentationFitError(f"ICA fit failed in {name}: {error}") from error
            if not np.isfinite(all_scores).all():
                raise RepresentationFitError(f"Nonfinite transformed ICA scores in {name}")
            aligned, order, signs, correlations, gap = _alignment(all_scores, references[name], alignment_mask)
            scores[name] = aligned
            diagnostics[name] = ContextFitDiagnostics(
                permutation=order,
                signs=signs,
                matched_abs_correlations=correlations,
                relative_assignment_gap=gap,
                alignment_tied=bool(gap <= 1e-12),
                convergence=True,
                n_iter=n_iter,
                fit_seed=fit_seed,
                alignment_cells=int(alignment_mask.sum()),
                alignment_scope="all_original_rows" if mode == "initialization_only" else "out_of_bag_rows",
            )

    return AssayBootstrapResult(
        scores=scores,
        inbag_indices=inbag,
        oob_mask=oob,
        contexts=diagnostics,
        normalization_mean=np.asarray(normalizer.mean_, dtype=float),
        normalization_scale=np.asarray(normalizer.scale_, dtype=float),
        library_scale=library_scale,
        feature_indices=feature_indices,
        provenance={
            "schema": SCHEMA,
            "mode": mode,
            "modality": modality,
            "reference_seed": int(reference_seed),
            "resample_seed": int(resample_seed),
            "initialization_seed": int(initialization_seed),
            "normalization_fit_on_inbag_only": True,
            "same_resample_across_feature_contexts": True,
            "same_assay_cell_order_required": True,
            "reference_used_only_for_axis_alignment": True,
            "reference_estimator_independent_of_alignment_cells": False,
            "all_original_rows_transformed": True,
            "cross_assay_cell_correspondence_used": False,
            "focal_features_accepted_as_separate_input": False,
            "nonconvergence_policy": "reject_whole_assay_replicate",
        },
    )
