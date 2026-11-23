"""Rank-three background representation from conditional count likelihood.

Observed depth is the count-row sum. Profiling the Poisson cell intercept
therefore gives a multinomial objective, not a fixed-exposure Poisson model:
    sum_i [N_i logsumexp(alpha + W z_i) - X_i (alpha + W z_i)].
Only its low-rank representation is changed here. The existing logcosh ICA
contrast subsequently chooses coordinates; this is not a full generative ICA
model, a shared prior, or an identifiability theorem. Source distributions are
not constrained to match between assays.

All feature parameters are fitted in-bag. Other cells receive fixed-loading
likelihood projections, which are reconstruction/profile fits, not held-out
predictive likelihoods. Zero-depth and boundary-MLE cells get no fabricated
state. Sampling is treated as multinomial working likelihood; extra dispersion
and latent population changes are not assumed absent.

Methodological inspiration: Townes et al. (2019), doi:10.1186/s13059-019-1861-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


SCHEMA = "ca_boss_v4.count_likelihood_background_representation.v1"


class CountLikelihoodFitError(ValueError):
    """The requested likelihood fit or state coordinate is unresolved."""


@dataclass(frozen=True)
class CountLikelihoodConfig:
    n_components: int = 3
    max_iter: int = 500
    gradient_tolerance: float = 1e-6
    function_tolerance: float = 1e-10
    projection_max_iter: int = 100
    projection_decrement_tolerance: float = 1e-8
    max_line_search_steps: int = 40
    fastica_max_iter: int = 3000
    fastica_tolerance: float = 1e-5
    random_seed: int = 0
    max_dense_entries: int = 50_000_000

    def __post_init__(self) -> None:
        if self.n_components != 3:
            raise ValueError("The existing representation contract has rank three")
        if min(self.max_iter, self.projection_max_iter, self.max_line_search_steps, self.fastica_max_iter) < 1:
            raise ValueError("Iteration limits must be positive")
        if min(self.gradient_tolerance, self.function_tolerance, self.projection_decrement_tolerance, self.fastica_tolerance) <= 0:
            raise ValueError("Numerical tolerances must be positive")
        if self.random_seed < 0 or self.max_dense_entries < 1:
            raise ValueError("Seed and allocation settings are invalid")


@dataclass(frozen=True)
class FactorProjection:
    factor_scores: np.ndarray
    informative_mask: np.ndarray
    row_status: np.ndarray
    fixed_loading_covariance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CountLikelihoodFit:
    intercepts: np.ndarray
    loadings: np.ndarray
    training_factor_scores: np.ndarray
    factor_scores: np.ndarray
    scores: np.ndarray
    fixed_loading_factor_covariance: np.ndarray
    fixed_loading_score_covariance: np.ndarray
    informative_mask: np.ndarray
    row_status: np.ndarray
    inbag_indices: np.ndarray
    ica_unmixing: np.ndarray
    ica_mean: np.ndarray
    diagnostics: dict[str, Any]


def _pack(intercepts: np.ndarray, factors: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    return np.concatenate([intercepts.ravel(), factors.ravel(), loadings.ravel()])


def _unpack(parameters: np.ndarray, n: int, p: int, rank: int = 3):
    return parameters[:p], parameters[p:p + n * rank].reshape(n, rank), parameters[p + n * rank:].reshape(p, rank)


def multinomial_objective_gradient(
    parameters: np.ndarray, counts: np.ndarray
) -> tuple[float, np.ndarray]:
    """Exact count-weighted objective and analytic gradient, scaled by total."""
    n, p = counts.shape
    intercepts, factors, loadings = _unpack(parameters, n, p)
    logits = intercepts[None, :] + factors @ loadings.T
    row_total = counts.sum(axis=1)
    total = float(row_total.sum())
    if total <= 0 or not np.isfinite(logits).all():
        raise CountLikelihoodFitError("Nonfinite parameters or no observed counts")
    log_normalizer = logsumexp(logits, axis=1)
    probabilities = np.exp(logits - log_normalizer[:, None])
    residual = row_total[:, None] * probabilities - counts
    objective = float((row_total @ log_normalizer - np.sum(counts * logits)) / total)
    gradient = _pack(residual.sum(axis=0), residual @ loadings, residual.T @ factors) / total
    return objective, gradient


def canonicalize_factorization(
    intercepts: np.ndarray, factors: np.ndarray, loadings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove feature shifts and whiten factors without changing probabilities."""
    intercepts = np.asarray(intercepts, dtype=float).copy()
    factors = np.asarray(factors, dtype=float)
    loadings = np.asarray(loadings, dtype=float).copy()
    mean = factors.mean(axis=0)
    intercepts += loadings @ mean
    centered = factors - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if np.linalg.matrix_rank(centered) != 3:
        raise CountLikelihoodFitError("Fitted factor subspace is numerically rank deficient")
    canonical_factors = u[:, :3] * np.sqrt(len(factors))
    transform = (singular[:3, None] * vt[:3]) / np.sqrt(len(factors))
    canonical_loadings = loadings @ transform.T
    intercepts -= intercepts.mean()
    canonical_loadings -= canonical_loadings.mean(axis=0)
    if np.linalg.matrix_rank(canonical_loadings) != 3:
        raise CountLikelihoodFitError("Feature-contrast loadings are rank deficient")
    return intercepts, canonical_factors, canonical_loadings


def _interior_mask(counts: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Finite multinomial MLE requires observed loading means in hull interior."""
    totals = counts.sum(axis=1)
    means = np.divide(counts @ loadings, totals[:, None], out=np.zeros((len(counts), 3)), where=totals[:, None] > 0)
    try:
        hull = ConvexHull(loadings)
    except QhullError as error:
        raise CountLikelihoodFitError("Loading convex hull is numerically degenerate") from error
    distances = means @ hull.equations[:, :3].T + hull.equations[:, 3]
    tolerance = np.finfo(float).eps * max(loadings.shape) * max(1.0, float(np.abs(loadings).max()))
    return (totals > 0) & np.all(distances < -tolerance, axis=1)


def infer_factor_scores(
    counts: np.ndarray, intercepts: np.ndarray, loadings: np.ndarray, *,
    config: CountLikelihoodConfig | None = None,
) -> FactorProjection:
    """Convex fixed-loading projection with per-cell Newton stationarity.

    Each positive row's objective is divided by its own count total; this
    changes no optimum and prevents high-depth cells from masking low-depth
    projection error. The reported decrement is invariant to invertible
    factor-coordinate changes. No ridge is added to Hessians.
    """
    config = config or CountLikelihoodConfig()
    counts = np.asarray(counts, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    if counts.ndim != 2 or intercepts.shape != (counts.shape[1],) or loadings.shape != (counts.shape[1], 3):
        raise ValueError("Projection dimensions do not agree")
    if not all(np.isfinite(x).all() for x in (counts, intercepts, loadings)) or np.any(counts < 0):
        raise ValueError("Projection requires finite nonnegative counts and finite parameters")
    informative = _interior_mask(counts, loadings)
    totals = counts.sum(axis=1)
    status = np.full(len(counts), "boundary_or_unidentified_finite_mle", dtype=object)
    status[totals == 0] = "zero_context_depth"
    factors = np.full((len(counts), 3), np.nan)
    covariance = np.full((len(counts), 3, 3), np.nan)
    if not informative.any():
        return FactorProjection(factors, informative, status, covariance, {"iterations": 0, "informative_rows": 0, "unresolved_rows": len(counts)})
    fractions = counts[informative] / totals[informative, None]
    z = np.zeros((len(fractions), 3))
    trajectory: list[float] = []

    def quantities(current):
        logits = intercepts[None, :] + current @ loadings.T
        normalizer = logsumexp(logits, axis=1)
        probabilities = np.exp(logits - normalizer[:, None])
        values = normalizer - np.sum(fractions * logits, axis=1)
        expected_loadings = probabilities @ loadings
        gradient = (probabilities - fractions) @ loadings
        hessian = np.einsum("ip,pj,pk->ijk", probabilities, loadings, loadings) - np.einsum("ij,ik->ijk", expected_loadings, expected_loadings)
        return values, gradient, hessian

    for iteration in range(config.projection_max_iter + 1):
        values, gradient, hessian = quantities(z)
        try:
            direction = np.linalg.solve(hessian, gradient[..., None])[..., 0]
        except np.linalg.LinAlgError as error:
            raise CountLikelihoodFitError("Singular fixed-loading projection Hessian") from error
        decrement = np.einsum("ij,ij->i", gradient, direction)
        if not np.isfinite(decrement).all() or np.any(decrement < 0):
            raise CountLikelihoodFitError("Invalid projection Newton decrement")
        trajectory.append(float(values @ totals[informative]))
        active = decrement > config.projection_decrement_tolerance
        if not active.any():
            break
        if iteration == config.projection_max_iter:
            raise CountLikelihoodFitError("Fixed-loading projection did not reach per-cell stationarity")
        step = np.ones(len(z))
        pending = active.copy()
        accepted = z.copy()
        for _ in range(config.max_line_search_steps):
            trial = z.copy()
            trial[pending] -= step[pending, None] * direction[pending]
            trial_values, _, _ = quantities(trial)
            numeric_slack = np.finfo(float).eps * np.maximum(np.abs(values), 1.0) * 8
            valid = pending & (trial_values <= values - 1e-4 * step * decrement + numeric_slack)
            accepted[valid] = trial[valid]
            pending[valid] = False
            if not pending.any():
                break
            step[pending] *= 0.5
        if pending.any():
            raise CountLikelihoodFitError("Fixed-loading projection line search failed")
        z = accepted
    factors[informative] = z
    # This is local inverse-Fisher curvature conditional on fixed feature
    # parameters, not a calibrated posterior covariance. It omits loading
    # uncertainty and extra-multinomial dispersion.
    covariance[informative] = np.linalg.inv(hessian) / totals[informative, None, None]
    status[informative] = "finite_stationary_projection"
    return FactorProjection(factors, informative, status, covariance, {
        "iterations": iteration,
        "informative_rows": int(informative.sum()),
        "unresolved_rows": int((~informative).sum()),
        "maximum_per_count_newton_decrement_squared": float(decrement.max()),
        "maximum_per_count_absolute_gradient": float(np.abs(gradient).max()),
        "maximum_native_absolute_gradient": float(np.abs(gradient * totals[informative, None]).max()),
        "maximum_factor_norm": float(np.linalg.norm(z, axis=1).max()),
        "native_objective_trajectory": trajectory,
        "criterion": "all informative cells satisfy per-count Newton decrement tolerance",
        "covariance_interpretation": "local inverse Fisher conditional on fixed loadings; excludes extra dispersion and loading uncertainty",
    })


def fit_count_likelihood_background(
    counts: np.ndarray, depth: np.ndarray, *,
    inbag_indices: np.ndarray | None = None,
    config: CountLikelihoodConfig | None = None,
) -> CountLikelihoodFit:
    """Fit one background-feature context without references or focal inputs.

    Depth must equal this context's count-row sum. For a feature fold the
    relevant conditional total is the fold's own sum, not full-library depth.
    Zero-depth training rows contribute no likelihood and are explicitly
    omitted. An all-zero training feature makes its finite intercept
    unidentifiable, so this minimal prototype rejects that fit.
    """
    config = config or CountLikelihoodConfig()
    counts = np.asarray(counts, dtype=float)
    depth = np.asarray(depth, dtype=float).reshape(-1)
    if counts.ndim != 2 or min(counts.shape) < 4 or counts.size > config.max_dense_entries:
        raise ValueError("Counts need at least four rows/features within allocation limit")
    if not np.isfinite(counts).all() or np.any(counts < 0) or depth.shape != (len(counts),) or not np.isfinite(depth).all() or not np.allclose(depth, counts.sum(axis=1), rtol=1e-10, atol=1e-10):
        raise ValueError("Nonnegative counts and observed context-row-sum depth are required")
    supplied_indices = np.arange(len(counts), dtype=int) if inbag_indices is None else np.asarray(inbag_indices)
    if supplied_indices.ndim != 1 or not np.issubdtype(supplied_indices.dtype, np.integer) or len(supplied_indices) < 4 or np.any(supplied_indices < 0) or np.any(supplied_indices >= len(counts)):
        raise ValueError("In-bag indices must identify at least four original rows")
    indices = supplied_indices[depth[supplied_indices] > 0].astype(int)
    if len(indices) < 4:
        raise CountLikelihoodFitError("Too few informative in-bag rows")
    training = counts[indices]
    if np.any(training.sum(axis=0) == 0):
        raise CountLikelihoodFitError("All-zero in-bag feature has no finite intercept")
    n, p = training.shape
    intercepts = np.log(training.sum(axis=0) / training.sum())
    # Half counts initialize the optimizer only; no pseudocount or penalty
    # appears in the likelihood or its analytic gradient.
    initial_log = np.log((training + 0.5) / (training.sum(axis=1, keepdims=True) + 0.5 * p))
    centered = initial_log - initial_log.mean(axis=0)
    with threadpool_limits(limits=1):
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        factors = u[:, :3] * np.sqrt(n)
        loadings = vt[:3].T * (singular[:3] / np.sqrt(n))
        parameters = _pack(intercepts, factors, loadings)
        trajectory = [multinomial_objective_gradient(parameters, training)[0]]
        result = minimize(
            multinomial_objective_gradient, parameters, args=(training,), jac=True,
            method="L-BFGS-B",
            options={"maxiter": config.max_iter, "gtol": config.gradient_tolerance,
                     "ftol": config.function_tolerance, "maxls": config.max_line_search_steps},
            callback=lambda current: trajectory.append(multinomial_objective_gradient(current, training)[0]),
        )
        final_objective, final_gradient = multinomial_objective_gradient(result.x, training)
        if not result.success or not np.isfinite(final_objective) or not np.isfinite(final_gradient).all():
            raise CountLikelihoodFitError(f"Count factorization did not converge: {result.message}")
        intercepts, training_factors, loadings = canonicalize_factorization(*_unpack(result.x, n, p))
        if not _interior_mask(training, loadings).all():
            raise CountLikelihoodFitError("Training factorization contains boundary/unidentified state MLEs")
        projection = infer_factor_scores(counts, intercepts, loadings, config=config)
        if not projection.informative_mask[indices].all():
            raise CountLikelihoodFitError("Training cells lost finite fixed-loading projections")
        # Use the fixed-loading reoptimized training coordinates for the
        # retained ICA contrast, ensuring identical rows use identical states.
        model = FastICA(n_components=3, whiten="unit-variance", algorithm="parallel",
                        fun="logcosh", max_iter=config.fastica_max_iter,
                        tol=config.fastica_tolerance, random_state=config.random_seed,
                        whiten_solver="svd")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(projection.factor_scores[indices])
        if any(issubclass(item.category, ConvergenceWarning) for item in caught) or model.n_iter_ >= config.fastica_max_iter:
            raise CountLikelihoodFitError("Independent-coordinate contrast did not converge")
        scores = np.full_like(projection.factor_scores, np.nan)
        scores[projection.informative_mask] = model.transform(projection.factor_scores[projection.informative_mask])
        score_covariance = np.einsum("aj,ijk,bk->iab", model.components_, projection.fixed_loading_covariance, model.components_)
    logits = intercepts[None, :] + projection.factor_scores[indices] @ loadings.T
    probabilities = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    expected = training.sum(axis=1, keepdims=True) * probabilities
    returned_objective = float((
        training.sum(axis=1) @ logsumexp(logits, axis=1) - np.sum(training * logits)
    ) / training.sum())
    if np.any((expected == 0) & (training > 0)):
        raise CountLikelihoodFitError("Positive training count received underflowed fitted probability")
    pearson = float(np.sum(np.divide(
        np.square(training - expected), expected,
        out=np.zeros_like(expected), where=expected > 0,
    )))
    return CountLikelihoodFit(
        intercepts=intercepts, loadings=loadings,
        training_factor_scores=projection.factor_scores[indices],
        factor_scores=projection.factor_scores, scores=scores,
        fixed_loading_factor_covariance=projection.fixed_loading_covariance,
        fixed_loading_score_covariance=score_covariance,
        informative_mask=projection.informative_mask, row_status=projection.row_status,
        inbag_indices=indices, ica_unmixing=np.asarray(model.components_),
        ica_mean=np.asarray(model.mean_), diagnostics={
            "schema": SCHEMA, "likelihood": "profiled_conditional_multinomial",
            "working_likelihood_assumption": "multinomial; extra dispersion not modeled",
            "objective_interpretation": "reconstruction/profile loss, not held-out predictive likelihood",
            "rank": 3, "training_rows": len(indices),
            "training_zero_depth_rows_omitted": int(len(supplied_indices) - len(indices)),
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
            "optimizer_iterations": int(result.nit), "optimizer_evaluations": int(result.nfev),
            "initial_objective_per_count": trajectory[0], "final_objective_per_count": returned_objective,
            "joint_optimizer_endpoint_objective_per_count": final_objective,
            "objective_per_count_trajectory": trajectory,
            "maximum_normalized_absolute_gradient": float(np.abs(final_gradient).max()),
            "maximum_native_absolute_gradient": float(np.abs(final_gradient).max() * training.sum()),
            "gradient_diagnostics_scope": "joint optimizer endpoint before fixed-loading coordinate refinement",
            "optimizer_success_proves_global_optimum": False,
            "projection": projection.diagnostics,
            "ica_iterations": int(model.n_iter_), "ica_coordinate_criterion": "retained logcosh contrast",
            "pearson_discrepancy_per_entry": pearson / training.size,
            "pearson_discrepancy_is_calibrated_test": False,
            "feature_parameters_fit_inbag_only": True,
            "source_distributions_forced_equal_across_assays": False,
            "references_or_focal_inputs_used": False,
        },
    )
