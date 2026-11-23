"""Unique reporter effects from joint depth-adjusted regression (v4.64).

Marginal correlation after depth projection can spread a single reporter
effect across conditionally correlated ICA axes.  Fit all three axes jointly
after residualizing on intercept and log1p(observed background depth), then
attribute each axis its semipartial explained-variance increment:

    energy[f,j] = beta[f,j]**2 * Var(Z_j | Z_-j, depth) / Var(Y_f | depth).

The Frisch--Waugh--Lovell identity makes this equal to the increase in R-squared
when adding axis j to the other axes and depth, using depth-residual outcome
variance as the denominator.  Scores are observational unique linear effects,
not causal coefficients or posterior probabilities.  The interpretation needs
an adequate linear conditional-mean model and identifiable, sufficiently
accurate background axes.  Count noise, rotations, and unequal assay noise
remain limitations.  Unique variance and hence amplitude also depend on the
within-assay conditional covariance; cross-assay values need not be equal.

No ridge or condition-number selection cutoff is applied.  Numerical rank
deficiency rejects attribution; full-rank near-collinearity is reported with
its condition number and per-axis unique variance for explicit uncertainty.
Primary reference: Lovell (1963), doi:10.1080/01621459.1963.10480682.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import soft_reporter_representation_ensemble_v4 as soft


SCHEMA = "ca_boss_v4.joint_reporter_regression_evidence.v1"


@dataclass(frozen=True)
class JointReporterAxisEvidence(soft.ReporterAxisEvidence):
    """Compatible soft evidence with full multivariable attribution diagnostics.

    ``partial_correlations`` is the inherited compatibility field; here it
    contains signed *semipartial* correlations (squares equal unique ΔR²), not
    the marginal depth-adjusted correlations used by v4.63.
    """

    coefficients: np.ndarray
    unique_axis_variance: np.ndarray
    residual_score_rank: int
    residual_score_condition_number: float
    identified: bool
    unresolved_reason: str


def _package(
    signed_effects: np.ndarray, coefficients: np.ndarray,
    unique_variance: np.ndarray, *, rank: int, condition_number: float,
    identified: bool, unresolved_reason: str,
) -> JointReporterAxisEvidence:
    energy = np.square(signed_effects)
    per_strength = energy.sum(axis=1)
    per_weights = np.divide(
        energy, per_strength[:, None], out=np.zeros_like(energy),
        where=per_strength[:, None] > 0,
    )
    mean_energy = energy.mean(axis=0)
    total = float(mean_energy.sum())
    usable = bool(identified and total > 0)
    weights = mean_energy / total if usable else np.zeros(3)
    nonzero = weights > 0
    information = float(1.0 + np.sum(weights[nonzero] * np.log(weights[nonzero])) / np.log(3.0)) if usable else 0.0
    ordered = np.sort(weights)
    return JointReporterAxisEvidence(
        partial_correlations=signed_effects,
        per_reporter_weights=per_weights,
        mean_squared_correlations=mean_energy,
        weights=weights,
        total_strength=total,
        information=float(np.clip(information, 0.0, 1.0)),
        weight_margin=float(ordered[-1] - ordered[-2]),
        minimum_reporter_best_squared_correlation=float(energy.max(axis=1).min()),
        nonzero_reporters=int(np.sum(per_strength > 0)),
        usable=usable,
        coefficients=coefficients,
        unique_axis_variance=unique_variance,
        residual_score_rank=rank,
        residual_score_condition_number=condition_number,
        identified=identified,
        unresolved_reason=unresolved_reason,
    )


def reporter_axis_evidence(
    focal: np.ndarray, scores: np.ndarray, observed_depth: np.ndarray
) -> JointReporterAxisEvidence:
    """Estimate unique linear reporter effects on one assay's rank-three state.

    Reporter alternatives contribute equal weight through their raw unique
    explained variance, before axis normalization.  A constant or numerically
    depth-explained reporter contributes zero.  The input arrays must describe
    the same assay and identical rows.  This function consumes no cross-assay
    observations, feature roles, simulator states, or truth.
    """
    focal = np.asarray(focal, dtype=float)
    scores = np.asarray(scores, dtype=float)
    depth = np.asarray(observed_depth, dtype=float).reshape(-1)
    if focal.ndim != 2 or focal.shape[1] < 1:
        raise ValueError("At least one two-dimensional reporter column is required")
    if scores.shape != (len(focal), 3) or depth.shape != (len(focal),):
        raise ValueError("Reporter, depth and rank-three score rows must agree")
    if len(focal) < 8 or not all(np.isfinite(values).all() for values in (focal, scores, depth)):
        raise ValueError("At least eight finite same-assay observations are required")
    if np.any(focal < 0) or np.any(depth <= 0):
        raise ValueError("Reporter counts must be nonnegative and depth positive")
    design = np.column_stack([np.ones(len(depth)), np.log1p(depth)])
    logged = np.log1p(focal)
    y = logged - design @ np.linalg.lstsq(design, logged, rcond=None)[0]
    x = scores - design @ np.linalg.lstsq(design, scores, rcond=None)[0]
    coefficient_columns, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    # If all axes are explained by depth, projection roundoff may have rank
    # three relative to its own tiny scale. Audit dimension in the original
    # augmented design as well. Column normalization removes arbitrary units;
    # matrix_rank then uses NumPy's machine-resolution threshold, not a
    # scientific collinearity cutoff.
    augmented = np.column_stack([design, scores])
    column_norms = np.linalg.norm(augmented, axis=0)
    augmented_scaled = augmented / np.where(column_norms > 0, column_norms, 1.0)
    augmented_rank = int(np.linalg.matrix_rank(augmented_scaled))
    nuisance_rank = int(np.linalg.matrix_rank(augmented_scaled[:, :design.shape[1]]))
    rank = min(int(rank), max(0, augmented_rank - nuisance_rank))
    condition = float(singular[0] / singular[-1]) if len(singular) == 3 and singular[-1] > 0 else float("inf")
    coefficients = coefficient_columns.T
    if rank != 3:
        return _package(
            np.zeros((focal.shape[1], 3)), np.zeros((focal.shape[1], 3)),
            np.zeros(3), rank=int(rank), condition_number=float("inf"),
            identified=False, unresolved_reason="depth_residual_score_design_rank_deficient",
        )

    unique_variance = np.empty(3, dtype=float)
    for component in range(3):
        others = x[:, [other for other in range(3) if other != component]]
        unique = x[:, component] - others @ np.linalg.lstsq(others, x[:, component], rcond=None)[0]
        unique_variance[component] = float(np.mean(np.square(unique)))
    outcome_variance = np.mean(np.square(y), axis=0)
    # The only zero-effect guard is floating-point resolution of the original
    # response: an exactly depth-explained reporter otherwise leaves tiny
    # projection roundoff that can be spuriously normalized into a full axis.
    residual_tolerance = np.finfo(float).eps * max(logged.shape) * np.maximum(np.linalg.norm(logged, axis=0), 1.0)
    resolved_reporters = np.linalg.norm(y, axis=0) > residual_tolerance
    signed_effects = np.zeros_like(coefficients)
    signed_effects[resolved_reporters] = coefficients[resolved_reporters] * np.sqrt(
        unique_variance[None, :] / outcome_variance[resolved_reporters, None]
    )
    if not np.isfinite(signed_effects).all() or not np.isfinite(unique_variance).all():
        return _package(
            np.zeros_like(coefficients), np.zeros_like(coefficients), np.zeros(3),
            rank=int(rank), condition_number=condition, identified=False,
            unresolved_reason="nonfinite_joint_regression_attribution",
        )
    return _package(
        signed_effects, coefficients, unique_variance, rank=int(rank),
        condition_number=condition, identified=True, unresolved_reason="",
    )
