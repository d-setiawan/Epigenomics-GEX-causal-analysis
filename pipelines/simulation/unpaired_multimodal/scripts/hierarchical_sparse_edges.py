#!/usr/bin/env python3
"""Hierarchical sparse within-assay edge evidence for CA-BOSS v4.

This module estimates *undirected* residual dependence between features observed
in the same assay cells.  It is deliberately independent of the rest of the
CA-BOSS implementation so that the statistical evidence layer can be tested and
changed without invoking Tetrad or reading simulation truth.

For every observed count/detection feature, a cross-fitted ridge-logistic model
estimates detection probability from:

* observed integration coordinates (background-derived state proxies),
* a K-means microstate indicator, and
* log1p leave-locus-out background library depth.

Counts are converted to binary detections using ``count > 0``.  For a feature
pair ``(X, Y)`` and microstate ``k``, conditional independence implies the
Bernoulli interaction score

    S_k = sum_i (X_i - p_Xi) (Y_i - p_Yi)

has information

    I_k = sum_i p_Xi(1-p_Xi) p_Yi(1-p_Yi).

The one-step residual log-odds estimate is ``S_k / I_k``.  Its uncertainty uses
the larger of model-based Bernoulli information and a sandwich score variance.
Local estimates are partially pooled with a normal-normal empirical-Bayes
random-effects approximation.  Those microstate estimates remain useful
heterogeneity diagnostics, but they discard information when detections are
very sparse.  The primary global evidence therefore also fits the symmetric
Bernoulli pseudo-likelihood

    logit P(Y=1 | X) = logit(p_Y) + theta X
    logit P(X=1 | Y) = logit(p_X) + theta Y,

where the cross-fitted marginal probabilities are fixed offsets and ``theta``
is one shared, undirected residual log-odds parameter.  A declared weak Normal
prior makes the estimate finite under separation.  Its posterior mean, scale,
and positive-effect probabilities use the Laplace approximation at the MAP.
Selection can use the microstate evidence, global evidence, or their
conservative intersection; every rule is positive-only and uncertainty-aware.
The intersection is the default because the global all-cell likelihood is a
powerful sparse-edge rescue but can mistake stable residual confounding for a
direct edge.  Its component probabilities are always returned so a separately
calibrated selector can opt into that rescue without refitting nuisance models.

The core API is modality agnostic and accepts only aligned observed rows.  The
table convenience API is intentionally leakage-safe: it accepts one within-
assay table, background analysis embeddings, and leave-locus-out depth.  It
never opens truth, pairing, semantics, DAG, or external-support files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


EDGE_COLUMNS = [
    "node1",
    "node2",
    "n_cells",
    "n_microstates_used",
    "total_information",
    "residual_log_odds",
    "residual_log_odds_sd",
    "ci95_lower",
    "ci95_upper",
    "posterior_probability_positive",
    "posterior_probability_negative",
    "posterior_probability_residual_dependence",
    "local_false_sign_rate",
    "approx_p_value",
    "between_microstate_sd",
    "heterogeneity_q",
    "unconditional_phi",
    "detection_rate_node1",
    "detection_rate_node2",
    "microstate_residual_log_odds",
    "microstate_residual_log_odds_sd",
    "microstate_posterior_probability_positive",
    "microstate_posterior_probability_negative",
    "microstate_posterior_probability_residual_dependence",
    "global_pseudolikelihood_log_odds",
    "global_pseudolikelihood_log_odds_sd",
    "global_pseudolikelihood_ci95_lower",
    "global_pseudolikelihood_ci95_upper",
    "global_posterior_probability_positive",
    "global_posterior_probability_gt_practical",
    "global_posterior_probability_negative",
    "global_posterior_probability_lt_negative_practical",
    "global_posterior_probability_residual_dependence",
    "global_local_false_sign_rate",
    "global_pseudolikelihood_information",
    "global_interaction_prior_sd",
    "global_conditional_weight",
    "global_pseudolikelihood_converged",
    "global_pseudolikelihood_iterations",
    "positive_evidence_rule",
    "selection_probability_positive",
    "selected",
    "interpretation",
]

LOCAL_COLUMNS = [
    "node1",
    "node2",
    "microstate",
    "n_cells",
    "detection_count_node1",
    "detection_count_node2",
    "information",
    "raw_residual_log_odds",
    "raw_sd",
    "shrunk_residual_log_odds",
    "shrunk_sd",
    "shrinkage_weight",
]

FORBIDDEN_PREFIXES = (
    "true_",
    "truth_",
    "latent_",
    "oracle_",
    "technical_",
    "counterfactual_",
    "external_",
    "pairing_",
)

FORBIDDEN_TOKENS = (
    "ground_truth",
    "truth__",
    "__truth",
    "true_z",
    "true_state",
    "true_cell_type",
    "latent__",
    "oracle__",
    "counterfactual__",
    "pairing__",
    "external_support",
    "technical_depth",
)

NON_FEATURE_COLUMNS = {
    "cell",
    "cell_id",
    "sample_id",
    "row_id",
    "modality",
    "retained_target",
    "observed_library_size",
    "observed_background_library_size",
}


@dataclass(frozen=True)
class HierarchicalSparseConfig:
    """Configuration for the hierarchical Bernoulli evidence approximation."""

    n_microstates: int = 20
    min_microstate_cells: int = 20
    min_peak_events: int = 5
    crossfit_folds: int = 5
    nuisance_c: float = 1.0
    nuisance_max_iter: int = 1_000
    probability_clip: float = 1e-3
    min_local_information: float = 0.20
    min_abs_residual_log_odds: float = 0.20
    evidence_threshold: float = 0.95
    global_interaction_prior_sd: float = 2.50
    global_conditional_weight: float = 0.50
    positive_evidence_rule: str = "intersection"
    global_root_tolerance: float = 1e-10
    global_max_root_iterations: int = 200
    random_state: int = 0

    def validate(self) -> None:
        if self.n_microstates < 1:
            raise ValueError("n_microstates must be >= 1")
        if self.min_microstate_cells < 4:
            raise ValueError("min_microstate_cells must be >= 4")
        if self.min_peak_events < 1:
            raise ValueError("min_peak_events must be >= 1")
        if self.crossfit_folds < 2:
            raise ValueError("crossfit_folds must be >= 2")
        if self.nuisance_c <= 0:
            raise ValueError("nuisance_c must be > 0")
        if not 0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must be in (0, 0.5)")
        if self.min_local_information <= 0:
            raise ValueError("min_local_information must be > 0")
        if self.min_abs_residual_log_odds < 0:
            raise ValueError("min_abs_residual_log_odds must be >= 0")
        if not 0.5 < self.evidence_threshold < 1:
            raise ValueError("evidence_threshold must be in (0.5, 1)")
        if self.global_interaction_prior_sd <= 0:
            raise ValueError("global_interaction_prior_sd must be > 0")
        if self.global_conditional_weight <= 0:
            raise ValueError("global_conditional_weight must be > 0")
        if self.positive_evidence_rule not in {
            "microstate",
            "global",
            "intersection",
        }:
            raise ValueError(
                "positive_evidence_rule must be one of: microstate, global, intersection"
            )
        if self.global_root_tolerance <= 0:
            raise ValueError("global_root_tolerance must be > 0")
        if self.global_max_root_iterations < 10:
            raise ValueError("global_max_root_iterations must be >= 10")


@dataclass
class HierarchicalSparseResult:
    """All outputs from one within-assay evidence fit."""

    edge_evidence: pd.DataFrame
    microstate_evidence: pd.DataFrame
    peak_diagnostics: pd.DataFrame
    assignments: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def selected_edges(self) -> pd.DataFrame:
        if self.edge_evidence.empty:
            return self.edge_evidence.copy()
        return self.edge_evidence.loc[self.edge_evidence["selected"].astype(bool)].copy()


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    """Return a deterministic undirected pair key."""

    return (left, right) if left < right else (right, left)


def _standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    scale = numeric.std(axis=0, ddof=0)
    live = [column for column in numeric.columns if float(scale[column]) > 1e-12]
    dropped = sorted(set(numeric.columns) - set(live))
    if not live:
        return pd.DataFrame(index=numeric.index), dropped
    centered = numeric[live] - numeric[live].mean(axis=0)
    return centered.divide(scale[live], axis=1), dropped


def assign_microstates(
    integration_state: pd.DataFrame,
    *,
    n_microstates: int,
    min_microstate_cells: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Cluster observed integration state without using feature measurements."""

    state, dropped = _standardize(integration_state.reset_index(drop=True))
    n_cells = len(state)
    max_by_size = max(1, n_cells // int(min_microstate_cells))
    actual = min(int(n_microstates), max_by_size)
    if actual <= 1 or state.shape[1] == 0:
        labels = np.zeros(n_cells, dtype=int)
        actual = 1
    else:
        labels = KMeans(
            n_clusters=actual,
            random_state=int(random_state),
            n_init=10,
        ).fit_predict(state.to_numpy(dtype=float))
        # Relabel by centroid order so diagnostics are stable across sklearn versions.
        centers = []
        for label in range(actual):
            center = state.loc[labels == label].mean(axis=0).to_numpy(dtype=float)
            centers.append((tuple(np.round(center, 12)), label))
        mapping = {old: new for new, (_, old) in enumerate(sorted(centers))}
        labels = np.asarray([mapping[int(value)] for value in labels], dtype=int)
    sizes = pd.Series(labels).value_counts().sort_index()
    return labels, state, {
        "requested_microstates": int(n_microstates),
        "actual_microstates": int(actual),
        "microstate_sizes": {str(int(k)): int(v) for k, v in sizes.items()},
        "state_columns": list(state.columns),
        "dropped_constant_state_columns": dropped,
    }


def _nuisance_design(
    standardized_state: pd.DataFrame,
    background_depth: np.ndarray,
    microstates: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    log_depth = np.log1p(np.asarray(background_depth, dtype=float))
    depth_sd = float(log_depth.std(ddof=0))
    depth_scaled = (
        (log_depth - float(log_depth.mean())) / depth_sd
        if depth_sd > 1e-12
        else np.zeros_like(log_depth)
    )
    parts = [standardized_state.to_numpy(dtype=float), depth_scaled[:, None]]
    names = [f"state::{column}" for column in standardized_state.columns] + [
        "log1p_background_depth"
    ]
    unique_states = sorted(int(value) for value in np.unique(microstates))
    # The first microstate is the reference category. Continuous state remains in
    # the model, while these indicators absorb coarse nonlinear state structure.
    if len(unique_states) > 1:
        one_hot = np.column_stack(
            [(microstates == value).astype(float) for value in unique_states[1:]]
        )
        parts.append(one_hot)
        names.extend([f"microstate::{value}" for value in unique_states[1:]])
    return np.column_stack(parts).astype(float), names


def _smoothed_rate(values: np.ndarray) -> float:
    return float((float(values.sum()) + 0.5) / (len(values) + 1.0))


def cross_fitted_detection_probabilities(
    detection: np.ndarray,
    design: np.ndarray,
    *,
    config: HierarchicalSparseConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return out-of-fold nuisance probabilities for one binary feature."""

    y = np.asarray(detection, dtype=int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("detection must be binary")
    counts = np.bincount(y, minlength=2)
    min_class = int(counts.min())
    if min_class < 2:
        probability = np.full(len(y), _smoothed_rate(y), dtype=float)
        probability = np.clip(
            probability,
            config.probability_clip,
            1.0 - config.probability_clip,
        )
        return probability, {
            "crossfit_folds_used": 0,
            "n_fallback_folds": 1,
            "fit_mode": "smoothed_intercept",
        }

    n_splits = min(int(config.crossfit_folds), min_class)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(seed),
    )
    probabilities = np.full(len(y), np.nan, dtype=float)
    fallback_folds = 0
    for train, test in splitter.split(design, y):
        if np.unique(y[train]).size < 2:
            probabilities[test] = _smoothed_rate(y[train])
            fallback_folds += 1
            continue
        try:
            model = LogisticRegression(
                C=float(config.nuisance_c),
                fit_intercept=True,
                solver="lbfgs",
                max_iter=int(config.nuisance_max_iter),
                random_state=int(seed),
            )
            model.fit(design[train], y[train])
            probabilities[test] = model.predict_proba(design[test])[:, 1]
        except (ValueError, FloatingPointError):
            probabilities[test] = _smoothed_rate(y[train])
            fallback_folds += 1
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Cross-fitting left non-finite detection probabilities")
    probabilities = np.clip(
        probabilities,
        config.probability_clip,
        1.0 - config.probability_clip,
    )
    return probabilities, {
        "crossfit_folds_used": int(n_splits),
        "n_fallback_folds": int(fallback_folds),
        "fit_mode": "cross_fitted_ridge_logistic",
    }


def _dersimonian_laird(
    effects: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """Normal-normal random-effects meta-analysis with conservative uncertainty."""

    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if effects.size == 0:
        raise ValueError("At least one local effect is required")
    if effects.size == 1:
        return (
            float(effects[0]),
            float(np.sqrt(variances[0])),
            0.0,
            0.0,
            np.asarray([1.0 / variances[0]]),
        )
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    q_value = float(np.sum(fixed_weights * np.square(effects - fixed_mean)))
    c_value = float(
        np.sum(fixed_weights)
        - np.sum(np.square(fixed_weights)) / np.sum(fixed_weights)
    )
    tau2 = max(0.0, (q_value - (effects.size - 1)) / max(c_value, 1e-12))
    random_weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(random_weights * effects) / np.sum(random_weights))
    # Hartung-like scale inflation prevents a heterogeneous set of microstates
    # from producing an overconfident edge merely because K is large.
    residual_q = float(np.sum(random_weights * np.square(effects - pooled)))
    scale = max(1.0, residual_q / max(effects.size - 1, 1))
    pooled_sd = float(np.sqrt(scale / np.sum(random_weights)))
    return pooled, pooled_sd, float(tau2), q_value, random_weights


def empirical_bayes_pool(
    local: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Pool local estimates and append their posterior shrinkage summaries.

    ``local`` must contain ``raw_residual_log_odds`` and ``raw_variance``.
    This public helper makes the partial-pooling behavior directly unit-testable.
    """

    required = {"raw_residual_log_odds", "raw_variance"}
    missing = required - set(local.columns)
    if missing:
        raise ValueError(f"Missing local-pooling columns: {sorted(missing)}")
    effects = local["raw_residual_log_odds"].to_numpy(dtype=float)
    variances = local["raw_variance"].to_numpy(dtype=float)
    if not np.isfinite(effects).all() or not np.isfinite(variances).all():
        raise ValueError("Local effects and variances must be finite")
    if np.any(variances <= 0):
        raise ValueError("Local variances must be positive")
    pooled, pooled_sd, tau2, q_value, weights = _dersimonian_laird(
        effects,
        variances,
    )
    out = local.copy()
    shrunk_means: list[float] = []
    shrunk_sds: list[float] = []
    shrinkage_weights: list[float] = []
    for effect, variance in zip(effects, variances):
        if tau2 <= 1e-14:
            shrinkage = 0.0
            mean = pooled
            sd = pooled_sd
        else:
            shrinkage = float(tau2 / (tau2 + variance))
            mean = float(pooled + shrinkage * (effect - pooled))
            conditional_variance = 1.0 / (1.0 / variance + 1.0 / tau2)
            sd = float(
                np.sqrt(conditional_variance + np.square(1.0 - shrinkage) * pooled_sd**2)
            )
        shrunk_means.append(mean)
        shrunk_sds.append(sd)
        shrinkage_weights.append(shrinkage)
    out["shrunk_residual_log_odds"] = shrunk_means
    out["shrunk_sd"] = shrunk_sds
    out["shrinkage_weight"] = shrinkage_weights
    return {
        "pooled_effect": pooled,
        "pooled_sd": pooled_sd,
        "between_microstate_variance": tau2,
        "heterogeneity_q": q_value,
        "sum_random_effects_weights": float(np.sum(weights)),
    }, out


def _probability_outside_interval(mean: float, sd: float, threshold: float) -> float:
    if sd <= 0:
        return float(abs(mean) > threshold)
    lower_tail = norm.cdf((-threshold - mean) / sd)
    upper_tail = norm.sf((threshold - mean) / sd)
    return float(np.clip(lower_tail + upper_tail, 0.0, 1.0))


def symmetric_global_pseudolikelihood(
    left: Sequence[int] | np.ndarray | pd.Series,
    right: Sequence[int] | np.ndarray | pd.Series,
    p_left: Sequence[float] | np.ndarray | pd.Series,
    p_right: Sequence[float] | np.ndarray | pd.Series,
    *,
    prior_sd: float = 2.50,
    conditional_weight: float = 0.50,
    practical_threshold: float = 0.20,
    probability_clip: float = 1e-3,
    root_tolerance: float = 1e-10,
    max_root_iterations: int = 200,
) -> dict[str, float | bool | int]:
    """Fit one symmetric residual log-odds parameter with fixed OOF offsets.

    The objective is a symmetrically weighted sum of the two conditional
    Bernoulli log likelihoods, ``Y | X`` and ``X | Y``, plus a
    ``Normal(0, prior_sd**2)`` log prior.  The default half-weight averages the
    two views of the same binary-pair information instead of pretending they
    are independent replicates; synthetic null calibration tests cover that
    declared generalized-posterior scale.  The objective is strictly concave
    because of the prior, so the unique MAP is the zero of its monotone score.
    The returned standard deviation is the inverse square root of posterior
    information at that MAP (the one-dimensional Laplace approximation).

    This function deliberately receives probabilities rather than nuisance
    covariates: callers must estimate those probabilities out of fold before
    inspecting the other feature in the pair.
    """

    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    px = np.asarray(p_left, dtype=float).reshape(-1)
    py = np.asarray(p_right, dtype=float).reshape(-1)
    lengths = {len(x), len(y), len(px), len(py)}
    if len(lengths) != 1 or not lengths or len(x) == 0:
        raise ValueError("Detections and OOF probabilities must have equal nonzero length")
    if not np.isfinite(np.column_stack([x, y, px, py])).all():
        raise ValueError("Detections and OOF probabilities must be finite")
    if not set(np.unique(x)).issubset({0.0, 1.0}) or not set(np.unique(y)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("Detections must be binary")
    if prior_sd <= 0:
        raise ValueError("prior_sd must be > 0")
    if conditional_weight <= 0:
        raise ValueError("conditional_weight must be > 0")
    if practical_threshold < 0:
        raise ValueError("practical_threshold must be >= 0")
    if not 0 < probability_clip < 0.5:
        raise ValueError("probability_clip must be in (0, 0.5)")
    if root_tolerance <= 0 or max_root_iterations < 10:
        raise ValueError("Root controls must be positive and allow at least 10 iterations")

    px = np.clip(px, probability_clip, 1.0 - probability_clip)
    py = np.clip(py, probability_clip, 1.0 - probability_clip)
    offset_x = logit(px)
    offset_y = logit(py)
    prior_precision = 1.0 / float(prior_sd) ** 2

    def score(theta: float) -> float:
        fitted_y = expit(offset_y + theta * x)
        fitted_x = expit(offset_x + theta * y)
        return float(
            conditional_weight
            * (np.dot(x, y - fitted_y) + np.dot(y, x - fitted_x))
            - theta * prior_precision
        )

    lower, upper = -1.0, 1.0
    while score(lower) <= 0.0 and abs(lower) < 256.0:
        lower *= 2.0
    while score(upper) >= 0.0 and abs(upper) < 256.0:
        upper *= 2.0
    if score(lower) <= 0.0 or score(upper) >= 0.0:
        raise RuntimeError("Could not bracket the strictly concave pseudo-posterior MAP")
    root, solver = brentq(
        score,
        lower,
        upper,
        xtol=float(root_tolerance),
        rtol=max(float(root_tolerance), 4.0 * np.finfo(float).eps),
        maxiter=int(max_root_iterations),
        full_output=True,
        disp=False,
    )
    fitted_y = expit(offset_y + root * x)
    fitted_x = expit(offset_x + root * y)
    information = float(
        conditional_weight
        * (
            np.dot(x, fitted_y * (1.0 - fitted_y))
            + np.dot(y, fitted_x * (1.0 - fitted_x))
        )
        + prior_precision
    )
    if not np.isfinite(information) or information <= 0:
        raise RuntimeError("Global pseudo-posterior information is not positive")
    posterior_sd = float(1.0 / np.sqrt(information))
    probability_positive = float(norm.cdf(root / posterior_sd))
    probability_negative = float(norm.cdf(-root / posterior_sd))
    probability_gt_practical = float(
        norm.sf((float(practical_threshold) - root) / posterior_sd)
    )
    probability_lt_negative_practical = float(
        norm.cdf((-float(practical_threshold) - root) / posterior_sd)
    )
    return {
        "posterior_mean": float(root),
        "posterior_sd": posterior_sd,
        "ci95_lower": float(root - 1.959963984540054 * posterior_sd),
        "ci95_upper": float(root + 1.959963984540054 * posterior_sd),
        "posterior_probability_positive": probability_positive,
        "posterior_probability_gt_practical": probability_gt_practical,
        "posterior_probability_negative": probability_negative,
        "posterior_probability_lt_negative_practical": (
            probability_lt_negative_practical
        ),
        "posterior_probability_residual_dependence": float(
            np.clip(
                probability_gt_practical + probability_lt_negative_practical,
                0.0,
                1.0,
            )
        ),
        "local_false_sign_rate": float(
            min(probability_positive, probability_negative)
        ),
        "posterior_information": information,
        "prior_sd": float(prior_sd),
        "conditional_weight": float(conditional_weight),
        "converged": bool(solver.converged),
        "iterations": int(solver.iterations),
    }


def _positive_selection_probability(
    microstate_probability: float,
    global_probability: float,
    *,
    rule: str,
) -> float:
    """Choose the positive-only evidence exposed to CA-BOSS thresholding."""

    if rule == "microstate":
        return float(microstate_probability)
    if rule == "global":
        return float(global_probability)
    if rule == "intersection":
        return float(min(microstate_probability, global_probability))
    raise ValueError(f"Unknown positive evidence rule: {rule}")


def _phi(left: np.ndarray, right: np.ndarray) -> float:
    if left.std(ddof=0) <= 0 or right.std(ddof=0) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _local_pair_scores(
    left: np.ndarray,
    right: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    microstates: np.ndarray,
    *,
    node1: str,
    node2: str,
    config: HierarchicalSparseConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for microstate in sorted(int(value) for value in np.unique(microstates)):
        use = microstates == microstate
        n_cells = int(use.sum())
        if n_cells < config.min_microstate_cells:
            continue
        x = left[use].astype(float)
        y = right[use].astype(float)
        px = p_left[use]
        py = p_right[use]
        residual_x = x - px
        residual_y = y - py
        # Centering within state makes the interaction score robust to small OOF
        # marginal-calibration errors without using the other peak's values.
        residual_x -= residual_x.mean()
        residual_y -= residual_y.mean()
        score_values = residual_x * residual_y
        score = float(score_values.sum())
        information = float(np.sum(px * (1.0 - px) * py * (1.0 - py)))
        if information < config.min_local_information:
            continue
        empirical_score_variance = float(
            np.sum(np.square(score_values - score_values.mean()))
        )
        score_variance = max(information, empirical_score_variance, 1e-12)
        effect = score / information
        variance = score_variance / information**2
        rows.append(
            {
                "node1": node1,
                "node2": node2,
                "microstate": microstate,
                "n_cells": n_cells,
                "detection_count_node1": int(x.sum()),
                "detection_count_node2": int(y.sum()),
                "information": information,
                "raw_residual_log_odds": effect,
                "raw_variance": variance,
                "raw_sd": float(np.sqrt(variance)),
            }
        )
    return pd.DataFrame(rows)


def estimate_hierarchical_sparse_edges(
    detections: pd.DataFrame,
    integration_state: pd.DataFrame,
    background_depth: Sequence[float] | np.ndarray | pd.Series,
    *,
    config: HierarchicalSparseConfig | None = None,
) -> HierarchicalSparseResult:
    """Estimate symmetric residual-dependence evidence for one observed assay.

    Parameters
    ----------
    detections:
        Aligned nonnegative within-assay count/detection columns. Values are
        binarized with ``> 0``. No feature-name convention is assumed.
    integration_state:
        Aligned observed/background-derived integration coordinates only.
    background_depth:
        Aligned leave-locus-out library depth. The core function trusts the
        caller; :func:`estimate_from_observed_tables` enforces safe naming.
    config:
        Statistical and selection controls.
    """

    cfg = config or HierarchicalSparseConfig()
    cfg.validate()
    depth_name = getattr(background_depth, "name", None)
    if depth_name is not None and _is_forbidden_analysis_name(str(depth_name)):
        raise ValueError("Truth/latent/technical depth cannot be a nuisance covariate")
    if (
        depth_name is not None
        and "library_size" in str(depth_name).lower()
        and "background" not in str(depth_name).lower()
        and "leave_locus" not in str(depth_name).lower()
    ):
        raise ValueError("Full library size is not accepted as background depth")
    detection_frame = detections.reset_index(drop=True).copy()
    detection_frame.columns = detection_frame.columns.map(str)
    state_frame = integration_state.reset_index(drop=True).copy()
    state_frame.columns = state_frame.columns.map(str)
    detection_frame = detection_frame.apply(
        pd.to_numeric,
        errors="raise",
    )
    state_frame = state_frame.apply(
        pd.to_numeric,
        errors="raise",
    )
    depth = np.asarray(background_depth, dtype=float).reshape(-1)
    n_cells = len(detection_frame)
    if n_cells < 8:
        raise ValueError("At least 8 observed cells are required")
    if len(state_frame) != n_cells or len(depth) != n_cells:
        raise ValueError("detections, integration_state, and background_depth must align")
    if detection_frame.columns.duplicated().any():
        raise ValueError("Detection feature names must be unique")
    if state_frame.columns.duplicated().any():
        raise ValueError("pipelines/empirical/integration-state column names must be unique")
    forbidden_features = [
        column for column in detection_frame.columns if _is_forbidden_analysis_name(column)
    ]
    if forbidden_features:
        raise ValueError(f"Truth/latent/external features are forbidden: {forbidden_features}")
    reserved_features = [
        column
        for column in detection_frame.columns
        if column in NON_FEATURE_COLUMNS or "library_size" in column.lower()
    ]
    if reserved_features:
        raise ValueError(
            f"Identifiers, modality labels, and library-size columns are not features: "
            f"{reserved_features}"
        )
    forbidden_state = [
        column for column in state_frame.columns if _is_forbidden_analysis_name(column)
    ]
    if forbidden_state:
        raise ValueError(
            f"Truth/latent/external columns cannot be integration state: {forbidden_state}"
        )
    if detection_frame.shape[1] < 2:
        raise ValueError("At least two within-assay features are required")
    if not np.isfinite(detection_frame.to_numpy(dtype=float)).all():
        raise ValueError("Detection values must be finite")
    if (detection_frame.to_numpy(dtype=float) < 0).any():
        raise ValueError("Detection/count values must be nonnegative")
    if not np.isfinite(state_frame.to_numpy(dtype=float)).all():
        raise ValueError("pipelines/empirical/integration state must be finite")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Background depth must be finite and nonnegative")

    binary = (detection_frame > 0).astype(np.int8)
    event_counts = binary.sum(axis=0)
    live = [
        column
        for column in binary.columns
        if int(event_counts[column]) >= cfg.min_peak_events
        and int(n_cells - event_counts[column]) >= cfg.min_peak_events
    ]
    dropped = sorted(set(binary.columns) - set(live))
    binary = binary[live]

    microstates, standardized_state, microstate_summary = assign_microstates(
        state_frame,
        n_microstates=cfg.n_microstates,
        min_microstate_cells=cfg.min_microstate_cells,
        random_state=cfg.random_state,
    )
    design, design_columns = _nuisance_design(
        standardized_state,
        depth,
        microstates,
    )

    probabilities: dict[str, np.ndarray] = {}
    peak_rows: list[dict[str, Any]] = []
    for feature_i, feature in enumerate(live):
        observed = binary[feature].to_numpy(dtype=int)
        probability, fit_summary = cross_fitted_detection_probabilities(
            observed,
            design,
            config=cfg,
            seed=cfg.random_state + 104_729 * (feature_i + 1),
        )
        probabilities[feature] = probability
        peak_rows.append(
            {
                "feature": feature,
                "status": "modeled",
                "n_cells": n_cells,
                "detection_count": int(observed.sum()),
                "detection_rate": float(observed.mean()),
                "mean_predicted_rate": float(probability.mean()),
                "brier_score": float(np.mean(np.square(observed - probability))),
                **fit_summary,
            }
        )
    for feature in dropped:
        observed = binary[feature] if feature in binary else (detection_frame[feature] > 0)
        peak_rows.append(
            {
                "feature": feature,
                "status": "dropped_too_few_events",
                "n_cells": n_cells,
                "detection_count": int(observed.sum()),
                "detection_rate": float(observed.mean()),
                "mean_predicted_rate": float("nan"),
                "brier_score": float("nan"),
                "crossfit_folds_used": 0,
                "n_fallback_folds": 0,
                "fit_mode": "not_fit",
            }
        )

    edge_rows: list[dict[str, Any]] = []
    all_local: list[pd.DataFrame] = []
    for left, right in combinations(live, 2):
        node1, node2 = canonical_edge(str(left), str(right))
        x = binary[node1].to_numpy(dtype=int)
        y = binary[node2].to_numpy(dtype=int)
        local = _local_pair_scores(
            x,
            y,
            probabilities[node1],
            probabilities[node2],
            microstates,
            node1=node1,
            node2=node2,
            config=cfg,
        )
        if local.empty:
            continue
        pooled, local = empirical_bayes_pool(local)
        all_local.append(local)
        effect = float(pooled["pooled_effect"])
        effect_sd = float(pooled["pooled_sd"])
        practical = float(cfg.min_abs_residual_log_odds)
        posterior_positive = float(norm.sf((practical - effect) / effect_sd))
        posterior_negative = float(norm.cdf((-practical - effect) / effect_sd))
        probability_dependence = _probability_outside_interval(
            effect,
            effect_sd,
            practical,
        )
        probability_gt_zero = float(norm.cdf(effect / effect_sd))
        local_false_sign_rate = float(min(probability_gt_zero, 1.0 - probability_gt_zero))
        z_value = effect / effect_sd
        global_fit = symmetric_global_pseudolikelihood(
            x,
            y,
            probabilities[node1],
            probabilities[node2],
            prior_sd=cfg.global_interaction_prior_sd,
            conditional_weight=cfg.global_conditional_weight,
            practical_threshold=practical,
            probability_clip=cfg.probability_clip,
            root_tolerance=cfg.global_root_tolerance,
            max_root_iterations=cfg.global_max_root_iterations,
        )
        selection_probability = _positive_selection_probability(
            posterior_positive,
            float(global_fit["posterior_probability_gt_practical"]),
            rule=cfg.positive_evidence_rule,
        )
        edge_rows.append(
            {
                "node1": node1,
                "node2": node2,
                "n_cells": n_cells,
                "n_microstates_used": int(len(local)),
                "total_information": float(local["information"].sum()),
                "residual_log_odds": effect,
                "residual_log_odds_sd": effect_sd,
                "ci95_lower": float(effect - 1.959963984540054 * effect_sd),
                "ci95_upper": float(effect + 1.959963984540054 * effect_sd),
                # This compatibility column is the positive-only probability
                # selected by ``positive_evidence_rule``.  The original
                # microstate probability is preserved explicitly below.
                "posterior_probability_positive": selection_probability,
                "posterior_probability_negative": posterior_negative,
                "posterior_probability_residual_dependence": probability_dependence,
                "local_false_sign_rate": local_false_sign_rate,
                "approx_p_value": float(2.0 * norm.sf(abs(z_value))),
                "between_microstate_sd": float(
                    np.sqrt(pooled["between_microstate_variance"])
                ),
                "heterogeneity_q": float(pooled["heterogeneity_q"]),
                "unconditional_phi": _phi(x, y),
                "detection_rate_node1": float(x.mean()),
                "detection_rate_node2": float(y.mean()),
                "microstate_residual_log_odds": effect,
                "microstate_residual_log_odds_sd": effect_sd,
                "microstate_posterior_probability_positive": posterior_positive,
                "microstate_posterior_probability_negative": posterior_negative,
                "microstate_posterior_probability_residual_dependence": (
                    probability_dependence
                ),
                "global_pseudolikelihood_log_odds": float(
                    global_fit["posterior_mean"]
                ),
                "global_pseudolikelihood_log_odds_sd": float(
                    global_fit["posterior_sd"]
                ),
                "global_pseudolikelihood_ci95_lower": float(
                    global_fit["ci95_lower"]
                ),
                "global_pseudolikelihood_ci95_upper": float(
                    global_fit["ci95_upper"]
                ),
                "global_posterior_probability_positive": float(
                    global_fit["posterior_probability_positive"]
                ),
                "global_posterior_probability_gt_practical": float(
                    global_fit["posterior_probability_gt_practical"]
                ),
                "global_posterior_probability_negative": float(
                    global_fit["posterior_probability_negative"]
                ),
                "global_posterior_probability_lt_negative_practical": float(
                    global_fit["posterior_probability_lt_negative_practical"]
                ),
                "global_posterior_probability_residual_dependence": float(
                    global_fit["posterior_probability_residual_dependence"]
                ),
                "global_local_false_sign_rate": float(
                    global_fit["local_false_sign_rate"]
                ),
                "global_pseudolikelihood_information": float(
                    global_fit["posterior_information"]
                ),
                "global_interaction_prior_sd": float(global_fit["prior_sd"]),
                "global_conditional_weight": float(
                    global_fit["conditional_weight"]
                ),
                "global_pseudolikelihood_converged": bool(global_fit["converged"]),
                "global_pseudolikelihood_iterations": int(global_fit["iterations"]),
                "positive_evidence_rule": cfg.positive_evidence_rule,
                "selection_probability_positive": selection_probability,
                "selected": bool(selection_probability >= cfg.evidence_threshold),
                "interpretation": (
                    "within_assay_positive_residual_adjacency_evidence"
                ),
            }
        )

    edge_evidence = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    if not edge_evidence.empty:
        edge_evidence = edge_evidence.sort_values(
            ["selection_probability_positive", "node1", "node2"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    microstate_evidence = (
        pd.concat(all_local, ignore_index=True)
        if all_local
        else pd.DataFrame(columns=[*LOCAL_COLUMNS, "raw_variance"])
    )
    if not microstate_evidence.empty:
        microstate_evidence = microstate_evidence[
            [*LOCAL_COLUMNS, "raw_variance"]
        ].sort_values(["node1", "node2", "microstate"]).reset_index(drop=True)
    assignments = pd.DataFrame(
        {
            "row_index": np.arange(n_cells, dtype=int),
            "microstate": microstates,
        }
    )
    peak_diagnostics = pd.DataFrame(peak_rows).sort_values("feature").reset_index(drop=True)
    metadata: dict[str, Any] = {
        "method": "hierarchical_sparse_bernoulli_score_eb",
        "config": asdict(cfg),
        "n_cells": int(n_cells),
        "n_features_input": int(detection_frame.shape[1]),
        "n_features_modeled": int(len(live)),
        "dropped_features": dropped,
        "n_pairs_scored": int(len(edge_evidence)),
        "n_edges_selected": int(
            edge_evidence.get("selected", pd.Series(dtype=bool)).sum()
        ),
        "nuisance_design_columns": design_columns,
        "microstates": microstate_summary,
        "global_symmetric_pseudolikelihood": {
            "enabled": True,
            "conditional_models": ["Y_given_X", "X_given_Y"],
            "shared_parameter": "residual_log_odds",
            "offsets": "cross_fitted_marginal_detection_logits",
            "prior": {
                "distribution": "Normal",
                "mean": 0.0,
                "sd": float(cfg.global_interaction_prior_sd),
            },
            "posterior_approximation": "one_dimensional_Laplace_at_MAP",
            "conditional_weight": float(cfg.global_conditional_weight),
            "positive_evidence_rule": cfg.positive_evidence_rule,
            "practical_threshold": float(cfg.min_abs_residual_log_odds),
        },
        "estimand": (
            "symmetric positive within-assay residual interaction after conditioning "
            "on observed integration state and leave-locus-out background depth"
        ),
        "orientation": "abstained",
    }
    return HierarchicalSparseResult(
        edge_evidence=edge_evidence,
        microstate_evidence=microstate_evidence,
        peak_diagnostics=peak_diagnostics,
        assignments=assignments,
        metadata=metadata,
    )


def _read_frame(value: Path | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(Path(value), sep="\t")


def _is_forbidden_analysis_name(column: str) -> bool:
    lowered = str(column).lower()
    return lowered.startswith(FORBIDDEN_PREFIXES) or any(
        token in lowered for token in FORBIDDEN_TOKENS
    )


def estimate_from_observed_tables(
    cell_table: Path | str | pd.DataFrame,
    embedding_table: Path | str | pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    cell_column: str = "cell",
    depth_column: str = "observed_background_library_size",
    state_prefix: str = "integration__",
    retained_cells_only: bool = False,
    config: HierarchicalSparseConfig | None = None,
) -> HierarchicalSparseResult:
    """Load one observed assay and its background-derived integration state.

    The join is within assay by cell identifier.  No cross-modality matching or
    pairing table is accepted.  The default depth column is the simulator's
    focal-feature-excluded background library size.
    """

    cells = _read_frame(cell_table)
    embeddings = _read_frame(embedding_table)
    if cell_column not in cells or cell_column not in embeddings:
        raise ValueError(f"Both tables must contain {cell_column!r}")
    if cells[cell_column].duplicated().any() or embeddings[cell_column].duplicated().any():
        raise ValueError("Cell identifiers must be unique within each observed table")
    if depth_column not in cells:
        raise ValueError(f"Cell table is missing depth column {depth_column!r}")
    safe_depth_name = depth_column.lower()
    if "background" not in safe_depth_name and "leave_locus" not in safe_depth_name:
        raise ValueError(
            "Depth must be leave-locus-out/background-derived; full library depth is not accepted"
        )
    state_columns = [
        column for column in embeddings.columns if str(column).startswith(state_prefix)
    ]
    if not state_columns:
        raise ValueError(f"No observed state columns begin with {state_prefix!r}")
    if any(_is_forbidden_analysis_name(column) for column in state_columns):
        raise ValueError("Truth/latent/external columns cannot be used as integration state")

    if feature_columns is None:
        excluded = set(NON_FEATURE_COLUMNS) | {cell_column, depth_column}
        inferred: list[str] = []
        for column in cells.columns:
            if column in excluded or str(column).startswith(state_prefix):
                continue
            if _is_forbidden_analysis_name(str(column)):
                continue
            if pd.api.types.is_numeric_dtype(cells[column]):
                inferred.append(str(column))
        feature_columns = inferred
    feature_columns = [str(column) for column in feature_columns]
    if len(feature_columns) < 2:
        raise ValueError("At least two observed within-assay features are required")
    missing_features = [column for column in feature_columns if column not in cells]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    forbidden_features = [
        column for column in feature_columns if _is_forbidden_analysis_name(column)
    ]
    if forbidden_features:
        raise ValueError(f"Truth/latent/external features are forbidden: {forbidden_features}")
    if depth_column in feature_columns or "observed_library_size" in feature_columns:
        raise ValueError("Library-size columns cannot be candidate features")

    embedding_columns = [cell_column, *state_columns]
    if "retained_target" in embeddings:
        embedding_columns.insert(1, "retained_target")
    joined = cells[[cell_column, depth_column, *feature_columns]].merge(
        embeddings[embedding_columns],
        on=cell_column,
        how="inner",
        validate="one_to_one",
    )
    if retained_cells_only:
        if "retained_target" not in joined:
            raise ValueError("retained_cells_only requires a retained_target column")
        retained = joined["retained_target"]
        if pd.api.types.is_bool_dtype(retained):
            retained_mask = retained.to_numpy(dtype=bool)
        elif pd.api.types.is_numeric_dtype(retained):
            retained_mask = retained.to_numpy(dtype=float) != 0
        else:
            normalized = retained.astype(str).str.strip().str.lower()
            unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
            if unknown:
                raise ValueError(f"Unrecognized retained_target values: {unknown}")
            retained_mask = normalized.isin({"true", "1"}).to_numpy(dtype=bool)
        joined = joined.loc[retained_mask].copy()
    joined = joined.reset_index(drop=True)
    result = estimate_hierarchical_sparse_edges(
        joined[feature_columns],
        joined[state_columns],
        joined[depth_column],
        config=config,
    )
    result.assignments.insert(0, cell_column, joined[cell_column].astype(str).to_numpy())
    result.metadata["observed_table_contract"] = {
        "cell_column": cell_column,
        "depth_column": depth_column,
        "state_columns": state_columns,
        "feature_columns": feature_columns,
        "retained_cells_only": bool(retained_cells_only),
        "n_cells_after_join": int(len(joined)),
        "truth_or_pairing_inputs": False,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-table", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--cell-column", default="cell")
    parser.add_argument("--depth-column", default="observed_background_library_size")
    parser.add_argument("--state-prefix", default="integration__")
    parser.add_argument("--retained-cells-only", action="store_true")
    parser.add_argument("--n-microstates", type=int, default=20)
    parser.add_argument("--min-microstate-cells", type=int, default=20)
    parser.add_argument("--min-peak-events", type=int, default=5)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--nuisance-c", type=float, default=1.0)
    parser.add_argument("--min-local-information", type=float, default=0.20)
    parser.add_argument("--min-abs-effect", type=float, default=0.20)
    parser.add_argument("--evidence-threshold", type=float, default=0.95)
    parser.add_argument("--global-interaction-prior-sd", type=float, default=2.50)
    parser.add_argument("--global-conditional-weight", type=float, default=0.50)
    parser.add_argument(
        "--positive-evidence-rule",
        choices=("microstate", "global", "intersection"),
        default="intersection",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = HierarchicalSparseConfig(
        n_microstates=args.n_microstates,
        min_microstate_cells=args.min_microstate_cells,
        min_peak_events=args.min_peak_events,
        crossfit_folds=args.crossfit_folds,
        nuisance_c=args.nuisance_c,
        min_local_information=args.min_local_information,
        min_abs_residual_log_odds=args.min_abs_effect,
        evidence_threshold=args.evidence_threshold,
        global_interaction_prior_sd=args.global_interaction_prior_sd,
        global_conditional_weight=args.global_conditional_weight,
        positive_evidence_rule=args.positive_evidence_rule,
        random_state=args.seed,
    )
    result = estimate_from_observed_tables(
        args.cell_table,
        args.embedding_table,
        feature_columns=args.feature or None,
        cell_column=args.cell_column,
        depth_column=args.depth_column,
        state_prefix=args.state_prefix,
        retained_cells_only=args.retained_cells_only,
        config=config,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.edge_evidence.to_csv(
        args.out_dir / "hierarchical_sparse_edge_evidence.tsv",
        sep="\t",
        index=False,
    )
    result.microstate_evidence.to_csv(
        args.out_dir / "hierarchical_sparse_microstate_evidence.tsv",
        sep="\t",
        index=False,
    )
    result.peak_diagnostics.to_csv(
        args.out_dir / "hierarchical_sparse_peak_diagnostics.tsv",
        sep="\t",
        index=False,
    )
    result.assignments.to_csv(
        args.out_dir / "hierarchical_sparse_microstate_assignments.tsv",
        sep="\t",
        index=False,
    )
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(result.metadata, indent=2, allow_nan=False)
    )
    print(
        "[hierarchical-sparse] "
        f"{result.metadata['n_cells']} cells, "
        f"{result.metadata['n_features_modeled']} features -> "
        f"{result.metadata['n_edges_selected']}/{result.metadata['n_pairs_scored']} "
        "selected residual adjacencies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
