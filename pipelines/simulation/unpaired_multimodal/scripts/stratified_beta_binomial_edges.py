#!/usr/bin/env python3
"""Observed-only sparse same-assay evidence for CA-BOSS v4.

This module is an isolated prototype.  It does not replace the production
hierarchical Bernoulli layer.  It estimates a *positive*, undirected residual
association between sparse feature detections after conditioning on an
observed background-derived state embedding and leave-focal-out library depth.

The procedure has three deliberately separate pieces:

1. Cells are blocked by K-means state clusters and within-cluster depth ranks.
2. Each block contributes a Jeffreys beta-binomial posterior log odds ratio;
   block effects are partially pooled with a random-effects approximation.
3. A conditional Mantel-Haenszel score is calibrated by independently
   permuting every feature within the state/depth blocks.  Thus every null draw
   preserves feature prevalence, state composition, and depth composition.

The empirical-null p-value is the primary calibration.  The beta-binomial
posterior supplies an effect-size/uncertainty guard; it is not treated as an
exact Bayesian posterior after the empirical-Bayes and random-effects steps.
No direction is claimed from symmetric co-detection evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


FORBIDDEN_TOKENS = (
    "truth",
    "true_",
    "latent",
    "oracle",
    "counterfactual",
    "pairing",
    "correspondence",
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

EDGE_COLUMNS = [
    "node1",
    "node2",
    "n_cells",
    "n_strata_used",
    "detection_count_node1",
    "detection_count_node2",
    "joint_detection_count",
    "cmh_score",
    "cmh_information",
    "efficient_residual_score",
    "pooled_log_odds",
    "pooled_log_odds_sd",
    "ci95_lower",
    "ci95_upper",
    "posterior_probability_positive_practical",
    "between_stratum_sd",
    "heterogeneity_q",
    "empirical_null_mean",
    "empirical_null_sd",
    "empirical_null_q90",
    "empirical_null_q95",
    "empirical_p_positive",
    "empirical_q_positive",
    "selected",
    "selection_reason",
    "interpretation",
]

LOCAL_COLUMNS = [
    "node1",
    "node2",
    "stratum",
    "n_cells",
    "neither",
    "node1_only",
    "node2_only",
    "both",
    "posterior_log_odds",
    "posterior_log_odds_variance",
    "posterior_log_odds_sd",
    "random_effect_weight",
]


@dataclass(frozen=True)
class StratifiedBetaBinomialConfig:
    """Frozen prototype settings.

    The defaults were chosen from the estimand and synthetic calibration, not
    from scorer truth.  With 399 permutations the smallest attainable p-value
    is 0.0025, which permits meaningful within-assay BH diagnostics for the 15
    pairs produced by six focal features.
    """

    n_state_clusters: int = 6
    n_depth_bins: int = 2
    min_stratum_cells: int = 20
    min_feature_events: int = 5
    nuisance_crossfit_folds: int = 5
    nuisance_c: float = 1.0
    probability_clip: float = 1e-3
    prior_alpha: float = 0.5
    prior_beta: float = 0.5
    practical_log_odds: float = 0.10
    min_joint_events: int = 2
    posterior_probability_threshold: float = 0.80
    empirical_p_threshold: float = 0.10
    empirical_q_threshold: float = 0.25
    n_permutations: int = 399
    random_state: int = 0

    def validate(self) -> None:
        if self.n_state_clusters < 1 or self.n_depth_bins < 1:
            raise ValueError("State clusters and depth bins must be positive")
        if self.min_stratum_cells < 4:
            raise ValueError("min_stratum_cells must be at least four")
        if self.min_feature_events < 2:
            raise ValueError("min_feature_events must be at least two")
        if self.nuisance_crossfit_folds < 2 or self.nuisance_c <= 0:
            raise ValueError("Nuisance cross-fit settings must be positive")
        if not 0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, 0.5)")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior shapes must be positive")
        if self.practical_log_odds < 0:
            raise ValueError("practical_log_odds must be non-negative")
        if self.min_joint_events < 1:
            raise ValueError("min_joint_events must be positive")
        if not 0.5 < self.posterior_probability_threshold < 1.0:
            raise ValueError("posterior_probability_threshold must lie in (0.5, 1)")
        if not 0.0 < self.empirical_p_threshold < 1.0:
            raise ValueError("empirical_p_threshold must lie in (0, 1)")
        if not 0.0 < self.empirical_q_threshold < 1.0:
            raise ValueError("empirical_q_threshold must lie in (0, 1)")
        if self.n_permutations < 19:
            raise ValueError("At least 19 permutations are required")


@dataclass
class StratifiedBetaBinomialResult:
    edge_evidence: pd.DataFrame
    stratum_evidence: pd.DataFrame
    assignments: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def selected_edges(self) -> pd.DataFrame:
        if self.edge_evidence.empty:
            return self.edge_evidence.copy()
        return self.edge_evidence.loc[self.edge_evidence["selected"].astype(bool)].copy()


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _is_forbidden_name(value: str) -> bool:
    lowered = str(value).lower()
    return any(token in lowered for token in FORBIDDEN_TOKENS)


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    scale = numeric.std(axis=0, ddof=0)
    live = [column for column in numeric if float(scale[column]) > 1e-12]
    if not live:
        return pd.DataFrame(index=numeric.index)
    return (numeric[live] - numeric[live].mean(axis=0)).divide(scale[live], axis=1)


def assign_state_depth_strata(
    integration_state: pd.DataFrame,
    background_depth: Sequence[float] | np.ndarray | pd.Series,
    *,
    config: StratifiedBetaBinomialConfig,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Return observed state clusters crossed with within-cluster depth ranks."""

    state = _standardize(integration_state.reset_index(drop=True))
    depth = np.asarray(background_depth, dtype=float).reshape(-1)
    if len(state) != len(depth):
        raise ValueError("State and background depth rows must align")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Background depth must be finite and non-negative")
    n_cells = len(state)
    maximum_clusters = max(
        1,
        n_cells // (config.n_depth_bins * config.min_stratum_cells),
    )
    actual_clusters = min(config.n_state_clusters, maximum_clusters)
    if actual_clusters == 1 or state.shape[1] == 0:
        state_labels = np.zeros(n_cells, dtype=int)
        actual_clusters = 1
    else:
        state_labels = KMeans(
            n_clusters=actual_clusters,
            random_state=config.random_state,
            n_init=10,
        ).fit_predict(state.to_numpy(dtype=float))

    # Depth bins are rank based, so tied integer library sizes cannot create an
    # empty qcut category.  A small state cluster automatically uses fewer bins.
    raw_labels: list[tuple[int, int]] = [(-1, -1)] * n_cells
    bins_per_cluster: dict[str, int] = {}
    for cluster in sorted(int(value) for value in np.unique(state_labels)):
        indices = np.flatnonzero(state_labels == cluster)
        actual_bins = min(
            config.n_depth_bins,
            max(1, len(indices) // config.min_stratum_cells),
        )
        order = np.argsort(depth[indices], kind="mergesort")
        ranked_bin = np.empty(len(indices), dtype=int)
        ranked_bin[order] = np.minimum(
            (np.arange(len(indices)) * actual_bins) // len(indices),
            actual_bins - 1,
        )
        bins_per_cluster[str(cluster)] = int(actual_bins)
        for local_i, row_i in enumerate(indices):
            raw_labels[int(row_i)] = (cluster, int(ranked_bin[local_i]))

    unique = sorted(set(raw_labels))
    relabel = {label: index for index, label in enumerate(unique)}
    labels = np.asarray([relabel[label] for label in raw_labels], dtype=int)
    sizes = pd.Series(labels).value_counts().sort_index()
    assignments = pd.DataFrame(
        {
            "row_index": np.arange(n_cells, dtype=int),
            "state_cluster": state_labels,
            "depth_bin": [label[1] for label in raw_labels],
            "stratum": labels,
            "background_depth": depth,
        }
    )
    return labels, assignments, {
        "requested_state_clusters": int(config.n_state_clusters),
        "actual_state_clusters": int(actual_clusters),
        "requested_depth_bins": int(config.n_depth_bins),
        "depth_bins_per_state_cluster": bins_per_cluster,
        "actual_strata": int(len(unique)),
        "stratum_sizes": {str(int(k)): int(v) for k, v in sizes.items()},
        "state_columns": list(state.columns),
    }


def _cmh_score(left: np.ndarray, right: np.ndarray, strata: np.ndarray) -> tuple[float, float]:
    score = 0.0
    information = 0.0
    for label in np.unique(strata):
        use = strata == label
        x = left[use]
        y = right[use]
        n = len(x)
        if n <= 1:
            continue
        n1 = float(x.sum())
        m1 = float(y.sum())
        expected = n1 * m1 / n
        score += float(np.dot(x, y)) - expected
        information += n1 * (n - n1) * m1 * (n - m1) / (n * n * (n - 1))
    if information <= 1e-12:
        return 0.0, 0.0
    return float(score / np.sqrt(information)), float(information)


def _nuisance_design(
    standardized_state: pd.DataFrame,
    background_depth: np.ndarray,
    assignments: pd.DataFrame,
) -> np.ndarray:
    """Continuous state/depth design plus coarse nonlinear state indicators."""

    log_depth = np.log1p(background_depth)
    depth_sd = float(log_depth.std(ddof=0))
    depth_column = (
        ((log_depth - log_depth.mean()) / depth_sd)[:, None]
        if depth_sd > 1e-12
        else np.zeros((len(log_depth), 1), dtype=float)
    )
    parts = [standardized_state.to_numpy(dtype=float), depth_column]
    clusters = assignments["state_cluster"].to_numpy(dtype=int)
    unique = sorted(int(value) for value in np.unique(clusters))
    if len(unique) > 1:
        parts.append(
            np.column_stack([(clusters == value).astype(float) for value in unique[1:]])
        )
    return np.column_stack(parts).astype(float)


def _cross_fitted_probabilities(
    detection: np.ndarray,
    design: np.ndarray,
    *,
    config: StratifiedBetaBinomialConfig,
    random_state: int,
) -> np.ndarray:
    """Out-of-fold detection probabilities under the state/depth null."""

    y = np.asarray(detection, dtype=int)
    counts = np.bincount(y, minlength=2)
    n_splits = min(config.nuisance_crossfit_folds, int(counts.min()))
    if n_splits < 2:
        rate = np.clip(
            (float(y.sum()) + 0.5) / (len(y) + 1.0),
            config.probability_clip,
            1.0 - config.probability_clip,
        )
        return np.full(len(y), rate, dtype=float)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    probabilities = np.full(len(y), np.nan, dtype=float)
    for train, test in splitter.split(design, y):
        model = LogisticRegression(
            C=config.nuisance_c,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=1_000,
            random_state=random_state,
        )
        model.fit(design[train], y[train])
        probabilities[test] = model.predict_proba(design[test])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Cross-fitting left non-finite probabilities")
    return np.clip(
        probabilities,
        config.probability_clip,
        1.0 - config.probability_clip,
    )


def _center_within_strata(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float).copy()
    for label in np.unique(strata):
        use = strata == label
        centered[use] -= float(centered[use].mean())
    return centered


def _efficient_residual_score(
    left_residual: np.ndarray,
    right_residual: np.ndarray,
) -> float:
    denominator = float(
        np.sqrt(
            np.dot(left_residual, left_residual)
            * np.dot(right_residual, right_residual)
        )
    )
    if denominator <= 1e-12:
        return 0.0
    return float(
        np.sqrt(len(left_residual))
        * np.dot(left_residual, right_residual)
        / denominator
    )


def _local_beta_binomial_tables(
    left: np.ndarray,
    right: np.ndarray,
    strata: np.ndarray,
    *,
    node1: str,
    node2: str,
    config: StratifiedBetaBinomialConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in sorted(int(value) for value in np.unique(strata)):
        use = strata == label
        x = left[use]
        y = right[use]
        if len(x) < config.min_stratum_cells or x.min() == x.max() or y.min() == y.max():
            continue
        both = int(np.sum((x == 1) & (y == 1)))
        x_only = int(np.sum((x == 1) & (y == 0)))
        y_only = int(np.sum((x == 0) & (y == 1)))
        neither = int(np.sum((x == 0) & (y == 0)))
        a = both + config.prior_alpha
        b = x_only + config.prior_beta
        c = y_only + config.prior_alpha
        d = neither + config.prior_beta
        # E[logit p] and Var[logit p] under independent beta posteriors.
        effect = float(digamma(a) - digamma(b) - digamma(c) + digamma(d))
        variance = float(
            polygamma(1, a) + polygamma(1, b) + polygamma(1, c) + polygamma(1, d)
        )
        rows.append(
            {
                "node1": node1,
                "node2": node2,
                "stratum": label,
                "n_cells": int(len(x)),
                "neither": neither,
                "node1_only": x_only,
                "node2_only": y_only,
                "both": both,
                "posterior_log_odds": effect,
                "posterior_log_odds_variance": variance,
                "posterior_log_odds_sd": float(np.sqrt(variance)),
            }
        )
    return pd.DataFrame(rows)


def _random_effect_pool(local: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    effects = local["posterior_log_odds"].to_numpy(dtype=float)
    variances = local["posterior_log_odds_variance"].to_numpy(dtype=float)
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    if len(effects) == 1:
        tau2 = 0.0
        q_value = 0.0
    else:
        q_value = float(np.sum(fixed_weights * np.square(effects - fixed_mean)))
        c_value = float(
            np.sum(fixed_weights)
            - np.sum(np.square(fixed_weights)) / np.sum(fixed_weights)
        )
        tau2 = max(0.0, (q_value - len(effects) + 1) / max(c_value, 1e-12))
    weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    residual_q = float(np.sum(weights * np.square(effects - pooled)))
    scale = max(1.0, residual_q / max(len(effects) - 1, 1))
    pooled_sd = float(np.sqrt(scale / np.sum(weights)))
    out = local.copy()
    out["random_effect_weight"] = weights
    return {
        "pooled_log_odds": pooled,
        "pooled_log_odds_sd": pooled_sd,
        "between_stratum_variance": float(tau2),
        "heterogeneity_q": q_value,
    }, out


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _permuted_null_scores(
    residuals: np.ndarray,
    strata: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    n_permutations: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    indices = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    null = np.empty((n_permutations, len(pairs)), dtype=float)
    permuted = np.empty_like(residuals)
    for repetition in range(n_permutations):
        for feature_i in range(residuals.shape[1]):
            for block in indices:
                permuted[block, feature_i] = rng.permutation(
                    residuals[block, feature_i]
                )
        for pair_i, (left_i, right_i) in enumerate(pairs):
            null[repetition, pair_i] = _efficient_residual_score(
                permuted[:, left_i],
                permuted[:, right_i],
            )
    return null


def estimate_stratified_beta_binomial_edges(
    detections: pd.DataFrame,
    integration_state: pd.DataFrame,
    background_depth: Sequence[float] | np.ndarray | pd.Series,
    *,
    config: StratifiedBetaBinomialConfig | None = None,
) -> StratifiedBetaBinomialResult:
    """Estimate positive sparse residual associations in one observed assay."""

    cfg = config or StratifiedBetaBinomialConfig()
    cfg.validate()
    depth_name = str(getattr(background_depth, "name", "") or "")
    if _is_forbidden_name(depth_name):
        raise ValueError("Truth/latent/technical depth is forbidden")
    if "library_size" in depth_name.lower() and not any(
        token in depth_name.lower() for token in ("background", "leave_focal", "leave_locus")
    ):
        raise ValueError("Depth must be background/leave-focal-out, not full library size")

    frame = detections.reset_index(drop=True).copy()
    frame.columns = frame.columns.map(str)
    state = integration_state.reset_index(drop=True).copy()
    state.columns = state.columns.map(str)
    if len(frame) < 20 or len(state) != len(frame):
        raise ValueError("At least 20 aligned observed cells are required")
    if frame.shape[1] < 2:
        raise ValueError("At least two within-assay features are required")
    if frame.columns.duplicated().any() or state.columns.duplicated().any():
        raise ValueError("Feature and state names must be unique")
    forbidden = [column for column in [*frame.columns, *state.columns] if _is_forbidden_name(column)]
    if forbidden:
        raise ValueError(f"Truth/latent/external columns are forbidden: {forbidden}")
    reserved = [
        column
        for column in frame.columns
        if column in NON_FEATURE_COLUMNS or "library_size" in column.lower()
    ]
    if reserved:
        raise ValueError(f"Identifiers and library-size columns are not features: {reserved}")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite")
    if np.any(numeric.to_numpy(dtype=float) < 0):
        raise ValueError("Feature values must be non-negative")
    binary_all = (numeric > 0).astype(np.int8)
    event_counts = binary_all.sum(axis=0)
    live = [
        column
        for column in binary_all
        if int(event_counts[column]) >= cfg.min_feature_events
        and int(len(binary_all) - event_counts[column]) >= cfg.min_feature_events
    ]
    dropped = sorted(set(binary_all.columns) - set(live))
    binary_frame = binary_all[live]
    if len(live) < 2:
        raise ValueError("Fewer than two features have sufficient detection events")

    strata, assignments, stratum_summary = assign_state_depth_strata(
        state,
        background_depth,
        config=cfg,
    )
    binary = binary_frame.to_numpy(dtype=np.int8)
    standardized_state = _standardize(state)
    design = _nuisance_design(
        standardized_state,
        np.asarray(background_depth, dtype=float),
        assignments,
    )
    probabilities = np.column_stack(
        [
            _cross_fitted_probabilities(
                binary[:, feature_i],
                design,
                config=cfg,
                random_state=cfg.random_state + 104_729 * (feature_i + 1),
            )
            for feature_i in range(binary.shape[1])
        ]
    )
    residuals = np.column_stack(
        [
            _center_within_strata(
                binary[:, feature_i] - probabilities[:, feature_i],
                strata,
            )
            for feature_i in range(binary.shape[1])
        ]
    )
    pair_indices = list(combinations(range(len(live)), 2))
    observed_scores = np.asarray(
        [
            _efficient_residual_score(residuals[:, i], residuals[:, j])
            for i, j in pair_indices
        ]
    )
    null_scores = _permuted_null_scores(
        residuals,
        strata,
        pair_indices,
        n_permutations=cfg.n_permutations,
        random_state=cfg.random_state + 1_000_003,
    )
    empirical_p = (
        1.0 + np.sum(null_scores >= observed_scores[None, :], axis=0)
    ) / (cfg.n_permutations + 1.0)
    empirical_q = _benjamini_hochberg(empirical_p)

    rows: list[dict[str, Any]] = []
    local_frames: list[pd.DataFrame] = []
    for pair_i, (left_i, right_i) in enumerate(pair_indices):
        node1, node2 = canonical_edge(live[left_i], live[right_i])
        # live is already the frame order; recover arrays by canonical names to
        # make every statistic invariant to caller column order.
        left = binary_frame[node1].to_numpy(dtype=np.int8)
        right = binary_frame[node2].to_numpy(dtype=np.int8)
        score, information = _cmh_score(left, right, strata)
        efficient_score = float(observed_scores[pair_i])
        local = _local_beta_binomial_tables(
            left,
            right,
            strata,
            node1=node1,
            node2=node2,
            config=cfg,
        )
        if local.empty:
            continue
        pooled, local = _random_effect_pool(local)
        local_frames.append(local)
        effect = pooled["pooled_log_odds"]
        effect_sd = pooled["pooled_log_odds_sd"]
        posterior_positive = float(
            norm.sf((cfg.practical_log_odds - effect) / max(effect_sd, 1e-12))
        )
        joint = int(np.dot(left, right))
        null = null_scores[:, pair_i]
        selected = bool(
            joint >= cfg.min_joint_events
            and effect >= cfg.practical_log_odds
            and posterior_positive >= cfg.posterior_probability_threshold
            and empirical_p[pair_i] <= cfg.empirical_p_threshold
            and empirical_q[pair_i] <= cfg.empirical_q_threshold
        )
        failures: list[str] = []
        if joint < cfg.min_joint_events:
            failures.append("insufficient_joint_events")
        if effect < cfg.practical_log_odds:
            failures.append("effect_below_practical_threshold")
        if posterior_positive < cfg.posterior_probability_threshold:
            failures.append("posterior_uncertain")
        if empirical_p[pair_i] > cfg.empirical_p_threshold:
            failures.append("empirical_null_not_exceeded")
        if empirical_q[pair_i] > cfg.empirical_q_threshold:
            failures.append("within_assay_multiplicity_not_controlled")
        rows.append(
            {
                "node1": node1,
                "node2": node2,
                "n_cells": int(len(binary)),
                "n_strata_used": int(len(local)),
                "detection_count_node1": int(left.sum()),
                "detection_count_node2": int(right.sum()),
                "joint_detection_count": joint,
                "cmh_score": score,
                "cmh_information": information,
                "efficient_residual_score": efficient_score,
                "pooled_log_odds": effect,
                "pooled_log_odds_sd": effect_sd,
                "ci95_lower": float(effect - 1.959963984540054 * effect_sd),
                "ci95_upper": float(effect + 1.959963984540054 * effect_sd),
                "posterior_probability_positive_practical": posterior_positive,
                "between_stratum_sd": float(np.sqrt(pooled["between_stratum_variance"])),
                "heterogeneity_q": pooled["heterogeneity_q"],
                "empirical_null_mean": float(null.mean()),
                "empirical_null_sd": float(null.std(ddof=1)),
                "empirical_null_q90": float(np.quantile(null, 0.90)),
                "empirical_null_q95": float(np.quantile(null, 0.95)),
                "empirical_p_positive": float(empirical_p[pair_i]),
                "empirical_q_positive": float(empirical_q[pair_i]),
                "selected": selected,
                "selection_reason": "selected" if selected else ";".join(failures),
                "interpretation": "within_assay_positive_residual_adjacency_evidence",
            }
        )

    evidence = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    if not evidence.empty:
        evidence = evidence.sort_values(
            [
                "selected",
                "empirical_p_positive",
                "efficient_residual_score",
                "node1",
                "node2",
            ],
            ascending=[False, True, False, True, True],
        ).reset_index(drop=True)
    local_evidence = (
        pd.concat(local_frames, ignore_index=True)[LOCAL_COLUMNS]
        if local_frames
        else pd.DataFrame(columns=LOCAL_COLUMNS)
    )
    diagnostics = pd.DataFrame(
        [
            {
                "feature": column,
                "status": "modeled" if column in live else "dropped_too_few_events",
                "n_cells": int(len(binary_all)),
                "detection_count": int(event_counts[column]),
                "detection_rate": float(event_counts[column] / len(binary_all)),
                "mean_predicted_detection_rate": float(
                    probabilities[:, live.index(column)].mean()
                )
                if column in live
                else float("nan"),
                "brier_score": float(
                    np.mean(
                        np.square(
                            binary[:, live.index(column)]
                            - probabilities[:, live.index(column)]
                        )
                    )
                )
                if column in live
                else float("nan"),
            }
            for column in binary_all
        ]
    ).sort_values("feature").reset_index(drop=True)
    metadata: dict[str, Any] = {
        "method": "state_depth_stratified_beta_binomial_empirical_null",
        "config": asdict(cfg),
        "n_cells": int(len(binary_all)),
        "n_features_input": int(binary_all.shape[1]),
        "n_features_modeled": int(len(live)),
        "dropped_features": dropped,
        "n_pairs_scored": int(len(evidence)),
        "n_edges_selected": int(evidence.get("selected", pd.Series(dtype=bool)).sum()),
        "strata": stratum_summary,
        "null_contract": {
            "permutation_unit": (
                "cross-fitted feature residuals within observed state/depth stratum"
            ),
            "preserves_feature_events_per_stratum": True,
            "continuous_state_and_depth_nuisance_cross_fitted": True,
            "preserves_cross_assay_cell_correspondence": False,
            "permutations": int(cfg.n_permutations),
            "one_sided_positive": True,
        },
        "estimand": (
            "positive symmetric within-assay residual detection odds after blocking "
            "on background-derived state and leave-focal-out depth"
        ),
        "orientation": "abstained",
    }
    return StratifiedBetaBinomialResult(
        edge_evidence=evidence,
        stratum_evidence=local_evidence,
        assignments=assignments,
        feature_diagnostics=diagnostics,
        metadata=metadata,
    )


def estimate_from_observed_tables(
    cell_table: Path | str | pd.DataFrame,
    embedding_table: Path | str | pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    cell_column: str = "cell",
    depth_column: str = "observed_background_library_size",
    state_prefix: str = "integration__",
    config: StratifiedBetaBinomialConfig | None = None,
) -> StratifiedBetaBinomialResult:
    """Leakage-safe convenience wrapper for one observed assay table."""

    cells = cell_table.copy() if isinstance(cell_table, pd.DataFrame) else pd.read_csv(cell_table, sep="\t")
    embedding = (
        embedding_table.copy()
        if isinstance(embedding_table, pd.DataFrame)
        else pd.read_csv(embedding_table, sep="\t")
    )
    if cell_column not in cells or cell_column not in embedding:
        raise ValueError(f"Both tables must contain {cell_column!r}")
    if cells[cell_column].duplicated().any() or embedding[cell_column].duplicated().any():
        raise ValueError("Cell identifiers must be unique")
    if depth_column not in cells or not any(
        token in depth_column.lower() for token in ("background", "leave_focal", "leave_locus")
    ):
        raise ValueError("Depth must be an observed background/leave-focal-out column")
    state_columns = [column for column in embedding if str(column).startswith(state_prefix)]
    if not state_columns or any(_is_forbidden_name(column) for column in state_columns):
        raise ValueError("A safe background-derived integration state is required")
    if feature_columns is None:
        feature_columns = [
            str(column)
            for column in cells
            if column not in NON_FEATURE_COLUMNS
            and column != cell_column
            and not _is_forbidden_name(str(column))
            and pd.api.types.is_numeric_dtype(cells[column])
        ]
    feature_columns = [str(column) for column in feature_columns]
    if any(column not in cells for column in feature_columns):
        raise ValueError("A requested feature is missing from the observed cell table")
    if any(_is_forbidden_name(column) for column in feature_columns):
        raise ValueError("Truth/latent/external feature names are forbidden")
    joined = cells[[cell_column, depth_column, *feature_columns]].merge(
        embedding[[cell_column, *state_columns]],
        on=cell_column,
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    result = estimate_stratified_beta_binomial_edges(
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
        "n_cells_after_join": int(len(joined)),
        "truth_pairing_or_external_inputs": False,
    }
    return result


def frame_sha256(frame: pd.DataFrame) -> str:
    """Stable hash used by the prototype freeze manifest."""

    return hashlib.sha256(frame.to_csv(sep="\t", index=False).encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-table", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--permutations", type=int, default=399)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = estimate_from_observed_tables(
        args.cell_table,
        args.embedding_table,
        feature_columns=args.feature or None,
        config=StratifiedBetaBinomialConfig(
            n_permutations=args.permutations,
            random_state=args.seed,
        ),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.edge_evidence.to_csv(args.out_dir / "edge_evidence.tsv", sep="\t", index=False)
    result.stratum_evidence.to_csv(args.out_dir / "stratum_evidence.tsv", sep="\t", index=False)
    result.assignments.to_csv(args.out_dir / "assignments.tsv", sep="\t", index=False)
    result.feature_diagnostics.to_csv(
        args.out_dir / "feature_diagnostics.tsv", sep="\t", index=False
    )
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(result.metadata, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "pairs": result.metadata["n_pairs_scored"],
        "selected": result.metadata["n_edges_selected"],
        "orientation": result.metadata["orientation"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
