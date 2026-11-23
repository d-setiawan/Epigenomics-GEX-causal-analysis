#!/usr/bin/env python3
"""CA-BOSS v4: latent-state decomposition for independently sampled assays.

The discovery contract is intentionally narrower than a counterfactual
single-cell DAG.  Background-only integration coordinates estimate biological
state.  Each observed feature is decomposed into a state-predicted component
and a residual component.  Cross-assay edges can represent shared state or an
unresolved association, but never a learned cell-level causal direction.
Residual evidence may become a direct candidate only when both variables were
measured in the same assay.

Truth, role, DAG, counterfactual, and oracle-state files are forbidden inputs.
The membership reader selects only ``sample_id``, ``modality``, and ``cell``
even when the simulation's scorer-only table contains additional columns.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as student_t
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


FORBIDDEN_INPUT_TOKENS = (
    "counterfactual",
    "generative_dag",
    "ground_truth",
    "cell_ground_truth",
    "oracle",
    "scorer",
    "truth",
    "role_assignment",
    "semantics",
    "truth_edges",
    "true_state",
)
EMBEDDING_PREFIX = "integration__"
STATE_PREFIX = "state__"
EDGE_TYPES = {
    "within_assay_direct_candidate",
    "within_assay_shared_state",
    "cross_assay_shared_state",
    "prior_constrained_peak_gene",
    "unresolved",
}


@dataclass(frozen=True)
class CABossV4Config:
    """Frozen knobs for one CA-BOSS v4 candidate.

    Values here are development defaults, not locked-validation claims.  A
    calibration artifact should replace them after the training split.
    """

    ridge_alpha: float = 2.0
    state_folds: int = 5
    alignment_mode: str = "soft"
    soft_n_microstates: int = 24
    soft_max_rotation_dispersion: float = 0.35
    soft_low_confidence_action: str = "partial_soft"
    min_state_r2: float = 0.05
    min_abs_total_r: float = 0.25
    min_abs_predicted_r: float = 0.75
    min_abs_loading_cosine: float = 0.70
    min_state_score: float = 0.18
    min_state_bootstrap_support: float = 0.70
    residual_nonsignificance_p: float = 0.10
    mutual_state_neighbors: int = 1
    min_abs_direct_residual_r: float = 0.18
    min_residual_to_pair_null95: float = 1.0
    direct_residual_q: float = 0.10
    min_direct_bootstrap_support: float = 0.70
    collider_separation_q: float = 0.20
    hierarchical_microstates: int = 8
    hierarchical_min_microstate_cells: int = 20
    hierarchical_min_peak_events: int = 5
    hierarchical_min_abs_effect: float = 0.20
    hierarchical_probability_threshold: float = 0.88
    hierarchical_positive_effect_only: bool = True
    loading_folds: int = 5
    loading_bootstrap_repetitions: int = 50
    loading_min_norm: float = 0.10
    loading_min_state_fit: float = 0.01
    loading_min_direction_stability: float = 0.50
    loading_min_abs_cosine: float = 0.70
    loading_max_unit_distance: float = 0.85
    loading_min_pair_state_score: float = 0.01
    loading_within_neighbors: int = 2
    loading_cross_neighbors: int = 1
    gene_state_neighbors: int = 1
    suppress_state_projection_through_direct_colliders: bool = True
    group_projection_empirical_q: float = 0.40
    group_projection_min_relative_margin: float = 0.10
    exclude_direct_collider_children_from_state_groups: bool = True
    replace_pairwise_state_layer_with_groups: bool = True
    residual_permutations: int = 199
    stability_bootstraps: int = 200
    null_strata: int = 6
    emit_unresolved: bool = True
    impose_gene_sink_prior: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.alignment_mode not in {"hard", "soft"}:
            raise ValueError("alignment_mode must be 'hard' or 'soft'")
        if self.soft_n_microstates <= 3:
            raise ValueError("soft_n_microstates must exceed the three state dimensions")
        if self.soft_low_confidence_action not in {
            "partial_soft",
            "fallback_hard",
            "abstain",
        }:
            raise ValueError(
                "soft_low_confidence_action must be 'partial_soft', "
                "'fallback_hard', or 'abstain'"
            )
        if (
            not np.isfinite(self.soft_max_rotation_dispersion)
            or self.soft_max_rotation_dispersion < 0.0
        ):
            raise ValueError(
                "soft_max_rotation_dispersion must be finite and non-negative"
            )
        if not 0.0 < self.hierarchical_probability_threshold < 1.0:
            raise ValueError("hierarchical_probability_threshold must lie in (0, 1)")
        if not 0.0 < self.group_projection_empirical_q <= 1.0:
            raise ValueError("group_projection_empirical_q must lie in (0, 1]")
        if not 0.0 <= self.group_projection_min_relative_margin <= 1.0:
            raise ValueError(
                "group_projection_min_relative_margin must lie in [0, 1]"
            )
        if min(
            self.mutual_state_neighbors,
            self.loading_within_neighbors,
            self.loading_cross_neighbors,
            self.gene_state_neighbors,
        ) < 1:
            raise ValueError("state-neighbor budgets must be positive")


@dataclass
class StateDecomposition:
    """Observable feature values split into predicted state and residual parts."""

    observed: pd.DataFrame
    # Out-of-fold state-only contribution, kept separate from depth prediction.
    predicted: pd.DataFrame
    depth_predicted: pd.DataFrame
    nuisance_predicted: pd.DataFrame
    residual: pd.DataFrame
    feature_models: pd.DataFrame
    state_loadings: pd.DataFrame
    state_basis: pd.DataFrame
    strata: np.ndarray
    preprocessing: dict[str, Any]


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def node_channel(node: str) -> str:
    if node.startswith("expr__"):
        return "RNA"
    if "__" in node:
        return node.rsplit("__", 1)[-1]
    return "unknown"


def channel_modality(channel: str) -> str:
    return "rna" if channel == "RNA" else f"chrom_{channel}"


def depth_column(channel: str) -> str:
    suffix = "rna" if channel == "RNA" else channel
    return f"libsize_background__{suffix}"


def is_biological(column: str) -> bool:
    return column.startswith("expr__") or column.startswith("chr")


def _assert_discovery_input(path: Path, *, allow_membership: bool = False) -> None:
    lowered = path.name.lower()
    forbidden = [token for token in FORBIDDEN_INPUT_TOKENS if token in lowered]
    forbidden_directories = {
        "scorer",
        "scorer_only",
        "truth",
        "ground_truth",
        "oracle",
        "counterfactual",
    }
    directory_hits = [
        part
        for part in (piece.lower() for piece in path.parent.parts)
        if part in forbidden_directories
        or part.startswith("scorer__")
        or part.startswith("oracle__")
    ]
    forbidden.extend(directory_hits)
    if forbidden:
        raise ValueError(f"Forbidden discovery input {path}: matched {forbidden}")
    if "membership" in lowered and not allow_membership:
        raise ValueError(f"Membership input is allowed only in the restricted reader: {path}")


def rank_gaussian(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    ranks = rankdata(array, method="average")
    return norm.ppf((ranks - 0.5) / max(len(array), 1))


def safe_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 4 or np.std(left[finite]) <= 1e-12 or np.std(right[finite]) <= 1e-12:
        return 0.0, 1.0
    x = left[finite] - float(np.mean(left[finite]))
    y = right[finite] - float(np.mean(right[finite]))
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-12:
        return 0.0, 1.0
    r_value = float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))
    degrees = int(finite.sum() - 2)
    if degrees <= 0 or abs(r_value) >= 1.0:
        p_value = 0.0 if abs(r_value) >= 1.0 else 1.0
    else:
        statistic = abs(r_value) * math.sqrt(degrees / max(1.0 - r_value**2, 1e-15))
        p_value = float(2.0 * student_t.sf(statistic, degrees))
    return r_value, p_value


def correlation_only(left: np.ndarray, right: np.ndarray) -> float:
    """Fast finite Pearson correlation for permutation/bootstrap inner loops."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 4:
        return 0.0
    x = x[finite] - float(np.mean(x[finite]))
    y = y[finite] - float(np.mean(y[finite]))
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0)) if denominator > 1e-12 else 0.0


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    clean = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    order = np.argsort(clean)
    ranked = clean[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _read_columns(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()


def load_state_centroids(
    embedding_path: Path,
    membership_path: Path,
    sample_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate allowed background coordinates to metacell-by-assay centroids."""

    _assert_discovery_input(embedding_path)
    _assert_discovery_input(membership_path, allow_membership=True)
    embedding_columns = _read_columns(embedding_path)
    coordinate_columns = sorted(
        column for column in embedding_columns if column.startswith(EMBEDDING_PREFIX)
    )
    required_embedding = ["modality", "cell", *coordinate_columns]
    missing_embedding = set(required_embedding) - set(embedding_columns)
    if missing_embedding or not coordinate_columns:
        raise ValueError(
            f"Embedding table lacks allowed coordinates: missing={sorted(missing_embedding)}"
        )
    embeddings = pd.read_csv(
        embedding_path,
        sep="\t",
        usecols=required_embedding,
    )
    membership_columns = _read_columns(membership_path)
    allowed_membership = ["sample_id", "modality", "cell"]
    missing_membership = set(allowed_membership) - set(membership_columns)
    if missing_membership:
        raise ValueError(f"Membership table missing {sorted(missing_membership)}")
    membership = pd.read_csv(
        membership_path,
        sep="\t",
        usecols=allowed_membership,
    )
    membership = membership.loc[membership["sample_id"].astype(str).isin(sample_ids)]
    merged = membership.merge(
        embeddings,
        on=["modality", "cell"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No membership rows matched the background embedding table")
    centroids = (
        merged.groupby(["sample_id", "modality"], sort=False)[coordinate_columns]
        .mean()
        .reset_index()
    )
    wide = centroids.pivot(index="sample_id", columns="modality", values=coordinate_columns)
    flattened = pd.DataFrame(index=wide.index)
    modalities = sorted(centroids["modality"].astype(str).unique())
    for coordinate in coordinate_columns:
        for modality in modalities:
            key = (coordinate, modality)
            if key in wide.columns:
                dim = coordinate.removeprefix(EMBEDDING_PREFIX)
                flattened[f"{STATE_PREFIX}{modality}__{dim}"] = wide[key]
        matching = [column for column in flattened if column.endswith(f"__{coordinate.removeprefix(EMBEDDING_PREFIX)}")]
        flattened[f"{STATE_PREFIX}consensus__{coordinate.removeprefix(EMBEDDING_PREFIX)}"] = (
            flattened[matching].mean(axis=1)
        )
    assay_columns = [column for column in flattened if not column.startswith(f"{STATE_PREFIX}consensus")]
    flattened[f"{STATE_PREFIX}assay_uncertainty"] = flattened[assay_columns].std(axis=1, ddof=0)
    flattened = flattened.reindex(sample_ids)
    if flattened.isna().all(axis=1).any():
        absent = flattened.index[flattened.isna().all(axis=1)].tolist()
        raise ValueError(f"State centroids are absent for samples: {absent[:8]}")
    flattened = flattened.apply(lambda column: column.fillna(column.mean()))
    return flattened.reset_index(names="sample_id"), {
        "embedding_input": str(embedding_path),
        "membership_input": str(membership_path),
        "embedding_columns_read": required_embedding,
        "membership_columns_read": allowed_membership,
        "n_joined_cells": int(len(merged)),
        "n_modalities": int(len(modalities)),
        "modalities": modalities,
        "n_state_coordinates": int(len(coordinate_columns)),
    }


def _standardized_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if frame.empty:
        return np.ones((len(frame), 1), dtype=float), ["intercept_only"]
    values = frame.to_numpy(dtype=float)
    live = np.nanstd(values, axis=0) > 1e-12
    columns = frame.columns[np.asarray(live)].astype(str).tolist()
    if not columns:
        return np.ones((len(frame), 1), dtype=float), ["intercept_only"]
    values = StandardScaler().fit_transform(frame[columns].to_numpy(dtype=float))
    return values, columns


def _optional_standardized_design(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Standardize live columns while allowing an intercept-only empty design."""

    if frame.shape[1] == 0:
        return np.empty((len(frame), 0), dtype=float), []
    values = frame.to_numpy(dtype=float)
    live = np.nanstd(values, axis=0) > 1e-12
    columns = frame.columns[np.asarray(live)].astype(str).tolist()
    if not columns:
        return np.empty((len(frame), 0), dtype=float), []
    return StandardScaler().fit_transform(frame[columns].to_numpy(dtype=float)), columns


def _feature_design_components(
    node: str,
    observed: pd.DataFrame,
    state_basis: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    channel = node_channel(node)
    modality = channel_modality(channel)
    state_columns = [
        column
        for column in state_basis.columns
        if column.startswith(f"{STATE_PREFIX}{modality}__")
        and not column.endswith("assay_uncertainty")
    ]
    if not state_columns:
        state_columns = [
            column for column in state_basis.columns if column.startswith(f"{STATE_PREFIX}consensus__")
        ]
    state_frame = state_basis[state_columns].copy()
    state_frame = state_frame.apply(lambda column: column.fillna(column.mean()))
    state_design, state_names = _optional_standardized_design(state_frame)
    depth_frame = pd.DataFrame(index=observed.index)
    depth = depth_column(channel)
    if depth in observed.columns:
        depth_frame[f"depth__{channel}"] = np.log1p(
            pd.to_numeric(observed[depth], errors="coerce").to_numpy(dtype=float)
        )
    depth_frame = depth_frame.apply(lambda column: column.fillna(column.mean()))
    depth_design, depth_names = _optional_standardized_design(depth_frame)
    if state_design.shape[1] and depth_design.shape[1]:
        # Background depth contains both technical sampling variation and real
        # state-dependent abundance.  Allocate their shared linear component to
        # biological state, then use only the state-orthogonal depth residual as
        # nuisance.  This avoids conditioning away biology while preventing a
        # purely technical depth axis from masquerading as state.
        augmented_state = np.column_stack([np.ones(len(state_design)), state_design])
        projection, *_ = np.linalg.lstsq(augmented_state, depth_design, rcond=None)
        depth_design = depth_design - augmented_state @ projection
        depth_scale = np.std(depth_design, axis=0, ddof=0)
        live_depth = depth_scale > 1e-12
        depth_design = depth_design[:, live_depth]
        depth_names = [
            name for name, live in zip(depth_names, live_depth, strict=True) if live
        ]
        if depth_design.shape[1]:
            depth_design = depth_design / np.std(depth_design, axis=0, ddof=0)
    return state_design, state_names, depth_design, depth_names


def _cross_fitted_ridge(
    design: np.ndarray,
    response: np.ndarray,
    *,
    alpha: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n_rows = len(response)
    if n_rows < 4:
        raise ValueError("At least four complete metacells are required for state decomposition")
    n_splits = min(max(2, folds), max(2, n_rows // 3))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.zeros(n_rows, dtype=float)
    for train, test in splitter.split(design):
        if design.shape[1] == 0:
            predictions[test] = float(np.mean(response[train]))
            continue
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(design[train], response[train])
        predictions[test] = model.predict(design[test])
    residual = response - predictions
    denominator = float(np.sum((response - np.mean(response)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-12)
    if design.shape[1] == 0:
        coefficients = np.empty(0, dtype=float)
    else:
        full_model = Ridge(alpha=float(alpha), fit_intercept=True).fit(design, response)
        coefficients = np.asarray(full_model.coef_, dtype=float)
    return predictions, residual, float(r2), coefficients


def _make_null_strata(
    observed: pd.DataFrame,
    state_basis: pd.DataFrame,
    *,
    requested: int,
    seed: int,
) -> np.ndarray:
    consensus = [
        column for column in state_basis.columns if column.startswith(f"{STATE_PREFIX}consensus__")
    ]
    nuisance = state_basis[consensus].copy()
    for column in observed.columns:
        if column.startswith("libsize_background__"):
            nuisance[column] = np.log1p(pd.to_numeric(observed[column], errors="coerce"))
    nuisance = nuisance.apply(lambda column: column.fillna(column.mean()))
    values, _ = _standardized_design(nuisance)
    n_clusters = min(max(1, int(requested)), max(1, len(values) // 4))
    if n_clusters <= 1:
        return np.zeros(len(values), dtype=int)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(values)


def decompose_features(
    observed: pd.DataFrame,
    state_basis: pd.DataFrame,
    cfg: CABossV4Config,
) -> StateDecomposition:
    """Cross-fit every biological feature on assay state and background depth."""

    features = [column for column in observed.columns if is_biological(str(column))]
    numeric = observed[features].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1) & state_basis.notna().all(axis=1)
    for column in observed.columns:
        if column.startswith("libsize_background__"):
            complete &= pd.to_numeric(observed[column], errors="coerce").notna()
    observed_complete = observed.loc[complete].reset_index(drop=True)
    state_complete = state_basis.loc[complete].reset_index(drop=True)
    transformed: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for feature in features:
        values = pd.to_numeric(observed_complete[feature], errors="coerce").to_numpy(dtype=float)
        if np.std(values) <= 1e-12:
            dropped.append(feature)
            continue
        transformed[feature] = rank_gaussian(values)
    feature_frame = pd.DataFrame(transformed)
    predictions = pd.DataFrame(index=feature_frame.index)
    depth_predictions = pd.DataFrame(index=feature_frame.index)
    nuisance_predictions = pd.DataFrame(index=feature_frame.index)
    residuals = pd.DataFrame(index=feature_frame.index)
    model_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    for feature_i, feature in enumerate(feature_frame.columns):
        state_design, state_columns, depth_design, depth_columns = _feature_design_components(
            feature, observed_complete, state_complete
        )
        joint_design = np.column_stack([depth_design, state_design])
        joint_columns = [*depth_columns, *state_columns]
        state_predicted, _, state_r2, state_coefficients = _cross_fitted_ridge(
            state_design,
            feature_frame[feature].to_numpy(dtype=float),
            alpha=cfg.ridge_alpha,
            folds=cfg.state_folds,
            seed=cfg.seed + feature_i,
        )
        depth_predicted, depth_residual, depth_r2, _ = _cross_fitted_ridge(
            depth_design,
            feature_frame[feature].to_numpy(dtype=float),
            alpha=cfg.ridge_alpha,
            folds=cfg.state_folds,
            seed=cfg.seed + feature_i,
        )
        joint_predicted, residual, joint_r2, joint_coefficients = _cross_fitted_ridge(
            joint_design,
            feature_frame[feature].to_numpy(dtype=float),
            alpha=cfg.ridge_alpha,
            folds=cfg.state_folds,
            seed=cfg.seed + feature_i,
        )
        sse_depth = float(np.sum(np.square(depth_residual)))
        sse_joint = float(np.sum(np.square(residual)))
        partial_state_r2 = max(0.0, (sse_depth - sse_joint) / max(sse_depth, 1e-12))
        predictions[feature] = state_predicted
        depth_predictions[feature] = depth_predicted
        nuisance_predictions[feature] = joint_predicted
        residuals[feature] = residual
        model_rows.append(
            {
                "feature": feature,
                "channel": node_channel(feature),
                "n_rows": int(len(feature_frame)),
                "oof_state_r2": float(state_r2),
                "oof_partial_state_r2": float(partial_state_r2),
                "oof_depth_r2": float(depth_r2),
                "oof_joint_r2": float(joint_r2),
                "observed_sd": float(feature_frame[feature].std(ddof=0)),
                "state_predicted_sd": float(np.std(state_predicted)),
                "depth_predicted_sd": float(np.std(depth_predicted)),
                "joint_predicted_sd": float(np.std(joint_predicted)),
                "residual_sd": float(np.std(residual)),
                "state_design_columns": ",".join(state_columns),
                "depth_design_columns": ",".join(depth_columns),
                "joint_design_columns": ",".join(joint_columns),
            }
        )
        conditional_state_coefficients = joint_coefficients[len(depth_columns) :]
        for column, coefficient, marginal_coefficient in zip(
            state_columns,
            conditional_state_coefficients,
            state_coefficients,
            strict=True,
        ):
            loading_rows.append(
                {
                    "feature": feature,
                    "channel": node_channel(feature),
                    "state_coordinate": column,
                    "coefficient": float(coefficient),
                    "marginal_state_coefficient": float(marginal_coefficient),
                    "loading_estimand": "state coefficient conditional on background depth",
                }
            )
    strata = _make_null_strata(
        observed_complete,
        state_complete,
        requested=cfg.null_strata,
        seed=cfg.seed,
    )
    return StateDecomposition(
        observed=feature_frame,
        predicted=predictions,
        depth_predicted=depth_predictions,
        nuisance_predicted=nuisance_predictions,
        residual=residuals,
        feature_models=pd.DataFrame(model_rows),
        state_loadings=pd.DataFrame(loading_rows),
        state_basis=state_complete,
        strata=strata,
        preprocessing={
            "n_input_rows": int(len(observed)),
            "n_complete_rows": int(len(observed_complete)),
            "n_features": int(len(feature_frame.columns)),
            "dropped_constant_features": dropped,
            "n_null_strata": int(len(np.unique(strata))),
            "focal_features_excluded_from_state_estimation": True,
            "state_component": "OOF prediction from background-derived state coordinates only",
            "depth_component": "OOF prediction from the state-orthogonal component of leave-locus-out background depth",
            "residual_component": "observed minus OOF joint(state+depth) prediction",
            "state_depth_separated": True,
            "state_depth_shared_variation_allocated_to_state": True,
        },
    )


def _loading_vector(
    feature: str,
    loadings: pd.DataFrame,
    coordinate_suffixes: list[str],
) -> np.ndarray:
    subset = loadings.loc[loadings["feature"] == feature]
    mapping = {
        str(row["state_coordinate"]).rsplit("__", 1)[-1]: float(row["coefficient"])
        for row in subset.to_dict(orient="records")
    }
    return np.asarray([mapping.get(suffix, 0.0) for suffix in coordinate_suffixes], dtype=float)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _stratified_permutation_p(
    left: np.ndarray,
    right: np.ndarray,
    strata: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    observed_r = correlation_only(left, right)
    if permutations <= 0:
        return 1.0, 0.0, 0.0
    null_values = np.zeros(permutations, dtype=float)
    labels = np.unique(strata)
    for repetition in range(permutations):
        permuted = np.asarray(right, dtype=float).copy()
        for label in labels:
            indices = np.flatnonzero(strata == label)
            if indices.size > 1:
                permuted[indices] = right[rng.permutation(indices)]
        null_values[repetition] = abs(correlation_only(left, permuted))
    p_value = (1.0 + float(np.sum(null_values >= abs(observed_r)))) / (permutations + 1.0)
    return p_value, float(np.quantile(null_values, 0.95)), float(np.mean(null_values))


def _bootstrap_support(
    total_left: np.ndarray,
    total_right: np.ndarray,
    predicted_left: np.ndarray,
    predicted_right: np.ndarray,
    residual_left: np.ndarray,
    residual_right: np.ndarray,
    *,
    state_score_scale: float,
    cfg: CABossV4Config,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if cfg.stability_bootstraps <= 0:
        return 1.0, 1.0
    n_rows = len(total_left)
    observed_total = correlation_only(total_left, total_right)
    observed_predicted = correlation_only(predicted_left, predicted_right)
    observed_residual = correlation_only(residual_left, residual_right)
    observed_state_score = state_score_scale * abs(observed_predicted)
    if abs(observed_total) <= 1e-12 or abs(observed_predicted) <= 1e-12:
        state_observable = False
    else:
        state_observable = True
    direct_observable = abs(observed_residual) > 1e-12
    state_hits = 0
    direct_hits = 0
    for _ in range(cfg.stability_bootstraps):
        index = rng.integers(0, n_rows, size=n_rows)
        total_r = correlation_only(total_left[index], total_right[index])
        predicted_r = correlation_only(predicted_left[index], predicted_right[index])
        residual_r = correlation_only(residual_left[index], residual_right[index])
        state_score = state_score_scale * abs(predicted_r)
        state_hits += int(
            state_observable
            and total_r * observed_total > 0
            and predicted_r * observed_predicted > 0
            and abs(total_r) >= 0.5 * abs(observed_total)
            and state_score >= 0.5 * observed_state_score
        )
        direct_hits += int(
            direct_observable
            and residual_r * observed_residual > 0
            and abs(residual_r) >= 0.5 * abs(observed_residual)
        )
    return (
        state_hits / cfg.stability_bootstraps,
        direct_hits / cfg.stability_bootstraps,
    )


def build_pair_evidence(
    decomposition: StateDecomposition,
    cfg: CABossV4Config,
) -> pd.DataFrame:
    """Build pairwise total, state-explained, residual, and stability evidence."""

    models = decomposition.feature_models.set_index("feature")
    model_r2 = models["oof_partial_state_r2"].to_dict()
    marginal_model_r2 = models["oof_state_r2"].to_dict()
    depth_model_r2 = models["oof_depth_r2"].to_dict()
    suffixes = sorted(
        {
            str(value).rsplit("__", 1)[-1]
            for value in decomposition.state_loadings.get(
                "state_coordinate", pd.Series(dtype=str)
            )
        }
    )
    rows: list[dict[str, Any]] = []
    for pair_i, (left, right) in enumerate(combinations(decomposition.observed.columns, 2)):
        rng = np.random.default_rng(cfg.seed + 100_003 + pair_i)
        total_left = decomposition.observed[left].to_numpy(dtype=float)
        total_right = decomposition.observed[right].to_numpy(dtype=float)
        predicted_left = decomposition.predicted[left].to_numpy(dtype=float)
        predicted_right = decomposition.predicted[right].to_numpy(dtype=float)
        depth_left = decomposition.depth_predicted[left].to_numpy(dtype=float)
        depth_right = decomposition.depth_predicted[right].to_numpy(dtype=float)
        nuisance_left = decomposition.nuisance_predicted[left].to_numpy(dtype=float)
        nuisance_right = decomposition.nuisance_predicted[right].to_numpy(dtype=float)
        residual_left = decomposition.residual[left].to_numpy(dtype=float)
        residual_right = decomposition.residual[right].to_numpy(dtype=float)
        total_r, total_p = safe_correlation(total_left, total_right)
        predicted_r, predicted_p = safe_correlation(predicted_left, predicted_right)
        depth_predicted_r, _ = safe_correlation(depth_left, depth_right)
        nuisance_predicted_r, _ = safe_correlation(nuisance_left, nuisance_right)
        residual_r, residual_p = safe_correlation(residual_left, residual_right)
        loading_left = _loading_vector(left, decomposition.state_loadings, suffixes)
        loading_right = _loading_vector(right, decomposition.state_loadings, suffixes)
        loading_cosine = _cosine(loading_left, loading_right)
        left_r2 = max(0.0, float(model_r2.get(left, 0.0)))
        right_r2 = max(0.0, float(model_r2.get(right, 0.0)))
        left_marginal_r2 = max(0.0, float(marginal_model_r2.get(left, 0.0)))
        right_marginal_r2 = max(0.0, float(marginal_model_r2.get(right, 0.0)))
        left_depth_r2 = max(0.0, float(depth_model_r2.get(left, 0.0)))
        right_depth_r2 = max(0.0, float(depth_model_r2.get(right, 0.0)))
        state_scale = math.sqrt(left_r2 * right_r2) * abs(loading_cosine)
        state_score = abs(predicted_r) * state_scale
        permutation_p, residual_null_95, residual_null_mean = _stratified_permutation_p(
            residual_left,
            residual_right,
            decomposition.strata,
            permutations=cfg.residual_permutations,
            rng=rng,
        )
        state_support, direct_support = _bootstrap_support(
            total_left,
            total_right,
            predicted_left,
            predicted_right,
            residual_left,
            residual_right,
            state_score_scale=state_scale,
            cfg=cfg,
            rng=rng,
        )
        covariance_total = float(np.cov(total_left, total_right, ddof=0)[0, 1])
        covariance_state = float(np.cov(predicted_left, predicted_right, ddof=0)[0, 1])
        explained_fraction = min(
            1.0,
            abs(covariance_state) / max(abs(covariance_total), 1e-12),
        )
        rows.append(
            {
                "node1": left,
                "node2": right,
                "channel1": node_channel(left),
                "channel2": node_channel(right),
                "same_assay": node_channel(left) == node_channel(right),
                "total_r": total_r,
                "total_p": total_p,
                "predicted_r": predicted_r,
                "predicted_p": predicted_p,
                "depth_predicted_r": depth_predicted_r,
                "joint_nuisance_predicted_r": nuisance_predicted_r,
                "residual_r": residual_r,
                "residual_parametric_p": residual_p,
                "residual_empirical_p": permutation_p,
                "residual_null_abs_r_95": residual_null_95,
                "residual_null_abs_r_mean": residual_null_mean,
                "residual_to_pair_null95": abs(residual_r)
                / max(residual_null_95, 1e-12),
                "loading_cosine": loading_cosine,
                "left_state_r2": left_r2,
                "right_state_r2": right_r2,
                "left_marginal_state_r2": left_marginal_r2,
                "right_marginal_state_r2": right_marginal_r2,
                "left_depth_r2": left_depth_r2,
                "right_depth_r2": right_depth_r2,
                "state_score": state_score,
                "state_explained_covariance_fraction": explained_fraction,
                "state_bootstrap_support": state_support,
                "direct_bootstrap_support": direct_support,
            }
        )
    evidence = pd.DataFrame(rows)
    # Direct residual structure is an assay-local estimand.  Combining its 15
    # tests with hundreds of unsupported cross-assay residual tests makes the
    # finite-permutation p-value floor dominate every within-assay q-value.
    # Adjust within each assay (and each cross-assay pair for unresolved calls),
    # then let the world-level sentinel calibration control graph-wide errors.
    evidence["residual_multiplicity_family"] = evidence.apply(
        lambda row: (
            f"within::{row['channel1']}"
            if bool(row["same_assay"])
            else "cross::" + "|".join(sorted((str(row["channel1"]), str(row["channel2"]))))
        ),
        axis=1,
    )
    evidence["residual_empirical_q"] = 1.0
    evidence["residual_family_size"] = 0
    for _, indices in evidence.groupby("residual_multiplicity_family").groups.items():
        index = list(indices)
        evidence.loc[index, "residual_empirical_q"] = benjamini_hochberg(
            evidence.loc[index, "residual_empirical_p"]
        )
        evidence.loc[index, "residual_family_size"] = len(index)
    evidence["residual_family_size"] = evidence["residual_family_size"].astype(int)
    return apply_evidence_thresholds(evidence, cfg)


def attach_cell_loading_evidence(
    evidence: pd.DataFrame,
    loading_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Attach complete cell-level loading evidence without cross-assay row joins.

    ``loading_pairs`` is produced from independently fitted assay-local cell
    models.  Only canonical feature-pair keys are joined here; no cell or
    metacell rows are compared across assays.  A complete one-row-per-pair
    contract prevents a partial external table from silently manufacturing
    stability for a hand-picked subset.
    """

    required = {
        "node1",
        "node2",
        "loading_cosine",
        "absolute_loading_cosine",
        "sign_invariant_unit_loading_distance",
        "minimum_loading_norm",
        "node1_oof_marginal_state_fit",
        "node2_oof_marginal_state_fit",
        "node1_oof_partial_state_fit",
        "node2_oof_partial_state_fit",
        "node1_loading_stability",
        "node2_loading_stability",
        "loading_stability_support",
        "pair_state_score",
        "loading_empirical_null_p",
        "loading_empirical_null_q",
        "unit_distance_null_percentile",
    }
    missing = sorted(required - set(loading_pairs.columns))
    if missing:
        raise ValueError(f"Cell-loading pair evidence lacks required columns: {missing}")
    loading_map: dict[tuple[str, str], dict[str, Any]] = {}
    for record in loading_pairs.to_dict(orient="records"):
        first, second = str(record["node1"]), str(record["node2"])
        key = canonical_edge(first, second)
        if key in loading_map:
            raise ValueError(f"Duplicate cell-loading pair evidence for {key}")
        loading_map[key] = {
            "record": record,
            "node_values": {
                first: {
                    "marginal_fit": float(record["node1_oof_marginal_state_fit"]),
                    "partial_fit": float(record["node1_oof_partial_state_fit"]),
                    "stability": float(record["node1_loading_stability"]),
                },
                second: {
                    "marginal_fit": float(record["node2_oof_marginal_state_fit"]),
                    "partial_fit": float(record["node2_oof_partial_state_fit"]),
                    "stability": float(record["node2_loading_stability"]),
                },
            },
        }
    evidence_keys = {
        canonical_edge(str(row["node1"]), str(row["node2"]))
        for row in evidence.to_dict(orient="records")
    }
    if evidence_keys != set(loading_map):
        missing_pairs = sorted(evidence_keys - set(loading_map))[:5]
        extra_pairs = sorted(set(loading_map) - evidence_keys)[:5]
        raise ValueError(
            "Cell-loading evidence must cover exactly the metacell focal-pair universe; "
            f"missing={missing_pairs}, extra={extra_pairs}"
        )

    output = evidence.copy()
    attached: dict[str, list[Any]] = {
        name: []
        for name in (
            "cell_loading_cosine",
            "cell_absolute_loading_cosine",
            "cell_sign_invariant_unit_loading_distance",
            "cell_minimum_loading_norm",
            "cell_left_oof_marginal_state_fit",
            "cell_right_oof_marginal_state_fit",
            "cell_left_oof_partial_state_fit",
            "cell_right_oof_partial_state_fit",
            "cell_left_loading_stability",
            "cell_right_loading_stability",
            "cell_loading_stability_support",
            "cell_pair_state_score",
            "cell_loading_empirical_null_p",
            "cell_loading_empirical_null_q",
            "cell_unit_distance_null_percentile",
            "cell_loading_default_selected_sparse",
        )
    }
    for row in output.to_dict(orient="records"):
        left, right = str(row["node1"]), str(row["node2"])
        item = loading_map[canonical_edge(left, right)]
        record = item["record"]
        node_values = item["node_values"]
        attached["cell_loading_cosine"].append(float(record["loading_cosine"]))
        attached["cell_absolute_loading_cosine"].append(
            float(record["absolute_loading_cosine"])
        )
        attached["cell_sign_invariant_unit_loading_distance"].append(
            float(record["sign_invariant_unit_loading_distance"])
        )
        attached["cell_minimum_loading_norm"].append(float(record["minimum_loading_norm"]))
        attached["cell_left_oof_marginal_state_fit"].append(
            node_values[left]["marginal_fit"]
        )
        attached["cell_right_oof_marginal_state_fit"].append(
            node_values[right]["marginal_fit"]
        )
        attached["cell_left_oof_partial_state_fit"].append(
            node_values[left]["partial_fit"]
        )
        attached["cell_right_oof_partial_state_fit"].append(
            node_values[right]["partial_fit"]
        )
        attached["cell_left_loading_stability"].append(node_values[left]["stability"])
        attached["cell_right_loading_stability"].append(node_values[right]["stability"])
        attached["cell_loading_stability_support"].append(
            float(record["loading_stability_support"])
        )
        attached["cell_pair_state_score"].append(float(record["pair_state_score"]))
        attached["cell_loading_empirical_null_p"].append(
            float(record["loading_empirical_null_p"])
        )
        attached["cell_loading_empirical_null_q"].append(
            float(record["loading_empirical_null_q"])
        )
        attached["cell_unit_distance_null_percentile"].append(
            float(record["unit_distance_null_percentile"])
        )
        attached["cell_loading_default_selected_sparse"].append(
            bool(record.get("selected_sparse", False))
        )
    for name, values in attached.items():
        output[name] = values
    output["cell_loading_evidence_available"] = True
    return output


def apply_evidence_thresholds(
    evidence: pd.DataFrame,
    cfg: CABossV4Config,
) -> pd.DataFrame:
    """Apply a candidate's selection thresholds to cached raw pair evidence."""

    output = evidence.copy()
    output["metacell_state_eligible"] = (
        (output["left_state_r2"] >= cfg.min_state_r2)
        & (output["right_state_r2"] >= cfg.min_state_r2)
        & (output["total_r"].abs() >= cfg.min_abs_total_r)
        & (output["predicted_r"].abs() >= cfg.min_abs_predicted_r)
        & (output["loading_cosine"].abs() >= cfg.min_abs_loading_cosine)
        & (output["state_score"] >= cfg.min_state_score)
        & (output["state_bootstrap_support"] >= cfg.min_state_bootstrap_support)
        & (output["residual_empirical_p"] >= cfg.residual_nonsignificance_p)
    )
    if "cell_loading_evidence_available" in output:
        output["cell_loading_state_eligible"] = (
            output["cell_loading_evidence_available"].astype(bool)
            & (output["cell_minimum_loading_norm"] >= cfg.loading_min_norm)
            & (output["cell_left_oof_partial_state_fit"] >= cfg.loading_min_state_fit)
            & (output["cell_right_oof_partial_state_fit"] >= cfg.loading_min_state_fit)
            & (
                output["cell_left_loading_stability"]
                >= cfg.loading_min_direction_stability
            )
            & (
                output["cell_right_loading_stability"]
                >= cfg.loading_min_direction_stability
            )
            & (
                output["cell_absolute_loading_cosine"]
                >= cfg.loading_min_abs_cosine
            )
            & (
                output["cell_sign_invariant_unit_loading_distance"]
                <= cfg.loading_max_unit_distance
            )
            & (output["cell_pair_state_score"] >= cfg.loading_min_pair_state_score)
            & (
                output["cell_loading_stability_support"]
                >= cfg.loading_min_direction_stability
            )
        )
    else:
        output["cell_loading_state_eligible"] = False
    output["state_eligible"] = (
        output["metacell_state_eligible"] | output["cell_loading_state_eligible"]
    )
    output["direct_eligible"] = (
        output["same_assay"]
        & (output["residual_r"].abs() >= cfg.min_abs_direct_residual_r)
        & (
            output["residual_to_pair_null95"]
            >= cfg.min_residual_to_pair_null95
        )
        & (output["residual_empirical_q"] <= cfg.direct_residual_q)
        & (output["direct_bootstrap_support"] >= cfg.min_direct_bootstrap_support)
    )
    return output


def _hierarchical_selection(
    hierarchy: pd.DataFrame | None,
    *,
    probability_threshold: float,
    positive_effect_only: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    if hierarchy is None or hierarchy.empty:
        return {}
    left_name = next(
        (column for column in ("node1", "feature1", "feature_a", "left") if column in hierarchy),
        None,
    )
    right_name = next(
        (column for column in ("node2", "feature2", "feature_b", "right") if column in hierarchy),
        None,
    )
    if left_name is None or right_name is None:
        raise ValueError("Hierarchical evidence requires two feature-name columns")
    selected_column = next(
        (
            column
            for column in (
                "selected",
                "edge_selected",
                "posterior_selected",
                "is_direct_candidate",
            )
            if column in hierarchy
        ),
        None,
    )
    probability_candidates = (
        ("posterior_probability_positive",)
        if positive_effect_only
        else (
            "posterior_probability",
            "posterior_prob_gt_delta",
            "posterior_probability_residual_dependence",
            "probability_nonzero",
            "prob_abs_effect_gt_delta",
        )
    )
    probability_column = next(
        (column for column in probability_candidates if column in hierarchy),
        None,
    )
    if positive_effect_only and probability_column is None and selected_column is not None:
        raise ValueError(
            "Positive-only hierarchical selection requires posterior_probability_positive"
        )
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for record in hierarchy.to_dict(orient="records"):
        selected = False
        if probability_column is not None:
            selected = float(record.get(probability_column, 0.0)) >= probability_threshold
        elif selected_column is not None:
            selected = bool(record.get(selected_column, False))
        if selected:
            key = canonical_edge(str(record[left_name]), str(record[right_name]))
            mapping[key] = {f"hierarchical_{key}": value for key, value in record.items()}
    return mapping


def _mutual_state_matches(
    evidence: pd.DataFrame,
    neighbors: int,
    *,
    eligibility_column: str = "state_eligible",
    score_column: str = "state_score",
) -> set[tuple[str, str]]:
    eligible = evidence.loc[evidence[eligibility_column].astype(bool)].copy()
    if eligible.empty:
        return set()
    choices: dict[tuple[str, str], set[str]] = {}
    channels = sorted(set(eligible["channel1"]) | set(eligible["channel2"]))
    nodes_by_channel: dict[str, set[str]] = {channel: set() for channel in channels}
    for row in eligible.to_dict(orient="records"):
        nodes_by_channel[str(row["channel1"])].add(str(row["node1"]))
        nodes_by_channel[str(row["channel2"])].add(str(row["node2"]))
    for source_channel, target_channel in combinations(channels, 2):
        pair = eligible.loc[
            eligible.apply(
                lambda row: {row["channel1"], row["channel2"]}
                == {source_channel, target_channel},
                axis=1,
            )
        ]
        for source in nodes_by_channel[source_channel]:
            candidates: list[tuple[float, str]] = []
            for row in pair.to_dict(orient="records"):
                if row["node1"] == source:
                    candidates.append((float(row[score_column]), str(row["node2"])))
                elif row["node2"] == source:
                    candidates.append((float(row[score_column]), str(row["node1"])))
            choices[(source, target_channel)] = {
                node for _, node in sorted(candidates, reverse=True)[: max(1, neighbors)]
            }
        for source in nodes_by_channel[target_channel]:
            candidates = []
            for row in pair.to_dict(orient="records"):
                if row["node1"] == source:
                    candidates.append((float(row[score_column]), str(row["node2"])))
                elif row["node2"] == source:
                    candidates.append((float(row[score_column]), str(row["node1"])))
            choices[(source, source_channel)] = {
                node for _, node in sorted(candidates, reverse=True)[: max(1, neighbors)]
            }
    for channel in channels:
        pair = eligible.loc[
            eligible["same_assay"]
            & eligible["channel1"].eq(channel)
            & eligible["channel2"].eq(channel)
        ]
        channel_nodes = nodes_by_channel[channel]
        for source in channel_nodes:
            candidates = []
            for row in pair.to_dict(orient="records"):
                if row["node1"] == source:
                    candidates.append((float(row[score_column]), str(row["node2"])))
                elif row["node2"] == source:
                    candidates.append((float(row[score_column]), str(row["node1"])))
            choices[(source, channel)] = {
                node for _, node in sorted(candidates, reverse=True)[: max(1, neighbors)]
            }
    selected: set[tuple[str, str]] = set()
    for row in eligible.to_dict(orient="records"):
        left, right = str(row["node1"]), str(row["node2"])
        left_channel, right_channel = str(row["channel1"]), str(row["channel2"])
        if (
            right in choices.get((left, right_channel), set())
            and left in choices.get((right, left_channel), set())
        ):
            selected.add(canonical_edge(left, right))
    return selected


def _global_cross_state_matches(
    evidence: pd.DataFrame,
    neighbors: int,
    *,
    eligibility_column: str,
    score_column: str,
) -> set[tuple[str, str]]:
    """Return a sparse mutual-neighbor graph across all other assays.

    Ranking separately for every assay pair creates at least one edge for most
    channel pairs even under a null.  A global cross-assay neighborhood gives
    every feature one small evidence budget and therefore represents a latent
    loading group as a sparse graph rather than a dense clique.
    """

    eligible = evidence.loc[
        evidence[eligibility_column].astype(bool) & ~evidence["same_assay"].astype(bool)
    ].copy()
    eligible = eligible.loc[
        ~eligible["node1"].astype(str).str.startswith("expr__")
        & ~eligible["node2"].astype(str).str.startswith("expr__")
    ].copy()
    if eligible.empty:
        return set()
    ranks: dict[tuple[str, str], int] = {}
    nodes = sorted(set(eligible["node1"].astype(str)) | set(eligible["node2"].astype(str)))
    for node in nodes:
        candidates = eligible.loc[
            eligible["node1"].eq(node) | eligible["node2"].eq(node)
        ].copy()
        candidates["__neighbor"] = candidates.apply(
            lambda row: str(row["node2"]) if str(row["node1"]) == node else str(row["node1"]),
            axis=1,
        )
        candidates = candidates.sort_values(
            [score_column, "__neighbor"],
            ascending=[False, True],
        )
        for rank, row in enumerate(candidates.to_dict(orient="records"), start=1):
            ranks[(node, str(row["__neighbor"]))] = rank
    selected: set[tuple[str, str]] = set()
    for row in eligible.to_dict(orient="records"):
        left, right = str(row["node1"]), str(row["node2"])
        if (
            0 < ranks.get((left, right), 0) <= max(1, neighbors)
            and 0 < ranks.get((right, left), 0) <= max(1, neighbors)
        ):
            selected.add(canonical_edge(left, right))
    return selected


def _global_gene_state_matches(
    evidence: pd.DataFrame,
    neighbors: int,
    *,
    eligibility_column: str,
    score_column: str,
) -> set[tuple[str, str]]:
    """Select a small global peak budget for each expression feature.

    Requiring a mutual nearest neighbor is inappropriate for a gene whose best
    state-matched peak may itself be closer to a same-loading peak.  Conversely,
    ranking separately by histone channel gives each gene one edge per assay and
    recreates the dense graph v4 is intended to avoid.  This selector therefore
    gives each expression feature one global, uncertainty-gated peak budget.
    Its arrowhead remains explicitly prior-constrained rather than learned.
    """

    eligible = evidence.loc[
        evidence[eligibility_column].astype(bool)
        & ~evidence["same_assay"].astype(bool)
        & (
            evidence["node1"].astype(str).str.startswith("expr__")
            ^ evidence["node2"].astype(str).str.startswith("expr__")
        )
    ].copy()
    if eligible.empty:
        return set()
    selected: set[tuple[str, str]] = set()
    genes = sorted(
        {
            node
            for column in ("node1", "node2")
            for node in eligible[column].astype(str)
            if node.startswith("expr__")
        }
    )
    for gene in genes:
        candidates = eligible.loc[
            eligible["node1"].eq(gene) | eligible["node2"].eq(gene)
        ].copy()
        candidates["__neighbor"] = candidates.apply(
            lambda row: str(row["node2"])
            if str(row["node1"]) == gene
            else str(row["node1"]),
            axis=1,
        )
        candidates = candidates.sort_values(
            [score_column, "__neighbor"],
            ascending=[False, True],
        )
        for row in candidates.head(max(1, neighbors)).to_dict(orient="records"):
            selected.add(canonical_edge(gene, str(row["__neighbor"])))
    return selected


def _state_match_sources(
    evidence: pd.DataFrame,
    cfg: CABossV4Config,
) -> dict[tuple[str, str], set[str]]:
    sources: dict[tuple[str, str], set[str]] = {}

    def add(keys: set[tuple[str, str]], source: str) -> None:
        for key in keys:
            sources.setdefault(key, set()).add(source)

    metacell_column = (
        "metacell_state_eligible"
        if "metacell_state_eligible" in evidence
        else "state_eligible"
        if "state_eligible" in evidence
        else None
    )
    if metacell_column is not None:
        metacell_within = evidence.loc[evidence["same_assay"].astype(bool)].copy()
        add(
            _mutual_state_matches(
                metacell_within,
                cfg.mutual_state_neighbors,
                eligibility_column=metacell_column,
                score_column="state_score",
            ),
            "metacell_state_decomposition_within_assay",
        )
        add(
            _global_cross_state_matches(
                evidence,
                cfg.mutual_state_neighbors,
                eligibility_column=metacell_column,
                score_column="state_score",
            ),
            "metacell_state_decomposition_global_cross_assay",
        )
        add(
            _global_gene_state_matches(
                evidence,
                cfg.gene_state_neighbors,
                eligibility_column=metacell_column,
                score_column="state_score",
            ),
            "metacell_state_decomposition_global_peak_gene",
        )
    if "cell_loading_state_eligible" in evidence:
        cell_within = evidence.loc[evidence["same_assay"].astype(bool)].copy()
        add(
            _mutual_state_matches(
                cell_within,
                cfg.loading_within_neighbors,
                eligibility_column="cell_loading_state_eligible",
                score_column="cell_pair_state_score",
            ),
            "cell_loading_within_assay",
        )
        add(
            _global_cross_state_matches(
                evidence,
                cfg.loading_cross_neighbors,
                eligibility_column="cell_loading_state_eligible",
                score_column="cell_pair_state_score",
            ),
            "cell_loading_global_cross_assay",
        )
        add(
            _global_gene_state_matches(
                evidence,
                cfg.gene_state_neighbors,
                eligibility_column="cell_loading_state_eligible",
                score_column="cell_pair_state_score",
            ),
            "cell_loading_global_peak_gene",
        )
    return sources


def _edge_row(
    left: str,
    right: str,
    *,
    edge_type: str,
    evidence: dict[str, Any],
    cfg: CABossV4Config,
) -> dict[str, Any]:
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unknown edge type: {edge_type}")
    endpoint_left = "CIRCLE"
    endpoint_right = "CIRCLE"
    orientation_source = "abstained"
    if (
        edge_type == "prior_constrained_peak_gene"
        and cfg.impose_gene_sink_prior
        and (left.startswith("expr__") ^ right.startswith("expr__"))
    ):
        if left.startswith("expr__"):
            endpoint_left = "ARROW"
        else:
            endpoint_right = "ARROW"
        orientation_source = "gene_sink_prior"
    return {
        "node1": left,
        "endpoint1": endpoint_left,
        "node2": right,
        "endpoint2": endpoint_right,
        "edge_type": edge_type,
        "interpretation": edge_type,
        "orientation_source": orientation_source,
        **evidence,
    }


def _orient_compelled_colliders(edges: pd.DataFrame, evidence: pd.DataFrame, cfg: CABossV4Config) -> pd.DataFrame:
    if edges.empty:
        return edges
    output = edges.copy()
    direct_indices = output.index[
        output["edge_type"].eq("within_assay_direct_candidate")
    ].tolist()
    direct_pairs = {
        canonical_edge(str(output.at[index, "node1"]), str(output.at[index, "node2"]))
        for index in direct_indices
    }
    evidence_map = {
        canonical_edge(str(row["node1"]), str(row["node2"])): row
        for row in evidence.to_dict(orient="records")
    }
    neighbors: dict[str, list[tuple[str, int]]] = {}
    for index in direct_indices:
        left, right = str(output.at[index, "node1"]), str(output.at[index, "node2"])
        neighbors.setdefault(left, []).append((right, index))
        neighbors.setdefault(right, []).append((left, index))
    for center, adjacent in neighbors.items():
        for (left, left_index), (right, right_index) in combinations(adjacent, 2):
            outer_key = canonical_edge(left, right)
            if outer_key in direct_pairs:
                continue
            outer = evidence_map.get(outer_key)
            if outer is not None and float(outer.get("residual_empirical_q", 1.0)) < cfg.collider_separation_q:
                continue
            for index in (left_index, right_index):
                if str(output.at[index, "node1"]) == center:
                    output.at[index, "endpoint1"] = "ARROW"
                else:
                    output.at[index, "endpoint2"] = "ARROW"
                output.at[index, "orientation_source"] = "unshielded_residual_collider"
    return output


def build_typed_graph(
    evidence: pd.DataFrame,
    cfg: CABossV4Config,
    *,
    hierarchical_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select sparse typed adjacencies and orient only supported colliders/priors."""

    hierarchy = _hierarchical_selection(
        hierarchical_evidence,
        probability_threshold=cfg.hierarchical_probability_threshold,
        positive_effect_only=cfg.hierarchical_positive_effect_only,
    )
    state_match_sources = _state_match_sources(evidence, cfg)
    rows: list[dict[str, Any]] = []
    for record in evidence.to_dict(orient="records"):
        left, right = str(record["node1"]), str(record["node2"])
        key = canonical_edge(left, right)
        edge_type: str | None = None
        selected_hierarchy = key in hierarchy
        if bool(record["same_assay"]) and (bool(record["direct_eligible"]) or selected_hierarchy):
            edge_type = "within_assay_direct_candidate"
        elif key in state_match_sources:
            edge_type = (
                "prior_constrained_peak_gene"
                if left.startswith("expr__") ^ right.startswith("expr__")
                else (
                    "within_assay_shared_state"
                    if bool(record["same_assay"])
                    else "cross_assay_shared_state"
                )
            )
        elif (
            cfg.emit_unresolved
            and not bool(record["same_assay"])
            and abs(float(record["residual_r"])) >= cfg.min_abs_direct_residual_r
            and float(record.get("residual_to_pair_null95", 0.0))
            >= cfg.min_residual_to_pair_null95
            and float(record["residual_empirical_q"]) <= cfg.direct_residual_q
            and float(record["direct_bootstrap_support"]) >= cfg.min_direct_bootstrap_support
        ):
            edge_type = "unresolved"
        if edge_type is None:
            continue
        compact = {
            field: record.get(field, float("nan"))
            for field in (
                "total_r",
                "predicted_r",
                "depth_predicted_r",
                "joint_nuisance_predicted_r",
                "residual_r",
                "residual_empirical_p",
                "residual_empirical_q",
                "residual_to_pair_null95",
                "residual_multiplicity_family",
                "residual_family_size",
                "loading_cosine",
                "left_state_r2",
                "right_state_r2",
                "left_marginal_state_r2",
                "right_marginal_state_r2",
                "left_depth_r2",
                "right_depth_r2",
                "state_score",
                "state_explained_covariance_fraction",
                "state_bootstrap_support",
                "direct_bootstrap_support",
                "cell_loading_cosine",
                "cell_absolute_loading_cosine",
                "cell_sign_invariant_unit_loading_distance",
                "cell_minimum_loading_norm",
                "cell_left_oof_partial_state_fit",
                "cell_right_oof_partial_state_fit",
                "cell_left_loading_stability",
                "cell_right_loading_stability",
                "cell_loading_stability_support",
                "cell_pair_state_score",
                "cell_loading_empirical_null_p",
                "cell_loading_empirical_null_q",
                "cell_unit_distance_null_percentile",
            )
        }
        compact["hierarchical_sparse_selected"] = selected_hierarchy
        compact["state_evidence_source"] = ",".join(sorted(state_match_sources.get(key, set())))
        rows.append(
            _edge_row(
                left,
                right,
                edge_type=edge_type,
                evidence=compact,
                cfg=cfg,
            )
        )
    columns = [
        "node1",
        "endpoint1",
        "node2",
        "endpoint2",
        "edge_type",
        "interpretation",
        "orientation_source",
    ]
    graph = pd.DataFrame(rows)
    if graph.empty:
        return pd.DataFrame(columns=columns)
    graph = _orient_compelled_colliders(graph, evidence, cfg)
    if cfg.suppress_state_projection_through_direct_colliders:
        collider_children: set[str] = set()
        for record in graph.loc[
            graph["edge_type"].eq("within_assay_direct_candidate")
            & graph["orientation_source"].eq("unshielded_residual_collider")
        ].to_dict(orient="records"):
            if str(record["endpoint1"]) == "ARROW":
                collider_children.add(str(record["node1"]))
            if str(record["endpoint2"]) == "ARROW":
                collider_children.add(str(record["node2"]))
        if collider_children:
            state_projection = graph["edge_type"].isin(
                {
                    "within_assay_shared_state",
                    "cross_assay_shared_state",
                    "prior_constrained_peak_gene",
                }
            )
            incident_to_collider = graph["node1"].isin(collider_children) | graph[
                "node2"
            ].isin(collider_children)
            graph = graph.loc[~(state_projection & incident_to_collider)].copy()
    symbol = {
        ("CIRCLE", "CIRCLE"): "o-o",
        ("CIRCLE", "ARROW"): "o->",
        ("ARROW", "CIRCLE"): "<-o",
        ("TAIL", "ARROW"): "-->",
        ("ARROW", "TAIL"): "<--",
    }
    graph["edge_text"] = graph.apply(
        lambda row: (
            f"{row['node1']} "
            f"{symbol.get((row['endpoint1'], row['endpoint2']), 'o-o')} "
            f"{row['node2']}"
        ),
        axis=1,
    )
    return graph.sort_values(["edge_type", "node1", "node2"]).reset_index(drop=True)


def state_node_edges(decomposition: StateDecomposition) -> pd.DataFrame:
    """Return the explicit latent-state measurement layer, separate from adjacency."""

    if decomposition.state_loadings.empty:
        return pd.DataFrame(
            columns=["node1", "endpoint1", "node2", "endpoint2", "edge_type", "coefficient"]
        )
    rows = []
    for record in decomposition.state_loadings.to_dict(orient="records"):
        coordinate = str(record["state_coordinate"])
        state_node = coordinate.replace("state__", "latent_state__", 1)
        rows.append(
            {
                "node1": state_node,
                "endpoint1": "TAIL",
                "node2": str(record["feature"]),
                "endpoint2": "ARROW",
                "edge_type": "state_measurement_loading",
                "coefficient": float(record["coefficient"]),
                "orientation_source": "measurement_model",
            }
        )
    return pd.DataFrame(rows)


def fit_hierarchical_cell_evidence(
    cell_tables: list[Path],
    embedding_table: Path,
    cfg: CABossV4Config,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Fit the uncertainty-aware Bernoulli layer once per observed assay."""

    if not cell_tables:
        return pd.DataFrame(), {}, []
    _assert_discovery_input(embedding_table)
    from hierarchical_sparse_edges import (
        HierarchicalSparseConfig,
        estimate_from_observed_tables,
    )

    edge_frames: list[pd.DataFrame] = []
    detail_frames: dict[str, list[pd.DataFrame]] = {
        "microstates": [],
        "peak_diagnostics": [],
        "assignments": [],
    }
    metadata: list[dict[str, Any]] = []
    for table_i, cell_table in enumerate(cell_tables):
        _assert_discovery_input(cell_table)
        feature_columns = [column for column in _read_columns(cell_table) if is_biological(column)]
        assay = cell_table.stem.removeprefix("observed_cells__").removeprefix(
            "observed_focal__"
        )
        if len(feature_columns) < 2:
            metadata.append(
                {
                    "source_assay": assay,
                    "status": "skipped_fewer_than_two_candidate_features",
                    "n_candidate_features": int(len(feature_columns)),
                }
            )
            continue
        result = estimate_from_observed_tables(
            cell_table,
            embedding_table,
            feature_columns=feature_columns,
            retained_cells_only=False,
            config=HierarchicalSparseConfig(
                n_microstates=cfg.hierarchical_microstates,
                min_microstate_cells=cfg.hierarchical_min_microstate_cells,
                min_peak_events=cfg.hierarchical_min_peak_events,
                min_abs_residual_log_odds=cfg.hierarchical_min_abs_effect,
                evidence_threshold=cfg.hierarchical_probability_threshold,
                random_state=cfg.seed + 10_007 * (table_i + 1),
            ),
        )
        edge_frame = result.edge_evidence.copy()
        edge_frame["source_assay"] = assay
        edge_frames.append(edge_frame)
        for name, frame in (
            ("microstates", result.microstate_evidence),
            ("peak_diagnostics", result.peak_diagnostics),
            ("assignments", result.assignments),
        ):
            tagged = frame.copy()
            tagged["source_assay"] = assay
            detail_frames[name].append(tagged)
        metadata.append({"source_assay": assay, **result.metadata})
    combined_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    combined_details = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in detail_frames.items()
    }
    return combined_edges, combined_details, metadata


def fit_cell_loading_group_evidence(
    discovery_dir: Path,
    embedding_table: Path,
    cfg: CABossV4Config,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit common-coordinate loading groups from independently sampled cells."""

    _assert_discovery_input(embedding_table)
    from cell_state_loading_groups import (
        LoadingGroupConfig,
        estimate_loading_groups,
        load_loading_inputs,
    )

    inputs = load_loading_inputs(discovery_dir, embedding_table)
    result = estimate_loading_groups(
        inputs,
        LoadingGroupConfig(
            folds=cfg.loading_folds,
            ridge_alpha=cfg.ridge_alpha,
            depth_ridge_alpha=cfg.ridge_alpha,
            bootstrap_repetitions=cfg.loading_bootstrap_repetitions,
            seed=cfg.seed,
            min_loading_norm=cfg.loading_min_norm,
            min_state_fit=cfg.loading_min_state_fit,
            min_direction_stability=cfg.loading_min_direction_stability,
            min_abs_loading_cosine=cfg.loading_min_abs_cosine,
            max_unit_loading_distance=cfg.loading_max_unit_distance,
            max_neighbors_per_feature=max(
                cfg.loading_within_neighbors,
                cfg.loading_cross_neighbors,
            ),
        ),
    )
    details = {
        "feature_models": result.feature_models,
        "loadings": result.loadings,
        "sparse_pairs": result.sparse_pairs,
        "groups": result.groups,
        "group_membership": result.group_membership,
    }
    return result.pair_evidence, details, result.provenance


def discover_v4(
    metacell_matrix: Path,
    embedding_table: Path,
    membership_table: Path,
    cfg: CABossV4Config,
    *,
    cell_tables: list[Path] | None = None,
    hierarchical_evidence: pd.DataFrame | None = None,
    cell_loading_evidence: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    StateDecomposition,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    _assert_discovery_input(metacell_matrix)
    observed = pd.read_csv(metacell_matrix, sep="\t")
    if "sample_id" not in observed:
        raise ValueError("Observed metacell matrix requires sample_id")
    sample_ids = observed["sample_id"].astype(str).tolist()
    state_basis, provenance = load_state_centroids(
        embedding_table,
        membership_table,
        sample_ids,
    )
    state_basis = state_basis.set_index("sample_id").reindex(sample_ids).reset_index(drop=True)
    decomposition = decompose_features(observed, state_basis, cfg)
    evidence = build_pair_evidence(decomposition, cfg)
    if cell_loading_evidence is not None:
        evidence = attach_cell_loading_evidence(evidence, cell_loading_evidence)
        evidence = apply_evidence_thresholds(evidence, cfg)
    externally_supplied_hierarchy = hierarchical_evidence is not None
    hierarchical_details: dict[str, pd.DataFrame] = {}
    hierarchical_metadata: list[dict[str, Any]] = []
    if hierarchical_evidence is None and cell_tables:
        hierarchical_evidence, hierarchical_details, hierarchical_metadata = (
            fit_hierarchical_cell_evidence(cell_tables, embedding_table, cfg)
        )
    graph = build_typed_graph(
        evidence,
        cfg,
        hierarchical_evidence=hierarchical_evidence,
    )
    latent_edges = state_node_edges(decomposition)
    type_counts = graph.get("edge_type", pd.Series(dtype=str)).value_counts().to_dict()
    summary = {
        "method": "ca_boss_v4_explicit_state",
        "config": asdict(cfg),
        "inputs": {
            "metacell_matrix": str(metacell_matrix),
            "embedding_table": str(embedding_table),
            "membership_table": str(membership_table),
            "cell_tables": [str(path) for path in (cell_tables or [])],
            "hierarchical_evidence_supplied": externally_supplied_hierarchy,
            "cell_loading_evidence_supplied": cell_loading_evidence is not None,
        },
        "provenance": provenance,
        "preprocessing": decomposition.preprocessing,
        "n_pair_evidence_rows": int(len(evidence)),
        "n_typed_edges": int(len(graph)),
        "edge_type_counts": {str(key): int(value) for key, value in type_counts.items()},
        "n_latent_state_loading_edges": int(len(latent_edges)),
        "hierarchical_cell_layer": hierarchical_metadata,
        "contract": {
            "within_assay_direct_candidate": "residual evidence from co-observed assay cells/metacells",
            "within_assay_shared_state": "same-assay mutual state-response match without residual support",
            "cross_assay_shared_state": "mutual state-response match; orientation prohibited",
            "prior_constrained_peak_gene": "gene arrowhead is prior-constrained, not learned",
            "unresolved": "cross-assay residual evidence without identifiable direction",
        },
    }
    return (
        graph,
        decomposition,
        evidence,
        latent_edges,
        hierarchical_evidence if hierarchical_evidence is not None else pd.DataFrame(),
        hierarchical_details,
        summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metacell-matrix", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--membership-table", type=Path, required=True)
    parser.add_argument("--cell-table", type=Path, action="append", default=[])
    parser.add_argument("--hierarchical-evidence", type=Path)
    parser.add_argument(
        "--hierarchical-summary",
        type=Path,
        help="Required provenance summary from hierarchical_sparse_edges.py when loading evidence.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_values: dict[str, Any] = {}
    if args.config_json:
        config_values = json.loads(args.config_json.read_text())
    config_values["seed"] = args.seed
    cfg = CABossV4Config(**config_values)
    if bool(args.hierarchical_evidence) != bool(args.hierarchical_summary):
        raise SystemExit(
            "--hierarchical-evidence and --hierarchical-summary must be supplied together"
        )
    hierarchy = None
    if args.hierarchical_evidence:
        _assert_discovery_input(args.hierarchical_evidence)
        _assert_discovery_input(args.hierarchical_summary)
        provenance = json.loads(args.hierarchical_summary.read_text())
        contract = provenance.get("observed_table_contract", {})
        if (
            provenance.get("method") != "hierarchical_sparse_bernoulli_score_eb"
            or contract.get("truth_or_pairing_inputs") is not False
        ):
            raise SystemExit("Hierarchical evidence provenance is missing or unsafe")
        hierarchy = pd.read_csv(args.hierarchical_evidence, sep="\t")
    (
        graph,
        decomposition,
        evidence,
        latent_edges,
        hierarchical_edges,
        hierarchical_details,
        summary,
    ) = discover_v4(
        args.metacell_matrix,
        args.embedding_table,
        args.membership_table,
        cfg,
        cell_tables=args.cell_table,
        hierarchical_evidence=hierarchy,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph.to_csv(args.out_dir / "edges.tsv", sep="\t", index=False)
    evidence.to_csv(args.out_dir / "pair_evidence.tsv", sep="\t", index=False)
    decomposition.feature_models.to_csv(
        args.out_dir / "feature_state_models.tsv", sep="\t", index=False
    )
    decomposition.state_loadings.to_csv(
        args.out_dir / "state_loadings.tsv", sep="\t", index=False
    )
    latent_edges.to_csv(args.out_dir / "latent_state_edges.tsv", sep="\t", index=False)
    hierarchical_edges.to_csv(
        args.out_dir / "hierarchical_sparse_edge_evidence.tsv", sep="\t", index=False
    )
    for name, frame in hierarchical_details.items():
        frame.to_csv(args.out_dir / f"hierarchical_sparse_{name}.tsv", sep="\t", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
