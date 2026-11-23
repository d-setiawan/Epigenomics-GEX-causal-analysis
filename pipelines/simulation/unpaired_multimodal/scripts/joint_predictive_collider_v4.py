#!/usr/bin/env python3
"""Cross-fitted joint predictive evidence for sparse within-assay colliders.

This layer is deliberately downstream of the permissive beta-binomial *screen*.
Screened pair evidence may nominate several adjacent arms, but a direct motif is
selected only when one center is uniquely better predicted by two positive
parents after background-state and leave-focal-out depth adjustment.  The
screen alone never creates an edge.

Only variables observed in the same assay are compared.  No role, truth,
cross-assay pairing, feature correspondence, or external data enter the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


FloatArray = NDArray[np.float64]

_FORBIDDEN_TOKENS = (
    "truth",
    "true_",
    "role",
    "oracle",
    "scorer",
    "counterfactual",
    "pairing",
    "correspondence",
    "latent_state",
)


@dataclass(frozen=True)
class JointColliderConfig:
    """Frozen screening and cross-fitted uniqueness thresholds."""

    max_screen_mean_empirical_p: float = 0.05
    min_screen_vote_fraction: float = 5.0 / 6.0
    min_screen_mean_log_odds: float = 1.0
    n_folds: int = 5
    logistic_c: float = 2.0
    min_mean_log_likelihood_gain: float = 0.001
    min_relative_log_likelihood_gain: float = 0.0075
    min_parent_coefficient: float = 0.30
    min_positive_coefficient_support: float = 0.80
    min_relative_gain_margin: float = 0.005
    random_state: int = 73

    def __post_init__(self) -> None:
        probabilities = (
            self.max_screen_mean_empirical_p,
            self.min_screen_vote_fraction,
            self.min_positive_coefficient_support,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Probability thresholds must lie in [0, 1]")
        if self.n_folds < 2 or self.logistic_c <= 0:
            raise ValueError("n_folds and logistic_c must be positive")
        if min(
            self.min_screen_mean_log_odds,
            self.min_mean_log_likelihood_gain,
            self.min_relative_log_likelihood_gain,
            self.min_parent_coefficient,
            self.min_relative_gain_margin,
        ) < 0:
            raise ValueError("Effect, gain, and margin thresholds may not be negative")


@dataclass(frozen=True)
class JointColliderResult:
    wedge_evidence: pd.DataFrame
    selected_wedges: pd.DataFrame
    relaxed_screen_edges: pd.DataFrame
    metadata: dict[str, Any]


def _assert_safe_columns(columns: list[str] | pd.Index) -> None:
    forbidden = sorted(
        column
        for column in map(str, columns)
        if any(token in column.lower() for token in _FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Joint collider evidence rejects privileged columns: {forbidden}")


def _canonical(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))  # type: ignore[return-value]


def _components(edges: set[tuple[str, str]]) -> list[set[str]]:
    neighbors: dict[str, set[str]] = {}
    for left, right in edges:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    output: list[set[str]] = []
    unseen = set(neighbors)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(neighbors.get(node, ()), reverse=True))
        unseen -= component
        output.append(component)
    return sorted(output, key=lambda value: tuple(sorted(value)))


def _scale_train_test(
    train: FloatArray, test: FloatArray
) -> tuple[FloatArray, FloatArray]:
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def _joint_gain(
    detections: pd.DataFrame,
    nuisance: FloatArray,
    center: str,
    outer1: str,
    outer2: str,
    config: JointColliderConfig,
) -> dict[str, float] | None:
    y = (detections[center].to_numpy(dtype=np.float64) > 0).astype(np.int64)
    minimum_class = int(min(y.sum(), len(y) - y.sum()))
    folds = min(config.n_folds, minimum_class)
    if folds < 2:
        return None
    outer = np.column_stack(
        [
            (detections[outer1].to_numpy(dtype=np.float64) > 0).astype(np.float64),
            (detections[outer2].to_numpy(dtype=np.float64) > 0).astype(np.float64),
        ]
    )
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.random_state,
    )
    gain = np.zeros(len(y), dtype=np.float64)
    baseline_loss = np.zeros(len(y), dtype=np.float64)
    coefficients: list[FloatArray] = []
    for train, test in splitter.split(nuisance, y):
        nuisance_train, nuisance_test = _scale_train_test(
            nuisance[train], nuisance[test]
        )
        baseline = LogisticRegression(
            C=config.logistic_c,
            max_iter=2000,
            solver="lbfgs",
        ).fit(nuisance_train, y[train])
        augmented = LogisticRegression(
            C=config.logistic_c,
            max_iter=2000,
            solver="lbfgs",
        ).fit(
            np.column_stack([nuisance_train, outer[train]]),
            y[train],
        )
        probability0 = np.clip(
            baseline.predict_proba(nuisance_test)[:, 1], 1e-8, 1.0 - 1e-8
        )
        probability1 = np.clip(
            augmented.predict_proba(
                np.column_stack([nuisance_test, outer[test]])
            )[:, 1],
            1e-8,
            1.0 - 1e-8,
        )
        log0 = xlogy(y[test], probability0) + xlogy(1 - y[test], 1 - probability0)
        log1 = xlogy(y[test], probability1) + xlogy(1 - y[test], 1 - probability1)
        gain[test] = log1 - log0
        baseline_loss[test] = -log0
        coefficients.append(np.asarray(augmented.coef_[0][-2:], dtype=np.float64))

    coefficient_array = np.vstack(coefficients)
    mean_gain = float(gain.mean())
    mean_baseline_loss = float(baseline_loss.mean())
    return {
        "n_cells": float(len(y)),
        "n_folds": float(folds),
        "center_detection_rate": float(y.mean()),
        "mean_log_likelihood_gain": mean_gain,
        "log_likelihood_gain_se": float(gain.std(ddof=1) / np.sqrt(len(gain))),
        "relative_log_likelihood_gain": float(
            mean_gain / max(mean_baseline_loss, 1e-12)
        ),
        "mean_parent1_coefficient": float(coefficient_array[:, 0].mean()),
        "mean_parent2_coefficient": float(coefficient_array[:, 1].mean()),
        "minimum_parent_coefficient": float(coefficient_array.mean(axis=0).min()),
        "positive_coefficient_support": float((coefficient_array > 0).mean()),
    }


def estimate_joint_predictive_colliders(
    detections: pd.DataFrame,
    integration_state: FloatArray,
    background_depth: FloatArray,
    aggregated_pair_evidence: pd.DataFrame,
    *,
    source_assay: str,
    config: JointColliderConfig | None = None,
) -> JointColliderResult:
    """Select uniquely predictive same-assay two-parent candidates."""

    cfg = config or JointColliderConfig()
    frame = detections.copy()
    _assert_safe_columns(frame.columns)
    evidence = aggregated_pair_evidence.copy()
    _assert_safe_columns(evidence.columns)
    required = {
        "node1",
        "node2",
        "selected_vote_fraction",
        "mean_empirical_p_positive",
        "mean_pooled_log_odds",
    }
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"Aggregated pair evidence lacks {sorted(missing)}")
    if "source_assay" in evidence:
        evidence = evidence.loc[
            evidence["source_assay"].astype(str).eq(str(source_assay))
        ].copy()
    state = np.asarray(integration_state, dtype=np.float64)
    depth = np.asarray(background_depth, dtype=np.float64)
    if state.ndim != 2 or len(state) != len(frame) or depth.shape != (len(frame),):
        raise ValueError("Detections, integration state, and depth must share rows")
    if not np.isfinite(state).all() or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("State and background depth must be finite and depth non-negative")
    features = set(map(str, frame.columns))
    referenced = set(evidence["node1"].astype(str)) | set(evidence["node2"].astype(str))
    if not referenced.issubset(features):
        raise ValueError("Pair evidence references features absent from detections")

    relaxed = evidence.loc[
        (pd.to_numeric(evidence["selected_vote_fraction"]) >= cfg.min_screen_vote_fraction)
        & (pd.to_numeric(evidence["mean_empirical_p_positive"]) <= cfg.max_screen_mean_empirical_p)
        & (pd.to_numeric(evidence["mean_pooled_log_odds"]) >= cfg.min_screen_mean_log_odds)
    ].copy()
    relaxed["pair_key"] = [
        "||".join(_canonical(left, right))
        for left, right in zip(relaxed["node1"], relaxed["node2"], strict=True)
    ]
    relaxed = relaxed.drop_duplicates("pair_key").sort_values("pair_key").reset_index(drop=True)
    edges = {
        _canonical(row.node1, row.node2) for row in relaxed.itertuples(index=False)
    }
    nuisance = np.column_stack([state, np.log1p(depth)])
    rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(_components(edges)):
        component_id = f"{source_assay}__joint_component{component_index}"
        neighbors: dict[str, set[str]] = {node: set() for node in component}
        for left, right in edges:
            if left in component and right in component:
                neighbors[left].add(right)
                neighbors[right].add(left)
        for center in sorted(component):
            for outer1, outer2 in combinations(sorted(neighbors[center]), 2):
                result = _joint_gain(frame, nuisance, center, outer1, outer2, cfg)
                if result is None:
                    continue
                rows.append(
                    {
                        "source_assay": str(source_assay),
                        "component_id": component_id,
                        "center": center,
                        "outer1": outer1,
                        "outer2": outer2,
                        **result,
                    }
                )
    columns = [
        "source_assay",
        "component_id",
        "center",
        "outer1",
        "outer2",
        "n_cells",
        "n_folds",
        "center_detection_rate",
        "mean_log_likelihood_gain",
        "log_likelihood_gain_se",
        "relative_log_likelihood_gain",
        "mean_parent1_coefficient",
        "mean_parent2_coefficient",
        "minimum_parent_coefficient",
        "positive_coefficient_support",
        "runner_up_relative_gain",
        "relative_gain_margin",
        "selected",
        "selection_reason",
    ]
    wedge = pd.DataFrame(rows)
    if wedge.empty:
        wedge = pd.DataFrame(columns=columns)
        selected = wedge.copy()
    else:
        output: list[pd.DataFrame] = []
        for _, group in wedge.groupby("component_id", sort=True):
            ranked = group.sort_values(
                [
                    "relative_log_likelihood_gain",
                    "mean_log_likelihood_gain",
                    "center",
                    "outer1",
                    "outer2",
                ],
                ascending=[False, False, True, True, True],
            ).reset_index(drop=True)
            runner = (
                float(ranked.loc[1, "relative_log_likelihood_gain"])
                if len(ranked) > 1
                else float("-inf")
            )
            ranked["runner_up_relative_gain"] = runner
            ranked["relative_gain_margin"] = (
                ranked["relative_log_likelihood_gain"] - runner
            )
            ranked["selected"] = False
            ranked["selection_reason"] = "not_component_winner"
            best = ranked.iloc[0]
            passes = bool(
                best["mean_log_likelihood_gain"] >= cfg.min_mean_log_likelihood_gain
                and best["relative_log_likelihood_gain"]
                >= cfg.min_relative_log_likelihood_gain
                and best["minimum_parent_coefficient"] >= cfg.min_parent_coefficient
                and best["positive_coefficient_support"]
                >= cfg.min_positive_coefficient_support
                and best["relative_gain_margin"] >= cfg.min_relative_gain_margin
            )
            ranked.loc[0, "selected"] = passes
            ranked.loc[0, "selection_reason"] = (
                "unique_joint_predictive_wedge"
                if passes
                else "joint_predictive_threshold_or_margin_failed"
            )
            output.append(ranked)
        wedge = pd.concat(output, ignore_index=True)[columns]
        selected = wedge.loc[wedge["selected"].astype(bool)].copy()
    return JointColliderResult(
        wedge_evidence=wedge,
        selected_wedges=selected,
        relaxed_screen_edges=relaxed,
        metadata={
            "schema_version": "joint_predictive_collider_v4.v1",
            "method": "cross_fitted_joint_child_prediction_after_relaxed_beta_screen",
            "config": asdict(cfg),
            "source_assay": str(source_assay),
            "n_cells": len(frame),
            "n_features": len(frame.columns),
            "n_relaxed_screen_edges": len(relaxed),
            "n_wedges_scored": len(wedge),
            "n_wedges_selected": len(selected),
            "screen_edges_create_graph_adjacencies": False,
            "same_assay_variables_only": True,
            "cross_assay_direction_identified": False,
            "truth_roles_or_external_data_used": False,
        },
    )


def estimate_from_observed_tables(
    cell_table: Path | str,
    embedding_table: Path | str,
    aggregated_pair_evidence: pd.DataFrame,
    *,
    source_assay: str,
    config: JointColliderConfig | None = None,
) -> JointColliderResult:
    """Leakage-safe table adapter for one observed assay."""

    cell_path = Path(cell_table)
    embedding_path = Path(embedding_table)
    for path in (cell_path, embedding_path):
        if any(token in part.lower() for part in path.parts for token in _FORBIDDEN_TOKENS):
            raise ValueError(f"Joint collider evidence rejects privileged paths: {path}")
    cells = pd.read_csv(cell_path, sep="\t", keep_default_na=False)
    embedding = pd.read_csv(embedding_path, sep="\t", keep_default_na=False)
    _assert_safe_columns(cells.columns)
    _assert_safe_columns(embedding.columns)
    required_cell = {"cell", "observed_background_library_size"}
    if not required_cell.issubset(cells.columns):
        raise ValueError("Observed cell table lacks cell/background-depth columns")
    if not {"cell", "modality"}.issubset(embedding.columns):
        raise ValueError("Embedding table lacks cell/modality columns")
    selected_embedding = embedding.loc[
        embedding["modality"].astype(str).eq(str(source_assay))
    ].copy()
    state_columns = sorted(
        column for column in selected_embedding.columns if column.startswith("integration__dim")
    )
    if not state_columns:
        raise ValueError("Embedding table has no integration state coordinates")
    merged = cells.merge(
        selected_embedding[["cell", *state_columns]],
        on="cell",
        how="left",
        validate="one_to_one",
    )
    if merged[state_columns].isna().any().any() or len(merged) != len(cells):
        raise ValueError("Observed cells and embedding rows do not match exactly")
    assay_evidence = aggregated_pair_evidence.copy()
    if "source_assay" in assay_evidence.columns:
        assay_evidence = assay_evidence.loc[
            assay_evidence["source_assay"].astype(str).eq(str(source_assay))
        ].copy()
    if not {"node1", "node2"}.issubset(assay_evidence.columns):
        raise ValueError("Aggregated pair evidence lacks node columns")
    feature_columns = sorted(
        set(assay_evidence["node1"].astype(str))
        | set(assay_evidence["node2"].astype(str))
    )
    missing_features = sorted(set(feature_columns) - set(cells.columns))
    if missing_features:
        raise ValueError(
            f"Aggregated pair evidence references absent observed features: {missing_features}"
        )
    return estimate_joint_predictive_colliders(
        merged[feature_columns],
        merged[state_columns].to_numpy(dtype=np.float64),
        merged["observed_background_library_size"].to_numpy(dtype=np.float64),
        assay_evidence,
        source_assay=source_assay,
        config=config,
    )


__all__ = [
    "JointColliderConfig",
    "JointColliderResult",
    "estimate_from_observed_tables",
    "estimate_joint_predictive_colliders",
]
