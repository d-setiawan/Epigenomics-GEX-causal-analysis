"""Truth-free soft reporter evidence across background-ICA refits (v4.63).

Each reporter supplies three depth-adjusted squared correlations.  Average
these *before* normalizing across axes: weak reporters then have weak influence
instead of receiving the same vote as strong alternatives.  These weights are
association scores, not posterior component probabilities.  Every converged
assay enters background-only distribution synchronization, including assays
whose reporter evidence is diffuse or zero.

The signed pair statistic is the centered dot product of aligned mean squared
correlations.  It retains negative values and gives zero for a uniform axis
profile.  One third is an algebraic centering constant, not an empirical null.
This module emits evidence only; it does not select edges or access truth.
The representation ensemble follows the reliability rationale of ICASSO
(Himberg et al., 2004; doi:10.1016/j.neuroimage.2004.03.027).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import diagnose_representation_bootstrap_v4 as v462


SCHEMA = "ca_boss_v4.soft_reporter_representation_ensemble.v1"
CONTEXTS = v462.CONTEXTS
MODES = v462.MODES
IDENTITY = v462.IDENTITY
PRIMARY_SCORE = "strength_centered_concordance"
PAIR_METRICS = (
    PRIMARY_SCORE,
    "centered_concordance",
    "minimum_endpoint_total_strength",
    "minimum_endpoint_information",
    "minimum_endpoint_weight_margin",
    "minimum_reporter_best_squared_correlation",
    "global_relative_margin",
)


@dataclass(frozen=True)
class ReporterAxisEvidence:
    partial_correlations: np.ndarray
    per_reporter_weights: np.ndarray
    mean_squared_correlations: np.ndarray
    weights: np.ndarray
    total_strength: float
    information: float
    weight_margin: float
    minimum_reporter_best_squared_correlation: float
    nonzero_reporters: int
    usable: bool


def reporter_axis_evidence(
    focal: np.ndarray, scores: np.ndarray, observed_depth: np.ndarray
) -> ReporterAxisEvidence:
    """Summarize all supplied reporter alternatives without choosing an axis.

    A zero-association reporter contributes a zero vector to the arithmetic
    mean; an assay abstains only when its aggregate strength is zero.  The
    weakest individual reporter's strength remains a separate audit field.
    """
    y_values = np.asarray(focal, dtype=float)
    x_values = np.asarray(scores, dtype=float)
    depth = np.asarray(observed_depth, dtype=float).reshape(-1)
    if y_values.ndim != 2 or y_values.shape[1] < 1:
        raise ValueError("At least one two-dimensional reporter column is required")
    if x_values.shape != (len(y_values), 3) or depth.shape != (len(y_values),):
        raise ValueError("Reporter, depth and rank-three score rows must agree")
    if len(y_values) < 8 or not all(np.isfinite(v).all() for v in (y_values, x_values, depth)):
        raise ValueError("At least eight finite same-assay observations are required")
    if np.any(y_values < 0) or np.any(depth <= 0):
        raise ValueError("Reporter counts must be nonnegative and depth positive")
    design = np.column_stack([np.ones(len(depth)), np.log1p(depth)])
    y = np.log1p(y_values)
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x = x_values - design @ np.linalg.lstsq(design, x_values, rcond=None)[0]
    denominator = np.linalg.norm(y, axis=0)[:, None] * np.linalg.norm(x, axis=0)
    correlation = np.divide(
        y.T @ x, denominator,
        out=np.zeros((y.shape[1], 3)), where=denominator > 1e-12,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    squared = np.square(correlation)
    per_strength = squared.sum(axis=1)
    per_weights = np.divide(squared, per_strength[:, None], out=np.zeros_like(squared), where=per_strength[:, None] > 1e-12)
    mean_squared = squared.mean(axis=0)
    total = float(mean_squared.sum())
    usable = bool(total > 1e-12)
    weights = mean_squared / total if usable else np.zeros(3)
    nonzero = weights > 0
    information = float(1.0 + np.sum(weights[nonzero] * np.log(weights[nonzero])) / np.log(3.0)) if usable else 0.0
    ordered = np.sort(weights)
    return ReporterAxisEvidence(
        partial_correlations=correlation,
        per_reporter_weights=per_weights,
        mean_squared_correlations=mean_squared,
        weights=weights,
        total_strength=total,
        information=float(np.clip(information, 0.0, 1.0)),
        weight_margin=float(ordered[-1] - ordered[-2]),
        minimum_reporter_best_squared_correlation=float(squared.max(axis=1).min()),
        nonzero_reporters=int(np.sum(per_strength > 1e-12)),
        usable=usable,
    )


def soft_pair_evidence(
    left: ReporterAxisEvidence,
    right: ReporterAxisEvidence,
    left_order: Sequence[int],
    right_order: Sequence[int],
) -> dict[str, Any]:
    """Align local axes to a shared permutation frame and compare evidence."""
    for order in (left_order, right_order):
        if sorted(order) != [0, 1, 2]:
            raise ValueError("Orders must map each global axis to one local axis")
    usable = left.usable and right.usable
    a = left.mean_squared_correlations[np.asarray(left_order, dtype=int)]
    b = right.mean_squared_correlations[np.asarray(right_order, dtype=int)]
    p = left.weights[np.asarray(left_order, dtype=int)]
    q = right.weights[np.asarray(right_order, dtype=int)]
    centered = float(np.dot(p - 1.0 / 3.0, q - 1.0 / 3.0)) if usable else 0.0
    # Center vectors before the dot product to avoid subtracting two almost
    # equal positive products when profiles are approximately uniform.
    strength_centered = float(np.dot(a - left.total_strength / 3.0, b - right.total_strength / 3.0)) if usable else 0.0
    return {
        "eligible": usable,
        PRIMARY_SCORE: strength_centered,
        "centered_concordance": centered,
        "minimum_endpoint_total_strength": min(left.total_strength, right.total_strength),
        "minimum_endpoint_information": min(left.information, right.information),
        "minimum_endpoint_weight_margin": min(left.weight_margin, right.weight_margin),
        "minimum_reporter_best_squared_correlation": min(left.minimum_reporter_best_squared_correlation, right.minimum_reporter_best_squared_correlation),
        "left_global_axis_weights_json": json.dumps(p.tolist()),
        "right_global_axis_weights_json": json.dumps(q.tolist()),
    }


def compute_array_soft_evidence(
    key: str, arrays: Mapping[str, Mapping[str, Any]], *,
    replicates: int, random_seed: int, modes: Sequence[str] = MODES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run crossed refits on one world's verified assay arrays from v4.62.

    This is sensitivity to representation fitting evaluated on a fixed
    original-cell subset per replicate. It is not a whole-estimator bootstrap:
    evaluation rows do not retain in-bag multiplicities, and cached references
    have seen these rows.  The fixed subset is paired across modes and contexts
    only within each assay.  Reporter columns affect no embedding or alignment.
    """
    if replicates < 1 or not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Positive replicates and unique supported modes are required")
    modalities = sorted(arrays)
    if len(modalities) < 2:
        raise ValueError("At least two independent assay arrays are required")
    block = key.split("__world_seed", 1)[0]
    pair_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for mode in modes:
            for replicate in range(replicates):
                seed = v462.mdcr._stable_seed(random_seed, f"{key}:{replicate}")
                context_scores: dict[str, dict[str, np.ndarray]] = {name: {} for name in CONTEXTS}
                context_reporters: dict[str, dict[str, ReporterAxisEvidence]] = {name: {} for name in CONTEXTS}
                for modality in modalities:
                    a = arrays[modality]
                    evaluation = v462._evaluation_indices(len(a["counts"]), modality, seed)
                    try:
                        if len(evaluation) < 8:
                            raise v462.representation.RepresentationFitError("Fewer than eight common evaluation cells")
                        fitted = None if mode == "cached_reference_subset" else v462.representation.bootstrap_fit_assay_contexts(
                            a["counts"], a["depth"], a["reference"], modality=modality,
                            reference_seed=a["reference_seed"], resample_seed=seed,
                            initialization_seed=seed, mode=mode, config=a["config"],
                        )
                    except v462.representation.RepresentationFitError as error:
                        for context in CONTEXTS:
                            fit_rows.append({"world_key": key, "mode": mode, "replicate": replicate, "modality": modality, "context": context, "fit_ok": False, "failure": str(error)})
                        continue
                    if fitted is not None and mode in ("cell_resample_only", "both"):
                        if not np.array_equal(evaluation, np.flatnonzero(fitted.oob_mask)):
                            raise ValueError("Representation and evaluation bootstrap rows disagree")
                    for context in CONTEXTS:
                        scores = (a["reference"][context] if fitted is None else fitted.scores[context])[evaluation]
                        # Background geometry must not depend on reporter usability.
                        context_scores[context][modality] = scores
                        reporter = reporter_axis_evidence(a["focal"][evaluation], scores, a["depth"][evaluation])
                        context_reporters[context][modality] = reporter
                        fit_rows.append({
                            "world_key": key, "mode": mode, "replicate": replicate,
                            "modality": modality, "context": context, "fit_ok": True,
                            "failure": "", "evaluation_cells": len(evaluation),
                            "evaluation_cells_out_of_bag_for_ica": mode in ("cell_resample_only", "both"),
                            "representation_refitted": fitted is not None,
                            "reporter_usable": reporter.usable,
                            "reporter_count": int(a["focal"].shape[1]),
                            "nonzero_reporters": reporter.nonzero_reporters,
                            "total_squared_correlation_strength": reporter.total_strength,
                            "reporter_axis_information": reporter.information,
                            "reporter_weight_margin": reporter.weight_margin,
                            "minimum_reporter_best_squared_correlation": reporter.minimum_reporter_best_squared_correlation,
                            "partial_correlations_json": json.dumps(reporter.partial_correlations.tolist()),
                            "per_reporter_axis_weights_json": json.dumps(reporter.per_reporter_weights.tolist()),
                            "mean_squared_correlations_json": json.dumps(reporter.mean_squared_correlations.tolist()),
                            "axis_weights_json": json.dumps(reporter.weights.tolist()),
                            "alignment_diagnostics_json": json.dumps({"reference_scores_reused": True} if fitted is None else fitted.contexts[context].__dict__, sort_keys=True),
                        })
                for context in CONTEXTS:
                    current = context_scores[context]
                    complete = len(current) == len(modalities)
                    # A failed assay must not change the global optimization
                    # problem.  Incomplete contexts contribute zero evidence.
                    fingerprints = {m: v462.joint._projection_quantiles(x, copula_only=False) for m, x in current.items()} if complete else {}
                    matrices = {(a, b): v462.joint._precomputed_cost_matrix(fingerprints[a], fingerprints[b]) for a, b in combinations(sorted(fingerprints), 2)}
                    table = v462.geometry._normalized_cost_table(current, matrices) if matrices else None
                    solution = v462.geometry.synchronization.solve_multiway_permutation_synchronization(table, sorted(current)) if table is not None else None
                    orders = {} if solution is None else dict(zip(solution.modalities, solution.best_orders))
                    for left, right in combinations(modalities, 2):
                        row: dict[str, Any] = {
                            "world_key": key, "block": block, "left_modality": left,
                            "right_modality": right, "mode": mode, "replicate": replicate,
                            "context": context, "fit_pair_available": False,
                            "eligible": False, "synchronized_assay_count": len(current),
                            "expected_assay_count": len(modalities),
                            "complete_context": complete,
                            **{metric: 0.0 for metric in PAIR_METRICS},
                        }
                        if left in orders and right in orders:
                            row.update(soft_pair_evidence(context_reporters[context][left], context_reporters[context][right], orders[left], orders[right]))
                            row.update(fit_pair_available=True, global_relative_margin=float(solution.global_relative_margin), left_global_order_json=json.dumps(orders[left].tolist()), right_global_order_json=json.dumps(orders[right].tolist()))
                        pair_rows.append(row)
    return pd.DataFrame(pair_rows), pd.DataFrame(fit_rows)


def compute_world_soft_evidence(
    key: str, artifacts: Any, formal_root: Path, *,
    replicates: int, random_seed: int, modes: Sequence[str] = MODES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load verified discovery/reference inputs and return pretruth evidence."""
    arrays, hashes = v462._load_world(key, artifacts, formal_root)
    pairs, fits = compute_array_soft_evidence(key, arrays, replicates=replicates, random_seed=random_seed, modes=modes)
    return pairs, fits, hashes


def aggregate_soft_evidence(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    """Aggregate with a fixed denominator; failed fits contribute zero evidence.

    ``*_minimum_context_mean`` means: average each context across all requested
    refits, then take the minimum of those three averages. Negative evidence is
    preserved. The reverse aggregation order is reported separately.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rows = []
    for identity, group in frame.groupby([*IDENTITY, "mode"], sort=True):
        expected = {(r, c) for r in range(replicates) for c in CONTEXTS}
        actual = list(zip(group["replicate"].astype(int), group["context"].astype(str)))
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Missing or duplicate replicate/context evidence")
        row = {**dict(zip([*IDENTITY, "mode"], identity)), "replicates": replicates}
        if group["expected_assay_count"].nunique() != 1:
            raise ValueError("Expected assay count changed across refits")
        row["expected_assay_count"] = int(group["expected_assay_count"].iloc[0])
        row["complete_context_fraction"] = float(group["complete_context"].astype(bool).mean())
        for field in ("fit_pair_available", "eligible"):
            row[f"{field}_frequency"] = float(group[field].astype(bool).mean())
            row[f"{field}_all_context_frequency"] = float(group.groupby("replicate")[field].all().mean())
        for metric in PAIR_METRICS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite evidence in {metric}")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_minimum_context_mean"] = float(group.groupby("context")[metric].mean().min())
            row[f"{metric}_mean_replicate_minimum_context"] = float(group.groupby("replicate")[metric].min().mean())
        row["minimum_context_mean_weighted_concordance"] = row[f"{PRIMARY_SCORE}_minimum_context_mean"]
        row["mean_weighted_concordance"] = row[f"{PRIMARY_SCORE}_mean"]
        rows.append(row)
    return pd.DataFrame(rows)
