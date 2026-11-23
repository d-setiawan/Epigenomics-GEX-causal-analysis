#!/usr/bin/env python3
"""Truth-barrier audit of a sparse, cross-fitted within-assay collider decoder.

This diagnostic uses only the discovery-side tables from retired TRAIN330,
TRAIN360, TRAIN370, and TRAIN380 worlds.  For each assay it enumerates every
possible ``outer1 -> center <- outer2`` star over the six observed focal
features.  It then adds three families of observed-only evidence:

* a quadratic held-out-background state/library-depth nuisance model;
* cross-fitted Bernoulli likelihood gains under a Normal(0, 2.5^2) Laplace
  prior, including both drop-one-parent gains and coefficient uncertainty;
* state/depth-preserving empirical-null residual correlations at two
  microstate resolutions.

All Phase-A tables and their hashes are written before scorer truth is opened.
Phase B labels the frozen candidates and evaluates fixed unsupervised and
leave-one-retired-block-out calibrated decoders.  The script is a diagnostic,
not a production selector, and refuses DEV/LOCKED paths or overwrites.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import expit, xlogy
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import audit_retired_direct_guard_rules_v4 as retired
import audit_retired_exhaustive_state_star_decoder_v1 as star_audit
import diagnose_formal_train_direct_failures_v4 as diagnostic
from observed_collider_structure import apply_observed_collider_refinements


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
TARGET_SEEDS = (38011, 38012, 38014, 38020)
SCHEMA = "ca_boss_v4.retired_crossfit_hierarchical_collider.v1"
N_FOLDS = 5
PARENT_PRIOR_SD = 2.5
NUISANCE_PRIOR_SD = 5.0
MICROSTATE_RESOLUTIONS = (8, 16)
N_NULL_PERMUTATIONS = 31
FORMULA_FAMILY = {
    "strong_generative": {
        "minimum_parent_positive_posterior": 0.98,
        "minimum_joint_gain_over_baseline": 0.005,
        "minimum_drop_one_gain": 0.0,
        "minimum_both_positive_fold_support": 0.80,
        "minimum_reverse_center_gain_margin": 0.0,
        "maximum_arm_empirical_positive_p": 0.125,
        "minimum_outer_empirical_null_compatibility_p": 0.03125,
    },
    "guard_failure_replacement": (
        "strong_generative AND both parents have positive partial-state fit "
        "AND conditional-Wald view support=1"
    ),
    "incomplete_motif_replacement": (
        "strong_generative AND (two-parent state/Wald consistency OR positive "
        "minimum collider separation in all six residual views)"
    ),
    "calibrated_models": (
        "leave-one-retired-block-out logistic C in {0.03,0.30} and random "
        "forest leaf=8; no held-out block chooses its own hyperparameters"
    ),
}

FloatArray = NDArray[np.float64]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*values: object) -> int:
    payload = "||".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _canonical(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _finite_correlation(left: FloatArray, right: FloatArray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _log_likelihood(y: FloatArray, probability: FloatArray) -> FloatArray:
    p = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return xlogy(y, p) + xlogy(1.0 - y, 1.0 - p)


@dataclass(frozen=True)
class _MapFit:
    coefficients: FloatArray
    covariance: FloatArray


def _fit_map_logistic(
    design: FloatArray,
    y: FloatArray,
    prior_precision: FloatArray,
    *,
    initial: FloatArray | None = None,
) -> _MapFit:
    """Small deterministic Newton solver for a Normal-prior Bernoulli GLM."""

    x = np.column_stack([np.ones(len(design)), design]).astype(np.float64)
    beta = (
        np.zeros(x.shape[1], dtype=np.float64)
        if initial is None or len(initial) != x.shape[1]
        else np.asarray(initial, dtype=np.float64).copy()
    )
    precision = np.r_[1e-8, np.asarray(prior_precision, dtype=np.float64)]

    def objective(value: FloatArray) -> float:
        probability = expit(x @ value)
        return float(
            _log_likelihood(y, probability).sum()
            - 0.5 * np.dot(precision * value, value)
        )

    current = objective(beta)
    for _ in range(40):
        probability = expit(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = x.T @ (y - probability) - precision * beta
        information = x.T @ (weight[:, None] * x) + np.diag(precision)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        if float(np.max(np.abs(step))) < 1e-7:
            break
        scale = 1.0
        accepted = False
        while scale >= 1.0 / 1024.0:
            proposal = beta + scale * step
            proposed = objective(proposal)
            if proposed >= current - 1e-10:
                beta = proposal
                current = proposed
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

    probability = expit(x @ beta)
    weight = np.maximum(probability * (1.0 - probability), 1e-7)
    information = x.T @ (weight[:, None] * x) + np.diag(precision)
    covariance = np.linalg.pinv(information)
    return _MapFit(coefficients=beta, covariance=covariance)


def _nuisance_design(frame: pd.DataFrame) -> FloatArray:
    state_columns = sorted(
        column for column in frame.columns if str(column).startswith("integration__dim")
    )
    if not state_columns:
        raise ValueError("Observed embedding has no integration coordinates")
    state = frame[state_columns].to_numpy(dtype=np.float64)
    log_depth = np.log1p(
        frame["observed_background_library_size"].to_numpy(dtype=np.float64)
    )[:, None]
    pieces = [state, log_depth, state * state, log_depth * log_depth, state * log_depth]
    if "integration_uncertainty" in frame:
        pieces.append(frame[["integration_uncertainty"]].to_numpy(dtype=np.float64))
    design = np.column_stack(pieces)
    if not np.isfinite(design).all():
        raise ValueError("Nuisance design contains non-finite values")
    return design


def _scale_train_test(
    train: FloatArray, test: FloatArray
) -> tuple[FloatArray, FloatArray]:
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def _crossfit_response(
    detections: FloatArray,
    nuisance: FloatArray,
    center_index: int,
    feature_names: tuple[str, ...],
    seed: int,
    assay: str,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Cross-fit baseline, single-parent, and all two-parent Bernoulli models."""

    y = detections[:, center_index]
    minimum_class = int(min(y.sum(), len(y) - y.sum()))
    n_folds = min(N_FOLDS, minimum_class)
    if n_folds < 2:
        raise ValueError(f"Too few detections for cross-fitting {seed} {assay}")
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=_stable_seed("fold", seed, assay, feature_names[center_index]),
    )
    parent_indices = [index for index in range(len(feature_names)) if index != center_index]
    parent_pairs = list(itertools.combinations(parent_indices, 2))
    baseline_ll = np.zeros(len(y), dtype=np.float64)
    baseline_probability = np.zeros(len(y), dtype=np.float64)
    single_ll = {
        parent: np.zeros(len(y), dtype=np.float64) for parent in parent_indices
    }
    joint_ll = {pair: np.zeros(len(y), dtype=np.float64) for pair in parent_pairs}
    fold_metrics: dict[tuple[int, int], list[dict[str, float]]] = {
        pair: [] for pair in parent_pairs
    }

    nuisance_precision = np.repeat(
        1.0 / (NUISANCE_PRIOR_SD * NUISANCE_PRIOR_SD), nuisance.shape[1]
    )
    parent_precision = 1.0 / (PARENT_PRIOR_SD * PARENT_PRIOR_SD)
    for fold, (train, test) in enumerate(splitter.split(nuisance, y)):
        train_nuisance, test_nuisance = _scale_train_test(
            nuisance[train], nuisance[test]
        )
        baseline = _fit_map_logistic(
            train_nuisance,
            y[train],
            nuisance_precision,
        )
        probability0 = expit(
            np.column_stack([np.ones(len(test)), test_nuisance])
            @ baseline.coefficients
        )
        baseline_probability[test] = probability0
        baseline_ll[test] = _log_likelihood(y[test], probability0)

        single_fits: dict[int, _MapFit] = {}
        for parent in parent_indices:
            train_design = np.column_stack([train_nuisance, detections[train, parent]])
            test_design = np.column_stack([test_nuisance, detections[test, parent]])
            fit = _fit_map_logistic(
                train_design,
                y[train],
                np.r_[nuisance_precision, parent_precision],
            )
            single_fits[parent] = fit
            probability = expit(
                np.column_stack([np.ones(len(test)), test_design]) @ fit.coefficients
            )
            single_ll[parent][test] = _log_likelihood(y[test], probability)

        for parent1, parent2 in parent_pairs:
            train_design = np.column_stack(
                [train_nuisance, detections[train, parent1], detections[train, parent2]]
            )
            test_design = np.column_stack(
                [test_nuisance, detections[test, parent1], detections[test, parent2]]
            )
            fit = _fit_map_logistic(
                train_design,
                y[train],
                np.r_[nuisance_precision, parent_precision, parent_precision],
            )
            probability = expit(
                np.column_stack([np.ones(len(test)), test_design]) @ fit.coefficients
            )
            joint_ll[(parent1, parent2)][test] = _log_likelihood(y[test], probability)
            coefficients = fit.coefficients[-2:]
            standard_errors = np.sqrt(np.maximum(np.diag(fit.covariance)[-2:], 1e-12))
            posterior_positive = norm.cdf(coefficients / standard_errors)
            fold_metrics[(parent1, parent2)].append(
                {
                    "fold": float(fold),
                    "coefficient1": float(coefficients[0]),
                    "coefficient2": float(coefficients[1]),
                    "posterior_positive1": float(posterior_positive[0]),
                    "posterior_positive2": float(posterior_positive[1]),
                }
            )

    rows: list[dict[str, Any]] = []
    for parent1, parent2 in parent_pairs:
        metrics = fold_metrics[(parent1, parent2)]
        coefficients = np.asarray(
            [[row["coefficient1"], row["coefficient2"]] for row in metrics]
        )
        posterior = np.asarray(
            [
                [row["posterior_positive1"], row["posterior_positive2"]]
                for row in metrics
            ]
        )
        joint = joint_ll[(parent1, parent2)]
        drop_parent1 = joint - single_ll[parent2]
        drop_parent2 = joint - single_ll[parent1]
        rows.append(
            {
                "center": feature_names[center_index],
                "outer1": min(feature_names[parent1], feature_names[parent2]),
                "outer2": max(feature_names[parent1], feature_names[parent2]),
                "crossfit_joint_gain_over_baseline": float(
                    np.mean(joint - baseline_ll)
                ),
                "crossfit_drop_parent1_gain": float(np.mean(drop_parent1)),
                "crossfit_drop_parent2_gain": float(np.mean(drop_parent2)),
                "crossfit_min_drop_one_gain": float(
                    min(np.mean(drop_parent1), np.mean(drop_parent2))
                ),
                "crossfit_joint_gain_fold_min": float(
                    min(
                        np.mean((joint - baseline_ll)[test])
                        for _, test in splitter.split(nuisance, y)
                    )
                ),
                "laplace_mean_parent_coefficient_min": float(
                    coefficients.mean(axis=0).min()
                ),
                "laplace_fold_parent_coefficient_min": float(coefficients.min()),
                "laplace_min_parent_positive_posterior": float(
                    posterior.mean(axis=0).min()
                ),
                "laplace_fold_min_parent_positive_posterior": float(
                    posterior.min()
                ),
                "laplace_both_positive_fold_support": float(
                    np.mean(np.all(coefficients > 0.0, axis=1))
                ),
                "laplace_both_practical_fold_support": float(
                    np.mean(np.all(coefficients >= 0.30, axis=1))
                ),
            }
        )
    return baseline_probability, rows


def _empirical_null_pair_metrics(
    residuals: FloatArray,
    nuisance: FloatArray,
    feature_names: tuple[str, ...],
    seed: int,
    assay: str,
) -> dict[tuple[str, str], dict[str, float]]:
    observed: dict[tuple[str, str], float] = {}
    for left, right in itertools.combinations(range(len(feature_names)), 2):
        observed[(feature_names[left], feature_names[right])] = _finite_correlation(
            residuals[:, left], residuals[:, right]
        )
    resolution_values: dict[tuple[str, str], list[tuple[float, float]]] = {
        key: [] for key in observed
    }
    scaled = (nuisance - nuisance.mean(axis=0)) / np.where(
        nuisance.std(axis=0) > 1e-8, nuisance.std(axis=0), 1.0
    )
    for resolution in MICROSTATE_RESOLUTIONS:
        labels = KMeans(
            n_clusters=resolution,
            n_init=10,
            random_state=_stable_seed("microstate", seed, assay, resolution),
        ).fit_predict(scaled)
        groups = [np.flatnonzero(labels == label) for label in range(resolution)]
        for (left_name, right_name), value in observed.items():
            left = feature_names.index(left_name)
            right = feature_names.index(right_name)
            rng = np.random.default_rng(
                _stable_seed("null", seed, assay, resolution, left_name, right_name)
            )
            null = np.zeros(N_NULL_PERMUTATIONS, dtype=np.float64)
            for permutation in range(N_NULL_PERMUTATIONS):
                order = np.arange(len(residuals))
                for group in groups:
                    order[group] = rng.permutation(group)
                null[permutation] = _finite_correlation(
                    residuals[:, left], residuals[order, right]
                )
            positive_p = float(
                (1 + np.sum(null >= value)) / (N_NULL_PERMUTATIONS + 1)
            )
            absolute_p = float(
                (1 + np.sum(np.abs(null) >= abs(value)))
                / (N_NULL_PERMUTATIONS + 1)
            )
            resolution_values[(left_name, right_name)].append(
                (positive_p, absolute_p)
            )
    return {
        key: {
            "residual_correlation": observed[key],
            "empirical_positive_p_worst_resolution": max(
                value[0] for value in values
            ),
            "empirical_absolute_p_worst_resolution": min(
                value[1] for value in values
            ),
        }
        for key, values in resolution_values.items()
    }


def _assay_generative_rows(
    block: str,
    seed: int,
    assay: str,
    world_root: str,
    embedding_root: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    discovery = Path(world_root) / f"world_seed{seed}" / "discovery"
    cell_path = discovery / f"observed_focal__{assay}.tsv"
    embedding_path = (
        Path(embedding_root)
        / f"world_seed{seed}"
        / "assay_local_state_embedding_v4"
        / "analysis_embeddings.tsv"
    )
    cells = pd.read_csv(cell_path, sep="\t", keep_default_na=False)
    embedding = pd.read_csv(embedding_path, sep="\t", keep_default_na=False)
    embedding = embedding.loc[embedding["modality"].astype(str).eq(assay)].copy()
    merged = cells.merge(embedding, on=["cell", "modality"], validate="one_to_one")
    metadata = {
        "cell",
        "modality",
        "observed_background_library_size",
        "observed_library_size",
    }
    feature_names = tuple(sorted(column for column in cells.columns if column not in metadata))
    if len(feature_names) != 6:
        raise ValueError(f"Expected six focal features in {seed} {assay}")
    detections = (merged[list(feature_names)].to_numpy(dtype=np.float64) > 0).astype(
        np.float64
    )
    nuisance = _nuisance_design(merged)
    baseline_probability = np.zeros_like(detections)
    row_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for center_index in range(len(feature_names)):
        probability, rows = _crossfit_response(
            detections, nuisance, center_index, feature_names, seed, assay
        )
        baseline_probability[:, center_index] = probability
        for row in rows:
            row_map[(row["center"], row["outer1"], row["outer2"])] = row

    denominator = np.sqrt(
        np.maximum(baseline_probability * (1.0 - baseline_probability), 0.01)
    )
    residuals = (detections - baseline_probability) / denominator
    pair_metrics = _empirical_null_pair_metrics(
        residuals, nuisance, feature_names, seed, assay
    )
    global_folds = np.asarray(
        [_stable_seed("stability-fold", seed, assay, cell) % N_FOLDS for cell in merged["cell"]],
        dtype=np.int64,
    )
    rows: list[dict[str, Any]] = []
    for key, likelihood in sorted(row_map.items()):
        center, outer1, outer2 = key
        arm1 = pair_metrics[_canonical(center, outer1)]
        arm2 = pair_metrics[_canonical(center, outer2)]
        outer = pair_metrics[_canonical(outer1, outer2)]
        center_index = feature_names.index(center)
        outer1_index = feature_names.index(outer1)
        outer2_index = feature_names.index(outer2)
        fold_arm_floor: list[float] = []
        fold_outer_absolute: list[float] = []
        for fold in range(N_FOLDS):
            selected = global_folds == fold
            if selected.sum() < 3:
                continue
            corr1 = _finite_correlation(
                residuals[selected, center_index], residuals[selected, outer1_index]
            )
            corr2 = _finite_correlation(
                residuals[selected, center_index], residuals[selected, outer2_index]
            )
            corr_outer = _finite_correlation(
                residuals[selected, outer1_index], residuals[selected, outer2_index]
            )
            fold_arm_floor.append(min(corr1, corr2))
            fold_outer_absolute.append(abs(corr_outer))
        rows.append(
            {
                "block": block,
                "seed": seed,
                "source_assay": assay,
                **likelihood,
                "center_detection_rate": float(detections[:, center_index].mean()),
                "minimum_parent_detection_rate": float(
                    min(
                        detections[:, outer1_index].mean(),
                        detections[:, outer2_index].mean(),
                    )
                ),
                "maximum_parent_detection_rate": float(
                    max(
                        detections[:, outer1_index].mean(),
                        detections[:, outer2_index].mean(),
                    )
                ),
                "minimum_arm_residual_correlation": float(
                    min(arm1["residual_correlation"], arm2["residual_correlation"])
                ),
                "outer_absolute_residual_correlation": float(
                    abs(outer["residual_correlation"])
                ),
                "residual_star_separation": float(
                    min(arm1["residual_correlation"], arm2["residual_correlation"])
                    - abs(outer["residual_correlation"])
                ),
                "maximum_arm_empirical_positive_p": float(
                    max(
                        arm1["empirical_positive_p_worst_resolution"],
                        arm2["empirical_positive_p_worst_resolution"],
                    )
                ),
                "outer_empirical_null_compatibility_p": float(
                    outer["empirical_absolute_p_worst_resolution"]
                ),
                "fold_minimum_arm_residual_correlation": float(
                    min(fold_arm_floor) if fold_arm_floor else math.nan
                ),
                "fold_median_arm_residual_correlation": float(
                    np.median(fold_arm_floor) if fold_arm_floor else math.nan
                ),
                "fold_maximum_outer_absolute_correlation": float(
                    max(fold_outer_absolute) if fold_outer_absolute else math.nan
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame["trio_key"] = [
        "||".join(sorted((row.center, row.outer1, row.outer2)))
        for row in frame.itertuples(index=False)
    ]
    for metric in (
        "crossfit_joint_gain_over_baseline",
        "crossfit_min_drop_one_gain",
        "laplace_min_parent_positive_posterior",
        "residual_star_separation",
    ):
        competitor = pd.Series(math.nan, index=frame.index, dtype=float)
        for _, group in frame.groupby("trio_key", sort=False):
            for index in group.index:
                peers = group.loc[group.index != index, metric]
                competitor.loc[index] = float(peers.max())
        frame[f"{metric}_reverse_center_margin"] = frame[metric] - competitor
    frame = frame.drop(columns="trio_key")
    return frame.to_dict("records"), {
        str(cell_path): _sha256(cell_path),
        str(embedding_path): _sha256(embedding_path),
    }


def _world_generative_rows(
    arguments: tuple[str, int, str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    block, seed, world_root, embedding_root = arguments
    discovery = Path(world_root) / f"world_seed{seed}" / "discovery"
    assays = sorted(
        path.name.removeprefix("observed_focal__").removesuffix(".tsv")
        for path in discovery.glob("observed_focal__chrom_*.tsv")
    )
    if len(assays) != 6:
        raise ValueError(f"Expected six chromatin assays for {block} seed {seed}")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for assay in assays:
        assay_rows, assay_hashes = _assay_generative_rows(
            block, seed, assay, world_root, embedding_root
        )
        rows.extend(assay_rows)
        hashes.update(assay_hashes)
    return rows, hashes


def _production_phase_a(
    block: str, world: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(world["seed"])
    motif, refinement = retired._selector_inputs(world)
    production = apply_observed_collider_refinements(motif, refinement)
    assay_map = diagnostic._feature_assay_map(world)
    edge_rows: list[dict[str, Any]] = []
    for parent, child in diagnostic._directed_edges_from_frame(
        production.direct_candidates
    ):
        assay = assay_map.get(child)
        if assay is None or assay != assay_map.get(parent):
            raise ValueError(f"Production direct edge is not assay-local: {parent}, {child}")
        edge_rows.append(
            {
                "block": block,
                "seed": seed,
                "source_assay": assay,
                "parent": parent,
                "child": child,
            }
        )

    decision_risk: dict[str, list[str]] = {}
    for _, decision in refinement.decisions.iterrows():
        assay = str(decision["source_assay"])
        features = retired._decision_features(
            decision, refinement, world["subspace_features"]
        )
        action = str(decision["action"])
        reasons: list[str] = []
        if action == "complete_dual_supported_unshielded_anchor":
            if not bool(features["candidate_new_parent_formal_state_selected"]):
                reasons.append("completion_parent_not_formal_state_selected")
            if float(features["selected_arm_outer_state_support_floor"]) <= 0.0:
                reasons.append("completion_nonpositive_state_floor")
        elif action == "complete_unique_rescue_only_state_loading_contrast":
            if float(features["anchor_base_view_count"]) < 1.0:
                reasons.append("rescue_without_base_anchor")
        elif action == "resolve_competing_multicenter_path":
            if float(features["median_collider_separation_score"]) <= 0.0:
                reasons.append("multicenter_nonpositive_separation")
            if int(features["winner_view_count"]) != 6:
                reasons.append("multicenter_not_six_of_six")
            if float(features["median_outer_state_support_floor"]) <= 0.0025:
                reasons.append("multicenter_weak_state_floor")
        decision_risk.setdefault(assay, []).extend(reasons)

    assays = sorted(set(assay_map.values()))
    revised_state_members = _redundancy_filtered_state_members(world)
    status_rows: list[dict[str, Any]] = []
    for assay in assays:
        assay_edges = [row for row in edge_rows if row["source_assay"] == assay]
        centers = {row["child"] for row in assay_edges}
        structural_risk = len(assay_edges) != 2 or len(centers) != 1
        reasons = list(decision_risk.get(assay, []))
        if structural_risk:
            reasons.append("production_not_exact_two_arm_star")
        if any(str(row["parent"]) not in revised_state_members for row in assay_edges):
            reasons.append("production_parent_redundancy_demoted")
        status_rows.append(
            {
                "block": block,
                "seed": seed,
                "source_assay": assay,
                "n_production_direct_edges": len(assay_edges),
                "n_production_centers": len(centers),
                "risk_trigger": bool(reasons),
                "risk_reasons": ";".join(sorted(set(reasons))),
            }
        )
    return edge_rows, status_rows


def _redundancy_filtered_state_members(world: dict[str, Any]) -> set[str]:
    """Replay the landed v7 subtractive direction-stability state rule."""

    frame = world["subspace_features"].copy()
    preliminary = frame.loc[
        frame["selection_status"].astype(str).eq("selected")
    ].copy()
    retained: set[str] = set()
    for _, assay in preliminary.groupby("modality", sort=True):
        stable = assay.loc[assay["direction_stability_diagnostic"].astype(bool)]
        if len(stable):
            retained.update(stable["feature"].astype(str))
            continue
        winner = assay.sort_values(
            ["feature_global_maxT_p", "reference_conditional_wald", "feature"],
            ascending=[True, False, True],
        ).iloc[0]
        retained.add(str(winner["feature"]))
    return retained


def _augment_candidate_state_consistency(
    rows: list[dict[str, Any]], world: dict[str, Any]
) -> list[dict[str, Any]]:
    membership = world["subspace_features"]
    if membership["feature"].astype(str).duplicated().any():
        raise ValueError("State membership has duplicate feature rows")
    state = {
        str(row.feature): row
        for row in membership.itertuples(index=False)
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        parents = [state[str(row["outer1"])], state[str(row["outer2"])] ]
        partial = [float(parent.oof_partial_state_fit) for parent in parents]
        wald_support = [float(parent.conditional_wald_view_support) for parent in parents]
        above_null = [
            float(parent.conditional_wald_above_null_95pct) for parent in parents
        ]
        reference_wald = [float(parent.reference_conditional_wald) for parent in parents]
        max_t = [
            bool(parent.feature_global_maxT_confirmation_gate) for parent in parents
        ]
        output.append(
            {
                **row,
                "outer_positive_partial_state_count": int(
                    sum(value > 0.0 for value in partial)
                ),
                "minimum_outer_partial_state_fit": float(min(partial)),
                "minimum_outer_conditional_wald_view_support": float(
                    min(wald_support)
                ),
                "minimum_outer_conditional_wald_above_null_95pct": float(
                    min(above_null)
                ),
                "minimum_outer_reference_conditional_wald": float(
                    min(reference_wald)
                ),
                "outer_feature_maxT_confirmation_count": int(sum(max_t)),
                "two_parent_positive_state_wald_consistency": bool(
                    min(partial) > 0.0 and min(wald_support) >= 1.0
                ),
            }
        )
    return output


def _verify_phase_a(out_dir: Path) -> dict[str, Any]:
    manifest = json.loads((out_dir / "phase_a_manifest.json").read_text())
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Phase-A truth barrier is missing")
    if manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Phase A reports truth/scorer access")
    for name, expected in manifest["output_sha256"].items():
        if _sha256(out_dir / name) != expected:
            raise ValueError(f"Phase-A output drift: {name}")
    return manifest


def _candidate_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    excluded = {
        "seed",
        "center",
        "outer1",
        "outer2",
        "arm1",
        "arm2",
        "outer_pair",
        "truth_exact_star",
        "target_train380_failure",
        "block",
        "source_assay",
    }
    numeric = [
        column
        for column in frame.columns
        if column not in excluded
        and (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]
    matrix = frame[numeric].copy()
    for column in numeric:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    group = frame.groupby(["seed", "source_assay"], sort=False)
    for column in numeric:
        matrix[f"rank__{column}"] = group[column].rank(
            method="average", pct=True, na_option="bottom"
        )
    matrix["source_assay"] = frame["source_assay"].astype(str)
    numeric_columns = [column for column in matrix if column != "source_assay"]
    return matrix, numeric_columns, ["source_assay"]


def _models(
    numeric_columns: list[str], categorical_columns: list[str]
) -> dict[str, Pipeline]:
    def preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "assay",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                ),
            ]
        )

    output: dict[str, Pipeline] = {}
    for c in (0.03, 0.30):
        output[f"calibrated_logit_C{c:g}"] = Pipeline(
            [
                ("features", preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=4000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    output["calibrated_rf_leaf8"] = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            SimpleImputer(strategy="median"),
                            numeric_columns,
                        ),
                        (
                            "assay",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_columns,
                        ),
                    ]
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    min_samples_leaf=8,
                    max_features=0.8,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=1701,
                ),
            ),
        ]
    )
    return output


def _top_predictions(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    selected = frame.copy()
    selected["selected"] = False
    selected["candidate_margin"] = math.nan
    for _, group in selected.groupby(["block", "seed", "source_assay"], sort=True):
        ordered = group.sort_values(
            [score_column, "center", "outer1", "outer2"],
            ascending=[False, True, True, True],
        )
        winner = ordered.index[0]
        runner = float(ordered.iloc[1][score_column]) if len(ordered) > 1 else -math.inf
        selected.loc[winner, "selected"] = True
        selected.loc[winner, "candidate_margin"] = float(
            ordered.iloc[0][score_column] - runner
        )
    return selected.loc[selected["selected"]].copy()


def _prediction_edges(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for parent in (str(row.outer1), str(row.outer2)):
            rows.append(
                {
                    "block": str(row.block),
                    "seed": int(row.seed),
                    "source_assay": str(row.source_assay),
                    "parent": parent,
                    "child": str(row.center),
                }
            )
    return pd.DataFrame(rows)


def _score_edges(
    method: str,
    predicted: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = truth[["block", "seed"]].drop_duplicates().itertuples(index=False)
    for block, seed in keys:
        selected = predicted.loc[
            predicted["block"].eq(block) & predicted["seed"].eq(seed)
        ]
        target = truth.loc[truth["block"].eq(block) & truth["seed"].eq(seed)]
        predicted_set = {
            (str(row.parent), str(row.child)) for row in selected.itertuples(index=False)
        }
        truth_set = {
            (str(row.parent), str(row.child)) for row in target.itertuples(index=False)
        }
        tp = len(predicted_set & truth_set)
        fp = len(predicted_set - truth_set)
        fn = len(truth_set - predicted_set)
        reverse = sum((child, parent) in truth_set for parent, child in predicted_set - truth_set)
        precision = tp / len(predicted_set) if predicted_set else math.nan
        recall = tp / len(truth_set) if truth_set else math.nan
        rows.append(
            {
                "method": method,
                "block": block,
                "seed": int(seed),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "strict_reverse": reverse,
                "precision": precision,
                "recall": recall,
                "direct_reliable": bool(
                    precision >= 0.80 and recall >= 0.90 and reverse == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(world_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, method_frame in world_metrics.groupby("method", sort=True):
        scopes = [("ALL_RETIRED", method_frame)] + [
            (block, method_frame.loc[method_frame["block"].eq(block)])
            for block in BLOCKS
        ]
        for scope, frame in scopes:
            tp = int(frame["tp"].sum())
            fp = int(frame["fp"].sum())
            fn = int(frame["fn"].sum())
            rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "strict_reverse": int(frame["strict_reverse"].sum()),
                    "precision": tp / (tp + fp) if tp + fp else math.nan,
                    "recall": tp / (tp + fn) if tp + fn else math.nan,
                    "reliable_worlds": int(frame["direct_reliable"].sum()),
                    "n_worlds": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_crossfit_hierarchical_collider_audit_v1"
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    # Phase A: enumerate and materialize every observed candidate before truth.
    held: dict[str, list[dict[str, Any]]] = {}
    base_rows: list[dict[str, Any]] = []
    production_edges: list[dict[str, Any]] = []
    production_status: list[dict[str, Any]] = []
    tasks: list[tuple[str, int, str, str]] = []
    freeze_hashes: dict[str, str] = {}
    for block in BLOCKS:
        worlds = retired._load_block_worlds(block)
        held[block] = worlds
        for world in worlds:
            base_rows.extend(
                _augment_candidate_state_consistency(
                    star_audit._phase_a_world(block, world), world
                )
            )
            edges, status = _production_phase_a(block, world)
            production_edges.extend(edges)
            production_status.extend(status)
            tasks.append(
                (
                    block,
                    int(world["seed"]),
                    str(retired.BLOCKS[block]["world_root"]),
                    str(
                        Path(
                            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_v1/artifacts"
                        )
                        if block == "TRAIN330"
                        else Path(retired.BLOCKS[block]["run_root"]) / "artifacts"
                    ),
                )
            )
        if block != "TRAIN330":
            freeze = Path(retired.BLOCKS[block]["run_root"]) / "global_pretruth_freeze.json"
            freeze_hashes[str(freeze)] = _sha256(freeze)

    generative_rows: list[dict[str, Any]] = []
    discovery_hashes: dict[str, str] = {}
    if args.workers == 1:
        results = map(_world_generative_rows, tasks)
        for rows, hashes in results:
            generative_rows.extend(rows)
            discovery_hashes.update(hashes)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for rows, hashes in executor.map(_world_generative_rows, tasks):
                generative_rows.extend(rows)
                discovery_hashes.update(hashes)

    base = pd.DataFrame(base_rows)
    generative = pd.DataFrame(generative_rows)
    keys = ["block", "seed", "source_assay", "center", "outer1", "outer2"]
    if base.duplicated(keys).any() or generative.duplicated(keys).any():
        raise ValueError("Candidate identity is not unique")
    candidates = base.merge(generative, on=keys, how="inner", validate="one_to_one")
    if len(candidates) != len(base) or len(candidates) != len(generative):
        raise ValueError("Observed and generative candidate universes differ")
    candidates = candidates.sort_values(keys).reset_index(drop=True)
    candidate_path = args.out_dir / "phase_a_crossfit_star_candidates.tsv"
    edge_path = args.out_dir / "phase_a_production_direct_edges.tsv"
    status_path = args.out_dir / "phase_a_production_assay_status.tsv"
    candidates.to_csv(candidate_path, sep="\t", index=False)
    pd.DataFrame(production_edges).sort_values(
        ["block", "seed", "source_assay", "parent", "child"]
    ).to_csv(edge_path, sep="\t", index=False)
    pd.DataFrame(production_status).sort_values(
        ["block", "seed", "source_assay"]
    ).to_csv(status_path, sep="\t", index=False)
    formula_path = args.out_dir / "phase_a_formula_family.json"
    _write_json(
        formula_path,
        {
            "schema_version": SCHEMA,
            "phase": "observed_only_before_truth",
            "family": FORMULA_FAMILY,
        },
    )
    phase_a_manifest = {
        "schema_version": SCHEMA,
        "phase": "observed_only_before_truth",
        "blocks": list(BLOCKS),
        "n_worlds": len(tasks),
        "n_candidates": len(candidates),
        "candidate_universe": "all 60 directed two-parent stars per six-feature chromatin assay",
        "state_nuisance": "quadratic assay-local held-out-background integration coordinates plus log background depth and uncertainty",
        "sparse_model": "cross-fitted Bernoulli MAP with Normal(0,2.5^2) parent prior and Laplace covariance",
        "empirical_null": {
            "state_depth_microstate_resolutions": list(MICROSTATE_RESOLUTIONS),
            "permutations_per_resolution": N_NULL_PERMUTATIONS,
        },
        "truth_or_scorer_opened": False,
        "dev_or_locked_opened": False,
        "input_global_pretruth_freeze_sha256": freeze_hashes,
        "input_discovery_sha256": discovery_hashes,
        "output_sha256": {
            path.name: _sha256(path)
            for path in (candidate_path, edge_path, status_path, formula_path)
        },
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.out_dir / "phase_a_manifest.json", phase_a_manifest)

    # Phase B: the manifest/hash barrier is re-read before scorer truth access.
    _verify_phase_a(args.out_dir)
    truth_frames: list[pd.DataFrame] = []
    truth_stars: dict[int, set[tuple[str, str, str]]] = {}
    for block in BLOCKS:
        world_root = Path(retired.BLOCKS[block]["world_root"])
        truth, _ = diagnostic._load_direct_truth_after_barrier(held[block], world_root)
        truth.insert(0, "block", block)
        truth_frames.append(truth)
        truth_stars.update(star_audit._truth_stars(truth))
    truth = pd.concat(truth_frames, ignore_index=True, sort=False)
    labels = candidates.copy()
    labels["truth_exact_star"] = [
        (str(row.center), str(row.outer1), str(row.outer2))
        in truth_stars.get(int(row.seed), set())
        for row in labels.itertuples(index=False)
    ]
    labels["target_train380_failure"] = labels["seed"].isin(TARGET_SEEDS)
    label_path = args.out_dir / "phase_b_labeled_crossfit_star_candidates.tsv"
    labels.to_csv(label_path, sep="\t", index=False)

    matrix, numeric_columns, categorical_columns = _candidate_matrix(labels)
    calibrated_models = _models(numeric_columns, categorical_columns)
    selected_frames: list[pd.DataFrame] = []
    world_metric_frames: list[pd.DataFrame] = []

    production = pd.read_csv(edge_path, sep="\t")
    world_metric_frames.append(_score_edges("production", production, truth))
    assay_status = pd.read_csv(status_path, sep="\t")

    # Fixed, truth-free likelihood guard.  It is intentionally conservative.
    strong = FORMULA_FAMILY["strong_generative"]
    strong_mask = (
        labels["laplace_min_parent_positive_posterior"].ge(
            strong["minimum_parent_positive_posterior"]
        )
        & labels["crossfit_joint_gain_over_baseline"].gt(
            strong["minimum_joint_gain_over_baseline"]
        )
        & labels["crossfit_min_drop_one_gain"].gt(
            strong["minimum_drop_one_gain"]
        )
        & labels["laplace_both_positive_fold_support"].ge(
            strong["minimum_both_positive_fold_support"]
        )
        & labels["crossfit_joint_gain_over_baseline_reverse_center_margin"].gt(
            strong["minimum_reverse_center_gain_margin"]
        )
        & labels["maximum_arm_empirical_positive_p"].le(
            strong["maximum_arm_empirical_positive_p"]
        )
        & labels["outer_empirical_null_compatibility_p"].ge(
            strong["minimum_outer_empirical_null_compatibility_p"]
        )
    )
    guarded = labels.loc[strong_mask].copy()
    guarded["guard_score"] = (
        guarded["crossfit_joint_gain_over_baseline"]
        + guarded["crossfit_min_drop_one_gain"]
        + 0.02 * guarded["laplace_min_parent_positive_posterior"]
        + 0.25 * guarded["residual_star_separation"]
    )
    guarded_selected = _top_predictions(guarded, "guard_score") if len(guarded) else guarded
    guarded_selected.insert(0, "method", "laplace_empirical_null_guarded")
    selected_frames.append(guarded_selected)
    guarded_edges = _prediction_edges(guarded_selected)
    world_metric_frames.append(
        _score_edges("laplace_empirical_null_guarded", guarded_edges, truth)
    )

    # A two-tier risk replacement: ordinary guard failures require both
    # proposed parents to survive conditional-Wald/state checks.  An actually
    # incomplete motif may instead use uniformly positive six-view collider
    # separation, but only with the same strong generative/null evidence.
    risk_candidates = labels.assign(_strong_generative=strong_mask).merge(
        assay_status.loc[assay_status["risk_trigger"].astype(bool)],
        on=["block", "seed", "source_assay"],
        how="inner",
        validate="many_to_one",
    )
    risk_strong = risk_candidates["_strong_generative"].astype(bool)
    incomplete = risk_candidates["risk_reasons"].fillna("").str.contains(
        "production_not_exact_two_arm_star", regex=False
    )
    state_wald = risk_candidates[
        "two_parent_positive_state_wald_consistency"
    ].astype(bool)
    stable_incomplete = risk_candidates["minimum_collider_separation"].gt(0.0)
    two_tier = risk_candidates.loc[
        risk_strong
        & ((~incomplete & state_wald) | (incomplete & (state_wald | stable_incomplete)))
    ].copy()
    two_tier["guard_score"] = (
        two_tier["crossfit_joint_gain_over_baseline"]
        + two_tier["crossfit_min_drop_one_gain"]
        + 0.02 * two_tier["laplace_min_parent_positive_posterior"]
        + 0.25 * two_tier["residual_star_separation"]
    )
    two_tier_selected = (
        _top_predictions(two_tier, "guard_score") if len(two_tier) else two_tier
    )
    two_tier_selected.insert(0, "method", "two_tier_state_wald_guard")
    selected_frames.append(two_tier_selected)
    two_tier_edges = _prediction_edges(two_tier_selected)
    world_metric_frames.append(
        _score_edges("two_tier_state_wald_guard", two_tier_edges, truth)
    )
    all_risky_keys = set(
        map(
            tuple,
            assay_status.loc[
                assay_status["risk_trigger"].astype(bool),
                ["block", "seed", "source_assay"],
            ].itertuples(index=False, name=None),
        )
    )
    replacement_keys = set(
        map(
            tuple,
            two_tier_edges[["block", "seed", "source_assay"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    for fallback in ("retain", "abstain"):
        removed_keys = replacement_keys if fallback == "retain" else all_risky_keys
        base = production.loc[
            [
                (str(row.block), int(row.seed), str(row.source_assay))
                not in removed_keys
                for row in production.itertuples(index=False)
            ]
        ]
        hybrid = pd.concat([base, two_tier_edges], ignore_index=True, sort=False)
        method = f"two_tier_risk_hybrid_{fallback}"
        world_metric_frames.append(_score_edges(method, hybrid, truth))

    for held_block in BLOCKS:
        train = labels["block"].ne(held_block)
        test = ~train
        for name, model in calibrated_models.items():
            model.fit(matrix.loc[train], labels.loc[train, "truth_exact_star"])
            scored = labels.loc[test].copy()
            scored["calibrated_score"] = model.predict_proba(matrix.loc[test])[:, 1]
            chosen = _top_predictions(scored, "calibrated_score")
            chosen.insert(0, "method", name)
            selected_frames.append(chosen)

    selected = pd.concat(selected_frames, ignore_index=True, sort=False)
    for method, chosen in selected.groupby("method", sort=True):
        if method == "laplace_empirical_null_guarded":
            continue
        chosen_edges = _prediction_edges(chosen)
        world_metric_frames.append(_score_edges(method, chosen_edges, truth))

        # Replace only production assays carrying an observed-only risk trigger.
        risky = assay_status.loc[assay_status["risk_trigger"].astype(bool), [
            "block", "seed", "source_assay"
        ]]
        risky_keys = set(map(tuple, risky.itertuples(index=False, name=None)))
        hybrid = production.loc[
            [
                (str(row.block), int(row.seed), str(row.source_assay)) not in risky_keys
                for row in production.itertuples(index=False)
            ]
        ].copy()
        replacement = chosen_edges.loc[
            [
                (str(row.block), int(row.seed), str(row.source_assay)) in risky_keys
                for row in chosen_edges.itertuples(index=False)
            ]
        ]
        hybrid = pd.concat([hybrid, replacement], ignore_index=True, sort=False)
        hybrid_method = f"risk_trigger_hybrid__{method}"
        world_metric_frames.append(_score_edges(hybrid_method, hybrid, truth))

    selected_path = args.out_dir / "phase_b_selected_stars.tsv"
    selected.to_csv(selected_path, sep="\t", index=False)
    world_metrics = pd.concat(world_metric_frames, ignore_index=True, sort=False)
    world_path = args.out_dir / "phase_b_world_direct_metrics.tsv"
    world_metrics.to_csv(world_path, sep="\t", index=False)
    aggregate = _aggregate_metrics(world_metrics)
    aggregate_path = args.out_dir / "phase_b_decoder_metrics.tsv"
    aggregate.to_csv(aggregate_path, sep="\t", index=False)
    target_path = args.out_dir / "phase_b_train380_target_metrics.tsv"
    world_metrics.loc[world_metrics["seed"].isin(TARGET_SEEDS)].sort_values(
        ["method", "seed"]
    ).to_csv(target_path, sep="\t", index=False)

    summary = {
        "schema_version": SCHEMA,
        "status": "posthoc_retired_train_diagnostic_not_production_certification",
        "phase_a_verified_before_truth_access": True,
        "dev_or_locked_opened": False,
        "n_phase_a_candidates": int(len(candidates)),
        "n_true_star_labels": int(labels["truth_exact_star"].sum()),
        "methods": sorted(world_metrics["method"].unique()),
        "target_seeds": list(TARGET_SEEDS),
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "phase_a_manifest_sha256": _sha256(args.out_dir / "phase_a_manifest.json"),
        "output_sha256": {
            path.name: _sha256(path)
            for path in (
                candidate_path,
                edge_path,
                status_path,
                label_path,
                selected_path,
                world_path,
                aggregate_path,
                target_path,
                summary_path,
            )
        },
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
