#!/usr/bin/env python3
"""Published-family baseline for unpaired multi-domain latent alignment.

This module implements the finite-sample matching step of Sturma, Squires,
Drton, and Uhler (NeurIPS 2023), *Unpaired Multi-Domain Causal
Representation Learning*.  Each assay is decomposed independently with linear
ICA.  Standardized component distributions are compared with the
sign-invariant two-sample Kolmogorov--Smirnov statistic.  A pair is matched only
when it is the mutual minimum for its two components and passes the paper's
Bonferroni-adjusted asymptotic critical value.  A shared component tuple must
match consistently across every assay.

The original theorem assumes a noiseless injective linear mixing model and
chooses the ICA dimension by thresholding the observed covariance rank.  The
simulation used by CA-BOSS has log-linear Poisson observations and therefore a
high observed rank.  This implementation reports that rank explicitly and
runs a declared fixed-rank-three adaptation because three background-state
coordinates are already part of the integration contract.  The adaptation is
a diagnostic baseline, not an application of the theorem.

Only held-out background matrices and their row sums are accepted.  Focal
features, cell correspondence across assays, simulator state, and scorer truth
are outside the API and filesystem boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.stats import ks_2samp, kstwobign
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]

SCHEMA = "unpaired_mdcr_ks_baseline_v4.v3"
ALGORITHM = "Sturma-et-al.-2023 Algorithm-3 KS matching, CA-BOSS adaptation"
PAPER_URL = (
    "https://proceedings.neurips.cc/paper_files/paper/2023/"
    "file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf"
)
_BACKGROUND_PREFIX = "background_matrix__"
_FORBIDDEN_PATH_PARTS = {
    "counterfactual",
    "ground_truth",
    "oracle",
    "scorer",
    "truth",
}


@dataclass(frozen=True)
class UnpairedMDCRConfig:
    """Frozen settings for the published-family diagnostic."""

    n_components: int = 3
    alpha: float = 0.05
    rank_gamma: float = 0.20
    random_seed: int = 0
    n_feature_folds: int = 2
    fastica_max_iter: int = 3_000
    fastica_tolerance: float = 1e-5
    max_dense_entries: int = 50_000_000

    def __post_init__(self) -> None:
        if self.n_components != 3:
            raise ValueError("The CA-BOSS integration contract has rank three")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.rank_gamma <= 0.0:
            raise ValueError("rank_gamma must be positive")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        if self.n_feature_folds < 2:
            raise ValueError("n_feature_folds must be at least two")
        if self.fastica_max_iter < 100 or self.fastica_tolerance <= 0.0:
            raise ValueError("FastICA iteration settings are invalid")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")


@dataclass(frozen=True)
class UnpairedMDCRResult:
    component_scores: dict[str, FloatArray]
    fold_component_scores: list[dict[str, FloatArray]]
    mixing_matrices: dict[str, FloatArray]
    empirical_covariance_ranks: dict[str, int]
    pairwise_matches: pd.DataFrame
    shared_component_tuples: pd.DataFrame
    consensus_component_tuples: pd.DataFrame
    fold_component_evidence: pd.DataFrame
    fold_pairwise_matches: pd.DataFrame
    fold_consensus_component_tuples: pd.DataFrame
    provenance: dict[str, Any]

    @property
    def n_shared_components(self) -> int:
        return int(len(self.shared_component_tuples))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(base: int, token: str) -> int:
    digest = hashlib.sha256(f"{base}::{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _as_matrix(matrix: ArrayLike, *, max_entries: int) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 20 or values.shape[1] < 6:
        raise ValueError("Each assay needs at least 20 cells and six backgrounds")
    if int(np.prod(values.shape)) > max_entries:
        raise ValueError("Observed background matrix exceeds max_dense_entries")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Background counts must be finite and nonnegative")
    return values


def _as_depth(depth: ArrayLike, *, matrix: FloatArray) -> FloatArray:
    values = np.asarray(depth, dtype=np.float64).reshape(-1)
    if values.shape != (len(matrix),):
        raise ValueError("Background depth must match the number of cells")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Background depth must be finite and positive")
    if not np.allclose(matrix.sum(axis=1), values, rtol=1e-8, atol=1e-8):
        raise ValueError("Background depth must equal the observed row sum")
    return values


def _normalize_counts(matrix: FloatArray, depth: FloatArray) -> FloatArray:
    scale = float(np.median(depth))
    logged = np.log1p(matrix * (scale / depth)[:, None])
    return np.asarray(StandardScaler().fit_transform(logged), dtype=np.float64)


def _empirical_rank(normalized: FloatArray, *, gamma: float) -> int:
    covariance = normalized.T @ normalized / float(len(normalized))
    singular = np.linalg.svd(covariance, compute_uv=False)
    return int(np.sum(singular > gamma))


def _fit_ica(
    normalized: FloatArray,
    *,
    modality: str,
    config: UnpairedMDCRConfig,
) -> tuple[FloatArray, FloatArray, bool, int]:
    model = FastICA(
        n_components=config.n_components,
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=config.fastica_max_iter,
        tol=config.fastica_tolerance,
        random_state=_stable_seed(config.random_seed, modality),
        whiten_solver="svd",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        scores = np.asarray(model.fit_transform(normalized), dtype=np.float64)
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    iterations = int(getattr(model, "n_iter_", config.fastica_max_iter))
    centered = scores - scores.mean(axis=0)
    scale = centered.std(axis=0, ddof=0)
    if np.any(scale <= 1e-12):
        raise ValueError(f"FastICA returned a degenerate component for {modality}")
    scores = centered / scale
    mixing = np.asarray(model.mixing_, dtype=np.float64) * scale[None, :]
    converged = not warned and iterations < config.fastica_max_iter
    return scores, mixing, converged, iterations


def _align_fold_scores(
    fold_scores: FloatArray, full_scores: FloatArray
) -> tuple[FloatArray, list[float], float]:
    """Align a same-assay feature-fold ICA fit to the full fit."""

    n_components = full_scores.shape[1]
    correlations = np.corrcoef(full_scores.T, fold_scores.T)[
        :n_components, n_components:
    ]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for order in permutations(range(n_components)):
        value = float(
            np.mean(
                [abs(correlations[index, order[index]]) for index in range(n_components)]
            )
        )
        candidates.append((value, tuple(int(component) for component in order)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_order = candidates[0]
    second_score = candidates[1][0]
    aligned = np.empty_like(full_scores)
    matched: list[float] = []
    for full_component, fold_component in enumerate(best_order):
        correlation = float(correlations[full_component, fold_component])
        aligned[:, full_component] = fold_scores[:, fold_component] * (
            1.0 if correlation >= 0.0 else -1.0
        )
        matched.append(abs(correlation))
    margin = float((best_score - second_score) / max(best_score, 1e-12))
    return aligned, matched, margin


def _signed_ks(
    left: FloatArray, right: FloatArray
) -> tuple[float, float, float, int]:
    same = float(ks_2samp(left, right, method="asymp").statistic)
    flipped = float(ks_2samp(left, -right, method="asymp").statistic)
    if same <= flipped:
        return same, same, flipped, 1
    return flipped, same, flipped, -1


def _pairwise_component_table(
    scores: Mapping[str, FloatArray], *, alpha: float
) -> tuple[pd.DataFrame, float, float]:
    modalities = sorted(scores)
    dimensions = {matrix.shape[1] for matrix in scores.values()}
    if len(dimensions) != 1:
        raise ValueError("Every assay must expose the same adapted ICA rank")
    n_components = dimensions.pop()
    tests = 2 * sum(
        n_components * n_components for _ in combinations(modalities, 2)
    )
    adjusted_alpha = alpha / float(tests)
    critical = float(kstwobign.ppf(1.0 - adjusted_alpha))
    rows: list[dict[str, Any]] = []
    for left_modality, right_modality in combinations(modalities, 2):
        left_scores = scores[left_modality]
        right_scores = scores[right_modality]
        statistic = np.empty((n_components, n_components), dtype=float)
        raw: dict[tuple[int, int], tuple[float, float, int]] = {}
        scale = np.sqrt(
            len(left_scores)
            * len(right_scores)
            / float(len(left_scores) + len(right_scores))
        )
        for left_component in range(n_components):
            for right_component in range(n_components):
                distance, same, flipped, sign = _signed_ks(
                    left_scores[:, left_component],
                    right_scores[:, right_component],
                )
                statistic[left_component, right_component] = scale * distance
                raw[(left_component, right_component)] = (same, flipped, sign)
        row_minimum = statistic.min(axis=1)
        column_minimum = statistic.min(axis=0)
        for left_component in range(n_components):
            for right_component in range(n_components):
                value = float(statistic[left_component, right_component])
                same, flipped, sign = raw[(left_component, right_component)]
                mutual = bool(
                    np.isclose(value, row_minimum[left_component])
                    and np.isclose(value, column_minimum[right_component])
                )
                rows.append(
                    {
                        "left_modality": left_modality,
                        "right_modality": right_modality,
                        "left_component": left_component,
                        "right_component": right_component,
                        "ks_distance_sign_invariant": min(same, flipped),
                        "ks_distance_same_sign": same,
                        "ks_distance_flipped_sign": flipped,
                        "right_sign_relative_to_left": sign,
                        "scaled_ks_statistic": value,
                        "mutual_minimum": mutual,
                        "bonferroni_adjusted_alpha": adjusted_alpha,
                        "asymptotic_critical_value": critical,
                        "matched": bool(mutual and value <= critical),
                    }
                )
    return pd.DataFrame(rows), adjusted_alpha, critical


def _match_lookup(
    pairwise: pd.DataFrame,
) -> tuple[dict[tuple[str, int, str, int], bool], dict[tuple[str, int, str, int], float]]:
    matched: dict[tuple[str, int, str, int], bool] = {}
    statistic: dict[tuple[str, int, str, int], float] = {}
    for row in pairwise.itertuples(index=False):
        forward = (
            str(row.left_modality),
            int(row.left_component),
            str(row.right_modality),
            int(row.right_component),
        )
        reverse = (forward[2], forward[3], forward[0], forward[1])
        for key in (forward, reverse):
            matched[key] = bool(row.matched)
            statistic[key] = float(row.scaled_ks_statistic)
    return matched, statistic


def _consistent_shared_tuples(
    scores: Mapping[str, FloatArray], pairwise: pd.DataFrame
) -> pd.DataFrame:
    modalities = sorted(scores)
    n_components = next(iter(scores.values())).shape[1]
    matched, statistic = _match_lookup(pairwise)
    candidates_by_reference: dict[int, list[tuple[int, ...]]] = {
        component: [] for component in range(n_components)
    }
    for reference_component in range(n_components):
        for remaining in product(
            range(n_components), repeat=len(modalities) - 1
        ):
            assignment = (reference_component, *remaining)
            consistent = all(
                matched[
                    (
                        modalities[left_index],
                        assignment[left_index],
                        modalities[right_index],
                        assignment[right_index],
                    )
                ]
                for left_index, right_index in combinations(
                    range(len(modalities)), 2
                )
            )
            if consistent:
                candidates_by_reference[reference_component].append(assignment)

    best: list[tuple[int, ...]] = []
    best_score = float("inf")

    def assignment_score(values: Sequence[tuple[int, ...]]) -> float:
        return float(
            sum(
                statistic[
                    (
                        modalities[left_index],
                        assignment[left_index],
                        modalities[right_index],
                        assignment[right_index],
                    )
                ]
                for assignment in values
                for left_index, right_index in combinations(
                    range(len(modalities)), 2
                )
            )
        )

    def search(
        reference_component: int,
        chosen: list[tuple[int, ...]],
        used: list[set[int]],
    ) -> None:
        nonlocal best, best_score
        if reference_component == n_components:
            score = assignment_score(chosen)
            if len(chosen) > len(best) or (
                len(chosen) == len(best) and score < best_score
            ):
                best = list(chosen)
                best_score = score
            return
        search(reference_component + 1, chosen, used)
        for assignment in candidates_by_reference[reference_component]:
            if any(
                assignment[index] in used[index]
                for index in range(len(modalities))
            ):
                continue
            for index, component in enumerate(assignment):
                used[index].add(component)
            chosen.append(assignment)
            search(reference_component + 1, chosen, used)
            chosen.pop()
            for index, component in enumerate(assignment):
                used[index].remove(component)

    search(0, [], [set() for _ in modalities])
    rows: list[dict[str, Any]] = []
    for shared_index, assignment in enumerate(sorted(best), start=1):
        pair_statistics = [
            statistic[
                (
                    modalities[left_index],
                    assignment[left_index],
                    modalities[right_index],
                    assignment[right_index],
                )
            ]
            for left_index, right_index in combinations(
                range(len(modalities)), 2
            )
        ]
        rows.append(
            {
                "shared_component_id": f"published_mdcr_shared_{shared_index:03d}",
                "component_assignment_json": json.dumps(
                    dict(zip(modalities, assignment, strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "modalities": ";".join(modalities),
                "n_modalities": len(modalities),
                "maximum_pairwise_scaled_ks": max(pair_statistics),
                "mean_pairwise_scaled_ks": float(np.mean(pair_statistics)),
                "all_pairwise_matches": True,
                "exact_biological_axis_claimed": False,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "shared_component_id",
            "component_assignment_json",
            "modalities",
            "n_modalities",
            "maximum_pairwise_scaled_ks",
            "mean_pairwise_scaled_ks",
            "all_pairwise_matches",
            "exact_biological_axis_claimed",
        ],
    )


def _best_global_consensus(
    scores: Mapping[str, FloatArray], pairwise: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synchronize all components by maximum KS-match consensus.

    This is a CA-BOSS relaxation, not part of published Algorithm 3.  It keeps
    the signed-permutation family but returns the best full permutation even
    when the paper's all-pairs acceptance rule abstains.
    """

    modalities = sorted(scores)
    n_components = next(iter(scores.values())).shape[1]
    matched, statistic = _match_lookup(pairwise)
    orders = tuple(permutations(range(n_components)))
    identity = tuple(range(n_components))
    best_assignment: tuple[tuple[int, ...], ...] | None = None
    best_matches = -1
    best_statistic = float("inf")
    second_matches = -1
    second_statistic = float("inf")
    evaluated = 0

    def objective(
        assignment: tuple[tuple[int, ...], ...]
    ) -> tuple[int, float]:
        match_count = 0
        total_statistic = 0.0
        for axis in range(n_components):
            for left_index, right_index in combinations(
                range(len(modalities)), 2
            ):
                key = (
                    modalities[left_index],
                    assignment[left_index][axis],
                    modalities[right_index],
                    assignment[right_index][axis],
                )
                match_count += int(matched[key])
                total_statistic += statistic[key]
        return match_count, total_statistic

    for remaining in product(orders, repeat=len(modalities) - 1):
        assignment = (identity, *remaining)
        match_count, total_statistic = objective(assignment)
        evaluated += 1
        if match_count > best_matches or (
            match_count == best_matches and total_statistic < best_statistic
        ):
            second_matches, second_statistic = best_matches, best_statistic
            best_assignment = assignment
            best_matches, best_statistic = match_count, total_statistic
        elif match_count > second_matches or (
            match_count == second_matches and total_statistic < second_statistic
        ):
            second_matches, second_statistic = match_count, total_statistic

    if best_assignment is None:  # pragma: no cover - product is nonempty
        raise RuntimeError("Global signed-permutation search produced no assignment")
    rows: list[dict[str, Any]] = []
    pairs_per_axis = len(modalities) * (len(modalities) - 1) // 2
    for axis in range(n_components):
        component_assignment = {
            modality: int(best_assignment[index][axis])
            for index, modality in enumerate(modalities)
        }
        axis_matches = 0
        axis_statistics: list[float] = []
        for left_index, right_index in combinations(
            range(len(modalities)), 2
        ):
            key = (
                modalities[left_index],
                best_assignment[left_index][axis],
                modalities[right_index],
                best_assignment[right_index][axis],
            )
            axis_matches += int(matched[key])
            axis_statistics.append(statistic[key])
        best_clique: tuple[str, ...] = ()
        best_clique_statistic = float("inf")
        for clique_size in range(len(modalities), 1, -1):
            candidates: list[tuple[float, tuple[str, ...]]] = []
            for subset in combinations(modalities, clique_size):
                if not all(
                    matched[
                        (
                            left,
                            component_assignment[left],
                            right,
                            component_assignment[right],
                        )
                    ]
                    for left, right in combinations(subset, 2)
                ):
                    continue
                total = sum(
                    statistic[
                        (
                            left,
                            component_assignment[left],
                            right,
                            component_assignment[right],
                        )
                    ]
                    for left, right in combinations(subset, 2)
                )
                candidates.append((float(total), tuple(subset)))
            if candidates:
                best_clique_statistic, best_clique = min(candidates)
                break
        rows.append(
            {
                "consensus_component_id": f"ca_boss_ks_consensus_{axis + 1:03d}",
                "component_assignment_json": json.dumps(
                    component_assignment, sort_keys=True, separators=(",", ":")
                ),
                "modalities": ";".join(modalities),
                "n_modalities": len(modalities),
                "matched_assay_pairs": axis_matches,
                "possible_assay_pairs": pairs_per_axis,
                "pairwise_match_support": float(axis_matches / pairs_per_axis),
                "maximum_pairwise_scaled_ks": max(axis_statistics),
                "mean_pairwise_scaled_ks": float(np.mean(axis_statistics)),
                "maximum_match_clique_modalities": ";".join(best_clique),
                "maximum_match_clique_size": len(best_clique),
                "maximum_match_clique_fraction": float(
                    len(best_clique) / len(modalities)
                ),
                "maximum_match_clique_total_scaled_ks": (
                    best_clique_statistic
                    if np.isfinite(best_clique_statistic)
                    else None
                ),
                "satisfies_published_all_pair_rule": axis_matches == pairs_per_axis,
                "published_algorithm_output": False,
                "exact_biological_axis_claimed": False,
            }
        )
    possible_total = n_components * pairs_per_axis
    if second_matches < 0:
        second_matches, second_statistic = 0, float("inf")
    diagnostics = {
        "signed_permutation_configurations_evaluated": evaluated,
        "best_matched_component_pair_count": best_matches,
        "possible_component_pair_count": possible_total,
        "best_global_pairwise_match_support": float(
            best_matches / possible_total
        ),
        "best_total_scaled_ks": best_statistic,
        "second_best_matched_component_pair_count": second_matches,
        "second_best_total_scaled_ks": (
            second_statistic if np.isfinite(second_statistic) else None
        ),
        "match_count_margin": int(best_matches - second_matches),
        "relative_statistic_margin_when_match_count_tied": (
            float((second_statistic - best_statistic) / max(best_statistic, 1e-12))
            if best_matches == second_matches and np.isfinite(second_statistic)
            else None
        ),
        "published_algorithm_output": False,
    }
    return pd.DataFrame(rows), diagnostics


def fit_unpaired_mdcr_backgrounds(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    *,
    config: UnpairedMDCRConfig | None = None,
) -> UnpairedMDCRResult:
    """Fit the truth-free fixed-rank adaptation of published Algorithm 3."""

    config = config or UnpairedMDCRConfig()
    modalities = sorted(str(value) for value in background_matrices)
    if len(modalities) < 2 or set(modalities) != set(background_depths):
        raise ValueError("At least two assays with matching depth vectors are required")
    scores: dict[str, FloatArray] = {}
    mixing: dict[str, FloatArray] = {}
    ranks: dict[str, int] = {}
    fit_diagnostics: dict[str, Any] = {}
    fold_scores: list[dict[str, FloatArray]] = [
        {} for _ in range(config.n_feature_folds)
    ]
    fold_rows: list[dict[str, Any]] = []
    for modality in modalities:
        matrix = _as_matrix(
            background_matrices[modality], max_entries=config.max_dense_entries
        )
        depth = _as_depth(background_depths[modality], matrix=matrix)
        normalized = _normalize_counts(matrix, depth)
        ranks[modality] = _empirical_rank(
            normalized, gamma=config.rank_gamma
        )
        assay_scores, assay_mixing, converged, iterations = _fit_ica(
            normalized, modality=modality, config=config
        )
        scores[modality] = assay_scores
        mixing[modality] = assay_mixing
        rng = np.random.default_rng(
            _stable_seed(config.random_seed, f"{modality}::feature_folds")
        )
        feature_folds = np.array_split(
            rng.permutation(normalized.shape[1]), config.n_feature_folds
        )
        for fold_index, indices in enumerate(feature_folds):
            local_scores, _, local_converged, local_iterations = _fit_ica(
                normalized[:, np.asarray(indices, dtype=int)],
                modality=f"{modality}__feature_fold{fold_index}",
                config=config,
            )
            aligned, correlations, margin = _align_fold_scores(
                local_scores, assay_scores
            )
            fold_scores[fold_index][modality] = aligned
            fold_rows.append(
                {
                    "modality": modality,
                    "feature_fold": fold_index,
                    "n_background_features": len(indices),
                    "minimum_full_fold_absolute_correlation": min(correlations),
                    "mean_full_fold_absolute_correlation": float(
                        np.mean(correlations)
                    ),
                    "component_assignment_relative_margin": margin,
                    "fastica_converged": local_converged,
                    "fastica_iterations": local_iterations,
                }
            )
        fit_diagnostics[modality] = {
            "cells": len(matrix),
            "background_features": matrix.shape[1],
            "empirical_covariance_rank_gamma": ranks[modality],
            "fastica_converged": converged,
            "fastica_iterations": iterations,
        }
    pairwise, adjusted_alpha, critical = _pairwise_component_table(
        scores, alpha=config.alpha
    )
    shared = _consistent_shared_tuples(scores, pairwise)
    consensus, consensus_diagnostics = _best_global_consensus(scores, pairwise)
    full_assignments = {
        str(row.consensus_component_id): str(row.component_assignment_json)
        for row in consensus.itertuples(index=False)
    }
    fold_consensus_frames: list[pd.DataFrame] = []
    fold_pairwise_frames: list[pd.DataFrame] = []
    fold_consensus_diagnostics: list[dict[str, Any]] = []
    for fold_index, current_scores in enumerate(fold_scores):
        fold_pairwise, _, _ = _pairwise_component_table(
            current_scores, alpha=config.alpha
        )
        fold_pairwise.insert(0, "feature_fold", fold_index)
        fold_pairwise_frames.append(fold_pairwise)
        current, diagnostics = _best_global_consensus(
            current_scores, fold_pairwise
        )
        current.insert(0, "feature_fold", fold_index)
        current["full_assignment_exact_match"] = current.apply(
            lambda row: str(row["component_assignment_json"])
            == full_assignments[str(row["consensus_component_id"])],
            axis=1,
        )
        fold_consensus_frames.append(current)
        fold_consensus_diagnostics.append(
            {"feature_fold": fold_index, **diagnostics}
        )
    fold_consensus = pd.concat(fold_consensus_frames, ignore_index=True)
    exact_rank_assumption = all(
        value == config.n_components for value in ranks.values()
    )
    provenance = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "paper_url": PAPER_URL,
        "config": asdict(config),
        "modalities": modalities,
        "fit_diagnostics": fit_diagnostics,
        "bonferroni_adjusted_alpha": adjusted_alpha,
        "asymptotic_ks_critical_value": critical,
        "n_pairwise_component_tests_including_sign": int(
            2 * len(pairwise)
        ),
        "n_shared_components": len(shared),
        "maximum_consensus_relaxation": consensus_diagnostics,
        "feature_fold_consensus": fold_consensus_diagnostics,
        "full_consensus_assignment_stability_rate": float(
            fold_consensus["full_assignment_exact_match"].astype(bool).mean()
        ),
        "information_boundary": {
            "held_out_background_counts_used": True,
            "focal_features_used": False,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "truth_or_scorer_used": False,
        },
        "assumption_audit": {
            "paper_noiseless_linear_injective_mixing_verified": False,
            "paper_independent_pairwise_distinct_nonsymmetric_errors_verified": False,
            "paper_observed_rank_equals_adapted_rank": exact_rank_assumption,
            "paper_two_partial_pure_children_per_shared_latent_verified": False,
            "poisson_measurement_noise_present_in_ca_boss_simulation": True,
            "fixed_rank_three_is_declared_adaptation": True,
            "theorem_backed_for_ca_boss_inputs": False,
        },
        "exact_cross_assay_feature_pair_claimed": False,
        "cross_assay_causal_direction_claimed": False,
    }
    return UnpairedMDCRResult(
        component_scores=scores,
        fold_component_scores=fold_scores,
        mixing_matrices=mixing,
        empirical_covariance_ranks=ranks,
        pairwise_matches=pairwise,
        shared_component_tuples=shared,
        consensus_component_tuples=consensus,
        fold_component_evidence=pd.DataFrame(fold_rows),
        fold_pairwise_matches=pd.concat(
            fold_pairwise_frames, ignore_index=True
        ),
        fold_consensus_component_tuples=fold_consensus,
        provenance=provenance,
    )


def _safe_discovery_dir(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if _FORBIDDEN_PATH_PARTS & {part.casefold() for part in resolved.parts}:
        raise ValueError(f"Background loader refuses scorer/truth path: {resolved}")
    return resolved


def freeze_unpaired_mdcr_evidence(
    discovery_dir: Path,
    out_dir: Path,
    *,
    config: UnpairedMDCRConfig | None = None,
) -> dict[str, Any]:
    """Read background archives only and content-address the diagnostic."""

    config = config or UnpairedMDCRConfig()
    discovery = _safe_discovery_dir(discovery_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    archives = sorted(discovery.glob(f"{_BACKGROUND_PREFIX}*.npz"))
    if len(archives) < 2:
        raise ValueError("At least two background archives are required")
    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    cell_ids: dict[str, NDArray[Any]] = {}
    input_hashes: dict[str, str] = {}
    for path in archives:
        modality = path.stem.removeprefix(_BACKGROUND_PREFIX)
        with np.load(path, allow_pickle=False) as archive:
            required = {"matrix", "background_depth", "cell_ids"}
            if missing := sorted(required.difference(archive.files)):
                raise ValueError(f"Background archive {path} is missing {missing}")
            matrices[modality] = np.asarray(archive["matrix"], dtype=float)
            depths[modality] = np.asarray(archive["background_depth"], dtype=float)
            cell_ids[modality] = np.asarray(archive["cell_ids"])
        input_hashes[path.name] = _sha256(path)
    result = fit_unpaired_mdcr_backgrounds(
        matrices, depths, config=config
    )
    out_dir.mkdir(parents=True)

    score_rows: list[dict[str, Any]] = []
    fold_score_rows: list[dict[str, Any]] = []
    mixing_rows: list[dict[str, Any]] = []
    for modality in sorted(result.component_scores):
        for row_index, cell_id in enumerate(cell_ids[modality]):
            score_rows.append(
                {
                    "modality": modality,
                    "cell_id": str(cell_id),
                    **{
                        f"component_{component}": float(
                            result.component_scores[modality][row_index, component]
                        )
                        for component in range(config.n_components)
                    },
                }
            )
        assay_mixing = result.mixing_matrices[modality]
        for feature_index in range(assay_mixing.shape[0]):
            mixing_rows.append(
                {
                    "modality": modality,
                    "background_feature_index": feature_index,
                    **{
                        f"component_{component}_loading": float(
                            assay_mixing[feature_index, component]
                        )
                        for component in range(config.n_components)
                    },
                }
            )
        for fold_index, fold_scores in enumerate(result.fold_component_scores):
            assay_fold_scores = fold_scores[modality]
            for row_index, cell_id in enumerate(cell_ids[modality]):
                fold_score_rows.append(
                    {
                        "modality": modality,
                        "feature_fold": fold_index,
                        "cell_id": str(cell_id),
                        **{
                            f"component_{component}": float(
                                assay_fold_scores[row_index, component]
                            )
                            for component in range(config.n_components)
                        },
                    }
                )
    outputs = {
        "component_scores.tsv": pd.DataFrame(score_rows),
        "fold_component_scores.tsv": pd.DataFrame(fold_score_rows),
        "mixing_matrices.tsv": pd.DataFrame(mixing_rows),
        "pairwise_component_matches.tsv": result.pairwise_matches,
        "shared_component_tuples.tsv": result.shared_component_tuples,
        "consensus_component_tuples.tsv": result.consensus_component_tuples,
        "fold_component_evidence.tsv": result.fold_component_evidence,
        "fold_pairwise_component_matches.tsv": result.fold_pairwise_matches,
        "fold_consensus_component_tuples.tsv": (
            result.fold_consensus_component_tuples
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    (out_dir / "provenance.json").write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source = Path(__file__).resolve(strict=True)
    output_names = [*outputs, "provenance.json"]
    manifest = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "algorithm_source_sha256": _sha256(source),
        "config": asdict(config),
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            name: _sha256(out_dir / name) for name in output_names
        },
        "frozen_before_scorer_read": True,
        "truth_or_scorer_used": False,
        "new_worlds_or_datasets_created": False,
        "published_method_assumptions_satisfied": False,
        "fixed_rank_three_adaptation": True,
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    (out_dir / "frozen_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rank-gamma", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = freeze_unpaired_mdcr_evidence(
        args.discovery_dir,
        args.out_dir,
        config=UnpairedMDCRConfig(
            alpha=args.alpha,
            rank_gamma=args.rank_gamma,
            random_seed=args.random_seed,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
