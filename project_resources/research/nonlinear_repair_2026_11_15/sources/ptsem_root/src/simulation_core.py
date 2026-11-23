"""Experiment 1: d-sweep for node-wise finite-library PT-SEM.

The proposed methods use the same local score.  For every node ``v`` and
candidate parent set ``C``, a plug-in BIC is computed for each of the six
exogenous working families; the smallest BIC is the local
score.  LibraryDP uses exact order-graph dynamic programming, whereas
LibraryGreedy uses greedy DAG search.

No fixed-family PT-SEM ablations are part of this experiment.  External
baselines are run only through genuine adapters; a requested but unavailable
baseline makes the experiment fail closed rather than substituting a proxy or
silently dropping a row.

Requires: numpy, scipy, pandas, matplotlib.
Optional for PC: causal-learn (``pip install causal-learn``).
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import io
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaincc, gammaln, logsumexp, xlogy

# KDEpy 1.1.4 in the frozen PB-SCM author implementation still calls the
# NumPy 1.x conversion alias removed in NumPy 2.  This compatibility mapping is
# installed in the core because Windows spawn workers import d.py directly and
# do not necessarily pass through a setting-specific wrapper module.  It is
# the documented floating-array behavior of the removed helper and does not
# modify the author source or algorithm.
if not hasattr(np, "asfarray"):
    def _numpy_asfarray_compat(values, dtype=float):
        requested = np.dtype(dtype)
        if requested.kind not in "fc":
            requested = np.dtype(float)
        return np.asarray(values, dtype=requested)

    np.asfarray = _numpy_asfarray_compat  # type: ignore[attr-defined]


SUPPORTED_FAMILIES: Tuple[str, ...] = (
    "Poisson",
    "NB",
    "ZIP",
    "Geom",
    "Binomial",
    "Bernoulli",
)
NUMERICAL_CORE_VERSION = "ptsem_submission_corrected_v3"


@dataclass(frozen=True)
class FamilyMetadata:
    continuous_dimension: int
    support: str


# Single authoritative family metadata table.  The Binomial integer size is
# a discrete model index; only p contributes to the continuous BIC dimension.
MODEL_FAMILY_METADATA: Dict[str, FamilyMetadata] = {
    "Poisson": FamilyMetadata(1, "nonnegative integers"),
    "NB": FamilyMetadata(2, "nonnegative integers"),
    "ZIP": FamilyMetadata(2, "nonnegative integers"),
    "Geom": FamilyMetadata(1, "nonnegative integers"),
    "Binomial": FamilyMetadata(1, "integers 0,...,n with n in {2,...,50}"),
    "Bernoulli": FamilyMetadata(1, "integers 0,1"),
}


def family_dimension(family: str) -> int:
    try:
        return MODEL_FAMILY_METADATA[family].continuous_dimension
    except KeyError as exc:
        raise ValueError(f"Unsupported family: {family}") from exc


@dataclass(frozen=True)
class EstimatorSpec:
    candidate_families: Tuple[str, ...] = SUPPORTED_FAMILIES

    def __post_init__(self) -> None:
        if not self.candidate_families:
            raise ValueError("candidate_families cannot be empty")
        if len(set(self.candidate_families)) != len(self.candidate_families):
            raise ValueError("candidate_families must be unique")
        unknown = set(self.candidate_families) - set(MODEL_FAMILY_METADATA)
        if unknown:
            raise ValueError(f"Unsupported candidate families: {sorted(unknown)}")


DEFAULT_ESTIMATOR_SPEC = EstimatorSpec()
# Compatibility name used by plotting/config validation.  It is fixed by the
# method specification and is not configurable from simulation metadata.
FAMLIB: Tuple[str, ...] = DEFAULT_ESTIMATOR_SPEC.candidate_families


def _configured_family_assignment() -> str:
    """Select the true-family assignment protocol without breaking old runs.

    ``cycle`` preserves the original four-family experiment, while
    ``balanced_shuffle`` preserves the first six-family sensitivity run.
    The paper-final six-family suite uses ``iid_uniform``: every node draws
    independently from the finite library with probability ``1/len(FAMLIB)``.
    """
    explicit = os.environ.get("PTSEM_FAMILY_ASSIGNMENT", "").strip().lower()
    aliases = {
        "cycle": "cycle",
        "balanced": "balanced_shuffle",
        "balanced_shuffle": "balanced_shuffle",
        "iid": "iid_uniform",
        "iid_uniform": "iid_uniform",
        "uniform": "iid_uniform",
    }
    if explicit:
        if explicit not in aliases:
            raise ValueError(
                "PTSEM_FAMILY_ASSIGNMENT must be one of "
                "cycle, balanced_shuffle, or iid_uniform"
            )
        return aliases[explicit]
    legacy_balanced = os.environ.get(
        "PTSEM_BALANCED_FAMILY_SHUFFLE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    return "balanced_shuffle" if legacy_balanced else "cycle"


FAMILY_ASSIGNMENT_MODE = _configured_family_assignment()
# Retain this public flag for older runners and manifests.
BALANCED_FAMILY_SHUFFLE = FAMILY_ASSIGNMENT_MODE == "balanced_shuffle"
DGP_BINOMIAL_N_MIN = int(os.environ.get("PTSEM_DGP_BINOMIAL_N_MIN", "2"))
DGP_BINOMIAL_N_MAX = int(os.environ.get("PTSEM_DGP_BINOMIAL_N_MAX", "20"))
# Fixed estimator/model-space constants.  These are deliberately literal and
# independent of every simulation generating bound and environment variable.
ESTIMATOR_BINOMIAL_N_MIN = 2
ESTIMATOR_BINOMIAL_N_MAX = 50
DGP_BINOMIAL_P_MIN = float(os.environ.get("PTSEM_DGP_BINOMIAL_P_MIN", "0.15"))
DGP_BINOMIAL_P_MAX = float(os.environ.get("PTSEM_DGP_BINOMIAL_P_MAX", "0.85"))
DGP_BERNOULLI_P_MIN = float(os.environ.get("PTSEM_DGP_BERNOULLI_P_MIN", "0.15"))
DGP_BERNOULLI_P_MAX = float(os.environ.get("PTSEM_DGP_BERNOULLI_P_MAX", "0.85"))
# Numerical projection distance used only when a finite empirical moment pair
# lies outside a family's natural inversion region.  It is a fixed method
# implementation constant, never a DGP or likelihood-tuning parameter.
MOMENT_MAP_EPS = 1e-5
if DGP_BINOMIAL_N_MIN < 2 or DGP_BINOMIAL_N_MAX < DGP_BINOMIAL_N_MIN:
    raise ValueError("Binomial family requires 2 <= n_min <= n_max")
if not (0.0 < DGP_BINOMIAL_P_MIN < DGP_BINOMIAL_P_MAX < 1.0):
    raise ValueError("Invalid Binomial probability range")
if not (0.0 < DGP_BERNOULLI_P_MIN < DGP_BERNOULLI_P_MAX < 1.0):
    raise ValueError("Invalid Bernoulli probability range")
INTERNAL_METHODS: Tuple[str, ...] = ("LibraryDP", "LibraryGreedy", "OracleDP")
EXTERNAL_METHODS: Tuple[str, ...] = (
    "PC-RCIT",
    "PBSCM",
    "PBSCM_PGF",
    "PoissonDAG-ODS",
    "CPCM",
)
ALL_METHODS: Tuple[str, ...] = INTERNAL_METHODS + EXTERNAL_METHODS
METHOD_STREAM: Dict[str, int] = {
    # Explicit streams preserve all previously used method seeds when a
    # baseline implementation or public label changes.
    "LibraryDP": 1,
    "LibraryGreedy": 2,
    "OracleDP": 3,
    "PC-RCIT": 4,
    "PBSCM": 5,
    "PBSCM_PGF": 6,
    "PoissonDAG-ODS": 7,
    "CPCM": 8,
}
MAIN_PLOT_METHODS: Tuple[str, ...] = (
    "LibraryDP",
    "LibraryGreedy",
    "PC-RCIT",
    "PBSCM",
    "PBSCM_PGF",
    "PoissonDAG-ODS",
)

RAW_COLUMNS: Tuple[str, ...] = (
    "d",
    "N",
    "rep",
    "method",
    "alpha_low",
    "alpha_high",
    "max_parents",
    "avg_indegree",
    "dag_generation_mode",
    "realized_avg_indegree",
    "parallel_workers",
    "seed",
    "method_seed",
    "runtime_sec",
    "local_score_runtime_sec",
    "search_runtime_sec",
    "n_local_scores_evaluated",
    "score",
    "directed_precision",
    "directed_recall",
    "directed_f1",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "exact_dag",
    "family_accuracy",
    "alpha_rmse_if_exact",
    "alpha_rmse_on_correct_edges",
    "alpha_mape_on_correct_edges_pct",
    "alpha_rmse_on_true_edges",
    "n_correct_directed_edges",
    "n_true_edges",
    "n_est_directed_edges",
    "n_est_undirected_edges",
    "true_families",
    "true_exogenous_params",
    "true_alpha_matrix",
    "estimated_alpha_matrix",
    "selected_families",
    "true_edges",
    "estimated_directed_edges",
    "estimated_undirected_edges",
    "numerical_guard_counts",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "d",
    "method",
    "alpha_low",
    "alpha_high",
    "max_parents",
    "avg_indegree",
    "dag_generation_mode",
    "mean_realized_avg_indegree",
    "se_realized_avg_indegree",
    "parallel_workers",
    "completed_reps",
    "mean_directed_f1",
    "se_directed_f1",
    "mean_skeleton_f1",
    "se_skeleton_f1",
    "mean_exact_dag",
    "se_exact_dag",
    "mean_family_accuracy",
    "se_family_accuracy",
    "mean_alpha_rmse_if_exact",
    "se_alpha_rmse_if_exact",
    "valid_alpha_rmse_reps",
    "mean_alpha_rmse_on_correct_edges",
    "se_alpha_rmse_on_correct_edges",
    "valid_alpha_correct_edge_reps",
    "mean_alpha_mape_on_correct_edges_pct",
    "se_alpha_mape_on_correct_edges_pct",
    "mean_alpha_rmse_on_true_edges",
    "se_alpha_rmse_on_true_edges",
    "mean_n_correct_directed_edges",
    "mean_runtime_sec",
    "se_runtime_sec",
    "mean_local_score_runtime_sec",
    "se_local_score_runtime_sec",
    "mean_search_runtime_sec",
    "se_search_runtime_sec",
    "mean_n_local_scores_evaluated",
)

Edge = Tuple[int, int]  # (parent, child)


# -----------------------------------------------------------------------------
# Data-generating PT-SEM
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyParam:
    family: str
    params: Dict[str, float]


def make_exogenous_specs(d: int, rng: np.random.Generator) -> List[FamilyParam]:
    """Draw true node families using the configured reproducible protocol."""
    if FAMILY_ASSIGNMENT_MODE == "iid_uniform":
        # This is the paper-final mixed-family design: no graph is forced to
        # contain every family, and every node has the same marginal chance.
        assigned_families = [
            str(value) for value in rng.choice(FAMLIB, size=d, replace=True)
        ]
    elif FAMILY_ASSIGNMENT_MODE == "balanced_shuffle":
        quotient, remainder = divmod(d, len(FAMLIB))
        assigned_families = list(FAMLIB) * quotient
        if remainder:
            assigned_families.extend(
                str(x) for x in rng.permutation(FAMLIB)[:remainder]
            )
        assigned_families = [
            str(x) for x in rng.permutation(assigned_families)
        ]
    elif FAMILY_ASSIGNMENT_MODE == "cycle":
        # Preserve the exact RNG consumption and assignments of the original
        # four-family experiments unless the sensitivity runner opts in.
        family_cycle = [str(x) for x in rng.permutation(FAMLIB)]
        assigned_families = [family_cycle[i % len(family_cycle)] for i in range(d)]
    else:  # pragma: no cover - validated during module import
        raise RuntimeError(f"Unknown family assignment mode: {FAMILY_ASSIGNMENT_MODE}")
    specs: List[FamilyParam] = []
    for fam in assigned_families:
        if fam == "Poisson":
            specs.append(FamilyParam(fam, {"lam": float(rng.uniform(2.0, 10.0))}))
        elif fam == "NB":
            # NumPy accepts a positive real-valued size parameter.
            size = float(rng.uniform(2.0, 10.0))
            probability = float(rng.uniform(0.15, 0.85))
            specs.append(FamilyParam(fam, {"r": size, "p": probability}))
        elif fam == "ZIP":
            specs.append(
                FamilyParam(
                    fam,
                    {
                        "rho": float(rng.uniform(0.15, 0.85)),
                        "lam": float(rng.uniform(2.0, 10.0)),
                    },
                )
            )
        elif fam == "Geom":
            # A geometric probability cannot lie in [2, 10], so draw its mean
            # uniformly on that interval and store the equivalent probability.
            mean = float(rng.uniform(2.0, 10.0))
            specs.append(FamilyParam(fam, {"mean": mean, "p": 1.0 / (1.0 + mean)}))
        elif fam == "Binomial":
            specs.append(
                FamilyParam(
                    fam,
                    {
                        "n": int(
                            rng.integers(
                                DGP_BINOMIAL_N_MIN, DGP_BINOMIAL_N_MAX + 1
                            )
                        ),
                        "p": float(
                            rng.uniform(DGP_BINOMIAL_P_MIN, DGP_BINOMIAL_P_MAX)
                        ),
                    },
                )
            )
        elif fam == "Bernoulli":
            specs.append(
                FamilyParam(
                    fam,
                    {
                        "p": float(
                            rng.uniform(
                                DGP_BERNOULLI_P_MIN, DGP_BERNOULLI_P_MAX
                            )
                        )
                    },
                )
            )
        else:  # pragma: no cover - guarded by FAMLIB
            raise ValueError(f"Unknown family: {fam}")
    return specs


def sample_exogenous(spec: FamilyParam, n: int, rng: np.random.Generator) -> np.ndarray:
    fam, par = spec.family, spec.params
    if fam == "Poisson":
        return rng.poisson(par["lam"], size=n).astype(int)
    if fam == "NB":
        return rng.negative_binomial(par["r"], par["p"], size=n).astype(int)
    if fam == "ZIP":
        x = rng.poisson(par["lam"], size=n).astype(int)
        x[rng.random(n) < par["rho"]] = 0
        return x
    if fam == "Geom":
        # NumPy's geometric is supported on {1, 2, ...}; the model uses {0, 1, ...}.
        return (rng.geometric(par["p"], size=n) - 1).astype(int)
    if fam == "Binomial":
        return rng.binomial(int(par["n"]), float(par["p"]), size=n).astype(int)
    if fam == "Bernoulli":
        return rng.binomial(1, float(par["p"]), size=n).astype(int)
    raise ValueError(fam)


def generate_random_dag(
    d: int,
    rng: np.random.Generator,
    avg_indegree: float = 1.6,
    max_parents: int = 3,
    alpha_low: float = 0.15,
    alpha_high: float = 0.85,
    generation_mode: str = "poisson_indegree",
) -> np.ndarray:
    """Generate a sparse ordered DAG; ``A[child, parent]`` stores alpha."""
    A = np.zeros((d, d), dtype=float)
    if max_parents == 0:
        return A
    if generation_mode == "exact_edges":
        candidates = [
            (parent, child)
            for child in range(1, d)
            for parent in range(child)
        ]
        capacity = sum(min(max_parents, child) for child in range(1, d))
        target_edges = int(round(d * avg_indegree))
        if target_edges < 0 or target_edges > capacity:
            raise ValueError(
                "Requested exact-edge density is infeasible: "
                f"round(d * avg_indegree)={target_edges}, capacity={capacity} "
                f"for d={d}, max_parents={max_parents}"
            )
        # Drawing a full permutation and alpha pool makes density cells paired:
        # lower-density graphs are prefixes of the same candidate ordering,
        # shared edges keep the same coefficients, and the RNG state entering
        # exogenous-family generation is independent of the target density.
        order = rng.permutation(len(candidates))
        alpha_pool = rng.uniform(alpha_low, alpha_high, size=len(candidates))
        indegrees = np.zeros(d, dtype=int)
        selected = 0
        for position, candidate_index in enumerate(order):
            parent, child = candidates[int(candidate_index)]
            if indegrees[child] >= max_parents:
                continue
            A[child, parent] = float(alpha_pool[position])
            indegrees[child] += 1
            selected += 1
            if selected == target_edges:
                break
        if selected != target_edges:  # pragma: no cover - capacity check guards this
            raise RuntimeError("Exact-edge DAG generator did not reach its target")
        return A
    if generation_mode != "poisson_indegree":
        raise ValueError(f"Unknown DAG generation mode: {generation_mode}")
    for child in range(1, d):
        cap = min(max_parents, child)
        indegree = max(1, min(int(rng.poisson(avg_indegree)), cap))
        parents = rng.choice(child, size=indegree, replace=False)
        for parent in parents:
            A[child, int(parent)] = float(rng.uniform(alpha_low, alpha_high))
    return A


def simulate_ptsem(
    d: int,
    n: int,
    seed: int,
    avg_indegree: float = 1.6,
    max_parents: int = 3,
    alpha_low: float = 0.15,
    alpha_high: float = 0.85,
    dag_generation_mode: str = "poisson_indegree",
) -> Tuple[np.ndarray, np.ndarray, List[FamilyParam]]:
    """Simulate one mixed-family PT-SEM data set from exactly one RNG stream."""
    rng = np.random.default_rng(seed)
    A = generate_random_dag(
        d,
        rng,
        avg_indegree,
        max_parents,
        alpha_low,
        alpha_high,
        dag_generation_mode,
    )
    specs = make_exogenous_specs(d, rng)
    X = np.zeros((n, d), dtype=int)
    for child in range(d):
        conditional_mean = np.zeros(n, dtype=float)
        for parent in range(child):
            if A[child, parent] > 0.0:
                conditional_mean += A[child, parent] * X[:, parent]
        thinned_sum = rng.poisson(conditional_mean).astype(int)
        X[:, child] = thinned_sum + sample_exogenous(specs[child], n, rng)
    return X, A, specs


# -----------------------------------------------------------------------------
# Plug-in local likelihood and family BIC
# -----------------------------------------------------------------------------


def empirical_moments(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = (centered.T @ centered) / X.shape[0]
    return mean, covariance


def pois_logpmf(k: int, lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    if k == 0:
        return np.where(lam == 0.0, 0.0, -lam)
    return np.where(
        lam == 0.0,
        -np.inf,
        k * np.log(np.maximum(lam, 1e-300)) - lam - gammaln(k + 1),
    )


def _validated_family_params(
    family: str, params: Dict[str, float]
) -> Dict[str, float]:
    """Validate parameters without silently clipping to another model."""
    if family == "Poisson":
        lam = float(params["lam"])
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError("Poisson requires finite lambda >= 0")
        return {"lam": lam}
    if family == "Geom":
        p = float(params["p"])
        if not math.isfinite(p) or not 0.0 < p < 1.0:
            raise ValueError("Geom requires finite 0 < p < 1")
        return {"p": p}
    if family == "NB":
        size, p = float(params["r"]), float(params["p"])
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("NB requires finite r > 0")
        if not math.isfinite(p) or not 0.0 < p < 1.0:
            raise ValueError("NB requires finite 0 < p < 1")
        return {"r": size, "p": p}
    if family == "ZIP":
        lam, rho = float(params["lam"]), float(params["rho"])
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError("ZIP requires finite lambda >= 0")
        if not math.isfinite(rho) or not 0.0 <= rho < 1.0:
            raise ValueError("ZIP requires finite 0 <= rho < 1")
        return {"lam": lam, "rho": rho}
    if family in ("Binomial", "Bernoulli"):
        if family == "Bernoulli":
            trials = 1
        else:
            raw_trials = float(params["n"])
            if not math.isfinite(raw_trials) or raw_trials != int(raw_trials):
                raise ValueError("Binomial requires an integer-valued n")
            trials = int(raw_trials)
        if family == "Binomial" and not (
            ESTIMATOR_BINOMIAL_N_MIN <= trials <= ESTIMATOR_BINOMIAL_N_MAX
        ):
            raise ValueError(
                "Binomial requires integer n in "
                f"{{{ESTIMATOR_BINOMIAL_N_MIN},...,{ESTIMATOR_BINOMIAL_N_MAX}}}"
            )
        p = float(params["p"])
        if not math.isfinite(p) or not 0.0 < p < 1.0:
            raise ValueError(f"{family} requires finite 0 < p < 1")
        result: Dict[str, float] = {"p": p}
        if family == "Binomial":
            result["n"] = trials
        return result
    raise ValueError(family)


def eps_logpmf(family: str, k: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    k = np.asarray(k, dtype=int)
    params = _validated_family_params(family, params)
    if family == "Poisson":
        lam = float(params["lam"])
        return xlogy(k, lam) - lam - gammaln(k + 1)
    if family == "Geom":
        p = float(params["p"])
        return np.log(p) + k * np.log1p(-p)
    if family == "NB":
        size = float(params["r"])
        p = float(params["p"])
        return (
            gammaln(k + size)
            - gammaln(size)
            - gammaln(k + 1)
            + size * np.log(p)
            + k * np.log1p(-p)
        )
    if family == "ZIP":
        lam = float(params["lam"])
        rho = float(params["rho"])
        out = np.empty_like(k, dtype=float)
        zero = k == 0
        out[zero] = np.logaddexp(np.log(rho) if rho > 0.0 else -np.inf, np.log1p(-rho) - lam)
        kp = k[~zero]
        out[~zero] = (
            np.log1p(-rho) + xlogy(kp, lam) - lam - gammaln(kp + 1)
        )
        return out
    if family in ("Binomial", "Bernoulli"):
        trials = 1 if family == "Bernoulli" else int(params["n"])
        p = float(params["p"])
        valid = (k >= 0) & (k <= trials)
        out = np.full(k.shape, -np.inf, dtype=float)
        kv = k[valid]
        out[valid] = (
            gammaln(trials + 1.0)
            - gammaln(kv + 1.0)
            - gammaln(trials - kv + 1.0)
            + kv * math.log(p)
            + (trials - kv) * math.log1p(-p)
        )
        return out
    raise ValueError(family)


class InvalidMomentInversion(ValueError):
    """The empirical residual moments do not belong to a family domain."""


def binomial_moment_map(
    mean: float,
    variance: float,
) -> Tuple[Dict[str, float], bool]:
    """Total fixed H_Binomial:R^2 -> Theta_Binomial.

    In the natural inversion region, the rounded moment index is projected
    into the fixed estimator space {2,...,50} before p=m/n is evaluated.
    Only an otherwise inadmissible finite pair is deterministically projected,
    without truth, DGP bounds, observed support, or likelihood profiling.  The
    boolean records use of the extension.
    """
    mean = float(mean)
    variance = float(variance)
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise InvalidMomentInversion("Binomial: nonfinite residual moments")

    eps = MOMENT_MAP_EPS

    if mean > 0.0 and 0.0 <= variance < mean:
        denominator = mean - variance
        # Algebraically identical to m^2/(m-v), but avoids an avoidable
        # intermediate overflow when the final continuous n is representable.
        n_continuous = mean * (mean / denominator)
        if math.isfinite(n_continuous):
            trials_raw = int(np.rint(n_continuous))
            trials = min(
                max(trials_raw, ESTIMATOR_BINOMIAL_N_MIN),
                ESTIMATOR_BINOMIAL_N_MAX,
            )
            probability = mean / trials
            if 0.0 < probability < 1.0:
                return {"n": trials, "p": float(probability)}, False

    mean_eff = max(mean, eps)
    implied_probability = 1.0 - variance / mean_eff
    probability_boundary = min(
        max(implied_probability, eps),
        1.0 - eps,
    )
    ratio = mean_eff / probability_boundary
    if math.isfinite(ratio):
        trials = min(
            max(ESTIMATOR_BINOMIAL_N_MIN, int(math.ceil(ratio))),
            ESTIMATOR_BINOMIAL_N_MAX,
        )
    else:
        trials = ESTIMATOR_BINOMIAL_N_MAX
    probability = mean_eff / trials
    if probability >= 1.0:
        # The total-map extension stays in the declared finite model space
        # even when the empirical mean exceeds its attainable mean boundary.
        probability = math.nextafter(1.0, 0.0)
    if not (
        ESTIMATOR_BINOMIAL_N_MIN <= trials <= ESTIMATOR_BINOMIAL_N_MAX
        and math.isfinite(probability)
        and 0.0 < probability < 1.0
    ):
        raise FloatingPointError(
            "Binomial total-map projection failed numerically"
        )
    return {"n": trials, "p": float(probability)}, True


def _moment_map_with_extension(
    family: str,
    mean: float,
    variance: float,
) -> Tuple[Dict[str, float], bool]:
    """Return the fixed total map H_r(mean, variance).

    Natural moment inversions remain exact.  A deterministic projection into
    the same family parameter space is used only for finite off-domain pairs.
    """
    mean = float(mean)
    variance = float(variance)
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise InvalidMomentInversion(f"{family}: nonfinite residual moments")

    eps = MOMENT_MAP_EPS
    if family == "Poisson":
        if mean > 0.0:
            return {"lam": mean}, False
        return {"lam": eps}, True

    if family == "Geom":
        if mean > 0.0:
            return {"p": 1.0 / (1.0 + mean)}, False
        mean_eff = eps
        return {"p": 1.0 / (1.0 + mean_eff)}, True

    if family == "Bernoulli":
        if 0.0 < mean < 1.0:
            return {"p": mean}, False
        probability = min(max(mean, eps), 1.0 - eps)
        return {"p": float(probability)}, True

    if family == "NB":
        if mean > 0.0 and variance > mean:
            # Algebraically identical to m^2/(v-m), with no m^2
            # intermediate overflow when the resulting size is representable.
            size = mean * (mean / (variance - mean))
            probability = mean / variance
            if (
                math.isfinite(size)
                and size > 0.0
                and 0.0 < probability < 1.0
            ):
                return {"r": size, "p": probability}, False
        mean_eff = max(mean, eps)
        probability = 1.0 - eps
        size = mean_eff * probability / (1.0 - probability)
        if not math.isfinite(size):
            # The prescribed Poisson-boundary projection can exceed the
            # largest representable r for an extreme but finite mean.  Use
            # the largest finite size and recompute p from the same NB mean
            # identity.  This is a machine-range guard, not an estimator cap.
            size = float(np.finfo(float).max)
            probability = 1.0 / (1.0 + mean_eff / size)
        return {"r": float(size), "p": float(probability)}, True

    if family == "ZIP":
        if mean > 0.0 and variance >= mean:
            extra = variance - mean
            lam = mean + extra / mean
            if math.isfinite(lam) and lam > 0.0:
                rho = 1.0 - mean / lam
                if 0.0 <= rho < 1.0:
                    return {"lam": float(lam), "rho": float(rho)}, False
        mean_eff = max(mean, eps)
        return {"lam": float(mean_eff), "rho": 0.0}, True

    if family == "Binomial":
        return binomial_moment_map(mean, variance)

    raise ValueError(family)


def invert_moments(
    family: str,
    mean: float,
    variance: float,
) -> Tuple[Dict[str, float], int]:
    """Apply the manuscript's fixed family-specific total moment map."""
    params, _ = _moment_map_with_extension(family, mean, variance)
    return params, family_dimension(family)


def convolution_loglik(
    observations: np.ndarray,
    thinning_mean: np.ndarray,
    family: str,
    params: Dict[str, float],
) -> float:
    """Fast exact log likelihood for ``X = Pois(mu) + epsilon``.

    Poisson and ZIP reduce to Poisson or two-Poisson-mixture likelihoods.
    Geometric uses the Poisson-CDF closed form.  NB uses the complete finite
    Delaporte convolution accumulated in log space, with a stable rising-
    factorial representation at the large-r Poisson boundary.
    Binomial and Bernoulli use their finite-support exact convolutions,
    vectorized over observations.  This avoids the former O(max(X))
    convolution for nearly all observations and is essential when thinning
    coefficients exceed one.
    """
    raw_observations = np.asarray(observations)
    if np.any(raw_observations < 0) or np.any(raw_observations != np.floor(raw_observations)):
        raise ValueError("observations must be nonnegative integer counts")
    observations = raw_observations.astype(int, copy=False)
    thinning_mean = np.asarray(thinning_mean, dtype=float)
    if observations.shape != thinning_mean.shape:
        raise ValueError("observations and thinning_mean must have the same shape")
    if np.any(~np.isfinite(thinning_mean)) or np.any(thinning_mean < 0.0):
        raise ValueError("thinning_mean must be finite and nonnegative")
    params = _validated_family_params(family, params)
    log_factorial = gammaln(observations + 1.0)

    def poisson_logpmf_vector(rate: np.ndarray) -> np.ndarray:
        rate = np.asarray(rate, dtype=float)
        return xlogy(observations, rate) - rate - log_factorial

    if family == "Poisson":
        rate = thinning_mean + float(params["lam"])
        return float(poisson_logpmf_vector(rate).sum())

    if family == "ZIP":
        lam = float(params["lam"])
        rho = float(params["rho"])
        structural_component = poisson_logpmf_vector(thinning_mean)
        poisson_component = poisson_logpmf_vector(thinning_mean + lam)
        log_rho = math.log(rho) if rho > 0.0 else -np.inf
        values = np.logaddexp(
            log_rho + structural_component,
            math.log1p(-rho) + poisson_component,
        )
        return float(values.sum())

    if family == "Geom":
        p = float(params["p"])
        q = 1.0 - p
        scaled_mean = thinning_mean / q
        poisson_cdf = gammaincc(observations + 1.0, scaled_mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (
                math.log(p)
                + observations * math.log(q)
                - thinning_mean
                + scaled_mean
                + np.log(poisson_cdf)
            )
        invalid = ~np.isfinite(values)
        if invalid.any():
            for index in np.flatnonzero(invalid):
                values[index] = _single_convolution_logpmf(
                    int(observations[index]),
                    float(thinning_mean[index]),
                    family,
                    params,
                )
        return float(values.sum())

    if family == "NB":
        size = float(params["r"])
        p = float(params["p"])
        q = 1.0 - p
        maximum = int(observations.max()) if observations.size else 0
        log_rising = np.zeros(maximum + 1, dtype=float)
        if maximum:
            log_rising[1:] = np.cumsum(
                np.log(size + np.arange(maximum, dtype=float))
            )
        # k=0 term: Pois(0;mu) NB(y;r,p).  The cumulative rising
        # factorial avoids gammaln(y+r)-gammaln(r) cancellation when moment
        # inversion puts r at the 1e8 Poisson boundary.
        log_term = (
            -thinning_mean
            + log_rising[observations]
            - log_factorial
            + size * math.log1p(-q)
            + observations * math.log(q)
        )
        log_total = log_term.copy()
        log_mean = xlogy(np.ones_like(thinning_mean), thinning_mean)
        for poisson_count in range(maximum):
            active = observations > poisson_count
            if not active.any():
                break
            remaining = observations[active] - poisson_count
            log_term[active] += (
                log_mean[active]
                - math.log(poisson_count + 1.0)
                + np.log(remaining)
                - np.log(remaining + size - 1.0)
                - math.log(q)
            )
            log_total[active] = np.logaddexp(
                log_total[active], log_term[active]
            )
        if np.any(~np.isfinite(log_total)):
            raise FloatingPointError(
                "nonfinite exact NB-Poisson convolution log probability"
            )
        return float(log_total.sum())

    if family in ("Binomial", "Bernoulli"):
        trials = 1 if family == "Bernoulli" else int(params["n"])
        probability = float(params["p"])
        maximum_epsilon = min(
            trials,
            int(observations.max()) if observations.size else 0,
        )
        log_p = math.log(probability)
        log_q = math.log1p(-probability)
        log_binomial = trials * log_q
        values = np.full(observations.shape, -np.inf, dtype=float)
        for epsilon_count in range(maximum_epsilon + 1):
            poisson_count = observations - epsilon_count
            valid = poisson_count >= 0
            if valid.any():
                local_count = poisson_count[valid]
                local_mean = thinning_mean[valid]
                log_poisson = (
                    xlogy(local_count, local_mean)
                    - local_mean
                    - gammaln(local_count + 1.0)
                )
                values[valid] = np.logaddexp(
                    values[valid], log_binomial + log_poisson
                )
            if epsilon_count < maximum_epsilon:
                log_binomial += (
                    math.log(trials - epsilon_count)
                    - math.log(epsilon_count + 1.0)
                    + log_p
                    - log_q
                )
        return float(values.sum())

    raise ValueError(family)


def _single_convolution_logpmf(
    observed: int,
    thinning_mean: float,
    family: str,
    params: Dict[str, float],
) -> float:
    """Stable exact fallback for one observation."""
    thin_counts = np.arange(observed + 1, dtype=int)
    local_mean = np.full(observed + 1, thinning_mean, dtype=float)
    log_poisson = (
        xlogy(thin_counts, local_mean)
        - local_mean
        - gammaln(thin_counts + 1.0)
    )
    log_epsilon = eps_logpmf(family, observed - thin_counts, params)
    return float(logsumexp(log_poisson + log_epsilon))


@dataclass
class LocalFit:
    bic: float
    family: Optional[str]
    params: Optional[Dict[str, float]]
    alpha: np.ndarray


def estimate_alpha_and_residual_moments(
    X: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
    audit: Optional[Counter[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    parents = [j for j in range(X.shape[1]) if (parent_mask >> j) & 1]
    if not parents:
        return (
            np.array([], dtype=float),
            np.zeros(X.shape[0], dtype=float),
            float(mean[child]),
            float(covariance[child, child]),
        )

    parent_covariance = covariance[np.ix_(parents, parents)]
    cross_covariance = covariance[parents, child]
    condition_number = float(np.linalg.cond(parent_covariance))
    condition_threshold = 1.0 / math.sqrt(np.finfo(float).eps)
    try:
        if not math.isfinite(condition_number) or condition_number > condition_threshold:
            raise np.linalg.LinAlgError("ill-conditioned covariance")
        alpha = np.linalg.solve(parent_covariance, cross_covariance)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(
            parent_covariance, cross_covariance, rcond=None
        )[0]
        if audit is not None:
            audit["alpha_singular_or_ill_conditioned_lstsq_fallback"] += 1
    negative = int(np.count_nonzero(alpha < 0.0))
    if negative and audit is not None:
        audit["alpha_nonnegative_projection_components"] += negative
    alpha = np.maximum(alpha, 0.0)

    thinning_mean = X[:, parents] @ alpha
    residual_mean = float(mean[child] - alpha @ mean[parents])
    residual_variance = float(
        covariance[child, child]
        - alpha @ mean[parents]
        - alpha @ parent_covariance @ alpha
    )
    return (
        alpha,
        thinning_mean,
        residual_mean,
        residual_variance,
    )


def local_score(
    X: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    child: int,
    parent_mask: int,
    families: Sequence[str],
    audit: Optional[Counter[str]] = None,
) -> LocalFit:
    n, d = X.shape
    parents = [j for j in range(d) if (parent_mask >> j) & 1]
    alpha, thinning_mean, residual_mean, residual_variance = estimate_alpha_and_residual_moments(
        X, mean, covariance, child, parent_mask, audit
    )
    best = LocalFit(float("inf"), None, None, alpha)
    for family in families:
        try:
            params, used_extension = _moment_map_with_extension(
                family, residual_mean, residual_variance
            )
        except InvalidMomentInversion:
            if audit is not None:
                audit[f"nonfinite_moment:{family}"] += 1
            continue
        if used_extension and audit is not None:
            audit[f"moment_map_extension:{family}"] += 1
        dimension = family_dimension(family)
        loglik = convolution_loglik(
            X[:, child], thinning_mean, family, params
        )
        if math.isnan(loglik) or loglik == math.inf:
            if audit is not None:
                audit[f"nonfinite_loglik:{family}"] += 1
            raise FloatingPointError(
                f"invalid {family} log likelihood for child={child}, "
                f"parent_mask={parent_mask}: {loglik}"
            )
        if loglik == -math.inf:
            if audit is not None:
                audit[f"impossible_support:{family}"] += 1
            continue
        parameter_count = len(parents) + dimension
        bic = -2.0 * loglik + parameter_count * math.log(n)
        if bic < best.bic:
            best = LocalFit(float(bic), family, params, alpha)
    return best


class LazyLibraryScoreCache:
    """Compute library-BIC local scores only when hill climbing requests them."""

    def __init__(
        self,
        X: np.ndarray,
        estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
        audit: Optional[Counter[str]] = None,
    ) -> None:
        start = time.perf_counter()
        self.X = X
        self.estimator_spec = estimator_spec
        self.audit = audit if audit is not None else Counter()
        self.mean, self.covariance = empirical_moments(X)
        self.fits: List[Dict[int, LocalFit]] = [dict() for _ in range(X.shape[1])]
        self.scoring_runtime = time.perf_counter() - start
        self.evaluation_count = 0

    def get_fit(self, child: int, parent_mask: int) -> LocalFit:
        if (parent_mask >> child) & 1:
            raise ValueError("A node cannot be its own parent")
        cached = self.fits[child].get(parent_mask)
        if cached is not None:
            return cached
        start = time.perf_counter()
        fit = local_score(
            self.X,
            self.mean,
            self.covariance,
            child,
            parent_mask,
            self.estimator_spec.candidate_families,
            self.audit,
        )
        self.scoring_runtime += time.perf_counter() - start
        self.evaluation_count += 1
        self.fits[child][parent_mask] = fit
        return fit


def precompute_local_scores(
    X: np.ndarray,
    family_options: Optional[Sequence[Sequence[str]]] = None,
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
    audit: Optional[Counter[str]] = None,
) -> Tuple[List[List[float]], List[List[LocalFit]]]:
    """Compute every node/parent-set score with no indegree restriction."""
    _, d = X.shape
    if family_options is None:
        family_options = [estimator_spec.candidate_families] * d
    if len(family_options) != d:
        raise ValueError("family_options must provide one family set per node")
    mask_count = 1 << d
    mean, covariance = empirical_moments(X)
    scores = [[float("inf")] * mask_count for _ in range(d)]
    fits: List[List[LocalFit]] = [
        [LocalFit(float("inf"), None, None, np.array([], dtype=float)) for _ in range(mask_count)]
        for _ in range(d)
    ]

    for child in range(d):
        families = tuple(family_options[child])
        if not families or set(families) - set(MODEL_FAMILY_METADATA):
            raise ValueError(f"Invalid family options for node {child}: {families}")

        for parent_mask in range(mask_count):
            if (parent_mask >> child) & 1:
                continue
            fit = local_score(
                X, mean, covariance, child, parent_mask, families, audit
            )
            scores[child][parent_mask] = fit.bic
            fits[child][parent_mask] = fit
    return scores, fits


# -----------------------------------------------------------------------------
# Exact order-graph DP and greedy DAG search
# -----------------------------------------------------------------------------


def exact_order_dp(scores: List[List[float]]) -> Tuple[Dict[int, int], float]:
    """Find the globally optimal DAG using subset-closed order-graph DP."""
    d = len(scores)
    mask_count = 1 << d
    closed_scores = [[float("inf")] * mask_count for _ in range(d)]
    best_parent_set = [[0] * mask_count for _ in range(d)]

    for child in range(d):
        for mask in range(mask_count):
            if not ((mask >> child) & 1):
                closed_scores[child][mask] = scores[child][mask]
                best_parent_set[child][mask] = mask
        for mask in sorted(
            (m for m in range(mask_count) if not ((m >> child) & 1)),
            key=int.bit_count,
        ):
            remaining = mask
            while remaining:
                bit = remaining & -remaining
                candidate = mask ^ bit
                if closed_scores[child][candidate] < closed_scores[child][mask]:
                    closed_scores[child][mask] = closed_scores[child][candidate]
                    best_parent_set[child][mask] = best_parent_set[child][candidate]
                remaining ^= bit

    best_order_score = [float("inf")] * mask_count
    last_node = [-1] * mask_count
    best_order_score[0] = 0.0
    for ordered_set in range(1, mask_count):
        remaining = ordered_set
        while remaining:
            bit = remaining & -remaining
            child = bit.bit_length() - 1
            predecessors = ordered_set ^ bit
            candidate_score = best_order_score[predecessors] + closed_scores[child][predecessors]
            if candidate_score < best_order_score[ordered_set]:
                best_order_score[ordered_set] = candidate_score
                last_node[ordered_set] = child
            remaining ^= bit

    parent_masks: Dict[int, int] = {}
    ordered_set = mask_count - 1
    while ordered_set:
        child = last_node[ordered_set]
        if child < 0:
            raise RuntimeError("Exact DP could not construct a finite-score DAG")
        predecessors = ordered_set ^ (1 << child)
        parent_masks[child] = best_parent_set[child][predecessors]
        ordered_set = predecessors
    return parent_masks, float(best_order_score[mask_count - 1])


def is_acyclic(parent_masks: Dict[int, int], d: int) -> bool:
    children = [[] for _ in range(d)]
    indegree = [0] * d
    for child, mask in parent_masks.items():
        for parent in range(d):
            if (mask >> parent) & 1:
                children[parent].append(child)
                indegree[child] += 1
    queue = [node for node in range(d) if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == d


def graph_score(parent_masks: Dict[int, int], scores: List[List[float]]) -> float:
    return float(sum(scores[child][parent_masks[child]] for child in range(len(scores))))


def greedy_hill_climb(
    scores: List[List[float]],
) -> Tuple[Dict[int, int], float]:
    """Deterministic best-improvement add/delete/reverse hill climbing."""
    d = len(scores)
    parent_masks = {node: 0 for node in range(d)}
    current_score = graph_score(parent_masks, scores)
    while True:
        best_masks: Optional[Dict[int, int]] = None
        best_score = current_score
        for parent in range(d):
            for child in range(d):
                if parent == child:
                    continue
                has_edge = bool((parent_masks[child] >> parent) & 1)
                if not has_edge:
                    candidate = dict(parent_masks)
                    candidate[child] |= 1 << parent
                    if is_acyclic(candidate, d):
                        candidate_score = graph_score(candidate, scores)
                        if candidate_score < best_score - 1e-8:
                            best_score, best_masks = candidate_score, candidate
                    continue

                candidate = dict(parent_masks)
                candidate[child] &= ~(1 << parent)
                candidate_score = graph_score(candidate, scores)
                if candidate_score < best_score - 1e-8:
                    best_score, best_masks = candidate_score, candidate

                candidate = dict(parent_masks)
                candidate[child] &= ~(1 << parent)
                candidate[parent] |= 1 << child
                if is_acyclic(candidate, d):
                    candidate_score = graph_score(candidate, scores)
                    if candidate_score < best_score - 1e-8:
                        best_score, best_masks = candidate_score, candidate

        if best_masks is None:
            return parent_masks, float(current_score)
        parent_masks, current_score = best_masks, best_score


def fit_library_greedy_lazy(
    X: np.ndarray,
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
    audit: Optional[Counter[str]] = None,
) -> Tuple[GraphEstimate, float, float, int]:
    """Best-improvement DAG hill climbing with on-demand local library scores.

    The returned graph is a one-edge local optimum under add/delete/reverse
    moves.  Local BIC values are cached by ``(child, parent_mask)``.
    """
    total_start = time.perf_counter()
    d = X.shape[1]
    cache = LazyLibraryScoreCache(X, estimator_spec, audit)
    parent_masks = {node: 0 for node in range(d)}
    current_score = float(sum(cache.get_fit(node, 0).bic for node in range(d)))

    while True:
        best_masks: Optional[Dict[int, int]] = None
        best_score = current_score
        for source in range(d):
            for target in range(d):
                if source == target:
                    continue
                has_edge = bool((parent_masks[target] >> source) & 1)
                if not has_edge:
                    candidate = dict(parent_masks)
                    candidate[target] |= 1 << source
                    if not is_acyclic(candidate, d):
                        continue
                    candidate_score = (
                        current_score
                        - cache.get_fit(target, parent_masks[target]).bic
                        + cache.get_fit(target, candidate[target]).bic
                    )
                    if candidate_score < best_score - 1e-8:
                        best_score, best_masks = candidate_score, candidate
                    continue

                # Delete source -> target.
                candidate = dict(parent_masks)
                candidate[target] &= ~(1 << source)
                candidate_score = (
                    current_score
                    - cache.get_fit(target, parent_masks[target]).bic
                    + cache.get_fit(target, candidate[target]).bic
                )
                if candidate_score < best_score - 1e-8:
                    best_score, best_masks = candidate_score, candidate

                # Reverse source -> target to target -> source.
                candidate = dict(parent_masks)
                candidate[target] &= ~(1 << source)
                candidate[source] |= 1 << target
                if is_acyclic(candidate, d):
                    candidate_score = (
                        current_score
                        - cache.get_fit(target, parent_masks[target]).bic
                        - cache.get_fit(source, parent_masks[source]).bic
                        + cache.get_fit(target, candidate[target]).bic
                        + cache.get_fit(source, candidate[source]).bic
                    )
                    if candidate_score < best_score - 1e-8:
                        best_score, best_masks = candidate_score, candidate

        if best_masks is None:
            break
        parent_masks, current_score = best_masks, best_score

    alpha_matrix = np.zeros((d, d), dtype=float)
    selected_families: List[Optional[str]] = [None] * d
    for child in range(d):
        fit = cache.get_fit(child, parent_masks[child])
        selected_families[child] = fit.family
        parents = [parent for parent in range(d) if (parent_masks[child] >> parent) & 1]
        for position, parent in enumerate(parents):
            alpha_matrix[child, parent] = fit.alpha[position]

    total_runtime = time.perf_counter() - total_start
    scoring_runtime = cache.scoring_runtime
    search_runtime = max(0.0, total_runtime - scoring_runtime)
    estimate = GraphEstimate(
        directed_edges=set(edges_from_parent_masks(parent_masks, d)),
        undirected_edges=set(),
        alpha_matrix=alpha_matrix,
        alpha_is_ptsem=True,
        selected_families=selected_families,
        score=float(current_score),
    )
    return estimate, scoring_runtime, search_runtime, cache.evaluation_count


@dataclass
class GraphEstimate:
    directed_edges: Set[Edge]
    undirected_edges: Set[Tuple[int, int]]
    alpha_matrix: Optional[np.ndarray] = None
    alpha_is_ptsem: bool = False
    selected_families: Optional[List[Optional[str]]] = None
    score: float = float("nan")
    diagnostics: Optional[Dict[str, object]] = None


def fit_internal_from_scores(
    d: int,
    scores: List[List[float]],
    fits: List[List[LocalFit]],
    search: str = "exact",
) -> GraphEstimate:
    """Run graph search from a precomputed score table and assemble estimates."""
    if search == "exact":
        parent_masks, best_score = exact_order_dp(scores)
    elif search == "greedy":
        parent_masks, best_score = greedy_hill_climb(scores)
    else:
        raise ValueError(f"Unknown graph search: {search}")

    alpha_matrix = np.zeros((d, d), dtype=float)
    selected_families: List[Optional[str]] = [None] * d
    for child in range(d):
        parent_mask = parent_masks[child]
        fit = fits[child][parent_mask]
        selected_families[child] = fit.family
        parents = [parent for parent in range(d) if (parent_mask >> parent) & 1]
        for position, parent in enumerate(parents):
            alpha_matrix[child, parent] = fit.alpha[position]
    return GraphEstimate(
        directed_edges=set(edges_from_parent_masks(parent_masks, d)),
        undirected_edges=set(),
        alpha_matrix=alpha_matrix,
        alpha_is_ptsem=True,
        selected_families=selected_families,
        score=best_score,
    )


@dataclass
class InternalFitDiagnostics:
    estimate: GraphEstimate
    local_score_runtime_sec: float
    search_runtime_sec: float
    n_local_scores_evaluated: int
    numerical_guard_counts: Dict[str, int]


def _fit_exact_with_family_options(
    X: np.ndarray,
    family_options: Sequence[Sequence[str]],
    estimator_spec: EstimatorSpec,
) -> InternalFitDiagnostics:
    audit: Counter[str] = Counter()
    score_start = time.perf_counter()
    scores, fits = precompute_local_scores(
        X,
        family_options=family_options,
        estimator_spec=estimator_spec,
        audit=audit,
    )
    score_runtime = time.perf_counter() - score_start
    search_start = time.perf_counter()
    estimate = fit_internal_from_scores(X.shape[1], scores, fits, "exact")
    search_runtime = time.perf_counter() - search_start
    return InternalFitDiagnostics(
        estimate=estimate,
        local_score_runtime_sec=score_runtime,
        search_runtime_sec=search_runtime,
        n_local_scores_evaluated=X.shape[1] * (1 << (X.shape[1] - 1)),
        numerical_guard_counts=dict(sorted(audit.items())),
    )


def _fit_proposed_with_diagnostics(
    X: np.ndarray,
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
) -> InternalFitDiagnostics:
    options = [estimator_spec.candidate_families] * X.shape[1]
    return _fit_exact_with_family_options(X, options, estimator_spec)


def fit_proposed(
    X: np.ndarray,
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
) -> GraphEstimate:
    """Fit Proposed DP-BIC from observed data and public estimator settings."""
    return _fit_proposed_with_diagnostics(X, estimator_spec).estimate


def _fit_greedy_with_diagnostics(
    X: np.ndarray,
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
) -> InternalFitDiagnostics:
    audit: Counter[str] = Counter()
    estimate, score_runtime, search_runtime, evaluations = (
        fit_library_greedy_lazy(X, estimator_spec, audit)
    )
    return InternalFitDiagnostics(
        estimate=estimate,
        local_score_runtime_sec=score_runtime,
        search_runtime_sec=search_runtime,
        n_local_scores_evaluated=evaluations,
        numerical_guard_counts=dict(sorted(audit.items())),
    )


def fit_greedy(
    X: np.ndarray,
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
) -> GraphEstimate:
    """Fit Greedy-BIC without access to any simulation truth."""
    return _fit_greedy_with_diagnostics(X, estimator_spec).estimate


def _fit_oracle_with_diagnostics(
    X: np.ndarray,
    true_families: Sequence[str],
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
) -> InternalFitDiagnostics:
    if len(true_families) != X.shape[1]:
        raise ValueError("Oracle requires exactly one true family per node")
    if set(true_families) - set(estimator_spec.candidate_families):
        raise ValueError("Oracle family is outside the public candidate library")
    options = [(family,) for family in true_families]
    return _fit_exact_with_family_options(X, options, estimator_spec)


def fit_oracle(
    X: np.ndarray,
    true_families: Sequence[str],
    estimator_spec: EstimatorSpec = DEFAULT_ESTIMATOR_SPEC,
) -> GraphEstimate:
    """Fit Oracle DP using only its explicitly allowed true-family vector."""
    return _fit_oracle_with_diagnostics(
        X, true_families, estimator_spec
    ).estimate


# -----------------------------------------------------------------------------
# Honest external-baseline adapters
# -----------------------------------------------------------------------------


class BaselineUnavailable(RuntimeError):
    """Raised when a requested external method has no usable real adapter."""


def _causal_graph_to_estimate(causal_graph) -> GraphEstimate:
    """Preserve directed/undirected semantics of a causal-learn CPDAG."""
    graph = causal_graph.G
    nodes = graph.get_nodes()
    directed: Set[Edge] = set()
    undirected: Set[Tuple[int, int]] = set()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not graph.is_adjacent_to(nodes[i], nodes[j]):
                continue
            i_to_j = bool(graph.is_directed_from_to(nodes[i], nodes[j]))
            j_to_i = bool(graph.is_directed_from_to(nodes[j], nodes[i]))
            if i_to_j and not j_to_i:
                directed.add((i, j))
            elif j_to_i and not i_to_j:
                directed.add((j, i))
            else:
                # Undirected, circle, or otherwise partially oriented adjacency:
                # skeleton evidence only, never two directed edges.
                undirected.add((i, j))
    return GraphEstimate(directed_edges=directed, undirected_edges=undirected)


def run_pc_rcit_baseline(
    X: np.ndarray, seed: Optional[int] = None
) -> GraphEstimate:
    """Run PC-Stable with scalable nonparametric RCIT on numeric counts.

    RCIT uses random Fourier features, so the replicate-specific method seed
    is applied explicitly.  The configuration is frozen for reproducibility.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise BaselineUnavailable(
            "PC-RCIT unavailable: install it with `pip install causal-learn`."
        ) from exc

    pc_kwargs: Dict[str, object] = {
        "alpha": 0.05,
        "indep_test": "rcit",
        "stable": True,
        "approx": "lpd4",
        "num_f": 100,
        "num_f2": 5,
        "rcit": True,
    }
    if "show_progress" in inspect.signature(pc).parameters:
        pc_kwargs["show_progress"] = False
    method_seed = 1 if seed is None else int(seed % (2**32))
    # causal-learn RCIT currently draws features from NumPy's global RNG.
    with temporary_global_seed(method_seed):
        causal_graph = pc(np.asarray(X, dtype=float), **pc_kwargs)
    return _causal_graph_to_estimate(causal_graph)


class _SilentProgress:
    """Minimal tqdm-compatible wrapper used to silence third-party progress bars."""

    def __init__(self, iterable: Optional[Iterable[object]] = None, **_: object) -> None:
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else ())

    def update(self, _: object = None) -> None:
        return None

    def close(self) -> None:
        return None


def _verify_pinned_external_source(name: str, source_directory: Path) -> None:
    """Fail closed unless an external author checkout is exactly pinned."""
    specification_path = (
        Path(__file__).resolve().parents[1] / "config/external_baselines.json"
    )
    try:
        entry = json.loads(specification_path.read_text(encoding="utf-8"))[name]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise BaselineUnavailable(
            f"{name} provenance specification is unavailable: {exc}"
        ) from exc
    if not (source_directory / ".git").is_dir():
        raise BaselineUnavailable(
            f"{name} unavailable: missing pinned git checkout at "
            f"{source_directory}. Run `python scripts/setup_external_baselines.py "
            "--fetch`."
        )
    completed = subprocess.run(
        ["git", "-C", str(source_directory), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode or head != entry["commit"]:
        raise BaselineUnavailable(
            f"{name} checkout is not at pinned commit {entry['commit']} "
            f"(found {head or 'unreadable'})"
        )
    missing = [
        relative
        for relative in entry["required_files"]
        if not (source_directory / relative).is_file()
    ]
    if missing:
        raise BaselineUnavailable(f"{name} checkout is missing files: {missing}")


def _load_pbscm_author_modules():
    """Load the pinned AAAI-2024 author implementation without copying it."""
    cached_name = "_ptsem_external_pbscm"
    if cached_name in sys.modules:
        return sys.modules[cached_name], sys.modules["_ptsem_external_pbscm_util"]

    source_directory = Path(__file__).resolve().parent / "external" / "PBSCM"
    _verify_pinned_external_source("PBSCM", source_directory)
    pb_path = source_directory / "PB_SCM.py"
    util_path = source_directory / "util.py"
    if not pb_path.exists() or not util_path.exists():
        raise BaselineUnavailable(
            "PBSCM unavailable: clone the author repository with "
            "`git clone https://github.com/DMIRLAB-Group/PBSCM.git external/PBSCM`."
        )
    try:
        util_spec = importlib.util.spec_from_file_location(
            "_ptsem_external_pbscm_util", util_path
        )
        if util_spec is None or util_spec.loader is None:
            raise ImportError("Could not create PB-SCM util module spec")
        util_module = importlib.util.module_from_spec(util_spec)
        sys.modules["_ptsem_external_pbscm_util"] = util_module
        # The author source uses `from util import *`.
        sys.modules["util"] = util_module
        util_spec.loader.exec_module(util_module)

        pb_spec = importlib.util.spec_from_file_location(cached_name, pb_path)
        if pb_spec is None or pb_spec.loader is None:
            raise ImportError("Could not create PB-SCM module spec")
        pb_module = importlib.util.module_from_spec(pb_spec)
        sys.modules[cached_name] = pb_module
        pb_spec.loader.exec_module(pb_module)
    except ImportError as exc:
        raise BaselineUnavailable(
            "PBSCM unavailable: its author implementation requires KDEpy, "
            "scikit-learn, scipy and tqdm. Run `python -m pip install "
            "KDEpy==1.1.4 scikit-learn scipy tqdm`."
        ) from exc

    pb_module.tqdm = _SilentProgress
    util_module.tqdm = _SilentProgress
    return pb_module, util_module


def run_pb_scm_baseline(
    X: np.ndarray, seed: Optional[int] = None
) -> GraphEstimate:
    """Run Qiao et al.'s genuine AAAI-2024 PB-SCM author code."""
    pb_module, util_module = _load_pbscm_author_modules()
    method_seed = 1 if seed is None else int(seed % (2**32))
    data_by_node = np.asarray(X, dtype=int).T
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        model = pb_module.PB_SCM(data_by_node, seed=method_seed)
        skeleton = model.Hill_Climb_search()
        adjacency = util_module.learning_causal_direction(data_by_node, skeleton)

    directed: Set[Edge] = set()
    undirected: Set[Tuple[int, int]] = set()
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[0]):
            i_to_j = bool(adjacency[i, j])
            j_to_i = bool(adjacency[j, i])
            if i_to_j and not j_to_i:
                directed.add((i, j))
            elif j_to_i and not i_to_j:
                directed.add((j, i))
            elif i_to_j or j_to_i:
                undirected.add((i, j))
    return GraphEstimate(directed_edges=directed, undirected_edges=undirected)


def _load_pbscm_pgf_author_modules():
    """Load the pinned NeurIPS-2024 PB-SCM-PGF author implementation."""
    module_name = "_ptsem_external_pbscm_pgf"
    util_name = "_ptsem_external_pbscm_pgf_util"
    cca_name = "_ptsem_external_pbscm_pgf_cca"
    if module_name in sys.modules:
        return sys.modules[module_name], sys.modules[util_name]

    source_directory = Path(__file__).resolve().parent / "external" / "PBSCM_PGF"
    _verify_pinned_external_source("PBSCM_PGF", source_directory)
    pb_path = source_directory / "PB_SCM_PGF.py"
    util_path = source_directory / "util.py"
    cca_path = source_directory / "CCARankTest.py"
    if not all(path.exists() for path in (pb_path, util_path, cca_path)):
        raise BaselineUnavailable(
            "PBSCM_PGF unavailable: run `python setup_external_baselines.py` "
            "to fetch the NeurIPS-2024 author repository."
        )

    previous_aliases = {
        name: sys.modules.get(name) for name in ("util", "CCARankTest", "torch")
    }
    try:
        cca_spec = importlib.util.spec_from_file_location(cca_name, cca_path)
        if cca_spec is None or cca_spec.loader is None:
            raise ImportError("Could not create PB-SCM-PGF CCA module spec")
        cca_module = importlib.util.module_from_spec(cca_spec)
        sys.modules[cca_name] = cca_module
        sys.modules["CCARankTest"] = cca_module
        cca_spec.loader.exec_module(cca_module)

        # The author util imports torch only for automatic differentiation.
        # Its own analytic NumPy derivative implements the same PGF derivative
        # and is substituted below, so a lightweight import stub avoids adding
        # a multi-gigabyte runtime dependency without modifying author files.
        if previous_aliases["torch"] is None:
            sys.modules["torch"] = types.ModuleType("torch")
        util_spec = importlib.util.spec_from_file_location(util_name, util_path)
        if util_spec is None or util_spec.loader is None:
            raise ImportError("Could not create PB-SCM-PGF util module spec")
        util_module = importlib.util.module_from_spec(util_spec)
        sys.modules[util_name] = util_module
        sys.modules["util"] = util_module
        util_spec.loader.exec_module(util_module)

        pb_spec = importlib.util.spec_from_file_location(module_name, pb_path)
        if pb_spec is None or pb_spec.loader is None:
            raise ImportError("Could not create PB-SCM-PGF module spec")
        pb_module = importlib.util.module_from_spec(pb_spec)
        sys.modules[module_name] = pb_module
        pb_spec.loader.exec_module(pb_module)
    except ImportError as exc:
        for name in (module_name, util_name, cca_name):
            sys.modules.pop(name, None)
        raise BaselineUnavailable(
            "PBSCM_PGF unavailable: install numpy, scipy, statsmodels and tqdm."
        ) from exc
    finally:
        for alias, previous in previous_aliases.items():
            if previous is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous

    pb_module.tqdm = _SilentProgress
    # These are mathematically identical author-provided derivatives.  The
    # analytic NumPy implementation is deterministic and avoids torch.
    author_numpy_derivative = util_module.diff_get_epgf_gradient_np

    def safe_numpy_derivative(data, z_list, order_list):
        # For d=2 the author's z list becomes [1, 1], which NumPy infers as
        # integer and then rejects negative derivative exponents.  Casting z
        # to float fixes this representation-only edge case.
        return author_numpy_derivative(
            data,
            np.asarray(z_list, dtype=float),
            order_list,
        )

    util_module.diff_get_epgf_gradient_np = safe_numpy_derivative
    pb_module.diff_get_epgf_gradient = safe_numpy_derivative

    def serial_parallel_subsample(data, batch, i, j):
        return np.asarray(
            [util_module.subsample_task(data, i, j) for _ in range(batch)],
            dtype=float,
        )

    # Avoid nested 12-process x 32-thread oversubscription while retaining the
    # author's exact bootstrap calculation and number of resamples.
    util_module.parallel_subsample = serial_parallel_subsample
    return pb_module, util_module


def run_pb_scm_pgf_baseline(
    X: np.ndarray, seed: Optional[int] = None
) -> GraphEstimate:
    """Run Xiang et al.'s genuine NeurIPS-2024 PB-SCM-PGF algorithm."""
    pb_module, _ = _load_pbscm_pgf_author_modules()
    data_by_node = np.asarray(X, dtype=int).T
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        model = pb_module.PBSCM_PGF(data_by_node)
        model.learn()
    adjacency = np.asarray(model.dag)

    directed: Set[Edge] = set()
    undirected: Set[Tuple[int, int]] = set()
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[0]):
            i_to_j = bool(adjacency[i, j])
            j_to_i = bool(adjacency[j, i])
            if i_to_j and not j_to_i:
                directed.add((i, j))
            elif j_to_i and not i_to_j:
                directed.add((j, i))
            elif i_to_j or j_to_i:
                undirected.add((i, j))
    return GraphEstimate(directed_edges=directed, undirected_edges=undirected)


def _poisson_overdispersion_score(
    response: np.ndarray,
    conditioning: np.ndarray,
    minimum_group_size: int,
) -> float:
    """Poisson ODS score from Park & Raskutti (2015), equation (4)."""
    response = np.asarray(response, dtype=float)
    conditioning = np.asarray(conditioning)
    if conditioning.size == 0:
        return float(response.var(ddof=1) - response.mean())
    if conditioning.ndim == 1:
        conditioning = conditioning[:, None]
    _, inverse, counts = np.unique(
        conditioning, axis=0, return_inverse=True, return_counts=True
    )
    eligible = np.flatnonzero(counts >= max(2, minimum_group_size))
    if eligible.size == 0:
        return float("inf")
    weighted_score = 0.0
    retained_count = int(counts[eligible].sum())
    for group in eligible:
        values = response[inverse == group]
        weighted_score += len(values) * (values.var(ddof=1) - values.mean())
    return float(weighted_score / retained_count)


def _poisson_glmlasso_support(
    response: np.ndarray,
    predictors: np.ndarray,
    predictor_nodes: Sequence[int],
    penalty: float = 0.1,
) -> Set[int]:
    """Select Poisson-GLM predictors using an L1 penalty.

    This mirrors glmnet's default predictor standardization, leaves the
    intercept unpenalized, and uses the NeurIPS-2015 simulation value
    lambda=0.1.  It is deliberately fixed rather than tuned against truth.
    """
    import statsmodels.api as sm

    response = np.asarray(response, dtype=float)
    predictors = np.asarray(predictors, dtype=float)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    if predictors.shape[1] != len(predictor_nodes):
        raise ValueError("predictor node list does not match design matrix")
    if predictors.shape[1] == 0 or float(response.var()) <= 1e-14:
        return set()

    means = predictors.mean(axis=0)
    scales = predictors.std(axis=0, ddof=0)
    varying = scales > 1e-12
    if not np.any(varying):
        return set()
    standardized = (predictors[:, varying] - means[varying]) / scales[varying]
    retained_nodes = np.asarray(predictor_nodes, dtype=int)[varying]
    design = sm.add_constant(standardized, prepend=True, has_constant="add")
    penalty_weights = np.concatenate(
        ([0.0], np.full(standardized.shape[1], float(penalty)))
    )
    start = np.zeros(design.shape[1], dtype=float)
    start[0] = math.log(max(float(response.mean()), 1e-8))
    model = sm.GLM(response, design, family=sm.families.Poisson())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        result = model.fit_regularized(
            method="elastic_net",
            alpha=penalty_weights,
            L1_wt=1.0,
            start_params=start,
            refit=False,
            maxiter=1000,
            cnvrg_tol=1e-8,
            zero_tol=1e-8,
        )
    coefficients = np.asarray(result.params[1:], dtype=float)
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Poisson GLMLasso produced non-finite coefficients")
    return {
        int(node)
        for node, coefficient in zip(retained_nodes, coefficients)
        if abs(float(coefficient)) > 1e-8
    }


def _poisson_glmlasso_moral_graph(
    X: np.ndarray, penalty: float = 0.1
) -> Tuple[Set[Tuple[int, int]], List[Set[int]]]:
    """ODS Step 1: node-wise GLMLasso and OR-rule symmetrization."""
    d = X.shape[1]
    directed_supports: List[Set[int]] = []
    for node in range(d):
        others = [candidate for candidate in range(d) if candidate != node]
        directed_supports.append(
            _poisson_glmlasso_support(
                X[:, node], X[:, others], others, penalty=penalty
            )
        )
    moral_edges: Set[Tuple[int, int]] = set()
    for first in range(d):
        for second in range(first + 1, d):
            if (
                second in directed_supports[first]
                or first in directed_supports[second]
            ):
                moral_edges.add((first, second))
    neighborhoods: List[Set[int]] = [set() for _ in range(d)]
    for first, second in moral_edges:
        neighborhoods[first].add(second)
        neighborhoods[second].add(first)
    return moral_edges, neighborhoods


def run_poisson_dag_ods_baseline(
    X: np.ndarray, seed: Optional[int] = None
) -> GraphEstimate:
    """Run the NeurIPS-2015 ODS algorithm with GLMLasso Steps 1 and 3.

    Step 1 learns a moral-graph neighborhood by node-wise L1-Poisson GLMs;
    Step 2 estimates an ordering using the paper's overdispersion score and
    c0=0.005; Step 3 performs a second L1-Poisson parent-selection regression.
    No PC skeleton or true-graph information is used.
    """
    del seed  # ODS-GLMLasso is deterministic for fixed data.
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    penalty = 0.1
    cutoff = max(2, int(math.ceil(0.005 * n)))

    # ODS Step 1: estimate the moral graph with GLMLasso.
    _, neighborhoods = _poisson_glmlasso_moral_graph(X, penalty=penalty)

    # ODS Step 2: recover a causal ordering by minimum overdispersion score.
    remaining: Set[int] = set(range(d))
    ordering: List[int] = []
    while remaining:
        if not ordering:
            candidates = sorted(remaining)
        else:
            candidates = sorted(neighborhoods[ordering[-1]] & remaining)
            if not candidates:  # Estimated moral graph may be disconnected.
                candidates = sorted(remaining)
        scored: List[Tuple[float, int]] = []
        for node in candidates:
            candidate_parents = sorted(neighborhoods[node] & set(ordering))
            conditioning = (
                X[:, candidate_parents]
                if candidate_parents
                else np.empty((n, 0), dtype=float)
            )
            score = _poisson_overdispersion_score(
                X[:, node], conditioning, cutoff
            )
            scored.append((score, node))
        _, selected_node = min(scored, key=lambda item: (item[0], item[1]))
        ordering.append(selected_node)
        remaining.remove(selected_node)

    # ODS Step 3: GLMLasso parent selection from earlier moral neighbors.
    directed: Set[Edge] = set()
    predecessors: Set[int] = set()
    for child in ordering:
        candidates = sorted(neighborhoods[child] & predecessors)
        if candidates:
            selected = _poisson_glmlasso_support(
                X[:, child],
                X[:, candidates],
                candidates,
                penalty=penalty,
            )
            directed.update((parent, child) for parent in selected)
        predecessors.add(child)
    return GraphEstimate(
        directed_edges=directed,
        undirected_edges=set(),
        diagnostics={
            "algorithm": "Park-Raskutti ODS Algorithm 1",
            "stages": (
                "GLMLasso moral graph",
                "conditional overdispersion ordering",
                "GLMLasso parent selection",
            ),
            "ordering": ordering,
            "moral_neighborhoods": [sorted(values) for values in neighborhoods],
            "c0": 0.005,
            "minimum_group_size": cutoff,
            "glmlasso_penalty": penalty,
        },
    )


def run_cpcm_baseline(X: np.ndarray) -> GraphEstimate:
    """Run Bodik & Chavez-Demoulin's genuine R CPCM implementation."""
    project_root = Path(__file__).resolve().parent
    rscript = _find_rscript()
    runner = project_root / "cpcm_runner.R"
    author_source = project_root / "external" / "CPCM" / "CPCM_function.R"
    r_library = project_root / "external" / "R_library"
    if rscript is None or not runner.exists() or not author_source.exists():
        raise BaselineUnavailable(
            "CPCM unavailable: Rscript, cpcm_runner.R, or the pinned author "
            "repository is missing. Run `python setup_external_baselines.py`."
        )
    with tempfile.TemporaryDirectory(prefix="ptsem_cpcm_") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "data.csv"
        output_path = temporary_path / "arcs.csv"
        # The author code constructs bnlearn node names X1, ..., Xd internally.
        columns = [f"X{node + 1}" for node in range(X.shape[1])]
        pd.DataFrame(np.asarray(X, dtype=int), columns=columns).to_csv(
            input_path, index=False
        )
        environment = os.environ.copy()
        environment["R_LIBS_USER"] = str(r_library)
        completed = subprocess.run(
            [
                str(rscript),
                "--vanilla",
                str(runner),
                str(input_path),
                str(output_path),
                str(author_source),
                str(r_library),
            ],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"CPCM R process failed: {detail}")
        arcs = pd.read_csv(output_path)
    directed: Set[Edge] = set()
    for row in arcs.itertuples(index=False):
        parent = int(str(row.from_node).removeprefix("X")) - 1
        child = int(str(row.to_node).removeprefix("X")) - 1
        directed.add((parent, child))
    return GraphEstimate(directed_edges=directed, undirected_edges=set())


def _find_rscript() -> Optional[Path]:
    located = shutil.which("Rscript")
    if located:
        return Path(located)
    candidates = sorted(
        Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True
    )
    return candidates[0] if candidates else None


BASELINE_ADAPTERS = {
    "PC-RCIT": run_pc_rcit_baseline,
    "PBSCM": run_pb_scm_baseline,
    "PBSCM_PGF": run_pb_scm_pgf_baseline,
    "PoissonDAG-ODS": run_poisson_dag_ods_baseline,
    "CPCM": run_cpcm_baseline,
}



# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def edges_from_A(A: np.ndarray, tolerance: float = 1e-12) -> List[Edge]:
    d = A.shape[0]
    return [
        (parent, child)
        for child in range(d)
        for parent in range(d)
        if parent != child and A[child, parent] > tolerance
    ]


def edges_from_parent_masks(parent_masks: Dict[int, int], d: int) -> List[Edge]:
    return [
        (parent, child)
        for child in range(d)
        for parent in range(d)
        if (parent_masks[child] >> parent) & 1
    ]


def precision_recall_f1(
    true_items: Set[object], estimated_items: Set[object]
) -> Tuple[float, float, float]:
    true_positive = len(true_items & estimated_items)
    false_positive = len(estimated_items - true_items)
    false_negative = len(true_items - estimated_items)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _f1(true_items: Set[object], estimated_items: Set[object]) -> float:
    return precision_recall_f1(true_items, estimated_items)[2]


def canonical_undirected(edge: Edge) -> Tuple[int, int]:
    return (min(edge), max(edge))


def alpha_rmse_on_true_edges(A: np.ndarray, estimated_A: np.ndarray, true_edges: Set[Edge]) -> float:
    if not true_edges:
        return float("nan")
    squared_errors = [(estimated_A[child, parent] - A[child, parent]) ** 2 for parent, child in true_edges]
    return float(np.sqrt(np.mean(squared_errors)))


def alpha_errors_on_correct_edges(
    A: np.ndarray,
    estimated_A: np.ndarray,
    correct_edges: Set[Edge],
) -> Tuple[float, float]:
    """Classical RMSE and MAPE (%) on correctly oriented recovered edges."""
    if not correct_edges:
        return float("nan"), float("nan")
    truth = np.asarray([A[child, parent] for parent, child in correct_edges])
    estimate = np.asarray(
        [estimated_A[child, parent] for parent, child in correct_edges]
    )
    rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    mape = float(100.0 * np.mean(np.abs((estimate - truth) / truth)))
    return rmse, mape


def evaluate_estimate(
    A: np.ndarray,
    true_families: Sequence[str],
    estimate: GraphEstimate,
    method: str,
) -> Dict[str, float]:
    true_directed = set(edges_from_A(A))
    estimated_directed = set(estimate.directed_edges)
    estimated_undirected = {canonical_undirected(edge) for edge in estimate.undirected_edges}
    true_skeleton = {canonical_undirected(edge) for edge in true_directed}
    estimated_skeleton = {canonical_undirected(edge) for edge in estimated_directed} | estimated_undirected
    correct_directed = true_directed & estimated_directed
    directed_precision, directed_recall, directed_f1 = precision_recall_f1(
        set(true_directed), set(estimated_directed)
    )
    skeleton_precision, skeleton_recall, skeleton_f1 = precision_recall_f1(
        set(true_skeleton), set(estimated_skeleton)
    )

    # A CPDAG is an exact DAG only when it is fully directed and equals the truth.
    exact = float(not estimated_undirected and estimated_directed == true_directed)
    if method in INTERNAL_METHODS and estimate.selected_families is not None:
        family_accuracy = float(
            np.mean(
                [
                    estimate.selected_families[node] == true_families[node]
                    for node in range(len(true_families))
                ]
            )
        )
    else:
        family_accuracy = float("nan")

    alpha_rmse = float("nan")
    alpha_rmse_correct = float("nan")
    alpha_mape_correct = float("nan")
    alpha_rmse_true_edges = float("nan")
    if estimate.alpha_matrix is not None and estimate.alpha_is_ptsem:
        alpha_rmse_correct, alpha_mape_correct = alpha_errors_on_correct_edges(
            A,
            estimate.alpha_matrix,
            correct_directed,
        )
        # Missing true edges are represented by zero in the estimated matrix.
        alpha_rmse_true_edges = alpha_rmse_on_true_edges(
            A,
            estimate.alpha_matrix,
            true_directed,
        )
    if exact == 1.0 and estimate.alpha_matrix is not None and estimate.alpha_is_ptsem:
        alpha_rmse = alpha_rmse_on_true_edges(A, estimate.alpha_matrix, true_directed)

    return {
        "directed_precision": directed_precision,
        "directed_recall": directed_recall,
        "directed_f1": directed_f1,
        "skeleton_precision": skeleton_precision,
        "skeleton_recall": skeleton_recall,
        "skeleton_f1": skeleton_f1,
        "exact_dag": exact,
        "family_accuracy": family_accuracy,
        "alpha_rmse_if_exact": alpha_rmse,
        "alpha_rmse_on_correct_edges": alpha_rmse_correct,
        "alpha_mape_on_correct_edges_pct": alpha_mape_correct,
        "alpha_rmse_on_true_edges": alpha_rmse_true_edges,
        "n_correct_directed_edges": float(len(correct_directed)),
        "n_true_edges": float(len(true_directed)),
        "n_est_directed_edges": float(len(estimated_directed)),
        "n_est_undirected_edges": float(len(estimated_undirected)),
    }


# -----------------------------------------------------------------------------
# Reproducibility, summaries, and figures
# -----------------------------------------------------------------------------


def derive_seed(base_seed: int, d: int, replicate: int, stream: int = 0) -> int:
    """Collision-resistant deterministic seed derived from (base, d, rep, stream)."""
    sequence = np.random.SeedSequence([base_seed, d, replicate, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


@contextmanager
def temporary_global_seed(seed: int) -> Iterable[None]:
    """Control optional baselines that use NumPy's or Python's global RNG."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(seed % (2**32))
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def setup_logger(outdir: Path, append: bool = False) -> logging.Logger:
    logger = logging.getLogger("ptsem_experiment1")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        outdir / "experiment1.log",
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    # Windows Defender, Explorer previews, and cloud/indexing services can
    # briefly hold the destination open.  Retrying the atomic rename keeps a
    # transient file lock from aborting a long, resumable experiment.
    for attempt in range(10):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (2 ** min(attempt, 6)))


def _mean_and_se(values: pd.Series) -> Tuple[float, float]:
    valid = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if valid.empty:
        return float("nan"), float("nan")
    mean = float(valid.mean())
    standard_error = float(valid.std(ddof=1) / math.sqrt(len(valid))) if len(valid) > 1 else float("nan")
    return mean, standard_error


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: List[Dict[str, object]] = []
    for (d, method), group in results.groupby(["d", "method"], sort=True):
        realized_indegree_mean, realized_indegree_se = _mean_and_se(
            group["realized_avg_indegree"]
        )
        directed_mean, directed_se = _mean_and_se(group["directed_f1"])
        skeleton_mean, skeleton_se = _mean_and_se(group["skeleton_f1"])
        exact_mean, exact_se = _mean_and_se(group["exact_dag"])
        family_mean, family_se = _mean_and_se(group["family_accuracy"])
        alpha_mean, alpha_se = _mean_and_se(group["alpha_rmse_if_exact"])
        alpha_correct_mean, alpha_correct_se = _mean_and_se(
            group["alpha_rmse_on_correct_edges"]
        )
        alpha_mape_mean, alpha_mape_se = _mean_and_se(
            group["alpha_mape_on_correct_edges_pct"]
        )
        alpha_true_mean, alpha_true_se = _mean_and_se(
            group["alpha_rmse_on_true_edges"]
        )
        correct_edge_mean, _ = _mean_and_se(group["n_correct_directed_edges"])
        runtime_mean, runtime_se = _mean_and_se(group["runtime_sec"])
        local_runtime_mean, local_runtime_se = _mean_and_se(
            group["local_score_runtime_sec"]
        )
        search_runtime_mean, search_runtime_se = _mean_and_se(
            group["search_runtime_sec"]
        )
        local_count_mean, _ = _mean_and_se(group["n_local_scores_evaluated"])
        rows.append(
            {
                "d": int(d),
                "method": str(method),
                "alpha_low": float(group["alpha_low"].iloc[0]),
                "alpha_high": float(group["alpha_high"].iloc[0]),
                "max_parents": int(group["max_parents"].iloc[0]),
                "avg_indegree": float(group["avg_indegree"].iloc[0]),
                "dag_generation_mode": str(
                    group["dag_generation_mode"].iloc[0]
                ),
                "mean_realized_avg_indegree": realized_indegree_mean,
                "se_realized_avg_indegree": realized_indegree_se,
                "parallel_workers": int(group["parallel_workers"].iloc[0]),
                "completed_reps": int(len(group)),
                "mean_directed_f1": directed_mean,
                "se_directed_f1": directed_se,
                "mean_skeleton_f1": skeleton_mean,
                "se_skeleton_f1": skeleton_se,
                "mean_exact_dag": exact_mean,
                "se_exact_dag": exact_se,
                "mean_family_accuracy": family_mean,
                "se_family_accuracy": family_se,
                "mean_alpha_rmse_if_exact": alpha_mean,
                "se_alpha_rmse_if_exact": alpha_se,
                "valid_alpha_rmse_reps": int(group["alpha_rmse_if_exact"].notna().sum()),
                "mean_alpha_rmse_on_correct_edges": alpha_correct_mean,
                "se_alpha_rmse_on_correct_edges": alpha_correct_se,
                "valid_alpha_correct_edge_reps": int(
                    group["alpha_rmse_on_correct_edges"].notna().sum()
                ),
                "mean_alpha_mape_on_correct_edges_pct": alpha_mape_mean,
                "se_alpha_mape_on_correct_edges_pct": alpha_mape_se,
                "mean_alpha_rmse_on_true_edges": alpha_true_mean,
                "se_alpha_rmse_on_true_edges": alpha_true_se,
                "mean_n_correct_directed_edges": correct_edge_mean,
                "mean_runtime_sec": runtime_mean,
                "se_runtime_sec": runtime_se,
                "mean_local_score_runtime_sec": local_runtime_mean,
                "se_local_score_runtime_sec": local_runtime_se,
                "mean_search_runtime_sec": search_runtime_mean,
                "se_search_runtime_sec": search_runtime_se,
                "mean_n_local_scores_evaluated": local_count_mean,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def make_family_confusion(results: pd.DataFrame) -> pd.DataFrame:
    """Long-form node-wise family confusion table for internal methods."""
    columns = (
        "d",
        "method",
        "true_family",
        "selected_family",
        "count",
        "row_rate",
    )
    counts: Dict[Tuple[int, str, str, str], int] = {}
    for row in results.itertuples(index=False):
        if row.method not in INTERNAL_METHODS or not row.selected_families:
            continue
        true_values = str(row.true_families).split(",")
        selected_values = str(row.selected_families).split(",")
        for truth, selected in zip(true_values, selected_values):
            key = (int(row.d), str(row.method), truth, selected)
            counts[key] = counts.get(key, 0) + 1
    output: List[Dict[str, object]] = []
    for d_value in sorted({key[0] for key in counts}):
        for method in INTERNAL_METHODS:
            for truth in FAMLIB:
                row_total = sum(
                    counts.get((d_value, method, truth, selected), 0)
                    for selected in FAMLIB
                )
                if row_total == 0:
                    continue
                for selected in FAMLIB:
                    count = counts.get((d_value, method, truth, selected), 0)
                    output.append(
                        {
                            "d": d_value,
                            "method": method,
                            "true_family": truth,
                            "selected_family": selected,
                            "count": count,
                            "row_rate": count / row_total,
                        }
                    )
    return pd.DataFrame(output, columns=columns)


def plot_metric(
    summary: pd.DataFrame,
    mean_column: str,
    se_column: str,
    ylabel: str,
    filename: str,
    outdir: Path,
    log_y: bool = False,
) -> None:
    # Imported lazily so spawned CPU workers do not each load Matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    plotted = False
    for method in MAIN_PLOT_METHODS:
        subset = summary.loc[summary["method"] == method].sort_values("d") if not summary.empty else summary
        subset = subset.loc[subset[mean_column].notna()] if not subset.empty else subset
        if subset.empty:
            continue
        means = subset[mean_column].to_numpy(dtype=float)
        errors = subset[se_column].fillna(0.0).to_numpy(dtype=float)
        if log_y:
            positive = means > 0.0
            subset = subset.loc[positive]
            means = means[positive]
            errors = errors[positive]
            # Symmetric error bars cannot extend to zero on a logarithmic axis.
            errors = np.minimum(errors, 0.95 * means)
            if subset.empty:
                continue
        axis.errorbar(
            subset["d"],
            means,
            yerr=errors,
            marker="o",
            linewidth=1.7,
            capsize=3,
            label=method,
        )
        plotted = True
    axis.set_xlabel("number of nodes d")
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    if not summary.empty:
        axis.set_xticks(sorted(summary["d"].unique()))
    axis.grid(alpha=0.2)
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No applicable results", ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    figure.savefig(outdir / filename, dpi=300)
    plt.close(figure)


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    plot_metric(
        summary,
        "mean_directed_f1",
        "se_directed_f1",
        "directed F1",
        "fig_directed_f1_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_skeleton_f1",
        "se_skeleton_f1",
        "skeleton F1",
        "fig_skeleton_f1_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_exact_dag",
        "se_exact_dag",
        "exact DAG recovery rate",
        "fig_exact_dag_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_family_accuracy",
        "se_family_accuracy",
        "node-wise family recovery accuracy",
        "fig_family_accuracy_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_alpha_rmse_if_exact",
        "se_alpha_rmse_if_exact",
        "alpha RMSE given exact DAG",
        "fig_alpha_rmse_if_exact_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_alpha_rmse_on_correct_edges",
        "se_alpha_rmse_on_correct_edges",
        "alpha RMSE on correctly directed edges",
        "fig_alpha_rmse_on_correct_edges_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_alpha_mape_on_correct_edges_pct",
        "se_alpha_mape_on_correct_edges_pct",
        "alpha MAPE on correctly directed edges (%)",
        "fig_alpha_mape_on_correct_edges_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_alpha_rmse_on_true_edges",
        "se_alpha_rmse_on_true_edges",
        "alpha RMSE on all true edges",
        "fig_alpha_rmse_on_true_edges_vs_d.png",
        outdir,
    )
    plot_metric(
        summary,
        "mean_runtime_sec",
        "se_runtime_sec",
        "total runtime (seconds, log scale)",
        "fig_runtime_vs_d.png",
        outdir,
        log_y=True,
    )
    plot_metric(
        summary,
        "mean_local_score_runtime_sec",
        "se_local_score_runtime_sec",
        "local-score runtime (seconds, log scale)",
        "fig_local_score_runtime_vs_d.png",
        outdir,
        log_y=True,
    )
    plot_metric(
        summary,
        "mean_search_runtime_sec",
        "se_search_runtime_sec",
        "graph-search runtime (seconds, log scale)",
        "fig_search_runtime_vs_d.png",
        outdir,
        log_y=True,
    )


def serialize_edges(edges: Iterable[Edge]) -> str:
    return ";".join(f"{parent}->{child}" for parent, child in sorted(edges))


def serialize_undirected_edges(edges: Iterable[Tuple[int, int]]) -> str:
    return ";".join(f"{min(a, b)}--{max(a, b)}" for a, b in sorted(edges))


@dataclass(frozen=True)
class ReplicateTask:
    d: int
    n: int
    replicate: int
    methods: Tuple[str, ...]
    max_parents: int
    avg_indegree: float
    dag_generation_mode: str
    alpha_low: float
    alpha_high: float
    base_seed: int
    blas_threads: int
    parallel_workers: int


@dataclass
class ReplicateOutput:
    rows: List[Dict[str, object]]
    warnings: List[str]


@contextmanager
def limited_blas_threads(thread_count: int) -> Iterable[None]:
    """Prevent BLAS oversubscription when many worker processes are active."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=thread_count):
        yield


def initialize_worker(methods: Sequence[str]) -> None:
    """Import optional baselines before their measured execution begins."""
    if "PC-RCIT" in methods:
        from causallearn.search.ConstraintBased.PC import pc as _pc

        del _pc
    if "PoissonDAG-ODS" in methods:
        import statsmodels.api as _sm

        del _sm
    if "PBSCM" in methods:
        _load_pbscm_author_modules()
    if "PBSCM_PGF" in methods:
        _load_pbscm_pgf_author_modules()


def run_replicate_task(task: ReplicateTask) -> ReplicateOutput:
    """Run all requested methods for one data set; safe for Windows spawn."""
    data_seed = derive_seed(task.base_seed, task.d, task.replicate, stream=0)
    warnings: List[str] = []
    rows: List[Dict[str, object]] = []

    with limited_blas_threads(task.blas_threads):
        X, A, specs = simulate_ptsem(
            task.d,
            task.n,
            data_seed,
            task.avg_indegree,
            task.max_parents,
            task.alpha_low,
            task.alpha_high,
            task.dag_generation_mode,
        )
        true_families = [spec.family for spec in specs]
        true_edges = set(edges_from_A(A))

        for method in task.methods:
            method_seed = derive_seed(
                task.base_seed,
                task.d,
                task.replicate,
                stream=METHOD_STREAM[method],
            )
            # Every method sees identical values but cannot mutate another method's input.
            method_data = X.copy()
            local_score_runtime = float("nan")
            search_runtime = float("nan")
            local_score_count = float("nan")
            numerical_guard_counts: Dict[str, int] = {}
            try:
                if method == "LibraryDP":
                    diagnostics = _fit_proposed_with_diagnostics(method_data)
                    estimate = diagnostics.estimate
                    local_score_runtime = diagnostics.local_score_runtime_sec
                    search_runtime = diagnostics.search_runtime_sec
                    local_score_count = float(
                        diagnostics.n_local_scores_evaluated
                    )
                    numerical_guard_counts = diagnostics.numerical_guard_counts
                    runtime = local_score_runtime + search_runtime
                elif method == "LibraryGreedy":
                    diagnostics = _fit_greedy_with_diagnostics(method_data)
                    estimate = diagnostics.estimate
                    local_score_runtime = diagnostics.local_score_runtime_sec
                    search_runtime = diagnostics.search_runtime_sec
                    local_score_count = float(
                        diagnostics.n_local_scores_evaluated
                    )
                    numerical_guard_counts = diagnostics.numerical_guard_counts
                    runtime = local_score_runtime + search_runtime
                elif method == "OracleDP":
                    diagnostics = _fit_oracle_with_diagnostics(
                        method_data, true_families
                    )
                    estimate = diagnostics.estimate
                    local_score_runtime = diagnostics.local_score_runtime_sec
                    search_runtime = diagnostics.search_runtime_sec
                    local_score_count = float(
                        diagnostics.n_local_scores_evaluated
                    )
                    numerical_guard_counts = diagnostics.numerical_guard_counts
                    runtime = local_score_runtime + search_runtime
                else:
                    start = time.perf_counter()
                    with temporary_global_seed(method_seed):
                        adapter = BASELINE_ADAPTERS[method]
                        if "seed" in inspect.signature(adapter).parameters:
                            estimate = adapter(method_data, seed=method_seed)
                        else:
                            estimate = adapter(method_data)
                    runtime = time.perf_counter() - start
            except BaselineUnavailable:
                raise
            except Exception:
                raise

            metrics = evaluate_estimate(A, true_families, estimate, method)
            selected = estimate.selected_families or []
            rows.append(
                {
                    "d": task.d,
                    "N": task.n,
                    "rep": task.replicate,
                    "method": method,
                    "alpha_low": task.alpha_low,
                    "alpha_high": task.alpha_high,
                    "max_parents": task.max_parents,
                    "avg_indegree": task.avg_indegree,
                    "dag_generation_mode": task.dag_generation_mode,
                    "realized_avg_indegree": len(true_edges) / task.d,
                    "parallel_workers": task.parallel_workers,
                    "seed": data_seed,
                    "method_seed": method_seed,
                    "runtime_sec": runtime,
                    "local_score_runtime_sec": local_score_runtime,
                    "search_runtime_sec": search_runtime,
                    "n_local_scores_evaluated": local_score_count,
                    "score": estimate.score,
                    "true_families": ",".join(true_families),
                    "true_exogenous_params": json.dumps(
                        [spec.params for spec in specs],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "true_alpha_matrix": json.dumps(
                        A.tolist(), separators=(",", ":")
                    ),
                    "estimated_alpha_matrix": json.dumps(
                        np.asarray(estimate.alpha_matrix, dtype=float).tolist(),
                        separators=(",", ":"),
                    ),
                    "selected_families": ",".join(
                        "" if value is None else value for value in selected
                    ),
                    "true_edges": serialize_edges(true_edges),
                    "estimated_directed_edges": serialize_edges(estimate.directed_edges),
                    "estimated_undirected_edges": serialize_undirected_edges(
                        estimate.undirected_edges
                    ),
                    "numerical_guard_counts": json.dumps(
                        numerical_guard_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    **metrics,
                }
            )
    return ReplicateOutput(rows=rows, warnings=warnings)


def available_methods(methods: Sequence[str], logger: logging.Logger) -> List[str]:
    """Remove unavailable external baselines before scheduling worker tasks."""
    enabled: List[str] = []
    for method in methods:
        if method == "PC-RCIT" and importlib.util.find_spec("causallearn") is None:
            logger.warning(
                "%s unavailable: install its PC-RCIT implementation with "
                "`pip install causal-learn`; skipping it.",
                method,
            )
            continue
        if method == "PoissonDAG-ODS" and importlib.util.find_spec(
            "statsmodels"
        ) is None:
            logger.warning(
                "%s unavailable: install its GLMLasso implementation with "
                "`pip install statsmodels`; skipping it.",
                method,
            )
            continue
        if method == "PBSCM":
            try:
                _load_pbscm_author_modules()
            except BaselineUnavailable as exc:
                logger.warning("%s Skipping PBSCM.", exc)
                continue
        if method == "PBSCM_PGF":
            try:
                _load_pbscm_pgf_author_modules()
            except BaselineUnavailable as exc:
                logger.warning("%s Skipping PBSCM_PGF.", exc)
                continue
        if method == "CPCM":
            project_root = Path(__file__).resolve().parent
            if (
                _find_rscript() is None
                or not (project_root / "cpcm_runner.R").exists()
                or not (project_root / "external" / "CPCM" / "CPCM_function.R").exists()
            ):
                logger.warning(
                    "CPCM unavailable: run `python setup_external_baselines.py`; "
                    "skipping CPCM without proxy results."
                )
                continue
        if method in EXTERNAL_METHODS and getattr(
            BASELINE_ADAPTERS[method], "_ptsem_placeholder", False
        ):
            logger.warning(
                "%s unavailable: no genuine adapter is connected; skipping it "
                "without emitting proxy results.",
                method,
            )
            continue
        enabled.append(method)
    return enabled


def ordered_results_frame(
    rows: Sequence[Dict[str, object]], methods: Sequence[str]
) -> pd.DataFrame:
    """Create deterministic CSV order independent of worker completion order."""
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if frame.empty:
        return frame
    method_order = {method: index for index, method in enumerate(methods)}
    frame["_method_order"] = frame["method"].map(method_order)
    frame = frame.sort_values(
        ["d", "rep", "_method_order"], kind="stable"
    ).drop(columns="_method_order")
    return frame.reset_index(drop=True)


def load_resume_checkpoint(
    raw_path: Path,
    d_values: Sequence[int],
    n: int,
    reps: int,
    methods: Sequence[str],
    max_parents: int,
    avg_indegree: float,
    dag_generation_mode: str,
    seed: int,
    alpha_low: float,
    alpha_high: float,
) -> Tuple[List[Dict[str, object]], Set[Tuple[int, int]], int]:
    """Load only complete replicate cells from a validated raw checkpoint."""
    if not raw_path.exists():
        return [], set(), 0
    checkpoint = pd.read_csv(raw_path)
    missing_columns = set(RAW_COLUMNS) - set(checkpoint.columns)
    if missing_columns:
        raise RuntimeError(
            "Resume checkpoint uses an incompatible schema; choose a new "
            f"--outdir. Missing columns: {sorted(missing_columns)}"
        )
    allowed_d = {int(value) for value in d_values}
    allowed_methods = set(methods)
    if not set(pd.to_numeric(checkpoint["d"]).astype(int)).issubset(allowed_d):
        raise RuntimeError("Resume checkpoint contains unexpected d values")
    if not set(checkpoint["method"].astype(str)).issubset(allowed_methods):
        raise RuntimeError("Resume checkpoint contains unexpected methods")
    if set(checkpoint["dag_generation_mode"].astype(str)) != {
        dag_generation_mode
    }:
        raise RuntimeError(
            "Resume checkpoint DAG generation mode does not match this run; "
            "choose a new --outdir"
        )

    numeric_expectations = {
        "N": float(n),
        "alpha_low": float(alpha_low),
        "alpha_high": float(alpha_high),
        "max_parents": float(max_parents),
        "avg_indegree": float(avg_indegree),
    }
    for column, expected in numeric_expectations.items():
        values = pd.to_numeric(checkpoint[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)) or not np.allclose(
            values,
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Resume checkpoint {column} does not match this run; "
                "choose a new --outdir"
            )

    valid_rep = pd.to_numeric(checkpoint["rep"], errors="coerce").astype(int)
    if ((valid_rep < 0) | (valid_rep >= reps)).any():
        raise RuntimeError("Resume checkpoint contains replicate indices outside --R")
    for row in checkpoint.itertuples(index=False):
        expected_seed = derive_seed(seed, int(row.d), int(row.rep), stream=0)
        if int(row.seed) != expected_seed:
            raise RuntimeError(
                "Resume checkpoint seed does not match this run; choose a new --outdir"
            )

    completed_keys: Set[Tuple[int, int]] = set()
    retained_frames: List[pd.DataFrame] = []
    for (d_value, replicate), group in checkpoint.groupby(["d", "rep"], sort=False):
        method_counts = group["method"].value_counts()
        complete = (
            set(method_counts.index.astype(str)) == allowed_methods
            and method_counts.eq(1).all()
        )
        if complete:
            completed_keys.add((int(d_value), int(replicate)))
            retained_frames.append(group)
    retained = (
        pd.concat(retained_frames, ignore_index=True)
        if retained_frames
        else pd.DataFrame(columns=RAW_COLUMNS)
    )
    return retained.to_dict("records"), completed_keys, len(checkpoint)


def run_experiment(
    d_values: Sequence[int],
    n: int,
    reps: int,
    methods: Sequence[str],
    max_parents: int,
    avg_indegree: float,
    seed: int,
    outdir: str,
    workers: int = 1,
    require_all_methods: bool = False,
    alpha_low: float = 0.15,
    alpha_high: float = 0.85,
    resume: bool = False,
    dag_generation_mode: str = "poisson_indegree",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_directory = Path(outdir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    provenance_source = Path(__file__).resolve().parent / "BASELINE_SOURCES.md"
    if provenance_source.exists():
        shutil.copyfile(
            provenance_source, output_directory / "BASELINE_SOURCES.md"
        )
    raw_path = output_directory / "experiment1_raw_results.csv"
    summary_path = output_directory / "experiment1_summary.csv"
    family_confusion_path = output_directory / "experiment1_family_confusion.csv"
    logger = setup_logger(
        output_directory,
        append=bool(resume and raw_path.exists()),
    )

    requested_methods = list(methods)
    methods = available_methods(requested_methods, logger)
    unavailable = [method for method in requested_methods if method not in methods]
    if require_all_methods and unavailable:
        raise RuntimeError(
            "Required methods are unavailable: " + ", ".join(unavailable)
        )
    total_tasks = len(d_values) * reps
    effective_workers = min(max(1, workers), max(1, total_tasks))
    blas_threads = 1 if effective_workers > 1 else 2
    logger.info(
        "Starting Experiment 1 with d=%s, N=%d, R=%d, max_parents=%d, "
        "alpha=[%.3f, %.3f], DAG_mode=%s, seed=%d, methods=%s, workers=%d, "
        "BLAS_threads_per_worker=%d",
        list(d_values),
        n,
        reps,
        max_parents,
        alpha_low,
        alpha_high,
        dag_generation_mode,
        seed,
        methods,
        effective_workers,
        blas_threads,
    )

    all_tasks = [
        ReplicateTask(
            d=d,
            n=n,
            replicate=replicate,
            methods=tuple(methods),
            max_parents=max_parents,
            avg_indegree=avg_indegree,
            dag_generation_mode=dag_generation_mode,
            alpha_low=alpha_low,
            alpha_high=alpha_high,
            base_seed=seed,
            blas_threads=blas_threads,
            parallel_workers=effective_workers,
        )
        for d in d_values
        for replicate in range(reps)
    ]
    rows: List[Dict[str, object]] = []
    completed_keys: Set[Tuple[int, int]] = set()
    if resume:
        rows, completed_keys, checkpoint_row_count = load_resume_checkpoint(
            raw_path,
            d_values,
            n,
            reps,
            methods,
            max_parents,
            avg_indegree,
            dag_generation_mode,
            seed,
            alpha_low,
            alpha_high,
        )
        if checkpoint_row_count:
            logger.info(
                "Resume checkpoint: retained %d complete replicate tasks "
                "(%d rows); %d incomplete checkpoint rows will be recomputed.",
                len(completed_keys),
                len(rows),
                checkpoint_row_count - len(rows),
            )
    tasks = [
        task
        for task in all_tasks
        if (task.d, task.replicate) not in completed_keys
    ]
    if resume:
        logger.info(
            "Resume scheduling %d/%d replicate tasks; %d already complete.",
            len(tasks),
            total_tasks,
            len(completed_keys),
        )
    logged_warnings: Set[str] = set()
    completed_tasks = len(completed_keys)

    def record_output(task: ReplicateTask, output: ReplicateOutput) -> None:
        nonlocal completed_tasks
        completed_tasks += 1
        rows.extend(output.rows)
        for warning_message in output.warnings:
            if warning_message not in logged_warnings:
                logger.warning(warning_message)
                logged_warnings.add(warning_message)
        for row in output.rows:
            family_value = float(row["family_accuracy"])
            family_text = f"{family_value:.3f}" if not math.isnan(family_value) else "NA"
            logger.info(
                "[%d/%d] d=%d rep=%03d %-13s directed_F1=%.3f "
                "skeleton_F1=%.3f exact=%.0f family=%s time=%.3fs",
                completed_tasks,
                total_tasks,
                task.d,
                task.replicate,
                row["method"],
                row["directed_f1"],
                row["skeleton_f1"],
                row["exact_dag"],
                family_text,
                row["runtime_sec"],
            )
        _atomic_to_csv(ordered_results_frame(rows, methods), raw_path)

    if not methods:
        logger.warning("No requested method is available; writing empty outputs.")
    elif not tasks:
        logger.info("All replicate tasks are already complete; rebuilding outputs.")
    elif effective_workers == 1:
        initialize_worker(methods)
        for task in tasks:
            record_output(task, run_replicate_task(task))
    else:
        logger.info("Launching %d replicate-level worker processes.", effective_workers)
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=initialize_worker,
            initargs=(tuple(methods),),
        ) as executor:
            future_to_task = {
                executor.submit(run_replicate_task, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    output = future.result()
                except Exception:
                    logger.exception(
                        "Fatal worker failure at d=%d replicate=%d.",
                        task.d,
                        task.replicate,
                    )
                    raise
                record_output(task, output)

    results = ordered_results_frame(rows, methods)
    _atomic_to_csv(results, raw_path)
    if require_all_methods:
        completed = results.groupby(["d", "method"]).size()
        missing_cells = [
            f"d={d}, method={method}: {int(completed.get((d, method), 0))}/{reps}"
            for d in d_values
            for method in methods
            if int(completed.get((d, method), 0)) != reps
        ]
        if missing_cells:
            raise RuntimeError(
                "Strict completeness check failed: " + "; ".join(missing_cells)
            )
    summary = make_summary(results)
    _atomic_to_csv(summary, summary_path)
    _atomic_to_csv(make_family_confusion(results), family_confusion_path)
    plot_summary(summary, output_directory)
    logger.info("Finished Experiment 1. Results saved to %s", output_directory)
    # Release Windows file handles so repeated test/calibration calls and
    # post-run promotion can inspect or move a completed run immediately.
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    return results, summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def resolve_methods(args: argparse.Namespace) -> List[str]:
    methods = list(dict.fromkeys(args.methods))
    additions = (
        (args.include_oracle, "OracleDP"),
        (args.include_pc, "PC-RCIT"),
        (args.include_pbscm, "PBSCM"),
        (args.include_pbscm_pgf, "PBSCM_PGF"),
        (args.include_poisson_dag, "PoissonDAG-ODS"),
        (args.include_cpcm, "CPCM"),
    )
    for include, method in additions:
        if include and method not in methods:
            methods.append(method)
    return methods


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 d-sweep for node-wise library-BIC PT-SEM.")
    parser.add_argument("--d-values", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--max-parents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--alpha-low", type=float, default=0.15)
    parser.add_argument("--alpha-high", type=float, default=0.85)
    parser.add_argument("--outdir", type=str, default="experiment1_results")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Replicate-level worker processes (recommended: 12 on this machine).",
    )
    parser.add_argument(
        "--require-all-methods",
        action="store_true",
        help="Fail unless every requested method completes every replicate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume complete (d, replicate) tasks from an existing raw CSV.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=["LibraryDP", "LibraryGreedy"],
        help="Explicit method list; --include-* flags append to this list.",
    )
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--include-pc", action="store_true")
    parser.add_argument("--include-pbscm", action="store_true")
    parser.add_argument("--include-pbscm-pgf", action="store_true")
    parser.add_argument("--include-poisson-dag", action="store_true")
    parser.add_argument("--include-cpcm", action="store_true")
    parser.add_argument("--avg-indegree", type=float, default=1.6, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if any(d < 1 for d in args.d_values):
        parser.error("all --d-values must be positive")
    if args.N < 1 or args.R < 1:
        parser.error("--N and --R must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_parents < 0:
        parser.error("--max-parents must be nonnegative")
    if args.avg_indegree < 0.0:
        parser.error("--avg-indegree must be nonnegative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.alpha_low < 0.0 or args.alpha_high <= args.alpha_low:
        parser.error("require 0 <= --alpha-low < --alpha-high")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    methods = resolve_methods(args)
    run_experiment(
        d_values=args.d_values,
        n=args.N,
        reps=args.R,
        methods=methods,
        max_parents=args.max_parents,
        avg_indegree=args.avg_indegree,
        seed=args.seed,
        outdir=args.outdir,
        workers=args.workers,
        require_all_methods=args.require_all_methods,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of src/d.py is disabled in PTSEM_FINAL. "
        "Use run.ps1 or run.sh so the locked All-Poisson design is enforced."
    )
