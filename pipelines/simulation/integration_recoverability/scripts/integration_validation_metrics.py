"""Finite-pool diagnostics; state scaling and cell-ID alignment belong to the caller."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix with features")
    return value


def _check_k(k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def transport_certificate(source_state, target_state, columns):
    """Bound squared error for one-Lipschitz state functions under a permutation.

    States must already use the same training-derived scale. The squared-cost
    oracle differs from the frozen benchmark's unsquared-distance assignment.
    """
    source = _matrix(source_state, "source_state")
    target = _matrix(target_state, "target_state")
    if source.shape != target.shape:
        raise ValueError("State pools must have the same row count and state dimension")
    n = len(source)
    columns = np.asarray(columns)
    if (columns.shape != (n,) or (n and columns.dtype.kind not in "iu")
            or not np.array_equal(np.sort(columns), np.arange(n))):
        raise ValueError("columns must be an integer permutation of the target row indices")
    result = dict(status="unavailable", n=n, matched_squared_cost=None,
                  oracle_squared_cost=None, random_expected_squared_cost=None,
                  normalized_gap=None, one_lipschitz_squared_error_bound=None)
    if n == 0:
        return {**result, "reason": "empty_pools"}
    costs = cdist(source, target, metric="sqeuclidean")
    if not np.isfinite(costs).all():
        raise ValueError("Squared state distances are nonfinite")
    rows, oracle_columns = linear_sum_assignment(costs)
    matched = float(costs[np.arange(n), columns].mean())
    oracle = float(costs[rows, oracle_columns].mean())
    random = float(costs.mean())
    if not np.isfinite([matched, oracle, random]).all():
        raise ValueError("Squared state costs are nonfinite")
    tolerance = 1e-12 * max(1.0, matched, oracle, random)
    gap, spread = matched - oracle, random - oracle
    if gap < -tolerance or spread < -tolerance:
        raise ValueError("Squared-cost oracle exceeds a feasible assignment or random expectation")
    result.update(status="ok", matched_squared_cost=matched, oracle_squared_cost=oracle,
                  random_expected_squared_cost=random,
                  one_lipschitz_squared_error_bound=matched)
    if spread <= tolerance:
        result["reason"] = "uninformative_reference"
    else:
        result["normalized_gap"] = max(0.0, gap) / spread
    return result


def neighbor_hubness(source_embedding, target_embedding, k=15):
    """Directed cross-assay kNN concentration before matching; ties use row order.

    Gini includes targets with zero indegree. The largest fraction uses all
    neighbor edges as its denominator, not the number of source cells.
    """
    source = _matrix(source_embedding, "source_embedding")
    target = _matrix(target_embedding, "target_embedding")
    k = _check_k(k)
    if source.shape[1] != target.shape[1]:
        raise ValueError("Cross-assay embeddings must have the same dimension")
    ns, nt = len(source), len(target)
    effective_k = min(k, nt) if ns else 0
    result = dict(status="unavailable", n_source=ns, n_target=nt, effective_k=effective_k,
                  indegree_gini=None, coverage=None, largest_indegree_fraction=None)
    if not ns or not nt:
        return {**result, "reason": "empty_pools"}
    costs = cdist(source, target, metric="sqeuclidean")
    if not np.isfinite(costs).all():
        raise ValueError("Embedding distances are nonfinite")
    neighbors = np.argsort(costs, axis=1, kind="stable")[:, :effective_k]
    indegree = np.bincount(neighbors.ravel(), minlength=nt)
    total = ns * effective_k
    gini = np.dot(2 * np.arange(1, nt + 1) - nt - 1, np.sort(indegree)) / (nt * total)
    return {**result, "status": "ok", "indegree_gini": float(gini),
            "coverage": float(np.count_nonzero(indegree) / nt),
            "largest_indegree_fraction": float(indegree.max() / total)}


def neighborhood_overlap(reference, embedding, k=15):
    """Mean same-cell kNN Jaccard; exclude self by row identity, including ties."""
    reference = _matrix(reference, "reference")
    embedding = _matrix(embedding, "embedding")
    k = _check_k(k)
    if len(reference) != len(embedding):
        raise ValueError("Reference and embedding must have the same aligned rows")
    n = len(reference)
    effective_k = min(k, max(0, n - 1))
    result = dict(status="unavailable", n=n, effective_k=effective_k, mean_knn_jaccard=None)
    if n <= 1:
        return {**result, "reason": "too_few_cells"}
    neighborhoods = []
    for values in (reference, embedding):
        costs = cdist(values, values, metric="sqeuclidean")
        if not np.isfinite(costs).all():
            raise ValueError("Embedding distances are nonfinite")
        np.fill_diagonal(costs, np.inf)
        neighborhoods.append(np.argsort(costs, axis=1, kind="stable")[:, :effective_k])
    overlaps = [len(set(first) & set(second)) for first, second in zip(*neighborhoods)]
    return {**result, "status": "ok",
            "mean_knn_jaccard": float(np.mean(np.asarray(overlaps) / (2 * effective_k - np.asarray(overlaps))))}
