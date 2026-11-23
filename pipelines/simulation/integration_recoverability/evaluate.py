"""Evaluator-only diagnostics for the integration-recoverability benchmark.

No function in this module is a deployable causal estimator from unpaired data.
Matching uses only opaque IDs and permitted representations.  Hidden values are
joined *after* matching, to diagnose the imposed coupling.  Count-PAG truth is
obtained from the measurement DAG, never from the activity skeleton.

The light-weight matching/scoring API needs NumPy and SciPy.  Discovery imports
causal-learn lazily, so validation can run in the separate AnnData environment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import io
import itertools
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
ENDPOINTS = {"TAIL", "ARROW", "CIRCLE"}


def _matrix(value: Any, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty-column 2D array")
    if rows is not None and len(array) != rows:
        raise ValueError(f"{name}: row count differs from IDs")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _ids(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=str)
    if array.ndim != 1 or len(set(array)) != len(array):
        raise ValueError(f"{name} must be a one-dimensional list of unique IDs")
    return array


def _indices(ids: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    lookup = {str(key): i for i, key in enumerate(ids)}
    try:
        return np.asarray([lookup[str(key)] for key in selected], dtype=int)
    except KeyError as error:
        raise ValueError(f"Unknown cell ID: {error.args[0]}") from error


def select_common_pools(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    observed_ids: Mapping[str, Sequence[str]],
    *,
    seed: int = 2718,
    assays: Sequence[str] = ASSAYS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Intersect eligible IDs across *all* supplied methods/seeds per assay.

    Nonfinite embeddings fail instead of quietly removing problematic cells.
    Equal-size subsets are sampled by IDs alone, with no access to truth.
    """
    if not runs:
        raise ValueError("At least one successfully encoded run is required")
    common: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {"seed": seed, "by_assay": {}}
    for assay in assays:
        permitted = set(_ids(observed_ids[assay], f"observed/{assay}"))
        eligible = permitted.copy()
        counts = {}
        for label, run in sorted(runs.items()):
            ids = _ids(run[assay]["cell_ids"], f"{label}/{assay}")
            _matrix(run[assay]["shared"], f"{label}/{assay}/shared", len(ids))
            if not set(ids) <= permitted:
                raise ValueError(f"{label}/{assay} contains non-observed cell IDs")
            eligible.intersection_update(ids)
            counts[label] = len(ids)
        common[assay] = np.asarray(sorted(eligible), dtype=str)
        audit["by_assay"][assay] = {
            "observed": len(permitted), "eligible_by_run": counts,
            "common_before_equalization": len(eligible),
        }
    n = min(map(len, common.values()))
    if n == 0:
        raise ValueError("There are no common eligible cells in at least one assay")
    for index, assay in enumerate(assays):
        if len(common[assay]) != n:
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            common[assay] = np.sort(rng.choice(common[assay], size=n, replace=False))
        audit["by_assay"][assay]["selected"] = n
        audit["by_assay"][assay]["selected_ids_sha256"] = hashlib.sha256(
            "\n".join(common[assay]).encode()
        ).hexdigest()
    return common, audit


def match_one_to_one(
    pools: Mapping[str, Sequence[str]],
    representations: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    anchor: str = "RNA",
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Euclidean minimum-total-cost assignment, or fixed random permutations.

    ``representations`` contains only cell_ids and shared arrays.  For an oracle
    comparator the caller may supply state as shared, clearly labeled as such.
    Counts and private activities cannot affect this matching function.
    """
    ids = {key: _ids(value, f"pool/{key}") for key, value in pools.items()}
    if anchor not in ids or len({len(value) for value in ids.values()}) != 1:
        raise ValueError("Matching requires an anchor and equal-size fixed pools")
    n = len(ids[anchor])
    if n == 0:
        raise ValueError("Cannot match empty pools")
    if (representations is None) == (random_seed is None):
        raise ValueError("Provide either representations or a random seed")
    selected = {anchor: ids[anchor].copy()}
    costs = {}
    embedded = {}
    if representations is not None:
        for assay in ids:
            rep = representations[assay]
            rep_ids = _ids(rep["cell_ids"], f"representation/{assay}")
            values = _matrix(rep["shared"], f"representation/{assay}", len(rep_ids))
            embedded[assay] = values[_indices(rep_ids, ids[assay])]
        if len({value.shape[1] for value in embedded.values()}) != 1:
            raise ValueError("Assays do not have a common representation dimension")
    for index, assay in enumerate(sorted(set(ids) - {anchor})):
        if random_seed is not None:
            rng = np.random.default_rng(np.random.SeedSequence([random_seed, index]))
            columns = rng.permutation(n)
            costs[assay] = None
        else:
            distances = cdist(embedded[anchor], embedded[assay], metric="euclidean")
            rows, columns = linear_sum_assignment(distances)
            if not np.array_equal(rows, np.arange(n)):
                raise RuntimeError("Assignment omitted an anchor")
            costs[assay] = {
                "sum_euclidean": float(distances[rows, columns].sum()),
                "mean_euclidean": float(distances[rows, columns].mean()),
            }
        selected[assay] = ids[assay][columns]
    return {
        "anchor": anchor, "cell_ids": selected, "costs": costs,
        "procedure": "random_one_to_one" if random_seed is not None else "euclidean_one_to_one",
        "random_seed": random_seed, "n_tuples": n,
        "interpretation": "An imposed coupling of different cells, not observed paired samples",
    }


def validate_truth(truth: Mapping[str, Any], regime: str) -> dict[str, np.ndarray]:
    """Check the simulator's evaluator-only compact NPZ contract."""
    keys = ("cell_ids", "assigned_assay", "state", "activities", "exposures",
            "candidate_feature_ids", "candidate_assays", "candidate_roles", "candidate_names")
    out = {key: np.asarray(truth[key]) for key in keys}
    out["cell_ids"] = _ids(out["cell_ids"], "truth/cell_ids")
    n = len(out["cell_ids"])
    names = _ids(out["candidate_names"], "truth/candidate_names")
    p = len(names)
    for key in ("candidate_feature_ids", "candidate_assays", "candidate_roles"):
        if out[key].shape != (p,):
            raise ValueError(f"truth/{key} is not aligned with candidate_names")
    if out["assigned_assay"].shape != (n,):
        raise ValueError("truth/assigned_assay is not aligned with cell_ids")
    for key in ("state", "activities", "exposures"):
        out[key] = _matrix(out[key], f"truth/{key}", n)
    if out["activities"].shape[1] != p or out["exposures"].shape[1] != len(ASSAYS):
        raise ValueError("Truth activity/exposure dimensions are inconsistent")
    counts = _matrix(truth[f"counts_{regime}"], f"truth/counts_{regime}", n)
    if counts.shape[1] != p or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Truth counts must be nonnegative integers aligned with candidates")
    out["counts"] = counts
    for key in ("conditional_mean_activities", "conditional_mean_standard_errors"):
        if key in truth:
            out[key] = _matrix(truth[key], f"truth/{key}", n)
    return out


def assemble_readout(assignment: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
    """Copy original selected observations, then private activities separately.

    Truth's complete panels are storage conveniences only: each observed count
    is read from the row belonging to the selected cell's *assigned* assay.
    ``validate-observed`` verifies these entries against original AnnData files.
    """
    ids = truth["cell_ids"]
    candidate_assays = np.asarray(truth["candidate_assays"], dtype=str)
    n = assignment["n_tuples"]
    p = len(candidate_assays)
    counts = np.empty((n, p))
    activities = np.empty((n, p))
    states = {}
    exposures = {}
    for assay, selected in assignment["cell_ids"].items():
        selected = _ids(selected, f"assignment/{assay}")
        if len(selected) != n:
            raise ValueError("Assignment row count differs across assays")
        rows = _indices(ids, selected)
        if not np.all(np.asarray(truth["assigned_assay"])[rows] == assay):
            raise ValueError(f"Matched {assay} cells were not observed in that assay")
        columns = np.flatnonzero(candidate_assays == assay)
        counts[:, columns] = truth["counts"][np.ix_(rows, columns)]
        activities[:, columns] = truth["activities"][np.ix_(rows, columns)]
        states[assay] = truth["state"][rows]
        exposures[assay] = truth["exposures"][rows, ASSAYS.index(assay)]
    if set(candidate_assays) != set(assignment["cell_ids"]):
        raise ValueError("Assignment does not cover every candidate assay")
    return {"counts": counts, "activities": activities, "state_by_assay": states,
            "exposure_by_assay": exposures, "cell_ids": assignment["cell_ids"]}


def state_matching_metrics(readout: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for first, second in itertools.combinations(ASSAYS, 2):
        difference = readout["state_by_assay"][first] - readout["state_by_assay"][second]
        rows[f"{first}--{second}"] = {
            "mean_absolute_difference": float(np.abs(difference).mean()),
            "mae_by_dimension": np.abs(difference).mean(axis=0).tolist(),
            "mean_euclidean_distance": float(np.linalg.norm(difference, axis=1).mean()),
        }
    return rows


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    x = first - np.mean(first)
    y = second - np.mean(second)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return None if norm == 0 else float(np.clip(x @ y / norm, -1, 1))


def dependence_metrics(
    values: np.ndarray, reference: np.ndarray,
    candidates: Sequence[Mapping[str, str]], *, scale: str,
) -> list[dict[str, Any]]:
    values = _matrix(values, "dependence values")
    reference = _matrix(reference, "dependence reference")
    if values.shape[1] != len(candidates) or reference.shape[1] != len(candidates):
        raise ValueError("Dependence columns do not match candidate order")
    ranked = np.apply_along_axis(rankdata, 0, values)
    reference_ranked = np.apply_along_axis(rankdata, 0, reference)
    result = []
    for i, j in itertools.combinations(range(len(candidates)), 2):
        row = {
            "node1": candidates[i]["name"], "node2": candidates[j]["name"],
            "assay_pair": "--".join(sorted((candidates[i]["assay"], candidates[j]["assay"]))),
            "cross_assay": candidates[i]["assay"] != candidates[j]["assay"],
            "sentinel_incident": "sentinel" in (candidates[i]["role"], candidates[j]["role"]),
            "scale": scale,
        }
        for label, current, paired in (("pearson", values, reference), ("spearman", ranked, reference_ranked)):
            estimate = _correlation(current[:, i], current[:, j])
            baseline = _correlation(paired[:, i], paired[:, j])
            row[label] = estimate
            row[f"paired_{label}"] = baseline
            row[f"{label}_absolute_error"] = (
                None if estimate is None or baseline is None else abs(estimate - baseline)
            )
        row["status"] = "undefined_constant" if row["pearson"] is None or row["paired_pearson"] is None else "ok"
        result.append(row)
    return result


@dataclass(frozen=True)
class DiscoveryConfig:
    alpha: float = 0.01
    seed: int = 0
    test: str = "rcit"
    # Native lpd4 catches exceptions and silently substitutes hbe. Pin hbe
    # directly so the chosen approximation has no such alternate backend.
    approximation: str = "hbe"
    rcit: bool = True
    num_f: int = 100
    num_f2: int = 5
    min_cells: int = 20


@contextmanager
def _numpy_seed(seed: int):
    old = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(old)


def _query_seed(seed: int, x: int, y: int, z: Sequence[int]) -> int:
    key = json.dumps([seed, min(x, y), max(x, y), sorted(z)])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def _register_strict_ci() -> str:
    from causallearn.utils.cit import CIT, CIT_Base, register_ci_test

    class StrictCI(CIT_Base):
        def __init__(self, data, **kwargs):
            super().__init__(data)
            self.check_cache_method_consistent("recoverability_strict", "in_memory_only")
            self.config = kwargs["recoverability_config"]
            self.log = kwargs["recoverability_test_log"]
            self.names = kwargs["recoverability_names"]
            backend_kwargs = {}
            if self.config.test == "rcit":
                backend_kwargs = dict(approx=self.config.approximation, rcit=self.config.rcit,
                                      num_f=self.config.num_f, num_f2=self.config.num_f2)
            self.backend = CIT(data, self.config.test, **backend_kwargs)

        def __call__(self, X, Y, condition_set=None):
            xs, ys, z, key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
            if key in self.pvalue_cache:
                return self.pvalue_cache[key]
            seed = _query_seed(self.config.seed, xs[0], ys[0], z)
            entry = {"x": self.names[xs[0]], "y": self.names[ys[0]],
                     "conditioning": [self.names[i] for i in z], "seed": seed}
            try:
                with _numpy_seed(seed), warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    value = float(self.backend(xs[0], ys[0], z))
                entry["warnings"] = sorted({str(w.message) for w in seen})
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise FloatingPointError(f"CI test returned invalid p-value {value}")
                self.pvalue_cache[key] = value
                entry.update(status="ok", p_value=value)
                self.log.append(entry)
                return value
            except Exception as error:
                entry.update(status="failed", p_value=None, error_type=type(error).__name__, error=str(error))
                self.log.append(entry)
                raise

    name = "recoverability_strict_ci_v1"
    register_ci_test(name, StrictCI)
    return name


def backend_provenance(config: DiscoveryConfig) -> dict[str, Any]:
    import causallearn.utils.cit as cit
    source_file = Path(inspect.getfile(cit.RCIT))
    result = {
        "package": "causal-learn", "version": importlib.metadata.version("causal-learn"),
        "requested_test": config.test, "configuration": asdict(config),
        "cit_source_path": str(source_file), "cit_source_sha256": _sha256(source_file),
        "fallback": "none", "seed_policy": "SHA256(base seed, canonical X/Y/conditioning); restore NumPy RNG",
    }
    if config.test == "rcit":
        import causallearn.utils.RCIT.RCIT as module
        source = Path(inspect.getfile(module))
        result.update(
            conditional_class="causallearn.utils.RCIT.RCIT.RCIT",
            marginal_class="causallearn.utils.RCIT.RCIT.RIT",
            variant="RCIT" if config.rcit else "RCoT",
            backend_source_sha256=_sha256(source),
            native_negative_pvalue_clipping=True,
        )
    return result


def _canonical_edges(graph) -> list[dict[str, str]]:
    result = []
    for edge in graph.get_graph_edges():
        first, second = edge.get_node1().get_name(), edge.get_node2().get_name()
        one, two = edge.get_endpoint1().name, edge.get_endpoint2().name
        if one not in ENDPOINTS or two not in ENDPOINTS:
            raise ValueError(f"Unsupported endpoint combination {one}/{two}")
        if second < first:
            first, second, one, two = second, first, two, one
        result.append(dict(node1=first, node2=second, endpoint1=one, endpoint2=two))
    return sorted(result, key=lambda edge: (edge["node1"], edge["node2"]))


def discover(
    values: np.ndarray, names: Sequence[str], *, method: str, scale: str,
    config: DiscoveryConfig | None = None,
    diagnostic_queries: Sequence[tuple[str, str, Sequence[str]]] = (),
) -> dict[str, Any]:
    """PC/FCI with an explicit backend and no test or direction fallback.

    ``counts`` means original nonnegative integer counts, transformed here with
    log1p exactly once. ``activities`` are passed through. A failed test does
    not become an empty graph; edges is None and the reason is retained.
    """
    config = config or DiscoveryConfig()
    names = list(map(str, names))
    started = time.monotonic()
    tests: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "status": "failed", "method": method, "scale": scale, "names": list(names),
        "edges": None, "tests": tests, "configuration": asdict(config),
        "directional_prior": None, "conditioning_depth": -1,
        "nominal_pvalues": "diagnostic; no claimed calibration after matching",
    }
    try:
        data = _matrix(values, "discovery data")
        if data.shape[1] != len(names) or len(set(names)) != len(names):
            raise ValueError("Unique explicit node names must match every column")
        if len(data) < config.min_cells:
            raise ValueError(f"Too few cells: {len(data)} < {config.min_cells}")
        if not 0 < config.alpha < 1 or config.test not in {"rcit", "kci", "fisherz"}:
            raise ValueError("Unsupported CI configuration")
        if config.test == "rcit" and config.approximation not in {"hbe", "gamma", "chi2", "perm"}:
            raise ValueError("Choose a fixed RCIT approximation without the lpd4 fallback")
        if scale == "counts":
            if np.any(data < 0) or np.any(data != np.floor(data)):
                raise ValueError("Counts must be raw nonnegative integers, not pre-logged inputs")
            data = np.log1p(data)
            result["input_transform"] = "log1p_once"
        elif scale == "activities":
            result["input_transform"] = "none"
        else:
            raise ValueError("scale must be activities or counts")
        constants = [str(names[i]) for i in np.flatnonzero(np.ptp(data, axis=0) == 0)]
        if constants:
            raise ValueError(f"Constant columns: {constants}")
        result["backend"] = backend_provenance(config)
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import CIT
        test_name = _register_strict_ci()
        kwargs = dict(recoverability_config=config, recoverability_test_log=tests,
                      recoverability_names=list(names))
        output = io.StringIO()
        with redirect_stdout(output):
            if method == "pc":
                graph = pc(data, alpha=config.alpha, indep_test=test_name, stable=True,
                           uc_rule=0, uc_priority=2, background_knowledge=None,
                           show_progress=False, node_names=list(names), **kwargs).G
            elif method == "fci":
                graph, _ = fci(data, independence_test_method=test_name, alpha=config.alpha,
                               depth=-1, max_path_length=-1, background_knowledge=None,
                               show_progress=False, node_names=list(names), **kwargs)
            else:
                raise ValueError("method must be pc or fci")
        result["backend_stdout"] = output.getvalue()
        result["edges"] = _canonical_edges(graph)
        result["status"] = "ok"
        result["diagnostic_queries"] = []
        query_test = CIT(data, test_name, **kwargs)
        for first, second, conditioned in diagnostic_queries:
            row = {"x": first, "y": second, "conditioning": list(conditioned)}
            try:
                p = query_test(names.index(first), names.index(second), [names.index(x) for x in conditioned])
                row.update(status="ok", p_value=p, rejects_independence=bool(p < config.alpha))
            except Exception as error:
                row.update(status="failed", p_value=None, error_type=type(error).__name__, error=str(error))
            result["diagnostic_queries"].append(row)
    except Exception as error:
        result.update(status="failed", edges=None, error_type=type(error).__name__, error=str(error))
    result["runtime_seconds"] = time.monotonic() - started
    result["n_rows"] = int(np.asarray(values).shape[0])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def d_separated(
    edges: Sequence[Sequence[str]], first: str, second: str, conditioned: Iterable[str] = (),
    *, nodes: Iterable[str] = (),
) -> bool:
    """Exact DAG d-separation by ancestral moralization (evaluator only)."""
    import networkx as nx
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(edges)
    z = set(conditioned)
    if first == second or {first, second} & z:
        raise ValueError("d-separation requires disjoint query and conditioning nodes")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Truth graph is not a DAG")
    relevant = {first, second} | z
    if not relevant <= set(dag):
        raise ValueError("Query references a node absent from the declared DAG")
    ancestors = relevant.copy()
    for node in relevant:
        ancestors.update(nx.ancestors(dag, node))
    restricted = dag.subgraph(ancestors)
    moral = restricted.to_undirected()
    for node in restricted:
        moral.add_edges_from(itertools.combinations(restricted.predecessors(node), 2))
    moral.remove_nodes_from(z)
    return not nx.has_path(moral, first, second)


def oracle_graph(
    dag_edges: Sequence[Sequence[str]], observed_names: Sequence[str], *,
    method: str = "fci", all_nodes: Iterable[str] = (),
) -> dict[str, Any]:
    """Construct the oracle CPDAG/PAG from exact d-separation, not samples.

    FCI permits arbitrary declared latent ancestors in dag_edges.  PC is only
    appropriate for causally sufficient activity controls.  Graphical truth
    assumes the DAG Markov/faithfulness model; it is not estimated from counts.
    """
    started = time.monotonic()
    result = {"status": "failed", "edges": None, "method": method,
              "names": list(observed_names), "test": "exact_d_separation"}
    try:
        from causallearn.utils.cit import CIT_Base, register_ci_test
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ConstraintBased.FCI import fci
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_nodes_from(set(all_nodes) | set(observed_names))
        dag.add_edges_from(dag_edges)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Truth graph contains a cycle")
        names = list(observed_names)
        if len(set(names)) != len(names):
            raise ValueError("Repeated observed truth node")
        test_log = []

        class OracleCI(CIT_Base):
            def __init__(self, data, **kwargs):
                super().__init__(data)
                self.check_cache_method_consistent("recoverability_oracle", "exact_dsep")

            def __call__(self, X, Y, condition_set=None):
                xs, ys, z, key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
                if key not in self.pvalue_cache:
                    separated = d_separated(dag_edges, names[xs[0]], names[ys[0]],
                                            [names[i] for i in z], nodes=dag.nodes)
                    self.pvalue_cache[key] = float(separated)
                    test_log.append({"x": names[xs[0]], "y": names[ys[0]],
                                     "conditioning": [names[i] for i in z], "separated": separated})
                return self.pvalue_cache[key]

        test_name = "recoverability_exact_dsep_v1"
        register_ci_test(test_name, OracleCI)
        # The custom test never accesses these numbers. They satisfy only the
        # library interface and are not simulated or measured data.
        dummy = np.zeros((max(20, len(names) + 1), len(names)))
        with redirect_stdout(io.StringIO()):
            if method == "pc":
                graph = pc(dummy, alpha=0.5, indep_test=test_name, show_progress=False,
                           stable=True, background_knowledge=None, node_names=names).G
            elif method == "fci":
                graph, _ = fci(dummy, independence_test_method=test_name, alpha=0.5,
                               depth=-1, max_path_length=-1, show_progress=False,
                               background_knowledge=None, node_names=names)
            else:
                raise ValueError("method must be pc or fci")
        result.update(status="ok", edges=_canonical_edges(graph), tests=test_log,
                      package_version=importlib.metadata.version("causal-learn"))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result["runtime_seconds"] = time.monotonic() - started
    return result


def measurement_dag(
    biological_edges: Sequence[Sequence[str]], candidates: Sequence[Mapping[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Activities and independent assay exposures cause noisy count leaves."""
    edges = [tuple(edge) for edge in biological_edges]
    measured = []
    for candidate in candidates:
        name = candidate["name"]
        count = f"count::{name}"
        measured.append(count)
        edges.extend(((name, count), (f"exposure::{candidate['assay']}", count)))
    return edges, measured


def _edge_map(edges: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], tuple[str, str]]:
    result = {}
    for edge in edges:
        a, b = edge["node1"], edge["node2"]
        one, two = edge["endpoint1"], edge["endpoint2"]
        if a == b or one not in ENDPOINTS or two not in ENDPOINTS:
            raise ValueError("Invalid canonical edge")
        if b < a:
            a, b, one, two = b, a, two, one
        if (a, b) in result:
            raise ValueError("Repeated adjacency")
        result[(a, b)] = (one, two)
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _set_scores(predicted: set, truth: set, universe: set | None = None) -> dict[str, Any]:
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    result = {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "n_predicted": len(predicted), "n_true": len(truth),
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
    if universe is not None:
        result["false_positive_rate"] = _ratio(fp, len(universe - truth))
    return result


def score_graph(
    discovery: Mapping[str, Any], reference: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]], *, target: str,
    biological_edges: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Score adjacencies/endpoints against the matching graphical target.

    Direct biological edge agreement is reported only as a separately labeled
    identifiability stress test, never confused with a latent-projection PAG.
    """
    base = {"target": target, "status": "not_applicable"}
    if discovery.get("status") != "ok" or reference.get("status") != "ok":
        return {**base, "reason": "Discovery or corresponding graph oracle failed", "metrics": None}
    predicted, truth = _edge_map(discovery["edges"]), _edge_map(reference["edges"])
    by_name = {item["name"]: item for item in candidates}
    universe = set(itertools.combinations(sorted(by_name), 2))
    if not set(predicted) <= universe or not set(truth) <= universe:
        raise ValueError("Graph contains nodes outside the declared scoring panel")
    result = {**base, "status": "ok", "adjacency": _set_scores(set(predicted), set(truth), universe)}
    correct_endpoints = sum(
        int(predicted[pair][end] == truth[pair][end])
        for pair in set(predicted) & set(truth) for end in (0, 1)
    )
    result["endpoints"] = {
        "n_correct": correct_endpoints,
        "precision": _ratio(correct_endpoints, 2 * len(predicted)),
        "recall": _ratio(correct_endpoints, 2 * len(truth)),
        "definition": "Exact TAIL/ARROW/CIRCLE equality; missing and extra adjacencies penalized",
    }
    scopes: dict[str, set] = {"within_assay": set(), "cross_assay": set()}
    for pair in universe:
        assay1, assay2 = by_name[pair[0]]["assay"], by_name[pair[1]]["assay"]
        scope = "within_assay" if assay1 == assay2 else "cross_assay"
        scopes[scope].add(pair)
        scopes.setdefault("--".join(sorted((assay1, assay2))), set()).add(pair)
    result["by_assay_pair"] = {
        scope: _set_scores(set(predicted) & pairs, set(truth) & pairs, pairs)
        for scope, pairs in sorted(scopes.items())
    }
    sentinel_pairs = {pair for pair in universe if any(by_name[node]["role"] == "sentinel" for node in pair)}
    result["sentinel"] = {
        "n_possible_incident_pairs": len(sentinel_pairs),
        "n_returned_incident_adjacencies": len(set(predicted) & sentinel_pairs),
        "incident_adjacency_rate": _ratio(len(set(predicted) & sentinel_pairs), len(sentinel_pairs)),
        "false_positive_rate_against_current_graph_target": _ratio(
            len((set(predicted) - set(truth)) & sentinel_pairs), len(sentinel_pairs - set(truth))
        ),
        "interpretation": (
            "Biological sentinel activities are independent; incident activity edges are false associations"
            if "activity" in target else
            "Sentinel counts share assay exposure; within-assay edges can be measurement-graph truth"
        ),
    }
    cross = scopes["cross_assay"]
    biological = {tuple(sorted(edge)) for edge in biological_edges if tuple(sorted(edge)) in cross}
    true_directed = {tuple(edge) for edge in biological_edges if tuple(sorted(edge)) in cross}
    predicted_directed = set()
    for pair, endpoints in predicted.items():
        if pair not in cross:
            continue
        if endpoints == ("TAIL", "ARROW"):
            predicted_directed.add(pair)
        elif endpoints == ("ARROW", "TAIL"):
            predicted_directed.add(pair[::-1])
    result["exact_cross_assay_stress_test"] = {
        "interpretation": "Identifiability stress test against biological direct edges; not a success requirement",
        "pair": _set_scores(set(predicted) & cross, biological, cross),
        "direction": _set_scores(predicted_directed, true_directed),
        "exact_pair_set": set(predicted) & cross == biological,
        "exact_directed_set": predicted_directed == true_directed,
    }
    return result


def optional_structure_metrics(
    candidates: Sequence[Mapping[str, str]], *,
    predicted_hyperedges: Sequence[Sequence[str]] | None = None,
    true_hyperedges: Sequence[Sequence[str]] | None = None,
    compatible_models: Sequence[str] | None = None,
    candidate_family: Sequence[str] | None = None,
    true_model: str | None = None,
) -> dict[str, Any]:
    """Only score objects a method actually returns; never infer hyperedges."""
    result: dict[str, Any] = {
        "hyperedges": {"status": "not_applicable", "reason": "No explicit hyperedge output and truth contract"},
        "equivalence_class": {"status": "not_applicable", "reason": "No justified candidate family and compatibility-set output"},
    }
    by_name = {row["name"]: row for row in candidates}
    if predicted_hyperedges is not None and true_hyperedges is not None:
        pred = {frozenset(edge) for edge in predicted_hyperedges}
        truth = {frozenset(edge) for edge in true_hyperedges}
        if any(len(edge) < 2 or not edge <= set(by_name) for edge in pred | truth):
            raise ValueError("Invalid hyperedge or unknown candidate")
        pred_members = set().union(*pred) if pred else set()
        true_members = set().union(*truth) if truth else set()
        pred_assays = {by_name[node]["assay"] for node in pred_members}
        true_assays = {by_name[node]["assay"] for node in true_members}
        sentinels = {name for name, row in by_name.items() if row["role"] == "sentinel"}
        result["hyperedges"] = {
            "status": "ok", "exact_membership": _set_scores(pred, truth),
            "candidate_membership": _set_scores(pred_members, true_members),
            "assay_participation_union": _set_scores(pred_assays, true_assays),
            "returned_assays_per_hyperedge": [sorted({by_name[x]["assay"] for x in edge}) for edge in sorted(pred, key=lambda x: sorted(x))],
            "sentinel_participation_rate": _ratio(len(pred_members & sentinels), len(sentinels)),
        }
    if compatible_models is not None and candidate_family is not None and true_model is not None:
        compatible, family = set(compatible_models), set(candidate_family)
        if not compatible <= family or true_model not in family:
            raise ValueError("Ambiguity-set members and truth must belong to the declared family")
        result["equivalence_class"] = {
            "status": "ok", "coverage": true_model in compatible, "ambiguity_set_size": len(compatible),
            "candidate_family_size": len(family),
            "scope": "Coverage of this declared finite family only, not all compatible unpaired models",
        }
    return result


def enumerate_three_node_dags(nodes: Sequence[str]) -> list[tuple[tuple[str, str], ...]]:
    """The 25 labeled DAGs, solely for causally sufficient three-node controls."""
    import networkx as nx
    if len(nodes) != 3 or len(set(nodes)) != 3:
        raise ValueError("Exactly three distinct nodes are required")
    pairs = list(itertools.combinations(sorted(nodes), 2))
    result = []
    for choices in itertools.product((0, 1, -1), repeat=3):
        edges = tuple(sorted(pair if choice == 1 else pair[::-1]
                             for pair, choice in zip(pairs, choices) if choice))
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        if nx.is_directed_acyclic_graph(graph):
            result.append(edges)
    return result


def markov_equivalence_class_three_nodes(
    nodes: Sequence[str], truth_edges: Sequence[Sequence[str]],
) -> dict[str, Any]:
    def signature(edges):
        edge_set = set(map(tuple, edges))
        skeleton = {frozenset(edge) for edge in edge_set}
        colliders = set()
        for first, second in itertools.combinations(nodes, 2):
            for middle in set(nodes) - {first, second}:
                if (first, middle) in edge_set and (second, middle) in edge_set and frozenset((first, second)) not in skeleton:
                    colliders.add((frozenset((first, second)), middle))
        return skeleton, colliders
    family = enumerate_three_node_dags(nodes)
    target = tuple(sorted(tuple(edge) for edge in truth_edges))
    if target not in family:
        raise ValueError("Truth is not a DAG in the declared three-node family")
    compatible = [edges for edges in family if signature(edges) == signature(target)]
    return {
        "candidate_family_size": len(family), "oracle_markov_equivalence_size": len(compatible),
        "compatible_dags": [[list(edge) for edge in edges] for edges in compatible],
        "scope": "Oracle Markov-equivalence diagnostic under causal sufficiency/faithfulness, not learned unpaired ambiguity output",
    }


def evaluate_probes(
    calibration_embeddings: Mapping[str, Mapping[str, Any]],
    assessment_embeddings: Mapping[str, Mapping[str, Any]],
    calibration_truth: Mapping[str, Any], assessment_truth: Mapping[str, Any],
    *, seed: int = 0,
) -> dict[str, Any]:
    """Frozen ridge/tree probes fitted only on private calibration cells.

    S-only versus (S,Z) prediction is an information diagnostic with matched
    probe capacities, not a claim of exact conditional-independence testing.
    When available, simulation Monte Carlo conditional means provide additional
    residual targets, explicitly labeled as approximate rather than exact.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cal_ids = _ids(calibration_truth["cell_ids"], "probe calibration IDs")
    test_ids = _ids(assessment_truth["cell_ids"], "probe assessment IDs")
    if set(cal_ids) & set(test_ids):
        raise ValueError("Probe calibration and assessment cells overlap")
    names = list(map(str, calibration_truth["candidate_names"]))
    if names != list(map(str, assessment_truth["candidate_names"])):
        raise ValueError("Probe target feature order changed across cohorts")
    states_cal = _matrix(calibration_truth["state"], "calibration state", len(cal_ids))
    states_test = _matrix(assessment_truth["state"], "assessment state", len(test_ids))
    activities_cal = _matrix(calibration_truth["activities"], "calibration activities", len(cal_ids))
    activities_test = _matrix(assessment_truth["activities"], "assessment activities", len(test_ids))
    target_assays = list(map(str, calibration_truth["candidate_assays"]))
    targets = [(f"state_{index}", "state", states_cal[:, index], states_test[:, index])
               for index in range(states_cal.shape[1])]
    targets += [(name, target_assays[index], activities_cal[:, index], activities_test[:, index])
                for index, name in enumerate(names)]
    have_residuals = "conditional_mean_activities" in calibration_truth and "conditional_mean_activities" in assessment_truth
    if have_residuals:
        mean_cal = _matrix(calibration_truth["conditional_mean_activities"], "calibration conditional mean", len(cal_ids))
        mean_test = _matrix(assessment_truth["conditional_mean_activities"], "assessment conditional mean", len(test_ids))
        if mean_cal.shape != activities_cal.shape or mean_test.shape != activities_test.shape:
            raise ValueError("Conditional mean activity dimensions differ")
        targets += [(f"residual::{name}", target_assays[index],
                     activities_cal[:, index] - mean_cal[:, index], activities_test[:, index] - mean_test[:, index])
                    for index, name in enumerate(names)]

    def estimator(kind):
        if kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                         max_depth=2, min_samples_leaf=10, random_state=seed,
                                         loss="squared_error", subsample=1.0)

    def scores(y, predicted):
        mse = float(np.mean((y - predicted) ** 2))
        variance = float(np.var(y))
        return {"mse": mse, "r2": None if variance == 0 else 1 - mse / variance}

    output: dict[str, Any] = {
        "status": "ok", "seed": seed, "rows": [],
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "fit_access": "private calibration only; no model/integration selection from probe scores",
        "residual_targets": "simulation Monte Carlo conditional means" if have_residuals else "not available; incremental prediction only",
        "caveat": "Restricted probes can miss information; S-only misspecification can inflate incremental gains",
        "configuration": {"ridge_alpha": 1.0, "tree_estimators": 100, "tree_depth": 2,
                          "tree_min_samples_leaf": 10, "tree_learning_rate": 0.05},
    }
    if have_residuals:
        output["conditional_mean_uncertainty"] = {}
        for label, truth in (("calibration", calibration_truth), ("assessment", assessment_truth)):
            if "conditional_mean_standard_errors" in truth:
                errors = _matrix(truth["conditional_mean_standard_errors"], f"{label} conditional-mean standard errors")
                output["conditional_mean_uncertainty"][label] = {
                    "median_standard_error_by_target": np.median(errors, axis=0).tolist(),
                    "max_standard_error_by_target": np.max(errors, axis=0).tolist(),
                    "monte_carlo_repetitions": int(truth["conditional_mean_repetitions"])
                    if "conditional_mean_repetitions" in truth else None,
                }
    # Share the S-only baseline across assays/representations when retained ID
    # pools are identical, but never across changed calibration/assessment pools.
    baseline_cache = {}
    for assay in ASSAYS:
        cal = calibration_embeddings[assay]
        test = assessment_embeddings[assay]
        cal_embed_ids = _ids(cal["cell_ids"], f"{assay} probe calibration embeddings")
        test_embed_ids = _ids(test["cell_ids"], f"{assay} probe assessment embeddings")
        ci, ti = _indices(cal_ids, cal_embed_ids), _indices(test_ids, test_embed_ids)
        zcal = _matrix(cal["shared"], f"{assay} calibration shared", len(ci))
        ztest = _matrix(test["shared"], f"{assay} assessment shared", len(ti))
        representations = {"shared": (zcal, ztest)}
        if "private" in cal or "private" in test:
            if "private" not in cal or "private" not in test:
                raise ValueError("Private embedding is missing from one probe cohort")
            pcal = _matrix(cal["private"], f"{assay} calibration private", len(ci))
            ptest = _matrix(test["private"], f"{assay} assessment private", len(ti))
            representations["shared_plus_private"] = (np.column_stack((zcal, pcal)), np.column_stack((ztest, ptest)))
        if len(ci) < 20 or len(ti) < 20:
            output["rows"].append({"assay": assay, "status": "failed", "reason": "Too few retained probe cells"})
            output["status"] = "partial_failure"
            continue
        for representation, (xcal, xtest) in representations.items():
            if xcal.shape[1] != xtest.shape[1]:
                raise ValueError("Embedding dimensions differ across probe cohorts")
            for kind in ("ridge", "shallow_boosted_tree"):
                for name, target_assay, ycal, ytest in targets:
                    row = {"source_assay": assay, "representation": representation, "probe": kind,
                           "target": name, "target_assay": target_assay,
                           "relationship": "state" if target_assay == "state" else ("same_assay" if assay == target_assay else "cross_assay"),
                           "n_calibration": len(ci), "n_assessment": len(ti)}
                    try:
                        predictions = estimator(kind).fit(xcal, ycal[ci]).predict(xtest)
                        row["embedding_only"] = scores(ytest[ti], predictions)
                        if target_assay != "state":
                            key = (kind, name, tuple(ci), tuple(ti))
                            if key not in baseline_cache:
                                baseline = estimator(kind).fit(states_cal[ci], ycal[ci]).predict(states_test[ti])
                                baseline_cache[key] = scores(ytest[ti], baseline)
                            row["state_only"] = baseline_cache[key]
                            augmented = estimator(kind).fit(np.column_stack((states_cal[ci], xcal)), ycal[ci]).predict(
                                np.column_stack((states_test[ti], xtest)))
                            row["state_plus_embedding"] = scores(ytest[ti], augmented)
                            row["mse_gain_over_true_state"] = row["state_only"]["mse"] - row["state_plus_embedding"]["mse"]
                        row["status"] = "ok"
                    except Exception as error:
                        row.update(status="failed", error_type=type(error).__name__, error=str(error))
                        output["status"] = "partial_failure"
                    output["rows"].append(row)
    return output


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite value reached JSON output; report explicitly as undefined")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite changed artifact: {path}")
        return
    with path.open("x") as handle:
        handle.write(encoded)


def _save_npz_new(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _load_npz(path)
        if set(existing) != set(arrays) or any(not np.array_equal(existing[key], value) for key, value in arrays.items()):
            raise FileExistsError(f"Refusing to overwrite changed readout: {path}")
        return
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def validate_observed_files(world: Path, regime: str, *, cohorts: Sequence[str] | None = None) -> dict[str, Any]:
    """Run in the AnnData environment; validate compact truth against raw X."""
    import anndata as ad
    from scipy import sparse
    manifest = json.loads((world / "truth" / "manifest.json").read_text())
    candidates = manifest["candidates"]
    paths = [world / "truth" / f"{name}.npz" for name in cohorts] if cohorts else sorted((world / "truth").glob("*.npz"))
    result = {"status": "ok", "regime": regime, "world": str(world.resolve()),
              "anndata_version": importlib.metadata.version("anndata"), "cohorts": {},
              "truth_manifest_sha256": _sha256(world / "truth" / "manifest.json")}
    for path in paths:
        truth = validate_truth(_load_npz(path), regime)
        if list(map(str, truth["candidate_names"])) != [row["name"] for row in candidates]:
            raise ValueError("Candidate manifest and NPZ order disagree")
        cohort = path.stem
        entries = {}
        paired = np.all(truth["assigned_assay"] == "paired")
        for assay in ASSAYS:
            source = world / ("private_inputs" if paired else "observed") / regime / cohort / f"{assay}.h5ad"
            adata = ad.read_h5ad(source, backed="r")
            try:
                ids = _ids(adata.obs_names, f"{source}/obs_names")
                expected_ids = truth["cell_ids"] if paired else truth["cell_ids"][truth["assigned_assay"] == assay]
                if set(ids) != set(expected_ids):
                    raise ValueError(f"Observed ID set differs from truth: {source}")
                rows = _indices(truth["cell_ids"], ids)
                columns = np.flatnonzero(truth["candidate_assays"] == assay)
                feature_ids = list(map(str, truth["candidate_feature_ids"][columns]))
                values = adata[:, feature_ids].X
                values = values.toarray() if sparse.issparse(values) else np.asarray(values)
                expected = truth["counts"][np.ix_(rows, columns)]
                if not np.array_equal(values, expected):
                    raise ValueError(f"Original observed candidate counts disagree with truth: {source}")
                entries[assay] = {"path": str(source.resolve()), "sha256": _sha256(source),
                                  "n_cells": len(ids), "n_scored_features": len(columns)}
            finally:
                adata.file.close()
        result["cohorts"][cohort] = {"truth_sha256": _sha256(path), "assays": entries}
    return result


def _load_embedding_run(
    directory: Path, cohort: str, *, require_common_dimension: bool = True,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    provenance = {"directory": str(directory.resolve()), "files": {}}
    label = directory.name
    for candidate in ("embedding_manifest.json", "manifest.json"):
        path = directory / candidate
        if path.exists():
            manifest = json.loads(path.read_text())
            label = str(manifest.get("label", label))
            provenance.update(manifest=manifest, manifest_sha256=_sha256(path))
            break
    else:
        provenance["manifest_warning"] = "Missing integration manifest; directory name used as label"
    arrays = {}
    for assay in ASSAYS:
        path = directory / cohort / f"{assay}.npz"
        values = _load_npz(path)
        _ids(values["cell_ids"], str(path))
        _matrix(values["shared"], str(path), len(values["cell_ids"]))
        arrays[assay] = values
        provenance["files"][assay] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    if require_common_dimension and len({value["shared"].shape[1] for value in arrays.values()}) != 1:
        raise ValueError("Shared embedding dimensions differ across assays")
    return label, arrays, provenance


def _rename_oracle_count_nodes(oracle: Mapping[str, Any]) -> dict[str, Any]:
    renamed = dict(oracle)
    renamed["names"] = [name.removeprefix("count::") for name in oracle["names"]]
    if oracle["edges"] is not None:
        renamed["edges"] = [
            {**edge, "node1": edge["node1"].removeprefix("count::"), "node2": edge["node2"].removeprefix("count::")}
            for edge in oracle["edges"]
        ]
    return renamed


def evaluate_world(
    world: Path, embedding_directories: Sequence[Path], output: Path, *, regime: str,
    cohort: str | None = None, config: DiscoveryConfig | None = None,
    pool_seed: int = 2718, matching_seed: int = 3141,
    validation_manifest: Path | None = None, include_probes: bool = True,
    preprocessing_directories: Sequence[Path] = (),
) -> dict[str, Any]:
    """Evaluate a cached world without regenerating or fitting an integration."""
    config = config or DiscoveryConfig()
    if regime == "sparse":
        regime = "realistic"
    world, output = world.resolve(), output.resolve()
    truth_manifest_path = world / "truth" / "manifest.json"
    manifest = json.loads(truth_manifest_path.read_text())
    world_manifest_path = world / "manifest.json"
    world_manifest = json.loads(world_manifest_path.read_text())
    if world_manifest.get("status") != "complete":
        raise ValueError("Evaluation requires a completely generated frozen world")
    count_calibration = {
        "status": world_manifest.get("count_calibration_status", "unreported"),
        "focal_calibration_passed": world_manifest.get("focal_calibration_passed"),
        "panel_calibration_passed": world_manifest.get("panel_calibration_passed"),
        "panel_calibration_audits": world_manifest.get("panel_calibration_audits", {}),
        "focal_calibration_audits": manifest.get("calibration_audit", {}),
        "interpretation": "Execution can complete despite failed count-model calibration; this is not validated empirical realism",
    }
    candidates = manifest["candidates"]
    names = [row["name"] for row in candidates]
    simple = len(candidates) == 6
    cohort = cohort or ("train" if simple else "discovery")
    raw_truth = _load_npz(world / "truth" / f"{cohort}.npz")
    truth = validate_truth(raw_truth, regime)
    if list(map(str, truth["candidate_names"])) != names:
        raise ValueError("Candidate manifest and NPZ column order disagree")
    validation_manifest = validation_manifest or world / "truth" / f"observation_validation_{regime}.json"
    verification = json.loads(validation_manifest.read_text())
    if verification.get("status") != "ok" or verification["regime"] != regime:
        raise ValueError("Counts have no successful matching observation validation")
    verified = verification["cohorts"][cohort]
    if verified["truth_sha256"] != _sha256(world / "truth" / f"{cohort}.npz"):
        raise ValueError("Truth changed after observation validation")
    if verification["truth_manifest_sha256"] != _sha256(truth_manifest_path):
        raise ValueError("Truth manifest changed after observation validation")
    for entry in verified["assays"].values():
        if _sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Observed counts changed after validation")

    runs, provenance, failures = {}, {}, []
    for directory in embedding_directories:
        try:
            label, arrays, audit = _load_embedding_run(directory, cohort)
            if label in runs:
                raise ValueError(f"Duplicate embedding run label {label}")
            runs[label], provenance[label] = arrays, audit
        except Exception as error:
            failures.append({"directory": str(directory), "status": "failed",
                             "error_type": type(error).__name__, "error": str(error)})
    observed_ids = {assay: truth["cell_ids"][truth["assigned_assay"] == assay] for assay in ASSAYS}
    extra_inputs = {}
    if not simple:
        for name in ("assessment", "probe"):
            path = world / "truth" / f"{name}.npz"
            extra_inputs[str(path)] = _sha256(path) if path.exists() else "missing"
            for directory in (*embedding_directories, *preprocessing_directories):
                for assay in ASSAYS:
                    path = directory / name / f"{assay}.npz"
                    extra_inputs[str(path.resolve())] = _sha256(path) if path.exists() else "missing"
        for directory in preprocessing_directories:
            for filename in ("manifest.json", "embedding_manifest.json"):
                path = directory / filename
                extra_inputs[str(path.resolve())] = _sha256(path) if path.exists() else "missing"
    try:
        ci_provenance = backend_provenance(config)
    except Exception as error:
        ci_provenance = {"status": "unavailable", "error_type": type(error).__name__, "error": str(error)}
    fingerprint_inputs = {
        "world": str(world), "truth_sha256": _sha256(world / "truth" / f"{cohort}.npz"),
        "world_manifest_sha256": _sha256(world_manifest_path), "count_calibration": count_calibration,
        "truth_manifest_sha256": _sha256(truth_manifest_path), "regime": regime, "cohort": cohort,
        "provenance": provenance, "failures": list(failures), "configuration": asdict(config),
        "pool_seed": pool_seed, "matching_seed": matching_seed, "include_probes": include_probes,
        "evaluation_source_sha256": _sha256(Path(__file__)),
        "extra_inputs": extra_inputs, "backend_provenance": ci_provenance,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "preprocessing_directories": [str(path.resolve()) for path in preprocessing_directories],
    }
    fingerprint = hashlib.sha256(json.dumps(_jsonable(fingerprint_inputs), sort_keys=True).encode()).hexdigest()
    report_path = output / "report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("input_fingerprint") != fingerprint:
            raise FileExistsError("Existing evaluation used different inputs/configuration; choose a new output directory")
        return existing
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running", "input_fingerprint": fingerprint, "inputs": fingerprint_inputs,
        "count_calibration": count_calibration,
        "failed_integrations": failures, "candidates": candidates, "results": {},
        "additional_structure_metrics": optional_structure_metrics(candidates),
        "limitations": [
            "Matching returns an imposed different-cell coupling; nominal p-values are diagnostics, not formal FDR control.",
            "Exact direct pair/direction recovery is an identifiability stress test.",
            "Count-PAG truth differs from biological activity truth and includes latent assay exposure.",
            "Pearson/Spearman errors do not characterize all nonlinear dependence.",
            "Integration seeds are not independent biological-world replicates.",
        ],
    }
    _write_json_new(output / "input_manifest.json", {"input_fingerprint": fingerprint, "inputs": fingerprint_inputs})

    def cached_stage(label, calculate):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label) + "_" + hashlib.sha256(label.encode()).hexdigest()[:8]
        path = output / "stages" / f"{safe}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["input_fingerprint"] != fingerprint:
                raise ValueError(f"Stage cache input fingerprint differs: {label}")
            return record["result"]
        result = calculate()
        _write_json_new(path, {"input_fingerprint": fingerprint, "stage": label, "result": result})
        return result

    if not runs:
        report.update(status="failed", reason="No integration produced valid shared embeddings", runtime_seconds=time.monotonic() - started)
        _write_json_new(report_path, report)
        return report
    try:
        pools, pool_audit = select_common_pools(runs, observed_ids, seed=pool_seed)
    except Exception as error:
        report.update(status="failed", reason="Common-pool selection failed", error_type=type(error).__name__,
                      error=str(error), runtime_seconds=time.monotonic() - started)
        _write_json_new(report_path, report)
        return report
    report["common_pools"] = {**pool_audit, "cell_ids": pools}
    activity_method = "pc" if simple else "fci"
    biological_edges = manifest["biological_dag_edges"]
    activity_oracle = cached_stage("oracle_activity", lambda: oracle_graph(biological_edges, names, method=activity_method))
    count_edges, count_names = measurement_dag(biological_edges, candidates)
    count_oracle = cached_stage("oracle_counts", lambda: _rename_oracle_count_nodes(oracle_graph(count_edges, count_names, method="fci")))
    report["graphical_truth"] = {"activity_projection": activity_oracle, "count_measurement_pag": count_oracle}
    if simple:
        cores = [item["name"] for item in candidates if item["role"] == "core"]
        core_edges = [edge for edge in biological_edges if set(edge) <= set(cores)]
        report["graphical_truth"]["core_markov_equivalence_oracle"] = markov_equivalence_class_three_nodes(cores, core_edges)
    core_query = []
    if simple:
        core_query = [("H3K27ac_core", "RNA_core", ["H3K4me3_core"])]
    anchor_rows = _indices(truth["cell_ids"], pools["RNA"])
    paired_activities, paired_counts = truth["activities"][anchor_rows], truth["counts"][anchor_rows]

    def uncached_graph_results(activities, counts):
        activity = discover(activities, names, method=activity_method, scale="activities",
                            config=config, diagnostic_queries=core_query)
        count = discover(counts, names, method="fci", scale="counts", config=config, diagnostic_queries=core_query)
        return {
            "activities": activity, "counts": count,
            "activity_graph_metrics": score_graph(activity, activity_oracle, candidates,
                                                   target="activity_projection", biological_edges=biological_edges),
            "count_graph_metrics": score_graph(count, count_oracle, candidates,
                                                target="count_measurement_pag", biological_edges=biological_edges),
        }

    def graph_results(label, activities, counts):
        return cached_stage(f"discovery_{label}", lambda: uncached_graph_results(activities, counts))

    print(f"Evaluating {world.name}/{regime}: paired reference, {len(pools['RNA'])} common cells", flush=True)
    report["results"]["paired_anchor_reference"] = graph_results("paired_anchor_reference", paired_activities, paired_counts)
    _save_npz_new(output / "readouts" / "paired_anchor_reference.npz", counts=paired_counts,
                  activities=paired_activities, cell_ids=pools["RNA"], candidate_names=np.asarray(names))
    state_rep = {
        assay: {"cell_ids": observed_ids[assay], "shared": truth["state"][_indices(truth["cell_ids"], observed_ids[assay])]}
        for assay in ASSAYS
    }
    assignments = {
        "random_reference": cached_stage("assignment_random", lambda: match_one_to_one(pools, random_seed=matching_seed)),
        "true_state_reference": cached_stage("assignment_true_state", lambda: match_one_to_one(pools, state_rep)),
    }
    for label, arrays in runs.items():
        if label in assignments or label == "paired_anchor_reference":
            raise ValueError("Integration label conflicts with a reference name")
        try:
            assignments[label] = cached_stage(f"assignment_method_{label}", lambda arrays=arrays: match_one_to_one(pools, arrays))
        except Exception as error:
            failures.append({"label": label, "status": "failed", "stage": "assignment",
                             "error_type": type(error).__name__, "error": str(error)})
    for label, assignment in assignments.items():
        print(f"Evaluating {world.name}/{regime}: {label}", flush=True)
        readout = assemble_readout(assignment, truth)
        result = graph_results(label, readout["activities"], readout["counts"])
        result.update(
            assignment=assignment, state_matching=state_matching_metrics(readout),
            activity_dependence=dependence_metrics(readout["activities"], paired_activities, candidates, scale="activities"),
            count_dependence=dependence_metrics(np.log1p(readout["counts"]), np.log1p(paired_counts), candidates, scale="log1p_counts"),
            activity_score_interpretation="Agreement with source activity projection, not discovery of the original paired joint",
        )
        report["results"][label] = result
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label) + "_" + hashlib.sha256(label.encode()).hexdigest()[:8]
        _save_npz_new(output / "readouts" / f"{safe}.npz", counts=readout["counts"],
                      activities=readout["activities"], candidate_names=np.asarray(names),
                      **{f"cell_ids_{assay}": ids for assay, ids in assignment["cell_ids"].items()})

    if not simple:
        assessment_raw = _load_npz(world / "truth" / "assessment.npz")
        assessment = validate_truth(assessment_raw, regime)
        report["results"]["independent_paired_assessment"] = graph_results("independent_paired_assessment", assessment["activities"], assessment["counts"])
        state_names = [f"S{index}" for index in range(assessment["state"].shape[1])]
        all_names = state_names + names
        all_values = np.column_stack((assessment["state"], assessment["activities"]))
        report["results"]["independent_activities_with_state"] = cached_stage("independent_activities_with_state", lambda: {
            "discovery": discover(all_values, all_names, method="pc", scale="activities", config=config),
            "oracle": oracle_graph(biological_edges, all_names, method="pc"),
        })
        report["probes"] = {}
        report["preprocessing_probes"] = {}
        if include_probes:
            calibration_raw = _load_npz(world / "truth" / "probe.npz")
            for directory in embedding_directories:
                try:
                    label, cal, cal_audit = _load_embedding_run(directory, "probe")
                    other_label, test, test_audit = _load_embedding_run(directory, "assessment")
                    if label != other_label:
                        raise ValueError("Probe and assessment method labels differ")
                    print(f"Evaluator-only retention probes: {label}", flush=True)
                    report["probes"][label] = cached_stage(f"probes_{label}", lambda: evaluate_probes(cal, test, calibration_raw, assessment_raw, seed=config.seed))
                    report["probes"][label]["embedding_provenance"] = {"calibration": cal_audit, "assessment": test_audit}
                except Exception as error:
                    report["probes"][str(directory)] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            for directory in preprocessing_directories:
                try:
                    label, cal, cal_audit = _load_embedding_run(directory, "probe", require_common_dimension=False)
                    other_label, test, test_audit = _load_embedding_run(directory, "assessment", require_common_dimension=False)
                    if label != other_label:
                        raise ValueError("Preprocessing probe/assessment labels differ")
                    if label in report["preprocessing_probes"]:
                        raise ValueError("Repeated preprocessing probe label")
                    print(f"Evaluator-only preprocessing retention probes: {label}", flush=True)
                    result = cached_stage(f"preprocessing_probes_{label}",
                                          lambda: evaluate_probes(cal, test, calibration_raw, assessment_raw, seed=config.seed))
                    result["embedding_provenance"] = {"calibration": cal_audit, "assessment": test_audit}
                    result["interpretation"] = "Probe-only native representations; different assay coordinate systems are not used for matching"
                    report["preprocessing_probes"][label] = result
                except Exception as error:
                    report["preprocessing_probes"][str(directory)] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            if not preprocessing_directories:
                report["preprocessing_probes"] = {"status": "not_run", "reason": "No --preprocessing-embeddings supplied; extended stage missing"}
        else:
            report["probes"] = {"status": "not_run", "reason": "Explicitly disabled; extended evaluation incomplete"}
            report["preprocessing_probes"] = {"status": "not_run", "reason": "Probes explicitly disabled"}
    failed_diagnostics = [
        f"{label}/{scale}" for label, result in report["results"].items()
        for scale in ("activities", "counts") if scale in result and result[scale]["status"] != "ok"
    ]
    for label, oracle in report["graphical_truth"].items():
        if "status" in oracle and oracle["status"] != "ok":
            failed_diagnostics.append(f"graphical_truth/{label}")
    for label, result in report["results"].items():
        for key in ("discovery", "oracle"):
            if key in result and result[key]["status"] != "ok":
                failed_diagnostics.append(f"{label}/{key}")
        for key in ("activities", "counts"):
            for index, query in enumerate(result.get(key, {}).get("diagnostic_queries", [])):
                if query["status"] != "ok":
                    failed_diagnostics.append(f"{label}/{key}/diagnostic_query_{index}")
    if not simple:
        for collection in ("probes", "preprocessing_probes"):
            if report[collection].get("status") == "not_run":
                failed_diagnostics.append(f"{collection}/not_run")
            else:
                for label, result in report[collection].items():
                    if result["status"] != "ok":
                        failed_diagnostics.append(f"{collection}/{label}")
    report["failed_diagnostics"] = failed_diagnostics
    report["status"] = "completed_with_failures" if failures or failed_diagnostics else "completed"
    report["runtime_seconds"] = time.monotonic() - started
    _write_json_new(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-observed", help="Run in the AnnData environment before scoring")
    validate.add_argument("--world", type=Path, required=True)
    validate.add_argument("--regime", choices=("realistic", "informative", "sparse"), required=True)
    validate.add_argument("--cohorts", nargs="+")
    validate.add_argument("--output", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="Run in the causal-learn environment")
    evaluate.add_argument("--world", type=Path, required=True)
    evaluate.add_argument("--regime", choices=("realistic", "informative", "sparse"), required=True)
    evaluate.add_argument("--embeddings", type=Path, nargs="+", required=True)
    evaluate.add_argument("--preprocessing-embeddings", type=Path, nargs="*", default=[],
                          help="Native representation directories used only for held-out probes, never matching")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--cohort")
    evaluate.add_argument("--ci-seed", type=int, default=0)
    evaluate.add_argument("--ci-test", choices=("rcit", "kci", "fisherz"), default="rcit")
    evaluate.add_argument("--pool-seed", type=int, default=2718)
    evaluate.add_argument("--matching-seed", type=int, default=3141)
    evaluate.add_argument("--validation-manifest", type=Path)
    evaluate.add_argument("--skip-probes", action="store_true", help="Diagnostic partial run, not completed extended evaluation")
    args = parser.parse_args(argv)
    regime = "realistic" if args.regime == "sparse" else args.regime
    if args.command == "validate-observed":
        result = validate_observed_files(args.world, regime, cohorts=args.cohorts)
        destination = args.output or args.world / "truth" / f"observation_validation_{regime}.json"
        _write_json_new(destination, result)
        print(json.dumps({"status": result["status"], "output": str(destination)}))
        return 0
    report = evaluate_world(
        args.world, args.embeddings, args.output, regime=regime, cohort=args.cohort,
        config=DiscoveryConfig(seed=args.ci_seed, test=args.ci_test), pool_seed=args.pool_seed,
        matching_seed=args.matching_seed, validation_manifest=args.validation_manifest,
        include_probes=not args.skip_probes,
        preprocessing_directories=args.preprocessing_embeddings,
    )
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"),
                      "failed_integrations": len(report["failed_integrations"]),
                      "failed_diagnostics": report.get("failed_diagnostics", [])}))
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
