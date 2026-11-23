#!/usr/bin/env python3
"""Run a small PC/FCI/DAGMA graph-discovery job on a locus matrix."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.search.ConstraintBased.FCI import (
    fci,
    getPossibleDsep,
    get_color_edges,
    reorientAllWith,
    rule0,
    rule8,
    rule9,
    rule10,
    ruleR3,
    ruleR4B,
    ruleR5,
    ruleR6,
    ruleR7,
    rulesR1R2cycle,
)
from causallearn.utils.FAS import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.PCUtils import Meek, UCSepset
from causallearn.utils.cit import CIT, CIT_Base, FastKCI, register_ci_test
from scipy.stats import norm, rankdata
from tqdm.auto import tqdm

from _common import infer_repo_root, resolve_path, sanitize_token
from dagma_mixed_family import MixedFamilyDagmaLinear
from plot_pc_graph import plot_saved_graph


DAGMA_FAMILY_CONFIGS = {
    "gaussian_gaussian": {
        "peak_family": "gaussian",
        "gene_family": "gaussian",
        "expr_suffix": "_log1p",
        "required_peak_quant_mode": "log1p_norm",
        "description": "Gaussian peaks and Gaussian gene expression.",
        "exact": True,
    },
    "bernoulli_peak_nb_gene": {
        "peak_family": "bernoulli",
        "gene_family": "nb2",
        "expr_suffix": "_raw_counts",
        "required_peak_quant_mode": "raw_counts",
        "description": "Bernoulli peaks with logit link and NB2 gene counts with log link.",
        "exact": True,
    },
    "gaussian_nb": {
        "peak_family": "gaussian",
        "gene_family": "nb2",
        "expr_suffix": "_raw_counts",
        "required_peak_quant_mode": "log1p_norm",
        "description": "Gaussian peaks and NB2 gene counts.",
        "exact": True,
    },
    "nb_nb": {
        "peak_family": "nb2",
        "gene_family": "nb2",
        "expr_suffix": "_raw_counts",
        "required_peak_quant_mode": "raw_counts",
        "description": "NB2 peaks and NB2 gene counts.",
        "exact": True,
    },
}


class SafeFastKCI(CIT_Base):
    """FastKCI wrapper that falls back when causal-learn hits degenerate blocks."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        fallback: str = "rcit",
        K: int = 3,
        J: int = 4,
        fastkci_alpha: float = 500,
        use_gp: bool = False,
        cache_path: str | None = None,
        **kwargs,
    ):
        super().__init__(data, cache_path=cache_path)
        params = {
            "fallback": fallback,
            "K": int(K),
            "J": int(J),
            "fastkci_alpha": float(fastkci_alpha),
            "use_gp": bool(use_gp),
        }
        self.check_cache_method_consistent("safe_fastkci", json.dumps(params, sort_keys=True))
        self.assert_input_data_is_valid()
        self.fastkci = FastKCI(
            data,
            K=int(K),
            J=int(J),
            alpha=float(fastkci_alpha),
            use_gp=bool(use_gp),
        )
        self.fallback_method = str(fallback)
        self.fallback_test = CIT(data, method=self.fallback_method)
        self.last_resort_method = "fisherz"
        self.last_resort_test = None if self.fallback_method == self.last_resort_method else CIT(data, method=self.last_resort_method)
        self.n_fallbacks = 0
        self.n_last_resort_fallbacks = 0
        self.n_conservative_failures = 0
        self.fallback_errors: list[str] = []

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]

        try:
            p_value = float(self.fastkci(Xs[0], Ys[0], condition_set))
            if not np.isfinite(p_value):
                raise ValueError(f"FastKCI returned non-finite p-value: {p_value}")
        except Exception as exc:
            self.n_fallbacks += 1
            if len(self.fallback_errors) < 20:
                self.fallback_errors.append(f"{type(exc).__name__}: {exc}")
            try:
                p_value = float(self.fallback_test(Xs[0], Ys[0], condition_set))
                fallback_message = f"fell back to {self.fallback_method}"
            except Exception as fallback_exc:
                if len(self.fallback_errors) < 20:
                    self.fallback_errors.append(f"{type(fallback_exc).__name__}: {fallback_exc}")
                if self.last_resort_test is not None:
                    try:
                        p_value = float(self.last_resort_test(Xs[0], Ys[0], condition_set))
                        self.n_last_resort_fallbacks += 1
                        fallback_message = (
                            f"{self.fallback_method} also failed, then fell back to {self.last_resort_method}"
                        )
                    except Exception as last_resort_exc:
                        if len(self.fallback_errors) < 20:
                            self.fallback_errors.append(f"{type(last_resort_exc).__name__}: {last_resort_exc}")
                        p_value = 0.0
                        self.n_conservative_failures += 1
                        fallback_message = (
                            f"{self.fallback_method} and {self.last_resort_method} both failed; "
                            "using conservative p=0.0"
                        )
                else:
                    p_value = 0.0
                    self.n_conservative_failures += 1
                    fallback_message = f"{self.fallback_method} also failed; using conservative p=0.0"
            if self.n_fallbacks <= 20:
                warnings.warn(
                    f"FastKCI failed for one CI test and {fallback_message}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif self.n_fallbacks == 21:
                warnings.warn(
                    f"FastKCI fallback warning limit reached; suppressing further {self.fallback_method} fallback warnings.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.pvalue_cache[cache_key] = p_value
        return p_value


register_ci_test("safe_fastkci", SafeFastKCI)


class ZeroDelFisherZ(CIT_Base):
    """Test-wise-deletion Fisher-Z CI test (Causal Dropout Model, Dai et al., ICLR 2024).

    For a test X _||_ Y | Z, every row whose *conditioning* variables contain a
    dropout zero is deleted, then the ordinary Fisher-Z partial-correlation test is
    run on the surviving rows. Theorem 2 of the paper shows these test-wise-deletion
    CI relations asymptotically equal the dropout-free CI relations, so PC/FCI run on
    top of this test recover the underlying (dropout-free) structure.

    Adapted (self-contained, no external import) from the reference implementation
    ``csl/utils/cit.py::ZeroDel_FisherZ`` in the scRNA-Causal-Dropout repo.

    IMPORTANT — zero pattern vs. transformed data
    ------------------------------------------------
    ``data`` handed to the CI test by this script's discovery path is usually
    z-scored (``--standardize``, default on) and may be otherwise transformed, so a
    naive ``data != 0`` no longer marks dropout: centering moves the zeros off 0.
    To stay correct we accept an explicit ``nonzero_mask`` built from the *raw*
    (pre-transform, pre-standardize) feature matrix, where ``True`` marks an observed
    (non-dropout) entry. Row deletion uses that mask; the partial correlation is still
    computed on the discovery matrix. Because z-scoring is a per-column *linear* map,
    Pearson/partial correlations on any fixed subset of rows are unchanged, so using
    the z-scored values here is exact. (A *rank_gaussian* transform is nonlinear and
    does change correlations — main() warns against combining it with this test.)
    If no mask is supplied we fall back to ``data != 0``.
    """

    def __init__(self, data: np.ndarray, *, nonzero_mask=None, cache_path: str | None = None, **kwargs):
        super().__init__(data, cache_path=cache_path)
        if nonzero_mask is None:
            nonzero_boolean_mat = np.asarray(data, dtype=float) != 0
            mask_tag = "self_zero"
        else:
            nonzero_boolean_mat = np.asarray(nonzero_mask, dtype=bool)
            if nonzero_boolean_mat.shape != data.shape:
                raise ValueError(
                    "nonzero_mask shape "
                    f"{nonzero_boolean_mat.shape} does not match data shape {data.shape}."
                )
            mask_tag = "external_zero"
        self.check_cache_method_consistent(
            "zerodel_fisherz",
            json.dumps({"mask": mask_tag}, sort_keys=True),
        )
        self.assert_input_data_is_valid()
        self.nonzero_boolean_mat = nonzero_boolean_mat

    def _get_index_no_dropout_rows(self, var: list[int]) -> np.ndarray:
        return np.where(np.all(self.nonzero_boolean_mat[:, var], axis=1))[0]

    def __call__(self, X, Y, condition_set=None, return_rho=False, return_sample_size=False):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        var = Xs + Ys + condition_set
        if cache_key in self.pvalue_cache:
            res = self.pvalue_cache[cache_key]
            if len(res) == 2:
                res = (res[0], res[1], self.sample_size)
            if return_rho:
                return res if return_sample_size else res[:2]
            return (res[0], res[2]) if return_sample_size else res[0]

        # Test-wise deletion: drop rows with a dropout zero in any conditioning var.
        kept = self._get_index_no_dropout_rows(condition_set)
        n_kept = int(len(kept))
        if n_kept <= len(condition_set) + 3:
            warnings.warn(
                "ZeroDel_FisherZ: too few rows after test-wise deletion "
                f"(n={n_kept}, |Z|={len(condition_set)}); returning p=1 (independent).",
                RuntimeWarning,
                stacklevel=2,
            )
            res = (1 - np.finfo(float).eps, 0.0, n_kept)
            self.pvalue_cache[cache_key] = res[:2]
            if return_rho:
                return res if return_sample_size else res[:2]
            return (res[0], res[2]) if return_sample_size else res[0]

        sub = self.data[kept][:, var]
        with np.errstate(invalid="ignore", divide="ignore"):
            sub_corr = np.corrcoef(sub.T)
        # After deletion a kept variable can become (near-)constant on the surviving
        # rows -> corrcoef returns NaN/Inf and the SVD in pinv diverges. On the very
        # sparse peak data this happens often; treat it as an uninformative test (p=1)
        # instead of crashing the discovery worker.
        if not np.all(np.isfinite(sub_corr)):
            warnings.warn(
                "ZeroDel_FisherZ: degenerate correlation submatrix after test-wise deletion "
                f"(n={n_kept}, |Z|={len(condition_set)}); returning p=1 (independent).",
                RuntimeWarning,
                stacklevel=2,
            )
            res = (1 - np.finfo(float).eps, 0.0, n_kept)
            self.pvalue_cache[cache_key] = res[:2]
            if return_rho:
                return res if return_sample_size else res[:2]
            return (res[0], res[2]) if return_sample_size else res[0]
        inv = np.linalg.pinv(sub_corr)
        r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
        if abs(r) >= 1:
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        z = 0.5 * np.log((1 + r) / (1 - r))
        stat = np.sqrt(n_kept - len(condition_set) - 3) * abs(z)
        p = 2 * (1 - norm.cdf(abs(stat)))
        res = (float(p), float(r), n_kept)
        self.pvalue_cache[cache_key] = res[:2]
        if return_rho:
            return res if return_sample_size else res[:2]
        return (res[0], res[2]) if return_sample_size else res[0]


register_ci_test("zerodel_fisherz", ZeroDelFisherZ)


def resolve_ci_test(args, *, nonzero_mask=None) -> tuple[str, dict[str, object]]:
    if args.indep_test == "fastkci":
        return "safe_fastkci", {
            "K": int(args.fastkci_k),
            "J": int(args.fastkci_j),
            "fastkci_alpha": float(args.fastkci_alpha_param),
            "use_gp": bool(args.fastkci_use_gp),
            "fallback": str(args.fastkci_fallback),
        }
    if args.indep_test == "zerodel_fisherz":
        # Pass the raw dropout mask so deletion is correct even though the discovery
        # matrix is z-scored/transformed. See ZeroDelFisherZ docstring.
        return "zerodel_fisherz", ({} if nonzero_mask is None else {"nonzero_mask": nonzero_mask})
    return args.indep_test, {}


def infer_default_graph_out_dir(
    repo_root: Path,
    matrix_path: Path,
    method: str,
    run_dir_name: str,
    *,
    dagma_family_config: str | None = None,
) -> Path:
    outputs_root = repo_root / "pipelines/empirical/analysis" / "outputs"
    methods_root = outputs_root / "methods" / sanitize_token(method)
    try:
        rel = matrix_path.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        return matrix_path.parent / run_dir_name

    parts = rel.parts
    if len(parts) >= 5 and parts[0] == "datasets":
        run_id = parts[1]
        dataset_name = parts[2]
        gene_name = parts[3]
        dataset_id = sanitize_token(f"{run_id}__{dataset_name}")
        if sanitize_token(method) == "dagma" and dagma_family_config:
            family_id = sanitize_token(dagma_family_config)
            return methods_root / dataset_id / family_id / gene_name / run_dir_name
        return methods_root / dataset_id / gene_name / run_dir_name

    if len(parts) >= 4 and parts[0] in {"locus_matrices_matched", "locus_matrices"}:
        dataset_family = parts[0]
        run_id = parts[1]
        locus_name = parts[2]
        dataset_id = sanitize_token(f"{run_id}__{dataset_family}")
        if sanitize_token(method) == "dagma" and dagma_family_config:
            family_id = sanitize_token(dagma_family_config)
            return methods_root / dataset_id / family_id / locus_name / run_dir_name
        return methods_root / dataset_id / locus_name / run_dir_name

    return matrix_path.parent / run_dir_name


def auto_select_columns(df: pd.DataFrame) -> list[str]:
    expr_log1p = [col for col in df.columns if col.startswith("expr__") and col.endswith("_log1p")]
    if expr_log1p:
        expr_cols = expr_log1p
    else:
        expr_cols = [
            col
            for col in df.columns
            if col.startswith("expr__") and not col.endswith("_raw_counts")
        ]
    histone_cols = [col for col in df.columns if "__H3" in col and not col.startswith("libsize__")]
    return expr_cols + histone_cols


def peak_mark_from_column(node_name: str) -> str | None:
    if "__" not in node_name:
        return None
    suffix = node_name.rsplit("__", 1)[-1]
    return suffix if suffix.startswith("H3") else None


def infer_nearby_quant_mode(matrix_path: Path) -> str | None:
    summary_path = matrix_path.parent / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    value = summary.get("quant_mode")
    return str(value) if value is not None else None


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    centered = df - df.mean(axis=0)
    std = df.std(axis=0, ddof=1)
    std = std.replace(0.0, np.nan)
    scaled = centered.divide(std, axis=1)
    return scaled


def rank_gaussian_series(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    n = values.shape[0]
    ranks = rankdata(values, method="average")
    quantiles = (ranks - 0.5) / n
    quantiles = np.clip(quantiles, 1e-6, 1 - 1e-6)
    transformed = norm.ppf(quantiles)
    return pd.Series(transformed, index=series.index, name=series.name)


def apply_transform(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    if transform == "none":
        return df.copy()
    if transform == "rank_gaussian":
        return df.apply(rank_gaussian_series, axis=0)
    raise ValueError(f"Unsupported transform: {transform}")


def exact_pattern(node_name: str) -> str:
    return "^" + re.escape(node_name) + "$"


def node_role(node_name: str) -> str:
    if node_name.startswith("conf__"):
        return "confounder"
    if node_name.startswith("expr__"):
        return "expr"
    if "__" not in node_name or "__H3" not in node_name:
        return "other"

    region = node_name.split("__", 1)[0].lower()
    if "promoter" in region:
        return "promoter"
    if "enhancer" in region or "fire" in region or region.startswith("ure"):
        return "distal"
    return "other"


CONFOUNDER_MARKS = ["H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3"]


def build_confounder_columns(df: pd.DataFrame, mode: str) -> list[str]:
    """Add observed-confounder columns (``conf__*``) to ``df`` in place and return their names.

    ``mode`` (comma-separated): ``none`` -> []; ``depth`` -> per-metacell library depths
    (RNA + mean chromatin), the technical confounders that drive both peak signal and expression
    and are already present in the matrix as ``libsize__*`` columns. Other tokens (``cellcycle``,
    ``donor``) are honored only if the corresponding covariate columns already exist in the matrix.
    """
    tokens = {t.strip() for t in str(mode).split(",") if t.strip() and t.strip() != "none"}
    added: list[str] = []
    # 'depth'         -> RNA depth + ONE averaged chromatin depth (conf__chrom_depth).
    # 'depth_permark' -> RNA depth + SIX per-mark chromatin depth roots. Use this to test
    #                    whether same-mark bidirected enrichment is technical: the averaged
    #                    root leaves per-mark (per-assay) depth fluctuation uncontrolled, which
    #                    can induce a `<->` edge between two peaks on the same channel.
    #                    See pipelines/empirical/evaluation/results/same_mark_finding.md. Mutually exclusive with
    #                    'depth'; if both are given, per-mark wins (it is the superset).
    if "depth" in tokens or "depth_permark" in tokens:
        if "libsize__rna" in df.columns:
            df["conf__rna_depth"] = np.log1p(df["libsize__rna"].astype(float))
            added.append("conf__rna_depth")
        mark_cols = [f"libsize__{m}" for m in CONFOUNDER_MARKS if f"libsize__{m}" in df.columns]
        if "depth_permark" in tokens:
            for m in CONFOUNDER_MARKS:
                col = f"libsize__{m}"
                if col in df.columns:
                    df[f"conf__depth_{m}"] = np.log1p(df[col].astype(float))
                    added.append(f"conf__depth_{m}")
        elif mark_cols:
            df["conf__chrom_depth"] = np.log1p(df[mark_cols].astype(float).mean(axis=1))
            added.append("conf__chrom_depth")
    if "activity" in tokens and "activity__global" in df.columns:
        # per-metacell GLOBAL chromatin-activity latent (leave-one-locus-out,
        # build_global_activity.py): a common-cause proxy for the shared state that
        # may drive the bidirected peak<->peak edges. Standardized here.
        v = df["activity__global"].astype(float)
        sd = v.std(ddof=0) or 1.0
        df["conf__global_activity"] = (v - v.mean()) / sd
        added.append("conf__global_activity")
    for extra in ("cellcycle", "donor"):
        if extra in tokens:
            # honored if a precomputed covariate column is present (e.g. cc__s_score, donor__*)
            for col in [c for c in df.columns if c.startswith(f"{extra[:4]}__") or c.startswith(f"{extra}__")]:
                cc = f"conf__{col.split('__', 1)[-1]}"
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[cc] = df[col].astype(float)
                    added.append(cc)
    return added


def build_background_knowledge(selected_cols: list[str], mode: str) -> tuple[BackgroundKnowledge | None, list[dict[str, str]]]:
    conf_nodes = [col for col in selected_cols if node_role(col) == "confounder"]
    if mode == "none" and not conf_nodes:
        return None, []

    bk = BackgroundKnowledge()
    rules: list[dict[str, str]] = []

    expr_nodes = [col for col in selected_cols if node_role(col) == "expr"]
    promoter_nodes = [col for col in selected_cols if node_role(col) == "promoter"]
    distal_nodes = [col for col in selected_cols if node_role(col) == "distal"]
    histone_nodes = promoter_nodes + distal_nodes + [col for col in selected_cols if node_role(col) == "other" and "__H3" in col]

    if mode in {"minimal_expr_sink", "tiered_distal_promoter_expr"}:
        for expr in expr_nodes:
            for histone in histone_nodes:
                bk.add_forbidden_by_pattern(exact_pattern(expr), exact_pattern(histone))
                rules.append({"rule_type": "forbidden", "from": expr, "to": histone, "reason": "expression_not_parent_of_histone"})

    if mode == "tiered_distal_promoter_expr":
        for promoter in promoter_nodes:
            for distal in distal_nodes:
                bk.add_forbidden_by_pattern(exact_pattern(promoter), exact_pattern(distal))
                rules.append({"rule_type": "forbidden", "from": promoter, "to": distal, "reason": "promoter_not_parent_of_distal"})

    # Observed confounders are ROOTS: nothing (peak or gene) causes them. Forbidding every
    # edge INTO a confounder makes it a source node, so FCI conditions on it (de-confounding)
    # and it can only point OUT (conf -> peak, conf -> gene). Equivalent to placing the
    # confounders in the lowest causal tier.
    if conf_nodes:
        non_conf = [col for col in selected_cols if col not in set(conf_nodes)]
        for conf in conf_nodes:
            for other in non_conf:
                bk.add_forbidden_by_pattern(exact_pattern(other), exact_pattern(conf))
                rules.append({"rule_type": "forbidden", "from": other, "to": conf, "reason": "confounder_is_root"})

    return bk, rules


def edge_to_record(edge) -> dict[str, str]:
    return {
        "node1": edge.get_node1().get_name(),
        "endpoint1": edge.get_endpoint1().name,
        "node2": edge.get_node2().get_name(),
        "endpoint2": edge.get_endpoint2().name,
        "edge_text": str(edge),
    }


def graph_to_edge_records(graph) -> list[dict[str, str]]:
    return [edge_to_record(edge) for edge in graph.get_graph_edges()]


def build_dagma_exclude_edges(selected_cols: list[str], bg_rules: list[dict[str, str]]) -> list[tuple[int, int]]:
    col_to_idx = {col: idx for idx, col in enumerate(selected_cols)}
    excluded: list[tuple[int, int]] = []
    for rule in bg_rules:
        if rule.get("rule_type") != "forbidden":
            continue
        src = str(rule.get("from", ""))
        dst = str(rule.get("to", ""))
        if src in col_to_idx and dst in col_to_idx:
            excluded.append((col_to_idx[src], col_to_idx[dst]))
    return sorted(set(excluded))


def run_dagma_search(
    data: np.ndarray,
    *,
    families: list[str],
    offsets: np.ndarray,
    nb2_dispersions: list[float],
    lambda1: float,
    w_threshold: float,
    T: int,
    exclude_edges: list[tuple[int, int]] | None,
    verbose: bool,
) -> tuple[np.ndarray, MixedFamilyDagmaLinear]:
    model = MixedFamilyDagmaLinear(families=families, verbose=verbose, dtype=np.float64)
    adjacency = model.fit(
        data,
        offsets=offsets,
        lambda1=lambda1,
        w_threshold=w_threshold,
        T=T,
        exclude_edges=exclude_edges if exclude_edges else None,
        nb2_dispersions=nb2_dispersions,
    )
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"DAGMA returned an invalid adjacency matrix shape: {adjacency.shape}")
    forbidden_before_clamp = 0
    max_forbidden_abs_weight = 0.0
    if exclude_edges:
        exc = np.asarray(list(exclude_edges), dtype=int)
        forbidden_values = adjacency[exc[:, 0], exc[:, 1]]
        forbidden_before_clamp = int(np.count_nonzero(forbidden_values))
        if forbidden_values.size:
            max_forbidden_abs_weight = float(np.max(np.abs(forbidden_values)))
        adjacency[exc[:, 0], exc[:, 1]] = 0.0
    model.forbidden_edges_nonzero_before_clamp_ = forbidden_before_clamp
    model.forbidden_edges_max_abs_weight_before_clamp_ = max_forbidden_abs_weight
    return adjacency, model


def dagma_edge_records(adjacency: np.ndarray, selected_cols: list[str], weight_tol: float = 0.0) -> list[dict[str, str | float]]:
    edge_records: list[dict[str, str | float]] = []
    for source_idx, source in enumerate(selected_cols):
        for target_idx, target in enumerate(selected_cols):
            if source_idx == target_idx:
                continue
            weight = float(adjacency[source_idx, target_idx])
            if abs(weight) <= weight_tol:
                continue
            edge_records.append(
                {
                    "node1": source,
                    "endpoint1": "TAIL",
                    "node2": target,
                    "endpoint2": "ARROW",
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "abs_weight": abs(weight),
                    "edge_text": f"{source} -> {target} ({weight:.6g})",
                }
            )
    return edge_records


def infer_gene_base_name(expr_cols: list[str]) -> str | None:
    if not expr_cols:
        return None
    col = expr_cols[0]
    if col.endswith("_log1p"):
        return col[: -len("_log1p")]
    if col.endswith("_raw_counts"):
        return col[: -len("_raw_counts")]
    return col


def select_dagma_columns(
    df: pd.DataFrame,
    *,
    matrix_path: Path,
    family_config: str,
) -> tuple[list[str], dict[str, object]]:
    if family_config not in DAGMA_FAMILY_CONFIGS:
        raise ValueError(f"Unsupported DAGMA family config: {family_config}")
    config = DAGMA_FAMILY_CONFIGS[family_config]
    peak_cols = [col for col in df.columns if "__H3" in col and not col.startswith("libsize__")]
    expr_base_names = sorted({infer_gene_base_name([col]) for col in df.columns if col.startswith("expr__")})
    expr_base_names = [name for name in expr_base_names if name is not None]
    if len(expr_base_names) != 1:
        raise ValueError(f"Expected exactly one gene expression base name in matrix, found: {expr_base_names}")
    expr_col = f"{expr_base_names[0]}{config['expr_suffix']}"
    if expr_col not in df.columns:
        raise ValueError(f"Missing required expression column for DAGMA config '{family_config}': {expr_col}")
    observed_quant_mode = infer_nearby_quant_mode(matrix_path)
    required_quant_mode = str(config["required_peak_quant_mode"])
    if observed_quant_mode is not None and observed_quant_mode != required_quant_mode:
        raise ValueError(
            f"DAGMA family config '{family_config}' requires peak quant_mode '{required_quant_mode}', "
            f"but matrix metadata reported '{observed_quant_mode}'."
        )
    selected_cols = [expr_col] + peak_cols
    meta = {
        "family_config": family_config,
        "description": str(config["description"]),
        "required_peak_quant_mode": required_quant_mode,
        "observed_peak_quant_mode": observed_quant_mode,
        "expression_column": expr_col,
        "peak_columns": peak_cols,
    }
    return selected_cols, meta


def prepare_dagma_inputs(
    selected_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    selected_cols: list[str],
    family_config: str,
    transform: str,
    standardize: bool,
    peak_binary_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, object]]:
    config = DAGMA_FAMILY_CONFIGS[family_config]
    transformed_df = selected_df.copy()
    model_df = selected_df.copy()
    offsets = np.zeros((selected_df.shape[0], len(selected_cols)), dtype=float)
    node_family_map: dict[str, str] = {}
    offset_columns: dict[str, str | None] = {}
    nb2_dispersions: list[float] = []

    for idx, col in enumerate(selected_cols):
        if col.startswith("expr__"):
            family = str(config["gene_family"])
            offset_col = "libsize__rna" if family == "nb2" else None
        else:
            family = str(config["peak_family"])
            mark = peak_mark_from_column(col)
            offset_col = f"libsize__{mark}" if family == "nb2" and mark is not None else None
        node_family_map[col] = family
        offset_columns[col] = offset_col

        series = selected_df[col].astype(float)
        transformed_series = series.copy()
        model_series = series.copy()

        if family == "gaussian":
            if transform == "rank_gaussian":
                transformed_series = rank_gaussian_series(series)
            else:
                transformed_series = series.copy()
            model_series = transformed_series.copy()
            if standardize:
                std = float(model_series.std(ddof=1))
                if std > 0:
                    model_series = (model_series - float(model_series.mean())) / std
                else:
                    model_series = model_series - float(model_series.mean())
            nb2_dispersions.append(np.nan)
        elif family == "bernoulli":
            transformed_series = (series > peak_binary_threshold).astype(float)
            model_series = transformed_series.copy()
            nb2_dispersions.append(np.nan)
        elif family == "nb2":
            transformed_series = series.copy()
            model_series = series.copy()
            mean = float(series.mean())
            var = float(series.var(ddof=1)) if series.shape[0] > 1 else mean
            if mean <= 0 or var <= mean:
                theta = 1e6
            else:
                theta = max((mean * mean) / max(var - mean, 1e-6), 1e-6)
            nb2_dispersions.append(float(theta))
            if offset_col is not None and offset_col in full_df.columns:
                offset_values = np.log(np.clip(full_df.loc[selected_df.index, offset_col].astype(float).to_numpy(), 1.0, None))
                offsets[:, idx] = offset_values
        else:
            raise ValueError(f"Unsupported DAGMA family: {family}")

        transformed_df[col] = transformed_series
        model_df[col] = model_series

    meta = {
        "family_config": family_config,
        "description": str(config["description"]),
        "node_family_map": node_family_map,
        "offset_columns": offset_columns,
        "nb_parameterization": "nb2",
        "nb2_dispersions": {col: float(theta) for col, theta in zip(selected_cols, nb2_dispersions) if not np.isnan(theta)},
        "peak_binary_threshold": float(peak_binary_threshold),
        "standardized_gaussian_columns": bool(standardize),
        "transform_applied_to_gaussian_columns": transform,
    }
    return selected_df.copy(), transformed_df, model_df, offsets, meta


def default_out_dir_name(args) -> str:
    if args.method == "dagma":
        return (
            f"dagma_{sanitize_token(args.dagma_family_config)}"
            f"_lambda1_{sanitize_token(str(args.dagma_lambda1))}"
            f"_wthr_{sanitize_token(str(args.dagma_w_threshold))}"
            f"_T_{sanitize_token(str(args.dagma_T))}"
            + ("" if args.transform == "none" else f"_xform_{sanitize_token(args.transform)}")
            + ("" if args.background_mode == "none" else f"_bg_{sanitize_token(args.background_mode)}")
        )
    return (
        f"{sanitize_token(args.method)}_{sanitize_token(args.indep_test)}_alpha_{sanitize_token(str(args.alpha))}"
        + ("" if args.max_depth < 0 else f"_depth_{args.max_depth}")
        + ("" if args.transform == "none" else f"_xform_{sanitize_token(args.transform)}")
        + ("" if args.background_mode == "none" else f"_bg_{sanitize_token(args.background_mode)}")
    )


def skeleton_discovery_limited(
    data: np.ndarray,
    alpha: float,
    indep_test,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: list[str] | None = None,
    max_depth: int = -1,
) -> CausalGraph:
    """Repo-local PC skeleton search with an optional maximum conditioning depth."""

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        if max_depth >= 0 and depth > max_depth:
            break

        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
                pbar.set_description(f"Depth={depth}, working on node {x}")
            neigh_x = cg.neighbors(x)
            if len(neigh_x) < depth - 1:
                continue
            for y in neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                    background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])
                ):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    edge_removal.append((x, y))
                    edge_removal.append((y, x))

                neigh_x_noy = np.delete(neigh_x, np.where(neigh_x == y))
                for cond_set in combinations(neigh_x_noy, depth):
                    p_val = cg.ci_test(x, y, cond_set)
                    if p_val > alpha:
                        if verbose:
                            print(f"{x} ind {y} | {cond_set} with p-value {p_val:f}\n")
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, cond_set)
                            append_value(cg.sepset, y, x, cond_set)
                            break
                        edge_removal.append((x, y))
                        edge_removal.append((y, x))
                        for s in cond_set:
                            sepsets.add(s)
                    elif verbose:
                        print(f"{x} dep {y} | {cond_set} with p-value {p_val:f}\n")
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg


def run_pc_search(
    data: np.ndarray,
    alpha: float,
    indep_test: str,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: list[str] | None = None,
    max_depth: int = -1,
    uc_rule: int = 0,
    uc_priority: int = 2,
    **kwargs,
) -> CausalGraph:
    """Local PC wrapper that optionally constrains conditioning-set depth."""

    indep_test_obj = CIT(data, indep_test, **kwargs)
    cg_1 = skeleton_discovery_limited(
        data,
        alpha,
        indep_test_obj,
        stable=stable,
        background_knowledge=background_knowledge,
        verbose=verbose,
        show_progress=show_progress,
        node_names=node_names,
        max_depth=max_depth,
    )

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        return Meek.meek(cg_2, background_knowledge=background_knowledge)
    if uc_rule == 1:
        cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        return Meek.meek(cg_2, background_knowledge=background_knowledge)
    if uc_rule == 2:
        cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        return Meek.meek(cg_before, background_knowledge=background_knowledge)
    raise ValueError("uc_rule should be in [0, 1, 2]")


def run_fci_search(
    data: np.ndarray,
    alpha: float,
    indep_test: str,
    background_knowledge: BackgroundKnowledge | None = None,
    show_progress: bool = True,
    node_names: list[str] | None = None,
    max_depth: int = -1,
    possible_dsep_depth: int = -1,
    max_path_length: int = -1,
    **kwargs,
):
    if possible_dsep_depth < 0 and max_path_length < 0:
        graph, edges = fci(
            data,
            independence_test_method=indep_test,
            alpha=alpha,
            depth=max_depth,
            background_knowledge=background_knowledge,
            show_progress=show_progress,
            node_names=node_names,
            **kwargs,
        )
        return graph, edges
    return run_fci_search_bounded(
        data,
        alpha=alpha,
        indep_test=indep_test,
        background_knowledge=background_knowledge,
        show_progress=show_progress,
        node_names=node_names,
        max_depth=max_depth,
        possible_dsep_depth=possible_dsep_depth,
        max_path_length=max_path_length,
        **kwargs,
    )


def remove_by_possible_dsep_limited(
    graph,
    independence_test,
    alpha: float,
    sep_sets: dict[tuple[int, int], set[int]],
    *,
    max_conditioning_depth: int,
) -> int:
    """Bounded version of causal-learn FCI Possible-D-Sep pruning.

    causal-learn's default `removeByPossibleDsep` enumerates all subsets of
    each Possible-D-Sep set. That is combinatorial on dense local peak graphs.
    This repo-local version caps the conditioning-set size.
    """

    if max_conditioning_depth < 2:
        return 0

    def _contains_all(set_a, set_b) -> bool:
        return all(node_b in set_a for node_b in set_b)

    removed = 0
    for edge in list(graph.get_graph_edges()):
        if not graph.contains_edge(edge):
            continue
        node_a = edge.get_node1()
        node_b = edge.get_node2()

        for left_node, right_node in [(node_a, node_b), (node_b, node_a)]:
            if not graph.contains_edge(edge):
                break
            possible_dsep = getPossibleDsep(left_node, right_node, graph, max_conditioning_depth)
            max_size = min(max_conditioning_depth, len(possible_dsep))
            for size in range(2, max_size + 1):
                found_independence = False
                for choice in combinations(range(len(possible_dsep)), size):
                    sepset_nodes = tuple(possible_dsep[index] for index in choice)
                    if _contains_all(set(graph.get_adjacent_nodes(node_a)), set(sepset_nodes)):
                        continue
                    if _contains_all(set(graph.get_adjacent_nodes(node_b)), set(sepset_nodes)):
                        continue
                    x_idx = graph.get_node_map()[left_node]
                    y_idx = graph.get_node_map()[right_node]
                    cond_set = tuple(graph.get_node_map()[possible_dsep[index]] for index in choice)
                    p_value = independence_test(x_idx, y_idx, cond_set)
                    if p_value > alpha:
                        graph.remove_edge(edge)
                        sep_sets[(x_idx, y_idx)] = set(cond_set)
                        removed += 1
                        found_independence = True
                        break
                if found_independence:
                    break
    return removed


def run_fci_search_bounded(
    data: np.ndarray,
    alpha: float,
    indep_test: str,
    background_knowledge: BackgroundKnowledge | None = None,
    show_progress: bool = True,
    node_names: list[str] | None = None,
    max_depth: int = -1,
    possible_dsep_depth: int = -1,
    max_path_length: int = -1,
    verbose: bool = False,
    **kwargs,
):
    independence_test = CIT(data, method=indep_test, **kwargs)
    if node_names is None:
        node_names = [f"X{i + 1}" for i in range(data.shape[1])]
    nodes = []
    for idx, name in enumerate(node_names):
        node = GraphNode(str(name))
        node.add_attribute("id", idx)
        nodes.append(node)

    graph, sep_sets, _ = fas(
        data,
        nodes,
        independence_test_method=independence_test,
        alpha=alpha,
        knowledge=background_knowledge,
        depth=max_depth,
        verbose=verbose,
        show_progress=show_progress,
    )

    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)
    remove_by_possible_dsep_limited(
        graph,
        independence_test,
        alpha,
        sep_sets,
        max_conditioning_depth=possible_dsep_depth,
    )
    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    change_flag = True
    first_time = True
    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)
        if change_flag or (
            first_time
            and background_knowledge is not None
            and len(background_knowledge.forbidden_rules_specs) > 0
            and len(background_knowledge.required_rules_specs) > 0
            and len(background_knowledge.tier_map.keys()) > 0
        ):
            change_flag = ruleR4B(
                graph,
                max_path_length,
                data,
                independence_test,
                alpha,
                sep_sets,
                change_flag,
                background_knowledge,
                verbose,
            )
            first_time = False
        change_flag = ruleR5(graph, change_flag, verbose)
        change_flag = ruleR6(graph, change_flag, verbose)
        change_flag = ruleR7(graph, change_flag, verbose)
        change_flag = rule8(graph, nodes, change_flag)
        change_flag = rule9(graph, nodes, change_flag)
        change_flag = rule10(graph, change_flag)

    graph.set_pag(True)
    return graph, get_color_edges(graph)


def drop_collinear_columns(selected_df, cols, transform, standardize, threshold: float = 0.9999):
    """Greedily drop feature columns that are near-perfectly collinear (|r| >= threshold)
    with an already-kept column, on the SAME matrix discovery will see (post transform +
    z-score). Perfectly collinear sparse peaks make the fisherz correlation matrix singular
    ("pipelines/empirical/data correlation matrix is singular") -- common at small n with 70-90%-zero peaks.
    Confounder and expression nodes are prioritized so a peak collinear with them is dropped,
    not the confounder/gene. Returns the columns to keep, in the original order."""
    tdf = apply_transform(selected_df[cols], transform)
    mdf = zscore_df(tdf) if standardize else tdf.copy()
    X = mdf.to_numpy(dtype=float)
    _keep_first = lambda c: c.startswith("conf__") or c.startswith("expr__")
    order = ([i for i, c in enumerate(cols) if _keep_first(c)] +
             [i for i, c in enumerate(cols) if not _keep_first(c)])
    kept: list[int] = []
    for i in order:
        xi = X[:, i]
        if not np.isfinite(xi).all() or xi.std() <= 0:
            continue
        collinear = False
        for j in kept:
            r = np.corrcoef(xi, X[:, j])[0, 1]
            if (not np.isfinite(r)) or abs(r) >= threshold:
                collinear = True
                break
        if not collinear:
            kept.append(i)
    keep_set = set(kept)
    return [c for i, c in enumerate(cols) if i in keep_set]


def main() -> int:
    p = argparse.ArgumentParser(description="Run a PC or FCI baseline on a locus matrix with causal-learn")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--matrix-tsv", required=True)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: pipelines/empirical/analysis/outputs/methods/<method>/<dataset>/<gene-or-locus>/<result-dir> when the matrix lives under the project outputs tree.",
    )
    p.add_argument("--method", default="pc", choices=["pc", "fci", "dagma"])
    p.add_argument("--columns", default="auto", help="Comma-separated column list or 'auto'")
    p.add_argument("--alpha", type=float, default=0.05)
    # Default = rank-fisherz (this flag + --transform rank_gaussian): fast, calibrated, and
    # captures monotone nonlinearity. 'rcit' is the nonlinearity guard (RCoT; randomized
    # Fourier-feature approximation to KCI). KCI itself is opt-in only — slow and brittle at
    # our n, and it returned empty graphs in the run audit. See project_resources/research/DECISIONS.md.
    p.add_argument("--indep-test", default="fisherz",
                   help="fisherz (with --transform rank_gaussian = rank-fisherz, the default) | "
                        "rcit (RCoT nonlinearity guard) | kci (opt-in, slow) | fastkci | "
                        "zerodel_fisherz (Causal Dropout, niche)")
    p.add_argument("--fastkci-k", type=int, default=3, help="Safe FastKCI mixture components; lower values reduce degenerate block failures.")
    p.add_argument("--fastkci-j", type=int, default=4, help="Safe FastKCI random partition repetitions.")
    p.add_argument("--fastkci-alpha-param", type=float, default=500.0, help="Safe FastKCI Dirichlet alpha parameter.")
    p.add_argument("--fastkci-use-gp", action="store_true", default=False, help="Use GP width estimation inside FastKCI conditional tests.")
    p.add_argument("--fastkci-fallback", default="rcit", choices=["rcit", "fisherz", "kci"], help="Fallback CI test if one FastKCI test fails.")
    p.add_argument("--stable", action="store_true", default=True)
    p.add_argument("--unstable", action="store_false", dest="stable")
    # rank_gaussian = van der Waerden / nonparanormal transform. With --indep-test fisherz this
    # IS rank-fisherz. Use --transform none for zerodel_fisherz (see the guard in main()).
    p.add_argument("--transform", default="rank_gaussian", choices=["none", "rank_gaussian"])
    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no-standardize", action="store_false", dest="standardize")
    p.add_argument("--background-mode", default="none", choices=["none", "minimal_expr_sink", "tiered_distal_promoter_expr"])
    p.add_argument(
        "--confounders",
        default="none",
        help="Comma-separated observed confounders to add as ROOT nodes (pc/fci only): "
             "'none' | 'depth' (per-metacell library depths from libsize__*) | 'depth,cellcycle' "
             "(cellcycle honored only if that covariate column exists). NB: donor is UNAVAILABLE (no shared donor structure across modalities; see project_resources/research/LIMITATIONS.md L1).",
    )
    p.add_argument("--min-unique", type=int, default=2)
    p.add_argument("--min-std", type=float, default=1e-8)
    p.add_argument("--max-depth", type=int, default=-1, help="Maximum conditioning-set size for PC skeleton search; -1 means no cap")
    p.add_argument(
        "--fci-possible-dsep-depth",
        type=int,
        default=-2,
        help="Maximum conditioning-set size for FCI Possible-D-Sep pruning. -2 means reuse --max-depth; -1 means causal-learn default unlimited.",
    )
    p.add_argument(
        "--fci-max-path-length",
        type=int,
        default=-2,
        help="Maximum discriminating-path length for FCI orientation. -2 means reuse --max-depth; -1 means causal-learn default unlimited.",
    )
    p.add_argument("--dropna-rows", action="store_true", default=True)
    p.add_argument("--keep-na-rows", action="store_false", dest="dropna_rows")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", action="store_false", dest="plot")
    p.add_argument("--plot-layout", default="auto", choices=["auto", "local", "spring"])
    p.add_argument("--plot-max-edges", type=int, default=30)
    p.add_argument("--plot-peaks-per-row", type=int, default=18)
    p.add_argument("--plot-layout-seed", type=int, default=13)
    p.add_argument("--plot-title", default=None)
    p.add_argument("--plot-out-prefix", default=None)
    p.add_argument("--dagma-loss-type", default="l2", choices=["l2", "logistic"])
    p.add_argument(
        "--dagma-family-config",
        "--dagma-family-preset",
        dest="dagma_family_config",
        default="gaussian_gaussian",
        choices=list(DAGMA_FAMILY_CONFIGS.keys()),
        help="Node-family DAGMA configuration for peaks and gene expression.",
    )
    p.add_argument("--dagma-lambda1", type=float, default=0.03)
    p.add_argument("--dagma-w-threshold", type=float, default=0.3)
    p.add_argument("--dagma-T", type=int, default=5)
    p.add_argument("--dagma-peak-binary-threshold", type=float, default=0.0)
    p.add_argument("--dagma-verbose", action="store_true", default=False)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = resolve_path(repo_root, args.matrix_tsv)
    if matrix_path is None or not matrix_path.exists():
        raise FileNotFoundError(f"Missing matrix TSV: {args.matrix_tsv}")

    df = pd.read_csv(matrix_path, sep="\t")
    dagma_config_meta: dict[str, object] | None = None
    if args.method == "dagma" and args.columns == "auto":
        requested_cols, dagma_config_meta = select_dagma_columns(
            df,
            matrix_path=matrix_path,
            family_config=args.dagma_family_config,
        )
    elif args.columns == "auto":
        requested_cols = auto_select_columns(df)
    else:
        requested_cols = [c.strip() for c in args.columns.split(",") if c.strip()]
        if args.method == "dagma":
            dagma_config_meta = {
                "family_config": args.dagma_family_config,
                "description": str(DAGMA_FAMILY_CONFIGS[args.dagma_family_config]["description"]),
                "required_peak_quant_mode": str(DAGMA_FAMILY_CONFIGS[args.dagma_family_config]["required_peak_quant_mode"]),
                "observed_peak_quant_mode": infer_nearby_quant_mode(matrix_path),
                "expression_column": next((col for col in requested_cols if col.startswith("expr__")), None),
                "peak_columns": [col for col in requested_cols if "__H3" in col and not col.startswith("libsize__")],
            }

    # Observed confounders (root nodes) — pc/fci only. Built from the matrix's libsize columns
    # and prepended so they enter discovery; build_background_knowledge makes them roots.
    confounder_cols: list[str] = []
    if args.method in {"pc", "fci"} and args.confounders != "none":
        confounder_cols = build_confounder_columns(df, args.confounders)
        requested_cols = confounder_cols + [c for c in requested_cols if c not in confounder_cols]

    missing_cols = [c for c in requested_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Requested columns not found: {missing_cols}")
    if not requested_cols:
        raise ValueError("No candidate columns selected for PC")

    work_df = df[requested_cols].copy()

    dropped_rows = []
    for col in requested_cols:
        series = work_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            dropped_rows.append({"column": col, "reason": "non_numeric"})
            continue
        if series.nunique(dropna=True) < args.min_unique:
            dropped_rows.append({"column": col, "reason": f"nunique_lt_{args.min_unique}"})
            continue
        if float(series.std(ddof=1)) <= args.min_std:
            dropped_rows.append({"column": col, "reason": f"std_le_{args.min_std}"})
            continue

    dropped_cols = {row["column"] for row in dropped_rows}
    selected_cols = [c for c in requested_cols if c not in dropped_cols]
    if len(selected_cols) < 2:
        raise ValueError("Fewer than two columns remain after filtering")

    selected_df = work_df[selected_cols].copy()
    n_rows_before_dropna = int(selected_df.shape[0])
    if args.dropna_rows:
        na_mask = selected_df.isna().any(axis=1)
        selected_df = selected_df.loc[~na_mask].copy()
    n_rows_after_dropna = int(selected_df.shape[0])
    if selected_df.empty:
        raise ValueError("No rows remain after dropping missing values")

    # Constraint-based methods invert the correlation matrix (fisherz); perfectly collinear
    # sparse peaks make it singular. Drop the redundant columns (keep expression + one of
    # each collinear group) so FCI/PC run on well-conditioned data.
    if args.method in ("pc", "fci"):
        keep_cols = drop_collinear_columns(selected_df, selected_cols, args.transform, args.standardize)
        if len(keep_cols) < len(selected_cols):
            for col in selected_cols:
                if col not in keep_cols:
                    dropped_rows.append({"column": col, "reason": "collinear"})
            selected_cols = keep_cols
            selected_df = selected_df[selected_cols].copy()
            if len(selected_cols) < 2:
                raise ValueError("Fewer than two columns remain after collinearity filtering")

    dagma_family = None
    resolved_dagma_loss = "mixed_family"
    dagma_offsets = np.zeros((selected_df.shape[0], len(selected_cols)), dtype=float)
    dagma_nb2_dispersions = [np.nan] * len(selected_cols)
    fci_possible_dsep_depth = args.max_depth if args.fci_possible_dsep_depth == -2 else args.fci_possible_dsep_depth
    fci_max_path_length = args.max_depth if args.fci_max_path_length == -2 else args.fci_max_path_length
    if args.method == "dagma":
        selected_df, transformed_df, model_df, dagma_offsets, dagma_family = prepare_dagma_inputs(
            selected_df,
            df,
            selected_cols=selected_cols,
            family_config=args.dagma_family_config,
            transform=args.transform,
            standardize=args.standardize,
            peak_binary_threshold=args.dagma_peak_binary_threshold,
        )
        dagma_nb2_dispersions = [
            float(dagma_family["nb2_dispersions"].get(col, np.nan))
            for col in selected_cols
        ]
    else:
        transformed_df = apply_transform(selected_df, args.transform)
        model_df = zscore_df(transformed_df) if args.standardize else transformed_df.copy()
        model_df = model_df.astype(float)

    feature_stats = []
    for col in selected_cols:
        series = selected_df[col]
        feature_stats.append(
            {
                "column": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "n_unique": int(series.nunique(dropna=True)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    model_df = model_df.astype(float)
    background_knowledge, bg_rules = build_background_knowledge(selected_cols, args.background_mode)
    dagma_exclude_edges = build_dagma_exclude_edges(selected_cols, bg_rules) if args.method == "dagma" else []

    # For the Causal Dropout Model test-wise-deletion test we need the *raw* dropout
    # pattern (before transform/standardize move zeros off 0). selected_df here is the
    # untransformed, row-filtered feature matrix in the same row/column order as model_df.
    zerodel_nonzero_mask = None
    if args.indep_test == "zerodel_fisherz" and args.method in {"pc", "fci"}:
        zerodel_nonzero_mask = selected_df[selected_cols].to_numpy(dtype=float) != 0
        zero_frac = float(1.0 - zerodel_nonzero_mask.mean())
        if args.transform == "rank_gaussian":
            warnings.warn(
                "--indep-test zerodel_fisherz with --transform rank_gaussian: the rank-Gaussian "
                "transform is nonlinear and destroys the count-magnitude semantics the dropout "
                "theorem relies on. Row deletion still uses the raw zero pattern, but the partial "
                "correlation is computed on transformed values. Use --transform none for a valid test.",
                RuntimeWarning,
            )
        if zero_frac < 1e-6:
            warnings.warn(
                f"--indep-test zerodel_fisherz: raw data has essentially no zeros (zero fraction "
                f"{zero_frac:.2e}); test-wise deletion is a no-op and this reduces to plain Fisher-Z. "
                "The Causal Dropout Model is only informative on data with genuine dropout zeros.",
                RuntimeWarning,
            )
    resolved_indep_test, ci_kwargs = resolve_ci_test(args, nonzero_mask=zerodel_nonzero_mask)

    if args.method == "pc":
        graph = run_pc_search(
            model_df.to_numpy(),
            alpha=args.alpha,
            indep_test=resolved_indep_test,
            stable=args.stable,
            show_progress=False,
            node_names=selected_cols,
            background_knowledge=background_knowledge,
            max_depth=args.max_depth,
            **ci_kwargs,
        ).G
    elif args.method == "fci":
        graph, _ = run_fci_search(
            model_df.to_numpy(),
            alpha=args.alpha,
            indep_test=resolved_indep_test,
            show_progress=False,
            node_names=selected_cols,
            background_knowledge=background_knowledge,
            max_depth=args.max_depth,
            possible_dsep_depth=fci_possible_dsep_depth,
            max_path_length=fci_max_path_length,
            **ci_kwargs,
        )
    elif args.method == "dagma":
        graph = None
        dagma_adjacency, dagma_model = run_dagma_search(
            model_df.to_numpy(),
            families=[str(dagma_family["node_family_map"][col]) for col in selected_cols],
            offsets=dagma_offsets,
            nb2_dispersions=dagma_nb2_dispersions,
            lambda1=args.dagma_lambda1,
            w_threshold=args.dagma_w_threshold,
            T=args.dagma_T,
            exclude_edges=dagma_exclude_edges,
            verbose=args.dagma_verbose,
        )
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else infer_default_graph_out_dir(
            repo_root,
            matrix_path,
            args.method,
            default_out_dir_name(args),
            dagma_family_config=(args.dagma_family_config if args.method == "dagma" else None),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "dagma":
        edge_records = dagma_edge_records(dagma_adjacency, selected_cols)
        edges_df = pd.DataFrame(
            edge_records,
            columns=["node1", "endpoint1", "node2", "endpoint2", "source", "target", "weight", "abs_weight", "edge_text"],
        )
        adjacency_df = pd.DataFrame(dagma_adjacency, index=selected_cols, columns=selected_cols)
    else:
        edge_records = graph_to_edge_records(graph)
        edges_df = pd.DataFrame(edge_records, columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"])
        adjacency_df = pd.DataFrame(graph.graph, index=selected_cols, columns=selected_cols)
    dropped_df = pd.DataFrame(dropped_rows, columns=["column", "reason"])
    stats_df = pd.DataFrame(feature_stats)
    bg_rules_df = pd.DataFrame(bg_rules, columns=["rule_type", "from", "to", "reason"])

    id_col = None
    for candidate in ["metacell_id", "sample_id", "row_id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        id_series = pd.Series([f"row_{i + 1:05d}" for i in range(df.shape[0])], name="row_id")
        id_col = "row_id"
        df = pd.concat([id_series, df], axis=1)

    selected_with_ids = df.loc[selected_df.index, [id_col]].reset_index(drop=True).join(selected_df.reset_index(drop=True))
    transformed_with_ids = df.loc[transformed_df.index, [id_col]].reset_index(drop=True).join(transformed_df.reset_index(drop=True))
    model_with_ids = df.loc[model_df.index, [id_col]].reset_index(drop=True).join(model_df.reset_index(drop=True))

    selected_path = out_dir / "selected_matrix.tsv"
    transformed_path = out_dir / "selected_matrix_transformed.tsv"
    model_path = out_dir / "selected_matrix_standardized.tsv"
    dropped_path = out_dir / "dropped_columns.tsv"
    stats_path = out_dir / "feature_stats.tsv"
    edges_path = out_dir / f"{args.method}_edges.tsv"
    adjacency_path = out_dir / f"{args.method}_adjacency.tsv"
    bg_rules_path = out_dir / "background_knowledge_rules.tsv"
    summary_path = out_dir / "run_summary.json"
    plot_png_path = None
    plot_svg_path = None

    selected_with_ids.to_csv(selected_path, sep="\t", index=False)
    transformed_with_ids.to_csv(transformed_path, sep="\t", index=False)
    model_with_ids.to_csv(model_path, sep="\t", index=False)
    dropped_df.to_csv(dropped_path, sep="\t", index=False)
    stats_df.to_csv(stats_path, sep="\t", index=False)
    edges_df.to_csv(edges_path, sep="\t", index=False)
    adjacency_df.to_csv(adjacency_path, sep="\t")
    bg_rules_df.to_csv(bg_rules_path, sep="\t", index=False)

    if args.plot:
        plot_max_edges = None if args.plot_max_edges <= 0 else args.plot_max_edges
        plot_out_prefix = resolve_path(repo_root, args.plot_out_prefix) if args.plot_out_prefix else None
        plot_png_path, plot_svg_path = plot_saved_graph(
            repo_root=repo_root,
            graph_dir=out_dir,
            out_prefix=plot_out_prefix,
            layout_seed=args.plot_layout_seed,
            title=args.plot_title,
            layout=args.plot_layout,
            max_edges=plot_max_edges,
            peaks_per_row=args.plot_peaks_per_row,
        )

    summary = {
        "inputs": {
            "matrix_tsv": str(matrix_path),
        },
        "params": vars(args),
        "requested_indep_test": args.indep_test,
        "resolved_indep_test": resolved_indep_test,
        "ci_kwargs": ci_kwargs,
        "resolved_fci_possible_dsep_depth": int(fci_possible_dsep_depth),
        "resolved_fci_max_path_length": int(fci_max_path_length),
        "n_rows_before_dropna": n_rows_before_dropna,
        "n_rows_after_dropna": n_rows_after_dropna,
        "n_columns_requested": len(requested_cols),
        "n_columns_selected": len(selected_cols),
        "n_columns_dropped": len(dropped_rows),
        "id_column": id_col,
        "selected_columns": selected_cols,
        "transform": args.transform,
        "background_mode": args.background_mode,
        "max_depth": args.max_depth,
        "n_background_rules": len(bg_rules),
        "n_edges": len(edge_records),
        "outputs": {
            "selected_matrix_tsv": str(selected_path),
            "selected_matrix_transformed_tsv": str(transformed_path),
            "selected_matrix_standardized_tsv": str(model_path),
            "dropped_columns_tsv": str(dropped_path),
            "feature_stats_tsv": str(stats_path),
            f"{args.method}_edges_tsv": str(edges_path),
            f"{args.method}_adjacency_tsv": str(adjacency_path),
            "background_knowledge_rules_tsv": str(bg_rules_path),
            "plot_png": str(plot_png_path) if plot_png_path is not None else None,
            "plot_svg": str(plot_svg_path) if plot_svg_path is not None else None,
        },
        "dagma": {
            "family_config": args.dagma_family_config,
            "family_config_details": dagma_config_meta,
            "requested_loss_type": args.dagma_loss_type,
            "resolved_loss_type": resolved_dagma_loss,
            "lambda1": args.dagma_lambda1,
            "w_threshold": args.dagma_w_threshold,
            "T": args.dagma_T,
            "verbose": args.dagma_verbose,
            "n_excluded_edges": len(dagma_exclude_edges),
            "excluded_edges": dagma_exclude_edges,
            "n_forbidden_edges_nonzero_before_clamp": int(getattr(dagma_model, "forbidden_edges_nonzero_before_clamp_", 0)),
            "max_forbidden_abs_weight_before_clamp": float(getattr(dagma_model, "forbidden_edges_max_abs_weight_before_clamp_", 0.0)),
            "preprocessing": dagma_family,
            "model_fit": {
                "h_final": float(getattr(dagma_model, "h_final", np.nan)),
                "score_final": float(getattr(dagma_model, "score_final", np.nan)),
                "column_family_stats": [
                    {
                        "column": col,
                        "family": stats.family,
                        "intercept": float(stats.intercept),
                        "dispersion": None if stats.dispersion is None else float(stats.dispersion),
                    }
                    for col, stats in zip(selected_cols, getattr(dagma_model, "column_family_stats_", []))
                ],
            },
        }
        if args.method == "dagma"
        else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote: {selected_path}")
    print(f"Wrote: {transformed_path}")
    print(f"Wrote: {model_path}")
    print(f"Wrote: {dropped_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {edges_path}")
    print(f"Wrote: {adjacency_path}")
    print(f"Wrote: {bg_rules_path}")
    print(f"Wrote: {summary_path}")
    if plot_png_path is not None and plot_svg_path is not None:
        print(f"Wrote: {plot_png_path}")
        print(f"Wrote: {plot_svg_path}")
    print(f"Method: {args.method}")
    if args.method == "dagma":
        print(f"DAGMA family config: {args.dagma_family_config}")
        print(f"DAGMA peak quant mode: {dagma_config_meta.get('required_peak_quant_mode') if dagma_config_meta else 'unknown'}")
        print(f"DAGMA resolved loss: {resolved_dagma_loss}")
    print(f"Selected columns: {len(selected_cols)}")
    print(f"Max depth: {args.max_depth}")
    print(f"Background rules: {len(bg_rules)}")
    print(f"Edges: {len(edge_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
