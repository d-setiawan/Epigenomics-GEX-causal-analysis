#!/usr/bin/env python3
"""Run a small PC/FCI/DAGMA graph-discovery job on a locus matrix."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.PCUtils import Meek, UCSepset
from causallearn.utils.cit import CIT
from scipy.stats import norm, rankdata
from tqdm.auto import tqdm

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


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


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


def build_background_knowledge(selected_cols: list[str], mode: str) -> tuple[BackgroundKnowledge | None, list[dict[str, str]]]:
    if mode == "none":
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


def build_dagma_family_mapping(selected_cols: list[str], family_config: str) -> list[str]:
    config = DAGMA_FAMILY_CONFIGS[family_config]
    families: list[str] = []
    for col in selected_cols:
        if col.startswith("expr__"):
            families.append(str(config["gene_family"]))
        else:
            families.append(str(config["peak_family"]))
    return families


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
    **kwargs,
):
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


def main() -> int:
    p = argparse.ArgumentParser(description="Run a PC or FCI baseline on a locus matrix with causal-learn")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--matrix-tsv", required=True)
    p.add_argument("--out-dir", default=None, help="Default: <matrix-dir>/<method>_<indep_test>_alpha_<alpha>")
    p.add_argument("--method", default="pc", choices=["pc", "fci", "dagma"])
    p.add_argument("--columns", default="auto", help="Comma-separated column list or 'auto'")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--indep-test", default="kci")
    p.add_argument("--stable", action="store_true", default=True)
    p.add_argument("--unstable", action="store_false", dest="stable")
    p.add_argument("--transform", default="none", choices=["none", "rank_gaussian"])
    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no-standardize", action="store_false", dest="standardize")
    p.add_argument("--background-mode", default="none", choices=["none", "minimal_expr_sink", "tiered_distal_promoter_expr"])
    p.add_argument("--min-unique", type=int, default=2)
    p.add_argument("--min-std", type=float, default=1e-8)
    p.add_argument("--max-depth", type=int, default=-1, help="Maximum conditioning-set size for PC skeleton search; -1 means no cap")
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

    dagma_family = None
    resolved_dagma_loss = "mixed_family"
    dagma_offsets = np.zeros((selected_df.shape[0], len(selected_cols)), dtype=float)
    dagma_nb2_dispersions = [np.nan] * len(selected_cols)
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

    if args.method == "pc":
        graph = run_pc_search(
            model_df.to_numpy(),
            alpha=args.alpha,
            indep_test=args.indep_test,
            stable=args.stable,
            show_progress=False,
            node_names=selected_cols,
            background_knowledge=background_knowledge,
            max_depth=args.max_depth,
        ).G
    elif args.method == "fci":
        graph, _ = run_fci_search(
            model_df.to_numpy(),
            alpha=args.alpha,
            indep_test=args.indep_test,
            show_progress=False,
            node_names=selected_cols,
            background_knowledge=background_knowledge,
            max_depth=args.max_depth,
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
        else matrix_path.parent / default_out_dir_name(args)
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
        plot_out_prefix = resolve_path(repo_root, args.plot_out_prefix) if args.plot_out_prefix else None
        plot_png_path, plot_svg_path = plot_saved_graph(
            repo_root=repo_root,
            graph_dir=out_dir,
            out_prefix=plot_out_prefix,
            layout_seed=args.plot_layout_seed,
            title=args.plot_title,
            layout=args.plot_layout,
            max_edges=args.plot_max_edges,
            peaks_per_row=args.plot_peaks_per_row,
        )

    summary = {
        "inputs": {
            "matrix_tsv": str(matrix_path),
        },
        "params": vars(args),
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
