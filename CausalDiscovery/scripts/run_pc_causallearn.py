#!/usr/bin/env python3
"""Run a small PC baseline on a locus matrix using causal-learn."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from scipy.stats import norm, rankdata


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def auto_select_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col.startswith("expr__"):
            cols.append(col)
        elif "__H3" in col:
            cols.append(col)
    return cols


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


def main() -> int:
    p = argparse.ArgumentParser(description="Run a PC baseline on a locus matrix with causal-learn")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--matrix-tsv", required=True)
    p.add_argument("--out-dir", default=None, help="Default: <matrix-dir>/pc_<indep_test>_alpha_<alpha>")
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
    p.add_argument("--dropna-rows", action="store_true", default=True)
    p.add_argument("--keep-na-rows", action="store_false", dest="dropna_rows")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matrix_path = resolve_path(repo_root, args.matrix_tsv)
    if matrix_path is None or not matrix_path.exists():
        raise FileNotFoundError(f"Missing matrix TSV: {args.matrix_tsv}")

    df = pd.read_csv(matrix_path, sep="\t")
    if args.columns == "auto":
        requested_cols = auto_select_columns(df)
    else:
        requested_cols = [c.strip() for c in args.columns.split(",") if c.strip()]

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

    transformed_df = apply_transform(selected_df, args.transform)
    model_df = zscore_df(transformed_df) if args.standardize else transformed_df.copy()
    model_df = model_df.astype(float)
    background_knowledge, bg_rules = build_background_knowledge(selected_cols, args.background_mode)

    cg = pc(
        model_df.to_numpy(),
        alpha=args.alpha,
        indep_test=args.indep_test,
        stable=args.stable,
        show_progress=False,
        node_names=selected_cols,
        background_knowledge=background_knowledge,
    )

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else matrix_path.parent / (
            f"pc_{sanitize_token(args.indep_test)}_alpha_{sanitize_token(str(args.alpha))}"
            + ("" if args.transform == "none" else f"_xform_{sanitize_token(args.transform)}")
            + ("" if args.background_mode == "none" else f"_bg_{sanitize_token(args.background_mode)}")
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_records = [edge_to_record(edge) for edge in cg.G.get_graph_edges()]
    edges_df = pd.DataFrame(edge_records, columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"])
    adjacency_df = pd.DataFrame(cg.G.graph, index=selected_cols, columns=selected_cols)
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
    edges_path = out_dir / "pc_edges.tsv"
    adjacency_path = out_dir / "pc_adjacency.tsv"
    bg_rules_path = out_dir / "background_knowledge_rules.tsv"
    summary_path = out_dir / "run_summary.json"

    selected_with_ids.to_csv(selected_path, sep="\t", index=False)
    transformed_with_ids.to_csv(transformed_path, sep="\t", index=False)
    model_with_ids.to_csv(model_path, sep="\t", index=False)
    dropped_df.to_csv(dropped_path, sep="\t", index=False)
    stats_df.to_csv(stats_path, sep="\t", index=False)
    edges_df.to_csv(edges_path, sep="\t", index=False)
    adjacency_df.to_csv(adjacency_path, sep="\t")
    bg_rules_df.to_csv(bg_rules_path, sep="\t", index=False)

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
        "n_background_rules": len(bg_rules),
        "n_edges": len(edge_records),
        "outputs": {
            "selected_matrix_tsv": str(selected_path),
            "selected_matrix_transformed_tsv": str(transformed_path),
            "selected_matrix_standardized_tsv": str(model_path),
            "dropped_columns_tsv": str(dropped_path),
            "feature_stats_tsv": str(stats_path),
            "pc_edges_tsv": str(edges_path),
            "pc_adjacency_tsv": str(adjacency_path),
            "background_knowledge_rules_tsv": str(bg_rules_path),
        },
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
    print(f"Selected columns: {len(selected_cols)}")
    print(f"Background rules: {len(bg_rules)}")
    print(f"Edges: {len(edge_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
