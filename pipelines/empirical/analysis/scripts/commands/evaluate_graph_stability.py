#!/usr/bin/env python3
"""Evaluate edge stability and support across saved graph-discovery runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

from _common import infer_repo_root, resolve_path, sanitize_token

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover - optional at runtime
    hypergeom = None


ENDPOINT_SYMBOLS = {
    "TAIL": "-",
    "ARROW": ">",
    "CIRCLE": "o",
    "NULL": ".",
}


def split_csv(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_params(summary: dict) -> dict:
    params = summary.get("params")
    return params if isinstance(params, dict) else {}


def summary_inputs(summary: dict) -> dict:
    inputs = summary.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def summary_section(summary: dict, key: str) -> dict:
    section = summary.get(key)
    return section if isinstance(section, dict) else {}


def find_edge_file(graph_dir: Path, method: str | None = None) -> Path | None:
    if method:
        candidate = graph_dir / f"{method}_edges.tsv"
        if candidate.exists():
            return candidate
    candidates = sorted(graph_dir.glob("*_edges.tsv"))
    return candidates[0] if candidates else None


def discover_graph_dirs(methods_root: Path, explicit_dirs: Iterable[Path] | None = None) -> list[Path]:
    dirs: set[Path] = set()
    if explicit_dirs:
        for graph_dir in explicit_dirs:
            if graph_dir.exists():
                dirs.add(graph_dir.resolve())
    if methods_root.exists():
        for summary_path in methods_root.rglob("run_summary.json"):
            graph_dir = summary_path.parent
            if find_edge_file(graph_dir):
                dirs.add(graph_dir.resolve())
    return sorted(dirs)


def infer_layout_fields(methods_root: Path, graph_dir: Path, summary: dict) -> dict[str, str]:
    params = summary_params(summary)
    method = str(params.get("method") or "")
    family_config = ""
    dataset = ""
    gene = ""

    try:
        rel_parts = graph_dir.resolve().relative_to(methods_root.resolve()).parts
    except ValueError:
        rel_parts = ()

    if rel_parts:
        method = method or rel_parts[0]
        if len(rel_parts) >= 4 and rel_parts[0] == "dagma":
            dataset = rel_parts[1]
            family_config = rel_parts[2]
            gene = rel_parts[3]
        elif len(rel_parts) >= 3:
            dataset = rel_parts[1]
            gene = rel_parts[2]

    if not family_config:
        family_config = str(
            params.get("dagma_family_config")
            or summary_section(summary, "dagma").get("family_config")
            or ""
        )

    matrix_tsv = summary_inputs(summary).get("matrix_tsv") or params.get("matrix_tsv") or ""
    if not gene and matrix_tsv:
        match = re.search(r"/([^/]+)/nearby_peaks/[^/]+_nearby_peak_matrix\.tsv$", str(matrix_tsv))
        if match:
            gene = match.group(1)

    graph_id_parts = [part for part in [method, dataset, family_config, gene, graph_dir.name] if part]
    return {
        "method": method,
        "dataset": dataset,
        "family_config": family_config,
        "gene": gene,
        "run_name": graph_dir.name,
        "graph_id": "::".join(graph_id_parts),
    }


def selected_nodes(summary: dict, graph_dir: Path) -> list[str]:
    nodes = summary.get("selected_columns")
    if isinstance(nodes, list) and nodes:
        return [str(node) for node in nodes]
    matrix_path = graph_dir / "selected_matrix.tsv"
    if matrix_path.exists():
        header = pd.read_csv(matrix_path, sep="\t", nrows=0).columns.tolist()
        return [col for col in header if col not in {"sample_id", "metacell_id", "row_id"}]
    return []


def endpoint_signature(node1: str, endpoint1: str, node2: str, endpoint2: str) -> tuple[str, str, str, str]:
    if node1 <= node2:
        return node1, endpoint1, endpoint2, node2
    return node2, endpoint2, endpoint1, node1


def edge_kind_and_direction(node1: str, endpoint1: str, node2: str, endpoint2: str) -> tuple[str, str, str]:
    e1 = endpoint1.upper()
    e2 = endpoint2.upper()
    if e1 == "TAIL" and e2 == "ARROW":
        return "directed", node1, node2
    if e1 == "ARROW" and e2 == "TAIL":
        return "directed", node2, node1
    if e1 == "TAIL" and e2 == "TAIL":
        return "undirected", "", ""
    if e1 == "ARROW" and e2 == "ARROW":
        return "bidirected", "", ""
    if "CIRCLE" in {e1, e2}:
        return "partial", "", ""
    return "other", "", ""


def normalize_edges(edges_df: pd.DataFrame, graph_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, row in edges_df.fillna("").iterrows():
        node1 = str(row.get("node1", ""))
        node2 = str(row.get("node2", ""))
        if not node1 or not node2:
            continue
        endpoint1 = str(row.get("endpoint1", "")).upper()
        endpoint2 = str(row.get("endpoint2", "")).upper()
        left, left_endpoint, right_endpoint, right = endpoint_signature(node1, endpoint1, node2, endpoint2)
        edge_kind, source, target = edge_kind_and_direction(node1, endpoint1, node2, endpoint2)
        endpoint_label = f"{ENDPOINT_SYMBOLS.get(left_endpoint, left_endpoint)}-{ENDPOINT_SYMBOLS.get(right_endpoint, right_endpoint)}"
        exact_key = f"{left}|{left_endpoint}|{right_endpoint}|{right}"
        adjacency_key = f"{left}||{right}"
        directed_key = f"{source}->{target}" if source and target else ""
        out = {
            "graph_id": graph_id,
            "edge_index": int(idx),
            "node_left": left,
            "node_right": right,
            "endpoint_left": left_endpoint,
            "endpoint_right": right_endpoint,
            "endpoint_label": endpoint_label,
            "adjacency_key": adjacency_key,
            "exact_edge_key": exact_key,
            "edge_kind": edge_kind,
            "source": source,
            "target": target,
            "directed_key": directed_key,
            "edge_text": str(row.get("edge_text", "")),
        }
        for numeric_col in ["weight", "abs_weight"]:
            if numeric_col in row:
                out[numeric_col] = pd.to_numeric(row[numeric_col], errors="coerce")
        rows.append(out)
    return pd.DataFrame(rows)


def is_expr_node(node: str) -> bool:
    return node.startswith("expr__")


def is_peak_node(node: str) -> bool:
    return bool(re.match(r"^chr[^:]+:\d+-\d+__", node))


def locate_metadata_path(summary: dict, repo_root: Path) -> Path | None:
    matrix_tsv = summary_inputs(summary).get("matrix_tsv") or summary_params(summary).get("matrix_tsv")
    matrix_path = resolve_path(repo_root, str(matrix_tsv)) if matrix_tsv else None
    if matrix_path is None or not matrix_path.exists():
        return None
    nearby_dir = matrix_path.parent
    matches = sorted(nearby_dir.glob("*_nearby_peak_metadata.tsv"))
    return matches[0] if matches else None


def load_peak_metadata(summary: dict, repo_root: Path) -> dict[str, dict[str, object]]:
    metadata_path = locate_metadata_path(summary, repo_root)
    if metadata_path is None:
        return {}
    metadata = pd.read_csv(metadata_path, sep="\t")
    if "variable_name" not in metadata.columns:
        return {}
    return {
        str(row["variable_name"]): row
        for row in metadata.to_dict(orient="records")
    }


def load_node_support(graph_dir: Path) -> dict[str, dict[str, object]]:
    support_path = graph_dir / "node_support.tsv"
    if not support_path.exists():
        return {}
    support = pd.read_csv(support_path, sep="\t")
    if "node" not in support.columns:
        return {}
    return {
        str(row["node"]): row
        for row in support.to_dict(orient="records")
    }


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def hypergeom_enrichment_pvalue(population: int, successes: int, draws: int, observed: int) -> float:
    if population <= 0 or successes <= 0 or draws <= 0 or observed <= 0:
        return 1.0
    if hypergeom is not None:
        return float(hypergeom.sf(observed - 1, population, successes, draws))
    total = 0.0
    denom = math.comb(population, draws)
    upper = min(successes, draws)
    for k in range(observed, upper + 1):
        total += math.comb(successes, k) * math.comb(population - successes, draws - k) / denom
    return float(min(max(total, 0.0), 1.0))


def support_metrics(edges: pd.DataFrame, nodes: list[str], peak_metadata: dict[str, dict[str, object]]) -> dict[str, object]:
    peak_nodes = [node for node in nodes if is_peak_node(node)]
    curated_peaks = {
        node
        for node in peak_nodes
        if boolish(peak_metadata.get(node, {}).get("overlaps_curated_region"))
    }

    expr_nodes = {node for node in nodes if is_expr_node(node)}
    expr_adjacent_peaks: set[str] = set()
    all_edge_peaks: set[str] = set()
    directed_peak_to_expr = 0
    directed_expr_to_peak = 0

    for row in edges.to_dict(orient="records"):
        left = str(row["node_left"])
        right = str(row["node_right"])
        for node in [left, right]:
            if is_peak_node(node):
                all_edge_peaks.add(node)
        if left in expr_nodes and is_peak_node(right):
            expr_adjacent_peaks.add(right)
        if right in expr_nodes and is_peak_node(left):
            expr_adjacent_peaks.add(left)
        if row.get("edge_kind") == "directed":
            source = str(row.get("source", ""))
            target = str(row.get("target", ""))
            if is_peak_node(source) and target in expr_nodes:
                directed_peak_to_expr += 1
            if source in expr_nodes and is_peak_node(target):
                directed_expr_to_peak += 1

    curated_expr_adjacent = expr_adjacent_peaks & curated_peaks
    curated_edge_peaks = all_edge_peaks & curated_peaks
    n_peak_nodes = len(peak_nodes)
    n_curated_peaks = len(curated_peaks)
    n_expr_adjacent = len(expr_adjacent_peaks)
    n_curated_expr_adjacent = len(curated_expr_adjacent)
    return {
        "n_peak_nodes": n_peak_nodes,
        "n_curated_peak_nodes": n_curated_peaks,
        "curated_peak_fraction": n_curated_peaks / n_peak_nodes if n_peak_nodes else 0.0,
        "n_edge_incident_peaks": len(all_edge_peaks),
        "n_curated_edge_incident_peaks": len(curated_edge_peaks),
        "curated_edge_incident_peak_fraction": len(curated_edge_peaks) / len(all_edge_peaks) if all_edge_peaks else 0.0,
        "n_expr_adjacent_peaks": n_expr_adjacent,
        "n_curated_expr_adjacent_peaks": n_curated_expr_adjacent,
        "curated_expr_adjacent_peak_fraction": n_curated_expr_adjacent / n_expr_adjacent if n_expr_adjacent else 0.0,
        "curated_expr_adjacent_hypergeom_p": hypergeom_enrichment_pvalue(
            n_peak_nodes,
            n_curated_peaks,
            n_expr_adjacent,
            n_curated_expr_adjacent,
        ),
        "expr_adjacent_peak_nodes": "|".join(sorted(expr_adjacent_peaks)),
        "curated_expr_adjacent_peak_nodes": "|".join(sorted(curated_expr_adjacent)),
        "n_directed_peak_to_expr_edges": directed_peak_to_expr,
        "n_directed_expr_to_peak_edges": directed_expr_to_peak,
    }


def safe_param(summary: dict, key: str, default: object = "") -> object:
    return summary_params(summary).get(key, default)


def graph_record(repo_root: Path, methods_root: Path, graph_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    summary_path = graph_dir / "run_summary.json"
    summary = read_json(summary_path)
    fields = infer_layout_fields(methods_root, graph_dir, summary)
    method = fields["method"]
    edge_path = find_edge_file(graph_dir, method)
    if edge_path is None:
        raise FileNotFoundError(f"Missing edge TSV in {graph_dir}")
    edges_df = pd.read_csv(edge_path, sep="\t")
    nodes = selected_nodes(summary, graph_dir)
    normalized = normalize_edges(edges_df, fields["graph_id"])

    peak_metadata = load_peak_metadata(summary, repo_root)
    node_support = load_node_support(graph_dir)
    metrics = support_metrics(normalized, nodes, peak_metadata)
    n_directed = int((normalized.get("edge_kind", pd.Series(dtype=str)) == "directed").sum())
    n_undirected = int((normalized.get("edge_kind", pd.Series(dtype=str)) == "undirected").sum())
    n_partial = int((normalized.get("edge_kind", pd.Series(dtype=str)) == "partial").sum())
    n_bidirected = int((normalized.get("edge_kind", pd.Series(dtype=str)) == "bidirected").sum())

    record = {
        **fields,
        "graph_dir": str(graph_dir),
        "edge_tsv": str(edge_path),
        "matrix_tsv": summary_inputs(summary).get("matrix_tsv") or safe_param(summary, "matrix_tsv"),
        "alpha": safe_param(summary, "alpha"),
        "indep_test": safe_param(summary, "indep_test"),
        "max_depth": summary.get("max_depth", safe_param(summary, "max_depth")),
        "transform": summary.get("transform", safe_param(summary, "transform")),
        "standardize": safe_param(summary, "standardize"),
        "background_mode": summary.get("background_mode", safe_param(summary, "background_mode")),
        "n_rows_after_dropna": summary.get("n_rows_after_dropna", ""),
        "n_columns_selected": summary.get("n_columns_selected", len(nodes)),
        "n_columns_dropped": summary.get("n_columns_dropped", ""),
        "n_nodes": len(nodes),
        "n_edges": int(normalized.shape[0]),
        "n_directed_edges": n_directed,
        "n_undirected_edges": n_undirected,
        "n_partial_edges": n_partial,
        "n_bidirected_edges": n_bidirected,
        "has_node_support": bool(node_support),
    }
    for key in ["dagma_lambda1", "dagma_w_threshold", "dagma_T"]:
        record[key] = safe_param(summary, key)
    dagma_summary = summary_section(summary, "dagma")
    if dagma_summary:
        record["dagma_lambda1"] = dagma_summary.get("lambda1", record["dagma_lambda1"])
        record["dagma_w_threshold"] = dagma_summary.get("w_threshold", record["dagma_w_threshold"])
        record["dagma_T"] = dagma_summary.get("T", record["dagma_T"])
    record.update(metrics)
    return record, normalized


def passes_filters(record: dict[str, object], args) -> bool:
    filters = {
        "method": split_csv(args.methods),
        "dataset": split_csv(args.datasets),
        "gene": split_csv(args.genes),
        "family_config": split_csv(args.family_configs),
    }
    for key, allowed in filters.items():
        if allowed and str(record.get(key, "")) not in set(allowed):
            return False
    return True


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def pairwise_rows(run_df: pd.DataFrame, edge_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    edges_by_graph = {
        graph_id: {
            "adjacency": set(sub["adjacency_key"].astype(str)),
            "exact": set(sub["exact_edge_key"].astype(str)),
            "directed": set(k for k in sub["directed_key"].astype(str) if k),
        }
        for graph_id, sub in edge_df.groupby("graph_id", dropna=False)
    }
    for _, group in run_df.groupby(group_cols, dropna=False):
        records = group.to_dict(orient="records")
        for left, right in combinations(records, 2):
            left_sets = edges_by_graph.get(left["graph_id"], {"adjacency": set(), "exact": set(), "directed": set()})
            right_sets = edges_by_graph.get(right["graph_id"], {"adjacency": set(), "exact": set(), "directed": set()})
            shared_adj = left_sets["adjacency"] & right_sets["adjacency"]
            shared_exact = left_sets["exact"] & right_sets["exact"]
            rows.append(
                {
                    **{col: left.get(col, "") for col in group_cols},
                    "graph_id_a": left["graph_id"],
                    "graph_id_b": right["graph_id"],
                    "run_name_a": left["run_name"],
                    "run_name_b": right["run_name"],
                    "n_edges_a": left["n_edges"],
                    "n_edges_b": right["n_edges"],
                    "adjacency_intersection": len(shared_adj),
                    "adjacency_union": len(left_sets["adjacency"] | right_sets["adjacency"]),
                    "adjacency_jaccard": jaccard(left_sets["adjacency"], right_sets["adjacency"]),
                    "exact_intersection": len(shared_exact),
                    "exact_union": len(left_sets["exact"] | right_sets["exact"]),
                    "exact_jaccard": jaccard(left_sets["exact"], right_sets["exact"]),
                    "directed_jaccard": jaccard(left_sets["directed"], right_sets["directed"]),
                    "orientation_agreement_given_adjacency": len(shared_exact) / len(shared_adj) if shared_adj else 1.0,
                }
            )
    return pd.DataFrame(rows)


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def edge_stability_rows(run_df: pd.DataFrame, edge_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group_runs in run_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_cols, group_values))
        graph_ids = set(group_runs["graph_id"].astype(str))
        n_runs = len(graph_ids)
        group_edges = edge_df[edge_df["graph_id"].isin(graph_ids)].copy()
        for adjacency_key, sub in group_edges.groupby("adjacency_key", dropna=False):
            orientation_counts = Counter(sub["exact_edge_key"].astype(str))
            kind_counts = Counter(sub["edge_kind"].astype(str))
            dominant_exact, dominant_count = orientation_counts.most_common(1)[0]
            dominant_kind = kind_counts.most_common(1)[0][0]
            present_graphs = set(sub["graph_id"].astype(str))
            first = sub.iloc[0]
            source_counts = Counter(k for k in sub["directed_key"].astype(str) if k)
            dominant_directed_key = source_counts.most_common(1)[0][0] if source_counts else ""
            weight_values = pd.to_numeric(sub.get("abs_weight", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(
                {
                    **group,
                    "node_left": first["node_left"],
                    "node_right": first["node_right"],
                    "adjacency_key": adjacency_key,
                    "n_runs": n_runs,
                    "n_present": len(present_graphs),
                    "adjacency_frequency": len(present_graphs) / n_runs if n_runs else 0.0,
                    "dominant_exact_edge_key": dominant_exact,
                    "dominant_orientation_frequency": dominant_count / n_runs if n_runs else 0.0,
                    "orientation_entropy": entropy(orientation_counts),
                    "dominant_edge_kind": dominant_kind,
                    "dominant_directed_key": dominant_directed_key,
                    "is_expr_adjacent": is_expr_node(str(first["node_left"])) or is_expr_node(str(first["node_right"])),
                    "orientation_counts": json.dumps(dict(sorted(orientation_counts.items())), sort_keys=True),
                    "present_graph_ids": "|".join(sorted(present_graphs)),
                    "mean_abs_weight": float(weight_values.mean()) if not weight_values.empty else "",
                    "max_abs_weight": float(weight_values.max()) if not weight_values.empty else "",
                }
            )
    return pd.DataFrame(rows)


def group_summary_rows(
    run_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    group_cols: list[str],
    stable_threshold: float,
    core_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairwise_groups = pairwise_df.groupby(group_cols, dropna=False) if not pairwise_df.empty else {}
    stability_groups = stability_df.groupby(group_cols, dropna=False) if not stability_df.empty else {}
    for group_values, group_runs in run_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_cols, group_values))
        key = group_values[0] if len(group_values) == 1 else group_values
        pair_sub = pairwise_groups.get_group(key) if hasattr(pairwise_groups, "groups") and key in pairwise_groups.groups else pd.DataFrame()
        stab_sub = stability_groups.get_group(key) if hasattr(stability_groups, "groups") and key in stability_groups.groups else pd.DataFrame()
        rows.append(
            {
                **group,
                "n_runs": int(group_runs.shape[0]),
                "mean_edges": float(group_runs["n_edges"].mean()),
                "min_edges": int(group_runs["n_edges"].min()),
                "max_edges": int(group_runs["n_edges"].max()),
                "mean_directed_edges": float(group_runs["n_directed_edges"].mean()),
                "mean_expr_adjacent_peaks": float(group_runs["n_expr_adjacent_peaks"].mean()),
                "mean_curated_expr_adjacent_peaks": float(group_runs["n_curated_expr_adjacent_peaks"].mean()),
                "mean_curated_expr_adjacent_peak_fraction": float(group_runs["curated_expr_adjacent_peak_fraction"].mean()),
                "adjacency_union_size": int(stab_sub.shape[0]) if not stab_sub.empty else 0,
                "n_stable_adjacencies": int((stab_sub["adjacency_frequency"] >= stable_threshold).sum()) if not stab_sub.empty else 0,
                "n_core_adjacencies": int((stab_sub["adjacency_frequency"] >= core_threshold).sum()) if not stab_sub.empty else 0,
                "mean_pairwise_adjacency_jaccard": float(pair_sub["adjacency_jaccard"].mean()) if not pair_sub.empty else "",
                "mean_pairwise_exact_jaccard": float(pair_sub["exact_jaccard"].mean()) if not pair_sub.empty else "",
                "mean_orientation_agreement_given_adjacency": float(pair_sub["orientation_agreement_given_adjacency"].mean()) if not pair_sub.empty else "",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    run_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    group_cols: list[str],
    stable_threshold: float,
    core_threshold: float,
) -> None:
    report_path = out_dir / "evaluation_report.md"
    lines = [
        "# Graph Stability pipelines/empirical/evaluation",
        "",
        "This report was generated from saved `PC`, `FCI`, and `DAGMA` graph output directories.",
        "",
        "Core stability summaries use edge-set overlap across graph settings. For two runs $a$ and $b$, adjacency stability is summarized with",
        "",
        "$$J(E_a,E_b)=\\frac{|E_a\\cap E_b|}{|E_a\\cup E_b|}.$$",
        "",
        f"Stable adjacencies use frequency >= `{stable_threshold}`; core adjacencies use frequency >= `{core_threshold}`.",
        "",
        f"Grouping columns: `{','.join(group_cols)}`.",
        "",
        f"Total graph runs: `{run_df.shape[0]}`.",
        "",
        "## Output Files",
        "",
        "- `graph_runs.tsv`: one row per graph run with graph-size and support metrics.",
        "- `normalized_edges.tsv`: one row per normalized edge per graph.",
        "- `edge_stability.tsv`: edge frequencies and orientation consistency within each group.",
        "- `pairwise_graph_similarity.tsv`: pairwise Jaccard and orientation agreement across runs.",
        "- `group_summary.tsv`: compact group-level stability and biological-support summaries.",
        "",
    ]
    if not group_summary_df.empty:
        preview_cols = [
            *group_cols,
            "n_runs",
            "mean_edges",
            "adjacency_union_size",
            "n_stable_adjacencies",
            "mean_pairwise_adjacency_jaccard",
            "mean_curated_expr_adjacent_peak_fraction",
        ]
        preview_cols = [col for col in preview_cols if col in group_summary_df.columns]
        lines.extend(["## Group Preview", "", group_summary_df[preview_cols].head(20).to_markdown(index=False), ""])
    if not stability_df.empty:
        top = stability_df.sort_values(
            ["adjacency_frequency", "dominant_orientation_frequency"],
            ascending=[False, False],
        ).head(20)
        preview_cols = [
            *group_cols,
            "node_left",
            "node_right",
            "adjacency_frequency",
            "dominant_orientation_frequency",
            "dominant_edge_kind",
            "dominant_directed_key",
        ]
        preview_cols = [col for col in preview_cols if col in top.columns]
        lines.extend(["## Most Stable Edges", "", top[preview_cols].to_markdown(index=False), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate edge stability across saved graph output directories.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--methods-root",
        default="pipelines/empirical/analysis/outputs/methods",
        help="Root containing method-first graph outputs.",
    )
    parser.add_argument("--graph-dir", action="append", default=[], help="Explicit graph directory to include. May be repeated.")
    parser.add_argument("--out-dir", default="pipelines/empirical/evaluation/outputs/graph_stability/latest")
    parser.add_argument(
        "--group-cols",
        default="dataset,gene,method,family_config",
        help="Comma-separated columns for within-group stability comparisons.",
    )
    parser.add_argument("--methods", default=None, help="Comma-separated methods to include.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset IDs to include.")
    parser.add_argument("--genes", default=None, help="Comma-separated gene/locus IDs to include.")
    parser.add_argument("--family-configs", default=None, help="Comma-separated DAGMA family configs to include.")
    parser.add_argument("--stable-threshold", type=float, default=0.5)
    parser.add_argument("--core-threshold", type=float, default=0.8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = resolve_path(infer_repo_root(), args.repo_root) if args.repo_root else infer_repo_root()
    assert repo_root is not None
    methods_root = resolve_path(repo_root, args.methods_root)
    out_dir = resolve_path(repo_root, args.out_dir)
    if methods_root is None or out_dir is None:
        raise ValueError("Could not resolve methods root or output directory")
    explicit_dirs = [resolve_path(repo_root, item) for item in args.graph_dir]
    graph_dirs = discover_graph_dirs(methods_root, [p for p in explicit_dirs if p is not None])
    if not graph_dirs:
        raise FileNotFoundError(f"No graph output directories found under {methods_root}")

    run_records: list[dict[str, object]] = []
    edge_frames: list[pd.DataFrame] = []
    for graph_dir in graph_dirs:
        try:
            record, edges = graph_record(repo_root, methods_root, graph_dir)
        except Exception as exc:
            print(f"[evaluate] skipping {graph_dir}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not passes_filters(record, args):
            continue
        run_records.append(record)
        edge_frames.append(edges)

    if not run_records:
        raise ValueError("No graph runs passed the requested filters")

    run_df = pd.DataFrame(run_records)
    edge_df = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    group_cols = split_csv(args.group_cols)
    missing_group_cols = [col for col in group_cols if col not in run_df.columns]
    if missing_group_cols:
        raise ValueError(f"Unknown group columns: {missing_group_cols}")

    pairwise_df = pairwise_rows(run_df, edge_df, group_cols) if not edge_df.empty else pd.DataFrame()
    stability_df = edge_stability_rows(run_df, edge_df, group_cols) if not edge_df.empty else pd.DataFrame()
    group_summary_df = group_summary_rows(
        run_df,
        pairwise_df,
        stability_df,
        group_cols,
        stable_threshold=args.stable_threshold,
        core_threshold=args.core_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    run_df.sort_values(["dataset", "gene", "method", "family_config", "run_name"]).to_csv(
        out_dir / "graph_runs.tsv",
        sep="\t",
        index=False,
    )
    edge_df.to_csv(out_dir / "normalized_edges.tsv", sep="\t", index=False)
    pairwise_df.to_csv(out_dir / "pairwise_graph_similarity.tsv", sep="\t", index=False)
    stability_df.sort_values(group_cols + ["adjacency_frequency", "dominant_orientation_frequency"], ascending=[True] * len(group_cols) + [False, False]).to_csv(
        out_dir / "edge_stability.tsv",
        sep="\t",
        index=False,
    )
    group_summary_df.to_csv(out_dir / "group_summary.tsv", sep="\t", index=False)
    write_report(
        out_dir,
        run_df=run_df,
        group_summary_df=group_summary_df,
        stability_df=stability_df,
        group_cols=group_cols,
        stable_threshold=args.stable_threshold,
        core_threshold=args.core_threshold,
    )

    print(f"Wrote graph run table: {out_dir / 'graph_runs.tsv'}")
    print(f"Wrote normalized edges: {out_dir / 'normalized_edges.tsv'}")
    print(f"Wrote edge stability: {out_dir / 'edge_stability.tsv'}")
    print(f"Wrote pairwise graph similarity: {out_dir / 'pairwise_graph_similarity.tsv'}")
    print(f"Wrote group summary: {out_dir / 'group_summary.tsv'}")
    print(f"Wrote report: {out_dir / 'evaluation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
