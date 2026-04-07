#!/usr/bin/env python3
"""Visualize a saved PC graph from run_pc_causallearn outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def region_label(region: str) -> str:
    mapping = {
        "promoter_primary_tss": "promoter",
        "promoter_e1_alt": "promoter\nE1",
        "fire_curated": "FIRE",
        "fire_5prime": "FIRE\n5p",
        "fire_core": "FIRE\ncore",
        "fire_3prime": "FIRE\n3p",
        "enhancer_e1_hmgxb3_3prime": "e1",
        "enhancer_e2_intragenic": "e2",
        "enhancer_e3_intragenic": "e3",
        "enhancer_e4_fire_proximal": "e4",
        "enhancer_e5_fire": "e5\nFIRE",
        "ltr_csf1r_promoter": "LTR",
        "ure_minus14kb": "URE\n-14kb",
        "enhancer_plus42kb": "enhancer\n+42kb",
        "enhancer_minus50kb_monocyte": "enhancer\n-50kb",
        "enhancer_downstream20kb_site1": "enhancer\n+20kb A",
        "enhancer_downstream20kb_site2": "enhancer\n+20kb B",
        "enhancer_downstream20kb_site3": "enhancer\n+20kb C",
        "enhancer_region6_downstream20kb": "region\n6",
        "enhancer_region7_downstream20kb": "region\n7",
        "enhancer_region8_downstream20kb": "region\n8",
    }
    if region in mapping:
        return mapping[region]
    if "promoter" in region:
        return region.replace("_", "\n")
    if "enhancer" in region:
        return region.replace("_", "\n")
    if "ure" in region:
        return region.upper().replace("_", "\n")
    return region.replace("_", "\n")


def pretty_node_label(node: str) -> str:
    if node.startswith("expr__"):
        gene = node.split("__", 1)[1]
        return f"expr\n{gene}"
    if "__" in node:
        region, mark = node.split("__", 1)
        return f"{region_label(region)}\n{mark}"
    return node


def node_region(node: str) -> str:
    if node.startswith("expr__"):
        return "expr"
    if "__" in node:
        return node.split("__", 1)[0]
    return "other"


def node_color(node: str) -> str:
    region = node_region(node)
    if region == "expr":
        return "#F4D35E"
    if region == "promoter_primary_tss":
        return "#78C0E0"
    if region == "fire_curated":
        return "#7BD389"
    if region == "ltr_csf1r_promoter":
        return "#F79D84"
    if "promoter" in region:
        return "#78C0E0"
    if "enhancer" in region or "ure" in region:
        return "#7BD389"
    return "#D3D3D3"


def node_shape(node: str) -> str:
    return "s" if node.startswith("expr__") else "o"


def edge_kind(endpoint1: str, endpoint2: str) -> tuple[str, tuple[str, str] | None]:
    if endpoint1 == "TAIL" and endpoint2 == "TAIL":
        return "undirected", None
    if endpoint1 == "TAIL" and endpoint2 == "ARROW":
        return "directed", ("node1", "node2")
    if endpoint1 == "ARROW" and endpoint2 == "TAIL":
        return "directed", ("node2", "node1")
    return "partial", None


def endpoint_symbol(endpoint: str) -> str:
    mapping = {"TAIL": "-", "ARROW": ">", "CIRCLE": "o"}
    return mapping.get(endpoint, "?")


def load_selected_nodes(pc_dir: Path) -> list[str]:
    summary_path = pc_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cols = summary.get("selected_columns")
        if cols:
            return [str(c) for c in cols]
    selected_path = pc_dir / "selected_matrix.tsv"
    if selected_path.exists():
        df = pd.read_csv(selected_path, sep="\t", nrows=1)
        return [c for c in df.columns if c != "metacell_id"]
    raise FileNotFoundError(f"Could not infer selected nodes from {pc_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot a PC graph from saved edge outputs")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--pc-dir", required=True, help="Directory containing pc_edges.tsv and run_summary.json")
    p.add_argument("--out-prefix", default=None, help="Default: <pc-dir>/pc_graph")
    p.add_argument("--layout-seed", type=int, default=13)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    pc_dir = resolve_path(repo_root, args.pc_dir)
    if pc_dir is None or not pc_dir.exists():
        raise FileNotFoundError(f"Missing PC directory: {args.pc_dir}")

    edges_path = pc_dir / "pc_edges.tsv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edge list: {edges_path}")

    edges_df = pd.read_csv(edges_path, sep="\t")
    nodes = load_selected_nodes(pc_dir)

    graph_for_layout = nx.Graph()
    graph_for_layout.add_nodes_from(nodes)
    for row in edges_df.to_dict(orient="records"):
        graph_for_layout.add_edge(row["node1"], row["node2"])

    pos = nx.spring_layout(graph_for_layout, seed=args.layout_seed)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()

    undirected_edges = []
    directed_edges = []
    partial_edges = []
    partial_labels = {}

    for row in edges_df.to_dict(orient="records"):
        kind, orientation = edge_kind(str(row["endpoint1"]), str(row["endpoint2"]))
        if kind == "undirected":
            undirected_edges.append((row["node1"], row["node2"]))
        elif kind == "directed" and orientation is not None:
            src_key, dst_key = orientation
            directed_edges.append((row[src_key], row[dst_key]))
        else:
            edge = (row["node1"], row["node2"])
            partial_edges.append(edge)
            partial_labels[edge] = f"{endpoint_symbol(str(row['endpoint1']))}{endpoint_symbol(str(row['endpoint2']))}"

    if undirected_edges:
        nx.draw_networkx_edges(
            graph_for_layout,
            pos,
            edgelist=undirected_edges,
            ax=ax,
            width=2.2,
            edge_color="#333333",
        )
    if directed_edges:
        nx.draw_networkx_edges(
            graph_for_layout,
            pos,
            edgelist=directed_edges,
            ax=ax,
            width=2.4,
            edge_color="#1F4E79",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22,
            min_source_margin=15,
            min_target_margin=15,
            connectionstyle="arc3,rad=0.05",
        )
    if partial_edges:
        nx.draw_networkx_edges(
            graph_for_layout,
            pos,
            edgelist=partial_edges,
            ax=ax,
            width=2.0,
            edge_color="#888888",
            style="dashed",
        )
        nx.draw_networkx_edge_labels(
            graph_for_layout,
            pos,
            edge_labels=partial_labels,
            ax=ax,
            font_size=8,
            font_color="#666666",
            rotate=False,
        )

    nodes_by_shape: dict[str, list[str]] = {"o": [], "s": []}
    for node in nodes:
        nodes_by_shape[node_shape(node)].append(node)

    for shape, shape_nodes in nodes_by_shape.items():
        if not shape_nodes:
            continue
        nx.draw_networkx_nodes(
            graph_for_layout,
            pos,
            nodelist=shape_nodes,
            node_shape=shape,
            node_color=[node_color(node) for node in shape_nodes],
            edgecolors="#222222",
            linewidths=1.2,
            node_size=2200 if shape == "s" else 1800,
            ax=ax,
        )

    nx.draw_networkx_labels(
        graph_for_layout,
        pos,
        labels={node: pretty_node_label(node) for node in nodes},
        font_size=9,
        font_color="#111111",
        ax=ax,
    )

    title = args.title or pc_dir.name.replace("_", " ")
    ax.set_title(title, fontsize=14)

    legend_text = "\n".join(
        [
            "Node colors:",
            "yellow = expression",
            "blue = promoter",
            "green = FIRE",
            "salmon = LTR promoter",
            "",
            "Edge styles:",
            "solid black = undirected",
            "blue arrow = directed",
            "dashed gray = partially oriented",
        ]
    )
    ax.text(
        1.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    out_prefix = (
        resolve_path(repo_root, args.out_prefix)
        if args.out_prefix
        else pc_dir / "pc_graph"
    )
    assert out_prefix is not None
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_prefix.with_suffix(".png")
    svg_path = out_prefix.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
