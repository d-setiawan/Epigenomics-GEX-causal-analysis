#!/usr/bin/env python3
"""Visualize a saved graph directory from run_pc_causallearn outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import networkx as nx
import pandas as pd


PEAK_PATTERN = re.compile(r"^(?P<chrom>chr[^:]+):(?P<start>\d+)-(?P<end>\d+)__(?P<mark>.+)$")
MARK_COLORS = {
    "H3K27ac": "#e76f51",
    "H3K27me3": "#6d597a",
    "H3K4me1": "#2a9d8f",
    "H3K4me2": "#457b9d",
    "H3K4me3": "#1d3557",
    "H3K9me3": "#8d99ae",
}


def get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("matplotlib is required to plot graph outputs.") from exc
    return plt


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def region_label(region: str) -> str:
    mapping = {
        "promoter_primary_tss": "promoter",
        "promoter_e1_alt": "promoter E1",
        "fire_curated": "FIRE",
        "fire_5prime": "FIRE 5p",
        "fire_core": "FIRE core",
        "fire_3prime": "FIRE 3p",
        "enhancer_e1_hmgxb3_3prime": "e1",
        "enhancer_e2_intragenic": "e2",
        "enhancer_e3_intragenic": "e3",
        "enhancer_e4_fire_proximal": "e4",
        "enhancer_e5_fire": "e5/FIRE",
        "ltr_csf1r_promoter": "LTR",
        "ure_minus14kb": "URE -14kb",
        "enhancer_plus42kb": "enhancer +42kb",
        "enhancer_minus50kb_monocyte": "enhancer -50kb",
        "enhancer_downstream20kb_site1": "enhancer +20kb A",
        "enhancer_downstream20kb_site2": "enhancer +20kb B",
        "enhancer_downstream20kb_site3": "enhancer +20kb C",
        "enhancer_region6_downstream20kb": "region 6",
        "enhancer_region7_downstream20kb": "region 7",
        "enhancer_region8_downstream20kb": "region 8",
    }
    if region in mapping:
        return mapping[region]
    return region.replace("_", " ")


def parse_peak_node(node: str) -> dict[str, str | int] | None:
    match = PEAK_PATTERN.match(node)
    if match is None:
        return None
    info = match.groupdict()
    return {
        "chrom": info["chrom"],
        "start": int(info["start"]),
        "end": int(info["end"]),
        "mark": info["mark"],
    }


def is_peak_node(node: str) -> bool:
    return parse_peak_node(node) is not None


def infer_method(graph_dir: Path) -> str:
    for candidate in ("pc", "fci", "dagma"):
        if (graph_dir / f"{candidate}_edges.tsv").exists():
            return candidate
    matches = sorted(graph_dir.glob("*_edges.tsv"))
    if len(matches) == 1:
        return matches[0].stem.replace("_edges", "")
    raise FileNotFoundError(f"Could not locate an edges TSV in {graph_dir}")


def load_edges_df(graph_dir: Path, method: str | None = None) -> tuple[str, pd.DataFrame]:
    inferred_method = method or infer_method(graph_dir)
    edges_path = graph_dir / f"{inferred_method}_edges.tsv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edge list: {edges_path}")
    return inferred_method, pd.read_csv(edges_path, sep="\t")


def load_selected_nodes(graph_dir: Path) -> list[str]:
    summary_path = graph_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cols = summary.get("selected_columns")
        if cols:
            return [str(c) for c in cols]
    selected_path = graph_dir / "selected_matrix.tsv"
    if selected_path.exists():
        df = pd.read_csv(selected_path, sep="\t", nrows=1)
        return [c for c in df.columns if c not in {"metacell_id", "sample_id", "row_id"}]
    raise FileNotFoundError(f"Could not infer selected nodes from {graph_dir}")


def load_node_support(graph_dir: Path, support_path: Path | None) -> pd.DataFrame:
    candidate = support_path or (graph_dir / "node_support.tsv")
    if not candidate.exists():
        return pd.DataFrame(columns=["node_name", "sources", "highlight_class"])
    df = pd.read_csv(candidate, sep="\t")
    required = {"node_name", "sources", "highlight_class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Support TSV missing required columns {sorted(missing)}: {candidate}")
    return df


def support_style(highlight_class: str) -> dict[str, str | float]:
    styles: dict[str, dict[str, str | float]] = {
        "eperturbdb_only": {
            "edgecolor": "#E67E22",
            "linewidth": 3.2,
            "size_delta": 320.0,
        },
        "encode_screen_only": {
            "edgecolor": "#8E5CF7",
            "linewidth": 3.2,
            "size_delta": 320.0,
        },
        "both": {
            "edgecolor": "#D7263D",
            "linewidth": 4.0,
            "size_delta": 440.0,
        },
    }
    return styles.get(
        highlight_class,
        {
            "edgecolor": "#222222",
            "linewidth": 0.0,
            "size_delta": 0.0,
        },
    )


def load_nearby_peak_metadata(graph_dir: Path, gene_name: str | None) -> pd.DataFrame:
    if gene_name is None:
        return pd.DataFrame()
    candidate = graph_dir.parent / f"{gene_name}_nearby_peak_metadata.tsv"
    if candidate.exists():
        return pd.read_csv(candidate, sep="\t")
    return pd.DataFrame()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def infer_gene_name(nodes: list[str]) -> str | None:
    expr_nodes = [node for node in nodes if node.startswith("expr__")]
    if len(expr_nodes) == 1:
        return expr_nodes[0].split("__", 1)[1]
    return None


def endpoint_symbol(endpoint: str) -> str:
    mapping = {"TAIL": "-", "ARROW": ">", "CIRCLE": "o"}
    return mapping.get(endpoint, "?")


def edge_kind(endpoint1: str, endpoint2: str) -> tuple[str, tuple[str, str] | None]:
    if endpoint1 == "TAIL" and endpoint2 == "TAIL":
        return "undirected", None
    if endpoint1 == "TAIL" and endpoint2 == "ARROW":
        return "directed", ("node1", "node2")
    if endpoint1 == "ARROW" and endpoint2 == "TAIL":
        return "directed", ("node2", "node1")
    return "partial", None


def edge_width(row: dict[str, object], *, base: float = 2.0, scale: float = 3.0) -> float:
    weight = row.get("abs_weight")
    if weight is None:
        return base
    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        return base
    return base + scale * max(weight_value, 0.0)


def select_edges_for_plot(
    edge_rows: list[dict[str, str]],
    *,
    gene_node: str | None,
    max_edges: int | None,
    metadata_lookup: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    if max_edges is None or len(edge_rows) <= max_edges:
        return edge_rows

    def edge_score(row: dict[str, str]) -> tuple[int, int, int, str]:
        nodes = [str(row["node1"]), str(row["node2"])]
        touches_gene = int(gene_node in nodes if gene_node is not None else 0)
        curated_overlap = int(
            any(as_bool(metadata_lookup.get(node, {}).get("overlaps_curated_region", False)) for node in nodes)
        )
        kind, _ = edge_kind(str(row["endpoint1"]), str(row["endpoint2"]))
        kind_score = {"directed": 2, "partial": 1, "undirected": 0}[kind]
        return (touches_gene, curated_overlap, kind_score, str(row["edge_text"]))

    return sorted(edge_rows, key=edge_score, reverse=True)[:max_edges]


def local_peak_sort_key(node: str, metadata_lookup: dict[str, dict[str, object]]) -> tuple[object, ...]:
    meta = metadata_lookup.get(node, {})
    chrom = meta.get("chrom")
    start = meta.get("start")
    end = meta.get("end")
    mark = meta.get("mark")
    if chrom is not None and start is not None and end is not None:
        return (str(chrom), int(start), int(end), str(mark or ""))
    peak_info = parse_peak_node(node)
    if peak_info is not None:
        return (
            str(peak_info["chrom"]),
            int(peak_info["start"]),
            int(peak_info["end"]),
            str(peak_info["mark"]),
        )
    return ("zzz", 0, 0, node)


def local_node_label(node: str, metadata_lookup: dict[str, dict[str, object]]) -> str:
    if node.startswith("expr__"):
        return node.split("__", 1)[1]

    peak_info = parse_peak_node(node)
    if peak_info is None:
        return node.replace("__", "\n")

    mark = str(peak_info["mark"])
    meta = metadata_lookup.get(node, {})
    curated_regions = str(meta.get("overlapping_curated_regions", "")).strip()
    if curated_regions:
        region_names = [region_label(token.strip()) for token in curated_regions.split(";") if token.strip()]
        if region_names:
            return f"{mark}\n{region_names[0]}"

    distance_bp = meta.get("signed_distance_to_tss_bp")
    if distance_bp is not None and not pd.isna(distance_bp):
        distance_kb = float(distance_bp) / 1000.0
        return f"{mark}\n{distance_kb:+.1f} kb"

    center_mb = (int(peak_info["start"]) + int(peak_info["end"])) / 2_000_000.0
    return f"{mark}\n{center_mb:.3f} Mb"


def local_node_color(node: str) -> str:
    if node.startswith("expr__"):
        return "#f4a261"
    peak_info = parse_peak_node(node)
    if peak_info is None:
        return "#d3d3d3"
    return MARK_COLORS.get(str(peak_info["mark"]), "#8ecae6")


def local_outline_color(node: str, metadata_lookup: dict[str, dict[str, object]]) -> str:
    if node.startswith("expr__"):
        return "#9c4221"
    if as_bool(metadata_lookup.get(node, {}).get("overlaps_curated_region", False)):
        return "#d7263d"
    return "#264653"


def graph_support_lookup(support_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {
        str(row["node_name"]): {
            "sources": str(row["sources"]),
            "highlight_class": str(row["highlight_class"]),
        }
        for row in support_df.to_dict(orient="records")
    }


def is_local_peak_graph(nodes: list[str]) -> bool:
    gene_nodes = [node for node in nodes if node.startswith("expr__")]
    peak_nodes = [node for node in nodes if is_peak_node(node)]
    return len(gene_nodes) == 1 and len(peak_nodes) >= max(1, len(nodes) - len(gene_nodes) - 2)


def plot_local_graph(
    *,
    graph_dir: Path,
    method: str,
    nodes: list[str],
    edges_df: pd.DataFrame,
    support_lookup: dict[str, dict[str, str]],
    metadata_df: pd.DataFrame,
    output_prefix: Path,
    title: str | None = None,
    max_edges: int | None = 30,
    peaks_per_row: int = 18,
) -> tuple[Path, Path]:
    plt = get_plt()
    metadata_lookup = {
        str(row["variable_name"]): row
        for row in metadata_df.to_dict(orient="records")
        if "variable_name" in row
    }

    gene_node = next((node for node in nodes if node.startswith("expr__")), None)
    if gene_node is None:
        raise ValueError(f"Could not identify an expression node in {graph_dir}")
    gene_name = gene_node.split("__", 1)[1]

    edge_rows = edges_df.to_dict(orient="records")
    edge_rows = select_edges_for_plot(edge_rows, gene_node=gene_node, max_edges=max_edges, metadata_lookup=metadata_lookup)

    peaks = [node for node in nodes if node != gene_node]
    peaks.sort(key=lambda node: local_peak_sort_key(node, metadata_lookup))

    n_rows = max(1, math.ceil(max(len(peaks), 1) / max(peaks_per_row, 1)))
    positions: dict[str, tuple[float, float]] = {}
    for row_idx in range(n_rows):
        row_peaks = peaks[row_idx * peaks_per_row : (row_idx + 1) * peaks_per_row]
        if not row_peaks:
            continue
        if len(row_peaks) > 1:
            xs = [-1.0 + (2.0 * i / (len(row_peaks) - 1)) for i in range(len(row_peaks))]
        else:
            xs = [0.0]
        y = 1.2 - 0.55 * row_idx
        for peak_name, x in zip(row_peaks, xs):
            positions[peak_name] = (float(x), float(y))
    positions[gene_node] = (0.0, -0.65)

    fig_height = max(4.8, 3.7 + 0.95 * n_rows)
    fig_width = max(8.0, min(16.0, peaks_per_row * 0.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    ax.set_title(title or f"{gene_name} {method.upper()} local graph", fontsize=13)

    partial_labels: dict[tuple[str, str], str] = {}
    for row in edge_rows:
        node1 = str(row["node1"])
        node2 = str(row["node2"])
        x0, y0 = positions[node1]
        x1, y1 = positions[node2]
        kind, orientation = edge_kind(str(row["endpoint1"]), str(row["endpoint2"]))

        if kind == "directed" and orientation is not None:
            src = str(row[orientation[0]])
            dst = str(row[orientation[1]])
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            connection = "arc3,rad=0.08" if src != gene_node and dst != gene_node else "arc3,rad=0.0"
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1f4e79",
                    linewidth=edge_width(row),
                    shrinkA=12,
                    shrinkB=12,
                    alpha=0.85,
                    connectionstyle=connection,
                ),
            )
            continue

        if kind == "undirected":
            ax.plot([x0, x1], [y0, y1], color="#333333", linewidth=edge_width(row, base=1.8, scale=2.4), alpha=0.85, zorder=1)
            continue

        ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=edge_width(row, base=1.6, scale=2.0), alpha=0.85, linestyle="--", zorder=1)
        partial_labels[(node1, node2)] = f"{endpoint_symbol(str(row['endpoint1']))}{endpoint_symbol(str(row['endpoint2']))}"

    peak_x = [positions[name][0] for name in peaks]
    peak_y = [positions[name][1] for name in peaks]
    peak_colors = [local_node_color(name) for name in peaks]
    peak_edgecolors = [local_outline_color(name, metadata_lookup) for name in peaks]
    peak_linewidths = [2.0 if as_bool(metadata_lookup.get(name, {}).get("overlaps_curated_region", False)) else 1.0 for name in peaks]
    if peaks:
        ax.scatter(
            peak_x,
            peak_y,
            s=130,
            color=peak_colors,
            edgecolor=peak_edgecolors,
            linewidth=peak_linewidths,
            zorder=3,
        )

    ax.scatter(
        [positions[gene_node][0]],
        [positions[gene_node][1]],
        s=260,
        color=local_node_color(gene_node),
        edgecolor=local_outline_color(gene_node, metadata_lookup),
        linewidth=1.4,
        marker="s",
        zorder=4,
    )

    highlighted = [node for node in nodes if support_lookup.get(node, {}).get("highlight_class") not in {"", "none", None}]
    for node in highlighted:
        style = support_style(str(support_lookup[node]["highlight_class"]))
        x, y = positions[node]
        ax.scatter(
            [x],
            [y],
            s=(360 if node == gene_node else 220) + float(style["size_delta"]),
            facecolors="none",
            edgecolors=str(style["edgecolor"]),
            linewidths=float(style["linewidth"]),
            marker="s" if node == gene_node else "o",
            zorder=5,
        )

    for peak_name in peaks:
        x, y = positions[peak_name]
        ax.text(x, y + 0.085, local_node_label(peak_name, metadata_lookup), ha="center", va="bottom", fontsize=6.8)

    gene_x, gene_y = positions[gene_node]
    ax.text(gene_x, gene_y - 0.12, gene_name, ha="center", va="top", fontsize=10, fontweight="bold")

    for (node1, node2), label in partial_labels.items():
        x0, y0 = positions[node1]
        x1, y1 = positions[node2]
        ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.04, label, ha="center", va="bottom", fontsize=7, color="#666666")

    legend_text = "\n".join(
        [
            "Node colors:",
            "orange square = gene expression",
            "peak colors = histone marks",
            "red outlines = overlaps curated region",
            "",
            "Edge styles:",
            "blue arrow = directed",
            "black line = undirected",
            "gray dashed = partial/PAG",
        ]
    )
    ax.text(
        1.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.05, 1.45)

    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


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


def node_color_spring(node: str) -> str:
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


def plot_spring_graph(
    *,
    nodes: list[str],
    edges_df: pd.DataFrame,
    support_lookup: dict[str, dict[str, str]],
    output_prefix: Path,
    title: str,
    layout_seed: int,
) -> tuple[Path, Path]:
    plt = get_plt()
    graph_for_layout = nx.Graph()
    graph_for_layout.add_nodes_from(nodes)
    for row in edges_df.to_dict(orient="records"):
        graph_for_layout.add_edge(row["node1"], row["node2"])

    pos = nx.spring_layout(graph_for_layout, seed=layout_seed)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()

    undirected_edges = []
    directed_edges = []
    partial_edges = []
    partial_labels = {}

    for row in edges_df.to_dict(orient="records"):
        kind, orientation = edge_kind(str(row["endpoint1"]), str(row["endpoint2"]))
        if kind == "undirected":
            undirected_edges.append((row["node1"], row["node2"], edge_width(row, base=2.2, scale=2.4)))
        elif kind == "directed" and orientation is not None:
            src_key, dst_key = orientation
            directed_edges.append((row[src_key], row[dst_key], edge_width(row, base=2.4, scale=3.0)))
        else:
            edge = (row["node1"], row["node2"])
            partial_edges.append(edge)
            partial_labels[edge] = f"{endpoint_symbol(str(row['endpoint1']))}{endpoint_symbol(str(row['endpoint2']))}"

    if undirected_edges:
        nx.draw_networkx_edges(
            graph_for_layout,
            pos,
            edgelist=[(u, v) for u, v, _ in undirected_edges],
            ax=ax,
            width=[width for _, _, width in undirected_edges],
            edge_color="#333333",
        )
    if directed_edges:
        nx.draw_networkx_edges(
            graph_for_layout,
            pos,
            edgelist=[(u, v) for u, v, _ in directed_edges],
            ax=ax,
            width=[width for _, _, width in directed_edges],
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
        size = 2200 if shape == "s" else 1800
        nx.draw_networkx_nodes(
            graph_for_layout,
            pos,
            nodelist=shape_nodes,
            node_shape=shape,
            node_color=[node_color_spring(node) for node in shape_nodes],
            edgecolors="#222222",
            linewidths=1.2,
            node_size=size,
            ax=ax,
        )
        highlighted_nodes = [node for node in shape_nodes if support_lookup.get(node, {}).get("highlight_class") not in {"", "none", None}]
        if highlighted_nodes:
            nx.draw_networkx_nodes(
                graph_for_layout,
                pos,
                nodelist=highlighted_nodes,
                node_shape=shape,
                node_color="none",
                edgecolors=[str(support_style(str(support_lookup[node]["highlight_class"]))["edgecolor"]) for node in highlighted_nodes],
                linewidths=[float(support_style(str(support_lookup[node]["highlight_class"]))["linewidth"]) for node in highlighted_nodes],
                node_size=[size + float(support_style(str(support_lookup[node]["highlight_class"]))["size_delta"]) for node in highlighted_nodes],
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

    ax.set_title(title, fontsize=14)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_saved_graph(
    *,
    repo_root: Path,
    graph_dir: Path,
    out_prefix: Path | None = None,
    node_support_tsv: Path | None = None,
    layout_seed: int = 13,
    title: str | None = None,
    layout: str = "auto",
    max_edges: int | None = 30,
    peaks_per_row: int = 18,
) -> tuple[Path, Path]:
    graph_dir = graph_dir.resolve()
    method, edges_df = load_edges_df(graph_dir)
    nodes = load_selected_nodes(graph_dir)
    support_df = load_node_support(graph_dir, node_support_tsv)
    support_lookup = graph_support_lookup(support_df)
    gene_name = infer_gene_name(nodes)
    metadata_df = load_nearby_peak_metadata(graph_dir, gene_name)

    chosen_layout = layout
    if chosen_layout == "auto":
        chosen_layout = "local" if is_local_peak_graph(nodes) else "spring"

    output_prefix = out_prefix or (graph_dir / "graph_plot")
    if chosen_layout == "local":
        return plot_local_graph(
            graph_dir=graph_dir,
            method=method,
            nodes=nodes,
            edges_df=edges_df,
            support_lookup=support_lookup,
            metadata_df=metadata_df,
            output_prefix=output_prefix,
            title=title,
            max_edges=max_edges,
            peaks_per_row=peaks_per_row,
        )

    return plot_spring_graph(
        nodes=nodes,
        edges_df=edges_df,
        support_lookup=support_lookup,
        output_prefix=output_prefix,
        title=title or graph_dir.name.replace("_", " "),
        layout_seed=layout_seed,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Plot a saved graph directory from run_pc_causallearn outputs")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--graph-dir", "--pc-dir", dest="graph_dir", required=True, help="Directory containing *_edges.tsv and run_summary.json")
    p.add_argument("--out-prefix", default=None, help="Default: <graph-dir>/graph_plot")
    p.add_argument(
        "--node-support-tsv",
        default=None,
        help="Optional node-level support TSV. Default: <graph-dir>/node_support.tsv if it exists.",
    )
    p.add_argument("--layout-seed", type=int, default=13)
    p.add_argument("--title", default=None)
    p.add_argument("--layout", default="auto", choices=["auto", "local", "spring"])
    p.add_argument("--max-edges", type=int, default=30)
    p.add_argument("--peaks-per-row", type=int, default=18)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    graph_dir = resolve_path(repo_root, args.graph_dir)
    if graph_dir is None or not graph_dir.exists():
        raise FileNotFoundError(f"Missing graph directory: {args.graph_dir}")

    support_path = resolve_path(repo_root, args.node_support_tsv)
    out_prefix = resolve_path(repo_root, args.out_prefix)
    png_path, svg_path = plot_saved_graph(
        repo_root=repo_root,
        graph_dir=graph_dir,
        out_prefix=out_prefix,
        node_support_tsv=support_path,
        layout_seed=args.layout_seed,
        title=args.title,
        layout=args.layout,
        max_edges=args.max_edges,
        peaks_per_row=args.peaks_per_row,
    )

    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
