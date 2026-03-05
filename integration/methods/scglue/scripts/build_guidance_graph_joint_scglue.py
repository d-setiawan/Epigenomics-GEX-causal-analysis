#!/usr/bin/env python3
"""Build one joint guidance graph for RNA + all histone marks.

Backends:
1. custom (default): memory-stable streaming builder with native interval semantics.
2. scglue: scGLUE-native rna_anchored_prior_graph backend.
"""

from __future__ import annotations

import argparse
import csv
import gc
import heapq
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

from build_gene_tss_table import iter_gtf
from build_guidance_graph_pilot_scglue import (
    MARK_DEFAULT_ALPHA,
    MARK_DEFAULT_DECAY,
    MARK_DEFAULT_WINDOW,
    MARK_SIGN,
    infer_repo_root,
    parse_bin_name,
    resolve_path,
    try_annotate_rna_with_gtf,
    write_tsv,
)


GeneInterval = Tuple[str, int, int]  # (gene_name, start, end)
BinInterval = Tuple[str, int, int]  # (bin_node, start, end)


def subset_chrom_features(chrom: ad.AnnData, max_features: int | None) -> tuple[ad.AnnData, int]:
    if max_features is None or max_features <= 0 or chrom.n_vars <= max_features:
        return chrom, 0
    x = chrom.layers["counts"] if "counts" in chrom.layers else chrom.X
    sums = np.asarray(x.sum(axis=0)).ravel()
    idx = np.argpartition(sums, -max_features)[-max_features:]
    idx = np.sort(idx)
    return chrom[:, idx].copy(), int(chrom.n_vars - max_features)


def prepare_chrom_for_scglue_backend(chrom: ad.AnnData) -> tuple[ad.AnnData, int]:
    chrom = chrom.copy()
    n_before = chrom.n_vars
    orig = (
        chrom.var["orig_feature"].astype(str).tolist()
        if "orig_feature" in chrom.var.columns
        else chrom.var_names.astype(str).tolist()
    )
    parsed = []
    keep_idx = []
    for i, feat in enumerate(orig):
        try:
            c, s, e = parse_bin_name(feat)
        except Exception:
            continue
        keep_idx.append(i)
        parsed.append((c, s, e))

    if len(keep_idx) < n_before:
        chrom = chrom[:, keep_idx].copy()
        orig = [orig[i] for i in keep_idx]

    chrom.var["orig_feature"] = orig
    chrom.var["chrom"] = [x[0] for x in parsed]
    chrom.var["chromStart"] = np.asarray([x[1] for x in parsed], dtype=int)
    chrom.var["chromEnd"] = np.asarray([x[2] for x in parsed], dtype=int)
    chrom.var["strand"] = "."
    dropped = int(n_before - chrom.n_vars)
    return chrom, dropped


def write_graphml_from_tsv(nodes_path: Path, edges_path: Path, graphml_path: Path) -> None:
    g = nx.MultiDiGraph()
    nodes = pd.read_csv(nodes_path, sep="\t")
    for row in nodes.itertuples(index=False):
        g.add_node(
            str(getattr(row, "node")),
            node_type=str(getattr(row, "node_type", "")),
            modality=str(getattr(row, "modality", "")),
            mark=str(getattr(row, "mark", "")),
        )
    edges = pd.read_csv(edges_path, sep="\t")
    for row in edges.itertuples(index=False):
        g.add_edge(
            str(getattr(row, "source")),
            str(getattr(row, "target")),
            weight=float(getattr(row, "weight", 1.0)),
            sign=int(getattr(row, "sign", 1)),
            distance_bp=int(getattr(row, "distance_bp", 0)),
            mark=str(getattr(row, "mark", "")),
            type=str(getattr(row, "type", "")),
        )
    nx.write_graphml(g, graphml_path)


def load_gene_coord_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required_identity = {"chrom", "gene_name", "gene_id"}
    missing_identity = required_identity - set(df.columns)
    if missing_identity:
        raise ValueError(f"Gene coord table missing required columns: {sorted(missing_identity)}")

    out = df.copy()
    out["chrom"] = out["chrom"].astype(str)
    out["gene_name"] = out["gene_name"].astype(str)
    out["gene_id"] = out["gene_id"].astype(str)

    if {"chromStart", "chromEnd", "strand"}.issubset(set(out.columns)):
        out["chromStart"] = out["chromStart"].astype(int)
        out["chromEnd"] = out["chromEnd"].astype(int)
        out["strand"] = out["strand"].astype(str)
    elif {"tss", "strand"}.issubset(set(out.columns)):
        out["tss"] = out["tss"].astype(int)
        out["strand"] = out["strand"].astype(str)
        out["chromStart"] = out["tss"]
        out["chromEnd"] = out["tss"] + 1
    else:
        raise ValueError(
            "Gene coord table must include either (chromStart, chromEnd, strand) "
            "or (tss, strand) columns."
        )

    out = out.loc[out["strand"].isin(["+", "-"])].copy()
    out = out.loc[out["chromEnd"] > out["chromStart"]].copy()
    return out[["chrom", "chromStart", "chromEnd", "strand", "gene_name", "gene_id"]]


def build_gene_coord_from_gtf(gtf_path: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for rec in iter_gtf(gtf_path):
        if rec["feature"] not in {"gene", "transcript"}:
            continue
        attrs = rec["attrs"]
        gene_id = str(attrs.get("gene_id", ""))
        gene_name = str(attrs.get("gene_name", ""))
        if not gene_id and not gene_name:
            continue
        if not gene_name:
            gene_name = gene_id
        key = gene_id or gene_name
        if key in seen:
            continue
        seen.add(key)

        start = int(rec["start"]) - 1  # GTF 1-based inclusive -> BED 0-based half-open
        end = int(rec["end"])
        if end <= start:
            continue

        rows.append(
            {
                "chrom": str(rec["chrom"]),
                "chromStart": int(start),
                "chromEnd": int(end),
                "strand": str(rec["strand"]),
                "gene_name": gene_name,
                "gene_id": gene_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No usable gene coordinates parsed from {gtf_path}")
    df = df.loc[df["strand"].isin(["+", "-"])].copy()
    df = df.loc[df["chromEnd"] > df["chromStart"]].copy()
    return df


def pick_rna_genes_custom(
    rna: ad.AnnData,
    gene_coords: pd.DataFrame,
    annotation_by: str,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    rna_var = rna.var.copy()
    rna_var["rna_gene"] = rna.var_names.astype(str)

    by_name = gene_coords[gene_coords["gene_name"].isin(set(rna_var["rna_gene"]))].copy()
    by_name["rna_gene"] = by_name["gene_name"]

    by_id = pd.DataFrame()
    if "gene_ids" in rna_var.columns:
        gid_map = rna_var[["rna_gene", "gene_ids"]].copy()
        gid_map["gene_ids"] = gid_map["gene_ids"].astype(str)
        gid_map["gene_id_trim"] = gid_map["gene_ids"].str.replace(r"\.\d+$", "", regex=True)
        gid_map = gid_map.drop_duplicates("gene_id_trim")

        t = gene_coords.copy()
        t["gene_id_trim"] = t["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        by_id = t.merge(gid_map[["gene_id_trim", "rna_gene"]], on="gene_id_trim", how="inner")

    if annotation_by == "gene_name":
        chosen = by_name
        mode = "gene_name"
    elif annotation_by == "gene_id":
        chosen = by_id
        mode = "gene_id"
    else:
        n_name = int(by_name["rna_gene"].nunique()) if not by_name.empty else 0
        n_id = int(by_id["rna_gene"].nunique()) if not by_id.empty else 0
        if n_name >= n_id:
            chosen = by_name
            mode = "gene_name"
        else:
            chosen = by_id
            mode = "gene_id"

    if chosen.empty:
        raise RuntimeError(
            "Custom backend failed to map RNA genes to genomic coordinates. "
            "Try a matching GTF build or pass --annotation-by gene_name/gene_id explicitly."
        )

    chosen = chosen[
        ["chrom", "chromStart", "chromEnd", "strand", "rna_gene", "gene_name", "gene_id"]
    ].copy()
    chosen["chrom"] = chosen["chrom"].astype(str)
    chosen["chromStart"] = chosen["chromStart"].astype(int)
    chosen["chromEnd"] = chosen["chromEnd"].astype(int)
    chosen["strand"] = chosen["strand"].astype(str)
    chosen["rna_gene"] = chosen["rna_gene"].astype(str)
    chosen = chosen.drop_duplicates("rna_gene")
    chosen = chosen.loc[chosen["chromEnd"] > chosen["chromStart"]].copy()

    stats = {
        "annotation_mode": mode,
        "annotation_coverage_before_filter": float(chosen["rna_gene"].nunique() / max(1, rna.n_vars)),
        "rna_genes_after_coord_filter": float(chosen["rna_gene"].nunique()),
        "backend": "custom",
    }
    return chosen, stats


def apply_gene_region_semantics(genes_df: pd.DataFrame, gene_region: str, promoter_len: int) -> pd.DataFrame:
    df = genes_df.copy()
    plus = df["strand"].to_numpy() == "+"
    minus = df["strand"].to_numpy() == "-"

    start = df["chromStart"].to_numpy(dtype=np.int64)
    end = df["chromEnd"].to_numpy(dtype=np.int64)

    if gene_region == "promoter":
        p_start = start.copy()
        p_end = end.copy()

        p_end[plus] = p_start[plus] + 1
        p_start[minus] = p_end[minus] - 1

        if promoter_len > 0:
            p_start[plus] -= int(promoter_len)
            p_end[minus] += int(promoter_len)
        start = p_start
        end = p_end
    elif gene_region == "combined":
        if promoter_len > 0:
            start[plus] -= int(promoter_len)
            end[minus] += int(promoter_len)
    elif gene_region != "gene_body":
        raise ValueError(f"Unrecognized gene_region: {gene_region}")

    start = np.maximum(start, 0)
    keep = end > start

    out = df.loc[keep].copy()
    out["chromStart"] = start[keep].astype(int)
    out["chromEnd"] = end[keep].astype(int)
    return out


def build_genes_by_chr(genes_df: pd.DataFrame) -> Dict[str, List[GeneInterval]]:
    out: Dict[str, List[GeneInterval]] = {}
    for chrom, sub in genes_df.groupby("chrom", sort=False):
        s = sub.sort_values(["chromStart", "chromEnd", "rna_gene"])
        out[str(chrom)] = [
            (str(r.rna_gene), int(r.chromStart), int(r.chromEnd))
            for r in s.itertuples(index=False)
        ]
    return out


def parse_prefixed_bin_features(chrom: ad.AnnData) -> tuple[List[Tuple[str, str, int, int]], int]:
    feats = chrom.var_names.astype(str).tolist()
    orig = (
        chrom.var["orig_feature"].astype(str).tolist()
        if "orig_feature" in chrom.var.columns
        else feats
    )
    parsed: List[Tuple[str, str, int, int]] = []
    n_dropped = 0
    for node_name, orig_feat in zip(feats, orig):
        try:
            chrom_name, start, end = parse_bin_name(orig_feat)
        except Exception:
            n_dropped += 1
            continue
        if int(end) <= int(start):
            n_dropped += 1
            continue
        parsed.append((node_name, str(chrom_name), int(start), int(end)))
    return parsed, n_dropped


def build_bins_by_chr(parsed_bins: Iterable[Tuple[str, str, int, int]]) -> Dict[str, List[BinInterval]]:
    out: Dict[str, List[BinInterval]] = {}
    for node, chrom, start, end in parsed_bins:
        out.setdefault(chrom, []).append((str(node), int(start), int(end)))
    for chrom in list(out.keys()):
        out[chrom].sort(key=lambda x: (x[1], x[2], x[0]))
    return out


def interval_dist(l_start: int, l_end: int, r_start: int, r_end: int) -> int:
    """Signed interval distance (same semantics as scglue.genomics.interval_dist)."""
    if l_start < r_end and r_start < l_end:
        return 0
    if l_end <= r_start:
        return l_end - r_start - 1
    # r_end <= l_start
    return l_start - r_end + 1


def iter_native_window_pairs(
    genes_by_chr: Dict[str, List[GeneInterval]],
    bins_by_chr: Dict[str, List[BinInterval]],
    window_bp: int,
) -> Iterable[Tuple[str, str, int]]:
    """Yield (gene, bin, abs_distance) pairs with scGLUE-native window semantics."""
    for chrom, genes in genes_by_chr.items():
        bins = bins_by_chr.get(chrom)
        if not bins:
            continue

        right_idx = 0
        window: List[BinInterval] = []
        n_bins = len(bins)

        for gene, g_start, g_end in genes:
            broke = False
            i = 0
            while i < len(window):
                node, b_start, b_end = window[i]
                d = interval_dist(g_start, g_end, b_start, b_end)
                if -window_bp <= d <= window_bp:
                    yield gene, node, abs(int(d))
                    i += 1
                elif d > window_bp:
                    window.pop(i)
                else:  # d < -window_bp
                    broke = True
                    break

            if not broke:
                while right_idx < n_bins:
                    node, b_start, b_end = bins[right_idx]
                    d = interval_dist(g_start, g_end, b_start, b_end)
                    if -window_bp <= d <= window_bp:
                        yield gene, node, abs(int(d))
                    elif d > window_bp:
                        right_idx += 1
                        continue

                    window.append((node, b_start, b_end))
                    right_idx += 1
                    if d < -window_bp:
                        break


def edge_weight(distance_bp: int, alpha: float, decay_bp: int, weight_mode: str) -> float:
    if weight_mode == "power":
        # Native scGLUE decay: dist_power_decay
        return float(alpha * (((float(distance_bp) + 1000.0) / 1000.0) ** (-0.75)))
    # Exponential decay used in earlier pipeline revisions
    return float(alpha * math.exp(-float(distance_bp) / float(max(1, decay_bp))))


def write_mark_edges_custom(
    ew: csv.DictWriter,
    mark: str,
    chrom: ad.AnnData,
    genes_by_chr: Dict[str, List[GeneInterval]],
    window_bp: int,
    decay_bp: int,
    alpha: float,
    sign: int,
    max_edges_per_bin: int,
    add_self_loops: bool,
    seen_gene_loops: set[str],
    weight_mode: str,
) -> Dict[str, int]:
    parsed_bins, dropped_parse = parse_prefixed_bin_features(chrom)
    bins_by_chr = build_bins_by_chr(parsed_bins)

    mark_edges_added = 0
    connected_genes = set()
    connected_bins = set()
    n_pairs_candidate = 0
    n_pairs_kept = 0

    # Optional nearest-gene pruning per bin.
    # 0 means "no pruning", which best matches native graph semantics.
    if max_edges_per_bin > 0:
        topk: Dict[str, List[Tuple[int, str, int, float]]] = {}
        for gene, node, dist in iter_native_window_pairs(genes_by_chr, bins_by_chr, window_bp):
            n_pairs_candidate += 1
            weight = edge_weight(dist, alpha=alpha, decay_bp=decay_bp, weight_mode=weight_mode)
            heap = topk.setdefault(node, [])
            item = (-int(dist), str(gene), int(dist), float(weight))
            if len(heap) < max_edges_per_bin:
                heapq.heappush(heap, item)
            else:
                worst_dist = -heap[0][0]
                worst_gene = heap[0][1]
                if int(dist) < worst_dist or (int(dist) == worst_dist and str(gene) < worst_gene):
                    heapq.heapreplace(heap, item)

        for node, heap in topk.items():
            selected = sorted(heap, key=lambda x: (x[2], x[1]))
            for _, gene, dist, weight in selected:
                ew.writerow(
                    {
                        "source": gene,
                        "target": node,
                        "weight": weight,
                        "sign": int(sign),
                        "distance_bp": int(dist),
                        "mark": mark,
                        "type": "fwd",
                    }
                )
                ew.writerow(
                    {
                        "source": node,
                        "target": gene,
                        "weight": weight,
                        "sign": int(sign),
                        "distance_bp": int(dist),
                        "mark": mark,
                        "type": "rev",
                    }
                )
                mark_edges_added += 2
                n_pairs_kept += 1
                connected_genes.add(gene)
                connected_bins.add(node)
    else:
        for gene, node, dist in iter_native_window_pairs(genes_by_chr, bins_by_chr, window_bp):
            n_pairs_candidate += 1
            weight = edge_weight(dist, alpha=alpha, decay_bp=decay_bp, weight_mode=weight_mode)
            ew.writerow(
                {
                    "source": gene,
                    "target": node,
                    "weight": weight,
                    "sign": int(sign),
                    "distance_bp": int(dist),
                    "mark": mark,
                    "type": "fwd",
                }
            )
            ew.writerow(
                {
                    "source": node,
                    "target": gene,
                    "weight": weight,
                    "sign": int(sign),
                    "distance_bp": int(dist),
                    "mark": mark,
                    "type": "rev",
                }
            )
            mark_edges_added += 2
            n_pairs_kept += 1
            connected_genes.add(gene)
            connected_bins.add(node)

    if add_self_loops:
        for node in map(str, chrom.var_names):
            ew.writerow(
                {
                    "source": node,
                    "target": node,
                    "weight": 1.0,
                    "sign": 1,
                    "distance_bp": 0,
                    "mark": "self",
                    "type": "loop",
                }
            )
            mark_edges_added += 1

        for genes in genes_by_chr.values():
            for gene, _, _ in genes:
                if gene in seen_gene_loops:
                    continue
                ew.writerow(
                    {
                        "source": gene,
                        "target": gene,
                        "weight": 1.0,
                        "sign": 1,
                        "distance_bp": 0,
                        "mark": "self",
                        "type": "loop",
                    }
                )
                seen_gene_loops.add(gene)
                mark_edges_added += 1

    return {
        "graph_edges_added": int(mark_edges_added),
        "connected_genes": int(len(connected_genes)),
        "connected_bins": int(len(connected_bins)),
        "chrom_features_dropped_by_parse": int(dropped_parse),
        "candidate_pairs_before_prune": int(n_pairs_candidate),
        "kept_pairs_after_prune": int(n_pairs_kept),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Build joint guidance graph for scGLUE")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--preprocess-dir", required=True, help="Joint preprocess directory")
    p.add_argument("--gtf", required=True, help="GTF/GTF.GZ annotation path")
    p.add_argument("--out-dir", default=None, help="Default: <preprocess-parent>/graph")
    p.add_argument("--backend", default="custom", choices=["custom", "scglue"])
    p.add_argument("--annotation-by", default="auto", choices=["auto", "gene_id", "gene_name"])
    p.add_argument(
        "--gene-tss",
        default=None,
        help="Optional prebuilt gene coordinate TSV for --backend custom",
    )
    p.add_argument("--gene-region", default="promoter", choices=["gene_body", "promoter", "combined"])
    p.add_argument("--promoter-len", type=int, default=1000)
    p.add_argument("--window-bp", type=int, default=None)
    p.add_argument("--decay-bp", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--weight-mode", default="exp", choices=["exp", "power"])
    p.add_argument("--max-chrom-features-per-mark", type=int, default=15000)
    p.add_argument(
        "--max-edges-per-bin",
        type=int,
        default=0,
        help="Custom backend only: keep nearest K genes per bin; 0 disables pruning.",
    )
    p.add_argument("--add-self-loops", action="store_true", default=True)
    p.add_argument("--no-add-self-loops", action="store_false", dest="add_self_loops")
    p.add_argument("--no-export-graphml", action="store_true", default=True)
    p.add_argument("--export-graphml", action="store_false", dest="no_export_graphml")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    preprocess_dir = resolve_path(repo_root, args.preprocess_dir)
    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else preprocess_dir.parent / "graph"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rna_path = preprocess_dir / "rna_preprocessed.h5ad"
    chrom_manifest_path = preprocess_dir / "chrom_preprocessed_manifest.tsv"
    if not rna_path.exists() or not chrom_manifest_path.exists():
        raise FileNotFoundError(f"Missing preprocess outputs: {rna_path} / {chrom_manifest_path}")

    chrom_manifest = pd.read_csv(chrom_manifest_path, sep="\t")
    if chrom_manifest.empty:
        raise RuntimeError(f"No chromatin rows in {chrom_manifest_path}")

    rna = ad.read_h5ad(rna_path)
    gtf_path = resolve_path(repo_root, args.gtf)

    if args.backend == "scglue":
        rna_annot, ann_stats = try_annotate_rna_with_gtf(rna, gtf_path, annotation_by=args.annotation_by)
        rna_genes = set(map(str, rna_annot.var_names))
        genes_by_chr = None
    else:
        if args.gene_tss:
            gene_coords = load_gene_coord_table(resolve_path(repo_root, args.gene_tss))
        else:
            print("[joint-graph] Building gene coordinate table from GTF for custom backend...")
            gene_coords = build_gene_coord_from_gtf(gtf_path)

        genes_df, ann_stats = pick_rna_genes_custom(rna, gene_coords, annotation_by=args.annotation_by)
        genes_df = apply_gene_region_semantics(
            genes_df,
            gene_region=args.gene_region,
            promoter_len=int(args.promoter_len),
        )
        genes_by_chr = build_genes_by_chr(genes_df)
        rna_genes = set(map(str, genes_df["rna_gene"].astype(str)))
        ann_stats["rna_genes_after_gene_region_transform"] = float(len(rna_genes))
        ann_stats["gene_region"] = args.gene_region
        ann_stats["promoter_len"] = int(args.promoter_len)

    edges_path = out_dir / "guidance_edges.tsv"
    nodes_path = out_dir / "guidance_nodes.tsv"
    summary_path = out_dir / "guidance_summary.json"
    graphml_path = out_dir / "guidance_graph.graphml"

    node_rows = [{"node": g, "node_type": "gene", "modality": "rna", "mark": ""} for g in sorted(rna_genes)]
    seen_nodes = set(rna_genes)

    n_edges_total = 0
    per_mark_stats: List[dict] = []
    seen_gene_loops = set()

    if args.backend == "scglue":
        import scglue

    with edges_path.open("w", newline="") as ef:
        ew = csv.DictWriter(
            ef,
            fieldnames=["source", "target", "weight", "sign", "distance_bp", "mark", "type"],
            delimiter="\t",
        )
        ew.writeheader()

        for row in chrom_manifest.to_dict(orient="records"):
            mark = str(row["mark"])
            print(f"[joint-graph] Building mark graph: {mark} (backend={args.backend})")

            chrom_path = Path(str(row["chrom_h5ad"]))
            if not chrom_path.is_absolute():
                chrom_path = (repo_root / chrom_path).resolve()

            chrom = ad.read_h5ad(chrom_path)
            chrom, dropped_cap = subset_chrom_features(chrom, args.max_chrom_features_per_mark)
            chrom_nodes = set(map(str, chrom.var_names))

            for n in sorted(chrom_nodes):
                if n not in seen_nodes:
                    node_rows.append({"node": n, "node_type": "chromatin", "modality": "chromatin", "mark": mark})
                    seen_nodes.add(n)

            window_bp = args.window_bp if args.window_bp is not None else MARK_DEFAULT_WINDOW.get(mark, 150_000)
            decay_bp = args.decay_bp if args.decay_bp is not None else MARK_DEFAULT_DECAY.get(mark, 75_000)
            alpha = args.alpha if args.alpha is not None else MARK_DEFAULT_ALPHA.get(mark, 1.0)
            sign = MARK_SIGN.get(mark, +1)

            if args.backend == "scglue":
                chrom_scglue, dropped_parse = prepare_chrom_for_scglue_backend(chrom)

                def _extend_fn(dist_bp: int) -> float:
                    return edge_weight(
                        int(dist_bp),
                        alpha=float(alpha),
                        decay_bp=int(decay_bp),
                        weight_mode=args.weight_mode,
                    )

                graph = scglue.genomics.rna_anchored_prior_graph(
                    rna_annot,
                    chrom_scglue,
                    gene_region=args.gene_region,
                    promoter_len=args.promoter_len,
                    extend_range=window_bp,
                    extend_fn=_extend_fn,
                    signs=[int(sign)],
                    propagate_highly_variable=False,
                )

                mark_edges_added = 0
                connected_genes = set()
                connected_bins = set()
                for u, v, data in graph.edges(data=True):
                    src = str(u)
                    tgt = str(v)
                    row_out = {
                        "source": src,
                        "target": tgt,
                        "weight": float(data.get("weight", 1.0)),
                        "sign": int(data.get("sign", 1)),
                        "distance_bp": int(data.get("distance_bp", data.get("dist", 0))),
                        "mark": mark if (src in chrom_nodes or tgt in chrom_nodes) else "",
                        "type": str(data.get("type", "")),
                    }
                    ew.writerow(row_out)
                    mark_edges_added += 1
                    n_edges_total += 1
                    if (src in chrom_nodes and tgt in rna_genes) or (src in rna_genes and tgt in chrom_nodes):
                        connected_genes.update([x for x in (src, tgt) if x in rna_genes])
                        connected_bins.update([x for x in (src, tgt) if x in chrom_nodes])

                per_mark_stats.append(
                    {
                        "mark": mark,
                        "chrom_features_used": int(chrom_scglue.n_vars),
                        "chrom_features_dropped_by_cap": int(dropped_cap),
                        "chrom_features_dropped_by_parse": int(dropped_parse),
                        "graph_edges_added": int(mark_edges_added),
                        "connected_genes": int(len(connected_genes)),
                        "connected_bins": int(len(connected_bins)),
                        "window_bp": int(window_bp),
                        "decay_bp": int(decay_bp),
                        "alpha": float(alpha),
                        "sign": int(sign),
                    }
                )

                del graph
                del chrom_scglue
            else:
                stats = write_mark_edges_custom(
                    ew=ew,
                    mark=mark,
                    chrom=chrom,
                    genes_by_chr=genes_by_chr,
                    window_bp=int(window_bp),
                    decay_bp=int(decay_bp),
                    alpha=float(alpha),
                    sign=int(sign),
                    max_edges_per_bin=int(args.max_edges_per_bin),
                    add_self_loops=bool(args.add_self_loops),
                    seen_gene_loops=seen_gene_loops,
                    weight_mode=args.weight_mode,
                )
                n_edges_total += int(stats["graph_edges_added"])
                per_mark_stats.append(
                    {
                        "mark": mark,
                        "chrom_features_used": int(chrom.n_vars),
                        "chrom_features_dropped_by_cap": int(dropped_cap),
                        "chrom_features_dropped_by_parse": int(stats["chrom_features_dropped_by_parse"]),
                        "graph_edges_added": int(stats["graph_edges_added"]),
                        "connected_genes": int(stats["connected_genes"]),
                        "connected_bins": int(stats["connected_bins"]),
                        "candidate_pairs_before_prune": int(stats["candidate_pairs_before_prune"]),
                        "kept_pairs_after_prune": int(stats["kept_pairs_after_prune"]),
                        "window_bp": int(window_bp),
                        "decay_bp": int(decay_bp),
                        "alpha": float(alpha),
                        "sign": int(sign),
                        "weight_mode": args.weight_mode,
                        "max_edges_per_bin": int(args.max_edges_per_bin),
                    }
                )

            del chrom
            gc.collect()

    write_tsv(nodes_path, ["node", "node_type", "modality", "mark"], node_rows)

    if not args.no_export_graphml:
        print("[joint-graph] Exporting GraphML from TSV (memory intensive)...")
        write_graphml_from_tsv(nodes_path, edges_path, graphml_path)

    summary = {
        "run_type": "joint",
        "inputs": {
            "rna_h5ad": str(rna_path),
            "chrom_manifest_tsv": str(chrom_manifest_path),
            "gtf": str(gtf_path),
            "gene_tss": str(resolve_path(repo_root, args.gene_tss)) if args.gene_tss else None,
        },
        "params": {
            "backend": args.backend,
            "annotation_by": args.annotation_by,
            "gene_region": args.gene_region,
            "promoter_len": int(args.promoter_len),
            "window_bp_override": args.window_bp,
            "decay_bp_override": args.decay_bp,
            "alpha_override": args.alpha,
            "weight_mode": args.weight_mode,
            "max_chrom_features_per_mark": int(args.max_chrom_features_per_mark),
            "max_edges_per_bin": int(args.max_edges_per_bin),
            "add_self_loops": bool(args.add_self_loops),
            "export_graphml": bool(not args.no_export_graphml),
        },
        "graph": {
            "n_nodes_total": int(len(node_rows)),
            "n_edges_total": int(n_edges_total),
            "n_rna_genes": int(sum(1 for r in node_rows if r["node_type"] == "gene")),
            "n_chrom_features": int(sum(1 for r in node_rows if r["node_type"] == "chromatin")),
        },
        "backend_meta": ann_stats,
        "per_mark": per_mark_stats,
        "outputs": {
            "edges_tsv": str(edges_path),
            "nodes_tsv": str(nodes_path),
            "graphml": str(graphml_path) if not args.no_export_graphml else None,
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {edges_path}")
    print(f"Wrote: {nodes_path}")
    if not args.no_export_graphml:
        print(f"Wrote: {graphml_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
