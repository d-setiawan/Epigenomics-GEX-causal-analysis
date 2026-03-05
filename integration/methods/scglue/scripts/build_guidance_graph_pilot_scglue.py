#!/usr/bin/env python3
"""Build a pilot guidance graph for scGLUE from preprocessed RNA/chromatin data.

Default backend uses scGLUE native utilities:
- scglue.data.get_gene_annotation
- scglue.genomics.rna_anchored_prior_graph

A custom fallback backend is also available for environments where scGLUE
graph utilities are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


MARK_SIGN = {
    "H3K27ac": +1,
    "H3K4me1": +1,
    "H3K4me2": +1,
    "H3K4me3": +1,
    "H3K27me3": -1,
    "H3K9me3": -1,
}

MARK_DEFAULT_WINDOW = {
    "H3K27ac": 150_000,
    "H3K4me1": 150_000,
    "H3K4me2": 150_000,
    "H3K4me3": 150_000,
    "H3K27me3": 500_000,
    "H3K9me3": 500_000,
}

MARK_DEFAULT_DECAY = {
    "H3K27ac": 75_000,
    "H3K4me1": 75_000,
    "H3K4me2": 75_000,
    "H3K4me3": 75_000,
    "H3K27me3": 200_000,
    "H3K9me3": 200_000,
}

MARK_DEFAULT_ALPHA = {
    "H3K27ac": 1.0,
    "H3K4me1": 1.0,
    "H3K4me2": 1.0,
    "H3K4me3": 1.0,
    "H3K27me3": 0.7,
    "H3K9me3": 0.7,
}


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def parse_bin_name(name: str) -> Tuple[str, int, int]:
    # Expect: chr:start-end
    chrom, rest = name.split(":", 1)
    start_s, end_s = rest.split("-", 1)
    return chrom, int(start_s), int(end_s)


def load_manifest_row(manifest_path: Path, mark: str) -> Dict[str, str]:
    df = pd.read_csv(manifest_path, sep="\t")
    hit = df.loc[df["mark"] == mark]
    if hit.empty:
        raise ValueError(f"Mark '{mark}' not found in manifest: {manifest_path}")
    return hit.iloc[0].to_dict()


def write_tsv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def prepare_chrom_bed_columns(chrom: ad.AnnData) -> tuple[ad.AnnData, int]:
    chrom = chrom.copy()
    parsed = []
    keep_idx = []
    for i, feat in enumerate(map(str, chrom.var_names)):
        try:
            c, s, e = parse_bin_name(feat)
        except Exception:
            continue
        keep_idx.append(i)
        parsed.append((c, s, e))

    if len(keep_idx) < chrom.n_vars:
        chrom = chrom[:, keep_idx].copy()

    chrom.var["chrom"] = [x[0] for x in parsed]
    chrom.var["chromStart"] = np.asarray([x[1] for x in parsed], dtype=int)
    chrom.var["chromEnd"] = np.asarray([x[2] for x in parsed], dtype=int)
    chrom.var["strand"] = "."
    dropped = int(len(keep_idx) != len(parsed) or (len(keep_idx) < chrom.n_vars))
    return chrom, dropped


def subset_chrom_features(chrom: ad.AnnData, max_features: int | None) -> tuple[ad.AnnData, int]:
    if max_features is None or max_features <= 0 or chrom.n_vars <= max_features:
        return chrom, 0
    x = chrom.layers["counts"] if "counts" in chrom.layers else chrom.X
    sums = np.asarray(x.sum(axis=0)).ravel()
    idx = np.argpartition(sums, -max_features)[-max_features:]
    idx = np.sort(idx)
    return chrom[:, idx].copy(), int(chrom.n_vars - max_features)


def try_annotate_rna_with_gtf(
    rna: ad.AnnData,
    gtf_path: Path,
    annotation_by: str,
) -> tuple[ad.AnnData, Dict[str, float]]:
    import scglue

    mode_candidates: List[str]
    if annotation_by == "auto":
        mode_candidates = ["gene_id", "gene_name"]
    else:
        mode_candidates = [annotation_by]

    best_rna = None
    best_mode = None
    best_cov = -1.0

    for mode in mode_candidates:
        candidate = rna.copy()
        try:
            if mode == "gene_id":
                if "gene_ids" not in candidate.var.columns:
                    continue
                scglue.data.get_gene_annotation(
                    candidate,
                    var_by="gene_ids",
                    gtf=gtf_path,
                    gtf_by="gene_id",
                    by_func=scglue.genomics.ens_trim_version,
                )
            elif mode == "gene_name":
                scglue.data.get_gene_annotation(
                    candidate,
                    gtf=gtf_path,
                    gtf_by="gene_name",
                )
            else:
                raise ValueError(f"Unsupported annotation mode: {mode}")
        except Exception:
            continue

        if "chrom" not in candidate.var.columns:
            continue
        cov = float(candidate.var["chrom"].notna().mean())
        if cov > best_cov:
            best_cov = cov
            best_mode = mode
            best_rna = candidate

    if best_rna is None or best_cov <= 0:
        raise RuntimeError(
            "Failed to annotate RNA genes from GTF using scGLUE native utility. "
            "Check genome build and whether RNA features match gene_name/gene_id in GTF."
        )

    # Keep only genes with valid genomic coordinates and strand for promoter expansion.
    required = ["chrom", "chromStart", "chromEnd", "strand"]
    mask = np.ones(best_rna.n_vars, dtype=bool)
    for col in required:
        if col not in best_rna.var.columns:
            raise RuntimeError(f"Missing required column after annotation: {col}")
        mask &= best_rna.var[col].notna().to_numpy()

    best_rna = best_rna[:, mask].copy()
    best_rna.var["chromStart"] = best_rna.var["chromStart"].astype(int)
    best_rna.var["chromEnd"] = best_rna.var["chromEnd"].astype(int)
    best_rna.var["strand"] = best_rna.var["strand"].astype(str)
    best_rna = best_rna[:, best_rna.var["strand"].isin(["+", "-"])].copy()

    stats = {
        "annotation_mode": best_mode,
        "annotation_coverage_before_filter": float(best_cov),
        "rna_genes_after_coord_filter": float(best_rna.n_vars),
    }
    return best_rna, stats


def build_graph_scglue_native(
    rna: ad.AnnData,
    chrom: ad.AnnData,
    mark: str,
    gtf_path: Path,
    annotation_by: str,
    window_bp: int,
    decay_bp: int,
    alpha: float,
    sign: int,
    gene_region: str,
    promoter_len: int,
) -> tuple[nx.MultiDiGraph, Dict[str, float]]:
    import scglue

    rna_annot, ann_stats = try_annotate_rna_with_gtf(rna, gtf_path, annotation_by=annotation_by)
    chrom_bed, _ = prepare_chrom_bed_columns(chrom)

    def _extend_fn(dist_bp: int) -> float:
        return float(alpha * math.exp(-float(dist_bp) / float(max(1, decay_bp))))

    graph = scglue.genomics.rna_anchored_prior_graph(
        rna_annot,
        chrom_bed,
        gene_region=gene_region,
        promoter_len=promoter_len,
        extend_range=window_bp,
        extend_fn=_extend_fn,
        signs=[int(sign)],
        propagate_highly_variable=False,
    )

    # Add node attributes for easier downstream parsing.
    rna_set = set(map(str, rna_annot.var_names))
    chrom_set = set(map(str, chrom_bed.var_names))
    for n in graph.nodes:
        if n in rna_set:
            graph.nodes[n]["node_type"] = "gene"
            graph.nodes[n]["modality"] = "rna"
        elif n in chrom_set:
            graph.nodes[n]["node_type"] = "chromatin"
            graph.nodes[n]["modality"] = "chromatin"
            graph.nodes[n]["mark"] = mark
        else:
            graph.nodes[n]["node_type"] = "unknown"
            graph.nodes[n]["modality"] = "unknown"

    meta = {
        **ann_stats,
        "backend": "scglue",
        "rna_genes_used": float(len(rna_set)),
        "chrom_features_used": float(len(chrom_set)),
    }
    return graph, meta


def load_gene_tss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"chrom", "tss", "gene_name", "gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gene_tss missing columns: {sorted(missing)}")
    df = df.copy()
    df["chrom"] = df["chrom"].astype(str)
    df["tss"] = df["tss"].astype(int)
    df["gene_name"] = df["gene_name"].astype(str)
    df["gene_id"] = df["gene_id"].astype(str)
    return df


def pick_rna_genes_custom(rna: ad.AnnData, gene_tss: pd.DataFrame) -> pd.DataFrame:
    rna_names = set(map(str, rna.var_names))
    rna_ids = set(map(str, rna.var["gene_ids"].fillna(""))) if "gene_ids" in rna.var.columns else set()

    by_name = gene_tss[gene_tss["gene_name"].isin(rna_names)].copy()
    by_name["rna_gene"] = by_name["gene_name"]
    if not by_name.empty:
        return by_name

    by_id = gene_tss[gene_tss["gene_id"].isin(rna_ids)].copy()
    by_id["rna_gene"] = by_id["gene_id"]
    return by_id


def build_edges_custom(
    chrom_names: Iterable[str],
    genes_df: pd.DataFrame,
    mark: str,
    window_bp: int,
    decay_bp: int,
    alpha: float,
) -> List[dict]:
    chrom_records: Dict[str, List[Tuple[str, int]]] = {}
    for feat in chrom_names:
        try:
            chrom, start, end = parse_bin_name(str(feat))
        except Exception:
            continue
        center = (start + end) // 2
        chrom_records.setdefault(chrom, []).append((str(feat), center))

    sign = MARK_SIGN.get(mark, +1)
    edges: List[dict] = []
    genes_by_chr = {chrom: sub.sort_values("tss") for chrom, sub in genes_df.groupby("chrom", sort=False)}

    for chrom, bins in chrom_records.items():
        g = genes_by_chr.get(chrom)
        if g is None or g.empty:
            continue
        tss = g["tss"].to_numpy(dtype=np.int64)
        names = g["rna_gene"].to_numpy(dtype=object)

        for chrom_feat, center in bins:
            left = np.searchsorted(tss, center - window_bp, side="left")
            right = np.searchsorted(tss, center + window_bp, side="right")
            if left >= right:
                continue
            local_tss = tss[left:right]
            local_names = names[left:right]
            dists = np.abs(local_tss - center)
            for gene, dist in zip(local_names, dists):
                weight = float(alpha * math.exp(-float(dist) / float(max(1, decay_bp))))
                edges.append(
                    {
                        "source": str(chrom_feat),
                        "target": str(gene),
                        "weight": weight,
                        "sign": int(sign),
                        "distance_bp": int(dist),
                        "mark": mark,
                        "type": "fwd",
                    }
                )
    return edges


def build_graph_custom(
    rna: ad.AnnData,
    chrom: ad.AnnData,
    mark: str,
    gene_tss_path: Path,
    window_bp: int,
    decay_bp: int,
    alpha: float,
    sign: int,
    max_edges_per_bin: int,
    add_self_loops: bool,
) -> tuple[nx.MultiDiGraph, Dict[str, float]]:
    gene_tss = load_gene_tss(gene_tss_path)
    genes_df = pick_rna_genes_custom(rna, gene_tss)
    if genes_df.empty:
        raise RuntimeError("No overlap between RNA genes and gene_tss table for custom backend")

    edges = build_edges_custom(chrom.var_names, genes_df, mark, window_bp, decay_bp, alpha)
    if max_edges_per_bin > 0 and edges:
        by_bin: Dict[str, List[dict]] = {}
        for e in edges:
            by_bin.setdefault(e["source"], []).append(e)
        trimmed: List[dict] = []
        for _, lst in by_bin.items():
            trimmed.extend(sorted(lst, key=lambda x: x["distance_bp"])[:max_edges_per_bin])
        edges = trimmed

    for e in edges:
        e["sign"] = int(sign)

    g = nx.MultiDiGraph()
    for n in map(str, rna.var_names):
        g.add_node(n, node_type="gene", modality="rna")
    for n in map(str, chrom.var_names):
        g.add_node(n, node_type="chromatin", modality="chromatin", mark=mark)

    for e in edges:
        g.add_edge(e["target"], e["source"], **e)
        g.add_edge(e["source"], e["target"], weight=e["weight"], sign=e["sign"], distance_bp=e["distance_bp"], mark=mark, type="rev")

    if add_self_loops:
        for node in g.nodes:
            g.add_edge(node, node, weight=1.0, sign=1, distance_bp=0, mark="self", type="loop")

    meta = {
        "backend": "custom",
        "gene_overlap_with_tss": float(genes_df["rna_gene"].nunique()),
    }
    return g, meta


def graph_to_rows(graph: nx.MultiDiGraph) -> tuple[List[dict], List[dict]]:
    edge_rows = []
    for u, v, data in graph.edges(data=True):
        edge_rows.append(
            {
                "source": str(u),
                "target": str(v),
                "weight": float(data.get("weight", 1.0)),
                "sign": int(data.get("sign", 1)),
                "distance_bp": int(data.get("distance_bp", 0)),
                "mark": str(data.get("mark", "")),
                "type": str(data.get("type", "")),
            }
        )

    node_rows = []
    for n, attrs in graph.nodes(data=True):
        node_rows.append(
            {
                "node": str(n),
                "node_type": str(attrs.get("node_type", "")),
                "modality": str(attrs.get("modality", "")),
                "mark": str(attrs.get("mark", "")),
            }
        )

    return edge_rows, node_rows


def main() -> int:
    p = argparse.ArgumentParser(description="Build pilot guidance graph for scGLUE")
    p.add_argument("--mark", required=True)
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--preprocess-dir", default=None, help="Default: integration/outputs/scglue/pilot/<MARK>/preprocess")
    p.add_argument("--out-dir", default=None, help="Default: integration/outputs/scglue/pilot/<MARK>/graph")

    p.add_argument("--backend", default="scglue", choices=["scglue", "custom"])
    p.add_argument("--gtf", default=None, help="GTF path (required for --backend scglue)")
    p.add_argument("--annotation-by", default="auto", choices=["auto", "gene_id", "gene_name"])
    p.add_argument("--gene-tss", default=None, help="gene_tss TSV (required for --backend custom)")

    p.add_argument("--gene-region", default="combined", choices=["gene_body", "promoter", "combined"])
    p.add_argument("--promoter-len", type=int, default=2000)

    p.add_argument("--window-bp", type=int, default=None)
    p.add_argument("--decay-bp", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--sign", type=int, default=None, choices=[-1, 1])
    p.add_argument(
        "--max-chrom-features",
        type=int,
        default=None,
        help="Cap chromatin features for graph building (top features by count)",
    )
    p.add_argument("--max-edges-per-bin", type=int, default=30)
    p.add_argument("--export-graphml", action="store_true", default=True)
    p.add_argument("--no-export-graphml", action="store_false", dest="export_graphml")
    p.add_argument("--add-self-loops", action="store_true", default=True)
    p.add_argument("--no-add-self-loops", action="store_false", dest="add_self_loops")

    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_path(repo_root, args.manifest)
    _ = load_manifest_row(manifest_path, args.mark)  # validate mark exists

    preprocess_dir = resolve_path(repo_root, args.preprocess_dir) if args.preprocess_dir else repo_root / f"integration/outputs/scglue/pilot/{args.mark}/preprocess"
    out_dir = resolve_path(repo_root, args.out_dir) if args.out_dir else repo_root / f"integration/outputs/scglue/pilot/{args.mark}/graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    rna_path = preprocess_dir / "rna_preprocessed.h5ad"
    chrom_path = preprocess_dir / f"chrom_{args.mark}_preprocessed.h5ad"
    if not rna_path.exists() or not chrom_path.exists():
        raise FileNotFoundError(f"Missing preprocessed inputs. Expected {rna_path} and {chrom_path}")

    rna = ad.read_h5ad(rna_path)
    chrom = ad.read_h5ad(chrom_path)
    chrom, dropped_chrom_features = subset_chrom_features(chrom, args.max_chrom_features)

    window_bp = args.window_bp if args.window_bp is not None else MARK_DEFAULT_WINDOW.get(args.mark, 150_000)
    decay_bp = args.decay_bp if args.decay_bp is not None else MARK_DEFAULT_DECAY.get(args.mark, 75_000)
    alpha = args.alpha if args.alpha is not None else MARK_DEFAULT_ALPHA.get(args.mark, 1.0)
    sign = args.sign if args.sign is not None else MARK_SIGN.get(args.mark, +1)

    if args.backend == "scglue":
        if not args.gtf:
            raise ValueError("--gtf is required for --backend scglue")
        gtf_path = resolve_path(repo_root, args.gtf)
        graph, meta = build_graph_scglue_native(
            rna=rna,
            chrom=chrom,
            mark=args.mark,
            gtf_path=gtf_path,
            annotation_by=args.annotation_by,
            window_bp=window_bp,
            decay_bp=decay_bp,
            alpha=alpha,
            sign=sign,
            gene_region=args.gene_region,
            promoter_len=args.promoter_len,
        )
        if args.max_edges_per_bin != 30:
            print("[WARN] --max-edges-per-bin is ignored in scglue backend")
    else:
        if not args.gene_tss:
            raise ValueError("--gene-tss is required for --backend custom")
        gene_tss_path = resolve_path(repo_root, args.gene_tss)
        graph, meta = build_graph_custom(
            rna=rna,
            chrom=chrom,
            mark=args.mark,
            gene_tss_path=gene_tss_path,
            window_bp=window_bp,
            decay_bp=decay_bp,
            alpha=alpha,
            sign=sign,
            max_edges_per_bin=args.max_edges_per_bin,
            add_self_loops=args.add_self_loops,
        )

    edge_rows, node_rows = graph_to_rows(graph)

    edges_path = out_dir / "guidance_edges.tsv"
    nodes_path = out_dir / "guidance_nodes.tsv"
    graphml_path = out_dir / "guidance_graph.graphml"
    summary_path = out_dir / "guidance_summary.json"

    write_tsv(edges_path, ["source", "target", "weight", "sign", "distance_bp", "mark", "type"], edge_rows)
    write_tsv(nodes_path, ["node", "node_type", "modality", "mark"], node_rows)
    if args.export_graphml:
        nx.write_graphml(graph, graphml_path)

    prior_edges = [e for e in edge_rows if e["type"] in {"fwd", "rev", "prior"}]
    connected_genes = {
        e["source"] for e in prior_edges if next((n for n in node_rows if n["node"] == e["source"] and n["node_type"] == "gene"), None)
    } | {
        e["target"] for e in prior_edges if next((n for n in node_rows if n["node"] == e["target"] and n["node_type"] == "gene"), None)
    }
    connected_bins = {
        e["source"] for e in prior_edges if next((n for n in node_rows if n["node"] == e["source"] and n["node_type"] == "chromatin"), None)
    } | {
        e["target"] for e in prior_edges if next((n for n in node_rows if n["node"] == e["target"] and n["node_type"] == "chromatin"), None)
    }

    summary = {
        "mark": args.mark,
        "inputs": {
            "rna_h5ad": str(rna_path),
            "chrom_h5ad": str(chrom_path),
            "manifest": str(manifest_path),
            "gtf": str(resolve_path(repo_root, args.gtf)) if args.gtf else None,
            "gene_tss": str(resolve_path(repo_root, args.gene_tss)) if args.gene_tss else None,
        },
        "params": {
            "backend": args.backend,
            "annotation_by": args.annotation_by,
            "gene_region": args.gene_region,
            "promoter_len": int(args.promoter_len),
            "window_bp": int(window_bp),
            "decay_bp": int(decay_bp),
            "alpha": float(alpha),
            "sign": int(sign),
            "max_edges_per_bin": int(args.max_edges_per_bin),
            "max_chrom_features": int(args.max_chrom_features) if args.max_chrom_features else None,
            "export_graphml": bool(args.export_graphml),
            "add_self_loops": bool(args.add_self_loops),
        },
        "graph": {
            "n_nodes_total": int(graph.number_of_nodes()),
            "n_edges_total": int(graph.number_of_edges()),
            "n_prior_edges": int(len(prior_edges)),
            "n_connected_genes": int(len(connected_genes)),
            "n_connected_bins": int(len(connected_bins)),
            "rna_genes_total": int(rna.n_vars),
            "chrom_features_total": int(chrom.n_vars),
        },
        "backend_meta": meta,
        "resource": {
            "chrom_features_dropped_by_cap": int(dropped_chrom_features),
        },
        "outputs": {
            "edges_tsv": str(edges_path),
            "nodes_tsv": str(nodes_path),
            "graphml": str(graphml_path) if args.export_graphml else None,
            "summary_json": str(summary_path),
        },
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {edges_path}")
    print(f"Wrote: {nodes_path}")
    print(f"Wrote: {graphml_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
