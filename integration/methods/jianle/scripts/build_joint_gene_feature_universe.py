#!/usr/bin/env python3
"""Build gene-level histone matrices on a shared gene universe.

This script is the first step toward a paper-closer Jianle/scMRDR setup:
it converts each histone peak matrix into a cell-by-gene matrix using
GTF-derived gene regions and peak-to-gene links.

Outputs:
1. One shared gene universe table derived from RNA and/or linked GTF genes.
2. One gene-level sparse matrix per histone mark.
3. One per-mark feature-availability table aligned to the shared gene universe.
3. A manifest describing the generated per-mark matrices.
4. A JSON summary with link and shape statistics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as spio
from scipy import sparse as sp

from preprocess_joint_jianle import parse_marks
from preprocess_pilot_jianle import infer_repo_root, pick_rna_h5, read_named_column, resolve_path


ATTR_RE = re.compile(r'\s*([^\s]+)\s+"([^"]*)"\s*;')

MARK_DEFAULT_WINDOW = {
    "H3K27ac": 150_000,
    "H3K4me1": 150_000,
    "H3K4me2": 150_000,
    "H3K4me3": 25_000,
    "H3K27me3": 500_000,
    "H3K9me3": 500_000,
}

MARK_DEFAULT_DECAY = {
    "H3K27ac": 75_000,
    "H3K4me1": 75_000,
    "H3K4me2": 75_000,
    "H3K4me3": 10_000,
    "H3K27me3": 200_000,
    "H3K9me3": 200_000,
}

MARK_SIGN = {
    "H3K27ac": +1,
    "H3K4me1": +1,
    "H3K4me2": +1,
    "H3K4me3": +1,
    "H3K27me3": -1,
    "H3K9me3": -1,
}

GeneInterval = Tuple[str, int, int]  # (rna_gene, start, end)
PeakInterval = Tuple[int, int, int]  # (peak_idx, start, end)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def parse_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for m in ATTR_RE.finditer(attr_text):
        attrs[m.group(1)] = m.group(2)
    return attrs


def iter_gtf(path: Path) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts[:9]
            yield {
                "chrom": chrom,
                "feature": feature,
                "start": int(start),
                "end": int(end),
                "strand": strand,
                "attrs": parse_attrs(attrs),
            }


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

        start = int(rec["start"]) - 1
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
    df["_coord_order"] = np.arange(df.shape[0], dtype=int)
    return df


def pick_rna_genes_from_gtf(
    rna,
    gene_coords: pd.DataFrame,
    annotation_by: str,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    rna_var = rna.var.copy()
    rna_var = rna_var.reset_index().rename(columns={"index": "rna_gene"})
    rna_var["rna_gene"] = rna_var["rna_gene"].astype(str)
    rna_var["_rna_order"] = np.arange(rna_var.shape[0], dtype=int)

    by_name = rna_var.merge(
        gene_coords,
        left_on="rna_gene",
        right_on="gene_name",
        how="inner",
    )

    by_id = pd.DataFrame()
    if "gene_ids" in rna_var.columns:
        tmp = rna_var[["rna_gene", "_rna_order", "gene_ids"]].copy()
        tmp["gene_ids"] = tmp["gene_ids"].astype(str)
        tmp["gene_id_trim"] = tmp["gene_ids"].str.replace(r"\.\d+$", "", regex=True)

        coords = gene_coords.copy()
        coords["gene_id_trim"] = coords["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        by_id = tmp.merge(coords, on="gene_id_trim", how="inner")

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
            "Failed to map RNA genes to the provided GTF. "
            "Try a matching annotation build or set --annotation-by gene_name/gene_id explicitly."
        )

    chosen = chosen.sort_values("_rna_order").drop_duplicates("rna_gene").copy()
    chosen = chosen.loc[chosen["strand"].isin(["+", "-"])].copy()
    chosen = chosen.loc[chosen["chromEnd"] > chosen["chromStart"]].copy()

    out = chosen[
        [
            "rna_gene",
            "gene_name",
            "gene_id",
            "chrom",
            "chromStart",
            "chromEnd",
            "strand",
            "_rna_order",
            "_coord_order",
        ]
    ].copy()
    out["chrom"] = out["chrom"].astype(str)
    out["chromStart"] = out["chromStart"].astype(int)
    out["chromEnd"] = out["chromEnd"].astype(int)
    out["strand"] = out["strand"].astype(str)
    out["rna_gene"] = out["rna_gene"].astype(str)

    stats = {
        "annotation_mode": mode,
        "annotation_coverage_before_filter": float(out["rna_gene"].nunique() / max(1, rna.n_vars)),
        "rna_genes_after_coord_filter": float(out["rna_gene"].nunique()),
    }
    return out, stats


def trim_gene_id(gene_id: str) -> str:
    return re.sub(r"\.\d+$", "", str(gene_id))


def choose_unmapped_feature_id(
    gene_name: str,
    gene_id: str,
    used: set[str],
    coord_order: int,
) -> str:
    gene_name = str(gene_name or "").strip()
    gene_id = str(gene_id or "").strip()
    gene_id_trim = trim_gene_id(gene_id)

    candidates = [
        gene_name,
        gene_id_trim,
        gene_id,
        f"{gene_name}|{gene_id_trim}" if gene_name and gene_id_trim else "",
        f"gene_{coord_order + 1}",
    ]
    for cand in candidates:
        if cand and cand not in used:
            return cand

    base = gene_name or gene_id_trim or gene_id or f"gene_{coord_order + 1}"
    suffix = 1
    candidate = f"{base}.{suffix}"
    while candidate in used:
        suffix += 1
        candidate = f"{base}.{suffix}"
    return candidate


def assign_feature_ids(gene_coords: pd.DataFrame, rna_mapped: pd.DataFrame) -> pd.DataFrame:
    mapped = (
        rna_mapped.sort_values("_rna_order")
        .drop_duplicates("_coord_order")
        .set_index("_coord_order")["rna_gene"]
        .astype(str)
        .to_dict()
    )

    used: set[str] = set()
    feature_ids: List[str] = []
    rna_gene_col: List[str | None] = []
    present_in_rna: List[bool] = []

    for row in gene_coords.sort_values("_coord_order").itertuples(index=False):
        mapped_gene = mapped.get(int(row._coord_order))
        if mapped_gene:
            feature_id = str(mapped_gene)
            if feature_id in used:
                raise RuntimeError(
                    f"Duplicate RNA-mapped feature id detected while building shared universe: {feature_id}"
                )
            used.add(feature_id)
            feature_ids.append(feature_id)
            rna_gene_col.append(feature_id)
            present_in_rna.append(True)
            continue

        feature_id = choose_unmapped_feature_id(
            gene_name=str(row.gene_name),
            gene_id=str(row.gene_id),
            used=used,
            coord_order=int(row._coord_order),
        )
        used.add(feature_id)
        feature_ids.append(feature_id)
        rna_gene_col.append(None)
        present_in_rna.append(False)

    out = gene_coords.sort_values("_coord_order").copy()
    out["feature_id"] = feature_ids
    out["rna_gene"] = rna_gene_col
    out["present_in_rna"] = np.asarray(present_in_rna, dtype=bool)
    return out


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
        sort_cols = [c for c in ["chromStart", "chromEnd", "_coord_order"] if c in sub.columns]
        s = sub.sort_values(sort_cols)
        out[str(chrom)] = [
            (str(r.feature_id), int(r.chromStart), int(r.chromEnd))
            for r in s.itertuples(index=False)
        ]
    return out


def parse_interval_name(name: str) -> Tuple[str, int, int]:
    chrom, rest = str(name).split(":", 1)
    start_s, end_s = rest.split("-", 1)
    return chrom, int(start_s), int(end_s)


def parse_peak_features(features: Sequence[str]) -> tuple[List[Tuple[int, str, int, int]], int]:
    parsed: List[Tuple[int, str, int, int]] = []
    dropped = 0
    for i, feat in enumerate(map(str, features)):
        try:
            chrom, start, end = parse_interval_name(feat)
        except Exception:
            dropped += 1
            continue
        if end <= start:
            dropped += 1
            continue
        parsed.append((i, str(chrom), int(start), int(end)))
    return parsed, dropped


def build_peaks_by_chr(parsed_peaks: Iterable[Tuple[int, str, int, int]]) -> Dict[str, List[PeakInterval]]:
    out: Dict[str, List[PeakInterval]] = {}
    for peak_idx, chrom, start, end in parsed_peaks:
        out.setdefault(str(chrom), []).append((int(peak_idx), int(start), int(end)))
    for chrom in list(out.keys()):
        out[chrom].sort(key=lambda x: (x[1], x[2], x[0]))
    return out


def interval_dist(l_start: int, l_end: int, r_start: int, r_end: int) -> int:
    if l_start < r_end and r_start < l_end:
        return 0
    if l_end <= r_start:
        return l_end - r_start - 1
    return l_start - r_end + 1


def iter_window_pairs(
    genes_by_chr: Dict[str, List[GeneInterval]],
    peaks_by_chr: Dict[str, List[PeakInterval]],
    window_bp: int,
) -> Iterable[Tuple[str, int, int]]:
    for chrom, genes in genes_by_chr.items():
        peaks = peaks_by_chr.get(chrom)
        if not peaks:
            continue

        right_idx = 0
        window: List[PeakInterval] = []
        n_peaks = len(peaks)

        for gene, g_start, g_end in genes:
            broke = False
            i = 0
            while i < len(window):
                peak_idx, p_start, p_end = window[i]
                d = interval_dist(g_start, g_end, p_start, p_end)
                if -window_bp <= d <= window_bp:
                    yield gene, peak_idx, abs(int(d))
                    i += 1
                elif d > window_bp:
                    window.pop(i)
                else:
                    broke = True
                    break

            if not broke:
                while right_idx < n_peaks:
                    peak_idx, p_start, p_end = peaks[right_idx]
                    d = interval_dist(g_start, g_end, p_start, p_end)
                    if -window_bp <= d <= window_bp:
                        yield gene, peak_idx, abs(int(d))
                    elif d > window_bp:
                        right_idx += 1
                        continue

                    window.append((peak_idx, p_start, p_end))
                    right_idx += 1
                    if d < -window_bp:
                        break


def edge_weight(distance_bp: int, alpha: float, decay_bp: int, weight_mode: str) -> float:
    if weight_mode == "binary":
        return 1.0
    if weight_mode == "power":
        return float(alpha * (((float(distance_bp) + 1000.0) / 1000.0) ** (-0.75)))
    return float(alpha * math.exp(-float(distance_bp) / float(max(1, decay_bp))))


def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def write_gzip_tsv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def workspace_peak_paths(repo_root: Path, mark: str) -> Dict[str, Path]:
    base = repo_root / "integration/workspace/data/chromatin" / mark
    return {
        "peak_mtx": base / "chromatin_clean.mtx",
        "peak_barcodes": base / "chromatin_clean_barcodes.tsv",
        "peak_features": base / "chromatin_clean_features.tsv",
        "clean_cells": base / "clean_cells.tsv",
        "peaks_bed": base / "peaks.bed",
    }


def build_links_for_mark(
    peak_features: Sequence[str],
    genes_by_chr: Dict[str, List[GeneInterval]],
    gene_index: Dict[str, int],
    window_bp: int,
    decay_bp: int,
    alpha: float,
    weight_mode: str,
) -> tuple[List[int], List[int], List[float], List[dict], dict]:
    parsed_peaks, dropped_parse = parse_peak_features(peak_features)
    keep_peak_idx = [peak_idx for peak_idx, _, _, _ in parsed_peaks]
    peaks_by_chr = build_peaks_by_chr(parsed_peaks)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    link_rows: List[dict] = []
    linked_peak_indices = set()
    linked_gene_indices = set()
    gene_link_counts = np.zeros(len(gene_index), dtype=np.int32)

    for gene, peak_idx, dist in iter_window_pairs(genes_by_chr, peaks_by_chr, window_bp):
        g_idx = gene_index.get(gene)
        if g_idx is None:
            continue
        weight = edge_weight(dist, alpha=alpha, decay_bp=decay_bp, weight_mode=weight_mode)
        rows.append(int(peak_idx))
        cols.append(int(g_idx))
        data.append(float(weight))
        linked_peak_indices.add(int(peak_idx))
        linked_gene_indices.add(int(g_idx))
        gene_link_counts[int(g_idx)] += 1
        link_rows.append(
            {
                "peak_index": int(peak_idx),
                "peak_feature": str(peak_features[peak_idx]),
                "feature_id": str(gene),
                "gene_index": int(g_idx),
                "distance_bp": int(dist),
                "weight": float(weight),
            }
        )

    stats = {
        "n_peak_features_input": int(len(peak_features)),
        "n_peak_features_parseable": int(len(parsed_peaks)),
        "n_peak_features_dropped_parse": int(dropped_parse),
        "n_link_pairs": int(len(rows)),
        "n_linked_peaks": int(len(linked_peak_indices)),
        "n_linked_genes": int(len(linked_gene_indices)),
        "keep_peak_idx": keep_peak_idx,
        "linked_gene_indices": sorted(int(x) for x in linked_gene_indices),
        "gene_link_counts": gene_link_counts,
    }
    return rows, cols, data, link_rows, stats


def main() -> int:
    p = argparse.ArgumentParser(description="Build joint gene-level histone feature universe")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--rna-h5", default=None)
    p.add_argument("--gtf", required=True, help="GTF or GTF.GZ used to define the gene universe")
    p.add_argument("--annotation-by", default="auto", choices=["auto", "gene_name", "gene_id"])
    p.add_argument("--marks", default=None, help="Comma-separated mark subset (default: all marks in manifest)")
    p.add_argument("--run-id", default="gene_space_v1")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: integration/outputs/jianle/gene_features/<RUN_ID>",
    )
    p.add_argument(
        "--gene-universe-mode",
        default="rna",
        choices=["rna", "union_linked", "gtf_all"],
        help=(
            "Shared feature universe choice: "
            "'rna' keeps only RNA-mapped genes, "
            "'union_linked' keeps RNA genes plus genes linked in at least one selected mark, "
            "'gtf_all' keeps all parsed GTF genes after region processing."
        ),
    )

    p.add_argument("--gene-region", default="promoter", choices=["gene_body", "promoter", "combined"])
    p.add_argument("--promoter-len", type=int, default=2000)
    p.add_argument(
        "--window-bp",
        type=int,
        default=None,
        help="Distance extension around each gene region. Default: mark-specific window.",
    )
    p.add_argument(
        "--weight-mode",
        default="binary",
        choices=["binary", "power", "exp"],
        help="Peak-to-gene link weighting scheme",
    )
    p.add_argument(
        "--decay-bp",
        type=int,
        default=None,
        help="Decay scale for --weight-mode exp; default: mark-specific decay.",
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Base link weight multiplier")
    p.add_argument("--export-links", action="store_true")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_path(repo_root, args.manifest)
    gtf_path = resolve_path(repo_root, args.gtf)
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "mark" not in manifest_df.columns:
        raise ValueError(f"Manifest missing 'mark' column: {manifest_path}")
    marks = parse_marks(manifest_df, args.marks)

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root / f"integration/outputs/jianle/gene_features/{args.run_id}"
    )
    matrices_dir = out_dir / "matrices"
    links_dir = out_dir / "links"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    if args.export_links:
        links_dir.mkdir(parents=True, exist_ok=True)

    rna_h5_path = pick_rna_h5(repo_root, args.rna_h5)
    rna = sc.read_10x_h5(rna_h5_path)

    gene_coords = build_gene_coord_from_gtf(gtf_path)
    rna_mapped_df, ann_stats = pick_rna_genes_from_gtf(rna, gene_coords, annotation_by=args.annotation_by)
    candidate_genes_df = assign_feature_ids(gene_coords, rna_mapped_df)
    candidate_genes_df = apply_gene_region_semantics(
        candidate_genes_df,
        gene_region=args.gene_region,
        promoter_len=int(args.promoter_len),
    )
    candidate_genes_df = candidate_genes_df.sort_values("_coord_order").drop_duplicates("feature_id").copy()

    if args.gene_universe_mode == "rna":
        working_genes_df = candidate_genes_df.loc[candidate_genes_df["present_in_rna"]].copy()
    else:
        working_genes_df = candidate_genes_df.copy()

    if working_genes_df.empty:
        raise RuntimeError("No genes remained after GTF mapping and gene-region processing")

    working_genes_df = working_genes_df.reset_index(drop=True)
    working_gene_order = working_genes_df["feature_id"].astype(str).tolist()
    gene_index = {gene: i for i, gene in enumerate(working_gene_order)}
    genes_by_chr = build_genes_by_chr(working_genes_df)
    mapped_gene_indices = np.flatnonzero(working_genes_df["present_in_rna"].to_numpy(dtype=bool))

    manifest_rows: List[dict] = []
    mark_summaries: List[dict] = []
    mark_buffers: List[dict] = []
    globally_linked_gene_indices: set[int] = set()

    for mark in marks:
        paths = workspace_peak_paths(repo_root, mark)
        missing = [str(pth) for pth in paths.values() if not pth.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing peak-matrix inputs for mark {mark}. "
                f"Expected workspace files are absent: {missing}"
            )

        peak_mtx = spio.mmread(paths["peak_mtx"]).tocsr().astype(np.float32)
        peak_barcodes = read_named_column(paths["peak_barcodes"], "barcode")
        peak_features = read_named_column(paths["peak_features"], "feature").astype(str)

        if peak_mtx.shape[0] != len(peak_barcodes):
            raise ValueError(f"{mark}: barcode length mismatch with matrix rows")
        if peak_mtx.shape[1] != len(peak_features):
            raise ValueError(f"{mark}: feature length mismatch with matrix columns")

        window_bp = int(args.window_bp) if args.window_bp is not None else int(MARK_DEFAULT_WINDOW.get(mark, 150_000))
        decay_bp = int(args.decay_bp) if args.decay_bp is not None else int(MARK_DEFAULT_DECAY.get(mark, 75_000))

        rows, cols, data, link_rows, link_stats = build_links_for_mark(
            peak_features=peak_features.tolist(),
            genes_by_chr=genes_by_chr,
            gene_index=gene_index,
            window_bp=window_bp,
            decay_bp=decay_bp,
            alpha=float(args.alpha),
            weight_mode=args.weight_mode,
        )
        keep_peak_idx = np.asarray(link_stats.pop("keep_peak_idx"), dtype=int)
        linked_gene_indices = np.asarray(link_stats.pop("linked_gene_indices"), dtype=int)
        gene_link_counts = np.asarray(link_stats.pop("gene_link_counts"), dtype=np.int32)
        globally_linked_gene_indices.update(linked_gene_indices.tolist())

        if rows:
            peak_mtx = peak_mtx[:, keep_peak_idx]
            remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_peak_idx.tolist())}
            remapped_rows = [remap[int(old_idx)] for old_idx in rows]
            link_mat = sp.csr_matrix(
                (
                    np.asarray(data, dtype=np.float32),
                    (np.asarray(remapped_rows, dtype=int), np.asarray(cols, dtype=int)),
                ),
                shape=(peak_mtx.shape[1], len(working_gene_order)),
                dtype=np.float32,
            )
            gene_mtx = (peak_mtx @ link_mat).tocsr().astype(np.float32)
            gene_mtx.eliminate_zeros()
        else:
            gene_mtx = sp.csr_matrix((peak_mtx.shape[0], len(working_gene_order)), dtype=np.float32)

        mark_buffers.append(
            {
                "mark": mark,
                "paths": paths,
                "peak_barcodes": peak_barcodes.astype(str).tolist(),
                "window_bp": int(window_bp),
                "decay_bp": int(decay_bp),
                "gene_mtx": gene_mtx,
                "gene_link_counts": gene_link_counts,
                "link_rows": link_rows,
                "link_stats": link_stats,
                "n_input_peak_features": int(len(peak_features)),
                "n_kept_peak_features_after_parse": int(len(keep_peak_idx)),
            }
        )

    if args.gene_universe_mode == "rna":
        retained_gene_idx = mapped_gene_indices.astype(int)
    elif args.gene_universe_mode == "union_linked":
        retained_gene_idx = np.asarray(
            sorted(set(mapped_gene_indices.tolist()) | globally_linked_gene_indices),
            dtype=int,
        )
    else:
        retained_gene_idx = np.arange(len(working_gene_order), dtype=int)

    if retained_gene_idx.size == 0:
        raise RuntimeError("Shared gene universe ended up empty after applying the requested universe mode")

    final_genes_df = working_genes_df.iloc[retained_gene_idx].copy().reset_index(drop=True)
    final_genes_df["gene_index"] = np.arange(final_genes_df.shape[0], dtype=int)
    final_gene_order = final_genes_df["feature_id"].astype(str).tolist()

    gene_universe_tsv = out_dir / "gene_universe.tsv"
    final_genes_df[
        [
            "feature_id",
            "rna_gene",
            "present_in_rna",
            "gene_name",
            "gene_id",
            "chrom",
            "chromStart",
            "chromEnd",
            "strand",
            "gene_index",
        ]
    ].to_csv(gene_universe_tsv, sep="\t", index=False)

    for buf in mark_buffers:
        mark = str(buf["mark"])
        gene_mtx = buf["gene_mtx"][:, retained_gene_idx].tocsr().astype(np.float32)
        gene_mtx.eliminate_zeros()
        gene_link_counts = np.asarray(buf["gene_link_counts"], dtype=np.int32)[retained_gene_idx]
        feature_available = gene_link_counts > 0

        mtx_out = matrices_dir / f"{mark}_gene_scores.mtx"
        barcodes_out = matrices_dir / f"{mark}_gene_scores_barcodes.tsv"
        features_out = matrices_dir / f"{mark}_gene_scores_features.tsv"
        availability_out = matrices_dir / f"{mark}_gene_scores_availability.tsv"

        spio.mmwrite(str(mtx_out), gene_mtx.tocoo())
        barcodes_out.write_text("barcode\n" + "\n".join(buf["peak_barcodes"]) + "\n")
        features_out.write_text("feature\n" + "\n".join(final_gene_order) + "\n")
        pd.DataFrame(
            {
                "feature": final_gene_order,
                "feature_available": feature_available.astype(int),
                "n_peak_links": gene_link_counts.astype(int),
            }
        ).to_csv(availability_out, sep="\t", index=False)

        if args.export_links:
            link_out = links_dir / f"{mark}_peak_to_gene_links.tsv.gz"
            write_gzip_tsv(
                link_out,
                (
                    {
                        **row,
                        "mark": mark,
                    }
                    for row in buf["link_rows"]
                ),
                ["mark", "peak_index", "peak_feature", "feature_id", "gene_index", "distance_bp", "weight"],
            )

        manifest_rows.append(
            {
                "mark": mark,
                "modality_key": f"chrom_{mark}",
                "gene_universe": str(gene_universe_tsv),
                "gene_universe_mode": args.gene_universe_mode,
                "gene_mtx": str(mtx_out),
                "gene_barcodes": str(barcodes_out),
                "gene_features": str(features_out),
                "gene_availability": str(availability_out),
                "clean_cells": str(buf["paths"]["clean_cells"]),
                "feature_space": "gene",
                "source_feature_space": "peak",
                "gene_region": args.gene_region,
                "promoter_len": int(args.promoter_len),
                "window_bp": int(buf["window_bp"]),
                "weight_mode": args.weight_mode,
                "decay_bp": int(buf["decay_bp"]),
                "sign": int(MARK_SIGN.get(mark, +1)),
                "n_cells": int(gene_mtx.shape[0]),
                "n_genes": int(gene_mtx.shape[1]),
                "n_available_genes": int(feature_available.sum()),
                "nnz": int(gene_mtx.nnz),
            }
        )

        mark_summaries.append(
            {
                "mark": mark,
                "source_paths": {k: str(v) for k, v in buf["paths"].items()},
                "window_bp": int(buf["window_bp"]),
                "decay_bp": int(buf["decay_bp"]),
                "alpha": float(args.alpha),
                "weight_mode": args.weight_mode,
                "n_input_cells": int(len(buf["peak_barcodes"])),
                "n_input_peak_features": int(buf["n_input_peak_features"]),
                "n_kept_peak_features_after_parse": int(buf["n_kept_peak_features_after_parse"]),
                "n_output_genes": int(gene_mtx.shape[1]),
                "n_available_genes": int(feature_available.sum()),
                "nnz_output": int(gene_mtx.nnz),
                **buf["link_stats"],
            }
        )

    manifest_tsv = out_dir / "joint_gene_feature_manifest.tsv"
    write_tsv(
        manifest_tsv,
        manifest_rows,
        [
            "mark",
            "modality_key",
            "gene_universe",
            "gene_universe_mode",
            "gene_mtx",
            "gene_barcodes",
            "gene_features",
            "gene_availability",
            "clean_cells",
            "feature_space",
            "source_feature_space",
            "gene_region",
            "promoter_len",
            "window_bp",
            "weight_mode",
            "decay_bp",
            "sign",
            "n_cells",
            "n_genes",
            "n_available_genes",
            "nnz",
        ],
    )

    summary = {
        "run_id": args.run_id,
        "manifest": str(manifest_path),
        "gtf": str(gtf_path),
        "rna_h5": str(rna_h5_path),
        "marks": marks,
        "gene_universe": {
            "path": str(gene_universe_tsv),
            "mode": args.gene_universe_mode,
            "n_candidate_genes_before_universe_filter": int(working_genes_df.shape[0]),
            "n_genes": int(len(final_gene_order)),
            "n_genes_present_in_rna": int(final_genes_df["present_in_rna"].sum()),
            **ann_stats,
            "gene_region": args.gene_region,
            "promoter_len": int(args.promoter_len),
        },
        "per_mark": mark_summaries,
        "outputs": {
            "out_dir": str(out_dir),
            "manifest_tsv": str(manifest_tsv),
            "gene_universe_tsv": str(gene_universe_tsv),
        },
        "params": vars(args),
    }
    summary_json = out_dir / "gene_feature_summary.json"
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {gene_universe_tsv}")
    print(f"Wrote: {manifest_tsv}")
    print(f"Wrote: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
