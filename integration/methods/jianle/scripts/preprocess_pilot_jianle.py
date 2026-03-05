#!/usr/bin/env python3
"""Pilot preprocessing for Jianle-track integration (step 4 only).

This script prepares one RNA dataset and one chromatin mark dataset and writes
method-ready pilot inputs without running integration itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as spio
from sklearn.decomposition import TruncatedSVD


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def strip_10x_suffix(barcodes: pd.Series) -> pd.Series:
    return barcodes.astype(str).str.replace(r"-\d+$", "", regex=True)


def pick_rna_h5(repo_root: Path, rna_h5: Optional[str]) -> Path:
    if rna_h5:
        p = resolve_path(repo_root, rna_h5)
        if not p.exists():
            raise FileNotFoundError(f"RNA h5 not found: {p}")
        return p

    candidates = [
        repo_root / "integration/workspace/data/rna/gex_feature_cell_matrix.h5",
        repo_root / "Data/GexData/GEX_feature_cell_matrix_HDF5.h5",
    ]
    for c in candidates:
        if c.exists():
            return c

    gex_dir = repo_root / "Data/GexData"
    for ext in ("*.h5", "*.hdf5"):
        found = sorted(gex_dir.glob(ext))
        if found:
            return found[0]
    raise FileNotFoundError("Could not find RNA h5 input. Provide --rna-h5 explicitly.")


def load_manifest_row(manifest_path: Path, mark: str) -> Dict[str, str]:
    df = pd.read_csv(manifest_path, sep="\t")
    if "mark" not in df.columns:
        raise ValueError(f"Manifest missing 'mark' column: {manifest_path}")
    hit = df.loc[df["mark"] == mark]
    if hit.empty:
        raise ValueError(f"Mark '{mark}' not found in manifest: {manifest_path}")
    return hit.iloc[0].to_dict()


def read_named_column(path: Path, preferred_col: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    if preferred_col in df.columns:
        return df[preferred_col].astype(str)
    return df.iloc[:, 0].astype(str)


def build_chrom_adata(repo_root: Path, row: Dict[str, str], mark: str) -> tuple[ad.AnnData, float]:
    mtx_path = resolve_path(repo_root, row["bin_mtx"])
    barcodes_path = resolve_path(repo_root, row["bin_barcodes"])
    features_path = resolve_path(repo_root, row["bin_features"])
    clean_cells_path = resolve_path(repo_root, row["clean_cells"])

    if not all(p.exists() for p in (mtx_path, barcodes_path, features_path, clean_cells_path)):
        missing = [str(p) for p in (mtx_path, barcodes_path, features_path, clean_cells_path) if not p.exists()]
        raise FileNotFoundError(f"Missing chromatin inputs: {missing}")

    x = spio.mmread(mtx_path).tocsr()
    barcodes = read_named_column(barcodes_path, "barcode")
    features = read_named_column(features_path, "feature")

    if x.shape[0] != len(barcodes):
        raise ValueError(f"Barcode length mismatch: matrix rows {x.shape[0]} vs {len(barcodes)}")
    if x.shape[1] != len(features):
        raise ValueError(f"Feature length mismatch: matrix cols {x.shape[1]} vs {len(features)}")

    adata = ad.AnnData(x)
    adata.obs_names = barcodes.values
    adata.var_names = features.values
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    clean_cells = pd.read_csv(clean_cells_path, sep="\t")
    if "barcode" not in clean_cells.columns:
        raise ValueError(f"clean_cells missing 'barcode' column: {clean_cells_path}")

    adata.obs["barcode_core"] = strip_10x_suffix(adata.obs.index.to_series())
    clean_index = clean_cells["barcode"].astype(str)
    matched_frac = float(adata.obs["barcode_core"].isin(clean_index).mean())
    adata.obs = adata.obs.join(clean_cells.set_index("barcode"), on="barcode_core")

    adata.obs["modality"] = "chromatin"
    adata.obs["mark"] = mark
    adata.layers["counts"] = adata.X.copy()

    return adata, matched_frac


def preprocess_rna(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    adata = adata.copy()
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=args.rna_min_genes)
    sc.pp.filter_genes(adata, min_cells=args.rna_min_cells)

    adata.layers["counts"] = adata.X.copy()

    n_top = min(args.rna_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor="cell_ranger",
        layer="counts",
    )

    sc.pp.normalize_total(adata, target_sum=args.rna_target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=args.rna_scale_max)

    n_pcs = max(1, min(args.rna_n_pcs, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True, svd_solver="arpack")

    adata.obs["modality"] = "rna"
    return adata


def tfidf_transform(x):
    n_cells = x.shape[0]
    row_sums = np.asarray(x.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    tf = x.multiply(1.0 / row_sums[:, None])

    doc_freq = np.asarray((x > 0).sum(axis=0)).ravel()
    doc_freq[doc_freq == 0] = 1.0
    idf = np.log1p(n_cells / doc_freq)
    return tf.multiply(idf)


def preprocess_chrom_for_jianle(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    adata = adata.copy()

    n_top = min(max(1, args.chrom_top_features), adata.n_vars)
    feature_sums = np.asarray(adata.X.sum(axis=0)).ravel()
    idx = np.argpartition(feature_sums, -n_top)[-n_top:]

    hv = np.zeros(adata.n_vars, dtype=bool)
    hv[idx] = True
    adata.var["selected_for_pilot"] = hv

    adata = adata[:, hv].copy()
    adata.var["selected_for_pilot"] = True

    x_tfidf = tfidf_transform(adata.layers["counts"])
    max_components = max(1, min(args.chrom_n_lsi, adata.n_vars - 1, adata.n_obs - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=args.random_state)
    adata.obsm["X_lsi"] = svd.fit_transform(x_tfidf)

    adata.uns["pilot_preprocess"] = {
        "chrom_top_features": int(n_top),
        "chrom_n_lsi": int(adata.obsm["X_lsi"].shape[1]),
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
    }
    return adata


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot preprocessing for Jianle-track integration")
    parser.add_argument("--mark", required=True, help="Histone mark key from manifest, e.g. H3K4me1")
    parser.add_argument(
        "--repo-root",
        default=str(infer_repo_root()),
        help="Repository root path",
    )
    parser.add_argument(
        "--manifest",
        default="integration/manifests/scglue_input_manifest.tsv",
        help="Manifest TSV path (repo-relative or absolute)",
    )
    parser.add_argument(
        "--rna-h5",
        default=None,
        help="10x RNA h5 file path (optional; auto-detected if omitted)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: integration/outputs/jianle/pilot/<mark>/preprocess)",
    )

    parser.add_argument("--rna-min-genes", type=int, default=200)
    parser.add_argument("--rna-min-cells", type=int, default=3)
    parser.add_argument("--rna-top-genes", type=int, default=3000)
    parser.add_argument("--rna-target-sum", type=float, default=1e4)
    parser.add_argument("--rna-scale-max", type=float, default=10.0)
    parser.add_argument("--rna-n-pcs", type=int, default=50)

    parser.add_argument("--chrom-top-features", type=int, default=50000)
    parser.add_argument("--chrom-n-lsi", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_path(repo_root, args.manifest)
    row = load_manifest_row(manifest_path, args.mark)
    rna_h5_path = pick_rna_h5(repo_root, args.rna_h5)

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root / f"integration/outputs/jianle/pilot/{args.mark}/preprocess"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rna = sc.read_10x_h5(rna_h5_path)
    rna = preprocess_rna(rna, args)

    chrom, chrom_meta_match = build_chrom_adata(repo_root, row, args.mark)
    chrom = preprocess_chrom_for_jianle(chrom, args)

    rna_out = out_dir / "rna_preprocessed.h5ad"
    chrom_out = out_dir / f"chrom_{args.mark}_preprocessed.h5ad"
    summary_out = out_dir / "preprocess_summary.json"

    rna.write_h5ad(rna_out, compression="gzip")
    chrom.write_h5ad(chrom_out, compression="gzip")

    summary = {
        "mark": args.mark,
        "manifest": str(manifest_path),
        "rna_h5": str(rna_h5_path),
        "outputs": {
            "rna_h5ad": str(rna_out),
            "chrom_h5ad": str(chrom_out),
        },
        "rna": {
            "n_cells": int(rna.n_obs),
            "n_genes": int(rna.n_vars),
            "n_hvg": int(rna.var["highly_variable"].sum()) if "highly_variable" in rna.var else 0,
            "n_pcs": int(rna.obsm["X_pca"].shape[1]) if "X_pca" in rna.obsm else 0,
        },
        "chrom": {
            "n_cells": int(chrom.n_obs),
            "n_features": int(chrom.n_vars),
            "n_lsi": int(chrom.obsm["X_lsi"].shape[1]) if "X_lsi" in chrom.obsm else 0,
            "clean_metadata_match_fraction": chrom_meta_match,
        },
        "params": vars(args),
    }

    with summary_out.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {rna_out}")
    print(f"Wrote: {chrom_out}")
    print(f"Wrote: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
