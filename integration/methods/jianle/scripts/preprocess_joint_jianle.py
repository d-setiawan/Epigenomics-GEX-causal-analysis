#!/usr/bin/env python3
"""Joint preprocessing for Jianle-track integration with RNA + all histone marks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as spio
from scipy import sparse as sp

from preprocess_pilot_jianle import (
    annotate_rna_covariates,
    build_chrom_adata,
    infer_repo_root,
    pick_rna_h5,
    preprocess_chrom_for_jianle,
    preprocess_rna,
    read_named_column,
    resolve_path,
    strip_10x_suffix,
)


def parse_marks(manifest_df: pd.DataFrame, marks_csv: str | None) -> List[str]:
    all_marks = list(dict.fromkeys(manifest_df["mark"].astype(str).tolist()))
    if not marks_csv:
        return all_marks
    wanted = [m.strip() for m in marks_csv.split(",") if m.strip()]
    missing = [m for m in wanted if m not in set(all_marks)]
    if missing:
        raise ValueError(f"Marks missing from manifest: {missing}")
    return wanted


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def normalize_gene_universe_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise RuntimeError(f"Shared gene universe is empty: {path}")

    if "feature_id" not in df.columns:
        if "rna_gene" in df.columns:
            df["feature_id"] = df["rna_gene"].astype(str)
        elif "feature" in df.columns:
            df["feature_id"] = df["feature"].astype(str)
        else:
            df["feature_id"] = df.iloc[:, 0].astype(str)

    if "rna_gene" not in df.columns:
        df["rna_gene"] = df["feature_id"].astype(str)
    df["rna_gene"] = df["rna_gene"].fillna("").astype(str)
    df["feature_id"] = df["feature_id"].astype(str)

    if "present_in_rna" not in df.columns:
        df["present_in_rna"] = df["rna_gene"].ne("")
    df["present_in_rna"] = df["present_in_rna"].astype(bool)

    if "gene_index" in df.columns:
        df = df.sort_values("gene_index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if df["feature_id"].duplicated().any():
        dupes = df.loc[df["feature_id"].duplicated(), "feature_id"].tolist()[:5]
        raise ValueError(f"Shared gene universe contains duplicate feature ids: {dupes}")
    return df


def align_sparse_matrix(
    x: sp.csr_matrix,
    source_features: list[str],
    target_features: list[str],
) -> tuple[sp.csr_matrix, np.ndarray]:
    source_index = {str(feat): i for i, feat in enumerate(source_features)}
    target_cols: list[int] = []
    source_cols: list[int] = []
    available = np.zeros(len(target_features), dtype=bool)

    for target_idx, feature in enumerate(target_features):
        source_idx = source_index.get(str(feature))
        if source_idx is None:
            continue
        target_cols.append(int(target_idx))
        source_cols.append(int(source_idx))
        available[target_idx] = True

    if not source_cols:
        aligned = sp.csr_matrix((x.shape[0], len(target_features)), dtype=np.float32)
        return aligned, available

    selected = x[:, np.asarray(source_cols, dtype=int)].tocoo()
    remapped_cols = np.asarray(target_cols, dtype=int)[selected.col]
    aligned = sp.csr_matrix(
        (selected.data.astype(np.float32), (selected.row, remapped_cols)),
        shape=(x.shape[0], len(target_features)),
        dtype=np.float32,
    )
    aligned.eliminate_zeros()
    return aligned, available


def annotate_shared_feature_universe(
    adata: ad.AnnData,
    gene_universe_df: pd.DataFrame,
    feature_available: np.ndarray,
) -> ad.AnnData:
    target_features = gene_universe_df["feature_id"].astype(str).tolist()
    var = gene_universe_df.copy()
    var.index = pd.Index(target_features, name="feature_id")
    var["feature_available"] = np.asarray(feature_available, dtype=bool)
    adata.var = var
    adata.var_names = pd.Index(target_features)
    adata.var_names_make_unique()
    adata.uns["shared_feature_universe"] = True
    adata.uns["feature_space"] = "shared_gene"
    return adata


def sparse_log_normalize_counts(x: sp.csr_matrix, target_sum: float) -> sp.csr_matrix:
    x = x.tocsr().astype(np.float32)
    lib = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(lib, dtype=np.float32)
    keep = lib > 0.0
    scale[keep] = float(target_sum) / lib[keep]
    out = x.multiply(scale[:, None]).tocsr().astype(np.float32)
    if out.nnz > 0:
        out.data = np.log1p(out.data)
    return out


def compute_sparse_feature_stats(x: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
    sq_mean = np.asarray(x.power(2).mean(axis=0)).ravel().astype(np.float32)
    var = np.maximum(0.0, sq_mean - mean**2).astype(np.float32)
    std = np.sqrt(var).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return mean, std


def prepare_shared_modality_features(
    adata: ad.AnnData,
    args: argparse.Namespace,
) -> ad.AnnData:
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if not sp.issparse(counts):
        counts = sp.csr_matrix(np.asarray(counts, dtype=np.float32))
    else:
        counts = counts.tocsr().astype(np.float32)
    adata.layers["counts"] = counts.copy()

    lognorm = sparse_log_normalize_counts(counts, target_sum=args.rna_target_sum)
    adata.X = lognorm

    if "feature_available" in adata.var:
        available_mask = adata.var["feature_available"].to_numpy(dtype=bool)
    else:
        available_mask = np.ones(adata.n_vars, dtype=bool)

    hvg_mask = np.zeros(adata.n_vars, dtype=bool)
    mean_arr = np.zeros(adata.n_vars, dtype=np.float32)
    std_arr = np.ones(adata.n_vars, dtype=np.float32)

    available_idx = np.flatnonzero(available_mask)
    if available_idx.size > 0:
        counts_available = counts[:, available_idx]
        gene_totals = np.asarray(counts_available.sum(axis=0)).ravel()
        nonzero_mask = gene_totals > 0
        usable_idx = available_idx[nonzero_mask]
        if usable_idx.size > 0:
            counts_usable = counts[:, usable_idx]
            lognorm_usable = lognorm[:, usable_idx]

            mean_sub, std_sub = compute_sparse_feature_stats(lognorm_usable)
            mean_arr[usable_idx] = mean_sub
            std_arr[usable_idx] = std_sub

            if int(args.shared_hvg_top_genes) > 0:
                n_top = min(int(args.shared_hvg_top_genes), usable_idx.size)
                if n_top > 0:
                    tmp_counts = ad.AnnData(counts_usable.copy())
                    tmp_counts.layers["counts"] = counts_usable.copy()
                    try:
                        sc.pp.highly_variable_genes(
                            tmp_counts,
                            n_top_genes=n_top,
                            flavor=args.shared_hvg_flavor,
                            layer="counts",
                        )
                        hvg_mask[usable_idx] = tmp_counts.var["highly_variable"].to_numpy(dtype=bool)
                    except ValueError as exc:
                        # Extremely sparse aggregated histone features can produce repeated
                        # mean-bin edges in Scanpy's cell_ranger-style HVG routine.
                        # Fall back to a normalized-data HVG call, then to a simple variance
                        # ranking so shared-gene preprocessing remains robust.
                        if "Bin edges must be unique" not in str(exc):
                            raise
                        tmp_lognorm = ad.AnnData(lognorm_usable.copy())
                        try:
                            sc.pp.highly_variable_genes(
                                tmp_lognorm,
                                n_top_genes=n_top,
                                flavor="seurat",
                            )
                            hvg_mask[usable_idx] = tmp_lognorm.var["highly_variable"].to_numpy(dtype=bool)
                        except Exception:
                            rank_order = np.argsort(-(std_sub.astype(np.float64) ** 2), kind="stable")
                            chosen = usable_idx[rank_order[:n_top]]
                            hvg_mask[chosen] = True

    adata.var["model_mean"] = mean_arr
    adata.var["model_std"] = std_arr
    adata.var["highly_variable"] = hvg_mask
    return adata


def compute_shared_rna_pca(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    use_mask = np.zeros(adata.n_vars, dtype=bool)
    if "feature_available" in adata.var:
        use_mask = adata.var["feature_available"].to_numpy(dtype=bool)
    if "highly_variable" in adata.var and adata.var["highly_variable"].to_numpy(dtype=bool).sum() > 1:
        use_mask = use_mask & adata.var["highly_variable"].to_numpy(dtype=bool)

    use_idx = np.flatnonzero(use_mask)
    if adata.n_obs > 1 and use_idx.size > 1:
        tmp = adata[:, use_idx].copy()
        sc.pp.scale(tmp, max_value=args.rna_scale_max)
        n_pcs = min(args.rna_n_pcs, tmp.n_obs - 1, tmp.n_vars - 1)
        if n_pcs >= 1:
            sc.tl.pca(tmp, n_comps=int(n_pcs), use_highly_variable=False, svd_solver="arpack")
            adata.obsm["X_pca"] = tmp.obsm["X_pca"]
    return adata


def subset_shared_modalities_to_hvg_union(
    rna: ad.AnnData,
    chrom_adatas: dict[str, ad.AnnData],
) -> tuple[ad.AnnData, dict[str, ad.AnnData], int]:
    union_mask = rna.var["highly_variable"].to_numpy(dtype=bool).copy() if "highly_variable" in rna.var else np.zeros(rna.n_vars, dtype=bool)
    for chrom in chrom_adatas.values():
        if "highly_variable" in chrom.var:
            union_mask |= chrom.var["highly_variable"].to_numpy(dtype=bool)

    union_count = int(union_mask.sum())
    if union_count <= 0:
        raise RuntimeError("Shared-gene HVG union ended up empty; try increasing --shared-hvg-top-genes")

    rna = rna[:, union_mask].copy()
    rna.var["shared_hvg_union"] = True

    out: dict[str, ad.AnnData] = {}
    for mark, chrom in chrom_adatas.items():
        sub = chrom[:, union_mask].copy()
        sub.var["shared_hvg_union"] = True
        out[mark] = sub
    return rna, out, union_count


def build_shared_gene_chrom_adata(
    repo_root: Path,
    row: dict,
    mark: str,
    gene_universe_df: pd.DataFrame,
) -> tuple[ad.AnnData, float]:
    mtx_path = resolve_path(repo_root, str(row["gene_mtx"]))
    barcodes_path = resolve_path(repo_root, str(row["gene_barcodes"]))
    features_path = resolve_path(repo_root, str(row["gene_features"]))
    clean_cells_path = resolve_path(repo_root, str(row["clean_cells"]))
    availability_path = None
    if "gene_availability" in row and str(row["gene_availability"]).strip():
        availability_path = resolve_path(repo_root, str(row["gene_availability"]))

    required = [mtx_path, barcodes_path, features_path, clean_cells_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shared gene-space chromatin inputs for {mark}: {missing}")

    x = spio.mmread(mtx_path).tocsr().astype(np.float32)
    barcodes = read_named_column(barcodes_path, "barcode").astype(str)
    features = read_named_column(features_path, "feature").astype(str)
    if x.shape[0] != len(barcodes):
        raise ValueError(f"{mark}: barcode length mismatch with matrix rows")
    if x.shape[1] != len(features):
        raise ValueError(f"{mark}: feature length mismatch with matrix columns")

    target_features = gene_universe_df["feature_id"].astype(str).tolist()
    x_aligned, available_by_presence = align_sparse_matrix(x, features.tolist(), target_features)

    if availability_path is not None and availability_path.exists():
        availability_df = pd.read_csv(availability_path, sep="\t")
        if "feature" not in availability_df.columns:
            availability_df["feature"] = availability_df.iloc[:, 0].astype(str)
        availability_col = "feature_available" if "feature_available" in availability_df.columns else availability_df.columns[1]
        availability_lookup = {}
        for _, avail_row in availability_df.iterrows():
            value = avail_row[availability_col]
            availability_lookup[str(avail_row["feature"])] = bool(int(value)) if pd.notna(value) else False
        feature_available = np.asarray(
            [bool(availability_lookup.get(feature, False)) for feature in target_features],
            dtype=bool,
        )
    else:
        feature_available = available_by_presence

    adata = ad.AnnData(x_aligned)
    adata.obs_names = pd.Index(barcodes.values)
    adata.obs_names_make_unique()
    adata = annotate_shared_feature_universe(adata, gene_universe_df, feature_available)

    clean_cells = pd.read_csv(clean_cells_path, sep="\t")
    if "barcode" not in clean_cells.columns:
        raise ValueError(f"clean_cells missing 'barcode' column: {clean_cells_path}")

    adata.obs["barcode_core"] = strip_10x_suffix(adata.obs.index.to_series())
    clean_index = clean_cells["barcode"].astype(str)
    matched_frac = float(adata.obs["barcode_core"].isin(clean_index).mean())
    adata.obs = adata.obs.join(clean_cells.set_index("barcode"), on="barcode_core")

    adata.obs["modality"] = "chromatin"
    adata.obs["mark"] = mark
    adata.obs["sample_id"] = str(row.get("source_prefix", row.get("prefix", mark)))
    adata.obs["batch_id"] = str(row.get("source_prefix", row.get("prefix", mark)))
    adata.layers["counts"] = adata.X.copy()
    return adata, matched_frac


def preprocess_rna_shared_gene_space(
    rna: ad.AnnData,
    gene_universe_df: pd.DataFrame,
    args: argparse.Namespace,
) -> ad.AnnData:
    rna = rna.copy()
    rna.var_names_make_unique()
    sc.pp.filter_cells(rna, min_genes=args.rna_min_genes)

    x = rna.X
    if not sp.issparse(x):
        x = sp.csr_matrix(np.asarray(x, dtype=np.float32))
    else:
        x = x.tocsr().astype(np.float32)

    target_features = gene_universe_df["feature_id"].astype(str).tolist()
    target_rna_features = gene_universe_df["rna_gene"].fillna("").astype(str).tolist()

    source_index = {str(feat): i for i, feat in enumerate(rna.var_names.astype(str).tolist())}
    target_cols: list[int] = []
    source_cols: list[int] = []
    feature_available = np.zeros(len(target_features), dtype=bool)
    for target_idx, rna_feature in enumerate(target_rna_features):
        if not rna_feature:
            continue
        source_idx = source_index.get(rna_feature)
        if source_idx is None:
            continue
        target_cols.append(int(target_idx))
        source_cols.append(int(source_idx))
        feature_available[target_idx] = True

    if source_cols:
        selected = x[:, np.asarray(source_cols, dtype=int)].tocoo()
        remapped_cols = np.asarray(target_cols, dtype=int)[selected.col]
        aligned = sp.csr_matrix(
            (selected.data.astype(np.float32), (selected.row, remapped_cols)),
            shape=(x.shape[0], len(target_features)),
            dtype=np.float32,
        )
        aligned.eliminate_zeros()
    else:
        aligned = sp.csr_matrix((x.shape[0], len(target_features)), dtype=np.float32)

    adata = ad.AnnData(aligned)
    adata.obs = rna.obs.copy()
    adata.obs_names = rna.obs_names.copy()
    adata = annotate_shared_feature_universe(adata, gene_universe_df, feature_available)
    adata.layers["counts"] = adata.X.copy()
    adata = prepare_shared_modality_features(adata, args)
    adata.obs["modality"] = "rna"
    adata = annotate_rna_covariates(adata, infer_repo_root())
    return adata


def main() -> int:
    p = argparse.ArgumentParser(description="Joint preprocessing for Jianle-track integration")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument(
        "--gene-feature-manifest",
        default=None,
        help="Optional shared gene-space manifest from build_joint_gene_feature_universe.py",
    )
    p.add_argument(
        "--gene-universe",
        default=None,
        help="Optional shared gene universe TSV; inferred from --gene-feature-manifest when omitted",
    )
    p.add_argument("--rna-h5", default=None)
    p.add_argument("--marks", default=None, help="Comma-separated subset of marks (default: all)")
    p.add_argument("--run-id", default="joint")
    p.add_argument("--out-dir", default=None, help="Default: integration/outputs/jianle/joint/<RUN_ID>/preprocess")

    p.add_argument("--rna-min-genes", type=int, default=200)
    p.add_argument("--rna-min-cells", type=int, default=3)
    p.add_argument("--rna-top-genes", type=int, default=3000)
    p.add_argument("--rna-target-sum", type=float, default=1e4)
    p.add_argument("--rna-scale-max", type=float, default=10.0)
    p.add_argument("--rna-n-pcs", type=int, default=50)
    p.add_argument(
        "--shared-hvg-top-genes",
        type=int,
        default=3000,
        help="For shared-gene mode, select up to this many HVGs per modality and keep the union",
    )
    p.add_argument(
        "--shared-hvg-flavor",
        default="cell_ranger",
        choices=["cell_ranger", "seurat"],
        help="Scanpy HVG flavor for shared-gene mode",
    )

    p.add_argument("--chrom-top-features", type=int, default=30000)
    p.add_argument("--chrom-n-lsi", type=int, default=50)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    gene_feature_manifest_path = resolve_path(repo_root, args.gene_feature_manifest) if args.gene_feature_manifest else None
    if gene_feature_manifest_path is not None:
        manifest_path = gene_feature_manifest_path
    else:
        manifest_path = resolve_path(repo_root, args.manifest)
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "mark" not in manifest_df.columns:
        raise ValueError(f"Manifest missing 'mark' column: {manifest_path}")
    marks = parse_marks(manifest_df, args.marks)

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root / f"integration/outputs/jianle/joint/{args.run_id}/preprocess"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rna_h5_path = pick_rna_h5(repo_root, args.rna_h5)
    rna = sc.read_10x_h5(rna_h5_path)

    shared_gene_mode = gene_feature_manifest_path is not None
    gene_universe_df = None
    gene_universe_path = None
    if shared_gene_mode:
        if args.gene_universe:
            gene_universe_path = resolve_path(repo_root, args.gene_universe)
        elif "gene_universe" in manifest_df.columns and manifest_df["gene_universe"].notna().any():
            gene_universe_path = resolve_path(repo_root, str(manifest_df["gene_universe"].dropna().iloc[0]))
        else:
            gene_universe_path = manifest_path.parent / "gene_universe.tsv"
        gene_universe_df = normalize_gene_universe_df(gene_universe_path)
        rna = preprocess_rna_shared_gene_space(rna, gene_universe_df, args)
    else:
        rna = preprocess_rna(rna, args)
        rna = annotate_rna_covariates(rna, repo_root)

    rna.obs["modality_key"] = "rna"
    rna.obs["mark"] = "RNA"
    chrom_adatas: dict[str, ad.AnnData] = {}
    chrom_meta: dict[str, dict] = {}
    for mark in marks:
        row = manifest_df.loc[manifest_df["mark"] == mark].iloc[0].to_dict()
        if shared_gene_mode:
            assert gene_universe_df is not None
            chrom, clean_match = build_shared_gene_chrom_adata(repo_root, row, mark, gene_universe_df)
            chrom = prepare_shared_modality_features(chrom, args)
        else:
            chrom, clean_match = build_chrom_adata(repo_root, row, mark)
            chrom = preprocess_chrom_for_jianle(chrom, args)

            # Namespace feature ids by mark for downstream clarity.
            chrom.var["orig_feature"] = chrom.var_names.astype(str)
            chrom.var_names = pd.Index([f"{mark}::{x}" for x in chrom.var["orig_feature"].astype(str)])
            chrom.var_names_make_unique()

        chrom.obs["modality_key"] = f"chrom_{mark}"
        chrom.obs["mark"] = mark

        chrom_adatas[mark] = chrom
        chrom_meta[mark] = {
            "clean_metadata_match_fraction": float(clean_match),
        }

    shared_hvg_union_size = None
    if shared_gene_mode:
        if int(args.shared_hvg_top_genes) > 0:
            rna, chrom_adatas, shared_hvg_union_size = subset_shared_modalities_to_hvg_union(rna, chrom_adatas)
        rna = compute_shared_rna_pca(rna, args)

    rna_out = out_dir / "rna_preprocessed.h5ad"
    rna.write_h5ad(rna_out, compression="gzip")

    chrom_rows: list[dict] = []
    for mark in marks:
        chrom = chrom_adatas[mark]
        chrom_out = out_dir / f"chrom_{mark}_preprocessed.h5ad"
        chrom.write_h5ad(chrom_out, compression="gzip")

        chrom_rows.append(
            {
                "mark": mark,
                "modality_key": f"chrom_{mark}",
                "chrom_h5ad": str(chrom_out),
                "n_cells": int(chrom.n_obs),
                "n_features": int(chrom.n_vars),
                "n_available_features": int(chrom.var["feature_available"].sum()) if "feature_available" in chrom.var else int(chrom.n_vars),
                "n_hvg": int(chrom.var["highly_variable"].sum()) if "highly_variable" in chrom.var else 0,
                "n_lsi": int(chrom.obsm["X_lsi"].shape[1]) if "X_lsi" in chrom.obsm else 0,
                "clean_metadata_match_fraction": float(chrom_meta[mark]["clean_metadata_match_fraction"]),
            }
        )

    chrom_manifest_tsv = out_dir / "chrom_preprocessed_manifest.tsv"
    write_tsv(
        chrom_manifest_tsv,
        chrom_rows,
        [
            "mark",
            "modality_key",
            "chrom_h5ad",
            "n_cells",
            "n_features",
            "n_available_features",
            "n_hvg",
            "n_lsi",
            "clean_metadata_match_fraction",
        ],
    )

    summary = {
        "run_id": args.run_id,
        "manifest": str(manifest_path),
        "rna_h5": str(rna_h5_path),
        "marks": marks,
        "feature_space_mode": "shared_gene" if shared_gene_mode else "native_modality",
        "outputs": {
            "rna_h5ad": str(rna_out),
            "chrom_manifest_tsv": str(chrom_manifest_tsv),
            "out_dir": str(out_dir),
        },
        "rna": {
            "n_cells": int(rna.n_obs),
            "n_genes": int(rna.n_vars),
            "n_available_features": int(rna.var["feature_available"].sum()) if "feature_available" in rna.var else int(rna.n_vars),
            "n_hvg": int(rna.var["highly_variable"].sum()) if "highly_variable" in rna.var else 0,
            "n_pcs": int(rna.obsm["X_pca"].shape[1]) if "X_pca" in rna.obsm else 0,
        },
        "shared_gene_universe": (
            {
                "path": str(gene_universe_path),
                "n_features_before_hvg_union": int(gene_universe_df.shape[0]),
                "n_features_after_hvg_union": int(rna.n_vars),
                "n_present_in_rna": int(gene_universe_df["present_in_rna"].sum()),
                "shared_hvg_top_genes_per_modality": int(args.shared_hvg_top_genes),
                "shared_hvg_union_size": int(shared_hvg_union_size) if shared_hvg_union_size is not None else None,
            }
            if shared_gene_mode and gene_universe_df is not None and gene_universe_path is not None
            else None
        ),
        "chrom": chrom_rows,
        "params": vars(args),
    }
    summary_path = out_dir / "joint_preprocess_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {rna_out}")
    print(f"Wrote: {chrom_manifest_tsv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
