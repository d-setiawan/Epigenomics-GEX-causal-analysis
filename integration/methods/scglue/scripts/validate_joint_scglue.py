#!/usr/bin/env python3
"""Validate and visualize joint scGLUE embeddings."""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from io import BytesIO
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def strip_10x_suffix(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"-\d+$", "", regex=True)


def load_rna_cell_annotations(path: Path, level: str) -> pd.DataFrame:
    with tarfile.open(path, "r:gz") as tf:
        member = "cell_types/cell_types.csv"
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Missing {member} in {path}")
        ann = pd.read_csv(BytesIO(f.read()))
    required = {"barcode", level}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")
    ann = ann[["barcode", level]].copy()
    ann["barcode"] = ann["barcode"].astype(str)
    ann["barcode_core"] = strip_10x_suffix(ann["barcode"])
    ann = ann.drop_duplicates(subset=["barcode_core"])
    return ann


def same_modality_neighbor_fraction(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = x.shape[0]
    if n <= 2:
        return float("nan")
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(x)
    idx = nn.kneighbors(x, return_distance=False)
    nbrs = idx[:, 1:]
    scores = []
    for i in range(n):
        scores.append(float(np.mean(labels[nbrs[i]] == labels[i])))
    return float(np.mean(scores))


def main() -> int:
    p = argparse.ArgumentParser(description="Validate joint scGLUE results")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--train-dir", required=True, help="Joint training output directory")
    p.add_argument("--out-dir", default=None, help="Default: <train-dir>/validation")
    p.add_argument(
        "--rna-annotation-tar",
        default="integration/workspace/data/rna/cell_type_annotation.tar.gz",
        help="Tar.gz containing cell_types/cell_types.csv for RNA annotation overlay",
    )
    p.add_argument(
        "--cell-type-level",
        default="coarse_cell_type",
        choices=["coarse_cell_type", "fine_cell_type"],
        help="Column from cell_types.csv to plot on the RNA overlay UMAP",
    )
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.3)
    p.add_argument("--leiden-resolution", type=float, default=1.0)
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_dir = resolve_path(repo_root, args.train_dir)
    out_dir = resolve_path(repo_root, args.out_dir) if args.out_dir else train_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = train_dir / "all_cells_glue_embeddings.tsv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings table: {emb_path}")
    emb = pd.read_csv(emb_path, sep="\t")
    glue_cols = [c for c in emb.columns if c.startswith("GLUE_")]
    if not glue_cols:
        raise RuntimeError("No GLUE_* columns found in all_cells_glue_embeddings.tsv")

    x = emb[glue_cols].to_numpy(dtype=float)
    obs = emb[["cell", "modality_key", "mark"]].copy()
    # Cell barcodes are reused across modalities; use a unique per-row obs id.
    obs["obs_id"] = obs["modality_key"].astype(str) + "::" + obs["cell"].astype(str)
    obs.index = obs["obs_id"].astype(str)
    obs.index.name = "obs_id"

    adata = ad.AnnData(X=np.zeros((x.shape[0], 1), dtype=np.float32), obs=obs)
    adata.obsm["X_glue"] = x

    sc.pp.neighbors(adata, n_neighbors=min(args.n_neighbors, max(2, adata.n_obs - 1)), use_rep="X_glue")
    sc.tl.umap(adata, min_dist=args.umap_min_dist, random_state=args.random_seed)
    sc.tl.leiden(adata, resolution=args.leiden_resolution, random_state=args.random_seed, key_added="leiden")

    umap = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1", "UMAP2"])
    out_cells = adata.obs.join(umap, how="left")
    out_cells["leiden"] = adata.obs["leiden"].astype(str).to_numpy()
    out_cells["barcode_core"] = strip_10x_suffix(out_cells["cell"])
    cells_tsv = out_dir / "cells_umap_clusters.tsv"
    out_cells.to_csv(cells_tsv, sep="\t", index_label="obs_id")

    modality_labels = adata.obs["modality_key"].astype(str).to_numpy()
    sil_modality = float("nan")
    if len(np.unique(modality_labels)) > 1 and adata.n_obs > len(np.unique(modality_labels)):
        sil_modality = float(silhouette_score(x, modality_labels, metric="euclidean"))
    same_mod_frac = same_modality_neighbor_fraction(x, modality_labels, k=args.n_neighbors)

    cluster_sizes = adata.obs["leiden"].value_counts().sort_index()
    cluster_sizes_tsv = out_dir / "cluster_sizes.tsv"
    cluster_sizes.rename("n_cells").to_csv(cluster_sizes_tsv, sep="\t", header=True, index_label="cluster")

    mod_by_cluster = pd.crosstab(adata.obs["leiden"], adata.obs["modality_key"])
    mod_by_cluster_tsv = out_dir / "cluster_by_modality.tsv"
    mod_by_cluster.to_csv(mod_by_cluster_tsv, sep="\t")

    mod_by_cluster_frac = mod_by_cluster.div(mod_by_cluster.sum(axis=1), axis=0).fillna(0.0)
    mod_by_cluster_frac_tsv = out_dir / "cluster_by_modality_fraction.tsv"
    mod_by_cluster_frac.to_csv(mod_by_cluster_frac_tsv, sep="\t")

    plt.figure(figsize=(7, 6))
    for mod, sub in out_cells.groupby("modality_key", sort=False):
        plt.scatter(sub["UMAP1"], sub["UMAP2"], s=4, alpha=0.7, label=mod)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Joint scGLUE UMAP by modality")
    plt.legend(markerscale=3, fontsize=8, loc="best")
    plt.tight_layout()
    umap_mod_png = out_dir / "umap_by_modality.png"
    plt.savefig(umap_mod_png, dpi=200)
    plt.close()

    modality_list = list(out_cells["modality_key"].astype(str).drop_duplicates())
    n_panels = len(modality_list)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, mod in zip(axes_flat, modality_list):
        fg = out_cells.loc[out_cells["modality_key"] == mod]
        ax.scatter(out_cells["UMAP1"], out_cells["UMAP2"], s=3, alpha=0.12, color="lightgray", rasterized=True)
        ax.scatter(fg["UMAP1"], fg["UMAP2"], s=4, alpha=0.8, rasterized=True)
        ax.set_title(str(mod))
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    plt.tight_layout()
    umap_mod_facets_png = out_dir / "umap_by_modality_facets.png"
    plt.savefig(umap_mod_facets_png, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    cats = out_cells["leiden"].astype("category")
    cmap = plt.get_cmap("tab20")
    for i, cl in enumerate(cats.cat.categories):
        sub = out_cells.loc[cats == cl]
        plt.scatter(sub["UMAP1"], sub["UMAP2"], s=4, alpha=0.7, color=cmap(i % 20), label=str(cl))
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Joint scGLUE UMAP by Leiden cluster")
    plt.tight_layout()
    umap_cluster_png = out_dir / "umap_by_leiden.png"
    plt.savefig(umap_cluster_png, dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, 1.2 * mod_by_cluster_frac.shape[1]), max(6, 0.35 * mod_by_cluster_frac.shape[0])))
    ax = plt.gca()
    im = ax.imshow(mod_by_cluster_frac.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(mod_by_cluster_frac.shape[1]))
    ax.set_xticklabels(mod_by_cluster_frac.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(mod_by_cluster_frac.shape[0]))
    ax.set_yticklabels(mod_by_cluster_frac.index.astype(str).tolist())
    ax.set_xlabel("Modality")
    ax.set_ylabel("Leiden cluster")
    ax.set_title("Cluster-by-modality fraction heatmap")
    plt.colorbar(im, ax=ax, label="Fraction within cluster")
    plt.tight_layout()
    cluster_heatmap_png = out_dir / "cluster_by_modality_heatmap.png"
    plt.savefig(cluster_heatmap_png, dpi=200)
    plt.close()

    annotation_path = resolve_path(repo_root, args.rna_annotation_tar)
    annotated_cells_tsv = None
    umap_celltype_png = None
    if annotation_path and annotation_path.exists():
        ann = load_rna_cell_annotations(annotation_path, args.cell_type_level)
        rna_cells = out_cells.loc[out_cells["modality_key"].astype(str) == "rna"].copy()
        rna_cells = rna_cells.merge(
            ann[["barcode_core", args.cell_type_level]],
            on="barcode_core",
            how="left",
        )
        annotated_cells_tsv = out_dir / f"rna_umap_{args.cell_type_level}.tsv"
        rna_cells.to_csv(annotated_cells_tsv, sep="\t", index=False)

        labeled = rna_cells.dropna(subset=[args.cell_type_level]).copy()
        if not labeled.empty:
            categories = sorted(labeled[args.cell_type_level].astype(str).unique().tolist())
            cmap = plt.get_cmap("tab20", max(len(categories), 1))
            color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

            plt.figure(figsize=(8, 6.5))
            plt.scatter(out_cells["UMAP1"], out_cells["UMAP2"], s=3, alpha=0.08, color="lightgray", rasterized=True)
            for cat in categories:
                sub = labeled.loc[labeled[args.cell_type_level].astype(str) == cat]
                plt.scatter(
                    sub["UMAP1"],
                    sub["UMAP2"],
                    s=6,
                    alpha=0.85,
                    color=color_map[cat],
                    label=cat,
                    rasterized=True,
                )
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
            plt.title(f"Joint UMAP with RNA {args.cell_type_level}")
            plt.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            umap_celltype_png = out_dir / f"umap_rna_{args.cell_type_level}.png"
            plt.savefig(umap_celltype_png, dpi=200, bbox_inches="tight")
            plt.close()

    metrics = {
        "n_cells_total": int(adata.n_obs),
        "n_modalities": int(adata.obs["modality_key"].nunique()),
        "n_clusters_leiden": int(adata.obs["leiden"].nunique()),
        "silhouette_by_modality": sil_modality,
        "mean_same_modality_neighbor_fraction": same_mod_frac,
        "notes": {
            "silhouette_by_modality": "Near 0 is usually better modality mixing; very high positive indicates separation.",
            "mean_same_modality_neighbor_fraction": "Lower indicates better cross-modality mixing (context dependent).",
        },
        "annotation_overlay": {
            "annotation_tar": str(annotation_path) if annotation_path and annotation_path.exists() else None,
            "cell_type_level": args.cell_type_level,
            "rna_annotation_overlay_written": bool(umap_celltype_png),
        },
    }
    metrics_json = out_dir / "validation_metrics.json"
    with metrics_json.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote: {cells_tsv}")
    print(f"Wrote: {cluster_sizes_tsv}")
    print(f"Wrote: {mod_by_cluster_tsv}")
    print(f"Wrote: {mod_by_cluster_frac_tsv}")
    print(f"Wrote: {umap_mod_png}")
    print(f"Wrote: {umap_mod_facets_png}")
    print(f"Wrote: {umap_cluster_png}")
    print(f"Wrote: {cluster_heatmap_png}")
    if annotated_cells_tsv:
        print(f"Wrote: {annotated_cells_tsv}")
    if umap_celltype_png:
        print(f"Wrote: {umap_celltype_png}")
    print(f"Wrote: {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
