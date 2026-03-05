#!/usr/bin/env python3
"""Validate and visualize joint scGLUE embeddings."""

from __future__ import annotations

import argparse
import json
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
    obs.index = obs["cell"].astype(str)

    adata = ad.AnnData(X=np.zeros((x.shape[0], 1), dtype=np.float32), obs=obs)
    adata.obsm["X_glue"] = x

    sc.pp.neighbors(adata, n_neighbors=min(args.n_neighbors, max(2, adata.n_obs - 1)), use_rep="X_glue")
    sc.tl.umap(adata, min_dist=args.umap_min_dist, random_state=args.random_seed)
    sc.tl.leiden(adata, resolution=args.leiden_resolution, random_state=args.random_seed, key_added="leiden")

    umap = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1", "UMAP2"])
    out_cells = adata.obs.join(umap, how="left")
    out_cells["leiden"] = adata.obs["leiden"].astype(str)
    cells_tsv = out_dir / "cells_umap_clusters.tsv"
    out_cells.to_csv(cells_tsv, sep="\t", index_label="cell")

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
    }
    metrics_json = out_dir / "validation_metrics.json"
    with metrics_json.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote: {cells_tsv}")
    print(f"Wrote: {cluster_sizes_tsv}")
    print(f"Wrote: {mod_by_cluster_tsv}")
    print(f"Wrote: {umap_mod_png}")
    print(f"Wrote: {umap_cluster_png}")
    print(f"Wrote: {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
