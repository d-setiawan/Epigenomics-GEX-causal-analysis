#!/usr/bin/env python3
"""Validate and visualize joint scGLUE embeddings."""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from collections import Counter
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


def load_rna_cell_annotations(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r:gz") as tf:
        member = "cell_types/cell_types.csv"
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Missing {member} in {path}")
        ann = pd.read_csv(BytesIO(f.read()))
    required = {"barcode", "coarse_cell_type", "fine_cell_type"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")
    keep_cols = ["barcode", "coarse_cell_type", "fine_cell_type"]
    if "cell_count_in_model" in ann.columns:
        keep_cols.append("cell_count_in_model")
    ann = ann[keep_cols].copy()
    ann["barcode"] = ann["barcode"].astype(str)
    ann["barcode_core"] = strip_10x_suffix(ann["barcode"])
    ann = ann.drop_duplicates(subset=["barcode_core"])
    return ann


def normalize_label_level(level: str) -> str:
    aliases = {
        "coarse_cell_type": "coarse_cell_type",
        "fine_cell_type": "fine_cell_type",
        "harmonized_coarse": "harmonized_coarse",
        "harmonized_fine": "harmonized_fine",
    }
    if level not in aliases:
        raise ValueError(f"Unsupported label level: {level}")
    return aliases[level]


def load_harmonization_table(path: Path) -> pd.DataFrame:
    harm = pd.read_csv(path, sep="\t")
    required = {"coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine"}
    missing = required - set(harm.columns)
    if missing:
        raise ValueError(f"Harmonization table missing columns: {sorted(missing)}")
    dup = harm.duplicated(subset=["coarse_cell_type", "fine_cell_type"])
    if dup.any():
        raise ValueError("Harmonization table has duplicate coarse/fine label pairs")
    if "cell_ontology_id" not in harm.columns:
        harm["cell_ontology_id"] = pd.NA
    return harm


def apply_harmonization(ann: pd.DataFrame, harm: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    merged = ann.copy()
    if harm is None:
        merged["harmonized_coarse"] = merged["coarse_cell_type"]
        merged["harmonized_fine"] = merged["fine_cell_type"]
        merged["cell_ontology_id"] = pd.NA
        return merged, {"mode": "identity", "n_unmapped_rows": 0}

    merged = merged.merge(
        harm[["coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine", "cell_ontology_id"]],
        on=["coarse_cell_type", "fine_cell_type"],
        how="left",
    )
    unmapped = int(merged["harmonized_coarse"].isna().sum() + merged["harmonized_fine"].isna().sum())
    merged["harmonized_coarse"] = merged["harmonized_coarse"].fillna(merged["coarse_cell_type"])
    merged["harmonized_fine"] = merged["harmonized_fine"].fillna(merged["fine_cell_type"])
    return merged, {"mode": "table", "n_unmapped_rows": unmapped}


def majority_vote(labels: list[str]) -> tuple[str, float]:
    labels = [str(x) for x in labels if pd.notna(x) and str(x) != ""]
    if not labels:
        return "unknown", 0.0
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    return label, float(count / len(labels))


def plot_label_overlay(
    adata: ad.AnnData,
    values: pd.Series,
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (8, 6.5),
) -> None:
    values = values.reindex(adata.obs_names)
    labeled = values.dropna()
    labeled = labeled.loc[labeled.astype(str) != ""]
    if labeled.empty:
        return
    temp_col = "__plot_label__"
    adata.obs[temp_col] = values
    try:
        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.umap(
            adata,
            color=temp_col,
            ax=ax,
            show=False,
            title=title,
            legend_loc="right margin",
            na_color="lightgray",
            na_in_legend=False,
            frameon=False,
            size=8,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    finally:
        del adata.obs[temp_col]


def save_scanpy_umap(
    adata: ad.AnnData,
    color: str,
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (7, 6),
    legend_loc: str | None = "right margin",
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.umap(
        adata,
        color=color,
        ax=ax,
        show=False,
        title=title,
        legend_loc=legend_loc,
        frameon=False,
        size=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scanpy_modality_facets(adata: ad.AnnData, out_path: Path) -> None:
    modality_list = list(adata.obs["modality_key"].astype(str).drop_duplicates())
    n_panels = len(modality_list)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab10", max(n_panels, 1))
    for i, (ax, mod) in enumerate(zip(axes_flat, modality_list)):
        temp_col = f"__highlight_modality_{i}"
        adata.obs[temp_col] = pd.Categorical(
            np.where(adata.obs["modality_key"].astype(str) == mod, mod, "other"),
            categories=[mod, "other"],
        )
        try:
            sc.pl.umap(
                adata,
                color=temp_col,
                ax=ax,
                show=False,
                title=str(mod),
                legend_loc=None,
                frameon=False,
                size=7,
                palette={mod: cmap(i), "other": "#d3d3d3"},
            )
        finally:
            del adata.obs[temp_col]
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def transfer_labels_from_rna(
    out_cells: pd.DataFrame,
    x: np.ndarray,
    ann: pd.DataFrame,
    k: int,
    min_confidence: float,
) -> pd.DataFrame:
    cells = out_cells.copy()
    cells["row_idx"] = np.arange(cells.shape[0])

    ann_cols = [
        "barcode_core",
        "coarse_cell_type",
        "fine_cell_type",
        "harmonized_coarse",
        "harmonized_fine",
        "cell_ontology_id",
    ]
    rna_ref = cells.loc[cells["modality_key"].astype(str) == "rna"].merge(
        ann[ann_cols],
        on="barcode_core",
        how="left",
    )
    rna_ref = rna_ref.dropna(subset=["harmonized_coarse"]).copy()
    if rna_ref.empty:
        raise RuntimeError("RNA annotation overlay exists, but no RNA cells matched the annotation table")

    ref_idx = rna_ref["row_idx"].to_numpy(dtype=int)
    ref_x = x[ref_idx]
    k_eff = min(max(1, k), ref_x.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(ref_x)

    result = cells[["cell", "modality_key", "mark", "barcode_core"]].copy()
    result["harmonized_coarse"] = pd.NA
    result["harmonized_fine"] = pd.NA
    result["coarse_confidence"] = np.nan
    result["fine_confidence"] = np.nan
    result["label_source"] = "unlabeled"
    result["cell_ontology_id"] = pd.NA

    rna_obs = rna_ref.set_index("obs_id")
    result.loc[rna_obs.index, "harmonized_coarse"] = rna_obs["harmonized_coarse"]
    result.loc[rna_obs.index, "harmonized_fine"] = rna_obs["harmonized_fine"]
    result.loc[rna_obs.index, "coarse_confidence"] = 1.0
    result.loc[rna_obs.index, "fine_confidence"] = 1.0
    result.loc[rna_obs.index, "label_source"] = "rna_annotation"
    result.loc[rna_obs.index, "cell_ontology_id"] = rna_obs["cell_ontology_id"].to_numpy()

    query_cells = cells.loc[cells["modality_key"].astype(str) != "rna"].copy()
    if not query_cells.empty:
        query_idx = query_cells["row_idx"].to_numpy(dtype=int)
        query_x = x[query_idx]
        nbr_idx = nn.kneighbors(query_x, return_distance=False)
        ref_labels = rna_ref.reset_index(drop=True)
        for obs_id, nbrs in zip(query_cells.index, nbr_idx):
            nbr_df = ref_labels.iloc[nbrs]
            coarse, coarse_conf = majority_vote(nbr_df["harmonized_coarse"].tolist())
            fine_pool = nbr_df
            if coarse != "unknown":
                fine_pool = nbr_df.loc[nbr_df["harmonized_coarse"] == coarse]
                if fine_pool.empty:
                    fine_pool = nbr_df
            fine, fine_conf = majority_vote(fine_pool["harmonized_fine"].tolist())
            if coarse_conf < min_confidence:
                coarse = "unknown"
            if fine_conf < min_confidence:
                fine = "unknown"
            result.loc[obs_id, "harmonized_coarse"] = coarse
            result.loc[obs_id, "harmonized_fine"] = fine
            result.loc[obs_id, "coarse_confidence"] = coarse_conf
            result.loc[obs_id, "fine_confidence"] = fine_conf
            result.loc[obs_id, "label_source"] = "rna_knn_transfer"
            if fine != "unknown":
                ontology, _ = majority_vote(fine_pool["cell_ontology_id"].tolist())
                if ontology != "unknown":
                    result.loc[obs_id, "cell_ontology_id"] = ontology

    return result


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
        "--harmonization-tsv",
        default="integration/manifests/rna_label_harmonization.tsv",
        help="Optional TSV mapping raw RNA labels to harmonized coarse/fine labels",
    )
    p.add_argument(
        "--cell-type-level",
        default="coarse_cell_type",
        choices=["coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine"],
        help="Label column to plot on the RNA overlay UMAP",
    )
    p.add_argument(
        "--transfer-labels",
        action="store_true",
        help="Transfer harmonized RNA labels to non-RNA modalities using kNN in GLUE space",
    )
    p.add_argument("--transfer-k", type=int, default=25, help="k for RNA-to-non-RNA label transfer")
    p.add_argument(
        "--transfer-min-confidence",
        type=float,
        default=0.5,
        help="If winning vote fraction is below this threshold, assign 'unknown'",
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

    umap_mod_png = out_dir / "umap_by_modality.png"
    save_scanpy_umap(adata, color="modality_key", out_path=umap_mod_png, title="Joint scGLUE UMAP by modality")

    umap_mod_facets_png = out_dir / "umap_by_modality_facets.png"
    save_scanpy_modality_facets(adata, out_path=umap_mod_facets_png)

    umap_cluster_png = out_dir / "umap_by_leiden.png"
    save_scanpy_umap(adata, color="leiden", out_path=umap_cluster_png, title="Joint scGLUE UMAP by Leiden cluster")

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
    harmonization_path = resolve_path(repo_root, args.harmonization_tsv)
    annotated_cells_tsv = None
    umap_celltype_png = None
    transferred_cells_tsv = None
    umap_joint_harm_coarse_png = None
    umap_joint_harm_fine_png = None
    annotation_summary = {
        "annotation_tar": str(annotation_path) if annotation_path and annotation_path.exists() else None,
        "harmonization_tsv": str(harmonization_path) if harmonization_path and harmonization_path.exists() else None,
        "cell_type_level": args.cell_type_level,
        "rna_annotation_overlay_written": False,
        "joint_transfer_written": False,
    }
    if annotation_path and annotation_path.exists():
        ann_raw = load_rna_cell_annotations(annotation_path)
        harm = load_harmonization_table(harmonization_path) if harmonization_path and harmonization_path.exists() else None
        ann, harm_summary = apply_harmonization(ann_raw, harm)
        label_level = normalize_label_level(args.cell_type_level)
        rna_cells = out_cells.loc[out_cells["modality_key"].astype(str) == "rna"].copy()
        rna_cells = rna_cells.merge(
            ann[
                [
                    "barcode_core",
                    "coarse_cell_type",
                    "fine_cell_type",
                    "harmonized_coarse",
                    "harmonized_fine",
                    "cell_ontology_id",
                ]
            ],
            on="barcode_core",
            how="left",
        )
        rna_cells.index = rna_cells["obs_id"].astype(str)
        annotated_cells_tsv = out_dir / f"rna_umap_{label_level}.tsv"
        rna_cells.to_csv(annotated_cells_tsv, sep="\t", index=False)
        plot_label_overlay(
            adata=adata,
            values=rna_cells[label_level],
            out_path=out_dir / f"umap_rna_{label_level}.png",
            title=f"Joint UMAP with RNA {label_level}",
        )
        umap_celltype_png = out_dir / f"umap_rna_{label_level}.png"
        annotation_summary["rna_annotation_overlay_written"] = umap_celltype_png.exists()
        annotation_summary["harmonization"] = harm_summary

        if args.transfer_labels:
            transferred = transfer_labels_from_rna(
                out_cells=out_cells,
                x=x,
                ann=ann,
                k=args.transfer_k,
                min_confidence=args.transfer_min_confidence,
            )
            transferred_cells_tsv = out_dir / "joint_harmonized_label_transfer.tsv"
            transferred.to_csv(transferred_cells_tsv, sep="\t", index_label="obs_id")

            transfer_plot = out_cells.join(
                transferred[["harmonized_coarse", "harmonized_fine", "coarse_confidence", "fine_confidence", "label_source"]],
                how="left",
            )
            umap_joint_harm_coarse_png = out_dir / "umap_joint_harmonized_coarse.png"
            plot_label_overlay(
                adata=adata,
                values=transfer_plot["harmonized_coarse"],
                out_path=umap_joint_harm_coarse_png,
                title="Joint UMAP with transferred harmonized coarse labels",
            )
            umap_joint_harm_fine_png = out_dir / "umap_joint_harmonized_fine.png"
            plot_label_overlay(
                adata=adata,
                values=transfer_plot["harmonized_fine"],
                out_path=umap_joint_harm_fine_png,
                title="Joint UMAP with transferred harmonized fine labels",
            )
            annotation_summary["joint_transfer_written"] = True
            annotation_summary["transfer"] = {
                "k": args.transfer_k,
                "min_confidence": args.transfer_min_confidence,
                "n_unknown_coarse": int((transferred["harmonized_coarse"].astype(str) == "unknown").sum()),
                "n_unknown_fine": int((transferred["harmonized_fine"].astype(str) == "unknown").sum()),
            }

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
        "annotation_overlay": annotation_summary,
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
    if transferred_cells_tsv:
        print(f"Wrote: {transferred_cells_tsv}")
    if umap_joint_harm_coarse_png and umap_joint_harm_coarse_png.exists():
        print(f"Wrote: {umap_joint_harm_coarse_png}")
    if umap_joint_harm_fine_png and umap_joint_harm_fine_png.exists():
        print(f"Wrote: {umap_joint_harm_fine_png}")
    print(f"Wrote: {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
