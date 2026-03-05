#!/usr/bin/env python3
"""Export PC-ready causal input tables from joint scGLUE outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def parse_gene_list(genes_csv: str | None, gene_file: str | None) -> List[str]:
    genes: List[str] = []
    if genes_csv:
        genes.extend([g.strip() for g in genes_csv.split(",") if g.strip()])
    if gene_file:
        with Path(gene_file).open() as f:
            genes.extend([ln.strip() for ln in f if ln.strip()])
    genes = list(dict.fromkeys(genes))
    if not genes:
        raise ValueError("Provide --genes or --gene-file")
    return genes


def mean_log1p_norm_for_features(adata: ad.AnnData, feature_names: List[str], target_sum: float = 1e4) -> np.ndarray:
    if len(feature_names) == 0:
        return np.full(adata.n_obs, np.nan, dtype=float)
    feat_idx = adata.var_names.get_indexer(feature_names)
    feat_idx = feat_idx[feat_idx >= 0]
    if len(feat_idx) == 0:
        return np.full(adata.n_obs, np.nan, dtype=float)

    x = adata.layers["counts"] if "counts" in adata.layers else adata.X
    row_sums = np.asarray(x.sum(axis=1)).ravel().astype(float)
    row_sums[row_sums <= 0] = 1.0
    sub = x[:, feat_idx]
    if hasattr(sub, "multiply"):
        norm = sub.multiply(target_sum / row_sums[:, None])
        mean_vals = np.asarray(norm.mean(axis=1)).ravel()
    else:
        norm = sub * (target_sum / row_sums[:, None])
        mean_vals = np.asarray(norm.mean(axis=1)).ravel()
    return np.log1p(mean_vals)


def log1p_norm_for_gene(adata: ad.AnnData, gene: str, target_sum: float = 1e4) -> np.ndarray:
    idx = adata.var_names.get_indexer([gene])[0]
    if idx < 0:
        return np.full(adata.n_obs, np.nan, dtype=float)
    x = adata.layers["counts"] if "counts" in adata.layers else adata.X
    row_sums = np.asarray(x.sum(axis=1)).ravel().astype(float)
    row_sums[row_sums <= 0] = 1.0
    col = x[:, idx]
    vals = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    vals = vals * (target_sum / row_sums)
    return np.log1p(vals)


def linked_features_for_gene(edges: pd.DataFrame, gene: str, mark: str) -> List[str]:
    prefix = f"{mark}::"
    src = edges["source"].astype(str)
    tgt = edges["target"].astype(str)
    m1 = (src == gene) & tgt.str.startswith(prefix)
    m2 = (tgt == gene) & src.str.startswith(prefix)
    feats = sorted(set(tgt[m1].tolist()) | set(src[m2].tolist()))
    return feats


def main() -> int:
    p = argparse.ArgumentParser(description="Export PC-ready causal inputs from joint scGLUE")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--train-dir", required=True, help="Joint train output directory")
    p.add_argument("--graph-dir", required=True, help="Joint graph output directory")
    p.add_argument("--genes", default=None, help="Comma-separated genes")
    p.add_argument("--gene-file", default=None, help="One gene per line")
    p.add_argument("--out-dir", default=None, help="Default: <train-dir>/causal_inputs")
    p.add_argument("--n-metacells", type=int, default=200)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--min-rna-cells", type=int, default=5)
    p.add_argument("--min-chrom-cells", type=int, default=2)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_dir = resolve_path(repo_root, args.train_dir)
    graph_dir = resolve_path(repo_root, args.graph_dir)
    out_dir = resolve_path(repo_root, args.out_dir) if args.out_dir else train_dir / "causal_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = parse_gene_list(args.genes, args.gene_file)

    all_cells_path = train_dir / "all_cells_glue_embeddings.tsv"
    modality_manifest_path = train_dir / "modality_outputs.tsv"
    edges_path = graph_dir / "guidance_edges.tsv"
    if not all(p.exists() for p in [all_cells_path, modality_manifest_path, edges_path]):
        raise FileNotFoundError("Missing required train/graph inputs")

    all_cells = pd.read_csv(all_cells_path, sep="\t")
    glue_cols = [c for c in all_cells.columns if c.startswith("GLUE_")]
    if not glue_cols:
        raise RuntimeError("No GLUE_* columns found in all_cells_glue_embeddings.tsv")
    x = all_cells[glue_cols].to_numpy(dtype=float)
    n_clusters = min(args.n_metacells, max(2, x.shape[0] // 20), x.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=args.random_seed, n_init=10)
    all_cells["metacell"] = km.fit_predict(x).astype(int)

    modality_manifest = pd.read_csv(modality_manifest_path, sep="\t")
    mod_to_h5ad = {}
    for r in modality_manifest.to_dict(orient="records"):
        key = str(r["modality_key"])
        h5ad_path = Path(str(r["h5ad"]))
        if not h5ad_path.is_absolute():
            h5ad_path = (repo_root / h5ad_path).resolve()
        mod_to_h5ad[key] = str(h5ad_path)

    if "rna" not in mod_to_h5ad:
        raise RuntimeError("Expected modality key 'rna' in modality_outputs.tsv")
    rna = ad.read_h5ad(mod_to_h5ad["rna"])

    chrom_adatas: Dict[str, ad.AnnData] = {}
    chrom_marks: Dict[str, str] = {}
    for r in modality_manifest.to_dict(orient="records"):
        key = str(r["modality_key"])
        mark = str(r["mark"])
        if key == "rna":
            continue
        chrom_adatas[key] = ad.read_h5ad(str(r["h5ad"]))
        chrom_marks[key] = mark

    edges = pd.read_csv(edges_path, sep="\t")
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    meta_counts = pd.crosstab(all_cells["metacell"], all_cells["modality_key"])
    meta_counts = meta_counts.sort_index()
    meta_counts.columns = [f"n_cells_{c}" for c in meta_counts.columns]

    cell_to_meta = all_cells.set_index("cell")["metacell"].astype(int)

    links_rows = []
    exported = []
    for gene in genes:
        expr_vals = log1p_norm_for_gene(rna, gene)
        if np.all(np.isnan(expr_vals)):
            print(f"[WARN] Gene not found in RNA matrix, skipping: {gene}")
            continue

        rna_df = pd.DataFrame({"cell": rna.obs_names.astype(str), "expr": expr_vals})
        rna_df["metacell"] = rna_df["cell"].map(cell_to_meta)
        rna_meta = rna_df.dropna(subset=["metacell"]).groupby("metacell", as_index=True)["expr"].mean()

        gene_df = pd.DataFrame(index=meta_counts.index.copy())
        gene_df.index.name = "metacell"
        gene_df[f"expr_{gene}"] = rna_meta

        for key, chrom in chrom_adatas.items():
            mark = chrom_marks[key]
            linked = linked_features_for_gene(edges, gene, mark)
            links_rows.extend(
                [{"gene": gene, "mark": mark, "feature": f} for f in linked]
            )
            signal_vals = mean_log1p_norm_for_features(chrom, linked)
            cdf = pd.DataFrame({"cell": chrom.obs_names.astype(str), f"hist_{mark}_linked_mean": signal_vals})
            cdf["metacell"] = cdf["cell"].map(cell_to_meta)
            cmeta = cdf.dropna(subset=["metacell"]).groupby("metacell", as_index=True)[f"hist_{mark}_linked_mean"].mean()
            gene_df[f"hist_{mark}_linked_mean"] = cmeta
            gene_df[f"n_linked_features_{mark}"] = len(linked)

        gene_df = gene_df.join(meta_counts, how="left")

        keep = gene_df.get("n_cells_rna", pd.Series(0, index=gene_df.index)) >= args.min_rna_cells
        for key in chrom_adatas:
            col = f"n_cells_{key}"
            if col in gene_df.columns:
                keep &= gene_df[col] >= args.min_chrom_cells
        gene_df = gene_df.loc[keep].copy()

        out_path = out_dir / f"{gene}_pc_input.tsv"
        gene_df.to_csv(out_path, sep="\t", index=True)
        exported.append({"gene": gene, "pc_input_tsv": str(out_path), "n_metacells": int(gene_df.shape[0])})
        print(f"Wrote: {out_path}")

    links_tsv = out_dir / "target_gene_linked_features.tsv"
    pd.DataFrame(links_rows).drop_duplicates().to_csv(links_tsv, sep="\t", index=False)

    summary = {
        "run_type": "joint",
        "inputs": {
            "train_dir": str(train_dir),
            "graph_dir": str(graph_dir),
            "all_cells_embeddings_tsv": str(all_cells_path),
            "modality_outputs_tsv": str(modality_manifest_path),
            "guidance_edges_tsv": str(edges_path),
        },
        "params": vars(args),
        "genes_requested": genes,
        "genes_exported": exported,
        "outputs": {
            "out_dir": str(out_dir),
            "linked_features_tsv": str(links_tsv),
        },
    }
    summary_path = out_dir / "causal_export_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {links_tsv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
