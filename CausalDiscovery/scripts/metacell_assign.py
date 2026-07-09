#!/usr/bin/env python3
"""Assign non-overlapping metacells over the scGLUE joint latent (native scanpy).

Replaces overlapping kNN neighbor-averaging with disjoint metacells: partition the
monocyte cells (RNA + all 6 histone-mark populations live in the SAME GLUE latent)
into K clusters via Leiden over-clustering; each cell belongs to exactly one metacell.
Because the design is UNPAIRED (RNA cells and each mark's cells are different cells),
a metacell must contain cells from RNA AND every requested mark to aggregate a full
row -- this tool reports that composition so we can pick the graining.

No new dependencies (scanpy/leidenalg already in scrna_r). Run with the scrna_r python:
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python CausalDiscovery/scripts/metacell_assign.py

Outputs: cell_metacell_assignments.tsv (obs_id, modality, mark, metacell_id) and
metacell_composition.tsv (per metacell: #rna + #cells per mark) + a summary JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--label-column", default="harmonized_coarse")
    p.add_argument("--cell-type", default="monocyte")
    p.add_argument("--min-label-confidence", type=float, default=0.5)
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--target-metacells", type=int, default=150,
                   help="Approx number of metacells (tunes Leiden resolution by search).")
    p.add_argument("--resolution", type=float, default=None,
                   help="Explicit Leiden resolution (overrides --target-metacells search).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args()

    import scanpy as sc
    import anndata as ad

    train_dir = a.train_dir or repo_root / f"integration/outputs/scglue/joint/{a.run_id}/train"
    emb = pd.read_csv(train_dir / "all_cells_glue_embeddings.tsv", sep="\t")
    labels = pd.read_csv(train_dir / "validation" / "joint_harmonized_label_transfer.tsv", sep="\t")

    glue_cols = [c for c in emb.columns if c.startswith("GLUE_")]
    # obs_id convention used across the repo: "<modality_key>::<cell>"
    def obs_id(df):
        return df["modality_key"].astype(str) + "::" + df["cell"].astype(str)
    emb = emb.copy(); emb["obs_id"] = obs_id(emb)
    if "obs_id" not in labels.columns:
        labels = labels.copy(); labels["obs_id"] = obs_id(labels)
    merged = emb.merge(labels[["obs_id", a.label_column] + [c for c in labels.columns if c.endswith("confidence")]],
                       on="obs_id", how="left")
    keep = merged[a.label_column].astype(str) == a.cell_type
    conf = [c for c in merged.columns if c.endswith("confidence")]
    if conf:
        keep &= merged[conf[0]].fillna(0.0) >= a.min_label_confidence
    cells = merged.loc[keep].copy()
    print(f"[metacell] monocyte cells in latent: {len(cells)}")

    X = cells[glue_cols].to_numpy(dtype=float)
    adata = ad.AnnData(X=X, obs=cells[["obs_id", "cell", "modality_key", "mark"]].reset_index(drop=True))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, random_state=a.seed)

    def cluster(res):
        sc.tl.leiden(adata, resolution=res, random_state=a.seed, key_added="metacell_id", flavor="igraph", n_iterations=2, directed=False)
        return adata.obs["metacell_id"].nunique()

    if a.resolution is not None:
        res = a.resolution; k = cluster(res)
    else:
        # simple search for a resolution near the target metacell count
        res, k = 1.0, cluster(1.0)
        for _ in range(8):
            if abs(k - a.target_metacells) <= max(5, 0.1 * a.target_metacells):
                break
            res *= a.target_metacells / max(k, 1)
            res = float(np.clip(res, 0.05, 60.0))
            k = cluster(res)
    print(f"[metacell] resolution={res:.3f} -> {k} metacells")

    obs = adata.obs.copy()
    obs["modality"] = np.where(obs["modality_key"].astype(str) == "rna", "rna", "chromatin")

    # composition: per metacell, count RNA + each mark
    marks = [m.strip() for m in a.marks.split(",") if m.strip()]
    comp = obs.pivot_table(index="metacell_id", columns="mark", values="obs_id", aggfunc="count", fill_value=0)
    rna_counts = obs[obs["modality"] == "rna"].groupby("metacell_id").size().rename("rna")
    comp = comp.join(rna_counts, how="outer").fillna(0).astype(int)
    for m in marks + ["rna"]:
        if m not in comp.columns:
            comp[m] = 0
    comp["complete"] = ((comp[marks] > 0).all(axis=1)) & (comp["rna"] > 0)

    out_dir = a.out_dir or repo_root / "CausalDiscovery/outputs/metacells" / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    obs[["obs_id", "cell", "modality_key", "mark", "modality", "metacell_id"]].to_csv(
        out_dir / "cell_metacell_assignments.tsv", sep="\t", index=False)
    comp.reset_index().to_csv(out_dir / "metacell_composition.tsv", sep="\t", index=False)

    n_complete = int(comp["complete"].sum())
    summary = {
        "n_cells": len(cells), "n_metacells": int(k), "resolution": res,
        "target_metacells": a.target_metacells,
        "n_complete_metacells": n_complete,
        "median_cells_per_metacell": int(obs.groupby("metacell_id").size().median()),
        "per_mark_median_cells": {m: int(comp[m].median()) for m in marks + ["rna"]},
        "min_cells_per_mark_over_complete": {m: int(comp.loc[comp["complete"], m].min()) if n_complete else 0 for m in marks + ["rna"]},
    }
    (out_dir / "metacell_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[metacell] complete metacells (all marks + rna present): {n_complete}/{k}")
    print(f"[metacell] median cells/metacell: {summary['median_cells_per_metacell']}  per-mark median: {summary['per_mark_median_cells']}")
    print(f"[metacell] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
