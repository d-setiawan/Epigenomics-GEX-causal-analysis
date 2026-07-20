#!/usr/bin/env python3
"""Measure the usable-sample-size ceiling as a function of metacell granularity.

Why this exists
---------------
Our honest n is the number of *complete* metacells — those containing at least one
cell from all 6 histone marks AND RNA. Clustering more finely buys more rows but
makes completeness harder, so n_complete does not rise monotonically with
resolution: there is an optimum, and nobody had measured where it is.

The trade-off has two arms and this script quantifies both:
  more metacells  -> larger n            -> better CI-test power
  fewer cells each -> noisier aggregates -> smaller effect size

It clusters once per resolution over the shared scGLUE latent (neighbors graph
built a single time, so this is cheap) and reports, per resolution:
  n_metacells, n_complete (usable n), cells per complete metacell, and which
  modality is the binding constraint on completeness.

    python CausalDiscovery/scripts/metacell_resolution_sweep.py
    python CausalDiscovery/scripts/metacell_resolution_sweep.py --targets 150,300,600 --cell-type tcell
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--label-column", default="harmonized_coarse")
    p.add_argument("--cell-type", default="monocyte")
    p.add_argument("--min-label-confidence", type=float, default=0.5)
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--targets", default="100,150,250,400,600,900",
                   help="comma-separated target metacell counts to sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "CausalDiscovery/outputs/metacells/resolution_sweep")
    a = p.parse_args()

    import scanpy as sc
    import anndata as ad

    train_dir = a.train_dir or REPO / f"Integration/outputs/scglue/joint/{a.run_id}/train"
    emb = pd.read_csv(train_dir / "all_cells_glue_embeddings.tsv", sep="\t")
    labels = pd.read_csv(train_dir / "validation" / "joint_harmonized_label_transfer.tsv", sep="\t")

    glue_cols = [c for c in emb.columns if c.startswith("GLUE_")]

    def obs_id(df):
        return df["modality_key"].astype(str) + "::" + df["cell"].astype(str)

    emb = emb.copy(); emb["obs_id"] = obs_id(emb)
    if "obs_id" not in labels.columns:
        labels = labels.copy(); labels["obs_id"] = obs_id(labels)
    conf_cols = [c for c in labels.columns if c.endswith("confidence")]
    merged = emb.merge(labels[["obs_id", a.label_column] + conf_cols], on="obs_id", how="left")

    keep = merged[a.label_column].astype(str) == a.cell_type
    if conf_cols:
        keep &= merged[conf_cols[0]].fillna(0.0) >= a.min_label_confidence
    cells = merged.loc[keep].copy()
    print(f"[sweep] {a.cell_type} cells in latent: {len(cells)}")

    adata = ad.AnnData(X=cells[glue_cols].to_numpy(dtype=float),
                       obs=cells[["obs_id", "cell", "modality_key", "mark"]].reset_index(drop=True))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, random_state=a.seed)  # built once
    marks = [m.strip() for m in a.marks.split(",") if m.strip()]

    def cluster(res: float) -> int:
        sc.tl.leiden(adata, resolution=res, random_state=a.seed, key_added="mc",
                     flavor="igraph", n_iterations=2, directed=False)
        return adata.obs["mc"].nunique()

    rows = []
    for target in [int(t) for t in a.targets.split(",")]:
        # same resolution search metacell_assign.py uses, so results are comparable
        res, k = 1.0, cluster(1.0)
        for _ in range(10):
            if abs(k - target) <= max(5, 0.1 * target):
                break
            res = float(np.clip(res * target / max(k, 1), 0.05, 200.0))
            k = cluster(res)

        obs = adata.obs.copy()
        obs["modality"] = np.where(obs["modality_key"].astype(str) == "rna", "rna", "chromatin")
        comp = obs.pivot_table(index="mc", columns="mark", values="obs_id",
                               aggfunc="count", fill_value=0)
        rna = obs[obs["modality"] == "rna"].groupby("mc").size().rename("rna")
        comp = comp.join(rna, how="outer").fillna(0).astype(int)
        for m in marks + ["rna"]:
            if m not in comp.columns:
                comp[m] = 0
        comp["complete"] = (comp[marks] > 0).all(axis=1) & (comp["rna"] > 0)

        sizes = obs.groupby("mc").size()
        complete_ids = comp.index[comp["complete"]]
        # which modality most often blocks completeness?
        missing = {m: int((comp[m] == 0).sum()) for m in marks + ["rna"]}
        binding = max(missing, key=missing.get)

        rows.append({
            "target": target, "resolution": round(res, 3), "n_metacells": int(k),
            "n_complete": int(comp["complete"].sum()),
            "pct_complete": round(100 * comp["complete"].mean(), 1),
            "median_cells_per_mc": int(sizes.median()),
            "median_cells_per_complete_mc": int(sizes.loc[complete_ids].median()) if len(complete_ids) else 0,
            "min_rna_in_complete": int(comp.loc[complete_ids, "rna"].min()) if len(complete_ids) else 0,
            "binding_modality": binding, "n_missing_binding": missing[binding],
        })
        print(f"  target={target:>4}  res={res:>6.3f}  metacells={k:>4}  "
              f"complete={rows[-1]['n_complete']:>4} ({rows[-1]['pct_complete']:>5.1f}%)  "
              f"median cells/complete={rows[-1]['median_cells_per_complete_mc']:>4}  "
              f"binding={binding}")

    df = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_dir / f"resolution_sweep_{a.cell_type}.tsv", sep="\t", index=False)

    best = df.loc[df.n_complete.idxmax()]
    cur = df.loc[(df.target - 150).abs().idxmin()]
    (a.out_dir / f"resolution_sweep_{a.cell_type}.json").write_text(json.dumps({
        "cell_type": a.cell_type, "n_cells": len(cells), "sweep": rows,
        "best_n_complete": int(best.n_complete), "best_target": int(best.target),
        "current_n_complete": int(cur.n_complete),
        "headroom_factor": round(float(best.n_complete) / max(float(cur.n_complete), 1), 2),
    }, indent=2))

    print(f"\n[sweep] usable n peaks at {int(best.n_complete)} complete metacells "
          f"(target={int(best.target)}, res={best.resolution})")
    print(f"[sweep] current config gives {int(cur.n_complete)} -> "
          f"headroom {float(best.n_complete)/max(float(cur.n_complete),1):.2f}x")
    print(f"[sweep] wrote {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
