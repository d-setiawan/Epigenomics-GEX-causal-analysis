#!/usr/bin/env python3
"""Turn metacell assignments into a dataset-stage-compatible matching dir.

Consumes cell_metacell_assignments.tsv (from metacell_assign.py) and emits a
matching dir the existing `causal_cli.py dataset` stage consumes UNCHANGED:

  matched_samples.tsv       one row per (complete) metacell; sample_id=cell_rna=MC id
  neighbor_assignments.tsv   one row per (metacell, mark, member chromatin cell),
                             weight = 1/n_members  -> the peak path weight-sums to a MEAN
  metacell_rna.h5ad          pseudobulk RNA: summed raw counts over each metacell's RNA
                             cells, obs index = MC id, layers["counts"] (so the RNA join
                             on cell_rna=MC id works with no dataset-stage code change)

Only COMPLETE metacells (>=1 cell of every requested mark AND >=1 RNA cell) are kept, so
every row can be fully aggregated. Non-overlapping by construction: each cell belongs to
exactly one metacell.

Run with the scrna_r python (needs anndata):
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
    pipelines/empirical/analysis/scripts/build_metacell_matching_dir.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignments", type=Path,
                   default=repo_root / "pipelines/empirical/analysis/outputs/metacells/joint_v2/cell_metacell_assignments.tsv")
    p.add_argument("--rna-h5ad", type=Path,
                   default=repo_root / "pipelines/empirical/integration/runs/scglue_joint_v2/train/modalities/rna_with_glue.h5ad")
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--out-dir", type=Path,
                   default=repo_root / "pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__metacell")
    p.add_argument("--min-rna-cells", type=int, default=1)
    a = p.parse_args()

    import anndata as ad
    from scipy import sparse

    marks = [m.strip() for m in a.marks.split(",") if m.strip()]
    asn = pd.read_csv(a.assignments, sep="\t")
    asn["metacell_id"] = asn["metacell_id"].astype(str)

    # complete metacells: >=1 cell per mark AND >=min RNA cells
    comp = asn.pivot_table(index="metacell_id", columns="mark", values="cell", aggfunc="count", fill_value=0)
    for m in marks + ["RNA"]:
        if m not in comp.columns:
            comp[m] = 0
    complete = comp[(comp[marks] > 0).all(axis=1) & (comp["RNA"] >= a.min_rna_cells)].index.tolist()
    complete = sorted(complete, key=lambda x: int(x))
    mc_name = {mid: f"MC_{i+1:04d}" for i, mid in enumerate(complete)}
    print(f"[mc] {len(complete)} complete metacells kept (of {asn['metacell_id'].nunique()})")

    a.out_dir.mkdir(parents=True, exist_ok=True)

    # --- matched_samples.tsv (one row per metacell) ---
    rows = []
    for i, mid in enumerate(complete):
        sub = asn[asn["metacell_id"] == mid]
        rows.append({
            "sample_id": mc_name[mid], "cell_rna": mc_name[mid], "anchor_rank": i + 1,
            "support_mean_neighbor_distance": 0.0, "support_max_neighbor_distance": 0.0,
            "mapping_strategy": "metacell", "n_neighbors": int(len(sub)),
            "neighbor_weight_mode": "uniform", "harmonized_coarse": "monocyte",
            "harmonized_fine": "CD14-positive monocyte", "label_source": "metacell",
            "coarse_confidence": 1.0, "fine_confidence": 1.0, "cell_ontology_id": "",
        })
    pd.DataFrame(rows).to_csv(a.out_dir / "matched_samples.tsv", sep="\t", index=False)

    # --- neighbor_assignments.tsv (disjoint members, weight=1/n_members per mark) ---
    na_rows = []
    chrom = asn[asn["modality"] == "chromatin"]
    for mid in complete:
        for mark in marks:
            members = chrom[(chrom["metacell_id"] == mid) & (chrom["mark"] == mark)]["cell"].tolist()
            w = 1.0 / len(members)
            for rank, cell in enumerate(members, start=1):
                na_rows.append({
                    "sample_id": mc_name[mid], "cell_rna": mc_name[mid], "target_mark": mark,
                    "neighbor_rank": rank, "matched_cell": cell, "match_distance": 0.0, "weight": w,
                })
    na = pd.DataFrame(na_rows)
    na.to_csv(a.out_dir / "neighbor_assignments.tsv", sep="\t", index=False)
    print(f"[mc] neighbor_assignments: {len(na)} rows; per-mark member medians:")
    print(na.groupby("target_mark").apply(lambda d: d.groupby("sample_id").size().median(), include_groups=False).to_string())

    # --- metacell RNA h5ad (summed raw counts per metacell) ---
    print("[mc] building metacell pseudobulk RNA ...")
    rna = ad.read_h5ad(a.rna_h5ad)
    counts = rna.layers["counts"]
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    counts = counts.tocsr()
    bc_to_row = {bc: i for i, bc in enumerate(rna.obs_names.astype(str))}
    rna_asn = asn[asn["modality"] == "rna"]

    mats, names, n_missing = [], [], 0
    for mid in complete:
        cells = rna_asn[rna_asn["metacell_id"] == mid]["cell"].astype(str).tolist()
        idx = [bc_to_row[c] for c in cells if c in bc_to_row]
        n_missing += len(cells) - len(idx)
        if not idx:
            continue
        mats.append(np.asarray(counts[idx].sum(axis=0)).ravel())
        names.append(mc_name[mid])
    X = sparse.csr_matrix(np.vstack(mats))
    mc_rna = ad.AnnData(X=X.copy(), obs=pd.DataFrame(index=names), var=rna.var.copy())
    mc_rna.layers["counts"] = X
    out_h5ad = a.out_dir / "metacell_rna.h5ad"
    mc_rna.write_h5ad(out_h5ad)
    print(f"[mc] wrote {out_h5ad}  ({mc_rna.n_obs} metacells x {mc_rna.n_vars} genes; {n_missing} member cells missing from RNA h5ad)")
    print(f"[mc] matching dir ready: {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
