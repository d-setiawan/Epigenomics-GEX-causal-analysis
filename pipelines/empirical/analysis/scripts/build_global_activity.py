#!/usr/bin/env python3
"""Build a per-metacell GLOBAL chromatin-activity covariate (leave-one-locus-out).

The global-activity-latent probe (peak_peak_validation_results.md): all three
biological axes are null, so the bidirected `<->` driver is most likely a shared
GLOBAL activity/state latent. To test that WITHOUT conditioning on a collider, the
proxy for gene g must be a *common cause* of g's peaks — computed from peaks OUTSIDE
g's locus. We use the mean z-scored accessibility across all OTHER panel genes'
peaks (all matrices share the same 133 metacells).

For each gene it writes an augmented matrix = original + `activity__global` into a
new dataset root, which `--confounders depth,activity` turns into a
`conf__global_activity` root node. Re-running discovery there and comparing `<->`
counts to the depth-only baseline says whether global state explains the edges.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]


def zscore_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd  # metacells x peaks, z-scored per peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2/"
                                   "monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    ap.add_argument("--dst-root", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2/"
                                   "monocyte_cuttag_peak_genes_300kb_metacell_expanded_globalact")
    ap.add_argument("--key", default="sample_id")
    a = ap.parse_args()

    genes = sorted(p.name for p in a.src_root.iterdir()
                   if (p / "nearby_peaks").is_dir())
    mats = {}
    for g in genes:
        hits = list((a.src_root / g / "nearby_peaks").glob("*_nearby_peak_matrix.tsv"))
        if hits:
            mats[g] = pd.read_csv(hits[0], sep="\t")
    # common metacell order (all share the same key set + order, but align to be safe)
    ref_key = list(mats[genes[0]][a.key])
    print(f"[globalact] {len(mats)} genes, {len(ref_key)} metacells")

    # accumulate global sum of z-scored peak values + peak count, per gene locus
    locus_sum: dict[str, np.ndarray] = {}
    locus_n: dict[str, int] = {}
    for g, df in mats.items():
        df = df.set_index(a.key).loc[ref_key].reset_index()  # align rows
        mats[g] = df
        peaks = [c for c in df.columns if str(c).startswith("chr")]
        z = zscore_cols(df, peaks)                 # metacells x peaks
        locus_sum[g] = np.nansum(z, axis=1)        # sum over this locus's peaks
        locus_n[g] = len(peaks)

    global_sum = np.sum(list(locus_sum.values()), axis=0)
    global_n = sum(locus_n.values())
    print(f"[globalact] total panel peaks = {global_n}")

    a.dst_root.mkdir(parents=True, exist_ok=True)
    for g, df in mats.items():
        # leave-THIS-locus-out mean z-accessibility = common-cause global-state proxy
        act = (global_sum - locus_sum[g]) / (global_n - locus_n[g])
        out = df.copy()
        out["activity__global"] = act
        dst = a.dst_root / g / "nearby_peaks"
        dst.mkdir(parents=True, exist_ok=True)
        hits = list((a.src_root / g / "nearby_peaks").glob("*_nearby_peak_matrix.tsv"))
        out.to_csv(dst / hits[0].name, sep="\t", index=False)
    # copy panel_summary if present
    ps = a.src_root / "panel_summary.json"
    if ps.exists():
        shutil.copy(ps, a.dst_root / "panel_summary.json")
    print(f"[globalact] wrote augmented matrices -> {a.dst_root}")
    print(f"[globalact] activity__global range: [{global_sum.min():.2f}, {global_sum.max():.2f}] (raw sum); "
          f"per-gene leave-one-out means computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
