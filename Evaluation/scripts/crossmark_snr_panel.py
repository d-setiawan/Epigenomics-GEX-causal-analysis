"""
Signal-share audit across the whole gene panel.

For every variable in every gene's panel, split its metacell-level variance
into the part that is real between-state signal and the part that is the
sampling noise of averaging ~10 cells:

    Var(X_j)  =  Var(mu_j)              +  E_c[ Var(V_j | c) / n_c ]
                 between-state signal      noise of the metacell mean

    signal share  =  1  -  noise / total

The noise term is estimated PER CLUSTER (E_c[S_c/n_c]) rather than as
Sw_pooled * E[1/n].  The pooled form is degrees-of-freedom weighted toward
large metacells while E[1/n] is dominated by small ones, and since small
Leiden clusters are internally tighter the product overestimates -- it returns
noise shares above 1, which is an impossible decomposition and the tell that
the estimator is wrong.

A variable with signal share <= 0 carries nothing this design can distinguish
from noise.  No search, no CI test, and no covariance repair can recover
structure from it, because there is no structure in it to recover.  This is
therefore a check that has to pass BEFORE any question about identifiability
or about which LV algorithm to use is worth asking.

Marks are loaded once and reused across genes; the per-gene work is only the
peak selection and the variance decomposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix, per_cell_values  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def decompose_panel(cols, ch, cell_vals, cell_mc, order, min_cells=2):
    """Per-variable (signal, noise, total) plus the same/cross correlation split."""
    p = len(cols)
    X = np.column_stack([
        np.array([np.nanmean(cell_vals[c][cell_mc[c] == m])
                  if np.any(cell_mc[c] == m) else np.nan for m in order])
        for c in cols])
    good = np.isfinite(X).all(axis=1)
    X = X[good]
    if X.shape[0] < 20:
        return None

    raw = np.cov(X, rowvar=False)
    noise = np.zeros((p, p))
    for a_i in np.unique(ch):
        vs = np.flatnonzero(ch == a_i)
        m = cell_mc[cols[vs[0]]]
        Vc = np.column_stack([cell_vals[cols[j]] for j in vs])
        acc, used = np.zeros((len(vs), len(vs))), 0
        for c in order:
            sel = m == c
            k = int(sel.sum())
            if k < min_cells:
                continue
            blk = Vc[sel] - Vc[sel].mean(axis=0, keepdims=True)
            acc += (blk.T @ blk) / (k - 1) / k
            used += 1
        if used:
            noise[np.ix_(vs, vs)] = acc / used

    tot = np.clip(np.diag(raw), 1e-12, None)
    share = 1.0 - np.diag(noise) / tot
    return dict(X=X, raw=raw, noise=noise, share=share, n_rows=X.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=str(
        REPO / "CausalDiscovery/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--assignments", default=str(
        REPO / "CausalDiscovery/outputs/metacells/joint_v2/cell_metacell_assignments.tsv"))
    ap.add_argument("--data-dir", default=str(REPO / "Data/outputs"))
    ap.add_argument("--min-cells", type=int, default=2)
    ap.add_argument("--out", default="crossmark_snr_panel.tsv")
    a = ap.parse_args()

    root = Path(a.dataset_root)
    mats = sorted(root.glob("*/nearby_peaks/*_nearby_peak_matrix.tsv"))
    print(f"{len(mats)} gene panels under {root.name}")

    assign = pd.read_csv(a.assignments, sep="\t")
    all_mods = sorted(assign["mark"].unique())
    counts = assign.groupby(["metacell_id", "mark"]).size().unstack(fill_value=0)
    order = sorted(counts.index[(counts[all_mods] > 0).all(axis=1)].to_numpy())
    print(f"complete metacells: {len(order)} of {counts.shape[0]}")

    # every peak needed, mark by mark, so each matrix is read once
    want = {}
    per_gene_cols = {}
    for m in mats:
        df = pd.read_csv(m, sep="\t", nrows=1)
        cols = [c for c in df.columns if c.startswith("chr") and "__" in c]
        gene = m.name.split("_nearby")[0]
        per_gene_cols[gene] = cols
        for c in cols:
            reg, mark = c.rsplit("__", 1)
            want.setdefault(mark, set()).add(reg)

    cell_vals, cell_mc = {}, {}
    for mark, regs in sorted(want.items()):
        sub = assign[assign["mark"] == mark]
        if sub.empty:
            print(f"  ! {mark}: absent from assignments")
            continue
        x, feats, bcs = load_mark_matrix(Path(a.data_dir) / mark, mark)
        vals = per_cell_values(x, feats, sorted(regs))
        pos = {b: i for i, b in enumerate(bcs)}
        keep = sub["cell"].map(pos)
        ok = keep.notna().to_numpy()
        rows = keep[ok].to_numpy(int)
        mcs = sub["metacell_id"].to_numpy()[ok]
        for reg in regs:
            col = f"{reg}__{mark}"
            cell_vals[col] = vals[reg][rows]
            cell_mc[col] = mcs
        print(f"  {mark:9s} {len(regs):4d} peaks  {ok.sum():5d} cells")

    marks = sorted(want)
    rows_out = []
    for gene, cols in sorted(per_gene_cols.items()):
        cols = [c for c in cols if c in cell_vals]
        if len(cols) < 3:
            continue
        ch = np.array([marks.index(c.rsplit("__", 1)[1]) for c in cols])
        res = decompose_panel(cols, ch, cell_vals, cell_mc, order, a.min_cells)
        if res is None:
            continue
        sh = res["share"]
        for j, c in enumerate(cols):
            rows_out.append(dict(gene=gene, variable=c,
                                 mark=c.rsplit("__", 1)[1],
                                 signal_share=float(sh[j]),
                                 total_var=float(res["raw"][j, j]),
                                 noise_var=float(res["noise"][j, j]),
                                 n_rows=res["n_rows"]))
        print(f"  {gene:8s} {len(cols):3d} vars   median share {np.median(sh):+.3f}   "
              f"usable(>0.10) {int((sh > 0.10).sum()):3d}   "
              f"usable(>0.25) {int((sh > 0.25).sum()):3d}")

    df = pd.DataFrame(rows_out)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}  ({len(df)} variables over {df.gene.nunique()} genes)")

    print("\n=== SIGNAL SHARE, POOLED OVER THE PANEL ===")
    print(f"  variables            : {len(df)}")
    print(f"  median signal share  : {df.signal_share.median():+.3f}")
    print(f"  share > 0            : {(df.signal_share > 0).mean()*100:5.1f}%  "
          f"(50% expected if the true share is 0 everywhere)")
    print(f"  share > 0.10         : {(df.signal_share > 0.10).mean()*100:5.1f}%")
    print(f"  share > 0.25         : {(df.signal_share > 0.25).mean()*100:5.1f}%")
    print(f"  share > 0.50         : {(df.signal_share > 0.50).mean()*100:5.1f}%")

    print("\n=== BY MARK ===")
    g = df.groupby("mark").agg(n=("signal_share", "size"),
                               median=("signal_share", "median"),
                               pct_gt_10=("signal_share", lambda s: 100*(s > 0.10).mean()),
                               pct_gt_25=("signal_share", lambda s: 100*(s > 0.25).mean()))
    print(g.round(3).to_string())

    print("\n=== BY GENE ===")
    g2 = df.groupby("gene").agg(n=("signal_share", "size"),
                                median=("signal_share", "median"),
                                n_usable=("signal_share", lambda s: int((s > 0.10).sum())))
    print(g2.round(3).to_string())

    json.dump({"median_share": float(df.signal_share.median()),
               "pct_gt_10": float((df.signal_share > 0.10).mean()),
               "n_variables": int(len(df)), "n_genes": int(df.gene.nunique())},
              open(str(Path(a.out).with_suffix(".json")), "w"), indent=1)


if __name__ == "__main__":
    main()
