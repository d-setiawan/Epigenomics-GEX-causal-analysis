"""
Where is the aggregation optimum? Signal share vs metacell count.

The audit in crossmark_snr_panel.py says the median panel variable has no
between-state signal: its metacell-level variance is fully accounted for by the
noise of averaging ~11 cells per mark. That noise term is Sigma_within / n, so it
falls as metacells get coarser -- and the number of rows available to the search
falls with it. There is an optimum, and it is computable rather than a matter of
taste:

    signal share(n)  =  1  -  E_c[ Var(V|c) / n_c ]  /  Var_c( metacell means )
    rows(n)          =  number of complete metacells

Finer metacells buy rows and lose signal per variable. Coarser metacells buy
signal and lose rows. The current setting (148 requested, 133 complete, ~11 cells
per mark) sits at one end of that curve, and the audit says it is the wrong end.

METHOD
------
Rather than re-running Leiden at many resolutions -- which would need the scGLUE
latent and would confound the comparison with clustering changes -- this
agglomerates the EXISTING metacells by their mean profiles (Ward on the
metacell x variable matrix). Merging metacells is exactly what a coarser
resolution would have done to first order, and it holds the underlying embedding,
the cells, and the completeness rule fixed, so the only thing varying is
granularity.

The curve is therefore a lower bound on what re-clustering could achieve, and an
honest one: any real coarser clustering has at least this much signal available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix, per_cell_values  # noqa: E402
from crossmark_screening_stability import complete_metacells  # noqa: E402

REPO = Path(__file__).resolve().parents[4]


def shares_at(cell_vals, cell_mc, cols, groups, min_cells=2):
    """Signal share per variable when metacells are merged per `groups`."""
    keys = sorted(set(groups.values()))
    p = len(cols)
    X = np.full((len(keys), p), np.nan)
    noise = np.full(p, np.nan)
    pos = {g: i for i, g in enumerate(keys)}

    for j, c in enumerate(cols):
        v, mc = cell_vals[c], cell_mc[c]
        g = np.array([groups.get(m, -1) for m in mc])
        acc, used = 0.0, 0
        for gi in keys:
            sel = g == gi
            k = int(sel.sum())
            if k == 0:
                continue
            X[pos[gi], j] = v[sel].mean()
            if k >= min_cells:
                acc += v[sel].var(ddof=1) / k
                used += 1
        noise[j] = acc / used if used else np.nan

    good = np.isfinite(X).all(axis=1)
    if good.sum() < 5:
        return None, 0
    tot = X[good].var(axis=0, ddof=1)
    return 1.0 - noise / np.clip(tot, 1e-12, None), int(good.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=str(
        REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--assignments", default=str(
        REPO / "pipelines/empirical/analysis/outputs/metacells/joint_v2/cell_metacell_assignments.tsv"))
    ap.add_argument("--data-dir", default=str(REPO / "pipelines/empirical/data/outputs"))
    ap.add_argument("--genes", nargs="+",
                    default=["CD14", "CD151", "CD52", "IL4R", "PTPRJ", "LGALS2"])
    ap.add_argument("--levels", nargs="+", type=int,
                    default=[133, 100, 80, 60, 45, 35, 25, 18, 12])
    ap.add_argument("--out", default="crossmark_resolution_tradeoff.tsv")
    a = ap.parse_args()

    assign = pd.read_csv(a.assignments, sep="\t")
    order = complete_metacells(assign)
    print(f"base: {len(order)} complete metacells, {len(assign)} cells")

    root = Path(a.dataset_root)
    rows = []
    for gene in a.genes:
        m = list(root.glob(f"{gene.lower()}/nearby_peaks/*_nearby_peak_matrix.tsv"))
        if not m:
            continue
        cols = [c for c in pd.read_csv(m[0], sep="\t", nrows=1).columns
                if c.startswith("chr") and "__" in c]
        want = {}
        for c in cols:
            r, k = c.rsplit("__", 1)
            want.setdefault(k, set()).add(r)

        cell_vals, cell_mc = {}, {}
        for mark, regs in sorted(want.items()):
            x, feats, bcs = load_mark_matrix(Path(a.data_dir) / mark, mark)
            vals = per_cell_values(x, feats, sorted(regs))
            sub = assign[assign["mark"] == mark]
            pos = {b: i for i, b in enumerate(bcs)}
            idx = sub["cell"].map(pos)
            ok = idx.notna().to_numpy()
            for r in regs:
                cell_vals[f"{r}__{mark}"] = vals[r][idx[ok].to_numpy(int)]
                cell_mc[f"{r}__{mark}"] = sub["metacell_id"].to_numpy()[ok]

        # metacell x variable profile, used only to decide which metacells merge
        base = np.full((len(order), len(cols)), np.nan)
        for j, c in enumerate(cols):
            v, mc = cell_vals[c], cell_mc[c]
            for i, mm in enumerate(order):
                sel = mc == mm
                if sel.any():
                    base[i, j] = v[sel].mean()
        keep_c = np.isfinite(base).all(axis=0)
        prof = base[:, keep_c]
        prof = (prof - prof.mean(0)) / np.clip(prof.std(0), 1e-9, None)
        Z = linkage(prof, method="ward")

        print(f"\n{gene}: {len(cols)} variables")
        print(f"  {'metacells':>10} {'rows':>6} {'cells/mark':>11} "
              f"{'median share':>13} {'>0.25':>7} {'>0.50':>7}")
        for lvl in a.levels:
            lvl = min(lvl, len(order))
            lab = fcluster(Z, t=lvl, criterion="maxclust")
            groups = {mm: int(lab[i]) for i, mm in enumerate(order)}
            sh, nrow = shares_at(cell_vals, cell_mc, cols, groups)
            if sh is None:
                continue
            per = np.array([np.sum([groups.get(mm, -1) == g
                                    for mm in cell_mc[cols[0]]])
                            for g in sorted(set(groups.values()))])
            cpm = float(np.median(per[per > 0]))
            rows.append(dict(gene=gene, level=int(lvl), n_rows=nrow,
                             cells_per_mark=cpm,
                             median_share=float(np.nanmedian(sh)),
                             frac_gt25=float(np.nanmean(sh > 0.25)),
                             frac_gt50=float(np.nanmean(sh > 0.50)),
                             n_vars=len(cols)))
            print(f"  {lvl:>10} {nrow:>6} {cpm:>11.1f} "
                  f"{np.nanmedian(sh):>+13.3f} "
                  f"{100*np.nanmean(sh > 0.25):>6.1f}% {100*np.nanmean(sh > 0.50):>6.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}")
    if df.empty:
        return
    print("\n=== POOLED OVER GENES ===")
    g = df.groupby("level").agg(rows=("n_rows", "mean"),
                                cells_per_mark=("cells_per_mark", "mean"),
                                median_share=("median_share", "mean"),
                                frac_gt25=("frac_gt25", "mean"),
                                frac_gt50=("frac_gt50", "mean"))
    print(g.round(3).to_string())
    print("\nThe search needs BOTH: enough rows for the CI test and enough")
    print("signal per variable to have something to test.  Read the optimum off")
    print("the point where frac_gt25 stops rising faster than rows falls.")


if __name__ == "__main__":
    main()
