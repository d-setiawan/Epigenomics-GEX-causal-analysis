"""
Is "no between-state signal" a property of the data, or of the Gaussian framing?

THE CIRCULARITY THIS ADDRESSES
------------------------------
`crossmark_snr_panel.py` reports that the median panel variable has a signal
share <= 0: its whole between-metacell variance is accounted for by the sampling
noise of averaging ~11 cells.  But that statistic is a *variance* decomposition
computed on log1p(CPM) values, so it presupposes that variance is the right
measure of information.  For sparse counts that is a strong assumption -- under a
Poisson-like model Var ~ mean, so counting noise mechanically eats the variance
of exactly the low-count features a count-native model would still learn from.

A conclusion drawn from a second-moment statistic cannot by itself rule out
signal that lives in the count distribution.  So the same question is asked here
four ways, three of them not variance decompositions:

  gaussian_share   1 - E_c[Var(V|c)/n_c] / Var_c(mean)   on log1p(CPM).  The
                   current statistic, reproduced for comparison.

  poisson_share    Count-native.  Pool RAW counts per metacell (Y_c) with the
                   pooled library size as offset (N_c), and compare the observed
                   dispersion of the rates Y_c/N_c against the Poisson variance
                   the same totals imply.  This is the analogue of gaussian_share
                   with the correct noise model for counts, and it does not
                   depend on the log1p(CPM) transform at all.

  deviance_p       Count-native TEST rather than effect size.  Poisson deviance
                   of the saturated model (a rate per metacell) against a single
                   shared rate, chi2 on n_metacells-1 df.  Directly asks "does
                   this feature's rate vary across metacells beyond counting
                   noise", with no Gaussian assumption anywhere.

  kw_p             Distribution-free.  Kruskal-Wallis across metacells on the
                   cell-level values.  Assumes nothing about the distribution,
                   only that metacells could shift it.

WHY POOLING MATTERS INDEPENDENTLY
---------------------------------
The pipeline computes the MEAN OF PER-CELL log1p(CPM).  The count-native route
computes log1p(CPM OF POOLED COUNTS).  These are not the same statistic, and for
sparse features the difference is large: the mean-of-logs is dominated by how
many cells had any count at all, while pooling first uses the full depth of the
metacell.  This is the standard pseudobulk argument, and it is a candidate
explanation for low signal share that has nothing to do with biology.

If the count-native measures agree with the Gaussian one, the noise-domination
finding is a property of the data and the earlier conclusion stands on firmer
ground than it did.  If they disagree, the conclusion was an artefact of the
transform and the whole cross-mark question deserves re-opening on a count model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix, parse_interval  # noqa: E402
from crossmark_screening_stability import complete_metacells  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def per_cell_counts(x: sparse.csr_matrix, feats, regions):
    """
    RAW summed count per cell for each region, plus each cell's library size.

    Deliberately NOT normalised and NOT logged: the count-native measures need
    the integers and the depth separately, which the production quantification
    (a per-cell mean of log1p(CPM)) has already destroyed by this stage.
    """
    depth = np.asarray(x.sum(axis=1)).ravel().astype(float)
    parsed = [parse_interval(f) for f in feats]
    by_chrom = {}
    for i, (c, s, e) in enumerate(parsed):
        by_chrom.setdefault(c, []).append((s, e, i))

    out = {}
    for reg in regions:
        rc, rs, re_ = parse_interval(reg)
        idx = np.array(sorted(i for (s, e, i) in by_chrom.get(rc, [])
                              if s < re_ and rs < e), dtype=int)
        out[reg] = (np.asarray(x[:, idx].sum(axis=1)).ravel().astype(float)
                    if idx.size else np.zeros(x.shape[0]))
    return out, depth


def gaussian_share(vals_log, mc, order, min_cells=2):
    """The current statistic: 1 - E_c[Var/n] / Var(metacell means)."""
    means, noise, used = [], 0.0, 0
    for c in order:
        sel = mc == c
        k = int(sel.sum())
        if k == 0:
            continue
        means.append(vals_log[sel].mean())
        if k >= min_cells:
            noise += vals_log[sel].var(ddof=1) / k
            used += 1
    if len(means) < 5 or used == 0:
        return np.nan
    tot = np.var(means, ddof=1)
    return 1.0 - (noise / used) / max(tot, 1e-12)


def poisson_measures(counts, depth, mc, order, min_total=10):
    """
    Count-native share and deviance test on POOLED counts per metacell.

    Under H0 (one rate for every metacell) Y_c ~ Poisson(mu * N_c), so
    Var(Y_c/N_c) = mu/N_c.  poisson_share compares the observed dispersion of the
    rates against exactly that, which is the count analogue of one minus the
    noise fraction.  deviance_p is the corresponding likelihood-ratio test.
    """
    Y, N = [], []
    for c in order:
        sel = mc == c
        if not sel.any():
            continue
        Y.append(counts[sel].sum())
        N.append(depth[sel].sum())
    Y, N = np.asarray(Y, float), np.asarray(N, float)
    if Y.size < 5 or Y.sum() < min_total or np.any(N <= 0):
        return np.nan, np.nan, float(Y.sum())

    mu = Y.sum() / N.sum()                      # pooled rate under H0
    rates = Y / N
    exp_var = mu * np.mean(1.0 / N)             # Poisson variance of the rates
    obs_var = np.var(rates, ddof=1)
    share = 1.0 - exp_var / max(obs_var, 1e-300)

    fit = mu * N
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(Y > 0, Y * np.log(Y / fit), 0.0)
    dev = 2.0 * np.sum(term - (Y - fit))
    p = float(stats.chi2.sf(dev, df=Y.size - 1)) if Y.size > 1 else np.nan
    return float(share), p, float(Y.sum())


def kruskal_p(vals, mc, order, min_cells=2):
    """Distribution-free: do the metacells shift the cell-level distribution?"""
    groups = [vals[mc == c] for c in order if (mc == c).sum() >= min_cells]
    if len(groups) < 3:
        return np.nan
    if all(np.ptp(np.concatenate(groups)) == 0 for _ in (0,)):
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=str(
        REPO / "CausalDiscovery/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--assignments", default=str(
        REPO / "CausalDiscovery/outputs/metacells/joint_v2/cell_metacell_assignments.tsv"))
    ap.add_argument("--data-dir", default=str(REPO / "Data/outputs"))
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--out", default="crossmark_count_signal.tsv")
    a = ap.parse_args()

    assign = pd.read_csv(a.assignments, sep="\t")
    order = complete_metacells(assign)
    print(f"{len(order)} complete metacells")

    root = Path(a.dataset_root)
    mats = sorted(root.glob("*/nearby_peaks/*_nearby_peak_matrix.tsv"))
    want, per_gene = {}, {}
    for m in mats:
        cols = [c for c in pd.read_csv(m, sep="\t", nrows=1).columns
                if c.startswith("chr") and "__" in c]
        per_gene[m.name.split("_nearby")[0]] = cols
        for c in cols:
            reg, mark = c.rsplit("__", 1)
            want.setdefault(mark, set()).add(reg)
    print(f"{len(mats)} genes, {sum(len(v) for v in want.values())} peak-mark variables")

    rows = []
    for mark, regs in sorted(want.items()):
        sub = assign[assign["mark"] == mark]
        if sub.empty:
            continue
        x, feats, bcs = load_mark_matrix(Path(a.data_dir) / mark, mark)
        counts, depth = per_cell_counts(x, feats, sorted(regs))
        pos = {b: i for i, b in enumerate(bcs)}
        idx = sub["cell"].map(pos)
        ok = idx.notna().to_numpy()
        rr = idx[ok].to_numpy(int)
        mc = sub["metacell_id"].to_numpy()[ok]
        dep = depth[rr]
        dep_safe = np.where(dep <= 0, 1.0, dep)

        for reg in sorted(regs):
            cnt = counts[reg][rr]
            # the production per-cell quantity: log1p(count * target/depth)
            vlog = np.log1p(cnt * (a.target_sum / dep_safe))
            g = gaussian_share(vlog, mc, order)
            ps, pdev, tot = poisson_measures(cnt, dep, mc, order)
            kw = kruskal_p(vlog, mc, order)
            det = float((cnt > 0).mean())
            rows.append(dict(variable=f"{reg}__{mark}", mark=mark,
                             gaussian_share=g, poisson_share=ps,
                             deviance_p=pdev, kw_p=kw,
                             total_counts=tot, detection_rate=det))
        print(f"  {mark:9s} {len(regs):4d} regions")

    df = pd.DataFrame(rows).drop_duplicates("variable")
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}  ({len(df)} unique variables)")

    from statsmodels.stats.multitest import multipletests
    for col in ("deviance_p", "kw_p"):
        m = df[col].notna()
        df.loc[m, col + "_fdr"] = multipletests(df.loc[m, col], method="fdr_bh")[1]

    print("\n=== SAME QUESTION, FOUR WAYS ===")
    print(f"  variables                          {len(df)}")
    print(f"  median gaussian_share (current)    {df.gaussian_share.median():+.3f}")
    print(f"  median poisson_share  (count)      {df.poisson_share.median():+.3f}")
    print(f"  gaussian_share > 0.25              {100*(df.gaussian_share>0.25).mean():5.1f}%")
    print(f"  poisson_share  > 0.25              {100*(df.poisson_share>0.25).mean():5.1f}%")
    print(f"  Poisson deviance FDR < 0.05        {100*(df.deviance_p_fdr<0.05).mean():5.1f}%")
    print(f"  Kruskal-Wallis  FDR < 0.05         {100*(df.kw_p_fdr<0.05).mean():5.1f}%")

    print("\n=== DO THEY AGREE? ===")
    m = df.gaussian_share.notna() & df.poisson_share.notna()
    if m.sum() > 10:
        print(f"  spearman(gaussian_share, poisson_share) = "
              f"{stats.spearmanr(df.gaussian_share[m], df.poisson_share[m]).statistic:+.3f}")
    g_ok = df.gaussian_share > 0.25
    d_ok = df.deviance_p_fdr < 0.05
    print(f"  variables the count test rescues (gaussian dead, deviance alive): "
          f"{int((~g_ok & d_ok).sum())}")
    print(f"  variables only the Gaussian view likes: {int((g_ok & ~d_ok).sum())}")
    print(f"  agreed alive: {int((g_ok & d_ok).sum())}   agreed dead: {int((~g_ok & ~d_ok).sum())}")

    print("\n=== IS LOW SIGNAL JUST LOW DEPTH? ===")
    q = pd.qcut(df.total_counts.rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    print(df.groupby(q, observed=True).agg(
        n=("variable", "size"), median_counts=("total_counts", "median"),
        detection=("detection_rate", "median"),
        gaussian=("gaussian_share", "median"), poisson=("poisson_share", "median"),
        pct_dev_sig=("deviance_p_fdr", lambda s: 100*(s < 0.05).mean())).round(3).to_string())


if __name__ == "__main__":
    main()
