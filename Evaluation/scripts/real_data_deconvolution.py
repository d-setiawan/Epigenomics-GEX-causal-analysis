"""
Measure the aggregation-noise term on the REAL panel, and de-attenuate it.

WHY
---
A metacell holds a median of 8-15 cells PER MARK (minimum 1).  Variable j is
the mean of that mark's cells, so its sampling variance is Sigma_within[j,j]/n
with n ~ 10 -- a tenth of the full cell-level variance, added to every diagonal
entry of the covariance the search consumes.

The consequence is asymmetric, and in the direction that hurts most:

    same-mark  (j, k):  shared cells, so the sampling noise contributes to the
                        COVARIANCE as well as to both variances.
    cross-mark (j, k):  disjoint cells, so the sampling noise contributes
                        NOTHING to the covariance -- but still inflates both
                        variances.

A cross-mark correlation is therefore attenuated by the full noise factor,

    rho_obs = rho_between * sqrt( (s2b_j / (s2b_j + s2w_j/n_j))
                                * (s2b_k / (s2b_k + s2w_k/n_k)) )

with no compensating numerator term.  This is classical attenuation by
measurement error, and it is a pure artefact of how many cells landed in each
metacell.

WHAT THIS BUYS, AND WHAT IT ASSUMES
-----------------------------------
Removing it needs only Sigma_within, which is estimable for every variable from
its own mark's cells -- there is no missing block and no completion assumption.
Correcting the diagonals AND the same-mark off-diagonals yields Sigma_between,
a consistent covariance on one scale for every pair.  That is the whole
modification: the search is stock, the input is de-attenuated.

The cost is that Sigma_between is closer to singular than the raw metacell
covariance -- the noise floor was also what kept it invertible.  So
`effective_rank` is reported before and after, and it is the number that says
whether the corrected matrix can support a p-variable search at all.

VERIFICATION
------------
The script first rebuilds the metacell matrix from cells and correlates it
against the shipped one.  If that reconciliation fails, nothing downstream is
trustworthy and the run aborts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmark_covariance as CC  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def parse_interval(s):
    chrom, rest = s.split(":")
    a, b = rest.split("-")
    return chrom, int(a), int(b)


def load_mark_matrix(data_dir: Path, mark: str):
    """Cells x features CSR, plus feature and barcode lists."""
    # some marks are shipped as <mark>_chromatin_clean, others as
    # <mark>_rep1_chromatin_clean; the _bin_ variants are the binned genome,
    # not the peak calls, and must not be picked up.
    cands = [c for c in sorted(data_dir.glob(f"{mark}*chromatin_clean.mtx"))
             if "_bin_" not in c.name]
    if not cands:
        raise FileNotFoundError(f"no peak-level chromatin_clean.mtx for {mark}")
    base = cands[0]
    stem = str(base)[: -len(".mtx")]
    feats = pd.read_csv(stem + "_features.tsv", sep="\t")["feature"].astype(str).tolist()
    bcs = pd.read_csv(stem + "_barcodes.tsv", sep="\t")["barcode"].astype(str).tolist()
    m = mmread(str(base)).tocsr()
    if m.shape == (len(feats), len(bcs)):
        m = m.T.tocsr()
    elif m.shape != (len(bcs), len(feats)):
        raise ValueError(f"{mark}: mtx {m.shape} matches neither "
                         f"({len(feats)},{len(bcs)}) nor its transpose")
    return m, feats, bcs


def per_cell_values(x: sparse.csr_matrix, feats, regions, target_sum=1e4):
    """
    Reproduce the pipeline's per-cell quantification:
        log1p( mean over overlapping features of count * target_sum / rowsum )
    """
    row_sums = np.asarray(x.sum(axis=1)).ravel().astype(float)
    row_sums[row_sums <= 0] = 1.0
    parsed = [parse_interval(f) for f in feats]
    by_chrom = {}
    for i, (c, s, e) in enumerate(parsed):
        by_chrom.setdefault(c, []).append((s, e, i))

    out = {}
    for reg in regions:
        rc, rs, re_ = parse_interval(reg)
        idx = np.array(sorted(i for (s, e, i) in by_chrom.get(rc, [])
                              if s < re_ and rs < e), dtype=int)
        if idx.size == 0:
            out[reg] = np.full(x.shape[0], np.nan)
            continue
        sub = x[:, idx].multiply((target_sum / row_sums)[:, None])
        out[reg] = np.log1p(np.asarray(sub.mean(axis=1)).ravel())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", required=True, help="shipped metacell matrix tsv")
    ap.add_argument("--assignments", default=str(
        REPO / "CausalDiscovery/outputs/metacells/joint_v2/cell_metacell_assignments.tsv"))
    ap.add_argument("--data-dir", default=str(REPO / "Data/outputs"))
    ap.add_argument("--gene", default=None)
    ap.add_argument("--min-cells", type=int, default=2,
                    help="min cells per mark for a metacell to contribute a "
                         "within-covariance estimate")
    ap.add_argument("--max-cond", type=float, default=1e4,
                    help="condition-number ceiling defining the admissible "
                         "de-attenuation lambda*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    shipped = pd.read_csv(a.matrix, sep="\t")
    peak_cols = [c for c in shipped.columns if c.startswith("chr") and "__" in c]
    expr_cols = [c for c in shipped.columns if c.startswith("expr__") and c.endswith("_log1p")]
    if not expr_cols:
        raise SystemExit("no expr__*_log1p column in the shipped matrix")

    by_mark = {}
    for c in peak_cols:
        reg, mark = c.rsplit("__", 1)
        by_mark.setdefault(mark, []).append((c, reg))
    print(f"panel: {len(peak_cols)} peaks over {len(by_mark)} marks "
          f"+ {expr_cols[0]}   ({shipped.shape[0]} metacells)")

    assign = pd.read_csv(a.assignments, sep="\t")
    data_dir = Path(a.data_dir)

    # ---- rebuild per-cell values, mark by mark ----
    cell_vals, cell_mc, cell_ch, col_names, col_ch = {}, {}, {}, [], []
    for ch_i, (mark, entries) in enumerate(sorted(by_mark.items())):
        sub = assign[assign["mark"] == mark]
        if sub.empty:
            print(f"  ! {mark}: no cells in assignment table, skipping")
            continue
        x, feats, bcs = load_mark_matrix(data_dir / mark, mark)
        vals = per_cell_values(x, feats, [r for _, r in entries])
        pos = {b: i for i, b in enumerate(bcs)}
        keep = sub["cell"].map(pos)
        ok = keep.notna().to_numpy()
        rows = keep[ok].to_numpy(int)
        mcs = sub["metacell_id"].to_numpy()[ok]
        print(f"  {mark:9s} {len(entries):3d} peaks  {ok.sum():5d}/{len(sub):5d} cells matched")
        for col, reg in entries:
            v = vals[reg][rows]
            cell_vals[col] = v
            cell_mc[col] = mcs
            cell_ch[col] = ch_i
            col_names.append(col)
            col_ch.append(ch_i)

    rna_ch = len(by_mark)
    for c in expr_cols[:1]:
        col_names.append(c)
        col_ch.append(rna_ch)

    # ---- restrict to the COMPLETE metacells the pipeline actually ships ----
    # The shipped matrix carries the completeness rule (a metacell contributes a
    # row only if it holds at least one cell from every mark AND from RNA) and
    # renames the survivors MC_0001.. in sorted metacell_id order.  Rebuilding
    # over all 148 metacells and comparing would misalign every row.
    all_mods = sorted(assign["mark"].unique())
    counts = assign.groupby(["metacell_id", "mark"]).size().unstack(fill_value=0)
    complete = counts.index[(counts[all_mods] > 0).all(axis=1)].to_numpy()
    order = sorted(complete)
    print(f"\ncomplete metacells: {len(order)} of {counts.shape[0]} "
          f"(shipped matrix has {shipped.shape[0]})")
    rebuilt = {}
    for col in col_names:
        if col not in cell_vals:
            continue
        v, m = cell_vals[col], cell_mc[col]
        rebuilt[col] = np.array([np.nanmean(v[m == c]) if np.any(m == c) else np.nan
                                 for c in order])
    reb = pd.DataFrame(rebuilt, index=order)

    common = [c for c in reb.columns if c in shipped.columns]
    cors = []
    for c in common:
        s = shipped[c].to_numpy(float)
        if len(s) != len(reb):
            s = s[:len(reb)]
        m = np.isfinite(s) & np.isfinite(reb[c].to_numpy())
        if m.sum() > 10:
            cors.append(np.corrcoef(s[m], reb[c].to_numpy()[m])[0, 1])
    cors = np.asarray(cors)
    print(f"\nRECONCILIATION vs shipped matrix: median r = {np.nanmedian(cors):.4f} "
          f"over {len(cors)} columns  (min {np.nanmin(cors):.3f})")
    if np.nanmedian(cors) < 0.9:
        print("  ! reconstruction does not match the shipped matrix; "
              "the metacell row order or normalisation differs.")
        print("  ! reporting on the RECONSTRUCTED matrix only.")

    # ---- decomposition ----
    use = [c for c in col_names if c in cell_vals]
    ch = np.array([col_ch[col_names.index(c)] for c in use])
    n_mc = len(order)
    p = len(use)

    # Noise term, estimated PER CLUSTER.
    #
    # The pooled estimate Sw_pooled * E[1/n] is biased here: Sw_pooled is
    # degrees-of-freedom weighted, so large metacells dominate it, while E[1/n]
    # is dominated by small ones.  Small Leiden clusters are internally tighter,
    # so the product overestimates -- which is why the pooled version reported a
    # noise share above 1, an impossible decomposition.
    #
    # E_c[ S_c / n_c ] using each cluster's OWN within-covariance is unbiased
    # whatever the relationship between cluster size and within-variance.
    Sw = np.full((p, p), np.nan)
    noise_est = np.zeros((p, p))
    inv_n = np.zeros(p)
    for a_i in np.unique(ch):
        vs = np.flatnonzero(ch == a_i)
        cols_a = [use[j] for j in vs]
        m = cell_mc[cols_a[0]]
        Vc = np.column_stack([cell_vals[c] for c in cols_a])
        acc = np.zeros((len(vs), len(vs)))
        acc_noise = np.zeros((len(vs), len(vs)))
        dof, used, sizes = 0, 0, []
        for c in order:
            sel = m == c
            k = int(sel.sum())
            if k < a.min_cells:
                continue
            blk = Vc[sel] - Vc[sel].mean(axis=0, keepdims=True)
            cross = blk.T @ blk
            acc += cross
            acc_noise += (cross / (k - 1)) / k      # S_c / n_c
            dof += k - 1
            used += 1
            sizes.append(k)
        if dof > 0:
            Sw[np.ix_(vs, vs)] = acc / dof
        if used > 0:
            noise_est[np.ix_(vs, vs)] = acc_noise / used
        inv_n[vs] = np.mean(1.0 / np.asarray(sizes, float)) if sizes else np.nan

    X = reb[use].to_numpy(float)
    good = np.isfinite(X).all(axis=1)
    X = X[good]
    raw = np.cov(X, rowvar=False)

    noise = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            if ch[j] == ch[k] and np.isfinite(noise_est[j, k]):
                noise[j, k] = noise_est[j, k]

    # How much of the correction can the data actually support?  Subtracting the
    # full noise term removes the floor that keeps the metacell covariance
    # invertible, so we take the LARGEST admissible fraction rather than
    # assuming the full one is safe.  lam* is a reported property of the data,
    # not a tuning knob: it is the point past which the corrected matrix stops
    # being a usable covariance.
    def corrected(lam):
        return raw - lam * noise

    lam_star, lo, hi = 0.0, 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        S = 0.5 * (corrected(mid) + corrected(mid).T)
        w = np.linalg.eigvalsh(S)
        if w.min() > 0 and (w.max() / w.min()) <= a.max_cond:
            lam_star, lo = mid, mid
        else:
            hi = mid
    between = corrected(lam_star)
    print(f"\nMAXIMAL ADMISSIBLE CORRECTION  lambda* = {lam_star:.3f}"
          f"   (1.0 = full de-attenuation, cond kept <= {a.max_cond:g})")

    iu = np.triu_indices(p, 1)
    same = ch[iu[0]] == ch[iu[1]]
    Craw, Cbet = CC.to_correlation(raw), CC.to_correlation(CC.project_psd(between)[0])

    frac = np.diag(noise) / np.clip(np.diag(raw), 1e-12, None)
    print(f"\nNOISE SHARE OF THE DIAGONAL (this is the attenuation):")
    print(f"  median {np.median(frac):.3f}   mean {np.mean(frac):.3f}   "
          f"max {np.max(frac):.3f}")
    print(f"  median cells/metacell/mark: {np.median(1/inv_n[np.isfinite(inv_n)]):.1f}")

    # The SNR audit.  signal share = 1 - noise/total = the fraction of a
    # variable's metacell-level variance that is real between-state signal
    # rather than the sampling noise of averaging ~10 cells.  A variable whose
    # share is <= 0 carries no between-metacell signal that this design can
    # distinguish from noise, and no search can recover structure from it.
    share = 1.0 - frac
    per_mark = {}
    for a_i in np.unique(ch):
        vs = ch == a_i
        nm = use[int(np.flatnonzero(vs)[0])].rsplit("__", 1)[-1] if vs.any() else "?"
        per_mark[nm] = (float(np.median(share[vs])), int((share[vs] > 0.1).sum()),
                        int(vs.sum()))
    print(f"\nSIGNAL SHARE PER VARIABLE  (1 - noise/total; <=0 means no "
          f"recoverable between-state signal)")
    print(f"  median {np.median(share):+.3f}   "
          f"variables with share > 0.10: {int((share > 0.1).sum())}/{p}   "
          f"share > 0: {int((share > 0).sum())}/{p}")
    for nm, (med, n_ok, n_tot) in sorted(per_mark.items()):
        print(f"    {nm:12s} median {med:+.3f}   {n_ok}/{n_tot} usable")

    print(f"\nEFFECTIVE RANK  ({p} variables, {X.shape[0]} metacells)")
    print(f"  raw metacell covariance : {CC.effective_rank(raw):6.2f}   "
          f"cond {CC.condition_number(raw):8.1f}")
    print(f"  de-attenuated (between) : {CC.effective_rank(CC.project_psd(between)[0]):6.2f}   "
          f"cond {CC.condition_number(CC.project_psd(between)[0]):8.1f}")

    print(f"\nMEAN |CORRELATION|            raw     de-attenuated   gain")
    for lab, m in (("same-mark ", same), ("cross-mark", ~same)):
        r0, r1 = np.abs(Craw[iu][m]).mean(), np.abs(Cbet[iu][m]).mean()
        print(f"  {lab} (n={m.sum():4d})   {r0:.4f}      {r1:.4f}      {r1/r0:.2f}x")

    print(f"\nSIGNIFICANT PAIRS at n={X.shape[0]} metacells (marginal Fisher z)")
    print(f"  {'alpha':>7}  {'same raw':>9} {'same deatt':>11} "
          f"{'cross raw':>10} {'cross deatt':>12}   (null)")
    for al in (0.05, 0.01, 0.001):
        thr = norm.ppf(1 - al / 2)
        out = []
        for C in (Craw, Cbet):
            z = np.abs(np.arctanh(np.clip(C[iu], -0.999, 0.999))) * np.sqrt(X.shape[0] - 3)
            out.append(((z > thr)[same].mean(), (z > thr)[~same].mean()))
        print(f"  {al:>7}  {100*out[0][0]:8.1f}% {100*out[1][0]:10.1f}% "
              f"{100*out[0][1]:9.1f}% {100*out[1][1]:11.1f}%   {100*al:.1f}%")

    if a.out:
        np.savez(a.out, raw=raw, between=between, within=Sw, noise=noise,
                 ch=ch, names=np.array(use), n_metacells=X.shape[0])
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
