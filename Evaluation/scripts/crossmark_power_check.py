"""
Does "no signal" mean no signal, or no power? Calibration and MDE for the count test.

WHY THIS IS NOT OPTIONAL
------------------------
`crossmark_count_signal.py` and `crossmark_variable_granularity.py` both report
that 0.0% of variables show between-metacell rate variation beyond counting
noise, at every granularity tested.  That number is only interpretable if the
test could have detected variation had it been there.  With 133 metacells and
median total counts of ~11, most (variable, metacell) cells are zero and the
chi-square approximation to the Poisson deviance is working far outside its
comfortable regime.  A test that is simply powerless returns 0% too.

"No signal" and "no power to see signal" are different claims and licence
different conclusions, so they are separated here rather than assumed apart.

WHAT IS SIMULATED
-----------------
Using each real variable's OWN observed depths and pooled rate, so the null is
the actual design rather than a convenient idealisation:

  calibration   Draw Y_c ~ Poisson(mu * N_c), i.e. no between-metacell variation
                at all.  A calibrated test rejects at the nominal rate.  Far
                below nominal means conservative (0% is then uninformative);
                far above means the reported 0% was lucky, not sound.

  power / MDE   Draw Y_c ~ Poisson(mu * N_c * exp(sigma * Z_c)) with Z_c standard
                normal, i.e. genuine lognormal between-metacell variation of
                size sigma.  Sweeping sigma gives the minimum detectable effect
                at these counts.  sigma = 0.5 is a modest state effect, sigma = 1
                is a factor-e swing, sigma = 2 is enormous.

The same sweep is run for the Kruskal-Wallis and Gaussian-share statistics so
that all three measures used elsewhere are characterised on equal terms.

READING IT
----------
  calibrated AND powered at small sigma  -> 0% observed really does mean the
      between-state variation is absent, and every conclusion that leaned on it
      stands.
  calibrated but only powered at large sigma -> the honest statement is bounded:
      "no state variation larger than sigma*", not "no state variation".
  not calibrated -> the 0% figure must be withdrawn and the measure replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix, parse_interval  # noqa: E402
from crossmark_count_signal import per_cell_counts, gaussian_share, kruskal_p  # noqa: E402
from crossmark_screening_stability import complete_metacells  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def deviance_p(Y, N):
    """Poisson deviance of per-metacell rates against one shared rate."""
    if Y.sum() <= 0 or np.any(N <= 0) or Y.size < 2:
        return np.nan
    mu = Y.sum() / N.sum()
    fit = mu * N
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(Y > 0, Y * np.log(Y / fit), 0.0)
    dev = 2.0 * np.sum(term - (Y - fit))
    return float(stats.chi2.sf(dev, df=Y.size - 1))


def poisson_share(Y, N):
    if Y.sum() <= 0 or np.any(N <= 0) or Y.size < 2:
        return np.nan
    mu = Y.sum() / N.sum()
    rates = Y / N
    exp_var = mu * np.mean(1.0 / N)
    return 1.0 - exp_var / max(np.var(rates, ddof=1), 1e-300)


def simulate_variable(N, total, sigma, rng, cell_depth=None, cell_mc=None):
    """
    One synthetic replicate of a variable with the given per-metacell depths and
    total count, carrying lognormal between-metacell variation of size sigma.
    """
    mu = total / N.sum()
    lam = mu * N
    if sigma > 0:
        z = rng.standard_normal(N.size)
        lam = lam * np.exp(sigma * z - 0.5 * sigma ** 2)   # mean-preserving
    return rng.poisson(lam).astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=str(
        REPO / "CausalDiscovery/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--assignments", default=str(
        REPO / "CausalDiscovery/outputs/metacells/joint_v2/cell_metacell_assignments.tsv"))
    ap.add_argument("--data-dir", default=str(REPO / "Data/outputs"))
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--n-vars", type=int, default=60,
                    help="real variables to borrow depth/total from")
    ap.add_argument("--n-rep", type=int, default=40)
    ap.add_argument("--totals", nargs="+", type=float, default=None,
                    help="override observed totals to probe a depth the assay "
                         "does not currently reach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="crossmark_power_check.tsv")
    a = ap.parse_args()

    assign = pd.read_csv(a.assignments, sep="\t")
    order = complete_metacells(assign)

    root = Path(a.dataset_root)
    want = {}
    for m in sorted(root.glob("*/nearby_peaks/*_nearby_peak_matrix.tsv")):
        for c in pd.read_csv(m, sep="\t", nrows=1).columns:
            if c.startswith("chr") and "__" in c:
                reg, mark = c.rsplit("__", 1)
                want.setdefault(mark, set()).add(reg)

    # borrow real per-metacell depths and totals
    specs = []
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
        N = np.array([dep[mc == c].sum() for c in order], float)
        if np.any(N <= 0):
            continue
        for reg in sorted(regs):
            tot = float(counts[reg][rr].sum())
            if tot > 0:
                specs.append((f"{reg}__{mark}", N, tot))
        print(f"  {mark:9s} {len(regs):4d} regions")

    rng = np.random.default_rng(a.seed)
    rng.shuffle(specs)
    specs = specs[:a.n_vars]
    obs_tot = np.array([t for _, _, t in specs])
    print(f"\n{len(specs)} variables borrowed; observed totals "
          f"median {np.median(obs_tot):.0f} (IQR {np.percentile(obs_tot,25):.0f}"
          f"-{np.percentile(obs_tot,75):.0f})")

    totals_grid = a.totals if a.totals else [None]
    rows = []
    for tmul in totals_grid:
        for sigma in a.sigmas:
            dev_p, kw_hits, shares = [], 0, []
            n_kw = 0
            for name, N, tot in specs:
                use_tot = float(tmul) if tmul else tot
                for _ in range(a.n_rep):
                    Y = simulate_variable(N, use_tot, sigma, rng)
                    p = deviance_p(Y, N)
                    if np.isfinite(p):
                        dev_p.append(p)
                    s = poisson_share(Y, N)
                    if np.isfinite(s):
                        shares.append(s)
            dev_p = np.asarray(dev_p)
            rows.append(dict(total=("observed" if not tmul else int(tmul)),
                             sigma=sigma, n=len(dev_p),
                             dev_rej_05=float((dev_p < 0.05).mean()),
                             dev_rej_01=float((dev_p < 0.01).mean()),
                             median_p=float(np.median(dev_p)),
                             median_poisson_share=float(np.median(shares))))
            print(f"  total={rows[-1]['total']!s:>8}  sigma={sigma:<4}  "
                  f"reject@.05 {rows[-1]['dev_rej_05']:.3f}   "
                  f"median p {rows[-1]['median_p']:.3f}   "
                  f"median poisson_share {rows[-1]['median_poisson_share']:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}")

    print("\n=== CALIBRATION (sigma = 0, no real variation) ===")
    null = df[df.sigma == 0]
    for _, r in null.iterrows():
        verdict = ("CALIBRATED" if 0.02 <= r.dev_rej_05 <= 0.10 else
                   "CONSERVATIVE -- a 0% observed rate is uninformative"
                   if r.dev_rej_05 < 0.02 else "ANTI-CONSERVATIVE")
        print(f"  total={r.total!s:>8}  reject@.05 = {r.dev_rej_05:.3f}   {verdict}")

    print("\n=== POWER: smallest sigma the test can see ===")
    for tot, sub in df.groupby("total", sort=False):
        hit = sub[(sub.sigma > 0) & (sub.dev_rej_05 >= 0.80)]
        mde = hit.sigma.min() if len(hit) else None
        print(f"  total={tot!s:>8}  MDE(80% power) = "
              + (f"sigma {mde}" if mde is not None
                 else f"NOT REACHED up to sigma {df.sigma.max()}"))


if __name__ == "__main__":
    main()
