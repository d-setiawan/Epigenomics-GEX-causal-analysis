"""
A simulator that behaves like single-cell chromatin data, with ground truth.

WHY THE EXISTING SIMULATOR IS NOT ENOUGH
----------------------------------------
The pairing harness generates `V = A V + Gamma C + e` with Gaussian errors: every
variable is continuous, always non-zero, and has real variance.  Every simulation
conclusion recorded here so far was drawn on that dense Gaussian data.  The real
panel is nothing like it, and differs in exactly the dimension the model-class
question is about.  Measured on the CD151 panel:

    detection rate per peak      median 1.26%   (IQR 0.43-2.11%)
    total counts per peak        median 26
    fraction of non-zero counts >1   0.015
    per-cell depth (peak-level)  median 7-242 by mark, log-sd 0.66
    cells per metacell per mark  14

The third line is the one that matters most: **98.5% of non-zero counts are
exactly 1**.  This is not Poisson-with-magnitude, it is close to Bernoulli
detection.  A simulator that omits that is not testing the regime any conclusion
about transforms or likelihoods needs to hold in.

THE GENERATIVE MODEL
--------------------
The causal structure stays where the biology puts it -- on latent chromatin
ACTIVITY -- and counting is a lossy observation on top:

    C                latent cell state (discrete or continuum)
    V = A V + Gamma C + e            cell-level activity; A is the ground-truth DAG
    lambda_ij = depth_i * exp(beta * V_ij + alpha_j)
    Y_ij ~ Poisson(lambda_ij)        observed counts

`alpha_j` is calibrated per peak by bisection so the realised detection rate hits
the real target, and `depth_i` is lognormal with the measured log-sd.  So the
observation layer is pinned to the real assay while `beta` (how strongly activity
modulates the rate) and `depth_scale` remain free to sweep.

WHAT THIS BUYS OVER THE REAL DATA
---------------------------------
Ground truth.  On real data a transform or test can only be compared on a proxy
(PCHi-C hub enrichment); here the true edge set is known, so recovery is measured
directly and split by channel type -- and the depth sweep converts "we need more
counts" from an assertion into a number that is validated end to end rather than
extrapolated from a marginal power calculation.

`--realism-only` runs the calibration report alone: a side-by-side of simulated
against real summary statistics, which is the check that the simulator earns the
right to be used for anything else.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

REPO = Path(__file__).resolve().parents[4]

# Measured on the CD151 panel; see the module docstring.
REAL = dict(detection_median=0.0126, detection_iqr=(0.0043, 0.0211),
            total_counts_median=26.0, frac_nonzero_gt1=0.015,
            depth_log_sd=0.66, cells_per_metacell=14)


# --------------------------------------------------------------------------
# observation layer
# --------------------------------------------------------------------------

def draw_depths(n, log_sd, median_depth, rng):
    """Per-cell library size: lognormal, matched to the measured spread."""
    return median_depth * np.exp(rng.normal(0.0, log_sd, size=n))


def calibrate_alpha(v, depth, beta, target_det, lo=-20.0, hi=10.0, iters=60):
    """
    Per-peak baseline giving the target detection rate.

    Detection under Poisson is 1 - exp(-lambda), monotone in alpha, so bisection
    is exact and needs no fitting.  Calibrating per peak rather than globally
    keeps the detection rate on target even though beta shifts the mean of the
    linear predictor.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        det = np.mean(1.0 - np.exp(-depth * np.exp(beta * v + mid)))
        if det < target_det:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def peak_detection_targets(p, rng, median=REAL["detection_median"],
                           log_sd=1.15):
    """
    A detection rate per peak, not one shared rate.

    Real peaks differ enormously in baseline accessibility -- the measured IQR is
    0.43%-2.11% around a 1.26% median, a five-fold spread.  Pinning every
    simulated peak to the median produces a panel far more homogeneous than the
    real one, which would flatter any method that benefits from uniform
    variables.
    """
    return np.clip(median * np.exp(rng.normal(0.0, log_sd, size=p)), 1e-5, 0.6)


def observe_counts(V, rng, beta=1.0, target_det=REAL["detection_median"],
                   depth_log_sd=REAL["depth_log_sd"], median_depth=1.0,
                   depth_scale=1.0, obs="bernoulli", det_log_sd=1.15):
    """
    Turn cell-level activity into observed counts.

    obs='bernoulli' emits 1{Poisson > 0}: a cell either carries a fragment at
    this peak or does not.  The real data is 98.5% ones among its non-zeros, so
    this is the closer model; Poisson with lognormal depth generates roughly 4x
    too many multi-counts.  obs='poisson' keeps the magnitudes for comparison.

    depth_scale multiplies every cell's depth and is the knob answering "how much
    deeper would the assay have to be".  Detection targets are calibrated at
    depth_scale=1, so raising it raises the realised detection rate exactly as
    deeper sequencing would, rather than being re-absorbed by the calibration.
    """
    n, p = V.shape
    base_depth = draw_depths(n, depth_log_sd, median_depth, rng)
    targets = peak_detection_targets(p, rng, target_det, det_log_sd)
    Y = np.empty((n, p))
    for j in range(p):
        a = calibrate_alpha(V[:, j], base_depth, beta, targets[j])
        lam = base_depth * depth_scale * np.exp(beta * V[:, j] + a)
        draw = rng.poisson(lam)
        Y[:, j] = (draw > 0).astype(float) if obs == "bernoulli" else draw
    return Y, base_depth * depth_scale


def realism_report(Y, depth, label=""):
    det = (Y > 0).mean(axis=0)
    tot = Y.sum(axis=0)
    nz = Y[Y > 0]
    got = dict(detection_median=float(np.median(det)),
               detection_q1=float(np.percentile(det, 25)),
               detection_q3=float(np.percentile(det, 75)),
               total_counts_median=float(np.median(tot)),
               frac_nonzero_gt1=float(np.mean(nz > 1)) if nz.size else np.nan,
               depth_log_sd=float(np.std(np.log(depth[depth > 0]))))
    if label:
        print(f"\n=== REALISM CHECK {label} ===")
        print(f"  {'statistic':<26} {'simulated':>12} {'real':>12}")
        print(f"  {'detection rate (median)':<26} {got['detection_median']:>12.4f} "
              f"{REAL['detection_median']:>12.4f}")
        print(f"  {'detection IQR':<26} "
              f"{got['detection_q1']:.4f}-{got['detection_q3']:<6.4f} "
              f"{REAL['detection_iqr'][0]:.4f}-{REAL['detection_iqr'][1]:.4f}")
        print(f"  {'frac nonzero counts >1':<26} {got['frac_nonzero_gt1']:>12.3f} "
              f"{REAL['frac_nonzero_gt1']:>12.3f}")
        print(f"  {'depth log-sd':<26} {got['depth_log_sd']:>12.2f} "
              f"{REAL['depth_log_sd']:>12.2f}")
    return got


# --------------------------------------------------------------------------
# aggregation (same transforms as the real-data arm, so results are comparable)
# --------------------------------------------------------------------------

def aggregate_counts(Y, depth, labels, ch, chan_idx, transform, target_sum=1e4):
    """Metacell x variable matrix, using each variable's own channel's cells."""
    p = Y.shape[1]
    lab_by_ch = [labels[ci] for ci in chan_idx]
    rows, kept = [], []
    for c in np.unique(labels):
        members = [chan_idx[i][lab_by_ch[i] == c] for i in range(len(chan_idx))]
        if any(m.size == 0 for m in members):
            continue
        row = np.empty(p)
        for j in range(p):
            m = members[ch[j]]
            d = depth[m]
            d_safe = np.where(d <= 0, 1.0, d)
            if transform == "mean_log1p":
                row[j] = np.log1p(Y[m, j] * (target_sum / d_safe)).mean()
            elif transform == "pseudobulk":
                tot = d.sum()
                row[j] = np.log1p(Y[m, j].sum() * target_sum / (tot if tot > 0 else 1.0))
            elif transform == "detection":
                row[j] = float((Y[m, j] > 0).mean())
            else:
                raise ValueError(transform)
        rows.append(row)
        kept.append(c)
    return np.asarray(rows)


def rank_gaussian(X):
    n = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / n)
                            for j in range(X.shape[1])])


def run_fci(X, names, alpha, depth_lim, test):
    import jpype
    import jpype.imports  # noqa: F401
    import edu.cmu.tetrad.data as td
    from edu.cmu.tetrad.search import Fci
    from java.util import ArrayList

    nodes = ArrayList()
    for nm in names:
        nodes.add(td.ContinuousVariable(str(nm)))
    box = td.DoubleDataBox(jpype.JArray(jpype.JDouble, 2)(
        np.asarray(X, float).tolist()))
    ds = td.BoxDataSet(box, nodes)
    if test == "fisherz":
        from edu.cmu.tetrad.search.test import IndTestFisherZ
        t = IndTestFisherZ(ds, float(alpha))
    elif test == "bf_lrt":
        from edu.cmu.tetrad.search.test import IndTestBasisFunctionLrt
        t = IndTestBasisFunctionLrt(ds, 3, 0.0)
        t.setAlpha(float(alpha))
    elif test == "dg_lrt":
        from edu.cmu.tetrad.search.test import IndTestDegenerateGaussianLrt
        t = IndTestDegenerateGaussianLrt(ds)
        t.setAlpha(float(alpha))
    else:
        raise ValueError(test)
    s = Fci(t)
    s.setDepth(int(depth_lim)); s.setStable(True)
    s.setCompleteRuleSetUsed(True); s.setVerbose(False)
    g = s.search()
    out = set()
    for e in g.getEdges():
        u, v = str(e.getNode1().getName()), str(e.getNode2().getName())
        out.add((u, v) if u < v else (v, u))
    return out


def stability_vs_truth(cells, sim, Y, dep, ch, chan_idx, A_mask, names,
                       n_partitions, n_clusters, transform, test, alpha,
                       search_depth, min_k, rng, Gamma_mask=None):
    """
    Is partition stability actually a good edge selector?

    The biology claim rests on it: §8 keeps edges recovered in >= min_k of 11
    metacell partitions, and §11 shows those are enriched for PCHi-C hubs.  Hub
    enrichment is a proxy, and with ground truth available the question can be
    asked directly.  But it must be asked against the RIGHT target.

    Two targets, because they are not the same claim:

      direct         A[i,j] != 0 -- a cell-level direct causal edge.  Matched
                     pairing structurally severs cross-CHANNEL direct edges
                     (§5a), so for cross-mark pairs this target is unreachable by
                     construction and every cross-mark edge found is a false
                     positive.  Precision against it is therefore a statement
                     about same-mark structure only.

      shared_latent  exists k with Gamma[i,k] != 0 and Gamma[j,k] != 0 -- the two
                     elements are driven by a common latent state dimension.
                     THIS is the simulation analogue of a shared PCHi-C promoter
                     hub, which is what §11 actually validated: two peaks looping
                     to a common bait share a domain, i.e. a common driver, not a
                     direct link between them.  It is also what MISSION §2 names
                     as the object of interest.

    Scoring §11's claim against `direct` would be a category error -- and running
    it that way first is how the distinction surfaced.
    """
    p = A_mask.shape[0]
    direct = set()
    for i in range(p):
        for j in range(p):
            if A_mask[i, j]:
                u, v = names[i], names[j]
                direct.add((u, v) if u < v else (v, u))

    shared = set()
    if Gamma_mask is not None:
        for i in range(p):
            for j in range(i + 1, p):
                if np.any(Gamma_mask[i] & Gamma_mask[j]):
                    u, v = names[i], names[j]
                    shared.add((u, v) if u < v else (v, u))
    truth = direct

    counts = {}
    n_ok = 0
    for r in range(n_partitions):
        labels = sim.cluster_labels(cells, n_clusters, seed=50_000 + r, oracle=False)
        X = aggregate_counts(Y, dep, labels, ch, chan_idx, transform)
        if X.shape[0] < 30:
            continue
        live = X.std(axis=0) > 1e-9
        if live.sum() < 5:
            continue
        idx = np.flatnonzero(live)
        try:
            est = run_fci(rank_gaussian(X[:, idx]), [names[j] for j in idx],
                          alpha, search_depth, test)
        except Exception:
            continue
        for e in est:
            counts[e] = counts.get(e, 0) + 1
        n_ok += 1

    if n_ok < 5:
        return None
    stable = {e for e, c in counts.items() if c >= min_k}
    unstable = {e for e, c in counts.items() if c < min_k}
    cross = lambda e: ch[names.index(e[0])] != ch[names.index(e[1])]
    sc, uc = {e for e in stable if cross(e)}, {e for e in unstable if cross(e)}
    pr = lambda s, tru: (len(s & tru) / len(s)) if s else float("nan")

    return dict(
        n_partitions=n_ok, n_seen=len(counts),
        n_stable=len(stable), n_unstable=len(unstable), n_stable_cross=len(sc),
        # target 1: cell-level direct edge (unreachable for cross-mark by design)
        prec_stable=pr(stable, direct), prec_unstable=pr(unstable, direct),
        prec_all=pr(set(counts), direct),
        prec_stable_cross=pr(sc, direct), prec_unstable_cross=pr(uc, direct),
        # target 2: shared latent driver -- the analogue of a shared PCHi-C hub
        sl_stable=pr(stable, shared), sl_unstable=pr(unstable, shared),
        sl_all=pr(set(counts), shared),
        sl_stable_cross=pr(sc, shared), sl_unstable_cross=pr(uc, shared),
        sl_base=(len(shared) / (p * (p - 1) / 2)) if p > 1 else float("nan"))


def score(est, A_mask, ch, names):
    """Recall and precision of true direct edges, split by channel type."""
    same_t, cross_t = set(), set()
    p = A_mask.shape[0]
    for i in range(p):
        for j in range(p):
            if not A_mask[i, j]:
                continue
            u, v = names[i], names[j]
            k = (u, v) if u < v else (v, u)
            (same_t if ch[i] == ch[j] else cross_t).add(k)
    rec = lambda tru: (len(est & tru) / len(tru)) if tru else float("nan")
    tru_all = same_t | cross_t
    return dict(n_edges=len(est),
                rec_same=rec(same_t), rec_cross=rec(cross_t),
                prec=(len(est & tru_all) / len(est)) if est else float("nan"),
                n_true_same=len(same_t), n_true_cross=len(cross_t))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paircheck", default=os.environ.get("PAIRCHECK_DIR"))
    ap.add_argument("--jar", default=str(REPO.parent.parent /
                    "py-tetrad/pytetrad/resources/tetrad-current.jar"))
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--n-clusters", type=int, default=133)
    ap.add_argument("--cells-per-metacell", type=int, default=REAL["cells_per_metacell"])
    ap.add_argument("--avg-degree", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--depth-scales", nargs="+", type=float, default=[1, 10, 100, 1000])
    ap.add_argument("--median-depth", type=float, default=1.0)
    ap.add_argument("--transforms", nargs="+",
                    default=["mean_log1p", "pseudobulk", "detection"])
    ap.add_argument("--tests", nargs="+", default=["fisherz", "bf_lrt", "dg_lrt"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--search-depth", type=int, default=2)
    ap.add_argument("--geometry", default="discrete")
    ap.add_argument("--obs", default="bernoulli", choices=["bernoulli", "poisson"])
    ap.add_argument("--det-log-sd", type=float, default=1.15,
                    help="spread of per-peak detection rates; 1.15 reproduces "
                         "the measured 0.43%%-2.11%% IQR")
    ap.add_argument("--realism-only", action="store_true")
    ap.add_argument("--stability", action="store_true",
                    help="validate the partition-stability selection rule "
                         "against ground truth instead of sweeping estimators")
    ap.add_argument("--n-partitions", type=int, default=11)
    ap.add_argument("--min-k", type=int, default=6)
    ap.add_argument("--out", default="crossmark_count_sim.tsv")
    a = ap.parse_args()

    if a.paircheck:
        sys.path.insert(0, a.paircheck)
    from paircheck import simulate as sim
    from paircheck import searches as S
    from paircheck.multichannel import assign_channels, split_cells_by_channel

    if not a.realism_only:
        S.start_jvm(a.jar, verbose=True)

    n_ch_total = a.n_channels + 1
    # enough cells that every channel can supply `cells_per_metacell` per cluster
    n_cells_total = a.n_clusters * a.cells_per_metacell * n_ch_total
    names = [f"V{i+1}" for i in range(a.p)]
    rows = []

    for seed in range(a.n_seeds):
        cfg = sim.GenConfig(p=a.p, d=a.d, n_cells=n_cells_total // 2,
                            avg_degree=a.avg_degree, p_state_parent=0.3,
                            geometry=a.geometry)
        cells = sim.draw_cells(cfg, seed=1000 + seed)
        A_mask, G_mask = sim.structure_masks(cells)
        rng = np.random.default_rng(7000 + seed)
        ch = assign_channels(a.p, a.n_channels, rng)
        chan_idx = split_cells_by_channel(cells.V.shape[0], n_ch_total, rng)
        labels = sim.cluster_labels(cells, a.n_clusters, seed=2000 + seed, oracle=False)

        for ds in a.depth_scales:
            Y, dep = observe_counts(cells.V, rng, beta=a.beta,
                                    median_depth=a.median_depth, depth_scale=ds,
                                    obs=a.obs, det_log_sd=a.det_log_sd)
            got = realism_report(Y, dep,
                                 label=f"(seed {seed}, depth x{ds:g})" if seed == 0 else "")
            if a.realism_only:
                continue

            if a.stability:
                for tf in a.transforms:
                    for test in a.tests:
                        st = stability_vs_truth(
                            cells, sim, Y, dep, ch, chan_idx, A_mask, names,
                            a.n_partitions, a.n_clusters, tf, test, a.alpha,
                            a.search_depth, a.min_k, rng, Gamma_mask=G_mask)
                        if st is None:
                            continue
                        st.update(seed=seed, depth_scale=ds, transform=tf, test=test)
                        rows.append(st)
                        print(f"  [x{ds:<5g} {tf:<11s} {test:<8s}] "
                              f"seen={st['n_seen']:3d} stable={st['n_stable']:3d} "
                              f"(cross {st['n_stable_cross']:2d})  "
                              f"DIRECT s={st['prec_stable']:.3f}/u={st['prec_unstable']:.3f}  "
                              f"SHARED-LATENT s={st['sl_stable']:.3f}/"
                              f"u={st['sl_unstable']:.3f} "
                              f"cross s={st['sl_stable_cross']:.3f}/"
                              f"u={st['sl_unstable_cross']:.3f} "
                              f"base={st['sl_base']:.3f}")
                continue

            for tf in a.transforms:
                X = aggregate_counts(Y, dep, labels, ch, chan_idx, tf)
                if X.shape[0] < 30:
                    continue
                live = X.std(axis=0) > 1e-9
                if live.sum() < 5:
                    print(f"  ! depth x{ds:g}/{tf}: only {live.sum()} live variables")
                    continue
                idx = np.flatnonzero(live)
                Xr = rank_gaussian(X[:, idx])
                nm = [names[j] for j in idx]
                for test in a.tests:
                    try:
                        est = run_fci(Xr, nm, a.alpha, a.search_depth, test)
                    except Exception as e:
                        print(f"  ! {tf}/{test}: {e}")
                        continue
                    m = score(est, A_mask, ch, names)
                    rows.append(dict(seed=seed, depth_scale=ds, transform=tf,
                                     test=test, n_rows=X.shape[0],
                                     n_live=int(live.sum()),
                                     detection=got["detection_median"],
                                     frac_gt1=got["frac_nonzero_gt1"], **m))
                    print(f"  [x{ds:<5g} {tf:<11s} {test:<8s}] rows={X.shape[0]:3d} "
                          f"live={live.sum():2d} det={got['detection_median']:.4f} "
                          f"rec_same={m['rec_same']:.3f} rec_cross={m['rec_cross']:.3f} "
                          f"prec={m['prec']:.3f}")

    if a.realism_only:
        return
    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}  ({len(df)} runs)")
    if df.empty:
        return

    if a.stability:
        print("\n=== IS PARTITION STABILITY A GOOD EDGE SELECTOR? (vs ground truth) ===")
        print("  the decisive column is prec_stable vs prec_unstable: edges that")
        print("  survived re-partitioning against edges the SAME search found and")
        print("  dropped.  That is the simulation analogue of §11's contrast.")
        g = df.groupby(["depth_scale"]).agg(
            seen=("n_seen", "mean"), stable=("n_stable", "mean"),
            stable_cross=("n_stable_cross", "mean"),
            dir_stable=("prec_stable", "mean"), dir_unstable=("prec_unstable", "mean"),
            sl_stable=("sl_stable", "mean"), sl_unstable=("sl_unstable", "mean"),
            sl_stable_cross=("sl_stable_cross", "mean"),
            sl_unstable_cross=("sl_unstable_cross", "mean"),
            sl_base=("sl_base", "mean"))
        print(g.round(3).to_string())
        print("\n  TARGET 1 -- cell-level DIRECT edge (unreachable for cross-mark by design):")
        print(f"    stable {df.prec_stable.mean():.3f}   unstable {df.prec_unstable.mean():.3f}"
              f"   lift {df.prec_stable.mean()/max(df.prec_unstable.mean(),1e-9):.2f}x")
        print("  TARGET 2 -- SHARED LATENT DRIVER (the analogue of a shared PCHi-C hub):")
        print(f"    stable {df.sl_stable.mean():.3f}   unstable {df.sl_unstable.mean():.3f}"
              f"   base rate {df.sl_base.mean():.3f}"
              f"   lift {df.sl_stable.mean()/max(df.sl_unstable.mean(),1e-9):.2f}x")
        print(f"    cross-mark only: stable {df.sl_stable_cross.mean():.3f}"
              f"   unstable {df.sl_unstable_cross.mean():.3f}")
        return

    print("\n=== DOES MODEL CLASS MATTER AT REALISTIC SPARSITY? (depth x1) ===")
    d1 = df[df.depth_scale == df.depth_scale.min()]
    if not d1.empty:
        print(d1.groupby(["transform", "test"]).agg(
            rec_same=("rec_same", "mean"), rec_cross=("rec_cross", "mean"),
            prec=("prec", "mean"), n_edges=("n_edges", "mean"),
            live=("n_live", "mean")).round(3).to_string())

    print("\n=== HOW MUCH DEEPER WOULD THE ASSAY HAVE TO BE? ===")
    print(df.groupby("depth_scale").agg(
        detection=("detection", "mean"), live_vars=("n_live", "mean"),
        rec_same=("rec_same", "mean"), rec_cross=("rec_cross", "mean"),
        prec=("prec", "mean")).round(3).to_string())

    print("\n=== BEST TEST AT EACH DEPTH ===")
    for ds, sub in df.groupby("depth_scale"):
        g = sub.groupby("test").rec_cross.mean().sort_values(ascending=False)
        best = g.index[0]
        print(f"  depth x{ds:<6g} best={best:<9s} " +
              "  ".join(f"{k}={v:.3f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
