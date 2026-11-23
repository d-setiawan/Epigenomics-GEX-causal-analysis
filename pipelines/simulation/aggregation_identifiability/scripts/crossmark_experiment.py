"""
Does changing the covariance the LV search sees recover cross-mark structure?

Two questions, one harness.

Q1  MECHANISM.  Is the same-mark / cross-mark recall gap caused by cell
    sharing, or is it real structure?  Three aggregation arms at MATCHED
    per-variable cell counts:

        shared    every variable in a channel averaged over all that channel's
                  cells in the cluster.  What the pipeline does.
        thinned   same, but over a random subset of size n/m.  Isolates the
                  effect of having fewer cells.
        disjoint  each variable gets its OWN subset of size n/m, so
                  same-channel pairs now have the disjoint-cell structure that
                  cross-channel pairs always had.

    `disjoint` vs `thinned` is the controlled contrast: identical n per
    variable, differing only in whether cells are shared.  If dir_same
    collapses to dir_cross under `disjoint`, the gap is the shared-cell
    sampling term and not biology.

Q2  REPAIR.  Feeding the search a completed cell-level covariance instead of
    the metacell mean matrix, do cross-mark direct edges become recoverable?
    Estimators from crossmark_covariance:

        raw           covariance of the metacell means.  Current pipeline.
        between-only  deconvolved between-cluster covariance.  Expected to
                      collapse to rank ~ d; included to show that it does.
        zero          Sigma_between + Sigma_within with the missing
                      cross-channel within-block set to 0.  Conservative.
        proportional  missing block filled by c * Sigma_between, c fit and
                      leave-one-channel-out validated on the same-channel
                      entries where both components are observed.

Everything is scored against the CELL-level DAG, split by channel type, on
adjacency AND on orientation -- adjacency alone cannot distinguish "found the
edge" from "found the confounding", which is the entire question here.

Sample size passed to every search is the number of metacells, for every
estimator.  The comparison is then purely about the covariance and never about
manufactured degrees of freedom.

Run:
    PYTHONPATH=<dir containing paircheck/> python crossmark_experiment.py --jar ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmark_covariance as CC  # noqa: E402


# --------------------------------------------------------------------------
# aggregation arms
# --------------------------------------------------------------------------

def aggregate_arm(V, labels, channel_cells, ch, arm, rng):
    """
    Build the metacell matrix under one of the three cell-sharing arms.
    Returns (X, per_variable_cell_counts, info).

    `thinned` and `disjoint` are matched on cells-per-variable by construction:
    both give each variable floor(n_ca / m_a) cells.  They differ only in
    whether two variables in the same channel get the same cells.
    """
    V = np.asarray(V, dtype=float)
    p = V.shape[1]
    ch = np.asarray(ch)
    channel_cells = [np.asarray(c) for c in channel_cells]
    labs = [labels[c] for c in channel_cells]
    n_ch = len(channel_cells)
    vars_in = {a: np.flatnonzero(ch == a) for a in range(n_ch)}

    rows, counts, dropped = [], [], 0
    for c in np.unique(labels):
        members = [channel_cells[a][labs[a] == c] for a in range(n_ch)]
        if any(m.size == 0 for m in members):
            dropped += 1
            continue

        assign = {}
        ok = True
        for a in range(n_ch):
            m, va = members[a], vars_in[a]
            if va.size == 0:
                continue
            if arm == "shared":
                for j in va:
                    assign[j] = m
                continue
            k = max(1, m.size // max(len(va), 1))
            perm = rng.permutation(m)
            if arm == "thinned":
                sub = perm[:k]
                for j in va:
                    assign[j] = sub
            elif arm == "disjoint":
                for t, j in enumerate(va):
                    assign[j] = perm[t * k:(t + 1) * k]
            else:
                raise ValueError(f"unknown arm {arm!r}")
            if any(assign[j].size == 0 for j in va):
                ok = False
        if not ok:
            dropped += 1
            continue

        rows.append([V[assign[j], j].mean() for j in range(p)])
        counts.append([assign[j].size for j in range(p)])

    X = np.asarray(rows)
    cnt = np.asarray(counts, dtype=float)
    return X, cnt, dict(arm=arm, n_kept=X.shape[0], n_dropped=dropped,
                        cells_per_var=float(cnt.mean()) if cnt.size else np.nan)


# --------------------------------------------------------------------------
# searches over a covariance matrix
# --------------------------------------------------------------------------

def _tests(cov, alpha, penalty):
    from edu.cmu.tetrad.search.test import IndTestFisherZ
    from edu.cmu.tetrad.search.score import SemBicScore
    test = IndTestFisherZ(cov, float(alpha))
    score = SemBicScore(cov)
    score.setPenaltyDiscount(float(penalty))
    return test, score


def search_cov(cov, name, alpha, penalty, depth):
    """Run one FCI-family / BOSS-family search against a CovarianceMatrix."""
    test, score = _tests(cov, alpha, penalty)

    if name == "fci":
        from edu.cmu.tetrad.search import Fci
        s = Fci(test)
        s.setDepth(int(depth)); s.setStable(True)
        s.setCompleteRuleSetUsed(True); s.setVerbose(False)
        return s.search()

    if name in ("bfci", "bfci-cp"):
        from edu.cmu.tetrad.search import Bfci
        s = Bfci(test, score)
        s.setDepth(int(depth)); s.setCompleteRuleSetUsed(True)
        s.setNumStarts(1); s.setVerbose(False)
        try:
            s.setGuaranteePag(name == "bfci-cp")
        except AttributeError:
            pass
        return s.search()

    if name == "gfci":
        from edu.cmu.tetrad.search import Gfci
        s = Gfci(test, score)
        s.setDepth(int(depth)); s.setCompleteRuleSetUsed(True); s.setVerbose(False)
        return s.search()

    if name == "lvh":
        # LV-Heuristic: PAG of the BOSS DAG.  Pure scoring, assumes
        # sufficiency, emits zero bidirected marks by construction.
        from edu.cmu.tetrad.search import Boss, PermutationSearch
        from edu.cmu.tetrad.graph import GraphTransforms
        boss = Boss(score)
        boss.setUseBes(False)
        boss.setNumStarts(1)
        ps = PermutationSearch(boss)
        dag = ps.search(False)          # False -> return the DAG, not the CPDAG
        return GraphTransforms.dagToPag(dag, False)

    raise ValueError(f"unknown search {name!r}")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_graph(est_g, A_mask, ch, names, truth_pag_adj=None):
    """
    Recall of the CELL-level direct edges, split by channel type, on both
    adjacency and orientation.

    'oriented' means the estimated edge carries an arrowhead at the child and
    no arrowhead at the parent -- i.e. it is compatible with parent -> child
    and is not bidirected.  Adjacency recall counts an edge found; orientation
    recall counts an edge found AND attributed a direction, which is what a
    causal claim needs and what a bidirected mark denies.
    """
    from paircheck import truth as T  # noqa: WPS433

    est = T.edge_dict(est_g)
    adj = set(est.keys())
    ch_of = {names[j]: int(ch[j]) for j in range(len(names))}
    p = A_mask.shape[0]

    same_t, cross_t, same_o, cross_o = set(), set(), 0, 0
    for i in range(p):
        for j in range(p):
            if not A_mask[i, j]:
                continue
            u, v = names[i], names[j]
            key = (u, v) if u < v else (v, u)
            (same_t if ch[i] == ch[j] else cross_t).add(key)
            if key not in est:
                continue
            eu, ev = est[key]
            end_child, end_parent = (ev, eu) if key[0] == u else (eu, ev)
            if end_child == "ARROW" and end_parent != "ARROW":
                if ch[i] == ch[j]:
                    same_o += 1
                else:
                    cross_o += 1

    rec = lambda hit, tru: (hit / len(tru)) if tru else float("nan")
    bi = {k for k, (a, b) in est.items() if a == "ARROW" and b == "ARROW"}
    bi_cross = [k for k in bi if ch_of[k[0]] != ch_of[k[1]]]

    out = dict(
        n_edges=len(adj), n_bi=len(bi),
        adj_rec_same=rec(len(adj & same_t), same_t),
        adj_rec_cross=rec(len(adj & cross_t), cross_t),
        ori_rec_same=rec(same_o, same_t),
        ori_rec_cross=rec(cross_o, cross_t),
        n_direct_same=len(same_t), n_direct_cross=len(cross_t),
        bi_cross_frac=(len(bi_cross) / len(bi)) if bi else float("nan"),
    )
    if truth_pag_adj is not None:
        out["adj_prec"] = (len(adj & truth_pag_adj) / len(adj)) if adj else float("nan")
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=None)
    ap.add_argument("--paircheck", default=os.environ.get("PAIRCHECK_DIR"),
                    help="directory CONTAINING the paircheck package dir")
    ap.add_argument("--out", default="crossmark_results.csv")
    ap.add_argument("--p", type=int, default=13)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--n-channels", type=int, default=6)
    ap.add_argument("--n-cells", type=int, default=4000, help="cells per channel")
    ap.add_argument("--n-clusters", type=int, default=133)
    ap.add_argument("--avg-degree", type=float, default=2.0)
    ap.add_argument("--p-state-parent", type=float, default=0.3)
    ap.add_argument("--geometry", nargs="+", default=["discrete", "continuum"])
    ap.add_argument("--arms", nargs="+", default=["shared", "thinned", "disjoint"])
    ap.add_argument("--estimators", nargs="+",
                    default=["raw", "between-only", "zero", "proportional"])
    ap.add_argument("--searches", nargs="+", default=["fci", "bfci-cp", "lvh"])
    ap.add_argument("--n-data-seeds", type=int, default=3)
    ap.add_argument("--n-cluster-seeds", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()

    if a.paircheck:
        sys.path.insert(0, a.paircheck)
    from paircheck import searches as S
    from paircheck import simulate as sim
    from paircheck import truth as T
    from paircheck.multichannel import (assign_channels, split_cells_by_channel,
                                        truth_pags_mc)

    S.start_jvm(a.jar, verbose=True)
    names = T.var_names(a.p)
    n_ch_total = a.n_channels + 1
    rows, fits = [], []

    for geom in a.geometry:
        for ds in range(a.n_data_seeds):
            cfg = sim.GenConfig(p=a.p, d=a.d,
                                n_cells=a.n_cells * n_ch_total // 2,
                                avg_degree=a.avg_degree,
                                p_state_parent=a.p_state_parent, geometry=geom)
            cells = sim.draw_cells(cfg, seed=1000 + ds)
            rng0 = np.random.default_rng(7000 + ds)
            ch = assign_channels(a.p, a.n_channels, rng0)
            chan_idx = split_cells_by_channel(cells.V.shape[0], n_ch_total, rng0)
            A_mask, G_mask = sim.structure_masks(cells)
            pags = truth_pags_mc(A_mask, G_mask, ch)
            adj_cell = T.adjacencies(pags["cell"])

            for cs in range(a.n_cluster_seeds):
                labels = sim.cluster_labels(cells, a.n_clusters, seed=2000 + cs,
                                            oracle=False)
                for arm in a.arms:
                    rng = np.random.default_rng(9000 + cs)
                    X, cnt, info = aggregate_arm(cells.V, labels, chan_idx, ch,
                                                 arm, rng)
                    if X.shape[0] < 20:
                        print(f"  ! {geom}/{arm} ds{ds} cs{cs}: only {X.shape[0]} rows")
                        continue

                    comp = CC.decompose(cells.V, labels, chan_idx, ch, X=X)
                    pfit = CC.fit_proportionality(comp, ch)
                    fits.append(dict(geometry=geom, arm=arm, data_seed=ds,
                                     cluster_seed=cs, scale=pfit.scale,
                                     r2_in_sample=pfit.r2_in_sample,
                                     r2_loco=pfit.r2_loco, n_pairs=pfit.n_pairs,
                                     eff_rank_between=CC.effective_rank(comp.between),
                                     eff_rank_raw=CC.effective_rank(comp.raw_aggregate),
                                     cond_between=CC.condition_number(comp.between)))

                    # the mechanism arms only need the pipeline's own estimator
                    ests = a.estimators if arm == "shared" else ["raw"]
                    for est in ests:
                        if est == "proportional" and not np.isfinite(pfit.scale):
                            continue
                        try:
                            Sig, meta = CC.complete_covariance(comp, ch, mode=est,
                                                               fit=pfit)
                            cov = CC.to_tetrad_covariance(Sig, names, X.shape[0])
                        except Exception as e:
                            print(f"  ! cov {est} failed: {e}")
                            continue
                        for sname in a.searches:
                            try:
                                g = search_cov(cov, sname, a.alpha, a.penalty, a.depth)
                            except Exception as e:
                                print(f"  ! {sname}/{est} failed: {e}")
                                continue
                            m = score_graph(g, A_mask, ch, names, adj_cell)
                            rows.append(dict(
                                geometry=geom, arm=arm, estimator=est,
                                search=sname, data_seed=ds, cluster_seed=cs,
                                n_rows=X.shape[0],
                                cells_per_var=info["cells_per_var"],
                                psd_shift=meta["psd_shift"],
                                eff_rank=meta["eff_rank"], cond=meta["cond"],
                                r2_loco=pfit.r2_loco, prop_scale=pfit.scale, **m))
                            print(f"[{geom}/{arm}/{est}/{sname}] ds{ds} cs{cs} "
                                  f"rows={X.shape[0]} "
                                  f"adj_same={m['adj_rec_same']:.3f} "
                                  f"adj_cross={m['adj_rec_cross']:.3f} "
                                  f"ori_cross={m['ori_rec_cross']:.3f} "
                                  f"effrank={meta['eff_rank']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.DataFrame(fits).to_csv(a.out.replace(".csv", "_fits.csv"), index=False)
    print(f"\nwrote {a.out}  ({len(df)} runs)")
    if df.empty:
        return

    print("\n=== Q1  MECHANISM: is the same/cross gap the shared cells? ===")
    print("    compare 'disjoint' with 'thinned' -- matched cells per variable")
    q1 = df[df.estimator == "raw"].groupby(["geometry", "arm", "search"]).agg(
        adj_same=("adj_rec_same", "mean"), adj_cross=("adj_rec_cross", "mean"),
        ori_same=("ori_rec_same", "mean"), ori_cross=("ori_rec_cross", "mean"),
        cells_per_var=("cells_per_var", "mean"))
    print(q1.round(3).to_string())

    print("\n=== Q2  REPAIR: does completing the covariance recover cross-mark? ===")
    q2 = df[df.arm == "shared"].groupby(["geometry", "estimator", "search"]).agg(
        adj_same=("adj_rec_same", "mean"), adj_cross=("adj_rec_cross", "mean"),
        ori_same=("ori_rec_same", "mean"), ori_cross=("ori_rec_cross", "mean"),
        adj_prec=("adj_prec", "mean"), n_bi=("n_bi", "mean"),
        eff_rank=("eff_rank", "mean"))
    print(q2.round(3).to_string())

    print("\n=== FALSIFICATION: does one scale c transfer across marks? ===")
    print("    r2_loco < 0 means proportionality is refuted; the completion")
    print("    is then not licensed and 'zero' is the honest estimator.")
    fd = pd.DataFrame(fits)
    print(fd.groupby(["geometry", "arm"]).agg(
        scale=("scale", "mean"), r2_in=("r2_in_sample", "mean"),
        r2_loco=("r2_loco", "mean"),
        eff_rank_between=("eff_rank_between", "mean"),
        eff_rank_raw=("eff_rank_raw", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
