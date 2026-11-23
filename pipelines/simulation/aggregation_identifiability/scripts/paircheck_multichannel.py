"""
Multi-channel extension of the paircheck harness.

WHY THIS EXISTS
---------------
The base harness has exactly two channels: one assay observes (S, V1..V_{p-1}),
the other observes (S, Vp).  All covariates therefore live in ONE channel and
share cells, so every covariate-covariate pair is effectively `pairing=true` and
no cross-channel covariate pair can occur.

The application it is modelling is not like that.  Six histone marks are six
SEPARATE assays on separate cells, so:

    same-mark  peak<->peak   -> one assay, SHARED cells    ~ pairing 'true'
    cross-mark peak<->peak   -> two assays, DISJOINT cells ~ pairing 'matched'
    peak -> gene             -> two assays, DISJOINT cells ~ pairing 'matched'

70.5% of the real candidate peak pairs are cross-mark.  So the majority of the
element<->element layer - the object of interest - sits in the `matched` regime,
and the base harness has no arm that reaches it.

WHAT THIS ADDS
--------------
Covariates are partitioned across M channels; the outcome gets channel M.  Cells
are split M+1 ways, disjointly.  A cluster contributes a row only if it holds at
least one cell from EVERY channel (the real pipeline's completeness rule, which
is itself an uncorrected selection).  Variable j is averaged over that cluster's
channel-ch(j) cells only.

GROUND TRUTH, generalized
-------------------------
The base harness derives `G_matched` by severing direct covariate->outcome edges,
because those two blocks are averaged over disjoint cells so their residuals are
independent.  The same argument applies to ANY pair of variables in different
channels.  So:

    G_matched_mc = cell DAG with all direct CROSS-CHANNEL edges deleted,
                   same-channel direct edges retained, C still latent.

That is a sharp prediction with a testable asymmetry the two-channel harness
cannot express: same-channel direct structure should survive, cross-channel
direct structure should not.

THE MEASUREMENT THAT MATTERS
----------------------------
If the construction MANUFACTURES bidirected marks (direct edge severed, only the
latent-state path left, which FCI must render `<->`), bidirected edges should
concentrate on CROSS-channel pairs.  `bi_same_frac` here is the simulation
counterpart of the real-data statistic (real: 0.437 of bidirected are same-mark
against a 0.292 baseline, mean +0.047 across 10 partitions).  This arm says
whether that statistic has any power to detect the artifact at all.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from . import searches as S
from . import simulate as sim
from . import truth as T


# --------------------------------------------------------------------------
# channel assignment
# --------------------------------------------------------------------------

def assign_channels(p: int, n_channels: int, rng) -> np.ndarray:
    """
    ch[j] = channel index of variable j.  The outcome (index p-1) always gets
    its own channel, mirroring RNA being a separate assay from every mark.
    Covariates are spread as evenly as possible over channels 0..n_channels-1.
    """
    ch = np.empty(p, dtype=int)
    cov = np.arange(p - 1)
    ch[: p - 1] = cov % n_channels
    rng.shuffle(ch[: p - 1])
    ch[p - 1] = n_channels                      # outcome in its own channel
    return ch


def split_cells_by_channel(n_tot: int, n_channels_total: int, rng) -> list[np.ndarray]:
    """Disjoint cell index sets, one per channel.  No cell in two assays."""
    idx = rng.permutation(n_tot)
    return [np.asarray(sorted(a)) for a in np.array_split(idx, n_channels_total)]


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def aggregate_mc(cells, labels, ch, chan_idx, pairing: str, rng):
    """
    Build the aggregated matrix under multi-channel pairing.

      'true'     every variable averaged over the SAME cells (channel 0's cells).
                 The jointly-observed ceiling; the assays cannot supply this.
      'matched'  variable j averaged over its own channel's cells in the cluster.
      'permuted' as matched, but each non-zero channel's cluster labels are
                 permuted, destroying cross-channel correspondence while leaving
                 every marginal and the whole apparatus intact.
    """
    p = cells.cfg.p
    n_ch = len(chan_idx)

    lab_by_ch = []
    for c_i in range(n_ch):
        lab = labels[chan_idx[c_i]]
        if pairing == "permuted" and c_i > 0:
            lab = rng.permutation(lab)
        lab_by_ch.append(lab)

    rows, kept, n_drop = [], [], 0
    for c in np.unique(labels):
        members = [chan_idx[i][lab_by_ch[i] == c] for i in range(n_ch)]
        if any(m.size == 0 for m in members):     # completeness rule
            n_drop += 1
            continue
        row = np.empty(p)
        for j in range(p):
            src = members[0] if pairing == "true" else members[ch[j]]
            row[j] = cells.V[src, j].mean()
        rows.append(row)
        kept.append(c)

    data = np.asarray(rows)
    info = dict(n_clusters_requested=int(np.unique(labels).size),
                n_clusters_kept=int(data.shape[0]),
                n_dropped_incomplete=int(n_drop),
                pct_dropped=round(100.0 * n_drop / max(np.unique(labels).size, 1), 1))
    return data, info


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def build_truth_dag_mc(A_mask, Gamma_mask, ch, drop_cross_channel=False,
                       isolate_outcome=False):
    """As truth.build_truth_dag, but severs CROSS-CHANNEL direct edges."""
    tg = T._tg()
    p, d = A_mask.shape[0], Gamma_mask.shape[1]
    y = p - 1
    vs = [tg.GraphNode(n) for n in T.var_names(p)]
    cs = []
    for k in range(d):
        c = tg.GraphNode(f"C{k + 1}")
        c.setNodeType(tg.NodeType.LATENT)
        cs.append(c)
    from java.util import ArrayList
    nodes = ArrayList()
    for n in vs + cs:
        nodes.add(n)
    g = tg.EdgeListGraph(nodes)

    for i in range(p):
        for j in range(p):
            if not A_mask[i, j]:
                continue
            if drop_cross_channel and ch[i] != ch[j]:
                continue                       # disjoint cells -> residuals independent
            g.addDirectedEdge(vs[i], vs[j])
    for j in range(p):
        for k in range(d):
            if Gamma_mask[j, k]:
                if isolate_outcome and j == y:
                    continue
                g.addDirectedEdge(cs[k], vs[j])
    return g


def truth_pags_mc(A_mask, Gamma_mask, ch):
    tg = T._tg()
    mk = lambda **kw: tg.GraphTransforms.dagToPag(
        build_truth_dag_mc(A_mask, Gamma_mask, ch, **kw), False)
    return {"cell": mk(),
            "matched": mk(drop_cross_channel=True),
            "null": mk(drop_cross_channel=True, isolate_outcome=True)}


def direct_edges_by_channel(A_mask, ch):
    """(same_channel_edges, cross_channel_edges) as sorted-name tuples."""
    p = A_mask.shape[0]
    names = T.var_names(p)
    same, cross = set(), set()
    for i in range(p):
        for j in range(p):
            if not A_mask[i, j]:
                continue
            u, v = names[i], names[j]
            key = (u, v) if u < v else (v, u)
            (same if ch[i] == ch[j] else cross).add(key)
    return same, cross


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_mc(est_g, A_mask, ch, p):
    """Recall of direct edges and bidirected composition, split by channel type."""
    names = T.var_names(p)
    ch_of = {names[j]: int(ch[j]) for j in range(p)}
    est = T.edge_dict(est_g)
    adj = set(est.keys())
    bi = {k for k, (a, b) in est.items() if a == "ARROW" and b == "ARROW"}

    same_true, cross_true = direct_edges_by_channel(A_mask, ch)
    rec = lambda tru: (len(adj & tru) / len(tru)) if tru else float("nan")

    same_pairs = [k for k in adj if ch_of[k[0]] == ch_of[k[1]]]
    cross_pairs = [k for k in adj if ch_of[k[0]] != ch_of[k[1]]]
    bi_same = [k for k in bi if ch_of[k[0]] == ch_of[k[1]]]
    bi_cross = [k for k in bi if ch_of[k[0]] != ch_of[k[1]]]

    # baseline: what fraction of ALL possible pairs are same-channel
    allp = list(itertools.combinations(names, 2))
    base_same = np.mean([ch_of[a] == ch_of[b] for a, b in allp])

    return dict(
        n_edges=len(adj), n_bi=len(bi),
        direct_same_ch_rec=rec(same_true), direct_cross_ch_rec=rec(cross_true),
        n_direct_same=len(same_true), n_direct_cross=len(cross_true),
        bi_rate_same_ch=(len(bi_same) / len(same_pairs)) if same_pairs else float("nan"),
        bi_rate_cross_ch=(len(bi_cross) / len(cross_pairs)) if cross_pairs else float("nan"),
        bi_same_frac=(len(bi_same) / len(bi)) if bi else float("nan"),
        baseline_same_frac=float(base_same),
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=None)
    ap.add_argument("--out", default="multichannel_results.csv")
    ap.add_argument("--p", type=int, default=13)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--n-channels", type=int, default=6, help="covariate channels (marks)")
    ap.add_argument("--n-cells", type=int, default=4000, help="cells PER CHANNEL")
    ap.add_argument("--n-clusters", type=int, default=133)
    ap.add_argument("--avg-degree", type=float, default=2.0)
    ap.add_argument("--p-state-parent", type=float, default=0.3)
    ap.add_argument("--geometry", nargs="+", default=["discrete", "continuum"])
    ap.add_argument("--pairing", nargs="+", default=["true", "matched", "permuted"])
    ap.add_argument("--searches", nargs="+", default=["fci", "bfci-cp"])
    ap.add_argument("--n-data-seeds", type=int, default=3)
    ap.add_argument("--n-cluster-seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()

    S.start_jvm(a.jar, verbose=True)
    n_ch_total = a.n_channels + 1
    rows = []

    for geom in a.geometry:
        for ds in range(a.n_data_seeds):
            cfg = sim.GenConfig(p=a.p, d=a.d, n_cells=a.n_cells * n_ch_total // 2,
                                avg_degree=a.avg_degree,
                                p_state_parent=a.p_state_parent, geometry=geom)
            cells = sim.draw_cells(cfg, seed=1000 + ds)
            n_tot = cells.V.shape[0]
            rng0 = np.random.default_rng(7000 + ds)
            ch = assign_channels(a.p, a.n_channels, rng0)
            chan_idx = split_cells_by_channel(n_tot, n_ch_total, rng0)
            A_mask, G_mask = sim.structure_masks(cells)
            pags = truth_pags_mc(A_mask, G_mask, ch)

            for cs in range(a.n_cluster_seeds):
                labels = sim.cluster_labels(cells, a.n_clusters, seed=2000 + cs, oracle=False)
                for pairing in a.pairing:
                    rng = np.random.default_rng(9000 + cs)
                    data, info = aggregate_mc(cells, labels, ch, chan_idx, pairing, rng)
                    if data.shape[0] < 20:
                        continue
                    ds_obj = S.to_dataset(data, T.var_names(a.p))
                    for sname in a.searches:
                        fn = S.SEARCHES[sname]
                        try:
                            g = fn(ds_obj, a.alpha, a.penalty, a.depth, True)
                        except Exception as e:
                            print(f"  ! {sname} failed: {e}")
                            continue
                        m = score_mc(g, A_mask, ch, a.p)
                        cmp_matched = T.compare(g, pags["matched"], outcome=f"V{a.p}")
                        rows.append(dict(geometry=geom, pairing=pairing, search=sname,
                                         data_seed=ds, cluster_seed=cs,
                                         n_rows=data.shape[0], **info, **m,
                                         adj_rec_vs_matched=cmp_matched.get("adj_rec"),
                                         adj_prec_vs_matched=cmp_matched.get("adj_prec")))
                        print(f"[{geom}/{pairing}/{sname}] ds{ds} cs{cs} rows={data.shape[0]} "
                              f"drop={info['pct_dropped']}% "
                              f"dir_same={m['direct_same_ch_rec']:.3f} "
                              f"dir_cross={m['direct_cross_ch_rec']:.3f} "
                              f"bi_same_frac={m['bi_same_frac']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} runs)")

    if not df.empty:
        print("\n=== DIRECT-EDGE RECALL BY CHANNEL TYPE (the identifiability question) ===")
        g = df.groupby(["geometry", "pairing"]).agg(
            dir_same=("direct_same_ch_rec", "mean"),
            dir_cross=("direct_cross_ch_rec", "mean"),
            n_rows=("n_rows", "mean"), pct_dropped=("pct_dropped", "mean"))
        print(g.round(3).to_string())
        print("\n=== BIDIRECTED COMPOSITION (the artifact question) ===")
        print("  bi_same_frac = fraction of bidirected edges that are SAME-channel")
        print("  artifact predicts this BELOW baseline; real data shows it ABOVE")
        g2 = df.groupby(["geometry", "pairing"]).agg(
            bi_same_frac=("bi_same_frac", "mean"),
            baseline=("baseline_same_frac", "mean"),
            bi_rate_same=("bi_rate_same_ch", "mean"),
            bi_rate_cross=("bi_rate_cross_ch", "mean"),
            n_bi=("n_bi", "mean"))
        print(g2.round(3).to_string())


if __name__ == "__main__":
    main()
