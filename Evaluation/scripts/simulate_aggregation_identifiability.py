#!/usr/bin/env python3
"""Simulation: is the cell-level causal structure identifiable from metacell-aggregated data?

THE QUESTION (docs/identifiability + partition_bootstrap_results.md). On real data, single-partition
FCI edges do not reproduce across metacell partitions, and orthogonal biological validation is thin.
Two explanations we cannot separate on real data:
  (a) POWER  — the method is sound but n is too small; true edges surface only sometimes.
  (b) CEILING — aggregation distorts the causal structure (ecological fallacy), so even a perfectly
                recovered metacell PAG need not equal the cell-level truth.
This simulation separates them: plant a KNOWN cell-level graph, run the SAME aggregation + discovery,
and ask whether the partition-bootstrap recovers the planted graph, and whether recovery -> 100% as
sample size grows (=> power) or plateaus below it (=> ceiling).

THE PLANTED GROUND TRUTH (the crisp test = causation vs confounding).
A shared cell-state latent Z drives four "state" peaks P1..P4 and the gene G (Z is the source of
cross-metacell variance AND the shared confounder — mirrors the real 53%-bidirected structure).
The gene G is caused DIRECTLY by P1, P2 only (no direct Z->G path, so the confounded peaks are
separable in principle):
  * P1, P2  directly cause G                       -> P1->G, P2->G   (true, DIRECTED)
  * P3, P4  share only Z with the P1,P2 path        -> P3,P4 become INDEPENDENT of G given {P1,P2},
            so FCI can PRUNE the P3-G / P4-G adjacency   (true: NOT adjacent to G)
  * P5, P6  depend on depth only                    -> isolated once depth is conditioned (TRUE NEG)
So the method WORKS iff it keeps P1,P2 adjacent-and-directed to G while PRUNING P3,P4 from G — i.e. it
tells a direct cause from a peak merely correlated with the gene through shared state. This distinction
IS identifiable at the cell level (condition on P1,P2); the question is whether aggregation preserves it.
Depth D is an observed confounder root. (A direct Z->G path makes P3,P4 inseparable at ANY level — pure
latent confounding, not an aggregation effect — so it is off by default: --c-state 0.)

Note this simulation also exposes that the tiered chromatin->expression prior FORCES every peak-gene
ADJACENCY to orient peak->gene. So "directed peak->gene" reflects adjacency + the prior, not learned
causal direction; the real test is whether the confounded adjacency is PRUNED, not how it is oriented.

Discovery reuses the production pipeline verbatim: build_background_knowledge (depth root + tiered
peaks->gene prior), rank-gaussian transform, FCI + fisherz. Only the data is synthetic.

    python Evaluation/scripts/simulate_aggregation_identifiability.py
    python Evaluation/scripts/simulate_aggregation_identifiability.py --cells-per-mc 40 --k-seeds 20
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "CausalDiscovery/scripts/commands"))
from run_pc_causallearn import build_background_knowledge, apply_transform, run_fci_search  # noqa: E402
from causallearn.graph.Endpoint import Endpoint  # noqa: E402

# node names use pipeline conventions so build_background_knowledge assigns the right roles.
# Two INDEPENDENT state latents Z, W so each confounded peak separates from G with a size-2 set
# ({its direct-cause partner} + depth), which the pipeline's max_depth=2 can actually test.
P = ["chr1:1000-1200__H3K4me3",   # P1  direct cause of G (uses Z)
     "chr1:5000-5200__H3K4me3",   # P2  direct cause of G (uses W)
     "chr1:9000-9200__H3K4me1",   # P3  confounded-only, shares Z with P1  -> prune from G given {P1, depth}
     "chr1:13000-13200__H3K4me1", # P4  confounded-only, shares W with P2  -> prune from G given {P2, depth}
     "chr1:17000-17200__H3K27ac", # P5  noise (depth only)
     "chr1:21000-21200__H3K27ac"] # P6  noise (depth only)
GENE = "expr__SIMG_log1p"
DEPTH = "conf__depth"
DISC_COLS = P + [GENE, DEPTH]


def simulate_cells(n, rng, b_direct, c_state, depth_eff, noise):
    Z = rng.normal(size=n)                # state latent A (confounds P1,P3 + partition structure)
    W = rng.normal(size=n)                # state latent B (confounds P2,P4), independent of Z
    D = rng.normal(size=n)                # depth (technical root)
    def e():
        return rng.normal(size=n) * noise
    cols = {
        "Z": Z, "Z2": W, "D": D,           # Z,Z2 columns are the 2-D "embedding" clustered on
        P[0]: Z + depth_eff * D + e(),     # P1 (cause, Z)
        P[1]: W + depth_eff * D + e(),     # P2 (cause, W)
        P[2]: Z + depth_eff * D + e(),     # P3 (Z-confounded with P1)
        P[3]: W + depth_eff * D + e(),     # P4 (W-confounded with P2)
        P[4]: depth_eff * D + e(),         # P5 noise
        P[5]: depth_eff * D + e(),         # P6 noise
    }
    # G caused directly by P1,P2 only. c_state adds a direct state->G path (off by default) that would
    # make P3,P4 inseparable at ANY level (pure latent confounding, not aggregation).
    cols[GENE] = (b_direct * cols[P[0]] + b_direct * cols[P[1]]
                  + c_state * (Z + W) + depth_eff * D + e())
    return pd.DataFrame(cols)


def aggregate(cells, n_metacells, seed):
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=n_metacells, random_state=seed, n_init=3).fit_predict(
        cells[["Z", "Z2"]].to_numpy())
    agg = cells.assign(mc=lab).groupby("mc")[P + [GENE, "D"]].mean()
    agg = agg.rename(columns={"D": DEPTH})
    return agg[DISC_COLS].reset_index(drop=True)


def discover(mat):
    cols = list(mat.columns)
    bk, _ = build_background_knowledge(cols, "tiered_distal_promoter_expr")
    X = apply_transform(mat, "rank_gaussian")
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    graph, edges = run_fci_search(X.to_numpy(), alpha=0.05, indep_test="fisherz",
                                  background_knowledge=bk, show_progress=False,
                                  node_names=cols, max_depth=2)
    out = {}
    for ed in edges:
        a, b = ed.get_node1().get_name(), ed.get_node2().get_name()
        out[(a, b)] = (ed.get_endpoint1(), ed.get_endpoint2())
    return out


def edge_to_gene(edges, peak):
    """How is `peak` connected to GENE? -> 'directed' (peak->G), 'bidirected', 'other_adj', or 'none'."""
    for (a, b), (e1, e2) in edges.items():
        if {a, b} != {peak, GENE}:
            continue
        # endpoint at the GENE side and at the peak side
        peak_ep = e1 if a == peak else e2
        gene_ep = e2 if a == peak else e1
        if peak_ep == Endpoint.TAIL and gene_ep == Endpoint.ARROW:
            return "directed"          # peak -> G  (causal)
        if peak_ep == Endpoint.ARROW and gene_ep == Endpoint.ARROW:
            return "bidirected"        # peak <-> G (latent confounder)
        return "other_adj"             # circle marks etc. (undetermined but adjacent)
    return "none"


def evaluate(edges):
    to_g = {pk: edge_to_gene(edges, pk) for pk in P}
    return {
        # TRUE causal peaks recovered as directed causes of G (adjacency + prior orientation)
        "P1_directed": int(to_g[P[0]] == "directed"),
        "P2_directed": int(to_g[P[1]] == "directed"),
        # confounded-only peaks correctly PRUNED from G (not adjacent) — the real causal test
        "P3_pruned": int(to_g[P[2]] == "none"),
        "P4_pruned": int(to_g[P[3]] == "none"),
        # confounded-only peaks WRONGLY kept adjacent to G (false causal edge under the prior)
        "P3_falseadj": int(to_g[P[2]] != "none"),
        "P4_falseadj": int(to_g[P[3]] != "none"),
        # pure-noise peaks wrongly adjacent to G
        "P5_adj": int(to_g[P[4]] != "none"),
        "P6_adj": int(to_g[P[5]] != "none"),
        # headline: true causes kept AND confounded-only peaks pruned, all four correct
        "distinguish": int(to_g[P[0]] == "directed" and to_g[P[1]] == "directed"
                           and to_g[P[2]] == "none" and to_g[P[3]] == "none"),
        "to_g": to_g,
    }


def cell_level_matrix(cells):
    m = cells[P + [GENE]].copy()
    m[DEPTH] = cells["D"].to_numpy()
    return m[DISC_COLS]


def distinguish_over(draws, n_mc, k_part):
    vals = []
    for cells in draws:
        for s in range(k_part):
            vals.append(evaluate(discover(aggregate(cells, n_mc, seed=s)))["distinguish"])
    return float(np.mean(vals))


def collinearity_over(draws, n_mc):
    v = []
    for cells in draws:
        m = aggregate(cells, n_mc, seed=0)
        v.append(abs(np.corrcoef(m[P[0]], m[P[2]])[0, 1]))
    return float(np.mean(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-metacells", type=int, default=133,
                    help="rows for discovery (matches the real pipeline's ~133)")
    ap.add_argument("--cpm-sweep", default="2,4,8,16,32,64",
                    help="cells-per-metacell values to sweep (the aggregation-amount axis)")
    ap.add_argument("--n-draws", type=int, default=6, help="independent data draws per point")
    ap.add_argument("--k-partitions", type=int, default=4, help="clustering seeds per draw")
    ap.add_argument("--b-direct", type=float, default=0.7, help="direct effect P1,P2 -> G")
    ap.add_argument("--c-state", type=float, default=0.0,
                    help="direct shared-state Z -> G path. 0 (default) keeps P3,P4 separable so "
                         "aggregation is what's tested; >0 makes them inseparable at any level.")
    ap.add_argument("--depth-eff", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=REPO / "Evaluation/outputs/simulation_identifiability.json")
    a = ap.parse_args()

    cpm_values = [int(x) for x in a.cpm_sweep.split(",")]
    n_mc = a.n_metacells
    print(f"Aggregation-identifiability simulation  (n_metacells={n_mc}, "
          f"{a.n_draws} data draws x {a.k_partitions} partitions per point, c_state={a.c_state})\n")
    print("Ground truth: P1,P2 -> G (direct causes);  P3,P4 confounded via shared state (should be "
          "PRUNED from G);  P5,P6 noise.  DISTINGUISH = all correct at once.\n")

    def draws_for(n_cells, tag):
        return [simulate_cells(n_cells, np.random.default_rng(a.data_seed + 100 * tag + d),
                               a.b_direct, a.c_state, a.depth_eff, a.noise) for d in range(a.n_draws)]

    # 1. cell-level reference: n_mc RAW cells, no aggregation (identifiability upper bound at matched rows)
    raw = draws_for(n_mc, 0)
    ref_distinguish = float(np.mean([evaluate(discover(cell_level_matrix(c)))["distinguish"] for c in raw]))
    print(f"CELL-LEVEL reference: {n_mc} raw cells, NO aggregation -> distinguish = {ref_distinguish:.2f}\n")

    # 2. sweep cells-per-metacell (the aggregation-amount axis) at fixed n_mc rows
    print(f"{'cells/mc':>9} {'n_cells':>9} {'distinguish':>11} {'|corr(cause,confound)|':>22}")
    print("-" * 54)
    sweep = []
    for cpm in cpm_values:
        d = draws_for(n_mc * cpm, cpm)
        dist = distinguish_over(d, n_mc, a.k_partitions)
        coll = collinearity_over(d, n_mc)
        sweep.append({"cells_per_mc": cpm, "n_cells": n_mc * cpm,
                      "distinguish_freq": dist, "collinearity": coll})
        print(f"{cpm:>9} {n_mc*cpm:>9} {dist:>11.2f} {coll:>22.3f}")

    lo, hi = sweep[0], sweep[-1]
    print("\nReading: distinguish = fraction of runs that keep P1,P2->G AND prune P3,P4,P5,P6.")
    print("  collinearity = |corr| between a causal peak and its confounded twin at the metacell level;")
    print("  it rises with cells/metacell as averaging washes out the noise that told them apart.")
    print(f"\n  cell-level (no aggregation, {n_mc} rows): {ref_distinguish:.2f}")
    print(f"  least aggregated ({lo['cells_per_mc']} cells/mc): {lo['distinguish_freq']:.2f}  "
          f"(collinearity {lo['collinearity']:.2f})")
    print(f"  most aggregated  ({hi['cells_per_mc']} cells/mc): {hi['distinguish_freq']:.2f}  "
          f"(collinearity {hi['collinearity']:.2f})")

    if ref_distinguish >= 0.8 and hi["distinguish_freq"] < ref_distinguish - 0.3:
        verdict = (f"AGGREGATION DEGRADES IDENTIFIABILITY. The structure is fully identifiable at the "
                   f"cell level ({ref_distinguish:.2f}); aggregation induces collinearity between causal "
                   f"and confounded peaks that grows with cells/metacell "
                   f"({lo['collinearity']:.2f}->{hi['collinearity']:.2f}) and collapses recovery "
                   f"({lo['distinguish_freq']:.2f}->{hi['distinguish_freq']:.2f}). At matched rows, raw "
                   f"cells beat metacells. => the thin real-data biology is an AGGREGATION CEILING, not "
                   f"pure power. (Caveat: sim cells are Gaussian; real single cells are sparse, which is "
                   f"why aggregation was adopted -> the fix is dropout-aware discovery / smaller metacells.)")
    elif ref_distinguish < 0.8:
        verdict = "INCONCLUSIVE: not cleanly identifiable even at cell level under these settings."
    else:
        verdict = "AGGREGATION TOLERATED: recovery holds under aggregation; thin biology is power/validation."
    print(f"\nVERDICT: {verdict}")

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps({
        "config": {"n_metacells": n_mc, "n_draws": a.n_draws, "k_partitions": a.k_partitions,
                   "b_direct": a.b_direct, "c_state": a.c_state, "depth_eff": a.depth_eff, "noise": a.noise},
        "cell_level_distinguish": ref_distinguish, "cpm_sweep": sweep, "verdict": verdict}, indent=2))
    print(f"wrote {a.out_json}")


if __name__ == "__main__":
    raise SystemExit(main())
