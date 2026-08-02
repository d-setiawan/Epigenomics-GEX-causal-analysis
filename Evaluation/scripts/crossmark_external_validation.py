"""
Do the partition-stable cross-mark edges carry a biological signature?

Partition stability (crossmark_deattenuated_stability.py) is INTERNAL
reproducibility: an edge that survives re-drawing the metacell boundaries.  The
permutation null shows those edges are not manufactured by the aggregation
apparatus.  Neither says the edges are biology.

This asks the biological question, reusing the repo's existing element<->element
machinery so the null is the one already agreed on:

    signature       PCHi-C shared hub -- both peaks loop to a common promoter
                    bait, i.e. they sit in one physical domain.  Independent of
                    discovery (PCHi-C never enters the search) and of the
                    peak->gene validation (which is target-gene aware).
    null            distance-matched permutation.  Genomic distance is THE
                    confound for element pairs, so the positive/negative label is
                    permuted WITHIN distance bins.  Optionally also degree-matched.

The positive class here is *partition-stable cross-mark edges* rather than the
`<->` mark class the validator normally scores, because the mark class is not
what survived the analysis -- orientation was never partition-stable at any
resolution, adjacency was.

Reported against two references:
  vs non_adjacent   the standard contrast: stable edges vs pairs with no edge
  vs all_adjacent   the sharper one: stable edges vs edges that were FOUND but
                    did NOT survive re-partitioning.  If stability is selecting
                    biology rather than just strong correlations, the stable set
                    should beat the unstable set, not merely the non-edges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peak_peak_validator import (PchicHub, build_pairs, gene_peaks,  # noqa: E402
                                 matched_permutation_test)
from real_data_deconvolution import load_mark_matrix, per_cell_values  # noqa: E402
from crossmark_screening_stability import complete_metacells, build_matrix  # noqa: E402
from crossmark_deattenuated_stability import rank_gaussian, fci_on_cov  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def stable_edges_for_gene(gene, cols, cell_vals, cell_bc, assigns, orders,
                          parts, alpha, depth, min_k, min_rows=30):
    """Adjacencies recovered in >= min_k partitions, and everything ever seen."""
    cnt = {}
    n_ok = 0
    for n, _ in parts:
        X = build_matrix(cols, cell_vals, cell_bc, assigns[n], orders[n])
        good = np.isfinite(X).all(axis=1)
        if good.sum() < min_rows:
            continue
        live = X[good].std(axis=0) > 1e-9
        if live.sum() < 5:
            continue
        sub = np.flatnonzero(live)
        adj, _ = fci_on_cov(np.cov(rank_gaussian(X[good][:, sub]), rowvar=False),
                            [cols[j] for j in sub], int(good.sum()), alpha, depth)
        for e in adj:
            cnt[e] = cnt.get(e, 0) + 1
        n_ok += 1
    stable = {e for e, c in cnt.items() if c >= min_k}
    return stable, set(cnt), n_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=str(REPO.parent.parent /
                    "py-tetrad/pytetrad/resources/tetrad-current.jar"))
    ap.add_argument("--dataset-root", default=str(
        REPO / "CausalDiscovery/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--pipeline-root", default=str(REPO / "CausalDiscovery/outputs/pipeline"))
    ap.add_argument("--metacell-root", default=str(REPO / "CausalDiscovery/outputs/metacells"))
    ap.add_argument("--data-dir", default=str(REPO / "Data/outputs"))
    ap.add_argument("--pchic-tsv", default=str(
        REPO / "Data/external/pchic/pchic_mon_gene_links.tsv"))
    ap.add_argument("--genes", nargs="+",
                    default=["CD14", "CD151", "CD52", "IL4R", "PTPRJ", "LGALS2"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-k", type=int, default=6)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--degree-bins", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="crossmark_external_validation.tsv")
    ap.add_argument("--edges-out", default=None,
                    help="write the partition-stable cross-mark edge list, "
                         "annotated with distance, PAG degree and PCHi-C hub "
                         "status. This is the actual finding; the p-values only "
                         "say the list is enriched.")
    a = ap.parse_args()

    import jpype
    import jpype.imports  # noqa: F401
    if not jpype.isJVMStarted():
        jpype.startJVM("-Xmx4g", classpath=[a.jar])

    pchic = Path(a.pchic_tsv)
    if not pchic.is_file():
        alt = REPO / "Data/raw/external/pchic/pchic_mon_gene_links.tsv"
        if alt.is_file():
            pchic = alt
        else:
            raise SystemExit(f"PCHi-C table not found at {a.pchic_tsv} or {alt}")
    hub = PchicHub(pchic)
    print(f"PCHi-C hub index from {pchic.name}")

    parts = [("main", Path(a.metacell_root) / "joint_v2")]
    parts += [(f"pb{i}", Path(a.metacell_root) / f"joint_v2_pb{i}") for i in range(10)]
    parts = [(n, p) for n, p in parts if (p / "cell_metacell_assignments.tsv").is_file()]
    assigns = {n: pd.read_csv(p / "cell_metacell_assignments.tsv", sep="\t") for n, p in parts}
    orders = {n: complete_metacells(df) for n, df in assigns.items()}

    root, pipe = Path(a.dataset_root), Path(a.pipeline_root)
    genes = [g for g in a.genes if (pipe / g / "edges_consensus.tsv").is_file()]
    print(f"{len(genes)} genes with a pipeline consensus: {', '.join(genes)}")

    stable_all, seen_all = set(), set()
    for gene in genes:
        m = list(root.glob(f"{gene.lower()}/nearby_peaks/*_nearby_peak_matrix.tsv"))
        if not m:
            continue
        cols = [c for c in pd.read_csv(m[0], sep="\t", nrows=1).columns
                if c.startswith("chr") and "__" in c]
        want = {}
        for c in cols:
            r, k = c.rsplit("__", 1)
            want.setdefault(k, set()).add(r)
        cell_vals, cell_bc = {}, {}
        for mark, regs in sorted(want.items()):
            x, feats, bcs = load_mark_matrix(Path(a.data_dir) / mark, mark)
            vals = per_cell_values(x, feats, sorted(regs))
            cell_bc[mark] = {b: i for i, b in enumerate(bcs)}
            for r in regs:
                cell_vals[f"{r}__{mark}"] = vals[r]

        st, seen, n_ok = stable_edges_for_gene(
            gene, cols, cell_vals, cell_bc, assigns, orders, parts,
            a.alpha, a.depth, a.min_k)
        cross = lambda e: e[0].rsplit("__", 1)[1] != e[1].rsplit("__", 1)[1]
        stable_all |= {frozenset(e) for e in st}
        seen_all |= {frozenset(e) for e in seen}
        print(f"  {gene:8s} {n_ok:2d} partitions   seen {len(seen):4d}   "
              f"stable {len(st):3d}   cross-mark stable {sum(cross(e) for e in st):3d}")

    print("\nbuilding candidate pairs + PCHi-C signature ...")
    pairs = build_pairs(pipe, genes, hub)
    if pairs.empty:
        raise SystemExit("no candidate pairs built")

    key = pairs.apply(lambda r: frozenset((r["peak_a"], r["peak_b"])), axis=1)
    is_cross = pairs.peak_a.str.rsplit("__", n=1).str[-1] != \
        pairs.peak_b.str.rsplit("__", n=1).str[-1]
    pairs["stability_class"] = np.where(
        key.isin(stable_all) & is_cross, "stable_cross",
        np.where(key.isin(seen_all), "unstable_adjacent", "non_adjacent"))

    n_pos = int((pairs.stability_class == "stable_cross").sum())
    print(f"pairs={len(pairs)}  stable_cross={n_pos}  "
          f"unstable_adjacent={int((pairs.stability_class=='unstable_adjacent').sum())}  "
          f"non_adjacent={int((pairs.stability_class=='non_adjacent').sum())}")
    if n_pos < 5:
        raise SystemExit(f"only {n_pos} stable cross-mark pairs matched the peak "
                         f"universe — too few to test")

    if a.edges_out:
        ed = pairs[pairs.stability_class == "stable_cross"].copy()
        ed["mark_a"] = ed.peak_a.str.rsplit("__", n=1).str[-1]
        ed["mark_b"] = ed.peak_b.str.rsplit("__", n=1).str[-1]
        ed = ed[["gene", "peak_a", "peak_b", "mark_a", "mark_b", "distance",
                 "deg_min", "deg_max", "pchic_hub", "edge_class"]]
        ed = ed.sort_values(["pchic_hub", "gene", "distance"],
                            ascending=[False, True, True])
        ed.to_csv(a.edges_out, sep="\t", index=False)
        print(f"wrote {a.edges_out}  ({len(ed)} stable cross-mark edges, "
              f"{int(ed.pchic_hub.sum())} on a shared PCHi-C hub)")

    rows = []
    for ref in ("non_adjacent", "unstable_adjacent"):
        sub = pairs[pairs.stability_class.isin(["stable_cross", ref])].copy()
        sub["edge_class"] = np.where(sub.stability_class == "stable_cross",
                                     "POS", "non_adjacent")
        res = matched_permutation_test(sub, "POS", "pchic_hub",
                                       a.n_bins, a.n_perm, a.seed,
                                       degree_bins=a.degree_bins)
        res.update(reference=ref, signature="pchic_hub")
        rows.append(res)
        print(f"\n=== stable cross-mark vs {ref} ===")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"    {k:24s} {v:.4f}")
            else:
                print(f"    {k:24s} {v}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}")
    print("\nThe 'vs unstable_adjacent' row is the one that matters: it asks whether")
    print("partition stability selects biology, or merely strong correlations.")


if __name__ == "__main__":
    main()
