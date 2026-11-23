"""
Does a different transform or model class change the cross-mark EDGE result?

WHY AT THE EDGE LEVEL
---------------------
The marginal per-variable audits (`crossmark_count_signal.py`,
`crossmark_variable_granularity.py`) cannot settle this.  `crossmark_power_check.py`
shows the count-native test rejects 0.000 of the time under its own null at the
observed depth and never reaches 80% power even for a 20-fold between-metacell
swing.  Those audits are therefore uninformative about whether signal exists, and
must not be used to pre-judge a model-class change.

Pairwise power is also simply not marginal power.  A test of "does this variable
vary across metacells" uses one column; a partial correlation across 133
metacells pools information across a pair.  That is how 41 partition-stable
cross-mark adjacencies with external PCHi-C enrichment coexist with variables
that are marginally indistinguishable from noise.  So the question is asked where
the answer lives: on the edges.

THE AXES
--------
transform -- how cell-level counts become a metacell value:

  mean_log1p   mean over the metacell's cells of log1p(count * 1e4 / depth).
               What the pipeline does.
  pseudobulk   log1p(1e4 * pooled counts / pooled depth).  Pools FIRST, so the
               metacell's whole sequencing depth is used instead of being spent
               on per-cell logs.  For 0.5-1.5% dense data the two differ a lot:
               mean-of-logs is dominated by how many cells had any count, and
               log-of-pooled by how much was seen overall.
  detection    fraction of the metacell's cells with a non-zero count.  Discards
               magnitude entirely and keeps only presence, which is the part of
               a very sparse count that is best measured.

test -- what the search assumes about the joint distribution:

  fisherz      Gaussian partial correlation on rank-transformed columns.  Current.
  bf_lrt       Basis-function likelihood ratio.  Nonparametric: expands each
               variable in a truncated basis, so it does not assume linear
               Gaussian dependence.
  dg_lrt       Degenerate-Gaussian likelihood ratio.  Built for mixed
               discrete/continuous data, which is what a mostly-zero count column
               effectively is.

EVALUATION
----------
The same metric used for every other claim in this analysis, so the arms are
comparable to what is already recorded: partition-stable cross-mark adjacencies
across 11 metacell partitions, a within-mark permutation null at matched edge
count, and PCHi-C shared-hub enrichment under a distance-matched permutation.

A config only wins if it beats the baseline on stable cross-mark edges AND those
extra edges survive the null AND the hub enrichment holds.  More edges alone is
not a win -- that lesson is already recorded in the de-attenuation arm, where a
gain vanished against an edge-count-matched control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix  # noqa: E402
from crossmark_count_signal import per_cell_counts  # noqa: E402
from crossmark_screening_stability import complete_metacells  # noqa: E402
from peak_peak_validator import PchicHub, build_pairs, matched_permutation_test  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
TARGET = 1e4


def build_metacell_matrix(cols, store, order, transform):
    """Metacell x variable matrix under one transform."""
    X = np.full((len(order), len(cols)), np.nan)
    pos = {m: i for i, m in enumerate(order)}
    for j, c in enumerate(cols):
        mark = c.rsplit("__", 1)[1]
        st = store[mark]
        cnt, dep, mc = st["counts"][c.rsplit("__", 1)[0]], st["depth"], st["mc"]
        dep_safe = np.where(dep <= 0, 1.0, dep)
        if transform == "mean_log1p":
            v = np.log1p(cnt * (TARGET / dep_safe))
            for m, i in pos.items():
                sel = mc == m
                if sel.any():
                    X[i, j] = v[sel].mean()
        elif transform == "pseudobulk":
            for m, i in pos.items():
                sel = mc == m
                if sel.any():
                    d = dep[sel].sum()
                    X[i, j] = np.log1p(cnt[sel].sum() * TARGET / (d if d > 0 else 1.0))
        elif transform == "detection":
            for m, i in pos.items():
                sel = mc == m
                if sel.any():
                    X[i, j] = float((cnt[sel] > 0).mean())
        else:
            raise ValueError(f"unknown transform {transform!r}")
    return X


def rank_gaussian(X):
    n = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / n)
                            for j in range(X.shape[1])])


def run_fci(X, names, alpha, depth, test):
    import jpype
    import jpype.imports  # noqa: F401
    import edu.cmu.tetrad.data as td
    from edu.cmu.tetrad.search import Fci
    from java.util import ArrayList

    alias = [f"X{j}" for j in range(len(names))]
    back = dict(zip(alias, names))
    nodes = ArrayList()
    for nm in alias:
        nodes.add(td.ContinuousVariable(nm))
    box = td.DoubleDataBox(jpype.JArray(jpype.JDouble, 2)(
        np.asarray(X, float).tolist()))
    ds = td.BoxDataSet(box, nodes)

    if test == "fisherz":
        from edu.cmu.tetrad.search.test import IndTestFisherZ
        t = IndTestFisherZ(ds, float(alpha))
    elif test == "bf_lrt":
        # ctor is (pipelines/empirical/dataSet, truncationLimit, singularityLambda) in this jar --
        # verified by reflection rather than assumed, since a wrong signature
        # throws per partition and silently empties the whole run.
        from edu.cmu.tetrad.search.test import IndTestBasisFunctionLrt
        t = IndTestBasisFunctionLrt(ds, 3, 0.0)
        t.setAlpha(float(alpha))
    elif test == "dg_lrt":
        from edu.cmu.tetrad.search.test import IndTestDegenerateGaussianLrt
        t = IndTestDegenerateGaussianLrt(ds)
        t.setAlpha(float(alpha))
    else:
        raise ValueError(f"unknown test {test!r}")

    s = Fci(t)
    s.setDepth(int(depth)); s.setStable(True)
    s.setCompleteRuleSetUsed(True); s.setVerbose(False)
    g = s.search()
    out = set()
    for e in g.getEdges():
        u, v = back[str(e.getNode1().getName())], back[str(e.getNode2().getName())]
        out.add((u, v) if u < v else (v, u))
    return out


def stable_from_sets(sets, min_k):
    cnt = {}
    for s in sets:
        for e in s:
            cnt[e] = cnt.get(e, 0) + 1
    return {e for e, c in cnt.items() if c >= min_k}, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=str(REPO.parent.parent /
                    "py-tetrad/pytetrad/resources/tetrad-current.jar"))
    ap.add_argument("--dataset-root", default=str(
        REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--pipeline-root", default=str(REPO / "pipelines/empirical/analysis/outputs/pipeline"))
    ap.add_argument("--metacell-root", default=str(REPO / "pipelines/empirical/analysis/outputs/metacells"))
    ap.add_argument("--data-dir", default=str(REPO / "pipelines/empirical/data/outputs"))
    ap.add_argument("--pchic-tsv", default=str(REPO / "pipelines/empirical/data/external/pchic/pchic_mon_gene_links.tsv"))
    ap.add_argument("--genes", nargs="+", default=None, help="default: all with a consensus")
    ap.add_argument("--transform", default="mean_log1p",
                    choices=["mean_log1p", "pseudobulk", "detection"])
    ap.add_argument("--test", default="fisherz", choices=["fisherz", "bf_lrt", "dg_lrt"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-k", type=int, default=6)
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--degree-bins", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--out", default="crossmark_modelclass.tsv")
    a = ap.parse_args()

    import jpype
    import jpype.imports  # noqa: F401
    if not jpype.isJVMStarted():
        jpype.startJVM("-Xmx4g", classpath=[a.jar])

    parts = [("main", Path(a.metacell_root) / "joint_v2")]
    parts += [(f"pb{i}", Path(a.metacell_root) / f"joint_v2_pb{i}") for i in range(10)]
    parts = [(n, p) for n, p in parts if (p / "cell_metacell_assignments.tsv").is_file()]
    assigns = {n: pd.read_csv(p / "cell_metacell_assignments.tsv", sep="\t") for n, p in parts}
    orders = {n: complete_metacells(df) for n, df in assigns.items()}

    root, pipe = Path(a.dataset_root), Path(a.pipeline_root)
    genes = a.genes or sorted(p.name for p in pipe.iterdir()
                              if (p / "edges_consensus.tsv").is_file())
    print(f"[{a.transform}/{a.test}] {len(genes)} genes, {len(parts)} partitions")

    stable_all, seen_all, per_gene_rows = set(), set(), []
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

        base = {}
        for mark, regs in sorted(want.items()):
            x, feats, bcs = load_mark_matrix(Path(a.data_dir) / mark, mark)
            counts, depth = per_cell_counts(x, feats, sorted(regs))
            base[mark] = (counts, depth, {b: i for i, b in enumerate(bcs)})

        sets, null_sets = [], []
        for n, _ in parts:
            store = {}
            for mark, (counts, depth, pos) in base.items():
                sub = assigns[n][assigns[n]["mark"] == mark]
                idx = sub["cell"].map(pos)
                ok = idx.notna().to_numpy()
                rr = idx[ok].to_numpy(int)
                store[mark] = dict(counts={r: v[rr] for r, v in counts.items()},
                                   depth=depth[rr],
                                   mc=sub["metacell_id"].to_numpy()[ok])
            X = build_metacell_matrix(cols, store, orders[n], a.transform)
            good = np.isfinite(X).all(axis=1)
            if good.sum() < 30:
                continue
            live = X[good].std(axis=0) > 1e-9
            if live.sum() < 5:
                continue
            sub_idx = np.flatnonzero(live)
            names = [cols[j] for j in sub_idx]
            try:
                sets.append(run_fci(rank_gaussian(X[good][:, sub_idx]), names,
                                    a.alpha, a.depth, a.test))
            except Exception as e:
                # Fail fast on the very first search: a bad test signature or an
                # unusable transform throws identically on every partition, and
                # discovering that after the whole grid has run wastes the run.
                if not sets and not per_gene_rows:
                    raise RuntimeError(
                        f"first search failed for {a.transform}/{a.test}: {e}") from e
                print(f"  ! {gene}/{n}: {e}")
                continue

            if a.permute:
                rng = np.random.default_rng(abs(hash(n)) % 10000)
                pstore = {}
                for mi, (mark, st) in enumerate(store.items()):
                    mc = st["mc"] if mi == 0 else rng.permutation(st["mc"])
                    pstore[mark] = dict(counts=st["counts"], depth=st["depth"], mc=mc)
                Xp = build_metacell_matrix([cols[j] for j in sub_idx], pstore,
                                           orders[n], a.transform)
                gp = np.isfinite(Xp).all(axis=1)
                if gp.sum() >= 30 and np.all(Xp[gp].std(axis=0) > 1e-9):
                    try:
                        null_sets.append(run_fci(rank_gaussian(Xp[gp]), names,
                                                 a.alpha, a.depth, a.test))
                    except Exception:
                        pass

        if len(sets) < 5:
            print(f"  {gene:8s} too few usable partitions")
            continue
        st, cnt = stable_from_sets(sets, a.min_k)
        cross = lambda e: e[0].rsplit("__", 1)[1] != e[1].rsplit("__", 1)[1]
        nst_cross = sum(cross(e) for e in st)
        null_cross = 0
        if null_sets:
            nst, _ = stable_from_sets(null_sets, a.min_k)
            null_cross = sum(cross(e) for e in nst)
        stable_all |= {frozenset(e) for e in st}
        seen_all |= {frozenset(e) for e in cnt}
        per_gene_rows.append(dict(gene=gene, n_partitions=len(sets),
                                  edges_per_part=float(np.mean([len(s) for s in sets])),
                                  n_stable=len(st), n_stable_cross=nst_cross,
                                  null_stable_cross=null_cross))
        print(f"  {gene:8s} edges/part {per_gene_rows[-1]['edges_per_part']:6.1f}   "
              f"stable {len(st):3d}   cross {nst_cross:3d}   null-cross {null_cross:3d}")

    pchic = Path(a.pchic_tsv)
    if not pchic.is_file():
        pchic = REPO / "pipelines/empirical/data/raw/external/pchic/pchic_mon_gene_links.tsv"
    pairs = build_pairs(pipe, [r["gene"] for r in per_gene_rows], PchicHub(pchic))
    key = pairs.apply(lambda r: frozenset((r["peak_a"], r["peak_b"])), axis=1)
    is_cross = pairs.peak_a.str.rsplit("__", n=1).str[-1] != \
        pairs.peak_b.str.rsplit("__", n=1).str[-1]
    pairs["stability_class"] = np.where(
        key.isin(stable_all) & is_cross, "stable_cross",
        np.where(key.isin(seen_all), "unstable_adjacent", "non_adjacent"))

    n_pos = int((pairs.stability_class == "stable_cross").sum())
    tot_cross = sum(r["n_stable_cross"] for r in per_gene_rows)
    tot_null = sum(r["null_stable_cross"] for r in per_gene_rows)
    print(f"\n[{a.transform}/{a.test}] stable cross-mark {tot_cross}  "
          f"null {tot_null}  matched-to-universe {n_pos}")

    res_rows = []
    if n_pos >= 5:
        for ref in ("non_adjacent", "unstable_adjacent"):
            sub = pairs[pairs.stability_class.isin(["stable_cross", ref])].copy()
            sub["edge_class"] = np.where(sub.stability_class == "stable_cross",
                                         "POS", "non_adjacent")
            r = matched_permutation_test(sub, "POS", "pchic_hub", a.n_bins,
                                         a.n_perm, 0, degree_bins=a.degree_bins)
            r.update(reference=ref, transform=a.transform, test=a.test,
                     stable_cross=tot_cross, null_cross=tot_null)
            res_rows.append(r)
            print(f"  vs {ref:20s} hub {r['pos_sig_rate']:.3f} vs {r['neg_sig_rate']:.3f}"
                  f"   diff {r['matched_rate_diff']:+.4f}   p {r['perm_p']:.4f}")
    else:
        print(f"  only {n_pos} positives matched the universe — enrichment not tested")

    pd.DataFrame(per_gene_rows).assign(transform=a.transform, test=a.test) \
        .to_csv(a.out.replace(".tsv", "_pergene.tsv"), sep="\t", index=False)
    pd.DataFrame(res_rows).to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
