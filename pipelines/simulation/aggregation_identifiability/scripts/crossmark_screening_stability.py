"""
Does signal-share screening produce partition-stable cross-mark edges?

THE PRIOR RESULT THIS ANSWERS TO
--------------------------------
Edges from this pipeline are partition-fragile: none of the previously
validated edges survived a change of metacell partition.  The SNR audit says
why -- roughly 85% of panel variables have no between-state signal at all,
their metacell-level variance being fully accounted for by the noise of
averaging ~10 cells.  Noise variables do not merely fail to yield edges; they
sit in every conditioning set and destroy power for the variables that do
carry signal.

THE MODIFICATION
----------------
Screen variables on signal share BEFORE the search, keeping only those whose
metacell variance is not explained by sampling noise.  The screen uses
marginal variances only -- never a pairwise association -- so it cannot bias
the CI tests toward finding edges, which a correlation-based screen would.

THE CONTROL THAT MAKES IT A TEST
--------------------------------
A screened panel is smaller, and smaller graphs are more stable for free:
fewer variables, fewer possible edges, fewer conditioning sets. Comparing
screened against full would confound "the screen selected signal" with "the
screen selected fewer variables".

So every screened arm is matched by RANDOM arms of IDENTICAL size, drawn from
the same panel.  The claim is only supported if screened beats random at
matched size.  Random draws are repeated so the comparison has a spread, not a
point.

Screening uses the signal share MEDIANED ACROSS PARTITIONS, then applies one
fixed variable set to every partition, so the retained set cannot vary with the
partition being scored and inflate agreement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix, per_cell_values  # noqa: E402

REPO = Path(__file__).resolve().parents[4]


def complete_metacells(assign):
    mods = sorted(assign["mark"].unique())
    counts = assign.groupby(["metacell_id", "mark"]).size().unstack(fill_value=0)
    return sorted(counts.index[(counts[mods] > 0).all(axis=1)].to_numpy())


def build_matrix(cols, cell_vals, cell_bc, assign, order):
    """Metacell means for `cols` under one partition's assignment table."""
    X = np.full((len(order), len(cols)), np.nan)
    pos = {m: i for i, m in enumerate(order)}
    for j, c in enumerate(cols):
        mark = c.rsplit("__", 1)[1]
        sub = assign[assign["mark"] == mark]
        idx = sub["cell"].map(cell_bc[mark])
        ok = idx.notna().to_numpy()
        v = cell_vals[c][idx[ok].to_numpy(int)]
        mc = sub["metacell_id"].to_numpy()[ok]
        for m, i in pos.items():
            sel = mc == m
            if sel.any():
                X[i, j] = np.nanmean(v[sel])
    return X


def signal_share(cols, cell_vals, cell_bc, assign, order, min_cells=2):
    """1 - E_c[Var(V|c)/n_c] / Var(metacell means), per variable."""
    X = build_matrix(cols, cell_vals, cell_bc, assign, order)
    good = np.isfinite(X).all(axis=1)
    Xg = X[good]
    tot = Xg.var(axis=0, ddof=1)
    noise = np.zeros(len(cols))
    for j, c in enumerate(cols):
        mark = c.rsplit("__", 1)[1]
        sub = assign[assign["mark"] == mark]
        idx = sub["cell"].map(cell_bc[mark])
        ok = idx.notna().to_numpy()
        v = cell_vals[c][idx[ok].to_numpy(int)]
        mc = sub["metacell_id"].to_numpy()[ok]
        acc, used = 0.0, 0
        for m in order:
            sel = mc == m
            k = int(sel.sum())
            if k < min_cells:
                continue
            acc += v[sel].var(ddof=1) / k
            used += 1
        noise[j] = acc / used if used else np.nan
    return 1.0 - noise / np.clip(tot, 1e-12, None), X, good


def rank_gaussian(X):
    n = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / n)
                            for j in range(X.shape[1])])


def run_fci(X, names, alpha, depth):
    import jpype
    import jpype.imports  # noqa: F401  -- installs the `edu.*` import hook
    import edu.cmu.tetrad.data as td
    from edu.cmu.tetrad.search import Fci
    from edu.cmu.tetrad.search.test import IndTestFisherZ
    from java.util import ArrayList
    # Tetrad parses variable names, and a peak label like
    # "chr1:47714591-47715464__H3K4me2" trips its number parser.  Search on
    # positional aliases and translate the edges back.
    alias = [f"X{j}" for j in range(len(names))]
    back = dict(zip(alias, names))

    nodes = ArrayList()
    for n in alias:
        nodes.add(td.ContinuousVariable(n))
    box = td.DoubleDataBox(jpype.JArray(jpype.JDouble, 2)(
        np.asarray(X, float).tolist()))
    ds = td.BoxDataSet(box, nodes)
    s = Fci(IndTestFisherZ(ds, float(alpha)))
    s.setDepth(int(depth)); s.setStable(True)
    s.setCompleteRuleSetUsed(True); s.setVerbose(False)
    g = s.search()
    out = set()
    for e in g.getEdges():
        u = back[str(e.getNode1().getName())]
        v = back[str(e.getNode2().getName())]
        out.add((u, v) if u < v else (v, u))
    return out


def stability(edge_sets, min_k):
    """Edges seen in at least min_k of the partitions, split by channel type."""
    cnt = {}
    for es in edge_sets:
        for e in es:
            cnt[e] = cnt.get(e, 0) + 1
    stable = {e for e, c in cnt.items() if c >= min_k}
    seen = set(cnt)
    cross = lambda e: e[0].rsplit("__", 1)[1] != e[1].rsplit("__", 1)[1]
    return dict(n_seen=len(seen), n_stable=len(stable),
                frac_stable=len(stable) / len(seen) if seen else float("nan"),
                n_stable_cross=sum(cross(e) for e in stable),
                n_stable_same=sum(not cross(e) for e in stable),
                n_seen_cross=sum(cross(e) for e in seen),
                mean_edges=float(np.mean([len(es) for es in edge_sets])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=str(REPO.parent.parent /
                    "py-tetrad/pytetrad/resources/tetrad-current.jar"))
    ap.add_argument("--dataset-root", default=str(
        REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--metacell-root", default=str(
        REPO / "pipelines/empirical/analysis/outputs/metacells"))
    ap.add_argument("--data-dir", default=str(REPO / "pipelines/empirical/data/outputs"))
    ap.add_argument("--genes", nargs="+", default=["CD14", "CD151", "PTPRJ", "IL4R", "CD52"])
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--min-vars", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-k", type=int, default=6, help="partitions an edge must appear in")
    ap.add_argument("--out", default="crossmark_screening_stability.tsv")
    a = ap.parse_args()

    import jpype
    import jpype.imports  # noqa: F401
    if not jpype.isJVMStarted():
        jpype.startJVM("-Xmx4g", classpath=[a.jar])

    parts = [("main", Path(a.metacell_root) / "joint_v2")]
    parts += [(f"pb{i}", Path(a.metacell_root) / f"joint_v2_pb{i}") for i in range(10)]
    parts = [(n, p) for n, p in parts if (p / "cell_metacell_assignments.tsv").is_file()]
    print(f"{len(parts)} partitions: {', '.join(n for n, _ in parts)}")

    assigns = {n: pd.read_csv(p / "cell_metacell_assignments.tsv", sep="\t")
               for n, p in parts}
    orders = {n: complete_metacells(df) for n, df in assigns.items()}

    root = Path(a.dataset_root)
    rows, summary = [], []
    for gene in a.genes:
        m = list(root.glob(f"{gene.lower()}/nearby_peaks/*_nearby_peak_matrix.tsv"))
        if not m:
            print(f"  ! {gene}: no matrix")
            continue
        cols = [c for c in pd.read_csv(m[0], sep="\t", nrows=1).columns
                if c.startswith("chr") and "__" in c]
        want = {}
        for c in cols:
            reg, mark = c.rsplit("__", 1)
            want.setdefault(mark, set()).add(reg)

        cell_vals, cell_bc = {}, {}
        for mark, regs in sorted(want.items()):
            x, feats, bcs = load_mark_matrix(Path(a.data_dir) / mark, mark)
            vals = per_cell_values(x, feats, sorted(regs))
            cell_bc[mark] = {b: i for i, b in enumerate(bcs)}
            for reg in regs:
                cell_vals[f"{reg}__{mark}"] = vals[reg]

        shares, mats = {}, {}
        for n, _ in parts:
            sh, X, good = signal_share(cols, cell_vals, cell_bc,
                                       assigns[n], orders[n])
            shares[n] = sh
            mats[n] = (X, good)
        S = np.column_stack([shares[n] for n, _ in parts])
        med = np.nanmedian(S, axis=1)
        # Screening on the cross-partition MEDIAN is deliberately stricter than
        # screening on any one partition: a variable has to carry signal
        # wherever the metacell boundaries fall.  Report the attrition, because
        # a variable whose signal share swings with the partition was never a
        # stable object to build an edge on.
        print(f"\n{gene}: share>{a.threshold} in "
              f"{(S > a.threshold).sum(axis=0).min()}-"
              f"{(S > a.threshold).sum(axis=0).max()} vars per single partition, "
              f"{int((med > a.threshold).sum())} on the cross-partition median")
        keep = np.flatnonzero(med > a.threshold)
        for j, c in enumerate(cols):
            rows.append(dict(gene=gene, variable=c, median_share=float(med[j]),
                             kept=bool(j in set(keep.tolist()))))
        if keep.size < a.min_vars:
            print(f"  {gene}: only {keep.size} vars above {a.threshold}, skipping")
            continue
        n_marks_kept = len({cols[j].rsplit('__', 1)[1] for j in keep})
        print(f"\n{gene}: {len(cols)} vars -> {keep.size} screened "
              f"(share > {a.threshold}) over {n_marks_kept} marks")

        def arm(idx, label):
            sets = []
            for n, _ in parts:
                X, good = mats[n]
                Xi = X[:, idx][good]
                if Xi.shape[0] < 30 or not np.isfinite(Xi).all():
                    continue
                # Peaks that are all-zero across metacells are constant, and a
                # constant column has no rank transform and no partial
                # correlation.  Drop those columns for this partition rather
                # than discarding the partition -- they carry no edges either
                # way, and dropping the partition silently shrank every arm.
                live = Xi.std(axis=0) > 1e-9
                if live.sum() < 3:
                    continue
                sets.append(run_fci(rank_gaussian(Xi[:, live]),
                                    [cols[j] for j in np.asarray(idx)[live]],
                                    a.alpha, a.depth))
            if len(sets) < 5:
                return None
            st = stability(sets, a.min_k)
            st.update(gene=gene, arm=label, n_vars=len(idx),
                      n_partitions=len(sets))
            return st

        full = arm(np.arange(len(cols)), "full")
        scr = arm(keep, "screened")
        rng = np.random.default_rng(0)
        rands = []
        for r in range(a.n_random):
            ridx = np.sort(rng.choice(len(cols), size=keep.size, replace=False))
            got = arm(ridx, f"random{r}")
            if got:
                rands.append(got)
        for s in filter(None, [full, scr]) :
            summary.append(s)
        summary.extend(rands)

        def show(s):
            if not s:
                return "        (insufficient)"
            return (f"        edges/partition {s['mean_edges']:6.1f}   "
                    f"stable(>={a.min_k}/{s['n_partitions']}) {s['n_stable']:3d}   "
                    f"frac {s['frac_stable']:.3f}   "
                    f"cross-mark stable {s['n_stable_cross']:3d}")
        print(f"    full     ({len(cols):3d} vars)\n{show(full)}")
        print(f"    screened ({keep.size:3d} vars)\n{show(scr)}")
        if rands:
            print(f"    random   ({keep.size:3d} vars, {len(rands)} draws)")
            print(f"        stable  mean {np.mean([r['n_stable'] for r in rands]):5.1f}  "
                  f"max {max(r['n_stable'] for r in rands):3d}   "
                  f"cross-mark stable mean "
                  f"{np.mean([r['n_stable_cross'] for r in rands]):5.1f}")

    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(a.out.replace(".tsv", "_summary.tsv"), sep="\t", index=False)
    print(f"\nwrote {a.out} and _summary.tsv")

    if sm.empty:
        return
    sm["kind"] = np.where(sm.arm.str.startswith("random"), "random", sm.arm)
    print("\n=== SCREENED vs RANDOM AT MATCHED SIZE (the controlled contrast) ===")
    g = sm.groupby("kind").agg(
        n=("n_stable", "size"), vars=("n_vars", "mean"),
        edges_per_partition=("mean_edges", "mean"),
        stable=("n_stable", "mean"), frac_stable=("frac_stable", "mean"),
        stable_cross=("n_stable_cross", "mean"),
        stable_same=("n_stable_same", "mean"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
