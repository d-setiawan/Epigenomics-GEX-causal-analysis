"""
Does de-attenuating the covariance yield partition-stable CROSS-MARK edges?

WHY THIS ARM AND NOT THE OTHER ONE
----------------------------------
Signal-share screening was tried first and failed: at matched panel size it
produced fewer edges than a random subset and zero partition-stable ones.  The
variables with marginal between-state signal are not the ones carrying pairwise
structure, so selecting on the marginal throws away the pairs.

De-attenuation is the surviving lever.  In simulation it moved cross-mark
adjacency recall from exactly 0.000 to 0.211 and cross-mark ORIENTATION from
0.000 to 0.158, by removing the same-mark sampling term that inflates same-mark
covariances while contributing nothing to cross-mark ones.  It changes the input
the search consumes and leaves the search itself stock.

THE TEST
--------
For each of 11 independent metacell partitions, run the same FCI on

    raw            covariance of the metacell means -- what the pipeline uses
    deattenuated   the same, minus lambda* x the same-mark sampling term,
                   with lambda* the largest admissible fraction

and count edges recovered in at least `min_k` partitions, split same/cross mark.
Both arms see identical variables, identical partitions and identical search
settings, so any difference is attributable to the covariance alone.

lambda* is refit per partition and reported: it is a property of that partition's
metacell sizes, not a knob.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_deconvolution import load_mark_matrix, per_cell_values  # noqa: E402
from crossmark_screening_stability import complete_metacells, build_matrix  # noqa: E402

REPO = Path(__file__).resolve().parents[4]


def rank_gaussian(X):
    n = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / n)
                            for j in range(X.shape[1])])


def decompose(cols, ch, cell_vals, cell_bc, assign, order, X, good, min_cells=2):
    """Raw covariance of the rank-transformed metacell means, and its noise term.

    The noise term is computed on the SAME rank-transformed scale the search
    sees.  Transforming the metacell means and then decomposing keeps the two
    matrices commensurable; decomposing raw values and correcting a transformed
    covariance would not be.
    """
    Xg = rank_gaussian(X[good])
    raw = np.cov(Xg, rowvar=False)

    # per-cell values pushed through the same monotone map, so the within-metacell
    # spread is expressed in the units the covariance is in
    noise = np.zeros_like(raw)
    for a_i in np.unique(ch):
        vs = np.flatnonzero(ch == a_i)
        mark = cols[vs[0]].rsplit("__", 1)[1]
        sub = assign[assign["mark"] == mark]
        idx = sub["cell"].map(cell_bc[mark])
        ok = idx.notna().to_numpy()
        rows = idx[ok].to_numpy(int)
        mc = sub["metacell_id"].to_numpy()[ok]
        V = np.column_stack([cell_vals[cols[j]][rows] for j in vs])
        # map cell values onto the metacell-mean scale via the empirical CDF of
        # the metacell means, so within- and between- are in the same units
        scale = np.array([Xg[:, j].std() / max(X[good][:, j].std(), 1e-12)
                          for j in vs])
        acc, used = np.zeros((len(vs), len(vs))), 0
        for m in order:
            sel = mc == m
            k = int(sel.sum())
            if k < min_cells:
                continue
            blk = (V[sel] - V[sel].mean(axis=0, keepdims=True)) * scale
            acc += (blk.T @ blk) / (k - 1) / k
            used += 1
        if used:
            noise[np.ix_(vs, vs)] = acc / used
    return raw, noise


def lambda_star(raw, noise, max_cond=1e4):
    lo, hi, best = 0.0, 1.0, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        S = raw - mid * noise
        S = 0.5 * (S + S.T)
        w = np.linalg.eigvalsh(S)
        if w.min() > 0 and w.max() / w.min() <= max_cond:
            best, lo = mid, mid
        else:
            hi = mid
    return best


def fci_on_cov(Sig, names, n, alpha, depth):
    import jpype
    import jpype.imports  # noqa: F401
    import edu.cmu.tetrad.data as td
    from edu.cmu.tetrad.search import Fci
    from edu.cmu.tetrad.search.test import IndTestFisherZ
    from java.util import ArrayList

    alias = [f"X{j}" for j in range(len(names))]
    back = dict(zip(alias, names))
    nodes = ArrayList()
    for nm in alias:
        nodes.add(td.ContinuousVariable(nm))
    arr = jpype.JArray(jpype.JDouble, 2)(np.asarray(Sig, float).tolist())
    mat = jpype.JClass("edu.cmu.tetrad.util.Matrix")(arr)
    cov = td.CovarianceMatrix(nodes, mat, int(n))
    s = Fci(IndTestFisherZ(cov, float(alpha)))
    s.setDepth(int(depth)); s.setStable(True)
    s.setCompleteRuleSetUsed(True); s.setVerbose(False)
    g = s.search()
    out, directed = set(), set()
    for e in g.getEdges():
        u, v = back[str(e.getNode1().getName())], back[str(e.getNode2().getName())]
        eu, ev = str(e.getEndpoint1()), str(e.getEndpoint2())
        key = (u, v) if u < v else (v, u)
        out.add(key)
        # A definitively DIRECTED mark (tail at one end, arrowhead at the
        # other) is what an identifiability claim needs; `o-o` and `<->` are
        # not orientations.  Tracked separately from adjacency because the two
        # questions have had different answers throughout this analysis.
        if {eu, ev} == {"TAIL", "ARROW"}:
            directed.add(key)
    return out, directed


def merge_metacells(assign, order, cols, cell_vals, cell_bc, target):
    """
    Coarsen a partition by merging its metacells, Ward on their mean profiles.

    This holds the embedding, the cells, and the completeness rule fixed and
    varies only granularity, so a comparison across `target` is a comparison of
    aggregation level and not of clustering method.  Returns a modified
    assignment table plus the new (smaller) list of groups.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    if target <= 0 or target >= len(order):
        return assign, order
    prof = build_matrix(cols, cell_vals, cell_bc, assign, order)
    keep = np.isfinite(prof).all(axis=0)
    if keep.sum() < 2:
        return assign, order
    P = prof[:, keep]
    P = (P - P.mean(0)) / np.clip(P.std(0), 1e-9, None)
    lab = fcluster(linkage(P, method="ward"), t=target, criterion="maxclust")
    mapping = {m: int(lab[i]) for i, m in enumerate(order)}

    out = assign.copy()
    out["metacell_id"] = out["metacell_id"].map(mapping)
    out = out[out["metacell_id"].notna()]
    out["metacell_id"] = out["metacell_id"].astype(int)
    return out, sorted(out["metacell_id"].unique())


def summarise(results, min_k):
    """`results` is a list of (adjacency_set, directed_set), one per partition."""
    cnt, dcnt = {}, {}
    for adj, dirc in results:
        for e in adj:
            cnt[e] = cnt.get(e, 0) + 1
        for e in dirc:
            dcnt[e] = dcnt.get(e, 0) + 1
    stable = {e for e, c in cnt.items() if c >= min_k}
    stable_dir = {e for e, c in dcnt.items() if c >= min_k}
    cross = lambda e: e[0].rsplit("__", 1)[1] != e[1].rsplit("__", 1)[1]
    return dict(mean_edges=float(np.mean([len(a) for a, _ in results])),
                n_seen=len(cnt), n_stable=len(stable),
                frac_stable=len(stable) / len(cnt) if cnt else float("nan"),
                n_stable_cross=sum(cross(e) for e in stable),
                n_stable_same=sum(not cross(e) for e in stable),
                n_stable_directed=len(stable_dir),
                n_stable_dir_cross=sum(cross(e) for e in stable_dir),
                stable_edges=sorted(stable))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", default=str(REPO.parent.parent /
                    "py-tetrad/pytetrad/resources/tetrad-current.jar"))
    ap.add_argument("--dataset-root", default=str(
        REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--metacell-root", default=str(REPO / "pipelines/empirical/analysis/outputs/metacells"))
    ap.add_argument("--data-dir", default=str(REPO / "pipelines/empirical/data/outputs"))
    ap.add_argument("--genes", nargs="+",
                    default=["CD14", "CD151", "CD52", "IL4R", "PTPRJ", "LGALS2"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-k", type=int, default=6)
    ap.add_argument("--max-cond", type=float, default=1e4)
    ap.add_argument("--permute", action="store_true",
                    help="add a null arm: permute metacell labels WITHIN each "
                         "mark (except the first), destroying cross-mark "
                         "correspondence while preserving every marginal, each "
                         "mark's own metacell structure, and the whole "
                         "aggregation apparatus")
    ap.add_argument("--permute-seed", type=int, default=0)
    ap.add_argument("--merge-to", type=int, default=0,
                    help="agglomerate each partition's metacells to this many "
                         "groups before building the matrix (0 = leave alone). "
                         "Trades rows for cells-per-metacell, i.e. for signal "
                         "per variable; see crossmark_resolution_tradeoff.py")
    ap.add_argument("--out", default="crossmark_deattenuated_stability.tsv")
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
    print(f"{len(parts)} partitions")

    root = Path(a.dataset_root)
    rows = []
    for gene in a.genes:
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

        marks = sorted(want)
        ch_all = np.array([marks.index(c.rsplit("__", 1)[1]) for c in cols])

        sets = {"raw": [], "deattenuated": [], "raw_matched": []}
        if a.permute:
            sets["permuted"] = []
        lams = []
        cache = []
        min_rows = 20 if a.merge_to else 30
        for n, _ in parts:
            an, on = (merge_metacells(assigns[n], orders[n], cols, cell_vals,
                                      cell_bc, a.merge_to)
                      if a.merge_to else (assigns[n], orders[n]))
            X = build_matrix(cols, cell_vals, cell_bc, an, on)
            good = np.isfinite(X).all(axis=1)
            if good.sum() < min_rows:
                continue
            live = X[good].std(axis=0) > 1e-9
            if live.sum() < 5:
                continue
            sub = np.flatnonzero(live)
            names = [cols[j] for j in sub]
            ch = ch_all[sub]
            raw, noise = decompose([cols[j] for j in sub], ch, cell_vals, cell_bc,
                                   an, on, X[:, sub], good)
            lam = lambda_star(raw, noise, a.max_cond)
            lams.append(lam)
            nrow = int(good.sum())
            sets["raw"].append(fci_on_cov(raw, names, nrow, a.alpha, a.depth))
            sets["deattenuated"].append(
                fci_on_cov(raw - lam * noise, names, nrow, a.alpha, a.depth))
            cache.append((raw, names, nrow))

            if a.permute:
                # Null: reassign metacell labels at random WITHIN each mark
                # after the first.  Each mark keeps its own metacell size
                # distribution and every variable keeps its marginal; only the
                # correspondence BETWEEN marks is destroyed.  Cross-mark edges
                # surviving here are produced by the aggregation apparatus
                # itself, not by shared cell state.
                rng = np.random.default_rng(a.permute_seed + hash(n) % 10000)
                pa = an.copy()
                for mk in sorted(pa["mark"].unique())[1:]:
                    sel = pa["mark"] == mk
                    pa.loc[sel, "metacell_id"] = rng.permutation(
                        pa.loc[sel, "metacell_id"].to_numpy())
                Xp = build_matrix([cols[j] for j in sub], cell_vals, cell_bc,
                                  pa, on)
                gp = np.isfinite(Xp).all(axis=1)
                if gp.sum() >= min_rows and np.all(Xp[gp].std(axis=0) > 1e-9):
                    sets["permuted"].append(fci_on_cov(
                        np.cov(rank_gaussian(Xp[gp]), rowvar=False),
                        names, int(gp.sum()), a.alpha, a.depth))

        if len(sets["raw"]) < 5:
            print(f"  {gene}: too few usable partitions")
            continue

        # THE CONTROL THAT MAKES THIS A TEST.
        # De-attenuation raises the correlations, so it also returns more edges,
        # and more edges yield more partition-stable edges for free.  Loosen the
        # RAW arm's alpha until it emits the same number of edges per partition,
        # then compare.  Only a difference that survives this is attributable to
        # the covariance rather than to an effectively looser threshold.
        target = float(np.mean([len(s[0]) for s in sets["deattenuated"]]))
        lo, hi, alpha_m = a.alpha, 0.9, a.alpha
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            got = np.mean([len(fci_on_cov(r, nm, nr, mid, a.depth)[0])
                           for r, nm, nr in cache])
            if got < target:
                lo, alpha_m = mid, mid
            else:
                hi, alpha_m = mid, mid
            if abs(got - target) / max(target, 1) < 0.05:
                break
        sets["raw_matched"] = [fci_on_cov(r, nm, nr, alpha_m, a.depth)
                               for r, nm, nr in cache]
        print(f"  edge-matched raw arm: alpha {a.alpha} -> {alpha_m:.4f}")
        print(f"\n{gene}: {len(cols)} vars, {len(sets['raw'])} partitions, "
              f"lambda* median {np.median(lams):.3f}"
              + (f", merged to {a.merge_to}" if a.merge_to else ""))
        for arm, es in sets.items():
            if not es:
                continue
            s = summarise(es, a.min_k)
            s.update(gene=gene, arm=arm, n_vars=len(cols),
                     lam=float(np.median(lams)), merge_to=a.merge_to)
            rows.append(s)
            print(f"    {arm:13s} edges/part {s['mean_edges']:6.1f}   "
                  f"stable(>={a.min_k}) {s['n_stable']:3d}   "
                  f"cross {s['n_stable_cross']:3d}   same {s['n_stable_same']:3d}   "
                  f"| DIRECTED stable {s['n_stable_directed']:3d}  "
                  f"cross {s['n_stable_dir_cross']:3d}")

    df = pd.DataFrame(rows)
    df.drop(columns=["stable_edges"]).to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}")
    if df.empty:
        return
    print("\n=== RAW vs DE-ATTENUATED, identical variables and partitions ===")
    g = df.groupby("arm").agg(genes=("gene", "nunique"),
                              edges_per_partition=("mean_edges", "mean"),
                              stable=("n_stable", "mean"),
                              frac_stable=("frac_stable", "mean"),
                              stable_cross=("n_stable_cross", "mean"),
                              stable_same=("n_stable_same", "mean"),
                              stable_directed=("n_stable_directed", "mean"),
                              stable_dir_cross=("n_stable_dir_cross", "mean"))
    print(g.round(3).to_string())
    print("\nTOTALS (summed over genes)")
    tot = df.groupby("arm").agg(cross=("n_stable_cross", "sum"),
                                same=("n_stable_same", "sum"),
                                dir_cross=("n_stable_dir_cross", "sum"))
    print(tot.to_string())

    # which cross-mark edges survive in the de-attenuated arm
    for _, r in df[df.arm == "deattenuated"].iterrows():
        cross = [e for e in r.stable_edges
                 if e[0].rsplit("__", 1)[1] != e[1].rsplit("__", 1)[1]]
        if cross:
            print(f"\n{r.gene}: {len(cross)} partition-stable cross-mark edges")
            for u, v in cross[:12]:
                print(f"    {u}  --  {v}")


if __name__ == "__main__":
    main()
