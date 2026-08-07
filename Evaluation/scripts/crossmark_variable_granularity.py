"""
Is the binding constraint the model class, or the variable definition?

WHAT THE COUNT AUDIT ESTABLISHED
--------------------------------
`crossmark_count_signal.py` asked "does this variable carry between-metacell
signal" four ways -- a Gaussian variance decomposition, a count-native Poisson
dispersion, a Poisson deviance test, and a distribution-free Kruskal-Wallis --
and they agree.  The count-native measures are if anything harsher, and not one
variable that the Gaussian view calls dead is revived by the count test.  So the
noise-domination finding is a property of the data, not of the Gaussian framing.

The depth table says why, and it is not subtle.  The peak x cell matrices are
**0.5-1.5% dense** (verified from the matrix headers, independent of any
extraction code).  A peak is detected in roughly 1% of cells, and a metacell
holds ~11 cells per mark, so the expected number of detections for a given
(peak, metacell) is ~0.1.  The overwhelming majority of entries in the metacell
matrix are zero because nothing was ever observed there.

No model class fixes a variable that is zero almost everywhere.  Gaussian,
Poisson, kernel, nonparametric -- all are being asked to find structure in a
column that is mostly the absence of measurement.  The lever is therefore the
**variable definition**, not the likelihood.

WHAT THIS TESTS
---------------
Pool ADJACENT peaks of the same mark into coarser regions and ask whether real
between-metacell signal appears.  Adjacency matters: merging peaks that are
megabases apart would produce a variable with counts but no interpretation, so
peaks are sorted by position and merged only with their neighbours, and the
resulting genomic span is reported so an uninterpretable merge is visible.

    peak    every peak its own variable (current pipeline)
    merge2  adjacent peaks pooled in twos
    merge4  adjacent peaks pooled in fours
    merge8  adjacent peaks pooled in eights
    mark    every peak of a mark at the locus pooled into one variable

At each level the same four measures are recomputed.  The question is whether
`pct_dev_sig` -- the fraction of variables whose rate genuinely varies across
metacells beyond counting noise -- rises from its current floor of zero.

Reading the outcome:
  * signal appears  -> the variable definition was the constraint, the cross-mark
                       question deserves re-running at that granularity, and the
                       interpretation shifts from "this element" to "this region".
  * signal does not -> the locus carries no state-dependent chromatin variation
                       this assay can resolve at any granularity, which is a hard
                       endpoint but a real one, and no algorithm change addresses it.
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
from crossmark_screening_stability import complete_metacells  # noqa: E402
from crossmark_count_signal import (per_cell_counts, gaussian_share,  # noqa: E402
                                    poisson_measures, kruskal_p)

REPO = Path(__file__).resolve().parents[2]

LEVELS = {"peak": 1, "merge2": 2, "merge4": 4, "merge8": 8, "mark": 10**6}


def group_adjacent(regions, k):
    """Sort peaks by start and pool neighbours in blocks of k."""
    parsed = [(parse_interval(r), r) for r in regions]
    parsed.sort(key=lambda t: (t[0][0], t[0][1]))
    out = []
    for i in range(0, len(parsed), k):
        blk = parsed[i:i + k]
        chrom = blk[0][0][0]
        if any(b[0][0] != chrom for b in blk):        # never merge across chromosomes
            for b in blk:
                out.append(([b[1]], b[0][2] - b[0][1]))
            continue
        span = max(b[0][2] for b in blk) - min(b[0][1] for b in blk)
        out.append(([b[1] for b in blk], span))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=str(
        REPO / "CausalDiscovery/outputs/datasets/joint_v2"
             / "monocyte_cuttag_peak_genes_300kb_metacell_expanded"))
    ap.add_argument("--assignments", default=str(
        REPO / "CausalDiscovery/outputs/metacells/joint_v2/cell_metacell_assignments.tsv"))
    ap.add_argument("--data-dir", default=str(REPO / "Data/outputs"))
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--out", default="crossmark_variable_granularity.tsv")
    a = ap.parse_args()

    assign = pd.read_csv(a.assignments, sep="\t")
    order = complete_metacells(assign)
    print(f"{len(order)} complete metacells")

    root = Path(a.dataset_root)
    per_gene = {}
    want = {}
    for m in sorted(root.glob("*/nearby_peaks/*_nearby_peak_matrix.tsv")):
        cols = [c for c in pd.read_csv(m, sep="\t", nrows=1).columns
                if c.startswith("chr") and "__" in c]
        per_gene[m.name.split("_nearby")[0]] = cols
        for c in cols:
            reg, mark = c.rsplit("__", 1)
            want.setdefault(mark, set()).add(reg)

    # per-cell raw counts and depth, once per mark
    store = {}
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
        store[mark] = dict(counts={r: v[rr] for r, v in counts.items()},
                           depth=depth[rr],
                           mc=sub["metacell_id"].to_numpy()[ok])
        print(f"  {mark:9s} {len(regs):4d} regions, {ok.sum():5d} cells")

    rows = []
    for level in a.levels:
        k = LEVELS[level]
        for gene, cols in sorted(per_gene.items()):
            by_mark = {}
            for c in cols:
                reg, mark = c.rsplit("__", 1)
                by_mark.setdefault(mark, []).append(reg)
            for mark, regs in by_mark.items():
                if mark not in store:
                    continue
                st = store[mark]
                dep = st["depth"]
                dep_safe = np.where(dep <= 0, 1.0, dep)
                for grp, span in group_adjacent(regs, k):
                    cnt = np.zeros_like(dep)
                    for r in grp:
                        cnt = cnt + st["counts"][r]
                    vlog = np.log1p(cnt * (a.target_sum / dep_safe))
                    g = gaussian_share(vlog, st["mc"], order)
                    ps, pdev, tot = poisson_measures(cnt, dep, st["mc"], order)
                    kw = kruskal_p(vlog, st["mc"], order)
                    rows.append(dict(level=level, gene=gene, mark=mark,
                                     n_peaks=len(grp), span_bp=span,
                                     gaussian_share=g, poisson_share=ps,
                                     deviance_p=pdev, kw_p=kw,
                                     total_counts=tot,
                                     detection_rate=float((cnt > 0).mean())))
        print(f"  level {level:7s} done")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}  ({len(df)} variable-instances)")

    from statsmodels.stats.multitest import multipletests
    for col in ("deviance_p", "kw_p"):
        df[col + "_fdr"] = np.nan
        for lv, sub in df.groupby("level"):
            m = sub[col].notna()
            if m.sum() > 1:
                df.loc[sub.index[m], col + "_fdr"] = multipletests(
                    sub.loc[m, col], method="fdr_bh")[1]

    print("\n=== DOES SIGNAL APPEAR AS VARIABLES GET COARSER? ===")
    g = df.groupby("level").agg(
        n_vars=("gene", "size"),
        median_counts=("total_counts", "median"),
        detection=("detection_rate", "median"),
        median_span_kb=("span_bp", lambda s: s.median() / 1000),
        gaussian=("gaussian_share", "median"),
        poisson=("poisson_share", "median"),
        pct_gauss_gt25=("gaussian_share", lambda s: 100 * (s > 0.25).mean()),
        pct_dev_sig=("deviance_p_fdr", lambda s: 100 * (s < 0.05).mean()),
        pct_kw_sig=("kw_p_fdr", lambda s: 100 * (s < 0.05).mean()))
    g = g.reindex([l for l in a.levels if l in g.index])
    print(g.round(3).to_string())

    print("\n  pct_dev_sig is the count-native answer: the fraction of variables")
    print("  whose rate genuinely varies across metacells beyond counting noise.")
    print("  It sits at 0.0 for per-peak variables.  If it rises with pooling,")
    print("  the variable definition was the constraint, not the model class.")


if __name__ == "__main__":
    main()
