#!/usr/bin/env python3
"""Implementation step (test harness) — confounder root-nodes + mechanistic orientation tiers.

Encodes the original design (notes/local_multiome_notears_plan.md) and MISSION.md direction:
observed confounders are ROOTS, chromatin is upstream of expression, the gene does NOT cause
chromatin. All three are expressed as ONE tiered-background-knowledge object:

    tier 0 = confounders  ->  tier 1 = peaks  ->  tier 2 = gene expression

Tiers forbid backward edges (peak->confounder, gene->peak, gene->confounder) and, by adding the
confounders as NODES, let FCI CONDITION on them -- so peak<->peak edges that were only shared
library-size/depth co-variation should drop, while genuine peak<->peak structure (the deliverable)
survives. Confounders available now: per-metacell library sizes (libsize__rna, libsize__<mark>);
donor is not threaded through metacells yet (future), cell-cycle optional (--cellcycle).

Compares FCI WITHOUT vs WITH the tiers on the same gene. Run in causaldisco env.
"""
from __future__ import annotations
import argparse, os, sys, re, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype_latent_factor import load_validated_intervals, is_validated, parse_peak  # noqa: E402

MARKS = ["H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3"]


def rank_gaussian(x):
    n = len(x); r = pd.Series(x).rank(method="average").to_numpy()
    return norm.ppf((r - 0.5) / n)


def decollinear(cols, Xr, thr=0.9999):
    order = [c for c in cols if c.startswith("conf__")] + \
            [c for c in cols if c.startswith("expr__")] + \
            [c for c in cols if c.startswith("chr")]
    kept = []
    for c in order:
        if Xr[c].std() <= 0:
            continue
        if all(abs(np.corrcoef(Xr[c], Xr[k])[0, 1]) < thr for k in kept):
            kept.append(c)
    return [c for c in cols if c in set(kept)]


def build_confounders(df):
    """Per-metacell confounders available now: RNA depth + chromatin depth (log)."""
    conf = {}
    if "libsize__rna" in df:
        conf["conf__rna_depth"] = np.log1p(df["libsize__rna"].to_numpy(float))
    marks = [f"libsize__{m}" for m in MARKS if f"libsize__{m}" in df.columns]
    if marks:
        conf["conf__chrom_depth"] = np.log1p(df[marks].mean(1).to_numpy(float))
    return conf


def run_fci(mat, gene, conf_names, peak_names, use_tiers):
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode
    names = list(mat.columns)
    Xr = {c: rank_gaussian(mat[c].to_numpy(float)) for c in names}
    keep = decollinear(names, Xr)
    names = [c for c in names if c in keep]
    X = np.column_stack([Xr[c] for c in names])
    bk = None
    if use_tiers:
        nodes = {n: GraphNode(n) for n in names}
        bk = BackgroundKnowledge()
        for n in names:
            if n in conf_names: bk.add_node_to_tier(nodes[n], 0)
            elif n == gene:     bk.add_node_to_tier(nodes[n], 2)
            else:               bk.add_node_to_tier(nodes[n], 1)
        bk.forbid_within_tier(0)   # confounders are independent roots
    g, edges = fci(X, alpha=0.05, background_knowledge=bk, node_names=names,
                   verbose=False, show_progress=False)
    out = []
    for e in edges:
        a = names[g.node_map[e.node1]]; b = names[g.node_map[e.node2]]; s = str(e)
        t = "<->" if "<->" in s else ("-->" if "-->" in s else ("o->" if "o->" in s else "o-o"))
        out.append((a, b, t))
    return out, names


def classify(edges, gene, conf_names):
    def kind(a, b):
        s = {a, b}
        if gene in s and (s & set(conf_names)): return "conf_gene"
        if gene in s: return "peak_gene"
        if s & set(conf_names): return "conf_peak"
        return "peak_peak"
    from collections import Counter
    k = Counter(kind(a, b) for a, b, t in edges)
    o = Counter(t for a, b, t in edges)
    return k, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--dataset-root", default="CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    a = ap.parse_args()
    g = a.gene
    gene_col = f"expr__{g}_log1p"
    df = pd.read_csv(f"{a.dataset_root}/{g.lower()}/nearby_peaks/{g.upper()}_nearby_peak_matrix.tsv", sep="\t")
    peaks = [c for c in df.columns if str(c).startswith("chr")]
    conf = build_confounders(df)
    conf_names = list(conf)
    base = pd.concat([pd.DataFrame(conf), df[[gene_col]], df[peaks]], axis=1).dropna()
    intervals = load_validated_intervals(g)
    print(f"\n===== {g} =====  metacells={len(base)}  peaks={len(peaks)}  confounders={conf_names}")

    for use in [False, True]:
        edges, used = run_fci(base, gene_col, conf_names, peaks, use)
        k, o = classify(edges, gene_col, conf_names)
        pg = [(a2, b2, t) for a2, b2, t in edges if gene_col in (a2, b2) and not ({a2, b2} & set(conf_names))]
        directed_pg = sum(1 for a2, b2, t in pg if t == "-->")
        val = 0
        for a2, b2, t in pg:
            pk = a2 if a2 != gene_col else b2
            pp = parse_peak(pk)
            if pp and is_validated(pp, intervals): val += 1
        tag = "WITH tiers (conf roots + chromatin->expr)" if use else "baseline (no knowledge)"
        print(f"\n-- {tag} --")
        print(f"  total={len(edges)}  peak_gene={k.get('peak_gene',0)}  peak_peak={k.get('peak_peak',0)}"
              f"  conf_peak={k.get('conf_peak',0)}  conf_gene={k.get('conf_gene',0)}")
        print(f"  orientations={dict(o)}  | peak->gene directed(-->)= {directed_pg}/{len(pg)}  validated peak_gene= {val}")
        for a2, b2, t in pg:
            pk = a2 if a2 != gene_col else b2
            pp = parse_peak(pk); v = is_validated(pp, intervals) if pp else ""
            arrow = f"{a2} {t} {b2}"
            print(f"       {arrow}   {'VALIDATED:'+v if v else ''}")


if __name__ == "__main__":
    main()
