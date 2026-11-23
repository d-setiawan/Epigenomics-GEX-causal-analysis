#!/usr/bin/env python3
"""EXPLORATORY PROTOTYPE (not production) — element-level causal modeling.

Motivation (see notes/element_level_modeling_design.md): at the peak level, ~92% of
discovered edges are peak<->peak and ~half are bidirected (latent common cause), because each
(region x mark) is treated as an independent causal variable while many peaks are redundant
readouts of one regulatory element's activity or are co-driven by shared latent regulators.

This prototype collapses co-located mark-peaks into REGULATORY ELEMENTS:
  * merge peaks (any mark) whose intervals are within GAP bp into one element;
  * element "activity" = PC1 of its z-scored member-peak signals (the shared axis the marks
    co-measure); PC1 explained-variance ratio is reported as a homogeneity check;
  * element labelled by chromatin state from its constituent marks.
Then rerun GFCI + FCI on {element activities + gene expression} and compare structure vs peaks.

Run in the causaldisco env with JAVA_HOME set (for GFCI):
  JAVA_HOME=/home/dgsetiawan/anaconda3/envs/causaldisco \
  /home/dgsetiawan/anaconda3/envs/causaldisco/bin/python \
  pipelines/empirical/analysis/scripts/prototypes/prototype_element_aggregation.py --gene CD14
"""
from __future__ import annotations
import argparse, re, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

MARK_STATE = None
ACTIVATING = {"H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3"}


def parse_peak(col):
    m = re.match(r"(chr[^:]+):(\d+)-(\d+)__(.+)", str(col))
    if not m:
        return None
    return dict(col=col, chrom=m.group(1), start=int(m.group(2)), end=int(m.group(3)), mark=m.group(4))


def chromatin_state(marks: set) -> str:
    if "H3K4me3" in marks:
        return "active_promoter" if "H3K27ac" in marks else "promoter"
    if "H3K27ac" in marks:
        return "active_enhancer"
    if "H3K4me1" in marks or "H3K4me2" in marks:
        return "poised_enhancer"
    if "H3K27me3" in marks:
        return "polycomb_repressed"
    if "H3K9me3" in marks:
        return "heterochromatin"
    return "other"


def merge_elements(peaks, gap=2000):
    peaks = sorted(peaks, key=lambda p: (p["chrom"], p["start"]))
    elems, cur = [], None
    for p in peaks:
        if cur and p["chrom"] == cur["chrom"] and p["start"] <= cur["end"] + gap:
            cur["members"].append(p); cur["end"] = max(cur["end"], p["end"])
        else:
            cur = dict(chrom=p["chrom"], start=p["start"], end=p["end"], members=[p]); elems.append(cur)
    return elems


def rank_gaussian(x):
    n = len(x); r = pd.Series(x).rank(method="average").to_numpy()
    return norm.ppf((r - 0.5) / n)


def build_element_matrix(df, gene_col, gap=2000):
    peaks = [p for p in (parse_peak(c) for c in df.columns if str(c).startswith("chr")) if p]
    elems = merge_elements(peaks, gap=gap)
    out = {gene_col: df[gene_col].to_numpy(float)}
    meta = []
    for i, e in enumerate(elems):
        cols = [m["col"] for m in e["members"]]
        marks = sorted({m["mark"] for m in e["members"]})
        Z = np.column_stack([(lambda v: (v - v.mean()) / (v.std() + 1e-9))(df[c].to_numpy(float)) for c in cols])
        if Z.shape[1] == 1:
            act = Z[:, 0]; ev = 1.0
        else:
            U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            act = U[:, 0] * S[0]; ev = float(S[0] ** 2 / (S ** 2).sum())
            if np.corrcoef(act, Z.mean(1))[0, 1] < 0:
                act = -act
        name = f"E{i:02d}_{chromatin_state(set(marks))}"
        out[name] = act
        meta.append(dict(element=name, chrom=e["chrom"], start=e["start"], end=e["end"],
                         span_kb=round((e["end"] - e["start"]) / 1000, 1), n_marks=len(marks),
                         marks=",".join(marks), state=chromatin_state(set(marks)),
                         pc1_explained=round(ev, 2), member_peaks=";".join(cols)))
    return pd.DataFrame(out), pd.DataFrame(meta)


def decollinear(mat, thr=0.9999):
    """Drop columns near-perfectly collinear with an earlier kept column (gene first) so the
    fisherz correlation matrix is non-singular (same guard as the production worker)."""
    cols = list(mat.columns)
    order = [c for c in cols if c.startswith("expr__")] + [c for c in cols if not c.startswith("expr__")]
    Xr = {c: rank_gaussian(mat[c].to_numpy(float)) for c in cols}
    kept = []
    for c in order:
        xi = Xr[c]
        if xi.std() <= 0:
            continue
        if all(abs(np.corrcoef(xi, Xr[k])[0, 1]) < thr for k in kept):
            kept.append(c)
    return mat[[c for c in cols if c in set(kept)]]


def run_fci(mat, alpha=0.05):
    from causallearn.search.ConstraintBased.FCI import fci
    mat = decollinear(mat)
    X = np.column_stack([rank_gaussian(mat[c].to_numpy(float)) for c in mat.columns])
    g, edges = fci(X, alpha=alpha, verbose=False, show_progress=False)
    names = list(mat.columns)
    out = []
    for e in edges:
        a = names[g.node_map[e.node1]]; b = names[g.node_map[e.node2]]
        s = str(e)
        etype = "<->" if "<->" in s else ("-->" if "-->" in s else ("o->" if "o->" in s else "o-o"))
        out.append((a, b, etype))
    return out


def run_gfci(mat, alpha=0.05, pd_=2):
    from pytetrad.tools import TetradSearch as ts
    mat = decollinear(mat)
    X = pd.DataFrame({c: rank_gaussian(mat[c].to_numpy(float)) for c in mat.columns})
    s = ts.TetradSearch(X); s.use_sem_bic(penalty_discount=pd_); s.use_fisher_z(alpha=alpha)
    s.run_gfci()
    out = []
    for line in str(s.get_string()).splitlines():
        line = line.strip()
        if ". " in line and any(x in line for x in ["-->", "o->", "<->", "o-o"]):
            body = line.split(". ", 1)[1]
            for sep in ["-->", "o->", "<->", "o-o"]:
                if sep in body:
                    a, b = [t.strip() for t in body.split(sep)]; out.append((a, b, sep)); break
    return out


def summarize(edges, gene_col, label):
    ge = [e for e in edges if gene_col in (e[0], e[1])]
    ee = [e for e in edges if gene_col not in (e[0], e[1])]
    from collections import Counter
    oc = Counter(e[2] for e in edges)
    print(f"  [{label}] edges={len(edges)}  node->gene={len(ge)}  node<->node={len(ee)}  orient={dict(oc)}")
    return ge, ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--dataset-root", default="pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    ap.add_argument("--gap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    g = a.gene
    mpath = f"{a.dataset_root}/{g.lower()}/nearby_peaks/{g.upper()}_nearby_peak_matrix.tsv"
    df = pd.read_csv(mpath, sep="\t")
    gene_col = f"expr__{g}_log1p"
    peak_cols = [c for c in df.columns if str(c).startswith("chr")]
    sub = df[[gene_col] + peak_cols].dropna()
    print(f"\n===== {g} =====  metacells={len(sub)}  peaks={len(peak_cols)}")

    emat, meta = build_element_matrix(pd.concat([sub[[gene_col]], df[peak_cols]], axis=1).dropna(), gene_col, gap=a.gap)
    print(f"peaks {len(peak_cols)} -> elements {len(meta)}  "
          f"(multi-mark elements: {(meta.n_marks>1).sum()}; median PC1 explained where multi-mark: "
          f"{meta.loc[meta.n_marks>1,'pc1_explained'].median() if (meta.n_marks>1).any() else float('nan')})")

    print("\n-- PEAK level --")
    pk = pd.concat([sub[[gene_col]], df[peak_cols]], axis=1).dropna()
    p_gfci = run_gfci(pk, a.alpha); pg_ge, pg_ee = summarize(p_gfci, gene_col, "GFCI peak")
    p_fci = run_fci(pk, a.alpha); pf_ge, pf_ee = summarize(p_fci, gene_col, "FCI  peak")

    print("\n-- ELEMENT level --")
    e_gfci = run_gfci(emat, a.alpha); eg_ge, eg_ee = summarize(e_gfci, gene_col, "GFCI elem")
    e_fci = run_fci(emat, a.alpha); ef_ge, ef_ee = summarize(e_fci, gene_col, "FCI  elem")

    print("\n-- ELEMENTS ADJACENT TO GENE (interpretability) --")
    adj = set()
    for a_, b_, t in eg_ge + ef_ge:
        adj.add(a_ if a_ != gene_col else b_)
    for _, r in meta[meta.element.isin(adj)].iterrows():
        algos = []
        if any((r.element in (x[0], x[1])) for x in eg_ge): algos.append("gfci")
        if any((r.element in (x[0], x[1])) for x in ef_ge): algos.append("fci")
        print(f"  {r.element:26s} {r.chrom}:{r.start}-{r.end} state={r.state} marks=[{r.marks}] "
              f"span={r.span_kb}kb PC1ev={r.pc1_explained} algos={'|'.join(algos)}")
    if not adj:
        print("  (no element adjacent to the gene)")
    meta.to_csv(f"/tmp/claude-1000/-home-dgsetiawan-MachineLearning-CLeaR-epigenomics/6b29a59b-e4ec-4121-9b99-ac07c133d0d6/scratchpad/elements_{g}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
