#!/usr/bin/env python3
"""EXPLORATORY PROTOTYPE (not production) — FUSED element + nuisance-regression model.

Synthesis of the two prior probes:
  * prototype_element_aggregation.py — merge co-located mark-peaks -> element PC1 activity
    (interpretable, keeps the gene link, but leaves residual element<->element confounding);
  * prototype_latent_factor.py — showed the dominant peak co-variation is gene-IRRELEVANT
    global confounding (factors explain gene R^2~0.05).

Fused idea: build ELEMENT nodes, then REGRESS OUT the top-k global factors (the dominant shared
modes = cell-state/donor confounders the FA surfaced) from every element AND the gene, then run
FCI/GFCI on the residualized elements + gene. Because the top factors don't predict the gene,
removing them should kill element<->element confounding WITHOUT destroying real gene->element
edges. THE TEST: do the externally-validated gene->element edges survive the nuisance removal?

Run in the causaldisco env with JAVA_HOME set.
"""
from __future__ import annotations
import argparse, os, sys, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype_element_aggregation import build_element_matrix, run_fci, run_gfci  # noqa: E402
from prototype_latent_factor import load_validated_intervals, is_validated, parse_peak  # noqa: E402


def summarize(edges, gene_col):
    fg = [e for e in edges if gene_col in (e[0], e[1])]
    ff = [e for e in edges if gene_col not in (e[0], e[1])]
    nbi = sum(1 for e in edges if e[2] == "<->")
    return dict(total=len(edges), gene=len(fg), ee=len(ff), bi=nbi, fg=fg)


def element_validated(member_peaks, intervals):
    tags = set()
    for c in str(member_peaks).split(";"):
        p = parse_peak(c)
        if p:
            t = is_validated(p, intervals)
            if t:
                tags.add(t)
    return ",".join(sorted(tags))


def residualize(mat, nuis):
    """Return mat with each column linearly regressed on the nuisance matrix (residuals)."""
    N = np.column_stack([np.ones(len(nuis)), nuis])
    beta, *_ = np.linalg.lstsq(N, mat.to_numpy(float), rcond=None)
    resid = mat.to_numpy(float) - N @ beta
    return pd.DataFrame(resid, columns=mat.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--dataset-root", default="pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    ap.add_argument("--k", type=int, default=2, help="# global nuisance PCs to regress out")
    ap.add_argument("--gap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    g = a.gene
    gene_col = f"expr__{g}_log1p"
    df = pd.read_csv(f"{a.dataset_root}/{g.lower()}/nearby_peaks/{g.upper()}_nearby_peak_matrix.tsv", sep="\t")
    peak_cols = [c for c in df.columns if str(c).startswith("chr")]
    base = pd.concat([df[[gene_col]], df[peak_cols]], axis=1).dropna()
    emat, meta = build_element_matrix(base, gene_col, gap=a.gap)
    intervals = load_validated_intervals(g)
    meta["validated"] = meta["member_peaks"].map(lambda mp: element_validated(mp, intervals))
    ecols = [c for c in emat.columns if c != gene_col]
    print(f"\n===== {g} =====  metacells={len(emat)}  elements={len(ecols)}  "
          f"validated elements={ (meta.validated!='').sum() }")

    # global nuisance = top-k PCs of the element-activity matrix
    E = emat[ecols].to_numpy(float); Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Ez - Ez.mean(0), full_matrices=False)
    k = min(a.k, len(ecols) - 1)
    nuis = U[:, :k] * S[:k]
    gene_v = emat[gene_col].to_numpy(float)
    varshare = (S[:k] ** 2 / (S ** 2).sum())
    corrs = [float(np.corrcoef(nuis[:, i], gene_v)[0, 1]) for i in range(k)]
    print(f"nuisance: top-{k} element PCs  varshare={np.round(varshare,2)}  corr(PC,gene)={np.round(corrs,2)} "
          f"(low corr => removing them should not hurt the gene signal)")

    emat_resid = residualize(emat, nuis)

    for label, mat in [("ELEMENT (no nuisance)", emat), ("FUSED  (nuisance-regressed)", emat_resid)]:
        print(f"\n-- {label} --")
        for algo, runner in [("GFCI", run_gfci), ("FCI ", run_fci)]:
            try:
                s = summarize(runner(mat, a.alpha), gene_col)
            except Exception as ex:
                print(f"  [{algo}] FAILED {type(ex).__name__}"); continue
            print(f"  [{algo}] edges={s['total']:2d}  elem->gene={s['gene']}  elem<->elem={s['ee']}  bidirected={s['bi']}")
            for aa, bb, tt in s["fg"]:
                en = aa if aa != gene_col else bb
                row = meta[meta.element == en]
                v = row.validated.iloc[0] if len(row) else ""
                st = row.state.iloc[0] if len(row) else "?"
                print(f"       {en} [{st}] {tt} GENE   {'VALIDATED:'+v if v else '(unvalidated)'}")


if __name__ == "__main__":
    main()
