#!/usr/bin/env python3
"""EXPLORATORY PROTOTYPE (not production) — latent-factor causal modeling.

Successor to prototype_element_aggregation.py. Instead of collapsing co-located marks into ONE
PC1 activity (lossy), treat the mark-peaks as MULTIPLE INDICATORS of a small number of latent
"regulatory factors" (factor analysis), then discover factor -> gene structure. This makes the
shared latent causes (that produce the peak<->peak confounding) into explicit variables.

Pipeline per gene:
  1. z-score peaks; choose K = #(eigenvalues>1) of the peak corr matrix (Kaiser), capped;
  2. FactorAnalysis(K) on peaks -> factor scores (n x K) + loadings (peaks x K);
  3. interpret each factor by top-loading peaks (marks, coords, chromatin state, validated?);
  4. FCI + GFCI on {factors + gene expression} -> factor->gene structure;
  5. GIN (causal-learn) on peaks as a fully data-driven latent-discovery cross-check;
  6. report gene-variance explained by the factors.

Run in the causaldisco env with JAVA_HOME set (for GFCI).
"""
from __future__ import annotations
import argparse, re, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

ACT = {"H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3"}


def parse_peak(c):
    m = re.match(r"(chr[^:]+):(\d+)-(\d+)__(.+)", str(c))
    return None if not m else dict(col=c, chrom=m.group(1), start=int(m.group(2)), end=int(m.group(3)), mark=m.group(4))


def rank_gaussian(x):
    n = len(x); r = pd.Series(x).rank(method="average").to_numpy()
    return norm.ppf((r - 0.5) / n)


def load_validated_intervals(gene, repo="."):
    """Union of validated (target-aware) intervals for this gene across rE2G/eQTL/PCHi-C."""
    iv = []
    srcs = [("Data/raw/external/re2g/re2g_cd14_monocyte_gene_links.tsv", "re2g"),
            ("Data/raw/external/eqtl/eqtl_mon_gene_links.tsv", "eqtl"),
            ("Data/raw/external/pchic/pchic_mon_gene_links.tsv", "pchic")]
    import os
    for path, tag in srcs:
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, sep="\t")
        gcol = "gene" if "gene" in d.columns else None
        if gcol is None:
            continue
        sub = d[d[gcol].astype(str).str.upper() == gene.upper()]
        for _, r in sub.iterrows():
            try:
                iv.append((str(r["chrom"]), int(r["start"]), int(r["end"]), tag))
            except Exception:
                pass
    return iv


def is_validated(peak, intervals):
    for chrom, s, e, tag in intervals:
        if peak["chrom"] == chrom and peak["start"] <= e and s <= peak["end"]:
            return tag
    return ""


def chromstate(marks):
    marks = set(marks)
    if "H3K4me3" in marks:
        return "promoter"
    if "H3K27ac" in marks:
        return "active_enh"
    if marks & {"H3K4me1", "H3K4me2"}:
        return "poised_enh"
    if "H3K27me3" in marks:
        return "polycomb"
    if "H3K9me3" in marks:
        return "hetchrom"
    return "other"


def run_fci(mat, alpha=0.05):
    from causallearn.search.ConstraintBased.FCI import fci
    X = np.column_stack([rank_gaussian(mat[c].to_numpy(float)) for c in mat.columns])
    g, edges = fci(X, alpha=alpha, verbose=False, show_progress=False)
    names = list(mat.columns); out = []
    for e in edges:
        a = names[g.node_map[e.node1]]; b = names[g.node_map[e.node2]]; s = str(e)
        t = "<->" if "<->" in s else ("-->" if "-->" in s else ("o->" if "o->" in s else "o-o"))
        out.append((a, b, t))
    return out


def run_gfci(mat, alpha=0.05, pd_=2):
    from pytetrad.tools import TetradSearch as ts
    X = pd.DataFrame({c: rank_gaussian(mat[c].to_numpy(float)) for c in mat.columns})
    s = ts.TetradSearch(X); s.use_sem_bic(penalty_discount=pd_); s.use_fisher_z(alpha=alpha); s.run_gfci()
    out = []
    for line in str(s.get_string()).splitlines():
        line = line.strip()
        if ". " in line and any(x in line for x in ["-->", "o->", "<->", "o-o"]):
            body = line.split(". ", 1)[1]
            for sep in ["-->", "o->", "<->", "o-o"]:
                if sep in body:
                    a, b = [t.strip() for t in body.split(sep)]; out.append((a, b, sep)); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--dataset-root", default="CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    g = a.gene
    df = pd.read_csv(f"{a.dataset_root}/{g.lower()}/nearby_peaks/{g.upper()}_nearby_peak_matrix.tsv", sep="\t")
    gene_col = f"expr__{g}_log1p"
    peaks0 = [p for p in (parse_peak(c) for c in df.columns if str(c).startswith("chr")) if p]
    sub = df[[gene_col] + [p["col"] for p in peaks0]].dropna()
    # drop constant columns (would make z-score / corr undefined)
    peaks = [p for p in peaks0 if sub[p["col"]].std() > 1e-9]
    pcols = [p["col"] for p in peaks]
    P = sub[pcols].to_numpy(float)
    Pz = (P - P.mean(0)) / P.std(0)
    intervals = load_validated_intervals(g)
    print(f"\n===== {g} =====  metacells={len(sub)}  peaks={len(pcols)}  validated_intervals={len(intervals)}")

    # K by Kaiser (eigenvalues>1 of the correlation matrix), via stable SVD (Pz is z-scored so
    # singular_value^2/(n-1) are the correlation-matrix eigenvalues).
    s_all = np.linalg.svd(Pz, compute_uv=False)
    ev = s_all ** 2 / (len(Pz) - 1)
    K = int(max(2, min(a.kmax, int((ev > 1).sum()), len(pcols) // 3)))
    fa = FactorAnalysis(n_components=K, random_state=0).fit(Pz)
    scores = fa.transform(Pz)                    # n x K
    load = fa.components_.T                       # peaks x K
    var_exp = (load ** 2).sum(0) / (load ** 2).sum()
    fac = {}
    print(f"K={K} latent factors (Kaiser eig>1={int((ev>1).sum())}); factor var share: {np.round(var_exp,2)}")
    for k in range(K):
        # orient so top-loading peak is positive
        top = np.argsort(-np.abs(load[:, k]))[:5]
        if load[top[0], k] < 0:
            scores[:, k] *= -1; load[:, k] *= -1
        marks = [peaks[i]["mark"] for i in top]
        name = f"F{k}_{chromstate(marks)}"
        fac[name] = scores[:, k]
        tops = []
        for i in top[:4]:
            v = is_validated(peaks[i], intervals)
            tops.append(f"{peaks[i]['mark']}@{peaks[i]['start']}({load[i,k]:+.2f}){'*'+v if v else ''}")
        print(f"  {name:16s} top: " + "  ".join(tops))

    # gene variance explained by the factors
    r2 = LinearRegression().fit(scores, sub[gene_col].to_numpy(float)).score(scores, sub[gene_col].to_numpy(float))
    print(f"gene expression R^2 explained by the {K} factors: {r2:.2f}")

    fmat = pd.DataFrame({**fac, gene_col: sub[gene_col].to_numpy(float)})
    for label, runner in [("GFCI", run_gfci), ("FCI ", run_fci)]:
        try:
            edges = runner(fmat, a.alpha)
        except Exception as ex:
            print(f"  [{label} factor] FAILED: {type(ex).__name__}"); continue
        fg = [e for e in edges if gene_col in (e[0], e[1])]
        ff = [e for e in edges if gene_col not in (e[0], e[1])]
        print(f"  [{label} factor] edges={len(edges)}  factor->gene={len(fg)}  factor-factor={len(ff)}")
        for aa, bb, tt in fg:
            fname = aa if aa != gene_col else bb
            print(f"      {fname} {tt} GENE")

    # GIN cross-check (data-driven latent discovery)
    try:
        from causallearn.search.HiddenCausal.GIN.GIN import GIN
        Gg, order = GIN(Pz)
        print(f"  [GIN peaks] latent causal order length={len(order) if order else 0}")
    except Exception as ex:
        print(f"  [GIN peaks] n/a ({type(ex).__name__})")


if __name__ == "__main__":
    main()
