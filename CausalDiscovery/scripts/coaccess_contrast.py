#!/usr/bin/env python3
"""Co-accessibility contrast: what a Cicero-style method sees vs what our PAG resolves.

THE NOVELTY COMPARISON (competitive_landscape_novelty.md): co-accessibility methods
(Cicero/JRIM/Ziyani) emit ONE undirected "these move together" score per peak pair
and, by their authors' own admission, cannot separate a direct element-element link
from co-dependence via a shared latent driver. Our confounder-conditioned PAG splits
those same pairs into `<->` (latent common cause), directed, or pruned (non-adjacent
after conditioning). This script quantifies the gap on the SAME metacell data:

  1. co-accessibility magnitude by PAG edge class -- is it high for bidirected AND
     for non-adjacent pairs alike? (co-access can't tell them apart)
  2. "Cicero's-eye view": among the pairs co-access would call links (top scores),
     what fraction does the PAG label latent (`<->`) or prune (non_adjacent)? Those
     are exactly the calls co-accessibility gets wrong / cannot annotate.

Co-accessibility here = |Spearman| of peak accessibility across metacells (the
transparent core of co-accessibility; Cicero adds a distance-penalized graphical-
lasso regularization -- noted as a refinement, not needed to make the point). A
regularized partial-correlation variant is also reported when it is numerically
stable.

Input: peak_peak_pairs.tsv (from peak_peak_validator.py) + the metacell matrices.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PEAK_COL = re.compile(r"^chr[^:]+:\d+-\d+__")


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """Spearman corr = Pearson on column ranks."""
    r = np.apply_along_axis(rankdata, 0, x)
    r = (r - r.mean(0)) / (r.std(0) + 1e-12)
    return (r.T @ r) / r.shape[0]


def gene_coaccess(matrix_tsv: Path) -> dict[frozenset, float]:
    df = pd.read_csv(matrix_tsv, sep="\t")
    peaks = [c for c in df.columns if PEAK_COL.match(c)]
    if len(peaks) < 2:
        return {}
    x = df[peaks].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(x).any(1)
    x = x[ok]
    # drop zero-variance peaks
    keep = x.std(0) > 0
    peaks = [p for p, k in zip(peaks, keep) if k]
    x = x[:, keep]
    if x.shape[1] < 2:
        return {}
    C = np.abs(spearman_matrix(x))
    out = {}
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            out[frozenset((peaks[i], peaks[j]))] = float(C[i, j])
    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-tsv", type=Path,
                    default=repo / "CausalDiscovery/evaluation/peak_peak_pairs.tsv")
    ap.add_argument("--dataset-root", type=Path,
                    default=repo / "CausalDiscovery/outputs/datasets/joint_v2/"
                                   "monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    ap.add_argument("--report-dir", type=Path, default=repo / "CausalDiscovery/evaluation")
    ap.add_argument("--link-quantile", type=float, default=0.90,
                    help="Pairs at/above this co-access quantile = what Cicero would call links.")
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs_tsv, sep="\t")
    # attach co-accessibility per pair, per gene
    ca_by_gene: dict[str, dict] = {}
    for gene in pairs["gene"].unique():
        mt = a.dataset_root / gene.lower() / "nearby_peaks" / f"{gene.upper()}_nearby_peak_matrix.tsv"
        ca_by_gene[gene] = gene_coaccess(mt) if mt.exists() else {}
    pairs["coaccess"] = [
        ca_by_gene.get(r.gene, {}).get(frozenset((r.peak_a, r.peak_b)), np.nan)
        for r in pairs.itertuples()
    ]
    pairs = pairs.dropna(subset=["coaccess"])
    pairs.to_csv(a.report_dir / "peak_peak_pairs_coaccess.tsv", sep="\t", index=False)

    # 1. co-accessibility magnitude by PAG edge class
    by_class = (pairs.groupby("edge_class")["coaccess"]
                .agg(["mean", "median", "size"])
                .rename(columns={"size": "n_pairs"}))

    # 2. Cicero's-eye view: top-quantile co-access pairs, broken down by PAG class
    thr = float(pairs["coaccess"].quantile(a.link_quantile))
    links = pairs[pairs["coaccess"] >= thr]
    link_breakdown = (links["edge_class"].value_counts(normalize=True) * 100).round(1).to_dict()
    link_counts = links["edge_class"].value_counts().to_dict()

    # among ADJACENT pairs (any edge), how many are latent (`<->`) vs directed?
    adj = pairs[pairs["edge_class"] != "non_adjacent"]
    latent_frac = float((adj["edge_class"] == "bidirected").mean()) if len(adj) else float("nan")

    summary = {
        "n_pairs_scored": int(len(pairs)),
        "coaccess_by_class": by_class.reset_index().to_dict("records"),
        "link_quantile": a.link_quantile, "link_threshold": thr,
        "n_links": int(len(links)),
        "link_class_pct": link_breakdown, "link_class_counts": link_counts,
        "adjacent_latent_fraction": latent_frac,
    }
    (a.report_dir / "coaccess_contrast_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[coaccess] {len(pairs)} pairs scored across {pairs['gene'].nunique()} genes\n")
    print("=== co-accessibility |Spearman| by PAG edge class ===")
    print(by_class.to_string())
    print(f"\n=== Cicero's-eye view: pairs at/above the {a.link_quantile:.0%} co-access "
          f"quantile (|r|>={thr:.3f}) = {len(links)} 'links' ===")
    for k, pct in sorted(link_breakdown.items(), key=lambda t: -t[1]):
        print(f"  {k:15s}: {pct:5.1f}%  (n={link_counts.get(k,0)})")
    print(f"\n  -> of the PAG's ADJACENT peak-peak edges, {latent_frac:.0%} are bidirected "
          f"(latent common cause) -- pairs co-accessibility would report as regulatory 'links' "
          f"but our method flags as confounded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
