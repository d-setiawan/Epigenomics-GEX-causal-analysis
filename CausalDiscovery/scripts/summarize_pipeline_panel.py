#!/usr/bin/env python3
"""Aggregate per-gene run_gene_pipeline outputs into one panel summary.

Reads <pipeline-root>/<GENE>/edges_consensus.tsv (+ pipeline_manifest.json) for
each requested gene and writes PANEL_SUMMARY.md + panel_peak_gene_edges.tsv.

Usage:
  python CausalDiscovery/scripts/summarize_pipeline_panel.py --genes CD14,CSF1R,IL1B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AXIS = {"re2g": "prediction", "pchic": "physical", "eqtl": "genetics"}


def s(val) -> str:
    return val if isinstance(val, str) and val else "—"


def axis_coverage(gene: str, pipeline_root: Path) -> dict:
    """Target-aware node-level axis coverage from the fci graph's support matches."""
    mpath = pipeline_root / gene / "fci" / "node_support_matches.tsv"
    if not mpath.exists():
        return dict(gene=gene, nodes=0, prediction=0, physical=0, genetics=0, ge2_axes=0, all3=0)
    m = pd.read_csv(mpath, sep="\t")
    m = m[m["source"].isin(AXIS) & (m["external_gene"].astype(str).str.upper() == gene)]
    node_axes: dict[str, set] = {}
    for _, r in m.iterrows():
        node_axes.setdefault(r["node_name"], set()).add(AXIS[r["source"]])
    vals = list(node_axes.values())
    return dict(
        gene=gene, nodes=len(vals),
        prediction=sum("prediction" in v for v in vals),
        physical=sum("physical" in v for v in vals),
        genetics=sum("genetics" in v for v in vals),
        ge2_axes=sum(len(v) >= 2 for v in vals),
        all3=sum(len(v) >= 3 for v in vals),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline-root", type=Path,
                   default=repo_root / "CausalDiscovery/outputs/pipeline")
    p.add_argument("--genes", default="CD14,CSF1R,IL1B")
    a = p.parse_args()
    genes = [g.strip().upper() for g in a.genes.split(",") if g.strip()]

    rows, peakgene, axis_rows = [], [], []
    for g in genes:
        cpath = a.pipeline_root / g / "edges_consensus.tsv"
        if not cpath.exists():
            print(f"[warn] missing {cpath}; skipping {g}")
            continue
        c = pd.read_csv(cpath, sep="\t")
        man = json.loads((a.pipeline_root / g / "pipeline_manifest.json").read_text())
        pg = c[c.edge_kind == "peak_gene"]
        rows.append(dict(
            gene=g, methods="+".join(man["methods"]),
            adjacencies=len(c),
            peak_peak=int((c.edge_kind == "peak_peak").sum()),
            peak_gene=len(pg),
            pg_multi=int((pg["n_methods"] >= 2).sum()) if not pg.empty else 0,
            pg_validated=int(pg["target_aware_validated"].sum()) if not pg.empty else 0,
        ))
        for _, r in pg.iterrows():
            peak = r["node1"] if not r["node1"].startswith("expr__") else r["node2"]
            peakgene.append(dict(gene=g, peak=peak, verdict=r["verdict"],
                                 algos=r["methods"], axes=s(r["axes"]),
                                 sources=s(r["target_aware_sources"])))
        axis_rows.append(axis_coverage(g, a.pipeline_root))

    summ = pd.DataFrame(rows)
    pgdf = pd.DataFrame(peakgene)
    axdf = pd.DataFrame(axis_rows)
    out_md = a.pipeline_root / "PANEL_SUMMARY.md"
    out_tsv = a.pipeline_root / "panel_peak_gene_edges.tsv"
    pgdf.to_csv(out_tsv, sep="\t", index=False)
    with open(out_md, "w") as f:
        f.write("# Pipeline panel summary — monocyte targets\n\n")
        f.write("Edge-validation suite (target-aware): **ENCODE-rE2G** (prediction) + "
                "**BLUEPRINT PCHi-C** (physical). DAGMA (Gaussian) is a sensitivity check.\n\n")
        f.write(summ.to_markdown(index=False) + "\n\n")
        f.write("`pg_*` = peak→gene. `pg_multi` = found by ≥2 algorithms. "
                "`pg_validated` = target-aware supported.\n\n")
        f.write("## 3-axis node coverage (target-gene-aware support)\n\n")
        f.write("How many discovered nodes each validation axis supports FOR THAT GENE "
                "(prediction=rE2G, physical=PCHi-C, genetics=eQTL). `ge2_axes`/`all3` = nodes "
                "corroborated by ≥2 / all 3 axes.\n\n")
        f.write(axdf.to_markdown(index=False) + "\n\n")
        f.write("## peak→gene edges (the interpretable ones)\n\n")
        f.write((pgdf.to_markdown(index=False) if not pgdf.empty
                 else "_none discovered_") + "\n\n")
        f.write("peak-peak edges have **no target-aware validator yet** "
                "(Phase B: ChIP-Atlas / Hi-C). Either-endpoint DB overlap is "
                "node-existence, not edge support.\n")
    print(f"wrote {out_md}\nwrote {out_tsv}")
    print("\n" + summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
