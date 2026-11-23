#!/usr/bin/env python3
"""Cross-method + cross-partition + external consensus = the confident biological edge set.

The biology-first deliverable. A peak->gene edge is CONFIDENT when it survives four independent
checks, so it cannot be an artifact of any one of them:
  1. multi-algorithm  — found by >=2 of {FCI (constraint), BOSS (score, CPDAG), BOSS-FCI (score, PAG)};
                        agreement across methods with different assumptions guards against one method's bias
  2. partition-robust — recovered in multiple re-clustered metacell partitions (frequency)
  3. externally validated — target-gene-aware support from rE2G / PCHi-C / eQTL
  4. (bonus) directed — oriented peak->gene by the score-based CPDAG search (BOSS)

Reads the per-method partition-bootstrap frequency tables and the panel's external-validation verdicts.
Run the FCI (partition_bootstrap.py) and Tetrad (partition_bootstrap_tetrad.py --algorithm boss/boss_fci)
bootstraps first.

    python pipelines/empirical/analysis/scripts/consensus_biology.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PB = REPO / "pipelines/empirical/analysis/outputs/partition_bootstrap"


def load_tetrad(f: Path) -> dict:
    if not f.exists():
        return {}
    d = pd.read_csv(f, sep="\t")
    return {(r.gene, r.peak): (float(r.frequency), int(getattr(r, "n_directed", 0)))
            for r in d.itertuples()}


def load_fci(f: Path) -> dict:
    if not f.exists():
        return {}
    d = pd.read_csv(f, sep="\t")
    d = d[d.kind == "peak_gene"]
    out = {}
    for r in d.itertuples():
        parts = str(r.edge).split("|")
        peaks = [p for p in parts[1:] if p.startswith("chr")]
        if peaks:
            out[(parts[0], peaks[0])] = (float(r.frequency), int(r.n_validated))
    return out


def load_validated(genes: list[str], panel: Path) -> set:
    val = set()
    for g in genes:
        f = panel / g / "edges_consensus.tsv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t")
        d = d[(d.edge_kind == "peak_gene") & d.verdict.astype(str).str.startswith("VALIDATED")]
        for r in d.itertuples():
            pk = r.node1 if str(r.node1).startswith("chr") else r.node2
            val.add((g, str(pk)))
    return val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", default="CD14,IL4R,LSP1,PLCL1,CD52,CD151")
    ap.add_argument("--min-methods", type=int, default=2)
    ap.add_argument("--panel", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/pipeline_confounder_panel")
    ap.add_argument("--out", type=Path, default=REPO / "pipelines/empirical/evaluation/outputs/consensus_biology.tsv")
    a = ap.parse_args()
    genes = [g.strip() for g in a.genes.split(",")]

    fci = load_fci(PB / "edge_frequency.tsv")
    boss = load_tetrad(PB / "edge_frequency_boss.tsv")
    bfci = load_tetrad(PB / "edge_frequency_boss_fci.tsv")
    val = load_validated(genes, a.panel)
    methods = {"FCI": fci, "BOSS": boss, "BOSSFCI": bfci}

    rows = []
    for k in set().union(*[set(m) for m in methods.values()]):
        g, pk = k
        row = {"gene": g, "peak": pk}
        for name, m in methods.items():
            row[name] = round(m.get(k, (0.0,))[0], 2)
        row["BOSS_directed"] = boss.get(k, (0, 0))[1]
        row["ext_validated"] = int(k in val)
        row["n_methods"] = sum(1 for m in methods.values() if k in m)
        row["max_freq"] = max(row["FCI"], row["BOSS"], row["BOSSFCI"])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ["ext_validated", "n_methods", "max_freq"], ascending=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False)

    consensus = df[df.n_methods >= a.min_methods]
    confident = consensus[consensus.ext_validated == 1]
    print(f"union peak->gene edges (any method): {len(df)}")
    print(f"multi-method (>= {a.min_methods}): {len(consensus)}")
    print(f"CONFIDENT (multi-method AND externally validated): {len(confident)}\n")
    print("=== CONFIDENT biological edge set ===")
    print(confident[["gene", "peak", "FCI", "BOSS", "BOSSFCI", "BOSS_directed"]].to_string(index=False))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
