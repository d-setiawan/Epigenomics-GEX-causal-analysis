#!/usr/bin/env python3
"""Compare two pipeline panel runs edge-for-edge.

Built for A/B questions where the whole panel is re-run under one changed setting —
a different CI test, a different aggregation — and the question is not "how many edges"
but **do the conclusions move**: are the same peak->gene edges found, do they stay
directed, do the validated ones survive?

    python Evaluation/scripts/compare_panels.py \
        --a CausalDiscovery/outputs/pipeline_confounder_panel --label-a fisherz \
        --b CausalDiscovery/outputs/pipeline_rcit           --label-b rcit

Reports per-gene counts, panel totals, and the Jaccard overlap of the peak->gene edge
sets — plus which edges are unique to each arm, since a validated edge that appears in
only one arm is the thing you actually need to look at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def genes_in(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "edges_consensus.tsv").exists())


def _is_directed(orient: str) -> bool:
    s = str(orient)
    return (("-->" in s) or ("<--" in s)) and not any(
        b in s for b in ("o-", "-o", "<->", "---"))


def load(root: Path) -> pd.DataFrame:
    rows = []
    for g in genes_in(root):
        df = pd.read_csv(root / g / "edges_consensus.tsv", sep="\t")
        df["gene"] = g
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["edge"] = df["gene"] + "|" + df["node1"].astype(str) + "|" + df["node2"].astype(str)
    df["directed"] = df["orientations"].map(_is_directed)
    return df


def summarize(df: pd.DataFrame, label: str) -> dict:
    pg = df[df["edge_kind"] == "peak_gene"]
    pp = df[df["edge_kind"] == "peak_peak"]
    val = pg[pg["verdict"].astype(str).str.startswith("VALIDATED")]
    return {
        "arm": label, "genes": df["gene"].nunique(),
        "peak_gene": len(pg), "pg_directed": int(pg["directed"].sum()),
        "pg_validated": len(val),
        "peak_peak": len(pp),
        "pp_bidirected": int(pp["orientations"].astype(str).str.contains("<->").sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-json", type=Path, default=None)
    x = ap.parse_args()

    A, B = load(x.a), load(x.b)
    if A.empty or B.empty:
        raise SystemExit(f"empty panel: {x.a if A.empty else x.b}")

    common = sorted(set(A["gene"]) & set(B["gene"]))
    A, B = A[A["gene"].isin(common)], B[B["gene"].isin(common)]
    sa, sb = summarize(A, x.label_a), summarize(B, x.label_b)

    print(f"\ncomparing {x.label_a} vs {x.label_b} over {len(common)} shared genes\n")
    print(pd.DataFrame([sa, sb]).to_string(index=False))

    ea = set(A.loc[A["edge_kind"] == "peak_gene", "edge"])
    eb = set(B.loc[B["edge_kind"] == "peak_gene", "edge"])
    inter, union = ea & eb, ea | eb
    jac = len(inter) / len(union) if union else float("nan")
    print(f"\npeak->gene edge sets:  shared {len(inter)}  "
          f"only-{x.label_a} {len(ea - eb)}  only-{x.label_b} {len(eb - ea)}  "
          f"Jaccard {jac:.3f}")

    va = set(A.loc[(A["edge_kind"] == "peak_gene")
                   & A["verdict"].astype(str).str.startswith("VALIDATED"), "edge"])
    vb = set(B.loc[(B["edge_kind"] == "peak_gene")
                   & B["verdict"].astype(str).str.startswith("VALIDATED"), "edge"])
    print(f"validated peak->gene:  shared {len(va & vb)}  "
          f"only-{x.label_a} {len(va - vb)}  only-{x.label_b} {len(vb - va)}")
    for lbl, s in ((x.label_a, va - vb), (x.label_b, vb - va)):
        for e in sorted(s):
            print(f"    only in {lbl}: {e}")

    print("\nper-gene peak->gene (found / directed / validated):")
    rows = []
    for g in common:
        a_, b_ = A[A.gene == g], B[B.gene == g]
        ag, bg = a_[a_.edge_kind == "peak_gene"], b_[b_.edge_kind == "peak_gene"]
        rows.append({
            "gene": g,
            x.label_a: f"{len(ag)}/{int(ag.directed.sum())}/"
                       f"{int(ag.verdict.astype(str).str.startswith('VALIDATED').sum())}",
            x.label_b: f"{len(bg)}/{int(bg.directed.sum())}/"
                       f"{int(bg.verdict.astype(str).str.startswith('VALIDATED').sum())}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    out = {"shared_genes": common, "summary": [sa, sb],
           "pg_jaccard": jac, "pg_shared": len(inter),
           f"pg_only_{x.label_a}": sorted(ea - eb), f"pg_only_{x.label_b}": sorted(eb - ea),
           "validated_shared": len(va & vb),
           f"validated_only_{x.label_a}": sorted(va - vb),
           f"validated_only_{x.label_b}": sorted(vb - va)}
    if x.out_json:
        x.out_json.parent.mkdir(parents=True, exist_ok=True)
        x.out_json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {x.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
