#!/usr/bin/env python3
"""Bootstrap edge-stability selection.

Noise edges flicker across resamples; real edges recur. This subsamples the rows
of a nearby-peak matrix B times, re-runs the SAME discovery, and reports each
undirected adjacency's recurrence frequency + its endpoint-mark distribution.
Edges above a frequency threshold are the "stable" set.

To prove stability separates signal from noise, pass --permute-null: the tool
then ALSO bootstraps a column-permuted (null) copy of the matrix and reports how
many edges reach the threshold under pure noise. Stability selection works when
`stable edges (real) >> stable edges (null)`.

Uses the exact `causal_cli.py graph` path (no re-implementation of discovery).
Outputs edge_stability.tsv + stable_edges.tsv (fci_edges-format, threshold-filtered
so null_calibration.py / support can consume it) + a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def run_graph_edges(cli: Path, python_bin: str, matrix: Path, out_dir: Path,
                    a: argparse.Namespace) -> pd.DataFrame:
    cmd = [
        sys.executable, str(cli), "--python-bin", python_bin, "graph",
        "--matrix-tsv", str(matrix), "--method", a.method,
        "--indep-test", a.indep_test, "--transform", a.transform,
        "--alpha", str(a.alpha), "--max-depth", str(a.max_depth),
        "--background-mode", a.background_mode, "--confounders", getattr(a, "confounders", "none"),
        "--out-dir", str(out_dir), "--no-plot",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"graph failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    ep = out_dir / f"{a.method}_edges.tsv"
    if not ep.exists():
        return pd.DataFrame(columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"])
    return pd.read_csv(ep, sep="\t")


def bootstrap(cli, python_bin, df, id_cols, a, permute: bool, tag: str) -> dict:
    rng = np.random.default_rng(a.seed + (7919 if permute else 0))
    n = len(df)
    k = int(round(a.subsample_frac * n))
    freq: dict[tuple, int] = defaultdict(int)
    marks: dict[tuple, list] = defaultdict(list)
    kinds: dict[tuple, str] = {}
    feat_cols = [c for c in df.columns if c not in id_cols]
    with tempfile.TemporaryDirectory() as tmp:
        for b in range(a.n_boot):
            idx = rng.choice(n, size=k, replace=a.with_replacement)
            sub = df.iloc[idx].reset_index(drop=True)
            if permute:
                for c in feat_cols:
                    sub[c] = sub[c].to_numpy()[rng.permutation(len(sub))]
            ppath = Path(tmp) / f"{tag}_{b}.tsv"
            sub.to_csv(ppath, sep="\t", index=False)
            ed = run_graph_edges(cli, python_bin, ppath, Path(tmp) / "run", a)
            for _, r in ed.iterrows():
                key = tuple(sorted([str(r["node1"]), str(r["node2"])]))
                freq[key] += 1
                marks[key].append(str(r.get("edge_text", "")))
                n1, n2 = key
                kinds[key] = ("peak_gene" if (n1.startswith("expr__") ^ n2.startswith("expr__"))
                              else "gene_gene" if n1.startswith("expr__")
                              else "peak_peak")
            if (b + 1) % 10 == 0:
                print(f"  [{tag}] {b+1}/{a.n_boot} boots", flush=True)
    rows = []
    for key, c in freq.items():
        rows.append({"node1": key[0], "node2": key[1], "edge_kind": kinds[key],
                     "frequency": c / a.n_boot, "n_boot_present": c,
                     "example_orientation": marks[key][0] if marks[key] else ""})
    tbl = pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True) if rows \
        else pd.DataFrame(columns=["node1", "node2", "edge_kind", "frequency", "n_boot_present", "example_orientation"])
    return {"table": tbl}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-tsv", type=Path, required=True)
    p.add_argument("--label", default="matrix")
    p.add_argument("--out-dir", type=Path, default=repo_root / "CausalDiscovery/outputs/evaluation/bootstrap_stability")
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--subsample-frac", type=float, default=0.8)
    p.add_argument("--with-replacement", action="store_true", help="Bootstrap w/ replacement instead of subsample.")
    p.add_argument("--stable-threshold", type=float, default=0.5)
    p.add_argument("--core-threshold", type=float, default=0.85)
    p.add_argument("--permute-null", action="store_true", help="Also bootstrap a column-permuted null for comparison.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--python-bin", default="/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python")
    p.add_argument("--method", default="fci")
    p.add_argument("--indep-test", default="fisherz")
    p.add_argument("--transform", default="rank_gaussian")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--background-mode", default="tiered_distal_promoter_expr")
    p.add_argument("--confounders", default="none",
                   help="Observed confounders as root nodes, forwarded to causal_cli graph (pc/fci).")
    p.add_argument("--id-cols", default="sample_id,cell_rna,anchor_rank,support_mean_nn_distance,metacell_id")
    a = p.parse_args()

    cli = repo_root / "CausalDiscovery" / "scripts" / "causal_cli.py"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.matrix_tsv, sep="\t")
    id_cols = [c for c in a.id_cols.split(",") if c in df.columns]
    print(f"[stability] {a.label}: {len(df)} rows, subsample {a.subsample_frac} x {a.n_boot} boots")

    real = bootstrap(cli, a.python_bin, df, id_cols, a, permute=False, tag="real")
    tbl = real["table"]
    tbl.to_csv(a.out_dir / f"edge_stability_{a.label}.tsv", sep="\t", index=False)
    stable = tbl[tbl["frequency"] >= a.stable_threshold]
    core = tbl[tbl["frequency"] >= a.core_threshold]
    # emit stable set in fci_edges format so downstream tools can consume it
    se = stable.rename(columns={"example_orientation": "edge_text"})
    se["endpoint1"] = ""; se["endpoint2"] = ""
    se[["node1", "endpoint1", "node2", "endpoint2", "edge_text"]].to_csv(
        a.out_dir / f"stable_edges_{a.label}.tsv", sep="\t", index=False)

    summary = {
        "label": a.label, "matrix": str(a.matrix_tsv), "n_rows": len(df),
        "n_boot": a.n_boot, "subsample_frac": a.subsample_frac,
        "stable_threshold": a.stable_threshold, "core_threshold": a.core_threshold,
        "n_edges_seen": len(tbl),
        "n_stable": int(len(stable)), "n_core": int(len(core)),
        "n_stable_peak_gene": int((stable["edge_kind"] == "peak_gene").sum()) if not stable.empty else 0,
        "n_core_peak_gene": int((core["edge_kind"] == "peak_gene").sum()) if not core.empty else 0,
    }
    if a.permute_null:
        print("[stability] bootstrapping column-permuted NULL for comparison ...")
        null = bootstrap(cli, a.python_bin, df, id_cols, a, permute=True, tag="null")
        nt = null["table"]
        summary["null_n_stable"] = int((nt["frequency"] >= a.stable_threshold).sum()) if not nt.empty else 0
        summary["null_n_core"] = int((nt["frequency"] >= a.core_threshold).sum()) if not nt.empty else 0
        summary["null_max_frequency"] = float(nt["frequency"].max()) if not nt.empty else 0.0

    (a.out_dir / f"stability_summary_{a.label}.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"STABILITY — {a.label}")
    print(f"  edges ever seen: {summary['n_edges_seen']}")
    print(f"  stable (>= {a.stable_threshold}): {summary['n_stable']}  (peak->gene {summary['n_stable_peak_gene']})")
    print(f"  core   (>= {a.core_threshold}): {summary['n_core']}  (peak->gene {summary['n_core_peak_gene']})")
    if a.permute_null:
        print(f"  NULL stable: {summary['null_n_stable']}  core: {summary['null_n_core']}  max-freq {summary['null_max_frequency']:.2f}")
        print(f"  -> signal separation: real {summary['n_stable']} vs null {summary['null_n_stable']} stable edges")
    if not stable.empty:
        print("  top stable edges:")
        for _, r in stable.head(12).iterrows():
            print(f"    {r['frequency']:.2f}  {r['edge_kind']:9} {r['node1']} — {r['node2']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
