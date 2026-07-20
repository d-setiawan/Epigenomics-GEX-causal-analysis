#!/usr/bin/env python3
"""Null-calibration test for an aggregation scheme.

The question aggregation must answer: does the discovery step's false-positive
rate match its nominal alpha? If rows are dependent (overlapping kNN-averaging),
the CI test sees more "information" than exists, so even a NULL matrix (all
columns mutually independent) yields many edges. This tool measures that.

Procedure, on a given nearby-peak matrix + graph config:
  * REAL run  -> edges the pipeline finds on the true matrix (signal + noise).
  * NULL runs -> independently permute EVERY feature column across rows
                 (destroys all cross-column structure, preserves each marginal),
                 re-run the SAME graph command, count surviving edges. Under a
                 correctly-calibrated method this should be ~= alpha * (#pairs).

A well-calibrated aggregation has a LOW null edge count. A high null count is
direct evidence the aggregation inflates type-I error and should be replaced.
Run the same tool on kNN-avg vs metacell matrices and compare the null floors.

Uses the exact `causal_cli.py graph` path (no re-implementation of discovery).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def run_graph(cli: Path, python_bin: str, matrix: Path, out_dir: Path,
              a: argparse.Namespace) -> dict:
    cmd = [
        sys.executable, str(cli), "--python-bin", python_bin, "graph",
        "--matrix-tsv", str(matrix), "--method", a.method,
        "--indep-test", a.indep_test, "--transform", a.transform,
        "--alpha", str(a.alpha), "--max-depth", str(a.max_depth),
        "--background-mode", a.background_mode,
        "--out-dir", str(out_dir), "--no-plot",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"graph failed on {matrix}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    summ = json.loads((out_dir / "run_summary.json").read_text())
    edges_path = out_dir / f"{a.method}_edges.tsv"
    n_peak_gene = 0
    if edges_path.exists():
        ed = pd.read_csv(edges_path, sep="\t")
        if not ed.empty:
            n_peak_gene = int(ed["edge_text"].astype(str).str.contains("expr__").sum())
    return {"n_edges": int(summ.get("n_edges", 0)),
            "n_vars": int(summ.get("n_columns_selected", 0)),
            "n_peak_gene": n_peak_gene}


def permute_matrix(df: pd.DataFrame, id_cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    feat_cols = [c for c in df.columns if c not in id_cols]
    n = len(out)
    for c in feat_cols:
        out[c] = out[c].to_numpy()[rng.permutation(n)]
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-tsv", type=Path, required=True)
    p.add_argument("--label", default="matrix", help="Name for this scheme in the report.")
    p.add_argument("--out-dir", type=Path,
                   default=repo_root / "Evaluation/outputs/null_calibration")
    p.add_argument("--n-permutations", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--python-bin",
                   default="/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python")
    # discovery config (match the pipeline defaults)
    p.add_argument("--method", default="fci")
    p.add_argument("--indep-test", default="fisherz")
    p.add_argument("--transform", default="rank_gaussian")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--background-mode", default="tiered_distal_promoter_expr")
    # id columns to NOT permute (kept intact so column-selection still works)
    p.add_argument("--id-cols", default="sample_id,cell_rna,anchor_rank,support_mean_nn_distance,metacell_id")
    a = p.parse_args()

    cli = repo_root / "CausalDiscovery" / "scripts" / "causal_cli.py"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.matrix_tsv, sep="\t")
    id_cols = [c for c in a.id_cols.split(",") if c in df.columns]

    print(f"[null] {a.label}: {len(df)} rows x {df.shape[1]} cols  (id cols kept: {id_cols})")
    real = run_graph(cli, a.python_bin, a.matrix_tsv, a.out_dir / "real", a)
    print(f"[null] REAL edges={real['n_edges']} (peak->gene={real['n_peak_gene']}) over {real['n_vars']} vars")

    rng = np.random.default_rng(a.seed)
    null_edges, null_pg = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(a.n_permutations):
            perm = permute_matrix(df, id_cols, rng)
            ppath = Path(tmp) / f"perm_{i}.tsv"
            perm.to_csv(ppath, sep="\t", index=False)
            res = run_graph(cli, a.python_bin, ppath, a.out_dir / "null_run", a)
            null_edges.append(res["n_edges"])
            null_pg.append(res["n_peak_gene"])
            print(f"  perm {i+1}/{a.n_permutations}: edges={res['n_edges']} peak->gene={res['n_peak_gene']}")

    ne = np.array(null_edges)
    npg = np.array(null_pg)
    n_vars = real["n_vars"]
    n_pairs = n_vars * (n_vars - 1) / 2 if n_vars else 1
    report = {
        "label": a.label,
        "matrix": str(a.matrix_tsv),
        "n_rows": len(df),
        "n_vars": n_vars,
        "config": {"method": a.method, "indep_test": a.indep_test,
                   "transform": a.transform, "alpha": a.alpha,
                   "max_depth": a.max_depth, "background_mode": a.background_mode},
        "real_edges": real["n_edges"],
        "real_peak_gene": real["n_peak_gene"],
        "n_permutations": a.n_permutations,
        "null_edges_mean": float(ne.mean()),
        "null_edges_median": float(np.median(ne)),
        "null_edges_p95": float(np.percentile(ne, 95)),
        "null_edges_max": int(ne.max()),
        "null_peak_gene_mean": float(npg.mean()),
        "null_peak_gene_rate": float((npg > 0).mean()),
        "null_false_edge_rate_per_pair": float(ne.mean() / n_pairs),
        "expected_false_edges_at_alpha": float(a.alpha * n_pairs),
        "null_edges": null_edges,
        "null_peak_gene": null_pg,
    }
    out = a.out_dir / f"null_calibration_{a.label}.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"NULL CALIBRATION — {a.label}")
    print(f"  rows={len(df)}  vars={n_vars}  pairs={int(n_pairs)}")
    print(f"  REAL edges:            {real['n_edges']}  (peak->gene {real['n_peak_gene']})")
    print(f"  NULL edges mean:       {ne.mean():.1f}  median {np.median(ne):.0f}  p95 {np.percentile(ne,95):.0f}  max {ne.max()}")
    print(f"  NULL false peak->gene: mean {npg.mean():.2f}/run, in {(npg>0).mean()*100:.0f}% of runs")
    print(f"  Expected false edges @ alpha={a.alpha}: {a.alpha*n_pairs:.1f}")
    print(f"  -> null/expected ratio: {ne.mean()/(a.alpha*n_pairs):.1f}x  (1.0 = well calibrated)")
    print(f"  wrote {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
