#!/usr/bin/env python3
"""Joint preprocessing for Jianle-track integration with RNA + all histone marks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import pandas as pd
import scanpy as sc

from preprocess_pilot_jianle import (
    build_chrom_adata,
    infer_repo_root,
    pick_rna_h5,
    preprocess_chrom_for_jianle,
    preprocess_rna,
    resolve_path,
)


def parse_marks(manifest_df: pd.DataFrame, marks_csv: str | None) -> List[str]:
    all_marks = list(dict.fromkeys(manifest_df["mark"].astype(str).tolist()))
    if not marks_csv:
        return all_marks
    wanted = [m.strip() for m in marks_csv.split(",") if m.strip()]
    missing = [m for m in wanted if m not in set(all_marks)]
    if missing:
        raise ValueError(f"Marks missing from manifest: {missing}")
    return wanted


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Joint preprocessing for Jianle-track integration")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--rna-h5", default=None)
    p.add_argument("--marks", default=None, help="Comma-separated subset of marks (default: all)")
    p.add_argument("--run-id", default="joint")
    p.add_argument("--out-dir", default=None, help="Default: integration/outputs/jianle/joint/<RUN_ID>/preprocess")

    p.add_argument("--rna-min-genes", type=int, default=200)
    p.add_argument("--rna-min-cells", type=int, default=3)
    p.add_argument("--rna-top-genes", type=int, default=3000)
    p.add_argument("--rna-target-sum", type=float, default=1e4)
    p.add_argument("--rna-scale-max", type=float, default=10.0)
    p.add_argument("--rna-n-pcs", type=int, default=50)

    p.add_argument("--chrom-top-features", type=int, default=30000)
    p.add_argument("--chrom-n-lsi", type=int, default=50)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_path(repo_root, args.manifest)
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "mark" not in manifest_df.columns:
        raise ValueError(f"Manifest missing 'mark' column: {manifest_path}")
    marks = parse_marks(manifest_df, args.marks)

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root / f"integration/outputs/jianle/joint/{args.run_id}/preprocess"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rna_h5_path = pick_rna_h5(repo_root, args.rna_h5)
    rna = sc.read_10x_h5(rna_h5_path)
    rna = preprocess_rna(rna, args)
    rna.obs["modality_key"] = "rna"
    rna.obs["mark"] = "RNA"

    rna_out = out_dir / "rna_preprocessed.h5ad"
    rna.write_h5ad(rna_out, compression="gzip")

    chrom_rows: list[dict] = []
    for mark in marks:
        row = manifest_df.loc[manifest_df["mark"] == mark].iloc[0].to_dict()
        chrom, clean_match = build_chrom_adata(repo_root, row, mark)
        chrom = preprocess_chrom_for_jianle(chrom, args)

        # Namespace feature ids by mark for downstream clarity.
        chrom.var["orig_feature"] = chrom.var_names.astype(str)
        chrom.var_names = pd.Index([f"{mark}::{x}" for x in chrom.var["orig_feature"].astype(str)])
        chrom.var_names_make_unique()
        chrom.obs["modality_key"] = f"chrom_{mark}"
        chrom.obs["mark"] = mark

        chrom_out = out_dir / f"chrom_{mark}_preprocessed.h5ad"
        chrom.write_h5ad(chrom_out, compression="gzip")

        chrom_rows.append(
            {
                "mark": mark,
                "modality_key": f"chrom_{mark}",
                "chrom_h5ad": str(chrom_out),
                "n_cells": int(chrom.n_obs),
                "n_features": int(chrom.n_vars),
                "n_lsi": int(chrom.obsm["X_lsi"].shape[1]) if "X_lsi" in chrom.obsm else 0,
                "clean_metadata_match_fraction": float(clean_match),
            }
        )

    chrom_manifest_tsv = out_dir / "chrom_preprocessed_manifest.tsv"
    write_tsv(
        chrom_manifest_tsv,
        chrom_rows,
        ["mark", "modality_key", "chrom_h5ad", "n_cells", "n_features", "n_lsi", "clean_metadata_match_fraction"],
    )

    summary = {
        "run_id": args.run_id,
        "manifest": str(manifest_path),
        "rna_h5": str(rna_h5_path),
        "marks": marks,
        "outputs": {
            "rna_h5ad": str(rna_out),
            "chrom_manifest_tsv": str(chrom_manifest_tsv),
            "out_dir": str(out_dir),
        },
        "rna": {
            "n_cells": int(rna.n_obs),
            "n_genes": int(rna.n_vars),
            "n_hvg": int(rna.var["highly_variable"].sum()) if "highly_variable" in rna.var else 0,
            "n_pcs": int(rna.obsm["X_pca"].shape[1]) if "X_pca" in rna.obsm else 0,
        },
        "chrom": chrom_rows,
        "params": vars(args),
    }
    summary_path = out_dir / "joint_preprocess_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {rna_out}")
    print(f"Wrote: {chrom_manifest_tsv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
