#!/usr/bin/env python3
"""Export multiple locus-level pseudo-paired matrices from one-to-one scGLUE matches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def parse_feature_interval(feature: str) -> tuple[str, int, int]:
    raw = feature.split("::")[-1]
    chrom, rest = raw.split(":")
    start_str, end_str = rest.split("-")
    return chrom, int(start_str), int(end_str)


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def read_string_dataset(ds: h5py.Dataset) -> list[str]:
    out = ds[()]
    values = []
    for item in out:
        values.append(item.decode() if isinstance(item, bytes) else str(item))
    return values


def read_csr_group(group: h5py.Group, shape: tuple[int, int]) -> sparse.csr_matrix:
    data = group["data"][()]
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def mean_log1p_norm_for_features(
    x: sparse.csr_matrix,
    feature_idx: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    if feature_idx.size == 0:
        return np.full(x.shape[0], np.nan, dtype=float)
    row_sums = np.asarray(x.sum(axis=1)).ravel().astype(float)
    row_sums[row_sums <= 0] = 1.0
    sub = x[:, feature_idx]
    norm = sub.multiply(target_sum / row_sums[:, None])
    mean_vals = np.asarray(norm.mean(axis=1)).ravel()
    return np.log1p(mean_vals)


def log1p_norm_for_gene(
    x: sparse.csr_matrix,
    gene_to_idx: dict[str, int],
    gene: str,
    target_sum: float,
) -> np.ndarray:
    idx = gene_to_idx.get(gene)
    if idx is None:
        return np.full(x.shape[0], np.nan, dtype=float)
    row_sums = np.asarray(x.sum(axis=1)).ravel().astype(float)
    row_sums[row_sums <= 0] = 1.0
    col = x[:, idx]
    vals = np.asarray(col.toarray()).ravel()
    vals = vals * (target_sum / row_sums)
    return np.log1p(vals)


def load_rna_counts(rna_h5ad: Path) -> tuple[sparse.csr_matrix, list[str], dict[str, int]]:
    with h5py.File(rna_h5ad, "r") as f:
        obs = read_string_dataset(f["obs"]["_index"])
        genes = read_string_dataset(f["var"]["_index"])
        counts = read_csr_group(f["layers"]["counts"], shape=(len(obs), len(genes)))
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    return counts, obs, gene_to_idx


def load_locus_configs(config_paths: list[Path]) -> tuple[list[dict], dict[str, list[dict]], dict[str, str]]:
    region_rows: list[dict] = []
    by_locus: dict[str, list[dict]] = {}
    locus_gene: dict[str, str] = {}
    required_cols = {"locus_id", "gene", "chrom", "start", "end", "strand", "region_name", "region_type"}

    for config_path in config_paths:
        df = pd.read_csv(config_path, sep="\t")
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{config_path} missing columns: {sorted(missing)}")

        for row in df.to_dict(orient="records"):
            locus_id = str(row["locus_id"])
            gene = str(row["gene"])
            if locus_id in locus_gene and locus_gene[locus_id] != gene:
                raise ValueError(f"Locus {locus_id} maps to multiple genes")
            locus_gene[locus_id] = gene
            clean_row = dict(row)
            clean_row["locus_id"] = locus_id
            clean_row["gene"] = gene
            clean_row["chrom"] = str(row["chrom"])
            clean_row["start"] = int(row["start"])
            clean_row["end"] = int(row["end"])
            clean_row["strand"] = str(row["strand"])
            clean_row["region_name"] = str(row["region_name"])
            clean_row["region_type"] = str(row["region_type"])
            region_rows.append(clean_row)
            by_locus.setdefault(locus_id, []).append(clean_row)

    return region_rows, by_locus, locus_gene


def main() -> int:
    p = argparse.ArgumentParser(description="Export multiple locus-level pseudo-paired matrices from one-to-one scGLUE matches")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--matching-dir", default=None, help="Default: CausalDiscovery/outputs/scglue_pairings/<RUN_ID>/harmonized_coarse__monocyte__rna_anchor")
    p.add_argument("--locus-config", action="append", required=True, help="TSV with region definitions; repeat for multiple loci")
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--rna-h5ad", default=None, help="Default: integration/outputs/scglue/joint/<RUN_ID>/train/modalities/rna_with_glue.h5ad")
    p.add_argument("--out-base", default=None, help="Default: CausalDiscovery/outputs/locus_matrices_matched/<RUN_ID>")
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--target-sum", type=float, default=1e4)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matching_dir = (
        resolve_path(repo_root, args.matching_dir)
        if args.matching_dir
        else repo_root / "CausalDiscovery" / "outputs" / "scglue_pairings" / sanitize_token(args.run_id) / "harmonized_coarse__monocyte__rna_anchor"
    )
    matched_path = matching_dir / "matched_samples.tsv"
    config_paths = [resolve_path(repo_root, path_str) for path_str in args.locus_config]
    manifest_path = resolve_path(repo_root, args.manifest)
    rna_h5ad = (
        resolve_path(repo_root, args.rna_h5ad)
        if args.rna_h5ad
        else repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train/modalities/rna_with_glue.h5ad"
    )
    out_base = (
        resolve_path(repo_root, args.out_base)
        if args.out_base
        else repo_root / "CausalDiscovery" / "outputs" / "locus_matrices_matched" / sanitize_token(args.run_id)
    )

    if not matched_path.exists():
        raise FileNotFoundError(f"Missing matched samples: {matched_path}")
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if rna_h5ad is None or not rna_h5ad.exists():
        raise FileNotFoundError(f"Missing RNA h5ad: {rna_h5ad}")
    if any(path is None or not path.exists() for path in config_paths):
        missing = [path_str for path_str, path in zip(args.locus_config, config_paths) if path is None or not path.exists()]
        raise FileNotFoundError(f"Missing locus config(s): {missing}")

    region_rows, regions_by_locus, locus_gene = load_locus_configs([path for path in config_paths if path is not None])
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Loading matched samples from {matched_path}", flush=True)
    matched = pd.read_csv(matched_path, sep="\t")
    panel_df = matched.copy()

    print(f"Loading RNA counts from {rna_h5ad}", flush=True)
    rna_counts, rna_obs, gene_to_idx = load_rna_counts(rna_h5ad)
    expr_df = pd.DataFrame({"cell_rna": rna_obs})
    for gene in sorted(set(locus_gene.values())):
        print(f"Computing expression column for {gene}", flush=True)
        expr_df[f"expr__{gene}"] = log1p_norm_for_gene(rna_counts, gene_to_idx, gene=gene, target_sum=args.target_sum)
    panel_df = panel_df.merge(expr_df, on="cell_rna", how="left")

    marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest = manifest.loc[manifest["mark"].astype(str).isin(marks)].copy()

    region_bin_rows: dict[str, list[dict]] = {locus_id: [] for locus_id in regions_by_locus}

    for row in manifest.to_dict(orient="records"):
        mark = str(row["mark"])
        feature_path = resolve_path(repo_root, str(row["bin_features"]))
        matrix_path = resolve_path(repo_root, str(row["bin_mtx"]))
        barcode_path = resolve_path(repo_root, str(row["bin_barcodes"]))
        if not all(path is not None and path.exists() for path in [feature_path, matrix_path, barcode_path]):
            raise FileNotFoundError(f"Missing raw chromatin inputs for {mark}")

        print(f"[{mark}] loading features", flush=True)
        feature_df = pd.read_csv(feature_path, sep="\t")
        features = feature_df.iloc[:, 0].astype(str).tolist()
        feature_intervals = []
        for idx, feat in enumerate(features):
            chrom, start, end = parse_feature_interval(feat)
            feature_intervals.append((idx, chrom, start, end))

        print(f"[{mark}] loading sparse matrix", flush=True)
        x = spio.mmread(matrix_path).tocsr()
        barcodes = pd.read_csv(barcode_path, sep="\t").iloc[:, 0].astype(str).tolist()

        sample_cell_col = f"cell_{mark}"
        if sample_cell_col not in panel_df.columns:
            raise ValueError(f"Matched samples file missing column: {sample_cell_col}")

        for reg in region_rows:
            reg_name = reg["region_name"]
            reg_chrom = reg["chrom"]
            reg_start = int(reg["start"])
            reg_end = int(reg["end"])
            idx = [
                feature_idx
                for feature_idx, chrom, start, end in feature_intervals
                if chrom == reg_chrom and overlaps(start, end, reg_start, reg_end)
            ]
            idx_arr = np.asarray(sorted(idx), dtype=int)
            value_name = f"{reg_name}__{mark}"
            print(
                f"[{mark}] computing {reg['locus_id']}:{reg_name} with {idx_arr.size} bins",
                flush=True,
            )
            per_cell_vals = mean_log1p_norm_for_features(x, idx_arr, target_sum=args.target_sum)
            per_cell_df = pd.DataFrame({sample_cell_col: barcodes, value_name: per_cell_vals})
            panel_df = panel_df.merge(per_cell_df, on=sample_cell_col, how="left")

            region_bin_rows[reg["locus_id"]].append(
                {
                    "locus_id": reg["locus_id"],
                    "gene": reg["gene"],
                    "region_name": reg_name,
                    "region_type": reg["region_type"],
                    "mark": mark,
                    "n_bins": int(idx_arr.size),
                    "bin_features": ";".join([features[i] for i in idx_arr[:200]]),
                    "bin_features_truncated": bool(idx_arr.size > 200),
                }
            )

    base_cols = [col for col in matched.columns]
    summary_records = []
    for locus_id, locus_regions in regions_by_locus.items():
        gene = locus_gene[locus_id]
        locus_dir = out_base / sanitize_token(locus_id)
        locus_dir.mkdir(parents=True, exist_ok=True)

        locus_cols = base_cols + [f"expr__{gene}"]
        for reg in locus_regions:
            for mark in marks:
                locus_cols.append(f"{reg['region_name']}__{mark}")

        locus_matrix = panel_df[locus_cols].copy()
        matrix_path = locus_dir / f"{gene}_matched_locus_matrix.tsv"
        region_bins_path = locus_dir / f"{gene}_region_bins.tsv"
        summary_path = locus_dir / "run_summary.json"

        locus_matrix.to_csv(matrix_path, sep="\t", index=False)
        pd.DataFrame(region_bin_rows[locus_id]).to_csv(region_bins_path, sep="\t", index=False)

        summary = {
            "inputs": {
                "matched_samples_tsv": str(matched_path),
                "locus_config": [str(path) for path in config_paths if path is not None and sanitize_token(pd.read_csv(path, sep='\t')['locus_id'].iloc[0]) == sanitize_token(locus_id)],
                "manifest": str(manifest_path),
                "rna_h5ad": str(rna_h5ad),
            },
            "params": vars(args),
            "locus_id": locus_id,
            "gene": gene,
            "n_samples": int(locus_matrix.shape[0]),
            "n_regions": len(locus_regions),
            "marks": marks,
            "outputs": {
                "matrix_tsv": str(matrix_path),
                "region_bins_tsv": str(region_bins_path),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summary_records.append({"locus_id": locus_id, "gene": gene, "matrix_tsv": str(matrix_path), "out_dir": str(locus_dir)})
        print(f"Wrote {matrix_path}", flush=True)
        print(f"Wrote {region_bins_path}", flush=True)
        print(f"Wrote {summary_path}", flush=True)

    panel_summary_path = out_base / "panel_summary.tsv"
    pd.DataFrame(summary_records).to_csv(panel_summary_path, sep="\t", index=False)
    print(f"Wrote {panel_summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
