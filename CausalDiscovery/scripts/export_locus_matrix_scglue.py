#!/usr/bin/env python3
"""Export a locus-level metacell matrix from joint scGLUE outputs and raw chromatin bins."""

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


def strip_10x_suffix(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"-\d+$", "", regex=True)


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
    genes: list[str],
    gene: str,
    target_sum: float,
) -> np.ndarray:
    try:
        idx = genes.index(gene)
    except ValueError:
        return np.full(x.shape[0], np.nan, dtype=float)
    row_sums = np.asarray(x.sum(axis=1)).ravel().astype(float)
    row_sums[row_sums <= 0] = 1.0
    col = x[:, idx]
    vals = np.asarray(col.toarray()).ravel()
    vals = vals * (target_sum / row_sums)
    return np.log1p(vals)


def aggregate_to_metacells(
    per_cell: pd.DataFrame,
    assignments: pd.DataFrame,
    value_col: str,
) -> pd.Series:
    merged = per_cell.merge(
        assignments[["cell", "metacell_id"]],
        on="cell",
        how="inner",
    )
    if merged.empty:
        return pd.Series(dtype=float)
    return merged.groupby("metacell_id", sort=True)[value_col].mean()


def load_rna_counts(rna_h5ad: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    with h5py.File(rna_h5ad, "r") as f:
        obs = read_string_dataset(f["obs"]["_index"])
        genes = read_string_dataset(f["var"]["_index"])
        counts = read_csr_group(f["layers"]["counts"], shape=(len(obs), len(genes)))
    return counts, obs, genes


def main() -> int:
    p = argparse.ArgumentParser(description="Export a locus-level metacell matrix from scGLUE outputs")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--metacell-dir", default=None, help="Default: CausalDiscovery/outputs/scglue_metacells/<RUN_ID>/harmonized_coarse__monocyte")
    p.add_argument("--locus-config", required=True, help="TSV with region definitions")
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--rna-h5ad", default=None, help="Default: integration/outputs/scglue/joint/<RUN_ID>/train/modalities/rna_with_glue.h5ad")
    p.add_argument("--out-dir", default=None, help="Default: CausalDiscovery/outputs/locus_matrices/<RUN_ID>/<LOCUS_ID>")
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--filter-qc-pass-metacells", action="store_true", default=True)
    p.add_argument("--no-filter-qc-pass-metacells", action="store_false", dest="filter_qc_pass_metacells")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metacell_dir = (
        resolve_path(repo_root, args.metacell_dir)
        if args.metacell_dir
        else repo_root / "CausalDiscovery" / "outputs" / "scglue_metacells" / sanitize_token(args.run_id) / "harmonized_coarse__monocyte"
    )
    assignments_path = metacell_dir / "cell_assignments.tsv"
    summary_path = metacell_dir / "metacell_summary.tsv"
    locus_config = resolve_path(repo_root, args.locus_config)
    manifest_path = resolve_path(repo_root, args.manifest)
    rna_h5ad = (
        resolve_path(repo_root, args.rna_h5ad)
        if args.rna_h5ad
        else repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train/modalities/rna_with_glue.h5ad"
    )

    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing metacell assignments: {assignments_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metacell summary: {summary_path}")
    if not locus_config.exists():
        raise FileNotFoundError(f"Missing locus config: {locus_config}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not rna_h5ad.exists():
        raise FileNotFoundError(f"Missing RNA h5ad: {rna_h5ad}")

    regions = pd.read_csv(locus_config, sep="\t")
    required_cols = {"locus_id", "gene", "chrom", "start", "end", "strand", "region_name", "region_type"}
    missing = required_cols - set(regions.columns)
    if missing:
        raise ValueError(f"Locus config missing columns: {sorted(missing)}")
    locus_id = str(regions["locus_id"].iloc[0])
    gene = str(regions["gene"].iloc[0])
    if regions["locus_id"].nunique() != 1 or regions["gene"].nunique() != 1:
        raise ValueError("Current exporter expects a single locus_id and gene per config")

    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root / "CausalDiscovery" / "outputs" / "locus_matrices" / sanitize_token(args.run_id) / sanitize_token(locus_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(assignments_path, sep="\t")
    summary = pd.read_csv(summary_path, sep="\t")
    if args.filter_qc_pass_metacells and "passes_default_qc" in summary.columns:
        keep_metacells = set(summary.loc[summary["passes_default_qc"].astype(bool), "metacell_id"].astype(str))
        assignments = assignments.loc[assignments["metacell_id"].astype(str).isin(keep_metacells)].copy()
        summary = summary.loc[summary["metacell_id"].astype(str).isin(keep_metacells)].copy()

    marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest = manifest.loc[manifest["mark"].astype(str).isin(marks)].copy()

    matrix_df = summary[["metacell_id"]].copy()

    rna_assign = assignments.loc[assignments["modality_key"].astype(str) == "rna", ["cell", "metacell_id"]].copy()
    rna_counts, rna_obs, rna_genes = load_rna_counts(rna_h5ad)
    expr = log1p_norm_for_gene(rna_counts, rna_genes, gene=gene, target_sum=args.target_sum)
    expr_df = pd.DataFrame({"cell": rna_obs, f"expr__{gene}": expr})
    expr_meta = aggregate_to_metacells(expr_df, rna_assign, value_col=f"expr__{gene}")
    matrix_df = matrix_df.merge(expr_meta.rename(f"expr__{gene}"), on="metacell_id", how="left")

    region_bin_rows: list[dict] = []
    for row in manifest.to_dict(orient="records"):
        mark = str(row["mark"])
        feature_path = resolve_path(repo_root, str(row["bin_features"]))
        matrix_path = resolve_path(repo_root, str(row["bin_mtx"]))
        barcode_path = resolve_path(repo_root, str(row["bin_barcodes"]))
        if not all(p.exists() for p in [feature_path, matrix_path, barcode_path]):
            raise FileNotFoundError(f"Missing raw chromatin inputs for {mark}")

        feature_df = pd.read_csv(feature_path, sep="\t")
        features = feature_df.iloc[:, 0].astype(str).tolist()
        feature_idx_map = {}
        for idx, feat in enumerate(features):
            chrom, start, end = parse_feature_interval(feat)
            feature_idx_map[feat] = (idx, chrom, start, end)

        x = spio.mmread(matrix_path).tocsr()
        barcodes = pd.read_csv(barcode_path, sep="\t").iloc[:, 0].astype(str).tolist()
        chrom_assign = assignments.loc[assignments["mark"].astype(str) == mark, ["cell", "metacell_id"]].copy()
        if chrom_assign.empty:
            continue

        for reg in regions.to_dict(orient="records"):
            reg_name = str(reg["region_name"])
            reg_chrom = str(reg["chrom"])
            reg_start = int(reg["start"])
            reg_end = int(reg["end"])

            idx = [
                feature_idx
                for _, (feature_idx, chrom, start, end) in feature_idx_map.items()
                if chrom == reg_chrom and overlaps(start, end, reg_start, reg_end)
            ]
            idx_arr = np.asarray(sorted(idx), dtype=int)
            value_name = f"{reg_name}__{mark}"
            per_cell_vals = mean_log1p_norm_for_features(x, idx_arr, target_sum=args.target_sum)
            per_cell_df = pd.DataFrame({"cell": barcodes, value_name: per_cell_vals})
            meta_vals = aggregate_to_metacells(per_cell_df, chrom_assign, value_col=value_name)
            matrix_df = matrix_df.merge(meta_vals.rename(value_name), on="metacell_id", how="left")

            region_bin_rows.append(
                {
                    "locus_id": locus_id,
                    "gene": gene,
                    "region_name": reg_name,
                    "region_type": reg["region_type"],
                    "mark": mark,
                    "n_bins": int(idx_arr.size),
                    "bin_features": ";".join([features[i] for i in idx_arr[:200]]),
                    "bin_features_truncated": bool(idx_arr.size > 200),
                }
            )

    matrix_df = matrix_df.merge(summary, on="metacell_id", how="left")

    matrix_path = out_dir / f"{gene}_locus_matrix.tsv"
    region_bins_path = out_dir / f"{gene}_region_bins.tsv"
    summary_json_path = out_dir / "run_summary.json"

    matrix_df.to_csv(matrix_path, sep="\t", index=False)
    pd.DataFrame(region_bin_rows).to_csv(region_bins_path, sep="\t", index=False)

    summary_json = {
        "locus_id": locus_id,
        "gene": gene,
        "inputs": {
            "run_id": args.run_id,
            "metacell_dir": str(metacell_dir),
            "locus_config_tsv": str(locus_config),
            "manifest_tsv": str(manifest_path),
            "rna_h5ad": str(rna_h5ad),
        },
        "params": vars(args),
        "n_metacells": int(matrix_df["metacell_id"].nunique()),
        "regions": regions.to_dict(orient="records"),
        "outputs": {
            "locus_matrix_tsv": str(matrix_path),
            "region_bins_tsv": str(region_bins_path),
        },
    }
    with summary_json_path.open("w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Wrote: {matrix_path}")
    print(f"Wrote: {region_bins_path}")
    print(f"Wrote: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
