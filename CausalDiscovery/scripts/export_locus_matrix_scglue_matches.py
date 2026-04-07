#!/usr/bin/env python3
"""Export a locus-level pseudo-paired matrix from one-to-one scGLUE matches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

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


def load_rna_counts(rna_h5ad: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    with h5py.File(rna_h5ad, "r") as f:
        obs = read_string_dataset(f["obs"]["_index"])
        genes = read_string_dataset(f["var"]["_index"])
        counts = read_csr_group(f["layers"]["counts"], shape=(len(obs), len(genes)))
    return counts, obs, genes


def load_chrom_counts_from_h5ad(chrom_h5ad: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    with h5py.File(chrom_h5ad, "r") as f:
        obs = read_string_dataset(f["obs"]["_index"])
        if "orig_feature" in f["var"]:
            features = read_string_dataset(f["var"]["orig_feature"])
        else:
            features = read_string_dataset(f["var"]["_index"])
        layer_group = f["layers"]["counts"] if "layers" in f and "counts" in f["layers"] else f["X"]
        counts = read_csr_group(layer_group, shape=(len(obs), len(features)))
    return counts, obs, features


def stream_raw_region_aggregates(
    matrix_path: Path,
    barcode_path: Path,
    feature_path: Path,
    selected_barcodes: Iterable[str],
    regions: pd.DataFrame,
    target_sum: float,
) -> tuple[pd.DataFrame, list[dict], int]:
    """Stream a raw Matrix Market file and aggregate only curated regions for selected cells."""

    barcode_series = pd.read_csv(barcode_path, sep="\t", header=None).iloc[:, 0].astype(str)
    barcode_to_row = {barcode: idx + 1 for idx, barcode in enumerate(barcode_series.tolist())}
    kept_barcodes = [bc for bc in selected_barcodes if bc in barcode_to_row]
    selected_rows = {barcode_to_row[bc] for bc in kept_barcodes}

    region_defs = []
    for reg in regions.to_dict(orient="records"):
        region_defs.append(
            {
                "region_name": str(reg["region_name"]),
                "region_type": str(reg["region_type"]),
                "chrom": str(reg["chrom"]),
                "start": int(reg["start"]),
                "end": int(reg["end"]),
            }
        )

    col_to_regions: dict[int, list[str]] = {}
    region_feature_lists: dict[str, list[str]] = {reg["region_name"]: [] for reg in region_defs}
    with feature_path.open("r") as f:
        header = next(f, None)
        for col_idx, line in enumerate(f, start=1):
            feat = line.strip().split("\t", 1)[0]
            if not feat:
                continue
            chrom, start, end = parse_feature_interval(feat)
            overlap_regions = [
                reg["region_name"]
                for reg in region_defs
                if chrom == reg["chrom"] and overlaps(start, end, reg["start"], reg["end"])
            ]
            if overlap_regions:
                col_to_regions[col_idx] = overlap_regions
                for reg_name in overlap_regions:
                    region_feature_lists[reg_name].append(feat)

    row_sum_map = {row_idx: 0.0 for row_idx in selected_rows}
    region_sum_maps: dict[str, dict[int, float]] = {
        reg["region_name"]: {row_idx: 0.0 for row_idx in selected_rows} for reg in region_defs
    }

    with matrix_path.open("r") as f:
        for line in f:
            if not line or line.startswith("%"):
                continue
            # Skip the shape line and begin processing triplets.
            parts = line.strip().split()
            if len(parts) == 3:
                break
        else:
            raise ValueError(f"Malformed Matrix Market file: {matrix_path}")

        # First non-comment 3-column line after header is the shape declaration.
        # Consume remaining triplets.
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            row_idx = int(parts[0])
            if row_idx not in selected_rows:
                continue
            col_idx = int(parts[1])
            value = float(parts[2])
            row_sum_map[row_idx] += value
            region_names = col_to_regions.get(col_idx)
            if not region_names:
                continue
            for reg_name in region_names:
                region_sum_maps[reg_name][row_idx] += value

    value_df = pd.DataFrame({"barcode": kept_barcodes})
    row_sums = np.array([row_sum_map[barcode_to_row[bc]] for bc in kept_barcodes], dtype=float)
    row_sums[row_sums <= 0] = 1.0
    for reg in region_defs:
        reg_name = reg["region_name"]
        n_bins = len(region_feature_lists[reg_name])
        region_vals = np.array([region_sum_maps[reg_name][barcode_to_row[bc]] for bc in kept_barcodes], dtype=float)
        if n_bins == 0:
            score = np.full_like(region_vals, np.nan, dtype=float)
        else:
            score = np.log1p((region_vals / float(n_bins)) * (target_sum / row_sums))
        value_df[reg_name] = score

    region_bin_rows = []
    for reg in region_defs:
        reg_name = reg["region_name"]
        feats = region_feature_lists[reg_name]
        region_bin_rows.append(
            {
                "region_name": reg_name,
                "region_type": reg["region_type"],
                "n_bins": len(feats),
                "bin_features": ";".join(feats[:200]),
                "bin_features_truncated": bool(len(feats) > 200),
            }
        )
    return value_df, region_bin_rows, len(barcode_series)


def main() -> int:
    p = argparse.ArgumentParser(description="Export a locus-level pseudo-paired matrix from one-to-one scGLUE matches")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--matching-dir", default=None, help="Default: CausalDiscovery/outputs/scglue_pairings/<RUN_ID>/harmonized_coarse__monocyte__rna_anchor")
    p.add_argument("--locus-config", required=True, help="TSV with region definitions")
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--rna-h5ad", default=None, help="Default: integration/outputs/scglue/joint/<RUN_ID>/train/modalities/rna_with_glue.h5ad")
    p.add_argument("--out-dir", default=None, help="Default: CausalDiscovery/outputs/locus_matrices_matched/<RUN_ID>/<LOCUS_ID>")
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--chrom-source", default="auto", choices=["auto", "h5ad", "raw"], help="Where to load chromatin counts from")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matching_dir = (
        resolve_path(repo_root, args.matching_dir)
        if args.matching_dir
        else repo_root / "CausalDiscovery" / "outputs" / "scglue_pairings" / sanitize_token(args.run_id) / "harmonized_coarse__monocyte__rna_anchor"
    )
    matched_path = matching_dir / "matched_samples.tsv"
    locus_config = resolve_path(repo_root, args.locus_config)
    manifest_path = resolve_path(repo_root, args.manifest)
    rna_h5ad = (
        resolve_path(repo_root, args.rna_h5ad)
        if args.rna_h5ad
        else repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train/modalities/rna_with_glue.h5ad"
    )

    if not matched_path.exists():
        raise FileNotFoundError(f"Missing matched samples: {matched_path}")
    if locus_config is None or not locus_config.exists():
        raise FileNotFoundError(f"Missing locus config: {args.locus_config}")
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if rna_h5ad is None or not rna_h5ad.exists():
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
        else repo_root / "CausalDiscovery" / "outputs" / "locus_matrices_matched" / sanitize_token(args.run_id) / sanitize_token(locus_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(matched_path, sep="\t")
    matrix_df = matched.copy()

    rna_counts, rna_obs, rna_genes = load_rna_counts(rna_h5ad)
    expr = log1p_norm_for_gene(rna_counts, rna_genes, gene=gene, target_sum=args.target_sum)
    expr_df = pd.DataFrame({"cell_rna": rna_obs, f"expr__{gene}": expr})
    matrix_df = matrix_df.merge(expr_df, on="cell_rna", how="left")

    marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest = manifest.loc[manifest["mark"].astype(str).isin(marks)].copy()

    region_bin_rows: list[dict] = []
    for row in manifest.to_dict(orient="records"):
        mark = str(row["mark"])
        chrom_h5ad = repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train/modalities/chrom_{mark}_with_glue.h5ad"
        if args.chrom_source != "raw" and chrom_h5ad.exists():
            x, barcodes, features = load_chrom_counts_from_h5ad(chrom_h5ad)
            source_used = "h5ad"
        else:
            feature_path = resolve_path(repo_root, str(row["bin_features"]))
            matrix_path = resolve_path(repo_root, str(row["bin_mtx"]))
            barcode_path = resolve_path(repo_root, str(row["bin_barcodes"]))
            if not all(p.exists() for p in [feature_path, matrix_path, barcode_path]):
                raise FileNotFoundError(f"Missing raw chromatin inputs for {mark}")
            source_used = "raw"

        sample_cell_col = f"cell_{mark}"
        if sample_cell_col not in matrix_df.columns:
            raise ValueError(f"Matched samples file missing column: {sample_cell_col}")

        if source_used == "raw":
            print(f"[{mark}] streaming raw Matrix Market for curated regions")
            raw_value_df, raw_region_rows, n_cells = stream_raw_region_aggregates(
                matrix_path=matrix_path,
                barcode_path=barcode_path,
                feature_path=feature_path,
                selected_barcodes=matrix_df[sample_cell_col].astype(str).tolist(),
                regions=regions,
                target_sum=args.target_sum,
            )
            print(f"[{mark}] loaded raw feature metadata and streamed counts for {n_cells} cells")
            per_cell_df = raw_value_df.rename(columns={"barcode": sample_cell_col})
            rename_map = {reg["region_name"]: f"{reg['region_name']}__{mark}" for reg in regions.to_dict(orient='records')}
            per_cell_df = per_cell_df.rename(columns=rename_map)
            matrix_df = matrix_df.merge(per_cell_df, on=sample_cell_col, how="left")
            for reg_row in raw_region_rows:
                region_bin_rows.append(
                    {
                        "locus_id": locus_id,
                        "gene": gene,
                        "region_name": reg_row["region_name"],
                        "region_type": reg_row["region_type"],
                        "mark": mark,
                        "n_bins": int(reg_row["n_bins"]),
                        "bin_features": reg_row["bin_features"],
                        "bin_features_truncated": reg_row["bin_features_truncated"],
                    }
                )
        else:
            feature_idx_map = {}
            for idx, feat in enumerate(features):
                chrom, start, end = parse_feature_interval(feat)
                feature_idx_map[feat] = (idx, chrom, start, end)

            print(f"[{mark}] loaded {source_used} feature matrix with {len(features)} features and {len(barcodes)} cells")

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
                per_cell_df = pd.DataFrame({sample_cell_col: barcodes, value_name: per_cell_vals})
                matrix_df = matrix_df.merge(per_cell_df, on=sample_cell_col, how="left")

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
        print(f"[{mark}] finished {len(regions)} regions")

    matrix_path = out_dir / f"{gene}_matched_locus_matrix.tsv"
    region_bins_path = out_dir / f"{gene}_region_bins.tsv"
    summary_json_path = out_dir / "run_summary.json"

    matrix_df.to_csv(matrix_path, sep="\t", index=False)
    pd.DataFrame(region_bin_rows).to_csv(region_bins_path, sep="\t", index=False)

    summary_json = {
        "locus_id": locus_id,
        "gene": gene,
        "inputs": {
            "run_id": args.run_id,
            "matching_dir": str(matching_dir),
            "locus_config_tsv": str(locus_config),
            "manifest_tsv": str(manifest_path),
            "rna_h5ad": str(rna_h5ad),
        },
        "params": vars(args),
        "n_samples": int(matrix_df["sample_id"].nunique()),
        "regions": regions.to_dict(orient="records"),
        "outputs": {
            "matched_locus_matrix_tsv": str(matrix_path),
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
