#!/usr/bin/env python3
"""Generate nearby CUT&Tag peak datasets from one-to-one scGLUE matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from export_locus_matrix_scglue_matches import (
    infer_repo_root,
    load_rna_counts,
    log1p_norm_for_gene,
    overlaps,
    raw_counts_for_gene,
    resolve_path,
    row_sums_for_matrix,
    sanitize_token,
    stream_raw_region_aggregates,
)


DEFAULT_MARKS = "H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3"
REGION_BIN_COLUMNS = [
    "gene",
    "mark",
    "region_name",
    "region_type",
    "dataset_kind",
    "variable_name",
    "n_bins",
    "bin_features",
    "bin_features_truncated",
]


def load_gene_panel(repo_root: Path, panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, sep="\t")
    required = {"gene", "chrom", "tss", "strand", "cell_type", "curated_locus_config"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Gene panel missing columns: {sorted(missing)}")
    panel = panel.copy()
    panel["gene"] = panel["gene"].astype(str)
    panel["chrom"] = panel["chrom"].astype(str)
    panel["tss"] = panel["tss"].astype(int)
    panel["strand"] = panel["strand"].astype(str)
    panel["cell_type"] = panel["cell_type"].astype(str)
    panel["curated_locus_config"] = panel["curated_locus_config"].astype(str)
    panel["curated_locus_config_resolved"] = panel["curated_locus_config"].map(
        lambda x: str(resolve_path(repo_root, x))
    )
    if panel["gene"].duplicated().any():
        dupes = sorted(panel.loc[panel["gene"].duplicated(), "gene"].astype(str).unique().tolist())
        raise ValueError(f"Gene panel has duplicated genes: {dupes}")
    if set(panel["cell_type"].unique()) != {"monocyte"}:
        raise ValueError("This generator currently supports only monocyte panels")
    for config_path in panel["curated_locus_config_resolved"]:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Missing curated locus config: {config_path}")
    return panel


def read_curated_regions(config_path: Path, expected_gene: str) -> pd.DataFrame:
    regions = pd.read_csv(config_path, sep="\t")
    required = {"locus_id", "gene", "chrom", "start", "end", "strand", "region_name", "region_type"}
    missing = required - set(regions.columns)
    if missing:
        raise ValueError(f"Curated locus config missing columns: {sorted(missing)}")
    if regions["gene"].astype(str).nunique() != 1:
        raise ValueError(f"{config_path} contains multiple genes")
    gene = str(regions["gene"].astype(str).iloc[0])
    if gene != expected_gene:
        raise ValueError(f"{config_path} gene {gene} did not match expected {expected_gene}")
    out = regions.copy()
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["strand"] = out["strand"].astype(str)
    out["region_name"] = out["region_name"].astype(str)
    out["region_type"] = out["region_type"].astype(str)
    return out


def read_peaks_bed(peaks_bed: Path, mark: str) -> pd.DataFrame:
    peaks = pd.read_csv(peaks_bed, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "peak_name"])
    peaks = peaks.copy()
    peaks["chrom"] = peaks["chrom"].astype(str)
    peaks["start"] = peaks["start"].astype(int)
    peaks["end"] = peaks["end"].astype(int)
    peaks["peak_name"] = peaks["peak_name"].astype(str)
    peaks["mark"] = mark
    return peaks


def signed_distance_to_tss(start: int, end: int, tss: int, strand: str) -> int:
    center = (int(start) + int(end)) // 2
    return int(center - tss) if strand == "+" else int(tss - center)


def curated_region_metadata(
    regions: pd.DataFrame,
    gene: str,
    tss: int,
    strand: str,
    curated_config_path: Path,
) -> pd.DataFrame:
    rows = []
    for reg in regions.to_dict(orient="records"):
        rows.append(
            {
                "dataset_kind": "curated_region",
                "gene": gene,
                "locus_id": str(reg["locus_id"]),
                "region_name": str(reg["region_name"]),
                "region_type": str(reg["region_type"]),
                "chrom": str(reg["chrom"]),
                "start": int(reg["start"]),
                "end": int(reg["end"]),
                "strand": strand,
                "tss": int(tss),
                "signed_distance_to_tss_bp": signed_distance_to_tss(reg["start"], reg["end"], tss, strand),
                "overlaps_curated_region": True,
                "overlapping_curated_regions": str(reg["region_name"]),
                "curated_locus_config": str(curated_config_path),
                "description": str(reg.get("description", "")),
                "source": str(reg.get("source", "")),
            }
        )
    return pd.DataFrame(rows)


def build_nearby_peak_regions(
    gene_row: pd.Series,
    curated_regions: pd.DataFrame,
    peaks_df: pd.DataFrame,
    mark: str,
    window_bp: int,
) -> pd.DataFrame:
    chrom = str(gene_row["chrom"])
    tss = int(gene_row["tss"])
    strand = str(gene_row["strand"])
    window_start = tss - int(window_bp)
    window_end = tss + int(window_bp)

    peaks = peaks_df.loc[peaks_df["chrom"] == chrom].copy()
    peaks = peaks.loc[~((peaks["end"] <= window_start) | (peaks["start"] >= window_end))].copy()
    if peaks.empty:
        return pd.DataFrame(
            columns=[
                "dataset_kind",
                "gene",
                "mark",
                "region_name",
                "region_type",
                "chrom",
                "start",
                "end",
                "peak_name",
                "tss",
                "strand",
                "window_start",
                "window_end",
                "peak_length_bp",
                "signed_distance_to_tss_bp",
                "overlaps_curated_region",
                "overlapping_curated_regions",
            ]
        )

    rows: list[dict[str, Any]] = []
    for peak in peaks.to_dict(orient="records"):
        overlaps_curated = [
            str(reg["region_name"])
            for reg in curated_regions.to_dict(orient="records")
            if str(reg["chrom"]) == str(peak["chrom"])
            and overlaps(int(reg["start"]), int(reg["end"]), int(peak["start"]), int(peak["end"]))
        ]
        rows.append(
            {
                "dataset_kind": "nearby_peak",
                "gene": str(gene_row["gene"]),
                "mark": mark,
                "region_name": f"{peak['chrom']}:{int(peak['start'])}-{int(peak['end'])}__{mark}",
                "region_type": "cuttag_peak",
                "chrom": str(peak["chrom"]),
                "start": int(peak["start"]),
                "end": int(peak["end"]),
                "peak_name": str(peak["peak_name"]),
                "tss": int(tss),
                "strand": strand,
                "window_start": int(window_start),
                "window_end": int(window_end),
                "peak_length_bp": int(peak["end"]) - int(peak["start"]),
                "signed_distance_to_tss_bp": signed_distance_to_tss(peak["start"], peak["end"], tss, strand),
                "overlaps_curated_region": bool(overlaps_curated),
                "overlapping_curated_regions": ";".join(overlaps_curated),
            }
        )
    return pd.DataFrame(rows)


def write_bed(df: pd.DataFrame, path: Path, name_col: str, strand_col: str = "strand") -> None:
    if df.empty:
        path.write_text("")
        return
    bed_df = df[["chrom", "start", "end", name_col]].copy()
    bed_df["score"] = 0
    if strand_col in df.columns:
        bed_df["strand"] = df[strand_col].astype(str)
    else:
        bed_df["strand"] = "."
    bed_df.to_csv(path, sep="\t", header=False, index=False)


def annotate_peak_stats(matrix_df: pd.DataFrame, peak_columns: list[str]) -> pd.DataFrame:
    if not peak_columns:
        return pd.DataFrame(
            columns=[
                "variable_name",
                "n_unique",
                "std",
                "zero_fraction",
                "missing_fraction",
                "min",
                "max",
            ]
        )
    rows = []
    for col in peak_columns:
        series = matrix_df[col]
        nonmissing = series.dropna()
        rows.append(
            {
                "variable_name": col,
                "n_unique": int(nonmissing.nunique(dropna=True)),
                "std": float(nonmissing.std(ddof=1)) if nonmissing.shape[0] > 1 else 0.0,
                "zero_fraction": float((nonmissing == 0).mean()) if nonmissing.shape[0] else 1.0,
                "missing_fraction": float(series.isna().mean()),
                "min": float(nonmissing.min()) if nonmissing.shape[0] else np.nan,
                "max": float(nonmissing.max()) if nonmissing.shape[0] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n")


def export_gene_datasets(
    repo_root: Path,
    matched: pd.DataFrame,
    gene_row: pd.Series,
    expr_log1p_values: pd.Series,
    expr_raw_values: pd.Series,
    rna_library_sizes: pd.Series,
    manifest: pd.DataFrame,
    target_sum: float,
    window_bp: int,
    quant_mode: str,
    gene_out_dir: Path,
) -> dict[str, Any]:
    gene = str(gene_row["gene"])
    chrom = str(gene_row["chrom"])
    tss = int(gene_row["tss"])
    strand = str(gene_row["strand"])
    curated_config = Path(str(gene_row["curated_locus_config_resolved"]))
    curated_regions = read_curated_regions(curated_config, expected_gene=gene)

    base_df = matched.copy()
    expr_log1p_col = f"expr__{gene}_log1p"
    expr_raw_col = f"expr__{gene}_raw_counts"
    expr_df = pd.DataFrame(
        {
            "cell_rna": expr_log1p_values.index.astype(str),
            expr_log1p_col: expr_log1p_values.to_numpy(),
            expr_raw_col: expr_raw_values.to_numpy(),
            "libsize__rna": rna_library_sizes.to_numpy(),
        }
    )
    base_df = base_df.merge(expr_df, on="cell_rna", how="left")
    if base_df[expr_log1p_col].isna().all() or base_df[expr_raw_col].isna().all():
        raise RuntimeError(f"Expression values for {gene} were all missing")

    nearby_dir = gene_out_dir / "nearby_peaks"
    nearby_dir.mkdir(parents=True, exist_ok=True)
    nearby_matrix = base_df.copy()
    nearby_col_order: list[str] = []
    nearby_bin_rows: list[dict[str, Any]] = []
    nearby_peak_metadata_parts: list[pd.DataFrame] = []

    curated_meta_df = curated_region_metadata(
        curated_regions,
        gene=gene,
        tss=tss,
        strand=strand,
        curated_config_path=curated_config,
    )

    for manifest_row in manifest.to_dict(orient="records"):
        mark = str(manifest_row["mark"])
        sample_cell_col = f"cell_{mark}"
        if sample_cell_col not in base_df.columns:
            raise ValueError(f"Matched samples file missing column: {sample_cell_col}")

        peaks_bed = resolve_path(repo_root, str(manifest_row["peaks_bed"]))
        feature_path = resolve_path(repo_root, str(manifest_row["bin_features"]))
        matrix_path = resolve_path(repo_root, str(manifest_row["bin_mtx"]))
        barcode_path = resolve_path(repo_root, str(manifest_row["bin_barcodes"]))
        if not all(path is not None and path.exists() for path in [peaks_bed, feature_path, matrix_path, barcode_path]):
            raise FileNotFoundError(f"Missing raw inputs for {gene} {mark}")

        peaks_df = read_peaks_bed(peaks_bed, mark=mark)
        nearby_regions = build_nearby_peak_regions(
            gene_row=gene_row,
            curated_regions=curated_regions,
            peaks_df=peaks_df,
            mark=mark,
            window_bp=window_bp,
        )
        nearby_peak_metadata_parts.append(nearby_regions.copy())

        raw_value_df, raw_region_rows, _, raw_library_df = stream_raw_region_aggregates(
            matrix_path=matrix_path,
            barcode_path=barcode_path,
            feature_path=feature_path,
            selected_barcodes=base_df[sample_cell_col].astype(str).tolist(),
            regions=nearby_regions[["region_name", "region_type", "chrom", "start", "end"]].copy(),
            target_sum=target_sum,
            quant_mode=quant_mode,
        )
        per_cell_df = raw_value_df.rename(columns={"barcode": sample_cell_col})
        raw_library_df = raw_library_df.rename(columns={"barcode": sample_cell_col, "library_size": f"libsize__{mark}"})
        base_df = base_df.merge(raw_library_df, on=sample_cell_col, how="left")
        nearby_matrix = nearby_matrix.merge(raw_library_df, on=sample_cell_col, how="left")

        nearby_cols = nearby_regions["region_name"].astype(str).tolist()
        if nearby_cols:
            nearby_values_df = per_cell_df[[sample_cell_col] + nearby_cols].copy()
            nearby_matrix = nearby_matrix.merge(nearby_values_df, on=sample_cell_col, how="left")
            nearby_col_order.extend(nearby_cols)

        region_meta_by_name = {
            str(reg["region_name"]): reg
            for reg in nearby_regions.assign(dataset_kind="nearby_peak").to_dict(orient="records")
        }
        for raw_row in raw_region_rows:
            region_name = str(raw_row["region_name"])
            region_meta = region_meta_by_name[region_name]
            out_row = {
                "gene": gene,
                "mark": mark,
                "region_name": region_name,
                "region_type": str(raw_row["region_type"]),
                "dataset_kind": str(region_meta["dataset_kind"]),
                "n_bins": int(raw_row["n_bins"]),
                "bin_features": str(raw_row["bin_features"]),
                "bin_features_truncated": bool(raw_row["bin_features_truncated"]),
            }
            out_row["variable_name"] = region_name
            nearby_bin_rows.append(out_row)

    base_cols = list(base_df.columns)
    nearby_matrix = nearby_matrix.loc[:, base_cols + nearby_col_order]

    nearby_peak_metadata = (
        pd.concat(nearby_peak_metadata_parts, ignore_index=True)
        if nearby_peak_metadata_parts
        else pd.DataFrame(
            columns=[
                "dataset_kind",
                "gene",
                "mark",
                "region_name",
                "region_type",
                "chrom",
                "start",
                "end",
                "peak_name",
                "tss",
                "strand",
                "window_start",
                "window_end",
                "peak_length_bp",
                "signed_distance_to_tss_bp",
                "overlaps_curated_region",
                "overlapping_curated_regions",
            ]
        )
    )
    nearby_peak_metadata = nearby_peak_metadata.rename(columns={"region_name": "variable_name"})

    nearby_stats = annotate_peak_stats(nearby_matrix, nearby_col_order)
    nearby_bins_df = pd.DataFrame(nearby_bin_rows, columns=REGION_BIN_COLUMNS)
    if not nearby_bins_df.empty:
        nearby_peak_metadata = nearby_peak_metadata.merge(
            nearby_bins_df[["variable_name", "n_bins", "bin_features", "bin_features_truncated"]],
            on="variable_name",
            how="left",
        )
    nearby_peak_metadata = nearby_peak_metadata.merge(nearby_stats, on="variable_name", how="left")

    nearby_matrix_path = nearby_dir / f"{gene}_nearby_peak_matrix.tsv"
    nearby_meta_path = nearby_dir / f"{gene}_nearby_peak_metadata.tsv"
    nearby_bins_path = nearby_dir / f"{gene}_nearby_peak_bins.tsv"
    nearby_bed_path = nearby_dir / f"{gene}_nearby_candidate_peaks.bed"
    eval_curated_meta_path = nearby_dir / f"{gene}_evaluation_curated_regions.tsv"
    eval_curated_bed_path = nearby_dir / f"{gene}_evaluation_curated_regions.bed"
    nearby_summary_path = nearby_dir / "run_summary.json"

    nearby_matrix.to_csv(nearby_matrix_path, sep="\t", index=False)
    nearby_peak_metadata.to_csv(nearby_meta_path, sep="\t", index=False)
    nearby_bins_df.to_csv(nearby_bins_path, sep="\t", index=False)
    write_bed(nearby_peak_metadata, nearby_bed_path, name_col="variable_name")
    curated_meta_df.to_csv(eval_curated_meta_path, sep="\t", index=False)
    write_bed(curated_meta_df, eval_curated_bed_path, name_col="region_name")

    nearby_summary = {
        "dataset_kind": "nearby_peaks",
        "gene": gene,
        "chrom": chrom,
        "tss": tss,
        "strand": strand,
        "window_bp": int(window_bp),
        "window_start": int(tss - window_bp),
        "window_end": int(tss + window_bp),
        "quant_mode": quant_mode,
        "rna_columns": [expr_log1p_col, expr_raw_col],
        "library_size_columns": ["libsize__rna", *[f"libsize__{mark}" for mark in manifest["mark"].astype(str).tolist()]],
        "n_samples": int(nearby_matrix["sample_id"].nunique()),
        "n_nearby_peak_variables": int(len(nearby_col_order)),
        "n_curated_regions_for_evaluation": int(curated_regions.shape[0]),
        "n_nearby_peaks_overlapping_curated_regions": int(
            nearby_peak_metadata["overlaps_curated_region"].astype("boolean").fillna(False).astype(bool).sum()
        ),
        "outputs": {
            "nearby_peak_matrix_tsv": str(nearby_matrix_path),
            "nearby_peak_metadata_tsv": str(nearby_meta_path),
            "nearby_peak_bins_tsv": str(nearby_bins_path),
            "nearby_candidate_peaks_bed": str(nearby_bed_path),
            "evaluation_curated_regions_tsv": str(eval_curated_meta_path),
            "evaluation_curated_regions_bed": str(eval_curated_bed_path),
        },
    }

    json_dump(nearby_summary_path, nearby_summary)

    return {
        "gene": gene,
        "gene_out_dir": str(gene_out_dir),
        "nearby_peak_variables": len(nearby_col_order),
        "curated_regions_for_evaluation": int(curated_regions.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate nearby CUT&Tag peak datasets from monocyte scGLUE pseudo-pairs")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--matching-dir", default=None, help="Default: CausalDiscovery/outputs/scglue_pairings/<RUN_ID>/harmonized_coarse__monocyte__rna_anchor")
    p.add_argument("--gene-panel", default="CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv")
    p.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p.add_argument("--rna-h5ad", default=None, help="Default: integration/outputs/scglue/joint/<RUN_ID>/train/modalities/rna_with_glue.h5ad")
    p.add_argument("--window-bp", type=int, default=10000)
    p.add_argument("--marks", default=DEFAULT_MARKS)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--quant-mode", default="log1p_norm", choices=["log1p_norm", "raw_counts"])
    p.add_argument("--out-root", default=None, help="Default: CausalDiscovery/outputs/datasets/<RUN_ID>/<GENE_PANEL_STEM>")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matching_dir = (
        resolve_path(repo_root, args.matching_dir)
        if args.matching_dir
        else repo_root / "CausalDiscovery" / "outputs" / "scglue_pairings" / sanitize_token(args.run_id) / "harmonized_coarse__monocyte__rna_anchor"
    )
    matched_path = matching_dir / "matched_samples.tsv"
    gene_panel_path = resolve_path(repo_root, args.gene_panel)
    manifest_path = resolve_path(repo_root, args.manifest)
    rna_h5ad = (
        resolve_path(repo_root, args.rna_h5ad)
        if args.rna_h5ad
        else repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train/modalities/rna_with_glue.h5ad"
    )

    if not matched_path.exists():
        raise FileNotFoundError(f"Missing matched samples: {matched_path}")
    if gene_panel_path is None or not gene_panel_path.exists():
        raise FileNotFoundError(f"Missing gene panel: {args.gene_panel}")
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if rna_h5ad is None or not rna_h5ad.exists():
        raise FileNotFoundError(f"Missing RNA h5ad: {rna_h5ad}")

    panel = load_gene_panel(repo_root, gene_panel_path)
    panel_token = sanitize_token(gene_panel_path.stem)
    out_root = (
        resolve_path(repo_root, args.out_root)
        if args.out_root
        else repo_root / "CausalDiscovery" / "outputs" / "datasets" / sanitize_token(args.run_id) / panel_token
    )
    out_root.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(matched_path, sep="\t")
    manifest = pd.read_csv(manifest_path, sep="\t")
    marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    manifest = manifest.loc[manifest["mark"].astype(str).isin(marks)].copy()
    if manifest.empty:
        raise RuntimeError("No marks remained after filtering the manifest")

    rna_counts, rna_obs, rna_genes = load_rna_counts(rna_h5ad)
    rna_index = pd.Index(rna_obs, name="cell_rna")
    rna_library_sizes = pd.Series(row_sums_for_matrix(rna_counts), index=rna_index, name="libsize__rna")
    expr_log1p_cache: dict[str, pd.Series] = {}
    expr_raw_cache: dict[str, pd.Series] = {}
    for gene in panel["gene"].astype(str).tolist():
        expr_log1p_values = log1p_norm_for_gene(rna_counts, rna_genes, gene=gene, target_sum=args.target_sum)
        expr_raw_values = raw_counts_for_gene(rna_counts, rna_genes, gene=gene)
        expr_log1p_cache[gene] = pd.Series(expr_log1p_values, index=rna_index, name=f"expr__{gene}_log1p")
        expr_raw_cache[gene] = pd.Series(expr_raw_values, index=rna_index, name=f"expr__{gene}_raw_counts")

    panel_summary_rows = []
    for gene_row in panel.to_dict(orient="records"):
        gene = str(gene_row["gene"])
        gene_dir = out_root / sanitize_token(gene)
        print(f"[{gene}] generating nearby peak datasets", flush=True)
        panel_summary_rows.append(
            export_gene_datasets(
                repo_root=repo_root,
                matched=matched,
                gene_row=pd.Series(gene_row),
                expr_log1p_values=expr_log1p_cache[gene],
                expr_raw_values=expr_raw_cache[gene],
                rna_library_sizes=rna_library_sizes,
                manifest=manifest,
                target_sum=args.target_sum,
                window_bp=args.window_bp,
                quant_mode=args.quant_mode,
                gene_out_dir=gene_dir,
            )
        )

    panel_summary = {
        "run_id": args.run_id,
        "matching_dir": str(matching_dir),
        "gene_panel": str(gene_panel_path),
        "window_bp": int(args.window_bp),
        "marks": marks,
        "target_sum": float(args.target_sum),
        "quant_mode": args.quant_mode,
        "genes": panel_summary_rows,
    }
    json_dump(out_root / "panel_summary.json", panel_summary)
    print(f"Wrote: {out_root / 'panel_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
