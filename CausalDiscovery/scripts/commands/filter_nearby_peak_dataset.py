#!/usr/bin/env python3
"""Filter nearby CUT&Tag peak datasets by structural peak metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import copy_if_exists, infer_repo_root, json_dump, load_panel_genes, resolve_path, sanitize_token, write_bed


def metric_rows(matrix: pd.DataFrame, peak_columns: list[str], min_nonzero_fraction: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in peak_columns:
        series = pd.to_numeric(matrix[col], errors="coerce")
        nonmissing = series.dropna()
        nonzero_fraction = float((nonmissing > 0).mean()) if nonmissing.shape[0] else 0.0
        zero_fraction = float((nonmissing == 0).mean()) if nonmissing.shape[0] else 1.0
        n_unique = int(nonmissing.nunique(dropna=True))
        std = float(nonmissing.std(ddof=1)) if nonmissing.shape[0] > 1 else 0.0
        keep = bool(nonzero_fraction > min_nonzero_fraction)
        rows.append(
            {
                "variable_name": col,
                "n_nonmissing": int(nonmissing.shape[0]),
                "n_unique": n_unique,
                "std": std,
                "zero_fraction": zero_fraction,
                "nonzero_fraction": nonzero_fraction,
                "min": float(nonmissing.min()) if nonmissing.shape[0] else np.nan,
                "max": float(nonmissing.max()) if nonmissing.shape[0] else np.nan,
                "keep": keep,
                "drop_reason": "" if keep else f"nonzero_fraction <= {min_nonzero_fraction:g}",
            }
        )
    return pd.DataFrame(rows)


def filter_one_gene(
    dataset_root: Path,
    out_root: Path,
    gene: str,
    *,
    min_nonzero_fraction: float,
) -> dict[str, Any]:
    gene_token = sanitize_token(gene)
    src_dir = dataset_root / gene_token / "nearby_peaks"
    out_dir = out_root / gene_token / "nearby_peaks"
    matrix_path = src_dir / f"{gene}_nearby_peak_matrix.tsv"
    metadata_path = src_dir / f"{gene}_nearby_peak_metadata.tsv"
    bins_path = src_dir / f"{gene}_nearby_peak_bins.tsv"
    summary_path = src_dir / "run_summary.json"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing matrix for {gene}: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for {gene}: {metadata_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(matrix_path, sep="\t")
    metadata = pd.read_csv(metadata_path, sep="\t")
    if "variable_name" not in metadata.columns:
        raise ValueError(f"Metadata missing variable_name column: {metadata_path}")

    candidate_peak_columns = metadata["variable_name"].astype(str).tolist()
    peak_columns = [col for col in candidate_peak_columns if col in matrix.columns]
    missing_peak_columns = sorted(set(candidate_peak_columns) - set(peak_columns))
    metrics = metric_rows(matrix, peak_columns, min_nonzero_fraction=min_nonzero_fraction)
    retained_peak_columns = metrics.loc[metrics["keep"], "variable_name"].astype(str).tolist()
    dropped = metrics.loc[~metrics["keep"]].copy()

    non_peak_columns = [col for col in matrix.columns if col not in peak_columns]
    filtered_matrix = matrix.loc[:, non_peak_columns + retained_peak_columns].copy()

    retained_metadata = metadata.loc[metadata["variable_name"].astype(str).isin(retained_peak_columns)].copy()
    retained_metadata = retained_metadata.drop(
        columns=[col for col in ["n_unique", "std", "zero_fraction", "missing_fraction", "min", "max"] if col in retained_metadata.columns],
        errors="ignore",
    )
    retained_metadata = retained_metadata.merge(
        metrics.drop(columns=["keep", "drop_reason"]),
        on="variable_name",
        how="left",
    )

    bins = pd.read_csv(bins_path, sep="\t") if bins_path.exists() else pd.DataFrame()
    if not bins.empty and "variable_name" in bins.columns:
        retained_bins = bins.loc[bins["variable_name"].astype(str).isin(retained_peak_columns)].copy()
    else:
        retained_bins = bins

    filtered_matrix_path = out_dir / f"{gene}_nearby_peak_matrix.tsv"
    filtered_metadata_path = out_dir / f"{gene}_nearby_peak_metadata.tsv"
    filtered_bins_path = out_dir / f"{gene}_nearby_peak_bins.tsv"
    filtered_bed_path = out_dir / f"{gene}_nearby_candidate_peaks.bed"
    dropped_path = out_dir / f"{gene}_dropped_peaks.tsv"
    metrics_path = out_dir / f"{gene}_peak_filter_metrics.tsv"
    filtered_summary_path = out_dir / "run_summary.json"

    filtered_matrix.to_csv(filtered_matrix_path, sep="\t", index=False)
    retained_metadata.to_csv(filtered_metadata_path, sep="\t", index=False)
    retained_bins.to_csv(filtered_bins_path, sep="\t", index=False)
    write_bed(retained_metadata, filtered_bed_path)
    dropped.to_csv(dropped_path, sep="\t", index=False)
    metrics.to_csv(metrics_path, sep="\t", index=False)

    copy_if_exists(src_dir / f"{gene}_evaluation_curated_regions.tsv", out_dir / f"{gene}_evaluation_curated_regions.tsv")
    copy_if_exists(src_dir / f"{gene}_evaluation_curated_regions.bed", out_dir / f"{gene}_evaluation_curated_regions.bed")

    source_summary: dict[str, Any] = {}
    if summary_path.exists():
        source_summary = json.loads(summary_path.read_text())

    summary = {
        **source_summary,
        "dataset_kind": "nearby_peaks_nonzero_filtered",
        "source_dataset_root": str(dataset_root),
        "source_matrix_tsv": str(matrix_path),
        "filter": {
            "metric": "nonzero_fraction",
            "comparison": ">",
            "min_nonzero_fraction": float(min_nonzero_fraction),
            "applied_to": "CUT&Tag nearby peak columns only",
        },
        "n_samples": int(filtered_matrix.shape[0]),
        "n_non_peak_columns": int(len(non_peak_columns)),
        "n_source_peak_variables": int(len(peak_columns)),
        "n_retained_peak_variables": int(len(retained_peak_columns)),
        "n_dropped_peak_variables": int(dropped.shape[0]),
        "missing_peak_columns": missing_peak_columns,
        "outputs": {
            "nearby_peak_matrix_tsv": str(filtered_matrix_path),
            "nearby_peak_metadata_tsv": str(filtered_metadata_path),
            "nearby_peak_bins_tsv": str(filtered_bins_path),
            "nearby_candidate_peaks_bed": str(filtered_bed_path),
            "dropped_peaks_tsv": str(dropped_path),
            "peak_filter_metrics_tsv": str(metrics_path),
        },
    }
    json_dump(filtered_summary_path, summary)

    return {
        "gene": gene,
        "source_peak_variables": int(len(peak_columns)),
        "retained_peak_variables": int(len(retained_peak_columns)),
        "dropped_peak_variables": int(dropped.shape[0]),
        "matrix_tsv": str(filtered_matrix_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Filter existing nearby peak datasets by peak nonzero fraction")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--gene-panel", default=None)
    p.add_argument("--genes", default=None, help="Comma-separated gene list. Defaults to --gene-panel or dataset subdirectories.")
    p.add_argument("--min-nonzero-fraction", type=float, default=0.01)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = resolve_path(repo_root, args.dataset_root)
    out_root = resolve_path(repo_root, args.out_root)
    gene_panel = resolve_path(repo_root, args.gene_panel) if args.gene_panel else None
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {args.dataset_root}")
    if out_root is None:
        raise ValueError("--out-root is required")
    if not (0 <= args.min_nonzero_fraction <= 1):
        raise ValueError("--min-nonzero-fraction must be between 0 and 1")

    if args.genes:
        genes = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    else:
        genes = load_panel_genes(gene_panel, allow_none=True)
    if genes is None:
        genes = sorted(path.name.upper() for path in dataset_root.iterdir() if path.is_dir())

    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for gene in genes:
        print(f"[{gene}] filtering nearby peaks by nonzero_fraction > {args.min_nonzero_fraction:g}", flush=True)
        rows.append(
            filter_one_gene(
                dataset_root=dataset_root,
                out_root=out_root,
                gene=gene,
                min_nonzero_fraction=args.min_nonzero_fraction,
            )
        )

    source_panel_summary = dataset_root / "panel_summary.json"
    panel_summary: dict[str, Any] = {}
    if source_panel_summary.exists():
        panel_summary = json.loads(source_panel_summary.read_text())
    panel_summary.update(
        {
            "dataset_filter": {
                "source_dataset_root": str(dataset_root),
                "metric": "nonzero_fraction",
                "comparison": ">",
                "min_nonzero_fraction": float(args.min_nonzero_fraction),
                "applied_to": "CUT&Tag nearby peak columns only",
            },
            "genes": rows,
        }
    )
    json_dump(out_root / "panel_summary.json", panel_summary)
    print(f"Wrote: {out_root / 'panel_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
