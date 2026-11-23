#!/usr/bin/env python3
"""Describe existing ADT markers by transferred label; never assign cell types.

Only the requested JSON output is written, with exclusive creation. The
confidence >= 0.9 subset is a descriptive sensitivity analysis, not a marker
classification threshold or accuracy criterion. No model is fitted.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKS = ("H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3")
MARKERS = ("CD14", "CD16", "CD3-1", "CD19", "CD20", "CD56-1")
TRANSFER_RELATIVE = (
    "pipelines/empirical/integration/runs/scglue_joint_v2/train/validation/"
    "joint_harmonized_label_transfer.tsv"
)
OUTPUT_RELATIVE = "pipelines/empirical/data/outputs/biology_review_2026-09-10/adt_label_audit.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_record(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        f"File changed while hashing: {path}",
    )
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "mtime_utc": datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat(),
        "sha256": digest.hexdigest(),
    }


def group_summary(frame: pd.DataFrame) -> list[dict]:
    groups = []
    for label, group in frame.groupby("harmonized_coarse", dropna=False, sort=True):
        groups.append({
            "harmonized_coarse": None if pd.isna(label) else str(label),
            "n": len(group),
            "coarse_confidence_median": float(group["coarse_confidence"].median()),
            "adt_total_median": float(group["adt_total"].median()),
            "raw_marker_medians": {
                marker: float(group[f"raw_{marker}"].median()) for marker in MARKERS
            },
            "centered_log1p_marker_medians": {
                marker: float(group[f"centered_{marker}"].median()) for marker in MARKERS
            },
        })
    return groups


def audit_assay(mark: str, matrix_path: Path, labels: pd.DataFrame) -> dict:
    before = file_record(matrix_path)
    with matrix_path.open(newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))[1:]
    require(len(header) == len(set(header)), f"Duplicate original ADT barcodes: {mark}")
    cores = [re.sub(r"\.1$", "", barcode) for barcode in header]
    require(len(cores) == len(set(cores)), f"Barcode normalization collision: {mark}")
    require(all(re.fullmatch(r"[ACGT]+", core) for core in cores),
            f"Unexpected ADT barcode format after terminal .1 removal: {mark}")

    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
    require(matrix.columns.tolist() == header, f"ADT header parse mismatch: {mark}")
    require(not matrix.index.duplicated().any(), f"Duplicate ADT features: {mark}")
    require(not matrix.index.isna().any(), f"Missing ADT feature names: {mark}")
    require(set(MARKERS).issubset(matrix.index), f"Missing required marker: {mark}")
    values = matrix.to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"Nonfinite ADT values: {mark}")
    require((values >= 0).all(), f"Negative ADT counts: {mark}")
    require(np.equal(values, np.floor(values)).all(), f"Noninteger ADT counts: {mark}")

    target = labels.loc[labels["mark"].eq(mark)].copy()
    require(len(target) > 0, f"No label rows for assay: {mark}")
    require(not target["barcode_core"].isna().any(), f"Missing label barcode: {mark}")
    require(not target["barcode_core"].duplicated().any(), f"Duplicate label barcode: {mark}")
    confidence = target["coarse_confidence"].to_numpy(dtype=float)
    require(np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all(),
            f"Invalid transfer confidence: {mark}")

    logged = np.log1p(values)
    centered = logged - logged.mean(axis=0, keepdims=True)
    raw_frame = pd.DataFrame(values, index=matrix.index, columns=cores)
    centered_frame = pd.DataFrame(centered, index=matrix.index, columns=cores)
    table = raw_frame.loc[list(MARKERS)].T.add_prefix("raw_")
    table = table.join(centered_frame.loc[list(MARKERS)].T.add_prefix("centered_"),
                       validate="one_to_one")
    table["adt_total"] = values.sum(axis=0)
    table.index.name = "barcode_core"
    joined = target.merge(table, left_on="barcode_core", right_index=True,
                          how="left", validate="one_to_one", indicator=True)
    matched = joined.loc[joined["_merge"].eq("both")].copy()
    unmatched_labels = joined.loc[joined["_merge"].eq("left_only"), "barcode_core"].tolist()
    unmatched_adt = sorted(set(cores) - set(target["barcode_core"]))
    require(len(matched) + len(unmatched_labels) == len(target), "Join accounting mismatch")
    require(len(matched) + len(unmatched_adt) == len(cores), "ADT accounting mismatch")
    sensitivity = matched.loc[matched["coarse_confidence"].ge(0.9)]
    after = file_record(matrix_path)
    require(before == after, f"ADT input changed during audit: {matrix_path}")
    return {
        "mark": mark,
        "input": before,
        "n_adt_features": len(matrix.index),
        "n_adt_cells": len(cores),
        "n_labels": len(target),
        "n_matched_one_to_one": len(matched),
        "n_unmatched_label_rows": len(unmatched_labels),
        "n_unmatched_adt_cells": len(unmatched_adt),
        "unmatched_label_barcodes": unmatched_labels,
        "unmatched_adt_barcodes": unmatched_adt,
        "n_missing_coarse_labels": int(target["harmonized_coarse"].isna().sum()),
        "n_explicit_unknown_labels": int(target["harmonized_coarse"].eq("unknown").sum()),
        "n_unknown_unassigned_unclassified_labels": int(
            target["harmonized_coarse"].fillna("").str.contains(
                "unknown|unassigned|unclassified", case=False, regex=True
            ).sum()
        ),
        "label_source_counts": {
            str(key): int(value) for key, value in
            target["label_source"].fillna("<missing>").value_counts().items()
        },
        "groups_all_matched": group_summary(matched),
        "sensitivity_confidence_ge_0_9": {
            "n": len(sensitivity),
            "groups": group_summary(sensitivity),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    transfer = (args.transfer or root / TRANSFER_RELATIVE).resolve(strict=True)
    output = (args.output or root / OUTPUT_RELATIVE).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    require(output.parent.is_dir(), f"Output parent must already exist: {output.parent}")
    transfer_before = file_record(transfer)
    labels = pd.read_csv(transfer, sep="\t")
    require({"mark", "barcode_core", "harmonized_coarse", "coarse_confidence", "label_source"}
            .issubset(labels.columns), "Transfer table lacks required columns")
    assays = []
    for mark in MARKS:
        paths = list((root / "pipelines/empirical/data" / "outputs" / mark).glob("*_adt_clean_matrix.tsv"))
        require(len(paths) == 1, f"Expected one clean ADT matrix for {mark}, found {len(paths)}")
        assays.append(audit_assay(mark, paths[0], labels))
    require(transfer_before == file_record(transfer), "Transfer input changed during audit")
    payload = {
        "schema_version": "adt-label-descriptive-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": file_record(Path(__file__)),
        "project_root": str(root),
        "runtime": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__},
        "transfer_input": transfer_before,
        "n_transfer_rows_all_modalities": len(labels),
        "methods": {
            "scope": "Read-only descriptive diagnostic of existing ADT counts and transferred labels",
            "markers": list(MARKERS),
            "barcode_join": "Within assay only; remove terminal .1 from ADT barcode; join barcode_core one-to-one",
            "normalization": "log1p(count) minus per-cell mean log1p(count) across ALL ADT features, including controls",
            "aggregation": "Median within transferred harmonized_coarse groups; all matched rows retained",
            "sensitivity": "Same summaries restricted to existing coarse_confidence >= 0.9; descriptive only",
            "interpretation": "No new labels, accuracy estimates, marker thresholds, hypothesis tests, or model fitting",
            "limitations": [
                "ADTs are an orthogonal measurement channel, not independent label truth.",
                "Background, sparse counts, antibody specificity, doublets and preprocessing can affect marker summaries.",
                "Transfer confidence is neighbor agreement, not validated biological correctness.",
                "Centered log1p is explicitly defined here; it is not a claim to reproduce Seurat CLR normalization.",
                "Unknown labels are retained; unmatched rows are counted and not included in marker medians.",
            ],
        },
        "totals": {
            key: sum(assay[key] for assay in assays) for key in (
                "n_adt_cells", "n_labels", "n_matched_one_to_one", "n_unmatched_label_rows",
                "n_unmatched_adt_cells", "n_missing_coarse_labels", "n_explicit_unknown_labels"
            )
        },
        "assays": assays,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(json.dumps({"output": str(output), "totals": payload["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
