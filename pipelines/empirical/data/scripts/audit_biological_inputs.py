#!/usr/bin/env python3
"""Read-only inventory/QC of the existing real-data integration inputs.

This does not filter cells, assign biological labels, train integration, or
modify source data. Output describes existing metadata and raw count arrays;
hashtag groups are explicitly not interpreted as donors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def quantiles(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {str(q): float(np.quantile(finite, q)) for q in (0, .1, .5, .9, 1)} if len(finite) else {}


def h5_strings(dataset):
    return dataset.asstr()[:]


def h5_column(node):
    if isinstance(node, h5py.Dataset):
        return h5_strings(node) if node.dtype.kind in "OSU" else node[:]
    categories = h5_column(node["categories"])
    return np.asarray([categories[c] if c >= 0 else None for c in node["codes"][:]], dtype=object)


def file_info(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns}


def scan_csr(group, shape, *, mito=None):
    """Also reads the transposed 10x CSC storage as cell-by-feature CSR."""
    pointer = group["indptr"][:]
    if len(pointer) != shape[0] + 1:
        raise ValueError("Sparse cell axis does not match indptr")
    totals, detected, mitochondrial = [], [], []
    bad = 0
    noninteger = 0
    for start in range(0, shape[0], 512):
        stop = min(start + 512, shape[0])
        lo, hi = int(pointer[start]), int(pointer[stop])
        data = group["data"][lo:hi]
        indices = group["indices"][lo:hi]
        bad += int(np.sum(~np.isfinite(data) | (data < 0)))
        noninteger += int(np.sum(np.isfinite(data) & (data != np.floor(data))))
        matrix = csr_matrix((data, indices, pointer[start:stop + 1] - lo), shape=(stop - start, shape[1]))
        totals.extend(np.asarray(matrix.sum(axis=1)).ravel())
        detected.extend(np.asarray((matrix > 0).sum(axis=1)).ravel())
        if mito is not None:
            mitochondrial.extend(np.asarray(matrix[:, mito].sum(axis=1)).ravel())
    totals = np.asarray(totals, dtype=float)
    result = {"cells": int(shape[0]), "features": int(shape[1]),
              "stored_entries": int(pointer[-1]), "invalid_count_entries": bad,
              "noninteger_count_entries": noninteger,
              "zero_count_cells": int(np.sum(totals == 0)),
              "counts_per_cell": quantiles(totals), "detected_features_per_cell": quantiles(detected)}
    if mito is not None:
        pct = np.divide(100 * np.asarray(mitochondrial), totals, out=np.zeros_like(totals), where=totals > 0)
        result.update(mitochondrial_percent=quantiles(pct), cells_mito_over_20_percent=int(np.sum(pct > 20)),
                      mitochondrial_threshold_is_descriptive_not_filter=True)
    return result


def audit(root):
    manifest = root / "pipelines/empirical/integration/manifests/scglue_input_manifest.tsv"
    marks = pd.read_csv(manifest, sep="\t")
    records = []
    for row in marks.itertuples():
        metadata_path = root / row.clean_cells
        metadata = pd.read_csv(metadata_path, sep="\t")
        record = {"mark": row.mark, "metadata": file_info(metadata_path),
                  "clean_cells": len(metadata), "duplicate_barcodes": int(metadata.barcode.duplicated().sum()),
                  "hashtag_group_counts": metadata.donor_id.value_counts(dropna=False).to_dict(),
                  "hashtag_is_not_verified_donor": True,
                  "fragment_rows_per_cell": quantiles(metadata.n_cuttag_fragments),
                  "available_qc_columns": list(metadata.columns)}
        demux = metadata_path.with_name(f"{row.prefix}_hto_adt_metadata.tsv")
        original = pd.read_csv(demux, sep="\t")
        record["demux_cells"] = len(original)
        record["demux_class_counts"] = original.hto_classification.value_counts(dropna=False).to_dict()
        peaks = pd.read_csv(root / row.peaks_bed, sep="\t", header=None)
        record["peak_count"] = len(peaks)
        record["peak_width_bp"] = quantiles(peaks[2] - peaks[1])
        h5ad = root / f"pipelines/empirical/integration/runs/scglue_joint_v2/train/modalities/chrom_{row.mark}_with_glue.h5ad"
        with h5py.File(h5ad, "r") as handle:
            counts = handle["layers/counts"]
            record["integration_counts"] = scan_csr(counts, tuple(counts.attrs["shape"]))
            obs = handle["obs"]
            record["integration_hashtag_counts"] = pd.Series(h5_column(obs["donor_id"])).value_counts(dropna=False).to_dict()
            record["integration_obs_columns"] = list(obs)
        records.append(record)

    rna_path = root / "pipelines/empirical/data/raw/GexData/GEX_feature_cell_matrix_HDF5.h5"
    with h5py.File(rna_path, "r") as handle:
        raw = handle["matrix"]
        genes = h5_strings(raw["features/name"])
        rna = scan_csr(raw, tuple(raw["shape"][:][::-1]), mito=np.char.startswith(genes.astype(str), "MT-"))
        rna["file"] = file_info(rna_path)
        rna["duplicate_barcodes"] = int(pd.Index(h5_strings(raw["barcodes"])).duplicated().sum())
        rna["feature_types"] = pd.Series(h5_strings(raw["features/feature_type"])).value_counts().to_dict()
    transfer_path = root / "pipelines/empirical/integration/runs/scglue_joint_v2/train/validation/joint_harmonized_label_transfer.tsv"
    transfer = pd.read_csv(transfer_path, sep="\t")
    label_counts = pd.crosstab(transfer.modality_key, transfer.harmonized_coarse)
    rna["existing_annotation_counts"] = label_counts.loc["rna"].to_dict()
    return {"schema": "biological_input_audit.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "existing real data only; no training, filtering, data replacement, or simulation",
            "manifest": file_info(manifest), "chromatin": records, "rna": rna,
            "label_transfer_file": file_info(transfer_path),
            "label_counts_by_modality": label_counts.to_dict(orient="index"),
            "limitations": ["Chromatin transferred labels are not independent biological truth.",
                            "Hashtag classes do not identify donors without a verified sample map.",
                            "pipelines/empirical/integration counts summarize retained bins, not all fragments or FRiP.",
                            "Input paths, size and mtime are recorded; large files are not content-hashed.",
                            "RNA QC uses raw 10x counts; MT fraction is descriptive, not a validated exclusion rule."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = audit(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "chromatin_cells": sum(r["clean_cells"] for r in result["chromatin"]),
                      "rna_cells": result["rna"]["cells"]}, indent=2))


if __name__ == "__main__":
    main()
