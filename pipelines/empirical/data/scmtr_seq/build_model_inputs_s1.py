"""Bridge only checksum-verified learner NPZ/TSV files into minimal AnnData."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

BASE = Path(__file__).resolve().parent
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
ANNOTATIONS = ["gene_id", "gene_name", "annotation_gene_name", "chrom", "tss", "strand", "start", "end", "gene_type"]


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ordered_sha256(values):
    return hashlib.sha256(("\n".join(str(value) for value in values) + "\n").encode()).hexdigest()


def convert_one(cohort, output, source_summary, assay, partition):
    relatives = {"counts": f"learner/{assay}_{partition}_counts.npz",
                 "cells": f"learner/{assay}_{partition}_cells.tsv",
                 "features": f"learner/{assay}_features.tsv"}
    checked_sources = {}
    for role, relative in relatives.items():
        path = cohort / relative
        expected = source_summary["files"][relative]
        digest = file_sha256(path)
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Frozen learner checksum mismatch: {relative}")
        checked_sources[role] = {"path": relative, "sha256": digest, "bytes": path.stat().st_size}
    counts = sparse.load_npz(cohort / relatives["counts"]).tocsr()
    cells = pd.read_csv(cohort / relatives["cells"], sep="\t")
    features = pd.read_csv(cohort / relatives["features"], sep="\t")
    if counts.shape != (len(cells), len(features)) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts.data < 0):
        raise ValueError("Learner counts are not dimensionally valid raw nonnegative integers")
    ids = cells.cell_id.astype(str).tolist()
    if len(set(ids)) != len(ids) or any(re.fullmatch(fr"{assay}\.{partition}\.\d{{6}}", value) is None for value in ids):
        raise ValueError("Expected unique opaque assay-specific learner IDs")
    if features.gene_id.duplicated().any() or features.gene_id.isna().any():
        raise ValueError("Expected unique annotated gene IDs")
    var = features.loc[:, [column for column in ANNOTATIONS if column in features]].copy()
    var.index = pd.Index([f"{assay}|{gene}" for gene in features.gene_id], name="feature_id")
    obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
    data = ad.AnnData(X=counts, obs=obs, var=var)
    destination = output / partition / f"{assay}.h5ad"
    if destination.exists():
        raise FileExistsError(destination)
    data.write_h5ad(destination, compression="gzip")
    restored = ad.read_h5ad(destination)
    if restored.shape != counts.shape or not np.issubdtype(restored.X.dtype, np.integer) or (restored.X != counts).nnz:
        raise ValueError("H5AD integer count roundtrip failed")
    if restored.obs_names.tolist() != ids or restored.var_names.tolist() != var.index.tolist():
        raise ValueError("H5AD ordered cell/feature identity roundtrip failed")
    pd.testing.assert_frame_equal(restored.var, var, check_dtype=False, check_categorical=False)
    if len(restored.obs.columns) or restored.uns or restored.obsm or restored.varm or restored.obsp or restored.varp or restored.layers or restored.raw is not None:
        raise ValueError("H5AD carries unexpected metadata or alternate data representations")
    return {"path": str(destination.relative_to(output)), "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size, "shape": list(counts.shape), "nnz": counts.nnz,
            "X_dtype": str(counts.dtype), "ordered_cell_ids_sha256": ordered_sha256(ids),
            "ordered_feature_ids_sha256": ordered_sha256(var.index), "ordered_gene_ids_sha256": ordered_sha256(var.gene_id),
            "var_columns": var.columns.tolist(), "obs_columns": [], "uns_keys": [],
            "source_files": checked_sources, "roundtrip_exact_counts_ids_gene_annotations": True}


def main():
    cohort = BASE / "cohort_s1"
    source_summary_path = cohort / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    output = cohort / "model_inputs"
    output.mkdir(exist_ok=False)
    for partition in ("train", "test"):
        (output / partition).mkdir()
    manifest = {"source_summary_path": "../summary.json", "source_summary_sha256": file_sha256(source_summary_path),
                "source_protocol_sha256": source_summary["protocol_sha256"],
                "helper_path": str(Path(__file__).resolve()), "helper_sha256": file_sha256(__file__),
                "python": sys.executable, "versions": {package: version(package) for package in ("anndata", "numpy", "pandas", "scipy", "h5py")},
                "policy": "Only learner count/cell/feature files and the checksum summary are read; no evaluator files are read. Raw integer CSR counts in X, opaque obs index only, fixed gene annotation in var; no obs columns, uns, obsm, varm, pairwise slots, layers or raw. Source library_depth and source feature_type are deliberately omitted.", "artifacts": {}}
    for partition in ("train", "test"):
        for assay in ASSAYS:
            record = convert_one(cohort, output, source_summary, assay, partition)
            manifest["artifacts"][f"{partition}/{assay}"] = record
            print(json.dumps({"path": record["path"], "shape": record["shape"], "roundtrip": True}), flush=True)
    if file_sha256(source_summary_path) != manifest["source_summary_sha256"]:
        raise ValueError("Source summary changed during conversion")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
