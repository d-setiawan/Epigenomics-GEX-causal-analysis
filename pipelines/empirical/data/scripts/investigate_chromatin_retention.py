#!/usr/bin/env python3
"""Read-only retention audit of existing chromatin matrices; no fitting or filtering.

Compare the clean genome-wide fixed-bin MatrixMarket inputs, 30k-bin LSI inputs,
and 15k-bin joint_v2 reconstruction counts. Counts are matrix units, not FRiP or
unique fragments. Inputs are streamed and never modified; outputs are exclusive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix


def quantiles(x):
    return {str(q): float(np.quantile(x, q)) for q in (0, .1, .5, .9, 1)}


def info(path):
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def read_column(node):
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]
    values = read_column(node["categories"])
    return np.asarray([values[i] if i >= 0 else None for i in node["codes"][:]])


def read_h5_counts(path):
    with h5py.File(path, "r") as h:
        group = h["layers/counts"]
        if group.attrs["encoding-type"] != "csr_matrix":
            raise ValueError(f"Not CSR: {path}")
        shape = tuple(group.attrs["shape"])
        matrix = csr_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape)
        matrix.check_format(full_check=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data) & (matrix.data >= 0) & (matrix.data == np.floor(matrix.data))):
            raise ValueError(f"Invalid counts: {path}")
        obs = read_column(h["obs/_index"])
        feature = read_column(h["var/orig_feature"])
        lsi = h["obsm/X_lsi"][:]
    if len(set(obs)) != len(obs) or len(set(feature)) != len(feature):
        raise ValueError(f"Duplicate identities in {path}")
    return {"matrix": matrix, "obs": obs, "features": feature, "lsi": lsi}


def summary(matrix):
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return {"cells": matrix.shape[0], "bins": matrix.shape[1], "nonzero_entries": matrix.nnz,
            "count_sum": float(totals.sum()), "zero_count_rows": int(np.sum(totals == 0)),
            "counts_per_cell": quantiles(totals),
            "detected_bins_per_cell": quantiles(np.diff(matrix.indptr))}


def scan_source(path, n_cells, n_features, selected_source_indices):
    """Stream MatrixMarket COO; retain only the selected 30k-bin sparse submatrix.

    Canonical row/column ordering is checked so per-row occupancy is well-defined
    and duplicate COO entries cannot inflate it. Dense arrays are one-dimensional.
    """
    totals = np.zeros(n_cells, dtype=np.float64)
    detected = np.zeros(n_cells, dtype=np.int64)
    feature_totals = np.zeros(n_features, dtype=np.float64)
    lookup = np.full(n_features, -1, dtype=np.int32)
    lookup[selected_source_indices] = np.arange(len(selected_source_indices))
    keep_rows, keep_cols, keep_data = [], [], []
    seen = 0
    previous_key = -1
    with path.open() as stream:
        header = stream.readline().strip().split()
        if header[:3] != ["%%MatrixMarket", "matrix", "coordinate"] or header[-1] != "general":
            raise ValueError(f"Unsupported MatrixMarket header: {header}")
        for line in stream:
            if not line.startswith("%"):
                dimensions = tuple(map(int, line.split()))
                break
        if dimensions[:2] != (n_cells, n_features):
            raise ValueError("Source dimensions do not match sidecars")
        for chunk in pd.read_csv(stream, sep=r"\s+", names=["row", "col", "value"], chunksize=250000):
            row = chunk.row.to_numpy(dtype=np.int64) - 1
            col = chunk.col.to_numpy(dtype=np.int64) - 1
            value = chunk.value.to_numpy(dtype=np.float64)
            if np.any((row < 0) | (row >= n_cells) | (col < 0) | (col >= n_features)):
                raise ValueError("Source matrix indices out of bounds")
            if not np.all(np.isfinite(value) & (value > 0) & (value == np.floor(value))):
                raise ValueError("Source entries are not positive finite integers")
            keys = row * n_features + col
            if keys[0] <= previous_key or np.any(np.diff(keys) <= 0):
                raise ValueError("Source COO is not unique row-major ordered")
            previous_key = keys[-1]
            totals += np.bincount(row, weights=value, minlength=n_cells)
            detected += np.bincount(row, minlength=n_cells)
            np.add.at(feature_totals, col, value)
            mapped = lookup[col]
            keep = mapped >= 0
            keep_rows.append(row[keep])
            keep_cols.append(mapped[keep])
            keep_data.append(value[keep])
            seen += len(chunk)
    if seen != dimensions[2]:
        raise ValueError("MatrixMarket entry count mismatch")
    subset = coo_matrix((np.concatenate(keep_data), (np.concatenate(keep_rows), np.concatenate(keep_cols))),
                        shape=(n_cells, len(selected_source_indices))).tocsr()
    return totals, detected, feature_totals, subset, seen


def scan_features(path, wanted):
    """Stream millions of labels; retain only 30k selected indices and chromosome spans."""
    indices, chrom_spans, width_counts = {}, {}, Counter()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        first = stream.readline()
        if first.strip() != b"feature":
            raise ValueError("Unexpected feature sidecar header")
        digest.update(first)
        for index, raw in enumerate(stream):
            digest.update(raw)
            feature = raw.decode().strip()
            chrom, position = feature.split(":")
            start, end = map(int, position.split("-"))
            width_counts[end - start] += 1
            if chrom not in chrom_spans:
                chrom_spans[chrom] = [index, index + 1]
            else:
                if chrom_spans[chrom][1] != index:
                    raise ValueError("Non-contiguous chromosome block")
                chrom_spans[chrom][1] = index + 1
            if feature in wanted:
                if feature in indices:
                    raise ValueError("Duplicate selected source feature")
                indices[feature] = index
    if set(indices) != set(wanted):
        raise ValueError("Selected features missing from source matrix")
    return indices, chrom_spans, width_counts, index + 1, digest.hexdigest()


def analyze_mark(root, row, guidance):
    joint = root / "pipelines/empirical/integration/runs/scglue_joint_v2"
    pre_path = joint / f"preprocess/chrom_{row.mark}_preprocessed.h5ad"
    train_path = joint / f"train/modalities/chrom_{row.mark}_with_glue.h5ad"
    pre, train = read_h5_counts(pre_path), read_h5_counts(train_path)
    source_cells = pd.read_csv(root / row.bin_barcodes, sep="\t").barcode.astype(str)
    if source_cells.duplicated().any():
        raise ValueError("Source cell identities are not unique")
    cell_index = pd.Index(source_cells).get_indexer(pre["obs"])
    if np.any(cell_index < 0):
        raise ValueError("Preprocessed cells missing from source")
    indices, spans, widths, n_features, feature_hash = scan_features(root / row.bin_features, set(pre["features"]))
    source_indices = np.asarray([indices[f] for f in pre["features"]])
    totals, detected, feature_totals, subset, entries = scan_source(root / row.bin_mtx, len(source_cells), n_features, source_indices)
    pre_equal = (subset[cell_index] != pre["matrix"]).nnz == 0
    train_cells = pd.Index(pre["obs"]).get_indexer(train["obs"])
    train_features = pd.Index(pre["features"]).get_indexer(train["features"])
    if np.any(train_cells < 0) or np.any(train_features < 0):
        raise ValueError("Training identities absent from preprocess")
    train_equal = (pre["matrix"][train_cells][:, train_features] != train["matrix"]).nnz == 0
    clean = pd.read_csv(root / row.clean_cells, sep="\t")
    stages = {"source_fixed_bins": {"cells": len(source_cells), "bins": n_features,
              "occupied_bins": int(np.count_nonzero(feature_totals)), "nonzero_entries": entries,
              "count_sum": float(totals.sum()), "counts_per_cell": quantiles(totals),
              "detected_bins_per_cell": quantiles(detected), "zero_count_rows": int(np.sum(totals == 0))},
              "preprocess_30k": summary(pre["matrix"]), "train_15k": summary(train["matrix"])}
    for stage, item in (("preprocess_30k", pre), ("train_15k", train)):
        ids = pd.Index(source_cells).get_indexer(item["obs"])
        retained = np.asarray(item["matrix"].sum(axis=1)).ravel()
        valid = totals[ids] > 0
        stages[stage]["fraction_source_count_mass"] = float(retained.sum() / totals.sum())
        stages[stage]["per_retained_cell_fraction_source_counts"] = quantiles(retained[valid] / totals[ids][valid])
        stages[stage]["fraction_source_occupied_bins"] = item["matrix"].shape[1] / np.count_nonzero(feature_totals)
        sums = np.asarray(item["matrix"].sum(axis=0)).ravel()
        top = np.argsort(-sums, kind="stable")[:20]
        stages[stage]["top_20_bins"] = [{"feature": str(item["features"][i]), "counts": float(sums[i])} for i in top]
        chroms = [f.split(":", 1)[0] for f in item["features"]]
        stages[stage]["chromosome_bin_counts"] = dict(Counter(chroms))
        stages[stage]["chromosome_count_mass"] = pd.Series(sums).groupby(chroms).sum().to_dict()
    peak_path = root / row.peaks_bed
    peaks = pd.read_csv(peak_path, sep="\t", header=None)
    log_path = peak_path.with_name(f"{row.prefix}_peaks_peaks.xls")
    peak_header = []
    with log_path.open() as stream:
        for line in stream:
            if line.startswith("#"):
                peak_header.append(line.strip())
            elif line.strip():
                break
    return {"mark": row.mark, "stages": stages,
            "source_counts_per_clean_fragment_row": float(totals.sum() / clean.n_cuttag_fragments.sum()),
            "source_chromosome_count_mass": {c: float(feature_totals[a:b].sum()) for c, (a, b) in spans.items()},
            "source_bin_width_counts": dict(widths), "source_features_sha256": feature_hash,
            "checks": {"preprocess_is_exact_source_subset": bool(pre_equal),
                       "train_is_exact_preprocess_subset": bool(train_equal),
                       "lsi_exactly_carried_to_train": bool(np.array_equal(pre["lsi"][train_cells], train["lsi"])),
                       "lsi_all_finite": bool(np.isfinite(train["lsi"]).all()),
                       "source_coo_unique_row_major": True},
            "source_cells_missing_preprocess": sorted(set(source_cells) - set(pre["obs"])),
            "peak_count": len(peaks), "peak_width_bp": quantiles(peaks[2] - peaks[1]),
            "peak_call_header": peak_header, "guidance_summary": guidance,
            "inputs": [info(root / p) for p in (row.bin_mtx, row.bin_features, row.bin_barcodes, row.clean_cells)] +
                      [info(p) for p in (pre_path, train_path, peak_path, log_path)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    root = args.root.resolve()
    manifest = root / "pipelines/empirical/integration/manifests/scglue_input_manifest.tsv"
    graph_summary = root / "pipelines/empirical/integration/runs/scglue_joint_v2/graph/guidance_summary.json"
    guidance = json.loads(graph_summary.read_text())
    per_mark = {x["mark"]: x for x in guidance["per_mark"]}
    records = []
    for row in pd.read_csv(manifest, sep="\t").itertuples(index=False):
        records.append(analyze_mark(root, row, per_mark[row.mark]))
        print(f"Audited {row.mark}", flush=True)
    result = {"schema": "chromatin_retention.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__,
              "scope": "Existing six-mark fixed-bin integration inputs only; no fitting, filtering or source mutation",
              "manifest": info(manifest), "guidance_file": info(graph_summary), "marks": records,
              "limitations": ["Count retention is not FRiP, biological information retention or integration accuracy.",
                              "Source bin units are not assumed to equal unique fragment rows.",
                              "LSI was generated from 30k bins and carried unchanged into 15k-bin reconstruction.",
                              "Feature sidecars and script are hashed; large matrices are identified by size and mtime only.",
                              "Peak calls are a separate branch from fixed-bin integration; no peak caller was rerun.",
                              "Guidance coverage is read from its saved summary, not recomputed from the graph."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
