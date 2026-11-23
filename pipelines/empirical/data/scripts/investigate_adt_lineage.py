#!/usr/bin/env python3
"""Read-only, exact-count lineage checks for existing ADT/HTO and GLUE inputs.

No labels are assigned, no model is trained, and only an exclusive-create JSON
report is written. Barcode normalization is strictly within each assay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import io, sparse


ROOT = Path(__file__).resolve().parents[4]
TRAIN = Path("pipelines/empirical/integration/runs/scglue_joint_v2/train")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(values):
    originals = list(map(str, values))
    cores = [re.sub(r"(?:\.1|-1)$", "", value) for value in originals]
    require(len(set(originals)) == len(originals), "Duplicate original barcodes")
    require(len(set(cores)) == len(cores), "Normalized barcode collision")
    require(all(re.fullmatch("[ACGT]+", core) for core in cores), "Unexpected barcode")
    return cores


def file_record(path, full_hash=True):
    stat = path.stat()
    result = {"path": str(path.resolve()), "size_bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if full_hash:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def matrix_frame(path):
    # Original source exports have no feature-column header and use whitespace.
    # Clean exports explicitly start with 'feature' and use tabs.
    with path.open() as handle:
        header = handle.readline().split()
        if header[0] == "feature":
            header = header[1:]
        cores = normalize(header)
        names, arrays = [], []
        for line in handle:
            fields = line.split()
            require(len(fields) == len(header) + 1, f"Invalid matrix row in {path}")
            names.append(fields[0])
            arrays.append(np.asarray(fields[1:], dtype=np.float64))
    require(len(names) == len(set(names)), f"Duplicate features in {path}")
    values = np.asarray(arrays)
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid counts")
    require(np.equal(values, np.floor(values)).all(), "Noninteger counts")
    return pd.DataFrame(values, index=names, columns=cores), header


def decode(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else v for v in values])


def h5_column(node):
    if isinstance(node, h5py.Dataset):
        return decode(node[:])
    require("codes" in node and "categories" in node, "Unsupported HDF5 column")
    codes, categories = node["codes"][:], decode(node["categories"][:])
    return np.asarray([categories[x] if x >= 0 else None for x in codes], dtype=object)


def h5_frame(group):
    index = h5_column(group[group.attrs["_index"]])
    return pd.DataFrame({name: h5_column(group[name])
                         for name in group.attrs["column-order"]}, index=index)


def disagreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape, "Compared arrays differ in shape")
    if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
        return int(np.count_nonzero(~np.isclose(a, b, rtol=1e-12, atol=1e-12,
                                              equal_nan=True)))
    return int(np.count_nonzero(a.astype(str) != b.astype(str)))


def order_check(left, right):
    left, right = list(left), list(right)
    require(len(set(left)) == len(left) and len(set(right)) == len(right),
            "Duplicate identities in order check")
    shared = set(left) & set(right)
    filtered_left = [x for x in left if x in shared]
    filtered_right = [x for x in right if x in shared]
    return {"n_left": len(left), "n_right": len(right), "n_shared": len(shared),
            "left_only": sorted(set(left) - set(right)),
            "right_only": sorted(set(right) - set(left)),
            "shared_relative_order_equal": filtered_left == filtered_right}


def compare_adt_hto(root, mark, prefix, clean_metadata, inputs):
    result = {}
    for kind in ("ADT", "HTO"):
        raw_path = root / "pipelines/empirical/data/raw" / mark / f"{prefix}_{kind}_counts.tsv"
        clean_path = root / "pipelines/empirical/data/outputs" / mark / f"{prefix}_{kind.lower()}_clean_matrix.tsv"
        inputs.extend([file_record(raw_path), file_record(clean_path)])
        raw, raw_header = matrix_frame(raw_path)
        clean, clean_header = matrix_frame(clean_path)
        require(set(clean.columns).issubset(raw.columns), "Clean barcode absent from raw")
        require(set(clean.index) == set(raw.index), "Clean feature set differs from raw")
        aligned = raw.loc[clean.index, clean.columns]
        difference = aligned.to_numpy() != clean.to_numpy()
        totals = raw.sum(axis=0)
        result[kind.lower()] = {
            "n_raw_cells": raw.shape[1], "n_clean_cells": clean.shape[1],
            "n_features": clean.shape[0],
            "n_compared_counts": int(difference.size),
            "n_mismatched_counts": int(difference.sum()),
            "n_cells_with_any_count_mismatch": int(difference.any(axis=0).sum()),
            "feature_order_equal": raw.index.tolist() == clean.index.tolist(),
            "raw_header_suffixes": sorted(set(re.sub("[ACGT]+", "", x) for x in raw_header)),
            "clean_header_suffixes": sorted(set(re.sub("[ACGT]+", "", x) for x in clean_header)),
            "raw_clean_barcode_order": order_check(raw.columns, clean.columns),
            "clean_metadata_barcode_order": order_check(clean.columns, clean_metadata.index),
            "n_clean_metadata_total_mismatches": disagreement(
                totals.loc[clean_metadata.index], clean_metadata[f"{kind.lower()}_total"]),
        }
        if kind == "ADT":
            result["adt"]["feature_names"] = raw.index.tolist()
            adt_totals = totals
        else:
            hto_totals = totals
    demux_path = root / "pipelines/empirical/data/outputs" / mark / f"{prefix}_hto_adt_metadata.tsv"
    inputs.append(file_record(demux_path))
    demux = pd.read_csv(demux_path, sep="\t").set_index("barcode")
    normalize(demux.index)
    require(set(demux.index).issubset(adt_totals.index), "Demux barcode absent from ADT")
    require(set(demux.index).issubset(hto_totals.index), "Demux barcode absent from HTO")
    result["demux"] = {
        "n_cells": len(demux),
        "adt_total_mismatches": disagreement(adt_totals.loc[demux.index], demux.adt_total),
        "hto_total_mismatches": disagreement(hto_totals.loc[demux.index], demux.hto_total),
        "clean_metadata_column_mismatches": {
            col: disagreement(demux.loc[clean_metadata.index, col], clean_metadata[col])
            for col in demux.columns if col in clean_metadata.columns
        },
        "hashtag_label_note": "donor_id is the stored HTO classification, not verified donor identity",
    }
    return result


def count_layer_check(h5, row, root, inputs):
    """Compare every stored selected count with the source cell x bin MTX."""
    barcodes_path, features_path, matrix_path = [root / row[key] for key in
                                                ("bin_barcodes", "bin_features", "bin_mtx")]
    inputs.extend(file_record(p, full_hash=False) for p in
                  (barcodes_path, features_path, matrix_path))
    barcodes = pd.read_csv(barcodes_path, sep="\t").barcode.tolist()
    normalize(barcodes)
    obs_names = h5_column(h5["obs/_index"])
    row_lookup = {barcode: i for i, barcode in enumerate(barcodes)}
    row_indices = [row_lookup[x] for x in obs_names]
    selected = h5_column(h5["var/orig_feature"])
    require(len(selected) == len(set(selected)), "Duplicate selected bin identifiers")
    selected_set = set(selected)
    # Millions of source bins: stream names instead of materializing a large table.
    column_lookup = {}
    with features_path.open() as handle:
        require(handle.readline().strip() == "feature", "Unexpected feature header")
        for i, line in enumerate(handle):
            name = line.strip()
            if name in selected_set:
                require(name not in column_lookup, "Duplicate selected source bin")
                column_lookup[name] = i
        n_source_features = i + 1
    require(selected_set == set(column_lookup), "Selected bin missing from source")
    original = io.mmread(matrix_path).tocsr()
    require(original.shape == (len(barcodes), n_source_features), "MTX identity dimensions mismatch")
    source_subset = original[row_indices][:, [column_lookup[x] for x in selected]]
    del original
    node = h5["layers/counts"]
    require(node.attrs["encoding-type"] == "csr_matrix", "Expected CSR counts")
    stored = sparse.csr_matrix((node["data"][:], node["indices"][:], node["indptr"][:]),
                               shape=tuple(node.attrs["shape"]))
    require(stored.shape == source_subset.shape, "Count comparison shape mismatch")
    diff = stored.astype(np.float64) - source_subset.astype(np.float64)
    diff.eliminate_zeros()
    return {"shape": list(stored.shape), "n_source_bins": n_source_features,
            "source_to_h5_barcode_order": order_check(barcodes, obs_names),
            "n_mismatched_count_entries": int(diff.nnz),
            "n_cells_with_any_count_mismatch": int(np.count_nonzero(np.diff(diff.indptr))),
            "maximum_absolute_count_difference": float(np.max(np.abs(diff.data))) if diff.nnz else 0.0}


def integration_check(root, key, all_embeddings, labels, clean_metadata, row, inputs):
    h5_path = root / TRAIN / "modalities" / f"{key}_with_glue.h5ad"
    embed_path = root / TRAIN / "modalities" / f"{key}_glue_embeddings.tsv"
    inputs.extend([file_record(h5_path, full_hash=False), file_record(embed_path)])
    modality = pd.read_csv(embed_path, sep="\t").set_index("cell")
    merged = all_embeddings.loc[all_embeddings.modality_key.eq(key)].set_index("cell")
    target = labels.loc[labels.modality_key.eq(key)].set_index("cell")
    with h5py.File(h5_path, "r") as handle:
        obs = h5_frame(handle["obs"])
        require(obs.index.is_unique, "Duplicate H5 observation identities")
        embedding = handle["obsm/X_glue"][:]
        require(np.isfinite(embedding).all(), "Nonfinite embeddings")
        columns = [f"GLUE_{i+1}" for i in range(embedding.shape[1])]
        require(set(obs.index) == set(modality.index) == set(merged.index) == set(target.index),
                "H5/embedding/label identity set mismatch")
        # TSV writers round floating-point representations: compare within
        # fixed 1e-6 tolerance, and also record the unrounded maximum error.
        embedding_comparisons = {}
        for name, frame in (("modality_tsv", modality), ("combined_tsv", merged)):
            delta = np.abs(embedding - frame.loc[obs.index, columns].to_numpy())
            embedding_comparisons[name] = {
                "maximum_absolute_difference": float(delta.max()),
                "entries_differing_by_more_than_1e_6": int((delta > 1e-6).sum()),
                "barcode_order_equal": obs.index.tolist() == frame.index.tolist(),
            }
        result = {"n_cells": len(obs), "embedding_comparisons": embedding_comparisons,
                  "label_barcode_order_equal": obs.index.tolist() == target.index.tolist(),
                  "label_barcode_core_mismatches": disagreement(normalize(obs.index),
                                                                 target.loc[obs.index].barcode_core),
                  "label_obs_id_mismatches": disagreement(
                      [f"{key}::{x}" for x in obs.index], target.loc[obs.index].obs_id)}
        if clean_metadata is not None:
            clean_aligned = clean_metadata.loc[normalize(obs.index)]
            result["h5_clean_metadata_column_mismatches"] = {
                col: disagreement(obs[col], clean_aligned[col])
                for col in clean_metadata.columns if col in obs.columns}
            result["count_layer_source_check"] = count_layer_check(handle, row, root, inputs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    output = (args.output or root / "pipelines/empirical/data/outputs/biology_investigation_2026-09-11/adt_lineage.json").resolve()
    require(not output.exists(), f"Refusing to overwrite {output}")
    manifest_path = root / "pipelines/empirical/integration/manifests/scglue_input_manifest.tsv"
    labels_path = root / TRAIN / "validation/joint_harmonized_label_transfer.tsv"
    embed_path = root / TRAIN / "all_cells_glue_embeddings.tsv"
    inputs = [file_record(p) for p in (manifest_path, labels_path, embed_path)]
    manifest = pd.read_csv(manifest_path, sep="\t")
    labels = pd.read_csv(labels_path, sep="\t")
    all_embeddings = pd.read_csv(embed_path, sep="\t")
    assays = []
    for row in manifest.to_dict("records"):
        mark, prefix = row["mark"], row["prefix"]
        clean_path = root / row["clean_cells"]
        inputs.append(file_record(clean_path))
        clean = pd.read_csv(clean_path, sep="\t").set_index("barcode")
        normalize(clean.index)
        result = {"mark": mark, "prefix": prefix}
        result.update(compare_adt_hto(root, mark, prefix, clean, inputs))
        result["integration"] = integration_check(root, f"chrom_{mark}", all_embeddings,
                                                   labels, clean, row, inputs)
        assays.append(result)
        print(json.dumps({"mark": mark, "adt_count_mismatches": result["adt"]["n_mismatched_counts"],
                          "hto_count_mismatches": result["hto"]["n_mismatched_counts"],
                          "chromatin_count_mismatches": result["integration"]["count_layer_source_check"]["n_mismatched_count_entries"]}), flush=True)
    rna = integration_check(root, "rna", all_embeddings, labels, None, None, inputs)
    for record in inputs:
        current = Path(record["path"]).stat()
        require((current.st_size, current.st_mtime_ns) == (record["size_bytes"], record["mtime_ns"]),
                f"Input changed during audit: {record['path']}")
    result = {"schema_version": "adt-lineage-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script": file_record(Path(__file__)), "inputs": inputs,
              "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                          "scipy": scipy.__version__, "h5py": h5py.__version__},
              "scope": "Existing local counts and identities only; no label assignment, model fitting or production writes",
              "limits": ["Does not certify original experimental antibody/feature annotations or biological donor identity.",
                         "Count equality tests local lineage, not biological alignment accuracy.",
                         "MTX and H5 large inputs have size/mtime provenance, not full-file hashes.",
                         "RNA embedding/label identity checked; RNA raw-to-count-layer values are not checked here."],
              "assays": assays, "rna_integration": rna}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
