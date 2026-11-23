#!/usr/bin/env python3
"""Read-only RNA barcode, annotation and marker/QC audit of existing inputs.

No labels are inferred or changed. Marker expression is descriptive, not an
independent accuracy estimate: these same expression data generated annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


PANELS = {
    "T": ["CD3D", "CD3E", "TRAC"],
    "B": ["MS4A1", "CD79A", "CD37"],
    "myeloid": ["LST1", "LYZ", "S100A8", "S100A9", "CD14"],
    "nonclassical_myeloid": ["FCGR3A", "MS4A7", "CTSS", "TYROBP", "CSF1R", "CX3CR1", "AIF1"],
    "microglia_review": ["TMEM119", "P2RY12", "TREM2", "C1QA", "C1QB", "C1QC", "APOE", "HEXB"],
    "NK": ["NKG7", "GNLY"],
    "plasma": ["JCHAIN", "MZB1", "SDC1"],
    "DC": ["FCER1A", "CLEC10A", "CD1C"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strings(node):
    return node.asstr()[:].astype(str)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {str(q): float(np.quantile(values, q)) for q in (0, .1, .5, .9, 1)} if len(values) else {}


def provenance(path, *, content_hash=False):
    result = {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    if content_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def read_counts(path):
    """Stream 10x feature-by-cell CSC as its cell-by-feature CSR transpose."""
    genes_requested = list(dict.fromkeys(gene for panel in PANELS.values() for gene in panel))
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        n_genes, n_cells = map(int, group["shape"][:])
        barcodes = strings(group["barcodes"])
        genes = strings(group["features/name"])
        require(len(set(barcodes)) == n_cells, "Raw barcodes are not unique")
        require(len(genes) == n_genes, "Gene axis does not match matrix")
        pointer = group["indptr"][:]
        require(len(pointer) == n_cells + 1 and pointer[0] == 0 and np.all(np.diff(pointer) >= 0),
                "Invalid CSC pointers")
        require(pointer[-1] == len(group["data"]) == len(group["indices"]), "Sparse length mismatch")
        marker_indices = {gene: np.flatnonzero(genes == gene) for gene in genes_requested}
        require(all(len(indices) <= 1 for indices in marker_indices.values()), "Ambiguous marker symbols")
        present = [gene for gene, indices in marker_indices.items() if len(indices)]
        columns = np.array([marker_indices[gene][0] for gene in present])
        mito = np.char.startswith(genes, "MT-")
        totals = np.zeros(n_cells, dtype=np.float64)
        detected = np.zeros(n_cells, dtype=np.int64)
        mt = np.zeros(n_cells, dtype=np.float64)
        marker_counts = np.zeros((n_cells, len(present)), dtype=np.float64)
        invalid = noninteger = 0
        for start in range(0, n_cells, 256):
            stop = min(n_cells, start + 256)
            low, high = int(pointer[start]), int(pointer[stop])
            data, indices = group["data"][low:high], group["indices"][low:high]
            require(np.all((indices >= 0) & (indices < n_genes)), "Sparse feature index out of range")
            invalid += int(np.sum(~np.isfinite(data) | (data < 0)))
            noninteger += int(np.sum(np.isfinite(data) & (data != np.floor(data))))
            matrix = csr_matrix((data, indices, pointer[start:stop + 1] - low),
                                shape=(stop - start, n_genes))
            matrix.sum_duplicates()
            totals[start:stop] = np.asarray(matrix.sum(axis=1)).ravel()
            detected[start:stop] = np.asarray((matrix > 0).sum(axis=1)).ravel()
            mt[start:stop] = np.asarray(matrix[:, mito].sum(axis=1)).ravel()
            marker_counts[start:stop] = matrix[:, columns].toarray()
        require(invalid == 0 and noninteger == 0, "Counts contain invalid or fractional entries")
        require(np.all(totals > 0), "Raw zero-count cells need explicit handling")
    qc = pd.DataFrame({"total_umis": totals, "detected_genes": detected,
                       "mitochondrial_percent": 100 * mt / totals}, index=barcodes)
    counts = pd.DataFrame(marker_counts, index=barcodes, columns=present)
    lognorm = np.log1p(counts.div(totals, axis=0) * 1e4)
    return qc, counts, lognorm, {"cells": n_cells, "features": n_genes,
        "mitochondrial_feature_count": int(mito.sum()), "markers_missing": sorted(set(genes_requested) - set(present)),
        "invalid_count_entries": invalid, "noninteger_count_entries": noninteger}


def summarize_subset(index, qc, counts, lognorm):
    sub = qc.loc[index]
    return {"cells": len(index), "total_umis": quantiles(sub.total_umis),
            "detected_genes": quantiles(sub.detected_genes),
            "mitochondrial_percent": quantiles(sub.mitochondrial_percent),
            "cells_mito_over_20_percent": int((sub.mitochondrial_percent > 20).sum()),
            "markers": {gene: {"detected_cells": int((counts.loc[index, gene] > 0).sum()),
                                "median_raw_count": float(counts.loc[index, gene].median()),
                                "median_log1p_10k": float(lognorm.loc[index, gene].median()),
                                "mean_log1p_10k": float(lognorm.loc[index, gene].mean())}
                        for gene in counts.columns},
            "panel_mean_log1p_10k_score_median": {
                panel: float(lognorm.loc[index, [g for g in genes if g in lognorm]].mean(axis=1).median())
                for panel, genes in PANELS.items()}}


def investigate(root):
    raw = root / "pipelines/empirical/data/raw/GexData/GEX_feature_cell_matrix_HDF5.h5"
    annotations_path = root / "pipelines/empirical/data/raw/GexData/cell_type_annotation.tar.gz"
    harmonization_path = root / "pipelines/empirical/integration/manifests/rna_label_harmonization.tsv"
    trained = root / "pipelines/empirical/integration/runs/scglue_joint_v2/train/modalities/rna_with_glue.h5ad"
    transferred_path = root / "pipelines/empirical/integration/runs/scglue_joint_v2/train/validation/joint_harmonized_label_transfer.tsv"
    with tarfile.open(annotations_path, "r:gz") as archive:
        with archive.extractfile("cell_types/cell_types.csv") as stream:
            annotations = pd.read_csv(stream)
        with archive.extractfile("cell_types/cell_annotation_results.json.gz") as stream:
            annotation_results = json.loads(gzip.decompress(stream.read()))
    require(not annotations.barcode.duplicated().any(), "Duplicate exact annotation barcode")
    raw_annotation_columns = list(annotations.columns)
    harmonization = pd.read_csv(harmonization_path, sep="\t")
    require(not harmonization.duplicated(["coarse_cell_type", "fine_cell_type"]).any(), "Duplicate mapping key")
    annotations = annotations.merge(harmonization, on=["coarse_cell_type", "fine_cell_type"],
                                    how="left", validate="many_to_one").set_index("barcode")
    require(annotations.harmonized_coarse.notna().all() and annotations.harmonized_fine.notna().all(),
            "Missing harmonization mapping")
    transferred = pd.read_csv(transferred_path, sep="\t")
    transferred = transferred.loc[transferred.modality_key == "rna"].set_index("cell")
    require(not transferred.index.duplicated().any(), "Duplicate retained RNA label barcode")
    with h5py.File(trained, "r") as handle:
        retained = pd.Index(strings(handle["obs/_index"]))
        require(not retained.duplicated().any(), "Duplicate retained count barcode")
    qc, counts, lognorm, count_info = read_counts(raw)
    require(set(qc.index) == set(annotations.index), "Original annotation/raw count exact barcode sets differ")
    require(set(retained) == set(transferred.index), "Retained count/label barcode sets differ")
    require(set(retained) <= set(qc.index), "Retained barcode missing in raw counts")
    mismatch = {key: int((transferred[key] != annotations.loc[transferred.index, key]).sum())
                for key in ["harmonized_coarse", "harmonized_fine"]}
    dropped = qc.index[~qc.index.isin(retained)]
    groups = {label: summarize_subset(group.index, qc, counts, lognorm)
              for label, group in transferred.groupby("harmonized_coarse", sort=True)}
    micro = transferred.index[transferred.harmonized_coarse.eq("microglial cell")]
    result_by_barcode = {row["barcode"]: row for row in annotation_results}
    require(len(result_by_barcode) == len(annotation_results), "Duplicate result JSON barcode")
    require(set(result_by_barcode) == set(qc.index), "Annotation result JSON barcode set differs")
    result_label_mismatch = 0
    for barcode, row in annotations.iterrows():
        matches = result_by_barcode[barcode].get("matches", [])
        require(len(matches) > 0, "Empty source annotation matches")
        result_label_mismatch += int(matches[0]["cell_type"] != row.fine_cell_type)
    micro_cells = []
    for barcode in micro:
        matches = result_by_barcode[barcode]["matches"]
        micro_cells.append({"barcode": barcode, "qc": qc.loc[barcode].to_dict(),
            "original_coarse": annotations.loc[barcode, "coarse_cell_type"],
            "original_fine": annotations.loc[barcode, "fine_cell_type"],
            "raw_marker_counts": counts.loc[barcode].to_dict(),
            "annotation_top_matches": [{"cell_type": match["cell_type"],
                "cell_count_in_model": match["cell_count_in_model"]} for match in matches[:5]]})
    return {"schema": "existing_rna_annotation_investigation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_provenance": [provenance(p, content_hash=p != trained) for p in
                             [raw, annotations_path, harmonization_path, trained, transferred_path]],
        "scope": "Existing data only; no training, relabeling, filtering or replacement",
        "count_structure": count_info,
        "method": {"barcode_join": "Exact barcode including suffix; no stripping or normalization",
            "normalization": "Per-cell log1p(10000 * marker UMI count / all raw gene UMIs)",
            "panel_score": "Mean log1p_10k across listed present panel genes, then median within existing label",
            "marker_panels": PANELS, "mitochondrial_rule": "Gene symbols beginning MT-; >20% is descriptive, not a filter",
            "annotation_archive_members": ["cell_types/cell_types.csv", "cell_types/cell_annotation_results.json.gz"],
            "original_annotation_columns": raw_annotation_columns},
        "joins": {"exact_raw_annotation_set_match": True, "exact_retained_label_set_match": True,
                  "retained_in_raw": True, "raw_cells": len(qc), "retained_cells": len(retained),
                  "dropped_cells": len(dropped), "retained_label_source_counts": transferred.label_source.value_counts().to_dict(),
                  "harmonization_to_saved_label_mismatches": mismatch,
                  "source_json_first_match_to_csv_fine_mismatches": result_label_mismatch},
        "raw": summarize_subset(qc.index, qc, counts, lognorm),
        "retained": summarize_subset(retained, qc, counts, lognorm),
        "dropped": summarize_subset(dropped, qc, counts, lognorm),
        "retention_equals_at_least_200_raw_detected_genes": set(retained) == set(qc.index[qc.detected_genes >= 200]),
        "retained_by_label": groups,
        "microglial_labeled_cells": micro_cells,
        "retained_mito_over_20_percent_by_label": transferred.loc[qc.loc[transferred.index].mitochondrial_percent > 20,
                                                                  "harmonized_coarse"].value_counts().to_dict(),
        "limitations": ["Expression checks are not independent label validation because source labels used expression.",
            "Marker panels are descriptive review aids, not a classifier or validated threshold set.",
            "Zero expression may reflect dropout; neither absent markers nor cross-lineage expression proves identity.",
            "cell_count_in_model values are source match support counts, not calibrated probabilities.",
            "No donor identities, doublet calls, new labels or automatic exclusions are inferred.",
            "The integration H5AD is read for barcode membership, not its matrix or embeddings."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"Refusing to overwrite {args.out}")
    result = investigate(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "joins": result["joins"],
                      "retained_mito_over_20": result["retained"]["cells_mito_over_20_percent"]}, indent=2))


if __name__ == "__main__":
    main()
