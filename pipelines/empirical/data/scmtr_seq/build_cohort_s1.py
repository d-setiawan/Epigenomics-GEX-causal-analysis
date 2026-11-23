"""Frozen unpaired S1 learner inputs and a strictly separate paired evaluator.

The source universe inherits the authors' six-target DNA filter. Our own split
precedes per-assay QC. Feature/panel selection reads only disjoint training cells.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from prepare_s1 import BASE, sha256

ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
PROTOCOL = {
    "split_seed": 2718, "paired_reserve_before_QC": 600,
    "split": "Sort SHA256('2718|partition|' + canonical_id); reserve first 600, round-robin remaining across RNA,H3K27ac,H3K4me3 before QC",
    "RNA_QC": {"counts_min": 1000, "detected_genes_min": 500, "dominant_RT_fraction_strict_min": 0.5, "mitochondrial_fraction_max": 0.05, "valid_RT_tags": ["01", "02", "03", "04"]},
    "histone_QC": {"genomic_fragments_min": 250, "training_uses_only_assigned_assay": True},
    "paired_reference_QC": "All three assay QC conditions; internal complete-case population within 600 reserved source cells",
    "panel": {"genes": 3, "minimum_detected_cells": 20, "training_detection_fraction": {"RNA": 0.10, "H3K27ac": 0.05, "H3K4me3": 0.05}, "hash_prefix": "20261101_causal_panel_v1|", "nonoverlapping_promoters_required": True, "insufficient_eligible_genes": "fail; never relax thresholds"},
    "encoder_features": "All own-training-positive annotation-matched genes, excluding every promoter overlapping selected panel loci and every associated RNA gene-name alias; no PCA or variance ranking",
    "identity_policy": "Independent per-assay/per-partition hash row order and opaque IDs; pairing, RT/day labels, source barcodes and panel counts evaluator-only; learner library_depth sums only retained encoder features",
    "source_QC_caveat": "Published S1 DNA files already used total six-target DNA counts >=2500; this inherited selection cannot be undone using released three-assay inputs",
}


def ordered_hash(values, prefix):
    return sorted(values, key=lambda value: hashlib.sha256((prefix + str(value)).encode()).hexdigest())


def assign_cells(canonical_ids, reserve=600):
    ordered = ordered_hash(canonical_ids, "2718|partition|")
    if len(ordered) <= reserve or len(set(ordered)) != len(ordered):
        raise ValueError("Invalid or insufficient canonical source cells")
    return {cell: ("paired_reserve" if i < reserve else ASSAYS[(i - reserve) % len(ASSAYS)]) for i, cell in enumerate(ordered)}


def assay_qc(cells, assay):
    if assay == "RNA":
        return ((cells.RNA_counts >= 1000) & (cells.RNA_detected_genes >= 500)
                & (cells.dominant_RT_fraction > .5) & (cells.mitochondrial_fraction <= .05)
                & cells.dominant_RT_tag.isin(["01", "02", "03", "04"])).to_numpy()
    return (cells[f"{assay}_counts"] >= 250).to_numpy()


def windows_overlap(left, right):
    return left.chrom == right.chrom and left.start < right.end and right.start < left.end


def select_panel(features, training_matrices):
    eligible = np.ones(len(features), dtype=bool)
    evidence = features[["gene_id", "gene_name", "chrom", "start", "end"]].copy()
    for assay in ASSAYS:
        matrix = training_matrices[assay]
        detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
        fraction = detected / matrix.shape[0] if matrix.shape[0] else np.zeros(len(features))
        eligible &= (detected >= 20) & (fraction >= PROTOCOL["panel"]["training_detection_fraction"][assay])
        evidence[f"{assay}_training_detected_cells"] = detected
        evidence[f"{assay}_training_detection_fraction"] = fraction
    evidence["eligible"] = eligible
    lookup = dict(zip(features.gene_id, features.index))
    selected = []
    for gene in ordered_hash(features.loc[eligible, "gene_id"], PROTOCOL["panel"]["hash_prefix"]):
        candidate = lookup[gene]
        if any(windows_overlap(features.loc[candidate], features.loc[other]) for other in selected):
            continue
        selected.append(candidate)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError(f"Only {len(selected)} eligible nonoverlapping panel genes; do not relax frozen rules")
    excluded = np.zeros(len(features), dtype=bool)
    for index in selected:
        locus = features.loc[index]
        excluded |= ((features.chrom == locus.chrom) & (features.start < locus.end) & (features.end > locus.start)).to_numpy()
    # RNA symbols can have annotation aliases. Remove all names associated with
    # overlapping loci so a second feature name cannot reintroduce a heldout locus.
    alias_columns = [c for c in ("gene_name", "annotation_gene_name") if c in features]
    aliases = set(features.loc[excluded, alias_columns].to_numpy().ravel()) - {"", None}
    for column in alias_columns:
        excluded |= features[column].isin(aliases).to_numpy()
    evidence["selected_panel"] = evidence.index.isin(selected)
    evidence["excluded_from_encoder"] = excluded
    return selected, excluded, evidence


def main():
    source, output = BASE / "full_s1", BASE / "cohort_s1"
    output.mkdir(exist_ok=False)
    learner, evaluator = output / "learner", output / "evaluator"
    learner.mkdir(); evaluator.mkdir()
    (output / "frozen_protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    cells = pd.read_csv(source / "cells.tsv", sep="\t", dtype={"dominant_RT_tag": str})
    features = pd.read_csv(source / "features.tsv", sep="\t")
    matrices = {assay: sparse.load_npz(source / f"{assay}.npz") for assay in ASSAYS}
    if any(matrix.shape != (len(cells), len(features)) for matrix in matrices.values()):
        raise ValueError("Full source dimensions mismatch")
    assignments = assign_cells(cells.canonical_id.tolist())
    assigned = cells.canonical_id.map(assignments)
    qc = {assay: assay_qc(cells, assay) for assay in ASSAYS}
    train_indices = {assay: np.flatnonzero((assigned == assay).to_numpy() & qc[assay]) for assay in ASSAYS}
    if any(len(indices) == 0 for indices in train_indices.values()):
        raise ValueError("At least one training assay has no eligible cells")
    # No paired/test counts are inspected for panel or encoder feature selection.
    training = {assay: matrices[assay][train_indices[assay]] for assay in ASSAYS}
    panel, excluded, evidence = select_panel(features, training)
    evidence.to_csv(evaluator / "training_feature_coverage.tsv", sep="\t", index=False)
    features.loc[panel].to_csv(evaluator / "panel_genes.tsv", sep="\t", index=False)
    paired_mask = (assigned == "paired_reserve").to_numpy() & np.logical_and.reduce(list(qc.values()))
    paired_indices = np.flatnonzero(paired_mask)
    lookup = dict(zip(cells.canonical_id, cells.index))
    pair_map = cells.loc[paired_indices, ["canonical_id", "RNA_raw_barcode", "dominant_RT_tag"]].copy().set_index("canonical_id")
    pair_map["day_evaluator_only"] = pair_map.dominant_RT_tag.map({"01": "D0", "02": "D1", "03": "D2", "04": "D3"})
    summary = {"source_n": len(cells), "reserved_before_QC": int((assigned == "paired_reserve").sum()),
               "paired_reference_n": len(paired_indices), "training": {}, "selected_panel_genes": features.loc[panel, ["gene_id", "gene_name"]].to_dict("records"),
               "panel_excluded_feature_count": int(excluded.sum()), "panel_eligible_gene_count": int(evidence.eligible.sum()), "files": {}}
    for assay in ASSAYS:
        feature_mask = (np.asarray((training[assay] > 0).sum(axis=0)).ravel() > 0) & ~excluded
        kept_features = features.loc[feature_mask].drop(columns=["rna_column"], errors="ignore")
        kept_features.to_csv(learner / f"{assay}_features.tsv", sep="\t", index=False)
        summary["training"][assay] = {"assigned_before_QC": int((assigned == assay).sum()), "retained_after_own_QC": len(train_indices[assay]), "encoder_feature_count": int(feature_mask.sum())}
        for partition, indices in (("train", train_indices[assay]), ("test", paired_indices)):
            ordered_cells = ordered_hash(cells.loc[indices, "canonical_id"], f"2718|{assay}|{partition}|")
            ordered_indices = np.array([lookup[cell] for cell in ordered_cells])
            opaque = [f"{assay}.{partition}.{i:06d}" for i in range(len(ordered_indices))]
            source_depth = cells.loc[ordered_indices, f"{assay}_counts"].to_numpy()
            learner_counts = matrices[assay][ordered_indices][:, feature_mask].tocsr()
            learner_depth = np.asarray(learner_counts.sum(axis=1)).ravel()
            if np.any(learner_depth == 0):
                raise ValueError(f"A {assay} {partition} cell has no encoder counts; freeze handling before fitting")
            pd.DataFrame({"cell_id": opaque, "library_depth": learner_depth}).to_csv(learner / f"{assay}_{partition}_cells.tsv", sep="\t", index=False)
            sparse.save_npz(learner / f"{assay}_{partition}_counts.npz", learner_counts)
            sparse.save_npz(evaluator / f"{assay}_{partition}_panel_counts.npz", matrices[assay][ordered_indices][:, panel].tocsr())
            pd.DataFrame({"cell_id": opaque, "canonical_id": ordered_cells, "library_depth": source_depth}).to_csv(evaluator / f"{assay}_{partition}_map.tsv", sep="\t", index=False)
            if partition == "test":
                pair_map[assay + "_cell_id"] = pd.Series(opaque, index=ordered_cells)
    pair_map.to_csv(evaluator / "heldout_pairs.tsv", sep="\t")
    paired = sparse.hstack([matrices[assay][paired_indices][:, panel] for assay in ASSAYS], format="csr")
    sparse.save_npz(evaluator / "paired_panel_counts.npz", paired)
    cells.loc[paired_indices, ["canonical_id"]].to_csv(evaluator / "paired_panel_rows.tsv", sep="\t", index=False)
    nodes = [{"node_id": f"{assay}|{features.loc[index, 'gene_id']}", "assay": assay, "gene_id": features.loc[index, "gene_id"], "gene_name": features.loc[index, "gene_name"]} for assay in ASSAYS for index in panel]
    pd.DataFrame(nodes).to_csv(evaluator / "panel_nodes.tsv", sep="\t", index=False)
    cells["assigned_before_QC"] = assigned
    for assay in ASSAYS:
        cells[f"{assay}_QC_evaluator_only"] = qc[assay]
    cells["paired_reference_retained"] = paired_mask
    cells.to_csv(evaluator / "source_cell_manifest.tsv", sep="\t", index=False)
    summary["paired_reference_day_counts_evaluator_only"] = pair_map.day_evaluator_only.value_counts().to_dict()
    summary["paired_reference_scope"] = "Internal cell holdout in one pooled-day S1 sublibrary; complete-case QC population, not independent biological replication or causal truth"
    summary["source_provenance_sha256"] = sha256(source / "provenance.json")
    summary["cohort_script_sha256"] = sha256(__file__)
    summary["protocol_sha256"] = sha256(output / "frozen_protocol.json")
    for path in sorted(output.rglob("*")):
        if path.is_file():
            summary["files"][str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
