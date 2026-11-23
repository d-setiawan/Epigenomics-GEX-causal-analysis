"""Prepare the fixed S1 source population and annotation-defined promoter counts.

This stage retains canonical identities and is evaluator/preparation-only. The
separate cohort builder removes these identities from learner inputs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
MARKS = ("H3K27ac", "H3K4me3")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_id(barcode, rna=False):
    parts = barcode.split(":")
    if len(parts) != 4 or any(not re.fullmatch(r"\d{2}", p) or not 1 <= int(p) <= 48 for p in parts[:3]):
        raise ValueError(f"Invalid split-pool barcode: {barcode}")
    if (rna and not re.fullmatch(r"\d{2}", parts[3])) or (not rna and parts[3] != "S1"):
        raise ValueError(f"Invalid S1 assay suffix: {barcode}")
    return ":".join(parts[:3]) + ":S1"


def collapse_rna(matrix, barcodes, features):
    """Choose the largest RT-tag row; flag ambiguous/invalid tags for later RNA QC.

    The denominator includes every observed RT tag. Ties cannot pass the >0.5
    fraction threshold. This does not sum different day-tag rows into a cell.
    """
    matrix = sparse.csr_matrix(matrix)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape != (len(barcodes), len(features)) or len(set(barcodes)) != len(barcodes):
        raise ValueError("RNA dimensions or barcode uniqueness failed")
    if np.any(matrix.data < 0) or np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError("RNA values must be nonnegative integer counts")
    if np.any(matrix.data > np.iinfo(np.int32).max):
        raise ValueError("RNA counts exceed int32 storage")
    matrix = matrix.astype(np.int32)
    keys = np.array([canonical_id(b, rna=True) for b in barcodes])
    unique, inverse = np.unique(keys, return_inverse=True)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    order = np.lexsort((np.arange(len(barcodes)), -totals, inverse))
    winners = order[np.r_[True, inverse[order][1:] != inverse[order][:-1]]]
    group_totals = np.bincount(inverse, weights=totals)
    dominant_fraction = np.divide(totals[winners], group_totals, out=np.zeros(len(unique)), where=group_totals > 0)
    mito = features.gene_name.str.upper().str.startswith("MT-").to_numpy()
    mt_counts = np.asarray(matrix[winners][:, mito].sum(axis=1)).ravel()
    mt_fraction = np.divide(mt_counts, totals[winners], out=np.zeros(len(unique)), where=totals[winners] > 0)
    chosen = np.asarray(barcodes)[winners]
    audit = pd.DataFrame({"canonical_id": unique, "RNA_raw_barcode": chosen,
                          "RNA_counts": totals[winners], "RNA_detected_genes": np.diff(matrix.indptr)[winners],
                          "dominant_RT_fraction": dominant_fraction, "mitochondrial_fraction": mt_fraction,
                          "dominant_RT_tag": [b.split(":")[-1] for b in chosen]})
    return matrix[winners].tocsr(), audit


def promoters_from_gtf(path, rna_features, flank=2000):
    """One gene-level TSS per unique Ensembl ID; zero-based half-open windows."""
    records = {}
    wanted = set(rna_features.gene_id)
    headers = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                headers.append(line.strip())
                continue
            chrom, _, kind, start, end, _, strand, _, attributes = line.rstrip("\n").split("\t")
            if kind != "gene":
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)"', attributes))
            gene = attrs["gene_id"].split(".")[0]
            if gene not in wanted or chrom not in CHROMS:
                continue
            if gene in records or strand not in {"+", "-"}:
                raise ValueError(f"Ambiguous annotation: {gene}")
            tss = int(start) - 1 if strand == "+" else int(end) - 1
            records[gene] = {"gene_id": gene, "annotation_gene_name": attrs.get("gene_name", gene),
                             "chrom": chrom, "tss": tss, "strand": strand,
                             "start": max(0, tss - flank), "end": tss + flank,
                             "gene_type": attrs.get("gene_type", "unknown")}
    if not any("GRCh38" in header for header in headers):
        raise ValueError("Expected an explicitly GRCh38 annotation")
    table = rna_features.merge(pd.DataFrame(records.values()), on="gene_id", validate="one_to_one")
    return table.reset_index(drop=True), headers


class PromoterIndex:
    """Exact overlap index for fixed-width, TSS-centered windows."""
    def __init__(self, features):
        self.chrom = {}
        for chrom, group in features.groupby("chrom"):
            group = group.sort_values(["start", "end"])
            starts, ends = group.start.tolist(), group.end.tolist()
            if ends != sorted(ends):
                raise ValueError("Promoter index requires monotonic window ends")
            self.chrom[chrom] = (starts, ends, group.index.tolist())

    def overlap(self, chrom, start, end):
        if chrom not in self.chrom:
            return []
        starts, ends, indices = self.chrom[chrom]
        return indices[bisect_right(ends, start):bisect_left(starts, end)]


def count_fragments(path, features):
    index = PromoterIndex(features)
    counters, depths, rows = {}, Counter(), Counter()
    fragment_rows = multiplicity_sum = overlapping_rows = multi_promoter_rows = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            chrom, start, end, barcode, multiplicity = line.rstrip("\n").split("\t")
            start, end, multiplicity = int(start), int(end), int(multiplicity)
            if chrom not in CHROMS or start < 0 or end <= start or multiplicity < 1:
                raise ValueError("Invalid fragment coordinates/count")
            cell = canonical_id(barcode)
            counts = counters.setdefault(cell, Counter())
            depths[cell] += multiplicity
            rows[cell] += 1
            overlap = index.overlap(chrom, start, end)
            for feature in overlap:
                counts[feature] += multiplicity
            fragment_rows += 1
            multiplicity_sum += multiplicity
            overlapping_rows += bool(overlap)
            multi_promoter_rows += len(overlap) > 1
    cells = sorted(counters)
    matrix_rows, matrix_cols, values = [], [], []
    for row, cell in enumerate(cells):
        for col, count in counters[cell].items():
            matrix_rows.append(row)
            matrix_cols.append(col)
            values.append(count)
    matrix = sparse.csr_matrix((values, (matrix_rows, matrix_cols)), shape=(len(cells), len(features)), dtype=np.int32)
    audit = pd.DataFrame({"canonical_id": cells, "counts": [depths[c] for c in cells],
                          "fragment_rows": [rows[c] for c in cells]})
    summary = {"fragment_rows": fragment_rows, "multiplicity_sum": multiplicity_sum,
               "overlapping_fragment_rows": overlapping_rows, "multiple_promoter_overlap_rows": multi_promoter_rows,
               "matrix_shape": list(matrix.shape), "matrix_nnz": matrix.nnz,
               "total_genomic_fragment_quantiles": np.quantile(audit.counts, [0, .25, .5, .75, 1]).tolist(),
               "promoter_count_quantiles": np.quantile(np.asarray(matrix.sum(axis=1)).ravel(), [0, .25, .5, .75, 1]).tolist(),
               "detected_promoter_quantiles": np.quantile(np.diff(matrix.indptr), [0, .25, .5, .75, 1]).tolist()}
    return matrix, audit, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--reuse-rna-audit", action="store_true", help="Reuse the previously audited sparse RNA collapse after source receipts and dimensions are verified")
    args = parser.parse_args()
    raw, prepared, output = BASE / "raw_s1", BASE / "prepared_s1", BASE / "full_s1"
    output.mkdir(exist_ok=False)
    receipts = {}
    for path in sorted(raw.glob("*.gz")):
        receipt = json.loads(Path(str(path) + ".json").read_text())
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Source receipt mismatch: {path}")
        receipts[path.name] = receipt
    features = pd.read_csv(raw / "RNA.features.tsv.gz", sep="\t", header=None, names=["gene_id", "gene_name", "feature_type"])
    features["gene_id"] = features.gene_id.str.split(".").str[0]
    features["rna_column"] = np.arange(len(features))
    if features.gene_id.duplicated().any():
        raise ValueError("RNA contains duplicated unversioned Ensembl IDs")
    if args.reuse_rna_audit:
        rna = sparse.load_npz(prepared / "RNA_dominant_all.npz")
        rna_audit = pd.read_csv(prepared / "RNA_barcode_audit.tsv", sep="\t", dtype={"dominant_RT_tag": str})
        for barcode in rna_audit.RNA_raw_barcode:
            canonical_id(barcode, rna=True)
        if rna.shape != (len(rna_audit), len(features)):
            raise ValueError("Existing RNA audit dimensions mismatch")
        if not np.array_equal(np.asarray(rna.sum(axis=1)).ravel(), rna_audit.RNA_counts):
            raise ValueError("Existing RNA audit counts mismatch")
    else:
        barcodes = pd.read_csv(raw / "RNA.barcodes.tsv.gz", header=None)[0].tolist()
        with gzip.open(raw / "RNA.matrix.mtx.gz", "rb") as handle, threadpool_limits(limits=1):
            matrix = io.mmread(handle).T.tocsr()
        rna, rna_audit = collapse_rna(matrix, barcodes, features)
        prepared.mkdir(exist_ok=True)
        sparse.save_npz(prepared / "RNA_dominant_all.npz", rna)
        rna_audit.to_csv(prepared / "RNA_barcode_audit.tsv", sep="\t", index=False)
    promoters, headers = promoters_from_gtf(args.gtf, features)
    promoters.to_csv(output / "features.tsv", sep="\t", index=False)
    audits, summaries = [], {}
    source_cells = None
    for mark in MARKS:
        matrix, audit, summary = count_fragments(raw / f"{mark}.fragments.tsv.gz", promoters)
        if source_cells is not None and source_cells != audit.canonical_id.tolist():
            raise ValueError("Released histone barcode universes differ; reassess source population")
        source_cells = audit.canonical_id.tolist()
        sparse.save_npz(output / f"{mark}.npz", matrix)
        audits.append(audit.rename(columns={"counts": f"{mark}_counts", "fragment_rows": f"{mark}_fragment_rows"}))
        summaries[mark] = summary
        print(json.dumps({mark: summary}), flush=True)
    rna_lookup = pd.Series(np.arange(len(rna_audit)), index=rna_audit.canonical_id)
    if rna_lookup.index.duplicated().any() or not set(source_cells) <= set(rna_lookup.index):
        raise ValueError("Canonical RNA rows cannot cover the released DNA source population")
    selected = rna_lookup.loc[source_cells].to_numpy()
    sparse.save_npz(output / "RNA.npz", rna[selected][:, promoters.rna_column.to_numpy()].tocsr())
    audit = audits[0].merge(audits[1], on="canonical_id", validate="one_to_one").merge(rna_audit, on="canonical_id", validate="one_to_one")
    if audit.canonical_id.tolist() != source_cells:
        raise ValueError("Audit/matrix cell order mismatch")
    audit.to_csv(output / "cells.tsv", sep="\t", index=False)
    summary = {"source_population": "Published S1 DNA-filtered canonical cells; inherited total-six-target DNA QC >=2500 cannot be undone", "n_source_cells": len(source_cells), "n_RNA_features": len(features), "n_promoters": len(promoters), "gtf": {"path": str(args.gtf.resolve()), "sha256": sha256(args.gtf), "headers": headers}, "window_definition": "zero-based half-open [gene_TSS-2000, gene_TSS+2000), clipped at zero; canonical chromosomes only", "fragment_rule": "Each fragment multiplicity contributes once to every overlapping promoter; overlapping windows can share fragments", "sources": receipts, "assays": summaries, "preparation_script_sha256": sha256(__file__), "reused_preexisting_RNA_audit": args.reuse_rna_audit}
    (output / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_source_cells": len(source_cells), "n_promoters": len(promoters)}), flush=True)


if __name__ == "__main__":
    main()
