"""Independent replay of scMTR split, training panel and encoder exclusions."""

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from scipy import sparse

import causal_reliability_pilot_empirical as empirical


def main():
    protocol, pools = empirical.context()
    empirical.load_evaluator(pools)  # Exact paired joins, opaque row order and disjointness.
    source = empirical.DATA.parent / "full_s1"
    cells = pd.read_csv(source / "cells.tsv", sep="\t", dtype={"dominant_RT_tag": str})
    features = pd.read_csv(source / "features.tsv", sep="\t")
    evidence = pd.read_csv(empirical.DATA / "evaluator/training_feature_coverage.tsv", sep="\t")
    assert cells.canonical_id.is_unique and features.gene_id.is_unique
    ordered = sorted(cells.canonical_id, key=lambda c: hashlib.sha256(("2718|partition|" + c).encode()).hexdigest())
    reserve = set(ordered[:600])
    assigned = {a: set(ordered[600 + i::3]) for i, a in enumerate(empirical.ASSAYS)}
    independent_qc = {
        "RNA": (cells.RNA_counts >= 1000) & (cells.RNA_detected_genes >= 500) & (cells.dominant_RT_fraction > .5)
               & (cells.mitochondrial_fraction <= .05) & cells.dominant_RT_tag.isin(["01", "02", "03", "04"]),
        "H3K27ac": cells.H3K27ac_counts >= 250,
        "H3K4me3": cells.H3K4me3_counts >= 250}
    training_ids = {a: set(cells.loc[cells.canonical_id.isin(assigned[a]) & independent_qc[a], "canonical_id"]) for a in assigned}
    paired_ids = set(cells.loc[cells.canonical_id.isin(reserve) & np.logical_and.reduce(list(independent_qc.values())), "canonical_id"])
    assert paired_ids == set(empirical.table("evaluator/heldout_pairs.tsv").canonical_id)
    eligible = np.ones(len(features), dtype=bool)
    positive, sources = {}, {str(source / n): empirical.digest(source / n) for n in ("cells.tsv", "features.tsv", "provenance.json")}
    for assay in empirical.ASSAYS:
        actual = empirical.table(f"evaluator/{assay}_train_map.tsv")
        assert training_ids[assay] == set(actual.canonical_id)
        path = source / f"{assay}.npz"
        sources[str(path)] = empirical.digest(path)
        matrix = sparse.load_npz(path)
        rows = cells.canonical_id.isin(training_ids[assay]).to_numpy()
        detected = np.asarray((matrix[rows] > 0).sum(axis=0)).ravel()
        positive[assay] = detected > 0
        threshold = .1 if assay == "RNA" else .05
        eligible &= (detected >= 20) & (detected / rows.sum() >= threshold)
        np.testing.assert_array_equal(detected, evidence[f"{assay}_training_detected_cells"])
        del matrix
    selected = []
    for i in sorted(np.flatnonzero(eligible), key=lambda i: hashlib.sha256(("20261101_causal_panel_v1|" + features.loc[i, "gene_id"]).encode()).hexdigest()):
        f = features.loc[i]
        if any(f.chrom == features.loc[j, "chrom"] and f.start < features.loc[j, "end"] and features.loc[j, "start"] < f.end for j in selected):
            continue
        selected.append(i)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    expected_genes = features.loc[selected, "gene_id"].tolist()
    assert expected_genes == empirical.table("evaluator/panel_genes.tsv").gene_id.tolist()
    excluded = np.zeros(len(features), dtype=bool)
    for i in selected:
        f = features.loc[i]
        excluded |= ((features.chrom == f.chrom) & (features.start < f.end) & (f.start < features.end)).to_numpy()
    aliases = set(features.loc[excluded, ["gene_name", "annotation_gene_name"]].to_numpy().ravel())
    aliases = {x for x in aliases if isinstance(x, str) and x}
    excluded |= features.gene_name.isin(aliases).to_numpy() | features.annotation_gene_name.isin(aliases).to_numpy()
    np.testing.assert_array_equal(excluded, evidence.excluded_from_encoder)
    for assay in empirical.ASSAYS:
        expected = features.loc[positive[assay] & ~excluded, "gene_id"].tolist()
        assert expected == empirical.table(f"learner/{assay}_features.tsv").gene_id.tolist()
        assert not set(expected) & set(expected_genes)
    report = {
        "status": "passed", "protocol_sha256": empirical.digest(empirical.ROOT / "protocol.json"),
        "reviewer": "Independent causal-reliability agent; no imports of cohort selection implementation",
        "script": empirical.pin(__file__), "source_files": sources,
        "checks": ["Cohort and bridge file seals", "Exact opaque-ID paired count joins",
                   "Canonical split and assigned-assay QC independently reconstructed", "No canonical train/test or cross-training overlap",
                   "All-gene detection coverage recomputed from source sparse counts using training rows only",
                   "Hash-ranked panel selected using training coverage only", "All panel overlaps and gene-name aliases removed",
                   "Every encoder feature list equals own-training-positive features after exclusions"],
        "counts": {"paired": len(paired_ids), "train": {a: len(x) for a, x in training_ids.items()},
                   "eligible_genes": int(eligible.sum()), "excluded_features": int(excluded.sum())},
        "panel": features.loc[selected, ["gene_id", "gene_name"]].to_dict("records"),
        "limits": ["This replay verifies prepared source-to-cohort logic, not the biological validity of canonical barcodes",
                   "No redownload or independent alignment; inherited six-target source filtering remains",
                   "Empirical graph agreement supplies no known causal edges or directions"]}
    empirical.write(empirical.ROOT / "independent_cohort_audit.json", report)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "panel": report["panel"]}))


if __name__ == "__main__":
    main()
