#!/usr/bin/env python3
"""Read-only replay and protein-contrast checks for the saved GLUE alignment.

No embedding is trained or altered, and no replacement cell labels are written.
Protein contrasts and neighborhood summaries are descriptive, not label truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from audit_biological_inputs import h5_column


PANELS = {"T": ["CD3-1", "CD7", "CD2"],
          "myeloid": ["CD14", "CD33", "CD11b-1", "CD64"],
          "B": ["CD19", "CD20", "CD79b", "CD22"]}


def provenance(path):
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()}


def vote(labels):
    winner, count = Counter(labels).most_common(1)[0]
    confidence = count / len(labels)
    return winner if confidence >= .5 else "unknown", confidence


def contrast_summary(table):
    output = []
    for label, group in table.groupby("harmonized_coarse", sort=True):
        output.append({"label": label, "n": len(group),
                       "median_T_minus_myeloid": float(group.T_minus_myeloid.median()),
                       "median_B_minus_T": float(group.B_minus_T.median()),
                       "median_adt_total": float(group.adt_total.median()),
                       "median_isotype_log1p": float(group.isotype_log1p.median())})
    return output


def neighbor_conservation(representation, score):
    if not np.isfinite(representation).all():
        return {"available": False, "reason": "nonfinite representation"}
    neighbor_index = NearestNeighbors(n_neighbors=26, metric="euclidean", n_jobs=1).fit(representation)
    indices = neighbor_index.kneighbors(representation, return_distance=False)
    # Explicit self removal rather than assuming self always occupies the first slot.
    neighbors = np.asarray([row[row != i][:25] for i, row in enumerate(indices)])
    smooth = score[neighbors].mean(axis=1)
    result = spearmanr(score, smooth)
    return {"available": True, "k": 25, "self_excluded": True,
            "spearman_score_vs_neighbor_mean": float(result.statistic),
            "interpretation": "within-assay protein-contrast conservation, not classification accuracy"}


def audit(root):
    train = root / "pipelines/empirical/integration/runs/scglue_joint_v2/train"
    emb_path = train / "all_cells_glue_embeddings.tsv"
    label_path = train / "validation/joint_harmonized_label_transfer.tsv"
    emb = pd.read_csv(emb_path, sep="\t")
    labels = pd.read_csv(label_path, sep="\t").set_index("obs_id")
    ids = emb.modality_key.astype(str) + "::" + emb.cell.astype(str)
    if ids.duplicated().any() or labels.index.duplicated().any() or set(ids) != set(labels.index):
        raise ValueError("Embedding/label identities are not one-to-one")
    labels = labels.loc[ids].copy()
    if not np.array_equal(labels.cell.to_numpy(), emb.cell.to_numpy()):
        raise ValueError("Cell order mismatch after explicit ID alignment")
    glue_cols = [c for c in emb if c.startswith("GLUE_")]
    x = emb[glue_cols].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite GLUE embedding")
    ref_mask = emb.modality_key.eq("rna").to_numpy()
    reference_labels = labels.loc[ref_mask, "harmonized_coarse"].to_numpy(dtype=str)
    if len(reference_labels) < 50:
        raise ValueError("Insufficient saved RNA references")
    nn = NearestNeighbors(n_neighbors=50, metric="euclidean", n_jobs=1).fit(x[ref_mask])
    records = []
    for mark in sorted(emb.loc[~ref_mask, "mark"].unique()):
        mask = emb.mark.eq(mark).to_numpy()
        target = labels.loc[mask].copy()
        distances, indices = nn.kneighbors(x[mask])
        variants = {}
        for k in (10, 25, 50):
            voted = [vote(reference_labels[row[:k]]) for row in indices]
            predicted = np.asarray([v[0] for v in voted])
            confidence = np.asarray([v[1] for v in voted])
            variants[str(k)] = {"n": len(target),
                "different_from_saved_coarse": int(np.sum(predicted != target.harmonized_coarse.to_numpy())),
                "max_confidence_difference_from_saved": float(np.max(np.abs(confidence - target.coarse_confidence.to_numpy()))),
                "unknowns": int(np.sum(predicted == "unknown"))}

        paths = list((root / "pipelines/empirical/data/outputs" / mark).glob("*_adt_clean_matrix.tsv"))
        if len(paths) != 1:
            raise ValueError(f"Nonunique ADT path for {mark}")
        adt = pd.read_csv(paths[0], sep="\t", index_col=0)
        adt.columns = adt.columns.str.replace(r"\.1$", "", regex=True)
        if adt.columns.duplicated().any() or adt.index.duplicated().any() or target.barcode_core.duplicated().any():
            raise ValueError(f"Nonunique ADT barcode/feature for {mark}")
        values = adt.loc[:, target.barcode_core].T
        if not np.isfinite(values.to_numpy()).all() or (values.to_numpy() < 0).any():
            raise ValueError("Invalid ADT values")
        logged = np.log1p(values)
        scores = {name: logged[features].mean(axis=1).to_numpy() for name, features in PANELS.items()}
        target["T_minus_myeloid"] = scores["T"] - scores["myeloid"]
        target["B_minus_T"] = scores["B"] - scores["T"]
        target["adt_total"] = values.sum(axis=1).to_numpy()
        isotypes = [c for c in values if "Isotype" in c]
        target["isotype_log1p"] = logged[isotypes].mean(axis=1).to_numpy()
        metadata_paths = list((root / "pipelines/empirical/data/outputs" / mark).glob("*_clean_cells.tsv"))
        if len(metadata_paths) != 1:
            raise ValueError("Nonunique metadata path")
        metadata = pd.read_csv(metadata_paths[0], sep="\t").set_index("barcode")
        metadata.index = metadata.index.str.replace(r"\.1$", "", regex=True).str.replace(r"-\d+$", "", regex=True)
        if metadata.index.duplicated().any():
            raise ValueError("Duplicate metadata cores")
        target["hashtag_group"] = metadata.loc[target.barcode_core, "donor_id"].to_numpy()
        h5_path = train / f"modalities/chrom_{mark}_with_glue.h5ad"
        with h5py.File(h5_path, "r") as handle:
            obs = handle["obs"]
            cores = pd.Index(h5_column(obs["barcode_core"]))
            positions = cores.get_indexer(target.barcode_core)
            if (positions < 0).any() or cores.duplicated().any():
                raise ValueError("H5AD representation barcode mismatch")
            lsi = handle["obsm/X_lsi"][:][positions]
        record = {"mark": mark, "n": len(target), "replay": variants,
            "median_nearest_rna_distance": float(np.median(distances[:, 0])),
            "adt_input": provenance(paths[0]), "metadata_input": provenance(metadata_paths[0]),
            "representation_input": {"path": str(h5_path), "bytes": h5_path.stat().st_size,
                                     "mtime_ns": h5_path.stat().st_mtime_ns},
            "panels": PANELS, "isotype_features": isotypes,
            "all_cells": contrast_summary(target),
            "confidence_ge_0_9": contrast_summary(target.loc[target.coarse_confidence >= .9]),
            "upper_adt_total_quartile": contrast_summary(target.loc[target.adt_total >= target.adt_total.quantile(.75)]),
            "by_hashtag": {str(tag): contrast_summary(group) for tag, group in target.groupby("hashtag_group")},
            "T_minus_myeloid_conservation": {
                "saved_glue": neighbor_conservation(x[mask], target.T_minus_myeloid.to_numpy()),
                "saved_lsi": neighbor_conservation(lsi, target.T_minus_myeloid.to_numpy())}}
        records.append(record)
        print(json.dumps({"mark": mark, "replay_k25": variants["25"],
                          "conservation": record["T_minus_myeloid_conservation"]}), flush=True)
    return {"schema": "saved_alignment_investigation.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": provenance(Path(__file__)), "embedding": provenance(emb_path), "labels": provenance(label_path),
            "software": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "method": {"distance": "Euclidean in unchanged saved embedding", "reference": "saved RNA annotation labels",
                       "votes": "unweighted majority; ties resolved by neighbor order; confidence below0.5 becomes unknown",
                       "contrasts": "difference of mean log1p ADT counts across prespecified marker panels; no thresholded biological labels",
                       "hashtag": "technical groups, not verified donors", "no_training_or_production_label_changes": True},
            "assays": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with threadpool_limits(limits=1):
        result = audit(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
