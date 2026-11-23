#!/usr/bin/env python3
"""Evaluate existing embeddings against ADTs excluded from integration training.

Development-only by default. This is an exploratory biological-consistency
diagnostic, not ground-truth accuracy, a causal test, or donor-level inference.
Source data, labels, and embeddings are never modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


SPLIT_SALT = "biology-integration-2026-09-16-v1"
PANELS = {"T": ["CD3-1", "CD7", "CD2"],
          "myeloid": ["CD14", "CD33", "CD11b-1", "CD64"],
          "B": ["CD19", "CD20", "CD79b", "CD22"]}
CONTRASTS = ["T_minus_myeloid", "B_minus_T"]
MARKS = {"H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3"}


def file_info(path, *, hash_content=True):
    stat = path.stat()
    result = {"path": str(path.resolve()), "bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def verify_unchanged(records):
    for record in records:
        current = file_info(Path(record["path"]), hash_content=False)
        if any(current[key] != record[key] for key in ("bytes", "mtime_ns")):
            raise RuntimeError(f"Input changed while evaluating: {record['path']}")


def validate_manifest(manifest):
    if "mark" not in manifest or manifest.mark.duplicated().any() or set(manifest.mark) != MARKS:
        raise ValueError("Require exactly six unique expected histone assays")


def core(value):
    return re.sub(r"-\d+$", "", re.sub(r"\.1$", "", str(value)))


def split_for(mark, barcode):
    digest = hashlib.sha256(f"{SPLIT_SALT}|{mark}|{core(barcode)}".encode()).digest()
    return "development" if int.from_bytes(digest[:8], "big") < 3 * 2**62 else "assessment"


def sample_positions(ids, limit):
    """Outcome-free, row-order invariant sampling, identical across methods."""
    if limit <= 0 or limit >= len(ids):
        return np.arange(len(ids))
    keys = [hashlib.sha256(f"query-v1|{x}".encode()).hexdigest() for x in ids]
    return np.sort(np.argsort(keys)[:limit])


def describe(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"n": len(values), "nonfinite": int(len(values) - len(finite)),
            "mean": float(finite.mean()) if len(finite) else None,
            "sd": float(finite.std()) if len(finite) else None,
            "quantiles": {str(q): float(np.quantile(finite, q))
                          for q in (0, .1, .5, .9, 1)} if len(finite) else {}}


def correlations(observed, predicted):
    """Columnwise Spearman effects; undefined columns are NaN, never zero."""
    a, b = np.asarray(observed, float), np.asarray(predicted, float)
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation inputs must be aligned finite 2D arrays")
    a, b = rankdata(a, axis=0), rankdata(b, axis=0)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), denominator,
                     out=np.full(a.shape[1], np.nan), where=denominator > 0)


def nuisance_design(meta):
    depth = np.log1p(meta[["adt_total", "n_cuttag_fragments"]].to_numpy(float))
    tags = pd.get_dummies(meta.donor_id.astype(str), dtype=float).to_numpy()
    return np.column_stack([np.ones(len(meta)), depth, depth**2, tags])


def adjusted_scores(scores, meta, fit_mask):
    design = nuisance_design(meta)
    fit_mask = np.asarray(fit_mask, bool)
    if fit_mask.sum() <= design.shape[1]:
        raise ValueError("Insufficient development cells for nuisance adjustment")
    coef = np.linalg.lstsq(design[fit_mask], scores[fit_mask], rcond=None)[0]
    return scores - design @ coef


def quantile_groups(values, bins=4):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def permutation_groups(meta):
    a = quantile_groups(meta.adt_total.to_numpy())
    b = quantile_groups(meta.n_cuttag_fragments.to_numpy())
    keys = np.asarray([f"{tag}|{i}|{j}" for tag, i, j in zip(meta.donor_id, a, b)])
    groups = [np.flatnonzero(keys == key) for key in sorted(set(keys))]
    return groups


def permute_within(groups, n, rng):
    indices = np.arange(n)
    for group in groups:
        indices[group] = rng.permutation(group)
    return indices


def neighbors(source_x, target_x, source_ids, target_ids, k):
    """Exclude matching identities explicitly, including tied embeddings."""
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("Duplicate neighbor identities")
    if k < 1 or len(source_x) <= k:
        raise ValueError("Need more source cells than k")
    if not np.isfinite(source_x).all() or not np.isfinite(target_x).all():
        raise ValueError("Nonfinite neighbor representation")
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1).fit(source_x)
    distances, candidates = model.kneighbors(target_x)
    indices, retained_distances = [], []
    for name, row, dist in zip(target_ids, candidates, distances):
        keep = source_ids[row] != name
        indices.append(row[keep][:k])
        retained_distances.append(dist[keep][:k])
    return np.asarray(indices), np.asarray(retained_distances)


def neighbor_mean(values, indices):
    # Accumulation avoids allocating queries x k x all proteins.
    result = np.zeros((len(indices), values.shape[1]))
    for j in range(indices.shape[1]):
        result += values[indices[:, j]]
    return result / indices.shape[1]


def h5_column(node):
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]
    categories = h5_column(node["categories"])
    return np.asarray([categories[i] if i >= 0 else None for i in node["codes"][:]])


def read_embedding(path, prefix):
    table = pd.read_csv(path, sep="\t")
    if not {"cell", "mark", "modality_key"}.issubset(table):
        raise ValueError("Missing embedding identity columns")
    table["barcode_core"] = table.cell.map(core)
    table["id"] = table.modality_key.astype(str) + "::" + table.barcode_core
    if table.id.duplicated().any():
        raise ValueError("Duplicate assay-qualified embedding identity")
    columns = [c for c in table if c.startswith(prefix)]
    if not columns or not np.isfinite(table[columns].to_numpy(float)).all():
        raise ValueError("Missing or nonfinite embedding columns")
    return table, columns


def load_assay(root, row, emb, columns, partition, lsi_dir):
    mark = str(row.mark)
    cells = emb.loc[emb.modality_key.eq(f"chrom_{mark}")].sort_values("barcode_core").copy()
    if cells.empty or not cells.mark.eq(mark).all():
        raise ValueError(f"Missing or inconsistent modality {mark}")
    cells["partition"] = [split_for(mark, b) for b in cells.barcode_core]
    # Assessment protein columns are NOT loaded on a development run.
    used = cells if partition == "assessment" else cells.loc[cells.partition.eq("development")]
    dev = used.partition.eq("development").to_numpy()
    query = used.partition.eq(partition).to_numpy()
    metadata_path = root / row.clean_cells
    adt_path = metadata_path.with_name(f"{row.prefix}_adt_clean_matrix.tsv")
    local_path = lsi_dir / f"chrom_{mark}_preprocessed.h5ad"
    snapshots = [file_info(p, hash_content=False) for p in (metadata_path, adt_path, local_path)]
    metadata = pd.read_csv(metadata_path, sep="\t")
    metadata["barcode_core"] = metadata.barcode.map(core)
    if metadata.barcode_core.duplicated().any():
        raise ValueError(f"Duplicate clean metadata identities: {mark}")
    meta = metadata.set_index("barcode_core").loc[used.barcode_core].copy()
    if meta[["donor_id", "adt_total", "n_cuttag_fragments"]].isna().any().any():
        raise ValueError("Missing technical covariates")
    if not np.isfinite(meta[["adt_total", "n_cuttag_fragments"]].to_numpy(float)).all() or (meta[["adt_total", "n_cuttag_fragments"]] < 0).any().any():
        raise ValueError("Invalid technical depth")
    with adt_path.open() as stream:
        header = next(csv.reader(stream, delimiter="\t"))
    raw_columns = header[1:]
    canonical = [core(c) for c in raw_columns]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate ADT barcode cores")
    lookup = dict(zip(canonical, raw_columns))
    selected_columns = [lookup[b] for b in used.barcode_core]
    adt = pd.read_csv(adt_path, sep="\t", usecols=[header[0], *selected_columns], index_col=0)
    if adt.index.duplicated().any():
        raise ValueError("Duplicate ADT features")
    values = adt.loc[:, selected_columns].T.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Invalid ADT counts")
    if not np.array_equal(values.sum(axis=1), meta.adt_total.to_numpy()):
        raise ValueError("ADT totals disagree with upstream metadata")
    proteins = adt.index.astype(str).tolist()
    logged = np.log1p(values)
    panel = {name: logged[:, [proteins.index(p) for p in panel]].mean(axis=1)
             for name, panel in PANELS.items()}
    isotypes = [i for i, name in enumerate(proteins) if "isotype" in name.lower()]
    if not isotypes:
        raise ValueError("No isotype controls")
    scores = np.column_stack([panel["T"] - panel["myeloid"], panel["B"] - panel["T"],
                              panel["T"], panel["myeloid"], panel["B"],
                              logged[:, isotypes].mean(axis=1),
                              np.log1p(meta.adt_total), np.log1p(meta.n_cuttag_fragments)])
    names = [*CONTRASTS, "T_panel", "myeloid_panel", "B_panel", "isotype_mean",
             "log_adt_total", "log_fragment_depth"]
    residual = adjusted_scores(scores, meta, dev)
    with h5py.File(local_path, "r") as handle:
        cores = pd.Index([core(c) for c in h5_column(handle["obs/barcode_core"])])
        positions = cores.get_indexer(used.barcode_core)
        if cores.duplicated().any() or (positions < 0).any():
            raise ValueError("Local LSI barcode mismatch")
        lsi = handle["obsm/X_lsi"][:][positions]
    groups = permutation_groups(meta.loc[dev])
    verify_unchanged(snapshots)
    report = {"mark": mark, "n_embedding_cells": len(cells), "n_development": int(dev.sum()),
              "n_query_partition": int(query.sum()), "n_assessment_reserved": int(cells.partition.eq("assessment").sum()),
              "adt_input": file_info(adt_path), "metadata_input": file_info(metadata_path),
              "local_lsi_input": file_info(local_path, hash_content=False),
              "isotypes": [proteins[i] for i in isotypes],
              "adt_zero_fraction": {p: float((values[:, j] == 0).mean()) for j, p in enumerate(proteins)},
              "scores": {name: describe(scores[:, j]) for j, name in enumerate(names)},
              "permutation_strata_sizes": [len(g) for g in groups],
              "permutable_source_fraction": float(sum(len(g) for g in groups if len(g) > 1) / dev.sum())}
    return {"ids": used.id.to_numpy(), "joint": used[columns].to_numpy(float), "lsi": lsi,
            "scores": scores, "adjusted": residual, "proteins": logged, "protein_names": proteins,
            "names": names, "meta": meta, "dev": dev, "query": query, "groups": groups,
            "report": report, "membership": cells[["id", "partition"]]}


def score_pair(source, target, *, k, limit, permutations, seed, representation="joint", within=False):
    source_pos = np.flatnonzero(source["dev"])
    target_pos = np.flatnonzero(target["query"])
    target_pos = target_pos[sample_positions(target["ids"][target_pos], limit)]
    indices, distances = neighbors(source[representation][source_pos], target[representation][target_pos],
                                   source["ids"][source_pos], target["ids"][target_pos], k)
    raw = source["scores"][source_pos]
    adjusted = source["adjusted"][source_pos]
    raw_prediction = neighbor_mean(raw, indices)
    adjusted_prediction = neighbor_mean(adjusted, indices)
    observed = target["scores"][target_pos]
    adjusted_observed = target["adjusted"][target_pos]
    raw_rho = correlations(observed, raw_prediction)
    adjusted_rho = correlations(adjusted_observed, adjusted_prediction)
    usage = np.bincount(indices.ravel(), minlength=len(source_pos))
    weights = usage / usage.sum()
    result = {"n_source": len(source_pos), "n_query": len(target_pos), "k": k,
              "source_ids_sha256": hashlib.sha256("\n".join(source["ids"][source_pos]).encode()).hexdigest(),
              "query_ids_sha256": hashlib.sha256("\n".join(target["ids"][target_pos]).encode()).hexdigest(),
              "source_cells_used_fraction": float((usage > 0).mean()),
              "source_effective_cells": float(1 / np.sum(weights**2)),
              "max_source_usage_fraction": float(weights.max()),
              "nearest_distance": describe(distances[:, 0]),
              "raw_spearman": dict(zip(source["names"], raw_rho)),
              # Depth columns are nuisance regressors themselves; their residual
              # ranks would be numerical noise, not an interpretable control.
              "adjusted_spearman": dict(zip(source["names"][:6], adjusted_rho[:6]))}
    if source["protein_names"] != target["protein_names"]:
        raise ValueError("Protein panels differ across assays")
    protein_pred = neighbor_mean(source["proteins"][source_pos], indices)
    result["individual_protein_raw_spearman"] = dict(zip(source["protein_names"],
        correlations(target["proteins"][target_pos], protein_pred)))
    # Describe heterogeneity without using these strata to select cells.
    target_meta = target["meta"].iloc[target_pos]
    result["target_hashtag_contrasts"] = {}
    for tag in sorted(target_meta.donor_id.unique()):
        mask = target_meta.donor_id.eq(tag).to_numpy()
        result["target_hashtag_contrasts"][str(tag)] = {"n": int(mask.sum()),
            "raw_spearman": dict(zip(CONTRASTS, correlations(observed[mask, :2], raw_prediction[mask, :2])))}
    result["target_adt_depth_quartile_contrasts"] = {}
    for group in range(4):
        mask = quantile_groups(target_meta.adt_total.to_numpy()) == group
        if mask.sum() < 2:
            continue
        result["target_adt_depth_quartile_contrasts"][str(group)] = {"n": int(mask.sum()),
            "raw_spearman": dict(zip(CONTRASTS, correlations(observed[mask, :2], raw_prediction[mask, :2])))}
    if not within:
        # Whole source rows are jointly shuffled; same permutation for raw/adjusted.
        rng = np.random.default_rng(seed)
        raw_null, adjusted_null, moved = [], [], []
        for _ in range(permutations):
            order = permute_within(source["groups"], len(source_pos), rng)
            moved.append(float(np.mean(order != np.arange(len(order)))))
            raw_null.append(correlations(observed[:, :6], neighbor_mean(raw[order, :6], indices)))
            adjusted_null.append(correlations(adjusted_observed[:, :6], neighbor_mean(adjusted[order, :6], indices)))
        result["conditional_shuffle"] = {"n": permutations, "moved_source_fraction": describe(moved),
            "raw": {name: describe(np.asarray(raw_null)[:, j]) for j, name in enumerate(source["names"][:6])},
            "adjusted": {name: describe(np.asarray(adjusted_null)[:, j]) for j, name in enumerate(source["names"][:6])}}
    return result


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--match-embedding", type=Path, required=True,
                        help="Fixed comparator identity universe, to ensure matched cohorts")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=["development", "assessment"], default="development")
    parser.add_argument("--release-assessment", action="store_true")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--max-queries", type=int, default=1000, help="0 uses every query")
    parser.add_argument("--permutations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--local-lsi-dir", type=Path,
                        default=Path("pipelines/empirical/integration/runs/scglue_joint_v2/preprocess"))
    args = parser.parse_args()
    if args.partition == "assessment" and not args.release_assessment:
        parser.error("Assessment is reserved; require explicit --release-assessment")
    if args.permutations < 2 or args.max_queries < 0 or args.k < 1:
        parser.error("Require permutations >= 2, max-queries >= 0, k >= 1")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    root = args.root.resolve()
    manifest_path = root / "pipelines/empirical/integration/manifests/scglue_input_manifest.tsv"
    snapshots = [file_info(p) for p in (manifest_path, args.embedding, args.match_embedding)]
    manifest = pd.read_csv(manifest_path, sep="\t")
    validate_manifest(manifest)
    emb, columns = read_embedding(args.embedding, args.prefix)
    comparator = pd.read_csv(args.match_embedding, sep="\t", usecols=["cell", "modality_key"])
    matched_ids = comparator.modality_key.astype(str) + "::" + comparator.cell.map(core)
    if matched_ids.duplicated().any():
        raise ValueError("Duplicate comparator identities")
    unmatched = emb.loc[~emb.id.isin(matched_ids), "modality_key"].value_counts().to_dict()
    emb = emb.loc[emb.id.isin(matched_ids)].copy()
    verify_unchanged(snapshots)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    protocol = {"schema": "biological_alignment.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "script": file_info(Path(__file__)), "embedding": file_info(args.embedding),
                "comparator": file_info(args.match_embedding), "manifest": file_info(manifest_path),
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "split_salt": SPLIT_SALT, "development_fraction": .75, "panels": PANELS,
                "dropped_outside_comparator": unmatched,
                "software": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                             "sklearn": sklearn.__version__, "h5py": h5py.__version__},
                "limitations": ["Exploratory; these data were previously examined. Reservation starts now.",
                    "Proteins excluded from integration training; same-embedding labels are not truth.",
                    "Development proteins only as source, including for future assessment queries.",
                    "Existing embedding fits are transductive; this is not unseen-cell embedding generalization.",
                    "HTO tags are technical groups, not verified donors or cross-assay matched groups.",
                    "Depth/HTO residualization is sensitivity analysis and can remove genuine biology.",
                    "Conditional shuffles give descriptive references, not calibrated tests or confidence intervals.",
                    "Neighbor reuse and cell dependence preclude donor-level inference.",
                    "Histone-to-histone protein agreement does not establish RNA alignment or causal validity.",
                    "No overall accuracy score, automatic relabeling, or acceptance threshold is produced.",
                    "Full files are hashed for provenance, but reserved protein columns are not analyzed in development.",
                    "Local LSI reference is the common historical scGLUE 30k-bin representation for both methods."]}
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    lsi_dir = args.local_lsi_dir if args.local_lsi_dir.is_absolute() else root / args.local_lsi_dir
    with threadpool_limits(limits=1):
        assays = {str(row.mark): load_assay(root, row, emb, columns, args.partition, lsi_dir)
                  for row in manifest.itertuples()}
        membership_path = args.out_dir / "cell_partitions.tsv"
        pd.concat([a["membership"] for a in assays.values()]).sort_values("id").to_csv(
            membership_path, sep="\t", index=False)
        results = {"protocol": protocol, "assays": [a["report"] for a in assays.values()],
                   "membership": file_info(membership_path),
                   "within_assay": [], "cross_assay": []}
        for source_name, source in assays.items():
            for rep in ("joint", "lsi"):
                scores = score_pair(source, source, k=args.k, limit=args.max_queries,
                                    permutations=args.permutations, seed=args.seed,
                                    representation=rep, within=True)
                results["within_assay"].append({"mark": source_name, "representation": rep, **scores})
            for target_name, target in assays.items():
                if source_name == target_name:
                    continue
                pair_seed = args.seed + int.from_bytes(hashlib.sha256(f"{source_name}|{target_name}".encode()).digest()[:4], "big")
                scores = score_pair(source, target, k=args.k, limit=args.max_queries,
                                    permutations=args.permutations, seed=pair_seed)
                result = {"source": source_name, "target": target_name, **scores}
                results["cross_assay"].append(result)
                print(json.dumps({"source": source_name, "target": target_name,
                                  "T_minus_myeloid": result["raw_spearman"]["T_minus_myeloid"]}), flush=True)
    if len(results["cross_assay"]) != 30 or len(results["within_assay"]) != 12:
        raise RuntimeError("Incomplete assay comparisons")
    for assay in assays.values():
        snapshots.extend(assay["report"][key] for key in ("adt_input", "metadata_input", "local_lsi_input"))
    verify_unchanged(snapshots)
    results["input_stat_recheck_passed"] = True
    (args.out_dir / "metrics.json").write_text(json.dumps(json_clean(results), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
