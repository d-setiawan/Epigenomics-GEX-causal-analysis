#!/usr/bin/env python3
"""Build cell-type-restricted metacells from a joint scGLUE run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def sanitize_token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "value"


def build_obs_id(df: pd.DataFrame) -> pd.Series:
    required = {"cell", "modality_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for obs_id construction: {sorted(missing)}")
    return df["modality_key"].astype(str) + "::" + df["cell"].astype(str)


def require_unique_obs_id(df: pd.DataFrame, name: str) -> None:
    dup_mask = df["obs_id"].duplicated()
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        raise ValueError(f"{name} contains {dup_count} duplicated obs_id values")


def pick_confidence_column(label_column: str, df: pd.DataFrame) -> str | None:
    if "coarse" in label_column and "coarse_confidence" in df.columns:
        return "coarse_confidence"
    if "fine" in label_column and "fine_confidence" in df.columns:
        return "fine_confidence"
    return None


def choose_n_metacells(
    filtered: pd.DataFrame,
    explicit_n: int | None,
    target_rna_cells_per_metacell: int,
    min_metacells: int,
    max_metacells: int | None,
) -> tuple[int, dict[str, Any]]:
    counts = filtered["modality_key"].astype(str).value_counts().sort_index()
    counts_dict = {k: int(v) for k, v in counts.items()}
    total_cells = int(filtered.shape[0])
    n_rna = int(counts.get("rna", 0))

    if explicit_n is not None:
        n_metacells = explicit_n
        reason = "explicit"
    else:
        if target_rna_cells_per_metacell <= 0:
            raise ValueError("--target-rna-cells-per-metacell must be > 0")
        anchor_n = n_rna if n_rna > 0 else total_cells
        proposed = int(round(anchor_n / target_rna_cells_per_metacell))
        n_metacells = max(min_metacells, proposed)
        reason = "rna_target_size" if n_rna > 0 else "total_target_size_no_rna_anchor"

    if max_metacells is not None:
        n_metacells = min(n_metacells, max_metacells)
    n_metacells = min(n_metacells, total_cells)
    if n_metacells < 2:
        raise ValueError("Need at least 2 metacells to proceed")

    expected_counts = {
        modality: float(count / n_metacells) for modality, count in counts_dict.items()
    }
    meta = {
        "selection_counts_by_modality": counts_dict,
        "selection_n_cells_total": total_cells,
        "selection_n_cells_rna": n_rna,
        "selection_expected_cells_per_metacell_by_modality": expected_counts,
        "selection_reason": reason,
    }
    return n_metacells, meta


def relabel_metacells(labels: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    order = counts.index.tolist()
    mapping = {int(old): f"MC_{i + 1:03d}" for i, old in enumerate(order)}
    remapped = np.array([mapping[int(x)] for x in labels], dtype=object)
    return remapped, mapping


def majority_label(series: pd.Series) -> tuple[str, float]:
    values = series.dropna().astype(str)
    values = values.loc[values != ""]
    if values.empty:
        return "unknown", float("nan")
    counts = values.value_counts()
    top_label = str(counts.index[0])
    purity = float(counts.iloc[0] / counts.sum())
    return top_label, purity


def summarize_cluster_sizes(assignments: pd.DataFrame) -> dict[str, float]:
    sizes = assignments.groupby("metacell_id").size()
    return {
        "min": float(sizes.min()),
        "median": float(sizes.median()),
        "mean": float(sizes.mean()),
        "max": float(sizes.max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Build cell-type-restricted metacells from joint scGLUE outputs")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--train-dir", default=None, help="Default: integration/outputs/scglue/joint/<RUN_ID>/train")
    p.add_argument(
        "--label-tsv",
        default=None,
        help="Default: <train-dir>/validation/joint_harmonized_label_transfer.tsv",
    )
    p.add_argument("--out-dir", default=None, help="Default: CausalDiscovery/outputs/scglue_metacells/<RUN_ID>/<LABEL>__<CELL_TYPE>")

    p.add_argument("--label-column", default="harmonized_coarse")
    p.add_argument("--cell-type", default="monocyte")
    p.add_argument(
        "--min-label-confidence",
        type=float,
        default=0.5,
        help="Applied only when the matching confidence column exists",
    )

    p.add_argument("--method", default="kmeans", choices=["kmeans"])
    p.add_argument("--n-metacells", type=int, default=None)
    p.add_argument("--target-rna-cells-per-metacell", type=int, default=20)
    p.add_argument("--min-metacells", type=int, default=2)
    p.add_argument("--max-metacells", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=0)

    p.add_argument("--qc-min-rna-cells", type=int, default=5)
    p.add_argument("--qc-min-other-cells", type=int, default=2)

    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_dir = (
        resolve_path(repo_root, args.train_dir)
        if args.train_dir
        else repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train"
    )
    label_tsv = (
        resolve_path(repo_root, args.label_tsv)
        if args.label_tsv
        else train_dir / "validation" / "joint_harmonized_label_transfer.tsv"
    )
    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root
        / "CausalDiscovery"
        / "outputs"
        / "scglue_metacells"
        / sanitize_token(args.run_id)
        / f"{sanitize_token(args.label_column)}__{sanitize_token(args.cell_type)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = train_dir / "all_cells_glue_embeddings.tsv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings table: {emb_path}")
    if not label_tsv.exists():
        raise FileNotFoundError(f"Missing label table: {label_tsv}")

    emb = pd.read_csv(emb_path, sep="\t")
    glue_cols = [c for c in emb.columns if c.startswith("GLUE_")]
    if not glue_cols:
        raise RuntimeError("No GLUE_* columns found in all_cells_glue_embeddings.tsv")
    emb = emb.copy()
    emb["obs_id"] = build_obs_id(emb)
    require_unique_obs_id(emb, "Embedding table")

    labels = pd.read_csv(label_tsv, sep="\t")
    labels = labels.copy()
    if "obs_id" not in labels.columns:
        labels["obs_id"] = build_obs_id(labels)
    require_unique_obs_id(labels, "Label table")
    if args.label_column not in labels.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in {label_tsv}")

    merged = emb.merge(
        labels,
        on="obs_id",
        how="left",
        suffixes=("", "_label"),
    )

    confidence_col = pick_confidence_column(args.label_column, merged)
    keep = merged[args.label_column].astype(str) == args.cell_type
    if confidence_col is not None:
        keep &= merged[confidence_col].fillna(0.0) >= args.min_label_confidence

    filtered = merged.loc[keep].copy()
    if filtered.empty:
        raise RuntimeError(
            f"No cells matched {args.label_column} == '{args.cell_type}' "
            f"with min confidence {args.min_label_confidence}"
        )

    x = filtered[glue_cols].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError("Filtered GLUE matrix contains non-finite values")

    n_metacells, selection_meta = choose_n_metacells(
        filtered=filtered,
        explicit_n=args.n_metacells,
        target_rna_cells_per_metacell=args.target_rna_cells_per_metacell,
        min_metacells=args.min_metacells,
        max_metacells=args.max_metacells,
    )
    if n_metacells >= filtered.shape[0]:
        raise RuntimeError("Number of metacells must be smaller than the number of selected cells")

    km = KMeans(
        n_clusters=n_metacells,
        random_state=args.random_seed,
        n_init=20,
    )
    labels_numeric = km.fit_predict(x).astype(int)
    distances = km.transform(x)
    min_dist = distances[np.arange(distances.shape[0]), labels_numeric]
    metacell_ids, metacell_mapping = relabel_metacells(labels_numeric)

    assignments = filtered.copy()
    assignments["metacell_id"] = metacell_ids
    assignments["distance_to_centroid"] = min_dist

    centroid_df = pd.DataFrame(
        km.cluster_centers_,
        columns=glue_cols,
    )
    centroid_df["cluster_numeric"] = np.arange(km.n_clusters, dtype=int)
    centroid_df["metacell_id"] = centroid_df["cluster_numeric"].map(metacell_mapping)
    centroid_df = centroid_df.drop(columns=["cluster_numeric"])
    centroid_df = centroid_df.set_index("metacell_id").sort_index().reset_index()

    modality_counts = pd.crosstab(assignments["metacell_id"], assignments["modality_key"]).sort_index()
    modality_counts = modality_counts.rename(columns={c: f"n_cells_{c}" for c in modality_counts.columns})

    summary_df = assignments.groupby("metacell_id").agg(
        n_cells_total=("obs_id", "size"),
        mean_distance_to_centroid=("distance_to_centroid", "mean"),
        median_distance_to_centroid=("distance_to_centroid", "median"),
    )
    summary_df = summary_df.join(modality_counts, how="left")

    if "harmonized_fine" in assignments.columns:
        fine_stats = assignments.groupby("metacell_id")["harmonized_fine"].apply(majority_label)
        summary_df["top_harmonized_fine"] = fine_stats.apply(lambda x: x[0])
        summary_df["top_harmonized_fine_purity"] = fine_stats.apply(lambda x: x[1])

    rna_count_col = "n_cells_rna"
    other_cols = [c for c in summary_df.columns if c.startswith("n_cells_") and c != rna_count_col]
    if rna_count_col not in summary_df.columns:
        summary_df[rna_count_col] = 0
    pass_mask = summary_df[rna_count_col] >= args.qc_min_rna_cells
    for col in other_cols:
        pass_mask &= summary_df[col] >= args.qc_min_other_cells
    summary_df["passes_default_qc"] = pass_mask
    summary_df = summary_df.reset_index().sort_values(["passes_default_qc", "n_cells_total"], ascending=[False, False])

    assignment_cols = [
        "obs_id",
        "cell",
        "modality_key",
        "mark",
        args.label_column,
        "metacell_id",
        "distance_to_centroid",
    ]
    for extra_col in ["harmonized_fine", "label_source", "coarse_confidence", "fine_confidence", "cell_ontology_id"]:
        if extra_col in assignments.columns and extra_col not in assignment_cols:
            assignment_cols.append(extra_col)
    assignments_out = assignments[assignment_cols].sort_values(["metacell_id", "modality_key", "cell"]).reset_index(drop=True)

    assignments_path = out_dir / "cell_assignments.tsv"
    centroids_path = out_dir / "metacell_centroids.tsv"
    summary_path = out_dir / "metacell_summary.tsv"
    json_path = out_dir / "run_summary.json"

    assignments_out.to_csv(assignments_path, sep="\t", index=False)
    centroid_df.to_csv(centroids_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)

    run_summary = {
        "inputs": {
            "run_id": args.run_id,
            "train_dir": str(train_dir),
            "all_cells_glue_embeddings_tsv": str(emb_path),
            "label_tsv": str(label_tsv),
        },
        "filter": {
            "label_column": args.label_column,
            "cell_type": args.cell_type,
            "min_label_confidence": args.min_label_confidence,
            "confidence_column_used": confidence_col,
            "n_cells_selected": int(filtered.shape[0]),
            "n_cells_with_any_label_row": int(merged[args.label_column].notna().sum()),
            "counts_by_modality": selection_meta["selection_counts_by_modality"],
        },
        "metacells": {
            "method": args.method,
            "n_metacells": int(n_metacells),
            "target_rna_cells_per_metacell": args.target_rna_cells_per_metacell,
            "selection_reason": selection_meta["selection_reason"],
            "expected_cells_per_metacell_by_modality": selection_meta["selection_expected_cells_per_metacell_by_modality"],
            "cluster_size_stats": summarize_cluster_sizes(assignments_out),
        },
        "qc": {
            "min_rna_cells": args.qc_min_rna_cells,
            "min_other_cells": args.qc_min_other_cells,
            "n_metacells_passing_default_qc": int(summary_df["passes_default_qc"].sum()),
            "fraction_metacells_passing_default_qc": float(summary_df["passes_default_qc"].mean()),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "cell_assignments_tsv": str(assignments_path),
            "metacell_centroids_tsv": str(centroids_path),
            "metacell_summary_tsv": str(summary_path),
        },
        "params": vars(args),
    }
    with json_path.open("w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Wrote: {assignments_path}")
    print(f"Wrote: {centroids_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
