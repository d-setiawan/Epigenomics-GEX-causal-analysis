#!/usr/bin/env python3
"""Build RNA-anchored kNN neighbor tables for averaged chromatin modality profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from build_scglue_one_to_one_matches import (
    build_obs_id,
    infer_repo_root,
    modality_summary,
    pick_confidence_column,
    require_unique_obs_id,
    resolve_path,
    sanitize_token,
)


def compute_neighbor_weights(distances: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.full(distances.shape, 1.0 / distances.shape[1], dtype=float)
    if mode == "inverse_distance":
        weights = 1.0 / np.maximum(distances, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights
    raise ValueError(f"Unsupported weight mode: {mode}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build RNA-anchored kNN neighbor tables for averaged CUT&Tag modality profiles"
    )
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--train-dir", default=None, help="Default: pipelines/empirical/integration/runs/scglue_<RUN_ID>/train")
    p.add_argument(
        "--label-tsv",
        default=None,
        help="Default: <train-dir>/validation/joint_harmonized_label_transfer.tsv",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Default: pipelines/empirical/analysis/outputs/scglue_pairings/<RUN_ID>/<LABEL>__<CELL_TYPE>__rna_anchor_neighbor_average_k<K>",
    )
    p.add_argument("--label-column", default="harmonized_coarse")
    p.add_argument("--cell-type", default="monocyte")
    p.add_argument("--min-label-confidence", type=float, default=0.5)
    p.add_argument("--anchor-modality", default="rna")
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument(
        "--neighbor-weight-mode",
        default="uniform",
        choices=["uniform", "inverse_distance"],
        help="How to average each mark's k nearest chromatin neighbors for each RNA anchor.",
    )
    p.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="Optional cap on RNA anchors after deterministic support-distance ranking. Default: all anchors.",
    )

    args = p.parse_args()

    if args.n_neighbors <= 0:
        raise ValueError("--n-neighbors must be > 0")
    if args.max_anchors is not None and args.max_anchors <= 0:
        raise ValueError("--max-anchors must be > 0 when provided")

    repo_root = Path(args.repo_root).resolve()
    train_dir = (
        resolve_path(repo_root, args.train_dir)
        if args.train_dir
        else repo_root / f"pipelines/empirical/integration/runs/scglue_{args.run_id}/train"
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
        / "pipelines/empirical/analysis"
        / "outputs"
        / "scglue_pairings"
        / sanitize_token(args.run_id)
        / (
            f"{sanitize_token(args.label_column)}__{sanitize_token(args.cell_type)}__"
            f"{sanitize_token(args.anchor_modality)}_anchor_neighbor_average_k{args.n_neighbors}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = train_dir / "all_cells_glue_embeddings.tsv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings table: {emb_path}")
    if label_tsv is None or not label_tsv.exists():
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

    merged = emb.merge(labels, on="obs_id", how="left", suffixes=("", "_label"))
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

    anchor_df = filtered.loc[filtered["modality_key"].astype(str) == args.anchor_modality].copy()
    if anchor_df.empty:
        raise RuntimeError(f"No anchor cells found for modality '{args.anchor_modality}'")

    requested_marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    modality_frames: dict[str, pd.DataFrame] = {}
    for mark in requested_marks:
        mod_df = filtered.loc[filtered["mark"].astype(str) == mark].copy()
        if mod_df.empty:
            raise RuntimeError(f"No cells found for mark '{mark}' after filtering")
        if mod_df.shape[0] < args.n_neighbors:
            raise ValueError(
                f"Requested {args.n_neighbors} neighbors for {mark}, but only {mod_df.shape[0]} cells are available"
            )
        modality_frames[mark] = mod_df

    anchor_x_all = anchor_df[glue_cols].to_numpy(dtype=float)
    support_df = anchor_df[["cell"]].copy()
    nearest_cols: list[str] = []
    max_cols: list[str] = []

    per_mark_neighbors: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for mark, mod_df in modality_frames.items():
        mod_x = mod_df[glue_cols].to_numpy(dtype=float)
        nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric="euclidean")
        nn.fit(mod_x)
        distances, indices = nn.kneighbors(anchor_x_all)
        per_mark_neighbors[mark] = (distances, indices, mod_df)
        mean_col = f"mean_neighbor_distance__{mark}"
        max_col = f"max_neighbor_distance__{mark}"
        support_df[mean_col] = distances.mean(axis=1).astype(float)
        support_df[max_col] = distances.max(axis=1).astype(float)
        nearest_cols.append(mean_col)
        max_cols.append(max_col)

    support_df["support_mean_neighbor_distance"] = support_df[nearest_cols].mean(axis=1)
    support_df["support_max_neighbor_distance"] = support_df[max_cols].max(axis=1)
    support_df = support_df.sort_values(
        ["support_mean_neighbor_distance", "support_max_neighbor_distance", "cell"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    support_df["anchor_rank"] = np.arange(1, support_df.shape[0] + 1, dtype=int)
    if args.max_anchors is not None:
        support_df = support_df.head(args.max_anchors).copy()

    selected_cells = set(support_df["cell"].astype(str))
    selected_anchor_df = anchor_df.loc[anchor_df["cell"].astype(str).isin(selected_cells)].copy()
    selected_anchor_df = selected_anchor_df.merge(
        support_df,
        on="cell",
        how="inner",
        suffixes=("", "_support"),
    )
    selected_anchor_df = selected_anchor_df.sort_values("anchor_rank").reset_index(drop=True)

    matched_samples = selected_anchor_df[
        ["cell", "anchor_rank", "support_mean_neighbor_distance", "support_max_neighbor_distance"]
    ].copy()
    matched_samples = matched_samples.rename(columns={"cell": f"cell_{args.anchor_modality}"})
    matched_samples.insert(0, "sample_id", [f"AVG_{i + 1:04d}" for i in range(matched_samples.shape[0])])
    matched_samples["mapping_strategy"] = "rna_anchor_neighbor_average"
    matched_samples["n_neighbors"] = int(args.n_neighbors)
    matched_samples["neighbor_weight_mode"] = args.neighbor_weight_mode

    anchor_meta_cols = [args.label_column]
    for extra in ["harmonized_fine", "label_source", "coarse_confidence", "fine_confidence", "cell_ontology_id"]:
        if extra in selected_anchor_df.columns:
            anchor_meta_cols.append(extra)
    anchor_meta = selected_anchor_df[["cell"] + anchor_meta_cols].copy().rename(
        columns={"cell": f"cell_{args.anchor_modality}"}
    )
    matched_samples = matched_samples.merge(anchor_meta, on=f"cell_{args.anchor_modality}", how="left")

    anchor_row_by_cell = {str(cell): idx for idx, cell in enumerate(anchor_df["cell"].astype(str).tolist())}

    neighbor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_lookup = dict(
        zip(
            matched_samples[f"cell_{args.anchor_modality}"].astype(str),
            matched_samples["sample_id"].astype(str),
        )
    )

    for mark, (distances, indices, mod_df) in per_mark_neighbors.items():
        mod_cells = mod_df["cell"].astype(str).to_numpy()
        selected_distances: list[float] = []
        for anchor_cell in matched_samples[f"cell_{args.anchor_modality}"].astype(str).tolist():
            anchor_idx = anchor_row_by_cell[anchor_cell]
            mark_distances = distances[anchor_idx, :]
            mark_indices = indices[anchor_idx, :]
            weights = compute_neighbor_weights(mark_distances.reshape(1, -1), args.neighbor_weight_mode).ravel()
            selected_distances.extend(mark_distances.astype(float).tolist())
            for rank, (neighbor_idx, distance, weight) in enumerate(
                zip(mark_indices, mark_distances, weights),
                start=1,
            ):
                neighbor_rows.append(
                    {
                        "sample_id": sample_lookup[anchor_cell],
                        f"cell_{args.anchor_modality}": anchor_cell,
                        "target_mark": mark,
                        "neighbor_rank": int(rank),
                        "matched_cell": str(mod_cells[int(neighbor_idx)]),
                        "match_distance": float(distance),
                        "weight": float(weight),
                    }
                )
        summary_rows.append(
            {
                "mark": mark,
                "n_anchor_samples": int(matched_samples.shape[0]),
                "n_neighbors": int(args.n_neighbors),
                "neighbor_weight_mode": args.neighbor_weight_mode,
                "mean_neighbor_distance": float(np.mean(selected_distances)),
                "median_neighbor_distance": float(np.median(selected_distances)),
                "max_neighbor_distance": float(np.max(selected_distances)),
            }
        )

    support_path = out_dir / "anchor_support.tsv"
    matched_path = out_dir / "matched_samples.tsv"
    neighbor_path = out_dir / "neighbor_assignments.tsv"
    summary_path = out_dir / "neighbor_summary_by_mark.tsv"
    run_summary_path = out_dir / "run_summary.json"

    support_df.to_csv(support_path, sep="\t", index=False)
    matched_samples.to_csv(matched_path, sep="\t", index=False)
    pd.DataFrame(neighbor_rows).to_csv(neighbor_path, sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

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
            "counts_by_modality": modality_summary(filtered),
        },
        "pairing": {
            "mapping_strategy": "rna_anchor_neighbor_average",
            "anchor_modality": args.anchor_modality,
            "marks": requested_marks,
            "n_anchor_candidates": int(anchor_df.shape[0]),
            "n_anchor_samples": int(matched_samples.shape[0]),
            "n_neighbors": int(args.n_neighbors),
            "neighbor_weight_mode": args.neighbor_weight_mode,
            "replacement": "with_replacement_across_rna_anchors",
            "selection_rule": "all_anchors_unless_max_anchors_is_set",
        },
        "distance_summary_by_mark": summary_rows,
        "outputs": {
            "out_dir": str(out_dir),
            "anchor_support_tsv": str(support_path),
            "matched_samples_tsv": str(matched_path),
            "neighbor_assignments_tsv": str(neighbor_path),
            "neighbor_summary_by_mark_tsv": str(summary_path),
        },
        "params": vars(args),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2) + "\n")

    print(f"Wrote: {support_path}")
    print(f"Wrote: {matched_path}")
    print(f"Wrote: {neighbor_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {run_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
