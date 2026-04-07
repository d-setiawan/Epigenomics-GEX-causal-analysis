#!/usr/bin/env python3
"""Build RNA-anchored one-to-one pseudo-matches from joint scGLUE embeddings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


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


def modality_summary(df: pd.DataFrame) -> dict[str, int]:
    counts = df["modality_key"].astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def select_anchor_cells(
    anchor_df: pd.DataFrame,
    modality_frames: dict[str, pd.DataFrame],
    glue_cols: list[str],
    n_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_x = anchor_df[glue_cols].to_numpy(dtype=float)
    support_df = anchor_df[["cell"]].copy()
    nearest_cols: list[str] = []

    for mark, mod_df in modality_frames.items():
        mod_x = mod_df[glue_cols].to_numpy(dtype=float)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(mod_x)
        distances, _ = nn.kneighbors(anchor_x)
        col = f"nearest_distance__{mark}"
        support_df[col] = distances.ravel().astype(float)
        nearest_cols.append(col)

    support_df["support_mean_nn_distance"] = support_df[nearest_cols].mean(axis=1)
    support_df["support_max_nn_distance"] = support_df[nearest_cols].max(axis=1)
    support_df = support_df.sort_values(
        ["support_mean_nn_distance", "support_max_nn_distance", "cell"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    support_df["anchor_rank"] = np.arange(1, support_df.shape[0] + 1, dtype=int)

    selected = support_df.head(n_pairs).copy()
    selected_cells = set(selected["cell"].astype(str))
    selected_anchor_df = anchor_df.loc[anchor_df["cell"].astype(str).isin(selected_cells)].copy()
    selected_anchor_df = selected_anchor_df.merge(selected, on="cell", how="inner")
    selected_anchor_df = selected_anchor_df.sort_values("anchor_rank").reset_index(drop=True)
    return selected_anchor_df, support_df


def main() -> int:
    p = argparse.ArgumentParser(description="Build RNA-anchored one-to-one pseudo-matches from joint scGLUE outputs")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--run-id", default="joint_v2")
    p.add_argument("--train-dir", default=None, help="Default: integration/outputs/scglue/joint/<RUN_ID>/train")
    p.add_argument(
        "--label-tsv",
        default=None,
        help="Default: <train-dir>/validation/joint_harmonized_label_transfer.tsv",
    )
    p.add_argument("--out-dir", default=None, help="Default: CausalDiscovery/outputs/scglue_pairings/<RUN_ID>/<LABEL>__<CELL_TYPE>__rna_anchor")

    p.add_argument("--label-column", default="harmonized_coarse")
    p.add_argument("--cell-type", default="monocyte")
    p.add_argument("--min-label-confidence", type=float, default=0.5)
    p.add_argument("--anchor-modality", default="rna")
    p.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p.add_argument("--n-pairs", type=int, default=None, help="Default: min(anchor count, target modality counts)")

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
        / "scglue_pairings"
        / sanitize_token(args.run_id)
        / f"{sanitize_token(args.label_column)}__{sanitize_token(args.cell_type)}__{sanitize_token(args.anchor_modality)}_anchor"
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

    anchor_df = filtered.loc[filtered["modality_key"].astype(str) == args.anchor_modality].copy()
    if anchor_df.empty:
        raise RuntimeError(f"No anchor cells found for modality '{args.anchor_modality}'")

    requested_marks = [m.strip() for m in args.marks.split(",") if m.strip()]
    modality_frames: dict[str, pd.DataFrame] = {}
    for mark in requested_marks:
        mod_df = filtered.loc[filtered["mark"].astype(str) == mark].copy()
        if mod_df.empty:
            raise RuntimeError(f"No cells found for mark '{mark}' after filtering")
        modality_frames[mark] = mod_df

    max_pairs = min(
        [int(anchor_df.shape[0])] + [int(mod_df.shape[0]) for mod_df in modality_frames.values()]
    )
    n_pairs = args.n_pairs if args.n_pairs is not None else max_pairs
    if n_pairs <= 0:
        raise ValueError("--n-pairs must be > 0")
    if n_pairs > max_pairs:
        raise ValueError(f"Requested {n_pairs} pairs but max possible is {max_pairs}")

    selected_anchor_df, support_df = select_anchor_cells(
        anchor_df=anchor_df,
        modality_frames=modality_frames,
        glue_cols=glue_cols,
        n_pairs=n_pairs,
    )
    anchor_x = selected_anchor_df[glue_cols].to_numpy(dtype=float)

    paired_wide = selected_anchor_df[
        ["cell", "anchor_rank", "support_mean_nn_distance", "support_max_nn_distance"]
    ].copy()
    paired_wide = paired_wide.rename(columns={"cell": f"cell_{args.anchor_modality}"})
    paired_wide.insert(0, "sample_id", [f"PAIR_{i + 1:04d}" for i in range(paired_wide.shape[0])])

    assignment_rows: list[dict[str, Any]] = []
    match_summary_rows: list[dict[str, Any]] = []

    for mark, mod_df in modality_frames.items():
        mod_x = mod_df[glue_cols].to_numpy(dtype=float)
        cost = cdist(anchor_x, mod_x, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(cost)
        if row_ind.size != n_pairs:
            raise RuntimeError(f"Expected {n_pairs} matches for {mark}, got {row_ind.size}")

        matched_cells = mod_df.iloc[col_ind]["cell"].astype(str).to_numpy()
        matched_dist = cost[row_ind, col_ind].astype(float)
        order = np.argsort(row_ind)
        matched_cells = matched_cells[order]
        matched_dist = matched_dist[order]
        row_ind = row_ind[order]
        if not np.array_equal(row_ind, np.arange(n_pairs)):
            raise RuntimeError(f"Anchor rows for {mark} are not aligned after assignment")

        paired_wide[f"cell_{mark}"] = matched_cells
        paired_wide[f"distance_{mark}"] = matched_dist

        for idx in range(n_pairs):
            assignment_rows.append(
                {
                    "sample_id": paired_wide.iloc[idx]["sample_id"],
                    f"cell_{args.anchor_modality}": paired_wide.iloc[idx][f"cell_{args.anchor_modality}"],
                    "target_mark": mark,
                    "matched_cell": matched_cells[idx],
                    "match_distance": float(matched_dist[idx]),
                }
            )

        match_summary_rows.append(
            {
                "mark": mark,
                "n_pairs": int(n_pairs),
                "mean_distance": float(np.mean(matched_dist)),
                "median_distance": float(np.median(matched_dist)),
                "max_distance": float(np.max(matched_dist)),
            }
        )

    anchor_meta_cols = [args.label_column]
    for extra in ["harmonized_fine", "label_source", "coarse_confidence", "fine_confidence", "cell_ontology_id"]:
        if extra in selected_anchor_df.columns:
            anchor_meta_cols.append(extra)
    anchor_meta = selected_anchor_df[["cell"] + anchor_meta_cols].copy().rename(columns={"cell": f"cell_{args.anchor_modality}"})
    paired_wide = paired_wide.merge(anchor_meta, on=f"cell_{args.anchor_modality}", how="left")

    paired_wide["mean_match_distance"] = paired_wide[[f"distance_{mark}" for mark in requested_marks]].mean(axis=1)
    paired_wide["max_match_distance"] = paired_wide[[f"distance_{mark}" for mark in requested_marks]].max(axis=1)

    support_path = out_dir / "anchor_support.tsv"
    paired_path = out_dir / "matched_samples.tsv"
    assignment_path = out_dir / "pair_assignments.tsv"
    match_summary_path = out_dir / "match_summary_by_mark.tsv"
    summary_json_path = out_dir / "run_summary.json"

    support_df.to_csv(support_path, sep="\t", index=False)
    paired_wide.to_csv(paired_path, sep="\t", index=False)
    pd.DataFrame(assignment_rows).to_csv(assignment_path, sep="\t", index=False)
    pd.DataFrame(match_summary_rows).to_csv(match_summary_path, sep="\t", index=False)

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
            "anchor_modality": args.anchor_modality,
            "marks": requested_marks,
            "n_anchor_candidates": int(anchor_df.shape[0]),
            "n_pairs": int(n_pairs),
            "selection_rule": "lowest_mean_nearest_neighbor_distance_across_modalities",
            "max_possible_pairs": int(max_pairs),
        },
        "distance_summary_by_mark": match_summary_rows,
        "outputs": {
            "out_dir": str(out_dir),
            "anchor_support_tsv": str(support_path),
            "matched_samples_tsv": str(paired_path),
            "pair_assignments_tsv": str(assignment_path),
            "match_summary_by_mark_tsv": str(match_summary_path),
        },
        "params": vars(args),
    }
    with summary_json_path.open("w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Wrote: {support_path}")
    print(f"Wrote: {paired_path}")
    print(f"Wrote: {assignment_path}")
    print(f"Wrote: {match_summary_path}")
    print(f"Wrote: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
