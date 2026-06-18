#!/usr/bin/env python3
"""Cross-fit elastic-net screening for nearby CUT&Tag peak matrices."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import copy_if_exists, infer_repo_root, json_dump, load_panel_genes, resolve_path, sanitize_token, write_bed


def parse_float_list(value: str) -> list[float]:
    vals = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def infer_report_root(repo_root: Path, dataset_root: Path, top_k: int) -> Path:
    outputs_root = repo_root / "CausalDiscovery" / "outputs"
    try:
        rel = dataset_root.resolve().relative_to((outputs_root / "datasets").resolve())
        parts = rel.parts
        dataset_id = sanitize_token("__".join(parts[:2])) if len(parts) >= 2 else sanitize_token(dataset_root.name)
    except ValueError:
        dataset_id = sanitize_token(dataset_root.name)
    return outputs_root / "methods" / "elastic_net_screen" / dataset_id / f"top{top_k}_crossfit"


def peak_sort_tuple(row: pd.Series) -> tuple[str, int, int, str, str]:
    chrom = str(row.get("chrom", ""))
    start = int(row.get("start", 0)) if pd.notna(row.get("start", np.nan)) else 0
    end = int(row.get("end", 0)) if pd.notna(row.get("end", np.nan)) else 0
    mark = str(row.get("mark", ""))
    name = str(row.get("variable_name", ""))
    return chrom, start, end, mark, name


def source_paths(dataset_root: Path, gene: str) -> dict[str, Path]:
    gene_token = sanitize_token(gene)
    src_dir = dataset_root / gene_token / "nearby_peaks"
    return {
        "src_dir": src_dir,
        "matrix": src_dir / f"{gene}_nearby_peak_matrix.tsv",
        "metadata": src_dir / f"{gene}_nearby_peak_metadata.tsv",
        "bins": src_dir / f"{gene}_nearby_peak_bins.tsv",
        "curated_tsv": src_dir / f"{gene}_evaluation_curated_regions.tsv",
        "curated_bed": src_dir / f"{gene}_evaluation_curated_regions.bed",
        "summary": src_dir / "run_summary.json",
    }


def output_paths(out_root: Path, gene: str) -> dict[str, Path]:
    gene_token = sanitize_token(gene)
    out_dir = out_root / gene_token / "nearby_peaks"
    return {
        "out_dir": out_dir,
        "matrix": out_dir / f"{gene}_nearby_peak_matrix.tsv",
        "metadata": out_dir / f"{gene}_nearby_peak_metadata.tsv",
        "bins": out_dir / f"{gene}_nearby_peak_bins.tsv",
        "bed": out_dir / f"{gene}_nearby_candidate_peaks.bed",
        "curated_tsv": out_dir / f"{gene}_evaluation_curated_regions.tsv",
        "curated_bed": out_dir / f"{gene}_evaluation_curated_regions.bed",
        "summary": out_dir / "run_summary.json",
    }


def score_grid_for_split(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    l1_ratios: list[float],
    alphas: np.ndarray,
    max_iter: int,
    tol: float,
    random_state: int,
    coef_tol: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    coefs: list[np.ndarray] = []
    x_train = x[train_idx, :]
    x_test = x[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            model = Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "elastic_net",
                        ElasticNet(
                            alpha=float(alpha),
                            l1_ratio=float(l1_ratio),
                            fit_intercept=True,
                            max_iter=int(max_iter),
                            tol=float(tol),
                            random_state=int(random_state),
                            selection="cyclic",
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred)) if np.unique(y_test).shape[0] > 1 else float("nan")
            coef = np.asarray(model.named_steps["elastic_net"].coef_, dtype=float)
            n_warnings = sum(1 for warning in caught if issubclass(warning.category, ConvergenceWarning))
            rows.append(
                {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "rmse": rmse,
                    "r2": r2,
                    "n_nonzero": int(np.sum(np.abs(coef) > coef_tol)),
                    "n_convergence_warnings": int(n_warnings),
                }
            )
            coefs.append(coef)

    scores = pd.DataFrame(rows)
    scores["rmse_rank"] = scores["rmse"].rank(method="min", ascending=True)
    scores["r2_rank"] = scores["r2"].rank(method="min", ascending=False, na_option="bottom")
    scores["rank_average"] = (scores["rmse_rank"] + scores["r2_rank"]) / 2.0
    sort_cols = ["rank_average", "rmse", "r2", "n_nonzero", "alpha", "l1_ratio"]
    ascending = [True, True, False, False, True, True]
    best_idx = int(scores.sort_values(sort_cols, ascending=ascending, kind="mergesort").index[0])
    scores["is_winner"] = False
    scores.loc[best_idx, "is_winner"] = True
    return scores, coefs[best_idx]


def screen_one_gene(
    *,
    repo_root: Path,
    dataset_root: Path,
    out_root: Path,
    report_root: Path,
    gene: str,
    top_k: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    l1_ratios: list[float],
    alphas: np.ndarray,
    max_iter: int,
    tol: float,
    coef_tol: float,
) -> dict[str, Any]:
    paths = source_paths(dataset_root, gene)
    for key in ["matrix", "metadata"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {key} for {gene}: {paths[key]}")

    matrix = pd.read_csv(paths["matrix"], sep="\t")
    metadata = pd.read_csv(paths["metadata"], sep="\t")
    if "variable_name" not in metadata.columns:
        raise ValueError(f"Metadata missing variable_name column: {paths['metadata']}")

    expr_col = f"expr__{gene}_log1p"
    if expr_col not in matrix.columns:
        raise ValueError(f"Missing expression outcome column {expr_col}: {paths['matrix']}")

    candidate_peak_cols = metadata["variable_name"].astype(str).tolist()
    peak_cols = [
        col
        for col in candidate_peak_cols
        if col in matrix.columns and "__H3" in col and not col.startswith("libsize__")
    ]
    if not peak_cols:
        raise ValueError(f"No nearby peak columns found for {gene}: {paths['matrix']}")

    numeric = matrix[peak_cols + [expr_col]].apply(pd.to_numeric, errors="coerce")
    valid_rows = numeric.notna().all(axis=1).to_numpy()
    if not bool(valid_rows.all()):
        raise ValueError(
            f"{gene} has missing/non-numeric values in elastic-net inputs; "
            "clean the matrix or extend the screen to drop rows explicitly."
        )

    x = numeric[peak_cols].to_numpy(dtype=float)
    y = numeric[expr_col].to_numpy(dtype=float)
    rkf = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=int(random_state))

    cv_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    selected_count = np.zeros(len(peak_cols), dtype=int)
    sum_abs_coef = np.zeros(len(peak_cols), dtype=float)
    sum_signed_coef = np.zeros(len(peak_cols), dtype=float)
    positive_count = np.zeros(len(peak_cols), dtype=int)
    negative_count = np.zeros(len(peak_cols), dtype=int)

    split_id = 0
    for train_idx, test_idx in rkf.split(x):
        repeat_idx = split_id // int(n_splits)
        fold_idx = split_id % int(n_splits)
        scores, winner_coef = score_grid_for_split(
            x,
            y,
            train_idx,
            test_idx,
            l1_ratios=l1_ratios,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state + split_id,
            coef_tol=coef_tol,
        )
        scores.insert(0, "fold", fold_idx)
        scores.insert(0, "repeat", repeat_idx)
        scores.insert(0, "split_id", split_id)
        cv_rows.append(scores)

        selected = np.abs(winner_coef) > coef_tol
        selected_count += selected.astype(int)
        sum_abs_coef += np.abs(winner_coef) * selected
        sum_signed_coef += winner_coef * selected
        positive_count += ((winner_coef > coef_tol) & selected).astype(int)
        negative_count += ((winner_coef < -coef_tol) & selected).astype(int)

        for row_idx in test_idx:
            fold_rows.append(
                {
                    "split_id": split_id,
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "row_index": int(row_idx),
                    "sample_id": matrix.loc[row_idx, "sample_id"] if "sample_id" in matrix.columns else "",
                    "role": "held_out",
                }
            )
        split_id += 1

    total_splits = int(n_splits) * int(n_repeats)
    cv_scores = pd.concat(cv_rows, ignore_index=True)
    fold_assignments = pd.DataFrame(fold_rows)

    metrics = []
    metadata_lookup = metadata.set_index(metadata["variable_name"].astype(str), drop=False)
    for idx, col in enumerate(peak_cols):
        series = pd.to_numeric(matrix[col], errors="coerce")
        selected_n = int(selected_count[idx])
        mean_abs = float(sum_abs_coef[idx] / selected_n) if selected_n else 0.0
        mean_signed = float(sum_signed_coef[idx] / selected_n) if selected_n else 0.0
        sign_stability = float(max(positive_count[idx], negative_count[idx]) / selected_n) if selected_n else 0.0
        row = metadata_lookup.loc[col] if col in metadata_lookup.index else pd.Series({"variable_name": col})
        sort_key = peak_sort_tuple(row)
        metrics.append(
            {
                "variable_name": col,
                "selection_count": selected_n,
                "selection_frequency": float(selected_n / total_splits),
                "mean_abs_coefficient_when_selected": mean_abs,
                "mean_signed_coefficient_when_selected": mean_signed,
                "positive_selection_count": int(positive_count[idx]),
                "negative_selection_count": int(negative_count[idx]),
                "sign_stability": sign_stability,
                "nonzero_fraction": float((series.dropna() > 0).mean()) if series.dropna().shape[0] else 0.0,
                "std": float(series.dropna().std(ddof=1)) if series.dropna().shape[0] > 1 else 0.0,
                "overlaps_curated_region": bool(row.get("overlaps_curated_region", False)),
                "overlapping_curated_regions": row.get("overlapping_curated_regions", ""),
                "chrom": sort_key[0],
                "start": sort_key[1],
                "end": sort_key[2],
                "mark": sort_key[3],
            }
        )

    ranking = pd.DataFrame(metrics)
    ranking = ranking.sort_values(
        ["selection_frequency", "mean_abs_coefficient_when_selected", "chrom", "start", "end", "mark", "variable_name"],
        ascending=[False, False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, ranking.shape[0] + 1))
    selected_peak_cols = ranking.head(min(int(top_k), ranking.shape[0]))["variable_name"].astype(str).tolist()
    selected_set = set(selected_peak_cols)
    ranking["selected"] = ranking["variable_name"].astype(str).isin(selected_set)

    dropped = ranking.loc[~ranking["selected"]].copy()
    dropped["drop_reason"] = f"not in top {top_k} by elastic-net cross-fit selection frequency"

    non_peak_cols = [col for col in matrix.columns if col not in peak_cols]
    screened_matrix = matrix.loc[:, non_peak_cols + selected_peak_cols].copy()
    retained_metadata = metadata.loc[metadata["variable_name"].astype(str).isin(selected_set)].copy()
    retained_metadata = retained_metadata.merge(
        ranking[
            [
                "variable_name",
                "rank",
                "selection_count",
                "selection_frequency",
                "mean_abs_coefficient_when_selected",
                "mean_signed_coefficient_when_selected",
                "sign_stability",
            ]
        ],
        on="variable_name",
        how="left",
    ).sort_values(["rank", "variable_name"], kind="mergesort")

    bins = pd.read_csv(paths["bins"], sep="\t") if paths["bins"].exists() else pd.DataFrame()
    retained_bins = (
        bins.loc[bins["variable_name"].astype(str).isin(selected_set)].copy()
        if not bins.empty and "variable_name" in bins.columns
        else bins
    )

    out_paths = output_paths(out_root, gene)
    report_dir = report_root / sanitize_token(gene)
    out_paths["out_dir"].mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    screened_matrix.to_csv(out_paths["matrix"], sep="\t", index=False)
    retained_metadata.to_csv(out_paths["metadata"], sep="\t", index=False)
    retained_bins.to_csv(out_paths["bins"], sep="\t", index=False)
    write_bed(retained_metadata, out_paths["bed"])
    copy_if_exists(paths["curated_tsv"], out_paths["curated_tsv"])
    copy_if_exists(paths["curated_bed"], out_paths["curated_bed"])

    ranking_path = report_dir / f"{gene}_elasticnet_peak_rankings.tsv"
    selected_path = report_dir / f"{gene}_elasticnet_selected_peaks.tsv"
    dropped_path = report_dir / f"{gene}_elasticnet_dropped_peaks.tsv"
    cv_path = report_dir / f"{gene}_elasticnet_cv_scores.tsv"
    fold_path = report_dir / f"{gene}_elasticnet_fold_assignments.tsv"
    winners_path = report_dir / f"{gene}_elasticnet_winning_hyperparameters.tsv"

    ranking.to_csv(ranking_path, sep="\t", index=False)
    ranking.loc[ranking["selected"]].to_csv(selected_path, sep="\t", index=False)
    dropped.to_csv(dropped_path, sep="\t", index=False)
    cv_scores.to_csv(cv_path, sep="\t", index=False)
    fold_assignments.to_csv(fold_path, sep="\t", index=False)
    cv_scores.loc[cv_scores["is_winner"]].to_csv(winners_path, sep="\t", index=False)

    source_summary: dict[str, Any] = {}
    if paths["summary"].exists():
        source_summary = json.loads(paths["summary"].read_text())

    summary = {
        **source_summary,
        "dataset_kind": "nearby_peaks_elastic_net_crossfit_screened",
        "source_dataset_root": str(dataset_root),
        "source_matrix_tsv": str(paths["matrix"]),
        "screening_method": "elastic_net_crossfit",
        "screening_note": "Association screen only; not causal evidence.",
        "target_column": expr_col,
        "predictor_source": "CUT&Tag nearby peak columns only",
        "n_samples": int(screened_matrix.shape[0]),
        "n_non_peak_columns": int(len(non_peak_cols)),
        "n_source_peak_variables": int(len(peak_cols)),
        "n_retained_peak_variables": int(len(selected_peak_cols)),
        "n_dropped_peak_variables": int(dropped.shape[0]),
        "selection": {
            "top_k": int(top_k),
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "n_total_splits": int(total_splits),
            "random_state": int(random_state),
            "l1_ratios": [float(v) for v in l1_ratios],
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "n_alphas": int(alphas.shape[0]),
            "metric": "rank_average of held-out RMSE rank and held-out R2 rank",
            "coef_tol": float(coef_tol),
        },
        "outputs": {
            "nearby_peak_matrix_tsv": str(out_paths["matrix"]),
            "nearby_peak_metadata_tsv": str(out_paths["metadata"]),
            "nearby_peak_bins_tsv": str(out_paths["bins"]),
            "nearby_candidate_peaks_bed": str(out_paths["bed"]),
            "peak_rankings_tsv": str(ranking_path),
            "selected_peaks_tsv": str(selected_path),
            "dropped_peaks_tsv": str(dropped_path),
            "cv_scores_tsv": str(cv_path),
            "fold_assignments_tsv": str(fold_path),
            "winning_hyperparameters_tsv": str(winners_path),
        },
    }
    json_dump(out_paths["summary"], summary)
    json_dump(report_dir / "run_summary.json", summary)

    return {
        "gene": gene,
        "source_peak_variables": int(len(peak_cols)),
        "retained_peak_variables": int(len(selected_peak_cols)),
        "dropped_peak_variables": int(dropped.shape[0]),
        "matrix_tsv": str(out_paths["matrix"]),
        "report_dir": str(report_dir),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Screen nearby peak datasets with cross-fit elastic net")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--report-root", default=None)
    p.add_argument("--gene-panel", required=True)
    p.add_argument("--genes", default=None, help="Comma-separated genes. Defaults to --gene-panel.")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--random-state", type=int, default=13)
    p.add_argument("--l1-ratios", default="0.1,0.25,0.5,0.75,0.9,0.95,1.0")
    p.add_argument("--alpha-min-exp", type=float, default=-4.0)
    p.add_argument("--alpha-max-exp", type=float, default=1.0)
    p.add_argument("--n-alphas", type=int, default=60)
    p.add_argument("--max-iter", type=int, default=20000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--coef-tol", type=float, default=1e-10)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = resolve_path(repo_root, args.dataset_root)
    out_root = resolve_path(repo_root, args.out_root)
    gene_panel = resolve_path(repo_root, args.gene_panel)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {args.dataset_root}")
    if out_root is None:
        raise ValueError("--out-root is required")
    if gene_panel is None or not gene_panel.exists():
        raise FileNotFoundError(f"Missing gene panel: {args.gene_panel}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1")
    if args.n_alphas < 1:
        raise ValueError("--n-alphas must be at least 1")

    report_root = resolve_path(repo_root, args.report_root) if args.report_root else infer_report_root(repo_root, dataset_root, args.top_k)
    if report_root is None:
        raise ValueError("--report-root resolved to None")

    if args.genes:
        genes = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    else:
        genes = load_panel_genes(gene_panel)

    l1_ratios = parse_float_list(args.l1_ratios)
    alphas = np.logspace(float(args.alpha_min_exp), float(args.alpha_max_exp), int(args.n_alphas))
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for gene in genes:
        print(f"[{gene}] elastic-net cross-fit screen: top_k={args.top_k}", flush=True)
        rows.append(
            screen_one_gene(
                repo_root=repo_root,
                dataset_root=dataset_root,
                out_root=out_root,
                report_root=report_root,
                gene=gene,
                top_k=args.top_k,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.random_state,
                l1_ratios=l1_ratios,
                alphas=alphas,
                max_iter=args.max_iter,
                tol=args.tol,
                coef_tol=args.coef_tol,
            )
        )

    source_panel_summary = dataset_root / "panel_summary.json"
    panel_summary: dict[str, Any] = {}
    if source_panel_summary.exists():
        panel_summary = json.loads(source_panel_summary.read_text())
    panel_summary.update(
        {
            "dataset_kind": "nearby_peaks_elastic_net_crossfit_screened",
            "source_dataset_root": str(dataset_root),
            "report_root": str(report_root),
            "screening_method": "elastic_net_crossfit",
            "screening_note": "Association screen only; not causal evidence.",
            "selection": {
                "top_k": int(args.top_k),
                "n_splits": int(args.n_splits),
                "n_repeats": int(args.n_repeats),
                "random_state": int(args.random_state),
                "l1_ratios": [float(v) for v in l1_ratios],
                "alpha_min": float(np.min(alphas)),
                "alpha_max": float(np.max(alphas)),
                "n_alphas": int(alphas.shape[0]),
                "metric": "rank_average of held-out RMSE rank and held-out R2 rank",
            },
            "genes": rows,
        }
    )
    json_dump(out_root / "panel_summary.json", panel_summary)
    json_dump(report_root / "panel_summary.json", panel_summary)
    print(f"Wrote: {out_root / 'panel_summary.json'}")
    print(f"Wrote reports: {report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
