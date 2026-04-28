#!/usr/bin/env python3
"""Analyze matched monocyte RNA raw counts for Poisson, NB1, and NB2 family fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from export_locus_matrix_scglue_matches import infer_repo_root, load_rna_counts, resolve_path, row_sums_for_matrix, sanitize_token


def poisson_loglik(y: np.ndarray, size_factors: np.ndarray) -> tuple[float, float]:
    beta = float(np.sum(y) / max(np.sum(size_factors), 1e-12))
    mu = np.clip(size_factors * beta, 1e-12, None)
    ll = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))
    return ll, beta


def nb2_loglik(y: np.ndarray, size_factors: np.ndarray, theta: float) -> tuple[float, float]:
    _, beta = poisson_loglik(y, size_factors)
    mu = np.clip(size_factors * beta, 1e-12, None)
    ll = np.sum(
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1.0)
        + theta * (np.log(theta) - np.log(theta + mu))
        + y * (np.log(mu) - np.log(theta + mu))
    )
    return float(ll), beta


def nb1_loglik(y: np.ndarray, size_factors: np.ndarray, alpha: float) -> tuple[float, float]:
    _, beta = poisson_loglik(y, size_factors)
    mu = np.clip(size_factors * beta, 1e-12, None)
    r = np.clip(mu / max(alpha, 1e-12), 1e-12, None)
    ll = np.sum(
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * (np.log(r) - np.log(r + mu))
        + y * (np.log(mu) - np.log(r + mu))
    )
    return float(ll), beta


def fit_nb2(y: np.ndarray, size_factors: np.ndarray) -> tuple[float, float, float]:
    theta_grid = np.logspace(-2, 6, 200)
    best_theta = float(theta_grid[0])
    best_ll = -np.inf
    best_beta = 0.0
    for theta in theta_grid:
        ll, beta = nb2_loglik(y, size_factors, float(theta))
        if ll > best_ll:
            best_ll = ll
            best_theta = float(theta)
            best_beta = float(beta)
    return best_ll, best_theta, best_beta


def fit_nb1(y: np.ndarray, size_factors: np.ndarray) -> tuple[float, float, float]:
    alpha_grid = np.logspace(-4, 4, 200)
    best_alpha = float(alpha_grid[0])
    best_ll = -np.inf
    best_beta = 0.0
    for alpha in alpha_grid:
        ll, beta = nb1_loglik(y, size_factors, float(alpha))
        if ll > best_ll:
            best_ll = ll
            best_alpha = float(alpha)
            best_beta = float(beta)
    return best_ll, best_alpha, best_beta


def aic(loglik: float, n_params: int) -> float:
    return float(2 * n_params - 2 * loglik)


def recommend_model(summary_df: pd.DataFrame) -> str:
    best_counts = summary_df["best_model_by_aic"].value_counts().to_dict()
    if best_counts.get("nb2", 0) >= 2:
        return "nb2"
    if best_counts.get("nb1", 0) >= 2:
        return "nb1"
    return "ambiguous"


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    *,
    run_id: str,
    genes: list[str],
    n_cells: int,
    library_size_stats: dict[str, float],
    summary_df: pd.DataFrame,
    recommendation: str,
) -> None:
    lines = [
        "# Matched RNA Count Family Analysis",
        "",
        "## Scope",
        "",
        f"- `run_id`: `{run_id}`",
        f"- genes: {', '.join(f'`{gene}`' for gene in genes)}",
        f"- matched monocyte RNA cells: `{n_cells}`",
        "",
        "## Library Size Context",
        "",
        f"- mean library size: `{library_size_stats['mean']:.2f}`",
        f"- std library size: `{library_size_stats['std']:.2f}`",
        f"- min library size: `{library_size_stats['min']:.2f}`",
        f"- max library size: `{library_size_stats['max']:.2f}`",
        "",
        "## Model Comparison",
        "",
        "The count-model comparison used an intercept-only log-link mean with RNA library-size offset and compared:",
        "",
        "- Poisson",
        "- NB1 with $\\operatorname{Var}(Y) = \\mu (1 + \\alpha)$",
        "- NB2 with $\\operatorname{Var}(Y) = \\mu + \\mu^2 / \\theta$",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Recommendation",
        "",
    ]

    if recommendation == "nb2":
        lines.extend(
            [
                "NB2 is the recommended first count family for the mixed-family DAGMA implementation.",
                "",
                "The matched monocyte RNA counts are clearly overdispersed relative to Poisson, and the NB2 fit is the best AIC choice for most of the focal genes. That is also consistent with the super-linear mean-variance growth we expect in this setting.",
            ]
        )
    elif recommendation == "nb1":
        lines.extend(
            [
                "NB1 is the recommended first count family for the mixed-family DAGMA implementation.",
                "",
                "The matched monocyte RNA counts are overdispersed and the NB1 fit is the best AIC choice for most of the focal genes.",
            ]
        )
    else:
        lines.extend(
            [
                "The NB1 versus NB2 choice is ambiguous on the focal genes.",
                "",
                "The current evidence still argues against Poisson, but the NB1 and NB2 fits are mixed enough that we should keep NB2 as the default implementation target and revisit with a broader matched-gene panel if needed.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Poisson, NB1, and NB2 fit on matched monocyte RNA counts.")
    parser.add_argument("--repo-root", default=str(infer_repo_root()))
    parser.add_argument("--run-id", default="joint_v2")
    parser.add_argument("--matching-dir", default=None)
    parser.add_argument("--rna-h5ad", default=None)
    parser.add_argument("--genes", default="CSF1R,CD14,IL1B")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    matching_dir = (
        resolve_path(repo_root, args.matching_dir)
        if args.matching_dir
        else repo_root / "CausalDiscovery" / "outputs" / "scglue_pairings" / sanitize_token(args.run_id) / "harmonized_coarse__monocyte__rna_anchor"
    )
    matched_path = matching_dir / "matched_samples.tsv"
    rna_h5ad = (
        resolve_path(repo_root, args.rna_h5ad)
        if args.rna_h5ad
        else repo_root / f"integration/outputs/scglue/joint/{args.run_id}/train/modalities/rna_with_glue.h5ad"
    )
    out_dir = (
        resolve_path(repo_root, args.out_dir)
        if args.out_dir
        else repo_root / "CausalDiscovery" / "outputs" / "analysis" / sanitize_token(args.run_id) / "matched_rna_count_families"
    )

    if matched_path is None or not matched_path.exists():
        raise FileNotFoundError(f"Missing matched samples: {matched_path}")
    if rna_h5ad is None or not rna_h5ad.exists():
        raise FileNotFoundError(f"Missing RNA h5ad: {rna_h5ad}")

    out_dir.mkdir(parents=True, exist_ok=True)
    genes = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    matched = pd.read_csv(matched_path, sep="\t")
    if "cell_rna" not in matched.columns:
        raise ValueError(f"Matched samples missing 'cell_rna': {matched_path}")

    rna_counts, rna_obs, rna_genes = load_rna_counts(rna_h5ad)
    obs_index = pd.Index(rna_obs, name="cell_rna")
    matched_cells = matched["cell_rna"].astype(str)
    missing_cells = matched_cells[~matched_cells.isin(obs_index)]
    if not missing_cells.empty:
        raise ValueError(f"Matched RNA barcodes were missing from RNA matrix: {missing_cells.iloc[:5].tolist()}")

    row_positions = obs_index.get_indexer(matched_cells)
    matched_counts = rna_counts[row_positions, :]
    library_sizes = row_sums_for_matrix(matched_counts)
    library_size_scale = np.clip(library_sizes / max(np.median(library_sizes[library_sizes > 0]), 1.0), 1e-6, None)

    gene_to_idx = {gene: idx for idx, gene in enumerate(rna_genes)}
    rows = []
    for gene in genes:
        if gene not in gene_to_idx:
            rows.append(
                {
                    "gene": gene,
                    "present": False,
                    "error": "missing_from_rna_matrix",
                }
            )
            continue

        y = np.asarray(matched_counts[:, gene_to_idx[gene]].toarray()).ravel().astype(float)
        mean = float(np.mean(y))
        var = float(np.var(y, ddof=1)) if y.shape[0] > 1 else 0.0
        zero_frac = float(np.mean(y == 0))

        poisson_ll, poisson_beta = poisson_loglik(y, library_size_scale)
        nb1_ll, nb1_alpha, nb1_beta = fit_nb1(y, library_size_scale)
        nb2_ll, nb2_theta, nb2_beta = fit_nb2(y, library_size_scale)

        poisson_aic = aic(poisson_ll, 1)
        nb1_aic = aic(nb1_ll, 2)
        nb2_aic = aic(nb2_ll, 2)
        best_model = min(
            [("poisson", poisson_aic), ("nb1", nb1_aic), ("nb2", nb2_aic)],
            key=lambda item: item[1],
        )[0]

        rows.append(
            {
                "gene": gene,
                "present": True,
                "n_cells": int(y.shape[0]),
                "mean_raw_counts": mean,
                "variance_raw_counts": var,
                "variance_to_mean": float(var / mean) if mean > 0 else np.nan,
                "zero_fraction": zero_frac,
                "poisson_loglik": poisson_ll,
                "nb1_loglik": nb1_ll,
                "nb2_loglik": nb2_ll,
                "poisson_aic": poisson_aic,
                "nb1_aic": nb1_aic,
                "nb2_aic": nb2_aic,
                "poisson_beta": poisson_beta,
                "nb1_alpha": nb1_alpha,
                "nb1_beta": nb1_beta,
                "nb2_theta": nb2_theta,
                "nb2_beta": nb2_beta,
                "best_model_by_aic": best_model,
            }
        )

    summary_df = pd.DataFrame(rows)
    present_df = summary_df.loc[summary_df.get("present", False) == True].copy()  # noqa: E712
    recommendation = recommend_model(present_df) if not present_df.empty else "ambiguous"

    summary_path = out_dir / "matched_rna_family_summary.tsv"
    report_path = out_dir / "matched_rna_family_report.md"
    manifest_path = out_dir / "matched_rna_family_report.json"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    library_stats = {
        "mean": float(np.mean(library_sizes)),
        "std": float(np.std(library_sizes, ddof=1)),
        "min": float(np.min(library_sizes)),
        "max": float(np.max(library_sizes)),
    }
    write_markdown_report(
        report_path,
        run_id=args.run_id,
        genes=genes,
        n_cells=int(matched.shape[0]),
        library_size_stats=library_stats,
        summary_df=summary_df,
        recommendation=recommendation,
    )
    manifest = {
        "run_id": args.run_id,
        "matching_dir": str(matching_dir),
        "rna_h5ad": str(rna_h5ad),
        "genes": genes,
        "n_cells": int(matched.shape[0]),
        "library_size_stats": library_stats,
        "recommendation": recommendation,
        "outputs": {
            "summary_tsv": str(summary_path),
            "report_md": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
