#!/usr/bin/env python3
"""Observed-only ordered-variance alignment for independently sampled assays.

Whitening makes independent PCA scores comparable only up to an arbitrary
orthogonal rotation.  When the observed state covariance has separated
eigenvalues, that destroys useful identifiability.  This prototype instead
keeps unwhitened PCA component order within every assay, verifies it with
disjoint background-feature folds using repeated cells only inside that assay,
and standardizes the ordered scores without rotating them across assays.

Marginal quantiles choose coordinate signs relative to RNA.  This compares
unpaired one-dimensional distributions and is immaterial to the projective
axis estimand, but gives mixed loadings a deterministic convention.  No focal
feature, scorer, simulator state, paired row, or cross-assay feature identity
enters the estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import soft_distributional_alignment as soft_alignment
import unpaired_state_alignment as hard_alignment


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrderedVarianceAlignmentConfig:
    """Frozen covariance-order and cross-fit abstention settings."""

    n_components: int = 3
    n_feature_folds: int = 2
    random_seed: int = 0
    min_fold_component_correlation: float = 0.60
    min_relative_variance_gap: float = 0.05
    abstention_uncertainty_penalty: float = 2.0
    sign_quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least two")
        if self.n_feature_folds < 2:
            raise ValueError("n_feature_folds must be at least two")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not 0.0 <= self.min_fold_component_correlation <= 1.0:
            raise ValueError("min_fold_component_correlation must lie in [0, 1]")
        if not 0.0 <= self.min_relative_variance_gap <= 1.0:
            raise ValueError("min_relative_variance_gap must lie in [0, 1]")
        if self.abstention_uncertainty_penalty <= 0.0:
            raise ValueError("abstention_uncertainty_penalty must be positive")
        if len(self.sign_quantiles) < 3 or any(
            not 0.0 < value < 1.0 for value in self.sign_quantiles
        ):
            raise ValueError("sign_quantiles must contain probabilities in (0, 1)")


@dataclass
class OrderedVarianceAlignmentResult:
    embeddings: dict[str, FloatArray]
    embedding_uncertainty: dict[str, FloatArray]
    assay_eligible: dict[str, bool]
    assay_alignment_confidence: dict[str, float]
    provenance: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, modality: str) -> int:
    digest = hashlib.sha256(f"{base_seed}::{modality}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _standardize_scores(values: FloatArray) -> FloatArray:
    return np.asarray(StandardScaler().fit_transform(values), dtype=np.float64)


def _fit_unwhitened_pca(
    matrix: FloatArray,
    depth: FloatArray,
    indices: NDArray[np.int64],
    n_components: int,
) -> tuple[FloatArray, FloatArray]:
    normalized = hard_alignment._normalize(matrix[:, indices], depth, "log_depth")
    model = PCA(n_components=n_components, whiten=False, svd_solver="full")
    scores = _standardize_scores(model.fit_transform(normalized))
    return scores, np.asarray(model.explained_variance_, dtype=np.float64)


def _match_fold_to_full(
    fold_scores: FloatArray,
    full_scores: FloatArray,
) -> tuple[FloatArray, FloatArray, list[int]]:
    dimensions = full_scores.shape[1]
    correlations = np.corrcoef(full_scores.T, fold_scores.T)[:dimensions, dimensions:]
    rows, columns = linear_sum_assignment(-np.abs(correlations))
    permutation = np.empty(dimensions, dtype=np.int64)
    for row, column in zip(rows, columns, strict=True):
        permutation[int(row)] = int(column)
    aligned = np.empty_like(full_scores)
    matched_correlation = np.empty(dimensions, dtype=np.float64)
    for dimension in range(dimensions):
        source = fold_scores[:, permutation[dimension]]
        correlation = float(correlations[dimension, permutation[dimension]])
        aligned[:, dimension] = source if correlation >= 0.0 else -source
        matched_correlation[dimension] = abs(correlation)
    return aligned, matched_correlation, permutation.tolist()


def _relative_variance_gaps(explained_variance: FloatArray) -> FloatArray:
    return np.asarray(
        (explained_variance[:-1] - explained_variance[1:])
        / np.maximum(explained_variance[:-1], 1e-12),
        dtype=np.float64,
    )


def align_ordered_variance_background_states(
    background_matrices: Mapping[str, FloatArray],
    background_depths: Mapping[str, FloatArray],
    *,
    reference_modality: str = "rna",
    config: OrderedVarianceAlignmentConfig | None = None,
) -> OrderedVarianceAlignmentResult:
    """Align assays by separated observed covariance ranks, not rotations."""

    cfg = config or OrderedVarianceAlignmentConfig()
    modalities = sorted(str(key) for key in background_matrices)
    if len(modalities) < 2 or set(modalities) != set(background_depths):
        raise ValueError("At least two assays with matching matrix/depth keys are required")
    if reference_modality not in modalities:
        raise ValueError("reference_modality is absent")
    embeddings: dict[str, FloatArray] = {}
    uncertainty: dict[str, FloatArray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    eligible: dict[str, bool] = {}
    confidence: dict[str, float] = {}
    for modality in modalities:
        matrix = np.asarray(background_matrices[modality], dtype=np.float64)
        depth = np.asarray(background_depths[modality], dtype=np.float64)
        if matrix.ndim != 2 or len(depth) != len(matrix):
            raise ValueError(f"Invalid observed background input for {modality}")
        if matrix.shape[1] < cfg.n_components * cfg.n_feature_folds:
            raise ValueError(f"Too few background features for {modality}")
        full_scores, full_variance = _fit_unwhitened_pca(
            matrix,
            depth,
            np.arange(matrix.shape[1], dtype=np.int64),
            cfg.n_components,
        )
        rng = np.random.default_rng(_stable_seed(cfg.random_seed, modality))
        folds = np.array_split(rng.permutation(matrix.shape[1]), cfg.n_feature_folds)
        aligned_folds: list[FloatArray] = []
        fold_rows: list[dict[str, Any]] = []
        all_correlations: list[float] = []
        for fold_index, feature_indices in enumerate(folds):
            fold_scores, fold_variance = _fit_unwhitened_pca(
                matrix,
                depth,
                np.asarray(feature_indices, dtype=np.int64),
                cfg.n_components,
            )
            aligned, correlations, permutation = _match_fold_to_full(
                fold_scores, full_scores
            )
            aligned_folds.append(aligned)
            all_correlations.extend(float(value) for value in correlations)
            fold_rows.append(
                {
                    "fold": fold_index,
                    "n_features": len(feature_indices),
                    "component_permutation_to_full": permutation,
                    "absolute_component_correlations": correlations.tolist(),
                    "explained_variance": fold_variance.tolist(),
                }
            )
        stack = np.stack(aligned_folds, axis=0)
        local_uncertainty = np.sqrt(
            np.mean(np.sum(np.square(stack - full_scores[None, :, :]), axis=2), axis=0)
        )
        gaps = _relative_variance_gaps(full_variance)
        minimum_correlation = float(np.min(all_correlations))
        minimum_gap = float(np.min(gaps))
        passed = bool(
            minimum_correlation >= cfg.min_fold_component_correlation
            and minimum_gap >= cfg.min_relative_variance_gap
        )
        eligible[modality] = passed
        confidence[modality] = (
            float(min(minimum_correlation, minimum_gap / 0.25, 1.0))
            if passed
            else 0.0
        )
        embeddings[modality] = full_scores
        uncertainty[modality] = np.sqrt(
            np.square(local_uncertainty)
            + (0.0 if passed else cfg.abstention_uncertainty_penalty**2)
        )
        diagnostics[modality] = {
            "n_cells": len(matrix),
            "n_background_features": matrix.shape[1],
            "full_explained_variance": full_variance.tolist(),
            "full_explained_variance_fraction_within_retained_rank": (
                full_variance / full_variance.sum()
            ).tolist(),
            "relative_variance_gaps": gaps.tolist(),
            "minimum_relative_variance_gap": minimum_gap,
            "minimum_fold_component_correlation": minimum_correlation,
            "feature_fold_diagnostics": fold_rows,
            "covariance_rank_eligible": passed,
            "alignment_confidence": confidence[modality],
        }

    # Choose coordinate signs by one-dimensional unpaired quantile signatures.
    # No rows are paired and no rotation/permutation is estimated here.
    quantiles = np.asarray(cfg.sign_quantiles, dtype=np.float64)
    reference = embeddings[reference_modality]
    reference_signatures = np.quantile(reference, quantiles, axis=0)
    for modality in modalities:
        sign_rows: list[dict[str, Any]] = []
        for dimension in range(cfg.n_components):
            values = embeddings[modality][:, dimension]
            positive = np.quantile(values, quantiles)
            negative = np.quantile(-values, quantiles)
            target = reference_signatures[:, dimension]
            positive_error = float(np.sqrt(np.mean(np.square(positive - target))))
            negative_error = float(np.sqrt(np.mean(np.square(negative - target))))
            sign = -1.0 if negative_error < positive_error else 1.0
            embeddings[modality][:, dimension] *= sign
            sign_rows.append(
                {
                    "dimension": dimension,
                    "selected_sign": int(sign),
                    "positive_quantile_rmse": positive_error,
                    "negative_quantile_rmse": negative_error,
                    "relative_sign_margin": abs(positive_error - negative_error)
                    / max(positive_error, negative_error, 1e-12),
                }
            )
        diagnostics[modality]["unpaired_marginal_sign_diagnostics"] = sign_rows

    provenance = {
        "schema_version": "ordered_variance_alignment_v4.v1",
        "method": "unwhitened_pca_covariance_rank_with_within_assay_feature_fold_stability",
        "reference_modality": reference_modality,
        "config": asdict(cfg),
        "assay_diagnostics": diagnostics,
        "assay_eligible": eligible,
        "assay_alignment_confidence": confidence,
        "accepted_assays": sorted(modality for modality in modalities if eligible[modality]),
        "abstained_assays": sorted(modality for modality in modalities if not eligible[modality]),
        "identifiability": {
            "whitening_used": False,
            "cross_assay_rotation_estimated": False,
            "component_order_source": "within_assay_observed_explained_variance_descending",
            "cross_assay_sign_source": "unpaired_marginal_quantile_rmse",
            "sign_is_irrelevant_to_projective_axis_estimand": True,
        },
        "information_boundary": {
            "consumed": [
                "observed_background_matrix_per_assay",
                "observed_leave_focal_out_depth_per_assay",
                "assay_key",
            ],
            "not_consumed": [
                "focal_features",
                "true_latent_state",
                "simulator_roles",
                "truth_graph",
                "scorer",
                "paired_cells",
                "cross_assay_feature_identity",
                "external_data",
            ],
            "same_assay_rows_used_for_feature_fold_checks": True,
            "cross_assay_rows_paired": False,
            "cross_assay_feature_columns_compared": False,
            "truth_or_scorer_used": False,
        },
    }
    return OrderedVarianceAlignmentResult(
        embeddings=embeddings,
        embedding_uncertainty=uncertainty,
        assay_eligible=eligible,
        assay_alignment_confidence=confidence,
        provenance=provenance,
    )


def freeze_ordered_variance_evidence(
    discovery_dir: Path | str,
    out_dir: Path | str,
    *,
    config: OrderedVarianceAlignmentConfig | None = None,
) -> dict[str, Any]:
    """Write content-addressed background-only evidence before scorer access."""

    cfg = config or OrderedVarianceAlignmentConfig()
    output = Path(out_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    root, matrices, depths, cell_ids, input_hashes = soft_alignment._load_background_archives(
        discovery_dir
    )
    if output.resolve(strict=False) == root or root in output.resolve(strict=False).parents:
        raise ValueError("Output must be outside the discovery input")
    output.mkdir(parents=True, exist_ok=True)
    result = align_ordered_variance_background_states(
        matrices, depths, reference_modality="rna", config=cfg
    )
    rows = []
    for modality in sorted(result.embeddings):
        coordinates = result.embeddings[modality]
        rows.append(
            pd.DataFrame(
                {
                    "modality": modality,
                    "cell": cell_ids[modality],
                    **{
                        f"integration__dim{dimension}": coordinates[:, dimension]
                        for dimension in range(coordinates.shape[1])
                    },
                    "integration_uncertainty": result.embedding_uncertainty[modality],
                }
            )
        )
    embedding_path = output / "analysis_embeddings.tsv"
    pd.concat(rows, ignore_index=True).to_csv(embedding_path, sep="\t", index=False)
    provenance_path = output / "alignment_provenance.json"
    provenance_path.write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    source = Path(__file__).resolve(strict=True)
    manifest = {
        "schema_version": "ordered_variance_alignment_v4.frozen_evidence.v1",
        "algorithm_source_sha256": _sha256(source),
        "config": asdict(cfg),
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            embedding_path.name: _sha256(embedding_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "frozen_before_scorer_read": True,
        "information_boundary": result.provenance["information_boundary"],
    }
    (output / "frozen_evidence_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = freeze_ordered_variance_evidence(
        args.discovery,
        args.out,
        config=OrderedVarianceAlignmentConfig(random_seed=args.random_seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OrderedVarianceAlignmentConfig",
    "OrderedVarianceAlignmentResult",
    "align_ordered_variance_background_states",
    "freeze_ordered_variance_evidence",
]
