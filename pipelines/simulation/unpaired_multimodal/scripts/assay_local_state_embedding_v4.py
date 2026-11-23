#!/usr/bin/env python3
"""Formal assay-local background-state embedding for CA-BOSS v4.

The simulator's biological state rank (three) is predeclared.  This module
estimates that state separately inside every assay from held-out background
features only.  Background columns are split in two, each fold receives a
rank-three PCA, and the two score bases are reconciled by orthogonal
Procrustes using repeated cells *within the same assay*.  No cell or feature
identity is compared across assays.

Parallel analysis is a diagnostic abstention guard, not a rank selector.  Both
feature folds of every assay must support at least the predeclared rank.  A
fresh world that does not meet that condition fails closed rather than padding
or silently changing the estimand.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]
SCHEMA_VERSION = "assay_local_state_embedding_v4.v1"
FROZEN_EVIDENCE_SCHEMA_VERSION = (
    "assay_local_state_embedding_v4.frozen_evidence.v1"
)
EXPECTED_MODALITIES = frozenset(
    {
        "rna",
        "chrom_H3K27ac",
        "chrom_H3K27me3",
        "chrom_H3K4me1",
        "chrom_H3K4me2",
        "chrom_H3K4me3",
        "chrom_H3K9me3",
    }
)
BACKGROUND_PREFIX = "background_matrix__"
EMBEDDING_NAME = "analysis_embeddings.tsv"
PROVENANCE_NAME = "assay_local_state_provenance.json"
MANIFEST_NAME = "frozen_evidence_manifest.json"


class AssayLocalStateAbstention(ValueError):
    """Raised when observed background evidence cannot support formal rank three."""


@dataclass(frozen=True)
class AssayLocalStateEmbeddingConfig:
    """Frozen observed-only settings for the formal simulator embedding."""

    state_dimension: int = 3
    n_feature_folds: int = 2
    parallel_max_dimension: int = 8
    parallel_repetitions: int = 99
    parallel_quantile: float = 0.95
    random_seed: int = 24_082_101

    def __post_init__(self) -> None:
        if self.state_dimension != 3:
            raise ValueError("The simulator benchmark predeclares state dimension three")
        if self.n_feature_folds != 2:
            raise ValueError("Formal embedding requires exactly two background folds")
        if self.parallel_max_dimension < self.state_dimension:
            raise ValueError("Parallel-analysis ceiling is below the formal rank")
        if self.parallel_repetitions < 39:
            raise ValueError("Parallel analysis requires at least 39 null repetitions")
        if not 0.5 < self.parallel_quantile < 1.0:
            raise ValueError("Parallel-analysis quantile must lie in (0.5, 1)")
        if self.random_seed < 0:
            raise ValueError("Embedding random seed may not be negative")


@dataclass
class AssayLocalStateEmbeddingResult:
    embedding: pd.DataFrame
    provenance: dict[str, Any]


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _stable_seed(base: int, *labels: str) -> int:
    digest = hashlib.sha256(
        (str(base) + "\0" + "\0".join(labels)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _validate_inputs(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    cell_ids: Mapping[str, ArrayLike],
    config: AssayLocalStateEmbeddingConfig,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray], dict[str, StringArray]]:
    modalities = set(map(str, background_matrices))
    if modalities != EXPECTED_MODALITIES or set(map(str, background_depths)) != modalities:
        raise ValueError("Formal embedding requires RNA plus the six registered marks")
    if set(map(str, cell_ids)) != modalities:
        raise ValueError("Cell-ID and background modality keys differ")
    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    identifiers: dict[str, StringArray] = {}
    seen_cells: set[str] = set()
    minimum_features = config.n_feature_folds * config.parallel_max_dimension
    for modality in sorted(modalities):
        matrix = np.asarray(background_matrices[modality], dtype=np.float64)
        depth = np.asarray(background_depths[modality], dtype=np.float64).reshape(-1)
        ids = np.asarray(cell_ids[modality], dtype=str).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError(f"Background matrix/cell IDs disagree for {modality}")
        if matrix.shape[1] < minimum_features or len(matrix) <= config.parallel_max_dimension:
            raise ValueError(f"Background matrix is too small for {modality}")
        if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
            raise ValueError(f"Background matrix is invalid for {modality}")
        if depth.shape != (len(matrix),) or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Background depth is invalid for {modality}")
        if not np.allclose(matrix.sum(axis=1), depth, rtol=1e-8, atol=1e-8):
            raise ValueError(f"Depth is not the background row sum for {modality}")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Cell IDs repeat for {modality}")
        if not seen_cells.isdisjoint(ids.tolist()):
            raise ValueError("Cell IDs overlap across independently sampled assays")
        seen_cells.update(ids.tolist())
        matrices[modality], depths[modality], identifiers[modality] = matrix, depth, ids
    return matrices, depths, identifiers


def _normalize(matrix: FloatArray, depth: FloatArray) -> FloatArray:
    scale = float(np.median(depth))
    normalized = np.log1p(matrix * (scale / depth)[:, None])
    return np.asarray(StandardScaler().fit_transform(normalized), dtype=np.float64)


def _eigenvalues(matrix: FloatArray, maximum: int) -> FloatArray:
    covariance = matrix.T @ matrix / max(len(matrix) - 1, 1)
    return np.asarray(np.linalg.eigvalsh(covariance)[::-1][:maximum], dtype=np.float64)


def _parallel_diagnostic(
    matrix: FloatArray,
    *,
    config: AssayLocalStateEmbeddingConfig,
    seed: int,
) -> dict[str, Any]:
    maximum = min(config.parallel_max_dimension, matrix.shape[1], len(matrix) - 1)
    observed = _eigenvalues(matrix, maximum)
    null = np.empty((config.parallel_repetitions, maximum), dtype=np.float64)
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(matrix)
    for repetition in range(config.parallel_repetitions):
        for feature in range(matrix.shape[1]):
            permuted[:, feature] = matrix[rng.permutation(len(matrix)), feature]
        null[repetition] = _eigenvalues(permuted, maximum)
    thresholds = np.quantile(null, config.parallel_quantile, axis=0)
    passes = list(map(bool, observed > thresholds))
    prefix_rank = 0
    for passed in passes:
        if not passed:
            break
        prefix_rank += 1
    return {
        "observed_eigenvalues": observed.tolist(),
        "null_quantiles": np.asarray(thresholds).tolist(),
        "prefix_passes": passes,
        "supported_prefix_rank": prefix_rank,
        "formal_rank_supported": prefix_rank >= config.state_dimension,
    }


def _pca_scores(matrix: FloatArray, dimension: int) -> FloatArray:
    scores = np.asarray(
        PCA(n_components=dimension, whiten=False, svd_solver="full").fit_transform(matrix),
        dtype=np.float64,
    )
    scores -= scores.mean(axis=0, keepdims=True)
    scale = scores.std(axis=0, ddof=0)
    if np.any(scale <= 1e-12):
        raise AssayLocalStateAbstention("degenerate_rank_three_fold_embedding")
    return np.asarray(scores / scale, dtype=np.float64)


def build_assay_local_state_embedding(
    background_matrices: Mapping[str, ArrayLike],
    background_depths: Mapping[str, ArrayLike],
    cell_ids: Mapping[str, ArrayLike],
    *,
    config: AssayLocalStateEmbeddingConfig | None = None,
) -> AssayLocalStateEmbeddingResult:
    """Build a common-width rank-three table without cross-assay alignment."""

    cfg = config or AssayLocalStateEmbeddingConfig()
    matrices, depths, identifiers = _validate_inputs(
        background_matrices, background_depths, cell_ids, cfg
    )
    rows: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    for modality in sorted(matrices):
        normalized = _normalize(matrices[modality], depths[modality])
        rng = np.random.default_rng(
            _stable_seed(cfg.random_seed, modality, "background_feature_folds")
        )
        folds = [
            np.asarray(values, dtype=np.int64)
            for values in np.array_split(rng.permutation(normalized.shape[1]), 2)
        ]
        fold_diagnostics = [
            _parallel_diagnostic(
                normalized[:, indices],
                config=cfg,
                seed=_stable_seed(
                    cfg.random_seed, modality, "parallel", str(fold_index)
                ),
            )
            for fold_index, indices in enumerate(folds)
        ]
        accepted_rank = min(
            int(item["supported_prefix_rank"]) for item in fold_diagnostics
        )
        if accepted_rank < cfg.state_dimension:
            raise AssayLocalStateAbstention(
                "parallel_rank_below_predeclared_three:"
                f"{modality}:supported_rank={accepted_rank}"
            )
        first, second = [
            _pca_scores(normalized[:, indices], cfg.state_dimension)
            for indices in folds
        ]
        rotation, _ = orthogonal_procrustes(second, first)
        aligned_second = np.asarray(second @ rotation, dtype=np.float64)
        consensus = np.asarray((first + aligned_second) / 2.0, dtype=np.float64)
        consensus -= consensus.mean(axis=0, keepdims=True)
        if np.any(consensus.std(axis=0, ddof=0) <= 1e-12):
            raise AssayLocalStateAbstention(
                f"degenerate_rank_three_consensus:{modality}"
            )
        difference = first - aligned_second
        uncertainty = np.sqrt(np.mean(np.square(difference), axis=1))
        frame: dict[str, Any] = {
            "modality": modality,
            "cell": identifiers[modality],
            "integration_uncertainty": uncertainty,
        }
        for dimension in range(cfg.state_dimension):
            frame[f"integration__dim{dimension}"] = consensus[:, dimension]
        rows.append(
            pd.DataFrame(frame)[
                [
                    "modality",
                    "cell",
                    *[
                        f"integration__dim{dimension}"
                        for dimension in range(cfg.state_dimension)
                    ],
                    "integration_uncertainty",
                ]
            ]
        )
        diagnostics[modality] = {
            "n_cells": len(normalized),
            "n_background_features": normalized.shape[1],
            "feature_fold_counts": [len(indices) for indices in folds],
            "feature_fold_index_sha256": [
                _array_sha256(indices) for indices in folds
            ],
            "parallel_diagnostics": fold_diagnostics,
            "minimum_fold_supported_rank": accepted_rank,
            "formal_state_dimension": cfg.state_dimension,
            "procrustes_relative_residual": float(
                np.linalg.norm(difference) / max(np.linalg.norm(first), 1e-12)
            ),
            "procrustes_rotation_determinant": float(np.linalg.det(rotation)),
            "median_cell_uncertainty": float(np.median(uncertainty)),
            "matrix_sha256": _array_sha256(matrices[modality]),
            "depth_sha256": _array_sha256(depths[modality]),
        }
    information_boundary = {
        "consumed": [
            "observed_background_matrix_per_assay",
            "observed_leave_focal_out_depth_per_assay",
            "assay_key",
        ],
        "not_consumed": [
            "observed_focal_values",
            "cross_assay_cell_identity",
            "cross_assay_feature_identity",
            "scorer",
            "truth_graph",
            "simulator_roles",
            "true_latent_state",
            "external_data",
        ],
        "truth_or_scorer_used": False,
        "cell_rows_compared_across_assays": False,
        "feature_columns_compared_across_assays": False,
        "same_assay_cells_used_for_fold_procrustes": True,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "method": "two_background_feature_fold_assay_local_pca_procrustes_consensus",
        "config": asdict(cfg),
        "formal_state_dimension": cfg.state_dimension,
        "rank_policy": (
            "predeclared_simulator_rank_three_with_parallel_analysis_abstention"
        ),
        "parallel_rank_affects_dimension_selection": False,
        "parallel_rank_below_three_causes_world_abstention": True,
        "common_coordinate_width": cfg.state_dimension,
        "zero_padding_used": False,
        "cross_assay_alignment_used": False,
        "assay_diagnostics": diagnostics,
        "accepted_assays": sorted(matrices),
        "information_boundary": information_boundary,
    }
    return AssayLocalStateEmbeddingResult(
        embedding=pd.concat(rows, ignore_index=True), provenance=provenance
    )


def _load_discovery_backgrounds(
    discovery_dir: Path,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray], dict[str, StringArray], dict[str, str]]:
    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    identifiers: dict[str, StringArray] = {}
    hashes: dict[str, str] = {}
    for path in sorted(discovery_dir.glob(f"{BACKGROUND_PREFIX}*.npz")):
        if path.is_symlink() or path.resolve().parent != discovery_dir:
            raise ValueError("Background input must be a direct discovery file")
        modality = path.name.removeprefix(BACKGROUND_PREFIX).removesuffix(".npz")
        with np.load(path, allow_pickle=False) as payload:
            required = {"matrix", "background_depth", "cell_ids"}
            if not required <= set(payload.files):
                raise ValueError(f"Background archive is incomplete for {modality}")
            matrices[modality] = np.asarray(payload["matrix"], dtype=np.float64)
            depths[modality] = np.asarray(payload["background_depth"], dtype=np.float64)
            identifiers[modality] = np.asarray(payload["cell_ids"], dtype=str)
        hashes[path.name] = file_sha256(path)
    return matrices, depths, identifiers, dict(sorted(hashes.items()))


def write_assay_local_state_embedding(
    focal_discovery_dir: Path | str,
    output_dir: Path | str,
    *,
    config: AssayLocalStateEmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Write and hash-lock one formal raw-background-only embedding bundle."""

    cfg = config or AssayLocalStateEmbeddingConfig()
    if cfg != AssayLocalStateEmbeddingConfig():
        raise ValueError("Formal writer requires the frozen embedding config")
    discovery_lexical = Path(focal_discovery_dir).absolute()
    if (
        discovery_lexical.name != "discovery"
        or discovery_lexical.is_symlink()
        or not discovery_lexical.is_dir()
    ):
        raise ValueError("Formal focal input must be a direct discovery/ directory")
    discovery = discovery_lexical.resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite embedding output {output}")
    resolved_output = output.resolve(strict=False)
    if resolved_output == discovery or discovery in resolved_output.parents:
        raise ValueError("Embedding output may not be inside raw discovery/")
    matrices, depths, identifiers, input_hashes = _load_discovery_backgrounds(
        discovery
    )
    result = build_assay_local_state_embedding(
        matrices, depths, identifiers, config=cfg
    )
    output.mkdir(parents=True, exist_ok=True)
    embedding_path = output / EMBEDDING_NAME
    provenance_path = output / PROVENANCE_NAME
    result.embedding.to_csv(embedding_path, sep="\t", index=False)
    provenance = {
        **result.provenance,
        "input_file_sha256": input_hashes,
        "algorithm_source_sha256": file_sha256(Path(__file__)),
    }
    _write_json(provenance_path, provenance)
    manifest: dict[str, Any] = {
        "schema_version": FROZEN_EVIDENCE_SCHEMA_VERSION,
        "frozen_before_scorer_read": True,
        "algorithm_source_sha256": file_sha256(Path(__file__)),
        "config": result.provenance["config"],
        "formal_state_dimension": result.provenance["formal_state_dimension"],
        "rank_policy": result.provenance["rank_policy"],
        "zero_padding_used": False,
        "cross_assay_alignment_used": False,
        "all_assays_parallel_rank_support_formal_dimension": True,
        "accepted_assays": result.provenance["accepted_assays"],
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            embedding_path.name: file_sha256(embedding_path),
            provenance_path.name: file_sha256(provenance_path),
        },
        "information_boundary": result.provenance["information_boundary"],
    }
    manifest_path = output / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    sidecar = manifest_path.with_suffix(".sha256")
    sidecar.write_text(f"{file_sha256(manifest_path)}  {manifest_path.name}\n")
    return {
        **manifest,
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_sidecar_sha256": file_sha256(sidecar),
    }


__all__ = [
    "AssayLocalStateAbstention",
    "AssayLocalStateEmbeddingConfig",
    "AssayLocalStateEmbeddingResult",
    "FROZEN_EVIDENCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_assay_local_state_embedding",
    "file_sha256",
    "write_assay_local_state_embedding",
]
