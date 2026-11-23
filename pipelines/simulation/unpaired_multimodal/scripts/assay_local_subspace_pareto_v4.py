#!/usr/bin/env python3
"""Observed-only assay-local state embeddings for the v4 score.

This standalone TRAIN-only experiment removes cross-assay alignment from the
state-response test.  For each assay it splits held-out background features in
two, fits a PCA to each fold, aligns the two score bases through orthogonal
Procrustes using only repeated cells inside that assay, and averages them.

Three choices are predeclared and frozen together: fixed rank three, fixed
rank eight as a higher-dimensional stress, and an assay-adaptive rank selected
by within-fold parallel analysis (2--8).  The
adaptive table is zero padded only for rectangular serialization; padded axes
are excluded from each assay's fitted rank.  No focal value, cross-assay row,
feature correspondence, scorer input, or external data enters either
embedding.  Both variants then use the unchanged shared_state_subspace_v4
orthogonal score, B=999 wild null, and hierarchical selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cell_state_loading_groups as loading_groups
import score_shared_state_subspace_v4 as scorer
import shared_state_subspace_v4 as state_score


FloatArray = NDArray[np.float64]
SCHEMA_VERSION = "shared_state_subspace_v4.assay_local_pca_pareto.v1"
VARIANTS = ("fixed3", "fixed8", "adaptive_parallel")


@dataclass(frozen=True)
class AssayLocalEmbeddingConfig:
    random_seed: int = 24_082_101
    n_feature_folds: int = 2
    fixed_dimension: int = 3
    higher_dimension: int = 8
    adaptive_min_dimension: int = 2
    adaptive_max_dimension: int = 8
    parallel_repetitions: int = 99
    parallel_quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.n_feature_folds != 2:
            raise ValueError("This experiment requires exactly two background folds")
        if self.fixed_dimension < 2:
            raise ValueError("Fixed state rank must be at least two")
        if self.higher_dimension < self.fixed_dimension:
            raise ValueError("Higher-dimensional stress must not be below fixed rank")
        if not 2 <= self.adaptive_min_dimension <= self.adaptive_max_dimension:
            raise ValueError("Adaptive rank interval is invalid")
        if self.parallel_repetitions < 39:
            raise ValueError("Parallel analysis needs at least 39 null repetitions")
        if not 0.5 < self.parallel_quantile < 1.0:
            raise ValueError("Parallel-analysis quantile must lie in (0.5, 1)")


@dataclass
class _BackgroundAssay:
    modality: str
    matrix: FloatArray
    cell_ids: NDArray[np.str_]
    depth: FloatArray
    source_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _stable_seed(base: int, *labels: str) -> int:
    digest = hashlib.sha256(
        (str(base) + "\0" + "\0".join(labels)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _world_seed(world: Path) -> int:
    try:
        return int(world.name.removeprefix("world_seed"))
    except ValueError as exc:
        raise ValueError(f"World lacks a numeric world_seed suffix: {world}") from exc


def _validate_world(world_path: str | Path) -> tuple[Path, Path, Path]:
    lexical = Path(world_path).absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"World must be a direct directory: {lexical}")
    world = lexical.resolve(strict=True)
    discovery_root, scorer_root = world / "discovery", world / "scorer"
    if (
        discovery_root.is_symlink()
        or scorer_root.is_symlink()
        or not discovery_root.is_dir()
        or not scorer_root.is_dir()
    ):
        raise ValueError("World must contain direct discovery/ and scorer/ domains")
    return world, discovery_root, scorer_root


def _load_backgrounds(discovery_root: Path) -> dict[str, _BackgroundAssay]:
    output: dict[str, _BackgroundAssay] = {}
    paths = sorted(discovery_root.glob("background_matrix__*.npz"))
    if len(paths) != 7:
        raise ValueError("Assay-local embedding requires RNA and six chromatin matrices")
    seen_cells: set[str] = set()
    for path in paths:
        if path.is_symlink() or path.resolve().parent != discovery_root.resolve():
            raise ValueError("Background matrices must be direct discovery files")
        modality = path.name.removeprefix("background_matrix__").removesuffix(".npz")
        with np.load(path, allow_pickle=False) as payload:
            if not {"matrix", "cell_ids", "background_depth"} <= set(payload.files):
                raise ValueError(f"Background archive is incomplete for {modality}")
            matrix = np.asarray(payload["matrix"], dtype=np.float64)
            cell_ids = np.asarray(payload["cell_ids"], dtype=str)
            depth = np.asarray(payload["background_depth"], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(cell_ids):
            raise ValueError(f"Background matrix shape is invalid for {modality}")
        if matrix.shape[1] < 2 * 8 or not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(f"Background matrix cannot support rank-8 folds: {modality}")
        if depth.shape != (len(matrix),) or np.any(depth <= 0) or not np.isfinite(depth).all():
            raise ValueError(f"Background depth is invalid for {modality}")
        if not np.allclose(matrix.sum(axis=1), depth, rtol=1e-8, atol=1e-8):
            raise ValueError(f"Background depth is not leave-focal-out row sum: {modality}")
        if len(np.unique(cell_ids)) != len(cell_ids):
            raise ValueError(f"Background cell IDs repeat for {modality}")
        if not seen_cells.isdisjoint(cell_ids.tolist()):
            raise ValueError("Cell IDs overlap across independently sampled assays")
        seen_cells.update(cell_ids.tolist())
        output[modality] = _BackgroundAssay(
            modality=modality,
            matrix=matrix,
            cell_ids=cell_ids,
            depth=depth,
            source_path=path,
        )
    return output


def _normalize(matrix: FloatArray, depth: FloatArray) -> FloatArray:
    scale = float(np.median(depth))
    normalized = np.log1p(matrix * (scale / depth)[:, None])
    return np.asarray(StandardScaler().fit_transform(normalized), dtype=np.float64)


def _ordered_eigenvalues(matrix: FloatArray, maximum: int) -> FloatArray:
    covariance = matrix.T @ matrix / max(len(matrix) - 1, 1)
    values = np.linalg.eigvalsh(covariance)[::-1]
    return np.asarray(values[:maximum], dtype=np.float64)


def _parallel_rank(
    matrix: FloatArray,
    config: AssayLocalEmbeddingConfig,
    *,
    seed: int,
) -> tuple[int, FloatArray, FloatArray, list[bool]]:
    maximum = min(config.adaptive_max_dimension, matrix.shape[1], len(matrix) - 1)
    observed = _ordered_eigenvalues(matrix, maximum)
    rng = np.random.default_rng(seed)
    null = np.empty((config.parallel_repetitions, maximum), dtype=np.float64)
    permuted = np.empty_like(matrix)
    for repetition in range(config.parallel_repetitions):
        for feature in range(matrix.shape[1]):
            permuted[:, feature] = matrix[rng.permutation(len(matrix)), feature]
        null[repetition] = _ordered_eigenvalues(permuted, maximum)
    thresholds = np.quantile(null, config.parallel_quantile, axis=0)
    passes = list(map(bool, observed > thresholds))
    prefix = 0
    for passed in passes:
        if not passed:
            break
        prefix += 1
    rank = int(
        np.clip(
            prefix,
            config.adaptive_min_dimension,
            config.adaptive_max_dimension,
        )
    )
    return rank, observed, np.asarray(thresholds, dtype=np.float64), passes


def _standardized_pca_scores(matrix: FloatArray, rank: int) -> FloatArray:
    scores = np.asarray(
        PCA(n_components=rank, whiten=False, svd_solver="full").fit_transform(matrix),
        dtype=np.float64,
    )
    scores -= scores.mean(axis=0, keepdims=True)
    scales = scores.std(axis=0, ddof=0)
    if np.any(scales <= 1e-12):
        raise ValueError("A local background component is degenerate")
    return np.asarray(scores / scales, dtype=np.float64)


def _fit_assay_embedding(
    assay: _BackgroundAssay,
    variant: str,
    config: AssayLocalEmbeddingConfig,
) -> tuple[FloatArray, FloatArray, int, dict[str, Any]]:
    normalized = _normalize(assay.matrix, assay.depth)
    split_rng = np.random.default_rng(
        _stable_seed(config.random_seed, assay.modality, "feature_split")
    )
    feature_folds = [
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(
            split_rng.permutation(assay.matrix.shape[1]), config.n_feature_folds
        )
    ]
    ranks: list[int] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for fold_index, indices in enumerate(feature_folds):
        fold_matrix = normalized[:, indices]
        if variant in {"fixed3", "fixed8"}:
            rank = (
                config.fixed_dimension
                if variant == "fixed3"
                else config.higher_dimension
            )
            diagnostics = {
                "parallel_observed_eigenvalues": [],
                "parallel_null_quantiles": [],
                "parallel_prefix_passes": [],
                "parallel_rank_before_crossfold_min": None,
            }
        elif variant == "adaptive_parallel":
            rank, observed, thresholds, passes = _parallel_rank(
                fold_matrix,
                config,
                seed=_stable_seed(
                    config.random_seed,
                    assay.modality,
                    "parallel_analysis",
                    str(fold_index),
                ),
            )
            diagnostics = {
                "parallel_observed_eigenvalues": observed.tolist(),
                "parallel_null_quantiles": thresholds.tolist(),
                "parallel_prefix_passes": passes,
                "parallel_rank_before_crossfold_min": rank,
            }
        else:
            raise ValueError(f"Unknown assay-local embedding variant {variant}")
        ranks.append(rank)
        fold_diagnostics.append(
            {
                "fold": fold_index,
                "n_background_features": len(indices),
                "feature_index_sha256": _array_sha256(indices),
                **diagnostics,
            }
        )
    rank = min(ranks)
    fold_scores = [
        _standardized_pca_scores(normalized[:, indices], rank)
        for indices in feature_folds
    ]
    rotation, _ = orthogonal_procrustes(fold_scores[1], fold_scores[0])
    aligned_second = np.asarray(fold_scores[1] @ rotation, dtype=np.float64)
    consensus = np.asarray((fold_scores[0] + aligned_second) / 2.0, dtype=np.float64)
    consensus -= consensus.mean(axis=0, keepdims=True)
    if np.any(consensus.std(axis=0, ddof=0) <= 1e-12):
        raise ValueError(f"Fold consensus is degenerate for {assay.modality}")
    discrepancy = fold_scores[0] - aligned_second
    uncertainty = np.sqrt(np.mean(np.square(discrepancy), axis=1))
    relative_residual = float(
        np.linalg.norm(discrepancy) / max(np.linalg.norm(fold_scores[0]), 1e-12)
    )
    diagnostics = {
        "modality": assay.modality,
        "variant": variant,
        "n_cells": len(assay.matrix),
        "n_background_features": assay.matrix.shape[1],
        "selected_local_rank": rank,
        "crossfold_rank_rule": (
            "minimum_of_two_within_fold_parallel_ranks"
            if variant == "adaptive_parallel"
            else "predeclared_fixed_rank"
        ),
        "fold_diagnostics": fold_diagnostics,
        "procrustes_relative_residual": relative_residual,
        "procrustes_rotation_determinant": float(np.linalg.det(rotation)),
        "median_cell_uncertainty": float(np.median(uncertainty)),
        "matrix_sha256": _array_sha256(assay.matrix),
        "depth_sha256": _array_sha256(assay.depth),
    }
    return consensus, np.asarray(uncertainty), rank, diagnostics


def build_assay_local_embedding(
    backgrounds: Mapping[str, _BackgroundAssay],
    variant: str,
    config: AssayLocalEmbeddingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fitted: dict[str, tuple[FloatArray, FloatArray, int, dict[str, Any]]] = {
        modality: _fit_assay_embedding(assay, variant, config)
        for modality, assay in sorted(backgrounds.items())
    }
    width = (
        (
            config.fixed_dimension
            if variant == "fixed3"
            else config.higher_dimension
        )
        if variant in {"fixed3", "fixed8"}
        else max(values[2] for values in fitted.values())
    )
    frames: list[pd.DataFrame] = []
    for modality, assay in sorted(backgrounds.items()):
        scores, uncertainty, rank, _ = fitted[modality]
        padded = np.zeros((len(scores), width), dtype=np.float64)
        padded[:, :rank] = scores
        frame: dict[str, Any] = {"modality": modality, "cell": assay.cell_ids}
        frame.update(
            {f"integration__dim{index}": padded[:, index] for index in range(width)}
        )
        frame["integration_uncertainty"] = uncertainty
        frames.append(pd.DataFrame(frame))
    embedding = pd.concat(frames, ignore_index=True)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "config": asdict(config),
        "serialization_coordinate_width": width,
        "assay_local_ranks": {
            modality: values[2] for modality, values in fitted.items()
        },
        "assay_diagnostics": {
            modality: values[3] for modality, values in fitted.items()
        },
        "information_boundary": {
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
        },
    }
    return embedding, provenance


def _write_candidate(
    discovery_root: Path,
    embedding_path: Path,
    out_dir: Path,
    embedding_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    config = state_score.SharedStateSubspaceConfig()
    raw_inputs = loading_groups.load_loading_inputs(discovery_root, embedding_path)
    inputs, scales = state_score._rotation_invariant_local_inputs(raw_inputs)
    observed = state_score._fit_observed_evidence(inputs, config)
    reference, null = state_score.estimate_conditional_wild_null(inputs, config=config)
    result = state_score.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    membership_path = out_dir / "subspace_feature_membership.tsv"
    hyperedge_path = out_dir / "typed_subspace_hyperedges.tsv"
    provenance_path = out_dir / "candidate_provenance.json"
    confirmation_path = out_dir / "feature_global_maxT_confirmation.tsv"
    result.feature_membership.to_csv(membership_path, sep="\t", index=False)
    result.typed_subspace_hyperedges.to_csv(hyperedge_path, sep="\t", index=False)
    result.feature_membership[
        ["feature", "modality", "selection_status", "feature_global_maxT_p"]
    ].assign(
        high_confidence_confirmed=lambda frame: (
            frame["selection_status"].eq("selected")
            & frame["feature_global_maxT_p"].le(
                config.maximum_hyperedge_error_probability
            )
        )
    ).to_csv(confirmation_path, sep="\t", index=False)
    provenance = {
        "schema_version": f"{SCHEMA_VERSION}.candidate",
        "truth_or_scorer_used": False,
        "discovery_eligible": True,
        "state_score_source_sha256": _sha256(Path(state_score.__file__)),
        "state_score_config": asdict(config),
        "state_score_config_sha256": _canonical_sha256(asdict(config)),
        "assay_coordinate_scales": scales,
        "embedding": embedding_provenance,
        "embedded_state_score_provenance": result.provenance,
        "feature_global_maxT_confirmation_is_diagnostic_only": True,
    }
    _write_json(provenance_path, provenance)
    artifacts = {
        path.name: _sha256(path)
        for path in (membership_path, hyperedge_path, provenance_path, confirmation_path)
    }
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.candidate_freeze",
        "artifacts": artifacts,
        "truth_or_scorer_used": False,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "candidate_manifest.json", manifest)
    return manifest


def _phase_a_world(world_path: str, output_root: str) -> dict[str, Any]:
    world, discovery_root, _ = _validate_world(world_path)
    seed = _world_seed(world)
    world_root = Path(output_root) / "phase_a" / f"world_seed{seed}"
    if world_root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-A world {world_root}")
    world_root.mkdir(parents=True)
    backgrounds = _load_backgrounds(discovery_root)
    config = AssayLocalEmbeddingConfig()
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        embedding, embedding_provenance = build_assay_local_embedding(
            backgrounds, variant, config
        )
        embedding_root = world_root / "embeddings" / variant
        embedding_root.mkdir(parents=True)
        embedding_path = embedding_root / "analysis_embeddings.tsv"
        embedding_provenance_path = embedding_root / "embedding_provenance.json"
        embedding.to_csv(embedding_path, sep="\t", index=False)
        embedding_provenance = {
            **embedding_provenance,
            "input_file_sha256": {
                assay.source_path.name: _sha256(assay.source_path)
                for assay in backgrounds.values()
            },
            "output_file_sha256": {
                embedding_path.name: _sha256(embedding_path)
            },
            "algorithm_source_sha256": _sha256(Path(__file__)),
        }
        _write_json(embedding_provenance_path, embedding_provenance)
        candidate_manifest = _write_candidate(
            discovery_root,
            embedding_path,
            world_root / "candidates" / variant,
            embedding_provenance,
        )
        variants[variant] = {
            "candidate_manifest_payload_sha256": candidate_manifest[
                "manifest_payload_sha256"
            ],
            "assay_local_ranks": embedding_provenance["assay_local_ranks"],
            "serialization_coordinate_width": embedding_provenance[
                "serialization_coordinate_width"
            ],
        }
    artifacts = {
        str(path.relative_to(world_root)): _sha256(path)
        for path in sorted(world_root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.phase_a",
        "world_seed": seed,
        "world_path": str(world),
        "all_variants_frozen_before_scorer_read": True,
        "embedding_config": asdict(config),
        "embedding_config_sha256": _canonical_sha256(asdict(config)),
        "state_score_config": asdict(state_score.SharedStateSubspaceConfig()),
        "state_score_source_sha256": _sha256(Path(state_score.__file__)),
        "runner_source_sha256": _sha256(Path(__file__)),
        "variants": variants,
        "artifacts": artifacts,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(world_root / "phase_a_manifest.json", manifest)
    return manifest


def _confirmation_metrics(feature_scores: pd.DataFrame) -> dict[str, Any]:
    frame = feature_scores.copy()
    frame["confirmed"] = (
        frame["selected"].eq(1) & frame["feature_global_maxT_p"].le(0.10)
    )
    expected = frame["expected_subspace_member"].astype(bool)
    selected = frame["confirmed"]
    true_selected = selected & expected
    chrom = frame["modality"].ne("rna")
    role_recalls = {
        role: float(frame.loc[frame["role"].eq(role), "confirmed"].mean())
        for role in ("P", "Q", "S", "C")
    }
    assay_recalls = {
        str(modality): float(group.loc[group["expected_subspace_member"], "confirmed"].mean())
        for modality, group in frame.groupby("modality", sort=True)
    }
    n_selected = int(selected.sum())
    n_true = int(true_selected.sum())
    return {
        "n_selected": n_selected,
        "n_true_selected": n_true,
        "n_false_selected": n_selected - n_true,
        "precision": float(n_true / n_selected) if n_selected else 0.0,
        "recall": float(n_true / expected.sum()),
        "chromatin_recall": float((true_selected & chrom).sum() / (expected & chrom).sum()),
        "sentinel_rate": float((selected & ~expected).sum() / (~expected).sum()),
        "role_recalls": role_recalls,
        "assay_recalls": assay_recalls,
        "minimum_chromatin_assay_recall": min(
            value for modality, value in assay_recalls.items() if modality != "rna"
        ),
    }


def _aggregate(per_world: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row[variant]["primary_metrics"] for row in per_world]
        confirms = [row[variant]["feature_maxT_confirmation"] for row in per_world]
        output[variant] = {
            "membership_world_successes": int(
                sum(int(row["membership_class_success"]) for row in rows)
            ),
            "hyperedge_world_successes": int(
                sum(int(row["hyperedge_class_success"]) for row in rows)
            ),
            "total_true_selected": int(
                sum(int(row["n_selected_true_subspace_features"]) for row in rows)
            ),
            "total_false_selected": int(
                sum(int(row["n_selected_ineligible_features"]) for row in rows)
            ),
            "mean_recall": float(np.mean([row["subspace_member_recall"] for row in rows])),
            "mean_precision": float(
                np.mean([row["subspace_member_precision"] for row in rows])
            ),
            "maxT_confirmation_total_true": int(
                sum(row["n_true_selected"] for row in confirms)
            ),
            "maxT_confirmation_total_false": int(
                sum(row["n_false_selected"] for row in confirms)
            ),
        }
    return output


def run_experiment(
    worlds: Sequence[Path], output_root: Path, *, workers: int = 1
) -> dict[str, Any]:
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if len({_world_seed(Path(world)) for world in worlds}) != len(worlds):
        raise ValueError("World seeds must be unique")
    phase_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_phase_a_world, str(world.absolute()), str(output_root)): world
            for world in worlds
        }
        for future in as_completed(futures):
            phase_records.append(future.result())
    phase_records.sort(key=lambda row: int(row["world_seed"]))
    freeze = {
        "schema_version": f"{SCHEMA_VERSION}.global_pretruth_freeze",
        "all_worlds_and_variants_complete": True,
        "truth_or_scorer_read_in_phase_a": False,
        "phase_a": phase_records,
        "runner_source_sha256": _sha256(Path(__file__)),
    }
    freeze["freeze_payload_sha256"] = _canonical_sha256(freeze)
    freeze_path = output_root / "global_pretruth_freeze.json"
    _write_json(freeze_path, freeze)

    loaded: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for record in phase_records:
        seed = int(record["world_seed"])
        world_root = output_root / "phase_a" / f"world_seed{seed}"
        for relative, digest in record["artifacts"].items():
            if _sha256(world_root / relative) != digest:
                raise ValueError(f"Frozen bytes changed for seed {seed}: {relative}")
        for variant in VARIANTS:
            candidate = world_root / "candidates" / variant
            loaded[(seed, variant)] = (
                pd.read_csv(
                    candidate / "subspace_feature_membership.tsv",
                    sep="\t",
                    keep_default_na=False,
                    float_precision="round_trip",
                ),
                pd.read_csv(
                    candidate / "typed_subspace_hyperedges.tsv",
                    sep="\t",
                    keep_default_na=False,
                    float_precision="round_trip",
                ),
                pd.read_csv(
                    candidate / "feature_global_maxT_confirmation.tsv",
                    sep="\t",
                    keep_default_na=False,
                    float_precision="round_trip",
                ),
            )

    per_world: list[dict[str, Any]] = []
    world_map = {_world_seed(Path(world)): Path(world) for world in worlds}
    for seed in sorted(world_map):
        _, _, scorer_root = _validate_world(world_map[seed])
        parameters = json.loads((scorer_root / "world_parameters.json").read_text())
        roles = pd.read_csv(
            scorer_root / "role_assignments.tsv", sep="\t", keep_default_na=False
        )
        world_summary: dict[str, Any] = {"world_seed": seed}
        for variant in VARIANTS:
            membership, hyperedge, confirmation = loaded[(seed, variant)]
            score = scorer.score_shared_state_subspace(
                world_parameters=parameters,
                role_assignments=roles,
                feature_membership=membership,
                typed_subspace_hyperedges=hyperedge,
            )
            confirmation_expected = (
                membership["selection_status"].eq("selected")
                & membership["feature_global_maxT_p"].le(0.10)
            )
            if not np.array_equal(
                confirmation["high_confidence_confirmed"].to_numpy(dtype=bool),
                confirmation_expected.to_numpy(dtype=bool),
            ):
                raise ValueError("Frozen feature-maxT confirmation reconstruction failed")
            diagnostic = _confirmation_metrics(score.feature_scores)
            score_root = output_root / "scores" / f"world_seed{seed}" / variant
            score_root.mkdir(parents=True)
            _write_json(score_root / "metrics.json", score.metrics)
            _write_json(score_root / "feature_maxT_confirmation_metrics.json", diagnostic)
            score.feature_scores.to_csv(
                score_root / "feature_scores.tsv", sep="\t", index=False
            )
            score.assay_scores.to_csv(
                score_root / "assay_scores.tsv", sep="\t", index=False
            )
            world_summary[variant] = {
                "primary_metrics": score.metrics,
                "feature_maxT_confirmation": diagnostic,
            }
        per_world.append(world_summary)
    summary = {
        "schema_version": f"{SCHEMA_VERSION}.summary",
        "global_pretruth_freeze_sha256": _sha256(freeze_path),
        "variants": list(VARIANTS),
        "per_world": per_world,
        "aggregate": _aggregate(per_world),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_experiment(args.world, args.out_root, workers=args.workers)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
