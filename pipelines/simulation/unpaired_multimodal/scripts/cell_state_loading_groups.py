#!/usr/bin/env python3
"""Cell-level state-response loadings for independently sampled assays.

The layer consumes only observed focal tables from a simulator ``discovery/``
directory and a separately recomputed ``analysis_embeddings.tsv``.  It never
reads a scorer, truth, supplied simulator embedding, metacell membership, or
cross-assay row correspondence.

For each assay, common background coordinates are fitted to each focal feature
at the original cell resolution.  Leave-focal log-depth is projected on state
inside every training fold; only the state-orthogonal residual is used as a
technical nuisance.  Binary chromatin features use cross-fitted logistic
regression and continuous RNA features use cross-fitted ridge regression.
Loadings therefore estimate an assay-local response to a common coordinate
system conditional on technical depth, without pairing cells across assays.

Sparse pair and group evidence is constructed from loading-vector cosine and
unit-vector distance.  It describes a shared state-response axis, not a direct
causal edge or cell correspondence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_FOCAL_PREFIX = "observed_focal__"
_EMBEDDING_FILENAME = "analysis_embeddings.tsv"
_FORBIDDEN_PATH_COMPONENTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
    "counterfactual",
}
_FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "true_",
    "role",
    "parent",
    "child",
    "latent",
    "oracle",
    "scorer",
    "correspondence",
    "pairing",
    "population_mode",
)
_COORDINATE_PATTERN = re.compile(r"^integration__dim([0-9]+)$")


@dataclass(frozen=True)
class LoadingGroupConfig:
    """Frozen estimation and sparse grouping settings."""

    folds: int = 5
    ridge_alpha: float = 2.0
    depth_ridge_alpha: float = 2.0
    logistic_c: float = 1.0
    logistic_max_iter: int = 2_000
    bootstrap_repetitions: int = 50
    seed: int = 0
    min_loading_norm: float = 0.10
    min_state_fit: float = 0.01
    min_direction_stability: float = 0.50
    min_abs_loading_cosine: float = 0.80
    max_unit_loading_distance: float = 0.65
    max_neighbors_per_feature: int = 2

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if min(self.ridge_alpha, self.depth_ridge_alpha, self.logistic_c) <= 0:
            raise ValueError("regularization parameters must be positive")
        if self.logistic_max_iter < 100:
            raise ValueError("logistic_max_iter must be at least 100")
        if self.bootstrap_repetitions < 0:
            raise ValueError("bootstrap_repetitions may not be negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.min_loading_norm < 0 or self.min_state_fit < -1:
            raise ValueError("loading norm and state-fit thresholds are invalid")
        if not 0 <= self.min_direction_stability <= 1:
            raise ValueError("min_direction_stability must lie in [0, 1]")
        if not 0 <= self.min_abs_loading_cosine <= 1:
            raise ValueError("min_abs_loading_cosine must lie in [0, 1]")
        if not 0 <= self.max_unit_loading_distance <= 2:
            raise ValueError("max_unit_loading_distance must lie in [0, 2]")
        if self.max_neighbors_per_feature < 1:
            raise ValueError("max_neighbors_per_feature must be positive")


@dataclass
class AssayData:
    modality: str
    cell_ids: NDArray[np.str_]
    coordinates: FloatArray
    coordinate_uncertainty: FloatArray
    log_depth: FloatArray
    focal: pd.DataFrame


@dataclass
class LoadingInputs:
    focal_root: Path
    embedding_path: Path
    coordinate_columns: tuple[str, ...]
    coordinate_center: FloatArray
    coordinate_scale: FloatArray
    assays: dict[str, AssayData]
    input_hashes: dict[str, str]


@dataclass
class LoadingGroupResult:
    feature_models: pd.DataFrame
    loadings: pd.DataFrame
    pair_evidence: pd.DataFrame
    sparse_pairs: pd.DataFrame
    groups: pd.DataFrame
    group_membership: pd.DataFrame
    provenance: dict[str, Any]


@dataclass
class _FeatureFit:
    feature_row: dict[str, Any]
    loading_rows: list[dict[str, Any]]
    conditional_loading: FloatArray


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_path_components(path: Path) -> list[str]:
    return sorted(
        {
            component.casefold()
            for component in path.parts
            if component.casefold() in _FORBIDDEN_PATH_COMPONENTS
        }
    )


def _validated_focal_root(path: Path | str) -> Path:
    lexical = Path(path).absolute()
    if lexical.name.casefold() != "discovery":
        raise ValueError("Focal input must be a simulator/world discovery/ directory")
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise ValueError("Focal discovery/ must be a direct existing directory")
    resolved = lexical.resolve(strict=True)
    if resolved.name.casefold() != "discovery":
        raise ValueError("Focal discovery/ may not be a symlink to another data domain")
    forbidden = _forbidden_path_components(resolved)
    if forbidden:
        raise ValueError(f"Refusing scorer/truth focal path: {forbidden}")
    return resolved


def _validated_embedding_path(path: Path | str) -> Path:
    lexical = Path(path).absolute()
    if lexical.name != _EMBEDDING_FILENAME:
        raise ValueError(f"Safe embedding input must be named {_EMBEDDING_FILENAME}")
    if lexical.parent.name.casefold() == "discovery":
        raise ValueError(
            "Refusing simulator-supplied embedding; use a separately recomputed safe embedding"
        )
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_file():
        raise ValueError("Safe embedding must be a direct regular file")
    resolved = lexical.resolve(strict=True)
    forbidden = _forbidden_path_components(resolved)
    if forbidden:
        raise ValueError(f"Refusing scorer/truth embedding path: {forbidden}")
    return resolved


def _forbidden_columns(columns: list[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(token in column.casefold() for token in _FORBIDDEN_COLUMN_TOKENS)
    )


def _biological_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column).startswith("expr__") or str(column).startswith("chr")
    ]


def _load_embeddings(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    try:
        frame = pd.read_csv(path, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Could not read safe analysis embeddings") from exc
    required = {"modality", "cell"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Safe embedding is missing required columns {missing}")
    forbidden = _forbidden_columns([str(column) for column in frame.columns])
    if forbidden:
        raise ValueError(f"Safe embedding contains forbidden columns {forbidden}")
    numbered: list[tuple[int, str]] = []
    for column in frame.columns:
        match = _COORDINATE_PATTERN.fullmatch(str(column))
        if match:
            numbered.append((int(match.group(1)), str(column)))
    numbered.sort()
    coordinate_columns = tuple(column for _, column in numbered)
    if len(coordinate_columns) < 2 or [index for index, _ in numbered] != list(
        range(len(numbered))
    ):
        raise ValueError("Safe embedding needs sequential integration__dim coordinates")
    allowed = required | set(coordinate_columns) | {"integration_uncertainty"}
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise ValueError(f"Safe embedding contains non-whitelisted columns {unexpected}")
    frame = frame.copy()
    if frame[["modality", "cell"]].isna().any().any():
        raise ValueError("Safe embedding modality and cell IDs may not be missing")
    frame["modality"] = frame["modality"].astype(str)
    frame["cell"] = frame["cell"].astype(str)
    if frame.duplicated(["modality", "cell"]).any():
        raise ValueError("Safe embedding keys must be unique")
    for column in coordinate_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    coordinates = frame[list(coordinate_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Safe integration coordinates must be finite")
    if "integration_uncertainty" in frame:
        frame["integration_uncertainty"] = pd.to_numeric(
            frame["integration_uncertainty"], errors="raise"
        )
    else:
        frame["integration_uncertainty"] = 0.0
    uncertainty = frame["integration_uncertainty"].to_numpy(dtype=np.float64)
    if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
        raise ValueError("pipelines/empirical/integration uncertainty must be finite and non-negative")
    return frame, coordinate_columns


def _load_focal_table(path: Path, modality: str) -> tuple[pd.DataFrame, list[str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Observed focal artifact must be a direct regular file: {path.name}")
    try:
        frame = pd.read_csv(path, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read observed focal table {path.name}") from exc
    required = {
        "cell",
        "modality",
        "observed_background_library_size",
        "observed_library_size",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns {missing}")
    forbidden = _forbidden_columns([str(column) for column in frame.columns])
    if forbidden:
        raise ValueError(f"{path.name} contains forbidden columns {forbidden}")
    biological = _biological_columns(frame)
    allowed = required | set(biological)
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise ValueError(f"{path.name} contains non-whitelisted columns {unexpected}")
    if not biological:
        raise ValueError(f"{path.name} contains no focal biological variables")
    frame = frame.copy()
    if frame[["cell", "modality"]].isna().any().any():
        raise ValueError(f"{path.name} has missing keys")
    frame["cell"] = frame["cell"].astype(str)
    frame["modality"] = frame["modality"].astype(str)
    if frame["cell"].duplicated().any():
        raise ValueError(f"{path.name} cell IDs must be unique")
    if len(frame) < 8:
        raise ValueError(f"{path.name} needs at least eight observed cells")
    if set(frame["modality"]) != {modality}:
        raise ValueError(f"{path.name} modality column does not equal {modality!r}")
    numeric = [
        "observed_background_library_size",
        "observed_library_size",
        *biological,
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[numeric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} numeric observations must be finite")
    depth = frame["observed_background_library_size"].to_numpy(dtype=np.float64)
    total = frame["observed_library_size"].to_numpy(dtype=np.float64)
    if np.any(depth <= 0) or np.any(total < depth):
        raise ValueError(f"{path.name} library sizes are invalid")
    if modality == "rna":
        if not all(feature.startswith("expr__") for feature in biological):
            raise ValueError("RNA variables must use expr__ names")
    elif modality.startswith("chrom_"):
        mark = modality.removeprefix("chrom_")
        if not all(
            feature.startswith("chr") and feature.endswith(f"__{mark}")
            for feature in biological
        ):
            raise ValueError(f"{modality} variables must carry their assay suffix")
        for feature in biological:
            observed = set(frame[feature].unique().tolist())
            if not observed <= {0, 1}:
                raise ValueError(f"Chromatin feature {feature} must be binary")
            if len(observed) < 2:
                raise ValueError(f"Chromatin feature {feature} must contain both outcomes")
    else:
        raise ValueError(f"Unsupported modality {modality!r}; expected rna or chrom_<mark>")
    return frame, biological


def load_loading_inputs(
    focal_discovery_dir: Path | str,
    safe_embedding_path: Path | str,
) -> LoadingInputs:
    """Load only focal observations and separately recomputed safe coordinates."""

    focal_root = _validated_focal_root(focal_discovery_dir)
    embedding_path = _validated_embedding_path(safe_embedding_path)
    embeddings, coordinate_columns = _load_embeddings(embedding_path)
    focal_paths: dict[str, Path] = {}
    for path in sorted(focal_root.glob(f"{_FOCAL_PREFIX}*.tsv")):
        if path.resolve().parent != focal_root:
            raise ValueError("Focal artifacts may not escape discovery/")
        modality = path.name.removeprefix(_FOCAL_PREFIX).removesuffix(".tsv")
        if not modality or modality in focal_paths:
            raise ValueError(f"Invalid or duplicate focal artifact {path.name}")
        focal_paths[modality] = path
    if "rna" not in focal_paths or len(focal_paths) < 2:
        raise ValueError("At least RNA and one chromatin focal table are required")
    embedding_modalities = set(embeddings["modality"])
    if embedding_modalities != set(focal_paths):
        raise ValueError(
            "Safe embedding and observed focal tables must have identical modalities"
        )

    pooled = embeddings[list(coordinate_columns)].to_numpy(dtype=np.float64)
    coordinate_center = pooled.mean(axis=0)
    coordinate_scale = pooled.std(axis=0, ddof=0)
    if np.any(coordinate_scale <= 1e-12):
        raise ValueError("Every common integration coordinate must vary")
    assays: dict[str, AssayData] = {}
    hashes = {embedding_path.name: _sha256(embedding_path)}
    all_cells: set[str] = set()
    all_features: set[str] = set()
    for modality, path in sorted(focal_paths.items()):
        focal, biological = _load_focal_table(path, modality)
        cells = set(focal["cell"])
        if not all_cells.isdisjoint(cells):
            raise ValueError("Cell IDs must be disjoint across independently sampled assays")
        all_cells.update(cells)
        overlap = all_features.intersection(biological)
        if overlap:
            raise ValueError(f"Feature names repeat across assays: {sorted(overlap)}")
        all_features.update(biological)
        assay_embedding = embeddings.loc[embeddings["modality"] == modality].copy()
        if set(assay_embedding["cell"]) != cells:
            raise ValueError(f"Safe embedding cells disagree for modality {modality}")
        merged = focal.merge(
            assay_embedding,
            on=["modality", "cell"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        coordinates = (
            merged[list(coordinate_columns)].to_numpy(dtype=np.float64)
            - coordinate_center[None, :]
        ) / coordinate_scale[None, :]
        uncertainty = merged["integration_uncertainty"].to_numpy(dtype=np.float64)
        log_depth = np.log1p(
            merged["observed_background_library_size"].to_numpy(dtype=np.float64)
        )
        log_depth = (log_depth - float(log_depth.mean())) / max(
            float(log_depth.std(ddof=0)), 1e-12
        )
        assays[modality] = AssayData(
            modality=modality,
            cell_ids=merged["cell"].to_numpy(dtype=str),
            coordinates=np.asarray(coordinates, dtype=np.float64),
            coordinate_uncertainty=np.asarray(uncertainty, dtype=np.float64),
            log_depth=np.asarray(log_depth, dtype=np.float64),
            focal=merged[biological].copy(),
        )
        hashes[path.name] = _sha256(path)
    return LoadingInputs(
        focal_root=focal_root,
        embedding_path=embedding_path,
        coordinate_columns=coordinate_columns,
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
        assays=assays,
        input_hashes=hashes,
    )


def _folds_for_response(
    response: FloatArray,
    *,
    family: str,
    requested: int,
    seed: int,
) -> list[tuple[IntArray, IntArray]]:
    if family == "logistic":
        counts = np.bincount(response.astype(np.int64), minlength=2)
        n_splits = min(requested, int(counts.min()))
        if n_splits < 2:
            raise ValueError("Binary features need at least two events and two non-events")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
            for train, test in splitter.split(np.zeros(len(response)), response)
        ]
    n_splits = min(requested, max(2, len(response) // 8))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(np.zeros(len(response)))
    ]


def _orthogonal_depth(
    state_train: FloatArray,
    depth_train: FloatArray,
    state_test: FloatArray,
    depth_test: FloatArray,
    *,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    model = Ridge(alpha=alpha, fit_intercept=True).fit(state_train, depth_train)
    train_residual = depth_train - model.predict(state_train)
    test_residual = depth_test - model.predict(state_test)
    center = float(train_residual.mean())
    scale = max(float(train_residual.std(ddof=0)), 1e-12)
    return (
        np.asarray((train_residual - center) / scale, dtype=np.float64),
        np.asarray((test_residual - center) / scale, dtype=np.float64),
    )


def _fit_predict(
    family: str,
    design_train: FloatArray,
    response_train: FloatArray,
    design_test: FloatArray,
    config: LoadingGroupConfig,
) -> tuple[FloatArray, FloatArray]:
    if family == "logistic":
        model = LogisticRegression(
            C=config.logistic_c,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=config.logistic_max_iter,
        ).fit(design_train, response_train.astype(np.int64))
        prediction = model.predict_proba(design_test)[:, 1]
        coefficient = np.asarray(model.coef_[0], dtype=np.float64)
    else:
        model = Ridge(alpha=config.ridge_alpha, fit_intercept=True).fit(
            design_train, response_train
        )
        prediction = model.predict(design_test)
        coefficient = np.asarray(model.coef_, dtype=np.float64)
    return np.asarray(prediction, dtype=np.float64), coefficient


def _loss(family: str, response: FloatArray, prediction: FloatArray) -> float:
    if family == "logistic":
        probability = np.clip(prediction, 1e-8, 1.0 - 1e-8)
        return float(
            -np.sum(response * np.log(probability) + (1.0 - response) * np.log1p(-probability))
        )
    return float(np.sum(np.square(response - prediction)))


def _explained_fraction(null_loss: float, model_loss: float) -> float:
    return float(1.0 - model_loss / max(null_loss, 1e-12))


def _cosine(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _benjamini_hochberg(values: FloatArray) -> FloatArray:
    clean = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    if clean.size == 0:
        return clean
    order = np.argsort(clean)
    ranked = clean[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _signed_permutation_null(
    left: FloatArray,
    right: FloatArray,
) -> tuple[float, float, float]:
    """Calibrate geometry by relabeling and reflecting common coordinates."""

    left_norm = max(float(np.linalg.norm(left)), 1e-12)
    right_norm = max(float(np.linalg.norm(right)), 1e-12)
    left_unit = left / left_norm
    observed_right = right / right_norm
    observed_cosine = abs(float(np.dot(left_unit, observed_right)))
    observed_distance = float(
        min(
            float(np.linalg.norm(left_unit - observed_right)),
            float(np.linalg.norm(left_unit + observed_right)),
        )
    )
    null_cosines: list[float] = []
    null_distances: list[float] = []
    dimensions = len(left)
    for order in permutations(range(dimensions)):
        permuted = observed_right[list(order)]
        for signs in product((-1.0, 1.0), repeat=dimensions):
            transformed = permuted * np.asarray(signs, dtype=np.float64)
            cosine = abs(float(np.dot(left_unit, transformed)))
            distance = float(
                min(
                    float(np.linalg.norm(left_unit - transformed)),
                    float(np.linalg.norm(left_unit + transformed)),
                )
            )
            null_cosines.append(cosine)
            null_distances.append(distance)
    cosine_null = np.asarray(null_cosines, dtype=np.float64)
    distance_null = np.asarray(null_distances, dtype=np.float64)
    p_value = (1.0 + float(np.sum(cosine_null >= observed_cosine - 1e-12))) / (
        len(cosine_null) + 1.0
    )
    # Small percentiles denote a closer pair than most relabeled/reflected axes.
    distance_percentile = (
        1.0 + float(np.sum(distance_null <= observed_distance + 1e-12))
    ) / (len(distance_null) + 1.0)
    return p_value, float(np.quantile(cosine_null, 0.95)), distance_percentile


def _direction_stability(draws: FloatArray, center: FloatArray) -> tuple[float, float]:
    if len(draws) == 0 or np.linalg.norm(center) <= 1e-12:
        return 0.0, 0.0
    cosines = np.asarray([_cosine(draw, center) for draw in draws], dtype=np.float64)
    return float(np.mean(cosines)), float(np.mean(cosines >= 0.80))


def _state_metric_transform(state: FloatArray) -> FloatArray:
    """Return a symmetric square root of the empirical state covariance.

    A coefficient vector depends on the particular scaling and rotation of an
    embedding.  In contrast, ``beta.T @ Cov(Z) @ gamma`` is the covariance of
    the two fitted linear state scores.  Mapping coefficients through this
    square root lets the rest of the projective-geometry code operate on that
    coordinate-invariant response metric.
    """

    centered = np.asarray(state, dtype=np.float64) - np.mean(
        state, axis=0, keepdims=True
    )
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return np.asarray(
        eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T,
        dtype=np.float64,
    )


def _bootstrap_coefficients(
    state: FloatArray,
    depth: FloatArray,
    response: FloatArray,
    *,
    family: str,
    config: LoadingGroupConfig,
    seed: int,
) -> FloatArray:
    if config.bootstrap_repetitions <= 0:
        return np.empty((0, state.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws: list[FloatArray] = []
    positive = np.flatnonzero(response == 1) if family == "logistic" else np.empty(0, int)
    negative = np.flatnonzero(response == 0) if family == "logistic" else np.empty(0, int)
    for _ in range(config.bootstrap_repetitions):
        if family == "logistic":
            indices = np.concatenate(
                [
                    rng.choice(negative, size=len(negative), replace=True),
                    rng.choice(positive, size=len(positive), replace=True),
                ]
            )
            rng.shuffle(indices)
        else:
            indices = rng.integers(0, len(response), size=len(response))
        sampled_state = state[indices]
        sampled_depth = depth[indices]
        depth_residual, _ = _orthogonal_depth(
            sampled_state,
            sampled_depth,
            sampled_state,
            sampled_depth,
            alpha=config.depth_ridge_alpha,
        )
        joint = np.column_stack([sampled_state, depth_residual])
        _, coefficients = _fit_predict(
            family,
            joint,
            response[indices],
            joint,
            config,
        )
        draws.append(np.asarray(coefficients[: state.shape[1]], dtype=np.float64))
    return np.vstack(draws)


def _estimate_feature(
    modality: str,
    feature: str,
    assay: AssayData,
    coordinate_columns: tuple[str, ...],
    config: LoadingGroupConfig,
    feature_seed: int,
) -> _FeatureFit:
    family = "ridge" if modality == "rna" else "logistic"
    raw_response = assay.focal[feature].to_numpy(dtype=np.float64)
    if family == "ridge":
        response_sd = float(raw_response.std(ddof=0))
        if response_sd <= 1e-12:
            raise ValueError(f"Continuous RNA feature {feature} is constant")
        response = (raw_response - float(raw_response.mean())) / response_sd
    else:
        response_sd = float(raw_response.std(ddof=0))
        response = raw_response
    folds = _folds_for_response(
        response,
        family=family,
        requested=config.folds,
        seed=feature_seed,
    )
    n = len(response)
    state_prediction = np.zeros(n, dtype=np.float64)
    depth_prediction = np.zeros(n, dtype=np.float64)
    joint_prediction = np.zeros(n, dtype=np.float64)
    null_prediction = np.zeros(n, dtype=np.float64)
    conditional_fold_coefficients: list[FloatArray] = []
    marginal_fold_coefficients: list[FloatArray] = []
    for train, test in folds:
        depth_train, depth_test = _orthogonal_depth(
            assay.coordinates[train],
            assay.log_depth[train],
            assay.coordinates[test],
            assay.log_depth[test],
            alpha=config.depth_ridge_alpha,
        )
        state_prediction[test], marginal = _fit_predict(
            family,
            assay.coordinates[train],
            response[train],
            assay.coordinates[test],
            config,
        )
        depth_prediction[test], _ = _fit_predict(
            family,
            depth_train[:, None],
            response[train],
            depth_test[:, None],
            config,
        )
        joint_train = np.column_stack([assay.coordinates[train], depth_train])
        joint_test = np.column_stack([assay.coordinates[test], depth_test])
        joint_prediction[test], conditional = _fit_predict(
            family,
            joint_train,
            response[train],
            joint_test,
            config,
        )
        baseline = float(response[train].mean())
        null_prediction[test] = baseline
        marginal_fold_coefficients.append(marginal)
        conditional_fold_coefficients.append(conditional[: assay.coordinates.shape[1]])

    conditional_folds = np.vstack(conditional_fold_coefficients)
    marginal_folds = np.vstack(marginal_fold_coefficients)
    conditional_loading = conditional_folds.mean(axis=0)
    marginal_loading = marginal_folds.mean(axis=0)
    null_loss = _loss(family, response, null_prediction)
    state_loss = _loss(family, response, state_prediction)
    depth_loss = _loss(family, response, depth_prediction)
    joint_loss = _loss(family, response, joint_prediction)
    marginal_state_fit = _explained_fraction(null_loss, state_loss)
    depth_fit = _explained_fraction(null_loss, depth_loss)
    joint_fit = _explained_fraction(null_loss, joint_loss)
    partial_state_fit = _explained_fraction(depth_loss, joint_loss)
    state_metric_transform = _state_metric_transform(assay.coordinates)
    conditional_response_loading = state_metric_transform @ conditional_loading
    marginal_response_loading = state_metric_transform @ marginal_loading
    conditional_response_folds = conditional_folds @ state_metric_transform
    marginal_response_folds = marginal_folds @ state_metric_transform
    fold_sd = conditional_folds.std(axis=0, ddof=1)
    fold_response_sd = conditional_response_folds.std(axis=0, ddof=1)
    fold_se = fold_sd / np.sqrt(len(conditional_folds))
    coefficient_fold_mean_cosine, coefficient_fold_stability = _direction_stability(
        conditional_folds, conditional_loading
    )
    fold_mean_cosine, fold_stability = _direction_stability(
        conditional_response_folds, conditional_response_loading
    )
    bootstrap = _bootstrap_coefficients(
        assay.coordinates,
        assay.log_depth,
        response,
        family=family,
        config=config,
        seed=feature_seed + 1_000_003,
    )
    if len(bootstrap) >= 2:
        bootstrap_response = bootstrap @ state_metric_transform
        bootstrap_sd = bootstrap.std(axis=0, ddof=1)
        bootstrap_response_sd = bootstrap_response.std(axis=0, ddof=1)
        ci_low = np.quantile(bootstrap, 0.025, axis=0)
        ci_high = np.quantile(bootstrap, 0.975, axis=0)
        coefficient_bootstrap_mean_cosine, coefficient_bootstrap_stability = (
            _direction_stability(bootstrap, conditional_loading)
        )
        bootstrap_mean_cosine, bootstrap_stability = _direction_stability(
            bootstrap_response, conditional_response_loading
        )
        sign_stability = np.mean(
            np.sign(bootstrap) == np.sign(conditional_loading)[None, :], axis=0
        )
        direction_stability = bootstrap_stability
        direction_mean_cosine = bootstrap_mean_cosine
        response_uncertainty = bootstrap_response_sd
    else:
        bootstrap_sd = np.full_like(conditional_loading, np.nan)
        ci_low = np.full_like(conditional_loading, np.nan)
        ci_high = np.full_like(conditional_loading, np.nan)
        sign_stability = np.mean(
            np.sign(conditional_folds) == np.sign(conditional_loading)[None, :], axis=0
        )
        direction_stability = fold_stability
        direction_mean_cosine = fold_mean_cosine
        response_uncertainty = fold_response_sd
        coefficient_bootstrap_mean_cosine = float("nan")
        coefficient_bootstrap_stability = float("nan")
    coefficient_loading_norm = float(np.linalg.norm(conditional_loading))
    loading_norm = float(np.linalg.norm(conditional_response_loading))
    uncertainty_norm = float(np.linalg.norm(response_uncertainty))
    feature_row = {
        "feature": feature,
        "modality": modality,
        "channel": "RNA" if modality == "rna" else modality.removeprefix("chrom_"),
        "model_family": family,
        "n_cells": n,
        "n_events": int(raw_response.sum()) if family == "logistic" else "",
        "event_fraction": float(raw_response.mean()) if family == "logistic" else "",
        "response_mean": float(raw_response.mean()),
        "response_sd": response_sd,
        "oof_marginal_state_fit": marginal_state_fit,
        "oof_partial_state_fit": partial_state_fit,
        "oof_depth_only_fit": depth_fit,
        "oof_joint_fit": joint_fit,
        "conditional_loading_norm": loading_norm,
        "conditional_coefficient_norm": coefficient_loading_norm,
        "conditional_state_response_sd": loading_norm,
        "loading_uncertainty_norm": uncertainty_norm,
        "relative_loading_uncertainty": uncertainty_norm / max(loading_norm, 1e-12),
        "fold_direction_mean_cosine": fold_mean_cosine,
        "fold_direction_stability": fold_stability,
        "coefficient_fold_direction_mean_cosine": coefficient_fold_mean_cosine,
        "coefficient_fold_direction_stability": coefficient_fold_stability,
        "bootstrap_direction_mean_cosine": (
            bootstrap_mean_cosine if len(bootstrap) >= 2 else float("nan")
        ),
        "bootstrap_direction_stability": (
            bootstrap_stability if len(bootstrap) >= 2 else float("nan")
        ),
        "coefficient_bootstrap_direction_mean_cosine": (
            coefficient_bootstrap_mean_cosine
            if len(bootstrap) >= 2
            else float("nan")
        ),
        "coefficient_bootstrap_direction_stability": (
            coefficient_bootstrap_stability
            if len(bootstrap) >= 2
            else float("nan")
        ),
        "loading_direction_stability": direction_stability,
        "loading_direction_mean_cosine": direction_mean_cosine,
        "mean_integration_uncertainty": float(assay.coordinate_uncertainty.mean()),
        "n_folds": len(folds),
        "n_bootstrap": len(bootstrap),
        "response_transform": "zscore" if family == "ridge" else "binary_logit",
        "loading_estimand": (
            "covariance-metric state-response score conditional on "
            "state-orthogonal depth"
        ),
    }
    loading_rows: list[dict[str, Any]] = []
    for index, coordinate in enumerate(coordinate_columns):
        loading_rows.append(
            {
                "feature": feature,
                "modality": modality,
                "channel": feature_row["channel"],
                "model_family": family,
                "state_coordinate": coordinate,
                "conditional_coefficient": float(conditional_loading[index]),
                "marginal_coefficient": float(marginal_loading[index]),
                "response_metric_coefficient": float(
                    conditional_response_loading[index]
                ),
                "marginal_response_metric_coefficient": float(
                    marginal_response_loading[index]
                ),
                "fold_coefficient_sd": float(fold_sd[index]),
                "fold_coefficient_se": float(fold_se[index]),
                "bootstrap_coefficient_sd": float(bootstrap_sd[index]),
                "bootstrap_ci_2.5": float(ci_low[index]),
                "bootstrap_ci_97.5": float(ci_high[index]),
                "coefficient_sign_stability": float(sign_stability[index]),
                "loading_estimand": feature_row["loading_estimand"],
            }
        )
    return _FeatureFit(
        feature_row=feature_row,
        loading_rows=loading_rows,
        conditional_loading=np.asarray(conditional_loading, dtype=np.float64),
    )


def _pair_evidence(
    feature_models: pd.DataFrame,
    vectors: Mapping[str, FloatArray],
    response_vectors: Mapping[str, FloatArray],
    config: LoadingGroupConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_lookup = feature_models.set_index("feature")
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(vectors), 2):
        left_vector, right_vector = vectors[left], vectors[right]
        left_model = model_lookup.loc[left]
        right_model = model_lookup.loc[right]
        same_assay = bool(left_model["modality"] == right_model["modality"])
        geometry_left = response_vectors[left] if same_assay else left_vector
        geometry_right = response_vectors[right] if same_assay else right_vector
        left_norm = float(np.linalg.norm(geometry_left))
        right_norm = float(np.linalg.norm(geometry_right))
        raw_coefficient_cosine = _cosine(left_vector, right_vector)
        cosine = _cosine(geometry_left, geometry_right)
        absolute_cosine = abs(cosine)
        if left_norm > 1e-12 and right_norm > 1e-12:
            left_unit = geometry_left / left_norm
            right_unit = geometry_right / right_norm
            signed_distance = float(np.linalg.norm(left_unit - right_unit))
            sign_invariant_distance = float(
                min(
                    float(np.linalg.norm(left_unit - right_unit)),
                    float(np.linalg.norm(left_unit + right_unit)),
                )
            )
        else:
            signed_distance = 2.0
            sign_invariant_distance = 2.0
        stability = float(
            np.sqrt(
                max(float(left_model["loading_direction_stability"]), 0.0)
                * max(float(right_model["loading_direction_stability"]), 0.0)
            )
        )
        marginal_fit = float(
            np.sqrt(
                max(float(left_model["oof_marginal_state_fit"]), 0.0)
                * max(float(right_model["oof_marginal_state_fit"]), 0.0)
            )
        )
        partial_fit = float(
            np.sqrt(
                max(float(left_model["oof_partial_state_fit"]), 0.0)
                * max(float(right_model["oof_partial_state_fit"]), 0.0)
            )
        )
        empirical_p, null_cosine_95, distance_percentile = _signed_permutation_null(
            geometry_left, geometry_right
        )
        pair_state_score = absolute_cosine * stability * partial_fit
        eligible = bool(
            min(left_norm, right_norm) >= config.min_loading_norm
            and min(
                float(left_model["oof_partial_state_fit"]),
                float(right_model["oof_partial_state_fit"]),
            )
            >= config.min_state_fit
            and min(
                float(left_model["loading_direction_stability"]),
                float(right_model["loading_direction_stability"]),
            )
            >= config.min_direction_stability
            and absolute_cosine >= config.min_abs_loading_cosine
            and sign_invariant_distance <= config.max_unit_loading_distance
        )
        rows.append(
            {
                "pair_key": f"{left}|||{right}",
                "node1": left,
                "node2": right,
                "channel1": str(left_model["channel"]),
                "channel2": str(right_model["channel"]),
                "modality1": str(left_model["modality"]),
                "modality2": str(right_model["modality"]),
                "same_assay": same_assay,
                "loading_cosine": cosine,
                "absolute_loading_cosine": absolute_cosine,
                "raw_coefficient_loading_cosine": raw_coefficient_cosine,
                "absolute_raw_coefficient_loading_cosine": abs(
                    raw_coefficient_cosine
                ),
                "state_response_correlation": cosine if same_assay else float("nan"),
                "absolute_state_response_correlation": (
                    absolute_cosine if same_assay else float("nan")
                ),
                "similarity_metric": (
                    "within_assay_state_score_correlation"
                    if same_assay
                    else "aligned_coefficient_cosine"
                ),
                "signed_unit_loading_distance": signed_distance,
                "sign_invariant_unit_loading_distance": sign_invariant_distance,
                "orientation_relation": "same" if cosine >= 0 else "opposite",
                "minimum_loading_norm": min(left_norm, right_norm),
                "node1_oof_marginal_state_fit": float(
                    left_model["oof_marginal_state_fit"]
                ),
                "node2_oof_marginal_state_fit": float(
                    right_model["oof_marginal_state_fit"]
                ),
                "node1_oof_partial_state_fit": float(
                    left_model["oof_partial_state_fit"]
                ),
                "node2_oof_partial_state_fit": float(
                    right_model["oof_partial_state_fit"]
                ),
                "geometric_mean_marginal_state_fit": marginal_fit,
                "geometric_mean_partial_state_fit": partial_fit,
                "node1_loading_stability": float(
                    left_model["loading_direction_stability"]
                ),
                "node2_loading_stability": float(
                    right_model["loading_direction_stability"]
                ),
                "loading_stability_support": stability,
                "pair_state_score": pair_state_score,
                "loading_empirical_null_p": empirical_p,
                "loading_null_absolute_cosine_95": null_cosine_95,
                "unit_distance_null_percentile": distance_percentile,
                "threshold_eligible": eligible,
            }
        )
    evidence = pd.DataFrame(rows)
    if evidence.empty:
        return evidence, evidence.copy()
    evidence["loading_empirical_null_q"] = _benjamini_hochberg(
        evidence["loading_empirical_null_p"].to_numpy(dtype=np.float64)
    )
    ranks: dict[tuple[str, str], int] = {}
    for feature in sorted(vectors):
        candidates = evidence.loc[
            evidence["threshold_eligible"]
            & ((evidence["node1"] == feature) | (evidence["node2"] == feature))
        ].copy()
        candidates = candidates.sort_values(
            ["pair_state_score", "sign_invariant_unit_loading_distance", "node1", "node2"],
            ascending=[False, True, True, True],
        )
        for rank, row in enumerate(candidates.itertuples(index=False), start=1):
            neighbor = row.node2 if row.node1 == feature else row.node1
            ranks[(feature, str(neighbor))] = rank
    rank1: list[int] = []
    rank2: list[int] = []
    selected: list[bool] = []
    for row in evidence.itertuples(index=False):
        left_rank = ranks.get((str(row.node1), str(row.node2)), 0)
        right_rank = ranks.get((str(row.node2), str(row.node1)), 0)
        rank1.append(left_rank)
        rank2.append(right_rank)
        selected.append(
            bool(
                row.threshold_eligible
                and 0 < left_rank <= config.max_neighbors_per_feature
                and 0 < right_rank <= config.max_neighbors_per_feature
            )
        )
    evidence["feature1_neighbor_rank"] = rank1
    evidence["feature2_neighbor_rank"] = rank2
    evidence["selected_sparse"] = selected
    return evidence, evidence.loc[evidence["selected_sparse"]].reset_index(drop=True)


def _loading_groups(
    sparse_pairs: pd.DataFrame,
    vectors: Mapping[str, FloatArray],
    coordinate_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "loading_group",
        "n_features",
        "n_assays",
        "features",
        "modalities",
        "n_sparse_edges",
        "mean_absolute_loading_cosine",
        "minimum_absolute_loading_cosine",
        *[
            f"centroid__{coordinate.removeprefix('integration__')}"
            for coordinate in coordinate_columns
        ],
    ]
    membership_columns = [
        "loading_group",
        "feature",
        "orientation_to_group",
        "unit_distance_to_group_centroid",
    ]
    if sparse_pairs.empty:
        return pd.DataFrame(columns=group_columns), pd.DataFrame(
            columns=membership_columns
        )
    parent = {feature: feature for feature in vectors}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row in sparse_pairs.itertuples(index=False):
        union(str(row.node1), str(row.node2))
    components: dict[str, list[str]] = {}
    for feature in sorted(vectors):
        root = find(feature)
        components.setdefault(root, []).append(feature)
    components = {root: members for root, members in components.items() if len(members) >= 2}
    group_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    ordered_components = sorted(components.values(), key=lambda values: tuple(values))
    for group_index, members in enumerate(ordered_components, start=1):
        group_id = f"loading_group_{group_index:03d}"
        reference = vectors[members[0]] / max(float(np.linalg.norm(vectors[members[0]])), 1e-12)
        aligned_units: list[FloatArray] = []
        signs: dict[str, int] = {}
        for feature in members:
            vector = vectors[feature]
            unit = vector / max(float(np.linalg.norm(vector)), 1e-12)
            sign = 1 if float(np.dot(unit, reference)) >= 0 else -1
            signs[feature] = sign
            aligned_units.append(sign * unit)
        centroid = np.mean(np.vstack(aligned_units), axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        internal = sparse_pairs.loc[
            sparse_pairs["node1"].isin(members) & sparse_pairs["node2"].isin(members)
        ]
        modalities = sorted(
            set(internal["modality1"].astype(str)) | set(internal["modality2"].astype(str))
        )
        group_row: dict[str, Any] = {
            "loading_group": group_id,
            "n_features": len(members),
            "n_assays": len(modalities),
            "features": ",".join(members),
            "modalities": ",".join(modalities),
            "n_sparse_edges": len(internal),
            "mean_absolute_loading_cosine": float(internal["absolute_loading_cosine"].mean()),
            "minimum_absolute_loading_cosine": float(internal["absolute_loading_cosine"].min()),
        }
        for coordinate, value in zip(coordinate_columns, centroid, strict=True):
            group_row[f"centroid__{coordinate.removeprefix('integration__')}"] = float(value)
        group_rows.append(group_row)
        for feature, aligned in zip(members, aligned_units, strict=True):
            membership_rows.append(
                {
                    "loading_group": group_id,
                    "feature": feature,
                    "orientation_to_group": signs[feature],
                    "unit_distance_to_group_centroid": float(np.linalg.norm(aligned - centroid)),
                }
            )
    return pd.DataFrame(group_rows, columns=group_columns), pd.DataFrame(
        membership_rows,
        columns=membership_columns,
    )


def estimate_loading_groups(
    inputs: LoadingInputs,
    config: LoadingGroupConfig | None = None,
) -> LoadingGroupResult:
    """Estimate assay-local state loadings and sparse shared-axis groups."""

    cfg = config or LoadingGroupConfig()
    feature_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    vectors: dict[str, FloatArray] = {}
    response_vectors: dict[str, FloatArray] = {}
    feature_index = 0
    for modality, assay in sorted(inputs.assays.items()):
        for feature in assay.focal.columns:
            fit = _estimate_feature(
                modality,
                str(feature),
                assay,
                inputs.coordinate_columns,
                cfg,
                cfg.seed + 7_919 * feature_index,
            )
            feature_rows.append(fit.feature_row)
            loading_rows.extend(fit.loading_rows)
            vectors[str(feature)] = fit.conditional_loading
            response_vectors[str(feature)] = np.asarray(
                [
                    row["response_metric_coefficient"]
                    for row in fit.loading_rows
                ],
                dtype=np.float64,
            )
            feature_index += 1
    feature_models = pd.DataFrame(feature_rows)
    loadings = pd.DataFrame(loading_rows)
    pair_evidence, sparse_pairs = _pair_evidence(
        feature_models, vectors, response_vectors, cfg
    )
    groups, membership = _loading_groups(
        sparse_pairs,
        response_vectors,
        inputs.coordinate_columns,
    )
    provenance = {
        "schema_version": "cell_state_loading_groups.v1",
        "estimand": (
            "assay-local focal response to aligned background state conditional on "
            "state-orthogonal leave-focal depth"
        ),
        "config": asdict(cfg),
        "coordinate_columns": list(inputs.coordinate_columns),
        "coordinate_center": inputs.coordinate_center.tolist(),
        "coordinate_scale": inputs.coordinate_scale.tolist(),
        "assay_cell_counts": {
            modality: int(len(assay.cell_ids)) for modality, assay in inputs.assays.items()
        },
        "input_file_sha256": inputs.input_hashes,
        "input_policy": {
            "files_read": sorted(inputs.input_hashes),
            "observed_focal_tables_read": True,
            "separately_recomputed_safe_embedding_read": True,
            "simulator_supplied_embedding_read": False,
            "background_matrices_read": False,
            "membership_or_metacells_read": False,
            "scorer_or_truth_read": False,
            "external_data_read": False,
        },
        "identifiability": {
            "cross_assay_rows_paired": False,
            "shared_loading_axis_is_adjacency": False,
            "shared_loading_axis_is_direct_causation": False,
            "evidence_interpretation": (
                "similar marginal response to common background state coordinates"
            ),
        },
        "state_depth_separation": {
            "depth_source": "observed leave-focal background library size",
            "depth_transform": "within-assay standardized log1p",
            "projection": "training-fold ridge depth_on_state",
            "nuisance": "training-standardized state-orthogonal depth residual",
            "shared_state_depth_variation_allocated_to_state": True,
        },
        "within_assay_geometry": {
            "metric": "correlation_of_fitted_linear_state_scores",
            "coefficient_inner_product": "beta_left.T @ Cov(Z_assay) @ beta_right",
            "invariant_to_invertible_coordinate_reparameterization": True,
            "raw_coefficient_cosine_used_for_within_assay_selection": False,
            "cross_assay_rows_compared": False,
        },
        "n_features": int(len(feature_models)),
        "n_pair_candidates": int(len(pair_evidence)),
        "n_sparse_pairs": int(len(sparse_pairs)),
        "n_loading_groups": int(len(groups)),
    }
    return LoadingGroupResult(
        feature_models=feature_models,
        loadings=loadings,
        pair_evidence=pair_evidence,
        sparse_pairs=sparse_pairs,
        groups=groups,
        group_membership=membership,
        provenance=provenance,
    )


def _validate_output_path(out_dir: Path | str, inputs: LoadingInputs) -> Path:
    output = Path(out_dir).absolute()
    resolved = output.resolve(strict=False)
    if resolved == inputs.focal_root or inputs.focal_root in resolved.parents:
        raise ValueError("Output must be outside the focal discovery/ tree")
    if resolved == inputs.embedding_path.parent or inputs.embedding_path.parent in resolved.parents:
        raise ValueError("Output must be outside the safe embedding input directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve(strict=True)


def run_loading_group_analysis(
    focal_discovery_dir: Path | str,
    safe_embedding_path: Path | str,
    out_dir: Path | str,
    config: LoadingGroupConfig | None = None,
) -> dict[str, Any]:
    """Load, estimate, and write the cell-level loading evidence layer."""

    cfg = config or LoadingGroupConfig()
    inputs = load_loading_inputs(focal_discovery_dir, safe_embedding_path)
    output = _validate_output_path(out_dir, inputs)
    result = estimate_loading_groups(inputs, cfg)
    tables = {
        "feature_state_models.tsv": result.feature_models,
        "conditional_state_loadings.tsv": result.loadings,
        "loading_pair_evidence.tsv": result.pair_evidence,
        "sparse_loading_pairs.tsv": result.sparse_pairs,
        "loading_groups.tsv": result.groups,
        "loading_group_membership.tsv": result.group_membership,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, sep="\t", index=False)
    _write_json(output / "analysis_provenance.json", result.provenance)
    summary = {
        "schema_version": "cell_state_loading_groups.run.v1",
        "n_assays": len(inputs.assays),
        "n_features": len(result.feature_models),
        "n_pair_candidates": len(result.pair_evidence),
        "n_sparse_pairs": len(result.sparse_pairs),
        "n_loading_groups": len(result.groups),
        "row_pairing_used": False,
        "truth_or_scorer_used": False,
        "files_written": [*tables, "analysis_provenance.json", "run_manifest.json"],
    }
    _write_json(output / "run_manifest.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal-discovery-dir", type=Path, required=True)
    parser.add_argument("--safe-embedding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--depth-ridge-alpha", type=float, default=2.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-loading-norm", type=float, default=0.10)
    parser.add_argument("--min-state-fit", type=float, default=0.01)
    parser.add_argument("--min-direction-stability", type=float, default=0.50)
    parser.add_argument("--min-abs-loading-cosine", type=float, default=0.80)
    parser.add_argument("--max-unit-loading-distance", type=float, default=0.65)
    parser.add_argument("--max-neighbors-per-feature", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = LoadingGroupConfig(
        folds=args.folds,
        ridge_alpha=args.ridge_alpha,
        depth_ridge_alpha=args.depth_ridge_alpha,
        logistic_c=args.logistic_c,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        min_loading_norm=args.min_loading_norm,
        min_state_fit=args.min_state_fit,
        min_direction_stability=args.min_direction_stability,
        min_abs_loading_cosine=args.min_abs_loading_cosine,
        max_unit_loading_distance=args.max_unit_loading_distance,
        max_neighbors_per_feature=args.max_neighbors_per_feature,
    )
    summary = run_loading_group_analysis(
        args.focal_discovery_dir,
        args.safe_embedding,
        args.out_dir,
        config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AssayData",
    "LoadingGroupConfig",
    "LoadingGroupResult",
    "LoadingInputs",
    "estimate_loading_groups",
    "load_loading_inputs",
    "run_loading_group_analysis",
]
