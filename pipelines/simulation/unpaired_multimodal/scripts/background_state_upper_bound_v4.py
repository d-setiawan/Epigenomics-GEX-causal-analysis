#!/usr/bin/env python3
"""Scorer-only background-state upper bound for CA-BOSS v4.

This module is a validation ceiling, not a discovery algorithm.  It deliberately
opens the simulator's ``scorer/`` domain, uses the true per-assay latent state,
and uses the typed truth table to define targeted observation-layer tests.  Its
outputs are therefore forbidden as inputs to CA-BOSS or any other discovery
method.

The ceiling is not a copy of the truth graph.  A truth-supported edge is emitted
only when the corresponding relationship is detectable in the *observed* focal
measurements:

* within-assay direct edges require a conditional likelihood-ratio test after
  adjustment for oracle state, state-orthogonal background depth, and the other
  registered collider parent;
* shared-state and slow peak--gene targets require significant focal response
  to oracle state and a matching signed loading direction;
* equivalence-class targets are credited through the strongest empirically
  supported member pair, never an arbitrarily privileged P/S representative;
* sentinel nulls are tested by the same conditional/loading rules and enter the
  false-adjacency gate when spuriously supported.

No rows are paired across assays.  The only cross-assay privilege is that the
simulator's true latent coordinates have a common coordinate system.  Passing
this arm establishes that the simulator observation layer and attainable typed
estimand are compatible with the preregistered gate; it does not establish that
an observed-only integration method can recover that coordinate system.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import expit, xlogy, xlog1py
from scipy.stats import chi2, f as f_distribution
from sklearn.linear_model import LogisticRegression

from score_multimodal_v4_graph import MAX_FOCAL_EDGES_PER_WORLD, score_v4


FloatArray = NDArray[np.float64]
SCHEMA_VERSION = "ca_boss_v4.background_state_upper_bound.v1"
V3_MEAN_BIOLOGICAL_EDGES = 22.725
TARGET_NULL_FPR = 0.10
EXPECTED_MODALITIES = {
    "rna",
    "chrom_H3K27ac",
    "chrom_H3K27me3",
    "chrom_H3K4me1",
    "chrom_H3K4me2",
    "chrom_H3K4me3",
    "chrom_H3K9me3",
}
DIRECT = "within_assay_direct"
WITHIN_SHARED = "within_assay_shared_state"
CROSS_SHARED = "cross_assay_shared_state"
SLOW_GENE = "slow_peak_gene"
SENTINEL_NULL = "sentinel_null"
POSITIVE_STATE_TYPES = {WITHIN_SHARED, CROSS_SHARED, SLOW_GENE}
CLASS_COLUMNS = (
    "direct_class_success",
    "collider_class_success",
    "shared_class_success",
    "slow_class_success",
    "null_class_success",
)
FACTOR_COLUMNS = (
    "A_cells",
    "B_detection",
    "C_population_divergence",
    "D_background_observability",
)
REGISTERED_SPLITS: dict[str, dict[str, Any]] = {
    "development": {
        "seeds": tuple(range(31_001, 31_017)),
        "required_world_successes": 14,
        "required_class_successes": 14,
        "required_factor_level_successes": 7,
        "required_design_cell_successes": 2,
    },
    "locked": {
        "seeds": tuple(range(42_001, 42_041)),
        "required_world_successes": 34,
        "required_class_successes": 34,
        "required_factor_level_successes": 16,
        "required_design_cell_successes": 3,
    },
}


@dataclass(frozen=True)
class UpperBoundConfig:
    """Frozen targeted-test settings for the validation ceiling."""

    direct_fdr: float = 0.10
    state_fdr: float = 0.10
    null_fdr: float = 0.10
    min_state_loading_norm: float = 0.10
    min_signed_loading_cosine: float = 0.60
    logistic_c: float = 10_000.0
    logistic_max_iter: int = 4_000

    def __post_init__(self) -> None:
        for name in ("direct_fdr", "state_fdr", "null_fdr"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.min_state_loading_norm < 0.0:
            raise ValueError("min_state_loading_norm may not be negative")
        if not -1.0 <= self.min_signed_loading_cosine <= 1.0:
            raise ValueError("min_signed_loading_cosine must lie in [-1, 1]")
        if self.logistic_c <= 0.0 or self.logistic_max_iter < 100:
            raise ValueError("logistic settings are invalid")


@dataclass
class _Assay:
    modality: str
    cell_ids: NDArray[np.str_]
    state: FloatArray
    depth_residual: FloatArray
    focal: pd.DataFrame


@dataclass
class _FeatureStateFit:
    node: str
    modality: str
    outcome_type: str
    loading: FloatArray
    loading_norm: float
    p_value: float
    q_value: float = 1.0
    supported: bool = False


@dataclass
class WorldUpperBoundResult:
    """In-memory scorer result; all members are forbidden discovery inputs."""

    world_seed: int | None
    metrics: dict[str, Any]
    edges: pd.DataFrame
    diagnostics: pd.DataFrame
    feature_state_fits: pd.DataFrame
    provenance: dict[str, Any]


class UpperBoundError(ValueError):
    """Raised when the scorer/discovery firewall or world schema is invalid."""


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _regular_file(path: Path, *, label: str, parent: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise UpperBoundError(f"{label} must be a direct regular file: {path}")
    resolved = path.resolve(strict=True)
    if resolved.parent != parent:
        raise UpperBoundError(f"{label} escapes its registered information domain")
    return resolved


def _validate_world_root(world_dir: Path | str) -> tuple[Path, Path, Path]:
    lexical = Path(world_dir).absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise UpperBoundError("Upper-bound input must be a direct v4 world directory")
    world = lexical.resolve(strict=True)
    discovery_lexical = lexical / "discovery"
    scorer_lexical = lexical / "scorer"
    if (
        discovery_lexical.is_symlink()
        or scorer_lexical.is_symlink()
        or not discovery_lexical.is_dir()
        or not scorer_lexical.is_dir()
    ):
        raise UpperBoundError("World must contain separate non-symlink discovery/ and scorer/")
    discovery = discovery_lexical.resolve(strict=True)
    scorer = scorer_lexical.resolve(strict=True)
    if discovery == scorer or discovery.parent != world or scorer.parent != world:
        raise UpperBoundError("Discovery and scorer domains must be direct, separate children")
    manifest_path = _regular_file(
        scorer / "manifest.json", label="scorer manifest", parent=scorer
    )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UpperBoundError("Scorer manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or not manifest.get("scorer_only", False):
        raise UpperBoundError("Scorer manifest must declare scorer_only=true")
    if not manifest.get("must_not_be_read_by_discovery", False):
        raise UpperBoundError("Scorer manifest lacks the discovery prohibition")
    return world, discovery, scorer


def _validate_output(output: Path | str, world: Path) -> Path:
    lexical = Path(output).absolute()
    resolved_parent = lexical.parent.resolve(strict=True)
    target = resolved_parent / lexical.name
    discovery = (world / "discovery").resolve(strict=True)
    try:
        target.relative_to(discovery)
    except ValueError:
        pass
    else:
        raise UpperBoundError("Scorer-only upper-bound outputs may not enter discovery/")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise UpperBoundError(f"Refusing to overwrite non-empty upper-bound output: {target}")
    return target


def _truth_nodes(raw: Any, fallback: str) -> tuple[str, ...]:
    if raw is None or pd.isna(raw) or not str(raw).strip():
        return (fallback,)
    return tuple(dict.fromkeys(node for node in str(raw).split(";") if node))


def _bh_adjust(p_values: Sequence[float]) -> FloatArray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise UpperBoundError("P-values must be a finite one-dimensional vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise UpperBoundError("P-values must lie in [0, 1]")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def _state_orthogonal_depth(state: FloatArray, raw_depth: FloatArray) -> FloatArray:
    log_depth = np.log1p(np.asarray(raw_depth, dtype=np.float64))
    design = np.column_stack([np.ones(len(state)), state])
    coefficients, *_ = np.linalg.lstsq(design, log_depth, rcond=None)
    residual = log_depth - design @ coefficients
    scale = float(residual.std(ddof=0))
    return np.zeros_like(residual) if scale <= 1e-12 else residual / scale


def _logistic_fit(
    y: FloatArray,
    x: FloatArray,
    config: UpperBoundConfig,
) -> tuple[FloatArray, float]:
    response = np.asarray(y, dtype=np.float64)
    design = np.asarray(x, dtype=np.float64)
    if response.ndim != 1 or design.ndim != 2 or len(response) != len(design):
        raise UpperBoundError("Logistic outcome and design are not aligned")
    if len(np.unique(response)) < 2:
        return np.zeros(design.shape[1], dtype=np.float64), float("-inf")
    model = LogisticRegression(
        C=config.logistic_c,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=config.logistic_max_iter,
        tol=1e-9,
    ).fit(design, response.astype(np.int64))
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    eta = np.asarray(model.intercept_[0] + design @ coefficients, dtype=np.float64)
    probabilities = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
    log_likelihood = float(
        np.sum(xlogy(response, probabilities) + xlog1py(1.0 - response, -probabilities))
    )
    return coefficients, log_likelihood


def _binary_nested_test(
    y: FloatArray,
    base: FloatArray,
    added: FloatArray,
    config: UpperBoundConfig,
) -> tuple[FloatArray, float, float]:
    base_design = np.asarray(base, dtype=np.float64)
    added_design = np.asarray(added, dtype=np.float64)
    if added_design.ndim == 1:
        added_design = added_design[:, None]
    full_design = np.column_stack([base_design, added_design])
    _, base_ll = _logistic_fit(y, base_design, config)
    full_coefficients, full_ll = _logistic_fit(y, full_design, config)
    if not np.isfinite(base_ll) or not np.isfinite(full_ll):
        return np.zeros(added_design.shape[1]), 1.0, 0.0
    statistic = max(0.0, 2.0 * (full_ll - base_ll))
    p_value = float(chi2.sf(statistic, added_design.shape[1]))
    return full_coefficients[-added_design.shape[1] :], p_value, statistic


def _continuous_nested_test(
    y: FloatArray,
    base: FloatArray,
    added: FloatArray,
) -> tuple[FloatArray, float, float]:
    response = np.asarray(y, dtype=np.float64)
    base_design = np.column_stack([np.ones(len(response)), np.asarray(base, dtype=np.float64)])
    added_design = np.asarray(added, dtype=np.float64)
    if added_design.ndim == 1:
        added_design = added_design[:, None]
    full_design = np.column_stack([base_design, added_design])
    base_coefficients, *_ = np.linalg.lstsq(base_design, response, rcond=None)
    full_coefficients, *_ = np.linalg.lstsq(full_design, response, rcond=None)
    base_rss = float(np.sum(np.square(response - base_design @ base_coefficients)))
    full_rss = float(np.sum(np.square(response - full_design @ full_coefficients)))
    numerator_df = added_design.shape[1]
    denominator_df = max(len(response) - full_design.shape[1], 1)
    if full_rss <= 1e-14:
        statistic = float("inf") if base_rss > full_rss else 0.0
    else:
        statistic = max(0.0, (base_rss - full_rss) / numerator_df) / (
            full_rss / denominator_df
        )
    p_value = float(f_distribution.sf(statistic, numerator_df, denominator_df))
    return (
        np.asarray(full_coefficients[-numerator_df:], dtype=np.float64),
        p_value,
        statistic,
    )


def _is_binary(values: FloatArray) -> bool:
    unique = np.unique(np.asarray(values, dtype=np.float64))
    return bool(len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0}))


def _load_world(
    world_dir: Path | str,
) -> tuple[Path, pd.DataFrame, dict[str, _Assay], dict[str, str], dict[str, str]]:
    world, discovery, scorer = _validate_world_root(world_dir)
    truth_path = _regular_file(
        scorer / "truth_typed_edges.tsv", label="typed truth", parent=scorer
    )
    truth = pd.read_csv(truth_path, sep="\t", keep_default_na=False)
    required_truth = {"node1", "node2", "node_set1", "node_set2", "truth_type", "label"}
    missing_truth = sorted(required_truth - set(truth.columns))
    if missing_truth:
        raise UpperBoundError(f"Typed truth is missing columns {missing_truth}")

    focal_paths = {
        path.name.removeprefix("observed_focal__").removesuffix(".tsv"): path
        for path in discovery.glob("observed_focal__*.tsv")
    }
    state_paths = {
        path.name.removeprefix("cell_ground_truth__").removesuffix(".npz"): path
        for path in scorer.glob("cell_ground_truth__*.npz")
    }
    if set(focal_paths) != EXPECTED_MODALITIES or set(state_paths) != EXPECTED_MODALITIES:
        raise UpperBoundError(
            "Upper-bound world must contain RNA plus exactly the six preregistered marks"
        )

    raw: dict[str, tuple[NDArray[np.str_], FloatArray, pd.DataFrame, FloatArray]] = {}
    input_hashes = {"truth_typed_edges.tsv": _sha256(truth_path)}
    pooled_states: list[FloatArray] = []
    for modality in sorted(EXPECTED_MODALITIES):
        focal_path = _regular_file(
            focal_paths[modality], label=f"observed focal {modality}", parent=discovery
        )
        state_path = _regular_file(
            state_paths[modality], label=f"oracle state {modality}", parent=scorer
        )
        focal = pd.read_csv(focal_path, sep="\t", keep_default_na=False)
        required = {
            "cell",
            "modality",
            "observed_background_library_size",
            "observed_library_size",
        }
        missing = sorted(required - set(focal.columns))
        if missing:
            raise UpperBoundError(f"Focal table {modality} is missing columns {missing}")
        if set(focal["modality"].astype(str)) != {modality}:
            raise UpperBoundError(f"Focal modality column does not match {modality}")
        if focal["cell"].astype(str).duplicated().any():
            raise UpperBoundError(f"Focal cell IDs are duplicated for {modality}")
        with np.load(state_path, allow_pickle=False) as payload:
            if not {"cell_ids", "latent_state"}.issubset(payload.files):
                raise UpperBoundError(f"Oracle state archive is incomplete for {modality}")
            state_ids = np.asarray(payload["cell_ids"], dtype=str)
            state = np.asarray(payload["latent_state"], dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 3 or not np.isfinite(state).all():
            raise UpperBoundError(f"Oracle state must be finite n-by-3 for {modality}")
        if len(np.unique(state_ids)) != len(state_ids):
            raise UpperBoundError(f"Oracle state cell IDs are duplicated for {modality}")
        focal_ids = focal["cell"].astype(str).to_numpy()
        state_index = {cell: index for index, cell in enumerate(state_ids)}
        if set(focal_ids) != set(state_ids):
            raise UpperBoundError(f"Observed and oracle-state cell sets differ for {modality}")
        order = np.asarray([state_index[cell] for cell in focal_ids], dtype=np.int64)
        aligned_state = state[order]
        depth = pd.to_numeric(
            focal["observed_background_library_size"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if np.any(depth < 0.0) or not np.isfinite(depth).all():
            raise UpperBoundError(f"Observed background depth is invalid for {modality}")
        raw[modality] = (focal_ids.astype(str), aligned_state, focal, depth)
        pooled_states.append(aligned_state)
        input_hashes[f"discovery/{focal_path.name}"] = _sha256(focal_path)
        input_hashes[f"scorer/{state_path.name}"] = _sha256(state_path)

    pooled = np.vstack(pooled_states)
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=0)
    if np.any(scale <= 1e-10):
        raise UpperBoundError("Oracle state has a degenerate coordinate")
    assays: dict[str, _Assay] = {}
    node_to_modality: dict[str, str] = {}
    for modality, (cell_ids, state, focal, depth) in raw.items():
        common_state = (state - center) / scale
        depth_residual = _state_orthogonal_depth(common_state, depth)
        assays[modality] = _Assay(
            modality=modality,
            cell_ids=cell_ids,
            state=common_state,
            depth_residual=depth_residual,
            focal=focal,
        )
        biological = [
            str(column)
            for column in focal.columns
            if str(column).startswith("chr") or str(column).startswith("expr__")
        ]
        if not biological:
            raise UpperBoundError(f"Focal table contains no biological variables for {modality}")
        for node in biological:
            if node in node_to_modality:
                raise UpperBoundError(f"Biological node is duplicated across assays: {node}")
            values = pd.to_numeric(focal[node], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise UpperBoundError(f"Biological variable is non-finite: {node}")
            focal[node] = values
            node_to_modality[node] = modality

    truth_nodes = {
        node
        for row in truth.to_dict(orient="records")
        for node in (
            *_truth_nodes(row.get("node_set1"), str(row["node1"])),
            *_truth_nodes(row.get("node_set2"), str(row["node2"])),
        )
    }
    unknown = sorted(truth_nodes - set(node_to_modality))
    if unknown:
        raise UpperBoundError(f"Typed truth references unknown observed nodes: {unknown}")
    return world, truth, assays, node_to_modality, input_hashes


def _feature_values(node: str, assays: Mapping[str, _Assay], mapping: Mapping[str, str]) -> FloatArray:
    assay = assays[mapping[node]]
    return assay.focal[node].to_numpy(dtype=np.float64)


def _fit_state_models(
    assays: Mapping[str, _Assay],
    node_to_modality: Mapping[str, str],
    config: UpperBoundConfig,
) -> dict[str, _FeatureStateFit]:
    fits: dict[str, _FeatureStateFit] = {}
    for node, modality in sorted(node_to_modality.items()):
        assay = assays[modality]
        y = assay.focal[node].to_numpy(dtype=np.float64)
        base = assay.depth_residual[:, None]
        if _is_binary(y):
            loading, p_value, _ = _binary_nested_test(y, base, assay.state, config)
            outcome_type = "binary_logistic"
        else:
            loading, p_value, _ = _continuous_nested_test(y, base, assay.state)
            outcome_type = "continuous_linear"
        fits[node] = _FeatureStateFit(
            node=node,
            modality=modality,
            outcome_type=outcome_type,
            loading=loading,
            loading_norm=float(np.linalg.norm(loading)),
            p_value=p_value,
        )
    q_values = _bh_adjust([fit.p_value for fit in fits.values()])
    for fit, q_value in zip(fits.values(), q_values, strict=True):
        fit.q_value = float(q_value)
        fit.supported = bool(
            fit.q_value <= config.state_fdr
            and fit.loading_norm >= config.min_state_loading_norm
        )
    return fits


def _signed_cosine(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _state_candidate(
    record: Mapping[str, Any],
    fits: Mapping[str, _FeatureStateFit],
    config: UpperBoundConfig,
) -> tuple[str, str, float, bool]:
    left_nodes = _truth_nodes(record.get("node_set1"), str(record["node1"]))
    right_nodes = _truth_nodes(record.get("node_set2"), str(record["node2"]))
    scored = [
        (left, right, _signed_cosine(fits[left].loading, fits[right].loading))
        for left in left_nodes
        for right in right_nodes
        if left != right
    ]
    if not scored:
        raise UpperBoundError(f"Truth target has no candidate members: {record['label']}")
    state_supported = [
        item for item in scored if fits[item[0]].supported and fits[item[1]].supported
    ]
    left, right, cosine = max(
        state_supported or scored,
        key=lambda item: (item[2], item[0], item[1]),
    )
    supported = bool(
        fits[left].supported
        and fits[right].supported
        and cosine >= config.min_signed_loading_cosine
    )
    return left, right, cosine, supported


def _direct_test_rows(
    truth: pd.DataFrame,
    assays: Mapping[str, _Assay],
    node_to_modality: Mapping[str, str],
    config: UpperBoundConfig,
) -> list[dict[str, Any]]:
    direct = truth.loc[truth["truth_type"] == DIRECT]
    raw: list[dict[str, Any]] = []
    by_child: dict[str, list[str]] = {}
    for row in direct.to_dict(orient="records"):
        by_child.setdefault(str(row.get("child", row["node2"])), []).append(
            str(row.get("parent", row["node1"]))
        )
    for row in direct.to_dict(orient="records"):
        parent = str(row.get("parent", row["node1"]))
        child = str(row.get("child", row["node2"]))
        modality = node_to_modality[parent]
        if node_to_modality[child] != modality:
            raise UpperBoundError("Registered direct edge is not within one observed assay")
        assay = assays[modality]
        other_parents = [node for node in by_child[child] if node != parent]
        nuisance_parts = [assay.state, assay.depth_residual[:, None]]
        nuisance_parts.extend(
            _feature_values(node, assays, node_to_modality)[:, None]
            for node in other_parents
        )
        nuisance = np.column_stack(nuisance_parts)
        coefficient, p_value, statistic = _binary_nested_test(
            _feature_values(child, assays, node_to_modality),
            nuisance,
            _feature_values(parent, assays, node_to_modality),
            config,
        )
        raw.append(
            {
                **row,
                "candidate_node1": parent,
                "candidate_node2": child,
                "evidence_method": "oracle_state_conditional_logistic_lrt",
                "coefficient": float(coefficient[0]),
                "p_value": p_value,
                "test_statistic": statistic,
                "loading_cosine": float("nan"),
            }
        )
    q_values = _bh_adjust([row["p_value"] for row in raw])
    for row, q_value in zip(raw, q_values, strict=True):
        row["q_value"] = float(q_value)
        row["observation_supported"] = bool(
            q_value <= config.direct_fdr and row["coefficient"] > 0.0
        )
    return raw


def _null_test_rows(
    truth: pd.DataFrame,
    assays: Mapping[str, _Assay],
    node_to_modality: Mapping[str, str],
    fits: Mapping[str, _FeatureStateFit],
    config: UpperBoundConfig,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for row in truth.loc[truth["truth_type"] == SENTINEL_NULL].to_dict(orient="records"):
        left, right = str(row["node1"]), str(row["node2"])
        left_modality, right_modality = node_to_modality[left], node_to_modality[right]
        if left_modality == right_modality:
            assay = assays[left_modality]
            nuisance = np.column_stack([assay.state, assay.depth_residual])
            coefficient, p_value, statistic = _binary_nested_test(
                _feature_values(right, assays, node_to_modality),
                nuisance,
                _feature_values(left, assays, node_to_modality),
                config,
            )
            raw.append(
                {
                    **row,
                    "candidate_node1": left,
                    "candidate_node2": right,
                    "evidence_method": "oracle_state_conditional_null_lrt",
                    "coefficient": float(coefficient[0]),
                    "p_value": p_value,
                    "test_statistic": statistic,
                    "loading_cosine": float("nan"),
                    "cross_assay": False,
                }
            )
        else:
            cosine = _signed_cosine(fits[left].loading, fits[right].loading)
            raw.append(
                {
                    **row,
                    "candidate_node1": left,
                    "candidate_node2": right,
                    "evidence_method": "oracle_state_loading_null_match",
                    "coefficient": float("nan"),
                    "p_value": max(fits[left].q_value, fits[right].q_value),
                    "test_statistic": float("nan"),
                    "loading_cosine": cosine,
                    "cross_assay": True,
                }
            )
    within_indexes = [index for index, row in enumerate(raw) if not row["cross_assay"]]
    within_q = _bh_adjust([raw[index]["p_value"] for index in within_indexes])
    for index, q_value in zip(within_indexes, within_q, strict=True):
        raw[index]["q_value"] = float(q_value)
        raw[index]["observation_supported"] = bool(q_value <= config.null_fdr)
    for row in raw:
        if row["cross_assay"]:
            left, right = row["candidate_node1"], row["candidate_node2"]
            row["q_value"] = float(max(fits[left].q_value, fits[right].q_value))
            row["observation_supported"] = bool(
                fits[left].supported
                and fits[right].supported
                and row["loading_cosine"] >= config.min_signed_loading_cosine
            )
    return raw


def _edge_record(
    node1: str,
    node2: str,
    edge_type: str,
    *,
    child: str = "",
) -> dict[str, Any]:
    endpoint1 = "ARROW" if node1 == child else "CIRCLE"
    endpoint2 = "ARROW" if node2 == child else "CIRCLE"
    return {
        "node1": node1,
        "node2": node2,
        "endpoint1": endpoint1,
        "endpoint2": endpoint2,
        "edge_type": edge_type,
        "evidence_source": "scorer_only_oracle_state_upper_bound",
    }


def evaluate_world_upper_bound(
    world_dir: Path | str,
    config: UpperBoundConfig | None = None,
) -> WorldUpperBoundResult:
    """Evaluate one v4 world with truth-aware oracle-state targeted tests.

    The result must remain in a scorer/validation domain.  It is intentionally
    impossible to call this function without opening scorer-only inputs.
    """

    settings = config or UpperBoundConfig()
    world, truth, assays, node_to_modality, input_hashes = _load_world(world_dir)
    fits = _fit_state_models(assays, node_to_modality, settings)
    diagnostics: list[dict[str, Any]] = []
    diagnostics.extend(
        _direct_test_rows(truth, assays, node_to_modality, settings)
    )
    diagnostics.extend(
        _null_test_rows(truth, assays, node_to_modality, fits, settings)
    )
    for row in truth.loc[truth["truth_type"].isin(POSITIVE_STATE_TYPES)].to_dict(
        orient="records"
    ):
        left, right, cosine, supported = _state_candidate(row, fits, settings)
        diagnostics.append(
            {
                **row,
                "candidate_node1": left,
                "candidate_node2": right,
                "evidence_method": "oracle_state_group_loading_match",
                "coefficient": float("nan"),
                "p_value": max(fits[left].p_value, fits[right].p_value),
                "q_value": max(fits[left].q_value, fits[right].q_value),
                "test_statistic": float("nan"),
                "loading_cosine": cosine,
                "observation_supported": supported,
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostics)
    if diagnostic_frame["label"].astype(str).duplicated().any():
        raise UpperBoundError("Upper-bound diagnostics must have one row per targeted truth label")

    edge_rows: list[dict[str, Any]] = []
    for row in diagnostic_frame.loc[diagnostic_frame["observation_supported"]].to_dict(
        orient="records"
    ):
        truth_type = str(row["truth_type"])
        left, right = str(row["candidate_node1"]), str(row["candidate_node2"])
        if truth_type == DIRECT:
            edge_rows.append(
                _edge_record(
                    left,
                    right,
                    "within_assay_direct_candidate",
                    child=str(row.get("child", right)),
                )
            )
        elif truth_type == WITHIN_SHARED:
            edge_rows.append(_edge_record(left, right, "within_assay_shared_state"))
        elif truth_type == CROSS_SHARED:
            edge_rows.append(_edge_record(left, right, "cross_assay_shared_state"))
        elif truth_type == SLOW_GENE:
            edge_rows.append(
                _edge_record(left, right, "prior_constrained_peak_gene", child=right)
            )
        elif truth_type == SENTINEL_NULL:
            edge_rows.append(_edge_record(left, right, "unresolved"))
    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "node1",
            "node2",
            "endpoint1",
            "endpoint2",
            "edge_type",
            "evidence_source",
        ],
    )
    if not edges.empty:
        keys = edges.apply(
            lambda row: tuple(sorted((str(row["node1"]), str(row["node2"])))), axis=1
        )
        if keys.duplicated().any():
            raise UpperBoundError("Targeted upper-bound tests emitted duplicate adjacencies")
    metrics, scored_pairs = score_v4(edges, truth)
    metrics.update(
        {
            "n_sentinel_nulls": int((truth["truth_type"] == SENTINEL_NULL).sum()),
            "n_sentinel_false_adjacencies": int(
                diagnostic_frame.loc[
                    diagnostic_frame["truth_type"] == SENTINEL_NULL,
                    "observation_supported",
                ].sum()
            ),
            "all_positive_targets_observation_supported": bool(
                diagnostic_frame.loc[
                    diagnostic_frame["truth_type"].isin(
                        {DIRECT, WITHIN_SHARED, CROSS_SHARED, SLOW_GENE}
                    ),
                    "observation_supported",
                ].all()
            ),
            "upper_bound_not_discovery_performance": True,
        }
    )
    diagnostic_frame = diagnostic_frame.merge(
        scored_pairs[["label", "present", "adjacency_present", "wrong_layer"]],
        on="label",
        how="left",
        validate="one_to_one",
    )
    feature_rows = [
        {
            "node": fit.node,
            "modality": fit.modality,
            "outcome_type": fit.outcome_type,
            "loading_dim0": float(fit.loading[0]),
            "loading_dim1": float(fit.loading[1]),
            "loading_dim2": float(fit.loading[2]),
            "loading_norm": fit.loading_norm,
            "state_p_value": fit.p_value,
            "state_q_value": fit.q_value,
            "state_supported": fit.supported,
        }
        for fit in fits.values()
    ]
    world_seed: int | None = None
    world_design: dict[str, Any] | None = None
    parameters_path = world / "scorer" / "world_parameters.json"
    if parameters_path.is_file() and not parameters_path.is_symlink():
        parameters = json.loads(parameters_path.read_text())
        if "seed" in parameters:
            world_seed = int(parameters["seed"])
        if isinstance(parameters.get("design"), dict):
            world_design = dict(parameters["design"])
        input_hashes["scorer/world_parameters.json"] = _sha256(parameters_path)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "information_domain": "scorer_only_validation_ceiling",
        "discovery_eligible": False,
        "must_not_be_consumed_by_discovery": True,
        "truth_used_for_target_definition": True,
        "oracle_latent_state_used": True,
        "observed_focal_measurements_used": True,
        "cross_assay_cell_rows_paired": False,
        "cross_assay_privilege": "common_true_latent_coordinate_system_only",
        "truth_graph_copied_without_observation_test": False,
        "world_design": world_design,
        "config": asdict(settings),
        "config_sha256": hashlib.sha256(_canonical_json_bytes(asdict(settings))).hexdigest(),
        "input_hashes": dict(sorted(input_hashes.items())),
    }
    return WorldUpperBoundResult(
        world_seed=world_seed,
        metrics=metrics,
        edges=edges,
        diagnostics=diagnostic_frame,
        feature_state_fits=pd.DataFrame(feature_rows),
        provenance=provenance,
    )


def write_world_upper_bound(
    result: WorldUpperBoundResult,
    out_dir: Path | str,
    *,
    world_dir: Path | str,
) -> dict[str, Any]:
    """Write one explicitly tainted scorer-only validation artifact bundle."""

    world, _, _ = _validate_world_root(world_dir)
    output = _validate_output(out_dir, world)
    output.mkdir(parents=True, exist_ok=True)
    result.edges.to_csv(output / "oracle_upper_bound_edges.tsv", sep="\t", index=False)
    result.diagnostics.to_csv(
        output / "oracle_upper_bound_target_diagnostics.tsv", sep="\t", index=False
    )
    result.feature_state_fits.to_csv(
        output / "oracle_upper_bound_feature_state_fits.tsv", sep="\t", index=False
    )
    _write_json(output / "oracle_upper_bound_metrics.json", result.metrics)
    manifest = {
        **result.provenance,
        "world_seed": result.world_seed,
        "world_success": bool(result.metrics["world_success"]),
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.name == "oracle_upper_bound_manifest.json":
            continue
        manifest["artifacts"][path.name] = _sha256(path)
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json(output / "oracle_upper_bound_manifest.json", manifest)
    return manifest


def _wilson_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 1.0
    # One-sided 95% Wilson bound; 1.64485 is Phi^-1(0.95).
    z = 1.6448536269514722 if confidence == 0.95 else 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float(min(1.0, (center + radius) / denominator))


def aggregate_upper_bound_results(
    results: Sequence[WorldUpperBoundResult],
    *,
    required_world_successes: int,
    required_class_successes: int,
    required_factor_level_successes: int | None = None,
    required_design_cell_successes: int | None = None,
) -> dict[str, Any]:
    """Apply the preregistered world/class/null/density ceiling gates."""

    if not results:
        raise UpperBoundError("At least one upper-bound world result is required")
    if not 1 <= required_world_successes <= len(results):
        raise UpperBoundError("required_world_successes is inconsistent with result count")
    if not 1 <= required_class_successes <= len(results):
        raise UpperBoundError("required_class_successes is inconsistent with result count")
    if (required_factor_level_successes is None) != (
        required_design_cell_successes is None
    ):
        raise UpperBoundError(
            "Factor-level and design-cell requirements must be supplied together"
        )
    rows = pd.DataFrame([result.metrics for result in results])
    false_nulls = int(rows["n_sentinel_false_adjacencies"].sum())
    total_nulls = int(rows["n_sentinel_nulls"].sum())
    null_upper = _wilson_upper(false_nulls, total_nulls)
    class_gates = {
        column: {
            "successes": int(pd.to_numeric(rows[column]).sum()),
            "required": required_class_successes,
            "pass": int(pd.to_numeric(rows[column]).sum()) >= required_class_successes,
        }
        for column in CLASS_COLUMNS
    }
    world_successes = int(pd.to_numeric(rows["world_success"]).sum())
    mean_density = float(pd.to_numeric(rows["n_estimated_focal_edges"]).mean())
    gates: dict[str, Any] = {
        "world_success": {
            "successes": world_successes,
            "required": required_world_successes,
            "pass": world_successes >= required_world_successes,
        },
        "classes": class_gates,
        "pooled_sentinel_null": {
            "false_adjacencies": false_nulls,
            "events": total_nulls,
            "wilson_one_sided_upper_95": null_upper,
            "maximum": TARGET_NULL_FPR,
            "pass": null_upper <= TARGET_NULL_FPR,
        },
        "density": {
            "mean_focal_edges": mean_density,
            "strictly_below_v3": V3_MEAN_BIOLOGICAL_EDGES,
            "all_worlds_at_most_30": bool(
                (pd.to_numeric(rows["n_estimated_focal_edges"]) <= MAX_FOCAL_EDGES_PER_WORLD).all()
            ),
            "pass": bool(
                mean_density < V3_MEAN_BIOLOGICAL_EDGES
                and (
                    pd.to_numeric(rows["n_estimated_focal_edges"])
                    <= MAX_FOCAL_EDGES_PER_WORLD
                ).all()
            ),
        },
    }
    if required_factor_level_successes is None:
        design_gates: dict[str, Any] = {
            "evaluated": False,
            "reason": "no registered split design requirements supplied",
            "pass": True,
        }
    else:
        assert required_factor_level_successes is not None
        assert required_design_cell_successes is not None
        factor_requirement = int(required_factor_level_successes)
        cell_requirement = int(required_design_cell_successes)
        design_rows: list[dict[str, Any]] = []
        for result in results:
            design = result.provenance.get("world_design")
            if not isinstance(design, Mapping):
                raise UpperBoundError(
                    "Registered split aggregation requires scorer world_design metadata"
                )
            values: dict[str, int] = {}
            for factor in FACTOR_COLUMNS:
                try:
                    value = int(design[factor])
                except (KeyError, TypeError, ValueError) as exc:
                    raise UpperBoundError(
                        f"World design lacks coded factor {factor}"
                    ) from exc
                if value not in {-1, 1}:
                    raise UpperBoundError(f"World design factor {factor} is not coded +/-1")
                values[factor] = value
            if values["D_background_observability"] != (
                values["A_cells"]
                * values["B_detection"]
                * values["C_population_divergence"]
            ):
                raise UpperBoundError("World design violates the registered D=A*B*C generator")
            design_rows.append(
                {
                    **values,
                    "world_success": int(bool(result.metrics["world_success"])),
                }
            )
        design_frame = pd.DataFrame(design_rows)
        factor_gates: dict[str, Any] = {}
        for factor in FACTOR_COLUMNS:
            for level in (-1, 1):
                subset = design_frame.loc[design_frame[factor] == level]
                successes = int(subset["world_success"].sum())
                key = f"{factor}:{level:+d}"
                factor_gates[key] = {
                    "worlds": int(len(subset)),
                    "successes": successes,
                    "required": factor_requirement,
                    "pass": successes >= factor_requirement,
                }
        cell_gates: dict[str, Any] = {}
        grouped = design_frame.groupby(
            ["A_cells", "B_detection", "C_population_divergence"], sort=True
        )
        for (a_value, b_value, c_value), subset in grouped:
            successes = int(subset["world_success"].sum())
            key = f"A={int(a_value):+d},B={int(b_value):+d},C={int(c_value):+d}"
            cell_gates[key] = {
                "worlds": int(len(subset)),
                "successes": successes,
                "required": cell_requirement,
                "pass": successes >= cell_requirement,
            }
        design_gates = {
            "evaluated": True,
            "factor_levels": factor_gates,
            "design_cells": cell_gates,
            "pass": bool(
                len(factor_gates) == 2 * len(FACTOR_COLUMNS)
                and len(cell_gates) == 8
                and all(value["pass"] for value in factor_gates.values())
                and all(value["pass"] for value in cell_gates.values())
            ),
        }
    gates["resolution_iv_design"] = design_gates
    passed = bool(
        gates["world_success"]["pass"]
        and all(value["pass"] for value in class_gates.values())
        and gates["pooled_sentinel_null"]["pass"]
        and gates["density"]["pass"]
        and design_gates["pass"]
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}.aggregate",
        "information_domain": "scorer_only_validation_ceiling",
        "discovery_eligible": False,
        "n_worlds": len(results),
        "gates": gates,
        "pass": passed,
    }


def aggregate_registered_upper_bound(
    results: Sequence[WorldUpperBoundResult],
    *,
    split: str,
) -> dict[str, Any]:
    """Apply the exact preregistered DEV or fresh-LOCKED ceiling contract."""

    if split not in REGISTERED_SPLITS:
        raise UpperBoundError("Registered upper-bound split must be development or locked")
    specification = REGISTERED_SPLITS[split]
    expected_seeds = tuple(int(seed) for seed in specification["seeds"])
    observed_seeds = tuple(
        sorted(int(result.world_seed) for result in results if result.world_seed is not None)
    )
    if len(observed_seeds) != len(results) or observed_seeds != expected_seeds:
        raise UpperBoundError(
            f"{split} upper bound requires the exact registered seed block"
        )
    aggregate = aggregate_upper_bound_results(
        results,
        required_world_successes=int(specification["required_world_successes"]),
        required_class_successes=int(specification["required_class_successes"]),
        required_factor_level_successes=int(
            specification["required_factor_level_successes"]
        ),
        required_design_cell_successes=int(
            specification["required_design_cell_successes"]
        ),
    )
    aggregate["registered_split"] = split
    aggregate["registered_seeds"] = list(expected_seeds)
    aggregate["exact_seed_block"] = True
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_world_upper_bound(args.world)
    manifest = write_world_upper_bound(result, args.out_dir, world_dir=args.world)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "world_seed": result.world_seed,
                "world_success": bool(result.metrics["world_success"]),
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "warning": "SCORER-ONLY: never use these artifacts for discovery",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
