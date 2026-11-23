#!/usr/bin/env python3
"""Freeze a CA-BOSS v4 configuration from training worlds only.

The calibrator deliberately runs after discovery.  It first reads cached pair
evidence produced without truth, verifies that each residual null was generated
by state/depth-stratified permutation, and only then opens scorer-only sentinel
labels and post-consensus benchmark scores.  Nested aggregation runs and
multiple sentinel pairs are never treated as independent worlds.

Selection is lexicographic:

1. require a world-cluster-bootstrap/Wilson upper bound on sentinel adjacency
   frequency at or below the requested false-positive target;
2. maximize the world-cluster-bootstrap lower bound on typed motif recall;
3. break ties by macro recall, worst-class recall, precision, and graph sparsity.

Only the preregistered training seeds 30001--30024 are accepted.  Historical
and fresh locked blocks, development worlds, and paths containing ``locked``
are rejected before any raw evidence or truth file is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark_ca_boss_v4 import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    DISCOVERY_MANIFEST_SCHEMA_VERSION,
    RUN_SUMMARY_SCHEMA_VERSION,
    SCORER_MANIFEST_SCHEMA_VERSION,
    WORLD_INPUT_BINDING_SCHEMA_VERSION,
    validate_world_input_binding,
)


TRAINING_SEED_MIN = 30001
TRAINING_SEED_MAX = 30024
DEFAULT_EXPECTED_WORLDS = 24
DEFAULT_EXPECTED_NESTED_RUNS = 6
SENTINEL_TRUTH_TYPE = "sentinel_null"
TYPED_RECALL_COLUMNS = (
    "direct_adjacency_macro_recall",
    "collider_arrowhead_recall",
    "shared_state_adjacency_recall",
    "slow_peak_gene_present",
)
RESULT_COLUMNS = (
    *TYPED_RECALL_COLUMNS,
    "matched_null_false_positive_rate",
    "primary_skeleton_precision",
    "primary_skeleton_f1",
    "n_estimated_focal_edges",
    "world_success",
)
RAW_NULL_COLUMNS = (
    "node1",
    "node2",
    "residual_r",
    "residual_empirical_p",
    "residual_empirical_q",
    "residual_null_abs_r_95",
    "residual_null_abs_r_mean",
    "state_score",
    "predicted_r",
    "direct_bootstrap_support",
    "state_bootstrap_support",
)
_SEED_PATTERN = re.compile(r"(?:^|[^a-z0-9])(?:world|seed)[_-]?(\d{4,8})(?:$|[^0-9])", re.I)
TRAINING_BINDING_AUDIT_SCHEMA_VERSION = "ca_boss_v4.training_world_bindings.v1"
CALIBRATION_AUDIT_SCHEMA_VERSION = "ca_boss_v4.calibration.v2"


class CalibrationError(ValueError):
    """Raised when calibration provenance or statistical requirements fail."""


@dataclass(frozen=True)
class CalibrationSpec:
    """Predeclared selection and uncertainty settings."""

    expected_worlds: int = DEFAULT_EXPECTED_WORLDS
    expected_nested_runs: int = DEFAULT_EXPECTED_NESTED_RUNS
    target_null_fpr: float = 0.10
    confidence: float = 0.95
    bootstrap_replicates: int = 10_000
    random_seed: int = 41_004
    simultaneous_candidate_control: bool = True

    def __post_init__(self) -> None:
        if self.expected_worlds < 2:
            raise ValueError("expected_worlds must be at least two")
        if self.expected_nested_runs < 1:
            raise ValueError("expected_nested_runs must be positive")
        if not 0.0 < self.target_null_fpr < 1.0:
            raise ValueError("target_null_fpr must lie strictly between zero and one")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0.5 and one")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")


@dataclass
class CalibrationArtifacts:
    """In-memory result plus paths written by :func:`calibrate_benchmark`."""

    selected_candidate: str
    frozen_config: dict[str, Any]
    ranking: pd.DataFrame
    null_world_summary: pd.DataFrame
    audit_manifest: dict[str, Any]
    config_path: Path
    ranking_path: Path
    null_summary_path: Path
    audit_path: Path


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_not_locked_path(path: Path | str) -> None:
    if "locked" in str(path).lower():
        raise CalibrationError(f"Locked path is forbidden during calibration: {path}")


def world_seed(value: Path | str) -> int:
    """Extract a marked ``world``/``seed`` identifier from a path or label."""

    text = str(value)
    _assert_not_locked_path(text)
    matches = [int(match) for match in _SEED_PATTERN.findall(text)]
    registered = sorted(
        {
            seed
            for seed in matches
            if TRAINING_SEED_MIN <= seed <= TRAINING_SEED_MAX
        }
    )
    nonregistered = sorted({seed for seed in matches if seed not in registered})
    if nonregistered:
        raise CalibrationError(
            f"Non-training world seed(s) found in calibration input {value!s}: {nonregistered}"
        )
    if len(registered) != 1:
        raise CalibrationError(
            f"Expected one preregistered training seed in {value!s}; found {registered}"
        )
    return registered[0]


def _canonical_pair(left: Any, right: Any) -> tuple[str, str]:
    left_text, right_text = str(left), str(right)
    return (
        (left_text, right_text)
        if left_text < right_text
        else (right_text, left_text)
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CalibrationError(f"Expected a JSON object in {path}")
    return value


def _candidate_configs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise CalibrationError("Benchmark manifest has no candidate configurations")
    configs: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
            raise CalibrationError("Malformed candidate entry in benchmark manifest")
        name = str(entry.get("name", ""))
        if not name or name in configs:
            raise CalibrationError(f"Missing or duplicate candidate name: {name!r}")
        configs[name] = dict(entry["config"])
    return configs


def _validate_manifest_worlds(
    manifest: Mapping[str, Any], spec: CalibrationSpec
) -> tuple[dict[int, Path], dict[int, dict[str, Any]]]:
    if manifest.get("schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise CalibrationError("Benchmark has a stale or unsupported manifest schema")
    if manifest.get("truth_opened_after_consensus_only") is not True:
        raise CalibrationError(
            "Benchmark did not attest that truth was opened only after consensus"
        )
    entries = manifest.get("worlds")
    if not isinstance(entries, list) or not entries:
        raise CalibrationError("Benchmark manifest has no worlds")
    sources: dict[int, Path] = {}
    for value in entries:
        source = Path(str(value))
        seed = world_seed(source)
        if seed in sources:
            raise CalibrationError(f"Duplicate training world seed in manifest: {seed}")
        sources[seed] = source
    expected = set(range(TRAINING_SEED_MIN, TRAINING_SEED_MIN + spec.expected_worlds))
    if set(sources) != expected:
        missing = sorted(expected - set(sources))
        unexpected = sorted(set(sources) - expected)
        raise CalibrationError(
            "Training worlds do not match the preregistered prefix; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if manifest.get("all_world_inputs_content_addressed_and_verified") is not True:
        raise CalibrationError("Benchmark did not verify every content-addressed world")
    if (
        manifest.get("world_input_binding_schema_version")
        != WORLD_INPUT_BINDING_SCHEMA_VERSION
    ):
        raise CalibrationError("Benchmark has a stale world-input binding schema")
    binding_entries = manifest.get("world_input_bindings")
    if not isinstance(binding_entries, list) or len(binding_entries) != len(sources):
        raise CalibrationError("Benchmark lacks one world-input binding per world")
    if manifest.get("world_input_bindings_canonical_sha256") != canonical_sha256(
        binding_entries
    ):
        raise CalibrationError("Benchmark world-input binding collection hash mismatch")

    recorded: dict[int, dict[str, Any]] = {}
    for value in binding_entries:
        if not isinstance(value, dict):
            raise CalibrationError("Malformed world-input binding in benchmark manifest")
        payload = dict(value)
        claimed_payload_hash = str(payload.pop("binding_payload_sha256", ""))
        if claimed_payload_hash != canonical_sha256(payload):
            raise CalibrationError("Benchmark world-input binding payload hash mismatch")
        if value.get("schema_version") != WORLD_INPUT_BINDING_SCHEMA_VERSION:
            raise CalibrationError("Benchmark contains a stale world-input binding")
        try:
            seed = int(value.get("seed", -1))
        except (TypeError, ValueError) as error:
            raise CalibrationError("World-input binding has an invalid seed") from error
        if seed not in sources or seed in recorded:
            raise CalibrationError(f"Unexpected or duplicate world-input binding seed: {seed}")
        if Path(str(value.get("world", ""))).absolute() != sources[seed].absolute():
            raise CalibrationError(f"World-input binding path mismatch for seed {seed}")
        for domain, expected_schema in (
            ("run_summary", RUN_SUMMARY_SCHEMA_VERSION),
            ("discovery_manifest", DISCOVERY_MANIFEST_SCHEMA_VERSION),
            ("scorer_manifest", SCORER_MANIFEST_SCHEMA_VERSION),
        ):
            record = value.get(domain)
            if not isinstance(record, dict):
                raise CalibrationError(
                    f"World-input binding lacks {domain} for seed {seed}"
                )
            digest = str(record.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise CalibrationError(
                    f"World-input binding has an invalid {domain} hash for seed {seed}"
                )
            if record.get("schema_version") != expected_schema:
                raise CalibrationError(
                    f"World-input binding has a stale {domain} schema for seed {seed}"
                )
            if domain != "run_summary" and record.get(
                "payloads_content_addressed_verified"
            ) is not True:
                raise CalibrationError(
                    f"World-input binding did not verify {domain} payloads for seed {seed}"
                )
        recorded[seed] = dict(value)
    if set(recorded) != set(sources):
        raise CalibrationError("World-input bindings do not cover the training worlds")

    # Preserve the information boundary: verify only run/discovery bytes here.
    # The current scorer bytes are not opened until cached discovery evidence
    # has been loaded below.
    for seed, source in sorted(sources.items()):
        try:
            observed = validate_world_input_binding(
                source,
                verify_scorer_payload=False,
            )
        except ValueError as error:
            raise CalibrationError(
                f"Training world {seed} failed its discovery binding: {error}"
            ) from error
        expected_binding = recorded[seed]
        for domain in ("run_summary", "discovery_manifest", "scorer_manifest"):
            if (
                observed[domain]["sha256"]
                != expected_binding[domain]["sha256"]
            ):
                raise CalibrationError(
                    f"Training world {seed} {domain} changed after benchmarking"
                )
    return sources, recorded


def _verify_full_world_bindings_after_discovery(
    sources: Mapping[int, Path],
    recorded: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Verify scorer payloads only after discovery evidence is in memory."""

    verified: list[dict[str, Any]] = []
    for seed, source in sorted(sources.items()):
        try:
            observed = validate_world_input_binding(
                source,
                verify_scorer_payload=True,
            )
        except ValueError as error:
            raise CalibrationError(
                f"Training world {seed} failed its full content binding: {error}"
            ) from error
        if observed != dict(recorded[seed]):
            raise CalibrationError(
                f"Training world {seed} content binding changed after benchmarking"
            )
        verified.append(observed)
    return verified


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise CalibrationError(f"Non-finite {column!r} values in {source}")
        frame[column] = converted


def _validate_discovery_summary(evidence_path: Path) -> Path:
    summary_path = evidence_path.with_name("discovery_summary.json")
    if not summary_path.exists():
        raise CalibrationError(
            f"Missing discovery_summary.json beside raw evidence {evidence_path}"
        )
    summary = _read_json(summary_path)
    config = summary.get("config")
    preprocessing = summary.get("preprocessing")
    if summary.get("method") != "ca_boss_v4_explicit_state":
        raise CalibrationError(f"Unexpected discovery method in {summary_path}")
    if not isinstance(config, dict) or not isinstance(preprocessing, dict):
        raise CalibrationError(f"Incomplete permutation provenance in {summary_path}")
    if int(config.get("residual_permutations", 0)) <= 0:
        raise CalibrationError(f"Residual permutations were disabled in {summary_path}")
    if int(config.get("null_strata", 0)) < 2:
        raise CalibrationError(f"State/depth null stratification was disabled in {summary_path}")
    if int(preprocessing.get("n_null_strata", 0)) < 2:
        raise CalibrationError(
            f"Fewer than two realized null strata were recorded in {summary_path}"
        )
    if preprocessing.get("focal_features_excluded_from_state_estimation") is not True:
        raise CalibrationError(
            f"Focal-feature exclusion was not attested in {summary_path}"
        )
    return summary_path


def load_raw_evidence_first(
    benchmark_dir: Path,
    expected_seeds: set[int],
    spec: CalibrationSpec,
) -> tuple[pd.DataFrame, list[Path], list[Path]]:
    """Load discovery-only evidence before any scorer artifact is opened."""

    paths = sorted(benchmark_dir.glob("**/raw_pair_evidence.tsv"))
    if not paths:
        raise CalibrationError(f"No raw_pair_evidence.tsv files below {benchmark_dir}")
    frames: list[pd.DataFrame] = []
    summary_paths: list[Path] = []
    per_world: dict[int, list[Path]] = {seed: [] for seed in expected_seeds}
    for path in paths:
        _assert_not_locked_path(path)
        resolved = path.resolve()
        _assert_not_locked_path(resolved)
        lexical_seed = world_seed(path)
        resolved_seed = world_seed(resolved)
        if lexical_seed != resolved_seed:
            raise CalibrationError(
                f"Raw-evidence symlink crosses world identities: {path} -> {resolved}"
            )
        seed = lexical_seed
        if seed not in expected_seeds:
            raise CalibrationError(f"Unexpected raw-evidence world {seed}: {path}")
        summary_paths.append(_validate_discovery_summary(path))
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        missing = set(RAW_NULL_COLUMNS) - set(frame.columns)
        if missing:
            raise CalibrationError(f"Raw evidence {path} lacks {sorted(missing)}")
        _numeric_frame(frame, RAW_NULL_COLUMNS[2:], path)
        frame["_world_seed"] = seed
        frame["_nested_run"] = str(path.parent.relative_to(benchmark_dir))
        frame["_source_file"] = str(path)
        frames.append(frame)
        per_world[seed].append(path)
    for seed, world_paths in sorted(per_world.items()):
        if len(world_paths) != spec.expected_nested_runs:
            raise CalibrationError(
                f"Training world {seed} has {len(world_paths)} raw-evidence runs; "
                f"expected {spec.expected_nested_runs}"
            )
    return pd.concat(frames, ignore_index=True), paths, summary_paths


def _truth_path(source: Path) -> Path:
    candidates = (
        source / "scorer" / "truth_typed_edges.tsv",
        source / "truth_typed_edges.tsv",
        source.parent / "scorer" / "truth_typed_edges.tsv",
    )
    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists():
            seen.add(resolved)
            existing.append(candidate)
    if not existing:
        raise CalibrationError(f"No scorer-only truth table found for {source}")
    scorer_paths = [path for path in existing if path.parent.name == "scorer"]
    return scorer_paths[0] if scorer_paths else existing[0]


def load_truth_after_discovery(
    sources: Mapping[int, Path],
) -> tuple[pd.DataFrame, list[Path]]:
    """Open scorer-only truth after cached discovery evidence is complete."""

    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    required = {"node1", "node2", "truth_type", "label"}
    for seed, source in sorted(sources.items()):
        path = _truth_path(source)
        _assert_not_locked_path(path)
        resolved = path.resolve()
        _assert_not_locked_path(resolved)
        if world_seed(path) != seed or world_seed(resolved) != seed:
            raise CalibrationError(f"Truth path does not match world {seed}: {path}")
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        missing = required - set(frame.columns)
        if missing:
            raise CalibrationError(f"Truth {path} lacks {sorted(missing)}")
        sentinels = frame.loc[frame["truth_type"].eq(SENTINEL_TRUTH_TYPE)]
        if sentinels.empty:
            raise CalibrationError(f"Truth {path} has no sentinel-null pairs")
        if sentinels["label"].astype(str).duplicated().any():
            raise CalibrationError(f"Truth {path} has duplicate sentinel labels")
        frame["_world_seed"] = seed
        frames.append(frame)
        paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def cluster_equal_weighted_quantile(
    values: Sequence[float], clusters: Sequence[Any], quantile: float
) -> float:
    """Quantile where each biological world has total weight one."""

    array = np.asarray(values, dtype=float)
    labels = np.asarray(clusters)
    if array.ndim != 1 or len(array) != len(labels) or len(array) == 0:
        raise ValueError("values and clusters must be non-empty one-dimensional arrays")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    unique, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1.0 / count_map[label] for label in labels], dtype=float)
    weights /= weights.sum()
    order = np.argsort(array, kind="mergesort")
    ordered_values, ordered_weights = array[order], weights[order]
    cumulative = np.cumsum(ordered_weights)
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(array) - 1)
    return float(ordered_values[index])


def _cluster_bootstrap_ratio(
    successes: Sequence[float],
    totals: Sequence[float],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    success = np.asarray(successes, dtype=float)
    total = np.asarray(totals, dtype=float)
    if len(success) < 2 or success.shape != total.shape or np.any(total <= 0):
        raise ValueError("Cluster bootstrap needs at least two positive world denominators")
    indices = rng.integers(0, len(success), size=(replicates, len(success)))
    numerator = success[indices].sum(axis=1)
    denominator = total[indices].sum(axis=1)
    return numerator / denominator


def _cluster_bootstrap_mean(
    values: Sequence[float],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Cluster bootstrap needs at least two finite worlds")
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    return array[indices].mean(axis=1)


def wilson_upper(successes: int, total: int, *, alpha: float) -> float:
    """One-sided Wilson upper confidence bound for a binomial proportion."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson inputs require 0 <= successes <= total and total > 0")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between zero and 0.5")
    z = NormalDist().inv_cdf(1.0 - alpha)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float(min(1.0, (center + radius) / denominator))


def sentinel_empirical_null(
    raw: pd.DataFrame,
    truth: pd.DataFrame,
    spec: CalibrationSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize detection-matched sentinel and conditional-permutation nulls."""

    sentinels = truth.loc[truth["truth_type"].eq(SENTINEL_TRUTH_TYPE)].copy()
    sentinel_pairs = sentinels.apply(
        lambda row: _canonical_pair(row["node1"], row["node2"]), axis=1
    )
    sentinels["_pair_left"] = [pair[0] for pair in sentinel_pairs]
    sentinels["_pair_right"] = [pair[1] for pair in sentinel_pairs]
    if sentinels.duplicated(["_world_seed", "_pair_left", "_pair_right"]).any():
        raise CalibrationError("A world contains duplicate sentinel-null pairs")

    evidence = raw.copy()
    evidence_pairs = evidence.apply(
        lambda row: _canonical_pair(row["node1"], row["node2"]), axis=1
    )
    evidence["_pair_left"] = [pair[0] for pair in evidence_pairs]
    evidence["_pair_right"] = [pair[1] for pair in evidence_pairs]
    matched = evidence.merge(
        sentinels[["_world_seed", "_pair_left", "_pair_right", "label"]],
        on=["_world_seed", "_pair_left", "_pair_right"],
        how="inner",
        validate="many_to_one",
    )
    expected_occurrences = len(sentinels) * spec.expected_nested_runs
    if len(matched) != expected_occurrences:
        raise CalibrationError(
            "Sentinel raw-evidence coverage is incomplete: "
            f"found {len(matched)}, expected {expected_occurrences}"
        )
    matched["abs_residual_r"] = matched["residual_r"].abs()
    denominator = matched["residual_null_abs_r_95"].clip(lower=1e-12)
    matched["residual_to_pair_null95"] = matched["abs_residual_r"] / denominator
    matched["exceeds_pair_null95"] = (
        matched["abs_residual_r"] >= matched["residual_null_abs_r_95"]
    ).astype(int)
    matched["permutation_p_le_005"] = (
        matched["residual_empirical_p"] <= 0.05
    ).astype(int)

    world_summary = (
        matched.groupby("_world_seed", as_index=False)
        .agg(
            n_sentinel_pair_runs=("label", "size"),
            n_sentinel_pairs=("label", "nunique"),
            exceeds_pair_null95_count=("exceeds_pair_null95", "sum"),
            exceeds_pair_null95_rate=("exceeds_pair_null95", "mean"),
            permutation_p_le_005_count=("permutation_p_le_005", "sum"),
            permutation_p_le_005_rate=("permutation_p_le_005", "mean"),
            mean_pair_null_abs_r_95=("residual_null_abs_r_95", "mean"),
            mean_pair_null_abs_r=("residual_null_abs_r_mean", "mean"),
            max_abs_residual_r=("abs_residual_r", "max"),
            max_state_score=("state_score", "max"),
        )
        .sort_values("_world_seed")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(spec.random_seed + 7_919)
    bootstrap = _cluster_bootstrap_ratio(
        world_summary["exceeds_pair_null95_count"],
        world_summary["n_sentinel_pair_runs"],
        replicates=spec.bootstrap_replicates,
        rng=rng,
    )
    upper = float(np.quantile(bootstrap, spec.confidence))
    clusters = matched["_world_seed"].to_numpy()
    target_quantile = 1.0 - spec.target_null_fpr
    summary: dict[str, Any] = {
        "n_worlds": int(matched["_world_seed"].nunique()),
        "n_unique_sentinel_pairs": int(
            matched[["_world_seed", "label"]].drop_duplicates().shape[0]
        ),
        "n_nested_sentinel_observations": int(len(matched)),
        "pair_null95_exceedance_rate": float(matched["exceeds_pair_null95"].mean()),
        "pair_null95_exceedance_cluster_bootstrap_upper": upper,
        "permutation_p_le_005_rate": float(matched["permutation_p_le_005"].mean()),
        "cluster_equal_threshold_estimates": {
            "abs_residual_r_upper": cluster_equal_weighted_quantile(
                matched["abs_residual_r"], clusters, target_quantile
            ),
            "residual_to_pair_null95_upper": cluster_equal_weighted_quantile(
                matched["residual_to_pair_null95"], clusters, target_quantile
            ),
            "state_score_upper": cluster_equal_weighted_quantile(
                matched["state_score"], clusters, target_quantile
            ),
            "residual_empirical_p_lower": cluster_equal_weighted_quantile(
                matched["residual_empirical_p"], clusters, spec.target_null_fpr
            ),
            "residual_empirical_q_lower": cluster_equal_weighted_quantile(
                matched["residual_empirical_q"], clusters, spec.target_null_fpr
            ),
            "pairwise_permutation_null_abs_r_95_median": cluster_equal_weighted_quantile(
                matched["residual_null_abs_r_95"], clusters, 0.50
            ),
        },
        "permutation_contract": {
            "stratification": "background-state and leave-focal-out depth strata",
            "summary_columns": [
                "residual_empirical_p",
                "residual_null_abs_r_95",
                "residual_null_abs_r_mean",
            ],
            "focal_truth_used_to_generate_permutations": False,
        },
    }
    return summary, world_summary


def _load_scored_results(
    benchmark_dir: Path,
    expected_seeds: set[int],
    candidates: set[str],
) -> tuple[pd.DataFrame, Path]:
    path = benchmark_dir / "world_results.tsv"
    if not path.exists():
        raise CalibrationError(f"Missing post-consensus benchmark scores: {path}")
    results = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"world", "candidate", *RESULT_COLUMNS}
    missing = required - set(results.columns)
    if missing:
        raise CalibrationError(f"World results lack {sorted(missing)}")
    results["_world_seed"] = results["world"].map(world_seed)
    if set(results["_world_seed"]) != expected_seeds:
        raise CalibrationError("Scored result worlds differ from the training manifest")
    observed_candidates = set(results["candidate"].astype(str))
    if observed_candidates != candidates:
        raise CalibrationError(
            "Candidate set differs between scores and manifest: "
            f"only_scores={sorted(observed_candidates - candidates)}, "
            f"only_manifest={sorted(candidates - observed_candidates)}"
        )
    if results.duplicated(["candidate", "_world_seed"]).any():
        raise CalibrationError("World results contain duplicate candidate/world rows")
    expected_rows = len(expected_seeds) * len(candidates)
    if len(results) != expected_rows:
        raise CalibrationError(f"Expected {expected_rows} scored rows, found {len(results)}")
    _numeric_frame(results, RESULT_COLUMNS, path)
    return results, path


def _candidate_null_counts(
    results: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    sentinels = truth.loc[truth["truth_type"].eq(SENTINEL_TRUTH_TYPE)]
    labels_by_world = {
        int(seed): group["label"].astype(str).tolist()
        for seed, group in sentinels.groupby("_world_seed")
    }
    rows: list[dict[str, Any]] = []
    for record in results.to_dict(orient="records"):
        seed = int(record["_world_seed"])
        labels = labels_by_world[seed]
        values: list[int] = []
        for label in labels:
            column = f"present__{label}"
            if column not in results.columns:
                raise CalibrationError(f"Missing scored sentinel column {column!r}")
            value = record.get(column)
            if value == "" or pd.isna(value) or float(value) not in (0.0, 1.0):
                raise CalibrationError(
                    f"Invalid scored sentinel event for candidate={record['candidate']}, "
                    f"world={seed}, label={label}: {value!r}"
                )
            values.append(int(float(value)))
        rate = sum(values) / len(values)
        reported = float(record["matched_null_false_positive_rate"])
        if not math.isclose(rate, reported, abs_tol=1e-10):
            raise CalibrationError(
                f"Sentinel events disagree with reported FPR for {record['candidate']} "
                f"in world {seed}: events={rate}, metric={reported}"
            )
        rows.append(
            {
                "candidate": str(record["candidate"]),
                "_world_seed": seed,
                "null_false_adjacencies": int(sum(values)),
                "null_adjacency_opportunities": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _candidate_rng(seed: int, candidate: str, stream: str) -> np.random.Generator:
    token = hashlib.sha256(f"{candidate}|{stream}".encode("utf-8")).digest()
    offset = int.from_bytes(token[:8], byteorder="big", signed=False)
    return np.random.default_rng(np.random.SeedSequence([seed, offset]))


def rank_candidates(
    results: pd.DataFrame,
    null_counts: pd.DataFrame,
    configs: Mapping[str, Mapping[str, Any]],
    spec: CalibrationSpec,
) -> pd.DataFrame:
    """Rank configurations after conservative, world-clustered null control."""

    n_candidates = len(configs)
    family_alpha = 1.0 - spec.confidence
    alpha = (
        family_alpha / n_candidates
        if spec.simultaneous_candidate_control
        else family_alpha
    )
    rows: list[dict[str, Any]] = []
    for candidate in sorted(configs):
        candidate_results = (
            results.loc[results["candidate"].astype(str).eq(candidate)]
            .sort_values("_world_seed")
            .reset_index(drop=True)
        )
        candidate_nulls = (
            null_counts.loc[null_counts["candidate"].eq(candidate)]
            .sort_values("_world_seed")
            .reset_index(drop=True)
        )
        if not np.array_equal(
            candidate_results["_world_seed"], candidate_nulls["_world_seed"]
        ):
            raise CalibrationError(f"Null counts do not align for candidate {candidate}")
        successes = candidate_nulls["null_false_adjacencies"].to_numpy(dtype=int)
        totals = candidate_nulls["null_adjacency_opportunities"].to_numpy(dtype=int)
        fpr_bootstrap = _cluster_bootstrap_ratio(
            successes,
            totals,
            replicates=spec.bootstrap_replicates,
            rng=_candidate_rng(spec.random_seed, candidate, "null"),
        )
        cluster_upper = float(np.quantile(fpr_bootstrap, 1.0 - alpha))
        pooled_successes, pooled_total = int(successes.sum()), int(totals.sum())
        wilson = wilson_upper(pooled_successes, pooled_total, alpha=alpha)
        null_upper = max(cluster_upper, wilson)

        typed_by_world = candidate_results[list(TYPED_RECALL_COLUMNS)].mean(axis=1)
        recall_bootstrap = _cluster_bootstrap_mean(
            typed_by_world,
            replicates=spec.bootstrap_replicates,
            rng=_candidate_rng(spec.random_seed, candidate, "recall"),
        )
        recall_lower = float(np.quantile(recall_bootstrap, 1.0 - spec.confidence))
        class_means = {
            column: float(candidate_results[column].mean())
            for column in TYPED_RECALL_COLUMNS
        }
        config = configs[candidate]
        row: dict[str, Any] = {
            "candidate": candidate,
            "null_false_adjacencies": pooled_successes,
            "null_adjacency_opportunities": pooled_total,
            "null_fpr": pooled_successes / pooled_total,
            "null_fpr_cluster_bootstrap_upper": cluster_upper,
            "null_fpr_wilson_upper": wilson,
            "null_fpr_conservative_upper": null_upper,
            "controls_target_null_fpr": int(null_upper <= spec.target_null_fpr),
            "typed_motif_macro_recall": float(typed_by_world.mean()),
            "typed_motif_recall_cluster_bootstrap_lower": recall_lower,
            "min_typed_class_recall": min(class_means.values()),
            "world_success_rate": float(candidate_results["world_success"].mean()),
            "primary_skeleton_precision": float(
                candidate_results["primary_skeleton_precision"].mean()
            ),
            "primary_skeleton_f1": float(
                candidate_results["primary_skeleton_f1"].mean()
            ),
            "mean_estimated_focal_edges": float(
                candidate_results["n_estimated_focal_edges"].mean()
            ),
            **{f"mean__{column}": value for column, value in class_means.items()},
        }
        for field in (
            "min_state_score",
            "min_abs_predicted_r",
            "hierarchical_probability_threshold",
            "direct_residual_q",
            "mutual_state_neighbors",
        ):
            if field in config:
                row[f"config__{field}"] = config[field]
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values(
        [
            "controls_target_null_fpr",
            "typed_motif_recall_cluster_bootstrap_lower",
            "typed_motif_macro_recall",
            "min_typed_class_recall",
            "primary_skeleton_precision",
            "mean_estimated_focal_edges",
            "candidate",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def _hash_records(paths: Iterable[Path], *, base: Path | None = None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(set(paths)):
        try:
            label = str(path.relative_to(base)) if base is not None else str(path)
        except ValueError:
            label = str(path)
        records.append({"path": label, "sha256": file_sha256(path)})
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def calibrate_benchmark(
    benchmark_dir: Path | str,
    out_dir: Path | str,
    spec: CalibrationSpec | None = None,
) -> CalibrationArtifacts:
    """Calibrate and freeze one v4 candidate from preregistered training worlds."""

    cfg = spec or CalibrationSpec()
    benchmark = Path(benchmark_dir)
    output = Path(out_dir)
    _assert_not_locked_path(benchmark)
    _assert_not_locked_path(output)
    manifest_path = benchmark / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise CalibrationError(f"Missing benchmark manifest: {manifest_path}")

    # The order below is part of the information boundary: provenance first,
    # then discovery-only evidence, then scorer truth and post-consensus scores.
    manifest = _read_json(manifest_path)
    configs = _candidate_configs(manifest)
    sources, benchmark_world_bindings = _validate_manifest_worlds(manifest, cfg)
    expected_seeds = set(sources)
    raw, raw_paths, discovery_summary_paths = load_raw_evidence_first(
        benchmark, expected_seeds, cfg
    )
    verified_world_bindings = _verify_full_world_bindings_after_discovery(
        sources,
        benchmark_world_bindings,
    )
    truth, truth_paths = load_truth_after_discovery(sources)
    null_summary, null_world_summary = sentinel_empirical_null(raw, truth, cfg)
    results, results_path = _load_scored_results(
        benchmark, expected_seeds, set(configs)
    )
    null_counts = _candidate_null_counts(results, truth)
    ranking = rank_candidates(results, null_counts, configs, cfg)
    controlled = ranking.loc[ranking["controls_target_null_fpr"].eq(1)]
    if controlled.empty:
        raise CalibrationError(
            "No candidate controls the predeclared sentinel-null false adjacency target; "
            "refusing to freeze a configuration"
        )
    selected_candidate = str(controlled.iloc[0]["candidate"])
    frozen_config = dict(configs[selected_candidate])

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "ca_boss_v4_frozen_config.json"
    ranking_path = output / "candidate_calibration_ranking.tsv"
    null_summary_path = output / "sentinel_null_world_summary.tsv"
    audit_path = output / "calibration_audit_manifest.json"
    _write_json(config_path, frozen_config)
    ranking.to_csv(ranking_path, sep="\t", index=False)
    null_world_summary.to_csv(null_summary_path, sep="\t", index=False)

    selected_row = ranking.loc[ranking["candidate"].eq(selected_candidate)].iloc[0]
    selected_metrics = {
        key: (
            int(value)
            if isinstance(value, (np.integer,))
            else float(value)
            if isinstance(value, (np.floating,))
            else value
        )
        for key, value in selected_row.to_dict().items()
    }
    source_path = Path(__file__).resolve()
    audit: dict[str, Any] = {
        "schema_version": CALIBRATION_AUDIT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CA-BOSS v4 explicit-state empirical-null selection",
        "selection_split": "training_only",
        "registered_training_seed_block": [TRAINING_SEED_MIN, TRAINING_SEED_MAX],
        "world_seeds": sorted(expected_seeds),
        "calibration_spec": asdict(cfg),
        "selection_rule": {
            "hard_gate": (
                "max(world-cluster-bootstrap upper, one-sided Wilson upper) "
                "<= target_null_fpr"
            ),
            "candidate_multiplicity_control": (
                "Bonferroni simultaneous one-sided bounds"
                if cfg.simultaneous_candidate_control
                else "candidate-wise one-sided bounds"
            ),
            "ranking_after_gate": [
                "typed motif recall world-cluster-bootstrap lower bound",
                "typed motif macro recall",
                "minimum typed class recall",
                "primary skeleton precision",
                "lower mean focal edge count",
                "candidate name",
            ],
            "independent_cluster": "simulated biological world",
            "nested_runs_are_independent_samples": False,
        },
        "truth_boundary": {
            "raw_discovery_evidence_loaded_before_truth": True,
            "scorer_payload_hashes_verified_after_discovery_before_truth_parse": True,
            "benchmark_attests_truth_opened_after_consensus_only": True,
            "truth_use": "sentinel labels and post-consensus scoring only",
            "locked_worlds_accessed": False,
            "external_data_accessed": False,
        },
        "training_world_bindings": {
            "schema_version": TRAINING_BINDING_AUDIT_SCHEMA_VERSION,
            "required_for_frozen_candidate_use": True,
            "benchmark_schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
            "world_binding_schema_version": WORLD_INPUT_BINDING_SCHEMA_VERSION,
            "benchmark_bindings_verified_against_current_worlds": True,
            "world_count": len(verified_world_bindings),
            "world_seeds": sorted(expected_seeds),
            "bindings_canonical_sha256": canonical_sha256(
                verified_world_bindings
            ),
            "bindings": verified_world_bindings,
        },
        "empirical_null": null_summary,
        "selected": {
            "candidate": selected_candidate,
            "metrics": selected_metrics,
            "config_sha256": file_sha256(config_path),
            "config_canonical_sha256": canonical_sha256(frozen_config),
        },
        "artifacts": {
            "frozen_config": config_path.name,
            "candidate_ranking": ranking_path.name,
            "candidate_ranking_sha256": file_sha256(ranking_path),
            "sentinel_world_summary": null_summary_path.name,
            "sentinel_world_summary_sha256": file_sha256(null_summary_path),
        },
        "inputs": {
            "benchmark_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
            },
            "world_run_and_domain_manifests": verified_world_bindings,
            "world_run_and_domain_manifests_canonical_sha256": canonical_sha256(
                verified_world_bindings
            ),
            "world_results": {
                "path": str(results_path),
                "sha256": file_sha256(results_path),
            },
            "raw_pair_evidence": _hash_records(raw_paths, base=benchmark),
            "discovery_summaries": _hash_records(
                discovery_summary_paths, base=benchmark
            ),
            "scorer_truth": _hash_records(truth_paths),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "source_path": str(source_path),
            "source_sha256": file_sha256(source_path),
        },
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    _write_json(audit_path, audit)
    return CalibrationArtifacts(
        selected_candidate=selected_candidate,
        frozen_config=frozen_config,
        ranking=ranking,
        null_world_summary=null_world_summary,
        audit_manifest=audit,
        config_path=config_path,
        ranking_path=ranking_path,
        null_summary_path=null_summary_path,
        audit_path=audit_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-null-fpr", type=float, default=0.10)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=41_004)
    parser.add_argument("--expected-worlds", type=int, default=DEFAULT_EXPECTED_WORLDS)
    parser.add_argument(
        "--expected-nested-runs", type=int, default=DEFAULT_EXPECTED_NESTED_RUNS
    )
    parser.add_argument(
        "--candidate-wise-bounds",
        action="store_true",
        help="Disable simultaneous Bonferroni bounds (exploratory use only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = calibrate_benchmark(
        args.benchmark_dir,
        args.out_dir,
        CalibrationSpec(
            expected_worlds=args.expected_worlds,
            expected_nested_runs=args.expected_nested_runs,
            target_null_fpr=args.target_null_fpr,
            confidence=args.confidence,
            bootstrap_replicates=args.bootstrap_replicates,
            random_seed=args.random_seed,
            simultaneous_candidate_control=not args.candidate_wise_bounds,
        ),
    )
    selected = artifacts.ranking.iloc[0]
    print(
        json.dumps(
            {
                "selected_candidate": artifacts.selected_candidate,
                "null_fpr": float(selected["null_fpr"]),
                "null_fpr_conservative_upper": float(
                    selected["null_fpr_conservative_upper"]
                ),
                "typed_motif_macro_recall": float(
                    selected["typed_motif_macro_recall"]
                ),
                "config": str(artifacts.config_path),
                "audit": str(artifacts.audit_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
