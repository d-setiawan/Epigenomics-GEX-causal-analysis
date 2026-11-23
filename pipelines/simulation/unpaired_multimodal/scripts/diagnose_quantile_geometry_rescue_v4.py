#!/usr/bin/env python3
"""Diagnose quantile-geometry synchronization on cached CA-BOSS v4 worlds.

This is a retrospective, truth-separated method-development diagnostic.  It
reuses the frozen held-out-background ICA scores and reporter-component
assignments that fed CA-BOSS v4.53.  For each assay pair and each of the full
fit plus two disjoint feature folds, it compares reporter-component
distributions with sign-invariant one- and two-Wasserstein quantile costs.
The costs are used both locally (reciprocal component matching) and globally
(exact rank-three multi-assay permutation synchronization).

Eight feature families are declared before scoring.  Each is evaluated with
the v4.53 nested empirical-copula, mutual-nearest-neighbour, other-block null,
and whole-world bootstrap protocol.  Positive labels are not used to fit a
score or threshold.  Other-block truth identifies null pairs only, while
target truth is opened only after predictions are frozen for retrospective
diagnostic scoring.  No new worlds, cells, or modalities are generated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_mutual_nearest_margin_rescue_v4 as v453  # noqa: E402
import crossfit_multivariate_reporter_rescue_v4 as v448  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import relaxed_reporter_synchronization_evidence_v4 as relaxed  # noqa: E402


SCHEMA = "ca_boss_v4.quantile_geometry_rescue_diagnostic.v1"
ALGORITHM = "CA-BOSS v4 quantile-geometry rescue diagnostic"
DEFAULT_RANDOM_SEED = 454000
DEFAULT_WORLDS_PER_BLOCK = 8
DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_BOOTSTRAP_FREQUENCY = 0.80
QUANTILE_GRID = np.linspace(0.01, 0.99, 99, dtype=np.float64)
METRICS = ("wasserstein1", "wasserstein2")
BASE_COLUMNS = (
    "statistical_context_support_count",
    "global_reporter_coassignment_context_count",
    "maximum_reporter_pair_scaled_ks_ratio",
    "mean_coassigned_pair_exclusion_relative_margin",
)
BASE_DIRECTIONS = (1.0, 1.0, -1.0, 1.0)
VARIANTS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "base_v453": (BASE_COLUMNS, BASE_DIRECTIONS),
    "w1_global_mirror": (
        (
            "statistical_context_support_count",
            "wasserstein1_global_coassignment_context_count",
            "wasserstein1_maximum_reporter_cost_ratio",
            "wasserstein1_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "w2_global_mirror": (
        (
            "statistical_context_support_count",
            "wasserstein2_global_coassignment_context_count",
            "wasserstein2_maximum_reporter_cost_ratio",
            "wasserstein2_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "w1_global_distribution_only": (
        (
            "wasserstein1_global_coassignment_context_count",
            "wasserstein1_maximum_reporter_cost_ratio",
            "wasserstein1_mean_coassigned_exclusion_margin",
        ),
        (1.0, -1.0, 1.0),
    ),
    "w2_global_distribution_only": (
        (
            "wasserstein2_global_coassignment_context_count",
            "wasserstein2_maximum_reporter_cost_ratio",
            "wasserstein2_mean_coassigned_exclusion_margin",
        ),
        (1.0, -1.0, 1.0),
    ),
    "w1_local_distribution_only": (
        (
            "wasserstein1_local_mutual_context_count",
            "wasserstein1_maximum_reporter_cost_ratio",
            "wasserstein1_mean_local_relative_margin",
        ),
        (1.0, -1.0, 1.0),
    ),
    "w2_local_distribution_only": (
        (
            "wasserstein2_local_mutual_context_count",
            "wasserstein2_maximum_reporter_cost_ratio",
            "wasserstein2_mean_local_relative_margin",
        ),
        (1.0, -1.0, 1.0),
    ),
    "base_plus_w1_global": (
        BASE_COLUMNS
        + (
            "wasserstein1_global_coassignment_context_count",
            "wasserstein1_maximum_reporter_cost_ratio",
            "wasserstein1_mean_coassigned_exclusion_margin",
        ),
        BASE_DIRECTIONS + (1.0, -1.0, 1.0),
    ),
}
CONSENSUS_RULES = {
    "wasserstein1_global_local_consensus": (
        "w1_global_mirror",
        "w1_local_distribution_only",
    ),
    "wasserstein2_global_local_consensus": (
        "w2_global_mirror",
        "w2_local_distribution_only",
    ),
}


@dataclass(frozen=True)
class _WorldArtifacts:
    relaxed_dir: Path
    mdcr_dir: Path
    assignment_path: Path
    evidence_manifest: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    stripped = {
        key: value for key, value in payload.items() if key != "manifest_payload_sha256"
    }
    encoded = json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def _pair_tuple(value: frozenset[str]) -> tuple[str, str]:
    ordered = sorted(value)
    if len(ordered) != 2:
        raise ValueError("An assay pair must contain exactly two modalities")
    return ordered[0], ordered[1]


def _parse_pairs(value: Any) -> set[frozenset[str]]:
    payload = json.loads(str(value))
    return {_pair(str(left), str(right)) for left, right in payload}


def _verified_manifest(path: Path, schema: str | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"Unexpected schema in {path}")
    expected = str(payload.get("manifest_payload_sha256", ""))
    if not expected or _canonical_sha256(payload) != expected:
        raise ValueError(f"Manifest payload hash mismatch: {path}")
    return payload


def _discover_v453(
    output_root: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, set[frozenset[str]]],
    dict[str, dict[str, Any]],
]:
    frames: dict[str, pd.DataFrame] = {}
    baseline: dict[str, set[frozenset[str]]] = defaultdict(set)
    manifests: dict[str, dict[str, Any]] = {}
    pattern = "bootstrap_mutual_nearest_margin_rescue_v453_source_current_target_*"
    for directory in sorted(output_root.glob(pattern)):
        if not directory.is_dir():
            continue
        manifest = _verified_manifest(
            directory / "calibration_manifest.json", v453.SCHEMA
        )
        block = str(manifest["target_block"])
        if block in frames:
            raise ValueError(f"Duplicate v4.53 target block: {block}")
        audit_path = directory / "bootstrap_mnn_pair_audit.tsv"
        if _sha256(audit_path) != str(
            manifest.get("output_sha256", {}).get(audit_path.name)
        ):
            raise ValueError(f"v4.53 pair-audit hash mismatch: {block}")
        frame = pd.read_csv(audit_path, sep="\t", keep_default_na=False)
        if set(frame["block"].astype(str)) != {block}:
            raise ValueError(f"v4.53 pair audit crosses target blocks: {block}")
        frames[block] = frame
        manifests[block] = manifest
        candidate_path = directory / "bootstrap_mnn_candidates.tsv"
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        if _sha256(candidate_path) != str(
            manifest.get("output_sha256", {}).get(candidate_path.name)
        ):
            raise ValueError(f"v4.53 candidate-table hash mismatch: {block}")
        candidates = pd.read_csv(candidate_path, sep="\t", keep_default_na=False)
        for row in candidates.itertuples(index=False):
            if not synchronization._as_bool(row.hypothesis_selected):
                raise ValueError("v4.53 candidate table contains an unselected row")
            baseline[str(row.world_key)].update(
                _parse_pairs(row.candidate_assay_pairs_json)
            )
    if len(frames) != 5:
        raise ValueError(f"Expected five frozen v4.53 blocks, found {len(frames)}")
    return frames, dict(baseline), manifests


def _discover_relaxed(output_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    worlds: dict[str, tuple[Path, dict[str, Any]]] = {}
    pattern = "relaxed_reporter_synchronization_evidence_v447_source_current_pretruth_*"
    for root in sorted(output_root.glob(pattern)):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/evidence_manifest.json")):
            manifest = _verified_manifest(path, relaxed.SCHEMA)
            key = path.parent.name
            expected_key = f"{manifest['block']}__world_seed{int(manifest['seed'])}"
            if key != expected_key or key in worlds:
                raise ValueError(f"Invalid or duplicate relaxed world: {key}")
            pair_path = path.parent / "relaxed_reporter_synchronization_pairs.tsv"
            if _sha256(pair_path) != str(
                manifest.get("output_sha256", {}).get(pair_path.name)
            ):
                raise ValueError(f"Relaxed pair-evidence hash mismatch: {key}")
            worlds[key] = (path.parent, manifest)
    return worlds


def _mdcr_index(output_root: Path) -> dict[tuple[str, str, str], list[Path]]:
    index: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    pattern = "published_unpaired_mdcr_ks_crossfit_v4*"
    for root in sorted(output_root.glob(pattern)):
        if not root.is_dir() or root.name.endswith("_scores"):
            continue
        for path in sorted(root.glob("*/frozen_evidence_manifest.json")):
            directory = path.parent
            if not (directory / "component_scores.tsv").is_file():
                continue
            manifest = _verified_manifest(path)
            pair_path = directory / "pairwise_component_matches.tsv"
            key = (
                directory.name,
                str(manifest["manifest_payload_sha256"]),
                _sha256(pair_path),
            )
            index[key].append(directory)
    return dict(index)


def _assignment_index(output_root: Path) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    pattern = "component_conditioned_bridge_v444_source_current_*"
    for root in sorted(output_root.glob(pattern)):
        if not root.is_dir() or "panel_summary" in root.name or "scores" in root.name:
            continue
        for path in sorted(root.glob("*/assay_axis_assignments.tsv")):
            index[(path.parent.name, _sha256(path))].append(path)
    return dict(index)


def _resolve_artifacts(
    output_root: Path,
    world_keys: Sequence[str],
) -> dict[str, _WorldArtifacts]:
    relaxed_worlds = _discover_relaxed(output_root)
    mdcr = _mdcr_index(output_root)
    assignments = _assignment_index(output_root)
    resolved: dict[str, _WorldArtifacts] = {}
    for key in sorted(world_keys):
        if key not in relaxed_worlds:
            raise ValueError(f"Missing relaxed evidence for {key}")
        relaxed_dir, manifest = relaxed_worlds[key]
        inputs = manifest.get("input_sha256", {})
        mdcr_key = (
            key,
            str(manifest["mdcr_manifest_payload_sha256"]),
            str(inputs["pairwise_component_matches.tsv"]),
        )
        mdcr_candidates = mdcr.get(mdcr_key, [])
        assignment_candidates = assignments.get(
            (key, str(inputs["assay_axis_assignments.tsv"])), []
        )
        if not mdcr_candidates or not assignment_candidates:
            raise ValueError(f"Cannot resolve hash-bound upstream artifacts for {key}")
        resolved[key] = _WorldArtifacts(
            relaxed_dir=relaxed_dir,
            mdcr_dir=sorted(mdcr_candidates)[0],
            assignment_path=sorted(assignment_candidates)[0],
            evidence_manifest=manifest,
        )
    return resolved


def _load_feature_caches(
    directories: Sequence[Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    cached: dict[str, pd.DataFrame] = {}
    manifests: dict[str, str] = {}
    for raw_directory in directories:
        directory = raw_directory.expanduser().absolute().resolve(strict=True)
        manifest_path = directory / "diagnostic_manifest.json"
        manifest = _verified_manifest(manifest_path, SCHEMA)
        table_path = directory / "quantile_geometry_pair_features.tsv"
        if _sha256(table_path) != str(
            manifest.get("output_sha256", {}).get(table_path.name)
        ):
            raise ValueError(f"Cached feature-table hash mismatch: {directory}")
        table = pd.read_csv(table_path, sep="\t", keep_default_na=False)
        for key, group in table.groupby("world_key", sort=False):
            world_key = str(key)
            normalized = group.reset_index(drop=True)
            if world_key in cached and not cached[world_key].equals(normalized):
                raise ValueError(f"Conflicting cached features for {world_key}")
            cached[world_key] = normalized
        manifests[str(directory)] = _sha256(manifest_path)
    return cached, manifests


def _sample_worlds(
    frames: Mapping[str, pd.DataFrame],
    *,
    worlds_per_block: int,
    random_seed: int,
    excluded_world_keys: set[str] | None = None,
) -> dict[str, list[str]]:
    excluded = excluded_world_keys or set()
    sampled: dict[str, list[str]] = {}
    for block_index, block in enumerate(sorted(frames)):
        keys = sorted(
            key
            for key in frames[block]["world_key"].astype(str).unique()
            if key not in excluded
        )
        if not keys:
            raise ValueError(f"Every world in {block} was excluded")
        count = len(keys) if worlds_per_block <= 0 else min(worlds_per_block, len(keys))
        rng = np.random.default_rng(
            np.random.SeedSequence([random_seed, block_index])
        )
        positions = np.sort(rng.choice(len(keys), size=count, replace=False))
        sampled[block] = [keys[int(position)] for position in positions]
    return sampled


def _score_contexts(mdcr_dir: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    full = pd.read_csv(mdcr_dir / "component_scores.tsv", sep="\t")
    fold = pd.read_csv(mdcr_dir / "fold_component_scores.tsv", sep="\t")
    component_columns = [
        column for column in full.columns if str(column).startswith("component_")
    ]
    if component_columns != ["component_0", "component_1", "component_2"]:
        raise ValueError(f"Unexpected component columns: {mdcr_dir}")
    fold_ids = sorted(pd.to_numeric(fold["feature_fold"], errors="raise").astype(int).unique())
    if len(fold_ids) != 2:
        raise ValueError(f"Expected two feature folds: {mdcr_dir}")
    contexts: dict[str, dict[str, np.ndarray]] = {"full_fit": {}}
    for modality, group in full.groupby("modality", sort=True):
        contexts["full_fit"][str(modality)] = group[component_columns].to_numpy(
            dtype=np.float64
        )
    for fold_id in fold_ids:
        label = f"feature_fold_{fold_id}"
        contexts[label] = {}
        selected = fold.loc[pd.to_numeric(fold["feature_fold"]).astype(int) == fold_id]
        for modality, group in selected.groupby("modality", sort=True):
            contexts[label][str(modality)] = group[component_columns].to_numpy(
                dtype=np.float64
            )
    return ["full_fit", *[f"feature_fold_{value}" for value in fold_ids]], contexts


def _reporter_contexts(
    assignment_path: Path,
    context_labels: Sequence[str],
) -> tuple[dict[str, list[int | None]], dict[str, list[bool]]]:
    assignment = pd.read_csv(assignment_path, sep="\t", keep_default_na=False)
    fold_ids = [int(label.rsplit("_", 1)[1]) for label in context_labels[1:]]
    components: dict[str, list[int | None]] = {}
    supports: dict[str, list[bool]] = {}
    for _, row in assignment.iterrows():
        _, values, flags = relaxed._reporter_contexts(row, fold_ids)
        modality = str(row["modality"])
        components[modality] = values
        supports[modality] = flags
    return components, supports


def _quantiles(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.quantile(np.asarray(values, dtype=np.float64), QUANTILE_GRID),
        dtype=np.float64,
    )


def _signed_quantile_cost(
    left: np.ndarray,
    right: np.ndarray,
    *,
    power: int,
) -> float:
    same = np.abs(left - right)
    flipped = np.abs(left + right[::-1])
    if power == 1:
        return float(min(same.mean(), flipped.mean()))
    if power == 2:
        return float(np.sqrt(min(np.square(same).mean(), np.square(flipped).mean())))
    raise ValueError(f"Unsupported Wasserstein power: {power}")


def _cost_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    power: int,
) -> np.ndarray:
    if left.shape[1] != 3 or right.shape[1] != 3:
        raise ValueError("Quantile geometry is fixed to rank three")
    left_quantiles = [_quantiles(left[:, index]) for index in range(3)]
    right_quantiles = [_quantiles(right[:, index]) for index in range(3)]
    return np.asarray(
        [
            [
                _signed_quantile_cost(
                    left_quantiles[left_index],
                    right_quantiles[right_index],
                    power=power,
                )
                for right_index in range(3)
            ]
            for left_index in range(3)
        ],
        dtype=np.float64,
    )


def _normalized_cost_table(
    scores: Mapping[str, np.ndarray],
    matrices: Mapping[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(scores), 2):
        matrix = matrices[(left, right)]
        positive = matrix[matrix > 1e-12]
        scale = float(np.median(positive)) if positive.size else 1.0
        for left_component in range(3):
            for right_component in range(3):
                rows.append(
                    {
                        "left_modality": left,
                        "right_modality": right,
                        "left_component": left_component,
                        "right_component": right_component,
                        "scaled_ks_statistic": float(
                            matrix[left_component, right_component]
                        ),
                        "asymptotic_critical_value": scale,
                    }
                )
    return pd.DataFrame(rows)


def _oriented_matrix(
    matrices: Mapping[tuple[str, str], np.ndarray], left: str, right: str
) -> np.ndarray:
    if (left, right) in matrices:
        return matrices[(left, right)]
    if (right, left) in matrices:
        return matrices[(right, left)].T
    raise KeyError((left, right))


def _local_diagnostics(matrix: np.ndarray, left: int, right: int) -> tuple[float, float, bool]:
    selected = float(matrix[left, right])
    competitors = np.concatenate(
        (np.delete(matrix[left, :], right), np.delete(matrix[:, right], left))
    )
    runner_up = float(competitors.min())
    positive = matrix[matrix > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    relative_margin = float((runner_up - selected) / max(scale, 1e-12))
    return selected / max(scale, 1e-12), relative_margin, bool(selected < runner_up)


def _world_features(
    pair_frame: pd.DataFrame,
    artifacts: _WorldArtifacts,
) -> pd.DataFrame:
    context_labels, contexts = _score_contexts(artifacts.mdcr_dir)
    components, supports = _reporter_contexts(
        artifacts.assignment_path, context_labels
    )
    matrices: dict[str, list[dict[tuple[str, str], np.ndarray]]] = {
        metric: [] for metric in METRICS
    }
    solutions: dict[
        str, list[synchronization.MultiwayPermutationSolution | None]
    ] = {metric: [] for metric in METRICS}
    tables: dict[str, list[pd.DataFrame | None]] = {metric: [] for metric in METRICS}
    for context_index, label in enumerate(context_labels):
        eligible = sorted(
            modality
            for modality, flags in supports.items()
            if flags[context_index] and modality in contexts[label]
        )
        for metric, power in (("wasserstein1", 1), ("wasserstein2", 2)):
            pair_matrices = {
                (left, right): _cost_matrix(
                    contexts[label][left], contexts[label][right], power=power
                )
                for left, right in combinations(sorted(contexts[label]), 2)
            }
            matrices[metric].append(pair_matrices)
            if len(eligible) >= 2:
                table = _normalized_cost_table(
                    {modality: contexts[label][modality] for modality in eligible},
                    pair_matrices,
                )
                solution = synchronization.solve_multiway_permutation_synchronization(
                    table, eligible
                )
            else:
                table = None
                solution = None
            tables[metric].append(table)
            solutions[metric].append(solution)

    enriched = pair_frame.copy()
    for metric in METRICS:
        aggregates: dict[str, list[Any]] = defaultdict(list)
        for row in pair_frame.itertuples(index=False):
            left = str(row.left_modality)
            right = str(row.right_modality)
            local_mutual: list[bool] = []
            local_margins: list[float] = []
            reporter_costs: list[float] = []
            global_coassigned: list[bool] = []
            global_margins: list[float] = []
            for context_index, _ in enumerate(context_labels):
                left_component = components.get(left, [None] * len(context_labels))[
                    context_index
                ]
                right_component = components.get(right, [None] * len(context_labels))[
                    context_index
                ]
                if left_component is None or right_component is None:
                    local_mutual.append(False)
                    local_margins.append(0.0)
                    reporter_costs.append(float("nan"))
                    global_coassigned.append(False)
                    continue
                matrix = _oriented_matrix(matrices[metric][context_index], left, right)
                cost, margin, mutual = _local_diagnostics(
                    matrix, int(left_component), int(right_component)
                )
                reporter_costs.append(cost)
                local_margins.append(margin)
                local_mutual.append(mutual)
                solution = solutions[metric][context_index]
                table = tables[metric][context_index]
                if (
                    solution is None
                    or table is None
                    or left not in solution.modalities
                    or right not in solution.modalities
                ):
                    global_coassigned.append(False)
                    continue
                diagnostics = synchronization.reporter_pair_diagnostics(
                    solution,
                    table,
                    left,
                    right,
                    int(left_component),
                    int(right_component),
                )
                coassigned = bool(diagnostics["coassigned_in_best_solution"])
                global_coassigned.append(coassigned)
                if coassigned:
                    global_margins.append(
                        float(diagnostics["pair_exclusion_relative_margin"])
                    )
            finite_costs = [value for value in reporter_costs if np.isfinite(value)]
            aggregates[f"{metric}_local_mutual_context_count"].append(
                int(sum(local_mutual))
            )
            aggregates[f"{metric}_heldout_local_mutual_context_count"].append(
                int(sum(local_mutual[1:]))
            )
            aggregates[f"{metric}_mean_local_relative_margin"].append(
                float(np.mean(local_margins)) if local_margins else 0.0
            )
            aggregates[f"{metric}_minimum_local_relative_margin"].append(
                float(min(local_margins)) if local_margins else 0.0
            )
            aggregates[f"{metric}_maximum_reporter_cost_ratio"].append(
                float(max(finite_costs)) if finite_costs else float("nan")
            )
            aggregates[f"{metric}_mean_reporter_cost_ratio"].append(
                float(np.mean(finite_costs)) if finite_costs else float("nan")
            )
            aggregates[f"{metric}_global_coassignment_context_count"].append(
                int(sum(global_coassigned))
            )
            aggregates[f"{metric}_heldout_global_coassignment_context_count"].append(
                int(sum(global_coassigned[1:]))
            )
            aggregates[f"{metric}_mean_coassigned_exclusion_margin"].append(
                float(np.mean(global_margins)) if global_margins else 0.0
            )
        for column, values in aggregates.items():
            enriched[column] = values
    return enriched


def _model_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Diagnostic features are missing: {missing}")
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _copula_score(
    training: pd.DataFrame,
    values: pd.DataFrame,
    *,
    columns: Sequence[str],
    directions: Sequence[float],
) -> np.ndarray:
    training_values = _model_matrix(training, columns) * np.asarray(directions)
    target_values = _model_matrix(values, columns) * np.asarray(directions)
    denominator = float(len(training_values) + 1)
    percentiles = [
        np.searchsorted(
            np.sort(training_values[:, index]),
            target_values[:, index],
            side="right",
        ).astype(float)
        / denominator
        for index in range(training_values.shape[1])
    ]
    return np.prod(np.column_stack(percentiles), axis=1)


def _truth_labels(
    frame: pd.DataFrame,
    truth: Mapping[str, set[frozenset[str]]],
) -> np.ndarray:
    return np.asarray(
        [
            _pair(row.left_modality, row.right_modality)
            in truth[str(row.world_key)]
            for row in frame.itertuples(index=False)
        ],
        dtype=bool,
    )


def _nested_selection(
    *,
    target: pd.DataFrame,
    target_block: str,
    frames: Mapping[str, pd.DataFrame],
    block_keys: Mapping[str, Sequence[str]],
    truth: Mapping[str, set[frozenset[str]]],
    columns: Sequence[str],
    directions: Sequence[float],
    world_frames: Mapping[str, pd.DataFrame] | None = None,
    sampled_training_keys: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if world_frames is None:
        world_frames = frames_by_world(frames)
    calibration_blocks = [block for block in sorted(frames) if block != target_block]
    fold_margins: list[np.ndarray] = []
    fold_mutual: list[np.ndarray] = []
    null_maxima: dict[str, float] = {}
    for calibration_block in calibration_blocks:
        training_keys: list[str] = []
        for training_block in calibration_blocks:
            if training_block == calibration_block:
                continue
            keys = (
                block_keys[training_block]
                if sampled_training_keys is None
                else sampled_training_keys[(calibration_block, training_block)]
            )
            training_keys.extend(keys)
        training = pd.concat(
            [world_frames[key] for key in training_keys],
            ignore_index=True,
        )
        calibration = pd.concat(
            [world_frames[key] for key in block_keys[calibration_block]],
            ignore_index=True,
        )
        target_scores = _copula_score(
            training, target, columns=columns, directions=directions
        )
        target_margins, target_mutual = v453._mutual_nearest_absolute_margin(
            target, target_scores
        )
        calibration_scores = _copula_score(
            training, calibration, columns=columns, directions=directions
        )
        calibration_margins, calibration_mutual = (
            v453._mutual_nearest_absolute_margin(calibration, calibration_scores)
        )
        null = (~_truth_labels(calibration, truth)) & calibration_mutual
        if not null.any():
            raise ValueError(
                f"No mutual-nearest null for {target_block}/{calibration_block}"
            )
        fold_margins.append(target_margins)
        fold_mutual.append(target_mutual)
        null_maxima[calibration_block] = float(calibration_margins[null].max())
    margins = np.vstack(fold_margins).T
    mutual = np.vstack(fold_mutual).T
    selected = mutual.all(axis=1) & (margins.min(axis=1) > max(null_maxima.values()))
    return selected, margins, null_maxima


def frames_by_world(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for frame in frames.values():
        for key, group in frame.groupby("world_key", sort=False):
            result[str(key)] = group.reset_index(drop=True)
    return result


def _load_truth(
    formal_worlds_root: Path,
    sampled: Mapping[str, Sequence[str]],
) -> dict[str, set[frozenset[str]]]:
    truth: dict[str, set[frozenset[str]]] = {}
    for block, keys in sampled.items():
        for key in keys:
            seed = int(key.rsplit("world_seed", 1)[1])
            path = (
                formal_worlds_root
                / block
                / f"world_seed{seed}"
                / "scorer"
                / "truth_typed_edges.tsv"
            )
            truth[key] = v448.v447._truth_assay_pairs(path)
    return truth


def _evaluate_variants(
    frames: Mapping[str, pd.DataFrame],
    block_keys: Mapping[str, Sequence[str]],
    truth: Mapping[str, set[frozenset[str]]],
    baseline: Mapping[str, set[frozenset[str]]],
    *,
    bootstrap_replicates: int,
    bootstrap_frequency: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    world_frames = frames_by_world(frames)
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant_index, (variant, (columns, directions)) in enumerate(VARIANTS.items()):
        variant_new_tp = 0
        variant_new_fp = 0
        variant_nominal_tp = 0
        variant_nominal_fp = 0
        for target_index, target_block in enumerate(sorted(frames)):
            target = frames[target_block].reset_index(drop=True)
            nominal, margins, null_maxima = _nested_selection(
                target=target,
                target_block=target_block,
                frames=frames,
                block_keys=block_keys,
                truth=truth,
                columns=columns,
                directions=directions,
                world_frames=world_frames,
            )
            baseline_mask = np.asarray(
                [
                    _pair(row.left_modality, row.right_modality)
                    in baseline.get(str(row.world_key), set())
                    for row in target.itertuples(index=False)
                ],
                dtype=bool,
            )
            proposal = nominal & ~baseline_mask
            selection_count = np.zeros(len(target), dtype=np.int64)
            calibration_blocks = [
                block for block in sorted(frames) if block != target_block
            ]
            for replicate in range(bootstrap_replicates):
                sampled_training: dict[tuple[str, str], list[str]] = {}
                for calibration_index, calibration_block in enumerate(
                    calibration_blocks
                ):
                    for training_index, training_block in enumerate(
                        block for block in calibration_blocks if block != calibration_block
                    ):
                        keys = list(block_keys[training_block])
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    random_seed,
                                    variant_index,
                                    target_index,
                                    replicate,
                                    calibration_index,
                                    training_index,
                                ]
                            )
                        )
                        sampled_training[(calibration_block, training_block)] = [
                            keys[int(index)]
                            for index in rng.integers(0, len(keys), size=len(keys))
                        ]
                selected, _, _ = _nested_selection(
                    target=target,
                    target_block=target_block,
                    frames=frames,
                    block_keys=block_keys,
                    truth=truth,
                    columns=columns,
                    directions=directions,
                    world_frames=world_frames,
                    sampled_training_keys=sampled_training,
                )
                selection_count += selected.astype(np.int64)
            frequencies = selection_count.astype(float) / float(bootstrap_replicates)
            stable = proposal & (frequencies >= bootstrap_frequency)
            labels = _truth_labels(target, truth)
            variant_nominal_tp += int(np.sum(proposal & labels))
            variant_nominal_fp += int(np.sum(proposal & ~labels))
            variant_new_tp += int(np.sum(stable & labels))
            variant_new_fp += int(np.sum(stable & ~labels))
            for index, row in enumerate(target.itertuples(index=False)):
                if not proposal[index] and not stable[index]:
                    continue
                pair_rows.append(
                    {
                        "variant": variant,
                        "target_block": target_block,
                        "world_key": str(row.world_key),
                        "left_modality": str(row.left_modality),
                        "right_modality": str(row.right_modality),
                        "truth_pair": bool(labels[index]),
                        "selected_by_v453_baseline": bool(baseline_mask[index]),
                        "nominal_new_proposal": bool(proposal[index]),
                        "minimum_nested_absolute_margin": float(margins[index].min()),
                        "null_margin_upper_envelope": float(max(null_maxima.values())),
                        "bootstrap_selection_count": int(selection_count[index]),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_selection_frequency": float(frequencies[index]),
                        "stable_new_proposal": bool(stable[index]),
                    }
                )
        summary_rows.append(
            {
                "variant": variant,
                "feature_columns_json": json.dumps(list(columns), separators=(",", ":")),
                "feature_directions_json": json.dumps(
                    list(directions), separators=(",", ":")
                ),
                "nominal_new_true_pairs": variant_nominal_tp,
                "nominal_new_false_pairs": variant_nominal_fp,
                "stable_new_true_pairs": variant_new_tp,
                "stable_new_false_pairs": variant_new_fp,
                "stable_new_precision": (
                    float(variant_new_tp / (variant_new_tp + variant_new_fp))
                    if variant_new_tp + variant_new_fp
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def _consensus_proposals(
    proposals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity_columns = ["world_key", "left_modality", "right_modality"]
    stable = proposals.loc[proposals["stable_new_proposal"]].copy()
    indexed = {
        variant: {
            tuple(str(row[column]) for column in identity_columns): row
            for _, row in group.iterrows()
        }
        for variant, group in stable.groupby("variant", sort=False)
    }
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    union_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rule, (global_variant, local_variant) in CONSENSUS_RULES.items():
        shared = sorted(
            set(indexed.get(global_variant, {}))
            & set(indexed.get(local_variant, {}))
        )
        true_count = 0
        false_count = 0
        for identity in shared:
            global_row = indexed[global_variant][identity]
            local_row = indexed[local_variant][identity]
            truth_pair = bool(global_row["truth_pair"])
            if truth_pair != bool(local_row["truth_pair"]):
                raise RuntimeError("Consensus views disagree on diagnostic truth")
            true_count += int(truth_pair)
            false_count += int(not truth_pair)
            row = {
                "consensus_rule": rule,
                "world_key": identity[0],
                "left_modality": identity[1],
                "right_modality": identity[2],
                "truth_pair": truth_pair,
                "global_variant": global_variant,
                "local_variant": local_variant,
                "minimum_view_bootstrap_selection_frequency": float(
                    min(
                        global_row["bootstrap_selection_frequency"],
                        local_row["bootstrap_selection_frequency"],
                    )
                ),
                "selected_by_global_local_consensus": True,
            }
            rows.append(row)
            union_rows.setdefault(identity, row)
        summaries.append(
            {
                "consensus_rule": rule,
                "stable_new_true_pairs": true_count,
                "stable_new_false_pairs": false_count,
                "stable_new_precision": (
                    float(true_count / (true_count + false_count))
                    if true_count + false_count
                    else float("nan")
                ),
            }
        )
    union_true = sum(int(row["truth_pair"]) for row in union_rows.values())
    union_false = len(union_rows) - union_true
    summaries.append(
        {
            "consensus_rule": "union_of_w1_and_w2_global_local_consensus",
            "stable_new_true_pairs": union_true,
            "stable_new_false_pairs": union_false,
            "stable_new_precision": (
                float(union_true / len(union_rows)) if union_rows else float("nan")
            ),
        }
    )
    for identity, source in sorted(union_rows.items()):
        rows.append(
            {
                **source,
                "consensus_rule": "union_of_w1_and_w2_global_local_consensus",
                "global_variant": "one_or_both_wasserstein_global_mirror_views",
                "local_variant": "matching_wasserstein_local_distribution_view",
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(rows)


def run_diagnostic(
    *,
    output_root: Path,
    formal_worlds_root: Path,
    out_dir: Path,
    worlds_per_block: int = DEFAULT_WORLDS_PER_BLOCK,
    random_seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_frequency: float = DEFAULT_BOOTSTRAP_FREQUENCY,
    exclude_sample_manifest: Path | None = None,
    feature_cache_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    if bootstrap_replicates <= 0 or not 0.0 < bootstrap_frequency <= 1.0:
        raise ValueError("Bootstrap settings are invalid")
    output_root = output_root.expanduser().absolute().resolve(strict=True)
    formal_worlds_root = formal_worlds_root.expanduser().absolute().resolve(
        strict=True
    )
    raw_frames, baseline, v453_manifests = _discover_v453(output_root)
    excluded_world_keys: set[str] = set()
    excluded_manifest_sha256: str | None = None
    if exclude_sample_manifest is not None:
        excluded_path = exclude_sample_manifest.expanduser().absolute().resolve(
            strict=True
        )
        excluded_manifest = _read_json(excluded_path)
        excluded_manifest_sha256 = _sha256(excluded_path)
        excluded_world_keys = {
            str(key)
            for keys in excluded_manifest.get(
                "sampled_world_keys_by_block", {}
            ).values()
            for key in keys
        }
        if not excluded_world_keys:
            raise ValueError("Exclusion manifest contains no sampled world keys")
    sampled = _sample_worlds(
        raw_frames,
        worlds_per_block=worlds_per_block,
        random_seed=random_seed,
        excluded_world_keys=excluded_world_keys,
    )
    sampled_keys = sorted(key for keys in sampled.values() for key in keys)
    cached_features, cache_manifest_hashes = _load_feature_caches(
        feature_cache_dirs
    )
    missing_keys = [key for key in sampled_keys if key not in cached_features]
    artifacts = _resolve_artifacts(output_root, missing_keys) if missing_keys else {}
    sampled_frames: dict[str, pd.DataFrame] = {}
    feature_frames: list[pd.DataFrame] = []
    for block in sorted(raw_frames):
        selected = raw_frames[block].loc[
            raw_frames[block]["world_key"].astype(str).isin(sampled[block])
        ].copy()
        enriched_worlds = []
        for key, group in selected.groupby("world_key", sort=True):
            world_key = str(key)
            current = group.reset_index(drop=True)
            if world_key in cached_features:
                enriched = cached_features[world_key].drop(
                    columns=[
                        "truth_pair",
                        "selected_by_v453_baseline",
                    ],
                    errors="ignore",
                )
                identity_columns = [
                    "world_key",
                    "left_modality",
                    "right_modality",
                ]
                if not enriched.loc[:, identity_columns].equals(
                    current.loc[:, identity_columns]
                ):
                    raise ValueError(
                        f"Cached and current pair identities disagree: {world_key}"
                    )
                for column in BASE_COLUMNS:
                    if not np.allclose(
                        pd.to_numeric(enriched[column], errors="raise"),
                        pd.to_numeric(current[column], errors="raise"),
                        rtol=1e-12,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"Cached base evidence changed for {world_key}: {column}"
                        )
            else:
                enriched = _world_features(current, artifacts[world_key])
            enriched_worlds.append(enriched)
            feature_frames.append(enriched)
        sampled_frames[block] = pd.concat(enriched_worlds, ignore_index=True)
    block_keys = {
        block: sorted(frame["world_key"].astype(str).unique())
        for block, frame in sampled_frames.items()
    }
    truth = _load_truth(formal_worlds_root, sampled)
    feature_table = pd.concat(feature_frames, ignore_index=True)
    feature_table["truth_pair"] = _truth_labels(feature_table, truth)
    feature_table["selected_by_v453_baseline"] = [
        _pair(row.left_modality, row.right_modality)
        in baseline.get(str(row.world_key), set())
        for row in feature_table.itertuples(index=False)
    ]
    summary, proposals = _evaluate_variants(
        sampled_frames,
        block_keys,
        truth,
        baseline,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_frequency=bootstrap_frequency,
        random_seed=random_seed + 1,
    )
    consensus_summary, consensus_proposals = _consensus_proposals(proposals)
    out_dir.mkdir(parents=True)
    outputs = {
        "quantile_geometry_pair_features.tsv": feature_table,
        "quantile_geometry_variant_summary.tsv": summary,
        "quantile_geometry_proposals.tsv": proposals,
        "quantile_geometry_consensus_summary.tsv": consensus_summary,
        "quantile_geometry_consensus_proposals.tsv": consensus_proposals,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "retrospective_truth_separated_diagnostic",
        "random_seed": random_seed,
        "worlds_per_block_requested": worlds_per_block,
        "sampled_world_keys_by_block": sampled,
        "sampled_world_count": len(sampled_keys),
        "excluded_prior_sample_world_count": len(excluded_world_keys),
        "excluded_sample_manifest_sha256": excluded_manifest_sha256,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_frequency_threshold": bootstrap_frequency,
        "quantile_grid": {
            "minimum": float(QUANTILE_GRID.min()),
            "maximum": float(QUANTILE_GRID.max()),
            "count": len(QUANTILE_GRID),
        },
        "variants_declared_before_scoring": {
            name: {
                "feature_columns": list(columns),
                "feature_directions": list(directions),
            }
            for name, (columns, directions) in VARIANTS.items()
        },
        "selection_level_consensus_rules": {
            rule: list(views) for rule, views in CONSENSUS_RULES.items()
        },
        "recommended_diagnostic_rule": (
            "union_of_w1_and_w2_global_local_consensus"
        ),
        "information_boundary": {
            "existing_cached_worlds_only": True,
            "new_worlds_or_datasets_created": False,
            "heldout_background_component_scores_only_for_new_features": True,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
            "other_block_truth_used_only_to_identify_nulls": True,
            "positive_labels_used_for_score_or_threshold_fit": False,
            "target_truth_opened_only_after_prediction_for_diagnostic_scoring": True,
        },
        "published_method_inspiration": {
            "unpaired_multi_domain_causal_representation": {
                "citation": "Sturma, Squires, Drton, and Uhler, NeurIPS 2023",
                "doi": "10.52202/075280-1496",
            },
            "unpaired_multiomics_optimal_transport": {
                "citation": "Demetci et al., Journal of Computational Biology 2022",
                "doi": "10.1089/cmb.2021.0446",
                "relationship": "distribution_geometry_inspiration_without_cell_coupling_or_barycentric_projection",
            },
        },
        "v453_manifest_payload_sha256_by_block": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v453_manifests.items()
        },
        "source_sha256": {
            Path(__file__).name: _sha256(Path(__file__).resolve())
        },
        "input_sha256": {
            key: {
                "relaxed_evidence_manifest.json": _sha256(
                    artifacts[key].relaxed_dir / "evidence_manifest.json"
                ),
                "component_scores.tsv": _sha256(
                    artifacts[key].mdcr_dir / "component_scores.tsv"
                ),
                "fold_component_scores.tsv": _sha256(
                    artifacts[key].mdcr_dir / "fold_component_scores.tsv"
                ),
                "assay_axis_assignments.tsv": _sha256(
                    artifacts[key].assignment_path
                ),
            }
            for key in sampled_keys
            if key in artifacts
        },
        "feature_cache_manifest_sha256": cache_manifest_hashes,
        "output_sha256": {
            name: _sha256(out_dir / name) for name in outputs
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worlds-per-block", type=int, default=DEFAULT_WORLDS_PER_BLOCK)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument(
        "--bootstrap-frequency", type=float, default=DEFAULT_BOOTSTRAP_FREQUENCY
    )
    parser.add_argument("--exclude-sample-manifest", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = run_diagnostic(
        output_root=args.output_root,
        formal_worlds_root=args.formal_worlds_root,
        out_dir=args.out_dir,
        worlds_per_block=args.worlds_per_block,
        random_seed=args.random_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_frequency=args.bootstrap_frequency,
        exclude_sample_manifest=args.exclude_sample_manifest,
        feature_cache_dirs=args.feature_cache_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
