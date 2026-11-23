#!/usr/bin/env python3
"""Confirmatory CA-BOSS v4 development and fresh-locked validation runner.

This runner is deliberately narrower than ``benchmark_ca_boss_v4.py``.  It
accepts one configuration frozen by the training-only calibrator, validates an
externally supplied calibration-audit hash, and runs one named candidate through
the benchmark's discovery-only and frozen-scoring APIs. There is no candidate
grid or threshold override.

The registered biological unit is a simulated world.  Crashed and timed-out
worlds therefore remain explicit failed rows rather than disappearing from an
aggregate denominator. All executable/configuration sources are hashed before
discovery. Every registered discovery attempt finishes and is bound into one
self-hashed global manifest before any successful handle may open scorer truth.

The scorer-only background-state upper-bound arm and the frozen-v3 and
production-FCI baselines run only after every frozen candidate output has been
produced. Baseline estimates use the same prepared observable matrices as v4,
are frozen before their diagnostic scorer opens truth, and can never enter
candidate selection. Crashes and timeouts remain explicit baseline rows and
block formal certification without changing the v4 candidate gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
from statistics import NormalDist
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from background_state_upper_bound_v4 import (
    aggregate_registered_upper_bound,
    evaluate_world_upper_bound,
    write_world_upper_bound,
)
from benchmark_ca_boss_v4 import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    DISCOVERY_MANIFEST_SCHEMA_VERSION,
    GLOBAL_PRETRUTH_FREEZE_SCHEMA_VERSION,
    RUN_SUMMARY_SCHEMA_VERSION,
    SCORER_MANIFEST_SCHEMA_VERSION,
    WORLD_INPUT_BINDING_SCHEMA_VERSION,
    FrozenWorldPretruth,
    discover_world_pretruth,
    evaluate_world,
    score_frozen_world,
)
from channel_aware_boss import (
    ChannelAwareConfig,
    discover as discover_frozen_v3,
    is_biological,
)
from channel_aware_boss_v4 import CABossV4Config
from multimodal_simulation_v4 import MODALITIES, design_for_seed


SCHEMA_VERSION = "ca_boss_v4.validation.v2"
DEVELOPMENT_SEEDS = tuple(range(31_001, 31_017))
FRESH_LOCKED_SEEDS = tuple(range(42_001, 42_041))
HISTORICAL_LOCKED_SEEDS = frozenset(range(10_001, 10_041))
TRAINING_SEEDS = tuple(range(30_001, 30_025))
V3_MEAN_BIOLOGICAL_EDGES = 22.725
TARGET_NULL_FPR = 0.10
CONFIDENCE = 0.95
EXPECTED_SENTINEL_NULLS_PER_SCORED_WORLD = 15
CALIBRATION_AUDIT_SCHEMA_VERSION = "ca_boss_v4.calibration.v2"
TRAINING_BINDING_AUDIT_SCHEMA_VERSION = "ca_boss_v4.training_world_bindings.v1"
BASELINE_METHODS = ("frozen_v3", "production_fci")
BASELINE_VIEW_SUPPORT = 2.0 / 3.0
BASELINE_EXPECTED_VIEWS = 6
BASELINE_ALLOWED_ENDPOINTS = frozenset({"TAIL", "ARROW", "CIRCLE"})
PRODUCTION_FCI_BASELINE_CONFIG = {
    "alpha": 0.05,
    "independence_test": "fisherz",
    "max_depth": 2,
    "background_mode": "minimal_expr_sink",
    "confounder_mode": "depth_permark",
    "transform": "rank_gaussian",
}

CLASS_SUCCESS_COLUMNS = (
    "direct_class_success",
    "collider_class_success",
    "shared_class_success",
    "slow_class_success",
    "null_class_success",
    "precision_class_success",
)

RESULT_DEFAULTS: dict[str, float | int] = {
    "direct_adjacency_macro_recall": 0.0,
    "collider_arrowhead_recall": 0.0,
    "strict_reverse_compelled_count": 0,
    "shared_state_adjacency_recall": 0.0,
    "shared_state_false_one_way_count": 0,
    "slow_peak_gene_present": 0,
    "matched_null_false_positive_rate": float("nan"),
    "primary_skeleton_precision": 0.0,
    "primary_skeleton_recall": 0.0,
    "primary_skeleton_f1": 0.0,
    "n_estimated_focal_edges": float("nan"),
    "n_unlabeled_focal_edges": float("nan"),
    "n_unsupported_focal_edges": float("nan"),
    "n_false_typed_focal_edges": float("nan"),
    "density_class_success": 0,
    "exact_pair_shared_class_success": 0,
    "exact_pair_slow_class_success": 0,
    "latent_state_class_success": 0,
    "selected_group_definition_axis_precision": 0.0,
    "selected_hyperedge_axis_precision": 0.0,
    "selected_hyperedge_typed_precision": 0.0,
    "selected_latent_axis_purity_precision": 0.0,
    "selected_latent_axis_axis_precision": 0.0,
    "selected_latent_axis_typed_precision": 0.0,
    "native_latent_axis_dimension_coverage": 0.0,
    "native_slow_rna_axis_support": 0,
    "selected_loading_edge_axis_precision": 0.0,
    "selected_loading_edge_typed_precision": 0.0,
    "combined_within_assay_ps_group_recall": 0.0,
    "combined_cross_assay_latent_axis_coverage": 0.0,
    "combined_slow_gene_axis_support": 0,
    "n_selected_axis_mismatch_claims": 0,
    "n_selected_wrong_layer_claims": 0,
    "world_success": 0,
    **{column: 0 for column in CLASS_SUCCESS_COLUMNS},
}

SOURCE_FILES = {
    "algorithm": "channel_aware_boss_v4.py",
    "hierarchical_sparse_layer": "hierarchical_sparse_edges.py",
    "cell_state_loading_groups": "cell_state_loading_groups.py",
    "loading_group_projection": "loading_group_projection_v4.py",
    "latent_axis_clustering": "latent_axis_clustering_v4.py",
    "stratified_beta_binomial": "stratified_beta_binomial_edges.py",
    "structure_aware_collider_motif": "structure_aware_collider_motif.py",
    "unpaired_state_alignment": "unpaired_state_alignment.py",
    "soft_distributional_alignment": "soft_distributional_alignment.py",
    "observable_input_preparation": "prepare_ca_boss_v4_inputs.py",
    "benchmark_engine": "benchmark_ca_boss_v4.py",
    "typed_consensus": "consensus_ca_boss_v4.py",
    "typed_scorer": "score_multimodal_v4_graph.py",
    "latent_loading_group_scorer": "score_latent_loading_groups_v4.py",
    "simulator": "multimodal_simulation_v4.py",
    "training_calibrator": "calibrate_ca_boss_v4.py",
    "background_state_upper_bound": "background_state_upper_bound_v4.py",
    "formal_validation_runner": "validate_ca_boss_v4.py",
}

BASELINE_SOURCE_FILES = {
    "frozen_v3_algorithm": Path(__file__).with_name("channel_aware_boss.py"),
    "production_fci_algorithm": (
        Path(__file__).resolve().parents[4]
        / "pipelines/empirical/analysis"
        / "scripts"
        / "commands"
        / "run_pc_causallearn.py"
    ),
    "tetrad_bridge_used_by_v3": (
        Path(__file__).resolve().parents[4]
        / "pipelines/empirical/analysis"
        / "scripts"
        / "run_gfci_tetrad.py"
    ),
}

PREREGISTERED_HOOKS: dict[str, dict[str, Any]] = {
    "background_state_upper_bound": {
        "implemented": True,
        "status": "implemented_after_candidate_freeze_and_required_for_certification",
        "required_before_locked_truth": False,
        "required_for_formal_certification": True,
    },
    "frozen_v3_baseline": {
        "implemented": True,
        "status": "implemented_after_candidate_freeze_diagnostic_only",
        "may_influence_v4_selection": False,
        "failures_and_timeouts_retained": True,
        "required_for_formal_certification": True,
    },
    "production_fci_baseline": {
        "implemented": True,
        "status": "implemented_after_candidate_freeze_diagnostic_only",
        "may_influence_v4_selection": False,
        "failures_and_timeouts_retained": True,
        "required_for_formal_certification": True,
    },
}

_SEED_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(?:world|seed)[_-]?(\d{4,8})(?:$|[^0-9])",
    re.IGNORECASE,
)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_DISCOVERY_PARTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
    "counterfactual",
}


class ValidationError(ValueError):
    """Raised before locked truth can be opened when a protocol check fails."""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seeds: tuple[int, ...]
    minimum_world_successes: int
    minimum_class_successes: int
    minimum_factor_level_successes: int
    minimum_design_cell_successes: int
    worlds_per_factor_level: int
    worlds_per_design_cell: int


SPLITS = {
    "development": SplitSpec(
        name="development",
        seeds=DEVELOPMENT_SEEDS,
        minimum_world_successes=14,
        minimum_class_successes=14,
        minimum_factor_level_successes=7,
        minimum_design_cell_successes=2,
        worlds_per_factor_level=8,
        worlds_per_design_cell=2,
    ),
    "locked": SplitSpec(
        name="fresh_locked",
        seeds=FRESH_LOCKED_SEEDS,
        minimum_world_successes=34,
        minimum_class_successes=34,
        minimum_factor_level_successes=16,
        minimum_design_cell_successes=3,
        worlds_per_factor_level=20,
        worlds_per_design_cell=5,
    ),
}


@dataclass(frozen=True)
class FrozenCandidate:
    name: str
    config: CABossV4Config
    config_bytes_sha256: str
    config_canonical_sha256: str
    calibration_audit_bytes_sha256: str
    calibration_audit_payload_sha256: str


@dataclass
class WorldExecution:
    status: str
    rows: list[dict[str, Any]]
    elapsed_seconds: float
    error: str = ""


@dataclass
class DiscoveryExecution:
    status: str
    frozen: FrozenWorldPretruth | None
    elapsed_seconds: float
    error: str = ""


@dataclass(frozen=True)
class BaselineObservableInputs:
    """Discovery-only inputs shared by both post-candidate baselines."""

    world_name: str
    world_seed: int
    matrices: tuple[Path, ...]
    embedding: Path
    chrom_cell_tables: tuple[Path, ...]
    file_sha256: dict[str, str]


@dataclass
class BaselineExecution:
    status: str
    elapsed_seconds: float
    result: dict[str, Any]
    error: str = ""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_expected_hash(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not _HASH_PATTERN.fullmatch(normalized):
        raise ValidationError(f"{label} must be exactly 64 hexadecimal SHA-256 characters")
    return normalized


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} must be a regular, non-symlink file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must contain one JSON object: {path}")
    return value


def _validate_content_addressed_domain(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Verify an exact, non-symlink file set against its manifest hashes."""

    listed_value = manifest.get("files")
    hashes_value = manifest.get("file_sha256")
    if not isinstance(listed_value, list) or not isinstance(hashes_value, Mapping):
        raise ValidationError(f"{label} lacks its content-addressed file contract")
    listed = [str(value) for value in listed_value]
    if len(listed) != len(set(listed)) or set(listed) != set(map(str, hashes_value)):
        raise ValidationError(f"{label} file list and hash keys do not match exactly")
    for relative in listed:
        path_value = Path(relative)
        if path_value.is_absolute() or ".." in path_value.parts or relative == "manifest.json":
            raise ValidationError(f"{label} contains an unsafe manifest path: {relative!r}")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValidationError(f"{label} contains a symlink: {path}")
        if path.is_file():
            relative = str(path.relative_to(root))
            if relative != "manifest.json":
                actual.add(relative)
    if actual != set(listed):
        raise ValidationError(
            f"{label} file set differs from its manifest: "
            f"missing={sorted(set(listed) - actual)}, extra={sorted(actual - set(listed))}"
        )
    for relative in listed:
        expected = _normalize_expected_hash(
            str(hashes_value[relative]), label=f"{label} hash for {relative}"
        )
        if file_sha256(root / relative) != expected:
            raise ValidationError(f"{label} payload hash mismatch: {relative}")


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    claimed = str(value.get(field, "")).lower()
    if not _HASH_PATTERN.fullmatch(claimed):
        raise ValidationError(f"{label} lacks a valid {field}")
    payload = dict(value)
    payload.pop(field, None)
    observed = canonical_sha256(payload)
    if observed != claimed:
        raise ValidationError(f"{label} payload hash mismatch: expected {claimed}, observed {observed}")
    return claimed


def load_frozen_candidate(
    config_path: Path | str,
    calibration_audit_path: Path | str,
    expected_calibration_audit_sha256: str,
) -> FrozenCandidate:
    """Validate one training-only frozen candidate and its content-addressed audit."""

    config_source = Path(config_path)
    audit_source = Path(calibration_audit_path)
    expected_audit_hash = _normalize_expected_hash(
        expected_calibration_audit_sha256,
        label="calibration audit hash",
    )
    observed_audit_hash = file_sha256(audit_source)
    if observed_audit_hash != expected_audit_hash:
        raise ValidationError(
            "Calibration audit file hash mismatch; refusing confirmatory validation"
        )
    config_values = _read_json_object(config_source, label="frozen configuration")
    audit = _read_json_object(audit_source, label="calibration audit")
    audit_payload_hash = _validate_self_hash(
        audit,
        field="audit_payload_sha256",
        label="calibration audit",
    )
    if audit.get("schema_version") != CALIBRATION_AUDIT_SCHEMA_VERSION:
        raise ValidationError("Unsupported calibration audit schema")
    if audit.get("selection_split") != "training_only":
        raise ValidationError("Frozen candidate was not selected on training worlds only")
    if audit.get("registered_training_seed_block") != [30_001, 30_024]:
        raise ValidationError("Calibration audit has the wrong registered training block")
    if tuple(audit.get("world_seeds", ())) != TRAINING_SEEDS:
        raise ValidationError("Calibration audit did not use exactly seeds 30001--30024")
    training_binding = audit.get("training_world_bindings")
    if not isinstance(training_binding, dict):
        raise ValidationError("Calibration audit lacks content-addressed training worlds")
    if training_binding.get("schema_version") != TRAINING_BINDING_AUDIT_SCHEMA_VERSION:
        raise ValidationError("Calibration audit has a stale training-binding schema")
    if training_binding.get("required_for_frozen_candidate_use") is not True:
        raise ValidationError("Training-world bindings are not mandatory for frozen use")
    if (
        training_binding.get("benchmark_bindings_verified_against_current_worlds")
        is not True
    ):
        raise ValidationError("Training-world bindings were not reverified during calibration")
    if training_binding.get("benchmark_schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise ValidationError("Training worlds came from a stale benchmark schema")
    if training_binding.get("world_binding_schema_version") != WORLD_INPUT_BINDING_SCHEMA_VERSION:
        raise ValidationError("Training worlds use a stale content-binding schema")
    bindings = training_binding.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != len(TRAINING_SEEDS):
        raise ValidationError("Calibration audit lacks one binding per training world")
    if int(training_binding.get("world_count", -1)) != len(TRAINING_SEEDS):
        raise ValidationError("Calibration audit has the wrong training-world binding count")
    if tuple(training_binding.get("world_seeds", ())) != TRAINING_SEEDS:
        raise ValidationError("Training-world bindings do not cover seeds 30001--30024")
    if training_binding.get("bindings_canonical_sha256") != canonical_sha256(bindings):
        raise ValidationError("Training-world binding collection hash mismatch")
    bound_seeds: set[int] = set()
    for value in bindings:
        if not isinstance(value, dict):
            raise ValidationError("Calibration audit contains a malformed world binding")
        payload = dict(value)
        claimed = str(payload.pop("binding_payload_sha256", ""))
        if claimed != canonical_sha256(payload):
            raise ValidationError("Training-world binding payload hash mismatch")
        if value.get("schema_version") != WORLD_INPUT_BINDING_SCHEMA_VERSION:
            raise ValidationError("Calibration audit contains a stale world binding")
        try:
            bound_seed = int(value.get("seed", -1))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Training-world binding has an invalid seed") from exc
        if bound_seed not in TRAINING_SEEDS or bound_seed in bound_seeds:
            raise ValidationError("Training-world bindings contain an unexpected or duplicate seed")
        bound_seeds.add(bound_seed)
        for domain, expected_schema in (
            ("run_summary", RUN_SUMMARY_SCHEMA_VERSION),
            ("discovery_manifest", DISCOVERY_MANIFEST_SCHEMA_VERSION),
            ("scorer_manifest", SCORER_MANIFEST_SCHEMA_VERSION),
        ):
            record = value.get(domain)
            if not isinstance(record, dict):
                raise ValidationError(f"Training-world binding lacks {domain}")
            _normalize_expected_hash(
                str(record.get("sha256", "")),
                label=f"training-world {bound_seed} {domain} hash",
            )
            if record.get("schema_version") != expected_schema:
                raise ValidationError(f"Training-world binding has a stale {domain} schema")
            if domain != "run_summary" and record.get(
                "payloads_content_addressed_verified"
            ) is not True:
                raise ValidationError(f"Training-world {domain} payload was not verified")
        if value.get("directory_seed_matches_run_summary") is not True:
            raise ValidationError("Training-world directory was not bound to its run-summary seed")
        if value.get("information_domains_separated") is not True:
            raise ValidationError("Training-world information domains were not separated")
        if value.get("scorer_payload_verification_performed") is not True:
            raise ValidationError("Training-world scorer payload was not verified after discovery")
    if bound_seeds != set(TRAINING_SEEDS):
        raise ValidationError("Training-world binding seed set is incomplete")
    boundary = audit.get("truth_boundary")
    if not isinstance(boundary, dict):
        raise ValidationError("Calibration audit lacks a truth-boundary record")
    if boundary.get("locked_worlds_accessed") is not False:
        raise ValidationError("Calibration audit reports locked-world access")
    if boundary.get("raw_discovery_evidence_loaded_before_truth") is not True:
        raise ValidationError("Calibration audit does not attest the discovery/truth barrier")
    selected = audit.get("selected")
    if not isinstance(selected, dict):
        raise ValidationError("Calibration audit lacks one selected candidate")
    candidate_name = str(selected.get("candidate", ""))
    if not _CANDIDATE_PATTERN.fullmatch(candidate_name) or candidate_name in {".", ".."}:
        raise ValidationError(f"Unsafe or missing frozen candidate name: {candidate_name!r}")
    metrics = selected.get("metrics")
    if not isinstance(metrics, dict) or int(metrics.get("controls_target_null_fpr", 0)) != 1:
        raise ValidationError("Frozen candidate did not pass the registered training null gate")
    config_bytes_hash = file_sha256(config_source)
    config_canonical_hash = canonical_sha256(config_values)
    if selected.get("config_sha256") != config_bytes_hash:
        raise ValidationError("Frozen configuration bytes do not match the calibration audit")
    if selected.get("config_canonical_sha256") != config_canonical_hash:
        raise ValidationError("Frozen configuration values do not match the calibration audit")
    try:
        config = CABossV4Config(**config_values)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid frozen CA-BOSS v4 configuration: {exc}") from exc
    return FrozenCandidate(
        name=candidate_name,
        config=config,
        config_bytes_sha256=config_bytes_hash,
        config_canonical_sha256=config_canonical_hash,
        calibration_audit_bytes_sha256=observed_audit_hash,
        calibration_audit_payload_sha256=audit_payload_hash,
    )


def extract_world_seed(path: Path | str) -> int:
    name = Path(path).name
    matches = sorted({int(value) for value in _SEED_PATTERN.findall(name)})
    if len(matches) != 1:
        raise ValidationError(
            f"World directory name must contain exactly one marked world/seed number: {name!r}"
        )
    seed = matches[0]
    if seed in HISTORICAL_LOCKED_SEEDS:
        raise ValidationError(f"Historical locked seed {seed} is permanently forbidden for v4")
    return seed


def _validate_safe_world_identity(world: Path, seed: int) -> None:
    """Bind a directory label to non-truth simulator identity metadata."""

    summary = _read_json_object(
        world / "run_summary.json", label=f"world {seed} run summary"
    )
    if summary.get("schema_version") != "multimodal_simulation_v4.run.v2":
        raise ValidationError(f"World {seed} does not use simulator run schema v2")
    if int(summary.get("seed", -1)) != seed:
        raise ValidationError(f"World {seed} run-summary seed does not match its directory")
    if summary.get("discovery_directory") != "discovery" or summary.get(
        "scorer_directory"
    ) != "scorer":
        raise ValidationError(f"World {seed} run-summary domains are invalid")
    if summary.get("information_domains_separated_by_directory") is not True:
        raise ValidationError(f"World {seed} lacks separated information domains")
    if summary.get("raw_only_preparer_required_for_discovery") is not True:
        raise ValidationError(f"World {seed} does not require raw-only preparation")
    if summary.get("domain_payloads_content_addressed") is not True:
        raise ValidationError(f"World {seed} lacks content-addressed information domains")
    if int(summary.get("n_modalities", -1)) != len(MODALITIES):
        raise ValidationError(f"World {seed} has the wrong modality count")

    discovery_manifest = _read_json_object(
        world / "discovery" / "manifest.json",
        label=f"world {seed} discovery manifest",
    )
    if discovery_manifest.get("schema_version") != "multimodal_simulation_v4.discovery.v2":
        raise ValidationError(f"World {seed} has an unsupported discovery schema")
    if set(discovery_manifest.get("modalities", ())) != set(MODALITIES):
        raise ValidationError(f"World {seed} discovery modalities do not match v4")
    discovery_manifest_path = world / "discovery" / "manifest.json"
    expected_discovery_manifest_hash = _normalize_expected_hash(
        str(summary.get("discovery_manifest_sha256", "")),
        label=f"world {seed} discovery manifest hash",
    )
    if file_sha256(discovery_manifest_path) != expected_discovery_manifest_hash:
        raise ValidationError(f"World {seed} discovery manifest is not bound to its run summary")
    _validate_content_addressed_domain(
        world / "discovery",
        discovery_manifest,
        label=f"world {seed} discovery domain",
    )
    scorer_manifest_path = world / "scorer" / "manifest.json"
    expected_scorer_manifest_hash = _normalize_expected_hash(
        str(summary.get("scorer_manifest_sha256", "")),
        label=f"world {seed} scorer manifest hash",
    )
    if file_sha256(scorer_manifest_path) != expected_scorer_manifest_hash:
        raise ValidationError(f"World {seed} scorer manifest is not bound to its run summary")


def validate_world_set(
    world_dirs: Sequence[Path | str],
    spec: SplitSpec,
) -> dict[int, Path]:
    """Require the entire registered split, with no historical, mixed, or duplicate worlds."""

    if not world_dirs:
        raise ValidationError(f"No {spec.name} worlds were supplied")
    worlds: dict[int, Path] = {}
    for value in world_dirs:
        path = Path(value).absolute()
        if path.is_symlink() or not path.is_dir():
            raise ValidationError(f"World must be a regular directory, not a symlink: {path}")
        seed = extract_world_seed(path)
        if seed not in spec.seeds:
            raise ValidationError(
                f"Seed {seed} is not in the registered {spec.name} block "
                f"{spec.seeds[0]}--{spec.seeds[-1]}"
            )
        if seed in worlds:
            raise ValidationError(f"Duplicate world seed {seed}")
        discovery = path / "discovery"
        scorer = path / "scorer"
        if not discovery.is_dir() or not scorer.is_dir():
            raise ValidationError(f"World {seed} lacks separate discovery/ and scorer/ domains")
        if discovery.is_symlink() or scorer.is_symlink():
            raise ValidationError(f"World {seed} information domains may not be symlinks")
        discovery_resolved = discovery.resolve(strict=True)
        scorer_resolved = scorer.resolve(strict=True)
        if discovery_resolved == scorer_resolved:
            raise ValidationError(f"World {seed} discovery and scorer domains resolve together")
        _validate_safe_world_identity(path, seed)
        truth = scorer / "truth_typed_edges.tsv"
        if truth.is_symlink() or not truth.is_file() or truth.resolve().parent != scorer_resolved:
            raise ValidationError(f"World {seed} lacks a direct scorer-only truth table")
        worlds[seed] = path
    expected = set(spec.seeds)
    if set(worlds) != expected:
        missing = sorted(expected - set(worlds))
        unexpected = sorted(set(worlds) - expected)
        raise ValidationError(
            f"Worlds do not exactly match registered {spec.name} seeds; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return dict(sorted(worlds.items()))


def source_hash_bundle() -> dict[str, dict[str, str]]:
    """Hash every code/preregistration artifact that can affect validation."""

    scripts = Path(__file__).resolve().parent
    records: dict[str, dict[str, str]] = {}
    for role, filename in SOURCE_FILES.items():
        path = scripts / filename
        if not path.is_file():
            raise ValidationError(f"Required source is missing: {path}")
        records[role] = {"path": str(path), "sha256": file_sha256(path)}
    for role, path in BASELINE_SOURCE_FILES.items():
        if not path.is_file():
            raise ValidationError(f"Required baseline source is missing: {path}")
        records[role] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    preregistration = scripts.parent / "results" / "channel_aware_boss_v4_experiment_log.md"
    if not preregistration.is_file():
        raise ValidationError(f"Required preregistration is missing: {preregistration}")
    records["preregistration"] = {
        "path": str(preregistration.resolve()),
        "sha256": file_sha256(preregistration),
    }
    return records


def _source_digest(bundle: Mapping[str, Mapping[str, str]]) -> str:
    content_hashes = {
        role: str(record["sha256"])
        for role, record in sorted(bundle.items())
    }
    return canonical_sha256(content_hashes)


def validate_development_manifest(
    path: Path | str,
    expected_file_sha256: str,
    *,
    frozen: FrozenCandidate,
    current_source_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate a content-addressed, passing DEV manifest before LOCKED."""

    source = Path(path)
    expected_hash = _normalize_expected_hash(
        expected_file_sha256,
        label="development manifest hash",
    )
    if file_sha256(source) != expected_hash:
        raise ValidationError("Development manifest file hash mismatch")
    manifest = _read_json_object(source, label="development validation manifest")
    _validate_self_hash(
        manifest,
        field="manifest_payload_sha256",
        label="development validation manifest",
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("Unsupported development validation schema")
    if manifest.get("split") != "development":
        raise ValidationError("LOCKED requires a DEVELOPMENT validation manifest")
    if manifest.get("candidate_gate_pass") is not True:
        raise ValidationError("Development candidate gate did not pass")
    if manifest.get("world_seeds") != list(DEVELOPMENT_SEEDS):
        raise ValidationError("Development manifest does not cover seeds 31001--31016")
    selected = manifest.get("frozen_candidate")
    if not isinstance(selected, dict):
        raise ValidationError("Development manifest lacks frozen-candidate provenance")
    expected_candidate = {
        "name": frozen.name,
        "config_bytes_sha256": frozen.config_bytes_sha256,
        "config_canonical_sha256": frozen.config_canonical_sha256,
        "calibration_audit_bytes_sha256": frozen.calibration_audit_bytes_sha256,
        "calibration_audit_payload_sha256": frozen.calibration_audit_payload_sha256,
    }
    for key, value in expected_candidate.items():
        if selected.get(key) != value:
            raise ValidationError(f"Development manifest frozen-candidate mismatch: {key}")
    recorded_sources = manifest.get("source_freeze", {}).get("files")
    if recorded_sources != current_source_hashes:
        raise ValidationError(
            "Executable sources changed after development validation; a new DEV gate is required"
        )
    if manifest.get("truth_boundary", {}).get("sources_hashed_before_truth") is not True:
        raise ValidationError("Development manifest lacks the pre-truth source freeze")
    if manifest.get("background_state_upper_bound", {}).get("pass") is not True:
        raise ValidationError(
            "Development background-state upper-bound gate did not pass"
        )
    if manifest.get("post_candidate_baseline_diagnostics", {}).get("pass") is not True:
        raise ValidationError(
            "Development post-candidate baseline diagnostics were incomplete"
        )
    if manifest.get("formal_certification_complete") is not True:
        raise ValidationError("Development formal certification was incomplete")
    return manifest


def _discovery_worker(
    connection: Any,
    world_dir: Path,
    candidate_name: str,
    config: CABossV4Config,
    evaluation_out: Path,
    discoverer: Callable[..., FrozenWorldPretruth],
) -> None:
    try:
        frozen = discoverer(
            world_dir,
            [(candidate_name, config)],
            out_dir=evaluation_out,
            save_candidate_names={candidate_name},
        )
        if not isinstance(frozen, FrozenWorldPretruth):
            raise TypeError("Discovery callable did not return FrozenWorldPretruth")
        connection.send(("ok", asdict(frozen), ""))
    except BaseException:  # noqa: BLE001 - child failures must become denominator rows
        connection.send(("crash", None, traceback.format_exc(limit=40)))
    finally:
        connection.close()


def discover_world_with_timeout(
    world_dir: Path,
    candidate_name: str,
    config: CABossV4Config,
    evaluation_out: Path,
    timeout_seconds: float,
    *,
    discoverer: Callable[..., FrozenWorldPretruth] = discover_world_pretruth,
) -> DiscoveryExecution:
    """Freeze one world's discovery artifacts in a killable truth-free child."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValidationError("World timeout must be a positive finite number")
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else methods[0])
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_discovery_worker,
        args=(send, world_dir, candidate_name, config, evaluation_out, discoverer),
    )
    started = time.monotonic()
    process.start()
    send.close()
    try:
        if not receive.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            return DiscoveryExecution(
                status="timeout",
                frozen=None,
                elapsed_seconds=time.monotonic() - started,
                error=f"Exceeded registered per-world timeout of {timeout_seconds:g} seconds",
            )
        try:
            status, frozen_payload, error = receive.recv()
        except EOFError:
            status, frozen_payload, error = (
                "crash",
                None,
                f"Worker exited without a result (exitcode={process.exitcode})",
            )
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        frozen: FrozenWorldPretruth | None = None
        if str(status) == "ok":
            try:
                if not isinstance(frozen_payload, dict):
                    raise TypeError("missing serialized frozen handle")
                frozen = FrozenWorldPretruth(**frozen_payload)
            except (TypeError, ValueError) as exc:
                status = "invalid_result"
                error = f"Invalid frozen discovery handle: {exc}"
        return DiscoveryExecution(
            status=str(status),
            frozen=frozen,
            elapsed_seconds=time.monotonic() - started,
            error=str(error),
        )
    finally:
        receive.close()


def _score_worker(
    connection: Any,
    frozen: FrozenWorldPretruth,
    global_freeze_manifest: Path,
    scorer: Callable[..., list[dict[str, Any]]],
) -> None:
    try:
        rows = scorer(
            frozen,
            global_freeze_manifest=global_freeze_manifest,
        )
        connection.send(("ok", rows, ""))
    except BaseException:  # noqa: BLE001 - child failures must become denominator rows
        connection.send(("crash", [], traceback.format_exc(limit=40)))
    finally:
        connection.close()


def score_world_with_timeout(
    frozen: FrozenWorldPretruth,
    global_freeze_manifest: Path,
    timeout_seconds: float,
    *,
    scorer: Callable[..., list[dict[str, Any]]] = score_frozen_world,
) -> WorldExecution:
    """Score one frozen world in a killable child after the global barrier."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValidationError("World timeout must be a positive finite number")
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else methods[0])
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_score_worker,
        args=(send, frozen, global_freeze_manifest, scorer),
    )
    started = time.monotonic()
    process.start()
    send.close()
    try:
        if not receive.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            return WorldExecution(
                status="timeout",
                rows=[],
                elapsed_seconds=time.monotonic() - started,
                error=(
                    "Exceeded registered per-world scoring timeout of "
                    f"{timeout_seconds:g} seconds"
                ),
            )
        try:
            status, rows, error = receive.recv()
        except EOFError:
            status, rows, error = (
                "crash",
                [],
                f"Scoring worker exited without a result (exitcode={process.exitcode})",
            )
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        normalized_rows: list[dict[str, Any]] = []
        if str(status) == "ok":
            if not isinstance(rows, list) or not all(
                isinstance(record, dict) for record in rows
            ):
                status = "invalid_result"
                error = "Scoring callable did not return a list of row dictionaries"
            else:
                normalized_rows = rows
        return WorldExecution(
            status=str(status),
            rows=normalized_rows,
            elapsed_seconds=time.monotonic() - started,
            error=str(error),
        )
    finally:
        receive.close()


def _world_worker(
    connection: Any,
    world_dir: Path,
    candidate_name: str,
    config: CABossV4Config,
    evaluation_out: Path,
    evaluator: Callable[..., list[dict[str, Any]]],
) -> None:
    try:
        rows = evaluator(
            world_dir,
            [(candidate_name, config)],
            out_dir=evaluation_out,
            save_candidate_names={candidate_name},
        )
        connection.send(("ok", rows, ""))
    except BaseException:  # noqa: BLE001 - child failures must become denominator rows
        connection.send(("crash", [], traceback.format_exc(limit=40)))
    finally:
        connection.close()


def evaluate_world_with_timeout(
    world_dir: Path,
    candidate_name: str,
    config: CABossV4Config,
    evaluation_out: Path,
    timeout_seconds: float,
    *,
    evaluator: Callable[..., list[dict[str, Any]]] = evaluate_world,
) -> WorldExecution:
    """Run one world in a killable child and return timeout/crash as data."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValidationError("World timeout must be a positive finite number")
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else methods[0])
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_world_worker,
        args=(send, world_dir, candidate_name, config, evaluation_out, evaluator),
    )
    started = time.monotonic()
    process.start()
    send.close()
    try:
        if not receive.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            return WorldExecution(
                status="timeout",
                rows=[],
                elapsed_seconds=time.monotonic() - started,
                error=f"Exceeded registered per-world timeout of {timeout_seconds:g} seconds",
            )
        try:
            status, rows, error = receive.recv()
        except EOFError:
            status, rows, error = (
                "crash",
                [],
                f"Worker exited without a result (exitcode={process.exitcode})",
            )
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        return WorldExecution(
            status=str(status),
            rows=list(rows),
            elapsed_seconds=time.monotonic() - started,
            error=str(error),
        )
    finally:
        receive.close()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def audit_world_output(
    world_dir: Path,
    evaluation_out: Path,
    candidate_name: str,
) -> dict[str, Any]:
    """Verify the single pre-truth consensus and discovery-only input provenance."""

    world_out = evaluation_out / world_dir.name
    candidate_out = world_out / candidate_name
    manifest_path = world_out / "pretruth_consensus_manifest.json"
    consensus_path = candidate_out / "consensus_edges.tsv"
    pair_scores_path = candidate_out / "typed_pair_scores.tsv"
    projection_dir = candidate_out / "loading_group_projection"
    group_hyperedges_path = projection_dir / "typed_group_hyperedges.tsv"
    group_pairs_path = projection_dir / "sparse_pair_projection.tsv"
    axis_dir = candidate_out / "latent_axis_projection"
    axis_hyperedges_path = axis_dir / "typed_axis_hyperedges.tsv"
    axis_provenance_path = axis_dir / "axis_provenance.json"
    axis_score_path = candidate_out / "latent_axis_hyperedge_scores.tsv"
    latent_metrics_path = candidate_out / "latent_group_metrics.json"
    reasons: list[str] = []
    try:
        seed = extract_world_seed(world_dir)
        _validate_safe_world_identity(world_dir, seed)
        parameters = _read_json_object(
            world_dir / "scorer" / "world_parameters.json",
            label=f"world {seed} scorer parameters",
        )
        scorer_manifest = _read_json_object(
            world_dir / "scorer" / "manifest.json",
            label=f"world {seed} scorer manifest",
        )
        if parameters.get("schema_version") != "multimodal_simulation_v4.parameters.v2":
            reasons.append("scorer parameters do not use simulator schema v2")
        if int(parameters.get("seed", -1)) != seed:
            reasons.append("scorer-parameter seed does not match the world directory")
        expected_design = design_for_seed(seed)
        observed_design = parameters.get("design")
        identity_fields = (
            "half_fraction_cell",
            "half_fraction_index_zero_based",
            "A_cells",
            "B_detection",
            "C_population_divergence",
            "D_background_observability",
            "base_cells",
            "background_integration_stress",
        )
        if not isinstance(observed_design, Mapping):
            reasons.append("scorer parameters lack world-design metadata")
        else:
            mismatched = [
                field
                for field in identity_fields
                if observed_design.get(field) != expected_design.get(field)
            ]
            if mismatched:
                reasons.append(
                    "scorer world design does not match its registered seed: "
                    f"{sorted(mismatched)}"
                )
            if observed_design.get("background_observability_stress_override") is not None:
                reasons.append("confirmatory world uses a background-stress override")
            if observed_design.get("alignment_derangement_diagnostic") is not False:
                reasons.append("confirmatory world enables alignment derangement diagnostic")
        if scorer_manifest.get("schema_version") != "multimodal_simulation_v4.scorer.v2":
            reasons.append("scorer manifest does not use simulator schema v2")
        _validate_content_addressed_domain(
            world_dir / "scorer",
            scorer_manifest,
            label=f"world {seed} scorer domain",
        )
        required_scorer_files = {
            "truth_typed_edges.tsv",
            "role_assignments.tsv",
            "generative_dag_nodes.tsv",
            "generative_dag_edges.tsv",
            "world_parameters.json",
        }
        if not required_scorer_files.issubset(set(scorer_manifest.get("files", ()))):
            reasons.append("scorer manifest lacks required truth artifacts")
    except (ValidationError, TypeError, ValueError) as exc:
        reasons.append(f"world identity audit failed: {exc}")
    pretruth: dict[str, Any] = {}
    try:
        pretruth = _read_json_object(manifest_path, label="pre-truth consensus manifest")
    except ValidationError as exc:
        reasons.append(str(exc))
    if pretruth:
        if pretruth.get("candidate_count") != 1:
            reasons.append("pre-truth manifest did not contain exactly one candidate")
        if pretruth.get("all_consensuses_built_before_truth_open") is not True:
            reasons.append("pre-truth consensus barrier was not attested")
        hashes = pretruth.get("consensus_sha256")
        if not isinstance(hashes, dict) or set(hashes) != {candidate_name}:
            reasons.append("pre-truth manifest candidate hash set is not singleton")
        elif not consensus_path.is_file():
            reasons.append("saved frozen consensus is missing")
        elif file_sha256(consensus_path) != hashes[candidate_name]:
            reasons.append("saved consensus differs from its pre-truth hash")
        projection_hashes = pretruth.get("loading_group_projection_sha256")
        if (
            not isinstance(projection_hashes, dict)
            or set(projection_hashes) != {candidate_name}
        ):
            reasons.append("pre-truth loading-group hash set is not singleton")
        elif not group_hyperedges_path.is_file() or not group_pairs_path.is_file():
            reasons.append("saved loading-group projection is missing")
        else:
            projection_digest = hashlib.sha256(
                (
                    group_hyperedges_path.read_text()
                    + group_pairs_path.read_text()
                ).encode("utf-8")
            ).hexdigest()
            if projection_digest != projection_hashes[candidate_name]:
                reasons.append("saved loading-group projection differs from its pre-truth hash")
        if pretruth.get("preliminary_pairwise_state_layer_replaced") is not True:
            reasons.append("pairwise state layer was not replaced by the group consensus")
        for attestation, message in (
            (
                "latent_axis_hyperedges_built_before_truth_open",
                "latent-axis hyperedges were not frozen before truth access",
            ),
            (
                "latent_axis_hyperedges_attached_once_post_consensus",
                "latent-axis hyperedges were not attached exactly once",
            ),
            (
                "ordinary_state_skeleton_contains_within_assay_groups_only",
                "ordinary state skeleton was not restricted to local groups",
            ),
            (
                "pairwise_cross_group_matches_are_diagnostics_only",
                "pairwise cross-group matches were not restricted to diagnostics",
            ),
            (
                "loading_views_reuse_same_observed_cells",
                "loading views were not identified as same-cell refits",
            ),
            (
                "loading_views_are_algorithmic_stability_repeats",
                "loading views were not identified as algorithmic stability repeats",
            ),
            (
                "loading_view_support_not_world_sample_size",
                "loading-view support was not disclaimed as a world sample size",
            ),
        ):
            if pretruth.get(attestation) is not True:
                reasons.append(message)
        if pretruth.get("latent_axis_hyperedges_expanded_into_pairwise_skeleton") is not False:
            reasons.append("latent-axis hyperedges were expanded into pairwise skeleton edges")
        if pretruth.get("loading_views_are_independent_worlds") is not False:
            reasons.append("loading views were incorrectly represented as independent worlds")
        axis_hashes = pretruth.get("latent_axis_projection_sha256")
        if not isinstance(axis_hashes, dict) or set(axis_hashes) != {candidate_name}:
            reasons.append("pre-truth latent-axis result hash set is not singleton")
        axis_file_hashes = pretruth.get("latent_axis_projection_file_sha256")
        if (
            not isinstance(axis_file_hashes, dict)
            or set(axis_file_hashes) != {candidate_name}
            or not isinstance(axis_file_hashes.get(candidate_name), dict)
        ):
            reasons.append("pre-truth latent-axis file hash set is not singleton")
        else:
            expected_axis_files = {
                "group_pair_view_evidence.tsv",
                "group_pair_evidence.tsv",
                "axis_candidates.tsv",
                "group_axis_view_evidence.tsv",
                "group_axis_evidence.tsv",
                "typed_axis_hyperedges.tsv",
                "representative_evidence.tsv",
                "sparse_pair_projection.tsv",
                "axis_provenance.json",
            }
            recorded = axis_file_hashes[candidate_name]
            if set(recorded) != expected_axis_files:
                reasons.append("saved latent-axis artifact set is incomplete")
            else:
                for filename in sorted(expected_axis_files):
                    path = axis_dir / filename
                    if not path.is_file():
                        reasons.append(f"saved latent-axis artifact is missing: {filename}")
                    elif file_sha256(path) != recorded[filename]:
                        reasons.append(
                            f"saved latent-axis artifact differs from its pre-truth hash: {filename}"
                        )
        if not axis_hyperedges_path.is_file() or not axis_provenance_path.is_file():
            reasons.append("saved latent-axis projection is missing")

    if not latent_metrics_path.is_file():
        reasons.append("latent loading-group score is missing")
    if not axis_score_path.is_file():
        reasons.append("latent-axis hyperedge score is missing")

    allowed_roots = [world_dir / "discovery", world_out / "prepared_discovery"]
    summaries = sorted(world_out.glob("cells*_seed*/discovery_summary.json"))
    if len(summaries) != 6:
        reasons.append(f"expected six discovery summaries, found {len(summaries)}")
    for summary_path in summaries:
        try:
            summary = _read_json_object(summary_path, label="discovery summary")
        except ValidationError as exc:
            reasons.append(str(exc))
            continue
        if summary.get("method") != "ca_boss_v4_explicit_state":
            reasons.append(f"unexpected discovery method in {summary_path}")
        inputs = summary.get("inputs")
        provenance = summary.get("provenance")
        if not isinstance(inputs, dict) or not isinstance(provenance, dict):
            reasons.append(f"missing discovery input provenance in {summary_path}")
            continue
        paths: list[str] = []
        for field in ("metacell_matrix", "embedding_table", "membership_table"):
            paths.append(str(inputs.get(field, "")))
        paths.extend(str(value) for value in inputs.get("cell_tables", []))
        paths.extend(
            str(provenance.get(field, ""))
            for field in ("embedding_input", "membership_input")
        )
        for raw in paths:
            path = Path(raw)
            lowered = {part.casefold() for part in path.parts}
            if not raw or lowered & _FORBIDDEN_DISCOVERY_PARTS:
                reasons.append(f"forbidden discovery input recorded in {summary_path}: {raw}")
                continue
            if not any(_is_below(path, root) for root in allowed_roots):
                reasons.append(f"discovery input escaped allowed domains: {raw}")

    null_false = 0
    null_total = 0
    if not pair_scores_path.is_file():
        reasons.append("typed pair scores are missing")
    else:
        try:
            pair_scores = pd.read_csv(pair_scores_path, sep="\t", keep_default_na=False)
            required = {"truth_type", "adjacency_present"}
            if not required <= set(pair_scores.columns):
                reasons.append("typed pair scores lack null-audit columns")
            else:
                nulls = pair_scores.loc[pair_scores["truth_type"].eq("sentinel_null")]
                null_total = int(len(nulls))
                values = pd.to_numeric(nulls["adjacency_present"], errors="coerce")
                if (
                    null_total != EXPECTED_SENTINEL_NULLS_PER_SCORED_WORLD
                    or values.isna().any()
                    or not set(values.astype(int)).issubset({0, 1})
                ):
                    reasons.append(
                        "typed pair scores do not contain exactly 15 binary sentinel-null events"
                    )
                else:
                    null_false = int(values.sum())
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            reasons.append(f"could not audit typed pair scores: {exc}")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "pretruth_manifest": str(manifest_path),
        "consensus": str(consensus_path),
        "consensus_sha256": file_sha256(consensus_path) if consensus_path.is_file() else "",
        "null_false_adjacencies": null_false,
        "null_adjacency_opportunities": null_total,
    }


def _failure_result(
    seed: int,
    world: Path,
    candidate: str,
    execution: WorldExecution,
) -> dict[str, Any]:
    design = design_for_seed(seed)
    return {
        "world": world.name,
        "world_seed": seed,
        "candidate": candidate,
        "execution_status": execution.status,
        "execution_error": execution.error,
        "elapsed_seconds": execution.elapsed_seconds,
        "provenance_audit_pass": 0,
        "provenance_violation": 0,
        "validation_world_success": 0,
        "null_false_adjacencies": 0,
        "null_adjacency_opportunities": 0,
        "design_cell": int(design["half_fraction_cell"]),
        **{
            field: int(design[field])
            for field in (
                "A_cells",
                "B_detection",
                "C_population_divergence",
                "D_background_observability",
            )
        },
        **RESULT_DEFAULTS,
    }


def _successful_result(
    seed: int,
    world: Path,
    candidate: str,
    execution: WorldExecution,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    if len(execution.rows) != 1:
        raise ValidationError(
            f"World {seed} returned {len(execution.rows)} rows; exactly one frozen candidate is required"
        )
    row = dict(execution.rows[0])
    if str(row.get("candidate", "")) != candidate:
        raise ValidationError(f"World {seed} returned the wrong candidate")
    if str(row.get("world", "")) != world.name:
        raise ValidationError(f"World {seed} returned the wrong world label")
    missing = set(RESULT_DEFAULTS) - set(row)
    if missing:
        raise ValidationError(f"World {seed} score row lacks required metrics: {sorted(missing)}")
    for field in (*CLASS_SUCCESS_COLUMNS, "density_class_success", "world_success"):
        value = int(row[field])
        if value not in (0, 1):
            raise ValidationError(f"World {seed} has non-binary {field}: {row[field]!r}")
        row[field] = value
    design = design_for_seed(seed)
    provenance_pass = int(bool(audit.get("pass")))
    row.update(
        {
            "world_seed": seed,
            "execution_status": "ok",
            "execution_error": "",
            "elapsed_seconds": execution.elapsed_seconds,
            "provenance_audit_pass": provenance_pass,
            "provenance_violation": int(not provenance_pass),
            "validation_world_success": int(bool(row["world_success"]) and provenance_pass),
            "null_false_adjacencies": int(audit.get("null_false_adjacencies", 0)),
            "null_adjacency_opportunities": int(
                audit.get("null_adjacency_opportunities", 0)
            ),
            "design_cell": int(design["half_fraction_cell"]),
            **{
                field: int(design[field])
                for field in (
                    "A_cells",
                    "B_detection",
                    "C_population_divergence",
                    "D_background_observability",
                )
            },
        }
    )
    return row


def wilson_upper(successes: int, total: int, *, confidence: float = CONFIDENCE) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson inputs require 0 <= successes <= total and total > 0")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and one")
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float(min(1.0, (center + radius) / denominator))


def aggregate_gates(results: pd.DataFrame, spec: SplitSpec) -> dict[str, Any]:
    """Apply registered world, class, null, density, factor, and design-cell gates."""

    if len(results) != len(spec.seeds) or set(results["world_seed"].astype(int)) != set(spec.seeds):
        raise ValidationError("Aggregate input does not contain exactly one row per registered world")
    if results["world_seed"].duplicated().any():
        raise ValidationError("Aggregate input contains duplicate world rows")
    success = pd.to_numeric(results["validation_world_success"], errors="raise").astype(int)
    if not set(success).issubset({0, 1}):
        raise ValidationError("validation_world_success must be binary")
    world_successes = int(success.sum())
    world_gate = {
        "successes": world_successes,
        "total": len(spec.seeds),
        "required": spec.minimum_world_successes,
        "pass": world_successes >= spec.minimum_world_successes,
    }

    class_details: dict[str, Any] = {}
    for column in CLASS_SUCCESS_COLUMNS:
        values = pd.to_numeric(results[column], errors="raise").astype(int)
        if not set(values).issubset({0, 1}):
            raise ValidationError(f"{column} must be binary")
        count = int(values.sum())
        class_details[column] = {
            "successes": count,
            "total": len(spec.seeds),
            "required": spec.minimum_class_successes,
            "pass": count >= spec.minimum_class_successes,
        }
    class_gate = {
        "classes": class_details,
        "pass": all(value["pass"] for value in class_details.values()),
    }

    completed = results.loc[results["execution_status"].eq("ok")]
    density = pd.to_numeric(completed["n_estimated_focal_edges"], errors="coerce")
    density = density.loc[np.isfinite(density)]
    mean_density = float(density.mean()) if len(density) else float("nan")
    per_world_density = pd.to_numeric(
        completed["density_class_success"], errors="coerce"
    ).fillna(0).astype(int)
    density_gate = {
        "mean_estimated_focal_edges": mean_density if math.isfinite(mean_density) else None,
        "v3_reference_mean": V3_MEAN_BIOLOGICAL_EDGES,
        "requires_strict_improvement": True,
        "completed_worlds": int(len(density)),
        "per_world_density_successes": int(per_world_density.sum()),
        "pass": bool(
            len(density) > 0
            and mean_density < V3_MEAN_BIOLOGICAL_EDGES
            and int(per_world_density.sum()) == len(completed)
        ),
    }

    null_false = int(pd.to_numeric(completed["null_false_adjacencies"], errors="raise").sum())
    null_total = int(
        pd.to_numeric(completed["null_adjacency_opportunities"], errors="raise").sum()
    )
    null_upper = wilson_upper(null_false, null_total) if null_total else 1.0
    null_gate = {
        "false_adjacencies": null_false,
        "opportunities": null_total,
        "one_sided_wilson_upper_95": null_upper,
        "maximum": TARGET_NULL_FPR,
        "pass": null_total > 0 and null_upper <= TARGET_NULL_FPR,
    }

    provenance_violations = int(
        pd.to_numeric(results["provenance_violation"], errors="raise").sum()
    )
    provenance_gate = {
        "audited_completed_worlds": int(
            pd.to_numeric(completed["provenance_audit_pass"], errors="raise").sum()
        ),
        "completed_worlds": int(len(completed)),
        "violations": provenance_violations,
        "pass": provenance_violations == 0,
    }

    factor_details: dict[str, Any] = {}
    for factor in (
        "A_cells",
        "B_detection",
        "C_population_divergence",
        "D_background_observability",
    ):
        levels: dict[str, Any] = {}
        for level in (-1, 1):
            subset = results.loc[pd.to_numeric(results[factor]).eq(level)]
            count = int(subset["validation_world_success"].sum())
            levels[str(level)] = {
                "successes": count,
                "total": int(len(subset)),
                "required": spec.minimum_factor_level_successes,
                "pass": bool(
                    len(subset) == spec.worlds_per_factor_level
                    and count >= spec.minimum_factor_level_successes
                ),
            }
        factor_details[factor] = levels
    factor_gate = {
        "levels": factor_details,
        "pass": all(
            level["pass"]
            for levels in factor_details.values()
            for level in levels.values()
        ),
    }

    design_cells: dict[str, Any] = {}
    for cell in range(1, 9):
        subset = results.loc[pd.to_numeric(results["design_cell"]).eq(cell)]
        count = int(subset["validation_world_success"].sum())
        design_cells[str(cell)] = {
            "successes": count,
            "total": int(len(subset)),
            "required": spec.minimum_design_cell_successes,
            "pass": bool(
                len(subset) == spec.worlds_per_design_cell
                and count >= spec.minimum_design_cell_successes
            ),
        }
    design_gate = {
        "cells": design_cells,
        "pass": all(value["pass"] for value in design_cells.values()),
    }

    execution_counts = {
        str(key): int(value)
        for key, value in results["execution_status"].value_counts().sort_index().items()
    }
    gates = {
        "execution_denominator": {
            "registered_worlds": len(spec.seeds),
            "status_counts": execution_counts,
            "failures_are_retained": True,
        },
        "world_gate": world_gate,
        "class_gate": class_gate,
        "density_gate": density_gate,
        "pooled_null_gate": null_gate,
        "provenance_gate": provenance_gate,
        "factor_level_gate": factor_gate,
        "design_cell_gate": design_gate,
    }
    gates["pass"] = all(
        gates[key]["pass"]
        for key in (
            "world_gate",
            "class_gate",
            "density_gate",
            "pooled_null_gate",
            "provenance_gate",
            "factor_level_gate",
            "design_cell_gate",
        )
    )
    return gates


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def collect_baseline_observable_inputs(
    world_dir: Path,
    evaluation_out: Path,
) -> BaselineObservableInputs:
    """Bind both baselines to the discovery inputs already prepared for v4."""

    seed = extract_world_seed(world_dir)
    prepared = evaluation_out / world_dir.name / "prepared_discovery"
    embedding = prepared / "analysis_embeddings.tsv"
    matrices = tuple(
        sorted(prepared.glob("nested/*/observed_metacell_matrix.tsv"))
    )
    chrom_cell_tables = tuple(
        sorted((world_dir / "discovery").glob("observed_focal__chrom_*.tsv"))
    )
    if len(matrices) != BASELINE_EXPECTED_VIEWS:
        raise ValidationError(
            f"World {seed} baseline expected {BASELINE_EXPECTED_VIEWS} prepared "
            f"matrices, found {len(matrices)}"
        )
    if len(chrom_cell_tables) != len(MODALITIES) - 1:
        raise ValidationError(
            f"World {seed} baseline expected {len(MODALITIES) - 1} chromatin "
            f"cell tables, found {len(chrom_cell_tables)}"
        )
    paths = (embedding, *matrices, *chrom_cell_tables)
    allowed_roots = (prepared, world_dir / "discovery")
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"Unsafe or missing baseline observable input: {path}")
        lowered = {part.casefold() for part in path.parts}
        if lowered & _FORBIDDEN_DISCOVERY_PARTS:
            raise ValidationError(f"Baseline observable input entered a forbidden domain: {path}")
        if not any(_is_below(path, root) for root in allowed_roots):
            raise ValidationError(f"Baseline observable input escaped allowed roots: {path}")
        if _is_below(path, prepared):
            label = f"prepared_discovery/{path.relative_to(prepared)}"
        else:
            label = f"discovery/{path.relative_to(world_dir / 'discovery')}"
        hashes[label] = file_sha256(path)
    return BaselineObservableInputs(
        world_name=world_dir.name,
        world_seed=seed,
        matrices=matrices,
        embedding=embedding,
        chrom_cell_tables=chrom_cell_tables,
        file_sha256=dict(sorted(hashes.items())),
    )


def _canonical_baseline_record(
    record: Mapping[str, Any],
) -> tuple[tuple[str, str], tuple[str, str]]:
    left, right = str(record["node1"]), str(record["node2"])
    if left <= right:
        return (left, right), (
            str(record.get("endpoint1", "CIRCLE")),
            str(record.get("endpoint2", "CIRCLE")),
        )
    return (right, left), (
        str(record.get("endpoint2", "CIRCLE")),
        str(record.get("endpoint1", "CIRCLE")),
    )


def consensus_baseline_graph(
    views: Sequence[pd.DataFrame],
    *,
    minimum_frequency: float = BASELINE_VIEW_SUPPORT,
) -> pd.DataFrame:
    """Apply the frozen two-thirds v3 rule over the six prepared v4 views."""

    if len(views) != BASELINE_EXPECTED_VIEWS:
        raise ValidationError(
            f"Baseline consensus requires exactly {BASELINE_EXPECTED_VIEWS} views"
        )
    adjacency_counts: Counter[tuple[str, str]] = Counter()
    endpoint1_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    endpoint2_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    interpretations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for frame in views:
        if not {"node1", "node2", "endpoint1", "endpoint2"} <= set(frame.columns):
            if frame.empty:
                continue
            raise ValidationError("Baseline view lacks canonical edge columns")
        seen: set[tuple[str, str]] = set()
        for record in frame.to_dict(orient="records"):
            key, endpoints = _canonical_baseline_record(record)
            invalid_endpoints = set(endpoints) - BASELINE_ALLOWED_ENDPOINTS
            if invalid_endpoints:
                raise ValidationError(
                    "Baseline view contains invalid endpoints: "
                    f"{sorted(invalid_endpoints)}"
                )
            if key in seen:
                continue
            seen.add(key)
            adjacency_counts[key] += 1
            endpoint1_votes[key][endpoints[0]] += 1
            endpoint2_votes[key][endpoints[1]] += 1
            evidence[key].update(
                token
                for token in str(record.get("evidence_layers", "")).split(",")
                if token
            )
            interpretation = str(record.get("interpretation", "")).strip()
            if interpretation:
                interpretations[key].add(interpretation)
    required = max(1, math.ceil(minimum_frequency * len(views) - 1e-12))
    rows: list[dict[str, Any]] = []
    for key, count in sorted(adjacency_counts.items()):
        if count < required:
            continue
        selected_endpoints: list[str] = []
        for votes in (endpoint1_votes[key], endpoint2_votes[key]):
            endpoint, endpoint_count = votes.most_common(1)[0]
            selected_endpoints.append(
                endpoint if endpoint_count / count >= 2.0 / 3.0 else "CIRCLE"
            )
        endpoints = (selected_endpoints[0], selected_endpoints[1])
        rows.append(
            {
                "node1": key[0],
                "endpoint1": endpoints[0],
                "node2": key[1],
                "endpoint2": endpoints[1],
                "edge_type": "unresolved",
                "baseline_edge_semantics": "untyped_pairwise_adjacency",
                "selection_count": int(count),
                "selection_frequency": float(count / len(views)),
                "n_nested_views": len(views),
                "evidence_layers": ",".join(sorted(evidence[key])),
                "interpretation": ";".join(sorted(interpretations[key])),
            }
        )
    columns = [
        "node1",
        "endpoint1",
        "node2",
        "endpoint2",
        "edge_type",
        "baseline_edge_semantics",
        "selection_count",
        "selection_frequency",
        "n_nested_views",
        "evidence_layers",
        "interpretation",
    ]
    return pd.DataFrame(rows, columns=columns)


def _frozen_v3_config(seed: int) -> ChannelAwareConfig:
    """Return the recorded channel_final_v3 configuration without tuning knobs."""

    return ChannelAwareConfig(
        seed=seed,
        within_sample_prior=5.0,
        sparse_alpha=0.25,
        within_profile_clusters=20,
        within_profile_min_abs_r=0.60,
        within_profile_alpha=0.05,
    )


def _baseline_algorithm_config(method: str, seed: int) -> dict[str, Any]:
    if method == "frozen_v3":
        return asdict(_frozen_v3_config(seed))
    if method == "production_fci":
        return dict(PRODUCTION_FCI_BASELINE_CONFIG)
    raise ValidationError(f"Unknown baseline method: {method}")


def _run_production_fci_view(matrix_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the repository's production rank-Gaussian causal-learn FCI."""

    command_dir = (
        Path(__file__).resolve().parents[4]
        / "pipelines/empirical/analysis"
        / "scripts"
        / "commands"
    )
    if str(command_dir) not in sys.path:
        sys.path.insert(0, str(command_dir))
    from run_pc_causallearn import (  # noqa: PLC0415
        apply_transform,
        build_background_knowledge,
        build_confounder_columns,
        edge_to_record,
        run_fci_search,
    )

    observed = pd.read_csv(matrix_path, sep="\t")
    analysis = observed.copy()
    depth_aliases: dict[str, str] = {}
    for background_column in [
        column
        for column in analysis.columns
        if column.startswith("libsize_background__")
    ]:
        assay = background_column.split("__", 1)[1]
        production_name = f"libsize__{assay}"
        if production_name not in analysis.columns:
            analysis[production_name] = analysis[background_column]
            depth_aliases[production_name] = background_column
    confounders = build_confounder_columns(
        analysis, str(PRODUCTION_FCI_BASELINE_CONFIG["confounder_mode"])
    )
    biological = [column for column in analysis.columns if is_biological(column)]
    columns = [*confounders, *biological]
    if len(analysis) < 8 or len(columns) < 4:
        return (
            pd.DataFrame(
                columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"]
            ),
            {
                "method": "production_causal_learn_fci",
                "error": "insufficient_rows_or_variables",
                "n_rows": int(len(analysis)),
                "n_selected_variables": int(len(columns)),
                "confounder_columns": confounders,
                "depth_aliases_from_observable_background": depth_aliases,
            },
        )
    numeric = analysis[columns].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    numeric = numeric.loc[complete]
    live = [column for column in columns if numeric[column].std(ddof=0) > 1e-10]
    if len(numeric) < 8 or len(live) < 4:
        return (
            pd.DataFrame(
                columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"]
            ),
            {
                "method": "production_causal_learn_fci",
                "error": "insufficient_complete_rows_or_live_variables",
                "n_rows": int(len(numeric)),
                "n_live_variables": int(len(live)),
                "confounder_columns": confounders,
                "depth_aliases_from_observable_background": depth_aliases,
            },
        )
    model = apply_transform(
        numeric[live].astype(float),
        str(PRODUCTION_FCI_BASELINE_CONFIG["transform"]),
    )
    model = (model - model.mean()) / model.std(ddof=0).replace(0, 1)
    knowledge, rules = build_background_knowledge(
        live, str(PRODUCTION_FCI_BASELINE_CONFIG["background_mode"])
    )
    _, graph_edges = run_fci_search(
        model.to_numpy(),
        alpha=float(PRODUCTION_FCI_BASELINE_CONFIG["alpha"]),
        indep_test=str(PRODUCTION_FCI_BASELINE_CONFIG["independence_test"]),
        background_knowledge=knowledge,
        show_progress=False,
        node_names=live,
        max_depth=int(PRODUCTION_FCI_BASELINE_CONFIG["max_depth"]),
    )
    edges = pd.DataFrame(
        [edge_to_record(edge) for edge in graph_edges],
        columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"],
    )
    if not edges.empty:
        edges["evidence_layers"] = "production_fci"
        edges["interpretation"] = "production_pairwise_pag"
    return edges, {
        "method": "production_causal_learn_fci",
        **PRODUCTION_FCI_BASELINE_CONFIG,
        "n_rows": int(len(model)),
        "n_live_variables": int(len(live)),
        "n_edges": int(len(edges)),
        "confounder_columns": confounders,
        "depth_aliases_from_observable_background": depth_aliases,
        "background_rule_count": int(len(rules)),
    }


def run_observable_baseline_world(
    method: str,
    inputs: BaselineObservableInputs,
    out_dir: Path,
) -> dict[str, Any]:
    """Freeze one six-view baseline consensus without opening scorer truth."""

    if method not in BASELINE_METHODS:
        raise ValidationError(f"Unknown baseline method: {method}")
    output = Path(out_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValidationError(f"Refusing to overwrite baseline output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    v3_embedding = inputs.embedding
    v3_embedding_adapter: dict[str, Any] | None = None
    if method == "frozen_v3":
        # Frozen v3 expects the historical retained_target column even though
        # its frozen retained_cells_only=False setting never filters on it.
        # V4's observable embedding contains the same cells/coordinates but no
        # label-transfer flag, so add an all-true compatibility column only.
        embedding_frame = pd.read_csv(inputs.embedding, sep="\t")
        if "retained_target" not in embedding_frame.columns:
            embedding_frame["retained_target"] = True
            v3_embedding = output / "v3_analysis_embeddings_compat.tsv"
            embedding_frame.to_csv(v3_embedding, sep="\t", index=False)
            v3_embedding_adapter = {
                "source": str(inputs.embedding),
                "source_sha256": file_sha256(inputs.embedding),
                "adapted": str(v3_embedding),
                "adapted_sha256": file_sha256(v3_embedding),
                "only_added_column": "retained_target",
                "added_value_for_every_cell": True,
                "used_for_cell_filtering": False,
                "coordinates_changed": False,
            }
    view_frames: list[pd.DataFrame] = []
    view_records: list[dict[str, Any]] = []
    for matrix in inputs.matrices:
        view_id = matrix.parent.name
        view_out = output / "views" / view_id
        view_out.mkdir(parents=True, exist_ok=True)
        if method == "frozen_v3":
            environment_java = Path(sys.prefix) / "bin" / "java"
            if "JAVA_HOME" not in os.environ and environment_java.is_file():
                # This mutation is confined to the killable baseline child.
                os.environ["JAVA_HOME"] = sys.prefix
            try:
                import pytetrad  # noqa: F401, PLC0415
            except ModuleNotFoundError as exc:
                raise ValidationError(
                    "Frozen v3 requires pytetrad; run formal validation in the "
                    "repository's causaldisco environment"
                ) from exc
            edges, summary = discover_frozen_v3(
                matrix,
                list(inputs.chrom_cell_tables),
                v3_embedding,
                _frozen_v3_config(inputs.world_seed),
            )
        else:
            edges, summary = _run_production_fci_view(matrix)
        if edges.empty and not {
            "node1",
            "endpoint1",
            "node2",
            "endpoint2",
        } <= set(edges.columns):
            edges = edges.reindex(
                columns=["node1", "endpoint1", "node2", "endpoint2"]
            )
        edges_path = view_out / "edges.tsv"
        edges.to_csv(edges_path, sep="\t", index=False)
        summary_path = view_out / "run_summary.json"
        summary_payload = {
            **summary,
            "baseline_method": method,
            "algorithm_config": _baseline_algorithm_config(
                method, inputs.world_seed
            ),
            "view_id": view_id,
            "observable_matrix": str(matrix),
            "observable_matrix_sha256": file_sha256(matrix),
            "truth_opened": False,
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        view_frames.append(edges)
        view_records.append(
            {
                "view_id": view_id,
                "matrix": str(matrix),
                "matrix_sha256": file_sha256(matrix),
                "edges": str(edges_path),
                "edges_sha256": file_sha256(edges_path),
                "summary": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
            }
        )
    consensus = consensus_baseline_graph(view_frames)
    consensus_path = output / "consensus_edges.tsv"
    consensus.to_csv(consensus_path, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": "ca_boss_v4.post_candidate_baseline_world.v1",
        "baseline_method": method,
        "world": inputs.world_name,
        "world_seed": inputs.world_seed,
        "candidate_outputs_frozen_before_hook": True,
        "diagnostic_only": True,
        "may_influence_v4_selection": False,
        "truth_opened_before_consensus_freeze": False,
        "observable_inputs": inputs.file_sha256,
        "same_prepared_matrices_as_v4": True,
        "v3_embedding_schema_adapter": v3_embedding_adapter,
        "algorithm_config": _baseline_algorithm_config(
            method, inputs.world_seed
        ),
        "n_views": len(view_frames),
        "minimum_view_frequency": BASELINE_VIEW_SUPPORT,
        "minimum_view_count": math.ceil(
            BASELINE_VIEW_SUPPORT * len(view_frames) - 1e-12
        ),
        "v3_algorithm_configuration_frozen": method == "frozen_v3",
        "v3_view_protocol_adapted_to_current_six_prepared_views": method == "frozen_v3",
        "endpoint_alphabet": sorted(BASELINE_ALLOWED_ENDPOINTS),
        "endpoint_consensus": (
            "independent_per_endpoint_at_two_thirds_among_adjacency_present_views"
        ),
        "runtime_python": sys.executable,
        "runtime_java_home": os.environ.get("JAVA_HOME", ""),
        "views": view_records,
        "consensus": str(consensus_path),
        "consensus_sha256": file_sha256(consensus_path),
        "n_consensus_edges": int(len(consensus)),
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = output / "baseline_pretruth_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "consensus": str(consensus_path),
        "consensus_sha256": manifest["consensus_sha256"],
        "n_consensus_edges": int(len(consensus)),
    }


def _baseline_worker(
    connection: Any,
    method: str,
    inputs: BaselineObservableInputs,
    out_dir: Path,
    evaluator: Callable[[str, BaselineObservableInputs, Path], dict[str, Any]],
) -> None:
    try:
        connection.send(("ok", evaluator(method, inputs, out_dir), ""))
    except BaseException:  # noqa: BLE001 - failures remain registered rows
        connection.send(("crash", {}, traceback.format_exc(limit=40)))
    finally:
        connection.close()


def evaluate_baseline_with_timeout(
    method: str,
    inputs: BaselineObservableInputs,
    out_dir: Path,
    timeout_seconds: float,
    *,
    evaluator: Callable[
        [str, BaselineObservableInputs, Path], dict[str, Any]
    ] = run_observable_baseline_world,
) -> BaselineExecution:
    """Run one method/world diagnostic in a killable process."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValidationError("Baseline timeout must be a positive finite number")
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else methods[0])
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_baseline_worker,
        args=(send, method, inputs, out_dir, evaluator),
    )
    started = time.monotonic()
    process.start()
    send.close()
    try:
        if not receive.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            return BaselineExecution(
                status="timeout",
                elapsed_seconds=time.monotonic() - started,
                result={},
                error=(
                    f"Exceeded registered per-method/world timeout of "
                    f"{timeout_seconds:g} seconds"
                ),
            )
        try:
            status, result, error = receive.recv()
        except EOFError:
            status, result, error = (
                "crash",
                {},
                f"Baseline worker exited without a result (exitcode={process.exitcode})",
            )
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        return BaselineExecution(
            status=str(status),
            elapsed_seconds=time.monotonic() - started,
            result=dict(result),
            error=str(error),
        )
    finally:
        receive.close()


def _current_baseline_input_hashes(
    inputs: BaselineObservableInputs,
) -> dict[str, str]:
    prepared = inputs.embedding.parent
    discovery = inputs.chrom_cell_tables[0].parent
    hashes: dict[str, str] = {}
    for path in (inputs.embedding, *inputs.matrices, *inputs.chrom_cell_tables):
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"Baseline observable input changed or vanished: {path}")
        if _is_below(path, prepared):
            label = f"prepared_discovery/{path.relative_to(prepared)}"
        elif _is_below(path, discovery):
            label = f"discovery/{path.relative_to(discovery)}"
        else:
            raise ValidationError(f"Baseline observable input escaped its frozen roots: {path}")
        hashes[label] = file_sha256(path)
    return dict(sorted(hashes.items()))


def _load_and_validate_frozen_baseline_result(
    method: str,
    inputs: BaselineObservableInputs,
    expected_root: Path,
    result: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and load a baseline estimate completely before scorer access."""

    root = expected_root.resolve(strict=True)
    manifest_path = Path(str(result.get("manifest", "")))
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or not _is_below(manifest_path, root)
    ):
        raise ValidationError("Baseline returned an unsafe pretruth manifest path")
    if file_sha256(manifest_path) != str(result.get("manifest_sha256", "")):
        raise ValidationError("Baseline pretruth manifest file hash mismatch")
    manifest = _read_json_object(
        manifest_path, label=f"{method} baseline pretruth manifest"
    )
    payload_hash = _validate_self_hash(
        manifest,
        field="manifest_payload_sha256",
        label=f"{method} baseline pretruth manifest",
    )
    if payload_hash != str(result.get("manifest_payload_sha256", "")):
        raise ValidationError("Baseline pretruth manifest payload hash mismatch")
    expected_config = _baseline_algorithm_config(method, inputs.world_seed)
    expected_values = {
        "schema_version": "ca_boss_v4.post_candidate_baseline_world.v1",
        "baseline_method": method,
        "world": inputs.world_name,
        "world_seed": inputs.world_seed,
        "candidate_outputs_frozen_before_hook": True,
        "diagnostic_only": True,
        "may_influence_v4_selection": False,
        "truth_opened_before_consensus_freeze": False,
        "same_prepared_matrices_as_v4": True,
        "n_views": BASELINE_EXPECTED_VIEWS,
        "minimum_view_frequency": BASELINE_VIEW_SUPPORT,
        "minimum_view_count": math.ceil(
            BASELINE_VIEW_SUPPORT * BASELINE_EXPECTED_VIEWS - 1e-12
        ),
        "algorithm_config": expected_config,
        "endpoint_alphabet": sorted(BASELINE_ALLOWED_ENDPOINTS),
        "endpoint_consensus": (
            "independent_per_endpoint_at_two_thirds_among_adjacency_present_views"
        ),
    }
    mismatched = [
        key for key, value in expected_values.items() if manifest.get(key) != value
    ]
    if mismatched:
        raise ValidationError(
            f"Baseline pretruth manifest contract mismatch: {sorted(mismatched)}"
        )
    current_input_hashes = _current_baseline_input_hashes(inputs)
    if current_input_hashes != inputs.file_sha256:
        raise ValidationError("Baseline observable inputs changed during execution")
    if manifest.get("observable_inputs") != current_input_hashes:
        raise ValidationError("Baseline manifest observable-input hashes do not match")

    expected_matrices = {matrix.parent.name: matrix for matrix in inputs.matrices}
    views = manifest.get("views")
    if not isinstance(views, list) or len(views) != BASELINE_EXPECTED_VIEWS:
        raise ValidationError("Baseline manifest does not contain exactly six views")
    observed_view_ids: set[str] = set()
    validated_view_frames: list[pd.DataFrame] = []
    for view in views:
        if not isinstance(view, Mapping):
            raise ValidationError("Baseline manifest contains a malformed view record")
        view_id = str(view.get("view_id", ""))
        if view_id in observed_view_ids or view_id not in expected_matrices:
            raise ValidationError(f"Baseline manifest has an invalid view id: {view_id!r}")
        observed_view_ids.add(view_id)
        matrix = expected_matrices[view_id]
        recorded_matrix = Path(str(view.get("matrix", "")))
        if recorded_matrix.resolve(strict=True) != matrix.resolve(strict=True):
            raise ValidationError(f"Baseline view {view_id} used the wrong matrix")
        if str(view.get("matrix_sha256", "")) != file_sha256(matrix):
            raise ValidationError(f"Baseline view {view_id} matrix hash mismatch")
        artifacts: dict[str, Path] = {}
        for path_key, hash_key in (
            ("edges", "edges_sha256"),
            ("summary", "summary_sha256"),
        ):
            artifact = Path(str(view.get(path_key, "")))
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or not _is_below(artifact, root)
                or file_sha256(artifact) != str(view.get(hash_key, ""))
            ):
                raise ValidationError(
                    f"Baseline view {view_id} {path_key} artifact hash mismatch"
                )
            artifacts[path_key] = artifact
        try:
            view_edges = pd.read_csv(
                artifacts["edges"], sep="\t", keep_default_na=False
            )
        except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
            raise ValidationError(
                f"Could not parse baseline view {view_id} edges: {exc}"
            ) from exc
        validated_view_frames.append(view_edges)
        summary = _read_json_object(
            artifacts["summary"], label=f"baseline view {view_id} summary"
        )
        if (
            summary.get("baseline_method") != method
            or summary.get("view_id") != view_id
            or summary.get("observable_matrix_sha256") != file_sha256(matrix)
            or summary.get("truth_opened") is not False
            or summary.get("algorithm_config") != expected_config
        ):
            raise ValidationError(f"Baseline view {view_id} summary contract mismatch")
    if observed_view_ids != set(expected_matrices):
        raise ValidationError("Baseline manifest view set does not match prepared inputs")

    consensus_path = Path(str(result.get("consensus", "")))
    recorded_consensus = Path(str(manifest.get("consensus", "")))
    if (
        consensus_path.is_symlink()
        or not consensus_path.is_file()
        or not _is_below(consensus_path, root)
        or recorded_consensus.resolve(strict=True) != consensus_path.resolve(strict=True)
    ):
        raise ValidationError("Baseline returned an unsafe consensus path")
    consensus_bytes = consensus_path.read_bytes()
    observed_consensus_hash = hashlib.sha256(consensus_bytes).hexdigest()
    if (
        observed_consensus_hash != str(result.get("consensus_sha256", ""))
        or observed_consensus_hash != str(manifest.get("consensus_sha256", ""))
    ):
        raise ValidationError("Baseline consensus hash mismatch")
    try:
        frozen_edges = pd.read_csv(
            io.BytesIO(consensus_bytes), sep="\t", keep_default_na=False
        )
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise ValidationError(f"Could not parse frozen baseline consensus: {exc}") from exc
    required_columns = {"node1", "endpoint1", "node2", "endpoint2"}
    if not required_columns <= set(frozen_edges.columns):
        raise ValidationError("Frozen baseline consensus lacks canonical edge columns")
    endpoints = set(frozen_edges["endpoint1"].astype(str)) | set(
        frozen_edges["endpoint2"].astype(str)
    )
    if not endpoints <= BASELINE_ALLOWED_ENDPOINTS:
        raise ValidationError(
            f"Frozen baseline consensus has invalid endpoints: "
            f"{sorted(endpoints - BASELINE_ALLOWED_ENDPOINTS)}"
        )
    if int(manifest.get("n_consensus_edges", -1)) != len(frozen_edges):
        raise ValidationError("Baseline consensus edge count differs from its manifest")
    recomputed = consensus_baseline_graph(validated_view_frames)
    try:
        pd.testing.assert_frame_equal(
            recomputed.reset_index(drop=True),
            frozen_edges.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValidationError(
            "Frozen baseline consensus does not match independently recomputed votes"
        ) from exc
    audit = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_payload_sha256": payload_hash,
        "consensus": str(consensus_path),
        "consensus_sha256": observed_consensus_hash,
        "consensus_bytes_loaded_before_truth": True,
        "consensus_frame_parsed_before_truth": True,
        "manifest_config_inputs_and_hashes_validated_before_truth": True,
    }
    return frozen_edges, audit


def _baseline_truth_candidate_keys(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    def nodes(column: str, fallback: str) -> tuple[str, ...]:
        raw = record.get(column, "")
        value = "" if raw is None or pd.isna(raw) else str(raw).strip()
        return tuple(token for token in value.split(";") if token) or (fallback,)

    left, right = str(record["node1"]), str(record["node2"])
    return {
        tuple(sorted((node1, node2)))
        for node1 in nodes("node_set1", left)
        for node2 in nodes("node_set2", right)
        if node1 != node2
    }


def score_untyped_baseline(
    edges: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score pairwise adjacency only; no v4 layer or latent-axis credit."""

    required_truth = {"node1", "node2", "truth_type", "label"}
    if not required_truth <= set(truth.columns):
        raise ValidationError("Baseline truth table lacks its typed pair contract")
    if not {"node1", "node2"} <= set(edges.columns):
        raise ValidationError("Baseline consensus lacks adjacency columns")
    truth_records = truth.to_dict(orient="records")
    focal_nodes = {
        node
        for record in truth_records
        for key in _baseline_truth_candidate_keys(record)
        for node in key
    }
    estimated = {
        tuple(sorted((str(record["node1"]), str(record["node2"]))))
        for record in edges.to_dict(orient="records")
        if str(record["node1"]) in focal_nodes and str(record["node2"]) in focal_nodes
    }
    rows: list[dict[str, Any]] = []
    for record in truth_records:
        candidates = _baseline_truth_candidate_keys(record)
        rows.append(
            {
                **record,
                "adjacency_present": int(bool(candidates & estimated)),
            }
        )
    pairs = pd.DataFrame(rows)
    direct = pairs.loc[pairs["truth_type"].eq("within_assay_direct")]
    shared = pairs.loc[
        pairs["truth_type"].isin(
            {"within_assay_shared_state", "cross_assay_shared_state"}
        )
    ]
    slow = pairs.loc[pairs["truth_type"].eq("slow_peak_gene")]
    nulls = pairs.loc[pairs["truth_type"].eq("sentinel_null")]
    if direct.empty or shared.empty or len(slow) != 1 or nulls.empty:
        raise ValidationError("Baseline truth lacks required pair classes")
    if "mark" in direct:
        direct_recall = float(direct.groupby("mark")["adjacency_present"].mean().mean())
    else:
        direct_recall = float(direct["adjacency_present"].mean())
    shared_recall = float(shared["adjacency_present"].mean())
    slow_present = int(slow["adjacency_present"].iloc[0])
    null_fpr = float(nulls["adjacency_present"].mean())
    positive_types = {
        "within_assay_direct",
        "within_assay_shared_state",
        "cross_assay_shared_state",
        "slow_peak_gene",
    }
    positive_keys = set().union(
        *(
            _baseline_truth_candidate_keys(record)
            for record in truth_records
            if str(record["truth_type"]) in positive_types
        )
    )
    true_estimated = estimated & positive_keys
    precision = float(len(true_estimated) / len(estimated)) if estimated else 0.0
    metrics = {
        "n_estimated_focal_edges": int(len(estimated)),
        "direct_adjacency_macro_recall": direct_recall,
        "shared_state_adjacency_recall": shared_recall,
        "slow_peak_gene_present": slow_present,
        "matched_null_false_positive_rate": null_fpr,
        "primary_skeleton_precision": precision,
        "pairwise_diagnostic_success": int(
            direct_recall >= 0.90
            and shared_recall >= 0.80
            and slow_present == 1
            and null_fpr <= 0.10
            and precision >= 0.80
        ),
        "typed_layer_credit": False,
        "latent_axis_credit": False,
        "diagnostic_only": True,
    }
    return metrics, pairs


def run_registered_baseline_diagnostics(
    worlds: Mapping[int, Path],
    split: str,
    evaluation_out: Path,
    out_dir: Path,
    timeout_seconds: float,
    candidate_freeze: Mapping[str, Any],
    *,
    baseline_executor: Callable[
        [str, BaselineObservableInputs, Path, float], BaselineExecution
    ] = evaluate_baseline_with_timeout,
) -> dict[str, Any]:
    """Run both required diagnostics after v4 execution has been frozen."""

    output = Path(out_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValidationError(f"Refusing to overwrite baseline output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for method in BASELINE_METHODS:
        for seed, world in worlds.items():
            try:
                inputs = collect_baseline_observable_inputs(world, evaluation_out)
            except (ValidationError, OSError) as exc:
                execution = BaselineExecution(
                    status="invalid_observable_inputs",
                    elapsed_seconds=0.0,
                    result={},
                    error=str(exc),
                )
                inputs = None
            else:
                try:
                    execution = baseline_executor(
                        method,
                        inputs,
                        output / method / world.name,
                        timeout_seconds,
                    )
                    if not isinstance(execution, BaselineExecution):
                        raise TypeError(
                            "baseline executor did not return BaselineExecution"
                        )
                except Exception as exc:  # noqa: BLE001 - retain and continue all worlds
                    execution = BaselineExecution(
                        status="executor_exception",
                        elapsed_seconds=0.0,
                        result={},
                        error=f"{type(exc).__name__}: {exc}",
                    )
            status = execution.status
            metrics: dict[str, Any] = {}
            score_paths: dict[str, str] = {}
            pretruth_audit: dict[str, Any] = {}
            if status == "ok":
                try:
                    if inputs is None:
                        raise ValidationError("Baseline scoring lacks observable inputs")
                    # Load bytes/frame and validate every discovery-side contract
                    # before this first scorer read. Scoring uses only the held
                    # in-memory frame even if the on-disk file later changes.
                    frozen_edges, pretruth_audit = (
                        _load_and_validate_frozen_baseline_result(
                            method,
                            inputs,
                            output / method / world.name,
                            execution.result,
                        )
                    )
                    truth = pd.read_csv(
                        world / "scorer" / "truth_typed_edges.tsv",
                        sep="\t",
                        keep_default_na=False,
                    )
                    metrics, pair_scores = score_untyped_baseline(frozen_edges, truth)
                    score_out = output / method / world.name / "diagnostic_score"
                    score_out.mkdir()
                    pair_path = score_out / "pair_scores.tsv"
                    metrics_path = score_out / "metrics.json"
                    pair_scores.to_csv(pair_path, sep="\t", index=False)
                    _write_json(metrics_path, metrics)
                    score_paths = {
                        "pair_scores": str(pair_path),
                        "pair_scores_sha256": file_sha256(pair_path),
                        "metrics": str(metrics_path),
                        "metrics_sha256": file_sha256(metrics_path),
                    }
                except (KeyError, OSError, TypeError, ValueError, ValidationError) as exc:
                    status = "scoring_error"
                    execution.error = str(exc)
            row = {
                "baseline_method": method,
                "world": world.name,
                "world_seed": seed,
                "execution_status": status,
                "elapsed_seconds": float(execution.elapsed_seconds),
                "execution_error": execution.error,
                **metrics,
            }
            rows.append(row)
            records.append(
                {
                    **row,
                    "candidate_outputs_frozen_before_hook": True,
                    "outputs_available_to_v4_selection": False,
                    "baseline_estimate_frozen_before_truth_score": status
                    not in {"invalid_observable_inputs", "crash", "timeout"},
                    "observable_inputs": inputs.file_sha256 if inputs else {},
                    "pretruth_result": execution.result,
                    "parent_pretruth_audit": pretruth_audit,
                    "diagnostic_score": score_paths,
                }
            )
    results = pd.DataFrame(rows)
    results_path = output / "baseline_world_results.tsv"
    results.to_csv(results_path, sep="\t", index=False)
    method_summaries: dict[str, Any] = {}
    for method in BASELINE_METHODS:
        subset = results.loc[results["baseline_method"].eq(method)]
        counts = {
            str(key): int(value)
            for key, value in subset["execution_status"].value_counts().sort_index().items()
        }
        ok = int(counts.get("ok", 0))
        method_summaries[method] = {
            "registered_worlds": len(worlds),
            "attempted_worlds": int(len(subset)),
            "execution_status_counts": counts,
            "successful_diagnostic_scores": ok,
            "biological_pairwise_diagnostic_successes": int(
                pd.to_numeric(
                    subset.get(
                        "pairwise_diagnostic_success",
                        pd.Series(0, index=subset.index),
                    ),
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "pass": bool(len(subset) == len(worlds) and ok == len(worlds)),
        }
    complete = all(summary["pass"] for summary in method_summaries.values())
    manifest: dict[str, Any] = {
        "schema_version": "ca_boss_v4.post_candidate_baselines.v1",
        "split": split,
        "world_seeds": list(worlds),
        "candidate_execution_freeze": dict(candidate_freeze),
        "candidate_outputs_frozen_before_hooks": True,
        "diagnostic_only": True,
        "outcomes_enter_candidate_gate": False,
        "outcomes_available_to_v4_selection": False,
        "same_prepared_matrix_and_embedding_inputs_as_v4": True,
        "v3_additional_inputs_are_observed_same_assay_cells": True,
        "baseline_estimates_frozen_before_diagnostic_truth_scoring": True,
        "parent_loads_and_validates_frozen_consensus_before_truth": True,
        "diagnostic_scoring_uses_parent_frozen_in_memory_frames": True,
        "failures_and_timeouts_retained_in_registered_denominator": True,
        "per_method_world_timeout_seconds": float(timeout_seconds),
        "methods": method_summaries,
        "worlds": records,
        "pass": complete,
        "artifacts": {
            "world_results": results_path.name,
            "world_results_sha256": file_sha256(results_path),
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = output / "baseline_diagnostics_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def run_registered_upper_bound_gate(
    worlds: Mapping[int, Path],
    split: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Run and freeze the scorer-only observation/estimand ceiling.

    This function is deliberately called only after executable sources and the
    training-selected candidate have been frozen.  Its artifacts are tainted
    scorer outputs and are never passed into ``evaluate_world``.
    """

    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise ValidationError(
            f"Refusing to overwrite non-empty upper-bound output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    results = []
    records: list[dict[str, Any]] = []
    for seed, world in worlds.items():
        result = evaluate_world_upper_bound(world)
        if result.world_seed != seed:
            raise ValidationError(
                f"Upper-bound seed mismatch for {world}: {result.world_seed} != {seed}"
            )
        world_output = output / world.name
        manifest = write_world_upper_bound(
            result,
            world_output,
            world_dir=world,
        )
        results.append(result)
        records.append(
            {
                "seed": seed,
                "world": world.name,
                "world_success": bool(result.metrics["world_success"]),
                "manifest": str(world_output / "oracle_upper_bound_manifest.json"),
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            }
        )
    aggregate = aggregate_registered_upper_bound(results, split=split)
    aggregate_path = output / "registered_upper_bound_gate.json"
    _write_json(aggregate_path, aggregate)
    return {
        "pass": bool(aggregate["pass"]),
        "scorer_only": True,
        "discovery_eligible": False,
        "candidate_outputs_frozen_before_gate": True,
        "aggregate": aggregate,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": file_sha256(aggregate_path),
        "worlds": records,
    }


def _validate_discovery_handle(
    frozen: FrozenWorldPretruth,
    *,
    world: Path,
    candidate_name: str,
    evaluation_out: Path,
) -> None:
    """Bind an injected discovery result to its registered observable world."""

    expected_world = world.resolve(strict=True)
    expected_world_out = (evaluation_out / world.name).resolve()
    expected_manifest = expected_world_out / "pretruth_consensus_manifest.json"
    try:
        observed_world = Path(frozen.world_dir).resolve(strict=True)
        observed_world_out = Path(frozen.world_out).resolve(strict=True)
        observed_manifest = Path(frozen.pretruth_manifest)
        if observed_manifest.is_symlink() or not observed_manifest.is_file():
            raise ValidationError("Frozen pretruth manifest is not a regular file")
        observed_manifest = observed_manifest.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError(f"Frozen discovery handle has unsafe paths: {exc}") from exc
    if frozen.world_name != world.name or observed_world != expected_world:
        raise ValidationError("Frozen discovery handle is bound to the wrong world")
    if observed_world_out != expected_world_out:
        raise ValidationError("Frozen discovery handle is bound to the wrong output directory")
    if tuple(frozen.candidate_names) != (candidate_name,):
        raise ValidationError("Frozen discovery handle is bound to the wrong candidate")
    if observed_manifest != expected_manifest:
        raise ValidationError("Frozen discovery handle uses an unexpected manifest path")
    if not isinstance(frozen.pretruth_manifest_sha256, str) or not (
        _HASH_PATTERN.fullmatch(frozen.pretruth_manifest_sha256)
    ):
        raise ValidationError("Frozen discovery handle has an invalid manifest hash")
    if file_sha256(observed_manifest) != frozen.pretruth_manifest_sha256:
        raise ValidationError("Frozen discovery manifest changed before the global freeze")


def _write_global_pretruth_freeze(
    path: Path,
    *,
    worlds: Mapping[int, Path],
    discoveries: Mapping[int, DiscoveryExecution],
) -> dict[str, Any]:
    """Write the exact all-world discovery barrier consumed by the scorer."""

    if set(discoveries) != set(worlds):
        raise ValidationError("Global pretruth freeze lacks a registered discovery attempt")
    world_records: list[dict[str, Any]] = []
    for seed, world in worlds.items():
        execution = discoveries[seed]
        record: dict[str, Any] = {
            "world_name": world.name,
            "world_dir": str(world.resolve(strict=True)),
            "status": execution.status,
        }
        if execution.status == "ok":
            if execution.frozen is None:
                raise ValidationError("Successful discovery lacks a frozen handle")
            record["pretruth_manifest_sha256"] = (
                execution.frozen.pretruth_manifest_sha256
            )
        world_records.append(record)
    manifest: dict[str, Any] = {
        "schema_version": GLOBAL_PRETRUTH_FREEZE_SCHEMA_VERSION,
        "all_registered_discovery_attempts_finished": True,
        "registered_world_names": [world.name for world in worlds.values()],
        "world_records": world_records,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    _write_json(path, manifest)
    return manifest


def run_validation(
    *,
    split: str,
    world_dirs: Sequence[Path | str],
    frozen_config: Path | str,
    calibration_audit: Path | str,
    calibration_audit_sha256: str,
    out_dir: Path | str,
    timeout_seconds: float,
    development_manifest: Path | str | None = None,
    development_manifest_sha256: str | None = None,
    discovery_executor: Callable[
        [Path, str, CABossV4Config, Path, float], DiscoveryExecution
    ] = discover_world_with_timeout,
    scoring_executor: Callable[
        [FrozenWorldPretruth, Path, float], WorldExecution
    ] = score_world_with_timeout,
    provenance_auditor: Callable[[Path, Path, str], Mapping[str, Any]] = audit_world_output,
    baseline_runner: Callable[
        [
            Mapping[int, Path],
            str,
            Path,
            Path,
            float,
            Mapping[str, Any],
        ],
        Mapping[str, Any],
    ] = run_registered_baseline_diagnostics,
    upper_bound_runner: Callable[
        [Mapping[int, Path], str, Path], Mapping[str, Any]
    ] = run_registered_upper_bound_gate,
) -> dict[str, Any]:
    """Run one exact confirmatory split and write a content-addressed manifest."""

    if split not in SPLITS:
        raise ValidationError(f"Unknown split {split!r}; choose development or locked")
    spec = SPLITS[split]
    worlds = validate_world_set(world_dirs, spec)
    frozen = load_frozen_candidate(
        frozen_config,
        calibration_audit,
        calibration_audit_sha256,
    )

    # This source freeze precedes the discovery-only global pretruth barrier.
    freeze_time = datetime.now(timezone.utc).isoformat()
    sources = source_hash_bundle()
    source_digest = _source_digest(sources)

    dev_manifest_record: dict[str, Any] | None = None
    if split == "development":
        if development_manifest is not None or development_manifest_sha256 is not None:
            raise ValidationError("Development execution may not consume an earlier DEV manifest")
    else:
        if development_manifest is None or development_manifest_sha256 is None:
            raise ValidationError(
                "Fresh LOCKED execution requires a passing immutable DEV manifest and SHA-256"
            )
        validated_dev = validate_development_manifest(
            development_manifest,
            development_manifest_sha256,
            frozen=frozen,
            current_source_hashes=sources,
        )
        dev_manifest_record = {
            "path": str(Path(development_manifest).resolve()),
            "sha256": file_sha256(Path(development_manifest)),
            "payload_sha256": validated_dev["manifest_payload_sha256"],
        }
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise ValidationError(f"Refusing to overwrite non-empty validation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    evaluation_out = output / "worlds"
    evaluation_out.mkdir()

    discoveries: dict[int, DiscoveryExecution] = {}
    for ordinal, (seed, world) in enumerate(worlds.items(), start=1):
        print(
            f"[discover {ordinal}/{len(worlds)}] {spec.name} seed {seed}",
            flush=True,
        )
        try:
            discovery = discovery_executor(
                world,
                frozen.name,
                frozen.config,
                evaluation_out,
                timeout_seconds,
            )
        except Exception as exc:  # injected executors must not shrink the denominator
            discovery = DiscoveryExecution(
                status="executor_exception",
                frozen=None,
                elapsed_seconds=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(discovery, DiscoveryExecution):
            discovery = DiscoveryExecution(
                status="invalid_result",
                frozen=None,
                elapsed_seconds=0.0,
                error="Discovery executor returned the wrong result type",
            )
        elif discovery.status == "ok":
            try:
                if discovery.frozen is None:
                    raise ValidationError("Successful discovery lacks a frozen handle")
                _validate_discovery_handle(
                    discovery.frozen,
                    world=world,
                    candidate_name=frozen.name,
                    evaluation_out=evaluation_out,
                )
            except ValidationError as exc:
                discovery = DiscoveryExecution(
                    status="invalid_result",
                    frozen=None,
                    elapsed_seconds=discovery.elapsed_seconds,
                    error=str(exc),
                )
        elif discovery.frozen is not None:
            discovery = DiscoveryExecution(
                status=discovery.status,
                frozen=None,
                elapsed_seconds=discovery.elapsed_seconds,
                error=discovery.error,
            )
        discoveries[seed] = discovery

    global_freeze_path = output / "global_pretruth_freeze_manifest.json"
    global_freeze = _write_global_pretruth_freeze(
        global_freeze_path,
        worlds=worlds,
        discoveries=discoveries,
    )
    global_freeze_file_sha256 = file_sha256(global_freeze_path)

    rows: list[dict[str, Any]] = []
    world_records: list[dict[str, Any]] = []
    for ordinal, (seed, world) in enumerate(worlds.items(), start=1):
        discovery = discoveries[seed]
        scoring_status = "not_run_due_to_discovery_failure"
        scoring_error = ""
        scoring_elapsed = 0.0
        if discovery.status != "ok":
            execution = WorldExecution(
                status=discovery.status,
                rows=[],
                elapsed_seconds=discovery.elapsed_seconds,
                error=discovery.error,
            )
            audit = {
                "pass": False,
                "not_run_due_to_discovery_failure": True,
                "reasons": [discovery.error],
            }
        else:
            print(
                f"[score {ordinal}/{len(worlds)}] {spec.name} seed {seed}",
                flush=True,
            )
            assert discovery.frozen is not None
            try:
                scored = scoring_executor(
                    discovery.frozen,
                    global_freeze_path,
                    timeout_seconds,
                )
            except Exception as exc:  # injected executors must not shrink the denominator
                scored = WorldExecution(
                    status="executor_exception",
                    rows=[],
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if not isinstance(scored, WorldExecution):
                scored = WorldExecution(
                    status="invalid_result",
                    rows=[],
                    elapsed_seconds=0.0,
                    error="Scoring executor returned the wrong result type",
                )
            scoring_status = scored.status
            scoring_error = scored.error
            scoring_elapsed = scored.elapsed_seconds
            execution = WorldExecution(
                status=scored.status,
                rows=scored.rows,
                elapsed_seconds=discovery.elapsed_seconds + scored.elapsed_seconds,
                error=scored.error,
            )
            if execution.status != "ok":
                audit = {
                    "pass": False,
                    "not_run_due_to_scoring_failure": True,
                    "reasons": [execution.error],
                }
            else:
                try:
                    audit = dict(provenance_auditor(world, evaluation_out, frozen.name))
                    row = _successful_result(
                        seed, world, frozen.name, execution, audit
                    )
                except Exception as exc:
                    execution = WorldExecution(
                        status="invalid_result",
                        rows=[],
                        elapsed_seconds=execution.elapsed_seconds,
                        error=str(exc),
                    )
                    audit = {
                        "pass": False,
                        "invalid_result": True,
                        "reasons": [str(exc)],
                    }

        if execution.status != "ok":
            row = _failure_result(seed, world, frozen.name, execution)
        rows.append(row)
        world_records.append(
            {
                "seed": seed,
                "world": str(world),
                "discovery_status": discovery.status,
                "discovery_elapsed_seconds": float(discovery.elapsed_seconds),
                "discovery_error": discovery.error,
                "scoring_status": scoring_status,
                "scoring_elapsed_seconds": float(scoring_elapsed),
                "scoring_error": scoring_error,
                "execution_status": row["execution_status"],
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "error": str(row["execution_error"]),
                "provenance_audit": audit,
                "validation_world_success": bool(row["validation_world_success"]),
            }
        )

    results = pd.DataFrame(rows).sort_values("world_seed").reset_index(drop=True)
    gates = aggregate_gates(results, spec)
    results_path = output / "world_validation_results.tsv"
    results.to_csv(results_path, sep="\t", index=False)
    pretruth_hashes = {
        worlds[seed].name: execution.frozen.pretruth_manifest_sha256
        for seed, execution in discoveries.items()
        if execution.status == "ok" and execution.frozen is not None
    }
    candidate_execution_freeze = {
        "created_after_all_registered_v4_execution_attempts_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "all_registered_v4_execution_attempts_finished": True,
        "registered_worlds": len(worlds),
        "global_pretruth_freeze_manifest": str(global_freeze_path),
        "global_pretruth_freeze_manifest_sha256": global_freeze_file_sha256,
        "global_pretruth_freeze_payload_sha256": global_freeze[
            "manifest_payload_sha256"
        ],
        "world_results": str(results_path),
        "world_results_sha256": file_sha256(results_path),
        "available_pretruth_consensus_manifest_sha256": pretruth_hashes,
        "missing_pretruth_consensus_worlds_are_retained_execution_failures": sorted(
            world.name for world in worlds.values() if world.name not in pretruth_hashes
        ),
    }
    baseline_diagnostics = dict(
        baseline_runner(
            worlds,
            split,
            evaluation_out,
            output / "post_candidate_baselines",
            timeout_seconds,
            candidate_execution_freeze,
        )
    )
    upper_bound = dict(
        upper_bound_runner(worlds, split, output / "upper_bound_scorer_only")
    )
    upper_bound_pass = upper_bound.get("pass") is True
    gates["upper_bound_gate"] = {
        "pass": upper_bound_pass,
        "candidate_outputs_frozen_before_gate": True,
    }
    gates["pass"] = bool(gates["pass"] and upper_bound_pass)
    candidate_gate_pass = bool(gates["pass"] and upper_bound_pass)
    baseline_diagnostics_complete = baseline_diagnostics.get("pass") is True
    formal_certification_complete = bool(
        candidate_gate_pass
        and baseline_diagnostics_complete
        and all(value["implemented"] for value in PREREGISTERED_HOOKS.values())
    )
    if formal_certification_complete:
        certification_status = "complete"
    elif not candidate_gate_pass:
        certification_status = "candidate_gate_failed"
    elif not baseline_diagnostics_complete:
        certification_status = "blocked_incomplete_preregistered_baseline_diagnostics"
    else:
        certification_status = "blocked_missing_preregistered_hooks"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "registered_split_name": spec.name,
        "world_seeds": list(spec.seeds),
        "single_candidate_only": True,
        "candidate_grid_permitted": False,
        "frozen_candidate": {
            "name": frozen.name,
            "config_bytes_sha256": frozen.config_bytes_sha256,
            "config_canonical_sha256": frozen.config_canonical_sha256,
            "calibration_audit_bytes_sha256": frozen.calibration_audit_bytes_sha256,
            "calibration_audit_payload_sha256": frozen.calibration_audit_payload_sha256,
        },
        "source_freeze": {
            "created_before_world_evaluation_utc": freeze_time,
            "files": sources,
            "content_digest_sha256": source_digest,
        },
        "truth_boundary": {
            "sources_hashed_before_truth": True,
            "discover_world_candidates_per_call": 1,
            "all_registered_discovery_attempts_finished_before_any_scorer_truth_open": True,
            "global_pretruth_freeze_manifest_required_by_every_score_call": True,
            "global_pretruth_freeze_manifest_sha256": global_freeze_file_sha256,
            "pretruth_consensus_audited_per_completed_world": True,
            "all_candidate_score_attempts_finished_before_baselines": True,
            "all_candidate_outputs_frozen_before_baselines": True,
            "baseline_estimates_frozen_before_diagnostic_truth_scoring": True,
            "baseline_outputs_available_to_v4_selection": False,
            "all_candidate_outputs_frozen_before_upper_bound": True,
            "upper_bound_outputs_available_to_discovery": False,
            "historical_locked_seeds_permitted": False,
        },
        "development_prerequisite": dev_manifest_record,
        "post_candidate_baseline_diagnostics": baseline_diagnostics,
        "background_state_upper_bound": upper_bound,
        "failure_policy": {
            "crashes_and_timeouts_remain_in_world_denominator": True,
            "per_world_timeout_seconds": float(timeout_seconds),
            "discovery_failures_are_not_scored": True,
            "discovery_and_scoring_failures_are_retained": True,
        },
        "registered_gates": {
            "world_successes_required": spec.minimum_world_successes,
            "each_class_successes_required": spec.minimum_class_successes,
            "each_factor_level_successes_required": spec.minimum_factor_level_successes,
            "each_design_cell_successes_required": spec.minimum_design_cell_successes,
            "pooled_null_wilson_upper_maximum": TARGET_NULL_FPR,
            "mean_graph_density_strictly_below": V3_MEAN_BIOLOGICAL_EDGES,
        },
        "gates": gates,
        "candidate_gate_pass": candidate_gate_pass,
        "preregistered_hooks": PREREGISTERED_HOOKS,
        "formal_certification_complete": formal_certification_complete,
        "formal_certification_status": certification_status,
        "worlds": world_records,
        "artifacts": {
            "world_results": results_path.name,
            "world_results_sha256": file_sha256(results_path),
            "global_pretruth_freeze_manifest": global_freeze_path.name,
            "global_pretruth_freeze_manifest_sha256": global_freeze_file_sha256,
            "global_pretruth_freeze_payload_sha256": global_freeze[
                "manifest_payload_sha256"
            ],
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = output / "validation_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_file_sha256"] = file_sha256(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument(
        "--world",
        type=Path,
        action="append",
        required=True,
        help="One registered world directory; repeat for every seed in the split.",
    )
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--calibration-audit-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7_200.0)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--development-manifest-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = run_validation(
            split=args.split,
            world_dirs=args.world,
            frozen_config=args.frozen_config,
            calibration_audit=args.calibration_audit,
            calibration_audit_sha256=args.calibration_audit_sha256,
            out_dir=args.out_dir,
            timeout_seconds=args.timeout_seconds,
            development_manifest=args.development_manifest,
            development_manifest_sha256=args.development_manifest_sha256,
        )
    except ValidationError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "candidate_gate_pass": manifest["candidate_gate_pass"],
                "formal_certification_complete": manifest[
                    "formal_certification_complete"
                ],
                "manifest": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_file_sha256"],
            },
            indent=2,
        )
    )
    return 0 if manifest["formal_certification_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
