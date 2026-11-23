#!/usr/bin/env python3
"""Run one preregistered CA-BOSS v4 shared-state-subspace block safely.

The runner has a hard two-phase boundary:

1. generate or validate every registered world, run the frozen one-candidate
   local CA-BOSS discovery, and freeze the rotation-invariant subspace output;
2. after *all* attempts (including failures) are recorded in one global
   pretruth manifest, load every successful local graph and subspace candidate
   into memory before opening any simulator scorer directory.

The local graph retains the formal HARD preparation from
:mod:`benchmark_ca_boss_v4`.  The state-membership layer separately builds a
rank-three assay-local PCA embedding from held-out background archives only,
using two within-assay feature folds and no cross-assay alignment.  Neither
path receives paired cells, scorer roles, latent truth, or cross-assay feature
links.  The n-ary subspace output is scored without expansion into pairwise
cross-assay edges.

The validation constants live in :mod:`score_shared_state_subspace_v4`; this
module deliberately does not duplicate them.  A run registration content-
addresses the full executable source bundle, configs, protocol, paths, and
seed block before a missing world can be generated.  Resume reuses only
individually completed, hash-valid Phase-A attempts.  It never overwrites a
partial or completed scientific artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

import assay_local_state_embedding_v4 as assay_local_embedding
import benchmark_ca_boss_v4 as local_benchmark
from channel_aware_boss_v4 import CABossV4Config
import layered_state_evidence_v4 as layered_evidence
from multimodal_simulation_v4 import SimulationV4Config, simulate_world
import score_layered_state_evidence_v4 as layered_scorer
import score_shared_state_subspace_v4 as subspace_scorer
from shared_state_subspace_v4 import (
    SharedStateSubspaceConfig,
    run_shared_state_subspace,
)


RUNNER_SCHEMA_VERSION = "shared_state_subspace_v4.block_runner.v15"
REGISTRATION_SCHEMA_VERSION = "shared_state_subspace_v4.block_registration.v15"
ATTEMPT_SCHEMA_VERSION = "shared_state_subspace_v4.discovery_attempt.v15"
PHASE_A_SCHEMA_VERSION = "shared_state_subspace_v4.phase_a.v15"
RESULT_RECORDS_SCHEMA_VERSION = "shared_state_subspace_v4.result_records.v15"
RUN_SUMMARY_SCHEMA_VERSION = "shared_state_subspace_v4.block_run.v15"
LAYERED_GLOBAL_EXTENSION_SCHEMA_VERSION = (
    "layered_state_evidence_v4.global_pretruth_extension.v13"
)
LAYERED_CANDIDATE_SCHEMA_VERSION = (
    layered_evidence.CANDIDATE_SCHEMA_VERSION
)
LAYERED_PROVENANCE_SCHEMA_VERSION = layered_evidence.PROVENANCE_SCHEMA_VERSION
RUNNER_BASENAME = "run_shared_state_subspace_block_v4.py"
SUBSPACE_DIRECTORY_NAME = "shared_state_subspace_v4"
LAYERED_DIRECTORY_NAME = "layered_state_evidence_v4"
GLOBAL_FREEZE_NAME = "global_pretruth_freeze.json"
REGISTRATION_NAME = "block_registration.json"
PHASE_A_NAME = "phase_a_attempts.json"
RESULT_RECORDS_NAME = "block_result_records.json"
BLOCK_VALIDATION_NAME = "block_validation.json"
RUN_SUMMARY_NAME = "block_run_summary.json"
DEFAULT_EMPIRICAL_NULL_CALIBRATOR = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_nested_empirical_null_v425_full_pool"
    / "phase_b_full_calibrator.json"
)
DEFAULT_MULTIEVIDENCE_CALIBRATOR = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_nested_multievidence_v426_full_pool"
    / "phase_b_full_calibrator.json"
)
DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "ca_boss_v4"
    / "cached_nested_topology_consensus_v427_full_pool"
    / "phase_b_full_calibrator.json"
)
_WORLD_PATTERN = re.compile(r"^world_seed(\d{4,8})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUBSPACE_OUTPUT_NAMES = frozenset(
    {
        "subspace_feature_membership.tsv",
        "typed_subspace_hyperedges.tsv",
        "subspace_provenance.json",
        "frozen_discovery_manifest.json",
    }
)
_SUBSPACE_EMBEDDING_OUTPUT_NAMES = frozenset(
    {
        assay_local_embedding.EMBEDDING_NAME,
        assay_local_embedding.PROVENANCE_NAME,
        assay_local_embedding.MANIFEST_NAME,
        "frozen_evidence_manifest.sha256",
    }
)
_LAYERED_OUTPUT_NAMES = frozenset(
    {
        "layered_feature_evidence.tsv",
        "typed_layered_hyperedges.tsv",
        "within_assay_direct_candidates.tsv",
        "layered_provenance.json",
        "frozen_layered_candidate_manifest.json",
    }
)
_LAYERED_DATA_OUTPUT_NAMES = frozenset(
    _LAYERED_OUTPUT_NAMES.difference({"frozen_layered_candidate_manifest.json"})
)


@dataclass(frozen=True)
class LoadedLayeredWorld:
    """All truth-blind tables loaded globally before Phase-B scorer access."""

    subspace_membership: pd.DataFrame
    subspace_hyperedges: pd.DataFrame
    local_edges: pd.DataFrame
    layered_feature_evidence: pd.DataFrame
    layered_hyperedges: pd.DataFrame
    layered_direct_candidates: pd.DataFrame
    audit: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:  # pragma: no cover - numpy is a project dependency
        pass
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one JSON artifact without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_ready(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = subspace_scorer.canonical_sha256(result)
    return result


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> None:
    expected = str(value.get(field, ""))
    payload = dict(value)
    payload.pop(field, None)
    if not _SHA256_PATTERN.fullmatch(expected) or (
        subspace_scorer.canonical_sha256(payload) != expected
    ):
        raise ValueError(f"{label} self-hash mismatch")


def _normalise_split(split_name: str) -> str:
    normalized = str(split_name).strip().upper()
    normalized = {"DEV": "DEVELOPMENT", "FRESH_LOCKED": "LOCKED"}.get(
        normalized, normalized
    )
    if normalized not in subspace_scorer.SPLIT_SEEDS:
        raise ValueError(f"Unknown preregistered split: {split_name}")
    return normalized


def registered_seeds(split_name: str) -> tuple[int, ...]:
    """Return the exact scorer-owned seed block."""

    return tuple(int(seed) for seed in subspace_scorer.SPLIT_SEEDS[_normalise_split(split_name)])


def world_name(seed: int) -> str:
    return f"world_seed{int(seed)}"


def _assert_separate_roots(worlds_root: Path, output_root: Path) -> None:
    worlds = worlds_root.resolve(strict=False)
    output = output_root.resolve(strict=False)
    if worlds == output or worlds in output.parents or output in worlds.parents:
        raise ValueError("World and runner-output roots must be disjoint sibling trees")
    if worlds_root.is_symlink() or output_root.is_symlink():
        raise ValueError("World and runner-output roots may not be symlinks")


def registered_world_paths(split_name: str, worlds_root: Path | str) -> dict[int, Path]:
    """Resolve the exact seed block and reject unexpected world directories."""

    root = Path(worlds_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"World root must be a regular directory: {root}")
    expected = set(registered_seeds(split_name))
    unexpected: list[str] = []
    for path in root.iterdir():
        if not path.name.startswith("world_seed"):
            continue
        match = _WORLD_PATTERN.fullmatch(path.name)
        if match is None or int(match.group(1)) not in expected:
            unexpected.append(path.name)
        elif path.is_symlink():
            raise ValueError(f"Registered world may not be a symlink: {path}")
    if unexpected:
        raise ValueError(
            "World root contains entries outside the exact registered seed block: "
            f"{sorted(unexpected)}"
        )
    return {seed: root / world_name(seed) for seed in sorted(expected)}


def _development_binding(
    split_name: str, development_validation_path: Path | str | None
) -> tuple[Path | None, str | None]:
    split = _normalise_split(split_name)
    if split != "LOCKED":
        if development_validation_path is not None:
            raise ValueError("Only LOCKED may consume a DEVELOPMENT validation manifest")
        return None, None
    if development_validation_path is None:
        raise ValueError("LOCKED requires a passing DEVELOPMENT validation manifest")
    path = Path(development_validation_path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("DEVELOPMENT validation manifest must be a regular file")
    return path, subspace_scorer.file_sha256(path)


def _validate_development_before_locked(
    path: Path, registration: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse LOCKED generation/access unless DEVELOPMENT already passed."""

    manifest = _read_json(path, label="DEVELOPMENT validation manifest")
    if manifest.get("schema_version") != layered_scorer.BLOCK_VALIDATION_SCHEMA_VERSION:
        raise ValueError("LOCKED received an unsupported DEVELOPMENT validation schema")
    _validate_self_hash(
        manifest,
        field="block_validation_payload_sha256",
        label="DEVELOPMENT validation manifest",
    )
    required = {
        "split": "DEVELOPMENT",
        "block_certified": True,
        "formal_candidate_and_global_freeze_binding_validated": True,
        "layered_algorithm_source_sha256": registration[
            "layered_algorithm_source_sha256"
        ],
        "layered_algorithm_config_sha256": registration[
            "layered_algorithm_config_sha256"
        ],
        "layered_validation_protocol_payload_sha256": registration[
            "protocol_payload_sha256"
        ],
        "empirical_null_calibrator_artifact_sha256": registration[
            "empirical_null_calibrator_artifact_sha256"
        ],
        "empirical_null_calibrator_payload_sha256": registration[
            "empirical_null_calibrator_payload_sha256"
        ],
        "multievidence_calibrator_artifact_sha256": registration[
            "multievidence_calibrator_artifact_sha256"
        ],
        "multievidence_calibrator_payload_sha256": registration[
            "multievidence_calibrator_payload_sha256"
        ],
        "topology_consensus_calibrator_artifact_sha256": registration[
            "topology_consensus_calibrator_artifact_sha256"
        ],
        "topology_consensus_calibrator_payload_sha256": registration[
            "topology_consensus_calibrator_payload_sha256"
        ],
        "subspace_algorithm_source_sha256": registration[
            "algorithm_source_sha256"
        ],
        "subspace_algorithm_config_sha256": registration[
            "algorithm_config_sha256"
        ],
        "subspace_embedding_method": registration[
            "subspace_embedding_method"
        ],
        "subspace_embedding_config_sha256": registration[
            "subspace_embedding_config_sha256"
        ],
        "subspace_embedding_source_sha256": registration[
            "subspace_embedding_source_sha256"
        ],
        "local_candidate_name": registration["local_candidate_name"],
        "local_config_sha256": registration["local_config_sha256"],
        "joint_executable_source_sha256": registration[
            "joint_executable_source_sha256"
        ],
        "layered_candidate_identity_sha256": registration[
            "layered_candidate_identity_sha256"
        ],
    }
    mismatched = [key for key, expected in required.items() if manifest.get(key) != expected]
    if mismatched:
        raise ValueError(
            "LOCKED requires a passing DEVELOPMENT artifact from the identical "
            f"candidate/protocol: {mismatched}"
        )
    return manifest


def _frozen_empirical_null_calibrator(
    path: Path | str,
) -> tuple[
    layered_evidence.empirical_null_state.FrozenEmpiricalNullStateCalibrator,
    dict[str, Any],
]:
    safe = layered_evidence._safe_observable_path(
        Path(path), "empirical_null_calibrator"
    )
    payload = layered_evidence._read_regular_bytes_once(
        safe, "empirical_null_calibrator"
    )
    calibrator, metadata = layered_evidence.load_empirical_null_state_calibrator(
        payload
    )
    return calibrator, {
        **metadata,
        "path": str(safe),
        "artifact_file_sha256": layered_evidence._bytes_sha256(payload),
    }


def _frozen_multievidence_calibrator(
    path: Path | str,
) -> tuple[
    layered_evidence.multievidence_state.FrozenMultievidenceStateCalibrator,
    dict[str, Any],
]:
    safe = layered_evidence._safe_observable_path(
        Path(path), "multievidence_calibrator"
    )
    payload = layered_evidence._read_regular_bytes_once(
        safe, "multievidence_calibrator"
    )
    calibrator, metadata = layered_evidence.load_multievidence_state_calibrator(
        payload
    )
    return calibrator, {
        **metadata,
        "path": str(safe),
        "artifact_file_sha256": layered_evidence._bytes_sha256(payload),
    }


def _frozen_topology_consensus_calibrator(
    path: Path | str,
) -> tuple[
    layered_evidence.topology_consensus_state.FrozenTopologyConsensusStateCalibrator,
    dict[str, Any],
]:
    safe = layered_evidence._safe_observable_path(
        Path(path), "topology_consensus_calibrator"
    )
    payload = layered_evidence._read_regular_bytes_once(
        safe, "topology_consensus_calibrator"
    )
    calibrator, metadata = (
        layered_evidence.load_topology_consensus_state_calibrator(payload)
    )
    return calibrator, {
        **metadata,
        "path": str(safe),
        "artifact_file_sha256": layered_evidence._bytes_sha256(payload),
    }


def _current_registration(
    *,
    split_name: str,
    worlds_root: Path,
    output_root: Path,
    generate_missing: bool,
    development_validation_sha256: str | None,
    empirical_null_calibrator_path: Path | str = DEFAULT_EMPIRICAL_NULL_CALIBRATOR,
    multievidence_calibrator_path: Path | str = DEFAULT_MULTIEVIDENCE_CALIBRATOR,
    topology_consensus_calibrator_path: Path | str = (
        DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR
    ),
) -> dict[str, Any]:
    split = _normalise_split(split_name)
    scripts_dir = Path(__file__).resolve(strict=True).parent
    source_hashes = subspace_scorer.collect_joint_executable_source_sha256(
        scripts_dir
    )
    if set(source_hashes) != set(subspace_scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES):
        raise RuntimeError("Scorer did not return the exact executable source bundle")
    if source_hashes.get(RUNNER_BASENAME) != subspace_scorer.file_sha256(__file__):
        raise RuntimeError("Joint executable bundle does not bind this runner")
    algorithm_source = source_hashes.get("shared_state_subspace_v4.py", "")
    if algorithm_source != subspace_scorer.FROZEN_DISCOVERY_SOURCE_SHA256:
        raise RuntimeError(
            "Current subspace source differs from the scorer's preregistered source"
        )
    subspace_protocol = subspace_scorer.validation_protocol()
    layered_protocol = layered_scorer.layered_validation_protocol()
    layered_config = asdict(_frozen_layered_config())
    _, empirical_null_binding = _frozen_empirical_null_calibrator(
        empirical_null_calibrator_path
    )
    _, multievidence_binding = _frozen_multievidence_calibrator(
        multievidence_calibrator_path
    )
    _, topology_consensus_binding = _frozen_topology_consensus_calibrator(
        topology_consensus_calibrator_path
    )
    value: dict[str, Any] = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "split": split,
        "registered_world_seeds": list(registered_seeds(split)),
        "worlds_root": str(worlds_root.resolve(strict=False)),
        "output_root": str(output_root.resolve(strict=False)),
        "generate_missing_worlds": bool(generate_missing),
        "maximum_workers": 2,
        "algorithm_source_sha256": algorithm_source,
        "algorithm_config": dict(subspace_scorer.FROZEN_DISCOVERY_CONFIG),
        "algorithm_config_sha256": subspace_scorer.canonical_sha256(
            subspace_scorer.FROZEN_DISCOVERY_CONFIG
        ),
        "local_candidate_name": subspace_scorer.FROZEN_LOCAL_CANDIDATE_NAME,
        "local_config": dict(subspace_scorer.FROZEN_LOCAL_CONFIG),
        "local_config_sha256": subspace_scorer.canonical_sha256(
            subspace_scorer.FROZEN_LOCAL_CONFIG
        ),
        "subspace_embedding_method": (
            subspace_scorer.FROZEN_SUBSPACE_EMBEDDING_METHOD
        ),
        "subspace_embedding_config": dict(
            subspace_scorer.FROZEN_SUBSPACE_EMBEDDING_CONFIG
        ),
        "subspace_embedding_config_sha256": subspace_scorer.canonical_sha256(
            subspace_scorer.FROZEN_SUBSPACE_EMBEDDING_CONFIG
        ),
        "subspace_embedding_source_sha256": source_hashes[
            "assay_local_state_embedding_v4.py"
        ],
        "protocol_payload_sha256": layered_protocol["protocol_payload_sha256"],
        "subspace_diagnostic_protocol_payload_sha256": subspace_protocol[
            "protocol_payload_sha256"
        ],
        "layered_algorithm_source_sha256": source_hashes[
            "layered_state_evidence_v4.py"
        ],
        "layered_algorithm_config": layered_config,
        "layered_algorithm_config_sha256": subspace_scorer.canonical_sha256(
            layered_config
        ),
        "layered_scorer_source_sha256": source_hashes[
            "score_layered_state_evidence_v4.py"
        ],
        "empirical_null_calibrator_path": empirical_null_binding["path"],
        "empirical_null_calibrator_artifact_sha256": (
            empirical_null_binding["artifact_file_sha256"]
        ),
        "empirical_null_calibrator_payload_sha256": (
            empirical_null_binding["artifact_payload_sha256"]
        ),
        "empirical_null_numeric_calibrator_payload_sha256": (
            empirical_null_binding["numeric_calibrator_payload_sha256"]
        ),
        "empirical_null_calibrator_schema_version": (
            layered_evidence.empirical_null_state.SCHEMA
        ),
        "empirical_null_calibrator_application_requires_truth": False,
        "multievidence_calibrator_path": multievidence_binding["path"],
        "multievidence_calibrator_artifact_sha256": (
            multievidence_binding["artifact_file_sha256"]
        ),
        "multievidence_calibrator_payload_sha256": (
            multievidence_binding["artifact_payload_sha256"]
        ),
        "multievidence_numeric_calibrator_payload_sha256": (
            multievidence_binding["numeric_calibrator_payload_sha256"]
        ),
        "multievidence_calibrator_schema_version": (
            layered_evidence.multievidence_state.SCHEMA
        ),
        "multievidence_calibrator_application_requires_truth": False,
        "topology_consensus_calibrator_path": topology_consensus_binding[
            "path"
        ],
        "topology_consensus_calibrator_artifact_sha256": (
            topology_consensus_binding["artifact_file_sha256"]
        ),
        "topology_consensus_calibrator_payload_sha256": (
            topology_consensus_binding["artifact_payload_sha256"]
        ),
        "topology_consensus_numeric_calibrator_payload_sha256": (
            topology_consensus_binding["numeric_calibrator_payload_sha256"]
        ),
        "topology_consensus_calibrator_schema_version": (
            layered_evidence.topology_consensus_state.SCHEMA
        ),
        "topology_consensus_calibrator_sentinel_certified": True,
        "topology_consensus_calibrator_application_requires_truth": False,
        "loading_group_state_guard_source_sha256": source_hashes[
            "loading_group_state_guard_v4.py"
        ],
        "loading_group_state_guard_schema_version": (
            layered_evidence.loading_group_state_guard.SCHEMA
        ),
        "loading_group_state_guard_fixed_rule": (
            layered_evidence.loading_group_state_guard.FIXED_RULE
        ),
        "loading_group_state_guard_state_route": (
            layered_evidence.loading_group_state_guard.STATE_ROUTE
        ),
        "loading_group_state_guard_axis_route": (
            layered_evidence.loading_group_state_guard.AXIS_ROUTE
        ),
        "loading_group_state_guard_state_route_requires_all_v427_members": True,
        "loading_group_state_guard_axis_route_requires_state_v427_anchor": True,
        "loading_group_state_guard_axis_route_requires_leave_one_group_out_"
        "cross_assay_consensus": True,
        "loading_group_state_guard_axis_route_changes_state_v427_membership": False,
        "loading_group_direct_structure_alone_is_insufficient": True,
        "loading_group_structure_uncertainty_guard_source_sha256": source_hashes[
            "loading_group_structure_uncertainty_guard_v4.py"
        ],
        "loading_group_structure_uncertainty_guard_schema_version": (
            layered_evidence.loading_group_structure_guard.SCHEMA
        ),
        "loading_group_structure_uncertainty_guard_fixed_rule": (
            layered_evidence.loading_group_structure_guard.FIXED_RULE
        ),
        "loading_group_minimum_state_route_member_centroid_cosine": (
            layered_evidence.loading_group_structure_guard
            .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        ),
        "loading_group_axis_route_bypasses_uncertainty_floor": True,
        "loading_group_both_routes_subject_to_direct_overlap_downgrade": True,
        "loading_group_direct_overlap_downgrade_means_absence": False,
        "loading_group_structure_guard_changes_state_v427_membership": False,
        "loading_group_structure_guard_changes_direct_candidates": False,
        "loading_group_competing_parent_observability_guard_source_sha256": source_hashes[
            "loading_group_competing_parent_observability_guard_v4.py"
        ],
        "loading_group_competing_parent_observability_guard_schema_version": (
            layered_evidence.loading_group_competitor_guard.SCHEMA
        ),
        "loading_group_competing_parent_observability_guard_fixed_rule": (
            layered_evidence.loading_group_competitor_guard.FIXED_RULE
        ),
        "loading_group_state_route_requires_exact_structure_anchor_competition": True,
        "loading_group_competing_parent_uses_frozen_v427_observability_rule": True,
        "loading_group_axis_route_bypasses_competing_parent_gate": True,
        "loading_group_competing_parent_downgrade_means_absence": False,
        "loading_group_competing_parent_guard_changes_state_v427_membership": False,
        "loading_group_competing_parent_guard_changes_direct_candidates": False,
        "loading_group_structure_equivalence_source_sha256": source_hashes[
            "loading_group_structure_equivalence_v4.py"
        ],
        "loading_group_structure_equivalence_schema_version": (
            layered_evidence.loading_group_structure_equivalence.SCHEMA
        ),
        "loading_group_structure_equivalence_fixed_rule": (
            layered_evidence.loading_group_structure_equivalence.FIXED_RULE
        ),
        "loading_group_state_route_candidate_pair_count": 4,
        "loading_group_state_route_exact_pair_identified": False,
        "loading_group_candidate_pairs_may_be_expanded_as_edges": False,
        "loading_group_axis_route_candidate_pair_count": 1,
        "loading_group_structure_equivalence_changes_v431_selection": False,
        "loading_group_structure_equivalence_changes_state_v427_membership": False,
        "loading_group_structure_equivalence_changes_direct_candidates": False,
        "loading_group_structure_equivalence_application_requires_truth": False,
        "loading_group_structure_equivalence_creates_cross_assay_relations": False,
        "joint_executable_source_sha256": source_hashes,
        "development_validation_manifest_sha256": development_validation_sha256,
        "frozen_before_missing_world_generation": True,
        "frozen_before_any_registered_scorer_read": True,
        "local_structure_alignment_mode": "hard",
        "subspace_membership_cross_assay_alignment_used": False,
        "layered_output_is_primary_validation_estimand": True,
        "full_subspace_completeness_is_diagnostic_only": True,
        "paired_cell_inputs_allowed": False,
        "external_or_scorer_inputs_allowed_during_discovery": False,
    }
    value["layered_candidate_identity_sha256"] = (
        subspace_scorer.canonical_sha256(_layered_candidate_identity(value))
    )
    return _self_hashed(value, "registration_payload_sha256")


def _load_or_create_registration(
    *,
    split_name: str,
    worlds_root: Path,
    output_root: Path,
    generate_missing: bool,
    development_validation_sha256: str | None,
    empirical_null_calibrator_path: Path | str,
    multievidence_calibrator_path: Path | str,
    topology_consensus_calibrator_path: Path | str,
    resume: bool,
) -> dict[str, Any]:
    _assert_separate_roots(worlds_root, output_root)
    expected = _current_registration(
        split_name=split_name,
        worlds_root=worlds_root,
        output_root=output_root,
        generate_missing=generate_missing,
        development_validation_sha256=development_validation_sha256,
        empirical_null_calibrator_path=empirical_null_calibrator_path,
        multievidence_calibrator_path=multievidence_calibrator_path,
        topology_consensus_calibrator_path=(
            topology_consensus_calibrator_path
        ),
    )
    registration_path = output_root / REGISTRATION_NAME
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise ValueError("Runner output root must be a regular directory")
    if registration_path.exists():
        if not resume:
            raise FileExistsError(
                "Runner output is already registered; pass --resume to verify and reuse it"
            )
        observed = _read_json(registration_path, label="block registration")
        _validate_self_hash(
            observed,
            field="registration_payload_sha256",
            label="block registration",
        )
        if observed != expected:
            raise RuntimeError(
                "Refusing resume because source, config, protocol, paths, or mode changed"
            )
        return observed
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            "Refusing unregistered non-empty runner output; choose an empty directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_new_json(registration_path, expected)
    return expected


def _verify_registration_is_current(registration: Mapping[str, Any]) -> None:
    """Fail closed if executable code, configs, or protocol changed mid-run."""

    scripts_dir = Path(__file__).resolve(strict=True).parent
    observed_sources = subspace_scorer.collect_joint_executable_source_sha256(
        scripts_dir
    )
    _, empirical_null_binding = _frozen_empirical_null_calibrator(
        str(registration["empirical_null_calibrator_path"])
    )
    _, multievidence_binding = _frozen_multievidence_calibrator(
        str(registration["multievidence_calibrator_path"])
    )
    _, topology_consensus_binding = _frozen_topology_consensus_calibrator(
        str(registration["topology_consensus_calibrator_path"])
    )
    checks = {
        "joint executable source": (
            dict(registration["joint_executable_source_sha256"]),
            observed_sources,
        ),
        "subspace config": (
            registration["algorithm_config_sha256"],
            subspace_scorer.canonical_sha256(
                subspace_scorer.FROZEN_DISCOVERY_CONFIG
            ),
        ),
        "local config": (
            registration["local_config_sha256"],
            subspace_scorer.canonical_sha256(subspace_scorer.FROZEN_LOCAL_CONFIG),
        ),
        "subspace embedding config": (
            registration["subspace_embedding_config_sha256"],
            subspace_scorer.canonical_sha256(
                subspace_scorer.FROZEN_SUBSPACE_EMBEDDING_CONFIG
            ),
        ),
        "subspace embedding method": (
            registration["subspace_embedding_method"],
            subspace_scorer.FROZEN_SUBSPACE_EMBEDDING_METHOD,
        ),
        "subspace embedding source": (
            registration["subspace_embedding_source_sha256"],
            observed_sources["assay_local_state_embedding_v4.py"],
        ),
        "layered config": (
            registration["layered_algorithm_config_sha256"],
            subspace_scorer.canonical_sha256(asdict(_frozen_layered_config())),
        ),
        "layered source": (
            registration["layered_algorithm_source_sha256"],
            observed_sources["layered_state_evidence_v4.py"],
        ),
        "layered scorer source": (
            registration["layered_scorer_source_sha256"],
            observed_sources["score_layered_state_evidence_v4.py"],
        ),
        "empirical-null calibrator artifact": (
            registration["empirical_null_calibrator_artifact_sha256"],
            empirical_null_binding["artifact_file_sha256"],
        ),
        "empirical-null calibrator payload": (
            registration["empirical_null_calibrator_payload_sha256"],
            empirical_null_binding["artifact_payload_sha256"],
        ),
        "empirical-null numeric calibrator": (
            registration[
                "empirical_null_numeric_calibrator_payload_sha256"
            ],
            empirical_null_binding["numeric_calibrator_payload_sha256"],
        ),
        "multievidence calibrator artifact": (
            registration["multievidence_calibrator_artifact_sha256"],
            multievidence_binding["artifact_file_sha256"],
        ),
        "multievidence calibrator payload": (
            registration["multievidence_calibrator_payload_sha256"],
            multievidence_binding["artifact_payload_sha256"],
        ),
        "multievidence numeric calibrator": (
            registration[
                "multievidence_numeric_calibrator_payload_sha256"
            ],
            multievidence_binding["numeric_calibrator_payload_sha256"],
        ),
        "topology-consensus calibrator artifact": (
            registration["topology_consensus_calibrator_artifact_sha256"],
            topology_consensus_binding["artifact_file_sha256"],
        ),
        "topology-consensus calibrator payload": (
            registration["topology_consensus_calibrator_payload_sha256"],
            topology_consensus_binding["artifact_payload_sha256"],
        ),
        "topology-consensus numeric calibrator": (
            registration[
                "topology_consensus_numeric_calibrator_payload_sha256"
            ],
            topology_consensus_binding[
                "numeric_calibrator_payload_sha256"
            ],
        ),
        "layered validation protocol": (
            registration["protocol_payload_sha256"],
            layered_scorer.layered_validation_protocol()[
                "protocol_payload_sha256"
            ],
        ),
        "subspace diagnostic protocol": (
            registration["subspace_diagnostic_protocol_payload_sha256"],
            subspace_scorer.validation_protocol()["protocol_payload_sha256"],
        ),
    }
    changed = [label for label, (expected, observed) in checks.items() if expected != observed]
    if changed:
        raise RuntimeError(
            "Refusing to cross the global freeze/scorer boundary after mid-run "
            f"changes: {changed}"
        )


def _verify_global_sources_are_current(global_manifest: Mapping[str, Any]) -> None:
    observed = subspace_scorer.collect_joint_executable_source_sha256(
        Path(__file__).resolve(strict=True).parent
    )
    if observed != dict(global_manifest["joint_executable_source_sha256"]):
        raise RuntimeError("Executable source changed after the global pretruth freeze")


def _frozen_local_candidate() -> list[tuple[str, CABossV4Config]]:
    config = CABossV4Config(**dict(subspace_scorer.FROZEN_LOCAL_CONFIG))
    if asdict(config) != dict(subspace_scorer.FROZEN_LOCAL_CONFIG):
        raise RuntimeError("Scorer/local CABossV4Config contracts differ")
    if config.alignment_mode != "hard":
        raise RuntimeError("Formal subspace block requires HARD local preparation")
    return [(subspace_scorer.FROZEN_LOCAL_CANDIDATE_NAME, config)]


def _frozen_subspace_config() -> SharedStateSubspaceConfig:
    config = SharedStateSubspaceConfig(
        **dict(subspace_scorer.FROZEN_DISCOVERY_CONFIG)
    )
    if asdict(config) != dict(subspace_scorer.FROZEN_DISCOVERY_CONFIG):
        raise RuntimeError("Scorer/discovery SharedStateSubspaceConfig contracts differ")
    return config


def _frozen_layered_config() -> layered_evidence.LayeredStateEvidenceConfig:
    config = layered_evidence.LayeredStateEvidenceConfig()
    expected = {
        "maximum_confirmatory_feature_maxT_p": 0.10,
        "maximum_state_equivalence_maxT_p": 0.01,
        "maximum_assay_participation_error_probability": 0.10,
        "minimum_confirmed_modalities": 2,
    }
    if asdict(config) != expected:
        raise RuntimeError("Layered evidence defaults differ from the formal contract")
    return config


def _artifact_paths(output_root: Path, seed: int) -> dict[str, Path]:
    world_out = output_root / "artifacts" / world_name(seed)
    local_candidate = world_out / subspace_scorer.FROZEN_LOCAL_CANDIDATE_NAME
    subspace_out = world_out / SUBSPACE_DIRECTORY_NAME
    layered_out = world_out / LAYERED_DIRECTORY_NAME
    subspace_embedding_root = world_out / "assay_local_state_embedding_v4"
    return {
        "world_out": world_out,
        "prepared_embedding": world_out / "prepared_discovery" / "analysis_embeddings.tsv",
        "subspace_embedding_root": subspace_embedding_root,
        "subspace_embedding": subspace_embedding_root
        / assay_local_embedding.EMBEDDING_NAME,
        "subspace_embedding_manifest": subspace_embedding_root
        / assay_local_embedding.MANIFEST_NAME,
        "subspace_embedding_provenance": subspace_embedding_root
        / assay_local_embedding.PROVENANCE_NAME,
        "subspace_embedding_sidecar": subspace_embedding_root
        / "frozen_evidence_manifest.sha256",
        "local_pretruth_manifest": world_out / "pretruth_consensus_manifest.json",
        "local_consensus_edges": local_candidate / "consensus_edges.tsv",
        "subspace_out": subspace_out,
        "candidate_manifest": subspace_out / "frozen_discovery_manifest.json",
        "feature_membership": subspace_out / "subspace_feature_membership.tsv",
        "typed_subspace_hyperedges": subspace_out / "typed_subspace_hyperedges.tsv",
        "loading_groups": local_candidate
        / "loading_group_projection"
        / "loading_groups.tsv",
        "loading_group_membership": local_candidate
        / "loading_group_projection"
        / "loading_group_membership.tsv",
        "structure_anchor_pair_evidence": local_candidate
        / "structure_anchored_loading_groups"
        / "structure_anchor_pair_evidence.tsv",
        "group_axis_evidence": local_candidate
        / "latent_axis_projection"
        / "group_axis_evidence.tsv",
        "axis_candidates": local_candidate
        / "latent_axis_projection"
        / "axis_candidates.tsv",
        "layered_out": layered_out,
        "layered_candidate_manifest": layered_out
        / "frozen_layered_candidate_manifest.json",
        "layered_feature_evidence": layered_out / "layered_feature_evidence.tsv",
        "typed_layered_hyperedges": layered_out / "typed_layered_hyperedges.tsv",
        "layered_direct_candidates": layered_out
        / "within_assay_direct_candidates.tsv",
        "layered_provenance": layered_out / "layered_provenance.json",
    }


def _attempt_path(output_root: Path, seed: int) -> Path:
    return output_root / "attempts" / f"{world_name(seed)}.json"


def _layered_output_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        "layered_feature_evidence.tsv": subspace_scorer.file_sha256(
            paths["layered_feature_evidence"]
        ),
        "layered_provenance.json": subspace_scorer.file_sha256(
            paths["layered_provenance"]
        ),
        "typed_layered_hyperedges.tsv": subspace_scorer.file_sha256(
            paths["typed_layered_hyperedges"]
        ),
        "within_assay_direct_candidates.tsv": subspace_scorer.file_sha256(
            paths["layered_direct_candidates"]
        ),
    }


def _attempt_payload(
    *,
    seed: int,
    registration_sha256: str,
    status: str,
    error: BaseException | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "world_seed": int(seed),
        "status": status,
        "registration_payload_sha256": registration_sha256,
        "scorer_domain_read_during_discovery": False,
    }
    if status == "ok":
        if output_root is None:
            raise ValueError("Successful attempt requires its artifact root")
        paths = _artifact_paths(output_root, seed)
        value.update(
            {
                "candidate_manifest_sha256": subspace_scorer.file_sha256(
                    paths["candidate_manifest"]
                ),
                "subspace_embedding_manifest_sha256": (
                    subspace_scorer.file_sha256(
                        paths["subspace_embedding_manifest"]
                    )
                ),
                "subspace_embedding_sha256": subspace_scorer.file_sha256(
                    paths["subspace_embedding"]
                ),
                "local_pretruth_manifest_sha256": subspace_scorer.file_sha256(
                    paths["local_pretruth_manifest"]
                ),
                "local_consensus_edges_sha256": subspace_scorer.file_sha256(
                    paths["local_consensus_edges"]
                ),
                "layered_candidate_manifest_sha256": (
                    subspace_scorer.file_sha256(
                        paths["layered_candidate_manifest"]
                    )
                ),
                "layered_output_sha256": _layered_output_hashes(paths),
                "layered_algorithm_source_sha256": subspace_scorer.file_sha256(
                    Path(layered_evidence.__file__).resolve(strict=True)
                ),
                "layered_algorithm_config_sha256": (
                    subspace_scorer.canonical_sha256(
                        asdict(_frozen_layered_config())
                    )
                ),
            }
        )
    elif error is not None:
        value.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
            }
        )
    return _self_hashed(value, "attempt_payload_sha256")


def _validate_attempt_record(
    record: Mapping[str, Any],
    *,
    seed: int,
    registration_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    if record.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        raise ValueError(f"Attempt {seed} has an unsupported schema")
    _validate_self_hash(record, field="attempt_payload_sha256", label=f"attempt {seed}")
    if int(record.get("world_seed", -1)) != int(seed):
        raise ValueError(f"Attempt record seed mismatch for {seed}")
    if record.get("registration_payload_sha256") != registration_sha256:
        raise ValueError(f"Attempt {seed} belongs to another registration")
    if record.get("scorer_domain_read_during_discovery") is not False:
        raise ValueError(f"Attempt {seed} reports scorer access during discovery")
    status = str(record.get("status", "")).casefold()
    if status not in {"ok", "timeout", "crash", "invalid", "error"}:
        raise ValueError(f"Attempt {seed} has an unsupported status")
    paths = _artifact_paths(output_root, seed)
    hash_fields = {
        "candidate_manifest_sha256": paths["candidate_manifest"],
        "subspace_embedding_manifest_sha256": paths[
            "subspace_embedding_manifest"
        ],
        "subspace_embedding_sha256": paths["subspace_embedding"],
        "local_pretruth_manifest_sha256": paths["local_pretruth_manifest"],
        "local_consensus_edges_sha256": paths["local_consensus_edges"],
        "layered_candidate_manifest_sha256": paths[
            "layered_candidate_manifest"
        ],
    }
    if status == "ok":
        for field, path in hash_fields.items():
            expected = str(record.get(field, ""))
            if not _SHA256_PATTERN.fullmatch(expected):
                raise ValueError(f"Attempt {seed} lacks {field}")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Attempt {seed} lacks frozen artifact {path.name}")
            if subspace_scorer.file_sha256(path) != expected:
                raise ValueError(f"Attempt {seed} frozen artifact changed: {path.name}")
        subspace_root = paths["subspace_out"]
        actual = {
            path.name
            for path in subspace_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if actual != _SUBSPACE_OUTPUT_NAMES:
            raise ValueError(f"Attempt {seed} subspace output set is not exact")
        embedding_root = paths["subspace_embedding_root"]
        embedding_actual = {
            path.name
            for path in embedding_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if embedding_actual != _SUBSPACE_EMBEDDING_OUTPUT_NAMES:
            raise ValueError(
                f"Attempt {seed} subspace-embedding output set is not exact"
            )
        layered_root = paths["layered_out"]
        if layered_root.is_symlink() or not layered_root.is_dir():
            raise ValueError(f"Attempt {seed} lacks a regular layered output root")
        layered_actual = {
            path.name
            for path in layered_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if layered_actual != _LAYERED_OUTPUT_NAMES:
            raise ValueError(f"Attempt {seed} layered output set is not exact")
        observed_layered_hashes = record.get("layered_output_sha256")
        if not isinstance(observed_layered_hashes, Mapping) or set(
            observed_layered_hashes
        ) != set(_LAYERED_DATA_OUTPUT_NAMES):
            raise ValueError(f"Attempt {seed} lacks exact layered output hashes")
        expected_layered_hashes = _layered_output_hashes(paths)
        if dict(observed_layered_hashes) != expected_layered_hashes:
            raise ValueError(f"Attempt {seed} frozen layered artifact changed")
        layered_manifest = _read_json(
            paths["layered_candidate_manifest"],
            label=f"attempt {seed} layered candidate manifest",
        )
        if (
            layered_manifest.get("schema_version")
            != LAYERED_CANDIDATE_SCHEMA_VERSION
            or layered_manifest.get("candidate_frozen") is not True
            or layered_manifest.get("truth_or_scorer_consumed_by_algorithm")
            is not False
        ):
            raise ValueError(
                f"Attempt {seed} layered candidate is unsupported or not truth-blind"
            )
        expected_source = subspace_scorer.file_sha256(
            Path(layered_evidence.__file__).resolve(strict=True)
        )
        expected_config = subspace_scorer.canonical_sha256(
            asdict(_frozen_layered_config())
        )
        if (
            record.get("layered_algorithm_source_sha256") != expected_source
            or layered_manifest.get("algorithm_source_sha256") != expected_source
        ):
            raise ValueError(f"Attempt {seed} binds another layered source")
        if (
            record.get("layered_algorithm_config_sha256") != expected_config
            or layered_manifest.get("config_sha256") != expected_config
            or layered_manifest.get("config") != asdict(_frozen_layered_config())
        ):
            raise ValueError(f"Attempt {seed} binds another layered config")
        if layered_manifest.get("output_sha256") != expected_layered_hashes:
            raise ValueError(f"Attempt {seed} layered manifest output hashes changed")
    elif any(field in record for field in hash_fields) or any(
        field in record
        for field in (
            "layered_output_sha256",
            "layered_algorithm_source_sha256",
            "layered_algorithm_config_sha256",
        )
    ):
        raise ValueError(f"Failed attempt {seed} unexpectedly binds frozen artifacts")
    return dict(record)


def _classify_failure(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (ValueError, FileNotFoundError, FileExistsError)):
        return "invalid"
    return "error"


def _discover_one(
    *,
    seed: int,
    worlds_root: Path,
    output_root: Path,
    registration_sha256: str,
    generate_missing: bool,
    empirical_null_calibrator_path: Path | str = DEFAULT_EMPIRICAL_NULL_CALIBRATOR,
    multievidence_calibrator_path: Path | str = DEFAULT_MULTIEVIDENCE_CALIBRATOR,
    topology_consensus_calibrator_path: Path | str = (
        DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR
    ),
) -> dict[str, Any]:
    """Run one truth-blind Phase-A attempt and persist its terminal record."""

    attempt_path = _attempt_path(output_root, seed)
    if attempt_path.exists():
        return _validate_attempt_record(
            _read_json(attempt_path, label=f"attempt {seed}"),
            seed=seed,
            registration_sha256=registration_sha256,
            output_root=output_root,
        )
    world_dir = worlds_root / world_name(seed)
    paths = _artifact_paths(output_root, seed)
    try:
        if not world_dir.exists():
            if not generate_missing:
                raise FileNotFoundError(f"Registered world is missing: {world_dir}")
            simulate_world(SimulationV4Config(seed=seed), world_dir)
        if paths["world_out"].exists():
            raise FileExistsError(
                "Refusing to overwrite partial unrecorded discovery output: "
                f"{paths['world_out']}"
            )
        # This validation intentionally leaves verify_scorer_payload=False.
        local_benchmark.validate_world_input_binding(
            world_dir, verify_scorer_payload=False
        )
        assay_local_embedding.write_assay_local_state_embedding(
            world_dir / "discovery",
            paths["subspace_embedding_root"],
        )
        # Freeze the independent state-membership decomposition before local
        # structure uses it to nominate non-collider state reporters.
        run_shared_state_subspace(
            world_dir / "discovery",
            paths["subspace_embedding"],
            paths["subspace_out"],
            config=_frozen_subspace_config(),
        )
        candidates = _frozen_local_candidate()
        frozen = local_benchmark.discover_world_pretruth(
            world_dir,
            candidates,
            out_dir=output_root / "artifacts",
            save_candidate_names={subspace_scorer.FROZEN_LOCAL_CANDIDATE_NAME},
            loading_embedding_path=paths["subspace_embedding"],
            state_membership_path=paths["feature_membership"],
        )
        if tuple(frozen.candidate_names) != (
            subspace_scorer.FROZEN_LOCAL_CANDIDATE_NAME,
        ):
            raise RuntimeError("Local benchmark froze a different candidate set")
        if Path(frozen.pretruth_manifest).resolve() != paths[
            "local_pretruth_manifest"
        ].resolve():
            raise RuntimeError("Local benchmark returned an unexpected output layout")
        layered_evidence.run_layered_state_evidence(
            paths["feature_membership"],
            paths["typed_subspace_hyperedges"],
            paths["local_consensus_edges"],
            paths["loading_groups"],
            paths["loading_group_membership"],
            paths["structure_anchor_pair_evidence"],
            paths["layered_out"],
            config=_frozen_layered_config(),
            empirical_null_calibrator_path=Path(
                empirical_null_calibrator_path
            ),
            multievidence_calibrator_path=Path(
                multievidence_calibrator_path
            ),
            topology_consensus_calibrator_path=Path(
                topology_consensus_calibrator_path
            ),
            group_axis_evidence_path=paths["group_axis_evidence"],
            axis_candidates_path=paths["axis_candidates"],
        )
        record = _attempt_payload(
            seed=seed,
            registration_sha256=registration_sha256,
            status="ok",
            output_root=output_root,
        )
        record = _validate_attempt_record(
            record,
            seed=seed,
            registration_sha256=registration_sha256,
            output_root=output_root,
        )
    except Exception as exc:  # retain every registered failure in the denominator
        record = _attempt_payload(
            seed=seed,
            registration_sha256=registration_sha256,
            status=_classify_failure(exc),
            error=exc,
        )
    _write_new_json(attempt_path, record)
    return record


def _discover_one_worker(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return _discover_one(
        seed=int(arguments["seed"]),
        worlds_root=Path(str(arguments["worlds_root"])),
        output_root=Path(str(arguments["output_root"])),
        registration_sha256=str(arguments["registration_sha256"]),
        generate_missing=bool(arguments["generate_missing"]),
        empirical_null_calibrator_path=str(
            arguments["empirical_null_calibrator_path"]
        ),
        multievidence_calibrator_path=str(
            arguments["multievidence_calibrator_path"]
        ),
        topology_consensus_calibrator_path=str(
            arguments["topology_consensus_calibrator_path"]
        ),
    )


def run_discovery_phase(
    *,
    split_name: str,
    worlds_root: Path,
    output_root: Path,
    registration: Mapping[str, Any],
    generate_missing: bool,
    workers: int = 2,
) -> list[dict[str, Any]]:
    """Attempt every registered world without opening any scorer payload."""

    if not 1 <= int(workers) <= 2:
        raise ValueError("workers must be one or two")
    seeds = registered_seeds(split_name)
    registration_sha256 = str(registration["registration_payload_sha256"])
    arguments = [
        {
            "seed": seed,
            "worlds_root": str(worlds_root),
            "output_root": str(output_root),
            "registration_sha256": registration_sha256,
            "generate_missing": generate_missing,
            "empirical_null_calibrator_path": registration[
                "empirical_null_calibrator_path"
            ],
            "multievidence_calibrator_path": registration[
                "multievidence_calibrator_path"
            ],
            "topology_consensus_calibrator_path": registration[
                "topology_consensus_calibrator_path"
            ],
        }
        for seed in seeds
    ]
    records: dict[int, dict[str, Any]] = {}
    if workers == 1:
        for argument in arguments:
            record = _discover_one_worker(argument)
            records[int(record["world_seed"])] = record
            print(
                f"[Phase A] {world_name(int(record['world_seed']))}: {record['status']}",
                file=sys.stderr,
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_seed = {
                executor.submit(_discover_one_worker, argument): int(argument["seed"])
                for argument in arguments
            }
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    record = future.result()
                except BaseException as exc:  # a dead worker is still a registered failure
                    record = _attempt_payload(
                        seed=seed,
                        registration_sha256=registration_sha256,
                        status="crash",
                        error=exc,
                    )
                    attempt_path = _attempt_path(output_root, seed)
                    if not attempt_path.exists():
                        _write_new_json(attempt_path, record)
                records[seed] = record
                print(
                    f"[Phase A] {world_name(seed)}: {record['status']}",
                    file=sys.stderr,
                    flush=True,
                )
    if set(records) != set(seeds):
        raise RuntimeError("Phase A did not retain every registered discovery attempt")
    ordered = [records[seed] for seed in seeds]
    phase_a = _self_hashed(
        {
            "schema_version": PHASE_A_SCHEMA_VERSION,
            "split": _normalise_split(split_name),
            "registered_world_seeds": list(seeds),
            "registration_payload_sha256": registration_sha256,
            "all_registered_discovery_attempts_finished": True,
            "all_attempts_completed_before_any_scorer_read": True,
            "scorer_domain_opened": False,
            "attempts": ordered,
        },
        "phase_a_payload_sha256",
    )
    phase_path = output_root / PHASE_A_NAME
    if phase_path.exists():
        observed = _read_json(phase_path, label="Phase-A manifest")
        _validate_self_hash(
            observed, field="phase_a_payload_sha256", label="Phase-A manifest"
        )
        if observed != phase_a:
            raise RuntimeError("Existing Phase-A manifest differs from current attempts")
    else:
        _write_new_json(phase_path, phase_a)
    return ordered


def _layered_candidate_identity(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "layered_algorithm_source_sha256": registration[
            "layered_algorithm_source_sha256"
        ],
        "layered_algorithm_config_sha256": registration[
            "layered_algorithm_config_sha256"
        ],
        "layered_scorer_source_sha256": registration[
            "layered_scorer_source_sha256"
        ],
        "layered_validation_protocol_payload_sha256": registration[
            "protocol_payload_sha256"
        ],
        "empirical_null_calibrator_artifact_sha256": registration[
            "empirical_null_calibrator_artifact_sha256"
        ],
        "empirical_null_calibrator_payload_sha256": registration[
            "empirical_null_calibrator_payload_sha256"
        ],
        "empirical_null_numeric_calibrator_payload_sha256": registration[
            "empirical_null_numeric_calibrator_payload_sha256"
        ],
        "empirical_null_calibrator_schema_version": registration[
            "empirical_null_calibrator_schema_version"
        ],
        "empirical_null_calibrator_path": registration[
            "empirical_null_calibrator_path"
        ],
        "multievidence_calibrator_artifact_sha256": registration[
            "multievidence_calibrator_artifact_sha256"
        ],
        "multievidence_calibrator_payload_sha256": registration[
            "multievidence_calibrator_payload_sha256"
        ],
        "multievidence_numeric_calibrator_payload_sha256": registration[
            "multievidence_numeric_calibrator_payload_sha256"
        ],
        "multievidence_calibrator_schema_version": registration[
            "multievidence_calibrator_schema_version"
        ],
        "multievidence_calibrator_path": registration[
            "multievidence_calibrator_path"
        ],
        "topology_consensus_calibrator_artifact_sha256": registration[
            "topology_consensus_calibrator_artifact_sha256"
        ],
        "topology_consensus_calibrator_payload_sha256": registration[
            "topology_consensus_calibrator_payload_sha256"
        ],
        "topology_consensus_numeric_calibrator_payload_sha256": registration[
            "topology_consensus_numeric_calibrator_payload_sha256"
        ],
        "topology_consensus_calibrator_schema_version": registration[
            "topology_consensus_calibrator_schema_version"
        ],
        "topology_consensus_calibrator_sentinel_certified": registration[
            "topology_consensus_calibrator_sentinel_certified"
        ],
        "topology_consensus_calibrator_path": registration[
            "topology_consensus_calibrator_path"
        ],
        "loading_group_state_guard_source_sha256": registration[
            "loading_group_state_guard_source_sha256"
        ],
        "loading_group_state_guard_schema_version": registration[
            "loading_group_state_guard_schema_version"
        ],
        "loading_group_state_guard_fixed_rule": registration[
            "loading_group_state_guard_fixed_rule"
        ],
        "loading_group_state_guard_state_route": registration[
            "loading_group_state_guard_state_route"
        ],
        "loading_group_state_guard_axis_route": registration[
            "loading_group_state_guard_axis_route"
        ],
        "loading_group_state_guard_state_route_requires_all_v427_members": registration[
            "loading_group_state_guard_state_route_requires_all_v427_members"
        ],
        "loading_group_state_guard_axis_route_requires_state_v427_anchor": registration[
            "loading_group_state_guard_axis_route_requires_state_v427_anchor"
        ],
        "loading_group_state_guard_axis_route_requires_leave_one_group_out_"
        "cross_assay_consensus": registration[
            "loading_group_state_guard_axis_route_requires_leave_one_group_out_"
            "cross_assay_consensus"
        ],
        "loading_group_state_guard_axis_route_changes_state_v427_membership": registration[
            "loading_group_state_guard_axis_route_changes_state_v427_membership"
        ],
        "loading_group_direct_structure_alone_is_insufficient": registration[
            "loading_group_direct_structure_alone_is_insufficient"
        ],
        "loading_group_structure_uncertainty_guard_source_sha256": registration[
            "loading_group_structure_uncertainty_guard_source_sha256"
        ],
        "loading_group_structure_uncertainty_guard_schema_version": registration[
            "loading_group_structure_uncertainty_guard_schema_version"
        ],
        "loading_group_structure_uncertainty_guard_fixed_rule": registration[
            "loading_group_structure_uncertainty_guard_fixed_rule"
        ],
        "loading_group_minimum_state_route_member_centroid_cosine": registration[
            "loading_group_minimum_state_route_member_centroid_cosine"
        ],
        "loading_group_axis_route_bypasses_uncertainty_floor": registration[
            "loading_group_axis_route_bypasses_uncertainty_floor"
        ],
        "loading_group_both_routes_subject_to_direct_overlap_downgrade": registration[
            "loading_group_both_routes_subject_to_direct_overlap_downgrade"
        ],
        "loading_group_direct_overlap_downgrade_means_absence": registration[
            "loading_group_direct_overlap_downgrade_means_absence"
        ],
        "loading_group_structure_guard_changes_state_v427_membership": registration[
            "loading_group_structure_guard_changes_state_v427_membership"
        ],
        "loading_group_structure_guard_changes_direct_candidates": registration[
            "loading_group_structure_guard_changes_direct_candidates"
        ],
        "loading_group_competing_parent_observability_guard_source_sha256": registration[
            "loading_group_competing_parent_observability_guard_source_sha256"
        ],
        "loading_group_competing_parent_observability_guard_schema_version": registration[
            "loading_group_competing_parent_observability_guard_schema_version"
        ],
        "loading_group_competing_parent_observability_guard_fixed_rule": registration[
            "loading_group_competing_parent_observability_guard_fixed_rule"
        ],
        "loading_group_state_route_requires_exact_structure_anchor_competition": registration[
            "loading_group_state_route_requires_exact_structure_anchor_competition"
        ],
        "loading_group_competing_parent_uses_frozen_v427_observability_rule": registration[
            "loading_group_competing_parent_uses_frozen_v427_observability_rule"
        ],
        "loading_group_axis_route_bypasses_competing_parent_gate": registration[
            "loading_group_axis_route_bypasses_competing_parent_gate"
        ],
        "loading_group_competing_parent_downgrade_means_absence": registration[
            "loading_group_competing_parent_downgrade_means_absence"
        ],
        "loading_group_competing_parent_guard_changes_state_v427_membership": registration[
            "loading_group_competing_parent_guard_changes_state_v427_membership"
        ],
        "loading_group_competing_parent_guard_changes_direct_candidates": registration[
            "loading_group_competing_parent_guard_changes_direct_candidates"
        ],
        "loading_group_structure_equivalence_source_sha256": registration[
            "loading_group_structure_equivalence_source_sha256"
        ],
        "loading_group_structure_equivalence_schema_version": registration[
            "loading_group_structure_equivalence_schema_version"
        ],
        "loading_group_structure_equivalence_fixed_rule": registration[
            "loading_group_structure_equivalence_fixed_rule"
        ],
        "loading_group_state_route_candidate_pair_count": registration[
            "loading_group_state_route_candidate_pair_count"
        ],
        "loading_group_state_route_exact_pair_identified": registration[
            "loading_group_state_route_exact_pair_identified"
        ],
        "loading_group_candidate_pairs_may_be_expanded_as_edges": registration[
            "loading_group_candidate_pairs_may_be_expanded_as_edges"
        ],
        "loading_group_axis_route_candidate_pair_count": registration[
            "loading_group_axis_route_candidate_pair_count"
        ],
        "loading_group_structure_equivalence_changes_v431_selection": registration[
            "loading_group_structure_equivalence_changes_v431_selection"
        ],
        "loading_group_structure_equivalence_changes_state_v427_membership": registration[
            "loading_group_structure_equivalence_changes_state_v427_membership"
        ],
        "loading_group_structure_equivalence_changes_direct_candidates": registration[
            "loading_group_structure_equivalence_changes_direct_candidates"
        ],
        "loading_group_structure_equivalence_application_requires_truth": registration[
            "loading_group_structure_equivalence_application_requires_truth"
        ],
        "loading_group_structure_equivalence_creates_cross_assay_relations": registration[
            "loading_group_structure_equivalence_creates_cross_assay_relations"
        ],
        "subspace_algorithm_source_sha256": registration[
            "algorithm_source_sha256"
        ],
        "subspace_algorithm_config_sha256": registration[
            "algorithm_config_sha256"
        ],
        "subspace_embedding_method": registration["subspace_embedding_method"],
        "subspace_embedding_config_sha256": registration[
            "subspace_embedding_config_sha256"
        ],
        "subspace_embedding_source_sha256": registration[
            "subspace_embedding_source_sha256"
        ],
        "local_candidate_name": registration["local_candidate_name"],
        "local_config_sha256": registration["local_config_sha256"],
        "joint_executable_source_sha256": dict(
            registration["joint_executable_source_sha256"]
        ),
    }


def _add_layered_global_bindings(
    base_manifest: Mapping[str, Any],
    *,
    attempts: Sequence[Mapping[str, Any]],
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every layered candidate into the one formal all-world freeze."""

    manifest = dict(base_manifest)
    manifest.pop("manifest_payload_sha256", None)
    identity = _layered_candidate_identity(registration)
    manifest.update(
        {
            key: value
            for key, value in identity.items()
            if key != "joint_executable_source_sha256"
        }
    )
    manifest["layered_candidate_identity_sha256"] = (
        subspace_scorer.canonical_sha256(identity)
    )
    manifest["layered_extension_schema_version"] = (
        LAYERED_GLOBAL_EXTENSION_SCHEMA_VERSION
    )
    attempts_by_seed = {
        int(attempt["world_seed"]): attempt for attempt in attempts
    }
    records: list[dict[str, Any]] = []
    for raw_record in manifest["world_records"]:
        record = dict(raw_record)
        attempt = attempts_by_seed[int(record["world_seed"])]
        if str(record["status"]).casefold() == "ok":
            for field in (
                "layered_candidate_manifest_sha256",
                "layered_output_sha256",
                "layered_algorithm_source_sha256",
                "layered_algorithm_config_sha256",
            ):
                record[field] = attempt[field]
        records.append(record)
    manifest["world_records"] = records
    manifest["all_layered_candidates_frozen_before_any_scorer_read"] = True
    manifest["all_layered_tables_must_load_before_first_scorer_read"] = True
    manifest["manifest_payload_sha256"] = subspace_scorer.canonical_sha256(
        manifest
    )
    return manifest


def _validate_layered_global_pretruth_freeze(
    manifest: Mapping[str, Any],
    *,
    split_name: str,
) -> dict[str, Any]:
    """Validate the base v9 freeze plus formal layered bindings."""

    base_audit = subspace_scorer.validate_global_pretruth_freeze(
        manifest, split=_normalise_split(split_name)
    )
    source_map = dict(base_audit["joint_executable_source_sha256"])
    current_config = asdict(_frozen_layered_config())
    _, empirical_null_binding = _frozen_empirical_null_calibrator(
        str(manifest.get("empirical_null_calibrator_path", ""))
    )
    _, multievidence_binding = _frozen_multievidence_calibrator(
        str(manifest.get("multievidence_calibrator_path", ""))
    )
    _, topology_consensus_binding = _frozen_topology_consensus_calibrator(
        str(manifest.get("topology_consensus_calibrator_path", ""))
    )
    expected = {
        "layered_algorithm_source_sha256": source_map[
            "layered_state_evidence_v4.py"
        ],
        "layered_algorithm_config_sha256": subspace_scorer.canonical_sha256(
            current_config
        ),
        "layered_scorer_source_sha256": source_map[
            "score_layered_state_evidence_v4.py"
        ],
        "layered_validation_protocol_payload_sha256": (
            layered_scorer.layered_validation_protocol()[
                "protocol_payload_sha256"
            ]
        ),
        "empirical_null_calibrator_artifact_sha256": (
            empirical_null_binding["artifact_file_sha256"]
        ),
        "empirical_null_calibrator_payload_sha256": (
            empirical_null_binding["artifact_payload_sha256"]
        ),
        "empirical_null_numeric_calibrator_payload_sha256": (
            empirical_null_binding["numeric_calibrator_payload_sha256"]
        ),
        "empirical_null_calibrator_schema_version": (
            layered_evidence.empirical_null_state.SCHEMA
        ),
        "empirical_null_calibrator_path": empirical_null_binding["path"],
        "multievidence_calibrator_artifact_sha256": (
            multievidence_binding["artifact_file_sha256"]
        ),
        "multievidence_calibrator_payload_sha256": (
            multievidence_binding["artifact_payload_sha256"]
        ),
        "multievidence_numeric_calibrator_payload_sha256": (
            multievidence_binding["numeric_calibrator_payload_sha256"]
        ),
        "multievidence_calibrator_schema_version": (
            layered_evidence.multievidence_state.SCHEMA
        ),
        "multievidence_calibrator_path": multievidence_binding["path"],
        "topology_consensus_calibrator_artifact_sha256": (
            topology_consensus_binding["artifact_file_sha256"]
        ),
        "topology_consensus_calibrator_payload_sha256": (
            topology_consensus_binding["artifact_payload_sha256"]
        ),
        "topology_consensus_numeric_calibrator_payload_sha256": (
            topology_consensus_binding["numeric_calibrator_payload_sha256"]
        ),
        "topology_consensus_calibrator_schema_version": (
            layered_evidence.topology_consensus_state.SCHEMA
        ),
        "topology_consensus_calibrator_sentinel_certified": True,
        "topology_consensus_calibrator_path": topology_consensus_binding[
            "path"
        ],
        "loading_group_state_guard_source_sha256": source_map[
            "loading_group_state_guard_v4.py"
        ],
        "loading_group_state_guard_schema_version": (
            layered_evidence.loading_group_state_guard.SCHEMA
        ),
        "loading_group_state_guard_fixed_rule": (
            layered_evidence.loading_group_state_guard.FIXED_RULE
        ),
        "loading_group_state_guard_state_route": (
            layered_evidence.loading_group_state_guard.STATE_ROUTE
        ),
        "loading_group_state_guard_axis_route": (
            layered_evidence.loading_group_state_guard.AXIS_ROUTE
        ),
        "loading_group_state_guard_state_route_requires_all_v427_members": True,
        "loading_group_state_guard_axis_route_requires_state_v427_anchor": True,
        "loading_group_state_guard_axis_route_requires_leave_one_group_out_"
        "cross_assay_consensus": True,
        "loading_group_state_guard_axis_route_changes_state_v427_membership": False,
        "loading_group_direct_structure_alone_is_insufficient": True,
        "loading_group_structure_uncertainty_guard_source_sha256": source_map[
            "loading_group_structure_uncertainty_guard_v4.py"
        ],
        "loading_group_structure_uncertainty_guard_schema_version": (
            layered_evidence.loading_group_structure_guard.SCHEMA
        ),
        "loading_group_structure_uncertainty_guard_fixed_rule": (
            layered_evidence.loading_group_structure_guard.FIXED_RULE
        ),
        "loading_group_minimum_state_route_member_centroid_cosine": (
            layered_evidence.loading_group_structure_guard
            .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        ),
        "loading_group_axis_route_bypasses_uncertainty_floor": True,
        "loading_group_both_routes_subject_to_direct_overlap_downgrade": True,
        "loading_group_direct_overlap_downgrade_means_absence": False,
        "loading_group_structure_guard_changes_state_v427_membership": False,
        "loading_group_structure_guard_changes_direct_candidates": False,
        "loading_group_competing_parent_observability_guard_source_sha256": source_map[
            "loading_group_competing_parent_observability_guard_v4.py"
        ],
        "loading_group_competing_parent_observability_guard_schema_version": (
            layered_evidence.loading_group_competitor_guard.SCHEMA
        ),
        "loading_group_competing_parent_observability_guard_fixed_rule": (
            layered_evidence.loading_group_competitor_guard.FIXED_RULE
        ),
        "loading_group_state_route_requires_exact_structure_anchor_competition": True,
        "loading_group_competing_parent_uses_frozen_v427_observability_rule": True,
        "loading_group_axis_route_bypasses_competing_parent_gate": True,
        "loading_group_competing_parent_downgrade_means_absence": False,
        "loading_group_competing_parent_guard_changes_state_v427_membership": False,
        "loading_group_competing_parent_guard_changes_direct_candidates": False,
        "loading_group_structure_equivalence_source_sha256": source_map[
            "loading_group_structure_equivalence_v4.py"
        ],
        "loading_group_structure_equivalence_schema_version": (
            layered_evidence.loading_group_structure_equivalence.SCHEMA
        ),
        "loading_group_structure_equivalence_fixed_rule": (
            layered_evidence.loading_group_structure_equivalence.FIXED_RULE
        ),
        "loading_group_state_route_candidate_pair_count": 4,
        "loading_group_state_route_exact_pair_identified": False,
        "loading_group_candidate_pairs_may_be_expanded_as_edges": False,
        "loading_group_axis_route_candidate_pair_count": 1,
        "loading_group_structure_equivalence_changes_v431_selection": False,
        "loading_group_structure_equivalence_changes_state_v427_membership": False,
        "loading_group_structure_equivalence_changes_direct_candidates": False,
        "loading_group_structure_equivalence_application_requires_truth": False,
        "loading_group_structure_equivalence_creates_cross_assay_relations": False,
        "subspace_algorithm_source_sha256": base_audit[
            "algorithm_source_sha256"
        ],
        "subspace_algorithm_config_sha256": base_audit[
            "algorithm_config_sha256"
        ],
        "subspace_embedding_method": base_audit["subspace_embedding_method"],
        "subspace_embedding_config_sha256": base_audit[
            "subspace_embedding_config_sha256"
        ],
        "subspace_embedding_source_sha256": base_audit[
            "subspace_embedding_source_sha256"
        ],
        "local_candidate_name": subspace_scorer.FROZEN_LOCAL_CANDIDATE_NAME,
        "local_config_sha256": subspace_scorer.canonical_sha256(
            subspace_scorer.FROZEN_LOCAL_CONFIG
        ),
        "joint_executable_source_sha256": source_map,
    }
    mismatched = [
        key
        for key, value in expected.items()
        if key != "joint_executable_source_sha256" and manifest.get(key) != value
    ]
    if mismatched:
        raise ValueError(f"Global freeze layered identity mismatch: {mismatched}")
    if manifest.get("layered_extension_schema_version") != (
        LAYERED_GLOBAL_EXTENSION_SCHEMA_VERSION
    ):
        raise ValueError("Global freeze layered extension schema mismatch")
    if manifest.get("layered_candidate_identity_sha256") != (
        subspace_scorer.canonical_sha256(expected)
    ):
        raise ValueError("Global freeze layered candidate identity mismatch")
    if (
        manifest.get("all_layered_candidates_frozen_before_any_scorer_read")
        is not True
        or manifest.get("all_layered_tables_must_load_before_first_scorer_read")
        is not True
    ):
        raise ValueError("Global freeze lacks the layered all-world barrier")
    for seed, record in base_audit["record_map"].items():
        status = str(record["status"]).casefold()
        layered_fields = {
            "layered_candidate_manifest_sha256",
            "layered_output_sha256",
            "layered_algorithm_source_sha256",
            "layered_algorithm_config_sha256",
        }
        if status != "ok":
            if layered_fields.intersection(record):
                raise ValueError("Failed global record binds layered artifacts")
            continue
        if record.get("layered_algorithm_source_sha256") != expected[
            "layered_algorithm_source_sha256"
        ] or record.get("layered_algorithm_config_sha256") != expected[
            "layered_algorithm_config_sha256"
        ]:
            raise ValueError(f"World {seed} layered source/config mismatch")
        manifest_hash = str(record.get("layered_candidate_manifest_sha256", ""))
        output_hashes = record.get("layered_output_sha256")
        if not _SHA256_PATTERN.fullmatch(manifest_hash):
            raise ValueError(f"World {seed} lacks a layered manifest hash")
        if not isinstance(output_hashes, Mapping) or set(output_hashes) != set(
            _LAYERED_DATA_OUTPUT_NAMES
        ) or any(
            not _SHA256_PATTERN.fullmatch(str(value))
            for value in output_hashes.values()
        ):
            raise ValueError(f"World {seed} lacks exact layered output hashes")
    return {
        **base_audit,
        **{
            key: value
            for key, value in expected.items()
            if key != "joint_executable_source_sha256"
        },
        "layered_candidate_identity_sha256": str(
            manifest["layered_candidate_identity_sha256"]
        ),
    }


def freeze_global_pretruth_barrier(
    *,
    split_name: str,
    attempts: Sequence[Mapping[str, Any]],
    output_root: Path,
    registration: Mapping[str, Any],
    development_validation_sha256: str | None,
) -> tuple[dict[str, Any], Path]:
    """Freeze all local and subspace hashes before scorer-domain access."""

    base_manifest = subspace_scorer.make_global_pretruth_freeze(
        split=_normalise_split(split_name),
        discovery_attempts=attempts,
        algorithm_source_sha256=str(registration["algorithm_source_sha256"]),
        algorithm_config_sha256=str(registration["algorithm_config_sha256"]),
        joint_executable_source_sha256=dict(
            registration["joint_executable_source_sha256"]
        ),
        development_validation_manifest_sha256=development_validation_sha256,
    )
    manifest = _add_layered_global_bindings(
        base_manifest,
        attempts=attempts,
        registration=registration,
    )
    path = output_root / GLOBAL_FREEZE_NAME
    if path.exists():
        observed = _read_json(path, label="global pretruth freeze")
        if observed != manifest:
            raise RuntimeError("Existing global freeze differs from Phase-A attempts")
    else:
        _write_new_json(path, manifest)
    _validate_layered_global_pretruth_freeze(
        _read_json(path, label="global pretruth freeze"),
        split_name=_normalise_split(split_name),
    )
    return manifest, path


def _load_frozen_layered_candidate(
    *,
    paths: Mapping[str, Path],
    global_freeze_path: Path,
    split_name: str,
    world_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate and parse every layered table without touching scorer truth."""

    global_manifest_bytes = subspace_scorer._read_regular_bytes_once(
        global_freeze_path, "global manifest"
    )
    global_manifest_preview = subspace_scorer._json_from_bytes(
        global_manifest_bytes, "global pretruth freeze"
    )
    calibrator_path = Path(
        str(global_manifest_preview.get("empirical_null_calibrator_path", ""))
    )
    multievidence_calibrator_path = Path(
        str(global_manifest_preview.get("multievidence_calibrator_path", ""))
    )
    topology_consensus_calibrator_path = Path(
        str(
            global_manifest_preview.get(
                "topology_consensus_calibrator_path", ""
            )
        )
    )
    layered_snapshot_paths = {
        "layered_manifest": paths["layered_candidate_manifest"],
        "layered_feature_evidence": paths["layered_feature_evidence"],
        "layered_hyperedges": paths["typed_layered_hyperedges"],
        "layered_direct": paths["layered_direct_candidates"],
        "layered_provenance": paths["layered_provenance"],
        "membership": paths["feature_membership"],
        "subspace_hyperedges": paths["typed_subspace_hyperedges"],
        "consensus_edges": paths["local_consensus_edges"],
        "loading_groups": paths["loading_groups"],
        "loading_group_membership": paths["loading_group_membership"],
        "structure_anchor_pair_evidence": paths[
            "structure_anchor_pair_evidence"
        ],
        "group_axis_evidence": paths["group_axis_evidence"],
        "axis_candidates": paths["axis_candidates"],
        "subspace_manifest": paths["candidate_manifest"],
        "local_manifest": paths["local_pretruth_manifest"],
        "empirical_null_calibrator": calibrator_path,
        "multievidence_calibrator": multievidence_calibrator_path,
        "topology_consensus_calibrator": topology_consensus_calibrator_path,
    }
    snapshot = {
        "global_manifest": global_manifest_bytes,
        **{
        label: subspace_scorer._read_regular_bytes_once(
            path, label.replace("_", " ")
        )
        for label, path in layered_snapshot_paths.items()
        },
    }
    global_manifest = subspace_scorer._json_from_bytes(
        snapshot["global_manifest"], "global pretruth freeze"
    )
    global_audit = _validate_layered_global_pretruth_freeze(
        global_manifest, split_name=split_name
    )
    seed = int(world_seed)
    record = global_audit["record_map"].get(seed)
    if record is None or str(record.get("status", "")).casefold() != "ok":
        raise ValueError("World is not a successful frozen layered attempt")
    if subspace_scorer._bytes_sha256(snapshot["subspace_manifest"]) != record[
        "candidate_manifest_sha256"
    ] or subspace_scorer._bytes_sha256(snapshot["local_manifest"]) != record[
        "local_pretruth_manifest_sha256"
    ] or subspace_scorer._bytes_sha256(snapshot["consensus_edges"]) != record[
        "local_consensus_edges_sha256"
    ]:
        raise ValueError("Layered upstream inputs differ from the global freeze")
    layered_root = paths["layered_out"]
    if layered_root.is_symlink() or not layered_root.is_dir():
        raise ValueError("Layered candidate root must be a regular directory")
    layered_entries = list(layered_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in layered_entries):
        raise ValueError("Layered candidate output set contains an unsafe entry")
    actual_files = {path.name for path in layered_entries}
    if actual_files != _LAYERED_OUTPUT_NAMES:
        raise ValueError("Layered candidate output set is missing, extra, or unsafe")
    manifest_path = paths["layered_candidate_manifest"]
    manifest_file_hash = subspace_scorer._bytes_sha256(
        snapshot["layered_manifest"]
    )
    if manifest_file_hash != record["layered_candidate_manifest_sha256"]:
        raise ValueError("Layered candidate manifest differs from the global freeze")
    manifest = subspace_scorer._json_from_bytes(
        snapshot["layered_manifest"], "layered candidate manifest"
    )
    if (
        manifest.get("schema_version")
        != LAYERED_CANDIDATE_SCHEMA_VERSION
        or manifest.get("candidate_frozen") is not True
        or manifest.get("truth_or_scorer_consumed_by_algorithm") is not False
    ):
        raise ValueError("Layered candidate is unsupported or not truth-blind")
    if manifest.get("algorithm_source_sha256") != global_audit[
        "layered_algorithm_source_sha256"
    ] or manifest.get("config_sha256") != global_audit[
        "layered_algorithm_config_sha256"
    ] or manifest.get("config") != asdict(_frozen_layered_config()):
        raise ValueError("Layered candidate source/config differs from the global freeze")
    expected_layered_dependencies = {
        name: global_audit["joint_executable_source_sha256"][name]
        for name in (
            "empirical_null_state_calibration_v4.py",
            "multievidence_empirical_null_state_calibration_v4.py",
            "topology_consensus_state_calibration_v4.py",
            "loading_group_state_guard_v4.py",
            "loading_group_structure_uncertainty_guard_v4.py",
            "loading_group_competing_parent_observability_guard_v4.py",
            "loading_group_structure_equivalence_v4.py",
            "shared_state_subspace_v4.py",
            "state_equivalence_membership_v4.py",
        )
    }
    if manifest.get("dependency_source_sha256") != expected_layered_dependencies:
        raise ValueError(
            "Layered candidate dependency sources differ from the global freeze"
        )

    input_paths = {
        "membership": paths["feature_membership"],
        "subspace_hyperedges": paths["typed_subspace_hyperedges"],
        "consensus_edges": paths["local_consensus_edges"],
        "loading_groups": paths["loading_groups"],
        "loading_group_membership": paths["loading_group_membership"],
        "structure_anchor_pair_evidence": paths[
            "structure_anchor_pair_evidence"
        ],
        "group_axis_evidence": paths["group_axis_evidence"],
        "axis_candidates": paths["axis_candidates"],
        "empirical_null_calibrator": calibrator_path,
        "multievidence_calibrator": multievidence_calibrator_path,
        "topology_consensus_calibrator": topology_consensus_calibrator_path,
    }
    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != set(
        input_paths
    ):
        raise ValueError("Layered candidate does not bind its exact observable inputs")
    for label in input_paths:
        if subspace_scorer._bytes_sha256(snapshot[label]) != input_hashes[label]:
            raise ValueError(f"Layered candidate observable input changed: {label}")
    calibrator_binding = manifest.get("empirical_null_calibrator")
    if not isinstance(calibrator_binding, Mapping):
        raise ValueError("Layered candidate lacks its empirical-null calibrator")
    if (
        calibrator_binding.get("active") is not True
        or calibrator_binding.get("artifact_file_sha256")
        != global_audit["empirical_null_calibrator_artifact_sha256"]
        or calibrator_binding.get("artifact_payload_sha256")
        != global_audit["empirical_null_calibrator_payload_sha256"]
        or calibrator_binding.get("numeric_calibrator_payload_sha256")
        != global_audit["empirical_null_numeric_calibrator_payload_sha256"]
        or calibrator_binding.get("application_requires_truth") is not False
    ):
        raise ValueError("Layered candidate binds another empirical-null calibrator")
    multievidence_binding = manifest.get("multievidence_calibrator")
    if not isinstance(multievidence_binding, Mapping):
        raise ValueError("Layered candidate lacks its multievidence calibrator")
    if (
        multievidence_binding.get("active") is not True
        or multievidence_binding.get("artifact_file_sha256")
        != global_audit["multievidence_calibrator_artifact_sha256"]
        or multievidence_binding.get("artifact_payload_sha256")
        != global_audit["multievidence_calibrator_payload_sha256"]
        or multievidence_binding.get("numeric_calibrator_payload_sha256")
        != global_audit["multievidence_numeric_calibrator_payload_sha256"]
        or multievidence_binding.get("application_requires_truth") is not False
    ):
        raise ValueError("Layered candidate binds another multievidence calibrator")
    topology_consensus_binding = manifest.get(
        "topology_consensus_calibrator"
    )
    if not isinstance(topology_consensus_binding, Mapping):
        raise ValueError(
            "Layered candidate lacks its topology-consensus calibrator"
        )
    if (
        topology_consensus_binding.get("active") is not True
        or topology_consensus_binding.get("sentinel_certified") is not True
        or topology_consensus_binding.get("artifact_file_sha256")
        != global_audit["topology_consensus_calibrator_artifact_sha256"]
        or topology_consensus_binding.get("artifact_payload_sha256")
        != global_audit["topology_consensus_calibrator_payload_sha256"]
        or topology_consensus_binding.get(
            "numeric_calibrator_payload_sha256"
        )
        != global_audit[
            "topology_consensus_numeric_calibrator_payload_sha256"
        ]
        or topology_consensus_binding.get("application_requires_truth")
        is not False
    ):
        raise ValueError(
            "Layered candidate binds another topology-consensus calibrator"
        )

    # Bind the standalone layered inputs back to the earlier world-level
    # pretruth freeze, rather than trusting a later self-consistent manifest.
    local_manifest = subspace_scorer._json_from_bytes(
        snapshot["local_manifest"], "world pretruth manifest"
    )
    candidate_name = str(global_audit["local_candidate_name"])
    all_group_hashes = local_manifest.get("loading_group_projection_file_sha256")
    if not isinstance(all_group_hashes, Mapping):
        raise ValueError("World pretruth manifest lacks loading-group file hashes")
    candidate_group_hashes = all_group_hashes.get(candidate_name)
    if not isinstance(candidate_group_hashes, Mapping):
        raise ValueError("World pretruth manifest lacks the selected loading group")
    all_axis_hashes = local_manifest.get("latent_axis_projection_file_sha256")
    if not isinstance(all_axis_hashes, Mapping):
        raise ValueError("World pretruth manifest lacks latent-axis file hashes")
    candidate_axis_hashes = all_axis_hashes.get(candidate_name)
    if not isinstance(candidate_axis_hashes, Mapping):
        raise ValueError("World pretruth manifest lacks the selected latent axis")
    all_structure_hashes = local_manifest.get(
        "structure_anchored_loading_file_sha256"
    )
    if not isinstance(all_structure_hashes, Mapping):
        raise ValueError("World pretruth manifest lacks structure-anchor file hashes")
    candidate_structure_hashes = all_structure_hashes.get(candidate_name)
    if not isinstance(candidate_structure_hashes, Mapping):
        raise ValueError("World pretruth manifest lacks the selected structure anchor")
    selected_projection_paths = {
        "loading_groups.tsv": paths["loading_groups"],
        "loading_group_membership.tsv": paths["loading_group_membership"],
        "group_axis_evidence.tsv": paths["group_axis_evidence"],
        "axis_candidates.tsv": paths["axis_candidates"],
        "structure_anchor_pair_evidence.tsv": paths[
            "structure_anchor_pair_evidence"
        ],
    }
    for filename, path in selected_projection_paths.items():
        snapshot_label = {
            "loading_groups.tsv": "loading_groups",
            "loading_group_membership.tsv": "loading_group_membership",
            "group_axis_evidence.tsv": "group_axis_evidence",
            "axis_candidates.tsv": "axis_candidates",
            "structure_anchor_pair_evidence.tsv": (
                "structure_anchor_pair_evidence"
            ),
        }[filename]
        projection_hashes = (
            candidate_group_hashes
            if filename in {"loading_groups.tsv", "loading_group_membership.tsv"}
            else (
                candidate_structure_hashes
                if filename == "structure_anchor_pair_evidence.tsv"
                else candidate_axis_hashes
            )
        )
        if projection_hashes.get(filename) != subspace_scorer._bytes_sha256(
            snapshot[snapshot_label]
        ):
            raise ValueError(
                f"Layered projection input differs from world freeze: {filename}"
            )

    subspace_manifest = subspace_scorer._json_from_bytes(
        snapshot["subspace_manifest"], "subspace candidate manifest"
    )
    subspace_outputs = subspace_manifest.get("output_sha256")
    if not isinstance(subspace_outputs, Mapping) or (
        subspace_outputs.get("subspace_feature_membership.tsv")
        != subspace_scorer._bytes_sha256(snapshot["membership"])
        or subspace_outputs.get("typed_subspace_hyperedges.tsv")
        != subspace_scorer._bytes_sha256(snapshot["subspace_hyperedges"])
    ):
        raise ValueError("Layered subspace inputs differ from the upstream freeze")
    local_file_hashes = local_manifest.get("consensus_file_sha256")
    if not isinstance(local_file_hashes, Mapping) or local_file_hashes.get(
        candidate_name
    ) != subspace_scorer._bytes_sha256(snapshot["consensus_edges"]):
        raise ValueError("Layered direct input differs from the upstream local freeze")

    expected_outputs = {
        "layered_feature_evidence.tsv": subspace_scorer._bytes_sha256(
            snapshot["layered_feature_evidence"]
        ),
        "layered_provenance.json": subspace_scorer._bytes_sha256(
            snapshot["layered_provenance"]
        ),
        "typed_layered_hyperedges.tsv": subspace_scorer._bytes_sha256(
            snapshot["layered_hyperedges"]
        ),
        "within_assay_direct_candidates.tsv": subspace_scorer._bytes_sha256(
            snapshot["layered_direct"]
        ),
    }
    output_hashes = manifest.get("output_sha256")
    if (
        not isinstance(output_hashes, Mapping)
        or dict(output_hashes) != expected_outputs
        or dict(record["layered_output_sha256"]) != expected_outputs
    ):
        raise ValueError("Frozen layered candidate output hashes changed")
    provenance = subspace_scorer._json_from_bytes(
        snapshot["layered_provenance"], "layered provenance"
    )
    if (
        provenance.get("schema_version") != LAYERED_PROVENANCE_SCHEMA_VERSION
        or provenance.get("config") != asdict(_frozen_layered_config())
    ):
        raise ValueError("Layered provenance/config is invalid")
    semantics = provenance.get("semantics")
    boundary = provenance.get("information_boundary")
    expected_semantics = {
        "layer_union_is_state_membership": False,
        "assay_participation_implies_feature_membership": False,
        "direct_layer_implies_state_membership": False,
        "loading_group_implies_cross_assay_identity": False,
        "state_route_loading_group_members_are_subset_of_state_v427_members": True,
        "axis_route_loading_group_may_contain_one_unresolved_state_member": True,
        "axis_route_changes_state_v427_membership": False,
        "axis_route_implies_exact_cross_assay_feature_pair": False,
        "loading_group_direct_structure_alone_is_insufficient": True,
        "loading_group_complete_direct_collider_overlap_is_unresolved": True,
        "loading_group_direct_overlap_downgrade_means_absence": False,
        "loading_group_uncertainty_floor_changes_feature_state_membership": False,
        "loading_group_structure_guard_changes_direct_candidates": False,
        "loading_group_competing_parent_downgrade_means_absence": False,
        "loading_group_axis_route_bypasses_competing_parent_gate": True,
        "loading_group_guard_changes_state_v427_membership": False,
        "state_route_loading_group_exact_pair_identified": False,
        "state_route_loading_group_candidate_pairs_are_joint_alternatives": True,
        "state_route_loading_group_candidate_pairs_may_be_expanded_as_edges": False,
        "axis_route_loading_group_remains_exact_assay_local_pair": True,
        "cross_assay_pair_claimed": False,
        "direct_cross_assay_effect_claimed": False,
        "exact_latent_axis_claimed": False,
        "state_equivalence_is_pairwise_structure": False,
        "state_equivalence_implies_exact_cross_assay_pair": False,
        "state_equivalence_implies_direct_causation": False,
        "strict_membership_changed_by_equivalence_tier": False,
        "empirical_null_membership_is_pairwise_structure": False,
        "empirical_null_membership_implies_direct_causation": False,
        "empirical_null_membership_implies_exact_cross_assay_pair": False,
        "existing_state_membership_changed_by_empirical_null_tier": False,
        "multievidence_membership_is_pairwise_structure": False,
        "multievidence_membership_implies_direct_causation": False,
        "multievidence_membership_implies_exact_cross_assay_pair": False,
        "existing_v425_membership_changed_by_multievidence_tier": False,
        "topology_consensus_membership_is_pairwise_structure": False,
        "topology_consensus_membership_implies_direct_causation": False,
        "topology_consensus_membership_implies_exact_cross_assay_pair": False,
        "direct_structure_alone_implies_state_membership": False,
        "existing_v426_membership_changed_by_topology_consensus_tier": False,
    }
    if not isinstance(semantics, Mapping) or dict(semantics) != expected_semantics:
        raise ValueError("Layered provenance makes an unsupported layer claim")
    expected_boundary = {
        "consumed": [
            "frozen_observable_subspace_membership",
            "frozen_observable_subspace_hyperedge",
            "frozen_observable_typed_consensus_graph",
            "frozen_observable_assay_local_loading_groups",
            "frozen_observable_redundancy_demotion_annotations",
            "optional_frozen_numeric_empirical_null_state_calibrator",
            "optional_frozen_numeric_multievidence_state_calibrator",
            "optional_frozen_topology_consensus_state_calibrator",
            "frozen_v427_state_membership_for_loading_group_guard",
            "optional_frozen_leave_one_group_out_group_axis_evidence",
            "optional_frozen_selected_cross_assay_axis_candidates",
            "frozen_selected_structure_anchor_pair_evidence",
        ],
        "not_consumed": [
            "true_latent_state",
            "simulator_roles",
            "truth_graph",
            "scorer",
            "paired_cells",
            "external_data",
            "row_level_calibrator_training_labels_at_application_time",
        ],
    }
    if not isinstance(boundary, Mapping) or dict(boundary) != expected_boundary:
        raise ValueError("Layered provenance does not exclude scorer information")
    provenance_calibrator = provenance.get("empirical_null_calibrator")
    if (
        not isinstance(provenance_calibrator, Mapping)
        or provenance_calibrator.get("active") is not True
        or provenance_calibrator.get("artifact_file_sha256")
        != global_audit["empirical_null_calibrator_artifact_sha256"]
        or provenance_calibrator.get("artifact_payload_sha256")
        != global_audit["empirical_null_calibrator_payload_sha256"]
        or provenance_calibrator.get("application_requires_truth") is not False
    ):
        raise ValueError("Layered provenance lacks the frozen empirical-null binding")
    provenance_multievidence = provenance.get("multievidence_calibrator")
    if (
        not isinstance(provenance_multievidence, Mapping)
        or provenance_multievidence.get("active") is not True
        or provenance_multievidence.get("artifact_file_sha256")
        != global_audit["multievidence_calibrator_artifact_sha256"]
        or provenance_multievidence.get("artifact_payload_sha256")
        != global_audit["multievidence_calibrator_payload_sha256"]
        or provenance_multievidence.get("application_requires_truth") is not False
    ):
        raise ValueError(
            "Layered provenance lacks the frozen multievidence binding"
        )
    provenance_topology_consensus = provenance.get(
        "topology_consensus_calibrator"
    )
    if (
        not isinstance(provenance_topology_consensus, Mapping)
        or provenance_topology_consensus.get("active") is not True
        or provenance_topology_consensus.get("sentinel_certified") is not True
        or provenance_topology_consensus.get("artifact_file_sha256")
        != global_audit["topology_consensus_calibrator_artifact_sha256"]
        or provenance_topology_consensus.get("artifact_payload_sha256")
        != global_audit["topology_consensus_calibrator_payload_sha256"]
        or provenance_topology_consensus.get("application_requires_truth")
        is not False
    ):
        raise ValueError(
            "Layered provenance lacks the frozen topology-consensus binding"
        )
    provenance_loading_guard = provenance.get("loading_group_state_guard")
    if (
        not isinstance(provenance_loading_guard, Mapping)
        or provenance_loading_guard.get("active") is not True
        or provenance_loading_guard.get("schema_version")
        != global_audit["loading_group_state_guard_schema_version"]
        or provenance_loading_guard.get("fixed_rule")
        != global_audit["loading_group_state_guard_fixed_rule"]
        or provenance_loading_guard.get("state_route")
        != global_audit["loading_group_state_guard_state_route"]
        or provenance_loading_guard.get("cross_assay_axis_route")
        != global_audit["loading_group_state_guard_axis_route"]
        or provenance_loading_guard.get("cross_assay_axis_evidence_active")
        is not True
        or provenance_loading_guard.get(
            "leave_one_group_out_axis_evidence_required_for_axis_route"
        )
        is not True
        or provenance_loading_guard.get("upstream_state_membership_schema")
        != layered_evidence.topology_consensus_state.SCHEMA
        or provenance_loading_guard.get(
            "direct_structure_alone_is_insufficient"
        )
        is not True
        or provenance_loading_guard.get("application_requires_truth") is not False
        or provenance_loading_guard.get(
            "creates_cross_assay_or_direct_relations"
        )
        is not False
    ):
        raise ValueError(
            "Layered provenance lacks the frozen loading-group state guard"
        )
    provenance_structure_guard = provenance.get(
        "loading_group_structure_uncertainty_guard"
    )
    if (
        not isinstance(provenance_structure_guard, Mapping)
        or provenance_structure_guard.get("active") is not True
        or provenance_structure_guard.get("schema_version")
        != global_audit[
            "loading_group_structure_uncertainty_guard_schema_version"
        ]
        or provenance_structure_guard.get("fixed_rule")
        != global_audit[
            "loading_group_structure_uncertainty_guard_fixed_rule"
        ]
        or provenance_structure_guard.get(
            "minimum_state_route_member_centroid_cosine"
        )
        != global_audit[
            "loading_group_minimum_state_route_member_centroid_cosine"
        ]
        or provenance_structure_guard.get(
            "axis_route_bypasses_uncertainty_floor"
        )
        is not True
        or provenance_structure_guard.get(
            "both_routes_subject_to_direct_overlap_downgrade"
        )
        is not True
        or provenance_structure_guard.get(
            "direct_overlap_downgrade_means_absence"
        )
        is not False
        or provenance_structure_guard.get("changes_state_v427_membership")
        is not False
        or provenance_structure_guard.get("changes_direct_candidates")
        is not False
        or provenance_structure_guard.get("application_requires_truth")
        is not False
        or provenance_structure_guard.get(
            "creates_cross_assay_or_direct_relations"
        )
        is not False
    ):
        raise ValueError(
            "Layered provenance lacks the frozen loading-group "
            "structure/uncertainty guard"
        )
    provenance_competitor_guard = provenance.get(
        "loading_group_competing_parent_observability_guard"
    )
    if (
        not isinstance(provenance_competitor_guard, Mapping)
        or provenance_competitor_guard.get("active") is not True
        or provenance_competitor_guard.get("schema_version")
        != global_audit[
            "loading_group_competing_parent_observability_guard_schema_version"
        ]
        or provenance_competitor_guard.get("fixed_rule")
        != global_audit[
            "loading_group_competing_parent_observability_guard_fixed_rule"
        ]
        or provenance_competitor_guard.get(
            "upstream_loading_group_guard_schema"
        )
        != layered_evidence.loading_group_structure_guard.SCHEMA
        or provenance_competitor_guard.get(
            "state_route_requires_complete_two_parent_comparison"
        )
        is not True
        or provenance_competitor_guard.get(
            "competing_parent_observability_definition"
        )
        != "frozen_v427_topology_consensus_fixed_rule_eligible"
        or provenance_competitor_guard.get(
            "axis_route_bypasses_competitor_gate"
        )
        is not True
        or provenance_competitor_guard.get("downgrade_means_absence")
        is not False
        or provenance_competitor_guard.get("changes_state_v427_membership")
        is not False
        or provenance_competitor_guard.get("changes_direct_candidates")
        is not False
        or provenance_competitor_guard.get("application_requires_truth")
        is not False
        or provenance_competitor_guard.get(
            "creates_cross_assay_or_direct_relations"
        )
        is not False
    ):
        raise ValueError(
            "Layered provenance lacks the frozen loading-group "
            "competing-parent observability guard"
        )
    provenance_structure_equivalence = provenance.get(
        "loading_group_structure_equivalence"
    )
    if (
        not isinstance(provenance_structure_equivalence, Mapping)
        or provenance_structure_equivalence.get("active") is not True
        or provenance_structure_equivalence.get("schema_version")
        != global_audit[
            "loading_group_structure_equivalence_schema_version"
        ]
        or provenance_structure_equivalence.get("fixed_rule")
        != global_audit["loading_group_structure_equivalence_fixed_rule"]
        or provenance_structure_equivalence.get(
            "upstream_loading_group_guard_schema"
        )
        != layered_evidence.loading_group_competitor_guard.SCHEMA
        or provenance_structure_equivalence.get(
            "state_route_candidate_pair_count"
        )
        != 4
        or provenance_structure_equivalence.get(
            "axis_route_candidate_pair_count"
        )
        != 1
        or provenance_structure_equivalence.get(
            "candidate_pairs_may_be_expanded_as_edges"
        )
        is not False
        or provenance_structure_equivalence.get("changes_v431_selection")
        is not False
        or provenance_structure_equivalence.get(
            "changes_state_v427_membership"
        )
        is not False
        or provenance_structure_equivalence.get("changes_direct_candidates")
        is not False
        or provenance_structure_equivalence.get("application_requires_truth")
        is not False
        or provenance_structure_equivalence.get(
            "creates_cross_assay_or_direct_relations"
        )
        is not False
    ):
        raise ValueError(
            "Layered provenance lacks the frozen v4.32 loading-structure "
            "equivalence contract"
        )

    read_options = {
        "sep": "\t",
        "keep_default_na": False,
        "float_precision": "round_trip",
    }
    feature_evidence = pd.read_csv(
        io.BytesIO(snapshot["layered_feature_evidence"]), **read_options
    )
    typed_hyperedges = pd.read_csv(
        io.BytesIO(snapshot["layered_hyperedges"]), **read_options
    )
    direct_candidates = pd.read_csv(
        io.BytesIO(snapshot["layered_direct"]), **read_options
    )
    return feature_evidence, typed_hyperedges, direct_candidates, {
        "layered_candidate_manifest_file_sha256": manifest_file_hash,
        "layered_output_sha256": expected_outputs,
        "layered_algorithm_source_sha256": global_audit[
            "layered_algorithm_source_sha256"
        ],
        "layered_algorithm_config_sha256": global_audit[
            "layered_algorithm_config_sha256"
        ],
        "layered_candidate_identity_sha256": global_audit[
            "layered_candidate_identity_sha256"
        ],
        "layered_tables_loaded_before_scorer_read": True,
    }


def load_all_candidates_before_scorer(
    *,
    split_name: str,
    attempts: Sequence[Mapping[str, Any]],
    output_root: Path,
    global_freeze_path: Path,
) -> dict[int, LoadedLayeredWorld]:
    """Load every subspace, local, and layered table before truth access."""

    _validate_layered_global_pretruth_freeze(
        _read_json(global_freeze_path, label="global pretruth freeze"),
        split_name=split_name,
    )
    loaded: dict[int, LoadedLayeredWorld] = {}
    for attempt in attempts:
        if str(attempt["status"]).casefold() != "ok":
            continue
        seed = int(attempt["world_seed"])
        paths = _artifact_paths(output_root, seed)
        membership, hyperedges, local_edges, base_audit = (
            subspace_scorer.load_frozen_subspace_candidate(
            feature_membership_path=paths["feature_membership"],
            typed_subspace_hyperedges_path=paths["typed_subspace_hyperedges"],
            candidate_manifest_path=paths["candidate_manifest"],
            subspace_embedding_manifest_path=paths[
                "subspace_embedding_manifest"
            ],
            subspace_embedding_path=paths["subspace_embedding"],
            local_pretruth_manifest_path=paths["local_pretruth_manifest"],
            local_consensus_edges_path=paths["local_consensus_edges"],
            global_pretruth_manifest_path=global_freeze_path,
            split=_normalise_split(split_name),
            world_seed=seed,
            )
        )
        (
            layered_features,
            layered_hyperedges,
            layered_direct,
            layered_audit,
        ) = _load_frozen_layered_candidate(
            paths=paths,
            global_freeze_path=global_freeze_path,
            split_name=split_name,
            world_seed=seed,
        )
        loaded[seed] = LoadedLayeredWorld(
            subspace_membership=membership,
            subspace_hyperedges=hyperedges,
            local_edges=local_edges,
            layered_feature_evidence=layered_features,
            layered_hyperedges=layered_hyperedges,
            layered_direct_candidates=layered_direct,
            audit={**base_audit, **layered_audit},
        )
    expected = {
        int(attempt["world_seed"])
        for attempt in attempts
        if str(attempt["status"]).casefold() == "ok"
    }
    if set(loaded) != expected:
        raise RuntimeError("Not every successful candidate was loaded before scorer access")
    return loaded


def _failure_result_record(
    attempt: Mapping[str, Any], global_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "world_seed": int(attempt["world_seed"]),
        "discovery_status": str(attempt["status"]).casefold(),
        "scoring_status": "not_attempted",
        "execution_status": str(attempt["status"]).casefold(),
        "metrics": None,
        "subspace_diagnostic_metrics": None,
        "algorithm_source_sha256": global_manifest["algorithm_source_sha256"],
        "algorithm_config_sha256": global_manifest["algorithm_config_sha256"],
        "joint_candidate_identity_sha256": global_manifest[
            "joint_candidate_identity_sha256"
        ],
        "layered_candidate_identity_sha256": global_manifest[
            "layered_candidate_identity_sha256"
        ],
        "global_manifest_payload_sha256": global_manifest[
            "manifest_payload_sha256"
        ],
    }


def _write_score_artifacts(
    *,
    output_root: Path,
    seed: int,
    layered_metrics: Mapping[str, Any],
    subspace_diagnostic: subspace_scorer.SharedStateSubspaceScore | None,
    subspace_diagnostic_error: Mapping[str, str] | None,
    audit: Mapping[str, Any],
    scorer_manifest_sha256: str,
) -> dict[str, str]:
    score_root = output_root / "scores" / world_name(seed)
    if score_root.is_symlink() or score_root.exists():
        raise FileExistsError(f"Refusing to overwrite score output: {score_root}")
    score_parent = score_root.parent
    if score_parent.is_symlink() or (
        score_parent.exists() and not score_parent.is_dir()
    ):
        raise ValueError("Score output parent must be a regular directory")
    score_parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {
        "metrics.json": dict(layered_metrics),
        "subspace_diagnostic_status.json": {
            "diagnostic_only": True,
            "status": "ok" if subspace_diagnostic is not None else "error",
            "error": dict(subspace_diagnostic_error or {}),
        },
        "candidate_load_audit.json": dict(audit),
    }
    table_outputs: dict[str, pd.DataFrame] = {}
    if subspace_diagnostic is not None:
        outputs["subspace_diagnostic_metrics.json"] = subspace_diagnostic.metrics
        table_outputs = {
            "subspace_diagnostic_feature_scores.tsv": (
                subspace_diagnostic.feature_scores
            ),
            "subspace_diagnostic_assay_scores.tsv": (
                subspace_diagnostic.assay_scores
            ),
            "subspace_diagnostic_hyperedge_scores.tsv": (
                subspace_diagnostic.hyperedge_scores
            ),
            "subspace_diagnostic_local_pair_scores.tsv": (
                subspace_diagnostic.local_pair_scores
            ),
        }
    # Build a complete sibling directory and publish it with one atomic rename.
    # A serialization or write failure is therefore an ordinary denominator
    # record and cannot leave a partial world directory that a first run treats
    # as complete (or that a later resume mistakes for a valid score bundle).
    with tempfile.TemporaryDirectory(
        dir=score_parent, prefix=f".{world_name(seed)}.tmp-"
    ) as temporary_name:
        staging_root = Path(temporary_name)
        for filename, value in outputs.items():
            _write_new_json(staging_root / filename, value)
        for filename, frame in table_outputs.items():
            frame.to_csv(staging_root / filename, sep="\t", index=False)
        hashes = {
            path.name: subspace_scorer.file_sha256(path)
            for path in sorted(staging_root.iterdir())
            if path.is_file()
        }
        manifest = _self_hashed(
            {
                "schema_version": "layered_state_evidence_v4.score_artifacts.v1",
                "world_seed": seed,
                "scorer_manifest_sha256": scorer_manifest_sha256,
                "output_sha256": hashes,
                "primary_score": "layered_state_evidence_v4",
                "full_subspace_score_is_diagnostic_only": True,
            },
            "score_manifest_payload_sha256",
        )
        _write_new_json(staging_root / "score_manifest.json", manifest)
        result = {
            **hashes,
            "score_manifest.json": subspace_scorer.file_sha256(
                staging_root / "score_manifest.json"
            ),
        }
        if score_root.is_symlink() or score_root.exists():
            raise FileExistsError(f"Refusing to overwrite score output: {score_root}")
        staging_root.rename(score_root)
    return result


def score_loaded_candidates(
    *,
    attempts: Sequence[Mapping[str, Any]],
    loaded: Mapping[int, LoadedLayeredWorld],
    worlds_root: Path,
    output_root: Path,
    global_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Open scorer domains only after ``loaded`` contains all candidates."""

    expected_loaded = {
        int(attempt["world_seed"])
        for attempt in attempts
        if str(attempt["status"]).casefold() == "ok"
    }
    if set(loaded) != expected_loaded:
        raise RuntimeError("Scorer access attempted before all candidates were loaded")
    if (output_root / "scores").exists():
        raise FileExistsError(
            "Refusing partial Phase-B resume; use the completed run or a new output root"
        )
    records: list[dict[str, Any]] = []
    for attempt in attempts:
        seed = int(attempt["world_seed"])
        if str(attempt["status"]).casefold() != "ok":
            records.append(_failure_result_record(attempt, global_manifest))
            print(
                f"[Phase B] {world_name(seed)}: not_attempted",
                file=sys.stderr,
                flush=True,
            )
            continue
        _verify_global_sources_are_current(global_manifest)
        loaded_world = loaded[seed]
        try:
            # This is the first scorer-domain access for this run.  The loop is
            # reached only after every candidate/local graph has been parsed.
            truth_edges, parameters, roles, scorer_binding = (
                local_benchmark.load_scorer_artifacts_after_pretruth_freeze(
                    worlds_root / world_name(seed),
                    _artifact_paths(output_root, seed)["world_out"],
                    expected_pretruth_manifest_sha256=str(
                        attempt["local_pretruth_manifest_sha256"]
                    ),
                    return_binding=True,
                )
            )
            scorer_manifest_sha256 = str(
                scorer_binding["scorer_manifest"]["sha256"]
            )
            layered_metrics = layered_scorer.score_layered_state_evidence(
                loaded_world.layered_feature_evidence,
                loaded_world.layered_hyperedges,
                loaded_world.layered_direct_candidates,
                roles,
                truth_edges,
                parameters,
            )
            subspace_diagnostic: subspace_scorer.SharedStateSubspaceScore | None
            subspace_diagnostic_error: dict[str, str] | None = None
            try:
                subspace_diagnostic = subspace_scorer.score_shared_state_subspace(
                    world_parameters=parameters,
                    role_assignments=roles,
                    feature_membership=loaded_world.subspace_membership,
                    typed_subspace_hyperedges=loaded_world.subspace_hyperedges,
                    typed_graph=loaded_world.local_edges,
                    truth_typed_edges=truth_edges,
                )
            except Exception as diagnostic_exc:
                subspace_diagnostic = None
                subspace_diagnostic_error = {
                    "error_type": type(diagnostic_exc).__name__,
                    "error_message": str(diagnostic_exc)[:2000],
                }
            score_artifact_sha256 = _write_score_artifacts(
                output_root=output_root,
                seed=seed,
                layered_metrics=layered_metrics,
                subspace_diagnostic=subspace_diagnostic,
                subspace_diagnostic_error=subspace_diagnostic_error,
                audit=loaded_world.audit,
                scorer_manifest_sha256=scorer_manifest_sha256,
            )
            record: dict[str, Any] = {
                "world_seed": seed,
                "discovery_status": "ok",
                "scoring_status": "ok",
                "execution_status": "ok",
                "metrics": dict(layered_metrics),
                "subspace_diagnostic_status": (
                    "ok" if subspace_diagnostic is not None else "error"
                ),
                "subspace_diagnostic_metrics": (
                    dict(subspace_diagnostic.metrics)
                    if subspace_diagnostic is not None
                    else None
                ),
                "subspace_diagnostic_error": subspace_diagnostic_error,
                "candidate_manifest_sha256": attempt[
                    "candidate_manifest_sha256"
                ],
                "subspace_embedding_manifest_sha256": attempt[
                    "subspace_embedding_manifest_sha256"
                ],
                "subspace_embedding_sha256": attempt[
                    "subspace_embedding_sha256"
                ],
                "local_pretruth_manifest_sha256": attempt[
                    "local_pretruth_manifest_sha256"
                ],
                "local_consensus_edges_sha256": attempt[
                    "local_consensus_edges_sha256"
                ],
                "layered_candidate_manifest_sha256": attempt[
                    "layered_candidate_manifest_sha256"
                ],
                "layered_output_sha256": dict(
                    attempt["layered_output_sha256"]
                ),
                "algorithm_source_sha256": global_manifest[
                    "algorithm_source_sha256"
                ],
                "algorithm_config_sha256": global_manifest[
                    "algorithm_config_sha256"
                ],
                "joint_candidate_identity_sha256": global_manifest[
                    "joint_candidate_identity_sha256"
                ],
                "layered_candidate_identity_sha256": global_manifest[
                    "layered_candidate_identity_sha256"
                ],
                "global_manifest_payload_sha256": global_manifest[
                    "manifest_payload_sha256"
                ],
                "score_artifact_sha256": score_artifact_sha256,
            }
        except Exception as exc:  # scoring failures also stay in the denominator
            record = {
                "world_seed": seed,
                "discovery_status": "ok",
                "scoring_status": "error",
                "execution_status": "error",
                "metrics": None,
                "subspace_diagnostic_metrics": None,
                "candidate_manifest_sha256": attempt[
                    "candidate_manifest_sha256"
                ],
                "subspace_embedding_manifest_sha256": attempt[
                    "subspace_embedding_manifest_sha256"
                ],
                "subspace_embedding_sha256": attempt[
                    "subspace_embedding_sha256"
                ],
                "local_pretruth_manifest_sha256": attempt[
                    "local_pretruth_manifest_sha256"
                ],
                "local_consensus_edges_sha256": attempt[
                    "local_consensus_edges_sha256"
                ],
                "layered_candidate_manifest_sha256": attempt[
                    "layered_candidate_manifest_sha256"
                ],
                "layered_output_sha256": dict(
                    attempt["layered_output_sha256"]
                ),
                "algorithm_source_sha256": global_manifest[
                    "algorithm_source_sha256"
                ],
                "algorithm_config_sha256": global_manifest[
                    "algorithm_config_sha256"
                ],
                "joint_candidate_identity_sha256": global_manifest[
                    "joint_candidate_identity_sha256"
                ],
                "layered_candidate_identity_sha256": global_manifest[
                    "layered_candidate_identity_sha256"
                ],
                "global_manifest_payload_sha256": global_manifest[
                    "manifest_payload_sha256"
                ],
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
            }
        records.append(record)
        print(
            f"[Phase B] {world_name(seed)}: {record['scoring_status']}",
            file=sys.stderr,
            flush=True,
        )
    _verify_global_sources_are_current(global_manifest)
    return records


def _write_result_records(
    *,
    split_name: str,
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    global_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = _self_hashed(
        {
            "schema_version": RESULT_RECORDS_SCHEMA_VERSION,
            "split": _normalise_split(split_name),
            "registered_world_seeds": list(registered_seeds(split_name)),
            "global_manifest_payload_sha256": global_manifest[
                "manifest_payload_sha256"
            ],
            "all_failures_retained_in_denominator": True,
            "records": list(records),
        },
        "result_records_payload_sha256",
    )
    _write_new_json(output_root / RESULT_RECORDS_NAME, value)
    return value


def _validate_result_records_against_global_freeze(
    *,
    split_name: str,
    records: Sequence[Mapping[str, Any]],
    global_manifest: Mapping[str, Any],
) -> None:
    """Bind every Phase-B result to its exact frozen per-world candidate."""

    audit = _validate_layered_global_pretruth_freeze(
        global_manifest, split_name=split_name
    )
    expected_seeds = registered_seeds(split_name)
    by_seed: dict[int, Mapping[str, Any]] = {}
    for result in records:
        seed = int(result["world_seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate Phase-B result for world {seed}")
        by_seed[seed] = result
    if set(by_seed) != set(expected_seeds):
        raise ValueError("Phase-B results do not cover the exact registered block")

    for seed in expected_seeds:
        result = by_seed[seed]
        frozen = audit["record_map"][seed]
        frozen_status = str(frozen["status"]).casefold()
        if str(result.get("discovery_status", "")).casefold() != frozen_status:
            raise ValueError(f"World {seed} result/discovery freeze status mismatch")
        identity_checks = {
            "algorithm_source_sha256": global_manifest[
                "algorithm_source_sha256"
            ],
            "algorithm_config_sha256": global_manifest[
                "algorithm_config_sha256"
            ],
            "global_manifest_payload_sha256": global_manifest[
                "manifest_payload_sha256"
            ],
            "joint_candidate_identity_sha256": global_manifest[
                "joint_candidate_identity_sha256"
            ],
            "layered_candidate_identity_sha256": global_manifest[
                "layered_candidate_identity_sha256"
            ],
        }
        if any(result.get(key) != value for key, value in identity_checks.items()):
            raise ValueError(f"World {seed} result lacks the frozen joint identity")

        execution_status = str(result.get("execution_status", "")).casefold()
        metrics = result.get("metrics")
        if frozen_status != "ok":
            if execution_status != frozen_status or metrics is not None:
                raise ValueError(f"World {seed} failed discovery result is inconsistent")
            forbidden_candidate_fields = {
                "candidate_manifest_sha256",
                "subspace_embedding_manifest_sha256",
                "subspace_embedding_sha256",
                "local_pretruth_manifest_sha256",
                "local_consensus_edges_sha256",
                "layered_candidate_manifest_sha256",
                "layered_output_sha256",
                "score_artifact_sha256",
            }
            if forbidden_candidate_fields.intersection(result):
                raise ValueError(f"World {seed} failed result binds candidate outputs")
            continue

        if execution_status not in {"ok", "error"}:
            raise ValueError(f"World {seed} has an invalid scoring execution status")
        if (execution_status == "ok") != isinstance(metrics, Mapping):
            raise ValueError(f"World {seed} score payload/status is inconsistent")
        score_hashes = result.get("score_artifact_sha256")
        if execution_status == "ok":
            if not isinstance(score_hashes, Mapping) or not score_hashes or (
                "score_manifest.json" not in score_hashes
            ):
                raise ValueError(
                    f"World {seed} successful result lacks its score bundle"
                )
        elif score_hashes is not None:
            raise ValueError(f"World {seed} scoring error binds score artifacts")
        frozen_candidate_fields = {
            "candidate_manifest_sha256": frozen["candidate_manifest_sha256"],
            "subspace_embedding_manifest_sha256": frozen[
                "subspace_embedding_manifest_sha256"
            ],
            "subspace_embedding_sha256": frozen["subspace_embedding_sha256"],
            "local_pretruth_manifest_sha256": frozen[
                "local_pretruth_manifest_sha256"
            ],
            "local_consensus_edges_sha256": frozen[
                "local_consensus_edges_sha256"
            ],
        }
        mismatched_candidate_fields = [
            field
            for field, expected in frozen_candidate_fields.items()
            if result.get(field) != expected
        ]
        if mismatched_candidate_fields:
            raise ValueError(
                f"World {seed} result/base candidate binding mismatch: "
                f"{mismatched_candidate_fields}"
            )
        if result.get("layered_candidate_manifest_sha256") != frozen.get(
            "layered_candidate_manifest_sha256"
        ) or result.get("layered_output_sha256") != frozen.get(
            "layered_output_sha256"
        ):
            raise ValueError(f"World {seed} result is not bound to its layered candidate")


def _formal_layered_block_binding(
    global_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the frozen joint identity into the primary block validator."""

    return {
        "layered_algorithm_source_sha256": global_manifest[
            "layered_algorithm_source_sha256"
        ],
        "layered_algorithm_config_sha256": global_manifest[
            "layered_algorithm_config_sha256"
        ],
        "layered_validation_protocol_payload_sha256": global_manifest[
            "layered_validation_protocol_payload_sha256"
        ],
        "empirical_null_calibrator_artifact_sha256": global_manifest[
            "empirical_null_calibrator_artifact_sha256"
        ],
        "empirical_null_calibrator_payload_sha256": global_manifest[
            "empirical_null_calibrator_payload_sha256"
        ],
        "multievidence_calibrator_artifact_sha256": global_manifest[
            "multievidence_calibrator_artifact_sha256"
        ],
        "multievidence_calibrator_payload_sha256": global_manifest[
            "multievidence_calibrator_payload_sha256"
        ],
        "topology_consensus_calibrator_artifact_sha256": global_manifest[
            "topology_consensus_calibrator_artifact_sha256"
        ],
        "topology_consensus_calibrator_payload_sha256": global_manifest[
            "topology_consensus_calibrator_payload_sha256"
        ],
        "subspace_algorithm_source_sha256": global_manifest[
            "subspace_algorithm_source_sha256"
        ],
        "subspace_algorithm_config_sha256": global_manifest[
            "subspace_algorithm_config_sha256"
        ],
        "subspace_embedding_method": global_manifest[
            "subspace_embedding_method"
        ],
        "subspace_embedding_config_sha256": global_manifest[
            "subspace_embedding_config_sha256"
        ],
        "subspace_embedding_source_sha256": global_manifest[
            "subspace_embedding_source_sha256"
        ],
        "local_candidate_name": global_manifest["local_candidate_name"],
        "local_config_sha256": global_manifest["local_config_sha256"],
        "joint_executable_source_sha256": dict(
            global_manifest["joint_executable_source_sha256"]
        ),
        "joint_candidate_identity_sha256": global_manifest[
            "joint_candidate_identity_sha256"
        ],
        "layered_candidate_identity_sha256": global_manifest[
            "layered_candidate_identity_sha256"
        ],
        "global_pretruth_manifest_payload_sha256": global_manifest[
            "manifest_payload_sha256"
        ],
    }


def _validate_score_artifact_tree(
    output_root: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    """Require the exact score-directory/file set declared by result records."""

    expected_score_directories: set[str] = set()
    for record in records:
        execution_status = str(record.get("execution_status", "")).casefold()
        has_score_hashes = "score_artifact_sha256" in record
        raw_score_hashes = record.get("score_artifact_sha256")
        if execution_status == "ok":
            if (
                not isinstance(raw_score_hashes, Mapping)
                or not raw_score_hashes
                or "score_manifest.json" not in raw_score_hashes
            ):
                raise ValueError("Successful result lacks a complete score bundle")
            score_hashes: Mapping[str, Any] = raw_score_hashes
        else:
            if has_score_hashes:
                raise ValueError("Failed result unexpectedly binds score artifacts")
            score_hashes = {}
        seed = int(record.get("world_seed", -1))
        if score_hashes:
            directory_name = world_name(seed)
            expected_score_directories.add(directory_name)
            score_directory = output_root / "scores" / directory_name
            if score_directory.is_symlink() or not score_directory.is_dir():
                raise ValueError("Completed score directory is missing or unsafe")
            score_entries = list(score_directory.iterdir())
            if any(path.is_symlink() or not path.is_file() for path in score_entries):
                raise ValueError("Completed score directory contains an unsafe entry")
            if {path.name for path in score_entries} != set(map(str, score_hashes)):
                raise ValueError(
                    "Completed score directory file set differs from its result record"
                )
        for filename, expected in score_hashes.items():
            path = output_root / "scores" / world_name(seed) / str(filename)
            if path.is_symlink() or not path.is_file() or (
                subspace_scorer.file_sha256(path) != str(expected)
            ):
                raise ValueError(f"Completed score artifact changed: {path}")

    score_root = output_root / "scores"
    if score_root.is_symlink() or (
        score_root.exists() and not score_root.is_dir()
    ):
        raise ValueError("Completed score root must be a regular directory")
    score_entries = list(score_root.iterdir()) if score_root.is_dir() else []
    if any(path.is_symlink() or not path.is_dir() for path in score_entries):
        raise ValueError("Completed score root contains an unsafe extra entry")
    actual_score_directories = {path.name for path in score_entries}
    if actual_score_directories != expected_score_directories:
        raise ValueError("Completed score directory set is missing, extra, or partial")


def _validate_completed_summary_container(
    summary: Mapping[str, Any], *, split_name: str
) -> None:
    split = _normalise_split(split_name)
    expected_keys = {
        "schema_version",
        "runner_schema_version",
        "split",
        "registered_world_seeds",
        "n_registered_worlds",
        "n_discovery_successes",
        "n_scoring_successes",
        "all_failures_retained_in_denominator",
        "all_candidates_loaded_before_any_scorer_read",
        "executable_source_unchanged_through_scoring",
        "maximum_workers",
        "workers_used",
        "block_certified",
        "artifact_sha256",
        "run_summary_payload_sha256",
    }
    if set(summary) != expected_keys:
        raise ValueError("Completed run summary field set is missing or extra")
    seeds = list(registered_seeds(split))
    if (
        summary.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION
        or summary.get("runner_schema_version") != RUNNER_SCHEMA_VERSION
        or summary.get("split") != split
        or summary.get("registered_world_seeds") != seeds
        or summary.get("n_registered_worlds") != len(seeds)
        or summary.get("maximum_workers") != 2
        or summary.get("all_failures_retained_in_denominator") is not True
        or summary.get("all_candidates_loaded_before_any_scorer_read") is not True
        or summary.get("executable_source_unchanged_through_scoring") is not True
        or summary.get("workers_used") not in {1, 2}
    ):
        raise ValueError("Completed run summary contract is not current")
    artifact_hashes = summary.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != {
        REGISTRATION_NAME,
        PHASE_A_NAME,
        GLOBAL_FREEZE_NAME,
        RESULT_RECORDS_NAME,
        BLOCK_VALIDATION_NAME,
    }:
        raise ValueError("Completed run summary artifact set is missing or extra")


def _validate_completed_result_container(
    results: Mapping[str, Any], *, split_name: str
) -> None:
    split = _normalise_split(split_name)
    expected_keys = {
        "schema_version",
        "split",
        "registered_world_seeds",
        "global_manifest_payload_sha256",
        "all_failures_retained_in_denominator",
        "records",
        "result_records_payload_sha256",
    }
    if set(results) != expected_keys:
        raise ValueError("Completed result-record field set is missing or extra")
    if (
        results.get("schema_version") != RESULT_RECORDS_SCHEMA_VERSION
        or results.get("split") != split
        or results.get("registered_world_seeds")
        != list(registered_seeds(split))
        or results.get("all_failures_retained_in_denominator") is not True
        or not isinstance(results.get("records"), list)
    ):
        raise ValueError("Completed result-record container contract is not current")


def _validate_completed_phase_a_container(
    phase_a: Mapping[str, Any], *, split_name: str, registration_sha256: str
) -> None:
    split = _normalise_split(split_name)
    expected_keys = {
        "schema_version",
        "split",
        "registered_world_seeds",
        "registration_payload_sha256",
        "all_registered_discovery_attempts_finished",
        "all_attempts_completed_before_any_scorer_read",
        "scorer_domain_opened",
        "attempts",
        "phase_a_payload_sha256",
    }
    if set(phase_a) != expected_keys:
        raise ValueError("Completed Phase-A field set is missing or extra")
    if (
        phase_a.get("schema_version") != PHASE_A_SCHEMA_VERSION
        or phase_a.get("split") != split
        or phase_a.get("registered_world_seeds")
        != list(registered_seeds(split))
        or phase_a.get("registration_payload_sha256") != registration_sha256
        or phase_a.get("all_registered_discovery_attempts_finished") is not True
        or phase_a.get("all_attempts_completed_before_any_scorer_read") is not True
        or phase_a.get("scorer_domain_opened") is not False
        or not isinstance(phase_a.get("attempts"), list)
    ):
        raise ValueError("Completed Phase-A container contract is not current")


def _load_completed_run(
    output_root: Path, *, split_name: str
) -> dict[str, Any] | None:
    summary_path = output_root / RUN_SUMMARY_NAME
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path, label="completed run summary")
    _validate_self_hash(
        summary, field="run_summary_payload_sha256", label="completed run summary"
    )
    _validate_completed_summary_container(summary, split_name=split_name)
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("Completed run summary lacks artifact hashes")
    for filename, expected in hashes.items():
        path = output_root / str(filename)
        if path.is_symlink() or not path.is_file() or (
            subspace_scorer.file_sha256(path) != str(expected)
        ):
            raise ValueError(f"Completed runner artifact changed: {filename}")
    validation = _read_json(
        output_root / BLOCK_VALIDATION_NAME, label="block validation"
    )
    if (
        validation.get("schema_version")
        != layered_scorer.BLOCK_VALIDATION_SCHEMA_VERSION
        or validation.get("split") != _normalise_split(split_name)
    ):
        raise ValueError("Completed block-validation container is not current")
    if bool(validation.get("block_certified")) != bool(
        summary.get("block_certified")
    ):
        raise ValueError("Completed run summary/block validation disagree")
    results = _read_json(
        output_root / RESULT_RECORDS_NAME, label="completed result records"
    )
    _validate_self_hash(
        results,
        field="result_records_payload_sha256",
        label="completed result records",
    )
    _validate_completed_result_container(results, split_name=split_name)
    _validate_score_artifact_tree(output_root, results.get("records", []))
    return summary


def run_block(
    *,
    split_name: str,
    worlds_root: Path | str,
    output_root: Path | str,
    generate_missing: bool = False,
    workers: int = 2,
    resume: bool = False,
    development_validation_path: Path | str | None = None,
    empirical_null_calibrator_path: Path | str = DEFAULT_EMPIRICAL_NULL_CALIBRATOR,
    multievidence_calibrator_path: Path | str = DEFAULT_MULTIEVIDENCE_CALIBRATOR,
    topology_consensus_calibrator_path: Path | str = (
        DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR
    ),
) -> dict[str, Any]:
    """Execute or safely resume one exact preregistered validation block."""

    if not 1 <= int(workers) <= 2:
        raise ValueError("workers must be one or two")
    split = _normalise_split(split_name)
    worlds = Path(worlds_root).absolute()
    output = Path(output_root).absolute()
    empirical_null_calibrator = Path(empirical_null_calibrator_path).absolute()
    multievidence_calibrator = Path(multievidence_calibrator_path).absolute()
    topology_consensus_calibrator = Path(
        topology_consensus_calibrator_path
    ).absolute()
    worlds.mkdir(parents=True, exist_ok=True)
    development_path, development_hash = _development_binding(
        split, development_validation_path
    )
    registration = _load_or_create_registration(
        split_name=split,
        worlds_root=worlds,
        output_root=output,
        generate_missing=generate_missing,
        development_validation_sha256=development_hash,
        empirical_null_calibrator_path=empirical_null_calibrator,
        multievidence_calibrator_path=multievidence_calibrator,
        topology_consensus_calibrator_path=topology_consensus_calibrator,
        resume=resume,
    )
    if development_path is not None:
        _validate_development_before_locked(development_path, registration)
    # Registration is on disk before this check and before any simulator call.
    registered_world_paths(split, worlds)
    if resume:
        complete = _load_completed_run(output, split_name=split)
        if complete is not None:
            phase_a = _read_json(output / PHASE_A_NAME, label="Phase-A manifest")
            _validate_self_hash(
                phase_a,
                field="phase_a_payload_sha256",
                label="Phase-A manifest",
            )
            _validate_completed_phase_a_container(
                phase_a,
                split_name=split,
                registration_sha256=str(
                    registration["registration_payload_sha256"]
                ),
            )
            attempts = [
                _validate_attempt_record(
                    value,
                    seed=int(value["world_seed"]),
                    registration_sha256=str(
                        registration["registration_payload_sha256"]
                    ),
                    output_root=output,
                )
                for value in phase_a.get("attempts", [])
            ]
            if tuple(int(value["world_seed"]) for value in attempts) != registered_seeds(
                split
            ):
                raise ValueError("Completed Phase-A attempt block is not exact")
            global_path = output / GLOBAL_FREEZE_NAME
            global_manifest = _read_json(
                global_path, label="global pretruth freeze"
            )
            _validate_layered_global_pretruth_freeze(
                global_manifest, split_name=split
            )
            completed_results = _read_json(
                output / RESULT_RECORDS_NAME, label="completed result records"
            )
            _validate_result_records_against_global_freeze(
                split_name=split,
                records=completed_results.get("records", []),
                global_manifest=global_manifest,
            )
            load_all_candidates_before_scorer(
                split_name=split,
                attempts=attempts,
                output_root=output,
                global_freeze_path=global_path,
            )
            return complete
    attempts = run_discovery_phase(
        split_name=split,
        worlds_root=worlds,
        output_root=output,
        registration=registration,
        generate_missing=generate_missing,
        workers=workers,
    )
    _verify_registration_is_current(registration)
    global_manifest, global_path = freeze_global_pretruth_barrier(
        split_name=split,
        attempts=attempts,
        output_root=output,
        registration=registration,
        development_validation_sha256=development_hash,
    )
    # This complete in-memory load is the global truth barrier.
    loaded = load_all_candidates_before_scorer(
        split_name=split,
        attempts=attempts,
        output_root=output,
        global_freeze_path=global_path,
    )
    _verify_registration_is_current(registration)
    records = score_loaded_candidates(
        attempts=attempts,
        loaded=loaded,
        worlds_root=worlds,
        output_root=output,
        global_manifest=global_manifest,
    )
    _validate_result_records_against_global_freeze(
        split_name=split,
        records=records,
        global_manifest=global_manifest,
    )
    # Validate the just-produced tree on the first run as well as on resume.
    # In particular, a caught score-write failure must not coexist with a
    # partially published world directory or a non-directory ``scores`` path.
    _validate_score_artifact_tree(output, records)
    _write_result_records(
        split_name=split,
        output_root=output,
        records=records,
        global_manifest=global_manifest,
    )
    block_attempts = [
        {
            "world_seed": int(record["world_seed"]),
            "execution_status": str(record["execution_status"]),
            "metrics": record.get("metrics"),
        }
        for record in records
    ]
    block_validation = layered_scorer.validate_layered_evidence_block(
        split,
        block_attempts,
        formal_binding=_formal_layered_block_binding(global_manifest),
        require_formal_binding=True,
    )
    _write_new_json(output / BLOCK_VALIDATION_NAME, block_validation)
    core_artifacts = (
        REGISTRATION_NAME,
        PHASE_A_NAME,
        GLOBAL_FREEZE_NAME,
        RESULT_RECORDS_NAME,
        BLOCK_VALIDATION_NAME,
    )
    summary = _self_hashed(
        {
            "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "split": split,
            "registered_world_seeds": list(registered_seeds(split)),
            "n_registered_worlds": len(registered_seeds(split)),
            "n_discovery_successes": sum(
                str(value["status"]).casefold() == "ok" for value in attempts
            ),
            "n_scoring_successes": sum(
                str(value["scoring_status"]).casefold() == "ok" for value in records
            ),
            "all_failures_retained_in_denominator": True,
            "all_candidates_loaded_before_any_scorer_read": True,
            "executable_source_unchanged_through_scoring": True,
            "maximum_workers": 2,
            "workers_used": int(workers),
            "block_certified": bool(block_validation["block_certified"]),
            "artifact_sha256": {
                filename: subspace_scorer.file_sha256(output / filename)
                for filename in core_artifacts
            },
        },
        "run_summary_payload_sha256",
    )
    _write_new_json(output / RUN_SUMMARY_NAME, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("TRAIN", "DEVELOPMENT", "LOCKED"), required=True
    )
    parser.add_argument("--worlds-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate only absent registered worlds; never replace partial worlds.",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--development-validation", type=Path)
    parser.add_argument(
        "--empirical-null-calibrator",
        type=Path,
        default=DEFAULT_EMPIRICAL_NULL_CALIBRATOR,
    )
    parser.add_argument(
        "--multievidence-calibrator",
        type=Path,
        default=DEFAULT_MULTIEVIDENCE_CALIBRATOR,
    )
    parser.add_argument(
        "--topology-consensus-calibrator",
        type=Path,
        default=DEFAULT_TOPOLOGY_CONSENSUS_CALIBRATOR,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_block(
        split_name=args.split,
        worlds_root=args.worlds_root,
        output_root=args.out_dir,
        generate_missing=args.generate_missing,
        workers=args.workers,
        resume=args.resume,
        development_validation_path=args.development_validation,
        empirical_null_calibrator_path=args.empirical_null_calibrator,
        multievidence_calibrator_path=args.multievidence_calibrator,
        topology_consensus_calibrator_path=(
            args.topology_consensus_calibrator
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "BLOCK_VALIDATION_NAME",
    "GLOBAL_FREEZE_NAME",
    "REGISTRATION_NAME",
    "RUNNER_SCHEMA_VERSION",
    "freeze_global_pretruth_barrier",
    "load_all_candidates_before_scorer",
    "registered_seeds",
    "registered_world_paths",
    "run_block",
    "run_discovery_phase",
    "score_loaded_candidates",
]
