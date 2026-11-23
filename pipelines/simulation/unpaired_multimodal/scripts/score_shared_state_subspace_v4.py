#!/usr/bin/env python3
"""Score the estimable CA-BOSS v4 shared-state-subspace contract.

The disjoint-cell design can identify whether assay-local focal variables
respond reproducibly to their estimated state subspaces.  Interpreting those
responses as one *shared* biological-state subspace additionally assumes the
common-state integration model; it is not established by focal responsiveness
alone.  The primary observable object is therefore a multi-assay
state-responsive set.  Its one n-ary
``cross_assay_shared_state_subspace`` label is hypothesis-only and conditional
on that model.  No common coordinate axis, cross-assay feature pair, or
cross-assay causal direction is credited.

Scorer-only simulator roles define the member contract:

* P, Q, S, and downstream collider child C are state-responsive members;
* C is additionally annotated as overlap with the direct residual layer;
* N1 and N2 are matched sentinel nulls and are not members;
* the RNA gene is state responsive, but receives no exact peak-link credit.

The module also preregisters independent fresh validation blocks and supplies
an all-world pretruth barrier.  Discovery failures and scoring failures remain
in the denominator.  Existing pairwise truth may optionally be passed to
``score_v4`` solely to retain within-assay direct/collider diagnostics; its
pairwise cross-assay metrics are not part of this estimand.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import assay_local_state_embedding_v4 as assay_local_embedding
from score_multimodal_v4_graph import score_v4


SCORER_SCHEMA_VERSION = "shared_state_subspace_v4.scorer.v14"
PROTOCOL_SCHEMA_VERSION = "shared_state_subspace_v4.validation_protocol.v16"
GLOBAL_FREEZE_SCHEMA_VERSION = "shared_state_subspace_v4.global_pretruth_freeze.v14"
BLOCK_VALIDATION_SCHEMA_VERSION = "shared_state_subspace_v4.block_validation.v14"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "shared_state_subspace_v4.frozen_discovery.v10"
PROVENANCE_SCHEMA_VERSION = "shared_state_subspace_v4.v10"
FROZEN_DISCOVERY_SOURCE_SHA256 = (
    "1c879a109cfc60a5db3b2c75f46a9bcb7baa6714d18507a405b5a68c601f0ba7"
)

SUBSPACE_ID = "shared_latent_state_subspace_001"
SUBSPACE_EDGE_TYPE = "cross_assay_shared_state_subspace"
STATE_ROLES = ("P", "Q", "S", "C")
CORE_STATE_ROLES = ("P", "Q", "S")
NEGATIVE_ROLES = ("N1", "N2")
ALL_ROLES = frozenset((*STATE_ROLES, *NEGATIVE_ROLES))

# Seeds 30001--30024 and the four retired layered TRAIN blocks 33001--33024,
# 36001--36024, 37001--37024, 38001--38024, and 40001--40024 informed the
# subspace redesign and subsequent conservative structural/loading repairs.
# TRAIN390 froze all discovery outputs but was retired before scorer access
# because a staged loader schema did not match the unchanged v7 discovery
# producer.  TRAIN410 passed its confirmation gate, then DEV340 exposed two
# exceptional direct-layer failures; both are now permanently development-only.
# LOCKED450 and LOCKED480 were never opened and are retired unopened because
# their executable candidates changed.  TRAIN460 passed confirmation, while
# DEV470 exposed the one sentinel that motivated the v10 subtractive strong-
# core guard; both opened blocks are now development-only.  The exact-axis
# DEV/LOCKED blocks 31001--31016 and 42001--42040 likewise remain unopened and
# reserved for their superseded protocols.  The registered blocks below are
# independent and untouched.
FRESH_TRAIN_SEEDS = tuple(range(49_001, 49_025))
FRESH_DEVELOPMENT_SEEDS = tuple(range(50_001, 50_017))
FRESH_LOCKED_SEEDS = tuple(range(51_001, 51_041))
SPLIT_SEEDS = {
    "TRAIN": FRESH_TRAIN_SEEDS,
    "DEVELOPMENT": FRESH_DEVELOPMENT_SEEDS,
    "LOCKED": FRESH_LOCKED_SEEDS,
}
MINIMUM_SUCCESSFUL_WORLDS = {
    "TRAIN": 22,
    # Two replicates per DOE cell make DEVELOPMENT an exact 16/16 gate; a
    # single failure would otherwise leave one stress cell with only 1/2.
    "DEVELOPMENT": 16,
    "LOCKED": 36,
}

MIN_MEMBER_PRECISION = 0.95
MIN_CHROMATIN_MEMBER_RECALL = 0.80
MIN_ROLE_RECALL = 0.80
MIN_ASSAY_COVERAGE = 1.00
MIN_EACH_CHROMATIN_ASSAY_MEMBER_RECALL = 0.75
MAX_SENTINEL_INCLUSION_RATE = 0.0
MIN_C_STATE_RESPONSIVE_RECALL = 0.80
MAX_HYPEREDGE_EMPIRICAL_P = 0.10
EXPECTED_ERROR_CONTROL_METHOD = (
    "synchronized_conditional_score_wild_feature_maxT_confirmation_"
    "borderline_direction_stability_"
    "within_assay_redundancy_confirmation_"
    "within_assay_strong_maxT_core_weak_tail_confirmation_"
    "assay_maxT_admission_"
    "within_assay_BH_conservative_order_2_partial_conjunction"
)
EXPECTED_BINARY_ELIGIBILITY_POLICY = (
    "pre_fit_exclusion_before_feature_models_BH_maxT_and_conjunction"
)
EXPECTED_INELIGIBLE_PLACEHOLDER_POLICY: dict[str, Any] = {
    "formal_and_descriptive_statistics": 0.0,
    "empirical_p_fdr_and_feature_maxT_p": 1.0,
    "selection_status": "unresolved",
    "abstention_reasons": [
        "insufficient_binary_events",
        "single_class_training_fold",
        "constant_continuous_response",
    ],
    "included_in_multiplicity_families": False,
}
FROZEN_DISCOVERY_CONFIG: dict[str, Any] = {
    "null_repetitions": 999,
    "random_seed": 0,
    "maximum_empirical_fdr": 0.10,
    "maximum_hyperedge_error_probability": 0.10,
    "minimum_conditional_wald_views": 4,
    "minimum_median_score_direction_stability": 0.80,
    "unstable_feature_maxT_fraction": 0.50,
    "within_assay_strong_core_maxT_p": 0.01,
    "within_assay_strong_core_minimum_members": 4,
    "minimum_modalities": 2,
    "observed_model_views": 6,
    "folds": 5,
    "nuisance_spline_knots": 12,
    "nuisance_spline_degree": 3,
    "nuisance_ridge_alpha": 0.0001,
    "nuisance_logistic_c": 0.1,
    "score_covariance_ridge": 1e-6,
    "minimum_binary_events": 20,
    "minimum_binary_nonevents": 20,
    "assay_state_participation_order": 2,
    "maximum_assay_state_participation_error_probability": 0.10,
    "minimum_assay_state_participation_views": 6,
}
FROZEN_SUBSPACE_EMBEDDING_CONFIG: dict[str, Any] = asdict(
    assay_local_embedding.AssayLocalStateEmbeddingConfig()
)
FROZEN_SUBSPACE_EMBEDDING_METHOD = (
    "two_background_feature_fold_assay_local_pca_procrustes_consensus"
)
FROZEN_LOCAL_CANDIDATE_NAME = "v4_ls0.010_lc0.85_hp0.880_q0.10_k1"
FROZEN_LOCAL_CONFIG: dict[str, Any] = {
    "ridge_alpha": 2.0,
    "state_folds": 5,
    "alignment_mode": "hard",
    "soft_n_microstates": 24,
    "soft_max_rotation_dispersion": 0.35,
    "soft_low_confidence_action": "partial_soft",
    "min_state_r2": 0.05,
    "min_abs_total_r": 0.25,
    "min_abs_predicted_r": 0.75,
    "min_abs_loading_cosine": 0.70,
    "min_state_score": 0.18,
    "min_state_bootstrap_support": 0.70,
    "residual_nonsignificance_p": 0.10,
    "mutual_state_neighbors": 1,
    "min_abs_direct_residual_r": 0.18,
    "min_residual_to_pair_null95": 1.0,
    "direct_residual_q": 0.10,
    "min_direct_bootstrap_support": 0.70,
    "collider_separation_q": 0.20,
    "hierarchical_microstates": 8,
    "hierarchical_min_microstate_cells": 20,
    "hierarchical_min_peak_events": 5,
    "hierarchical_min_abs_effect": 0.20,
    "hierarchical_probability_threshold": 0.88,
    "hierarchical_positive_effect_only": True,
    "loading_folds": 5,
    "loading_bootstrap_repetitions": 50,
    "loading_min_norm": 0.10,
    "loading_min_state_fit": 0.01,
    "loading_min_direction_stability": 0.50,
    "loading_min_abs_cosine": 0.85,
    "loading_max_unit_distance": 0.85,
    "loading_min_pair_state_score": 0.01,
    "loading_within_neighbors": 2,
    "loading_cross_neighbors": 1,
    "gene_state_neighbors": 1,
    "suppress_state_projection_through_direct_colliders": True,
    "group_projection_empirical_q": 0.40,
    "group_projection_min_relative_margin": 0.10,
    "exclude_direct_collider_children_from_state_groups": True,
    "replace_pairwise_state_layer_with_groups": True,
    "residual_permutations": 199,
    "stability_bootstraps": 200,
    "null_strata": 6,
    "emit_unresolved": True,
    "impose_gene_sink_prior": True,
    "seed": 0,
}
REQUIRED_JOINT_EXECUTABLE_SOURCES = frozenset(
    {
        "benchmark_ca_boss_v4.py",
        "assay_local_state_embedding_v4.py",
        "cell_state_loading_groups.py",
        "channel_aware_boss_v4.py",
        "consensus_ca_boss_v4.py",
        "crossfit_hierarchical_collider_v4.py",
        "empirical_null_state_calibration_v4.py",
        "multievidence_empirical_null_state_calibration_v4.py",
        "topology_consensus_state_calibration_v4.py",
        "hierarchical_sparse_edges.py",
        "latent_axis_clustering_v4.py",
        "layered_state_evidence_v4.py",
        "loading_group_projection_v4.py",
        "loading_group_state_guard_v4.py",
        "loading_group_structure_uncertainty_guard_v4.py",
        "loading_group_competing_parent_observability_guard_v4.py",
        "loading_group_structure_equivalence_v4.py",
        "multimodal_simulation_v4.py",
        "observed_collider_structure.py",
        "prepare_ca_boss_v4_inputs.py",
        "run_shared_state_subspace_block_v4.py",
        "score_latent_loading_groups_v4.py",
        "score_multimodal_v4_graph.py",
        "score_layered_state_evidence_v4.py",
        "score_shared_state_subspace_v4.py",
        "shared_state_subspace_v4.py",
        "soft_distributional_alignment.py",
        "state_equivalence_membership_v4.py",
        "stratified_beta_binomial_edges.py",
        "structure_anchored_loading_groups_v4.py",
        "structure_aware_collider_rescue.py",
        "structure_aware_collider_motif.py",
        "unpaired_state_alignment.py",
    }
)
MIN_LOCAL_DIRECT_RECALL = 0.90
MIN_LOCAL_COLLIDER_RECALL = 0.80
MIN_LOCAL_PRIMARY_SKELETON_PRECISION = 0.80
MAX_LOCAL_MATCHED_NULL_FALSE_POSITIVE_RATE = 0.10
MAX_LOCAL_FOCAL_EDGES_PER_WORLD = 24

# Block-level safeguards tolerate a small number of honest world abstentions,
# while claim-schema violations remain forbidden.
MIN_BLOCK_MEAN_MEMBER_PRECISION = 0.90
MIN_BLOCK_MEAN_CHROMATIN_RECALL = 0.80
MIN_BLOCK_MEAN_ASSAY_COVERAGE = 0.90
MIN_BLOCK_MEAN_MINIMUM_CHROMATIN_ASSAY_RECALL = 0.75
MAX_BLOCK_MEAN_SENTINEL_RATE = 0.02
MAX_BLOCK_SENTINEL_WILSON_UCB = 0.02
MIN_BLOCK_MEAN_C_STATE_RESPONSIVE_RECALL = 0.80
MIN_BLOCK_MEAN_LOCAL_DIRECT_RECALL = 0.90
MIN_BLOCK_MEAN_LOCAL_COLLIDER_RECALL = 0.80
MIN_BLOCK_MEAN_LOCAL_PRIMARY_PRECISION = 0.80
MAX_BLOCK_MEAN_LOCAL_NULL_RATE = 0.10
MAX_BLOCK_MEAN_LOCAL_FOCAL_EDGES = 20.0
MIN_BLOCK_MEAN_ROLE_RECALL = 0.80

_HALF_FRACTION = (
    (-1, -1, -1),
    (-1, -1, +1),
    (-1, +1, -1),
    (-1, +1, +1),
    (+1, -1, -1),
    (+1, -1, +1),
    (+1, +1, -1),
    (+1, +1, +1),
)
MIN_DESIGN_CELL_SUCCESSES = {
    "TRAIN": 2,       # 2 / 3 per half-fraction cell
    "DEVELOPMENT": 2, # 2 / 2 per half-fraction cell
    "LOCKED": 3,      # 3 / 5 per half-fraction cell
}
MIN_FACTOR_LEVEL_SUCCESSES = {
    "TRAIN": 10,      # 10 / 12 per level
    "DEVELOPMENT": 7, # 7 / 8 per level
    "LOCKED": 16,     # 16 / 20 per level
}

MEMBERSHIP_COLUMNS = (
    "feature",
    "modality",
    "channel",
    "n_views",
    "n_cells",
    "n_events",
    "n_nonevents",
    "binary_event_gate",
    "formal_test_eligible",
    "formal_test_abstention_reason",
    "oof_marginal_state_fit",
    "oof_depth_only_fit",
    "oof_joint_fit",
    "oof_partial_state_fit",
    "median_conditional_wald",
    "lower_quartile_conditional_wald",
    "median_conditional_score_norm",
    "minimum_conditional_score_norm",
    "median_conditional_score_direction_stability",
    "minimum_conditional_score_direction_stability",
    "conditional_wald__model_seed0",
    "conditional_wald__model_seed1",
    "conditional_wald__model_seed2",
    "conditional_wald__model_seed3",
    "conditional_wald__model_seed4",
    "conditional_wald__model_seed5",
    "empirical_null_p",
    "reference_conditional_wald",
    "empirical_null_fdr",
    "empirical_null_family_size",
    "conditional_wald_null_median",
    "conditional_wald_null_95pct",
    "conditional_wald_above_null_95pct",
    "conditional_wald_view_support",
    "assay_participation_partial_conjunction_p",
    "assay_participation_holm_p",
    "assay_participation_single_feature_order_statistic_p",
    "assay_participation_single_feature_p",
    "assay_participation_single_feature_witness_view_count",
    "assay_participation_single_feature_multiview_replication_gate",
    "assay_participation_partial_conjunction_multiview_replication_gate",
    "assay_participation_omnibus_route_count",
    "assay_participation_omnibus_p",
    "assay_participation_omnibus_holm_p",
    "assay_participation_winning_route",
    "assay_participation_selection_route",
    "assay_participation_selected",
    "assay_participation_order",
    "assay_participation_family_size",
    "assay_participation_contributing_feature_count",
    "assay_participation_minimum_witness_view_count",
    "assay_participation_multiview_replication_gate",
    "assay_participation_reporter",
    "feature_global_maxT_p",
    "feature_global_maxT_confirmation_gate",
    "assay_global_maxT_p",
    "assay_global_maxT_admission_gate",
    "direction_stability_diagnostic",
    "feature_maxT_unconditional_confirmation_gate",
    "feature_direction_stability_confirmation_gate",
    "pre_redundancy_selection_status",
    "pre_redundancy_abstention_reason",
    "selection_status",
    "abstention_reason",
    "redundancy_direction_stability_gate",
    "redundancy_fallback_selected",
    "redundancy_demoted",
    "within_assay_strong_core_member",
    "within_assay_strong_core_count",
    "within_assay_weak_maxT_tail_demoted",
    "direct_layer_overlap",
    "direct_layer_annotation_available",
    "subspace_id",
)
MEMBERSHIP_REQUIRED_COLUMNS = frozenset(MEMBERSHIP_COLUMNS)
HYPEREDGE_COLUMNS = (
    "subspace_id",
    "edge_type",
    "modalities",
    "member_features",
    "n_modalities",
    "n_features",
    "selection_status",
    "empirical_multimodality_p",
    "error_control_method",
    "conditional_on_common_state_model",
    "causal_interpretation",
    "direction_claimed",
    "exact_axis_pairing_claimed",
    "direct_causal_claimed",
    "abstention_reason",
)
HYPEREDGE_REQUIRED_COLUMNS = frozenset(HYPEREDGE_COLUMNS)
DESCRIPTIVE_DECOMPOSITION_FIELDS = (
    "oof_marginal_state_fit",
    "oof_depth_only_fit",
    "oof_joint_fit",
    "oof_partial_state_fit",
)
EXPECTED_DESCRIPTIVE_DECOMPOSITION = {
    "method": "cross_fitted_state_plus_orthogonal_depth_predictive_fit",
    "fields": list(DESCRIPTIVE_DECOMPOSITION_FIELDS),
    "affects_selection": False,
    "shared_state_depth_allocation": "state",
    "calibrated_primary_estimand": "depth_orthogonal_state_response",
}
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SharedStateSubspaceTruth:
    """Dimension-agnostic scorer truth derived after discovery is frozen."""

    features: pd.DataFrame
    assay_groups: pd.DataFrame
    marks: tuple[str, ...]
    gene: str


@dataclass(frozen=True)
class SharedStateSubspaceScore:
    """Metrics plus row-level audit tables."""

    metrics: dict[str, Any]
    feature_scores: pd.DataFrame
    assay_scores: pd.DataFrame
    hyperedge_scores: pd.DataFrame
    truth: SharedStateSubspaceTruth
    local_pair_scores: pd.DataFrame


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
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


def _read_regular_bytes_once(path: Path | str, label: str) -> bytes:
    """Open a non-symlink regular file once for exact-byte hash and parsing."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"Could not open {label} safely: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {source}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def collect_joint_executable_source_sha256(
    scripts_dir: Path | str,
) -> dict[str, str]:
    """Hash the exact local/direct/subspace executable dependency set."""

    root = Path(scripts_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Joint executable source root must be a regular directory")
    hashes: dict[str, str] = {}
    for filename in sorted(REQUIRED_JOINT_EXECUTABLE_SOURCES):
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular joint executable source: {filename}")
        hashes[filename] = file_sha256(path)
    return hashes


def _valid_hash(value: Any, label: str) -> str:
    text = str(value).strip().lower()
    if not _HASH.fullmatch(text):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return text


def _read_json(path: Path | str, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {source}")
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _validate_self_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    claimed = _valid_hash(value.get(field, ""), f"{label} {field}")
    payload = dict(value)
    payload.pop(field, None)
    observed = canonical_sha256(payload)
    if claimed != observed:
        raise ValueError(f"{label} self-hash mismatch")
    return claimed


def _split(value: Any) -> tuple[str, ...]:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ()
    parts = tuple(part.strip() for part in str(value).split(";") if part.strip())
    if len(parts) != len(set(parts)):
        raise ValueError("Semicolon-delimited identifiers may not repeat")
    return parts


def _parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} is not an explicit Boolean")


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _benjamini_hochberg(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return deterministic BH-adjusted p-values for one frozen global family."""

    p_values = np.asarray(values, dtype=np.float64)
    if p_values.ndim != 1 or not np.isfinite(p_values).all() or (
        (p_values < 0.0) | (p_values > 1.0)
    ).any():
        raise ValueError("BH inputs must be one finite probability vector")
    if not len(p_values):
        return np.asarray([], dtype=np.float64)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _holm(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return deterministic Holm-adjusted p-values with strong FWER control."""

    p_values = np.asarray(values, dtype=np.float64)
    if p_values.ndim != 1 or not len(p_values) or not np.isfinite(p_values).all():
        raise ValueError("Holm inputs must be one non-empty finite probability vector")
    if ((p_values < 0.0) | (p_values > 1.0)).any():
        raise ValueError("Holm inputs must lie in [0, 1]")
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order] * np.arange(len(p_values), 0, -1)
    ranked = np.maximum.accumulate(ranked)
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _doe_design_for_seed(seed: int) -> dict[str, int]:
    index = (int(seed) - 1) % len(_HALF_FRACTION)
    a, b, c = _HALF_FRACTION[index]
    return {
        "half_fraction_cell": index + 1,
        "A_cells": a,
        "B_detection": b,
        "C_population_divergence": c,
        "D_background_observability": a * b * c,
    }


def _one_sided_wilson_upper(
    successes: int, total: int, *, z: float = 1.6448536269514722
) -> float:
    """One-sided 95% Wilson upper confidence bound for a binomial rate."""

    if total <= 0:
        return 1.0
    if successes < 0 or successes > total:
        raise ValueError("Binomial successes must lie between zero and total")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return float((center + radius) / denominator)


def _normalise_split(split: str) -> str:
    normalized = split.strip().upper()
    aliases = {"DEV": "DEVELOPMENT", "FRESH_LOCKED": "LOCKED"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SPLIT_SEEDS:
        raise ValueError(f"Unknown validation split: {split}")
    return normalized


def validation_protocol() -> dict[str, Any]:
    """Return the self-hashed, preregistered subspace validation protocol."""

    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "status": "preregistered_before_fresh_world_generation",
        "estimand": {
            "edge_type": SUBSPACE_EDGE_TYPE,
            "primary_observable_type": "multi_assay_state_responsive_set",
            "shared_subspace_interpretation": (
                "hypothesis_only_conditional_on_common_state_model"
            ),
            "common_state_model_identified_by_focal_responsiveness_alone": False,
            "dimension_agnostic_shared_state_subspace": True,
            "exact_latent_axis_identity_scored": False,
            "exact_cross_assay_feature_pairs_scored": False,
            "cross_assay_causal_direction_scored": False,
            "rna_gene_is_state_responsive_support_not_exact_peak_link": True,
            "subspace_hyperedge_expanded_into_pairwise_edges": False,
            "assay_participation_is_distinct_from_strict_feature_membership": True,
            "assay_participation_does_not_rescue_feature_membership": True,
        },
        "fresh_seed_blocks": {
            split: list(seeds) for split, seeds in SPLIT_SEEDS.items()
        },
        "minimum_successful_worlds": MINIMUM_SUCCESSFUL_WORLDS,
        "world_thresholds": {
            "minimum_member_precision": MIN_MEMBER_PRECISION,
            "minimum_chromatin_member_recall": MIN_CHROMATIN_MEMBER_RECALL,
            "minimum_each_P_Q_S_role_recall": MIN_ROLE_RECALL,
            "minimum_assay_coverage": MIN_ASSAY_COVERAGE,
            "minimum_each_chromatin_assay_member_recall": (
                MIN_EACH_CHROMATIN_ASSAY_MEMBER_RECALL
            ),
            "maximum_sentinel_inclusion_rate": MAX_SENTINEL_INCLUSION_RATE,
            "minimum_C_state_responsive_recall": (
                MIN_C_STATE_RESPONSIVE_RECALL
            ),
            "minimum_local_direct_recall": MIN_LOCAL_DIRECT_RECALL,
            "minimum_local_collider_recall": MIN_LOCAL_COLLIDER_RECALL,
            "minimum_local_primary_skeleton_precision": (
                MIN_LOCAL_PRIMARY_SKELETON_PRECISION
            ),
            "maximum_local_matched_null_false_positive_rate": (
                MAX_LOCAL_MATCHED_NULL_FALSE_POSITIVE_RATE
            ),
            "maximum_local_focal_edges_per_world": MAX_LOCAL_FOCAL_EDGES_PER_WORLD,
            "zero_strict_reverse_compelled": True,
            "zero_exact_axis_or_direct_cross_assay_claims": True,
            "maximum_hyperedge_empirical_multimodality_p": (
                MAX_HYPEREDGE_EMPIRICAL_P
            ),
        },
        "block_thresholds": {
            "minimum_mean_member_precision_all_registered_worlds": (
                MIN_BLOCK_MEAN_MEMBER_PRECISION
            ),
            "minimum_mean_chromatin_recall_all_registered_worlds": (
                MIN_BLOCK_MEAN_CHROMATIN_RECALL
            ),
            "minimum_mean_assay_coverage_all_registered_worlds": (
                MIN_BLOCK_MEAN_ASSAY_COVERAGE
            ),
            "minimum_mean_of_world_minimum_chromatin_assay_recall": (
                MIN_BLOCK_MEAN_MINIMUM_CHROMATIN_ASSAY_RECALL
            ),
            "maximum_mean_sentinel_rate": MAX_BLOCK_MEAN_SENTINEL_RATE,
            "maximum_one_sided_95pct_Wilson_sentinel_UCB": (
                MAX_BLOCK_SENTINEL_WILSON_UCB
            ),
            "minimum_mean_C_state_responsive_recall": (
                MIN_BLOCK_MEAN_C_STATE_RESPONSIVE_RECALL
            ),
            "minimum_mean_each_P_Q_S_role_recall": MIN_BLOCK_MEAN_ROLE_RECALL,
            "minimum_mean_local_direct_recall_all_registered_worlds": (
                MIN_BLOCK_MEAN_LOCAL_DIRECT_RECALL
            ),
            "minimum_mean_local_collider_recall_all_registered_worlds": (
                MIN_BLOCK_MEAN_LOCAL_COLLIDER_RECALL
            ),
            "minimum_mean_local_primary_precision_all_registered_worlds": (
                MIN_BLOCK_MEAN_LOCAL_PRIMARY_PRECISION
            ),
            "maximum_mean_local_matched_null_rate": MAX_BLOCK_MEAN_LOCAL_NULL_RATE,
            "maximum_mean_local_focal_edges": MAX_BLOCK_MEAN_LOCAL_FOCAL_EDGES,
            "unsupported_claim_flag_count": 0,
            "minimum_design_cell_successes": MIN_DESIGN_CELL_SUCCESSES,
            "minimum_factor_level_successes": MIN_FACTOR_LEVEL_SUCCESSES,
        },
        "barrier": {
            "all_registered_discovery_attempts_finish_before_any_scorer_read": True,
            "candidate_outputs_and_algorithm_source_are_content_addressed": True,
            "joint_local_and_subspace_dependency_sources_are_content_addressed": True,
            "formal_assay_local_embedding_bundle_is_content_addressed": True,
            "local_pretruth_manifest_and_consensus_bytes_loaded_before_scorer": True,
            "discovery_and_scoring_failures_remain_in_denominator": True,
            "development_source_hash_must_bind_locked": True,
            "training_worlds_are_noncertifying": True,
            "TRAIN_49001_49024_is_one_nonadaptive_confirmation_block": True,
            "source_config_protocol_frozen_before_any_fresh_TRAIN_truth_read": True,
            "source_changes_after_first_fresh_TRAIN_truth_read_forbidden": True,
            "observable_only_null_calibration_must_precede_global_freeze": True,
        },
        "history_boundary": {
            "seeds_30001_30024": "diagnostic_and_algorithm_development_only",
            "seeds_33001_33024": "diagnostic_and_algorithm_development_only",
            "seeds_36001_36024": "diagnostic_and_algorithm_development_only",
            "seeds_37001_37024": "diagnostic_and_algorithm_development_only",
            "seeds_38001_38024": "diagnostic_and_algorithm_development_only",
            "seeds_39001_39024": (
                "aborted_pretruth_infrastructure_schema_mismatch_no_scorer_opened"
            ),
            "seeds_40001_40024": (
                "failed_scientific_confirmation_then_algorithm_development_only"
            ),
            "seeds_41001_41024": (
                "passed_confirmation_then_algorithm_changed_and_development_only"
            ),
            "seeds_34001_34016": (
                "opened_development_failed_v43_then_posthoc_passed_v44_"
                "development_only"
            ),
            "seeds_45001_45040": (
                "unopened_reserved_superseded_before_v44_fresh_validation"
            ),
            "seeds_46001_46024": (
                "passed_confirmation_then_DEV470_informed_v45_and_"
                "development_only"
            ),
            "seeds_47001_47016": (
                "opened_development_failed_v44_then_posthoc_passed_v45_"
                "development_only"
            ),
            "seeds_48001_48040": (
                "unopened_reserved_superseded_before_v45_fresh_validation"
            ),
            "seeds_31001_31016": "unopened_reserved_superseded_exact_axis_DEV",
            "seeds_42001_42040": "unopened_reserved_superseded_exact_axis_LOCKED",
        },
        "postcandidate_baseline_diagnostics": {
            "frozen_v3_BOSS_FCI": "run_only_after_candidate_freeze;never_select_candidate",
            "production_FCI": "run_only_after_candidate_freeze;never_select_candidate",
            "baseline_results_available_to_candidate_selection": False,
            "baseline_results_enter_subspace_pass_fail_gate": False,
        },
        "density_objective": {
            "historical_ordinary_graph_mean_edges": "approximately_23",
            "fresh_per_world_ceiling": MAX_LOCAL_FOCAL_EDGES_PER_WORLD,
            "fresh_block_scored_world_mean_ceiling": (
                MAX_BLOCK_MEAN_LOCAL_FOCAL_EDGES
            ),
            "unresolved_pair_edges_count_toward_density": True,
            "nary_subspace_hyperedge_counts_as_one_and_is_not_pair_expanded": True,
        },
        "empirical_error_control": {
            "binary_eligibility": EXPECTED_BINARY_ELIGIBILITY_POLICY,
            "ineligible_placeholder_policy": (
                EXPECTED_INELIGIBLE_PLACEHOLDER_POLICY
            ),
            "feature_level": (
                "world_feature_maxT_confirmation_and_within_assay_BH_"
                "depth_orthogonal_score"
            ),
            "assay_admission": "synchronized_world_assay_maxT_FWER",
            "feature_global_maxT": "membership_confirmation_gate",
            "direction_stability": "diagnostic_only",
            "strong_core_weak_tail_confirmation": {
                "strong_core_maxT_p": FROZEN_DISCOVERY_CONFIG[
                    "within_assay_strong_core_maxT_p"
                ],
                "minimum_strong_core_members": FROZEN_DISCOVERY_CONFIG[
                    "within_assay_strong_core_minimum_members"
                ],
                "evaluated_after_redundancy_confirmation": True,
                "strictly_subtractive": True,
                "affects_assay_participation": False,
            },
            "minimum_conditional_wald_views": FROZEN_DISCOVERY_CONFIG[
                "minimum_conditional_wald_views"
            ],
            "observed_model_views": FROZEN_DISCOVERY_CONFIG[
                "observed_model_views"
            ],
            "assay_and_hyperedge_level": EXPECTED_ERROR_CONTROL_METHOD,
            "assay_state_participation": {
                "null": "no_state_responsive_features_in_the_assay",
                "within_assay_single_feature_test": (
                    "bonferroni_minP_valid_under_arbitrary_feature_dependence"
                ),
                "within_assay_partial_conjunction_test": (
                    "bonferroni_partial_conjunction_valid_under_arbitrary_"
                    "feature_dependence"
                ),
                "order": FROZEN_DISCOVERY_CONFIG[
                    "assay_state_participation_order"
                ],
                "within_assay_omnibus_combination": (
                    "bonferroni_over_predeclared_replication_gated_routes"
                ),
                "omnibus_route_count_policy": (
                    "one_for_structural_singleton_otherwise_two"
                ),
                "across_assay_error_control": (
                    "holm_strong_FWER_without_cross_assay_dependence_assumption"
                ),
                "maximum_error_probability": FROZEN_DISCOVERY_CONFIG[
                    "maximum_assay_state_participation_error_probability"
                ],
                "minimum_witness_views": FROZEN_DISCOVERY_CONFIG[
                    "minimum_assay_state_participation_views"
                ],
                "complete_null_order_statistic_minP_used": True,
                "affects_strict_feature_membership": False,
                "affects_existing_typed_subspace_hyperedge": False,
            },
            "maximum_probability": MAX_HYPEREDGE_EMPIRICAL_P,
            "synchronized_null_required": True,
        },
        "frozen_discovery_source_sha256": FROZEN_DISCOVERY_SOURCE_SHA256,
        "output_contract": {
            "candidate_manifest_schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "membership_columns_in_order": list(MEMBERSHIP_COLUMNS),
            "hyperedge_columns_in_order": list(HYPEREDGE_COLUMNS),
            "descriptive_state_depth_decomposition": (
                EXPECTED_DESCRIPTIVE_DECOMPOSITION
            ),
            "typed_subspace_hyperedge_is_native_nary": True,
            "pairwise_expansion_forbidden": True,
            "assay_participation_is_a_separate_first_class_table_in_provenance": True,
        },
        "frozen_discovery_config": FROZEN_DISCOVERY_CONFIG,
        "formal_subspace_embedding": {
            "method": FROZEN_SUBSPACE_EMBEDDING_METHOD,
            "config": FROZEN_SUBSPACE_EMBEDDING_CONFIG,
            "config_sha256": canonical_sha256(
                FROZEN_SUBSPACE_EMBEDDING_CONFIG
            ),
            "source_basename": "assay_local_state_embedding_v4.py",
            "source_sha256": file_sha256(
                Path(assay_local_embedding.__file__).resolve(strict=True)
            ),
            "input_domain": "held_out_background_matrices_and_leave_focal_out_depth",
            "formal_state_dimension": 3,
            "feature_folds": 2,
            "same_assay_procrustes_consensus": True,
            "cross_assay_alignment_used": False,
            "zero_padding_used": False,
            "parallel_analysis_role": "observed_only_rank_three_abstention_guard",
            "parallel_rank_below_three_action": "world_abstention",
        },
        "frozen_local_structure_candidate": {
            "candidate_name": FROZEN_LOCAL_CANDIDATE_NAME,
            "config": FROZEN_LOCAL_CONFIG,
            "config_sha256": canonical_sha256(FROZEN_LOCAL_CONFIG),
            "hard_preparation_is_formal_primary_for_local_structure": True,
            "hard_preparation_is_formal_primary_for_subspace_membership": False,
        },
    }
    protocol["protocol_payload_sha256"] = canonical_sha256(protocol)
    return protocol


def build_subspace_truth(
    world_parameters: Mapping[str, Any], role_assignments: pd.DataFrame
) -> SharedStateSubspaceTruth:
    """Build scorer truth without using latent coordinates or exact axes."""

    required_parameters = {"seed", "marks", "gene"}
    missing_parameters = sorted(required_parameters - set(world_parameters))
    if missing_parameters:
        raise ValueError(f"World parameters are missing {missing_parameters}")
    required_roles = {"node", "mark", "role"}
    missing_roles = sorted(required_roles - set(role_assignments.columns))
    if missing_roles:
        raise ValueError(f"Role assignments are missing {missing_roles}")

    marks = tuple(str(mark) for mark in world_parameters["marks"])
    if len(marks) != 6 or len(set(marks)) != 6:
        raise ValueError("The v4 subspace contract requires six unique chromatin assays")
    roles = role_assignments.copy()
    roles["node"] = roles["node"].astype(str)
    roles["mark"] = roles["mark"].astype(str)
    roles["role"] = roles["role"].astype(str)
    if roles["node"].duplicated().any():
        raise ValueError("Role-assignment nodes must be unique")
    if set(roles["mark"]) != set(marks):
        raise ValueError("Role-assignment marks do not match world parameters")
    if set(roles["role"]) != ALL_ROLES:
        raise ValueError("Role assignments do not contain exactly the v4 role alphabet")
    for mark in marks:
        counts = roles.loc[roles["mark"] == mark, "role"].value_counts().to_dict()
        if any(int(counts.get(role, 0)) != 1 for role in ALL_ROLES):
            raise ValueError(f"Mark {mark} must have exactly one feature per v4 role")

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for mark in marks:
        mark_roles = roles.loc[roles["mark"] == mark].set_index("role")
        expected_members = tuple(
            sorted(str(mark_roles.loc[role, "node"]) for role in STATE_ROLES)
        )
        groups.append(
            {
                "assay_group": f"state_responsive_subspace__{mark}",
                "modality": f"chrom_{mark}",
                "channel": mark,
                "expected_members": ";".join(expected_members),
                "n_expected_members": len(expected_members),
            }
        )
        for record in mark_roles.reset_index().to_dict(orient="records"):
            role = str(record["role"])
            if role in STATE_ROLES:
                truth_class = (
                    "state_responsive_chromatin_direct_layer_overlap"
                    if role == "C"
                    else "state_responsive_chromatin"
                )
                eligible = True
            else:
                truth_class = "sentinel_null"
                eligible = False
            rows.append(
                {
                    "feature": str(record["node"]),
                    "modality": f"chrom_{mark}",
                    "channel": mark,
                    "mark": mark,
                    "role": role,
                    "truth_class": truth_class,
                    "expected_subspace_member": bool(eligible),
                    "true_direct_layer_overlap": role == "C",
                }
            )

    gene = str(world_parameters["gene"]).strip()
    if not gene or gene in set(roles["node"]):
        raise ValueError("The RNA gene must be a distinct non-empty focal node")
    rows.append(
        {
            "feature": gene,
            "modality": "rna",
            "channel": "RNA",
            "mark": "",
            "role": "GENE",
            "truth_class": "state_responsive_gene",
            "expected_subspace_member": True,
            "true_direct_layer_overlap": False,
        }
    )
    groups.append(
        {
            "assay_group": "state_responsive_subspace__RNA",
            "modality": "rna",
            "channel": "RNA",
            "expected_members": gene,
            "n_expected_members": 1,
        }
    )
    features = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    return SharedStateSubspaceTruth(
        features=features,
        assay_groups=pd.DataFrame(groups),
        marks=marks,
        gene=gene,
    )


def _validate_membership_table(
    membership: pd.DataFrame, truth: SharedStateSubspaceTruth
) -> pd.DataFrame:
    missing = sorted(MEMBERSHIP_REQUIRED_COLUMNS - set(membership.columns))
    if missing:
        raise ValueError(f"Subspace membership is missing columns {missing}")
    extra = sorted(set(membership.columns) - MEMBERSHIP_REQUIRED_COLUMNS)
    if extra:
        raise ValueError(f"Subspace membership has unregistered columns {extra}")
    if tuple(membership.columns) != MEMBERSHIP_COLUMNS:
        raise ValueError("Subspace membership columns are not in the frozen v10 order")
    frame = membership.copy().reset_index(drop=True)
    frame["feature"] = frame["feature"].astype(str)
    if frame["feature"].duplicated().any():
        raise ValueError("Subspace membership features must be unique")
    expected_features = set(truth.features["feature"].astype(str))
    if set(frame["feature"]) != expected_features:
        raise ValueError("Subspace membership must enumerate every focal feature exactly once")

    expected = truth.features.set_index("feature")
    for record in frame.to_dict(orient="records"):
        feature = str(record["feature"])
        if str(record["modality"]) != str(expected.loc[feature, "modality"]):
            raise ValueError(f"Candidate modality mismatch for {feature}")
        if str(record["channel"]) != str(expected.loc[feature, "channel"]):
            raise ValueError(f"Candidate channel mismatch for {feature}")
    allowed_statuses = {"selected", "unresolved"}
    statuses = frame["selection_status"].astype(str).str.casefold()
    if not set(statuses) <= allowed_statuses:
        raise ValueError("Unknown subspace membership selection status")
    frame["selection_status"] = statuses
    boolean_columns = (
        "binary_event_gate",
        "formal_test_eligible",
        "assay_participation_selected",
        "assay_participation_multiview_replication_gate",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_reporter",
        "feature_global_maxT_confirmation_gate",
        "assay_global_maxT_admission_gate",
        "direction_stability_diagnostic",
        "feature_maxT_unconditional_confirmation_gate",
        "feature_direction_stability_confirmation_gate",
        "redundancy_direction_stability_gate",
        "redundancy_fallback_selected",
        "redundancy_demoted",
        "within_assay_strong_core_member",
        "within_assay_weak_maxT_tail_demoted",
        "direct_layer_overlap",
        "direct_layer_annotation_available",
    )
    for column in boolean_columns:
        frame[column] = frame[column].map(
            lambda value, label=column: _parse_bool(value, label)
        )
    frame["formal_test_abstention_reason"] = (
        frame["formal_test_abstention_reason"].fillna("").astype(str)
    )
    formal_eligible = frame["formal_test_eligible"].to_numpy(dtype=bool)

    numeric_columns = [
        "n_views",
        "n_cells",
        "n_events",
        "n_nonevents",
        "oof_marginal_state_fit",
        "oof_depth_only_fit",
        "oof_joint_fit",
        "oof_partial_state_fit",
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "median_conditional_score_norm",
        "minimum_conditional_score_norm",
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
        *[
            f"conditional_wald__model_seed{view}"
            for view in range(FROZEN_DISCOVERY_CONFIG["observed_model_views"])
        ],
        "empirical_null_p",
        "reference_conditional_wald",
        "empirical_null_fdr",
        "empirical_null_family_size",
        "conditional_wald_null_median",
        "conditional_wald_null_95pct",
        "conditional_wald_above_null_95pct",
        "conditional_wald_view_support",
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p",
        "assay_participation_single_feature_p",
        "assay_participation_single_feature_witness_view_count",
        "assay_participation_omnibus_route_count",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "feature_global_maxT_p",
        "assay_global_maxT_p",
        "within_assay_strong_core_count",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Subspace evidence columns must be finite numeric values")
    integer_columns = (
        "n_views",
        "n_cells",
        "n_events",
        "n_nonevents",
        "empirical_null_family_size",
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_single_feature_witness_view_count",
        "assay_participation_omnibus_route_count",
        "within_assay_strong_core_count",
    )
    if not np.allclose(
        numeric[list(integer_columns)], np.round(numeric[list(integer_columns)])
    ):
        raise ValueError("Count evidence must contain integers")
    if not (
        numeric["n_views"]
        == int(FROZEN_DISCOVERY_CONFIG["observed_model_views"])
    ).all():
        raise ValueError("n_views disagrees with the frozen discovery config")
    if (numeric["n_cells"] < 1).any():
        raise ValueError("n_cells must contain positive integers")
    rna_mask = frame["modality"].astype(str).eq("rna")
    if not (
        (numeric.loc[rna_mask, "n_events"] == -1).all()
        and (numeric.loc[rna_mask, "n_nonevents"] == -1).all()
    ):
        raise ValueError("RNA rows must use -1 binary-event sentinels")
    chrom_mask = ~rna_mask
    if (
        (numeric.loc[chrom_mask, ["n_events", "n_nonevents"]] < 0).any().any()
        or not np.allclose(
            numeric.loc[chrom_mask, "n_events"]
            + numeric.loc[chrom_mask, "n_nonevents"],
            numeric.loc[chrom_mask, "n_cells"],
        )
    ):
        raise ValueError("Chromatin event counts must be nonnegative and sum to n_cells")
    if (numeric["empirical_null_family_size"] < 0).any():
        raise ValueError("empirical_null_family_size must be nonnegative")
    descriptive_fits = list(DESCRIPTIVE_DECOMPOSITION_FIELDS)
    if (numeric[descriptive_fits] > 1.0 + 1e-10).any().any():
        raise ValueError("OOF descriptive fit fractions must not exceed one")
    depth_denominator = 1.0 - numeric["oof_depth_only_fit"]
    algebra_mask = depth_denominator.abs() > 1e-10
    expected_partial_fit = (
        numeric.loc[algebra_mask, "oof_joint_fit"]
        - numeric.loc[algebra_mask, "oof_depth_only_fit"]
    ) / depth_denominator.loc[algebra_mask]
    if not np.allclose(
        numeric.loc[algebra_mask, "oof_partial_state_fit"],
        expected_partial_fit,
        atol=1e-8,
        rtol=1e-7,
    ):
        raise ValueError("OOF joint/depth/partial fit decomposition is inconsistent")
    bounded = [
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
        "empirical_null_p",
        "empirical_null_fdr",
        "conditional_wald_view_support",
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p",
        "assay_participation_single_feature_p",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "feature_global_maxT_p",
        "assay_global_maxT_p",
    ]
    if ((numeric[bounded] < 0.0) | (numeric[bounded] > 1.0)).any().any():
        raise ValueError("Probability/stability evidence must lie in [0, 1]")
    nonnegative = [
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "median_conditional_score_norm",
        "minimum_conditional_score_norm",
        *[
            f"conditional_wald__model_seed{view}"
            for view in range(FROZEN_DISCOVERY_CONFIG["observed_model_views"])
        ],
        "reference_conditional_wald",
        "conditional_wald_null_median",
        "conditional_wald_null_95pct",
        "within_assay_strong_core_count",
    ]
    if (numeric[nonnegative] < 0.0).any().any():
        raise ValueError("Conditional-Wald and score-norm evidence must be nonnegative")
    if (
        (numeric["lower_quartile_conditional_wald"] > numeric["median_conditional_wald"]).any()
        or (
            numeric["minimum_conditional_score_norm"]
            > numeric["median_conditional_score_norm"]
        ).any()
        or (
            numeric["minimum_conditional_score_direction_stability"]
            > numeric["median_conditional_score_direction_stability"]
        ).any()
        or (
            numeric["conditional_wald_null_median"]
            > numeric["conditional_wald_null_95pct"]
        ).any()
    ):
        raise ValueError("Subspace summary statistics have an invalid ordering")

    view_columns = [
        f"conditional_wald__model_seed{view}"
        for view in range(FROZEN_DISCOVERY_CONFIG["observed_model_views"])
    ]
    view_statistics = numeric[view_columns].to_numpy(dtype=float)
    if not np.allclose(
        numeric["median_conditional_wald"],
        np.median(view_statistics, axis=1),
        atol=1e-10,
        rtol=1e-8,
    ) or not np.allclose(
        numeric["lower_quartile_conditional_wald"],
        np.quantile(view_statistics, 0.25, axis=1),
        atol=1e-10,
        rtol=1e-8,
    ):
        raise ValueError("Descriptive multiview conditional-Wald summaries are inconsistent")
    expected_view_passes = formal_eligible[:, None] & (
        view_statistics
        >= numeric["conditional_wald_null_95pct"].to_numpy(dtype=float)[:, None]
    )
    expected_view_support = np.mean(expected_view_passes, axis=1)
    if not np.allclose(
        numeric["conditional_wald_view_support"], expected_view_support
    ):
        raise ValueError("conditional_wald_view_support is inconsistent with its views")

    expected_binary_gate = rna_mask | (
        (numeric["n_events"] >= FROZEN_DISCOVERY_CONFIG["minimum_binary_events"])
        & (
            numeric["n_nonevents"]
            >= FROZEN_DISCOVERY_CONFIG["minimum_binary_nonevents"]
        )
    )
    if not np.array_equal(
        frame["binary_event_gate"].to_numpy(dtype=bool),
        expected_binary_gate.to_numpy(dtype=bool),
    ):
        raise ValueError("binary_event_gate disagrees with observed event counts")
    formal_eligible = frame["formal_test_eligible"].to_numpy(dtype=bool)
    formal_reasons = frame["formal_test_abstention_reason"].astype(str)
    count_ineligible = ~frame["binary_event_gate"].to_numpy(dtype=bool)
    allowed_formal_reasons = {
        "",
        "insufficient_binary_events",
        "single_class_training_fold",
        "constant_continuous_response",
    }
    if not set(formal_reasons) <= allowed_formal_reasons:
        raise ValueError("Unknown formal-test abstention reason")
    if np.any(formal_eligible & count_ineligible):
        raise ValueError("Count-ineligible features cannot enter formal tests")
    if np.any(formal_eligible & formal_reasons.ne("").to_numpy(dtype=bool)):
        raise ValueError("Formally eligible features cannot carry an abstention reason")
    if np.any(
        count_ineligible
        & formal_reasons.ne("insufficient_binary_events").to_numpy(dtype=bool)
    ):
        raise ValueError("Count-ineligible features need the exact abstention reason")
    if np.any(
        (~formal_eligible)
        & (~count_ineligible)
        & (~formal_reasons.isin(
            {"single_class_training_fold", "constant_continuous_response"}
        ).to_numpy(dtype=bool))
    ):
        raise ValueError("Fit-ineligible features need an auditable abstention reason")
    if np.any(
        formal_reasons.eq("single_class_training_fold").to_numpy(dtype=bool)
        & rna_mask.to_numpy(dtype=bool)
    ) or np.any(
        formal_reasons.eq("constant_continuous_response").to_numpy(dtype=bool)
        & chrom_mask.to_numpy(dtype=bool)
    ):
        raise ValueError("Formal-test abstention reason disagrees with modality")
    expected_feature_diagnostics = (
        numeric["feature_global_maxT_p"] <= MAX_HYPEREDGE_EMPIRICAL_P
    )
    if not np.array_equal(
        frame["feature_global_maxT_confirmation_gate"].to_numpy(dtype=bool),
        expected_feature_diagnostics.to_numpy(dtype=bool),
    ):
        raise ValueError(
            "feature_global_maxT_confirmation_gate disagrees with its p-value"
        )
    expected_assay_admission = (
        numeric["assay_global_maxT_p"] <= MAX_HYPEREDGE_EMPIRICAL_P
    )
    if not np.array_equal(
        frame["assay_global_maxT_admission_gate"].to_numpy(dtype=bool),
        expected_assay_admission.to_numpy(dtype=bool),
    ):
        raise ValueError(
            "assay_global_maxT_admission_gate disagrees with assay_global_maxT_p"
        )
    expected_direction_diagnostic = (
        numeric["median_conditional_score_direction_stability"]
        >= FROZEN_DISCOVERY_CONFIG["minimum_median_score_direction_stability"]
    )
    if not np.array_equal(
        frame["direction_stability_diagnostic"].to_numpy(dtype=bool),
        expected_direction_diagnostic.to_numpy(dtype=bool),
    ):
        raise ValueError("direction_stability_diagnostic disagrees with its statistic")
    expected_unconditional_confirmation = (
        numeric["feature_global_maxT_p"]
        <= (
            MAX_HYPEREDGE_EMPIRICAL_P
            * FROZEN_DISCOVERY_CONFIG["unstable_feature_maxT_fraction"]
        )
    )
    if not np.array_equal(
        frame["feature_maxT_unconditional_confirmation_gate"].to_numpy(dtype=bool),
        expected_unconditional_confirmation.to_numpy(dtype=bool),
    ):
        raise ValueError(
            "feature_maxT_unconditional_confirmation_gate disagrees with its p-value"
        )
    expected_stability_confirmation = (
        expected_unconditional_confirmation | expected_direction_diagnostic
    )
    if not np.array_equal(
        frame["feature_direction_stability_confirmation_gate"].to_numpy(dtype=bool),
        expected_stability_confirmation.to_numpy(dtype=bool),
    ):
        raise ValueError(
            "feature_direction_stability_confirmation_gate disagrees with its evidence"
        )
    ineligible = ~formal_eligible
    zero_placeholder_columns = [
        *DESCRIPTIVE_DECOMPOSITION_FIELDS,
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "median_conditional_score_norm",
        "minimum_conditional_score_norm",
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
        *view_columns,
        "reference_conditional_wald",
        "conditional_wald_null_median",
        "conditional_wald_null_95pct",
        "conditional_wald_above_null_95pct",
        "conditional_wald_view_support",
    ]
    if ineligible.any() and not np.allclose(
        numeric.loc[ineligible, zero_placeholder_columns], 0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("Formally ineligible features need exact zero diagnostics")
    one_placeholder_columns = [
        "empirical_null_p",
        "empirical_null_fdr",
        "feature_global_maxT_p",
    ]
    if ineligible.any() and not np.allclose(
        numeric.loc[ineligible, one_placeholder_columns], 1.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("Formally ineligible features need exact unit p-value placeholders")
    if ineligible.any() and (
        frame.loc[ineligible, "feature_global_maxT_confirmation_gate"].any()
        or frame.loc[ineligible, "direction_stability_diagnostic"].any()
        or frame.loc[
            ineligible, "feature_maxT_unconditional_confirmation_gate"
        ].any()
        or frame.loc[
            ineligible, "feature_direction_stability_confirmation_gate"
        ].any()
    ):
        raise ValueError("Formally ineligible features cannot pass diagnostics")
    view_count = np.sum(expected_view_passes, axis=1)
    participation_expectations: dict[str, dict[str, Any]] = {}
    for modality, group in frame.groupby("modality", sort=True):
        if group["assay_global_maxT_p"].nunique() != 1 or group[
            "assay_global_maxT_admission_gate"
        ].nunique() != 1:
            raise ValueError("Assay-global error evidence must be constant within assay")
        group_formal = frame.loc[group.index, "formal_test_eligible"].to_numpy(
            dtype=bool
        )
        family_size = int(group_formal.sum())
        if not (
            numeric.loc[group.index, "empirical_null_family_size"] == family_size
        ).all():
            raise ValueError(
                "Empirical-null family size disagrees with eligible assay family"
            )
        expected_fdr = np.ones(len(group), dtype=np.float64)
        if family_size:
            expected_fdr[group_formal] = _benjamini_hochberg(
                numeric.loc[
                    np.asarray(group.index)[group_formal], "empirical_null_p"
                ].to_numpy(dtype=float)
            )
        if not np.allclose(
            numeric.loc[group.index, "empirical_null_fdr"],
            expected_fdr,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("empirical_null_fdr is not eligible-only within-assay BH")
        expected_assay_p = (
            float(
                numeric.loc[
                    np.asarray(group.index)[group_formal], "feature_global_maxT_p"
                ].min()
            )
            if family_size
            else 1.0
        )
        if not np.isclose(
            float(group["assay_global_maxT_p"].iloc[0]),
            expected_assay_p,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "Assay maxT p-value is not its strongest eligible feature maxT p-value"
            )
        participation_order = min(
            int(FROZEN_DISCOVERY_CONFIG["assay_state_participation_order"]),
            len(group),
        )
        ranked_eligible = frame.loc[
            np.asarray(group.index)[group_formal]
        ].copy()
        ranked_eligible["_empirical_null_p"] = numeric.loc[
            ranked_eligible.index, "empirical_null_p"
        ]
        ranked_eligible["_reference_conditional_wald"] = numeric.loc[
            ranked_eligible.index, "reference_conditional_wald"
        ]
        ranked_eligible = ranked_eligible.sort_values(
            ["_empirical_null_p", "_reference_conditional_wald", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        leading_reporter = (
            str(ranked_eligible.iloc[0]["feature"])
            if family_size
            else ""
        )
        if family_size:
            leading_index = int(ranked_eligible.index[0])
            single_feature_order_p = float(
                numeric.loc[leading_index, "empirical_null_p"]
            )
            single_feature_p = min(
                1.0, family_size * single_feature_order_p
            )
            single_feature_witness_views = int(view_count[leading_index])
        else:
            single_feature_order_p = 1.0
            single_feature_p = 1.0
            single_feature_witness_views = 0
        single_feature_replication = bool(
            family_size >= 1
            and single_feature_witness_views
            >= int(
                FROZEN_DISCOVERY_CONFIG[
                    "minimum_assay_state_participation_views"
                ]
            )
        )
        enough_witnesses = family_size >= participation_order > 0
        if enough_witnesses:
            witness_indices = list(ranked_eligible.index[:participation_order])
            order_p = float(
                numeric.loc[witness_indices[-1], "empirical_null_p"]
            )
            partial_conjunction_p = min(
                1.0,
                (family_size - participation_order + 1) * order_p,
            )
            minimum_witness_views = int(
                np.min(view_count[np.asarray(witness_indices, dtype=int)])
            )
            contributing_count = participation_order
        else:
            partial_conjunction_p = 1.0
            minimum_witness_views = 0
            contributing_count = 0
        partial_conjunction_replication = bool(
            enough_witnesses
            and minimum_witness_views
            >= int(
                FROZEN_DISCOVERY_CONFIG[
                    "minimum_assay_state_participation_views"
                ]
            )
        )
        omnibus_route_count = 1 if len(group) == 1 else 2
        live_routes = {
            "single_feature": (
                single_feature_p if single_feature_replication else 1.0
            )
        }
        if omnibus_route_count == 2:
            live_routes["partial_conjunction"] = (
                partial_conjunction_p
                if partial_conjunction_replication
                else 1.0
            )
        if any(value < 1.0 for value in live_routes.values()):
            winning_route = min(
                live_routes,
                key=lambda route: (
                    live_routes[route],
                    0 if route == "partial_conjunction" else 1,
                ),
            )
        else:
            winning_route = "none"
        omnibus_p = min(
            1.0, omnibus_route_count * min(live_routes.values())
        )
        constant_columns = (
            "assay_participation_partial_conjunction_p",
            "assay_participation_holm_p",
            "assay_participation_single_feature_order_statistic_p",
            "assay_participation_single_feature_p",
            "assay_participation_single_feature_witness_view_count",
            "assay_participation_single_feature_multiview_replication_gate",
            "assay_participation_partial_conjunction_multiview_replication_gate",
            "assay_participation_omnibus_route_count",
            "assay_participation_omnibus_p",
            "assay_participation_omnibus_holm_p",
            "assay_participation_winning_route",
            "assay_participation_selection_route",
            "assay_participation_selected",
            "assay_participation_order",
            "assay_participation_family_size",
            "assay_participation_contributing_feature_count",
            "assay_participation_minimum_witness_view_count",
            "assay_participation_multiview_replication_gate",
        )
        if any(group[column].nunique(dropna=False) != 1 for column in constant_columns):
            raise ValueError(
                f"Assay-participation annotations vary within {modality}"
            )
        if not np.isclose(
            float(group["assay_participation_partial_conjunction_p"].iloc[0]),
            partial_conjunction_p,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"Assay partial-conjunction p-value is inconsistent for {modality}"
            )
        for column, expected_value in (
            (
                "assay_participation_single_feature_order_statistic_p",
                single_feature_order_p,
            ),
            ("assay_participation_single_feature_p", single_feature_p),
            ("assay_participation_omnibus_p", omnibus_p),
        ):
            if not np.isclose(
                float(group[column].iloc[0]),
                expected_value,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"Assay omnibus evidence is inconsistent for {modality}: {column}"
                )
        if (
            int(group["assay_participation_order"].iloc[0])
            != participation_order
            or int(group["assay_participation_family_size"].iloc[0])
            != family_size
            or int(
                group[
                    "assay_participation_contributing_feature_count"
                ].iloc[0]
            )
            != contributing_count
            or int(
                group[
                    "assay_participation_minimum_witness_view_count"
                ].iloc[0]
            )
            != minimum_witness_views
            or int(
                group[
                    "assay_participation_single_feature_witness_view_count"
                ].iloc[0]
            )
            != single_feature_witness_views
            or int(
                group["assay_participation_omnibus_route_count"].iloc[0]
            )
            != omnibus_route_count
        ):
            raise ValueError(
                f"Assay partial-conjunction counts are inconsistent for {modality}"
            )
        if bool(
            group["assay_participation_multiview_replication_gate"].iloc[0]
        ) != partial_conjunction_replication or bool(
            group[
                "assay_participation_partial_conjunction_multiview_"
                "replication_gate"
            ].iloc[0]
        ) != partial_conjunction_replication or bool(
            group[
                "assay_participation_single_feature_multiview_replication_gate"
            ].iloc[0]
        ) != single_feature_replication:
            raise ValueError(
                f"Assay participation branch replication is inconsistent for {modality}"
            )
        if str(group["assay_participation_winning_route"].iloc[0]) != (
            winning_route
        ):
            raise ValueError(
                f"Assay participation winning route is inconsistent for {modality}"
            )
        participation_expectations[str(modality)] = {
            "indices": list(group.index),
            "partial_conjunction_p": partial_conjunction_p,
            "single_feature_order_p": single_feature_order_p,
            "single_feature_p": single_feature_p,
            "single_feature_witness_views": single_feature_witness_views,
            "single_feature_replication": single_feature_replication,
            "partial_conjunction_replication": (
                partial_conjunction_replication
            ),
            "omnibus_route_count": omnibus_route_count,
            "omnibus_p": omnibus_p,
            "winning_route": winning_route,
            "leading_reporter": leading_reporter,
        }
    participation_modalities = sorted(participation_expectations)
    expected_partial_conjunction_holm = _holm(
        [
            participation_expectations[modality]["partial_conjunction_p"]
            for modality in participation_modalities
        ]
    )
    expected_omnibus_holm = _holm(
        [
            participation_expectations[modality]["omnibus_p"]
            for modality in participation_modalities
        ]
    )
    for modality, partial_holm_p, omnibus_holm_p in zip(
        participation_modalities,
        expected_partial_conjunction_holm,
        expected_omnibus_holm,
        strict=True,
    ):
        expectation = participation_expectations[modality]
        indices = expectation["indices"]
        if not np.allclose(
            numeric.loc[indices, "assay_participation_holm_p"],
            partial_holm_p,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"Assay participation Holm p-value is inconsistent for {modality}"
            )
        if not np.allclose(
            numeric.loc[indices, "assay_participation_omnibus_holm_p"],
            omnibus_holm_p,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"Assay omnibus Holm p-value is inconsistent for {modality}"
            )
        expected_participation = bool(
            omnibus_holm_p
            <= float(
                FROZEN_DISCOVERY_CONFIG[
                    "maximum_assay_state_participation_error_probability"
                ]
            )
            and expectation["winning_route"] != "none"
        )
        observed_participation = frame.loc[
            indices, "assay_participation_selected"
        ]
        if not (observed_participation == expected_participation).all():
            raise ValueError(
                f"Assay participation selection is inconsistent for {modality}"
            )
        expected_selection_route = (
            expectation["winning_route"] if expected_participation else ""
        )
        if not (
            frame.loc[indices, "assay_participation_selection_route"]
            .astype(str)
            .eq(expected_selection_route)
            .all()
        ):
            raise ValueError(
                f"Assay participation selection route is inconsistent for {modality}"
            )
        reporter_indices = frame.index[
            frame.index.isin(indices) & frame["assay_participation_reporter"]
        ].tolist()
        expected_reporter = (
            expectation["leading_reporter"] if expected_participation else ""
        )
        observed_reporters = list(frame.loc[reporter_indices, "feature"].astype(str))
        if observed_reporters != ([expected_reporter] if expected_reporter else []):
            raise ValueError(
                f"Assay participation reporter is inconsistent for {modality}"
            )
    expected_delta = (
        numeric["reference_conditional_wald"]
        - numeric["conditional_wald_null_95pct"]
    )
    if not np.allclose(
        numeric["conditional_wald_above_null_95pct"],
        expected_delta,
        atol=1e-10,
        rtol=1e-8,
    ):
        raise ValueError(
            "conditional_wald_above_null_95pct is inconsistent with its evidence"
        )

    selection_gate = (
        (numeric["empirical_null_fdr"] <= FROZEN_DISCOVERY_CONFIG["maximum_empirical_fdr"])
        & frame["assay_global_maxT_admission_gate"]
        & frame["feature_global_maxT_confirmation_gate"]
        & frame["feature_direction_stability_confirmation_gate"]
        & (view_count >= FROZEN_DISCOVERY_CONFIG["minimum_conditional_wald_views"])
        & frame["formal_test_eligible"]
    )
    expected_pre_status = np.where(selection_gate, "selected", "unresolved")
    expected_pre_reasons: list[str] = []
    for index in frame.index:
        if not bool(frame.loc[index, "formal_test_eligible"]):
            expected_pre_reasons.append(
                str(frame.loc[index, "formal_test_abstention_reason"])
            )
            continue
        failures: list[str] = []
        if numeric.loc[index, "empirical_null_fdr"] > FROZEN_DISCOVERY_CONFIG[
            "maximum_empirical_fdr"
        ]:
            failures.append("empirical_null_fdr")
        if not bool(frame.loc[index, "assay_global_maxT_admission_gate"]):
            failures.append("assay_global_maxT_admission")
        if not bool(frame.loc[index, "feature_global_maxT_confirmation_gate"]):
            failures.append("feature_global_maxT_confirmation")
        if not bool(
            frame.loc[index, "feature_direction_stability_confirmation_gate"]
        ):
            failures.append("feature_direction_stability_confirmation")
        if int(view_count[index]) < FROZEN_DISCOVERY_CONFIG[
            "minimum_conditional_wald_views"
        ]:
            failures.append("conditional_wald_view_support")
        expected_pre_reasons.append(";".join(failures))
    if not np.array_equal(
        frame["pre_redundancy_selection_status"].to_numpy(),
        expected_pre_status,
    ):
        raise ValueError(
            "pre_redundancy_selection_status disagrees with calibrated evidence"
        )
    if list(
        frame["pre_redundancy_abstention_reason"].fillna("").astype(str)
    ) != expected_pre_reasons:
        raise ValueError(
            "pre_redundancy_abstention_reason disagrees with calibrated evidence"
        )
    if not np.array_equal(
        frame["redundancy_direction_stability_gate"].to_numpy(dtype=bool),
        expected_direction_diagnostic.to_numpy(dtype=bool),
    ):
        raise ValueError(
            "redundancy direction-stability gate disagrees with its statistic"
        )
    expected_status = np.asarray(expected_pre_status, dtype=object).copy()
    expected_reasons = list(expected_pre_reasons)
    expected_fallback = np.zeros(len(frame), dtype=bool)
    expected_demoted = np.zeros(len(frame), dtype=bool)
    preliminary_indices = frame.index[expected_pre_status == "selected"]
    for _, group in frame.loc[preliminary_indices].groupby(
        "modality", sort=True
    ):
        group_indices = list(group.index)
        stable_indices = [
            index
            for index in group_indices
            if bool(expected_direction_diagnostic.loc[index])
        ]
        if stable_indices:
            retained = set(stable_indices)
        else:
            ranked = frame.loc[group_indices].copy()
            ranked["_feature_global_maxT_p"] = numeric.loc[
                group_indices, "feature_global_maxT_p"
            ]
            ranked["_reference_conditional_wald"] = numeric.loc[
                group_indices, "reference_conditional_wald"
            ]
            ranked = ranked.sort_values(
                [
                    "_feature_global_maxT_p",
                    "_reference_conditional_wald",
                    "feature",
                ],
                ascending=[True, False, True],
                kind="mergesort",
            )
            fallback_index = int(ranked.index[0])
            retained = {fallback_index}
            expected_fallback[fallback_index] = True
        for index in group_indices:
            if index in retained:
                continue
            expected_status[index] = "unresolved"
            expected_reasons[index] = (
                "within_assay_redundancy_direction_instability"
            )
            expected_demoted[index] = True
    post_redundancy_selected = expected_status == "selected"
    expected_strong_core_member = (
        post_redundancy_selected
        & numeric["feature_global_maxT_p"]
        .le(FROZEN_DISCOVERY_CONFIG["within_assay_strong_core_maxT_p"])
        .to_numpy(dtype=bool)
    )
    expected_strong_core_count = np.zeros(len(frame), dtype=int)
    for _, group in frame.groupby("modality", sort=True):
        group_indices = np.asarray(list(group.index), dtype=int)
        expected_strong_core_count[group_indices] = int(
            expected_strong_core_member[group_indices].sum()
        )
    expected_weak_tail_demoted = (
        post_redundancy_selected
        & (
            expected_strong_core_count
            >= FROZEN_DISCOVERY_CONFIG[
                "within_assay_strong_core_minimum_members"
            ]
        )
        & numeric["feature_global_maxT_p"]
        .gt(FROZEN_DISCOVERY_CONFIG["within_assay_strong_core_maxT_p"])
        .to_numpy(dtype=bool)
    )
    for index in np.flatnonzero(expected_weak_tail_demoted):
        expected_status[index] = "unresolved"
        expected_reasons[index] = "within_assay_strong_maxT_core_weak_tail"
    if not np.array_equal(
        frame["within_assay_strong_core_member"].to_numpy(dtype=bool),
        expected_strong_core_member,
    ):
        raise ValueError("Within-assay strong-core membership flags are inconsistent")
    if not np.array_equal(
        numeric["within_assay_strong_core_count"].to_numpy(dtype=int),
        expected_strong_core_count,
    ):
        raise ValueError("Within-assay strong-core counts are inconsistent")
    if not np.array_equal(
        frame["within_assay_weak_maxT_tail_demoted"].to_numpy(dtype=bool),
        expected_weak_tail_demoted,
    ):
        raise ValueError("Within-assay weak-tail demotion flags are inconsistent")
    if not np.array_equal(frame["selection_status"].to_numpy(), expected_status):
        raise ValueError(
            "selection_status disagrees with calibrated redundancy/strong-core evidence"
        )
    if list(frame["abstention_reason"].fillna("").astype(str)) != expected_reasons:
        raise ValueError(
            "abstention_reason disagrees with calibrated redundancy/strong-core evidence"
        )
    if not np.array_equal(
        frame["redundancy_fallback_selected"].to_numpy(dtype=bool),
        expected_fallback,
    ) or not np.array_equal(
        frame["redundancy_demoted"].to_numpy(dtype=bool), expected_demoted
    ):
        raise ValueError("Redundancy fallback/demotion flags are inconsistent")
    expected_ids = np.where(
        frame["selection_status"].eq("selected"), SUBSPACE_ID, ""
    )
    if not np.array_equal(
        frame["subspace_id"].fillna("").astype(str).to_numpy(), expected_ids
    ):
        raise ValueError("Membership subspace_id disagrees with selection_status")
    if frame["direct_layer_annotation_available"].nunique() != 1:
        raise ValueError("Direct-layer annotation availability must be table-global")
    if not bool(frame["direct_layer_annotation_available"].iloc[0]) and frame[
        "direct_layer_overlap"
    ].any():
        raise ValueError("Unavailable direct-layer evidence cannot annotate overlap")
    frame[numeric_columns] = numeric
    return frame


def _validate_assay_participation_provenance(
    membership: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind repeated assay annotations to the full pretruth provenance table."""

    annotation_columns = [
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p",
        "assay_participation_single_feature_p",
        "assay_participation_single_feature_witness_view_count",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_omnibus_route_count",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "assay_participation_winning_route",
        "assay_participation_selection_route",
        "assay_participation_selected",
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_multiview_replication_gate",
        "assay_participation_reporter",
    ]
    missing = sorted(
        {"feature", "modality", "formal_test_eligible", *annotation_columns}
        - set(membership.columns)
    )
    if missing:
        raise ValueError(
            f"Membership lacks assay-participation annotations: {missing}"
        )
    section = provenance.get("assay_state_participation")
    if not isinstance(section, Mapping):
        raise ValueError("Subspace provenance lacks assay-state participation")
    expected_contract = {
        "edge_type": "assay_shared_state_participation",
        "feature_empirical_p_calibration": (
            "existing_state_depth_orthogonal_synchronized_conditional_wild_null"
        ),
        "within_assay_single_feature_test": (
            "bonferroni_minP_valid_under_arbitrary_feature_dependence"
        ),
        "within_assay_partial_conjunction_test": (
            "bonferroni_partial_conjunction_valid_under_arbitrary_"
            "feature_dependence"
        ),
        "assay_participation_null": (
            "no_state_responsive_features_in_the_assay"
        ),
        "partial_conjunction_null": (
            "fewer_than_r_state_responsive_features_in_the_assay"
        ),
        "within_assay_omnibus_combination": (
            "bonferroni_over_predeclared_replication_gated_routes"
        ),
        "omnibus_route_count_policy": (
            "one_for_structural_singleton_otherwise_two"
        ),
        "failed_replication_route_p_value": 1.0,
        "across_assay_error_control": (
            "holm_strong_FWER_without_cross_assay_dependence_assumption"
        ),
        "assay_multiplicity_family": "all_observed_assays",
        "selection_rule": (
            "assay_participation_omnibus_holm_p_at_most_configured_"
            "error_probability"
        ),
        "complete_null_order_statistic_minP_used": True,
        "minimum_witness_views": FROZEN_DISCOVERY_CONFIG[
            "minimum_assay_state_participation_views"
        ],
        "affects_feature_membership": False,
        "affects_existing_typed_subspace_hyperedge": False,
        "reporter_or_witness_implies_feature_membership": False,
    }
    expected_section_keys = {
        *expected_contract,
        "n_selected_assays",
        "selected_modalities",
        "membership_annotation_columns",
        "records",
    }
    if set(section) != expected_section_keys:
        raise ValueError(
            "Assay-participation provenance fields are missing or extra"
        )
    for key, expected in expected_contract.items():
        if section.get(key) != expected:
            raise ValueError(
                f"Assay-participation provenance contract mismatch: {key}"
            )
    if list(section.get("membership_annotation_columns", [])) != annotation_columns:
        raise ValueError("Assay-participation annotation column contract changed")
    records = section.get("records")
    if not isinstance(records, list):
        raise ValueError("Assay-participation provenance records must be a list")
    expected_record_keys = {
        "assay_state_id",
        "modality",
        "edge_type",
        "n_observed_features",
        "n_formally_eligible_features",
        "partial_conjunction_order",
        "partial_conjunction_order_statistic_p",
        "partial_conjunction_empirical_p",
        "assay_participation_holm_p",
        "single_feature_order_statistic_p",
        "single_feature_bonferroni_p",
        "single_feature_witness_feature",
        "single_feature_witness_view_count",
        "single_feature_multiview_replication_gate",
        "minimum_witness_view_count",
        "multiview_replication_gate",
        "partial_conjunction_multiview_replication_gate",
        "assay_participation_omnibus_route_count",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "assay_participation_winning_route",
        "assay_participation_selection_route",
        "leading_reporter",
        "selected_reporter",
        "contributing_feature_count",
        "partial_conjunction_witness_features",
        "all_eligible_feature_evidence",
        "selection_status",
        "abstention_reason",
        "conditional_on_assay_local_state_model",
        "feature_membership_claimed",
        "exact_axis_pairing_claimed",
        "direct_causal_claimed",
    }
    record_map: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Assay-participation provenance record is not an object")
        if set(record) != expected_record_keys:
            raise ValueError(
                "Assay-participation provenance record fields are missing or extra"
            )
        modality = str(record.get("modality", ""))
        if not modality or modality in record_map:
            raise ValueError("Assay-participation provenance modalities must be unique")
        record_map[modality] = record
    observed_modalities = sorted(set(membership["modality"].astype(str)))
    if set(record_map) != set(observed_modalities):
        raise ValueError("Assay-participation provenance modality set changed")
    selected_modalities: list[str] = []
    for modality in observed_modalities:
        group = membership.loc[
            membership["modality"].astype(str).eq(modality)
        ].copy()
        record = record_map[modality]
        constant_columns = annotation_columns[:-1]
        if any(group[column].nunique(dropna=False) != 1 for column in constant_columns):
            raise ValueError(
                f"Repeated assay-participation bytes vary within {modality}"
            )
        first = group.iloc[0]
        selected = _parse_bool(
            first["assay_participation_selected"],
            "assay_participation_selected",
        )
        replicated = _parse_bool(
            first["assay_participation_multiview_replication_gate"],
            "assay_participation_multiview_replication_gate",
        )
        single_replicated = _parse_bool(
            first[
                "assay_participation_single_feature_multiview_replication_gate"
            ],
            "assay_participation_single_feature_multiview_replication_gate",
        )
        partial_replicated = _parse_bool(
            first[
                "assay_participation_partial_conjunction_multiview_"
                "replication_gate"
            ],
            "assay_participation_partial_conjunction_multiview_replication_gate",
        )
        reporters = [
            str(row.feature)
            for row in group.itertuples(index=False)
            if _parse_bool(
                getattr(row, "assay_participation_reporter"),
                "assay_participation_reporter",
            )
        ]
        expected_reporter = str(record.get("selected_reporter", ""))
        if reporters != ([expected_reporter] if expected_reporter else []):
            raise ValueError(
                f"Assay-participation reporter bytes/provenance differ for {modality}"
            )
        formal_mask = group["formal_test_eligible"].map(
            lambda value: _parse_bool(value, "formal_test_eligible")
        )
        ranked = group.loc[formal_mask].copy()
        ranked["_empirical_null_p"] = pd.to_numeric(
            ranked["empirical_null_p"], errors="raise"
        )
        ranked["_reference_conditional_wald"] = pd.to_numeric(
            ranked["reference_conditional_wald"], errors="raise"
        )
        ranked = ranked.sort_values(
            ["_empirical_null_p", "_reference_conditional_wald", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        family_size = len(ranked)
        order = min(
            int(FROZEN_DISCOVERY_CONFIG["assay_state_participation_order"]),
            len(group),
        )
        enough = family_size >= order > 0
        witnesses = ranked.head(order) if enough else ranked.iloc[0:0]
        leading = str(ranked.iloc[0]["feature"]) if family_size else ""
        order_statistic_p = (
            float(witnesses.iloc[-1]["_empirical_null_p"])
            if enough
            else 1.0
        )
        witness_features = ";".join(witnesses["feature"].astype(str))
        evidence_records = [
            {
                "feature": str(row["feature"]),
                "empirical_null_p": float(row["_empirical_null_p"]),
                "reference_conditional_wald": float(
                    row["_reference_conditional_wald"]
                ),
                "conditional_wald_view_support": float(
                    row["conditional_wald_view_support"]
                ),
                "single_feature_witness": bool(index == 0),
                "partial_conjunction_witness": bool(enough and index < order),
            }
            for index, row in enumerate(ranked.to_dict(orient="records"))
        ]
        expected_abstention: list[str] = []
        if family_size < 1:
            expected_abstention.append("insufficient_formally_eligible_features")
        if str(first["assay_participation_winning_route"]) == "none":
            expected_abstention.append("no_replicated_omnibus_route")
        if float(first["assay_participation_omnibus_holm_p"]) > float(
            FROZEN_DISCOVERY_CONFIG[
                "maximum_assay_state_participation_error_probability"
            ]
        ):
            expected_abstention.append("omnibus_holm_FWER")
        expected_values = {
            "partial_conjunction_empirical_p": float(
                first["assay_participation_partial_conjunction_p"]
            ),
            "assay_participation_holm_p": float(
                first["assay_participation_holm_p"]
            ),
            "single_feature_order_statistic_p": float(
                first[
                    "assay_participation_single_feature_order_statistic_p"
                ]
            ),
            "single_feature_bonferroni_p": float(
                first["assay_participation_single_feature_p"]
            ),
            "single_feature_witness_view_count": int(
                first[
                    "assay_participation_single_feature_witness_view_count"
                ]
            ),
            "single_feature_multiview_replication_gate": single_replicated,
            "partial_conjunction_multiview_replication_gate": (
                partial_replicated
            ),
            "assay_participation_omnibus_route_count": int(
                first["assay_participation_omnibus_route_count"]
            ),
            "assay_participation_omnibus_p": float(
                first["assay_participation_omnibus_p"]
            ),
            "assay_participation_omnibus_holm_p": float(
                first["assay_participation_omnibus_holm_p"]
            ),
            "assay_participation_winning_route": str(
                first["assay_participation_winning_route"]
            ),
            "assay_participation_selection_route": str(
                first["assay_participation_selection_route"]
            ),
            "partial_conjunction_order": int(
                first["assay_participation_order"]
            ),
            "n_formally_eligible_features": int(
                first["assay_participation_family_size"]
            ),
            "contributing_feature_count": int(
                first["assay_participation_contributing_feature_count"]
            ),
            "minimum_witness_view_count": int(
                first["assay_participation_minimum_witness_view_count"]
            ),
            "multiview_replication_gate": replicated,
            "selection_status": "selected" if selected else "unresolved",
            "assay_state_id": f"assay_state_participation__{modality}",
            "modality": modality,
            "edge_type": "assay_shared_state_participation",
            "n_observed_features": len(group),
            "partial_conjunction_order_statistic_p": order_statistic_p,
            "single_feature_witness_feature": leading,
            "leading_reporter": leading,
            "partial_conjunction_witness_features": witness_features,
            "all_eligible_feature_evidence": json.dumps(
                evidence_records, sort_keys=True, separators=(",", ":")
            ),
            "abstention_reason": ";".join(expected_abstention),
            "conditional_on_assay_local_state_model": True,
            "feature_membership_claimed": False,
            "exact_axis_pairing_claimed": False,
            "direct_causal_claimed": False,
        }
        for key, expected in expected_values.items():
            observed = record.get(key)
            if isinstance(expected, float):
                try:
                    matches = np.isclose(
                        float(observed), expected, atol=1e-12, rtol=1e-10
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = observed == expected
            if not matches:
                raise ValueError(
                    "Assay-participation membership/provenance mismatch for "
                    f"{modality}: {key}"
                )
        if selected:
            selected_modalities.append(modality)
    if int(section.get("n_selected_assays", -1)) != len(selected_modalities) or list(
        section.get("selected_modalities", [])
    ) != sorted(selected_modalities):
        raise ValueError("Assay-participation selected-modality summary changed")
    return {
        "selected_modalities": sorted(selected_modalities),
        "n_selected_assays": len(selected_modalities),
        "membership_annotation_columns": annotation_columns,
    }


def _score_hyperedge(
    hyperedges: pd.DataFrame, membership: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = sorted(HYPEREDGE_REQUIRED_COLUMNS - set(hyperedges.columns))
    if missing:
        raise ValueError(f"Typed subspace hyperedges are missing columns {missing}")
    extra = sorted(set(hyperedges.columns) - HYPEREDGE_REQUIRED_COLUMNS)
    if extra:
        raise ValueError(f"Typed subspace hyperedges have unregistered columns {extra}")
    if tuple(hyperedges.columns) != HYPEREDGE_COLUMNS:
        raise ValueError("Typed subspace hyperedge columns are not in frozen v4 order")
    if len(hyperedges) != 1:
        raise ValueError("The dimension-agnostic contract emits exactly one subspace")
    record = hyperedges.iloc[0].to_dict()
    selected = membership.loc[membership["selection_status"] == "selected"]
    expected_members = tuple(sorted(selected["feature"].astype(str)))
    expected_modalities = tuple(sorted(set(selected["modality"].astype(str))))
    observed_members = tuple(sorted(_split(record["member_features"])))
    observed_modalities = tuple(sorted(_split(record["modalities"])))
    try:
        n_features = int(record["n_features"])
        n_modalities = int(record["n_modalities"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Hyperedge counts must be integers") from exc
    status = str(record["selection_status"]).strip().casefold()
    if status not in {"selected", "unresolved"}:
        raise ValueError("Hyperedge selection status must be selected or unresolved")
    try:
        multimodality_p = float(record["empirical_multimodality_p"])
    except (TypeError, ValueError) as exc:
        raise ValueError("empirical_multimodality_p must be numeric") from exc
    if not np.isfinite(multimodality_p) or not 0.0 <= multimodality_p <= 1.0:
        raise ValueError("empirical_multimodality_p must lie in [0, 1]")
    expected_status = (
        "selected"
        if len(expected_modalities) >= 2
        and multimodality_p <= MAX_HYPEREDGE_EMPIRICAL_P
        else "unresolved"
    )
    expected_abstention_reason = (
        ""
        if expected_status == "selected"
        else (
            "insufficient_modalities"
            if len(expected_modalities) < 2
            else "multimodality_maxT"
        )
    )
    # Pandas' default TSV reader represents a terminal empty field as NaN.
    # The writer deliberately emits an empty abstention reason for a selected
    # hyperedge, so normalize only a genuinely missing scalar back to that
    # canonical empty representation before checking the frozen contract.
    raw_abstention_reason = record["abstention_reason"]
    observed_abstention_reason = (
        "" if pd.isna(raw_abstention_reason) else str(raw_abstention_reason).strip()
    )
    claims = {
        key: _parse_bool(record[key], key)
        for key in (
            "direction_claimed",
            "exact_axis_pairing_claimed",
            "direct_causal_claimed",
        )
    }
    conditional_on_model = _parse_bool(
        record["conditional_on_common_state_model"],
        "conditional_on_common_state_model",
    )
    causal_interpretation = str(record["causal_interpretation"]).strip().casefold()
    checks = {
        "subspace_id_correct": int(str(record["subspace_id"]) == SUBSPACE_ID),
        "edge_type_correct": int(str(record["edge_type"]) == SUBSPACE_EDGE_TYPE),
        "member_set_exact": int(observed_members == expected_members),
        "modality_set_exact": int(observed_modalities == expected_modalities),
        "feature_count_exact": int(n_features == len(expected_members)),
        "modality_count_exact": int(n_modalities == len(expected_modalities)),
        "selection_status_coherent": int(status == expected_status),
        "abstention_reason_coherent": int(
            observed_abstention_reason == expected_abstention_reason
        ),
        "hyperedge_selected": int(status == "selected"),
        "conditional_common_state_interpretation": int(conditional_on_model),
        "hypothesis_only_interpretation": int(
            causal_interpretation == "hypothesis_only"
        ),
        "empirical_multimodality_gate": int(
            multimodality_p <= MAX_HYPEREDGE_EMPIRICAL_P
        ),
        "error_control_method_correct": int(
            str(record["error_control_method"]) == EXPECTED_ERROR_CONTROL_METHOD
        ),
        "membership_subspace_ids_coherent": int(
            bool(
                (
                    membership.loc[
                        membership["selection_status"] == "selected", "subspace_id"
                    ].astype(str)
                    == SUBSPACE_ID
                ).all()
            )
            and bool(
                (
                    membership.loc[
                        membership["selection_status"] != "selected", "subspace_id"
                    ].fillna("").astype(str)
                    == ""
                ).all()
            )
        ),
        "n_direction_claims": int(claims["direction_claimed"]),
        "n_exact_axis_pairing_claims": int(
            claims["exact_axis_pairing_claimed"]
        ),
        "n_direct_causal_claims": int(claims["direct_causal_claimed"]),
    }
    checks["n_unsupported_claim_flags"] = int(
        checks["n_direction_claims"]
        + checks["n_exact_axis_pairing_claims"]
        + checks["n_direct_causal_claims"]
    )
    checks["hyperedge_schema_consistent"] = int(
        all(
            checks[key] == 1
            for key in (
                "subspace_id_correct",
                "edge_type_correct",
                "member_set_exact",
                "modality_set_exact",
                "feature_count_exact",
                "modality_count_exact",
                "selection_status_coherent",
                "abstention_reason_coherent",
                "membership_subspace_ids_coherent",
                "conditional_common_state_interpretation",
                "hypothesis_only_interpretation",
                "error_control_method_correct",
            )
        )
        and checks["n_unsupported_claim_flags"] == 0
    )
    row = {
        **record,
        **checks,
        "expected_member_features": ";".join(expected_members),
        "expected_modalities": ";".join(expected_modalities),
        "expected_selection_status": expected_status,
        "expected_abstention_reason": expected_abstention_reason,
    }
    return pd.DataFrame([row]), checks


def _observable_direct_child_set(typed_graph: pd.DataFrame) -> set[str]:
    """Return arrowhead endpoints of typed within-assay direct candidates."""

    required = {"node1", "endpoint1", "node2", "endpoint2", "edge_type"}
    if not required <= set(typed_graph.columns):
        return set()
    children: set[str] = set()
    direct = typed_graph.loc[
        typed_graph["edge_type"].astype(str) == "within_assay_direct_candidate"
    ]
    for record in direct.to_dict(orient="records"):
        endpoint1 = str(record["endpoint1"])
        endpoint2 = str(record["endpoint2"])
        if endpoint1 == "ARROW" and endpoint2 != "ARROW":
            children.add(str(record["node1"]))
        if endpoint2 == "ARROW" and endpoint1 != "ARROW":
            children.add(str(record["node2"]))
    return children


def score_shared_state_subspace(
    *,
    world_parameters: Mapping[str, Any],
    role_assignments: pd.DataFrame,
    feature_membership: pd.DataFrame,
    typed_subspace_hyperedges: pd.DataFrame,
    typed_graph: pd.DataFrame | None = None,
    truth_typed_edges: pd.DataFrame | None = None,
) -> SharedStateSubspaceScore:
    """Score one frozen dimension-agnostic shared-state-subspace estimate."""

    if (typed_graph is None) != (truth_typed_edges is None):
        raise ValueError("typed_graph and truth_typed_edges must be supplied together")
    world_seed = int(world_parameters["seed"])
    expected_design = _doe_design_for_seed(world_seed)
    observed_design = world_parameters.get("design")
    if observed_design is not None:
        if not isinstance(observed_design, Mapping):
            raise ValueError("World DOE design must be a mapping")
        for key, expected_value in expected_design.items():
            if int(observed_design.get(key, 0)) != expected_value:
                raise ValueError(f"World DOE design disagrees with seed for {key}")
    truth = build_subspace_truth(world_parameters, role_assignments)
    membership = _validate_membership_table(feature_membership, truth)
    hyperedge_scores, hyperedge_checks = _score_hyperedge(
        typed_subspace_hyperedges, membership
    )

    feature_scores = truth.features.merge(
        membership, on=["feature", "modality", "channel"], how="left", validate="1:1"
    )
    feature_scores["selected"] = (
        feature_scores["selection_status"] == "selected"
    ).astype(int)
    feature_scores["selection_correct"] = (
        feature_scores["selected"]
        == feature_scores["expected_subspace_member"].astype(int)
    ).astype(int)
    selected = feature_scores.loc[feature_scores["selected"] == 1]
    eligible = feature_scores.loc[feature_scores["expected_subspace_member"]]
    chromatin_eligible = eligible.loc[eligible["modality"] != "rna"]
    selected_true = selected.loc[selected["expected_subspace_member"]]
    selected_chromatin = selected.loc[selected["modality"] != "rna"]
    selected_chromatin_true = selected.loc[
        (selected["modality"] != "rna")
        & selected["expected_subspace_member"]
    ]
    sentinel = feature_scores.loc[feature_scores["truth_class"] == "sentinel_null"]
    direct_overlap = feature_scores.loc[feature_scores["role"] == "C"]
    gene = feature_scores.loc[feature_scores["truth_class"] == "state_responsive_gene"]

    assay_rows: list[dict[str, Any]] = []
    for group in truth.assay_groups.to_dict(orient="records"):
        expected_members = set(_split(group["expected_members"]))
        assay_frame = feature_scores.loc[
            feature_scores["modality"] == str(group["modality"])
        ]
        observed_members = set(
            assay_frame.loc[
                (assay_frame["selected"] == 1)
                & assay_frame["expected_subspace_member"],
                "feature",
            ].astype(str)
        )
        false_members = set(
            assay_frame.loc[
                (assay_frame["selected"] == 1)
                & ~assay_frame["expected_subspace_member"],
                "feature",
            ].astype(str)
        )
        assay_rows.append(
            {
                **group,
                "selected_expected_members": ";".join(sorted(observed_members)),
                "selected_ineligible_members": ";".join(sorted(false_members)),
                "n_selected_expected_members": len(observed_members),
                "n_selected_ineligible_members": len(false_members),
                "member_recall": _safe_rate(len(observed_members), len(expected_members)),
                "assay_covered": int(bool(observed_members)),
                "complete_assay_group": int(observed_members == expected_members),
            }
        )
    assay_scores = pd.DataFrame(assay_rows)

    role_recalls = {
        role: _safe_rate(
            int(
                (
                    (feature_scores["role"] == role)
                    & (feature_scores["selected"] == 1)
                ).sum()
            ),
            int((feature_scores["role"] == role).sum()),
        )
        for role in STATE_ROLES
    }
    member_precision = _safe_rate(len(selected_true), len(selected))
    chromatin_member_precision = _safe_rate(
        len(selected_chromatin_true), len(selected_chromatin)
    )
    total_recall = _safe_rate(len(selected_true), len(eligible))
    chromatin_recall = _safe_rate(
        len(selected_chromatin_true), len(chromatin_eligible)
    )
    sentinel_rate = _safe_rate(int(sentinel["selected"].sum()), len(sentinel))
    assay_coverage = float(assay_scores["assay_covered"].mean())
    chromatin_assays = assay_scores.loc[assay_scores["modality"] != "rna"]
    chromatin_assay_coverage = float(chromatin_assays["assay_covered"].mean())
    minimum_chromatin_assay_member_recall = float(
        chromatin_assays["member_recall"].min()
    )
    gene_selected = int(gene["selected"].iloc[0])

    membership_success = int(
        member_precision >= MIN_MEMBER_PRECISION
        and chromatin_recall >= MIN_CHROMATIN_MEMBER_RECALL
        and min(role_recalls[role] for role in CORE_STATE_ROLES) >= MIN_ROLE_RECALL
        and assay_coverage >= MIN_ASSAY_COVERAGE
        and minimum_chromatin_assay_member_recall
        >= MIN_EACH_CHROMATIN_ASSAY_MEMBER_RECALL
        and sentinel_rate <= MAX_SENTINEL_INCLUSION_RATE
        and role_recalls["C"] >= MIN_C_STATE_RESPONSIVE_RECALL
        and gene_selected == 1
    )
    hyperedge_success = int(
        hyperedge_checks["hyperedge_schema_consistent"] == 1
        and hyperedge_checks["hyperedge_selected"] == 1
        and hyperedge_checks["empirical_multimodality_gate"] == 1
    )
    subspace_success = int(membership_success and hyperedge_success)

    local_metrics: dict[str, Any] = {}
    local_pair_scores = pd.DataFrame()
    if typed_graph is not None and truth_typed_edges is not None:
        raw_local_metrics, local_pair_scores = score_v4(typed_graph, truth_typed_edges)
        local_metrics = {
            "local_structure_scored": True,
            "within_assay_direct_typed_candidate_macro_recall": float(
                raw_local_metrics["direct_typed_candidate_macro_recall"]
            ),
            "within_assay_direct_raw_adjacency_macro_recall": float(
                raw_local_metrics["direct_raw_adjacency_macro_recall"]
            ),
            "within_assay_collider_arrowhead_recall": float(
                raw_local_metrics["collider_arrowhead_recall"]
            ),
            "within_assay_strict_reverse_compelled_count": int(
                raw_local_metrics["strict_reverse_compelled_count"]
            ),
            "within_assay_direct_class_success": int(
                raw_local_metrics["direct_class_success"]
            ),
            "within_assay_collider_class_success": int(
                raw_local_metrics["collider_class_success"]
            ),
            "local_graph_primary_skeleton_precision": float(
                raw_local_metrics["primary_skeleton_precision"]
            ),
            "local_graph_matched_null_false_positive_rate": float(
                raw_local_metrics["matched_null_false_positive_rate"]
            ),
            "local_graph_n_estimated_focal_edges": int(
                raw_local_metrics["n_estimated_focal_edges"]
            ),
            "local_graph_n_unsupported_focal_edges": int(
                raw_local_metrics["n_unsupported_focal_edges"]
            ),
            "local_graph_n_wrong_layer_primary_edges": int(
                raw_local_metrics["n_wrong_layer_primary_edges"]
            ),
            "local_graph_precision_class_success": int(
                raw_local_metrics["precision_class_success"]
            ),
            "local_graph_null_class_success": int(
                raw_local_metrics["null_class_success"]
            ),
            "local_graph_density_class_success": int(
                raw_local_metrics["density_class_success"]
            ),
        }
    else:
        local_metrics = {
            "local_structure_scored": False,
            "within_assay_direct_typed_candidate_macro_recall": 0.0,
            "within_assay_direct_raw_adjacency_macro_recall": 0.0,
            "within_assay_collider_arrowhead_recall": 0.0,
            "within_assay_strict_reverse_compelled_count": 0,
            "within_assay_direct_class_success": 0,
            "within_assay_collider_class_success": 0,
            "local_graph_primary_skeleton_precision": 0.0,
            "local_graph_matched_null_false_positive_rate": 0.0,
            "local_graph_n_estimated_focal_edges": 0,
            "local_graph_n_unsupported_focal_edges": 0,
            "local_graph_n_wrong_layer_primary_edges": 0,
            "local_graph_precision_class_success": 0,
            "local_graph_null_class_success": 0,
            "local_graph_density_class_success": 0,
        }
    overlap_annotation_present = bool(
        membership["direct_layer_annotation_available"].iloc[0]
    )
    overlap_annotation_metrics: dict[str, Any] = {
        "direct_layer_overlap_annotation_present": overlap_annotation_present,
        "direct_layer_overlap_annotation_audited_against_typed_graph": False,
        "direct_layer_overlap_annotation_precision": 0.0,
        "direct_layer_overlap_annotation_recall": 0.0,
        "direct_layer_overlap_annotation_exact": 0,
    }
    if overlap_annotation_present and typed_graph is not None:
        annotated = set(
            membership.loc[membership["direct_layer_overlap"], "feature"].astype(str)
        )
        graph_children = _observable_direct_child_set(typed_graph)
        overlap_annotation_metrics = {
            "direct_layer_overlap_annotation_present": True,
            "direct_layer_overlap_annotation_audited_against_typed_graph": True,
            "direct_layer_overlap_annotation_precision": _safe_rate(
                len(annotated & graph_children), len(annotated)
            ),
            "direct_layer_overlap_annotation_recall": _safe_rate(
                len(annotated & graph_children), len(graph_children)
            ),
            "direct_layer_overlap_annotation_exact": int(
                annotated == graph_children
            ),
        }
    local_success = int(
        bool(local_metrics["local_structure_scored"])
        and local_metrics["within_assay_direct_typed_candidate_macro_recall"]
        >= MIN_LOCAL_DIRECT_RECALL
        and local_metrics["within_assay_collider_arrowhead_recall"]
        >= MIN_LOCAL_COLLIDER_RECALL
        and local_metrics["within_assay_strict_reverse_compelled_count"] == 0
        and local_metrics["local_graph_primary_skeleton_precision"]
        >= MIN_LOCAL_PRIMARY_SKELETON_PRECISION
        and local_metrics["local_graph_matched_null_false_positive_rate"]
        <= MAX_LOCAL_MATCHED_NULL_FALSE_POSITIVE_RATE
        and local_metrics["local_graph_n_estimated_focal_edges"]
        <= MAX_LOCAL_FOCAL_EDGES_PER_WORLD
        and local_metrics["local_graph_precision_class_success"] == 1
        and local_metrics["local_graph_null_class_success"] == 1
        and local_metrics["local_graph_density_class_success"] == 1
    )

    metrics: dict[str, Any] = {
        "schema_version": SCORER_SCHEMA_VERSION,
        "world_seed": world_seed,
        **{f"DOE_{key}": value for key, value in expected_design.items()},
        "scoring_contract": "dimension_agnostic_shared_latent_state_subspace",
        "primary_observable_scoring_object": "multi_assay_state_responsive_set",
        "shared_subspace_interpretation": (
            "hypothesis_only_conditional_on_common_state_model"
        ),
        "common_state_model_identified_by_focal_responsiveness_alone": False,
        "subspace_edge_type": SUBSPACE_EDGE_TYPE,
        "exact_latent_axis_identity_scored": False,
        "exact_cross_assay_feature_pairs_scored": False,
        "cross_assay_causal_direction_scored": False,
        "rna_gene_exact_peak_link_scored": False,
        "n_truth_state_responsive_features": len(eligible),
        "n_truth_state_responsive_chromatin_features": len(chromatin_eligible),
        "n_truth_sentinel_null_features": len(sentinel),
        "n_truth_direct_layer_overlap_features": len(direct_overlap),
        "n_selected_subspace_features": len(selected),
        "n_selected_true_subspace_features": len(selected_true),
        "n_selected_ineligible_features": int(
            len(selected) - len(selected_true)
        ),
        "subspace_member_precision": member_precision,
        "chromatin_subspace_member_precision": chromatin_member_precision,
        "subspace_member_recall": total_recall,
        "chromatin_state_responsive_member_recall": chromatin_recall,
        "P_member_recall": role_recalls["P"],
        "Q_member_recall": role_recalls["Q"],
        "S_member_recall": role_recalls["S"],
        "minimum_P_Q_S_member_recall": min(
            role_recalls[role] for role in CORE_STATE_ROLES
        ),
        "C_state_responsive_member_recall": role_recalls["C"],
        "sentinel_null_inclusion_rate": sentinel_rate,
        "n_selected_sentinel_null_features": int(sentinel["selected"].sum()),
        "n_selected_direct_layer_overlap_features": int(
            direct_overlap["selected"].sum()
        ),
        "rna_gene_subspace_support": gene_selected,
        "n_assays_covered": int(assay_scores["assay_covered"].sum()),
        "n_truth_assays": len(assay_scores),
        "assay_coverage": assay_coverage,
        "chromatin_assay_coverage": chromatin_assay_coverage,
        "minimum_chromatin_assay_member_recall": (
            minimum_chromatin_assay_member_recall
        ),
        "n_complete_chromatin_assay_groups": int(
            chromatin_assays["complete_assay_group"].sum()
        ),
        **hyperedge_checks,
        "membership_class_success": membership_success,
        "hyperedge_class_success": hyperedge_success,
        "subspace_world_success": subspace_success,
        **local_metrics,
        **overlap_annotation_metrics,
        "local_structure_success": local_success,
        "joint_world_success": int(subspace_success and local_success),
        "joint_success_requires_local_graph_score": True,
        "thresholds": validation_protocol()["world_thresholds"],
    }
    return SharedStateSubspaceScore(
        metrics=metrics,
        feature_scores=feature_scores,
        assay_scores=assay_scores,
        hyperedge_scores=hyperedge_scores,
        truth=truth,
        local_pair_scores=local_pair_scores,
    )


def make_global_pretruth_freeze(
    *,
    split: str,
    discovery_attempts: Sequence[Mapping[str, Any]],
    algorithm_source_sha256: str,
    algorithm_config_sha256: str,
    joint_executable_source_sha256: Mapping[str, str],
    development_validation_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the self-hashed all-world barrier before any truth is opened.

    Each attempt must contain ``world_seed`` and ``status``.  A successful
    attempt additionally contains hashes for the subspace candidate, its
    formal assay-local embedding manifest/table, the one-candidate ordinary
    graph freeze, and the graph bytes consumed by joint scoring.  Failure and
    timeout records are retained and expose none of those hashes.
    """

    normalized = _normalise_split(split)
    expected_seeds = SPLIT_SEEDS[normalized]
    source_hash = _valid_hash(algorithm_source_sha256, "algorithm source hash")
    if source_hash != FROZEN_DISCOVERY_SOURCE_SHA256:
        raise ValueError("Global freeze does not bind the preregistered discovery source")
    config_hash = _valid_hash(algorithm_config_sha256, "algorithm config hash")
    required_config_hash = canonical_sha256(FROZEN_DISCOVERY_CONFIG)
    if config_hash != required_config_hash:
        raise ValueError("Global freeze does not use the preregistered discovery config")
    if not isinstance(joint_executable_source_sha256, Mapping) or not (
        joint_executable_source_sha256
    ):
        raise ValueError("Joint executable source hash map must be non-empty")
    source_map: dict[str, str] = {}
    for name, digest in joint_executable_source_sha256.items():
        source_name = str(name).strip()
        if not source_name or Path(source_name).name != source_name or source_name in source_map:
            raise ValueError("Joint executable source names must be unique basenames")
        source_map[source_name] = _valid_hash(
            digest, f"joint executable source hash for {source_name}"
        )
    if source_map.get("shared_state_subspace_v4.py") != source_hash:
        raise ValueError("Joint source map does not bind the subspace algorithm source")
    if set(source_map) != set(REQUIRED_JOINT_EXECUTABLE_SOURCES):
        raise ValueError(
            "Joint source map must exactly match the preregistered executable bundle"
        )
    local_config_hash = canonical_sha256(FROZEN_LOCAL_CONFIG)
    embedding_config_hash = canonical_sha256(FROZEN_SUBSPACE_EMBEDDING_CONFIG)
    embedding_source_hash = source_map.get("assay_local_state_embedding_v4.py")
    joint_identity = {
        "subspace_config_sha256": config_hash,
        "subspace_embedding_method": FROZEN_SUBSPACE_EMBEDDING_METHOD,
        "subspace_embedding_config_sha256": embedding_config_hash,
        "subspace_embedding_source_sha256": embedding_source_hash,
        "local_candidate_name": FROZEN_LOCAL_CANDIDATE_NAME,
        "local_config_sha256": local_config_hash,
        "joint_executable_source_sha256": source_map,
    }
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for value in discovery_attempts:
        seed = int(value.get("world_seed", -1))
        if seed in seen:
            raise ValueError("Discovery attempts repeat a world seed")
        seen.add(seed)
        status = str(value.get("status", "")).strip().casefold()
        if status not in {"ok", "timeout", "crash", "invalid", "error"}:
            raise ValueError(f"Unsupported discovery status for seed {seed}")
        record: dict[str, Any] = {"world_seed": seed, "status": status}
        if status == "ok":
            record["candidate_manifest_sha256"] = _valid_hash(
                value.get("candidate_manifest_sha256", ""),
                f"candidate manifest hash for seed {seed}",
            )
            record["subspace_embedding_manifest_sha256"] = _valid_hash(
                value.get("subspace_embedding_manifest_sha256", ""),
                f"subspace embedding manifest hash for seed {seed}",
            )
            record["subspace_embedding_sha256"] = _valid_hash(
                value.get("subspace_embedding_sha256", ""),
                f"subspace embedding table hash for seed {seed}",
            )
            record["local_pretruth_manifest_sha256"] = _valid_hash(
                value.get("local_pretruth_manifest_sha256", ""),
                f"local pretruth manifest hash for seed {seed}",
            )
            record["local_consensus_edges_sha256"] = _valid_hash(
                value.get("local_consensus_edges_sha256", ""),
                f"local consensus-edge hash for seed {seed}",
            )
        elif value.get("candidate_manifest_sha256"):
            raise ValueError("Failed discovery attempts may not expose a candidate")
        elif value.get("subspace_embedding_manifest_sha256") or value.get(
            "subspace_embedding_sha256"
        ):
            raise ValueError("Failed discovery attempts may not expose an embedding")
        elif value.get("local_pretruth_manifest_sha256"):
            raise ValueError("Failed discovery attempts may not expose local structure")
        elif value.get("local_consensus_edges_sha256"):
            raise ValueError("Failed discovery attempts may not expose local edges")
        records.append(record)
    if seen != set(expected_seeds):
        raise ValueError("Discovery attempts do not exactly cover the preregistered split")
    records.sort(key=lambda row: int(row["world_seed"]))
    protocol_hash = validation_protocol()["protocol_payload_sha256"]
    manifest: dict[str, Any] = {
        "schema_version": GLOBAL_FREEZE_SCHEMA_VERSION,
        "split": normalized,
        "registered_world_seeds": list(expected_seeds),
        "algorithm_source_sha256": source_hash,
        "algorithm_config_sha256": config_hash,
        "subspace_embedding_method": FROZEN_SUBSPACE_EMBEDDING_METHOD,
        "subspace_embedding_config_sha256": embedding_config_hash,
        "subspace_embedding_source_sha256": embedding_source_hash,
        "local_candidate_name": FROZEN_LOCAL_CANDIDATE_NAME,
        "local_config_sha256": local_config_hash,
        "joint_executable_source_sha256": source_map,
        "joint_candidate_identity_sha256": canonical_sha256(joint_identity),
        "protocol_payload_sha256": protocol_hash,
        "all_registered_discovery_attempts_finished": True,
        "created_before_any_scorer_read": True,
        "scorer_domain_opened": False,
        "world_records": records,
    }
    if normalized == "LOCKED":
        manifest["development_validation_manifest_sha256"] = _valid_hash(
            development_validation_manifest_sha256 or "",
            "development validation manifest hash",
        )
    elif development_validation_manifest_sha256 is not None:
        raise ValueError("Only LOCKED may bind a development validation manifest")
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_global_pretruth_freeze(
    manifest: Mapping[str, Any], *, split: str | None = None
) -> dict[str, Any]:
    """Validate an all-world barrier without reading scorer-domain inputs."""

    if manifest.get("schema_version") != GLOBAL_FREEZE_SCHEMA_VERSION:
        raise ValueError("Unsupported global pretruth freeze schema")
    _validate_self_hash(manifest, "manifest_payload_sha256", "global freeze")
    normalized = _normalise_split(str(manifest.get("split", "")))
    if split is not None and normalized != _normalise_split(split):
        raise ValueError("Global freeze split mismatch")
    if manifest.get("registered_world_seeds") != list(SPLIT_SEEDS[normalized]):
        raise ValueError("Global freeze does not bind the preregistered seeds")
    if manifest.get("all_registered_discovery_attempts_finished") is not True:
        raise ValueError("Not every discovery attempt finished before the freeze")
    if manifest.get("created_before_any_scorer_read") is not True:
        raise ValueError("Global freeze was not created before scorer access")
    if manifest.get("scorer_domain_opened") is not False:
        raise ValueError("Global freeze reports premature scorer access")
    source_hash = _valid_hash(
        manifest.get("algorithm_source_sha256", ""), "algorithm source hash"
    )
    if source_hash != FROZEN_DISCOVERY_SOURCE_SHA256:
        raise ValueError("Global freeze binds a non-preregistered discovery source")
    config_hash = _valid_hash(
        manifest.get("algorithm_config_sha256", ""), "algorithm config hash"
    )
    if config_hash != canonical_sha256(FROZEN_DISCOVERY_CONFIG):
        raise ValueError("Global freeze binds a non-preregistered discovery config")
    if manifest.get("local_candidate_name") != FROZEN_LOCAL_CANDIDATE_NAME or (
        manifest.get("local_config_sha256")
        != canonical_sha256(FROZEN_LOCAL_CONFIG)
    ):
        raise ValueError("Global freeze binds another local-structure candidate")
    raw_source_map = manifest.get("joint_executable_source_sha256")
    if not isinstance(raw_source_map, Mapping) or not raw_source_map:
        raise ValueError("Global freeze lacks joint executable source hashes")
    source_map = {
        str(name): _valid_hash(digest, f"joint source hash for {name}")
        for name, digest in raw_source_map.items()
    }
    if source_map.get("shared_state_subspace_v4.py") != source_hash:
        raise ValueError("Global joint sources do not bind the subspace source")
    if set(source_map) != set(REQUIRED_JOINT_EXECUTABLE_SOURCES):
        raise ValueError("Global freeze does not bind the exact executable bundle")
    embedding_config_hash = canonical_sha256(FROZEN_SUBSPACE_EMBEDDING_CONFIG)
    embedding_source_hash = source_map.get("assay_local_state_embedding_v4.py")
    if (
        manifest.get("subspace_embedding_method")
        != FROZEN_SUBSPACE_EMBEDDING_METHOD
        or manifest.get("subspace_embedding_config_sha256")
        != embedding_config_hash
        or manifest.get("subspace_embedding_source_sha256")
        != embedding_source_hash
    ):
        raise ValueError("Global freeze binds another subspace embedding")
    joint_identity = {
        "subspace_config_sha256": config_hash,
        "subspace_embedding_method": FROZEN_SUBSPACE_EMBEDDING_METHOD,
        "subspace_embedding_config_sha256": embedding_config_hash,
        "subspace_embedding_source_sha256": embedding_source_hash,
        "local_candidate_name": FROZEN_LOCAL_CANDIDATE_NAME,
        "local_config_sha256": canonical_sha256(FROZEN_LOCAL_CONFIG),
        "joint_executable_source_sha256": source_map,
    }
    if manifest.get("joint_candidate_identity_sha256") != canonical_sha256(
        joint_identity
    ):
        raise ValueError("Global joint candidate identity mismatch")
    if manifest.get("protocol_payload_sha256") != validation_protocol()[
        "protocol_payload_sha256"
    ]:
        raise ValueError("Global freeze binds a different validation protocol")
    records = manifest.get("world_records")
    if not isinstance(records, list):
        raise ValueError("Global freeze world_records must be a list")
    record_map: dict[int, dict[str, Any]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise ValueError("Malformed global freeze world record")
        seed = int(value.get("world_seed", -1))
        if seed in record_map:
            raise ValueError("Global freeze repeats a world seed")
        status = str(value.get("status", "")).casefold()
        if status not in {"ok", "timeout", "crash", "invalid", "error"}:
            raise ValueError("Global freeze contains an unknown discovery status")
        if status == "ok":
            _valid_hash(
                value.get("candidate_manifest_sha256", ""),
                f"candidate manifest hash for seed {seed}",
            )
            _valid_hash(
                value.get("subspace_embedding_manifest_sha256", ""),
                f"subspace embedding manifest hash for seed {seed}",
            )
            _valid_hash(
                value.get("subspace_embedding_sha256", ""),
                f"subspace embedding table hash for seed {seed}",
            )
            _valid_hash(
                value.get("local_pretruth_manifest_sha256", ""),
                f"local pretruth manifest hash for seed {seed}",
            )
            _valid_hash(
                value.get("local_consensus_edges_sha256", ""),
                f"local consensus-edge hash for seed {seed}",
            )
        elif any(
            field in value
            for field in (
                "candidate_manifest_sha256",
                "subspace_embedding_manifest_sha256",
                "subspace_embedding_sha256",
                "local_pretruth_manifest_sha256",
                "local_consensus_edges_sha256",
            )
        ):
            raise ValueError("Failed global record unexpectedly binds a candidate")
        record_map[seed] = dict(value)
    if set(record_map) != set(SPLIT_SEEDS[normalized]):
        raise ValueError("Global freeze world records do not match its seed block")
    if normalized == "LOCKED":
        _valid_hash(
            manifest.get("development_validation_manifest_sha256", ""),
            "development validation manifest hash",
        )
    return {
        "split": normalized,
        "algorithm_source_sha256": source_hash,
        "algorithm_config_sha256": config_hash,
        "subspace_embedding_method": FROZEN_SUBSPACE_EMBEDDING_METHOD,
        "subspace_embedding_config_sha256": embedding_config_hash,
        "subspace_embedding_source_sha256": str(embedding_source_hash),
        "joint_executable_source_sha256": source_map,
        "joint_candidate_identity_sha256": str(
            manifest["joint_candidate_identity_sha256"]
        ),
        "record_map": record_map,
        "manifest_payload_sha256": str(manifest["manifest_payload_sha256"]),
    }


def load_frozen_subspace_candidate(
    *,
    feature_membership_path: Path | str,
    typed_subspace_hyperedges_path: Path | str,
    candidate_manifest_path: Path | str,
    subspace_embedding_manifest_path: Path | str,
    subspace_embedding_path: Path | str,
    local_pretruth_manifest_path: Path | str,
    local_consensus_edges_path: Path | str,
    global_pretruth_manifest_path: Path | str,
    split: str,
    world_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate/load candidate bytes before the caller reads scorer truth."""

    membership_path = Path(feature_membership_path)
    hyperedge_path = Path(typed_subspace_hyperedges_path)
    candidate_path = Path(candidate_manifest_path)
    embedding_manifest_path = Path(subspace_embedding_manifest_path)
    embedding_path = Path(subspace_embedding_path)
    local_manifest_path = Path(local_pretruth_manifest_path)
    local_edges_path = Path(local_consensus_edges_path)
    global_path = Path(global_pretruth_manifest_path)
    for path, label in (
        (membership_path, "feature membership"),
        (hyperedge_path, "typed hyperedges"),
        (candidate_path, "candidate manifest"),
        (embedding_manifest_path, "subspace embedding manifest"),
        (embedding_path, "subspace embedding table"),
        (local_manifest_path, "local pretruth manifest"),
        (local_edges_path, "local consensus edges"),
        (global_path, "global pretruth manifest"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular non-symlink file")
    candidate_root = candidate_path.parent.resolve(strict=True)
    if membership_path.resolve(strict=True).parent != candidate_root or (
        hyperedge_path.resolve(strict=True).parent != candidate_root
    ):
        raise ValueError("Candidate outputs and manifest must share one directory")
    if membership_path.name != "subspace_feature_membership.tsv" or (
        hyperedge_path.name != "typed_subspace_hyperedges.tsv"
    ) or candidate_path.name != "frozen_discovery_manifest.json":
        raise ValueError("Candidate files do not use the frozen output names")
    raw_embedding_root = embedding_manifest_path.parent
    if raw_embedding_root.is_symlink() or not raw_embedding_root.is_dir():
        raise ValueError("Formal subspace embedding root must be a regular directory")
    embedding_root = raw_embedding_root.resolve(strict=True)
    if embedding_path.resolve(strict=True).parent != embedding_root or (
        embedding_manifest_path.name != assay_local_embedding.MANIFEST_NAME
    ) or embedding_path.name != assay_local_embedding.EMBEDDING_NAME:
        raise ValueError("Formal subspace embedding files do not share its frozen layout")

    embedding_provenance_path = embedding_root / assay_local_embedding.PROVENANCE_NAME
    embedding_sidecar_path = embedding_root / "frozen_evidence_manifest.sha256"
    expected_embedding_files = {
        assay_local_embedding.EMBEDDING_NAME,
        assay_local_embedding.PROVENANCE_NAME,
        assay_local_embedding.MANIFEST_NAME,
        "frozen_evidence_manifest.sha256",
    }
    embedding_entries = list(embedding_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in embedding_entries):
        raise ValueError("Formal subspace embedding directory contains an unsafe entry")
    if {path.name for path in embedding_entries} != expected_embedding_files:
        raise ValueError(
            "Formal subspace embedding directory contains missing or extra files"
        )
    provenance_path = candidate_root / "subspace_provenance.json"
    snapshot_paths = {
        "membership": membership_path,
        "hyperedges": hyperedge_path,
        "candidate_manifest": candidate_path,
        "embedding_manifest": embedding_manifest_path,
        "embedding": embedding_path,
        "embedding_provenance": embedding_provenance_path,
        "embedding_sidecar": embedding_sidecar_path,
        "local_manifest": local_manifest_path,
        "local_edges": local_edges_path,
        "global_manifest": global_path,
        "subspace_provenance": provenance_path,
    }
    snapshot = {
        label: _read_regular_bytes_once(path, label.replace("_", " "))
        for label, path in snapshot_paths.items()
    }

    global_manifest = _json_from_bytes(
        snapshot["global_manifest"], "global pretruth manifest"
    )
    global_audit = validate_global_pretruth_freeze(global_manifest, split=split)
    seed = int(world_seed)
    record = global_audit["record_map"].get(seed)
    if record is None or record.get("status") != "ok":
        raise ValueError("World is not a successful frozen discovery attempt")
    candidate_file_hash = _bytes_sha256(snapshot["candidate_manifest"])
    if candidate_file_hash != record.get("candidate_manifest_sha256"):
        raise ValueError("Candidate manifest differs from the global freeze")
    embedding_manifest_file_hash = _bytes_sha256(snapshot["embedding_manifest"])
    if embedding_manifest_file_hash != record.get(
        "subspace_embedding_manifest_sha256"
    ):
        raise ValueError("Subspace embedding manifest differs from the global freeze")
    embedding_file_hash = _bytes_sha256(snapshot["embedding"])
    if embedding_file_hash != record.get("subspace_embedding_sha256"):
        raise ValueError("Subspace embedding table differs from the global freeze")
    embedding_manifest = _json_from_bytes(
        snapshot["embedding_manifest"], "subspace embedding manifest"
    )
    if embedding_manifest.get("schema_version") != (
        assay_local_embedding.FROZEN_EVIDENCE_SCHEMA_VERSION
    ) or embedding_manifest.get("frozen_before_scorer_read") is not True:
        raise ValueError("Unsupported or unfrozen formal subspace embedding")
    if embedding_manifest.get("algorithm_source_sha256") != global_audit[
        "subspace_embedding_source_sha256"
    ] or embedding_manifest.get("config") != FROZEN_SUBSPACE_EMBEDDING_CONFIG:
        raise ValueError("Formal subspace embedding source/config is not frozen")
    if (
        int(embedding_manifest.get("formal_state_dimension", -1)) != 3
        or embedding_manifest.get("zero_padding_used") is not False
        or embedding_manifest.get("cross_assay_alignment_used") is not False
        or embedding_manifest.get(
            "all_assays_parallel_rank_support_formal_dimension"
        )
        is not True
    ):
        raise ValueError("Formal subspace embedding violates the rank/boundary contract")
    embedding_boundary = embedding_manifest.get("information_boundary")
    if not isinstance(embedding_boundary, Mapping) or (
        embedding_boundary.get("truth_or_scorer_used") is not False
        or embedding_boundary.get("cell_rows_compared_across_assays") is not False
        or embedding_boundary.get("feature_columns_compared_across_assays") is not False
    ):
        raise ValueError("Formal subspace embedding information boundary is invalid")
    if set(map(str, embedding_boundary.get("consumed", []))) != {
        "observed_background_matrix_per_assay",
        "observed_leave_focal_out_depth_per_assay",
        "assay_key",
    } or not {
        "scorer",
        "truth_graph",
        "simulator_roles",
        "true_latent_state",
        "external_data",
    } <= set(map(str, embedding_boundary.get("not_consumed", []))):
        raise ValueError("Formal subspace embedding did not preserve its raw boundary")
    if set(map(str, embedding_manifest.get("accepted_assays", []))) != set(
        assay_local_embedding.EXPECTED_MODALITIES
    ):
        raise ValueError("Formal subspace embedding assay set is incomplete")
    embedding_inputs = embedding_manifest.get("input_file_sha256")
    expected_background_names = {
        f"background_matrix__{modality}.npz"
        for modality in assay_local_embedding.EXPECTED_MODALITIES
    }
    if not isinstance(embedding_inputs, Mapping) or set(
        map(str, embedding_inputs)
    ) != expected_background_names:
        raise ValueError("Formal subspace embedding does not bind all raw backgrounds")
    for name, digest in embedding_inputs.items():
        _valid_hash(digest, f"formal embedding raw input hash for {name}")
    embedding_output_hashes = embedding_manifest.get("output_file_sha256")
    if not isinstance(embedding_output_hashes, Mapping) or (
        embedding_output_hashes.get(embedding_path.name) != embedding_file_hash
    ) or embedding_output_hashes.get(embedding_provenance_path.name) != _bytes_sha256(
        snapshot["embedding_provenance"]
    ):
        raise ValueError("Formal subspace embedding output hashes disagree")
    try:
        sidecar_parts = snapshot["embedding_sidecar"].decode("utf-8").split()
    except UnicodeDecodeError as exc:
        raise ValueError("Subspace embedding sidecar is not UTF-8") from exc
    if not sidecar_parts or sidecar_parts[0] != embedding_manifest_file_hash:
        raise ValueError("Formal subspace embedding sidecar disagrees")
    local_manifest_file_hash = _bytes_sha256(snapshot["local_manifest"])
    if local_manifest_file_hash != record.get("local_pretruth_manifest_sha256"):
        raise ValueError("Local pretruth manifest differs from the global freeze")
    local_edges_file_hash = _bytes_sha256(snapshot["local_edges"])
    if local_edges_file_hash != record.get("local_consensus_edges_sha256"):
        raise ValueError("Local consensus edges differ from the global freeze")
    local_manifest = _json_from_bytes(
        snapshot["local_manifest"], "local pretruth manifest"
    )
    if local_manifest.get("all_consensuses_built_before_truth_open") is not True:
        raise ValueError("Local graph was not frozen before scorer access")
    local_logical_hashes = local_manifest.get("consensus_sha256")
    local_file_hashes = local_manifest.get("consensus_file_sha256")
    if (
        not isinstance(local_logical_hashes, Mapping)
        or not isinstance(local_file_hashes, Mapping)
        or int(local_manifest.get("candidate_count", -1)) != 1
        or set(local_logical_hashes) != {FROZEN_LOCAL_CANDIDATE_NAME}
        or set(local_file_hashes) != {FROZEN_LOCAL_CANDIDATE_NAME}
        or local_file_hashes.get(FROZEN_LOCAL_CANDIDATE_NAME)
        != local_edges_file_hash
    ):
        raise ValueError(
            "Local pretruth manifest does not byte-bind the selected consensus"
        )
    expected_local_edges = (
        local_manifest_path.parent
        / FROZEN_LOCAL_CANDIDATE_NAME
        / "consensus_edges.tsv"
    )
    if local_edges_path.resolve(strict=True) != expected_local_edges.resolve(strict=True):
        raise ValueError("Local consensus path is not the frozen selected candidate")
    candidate = _json_from_bytes(snapshot["candidate_manifest"], "candidate manifest")
    if candidate.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate manifest schema")
    if candidate.get("frozen_before_scorer_read") is not True:
        raise ValueError("Candidate was not frozen before scorer access")
    if candidate.get("algorithm_source_sha256") != global_audit[
        "algorithm_source_sha256"
    ]:
        raise ValueError("Candidate algorithm source differs from the global freeze")
    dependency_hashes = candidate.get("dependency_source_sha256")
    if not isinstance(dependency_hashes, Mapping) or not dependency_hashes:
        raise ValueError("Candidate manifest lacks dependency source hashes")
    candidate_sources = {
        "shared_state_subspace_v4.py": str(candidate["algorithm_source_sha256"]),
        **{str(name): str(digest) for name, digest in dependency_hashes.items()},
    }
    if set(dependency_hashes) != {
        "cell_state_loading_groups.py",
        "assay_local_state_embedding_v4.py",
    }:
        raise ValueError("Candidate does not bind the exact formal dependency set")
    for name, digest in candidate_sources.items():
        if global_audit["joint_executable_source_sha256"].get(name) != _valid_hash(
            digest, f"candidate dependency source hash for {name}"
        ):
            raise ValueError(f"Candidate dependency source is not globally bound: {name}")
    candidate_config = candidate.get("config")
    if not isinstance(candidate_config, Mapping) or canonical_sha256(
        candidate_config
    ) != global_audit["algorithm_config_sha256"]:
        raise ValueError("Candidate algorithm config differs from the global freeze")
    boundary = candidate.get("information_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("truth_or_scorer_used") is not False:
        raise ValueError("Candidate information boundary does not exclude scorer truth")
    safe_embedding = candidate.get("safe_embedding_freeze_evidence")
    if not isinstance(safe_embedding, Mapping) or safe_embedding.get(
        "layout"
    ) != "assay_local_state_embedding_v4":
        raise ValueError("Candidate did not use the formal assay-local embedding")
    expected_embedding_files = {
        embedding_path.name: embedding_file_hash,
        embedding_manifest_path.name: embedding_manifest_file_hash,
        embedding_sidecar_path.name: _bytes_sha256(snapshot["embedding_sidecar"]),
        embedding_provenance_path.name: _bytes_sha256(
            snapshot["embedding_provenance"]
        ),
    }
    if safe_embedding.get("files") != expected_embedding_files or (
        safe_embedding.get("upstream_algorithm_source_sha256")
        != {
            "assay_local_state_embedding_v4.py": global_audit[
                "subspace_embedding_source_sha256"
            ]
        }
    ) or int(safe_embedding.get("formal_state_dimension", -1)) != 3:
        raise ValueError("Candidate/formal subspace embedding evidence disagrees")
    output_hashes = candidate.get("output_sha256")
    expected_outputs = {
        "subspace_feature_membership.tsv",
        "typed_subspace_hyperedges.tsv",
        "subspace_provenance.json",
    }
    if not isinstance(output_hashes, Mapping) or set(output_hashes) != expected_outputs:
        raise ValueError("Candidate manifest output file set is not exact")
    candidate_entries = list(candidate_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in candidate_entries):
        raise ValueError("Candidate directory contains an unsafe entry")
    actual_files = {
        path.name for path in candidate_entries if path.name != candidate_path.name
    }
    if actual_files != expected_outputs:
        raise ValueError("Candidate directory contains missing or extra output files")
    for filename, digest in output_hashes.items():
        expected_hash = _valid_hash(digest, f"candidate output hash for {filename}")
        snapshot_label = {
            "subspace_feature_membership.tsv": "membership",
            "typed_subspace_hyperedges.tsv": "hyperedges",
            "subspace_provenance.json": "subspace_provenance",
        }[str(filename)]
        if _bytes_sha256(snapshot[snapshot_label]) != expected_hash:
            raise ValueError(f"Frozen candidate output changed: {filename}")
    provenance = _json_from_bytes(
        snapshot["subspace_provenance"], "subspace provenance"
    )
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported subspace provenance schema")
    if provenance.get("config") != dict(candidate_config):
        raise ValueError("Subspace provenance/config manifest mismatch")
    if provenance.get("information_boundary") != boundary:
        raise ValueError("Subspace provenance information boundary mismatch")
    if provenance.get("formal_state_embedding") != safe_embedding:
        raise ValueError("Subspace provenance/formal embedding evidence mismatch")
    if provenance.get("descriptive_state_depth_decomposition") != (
        EXPECTED_DESCRIPTIVE_DECOMPOSITION
    ):
        raise ValueError(
            "Subspace provenance does not bind the non-selecting OOF decomposition"
        )
    error_control = provenance.get("error_control")
    if not isinstance(error_control, Mapping) or error_control.get(
        "binary_eligibility"
    ) != EXPECTED_BINARY_ELIGIBILITY_POLICY or error_control.get(
        "ineligible_placeholder_policy"
    ) != EXPECTED_INELIGIBLE_PLACEHOLDER_POLICY:
        raise ValueError(
            "Subspace provenance does not bind pre-fit eligibility and placeholders"
        )

    # Parsing occurs only after every candidate/global byte and source binding
    # has been checked.  The caller may now open scorer-domain roles/parameters.
    # Preserve the writer's exact binary float when reconstructing gates.  In
    # particular, an adjusted p-value one ULP above a threshold must not be
    # rounded onto the threshold during the scorer-domain reload.
    membership = pd.read_csv(
        io.BytesIO(snapshot["membership"]),
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )
    hyperedges = pd.read_csv(
        io.BytesIO(snapshot["hyperedges"]),
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )
    local_edges = pd.read_csv(
        io.BytesIO(snapshot["local_edges"]),
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )
    participation_audit = _validate_assay_participation_provenance(
        membership, provenance
    )
    return membership, hyperedges, local_edges, {
        "global_manifest_payload_sha256": global_audit["manifest_payload_sha256"],
        "global_manifest_file_sha256": _bytes_sha256(snapshot["global_manifest"]),
        "candidate_manifest_file_sha256": candidate_file_hash,
        "subspace_embedding_manifest_file_sha256": (
            embedding_manifest_file_hash
        ),
        "subspace_embedding_file_sha256": embedding_file_hash,
        "local_pretruth_manifest_file_sha256": local_manifest_file_hash,
        "local_consensus_edges_file_sha256": local_edges_file_hash,
        "algorithm_source_sha256": global_audit["algorithm_source_sha256"],
        "algorithm_config_sha256": global_audit["algorithm_config_sha256"],
        "joint_candidate_identity_sha256": global_audit[
            "joint_candidate_identity_sha256"
        ],
        "descriptive_decomposition_provenance_validated": True,
        "prefit_eligibility_provenance_validated": True,
        "assay_participation_provenance_validated": True,
        "assay_participation_selected_modalities": participation_audit[
            "selected_modalities"
        ],
        "candidate_bytes_loaded_before_scorer_read": True,
        "formal_subspace_embedding_bytes_loaded_before_scorer_read": True,
        "local_graph_bytes_loaded_before_scorer_read": True,
    }


def _metric_value(record: Mapping[str, Any], key: str) -> float:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    try:
        value = float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def validate_block_results(
    *,
    split: str,
    result_records: Sequence[Mapping[str, Any]],
    global_pretruth_manifest: Mapping[str, Any],
    development_validation_manifest: Mapping[str, Any] | None = None,
    development_validation_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply the preregistered block gate with failures in the denominator."""

    normalized = _normalise_split(split)
    global_audit = validate_global_pretruth_freeze(
        global_pretruth_manifest, split=normalized
    )
    expected_seeds = SPLIT_SEEDS[normalized]
    records: dict[int, Mapping[str, Any]] = {}
    for record in result_records:
        seed = int(record.get("world_seed", -1))
        if seed in records:
            raise ValueError("Block results repeat a world seed")
        records[seed] = record
    if set(records) != set(expected_seeds):
        raise ValueError("Block results do not exactly cover the preregistered split")

    source_hash = global_audit["algorithm_source_sha256"]
    config_hash = global_audit["algorithm_config_sha256"]
    if normalized == "LOCKED":
        if development_validation_manifest is None:
            raise ValueError("LOCKED requires a passing DEVELOPMENT manifest")
        if development_validation_manifest.get("schema_version") != BLOCK_VALIDATION_SCHEMA_VERSION:
            raise ValueError("Unsupported DEVELOPMENT validation schema")
        _validate_self_hash(
            development_validation_manifest,
            "block_validation_payload_sha256",
            "development validation",
        )
        if development_validation_manifest.get("split") != "DEVELOPMENT" or (
            development_validation_manifest.get("block_pass") is not True
        ):
            raise ValueError("LOCKED requires a passing DEVELOPMENT gate")
        if development_validation_manifest.get("algorithm_source_sha256") != source_hash:
            raise ValueError("Algorithm source changed after DEVELOPMENT")
        if development_validation_manifest.get("algorithm_config_sha256") != config_hash:
            raise ValueError("Algorithm config changed after DEVELOPMENT")
        if development_validation_manifest.get(
            "joint_candidate_identity_sha256"
        ) != global_audit["joint_candidate_identity_sha256"]:
            raise ValueError("Joint executable candidate changed after DEVELOPMENT")
        if development_validation_manifest.get("protocol_payload_sha256") != (
            validation_protocol()["protocol_payload_sha256"]
        ):
            raise ValueError("Validation protocol changed after DEVELOPMENT")
        expected_dev_file_hash = _valid_hash(
            development_validation_manifest_file_sha256 or "",
            "development validation manifest file hash",
        )
        if global_pretruth_manifest.get(
            "development_validation_manifest_sha256"
        ) != expected_dev_file_hash:
            raise ValueError("LOCKED global freeze binds another DEVELOPMENT file")
    elif development_validation_manifest is not None or (
        development_validation_manifest_file_sha256 is not None
    ):
        raise ValueError("Only LOCKED consumes a DEVELOPMENT validation manifest")

    successful_metrics: list[Mapping[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    for seed in expected_seeds:
        record = records[seed]
        discovery_record = global_audit["record_map"][seed]
        discovery_status = str(record.get("discovery_status", "")).casefold()
        scoring_status = str(record.get("scoring_status", "")).casefold()
        if discovery_status != discovery_record["status"]:
            raise ValueError(f"Result/global discovery status mismatch for seed {seed}")
        if record.get("algorithm_source_sha256") != source_hash:
            raise ValueError(f"Result source hash mismatch for seed {seed}")
        if record.get("algorithm_config_sha256") != config_hash:
            raise ValueError(f"Result config hash mismatch for seed {seed}")
        if record.get("joint_candidate_identity_sha256") != global_audit[
            "joint_candidate_identity_sha256"
        ]:
            raise ValueError(f"Result joint-candidate binding mismatch for seed {seed}")
        if record.get("global_manifest_payload_sha256") != global_audit[
            "manifest_payload_sha256"
        ]:
            raise ValueError(f"Result global-freeze binding mismatch for seed {seed}")
        if discovery_status == "ok":
            if record.get("candidate_manifest_sha256") != discovery_record.get(
                "candidate_manifest_sha256"
            ):
                raise ValueError(f"Result candidate binding mismatch for seed {seed}")
            if record.get(
                "subspace_embedding_manifest_sha256"
            ) != discovery_record.get("subspace_embedding_manifest_sha256") or (
                record.get("subspace_embedding_sha256")
                != discovery_record.get("subspace_embedding_sha256")
            ):
                raise ValueError(
                    f"Result subspace-embedding binding mismatch for seed {seed}"
                )
            if record.get("local_pretruth_manifest_sha256") != discovery_record.get(
                "local_pretruth_manifest_sha256"
            ):
                raise ValueError(f"Result local-structure binding mismatch for seed {seed}")
            if record.get("local_consensus_edges_sha256") != discovery_record.get(
                "local_consensus_edges_sha256"
            ):
                raise ValueError(f"Result local-edge binding mismatch for seed {seed}")
        elif scoring_status == "ok":
            raise ValueError("A failed discovery attempt may not have a successful score")
        metrics = record.get("metrics")
        scored = bool(
            discovery_status == "ok"
            and scoring_status == "ok"
            and isinstance(metrics, Mapping)
            and metrics.get("schema_version") == SCORER_SCHEMA_VERSION
            and int(metrics.get("world_seed", -1)) == seed
        )
        if scored:
            expected_design = _doe_design_for_seed(seed)
            for key, expected_value in expected_design.items():
                if int(metrics.get(f"DOE_{key}", 0)) != expected_value:
                    raise ValueError(f"Scored DOE assignment mismatch for seed {seed}")
            if int(metrics.get("n_truth_sentinel_null_features", -1)) != 12:
                raise ValueError("Scored sentinel-null denominator changed")
            if int(metrics.get("n_truth_direct_layer_overlap_features", -1)) != 6:
                raise ValueError("Scored direct-layer-overlap denominator changed")
            successful_metrics.append(metrics)
        world_success = int(scored and int(metrics.get("joint_world_success", 0)) == 1)
        normalized_records.append(
            {
                "world_seed": seed,
                "discovery_status": discovery_status,
                "scoring_status": scoring_status,
                "scored": int(scored),
                "joint_world_success": world_success,
            }
        )

    denominator = len(expected_seeds)
    def mean_all(key: str) -> float:
        return float(
            sum(_metric_value(records[seed], key) for seed in expected_seeds)
            / denominator
        )

    def mean_scored(key: str) -> float:
        if not successful_metrics:
            return 0.0
        return float(
            sum(float(metrics.get(key, 0.0)) for metrics in successful_metrics)
            / len(successful_metrics)
        )

    # _metric_value expects a record with a nested metrics object, so every
    # failed attempt contributes zero to recall/precision/coverage summaries.
    world_successes = sum(row["joint_world_success"] for row in normalized_records)
    unsupported_claims = int(
        sum(_metric_value(records[seed], "n_unsupported_claim_flags") for seed in expected_seeds)
    )
    selected_sentinels = int(
        sum(
            _metric_value(records[seed], "n_selected_sentinel_null_features")
            for seed in expected_seeds
        )
    )
    # Only scored worlds contribute observed negative-control trials.  A
    # crash/timeout is already a world failure and must not dilute the false
    # inclusion rate by masquerading as twelve/six observed negatives.
    sentinel_denominator = 12 * len(successful_metrics)
    sentinel_ucb = _one_sided_wilson_upper(
        selected_sentinels, sentinel_denominator
    )

    design_cell_successes: dict[str, dict[str, int]] = {}
    for cell in range(1, 9):
        seeds = [
            seed
            for seed in expected_seeds
            if _doe_design_for_seed(seed)["half_fraction_cell"] == cell
        ]
        successes = sum(
            next(
                row["joint_world_success"]
                for row in normalized_records
                if row["world_seed"] == seed
            )
            for seed in seeds
        )
        design_cell_successes[str(cell)] = {
            "successes": int(successes),
            "total": len(seeds),
        }
    factor_level_successes: dict[str, dict[str, dict[str, int]]] = {}
    for factor in (
        "A_cells",
        "B_detection",
        "C_population_divergence",
        "D_background_observability",
    ):
        factor_level_successes[factor] = {}
        for level in (-1, 1):
            seeds = [
                seed
                for seed in expected_seeds
                if _doe_design_for_seed(seed)[factor] == level
            ]
            successes = sum(
                next(
                    row["joint_world_success"]
                    for row in normalized_records
                    if row["world_seed"] == seed
                )
                for seed in seeds
            )
            factor_level_successes[factor][str(level)] = {
                "successes": int(successes),
                "total": len(seeds),
            }
    summaries = {
        "n_registered_worlds": denominator,
        "n_scored_worlds": len(successful_metrics),
        "n_joint_successful_worlds": world_successes,
        "joint_world_success_rate": _safe_rate(world_successes, denominator),
        "mean_subspace_member_precision": mean_all("subspace_member_precision"),
        "mean_chromatin_state_responsive_member_recall": mean_all(
            "chromatin_state_responsive_member_recall"
        ),
        "mean_assay_coverage": mean_all("assay_coverage"),
        "mean_minimum_chromatin_assay_member_recall": mean_all(
            "minimum_chromatin_assay_member_recall"
        ),
        "mean_sentinel_null_inclusion_rate": mean_all(
            "sentinel_null_inclusion_rate"
        ),
        "selected_sentinel_null_features": selected_sentinels,
        "sentinel_null_feature_denominator": sentinel_denominator,
        "sentinel_null_inclusion_rate_one_sided_95pct_Wilson_UCB": sentinel_ucb,
        "mean_P_member_recall": mean_all("P_member_recall"),
        "mean_Q_member_recall": mean_all("Q_member_recall"),
        "mean_S_member_recall": mean_all("S_member_recall"),
        "mean_C_state_responsive_member_recall": mean_all(
            "C_state_responsive_member_recall"
        ),
        "mean_within_assay_direct_typed_candidate_macro_recall": mean_all(
            "within_assay_direct_typed_candidate_macro_recall"
        ),
        "mean_within_assay_collider_arrowhead_recall": mean_all(
            "within_assay_collider_arrowhead_recall"
        ),
        "mean_local_graph_primary_skeleton_precision": mean_all(
            "local_graph_primary_skeleton_precision"
        ),
        "mean_local_graph_matched_null_false_positive_rate": mean_scored(
            "local_graph_matched_null_false_positive_rate"
        ),
        "mean_local_graph_n_estimated_focal_edges": mean_scored(
            "local_graph_n_estimated_focal_edges"
        ),
        "total_local_graph_unsupported_focal_edges": int(
            sum(
                _metric_value(records[seed], "local_graph_n_unsupported_focal_edges")
                for seed in expected_seeds
            )
        ),
        "total_local_graph_wrong_layer_primary_edges": int(
            sum(
                _metric_value(records[seed], "local_graph_n_wrong_layer_primary_edges")
                for seed in expected_seeds
            )
        ),
        "total_unsupported_claim_flags": unsupported_claims,
        "design_cell_successes": design_cell_successes,
        "factor_level_successes": factor_level_successes,
    }
    design_cells_pass = all(
        value["successes"] >= MIN_DESIGN_CELL_SUCCESSES[normalized]
        for value in design_cell_successes.values()
    )
    factor_levels_pass = all(
        value["successes"] >= MIN_FACTOR_LEVEL_SUCCESSES[normalized]
        for levels in factor_level_successes.values()
        for value in levels.values()
    )
    gate_checks = {
        "minimum_successful_worlds": world_successes
        >= MINIMUM_SUCCESSFUL_WORLDS[normalized],
        "mean_member_precision": summaries["mean_subspace_member_precision"]
        >= MIN_BLOCK_MEAN_MEMBER_PRECISION,
        "mean_chromatin_recall": summaries[
            "mean_chromatin_state_responsive_member_recall"
        ] >= MIN_BLOCK_MEAN_CHROMATIN_RECALL,
        "mean_assay_coverage": summaries["mean_assay_coverage"]
        >= MIN_BLOCK_MEAN_ASSAY_COVERAGE,
        "mean_minimum_chromatin_assay_recall": summaries[
            "mean_minimum_chromatin_assay_member_recall"
        ] >= MIN_BLOCK_MEAN_MINIMUM_CHROMATIN_ASSAY_RECALL,
        "mean_sentinel_rate": summaries["mean_sentinel_null_inclusion_rate"]
        <= MAX_BLOCK_MEAN_SENTINEL_RATE,
        "sentinel_rate_Wilson_UCB": sentinel_ucb
        <= MAX_BLOCK_SENTINEL_WILSON_UCB,
        "mean_P_role_recall": summaries["mean_P_member_recall"]
        >= MIN_BLOCK_MEAN_ROLE_RECALL,
        "mean_Q_role_recall": summaries["mean_Q_member_recall"]
        >= MIN_BLOCK_MEAN_ROLE_RECALL,
        "mean_S_role_recall": summaries["mean_S_member_recall"]
        >= MIN_BLOCK_MEAN_ROLE_RECALL,
        "mean_C_state_responsive_recall": summaries[
            "mean_C_state_responsive_member_recall"
        ] >= MIN_BLOCK_MEAN_C_STATE_RESPONSIVE_RECALL,
        "mean_local_direct_recall": summaries[
            "mean_within_assay_direct_typed_candidate_macro_recall"
        ] >= MIN_BLOCK_MEAN_LOCAL_DIRECT_RECALL,
        "mean_local_collider_recall": summaries[
            "mean_within_assay_collider_arrowhead_recall"
        ] >= MIN_BLOCK_MEAN_LOCAL_COLLIDER_RECALL,
        "mean_local_primary_precision": summaries[
            "mean_local_graph_primary_skeleton_precision"
        ] >= MIN_BLOCK_MEAN_LOCAL_PRIMARY_PRECISION,
        "mean_local_matched_null_rate": summaries[
            "mean_local_graph_matched_null_false_positive_rate"
        ] <= MAX_BLOCK_MEAN_LOCAL_NULL_RATE,
        "mean_local_focal_edges": summaries[
            "mean_local_graph_n_estimated_focal_edges"
        ] <= MAX_BLOCK_MEAN_LOCAL_FOCAL_EDGES,
        "zero_unsupported_claim_flags": unsupported_claims == 0,
        "every_half_fraction_design_cell": design_cells_pass,
        "every_DOE_factor_level": factor_levels_pass,
    }
    output: dict[str, Any] = {
        "schema_version": BLOCK_VALIDATION_SCHEMA_VERSION,
        "split": normalized,
        "world_seeds": list(expected_seeds),
        "algorithm_source_sha256": source_hash,
        "algorithm_config_sha256": config_hash,
        "subspace_embedding_method": global_audit[
            "subspace_embedding_method"
        ],
        "subspace_embedding_config_sha256": global_audit[
            "subspace_embedding_config_sha256"
        ],
        "subspace_embedding_source_sha256": global_audit[
            "subspace_embedding_source_sha256"
        ],
        "local_candidate_name": FROZEN_LOCAL_CANDIDATE_NAME,
        "local_config_sha256": canonical_sha256(FROZEN_LOCAL_CONFIG),
        "joint_executable_source_sha256": global_audit[
            "joint_executable_source_sha256"
        ],
        "joint_candidate_identity_sha256": global_audit[
            "joint_candidate_identity_sha256"
        ],
        "protocol_payload_sha256": validation_protocol()[
            "protocol_payload_sha256"
        ],
        "global_pretruth_manifest_payload_sha256": global_audit[
            "manifest_payload_sha256"
        ],
        "all_failures_retained_in_denominator": True,
        "result_records": normalized_records,
        "summaries": summaries,
        "gate_checks": gate_checks,
        "block_pass": bool(all(gate_checks.values())),
    }
    output["block_validation_payload_sha256"] = canonical_sha256(output)
    return output


def _write_score(result: SharedStateSubspaceScore, out_dir: Path) -> None:
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty score output {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.feature_scores.to_csv(out_dir / "subspace_feature_scores.tsv", sep="\t", index=False)
    result.assay_scores.to_csv(out_dir / "subspace_assay_scores.tsv", sep="\t", index=False)
    result.hyperedge_scores.to_csv(out_dir / "subspace_hyperedge_scores.tsv", sep="\t", index=False)
    result.truth.features.to_csv(out_dir / "subspace_truth_features.tsv", sep="\t", index=False)
    result.truth.assay_groups.to_csv(out_dir / "subspace_truth_assay_groups.tsv", sep="\t", index=False)
    if not result.local_pair_scores.empty:
        result.local_pair_scores.to_csv(out_dir / "local_pair_scores.tsv", sep="\t", index=False)
    (out_dir / "subspace_metrics.json").write_text(
        json.dumps(_json_ready(result.metrics), indent=2, sort_keys=True) + "\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("write-protocol")
    protocol.add_argument("--out", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--world-parameters", type=Path, required=True)
    score.add_argument("--role-assignments", type=Path, required=True)
    score.add_argument("--feature-membership", type=Path, required=True)
    score.add_argument("--typed-subspace-hyperedges", type=Path, required=True)
    score.add_argument("--typed-graph", type=Path)
    score.add_argument("--truth-typed-edges", type=Path)
    score.add_argument("--candidate-manifest", type=Path)
    score.add_argument("--subspace-embedding-manifest", type=Path)
    score.add_argument("--subspace-embedding", type=Path)
    score.add_argument("--local-pretruth-manifest", type=Path)
    score.add_argument("--global-pretruth-manifest", type=Path)
    score.add_argument("--split")
    score.add_argument("--world-seed", type=int)
    score.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "write-protocol":
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(validation_protocol(), indent=2, sort_keys=True) + "\n"
        )
        return 0

    frozen_arguments = (
        args.candidate_manifest,
        args.subspace_embedding_manifest,
        args.subspace_embedding,
        args.local_pretruth_manifest,
        args.global_pretruth_manifest,
        args.split,
        args.world_seed,
    )
    if any(value is not None for value in frozen_arguments) and not all(
        value is not None for value in frozen_arguments
    ):
        raise ValueError(
            "Certification scoring requires candidate/global manifests, split, and seed"
        )
    if all(value is not None for value in frozen_arguments):
        if args.typed_graph is None:
            raise ValueError("Certification scoring requires frozen local consensus edges")
        membership, hyperedges, graph, _ = load_frozen_subspace_candidate(
            feature_membership_path=args.feature_membership,
            typed_subspace_hyperedges_path=args.typed_subspace_hyperedges,
            candidate_manifest_path=args.candidate_manifest,
            subspace_embedding_manifest_path=args.subspace_embedding_manifest,
            subspace_embedding_path=args.subspace_embedding,
            local_pretruth_manifest_path=args.local_pretruth_manifest,
            local_consensus_edges_path=args.typed_graph,
            global_pretruth_manifest_path=args.global_pretruth_manifest,
            split=args.split,
            world_seed=args.world_seed,
        )
    else:
        # Explicitly diagnostic: no certification barrier was supplied.
        membership = pd.read_csv(
            args.feature_membership,
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        hyperedges = pd.read_csv(
            args.typed_subspace_hyperedges,
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        graph = (
            pd.read_csv(
                args.typed_graph,
                sep="\t",
                keep_default_na=False,
                float_precision="round_trip",
            )
            if args.typed_graph is not None
            else None
        )
    # Scorer-domain reads occur only after frozen candidate loading above.
    parameters = json.loads(args.world_parameters.read_text())
    roles = pd.read_csv(args.role_assignments, sep="\t", keep_default_na=False)
    pair_truth = (
        pd.read_csv(args.truth_typed_edges, sep="\t", keep_default_na=False)
        if args.truth_typed_edges is not None
        else None
    )
    result = score_shared_state_subspace(
        world_parameters=parameters,
        role_assignments=roles,
        feature_membership=membership,
        typed_subspace_hyperedges=hyperedges,
        typed_graph=graph,
        truth_typed_edges=pair_truth,
    )
    if args.world_seed is not None and result.metrics["world_seed"] != args.world_seed:
        raise ValueError("Scorer world seed differs from the global freeze")
    _write_score(result, args.out_dir)
    print(json.dumps(_json_ready(result.metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_VALIDATION_SCHEMA_VERSION",
    "CANDIDATE_MANIFEST_SCHEMA_VERSION",
    "DESCRIPTIVE_DECOMPOSITION_FIELDS",
    "EXPECTED_DESCRIPTIVE_DECOMPOSITION",
    "EXPECTED_ERROR_CONTROL_METHOD",
    "FRESH_DEVELOPMENT_SEEDS",
    "FRESH_LOCKED_SEEDS",
    "FRESH_TRAIN_SEEDS",
    "FROZEN_DISCOVERY_CONFIG",
    "FROZEN_DISCOVERY_SOURCE_SHA256",
    "FROZEN_LOCAL_CANDIDATE_NAME",
    "FROZEN_LOCAL_CONFIG",
    "GLOBAL_FREEZE_SCHEMA_VERSION",
    "HYPEREDGE_COLUMNS",
    "MEMBERSHIP_COLUMNS",
    "PROTOCOL_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "REQUIRED_JOINT_EXECUTABLE_SOURCES",
    "SCORER_SCHEMA_VERSION",
    "SPLIT_SEEDS",
    "SharedStateSubspaceScore",
    "SharedStateSubspaceTruth",
    "build_subspace_truth",
    "canonical_sha256",
    "collect_joint_executable_source_sha256",
    "load_frozen_subspace_candidate",
    "make_global_pretruth_freeze",
    "score_shared_state_subspace",
    "validate_block_results",
    "validate_global_pretruth_freeze",
    "validation_protocol",
]
