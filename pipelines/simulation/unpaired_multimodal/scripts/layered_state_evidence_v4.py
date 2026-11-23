#!/usr/bin/env python3
"""Build an explicitly layered CA-BOSS v4 evidence graph.

This module is a conservative post-processor for *observable* CA-BOSS v4
candidate artifacts.  It does not turn every selected feature, loading group,
or direct-graph endpoint into the same kind of claim:

* a feature is a confirmatory state-response member only when it passes both
  the upstream hierarchical rule and the synchronized feature-global maxT
  test;
* assay participation is a separate Bonferroni/partial-conjunction omnibus
  call and never promotes its reporter or witnesses into feature membership;
* weaker upstream members remain conditional candidates, not confirmed
  members;
* coherent loading groups are retained as assay-local shared-state support;
* collider-motif edges remain within-assay direct candidates; and
* all remaining variables are unresolved.

The resulting graph is deliberately multi-label.  In particular, a collider
child may also be a confirmed state-response member.  The union of layers is
never represented as a state-membership set, and no cross-assay pair,
direction, or direct causal effect is emitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import empirical_null_state_calibration_v4 as empirical_null_state
import loading_group_competing_parent_observability_guard_v4 as loading_group_competitor_guard
import loading_group_state_guard_v4 as loading_group_state_guard
import loading_group_structure_equivalence_v4 as loading_group_structure_equivalence
import loading_group_structure_uncertainty_guard_v4 as loading_group_structure_guard
import multievidence_empirical_null_state_calibration_v4 as multievidence_state
import shared_state_subspace_v4 as state_subspace
import state_equivalence_membership_v4 as state_equivalence
import topology_consensus_state_calibration_v4 as topology_consensus_state


PROVENANCE_SCHEMA_VERSION = "layered_state_evidence_v4.v12"
CANDIDATE_SCHEMA_VERSION = "layered_state_evidence_v4.frozen_candidate.v12"
ALGORITHM_VERSION = "CA-BOSS v4.32"


_FORBIDDEN_PATH_PARTS = {
    "counterfactual",
    "ground_truth",
    "oracle",
    "scorer",
    "truth",
}

_MEMBERSHIP_REQUIRED = {
    "feature",
    "modality",
    "n_views",
    "formal_test_eligible",
    "empirical_null_p",
    "reference_conditional_wald",
    "conditional_wald_view_support",
    "feature_global_maxT_p",
    "pre_redundancy_selection_status",
    "redundancy_demoted",
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
    "selection_status",
}
_SUBSPACE_REQUIRED = {
    "subspace_id",
    "edge_type",
    "selection_status",
    "empirical_multimodality_p",
}
_CONSENSUS_REQUIRED = {
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "edge_type",
}
_GROUPS_REQUIRED = {
    "loading_group",
    "modality",
    "members",
    "group_confidence",
    "minimum_member_centroid_cosine",
    "selection_status",
}
_GROUP_MEMBERSHIP_REQUIRED = {"loading_group", "feature", "modality"}
_STRUCTURE_ANCHOR_REQUIRED = {
    "pair_key",
    "opposite_parent_pair_key",
    "modality",
    "parent_candidate",
    "noncollider_sibling_candidate",
    "collider_center",
    "other_parent_candidate",
    "selected",
    "selection_stage",
}


@dataclass(frozen=True)
class LayeredStateEvidenceConfig:
    """Frozen rules for assigning evidence to separate graph layers."""

    maximum_confirmatory_feature_maxT_p: float = 0.10
    maximum_state_equivalence_maxT_p: float = 0.01
    maximum_assay_participation_error_probability: float = 0.10
    minimum_confirmed_modalities: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_confirmatory_feature_maxT_p <= 0.10:
            raise ValueError(
                "maximum_confirmatory_feature_maxT_p must lie in (0, 0.10]"
            )
        if not 0.0 < self.maximum_state_equivalence_maxT_p <= 0.01:
            raise ValueError(
                "maximum_state_equivalence_maxT_p must lie in (0, 0.01]"
            )
        if not 0.0 < self.maximum_assay_participation_error_probability <= 0.10:
            raise ValueError(
                "maximum_assay_participation_error_probability must lie in "
                "(0, 0.10]"
            )
        if self.minimum_confirmed_modalities < 2:
            raise ValueError("minimum_confirmed_modalities must be at least two")


@dataclass
class LayeredStateEvidenceResult:
    feature_evidence: pd.DataFrame
    typed_hyperedges: pd.DataFrame
    within_assay_direct_candidates: pd.DataFrame
    provenance: dict[str, Any]


def _assert_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    invalid = sorted(set(normalized).difference(allowed))
    if invalid:
        raise ValueError(f"{label} contains invalid boolean values: {invalid}")
    return normalized.isin({"true", "1"})


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes_once(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Could not open {label} safely: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_empirical_null_state_calibrator(
    payload: bytes,
) -> tuple[
    empirical_null_state.FrozenEmpiricalNullStateCalibrator,
    dict[str, Any],
]:
    """Validate one frozen v4.25 artifact and expose only numeric application state."""

    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse empirical-null calibrator: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ValueError("Empirical-null calibrator must contain one JSON object")
    unsigned = dict(artifact)
    observed_payload_hash = str(
        unsigned.pop("artifact_payload_sha256", "")
    ).strip()
    expected_payload_hash = _canonical_sha256(unsigned)
    if observed_payload_hash != expected_payload_hash:
        raise ValueError("Empirical-null calibrator payload hash changed")
    required_metadata = {
        "artifact_role": "future_application_only",
        "application_requires_truth": False,
        "training_rows_are_not_validation_rows": True,
        "new_worlds_or_datasets_created": False,
        "exact_cross_assay_pair_claimed": False,
        "direct_causal_claimed": False,
        "pairwise_expansion_permitted": False,
    }
    mismatched = {
        key: artifact.get(key)
        for key, expected in required_metadata.items()
        if artifact.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            "Empirical-null calibrator violates its application boundary: "
            f"{mismatched}"
        )
    fields = set(
        empirical_null_state.FrozenEmpiricalNullStateCalibrator.__dataclass_fields__
    )
    missing = sorted(fields.difference(artifact))
    if missing:
        raise ValueError(f"Empirical-null calibrator is missing fields: {missing}")
    calibrator = empirical_null_state.FrozenEmpiricalNullStateCalibrator.from_payload(
        {field: artifact[field] for field in fields}
    )
    metadata = {
        "active": True,
        "artifact_payload_sha256": observed_payload_hash,
        "numeric_calibrator_payload_sha256": _canonical_sha256(
            calibrator.to_payload()
        ),
        "artifact_role": str(artifact["artifact_role"]),
        "phase_a_manifest_sha256": str(
            artifact.get("phase_a_manifest_sha256", "")
        ),
        "training_label_input_sha256": str(
            artifact.get("training_label_input_sha256", "")
        ),
        "application_requires_truth": False,
        "calibrator_training_used_simulation_null_labels": True,
    }
    return calibrator, metadata


def load_multievidence_state_calibrator(
    payload: bytes,
) -> tuple[
    multievidence_state.FrozenMultievidenceStateCalibrator,
    dict[str, Any],
]:
    """Validate one frozen v4.26 artifact and expose numeric application state."""

    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse multievidence calibrator: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ValueError("Multievidence calibrator must contain one JSON object")
    unsigned = dict(artifact)
    observed_payload_hash = str(
        unsigned.pop("artifact_payload_sha256", "")
    ).strip()
    expected_payload_hash = _canonical_sha256(unsigned)
    if observed_payload_hash != expected_payload_hash:
        raise ValueError("Multievidence calibrator payload hash changed")
    required_metadata = {
        "artifact_role": "future_application_only",
        "application_requires_truth": False,
        "training_rows_are_not_validation_rows": True,
        "new_simulation_worlds_or_datasets_created": False,
        "fixed_route_family_selected_without_application_labels": True,
        "exact_cross_assay_pair_claimed": False,
        "direct_causal_claimed": False,
        "pairwise_expansion_permitted": False,
    }
    mismatched = {
        key: artifact.get(key)
        for key, expected in required_metadata.items()
        if artifact.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            "Multievidence calibrator violates its application boundary: "
            f"{mismatched}"
        )
    fields = set(
        multievidence_state.FrozenMultievidenceStateCalibrator.__dataclass_fields__
    )
    missing = sorted(fields.difference(artifact))
    if missing:
        raise ValueError(f"Multievidence calibrator is missing fields: {missing}")
    calibrator = (
        multievidence_state.FrozenMultievidenceStateCalibrator.from_payload(
            {field: artifact[field] for field in fields}
        )
    )
    metadata = {
        "active": True,
        "artifact_payload_sha256": observed_payload_hash,
        "numeric_calibrator_payload_sha256": _canonical_sha256(
            calibrator.to_payload()
        ),
        "artifact_role": str(artifact["artifact_role"]),
        "phase_a_manifest_sha256": str(
            artifact.get("phase_a_manifest_sha256", "")
        ),
        "training_label_input_sha256": str(
            artifact.get("training_label_input_sha256", "")
        ),
        "application_requires_truth": False,
        "calibrator_training_used_simulation_null_labels": True,
        "fixed_route_family": list(multievidence_state.ROUTE_NAMES),
    }
    return calibrator, metadata


def load_topology_consensus_state_calibrator(
    payload: bytes,
) -> tuple[
    topology_consensus_state.FrozenTopologyConsensusStateCalibrator,
    dict[str, Any],
]:
    """Validate one frozen v4.27 artifact for truth-free application."""

    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not parse topology-consensus calibrator: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise ValueError(
            "Topology-consensus calibrator must contain one JSON object"
        )
    unsigned = dict(artifact)
    observed_payload_hash = str(
        unsigned.pop("artifact_payload_sha256", "")
    ).strip()
    if observed_payload_hash != _canonical_sha256(unsigned):
        raise ValueError("Topology-consensus calibrator payload hash changed")
    required_metadata = {
        "artifact_role": "future_application_only",
        "application_requires_truth": False,
        "training_rows_are_not_validation_rows": True,
        "new_simulation_worlds_or_datasets_created": False,
        "fixed_state_topology_conjunction_selected_without_application_labels": True,
        "exact_cross_assay_pair_claimed": False,
        "direct_causal_claimed": False,
        "pairwise_expansion_permitted": False,
    }
    mismatched = {
        key: artifact.get(key)
        for key, expected in required_metadata.items()
        if artifact.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            "Topology-consensus calibrator violates its application boundary: "
            f"{mismatched}"
        )
    fields = set(
        topology_consensus_state.FrozenTopologyConsensusStateCalibrator
        .__dataclass_fields__
    )
    missing = sorted(fields.difference(artifact))
    if missing:
        raise ValueError(
            f"Topology-consensus calibrator is missing fields: {missing}"
        )
    calibrator = (
        topology_consensus_state.FrozenTopologyConsensusStateCalibrator
        .from_payload({field: artifact[field] for field in fields})
    )
    if not calibrator.sentinel_certified:
        raise ValueError(
            "Topology-consensus calibrator is not sentinel certified"
        )
    metadata = {
        "active": True,
        "artifact_payload_sha256": observed_payload_hash,
        "numeric_calibrator_payload_sha256": _canonical_sha256(
            calibrator.to_payload()
        ),
        "artifact_role": str(artifact["artifact_role"]),
        "phase_a_manifest_sha256": str(
            artifact.get("phase_a_manifest_sha256", "")
        ),
        "training_label_input_sha256": str(
            artifact.get("training_label_input_sha256", "")
        ),
        "application_requires_truth": False,
        "calibrator_training_used_simulation_null_labels": True,
        "sentinel_certified": bool(calibrator.sentinel_certified),
        "fixed_state_topology_conjunction": True,
    }
    return calibrator, metadata


def _safe_observable_path(path: Path, label: str) -> Path:
    expanded = path.expanduser().absolute()
    if expanded.is_symlink() or not expanded.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = expanded.resolve(strict=True)
    forbidden = sorted(
        part for part in resolved.parts if part.lower() in _FORBIDDEN_PATH_PARTS
    )
    if forbidden:
        raise ValueError(f"{label} points into a forbidden scorer domain: {forbidden}")
    return resolved


def _normalize_direct_candidates(
    consensus_edges: pd.DataFrame,
    feature_to_modality: Mapping[str, str],
) -> pd.DataFrame:
    direct = consensus_edges.loc[
        consensus_edges["edge_type"].astype(str).eq(
            "within_assay_direct_candidate"
        )
    ].copy()
    direct = direct.sort_values(["node1", "node2"]).reset_index(drop=True)
    if direct.empty:
        direct.insert(0, "relation_id", pd.Series(dtype=str))
        direct.insert(1, "layer", pd.Series(dtype=str))
        return direct

    unknown = sorted(
        (set(direct["node1"].astype(str)) | set(direct["node2"].astype(str)))
        .difference(feature_to_modality)
    )
    if unknown:
        raise ValueError(f"Direct candidates contain unknown features: {unknown}")

    source_assays: list[str] = []
    for row in direct.itertuples(index=False):
        left = feature_to_modality[str(row.node1)]
        right = feature_to_modality[str(row.node2)]
        if left != right:
            raise ValueError(
                "A within_assay_direct_candidate joins different modalities: "
                f"{row.node1}, {row.node2}"
            )
        source_assays.append(left)
    direct.insert(
        0,
        "relation_id",
        [f"within_assay_direct_{index:03d}" for index in range(1, len(direct) + 1)],
    )
    direct.insert(1, "layer", "within_assay_direct_candidate")
    if "source_assay" not in direct.columns:
        direct.insert(2, "source_assay", source_assays)
    else:
        observed = direct["source_assay"].astype(str).tolist()
        if observed != source_assays:
            raise ValueError("Direct-candidate source_assay disagrees with membership")
    direct["cross_assay_claim"] = False
    direct["exact_axis_pairing_claimed"] = False
    return direct


def _annotate_empirical_null_state_layer(
    features: pd.DataFrame,
    calibrator: empirical_null_state.FrozenEmpiricalNullStateCalibrator | None,
) -> pd.DataFrame:
    """Add the v4.25 state tier without changing existing membership or edges."""

    output = features.copy()
    candidate = (
        output["formal_test_eligible"].astype(bool)
        & ~output["state_supported_member"].astype(bool)
    )
    if calibrator is None:
        output["empirical_null_calibration_active"] = False
        output["empirical_null_calibration_candidate"] = candidate
        output["empirical_null_marginal_state_z"] = 0.0
        output["empirical_null_conditional_excess_z"] = 0.0
        output["empirical_null_composite_score"] = 0.0
        output["empirical_null_selection_threshold"] = 0.0
        output["empirical_null_state_member"] = False
        output["state_calibrated_supported_member"] = output[
            "state_supported_member"
        ].astype(bool)
        output["state_calibrated_membership_tier"] = output[
            "state_membership_tier"
        ].astype(str)
        output["empirical_null_calibration_schema"] = ""
        output["empirical_null_threshold_method"] = ""
        return output

    application = output.loc[
        :,
        [
            "feature",
            "modality",
            "formal_test_eligible",
            "state_supported_member",
            "state_membership_tier",
            "oof_marginal_state_fit",
            "conditional_wald_above_null_95pct",
        ],
    ].copy()
    application.insert(0, "seed", 0)
    application.insert(0, "block", "production_application")
    annotated = empirical_null_state.annotate_empirical_null_state_membership(
        application,
        calibrator=calibrator,
    )
    if not annotated["feature"].astype(str).equals(output["feature"].astype(str)):
        raise RuntimeError("Empirical-null annotation changed feature order")
    if not annotated["modality"].astype(str).equals(
        output["modality"].astype(str)
    ):
        raise RuntimeError("Empirical-null annotation changed feature modalities")
    transfer_columns = (
        "empirical_null_calibration_candidate",
        "empirical_null_marginal_state_z",
        "empirical_null_conditional_excess_z",
        "empirical_null_composite_score",
        "empirical_null_selection_threshold",
        "empirical_null_state_member",
        "state_calibrated_supported_member",
        "state_calibrated_membership_tier",
        "empirical_null_calibration_schema",
        "empirical_null_threshold_method",
    )
    output["empirical_null_calibration_active"] = True
    for column in transfer_columns:
        output[column] = annotated[column].to_numpy(copy=True)
    for column in (
        "exact_cross_assay_pair_claimed",
        "direct_causal_claimed",
        "pairwise_expansion_permitted",
    ):
        if annotated[column].astype(bool).any():
            raise RuntimeError(
                f"Empirical-null state layer emitted unsupported claim: {column}"
            )
    return output


def _annotate_multievidence_state_layer(
    features: pd.DataFrame,
    calibrator: (
        multievidence_state.FrozenMultievidenceStateCalibrator | None
    ),
) -> pd.DataFrame:
    """Add v4.26 membership and stack it additively on the v4.25 union."""

    output = features.copy()
    candidate = (
        output["formal_test_eligible"].astype(bool)
        & ~output["state_supported_member"].astype(bool)
    )
    state_gate = (
        pd.to_numeric(output["oof_marginal_state_fit"], errors="raise").gt(0.0)
        & pd.to_numeric(output["oof_partial_state_fit"], errors="raise").gt(0.0)
    )
    if calibrator is None:
        output["multievidence_calibration_active"] = False
        output["multievidence_calibration_candidate"] = candidate
        output["multievidence_state_specific_gate"] = state_gate
        for column in multievidence_state.SCORE_COMPONENT_COLUMNS:
            output[f"multievidence_component_z__{column}"] = 0.0
        for route in multievidence_state.ROUTE_NAMES:
            output[f"multievidence_route_z__{route}"] = 0.0
        output["multievidence_winning_route"] = ""
        output["multievidence_family_score"] = 0.0
        output["multievidence_selection_threshold"] = 0.0
        output["multievidence_empirical_null_state_member"] = False
        output["state_multievidence_supported_member"] = output[
            "state_supported_member"
        ].astype(bool)
        output["state_multievidence_membership_tier"] = output[
            "state_membership_tier"
        ].astype(str)
        output["multievidence_calibration_schema"] = ""
        output["multievidence_threshold_method"] = ""
    else:
        application = output.loc[
            :,
            [
                "feature",
                "modality",
                "formal_test_eligible",
                "state_supported_member",
                "state_membership_tier",
                "oof_marginal_state_fit",
                "oof_partial_state_fit",
                "oof_joint_fit",
                "conditional_wald_above_null_95pct",
                "median_conditional_score_norm",
            ],
        ].copy()
        application.insert(0, "seed", 0)
        application.insert(0, "block", "production_application")
        annotated = multievidence_state.annotate_multievidence_state_membership(
            application, calibrator=calibrator
        )
        if not annotated["feature"].astype(str).equals(
            output["feature"].astype(str)
        ):
            raise RuntimeError("Multievidence annotation changed feature order")
        if not annotated["modality"].astype(str).equals(
            output["modality"].astype(str)
        ):
            raise RuntimeError("Multievidence annotation changed feature modalities")
        transfer_columns = (
            "multievidence_calibration_candidate",
            "multievidence_state_specific_gate",
            *(
                f"multievidence_component_z__{column}"
                for column in multievidence_state.SCORE_COMPONENT_COLUMNS
            ),
            *(
                f"multievidence_route_z__{route}"
                for route in multievidence_state.ROUTE_NAMES
            ),
            "multievidence_winning_route",
            "multievidence_family_score",
            "multievidence_selection_threshold",
            "multievidence_empirical_null_state_member",
            "state_multievidence_supported_member",
            "state_multievidence_membership_tier",
            "multievidence_calibration_schema",
            "multievidence_threshold_method",
        )
        output["multievidence_calibration_active"] = True
        for column in transfer_columns:
            output[column] = annotated[column].to_numpy(copy=True)
        for column in (
            "exact_cross_assay_pair_claimed",
            "direct_causal_claimed",
            "pairwise_expansion_permitted",
        ):
            if annotated[column].astype(bool).any():
                raise RuntimeError(
                    f"Multievidence state layer emitted unsupported claim: {column}"
                )

    addition = output["multievidence_empirical_null_state_member"].astype(bool)
    upstream = output["state_calibrated_supported_member"].astype(bool)
    output["state_v426_supported_member"] = upstream | addition
    output["state_v426_membership_tier"] = output[
        "state_calibrated_membership_tier"
    ].astype(str)
    output.loc[addition & ~upstream, "state_v426_membership_tier"] = (
        multievidence_state.TIER
    )
    if (upstream & ~output["state_v426_supported_member"]).any():
        raise RuntimeError("v4.26 removed an upstream v4.25 state member")
    if (
        output["state_v426_supported_member"]
        & ~upstream
        & ~addition
    ).any():
        raise RuntimeError("v4.26 stacked union contains an unsupported promotion")
    return output


def _annotate_topology_consensus_state_layer(
    features: pd.DataFrame,
    calibrator: (
        topology_consensus_state.FrozenTopologyConsensusStateCalibrator | None
    ),
) -> pd.DataFrame:
    """Add v4.27 joint weak-state/residual-topology support."""

    output = features.copy()
    upstream = output["state_v426_supported_member"].astype(bool)
    candidate = output["formal_test_eligible"].astype(bool) & ~upstream
    state_gate = (
        pd.to_numeric(output["oof_marginal_state_fit"], errors="raise").gt(0.0)
        & pd.to_numeric(output["oof_partial_state_fit"], errors="raise").gt(0.0)
    )
    residual_support = pd.to_numeric(
        output["within_assay_direct_candidate_degree"], errors="raise"
    ).ge(topology_consensus_state.MINIMUM_DIRECT_CANDIDATE_DEGREE)
    if calibrator is None:
        output["topology_consensus_calibration_active"] = False
        output["topology_consensus_calibration_candidate"] = candidate
        output["topology_consensus_state_specific_gate"] = state_gate
        output["topology_consensus_residual_structure_support"] = residual_support
        output["topology_consensus_fixed_rule_eligible"] = (
            output["formal_test_eligible"].astype(bool)
            & state_gate
            & residual_support
        )
        output["topology_consensus_empirical_null_certified"] = False
        output["topology_consensus_state_member"] = False
        output["state_v427_supported_member"] = upstream
        output["state_v427_membership_tier"] = output[
            "state_v426_membership_tier"
        ].astype(str)
        output["topology_consensus_calibration_schema"] = ""
        output["topology_consensus_null_control_method"] = ""
        return output

    application = output.loc[
        :,
        [
            "feature",
            "modality",
            "formal_test_eligible",
            "oof_marginal_state_fit",
            "oof_partial_state_fit",
            "within_assay_direct_candidate_degree",
        ],
    ].copy()
    application.insert(0, "seed", 0)
    application.insert(0, "block", "production_application")
    annotated = (
        topology_consensus_state.annotate_topology_consensus_state_membership(
            application,
            upstream_supported=upstream,
            upstream_tier=output["state_v426_membership_tier"],
            calibrator=calibrator,
        )
    )
    if not annotated["feature"].astype(str).equals(
        output["feature"].astype(str)
    ):
        raise RuntimeError("Topology-consensus annotation changed feature order")
    if not annotated["modality"].astype(str).equals(
        output["modality"].astype(str)
    ):
        raise RuntimeError(
            "Topology-consensus annotation changed feature modalities"
        )
    transfer_columns = (
        "topology_consensus_calibration_active",
        "topology_consensus_calibration_candidate",
        "topology_consensus_state_specific_gate",
        "topology_consensus_residual_structure_support",
        "topology_consensus_fixed_rule_eligible",
        "topology_consensus_empirical_null_certified",
        "topology_consensus_state_member",
        "state_v427_supported_member",
        "state_v427_membership_tier",
        "topology_consensus_calibration_schema",
        "topology_consensus_null_control_method",
    )
    for column in transfer_columns:
        output[column] = annotated[column].to_numpy(copy=True)
    for column in (
        "exact_cross_assay_pair_claimed",
        "direct_causal_claimed",
        "pairwise_expansion_permitted",
    ):
        if annotated[column].astype(bool).any():
            raise RuntimeError(
                f"Topology-consensus layer emitted unsupported claim: {column}"
            )
    if (upstream & ~output["state_v427_supported_member"]).any():
        raise RuntimeError("v4.27 removed an upstream v4.26 state member")
    addition = output["topology_consensus_state_member"].astype(bool)
    if (
        output["state_v427_supported_member"]
        & ~upstream
        & ~addition
    ).any():
        raise RuntimeError("v4.27 stacked union contains an unsupported promotion")
    return output


def build_layered_state_evidence(
    membership: pd.DataFrame,
    subspace_hyperedges: pd.DataFrame,
    consensus_edges: pd.DataFrame,
    loading_groups: pd.DataFrame,
    loading_group_membership: pd.DataFrame,
    structure_anchor_pair_evidence: pd.DataFrame,
    config: LayeredStateEvidenceConfig | None = None,
    empirical_null_calibrator: (
        empirical_null_state.FrozenEmpiricalNullStateCalibrator | None
    ) = None,
    multievidence_calibrator: (
        multievidence_state.FrozenMultievidenceStateCalibrator | None
    ) = None,
    topology_consensus_calibrator: (
        topology_consensus_state.FrozenTopologyConsensusStateCalibrator | None
    ) = None,
    group_axis_evidence: pd.DataFrame | None = None,
    axis_candidates: pd.DataFrame | None = None,
) -> LayeredStateEvidenceResult:
    """Assign observable candidate evidence to non-interchangeable layers."""

    config = config or LayeredStateEvidenceConfig()
    _assert_columns(membership, _MEMBERSHIP_REQUIRED, "membership")
    _assert_columns(subspace_hyperedges, _SUBSPACE_REQUIRED, "subspace_hyperedges")
    _assert_columns(consensus_edges, _CONSENSUS_REQUIRED, "consensus_edges")
    _assert_columns(loading_groups, _GROUPS_REQUIRED, "loading_groups")
    _assert_columns(
        loading_group_membership,
        _GROUP_MEMBERSHIP_REQUIRED,
        "loading_group_membership",
    )
    _assert_columns(
        structure_anchor_pair_evidence,
        _STRUCTURE_ANCHOR_REQUIRED,
        "structure_anchor_pair_evidence",
    )

    features = membership.copy()
    features["feature"] = features["feature"].astype(str)
    features["modality"] = features["modality"].astype(str)
    if features["feature"].duplicated().any():
        duplicate = sorted(features.loc[features["feature"].duplicated(), "feature"])
        raise ValueError(f"membership has duplicate features: {duplicate}")
    if features[["feature", "modality"]].isna().any().any():
        raise ValueError("membership contains missing feature/modality identifiers")

    features["formal_test_eligible"] = _as_bool(
        features["formal_test_eligible"], "formal_test_eligible"
    )
    for column in (
        "empirical_null_p",
        "reference_conditional_wald",
        "conditional_wald_view_support",
    ):
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
    if not features["empirical_null_p"].between(0.0, 1.0).all() or not features[
        "conditional_wald_view_support"
    ].between(0.0, 1.0).all():
        raise ValueError("Empirical p-values/view support must lie in [0, 1]")
    features["feature_global_maxT_p"] = pd.to_numeric(
        features["feature_global_maxT_p"], errors="raise"
    )
    if not features["feature_global_maxT_p"].between(0.0, 1.0).all():
        raise ValueError("feature_global_maxT_p must lie in [0, 1]")

    participation_boolean_columns = (
        "assay_participation_selected",
        "assay_participation_multiview_replication_gate",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_reporter",
    )
    for column in participation_boolean_columns:
        features[column] = _as_bool(features[column], column)
    participation_probability_columns = (
        "assay_participation_partial_conjunction_p",
        "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p",
        "assay_participation_single_feature_p",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
    )
    for column in participation_probability_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not features[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must lie in [0, 1]")
    participation_count_columns = (
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_single_feature_witness_view_count",
        "assay_participation_omnibus_route_count",
    )
    for column in participation_count_columns:
        numeric = pd.to_numeric(features[column], errors="raise")
        if (numeric < 0).any() or not (numeric == numeric.round()).all():
            raise ValueError(f"{column} must contain nonnegative integers")
        features[column] = numeric.astype(int)

    assay_participation_rows: list[dict[str, Any]] = []
    constant_participation_columns = (
        *participation_probability_columns,
        "assay_participation_selected",
        "assay_participation_order",
        "assay_participation_family_size",
        "assay_participation_contributing_feature_count",
        "assay_participation_minimum_witness_view_count",
        "assay_participation_multiview_replication_gate",
        "assay_participation_single_feature_multiview_replication_gate",
        "assay_participation_partial_conjunction_multiview_replication_gate",
        "assay_participation_omnibus_route_count",
        "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p",
        "assay_participation_winning_route",
        "assay_participation_selection_route",
    )
    recomputed_participation = state_subspace._select_assay_state_participation(
        features,
        state_subspace.SharedStateSubspaceConfig(
            maximum_assay_state_participation_error_probability=(
                config.maximum_assay_participation_error_probability
            )
        ),
    ).set_index("modality")
    for modality, group in features.groupby("modality", sort=True):
        inconsistent = [
            column
            for column in constant_participation_columns
            if group[column].nunique(dropna=False) != 1
        ]
        if inconsistent:
            raise ValueError(
                "Assay participation evidence must be constant within a modality: "
                f"{modality}: {inconsistent}"
            )
        first = group.iloc[0]
        order = int(first["assay_participation_order"])
        family_size = int(first["assay_participation_family_size"])
        contributing = int(
            first["assay_participation_contributing_feature_count"]
        )
        replicated = bool(
            first["assay_participation_multiview_replication_gate"]
        )
        single_replicated = bool(
            first[
                "assay_participation_single_feature_multiview_replication_gate"
            ]
        )
        partial_replicated = bool(
            first[
                "assay_participation_partial_conjunction_multiview_"
                "replication_gate"
            ]
        )
        selected_assay = bool(first["assay_participation_selected"])
        n_views_values = pd.to_numeric(group["n_views"], errors="raise")
        if (
            n_views_values.nunique(dropna=False) != 1
            or int(n_views_values.iloc[0])
            != state_subspace.SharedStateSubspaceConfig().observed_model_views
        ):
            raise ValueError(
                f"n_views must equal the frozen six-view contract: {modality}"
            )
        n_views = int(n_views_values.iloc[0])
        expected_family_size = int(
            group["formal_test_eligible"].astype(bool).sum()
        )
        expected_order = 1 if len(group) == 1 else 2
        expected_contributing = (
            expected_order if expected_family_size >= expected_order else 0
        )
        if family_size != expected_family_size:
            raise ValueError(
                f"Assay participation family size is inconsistent for {modality}"
            )
        if order != expected_order:
            raise ValueError(
                f"Assay participation order is inconsistent for {modality}"
            )
        if contributing != expected_contributing:
            raise ValueError(
                "Assay participation contributing-feature count is inconsistent "
                f"for {modality}"
            )
        minimum_views = int(
            first["assay_participation_minimum_witness_view_count"]
        )
        single_views = int(
            first[
                "assay_participation_single_feature_witness_view_count"
            ]
        )
        if (
            minimum_views > n_views
            or single_views > n_views
            or replicated != (minimum_views == n_views)
            or partial_replicated != replicated
            or single_replicated != (single_views == n_views)
        ):
            raise ValueError(
                f"Assay participation replication evidence is inconsistent for {modality}"
            )
        expected = recomputed_participation.loc[str(modality)]
        expected_selected = str(expected["selection_status"]) == "selected"
        numeric_contract = {
            "assay_participation_partial_conjunction_p": (
                "partial_conjunction_empirical_p"
            ),
            "assay_participation_holm_p": "assay_participation_holm_p",
            "assay_participation_single_feature_order_statistic_p": (
                "single_feature_order_statistic_p"
            ),
            "assay_participation_single_feature_p": (
                "single_feature_bonferroni_p"
            ),
            "assay_participation_omnibus_p": (
                "assay_participation_omnibus_p"
            ),
            "assay_participation_omnibus_holm_p": (
                "assay_participation_omnibus_holm_p"
            ),
        }
        for observed_column, expected_column in numeric_contract.items():
            if not np.isclose(
                float(first[observed_column]),
                float(expected[expected_column]),
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    "Assay participation p-value is inconsistent for "
                    f"{modality}: {observed_column}"
                )
        exact_contract = {
            "assay_participation_single_feature_witness_view_count": (
                "single_feature_witness_view_count"
            ),
            "assay_participation_single_feature_multiview_replication_gate": (
                "single_feature_multiview_replication_gate"
            ),
            "assay_participation_partial_conjunction_multiview_replication_gate": (
                "partial_conjunction_multiview_replication_gate"
            ),
            "assay_participation_omnibus_route_count": (
                "assay_participation_omnibus_route_count"
            ),
            "assay_participation_winning_route": (
                "assay_participation_winning_route"
            ),
            "assay_participation_selection_route": (
                "assay_participation_selection_route"
            ),
        }
        for observed_column, expected_column in exact_contract.items():
            if first[observed_column] != expected[expected_column]:
                raise ValueError(
                    "Assay participation route evidence is inconsistent for "
                    f"{modality}: {observed_column}"
                )
        if selected_assay != expected_selected:
            raise ValueError(
                f"Assay participation selection is inconsistent for {modality}"
            )
        reporter_rows = group.loc[group["assay_participation_reporter"]]
        if len(reporter_rows) != int(selected_assay):
            raise ValueError(
                "A selected assay must have exactly one non-membership reporter "
                f"and an unresolved assay none: {modality}"
            )
        observed_reporter = (
            str(reporter_rows.iloc[0]["feature"])
            if selected_assay
            else ""
        )
        if observed_reporter != str(expected["selected_reporter"]):
            raise ValueError(
                f"Assay participation reporter is inconsistent for {modality}"
            )
        assay_participation_rows.append(
            {
                "modality": str(modality),
                "selected": selected_assay,
                "partial_conjunction_p": float(
                    first["assay_participation_partial_conjunction_p"]
                ),
                "holm_p": float(first["assay_participation_holm_p"]),
                "single_feature_order_statistic_p": float(
                    first[
                        "assay_participation_single_feature_order_statistic_p"
                    ]
                ),
                "single_feature_p": float(
                    first["assay_participation_single_feature_p"]
                ),
                "single_feature_witness_view_count": single_views,
                "single_feature_multiview_replication_gate": (
                    single_replicated
                ),
                "partial_conjunction_multiview_replication_gate": (
                    partial_replicated
                ),
                "omnibus_route_count": int(
                    first["assay_participation_omnibus_route_count"]
                ),
                "omnibus_p": float(
                    first["assay_participation_omnibus_p"]
                ),
                "omnibus_holm_p": float(
                    first["assay_participation_omnibus_holm_p"]
                ),
                "winning_route": str(
                    first["assay_participation_winning_route"]
                ),
                "selection_route": str(
                    first["assay_participation_selection_route"]
                ),
                "order": order,
                "family_size": family_size,
                "contributing_feature_count": contributing,
                "minimum_witness_view_count": int(
                    first["assay_participation_minimum_witness_view_count"]
                ),
                "multiview_replication_gate": replicated,
                "reporter": observed_reporter,
            }
        )
    assay_participation = pd.DataFrame(assay_participation_rows)

    feature_to_modality = dict(zip(features["feature"], features["modality"]))
    upstream_selected = features["selection_status"].astype(str).eq("selected")
    confirmed = (
        upstream_selected
        & features["formal_test_eligible"]
        & features["feature_global_maxT_p"].le(
            config.maximum_confirmatory_feature_maxT_p
        )
    )
    features["upstream_conditional_state_candidate"] = upstream_selected
    features["confirmed_depth_orthogonal_state_response"] = confirmed
    features = state_equivalence.annotate_state_equivalence_membership(
        features,
        config=state_equivalence.StateEquivalenceConfig(
            maximum_feature_global_maxT_p=(
                config.maximum_state_equivalence_maxT_p
            )
        ),
        strict_membership=confirmed,
    )
    if not features["strict_state_member"].equals(confirmed.astype(bool)):
        raise RuntimeError(
            "State-equivalence annotation changed strict feature membership"
        )
    state_equivalence_groups = state_equivalence.build_state_equivalence_groups(
        features
    )
    features = _annotate_empirical_null_state_layer(
        features,
        empirical_null_calibrator,
    )
    features = _annotate_multievidence_state_layer(
        features,
        multievidence_calibrator,
    )
    features["state_response_status"] = "unresolved"
    features.loc[
        upstream_selected, "state_response_status"
    ] = "conditional_state_response_candidate_not_featurewise_confirmed"
    features.loc[
        confirmed, "state_response_status"
    ] = "confirmed_depth_orthogonal_state_response"
    features.loc[
        features["empirical_null_state_member"], "state_response_status"
    ] = "within_assay_empirical_null_state_member"
    features.loc[
        features["multievidence_empirical_null_state_member"],
        "state_response_status",
    ] = "within_assay_multievidence_empirical_null_state_member"

    groups = loading_groups.copy()
    groups["loading_group"] = groups["loading_group"].astype(str)
    group_membership = loading_group_membership.copy()
    group_membership["loading_group"] = group_membership["loading_group"].astype(str)
    group_membership["feature"] = group_membership["feature"].astype(str)
    group_membership["modality"] = group_membership["modality"].astype(str)
    if group_membership.duplicated(["loading_group", "feature"]).any():
        raise ValueError("loading_group_membership contains duplicate memberships")
    unknown_group_members = sorted(
        set(group_membership["feature"]).difference(feature_to_modality)
    )
    if unknown_group_members:
        raise ValueError(
            f"loading groups contain unknown features: {unknown_group_members}"
        )
    for row in group_membership.itertuples(index=False):
        if feature_to_modality[str(row.feature)] != str(row.modality):
            raise ValueError(
                f"Loading-group modality disagrees for feature {row.feature}"
            )

    selected_status_group_ids = set(
        groups.loc[
            groups["selection_status"].astype(str).eq("selected"), "loading_group"
        ].astype(str)
    )
    singleton_group_ids = set(
        groups.loc[
            groups["selection_status"].astype(str).eq("singleton_candidate"),
            "loading_group",
        ].astype(str)
    )
    group_sizes = group_membership.groupby("loading_group")["feature"].nunique()
    # RNA is represented upstream by a selected one-feature loading record.
    # It is useful feature evidence but is not a within-assay *shared* loading
    # group, so only selected groups with at least two observed members enter
    # this separate layer.
    pre_v428_selected_group_ids = {
        group_id
        for group_id in selected_status_group_ids
        if int(group_sizes.get(group_id, 0)) >= 2
    }
    pre_v428_selected_membership = group_membership.loc[
        group_membership["loading_group"].isin(pre_v428_selected_group_ids)
    ]
    singleton_membership = group_membership.loc[
        group_membership["loading_group"].isin(singleton_group_ids)
    ]
    pre_v428_selected_group_map = (
        pre_v428_selected_membership.groupby("feature")["loading_group"]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .to_dict()
    )
    singleton_group_map = (
        singleton_membership.groupby("feature")["loading_group"]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .to_dict()
    )
    features["assay_local_loading_group_pre_v428_ids"] = features[
        "feature"
    ].map(
        pre_v428_selected_group_map
    ).fillna("")
    features["assay_local_loading_group_pre_v428_support"] = features[
        "assay_local_loading_group_pre_v428_ids"
    ].ne("")
    features["assay_local_singleton_candidate_ids"] = features["feature"].map(
        singleton_group_map
    ).fillna("")
    features["assay_local_singleton_candidate"] = features[
        "assay_local_singleton_candidate_ids"
    ].ne("")

    direct = _normalize_direct_candidates(consensus_edges, feature_to_modality)
    direct_degree: dict[str, int] = {}
    direct_center: set[str] = set()
    for row in direct.itertuples(index=False):
        left, right = str(row.node1), str(row.node2)
        direct_degree[left] = direct_degree.get(left, 0) + 1
        direct_degree[right] = direct_degree.get(right, 0) + 1
        center = getattr(row, "collider_motif_center", "")
        if str(center) and str(center).lower() != "nan":
            direct_center.add(str(center))
        else:
            if str(row.endpoint1).upper() == "ARROW":
                direct_center.add(left)
            if str(row.endpoint2).upper() == "ARROW":
                direct_center.add(right)
    features["within_assay_direct_candidate_degree"] = (
        features["feature"].map(direct_degree).fillna(0).astype(int)
    )
    features["within_assay_direct_candidate_overlap"] = features[
        "within_assay_direct_candidate_degree"
    ].gt(0)
    features["within_assay_direct_role"] = "none"
    direct_mask = features["within_assay_direct_candidate_overlap"]
    features.loc[direct_mask, "within_assay_direct_role"] = "parent_candidate"
    features.loc[
        features["feature"].isin(direct_center), "within_assay_direct_role"
    ] = "collider_center_candidate"
    features = _annotate_topology_consensus_state_layer(
        features,
        topology_consensus_calibrator,
    )
    features.loc[
        features["topology_consensus_state_member"],
        "state_response_status",
    ] = topology_consensus_state.TIER

    v429_loading_guard = loading_group_state_guard.apply_loading_group_state_guard(
        groups,
        group_membership,
        features,
        group_axis_evidence,
        axis_candidates,
    )
    v430_loading_guard = (
        loading_group_structure_guard
        .apply_loading_group_structure_uncertainty_guard(
            v429_loading_guard.decisions,
            groups,
            direct,
        )
    )
    loading_guard = (
        loading_group_competitor_guard
        .apply_loading_group_competing_parent_observability_guard(
            v430_loading_guard.decisions,
            structure_anchor_pair_evidence,
            features,
        )
    )
    loading_structure_equivalence = (
        loading_group_structure_equivalence
        .build_loading_group_structure_equivalence(
            loading_guard.decisions.loc[
                :, sorted(loading_group_structure_equivalence.REQUIRED_COLUMNS)
            ]
        )
    )
    selected_group_ids = set(
        loading_structure_equivalence.selected_hypotheses[
            "loading_group"
        ].astype(str)
    )
    rejected_group_ids = loading_guard.rejected_group_ids
    selected_membership = group_membership.loc[
        group_membership["loading_group"].isin(selected_group_ids)
    ]
    rejected_membership = group_membership.loc[
        group_membership["loading_group"].isin(rejected_group_ids)
    ]
    selected_group_map = (
        selected_membership.groupby("feature")["loading_group"]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .to_dict()
    )
    rejected_group_map = (
        rejected_membership.groupby("feature")["loading_group"]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .to_dict()
    )
    features["assay_local_loading_group_ids"] = features["feature"].map(
        selected_group_map
    ).fillna("")
    features["assay_local_loading_group_rejected_ids"] = features[
        "feature"
    ].map(rejected_group_map).fillna("")
    features["assay_local_loading_group_support"] = features[
        "assay_local_loading_group_ids"
    ].ne("")
    features["assay_local_loading_group_guard_status"] = (
        "not_pre_v431_loading_group_member"
    )
    rejected_mask = features["assay_local_loading_group_rejected_ids"].ne("")
    retained_mask = features["assay_local_loading_group_support"]
    features.loc[
        rejected_mask,
        "assay_local_loading_group_guard_status",
    ] = "unresolved_by_v431_loading_group_guard"
    retained_state_ids = set(
        loading_guard.decisions.loc[
            loading_guard.decisions["guard_route"].astype(str).eq(
                loading_group_state_guard.STATE_ROUTE
            ),
            "loading_group",
        ].astype(str)
    )
    retained_axis_ids = set(
        loading_guard.decisions.loc[
            loading_guard.decisions["guard_route"].astype(str).eq(
                loading_group_state_guard.AXIS_ROUTE
            ),
            "loading_group",
        ].astype(str)
    )
    retained_state_members = set(
        group_membership.loc[
            group_membership["loading_group"].isin(retained_state_ids),
            "feature",
        ].astype(str)
    )
    retained_axis_members = set(
        group_membership.loc[
            group_membership["loading_group"].isin(retained_axis_ids),
            "feature",
        ].astype(str)
    )
    features.loc[
        features["feature"].isin(retained_state_members),
        "assay_local_loading_group_guard_status",
    ] = loading_group_competitor_guard.STATE_RETAINED_STATUS
    features.loc[
        features["feature"].isin(retained_axis_members),
        "assay_local_loading_group_guard_status",
    ] = loading_group_competitor_guard.AXIS_RETAINED_STATUS
    features.loc[
        retained_mask & rejected_mask,
        "assay_local_loading_group_guard_status",
    ] = "retained_in_one_group_and_rejected_in_another"
    features["assay_local_state_support_status"] = "none"
    features.loc[
        features["assay_local_singleton_candidate"],
        "assay_local_state_support_status",
    ] = "singleton_candidate_not_group_support"
    features.loc[
        rejected_mask,
        "assay_local_state_support_status",
    ] = "loading_group_unresolved_by_v431_fair_competitor_guard"
    features.loc[
        retained_mask,
        "assay_local_state_support_status",
    ] = "selected_state_certified_assay_local_loading_group"
    features.loc[
        features["feature"].isin(retained_axis_members),
        "assay_local_state_support_status",
    ] = "selected_cross_assay_axis_consensus_loading_group"

    def support_layers(row: pd.Series) -> str:
        layers: list[str] = []
        if bool(row["confirmed_depth_orthogonal_state_response"]):
            layers.append("confirmed_depth_orthogonal_state_response")
        elif bool(row["state_equivalence_only_member"]):
            layers.append("within_assay_state_equivalence_member")
        elif bool(row["empirical_null_state_member"]):
            layers.append("within_assay_empirical_null_state_member")
        elif bool(row["multievidence_empirical_null_state_member"]):
            layers.append(
                "within_assay_multievidence_empirical_null_state_member"
            )
        elif bool(row["topology_consensus_state_member"]):
            layers.append(
                "within_assay_topology_consensus_state_member"
            )
        elif bool(row["upstream_conditional_state_candidate"]):
            layers.append("conditional_state_response_candidate")
        if bool(row["assay_local_loading_group_support"]):
            layers.append("assay_local_loading_group_support")
        elif bool(row["assay_local_singleton_candidate"]):
            layers.append("assay_local_singleton_candidate")
        if bool(row["within_assay_direct_candidate_overlap"]):
            layers.append("within_assay_direct_candidate")
        return ";".join(layers) if layers else "unresolved"

    features["support_layers"] = features.apply(support_layers, axis=1)
    features["layer_union_is_state_membership"] = False
    features["assay_participation_implies_feature_membership"] = False
    features["cross_assay_pair_claimed"] = False
    features["direct_cross_assay_effect_claimed"] = False
    features = features.sort_values(["modality", "feature"]).reset_index(drop=True)

    hyperedge_rows: list[dict[str, Any]] = []
    confirmed_features = features.loc[
        features["confirmed_depth_orthogonal_state_response"]
    ]
    confirmed_modalities = sorted(set(confirmed_features["modality"].astype(str)))
    participating_modalities = sorted(
        assay_participation.loc[
            assay_participation["selected"], "modality"
        ].astype(str)
    )
    hyperedge_member_features = confirmed_features.loc[
        confirmed_features["modality"].astype(str).isin(
            participating_modalities
        )
    ]
    source_subspace = subspace_hyperedges.loc[
        subspace_hyperedges["selection_status"].astype(str).eq("selected")
        & subspace_hyperedges["edge_type"].astype(str).eq(
            "cross_assay_shared_state_subspace"
        )
    ]
    if (
        not source_subspace.empty
        and len(participating_modalities) >= config.minimum_confirmed_modalities
    ):
        source_p = float(
            pd.to_numeric(
                source_subspace["empirical_multimodality_p"], errors="raise"
            ).min()
        )
        maximum_selected_holm_p = float(
            assay_participation.loc[
                assay_participation["selected"], "omnibus_holm_p"
            ].max()
        )
        hyperedge_rows.append(
            {
                "relation_id": "confirmed_shared_state_subspace_001",
                "edge_type": "cross_assay_shared_state_subspace",
                "layer": "confirmed_depth_orthogonal_state_response",
                "modalities": ";".join(participating_modalities),
                "members": ";".join(
                    sorted(hyperedge_member_features["feature"].astype(str))
                ),
                "n_modalities": len(participating_modalities),
                "n_members": len(hyperedge_member_features),
                "selection_status": "selected",
                "evidence_basis": (
                    "replication_gated_assay_omnibus_modalities_with_Holm_"
                    "FWER_and_strict_feature_global_maxT_member_"
                    "annotations;confidence_is_one_minus_max_selected_Holm_p"
                ),
                "confidence": 1.0 - maximum_selected_holm_p,
                "empirical_p": source_p,
                "conditional_on_common_state_model": True,
                "hypothesis_only": True,
                "direct_causal_claimed": False,
                "direction_claimed": False,
                "exact_axis_pairing_claimed": False,
                "strict_members": ";".join(
                    sorted(hyperedge_member_features["feature"].astype(str))
                ),
                "equivalence_only_members": "",
                "pairwise_expansion_permitted": False,
                "exact_cross_assay_pair_claimed": False,
                "interpretation": (
                    "replicated assay-local state participation pooled as a "
                    "common-state hypothesis; members are only strict feature "
                    "annotations, with no feature pairing or direction"
                ),
            }
        )

    pre_v428_selected_groups = groups.loc[
        groups["loading_group"].isin(pre_v428_selected_group_ids)
    ].sort_values(["modality", "loading_group"])
    selected_groups = pre_v428_selected_groups.loc[
        pre_v428_selected_groups["loading_group"].isin(selected_group_ids)
    ]
    guard_by_group = loading_guard.decisions.set_index("loading_group")
    equivalence_by_group = (
        loading_structure_equivalence.decisions.set_index("loading_group")
    )
    for row in pre_v428_selected_groups.itertuples(index=False):
        nominal_members = sorted(
            group_membership.loc[
                group_membership["loading_group"].eq(str(row.loading_group)),
                "feature",
            ].astype(str)
        )
        if len(nominal_members) < 2:
            raise ValueError(
                f"Selected loading group {row.loading_group} has fewer than two members"
            )
        guard_row = guard_by_group.loc[str(row.loading_group)]
        equivalence_row = equivalence_by_group.loc[str(row.loading_group)]
        retained = bool(guard_row["retained"])
        guard_route = str(guard_row["guard_route"])
        guard_status = str(guard_row["guard_status"])
        axis_retained = guard_route == loading_group_state_guard.AXIS_ROUTE
        structure_equivalence_selected = bool(
            equivalence_row["structure_equivalence_selected"]
        )
        exact_loading_group_selected = bool(
            equivalence_row["exact_loading_group_selected"]
        )
        if structure_equivalence_selected:
            members = sorted(
                {
                    str(guard_row["selected_parent_candidate"]),
                    str(guard_row["competing_parent_candidate"]),
                    str(guard_row["noncollider_sibling_candidate"]),
                    str(guard_row["collider_child_candidate"]),
                }
            )
            missing_equivalence_nodes = sorted(
                set(members).difference(features["feature"].astype(str))
            )
            if missing_equivalence_nodes:
                raise ValueError(
                    "State-structure equivalence contains nodes outside the "
                    f"feature table: {missing_equivalence_nodes}"
                )
        else:
            members = nominal_members
        if retained and guard_route not in {
            loading_group_state_guard.STATE_ROUTE,
            loading_group_state_guard.AXIS_ROUTE,
        }:
            raise RuntimeError(
                f"Retained loading group has an unknown route: {guard_route}"
            )
        hyperedge_rows.append(
            {
                "relation_id": f"assay_local__{row.loading_group}",
                "edge_type": (
                    loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE
                    if structure_equivalence_selected
                    else "within_assay_shared_state_support"
                ),
                "layer": (
                    "assay_local_state_structure_equivalence"
                    if structure_equivalence_selected
                    else (
                        "assay_local_loading_group_support"
                        if retained
                        else "unresolved"
                    )
                ),
                "modalities": str(row.modality),
                "members": ";".join(members),
                "n_modalities": 1,
                "n_members": len(members),
                "selection_status": "selected" if retained else "unresolved",
                "evidence_basis": (
                    "v431_fair_selected_competing_parent_comparison_emitted_"
                    "as_unexpanded_two_by_two_state_structure_equivalence"
                    if structure_equivalence_selected
                    else (
                        "leave_one_group_out_cross_assay_axis_consensus_with_"
                        "at_least_one_state_v427_supported_member_and_no_"
                        "direct_layer_conflict"
                        if axis_retained
                        else guard_status
                    )
                ),
                "confidence": float(row.group_confidence),
                "empirical_p": float("nan"),
                "conditional_on_common_state_model": retained,
                "hypothesis_only": True,
                "direct_causal_claimed": False,
                "direction_claimed": False,
                "exact_axis_pairing_claimed": False,
                "strict_members": "",
                "equivalence_only_members": "",
                "pairwise_expansion_permitted": False,
                "exact_cross_assay_pair_claimed": False,
                "loading_group_v431_guard_route": guard_route,
                "loading_group_v431_guard_status": guard_status,
                "loading_group_v431_retained": retained,
                "structure_equivalence_schema": str(
                    equivalence_row["structure_equivalence_schema"]
                ),
                "structure_equivalence_fixed_rule": str(
                    equivalence_row["fixed_rule"]
                ),
                "structure_equivalence_hypothesis_type": str(
                    equivalence_row["hypothesis_type"]
                ),
                "structure_equivalence_hypothesis_status": str(
                    equivalence_row["hypothesis_status"]
                ),
                "nominal_members": ";".join(nominal_members),
                "parent_equivalence_members": str(
                    equivalence_row["parent_equivalence_members"]
                ),
                "counterpart_equivalence_members": str(
                    equivalence_row["counterpart_equivalence_members"]
                ),
                "candidate_pairs_json": str(
                    equivalence_row["candidate_pairs_json"]
                ),
                "candidate_pair_count": int(
                    equivalence_row["candidate_pair_count"]
                ),
                "exact_loading_group_selected": exact_loading_group_selected,
                "structure_equivalence_selected": structure_equivalence_selected,
                "exact_pair_identified": bool(
                    equivalence_row["exact_pair_identified"]
                ),
                "shared_state_hypothesis_supported": bool(
                    equivalence_row["shared_state_hypothesis_supported"]
                ),
                "candidate_pairs_may_be_expanded_as_edges": bool(
                    equivalence_row[
                        "candidate_pairs_may_be_expanded_as_edges"
                    ]
                ),
                "loading_group_guard_route": guard_route,
                "loading_group_guard_status": guard_status,
                "loading_group_v429_guard_route": str(
                    guard_row["v429_guard_route"]
                ),
                "loading_group_v429_guard_status": str(
                    guard_row["v429_guard_status"]
                ),
                "loading_group_v429_retained": bool(
                    guard_row["v429_retained"]
                ),
                "loading_group_v430_guard_route": str(
                    guard_row["v430_guard_route"]
                ),
                "loading_group_v430_guard_status": str(
                    guard_row["v430_guard_status"]
                ),
                "loading_group_v430_retained": bool(
                    guard_row["v430_retained"]
                ),
                "minimum_member_centroid_cosine": float(
                    guard_row["minimum_member_centroid_cosine"]
                ),
                "minimum_state_route_member_centroid_cosine": float(
                    guard_row[
                        "minimum_state_route_member_centroid_cosine"
                    ]
                ),
                "passes_loading_uncertainty_floor": bool(
                    guard_row["passes_loading_uncertainty_floor"]
                ),
                "complete_direct_collider_overlap": bool(
                    guard_row["complete_direct_collider_overlap"]
                ),
                "direct_collider_child_overlap": str(
                    guard_row["direct_collider_child_overlap"]
                ),
                "shared_direct_parent_children": str(
                    guard_row["shared_direct_parent_children"]
                ),
                "cross_assay_axis_id": str(
                    guard_row["cross_assay_axis_id"]
                ),
                "cross_assay_axis_modalities": str(
                    guard_row["cross_assay_axis_modalities"]
                ),
                "cross_assay_axis_n_groups": int(
                    guard_row["cross_assay_axis_n_groups"]
                ),
                "cross_assay_axis_n_modalities": int(
                    guard_row["cross_assay_axis_n_modalities"]
                ),
                "cross_assay_axis_confidence": float(
                    guard_row["cross_assay_axis_confidence"]
                ),
                "group_axis_lower_cosine": float(
                    guard_row["group_axis_lower_cosine"]
                ),
                "group_axis_view_support": float(
                    guard_row["group_axis_view_support"]
                ),
                "group_axis_relative_margin": float(
                    guard_row["group_axis_relative_margin"]
                ),
                "competing_parent_observability_required": bool(
                    guard_row["competing_parent_observability_required"]
                ),
                "complete_two_parent_comparison_available": bool(
                    guard_row["complete_two_parent_comparison_available"]
                ),
                "selected_parent_candidate": str(
                    guard_row["selected_parent_candidate"]
                ),
                "competing_parent_candidate": str(
                    guard_row["competing_parent_candidate"]
                ),
                "noncollider_sibling_candidate": str(
                    guard_row["noncollider_sibling_candidate"]
                ),
                "collider_child_candidate": str(
                    guard_row["collider_child_candidate"]
                ),
                "competition_selection_stage": str(
                    guard_row["competition_selection_stage"]
                ),
                "competing_parent_formal_test_eligible": bool(
                    guard_row["competing_parent_formal_test_eligible"]
                ),
                "competing_parent_oof_marginal_state_fit": float(
                    guard_row["competing_parent_oof_marginal_state_fit"]
                ),
                "competing_parent_oof_partial_state_fit": float(
                    guard_row["competing_parent_oof_partial_state_fit"]
                ),
                "competing_parent_direct_candidate_degree": int(
                    guard_row["competing_parent_direct_candidate_degree"]
                ),
                "competing_parent_state_specific_gate": bool(
                    guard_row["competing_parent_state_specific_gate"]
                ),
                "competing_parent_residual_structure_support": bool(
                    guard_row["competing_parent_residual_structure_support"]
                ),
                "competing_parent_fixed_rule_eligible": bool(
                    guard_row["competing_parent_fixed_rule_eligible"]
                ),
                "passes_competing_parent_observability_guard": bool(
                    guard_row["passes_competing_parent_observability_guard"]
                ),
                "interpretation": (
                    "four candidate pairs form one assay-local shared-state/"
                    "collider-role equivalence class; alternatives must not be "
                    "expanded as edges and no exact pair or direction is identified"
                    if structure_equivalence_selected
                    else (
                        "features form an assay-local loading group that aligns "
                        "to a selected cross-assay axis estimated without this "
                        "group; no exact cross-assay feature pairing, state-"
                        "membership promotion, or direct causation"
                        if axis_retained
                        else "structure-proposed loading group is unresolved "
                        f"under the v4.31 guard: {guard_status}"
                    )
                ),
            }
        )

    selected_equivalence_groups = state_equivalence_groups.loc[
        state_equivalence_groups["selection_status"].astype(str).eq("selected")
    ].sort_values(["modality", "state_equivalence_group_id"])
    feature_max_t = features.set_index("feature")["feature_global_maxT_p"]
    for row in selected_equivalence_groups.itertuples(index=False):
        members = sorted(
            member.strip()
            for member in str(row.supported_members).split(";")
            if member.strip()
        )
        if not members:
            raise ValueError(
                f"Selected state-equivalence group {row.state_equivalence_group_id} "
                "has no members"
            )
        group_p = float(feature_max_t.loc[members].max())
        hyperedge_rows.append(
            {
                "relation_id": str(row.state_equivalence_group_id),
                "edge_type": state_equivalence.GROUP_EDGE_TYPE,
                "layer": "within_assay_state_equivalence",
                "modalities": str(row.modality),
                "members": ";".join(members),
                "n_modalities": 1,
                "n_members": len(members),
                "selection_status": "selected",
                "evidence_basis": (
                    "strict_members_plus_pre_redundancy_selected_"
                    "redundancy_demoted_world_maxT_strong_core"
                ),
                "confidence": 1.0 - group_p,
                "empirical_p": group_p,
                "conditional_on_common_state_model": True,
                "hypothesis_only": True,
                "direct_causal_claimed": False,
                "direction_claimed": False,
                "exact_axis_pairing_claimed": False,
                "strict_members": str(row.strict_members),
                "equivalence_only_members": str(
                    row.equivalence_only_members
                ),
                "pairwise_expansion_permitted": False,
                "exact_cross_assay_pair_claimed": False,
                "interpretation": (
                    "within-assay strict and strong redundancy-demoted "
                    "shared-state reporters represented as one set; no "
                    "pairwise, cross-assay, directional, or direct claim"
                ),
            }
        )

    hyperedge_columns = [
        "relation_id",
        "edge_type",
        "layer",
        "modalities",
        "members",
        "n_modalities",
        "n_members",
        "selection_status",
        "evidence_basis",
        "confidence",
        "empirical_p",
        "conditional_on_common_state_model",
        "hypothesis_only",
        "direct_causal_claimed",
        "direction_claimed",
        "exact_axis_pairing_claimed",
        "strict_members",
        "equivalence_only_members",
        "pairwise_expansion_permitted",
        "exact_cross_assay_pair_claimed",
        "loading_group_v431_guard_route",
        "loading_group_v431_guard_status",
        "loading_group_v431_retained",
        "structure_equivalence_schema",
        "structure_equivalence_fixed_rule",
        "structure_equivalence_hypothesis_type",
        "structure_equivalence_hypothesis_status",
        "nominal_members",
        "parent_equivalence_members",
        "counterpart_equivalence_members",
        "candidate_pairs_json",
        "candidate_pair_count",
        "exact_loading_group_selected",
        "structure_equivalence_selected",
        "exact_pair_identified",
        "shared_state_hypothesis_supported",
        "candidate_pairs_may_be_expanded_as_edges",
        "loading_group_guard_route",
        "loading_group_guard_status",
        "loading_group_v429_guard_route",
        "loading_group_v429_guard_status",
        "loading_group_v429_retained",
        "loading_group_v430_guard_route",
        "loading_group_v430_guard_status",
        "loading_group_v430_retained",
        "minimum_member_centroid_cosine",
        "minimum_state_route_member_centroid_cosine",
        "passes_loading_uncertainty_floor",
        "complete_direct_collider_overlap",
        "direct_collider_child_overlap",
        "shared_direct_parent_children",
        "cross_assay_axis_id",
        "cross_assay_axis_modalities",
        "cross_assay_axis_n_groups",
        "cross_assay_axis_n_modalities",
        "cross_assay_axis_confidence",
        "group_axis_lower_cosine",
        "group_axis_view_support",
        "group_axis_relative_margin",
        "competing_parent_observability_required",
        "complete_two_parent_comparison_available",
        "selected_parent_candidate",
        "competing_parent_candidate",
        "noncollider_sibling_candidate",
        "collider_child_candidate",
        "competition_selection_stage",
        "competing_parent_formal_test_eligible",
        "competing_parent_oof_marginal_state_fit",
        "competing_parent_oof_partial_state_fit",
        "competing_parent_direct_candidate_degree",
        "competing_parent_state_specific_gate",
        "competing_parent_residual_structure_support",
        "competing_parent_fixed_rule_eligible",
        "passes_competing_parent_observability_guard",
        "interpretation",
    ]
    typed_hyperedges = pd.DataFrame(hyperedge_rows, columns=hyperedge_columns)
    if not typed_hyperedges.empty:
        typed_hyperedges = typed_hyperedges.sort_values("relation_id").reset_index(
            drop=True
        )

    calibrator_payload = (
        empirical_null_calibrator.to_payload()
        if empirical_null_calibrator is not None
        else None
    )
    multievidence_calibrator_payload = (
        multievidence_calibrator.to_payload()
        if multievidence_calibrator is not None
        else None
    )
    topology_consensus_calibrator_payload = (
        topology_consensus_calibrator.to_payload()
        if topology_consensus_calibrator is not None
        else None
    )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "method": (
            "replication_gated_assay_omnibus_with_conservative_feature_"
            "maxT_confirmation_redundancy_aware_state_equivalence_and_"
            "frozen_empirical_null_state_calibration_familywise_multievidence_"
            "state_calibration_sentinel_certified_state_topology_consensus_"
            "state_or_leave_one_group_out_cross_assay_axis_certified_loading_"
            "group_guard_structure_uncertainty_and_fair_competing_parent_"
            "observability_downgrades_with_unexpanded_state_structure_"
            "equivalence_and_separate_local_layers"
        ),
        "config": asdict(config),
        "empirical_null_calibrator": {
            "active": empirical_null_calibrator is not None,
            "schema_version": (
                empirical_null_calibrator.schema_version
                if empirical_null_calibrator is not None
                else ""
            ),
            "numeric_payload_sha256": (
                _canonical_sha256(calibrator_payload)
                if calibrator_payload is not None
                else ""
            ),
            "selection_threshold": (
                float(empirical_null_calibrator.selection_threshold)
                if empirical_null_calibrator is not None
                else None
            ),
            "application_requires_truth": False,
            "creates_pairwise_or_direct_relations": False,
        },
        "multievidence_calibrator": {
            "active": multievidence_calibrator is not None,
            "schema_version": (
                multievidence_calibrator.schema_version
                if multievidence_calibrator is not None
                else ""
            ),
            "numeric_payload_sha256": (
                _canonical_sha256(multievidence_calibrator_payload)
                if multievidence_calibrator_payload is not None
                else ""
            ),
            "selection_threshold": (
                float(multievidence_calibrator.selection_threshold)
                if multievidence_calibrator is not None
                else None
            ),
            "fixed_route_family": list(multievidence_state.ROUTE_NAMES),
            "state_specific_gate": "positive_marginal_and_partial_state_fit",
            "application_requires_truth": False,
            "creates_pairwise_or_direct_relations": False,
        },
        "topology_consensus_calibrator": {
            "active": bool(
                topology_consensus_calibrator is not None
                and topology_consensus_calibrator.sentinel_certified
            ),
            "schema_version": (
                topology_consensus_calibrator.schema_version
                if topology_consensus_calibrator is not None
                else ""
            ),
            "numeric_payload_sha256": (
                _canonical_sha256(topology_consensus_calibrator_payload)
                if topology_consensus_calibrator_payload is not None
                else ""
            ),
            "sentinel_certified": bool(
                topology_consensus_calibrator is not None
                and topology_consensus_calibrator.sentinel_certified
            ),
            "minimum_direct_candidate_degree": (
                topology_consensus_calibrator.minimum_direct_candidate_degree
                if topology_consensus_calibrator is not None
                else None
            ),
            "state_specific_gate": topology_consensus_state.STATE_SPECIFIC_GATE,
            "direct_structure_alone_is_insufficient": True,
            "application_requires_truth": False,
            "creates_pairwise_or_direct_relations": False,
        },
        "loading_group_state_guard": {
            "active": True,
            "schema_version": loading_group_state_guard.SCHEMA,
            "fixed_rule": loading_group_state_guard.FIXED_RULE,
            "state_route": loading_group_state_guard.STATE_ROUTE,
            "cross_assay_axis_route": loading_group_state_guard.AXIS_ROUTE,
            "cross_assay_axis_evidence_active": bool(
                group_axis_evidence is not None and axis_candidates is not None
            ),
            "leave_one_group_out_axis_evidence_required_for_axis_route": True,
            "upstream_state_membership_schema": (
                topology_consensus_state.SCHEMA
            ),
            "direct_structure_alone_is_insufficient": True,
            "application_requires_truth": False,
            "creates_cross_assay_or_direct_relations": False,
        },
        "loading_group_structure_uncertainty_guard": {
            "active": True,
            "schema_version": loading_group_structure_guard.SCHEMA,
            "fixed_rule": loading_group_structure_guard.FIXED_RULE,
            "minimum_state_route_member_centroid_cosine": (
                loading_group_structure_guard
                .MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
            ),
            "upstream_loading_group_guard_schema": (
                loading_group_state_guard.SCHEMA
            ),
            "axis_route_bypasses_uncertainty_floor": True,
            "both_routes_subject_to_direct_overlap_downgrade": True,
            "direct_overlap_downgrade_means_absence": False,
            "changes_state_v427_membership": False,
            "changes_direct_candidates": False,
            "application_requires_truth": False,
            "creates_cross_assay_or_direct_relations": False,
        },
        "loading_group_competing_parent_observability_guard": {
            "active": True,
            "schema_version": loading_group_competitor_guard.SCHEMA,
            "fixed_rule": loading_group_competitor_guard.FIXED_RULE,
            "upstream_loading_group_guard_schema": (
                loading_group_structure_guard.SCHEMA
            ),
            "state_route_requires_complete_two_parent_comparison": True,
            "competing_parent_observability_definition": (
                "frozen_v427_topology_consensus_fixed_rule_eligible"
            ),
            "axis_route_bypasses_competitor_gate": True,
            "downgrade_means_absence": False,
            "changes_state_v427_membership": False,
            "changes_direct_candidates": False,
            "application_requires_truth": False,
            "creates_cross_assay_or_direct_relations": False,
        },
        "loading_group_structure_equivalence": {
            "active": True,
            "schema_version": loading_group_structure_equivalence.SCHEMA,
            "fixed_rule": loading_group_structure_equivalence.FIXED_RULE,
            "upstream_loading_group_guard_schema": (
                loading_group_competitor_guard.SCHEMA
            ),
            "state_route_hypothesis_type": (
                loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE
            ),
            "state_route_candidate_pair_count": 4,
            "axis_route_hypothesis_type": (
                loading_group_structure_equivalence.AXIS_EXACT_TYPE
            ),
            "axis_route_candidate_pair_count": 1,
            "candidate_pairs_may_be_expanded_as_edges": False,
            "changes_v431_selection": False,
            "changes_state_v427_membership": False,
            "changes_direct_candidates": False,
            "application_requires_truth": False,
            "creates_cross_assay_or_direct_relations": False,
        },
        "estimands": {
            "confirmed_feature_layer": "depth_orthogonal_assay_local_state_response",
            "assay_participation_layer": (
                "replicated_depth_orthogonal_assay_local_state_response"
            ),
            "cross_assay_hyperedge": (
                "hypothesis_only_conditional_on_a_common_state_model"
            ),
            "loading_group_layer": (
                "state_route_two_by_two_assay_local_structure_equivalence_or_"
                "leave_one_group_out_cross_assay_axis_certified_exact_assay_"
                "local_loading_support"
            ),
            "direct_layer": "within_assay_direct_candidate",
            "state_equivalence_layer": (
                "within_assay_set_valued_shared_state_membership_uncertainty"
            ),
            "empirical_null_state_layer": (
                "within_assay_feature_participation_in_shared_latent_state"
            ),
            "multievidence_state_layer": (
                "familywise_multiroute_within_assay_participation_in_shared_"
                "latent_state"
            ),
            "topology_consensus_state_layer": (
                "weak_within_assay_state_participation_supported_by_separately_"
                "typed_residual_topology"
            ),
        },
        "semantics": {
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
        },
        "information_boundary": {
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
        },
        "counts": {
            "n_features": len(features),
            "n_upstream_conditional_candidates": int(upstream_selected.sum()),
            "n_confirmed_state_response_members": int(confirmed.sum()),
            "n_selected_assay_participation_calls": len(participating_modalities),
            "selected_assay_participation_modalities": participating_modalities,
            "strict_confirmed_feature_modalities": confirmed_modalities,
            "n_strict_confirmed_hyperedge_members": len(
                hyperedge_member_features
            ),
            "n_state_equivalence_only_members": int(
                features["state_equivalence_only_member"].sum()
            ),
            "n_state_supported_members": int(
                features["state_supported_member"].sum()
            ),
            "n_empirical_null_state_members": int(
                features["empirical_null_state_member"].sum()
            ),
            "n_state_calibrated_supported_members": int(
                features["state_calibrated_supported_member"].sum()
            ),
            "n_multievidence_state_members": int(
                features["multievidence_empirical_null_state_member"].sum()
            ),
            "n_state_v426_supported_members": int(
                features["state_v426_supported_member"].sum()
            ),
            "n_topology_consensus_state_members": int(
                features["topology_consensus_state_member"].sum()
            ),
            "n_state_v427_supported_members": int(
                features["state_v427_supported_member"].sum()
            ),
            "n_selected_state_equivalence_groups": len(
                selected_equivalence_groups
            ),
            "n_pre_v429_selected_assay_local_loading_groups": len(
                pre_v428_selected_groups
            ),
            "n_selected_assay_local_loading_groups": len(selected_groups),
            "n_v429_state_route_assay_local_loading_groups": len(
                retained_state_ids
            ),
            "n_v429_axis_route_assay_local_loading_groups": len(
                retained_axis_ids
            ),
            "n_v429_rejected_assay_local_loading_groups": len(
                rejected_group_ids
            ),
            "n_v430_retained_assay_local_loading_groups": int(
                loading_guard.decisions["v430_retained"].astype(bool).sum()
            ),
            "n_v431_competing_parent_observability_downgrades": int(
                (
                    loading_guard.decisions["v430_retained"].astype(bool)
                    & ~loading_guard.decisions["retained"].astype(bool)
                ).sum()
            ),
            "n_v432_state_structure_equivalence_hyperedges": int(
                loading_structure_equivalence.decisions[
                    "structure_equivalence_selected"
                ].astype(bool).sum()
            ),
            "n_v432_exact_axis_loading_groups": int(
                loading_structure_equivalence.decisions[
                    "exact_loading_group_selected"
                ].astype(bool).sum()
            ),
            "n_assay_local_loading_group_members": int(
                features["assay_local_loading_group_support"].sum()
            ),
            "n_assay_local_singleton_candidates": int(
                features["assay_local_singleton_candidate"].sum()
            ),
            "n_within_assay_direct_candidates": len(direct),
            "n_within_assay_direct_candidate_endpoints": int(
                features["within_assay_direct_candidate_overlap"].sum()
            ),
        },
    }
    return LayeredStateEvidenceResult(
        feature_evidence=features,
        typed_hyperedges=typed_hyperedges,
        within_assay_direct_candidates=direct,
        provenance=provenance,
    )


def run_layered_state_evidence(
    membership_path: Path,
    subspace_hyperedges_path: Path,
    consensus_edges_path: Path,
    loading_groups_path: Path,
    loading_group_membership_path: Path,
    structure_anchor_pair_evidence_path: Path,
    out_dir: Path,
    config: LayeredStateEvidenceConfig | None = None,
    empirical_null_calibrator_path: Path | None = None,
    multievidence_calibrator_path: Path | None = None,
    topology_consensus_calibrator_path: Path | None = None,
    group_axis_evidence_path: Path | None = None,
    axis_candidates_path: Path | None = None,
) -> dict[str, Any]:
    """Load safe observable artifacts, build layers, and content-address output."""

    config = config or LayeredStateEvidenceConfig()
    inputs = {
        "membership": _safe_observable_path(membership_path, "membership"),
        "subspace_hyperedges": _safe_observable_path(
            subspace_hyperedges_path, "subspace_hyperedges"
        ),
        "consensus_edges": _safe_observable_path(consensus_edges_path, "consensus_edges"),
        "loading_groups": _safe_observable_path(loading_groups_path, "loading_groups"),
        "loading_group_membership": _safe_observable_path(
            loading_group_membership_path, "loading_group_membership"
        ),
        "structure_anchor_pair_evidence": _safe_observable_path(
            structure_anchor_pair_evidence_path,
            "structure_anchor_pair_evidence",
        ),
    }
    if empirical_null_calibrator_path is not None:
        inputs["empirical_null_calibrator"] = _safe_observable_path(
            empirical_null_calibrator_path,
            "empirical_null_calibrator",
        )
    if multievidence_calibrator_path is not None:
        inputs["multievidence_calibrator"] = _safe_observable_path(
            multievidence_calibrator_path,
            "multievidence_calibrator",
        )
    if topology_consensus_calibrator_path is not None:
        inputs["topology_consensus_calibrator"] = _safe_observable_path(
            topology_consensus_calibrator_path,
            "topology_consensus_calibrator",
        )
    if (group_axis_evidence_path is None) != (axis_candidates_path is None):
        raise ValueError(
            "group-axis evidence and axis candidates must be supplied together"
        )
    if group_axis_evidence_path is not None:
        assert axis_candidates_path is not None
        inputs["group_axis_evidence"] = _safe_observable_path(
            group_axis_evidence_path,
            "group_axis_evidence",
        )
        inputs["axis_candidates"] = _safe_observable_path(
            axis_candidates_path,
            "axis_candidates",
        )
    input_bytes = {
        label: _read_regular_bytes_once(path, label)
        for label, path in inputs.items()
    }
    output = out_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tsv_options = {
        "sep": "\t",
        "keep_default_na": False,
        "float_precision": "round_trip",
    }
    calibrator = None
    calibrator_metadata: dict[str, Any] | None = None
    if "empirical_null_calibrator" in input_bytes:
        calibrator, calibrator_metadata = load_empirical_null_state_calibrator(
            input_bytes["empirical_null_calibrator"]
        )
    multievidence_calibrator = None
    multievidence_calibrator_metadata: dict[str, Any] | None = None
    if "multievidence_calibrator" in input_bytes:
        (
            multievidence_calibrator,
            multievidence_calibrator_metadata,
        ) = load_multievidence_state_calibrator(
            input_bytes["multievidence_calibrator"]
        )
    topology_consensus_calibrator = None
    topology_consensus_calibrator_metadata: dict[str, Any] | None = None
    if "topology_consensus_calibrator" in input_bytes:
        (
            topology_consensus_calibrator,
            topology_consensus_calibrator_metadata,
        ) = load_topology_consensus_state_calibrator(
            input_bytes["topology_consensus_calibrator"]
        )
    result = build_layered_state_evidence(
        pd.read_csv(
            io.BytesIO(input_bytes["membership"]),
            **tsv_options,
        ),
        pd.read_csv(
            io.BytesIO(input_bytes["subspace_hyperedges"]),
            **tsv_options,
        ),
        pd.read_csv(
            io.BytesIO(input_bytes["consensus_edges"]),
            **tsv_options,
        ),
        pd.read_csv(
            io.BytesIO(input_bytes["loading_groups"]),
            **tsv_options,
        ),
        pd.read_csv(
            io.BytesIO(input_bytes["loading_group_membership"]),
            **tsv_options,
        ),
        pd.read_csv(
            io.BytesIO(input_bytes["structure_anchor_pair_evidence"]),
            **tsv_options,
        ),
        config=config,
        empirical_null_calibrator=calibrator,
        multievidence_calibrator=multievidence_calibrator,
        topology_consensus_calibrator=topology_consensus_calibrator,
        group_axis_evidence=(
            pd.read_csv(
                io.BytesIO(input_bytes["group_axis_evidence"]),
                **tsv_options,
            )
            if "group_axis_evidence" in input_bytes
            else None
        ),
        axis_candidates=(
            pd.read_csv(
                io.BytesIO(input_bytes["axis_candidates"]),
                **tsv_options,
            )
            if "axis_candidates" in input_bytes
            else None
        ),
    )
    if calibrator_metadata is not None:
        result.provenance["empirical_null_calibrator"].update(
            calibrator_metadata
        )
        result.provenance["empirical_null_calibrator"][
            "artifact_file_sha256"
        ] = _bytes_sha256(input_bytes["empirical_null_calibrator"])
    if multievidence_calibrator_metadata is not None:
        result.provenance["multievidence_calibrator"].update(
            multievidence_calibrator_metadata
        )
        result.provenance["multievidence_calibrator"][
            "artifact_file_sha256"
        ] = _bytes_sha256(input_bytes["multievidence_calibrator"])
    if topology_consensus_calibrator_metadata is not None:
        result.provenance["topology_consensus_calibrator"].update(
            topology_consensus_calibrator_metadata
        )
        result.provenance["topology_consensus_calibrator"][
            "artifact_file_sha256"
        ] = _bytes_sha256(input_bytes["topology_consensus_calibrator"])
    paths = {
        "layered_feature_evidence.tsv": output / "layered_feature_evidence.tsv",
        "typed_layered_hyperedges.tsv": output / "typed_layered_hyperedges.tsv",
        "within_assay_direct_candidates.tsv": (
            output / "within_assay_direct_candidates.tsv"
        ),
        "layered_provenance.json": output / "layered_provenance.json",
    }
    result.feature_evidence.to_csv(paths["layered_feature_evidence.tsv"], sep="\t", index=False)
    result.typed_hyperedges.to_csv(paths["typed_layered_hyperedges.tsv"], sep="\t", index=False)
    result.within_assay_direct_candidates.to_csv(
        paths["within_assay_direct_candidates.tsv"], sep="\t", index=False
    )
    paths["layered_provenance.json"].write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )

    source = Path(__file__).resolve(strict=True)
    manifest = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_source_sha256": _sha256(source),
        "dependency_source_sha256": {
            "empirical_null_state_calibration_v4.py": _sha256(
                Path(empirical_null_state.__file__).resolve(strict=True)
            ),
            "loading_group_state_guard_v4.py": _sha256(
                Path(loading_group_state_guard.__file__).resolve(strict=True)
            ),
            "loading_group_structure_uncertainty_guard_v4.py": _sha256(
                Path(loading_group_structure_guard.__file__).resolve(strict=True)
            ),
            "loading_group_competing_parent_observability_guard_v4.py": _sha256(
                Path(loading_group_competitor_guard.__file__).resolve(strict=True)
            ),
            "loading_group_structure_equivalence_v4.py": _sha256(
                Path(loading_group_structure_equivalence.__file__).resolve(
                    strict=True
                )
            ),
            "multievidence_empirical_null_state_calibration_v4.py": _sha256(
                Path(multievidence_state.__file__).resolve(strict=True)
            ),
            "shared_state_subspace_v4.py": _sha256(
                Path(state_subspace.__file__).resolve(strict=True)
            ),
            "state_equivalence_membership_v4.py": _sha256(
                Path(state_equivalence.__file__).resolve(strict=True)
            ),
            "topology_consensus_state_calibration_v4.py": _sha256(
                Path(topology_consensus_state.__file__).resolve(strict=True)
            ),
        },
        "config": asdict(config),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_sha256": {
            label: _bytes_sha256(input_bytes[label]) for label in sorted(inputs)
        },
        "output_sha256": {
            name: _sha256(path) for name, path in sorted(paths.items())
        },
        "empirical_null_calibrator": (
            {
                **calibrator_metadata,
                "artifact_file_sha256": _bytes_sha256(
                    input_bytes["empirical_null_calibrator"]
                ),
                "input_path": str(inputs["empirical_null_calibrator"]),
            }
            if calibrator_metadata is not None
            else {
                "active": False,
                "application_requires_truth": False,
            }
        ),
        "multievidence_calibrator": (
            {
                **multievidence_calibrator_metadata,
                "artifact_file_sha256": _bytes_sha256(
                    input_bytes["multievidence_calibrator"]
                ),
                "input_path": str(inputs["multievidence_calibrator"]),
            }
            if multievidence_calibrator_metadata is not None
            else {
                "active": False,
                "application_requires_truth": False,
            }
        ),
        "topology_consensus_calibrator": (
            {
                **topology_consensus_calibrator_metadata,
                "artifact_file_sha256": _bytes_sha256(
                    input_bytes["topology_consensus_calibrator"]
                ),
                "input_path": str(inputs["topology_consensus_calibrator"]),
            }
            if topology_consensus_calibrator_metadata is not None
            else {
                "active": False,
                "application_requires_truth": False,
            }
        ),
        "candidate_frozen": True,
        "scorer_access_chronology": "not_asserted_by_this_standalone_module",
        "truth_or_scorer_consumed_by_algorithm": False,
        "calibrator_training_used_simulation_null_labels": bool(
            calibrator_metadata is not None
        ),
        "calibrator_application_requires_truth": False,
        "multievidence_calibrator_application_requires_truth": False,
        "topology_consensus_calibrator_application_requires_truth": False,
        "loading_group_axis_consensus_active": bool(
            "group_axis_evidence" in input_bytes
        ),
        "loading_group_axis_consensus_application_requires_truth": False,
        "loading_group_competing_parent_observability_guard_active": True,
        "loading_group_competing_parent_observability_application_requires_truth": False,
        "loading_group_structure_equivalence_active": True,
        "loading_group_structure_equivalence_schema_version": (
            loading_group_structure_equivalence.SCHEMA
        ),
        "loading_group_structure_equivalence_application_requires_truth": False,
        "loading_group_structure_equivalence_pairwise_expansion_permitted": False,
    }
    manifest_path = output / "frozen_layered_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--subspace-hyperedges", type=Path, required=True)
    parser.add_argument("--consensus-edges", type=Path, required=True)
    parser.add_argument("--loading-groups", type=Path, required=True)
    parser.add_argument("--loading-group-membership", type=Path, required=True)
    parser.add_argument(
        "--structure-anchor-pair-evidence", type=Path, required=True
    )
    parser.add_argument("--empirical-null-calibrator", type=Path)
    parser.add_argument("--multievidence-calibrator", type=Path)
    parser.add_argument("--topology-consensus-calibrator", type=Path)
    parser.add_argument("--group-axis-evidence", type=Path)
    parser.add_argument("--axis-candidates", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-maxT-alpha", type=float, default=0.10)
    parser.add_argument(
        "--state-equivalence-maxT-alpha", type=float, default=0.01
    )
    parser.add_argument("--minimum-confirmed-modalities", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = LayeredStateEvidenceConfig(
        maximum_confirmatory_feature_maxT_p=args.feature_maxT_alpha,
        maximum_state_equivalence_maxT_p=args.state_equivalence_maxT_alpha,
        minimum_confirmed_modalities=args.minimum_confirmed_modalities,
    )
    summary = run_layered_state_evidence(
        membership_path=args.membership,
        subspace_hyperedges_path=args.subspace_hyperedges,
        consensus_edges_path=args.consensus_edges,
        loading_groups_path=args.loading_groups,
        loading_group_membership_path=args.loading_group_membership,
        structure_anchor_pair_evidence_path=(
            args.structure_anchor_pair_evidence
        ),
        out_dir=args.out_dir,
        config=config,
        empirical_null_calibrator_path=args.empirical_null_calibrator,
        multievidence_calibrator_path=args.multievidence_calibrator,
        topology_consensus_calibrator_path=args.topology_consensus_calibrator,
        group_axis_evidence_path=args.group_axis_evidence,
        axis_candidates_path=args.axis_candidates,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "LayeredStateEvidenceConfig",
    "LayeredStateEvidenceResult",
    "build_layered_state_evidence",
    "load_empirical_null_state_calibrator",
    "load_multievidence_state_calibrator",
    "load_topology_consensus_state_calibrator",
    "run_layered_state_evidence",
]
