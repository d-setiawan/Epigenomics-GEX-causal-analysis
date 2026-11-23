#!/usr/bin/env python3
"""Sentinel-certified state/topology consensus for CA-BOSS v4.27.

This module promotes a feature only when two separately typed observable
signals agree:

* marginal and depth-adjusted state fits are both positive; and
* the feature is an endpoint of a frozen within-assay direct candidate.

The direct relation is not itself interpreted as state membership.  It is a
residual-structure support channel for otherwise unresolved weak state
evidence.  Fitting only certifies that the fixed conjunction has zero
eligibility among training sentinel nulls in every available source block.
Application consumes no labels and cannot create pairs, directions, or edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "topology_consensus_state_calibration.v4.27"
TIER = "within_assay_topology_consensus_state_member"
KEY_COLUMNS = ("block", "seed", "feature", "modality")
STATE_GATE_COLUMNS = (
    "oof_marginal_state_fit",
    "oof_partial_state_fit",
)
OBSERVABLE_COLUMNS = (
    *KEY_COLUMNS,
    "formal_test_eligible",
    *STATE_GATE_COLUMNS,
    "within_assay_direct_candidate_degree",
)
MINIMUM_DIRECT_CANDIDATE_DEGREE = 1
STATE_SPECIFIC_GATE = "positive_marginal_and_partial_state_fit"
NULL_CONTROL_METHOD = (
    "all_training_source_block_sentinels_must_have_zero_fixed_"
    "state_topology_conjunction_eligibility"
)
FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "outcome",
    "scorer",
    "ground_truth",
)


def _boolean(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        values = sorted(set(normalized.loc[invalid]))
        raise ValueError(f"{name} contains non-boolean values {values}")
    return normalized.isin({"true", "1"})


def observable_topology_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact truth-free evidence accepted by v4.27."""

    forbidden = sorted(
        column
        for column in frame.columns
        if any(token in str(column).casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Observable topology evidence contains forbidden columns {forbidden}")
    missing = sorted(set(OBSERVABLE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Observable topology evidence is missing columns {missing}")
    output = frame.loc[:, OBSERVABLE_COLUMNS].copy()
    if output.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Observable topology evidence contains duplicate feature keys")
    for column in KEY_COLUMNS:
        if (
            output[column].isna().any()
            or output[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Observable key column {column} contains empty values")
    output["seed"] = pd.to_numeric(output["seed"], errors="raise").astype(int)
    output["formal_test_eligible"] = _boolean(
        output["formal_test_eligible"], name="formal_test_eligible"
    )
    for column in STATE_GATE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="raise")
        if not np.isfinite(output[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Observable evidence column {column} must be finite")
    degree = pd.to_numeric(
        output["within_assay_direct_candidate_degree"], errors="raise"
    )
    if (
        not np.isfinite(degree.to_numpy(dtype=float)).all()
        or (degree < 0).any()
        or not np.equal(degree, np.floor(degree)).all()
    ):
        raise ValueError(
            "within_assay_direct_candidate_degree must contain nonnegative integers"
        )
    output["within_assay_direct_candidate_degree"] = degree.astype(int)
    output["topology_consensus_state_specific_gate"] = (
        output["oof_marginal_state_fit"].gt(0.0)
        & output["oof_partial_state_fit"].gt(0.0)
    )
    output["topology_consensus_residual_structure_support"] = output[
        "within_assay_direct_candidate_degree"
    ].ge(MINIMUM_DIRECT_CANDIDATE_DEGREE)
    output["topology_consensus_fixed_rule_eligible"] = (
        output["formal_test_eligible"]
        & output["topology_consensus_state_specific_gate"]
        & output["topology_consensus_residual_structure_support"]
    )
    return output


@dataclass(frozen=True)
class FrozenTopologyConsensusStateCalibrator:
    """Truth-free application artifact for the fixed v4.27 conjunction."""

    schema_version: str
    minimum_direct_candidate_degree: int
    state_specific_gate: str
    null_control_method: str
    training_source_blocks: tuple[str, ...]
    training_observable_rows: int
    training_null_rows: int
    training_rule_eligible_null_rows: int
    sentinel_certified: bool
    exact_cross_assay_pair_claimed: bool = False
    direct_causal_claimed: bool = False
    pairwise_expansion_permitted: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA:
            raise ValueError(f"Unsupported calibrator schema {self.schema_version}")
        if self.minimum_direct_candidate_degree != MINIMUM_DIRECT_CANDIDATE_DEGREE:
            raise ValueError("Calibrator direct-degree rule does not match v4.27")
        if self.state_specific_gate != STATE_SPECIFIC_GATE:
            raise ValueError("Unsupported state-specific gate")
        if self.null_control_method != NULL_CONTROL_METHOD:
            raise ValueError("Unsupported sentinel-null control method")
        if len(self.training_source_blocks) < 2:
            raise ValueError("At least two source blocks are required")
        if self.training_observable_rows <= 0 or self.training_null_rows <= 0:
            raise ValueError("Training observable and null counts must be positive")
        if self.training_rule_eligible_null_rows < 0:
            raise ValueError("Eligible-null count cannot be negative")
        if self.sentinel_certified != (
            self.training_rule_eligible_null_rows == 0
        ):
            raise ValueError("Sentinel certification/count are inconsistent")
        if self.exact_cross_assay_pair_claimed or self.direct_causal_claimed:
            raise ValueError("State calibrator cannot claim pairing or causation")
        if self.pairwise_expansion_permitted:
            raise ValueError("State calibrator cannot permit pairwise expansion")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_source_blocks"] = list(self.training_source_blocks)
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "FrozenTopologyConsensusStateCalibrator":
        values = dict(payload)
        values["training_source_blocks"] = tuple(values["training_source_blocks"])
        return cls(**values)


def fit_topology_consensus_state_calibrator(
    observable_evidence: pd.DataFrame,
    *,
    null_feature: pd.Series | Sequence[bool],
) -> FrozenTopologyConsensusStateCalibrator:
    """Certify the fixed conjunction on labeled training sentinel nulls."""

    evidence = observable_topology_evidence(observable_evidence)
    if isinstance(null_feature, pd.Series):
        if not null_feature.index.equals(evidence.index):
            raise ValueError("null_feature index must exactly match evidence")
        null_mask = _boolean(null_feature, name="null_feature")
    else:
        if len(null_feature) != len(evidence):
            raise ValueError("null_feature length must match evidence")
        null_mask = _boolean(
            pd.Series(null_feature, index=evidence.index), name="null_feature"
        )
    null_blocks = sorted(set(evidence.loc[null_mask, "block"].astype(str)))
    if len(null_blocks) < 2:
        raise ValueError("Calibration nulls must span at least two source blocks")
    block_counts = (
        evidence.loc[null_mask]
        .groupby("block", sort=True)
        .size()
        .reindex(null_blocks, fill_value=0)
    )
    if (block_counts <= 0).any():
        raise ValueError("Every calibration source block needs a null feature")
    eligible_nulls = (
        evidence["topology_consensus_fixed_rule_eligible"] & null_mask
    )
    return FrozenTopologyConsensusStateCalibrator(
        schema_version=SCHEMA,
        minimum_direct_candidate_degree=MINIMUM_DIRECT_CANDIDATE_DEGREE,
        state_specific_gate=STATE_SPECIFIC_GATE,
        null_control_method=NULL_CONTROL_METHOD,
        training_source_blocks=tuple(null_blocks),
        training_observable_rows=int(len(evidence)),
        training_null_rows=int(null_mask.sum()),
        training_rule_eligible_null_rows=int(eligible_nulls.sum()),
        sentinel_certified=not bool(eligible_nulls.any()),
    )


def annotate_topology_consensus_state_membership(
    observable_evidence: pd.DataFrame,
    *,
    upstream_supported: pd.Series | Sequence[bool],
    upstream_tier: pd.Series | Sequence[str],
    calibrator: FrozenTopologyConsensusStateCalibrator,
) -> pd.DataFrame:
    """Apply v4.27 without labels and preserve the complete v4.26 union."""

    evidence = observable_topology_evidence(observable_evidence)
    if isinstance(upstream_supported, pd.Series):
        if not upstream_supported.index.equals(evidence.index):
            raise ValueError("upstream_supported index must exactly match evidence")
        upstream = _boolean(upstream_supported, name="upstream_supported")
    else:
        if len(upstream_supported) != len(evidence):
            raise ValueError("upstream_supported length must match evidence")
        upstream = _boolean(
            pd.Series(upstream_supported, index=evidence.index),
            name="upstream_supported",
        )
    if isinstance(upstream_tier, pd.Series):
        if not upstream_tier.index.equals(evidence.index):
            raise ValueError("upstream_tier index must exactly match evidence")
        tiers = upstream_tier.fillna("").astype(str)
    else:
        if len(upstream_tier) != len(evidence):
            raise ValueError("upstream_tier length must match evidence")
        tiers = pd.Series(upstream_tier, index=evidence.index).fillna("").astype(str)

    candidate = evidence["formal_test_eligible"] & ~upstream
    addition = (
        bool(calibrator.sentinel_certified)
        & candidate
        & evidence["topology_consensus_state_specific_gate"]
        & evidence["topology_consensus_residual_structure_support"]
    )
    output = evidence.copy()
    output["topology_consensus_calibration_active"] = bool(
        calibrator.sentinel_certified
    )
    output["topology_consensus_calibration_candidate"] = candidate
    output["topology_consensus_empirical_null_certified"] = bool(
        calibrator.sentinel_certified
    )
    output["topology_consensus_state_member"] = addition
    output["state_v427_supported_member"] = upstream | addition
    output["state_v427_membership_tier"] = tiers
    output.loc[addition, "state_v427_membership_tier"] = TIER
    output["topology_consensus_calibration_schema"] = calibrator.schema_version
    output["topology_consensus_null_control_method"] = (
        calibrator.null_control_method
    )
    output["exact_cross_assay_pair_claimed"] = False
    output["direct_causal_claimed"] = False
    output["pairwise_expansion_permitted"] = False
    if (upstream & ~output["state_v427_supported_member"]).any():
        raise RuntimeError("v4.27 removed an upstream v4.26 state member")
    return output


__all__ = [
    "FrozenTopologyConsensusStateCalibrator",
    "KEY_COLUMNS",
    "MINIMUM_DIRECT_CANDIDATE_DEGREE",
    "NULL_CONTROL_METHOD",
    "OBSERVABLE_COLUMNS",
    "SCHEMA",
    "STATE_GATE_COLUMNS",
    "STATE_SPECIFIC_GATE",
    "TIER",
    "annotate_topology_consensus_state_membership",
    "fit_topology_consensus_state_calibrator",
    "observable_topology_evidence",
]
