#!/usr/bin/env python3
"""Familywise empirical-null state calibration for CA-BOSS v4.26.

This module promotes only *within-assay shared-state membership*.  It combines
three predeclared, observable evidence routes and calibrates their maximum on
simulation sentinel nulls from training source blocks.  The maximum-over-route
score makes route multiplicity part of the null distribution instead of
silently trying several thresholds.

The three routes deliberately separate complementary evidence:

* joint state-and-depth predictive fit plus conditional state evidence;
* joint state-and-depth predictive fit plus conditional Wald excess; and
* marginal state fit plus conditional state evidence.

Joint fit can improve sparse-feature reliability, but it is never sufficient:
application also requires positive marginal and depth-adjusted partial state
fit.  The output cannot create a feature pair, a direction, or a direct edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "multievidence_empirical_null_state_calibration.v4.26"
TIER = "within_assay_multievidence_empirical_null_state_member"
KEY_COLUMNS = ("block", "seed", "feature", "modality")
SCORE_COMPONENT_COLUMNS = (
    "oof_marginal_state_fit",
    "oof_joint_fit",
    "conditional_wald_above_null_95pct",
    "median_conditional_score_norm",
)
STATE_GATE_COLUMNS = (
    "oof_marginal_state_fit",
    "oof_partial_state_fit",
)
OBSERVABLE_COLUMNS = (
    *KEY_COLUMNS,
    "formal_test_eligible",
    "state_supported_member",
    "state_membership_tier",
    *STATE_GATE_COLUMNS,
    "oof_joint_fit",
    "conditional_wald_above_null_95pct",
    "median_conditional_score_norm",
)
ROUTE_NAMES = (
    "joint_conditional_magnitude",
    "joint_conditional_excess",
    "marginal_conditional_magnitude",
)
# Columns follow SCORE_COMPONENT_COLUMNS.  These weights are intentionally
# fixed; v4.26 does not select a route or weight after seeing an application
# block.
ROUTE_WEIGHTS = (
    (0.0, 1.0, 1.0, 1.0),
    (0.0, 1.0, 0.5, 0.0),
    (1.0, 0.0, 1.0, 1.5),
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


def observable_feature_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact truth-free feature evidence accepted by v4.26."""

    forbidden = sorted(
        column
        for column in frame.columns
        if any(token in str(column).casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Observable evidence contains forbidden columns {forbidden}")
    missing = sorted(set(OBSERVABLE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Observable feature evidence is missing columns {missing}")
    output = frame.loc[:, OBSERVABLE_COLUMNS].copy()
    if output.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Observable feature evidence contains duplicate feature keys")
    for column in KEY_COLUMNS:
        if output[column].isna().any() or output[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Observable key column {column} contains empty values")
    output["seed"] = pd.to_numeric(output["seed"], errors="raise").astype(int)
    output["formal_test_eligible"] = _boolean(
        output["formal_test_eligible"], name="formal_test_eligible"
    )
    output["state_supported_member"] = _boolean(
        output["state_supported_member"], name="state_supported_member"
    )
    for column in sorted(set((*SCORE_COMPONENT_COLUMNS, *STATE_GATE_COLUMNS))):
        output[column] = pd.to_numeric(output[column], errors="raise")
        if not np.isfinite(output[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Observable evidence column {column} must be finite")
    if (output["median_conditional_score_norm"] < 0).any():
        raise ValueError("median_conditional_score_norm cannot be negative")
    output["multievidence_calibration_candidate"] = (
        output["formal_test_eligible"] & ~output["state_supported_member"]
    )
    output["multievidence_state_specific_gate"] = (
        output["oof_marginal_state_fit"].gt(0.0)
        & output["oof_partial_state_fit"].gt(0.0)
    )
    return output


@dataclass(frozen=True)
class FrozenMultievidenceStateCalibrator:
    """Numeric, truth-free application artifact for the v4.26 state layer."""

    schema_version: str
    component_columns: tuple[str, ...]
    component_null_medians: tuple[float, ...]
    component_null_iqrs: tuple[float, ...]
    route_names: tuple[str, ...]
    route_weights: tuple[tuple[float, ...], ...]
    route_null_medians: tuple[float, ...]
    route_null_iqrs: tuple[float, ...]
    maximum_null_family_score: float
    second_largest_block_maximum_null_family_score: float
    block_maximum_extrapolation: float
    selection_threshold: float
    training_source_blocks: tuple[str, ...]
    training_candidate_rows: int
    training_null_rows: int
    threshold_method: str = (
        "max_over_fixed_routes_then_one_source_block_maximum_gap_beyond_null_maximum"
    )
    state_specific_gate: str = "positive_marginal_and_partial_state_fit"
    exact_cross_assay_pair_claimed: bool = False
    direct_causal_claimed: bool = False
    pairwise_expansion_permitted: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA:
            raise ValueError(f"Unsupported calibrator schema {self.schema_version}")
        if self.component_columns != SCORE_COMPONENT_COLUMNS:
            raise ValueError("Calibrator component columns do not match v4.26")
        if self.route_names != ROUTE_NAMES:
            raise ValueError("Calibrator route names do not match v4.26")
        if len(self.route_weights) != len(ROUTE_WEIGHTS) or not np.allclose(
            np.asarray(self.route_weights, dtype=float),
            np.asarray(ROUTE_WEIGHTS, dtype=float),
        ):
            raise ValueError("Calibrator route weights do not match v4.26")
        if len(self.component_null_medians) != len(SCORE_COMPONENT_COLUMNS):
            raise ValueError("Component null medians have the wrong length")
        if len(self.component_null_iqrs) != len(SCORE_COMPONENT_COLUMNS):
            raise ValueError("Component null IQRs have the wrong length")
        if len(self.route_null_medians) != len(ROUTE_NAMES):
            raise ValueError("Route null medians have the wrong length")
        if len(self.route_null_iqrs) != len(ROUTE_NAMES):
            raise ValueError("Route null IQRs have the wrong length")
        numeric = (
            *self.component_null_medians,
            *self.component_null_iqrs,
            *self.route_null_medians,
            *self.route_null_iqrs,
            self.maximum_null_family_score,
            self.second_largest_block_maximum_null_family_score,
            self.block_maximum_extrapolation,
            self.selection_threshold,
        )
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError("Calibrator numeric values must be finite")
        if any(value <= 0 for value in self.component_null_iqrs):
            raise ValueError("Component null IQRs must be positive")
        if any(value <= 0 for value in self.route_null_iqrs):
            raise ValueError("Route null IQRs must be positive")
        if len(self.training_source_blocks) < 2:
            raise ValueError("At least two source blocks are required")
        if self.training_candidate_rows <= 0 or self.training_null_rows <= 0:
            raise ValueError("Training candidate and null counts must be positive")
        if self.block_maximum_extrapolation < 0:
            raise ValueError("Block-maximum extrapolation cannot be negative")
        if self.selection_threshold <= self.maximum_null_family_score:
            raise ValueError("Selection threshold must strictly exceed every training null")
        if self.state_specific_gate != "positive_marginal_and_partial_state_fit":
            raise ValueError("Unsupported state-specific gate")
        if self.exact_cross_assay_pair_claimed or self.direct_causal_claimed:
            raise ValueError("State calibrator cannot claim pairing or direct causation")
        if self.pairwise_expansion_permitted:
            raise ValueError("State calibrator cannot permit pairwise expansion")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "component_columns",
            "component_null_medians",
            "component_null_iqrs",
            "route_names",
            "route_null_medians",
            "route_null_iqrs",
            "training_source_blocks",
        ):
            payload[key] = list(payload[key])
        payload["route_weights"] = [list(values) for values in self.route_weights]
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "FrozenMultievidenceStateCalibrator":
        values = dict(payload)
        for key in (
            "component_columns",
            "component_null_medians",
            "component_null_iqrs",
            "route_names",
            "route_null_medians",
            "route_null_iqrs",
            "training_source_blocks",
        ):
            values[key] = tuple(values[key])
        values["route_weights"] = tuple(
            tuple(float(value) for value in row) for row in values["route_weights"]
        )
        return cls(**values)


def _median_iqr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.quantile(values, (0.25, 0.5, 0.75), axis=0)
    medians = np.asarray(quantiles[1], dtype=float)
    iqrs = np.asarray(quantiles[2] - quantiles[0], dtype=float)
    if np.any(iqrs <= 0) or not np.isfinite(iqrs).all():
        raise ValueError("Calibration null evidence has a degenerate IQR")
    return medians, iqrs


def _score(
    evidence: pd.DataFrame,
    calibrator: FrozenMultievidenceStateCalibrator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    components = evidence.loc[:, SCORE_COMPONENT_COLUMNS].to_numpy(dtype=float)
    component_z = (
        components - np.asarray(calibrator.component_null_medians, dtype=float)
    ) / np.asarray(calibrator.component_null_iqrs, dtype=float)
    route_raw = component_z @ np.asarray(calibrator.route_weights, dtype=float).T
    route_z = (
        route_raw - np.asarray(calibrator.route_null_medians, dtype=float)
    ) / np.asarray(calibrator.route_null_iqrs, dtype=float)
    family_score = np.max(route_z, axis=1)
    winning_route = np.argmax(route_z, axis=1)
    return component_z, route_z, family_score, winning_route


def fit_multievidence_state_calibrator(
    observable_evidence: pd.DataFrame,
    *,
    null_feature: pd.Series | Sequence[bool],
) -> FrozenMultievidenceStateCalibrator:
    """Fit v4.26 on labeled simulation nulls from training source blocks."""

    evidence = observable_feature_evidence(observable_evidence)
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
    candidates = evidence["multievidence_calibration_candidate"]
    calibration_null = candidates & null_mask
    null_blocks = sorted(set(evidence.loc[calibration_null, "block"].astype(str)))
    if len(null_blocks) < 2:
        raise ValueError("Calibration nulls must span at least two source blocks")
    block_counts = (
        evidence.loc[calibration_null]
        .groupby("block", sort=True)
        .size()
        .reindex(null_blocks, fill_value=0)
    )
    if (block_counts <= 0).any():
        raise ValueError("Every calibration source block needs a null feature")

    null_components = evidence.loc[
        calibration_null, SCORE_COMPONENT_COLUMNS
    ].to_numpy(dtype=float)
    component_medians, component_iqrs = _median_iqr(null_components)
    all_components = evidence.loc[:, SCORE_COMPONENT_COLUMNS].to_numpy(dtype=float)
    component_z = (all_components - component_medians) / component_iqrs
    route_raw = component_z @ np.asarray(ROUTE_WEIGHTS, dtype=float).T
    route_medians, route_iqrs = _median_iqr(route_raw[calibration_null.to_numpy()])
    route_z = (route_raw - route_medians) / route_iqrs
    family_score = np.max(route_z, axis=1)
    block_maxima = (
        pd.DataFrame(
            {
                "block": evidence.loc[calibration_null, "block"].astype(str),
                "score": family_score[calibration_null.to_numpy()],
            }
        )
        .groupby("block", sort=True)["score"]
        .max()
        .sort_values()
    )
    maximum = float(block_maxima.iloc[-1])
    second = float(block_maxima.iloc[-2])
    extrapolation = float(maximum - second)
    threshold = float(np.nextafter(maximum + extrapolation, np.inf))
    if threshold <= maximum:
        threshold = float(np.nextafter(maximum, np.inf))
    return FrozenMultievidenceStateCalibrator(
        schema_version=SCHEMA,
        component_columns=SCORE_COMPONENT_COLUMNS,
        component_null_medians=tuple(float(value) for value in component_medians),
        component_null_iqrs=tuple(float(value) for value in component_iqrs),
        route_names=ROUTE_NAMES,
        route_weights=ROUTE_WEIGHTS,
        route_null_medians=tuple(float(value) for value in route_medians),
        route_null_iqrs=tuple(float(value) for value in route_iqrs),
        maximum_null_family_score=maximum,
        second_largest_block_maximum_null_family_score=second,
        block_maximum_extrapolation=extrapolation,
        selection_threshold=threshold,
        training_source_blocks=tuple(null_blocks),
        training_candidate_rows=int(candidates.sum()),
        training_null_rows=int(calibration_null.sum()),
    )


def annotate_multievidence_state_membership(
    observable_evidence: pd.DataFrame,
    *,
    calibrator: FrozenMultievidenceStateCalibrator,
) -> pd.DataFrame:
    """Apply a frozen v4.26 calibrator without labels or scorer inputs."""

    evidence = observable_feature_evidence(observable_evidence)
    component_z, route_z, family_score, winning_route = _score(evidence, calibrator)
    candidate = evidence["multievidence_calibration_candidate"].to_numpy(dtype=bool)
    state_gate = evidence["multievidence_state_specific_gate"].to_numpy(dtype=bool)
    addition = candidate & state_gate & (family_score >= calibrator.selection_threshold)
    output = evidence.copy()
    for index, column in enumerate(SCORE_COMPONENT_COLUMNS):
        output[f"multievidence_component_z__{column}"] = component_z[:, index]
    for index, route in enumerate(ROUTE_NAMES):
        output[f"multievidence_route_z__{route}"] = route_z[:, index]
    output["multievidence_winning_route"] = [
        ROUTE_NAMES[int(index)] for index in winning_route
    ]
    output["multievidence_family_score"] = family_score
    output["multievidence_selection_threshold"] = calibrator.selection_threshold
    output["multievidence_empirical_null_state_member"] = addition
    output["state_multievidence_supported_member"] = (
        output["state_supported_member"] | addition
    )
    output["state_multievidence_membership_tier"] = output[
        "state_membership_tier"
    ].astype(str)
    output.loc[addition, "state_multievidence_membership_tier"] = TIER
    output["multievidence_calibration_schema"] = SCHEMA
    output["multievidence_threshold_method"] = calibrator.threshold_method
    output["exact_cross_assay_pair_claimed"] = False
    output["direct_causal_claimed"] = False
    output["pairwise_expansion_permitted"] = False
    if (
        output["state_multievidence_supported_member"]
        & ~output["state_supported_member"]
        & ~output["multievidence_empirical_null_state_member"]
    ).any():
        raise RuntimeError("Multievidence membership contains an unsupported promotion")
    if (
        output["multievidence_empirical_null_state_member"]
        & ~output["multievidence_state_specific_gate"]
    ).any():
        raise RuntimeError("Multievidence membership bypassed the state-specific gate")
    if (
        output["state_supported_member"]
        & ~output["state_multievidence_supported_member"]
    ).any():
        raise RuntimeError("Multievidence tier changed existing membership")
    return output
