#!/usr/bin/env python3
"""Simulation-calibrated empirical-null state membership for CA-BOSS v4.

The production application path consumes only observable feature evidence and
a frozen numeric calibrator.  Simulator truth is used only by a separate
training/audit path to identify null rows while fitting that calibrator.

The score deliberately combines two distinct pieces of state evidence:

* cross-fitted marginal background-state fit; and
* conditional Wald excess above the feature's permutation-null 95th percentile.

Both are robustly standardized on training null features.  The selection
threshold is extrapolated one source-block-maximum spacing beyond the largest
training-null score.  This makes a feature clear not merely the pooled null
maximum, but also the largest observed between-environment extreme shift.

This module annotates shared-state membership only.  It never creates a direct
edge, a direction, an exact cross-assay feature match, or a pairwise expansion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "empirical_null_state_calibration.v4.25"
TIER = "within_assay_empirical_null_state_member"
DEFAULT_CONDITIONAL_WEIGHT = 1.5
DEFAULT_CONDITIONAL_WEIGHT_GRID = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
KEY_COLUMNS = ("block", "seed", "feature", "modality")
OBSERVABLE_COLUMNS = (
    *KEY_COLUMNS,
    "formal_test_eligible",
    "state_supported_member",
    "state_membership_tier",
    "oof_marginal_state_fit",
    "conditional_wald_above_null_95pct",
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
    """Return the exact truth-free feature evidence accepted by the selector."""

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
    for column in (
        "oof_marginal_state_fit",
        "conditional_wald_above_null_95pct",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise")
        if not np.isfinite(output[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Observable evidence column {column} must be finite")
    output["empirical_null_calibration_candidate"] = (
        output["formal_test_eligible"] & ~output["state_supported_member"]
    )
    return output


@dataclass(frozen=True)
class FrozenEmpiricalNullStateCalibrator:
    """Numeric artifact needed for truth-free state-membership application."""

    schema_version: str
    conditional_weight: float
    marginal_null_median: float
    marginal_null_iqr: float
    conditional_excess_null_median: float
    conditional_excess_null_iqr: float
    maximum_null_score: float
    second_largest_block_maximum_null_score: float
    block_maximum_extrapolation: float
    selection_threshold: float
    training_source_blocks: tuple[str, ...]
    training_candidate_rows: int
    training_null_rows: int
    threshold_method: str = "one_source_block_maximum_gap_beyond_null_maximum"
    exact_cross_assay_pair_claimed: bool = False
    direct_causal_claimed: bool = False
    pairwise_expansion_permitted: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA:
            raise ValueError(f"Unsupported calibrator schema {self.schema_version}")
        if not np.isfinite(self.conditional_weight) or self.conditional_weight <= 0:
            raise ValueError("conditional_weight must be positive and finite")
        if self.marginal_null_iqr <= 0 or self.conditional_excess_null_iqr <= 0:
            raise ValueError("Null IQR values must be positive")
        if len(self.training_source_blocks) < 2:
            raise ValueError("At least two source blocks are required")
        if self.training_candidate_rows <= 0 or self.training_null_rows <= 0:
            raise ValueError("Training candidate and null counts must be positive")
        if self.block_maximum_extrapolation < 0:
            raise ValueError("Block-maximum extrapolation cannot be negative")
        if self.selection_threshold <= self.maximum_null_score:
            raise ValueError("Selection threshold must strictly exceed every training null")
        if self.exact_cross_assay_pair_claimed or self.direct_causal_claimed:
            raise ValueError("State calibrator cannot claim pairing or direct causation")
        if self.pairwise_expansion_permitted:
            raise ValueError("State calibrator cannot permit pairwise expansion")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_source_blocks"] = list(self.training_source_blocks)
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "FrozenEmpiricalNullStateCalibrator":
        values = dict(payload)
        values["training_source_blocks"] = tuple(
            str(value) for value in values["training_source_blocks"]
        )
        return cls(**values)


def _score(
    evidence: pd.DataFrame,
    calibrator: FrozenEmpiricalNullStateCalibrator,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    marginal_z = (
        evidence["oof_marginal_state_fit"] - calibrator.marginal_null_median
    ) / calibrator.marginal_null_iqr
    conditional_z = (
        evidence["conditional_wald_above_null_95pct"]
        - calibrator.conditional_excess_null_median
    ) / calibrator.conditional_excess_null_iqr
    score = marginal_z + calibrator.conditional_weight * conditional_z
    return marginal_z, conditional_z, score


def fit_empirical_null_state_calibrator(
    observable_evidence: pd.DataFrame,
    *,
    null_feature: pd.Series | Sequence[bool],
    conditional_weight: float = DEFAULT_CONDITIONAL_WEIGHT,
) -> FrozenEmpiricalNullStateCalibrator:
    """Fit a calibrator using labeled simulation nulls from training blocks.

    ``observable_evidence`` must already be stripped of truth columns.  The
    separate ``null_feature`` mask is consumed only here and is not serialized
    row by row.  Application via :func:`annotate_empirical_null_state_membership`
    needs no labels.
    """

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
    if not np.isfinite(float(conditional_weight)) or conditional_weight <= 0:
        raise ValueError("conditional_weight must be positive and finite")
    candidates = evidence["empirical_null_calibration_candidate"]
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
    nulls = evidence.loc[calibration_null]
    marginal_median = float(nulls["oof_marginal_state_fit"].median())
    marginal_iqr = float(
        nulls["oof_marginal_state_fit"].quantile(0.75)
        - nulls["oof_marginal_state_fit"].quantile(0.25)
    )
    conditional_median = float(
        nulls["conditional_wald_above_null_95pct"].median()
    )
    conditional_iqr = float(
        nulls["conditional_wald_above_null_95pct"].quantile(0.75)
        - nulls["conditional_wald_above_null_95pct"].quantile(0.25)
    )
    if marginal_iqr <= 0 or conditional_iqr <= 0:
        raise ValueError("Calibration null evidence has a degenerate IQR")
    provisional = FrozenEmpiricalNullStateCalibrator(
        schema_version=SCHEMA,
        conditional_weight=float(conditional_weight),
        marginal_null_median=marginal_median,
        marginal_null_iqr=marginal_iqr,
        conditional_excess_null_median=conditional_median,
        conditional_excess_null_iqr=conditional_iqr,
        maximum_null_score=0.0,
        second_largest_block_maximum_null_score=-1.0,
        block_maximum_extrapolation=1.0,
        selection_threshold=1.0,
        training_source_blocks=tuple(null_blocks),
        training_candidate_rows=int(candidates.sum()),
        training_null_rows=int(calibration_null.sum()),
    )
    _, _, scores = _score(evidence, provisional)
    block_maxima = (
        pd.DataFrame(
            {
                "block": evidence.loc[calibration_null, "block"].astype(str),
                "score": scores.loc[calibration_null].astype(float),
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
    return FrozenEmpiricalNullStateCalibrator(
        schema_version=SCHEMA,
        conditional_weight=float(conditional_weight),
        marginal_null_median=marginal_median,
        marginal_null_iqr=marginal_iqr,
        conditional_excess_null_median=conditional_median,
        conditional_excess_null_iqr=conditional_iqr,
        maximum_null_score=maximum,
        second_largest_block_maximum_null_score=second,
        block_maximum_extrapolation=extrapolation,
        selection_threshold=threshold,
        training_source_blocks=tuple(null_blocks),
        training_candidate_rows=int(candidates.sum()),
        training_null_rows=int(calibration_null.sum()),
    )


def annotate_empirical_null_state_membership(
    observable_evidence: pd.DataFrame,
    *,
    calibrator: FrozenEmpiricalNullStateCalibrator,
) -> pd.DataFrame:
    """Apply a frozen calibrator without labels or scorer-domain inputs."""

    evidence = observable_feature_evidence(observable_evidence)
    marginal_z, conditional_z, score = _score(evidence, calibrator)
    addition = (
        evidence["empirical_null_calibration_candidate"]
        & score.ge(calibrator.selection_threshold)
    )
    output = evidence.copy()
    output["empirical_null_marginal_state_z"] = marginal_z
    output["empirical_null_conditional_excess_z"] = conditional_z
    output["empirical_null_composite_score"] = score
    output["empirical_null_selection_threshold"] = calibrator.selection_threshold
    output["empirical_null_state_member"] = addition
    output["state_calibrated_supported_member"] = (
        output["state_supported_member"] | addition
    )
    output["state_calibrated_membership_tier"] = output[
        "state_membership_tier"
    ].astype(str)
    output.loc[addition, "state_calibrated_membership_tier"] = TIER
    output["empirical_null_calibration_schema"] = SCHEMA
    output["empirical_null_threshold_method"] = calibrator.threshold_method
    output["exact_cross_assay_pair_claimed"] = False
    output["direct_causal_claimed"] = False
    output["pairwise_expansion_permitted"] = False
    if (
        output["state_calibrated_supported_member"]
        & ~output["state_supported_member"]
        & ~output["empirical_null_state_member"]
    ).any():
        raise RuntimeError("Calibrated membership contains an unsupported promotion")
    if (
        output["state_supported_member"]
        & ~output["state_calibrated_supported_member"]
    ).any():
        raise RuntimeError("Empirical-null tier changed existing membership")
    return output
