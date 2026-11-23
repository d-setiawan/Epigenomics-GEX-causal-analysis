from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import multievidence_empirical_null_state_calibration_v4 as calibration  # noqa: E402


def _evidence() -> pd.DataFrame:
    rows = []
    seed = 1
    for block_index, block in enumerate(("block_a", "block_b", "block_c")):
        for null_index in range(4):
            offset = 0.01 * block_index + 0.005 * null_index
            rows.append(
                {
                    "block": block,
                    "seed": seed,
                    "feature": f"{block}_null_{null_index}",
                    "modality": "chrom_A",
                    "formal_test_eligible": True,
                    "state_supported_member": False,
                    "state_membership_tier": "unresolved",
                    "oof_marginal_state_fit": 0.01 + offset,
                    "oof_partial_state_fit": 0.008 + offset,
                    "oof_joint_fit": 0.02 + 1.5 * offset,
                    "conditional_wald_above_null_95pct": 0.5 + 20 * offset,
                    "median_conditional_score_norm": 0.01 + 0.5 * offset,
                }
            )
        rows.append(
            {
                "block": block,
                "seed": seed,
                "feature": f"{block}_signal",
                "modality": "chrom_A",
                "formal_test_eligible": True,
                "state_supported_member": False,
                "state_membership_tier": "unresolved",
                "oof_marginal_state_fit": 0.25,
                "oof_partial_state_fit": 0.22,
                "oof_joint_fit": 0.30,
                "conditional_wald_above_null_95pct": 15.0,
                "median_conditional_score_norm": 0.20,
            }
        )
        rows.append(
            {
                "block": block,
                "seed": seed,
                "feature": f"{block}_depth_only",
                "modality": "chrom_A",
                "formal_test_eligible": True,
                "state_supported_member": False,
                "state_membership_tier": "unresolved",
                "oof_marginal_state_fit": -0.01,
                "oof_partial_state_fit": -0.01,
                "oof_joint_fit": 0.50,
                "conditional_wald_above_null_95pct": 20.0,
                "median_conditional_score_norm": 0.30,
            }
        )
        rows.append(
            {
                "block": block,
                "seed": seed,
                "feature": f"{block}_base",
                "modality": "chrom_A",
                "formal_test_eligible": True,
                "state_supported_member": True,
                "state_membership_tier": "strict_state_member",
                "oof_marginal_state_fit": 0.01,
                "oof_partial_state_fit": 0.01,
                "oof_joint_fit": 0.02,
                "conditional_wald_above_null_95pct": 0.5,
                "median_conditional_score_norm": 0.01,
            }
        )
        seed += 1
    return pd.DataFrame(rows)


def test_familywise_calibrator_excludes_training_nulls_and_selects_signal() -> None:
    evidence = _evidence()
    null = evidence["feature"].str.contains("_null_")
    fitted = calibration.fit_multievidence_state_calibrator(
        evidence, null_feature=null
    )

    assert fitted.selection_threshold > fitted.maximum_null_family_score
    assert fitted.route_names == calibration.ROUTE_NAMES
    assert fitted.route_weights == calibration.ROUTE_WEIGHTS
    annotated = calibration.annotate_multievidence_state_membership(
        evidence, calibrator=fitted
    )
    assert not annotated.loc[
        null, "multievidence_empirical_null_state_member"
    ].any()
    assert annotated.loc[
        annotated["feature"].str.endswith("_signal"),
        "multievidence_empirical_null_state_member",
    ].all()
    base = annotated["feature"].str.endswith("_base")
    assert annotated.loc[base, "state_multievidence_supported_member"].all()
    assert not annotated.loc[
        base, "multievidence_empirical_null_state_member"
    ].any()
    assert not annotated["exact_cross_assay_pair_claimed"].any()
    assert not annotated["direct_causal_claimed"].any()
    assert not annotated["pairwise_expansion_permitted"].any()


def test_joint_fit_cannot_bypass_positive_state_specific_gate() -> None:
    evidence = _evidence()
    fitted = calibration.fit_multievidence_state_calibrator(
        evidence,
        null_feature=evidence["feature"].str.contains("_null_"),
    )
    annotated = calibration.annotate_multievidence_state_membership(
        evidence, calibrator=fitted
    )
    depth_only = annotated["feature"].str.endswith("_depth_only")

    assert not annotated.loc[depth_only, "multievidence_state_specific_gate"].any()
    assert not annotated.loc[
        depth_only, "multievidence_empirical_null_state_member"
    ].any()


def test_observable_adapter_fails_closed_on_truth_columns() -> None:
    evidence = _evidence()
    evidence["truth_state_member"] = True

    with pytest.raises(ValueError, match="forbidden columns"):
        calibration.observable_feature_evidence(evidence)


def test_frozen_calibrator_payload_round_trip_is_exact() -> None:
    evidence = _evidence()
    fitted = calibration.fit_multievidence_state_calibrator(
        evidence,
        null_feature=evidence["feature"].str.contains("_null_"),
    )

    restored = calibration.FrozenMultievidenceStateCalibrator.from_payload(
        fitted.to_payload()
    )
    assert restored == fitted
