from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_nested_empirical_null_state_calibration_v4 as audit  # noqa: E402
import empirical_null_state_calibration_v4 as calibration  # noqa: E402


def _evidence() -> pd.DataFrame:
    rows = []
    values = {
        "block_a": ((0.01, 1.0), (0.02, 2.0)),
        "block_b": ((0.02, 1.5), (0.03, 2.5)),
        "block_c": ((0.03, 2.0), (0.04, 3.0)),
    }
    seed = 1
    for block, null_values in values.items():
        for index, (marginal, conditional) in enumerate(null_values):
            rows.append(
                {
                    "block": block,
                    "seed": seed,
                    "feature": f"{block}_null_{index}",
                    "modality": "chrom_A",
                    "formal_test_eligible": True,
                    "state_supported_member": False,
                    "state_membership_tier": "unresolved",
                    "oof_marginal_state_fit": marginal,
                    "conditional_wald_above_null_95pct": conditional,
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
                "oof_marginal_state_fit": 0.20,
                "conditional_wald_above_null_95pct": 12.0,
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
                "conditional_wald_above_null_95pct": 1.0,
            }
        )
        seed += 1
    return pd.DataFrame(rows)


def test_calibrator_extrapolates_beyond_every_training_block_maximum() -> None:
    evidence = _evidence()
    null = evidence["feature"].str.contains("_null_")
    fitted = calibration.fit_empirical_null_state_calibrator(
        evidence,
        null_feature=null,
        conditional_weight=1.5,
    )

    assert fitted.selection_threshold > fitted.maximum_null_score
    assert fitted.block_maximum_extrapolation == pytest.approx(
        fitted.maximum_null_score
        - fitted.second_largest_block_maximum_null_score
    )
    annotated = calibration.annotate_empirical_null_state_membership(
        evidence, calibrator=fitted
    )
    assert not annotated.loc[null, "empirical_null_state_member"].any()
    assert annotated.loc[
        annotated["feature"].str.endswith("_signal"),
        "empirical_null_state_member",
    ].all()
    base = annotated["feature"].str.endswith("_base")
    assert annotated.loc[base, "state_calibrated_supported_member"].all()
    assert not annotated.loc[base, "empirical_null_state_member"].any()
    assert not annotated["exact_cross_assay_pair_claimed"].any()
    assert not annotated["direct_causal_claimed"].any()
    assert not annotated["pairwise_expansion_permitted"].any()


def test_observable_adapter_fails_closed_on_truth_columns() -> None:
    evidence = _evidence()
    evidence["truth_state_member"] = True

    with pytest.raises(ValueError, match="forbidden columns"):
        calibration.observable_feature_evidence(evidence)


def test_frozen_calibrator_payload_round_trip_is_exact() -> None:
    evidence = _evidence()
    fitted = calibration.fit_empirical_null_state_calibrator(
        evidence,
        null_feature=evidence["feature"].str.contains("_null_"),
    )

    restored = calibration.FrozenEmpiricalNullStateCalibrator.from_payload(
        fitted.to_payload()
    )
    assert restored == fitted


def test_random_panels_are_reproducible_and_block_balanced() -> None:
    worlds = pd.DataFrame(
        [
            {"block": block, "seed": seed}
            for block in ("a", "b", "c")
            for seed in range(10)
        ]
    )
    first = audit._balanced_random_panels(
        worlds, selection_seed=17, panels=3, panel_size=7
    )
    second = audit._balanced_random_panels(
        worlds, selection_seed=17, panels=3, panel_size=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 21
    for _, panel in first.groupby("panel_id"):
        counts = panel.groupby("block").size()
        assert counts.max() - counts.min() <= 1
        assert not panel.duplicated(["block", "seed"]).any()


def test_weight_choice_requires_zero_inner_false_additions() -> None:
    rows = [
        {
            "grid_index": 0,
            "conditional_weight": 1.25,
            "true_additions": 20,
            "false_additions": 1,
        },
        {
            "grid_index": 1,
            "conditional_weight": 1.5,
            "true_additions": 18,
            "false_additions": 0,
        },
        {
            "grid_index": 2,
            "conditional_weight": 1.75,
            "true_additions": 18,
            "false_additions": 0,
        },
    ]

    assert audit._choose_weight(rows, tie_target=1.5) == 1.5

