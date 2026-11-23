from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import topology_consensus_state_calibration_v4 as calibration  # noqa: E402


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block": "train_a",
                "seed": 1,
                "feature": "null_a",
                "modality": "mark_a",
                "formal_test_eligible": True,
                "oof_marginal_state_fit": -0.01,
                "oof_partial_state_fit": -0.01,
                "within_assay_direct_candidate_degree": 0,
            },
            {
                "block": "train_a",
                "seed": 1,
                "feature": "signal_a",
                "modality": "mark_a",
                "formal_test_eligible": True,
                "oof_marginal_state_fit": 0.01,
                "oof_partial_state_fit": 0.02,
                "within_assay_direct_candidate_degree": 1,
            },
            {
                "block": "train_b",
                "seed": 2,
                "feature": "null_b",
                "modality": "mark_b",
                "formal_test_eligible": True,
                "oof_marginal_state_fit": 0.01,
                "oof_partial_state_fit": 0.01,
                "within_assay_direct_candidate_degree": 0,
            },
            {
                "block": "train_b",
                "seed": 2,
                "feature": "signal_b",
                "modality": "mark_b",
                "formal_test_eligible": True,
                "oof_marginal_state_fit": 0.02,
                "oof_partial_state_fit": 0.01,
                "within_assay_direct_candidate_degree": 2,
            },
        ]
    )


def test_certified_conjunction_adds_only_weak_state_direct_overlap() -> None:
    evidence = _evidence()
    null = evidence["feature"].str.startswith("null")
    calibrator = calibration.fit_topology_consensus_state_calibrator(
        evidence, null_feature=null
    )
    assert calibrator.sentinel_certified is True
    assert calibrator.training_rule_eligible_null_rows == 0
    upstream = pd.Series([False, False, False, True], index=evidence.index)
    tier = pd.Series(["unresolved", "unresolved", "unresolved", "strict"], index=evidence.index)
    annotated = calibration.annotate_topology_consensus_state_membership(
        evidence,
        upstream_supported=upstream,
        upstream_tier=tier,
        calibrator=calibrator,
    )
    assert annotated["topology_consensus_state_member"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert annotated["state_v427_supported_member"].tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert annotated.loc[1, "state_v427_membership_tier"] == calibration.TIER
    assert not annotated["exact_cross_assay_pair_claimed"].any()
    assert not annotated["direct_causal_claimed"].any()
    assert not annotated["pairwise_expansion_permitted"].any()


def test_training_sentinel_hit_disables_all_future_additions() -> None:
    evidence = _evidence()
    evidence.loc[0, "oof_marginal_state_fit"] = 0.01
    evidence.loc[0, "oof_partial_state_fit"] = 0.01
    evidence.loc[0, "within_assay_direct_candidate_degree"] = 1
    calibrator = calibration.fit_topology_consensus_state_calibrator(
        evidence,
        null_feature=evidence["feature"].str.startswith("null"),
    )
    assert calibrator.sentinel_certified is False
    assert calibrator.training_rule_eligible_null_rows == 1
    annotated = calibration.annotate_topology_consensus_state_membership(
        evidence,
        upstream_supported=[False] * len(evidence),
        upstream_tier=["unresolved"] * len(evidence),
        calibrator=calibrator,
    )
    assert not annotated["topology_consensus_state_member"].any()
    assert not annotated["topology_consensus_calibration_active"].any()


def test_state_fit_and_direct_support_are_both_required() -> None:
    evidence = _evidence()
    calibrator = calibration.fit_topology_consensus_state_calibrator(
        evidence,
        null_feature=evidence["feature"].str.startswith("null"),
    )
    application = evidence.copy()
    application.loc[1, "oof_partial_state_fit"] = 0.0
    application.loc[3, "within_assay_direct_candidate_degree"] = 0
    annotated = calibration.annotate_topology_consensus_state_membership(
        application,
        upstream_supported=[False] * len(application),
        upstream_tier=["unresolved"] * len(application),
        calibrator=calibrator,
    )
    assert not annotated["topology_consensus_state_member"].any()


def test_observable_adapter_rejects_truth_columns_and_roundtrips_payload() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="forbidden columns"):
        calibration.observable_topology_evidence(
            evidence.assign(truth_role="N1")
        )
    calibrator = calibration.fit_topology_consensus_state_calibrator(
        evidence,
        null_feature=evidence["feature"].str.startswith("null"),
    )
    assert (
        calibration.FrozenTopologyConsensusStateCalibrator.from_payload(
            calibrator.to_payload()
        )
        == calibrator
    )
