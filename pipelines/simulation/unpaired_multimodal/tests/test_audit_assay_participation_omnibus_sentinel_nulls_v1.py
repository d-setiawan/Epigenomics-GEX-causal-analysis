from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_assay_participation_omnibus_sentinel_nulls_v1 as audit


def _sentinels(
    p1: float,
    p2: float,
    support1: float,
    support2: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["N1", "N2"],
            "n_views": [6, 6],
            "formal_test_eligible": [True, True],
            "empirical_null_p": [p1, p2],
            "reference_conditional_wald": [2.0, 1.0],
            "conditional_wald_view_support": [support1, support2],
        }
    )


def test_single_route_can_live_when_partial_conjunction_replication_fails() -> None:
    result = audit._one_pseudo_assay(_sentinels(0.002, 0.20, 1.0, 5 / 6))
    assert result["single_feature_p"] == pytest.approx(0.004)
    assert result["single_feature_replication_gate"] is True
    assert result["partial_conjunction_replication_gate"] is False
    assert result["winning_route"] == "single_feature"
    assert result["omnibus_p"] == pytest.approx(0.008)


def test_tied_live_routes_prefer_partial_conjunction_for_audit_identity() -> None:
    result = audit._one_pseudo_assay(_sentinels(0.01, 0.02, 1.0, 1.0))
    assert result["single_feature_p"] == pytest.approx(0.02)
    assert result["partial_conjunction_p"] == pytest.approx(0.02)
    assert result["winning_route"] == "partial_conjunction"
    assert result["omnibus_p"] == pytest.approx(0.04)


def test_holm_and_exact_interval_match_reviewed_world_level_audit() -> None:
    assert audit._holm([0.012, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[0] == pytest.approx(
        0.084
    )
    interval = audit._exact_interval(5, 120)
    assert interval["low"] == pytest.approx(0.013665384116662828)
    assert interval["high"] == pytest.approx(0.09455879997593412)


def test_audit_rejects_any_noncanonical_replay_root(tmp_path: Path) -> None:
    replay = tmp_path / "not_the_opened_replay"
    replay.mkdir()
    with pytest.raises(ValueError, match="only the canonical opened TRAIN replay"):
        audit.run_audit(replay, tmp_path / "out")

