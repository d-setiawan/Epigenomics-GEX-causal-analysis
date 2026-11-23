from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_joint_sliced_null_stability_v4 as diagnostic  # noqa: E402


def _feature_frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=diagnostic.FEATURE_COLUMNS)


def test_joint_stability_feature_contract_is_raw_joint_not_copula() -> None:
    assert diagnostic.FEATURE_DIRECTIONS == (1.0, 1.0, -1.0, 1.0)
    assert len(diagnostic.FEATURE_COLUMNS) == 4
    assert any("joint_sliced_w2" in value for value in diagnostic.FEATURE_COLUMNS)
    assert not any("copula_sliced_w2" in value for value in diagnostic.FEATURE_COLUMNS)


def test_model_matrix_quantizes_before_empirical_ranking() -> None:
    left = _feature_frame([[1.0 + 4e-14, 2.0, 3.0, 4.0]])
    right = _feature_frame([[1.0, 2.0, 3.0, 4.0]])
    assert np.array_equal(
        diagnostic._model_matrix(left), diagnostic._model_matrix(right)
    )


def test_empirical_score_reverses_reporter_cost_direction() -> None:
    training = _feature_frame(
        [[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 3, 1]]
    )
    lower_cost = _feature_frame([[1, 1, 0.5, 1]])
    higher_cost = _feature_frame([[1, 1, 4.0, 1]])
    assert diagnostic._copula_score(training, lower_cost)[0] > (
        diagnostic._copula_score(training, higher_cost)[0]
    )


def test_selection_rule_is_strict_and_requires_nominal_new_proposal() -> None:
    selected, _ = diagnostic._selection_decision(
        nominal_new_proposal=True, candidate_count=90, null_count=10
    )
    assert selected
    tied, _ = diagnostic._selection_decision(
        nominal_new_proposal=True, candidate_count=90, null_count=90
    )
    assert not tied
    bare_majority, _ = diagnostic._selection_decision(
        nominal_new_proposal=True, candidate_count=50, null_count=0
    )
    assert not bare_majority
    baseline_pair, _ = diagnostic._selection_decision(
        nominal_new_proposal=False, candidate_count=100, null_count=0
    )
    assert not baseline_pair
