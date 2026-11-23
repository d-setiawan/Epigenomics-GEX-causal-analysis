from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crossfit_multivariate_reporter_rescue_v4 as rescue  # noqa: E402


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=rescue.FEATURE_COLUMNS)


def test_model_matrix_uses_only_frozen_features_and_zero_imputes() -> None:
    frame = _frame([[1.0, 2.0, np.nan, np.inf]])
    frame["modality_identity"] = "must_not_enter_model"

    observed = rescue._model_matrix(frame)

    assert observed.shape == (1, 4)
    assert observed.tolist() == [[1.0, 2.0, 0.0, 0.0]]
    assert "modality_identity" not in rescue.FEATURE_COLUMNS


def test_nested_fold_is_deterministic_and_respects_sign_guard() -> None:
    negative = [
        [0.0, 0.0, 4.0, 0.0],
        [0.2, 0.1, 3.8, 0.1],
        [0.1, 0.3, 3.5, 0.2],
        [0.3, 0.2, 3.7, 0.1],
    ]
    positive = [
        [3.5, 3.8, 0.2, 3.4],
        [4.0, 3.6, 0.0, 3.8],
        [3.7, 4.0, 0.3, 4.0],
        [3.9, 3.7, 0.1, 3.6],
    ]
    training = _frame(negative + positive)
    labels = np.asarray([False] * len(negative) + [True] * len(positive))
    calibration = _frame([negative[0], positive[0]])
    calibration_labels = np.asarray([False, True])
    target = _frame([negative[1], positive[1]])

    first = rescue._fit_fold(
        training, labels, calibration, calibration_labels, target
    )
    second = rescue._fit_fold(
        training, labels, calibration, calibration_labels, target
    )

    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert first[2] == second[2]
    coefficients = first[2]["standardized_logistic_coefficients_json"]
    assert first[2]["coefficient_sign_guard_passed"]
    assert coefficients.startswith("[")
    assert first[1][1] > first[1][0]


def test_coefficient_sign_guard_abstains_on_wrong_or_zero_sign() -> None:
    assert rescue._coefficient_sign_guard([1.0, 2.0, -3.0, 4.0])
    assert not rescue._coefficient_sign_guard([1.0, 2.0, 3.0, 4.0])
    assert not rescue._coefficient_sign_guard([1.0, 2.0, -3.0, 0.0])

