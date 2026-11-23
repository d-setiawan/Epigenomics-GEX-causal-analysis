from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_joint_sliced_geometry_rescue_v4 as diagnostic  # noqa: E402


def _asymmetric_scores(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = rng.gamma(shape=2.0, scale=0.8, size=1200)
    second = 0.45 * first + rng.standard_t(df=5, size=1200)
    third = -0.25 * first + 0.35 * np.square(second) + rng.normal(size=1200)
    return np.column_stack((first, second, third))


def test_direction_set_is_closed_under_every_signed_permutation() -> None:
    assert len(diagnostic.INTEGER_DIRECTIONS) == 49
    assert len(diagnostic.SIGNED_PERMUTATIONS) == 48
    for indexes, orientations in diagnostic.TRANSFORM_DIRECTION_MAPS:
        assert sorted(indexes.tolist()) == list(range(49))
        assert set(orientations.tolist()) <= {-1, 1}


def test_joint_cost_recovers_known_permutation_without_row_pairing() -> None:
    left = _asymmetric_scores()
    order = (2, 0, 1)
    signs = np.asarray([-1.0, 1.0, -1.0])
    right = left[:, order] * signs
    right = right[np.random.default_rng(19).permutation(len(right))]

    matrix = diagnostic._restricted_signed_permutation_cost_matrix(left, right)
    expected = np.argsort(np.asarray(order))

    assert np.array_equal(np.argmin(matrix, axis=1), expected)
    assert np.all(matrix[np.arange(3), expected] < 1e-10)


def test_copula_geometry_is_invariant_to_monotone_marginal_changes() -> None:
    left = _asymmetric_scores(13)
    altered = np.column_stack(
        (
            np.exp(left[:, 0] / 3.0),
            np.sign(left[:, 1]) * np.square(left[:, 1]),
            np.arcsinh(left[:, 2]),
        )
    )
    altered = altered[np.random.default_rng(23).permutation(len(altered))]

    matrix = diagnostic._restricted_signed_permutation_cost_matrix(
        left, altered, copula_only=True
    )

    assert np.array_equal(np.argmin(matrix, axis=1), np.arange(3))
    assert np.all(matrix[np.arange(3), np.arange(3)] < 1e-10)
