from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_depth_conditioned_geometry_rescue_v4 as diagnostic


def test_balanced_depth_bins_are_deterministic_and_balanced() -> None:
    depth = np.asarray([2, 1, 1, 4, 3, 2, 4, 3], dtype=float)
    cells = [f"cell_{index}" for index in range(len(depth))]
    first = diagnostic._balanced_depth_bins(depth, cells)
    second = diagnostic._balanced_depth_bins(depth, cells)
    assert np.array_equal(first, second)
    assert np.bincount(first, minlength=4).tolist() == [2, 2, 2, 2]


def test_depth_cost_is_invariant_to_component_sign() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(80, 3))
    depth = rng.integers(10, 100, size=80).astype(float)
    cells = [f"cell_{index:03d}" for index in range(80)]
    left = diagnostic._depth_fingerprint(
        scores, depth, cells, standardize=True
    )
    right = diagnostic._depth_fingerprint(
        -scores, depth, cells, standardize=True
    )
    matrix = diagnostic._fingerprint_cost_matrix(left, right)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-12)


def test_depth_cost_prefers_matching_components() -> None:
    rng = np.random.default_rng(11)
    depth = np.arange(120, dtype=float)
    cells = [f"cell_{index:03d}" for index in range(120)]
    base = np.column_stack(
        [
            rng.normal(loc=np.repeat(np.arange(4), 30), scale=0.2),
            rng.standard_t(df=4, size=120),
            rng.uniform(-2, 2, size=120),
        ]
    )
    left = diagnostic._depth_fingerprint(
        base, depth, cells, standardize=False
    )
    right = diagnostic._depth_fingerprint(
        base[:, [1, 2, 0]], depth, cells, standardize=False
    )
    matrix = diagnostic._fingerprint_cost_matrix(left, right)
    assert matrix[0, 2] < matrix[0, 0]
    assert matrix[1, 0] < matrix[1, 1]
    assert matrix[2, 1] < matrix[2, 2]
