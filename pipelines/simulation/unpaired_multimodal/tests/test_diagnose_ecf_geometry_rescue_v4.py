from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_ecf_geometry_rescue_v4 as diagnostic


def test_ecf_cost_is_sign_invariant() -> None:
    rng = np.random.default_rng(17)
    scores = rng.standard_t(df=5, size=(240, 3))
    left = diagnostic._ecf_fingerprint(scores, standardize=True)
    right = diagnostic._ecf_fingerprint(-scores, standardize=True)
    matrix = diagnostic._ecf_cost_matrix(left, right)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-12)


def test_ecf_cost_recovers_component_permutation() -> None:
    rng = np.random.default_rng(23)
    scores = np.column_stack(
        [
            rng.exponential(size=1200) - 1.0,
            rng.standard_t(df=3, size=1200),
            rng.uniform(-2.0, 2.0, size=1200),
        ]
    )
    left = diagnostic._ecf_fingerprint(scores, standardize=True)
    right = diagnostic._ecf_fingerprint(
        scores[:, [1, 2, 0]], standardize=True
    )
    matrix = diagnostic._ecf_cost_matrix(left, right)
    assert matrix[0, 2] < matrix[0, 0]
    assert matrix[1, 0] < matrix[1, 1]
    assert matrix[2, 1] < matrix[2, 2]
