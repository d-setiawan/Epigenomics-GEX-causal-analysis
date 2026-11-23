from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import joint_reporter_regression_evidence_v4 as joint
import soft_reporter_representation_ensemble_v4 as marginal


def _orthogonal_example():
    rng = np.random.default_rng(41)
    values = rng.normal(size=(300, 5))
    values -= values.mean(axis=0)
    q, _ = np.linalg.qr(values)
    q *= np.sqrt(len(q))
    scores = q[:, :3]
    log_depth = 4 + 0.2 * (q[:, 0] + q[:, 1] + np.sqrt(0.1) * q[:, 3])
    depth = np.expm1(log_depth)
    focal = np.expm1(4 + 0.2 * q[:, [0]])
    return focal, scores, depth


def test_joint_effect_isolates_source_after_depth_induces_marginal_spread():
    focal, scores, depth = _orthogonal_example()
    old = marginal.reporter_axis_evidence(focal, scores, depth)
    corrected = joint.reporter_axis_evidence(focal, scores, depth)
    assert old.weights[1] > 0.40
    assert corrected.identified
    assert corrected.weights[0] == pytest.approx(1.0, abs=1e-12)
    assert corrected.weights[1] < 1e-24
    np.testing.assert_allclose(corrected.coefficients[0], [0.2, 0, 0], atol=1e-14)


def test_energy_equals_nested_regression_r_squared_increment():
    focal, scores, depth = _orthogonal_example()
    rng = np.random.default_rng(88)
    y = np.log1p(focal) + rng.normal(scale=0.12, size=focal.shape)
    focal = np.expm1(y)
    result = joint.reporter_axis_evidence(focal, scores, depth)
    nuisance = np.column_stack([np.ones(len(depth)), np.log1p(depth)])
    base_residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    full = np.column_stack([nuisance, scores])
    full_residual = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    denominator = np.square(base_residual).sum()
    for component in range(3):
        reduced = np.column_stack([nuisance, np.delete(scores, component, axis=1)])
        residual = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        expected = (np.square(residual).sum() - np.square(full_residual).sum()) / denominator
        assert result.mean_squared_correlations[component] == pytest.approx(expected, abs=1e-12)


def test_semipartial_effect_is_invariant_to_axis_scale_sign_and_permutation():
    focal, scores, depth = _orthogonal_example()
    original = joint.reporter_axis_evidence(focal, scores, depth)
    order = [2, 0, 1]
    altered = joint.reporter_axis_evidence(focal, scores[:, order] * [-2, 4, -0.3], depth)
    np.testing.assert_allclose(altered.mean_squared_correlations, original.mean_squared_correlations[order], atol=1e-12)
    same = marginal.soft_pair_evidence(original, original, [0, 1, 2], [0, 1, 2])
    aligned = marginal.soft_pair_evidence(original, altered, [0, 1, 2], [1, 2, 0])
    assert same[marginal.PRIMARY_SCORE] == pytest.approx(aligned[marginal.PRIMARY_SCORE])


def test_exact_collinearity_is_unresolved_without_arbitrary_pseudoinverse_split():
    focal, scores, depth = _orthogonal_example()
    scores[:, 1] = scores[:, 0]
    result = joint.reporter_axis_evidence(focal, scores, depth)
    assert result.identified is False
    assert result.usable is False
    assert result.residual_score_rank == 2
    assert result.unresolved_reason == "depth_residual_score_design_rank_deficient"
    np.testing.assert_array_equal(result.mean_squared_correlations, np.zeros(3))


def test_near_collinearity_has_explicit_condition_and_unique_variance():
    focal, scores, depth = _orthogonal_example()
    scores[:, 1] = scores[:, 0] + 1e-7 * scores[:, 1]
    result = joint.reporter_axis_evidence(focal, scores, depth)
    assert result.identified
    assert result.residual_score_rank == 3
    assert result.residual_score_condition_number > 1e6
    assert result.unique_axis_variance[0] < 1e-12
    assert result.unique_axis_variance[1] < 1e-12
    assert np.isfinite(result.mean_squared_correlations).all()


def test_all_depth_explained_axes_do_not_promote_projection_roundoff():
    focal, _, depth = _orthogonal_example()
    d = np.log1p(depth)
    scores = np.column_stack([d, 2 * d + 1, 3 * d + 4])
    result = joint.reporter_axis_evidence(focal, scores, depth)
    assert result.identified is False
    assert result.residual_score_rank == 0
    assert result.usable is False
    np.testing.assert_array_equal(result.mean_squared_correlations, np.zeros(3))


def test_zero_and_depth_explained_reporters_contribute_zero_to_raw_mean():
    focal, scores, depth = _orthogonal_example()
    active = joint.reporter_axis_evidence(focal, scores, depth)
    alternatives = np.column_stack([focal, np.zeros(len(depth)), np.expm1(0.3 + 0.2 * np.log1p(depth))])
    combined = joint.reporter_axis_evidence(alternatives, scores, depth)
    assert combined.nonzero_reporters == 1
    np.testing.assert_allclose(combined.mean_squared_correlations, active.mean_squared_correlations / 3, atol=1e-15)
    np.testing.assert_allclose(combined.weights, active.weights, atol=1e-12)
    np.testing.assert_array_equal(combined.partial_correlations[1:], np.zeros((2, 3)))
