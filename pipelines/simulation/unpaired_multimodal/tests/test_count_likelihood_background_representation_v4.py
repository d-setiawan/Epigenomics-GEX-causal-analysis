from __future__ import annotations

from pathlib import Path
from itertools import product
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import softmax

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import count_likelihood_background_representation_v4 as count_model


def _counts():
    """Small numerical optimizer fixture; never written as a benchmark world."""
    rng = np.random.default_rng(153)
    state = np.asarray(list(product([-1, -0.5, 0, 0.4, 3], [-1.5, -1, 0.25, 1, 1.25], [-3, -0.1, 0.1, 0.2, 0.3])))
    logits = state @ rng.normal(scale=0.2, size=(3, 12))
    return np.asarray([rng.multinomial(6000, p) for p in softmax(logits, axis=1)], dtype=float)


def test_analytic_multinomial_gradient_matches_finite_difference():
    rng = np.random.default_rng(100)
    counts = rng.integers(1, 15, size=(7, 8)).astype(float)
    theta = count_model._pack(rng.normal(scale=0.1, size=8), rng.normal(scale=0.2, size=(7, 3)), rng.normal(scale=0.2, size=(8, 3)))
    error = check_grad(lambda p: count_model.multinomial_objective_gradient(p, counts)[0], lambda p: count_model.multinomial_objective_gradient(p, counts)[1], theta)
    assert error < 2e-6


def test_gauge_canonicalization_preserves_exact_reconstruction_probabilities():
    rng = np.random.default_rng(66)
    alpha = rng.normal(size=9)
    z = rng.normal(size=(40, 3)) + [1, 2, 3]
    w = rng.normal(size=(9, 3))
    a2, z2, w2 = count_model.canonicalize_factorization(alpha, z, w)
    np.testing.assert_allclose(softmax(alpha + z @ w.T, axis=1), softmax(a2 + z2 @ w2.T, axis=1), atol=1e-14)
    np.testing.assert_allclose(z2.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(z2.T @ z2 / len(z2), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(w2.mean(axis=0), 0, atol=1e-14)


def test_fixed_loading_projection_has_per_cell_stationarity_and_depth_invariance():
    rng = np.random.default_rng(8)
    w = rng.normal(scale=0.3, size=(12, 3))
    w -= w.mean(axis=0)
    alpha = rng.normal(scale=0.2, size=12)
    counts = rng.integers(1, 20, size=(10, 12)).astype(float)
    projected = count_model.infer_factor_scores(counts, alpha, w)
    scaled = count_model.infer_factor_scores(counts * np.arange(1, 11)[:, None], alpha, w)
    assert projected.informative_mask.all()
    np.testing.assert_allclose(projected.factor_scores, scaled.factor_scores, atol=1e-12)
    np.testing.assert_allclose(scaled.fixed_loading_covariance, projected.fixed_loading_covariance / np.arange(1, 11)[:, None, None], atol=1e-12)
    assert projected.diagnostics["maximum_per_count_newton_decrement_squared"] <= 1e-8
    objective = projected.diagnostics["native_objective_trajectory"]
    assert np.all(np.diff(objective) <= 1e-9)


def test_zero_depth_and_boundary_cells_do_not_receive_fabricated_latent_scores():
    w = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    counts = np.array([[2, 2, 2, 2], [0, 0, 0, 0], [8, 0, 0, 0]], dtype=float)
    result = count_model.infer_factor_scores(counts, np.zeros(4), w)
    np.testing.assert_array_equal(result.informative_mask, [True, False, False])
    assert np.isnan(result.factor_scores[1:]).all()
    assert result.row_status[1] == "zero_context_depth"
    assert result.row_status[2] == "boundary_or_unidentified_finite_mle"


def test_count_fit_improves_likelihood_and_exposes_optimizer_and_projection_audits():
    counts = _counts()
    result = count_model.fit_count_likelihood_background(counts, counts.sum(axis=1))
    diagnostics = result.diagnostics
    assert diagnostics["final_objective_per_count"] < diagnostics["initial_objective_per_count"]
    assert diagnostics["optimizer_success"]
    assert not diagnostics["optimizer_success_proves_global_optimum"]
    assert diagnostics["projection"]["maximum_per_count_newton_decrement_squared"] <= 1e-8
    assert result.scores.shape == (len(counts), 3)
    assert np.isfinite(result.scores).all()
    np.testing.assert_allclose(result.scores.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(result.scores.std(axis=0), 1, atol=1e-12)


def test_feature_fit_cannot_see_out_of_bag_counts():
    counts = _counts()
    indices = np.arange(100)
    original = count_model.fit_count_likelihood_background(counts, counts.sum(axis=1), inbag_indices=indices)
    changed = counts.copy()
    changed[100:] = changed[100:, ::-1]
    other = count_model.fit_count_likelihood_background(changed, changed.sum(axis=1), inbag_indices=indices)
    np.testing.assert_array_equal(original.intercepts, other.intercepts)
    np.testing.assert_array_equal(original.loadings, other.loadings)
    np.testing.assert_allclose(original.scores[indices], other.scores[indices], atol=1e-12)
    assert not np.allclose(original.factor_scores[100:], other.factor_scores[100:])


def test_invalid_depth_zero_feature_and_optimizer_failure_are_explicit(monkeypatch):
    counts = _counts()
    with pytest.raises(ValueError, match="context-row-sum"):
        count_model.fit_count_likelihood_background(counts, counts.sum(axis=1) + 1)
    zero_feature = counts.copy()
    zero_feature[:, 0] = 0
    with pytest.raises(count_model.CountLikelihoodFitError, match="finite intercept"):
        count_model.fit_count_likelihood_background(zero_feature, zero_feature.sum(axis=1))
    monkeypatch.setattr(count_model, "minimize", lambda fun, initial, **kwargs: SimpleNamespace(x=initial, success=False, message="test iteration limit"))
    with pytest.raises(count_model.CountLikelihoodFitError, match="did not converge"):
        count_model.fit_count_likelihood_background(counts, counts.sum(axis=1))
