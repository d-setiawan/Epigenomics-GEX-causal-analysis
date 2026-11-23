"""Fixed algebra/likelihood fixtures, not generated recovery benchmarks."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import count_transport_model as model


def fixture():
    return {"a": np.array([[0, 1, 0], [2, 0, 1], [1, 2, 0], [0, 0, 3]], dtype=float),
            "b": np.array([[1, 0], [0, 2], [1, 1], [3, 0], [0, 1]], dtype=float)}


def test_source_grid_has_independent_unit_variance_coordinates():
    bits, z, w = model.source_grid([.12, .27, .41])
    np.testing.assert_allclose(w.sum(), 1, atol=1e-14)
    np.testing.assert_allclose(w @ z, 0, atol=1e-14)
    np.testing.assert_allclose(z.T @ (w[:, None] * z), np.eye(3), atol=1e-14)
    for j, p in enumerate((.12, .27, .41)):
        np.testing.assert_allclose(w[bits[:, j] == 1].sum(), p, atol=1e-14)


def test_likelihood_matches_manual_poisson_mixture():
    x = np.array([[0, 1], [2, 0], [1, 3]])
    weights = np.array([.7, .3])
    rates = np.array([[.4, 1.2], [2.3, .8]])
    manual = []
    for row in x:
        probability = sum(w * np.prod(np.exp(row * np.log(r) - r - gammaln(row + 1)))
                          for w, r in zip(weights, rates))
        manual.append(np.log(probability))
    scores, _ = model.score_rows(x, weights, rates)
    np.testing.assert_allclose(scores, manual, atol=1e-13)
    np.testing.assert_allclose(model.posterior(x, weights, rates).sum(axis=1), 1, atol=1e-13)


def test_transport_and_cross_covariance_match_complete_state_enumeration():
    _, z, w = model.source_grid([.17, .36])
    s = z + np.array([1.4, -.8])
    a = np.array([[.3, -.1], [.2, .4], [-.1, .2]])
    b = np.array([[.1, .2], [-.3, .1]])
    log_a, log_b = s @ a.T, s @ b.T
    transport = b @ np.linalg.pinv(a)
    np.testing.assert_allclose(log_a @ transport.T, log_b, atol=1e-14)
    ra, rb = np.exp(log_a), np.exp(log_b)
    explicit = sum(wi * np.outer(x, y) for wi, x, y in zip(w, ra, rb)) - np.outer(w @ ra, w @ rb)
    np.testing.assert_allclose(model.cross_covariance(w, ra, rb), explicit, atol=1e-14)


def test_strict_initialization_covers_both_polarities_with_same_mean_logits():
    data = fixture()
    widths = {a: x.shape[1] for a, x in data.items()}
    for seed, expected_sign in ((10, 1), (11, -1)):
        _, theta, _, _ = model._setup(data, 2, "strict", seed)
        _, _, _, _, weights, mu, matrices, _, logits = model._decode(theta, widths, 2, "strict")
        assert mu[0] == expected_sign * 8
        for assay, x in data.items():
            expected_logits = np.log(np.maximum(x.mean(axis=0), 1e-4))
            np.testing.assert_allclose(matrices[assay] @ mu, expected_logits, atol=1e-14)
            np.testing.assert_allclose(weights @ logits[assay], expected_logits, atol=1e-14)


@pytest.mark.parametrize("variant", ["strict", "affine"])
def test_analytic_gradient_matches_finite_differences(variant):
    data, theta, _, _ = model._setup(fixture(), 2, variant, 31)
    _, analytic = model._objective(theta, data, 2, variant)
    numeric = np.empty_like(theta)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        numeric[j] = (model._objective(plus, data, 2, variant)[0]
                      - model._objective(minus, data, 2, variant)[0]) / 2e-5
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("variant", ["strict", "affine"])
def test_per_assay_objective_is_invariant_to_repeating_its_rows(variant):
    data, theta, _, _ = model._setup(fixture(), 2, variant, 13)
    repeated = dict(data)
    repeated["a"] = np.repeat(data["a"], 3, axis=0)
    loss, grad = model._objective(theta, data, 2, variant)
    other, other_grad = model._objective(theta, repeated, 2, variant)
    np.testing.assert_allclose(other, loss, atol=1e-14)
    np.testing.assert_allclose(other_grad, grad, atol=1e-13)


@pytest.mark.parametrize("variant", ["strict", "affine"])
def test_small_fixed_fit_preserves_exact_model_and_warm_start(variant):
    data = fixture()
    fit = model.fit_model(data, 2, variant, 11, maxiter=30)
    json.dumps(fit, allow_nan=False)
    assert fit["loss"] <= fit["initial_loss"] + 1e-9
    assert fit["iterations"] <= 30
    assert isinstance(fit["converged"], bool)
    scores = [model.score_rows(data[a], fit["state_weights"], fit["state_rates"][a])[0].mean()
              / data[a].shape[1] for a in data]
    np.testing.assert_allclose(fit["loss"], -np.mean(scores), atol=1e-12)
    warm = model.fit_model(data, 2, variant, 99, maxiter=1, initial=fit)
    np.testing.assert_allclose(warm["initial_loss"], fit["loss"], atol=1e-12)
    bad = dict(fit, dimension=3)
    with pytest.raises(ValueError, match="schema"):
        model.fit_model(data, 2, variant, 99, maxiter=1, initial=bad)


@pytest.mark.parametrize("bad", [np.array([[.5]]), np.array([[-1]]), np.array([[np.nan]]),
                                 np.array([[np.inf]]), np.empty((0, 2)), np.zeros(2)])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError, match="counts"):
        model.fit_model({"a": bad}, 1, "strict", 0)


def test_invalid_model_and_mixture_rejected():
    with pytest.raises(ValueError, match="dimension"):
        model.fit_model(fixture(), 4, "strict", 0)
    with pytest.raises(ValueError, match="variant"):
        model.fit_model(fixture(), 1, "other", 0)
    with pytest.raises(ValueError, match="positive-rate"):
        model.score_rows([[1]], [1], [[0]])
    with pytest.raises(ValueError, match="positive-rate"):
        model.score_rows([[1]], [.5], [[1]])
    with pytest.raises(ValueError, match="width"):
        model.score_rows([[1, 2]], [1], [[1]])
