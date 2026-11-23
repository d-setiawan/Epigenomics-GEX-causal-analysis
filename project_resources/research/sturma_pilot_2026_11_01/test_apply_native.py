"""Checks new heldout encoding and joint-moment calculations against known laws."""
import numpy as np
from apply_native import encode, model_cross_correlation, cross_correlation


def test_fixed_encoder_recovers_sources_without_test_recentering():
    rng = np.random.default_rng(13)
    mixing = rng.normal(size=(7, 3))
    mean = rng.normal(size=7)
    shifted_sources = rng.normal(size=(30, 3)) + 4
    x = shifted_sources @ mixing.T + mean
    assert np.allclose(encode(x, mean, mixing), shifted_sources)


def test_covariance_uses_shared_sources_but_variance_includes_private_sources():
    a = np.array([[2., 3.], [1., 4.]])
    b = np.array([[5., 6.]])
    cov, corr = model_cross_correlation(a, b, a[:, :1], b[:, :1])
    assert np.allclose(cov, [[10.], [5.]])
    assert np.allclose(corr[:, 0], [10 / np.sqrt(13 * 61), 5 / np.sqrt(17 * 61)])
    zero_cov, zero_corr = model_cross_correlation(a, b, a[:, :0], b[:, :0])
    assert np.allclose(zero_cov, 0) and np.allclose(zero_corr, 0)


def test_undefined_reference_correlation_is_not_silently_zero():
    x = np.column_stack([np.arange(6), np.ones(6)])
    actual = cross_correlation(x, -np.arange(6)[:, None])
    assert np.isclose(actual[0, 0], -1)
    assert np.isnan(actual[1, 0])
