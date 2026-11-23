from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import (assess_profile_fit, canonical_hash, fit_count_parameters,
                         sample_counts, select_focal_profile, summarize_count_matrix)


def test_sparse_moments_are_exact_and_explicit_zeros_not_detections():
    matrix = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 0], [3, 0, 0]])
    result = summarize_count_matrix(sparse.csr_matrix(matrix), ["a", "b", "c"], 2)
    for j, record in enumerate(result["features"]):
        assert record["mean"] == np.mean(matrix[:, j])
        assert record["variance"] == np.var(matrix[:, j], ddof=1)
        assert record["detection"] == np.mean(matrix[:, j] > 0)
        assert record["positive_mean"] == np.mean(matrix[matrix[:, j] > 0, j])
    assert result["library_totals"] == [3.0, 4.0, 1.0, 3.0]


def test_focal_lower_median_tie_id_and_coherent_profile():
    features = [{"feature_id": name, "n_detected": 1, "detection": d, "mean": mean}
                for name, d, mean in [("z", 0.1, 3), ("b", 0.1, 2), ("a", 0.1, 1), ("k", 0.9, 7)]]
    features.append({"feature_id": "zero", "n_detected": 0, "detection": 0, "mean": 0})
    assert select_focal_profile(features) == features[1]


def test_noninteger_and_negative_sources_rejected():
    for values in ([[1.2], [2]], [[-1], [2]]):
        with pytest.raises(ValueError):
            summarize_count_matrix(sparse.csr_matrix(values), ["a"])


def test_frozen_link_matches_mean_and_reports_incompatible_variance():
    rng = np.random.default_rng(4)
    activity = rng.normal(size=2000)
    exposure = rng.lognormal(-0.125, 0.5, size=2000)
    target = {"mean": 0.05, "variance": 0.02, "detection": 0.045, "positive_mean": 0.05 / 0.045}
    fit = fit_count_parameters(activity, exposure, target)
    assert fit["expected"]["mean"] == pytest.approx(target["mean"], rel=1e-10)
    audit = assess_profile_fit(target, fit["expected"])
    assert audit["checks"]["mean"]["passed"]
    assert not audit["checks"]["variance"]["passed"]
    assert not audit["passed"]


def test_nb_variance_and_counts_without_forced_detection():
    activity = np.zeros(3000)
    exposure = np.ones(3000)
    target = {"mean": 0.1, "variance": 0.2, "detection": 0.07, "positive_mean": 0.1 / 0.07}
    fit = fit_count_parameters(activity, exposure, target, "nb")
    assert fit["expected"]["variance"] == pytest.approx(0.2, rel=1e-8)
    result = sample_counts(np.random.default_rng(7), activity, exposure, fit)
    assert result.dtype == np.int32
    assert np.count_nonzero(result) < 400
    assert (result == 0).any()
    empty = sample_counts(np.random.default_rng(7), activity, exposure,
                          dict(fit, structural_zero=True))
    assert not np.any(empty)


def test_canonical_hash_ignores_dict_order_not_parameters():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_high_mean_feature_is_not_clipped_by_intercept_bracket():
    target = {"mean": 500.0, "variance": 700.0, "detection": 1.0, "positive_mean": 500.0}
    fitted = fit_count_parameters(np.zeros(100), np.ones(100), target, "nb")
    assert fitted["expected"]["mean"] == pytest.approx(500.0, rel=1e-10)
