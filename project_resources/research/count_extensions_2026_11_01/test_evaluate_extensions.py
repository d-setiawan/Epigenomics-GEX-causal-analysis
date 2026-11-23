"""Fixed numerical fixtures only; no new scientific worlds or fitting."""
import numpy as np
import pytest

import evaluate_extensions as ev


@pytest.mark.parametrize("width", [3, 30, 300])
def test_gram_bootstrap_matches_explicit_complete_pair_resamples(width):
    x = ((np.arange(7 * width).reshape(7, width) * 5) % 13).astype(float)
    y = ((np.arange(7 * width).reshape(7, width) * 3 + 1) % 17).astype(float)
    a = np.linspace(1, 2, width)
    b = np.linspace(2, 3, width)
    prediction = np.outer(np.linspace(-.2, .2, width), np.linspace(.1, .3, width))
    indices = np.array([[0, 0, 1, 2, 4, 5, 6], [6, 4, 3, 2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3]])
    weights = np.array([np.bincount(row, minlength=7) for row in indices])
    out = ev.bootstrap_cross_covariance_errors(x, y, prediction, a, b, weights)
    direct_model, direct_zero = [], []
    for row in indices:
        observed = ev.covariance(x[row], y[row]) / np.outer(a, b)
        direct_zero.append(np.sqrt(np.mean(observed**2)))
        direct_model.append(np.sqrt(np.mean((observed - prediction / np.outer(a, b))**2)))
    np.testing.assert_allclose(out["model_rmse"], direct_model, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(out["zero_rmse"], direct_zero, atol=2e-7, rtol=2e-12)


def test_cached_observed_bootstrap_and_zero_prediction_stay_exactly_equal():
    x = np.arange(20).reshape(5, 4).astype(float)
    y = x[:, ::-1] + 3
    _, weights = ev.bootstrap_multiplicities(5, repetitions=11)
    zero = np.zeros((4, 4))
    first = ev.bootstrap_cross_covariance_errors(x, y, zero, np.ones(4), np.ones(4), weights)
    second = ev.bootstrap_cross_covariance_errors(x, y, zero, np.ones(4), np.ones(4), weights,
                                                  first["observed_squared_frobenius_cache"])
    np.testing.assert_array_equal(first["zero_rmse"], second["zero_rmse"])
    np.testing.assert_array_equal(second["rmse_improvement_over_zero"], np.zeros(11))


def test_nb_and_retention_noise_reduces_to_poisson_and_no_double_count():
    counts = np.array([[0., 1., 2.], [1., 4., 3.], [2., 2., 0.], [4., 0., 1.]])
    exposure = np.array([1., 2., 1.5, 3.])
    np.testing.assert_allclose(ev.noise_diagonal_from_kernel(counts, exposure),
                               np.mean(counts / exposure[:, None]**2, axis=0), atol=1e-14)
    nb = ev.noise_diagonal_from_kernel(counts, exposure, theta=2, retention=.8)
    poisson = ev.noise_diagonal_from_kernel(counts, exposure)
    assert np.all(nb >= poisson)
    np.testing.assert_array_equal(ev.noise_diagonal_from_kernel(counts, exposure, theta=2,
                                                               retention=.8, correction=False), np.zeros(3))


def test_invalid_fixed_scale_is_not_silently_masked():
    x = np.arange(12).reshape(4, 3).astype(float)
    with pytest.raises(ValueError, match="no pairs removed"):
        ev.bootstrap_cross_covariance_errors(x, x, np.zeros((3, 3)), np.array([1., 0., 2.]),
                                              np.ones(3), np.ones((1, 4)))


@pytest.mark.parametrize("totals,expected", [([3, 6, 12], 1.), ([4, 4, 4], 0.), ([1, 4, 16], 2.)])
def test_convex_depth_profile_recovers_fixed_power_fixture(totals, expected):
    counts = np.zeros((3, 300))
    counts[:, 0] = totals
    beta, normalizer, _ = ev.profile_beta(counts, np.array([1., 2., 4.]), np.arange(3))
    assert abs(beta - expected) < 1e-10
    assert normalizer == 2


def test_exposure_cannot_accidentally_subtract_only_the30_column_view():
    counts = np.arange(3 * 300).reshape(3, 300) % 5
    public = {"raw_counts": counts, "library_depth": counts.sum(1) + np.array([10, 20, 30])}
    parameters = {"mode": "offset", "normalization": 20, "beta": 1}
    np.testing.assert_array_equal(ev.rates_exposure(public, parameters), [.5, 1., 1.5])
    with pytest.raises(ValueError, match="complete frozen 300"):
        ev.rates_exposure({**public, "raw_counts": counts[:, :30]}, parameters)


def test_whole300_permutation_keeps_common30_control_and_exposure_fixed():
    counts = np.arange(64 * 300).reshape(64, 300) % 7
    outside = np.arange(64) + 100
    public = {"raw_counts": counts, "library_depth": counts.sum(1) + outside,
              "cell_ids": np.array([f"cell-{i}" for i in range(64)])}
    fit, _ = ev.utility.split_indices(public["cell_ids"])
    parameters = {"mode": "offset", "normalization": float(np.median(outside[fit])), "beta": 1}
    base = {"dataset": "fixture", "control": "independent_columns", "variant": {"depth": "offset", "width": 30},
            "exposure_parameters": {"RNA": parameters}}
    cache = {}
    small = ev.training_replay(public, base, "RNA", {"control_seed": 1000}, cache)
    large = ev.training_replay(public, {**base, "variant": {"depth": "offset", "width": 300}},
                               "RNA", {"control_seed": 1000}, cache)
    assert small is large
    assert np.any(small["counts"] != counts)
    np.testing.assert_array_equal(small["exposure"], outside / parameters["normalization"])


def test_original30_anchor_targets_are_identical_between_panel_widths():
    counts = (np.arange(7 * 300).reshape(7, 300) * 3) % 11
    public = {"RNA": {"test": {"raw_counts": counts, "library_depth": counts.sum(1) + 20}}}
    parameters = {"mode": "offset", "normalization": 20, "beta": 1}
    outputs = []
    for width in [30, 300]:
        result = {"variant": {"width": width}, "exposure_parameters": {"RNA": parameters}}
        arrays = {"predicted_observed_covariance_RNA": np.eye(width), "scale_RNA": np.ones(width),
                  "mean_RNA": np.zeros(width)}
        _, scores = ev.within_scores(result, arrays, public, ["RNA"])
        outputs.append(scores["RNA"]["views"]["original30_anchor"])
    assert outputs[0] == outputs[1]


def test_complete_extension_seal_precedes_any_npz_read(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    data = study / "data_v1"
    data.mkdir()
    (data / "manifest.json").write_text('{}')
    variant = {"name": "fixture", "width": 30, "depth": "unit", "theta": None,
               "retention": 1, "correction": True, "orientation": "c3"}
    protocol = {"datasets": ["fixture"], "variants": [variant], "controls": {"original": [0]},
                "expected_arm_records": 1, "data": "data_v1", "data_manifest_sha256": ev.sha(data / "manifest.json")}
    ev.utility.write_json(study / "protocol.json", protocol)
    runs = study / "runs_v1"
    runs.mkdir()
    name = "fixture__fixture__seed0"
    (runs / name).mkdir()
    ev.utility.write_json(runs / name / "result.json", {"dataset": "fixture", "variant": variant,
        "source_seed": 0, "control": "original", "status": "failed"})
    ev.utility.write_json(runs / "summary.json", [{"path": name, "status": "failed"}])
    seal = {"evaluator_read": False, "source_sha256": {"protocol.json": ev.sha(study / "protocol.json")},
            "support_sha256": {}, "data_manifest_sha256": protocol["data_manifest_sha256"],
            "artifacts": {str(p.relative_to(runs)): ev.sha(p) for p in runs.rglob('*.json')}}
    ev.utility.write_json(runs / "provenance.json", seal)
    def forbid(*_args, **_kwargs):
        raise RuntimeError("NPZ read before all extension fits are verified")
    monkeypatch.setattr(np, 'load', forbid)
    ev.verify_seal(study)
    (runs / name / "result.json").write_text('{}')
    with pytest.raises(ValueError, match="Fit artifact changed"):
        ev.verify_seal(study)
