"""Small deterministic boundary/decision checks, not new benchmark worlds."""
import numpy as np

from apply_extensions import (FULL_MARGINS, MOMENT_MARGINS, bootstrap_rows,
                              depth_parameters, effective_gate, equivalence,
                              exposure_values, unique_matches)


def test_depth_elasticity_is_training_only_and_preserves_common_proxy():
    outside = np.arange(1, 13, dtype=float) * 10
    counts = np.zeros((12, 300))
    counts[:, 0] = 3 * outside
    depth = outside + counts.sum(axis=1)
    rows = np.arange(8)
    params = depth_parameters(counts, depth, rows, "elastic")
    np.testing.assert_allclose(params["beta"], 1., atol=1e-6)
    modified, modified_depth = counts.copy(), depth.copy()
    modified[8:, 100] += 1000
    modified_depth[8:] += 1000
    assert depth_parameters(modified, modified_depth, rows, "elastic") == params
    np.testing.assert_allclose(exposure_values(counts, depth, params),
                               exposure_values(modified, modified_depth, params))
    unit = depth_parameters(counts, depth, rows, "unit")
    np.testing.assert_array_equal(exposure_values(counts, depth, unit), np.ones(12))


def test_cf_can_resolve_moment_sign_ambiguity_without_relaxing_equivalence():
    moment = dict(value=np.zeros((1, 2)), bootstrap=np.zeros((32, 1, 2)), eligible=np.array([True]))
    first = [equivalence("RNA", "H3K27ac", 0, 0, sign, moment, moment, MOMENT_MARGINS) for sign in [1, -1]]
    assert all(d["equivalent"] for d in first)
    assert unique_matches(first) == []
    point = np.array([[0., 0., .9, .8, .7, .2, .3, .4]])
    full = dict(value=point, bootstrap=np.repeat(point[None], 32, axis=0), eligible=np.array([True]))
    confirmed = [equivalence("RNA", "H3K27ac", 0, 0, d["sign"], full, full, FULL_MARGINS) for d in first]
    matches = unique_matches([d for d in confirmed if d["equivalent"]])
    assert len(matches) == 1 and matches[0]["sign"] == 1
    uncertain = dict(full, bootstrap=full["bootstrap"].copy())
    uncertain["bootstrap"][::2, 0, 0] += 10
    assert not equivalence("RNA", "H3K27ac", 0, 0, 1, full, uncertain, FULL_MARGINS)["equivalent"]


def test_pgf_effective_cell_gate_uses_worst_processing_point():
    model = dict(orientation="pgf", k_fitted=2, pgf_effective_cells=np.array([500.] * 7 + [49.]))
    assert not effective_gate(model, 200)
    model["pgf_effective_cells"][-1] = 50
    assert effective_gate(model, 200)
    assert not effective_gate(model, 1000)
    model["orientation"] = "c3"
    assert effective_gate(model, 1000)


def test_bootstrap_replay_draws_fit_before_calibration_and_is_repeatable():
    left = bootstrap_rows(20, 7, 42, 3)
    right = bootstrap_rows(20, 7, 42, 3)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)
    assert left[0].shape == (20,) and left[1].shape == (7,)
    assert np.all(left[0] < 20) and np.all(left[1] < 7)


def test_rejected_cf_values_remain_auditable_without_becoming_eligible(monkeypatch):
    import apply_extensions as runner
    dataset, assay = "audit_fixture", "RNA"
    variant = dict(name="fixture", width=1, correction=True, theta=None, retention=1.)
    monkeypatch.setitem(runner.TRAINING, dataset, {assay: dict(fit=np.arange(4), cal=np.arange(4, 8))})
    monkeypatch.setitem(runner.PREPARED, (dataset, "fixture", "original", assay),
                        dict(counts=np.zeros((8, 1)), exposure=np.ones(8)))
    def rejected(*args, **kwargs):
        return dict(value=np.full((1, 3), 2.+0j), se=np.full((1, 3), .01), valid=np.array([False]),
                    max_logweight=np.zeros((1, 3)), mean_absolute_conditioning_proxy=np.zeros((1, 3)),
                    numerical_valid=np.ones((1, 3), bool))
    monkeypatch.setattr(runner, "generalized_cf", rejected)
    fp = dict(value=np.zeros((1, 2)), bootstrap=np.zeros((32, 1, 2)), eligible=np.array([True]),
              bootstrap_unmixing=np.ones((32, 1, 1)), bootstrap_mean=np.zeros((32, 1)))
    full, saved, _ = runner.confirmation_fingerprints(dataset, variant, 0, "original", assay,
        dict(unmixing=np.ones((1, 1)), mean=np.zeros(1)), fp, [0])
    assert not full["eligible"].any() and np.isnan(saved["bootstrap"]).all()
    np.testing.assert_array_equal(saved["bootstrap_raw_value"], np.full((32, 1, 3), 2.+0j))
    np.testing.assert_allclose(saved["bootstrap_se"], .01)
    assert saved["bootstrap_numerical_valid"].all()
