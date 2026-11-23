"""Deterministic numerical/contract fixtures; no generated experimental worlds."""
import json

import numpy as np
import pytest

import evaluate_count_sources as ev


def test_covariance_uses_same_fixed_training_scale_and_zero_is_not_improvement():
    left = np.array([[0., 2.], [2., 0.], [4., 4.], [6., 2.]])
    right = np.array([[3., 0.], [1., 2.], [5., 4.], [3., 6.]])
    observed = np.array([[1., 5.], [2., 1.]])
    np.testing.assert_allclose(ev.covariance(left, right), observed)
    scale = np.outer([2., 4.], [5., 10.])
    exact = ev.covariance_comparison(observed / scale, observed / scale)
    zero = ev.covariance_comparison(np.zeros((2, 2)), observed / scale)
    assert exact["model"]["rmse"] == 0
    assert exact["relative_rmse_improvement_over_zero"] == 1
    assert zero["rmse_improvement_over_zero"] == 0
    assert zero["relative_rmse_improvement_over_zero"] == 0


def test_paired_bootstrap_resamples_rows_jointly_and_preserves_zero_baseline():
    left = np.arange(12, dtype=float).reshape(6, 2)
    right = 3 * left + 7
    indices = np.array([[0, 2, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]])
    covariances = ev.bootstrap_covariances(left, right, indices)
    for b, idx in enumerate(indices):
        np.testing.assert_allclose(covariances[b], 3 * ev.covariance(left[idx]))
    arrays, summary = ev.bootstrap_errors(np.zeros((2, 2)), covariances, np.ones((2, 2)))
    np.testing.assert_array_equal(arrays["rmse_improvement_over_zero"], [0., 0.])
    assert summary["fraction_with_positive_improvement"] == 0


def test_proxy_exposure_is_frozen_before_control_permutations():
    counts = (np.arange(64 * 3).reshape(64, 3) % 7).astype(int)
    public = {"raw_counts": counts, "library_depth": counts.sum(axis=1) + np.arange(64) + 100,
              "cell_ids": np.array([f"cell-{i}" for i in range(64)])}
    fit, _ = ev.split_indices(public["cell_ids"])
    normalizer = float(np.median(np.arange(64)[fit] + 100))
    result = {"dataset": "fixture", "exposure": "leave_panel_out", "control": "independent_columns",
              "exposure_normalization": {"RNA": normalizer}}
    changed, exposure, _, _ = ev.replay_training_counts(public, result, "RNA", {"control_seed": 1000})
    assert np.any(changed != counts)
    np.testing.assert_allclose(exposure, (np.arange(64) + 100) / normalizer)
    assert np.any(np.abs(exposure - (public["library_depth"] - changed.sum(axis=1)) / normalizer) > 0)
    np.testing.assert_array_equal(public["raw_counts"], counts)


def test_zero_sources_have_unavailable_state_recovery_not_perfect_recovery():
    ids = np.array(["a", "b", "c", "d"])
    public = {"RNA": {"test": {"cell_ids": ids}}}
    reference = {"ids_RNA": ids, "state_RNA": np.array([[0., 1.], [1., 0.], [2., 2.], [3., 4.]])}
    result = {"pair_matches": [], "all_assay_matches": []}
    arrays = {"mean_RNA": np.zeros(2), "unmixing_RNA": np.empty((0, 2))}
    saved = {}
    report = ev.evaluate_state(result, arrays, public, {"RNA": np.ones((4, 2))}, reference, ["RNA"], saved)
    assert report["RNA"]["all_sources"]["score_count"] == 0
    assert report["RNA"]["all_sources"]["max_absolute_correlation_per_state"] == [None, None]
    assert saved["state_correlation_all_sources_RNA"].shape == (0, 2)


def test_paired_reference_requires_actual_index_join_not_row_order():
    assays = ["RNA", "H3K27ac", "H3K4me3"]
    public, reference, blocks, features = {}, {"canonical_ids": np.array([f"pair-{i}" for i in range(595)])}, [], []
    for a, assay in enumerate(assays):
        counts = np.arange(595 * 30).reshape(595, 30) + a
        feature_ids = np.array([f"{assay}-{j}" for j in range(30)])
        public[assay] = {"test": {"raw_counts": counts, "feature_ids": feature_ids}}
        idx = np.roll(np.arange(595), a + 1)
        reference[f"paired_indices_{assay}"] = idx
        blocks.append(counts[idx])
        features.extend(feature_ids)
    reference.update(raw_counts=np.concatenate(blocks, axis=1), column_ids=np.array(features),
                     column_assays=np.repeat(assays, 30))
    ev.verify_real_reference(reference, public, assays)
    reference["paired_indices_H3K27ac"] = np.arange(595)
    with pytest.raises(ValueError, match="Paired count join differs"):
        ev.verify_real_reference(reference, public, assays)


def test_missing_or_changed_fit_seal_fails_before_any_array_read(tmp_path, monkeypatch):
    study = tmp_path / "research" / "study"
    study.mkdir(parents=True)
    data = study / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{}')
    protocol = {"datasets": ["fixture"], "exposures": ["unit"], "methods": ["poisson_dica"],
                "controls": {"original": [0]}, "data": "data", "expected_arm_records": 1,
                "data_manifest_sha256": ev.sha(data / "manifest.json")}
    ev.write_json(study / "protocol.json", protocol)
    runs = study / "runs_v1"
    runs.mkdir()
    name = "fixture__unit__poisson_dica__seed0"
    (runs / name).mkdir()
    ev.write_json(runs / name / "result.json", {"dataset": "fixture", "exposure": "unit", "method": "poisson_dica",
                                                "source_seed": 0, "control": "original", "status": "failed"})
    ev.write_json(runs / "summary.json", [{"path": name, "status": "failed"}])
    seal = {"evaluator_read": False, "source_sha256": {"protocol.json": ev.sha(study / "protocol.json")},
            "support_sha256": {}, "data_manifest_sha256": protocol["data_manifest_sha256"],
            "artifacts": {str(p.relative_to(runs)): ev.sha(p) for p in runs.rglob("*.json")}}
    ev.write_json(runs / "provenance.json", seal)
    def forbid(*_args, **_kwargs):
        raise RuntimeError("No NPZ may be read during fit-seal verification")
    monkeypatch.setattr(np, "load", forbid)
    ev.verify_fit_seal(study)
    (runs / name / "result.json").write_text('{}')
    with pytest.raises(ValueError, match="Sealed fit artifact changed"):
        ev.verify_fit_seal(study)


def test_noise_diagonal_cannot_be_added_to_uncorrected_covariance_twice():
    assay = "RNA"
    counts = (np.arange(60 * 30).reshape(60, 30) % 11).astype(int)
    train = {"raw_counts": counts, "library_depth": counts.sum(axis=1) + 1000,
             "cell_ids": np.array([f"cell-{i}" for i in range(60)]),
             "feature_ids": np.array([f"f-{i}" for i in range(30)])}
    fit, cal = ev.split_indices(train["cell_ids"])
    rate = counts[fit].astype(float)
    mixing = np.ones((30, 1))
    arrays = {"features_RNA": train["feature_ids"], "fit_indices_RNA": fit, "calibration_indices_RNA": cal,
              "mean_RNA": rate.mean(axis=0), "scale_RNA": rate.std(axis=0),
              "noise_diagonal_RNA": np.zeros(30), "poisson_noise_diagonal_RNA": rate.mean(axis=0),
              "mixing_RNA": mixing, "unmixing_RNA": np.ones((1, 30)) / 30,
              "predicted_observed_covariance_RNA": mixing @ mixing.T}
    result = {"dataset": "fixture", "exposure": "unit", "control": "original", "method": "uncorrected_rank_matched",
              "exposure_normalization": {"RNA": 1}, "assays": [{"assay": assay, "status": "ok", "k": 1}],
              "pair_matches": []}
    protocol = {"assays": [assay], "control_seed": 1000}
    ev.validate_arm_arrays(result, arrays, {assay: {"train": train}}, protocol)
    arrays["predicted_observed_covariance_RNA"] += np.diag(rate.mean(axis=0))
    with pytest.raises(ValueError, match="wrong measurement-noise semantics"):
        ev.validate_arm_arrays(result, arrays, {assay: {"train": train}}, protocol)


def test_undefined_secondary_pairs_are_not_perfect_scores():
    result = ev.errors(np.full((2, 2), np.nan), np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
    assert result["status"] == "unscorable"
    assert result["entries"] == 0
    assert result["rmse"] is None


def test_abstained_endpoint_keeps_default_zero_distinct_from_independence():
    assays = ["RNA", "H3K27ac", "H3K4me3"]
    result = {"exposure": "unit", "exposure_normalization": dict.fromkeys(assays, 1.),
              "pair_matches": [], "assays": [
                  {"assay": assay, "status": "abstained", "reason": "fixture failure", "k": 1}
                  for assay in assays]}
    rates = {a: np.zeros((595, 30)) for a in assays}
    reference = {f"paired_indices_{a}": np.arange(595) for a in assays}
    arrays = {}
    for a in assays:
        arrays[f"scale_{a}"] = np.ones(30)
        arrays[f"predicted_observed_covariance_{a}"] = np.full((30, 30), np.nan)
    for a, b in [(assays[0], assays[1]), (assays[0], assays[2]), (assays[1], assays[2])]:
        arrays[f"predicted_cross_covariance_{a}_{b}"] = np.zeros((30, 30))
    cache = {("unit", (1., 1., 1.)): {(a, b): np.zeros((200, 30, 30))
              for a, b in [(assays[0], assays[1]), (assays[0], assays[2]), (assays[1], assays[2])]}}
    report = ev.evaluate_real(result, arrays, rates, reference, assays, {}, cache)
    pair = report["RNA__H3K27ac"]
    assert pair["endpoint_source_fits"]["RNA"]["status"] == "abstained"
    assert pair["zero_predictor_meaning"].startswith("Empty-candidate default")
    assert pair["primary"]["rmse_improvement_over_zero"] == 0
    assert pair["secondary_pearson"]["model"]["status"] == "unscorable"
