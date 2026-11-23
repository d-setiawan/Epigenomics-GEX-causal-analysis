from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import ASSAYS, DEFAULT_TOLERANCES, canonical_hash, file_sha256, select_focal_profile, summarize_count_matrix
from simulator import (apply_mechanism, biological_edges, conditional_activity_means,
                       freeze_biology, make_extended_specs, make_simple_spec,
                       prepare_world, rng_for, sample_activities, sample_state,
                       stream_seed, validate_world, write_world)


@pytest.fixture
def fixture_calibration():
    # Deliberately artificial test fixtures, never used by the production CLI.
    result = {"test_fixture_only": True, "tolerances": DEFAULT_TOLERANCES, "assays": {}}
    for assay in ASSAYS:
        rng = rng_for(1234, assay)
        matrix = rng.poisson(rng.uniform(0.03, 0.5, 40), size=(120, 40))
        names = [f"source_{j}" for j in range(40)]
        encoder = summarize_count_matrix(sparse.csr_matrix(matrix), names)
        retained = summarize_count_matrix(sparse.csr_matrix(matrix[:, :20]), names[:20])
        result["assays"][assay] = {
            "encoder": encoder, "retained": retained, "retained_feature_ids": names[:20],
            "focal_profile": select_focal_profile(retained["features"]),
            "relative_exposures": [0.5, 1.0, 1.5],
        }
    result["calibration_hash"] = canonical_hash(result)
    return result


def test_bank_has_locked_30_and_no_unbalanced_histone_order():
    bank = make_extended_specs()
    assert len(bank) == 30
    assert len({x["world_id"] for x in bank}) == 30
    assert sum(x["split"] == "development" for x in bank) == 9
    assert sum(x["split"] == "assessment" for x in bank) == 21
    for family in ("linear", "constrained", "unrestricted"):
        chains = [x for x in bank if x["family"] == family and x["motif"] == "chain"]
        assert sum(x["histone_order"][0] == "H3K27ac" for x in chains) == 2


def test_named_streams_are_stable_and_separate():
    assert stream_seed(5, "state") == stream_seed(5, "state")
    assert stream_seed(5, "state") != stream_seed(5, "exposure")
    a = rng_for(5, "state").normal(size=10)
    rng_for(5, "unrelated").normal(size=100)
    np.testing.assert_array_equal(a, rng_for(5, "state").normal(size=10))


def test_simple_exact_equations_and_sentinel_independence():
    spec = make_simple_spec(17)
    biology = freeze_biology(spec)
    state = sample_state(spec, 30000, rng_for(17, "s"))
    seed = 9
    activities = sample_activities(spec, biology, state, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    a = state[:, 0] + 0.5 * rng.normal(size=len(state))
    b = np.tanh(a) + 0.5 * rng.normal(size=len(state))
    r = np.tanh(b) + 0.5 * rng.normal(size=len(state))
    np.testing.assert_allclose(activities[:, [0, 2, 4]], np.column_stack((r, a, b)))
    assert abs(np.corrcoef(activities[:, 1], activities[:, 0])[0, 1]) < 0.03
    assert ["S0", "RNA_core"] not in biological_edges(spec, biology)


def test_constrained_network_is_monotone_and_frozen():
    spec = next(x for x in make_extended_specs() if x["world_id"] == "constrained_collider_0")
    biology = freeze_biology(spec, 256)
    mechanism = biology["mechanisms"]["RNA"]
    grid = np.column_stack((np.linspace(-4, 4, 1000), np.zeros(1000)))
    predicted = apply_mechanism(grid, mechanism)
    assert np.all(np.diff(predicted) > 0)
    np.testing.assert_array_equal(predicted, apply_mechanism(grid, mechanism))
    assert all(x > 0 for x in biology["noise_scales"].values())
    assert len(mechanism["output"]) == 32


def test_world_io_exact_readout_boundary_and_no_overwrite(tmp_path, fixture_calibration):
    pytest.importorskip("anndata")
    spec = make_simple_spec(8)
    root = tmp_path / "simple"
    manifest = write_world(spec, fixture_calibration, root,
                            feature_counts={a: 18 for a in ASSAYS}, cohort_sizes={"train": 25},
                            calibration_n=96, background_calibration_n=48,
                            feature_chunk_size=8, conditional_mean_repetitions=16)
    assert manifest["status"] == "complete"
    assert validate_world(root)["public_cells"] == 75
    with np.load(root / "truth/train.npz") as truth:
        assert truth["activities"].shape == (75, 6)
        assert truth["counts_realistic"].shape == (75, 6)
        assert truth["conditional_mean_activities"].shape == (75, 6)
    with pytest.raises(FileExistsError):
        write_world(spec, fixture_calibration, root)
    private = json.loads((root / "truth/manifest.json").read_text())
    assert all("feature_id" in c for c in private["candidates"])


def test_ambiguity_visible_bytes_identical_joint_truth_differs(tmp_path, fixture_calibration):
    pytest.importorskip("anndata")
    twins = make_extended_specs(7)[-2:]
    settings = dict(feature_counts={a: 16 for a in ASSAYS},
                    cohort_sizes={"train": 30, "probe": 15, "assessment": 20},
                    calibration_n=96, background_calibration_n=48,
                    feature_chunk_size=8, conditional_mean_repetitions=0)
    first = tmp_path / "zero"
    second = tmp_path / "coupled"
    write_world(twins[0], fixture_calibration, first, **settings)
    write_world(twins[1], fixture_calibration, second, visible_reference=first, **settings)
    for regime in ("realistic", "informative"):
        for assay in ASSAYS:
            relative = Path("observed") / regime / "train" / f"{assay}.h5ad"
            assert file_sha256(first / relative) == file_sha256(second / relative)
    validate_world(first)
    validate_world(second)
    with np.load(first / "truth/train.npz") as a, np.load(second / "truth/train.npz") as b:
        assert not np.array_equal(a["activities"], b["activities"])
        for assay in ASSAYS:
            rows = np.flatnonzero(a["assigned_assay"] == assay)
            cols = np.flatnonzero(a["candidate_assays"] == assay)
            np.testing.assert_array_equal(a["activities"][np.ix_(rows, cols)], b["activities"][np.ix_(rows, cols)])
    assert file_sha256(first / "truth/assessment.npz") != file_sha256(second / "truth/assessment.npz")


def test_regimes_keep_biology_and_parameters_frozen(tmp_path, fixture_calibration):
    pytest.importorskip("anndata")
    spec = make_extended_specs(17)[0]
    root = tmp_path / "extended"
    write_world(spec, fixture_calibration, root, feature_counts={a: 16 for a in ASSAYS},
                cohort_sizes={"train": 80, "val": 7, "discovery": 9, "probe": 8, "assessment": 11},
                calibration_n=128, background_calibration_n=64, conditional_mean_repetitions=0)
    validate_world(root)
    with np.load(root / "truth/train.npz") as truth:
        assert truth["counts_informative"].mean() > truth["counts_realistic"].mean() * 3
        assert len(truth["candidate_names"]) == 9
    params = json.loads((root / "truth/frozen_parameters.json").read_text())
    assert "informative" not in params["observation_parameters"]


def test_conditional_means_are_monte_carlo_not_zero_noise(fixture_calibration):
    spec = make_simple_spec(8)
    prepared = prepare_world(spec, fixture_calibration, {a: 12 for a in ASSAYS}, 96, 48)
    state = np.ones((4, 1))
    means, errors = conditional_activity_means(prepared, state, 111, 5000, 2)
    # Jensen effects propagate through tanh; conditional B mean differs from tanh(S).
    assert np.all(means[:, 4] < np.tanh(1) - 0.01)
    assert np.all(errors >= 0)


def test_tampered_manifest_rejected(tmp_path, fixture_calibration):
    pytest.importorskip("anndata")
    root = tmp_path / "world"
    write_world(make_simple_spec(9), fixture_calibration, root,
                feature_counts={a: 10 for a in ASSAYS}, cohort_sizes={"train": 8},
                calibration_n=32, background_calibration_n=32, conditional_mean_repetitions=0)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["world_id"] = "tampered"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        validate_world(root)


def test_ambiguity_residual_correlations_are_exact_target_in_population():
    twins = make_extended_specs(31)[-2:]
    n = 60000
    state = sample_state(twins[0], n, rng_for(91, "state"))
    assigned = np.tile(np.asarray(ASSAYS), n // 3)
    visible_core = []
    for spec in twins:
        values = sample_activities(spec, freeze_biology(spec, 96), state,
                                    rng_for(91, "activity"), assigned)
        ac_residual = values[:, 3] - state[:, 0]
        rna_residual = values[:, 0] - 0.8 * state[:, 0]
        assert np.corrcoef(ac_residual, rna_residual)[0, 1] == pytest.approx(spec["rho"], abs=0.02)
        own = np.column_stack([values[assigned == a, 3 * j] for j, a in enumerate(ASSAYS)])
        visible_core.append(own)
    np.testing.assert_array_equal(*visible_core)


def test_shared_state_control_has_no_inherited_cross_assay_residuals():
    spec = next(x for x in make_extended_specs(71) if x["world_id"] == "shared_state_0")
    biology = freeze_biology(spec, 96)
    state = sample_state(spec, 30000, rng_for(21, "state"))
    values = sample_activities(spec, biology, state, rng_for(21, "activities"))
    residuals = np.column_stack([values[:, 3 * j] - state @ np.asarray(biology["state_weights"][a])
                                 for j, a in enumerate(ASSAYS)])
    off_diagonal = np.corrcoef(residuals.T)[np.triu_indices(3, 1)]
    assert np.max(np.abs(off_diagonal)) < 0.03


def test_mutated_calibration_is_rejected(fixture_calibration):
    fixture_calibration["assays"]["RNA"]["focal_profile"]["mean"] += 1
    with pytest.raises(ValueError, match="hash"):
        prepare_world(make_simple_spec(), fixture_calibration, {a: 12 for a in ASSAYS}, 32, 32)
