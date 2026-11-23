"""Population identities and observation boundary of the matched generator."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import count_transport_matched as matched


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_true_model_is_inside_strict_model_class(dimension):
    model = matched.true_model(dimension, 1)
    w, s = np.array(model["state_weights"]), np.array(model["state_sources"])
    mean = np.array(model["source_mean"])
    np.testing.assert_allclose(w@s, mean, atol=1e-14)
    np.testing.assert_allclose((s-mean).T@(w[:, None]*(s-mean)), np.eye(dimension), atol=1e-14)
    for assay, matrix in model["loadings"].items():
        m = np.array(matrix)
        assert np.linalg.matrix_rank(m) == dimension
        np.testing.assert_allclose(s@m.T, model["state_log_rates"][assay], atol=1e-14)
        np.testing.assert_allclose(np.exp(s@m.T), model["state_rates"][assay], atol=1e-14)
    source, target = (np.array(model["loadings"][a]) for a in ("RNA", "H3K27ac"))
    np.testing.assert_allclose(target@np.linalg.pinv(source)@source, target, atol=1e-14)


def test_training_source_streams_are_distinct_and_reproducible():
    model = matched.true_model(2, 0)
    first = matched.draw_sources(model, 100, "fixture", "fit", "RNA")
    np.testing.assert_array_equal(first, matched.draw_sources(model, 100, "fixture", "fit", "RNA"))
    assert not np.array_equal(first, matched.draw_sources(model, 100, "fixture", "fit", "H3K27ac"))
    assert not np.array_equal(first, matched.draw_sources(model, 100, "fixture", "calibration", "RNA"))
    counts = matched.draw_counts(model, "RNA", first, "fixture", "fit")
    assert np.issubdtype(counts.dtype, np.integer) and np.min(counts) >= 0
    np.testing.assert_array_equal(counts, matched.draw_counts(model, "RNA", first, "fixture", "fit"))


def test_symmetry_countercontrol_preserves_marginals_and_changes_joint():
    report = matched.symmetry_control()
    assert report["maximum_marginal_rate_atom_error"] < 1e-12
    assert report["relative_joint_crosscovariance_difference"] > .01


def test_generated_archives_cannot_be_overwritten(tmp_path):
    path = tmp_path / "counts.npz"
    matched.save_npz(path, raw_counts=np.array([[1, 0]]))
    with pytest.raises(FileExistsError):
        matched.save_npz(path, raw_counts=np.array([[0, 1]]))
    with np.load(path) as result:
        np.testing.assert_array_equal(result["raw_counts"], [[1, 0]])
