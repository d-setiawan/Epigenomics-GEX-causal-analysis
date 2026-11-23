from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_background_ica_representation_v4 as bootstrap


def _example() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(95)
    latent = np.column_stack([rng.exponential(size=400), rng.uniform(-1, 1, 400), rng.laplace(size=400)])
    loadings = rng.normal(scale=0.18, size=(3, 18))
    counts = rng.poisson(np.exp(3.0 + latent @ loadings)).astype(float)
    return counts, {name: latent.copy() for name in ("full", "fold0", "fold1")}


def _fit(counts, references, **kwargs):
    options = dict(modality="assay_a", reference_seed=31, resample_seed=83, initialization_seed=102, mode="both")
    options.update(kwargs)
    return bootstrap.bootstrap_fit_assay_contexts(counts, counts.sum(axis=1), references, **options)


def test_reproducible_finite_refits_preserve_original_feature_folds():
    counts, references = _example()
    first = _fit(counts, references)
    second = _fit(counts, references)
    expected = np.array_split(
        np.random.default_rng(bootstrap.mdcr._stable_seed(31, "assay_a::feature_folds")).permutation(counts.shape[1]), 2
    )
    assert first.oob_mask.sum() >= 8
    assert not first.oob_mask[np.unique(first.inbag_indices)].any()
    for index, values in enumerate(expected):
        np.testing.assert_array_equal(first.feature_indices[f"fold{index}"], values)
    for context in references:
        np.testing.assert_allclose(first.scores[context], second.scores[context], atol=0, rtol=0)
        assert first.scores[context].shape == (400, 3)
        assert np.isfinite(first.scores[context]).all()
        assert first.contexts[context].convergence
        assert first.contexts[context].alignment_cells == first.oob_mask.sum()
    assert first.provenance["reference_estimator_independent_of_alignment_cells"] is False


def test_reference_only_relabels_axes_without_changing_estimated_representation():
    counts, references = _example()
    original = _fit(counts, references)
    order = [2, 0, 1]
    signs = np.array([-1, 1, -1])
    relabeled = _fit(counts, {key: value[:, order] * signs for key, value in references.items()})
    np.testing.assert_array_equal(original.inbag_indices, relabeled.inbag_indices)
    np.testing.assert_array_equal(original.normalization_mean, relabeled.normalization_mean)
    for name in references:
        np.testing.assert_allclose(relabeled.scores[name], original.scores[name][:, order] * signs, atol=1e-12)


def test_normalization_parameters_are_inbag_only_even_when_oob_depth_changes():
    counts, references = _example()
    first = _fit(counts, references)
    changed = counts.copy()
    changed[first.oob_mask, 0] += 50_000
    second = _fit(changed, references)
    np.testing.assert_array_equal(first.normalization_mean, second.normalization_mean)
    np.testing.assert_array_equal(first.normalization_scale, second.normalization_scale)
    assert first.library_scale == second.library_scale
    expected_logged = np.log1p(counts[first.inbag_indices] * (first.library_scale / counts[first.inbag_indices].sum(axis=1))[:, None])
    np.testing.assert_allclose(first.normalization_mean, expected_logged.mean(axis=0))


def test_crossed_modes_separate_resampling_and_initialization():
    counts, references = _example()
    init = _fit(counts, references, mode="initialization_only")
    cell = _fit(counts, references, mode="cell_resample_only", initialization_seed=90)
    cell_other_init = _fit(counts, references, mode="cell_resample_only", initialization_seed=91)
    both = _fit(counts, references, mode="both")
    np.testing.assert_array_equal(init.inbag_indices, np.arange(len(counts)))
    assert not init.oob_mask.any()
    assert all(item.alignment_scope == "all_original_rows" for item in init.contexts.values())
    np.testing.assert_array_equal(cell.inbag_indices, both.inbag_indices)
    for name in references:
        np.testing.assert_array_equal(cell.scores[name], cell_other_init.scores[name])
        token = "assay_a" if name == "full" else f"assay_a__feature_{name}"
        assert cell.contexts[name].fit_seed == bootstrap.mdcr._stable_seed(31, token)
        assert both.contexts[name].fit_seed == bootstrap.mdcr._stable_seed(102, token)
    other_assay_indices, _ = bootstrap._resample_indices(len(counts), modality="assay_b", resample_seed=83, mode="both")
    assert not np.array_equal(cell.inbag_indices, other_assay_indices)


def test_nonconvergence_rejects_whole_replicate(monkeypatch):
    class NonconvergentICA:
        def __init__(self, **kwargs):
            self.n_iter_ = 1

        def fit_transform(self, values):
            warnings.warn("test failed fit", ConvergenceWarning)
            return values[:, :3]

        def transform(self, values):
            pytest.fail("A warned fit must never transform or return evidence")

    monkeypatch.setattr(bootstrap, "FastICA", NonconvergentICA)
    counts, references = _example()
    with pytest.raises(bootstrap.RepresentationFitError, match="did not converge"):
        _fit(counts, references)


def test_api_rejects_misaligned_reference_and_nonobserved_depth():
    counts, references = _example()
    with pytest.raises(ValueError, match="same-assay reference"):
        _fit(counts, {**references, "fold0": references["fold0"][:-1]})
    with pytest.raises(ValueError, match="observed row sum"):
        bootstrap.bootstrap_fit_assay_contexts(counts, counts.sum(axis=1) + 1, references, modality="assay_a", reference_seed=0, resample_seed=0, initialization_seed=0)


def test_alignment_uses_only_requested_same_assay_rows():
    rng = np.random.default_rng(29)
    reference = rng.normal(size=(40, 3))
    scores = reference[:, [1, 2, 0]] * np.array([-1, 1, -1])
    mask = np.arange(40) < 20
    modified_reference = reference.copy()
    modified_reference[~mask] = rng.normal(size=(20, 3)) * 100
    first = bootstrap._alignment(scores, reference, mask)
    second = bootstrap._alignment(scores, modified_reference, mask)
    np.testing.assert_allclose(first[0], reference)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
