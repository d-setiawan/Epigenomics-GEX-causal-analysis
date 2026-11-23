from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ordered_variance_alignment_v4 as ordered_alignment  # noqa: E402


def _anisotropic_backgrounds(
    *, seed: int = 4
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Independent cells/features with three reproducibly ordered factors."""

    rng = np.random.default_rng(seed)
    matrices: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    for modality in ("rna", "chrom_A", "chrom_B"):
        n_cells = 900
        state = rng.normal(size=(n_cells, 3))
        # Unequal numbers of exchangeable indicators make the three covariance
        # eigenspaces separated without sharing feature identities across assays.
        axes = np.repeat(np.arange(3), [90, 55, 35])
        signs = rng.choice([-1.0, 1.0], size=len(axes))
        baselines = rng.normal(0.0, 0.08, size=len(axes))
        rates = np.exp(
            3.0
            + 0.55 * state[:, axes] * signs[None, :]
            + baselines[None, :]
        )
        matrix = rng.poisson(rates).astype(np.float64)
        matrices[modality] = matrix
        depths[modality] = matrix.sum(axis=1)
        states[modality] = state
    return matrices, depths, states


def test_ordered_variance_recovers_projective_state_order_without_pairing() -> None:
    matrices, depths, states = _anisotropic_backgrounds()

    result = ordered_alignment.align_ordered_variance_background_states(
        matrices,
        depths,
        config=ordered_alignment.OrderedVarianceAlignmentConfig(
            min_fold_component_correlation=0.55,
            min_relative_variance_gap=0.05,
        ),
    )

    assert all(result.assay_eligible.values())
    for modality, embedding in result.embeddings.items():
        correlations = np.abs(
            np.corrcoef(embedding.T, states[modality].T)[:3, 3:]
        )
        # The estimator keeps covariance rank: no cross-assay permutation or
        # rotation is learned, and each ordered score identifies its own factor.
        assert np.all(np.diag(correlations) > 0.94)
        assert np.all(correlations - np.diag(np.diag(correlations)) < 0.25)
    boundary = result.provenance["information_boundary"]
    assert not boundary["cross_assay_rows_paired"]
    assert not boundary["cross_assay_feature_columns_compared"]
    assert not boundary["truth_or_scorer_used"]


def test_nearly_tied_covariance_rank_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    matrices = {
        modality: rng.poisson(10.0, size=(80, 18)).astype(np.float64)
        for modality in ("rna", "chrom_A")
    }
    depths = {modality: matrix.sum(axis=1) for modality, matrix in matrices.items()}
    stable_scores = rng.normal(size=(80, 3))

    def tied_fit(
        matrix: np.ndarray,
        depth: np.ndarray,
        indices: np.ndarray,
        n_components: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del matrix, depth, indices
        return stable_scores[:, :n_components].copy(), np.array([1.0, 0.99, 0.40])

    monkeypatch.setattr(ordered_alignment, "_fit_unwhitened_pca", tied_fit)
    result = ordered_alignment.align_ordered_variance_background_states(
        matrices,
        depths,
        config=ordered_alignment.OrderedVarianceAlignmentConfig(
            min_fold_component_correlation=0.60,
            min_relative_variance_gap=0.05,
        ),
    )

    assert not any(result.assay_eligible.values())
    assert result.provenance["abstained_assays"] == ["chrom_A", "rna"]
    assert all(
        value >= 2.0 for array in result.embedding_uncertainty.values() for value in array
    )


def test_freeze_rejects_scorer_path_before_read(tmp_path: Path) -> None:
    discovery = tmp_path / "scorer" / "world" / "discovery"
    discovery.mkdir(parents=True)

    with pytest.raises(ValueError, match="scorer/truth"):
        ordered_alignment.freeze_ordered_variance_evidence(
            discovery, tmp_path / "safe_output"
        )


def test_freeze_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    discovery = tmp_path / "world" / "discovery"
    discovery.mkdir(parents=True)
    output = tmp_path / "frozen"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ordered_alignment.freeze_ordered_variance_evidence(discovery, output)
