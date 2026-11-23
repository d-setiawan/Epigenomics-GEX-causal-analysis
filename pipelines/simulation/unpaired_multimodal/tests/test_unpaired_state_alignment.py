from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
from scipy import sparse


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from unpaired_state_alignment import (  # noqa: E402
    AlignmentConfig,
    align_cross_fitted_background_states,
    align_unpaired_background_states,
)


def _orthonormal_loadings(
    rng: np.random.Generator, n_features: int, n_state: int
) -> np.ndarray:
    raw = rng.normal(size=(n_features, n_state))
    loadings, _ = np.linalg.qr(raw)
    return loadings[:, :n_state]


def _unpaired_assay(
    *,
    seed: int,
    centers: np.ndarray,
    probabilities: np.ndarray,
    n_cells: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(centers), size=n_cells, p=probabilities)
    latent = centers[labels] + rng.normal(scale=0.10, size=(n_cells, centers.shape[1]))
    rotation, _ = np.linalg.qr(rng.normal(size=(centers.shape[1], centers.shape[1])))
    latent = latent @ rotation
    loadings = _orthonormal_loadings(rng, n_features, centers.shape[1])
    continuous = latent @ loadings.T + rng.normal(scale=0.025, size=(n_cells, n_features))
    # Positive observations with independently named/dimensioned columns.  Depth
    # is observed and varies by microstate, but row identities are never shared.
    depth = rng.lognormal(mean=7.4 + 0.06 * labels, sigma=0.18, size=n_cells)
    positive = continuous - continuous.min(axis=0, keepdims=True) + 0.05
    observed = positive * depth[:, None] / np.median(depth)
    return observed, depth, labels


def _truth_centroids(embedding: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    return np.vstack([embedding[labels == state].mean(axis=0) for state in range(n_states)])


def _relative_centroid_error(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    scale = np.linalg.norm(right) / max(np.linalg.norm(left), 1e-12)
    return float(np.linalg.norm(scale * left - right) / max(np.linalg.norm(right), 1e-12))


def test_synthetic_unpaired_rotations_align_without_feature_or_cell_correspondence() -> None:
    centers = np.asarray(
        [
            [-2.2, -0.4, 0.1],
            [-0.7, 1.8, -0.5],
            [0.1, -1.2, 1.7],
            [1.4, 0.3, -1.0],
            [2.7, 1.1, 0.8],
            [0.5, 2.9, 1.9],
        ]
    )
    probabilities = np.asarray([0.09, 0.14, 0.18, 0.21, 0.16, 0.22])
    rna_x, rna_depth, rna_labels = _unpaired_assay(
        seed=100, centers=centers, probabilities=probabilities, n_cells=1800, n_features=17
    )
    mark_x, mark_depth, mark_labels = _unpaired_assay(
        seed=200, centers=centers, probabilities=probabilities, n_cells=1500, n_features=29
    )
    third_x, third_depth, third_labels = _unpaired_assay(
        seed=300, centers=centers, probabilities=probabilities, n_cells=1650, n_features=23
    )
    result = align_unpaired_background_states(
        {
            "rna": rna_x,
            "chrom_H3K27ac": sparse.csr_matrix(mark_x),
            "chrom_H3K27me3": third_x,
        },
        {
            "rna": rna_depth,
            "chrom_H3K27ac": mark_depth,
            "chrom_H3K27me3": third_depth,
        },
        reference_modality="rna",
        config=AlignmentConfig(
            n_components=3,
            n_microstates=6,
            min_cells_per_microstate=30,
            random_seed=17,
        ),
    )

    reference = _truth_centroids(result.embeddings["rna"], rna_labels, len(centers))
    mark = _truth_centroids(
        result.embeddings["chrom_H3K27ac"], mark_labels, len(centers)
    )
    third = _truth_centroids(
        result.embeddings["chrom_H3K27me3"], third_labels, len(centers)
    )
    assert _relative_centroid_error(mark, reference) < 0.35
    assert _relative_centroid_error(third, reference) < 0.35
    assert result.embeddings["rna"].shape == (1800, 3)
    assert result.embeddings["chrom_H3K27ac"].shape == (1500, 3)

    boundary = result.provenance["information_boundary"]
    assert boundary["feature_columns_compared_across_assays"] is False
    assert boundary["cell_rows_compared_across_assays"] is False
    assert "true_latent_state" in boundary["not_consumed"]
    assert result.provenance["per_assay"]["chrom_H3K27ac"]["input_was_sparse"] is True


def test_derangement_changes_every_assignment_and_preserves_properties() -> None:
    centers = np.asarray(
        [
            [-2.0, 0.0],
            [-0.8, 1.7],
            [0.4, -1.3],
            [1.5, 0.5],
            [2.8, 1.9],
            [0.2, 3.0],
        ]
    )
    probabilities = np.asarray([0.10, 0.14, 0.18, 0.20, 0.16, 0.22])
    first_x, first_depth, _ = _unpaired_assay(
        seed=11, centers=centers, probabilities=probabilities, n_cells=900, n_features=11
    )
    second_x, second_depth, _ = _unpaired_assay(
        seed=12, centers=centers, probabilities=probabilities, n_cells=850, n_features=15
    )
    inputs = {"rna": first_x, "mark": second_x}
    depths = {"rna": first_depth, "mark": second_depth}
    base = align_unpaired_background_states(
        inputs,
        depths,
        reference_modality="rna",
        config=AlignmentConfig(n_components=2, n_microstates=6, random_seed=9),
    )
    deranged = align_unpaired_background_states(
        inputs,
        depths,
        reference_modality="rna",
        config=AlignmentConfig(
            n_components=2, n_microstates=6, random_seed=9, derange=True
        ),
    )
    assert np.all(
        base.microstate_assignments["mark"] != deranged.microstate_assignments["mark"]
    )
    metadata = deranged.provenance["alignments"]["mark"]
    assert metadata["deranged"] is True
    summary = metadata["derangement_property_summary"]
    assert 0.0 < summary["mean_prevalence_rank_change"] <= 1.0
    assert 0.0 < summary["mean_depth_rank_change"] <= 1.0


def test_public_api_has_no_truth_pairing_or_correspondence_arguments() -> None:
    parameters = set(inspect.signature(align_unpaired_background_states).parameters) | set(
        inspect.signature(align_cross_fitted_background_states).parameters
    )
    forbidden = {
        "labels",
        "true_labels",
        "true_z",
        "truth",
        "feature_names",
        "feature_links",
        "guidance_graph",
        "cell_ids",
        "pairs",
        "correspondences",
        "focal_features",
    }
    assert parameters.isdisjoint(forbidden)


def test_cross_fitted_alignment_uses_disjoint_within_assay_feature_folds() -> None:
    centers = np.asarray(
        [[-2.0, -0.2], [-0.7, 1.6], [0.5, -1.4], [1.8, 0.4], [2.8, 1.9]]
    )
    probabilities = np.asarray([0.12, 0.18, 0.24, 0.20, 0.26])
    first_x, first_depth, _ = _unpaired_assay(
        seed=51, centers=centers, probabilities=probabilities, n_cells=700, n_features=16
    )
    second_x, second_depth, _ = _unpaired_assay(
        seed=52, centers=centers, probabilities=probabilities, n_cells=650, n_features=22
    )

    result = align_cross_fitted_background_states(
        {"rna": first_x, "mark": second_x},
        {"rna": first_depth, "mark": second_depth},
        reference_modality="rna",
        n_feature_folds=2,
        config=AlignmentConfig(
            n_components=2,
            n_microstates=5,
            min_cells_per_microstate=30,
            random_seed=7,
        ),
    )

    assert result.embeddings["rna"].shape == (700, 2)
    assert result.embeddings["mark"].shape == (650, 2)
    assert result.embedding_uncertainty["mark"].shape == (650,)
    assert np.isfinite(result.embedding_uncertainty["mark"]).all()
    assert result.provenance["fold_feature_counts"]["rna"] == [8, 8]
    boundary = result.provenance["information_boundary"]
    assert boundary["feature_columns_compared_across_assays"] is False
    assert boundary["cell_rows_compared_across_assays"] is False
