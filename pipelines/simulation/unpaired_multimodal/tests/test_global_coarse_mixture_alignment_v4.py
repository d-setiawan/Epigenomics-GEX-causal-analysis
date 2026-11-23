from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import global_coarse_mixture_alignment_v4 as mixture_alignment  # noqa: E402
import filter_axis_groups_by_alignment_v4 as eligibility_filter  # noqa: E402


def _uniform_whiten(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        centered.T @ centered / len(centered)
    )
    return centered @ (
        eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    )


def _prototype_config() -> mixture_alignment.GlobalCoarseMixtureConfig:
    return mixture_alignment.GlobalCoarseMixtureConfig(
        random_rotation_starts=8,
        shape_alignment_iterations=20,
        template_iterations=8,
        empirical_null_repetitions=79,
    )


def test_global_unlabeled_registration_recovers_shared_shape_and_abstains_null() -> None:
    rng = np.random.default_rng(8)
    shared = _uniform_whiten(
        np.array(
            [
                [-2.0, -0.5, 0.1],
                [-0.7, 1.8, 0.4],
                [0.3, -1.4, 1.5],
                [1.7, 0.2, -0.8],
                [0.8, 1.2, 1.6],
                [-0.2, -0.8, -1.5],
            ]
        )
    )
    modalities = ["rna", *(f"chrom_{index}" for index in range(6))]
    centroids: dict[str, np.ndarray] = {}
    folds: dict[str, tuple[np.ndarray, ...]] = {}
    for modality in modalities:
        # One assay deliberately has unrelated geometry.  Every assay otherwise
        # receives an arbitrary continuous rotation and unlabeled mode order.
        shape = (
            _uniform_whiten(rng.normal(size=(6, 3)))
            if modality == "chrom_5"
            else shared
        )
        rotation = mixture_alignment._random_orthogonal(3, rng)
        values = shape[rng.permutation(6)] @ rotation
        centroids[modality] = values
        folds[modality] = (values.copy(), values.copy())

    rotations, eligible, _, diagnostics = (
        mixture_alignment.register_global_coarse_mixtures(
            centroids,
            folds,
            reference_modality="rna",
            config=_prototype_config(),
        )
    )

    assert all(eligible[modality] for modality in modalities if modality != "chrom_5")
    assert not eligible["chrom_5"]
    assert "shape_not_separated_from_scrambled_null" in diagnostics[
        "assay_registration_diagnostics"
    ]["chrom_5"]["abstention_reasons"]
    np.testing.assert_allclose(rotations["rna"], np.eye(3), atol=1e-12)
    reference = centroids["rna"]
    for modality in modalities:
        if modality == "chrom_5":
            continue
        mapped = centroids[modality] @ rotations[modality]
        comparison = mixture_alignment._align_shape(
            mapped,
            reference,
            seed=1,
            config=_prototype_config(),
        )
        assert comparison.objective < 1e-12


def test_uniform_mode_whitening_ignores_population_mass_in_geometry() -> None:
    rng = np.random.default_rng(12)
    means = np.array(
        [
            [-2.0, -0.5, 0.1],
            [-0.7, 1.8, 0.4],
            [0.3, -1.4, 1.5],
            [1.7, 0.2, -0.8],
            [0.8, 1.2, 1.6],
            [-0.2, -0.8, -1.5],
        ]
    )
    counts = np.array([500, 220, 120, 80, 50, 30])
    coordinates = np.vstack(
        [
            mean + rng.normal(scale=0.02, size=(count, 3))
            for mean, count in zip(means, counts, strict=True)
        ]
    )

    _, centroids, mass, diagnostics = mixture_alignment._uniform_centroid_whiten(
        coordinates,
        n_modes=6,
        seed=4,
        kmeans_n_init=20,
    )

    np.testing.assert_allclose(centroids.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(centroids.T @ centroids / 6, np.eye(3), atol=1e-9)
    assert np.max(mass) > 0.45
    assert diagnostics["mode_mass"] == pytest.approx(mass.tolist())


def test_safe_freeze_rejects_scorer_path_before_read(tmp_path: Path) -> None:
    discovery = tmp_path / "scorer" / "world" / "discovery"
    discovery.mkdir(parents=True)

    with pytest.raises(ValueError, match="scorer/truth"):
        mixture_alignment.freeze_global_coarse_mixture_evidence(
            discovery, tmp_path / "frozen"
        )


def test_axis_filter_keeps_only_observably_eligible_modalities(tmp_path: Path) -> None:
    projection = tmp_path / "projection"
    projection.mkdir()
    pd.DataFrame(
        {
            "loading_group": ["group_a", "group_b", "group_r"],
            "modality": ["chrom_A", "chrom_B", "rna"],
        }
    ).to_csv(projection / "loading_groups.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "loading_group": ["group_a", "group_b", "group_r"],
            "modality": ["chrom_A", "chrom_B", "rna"],
            "feature": ["a", "b", "r"],
        }
    ).to_csv(projection / "loading_group_membership.tsv", sep="\t", index=False)
    evidence = tmp_path / "alignment_provenance.json"
    evidence.write_text(
        """{
          "accepted_assays": ["chrom_A", "rna"],
          "abstained_assays": ["chrom_B"],
          "information_boundary": {"truth_or_scorer_used": false}
        }\n"""
    )

    manifest = eligibility_filter.filter_axis_groups(
        projection, evidence, tmp_path / "filtered"
    )

    selected = pd.read_csv(tmp_path / "filtered" / "loading_groups.tsv", sep="\t")
    assert set(selected["modality"]) == {"chrom_A", "rna"}
    assert manifest["n_output_groups"] == 2
    assert manifest["truth_or_scorer_used"] is False
