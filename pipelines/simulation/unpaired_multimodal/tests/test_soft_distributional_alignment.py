from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import soft_distributional_alignment as soft_alignment  # noqa: E402


CENTERS = np.asarray(
    [
        [-1.55, -0.55, +0.15],
        [-0.25, +1.45, -0.35],
        [+1.35, +0.20, +0.75],
        [-0.95, +0.75, +1.25],
        [+0.45, -1.35, -0.85],
        [+1.45, +0.55, -0.65],
    ],
    dtype=np.float64,
)


def _private_assay(
    seed: int,
    *,
    n_cells: int,
    n_features: int,
    target_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    within = np.asarray([0.46, 0.34, 0.20])
    probabilities = np.concatenate(
        [target_fraction * within, (1.0 - target_fraction) * within]
    )
    modes = rng.choice(len(CENTERS), size=n_cells, p=probabilities)
    state = CENTERS[modes] + rng.normal(scale=0.44, size=(n_cells, 3))
    state[:, 1] += 0.22 * state[:, 0]
    loadings = rng.normal(size=(n_features, 3))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    technical = np.exp(rng.normal(scale=0.30, size=n_cells))
    baseline = rng.uniform(0.7, 3.0, size=n_features)
    rate = np.exp(
        np.log(baseline)[None, :]
        + 0.58 * state @ loadings.T
        + rng.normal(scale=0.08, size=(n_cells, n_features))
    )
    observed = rng.poisson(rate * technical[:, None]).astype(np.float64)
    depth = observed.sum(axis=1)
    zero = np.flatnonzero(depth == 0)
    if zero.size:
        observed[zero, rng.integers(0, n_features, size=len(zero))] = 1.0
        depth = observed.sum(axis=1)
    return observed, depth, state


def _loading(embedding: np.ndarray, response: np.ndarray) -> np.ndarray:
    standardized = (response - response.mean()) / response.std(ddof=0)
    return np.asarray(Ridge(alpha=2.0).fit(embedding, standardized).coef_, dtype=float)


def _absolute_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return abs(
        float(np.dot(left, right))
        / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    )


def _ambiguous_background_assay(
    seed: int,
    *,
    n_cells: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed count matrix with depth but no reproducible biological axes."""

    rng = np.random.default_rng(seed)
    technical = np.exp(rng.normal(scale=0.30, size=n_cells))
    baseline = rng.uniform(0.8, 3.0, size=n_features)
    observed = rng.poisson(technical[:, None] * baseline[None, :]).astype(float)
    return observed, observed.sum(axis=1)


def test_soft_distributional_alignment_recovers_shared_loading_axis_under_shift() -> None:
    specifications = {
        "rna": (101, 950, 48, 0.47),
        "chrom_A": (202, 760, 61, 0.60),
        "chrom_B": (303, 840, 55, 0.72),
    }
    matrices: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    for modality, (seed, cells, features, fraction) in specifications.items():
        matrix, depth, state = _private_assay(
            seed,
            n_cells=cells,
            n_features=features,
            target_fraction=fraction,
        )
        matrices[modality] = matrix
        depths[modality] = depth
        states[modality] = state

    result = soft_alignment.align_soft_distributional_background_states(
        matrices,
        depths,
        reference_modality="rna",
        config=soft_alignment.SoftAlignmentConfig(
            n_components=3,
            n_microstates=18,
            min_cells_per_microstate=20,
            n_feature_folds=2,
            random_seed=17,
            n_random_starts=8,
            max_refinement_iterations=30,
            sinkhorn_iterations=160,
        ),
    )

    biological_axis = np.asarray([1.10, -0.55, 0.35])
    loadings = {
        modality: _loading(
            result.embeddings[modality],
            states[modality] @ biological_axis,
        )
        for modality in specifications
    }
    assert _absolute_cosine(loadings["rna"], loadings["chrom_A"]) > 0.80
    assert _absolute_cosine(loadings["rna"], loadings["chrom_B"]) > 0.80
    assert _absolute_cosine(loadings["chrom_A"], loadings["chrom_B"]) > 0.80
    assert all(
        result.embeddings[modality].shape == (specifications[modality][1], 3)
        for modality in specifications
    )
    assert all(
        np.isfinite(result.embedding_uncertainty[modality]).all()
        for modality in specifications
    )


def test_soft_alignment_api_and_provenance_exclude_truth_and_correspondence() -> None:
    parameters = set(
        inspect.signature(
            soft_alignment.align_soft_distributional_background_states
        ).parameters
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

    first, first_depth, _ = _private_assay(
        71, n_cells=260, n_features=24, target_fraction=0.55
    )
    second, second_depth, _ = _private_assay(
        72, n_cells=240, n_features=28, target_fraction=0.65
    )
    result = soft_alignment.align_soft_distributional_background_states(
        {"rna": first, "mark": second},
        {"rna": first_depth, "mark": second_depth},
        reference_modality="rna",
        config=soft_alignment.SoftAlignmentConfig(
            n_microstates=8,
            min_cells_per_microstate=20,
            n_random_starts=0,
            max_refinement_iterations=12,
            sinkhorn_iterations=80,
            random_seed=5,
        ),
    )
    boundary = result.provenance["information_boundary"]
    assert boundary["feature_columns_compared_across_assays"] is False
    assert boundary["cell_rows_compared_across_assays"] is False
    assert boundary["transport_plan_returned_as_pairing"] is False
    assert boundary["confidence_gates_use_background_distributions_only"] is True
    assert boundary["synchronization_uses_pairwise_rotations_only"] is True
    assert "true_latent_state" in boundary["not_consumed"]
    diagnostics = result.provenance["distribution_diagnostics"]
    assert "mark" in diagnostics
    assert diagnostics["mark"]["n_starts"] > 0
    assert len(result.provenance["fold_reconciliation"]["mark"]) == 2


def test_invalid_soft_alignment_configuration_is_rejected() -> None:
    for kwargs in (
        {"population_uniform_shrinkage": -0.1},
        {"population_uniform_shrinkage": 1.1},
        {"uncertainty_floor": 0.0},
        {"entropy_fraction": 0.0},
        {"n_random_starts": -1},
        {"max_pair_rotation_dispersion": 0.0},
        {"min_pair_transport_separation": 1.0},
        {"min_synchronization_neighbors": 0},
        {"min_cycle_support": 0},
        {"max_cycle_inconsistency": 0.0},
        {"synchronization_iterations": 0},
        {"abstention_uncertainty_penalty": 0.0},
    ):
        try:
            soft_alignment.SoftAlignmentConfig(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive assertion style
            raise AssertionError(f"Expected invalid configuration to fail: {kwargs}")


def test_ambiguous_assay_abstains_without_contaminating_rotation_consensus() -> None:
    specifications = {
        "rna": (101, 360, 32, 0.47),
        "chrom_A": (202, 340, 34, 0.60),
        "chrom_B": (303, 350, 36, 0.72),
        "chrom_C": (404, 330, 38, 0.57),
    }
    good_matrices: dict[str, np.ndarray] = {}
    good_depths: dict[str, np.ndarray] = {}
    for modality, (seed, cells, features, fraction) in specifications.items():
        matrix, depth, _ = _private_assay(
            seed,
            n_cells=cells,
            n_features=features,
            target_fraction=fraction,
        )
        good_matrices[modality] = matrix
        good_depths[modality] = depth
    bad_matrix, bad_depth = _ambiguous_background_assay(
        991, n_cells=360, n_features=32
    )
    all_matrices = {**good_matrices, "zz_ambiguous": bad_matrix}
    all_depths = {**good_depths, "zz_ambiguous": bad_depth}
    config = soft_alignment.SoftAlignmentConfig(
        n_microstates=7,
        min_cells_per_microstate=20,
        n_random_starts=0,
        max_refinement_iterations=12,
        sinkhorn_iterations=60,
        random_seed=17,
    )

    with_bad = soft_alignment.align_soft_distributional_background_states(
        all_matrices,
        all_depths,
        reference_modality="rna",
        config=config,
    )
    good_only = soft_alignment.align_soft_distributional_background_states(
        good_matrices,
        good_depths,
        reference_modality="rna",
        config=config,
    )

    assert with_bad.assay_synchronized["zz_ambiguous"] is False
    assert with_bad.assay_alignment_confidence["zz_ambiguous"] == 0.0
    assert set(with_bad.provenance["synchronization"]["abstained_assays"]) == {
        "zz_ambiguous"
    }
    assert all(
        with_bad.assay_synchronized[modality] for modality in specifications
    )
    # Pair seeds, quantization seeds, confidence gates, and synchronization are
    # invariant to an unrelated rejected assay, so its addition changes none of
    # the accepted coordinates—not merely their average quality.
    for modality in specifications:
        np.testing.assert_allclose(
            with_bad.embeddings[modality],
            good_only.embeddings[modality],
            rtol=0.0,
            atol=0.0,
        )
    assert float(
        np.min(with_bad.embedding_uncertainty["zz_ambiguous"])
    ) >= config.abstention_uncertainty_penalty


def test_freeze_runner_reads_only_background_archives_and_hash_locks_outputs(
    tmp_path: Path,
) -> None:
    discovery = tmp_path / "world" / "discovery"
    discovery.mkdir(parents=True)
    assays = {
        "rna": _private_assay(
            801, n_cells=180, n_features=20, target_fraction=0.55
        ),
        "chrom_A": _private_assay(
            802, n_cells=170, n_features=24, target_fraction=0.63
        ),
    }
    for modality, (matrix, depth, _) in assays.items():
        cells = np.asarray(
            [f"{modality}_{index:04d}" for index in range(len(matrix))]
        )
        np.savez_compressed(
            discovery / f"background_matrix__{modality}.npz",
            matrix=matrix,
            cell_ids=cells,
            background_depth=depth,
        )
    # These tempting artifacts must not be opened by the freeze runner.
    (discovery / "observed_focal__rna.tsv").write_text("forbidden\n")
    (discovery / "analysis_embeddings.tsv").write_text("forbidden\n")
    output = tmp_path / "frozen"
    manifest = soft_alignment.freeze_soft_alignment_evidence(
        discovery,
        output,
        config=soft_alignment.SoftAlignmentConfig(
            n_microstates=6,
            min_cells_per_microstate=20,
            n_random_starts=0,
            max_refinement_iterations=10,
            sinkhorn_iterations=60,
            random_seed=4,
        ),
    )
    on_disk = json.loads((output / "frozen_evidence_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["frozen_before_scorer_read"] is True
    boundary = manifest["information_boundary"]
    assert boundary["focal_files_read"] is False
    assert boundary["supplied_embeddings_read"] is False
    assert boundary["scorer_or_truth_read"] is False
    assert boundary["files_read"] == [
        "background_matrix__chrom_A.npz",
        "background_matrix__rna.npz",
    ]
    assert set(manifest["output_file_sha256"]) == {
        "alignment_provenance.json",
        "analysis_embeddings.tsv",
    }
    frozen_columns = set(
        pd.read_csv(output / "analysis_embeddings.tsv", sep="\t", nrows=1).columns
    )
    assert frozen_columns == {
        "modality",
        "cell",
        "integration__dim0",
        "integration__dim1",
        "integration__dim2",
        "integration_uncertainty",
    }
