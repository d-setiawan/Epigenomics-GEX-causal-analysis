from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import component_identified_alignment_v4 as alignment  # noqa: E402


def _signed_matrix(order: tuple[int, int, int], signs: tuple[int, int, int]) -> np.ndarray:
    identity = np.eye(3)
    return identity[:, list(order)] * np.asarray(signs, dtype=float)[None, :]


def _background(seed: int, *, n_cells: int = 240, n_features: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    modes = rng.choice(4, size=n_cells, p=[0.35, 0.30, 0.22, 0.13])
    centers = np.asarray(
        [
            [-2.0, -0.4, 0.2],
            [1.6, -1.1, 0.0],
            [0.5, 1.8, -0.9],
            [0.2, 0.4, 2.2],
        ]
    )
    state = centers[modes] + rng.normal(scale=[0.35, 0.45, 0.30], size=(n_cells, 3))
    loadings = rng.normal(size=(n_features, 3))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    baseline = rng.uniform(1.0, 3.0, size=n_features)
    technical = np.exp(rng.normal(scale=0.2, size=n_cells))
    rate = np.exp(np.log(baseline)[None, :] + 0.55 * state @ loadings.T)
    matrix = rng.poisson(rate * technical[:, None]).astype(float)
    zero = np.flatnonzero(matrix.sum(axis=1) == 0)
    matrix[zero, 0] = 1.0
    return matrix, matrix.sum(axis=1)


def test_exact_signed_permutation_transport_recovers_joint_geometry() -> None:
    rng = np.random.default_rng(41)
    target_centroids = rng.normal(size=(6, 3)) * np.asarray([1.8, 1.0, 0.5])
    expected = _signed_matrix((2, 0, 1), (-1, 1, -1))
    target = alignment._QuantizedDistribution(
        centroids=target_centroids,
        mass=np.full(6, 1.0 / 6.0),
        effective_cells=200.0,
    )
    reference = alignment._QuantizedDistribution(
        centroids=target_centroids @ expected,
        mass=np.full(6, 1.0 / 6.0),
        effective_cells=180.0,
    )
    config = alignment.ComponentAlignmentConfig(
        n_transport_microstates=6,
        sinkhorn_iterations=400,
    )
    observed, diagnostics = alignment._fit_pair_transform(
        target,
        reference,
        transforms=alignment._signed_permutation_matrices(3),
        config=config,
    )

    assert np.array_equal(observed, expected)
    assert diagnostics["signed_permutation_candidates_evaluated"] == 48
    assert diagnostics["relative_signed_permutation_margin"] > 0.0


def test_discrete_global_synchronization_recovers_cycle_consistent_transforms() -> None:
    transforms = alignment._signed_permutation_matrices(3)
    global_transform = {
        "rna": np.eye(3),
        "a": _signed_matrix((1, 2, 0), (1, -1, 1)),
        "b": _signed_matrix((2, 0, 1), (-1, 1, 1)),
        "c": _signed_matrix((0, 2, 1), (1, 1, -1)),
    }
    rotations: dict[tuple[str, str], np.ndarray] = {}
    edges: set[tuple[str, str]] = set()
    confidences: dict[tuple[str, str], float] = {}
    modalities = sorted(global_transform)
    for left_index, left in enumerate(modalities):
        for right in modalities[left_index + 1 :]:
            edge = (left, right)
            edges.add(edge)
            confidences[edge] = 1.0
            relative = global_transform[left] @ global_transform[right].T
            rotations[(left, right)] = relative
            rotations[(right, left)] = relative.T

    synchronized, margins, diagnostics = alignment._synchronize_discrete_transforms(
        "rna",
        rotations,
        confidences,
        set(modalities),
        edges,
        transforms,
        alignment.ComponentAlignmentConfig(),
    )

    for modality in modalities:
        assert np.array_equal(synchronized[modality], global_transform[modality])
        assert margins[modality] > 0.0
    assert diagnostics["weighted_mean_edge_residual"] == 0.0
    assert diagnostics["exact_consensus_edges"] == len(edges)


def test_public_api_excludes_truth_focal_and_correspondence_inputs() -> None:
    parameters = set(
        inspect.signature(
            alignment.align_component_identified_background_states
        ).parameters
    )
    assert parameters == {
        "background_matrices",
        "background_depths",
        "reference_modality",
        "config",
    }
    assert not parameters.intersection(
        {
            "truth",
            "roles",
            "focal",
            "latent_state",
            "cell_ids",
            "feature_ids",
            "pairing",
        }
    )


def test_pca_covariance_gap_gate_abstains_without_relaxing_the_transform_family() -> None:
    matrices: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    for modality, seed in (("rna", 12), ("chrom_A", 13), ("chrom_B", 14)):
        matrices[modality], depths[modality] = _background(seed)
    result = alignment.align_component_identified_background_states(
        matrices,
        depths,
        config=alignment.ComponentAlignmentConfig(
            method="pca",
            n_transport_microstates=6,
            min_pca_relative_variance_gap=1.0,
        ),
    )

    assert not any(result.assay_eligible.values())
    assert result.provenance["identifiability"]["cross_assay_rotation_fitted"] is False
    assert (
        result.provenance["identifiability"]["cross_assay_transform_family"]
        == "signed_permutation_only"
    )


def test_freeze_runner_reads_background_archives_only_and_hash_locks_outputs(
    tmp_path: Path,
) -> None:
    discovery = tmp_path / "world" / "discovery"
    discovery.mkdir(parents=True)
    for modality, seed in (("rna", 21), ("chrom_A", 22)):
        matrix, depth = _background(seed, n_cells=180, n_features=18)
        np.savez_compressed(
            discovery / f"background_matrix__{modality}.npz",
            matrix=matrix,
            cell_ids=np.asarray(
                [f"{modality}_{index:04d}" for index in range(len(matrix))]
            ),
            background_depth=depth,
        )
    output = tmp_path / "frozen_component_alignment"
    manifest = alignment.freeze_component_alignment_evidence(
        discovery,
        output,
        config=alignment.ComponentAlignmentConfig(
            method="pca",
            n_transport_microstates=6,
            min_pca_relative_variance_gap=1.0,
        ),
    )

    expected = {
        "analysis_embeddings.tsv",
        "component_evidence.tsv",
        "alignment_provenance.json",
        "frozen_evidence_manifest.json",
        "frozen_evidence_manifest.sha256",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert manifest["frozen_before_scorer_read"] is True
    assert manifest["information_boundary"]["truth_or_scorer_used"] is False
    for filename, expected_hash in manifest["output_file_sha256"].items():
        observed = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert observed == expected_hash
    frozen = json.loads((output / "frozen_evidence_manifest.json").read_text())
    assert frozen["input_file_sha256"]
    assert all(name.startswith("background_matrix__") for name in frozen["input_file_sha256"])


def test_axis_freeze_rejects_scorer_paths(tmp_path: Path) -> None:
    scorer_root = tmp_path / "scorer" / "axis"
    scorer_root.mkdir(parents=True)
    (scorer_root / "evidence.tsv").write_text("x\n1\n")
    source = tmp_path / "prototype.py"
    source.write_text("pass\n")

    try:
        alignment.freeze_observed_axis_directory(
            scorer_root, source_paths=[source]
        )
    except ValueError as exc:
        assert "scorer/truth" in str(exc)
    else:  # pragma: no cover - a firewall failure must be loud
        raise AssertionError("scorer path was accepted")
