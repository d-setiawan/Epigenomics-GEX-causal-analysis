from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cell_state_loading_groups as loading_groups  # noqa: E402


RNA_FEATURE = "expr__aligned"
A_ALIGNED = "chr1:100-199__A"
A_TECHNICAL = "chr1:300-399__A"
B_ALIGNED = "chr2:100-199__B"
B_ORTHOGONAL = "chr2:300-399__B"
ALIGNED_FEATURES = {RNA_FEATURE, A_ALIGNED, B_ALIGNED}


@dataclass(frozen=True)
class SyntheticPaths:
    discovery: Path
    embedding: Path
    common_loading: np.ndarray
    rotations: dict[str, np.ndarray]


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _write_synthetic_inputs(
    root: Path,
    *,
    aligned: bool,
    n_cells: int = 480,
) -> SyntheticPaths:
    discovery = root / "world" / "discovery"
    safe = root / "safe_recomputed"
    discovery.mkdir(parents=True)
    safe.mkdir()
    rotations = {
        "rna": _rotation(0.0),
        "chrom_A": _rotation(np.pi / 2.0),
        "chrom_B": _rotation(-np.pi / 3.0),
    }
    common_loading = np.asarray([1.15, 0.55, 0.0])
    embedding_rows: list[pd.DataFrame] = []
    for assay_index, modality in enumerate(rotations):
        rng = np.random.default_rng(1_100 + assay_index)
        common_state = rng.normal(size=(n_cells, 3))
        local_state = common_state @ rotations[modality].T
        safe_state = local_state @ rotations[modality] if aligned else local_state
        technical = rng.normal(size=n_cells)
        log_depth = (
            5.2
            + 0.85 * common_state[:, 0]
            - 0.35 * common_state[:, 2]
            + 0.70 * technical
        )
        depth = np.exp(log_depth)
        prefix = modality.replace("chrom_", "").lower()
        cells = np.asarray([f"{prefix}_{index:05d}" for index in range(n_cells)])
        embedding_rows.append(
            pd.DataFrame(
                {
                    "modality": modality,
                    "cell": cells,
                    "integration__dim0": safe_state[:, 0],
                    "integration__dim1": safe_state[:, 1],
                    "integration__dim2": safe_state[:, 2],
                    "integration_uncertainty": np.full(n_cells, 0.04),
                }
            ).sample(frac=1.0, random_state=90 + assay_index)
        )
        shared_score = common_state @ common_loading
        if modality == "rna":
            response = shared_score + 0.70 * technical + rng.normal(scale=0.35, size=n_cells)
            focal = pd.DataFrame(
                {
                    "cell": cells,
                    "modality": modality,
                    "observed_background_library_size": depth,
                    "observed_library_size": depth + response - response.min() + 1.0,
                    RNA_FEATURE: response,
                }
            )
        elif modality == "chrom_A":
            aligned_response = rng.binomial(
                1,
                expit(-0.75 + 1.25 * shared_score + 0.75 * technical),
            )
            technical_response = rng.binomial(1, expit(-0.75 + 1.45 * technical))
            focal = pd.DataFrame(
                {
                    "cell": cells,
                    "modality": modality,
                    "observed_background_library_size": depth,
                    "observed_library_size": depth + aligned_response + technical_response,
                    A_ALIGNED: aligned_response,
                    A_TECHNICAL: technical_response,
                }
            )
        else:
            aligned_response = rng.binomial(
                1,
                expit(-0.75 + 1.25 * shared_score + 0.75 * technical),
            )
            orthogonal_response = rng.binomial(
                1,
                expit(-0.75 + 1.55 * common_state[:, 2] + 0.25 * technical),
            )
            focal = pd.DataFrame(
                {
                    "cell": cells,
                    "modality": modality,
                    "observed_background_library_size": depth,
                    "observed_library_size": depth + aligned_response + orthogonal_response,
                    B_ALIGNED: aligned_response,
                    B_ORTHOGONAL: orthogonal_response,
                }
            )
        focal.sample(frac=1.0, random_state=40 + assay_index).to_csv(
            discovery / f"observed_focal__{modality}.tsv",
            sep="\t",
            index=False,
        )
    pd.concat(embedding_rows, ignore_index=True).to_csv(
        safe / "analysis_embeddings.tsv", sep="\t", index=False
    )
    return SyntheticPaths(
        discovery=discovery,
        embedding=safe / "analysis_embeddings.tsv",
        common_loading=common_loading,
        rotations=rotations,
    )


@pytest.fixture(scope="module")
def aligned_paths(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPaths:
    return _write_synthetic_inputs(
        tmp_path_factory.mktemp("aligned-loading"),
        aligned=True,
    )


@pytest.fixture(scope="module")
def aligned_result(
    aligned_paths: SyntheticPaths,
) -> loading_groups.LoadingGroupResult:
    inputs = loading_groups.load_loading_inputs(
        aligned_paths.discovery,
        aligned_paths.embedding,
    )
    return loading_groups.estimate_loading_groups(
        inputs,
        loading_groups.LoadingGroupConfig(
            bootstrap_repetitions=24,
            min_state_fit=0.02,
            min_direction_stability=0.55,
            min_abs_loading_cosine=0.82,
            max_unit_loading_distance=0.62,
            max_neighbors_per_feature=2,
            seed=7,
        ),
    )


def _loading_vector(
    result: loading_groups.LoadingGroupResult,
    feature: str,
) -> np.ndarray:
    subset = result.loadings.loc[result.loadings["feature"] == feature].sort_values(
        "state_coordinate"
    )
    return subset["conditional_coefficient"].to_numpy(dtype=float)


def _pair_row(
    result: loading_groups.LoadingGroupResult,
    left: str,
    right: str,
) -> pd.Series:
    node1, node2 = sorted((left, right))
    return result.pair_evidence.loc[
        (result.pair_evidence["node1"] == node1)
        & (result.pair_evidence["node2"] == node2)
    ].iloc[0]


def test_rotated_assays_recover_aligned_common_loadings_and_sparse_group(
    aligned_paths: SyntheticPaths,
    aligned_result: loading_groups.LoadingGroupResult,
) -> None:
    # The assay-local systems are genuinely rotated, while the safe embedding
    # has mapped each one back to the shared coordinates consumed by the layer.
    assert not np.allclose(aligned_paths.rotations["chrom_A"], np.eye(3))
    for feature in ALIGNED_FEATURES:
        estimated = _loading_vector(aligned_result, feature)
        cosine = float(
            np.dot(estimated, aligned_paths.common_loading)
            / (np.linalg.norm(estimated) * np.linalg.norm(aligned_paths.common_loading))
        )
        assert cosine > 0.88

    expected_pairs = {
        tuple(sorted(pair))
        for pair in (
            (RNA_FEATURE, A_ALIGNED),
            (RNA_FEATURE, B_ALIGNED),
            (A_ALIGNED, B_ALIGNED),
        )
    }
    selected = {
        (str(row.node1), str(row.node2))
        for row in aligned_result.sparse_pairs.itertuples(index=False)
    }
    assert expected_pairs <= selected
    memberships = aligned_result.group_membership
    aligned_groups = memberships.loc[memberships["feature"].isin(ALIGNED_FEATURES)]
    assert len(aligned_groups) == 3
    assert aligned_groups["loading_group"].nunique() == 1
    assert B_ORTHOGONAL not in set(aligned_groups["feature"])


def test_depth_is_state_orthogonal_and_feature_fit_reports_uncertainty(
    aligned_result: loading_groups.LoadingGroupResult,
) -> None:
    models = aligned_result.feature_models.set_index("feature")
    technical = models.loc[A_TECHNICAL]
    assert technical["oof_depth_only_fit"] > 0.08
    assert technical["oof_partial_state_fit"] < 0.03
    assert technical["conditional_loading_norm"] < models.loc[A_ALIGNED, "conditional_loading_norm"]

    required_model_columns = {
        "oof_marginal_state_fit",
        "oof_partial_state_fit",
        "oof_depth_only_fit",
        "oof_joint_fit",
        "conditional_loading_norm",
        "relative_loading_uncertainty",
        "loading_direction_stability",
        "bootstrap_direction_stability",
    }
    assert required_model_columns <= set(aligned_result.feature_models)
    assert np.isfinite(
        aligned_result.loadings[
            [
                "conditional_coefficient",
                "marginal_coefficient",
                "fold_coefficient_sd",
                "bootstrap_coefficient_sd",
                "bootstrap_ci_2.5",
                "bootstrap_ci_97.5",
                "coefficient_sign_stability",
            ]
        ].to_numpy(dtype=float)
    ).all()


def test_pair_evidence_is_complete_canonical_and_rethresholdable(
    aligned_result: loading_groups.LoadingGroupResult,
) -> None:
    n_features = len(aligned_result.feature_models)
    evidence = aligned_result.pair_evidence
    assert len(evidence) == n_features * (n_features - 1) // 2
    assert (evidence["node1"] < evidence["node2"]).all()
    assert evidence["pair_key"].is_unique
    required = {
        "pair_key",
        "node1",
        "node2",
        "channel1",
        "channel2",
        "same_assay",
        "loading_cosine",
        "absolute_loading_cosine",
        "sign_invariant_unit_loading_distance",
        "node1_oof_partial_state_fit",
        "node2_oof_partial_state_fit",
        "pair_state_score",
        "node1_loading_stability",
        "node2_loading_stability",
        "loading_stability_support",
        "loading_empirical_null_p",
        "loading_empirical_null_q",
        "loading_null_absolute_cosine_95",
        "unit_distance_null_percentile",
        "threshold_eligible",
        "selected_sparse",
    }
    assert required <= set(evidence)
    assert evidence["loading_empirical_null_p"].between(0, 1).all()
    assert evidence["loading_empirical_null_q"].between(0, 1).all()
    assert evidence["unit_distance_null_percentile"].between(0, 1).all()
    assert np.isfinite(evidence["pair_state_score"]).all()
    assert len(aligned_result.sparse_pairs) < len(evidence)

    aligned_pair = _pair_row(aligned_result, A_ALIGNED, B_ALIGNED)
    orthogonal_pair = _pair_row(aligned_result, A_ALIGNED, B_ORTHOGONAL)
    assert aligned_pair["absolute_loading_cosine"] > 0.90
    assert aligned_pair["pair_state_score"] > orthogonal_pair["pair_state_score"]


def test_within_assay_state_response_similarity_is_coordinate_invariant() -> None:
    rng = np.random.default_rng(91)
    state = rng.normal(size=(600, 3)) @ np.asarray(
        [[1.0, 0.7, 0.1], [0.0, 1.5, 0.4], [0.0, 0.0, 0.6]]
    )
    transform = np.asarray(
        [[1.8, 0.4, -0.2], [0.1, 0.7, 0.3], [0.2, -0.1, 1.3]]
    )
    coefficients = {
        "a": np.asarray([0.9, -0.35, 0.2]),
        "b": np.asarray([0.7, -0.15, 0.4]),
    }

    def evidence_for(
        coordinates: np.ndarray,
        vectors: dict[str, np.ndarray],
    ) -> pd.Series:
        metric = loading_groups._state_metric_transform(coordinates)
        response_vectors = {
            feature: metric @ vector for feature, vector in vectors.items()
        }
        models = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "modality": "chrom_A",
                    "channel": "A",
                    "conditional_loading_norm": float(
                        np.linalg.norm(response_vectors[feature])
                    ),
                    "oof_marginal_state_fit": 0.3,
                    "oof_partial_state_fit": 0.25,
                    "loading_direction_stability": 1.0,
                }
                for feature in sorted(vectors)
            ]
        )
        pair, _ = loading_groups._pair_evidence(
            models,
            vectors,
            response_vectors,
            loading_groups.LoadingGroupConfig(
                bootstrap_repetitions=0,
                min_loading_norm=0.0,
                min_state_fit=0.0,
                min_direction_stability=0.0,
                min_abs_loading_cosine=0.0,
                max_unit_loading_distance=2.0,
            ),
        )
        return pair.iloc[0]

    original = evidence_for(state, coefficients)
    transformed_state = state @ transform
    transformed_coefficients = {
        feature: np.linalg.solve(transform, vector)
        for feature, vector in coefficients.items()
    }
    reparameterized = evidence_for(transformed_state, transformed_coefficients)
    assert original["absolute_loading_cosine"] == pytest.approx(
        reparameterized["absolute_loading_cosine"], abs=1e-12
    )
    assert original["pair_state_score"] == pytest.approx(
        reparameterized["pair_state_score"], abs=1e-12
    )
    assert original["absolute_raw_coefficient_loading_cosine"] != pytest.approx(
        reparameterized["absolute_raw_coefficient_loading_cosine"], abs=1e-3
    )


def test_unaligned_rotated_coordinates_do_not_masquerade_as_shared_loading(
    tmp_path: Path,
) -> None:
    paths = _write_synthetic_inputs(tmp_path, aligned=False, n_cells=380)
    inputs = loading_groups.load_loading_inputs(paths.discovery, paths.embedding)
    result = loading_groups.estimate_loading_groups(
        inputs,
        loading_groups.LoadingGroupConfig(
            bootstrap_repetitions=6,
            min_state_fit=0.0,
            min_direction_stability=0.0,
            seed=9,
        ),
    )
    pair = _pair_row(result, RNA_FEATURE, A_ALIGNED)
    assert pair["absolute_loading_cosine"] < 0.35
    assert not bool(pair["selected_sparse"])


def test_strict_paths_and_leakage_columns_are_rejected(
    aligned_paths: SyntheticPaths,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="simulator-supplied"):
        loading_groups.load_loading_inputs(
            aligned_paths.discovery,
            aligned_paths.discovery / "analysis_embeddings.tsv",
        )

    truth_root = tmp_path / "truth"
    truth_discovery = truth_root / "discovery"
    truth_root.mkdir()
    shutil.copytree(aligned_paths.discovery, truth_discovery)
    with pytest.raises(ValueError, match="scorer/truth"):
        loading_groups.load_loading_inputs(truth_discovery, aligned_paths.embedding)

    unsafe = tmp_path / "unsafe" / "analysis_embeddings.tsv"
    unsafe.parent.mkdir()
    embeddings = pd.read_csv(aligned_paths.embedding, sep="\t")
    embeddings["true_z0"] = 99.0
    embeddings.to_csv(unsafe, sep="\t", index=False)
    with pytest.raises(ValueError, match="forbidden columns"):
        loading_groups.load_loading_inputs(aligned_paths.discovery, unsafe)


def test_api_writes_safe_standalone_tables(
    aligned_paths: SyntheticPaths,
    tmp_path: Path,
) -> None:
    output = tmp_path / "loading-output"
    summary = loading_groups.run_loading_group_analysis(
        aligned_paths.discovery,
        aligned_paths.embedding,
        output,
        loading_groups.LoadingGroupConfig(
            bootstrap_repetitions=6,
            min_state_fit=0.02,
            min_abs_loading_cosine=0.82,
            seed=7,
        ),
    )
    expected = {
        "feature_state_models.tsv",
        "conditional_state_loadings.tsv",
        "loading_pair_evidence.tsv",
        "sparse_loading_pairs.tsv",
        "loading_groups.tsv",
        "loading_group_membership.tsv",
        "analysis_provenance.json",
        "run_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert summary["n_features"] == 5
    assert summary["n_pair_candidates"] == 10
    assert summary["row_pairing_used"] is False
    assert summary["truth_or_scorer_used"] is False
    provenance = json.loads((output / "analysis_provenance.json").read_text())
    assert provenance["identifiability"]["cross_assay_rows_paired"] is False
    assert provenance["input_policy"]["scorer_or_truth_read"] is False
    assert provenance["input_policy"]["membership_or_metacells_read"] is False
    assert set(provenance["assay_cell_counts"].values()) == {480}
