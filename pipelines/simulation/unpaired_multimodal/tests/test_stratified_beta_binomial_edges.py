"""Tests for the observed-only stratified beta-binomial rescue prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from stratified_beta_binomial_edges import (  # noqa: E402
    StratifiedBetaBinomialConfig,
    _local_beta_binomial_tables,
    assign_state_depth_strata,
    estimate_from_observed_tables,
    estimate_stratified_beta_binomial_edges,
    frame_sha256,
)


def _synthetic_sparse_world(
    n_cells: int = 2_400,
    seed: int = 812,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n_cells, 3))
    log_depth = 3.0 + 0.45 * state[:, 0] - 0.25 * state[:, 1] + rng.normal(
        scale=0.30, size=n_cells
    )
    depth = rng.poisson(np.exp(log_depth)).astype(float)
    centered_depth = (np.log1p(depth) - np.log1p(depth).mean()) / np.log1p(depth).std()
    common = 0.85 * state[:, 0] - 0.40 * state[:, 1] + 0.25 * centered_depth
    trigger = rng.binomial(1, expit(-3.0 + common))
    shared_state_null = rng.binomial(1, expit(-3.0 + 0.95 * common))
    dependent = rng.binomial(1, expit(-3.2 + 0.80 * common + 1.85 * trigger))
    independent_null = rng.binomial(1, expit(-3.0 - 0.30 * state[:, 2]))
    detections = pd.DataFrame(
        {
            "peak-trigger": trigger,
            "peak-dependent": dependent,
            "peak-shared-null": shared_state_null,
            "peak-independent-null": independent_null,
        }
    )
    embedding = pd.DataFrame(state, columns=["integration__0", "integration__1", "integration__2"])
    return detections, embedding, pd.Series(depth, name="observed_background_library_size")


def _config(seed: int = 17) -> StratifiedBetaBinomialConfig:
    return StratifiedBetaBinomialConfig(
        n_state_clusters=6,
        n_depth_bins=2,
        min_stratum_cells=50,
        min_feature_events=10,
        n_permutations=99,
        random_state=seed,
    )


def _edge(result, left: str, right: str) -> pd.Series:
    wanted = {left, right}
    matches = result.edge_evidence.loc[
        result.edge_evidence.apply(
            lambda row: {str(row["node1"]), str(row["node2"])} == wanted,
            axis=1,
        )
    ]
    assert len(matches) == 1
    return matches.iloc[0]


def test_sparse_direct_edge_survives_state_depth_null_but_shared_state_does_not() -> None:
    detections, state, depth = _synthetic_sparse_world()
    result = estimate_stratified_beta_binomial_edges(
        detections,
        state,
        depth,
        config=_config(),
    )

    direct = _edge(result, "peak-trigger", "peak-dependent")
    shared_null = _edge(result, "peak-trigger", "peak-shared-null")
    independent_null = _edge(result, "peak-trigger", "peak-independent-null")
    assert direct["joint_detection_count"] >= 10
    assert direct["pooled_log_odds"] > 0.6
    assert direct["posterior_probability_positive_practical"] > 0.95
    assert direct["empirical_p_positive"] <= 0.02
    assert bool(direct["selected"])

    assert shared_null["empirical_p_positive"] > 0.10
    assert not bool(shared_null["selected"])
    assert independent_null["empirical_p_positive"] > 0.10
    assert not bool(independent_null["selected"])
    assert result.metadata["orientation"] == "abstained"
    assert result.metadata["null_contract"]["preserves_feature_events_per_stratum"]


def test_beta_binomial_local_log_odds_is_symmetric() -> None:
    detections, state, depth = _synthetic_sparse_world(n_cells=800)
    config = _config(seed=5)
    strata, _, _ = assign_state_depth_strata(state, depth, config=config)
    left = detections["peak-trigger"].to_numpy(dtype=np.int8)
    right = detections["peak-dependent"].to_numpy(dtype=np.int8)
    forward = _local_beta_binomial_tables(
        left,
        right,
        strata,
        node1="a",
        node2="b",
        config=config,
    ).sort_values("stratum")
    reverse = _local_beta_binomial_tables(
        right,
        left,
        strata,
        node1="b",
        node2="a",
        config=config,
    ).sort_values("stratum")
    np.testing.assert_allclose(
        forward["posterior_log_odds"],
        reverse["posterior_log_odds"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        forward["posterior_log_odds_variance"],
        reverse["posterior_log_odds_variance"],
        rtol=0,
        atol=1e-12,
    )


def test_observed_table_contract_rejects_truth_and_full_depth() -> None:
    detections, state, depth = _synthetic_sparse_world(n_cells=900)
    cell_ids = np.asarray([f"cell-{i}" for i in range(len(detections))])
    cells = detections.copy()
    cells.insert(0, "observed_background_library_size", depth.to_numpy())
    cells.insert(0, "observed_library_size", depth.to_numpy() + detections.sum(axis=1))
    cells.insert(0, "cell", cell_ids)
    embeddings = state.copy()
    embeddings.insert(0, "cell", cell_ids)
    result = estimate_from_observed_tables(
        cells,
        embeddings,
        feature_columns=list(detections.columns),
        config=_config(seed=21),
    )
    assert result.metadata["observed_table_contract"]["truth_pairing_or_external_inputs"] is False
    assert set(result.assignments["cell"]) == set(cell_ids)
    assert len(frame_sha256(result.edge_evidence)) == 64

    with pytest.raises(ValueError, match="forbidden"):
        estimate_from_observed_tables(
            cells.rename(columns={"peak-trigger": "truth__trigger"}),
            embeddings,
            feature_columns=["truth__trigger", "peak-dependent"],
            config=_config(),
        )
    with pytest.raises(ValueError, match="background/leave-focal-out"):
        estimate_from_observed_tables(
            cells,
            embeddings,
            feature_columns=list(detections.columns),
            depth_column="observed_library_size",
            config=_config(),
        )


def test_permutation_evidence_is_reproducible() -> None:
    detections, state, depth = _synthetic_sparse_world(n_cells=1_000, seed=44)
    first = estimate_stratified_beta_binomial_edges(
        detections,
        state,
        depth,
        config=_config(seed=2),
    )
    second = estimate_stratified_beta_binomial_edges(
        detections,
        state,
        depth,
        config=_config(seed=2),
    )
    assert frame_sha256(first.edge_evidence) == frame_sha256(second.edge_evidence)
