"""Tests for leakage-safe hierarchical within-assay edge evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from hierarchical_sparse_edges import (  # noqa: E402
    HierarchicalSparseConfig,
    empirical_bayes_pool,
    estimate_from_observed_tables,
    estimate_hierarchical_sparse_edges,
    symmetric_global_pseudolikelihood,
)


def _synthetic_conditional_dependence(
    n_cells: int = 3_000,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Make a confounded null pair and one genuine residual interaction."""

    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n_cells, 2))
    log_depth = (
        2.2
        + 0.55 * state[:, 0]
        - 0.25 * state[:, 1]
        + rng.normal(scale=0.45, size=n_cells)
    )
    background_depth = rng.poisson(np.exp(log_depth) * 4.0)
    observed_log_depth = np.log1p(background_depth)
    centered_depth = observed_log_depth - observed_log_depth.mean()

    peak_x = rng.binomial(
        1,
        expit(
            -2.0
            + 0.90 * state[:, 0]
            - 0.40 * state[:, 1]
            + 0.25 * centered_depth
        ),
    )
    confounded_null = rng.binomial(
        1,
        expit(
            -1.8
            + 1.00 * state[:, 0]
            - 0.45 * state[:, 1]
            + 0.30 * centered_depth
        ),
    )
    dependent_peak = rng.binomial(
        1,
        expit(
            -2.1
            + 0.85 * state[:, 0]
            - 0.35 * state[:, 1]
            + 0.25 * centered_depth
            + 1.60 * peak_x
        ),
    )
    detections = pd.DataFrame(
        {
            "peak-X": peak_x,
            "peak-null": confounded_null,
            "peak-dependent": dependent_peak,
        }
    )
    integration_state = pd.DataFrame(state, columns=["state-1", "state-2"])
    return detections, integration_state, background_depth


def _test_config(seed: int = 7) -> HierarchicalSparseConfig:
    return HierarchicalSparseConfig(
        n_microstates=10,
        min_microstate_cells=80,
        min_peak_events=20,
        crossfit_folds=5,
        min_abs_residual_log_odds=0.20,
        evidence_threshold=0.95,
        random_state=seed,
    )


def _edge(result, left: str, right: str) -> pd.Series:
    names = {left, right}
    use = result.edge_evidence.apply(
        lambda row: {row["node1"], row["node2"]} == names,
        axis=1,
    )
    matches = result.edge_evidence.loc[use]
    assert len(matches) == 1
    return matches.iloc[0]


def test_conditioning_removes_shared_state_but_keeps_residual_edge() -> None:
    detections, state, depth = _synthetic_conditional_dependence()
    result = estimate_hierarchical_sparse_edges(
        detections,
        state,
        depth,
        config=_test_config(),
    )

    null_edge = _edge(result, "peak-X", "peak-null")
    dependent_edge = _edge(result, "peak-X", "peak-dependent")

    # The null pair looks associated marginally because both peaks share state.
    assert null_edge["unconditional_phi"] > 0.12
    assert null_edge["posterior_probability_residual_dependence"] < 0.80
    assert not bool(null_edge["selected"])

    # Conditioning retains the deliberately injected within-assay interaction.
    assert dependent_edge["residual_log_odds"] > 1.0
    assert dependent_edge["ci95_lower"] > 0.8
    assert dependent_edge["posterior_probability_residual_dependence"] > 0.99
    assert bool(dependent_edge["selected"])
    assert dependent_edge["residual_log_odds_sd"] > 0
    assert dependent_edge["n_microstates_used"] >= 8

    assert result.metadata["orientation"] == "abstained"
    assert result.metadata["n_edges_selected"] == 1
    assert set(result.selected_edges[["node1", "node2"]].iloc[0]) == {
        "peak-X",
        "peak-dependent",
    }


def test_empirical_bayes_pool_shrinks_a_noisy_microstate() -> None:
    local = pd.DataFrame(
        {
            "microstate": [0, 1, 2],
            "raw_residual_log_odds": [0.00, 0.30, 2.50],
            "raw_variance": [0.04, 0.04, 1.50],
        }
    )
    summary, pooled = empirical_bayes_pool(local)

    noisy = pooled.loc[pooled["microstate"] == 2].iloc[0]
    precise = pooled.loc[pooled["microstate"] == 0].iloc[0]
    pooled_effect = summary["pooled_effect"]
    assert summary["pooled_sd"] > 0
    assert summary["between_microstate_variance"] >= 0
    assert abs(noisy["shrunk_residual_log_odds"] - pooled_effect) < abs(
        noisy["raw_residual_log_odds"] - pooled_effect
    )
    assert noisy["shrinkage_weight"] < precise["shrinkage_weight"]
    assert noisy["shrunk_sd"] > 0


def test_observed_table_contract_is_modality_agnostic_and_leakage_safe() -> None:
    rng = np.random.default_rng(91)
    n_cells = 800
    cell_ids = np.asarray([f"assay_cell_{i}" for i in range(n_cells)])
    state = rng.normal(size=(n_cells, 2))
    background_depth = rng.poisson(np.exp(2.5 + 0.3 * state[:, 0]))
    first = rng.binomial(1, expit(-1.3 + 0.7 * state[:, 0]))
    second = rng.binomial(1, expit(-1.4 + 0.6 * state[:, 0] + 1.1 * first))
    retained = np.arange(n_cells) % 5 != 0

    # Counts are intentionally not just zero/one, and feature names assume no
    # RNA/chromatin/protein naming convention.
    cells = pd.DataFrame(
        {
            "cell": cell_ids,
            "observed_library_size": background_depth + 100 + first + second,
            "observed_background_library_size": background_depth,
            "feature::alpha/1": first * rng.integers(1, 5, size=n_cells),
            "feature::beta-2": second * rng.integers(1, 8, size=n_cells),
            "truth__oracle_peak": first,
            "external__support": rng.normal(size=n_cells),
        }
    )
    embeddings = pd.DataFrame(
        {
            "cell": cell_ids,
            "retained_target": np.where(retained, "TRUE", "FALSE"),
            "integration__dim0": state[:, 0],
            "integration__dim1": state[:, 1],
            "truth__latent_state": rng.normal(size=n_cells),
        }
    )
    config = HierarchicalSparseConfig(
        n_microstates=6,
        min_microstate_cells=35,
        min_peak_events=10,
        random_state=9,
    )
    result = estimate_from_observed_tables(
        cells,
        embeddings,
        retained_cells_only=True,
        config=config,
    )

    contract = result.metadata["observed_table_contract"]
    assert contract["feature_columns"] == ["feature::alpha/1", "feature::beta-2"]
    assert contract["state_columns"] == ["integration__dim0", "integration__dim1"]
    assert contract["depth_column"] == "observed_background_library_size"
    assert contract["truth_or_pairing_inputs"] is False
    assert result.metadata["n_cells"] == int(retained.sum())
    assert set(result.assignments["cell"]) == set(cell_ids[retained])
    expected_events = int((cells.loc[retained, "feature::alpha/1"] > 0).sum())
    observed_events = int(
        result.peak_diagnostics.set_index("feature").loc[
            "feature::alpha/1", "detection_count"
        ]
    )
    assert observed_events == expected_events

    with pytest.raises(ValueError, match="forbidden"):
        estimate_from_observed_tables(
            cells,
            embeddings,
            feature_columns=["feature::alpha/1", "truth__oracle_peak"],
            config=config,
        )
    with pytest.raises(ValueError, match="background-derived"):
        estimate_from_observed_tables(
            cells,
            embeddings,
            feature_columns=["feature::alpha/1", "feature::beta-2"],
            depth_column="observed_library_size",
            config=config,
        )


def test_core_api_rejects_truth_named_inputs_and_drops_uninformative_peaks() -> None:
    detections, state, depth = _synthetic_conditional_dependence(n_cells=600)
    detections["never-detected"] = 0
    config = HierarchicalSparseConfig(
        n_microstates=4,
        min_microstate_cells=30,
        min_peak_events=8,
        random_state=3,
    )
    result = estimate_hierarchical_sparse_edges(
        detections,
        state,
        depth,
        config=config,
    )
    assert "never-detected" in result.metadata["dropped_features"]
    diagnostic = result.peak_diagnostics.set_index("feature").loc["never-detected"]
    assert diagnostic["status"] == "dropped_too_few_events"
    assert not (
        result.edge_evidence[["node1", "node2"]] == "never-detected"
    ).to_numpy().any()

    with pytest.raises(ValueError, match="forbidden"):
        estimate_hierarchical_sparse_edges(
            detections.rename(columns={"peak-X": "truth__peak-X"}),
            state,
            depth,
            config=config,
        )
    with pytest.raises(ValueError, match="cannot be integration state"):
        estimate_hierarchical_sparse_edges(
            detections,
            state.rename(columns={"state-1": "latent__state-1"}),
            depth,
            config=config,
        )
    with pytest.raises(ValueError, match="library-size columns"):
        estimate_hierarchical_sparse_edges(
            detections.rename(columns={"peak-X": "observed_library_size"}),
            state,
            depth,
            config=config,
        )
    with pytest.raises(ValueError, match="Full library size"):
        estimate_hierarchical_sparse_edges(
            detections,
            state,
            pd.Series(depth, name="observed_library_size"),
            config=config,
        )


def test_global_pseudolikelihood_is_symmetric_and_positive_only() -> None:
    rng = np.random.default_rng(414)
    n_cells = 2_500
    state = rng.normal(size=n_cells)
    p_left = expit(-1.5 + 0.45 * state)
    p_right = expit(-1.8 - 0.30 * state)
    left = rng.binomial(1, p_left)
    positive_right = rng.binomial(
        1,
        expit(np.log(p_right / (1.0 - p_right)) + 1.1 * left),
    )
    fit = symmetric_global_pseudolikelihood(
        left,
        positive_right,
        p_left,
        p_right,
        practical_threshold=0.20,
    )
    swapped = symmetric_global_pseudolikelihood(
        positive_right,
        left,
        p_right,
        p_left,
        practical_threshold=0.20,
    )

    assert fit["converged"]
    assert fit["posterior_mean"] > 0.65
    assert fit["posterior_probability_gt_practical"] > 0.99
    assert fit["posterior_sd"] > 0
    assert fit["prior_sd"] == pytest.approx(2.50)
    assert fit["conditional_weight"] == pytest.approx(0.50)
    for field in (
        "posterior_mean",
        "posterior_sd",
        "posterior_probability_positive",
        "posterior_probability_gt_practical",
    ):
        assert fit[field] == pytest.approx(swapped[field], abs=1e-12)

    negative_right = rng.binomial(
        1,
        expit(np.log(p_right / (1.0 - p_right)) - 1.3 * left),
    )
    negative = symmetric_global_pseudolikelihood(
        left,
        negative_right,
        p_left,
        p_right,
        practical_threshold=0.20,
    )
    assert negative["posterior_mean"] < -0.8
    assert negative["posterior_probability_lt_negative_practical"] > 0.99
    assert negative["posterior_probability_gt_practical"] < 0.01


def test_global_laplace_evidence_has_synthetic_null_calibration_and_power() -> None:
    null_z: list[float] = []
    null_selected: list[bool] = []
    powered: list[bool] = []
    for replicate in range(80):
        rng = np.random.default_rng(18_000 + replicate)
        n_cells = 1_200
        state = rng.normal(size=n_cells)
        p_left = expit(-1.6 + 0.55 * state)
        p_right = expit(-1.8 - 0.40 * state)
        left = rng.binomial(1, p_left)
        null_right = rng.binomial(1, p_right)
        null_fit = symmetric_global_pseudolikelihood(
            left,
            null_right,
            p_left,
            p_right,
            practical_threshold=0.20,
        )
        null_z.append(null_fit["posterior_mean"] / null_fit["posterior_sd"])
        null_selected.append(
            bool(null_fit["posterior_probability_gt_practical"] >= 0.95)
        )

        powered_right = rng.binomial(
            1,
            expit(np.log(p_right / (1.0 - p_right)) + 1.0 * left),
        )
        powered_fit = symmetric_global_pseudolikelihood(
            left,
            powered_right,
            p_left,
            p_right,
            practical_threshold=0.20,
        )
        powered.append(
            bool(powered_fit["posterior_probability_gt_practical"] >= 0.95)
        )

    # The half-weighted symmetric composite avoids counting the same pair twice:
    # under a known-offset null, its Laplace z scores are approximately N(0, 1).
    assert abs(float(np.mean(null_z))) < 0.25
    assert 0.75 < float(np.std(null_z, ddof=1)) < 1.25
    assert float(np.mean(np.abs(null_z) <= 1.959963984540054)) >= 0.90
    assert float(np.mean(null_selected)) <= 0.05
    assert float(np.mean(powered)) >= 0.90


def test_positive_evidence_selector_is_explicit_and_backward_compatible() -> None:
    detections, state, depth = _synthetic_conditional_dependence(n_cells=1_800, seed=72)
    outputs = {}
    for rule in ("microstate", "global", "intersection"):
        cfg = HierarchicalSparseConfig(
            n_microstates=8,
            min_microstate_cells=70,
            min_peak_events=20,
            positive_evidence_rule=rule,
            random_state=29,
        )
        outputs[rule] = estimate_hierarchical_sparse_edges(
            detections,
            state,
            depth,
            config=cfg,
        )

    for rule, result in outputs.items():
        edge = _edge(result, "peak-X", "peak-dependent")
        assert edge["positive_evidence_rule"] == rule
        assert edge["posterior_probability_positive"] == pytest.approx(
            edge["selection_probability_positive"]
        )
        assert edge["microstate_residual_log_odds"] == pytest.approx(
            edge["residual_log_odds"]
        )
        assert edge["global_pseudolikelihood_log_odds_sd"] > 0
        assert bool(edge["global_pseudolikelihood_converged"])

    micro = _edge(outputs["microstate"], "peak-X", "peak-dependent")
    global_edge = _edge(outputs["global"], "peak-X", "peak-dependent")
    intersection = _edge(outputs["intersection"], "peak-X", "peak-dependent")
    assert micro["selection_probability_positive"] == pytest.approx(
        micro["microstate_posterior_probability_positive"]
    )
    assert global_edge["selection_probability_positive"] == pytest.approx(
        global_edge["global_posterior_probability_gt_practical"]
    )
    assert intersection["selection_probability_positive"] == pytest.approx(
        min(
            intersection["microstate_posterior_probability_positive"],
            intersection["global_posterior_probability_gt_practical"],
        )
    )
