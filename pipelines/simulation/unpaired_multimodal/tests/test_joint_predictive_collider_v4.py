from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import joint_predictive_collider_v4 as joint  # noqa: E402


def _synthetic(seed: int = 4) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 2400
    state = rng.normal(size=(n, 3))
    depth = np.exp(0.3 * state[:, 0] + rng.normal(scale=0.4, size=n))
    p1 = rng.binomial(1, 1 / (1 + np.exp(-(-2.5 + 0.5 * state[:, 0]))))
    p2 = rng.binomial(1, 1 / (1 + np.exp(-(-2.3 - 0.4 * state[:, 1]))))
    child_probability = 1 / (
        1 + np.exp(-(-3.1 + 0.25 * state[:, 2] + 1.6 * p1 + 1.4 * p2))
    )
    child = rng.binomial(1, child_probability)
    null = rng.binomial(1, 0.08, size=n)
    detections = pd.DataFrame({"p1": p1, "p2": p2, "child": child, "null": null})
    evidence = pd.DataFrame(
        [
            {
                "source_assay": "chrom_test",
                "node1": "p1",
                "node2": "child",
                "selected_vote_fraction": 1.0,
                "mean_empirical_p_positive": 0.0025,
                "mean_pooled_log_odds": 1.6,
            },
            {
                "source_assay": "chrom_test",
                "node1": "p2",
                "node2": "child",
                "selected_vote_fraction": 1.0,
                "mean_empirical_p_positive": 0.03,
                "mean_pooled_log_odds": 1.3,
            },
            {
                "source_assay": "chrom_test",
                "node1": "p2",
                "node2": "null",
                "selected_vote_fraction": 1.0,
                "mean_empirical_p_positive": 0.04,
                "mean_pooled_log_odds": 1.1,
            },
        ]
    )
    return detections, state, depth, evidence


def test_joint_prediction_selects_the_positive_two_parent_center() -> None:
    detections, state, depth, evidence = _synthetic()

    result = joint.estimate_joint_predictive_colliders(
        detections,
        state,
        depth,
        evidence,
        source_assay="chrom_test",
    )

    assert len(result.selected_wedges) == 1
    selected = result.selected_wedges.iloc[0]
    assert selected["center"] == "child"
    assert {selected["outer1"], selected["outer2"]} == {"p1", "p2"}
    assert selected["relative_gain_margin"] > 0
    assert result.metadata["screen_edges_create_graph_adjacencies"] is False


def test_nonunique_winner_abstains() -> None:
    detections, state, depth, evidence = _synthetic()
    result = joint.estimate_joint_predictive_colliders(
        detections,
        state,
        depth,
        evidence,
        source_assay="chrom_test",
        config=joint.JointColliderConfig(min_relative_gain_margin=1.0),
    )

    assert result.selected_wedges.empty
    assert (
        result.wedge_evidence.loc[0, "selection_reason"]
        == "joint_predictive_threshold_or_margin_failed"
    )


def test_privileged_columns_are_rejected() -> None:
    detections, state, depth, evidence = _synthetic()
    with pytest.raises(ValueError, match="privileged"):
        joint.estimate_joint_predictive_colliders(
            detections.assign(truth_role="unknown"),
            state,
            depth,
            evidence,
            source_assay="chrom_test",
        )
