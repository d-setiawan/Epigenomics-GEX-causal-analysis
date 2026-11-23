from __future__ import annotations

import pandas as pd
import pytest

from pipelines.simulation.unpaired_multimodal.scripts.score_structure_anchored_loading_groups_v4 import (
    score_structure_anchored_groups,
)


def _groups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_id": "g1",
                "edge_type": "within_assay_shared_state_support",
                "selection_status": "selected",
                "modality": "chrom_M1",
                "members": "a__M1;b__M1",
                "n_members": 2,
                "direct_causal_claimed": False,
                "cross_assay_membership_claimed": False,
            },
            {
                "relation_id": "g2",
                "edge_type": "within_assay_shared_state_support",
                "selection_status": "selected",
                "modality": "chrom_M2",
                "members": "x__M2;y__M2",
                "n_members": 2,
                "direct_causal_claimed": False,
                "cross_assay_membership_claimed": False,
            },
        ]
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node1": "a__M1", "node2": "b__M1", "truth_type": "within_assay_shared_state", "mark": "M1"},
            {"node1": "x__M2", "node2": "z__M2", "truth_type": "within_assay_shared_state", "mark": "M2"},
        ]
    )


def test_scores_pairs_and_assay_completeness() -> None:
    metrics, pairs, assays = score_structure_anchored_groups(_groups(), _truth(), 7)
    assert metrics["n_true_positive_pairs"] == 1
    assert metrics["n_false_positive_pairs"] == 1
    assert metrics["n_false_negative_pairs"] == 1
    assert metrics["pair_precision"] == 0.5
    assert metrics["pair_recall"] == 0.5
    assert metrics["n_assays_complete_without_false_pair"] == 1
    assert set(pairs["classification"]) == {"true_positive", "false_positive", "false_negative"}
    assert assays.set_index("modality").loc["chrom_M1", "complete_without_false_pair"]


def test_rejects_semantic_overclaim() -> None:
    groups = _groups()
    groups.loc[0, "cross_assay_membership_claimed"] = True
    with pytest.raises(ValueError, match="cross-assay membership"):
        score_structure_anchored_groups(groups, _truth(), 7)


def test_rejects_more_than_one_pair_per_assay() -> None:
    groups = pd.concat([_groups(), _groups().iloc[[0]].assign(relation_id="g3", members="c__M1;d__M1")])
    with pytest.raises(ValueError, match="At most one"):
        score_structure_anchored_groups(groups, _truth(), 7)
