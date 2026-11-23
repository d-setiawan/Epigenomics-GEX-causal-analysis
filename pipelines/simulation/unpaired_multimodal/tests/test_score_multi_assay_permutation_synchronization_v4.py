from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as scorer  # noqa: E402


def _core(*, expands: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hyperedge_id": "core_1",
                "selection_status": "selected",
                "hypothesis_selected": True,
                "modalities": "a;b;c",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["a_s", "b_s"], ["a_s", "c_s"]]
                ),
                "candidate_cross_assay_feature_pair_count": 2,
                "candidate_assay_pair_count": 2,
                "exact_assay_participation_claimed": False,
                "assay_participation_is_set_valued": True,
                "candidate_pairs_may_be_expanded_as_edges": expands,
            }
        ]
    )


def _synchronized() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "synchronized_hyperedge_id": "sync_1",
                "selection_status": "selected",
                "hypothesis_selected": True,
                "modalities": "c;d",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["c_s", "d_s"]]
                ),
                "candidate_cross_assay_feature_pair_count": 1,
                "candidate_assay_pair_count": 1,
                "exact_assay_participation_claimed": True,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        ]
    )


def _truth(
    truth_id: str, left: str, right: str, modalities: set[str]
) -> dict:
    pair = core_scorer._pair(left, right)
    return {
        "truth_id": truth_id,
        "equivalence_pairs": {pair},
        "exact_pair": pair,
        "modalities": modalities,
        "direction_identifiable": False,
    }


def test_combined_scoring_preserves_sets_and_adds_incremental_truth() -> None:
    predictions = scorer.normalize_predictions(_core(), _synchronized())
    truths = [
        _truth("t1", "a_s", "b_s", {"a", "b"}),
        _truth("t2", "c_s", "d_s", {"c", "d"}),
    ]

    scored, metrics = scorer.score_predictions(predictions, truths, set())

    shared = metrics["shared_state_hyperedge"]
    assert shared["selected_hyperedges"] == 2
    assert shared["true_positive_hyperedges"] == 2
    assert shared["false_positive_hyperedges"] == 0
    assert shared["shared_state_hyperedge_precision"] == 1.0
    assert shared["shared_state_hyperedge_recall"] == 1.0

    participation = metrics[
        "assay_participation_and_sentinel_false_positive"
    ]
    assert participation["predicted_participating_assays"] == 2
    assert participation["correctly_recovered_assay_participations"] == 2
    assert participation["ambiguous_assay_participation_hypotheses"] == 1
    assert participation["candidate_assay_pair_alternatives"] == 2

    equivalence = metrics["equivalence_class"]
    assert equivalence["average_ambiguity_set_size"] == 1.5
    assert not equivalence["candidate_pair_alternatives_expanded_as_edges"]

    stress = metrics["identifiability_stress_test"]
    assert stress["exact_cross_assay_pair_predictions"] == 1
    assert stress["exact_cross_assay_pair_true_positives"] == 1
    assert stress["included_in_reliability_gate"] is False
    assert set(scored["prediction_source"]) == {
        "v444_component_conditioned_bridge",
        "v445_multi_assay_synchronization",
    }


def test_normalization_rejects_pair_expansion() -> None:
    with pytest.raises(ValueError, match="attempted pair expansion"):
        scorer.normalize_predictions(_core(expands=True), _synchronized())
