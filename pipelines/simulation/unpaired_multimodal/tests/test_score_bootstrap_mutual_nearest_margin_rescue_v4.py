from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_bootstrap_mutual_nearest_margin_rescue_v4 as scorer  # noqa: E402


def test_normalize_predictions_keeps_candidate_set_unexpanded() -> None:
    core = pd.DataFrame(
        [
            {
                "hyperedge_id": "core_1",
                "selection_route": "core",
                "hypothesis_selected": True,
                "modalities": "x;y",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["x_s", "y_s"]]
                ),
                "candidate_cross_assay_feature_pair_count": 1,
                "candidate_assay_pair_count": 1,
                "exact_assay_participation_claimed": True,
                "assay_participation_is_set_valued": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "mutual_nearest_hyperedge_id": "mnn_001",
                "selection_status": "selected_as_mutual_nearest_stable_ambiguity",
                "hypothesis_selected": True,
                "modalities": "a;b;c",
                "candidate_assay_pairs_json": json.dumps(
                    [["a", "b"], ["b", "c"]]
                ),
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["a_s", "b_s"], ["b_s", "c_s"]]
                ),
                "candidate_cross_assay_feature_pair_count": 2,
                "candidate_assay_pair_count": 2,
                "assay_participation_is_set_valued": True,
                "exact_assay_participation_claimed": False,
                "hypothesis_type": "cross_assay_shared_state_equivalence",
                "within_assay_direct_structure_claimed": False,
                "exact_cross_assay_pair_claimed": False,
                "cross_assay_direction_claimed": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        ]
    )

    predictions = scorer.normalize_predictions(core, candidates)

    assert len(predictions) == 2
    assert predictions.iloc[1]["prediction_source"] == (
        "v453_bootstrap_mutual_nearest_margin_rescue"
    )
    assert not bool(
        predictions.iloc[1]["candidate_pairs_may_be_expanded_as_edges"]
    )
    assert scorer.SCHEMA.endswith(".scorer.v1")
