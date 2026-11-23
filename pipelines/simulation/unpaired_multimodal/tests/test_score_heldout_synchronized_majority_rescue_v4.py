from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_heldout_synchronized_majority_rescue_v4 as scorer  # noqa: E402


def test_rescue_normalization_preserves_provenance_and_no_expansion() -> None:
    core = pd.DataFrame(
        [
            {
                "hyperedge_id": "core_1",
                "selection_route": "core",
                "hypothesis_selected": True,
                "modalities": "a;b",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["a_s", "b_s"]]
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
                "synchronized_majority_hyperedge_id": "rescue_1",
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

    normalized = scorer.normalize_rescue_predictions(core, candidates)

    assert normalized["hyperedge_id"].tolist() == ["core_1", "rescue_1"]
    assert normalized["prediction_source"].tolist() == [
        "v444_component_conditioned_bridge",
        "v446_heldout_synchronized_majority",
    ]
    assert normalized["exact_assay_participation_claimed"].all()
    assert not normalized["candidate_pairs_may_be_expanded_as_edges"].any()
