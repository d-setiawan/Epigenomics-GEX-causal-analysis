from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_bootstrap_stable_sparse_completion_v4 as scorer  # noqa: E402


def test_stable_completion_normalization_preserves_unexpanded_ambiguity() -> None:
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
                "stable_hyperedge_id": "stable_1",
                "selection_status": "selected_as_bootstrap_stable_ambiguity",
                "hypothesis_selected": True,
                "modalities": "c;d;e",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["c_s", "d_s"], ["d_s", "e_s"]]
                ),
                "candidate_cross_assay_feature_pair_count": 2,
                "candidate_assay_pair_count": 2,
                "exact_assay_participation_claimed": False,
                "assay_participation_is_set_valued": True,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        ]
    )

    normalized = scorer.normalize_predictions(core, candidates)

    assert normalized["hyperedge_id"].tolist() == ["core_1", "stable_1"]
    assert normalized["prediction_source"].tolist() == [
        "v444_component_conditioned_bridge",
        "v451_bootstrap_stable_sparse_completion",
    ]
    assert normalized.iloc[1]["assay_participation_is_set_valued"]
    assert not normalized.iloc[1]["exact_assay_participation_claimed"]
    assert not normalized["candidate_pairs_may_be_expanded_as_edges"].any()
