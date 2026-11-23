from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_null_calibrated_stability_rescue_v4 as rescue  # noqa: E402


def test_baseline_counts_are_verified() -> None:
    frame = pd.DataFrame(
        [
            {
                "world_key": "block_a__world_seed7",
                "candidate_assay_pairs_json": '[["a","b"]]',
                "candidate_assay_pair_count": 2,
                "candidate_cross_assay_feature_pairs_json": '[["a_s","b_s"]]',
                "candidate_cross_assay_feature_pair_count": 1,
                "candidate_pair_selection_audit_json": "[]",
            }
        ]
    )
    with pytest.raises(ValueError, match="assay-pair count mismatch"):
        rescue._baseline_components(frame)


def test_union_preserves_connected_ambiguity_without_edge_expansion() -> None:
    world = "block_a__world_seed7"
    world_frames = {
        world: pd.DataFrame(
            [{"world_key": world, "block": "block_a", "seed": 7}]
        )
    }
    baseline = {
        world: [
            {
                "pairs": {frozenset(("a", "b"))},
                "features": {("a_s", "b_s")},
                "audit": [{"source": "target_safe_v454"}],
                "source": "target_safe_v454",
            }
        ]
    }
    stable = {
        world: [
            {
                "left_modality": "b",
                "right_modality": "c",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["b_s", "c_s"]]
                ),
                "bootstrap_selection_frequency": 0.52,
                "empirical_null_maximum_selection_count": 27,
                "posterior_probability_exceeds_worst_null": 0.999,
            }
        ]
    }

    candidates, merged = rescue._union_candidates(
        target_keys=[world],
        world_frames=world_frames,
        baseline=baseline,
        stable_rows=stable,
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["modalities"] == "a;b;c"
    assert int(row["candidate_assay_pair_count"]) == 2
    assert bool(row["assay_participation_is_set_valued"])
    assert not bool(row["candidate_pairs_may_be_expanded_as_edges"])
    assert not bool(row["exact_cross_assay_pair_claimed"])
    assert not bool(row["cross_assay_direction_claimed"])
    assert merged[(world, frozenset(("a", "b")))] == row[
        "null_calibrated_hyperedge_id"
    ]
    assert merged[(world, frozenset(("b", "c")))] == row[
        "null_calibrated_hyperedge_id"
    ]

