from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_mutual_nearest_margin_rescue_v4 as mnn  # noqa: E402


def test_mutual_nearest_margin_requires_reciprocal_unique_winner() -> None:
    frame = pd.DataFrame(
        [
            {"world_key": "w", "left_modality": "a", "right_modality": "b"},
            {"world_key": "w", "left_modality": "a", "right_modality": "c"},
            {"world_key": "w", "left_modality": "b", "right_modality": "c"},
            {"world_key": "x", "left_modality": "d", "right_modality": "e"},
        ]
    )
    margins, mutual = mnn._mutual_nearest_absolute_margin(
        frame, np.asarray([0.9, 0.4, 0.3, 0.8])
    )

    assert np.allclose(margins, [0.5, -0.5, -0.6, 0.0])
    assert mutual.tolist() == [True, False, False, False]


def test_mutual_nearest_bootstrap_threshold_is_inclusive() -> None:
    assert mnn._bootstrap_frequency_passes(80, 100)
    assert not mnn._bootstrap_frequency_passes(79, 100)
    assert not mnn._bootstrap_frequency_passes(0, 0)
    assert "perfect_matching" in mnn.FIXED_SELECTION_RULE


def test_mutual_nearest_union_keeps_overlap_as_one_ambiguity() -> None:
    world = "block_a__world_seed7"
    evidence_frames = {
        world: pd.DataFrame(
            [{"world_key": world, "block": "block_a", "seed": 7}]
        )
    }
    baseline = {
        world: [
            {
                "pairs": {frozenset(("a", "b"))},
                "features": {("a_s", "b_s")},
                "audit": [{"source": "target_safe_v452"}],
                "source": "target_safe_v452",
            }
        ]
    }
    stable_rows = {
        world: [
            {
                "left_modality": "b",
                "right_modality": "c",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["b_s", "c_s"]]
                ),
                "minimum_nested_absolute_margin": 0.5,
                "bootstrap_selection_frequency": 0.99,
            }
        ]
    }

    candidates, merged = mnn._union_candidates(
        target_keys=[world],
        evidence_frames=evidence_frames,
        baseline=baseline,
        stable_rows=stable_rows,
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["modalities"] == "a;b;c"
    assert int(row["candidate_assay_pair_count"]) == 2
    assert bool(row["assay_participation_is_set_valued"])
    assert not bool(row["candidate_pairs_may_be_expanded_as_edges"])
    assert row["selection_sources"] == (
        "target_safe_v452;v453_bootstrap_mutual_nearest_margin"
    )
    assert merged[(world, frozenset(("a", "b")))] == row[
        "mutual_nearest_hyperedge_id"
    ]
    assert merged[(world, frozenset(("b", "c")))] == row[
        "mutual_nearest_hyperedge_id"
    ]
