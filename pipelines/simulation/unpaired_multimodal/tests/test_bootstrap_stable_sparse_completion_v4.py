from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_stable_sparse_completion_v4 as stable  # noqa: E402


def test_bootstrap_frequency_threshold_is_inclusive_and_requires_replicates() -> None:
    assert stable._bootstrap_frequency_passes(80, 100)
    assert not stable._bootstrap_frequency_passes(79, 100)
    assert not stable._bootstrap_frequency_passes(0, 0)


def test_candidate_identity_excludes_modality_and_feature_from_model_inputs() -> None:
    row = pd.DataFrame(
        [
            {
                "world_key": "block_a__world_seed7",
                "modality": "assay_a",
                "feature": "assay_a_S",
            }
        ]
    ).itertuples(index=False).__next__()

    assert stable._candidate_identity(row) == (
        "block_a__world_seed7",
        "assay_a",
        "assay_a_S",
    )
    assert "modality" not in stable.FEATURE_COLUMNS
    assert "feature" not in stable.FEATURE_COLUMNS


def test_union_merges_stable_pair_with_overlapping_v447_component() -> None:
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
                "audit": [{"source": "target_safe_v447"}],
                "source": "target_safe_v447",
            }
        ]
    }
    selected = {
        world: [
            {
                "unresolved_modality": "b",
                "partner_modality": "c",
                "candidate_reporter_feature": "b_s",
                "partner_reporter_feature": "c_s",
                "reporter_bootstrap_selection_frequency": 0.98,
            }
        ]
    }

    candidates, merged = stable._union_candidates(
        target_keys=[world],
        evidence_frames=evidence_frames,
        baseline=baseline,
        stable_pair_rows=selected,
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["modalities"] == "a;b;c"
    assert int(row["candidate_assay_pair_count"]) == 2
    assert int(row["candidate_cross_assay_feature_pair_count"]) == 2
    assert bool(row["assay_participation_is_set_valued"])
    assert not bool(row["candidate_pairs_may_be_expanded_as_edges"])
    assert row["selection_sources"] == (
        "target_safe_v447;v451_bootstrap_stable_sparse_completion"
    )
    assert merged[(world, frozenset(("a", "b")))] == row[
        "stable_hyperedge_id"
    ]
    assert merged[(world, frozenset(("b", "c")))] == row[
        "stable_hyperedge_id"
    ]
