from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_empirical_copula_pair_rescue_v4 as copula  # noqa: E402


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=copula.FEATURE_COLUMNS)


def test_empirical_copula_score_respects_all_four_fixed_directions() -> None:
    training = _frame(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.8, 0.1],
            [2.0, 2.0, 0.5, 0.2],
        ]
    )
    values = _frame(
        [
            [3.0, 3.0, 0.2, 0.3],
            [0.0, 0.0, 1.2, 0.0],
        ]
    )

    first = copula._empirical_copula_product(training, values)
    second = copula._empirical_copula_product(training, values)

    assert first.tolist() == second.tolist()
    assert first[0] > first[1]
    assert "left_modality" not in copula.FEATURE_COLUMNS
    assert "right_modality" not in copula.FEATURE_COLUMNS


def test_copula_bootstrap_threshold_is_inclusive() -> None:
    assert copula._bootstrap_frequency_passes(80, 100)
    assert not copula._bootstrap_frequency_passes(79, 100)
    assert not copula._bootstrap_frequency_passes(0, 0)


def test_copula_union_preserves_overlapping_pair_as_one_ambiguity() -> None:
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
                "audit": [{"source": "target_safe_v451"}],
                "source": "target_safe_v451",
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
                "minimum_nested_copula_score": 0.95,
                "bootstrap_selection_frequency": 0.99,
            }
        ]
    }

    candidates, merged = copula._union_candidates(
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
        "target_safe_v451;v452_bootstrap_empirical_copula"
    )
    assert merged[(world, frozenset(("a", "b")))] == row[
        "copula_hyperedge_id"
    ]
    assert merged[(world, frozenset(("b", "c")))] == row[
        "copula_hyperedge_id"
    ]
