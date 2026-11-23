from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crossfit_interaction_reporter_rescue_v4 as rescue  # noqa: E402


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=rescue.FEATURE_COLUMNS)


def test_interaction_model_matrix_is_identity_blind_and_zero_imputes() -> None:
    frame = _frame([[1.0, 2.0, np.nan, np.inf]])
    frame["modality"] = "must_not_enter_model"

    observed = rescue._model_matrix(frame)

    assert observed.tolist() == [[1.0, 2.0, 0.0, 0.0]]
    assert "modality" not in rescue.FEATURE_COLUMNS


def test_interaction_fold_is_deterministic_and_converges() -> None:
    negatives = [
        [0.0, 0.0, 4.0, 0.0],
        [0.2, 0.1, 3.8, 0.1],
        [0.1, 0.3, 3.5, 0.2],
        [0.3, 0.2, 3.7, 0.1],
    ]
    positives = [
        [3.5, 3.8, 0.2, 3.4],
        [4.0, 3.6, 0.0, 3.8],
        [3.7, 4.0, 0.3, 4.0],
        [3.9, 3.7, 0.1, 3.6],
    ]
    training = _frame(negatives + positives)
    labels = np.asarray([False] * 4 + [True] * 4)
    calibration = _frame([negatives[0], positives[0]])
    calibration_labels = np.asarray([False, True])
    target = _frame([negatives[1], positives[1]])

    first = rescue._fit_fold(
        training,
        labels,
        calibration,
        calibration_labels,
        target,
        model_c=0.02,
    )
    second = rescue._fit_fold(
        training,
        labels,
        calibration,
        calibration_labels,
        target,
        model_c=0.02,
    )

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["model_converged"]
    assert len(json.loads(first[2]["polynomial_feature_names_json"])) == 14
    assert first[1][1] > first[1][0]


def test_interaction_union_merges_overlapping_pairs_without_expansion() -> None:
    world = "block_a__world_seed7"
    evidence = pd.DataFrame(
        [
            {
                "world_key": world,
                "block": "block_a",
                "seed": 7,
                "left_modality": "b",
                "right_modality": "c",
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    [["b_s", "c_s"]]
                ),
            }
        ]
    )
    baseline = {
        world: [
            {
                "pairs": {frozenset(("a", "b"))},
                "features": {("a_s", "b_s")},
                "audit": [{"source": "target_safe_v449"}],
                "source": "target_safe_v449",
            }
        ]
    }
    pair = frozenset(("b", "c"))
    score_lookup = {
        (world, pair): {
            "minimum_probability_by_c": {0.02: 0.9, 0.03: 0.91}
        }
    }

    candidates, merged = rescue._union_candidates(
        target_keys=[world],
        evidence_frames={world: evidence},
        baseline=baseline,
        selected_pairs={world: {pair}},
        score_lookup=score_lookup,
        thresholds={0.02: 0.8, 0.03: 0.81},
        all_models_converged=True,
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["modalities"] == "a;b;c"
    assert int(row["candidate_assay_pair_count"]) == 2
    assert int(row["candidate_cross_assay_feature_pair_count"]) == 2
    assert bool(row["assay_participation_is_set_valued"])
    assert not bool(row["candidate_pairs_may_be_expanded_as_edges"])
    assert row["selection_sources"] == (
        "target_safe_v449;v450_nested_interaction_stability"
    )
    assert merged[(world, pair)] == row["interaction_hyperedge_id"]

