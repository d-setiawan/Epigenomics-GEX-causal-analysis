from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crossfit_sparse_reporter_completion_v4 as completion  # noqa: E402


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=completion.FEATURE_COLUMNS)


def test_model_matrix_uses_only_frozen_features_and_zero_imputes() -> None:
    frame = _frame([[0.1, 0.2, np.nan, np.inf, 0.3, 1.0, 2.0]])
    frame["modality"] = "must_not_enter_model"
    frame["feature"] = "must_not_enter_model"

    observed = completion._model_matrix(frame)

    assert observed.shape == (1, 7)
    assert observed.tolist() == [[0.1, 0.2, 0.0, 0.0, 0.3, 1.0, 2.0]]
    assert "modality" not in completion.FEATURE_COLUMNS
    assert "feature" not in completion.FEATURE_COLUMNS


def test_polynomial_nested_fold_is_deterministic() -> None:
    negatives = [
        [0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0],
        [0.3, 0.1, 0.1, 0.2, 0.0, 0.0, 1.0],
        [0.2, 0.3, 0.0, 0.2, 0.1, 0.0, 0.0],
    ]
    positives = [
        [0.8, 0.8, 0.6, 0.7, 0.5, 1.0, 3.0],
        [0.9, 0.7, 0.5, 0.8, 0.6, 1.0, 3.0],
        [0.7, 0.9, 0.7, 0.7, 0.7, 1.0, 2.0],
        [0.8, 0.8, 0.8, 0.9, 0.6, 1.0, 3.0],
    ]
    training = _frame(negatives + positives)
    labels = np.asarray([False] * 4 + [True] * 4)
    calibration = _frame([negatives[0], positives[0]])
    calibration_labels = np.asarray([False, True])
    target = _frame([negatives[1], positives[1]])

    first = completion._fit_fold(
        training,
        labels,
        calibration,
        calibration_labels,
        target,
        model_c=0.1,
    )
    second = completion._fit_fold(
        training,
        labels,
        calibration,
        calibration_labels,
        target,
        model_c=0.1,
    )

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["model_converged"]
    assert first[1][1] > first[1][0]
    assert len(json.loads(first[2]["polynomial_feature_names_json"])) == 35


def test_union_merges_new_pair_with_overlapping_v448_component() -> None:
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
                "audit": [{"source": "target_safe_v448"}],
                "source": "target_safe_v448",
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
            }
        ]
    }

    candidates, merged = completion._union_candidates(
        target_keys=[world],
        evidence_frames=evidence_frames,
        baseline=baseline,
        selected_pair_rows=selected,
        thresholds={0.1: 0.8, 0.3: 0.9},
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
        "target_safe_v448;v449_sparse_reporter_completion"
    )
    assert merged[(world, frozenset(("a", "b")))] == row[
        "completion_hyperedge_id"
    ]
    assert merged[(world, frozenset(("b", "c")))] == row[
        "completion_hyperedge_id"
    ]

