from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_wasserstein_geometry_rescue_v4 as rescue  # noqa: E402


def _features(*values: float) -> dict[str, float]:
    return dict(zip(rescue.FEATURE_COLUMNS, values, strict=True))


def test_copula_score_respects_all_four_declared_directions() -> None:
    training = pd.DataFrame(
        [
            _features(0.0, 0.0, 3.0, 0.0),
            _features(1.0, 1.0, 2.0, 1.0),
            _features(2.0, 2.0, 1.0, 2.0),
        ]
    )
    values = pd.DataFrame(
        [
            _features(3.0, 3.0, 0.0, 3.0),
            _features(-1.0, -1.0, 4.0, -1.0),
        ]
    )

    scores = rescue._copula_score(training, values)

    assert scores[0] == pytest.approx((3.0 / 4.0) ** 4)
    assert scores[1] == pytest.approx(0.0)


def test_model_matrix_quantizes_machine_precision_noise() -> None:
    first = pd.DataFrame([_features(1.0, 2.0, 0.3, 4.0)])
    second = pd.DataFrame([_features(1.0, 2.0, 0.3 + 4e-16, 4.0)])

    assert np.array_equal(
        rescue._model_matrix(first), rescue._model_matrix(second)
    )


def test_bootstrap_seed_schedule_is_bound_to_predeclared_w2_variant() -> None:
    assert list(rescue.diagnostic.VARIANTS)[
        rescue.DIAGNOSTIC_VARIANT_INDEX
    ] == "w2_global_mirror"
    assert rescue.BOOTSTRAP_BASE_SEED == 454002


def test_baseline_component_counts_are_verified() -> None:
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


def test_union_keeps_connected_pairs_as_unexpanded_ambiguity() -> None:
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
                "audit": [{"source": "target_safe_v453"}],
                "source": "target_safe_v453",
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
                "bootstrap_selection_frequency": 0.9,
            }
        ]
    }

    candidates, merged = rescue._union_candidates(
        target_keys=[world],
        world_frames=world_frames,
        baseline=baseline,
        stable_rows=stable_rows,
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
        "wasserstein_geometry_hyperedge_id"
    ]
    assert merged[(world, frozenset(("b", "c")))] == row[
        "wasserstein_geometry_hyperedge_id"
    ]
