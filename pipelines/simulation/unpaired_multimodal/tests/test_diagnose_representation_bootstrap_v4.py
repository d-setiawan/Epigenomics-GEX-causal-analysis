from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_representation_bootstrap_v4 as diagnostic


def test_reporter_depth_adjustment_and_coordinate_relabeling():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(500, 3))
    log_depth = rng.uniform(3, 6, size=500)
    focal = np.expm1(3 + 0.2 * scores[:, [1]] + 0.3 * log_depth[:, None])
    observed_depth = np.expm1(log_depth)
    component, association, margin = diagnostic._reporter_assignment(focal, scores, observed_depth)
    assert component == 1
    assert association > 0.99
    assert margin > 0.8
    moved, _, _ = diagnostic._reporter_assignment(focal, -scores[:, [1, 0, 2]], observed_depth)
    assert moved == 0


def test_reporter_ties_and_disagreeing_alternatives_abstain():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(500, 3))
    depth = np.ones(500) * 50
    focal = np.expm1(4 + scores[:, [0, 1]] / 4)
    assert diagnostic._reporter_assignment(focal, scores, depth)[0] is None
    scores[:, 1] = scores[:, 0]
    assert diagnostic._reporter_assignment(focal[:, [0]], scores, depth)[0] is None


def _rows():
    return pd.DataFrame([
        {"world_key": "w", "block": "b", "left_modality": "a", "right_modality": "c",
         "mode": "both", "replicate": replicate, "context": context,
         "eligible": replicate == 0, "local_mutual": replicate == 0,
         "global_coassigned": replicate == 0}
        for replicate in range(2) for context in diagnostic.CONTEXTS
    ])


def test_failed_fit_remains_in_frequency_denominator():
    row = diagnostic._aggregate_replicates(_rows(), 2).iloc[0]
    assert row["global_coassigned_all_context_frequency"] == 0.5
    assert row["global_coassigned_all_context_count"] == 1


def test_missing_or_duplicate_context_is_rejected():
    rows = _rows()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        diagnostic._aggregate_replicates(rows.iloc[:-1], 2)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        diagnostic._aggregate_replicates(pd.concat([rows, rows.iloc[[0]]]), 2)


def test_evaluation_subset_is_repeatable_and_assay_specific():
    first = diagnostic._evaluation_indices(500, "assay_a", 7)
    assert np.array_equal(first, diagnostic._evaluation_indices(500, "assay_a", 7))
    assert not np.array_equal(first, diagnostic._evaluation_indices(500, "assay_b", 7))
    _, expected = diagnostic.representation._resample_indices(
        500, modality="assay_a", resample_seed=7, mode="both")
    assert np.array_equal(first, np.flatnonzero(expected))
