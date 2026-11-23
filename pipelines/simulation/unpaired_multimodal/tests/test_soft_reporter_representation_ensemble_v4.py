from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import soft_reporter_representation_ensemble_v4 as soft


def _axis(values):
    a = np.asarray(values, dtype=float)
    strength = a.sum()
    weights = a / strength if strength else np.zeros(3)
    return soft.ReporterAxisEvidence(
        partial_correlations=np.sqrt(a)[None], per_reporter_weights=weights[None],
        mean_squared_correlations=a, weights=weights, total_strength=float(strength),
        information=0.0, weight_margin=0.0,
        minimum_reporter_best_squared_correlation=float(a.max()),
        nonzero_reporters=int(strength > 0), usable=bool(strength > 0),
    )


def test_uniform_is_neutral_and_discordance_keeps_its_negative_sign():
    uniform = _axis([0.2, 0.2, 0.2])
    concentrated = _axis([0.8, 0, 0])
    opposite = _axis([0, 0.5, 0])
    neutral = soft.soft_pair_evidence(uniform, concentrated, [0, 1, 2], [0, 1, 2])
    negative = soft.soft_pair_evidence(concentrated, opposite, [0, 1, 2], [0, 1, 2])
    same = soft.soft_pair_evidence(concentrated, opposite, [0, 1, 2], [1, 0, 2])
    assert neutral[soft.PRIMARY_SCORE] == pytest.approx(0.0, abs=1e-15)
    assert negative[soft.PRIMARY_SCORE] == pytest.approx(-0.8 * 0.5 / 3)
    assert negative["centered_concordance"] == pytest.approx(-1 / 3)
    assert same[soft.PRIMARY_SCORE] == pytest.approx(2 * 0.8 * 0.5 / 3)
    zero = soft.soft_pair_evidence(_axis([0, 0, 0]), concentrated, [0, 1, 2], [0, 1, 2])
    assert zero[soft.PRIMARY_SCORE] == 0.0
    assert zero["eligible"] is False


def test_pool_raw_squared_correlations_before_normalization_downweights_weak_reporter():
    rng = np.random.default_rng(80)
    centered = rng.normal(size=(200, 4))
    centered -= centered.mean(axis=0)
    q, _ = np.linalg.qr(centered)
    scores = q[:, :3]
    focal = np.expm1(3 + np.column_stack([q[:, 0], 0.01 * q[:, 2] + q[:, 3]]))
    result = soft.reporter_axis_evidence(focal, scores, np.ones(200))
    assert result.per_reporter_weights[0, 0] > 0.999
    assert result.per_reporter_weights[1, 2] > 0.999
    # Equal averaging after normalization would put about half the weight on
    # the extremely weak second reporter. Pooling raw evidence avoids that.
    assert result.weights[0] > 0.999
    np.testing.assert_allclose(result.mean_squared_correlations, np.square(result.partial_correlations).mean(axis=0))


def test_depth_adjustment_and_zero_reporter_abstention():
    rng = np.random.default_rng(9)
    depth = np.exp(rng.uniform(1, 3, 120))
    scores = rng.normal(size=(120, 3)) + np.log1p(depth)[:, None]
    focal = np.expm1(np.column_stack([1.0 + 0.2 * np.log1p(depth), np.zeros(120)]))
    result = soft.reporter_axis_evidence(focal, scores, depth)
    assert result.usable is False
    assert result.nonzero_reporters == 0
    assert result.information == 0.0
    np.testing.assert_array_equal(result.weights, np.zeros(3))


def test_score_sign_and_axis_permutation_invariance():
    rng = np.random.default_rng(64)
    scores = rng.normal(size=(150, 3))
    focal = np.expm1(3 + 0.2 * scores[:, [0]] + rng.normal(scale=0.1, size=(150, 1)))
    depth = np.exp(rng.normal(size=150))
    a = soft.reporter_axis_evidence(focal, scores, depth)
    order = [2, 0, 1]
    b = soft.reporter_axis_evidence(focal, scores[:, order] * [-1, 1, -1], depth)
    np.testing.assert_allclose(b.mean_squared_correlations, a.mean_squared_correlations[order])
    aligned = soft.soft_pair_evidence(a, b, [0, 1, 2], [1, 2, 0])
    same = soft.soft_pair_evidence(a, a, [0, 1, 2], [0, 1, 2])
    assert aligned[soft.PRIMARY_SCORE] == pytest.approx(same[soft.PRIMARY_SCORE])


def _arrays():
    rng = np.random.default_rng(71)
    arrays = {}
    for modality in ("assay_a", "assay_b"):
        counts = rng.poisson(10, size=(120, 12)).astype(float)
        scores = rng.normal(size=(120, 3))
        arrays[modality] = {
            "counts": counts, "depth": counts.sum(axis=1),
            "reference": {context: scores.copy() for context in soft.CONTEXTS},
            "focal": np.zeros((120, 1)) if modality == "assay_b" else np.expm1(4 + 0.2 * scores[:, [0]]),
            "reference_seed": 0, "config": soft.v462.mdcr.UnpairedMDCRConfig(),
        }
    return arrays


def test_degenerate_reporter_does_not_remove_assay_from_background_synchronization(monkeypatch):
    monkeypatch.setattr(soft.v462.joint, "_projection_quantiles", lambda values, **kwargs: values)
    monkeypatch.setattr(soft.v462.joint, "_precomputed_cost_matrix", lambda left, right: np.ones((3, 3)) - np.eye(3))
    pairs, fits = soft.compute_array_soft_evidence(
        "block__world_seed1", _arrays(), replicates=1, random_seed=80,
        modes=("cached_reference_subset",),
    )
    assert len(pairs) == 3
    assert pairs["synchronized_assay_count"].eq(2).all()
    assert pairs["complete_context"].all()
    assert pairs["fit_pair_available"].all()
    assert not pairs["eligible"].any()
    assert pairs[soft.PRIMARY_SCORE].eq(0).all()
    assert fits["fit_ok"].all()


def test_failed_assay_zeroes_whole_context_without_reoptimizing_subset(monkeypatch):
    arrays = _arrays()
    arrays["assay_c"] = arrays["assay_a"]

    def refit(*args, modality, **kwargs):
        if modality == "assay_b":
            raise soft.v462.representation.RepresentationFitError("test failure")
        class Result:
            scores = arrays[modality]["reference"]
            contexts = {name: type("Context", (), {})() for name in soft.CONTEXTS}
        return Result()

    monkeypatch.setattr(soft.v462.representation, "bootstrap_fit_assay_contexts", refit)
    monkeypatch.setattr(soft.v462.joint, "_projection_quantiles", lambda *args, **kwargs: pytest.fail("Incomplete assay set must not be synchronized"))
    pairs, _ = soft.compute_array_soft_evidence("block__world_seed1", arrays, replicates=1, random_seed=80, modes=("initialization_only",))
    assert len(pairs) == 9
    assert pairs["expected_assay_count"].eq(3).all()
    assert pairs["synchronized_assay_count"].eq(2).all()
    assert not pairs["complete_context"].any()
    assert pairs[soft.PRIMARY_SCORE].eq(0).all()


def test_aggregation_means_each_context_before_min_and_retains_failed_replicates():
    rows = []
    # Each replicate's minimum is -0.1, but the weakest context mean is 0.05.
    values = [[0.2, -0.1, 0.3], [-0.1, 0.2, 0.3]]
    for replicate in range(2):
        for index, context in enumerate(soft.CONTEXTS):
            rows.append({
                "world_key": "w", "block": "b", "left_modality": "a", "right_modality": "b", "mode": "both",
                "replicate": replicate, "context": context,
                "expected_assay_count": 6, "complete_context": True,
                "fit_pair_available": True, "eligible": True,
                **{metric: values[replicate][index] for metric in soft.PAIR_METRICS},
            })
    frame = pd.DataFrame(rows)
    row = soft.aggregate_soft_evidence(frame, 2).iloc[0]
    assert row["minimum_context_mean_weighted_concordance"] == pytest.approx(0.05)
    assert row[f"{soft.PRIMARY_SCORE}_mean_replicate_minimum_context"] == pytest.approx(-0.1)
    frame.loc[3, list(soft.PAIR_METRICS)] = 0.0
    frame.loc[3, ["complete_context", "fit_pair_available", "eligible"]] = False
    row = soft.aggregate_soft_evidence(frame, 2).iloc[0]
    assert row["complete_context_fraction"] == pytest.approx(5 / 6)
    assert row["mean_weighted_concordance"] == pytest.approx((0.2 - 0.1 + 0.3 + 0 + 0.2 + 0.3) / 6)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        soft.aggregate_soft_evidence(frame.iloc[:-1], 2)
