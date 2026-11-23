from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import compare_reporter_learning_rules_v4 as comparison


def _arrays():
    rng = np.random.default_rng(464)
    result = {}
    for modality in ("assay_a", "assay_b"):
        counts = rng.poisson(10, size=(120, 12)).astype(float)
        scores = rng.normal(size=(120, 3))
        result[modality] = {
            "counts": counts, "depth": counts.sum(axis=1),
            "reference": {context: scores.copy() for context in comparison.hard.CONTEXTS},
            "focal": np.expm1(3 + 0.2 * scores[:, [0]]),
            "reference_seed": 0, "config": comparison.hard.mdcr.UnpairedMDCRConfig(),
        }
    return result


def _mock_geometry(monkeypatch):
    calls = {"projections": 0}
    def project(values, **kwargs):
        calls["projections"] += 1
        return values
    monkeypatch.setattr(comparison.hard.joint, "_projection_quantiles", project)
    monkeypatch.setattr(comparison.hard.joint, "_precomputed_cost_matrix", lambda left, right: np.ones((3, 3)) - np.eye(3))
    return calls


def test_representation_evaluation_and_synchronization_are_shared_between_models(monkeypatch):
    arrays = _arrays()
    calls = _mock_geometry(monkeypatch)
    refits = []
    def fit(*args, modality, **kwargs):
        refits.append(modality)
        return SimpleNamespace(
            scores=arrays[modality]["reference"],
            contexts={context: SimpleNamespace(test=True) for context in comparison.hard.CONTEXTS},
            oob_mask=np.arange(120) < 60,
        )
    monkeypatch.setattr(comparison.hard, "_evaluation_indices", lambda *args: np.arange(60))
    monkeypatch.setattr(comparison.hard.representation, "bootstrap_fit_assay_contexts", fit)
    pairs, reporters = comparison.compute_array_comparison(
        "block__world_seed1", arrays, replicates=2, random_seed=463000, modes=("both",))
    assert len(refits) == 2 * 2  # One all-context fit per assay/replicate.
    assert calls["projections"] == 2 * 2 * 3  # Not multiplied by reporter models.
    assert len(pairs) == 2 * 3 * 2
    paired = pairs.groupby(["replicate", "context"])
    assert paired.shared_evaluation_sha256.nunique().eq(1).all()
    assert paired.shared_geometry_sha256.nunique().eq(1).all()
    assert reporters.groupby(["replicate", "context", "modality"]).evaluation_indices_sha256.nunique().eq(1).all()
    joint = reporters[reporters.reporter_model.eq(comparison.PRIMARY_MODEL)]
    assert joint.reporter_model_identified.all()
    assert joint.residual_score_rank.eq(3).all()
    assert joint.unique_axis_variance_json.notna().all()
    assert joint.joint_coefficients_json.notna().all()
    assert len(comparison.aggregate_comparison(pairs, 2)) == 2


def test_marginal_control_exactly_reproduces_existing_v463_evidence(monkeypatch):
    _mock_geometry(monkeypatch)
    arrays = _arrays()
    old, _ = comparison.soft.compute_array_soft_evidence(
        "block__world_seed1", arrays, replicates=2, random_seed=463000, modes=("cached_reference_subset",))
    new, _ = comparison.compute_array_comparison(
        "block__world_seed1", arrays, replicates=2, random_seed=463000, modes=("cached_reference_subset",))
    marginal = new[new.reporter_model.eq(comparison.REPORTER_MODELS[0])].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, marginal[old.columns], check_exact=True)


def test_failed_background_context_zeroes_both_models_without_subset_alignment(monkeypatch):
    arrays = _arrays()
    def fit(*args, modality, **kwargs):
        if modality == "assay_b":
            raise comparison.hard.representation.RepresentationFitError("failed background fit")
        return SimpleNamespace(scores=arrays[modality]["reference"],
            contexts={context: SimpleNamespace() for context in comparison.hard.CONTEXTS})
    monkeypatch.setattr(comparison.hard.representation, "bootstrap_fit_assay_contexts", fit)
    monkeypatch.setattr(comparison.hard.joint, "_projection_quantiles", lambda *args, **kwargs: pytest.fail("Incomplete context must not realign a subset"))
    pairs, reporters = comparison.compute_array_comparison(
        "block__world_seed1", arrays, replicates=2, random_seed=463000, modes=("initialization_only",))
    assert pairs[comparison.soft.PRIMARY_SCORE].eq(0).all()
    assert not pairs.complete_context.any()
    assert len(reporters[~reporters.fit_ok]) == 2 * 3 * 2
    aggregate = comparison.aggregate_comparison(pairs, 2)
    assert aggregate[comparison.PRIMARY_SCORE].eq(0).all()


def _context_table():
    rows = []
    values = [[0.2, -0.1, 0.3], [-0.1, 0.2, 0.3]]
    for model_index, model in enumerate(comparison.REPORTER_MODELS):
        for replicate in range(2):
            for context_index, context in enumerate(comparison.hard.CONTEXTS):
                rows.append({"world_key": "w", "block": "b", "left_modality": "a", "right_modality": "b",
                    "mode": "both", "reporter_model": model, "replicate": replicate, "context": context,
                    "expected_assay_count": 6, "complete_context": True, "fit_pair_available": True,
                    "eligible": True, "shared_evaluation_sha256": "same_rows", "shared_geometry_sha256": "same_fit",
                    **{metric: values[replicate][context_index] * (model_index + 1) for metric in comparison.soft.PAIR_METRICS}})
    return pd.DataFrame(rows)


def test_model_aggregation_is_separate_and_keeps_mean_then_min_order():
    aggregate = comparison.aggregate_comparison(_context_table(), 2).set_index("reporter_model")
    assert aggregate.loc[comparison.REPORTER_MODELS[0], comparison.PRIMARY_SCORE] == pytest.approx(0.05)
    assert aggregate.loc[comparison.REPORTER_MODELS[1], comparison.PRIMARY_SCORE] == pytest.approx(0.10)


def test_missing_model_or_different_shared_geometry_fails_closed():
    table = _context_table()
    with pytest.raises(ValueError, match="Missing or duplicate paired-model"):
        comparison.aggregate_comparison(table.iloc[1:], 2)
    table.loc[0, "shared_geometry_sha256"] = "other_fit"
    with pytest.raises(ValueError, match="do not share"):
        comparison.aggregate_comparison(table, 2)


def _selection_table():
    return pd.DataFrame([
        {"reporter_model": model, "mode": "both", "block": block,
         "expected_assay_count": assays, "truth_pair": truth, comparison.PRIMARY_SCORE: score}
        for model, rows in (
            (comparison.REPORTER_MODELS[0], [("target", 6, True, .12), ("train", 6, False, .10), ("train", 5, False, .90)]),
            (comparison.REPORTER_MODELS[1], [("target", 6, True, .12), ("train", 6, False, .20), ("train", 6, True, .99)]),
        ) for block, assays, truth, score in rows
    ])


def test_null_calibration_never_mixes_models_or_assay_counts():
    result = comparison.select_diagnostic(_selection_table())
    target = result[result.block.eq("target")].set_index("reporter_model")
    assert target.loc[comparison.REPORTER_MODELS[0], "selected_diagnostic_proposal"]
    assert not target.loc[comparison.REPORTER_MODELS[1], "selected_diagnostic_proposal"]
    assert target.loc[comparison.REPORTER_MODELS[0], "null_upper_envelope"] == .10
    assert target.loc[comparison.REPORTER_MODELS[1], "null_upper_envelope"] == .20


def test_target_labels_do_not_change_target_decision_and_missing_strata_abstain():
    table = _selection_table()
    first = comparison.select_diagnostic(table)
    table.loc[table.block.eq("target"), "truth_pair"] = False
    changed = comparison.select_diagnostic(table)
    pd.testing.assert_series_equal(first.loc[first.block.eq("target"), "selected_diagnostic_proposal"],
                                   changed.loc[changed.block.eq("target"), "selected_diagnostic_proposal"])
    table.loc[table.block.eq("target"), "expected_assay_count"] = 4
    result = comparison.select_diagnostic(table)
    assert not result.loc[result.block.eq("target"), "null_reference_available"].any()
    assert not result.loc[result.block.eq("target"), "selected_diagnostic_proposal"].any()


def test_changed_evidence_is_rejected_before_truth_access(tmp_path, monkeypatch):
    path = tmp_path / "evidence.tsv"
    path.write_text("tampered")
    manifest = {"primary_reporter_model": comparison.PRIMARY_MODEL, "primary_mode": comparison.PRIMARY_MODE,
                "primary_score": comparison.PRIMARY_SCORE, "information_boundary": {"truth_read": False},
                "source_sha256": {}, "output_sha256": {path.name: "incorrect_hash"}}
    monkeypatch.setattr(comparison.geometry, "_verified_manifest", lambda *args: manifest)
    monkeypatch.setattr(comparison.hard.depth, "_load_truth", lambda *args: pytest.fail("Truth accessed before evidence verification"))
    with pytest.raises(ValueError, match="Changed evidence"):
        comparison.score(formal_root=tmp_path, evidence_dir=tmp_path)
