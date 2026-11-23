"""Empirical agreement and leakage boundaries, without fitting integration."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import causal_reliability_pilot_empirical as empirical


def fixture_data(n=10):
    ids = {a: np.asarray([f"{a}_{i}" for i in range(n)]) for a in empirical.ASSAYS}
    canonical = {a: {cell: f"c{(i + j) % n}" for i, cell in enumerate(cells)} for j, (a, cells) in enumerate(ids.items())}
    pairs = pd.DataFrame({"canonical_id": [f"c{i}" for i in range(n)]})
    for a in ids:
        inverse = {canon: cell for cell, canon in canonical[a].items()}
        pairs[a + "_cell_id"] = pairs.canonical_id.map(inverse)
    candidates = [{"name": f"{a}|g{i}", "assay": a, "role": "heldout_panel"} for a in ids for i in range(3)]
    return {"ids": ids, "canonical": canonical, "pairs": pairs,
            "blocks": {a: np.arange(n * 3).reshape(n, 3) + j * 100 for j, a in enumerate(ids)}, "candidates": candidates}


def test_empty_paired_graph_does_not_award_recovery_or_causal_truth():
    data = fixture_data()
    empty = {"status": "ok", "edges": []}
    score = empirical.graph_agreement(empty, empty, data["candidates"])
    cross = score["by_assay_pair"]["cross_assay"]
    assert cross["reference_coverage"] is None and cross["jaccard"] is None
    assert cross["prediction_overlap_fraction"] is None
    assert "causal truth" in score["reference"]
    failed = empirical.graph_agreement({"status": "failed", "edges": None}, empty, data["candidates"])
    assert failed["status"] == "unavailable" and "by_assay_pair" not in failed


def test_endpoint_agreement_is_separate_from_adjacency():
    data = fixture_data()
    a, b = data["candidates"][0]["name"], data["candidates"][3]["name"]
    reference = {"status": "ok", "edges": [{"node1": a, "node2": b, "endpoint1": "TAIL", "endpoint2": "ARROW"}]}
    reverse = {"status": "ok", "edges": [{"node1": b, "node2": a, "endpoint1": "TAIL", "endpoint2": "ARROW"}]}
    score = empirical.graph_agreement(reverse, reference, data["candidates"])["by_assay_pair"]["cross_assay"]
    assert score["overlap"] == 1 and score["jaccard"] == 1
    assert score["endpoint_agreement_on_overlap"] == 0


def test_canonical_join_and_assignment_preserve_raw_assay_blocks():
    data = fixture_data()
    assignment = empirical.paired_assignment(data["ids"]["RNA"], data)
    counts = empirical.assemble(assignment, data)
    for j, a in enumerate(empirical.ASSAYS):
        rows = empirical.ev._indices(assignment["cell_ids"][a], data["ids"][a])
        np.testing.assert_array_equal(counts[rows, j * 3:(j + 1) * 3], data["blocks"][a])
    pairs = empirical.retrieval(assignment, None, data)["assignment"]
    assert all(result["exact_pair_fraction"] == 1 for result in pairs.values())
    positional = {"cell_ids": data["ids"]}
    assert empirical.retrieval(positional, None, data)["assignment"]["RNA--H3K27ac"]["exact_pair_fraction"] == 0


def test_collapsed_embeddings_do_not_win_retrieval_by_tie_order():
    data = fixture_data()
    views = {a: {"cell_ids": ids, "shared": np.ones((len(ids), 2))} for a, ids in data["ids"].items()}
    assignment = empirical.paired_assignment(data["ids"]["RNA"], data)
    score = empirical.retrieval(assignment, views, data)
    assert all(r["mean_true_pair_rank"] == 10 and r["recall_at_1"] == 0 and r["recall_at_5"] == 0 for r in score["nearest_neighbor"].values())


def test_model_loader_uses_declared_test_cohort_and_rejects_wrong_source(monkeypatch):
    data = fixture_data()
    inputs = {f"{cohort}/{a}": {"path": f"{cohort}_{a}.h5ad", "sha256": "fixture"} for cohort in ("train", "test") for a in empirical.ASSAYS}
    provenance = {"training_inputs": {a: inputs[f"train/{a}"] for a in empirical.ASSAYS},
                  "encoding_inputs": {a: inputs[f"test/{a}"] for a in empirical.ASSAYS}, "selection_references": {}}
    views = {a: {"cell_ids": ids, "shared": np.ones((len(ids), 2))} for a, ids in data["ids"].items()}

    def loader(directory, seed):
        assert empirical.pilot.COHORT == "test"
        return views, provenance

    monkeypatch.setattr(empirical.pilot, "external_views", loader)
    monkeypatch.setattr(empirical, "table", lambda relative: pd.DataFrame({"cell_id": data["ids"][Path(relative).name.split("_test")[0]]}))
    empirical.method_views("fixture", 0, {"inputs": inputs}, "test")
    assert empirical.pilot.COHORT == "discovery"
    provenance["encoding_inputs"]["RNA"] = {"path": "wrong.h5ad", "sha256": "fixture"}
    with pytest.raises(ValueError, match="different empirical inputs"):
        empirical.method_views("fixture", 0, {"inputs": inputs}, "test")
    assert empirical.pilot.COHORT == "discovery"


def test_random_assignment_precedes_paired_read_and_failures_survive(tmp_path, monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(empirical, "ROOT", tmp_path)
    (tmp_path / "protocol.json").write_text("{}")
    monkeypatch.setattr(empirical, "context", lambda: ({}, data["ids"]))
    calls = []
    original = empirical.ev.match_one_to_one

    def match(*args, **kwargs):
        assert "paired" not in calls
        calls.append("match")
        return original(*args, **kwargs)

    def load(pools):
        assert calls == ["match"] * 4
        calls.append("paired")
        return data

    monkeypatch.setattr(empirical.ev, "match_one_to_one", match)
    monkeypatch.setattr(empirical, "load_evaluator", load)
    monkeypatch.setattr(empirical.ev, "discover", lambda *args, **kwargs: {"status": "failed", "edges": None})
    empirical.run_arm("random_reference")
    saved = json.loads((tmp_path / "arms/random_reference.json").read_text())
    assert saved["status"] == "complete_with_failures"
    assert saved["results"]["full"]["paired_graph_agreement"]["status"] == "unavailable"
    assert saved["count_subsample_stability"]["n_failed"] == 3
