"""Small boundary and failure-semantics checks; no integration fitting or GPU."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import causal_reliability_pilot as pilot


def test_independent_subsamples_keep_fixed_sizes_and_reproduce():
    pools = {a: np.asarray([f"{a}_{i:03d}" for i in range(100)]) for a in pilot.ASSAYS}
    first, repeat = pilot.subsample(pools, 7000), pilot.subsample(pools, 7000)
    for a in pools:
        np.testing.assert_array_equal(first[a], repeat[a])
        assert len(first[a]) == len(set(first[a])) == 80
        assert set(first[a]) <= set(pools[a])
    assert {x.rsplit("_", 1)[-1] for x in first["RNA"]} != {x.rsplit("_", 1)[-1] for x in first["H3K27ac"]}


def test_missing_pool_cells_are_not_silently_removed():
    pools = {a: np.asarray([f"{a}_0", f"{a}_1"]) for a in pilot.ASSAYS}
    views = {a: {"cell_ids": ids[:1], "shared": np.ones((1, 2))} for a, ids in pools.items()}
    with pytest.raises(ValueError, match="do not shrink"):
        pilot.require_pool_coverage(views, pools)


def test_failed_discovery_is_not_an_empty_graph_or_stable_absence():
    edge = {"node1": "a", "node2": "b", "endpoint1": "CIRCLE", "endpoint2": "ARROW"}
    result = pilot.edge_stability([{"status": "ok", "edges": [edge]}, {"status": "ok", "edges": []},
                                   {"status": "failed", "edges": None}])
    assert result["n_failed"] == 1 and result["n_successful"] == 2
    assert result["adjacencies"][0]["frequency"] == 0.5
    failed = pilot.edge_stability([{"status": "failed", "edges": None}])
    assert failed["status"] == "unavailable" and failed["n_successful"] == 0


def test_private_world_rejected_before_any_read(monkeypatch):
    monkeypatch.setattr(pilot, "read", lambda _: pytest.fail("must reject before file access"))
    with pytest.raises(ValueError, match="predeclared"):
        pilot.context("linear_chain_1")


def test_external_export_seals_and_extra_truth_fields(tmp_path):
    for name in ("checkpoint.pt", "preprocessing.json", "input.h5ad", "adapter.py"):
        (tmp_path / name).write_bytes(b"fixture")
    source = pilot.pin(tmp_path / "input.h5ad")
    model = {"status": "complete", "method": "scconfluence", "configuration": {"seed": 0},
             "checkpoint_sha256": pilot.digest(tmp_path / "checkpoint.pt"),
             "preprocessing_sha256": pilot.digest(tmp_path / "preprocessing.json"),
             "inputs": {a: source for a in pilot.ASSAYS},
             "source_files": {str(tmp_path / "adapter.py"): pilot.digest(tmp_path / "adapter.py")}}
    (tmp_path / "model_manifest.json").write_text(json.dumps(model))
    cohort = tmp_path / "discovery"
    cohort.mkdir()
    encoding = {"status": "complete", "cohort": "discovery", "checkpoint_sha256": model["checkpoint_sha256"],
                "preprocessing_sha256": model["preprocessing_sha256"], "inputs": model["inputs"], "exports": {}}
    for a in pilot.ASSAYS:
        ids = np.asarray([f"{a}_0", f"{a}_1"])
        path = cohort / f"{a}.npz"
        np.savez(path, cell_ids=ids, shared=np.ones((2, 3)))
        encoding["exports"][a] = {**pilot.pin(path), "n_cells": 2, "latent_dim": 3,
                                  "cell_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "dropped_cell_ids": []}
    enc_path = cohort / "encoding_manifest.json"
    enc_path.write_text(json.dumps(encoding))
    arrays, _ = pilot.external_views(tmp_path, 0)
    assert arrays["RNA"]["shared"].shape == (2, 3)
    with pytest.raises(ValueError, match="seed/discovery"):
        pilot.external_views(tmp_path, 1)
    np.savez(cohort / "RNA.npz", cell_ids=arrays["RNA"]["cell_ids"], shared=np.ones((2, 3)), state=np.ones((2, 1)))
    encoding["exports"]["RNA"]["sha256"] = pilot.digest(cohort / "RNA.npz")
    enc_path.write_text(json.dumps(encoding))
    with pytest.raises(ValueError, match="only opaque IDs"):
        pilot.external_views(tmp_path, 0)


def test_learned_matching_precedes_truth_and_failures_survive(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    (tmp_path / "protocol.json").write_text("{}")
    pools = {a: np.asarray([f"{a}_{i}" for i in range(4)]) for a in pilot.ASSAYS}
    ids = np.concatenate(list(pools.values()))
    names = np.asarray([f"{a}_{r}" for a in pilot.ASSAYS for r in ("core", "decoy", "sentinel")])
    assays = np.repeat(pilot.ASSAYS, 3)
    candidates = [{"name": n, "assay": a, "role": n.rsplit("_", 1)[-1]} for n, a in zip(names, assays)]
    manifest_path = tmp_path / "truth_manifest.json"
    manifest_path.write_text(json.dumps({"candidates": candidates, "biological_dag_edges": []}))
    truth = {"cell_ids": ids, "assigned_assay": np.repeat(pilot.ASSAYS, 4),
             "state": np.arange(12)[:, None], "activities": np.arange(108).reshape(12, 9),
             "exposures": np.ones((12, 3)), "candidate_names": names, "candidate_assays": assays,
             "candidate_roles": np.tile(["core", "decoy", "sentinel"], 3), "candidate_feature_ids": names,
             "counts_informative": np.arange(108).reshape(12, 9)}
    record = {"public_truth": {"path": "discovery_only.npz", "sha256": "fixture"},
              "truth_manifest": {"path": str(manifest_path)}}
    monkeypatch.setattr(pilot, "context", lambda _: ({}, record, pools))
    monkeypatch.setattr(pilot.previous, "prepare", lambda: ({}, None))
    views = {a: {"cell_ids": v, "shared": np.arange(4)[:, None]} for a, v in pools.items()}
    monkeypatch.setattr(pilot.experiment, "load_fit", lambda *args: (views, {}))
    calls = []
    original_match = pilot.ev.match_one_to_one

    def match(*args, **kwargs):
        assert "truth" not in calls
        calls.append("match")
        return original_match(*args, **kwargs)

    def load(path):
        assert str(path) == "discovery_only.npz" and calls == ["match"] * 4
        calls.append("truth")
        return truth

    monkeypatch.setattr(pilot.ev, "match_one_to_one", match)
    monkeypatch.setattr(pilot.ev, "_load_npz", load)
    monkeypatch.setattr(pilot.ev, "oracle_graph", lambda *args, **kwargs: {"status": "ok", "names": list(args[1]), "edges": []})
    monkeypatch.setattr(pilot.ev, "discover", lambda *args, **kwargs: {"status": "failed", "edges": None})
    pilot.run_arm("linear_chain_0", "glue_compressed")
    saved = json.loads((tmp_path / "linear_chain_0/arms/glue_compressed.json").read_text())
    assert saved["status"] == "complete_with_failures"
    assert saved["count_subsample_stability"]["n_failed"] == 3
    assert saved["results"]["full"]["scales"]["counts"]["metrics"]["status"] == "not_applicable"
