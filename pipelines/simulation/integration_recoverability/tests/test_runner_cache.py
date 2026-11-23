"""Fail-closed cache contracts; tiny synthetic artifacts, no model training."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")
pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_benchmark as runner
import scmrdr_adapter as wrapper
from gene_features import GeneConfig


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def fixture_run(tmp_path, monkeypatch, *, external=False):
    base, repo = tmp_path / "base", tmp_path / "repo"
    base.mkdir()
    (base / "scmrdr_adapter.py").write_text("wrapper fixture\n")
    core = repo / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
    (core / "upstream").mkdir(parents=True)
    (core / "adapter.py").write_text("core fixture\n")
    for name in ("model.py", "loss.py", "LICENSE"):
        (core / "upstream" / name).write_text(name + " fixture\n")
    monkeypatch.setattr(runner, "BASE", base)
    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(runner, "_scmrdr_versions", lambda: {"fixture_version": "1"})
    run, transform = tmp_path / "run", tmp_path / "gene_transform"
    (run / "model").mkdir(parents=True)
    (run / "model/checkpoint.pt").write_bytes(b"unit test checkpoint, not a trained model")
    sources, inputs = {}, {}
    for cohort in ("train", "val", "discovery", "probe", "assessment"):
        source = tmp_path / "inputs" / cohort
        source.mkdir(parents=True)
        sources[cohort], inputs[cohort] = source, {}
        for assay in runner.ASSAYS:
            ids = [f"{cohort}_{assay}_{i}" for i in range(4)]
            data = ad.AnnData(sparse.csr_matrix(np.asarray([[0, 0], [1, 2], [2, 1], [1, 1]], dtype=np.float32)),
                             obs=pd.DataFrame(index=ids),
                             var=pd.DataFrame({"gene_id": ["g0", "g1"]}, index=["f0", "f1"]))
            path = source / f"{assay}.h5ad"
            data.write_h5ad(path)
            inputs[cohort][assay] = {"path": str(path), "sha256": runner.digest(path),
                                     "retained_cells": 3, "dropped_cell_ids": ids[:1]}
            (run / cohort).mkdir(exist_ok=True)
            np.savez_compressed(run / cohort / f"{assay}.npz", cell_ids=np.asarray(ids[1:]),
                                shared=np.zeros((3, 20), dtype=np.float32),
                                private=np.zeros((3, 20), dtype=np.float32))
    gene = {"config": asdict(GeneConfig()), "train_limit": 2000, "excluded_features": [],
            "feature_names": ["g0", "g1"],
            "source_sha256": runner.digest(Path(wrapper.__file__).with_name("gene_features.py")),
            "assays": {a: {"source_sha256": inputs["train"][a]["sha256"]} for a in runner.ASSAYS}}
    write_json(transform / "manifest.json", gene)
    config = json.loads(json.dumps(asdict(wrapper.SCmRDRConfig(seed=0, device="cpu"))))
    hashes = runner._scmrdr_source_hashes()
    manifest = {"schema_version": 1, "status": "complete", "method": "scmrdr_paper_faithful",
                "seed": 0, "config": config, "train_limit": 2000,
                "train_inputs": inputs["train"], "validation_inputs": inputs["val"] if external else None,
                "gene_transform_dir": str(transform), "gene_manifest_sha256": runner.digest(transform / "manifest.json"),
                "feature_count": 2, "checkpoint_sha256": runner.digest(run / "model/checkpoint.pt"),
                "adapter_source_sha256": hashes["wrapper"], "versions": runner._scmrdr_versions()}
    summary = {"method": "scmrdr_paper_faithful", "config": config, "n_features": 2,
               "modalities": list(runner.ASSAYS),
               "training_rows": {a: 3 if external else 2 for a in runner.ASSAYS},
               "validation_rows": {a: 3 if external else 1 for a in runner.ASSAYS},
               "source": {"adapter_sha256": hashes["core"], "vendored_sha256": hashes["vendored"]}}
    write_json(run / "manifest.json", manifest)
    write_json(run / "model/fit_summary.json", summary)
    for cohort in ("val", "discovery", "probe", "assessment"):
        write_json(run / cohort / "encoding_manifest.json", {
            "cohort": cohort, "limit": 1000 if cohort == "discovery" else None,
            "post_fit_only": True, "inputs": inputs[cohort],
            "checkpoint_sha256": manifest["checkpoint_sha256"]})
    return {"run": run, "transform": transform, "sources": sources, "manifest": manifest,
            "summary": summary, "core": core, "external": external}


def check_reuse(fixture, **overrides):
    options = {"validation_dir": fixture["sources"]["val"] if fixture["external"] else None,
               "device": "cpu"}
    options.update(overrides)
    return runner._reuse_scmrdr(fixture["run"], 0, fixture["sources"]["train"],
                                fixture["transform"], [], **options)


@pytest.mark.parametrize("external", [False, True])
def test_identical_internal_or_external_fit_is_reusable(tmp_path, monkeypatch, external):
    fixture = fixture_run(tmp_path, monkeypatch, external=external)
    assert check_reuse(fixture)["status"] == "complete"


@pytest.mark.parametrize("changed", ["train_hash", "limit", "gene_link", "wrapper", "core", "model", "loss",
                                    "summary_config", "summary_features", "summary_rows", "versions", "checkpoint"])
def test_fit_cache_rejects_changed_identity(tmp_path, monkeypatch, changed):
    f = fixture_run(tmp_path, monkeypatch)
    manifest, summary = f["manifest"], f["summary"]
    if changed == "train_hash":
        manifest["train_inputs"]["RNA"]["sha256"] = "0" * 64
    elif changed == "limit":
        manifest["train_limit"] = 1999
    elif changed == "gene_link":
        manifest["gene_manifest_sha256"] = "0" * 64
    elif changed == "wrapper":
        (runner.BASE / "scmrdr_adapter.py").write_text("changed wrapper")
    elif changed == "core":
        (f["core"] / "adapter.py").write_text("changed core")
    elif changed in {"model", "loss"}:
        (f["core"] / "upstream" / f"{changed}.py").write_text("changed native code")
    elif changed == "summary_config":
        summary["config"] = {**summary["config"], "gamma": 99}
    elif changed == "summary_features":
        summary["n_features"] = 3
    elif changed == "summary_rows":
        summary["training_rows"]["RNA"] = 3
    elif changed == "versions":
        manifest["versions"] = {"fixture_version": "changed"}
    else:
        (f["run"] / "model/checkpoint.pt").write_bytes(b"different checkpoint")
    write_json(f["run"] / "manifest.json", manifest)
    write_json(f["run"] / "model/fit_summary.json", summary)
    with pytest.raises(ValueError):
        check_reuse(f)


def test_changed_validation_mode_or_source_cannot_reuse_fit(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch, external=True)
    with pytest.raises(ValueError, match="validation mode"):
        check_reuse(f, validation_dir=None)
    f["manifest"]["validation_inputs"]["RNA"]["sha256"] = "0" * 64
    write_json(f["run"] / "manifest.json", f["manifest"])
    with pytest.raises(ValueError, match="external validation"):
        check_reuse(f)


def test_internal_fit_cannot_be_mistaken_for_external_validation(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="validation"):
        check_reuse(f, validation_dir=f["sources"]["val"])


def test_existing_unsealed_outputs_fail_closed_except_explicit_simple_train(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Missing immutable"):
        runner._verify_scmrdr_encoding(f["run"], "discovery", f["sources"]["discovery"], 1000)
    with pytest.raises(ValueError, match="Missing immutable"):
        runner._verify_scmrdr_encoding(f["run"], "train", f["sources"]["train"], 2000)
    receipt = runner._verify_scmrdr_encoding(f["run"], "train", f["sources"]["train"], 2000,
                                           allow_preexisting_simple_train=True)
    assert receipt["verification_origin"] == "preexisting_simple_train"
    assert "not the unrecorded interval" in receipt["integrity_scope"]
    assert receipt["verified_at_utc"]
    # Identical reuse must not rewrite the first-verification timestamp.
    assert runner._verify_scmrdr_encoding(f["run"], "train", f["sources"]["train"], 2000) == receipt


@pytest.mark.parametrize("changed", ["input_hash", "limit", "checkpoint", "post_fit", "cohort"])
def test_encoding_context_rejects_wrong_inputs_or_checkpoint(tmp_path, monkeypatch, changed):
    f = fixture_run(tmp_path, monkeypatch)
    path = f["run"] / "discovery/encoding_manifest.json"
    audit = runner.read(path)
    if changed == "input_hash":
        audit["inputs"]["RNA"]["sha256"] = "0" * 64
    elif changed == "limit":
        audit["limit"] = None
    elif changed == "checkpoint":
        audit["checkpoint_sha256"] = "0" * 64
    elif changed == "post_fit":
        audit["post_fit_only"] = False
    else:
        audit["cohort"] = "assessment"
    write_json(path, audit)
    with pytest.raises(ValueError):
        runner._verify_scmrdr_encoding(f["run"], "discovery", f["sources"]["discovery"], 1000,
                                        newly_exported=True)


def test_modified_embedding_cannot_be_resealed(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch)
    runner._verify_scmrdr_encoding(f["run"], "discovery", f["sources"]["discovery"], 1000, newly_exported=True)
    path = f["run"] / "discovery/RNA.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["shared"] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="receipt changed"):
        runner._verify_scmrdr_encoding(f["run"], "discovery", f["sources"]["discovery"], 1000,
                                        newly_exported=True)


def test_npz_cell_id_substitution_is_rejected_before_first_receipt(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch)
    path = f["run"] / "train/RNA.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["cell_ids"] = arrays["cell_ids"][::-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="exported IDs"):
        runner._verify_scmrdr_encoding(f["run"], "train", f["sources"]["train"], 2000,
                                        allow_preexisting_simple_train=True)


def test_public_scmrdr_manifest_covers_all_and_only_public_exports(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch, external=True)
    for cohort in ("train", "val", "discovery", "probe", "assessment"):
        limit = 2000 if cohort == "train" else 1000 if cohort == "discovery" else None
        runner._verify_scmrdr_encoding(f["run"], cohort, f["sources"][cohort], limit, newly_exported=True)
    receipt = runner._public_embedding_manifest(f["run"], "scmrdr", ("train", "val", "discovery"))
    expected = {f"{cohort}/{assay}.npz" for cohort in ("train", "val", "discovery") for assay in runner.ASSAYS}
    assert set(receipt["files"]) == expected
    assert receipt["checkpoint_sha256"] == f["manifest"]["checkpoint_sha256"]
    assert receipt["status"] == "complete"
    assert runner._public_embedding_manifest(f["run"], "scmrdr", ("train", "val", "discovery")) == receipt
    with pytest.raises(ValueError, match="Public embedding manifest"):
        runner._public_embedding_manifest(f["run"], "scmrdr", ("train",))


def test_public_manifest_rejects_missing_public_cohort_receipt(tmp_path, monkeypatch):
    f = fixture_run(tmp_path, monkeypatch)
    runner._verify_scmrdr_encoding(f["run"], "train", f["sources"]["train"], 2000, newly_exported=True)
    with pytest.raises(FileNotFoundError):
        runner._public_embedding_manifest(f["run"], "scmrdr", ("train", "val", "discovery"))
    assert not (f["run"] / "public_embedding_manifest.json").exists()


def test_scglue_public_receipt_uses_native_export_hashes(tmp_path):
    run = tmp_path / "glue"
    run.mkdir()
    (run / "scglue_model.dill").write_bytes(b"checkpoint fixture")
    checkpoint = runner.digest(run / "scglue_model.dill")
    write_json(run / "model_manifest.json", {"status": "complete", "checkpoint_sha256": checkpoint})
    files = {}
    for cohort in ("train", "val", "discovery", "probe"):
        (run / cohort).mkdir()
        for assay in runner.ASSAYS:
            path = run / cohort / f"{assay}.npz"
            path.write_bytes(b"export fixture")
            files[str(path.relative_to(run))] = runner.digest(path)
    write_json(run / "embedding_manifest.json", {
        "status": "complete", "checkpoint_sha256": checkpoint, "files": files,
        "inputs": {"model_manifest_sha256": runner.digest(run / "model_manifest.json")}})
    receipt = runner._public_embedding_manifest(run, "scglue", ("train", "val", "discovery"))
    assert len(receipt["files"]) == 9
    assert not any(name.startswith("probe/") for name in receipt["files"])
    (run / "train/RNA.npz").write_bytes(b"changed export fixture")
    with pytest.raises(ValueError, match="public embedding"):
        runner._public_embedding_manifest(run, "scglue", ("train", "val", "discovery"))
