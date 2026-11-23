"""Tiny public-cohort fixtures; native fitting and checkpoint loading are mocked."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import scmrdr_adapter as paper

spec = importlib.util.spec_from_file_location("integration_validation_fit_unit", BASE / "scripts/integration_validation_fit.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    base = tmp_path / "benchmark"
    world = base / "runs/worlds/linear_chain_0"
    write(base / "runs/worlds/extended_bank_manifest.json",
          {"specifications": [{"world_id": "linear_chain_0", "split": "development"}]})
    write(world / "manifest.json", {"world_id": "linear_chain_0", "split": "development",
          "status": "complete", "regimes": ["informative"], "count_calibration_status": "passed_declared_checks"})
    primary = base / "runs/extended/linear_chain_0/informative/primary"
    observed = world / "observed/informative"
    prepared = primary / "scglue_preprocessing"
    for cohort in bridge.COHORTS:
        inputs = {}
        for assay in bridge.ASSAYS:
            path = observed / cohort / f"{assay}.h5ad"
            write(path, f"fake public counts {cohort} {assay}".encode())
            inputs[assay] = bridge._input_record(path)
        write(prepared / ("manifest.json" if cohort == "train" else f"cohorts/{cohort}/manifest.json"),
              {"status": "complete", "inputs": inputs})
    gene = {"config": asdict(paper.GeneConfig()), "train_limit": 2000, "excluded_features": [],
            "source_sha256": bridge.glue.file_hash(BASE / "gene_features.py"), "feature_names": ["g0", "g1"],
            "assays": {assay: {"source_sha256": bridge.glue.file_hash(observed / "train" / f"{assay}.h5ad")}
                       for assay in bridge.ASSAYS}}
    write(primary / "gene_transform/manifest.json", gene)
    validated = Mock()
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name:
                        paper if name == "scmrdr_adapter" else SimpleNamespace(_validate_saved_outputs=validated))
    fit_glue, fit_paper = Mock(side_effect=AssertionError("unexpected native fit")), Mock(side_effect=AssertionError("unexpected native fit"))
    monkeypatch.setattr(bridge.glue, "fit_prepared", fit_glue)
    monkeypatch.setattr(paper, "fit", fit_paper)
    # A truth or private-cohort read fails even if a future edit adds one.
    original_open = Path.open
    def public_only(path, *args, **kwargs):
        assert not {"truth", "private_inputs"} & set(path.parts)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", public_only)
    return SimpleNamespace(base=base, world=world, primary=primary, prepared=prepared,
                           observed=observed, out=tmp_path / "candidate", validated=validated,
                           fit_glue=fit_glue, fit_paper=fit_paper)


def fit(fixture, method="scglue", overrides=None, **kwargs):
    arguments = dict(base=fixture.base, world_id="linear_chain_0", regime="informative",
                     method=method, candidate="alignment_half", overrides=overrides or {},
                     device="cpu", output=fixture.out)
    arguments.update(kwargs)
    return bridge.fit_candidate(**arguments)


def test_scglue_forwards_complete_config_and_explicit_public_cohorts(fixture):
    def native(prepared, guidance, output, *, config, label, cohorts):
        assert prepared == fixture.prepared
        assert guidance == fixture.prepared / "guidance.graphml.gz"
        assert cohorts == ("train", "val", "discovery")
        assert config.lam_align == .025 and config.latent_dim == 12 and config.max_epochs == 120
        output.mkdir()
        write(output / "scglue_model.dill", b"fake native checkpoint")
        write(output / "model_manifest.json", {
            "checkpoint_sha256": bridge.glue.file_hash(output / "scglue_model.dill"),
            "inputs": {"transformations": {"fingerprint": "frozen"}, "package_provenance": {"fixture": True}},
            "training_history": {"pretrain": {"last_recorded_epoch": 2}}})
        for cohort in cohorts:
            for assay in bridge.ASSAYS:
                write(output / cohort / f"{assay}.npz", b"mock public export")
        return {"retention": {f"{cohort}/{assay}": {"n_cells": 3}
                              for cohort in cohorts for assay in bridge.ASSAYS}}
    fixture.fit_glue.side_effect = native
    result = fit(fixture, overrides={"lam_align": .025, "latent_dim": 12})
    assert result["status"] == "complete" and result["configuration"]["lam_align"] == .025
    assert result["overrides"] == {"lam_align": .025, "latent_dim": 12}
    assert fixture.validated.call_count == 3
    assert len(result["retention"]) == 9
    assert set(result["files"]) == {f"{cohort}/{assay}.npz" for cohort in bridge.COHORTS for assay in bridge.ASSAYS}
    assert bridge.read(fixture.out / "validation_fit.json") == result
    assert result["runner_source_sha256"] == bridge.glue.file_hash(Path(bridge.__file__))
    fixture.fit_paper.assert_not_called()


def test_scMRDR_core_config_and_existing_frozen_encoder_are_used(fixture, monkeypatch):
    calls, models = [], []
    def transform(path, assay, gene, limit=None):
        calls.append((path.parent.name, assay, limit))
        n = 4 if path.parent.name == "train" else 3
        return {"cell_ids": np.asarray([f"{path.parent.name}_{assay}_{i}" for i in range(n)]),
                "counts": np.ones((n, 2), dtype=int), "availability": np.ones(2, dtype=bool),
                "dropped_cell_ids": np.asarray([], dtype=str)}
    monkeypatch.setattr(paper, "transform_gene_features", transform)
    def native(counts, availability, features, output, *, config, validation_counts_by_modality):
        assert config.lambda_adv == 5 and config.latent_dim_shared == 7 and config.gamma == 10
        assert all(len(values) == 4 for values in counts.values())
        assert all(len(values) == 3 for values in validation_counts_by_modality.values())
        assert set(counts) == set(availability) == set(bridge.ASSAYS)
        assert features == ["g0", "g1"]
        write(output / "checkpoint.pt", b"fake paper checkpoint")
        write(output / "fit_summary.json", {"source": {"fixture": True}, "best_epoch": 2})
        model = SimpleNamespace(encode=lambda values, assay, names: {
            "shared": np.zeros((len(values), config.latent_dim_shared)),
            "specific": np.zeros((len(values), config.latent_dim_specific))})
        models.append(model)
        return model
    fixture.fit_paper.side_effect = native
    monkeypatch.setattr(paper.ScMRDRAdapter, "load", lambda path, device: models[0])
    result = fit(fixture, method="scmrdr", overrides={"lambda_adv": 5, "latent_dim_shared": 7})
    assert result["configuration"]["latent_dim_shared"] == 7 and len(result["retention"]) == 9
    assert result["training"]["best_epoch"] == 2
    assert bridge.read(fixture.out / "manifest.json")["config"]["lambda_adv"] == 5
    for cohort in bridge.COHORTS:
        for assay in bridge.ASSAYS:
            with np.load(fixture.out / cohort / f"{assay}.npz", allow_pickle=False) as values:
                assert values["shared"].shape[1] == 7 and values["private"].shape[1] == 20
    assert {cohort for cohort, _, _ in calls} == set(bridge.COHORTS)
    assert all(limit == 2000 for cohort, _, limit in calls if cohort == "train")
    assert all(limit == 1000 for cohort, _, limit in calls if cohort == "discovery")
    fixture.fit_glue.assert_not_called()


@pytest.mark.parametrize("overrides", [{"unknown": 1}, {"seed": 4}, {"device": "cuda"},
                         {"learning_rate": float("nan")}, {"lam_align": float("inf")},
                         {"learning_rate": 0}, {"lam_kl": -1}, {"dropout": 1},
                         {"h_dim": 0}, {"max_epochs": True}])
def test_invalid_overrides_rejected_before_output_or_fit(fixture, overrides):
    with pytest.raises(ValueError):
        fit(fixture, overrides=overrides)
    assert not fixture.out.exists()
    fixture.fit_glue.assert_not_called()
    fixture.fit_paper.assert_not_called()


@pytest.mark.parametrize("changed", [{"world_id": "../linear_chain_0"}, {"candidate": "../bad"},
                         {"method": "jianle"}, {"regime": "assessment"}, {"seed": -1}])
def test_invalid_scope_rejected(fixture, changed):
    with pytest.raises(ValueError):
        fit(fixture, **changed)
    fixture.fit_glue.assert_not_called()
    assert not fixture.out.exists()


def test_existing_output_and_reserved_world_are_rejected(fixture):
    fixture.out.mkdir()
    with pytest.raises(FileExistsError):
        fit(fixture)
    fixture.out.rmdir()
    path = fixture.world / "manifest.json"
    manifest = bridge.read(path)
    for key, value in (("split", "assessment"), ("count_calibration_status", "failed")):
        write(path, {**manifest, key: value})
        with pytest.raises(ValueError):
            fit(fixture)
    fixture.fit_glue.assert_not_called()


def test_changed_preprocessing_or_gene_inputs_are_rejected(fixture):
    path = fixture.observed / "train/RNA.h5ad"
    path.write_bytes(b"different public counts")
    for method in ("scglue", "scmrdr"):
        with pytest.raises(ValueError):
            fit(fixture, method=method)
    fixture.fit_glue.assert_not_called()
    fixture.fit_paper.assert_not_called()
    assert not fixture.out.exists()
