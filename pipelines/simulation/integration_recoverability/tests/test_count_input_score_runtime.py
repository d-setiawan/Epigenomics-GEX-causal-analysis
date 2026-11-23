"""Analysis-only runtime correction on dummy sealed fits; no CUDA or data reads."""

import ast
from copy import deepcopy
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
sys.path[:0] = [str(BASE / "scripts"), str(BASE)]
SPEC = importlib.util.spec_from_file_location("count_input_runtime_score_unit", BASE / "scripts/score_count_input_followup.py")
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    count = scoring.experiment
    monkeypatch.setattr(count, "ROOT", tmp_path)
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(key, raising=False)
    source = str(Path(count.__file__).resolve())
    spec = {"configuration": asdict(count.glue.ScGLUEConfig(lam_align=.2)),
            "analysis_sources": {source: count.glue.file_hash(Path(source))},
            "runtime_flags": dict(count.RUNTIME_FLAGS),
            "inherited_loader_environment": {"CUDA_VISIBLE_DEVICES": None, "LD_LIBRARY_PATH": None}}
    protocol = tmp_path / "protocol.json"
    write(protocol, spec)
    monkeypatch.setattr(scoring, "PROTOCOL_SHA256", count.glue.file_hash(protocol))
    checked = Mock(side_effect=lambda: deepcopy(spec))
    monkeypatch.setattr(count, "checked_protocol", checked)
    directories = {}
    for mode in count.MODES:
        directory = tmp_path / "fits" / count.WORLDS[0] / mode
        directories[mode] = directory
        write(directory / "scglue_model.dill", b"opaque fake checkpoint, never deserialized")
        checkpoint = count.glue.file_hash(directory / "scglue_model.dill")
        model = {"status": "complete", "configuration": spec["configuration"], "checkpoint_sha256": checkpoint,
                 "training_audit": {"encoder_mode": mode}, "inputs": {"package_provenance": {
                     "input_ablation": {"mode": mode, "source": source, "source_sha256": spec["analysis_sources"][source]}}}}
        write(directory / "model_manifest.json", model)
        files = {}
        for cohort in count.fitting.COHORTS:
            for assay in count.ASSAYS:
                relative = f"{cohort}/{assay}.npz"
                write(directory / relative, b"opaque fake export, never decoded")
                files[relative] = count.glue.file_hash(directory / relative)
        seal = {"status": "complete", "checkpoint_sha256": checkpoint, "files": files,
                "fit_manifest": {"sha256": count.glue.file_hash(directory / "model_manifest.json")}}
        write(directory / "validation_fit.json", seal)
        write(directory / "experiment.json", {
            "status": "complete", "world": count.WORLDS[0], "mode": mode,
            "protocol_sha256": scoring.PROTOCOL_SHA256,
            "validation_fit_sha256": count.glue.file_hash(directory / "validation_fit.json"),
            "runtime_environment": {**spec["runtime_flags"], **spec["inherited_loader_environment"], "CUDA_VISIBLE_DEVICES": "0"}})
    loader = Mock(return_value=(None, {"fixture": True}, {"files": {}}))
    monkeypatch.setattr(count.validation.evaluation, "_load_embedding_run", loader)
    evaluate = Mock(return_value={"status": "complete", "results": {
        mode: {"worst_pair_gap": 0} for mode in count.MODES}})
    monkeypatch.setattr(count, "evaluate_views", evaluate)
    return SimpleNamespace(root=tmp_path, spec=spec, checked=checked, loader=loader,
                           evaluate=evaluate, directories=directories)


def test_native_autodevice_sets_gpu_mask_without_calling_cuda():
    source = scoring.AUTODEVICE_SOURCE.read_text()
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == "autodevice")
    node.decorator_list = []
    environment = {}
    namespace = {"os": SimpleNamespace(environ=environment), "np": np,
                 "config": SimpleNamespace(CPU_ONLY=False, MASKED_GPUS=[]),
                 "torch": SimpleNamespace(device=lambda name: name),
                 "pynvml": SimpleNamespace(NVMLError=RuntimeError, nvmlInit=lambda: None,
                     nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: i,
                     nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(free=100))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(scoring.AUTODEVICE_SOURCE), "exec"), namespace)
    namespace["autodevice"].logger = SimpleNamespace(info=Mock())
    assert namespace["autodevice"]() == "cuda"
    assert environment == {"CUDA_VISIBLE_DEVICES": "0"}


def test_plan_is_immutable_deepcopy_and_never_reads_embeddings_or_scores(fixture):
    before = {path: path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()}
    original = deepcopy(fixture.spec)
    expected, amendment = scoring.prepare()
    assert fixture.spec == original
    assert expected == {**original, "inherited_loader_environment": {
        **original["inherited_loader_environment"], "CUDA_VISIBLE_DEVICES": "0"}}
    record = json.loads(amendment.read_text())
    assert record["original_protocol"]["sha256"] == scoring.PROTOCOL_SHA256
    assert record["original_runner"]["sha256"] == scoring.RUNNER_SHA256
    assert record["native_autodevice"]["sha256"] == scoring.AUTODEVICE_SHA256
    assert record["change"] == {"field": "inherited_loader_environment.CUDA_VISIBLE_DEVICES", "from": None, "to": "0"}
    assert scoring.prepare()[1].read_bytes() == amendment.read_bytes()
    assert all(path.read_bytes() == value for path, value in before.items())
    fixture.loader.assert_not_called()
    fixture.evaluate.assert_not_called()
    record["change"]["to"] = "1"
    write(amendment, record)
    with pytest.raises(FileExistsError):
        scoring.prepare()


@pytest.mark.parametrize("value", ["0", "1", ""])
def test_only_initially_unset_gpu_mask_is_eligible(fixture, monkeypatch, value):
    fixture.spec["inherited_loader_environment"]["CUDA_VISIBLE_DEVICES"] = value
    write(fixture.root / "protocol.json", fixture.spec)
    monkeypatch.setattr(scoring, "PROTOCOL_SHA256", scoring.experiment.glue.file_hash(fixture.root / "protocol.json"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with pytest.raises(ValueError, match="null-to-0"):
        scoring.prepare()
    fixture.checked.assert_not_called()
    assert not (fixture.root / "runtime_scoring_amendment.json").exists()


def test_original_initial_environment_required_before_checked_protocol(fixture, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="original initial loader environment"):
        scoring.prepare()
    fixture.checked.assert_not_called()


@pytest.mark.parametrize("key,value", [("CUDA_VISIBLE_DEVICES", "1"), ("LD_LIBRARY_PATH", "/different"),
                                      ("CUDA_LAUNCH_BLOCKING", "0")])
def test_observed_environment_may_only_differ_by_null_to_zero(fixture, key, value):
    path = fixture.directories["compressed"] / "experiment.json"
    receipt = json.loads(path.read_text())
    receipt["runtime_environment"][key] = value
    write(path, receipt)
    with pytest.raises(ValueError, match="exact amendment"):
        scoring.prepare()
    assert not (fixture.root / "runtime_scoring_amendment.json").exists()


def test_corrected_scorer_keeps_original_fit_guards_and_reports(fixture):
    legacy = fixture.root / "reports" / "linear_chain_0.json"
    write(legacy, {"original_evaluator_failure": "retained"})
    before = legacy.read_bytes()
    def check_amendment(*args):
        assert (fixture.root / "runtime_scoring_amendment.json").exists()
        return None, {"fixture": True}, {"files": {}}
    fixture.loader.side_effect = check_amendment
    result = scoring.evaluate("linear_chain_0")
    assert fixture.loader.call_count == 4 and fixture.evaluate.call_count == 1
    assert Path(result["report"]).parent == fixture.root / "reports_runtime_corrected"
    report = json.loads(Path(result["report"]).read_text())
    assert report["protocol_sha256"] == scoring.PROTOCOL_SHA256 and report["runtime_scoring_amendment"]["sha256"]
    assert report["scoring_source_sha256"] == scoring.experiment.glue.file_hash(Path(scoring.__file__))
    assert legacy.read_bytes() == before
    receipt_path = fixture.directories["native_counts"] / "experiment.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["runtime_environment"]["CUDA_LAUNCH_BLOCKING"] = "0"
    write(receipt_path, receipt)
    fixture.evaluate.reset_mock()
    with pytest.raises(ValueError, match="sealed artifact"):
        scoring.evaluate("linear_chain_0")
    fixture.evaluate.assert_not_called()


def test_changed_native_source_or_unplanned_world_stops_before_scores(fixture, monkeypatch):
    native = fixture.root / "changed_nn.py"
    write(native, b"changed native source")
    monkeypatch.setattr(scoring, "AUTODEVICE_SOURCE", native)
    with pytest.raises(ValueError, match="autodevice source changed"):
        scoring.prepare()
    with pytest.raises(ValueError, match="Unplanned development world"):
        scoring.evaluate("constrained_chain_1")
    fixture.loader.assert_not_called()
    fixture.evaluate.assert_not_called()
