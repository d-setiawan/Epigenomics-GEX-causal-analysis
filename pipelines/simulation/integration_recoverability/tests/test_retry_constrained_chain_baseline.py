"""Only temporary dummy files and tiny CPU subprocesses; no scientific datasets."""

import importlib.util
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/retry_constrained_chain_baseline.py"
spec = importlib.util.spec_from_file_location("constrained_chain_retry_test", SCRIPT)
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)
suite = retry.suite

WORKER = '''import hashlib, json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
mode = sys.argv[2]
print(json.dumps({"mode": mode}), flush=True)
if mode == "fail":
    sys.exit(7)
if mode == "missing":
    sys.exit(0)
root = pathlib.Path(payload["root"])
root.mkdir(parents=True)
(root / "fit_inputs.json").write_text(payload["fit_inputs"] + (" " if mode == "changed_inputs" else ""))
checkpoint = root / "scglue_model.dill"
checkpoint.write_bytes(b"tiny fixture weights")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
model = {"status": "complete", "checkpoint_sha256": digest(checkpoint)}
(root / "model_manifest.json").write_text(json.dumps(model))
files = {}
for cohort in ("train", "val", "discovery"):
    for assay in ("RNA", "H3K4me3", "H3K27ac"):
        path = root / cohort / (assay + ".npz")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"dummy public embedding")
        files[str(path.relative_to(root))] = digest(path)
exported = root / "embedding_manifest.json"
exported.write_text(json.dumps({**model, "files": files, "inputs": {"model_manifest_sha256": digest(root / "model_manifest.json")}}))
(root / "public_embedding_manifest.json").write_text(json.dumps({**model, "schema_version": 1, "files": files,
    "verified_at_utc": "fixture", "identity": {"method": "scglue", "cohorts": ["train", "val", "discovery"],
    "checkpoint_sha256": digest(checkpoint), "export_receipts": {"embedding_manifest.json": digest(exported)}}}))
if mode == "changed_export":
    (root / "train/RNA.npz").write_bytes(b"changed")
if mode == "changed_checkpoint":
    checkpoint.write_bytes(b"changed")
pathlib.Path(payload["report"]).write_text(json.dumps({"status": "completed_with_failures"}))
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def build(*, mode="ok", complete=True):
        base = tmp_path / "pipelines/simulation/integration_recoverability"
        base.mkdir(parents=True)
        worker = base / "fake_worker.py"
        worker.write_text(WORKER)
        root = base / "runs/suite"
        bank, calibration = write(tmp_path / "bank.json", {}), write(tmp_path / "calibration.json", {})
        world = write(base / "runs/worlds/constrained_chain_1/manifest.json", {"status": "complete"})
        pins = suite.source_pins(base, tmp_path)
        lock = write(root / "assessment_lock.json", {"source_pins": pins, "bank_sha256": suite.digest(bank),
                    "calibration_sha256": suite.digest(calibration), "development_reports": {}})
        schedule = {"base_directory": str(base), "repository": str(tmp_path), "runs_directory": str(base / "runs"),
            "assessment_lock": str(lock), "development_reports": [], "environment": {"OMP_NUM_THREADS": "1"},
            "inputs": {"bank_path": str(bank), "bank_sha256": suite.digest(bank), "calibration_path": str(calibration),
                       "calibration_sha256": suite.digest(calibration), "source_pins": pins}, "stages": []}
        preparation = {"schedule_hash": None, "conditions": {}, "failed_stage_receipts": {},
            "assessment_lock_sha256": suite.digest(lock), "world_manifest_sha256": suite.digest(world),
            "frozen_source_checks": {name: {"expected_sha256": digest} for name, digest in pins.items()}}
        required = [f"{cohort}/{assay}.npz" for cohort in ("train", "val", "discovery")
                    for assay in ("RNA", "H3K4me3", "H3K27ac")]
        for regime in retry.REGIMES:
            partial = base / f"runs/extended/constrained_chain_1/{regime}/primary/scglue_seed0"
            partial.mkdir(parents=True)
            prepared = partial.parent / "scglue_preprocessing"
            manifest = write(prepared / "manifest.json", {"status": "complete", "fingerprint": "frozen", "assays": ["RNA"]})
            training = write(prepared / "train/RNA.h5ad", {"tiny": "public training file"})
            guidance = write(prepared / "guidance.graphml.gz", {})
            transform = write(prepared / "transforms/RNA.json", {})
            inputs = {"configuration": {"seed": 0}, "training_files": {"RNA": {"path": str(training), "sha256": suite.digest(training)}},
                "guidance": {"path": str(guidance), "sha256": suite.digest(guidance)},
                "transformations": {"files": {"transforms/RNA.json": suite.digest(transform)}}, "package_provenance": {"packages": {}}}
            fit = write(partial / "fit_inputs.json", {"input_fingerprint": suite.canonical(inputs), "inputs": inputs})
            report = partial.parent / "evaluation.json"
            payload = write(tmp_path / f"{regime}_worker_inputs.json", {"root": str(partial), "report": str(report), "fit_inputs": fit.read_text()})
            argv = [sys.executable, str(worker), str(payload), mode if regime == "realistic" else "ok"]
            model, public = partial / "model_manifest.json", partial / "embedding_manifest.json"
            stage = {"stage_id": retry.STAGES[regime], "phase": "F", "kind": "pipeline", "dependencies": [],
                "argv": argv, "fit_keys": [], "reuse_fit_keys": [], "requires_assessment_lock": True,
                "fit_runs": [{"directory": str(partial), "model_manifest": str(model), "embedding_manifest": str(public)}],
                "resources": {"gpu_slots": 1, "large_memory_slots": 1}, "artifacts": [
                    {"path": str(report), "kind": "evaluation", "statuses": ["completed", "completed_with_failures"]},
                    {"path": str(model), "kind": "model", "statuses": ["complete"]},
                    {"path": str(public), "kind": "public_receipt", "statuses": ["complete"], "required_files": required,
                     "model_manifest": str(model)}]}
            schedule["stages"].append(stage)
            preparation["conditions"][regime] = {"stage_id": retry.STAGES[regime], "partial_run": str(partial), "frozen_argv": argv,
                "fit_input_sha256": suite.digest(fit), "input_fingerprint": suite.canonical(inputs), "configuration": inputs["configuration"],
                "training_files_recorded": inputs["training_files"], "preprocessing_manifest_sha256": suite.digest(manifest),
                "native_source_hashes": {}, "partial_inventory": [{"path": "fit_inputs.json", "sha256": suite.digest(fit), "size_bytes": fit.stat().st_size}]}
        schedule["schedule_hash"] = suite.canonical(schedule)
        plan = write(root / "schedule_v1.json", schedule)
        preparation.update(schedule_hash=schedule["schedule_hash"], schedule_sha256=suite.digest(plan))
        attempts = root / "schedule_v1_attempts"
        for regime in retry.REGIMES:
            log = tmp_path / f"old_{regime}.log"
            log.write_text("original native failure\n")
            receipt = suite.record_attempt(schedule, retry.STAGES[regime], 1, log, attempts)
            preparation["failed_stage_receipts"][retry.STAGES[regime]] = {
                "path": str(receipt), "sha256": suite.digest(receipt), "log": str(log), "log_sha256": suite.digest(log)}
        prep = write(tmp_path / retry.PREPARATION, preparation)
        monkeypatch.setattr(retry, "PREPARATION_SHA256", suite.digest(prep))
        study = base / "runs/integration_validation/20261025_v1"
        protocol = write(base / "configs/integration_validation_v1.json", {})
        driver = write(study / "matrix_orchestration.py.txt", {})
        analysis = write(base / "scripts/validate_integration.py", {})
        runtime = write(tmp_path / "dummy_runtime.so", {})
        matrix_plan = write(study / "execution_plan.json", {"runtime_library": str(runtime), "runtime_library_sha256": suite.digest(runtime)})
        selected = write(study / "selected_settings.json", {})
        observed = {"observed_file_hashes": {str(path.relative_to(tmp_path)): suite.digest(path) for path in (protocol, driver, matrix_plan)},
                    "reviewed_analysis_sources": {analysis.name: suite.digest(analysis)}}
        observation = write(study / "matrix_execution_observed_start.json", observed)
        monkeypatch.setattr(retry, "OBSERVATION_SHA256", suite.digest(observation))
        reports = [write(study / f"report{i}.json", {}) for i in range(16)]
        completion = {"status": "complete", "planned_fit_count": 32, "analysis_sources": observed["reviewed_analysis_sources"],
            "protocol_sha256": suite.digest(protocol), "selection_plan_sha256": suite.digest(matrix_plan),
            "selected_settings_sha256": suite.digest(selected), "confirmation_reports": [{"path": str(path), "sha256": suite.digest(path)} for path in reports]}
        if complete:
            write(study / "matrix_execution_complete.json", completion)
        return SimpleNamespace(base=base, root=root, study=study, preparation=preparation, schedule=schedule,
                               attempts=attempts, ledger=root / retry.LEDGER, driver=driver, analysis=analysis)
    return build


def test_preview_has_no_writes_or_launch(fixture, monkeypatch):
    f = fixture(complete=False)
    before = sorted(f.base.parents[2].rglob("*"))
    monkeypatch.setattr(retry.subprocess, "Popen", lambda *a, **k: pytest.fail("Preview launched a child"))
    result = retry.operate(f.base)
    assert result["status"] == "preview" and result["executed_commands"] == 0
    assert result["commands"] == [f.preparation["conditions"][r]["frozen_argv"] for r in retry.REGIMES]
    assert sorted(f.base.parents[2].rglob("*")) == before


def test_missing_matrix_completion_prevents_archive_or_launch(fixture):
    f = fixture(complete=False)
    with pytest.raises(FileNotFoundError, match="matrix_execution_complete"):
        retry.operate(f.base, execute=True)
    assert not f.ledger.exists()
    assert all(Path(c["partial_run"]).exists() for c in f.preparation["conditions"].values())


@pytest.mark.parametrize("target,relative", [("driver", False), ("driver", True), ("analysis", False), ("analysis", True)])
def test_controller_and_orphan_child_are_detected(fixture, tmp_path, target, relative):
    f = fixture()
    proc = tmp_path / "proc/123456789"
    proc.mkdir(parents=True)
    path = getattr(f, target)
    arg = os.path.relpath(path, path.parent) if relative else str(path)
    (proc / "cmdline").write_bytes(f"python\0-u\0{arg}\0".encode())
    (proc / "cwd").symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="still running"):
        retry.assert_matrix_stopped(f.base, proc.parent)


def test_live_matrix_blocks_even_with_completion_receipt(fixture, monkeypatch):
    f = fixture()
    monkeypatch.setattr(retry, "assert_matrix_stopped", lambda base: (_ for _ in ()).throw(RuntimeError("matrix child alive")))
    with pytest.raises(RuntimeError, match="matrix child alive"):
        retry.operate(f.base, execute=True)
    assert not f.ledger.exists()


@pytest.mark.parametrize("extra", ["file", "directory", "symlink"])
def test_unknown_inventory_is_refused_without_moves(fixture, extra):
    f = fixture()
    root = Path(f.preparation["conditions"]["realistic"]["partial_run"])
    path = root / "unexpected"
    if extra == "file":
        path.write_text("unreviewed")
    elif extra == "directory":
        path.mkdir()
    else:
        path.symlink_to(root / "fit_inputs.json")
    with pytest.raises(ValueError, match="inventory|Symlink"):
        retry.operate(f.base, execute=True)
    assert root.exists() and not f.ledger.exists()


def test_existing_archive_is_never_overwritten(fixture):
    f = fixture()
    archive = f.ledger / "archives/realistic/scglue_seed0"
    archive.mkdir(parents=True)
    sentinel = archive / "evidence"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="overwrite archive"):
        retry.archive_partials(f.preparation, f.ledger)
    assert sentinel.read_text() == "preserve"
    assert not (f.ledger / "archive_mapping.json").exists()


def test_success_archives_exact_evidence_and_appends_after_validation(fixture):
    f = fixture()
    originals = {Path(item["path"]): Path(item["path"]).read_bytes() for item in f.preparation["failed_stage_receipts"].values()}
    result = retry.operate(f.base, execute=True)
    assert result["status"] == "completed_two_retries" and result["executed_commands"] == 2
    mapping = suite.read(f.ledger / "archive_mapping.json")
    assert set(mapping) == set(retry.REGIMES)
    for regime, outcome in zip(retry.REGIMES, result["outcomes"]):
        condition = f.preparation["conditions"][regime]
        assert outcome["successful"] and outcome["status"] == "complete_with_failures"
        assert outcome["child_exit_code"] == outcome["effective_exit_code"] == 0
        archived = Path(mapping[regime]["archive"])
        assert retry.inventory(archived) == retry.expected_inventory(condition)
        assert set(mapping[regime]["files"]) == {str(Path(condition["partial_run"]) / "fit_inputs.json")}
        for original, item in mapping[regime]["files"].items():
            assert Path(item["archive"]).relative_to(archived) == Path(original).relative_to(condition["partial_run"])
            assert suite.digest(item["archive"]) == item["sha256"]
        history = suite.attempt_history(f.attempts, f.schedule)[condition["stage_id"]]
        assert [entry["exit_code"] for entry in history] == [1, 0]
        start = suite.read(Path(outcome["log"]).parent / "start.json")
        assert start["argv"] == condition["frozen_argv"]
        assert start["runtime_overrides"]["CUDA_LAUNCH_BLOCKING"] == "1"
        assert start["environment"] == f.schedule["environment"]
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert not list(f.base.rglob("gpu_training.lock"))


@pytest.mark.parametrize("mode,raw_code", [("fail", 7), ("missing", 0), ("changed_inputs", 0),
                                         ("changed_export", 0), ("changed_checkpoint", 0)])
def test_new_failure_stops_and_cannot_be_retried(fixture, mode, raw_code):
    f = fixture(mode=mode)
    result = retry.operate(f.base, execute=True)
    assert result["status"] == "stopped_command_failure" and result["executed_commands"] == 1
    assert result["outcomes"][0]["child_exit_code"] == raw_code
    assert not result["outcomes"][0]["successful"]
    assert result["outcomes"][0]["effective_exit_code"] != 0
    history = suite.attempt_history(f.attempts, f.schedule)
    assert len(history[retry.STAGES["realistic"]]) == 2
    assert len(history[retry.STAGES["informative"]]) == 1
    assert all((f.ledger / f"archive_{regime}_complete.json").exists() for regime in retry.REGIMES)
    before = {path: path.read_bytes() for path in f.attempts.glob("*.json")}
    with pytest.raises((ValueError, FileNotFoundError)):
        retry.operate(f.base, execute=True)
    assert {path: path.read_bytes() for path in f.attempts.glob("*.json")} == before


def test_extra_attempt_refused_even_without_helper_journal(fixture):
    f = fixture()
    log = f.root / "another_failure.log"
    log.write_text("failure")
    suite.record_attempt(f.schedule, retry.STAGES["realistic"], 1, log, f.attempts)
    with pytest.raises(ValueError, match="already attempted"):
        retry.operate(f.base, execute=True)
    assert not f.ledger.exists()


def test_attempt_appended_before_lock_is_refused_before_archiving(fixture, monkeypatch):
    f = fixture()
    original_lock = retry.operator.global_lock
    @contextmanager
    def late_attempt(directory):
        with original_lock(directory):
            log = f.root / "late_failure.log"
            log.write_text("failure")
            suite.record_attempt(f.schedule, retry.STAGES["realistic"], 1, log, f.attempts)
            yield
    monkeypatch.setattr(retry.operator, "global_lock", late_attempt)
    with pytest.raises(ValueError, match="already attempted"):
        retry.operate(f.base, execute=True)
    assert not f.ledger.exists()
    assert all(Path(c["partial_run"]).exists() for c in f.preparation["conditions"].values())


def test_external_native_guard_and_global_lock_are_enforced(fixture, monkeypatch):
    f = fixture()
    with retry.operator.global_lock(f.root):
        with pytest.raises(RuntimeError, match="global lock"):
            retry.operate(f.base, execute=True)
    monkeypatch.setattr(retry.operator, "assert_no_external_jobs", lambda schedule: (_ for _ in ()).throw(RuntimeError("native child alive")))
    with pytest.raises(RuntimeError, match="native child alive"):
        retry.operate(f.base, execute=True)
    assert not f.ledger.exists()


@pytest.mark.parametrize("target", ["source", "completion"])
def test_pin_changes_refused_before_archiving(fixture, target):
    f = fixture()
    if target == "source":
        (f.base / "fake_worker.py").write_text("# changed frozen source\n")
    else:
        path = f.study / "matrix_execution_complete.json"
        value = suite.read(path)
        value["analysis_sources"] = {}
        write(path, value)
    with pytest.raises(ValueError, match="Implementation changed|pins differ"):
        retry.operate(f.base, execute=True)
    assert not f.ledger.exists()
