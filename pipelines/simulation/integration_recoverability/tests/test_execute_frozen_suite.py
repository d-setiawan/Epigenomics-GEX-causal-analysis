"""Tiny subprocess fixtures only: no simulation, integration, CI, or GPU work."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/execute_frozen_suite.py"
spec = importlib.util.spec_from_file_location("frozen_suite_executor_test", SCRIPT)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)
suite = executor.suite

WORKER = '''import json, os, pathlib, signal, subprocess, sys, time
output, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
print(json.dumps({"cwd": os.getcwd(), "token": os.environ.get("EXECUTOR_UNIT_TOKEN"), "argv": sys.argv}), flush=True)
if mode == "fail":
    sys.exit(7)
if mode == "missing":
    sys.exit(0)
if mode == "sleep":
    child = subprocess.Popen([sys.executable, __file__, str(output), "grandchild"])
    output.with_suffix(".child").write_text(str(child.pid))
    time.sleep(30)
if mode == "grandchild":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    output.with_suffix(".grandchild_ready").write_text(str(os.getpid()))
    time.sleep(30)
    sys.exit(0)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"status": "completed_with_failures" if mode == "scientific_failure" else "completed"}))
if mode == "change_source":
    pathlib.Path(__file__).write_text("# source changed by the tiny fixture\\n")
'''


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / "pipelines/simulation/integration_recoverability"
    base.mkdir(parents=True)
    worker = base / "fake_worker.py"
    worker.write_text(WORKER)
    bank, calibration = tmp_path / "bank.json", tmp_path / "calibration.json"
    bank.write_text("{}")
    calibration.write_text("{}")
    runs = base / "runs"
    plan = tmp_path / "schedule.json"
    schedule = {
        "base_directory": str(base), "repository": str(tmp_path), "runs_directory": str(runs),
        "environment": {"EXECUTOR_UNIT_TOKEN": "exact frozen value"},
        "assessment_lock": str(tmp_path / "assessment_lock.json"), "development_reports": [],
        "inputs": {"bank_path": str(bank), "bank_sha256": suite.digest(bank),
                   "calibration_path": str(calibration), "calibration_sha256": suite.digest(calibration),
                   "source_pins": suite.source_pins(base, tmp_path)},
        "stages": [],
    }

    def add(name, *, phase="D", dependencies=(), mode="ok", kind="evaluate", output=None):
        output = output or tmp_path / "outputs" / name / "report.json"
        command = [sys.executable, str(worker), str(output), mode, "literal;$(not_a_command)"]
        stage = {"stage_id": name, "phase": phase, "kind": kind, "dependencies": list(dependencies),
                 "argv": command, "fit_keys": [], "reuse_fit_keys": [], "requires_assessment_lock": phase == "F",
                 "resources": {"gpu_slots": 0, "large_memory_slots": 0},
                 "artifacts": [{"path": str(output), "kind": "evaluation", "statuses": ["completed", "completed_with_failures"]}]}
        schedule["stages"].append(stage)
        return stage

    def freeze():
        schedule.pop("schedule_hash", None)
        schedule["schedule_hash"] = suite.canonical(schedule)
        suite.write_new(plan, schedule)
        return plan

    return schedule, add, freeze


def artifact(stage):
    return Path(stage["artifacts"][0]["path"])


def test_default_preview_has_no_outputs_logs_lock_or_execution(fixture):
    schedule, add, freeze = fixture
    stage = add("one")
    plan = freeze()
    before = sorted(Path(schedule["repository"]).rglob("*"))
    result = executor.operate(plan, ["D"])
    assert result["status"] == "preview" and result["executed_commands"] == 0
    assert result["commands"][0]["argv"] == stage["argv"]
    assert sorted(Path(schedule["repository"]).rglob("*")) == before
    assert not artifact(stage).exists()


def test_exact_serial_commands_and_durable_provenance(fixture):
    schedule, add, freeze = fixture
    first = add("first", mode="scientific_failure")
    second = add("second", dependencies=["first"])
    result = executor.operate(freeze(), ["D"], execute=True)
    assert result["status"] == "completed_selected_phases" and result["executed_commands"] == 2
    assert result["outcomes"][0]["status"] == "complete_with_failures"
    for outcome, stage in zip(result["outcomes"], [first, second]):
        assert outcome["successful"] and outcome["effective_exit_code"] == 0
        assert outcome["executor_sha256"] == suite.digest(SCRIPT)
        directory = Path(outcome["log"]).parent
        start = suite.read(directory / "start.json")
        assert start["argv"] == stage["argv"] and start["cwd"] == schedule["repository"]
        line = json.loads(Path(outcome["log"]).read_text().splitlines()[0])
        assert line["token"] == "exact frozen value" and line["cwd"] == schedule["repository"]
        assert line["argv"][-1] == "literal;$(not_a_command)"
        receipt = suite.read(outcome["suite_receipt"])
        assert receipt["log_sha256"] == suite.digest(outcome["log"])
        assert receipt["artifact_sha256"][str(artifact(stage))] == suite.digest(artifact(stage))
    again = executor.operate(Path(result["plan_path"]), ["D"], execute=True)
    assert again["executed_commands"] == 0


@pytest.mark.parametrize("mode,raw_code", [("fail", 7), ("missing", 0)])
def test_failure_or_missing_artifacts_stops_and_never_retries(fixture, mode, raw_code):
    _, add, freeze = fixture
    first = add("first", mode=mode)
    second = add("second", dependencies=["first"])
    plan = freeze()
    result = executor.operate(plan, ["D"], execute=True)
    assert result["status"] == "stopped_command_failure" and result["executed_commands"] == 1
    outcome = result["outcomes"][0]
    assert not outcome["successful"] and outcome["child_exit_code"] == raw_code
    assert outcome["effective_exit_code"] != 0 and not artifact(second).exists()
    original = Path(outcome["suite_receipt"]).read_bytes()
    again = executor.operate(plan, ["D"], execute=True)
    assert again["executed_commands"] == 0 and again["status"] == "stopped_no_safe_stage"
    assert Path(outcome["suite_receipt"]).read_bytes() == original


def test_partial_artifact_stage_not_resumed(fixture):
    _, add, freeze = fixture
    stage = add("partial")
    other = artifact(stage).with_name("second.json")
    stage["artifacts"].append({**stage["artifacts"][0], "path": str(other)})
    artifact(stage).parent.mkdir(parents=True)
    artifact(stage).write_text('{"status":"completed"}')
    result = executor.operate(freeze(), ["D"], execute=True)
    assert result["executed_commands"] == 0 and result["blocked_stages"][0]["status"] == "partial"
    assert not other.exists()


def test_existing_nonmodel_output_cache_not_resumed(fixture):
    _, add, freeze = fixture
    stage = add("external_partial")
    stage["argv"] += ["--output", str(artifact(stage).parent)]
    artifact(stage).parent.mkdir(parents=True)
    (artifact(stage).parent / "input_manifest.json").write_text("{}")
    result = executor.operate(freeze(), ["D"], execute=True)
    assert result["executed_commands"] == 0
    assert result["blocked_stages"][0]["status"] == "blocked_partial_output"


@pytest.mark.parametrize("cache", ["scglue_preprocessing", "scglue_preprocessing/cohorts/probe", "gene_transform"])
def test_partial_preprocessing_before_any_model_is_blocked(fixture, cache):
    schedule, add, freeze = fixture
    stage = add("pipeline", kind="pipeline")
    root = Path(schedule["runs_directory"]) / "unit"
    stage["fit_runs"] = [{"directory": str(root / "scglue_seed0"), "model_manifest": str(root / "model.json"),
                          "embedding_manifest": str(root / "embedding.json")}]
    directory = root / cache
    directory.mkdir(parents=True)
    (directory / "partial.npz").write_bytes(b"not scientific data")
    result = executor.operate(freeze(), ["D"])
    assert not result["commands"] and result["stage_counts"]["blocked_partial_output"] == 1


def test_phase_e_never_accepted_and_f_needs_valid_lock(fixture):
    schedule, add, freeze = fixture
    development = add("development")
    assessment = add("assessment", phase="F")
    plan = freeze()
    with pytest.raises(ValueError, match="always manual"):
        executor.operate(plan, ["E"], execute=True)
    assert not executor.operate(plan, ["F"])["commands"]
    lock = {key: schedule["inputs"][key] for key in ("source_pins", "bank_sha256", "calibration_sha256")}
    lock["development_reports"] = {}
    Path(schedule["assessment_lock"]).write_text(json.dumps(lock))
    result = executor.operate(plan, ["D"], execute=True)
    assert artifact(development).exists() and not artifact(assessment).exists()
    result = executor.operate(plan, ["F"], execute=True)
    assert result["status"] == "completed_selected_phases" and artifact(assessment).exists()


def test_source_mutation_after_a_command_stops_before_next_launch(fixture):
    _, add, freeze = fixture
    add("first", mode="change_source")
    second = add("second", dependencies=["first"])
    result = executor.operate(freeze(), ["D"], execute=True)
    assert result["status"] == "stopped_command_failure" and result["executed_commands"] == 1
    assert not artifact(second).exists()
    assert not result["outcomes"][0]["successful"]
    assert "Implementation changed" in result["outcomes"][0]["error"]


def test_plan_and_executor_hashes_rechecked_before_launch(fixture):
    _, add, freeze = fixture
    add("one")
    plan = freeze()
    expected = suite.digest(plan)
    with pytest.raises(ValueError, match="Executor source changed"):
        executor.inspect_plan(plan, ["D"], expected_executor="not_the_current_hash")
    # Even semantically identical JSON bytes cannot replace a running plan.
    plan.write_text(plan.read_text() + "\n")
    with pytest.raises(ValueError, match="Plan bytes changed"):
        executor.inspect_plan(plan, ["D"], expected_plan=expected)


def test_completed_gene_manifest_does_not_require_invented_status(fixture):
    schedule, add, freeze = fixture
    stage = add("pipeline", kind="pipeline")
    root = Path(schedule["runs_directory"]) / "unit"
    stage["fit_runs"] = [{"directory": str(root / "scmrdr_seed0"), "model_manifest": str(root / "model.json"),
                          "embedding_manifest": str(root / "embedding.json")}]
    directory = root / "gene_transform"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "config": {}, "feature_names": ["unit"], "assays": {}, "source_sha256": "unit",
        "train_limit": 2, "excluded_features": [],
    }))
    assert executor.operate(freeze(), ["D"])["commands"][0]["stage_id"] == "pipeline"


def test_global_lock_excludes_second_executor(fixture):
    schedule, add, freeze = fixture
    add("one")
    plan = freeze()
    with executor.global_lock(Path(schedule["runs_directory"]) / "suite"):
        with pytest.raises(RuntimeError, match="global lock"):
            executor.operate(plan, ["D"], execute=True)


def test_stale_started_attempt_blocks_automatic_restart(fixture):
    schedule, add, freeze = fixture
    add("one")
    plan = freeze()
    unfinished = Path(schedule["runs_directory"]) / "suite/executions/prior/attempt_unfinished/start.json"
    executor.durable_json(unfinished, {"status": "started"})
    with pytest.raises(RuntimeError, match="Unresolved executor attempt"):
        executor.operate(plan, ["D"], execute=True)


def test_independent_same_suite_job_blocks_execution(fixture):
    schedule, add, freeze = fixture
    stage = add("one")
    plan = freeze()
    worker = Path(schedule["base_directory"]) / "fake_worker.py"
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", str(worker)])
    try:
        result = executor.operate(plan, ["D"], execute=True)
        assert result["status"] == "stopped_validation_failure" and result["executed_commands"] == 0
        assert "Independent benchmark process" in result["error"]
        assert not artifact(stage).exists()
    finally:
        process.terminate()
        process.wait(timeout=3)


def test_irrelevant_protected_system_process_needs_no_cwd_access(tmp_path):
    script = tmp_path / "evaluate.py"

    def protected_cwd():
        raise PermissionError("sd-pam cwd is protected")

    assert not executor.process_runs_pinned_script(
        ["(sd-pam)"], {str(script)}, {"benchmark/evaluate.py"}, protected_cwd)
    assert not executor.process_runs_pinned_script(
        ["python", str(tmp_path / "unrelated.py")], {str(script)}, {"benchmark/evaluate.py"}, protected_cwd)


def test_pinned_script_detection_keeps_unreadable_relevant_process_fail_closed(tmp_path):
    script = tmp_path / "evaluate.py"

    def protected_cwd():
        raise PermissionError("Relevant cwd is protected")

    assert executor.process_runs_pinned_script(
        ["python", str(script)], {str(script)}, {"benchmark/evaluate.py"}, protected_cwd)
    assert executor.process_runs_pinned_script(
        ["python", "evaluate.py"], {str(script)}, {"benchmark/evaluate.py"}, lambda: tmp_path)
    with pytest.raises(PermissionError, match="Relevant cwd"):
        executor.process_runs_pinned_script(
            ["python", "evaluate.py"], {str(script)}, {"benchmark/evaluate.py"}, protected_cwd)


def test_sigterm_kills_child_group_and_preserves_interrupted_receipt(fixture):
    schedule, add, freeze = fixture
    stage = add("sleep", mode="sleep")
    # Coordination files are separate from the declared absent report.
    artifact(stage).parent.mkdir(parents=True)
    plan = freeze()
    process = subprocess.Popen([sys.executable, str(SCRIPT), "--plan", str(plan), "--phase", "D", "--execute"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready = artifact(stage).with_suffix(".grandchild_ready")
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.02)
        assert ready.exists(), process.communicate(timeout=1)
        descendant = int(ready.read_text())
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 1, stderr
        journal = Path(schedule["runs_directory"]) / "suite/executions"
        result = suite.read(next(journal.glob("*/attempt_*/result.json")))
        assert result["status"] == "interrupted" and result["signal"] == signal.SIGTERM
        assert not result["successful"] and suite.read(result["suite_receipt"])["exit_code"] != 0
        state = Path(f"/proc/{descendant}/stat")
        deadline = time.monotonic() + 2
        while state.exists() and state.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not state.exists() or state.read_text().split()[2] == "Z"
        assert executor.operate(plan, ["D"], execute=True)["executed_commands"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
