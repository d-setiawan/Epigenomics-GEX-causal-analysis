"""Preview the two reviewed constrained_chain_1 baseline retries; --execute is explicit.

This one-use operational exception preserves the old frozen commands and evidence.
It must be reviewed before use. It never resumes the separate integration matrix.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import execute_frozen_suite as operator
import run_benchmark as native

suite = operator.suite
BASE = Path(__file__).resolve().parents[1]
REGIMES = ("realistic", "informative")
STAGES = {regime: f"pipeline:constrained_chain_1:{regime}:primary:seeds0" for regime in REGIMES}
PREPARATION = "project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_retry_preparation_readonly_2026_10_25.json"
PREPARATION_SHA256 = "7f96b310d6303a1629805006eebac589ec69c0c3f15a3233e0586b888f9bb7c9"
OBSERVATION_SHA256 = "136d6076a733b5d1874d9f5c897c9769f6e42ab14122ddeb77f99d588de5c861"
OPERATOR_SHA256 = "6bde64d8d361eea62fe9f51fa5a8197a1c87661dca61f8981d24b116c4f7ce9f"
EVENT_SHA256 = "7c47040d7cb17d948fcbeb9803c9e4a599a3412bf6c097b1ccd25168dd8a1e99"
LEDGER = "constrained_chain_1_retry_20261025_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path, expected):
    require(suite.digest(path) == expected, f"Hash changed: {path}")


def inventory(directory):
    require(directory.is_dir() and not directory.is_symlink(), f"Invalid partial directory: {directory}")
    files, directories = {}, set()
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"Symlink in partial inventory: {path}")
        relative = str(path.relative_to(directory))
        if path.is_dir():
            directories.add(relative)
        else:
            require(path.is_file(), f"Non-file in partial inventory: {path}")
            files[relative] = {"size_bytes": path.stat().st_size, "sha256": suite.digest(path)}
    ancestors = {str(parent) for name in files for parent in Path(name).parents if str(parent) != "."}
    require(directories == ancestors, f"Unknown empty directory in partial inventory: {directory}")
    return files


def assert_matrix_stopped(base, proc=Path("/proc")):
    """Also catch an orphan validation child and cwd-relative .py.txt controller."""
    require(proc.is_dir(), "Cannot inspect matrix processes without /proc")
    paths = {str((base / "runs/integration_validation/20261025_v1/matrix_orchestration.py.txt").resolve()),
             str((base / "scripts/validate_integration.py").resolve())}
    relative = {str(Path(path).relative_to(base.parents[2])) for path in paths}
    for folder in proc.iterdir():
        if not folder.name.isdecimal() or int(folder.name) == os.getpid():
            continue
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            args = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
            matches = operator.process_runs_pinned_script(args, paths, relative, lambda: (folder / "cwd").resolve(strict=True))
            controllers = [Path(arg) for arg in args if arg.endswith(".py.txt")]
            matches |= any(str(path.resolve()) in paths for path in controllers if path.is_absolute())
            controllers = [path for path in controllers if not path.is_absolute()]
            if controllers:
                cwd = (folder / "cwd").resolve(strict=True)
                matches |= any(str((cwd / path).resolve()) in paths for path in controllers)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError(f"Cannot inspect same-user process {folder.name}") from error
        require(not matches, f"Matrix controller or validation child {folder.name} is still running")


def context(base, *, completed=False, archived=None, executor_sha256=None):
    repo, root = base.parents[2], base / "runs/suite"
    bound(repo / PREPARATION, PREPARATION_SHA256)
    bound(operator.__file__, OPERATOR_SHA256)
    if executor_sha256:
        bound(__file__, executor_sha256)
    preparation = suite.read(repo / PREPARATION)
    plan = root / "schedule_v1.json"
    bound(plan, preparation["schedule_sha256"])
    schedule = suite.load_schedule(plan)
    require(schedule["schedule_hash"] == preparation["schedule_hash"], "Wrong frozen schedule")
    suite.validate_assessment_lock(schedule)
    bound(schedule["assessment_lock"], preparation["assessment_lock_sha256"])
    bound(base / "runs/worlds/constrained_chain_1/manifest.json", preparation["world_manifest_sha256"])
    require(schedule["inputs"]["source_pins"] == {
        name: item["expected_sha256"] for name, item in preparation["frozen_source_checks"].items()}, "Wrong frozen sources")
    stages = {stage["stage_id"]: stage for stage in schedule["stages"]}
    for regime in REGIMES:
        condition, stage = preparation["conditions"][regime], stages[STAGES[regime]]
        original = base / f"runs/extended/constrained_chain_1/{regime}/primary/scglue_seed0"
        require(condition["stage_id"] == STAGES[regime] and Path(condition["partial_run"]) == original,
                "Unexpected partial-run destination")
        require(stage["argv"] == condition["frozen_argv"] and stage["phase"] == "F" and stage["kind"] == "pipeline",
                "Wrong original pipeline command")
        require(not operator.partial_output_reason(stage), "Partial preprocessing requires manual review")
        old = preparation["failed_stage_receipts"][STAGES[regime]]
        bound(old["path"], old["sha256"])
        bound(old["log"], old["log_sha256"])
        receipt = suite.read(old["path"])
        require(receipt["stage_id"] == STAGES[regime] and receipt["schedule_hash"] == schedule["schedule_hash"]
                and receipt["exit_code"] == 1 and receipt["argv_sha256"] == suite.canonical(stage["argv"])
                and receipt["artifact_sha256"] == {}, "Unexpected failed-stage receipt")
        saved = (archived[regime] if archived else original) / "fit_inputs.json"
        if archived:
            require(inventory(archived[regime]) == expected_inventory(condition), "Archived evidence changed")
        bound(saved, condition["fit_input_sha256"])
        fit = suite.read(saved)
        inputs = fit["inputs"]
        require(fit["input_fingerprint"] == condition["input_fingerprint"] == suite.canonical(inputs)
                and inputs["configuration"] == condition["configuration"]
                and inputs["training_files"] == condition["training_files_recorded"], "Original fit inputs differ")
        prepared = original.parent / "scglue_preprocessing"
        bound(prepared / "manifest.json", condition["preprocessing_manifest_sha256"])
        for item in [*inputs["training_files"].values(), inputs["guidance"]]:
            bound(item["path"], item["sha256"])
        for name, digest in inputs["transformations"]["files"].items():
            bound(prepared / name, digest)
        for item in condition["native_source_hashes"].values():
            bound(item["path"], item["recorded_sha256"])
        for package, version in inputs["package_provenance"]["packages"].items():
            require(importlib.metadata.version(package) == version, f"Native package version changed: {package}")
    study = base / "runs/integration_validation/20261025_v1"
    bound(study / "matrix_execution_observed_start.json", OBSERVATION_SHA256)
    observed = suite.read(study / "matrix_execution_observed_start.json")
    for name, digest in observed["observed_file_hashes"].items():
        bound(repo / name, digest)
    for name, digest in observed["reviewed_analysis_sources"].items():
        bound(base / "scripts" / name, digest)
    matrix_plan = suite.read(study / "execution_plan.json")
    bound(matrix_plan["runtime_library"], matrix_plan["runtime_library_sha256"])
    if completed:
        finish = suite.read(study / "matrix_execution_complete.json")
        require(finish["status"] == "complete" and finish["planned_fit_count"] == 32
                and finish["analysis_sources"] == observed["reviewed_analysis_sources"]
                and finish["protocol_sha256"] == suite.digest(base / "configs/integration_validation_v1.json")
                and finish["selection_plan_sha256"] == suite.digest(study / "execution_plan.json"), "Matrix completion pins differ")
        bound(study / "selected_settings.json", finish["selected_settings_sha256"])
        require(len(finish["confirmation_reports"]) == len({item["path"] for item in finish["confirmation_reports"]}) == 16,
                "Matrix completion lacks unique confirmation receipts")
        for report in finish["confirmation_reports"]:
            bound(report["path"], report["sha256"])
        assert_matrix_stopped(base)
        operator.assert_no_external_jobs(schedule)
    return preparation, schedule, root / "schedule_v1_attempts", matrix_plan


def expected_inventory(condition):
    return {item["path"]: {"size_bytes": item["size_bytes"], "sha256": item["sha256"] or EVENT_SHA256}
            for item in condition["partial_inventory"]}


def retry_command(schedule, attempts, condition):
    status = suite.schedule_status(schedule, attempts_dir=attempts)
    row = next(row for row in status["stages"] if row["stage_id"] == condition["stage_id"])
    require(len(row["attempts"]) == 1 and row["attempts"][0]["exit_code"] == 1, "A retry was already attempted")
    require(row["status"] == "failed" and row["dependencies_complete"]
            and all(item["status"] == "missing" for item in row["artifacts"])
            and not Path(condition["partial_run"]).exists(), "Stage is not the reviewed failed-only retry")
    commands = suite.emit_commands(schedule, status, stage_ids=[condition["stage_id"]], retry_failed=True)
    require(len(commands) == 1 and commands[0]["argv"] == condition["frozen_argv"], "Frozen retry command is unavailable")
    return commands[0]


def archive_partials(preparation, ledger):
    mapping, archived = {}, {}
    for regime in REGIMES:
        condition = preparation["conditions"][regime]
        source, destination = Path(condition["partial_run"]), ledger / "archives" / regime / "scglue_seed0"
        require(not os.path.lexists(destination), f"Refusing to overwrite archive: {destination}")
        files = inventory(source)
        require(files == expected_inventory(condition), f"Unknown or changed partial inventory: {source}")
        archived[regime] = destination
        mapping[regime] = {"source": str(source), "archive": str(destination), "files": {
            str(source / name): {**item, "archive": str(destination / name)} for name, item in files.items()}}
    operator.durable_json(ledger / "archive_mapping.json", mapping)
    for regime in REGIMES:
        source, destination = Path(mapping[regime]["source"]), archived[regime]
        destination.parent.mkdir(parents=True, exist_ok=False)
        require(inventory(source) == expected_inventory(preparation["conditions"][regime]), "Partial inventory changed before archive")
        source.rename(destination)
        require(inventory(destination) == expected_inventory(preparation["conditions"][regime]), "Archive verification failed")
        for directory in (source.parent, destination.parent):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        operator.durable_json(ledger / f"archive_{regime}_complete.json", mapping[regime])
    return archived


def run_one(base, regime, ledger, archived, stop, identity):
    directory, child, code, error, row = ledger / f"attempt_{regime}", None, None, None, None
    bound(base / "runs/integration_validation/20261025_v1/matrix_execution_complete.json", identity["matrix_completion_sha256"])
    preparation, schedule, attempts, runtime = context(base, completed=True, archived=archived,
                                                      executor_sha256=identity["executor_sha256"])
    condition = preparation["conditions"][regime]
    command = retry_command(schedule, attempts, condition)
    environment = os.environ.copy()
    environment.update(command["environment"])
    overrides = {"LD_PRELOAD": runtime["runtime_library"], "CUDA_LAUNCH_BLOCKING": "1"}
    environment.update(overrides)
    log = directory / "output.log"
    operator.durable_json(directory / "start.json", {**identity, **command, "runtime_overrides": overrides,
        "runtime_library_sha256": runtime["runtime_library_sha256"], "executor_pid": os.getpid(),
        "resolved_environment_sha256": suite.canonical(environment), "started_unix_ns": time.time_ns()})
    with log.open("xb") as handle:
        try:
            require(not stop.signum, "Termination requested before launch")
            require(Path(sys.executable).resolve() == Path(command["argv"][0]).resolve(), "Use the frozen Python interpreter")
            child = subprocess.Popen(command["argv"], cwd=command["cwd"], env=environment, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            operator.durable_json(directory / "child.json", {"pid": child.pid, "process_group": child.pid})
            while child.poll() is None:
                if stop.signum:
                    operator.stop_group(child)
                    break
                time.sleep(0.1)
            code = child.wait()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            if child is not None:
                operator.stop_group(child)
                code = child.returncode
            handle.write((error + "\n").encode())
        finally:
            handle.flush()
            os.fsync(handle.fileno())
    try:
        bound(base / "runs/integration_validation/20261025_v1/matrix_execution_complete.json", identity["matrix_completion_sha256"])
        _, schedule, _, _ = context(base, completed=True, archived=archived, executor_sha256=identity["executor_sha256"])
        row = next(row for row in suite.schedule_status(schedule, attempts_dir=attempts)["stages"]
                   if row["stage_id"] == condition["stage_id"])
        require(row["artifacts"] and all(item["status"] in suite.COMPLETE for item in row["artifacts"]),
                "Declared output artifacts are missing or invalid")
        stage = next(stage for stage in schedule["stages"] if stage["stage_id"] == condition["stage_id"])
        for run in stage["fit_runs"]:
            directory_run = Path(run["directory"])
            require((directory_run / "public_embedding_manifest.json").is_file(), "Missing native public-export seal")
            native._public_embedding_manifest(directory_run, directory_run.name.rsplit("_seed", 1)[0],
                                              ("train", "val", "discovery"))
        # Old nonzero receipts still control row.status until the new receipt is appended.
        bound(Path(condition["partial_run"]) / "fit_inputs.json", condition["fit_input_sha256"])
    except Exception as exception:
        error = error or f"{type(exception).__name__}: {exception}"
    successful = code == 0 and error is None and not stop.signum
    effective = 0 if successful else (128 + stop.signum if stop.signum else code or 65)
    outcome = {**identity, "stage_id": condition["stage_id"], "successful": successful, "launched": child is not None,
        "status": "interrupted" if stop.signum else "complete" if successful else "failed", "child_exit_code": code,
        "effective_exit_code": effective, "signal": stop.signum, "error": error, "log": str(log),
        "log_sha256": suite.digest(log), "artifact_validation": row["artifacts"] if row else None}
    try:
        receipt = suite.record_attempt(schedule, condition["stage_id"], effective, log, attempts,
                                       note=f"One bounded baseline retry; journal {directory}; executor {identity['executor_sha256']}")
        operator.sync_file_and_directory(receipt)
        outcome["suite_receipt"] = str(receipt)
        if successful:
            refreshed = next(row for row in suite.schedule_status(schedule, attempts_dir=attempts)["stages"]
                             if row["stage_id"] == condition["stage_id"])
            require(refreshed["status"] in suite.COMPLETE, "Stage did not become complete after the success receipt")
            outcome["status"] = refreshed["status"]
    except Exception as exception:
        outcome.update(successful=False, status="failed", effective_exit_code=74,
                       receipt_error=f"{type(exception).__name__}: {exception}")
    operator.durable_json(directory / "result.json", {**outcome, "finished_unix_ns": time.time_ns()})
    return outcome


def reviewed_partials(preparation, schedule, attempts):
    status = {row["stage_id"]: row for row in suite.schedule_status(schedule, attempts_dir=attempts)["stages"]}
    for regime in REGIMES:
        condition = preparation["conditions"][regime]
        require(inventory(Path(condition["partial_run"])) == expected_inventory(condition), "Unknown or changed partial inventory")
        row = status[condition["stage_id"]]
        require(len(row["attempts"]) == 1, "A retry was already attempted")
        require(row["dependencies_complete"] and row["status"] == "blocked_partial_run"
                and {run["directory"] for run in row["incomplete_runs"]} == {condition["partial_run"]}
                and all(item["status"] == "missing" for item in row["artifacts"]), "Unexpected stage outputs or dependencies")


def operate(base=BASE, *, execute=False):
    base = Path(base).resolve()
    preparation, schedule, attempts, _ = context(base)
    root, ledger = base / "runs/suite", base / "runs/suite" / LEDGER
    reviewed_partials(preparation, schedule, attempts)
    require(not os.path.lexists(ledger), "One-use retry journal already exists; manual review required")
    if not execute:
        return {"status": "preview", "executed_commands": 0, "matrix_completion_present": (
            base / "runs/integration_validation/20261025_v1/matrix_execution_complete.json").exists(),
            "commands": [preparation["conditions"][regime]["frozen_argv"] for regime in REGIMES],
            "archive_root": str(ledger / "archives"), "preparation_sha256": PREPARATION_SHA256}
    identity = {"schedule_hash": schedule["schedule_hash"], "plan_sha256": preparation["schedule_sha256"],
                "executor_sha256": suite.digest(__file__), "preparation_sha256": PREPARATION_SHA256}
    with operator.global_lock(root), operator.StopRequest().handlers() as stop:
        preparation, schedule, attempts, _ = context(base, completed=True, executor_sha256=identity["executor_sha256"])
        operator.check_journal(root / "executions")
        reviewed_partials(preparation, schedule, attempts)
        require(not stop.signum, "Termination requested before archive")
        require(all(Path(sys.executable).resolve() == Path(preparation["conditions"][regime]["frozen_argv"][0]).resolve()
                    for regime in REGIMES), "Use the frozen Python interpreter")
        identity["matrix_completion_sha256"] = suite.digest(base / "runs/integration_validation/20261025_v1/matrix_execution_complete.json")
        ledger.mkdir()  # Atomic one-use claim; existing evidence is never overwritten.
        operator.durable_json(ledger / "session.json", {**identity, "executor_pid": os.getpid(), "started_unix_ns": time.time_ns()})
        outcomes = []
        core_limit = resource.getrlimit(resource.RLIMIT_CORE)
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, core_limit[1]))
            archived = archive_partials(preparation, ledger)
            for regime in REGIMES:
                require(not stop.signum, "Termination requested before next stage")
                outcome = run_one(base, regime, ledger, archived, stop, identity)
                outcomes.append(outcome)
                if not outcome["successful"]:
                    break
            result = {"status": "completed_two_retries" if len(outcomes) == 2 and all(item["successful"] for item in outcomes)
                      else "interrupted" if stop.signum else "stopped_command_failure"}
        except Exception as exception:
            result = {"status": "interrupted" if stop.signum else "stopped_validation_failure",
                      "error": f"{type(exception).__name__}: {exception}"}
        finally:
            resource.setrlimit(resource.RLIMIT_CORE, core_limit)
        result.update(identity, outcomes=outcomes, executed_commands=sum(item["launched"] for item in outcomes), journal=str(ledger))
        operator.durable_json(ledger / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the two exact stages once, only after all matrix processes end")
    args = parser.parse_args(argv)
    try:
        result = operate(execute=args.execute)
    except Exception as exception:
        result = {"status": "refused", "error": f"{type(exception).__name__}: {exception}"}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] in {"preview", "completed_two_retries"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
