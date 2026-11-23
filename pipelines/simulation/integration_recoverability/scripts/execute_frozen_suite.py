"""Serial operator for an existing frozen suite; previews unless --execute.

Runs only pristine, ready stages in explicitly selected phases. Never creates
an assessment lock, retries a failed/partial stage, or modifies scientific
configuration. The advisory lock coordinates instances of this script, not
independently launched jobs: finish other benchmark jobs before execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import suite_plan as suite

PHASES = {"simple", "C", "D", "F"}
TERMINATION_GRACE_SECONDS = 5


def durable_json(path, value):
    """Create an immutable journal entry and flush file and directory metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    sync_file_and_directory(path)


def sync_file_and_directory(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def global_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Never unlink an advisory-lock file: doing so permits locking another inode.
    with (directory / "executor.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another frozen-suite executor holds the global lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class StopRequest:
    def __init__(self):
        self.signum = None

    def receive(self, signum, _frame):
        self.signum = self.signum or signum

    @contextmanager
    def handlers(self):
        old = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        for number in old:
            signal.signal(number, self.receive)
        try:
            yield self
        finally:
            for number, handler in old.items():
                signal.signal(number, handler)


def stop_group(child):
    """Terminate the whole child session, escalating even if its leader exits."""
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, number)
        except ProcessLookupError:
            pass
        if number == signal.SIGTERM:
            try:
                child.wait(timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
    child.wait()


def partial_output_reason(stage):
    """Guard partial caches that predate this operator and lack stage receipts."""
    argv = stage["argv"]
    if stage["kind"] in {"evaluate", "twin_prepare"} and "--output" in argv:
        output = Path(argv[argv.index("--output") + 1])
        if output.exists():
            return f"Existing unfinished output directory needs manual review: {output}"
    for root in {Path(run["directory"]).parent for run in stage.get("fit_runs", [])}:
        prepared = root / "scglue_preprocessing"
        checks = [(prepared, "manifest.json", {"status": "complete"}, ("fingerprint", "assays"))]
        checks += [(prepared / "cohorts" / cohort, "manifest.json", {"status": "complete"}, ("fingerprint", "assays"))
                   for cohort in ("val", "discovery", "probe", "assessment")]
        checks += [(root / "gene_transform", "manifest.json", {},
                    ("schema_version", "config", "feature_names", "assays", "source_sha256", "train_limit", "excluded_features"))]
        for directory, filename, required, fields in checks:
            if directory.exists():
                try:
                    receipt = suite.read(directory / filename)
                    if any(receipt.get(key) != value for key, value in required.items()) or not all(key in receipt for key in fields):
                        raise ValueError("Incomplete receipt")
                except (OSError, ValueError, TypeError):
                    return f"Partial preprocessing/gene-transform cache needs manual review: {directory}"
    return None


def process_runs_pinned_script(args, scripts, relative, read_cwd):
    """Inspect cwd only when a relative Python-script argument needs resolution.

    Same-user system helpers such as sd-pam can expose their command line but
    protect cwd. They cannot match the existing script-argument heuristic and
    need no cwd access. A relevant, unreadable relative-script cwd still fails
    closed in the caller.
    """
    text = " ".join(args)
    if any(path in text for path in scripts | relative):
        return True
    paths = [Path(arg) for arg in args if arg.endswith(".py")]
    if any(str(path.resolve()) in scripts for path in paths if path.is_absolute()):
        return True
    paths = [path for path in paths if not path.is_absolute()]
    if not paths:
        return False
    cwd = read_cwd()
    return any(str((cwd / path).resolve()) in scripts for path in paths)


def assert_no_external_jobs(schedule):
    """Linux-only conservative check; the advisory lock cannot cover manual jobs."""
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Cannot check independently running jobs: Linux /proc is unavailable")
    scripts = {str((Path(schedule["repository"]) / path).resolve()) for path in schedule["inputs"]["source_pins"]}
    relative = {str(Path(path).relative_to(schedule["repository"])) for path in scripts}
    for folder in proc.iterdir():
        if not folder.name.isdecimal() or int(folder.name) == os.getpid():
            continue
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            args = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
            matches = process_runs_pinned_script(
                args, scripts, relative, lambda: (folder / "cwd").resolve(strict=True))
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during inspection.
        except PermissionError as error:
            raise RuntimeError(f"Cannot inspect same-user process {folder.name}") from error
        if matches:
            raise RuntimeError(f"Independent benchmark process {folder.name} is still running; no concurrent suite execution")


def inspect_plan(path, phases, *, expected_plan=None, expected_executor=None):
    if not phases or not set(phases) <= PHASES:
        raise ValueError("Explicit phases must be simple, C, D, or F; phase E is always manual")
    if expected_plan and suite.digest(path) != expected_plan:
        raise ValueError("Plan bytes changed during execution")
    if expected_executor and suite.digest(__file__) != expected_executor:
        raise ValueError("Executor source changed during execution")
    schedule = suite.load_schedule(path)
    attempts = Path(path).parent / (Path(path).stem + "_attempts")
    status = suite.schedule_status(schedule, attempts_dir=attempts)
    rows = {row["stage_id"]: row for row in status["stages"]}
    stages = {stage["stage_id"]: stage for stage in schedule["stages"]}
    for key, row in rows.items():
        if row["status"] == "ready" and (reason := partial_output_reason(stages[key])):
            row.update(status="blocked_partial_output", eligible=False)
            row["reasons"].append(reason)
    status["counts"] = dict(Counter(row["status"] for row in rows.values()))
    commands = suite.emit_commands(schedule, status, phases=phases)
    # The generic planner also offers partial cache resumes. This unattended
    # operator deliberately does not: those require an explicit human audit.
    commands = [command for command in commands if rows[command["stage_id"]]["status"] == "ready"
                and not rows[command["stage_id"]]["attempts"]
                and rows[command["stage_id"]]["dependencies_complete"]]
    commands = [command for command in commands
                if not stages[command["stage_id"]].get("manual_review_required")
                and stages[command["stage_id"]]["kind"] != "assessment_lock"
                and "lock-assessment" not in command["argv"]]
    return schedule, status, commands, attempts


def check_journal(directory):
    """An uncatchable interruption never becomes an implicit relaunch."""
    blocked = set()
    for start in sorted(Path(directory).glob("*/attempt_*/start.json")):
        finish = start.with_name("result.json")
        if not finish.exists():
            raise RuntimeError(f"Unresolved executor attempt requires manual review: {start}")
        result = suite.read(finish)
        if not result["successful"]:
            blocked.add((result["schedule_hash"], result["stage_id"]))
    return blocked


def run_stage(plan, phases, command, session, attempts, stop, identity):
    directory = session / f"attempt_{time.time_ns()}_{uuid.uuid4().hex}"
    log = directory / "output.log"
    environment = os.environ.copy()
    environment.update(command["environment"])
    start = {**identity, **command, "status": "started", "executor_pid": os.getpid(),
             "started_unix_ns": time.time_ns(), "resolved_environment_sha256": suite.canonical(environment),
             "environment_policy": "Inherited process environment with exact frozen overrides; unrelated values are not logged"}
    durable_json(directory / "start.json", start)
    child, code, error, row = None, None, None, None
    with log.open("xb") as handle:
        try:
            # Re-read immutable plan/source hashes immediately before each launch.
            schedule, _, eligible, _ = inspect_plan(plan, phases, expected_plan=identity["plan_sha256"],
                                                    expected_executor=identity["executor_sha256"])
            if command not in eligible:
                raise RuntimeError("Stage is no longer pristine and eligible")
            assert_no_external_jobs(schedule)
            if stop.signum:
                raise InterruptedError("Termination requested before launch")
            child = subprocess.Popen(command["argv"], cwd=command["cwd"], env=environment,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            durable_json(directory / "child.json", {"pid": child.pid, "process_group": child.pid})
            while child.poll() is None:
                if stop.signum:
                    stop_group(child)
                    break
                time.sleep(0.1)
            code = child.wait()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            if child is not None:
                stop_group(child)
                code = child.returncode
            handle.write((error + "\n").encode())
        finally:
            handle.flush()
            os.fsync(handle.fileno())
    try:
        schedule, after, _, _ = inspect_plan(plan, phases, expected_plan=identity["plan_sha256"],
                                            expected_executor=identity["executor_sha256"])
        row = next(row for row in after["stages"] if row["stage_id"] == command["stage_id"])
        valid = bool(row["artifacts"]) and all(item["status"] in suite.COMPLETE for item in row["artifacts"])
        if not valid:
            error = error or "Declared output artifacts are missing or invalid after command completion"
    except Exception as exception:
        valid = False
        error = error or f"{type(exception).__name__}: {exception}"
    successful = code == 0 and valid and not stop.signum and error is None
    effective = 0 if successful else (128 + stop.signum if stop.signum else code or 65)
    outcome = {**identity, "stage_id": command["stage_id"], "status": (
        "interrupted" if stop.signum else row["status"] if successful else "failed"),
        "successful": successful, "launched": child is not None, "child_exit_code": code, "effective_exit_code": effective,
        "signal": stop.signum, "error": error, "log": str(log), "log_sha256": suite.digest(log),
        "artifact_validation": row["artifacts"] if row else None, "finished_unix_ns": time.time_ns()}
    try:
        receipt = suite.record_attempt(schedule, command["stage_id"], effective, log, attempts,
                                       note=f"Serial executor {identity['executor_sha256']}; journal {directory}")
        sync_file_and_directory(receipt)
        outcome["suite_receipt"] = str(receipt)
    except Exception as exception:
        outcome.update(successful=False, status="failed", effective_exit_code=74,
                       receipt_error=f"{type(exception).__name__}: {exception}")
    durable_json(directory / "result.json", outcome)
    return outcome


def operate(plan, phases, *, execute=False):
    plan = Path(plan).resolve()
    plan_sha256 = suite.digest(plan)
    schedule, status, commands, attempts = inspect_plan(plan, phases, expected_plan=plan_sha256)
    root = Path(schedule["runs_directory"]) / "suite"
    journal = root / "executions"
    blocked = check_journal(journal)
    commands = [command for command in commands if (schedule["schedule_hash"], command["stage_id"]) not in blocked]
    if not execute:
        return {"status": "preview", "schedule_hash": schedule["schedule_hash"], "phases": list(phases),
                "commands": commands, "executed_commands": 0, "stage_counts": status["counts"]}
    identity = {"plan_path": str(plan), "plan_sha256": plan_sha256, "schedule_hash": schedule["schedule_hash"],
                "executor_path": str(Path(__file__).resolve()), "executor_sha256": suite.digest(__file__)}
    outcomes = []
    with global_lock(root), StopRequest().handlers() as stop:
        blocked = check_journal(journal)
        session = journal / f"{time.time_ns()}_{uuid.uuid4().hex}"
        durable_json(session / "session.json", {**identity, "phases": list(phases), "status": "started",
                                                "created_unix_ns": time.time_ns(), "executor_pid": os.getpid()})
        while True:
            try:
                schedule, status, commands, attempts = inspect_plan(plan, phases, expected_plan=identity["plan_sha256"],
                                                                    expected_executor=identity["executor_sha256"])
                assert_no_external_jobs(schedule)
                commands = [command for command in commands if (schedule["schedule_hash"], command["stage_id"]) not in blocked]
            except Exception as exception:
                result = {"status": "stopped_validation_failure", "error": f"{type(exception).__name__}: {exception}"}
                break
            if stop.signum:
                result = {"status": "interrupted", "signal": stop.signum}
                break
            if not commands:
                selected = [row for row in status["stages"] if row["phase"] in phases]
                result = {"status": "completed_selected_phases" if all(row["status"] in suite.COMPLETE for row in selected)
                          else "stopped_no_safe_stage", "stage_counts": dict(Counter(row["status"] for row in selected)),
                          "blocked_stages": [{"stage_id": row["stage_id"], "status": row["status"], "reasons": row["reasons"]}
                                             for row in selected if row["status"] not in suite.COMPLETE]}
                break
            outcome = run_stage(plan, phases, commands[0], session, attempts, stop, identity)
            outcomes.append(outcome)
            print(json.dumps({"stage_id": outcome["stage_id"], "status": outcome["status"], "log": outcome["log"]}), flush=True)
            if not outcome["successful"]:
                result = {"status": "interrupted" if stop.signum else "stopped_command_failure", "failed_stage": outcome["stage_id"]}
                break
        result.update(identity, executed_commands=sum(item["launched"] for item in outcomes),
                      attempted_commands=len(outcomes), session=str(session), outcomes=outcomes)
        durable_json(session / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", nargs="+", choices=sorted(PHASES), required=True)
    parser.add_argument("--execute", action="store_true", help="Launch exact frozen commands serially; default only previews")
    args = parser.parse_args(argv)
    try:
        result = operate(args.plan, args.phase, execute=args.execute)
    except Exception as exception:
        result = {"status": "refused", "error": f"{type(exception).__name__}: {exception}"}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] in {"preview", "completed_selected_phases"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
