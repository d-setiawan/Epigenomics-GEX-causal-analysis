"""Freeze and inspect a benchmark schedule without executing scientific work.

Only ``freeze`` writes a schedule, and only ``record`` appends an execution
receipt. Commands are emitted as argv arrays for an operator or external job
manager; this module never starts subprocesses, generates worlds, or imports
the simulator, integration adapters, or evaluator. Failed attempts remain in
an append-only history. Assessment is unavailable until the exact development
evidence and implementation have a valid, explicitly reviewed lock.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import uuid


BASE = Path(__file__).resolve().parent
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
METHODS = ("scglue", "scmrdr")
REGIMES = ("realistic", "informative")
PHASE_C = ("linear_chain_0", "constrained_chain_0")
SENTINELS = (*PHASE_C, "unrestricted_chain_0", "shared_state_0")
SIMPLE = "simple_saturating_chain"
TWIN_BASE, TWIN_OTHER = "ambiguity_rho_0.0", "ambiguity_rho_0.7"
COMPLETE = {"complete", "complete_with_failures", "not_applicable"}
DEFAULT_PYTHON = "/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python"


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_new(path: str | Path, value: Any) -> None:
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to change frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def source_pins(base: Path, repo: Path) -> dict[str, str]:
    """Same source universe as the runner's assessment lock, without import."""
    paper = repo / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
    files = list(base.glob("*.py")) + list(paper.glob("*.py")) + list((paper / "upstream").glob("*.py"))
    return {str(path.relative_to(repo)): digest(path) for path in sorted(files)}


def validate_bank(bank: Mapping[str, Any]) -> None:
    if canonical({key: value for key, value in bank.items() if key != "bank_hash"}) != bank["bank_hash"]:
        raise ValueError("Frozen bank content hash changed")
    specs = bank["specifications"]
    names = [spec["world_id"] for spec in specs]
    splits = Counter(spec["split"] for spec in specs)
    if len(names) != 30 or len(set(names)) != 30 or splits != {"development": 9, "assessment": 21}:
        raise ValueError("Expected the existing 30-specification, 9/21 frozen bank")
    if tuple(bank["development_sentinel_panel"]) != SENTINELS:
        raise ValueError("The preselected four-world sentinel panel differs")
    if tuple(bank["generation_config"]["regimes"]) != REGIMES:
        raise ValueError("The frozen bank must retain both existing count regimes")
    by_id = {spec["world_id"]: spec for spec in specs}
    if any(by_id[name]["split"] != "development" for name in (*SENTINELS, TWIN_BASE, TWIN_OTHER)):
        raise ValueError("A development control was moved to assessment")
    if [by_id[name].get("rho") for name in (TWIN_BASE, TWIN_OTHER)] != [0.0, 0.7]:
        raise ValueError("The exact ambiguity pair changed")


def build_schedule(base: str | Path = BASE, *, python: str = DEFAULT_PYTHON,
                   device: str = "cuda", assessment_lock: str | Path | None = None) -> dict[str, Any]:
    """Read the already-frozen bank and create an in-memory execution DAG."""
    base = Path(base).resolve()
    repo, runs = base.parents[2], base / "runs"
    bank_path, calibration_path = runs / "worlds/extended_bank_manifest.json", runs / "calibration/empirical_count_profiles.json"
    bank, calibration = read(bank_path), read(calibration_path)
    validate_bank(bank)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be explicitly cpu or cuda")
    if calibration["calibration_hash"] != bank["calibration_hash"] or canonical({
            key: value for key, value in calibration.items() if key != "calibration_hash"}) != calibration["calibration_hash"]:
        raise ValueError("Calibration content differs from the frozen bank")
    for filename, expected in bank["source_sha256"].items():
        if digest(base / filename) != expected:
            raise ValueError(f"Generator changed after bank freeze: {filename}")
    lock_path = Path(assessment_lock).resolve() if assessment_lock else runs / "suite/assessment_lock.json"
    stages: dict[str, dict[str, Any]] = {}
    reports: list[str] = []
    environment = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "NUMBA_CACHE_DIR": "/tmp/ir_numba_cache", "MPLCONFIGDIR": "/tmp/ir_mpl_cache"}

    def argv(script, *args):
        return [str(python), "-u", str(base / script), *map(str, args)]

    def artifact(path, kind="json", statuses=("complete",), **extra):
        return {"path": str(path), "kind": kind, "statuses": list(statuses), **extra}

    def add(stage_id, phase, kind, command=(), dependencies=(), artifacts=(), **metadata):
        value = {"stage_id": stage_id, "phase": phase, "kind": kind, "argv": list(command),
                 "dependencies": list(dict.fromkeys(dependencies)), "artifacts": list(artifacts),
                 "fit_keys": [], "reuse_fit_keys": [], "requires_assessment_lock": phase == "F",
                 "resources": {"gpu_slots": int(device == "cuda" and kind in {"pipeline", "twin_prepare"}),
                               "large_memory_slots": int(kind in {"generate", "pipeline", "twin_prepare", "evaluate"})},
                 **metadata}
        if stage_id in stages and stages[stage_id] != value:
            raise ValueError(f"Conflicting stage definition: {stage_id}")
        stages[stage_id] = value
        return stage_id

    def root_for(world_id, regime, arm="primary"):
        return runs / "simple" / regime if world_id == SIMPLE else runs / "extended" / world_id / regime / arm

    def fit_key(world_id, regime, arm, seed, method):
        return f"{world_id}/{regime}/{arm}/{method}/seed{seed}"

    def generation(world_id, phase, dependencies=()):
        key = f"world:{world_id}"
        if key in stages:
            return key
        return add(key, phase, "generate", argv("run_benchmark.py", "generate", "--world-id", world_id),
                   dependencies, [artifact(runs / "worlds" / world_id / "manifest.json", "world",
                                            require_values={"world_id": world_id, "count_calibration_status": "passed_declared_checks"})],
                   world_id=world_id, no_replacement=True)

    def pipeline(world_id, regime, seeds, phase, dependencies=(), arm="primary", methods=METHODS):
        key = f"pipeline:{world_id}:{regime}:{arm}:seeds{'-'.join(map(str, seeds))}"
        if key in stages:
            return key
        root = root_for(world_id, regime, arm)
        command = argv("run_benchmark.py", "run-world", "--world-id", world_id, "--regime", regime,
                       "--seeds", *seeds, "--methods", *methods, "--device", device, "--arm", arm)
        if phase == "F":
            command += ["--assessment-lock", str(lock_path)]
        name = "evaluation_" + "_".join(methods) + "_seeds" + "-".join(map(str, seeds))
        report = root / name / "report.json"
        outputs = [artifact(report, "evaluation", ("completed", "completed_with_failures"))]
        fit_runs = []
        for seed in seeds:
            for method in methods:
                run = root / f"{method}_seed{seed}"
                filename = "embedding_manifest.json" if method == "scglue" else "public_embedding_manifest.json"
                model_manifest = run / ("model_manifest.json" if method == "scglue" else "manifest.json")
                cohorts = ["train"] if world_id == SIMPLE else ["train", "val", "discovery"]
                outputs.append(artifact(model_manifest, "model"))
                outputs.append(artifact(run / filename, "public_receipt", required_files=[
                    f"{cohort}/{assay}.npz" for cohort in cohorts for assay in ASSAYS], model_manifest=str(model_manifest)))
                fit_runs.append({"method": method, "directory": str(run), "model_manifest": str(model_manifest),
                                 "embedding_manifest": str(run / filename)})
        add(key, phase, "pipeline", command, dependencies, outputs, world_id=world_id, regime=regime, arm=arm,
            seeds=list(seeds), methods=list(methods), fit_keys=[fit_key(world_id, regime, arm, seed, method)
                for seed in seeds for method in methods], fit_runs=fit_runs, includes_matching_and_evaluation=True)
        if phase != "F":
            reports.append(str(report))
        return key

    def evaluation(world_id, regime, seeds, phase, dependencies, *, test="rcit", suffix="all_seeds"):
        root = root_for(world_id, regime)
        output = root / f"evaluation_suite_{suffix}"
        command = argv("evaluate.py", "evaluate", "--world", runs / "worlds" / world_id,
                       "--regime", regime, "--embeddings", *[root / f"{method}_seed{seed}" for seed in seeds for method in METHODS],
                       "--output", output, "--ci-test", test, "--ci-seed", 0)
        if world_id != SIMPLE:
            command += ["--preprocessing-embeddings", str(root / "scglue_preprocessing/representations")]
        if test != "rcit":
            command.append("--skip-probes")
        report = output / "report.json"
        key = add(f"evaluation:{world_id}:{regime}:{suffix}", phase, "evaluate", command, dependencies,
                  [artifact(report, "evaluation", ("completed", "completed_with_failures"))],
                  world_id=world_id, regime=regime, ci_test=test,
                  evaluation_scope=("partial_CI_only_diagnostic; primary RCIT retains all probes" if test != "rcit"
                                    else "complete_matching_and_retention_evaluation"),
                  diagnostic_label=("linear_activity_control; count log1p is not Gaussian" if test == "fisherz" else
                                    "preselected_secondary_CI_check" if test == "kci" else "all_seeds_common_pool"))
        reports.append(str(report))
        return key

    simple_world = generation(SIMPLE, "simple")
    simple_zero = pipeline(SIMPLE, "realistic", (0,), "simple", (simple_world,))
    simple_repeat = pipeline(SIMPLE, "realistic", (1, 2), "simple", (simple_zero,))
    evaluation(SIMPLE, "realistic", (0, 1, 2), "simple", (simple_zero, simple_repeat))
    smoke = []
    for world_id in PHASE_C:
        world = generation(world_id, "C", (simple_zero,))
        smoke.append(pipeline(world_id, "informative", (0,), "C", (world, simple_zero)))
    primary = {}
    development = [spec["world_id"] for spec in bank["specifications"] if spec["split"] == "development"]
    for world_id in development:
        if world_id == TWIN_OTHER:
            continue
        world = generation(world_id, "D", smoke)
        for regime in REGIMES:
            primary[world_id, regime] = pipeline(world_id, regime, (0,), "D", (world, *smoke))

    for regime in REGIMES:
        world = generation(TWIN_OTHER, "D", (f"world:{TWIN_BASE}", *smoke))
        base_root, other_root = root_for(TWIN_BASE, regime), root_for(TWIN_OTHER, regime)
        preparations, embedding_dirs = [], []
        native = None
        for method in METHODS:
            control = other_root / f"{method}_seed0_twin_control"
            command = argv("twin_controls.py", "prepare", "--base-world", runs / "worlds" / TWIN_BASE,
                           "--counterpart-world", runs / "worlds" / TWIN_OTHER, "--regime", regime,
                           "--method", method, "--base-run", base_root / f"{method}_seed0",
                           "--output", control, "--device", device)
            if method == "scglue":
                command += ["--base-preprocessing", str(base_root / "scglue_preprocessing")]
                native = control / "counterpart_preprocessing/representations"
            preparations.append(add(f"twin_prepare:{regime}:{method}", "D", "twin_prepare", command,
                (world, primary[TWIN_BASE, regime]), [artifact(control / "twin_control.json")],
                world_id=TWIN_OTHER, regime=regime, method=method, immutable_public_receipt_required=True,
                reuse_fit_keys=[fit_key(TWIN_BASE, regime, "primary", 0, method)], training_performed_here=False))
            embedding_dirs.append(control / "embeddings")
        observed = add(f"twin_validate_observed:{regime}", "D", "validate_observed",
                      argv("evaluate.py", "validate-observed", "--world", runs / "worlds" / TWIN_OTHER, "--regime", regime),
                      (world,), [artifact(runs / "worlds" / TWIN_OTHER / "truth" / f"observation_validation_{regime}.json", statuses=("ok",))])
        output = other_root / "evaluation_scglue_scmrdr_seeds0"
        other_eval = add(f"twin_evaluate:{regime}", "D", "evaluate", argv("evaluate.py", "evaluate",
            "--world", runs / "worlds" / TWIN_OTHER, "--regime", regime, "--embeddings", *embedding_dirs,
            "--preprocessing-embeddings", native, "--output", output, "--ci-test", "rcit", "--ci-seed", 0),
            (*preparations, observed), [artifact(output / "report.json", "evaluation", ("completed", "completed_with_failures"))],
            world_id=TWIN_OTHER, regime=regime, training_performed_here=False)
        reports.append(str(output / "report.json"))
        add(f"twin_compare:{regime}", "D", "twin_compare", argv("twin_controls.py", "compare",
            "--base-world", runs / "worlds" / TWIN_BASE, "--counterpart-world", runs / "worlds" / TWIN_OTHER,
            "--regime", regime, "--base-report", base_root / "evaluation_scglue_scmrdr_seeds0/report.json",
            "--counterpart-report", output / "report.json", "--output", other_root / "twin_comparison.json"),
            (other_eval, primary[TWIN_BASE, regime]), [artifact(other_root / "twin_comparison.json", "comparison",
                ("completed", "completed_with_failed_discovery"))], world_id=TWIN_OTHER, regime=regime,
            interpretation="Same public outputs scored under distinct hidden truths; not two independent trials")
        reports.append(str(other_root / "twin_comparison.json"))

    for world_id in SENTINELS:
        for regime in REGIMES:
            original = primary[world_id, regime]
            repeat = pipeline(world_id, regime, (1, 2), "D", (original,))
            evaluation(world_id, regime, (0, 1, 2), "D", (original, repeat))
            background = pipeline(world_id, regime, (0,), "D", (original,), arm="background_only")
            shuffled = pipeline(world_id, regime, (0,), "D", (original,), arm="shuffled_guidance", methods=("scglue",))
            add(f"not_applicable:{world_id}:{regime}:scmrdr_shuffled_guidance", "D", "not_applicable",
                dependencies=(original,), world_id=world_id, regime=regime, method="scmrdr",
                reason="scMRDR has no guidance-graph input; rerunning it would duplicate the primary fit")
            for arm, dependency in (("primary", original), ("background_only", background), ("shuffled_guidance", shuffled)):
                root = root_for(world_id, regime, arm)
                add(f"guidance_prior:{world_id}:{regime}:{arm}", "D", "guidance_prior",
                    argv("guidance_baseline.py", "--preprocessing-dir", root / "scglue_preprocessing",
                         "--truth-manifest", runs / "worlds" / world_id / "truth/manifest.json",
                         "--output", root / "guidance_only_baseline.json"), (dependency,),
                    [artifact(root / "guidance_only_baseline.json", "prior", ("ok", "not_applicable"))],
                    world_id=world_id, regime=regime, arm=arm,
                    interpretation="Undirected exposed annotation prior only; missing scored vertices abstain and unsupported metrics remain N/A")
                reports.append(str(root / "guidance_only_baseline.json"))
    for world_id in PHASE_C:
        evaluation(world_id, "informative", (0,), "D", (primary[world_id, "informative"],), test="kci", suffix="secondary_kci")
    evaluation("linear_chain_0", "informative", (0,), "D", (primary["linear_chain_0", "informative"],),
               test="fisherz", suffix="linear_fisherz_diagnostic")

    reports = sorted(set(reports))
    development_stages = list(stages)
    add("freeze_assessment", "E", "assessment_lock", argv("run_benchmark.py", "lock-assessment",
        "--development-reports", *reports, "--out", lock_path), development_stages,
        [artifact(lock_path, "lock", ())], manual_review_required=True,
        review_scope="Review every development outcome, including failures; no favorable-recovery threshold")
    for spec in bank["specifications"]:
        if spec["split"] == "assessment":
            world = generation(spec["world_id"], "F", ("freeze_assessment",))
            for regime in REGIMES:
                pipeline(spec["world_id"], regime, (0,), "F", (world, "freeze_assessment"))
    schedule = {
        "schema_version": 1, "status": "frozen_execution_schedule", "base_directory": str(base),
        "repository": str(repo), "runs_directory": str(runs), "python": str(python), "device": device,
        "environment": environment, "assessment_lock": str(lock_path), "development_reports": reports,
        "execution_policy": {"maximum_gpu_slots": 1, "maximum_large_memory_slots": 1,
                             "enforcement": "Operator/external executor must serialize resource-demanding commands; this planner launches nothing"},
        "inputs": {"bank_path": str(bank_path), "bank_sha256": digest(bank_path), "bank_hash": bank["bank_hash"],
                   "calibration_path": str(calibration_path), "calibration_sha256": digest(calibration_path),
                   "source_pins": source_pins(base, repo)},
        "protocol": {"simple_seeds": [0, 1, 2], "phase_c_worlds": list(PHASE_C), "phase_c_regime": "informative",
                     "development_specifications": development, "assessment_specifications": [spec["world_id"] for spec in bank["specifications"] if spec["split"] == "assessment"],
                     "sentinel_worlds": list(SENTINELS), "extended_regimes": list(REGIMES), "primary_seed": 0,
                     "sentinel_extra_seeds": [1, 2], "ablation_regimes": list(REGIMES),
                     "background_only_methods": list(METHODS), "shuffled_guidance_methods": ["scglue"],
                     "guidance_only_baseline": "All three scGLUE sentinel arms; report coverage and N/A without inventing causal directions",
                     "guidance_masking": "Primary already masks all scored cross-feature links; no separate fit",
                     "kci_worlds": list(PHASE_C), "secondary_regime": "informative", "fisherz_worlds": ["linear_chain_0"],
                     "secondary_scope": "CI-only partial diagnostics skip duplicate probe fits; not complete extended results",
                     "assessment_runs_per_world_regime": "both methods, seed0 only; validated source/evidence lock required",
                     "no_automatic_execution": True, "no_world_replacement": True,
                     "independence": "Twins, count regimes, and integration seeds are not independent biological worlds"},
        "stages": list(stages.values()),
    }
    validate_schedule(schedule)
    schedule["schedule_hash"] = canonical(schedule)
    return schedule


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    stages = {stage["stage_id"]: stage for stage in schedule["stages"]}
    if len(stages) != len(schedule["stages"]):
        raise ValueError("Repeated stage IDs")
    owners = {}
    for key, stage in stages.items():
        if set(stage["dependencies"]) - stages.keys() or key in stage["dependencies"]:
            raise ValueError(f"Invalid dependencies: {key}")
        for fit in stage["fit_keys"]:
            if fit in owners:
                raise ValueError(f"Duplicated planned fit: {fit}")
            owners[fit] = key
        if stage.get("world_id") == TWIN_OTHER and stage["fit_keys"]:
            raise ValueError("The ambiguity counterpart must never be refitted")
        if stage["phase"] == "F" and not stage["requires_assessment_lock"]:
            raise ValueError("Assessment stage lacks its release gate")
    resolved = set()
    while len(resolved) < len(stages):
        ready = {key for key, stage in stages.items() if set(stage["dependencies"]) <= resolved} - resolved
        if not ready:
            raise ValueError("Cyclic stage dependencies")
        resolved.update(ready)
    for stage in stages.values():
        if set(stage["reuse_fit_keys"]) - owners.keys():
            raise ValueError("Reuse references an unplanned fit")


def load_schedule(path: str | Path) -> dict[str, Any]:
    schedule = read(path)
    if canonical({key: value for key, value in schedule.items() if key != "schedule_hash"}) != schedule["schedule_hash"]:
        raise ValueError("Frozen execution schedule changed")
    validate_schedule(schedule)
    return schedule


def validate_inputs(schedule: Mapping[str, Any]) -> None:
    inputs = schedule["inputs"]
    for key in ("bank", "calibration"):
        if digest(inputs[f"{key}_path"]) != inputs[f"{key}_sha256"]:
            raise ValueError(f"Frozen {key} changed")
    current = source_pins(Path(schedule["base_directory"]), Path(schedule["repository"]))
    if current != inputs["source_pins"]:
        raise ValueError("Implementation changed after schedule freeze; freeze a separate schedule, never overwrite this one")


def validate_assessment_lock(schedule: Mapping[str, Any]) -> None:
    validate_inputs(schedule)
    lock = read(schedule["assessment_lock"])
    for key in ("source_pins", "bank_sha256", "calibration_sha256"):
        if lock[key] != schedule["inputs"][key]:
            raise ValueError(f"Assessment lock differs from frozen {key}")
    if set(lock["development_reports"]) != set(schedule["development_reports"]):
        raise ValueError("Assessment lock omits or substitutes scheduled development reports")
    for path, expected in lock["development_reports"].items():
        if digest(path) != expected:
            raise ValueError("Development evidence changed after assessment lock")


def attempt_history(directory: str | Path | None, schedule: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {stage["stage_id"]: [] for stage in schedule["stages"]}
    if directory is None or not Path(directory).exists():
        return result
    for path in sorted(Path(directory).glob("*.json")):
        value = read(path)
        if value["schedule_hash"] != schedule["schedule_hash"] or value["stage_id"] not in result:
            raise ValueError("Attempt receipt belongs to another schedule")
        result[value["stage_id"]].append({**value, "receipt_path": str(path)})
    return result


def _artifact_status(entry: Mapping[str, Any], schedule: Mapping[str, Any]) -> tuple[str, str | None]:
    path = Path(entry["path"])
    if not path.exists():
        return "missing", None
    try:
        value = read(path)
        if entry["kind"] == "lock":
            validate_assessment_lock(schedule)
            return "complete", None
        if value.get("status") not in entry["statuses"]:
            return "failed", f"Unexpected artifact status: {value.get('status')}"
        for key, expected in entry.get("require_values", {}).items():
            if value.get(key) != expected:
                return "failed", f"Artifact does not satisfy {key}={expected!r}"
        if entry["kind"] == "public_receipt":
            if not value.get("checkpoint_sha256") or not set(entry["required_files"]) <= set(value.get("files", {})):
                return "failed", "Incomplete immutable public-embedding receipt"
            model = read(entry["model_manifest"])
            if model.get("status") != "complete" or model.get("checkpoint_sha256") != value["checkpoint_sha256"]:
                return "failed", "Public-embedding receipt differs from completed model manifest"
        if entry["kind"] == "model" and not value.get("checkpoint_sha256"):
            return "failed", "Completed model manifest lacks a checkpoint identity"
        if entry["kind"] == "world" and "manifest_hash" in value:
            if canonical({key: item for key, item in value.items() if key != "manifest_hash"}) != value["manifest_hash"]:
                return "failed", "World manifest content hash changed"
        if value.get("status") == "not_applicable":
            return "not_applicable", None
        return ("complete_with_failures" if value.get("status") in {"completed_with_failures", "completed_with_failed_discovery"} else "complete"), None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "failed", f"{type(error).__name__}: {error}"


def schedule_status(schedule: Mapping[str, Any], *, attempts_dir: str | Path | None = None) -> dict[str, Any]:
    """Inspect small declared manifests, not fit models or audit every count byte."""
    validate_inputs(schedule)
    history = attempt_history(attempts_dir, schedule)
    try:
        validate_assessment_lock(schedule)
        locked, lock_reason = True, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        locked, lock_reason = False, str(error)
    pending = {stage["stage_id"]: stage for stage in schedule["stages"]}
    states = {}
    while pending:
        for key, stage in list(pending.items()):
            if set(stage["dependencies"]) - states.keys():
                continue
            dependencies_complete = all(states[dep]["status"] in COMPLETE for dep in stage["dependencies"])
            artifacts = [{**entry, **dict(zip(("status", "reason"), _artifact_status(entry, schedule)))} for entry in stage["artifacts"]]
            attempts = history[key]
            reasons = [entry["reason"] for entry in artifacts if entry["reason"]]
            if stage["kind"] == "not_applicable":
                state = "not_applicable"
            elif any(entry["status"] == "failed" for entry in artifacts):
                state = "failed"
            elif attempts and attempts[-1]["exit_code"] != 0:
                state = "failed"
            elif artifacts and all(entry["status"] in COMPLETE for entry in artifacts):
                state = ("not_applicable" if all(entry["status"] == "not_applicable" for entry in artifacts) else
                         "complete_with_failures" if any(entry["status"] == "complete_with_failures" for entry in artifacts) else "complete")
            elif not dependencies_complete:
                state = "blocked_dependencies"
            elif stage["requires_assessment_lock"] and not locked:
                state = "blocked_assessment_lock"
            elif stage.get("manual_review_required"):
                state = "awaiting_review"
            elif any(entry["status"] in COMPLETE for entry in artifacts):
                state = "partial"
            else:
                state = "ready"
            if stage["kind"] == "generate" and artifacts[0]["status"] == "missing" and Path(artifacts[0]["path"]).parent.exists():
                state = "failed"
                reasons.append("Partial saved world exists without a complete manifest; replacement/regeneration is prohibited")
            incomplete_runs = []
            for run in stage.get("fit_runs", []):
                directory = Path(run["directory"])
                if directory.exists() and any(directory.iterdir()) and any(
                        not Path(run[key]).exists() for key in ("model_manifest", "embedding_manifest")):
                    incomplete_runs.append(run)
            if incomplete_runs:
                state = "blocked_partial_run"
                reasons.append("Preserved fit/export artifacts are incomplete; require explicit audited recovery, never emit a training rerun")
            if stage["requires_assessment_lock"] and not locked:
                # Existing artifacts do not retroactively authorize assessment.
                state = "blocked_assessment_lock"
                reasons.append(lock_reason)
            if attempts and attempts[-1]["exit_code"] == 0:
                for path, expected in attempts[-1]["artifact_sha256"].items():
                    if not Path(path).exists() or digest(path) != expected:
                        state = "failed"
                        reasons.append("Declared output changed after its successful attempt receipt")
            states[key] = {"stage_id": key, "phase": stage["phase"], "kind": stage["kind"], "status": state,
                           "eligible": state in {"ready", "partial"}, "dependencies_complete": dependencies_complete,
                           "artifacts": artifacts, "reasons": reasons, "attempts": attempts,
                           "incomplete_runs": incomplete_runs,
                           "failed_attempts": sum(item["exit_code"] != 0 for item in attempts)}
            del pending[key]
    return {"schedule_hash": schedule["schedule_hash"], "assessment_lock_valid": locked,
            "assessment_lock_reason": lock_reason, "counts": dict(Counter(row["status"] for row in states.values())),
            "verification_scope": "Declared small manifests/receipts and frozen sources; stage runners perform full data/checkpoint audits before reuse",
            "stages": list(states.values())}


def emit_commands(schedule: Mapping[str, Any], status: Mapping[str, Any], *, phases: Sequence[str] = (),
                  stage_ids: Sequence[str] = (), retry_failed: bool = False,
                  acknowledge_review: bool = False) -> list[dict[str, Any]]:
    states = {row["stage_id"]: row for row in status["stages"]}
    if status["schedule_hash"] != schedule["schedule_hash"]:
        raise ValueError("Status belongs to another schedule")
    if set(stage_ids) - states.keys():
        raise ValueError("Unknown requested stage ID")
    commands = []
    for stage in schedule["stages"]:
        row = states[stage["stage_id"]]
        if (phases and stage["phase"] not in phases) or (stage_ids and stage["stage_id"] not in stage_ids):
            continue
        eligible = row["eligible"] or (acknowledge_review and row["status"] == "awaiting_review")
        if retry_failed and row["status"] == "failed" and row["dependencies_complete"]:
            eligible = not stage["requires_assessment_lock"] or status["assessment_lock_valid"]
            if stage["kind"] == "generate" and Path(stage["artifacts"][0]["path"]).parent.exists():
                eligible = False  # No retry may replace a partially or fully saved world.
        if eligible and stage["argv"]:
            commands.append({"stage_id": stage["stage_id"], "phase": stage["phase"], "argv": stage["argv"],
                             "cwd": schedule["repository"], "environment": schedule["environment"],
                             "resources": stage["resources"],
                             "previous_failed_attempts": row["failed_attempts"],
                             "manual_review_acknowledged": bool(acknowledge_review and stage.get("manual_review_required"))})
    return commands


def record_attempt(schedule: Mapping[str, Any], stage_id: str, exit_code: int, log: str | Path,
                   directory: str | Path, *, note: str = "") -> Path:
    """Append an operator-reported outcome; never delete an earlier failure."""
    validate_inputs(schedule)
    stage = next((item for item in schedule["stages"] if item["stage_id"] == stage_id), None)
    if stage is None or not stage["argv"]:
        raise ValueError("Unknown or non-executable stage")
    log = Path(log).resolve()
    record = {"schedule_hash": schedule["schedule_hash"], "stage_id": stage_id, "argv_sha256": canonical(stage["argv"]),
              "recorded_unix_ns": time.time_ns(), "exit_code": int(exit_code), "log": str(log),
              "log_sha256": digest(log), "note": note,
              "provenance": "Operator-reported outcome; this command emitter did not supervise execution",
              "artifact_sha256": {entry["path"]: digest(entry["path"]) for entry in stage["artifacts"] if Path(entry["path"]).exists()}}
    path = Path(directory) / f"{record['recorded_unix_ns']}_{uuid.uuid4().hex}.json"
    write_new(path, record)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Write a new immutable schedule; execute nothing")
    freeze.add_argument("--base", type=Path, default=BASE)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--python", default=DEFAULT_PYTHON)
    freeze.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    freeze.add_argument("--assessment-lock", type=Path)
    for name in ("status", "commands", "record"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--attempts-dir", type=Path)
        if name == "commands":
            command.add_argument("--phase", nargs="+", choices=("simple", "C", "D", "E", "F"), default=[])
            command.add_argument("--stage", nargs="+", default=[])
            command.add_argument("--retry-failed", action="store_true")
            command.add_argument("--acknowledge-review", action="store_true")
        if name == "record":
            command.add_argument("--stage", required=True)
            command.add_argument("--exit-code", type=int, required=True)
            command.add_argument("--log", type=Path, required=True)
            command.add_argument("--note", default="")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        schedule = build_schedule(args.base, python=args.python, device=args.device, assessment_lock=args.assessment_lock)
        write_new(args.output, schedule)
        result = {"schedule": str(args.output), "schedule_hash": schedule["schedule_hash"],
                  "stages": len(schedule["stages"]), "unique_fits": sum(len(stage["fit_keys"]) for stage in schedule["stages"]),
                  "executed_commands": 0}
    else:
        schedule = load_schedule(args.plan)
        attempts = args.attempts_dir or args.plan.parent / (args.plan.stem + "_attempts")
        if args.command == "record":
            result = {"attempt_receipt": str(record_attempt(schedule, args.stage, args.exit_code, args.log, attempts, note=args.note))}
        else:
            result = schedule_status(schedule, attempts_dir=attempts)
            if args.command == "commands":
                result = emit_commands(schedule, result, phases=args.phase, stage_ids=args.stage,
                                       retry_failed=args.retry_failed, acknowledge_review=args.acknowledge_review)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
