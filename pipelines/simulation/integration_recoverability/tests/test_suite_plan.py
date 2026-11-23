"""Schedule-only fixtures: no counts, simulation, training, or CI jobs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import suite_plan as suite


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def schedule(tmp_path):
    base = tmp_path / "pipelines/simulation/integration_recoverability"
    base.mkdir(parents=True)
    (base / "run_benchmark.py").write_text("# source pin fixture\n")
    specs = []
    for family in ("linear", "constrained", "unrestricted"):
        for motif in ("chain", "collider"):
            for draw in range(4):
                specs.append({"world_id": f"{family}_{motif}_{draw}", "family": family,
                              "motif": motif, "split": "development" if draw == 0 else "assessment"})
    for draw in range(4):
        specs.append({"world_id": f"shared_state_{draw}", "motif": "shared_state",
                      "split": "development" if draw == 0 else "assessment"})
    for rho in (0.0, 0.7):
        specs.append({"world_id": f"ambiguity_rho_{rho:.1f}", "motif": "ambiguity", "rho": rho,
                      "split": "development"})
    calibration = {"unit_fixture": True}
    calibration["calibration_hash"] = suite.canonical(calibration)
    bank = {"specifications": specs, "generation_config": {"regimes": list(suite.REGIMES)},
            "development_sentinel_panel": list(suite.SENTINELS), "source_sha256": {},
            "calibration_hash": calibration["calibration_hash"]}
    bank["bank_hash"] = suite.canonical(bank)
    save(base / "runs/worlds/extended_bank_manifest.json", bank)
    save(base / "runs/calibration/empirical_count_profiles.json", calibration)
    return suite.build_schedule(base, python="/unit/python", device="cpu")


def stage_map(schedule):
    return {item["stage_id"]: item for item in schedule["stages"]}


def complete_stage(stage):
    for artifact in stage["artifacts"]:
        assert artifact["kind"] != "lock"
        value = {"status": artifact["statuses"][0], **artifact.get("require_values", {})}
        if artifact["kind"] == "public_receipt":
            value.update(checkpoint_sha256="unit_checkpoint", files={path: "unit_hash" for path in artifact["required_files"]})
        if artifact["kind"] == "model":
            value["checkpoint_sha256"] = "unit_checkpoint"
        if artifact["kind"] == "world":
            value["manifest_hash"] = suite.canonical(value)
        save(Path(artifact["path"]), value)


def completed_development(schedule):
    for stage in schedule["stages"]:
        if stage["phase"] not in {"E", "F"}:
            complete_stage(stage)


def valid_lock(schedule):
    completed_development(schedule)
    value = {key: schedule["inputs"][key] for key in ("source_pins", "bank_sha256", "calibration_sha256")}
    value["development_reports"] = {path: suite.digest(path) for path in schedule["development_reports"]}
    save(Path(schedule["assessment_lock"]), value)


def test_import_has_no_scientific_work_or_filesystem_write(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import attempted data access or a filesystem mutation")
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    spec = importlib.util.spec_from_file_location("isolated_suite_plan", suite.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PHASE_C == suite.PHASE_C


def test_exact_frozen_fit_matrix_no_duplicate_or_twin_refit(schedule):
    fits = [fit for stage in schedule["stages"] for fit in stage["fit_keys"]]
    assert len(fits) == len(set(fits)) == 178
    assert sum(key.startswith(suite.SIMPLE + "/") for key in fits) == 6
    assessment_fits = [fit for stage in schedule["stages"] if stage["phase"] == "F" for fit in stage["fit_keys"]]
    assert len(assessment_fits) == 84
    assert all(key.endswith("seed0") for key in assessment_fits)
    assert not any(key.startswith(suite.TWIN_OTHER + "/") for key in fits)


def test_phase_c_fits_are_reused_by_development_and_have_technical_gate(schedule):
    stages = stage_map(schedule)
    for world in suite.PHASE_C:
        matching = [stage for stage in stages.values() if stage.get("world_id") == world and
                    stage.get("regime") == "informative" and stage.get("arm") == "primary" and stage.get("seeds") == [0]]
        assert len(matching) == 1 and matching[0]["phase"] == "C"
        assert matching[0]["methods"] == list(suite.METHODS)
        assert f"pipeline:{suite.SIMPLE}:realistic:primary:seeds0" in matching[0]["dependencies"]
    example = stages["pipeline:unrestricted_chain_0:realistic:primary:seeds0"]
    assert all(f"pipeline:{world}:informative:primary:seeds0" in example["dependencies"] for world in suite.PHASE_C)


def test_ablations_have_both_regimes_and_guidance_na_is_not_a_refit(schedule):
    stages = schedule["stages"]
    background = [stage for stage in stages if stage["kind"] == "pipeline" and stage.get("arm") == "background_only"]
    shuffled = [stage for stage in stages if stage["kind"] == "pipeline" and stage.get("arm") == "shuffled_guidance"]
    assert len(background) == len(shuffled) == 8
    assert all(stage["seeds"] == [0] and stage["methods"] == list(suite.METHODS) for stage in background)
    assert all(stage["seeds"] == [0] and stage["methods"] == ["scglue"] for stage in shuffled)
    na = [stage for stage in stages if stage["kind"] == "not_applicable"]
    assert len(na) == 8 and all(not stage["argv"] and not stage["fit_keys"] for stage in na)
    priors = [stage for stage in stages if stage["kind"] == "guidance_prior"]
    assert len(priors) == 24
    assert all("not_applicable" in stage["artifacts"][0]["statuses"] for stage in priors)


def test_secondary_ci_checks_are_preselected_not_new_fits(schedule):
    checks = [stage for stage in schedule["stages"] if stage.get("ci_test") in {"kci", "fisherz"}]
    assert sorted((stage["world_id"], stage["ci_test"]) for stage in checks) == [
        ("constrained_chain_0", "kci"), ("linear_chain_0", "fisherz"), ("linear_chain_0", "kci")]
    assert all(stage["regime"] == "informative" and not stage["fit_keys"] for stage in checks)
    assert all("--skip-probes" in stage["argv"] and "partial_CI_only" in stage["evaluation_scope"] for stage in checks)
    assert not any("--skip-probes" in stage["argv"] for stage in schedule["stages"] if stage.get("ci_test") == "rcit")
    assert "count log1p is not Gaussian" in next(stage for stage in checks if stage["ci_test"] == "fisherz")["diagnostic_label"]


def test_twin_stages_use_receipts_private_encode_and_same_base_models(schedule):
    stages = schedule["stages"]
    preparations = [stage for stage in stages if stage["kind"] == "twin_prepare"]
    assert len(preparations) == 4
    for stage in preparations:
        assert stage["immutable_public_receipt_required"] and stage["training_performed_here"] is False
        assert not stage["fit_keys"] and len(stage["reuse_fit_keys"]) == 1
        assert stage["reuse_fit_keys"][0].startswith(suite.TWIN_BASE + "/")
        assert "--base-preprocessing" in stage["argv"] if stage["method"] == "scglue" else "--base-preprocessing" not in stage["argv"]
    assert len([stage for stage in stages if stage["kind"] == "twin_compare"]) == 2


def test_freeze_and_status_do_not_generate_or_execute(schedule, tmp_path):
    worlds = Path(schedule["runs_directory"]) / "worlds"
    before = sorted(worlds.rglob("*"))
    path = tmp_path / "plan.json"
    suite.write_new(path, schedule)
    suite.write_new(path, schedule)
    loaded = suite.load_schedule(path)
    status = suite.schedule_status(loaded)
    commands = suite.emit_commands(loaded, status)
    assert commands and all(isinstance(command["argv"], list) for command in commands)
    assert sorted(worlds.rglob("*")) == before
    assert not Path(schedule["assessment_lock"]).exists()
    assert all(stage["status"] == "blocked_assessment_lock" for stage in status["stages"] if stage["phase"] == "F")
    assert suite.emit_commands(loaded, status, phases=["F"]) == []


def test_development_review_required_before_lock_command(schedule):
    for stage in schedule["stages"]:
        if stage["kind"] in {"guidance_prior", "twin_compare"}:
            assert all(artifact["path"] in schedule["development_reports"] for artifact in stage["artifacts"])
    completed_development(schedule)
    status = suite.schedule_status(schedule)
    assert next(stage for stage in status["stages"] if stage["stage_id"] == "freeze_assessment")["status"] == "awaiting_review"
    assert not suite.emit_commands(schedule, status, phases=["E"])
    commands = suite.emit_commands(schedule, status, phases=["E"], acknowledge_review=True)
    assert len(commands) == 1 and commands[0]["manual_review_acknowledged"]
    assert not suite.emit_commands(schedule, status, phases=["F"])


def test_assessment_only_eligible_with_complete_valid_source_and_evidence_lock(schedule):
    valid_lock(schedule)
    status = suite.schedule_status(schedule)
    assert status["assessment_lock_valid"]
    commands = suite.emit_commands(schedule, status, phases=["F"])
    assert len(commands) == 21  # generation only; integration waits for saved worlds
    assert all("generate" in command["argv"] for command in commands)
    lock = suite.read(schedule["assessment_lock"])
    lock["development_reports"].pop(next(iter(lock["development_reports"])))
    save(Path(schedule["assessment_lock"]), lock)
    assert not suite.schedule_status(schedule)["assessment_lock_valid"]


def test_preexisting_assessment_artifacts_do_not_bypass_missing_lock(schedule):
    assessment = next(stage for stage in schedule["stages"] if stage["phase"] == "F" and stage["kind"] == "pipeline")
    complete_stage(assessment)
    status = suite.schedule_status(schedule)
    row = next(row for row in status["stages"] if row["stage_id"] == assessment["stage_id"])
    assert row["status"] == "blocked_assessment_lock" and not row["eligible"]
    assert suite.emit_commands(schedule, status, phases=["F"], retry_failed=True) == []


def test_failed_attempts_are_immutable_visible_and_never_erased_by_success(schedule, tmp_path):
    key = f"world:{suite.SIMPLE}"
    stage = stage_map(schedule)[key]
    log = tmp_path / "attempt.log"
    log.write_text("unit execution receipt, no command actually executed")
    receipts = tmp_path / "attempts"
    failed = suite.record_attempt(schedule, key, 1, log, receipts)
    failed_hash = suite.digest(failed)
    status = suite.schedule_status(schedule, attempts_dir=receipts)
    assert next(row for row in status["stages"] if row["stage_id"] == key)["status"] == "failed"
    complete_stage(stage)
    suite.record_attempt(schedule, key, 0, log, receipts)
    status = suite.schedule_status(schedule, attempts_dir=receipts)
    row = next(row for row in status["stages"] if row["stage_id"] == key)
    assert row["status"] == "complete" and row["failed_attempts"] == 1 and len(row["attempts"]) == 2
    assert suite.digest(failed) == failed_hash
    save(Path(stage["artifacts"][0]["path"]), {"status": "changed"})
    row = next(row for row in suite.schedule_status(schedule, attempts_dir=receipts)["stages"] if row["stage_id"] == key)
    assert row["status"] == "failed"


def test_partial_saved_world_is_never_silently_regenerated(schedule):
    key = f"world:{suite.SIMPLE}"
    Path(stage_map(schedule)[key]["artifacts"][0]["path"]).parent.mkdir()
    status = suite.schedule_status(schedule)
    row = next(row for row in status["stages"] if row["stage_id"] == key)
    assert row["status"] == "failed"
    assert suite.emit_commands(schedule, status, stage_ids=[key], retry_failed=True) == []


def test_source_changes_and_schedule_changes_are_rejected(schedule, tmp_path):
    path = tmp_path / "plan.json"
    suite.write_new(path, schedule)
    modified = json.loads(json.dumps(schedule))
    modified["protocol"]["primary_seed"] = 17
    save(path, modified)
    with pytest.raises(ValueError, match="schedule changed"):
        suite.load_schedule(path)
    (Path(schedule["base_directory"]) / "run_benchmark.py").write_text("# changed after freeze")
    with pytest.raises(ValueError, match="Implementation changed"):
        suite.schedule_status(schedule)


def test_failed_discovery_and_prior_na_remain_explicit_outcomes(schedule):
    completed_development(schedule)
    stages = stage_map(schedule)
    report = next(stage for stage in stages.values() if stage["kind"] == "evaluate" and stage.get("ci_test") == "kci")["artifacts"][0]
    save(Path(report["path"]), {"status": "completed_with_failures"})
    prior = next(stage for stage in stages.values() if stage["kind"] == "guidance_prior")["artifacts"][0]
    save(Path(prior["path"]), {"status": "not_applicable"})
    status = suite.schedule_status(schedule)
    assert status["counts"]["complete_with_failures"] == 1
    row = next(row for row in status["stages"] if any(entry["path"] == prior["path"] for entry in row["artifacts"]))
    assert row["status"] == "not_applicable"
    assert row["artifacts"][0]["status"] == "not_applicable"


def test_existing_but_incomplete_public_receipt_is_not_a_completed_pipeline(schedule):
    stage = stage_map(schedule)[f"pipeline:{suite.SIMPLE}:realistic:primary:seeds0"]
    complete_stage(stage)
    receipt = next(item for item in stage["artifacts"] if item["kind"] == "public_receipt")
    save(Path(receipt["path"]), {"status": "complete", "checkpoint_sha256": "unit_checkpoint", "files": {}})
    row = next(row for row in suite.schedule_status(schedule)["stages"] if row["stage_id"] == stage["stage_id"])
    assert row["status"] == "failed"
    assert "Incomplete immutable" in " ".join(row["reasons"])


def test_gpu_schedule_declares_serial_resource_slots_without_loading_cuda(schedule):
    gpu = suite.build_schedule(schedule["base_directory"], python="/unit/python", device="cuda")
    assert gpu["execution_policy"]["maximum_gpu_slots"] == 1
    assert gpu["execution_policy"]["maximum_large_memory_slots"] == 1
    for stage in gpu["stages"]:
        assert stage["resources"]["gpu_slots"] == int(stage["kind"] in {"pipeline", "twin_prepare"})


def test_partial_native_checkpoint_blocks_rerun_until_real_recovery_manifests(schedule):
    key = f"pipeline:{suite.SIMPLE}:realistic:primary:seeds0"
    stage = stage_map(schedule)[key]
    complete_stage(stage_map(schedule)[f"world:{suite.SIMPLE}"])
    partial = Path(next(run for run in stage["fit_runs"] if run["method"] == "scglue")["directory"])
    save(partial / "failure.json", {"status": "failed", "stage": "native_training_history_validation"})
    status = suite.schedule_status(schedule)
    row = next(row for row in status["stages"] if row["stage_id"] == key)
    assert row["status"] == "blocked_partial_run"
    assert suite.emit_commands(schedule, status, stage_ids=[key], retry_failed=True) == []
    # The historical failure remains; completed standard manifests, not an
    # optimistic recovery note, establish the recovered output contract.
    complete_stage(stage)
    save(partial / "recovery_provenance.json", {"training_performed": False})
    row = next(row for row in suite.schedule_status(schedule)["stages"] if row["stage_id"] == key)
    assert row["status"] == "complete" and (partial / "failure.json").exists()
