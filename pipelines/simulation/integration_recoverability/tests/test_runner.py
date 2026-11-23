"""Independent orchestration tests; no production worlds or training are run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_benchmark as runner
import simulator
from calibration import canonical_hash


def _write(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _bank_fixture(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    runs = tmp_path / "runs"
    (runs / "worlds").mkdir(parents=True)
    monkeypatch.setattr(runner, "BASE", base)
    monkeypatch.setattr(runner, "RUNS", runs)
    monkeypatch.setattr(runner, "REPO", tmp_path)
    spec = {"world_id": "linear_chain_0", "design": "extended", "split": "development", "seed": 17}
    calibration = {"assays": {}}
    calibration["calibration_hash"] = canonical_hash(calibration)
    config = {"feature_counts": {"RNA": 20, "H3K27ac": 40, "H3K4me3": 40},
              "cohort_sizes": {"train": 11, "val": 7, "discovery": 13, "probe": 5, "assessment": 17},
              "regimes": ["realistic", "informative"], "calibration_n": 23,
              "background_calibration_n": 19, "feature_chunk_size": 8,
              "conditional_mean_repetitions": 31}
    biology = {"weights": [0.2, 0.7]}
    bank = {"specifications": [spec], "biology_parameters": {spec["world_id"]: biology},
            "calibration_hash": calibration["calibration_hash"], "source_sha256": {},
            "generation_config": config}
    bank["bank_hash"] = canonical_hash(bank)
    _write(runs / "worlds/extended_bank_manifest.json", bank)
    _write(runs / "calibration/empirical_count_profiles.json", calibration)
    return spec, calibration, bank


def test_command_records_both_logs_with_valid_suffixes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS", tmp_path)
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "PYTHON", sys.executable)
    runner.command("tiny", "-c", "print('unit fixture')", log_dir=tmp_path / "logs")
    assert len(list((tmp_path / "logs").glob("*.command.json"))) == 1
    results = list((tmp_path / "logs").glob("*.result.json"))
    assert len(results) == 1
    assert json.loads(results[0].read_text())["returncode"] == 0


def test_frozen_worker_passes_exact_saved_spec_and_generation_settings(tmp_path, monkeypatch):
    spec, calibration, bank = _bank_fixture(tmp_path, monkeypatch)
    seen = {}

    def prepare(*args):
        seen["prepare_args"] = args
        return {"biology": bank["biology_parameters"][spec["world_id"]]}

    def write(*args, **kwargs):
        seen["write_args"], seen["write_kwargs"] = args, kwargs
        return {"count_calibration_status": "passed_declared_checks"}

    monkeypatch.setattr(simulator, "prepare_world", prepare)
    monkeypatch.setattr(simulator, "write_world", write)
    monkeypatch.setattr(simulator, "validate_world", lambda _: {"passed": True})
    runner.generate_frozen_worker(spec["world_id"])
    assert seen["prepare_args"][0] == spec
    assert seen["prepare_args"][1] == calibration
    assert seen["prepare_args"][2:] == (bank["generation_config"]["feature_counts"], 23, 19, 8)
    for key, value in bank["generation_config"].items():
        assert seen["write_kwargs"][key] == value
    assert seen["write_kwargs"]["visible_reference"] is None


def test_frozen_worker_rejects_changed_bank_before_simulation(tmp_path, monkeypatch):
    spec, _, bank = _bank_fixture(tmp_path, monkeypatch)
    bank["generation_config"]["calibration_n"] += 1
    _write(runner.RUNS / "worlds/extended_bank_manifest.json", bank)
    monkeypatch.setattr(simulator, "prepare_world", lambda *args: pytest.fail("Must reject before calibration"))
    with pytest.raises(ValueError, match="manifest"):
        runner.generate_frozen_worker(spec["world_id"])


def test_frozen_worker_rejects_changed_generator_source(tmp_path, monkeypatch):
    spec, _, bank = _bank_fixture(tmp_path, monkeypatch)
    source = runner.BASE / "simulator.py"
    source.write_text("unchanged")
    bank["source_sha256"] = {"simulator.py": runner.digest(source)}
    bank["bank_hash"] = canonical_hash({k: v for k, v in bank.items() if k != "bank_hash"})
    _write(runner.RUNS / "worlds/extended_bank_manifest.json", bank)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Simulator changed"):
        runner.generate_frozen_worker(spec["world_id"])


def test_frozen_worker_rejects_different_biological_parameters(tmp_path, monkeypatch):
    spec, _, _ = _bank_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(simulator, "prepare_world", lambda *args: {"biology": {"weights": [999]}})
    monkeypatch.setattr(simulator, "write_world", lambda *args, **kwargs: pytest.fail("Do not generate changed biology"))
    with pytest.raises(ValueError, match="mechanisms"):
        runner.generate_frozen_worker(spec["world_id"])


def test_assessment_lock_checks_frozen_calibration(tmp_path, monkeypatch):
    _bank_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "source_pins", lambda: {})
    lock = tmp_path / "lock.json"
    _write(lock, {"source_pins": {}, "bank_sha256": runner.digest(runner.RUNS / "worlds/extended_bank_manifest.json"),
                  "calibration_sha256": "incorrect", "development_reports": {}})
    with pytest.raises(ValueError, match="[Cc]alibration"):
        runner.check_lock(lock)


def test_changed_json_record_cannot_silently_overwrite(tmp_path):
    path = tmp_path / "record.json"
    runner.new_json(path, {"result": 1})
    runner.new_json(path, {"result": 1})
    with pytest.raises(FileExistsError):
        runner.new_json(path, {"result": 2})


@pytest.mark.parametrize("changed", ["spec", "biology", "calibration_hash", "feature_counts", "cohort_sizes", "regimes"])
def test_cached_world_is_compared_to_frozen_bank_not_merely_self_validated(tmp_path, monkeypatch, changed):
    spec, _, bank = _bank_fixture(tmp_path, monkeypatch)
    root = runner.RUNS / "worlds" / spec["world_id"]
    cached = {"calibration_hash": bank["calibration_hash"],
              **{key: bank["generation_config"][key] for key in ("feature_counts", "cohort_sizes", "regimes")}}
    frozen = {"spec": spec, "biology": bank["biology_parameters"][spec["world_id"]]}
    if changed in frozen:
        frozen[changed] = {"changed": True}
    else:
        cached[changed] = "changed"
    _write(root / "manifest.json", cached)
    _write(root / "truth/frozen_parameters.json", frozen)
    # Internal artifact validation is a distinct responsibility; this test
    # isolates whether a valid but wrong world can be substituted into the bank.
    monkeypatch.setattr(runner, "command", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Cached world differs"):
        runner.generate(spec["world_id"])


def test_direct_run_world_does_not_bypass_frozen_world_validation(tmp_path, monkeypatch):
    spec, _, _ = _bank_fixture(tmp_path, monkeypatch)
    (runner.RUNS / "worlds" / spec["world_id"]).mkdir()
    checked = []

    def reject_changed_world(world_id):
        checked.append(world_id)
        raise ValueError("frozen world mismatch fixture")

    monkeypatch.setattr(runner, "generate", reject_changed_world)
    with pytest.raises(ValueError, match="frozen world mismatch"):
        runner.run_world(spec["world_id"], "realistic", [0])
    assert checked == [spec["world_id"]]


def test_failed_calibration_blocks_integration_by_default(tmp_path, monkeypatch):
    spec, _, _ = _bank_fixture(tmp_path, monkeypatch)
    root = runner.RUNS / "worlds" / spec["world_id"]
    _write(root / "manifest.json", {"status": "complete", "regimes": ["realistic"],
                                     "count_calibration_status": "failed_declared_checks"})
    monkeypatch.setattr(runner, "generate", lambda _: root)
    with pytest.raises(ValueError, match="Count calibration failed"):
        runner.run_world(spec["world_id"], "realistic", [0])
    assert not (runner.RUNS / "extended").exists()


def test_explicit_failed_calibration_continuation_is_durably_labeled(tmp_path, monkeypatch):
    spec, _, _ = _bank_fixture(tmp_path, monkeypatch)
    root = runner.RUNS / "worlds" / spec["world_id"]
    _write(root / "manifest.json", {"status": "complete", "regimes": ["realistic"],
                                     "count_calibration_status": "failed_declared_checks"})
    monkeypatch.setattr(runner, "generate", lambda _: root)
    original_read = runner.read

    def stop_before_any_preprocessing(path):
        if Path(path) == root / "truth/manifest.json":
            raise RuntimeError("stop after scope audit fixture")
        return original_read(path)

    monkeypatch.setattr(runner, "read", stop_before_any_preprocessing)
    with pytest.raises(RuntimeError, match="stop after scope audit"):
        runner.run_world(spec["world_id"], "realistic", [0], allow_failed_calibration=True)
    scope = original_read(runner.RUNS / "extended" / spec["world_id"] / "realistic/primary/benchmark_scope.json")
    assert scope["allow_failed_calibration"] is True
    assert scope["count_calibration_status"] == "failed_declared_checks"
    assert scope["interpretation"].startswith("diagnostic_only:")
