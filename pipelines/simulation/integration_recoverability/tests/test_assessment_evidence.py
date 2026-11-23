"""Assessment release accepts typed terminal outcomes, never status-only stubs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_benchmark as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    base = tmp_path / "source"
    base.mkdir()
    for name in ("evaluate.py", "guidance_baseline.py", "twin_controls.py"):
        (base / name).write_text(f"# frozen {name} test source\n")
    monkeypatch.setattr(runner, "BASE", base)
    return base


def evaluation(world, *, failures=False):
    inputs = {
        "world": str(world.resolve()), "world_manifest_sha256": "1" * 64,
        "truth_manifest_sha256": "2" * 64, "truth_sha256": "3" * 64,
        "evaluation_source_sha256": runner.digest(runner.BASE / "evaluate.py"),
        "configuration": {"seed": 0}, "regime": "realistic", "cohort": "discovery",
        "pool_seed": 42, "matching_seed": 17,
    }
    return {
        "status": "completed_with_failures" if failures else "completed",
        "inputs": inputs,
        "input_fingerprint": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "candidates": [{"name": "A", "feature_id": "f_a", "assay": "RNA", "role": "core"}],
        "results": {"matched": {"assignment": {"cell_ids": ["a", "b"]}}},
        "common_pools": {"cell_ids": {"RNA": ["a"]}}, "graphical_truth": {"activities": {"status": "ok"}},
        "count_calibration": {"status": "passed_declared_checks"}, "failed_integrations": [],
        "failed_diagnostics": ["matched/counts"] if failures else [],
    }


def guidance(*, applicable=False):
    status = "ok" if applicable else "not_applicable"
    return {
        "status": status,
        "prior": {
            "method": "guidance_only_undirected_prior", "status": status,
            "baseline_kind": "exposed_annotation_prior" if applicable else "masked_scored_feature_negative_control",
            "direction": {"status": "not_applicable"}, "edges": [] if applicable else None,
            "reason": None if applicable else "Scored vertices were masked by construction",
        },
        "scoring": {
            "status": status, "direction": {"status": "not_applicable"},
            "adjacency": {"status": "ok", "n_predicted": 0} if applicable else None,
            "coverage": {"n_scored_features": 3, "n_represented_features": 2,
                         "n_scored_pairs": 3, "n_represented_pairs": 1,
                         "n_link_eligible_pairs": 1 if applicable else 0},
        },
        "inputs": {
            "preprocessing_manifest_sha256": "1" * 64, "guidance_sha256": "2" * 64,
            "truth_manifest_sha256": "3" * 64, "world_manifest_sha256": "4" * 64,
            "source_sha256": runner.digest(runner.BASE / "guidance_baseline.py"),
            "matched_count_regimes": ["realistic"],
        },
    }


def twin(first_path, second_path, *, failures=False):
    first, second = (runner.read(path)["inputs"] for path in (first_path, second_path))
    counts = ({"status": "not_evaluable", "reason": "Discovery failed", "discovery_statuses": ["ok", "failed"]}
              if failures else {"status": "ok", "graph_output_identical": True})
    return {
        "status": "completed_with_failed_discovery" if failures else "completed",
        "twin_audit": {
            "status": "verified_exact_public_equivalence", "rho": [0.0, 0.7],
            "base_world": first["world"], "counterpart_world": second["world"], "regime": "realistic",
            "world_manifest_sha256": [item["world_manifest_sha256"] for item in (first, second)],
            "truth_manifest_sha256": [item["truth_manifest_sha256"] for item in (first, second)],
            "public_files": {"observed/realistic/train/RNA.h5ad": "4" * 64},
            "joint_audits": {"assessment": {"joint_activities_differ": True}},
        },
        "comparisons": {"matched": {"matching_identical": True, "scales": {
            "activities": {"status": "ok", "graph_output_identical": True}, "counts": counts}}},
        "report_sha256": [runner.digest(first_path), runner.digest(second_path)],
        "ambiguity_set_metric": {"status": "not_applicable"},
        "success_criterion": "Do not claim discrimination between identical public inputs",
    }


@pytest.mark.parametrize("status", ["completed", "completed_with_failures", "ok", "not_applicable", "completed_with_failed_discovery"])
def test_status_only_stubs_never_unlock_assessment(status):
    with pytest.raises(ValueError, match="schema"):
        runner._terminal_evidence({"status": status})


@pytest.mark.parametrize("factory", [evaluation, guidance])
@pytest.mark.parametrize("status", ["running", "failed", "complete"])
def test_nonterminal_or_other_producer_status_rejected(source_tree, tmp_path, factory, status):
    report = factory(tmp_path / "world") if factory is evaluation else factory()
    report["status"] = status
    with pytest.raises(ValueError, match="not terminal"):
        runner._terminal_evidence(report)


@pytest.mark.parametrize("failures", [False, True])
def test_evaluation_terminal_outcomes_preserved(source_tree, tmp_path, failures):
    report = evaluation(tmp_path / "world", failures=failures)
    result = runner._terminal_evidence(report)
    assert result == {"kind": "evaluation", "status": report["status"], "producer": "evaluate.py"}


@pytest.mark.parametrize("applicable", [False, True])
def test_guidance_terminal_outcomes_preserved(source_tree, applicable):
    report = guidance(applicable=applicable)
    result = runner._terminal_evidence(report)
    assert result["kind"] == "guidance_prior"
    assert result["status"] == report["status"]


def test_evaluation_fingerprint_and_failure_summary_validated(source_tree, tmp_path):
    report = evaluation(tmp_path / "world")
    changed = copy.deepcopy(report)
    changed["inputs"]["configuration"]["seed"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        runner._terminal_evidence(changed)
    changed = copy.deepcopy(report)
    changed["failed_diagnostics"] = ["unreported failure"]
    with pytest.raises(ValueError, match="contradicts"):
        runner._terminal_evidence(changed)


@pytest.mark.parametrize("factory,producer", [(evaluation, "evaluate.py"), (guidance, "guidance_baseline.py")])
def test_changed_producer_source_rejected(source_tree, tmp_path, factory, producer):
    report = factory(tmp_path / "world") if factory is evaluation else factory()
    (source_tree / producer).write_text("# modified producer\n")
    with pytest.raises(ValueError, match="producer source changed"):
        runner._terminal_evidence(report)


@pytest.mark.parametrize("mutation", ["hash", "direction", "no_eligible_pairs", "missing_coverage", "status"])
def test_invalid_guidance_schema_rejected(source_tree, mutation):
    report = guidance()
    if mutation == "hash":
        report["inputs"]["truth_manifest_sha256"] = "z" * 64
    elif mutation == "direction":
        report["prior"]["direction"]["status"] = "ok"
    elif mutation == "no_eligible_pairs":
        report["scoring"]["coverage"]["n_link_eligible_pairs"] = 1
    elif mutation == "missing_coverage":
        del report["scoring"]["coverage"]
    else:
        report["scoring"]["status"] = "ok"
    with pytest.raises(ValueError):
        runner._terminal_evidence(report)


@pytest.mark.parametrize("failures", [False, True])
def test_twin_terminal_outcomes_require_linked_evaluation_reports(source_tree, tmp_path, failures):
    first = write(tmp_path / "first.json", evaluation(tmp_path / "base"))
    second = write(tmp_path / "second.json", evaluation(tmp_path / "counterpart", failures=failures))
    compared = write(tmp_path / "comparison.json", twin(first, second, failures=failures))
    reports, outcomes = runner.validate_development_reports([first, second, compared])
    assert reports[str(compared.resolve())] == runner.digest(compared)
    assert outcomes[str(compared.resolve())]["status"] == runner.read(compared)["status"]
    assert outcomes[str(compared.resolve())]["kind"] == "twin_comparison"
    with pytest.raises(ValueError, match="included, validated evaluation"):
        runner.validate_development_reports([first, compared])


@pytest.mark.parametrize("mutation", ["status", "graph", "matching", "world", "truth", "reference", "rho", "private_truth"])
def test_invalid_twin_schema_or_evidence_links_rejected(source_tree, tmp_path, mutation):
    first = write(tmp_path / "first.json", evaluation(tmp_path / "base"))
    second = write(tmp_path / "second.json", evaluation(tmp_path / "counterpart"))
    report = twin(first, second)
    if mutation == "status":
        report["status"] = "completed_with_failed_discovery"
    elif mutation == "graph":
        report["comparisons"]["matched"]["scales"]["counts"]["graph_output_identical"] = False
    elif mutation == "matching":
        report["comparisons"]["matched"]["matching_identical"] = False
    elif mutation == "world":
        report["twin_audit"]["base_world"] = str(tmp_path / "different")
    elif mutation == "truth":
        report["twin_audit"]["truth_manifest_sha256"][0] = "8" * 64
    elif mutation == "reference":
        report["report_sha256"][0] = "9" * 64
    elif mutation == "rho":
        report["twin_audit"]["rho"] = [0, 0]
    else:
        report["twin_audit"]["joint_audits"]["assessment"]["joint_activities_differ"] = False
    compared = write(tmp_path / "comparison.json", report)
    with pytest.raises(ValueError):
        runner.validate_development_reports([first, second, compared])


def test_empty_duplicate_and_ambiguous_evidence_rejected(source_tree, tmp_path):
    path = write(tmp_path / "report.json", guidance())
    with pytest.raises(ValueError, match="without development"):
        runner.validate_development_reports([])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_development_reports([path, path])
    combined = guidance()
    combined["input_fingerprint"] = "1" * 64
    with pytest.raises(ValueError, match="ambiguous"):
        runner._terminal_evidence(combined)


def test_cli_lock_keeps_na_and_failure_metadata_without_claiming_success(source_tree, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    write(runs / "worlds/extended_bank_manifest.json", {"test": "fixed bank"})
    write(runs / "calibration/empirical_count_profiles.json", {"test": "fixed calibration"})
    monkeypatch.setattr(runner, "RUNS", runs)
    monkeypatch.setattr(runner, "source_pins", lambda: {"test_source.py": "1" * 64})
    failed = write(tmp_path / "failed_evaluation.json", evaluation(tmp_path / "world", failures=True))
    masked = write(tmp_path / "masked_guidance.json", guidance())
    output = tmp_path / "assessment_lock.json"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "lock-assessment", "--development-reports", str(failed), str(masked), "--out", str(output)])
    runner.main()
    lock = runner.read(output)
    assert lock["development_reports"] == {str(path.resolve()): runner.digest(path) for path in (failed, masked)}
    assert lock["development_evidence"][str(masked.resolve())]["status"] == "not_applicable"
    assert lock["development_evidence"][str(failed.resolve())]["status"] == "completed_with_failures"
    assert "not recovery success" in lock["note"]
    runner.check_lock(output)

