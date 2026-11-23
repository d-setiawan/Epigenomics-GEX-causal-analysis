"""Fake JSON fixtures only: no production worlds, fits, probes, or CI tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts/summarize_frozen_suite.py"
spec = importlib.util.spec_from_file_location("frozen_summary_test_module", MODULE_PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


@pytest.fixture
def fixture_plan(tmp_path):
    repo = tmp_path / "repository"
    base = repo / "pipelines/simulation/integration_recoverability"
    base.mkdir(parents=True)
    source = base / "evaluate.py"
    source.write_text("# Mock frozen evaluator, never executed.\n")
    runs = base / "runs"
    specs = [{"world_id": w, "family": f, "motif": m, "draw": d,
              "split": "development" if d == 0 else "assessment"}
             for w, (f, m, d) in summary.expected_worlds().items()]
    bank = {"specifications": specs}
    bank["bank_hash"] = summary.canonical(bank)
    bank_path = put(runs / "worlds/bank.json", bank)
    calibration = put(runs / "calibration.json", {"status": "frozen"})
    stages = []
    for s in specs:
        if s["motif"] == "ambiguity":
            continue
        for regime in summary.REGIMES:
            output = runs / "extended" / s["world_id"] / regime / "primary/evaluation_scglue_scmrdr_seeds0/report.json"
            stages.append({"stage_id": f"primary:{s['world_id']}:{regime}", "world_id": s["world_id"],
                "phase": "D" if s["split"] == "development" else "F", "kind": "pipeline", "regime": regime,
                "arm": "primary", "seeds": [0], "methods": ["scglue", "scmrdr"], "argv": ["mock_pipeline"],
                "artifacts": [{"kind": "evaluation", "path": str(output),
                               "statuses": ["completed", "completed_with_failures"]}]})
    plan = {"repository": str(repo), "base_directory": str(base), "runs_directory": str(runs),
        "assessment_lock": str(runs / "suite/assessment_lock.json"), "stages": stages,
        "development_reports": [], "inputs": {"bank_path": str(bank_path), "bank_sha256": summary.digest(bank_path),
        "bank_hash": bank["bank_hash"], "calibration_path": str(calibration), "calibration_sha256": summary.digest(calibration),
        "source_pins": {str(source.relative_to(repo)): summary.digest(source)}},
        "protocol": {"assessment_specifications": [s["world_id"] for s in specs if s["split"] == "assessment"]}}
    path = runs / "suite/schedule.json"
    plan["schedule_hash"] = summary.canonical(plan)
    put(path, plan)
    for world in summary.SUPPLEMENTAL_WORLDS:
        put(runs / "secondary" / world / "informative/anchor_readout_v1/freeze.json", {
            "schema": "secondary_matching_freeze_v1", "status": "secondary_frozen", "world_id": world,
            "primary_report": str(runs / "extended" / world / "informative/primary/evaluation_scglue_scmrdr_seeds0/report.json"),
            "configuration": {**summary.CI_CONFIGURATION, "test": "rcit"},
            "conditions": [{"id": c} for c in summary.SUPPLEMENTAL_CONDITIONS]})
    return path, plan, {s["world_id"]: s for s in specs}


def adjacency(tp=1, fp=1, truth=2):
    predicted = tp + fp
    return {"precision": tp / predicted if predicted else None, "recall": tp / truth if truth else None,
            "f1": 2 * tp / (predicted + truth) if predicted + truth else None,
            "false_positive_rate": fp / 10, "true_positive": tp, "false_positive": fp,
            "false_negative": truth - tp, "n_predicted": predicted, "n_true": truth}


def make_report(plan, stage, *, skipped=False, ci=None):
    world = Path(plan["runs_directory"]) / "worlds" / stage["world_id"]
    truth_manifest = put(world / "truth/manifest.json", {"fixture": "graph definitions only"})
    manifest = put(world / "manifest.json", {"status": "complete", "files": {
        "truth/discovery.npz": "1" * 64, "truth/manifest.json": summary.digest(truth_manifest)}})
    provenance = {}
    for label in summary.METHODS:
        directory = Path(stage["artifacts"][0]["path"]).parent / label
        entry = {"status": "complete", "feature_count": 9}
        mpath = put(directory / "embedding_manifest.json", entry)
        provenance[label] = {"directory": str(directory), "manifest": entry, "manifest_sha256": summary.digest(mpath)}
    source = str(Path(plan["base_directory"]).relative_to(plan["repository"]) / "evaluate.py")
    inputs = {"configuration": {**summary.CI_CONFIGURATION, "test": ci or stage.get("ci_test", "rcit")},
        "pool_seed": 2718, "matching_seed": 3141, "include_probes": not skipped, "cohort": "discovery",
        "world": str(world), "regime": stage["regime"], "provenance": provenance,
        "world_manifest_sha256": summary.digest(manifest), "truth_manifest_sha256": summary.digest(truth_manifest),
        "truth_sha256": "1" * 64, "evaluation_source_sha256": plan["inputs"]["source_pins"][source]}
    results = {}
    for label in [*summary.METHODS, "paired_anchor_reference", "independent_paired_assessment", "random_reference", "true_state_reference"]:
        result = {"state_matching": {p: {"mean_absolute_difference": .4, "mean_euclidean_distance": .6,
                                        "mae_by_dimension": [.3, .5]} for p in summary.STATE_PAIRS}}
        for scale, score_key in (("activities", "activity_graph_metrics"), ("counts", "count_graph_metrics")):
            result[scale] = {"status": "ok", "edges": [], "n_rows": 20}
            result[score_key] = {"status": "ok", "target": summary.TARGETS[scale], "adjacency": adjacency(),
                "by_assay_pair": {scope: adjacency() for scope in summary.SCOPES if scope != "all"},
                "endpoints": {"precision": .5, "recall": .5, "n_correct": 1},
                "sentinel": {"false_positive_rate_against_current_graph_target": .1,
                             "incident_adjacency_rate": .2, "n_possible_incident_pairs": 10, "n_returned_incident_adjacencies": 2},
                "exact_cross_assay_stress_test": {"pair": adjacency(), "direction": adjacency(0, 0),
                                                  "exact_pair_set": False, "exact_directed_set": False}}
        results[label] = result
    results["independent_activities_with_state"] = {"discovery": {"status": "ok"}, "oracle": {"status": "ok"}}
    probe = {"status": "ok", "rows": [{"status": "ok", "probe": "ridge", "representation": "shared",
        "source_assay": "RNA", "target_assay": "H3K27ac", "target": "H3K27ac_core_residual",
        "relationship": "cross_assay", "n_calibration": 10, "n_assessment": 20,
        "embedding_only": {"r2": -.1, "mse": 1.1}, "state_only": {"r2": -.2, "mse": 1.2},
        "state_plus_embedding": {"r2": -.05, "mse": 1.05}, "mse_gain_over_true_state": .15}],
        "conditional_mean_uncertainty": {"repetitions": 128, "fixture_se": .1}}
    disabled = {"status": "not_run", "reason": "Explicitly disabled; extended evaluation incomplete"}
    report = {"status": "completed_with_failures" if skipped else "completed", "inputs": inputs,
        "input_fingerprint": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "results": results, "failed_diagnostics": ["probes/not_run", "preprocessing_probes/not_run"] if skipped else [],
        "failed_integrations": [], "probes": disabled if skipped else {m: copy.deepcopy(probe) for m in summary.METHODS},
        "preprocessing_probes": disabled if skipped else {"representations": copy.deepcopy(probe)},
        "common_pools": {"cell_ids": {a: [f"{a}_c{i}" for i in range(20)] for a in summary.ASSAYS},
                         "by_assay": {a: {"observed": 30, "selected": 20, "common_before_equalization": 20,
                                          "eligible_by_run": {m: 20 for m in summary.METHODS}} for a in summary.ASSAYS}},
        "graphical_truth": {t: {"status": "ok"} for t in summary.TARGETS.values()},
        "count_calibration": {"status": "passed_declared_checks", "panel_calibration_passed": True, "focal_calibration_passed": True},
        "additional_structure_metrics": {k: {"status": "not_applicable", "reason": "No output/truth contract"}
                                         for k in ("hyperedges", "equivalence_class")}, "limitations": ["fixture limitation"]}
    return report


def stage_for(plan, world="linear_chain_0", regime="realistic"):
    return next(s for s in plan["stages"] if s["world_id"] == world and s["regime"] == regime)


def base_for(stage, specs):
    return summary.context(stage, specs, stage["artifacts"][0]["path"])


def pin_plan(path):
    pin = path.with_name("synthesis_pin.json")
    summary.freeze(path, pin)
    return pin


def release(plan):
    put(plan["assessment_lock"], {k: plan["inputs"][k] for k in ("source_pins", "bank_sha256", "calibration_sha256")} |
        {"development_reports": {}})


def test_freeze_pins_exact_worlds_metrics_paths_and_refuses_overwrite(fixture_plan):
    path, plan, _ = fixture_plan
    pin = pin_plan(path)
    value = summary.read(pin)
    assert len(value["primary_paths"]) == 56
    assert len(value["worlds"]) == 30
    assert value["metric_schema"]["bootstrap"]["seed"] == 61709
    with pytest.raises(FileExistsError):
        summary.freeze(path, pin)
    release(plan)
    with pytest.raises(ValueError, match="retrospective"):
        summary.freeze(path, path.with_name("late_pin.json"))


def test_frozen_source_universe_and_membership_are_enforced(fixture_plan):
    path, plan, _ = fixture_plan
    extra = Path(plan["base_directory"]) / "unexpected.py"
    extra.write_text("# fixture\n")
    with pytest.raises(ValueError, match="source universe"):
        summary.load_schedule(path)
    extra.unlink()
    bank_path = Path(plan["inputs"]["bank_path"])
    bank = summary.read(bank_path)
    bank["specifications"][0]["draw"] = 99
    bank["bank_hash"] = summary.canonical({k: v for k, v in bank.items() if k != "bank_hash"})
    put(bank_path, bank)
    plan["inputs"].update(bank_hash=bank["bank_hash"], bank_sha256=summary.digest(bank_path))
    plan["schedule_hash"] = summary.canonical({k: v for k, v in plan.items() if k != "schedule_hash"})
    put(path, plan)
    with pytest.raises(ValueError, match="membership"):
        summary.load_schedule(path)


@pytest.mark.parametrize("ci", ["kci", "fisherz"])
def test_actual_secondary_skip_shape_does_not_crash(fixture_plan, ci):
    _, plan, specs = fixture_plan
    stage = copy.deepcopy(stage_for(plan))
    stage.update(kind="evaluate", ci_test=ci, argv=["mock_evaluate", "--skip-probes"])
    report = make_report(plan, stage, skipped=True, ci=ci)
    summary.validate_report(report, stage, plan)
    rows, _, components = summary.flatten_report(report, base_for(stage, specs))
    assert {r["status"] for r in rows if r["stage"] in {"probes", "preprocessing_probes"}} == {"planned_skip"}
    assert sum(r["status"] == "planned_skip" for r in components) == 2
    assert summary.saved_probe_metadata(report)["probes"]["all"]["status"] == "not_run"
    assert not [r for r in components if r["status"] == "failed"]


def test_zero_predictions_undefined_not_failed_and_failed_not_empty(fixture_plan):
    _, plan, specs = fixture_plan
    stage = stage_for(plan)
    report = make_report(plan, stage)
    report["results"]["scglue_seed0"]["count_graph_metrics"]["adjacency"] = adjacency(0, 0)
    report["results"]["scmrdr_paper_seed0"]["counts"] = {"status": "failed", "edges": None,
        "error_type": "LinAlgError", "error": "Eigenvalues did not converge"}
    rows, flags, parts = summary.flatten_report(report, base_for(stage, specs))
    select = lambda m, metric: next(r for r in rows if r["method"] == m and r["stage"] == "graph" and r["target"] == "count_measurement_pag" and r["scope"] == "all" and r["metric"] == metric)
    assert select("scglue_seed0", "precision")["status"] == "undefined"
    assert select("scglue_seed0", "recall")["value"] == 0
    assert select("scmrdr_paper_seed0", "precision")["status"] == "failed"
    assert select("scmrdr_paper_seed0", "n_predicted")["value"] is None
    assert len([r for r in parts if r["status"] == "failed"]) == 1
    assert any(r["flag"] == "recorded_graph_failure" for r in flags)
    report["results"]["scglue_seed0"]["count_graph_metrics"]["adjacency"]["precision"] = 0
    with pytest.raises(ValueError, match="zero-denominator"):
        summary.flatten_report(report, base_for(stage, specs))


def test_unrecorded_denominators_probe_metadata_and_missing_statuses(fixture_plan):
    _, plan, specs = fixture_plan
    stage = stage_for(plan)
    report = make_report(plan, stage)
    report["results"]["scglue_seed0"]["state_matching"] = None
    report["probes"]["scglue_seed0"] = {"status": "failed", "error": "fixture failure"}
    rows, _, parts = summary.flatten_report(report, base_for(stage, specs))
    fprs = [r for r in rows if r["metric"] in {"false_positive_rate", "false_positive_rate_against_current_graph_target"}]
    assert all(r["denominator"] is None and r["denominator_status"] == "not_recorded" for r in fprs)
    incidences = [r for r in rows if r["metric"] == "incident_adjacency_rate"]
    assert all(r["denominator"] == 10 for r in incidences)
    assert all(r["status"] == "missing" for r in rows if r["method"] == "scglue_seed0" and r["stage"] == "state_matching")
    assert any(r["component"] == "probes" and r["status"] == "failed" for r in parts)
    assert summary.saved_probe_metadata(report)["preprocessing_probes"]["representations"]["conditional_mean_uncertainty"]["repetitions"] == 128
    assert any(r.get("n_assessment") == 20 and r["metric"] == "mse_gain_over_true_state" for r in rows)


def test_report_configuration_and_manifest_identity_are_not_substituted(fixture_plan):
    _, plan, _ = fixture_plan
    stage = stage_for(plan)
    report = make_report(plan, stage)
    summary.validate_report(report, stage, plan)
    report["inputs"]["provenance"]["scglue_seed0"]["manifest"]["feature_count"] = 99
    report["input_fingerprint"] = hashlib.sha256(json.dumps(report["inputs"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="integration manifest"):
        summary.validate_report(report, stage, plan)


def test_default_never_opens_assessment_files(fixture_plan, monkeypatch):
    path, plan, _ = fixture_plan
    pin = pin_plan(path)
    assessment = stage_for(plan, "linear_chain_1")
    forbidden = Path(assessment["artifacts"][0]["path"])
    put(forbidden, {"must_not_be_opened": True})
    original = summary.read

    def guarded(path):
        if Path(path) == forbidden:
            pytest.fail("Assessment report was opened without flag")
        return original(path)

    monkeypatch.setattr(summary, "read", guarded)
    output, _ = summary.synthesize(path, pin)
    record = next(r for r in output["inventory"] if r["stage_id"] == assessment["stage_id"])
    assert record["status"] == "blocked_assessment_lock"
    assert all(r["value"] is None for r in output["metrics"] if r["split"] == "assessment")
    with pytest.raises(FileNotFoundError):
        summary.synthesize(path, pin, include_assessment=True)


def test_explicit_lock_access_and_default_receipt_suppression(fixture_plan, monkeypatch):
    path, plan, _ = fixture_plan
    pin = pin_plan(path)
    release(plan)
    attempts = path.with_name(path.stem + "_attempts")
    forbidden = put(attempts / "opaque.json", {"not_to_be_read": True})
    original = summary.read

    def guarded(path):
        if Path(path) == forbidden:
            pytest.fail("Mixed receipt opened without assessment permission")
        return original(path)

    monkeypatch.setattr(summary, "read", guarded)
    output, _ = summary.synthesize(path, pin)
    assert output["attempt_receipt_scope"]["status"] == "receipt_inspection_suppressed"
    forbidden.unlink()
    output, _ = summary.synthesize(path, pin, include_assessment=True)
    assert output["assessment_lock"]["sha256"] == summary.digest(plan["assessment_lock"])
    assert not [r for r in output["inventory"] if r["status"] == "blocked_assessment_lock"]


def receipt_for(path, plan, stage, exit_code=1):
    return put(path.with_name(path.stem + "_attempts") / "receipt.json", {
        "schedule_hash": plan["schedule_hash"], "stage_id": stage["stage_id"],
        "argv_sha256": summary.canonical(stage["argv"]), "recorded_unix_ns": 123,
        "exit_code": exit_code, "log": "fixture.log", "note": "preserved failure", "artifact_sha256": {}})


def test_failed_attempt_without_report_and_operator_refusal_separate(fixture_plan):
    path, plan, _ = fixture_plan
    pin = pin_plan(path)
    stage = stage_for(plan)
    receipt_for(path, plan, stage)
    journal = Path(plan["runs_directory"]) / "suite/executions/refusal"
    put(journal / "session.json", {"schedule_hash": plan["schedule_hash"], "phases": ["D"], "status": "started"})
    put(journal / "result.json", {"status": "stopped_validation_failure", "executed_commands": 0})
    output, _ = summary.synthesize(path, pin)
    row = next(r for r in output["inventory"] if r["stage_id"] == stage["stage_id"])
    assert row["status"] == "failed_without_complete_artifacts"
    assert row["failed_recorded_attempts"] == 1
    assert row["attempts"][0]["note"] == "preserved failure"
    assert output["operator_sessions"][0]["saved_result"]["executed_commands"] == 0
    assert all(r["status"] != "failed" for r in output["component_outcomes"])
    assert all(r["status"] == "blocked_operational_failure" for r in output["component_outcomes"] if r["stage_id"] == stage["stage_id"])


def test_simple_probes_are_not_applicable_not_missing(fixture_plan):
    _, plan, specs = fixture_plan
    stage = stage_for(plan)
    base = {**base_for(stage, specs), "split": "simple"}
    report = make_report(plan, stage)
    report.pop("probes")
    report.pop("preprocessing_probes")
    rows, _, components = summary.flatten_report(report, base)
    assert all(r["status"] == "not_applicable" for r in rows if r["stage"] in {"probes", "preprocessing_probes"})
    assert all(r["status"] == "not_applicable" for r in components if r["component"] in {"probes", "preprocessing_probes"})


def test_unreleased_assessment_receipt_rejected(fixture_plan):
    path, plan, _ = fixture_plan
    pin = pin_plan(path)
    receipt_for(path, plan, stage_for(plan, "linear_chain_1"))
    with pytest.raises(ValueError, match="Unauthorized assessment"):
        summary.synthesize(path, pin)


def paired_rows(specs, missing=None):
    result = []
    for world, spec in specs.items():
        if spec["split"] != "assessment":
            continue
        for regime in summary.REGIMES:
            for method in summary.METHODS:
                status = "undefined" if world == missing and method == summary.METHODS[1] else "ok"
                value = None if status != "ok" else (.25 if method == summary.METHODS[0] else .75)
                result.append({"world_id": world, "split": "assessment", "analysis_role": "primary",
                    "regime": regime, "method": method, "stage": "graph", "target": "count_measurement_pag",
                    "scope": "cross_assay", "metric": "precision", "status": status, "value": value})
    return result


def test_world_bootstrap_identical_indices_regimes_and_no_control_intervals(fixture_plan):
    _, _, specs = fixture_plan
    draws = summary.world_bootstrap(specs)
    assert draws[1].shape == (10000, 6, 3)
    assert np.array_equal(draws[1], summary.world_bootstrap(specs)[1])
    differences, estimates = summary.paired_summaries(paired_rows(specs), specs, draws)
    causal = [r for r in estimates if r["population"] == "causal"]
    assert len(causal) == 2 and all(r["interval"] == [.5, .5] for r in causal)
    assert all(r["n_expected"] == 18 and r["n_valid"] == 18 for r in causal)
    controls = [r for r in estimates if r["population"] == "shared_state_controls"]
    assert all(r["n_expected"] == 3 and r["interval"] is None for r in controls)
    assert len(differences) == 42


def test_missing_precision_no_imputation_or_complete_interval(fixture_plan):
    _, _, specs = fixture_plan
    diffs, estimates = summary.paired_summaries(paired_rows(specs, "linear_chain_1"), specs, summary.world_bootstrap(specs))
    row = next(r for r in estimates if r["population"] == "causal")
    assert row["n_valid"] == 17
    assert row["available_pair_mean"] == .5 and row["complete_bank_mean"] is None and row["interval"] is None
    assert row["unavailable_world_ids"] == ["linear_chain_1"]
    assert "varying-world-set" in row["available_pair_label"]
    assert all(r["difference_scmrdr_minus_scglue"] is None for r in diffs if r["world_id"] == "linear_chain_1")


def test_available_mean_does_not_reweight_surviving_strata(fixture_plan):
    _, _, specs = fixture_plan
    rows = paired_rows(specs, "linear_chain_1")
    for row in rows:
        if row["method"] == summary.METHODS[1] and row["status"] == "ok":
            row["value"] = .25 + (1.0 if row["world_id"].startswith("linear_chain") else 0.0)
    _, estimates = summary.paired_summaries(rows, specs, summary.world_bootstrap(specs))
    causal = [r for r in estimates if r["population"] == "causal"]
    assert all(r["available_pair_mean"] == pytest.approx(2 / 17) for r in causal)
    assert all(r["interval"] is None for r in causal)


def test_secondary_skips_remain_visible_in_full_snapshot(fixture_plan):
    path, plan, _ = fixture_plan
    stage = copy.deepcopy(stage_for(plan))
    stage.update(stage_id="secondary_fixture", kind="evaluate", ci_test="kci", argv=["mock_evaluate", "--skip-probes"])
    stage["artifacts"][0]["path"] = str(Path(stage["artifacts"][0]["path"]).parent.parent / "secondary/report.json")
    plan["stages"].append(stage)
    plan["schedule_hash"] = summary.canonical({k: v for k, v in plan.items() if k != "schedule_hash"})
    put(path, plan)
    put(stage["artifacts"][0]["path"], make_report(plan, stage, skipped=True, ci="kci"))
    output, _ = summary.synthesize(path, pin_plan(path))
    inventory = next(r for r in output["inventory"] if r["stage_id"] == "secondary_fixture")
    assert inventory["status"] == "complete_with_failures"
    parts = [r for r in output["component_outcomes"] if r["stage_id"] == "secondary_fixture"]
    assert sum(r["status"] == "planned_skip" for r in parts) == 2
    assert not any(r["status"] == "failed" for r in parts)
    assert all(r["analysis_role"] == "secondary_ci" for r in parts)


def test_changed_artifact_quarantined_against_immutable_attempt(fixture_plan):
    path, plan, _ = fixture_plan
    stage = stage_for(plan)
    report = make_report(plan, stage)
    report_path = put(stage["artifacts"][0]["path"], report)
    receipt_path = receipt_for(path, plan, stage, exit_code=0)
    receipt = summary.read(receipt_path)
    receipt["artifact_sha256"] = {str(report_path): summary.digest(report_path)}
    put(receipt_path, receipt)
    pin = pin_plan(path)
    report["limitations"].append("Changed after immutable receipt")
    put(report_path, report)
    output, _ = summary.synthesize(path, pin)
    row = next(r for r in output["inventory"] if r["stage_id"] == stage["stage_id"])
    assert row["status"] == "invalid"
    assert "immutable attempt receipt" in row["artifacts"][0]["reason"]
    assert all(r["value"] is None for r in output["metrics"] if r["stage_id"] == stage["stage_id"])


def test_secondary_twins_and_all_seed_copies_not_pooled(fixture_plan):
    _, _, specs = fixture_plan
    rows = paired_rows(specs)
    extras = [{**r, "analysis_role": role} for r in rows for role in ("secondary_ci", "twins", "development_diagnostic")]
    expected = summary.paired_summaries(rows, specs, summary.world_bootstrap(specs))
    assert summary.paired_summaries(rows + extras, specs, summary.world_bootstrap(specs)) == expected
    with pytest.raises(ValueError, match="Duplicate primary"):
        summary.paired_summaries(rows + [rows[0]], specs, summary.world_bootstrap(specs))


def test_snapshot_preserves_inputs_and_exports_immutable_outputs(fixture_plan):
    path, plan, _ = fixture_plan
    stage = stage_for(plan)
    report = make_report(plan, stage)
    report_path = put(stage["artifacts"][0]["path"], report)
    before = summary.digest(report_path)
    pin = pin_plan(path)
    output, draws = summary.synthesize(path, pin)
    record = next(r for r in output["reports"] if r["report_path"] == str(report_path))
    assert record["status"] == "ok" and record["saved_inputs"] == report["inputs"]
    assert record["saved_exact_set_stress"]["scglue_seed0"]["counts"]["exact_pair_set"] is False
    assert record["saved_limitations"] == ["fixture limitation"]
    out = path.parent / "new_summary"
    summary.export_snapshot(output, draws, out, markdown=True)
    manifest = summary.read(out / "output_manifest.json")
    assert all(summary.digest(out / name) == digest for name, digest in manifest["files"].items())
    assert np.array_equal(np.load(out / "bootstrap_world_indices.npz")["indices"], draws)
    assert summary.digest(report_path) == before
    with pytest.raises(FileExistsError):
        summary.export_snapshot(output, draws, out)


def test_supplementary_freezes_pending_and_sealed_reports_kept_separate(fixture_plan):
    path, plan, _ = fixture_plan
    pin = pin_plan(path)
    entries = summary.read(pin)["supplemental_evidence"]
    assert len(entries) == 2
    assert all(r["status"] == "pending" for r in summary.supplemental_inventory(entries))
    entry = entries[0]
    directory = Path(entry["report_path"]).parent
    report = {"schema": "secondary_matching_report_v1", "status": "secondary_complete_with_failures",
              "freeze_sha256": entry["freeze_sha256"], "conditions": {c: {} for c in summary.SUPPLEMENTAL_CONDITIONS},
              "failures": ["preserved fake CI failure"]}
    report_path = put(entry["report_path"], report)
    seal = put(entry["artifact_manifest_path"], {"schema": "secondary_matching_artifacts_v1", "status": "secondary_sealed",
               "freeze_sha256": entry["freeze_sha256"], "files": {"secondary_report.json": summary.digest(report_path),
               "freeze.json": entry["freeze_sha256"]}})
    assert summary.supplemental_inventory(entries)[0]["status"] == "invalid"
    put(directory / "attempts/fixture/result.json", {"status": "secondary_complete", "freeze_sha256": entry["freeze_sha256"],
        "report_sha256": summary.digest(report_path), "artifact_manifest_sha256": summary.digest(seal)})
    output, _ = summary.synthesize(path, pin)
    assert output["supplemental_evidence"][0]["saved_report"] == report
    assert output["supplemental_evidence"][0]["status"] == "secondary_complete_with_failures"
    assert all(r["analysis_role"] != "supplemental_anchor_readout" for r in output["metrics"])
    report["failures"] = []
    put(report_path, report)
    assert summary.supplemental_inventory(entries)[0]["status"] == "invalid"
