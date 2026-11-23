#!/usr/bin/env python3
"""Read-only synthesis of frozen reports; no simulation, fitting, CI, or rescoring.

Freeze this helper and its protocol before assessment. Outputs are new snapshots,
never replacements of evaluator reports. Only NumPy is needed beyond stdlib.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

SCRIPT = Path(__file__).resolve()
PROTOCOL = SCRIPT.parents[1] / "SUITE_SYNTHESIS_PROTOCOL.md"
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
STRATA = tuple(itertools.product(("linear", "constrained", "unrestricted"), ("chain", "collider")))
REGIMES = ("realistic", "informative")
METHODS = ("scglue_seed0", "scmrdr_paper_seed0")
SCOPES = ("all", "cross_assay", "within_assay", "H3K27ac--RNA", "H3K4me3--RNA",
          "H3K27ac--H3K4me3", "RNA--RNA", "H3K27ac--H3K27ac", "H3K4me3--H3K4me3")
GRAPH_METRICS = ("precision", "recall", "f1", "false_positive_rate", "true_positive",
                 "false_positive", "false_negative", "n_predicted", "n_true")
TARGETS = {"activities": "activity_projection", "counts": "count_measurement_pag"}
STATE_PAIRS = ("RNA--H3K27ac", "RNA--H3K4me3", "H3K27ac--H3K4me3")
CI_CONFIGURATION = {"alpha": 0.01, "approximation": "hbe", "min_cells": 20,
                    "num_f": 100, "num_f2": 5, "rcit": True, "seed": 0}
SUPPLEMENTAL_WORLDS = ("linear_chain_0", "constrained_chain_0")
SUPPLEMENTAL_CONDITIONS = ("anchor_H3K27ac", "anchor_H3K4me3", "readout_repeat_1", "readout_repeat_2")
BOOTSTRAP = {"replicates": 10000, "seed": 61709, "rng": "numpy.PCG64",
             "quantiles": [0.025, 0.975], "quantile_method": "linear",
             "unit": "paired world, stratified 3 draws within each of 6 strata"}
SCHEMA = {"version": 1, "primary_seed": 0, "primary_test": "rcit", "alpha": 0.01,
          "primary_arm": "primary", "regimes": REGIMES, "strata": STRATA,
          "scopes": SCOPES, "graph_metrics": GRAPH_METRICS, "targets": TARGETS,
          "endpoints": ("precision", "recall", "n_correct"),
          "sentinel": ("false_positive_rate_against_current_graph_target", "incident_adjacency_rate",
                       "n_possible_incident_pairs", "n_returned_incident_adjacencies"),
          "probe_metrics": ("embedding_only.r2", "embedding_only.mse", "state_only.r2", "state_only.mse",
                            "state_plus_embedding.r2", "state_plus_embedding.mse", "mse_gain_over_true_state"),
          "state_metrics": ("mean_absolute_difference", "mean_euclidean_distance"),
          "state_pairs": STATE_PAIRS, "ci_configuration": CI_CONFIGURATION,
          "paired_summary_stages": ("graph", "endpoints", "sentinel", "direct_stress", "state_matching"),
          "statuses": ("ok", "undefined", "failed", "pending", "blocked_assessment_lock",
                       "not_applicable", "planned_skip", "missing", "invalid", "blocked_operational_failure"),
          "bootstrap": BOOTSTRAP}


def expected_worlds():
    result = {}
    for family, motif in STRATA:
        for draw in range(4):
            result[f"{family}_{motif}_{draw}"] = (family, motif, draw)
    result.update({f"shared_state_{draw}": ("linear", "shared_state", draw) for draw in range(4)})
    result.update({f"ambiguity_rho_{rho}": ("linear", "ambiguity", 0) for rho in ("0.0", "0.7")})
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    json.dumps(value, allow_nan=False)  # Reject nonstandard JSON NaN/Infinity.
    return value


def write_new(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(text)


def load_schedule(path):
    plan = read(path)
    if canonical({k: v for k, v in plan.items() if k != "schedule_hash"}) != plan["schedule_hash"]:
        raise ValueError("Frozen schedule hash changed")
    ids = [s["stage_id"] for s in plan["stages"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scheduled stage")
    for name in ("bank", "calibration"):
        if digest(plan["inputs"][name + "_path"]) != plan["inputs"][name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    repository, base = Path(plan["repository"]), Path(plan["base_directory"])
    paper = repository / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
    source_files = list(base.glob("*.py")) + list(paper.glob("*.py")) + list((paper / "upstream").glob("*.py"))
    if {str(p.relative_to(repository)) for p in source_files} != set(plan["inputs"]["source_pins"]):
        raise ValueError("Frozen scientific source universe changed")
    for rel, expected in plan["inputs"]["source_pins"].items():
        if digest(Path(plan["repository"]) / rel) != expected:
            raise ValueError(f"Frozen implementation changed: {rel}")
    bank = read(plan["inputs"]["bank_path"])
    if canonical({k: v for k, v in bank.items() if k != "bank_hash"}) != bank["bank_hash"] or bank["bank_hash"] != plan["inputs"]["bank_hash"]:
        raise ValueError("Frozen bank content hash changed")
    specs = {s["world_id"]: s for s in bank["specifications"]}
    if len(specs) != 30 or len(bank["specifications"]) != 30:
        raise ValueError("Expected the frozen 30-specification bank")
    if set(specs) != set(expected_worlds()):
        raise ValueError("Frozen world IDs differ from the predeclared bank")
    for world, (family, motif, draw) in expected_worlds().items():
        spec = specs[world]
        split = "development" if draw == 0 else "assessment"
        if (spec["family"], spec["motif"], spec["draw"], spec["split"]) != (family, motif, draw, split):
            raise ValueError("Frozen world stratum/draw/split membership changed")
    assessment = [s for s in specs.values() if s["split"] == "assessment"]
    causal = [s for s in assessment if (s["family"], s["motif"]) in STRATA]
    if len(assessment) != 21 or Counter((s["family"], s["motif"]) for s in causal) != {s: 3 for s in STRATA}:
        raise ValueError("Assessment must contain 6 causal strata × 3 worlds plus 3 controls")
    if set(plan["protocol"]["assessment_specifications"]) != {s["world_id"] for s in assessment}:
        raise ValueError("Schedule and bank assessment worlds differ")
    return plan, specs


def primary_paths(plan, specs):
    result = {}
    for s in plan["stages"]:
        if s["kind"] != "pipeline" or s.get("arm") != "primary" or s.get("seeds") != [0]:
            continue
        world = s.get("world_id")
        if world not in specs or specs[world]["motif"] == "ambiguity":
            continue
        if set(s.get("methods", [])) != {"scglue", "scmrdr"}:
            raise ValueError("Primary method set changed")
        expected = Path(plan["runs_directory"]) / "extended" / world / s["regime"] / "primary" / "evaluation_scglue_scmrdr_seeds0/report.json"
        actual = [a["path"] for a in s["artifacts"] if a["kind"] == "evaluation"]
        if actual != [str(expected)]:
            raise ValueError("Primary report must use the exact scheduled seed-0 path")
        key = f"{world}/{s['regime']}"
        if key in result:
            raise ValueError("Duplicate primary world/regime")
        result[key] = str(expected)
    expected_keys = {f"{w}/{r}" for w, s in specs.items() if s["motif"] != "ambiguity" for r in REGIMES}
    if set(result) != expected_keys:
        raise ValueError("Schedule omits or substitutes a primary world/regime report")
    return result


def identity(schedule_path):
    plan, specs = load_schedule(schedule_path)
    return plan, specs, {"schema_version": 1, "helper_sha256": digest(SCRIPT),
        "protocol_sha256": digest(PROTOCOL), "schedule_sha256": digest(schedule_path),
        "schedule_hash": plan["schedule_hash"], "input_pins": plan["inputs"],
        "metric_schema": SCHEMA, "numpy_version": np.__version__,
        "attempts_directory": str(Path(schedule_path).with_name(Path(schedule_path).stem + "_attempts")),
        "executions_directory": str(Path(plan["runs_directory"]) / "suite/executions"),
        "supplemental_evidence": supplemental_inputs(plan),
        "worlds": [{k: s.get(k) for k in ("world_id", "split", "family", "motif", "draw")}
                   for _, s in sorted(specs.items())], "primary_paths": primary_paths(plan, specs)}


def supplemental_inputs(plan):
    result = []
    for world in SUPPLEMENTAL_WORLDS:
        directory = Path(plan["runs_directory"]) / "secondary" / world / "informative/anchor_readout_v1"
        path = directory / "freeze.json"
        freeze = read(path)
        primary = Path(plan["runs_directory"]) / "extended" / world / "informative/primary/evaluation_scglue_scmrdr_seeds0/report.json"
        if (freeze.get("schema"), freeze.get("status"), freeze.get("world_id"), freeze.get("primary_report")) != (
                "secondary_matching_freeze_v1", "secondary_frozen", world, str(primary)):
            raise ValueError("Supplemental input receipt differs from its fixed world/primary report")
        if freeze.get("configuration") != {**CI_CONFIGURATION, "test": "rcit"}:
            raise ValueError("Supplemental input configuration changed")
        if tuple(c["id"] for c in freeze.get("conditions", [])) != SUPPLEMENTAL_CONDITIONS:
            raise ValueError("Supplemental condition set changed")
        result.append({"world_id": world, "regime": "informative", "analysis_role": "supplemental_anchor_readout",
                       "freeze_path": str(path), "freeze_sha256": digest(path),
                       "report_path": str(directory / "secondary_report.json"),
                       "artifact_manifest_path": str(directory / "secondary_artifact_manifest.json")})
    return result


def supplemental_inventory(entries):
    result = []
    for entry in entries:
        row = {**entry, "status": "pending", "reason": "Frozen inputs alone are not completed scientific evidence"}
        directory = Path(entry["report_path"]).parent
        attempts = [{"path": str(p), "sha256": digest(p), "saved_result": read(p)}
                    for p in sorted((directory / "attempts").glob("*/result.json"))]
        row["attempts"] = attempts
        report_path, manifest_path = Path(entry["report_path"]), Path(entry["artifact_manifest_path"])
        if report_path.is_file() or manifest_path.is_file():
            try:
                report, seal = read(report_path), read(manifest_path)
                report_hash, seal_hash = digest(report_path), digest(manifest_path)
                if (seal.get("schema"), seal.get("status"), seal.get("freeze_sha256")) != (
                        "secondary_matching_artifacts_v1", "secondary_sealed", entry["freeze_sha256"]):
                    raise ValueError("Supplemental artifact seal identity differs")
                if seal["files"].get("secondary_report.json") != report_hash or seal["files"].get("freeze.json") != entry["freeze_sha256"]:
                    raise ValueError("Supplemental report/input bytes differ from artifact seal")
                if report.get("schema") != "secondary_matching_report_v1" or report.get("freeze_sha256") != entry["freeze_sha256"]:
                    raise ValueError("Supplemental report identity differs")
                if report.get("status") not in {"secondary_complete", "secondary_complete_with_failures"}:
                    raise ValueError("Supplemental report is not terminal")
                if set(report.get("conditions", {})) != set(SUPPLEMENTAL_CONDITIONS):
                    raise ValueError("Supplemental report omits a condition")
                if not any(r["saved_result"].get("status") == "secondary_complete"
                    and r["saved_result"].get("freeze_sha256") == entry["freeze_sha256"]
                    and r["saved_result"].get("report_sha256") == report_hash
                    and r["saved_result"].get("artifact_manifest_sha256") == seal_hash for r in attempts):
                    raise ValueError("No immutable supplemental completion receipt binds the seal/report")
                row.update(status=report["status"], reason=None, report_sha256=report_hash,
                           artifact_manifest_sha256=seal_hash, saved_report=report, saved_artifact_manifest=seal)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="invalid", reason=f"{type(error).__name__}: {error}")
        elif attempts:
            row.update(status="incomplete_with_attempt_receipts", reason="No sealed report; original attempt outcomes retained")
        result.append(row)
    return result


def freeze(schedule_path, pin_path):
    plan, _, body = identity(schedule_path)
    if Path(plan["assessment_lock"]).exists() or any(
            Path(a["path"]).exists() for s in plan["stages"] if s["phase"] == "F" for a in s["artifacts"]):
        raise ValueError("Refuse retrospective synthesis freeze after assessment artifacts or release lock exist")
    body = json.loads(json.dumps(body))
    body.update(status="frozen_before_assessment", created_utc=datetime.now(timezone.utc).isoformat(),
                assessment_artifacts_present_at_freeze=False)
    body["pin_hash"] = canonical(body)
    write_new(pin_path, body)
    return body


def validate_lock(plan):
    lock = read(plan["assessment_lock"])
    for key in ("source_pins", "bank_sha256", "calibration_sha256"):
        if lock.get(key) != plan["inputs"][key]:
            raise ValueError(f"Assessment lock differs from {key}")
    if set(lock.get("development_reports", {})) != set(plan["development_reports"]):
        raise ValueError("Assessment lock omits scheduled development evidence")
    for path, expected in lock["development_reports"].items():
        if digest(path) != expected:
            raise ValueError("Locked development evidence changed")
    return {"path": plan["assessment_lock"], "sha256": digest(plan["assessment_lock"])}


def context(stage, specs, path):
    world = stage.get("world_id", "")
    spec = specs.get(world, {})
    split = "twins" if spec.get("motif") == "ambiguity" else spec.get("split", stage["phase"])
    role = ("primary" if stage["kind"] == "pipeline" and stage.get("arm") == "primary"
            and stage.get("seeds") == [0] and world in specs and split != "twins" else
            "secondary_ci" if stage.get("ci_test", "rcit") != "rcit" else
            "twins" if split == "twins" else "development_diagnostic")
    seeds = stage.get("seeds", [0, 1, 2] if stage.get("diagnostic_label") == "all_seeds_common_pool" else [0])
    methods = stage.get("methods", ["scglue", "scmrdr"])
    return {"stage_id": stage["stage_id"], "report_path": str(path), "world_id": world,
            "split": split, "family": spec.get("family"), "motif": spec.get("motif"),
            "draw": spec.get("draw"), "regime": stage.get("regime"),
            "arm": stage.get("arm", "primary"), "ci_test": stage.get("ci_test", "rcit"),
            "analysis_role": role, "expected_methods": [f"{'scmrdr_paper' if m == 'scmrdr' else m}_seed{seed}"
                                                        for m in methods for seed in seeds]}


def validate_report(report, stage, plan):
    inputs = report["inputs"]
    if hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest() != report["input_fingerprint"]:
        raise ValueError("Evaluation input fingerprint changed")
    if inputs["regime"] != stage["regime"] or Path(inputs["world"]).name != stage["world_id"]:
        raise ValueError("Evaluation world/regime differs from scheduled outcome")
    if inputs["configuration"] != {**CI_CONFIGURATION, "test": stage.get("ci_test", "rcit")}:
        raise ValueError("Evaluation CI configuration differs from frozen stage")
    if inputs["configuration"]["seed"] != 0 or inputs["pool_seed"] != 2718 or inputs["matching_seed"] != 3141:
        raise ValueError("Evaluation/readout seed changed")
    if inputs["include_probes"] != ("--skip-probes" not in stage["argv"]):
        raise ValueError("Probe scope differs from the scheduled command")
    if inputs["cohort"] != ("train" if stage["phase"] == "simple" else "discovery"):
        raise ValueError("Evaluation cohort differs from the scheduled design")
    expected_source = plan["inputs"]["source_pins"][str(Path(plan["base_directory"]).relative_to(plan["repository"]) / "evaluate.py")]
    if inputs["evaluation_source_sha256"] != expected_source:
        raise ValueError("Evaluation was produced by different source")
    world = Path(plan["runs_directory"]) / "worlds" / stage["world_id"]
    if Path(inputs["world"]).resolve() != world.resolve():
        raise ValueError("Evaluation points outside the scheduled world")
    if digest(world / "manifest.json") != inputs["world_manifest_sha256"]:
        raise ValueError("World manifest changed since evaluation")
    manifest = read(world / "manifest.json")
    if manifest["files"][f"truth/{inputs['cohort']}.npz"] != inputs["truth_sha256"]:
        raise ValueError("Evaluation truth differs from frozen world receipt")
    if digest(world / "truth/manifest.json") != inputs["truth_manifest_sha256"]:
        raise ValueError("Truth definition changed since evaluation")
    for entry in inputs["provenance"].values():
        candidates = [Path(entry["directory"]) / n for n in ("embedding_manifest.json", "manifest.json")]
        if not any(p.is_file() and digest(p) == entry["manifest_sha256"] and read(p) == entry["manifest"] for p in candidates):
            raise ValueError("Referenced integration manifest changed or is missing")


def numeric_rows(base, label, stage, target, scope, block, metric_names, *, status="ok", reason=None, extra=None):
    rows = []
    for name in metric_names:
        value = block.get(name) if status == "ok" else None
        row_status, row_reason = status, reason
        numerator = denominator = None
        if name == "precision":
            numerator, denominator = block.get("true_positive"), block.get("n_predicted")
        elif name == "recall":
            numerator, denominator = block.get("true_positive"), block.get("n_true")
        if status == "ok" and denominator == 0:
            if value is not None:
                raise ValueError("A zero-denominator score must be null, not zero/perfect precision")
            row_status, row_reason = "undefined", "zero_predictions" if name == "precision" else "zero_true_edges"
        elif status == "ok" and value is None:
            row_status, row_reason = "undefined", "undefined_or_not_recorded_in_saved_metric"
        elif value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("Invalid saved numeric metric")
        ratio = name in {"precision", "recall", "false_positive_rate", "false_positive_rate_against_current_graph_target", "incident_adjacency_rate"}
        if name == "incident_adjacency_rate":
            numerator, denominator = block.get("n_returned_incident_adjacencies"), block.get("n_possible_incident_pairs")
        rows.append({**base, "method": label, "stage": stage, "target": target, "scope": scope,
                     "metric": name, "value": value, "numerator": numerator, "denominator": denominator,
                     "denominator_status": ("recorded" if denominator is not None else "not_recorded") if ratio else "not_a_ratio",
                     "status": row_status, "reason": row_reason, **(extra or {})})
    return rows


def probe_outputs(report, collection, expected, applicable=True):
    value = report.get(collection) or {}
    if not value and not applicable:
        return {"all": {"status": "not_applicable", "reason": "Simple design has no independent private probe/assessment cohorts"}}
    if "status" in value:  # Root-level skipped/failed/N/A payload, not a label mapping.
        return {"all": value}
    return {name: value.get(name, {"status": "missing", "reason": "No saved probe component"})
            for name in sorted(set(expected) | set(value))}


def expected_labels(base):
    labels = list(base.get("expected_methods", METHODS))
    labels += ["paired_anchor_reference", "random_reference", "true_state_reference"]
    if base["split"] != "simple":
        labels.append("independent_paired_assessment")
    return labels


def saved_probe_metadata(report):
    return {collection: {label: {k: v for k, v in output.items() if k != "rows"}
                         for label, output in probe_outputs(report, collection, ()).items()}
            for collection in ("probes", "preprocessing_probes")}


def component_inventory(report, base, unavailable=None):
    """One row per planned component, not per metric/cell/edge as a trial."""
    rows = []

    def add(method, component, payload, applicable=True):
        status = unavailable or (payload.get("status", "missing") if applicable else "not_applicable")
        if status == "not_run" and report.get("inputs", {}).get("include_probes") is False:
            status = "planned_skip"
        if status == "partial_failure":
            status = "failed"
        rows.append({**base, "method": method, "component": component, "status": status,
                     "reason": payload.get("reason") or payload.get("error"),
                     "error_type": payload.get("error_type"), "producer_status": payload.get("status")})

    results = report.get("results", {})
    for label in sorted(set(expected_labels(base)) | (set(results) - {"independent_activities_with_state"})):
        result = results.get(label, {})
        for scale in TARGETS:
            add(label, "discovery/" + scale, result.get(scale) or {})
        if label not in {"paired_anchor_reference", "independent_paired_assessment"}:
            for pair in STATE_PAIRS:
                state = (result.get("state_matching") or {}).get(pair)
                add(label, "state_matching/" + pair, {"status": "ok"} if state else {})
    for target in TARGETS.values():
        add("oracle", target, (report.get("graphical_truth") or {}).get(target) or {})
    state_control = results.get("independent_activities_with_state") or {}
    for part in ("discovery", "oracle"):
        add("independent_activities_with_state", part, state_control.get(part) or {}, base["split"] != "simple")
    for collection in ("probes", "preprocessing_probes"):
        expected = base.get("expected_methods", METHODS) if collection == "probes" else ("representations",)
        for label, output in probe_outputs(report, collection, expected, base["split"] != "simple").items():
            add(label, collection, output)
    return rows


def flatten_report(report, base, *, unavailable=None):
    rows, flags = [], []
    results = report.get("results", {})
    labels = (set(results) | set(expected_labels(base))) - {"independent_activities_with_state"}
    for label in sorted(labels):
        result = results.get(label, {})
        for scale, target in TARGETS.items():
            discovery = result.get(scale) or {}
            score = result.get("activity_graph_metrics" if scale == "activities" else "count_graph_metrics") or {}
            status = unavailable or ("failed" if discovery.get("status") == "failed" else
                                    "ok" if score.get("status") == "ok" else score.get("status", "missing"))
            reason = discovery.get("error") or score.get("reason") or ("report_not_available" if unavailable else None)
            if discovery.get("status") == "failed":
                flags.append({**base, "method": label, "flag": "recorded_graph_failure", "scale": scale,
                              "error_type": discovery.get("error_type"), "evidence": reason,
                              "interpretation": "Execution failure; not an empty graph or proof of information loss"})
            for scope in SCOPES:
                block = score.get("adjacency", {}) if scope == "all" else score.get("by_assay_pair", {}).get(scope, {})
                rows += numeric_rows(base, label, "graph", target, scope, block, GRAPH_METRICS, status=status, reason=reason)
            rows += numeric_rows(base, label, "endpoints", target, "all", score.get("endpoints", {}),
                                 SCHEMA["endpoints"], status=status, reason=reason)
            rows += numeric_rows(base, label, "sentinel", target, "sentinel_incident", score.get("sentinel", {}),
                                 SCHEMA["sentinel"], status=status, reason=reason,
                                 extra={"interpretation": score.get("sentinel", {}).get("interpretation")})
            for kind in ("pair", "direction"):
                stress = score.get("exact_cross_assay_stress_test", {})
                rows += numeric_rows(base, label, "direct_stress", "biological_direct_edges", scale + "/" + kind,
                                     stress.get(kind, {}), GRAPH_METRICS, status=status, reason=reason,
                                     extra={"interpretation": "Identifiability stress test, not an expected success criterion"})
        for pair in STATE_PAIRS:
            scores = (result.get("state_matching") or {}).get(pair, {})
            state_status = unavailable or ("ok" if scores else "not_applicable" if label in {
                "paired_anchor_reference", "independent_paired_assessment"} else "missing")
            rows += numeric_rows(base, label, "state_matching", "true_state", pair, scores,
                                 SCHEMA["state_metrics"], status=state_status,
                                 extra={"mae_by_dimension": scores.get("mae_by_dimension")})
    for collection in ("probes", "preprocessing_probes"):
        expected = base.get("expected_methods", METHODS) if collection == "probes" else ("representations",)
        for label, output in probe_outputs(report, collection, expected, base["split"] != "simple").items():
            if not output.get("rows"):
                status = unavailable or output.get("status", "missing")
                if status == "not_run" and report.get("inputs", {}).get("include_probes") is False:
                    status = "planned_skip"
                rows.append({**base, "method": label, "stage": collection, "target": "retention", "scope": "all",
                             "metric": "execution_status", "value": None, "status": status,
                             "reason": output.get("reason") or output.get("error")})
            for probe in output.get("rows", []):
                block = {"mse_gain_over_true_state": probe.get("mse_gain_over_true_state")}
                for pred in ("embedding_only", "state_only", "state_plus_embedding"):
                    block.update({f"{pred}.{m}": probe.get(pred, {}).get(m) for m in ("r2", "mse")})
                rows += numeric_rows(base, label, collection, probe.get("target"), probe.get("source_assay"),
                    block, SCHEMA["probe_metrics"], status=probe.get("status", "missing"), reason=probe.get("error"),
                    extra={k: probe.get(k) for k in ("probe", "representation", "relationship", "target_assay", "n_calibration", "n_assessment")})
                if (probe.get("status") == "ok" and "residual" in str(probe.get("target"))
                        and probe.get("embedding_only", {}).get("r2", 1) <= 0):
                    flags.append({**base, "method": label, "flag": "fixed_probe_nonpositive_residual_r2",
                                  "source_assay": probe.get("source_assay"), "target": probe.get("target"),
                                  "probe": probe.get("probe"), "representation": probe.get("representation"),
                                  "interpretation": "This fitted readout did not outperform the assessment-mean baseline; not proof of absent information or its cause"})
    for assay in ASSAYS:
        pool = (report.get("common_pools") or {}).get("by_assay", {}).get(assay, {})
        rows += numeric_rows(base, "common_pool", "retention", "cells", assay, pool,
                             ("observed", "selected", "common_before_equalization"), status=unavailable or ("ok" if pool else "missing"))
        for label, count in pool.get("eligible_by_run", {}).items():
            rows += numeric_rows(base, label, "retention", "cells", assay, {"native_eligible": count}, ("native_eligible",))
    for label, entry in report.get("inputs", {}).get("provenance", {}).items():
        manifest = entry.get("manifest", {})
        for cohort_assay, item in manifest.get("retention", {}).items():
            rows += numeric_rows(base, label, "retention", "native_representation", cohort_assay, item,
                                 ("n_cells", "n_features", "latent_dimension"))
        if "feature_count" in manifest:
            rows += numeric_rows(base, label, "retention", "gene_union", "all", manifest, ("feature_count",))
    for name in ("hyperedges", "equivalence_class"):
        item = (report.get("additional_structure_metrics") or {}).get(name, {})
        rows.append({**base, "method": "all", "stage": "additional_structure", "target": name, "scope": "all",
                     "metric": "availability", "value": None, "status": unavailable or item.get("status", "missing"), "reason": item.get("reason")})
    for flag, field in (("declared_panel_calibration_failure", "panel_calibration_passed"),
                        ("declared_focal_calibration_failure", "focal_calibration_passed")):
        if report.get("count_calibration", {}).get(field) is False:
            flags.append({**base, "method": "world", "flag": flag,
                          "interpretation": "Declared simulation calibration checks failed; no attribution to integration"})
    if base["split"] != "simple" and not unavailable:
        flags.append({**base, "method": "all", "flag": "source_population_target_caveat",
                      "interpretation": "Source oracle is not a QC-selection-augmented graph; no claim that selection caused any observed discrepancy"})
    return rows, flags, component_inventory(report, base, unavailable)


def world_bootstrap(specs):
    world_ids = [[w for w, s in sorted(specs.items()) if s["split"] == "assessment" and (s["family"], s["motif"]) == stratum]
                 for stratum in STRATA]
    if any(len(w) != 3 for w in world_ids):
        raise ValueError("Bootstrap requires exactly three independent worlds per causal stratum")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP["seed"]))
    return world_ids, rng.integers(0, 3, size=(BOOTSTRAP["replicates"], 6, 3), dtype=np.int8)


def attempt_receipts(plan, directory, include_assessment):
    """Operational receipts only; never use logs to infer scientific scores."""
    if Path(plan["assessment_lock"]).exists() and not include_assessment:
        return {}, {"status": "receipt_inspection_suppressed",
                    "reason": "Opaque receipt names mix splits; assessment lock exists without explicit assessment permission"}
    stages = {s["stage_id"]: s for s in plan["stages"]}
    history = defaultdict(list)
    for path in sorted(Path(directory).glob("*.json")):
        receipt = read(path)
        stage = stages.get(receipt.get("stage_id"))
        if stage is None or receipt.get("schedule_hash") != plan["schedule_hash"]:
            raise ValueError("Attempt receipt belongs to a different schedule")
        if stage["phase"] == "F" and not include_assessment:
            raise ValueError("Unauthorized assessment attempt before explicit release")
        if receipt.get("argv_sha256") != canonical(stage["argv"]):
            raise ValueError("Attempt receipt command differs from frozen stage")
        if not isinstance(receipt.get("exit_code"), int) or isinstance(receipt["exit_code"], bool):
            raise ValueError("Invalid attempt exit code")
        allowed = {a["path"] for a in stage["artifacts"]}
        if not set(receipt.get("artifact_sha256", {})) <= allowed:
            raise ValueError("Attempt receipt contains unscheduled artifacts")
        history[stage["stage_id"]].append({**receipt, "receipt_path": str(path), "receipt_sha256": digest(path),
                                         "failure_class": "nonzero_process_exit" if receipt["exit_code"] else None})
    for attempts in history.values():
        attempts.sort(key=lambda r: (r["recorded_unix_ns"], r["receipt_path"]))
    return dict(history), {"status": "inspected", "directory": str(directory),
                           "scope": "Saved receipts only; absence does not prove no historical/manual attempt"}


def operator_sessions(plan, directory, include_assessment):
    if Path(plan["assessment_lock"]).exists() and not include_assessment:
        return [{"category": "operator_session", "status": "receipt_inspection_suppressed"}]
    result = []
    for path in sorted(Path(directory).glob("*/session.json")):
        session = read(path)
        if session.get("schedule_hash") != plan["schedule_hash"]:
            continue  # Other frozen schedules are not part of this snapshot.
        if "F" in session.get("phases", []) and not include_assessment:
            raise ValueError("Unauthorized assessment operator session before explicit release")
        record = {"category": "operator_session", "session_path": str(path),
                  "session_sha256": digest(path), "saved_session": session}
        outcome = path.with_name("result.json")
        if outcome.is_file():
            record.update(result_sha256=digest(outcome), saved_result=read(outcome))
        result.append(record)
    return result


def paired_summaries(rows, specs, draws):
    group_fields = ("split", "regime", "stage", "target", "scope", "metric")
    grouped = defaultdict(dict)
    for row in rows:
        if row["analysis_role"] != "primary" or row["method"] not in METHODS or row["stage"] not in SCHEMA["paired_summary_stages"]:
            continue
        key = tuple(row.get(k) for k in group_fields)
        cell = grouped[key].setdefault(row["world_id"], {})
        if row["method"] in cell:
            raise ValueError("Duplicate primary metric; never pool all-seed copies")
        cell[row["method"]] = row
    differences, summaries = [], []
    bootstrap_worlds, indices = draws
    for key, worlds in sorted(grouped.items()):
        base = dict(zip(group_fields, key))
        expected = [w for w, s in sorted(specs.items()) if s["split"] == base["split"] and (s["family"], s["motif"]) in STRATA]
        controls = [w for w in worlds if w not in expected]
        for population, world_set in (("causal", expected), ("shared_state_controls", controls)):
            if not world_set:
                continue
            valid = {}
            for world in world_set:
                pair = worlds.get(world, {})
                a, b = (pair.get(m, {}) for m in METHODS)
                okay = all(x.get("status") == "ok" and x.get("value") is not None for x in (a, b))
                value = b["value"] - a["value"] if okay else None
                differences.append({**base, "population": population, "world_id": world,
                    "scglue": a.get("value"), "scmrdr": b.get("value"), "difference_scmrdr_minus_scglue": value,
                    "status": "ok" if okay else "unavailable_pair", "method_statuses": [a.get("status", "missing"), b.get("status", "missing")]})
                if okay:
                    valid[world] = value
            stratum_rows = []
            for family, motif in STRATA:
                ids = [w for w in world_set if specs[w]["family"] == family and specs[w]["motif"] == motif]
                if ids:
                    values = [valid[w] for w in ids if w in valid]
                    stratum_rows.append({"family": family, "motif": motif, "n_expected": len(ids),
                        "valid_world_ids": [w for w in ids if w in valid], "available_pair_mean": float(np.mean(values)) if values else None,
                        "median": float(np.median(values)) if values else None, "range": [min(values), max(values)] if values else None})
            complete = len(valid) == len(world_set)
            summary = {**base, "population": population, "n_expected": len(world_set), "n_valid": len(valid),
                "valid_world_ids": list(valid), "unavailable_world_ids": [w for w in world_set if w not in valid],
                "available_pair_mean": float(np.mean(list(valid.values()))) if valid else None,
                "available_pair_label": "varying-world-set descriptive arithmetic mean; no stratum reweighting",
                "complete_bank_mean": float(np.mean(list(valid.values()))) if complete else None,
                "strata": stratum_rows, "interval": None,
                "interval_reason": "No inference interval for development, controls, small strata, or incomplete 18-world metrics"}
            if base["split"] == "assessment" and population == "causal" and complete and len(world_set) == 18:
                matrix = np.asarray([[valid[w] for w in group] for group in bootstrap_worlds])
                values = matrix[np.arange(6)[None, :, None], indices].mean(axis=(1, 2))
                summary.update(interval=np.quantile(values, BOOTSTRAP["quantiles"], method="linear").tolist(),
                    interval_reason="Approximate pointwise stratified paired-world percentile interval; not a reliability guarantee")
            summaries.append(summary)
    return differences, summaries


def synthesize(schedule_path, pin_path, *, include_assessment=False):
    plan, specs, expected = identity(schedule_path)
    pin = read(pin_path)
    if canonical({k: v for k, v in pin.items() if k != "pin_hash"}) != pin["pin_hash"]:
        raise ValueError("Synthesis pin changed")
    if pin.get("status") != "frozen_before_assessment" or pin.get("assessment_artifacts_present_at_freeze") is not False:
        raise ValueError("Synthesis was not declared frozen before assessment")
    if any(pin.get(k) != json.loads(json.dumps(v)) for k, v in expected.items()):
        raise ValueError("Helper/protocol/schema/input identity changed after synthesis freeze")
    lock = validate_lock(plan) if include_assessment else None
    attempts, attempt_scope = attempt_receipts(plan, expected["attempts_directory"], include_assessment)
    inventory, metrics, reports, flags, components = [], [], [], [], []
    seen_reports = set()
    for stage in plan["stages"]:
        item = {"stage_id": stage["stage_id"], "kind": stage["kind"], "phase": stage["phase"],
                "world_id": stage.get("world_id"), "regime": stage.get("regime"),
                "status": "pending", "artifacts": [], "attempts": attempts.get(stage["stage_id"], []),
                "attempt_inspection": attempt_scope["status"]}
        item["failed_recorded_attempts"] = sum(a["exit_code"] != 0 for a in item["attempts"])
        blocked = stage["phase"] == "F" and not include_assessment
        if stage["kind"] == "not_applicable":
            item.update(status="not_applicable", reason=stage.get("reason"))
        for artifact in stage["artifacts"]:
            path = Path(artifact["path"])
            record = {"kind": artifact["kind"], "path": str(path), "status": "blocked_assessment_lock" if blocked else "pending"}
            if not blocked and not path.is_file() and item["attempts"] and item["attempts"][-1]["exit_code"]:
                record.update(status="blocked_operational_failure",
                              reason="Report unavailable after nonzero stage exit; individual diagnostic execution is unknown")
            payload = None
            if not blocked and path.is_file():
                try:
                    payload = read(path)
                    record.update(sha256=digest(path), producer_status=payload.get("status"), status="ok")
                    if artifact["kind"] == "lock":
                        validate_lock(plan)
                    elif payload.get("status") not in artifact["statuses"]:
                        record.update(status="failed", reason="producer_status_not_accepted", producer_error=payload.get("error"))
                    if any(payload.get(k) != v for k, v in artifact.get("require_values", {}).items()):
                        record.update(status="invalid", reason="declared_artifact_values_differ")
                    if artifact["kind"] == "evaluation" and record["status"] == "ok":
                        validate_report(payload, stage, plan)
                    last_receipt = next((r for r in reversed(item["attempts"]) if str(path) in r.get("artifact_sha256", {})), None)
                    if last_receipt and last_receipt["artifact_sha256"][str(path)] != record["sha256"]:
                        raise ValueError("Artifact changed after its latest immutable attempt receipt")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    record.update(status="invalid", reason=f"{type(error).__name__}: {error}")
                    payload = None
            item["artifacts"].append(record)
            if artifact["kind"] == "evaluation" and str(path) not in seen_reports:
                seen_reports.add(str(path))
                base = context(stage, specs, path)
                base["report_sha256"] = record.get("sha256")
                okay = record["status"] == "ok" and payload is not None
                if okay:
                    base["common_pool_sha256"] = canonical(payload.get("common_pools", {}))
                try:
                    rows, issues, parts = flatten_report(payload if okay else {}, base, unavailable=None if okay else record["status"])
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    record.update(status="invalid", reason=f"Malformed saved metrics: {type(error).__name__}: {error}")
                    okay = False
                    rows, issues, parts = flatten_report({}, base, unavailable="invalid")
                metrics.extend(rows)
                flags.extend(issues)
                components.extend(parts)
                reports.append({**base, "status": record["status"], "producer_status": record.get("producer_status"),
                    "reason": record.get("reason"), "saved_inputs": payload.get("inputs") if okay else None,
                    "saved_failed_diagnostics": payload.get("failed_diagnostics") if okay else None,
                    "saved_failed_integrations": payload.get("failed_integrations") if okay else None,
                    "saved_count_calibration": payload.get("count_calibration") if okay else None,
                    "saved_activity_state_control": payload.get("results", {}).get("independent_activities_with_state") if okay else None,
                    "saved_probe_metadata": saved_probe_metadata(payload) if okay else None,
                    "saved_limitations": payload.get("limitations") if okay else None,
                    "saved_graphical_truth": payload.get("graphical_truth") if okay else None,
                    "saved_additional_structure_metrics": payload.get("additional_structure_metrics") if okay else None,
                    "saved_exact_set_stress": {label: {scale: (r.get(key) or {}).get("exact_cross_assay_stress_test")
                        for scale, key in (("activities", "activity_graph_metrics"), ("counts", "count_graph_metrics"))}
                        for label, r in payload.get("results", {}).items()} if okay else None})
            elif artifact["kind"] in {"comparison", "guidance_prior", "prior"} and payload is not None:
                reports.append({**context(stage, specs, path), "status": record["status"], "sha256": record.get("sha256"),
                                "saved_diagnostic": payload, "analysis_role": "twins" if stage["kind"] == "twin_compare" else "guidance_prior"})
        if blocked:
            item["status"] = "blocked_assessment_lock"
        elif item["artifacts"]:
            statuses = {a["status"] for a in item["artifacts"]}
            item["status"] = ("invalid" if "invalid" in statuses else "failed" if "failed" in statuses else
                              "failed_without_complete_artifacts" if "blocked_operational_failure" in statuses else
                              "pending" if statuses == {"pending"} else "partial" if "pending" in statuses else "ok")
            if item["status"] == "ok" and any(a.get("producer_status") in {
                    "completed_with_failures", "completed_with_failed_discovery"} for a in item["artifacts"]):
                item["status"] = "complete_with_failures"
            if item["status"] in {"pending", "partial"} and item["attempts"]:
                item["status"] = "failed_without_complete_artifacts" if item["attempts"][-1]["exit_code"] else "incomplete_after_zero_exit"
        inventory.append(item)
    draws = world_bootstrap(specs)
    differences, summaries = paired_summaries(metrics, specs, draws)
    count_groups = defaultdict(Counter)
    for row in components:
        key = tuple(row.get(k) for k in ("split", "analysis_role", "regime", "method", "component"))
        count_groups[key][row["status"]] += 1
    component_counts = [{**dict(zip(("split", "analysis_role", "regime", "method", "component"), key)),
                         "n_component_rows": sum(counts.values()), "status_counts": dict(counts)}
                        for key, counts in sorted(count_groups.items())]
    output = {"schema_version": 1, "status": "descriptive_frozen_suite_snapshot", "pin_sha256": digest(pin_path),
        "schedule_sha256": digest(schedule_path), "helper_sha256": digest(SCRIPT), "protocol_sha256": digest(PROTOCOL),
        "created_utc": datetime.now(timezone.utc).isoformat(), "include_assessment": include_assessment,
        "assessment_lock": lock, "worlds": expected["worlds"], "metric_schema": SCHEMA,
        "inventory": inventory, "inventory_counts": dict(Counter(r["status"] for r in inventory)),
        "metrics": metrics, "reports": reports, "paired_differences": differences, "summaries": summaries,
        "component_outcomes": components, "component_outcome_counts": component_counts,
        "attempt_receipt_scope": attempt_scope,
        "operator_sessions": operator_sessions(plan, expected["executions_directory"], include_assessment),
        "supplemental_evidence": supplemental_inventory(expected["supplemental_evidence"]),
        "stage_evidence_flags": flags, "bootstrap": {**BOOTSTRAP, "world_id_order": draws[0]},
        "interpretation": "No causal reliability cutoff, no preferred test/seed, no p-values. Available-pair summaries condition on metric availability.",
        "integrity_scope": "Frozen sources, small manifests, report fingerprints and hashes; does not rehash all data or rerun scoring"}
    return output, draws[1]


def csv_write(path, rows):
    fields = sorted({k for r in rows for k in r})
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def export_snapshot(output, indices, directory, markdown=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "synthesis.json", output)
    csv_write(directory / "metrics.csv", output["metrics"])
    csv_write(directory / "paired_differences.csv", output["paired_differences"])
    csv_write(directory / "stage_status.csv", output["inventory"])
    csv_write(directory / "component_status.csv", output["component_outcomes"])
    np.savez_compressed(directory / "bootstrap_world_indices.npz", indices=indices,
                        world_ids=np.asarray(output["bootstrap"]["world_id_order"]))
    if markdown:
        text = "# Frozen-suite synthesis snapshot\n\nDescriptive reporting only; no reliability threshold or preferred-score selection.\n\n"
        text += "| Stage status | Count |\n| --- | ---: |\n" + "".join(f"| {k} | {v} |\n" for k, v in sorted(output["inventory_counts"].items()))
        text += "\nSee [full synthesis](synthesis.json), [individual metrics](metrics.csv), and [paired differences](paired_differences.csv).\n"
        text += "\nUndefined precision and failed tests remain unavailable. Available-pair means use explicitly varying world sets.\n"
        with (directory / "summary.md").open("x") as f:
            f.write(text)
    hashes = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
    write_new(directory / "output_manifest.json", {"status": "complete", "files": hashes,
                                                   "helper_sha256": digest(SCRIPT), "protocol_sha256": digest(PROTOCOL)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "summarize"))
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--pin", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-assessment", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze(args.schedule, args.pin)
        print(json.dumps({"status": result["status"], "pin_hash": result["pin_hash"]}))
    else:
        if args.output is None:
            parser.error("summarize requires --output NEW_DIRECTORY")
        result, indices = synthesize(args.schedule, args.pin, include_assessment=args.include_assessment)
        export_snapshot(result, indices, args.output, args.markdown)
        print(json.dumps({"status": result["status"], "inventory_counts": result["inventory_counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
