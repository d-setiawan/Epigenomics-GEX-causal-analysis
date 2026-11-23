"""Resumable, serialized benchmark stages; never replace a frozen world.

Run `generate` once per specification, `run-world` on development first, then
freeze the implementation with `lock-assessment` before assessment execution.
Subprocesses release GPU/RAM between methods. Every command has a durable log.
This runner does not infer successful recovery from successful execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
RUNS = BASE / "runs"
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
PYTHON = "/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def new_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing changed record: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def source_pins():
    files = list(BASE.glob("*.py"))
    paper = REPO / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
    files += list(paper.glob("*.py")) + list((paper / "upstream").glob("*.py"))
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(files)}


def command(stage, *args, gpu=False, log_dir=None):
    logs = Path(log_dir or RUNS / "execution_logs")
    logs.mkdir(parents=True, exist_ok=True)
    timestamp = time.time_ns()
    log = logs / f"{timestamp}_{stage}.log"
    cmd = [PYTHON, "-u", *map(str, args)]
    record = {"command": cmd, "cwd": str(REPO), "stage": stage,
              "started_unix": time.time(), "gpu_serialized": gpu}
    new_json(log.with_suffix(".command.json"), record)
    env = dict(os.environ, NUMBA_CACHE_DIR="/tmp/ir_numba_cache", MPLCONFIGDIR="/tmp/ir_mpl_cache",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    lock_path = RUNS / "gpu_training.lock"
    with lock_path.open("a") as lock, log.open("x") as output:
        if gpu:
            fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps({"stage": stage, "log": str(log), "command": cmd}), flush=True)
        result = subprocess.run(cmd, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
    new_json(log.with_suffix(".result.json"), {**record, "returncode": result.returncode,
                                             "finished_unix": time.time(), "log_sha256": digest(log)})
    if result.returncode:
        raise RuntimeError(f"Stage {stage} failed ({result.returncode}); preserved log {log}")


def bank_entry(world_id):
    if world_id == "simple_saturating_chain":
        return {"world_id": world_id, "design": "simple", "split": "development"}
    bank = read(RUNS / "worlds/extended_bank_manifest.json")
    matches = [item for item in bank["specifications"] if item["world_id"] == world_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown frozen specification {world_id}")
    return matches[0]


def generate(world_id):
    spec = bank_entry(world_id)
    root = RUNS / "worlds" / world_id
    bank = validate_bank_manifest() if spec["design"] == "extended" else None
    if root.exists():
        command("validate_world", BASE / "simulator.py", "--out", root, "--validate-only")
        if bank:
            cached, frozen = read(root / "manifest.json"), read(root / "truth/frozen_parameters.json")
            if (frozen["spec"] != spec or frozen["biology"] != bank["biology_parameters"][world_id]
                    or cached["calibration_hash"] != bank["calibration_hash"]):
                raise ValueError("Cached world differs from frozen specification, biology, or calibration")
            for key in ("feature_counts", "cohort_sizes", "regimes"):
                if cached[key] != bank["generation_config"][key]:
                    raise ValueError(f"Cached world differs from frozen {key}")
        return root
    arguments = ([BASE / "simulator.py", "--calibration", RUNS / "calibration/empirical_count_profiles.json",
                  "--out", root, "--design", "simple"] if spec["design"] == "simple" else
                 [BASE / "run_benchmark.py", "generate-frozen-worker", "--world-id", world_id])
    command("generate_world", *arguments)
    if spec["design"] == "extended":
        frozen = read(root / "truth/frozen_parameters.json")
        bank = read(RUNS / "worlds/extended_bank_manifest.json")
        if frozen["biology"] != bank["biology_parameters"][world_id]:
            raise ValueError("Generated biological mechanisms differ from frozen bank")
    return root


def validate_bank_manifest():
    from calibration import canonical_hash
    bank = read(RUNS / "worlds/extended_bank_manifest.json")
    unhashed = {k: v for k, v in bank.items() if k != "bank_hash"}
    if canonical_hash(unhashed) != bank["bank_hash"]:
        raise ValueError("Frozen bank manifest was modified")
    for filename, expected in bank["source_sha256"].items():
        if digest(BASE / filename) != expected:
            raise ValueError(f"Simulator changed after freezing the bank: {filename}")
    calibration = read(RUNS / "calibration/empirical_count_profiles.json")
    if calibration["calibration_hash"] != bank["calibration_hash"]:
        raise ValueError("Calibration differs from the fixed bank")
    if canonical_hash({k: v for k, v in calibration.items() if k != "calibration_hash"}) != calibration["calibration_hash"]:
        raise ValueError("Frozen calibration content was modified")
    return bank


def generate_frozen_worker(world_id):
    """Consume saved bank entries/settings, not a newly selected world bank."""
    from simulator import prepare_world, write_world, validate_world
    bank = validate_bank_manifest()
    calibration = read(RUNS / "calibration/empirical_count_profiles.json")
    spec = bank_entry(world_id)
    if spec["design"] != "extended":
        raise ValueError("Frozen worker requires an extended specification")
    config = bank["generation_config"]
    prepared = prepare_world(spec, calibration, config["feature_counts"], config["calibration_n"],
                              config["background_calibration_n"], config["feature_chunk_size"])
    if prepared["biology"] != bank["biology_parameters"][world_id]:
        raise ValueError("Prepared mechanisms differ from fixed biological parameters")
    reference = None
    if spec.get("motif") == "ambiguity" and spec["rho"] != 0:
        reference = RUNS / "worlds/ambiguity_rho_0.0"
        if not (reference / "manifest.json").exists():
            raise ValueError("Generate the zero-rho ambiguity member first")
    root = RUNS / "worlds" / world_id
    result = write_world(spec, calibration, root, prepared=prepared, visible_reference=reference, **config)
    print(json.dumps({"world_id": world_id, "calibration_status": result["count_calibration_status"],
                      "validation": validate_world(root)}), flush=True)


def check_lock(path):
    lock = read(path)
    if lock["source_pins"] != source_pins():
        raise ValueError("Implementation changed after assessment lock")
    if lock["bank_sha256"] != digest(RUNS / "worlds/extended_bank_manifest.json"):
        raise ValueError("World bank changed after assessment lock")
    if lock["calibration_sha256"] != digest(RUNS / "calibration/empirical_count_profiles.json"):
        raise ValueError("Calibration changed after assessment lock")
    for report, expected in lock["development_reports"].items():
        if digest(report) != expected:
            raise ValueError("Development evidence changed after assessment lock")


def _evidence_hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Development evidence has a missing or invalid SHA-256")
    return value


def _terminal_evidence(report):
    """Validate a producer-specific terminal outcome, never a success string."""
    if not isinstance(report, dict):
        raise ValueError("Development evidence must be a report object")
    markers = {"evaluation": "input_fingerprint" in report,
               "guidance_prior": "prior" in report,
               "twin_comparison": "twin_audit" in report}
    kinds = [kind for kind, present in markers.items() if present]
    if len(kinds) != 1:
        raise ValueError("Unknown or ambiguous development evidence schema")
    kind, status = kinds[0], report.get("status")

    def require(condition, reason):
        if not condition:
            raise ValueError(f"Invalid {kind} development evidence: {reason}")

    try:
        if kind == "evaluation":
            require(status in {"completed", "completed_with_failures"}, "not terminal")
            inputs = report["inputs"]
            require(isinstance(inputs, dict) and isinstance(inputs["configuration"], dict), "missing input configuration")
            for key in ("world_manifest_sha256", "truth_manifest_sha256", "truth_sha256", "evaluation_source_sha256"):
                _evidence_hash(inputs[key])
            require(inputs["evaluation_source_sha256"] == digest(BASE / "evaluate.py"), "producer source changed")
            _evidence_hash(report["input_fingerprint"])
            require(report["input_fingerprint"] == hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                    "input fingerprint differs from recorded inputs")
            require(isinstance(inputs["world"], str) and bool(inputs["world"]), "missing world")
            require(inputs["regime"] in {"realistic", "informative", "sparse"} and inputs["cohort"] in {"train", "discovery"},
                    "unknown count regime or scoring cohort")
            require(isinstance(report["candidates"], list) and bool(report["candidates"]), "missing scored candidates")
            for key in ("results", "common_pools", "graphical_truth", "count_calibration"):
                require(isinstance(report[key], dict) and bool(report[key]), f"missing {key}")
            for key in ("failed_integrations", "failed_diagnostics"):
                require(isinstance(report[key], list), f"missing {key}")
            has_failures = bool(report["failed_integrations"] or report["failed_diagnostics"])
            require((status == "completed_with_failures") == has_failures, "failure status contradicts recorded outcomes")
            producer = "evaluate.py"
        elif kind == "guidance_prior":
            require(status in {"ok", "not_applicable"}, "not terminal")
            prior, scoring, inputs = report["prior"], report["scoring"], report["inputs"]
            require(prior["method"] == "guidance_only_undirected_prior", "unknown prior method")
            require(prior["status"] == scoring["status"] == status, "contradictory terminal statuses")
            require(prior["baseline_kind"] in {"exposed_annotation_prior", "masked_scored_feature_negative_control"},
                    "missing baseline interpretation")
            require(prior["direction"]["status"] == scoring["direction"]["status"] == "not_applicable",
                    "annotation prior cannot assert causal directions")
            coverage = scoring["coverage"]
            for key in ("n_scored_features", "n_represented_features", "n_scored_pairs", "n_represented_pairs", "n_link_eligible_pairs"):
                require(type(coverage[key]) is int and coverage[key] >= 0, f"invalid {key}")
            require(coverage["n_link_eligible_pairs"] <= coverage["n_represented_pairs"] <= coverage["n_scored_pairs"],
                    "pair coverage is inconsistent")
            require(coverage["n_represented_features"] <= coverage["n_scored_features"], "feature coverage is inconsistent")
            if status == "not_applicable":
                require(coverage["n_link_eligible_pairs"] == 0 and prior["edges"] is None and scoring["adjacency"] is None,
                        "N/A must not claim an evaluated pair universe")
                require(isinstance(prior["reason"], str) and bool(prior["reason"]), "N/A needs a reason")
            else:
                require(coverage["n_link_eligible_pairs"] > 0 and isinstance(prior["edges"], list)
                        and scoring["adjacency"]["status"] == "ok", "missing eligible adjacency outcome")
            for key in ("preprocessing_manifest_sha256", "guidance_sha256", "truth_manifest_sha256", "world_manifest_sha256", "source_sha256"):
                _evidence_hash(inputs[key])
            require(inputs["source_sha256"] == digest(BASE / "guidance_baseline.py"), "producer source changed")
            require(isinstance(inputs["matched_count_regimes"], list) and bool(inputs["matched_count_regimes"]),
                    "no count regime bound to the scored world")
            producer = "guidance_baseline.py"
        else:
            require(status in {"completed", "completed_with_failed_discovery"}, "not terminal")
            audit, comparisons = report["twin_audit"], report["comparisons"]
            require(audit["status"] == "verified_exact_public_equivalence" and audit["rho"] == [0.0, 0.7],
                    "missing declared observational-equivalence audit")
            require(isinstance(audit["public_files"], dict) and bool(audit["public_files"]), "missing public-input equality receipts")
            for value in audit["public_files"].values():
                _evidence_hash(value)
            require(audit["joint_audits"]["assessment"]["joint_activities_differ"] is True,
                    "hidden paired structures do not differ")
            for key in ("world_manifest_sha256", "truth_manifest_sha256"):
                require(isinstance(audit[key], list) and len(audit[key]) == 2, f"missing {key}")
                for value in audit[key]:
                    _evidence_hash(value)
            require(report["ambiguity_set_metric"]["status"] == "not_applicable", "invented ambiguity-set recovery")
            require(isinstance(report["success_criterion"], str) and bool(report["success_criterion"]), "missing interpretation")
            require(isinstance(report["report_sha256"], list) and len(report["report_sha256"]) == 2,
                    "comparison must identify two evaluation reports")
            for value in report["report_sha256"]:
                _evidence_hash(value)
            require(len(set(report["report_sha256"])) == 2, "comparison needs two different truth reports")
            require(isinstance(comparisons, dict) and bool(comparisons), "missing comparison outcomes")
            any_failed = False
            for comparison in comparisons.values():
                require(comparison["matching_identical"] is True, "matching differs across observational twins")
                require(set(comparison["scales"]) == {"activities", "counts"}, "missing comparison scale")
                for outcome in comparison["scales"].values():
                    if outcome["status"] == "ok":
                        require(outcome["graph_output_identical"] is True, "graphs differ across observational twins")
                    else:
                        require(outcome["status"] == "not_evaluable" and isinstance(outcome["reason"], str)
                                and isinstance(outcome["discovery_statuses"], list) and len(outcome["discovery_statuses"]) == 2
                                and outcome["discovery_statuses"] != ["ok", "ok"], "invalid failed-discovery outcome")
                        any_failed = True
            require((status == "completed_with_failed_discovery") == any_failed, "failure status contradicts comparison outcomes")
            producer = "twin_controls.py"
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed {kind} development evidence: {error}") from error
    return {"kind": kind, "status": status, "producer": producer}


def validate_development_reports(paths):
    """Lock reviewed outcomes; twin comparisons must reference included reports."""
    reports, outcomes, payloads = {}, {}, {}
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved in reports:
            raise ValueError("Duplicate development evidence path")
        payload = read(path)
        outcomes[resolved] = _terminal_evidence(payload)
        reports[resolved] = digest(path)
        payloads[resolved] = payload
    if not reports:
        raise ValueError("Cannot lock without development evidence")
    evaluations = {reports[path]: payloads[path] for path, item in outcomes.items() if item["kind"] == "evaluation"}
    for path, item in outcomes.items():
        if item["kind"] != "twin_comparison":
            continue
        twin = payloads[path]
        audit = twin["twin_audit"]
        try:
            linked = [evaluations[value]["inputs"] for value in twin["report_sha256"]]
        except KeyError as error:
            raise ValueError("Twin comparison must reference two included, validated evaluation reports") from error
        for index, world_key in enumerate(("base_world", "counterpart_world")):
            inputs = linked[index]
            if (Path(inputs["world"]).resolve() != Path(audit[world_key]).resolve()
                    or inputs["regime"] != audit["regime"]
                    or inputs["world_manifest_sha256"] != audit["world_manifest_sha256"][index]
                    or inputs["truth_manifest_sha256"] != audit["truth_manifest_sha256"][index]):
                raise ValueError("Twin comparison references a different evaluation world or truth")
        for key in ("configuration", "pool_seed", "matching_seed", "cohort", "evaluation_source_sha256"):
            if linked[0][key] != linked[1][key]:
                raise ValueError(f"Twin comparison evaluation {key} differs")
    return reports, outcomes


def _scmrdr_source_hashes():
    core = REPO / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
    return {"wrapper": digest(BASE / "scmrdr_adapter.py"),
            "core": digest(core / "adapter.py"),
            "vendored": {name: digest(core / "upstream" / name)
                         for name in ("model.py", "loss.py", "LICENSE")}}


def _scmrdr_versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "torch", "scanpy", "anndata", "scikit-misc")}


def _input_hashes(directory):
    return {assay: digest(Path(directory) / f"{assay}.h5ad") for assay in ASSAYS}


def _recorded_input_hashes(records, description):
    if not isinstance(records, dict) or set(records) != set(ASSAYS):
        raise ValueError(f"Missing/incomplete {description} input provenance")
    hashes = {assay: records[assay].get("sha256") for assay in ASSAYS}
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError(f"Missing {description} input hashes")
    return hashes


def _reuse_scmrdr(run, seed, train_dir, gene_transform, excluded, limit=2000,
                  validation_dir=None, device=None):
    """Bind a cached fit to its actual data, transform, configuration and code."""
    run, gene_transform = Path(run), Path(gene_transform)
    manifest = read(run / "manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("seed") != seed
            or manifest.get("method") != "scmrdr_paper_faithful"):
        raise ValueError("Incomplete/different scMRDR fit")
    if manifest.get("train_limit") != limit:
        raise ValueError("scMRDR training cell limit changed")
    if _recorded_input_hashes(manifest.get("train_inputs"), "training") != _input_hashes(train_dir):
        raise ValueError("scMRDR training inputs changed")
    if validation_dir is None:
        if manifest.get("validation_inputs", "missing") is not None:
            raise ValueError("scMRDR validation mode changed or is missing")
    elif (_recorded_input_hashes(manifest.get("validation_inputs"), "validation")
          != _input_hashes(validation_dir)):
        raise ValueError("scMRDR external validation inputs changed")
    if manifest.get("gene_manifest_sha256") != digest(gene_transform / "manifest.json"):
        raise ValueError("scMRDR fit is linked to a different gene transform")
    recorded_transform = Path(manifest["gene_transform_dir"]) / "manifest.json"
    if not recorded_transform.is_file() or digest(recorded_transform) != manifest["gene_manifest_sha256"]:
        raise ValueError("Recorded scMRDR gene transform is missing or changed")
    from scmrdr_adapter import _reuse_gene_manifest
    from gene_features import GeneConfig
    gene = _reuse_gene_manifest(gene_transform, {a: Path(train_dir) / f"{a}.h5ad" for a in ASSAYS},
                                GeneConfig(), excluded, limit)
    summary = read(run / "model/fit_summary.json")
    from scmrdr_adapter import SCmRDRConfig
    from dataclasses import asdict
    expected = asdict(SCmRDRConfig(seed=seed, device=device or manifest["config"]["device"]))
    expected = json.loads(json.dumps(expected))
    if manifest["config"] != expected or summary.get("config") != expected:
        raise ValueError("scMRDR fit differs from frozen primary hyperparameters")
    if (summary.get("method") != "scmrdr_paper_faithful"
            or set(summary.get("modalities", [])) != set(ASSAYS)
            or summary.get("n_features") != len(gene["feature_names"])
            or manifest.get("feature_count") != len(gene["feature_names"])):
        raise ValueError("scMRDR fit summary and gene universe disagree")
    for assay in ASSAYS:
        retained = manifest["train_inputs"][assay]["retained_cells"]
        fitted = summary["training_rows"][assay]
        held_out = summary["validation_rows"][assay]
        if min(retained, fitted, held_out) < 1:
            raise ValueError("Missing/nonpositive scMRDR fit cohort sizes")
        if fitted + (held_out if validation_dir is None else 0) != retained:
            raise ValueError("scMRDR training/validation sizes disagree with retained cells")
    if digest(run / "model/checkpoint.pt") != manifest["checkpoint_sha256"]:
        raise ValueError("scMRDR checkpoint changed")
    sources = _scmrdr_source_hashes()
    if manifest.get("adapter_source_sha256") != sources["wrapper"]:
        raise ValueError("scMRDR benchmark wrapper changed since the cached fit")
    if summary["source"].get("adapter_sha256") != sources["core"]:
        raise ValueError("scMRDR implementation changed since the cached fit")
    if summary["source"].get("vendored_sha256") != sources["vendored"]:
        raise ValueError("scMRDR vendored model/loss source changed since the cached fit")
    if manifest.get("versions") != _scmrdr_versions():
        raise ValueError("scMRDR software versions changed since the cached fit")
    return manifest


def _scmrdr_encoding_context(run, cohort, source, limit):
    """Validate an export's checkpoint, source cohort, limit and frozen code."""
    run, source = Path(run), Path(source)
    if cohort not in {"train", "val", "discovery", "probe", "assessment"}:
        raise ValueError("Unknown scMRDR cohort")
    manifest = read(run / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("method") != "scmrdr_paper_faithful":
        raise ValueError("A completed scMRDR fit is required before using embeddings")
    checkpoint_hash = digest(run / "model/checkpoint.pt")
    if checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise ValueError("scMRDR encoding checkpoint changed")
    gene_path = Path(manifest["gene_transform_dir"]) / "manifest.json"
    if digest(gene_path) != manifest.get("gene_manifest_sha256"):
        raise ValueError("scMRDR encoding gene transform changed")
    sources = _scmrdr_source_hashes()
    summary = read(run / "model/fit_summary.json")
    if (manifest.get("adapter_source_sha256") != sources["wrapper"]
            or summary["source"].get("adapter_sha256") != sources["core"]
            or summary["source"].get("vendored_sha256") != sources["vendored"]):
        raise ValueError("scMRDR encoding source implementation changed")
    versions = _scmrdr_versions()
    if manifest.get("versions") != versions:
        raise ValueError("scMRDR encoding software versions changed")
    hashes = _input_hashes(source)
    if cohort == "train":
        audit = manifest
        if limit != manifest.get("train_limit"):
            raise ValueError("scMRDR training encoding limit changed")
        if _recorded_input_hashes(manifest.get("train_inputs"), "training") != hashes:
            raise ValueError("scMRDR training encoding inputs changed")
        audit_path = run / "manifest.json"
    else:
        audit_path = run / cohort / "encoding_manifest.json"
        audit = read(audit_path)
        if (audit.get("cohort") != cohort or audit.get("limit", "missing") != limit
                or audit.get("post_fit_only") is not True):
            raise ValueError("Cached scMRDR encoding cohort/limit/protocol changed")
        if audit.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Cached scMRDR encoding used a different checkpoint")
        if _recorded_input_hashes(audit.get("inputs"), "encoding") != hashes:
            raise ValueError("Cached scMRDR encoding inputs changed")
    return {"method": "scmrdr_paper_faithful", "cohort": cohort, "limit": limit,
            "checkpoint_sha256": checkpoint_hash,
            "fit_manifest_sha256": digest(run / "manifest.json"),
            "gene_manifest_sha256": digest(gene_path),
            "encoding_manifest_sha256": digest(audit_path),
            "input_sha256": hashes, "source_sha256": sources, "versions": versions}


def _verified_scmrdr_npzs(run, cohort, source, limit):
    """Check dimensions, finiteness and exact source IDs before sealing files."""
    import anndata as ad
    import numpy as np

    run = Path(run)
    manifest = read(run / "manifest.json")
    audit = (manifest["train_inputs"] if cohort == "train" else
             read(run / cohort / "encoding_manifest.json")["inputs"])
    files = {}
    for assay in ASSAYS:
        data = ad.read_h5ad(Path(source) / f"{assay}.h5ad", backed="r")
        try:
            ids = data.obs_names.astype(str).tolist()
        finally:
            data.file.close()
        if limit is not None:
            ids = ids[:limit]
        dropped = audit[assay]["dropped_cell_ids"]
        if (len(ids) != len(set(ids)) or len(dropped) != len(set(dropped))
                or not set(dropped).issubset(ids)):
            raise ValueError("Invalid source or dropped cell IDs in scMRDR export")
        drop_set = set(dropped)
        expected_ids = [cell for cell in ids if cell not in drop_set]
        path = run / cohort / f"{assay}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"cell_ids", "shared", "private"}:
                raise ValueError("Unknown/missing scMRDR embedding arrays")
            if saved["cell_ids"].tolist() != expected_ids or not expected_ids:
                raise ValueError("scMRDR exported IDs differ from their source/retention audit")
            if len(expected_ids) != audit[assay]["retained_cells"]:
                raise ValueError("scMRDR export retention count disagrees with its audit")
            for key, dimension in (("shared", "latent_dim_shared"), ("private", "latent_dim_specific")):
                values = saved[key]
                if (values.shape != (len(expected_ids), manifest["config"][dimension])
                        or not np.isfinite(values).all()):
                    raise ValueError("Invalid scMRDR embedding dimensions/values")
        files[f"{assay}.npz"] = digest(path)
    return files


def _receipt(path, identity, files, *, create=False, origin=None):
    """Immutable post-export receipt; never claim pre-receipt integrity."""
    path = Path(path)
    if path.exists():
        saved = read(path)
        if (saved.get("schema_version") != 1 or saved.get("status") != "complete"
                or saved.get("identity") != identity or saved.get("files") != files
                or not saved.get("verified_at_utc") or not saved.get("verification_origin")):
            raise ValueError(f"Cached embedding receipt changed or is incomplete: {path}")
        return saved
    if not create or origin not in {"new_export", "preexisting_simple_train"}:
        raise ValueError(f"Missing immutable embedding receipt; refusing unknown cached output: {path}")
    value = {"schema_version": 1, "status": "complete", "identity": identity, "files": files,
             "verified_at_utc": datetime.now(timezone.utc).isoformat(),
             "verification_origin": origin,
             "integrity_scope": "First post-export orchestration verification; hashes protect subsequent reuse, not the unrecorded interval before this receipt",
             "receipt_writer_sha256": digest(Path(__file__))}
    new_json(path, value)
    return value


def _verify_scmrdr_encoding(run, cohort, source, limit, *, newly_exported=False,
                            allow_preexisting_simple_train=False):
    context = _scmrdr_encoding_context(run, cohort, source, limit)
    files = _verified_scmrdr_npzs(run, cohort, source, limit)
    bootstrap = allow_preexisting_simple_train and cohort == "train"
    origin = "new_export" if newly_exported else "preexisting_simple_train" if bootstrap else None
    return _receipt(Path(run) / cohort / "artifact_receipt.json", context, files,
                    create=newly_exported or bootstrap, origin=origin)


def _public_embedding_manifest(run, method, cohorts):
    """Seal public embeddings for evaluator/twin reuse, excluding private views."""
    run, cohorts = Path(run), tuple(cohorts)
    if cohorts not in (("train",), ("train", "val", "discovery")):
        raise ValueError("Public embedding scope must be simple train or extended train/val/discovery")
    files, receipts = {}, {}
    if method == "scmrdr":
        fit = read(run / "manifest.json")
        checkpoint = digest(run / "model/checkpoint.pt")
        if fit.get("status") != "complete" or checkpoint != fit.get("checkpoint_sha256"):
            raise ValueError("Incomplete/changed scMRDR public checkpoint")
        for cohort in cohorts:
            path = run / cohort / "artifact_receipt.json"
            receipt = read(path)
            expected = {f"{a}.npz" for a in ASSAYS}
            if (receipt.get("status") != "complete" or set(receipt.get("files", {})) != expected
                    or receipt.get("identity", {}).get("checkpoint_sha256") != checkpoint
                    or receipt["identity"].get("fit_manifest_sha256") != digest(run / "manifest.json")):
                raise ValueError("Incomplete or unrelated scMRDR public export receipt")
            for relative, saved_hash in receipt["files"].items():
                if digest(run / cohort / relative) != saved_hash:
                    raise ValueError("scMRDR public embedding changed after its export receipt")
                files[f"{cohort}/{relative}"] = saved_hash
            receipts[str(path.relative_to(run))] = digest(path)
    elif method == "scglue":
        fit = read(run / "model_manifest.json")
        exported = read(run / "embedding_manifest.json")
        checkpoint = digest(run / "scglue_model.dill")
        if (fit.get("status") != "complete" or exported.get("status") != "complete"
                or checkpoint != fit.get("checkpoint_sha256")
                or checkpoint != exported.get("checkpoint_sha256")
                or exported["inputs"].get("model_manifest_sha256") != digest(run / "model_manifest.json")):
            raise ValueError("Incomplete/unrelated scGLUE checkpoint and embeddings")
        for cohort in cohorts:
            for assay in ASSAYS:
                relative = f"{cohort}/{assay}.npz"
                if exported.get("files", {}).get(relative) != digest(run / relative):
                    raise ValueError("Missing/changed scGLUE public embedding")
                files[relative] = exported["files"][relative]
        receipts["embedding_manifest.json"] = digest(run / "embedding_manifest.json")
    else:
        raise ValueError("Unknown integration method for public embedding receipt")
    path = run / "public_embedding_manifest.json"
    identity = {"method": method, "cohorts": list(cohorts),
                "checkpoint_sha256": checkpoint, "export_receipts": receipts}
    if path.exists():
        saved = read(path)
        if (saved.get("schema_version") != 1 or saved.get("status") != "complete"
                or saved.get("checkpoint_sha256") != checkpoint or saved.get("files") != files
                or saved.get("identity") != identity or not saved.get("verified_at_utc")):
            raise ValueError("Public embedding manifest changed or belongs to another export")
        return saved
    value = {"schema_version": 1, "status": "complete", "checkpoint_sha256": checkpoint,
             "files": files, "identity": identity,
             "verified_at_utc": datetime.now(timezone.utc).isoformat(),
             "verification_origin": "first_post_export_orchestration_verification",
             "integrity_scope": "Not pre-fit provenance; records validated exported files at receipt creation",
             "receipt_writer_sha256": digest(Path(__file__))}
    new_json(path, value)
    return value


def run_world(world_id, regime, seeds, *, device="cuda", arm="primary", assessment_lock=None,
               allow_failed_calibration=False, methods=("scglue", "scmrdr")):
    spec = bank_entry(world_id)
    if spec["split"] == "assessment":
        if assessment_lock is None:
            raise ValueError("Assessment needs a frozen lock, after development review")
        check_lock(assessment_lock)
    world = RUNS / "worlds" / world_id
    if not world.exists():
        raise FileNotFoundError("Generate this frozen specification before running integration")
    # Direct run-world invocations get the same source/world/cache checks.
    generate(world_id)
    manifest = read(world / "manifest.json")
    if manifest["status"] != "complete" or regime not in manifest["regimes"]:
        raise ValueError("Missing/incomplete world or count regime")
    if manifest["count_calibration_status"] != "passed_declared_checks" and not allow_failed_calibration:
        raise ValueError("Count calibration failed; report it before any explicitly labeled diagnostic continuation")
    simple = spec["design"] == "simple"
    root = RUNS / "simple" / regime if simple else RUNS / "extended" / world_id / regime / arm
    if simple and arm != "primary":
        raise ValueError("Ablations are not part of the simple study")
    if spec.get("motif") == "ambiguity" and spec.get("rho") != 0:
        raise ValueError("Twin must reuse first member's fitted models exactly; use the separate twin evaluator")
    root.mkdir(parents=True, exist_ok=True)
    new_json(root / "benchmark_scope.json", {
        "world_manifest_sha256": digest(world / "manifest.json"), "world_id": world_id,
        "regime": regime, "arm": arm, "split": spec["split"],
        "count_calibration_status": manifest["count_calibration_status"],
        "interpretation": ("passed_declared_count_realism_checks" if manifest["count_calibration_status"] == "passed_declared_checks"
                           else "diagnostic_only: failed count calibration; not a validated-realism benchmark"),
        "allow_failed_calibration": allow_failed_calibration})
    # Evaluator supplies feature exclusion IDs only. No roles, states, edges,
    # hidden counts or partner IDs enter either integration process.
    candidates = read(world / "truth/manifest.json")["candidates"]
    excluded = [item["feature_id"] for item in candidates] if arm == "background_only" else []
    from preprocessing import PreprocessingConfig
    from dataclasses import asdict
    config = PreprocessingConfig(
        guidance_excluded_feature_ids=[item["feature_id"] for item in candidates],
        excluded_feature_ids=({a: [x["feature_id"] for x in candidates if x["assay"] == a] for a in ASSAYS}
                              if excluded else {}),
        guidance_mode="shuffled" if arm == "shuffled_guidance" else "normal")
    pp_config = root / "scglue_preprocessing_config.json"
    if pp_config.exists():
        if asdict(PreprocessingConfig(**read(pp_config))) != asdict(config):
            raise ValueError("Saved preprocessing settings differ from requested protocol")
    else:
        new_json(pp_config, asdict(config))
    exclusion_file = root / "excluded_features.json"
    new_json(exclusion_file, excluded)
    observed = world / "observed" / regime
    preprocess = root / "scglue_preprocessing"
    if "scglue" in methods:
        command("preprocess_fit", BASE / "preprocessing.py", "fit", "--input-dir", observed / "train",
                "--out", preprocess, "--config", pp_config)
        if not simple:
            for cohort in ("val", "discovery", "probe", "assessment"):
                source = observed / cohort if cohort in {"val", "discovery"} else world / "private_inputs" / regime / cohort
                command("preprocess_transform", BASE / "preprocessing.py", "transform", "--input-dir", source,
                        "--fit-dir", preprocess, "--cohort", cohort)
    embeddings = []
    for seed in seeds:
        for method in methods:
            destination = root / f"{method}_seed{seed}"
            embeddings.append(destination)
            if method == "scglue":
                command("scglue_fit_encode", BASE / "scglue_adapter.py", "fit", "--preprocessing-dir", preprocess,
                        "--guidance", preprocess / "guidance.graphml.gz", "--output", destination,
                        "--seed", seed, "--device", device, gpu=device == "cuda")
            else:
                if arm == "shuffled_guidance":
                    raise ValueError("scMRDR has no guidance graph; label this ablation not applicable rather than refitting an identical model")
                transform = root / "gene_transform"
                new_fit = not (destination / "manifest.json").exists()
                if new_fit:
                    args = [BASE / "scmrdr_adapter.py", "fit", "--train-dir", observed / "train", "--out", destination,
                            "--gene-transform-dir", transform, "--seed", seed, "--device", device,
                            "--excluded-features", exclusion_file]
                    if not simple:
                        args += ["--validation-dir", observed / "val"]
                    command("scmrdr_fit", *args, gpu=device == "cuda")
                _reuse_scmrdr(destination, seed, observed / "train", transform, excluded,
                              validation_dir=None if simple else observed / "val", device=device)
                _verify_scmrdr_encoding(destination, "train", observed / "train", 2000,
                                        newly_exported=new_fit,
                                        allow_preexisting_simple_train=simple and not new_fit)
                if not simple:
                    for cohort in ("val", "discovery", "probe", "assessment"):
                        saved = destination / cohort / "encoding_manifest.json"
                        source = observed / cohort if cohort in {"val", "discovery"} else world / "private_inputs" / regime / cohort
                        limit = 1000 if cohort == "discovery" else None
                        if saved.exists():
                            _verify_scmrdr_encoding(destination, cohort, source, limit)
                            continue
                        args = [BASE / "scmrdr_adapter.py", "encode", "--run", destination,
                                "--input-dir", source, "--cohort", cohort, "--device", device]
                        if cohort == "discovery":
                            args += ["--limit", 1000]
                        command("scmrdr_encode", *args, gpu=device == "cuda")
                        _verify_scmrdr_encoding(destination, cohort, source, limit, newly_exported=True)
            _public_embedding_manifest(destination, method,
                                       ("train",) if simple else ("train", "val", "discovery"))
    validation = world / "truth" / f"observation_validation_{regime}.json"
    if not validation.exists():
        command("validate_observed", BASE / "evaluate.py", "validate-observed", "--world", world, "--regime", regime)
    result_name = "evaluation_" + "_".join(methods) + "_seeds" + "-".join(map(str, seeds))
    args = [BASE / "evaluate.py", "evaluate", "--world", world, "--regime", regime,
            "--embeddings", *embeddings, "--output", root / result_name]
    if not simple and "scglue" in methods:
        args += ["--preprocessing-embeddings", preprocess / "representations"]
    command("evaluate", *args)
    return root / result_name / "report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--world-id", required=True)
    worker = sub.add_parser("generate-frozen-worker", help=argparse.SUPPRESS)
    worker.add_argument("--world-id", required=True)
    run = sub.add_parser("run-world")
    run.add_argument("--world-id", required=True)
    run.add_argument("--regime", choices=("realistic", "informative"), default="realistic")
    run.add_argument("--seeds", nargs="+", type=int, default=[0])
    run.add_argument("--methods", nargs="+", choices=("scglue", "scmrdr"), default=["scglue", "scmrdr"])
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--arm", choices=("primary", "background_only", "shuffled_guidance"), default="primary")
    run.add_argument("--assessment-lock", type=Path)
    run.add_argument("--allow-failed-calibration", action="store_true")
    lock = sub.add_parser("lock-assessment")
    lock.add_argument("--development-reports", type=Path, nargs="+", required=True)
    lock.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        print(generate(args.world_id))
    elif args.command == "generate-frozen-worker":
        generate_frozen_worker(args.world_id)
    elif args.command == "run-world":
        if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
            parser.error("Seeds must be distinct nonnegative integers")
        print(run_world(args.world_id, args.regime, args.seeds, device=args.device, arm=args.arm,
                        assessment_lock=args.assessment_lock, allow_failed_calibration=args.allow_failed_calibration,
                        methods=tuple(args.methods)))
    else:
        reports, outcomes = validate_development_reports(args.development_reports)
        new_json(args.out, {"source_pins": source_pins(), "development_reports": reports,
                            "development_evidence": outcomes,
                            "bank_sha256": digest(RUNS / "worlds/extended_bank_manifest.json"),
                            "calibration_sha256": digest(RUNS / "calibration/empirical_count_profiles.json"),
                            "note": "No assessment-driven configuration changes; failures and N/A remain outcomes, not recovery success"})
        print(args.out)


if __name__ == "__main__":
    main()
