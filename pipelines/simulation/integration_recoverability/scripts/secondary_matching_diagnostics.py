"""Fixed, development-only anchor/readout diagnostics; never fit or generate.

``freeze`` audits existing primary evidence and writes an immutable receipt.
``run`` requires that receipt's SHA-256 and executes all four protocol rows.
This is a separate secondary evidence schema, not a primary suite report.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import uuid

import numpy as np

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
sys.path.insert(0, str(BASE))
import evaluate as ev
import suite_plan as suite

# The separately pinned operator supplies the SAME advisory lock as the suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import execute_frozen_suite as operator

PROTOCOL = BASE / "SECONDARY_DIAGNOSTICS_PROTOCOL.md"
SCHEDULE = BASE / "runs/suite/schedule_v1.json"
WORLD_IDS = ("linear_chain_0", "constrained_chain_0")
METHODS = ("scglue_seed0", "scmrdr_paper_seed0")
ARMS = ("paired_anchor_reference", "random_reference", "true_state_reference", *METHODS)
CONDITIONS = (
    {"id": "anchor_H3K27ac", "anchor": "H3K27ac", "ci_seed": 0, "random_seed": 3141},
    {"id": "anchor_H3K4me3", "anchor": "H3K4me3", "ci_seed": 0, "random_seed": 3141},
    {"id": "readout_repeat_1", "anchor": "RNA", "ci_seed": 1, "random_seed": 3142},
    {"id": "readout_repeat_2", "anchor": "RNA", "ci_seed": 2, "random_seed": 3143},
)
SCALES = ("activities", "counts")


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return suite.canonical(ev._jsonable(value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(bindings, path, expected=None):
    path = Path(path).resolve()
    digest = ev._sha256(path)
    if expected is not None:
        require(digest == expected, f"Recorded file changed: {path}")
    if str(path) in bindings:
        require(bindings[str(path)] == digest, f"File changed during audit: {path}")
    bindings[str(path)] = digest
    return digest


def recorded_paths(value, bindings):
    """Verify nested existing path/SHA receipts, without loading any model."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            pin(bindings, value["path"], value["sha256"])
        for item in value.values():
            recorded_paths(item, bindings)
    elif isinstance(value, list):
        for item in value:
            recorded_paths(item, bindings)


def safe_relative(root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Receipt path escapes its artifact directory")
    return path


def safe_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label) + "_" + hashlib.sha256(label.encode()).hexdigest()[:8]


def audit_model(directory, label, provenance, bindings):
    """Fail closed on missing model/export provenance (unlike the light loader)."""
    directory = Path(directory).resolve()
    glue = label == METHODS[0]
    model_path = directory / ("model_manifest.json" if glue else "manifest.json")
    export_path = directory / ("embedding_manifest.json" if glue else "manifest.json")
    model, exported = read(model_path), read(export_path)
    pin(bindings, model_path)
    pin(bindings, export_path, provenance["manifest_sha256"])
    require(exported == provenance["manifest"], "Primary export manifest differs from saved provenance")
    require(model["status"] == "complete" and model["seed"] == 0 and model["label"] == label,
            "Only complete primary seed-0 models are allowed")
    require(model["method"] == ("scglue" if glue else "scmrdr_paper_faithful"), "Wrong integration method")
    checkpoint = directory / ("scglue_model.dill" if glue else "model/checkpoint.pt")
    pin(bindings, checkpoint, model["checkpoint_sha256"])
    public_path = directory / "public_embedding_manifest.json"
    public = read(public_path)
    pin(bindings, public_path)
    expected_public = {f"{cohort}/{assay}.npz" for cohort in ("train", "val", "discovery") for assay in ev.ASSAYS}
    require(public["status"] == "complete" and public["checkpoint_sha256"] == model["checkpoint_sha256"]
            and set(public["files"]) == expected_public, "Incomplete public export receipt")
    for relative, digest in public["files"].items():
        pin(bindings, safe_relative(directory, relative), digest)
    if glue:
        require(exported["status"] == "complete" and exported["checkpoint_sha256"] == model["checkpoint_sha256"],
                "scGLUE export and checkpoint disagree")
        pin(bindings, model_path, exported["inputs"]["model_manifest_sha256"])
        pin(bindings, BASE / "scglue_adapter.py", model["inputs"]["adapter_sha256"])
        for relative, digest in exported["files"].items():
            pin(bindings, safe_relative(directory, relative), digest)
        prepared = directory.parent / "scglue_preprocessing"
        pin(bindings, prepared / "manifest.json")
        for relative, digest in model["inputs"]["transformations"]["files"].items():
            pin(bindings, safe_relative(prepared, relative), digest)
    else:
        pin(bindings, BASE / "scmrdr_adapter.py", model["adapter_source_sha256"])
        gene_path = Path(model["gene_transform_dir"]) / "manifest.json"
        pin(bindings, gene_path, model["gene_manifest_sha256"])
        gene = read(gene_path)
        require(gene["excluded_features"] == [], "Background-only fits are outside this secondary protocol")
        pin(bindings, BASE / "gene_features.py", gene["source_sha256"])
        summary_path = directory / "model/fit_summary.json"
        summary = read(summary_path)
        pin(bindings, summary_path)
        require(summary["config"] == model["config"] and summary["method"] == model["method"],
                "scMRDR fit summary disagrees with model manifest")
        paper = REPO / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
        pin(bindings, paper / "adapter.py", summary["source"]["adapter_sha256"])
        for relative, digest in summary["source"]["vendored_sha256"].items():
            pin(bindings, safe_relative(paper / "upstream", relative), digest)
        for cohort in ("train", "val", "discovery"):
            receipt_path = directory / cohort / "artifact_receipt.json"
            receipt = read(receipt_path)
            pin(bindings, receipt_path)
            require(receipt["status"] == "complete" and receipt["identity"]["checkpoint_sha256"] == model["checkpoint_sha256"]
                    and receipt["identity"]["fit_manifest_sha256"] == ev._sha256(model_path), "scMRDR export identity mismatch")
            require(set(receipt["files"]) == {f"{assay}.npz" for assay in ev.ASSAYS}, "Incomplete cohort export")
            for relative, digest in receipt["files"].items():
                pin(bindings, safe_relative(directory / cohort, relative), digest)
            if cohort != "train":
                audit_path = directory / cohort / "encoding_manifest.json"
                audit = read(audit_path)
                pin(bindings, audit_path, receipt["identity"]["encoding_manifest_sha256"])
                require(audit["checkpoint_sha256"] == model["checkpoint_sha256"] and audit["post_fit_only"] is True,
                        "Encoding is not a frozen-checkpoint export")
                recorded_paths(audit, bindings)
        recorded_paths(gene, bindings)
    recorded_paths(model, bindings)
    recorded_paths(exported, bindings)
    return {"model_manifest": str(model_path), "checkpoint": str(checkpoint),
            "checkpoint_sha256": model["checkpoint_sha256"], "public_export_manifest": str(public_path)}


def primary_readout_path(report_path, arm):
    filename = "paired_anchor_reference" if arm == ARMS[0] else safe_label(arm)
    return Path(report_path).parent / "readouts" / f"{filename}.npz"


def paired_readout(truth, pools, anchor):
    rows = ev._indices(truth["cell_ids"], pools[anchor])
    return {"counts": truth["counts"][rows], "activities": truth["activities"][rows],
            "cell_ids": np.asarray(pools[anchor]), "candidate_names": truth["candidate_names"]}


def verify_primary_readouts(report_path, report, truth, pools, bindings):
    """Bind original arrays to the recorded assignment, not just a new checksum."""
    for arm in ARMS:
        path = primary_readout_path(report_path, arm)
        pin(bindings, path)
        saved = ev._load_npz(path)
        if arm == ARMS[0]:
            expected = paired_readout(truth, pools, "RNA")
        else:
            assignment = report["results"][arm]["assignment"]
            validate_assignment(assignment, pools, "RNA")
            selected = ev.assemble_readout(assignment, truth)
            expected = {scale: selected[scale] for scale in SCALES}
            expected.update(candidate_names=truth["candidate_names"],
                            **{f"cell_ids_{assay}": ids for assay, ids in assignment["cell_ids"].items()})
        require(set(saved) == set(expected) and all(np.array_equal(saved[key], value) for key, value in expected.items()),
                f"Primary readout differs from original values/assignment: {arm}")


def validate_assignment(assignment, pools, anchor):
    require(assignment["anchor"] == anchor and set(assignment["cell_ids"]) == set(ev.ASSAYS), "Wrong assignment anchor/assays")
    require(assignment["n_tuples"] == len(pools[anchor]), "Assignment pool size changed")
    for assay in ev.ASSAYS:
        selected = ev._ids(assignment["cell_ids"][assay], assay)
        require(set(selected) == set(pools[assay]), "Assignment substituted or omitted pool cells")
    require(np.array_equal(assignment["cell_ids"][anchor], pools[anchor]), "Assignment reordered anchor cells")


def audit_primary(report_path):
    """Read-only input inspection; no integration, CI, oracle, or probe fit."""
    report_path = Path(report_path).resolve()
    report = read(report_path)
    inputs = report["inputs"]
    world = Path(inputs["world"]).resolve()
    require(world.name in WORLD_IDS, "Assessment/other worlds are outside the fixed secondary protocol")
    expected_report = BASE / "runs/extended" / world.name / "informative/primary/evaluation_scglue_scmrdr_seeds0/report.json"
    require(report_path == expected_report.resolve() and world == (BASE / "runs/worlds" / world.name).resolve(),
            "Only the saved primary seed-0 report and its original world are allowed")
    require(report["status"] in {"completed", "completed_with_failures"}, "Primary evaluation is not terminal")
    require(inputs["configuration"] == asdict(ev.DiscoveryConfig()) and inputs["regime"] == "informative"
            and inputs["cohort"] == "discovery" and inputs["pool_seed"] == 2718 and inputs["matching_seed"] == 3141,
            "Primary configuration differs from the secondary protocol")
    require(not report["failed_integrations"] and set(inputs["provenance"]) == set(METHODS), "Both original seed-0 fits are required")
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    require(fingerprint == report["input_fingerprint"], "Primary fingerprint disagrees with recorded inputs")
    require(read(report_path.parent / "input_manifest.json") == {"inputs": inputs, "input_fingerprint": fingerprint},
            "Primary input receipt differs from report")
    schedule = suite.load_schedule(SCHEDULE)
    suite.validate_inputs(schedule)
    bindings = {}
    for path in (report_path, report_path.parent / "input_manifest.json", SCHEDULE, PROTOCOL, Path(__file__), Path(operator.__file__)):
        pin(bindings, path)
    for relative, digest in schedule["inputs"]["source_pins"].items():
        pin(bindings, REPO / relative, digest)
    for key in ("bank", "calibration"):
        pin(bindings, schedule["inputs"][f"{key}_path"], schedule["inputs"][f"{key}_sha256"])
    pin(bindings, Path(ev.__file__), inputs["evaluation_source_sha256"])
    current_backend = ev.backend_provenance(ev.DiscoveryConfig())
    require(current_backend == inputs["backend_provenance"], "Primary CI backend/version/source changed")
    backend_files = [Path(current_backend["cit_source_path"])]
    # Pin causal-learn AND its external chosen HBE approximation implementation.
    # The primary report did not record momentchi2; its hashes are first recorded
    # here, not represented as historical pre-primary anti-tampering evidence.
    backend_root = backend_files[0].parents[1]
    for path in sorted(backend_root.rglob("*.py")):
        pin(bindings, path)
    approximation = importlib.util.find_spec("momentchi2")
    require(approximation is not None and approximation.origin is not None, "Missing momentchi2 HBE source")
    for path in sorted(Path(approximation.origin).parent.rglob("*.py")):
        pin(bindings, path)
    world_manifest_path, truth_manifest_path = world / "manifest.json", world / "truth/manifest.json"
    pin(bindings, world_manifest_path, inputs["world_manifest_sha256"])
    pin(bindings, truth_manifest_path, inputs["truth_manifest_sha256"])
    world_manifest, truth_manifest = read(world_manifest_path), read(truth_manifest_path)
    require(world_manifest["status"] == "complete" and world_manifest["split"] == "development"
            and world_manifest["design"] == "extended" and world_manifest["world_id"] == world.name,
            "World is not the completed approved development specification")
    require(canonical({k: v for k, v in world_manifest.items() if k != "manifest_hash"}) == world_manifest["manifest_hash"],
            "World content receipt changed")
    truth_path = world / "truth/discovery.npz"
    pin(bindings, truth_path, inputs["truth_sha256"])
    validation_path = world / "truth/observation_validation_informative.json"
    validation = read(validation_path)
    pin(bindings, validation_path)
    require(validation["status"] == "ok" and validation["regime"] == "informative"
            and Path(validation["world"]).resolve() == world
            and validation["truth_manifest_sha256"] == inputs["truth_manifest_sha256"], "Invalid observation validation")
    for cohort in ("discovery", "assessment", "probe"):
        entry = validation["cohorts"][cohort]
        pin(bindings, world / "truth" / f"{cohort}.npz", entry["truth_sha256"])
        recorded_paths(entry, bindings)
        for assay in ev.ASSAYS:
            source = Path(entry["assays"][assay]["path"]).resolve()
            relative = str(source.relative_to(world))
            require(world_manifest["files"][relative] == entry["assays"][assay]["sha256"], "Observed file is not root-world bound")
    # Native preprocessing representations have no dedicated export manifest;
    # the primary evaluator records those two optional filenames as absent.
    # Their parent preprocessing manifest and every actual NPZ remain pinned.
    optional_absent = {str((report_path.parents[1] / "scglue_preprocessing/representations" / filename).resolve())
                       for filename in ("manifest.json", "embedding_manifest.json")}
    absent_paths = []
    for path, digest in inputs["extra_inputs"].items():
        if digest == "missing":
            require(path in optional_absent and not Path(path).exists(), "Required contextual primary input was missing or optional absence changed")
            absent_paths.append(path)
        else:
            pin(bindings, path, digest)
    for path in (truth_manifest_path, truth_path, world / "truth/assessment.npz", world / "truth/probe.npz"):
        pin(bindings, path, world_manifest["files"][str(path.relative_to(world))])
    truth = ev.validate_truth(ev._load_npz(truth_path), "informative")
    candidates = truth_manifest["candidates"]
    require(report["candidates"] == candidates and len(candidates) == 9
            and list(truth["candidate_names"]) == [row["name"] for row in candidates], "Candidate panel/order changed")
    pools = {assay: ev._ids(report["common_pools"]["cell_ids"][assay], assay) for assay in ev.ASSAYS}
    require(len({len(value) for value in pools.values()}) == 1 and len(pools["RNA"]) >= 20, "Invalid primary common pool")
    runs, models = {}, {}
    for label in METHODS:
        provenance = inputs["provenance"][label]
        directory = Path(provenance["directory"])
        expected_name = "scglue_seed0" if label == METHODS[0] else "scmrdr_seed0"
        require(directory.resolve() == report_path.parents[1] / expected_name, "Non-primary model directory")
        models[label] = audit_model(directory, label, provenance, bindings)
        loaded_label, arrays, loaded_provenance = ev._load_embedding_run(directory, "discovery")
        require(loaded_label == label and loaded_provenance == provenance, "Embedding provenance differs from primary report")
        for assay in ev.ASSAYS:
            pin(bindings, provenance["files"][assay]["path"], provenance["files"][assay]["sha256"])
            require(set(pools[assay]) <= set(arrays[assay]["cell_ids"]), "Primary pool not contained in current embeddings")
        runs[label] = {assay: {key: arrays[assay][key] for key in ("cell_ids", "shared")} for assay in ev.ASSAYS}
    for assay, ids in pools.items():
        audit = report["common_pools"]["by_assay"][assay]
        require(audit["selected"] == len(ids) and hashlib.sha256("\n".join(ids).encode()).hexdigest() == audit["selected_ids_sha256"],
                "Primary pool ID hash/order changed")
        require(np.all(truth["assigned_assay"][ev._indices(truth["cell_ids"], ids)] == assay), "Pool uses another assay's cells")
    for arm in ARMS:
        for scale in SCALES:
            result = report["results"][arm][scale]
            require(result["configuration"] == asdict(ev.DiscoveryConfig()) and result["names"] == list(truth["candidate_names"])
                    and result["n_rows"] == len(pools["RNA"]) and result["method"] == "fci", "Primary graph configuration/order changed")
    verify_primary_readouts(report_path, report, truth, pools, bindings)
    for path in sorted((report_path.parent / "stages").glob("*.json")):
        pin(bindings, path)
        require(read(path)["input_fingerprint"] == fingerprint, "Primary stage has different input fingerprint")
    return {"report": report, "report_path": report_path, "truth": truth, "truth_manifest": truth_manifest,
            "pools": pools, "runs": runs, "bindings": bindings, "models": models, "schedule": schedule,
            "backend": current_backend, "world_id": world.name, "absent_optional_representation_manifests": sorted(absent_paths)}


def freeze(primary_report, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Secondary output already exists; no receipt overwrite")
    context = audit_primary(primary_report)
    receipt = {"schema": "secondary_matching_freeze_v1", "status": "secondary_frozen",
               "created_unix_ns": time.time_ns(), "world_id": context["world_id"],
               "primary_report": str(context["report_path"]), "primary_fingerprint": context["report"]["input_fingerprint"],
               "conditions": list(CONDITIONS), "arms": list(ARMS), "scales": list(SCALES),
               "file_sha256": context["bindings"], "models": context["models"],
               "absent_optional_representation_manifests": context["absent_optional_representation_manifests"],
               "backend": context["backend"], "configuration": asdict(ev.DiscoveryConfig()),
               "common_pools_sha256": canonical(context["report"]["common_pools"]),
               "packages": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "causal-learn", "momentchi2")},
               "integrity_scope": "New secondary receipt; historical primary hashes are checked where recorded, not invented. momentchi2 HBE source/version first recorded here, absent from the primary backend receipt.",
               "interpretation": "Development outcomes were known; all conditions are secondary, not independent world replicates"}
    output.mkdir(parents=True, exist_ok=False)
    operator.durable_json(output / "freeze.json", receipt)
    return {"status": receipt["status"], "receipt": str(output / "freeze.json"),
            "freeze_sha256": ev._sha256(output / "freeze.json"), "new_ci_calculations": 0}


def verify_frozen(output, expected_sha256):
    path = Path(output) / "freeze.json"
    require(ev._sha256(path) == expected_sha256, "Secondary freeze receipt changed")
    receipt = read(path)
    require(receipt["schema"] == "secondary_matching_freeze_v1" and receipt["status"] == "secondary_frozen"
            and receipt["world_id"] in WORLD_IDS and receipt["conditions"] == list(CONDITIONS)
            and receipt["arms"] == list(ARMS) and receipt["scales"] == list(SCALES)
            and receipt["configuration"] == asdict(ev.DiscoveryConfig()), "Secondary protocol choices changed")
    required_sources = {str(Path(__file__).resolve()), str(PROTOCOL.resolve()), str(Path(ev.__file__).resolve()),
                        str(Path(operator.__file__).resolve()), str(SCHEDULE.resolve())}
    require(required_sources <= set(receipt["file_sha256"]), "Missing required source/protocol pins")
    for filename, expected in receipt["file_sha256"].items():
        require(ev._sha256(Path(filename)) == expected, f"Frozen secondary input changed: {filename}")
    for filename in receipt["absent_optional_representation_manifests"]:
        require(not Path(filename).exists(), f"Optional preprocessing manifest absence changed: {filename}")
    require(receipt["backend"] == ev.backend_provenance(ev.DiscoveryConfig()), "Secondary CI backend changed")
    for package, expected in receipt["packages"].items():
        require(importlib.metadata.version(package) == expected, f"Software version changed: {package}")
    return receipt


class ExecutionInterrupted(BaseException):
    pass


@contextmanager
def interruption_signals():
    def interrupt(signum, _frame):
        raise ExecutionInterrupted(f"Received signal {signum}")
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    for number in previous:
        signal.signal(number, interrupt)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


class StageStore:
    """Immutable results, including numerical failures; only interrupted work resumes."""
    def __init__(self, output, freeze_sha256):
        self.output, self.freeze_sha256 = Path(output), freeze_sha256

    def stage(self, label, calculate):
        verify_frozen(self.output, self.freeze_sha256)
        directory = self.output / "stages" / safe_label(label)
        result_path = directory / "result.json"
        if result_path.exists():
            record = read(result_path)
            require(record["freeze_sha256"] == self.freeze_sha256 and record["stage"] == label
                    and canonical(record["result"]) == record["result_sha256"], "Secondary stage cache changed")
            recorded_paths(record["result"], {})
            return record["result"]
        attempt = directory / f"attempt_{time.time_ns()}_{uuid.uuid4().hex}.json"
        operator.durable_json(attempt, {"stage": label, "freeze_sha256": self.freeze_sha256,
                                        "status": "started", "created_unix_ns": time.time_ns()})
        result = calculate()
        # Input mutation during computation invalidates the result, not just the next stage.
        verify_frozen(self.output, self.freeze_sha256)
        operator.durable_json(result_path, {"stage": label, "freeze_sha256": self.freeze_sha256,
                                            "attempt": str(attempt), "result": ev._jsonable(result),
                                            "result_sha256": canonical(result)})
        return result


def condition_assignment(condition, arm, context):
    pools, truth = context["pools"], context["truth"]
    if arm == "random_reference":
        assignment = ev.match_one_to_one(pools, anchor=condition["anchor"], random_seed=condition["random_seed"])
    elif condition["anchor"] == "RNA":
        # Reuse EXACT recorded assignments; no re-optimization or pool reselection.
        assignment = deepcopy(context["report"]["results"][arm]["assignment"])
    elif arm == "true_state_reference":
        state = {assay: {"cell_ids": ids, "shared": truth["state"][ev._indices(truth["cell_ids"], ids)]}
                 for assay, ids in pools.items()}
        assignment = ev.match_one_to_one(pools, state, anchor=condition["anchor"])
    else:
        # These dictionaries contain ONLY public opaque cell IDs and shared embeddings.
        assignment = ev.match_one_to_one(pools, context["runs"][arm], anchor=condition["anchor"])
    validate_assignment(assignment, pools, condition["anchor"])
    return assignment


def save_readout(output, condition, arm, context):
    paired = paired_readout(context["truth"], context["pools"], condition["anchor"])
    if arm == "paired_anchor_reference":
        arrays, summary = paired, {"anchor": condition["anchor"], "interpretation": "Hidden same-cell panel of this anchor's cells"}
    else:
        assignment = condition_assignment(condition, arm, context)
        values = ev.assemble_readout(assignment, context["truth"])
        marginal_audit = verify_marginals(values, context)
        arrays = {scale: values[scale] for scale in SCALES}
        arrays.update(candidate_names=context["truth"]["candidate_names"],
                      **{f"cell_ids_{assay}": ids for assay, ids in assignment["cell_ids"].items()})
        summary = {"assignment": assignment, "fixed_pool_marginals": marginal_audit,
                   "state_matching": ev.state_matching_metrics(values),
                   "activity_dependence": ev.dependence_metrics(values["activities"], paired["activities"], context["report"]["candidates"], scale="activities"),
                   "count_dependence": ev.dependence_metrics(np.log1p(values["counts"]), np.log1p(paired["counts"]), context["report"]["candidates"], scale="log1p_counts")}
    path = Path(output) / "readouts" / condition["id"] / f"{arm}.npz"
    ev._save_npz_new(path, **arrays)
    return {**summary, "readout": {"path": str(path), "sha256": ev._sha256(path)}}


def verify_marginals(values, context):
    """A one-to-one permutation cannot alter any feature marginal or detection."""
    truth, pools = context["truth"], context["pools"]
    audit = {"status": "verified_exact_marginals", "candidate_order": list(truth["candidate_names"]), "detections": {}}
    for assay in ev.ASSAYS:
        rows = ev._indices(truth["cell_ids"], pools[assay])
        columns = np.flatnonzero(truth["candidate_assays"] == assay)
        for scale in SCALES:
            original = truth[scale][np.ix_(rows, columns)]
            current = values[scale][:, columns]
            require(np.array_equal(np.sort(original, axis=0), np.sort(current, axis=0)),
                    f"Matching altered a fixed-pool feature marginal: {assay}/{scale}")
        audit["detections"][assay] = np.count_nonzero(values["counts"][:, columns], axis=0).tolist()
    return audit


def graph_result(values, scale, condition, context):
    config = replace(ev.DiscoveryConfig(), seed=condition["ci_seed"])
    names = list(context["truth"]["candidate_names"])
    discovery = ev.discover(values, names, method="fci", scale=scale, config=config)
    target = "activity_projection" if scale == "activities" else "count_measurement_pag"
    metrics = ev.score_graph(discovery, context["report"]["graphical_truth"][target], context["report"]["candidates"],
                             target=target, biological_edges=context["truth_manifest"]["biological_dag_edges"])
    return {"discovery": discovery, "graph_metrics": metrics}


def contextual_controls(context, ci_seed):
    result = {}
    for key in ("independent_paired_assessment", "independent_activities_with_state"):
        value = context["report"]["results"][key]
        graphs = [value["discovery"]] if "discovery" in value else [value[scale] for scale in SCALES]
        same = all(graph["configuration"] == asdict(replace(ev.DiscoveryConfig(), seed=ci_seed)) for graph in graphs)
        result[key] = ({"status": "cached_primary_context", "result": deepcopy(value),
                        "interpretation": "Same primary evidence, not a new independent result"} if same else
                       {"status": "not_run", "reason": "No independent primary reference with this exact CI seed/configuration"})
    return result


def expected_stages():
    return {f"{row['id']}::{arm}::{part}" for row in CONDITIONS for arm in ARMS for part in ("readout", *SCALES)}


def validate_finished(output, freeze_sha256):
    manifest_path = Path(output) / "secondary_artifact_manifest.json"
    manifest = read(manifest_path)
    completions = [read(path) for path in (Path(output) / "attempts").glob("*/result.json")]
    require(any(row.get("status") == "secondary_complete" and row.get("freeze_sha256") == freeze_sha256
                and row.get("artifact_manifest_sha256") == ev._sha256(manifest_path) for row in completions),
            "Missing independently recorded completion-manifest hash")
    require(manifest["schema"] == "secondary_matching_artifacts_v1" and manifest["freeze_sha256"] == freeze_sha256,
            "Secondary output receipt mismatch")
    for relative, digest in manifest["files"].items():
        require(ev._sha256(safe_relative(output, relative)) == digest, "Secondary output changed after completion")
    report = read(Path(output) / "secondary_report.json")
    require(report["schema"] == "secondary_matching_report_v1" and report["freeze_sha256"] == freeze_sha256
            and report["status"] in {"secondary_complete", "secondary_complete_with_failures"}, "Incomplete secondary report")
    require(set(report["conditions"]) == {row["id"] for row in CONDITIONS}, "Missing secondary condition")
    return report


def seal_json(path, value, *, resume):
    """Only explicit interrupted continuation may retain identical final bytes."""
    path = Path(path)
    if path.exists():
        encoded = json.dumps(ev._jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
        require(resume and path.read_text() == encoded, "Existing secondary final artifact differs or was not explicitly resumed")
        return
    operator.durable_json(path, ev._jsonable(value))


def run(output, freeze_sha256, *, resume=False):
    output = Path(output).resolve()
    receipt = verify_frozen(output, freeze_sha256)
    # Shared with execute_frozen_suite.py; never overlap a GPU/large-memory stage.
    with operator.global_lock(BASE / "runs/suite"):
        schedule = suite.load_schedule(SCHEDULE)
        operator.assert_no_external_jobs(schedule)
        verify_frozen(output, freeze_sha256)
        prior = sorted((output / "attempts").glob("*/start.json"))
        completed_attempts = [read(path.with_name("result.json")) for path in prior if path.with_name("result.json").exists()]
        if any(row["status"] == "secondary_complete" for row in completed_attempts):
            return validate_finished(output, freeze_sha256)
        require(not prior or resume, "Interrupted output requires explicit --resume with the exact freeze receipt")
        require(prior or not resume, "There is no interrupted secondary execution to resume")
        for start in prior:
            require(read(start)["freeze_sha256"] == freeze_sha256, "Prior attempt belongs to another receipt")
            finished = start.with_name("result.json")
            require(not finished.exists() or read(finished)["status"] == "interrupted", "Only interrupted executions may resume")
        context = audit_primary(receipt["primary_report"])
        require(context["bindings"] == receipt["file_sha256"] and context["models"] == receipt["models"],
                "Current audited inputs differ from frozen secondary receipt")
        require(context["absent_optional_representation_manifests"] == receipt["absent_optional_representation_manifests"],
                "Recorded optional preprocessing absences changed")
        attempt = output / "attempts" / f"{time.time_ns()}_{uuid.uuid4().hex}"
        operator.durable_json(attempt / "start.json", {"status": "started", "freeze_sha256": freeze_sha256,
                                                       "pid": os.getpid(), "resume": resume, "prior_attempts": [str(path.parent) for path in prior]})
        if prior:
            operator.durable_json(attempt / "interruption_record.json", {
                "freeze_sha256": freeze_sha256, "prior_attempts": [str(path.parent) for path in prior],
                "interpretation": "Explicit continuation; earlier incomplete attempts retained, never erased",
                "incomplete_stages": [str(path) for path in (output / "stages").glob("*") if not (path / "result.json").exists()]})
        store = StageStore(output, freeze_sha256)
        try:
            with interruption_signals():
                report = {"schema": "secondary_matching_report_v1", "status": "secondary_running",
                          "freeze_sha256": freeze_sha256, "primary_report": receipt["primary_report"],
                          "world_id": context["world_id"], "common_pools": context["report"]["common_pools"],
                          "candidates": context["report"]["candidates"], "graphical_truth": context["report"]["graphical_truth"],
                          "cached_primary_RNA0_context": {arm: context["report"]["results"][arm] for arm in ARMS},
                          "primary_retention_context": {key: context["report"][key] for key in ("probes", "preprocessing_probes")},
                          "conditions": {}, "failures": [],
                          "limitations": [*context["report"]["limitations"],
                              "Secondary development diagnostics after known primary outcomes; no best-anchor or best-seed selection.",
                              "Combined random-permutation/CI-seed sensitivity; these conditions are not independent worlds.",
                              "Anchor changes include row-order-dependent RCIT/RIT bandwidth and finite paired-reference variation; full matching/readout/test sensitivity.",
                              "Two extra CI seeds and one random permutation per condition do not estimate stable variance or false-adjacency rates.",
                              "Source measurement PAG does not model conditioning on count-based QC selection.",
                              "Nominal sparse-count CI null calibration is not guaranteed; failures are not empty graphs."]}
                for condition in CONDITIONS:
                    entry = {"configuration": condition, "results": {},
                             "contextual_controls": contextual_controls(context, condition["ci_seed"])}
                    for arm in ARMS:
                        stem = f"{condition['id']}::{arm}"
                        readout = store.stage(f"{stem}::readout", lambda: save_readout(output, condition, arm, context))
                        values = ev._load_npz(Path(readout["readout"]["path"]))
                        result = {"readout": readout}
                        for scale in SCALES:
                            print(json.dumps({"secondary_condition": condition["id"], "arm": arm, "scale": scale}), flush=True)
                            result[scale] = store.stage(f"{stem}::{scale}", lambda: graph_result(values[scale], scale, condition, context))
                            if result[scale]["discovery"]["status"] != "ok":
                                report["failures"].append(f"{stem}::{scale}")
                        entry["results"][arm] = result
                    report["conditions"][condition["id"]] = entry
                stages = [read(path) for path in (output / "stages").glob("*/result.json")]
                require({row["stage"] for row in stages} == expected_stages(), "Secondary stage set is incomplete or unexpected")
                report["status"] = "secondary_complete_with_failures" if report["failures"] else "secondary_complete"
                verify_frozen(output, freeze_sha256)
                seal_json(output / "secondary_report.json", report, resume=resume)
                artifacts = [path for path in output.rglob("*") if path.is_file() and "attempts" not in path.relative_to(output).parts
                             and path.name != "secondary_artifact_manifest.json"]
                seal_json(output / "secondary_artifact_manifest.json", {
                    "schema": "secondary_matching_artifacts_v1", "status": "secondary_sealed", "freeze_sha256": freeze_sha256,
                    "files": {str(path.relative_to(output)): ev._sha256(path) for path in sorted(artifacts)}}, resume=resume)
                operator.durable_json(attempt / "result.json", {"status": "secondary_complete", "freeze_sha256": freeze_sha256,
                                                                 "secondary_status": report["status"], "report_sha256": ev._sha256(output / "secondary_report.json"),
                                                                 "artifact_manifest_sha256": ev._sha256(output / "secondary_artifact_manifest.json")})
                return report
        except BaseException as error:
            interrupted = isinstance(error, (ExecutionInterrupted, KeyboardInterrupt))
            operator.durable_json(attempt / "result.json", {"status": "interrupted" if interrupted else "operational_failure",
                                                            "freeze_sha256": freeze_sha256, "error_type": type(error).__name__, "error": str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze", help="Audit and freeze inputs; no CI or model calculations")
    freezing.add_argument("--primary-report", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    executing = commands.add_parser("run", help="Run exactly four declared secondary conditions")
    executing.add_argument("--output", type=Path, required=True)
    executing.add_argument("--freeze-sha256", required=True)
    executing.add_argument("--resume", action="store_true", help="Explicitly continue only an interrupted run with identical inputs")
    args = parser.parse_args(argv)
    result = freeze(args.primary_report, args.output) if args.command == "freeze" else run(args.output, args.freeze_sha256, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("status", "receipt", "freeze_sha256", "world_id", "failures") if key in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
