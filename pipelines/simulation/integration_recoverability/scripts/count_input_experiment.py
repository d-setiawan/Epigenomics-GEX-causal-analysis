"""Small count-encoder comparison; reuse cached development data and native GLUE."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import fcntl
import itertools
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import scglue_adapter as glue
import integration_validation_fit as fitting
import validate_integration as validation
import integration_validation_metrics as metrics
import execute_frozen_suite as operator

ASSAYS = glue.ASSAYS
MODES = ("compressed", "native_counts")
WORLDS = ("linear_chain_0", "constrained_collider_0")
ROOT = BASE / "runs/integration_validation/20261031_count_input_v1"
RUNTIME = Path("/tmp/integration_followup_runtime_20261031/libcudart.so.13.0.96")
RUNTIME_SHA256 = "96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"
RUNTIME_FLAGS = {"LD_PRELOAD": str(RUNTIME), "CUDA_LAUNCH_BLOCKING": "1",
                 "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def nonzero_libraries(data):
    if np.any(np.asarray(data.layers["counts"].sum(axis=1)).ravel() <= 0):
        raise ValueError("Count encoder requires positive observed libraries; no implicit cell exclusion")


def native_encoder_record(model, prepared, mode):
    records = {}
    for assay, data in prepared.items():
        config = model.domains[assay]
        expected_rep = glue.REPRESENTATIONS[assay] if mode == "compressed" else None
        expected_width = data.obsm[expected_rep].shape[1] if expected_rep else data.n_vars
        width = model.net.x2u[assay].linear_0.in_features
        if config["use_rep"] != expected_rep or width != expected_width:
            raise ValueError("Native encoder mode or first-layer width differs from the declared experiment")
        if config["use_layer"] != "counts" or config["prob_model"] != "NB":
            raise ValueError("The original count likelihood must remain unchanged")
        records[assay] = {"use_rep": config["use_rep"], "rep_dim": config["rep_dim"],
                          "input_width": width, "count_features": data.n_vars}
    return records


@contextmanager
def input_mode(mode):
    """Change only the documented encoder option; retain native training/export checks."""
    if mode not in MODES:
        raise ValueError("Unknown encoder mode")
    import scglue
    original_fit, original_encode = glue.fit_native, glue.encode_frozen
    original_provenance = glue.package_provenance
    original_configure = scglue.models.configure_dataset

    def provenance():
        return {**original_provenance(), "input_ablation": {
            "mode": mode, "source": str(Path(__file__).resolve()),
            "source_sha256": glue.file_hash(Path(__file__)),
            "count_input_normalization": "native library-size 1e4 and log1p" if mode == "native_counts" else None}}

    def configure(data, prob_model, **kwargs):
        if mode == "native_counts":
            nonzero_libraries(data)
            kwargs["use_rep"] = None
        return original_configure(data, prob_model, **kwargs)

    def fit(prepared, graph, config, directory):
        for data in prepared.values():
            nonzero_libraries(data)
        with patch.object(scglue.models, "configure_dataset", configure):
            model, audit = original_fit(prepared, graph, config, directory)
        audit["encoder_mode"] = mode
        audit["actual_native_encoders"] = native_encoder_record(model, prepared, mode)
        audit["prepared_PCA_LSI_arrays_are_inert"] = mode == "native_counts"
        return model, audit

    def encode(model, assay, data, contract, *, batch_size):
        nonzero_libraries(data)
        native_encoder_record(model, {assay: data}, mode)
        return original_encode(model, assay, data, contract, batch_size=batch_size)

    with ExitStack() as stack:
        stack.enter_context(patch.object(glue, "fit_native", fit))
        stack.enter_context(patch.object(glue, "encode_frozen", encode))
        stack.enter_context(patch.object(glue, "package_provenance", provenance))
        yield


def protocol():
    previous = validation.protocol()
    config = asdict(glue.ScGLUEConfig(lam_align=0.2, seed=0, device="cuda"))
    return {
        "study_id": ROOT.name, "worlds": list(WORLDS), "regime": "informative", "seed": 0,
        "modes": list(MODES), "configuration": config,
        "source_pins": previous["source_pins"],
        "analysis_sources": {str(Path(p).resolve()): glue.file_hash(Path(p)) for p in
                             (__file__, fitting.__file__, validation.__file__, metrics.__file__)},
        "runtime_library": str(RUNTIME), "runtime_sha256": RUNTIME_SHA256,
        "runtime_flags": RUNTIME_FLAGS,
        "inherited_loader_environment": {name: os.environ.get(name) for name in
                                         ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")},
        "training": "Four fresh native fits; identical selected cells/features/guidance/settings; no checkpoint reuse",
        "scoring": "Public val cohort; common cells across both arms and old method references; no causal discovery",
        "pool_seed": 2718, "knn": 15,
        "primary": "Worst of the three normalized squared-state transport gaps on each world",
        "limits": ["Exploratory two-world comparison; regimes/seeds are not independent biological replicates",
                   "Native-count mode also changes normalization; this is not a pure PCA-rank ablation",
                   "Input normalization, filtering and shared latent bottleneck still impose assumptions",
                   "PCA/LSI neighborhood retention is reference fidelity, not neutral biological validity"],
    }


def checked_protocol():
    spec = protocol()
    validation.write_new(ROOT / "protocol.json", spec)
    return spec


def fit_one(world, mode):
    if world not in WORLDS or mode not in MODES:
        raise ValueError("Unplanned world or mode")
    spec = checked_protocol()
    if (not RUNTIME.exists() or glue.file_hash(RUNTIME) != RUNTIME_SHA256
            or any(os.environ.get(k) != v for k, v in spec["runtime_flags"].items())):
        raise ValueError("Use the verified process-local runtime recorded in the protocol")
    primary = BASE / "runs/extended" / world / "informative/primary/scglue_preprocessing"
    prep = validation.read(primary / "manifest.json")
    for assay, entry in prep["assays"].items():
        meta = validation.read(entry["transform_manifest"])
        if not (meta["normalization_feature_ids"] == meta["encoder_feature_ids"]
                == meta["reconstruction_feature_ids"]):
            raise ValueError(f"{assay} feature coverage would change; this comparison requires identical feature sets")
    output = ROOT / "fits" / world / mode
    if output.exists():
        raise FileExistsError(f"Preserve the existing attempt; refusing {output}")
    with operator.global_lock(BASE / "runs/suite"):
        schedule = operator.suite.load_schedule(BASE / "runs/suite/schedule_v1.json")
        operator.assert_no_external_jobs(schedule)
        with (BASE / "runs/gpu_training.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with input_mode(mode):
                result = fitting.fit_candidate(BASE, world, "informative", "scglue", mode,
                                               {"lam_align": 0.2}, seed=0, device="cuda", output=output)
            receipt = {"status": "complete", "mode": mode, "world": world,
                       "protocol_sha256": glue.file_hash(ROOT / "protocol.json"),
                       "validation_fit_sha256": glue.file_hash(output / "validation_fit.json"),
                       "configuration": result["configuration"],
                       "runtime_environment": {name: os.environ.get(name) for name in
                                               (*RUNTIME_FLAGS, "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")},
                       "scope": "Public train/val/discovery exports only; no truth read by fitting"}
            validation.write_new(output / "experiment.json", receipt)
    return receipt


def load_fit(world, mode, cohort, spec):
    directory = ROOT / "fits" / world / mode
    receipt = validation.read(directory / "experiment.json")
    seal = validation.read(directory / "validation_fit.json")
    manifest = validation.read(directory / "model_manifest.json")
    ablation = manifest["inputs"]["package_provenance"]["input_ablation"]
    source = str(Path(__file__).resolve())
    if (receipt["status"] != "complete" or receipt["mode"] != mode or receipt["world"] != world
            or manifest.get("status") != "complete" or seal.get("status") != "complete"
            or seal["checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or receipt["protocol_sha256"] != glue.file_hash(ROOT / "protocol.json")
            or receipt["validation_fit_sha256"] != glue.file_hash(directory / "validation_fit.json")
            or seal["fit_manifest"]["sha256"] != glue.file_hash(directory / "model_manifest.json")
            or manifest["configuration"] != spec["configuration"]
            or manifest["training_audit"]["encoder_mode"] != mode
            or ablation["mode"] != mode or ablation["source"] != source
            or ablation["source_sha256"] != spec["analysis_sources"][source]
            or receipt["runtime_environment"] != {**spec["runtime_flags"], **spec["inherited_loader_environment"]}
            or manifest["checkpoint_sha256"] != glue.file_hash(directory / "scglue_model.dill")):
        raise ValueError("Fit identity, configuration, encoder mode or sealed artifact changed")
    for assay in ASSAYS:
        name = f"{cohort}/{assay}.npz"
        if seal["files"][name] != glue.file_hash(directory / name):
            raise ValueError("Embedding changed after fit")
    _, views, evidence = validation.evaluation._load_embedding_run(directory, cohort)
    return views, {"fit_manifest_sha256": glue.file_hash(directory / "model_manifest.json"),
                   "experiment_sha256": glue.file_hash(directory / "experiment.json"), "files": evidence["files"]}


def evaluate_views(world_id, runs, train_runs, evidence, *, cohort="val"):
    """Use the existing state metrics; state labels enter only after matching."""
    previous = validation.protocol()
    world, manifest = validation.development_world(previous, world_id)
    ev = validation.evaluation
    pools, pool_audit, reference_evidence = {}, {}, {}
    for c, views in (("train", train_runs), (cohort, runs)):
        identities = validation.truth_fields(world, manifest, c)
        eligibility = dict(views)
        for method in ("scglue", "scmrdr"):
            eligibility[f"pool_reference/{method}"], reference_evidence[f"{c}/{method}"] = validation.public_run(
                previous, world_id, "informative", method, "baseline", 0, c, reference=True)
        observed = {a: identities["cell_ids"][identities["assigned_assay"] == a] for a in ASSAYS}
        pools[c], pool_audit[c] = ev.select_common_pools(eligibility, observed, seed=2718)
    assignments = {label: ev.match_one_to_one(pools[cohort], views) for label, views in runs.items()}
    states = {}
    for c in ("train", cohort):
        truth = validation.truth_fields(world, manifest, c, states=True)
        states[c] = {a: truth["state"][ev._indices(truth["cell_ids"], pools[c][a])] for a in ASSAYS}
    pooled = np.concatenate(list(states["train"].values()))
    center, scale = pooled.mean(axis=0), pooled.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Uninformative training state scale")
    primary = BASE / "runs/extended" / world_id / "informative/primary"
    reference, geometry_evidence = validation.reference_geometry(primary, cohort, pools[cohort])
    results = {}
    for label, views in runs.items():
        x = {a: views[a]["shared"][ev._indices(views[a]["cell_ids"], pools[cohort][a])] for a in ASSAYS}
        tx = {a: train_runs[label][a]["shared"][ev._indices(train_runs[label][a]["cell_ids"], pools["train"][a])] for a in ASSAYS}
        chosen = {a: ev._indices(pools[cohort][a], assignments[label]["cell_ids"][a]) for a in ASSAYS}
        costs = {f"{a}--{b}": metrics.transport_certificate(
            (states[cohort][a][chosen[a]] - center) / scale,
            (states[cohort][b] - center) / scale, chosen[b]) for a, b in itertools.combinations(ASSAYS, 2)}
        gaps = [v["normalized_gap"] for v in costs.values()]
        results[label] = {"pairwise_state_transport": costs,
            "worst_pair_gap": max(gaps) if all(g is not None for g in gaps) else None,
            "source_state_transfer": validation.state_probes(tx, states["train"], x, states[cohort]),
            "within_assay_neighbor_retention": {a: metrics.neighborhood_overlap(reference[a], x[a], 15) for a in ASSAYS},
            "pre_assignment_hubness": {f"{a}->{b}": metrics.neighbor_hubness(x[a], x[b], 15) for a in ASSAYS for b in ASSAYS if a != b},
            "shared_variance": {a: np.var(x[a], axis=0).tolist() for a in ASSAYS}}
    return {"status": "complete", "world_id": world_id, "regime": "informative", "cohort": cohort,
            "results": results, "pools": {c: {a: ids.tolist() for a, ids in p.items()} for c, p in pools.items()},
            "pool_audit": pool_audit, "assignments": ev._jsonable(assignments),
            "state_center": center.tolist(), "state_scale": scale.tolist(), "evidence": evidence,
            "pool_references": reference_evidence, "reference_geometry": geometry_evidence,
            "truth_hashes": {c: glue.file_hash(world / "truth" / f"{c}.npz") for c in ("train", cohort)},
            "analysis_source_sha256": glue.file_hash(Path(__file__)),
            "scope": "Existing public development cohorts; no causal graph or residual-joint claim"}


def evaluate(world):
    spec = checked_protocol()
    if world not in WORLDS:
        raise ValueError("Unplanned world")
    runs, train_runs, evidence = {}, {}, {}
    for mode in MODES:
        runs[mode], evidence[mode] = load_fit(world, mode, "val", spec)
        train_runs[mode], _ = load_fit(world, mode, "train", spec)
    report = evaluate_views(world, runs, train_runs, evidence)
    report["protocol_sha256"] = glue.file_hash(ROOT / "protocol.json")
    validation.write_new(ROOT / "reports" / f"{world}.json", report)
    return {mode: r["worst_pair_gap"] for mode, r in report["results"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "fit", "evaluate"))
    parser.add_argument("--world", choices=WORLDS)
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    if args.action == "plan":
        result = checked_protocol()
    elif args.action == "fit":
        result = fit_one(args.world, args.mode)
    else:
        result = evaluate(args.world)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
