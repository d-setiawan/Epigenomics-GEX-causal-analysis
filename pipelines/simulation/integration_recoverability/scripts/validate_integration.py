"""Integration-only development study; never calls causal discovery."""

import argparse
import fcntl
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import evaluate as evaluation
import suite_plan
import integration_validation_metrics as metrics

ASSAYS = evaluation.ASSAYS
DEFAULT_PROTOCOL = BASE / "configs/integration_validation_v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Changed output; do not overwrite {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)
    return path


def protocol(path=DEFAULT_PROTOCOL):
    value = read(path)
    if value["source_pins"] != suite_plan.source_pins(BASE, BASE.parents[2]):
        raise ValueError("Frozen scientific sources changed")
    names = value["selection_worlds"] + value["confirmation_worlds"]
    names += [c for choices in value["methods"].values() for c in choices]
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        raise ValueError("Unsafe study identifiers")
    root = (BASE / value["output_root"]).resolve()
    permitted = (BASE / "runs/integration_validation").resolve()
    if not root.is_relative_to(permitted) or root == permitted:
        raise ValueError("Study outputs must stay in a dedicated integration-validation directory")
    value["protocol_sha256"] = suite_plan.digest(path)
    return value


def development_world(spec, world_id):
    allowed = spec["selection_worlds"] + spec["confirmation_worlds"]
    if world_id not in allowed:
        raise ValueError("World is not predeclared for development validation")
    world = BASE / "runs/worlds" / world_id
    manifest = read(world / "manifest.json")
    if (manifest.get("split") != "development" or manifest.get("world_id") != world_id
            or manifest.get("status") != "complete"
            or manifest.get("count_calibration_status") != "passed_declared_checks"):
        raise ValueError("Require an existing calibrated development world")
    return world, manifest


def truth_fields(world, manifest, cohort, *, states=False):
    if cohort not in {"train", "val", "discovery"}:
        raise ValueError("Private probe/assessment cohorts are prohibited")
    path = world / "truth" / f"{cohort}.npz"
    if suite_plan.digest(path) != manifest["files"][f"truth/{cohort}.npz"]:
        raise ValueError("Development truth archive changed")
    with np.load(path, allow_pickle=False) as saved:
        ids = evaluation._ids(saved["cell_ids"], "truth/cell_ids")
        assays = np.asarray(saved["assigned_assay"], dtype=str)
        if assays.shape != ids.shape or not set(assays) <= set(ASSAYS):
            raise ValueError("Invalid public-cohort assay identities")
        result = {"cell_ids": ids, "assigned_assay": assays}
        if states:
            result["state"] = evaluation._matrix(saved["state"], "state", len(ids))
    return result


def candidate_directory(spec, world_id, regime, method, candidate, seed, *, reference=False):
    if world_id not in spec["selection_worlds"] + spec["confirmation_worlds"]:
        raise ValueError("World is not predeclared for development validation")
    if regime not in {"realistic", "informative"} or type(seed) is not int or seed < 0:
        raise ValueError("Invalid regime or seed")
    if method not in spec["methods"] or candidate not in spec["methods"][method]:
        raise ValueError("Unknown predeclared method/candidate")
    if reference and candidate == "baseline":
        return BASE / "runs/extended" / world_id / regime / "primary" / f"{method}_seed{seed}"
    return (BASE / spec["output_root"] / "fits" / world_id / regime / method
            / f"{candidate}_seed{seed}")


def expected_configuration(spec, method, candidate, seed):
    return {**spec["defaults"][method], **spec["methods"][method][candidate], "seed": seed}


def public_run(spec, world_id, regime, method, candidate, seed, cohort, *, reference=False):
    directory = candidate_directory(spec, world_id, regime, method, candidate, seed, reference=reference)
    model_path = directory / ("model_manifest.json" if method == "scglue" else "manifest.json")
    model = read(model_path)
    actual = model["configuration" if method == "scglue" else "config"]
    if model.get("status") != "complete" or actual != expected_configuration(spec, method, candidate, seed):
        raise ValueError("Incomplete model or mismatched candidate configuration")
    checkpoint = directory / ("scglue_model.dill" if method == "scglue" else "model/checkpoint.pt")
    if suite_plan.digest(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    seal_path = directory / ("public_embedding_manifest.json" if reference else "validation_fit.json")
    seal = read(seal_path)
    if seal.get("status") != "complete" or seal["checkpoint_sha256"] != model["checkpoint_sha256"]:
        raise ValueError("Incomplete or unrelated public export seal")
    if not reference:
        study = read(directory / "study.json")
        if study["protocol_sha256"] != spec["protocol_sha256"]:
            raise ValueError("Candidate belongs to another validation protocol")
        if study["validation_fit_sha256"] != suite_plan.digest(seal_path):
            raise ValueError("Public export seal changed after the study receipt")
        if seal["fit_manifest"]["sha256"] != suite_plan.digest(model_path):
            raise ValueError("Fit manifest changed after encoding")
    for assay in ASSAYS:
        relative = f"{cohort}/{assay}.npz"
        if suite_plan.digest(directory / relative) != seal["files"][relative]:
            raise ValueError("Public embedding changed after encoding")
    _, arrays, loaded = evaluation._load_embedding_run(directory, cohort)
    arrays = {a: {"cell_ids": x["cell_ids"], "shared": x["shared"]} for a, x in arrays.items()}
    evidence = {"directory": str(directory), "files": loaded["files"], "configuration": actual,
                "model_manifest_sha256": suite_plan.digest(model_path),
                "export_seal_sha256": suite_plan.digest(seal_path), "reference_only": reference}
    return arrays, evidence


def state_probes(train_x, train_y, test_x, test_y):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    result = {}
    for source in ASSAYS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(train_x[source], train_y[source])
        for target in ASSAYS:
            predicted = model.predict(test_x[target])
            total = np.square(test_y[target] - test_y[target].mean(axis=0)).sum(axis=0)
            residual = np.square(test_y[target] - predicted).sum(axis=0)
            values = [float(1 - error / variance) if variance > 0 else None
                      for error, variance in zip(residual, total)]
            finite = [x for x in values if x is not None]
            result[f"{source}->{target}"] = {"r2_by_state": values,
                "mean_r2": float(np.mean(finite)) if finite else None, "target_refit": False}
    return result


def reference_geometry(primary, cohort, pools):
    import anndata as ad
    values, evidence = {}, {}
    for assay in ASSAYS:
        directory = primary / "scglue_preprocessing"
        path = directory / ("train" if cohort == "train" else f"cohorts/{cohort}") / f"{assay}.h5ad"
        data = ad.read_h5ad(path, backed="r")
        try:
            rows = evaluation._indices(data.obs_names, pools[assay])
            representation = "X_pca" if assay == "RNA" else "X_lsi"
            values[assay] = np.asarray(data.obsm[representation])[rows]
        finally:
            data.file.close()
        evidence[assay] = {"path": str(path), "sha256": suite_plan.digest(path), "representation": representation}
    return values, evidence


def selected_candidates(spec, mode, methods):
    if not methods or len(set(methods)) != len(methods) or any(m not in spec["methods"] for m in methods):
        raise ValueError("Unknown or repeated integration method")
    if mode == "all":
        return {m: list(spec["methods"][m]) for m in methods}
    if mode == "baseline":
        return {m: ["baseline"] for m in methods}
    if mode != "selected":
        if any(mode not in spec["methods"][m] for m in methods):
            raise ValueError("Candidate must be predeclared for every requested method")
        return {m: [mode] for m in methods}
    selection = read(BASE / spec["output_root"] / "selected_settings.json")
    if selection["protocol_sha256"] != spec["protocol_sha256"]:
        raise ValueError("Selection belongs to another protocol")
    return {m: [selection["selected"][m]] for m in methods}


def audit(spec, world_id, regime, seed, cohort, mode="baseline", methods=("scglue", "scmrdr")):
    role = "selection" if cohort == "val" else "confirmation"
    if (cohort not in {"val", "discovery"} or world_id not in spec[f"{role}_worlds"]
            or regime not in spec[f"{role}_regimes"] or seed not in spec[f"{role}_seeds"]):
        raise ValueError("Cohort/world/regime/seed is outside its declared study role")
    if role == "confirmation" and mode != "selected":
        raise ValueError("Freeze settings before opening confirmation state labels")
    candidates = selected_candidates(spec, mode, methods)
    world, manifest = development_world(spec, world_id)
    reference = mode == "baseline"
    runs, train_runs, evidence, labels = {}, {}, {}, {}
    for method, choices in candidates.items():
        for candidate in choices:
            label = f"{method}/{candidate}"
            labels[label] = {"method": method, "candidate": candidate}
            runs[label], evidence[label] = public_run(spec, world_id, regime, method, candidate, seed, cohort, reference=reference)
            train_runs[label], _ = public_run(spec, world_id, regime, method, candidate, seed, "train", reference=reference)
    identities = {c: truth_fields(world, manifest, c) for c in ("train", cohort)}
    pools, pool_audit, pool_references = {}, {}, {}
    for c, views in (("train", train_runs), (cohort, runs)):
        eligibility = dict(views)
        pool_references[c] = {}
        # Keep interim and full audits eligible on the same cached baseline IDs.
        for method in spec["methods"]:
            label = f"pool_reference/{method}"
            eligibility[label], pool_references[c][method] = public_run(
                spec, world_id, regime, method, "baseline", seed, c, reference=True)
        ids = identities[c]
        observed = {a: ids["cell_ids"][ids["assigned_assay"] == a] for a in ASSAYS}
        pools[c], pool_audit[c] = evaluation.select_common_pools(eligibility, observed, seed=spec["pool_seed"])
    assignments = {label: evaluation.match_one_to_one(pools[cohort], views) for label, views in runs.items()}
    # Truth joins follow outcome-free pool selection and embedding-only matching.
    states = {}
    for c in ("train", cohort):
        truth = truth_fields(world, manifest, c, states=True)
        states[c] = {a: truth["state"][evaluation._indices(truth["cell_ids"], pools[c][a])] for a in ASSAYS}
    pooled_train = np.concatenate(list(states["train"].values()))
    center, scale = pooled_train.mean(axis=0), pooled_train.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Training state scale is uninformative")
    primary = BASE / "runs/extended" / world_id / regime / "primary"
    references, reference_evidence = reference_geometry(primary, cohort, pools[cohort])
    results = {}
    for label, views in runs.items():
        x = {a: views[a]["shared"][evaluation._indices(views[a]["cell_ids"], pools[cohort][a])] for a in ASSAYS}
        train_x = {a: train_runs[label][a]["shared"][evaluation._indices(train_runs[label][a]["cell_ids"], pools["train"][a])] for a in ASSAYS}
        chosen = {a: evaluation._indices(pools[cohort][a], assignments[label]["cell_ids"][a]) for a in ASSAYS}
        costs = {}
        for first, second in itertools.combinations(ASSAYS, 2):
            source = (states[cohort][first][chosen[first]] - center) / scale
            target = (states[cohort][second] - center) / scale
            costs[f"{first}--{second}"] = metrics.transport_certificate(source, target, chosen[second])
        gaps = [v["normalized_gap"] for v in costs.values()]
        results[label] = {**labels[label], "pairwise_state_transport": costs,
            "worst_pair_gap": max(gaps) if all(g is not None for g in gaps) else None,
            "source_state_transfer": state_probes(train_x, states["train"], x, states[cohort]),
            "within_assay_neighbor_retention": {a: metrics.neighborhood_overlap(references[a], x[a], spec["knn"]) for a in ASSAYS},
            "pre_assignment_hubness": {f"{a}->{b}": metrics.neighbor_hubness(x[a], x[b], spec["knn"]) for a in ASSAYS for b in ASSAYS if a != b},
            "shared_variance": {a: np.var(x[a], axis=0).tolist() for a in ASSAYS}}
    report = {"status": "complete", "study_id": spec["study_id"], "protocol_sha256": spec["protocol_sha256"],
        "world_id": world_id, "regime": regime, "seed": seed, "cohort": cohort, "role": role,
        "mode": mode, "reference_only": reference, "results": results, "pool_audit": pool_audit,
        "assignments": evaluation._jsonable(assignments), "state_center": center.tolist(), "state_scale": scale.tolist(),
        "evidence": evidence, "pool_reference_evidence": pool_references, "reference_geometry": reference_evidence,
        "truth_files": {c: suite_plan.digest(world / "truth" / f"{c}.npz") for c in ("train", cohort)},
        "analysis_sources": {Path(__file__).name: suite_plan.digest(__file__), Path(metrics.__file__).name: suite_plan.digest(metrics.__file__)},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "anndata")},
        "scope": "Computation complete, not a binary correctness verdict; no causal discovery, private cohorts, or assessment worlds. Pairwise oracle floors are not a joint three-assay optimum."}
    name = f"{world_id}_{regime}_{cohort}_seed{seed}_{'-'.join(methods)}_{mode}_{suite_plan.digest(__file__)[:8]}.json"
    destination = BASE / spec["output_root"] / "reports" / name
    return write_new(destination, report), report


def choose(spec):
    values, evidence = {}, []
    for world in spec["selection_worlds"]:
        for regime in spec["selection_regimes"]:
            for seed in spec["selection_seeds"]:
                path, report = audit(spec, world, regime, seed, "val", "all")
                evidence.append({"path": str(path), "sha256": suite_plan.digest(path)})
                for label, result in report["results"].items():
                    if result["worst_pair_gap"] is None:
                        raise ValueError("Selection has an uninformative state reference")
                    values.setdefault(label, []).append(result["worst_pair_gap"])
    means = {label: float(np.mean(items)) for label, items in values.items()}
    selected = {m: min(spec["methods"][m], key=lambda c: (means[f"{m}/{c}"], c)) for m in spec["methods"]}
    result = {"protocol_sha256": spec["protocol_sha256"], "selected": selected, "mean_worst_pair_gaps": means,
              "evidence": evidence, "interpretation": "Best among tested development settings, not proof of correct integration or causal structure"}
    return write_new(BASE / spec["output_root"] / "selected_settings.json", result), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "audit"):
        child = commands.add_parser(name)
        child.add_argument("--world", required=True)
        child.add_argument("--regime", choices=("realistic", "informative"), required=True)
        child.add_argument("--seed", type=int, default=0)
        if name == "fit":
            child.add_argument("--method", choices=("scglue", "scmrdr"), required=True)
            child.add_argument("--candidate", required=True)
        else:
            child.add_argument("--cohort", choices=("val", "discovery"), default="val")
            child.add_argument("--candidates", default="baseline",
                               help="baseline (cached), all (refitted), selected, or a predeclared candidate for an interim audit")
            child.add_argument("--method", choices=("both", "scglue", "scmrdr"), default="both")
    commands.add_parser("select")
    args = parser.parse_args()
    spec = protocol(args.protocol)
    if args.command == "audit":
        methods = tuple(spec["methods"]) if args.method == "both" else (args.method,)
        path, report = audit(spec, args.world, args.regime, args.seed, args.cohort, args.candidates, methods)
    elif args.command == "select":
        path, report = choose(spec)
    else:
        from integration_validation_fit import fit_candidate
        import execute_frozen_suite as operator
        development_world(spec, args.world)
        if args.world in spec["selection_worlds"]:
            if args.regime not in spec["selection_regimes"] or args.seed not in spec["selection_seeds"]:
                raise ValueError("Fit is outside the declared selection schedule")
        else:
            selected = selected_candidates(spec, "selected", [args.method])[args.method]
            if (args.candidate not in selected or args.regime not in spec["confirmation_regimes"]
                    or args.seed not in spec["confirmation_seeds"]):
                raise ValueError("Only frozen selected settings may enter confirmation")
        output = candidate_directory(spec, args.world, args.regime, args.method, args.candidate, args.seed)
        if output.exists():
            raise FileExistsError(f"Refusing an existing candidate output: {output}")
        with operator.global_lock(BASE / "runs/suite"), (BASE / "runs/gpu_training.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            operator.assert_no_external_jobs(suite_plan.load_schedule(BASE / "runs/suite/schedule_v1.json"))
            try:
                report = fit_candidate(BASE, args.world, args.regime, args.method, args.candidate,
                    spec["methods"][args.method][args.candidate], seed=args.seed,
                    device=spec["defaults"][args.method]["device"], output=output)
                path = write_new(output / "study.json", {"protocol_sha256": spec["protocol_sha256"],
                    "runner_source_sha256": suite_plan.digest(__file__), "source_pins": spec["source_pins"],
                    "validation_fit_sha256": suite_plan.digest(output / "validation_fit.json"),
                    "runtime_overrides": {key: os.environ.get(key) for key in (
                        "LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING")}})
            except Exception as error:
                if output.exists():
                    write_new(output / "validation_failure.json", {"status": "failed", "error_type": type(error).__name__,
                        "error": str(error), "protocol_sha256": spec["protocol_sha256"]})
                raise
    print(json.dumps({"status": report.get("status", "selected"), "output": str(path)}), flush=True)


if __name__ == "__main__":
    main()
