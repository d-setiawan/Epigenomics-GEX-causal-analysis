"""Public-discovery causal diagnostics for fixed cached worlds; never fit or generate."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import score_count_input_followup as previous

experiment = previous.experiment
ev = experiment.validation.evaluation
BASE = experiment.BASE
ROOT = BASE / "runs/integration_validation/20261101_causal_reliability_v1"
WORLDS = experiment.WORLDS
ASSAYS = experiment.ASSAYS
COHORT = "discovery"
CONFIG = ev.DiscoveryConfig()
POOL_SEED = 2718
MATCHING_SEED = 3141
SUBSAMPLE_SEEDS = (7000, 7001, 7002)
ARMS = ("paired_reference", "random_reference", "true_state_reference", "glue_compressed",
        "glue_native_counts", "scconfluence_seed0", "scconfluence_seed1")
read = experiment.validation.read
write = experiment.validation.write_new
digest = experiment.glue.file_hash


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def verify(record):
    if digest(Path(record["path"])) != record["sha256"]:
        raise ValueError(f"Changed artifact: {record['path']}")


def freeze():
    """Freeze pools using public IDs/embeddings, without reading latent values."""
    expected, amendment = previous.prepare()
    old = experiment.validation.protocol()
    worlds = {}
    for name in WORLDS:
        world, manifest = experiment.validation.development_world(old, name)
        identities = experiment.validation.truth_fields(world, manifest, COHORT)
        observed = {a: identities["cell_ids"][identities["assigned_assay"] == a] for a in ASSAYS}
        runs, evidence = {}, {}
        for mode in experiment.MODES:
            runs[f"glue_{mode}"], evidence[mode] = experiment.load_fit(name, mode, COHORT, expected)
        for method in ("scglue", "scmrdr"):
            runs[f"pool_reference/{method}"], evidence[method] = experiment.validation.public_run(
                old, name, "informative", method, "baseline", 0, COHORT, reference=True)
        pools, pool_audit = ev.select_common_pools(runs, observed, seed=POOL_SEED)
        observation = world / "truth/observation_validation_informative.json"
        check = read(observation)
        if check["status"] != "ok" or check["regime"] != "informative":
            raise ValueError("Require original observed-count validation")
        public = check["cohorts"][COHORT]
        truth_path, truth_manifest = world / f"truth/{COHORT}.npz", world / "truth/manifest.json"
        if public["truth_sha256"] != digest(truth_path) or check["truth_manifest_sha256"] != digest(truth_manifest):
            raise ValueError("Public truth no longer matches count validation")
        for item in public["assays"].values():
            verify(item)
        prep = read(BASE / "runs/extended" / name / "informative/primary/scglue_preprocessing/manifest.json")
        selection = {a: {"path": item["h5ad"], "sha256": item["h5ad_sha256"],
                         "cell_ids_sha256": hashlib.sha256("\n".join(item["cell_ids"]).encode()).hexdigest(),
                         "feature_ids_sha256": hashlib.sha256("\n".join(item["feature_ids"]).encode()).hexdigest()}
                     for a, item in prep["assays"].items()}
        for item in selection.values():
            verify(item)
        worlds[name] = {"pools": ev._jsonable(pools), "pool_audit": pool_audit,
                        "evidence": evidence, "world_manifest": pin(world / "manifest.json"),
                        "truth_manifest": pin(truth_manifest), "public_truth": pin(truth_path),
                        "observed_validation": pin(observation), "observed_counts": public["assays"],
                        "training_selection_references": selection}
    protocol = {"status": "frozen", "worlds": worlds, "cohort": COHORT, "regime": "informative",
                "arms": list(ARMS), "discovery": asdict(CONFIG), "backend": ev.backend_provenance(CONFIG),
                "pool_seed": POOL_SEED, "matching_seed": MATCHING_SEED,
                "subsample_seeds": list(SUBSAMPLE_SEEDS), "subsample_fraction": 0.8,
                "sources": {str(Path(p).resolve()): digest(Path(p)) for p in (__file__, ev.__file__)},
                "original_source_pins": old["source_pins"], "previous_runtime_amendment": pin(amendment),
                "primary": "Observed-count measurement-PAG cross-assay adjacency precision/recall and sentinel false positives",
                "directions": "Exact biological direct pairs/directions are identifiability stress tests, not success requirements",
                "stability": "Three independent 80% per-assay subsamples with rematching; fixed fitted encoders and fixed CI seed, coarse conditional stability only",
                "scope": "Nine-node cached panels; public discovery only; original full feature universes include focal variables; no private assessment/probe or retraining",
                "limits": ["Nominal CI p-values after matching have no established FDR calibration",
                           "Stable inferred edges can still be false; paired count controls also have measurement noise",
                           "Integration seeds and cell subsamples are not independent biological replicates",
                           "Synthetic graphical oracle assumes the stated Markov/faithfulness model",
                           "Empirical paired graphs would be agreement references, not known causal truth"]}
    write(ROOT / "protocol.json", protocol)
    return protocol


def context(world):
    if world not in WORLDS:
        raise ValueError("Only the two predeclared development worlds are allowed")
    protocol = read(ROOT / "protocol.json")
    if protocol["discovery"] != asdict(CONFIG) or protocol["cohort"] != "discovery":
        raise ValueError("Declared discovery configuration changed")
    for path, sha in protocol["sources"].items():
        verify({"path": path, "sha256": sha})
    if experiment.validation.protocol()["source_pins"] != protocol["original_source_pins"]:
        raise ValueError("Original scientific sources changed")
    record = protocol["worlds"][world]
    for key in ("world_manifest", "truth_manifest", "public_truth", "observed_validation"):
        verify(record[key])
    for value in record["observed_counts"].values():
        verify(value)
    for value in record["training_selection_references"].values():
        verify(value)
    pools = {a: np.asarray(ids, dtype=str) for a, ids in record["pools"].items()}
    return protocol, record, pools


def subsample(pools, seed):
    n = int(0.8 * min(map(len, pools.values())))
    return {a: np.sort(np.random.default_rng(np.random.SeedSequence([seed, i])).choice(
        ids, n, replace=False)) for i, (a, ids) in enumerate(pools.items())}


def require_pool_coverage(views, pools):
    for assay in ASSAYS:
        ids = ev._ids(views[assay]["cell_ids"], assay)
        ev._matrix(views[assay]["shared"], assay, len(ids))
        if not set(pools[assay]) <= set(ids):
            raise ValueError(f"{assay} misses frozen discovery-pool cells; do not shrink the pool")


def external_views(directory, seed):
    directory = Path(directory).resolve()
    model_path, encoding_path = directory / "model_manifest.json", directory / f"{COHORT}/encoding_manifest.json"
    model, encoding = read(model_path), read(encoding_path)
    if (model["status"] != "complete" or model["method"] != "scconfluence"
            or model["configuration"]["seed"] != seed or encoding["status"] != "complete"
            or encoding["cohort"] != COHORT):
        raise ValueError("Require the declared complete scConfluence seed/discovery export")
    for key, filename in (("checkpoint_sha256", "checkpoint.pt"), ("preprocessing_sha256", "preprocessing.json")):
        if model[key] != encoding[key] or model[key] != digest(directory / filename):
            raise ValueError("Checkpoint/preprocessing changed between model and export")
    for record in (*model["inputs"].values(), *encoding["inputs"].values()):
        verify(record)
    for path, sha in model["source_files"].items():
        verify({"path": path, "sha256": sha})
    for entry in model.get("selection_references", {}).values():
        verify(entry)
    arrays = {}
    for assay in ASSAYS:
        file = directory / COHORT / f"{assay}.npz"
        entry = encoding["exports"][assay]
        if Path(entry["path"]).resolve() != file or entry["sha256"] != digest(file):
            raise ValueError("Export identity or bytes changed")
        with np.load(file, allow_pickle=False) as saved:
            if set(saved.files) != {"cell_ids", "shared"}:
                raise ValueError("External embeddings must contain only opaque IDs and coordinates")
            ids, shared = ev._ids(saved["cell_ids"], assay), ev._matrix(saved["shared"], assay)
        if (shared.shape != (entry["n_cells"], entry["latent_dim"]) or len(ids) != len(shared)
                or hashlib.sha256("\n".join(ids).encode()).hexdigest() != entry["cell_ids_sha256"]
                or set(ids) & set(entry["dropped_cell_ids"])):
            raise ValueError("Export dimensions, IDs or retention receipt disagree")
        arrays[assay] = {"cell_ids": ids, "shared": shared}
    return arrays, {"model": pin(model_path), "encoding": pin(encoding_path), "configuration": model["configuration"],
                    "training_inputs": model["inputs"], "encoding_inputs": encoding["inputs"],
                    "selection_references": model.get("selection_references", {})}


def stage(world, name, inputs, calculate):
    path = ROOT / world / "stages" / f"{name}.json"
    binding = {"protocol_sha256": digest(ROOT / "protocol.json"), "inputs": inputs}
    if path.exists():
        saved = read(path)
        if saved["binding"] != binding:
            raise ValueError("Saved causal stage belongs to different inputs")
        return saved["result"]
    result = calculate()
    write(path, {"binding": binding, "result": ev._jsonable(result)})
    return result


def edge_stability(results):
    good = [r for r in results if r["status"] == "ok"]
    counts = {}
    for graph in good:
        for pair in ev._edge_map(graph["edges"]):
            counts[pair] = counts.get(pair, 0) + 1
    return {"status": "ok" if good else "unavailable", "n_attempted": len(results), "n_successful": len(good), "n_failed": len(results) - len(good),
            "denominator": "Successful graphs only; failures are not counted as absent edges",
            "adjacencies": [{"node1": a, "node2": b, "frequency": n / len(good)}
                            for (a, b), n in sorted(counts.items())],
            "interpretation": "Three-draw conditional subsampling stability, not a confidence interval or false-discovery control"}


def run_arm(world, arm, directory=None):
    if arm not in ARMS:
        raise ValueError("Unplanned arm")
    protocol, record, pools = context(world)
    views, provenance = None, {}
    if arm.startswith("glue_"):
        expected, _ = previous.prepare()
        views, provenance = experiment.load_fit(world, arm.removeprefix("glue_"), COHORT, expected)
    elif arm.startswith("scconfluence_"):
        views, provenance = external_views(directory, int(arm[-1]))
        world_path = Path(record["world_manifest"]["path"]).parent
        manifest = read(record["world_manifest"]["path"])
        for assay in ASSAYS:
            expected_selection = record["training_selection_references"][assay]
            actual_selection = provenance["selection_references"].get(assay, {})
            if any(actual_selection.get(k) != v for k, v in expected_selection.items()):
                raise ValueError("External model uses a different training cell/feature selection")
            for cohort, kind in (("train", "training_inputs"), (COHORT, "encoding_inputs")):
                source = world_path / "observed/informative" / cohort / f"{assay}.h5ad"
                entry = provenance[kind][assay]
                if (Path(entry["path"]).resolve() != source or entry["sha256"] != manifest["files"][str(source.relative_to(world_path))]):
                    raise ValueError("External model belongs to different observed world/cohort inputs")
    if views is not None:
        require_pool_coverage(views, pools)
    selections = {"full": pools, **{f"subsample_{s}": subsample(pools, s) for s in SUBSAMPLE_SEEDS}}
    assignments = {}
    if arm not in {"paired_reference", "true_state_reference"}:
        for key, selected in selections.items():
            assignments[key] = ev.match_one_to_one(selected, views) if views is not None else ev.match_one_to_one(
                selected, random_seed=MATCHING_SEED)
    # Learned/random assignments are fixed before loading any latent/count values.
    truth = ev.validate_truth(ev._load_npz(Path(record["public_truth"]["path"])), "informative")
    manifest = read(record["truth_manifest"]["path"])
    candidates, biological = manifest["candidates"], manifest["biological_dag_edges"]
    names = [r["name"] for r in candidates]
    if names != truth["candidate_names"].tolist() or len(names) != 9:
        raise ValueError("Require the unchanged nine-node candidate panel")
    oracle_inputs = {"truth_manifest": record["truth_manifest"]}
    activity_oracle = stage(world, "oracle_activity", oracle_inputs,
                            lambda: ev.oracle_graph(biological, names, method="fci"))
    edges, measured = ev.measurement_dag(biological, candidates)
    count_oracle = stage(world, "oracle_counts", oracle_inputs,
                         lambda: ev._rename_oracle_count_nodes(ev.oracle_graph(edges, measured, method="fci")))
    results = {}
    for key, selected in selections.items():
        if arm == "paired_reference":
            rows = ev._indices(truth["cell_ids"], selected["RNA"])
            arrays = {s: truth[s][rows] for s in ("activities", "counts")}
            assignment = {"cell_ids": {a: selected["RNA"] for a in ASSAYS}, "procedure": "paired_evaluator_reference"}
        else:
            if arm == "true_state_reference":
                state = {a: {"cell_ids": selected[a], "shared": truth["state"][ev._indices(truth["cell_ids"], selected[a])]} for a in ASSAYS}
                assignments[key] = ev.match_one_to_one(selected, state)
            assignment = assignments[key]
            arrays = ev.assemble_readout(assignment, truth)
            for assay in ASSAYS:
                columns = np.flatnonzero(truth["candidate_assays"] == assay)
                ordered = ev._indices(assignment["cell_ids"][assay], selected[assay])
                expected = truth["counts"][np.ix_(ev._indices(truth["cell_ids"], selected[assay]), columns)]
                if not np.array_equal(arrays["counts"][np.ix_(ordered, columns)], expected):
                    raise ValueError("Within-assay observed count block changed")
        binding = {"method": provenance, "assignment": ev._jsonable(assignment), "public_truth": record["public_truth"]}
        readout_path = ROOT / world / "readouts" / f"{arm}_{key}.npz"
        ev._save_npz_new(readout_path, counts=arrays["counts"], candidate_names=np.asarray(names),
                         **{f"cell_ids_{a}": ids for a, ids in assignment["cell_ids"].items()})
        result = {"assignment": ev._jsonable(assignment), "readout": pin(readout_path), "scales": {}}
        for scale in (("counts", "activities") if key == "full" else ("counts",)):
            print(f"{world}/{arm}/{key}/{scale}", flush=True)
            discovered = stage(world, f"{arm}_{key}_{scale}", binding,
                               lambda scale=scale: ev.discover(arrays[scale], names, method="fci", scale=scale, config=CONFIG))
            oracle = count_oracle if scale == "counts" else activity_oracle
            result["scales"][scale] = {"discovery": discovered, "metrics": ev.score_graph(
                discovered, oracle, candidates, target="count_measurement_pag" if scale == "counts" else "activity_projection",
                biological_edges=biological)}
        results[key] = result
    failed = any(item["discovery"]["status"] != "ok" for result in results.values() for item in result["scales"].values())
    output = {"status": "complete_with_failures" if failed else "complete", "arm": arm, "world": world, "protocol_sha256": digest(ROOT / "protocol.json"),
              "provenance": provenance, "results": results, "count_subsample_stability": edge_stability([
                  results[f"subsample_{s}"]["scales"]["counts"]["discovery"] for s in SUBSAMPLE_SEEDS]),
              "scope": "Known synthetic graphical truth; shared-state and direction claims remain separate"}
    write(ROOT / world / "arms" / f"{arm}.json", output)
    return {"world": world, "arm": arm, "counts": results["full"]["scales"]["counts"]["metrics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "baseline", "method"))
    parser.add_argument("--world", choices=WORLDS)
    parser.add_argument("--seed", type=int, choices=(0, 1))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
        result = {"protocol": str(ROOT / "protocol.json"), "sha256": digest(ROOT / "protocol.json")}
    else:
        if args.world is None:
            parser.error("A declared development world is required")
        if args.action == "baseline":
            result = [run_arm(args.world, a) for a in ARMS[:5]]
        else:
            if args.seed is None or args.directory is None:
                parser.error("method requires --seed and --directory")
            result = run_arm(args.world, f"scconfluence_seed{args.seed}", args.directory)
    print(json.dumps(ev._jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
