"""Frozen-pool state diagnostics for the four declared simulated scConfluence fits."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "pipelines/simulation/integration_recoverability/scripts"))
import causal_reliability_pilot as pilot

experiment = pilot.experiment
ev = pilot.ev
read, write, digest, pin, verify = pilot.read, pilot.write, pilot.digest, pilot.pin, pilot.verify
RESEARCH = REPO / "project_resources/research/scconfluence_scmtr_pilot_2026_11_01"
ROOT = experiment.BASE / "runs/integration_validation/20261101_scconfluence_state_v1"
PROTOCOL = ROOT / "protocol.json"


def subset_frozen(views, pools):
    """Reject missing cells before restricting to exactly the prior ordered pool."""
    pilot.require_pool_coverage(views, pools)
    return {a: {"cell_ids": np.asarray(pools[a], dtype=str),
                "shared": views[a]["shared"][ev._indices(views[a]["cell_ids"], pools[a])]}
            for a in pilot.ASSAYS}


def require_same_pools(actual, expected):
    if ev._jsonable(actual) != ev._jsonable(expected):
        raise ValueError("Frozen state-evaluation pools changed; no shrinking or reselection")


def freeze():
    plan_path = RESEARCH / "execution_plan.json"
    plan = read(plan_path)
    jobs = [j for j in plan["jobs"] if j["name"] in experiment.WORLDS]
    if {(j["name"], j["seed"]) for j in jobs} != {
            (w, s) for w in experiment.WORLDS for s in (0, 1)} or len(jobs) != 4:
        raise ValueError("Require exactly the four declared simulated fits")
    previous = experiment.ROOT
    completion = read(previous / "completion.json")
    reports, pools = {}, {}
    for world in experiment.WORLDS:
        record = completion["corrected_reports"][world]
        verify(record)
        report = read(record["path"])
        if report["status"] != "complete" or report["world_id"] != world or report["cohort"] != "val":
            raise ValueError("Require the completed prior public validation report")
        reports[world], pools[world] = record, report["pools"]
        if set(pools[world]) != {"train", "val"}:
            raise ValueError("Unexpected prior cohorts")
    sources = [__file__, pilot.__file__, experiment.__file__, pilot.previous.__file__,
               ev.__file__, experiment.validation.__file__, experiment.metrics.__file__]
    protocol = {
        "status": "frozen", "worlds": list(experiment.WORLDS), "seeds": [0, 1],
        "cohort": "val", "execution_plan": pin(plan_path),
        "previous_protocol": pin(previous / "protocol.json"),
        "previous_runtime_amendment": pin(previous / "runtime_scoring_amendment.json"),
        "previous_reports": reports, "pools": pools,
        "sources": {str(Path(p).resolve()): digest(Path(p)) for p in sources},
        "original_source_pins": experiment.validation.protocol()["source_pins"],
        "primary": "Worst of all three normalized squared-state transport gaps; lower is better",
        "secondary": "All pair gaps, train-source state transfer R2 without target refit, pre-assignment hubness and variance",
        "reference_fidelity_only": "Within-assay PCA/LSI neighbor retention is not biological truth",
        "matching": "Unchanged evaluate_views: RNA-anchored Euclidean one-to-one assignment",
        "pool_policy": "Exactly the previous ordered train/val pools; missing coverage is failure, never shrink",
        "limits": ["State labels enter only evaluator calculations after matching; no causal graph inference",
                   "Full-pool forced matching is a separate diagnostic from native half-mass partial OT",
                   "Two previously inspected development worlds, two scConfluence seeds, one seed per GLUE arm",
                   "Repeated seeds are not independent biological replicates; no tuning or assessment access"]}
    write(PROTOCOL, protocol)
    return protocol


def context(world, seed):
    spec = read(PROTOCOL)
    if world not in spec["worlds"] or seed not in spec["seeds"]:
        raise ValueError("Unplanned world or integration seed")
    for key in ("execution_plan", "previous_protocol", "previous_runtime_amendment"):
        verify(spec[key])
    for record in spec["previous_reports"].values():
        verify(record)
    for path, sha in spec["sources"].items():
        verify({"path": path, "sha256": sha})
    if experiment.validation.protocol()["source_pins"] != spec["original_source_pins"]:
        raise ValueError("Original scientific sources changed")
    plan = read(spec["execution_plan"]["path"])
    for path, sha in plan["sources"].items():
        verify({"path": path, "sha256": sha})
    job = next(j for j in plan["jobs"] if j["name"] == world and j["seed"] == seed)
    expected = read(spec["previous_protocol"]["path"])
    amendment = read(spec["previous_runtime_amendment"]["path"])
    if amendment["change"] != {"field": "inherited_loader_environment.CUDA_VISIBLE_DEVICES",
                               "from": None, "to": "0"}:
        raise ValueError("Unexpected prior runtime amendment")
    expected["inherited_loader_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    return spec, plan, job, expected


def external(job, plan, cohort):
    directory = Path(job["output"])
    # The existing verifier is shared with empirical scoring; restore its cohort immediately.
    with patch.object(pilot, "COHORT", cohort):
        views, evidence = pilot.external_views(directory, job["seed"])
    model = read(directory / "model_manifest.json")
    encoding = read(directory / cohort / "encoding_manifest.json")
    if (model["configuration"] != job["configuration"]
            or model["inputs"] != job["input_records"]["train"]
            or encoding["inputs"] != job["input_records"][cohort]):
        raise ValueError("Model configuration or source cohorts differ from the fixed execution plan")
    if any(plan["sources"].get(path) != sha for path, sha in model["source_files"].items()):
        raise ValueError("Model source evidence differs from the fixed execution plan")
    selections = model["selection_references"]
    if set(selections) != set(pilot.ASSAYS) or any(
            {k: selections[a][k] for k in ("path", "sha256")} != job["selection_records"][a]
            for a in pilot.ASSAYS):
        raise ValueError("Training selection references differ from the fixed execution plan")
    return views, evidence


def score(world, seed):
    spec, plan, job, expected = context(world, seed)
    target = ROOT / "reports" / f"{world}_seed{seed}.json"
    if target.exists():
        raise FileExistsError(f"Preserve existing state report: {target}")
    pools = spec["pools"][world]
    runs, train_runs, evidence = {}, {}, {}
    for mode in experiment.MODES:
        label = f"glue_{mode}"
        evidence[label] = {}
        for cohort, destination in (("train", train_runs), ("val", runs)):
            views, proof = experiment.load_fit(world, mode, cohort, expected)
            destination[label] = subset_frozen(views, pools[cohort])
            evidence[label][cohort] = proof
    label = f"scconfluence_seed{seed}"
    evidence[label] = {}
    for cohort, destination in (("train", train_runs), ("val", runs)):
        views, proof = external(job, plan, cohort)
        destination[label] = subset_frozen(views, pools[cohort])
        evidence[label][cohort] = proof
    report = experiment.evaluate_views(world, runs, train_runs, evidence, cohort="val")
    require_same_pools(report["pools"], pools)
    previous = read(spec["previous_reports"][world]["path"])
    for mode in experiment.MODES:
        before, after = previous["results"][mode], report["results"][f"glue_{mode}"]
        np.testing.assert_allclose(after["worst_pair_gap"], before["worst_pair_gap"], rtol=0, atol=1e-8)
        for pair in before["pairwise_state_transport"]:
            np.testing.assert_allclose(after["pairwise_state_transport"][pair]["normalized_gap"],
                                       before["pairwise_state_transport"][pair]["normalized_gap"], rtol=0, atol=1e-8)
    report.update(state_protocol=pin(PROTOCOL), execution_plan=spec["execution_plan"],
                  previous_report=spec["previous_reports"][world], scoring_sources=spec["sources"],
                  seed=seed, pools_exactly_preserved=True, baseline_transport_reproduced_atol=1e-8,
                  interpretation="State alignment diagnostic; no causal recovery or residual coupling claim")
    write(target, report)
    return {"report": str(target), "sha256": digest(target),
            "worst_pair_gap": {k: r["worst_pair_gap"] for k, r in report["results"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--world", choices=experiment.WORLDS)
    parser.add_argument("--seed", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
        result = pin(PROTOCOL)
    else:
        if args.world is None or args.seed is None:
            parser.error("score requires --world and --seed")
        result = score(args.world, args.seed)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
