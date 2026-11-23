"""Sealed exploratory source-search comparison on existing count datasets.

Only orchestration is new here. Public input validation, calibration selection,
exact transport metrics and the post-seal generating-point witness are reused.
No primary or previous-followup source/artifact is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

import count_transport_label_run as prior
from count_transport_source_search import fit_source_search

HERE, REPO = prior.HERE, prior.REPO
PRIMARY_RUN = prior.PRIMARY_RUN
DEFAULT_RUN = HERE / "runs" / "count_transport_search_v1"
MODES = ("source_search", "plain_unbounded")
N_FIT = 12000
OUTPUTS = prior.OUTPUTS
metrics = prior.metrics


def group_key(world: str, mode: str) -> str:
    return f"{world}__n{N_FIT}__{mode}"


def freeze(run_dir: Path) -> dict:
    """Freeze code, provenance, and finite start inventory before any new fit."""
    run_dir = Path(run_dir).resolve()
    base = json.loads((PRIMARY_RUN / "protocol.json").read_text())
    protocol = {key: base[key] for key in ("worlds", "assays", "widths", "calibration_rows", "assessment_rows",
                                         "numerical_gates", "recovery_thresholds")}
    protocol.update({
        "version": "count_transport_search_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory": True,
        "assessment_reuse": "The same nine worlds and paired assessments were already inspected. This is exploratory algorithm development, not fresh confirmatory evidence.",
        "fit_sizes": [N_FIT], "seeds": [0, 1, 2, 3], "optimizer_modes": list(MODES),
        "mixture_starts": 4, "refine_candidates": 4,
        "chunk_iterations": 600, "chunk_evaluations": 15000, "gtol": 1e-6, "workers": 2,
        "em_gtol": 1e-6, "relative_objective_tolerance": 0, "line_search_max_steps": 40,
        "maximum_marginal_mixture_runs": 432, "maximum_joint_refinements": 180,
        "total_iteration_limit": None, "total_evaluation_limit": None, "finite_parameter_boxes": False,
        "dimension": "Known true source dimension supplied as an experimental condition",
        "optimizer_change": "Exact log-rate reparameterization eta=(bits+h)A with source probabilities entering mixture weights only; raw-count Poisson-mixture and cube-orbit informed starts versus physical-primary-equivalent plain starts",
        "finite_search_inventory": "Four outer seeds per mode and world; informed mode has four marginal-mixture starts per assay, exhaustive cube orbits (1/3/840 for d=1/2/3), up to64 cross-assay combinations scored by training likelihood, and the best four candidates refined. This finite search does not guarantee a global optimum.",
        "inner_selection": "Public training objective only; no calibration, truth, state, pairing or assessment data enter initialization or inner candidate selection",
        "selection": "Outer starts selected by the unchanged calibration-only rule: highest equal-assay per-feature calibration LL among gradient-converged finite starts, otherwise highest finite diagnostic; no finite start leaves explicit null selection.",
        "uncertainty": "Descriptive fixed-fit paired-row percentile95 intervals using1000resamples; no familywise, global-optimum or fresh-assessment guarantee",
        "primary_run": str(PRIMARY_RUN.relative_to(REPO)),
        "primary_protocol_sha256": prior.sha(PRIMARY_RUN / "protocol.json"),
        "primary_generation_seal_sha256": prior.sha(PRIMARY_RUN / "generation_seal.json"),
        "primary_data_manifest_sha256": prior.sha(PRIMARY_RUN / "data_manifest.json"),
        "primary_evaluation_seal_sha256": prior.sha(PRIMARY_RUN / "evaluation_seal.json"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    })
    paths = [HERE / name for name in (
        "count_transport_search_run.py", "count_transport_source_search.py",
        "tests/test_count_transport_search_run.py", "tests/test_count_transport_source_search.py",
        "count_transport_label_run.py", "count_transport_label_optimizer.py",
        "count_transport_model.py", "count_transport_converged.py", "count_transport_run.py",
        "count_transport_matched.py", "count_transport_matched_evaluate.py", "count_transport_evaluate.py",
        "tests/test_count_transport_matched_evaluate.py")]
    paths += [REPO / "project_resources/research/nonlinear_repair_2026_11_15/experiments/C01/model.py",
              REPO / "project_resources/research/sturma_pilot_2026_11_01/prepare_cached_data_pilot.py", run_dir / "PLAN.md"]
    protocol["sources"] = {str(path.relative_to(REPO)): prior.sha(path) for path in paths}
    prior._primary(run_dir, protocol)
    prior.write_exclusive(run_dir / "protocol.json", protocol)
    return protocol


def verify_sources(run_dir: Path) -> dict:
    protocol = json.loads((run_dir / "protocol.json").read_text())
    for name, digest in protocol["sources"].items():
        prior.require_hash(prior.within(REPO, name), digest)
    if (protocol["fit_sizes"] != [N_FIT] or protocol["seeds"] != [0, 1, 2, 3]
            or protocol["optimizer_modes"] != list(MODES) or protocol["exploratory"] is not True
            or protocol["total_iteration_limit"] is not None or protocol["total_evaluation_limit"] is not None
            or protocol["finite_parameter_boxes"] is not False):
        raise ValueError("Unexpected source-search comparison design")
    return protocol


def fit_one(primary: Path, protocol: dict, world: str, entry: dict, mode: str, seed: int) -> dict:
    started = time.monotonic()
    result = {"world": world, "dimension": entry["dimension"], "n_fit": N_FIT,
              "optimizer_mode": mode, "variant": "strict", "seed": seed}
    try:
        fit, calibration, scales = prior.matched.training_arrays(primary, entry, protocol, N_FIT)
        with threadpool_limits(limits=1):
            fitted = fit_source_search(fit, entry["dimension"], seed, mode=mode,
                                       mixture_starts=protocol["mixture_starts"],
                                       refine_candidates=protocol["refine_candidates"],
                                       chunk_iterations=protocol["chunk_iterations"],
                                       chunk_evaluations=protocol["chunk_evaluations"], gtol=protocol["gtol"])
            conflicts = [key for key, value in result.items() if key in fitted and fitted[key] != value]
            if conflicts:
                result["rejected_search_payload"] = fitted
                raise ValueError(f"Search payload changes outer identity: {conflicts}")
            # Retain every inner model, initialization/search diagnostic,
            # numerical failure and continuation record returned by the search.
            result.update(fitted)
            result["fit_scale"] = scales
            if result["model"] is None:
                result.update(calibration_score=None, acceptance={"pass": False})
            else:
                model = result["model"]
                scores = {a: float(prior.score_rows(calibration[a], model["state_weights"], model["state_rates"][a])[0].mean() /
                                   protocol["widths"][a]) for a in protocol["assays"]}
                result.update(calibration_scores=scores, calibration_score=float(np.mean(list(scores.values()))),
                              acceptance=prior.acceptance(model))
    except Exception as error:
        if result.get("model") is not None:
            result["model_before_scoring_failure"] = result["model"]
        result.update(model=None, calibration_score=None, acceptance={"pass": False},
                      error=f"{type(error).__name__}: {error}")
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def fit_all(run_dir: Path) -> dict:
    if (run_dir / "fit_seal.json").exists() or (run_dir / "evaluation.json").exists():
        raise FileExistsError("Sealed source-search runs cannot be fitted again")
    protocol = verify_sources(run_dir)
    primary, manifest, _ = prior._primary(run_dir, protocol)
    (run_dir / "fits").mkdir(exist_ok=False)
    tasks = list(itertools.product(protocol["worlds"], protocol["optimizer_modes"], protocol["seeds"]))
    results, artifacts = {}, {}
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        pending = {pool.submit(fit_one, primary, protocol, world, manifest["worlds"][world], mode, seed): (world, mode, seed)
                   for world, mode, seed in tasks}
        for future in as_completed(pending):
            world, mode, seed = pending[future]
            result = future.result()
            name = f"fits/{group_key(world, mode)}__seed{seed}.json"
            prior.write_exclusive(run_dir / name, result)
            results[name], artifacts[name] = result, prior.sha(run_dir / name)
            print(json.dumps({"completed": len(results), "total": len(tasks), "fit": name,
                              "calibration": result["calibration_score"], "accepted": result["acceptance"]["pass"],
                              "error": result.get("error")}), flush=True)
    selected, restarts, statuses = {}, {}, {}
    for world, mode in itertools.product(protocol["worlds"], protocol["optimizer_modes"]):
        group = group_key(world, mode)
        candidates = {name: result for name, result in results.items() if result["world"] == world and result["optimizer_mode"] == mode}
        selected[group], statuses[group] = prior.select_group(candidates)
        restarts[group] = sorted(candidates, key=lambda name: candidates[name]["seed"])
    seal = {"protocol_sha256": prior.sha(run_dir / "protocol.json"),
            "primary_generation_seal_sha256": protocol["primary_generation_seal_sha256"],
            "primary_data_manifest_sha256": protocol["primary_data_manifest_sha256"],
            "artifacts": artifacts, "selected": selected, "selected_dimension_restarts": restarts,
            "selection_status": statuses, "evaluator_read": False, "exploratory": True}
    prior.write_exclusive(run_dir / "fit_seal.json", seal)
    return seal


def verify_frozen(run_dir: Path) -> tuple:
    seal = json.loads((run_dir / "fit_seal.json").read_text())
    prior.require_hash(run_dir / "protocol.json", seal["protocol_sha256"])
    if seal.get("evaluator_read") is not False or seal.get("exploratory") is not True:
        raise ValueError("Invalid exploratory fit seal or prior evaluator access")
    protocol = verify_sources(run_dir)
    for key in ("primary_generation_seal_sha256", "primary_data_manifest_sha256"):
        if seal[key] != protocol[key]:
            raise ValueError("Primary data pins differ between fit seal and protocol")
    primary, manifest, input_hashes = prior._primary(run_dir, protocol, include_private=True)
    artifacts = {}
    for name, digest in seal["artifacts"].items():
        path = prior.within(run_dir, name)
        prior.require_hash(path, digest)
        artifacts[name] = json.loads(path.read_text())
    groups = {group_key(world, mode) for world, mode in itertools.product(protocol["worlds"], protocol["optimizer_modes"])}
    if (len(artifacts) != len(groups) * len(protocol["seeds"]) or set(seal["selected"]) != groups
            or set(seal["selected_dimension_restarts"]) != groups or set(seal["selection_status"]) != groups):
        raise ValueError("Incomplete source-search fit inventory")
    all_names = []
    for world, mode in itertools.product(protocol["worlds"], protocol["optimizer_modes"]):
        group = group_key(world, mode)
        names = seal["selected_dimension_restarts"][group]
        if len(names) != len(protocol["seeds"]) or len(set(names)) != len(names):
            raise ValueError("Every mode/world must retain every outer start")
        all_names.extend(names)
        seeds = []
        for name in names:
            if name not in artifacts:
                raise ValueError("Unsealed outer start")
            item = artifacts[name]
            if (item["world"] != world or item["optimizer_mode"] != mode or item["n_fit"] != N_FIT
                    or item["dimension"] != manifest["worlds"][world]["dimension"]):
                raise ValueError("Outer start identity or known dimension differs")
            seeds.append(item["seed"])
            if item.get("model") is not None:
                metrics.validate_model(item["model"], protocol["assays"], protocol["widths"])
        if set(seeds) != set(protocol["seeds"]):
            raise ValueError("Outer seed inventory differs")
        expected_name, expected_status = prior.select_group({name: artifacts[name] for name in names})
        if seal["selected"][group] != expected_name or seal["selection_status"][group] != expected_status:
            raise ValueError("Sealed outer selection differs from calibration-only selection")
    if len(set(all_names)) != len(artifacts) or set(all_names) != set(artifacts):
        raise ValueError("Groups duplicate or omit sealed outer starts")
    return protocol, seal, primary, manifest, artifacts, input_hashes


def evaluate(run_dir: Path) -> dict:
    if any((run_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("Source-search evaluation outputs already exist")
    protocol, seal, primary, manifest, artifacts, input_hashes = verify_frozen(run_dir)
    counts, truths, audits = metrics.load_evaluation_inputs(primary, protocol, manifest)
    oracle_losses = metrics.oracle_training_objectives(primary, protocol, manifest, truths)
    bootstrap = {world: prior.bootstrap_indices(len(next(iter(block.values()))), f"matched:{world}", 1000)
                 for world, block in counts.items()}
    report = {key: protocol[key] for key in ("exploratory", "assessment_reuse", "optimizer_change", "finite_search_inventory",
                                           "inner_selection", "selection", "uncertainty", "recovery_thresholds")}
    report.update(target="Known-dimension count-law and log-rate transport, not causal directions",
                  optimization_interpretation="No finite parameter boxes or cumulative iteration/evaluation caps; finite start/candidate inventory and numerical stopping remain explicit. Gradient convergence does not certify a global optimum.",
                  direction_interpretation="Same-model marginal per-cell predictive gains are symmetric associations; ordered assay labels are not causal arrows.",
                  input_audits=audits, selected=seal["selected"], selection_status=seal["selection_status"],
                  fits={}, restart_stability={})
    rows = []
    for world, mode in itertools.product(protocol["worlds"], protocol["optimizer_modes"]):
        group = group_key(world, mode)
        selected, names = seal["selected"][group], seal["selected_dimension_restarts"][group]
        for name in names:
            point, _ = metrics.evaluate_fit(artifacts[name], counts[world], truths[world], protocol["assays"], protocol["widths"],
                                            bootstrap[world] if name == selected else None, protocol["recovery_thresholds"],
                                            oracle_losses[world][N_FIT])
            point["search_diagnostics"] = artifacts[name].get("search_diagnostics")
            point["candidate_records_in_fit_artifact"] = len(artifacts[name].get("candidates") or [])
            report["fits"][name] = point
        report["restart_stability"][group] = metrics.restart_summary(report["fits"], names)
        summary = metrics.fit_summary(report["fits"][selected if selected is not None else names[0]])
        if selected is None:
            summary["seed"] = None
        summary.update(selection_status=seal["selection_status"][group], selected_artifact=selected, exploratory=True)
        rows.append(summary)
    report["aggregate"] = metrics.aggregate_rows(rows)
    for row in report["aggregate"]:
        subset = [r for r in rows if r["dimension"] == row["dimension"] and r["optimizer_mode"] == row["optimizer_mode"]]
        row["no_finite_fit_groups"] = sum(r["selection_status"] == "no_finite_fit" for r in subset)
        row["exploratory"] = True
    prior.write_exclusive(run_dir / "evaluation.json", report)
    metrics.write_table(run_dir / "comparison.tsv", rows)
    metrics.write_table(run_dir / "aggregate.tsv", report["aggregate"])
    prior.write_exclusive(run_dir / "evaluation_seal.json", {
        "protocol_sha256": prior.sha(run_dir / "protocol.json"), "fit_seal_sha256": prior.sha(run_dir / "fit_seal.json"),
        "primary_generation_seal_sha256": protocol["primary_generation_seal_sha256"],
        "primary_data_manifest_sha256": protocol["primary_data_manifest_sha256"],
        "primary_evaluation_seal_sha256": protocol["primary_evaluation_seal_sha256"],
        "primary_inputs": input_hashes, "outputs": {name: prior.sha(run_dir / name) for name in OUTPUTS if name != "evaluation_seal.json"},
        "exploratory": True, "truth_used_only_after_fit_seal": True, "no_reselection": True,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "evaluate"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = {"freeze": freeze, "fit": fit_all, "evaluate": evaluate}[args.stage](run_dir)
    print(json.dumps({"stage": args.stage, "run_dir": str(run_dir), "selected_groups": len(result.get("selected", {}))}))


if __name__ == "__main__":
    main()
