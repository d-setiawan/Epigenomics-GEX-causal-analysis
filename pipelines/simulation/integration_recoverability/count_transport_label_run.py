"""Exploratory two-sided Bernoulli-label followup on already assessed worlds.

No new data are generated. Fitting reads public disjoint counts only. True
parameters and paired assessment counts are read only after a new fit seal.
Primary matched-experiment code and artifacts are never modified.
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

import count_transport_matched as matched
import count_transport_matched_evaluate as metrics
from count_transport_evaluate import sha, require_hash, within, write_exclusive, bootstrap_indices
from count_transport_label_optimizer import fit_label_model
from count_transport_model import score_rows
from count_transport_run import acceptance, choose

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIMARY_RUN = HERE / "runs" / "count_transport_matched_v1"
DEFAULT_RUN = HERE / "runs" / "count_transport_labels_v1"
MODE = "two_sided_labels"
N_FIT = 12000
OUTPUTS = ("evaluation.json", "comparison.tsv", "aggregate.tsv", "evaluation_seal.json")


def group_key(world: str) -> str:
    return f"{world}__n{N_FIT}__{MODE}"


def _primary(run_dir: Path, protocol: dict, include_private: bool = False) -> tuple[Path, dict, dict]:
    """Check primary provenance; before evaluation hash only public count files."""
    primary = within(REPO, protocol["primary_run"])
    for name, key in (("protocol.json", "primary_protocol_sha256"),
                      ("generation_seal.json", "primary_generation_seal_sha256"),
                      ("data_manifest.json", "primary_data_manifest_sha256"),
                      ("evaluation_seal.json", "primary_evaluation_seal_sha256")):
        require_hash(primary / name, protocol[key])
    generation = json.loads((primary / "generation_seal.json").read_text())
    if (generation["protocol_sha256"] != protocol["primary_protocol_sha256"]
            or generation["data_manifest_sha256"] != protocol["primary_data_manifest_sha256"]):
        raise ValueError("Primary generation provenance differs")
    manifest = json.loads((primary / "data_manifest.json").read_text())
    if set(manifest["worlds"]) != set(protocol["worlds"]):
        raise ValueError("Followup world inventory differs from primary data")
    expected, public = {}, {}
    for world in protocol["worlds"]:
        entry = manifest["worlds"][world]
        if set(entry["public"]) != set(protocol["assays"]):
            raise ValueError("Followup assay inventory differs")
        for item in entry["public"].values():
            public[item["path"]] = item["sha256"]
        for item in [*entry["public"].values(), entry["assessment"], entry["truth"]]:
            if item["path"] in expected:
                raise ValueError("Primary inputs alias a repeated file")
            expected[item["path"]] = item["sha256"]
    if "symmetry_control.json" not in generation["artifacts"]:
        raise ValueError("Primary symmetry control is absent")
    expected["symmetry_control.json"] = generation["artifacts"]["symmetry_control.json"]
    if expected != generation["artifacts"]:
        raise ValueError("Primary generation seal has an unexpected input inventory")
    for name, digest in (expected if include_private else public).items():
        require_hash(within(primary, name), digest)
    return primary, manifest, expected


def verify_sources(run_dir: Path) -> dict:
    protocol = json.loads((run_dir / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        require_hash(within(REPO, path), digest)
    if (protocol["fit_sizes"] != [N_FIT] or protocol["seeds"] != [0, 1, 2, 3]
            or protocol["optimizer_modes"] != [MODE] or protocol["exploratory"] is not True):
        raise ValueError("Unexpected exploratory followup design")
    return protocol


def freeze(run_dir: Path) -> dict:
    """Freeze metadata and code only; generation and assessment are not run."""
    run_dir = Path(run_dir).resolve()
    base = json.loads((PRIMARY_RUN / "protocol.json").read_text())
    if not (PRIMARY_RUN / "evaluation_seal.json").is_file():
        raise ValueError("This declared exploratory followup requires the completed primary evaluation")
    protocol = {"version": "count_transport_labels_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "exploratory": True,
                "assessment_reuse": "These nine worlds and paired assessments were already inspected in the primary experiment; this is not fresh confirmatory evidence.",
                "worlds": base["worlds"], "assays": base["assays"], "widths": base["widths"],
                "fit_sizes": [N_FIT], "calibration_rows": base["calibration_rows"],
                "assessment_rows": base["assessment_rows"], "seeds": [0, 1, 2, 3],
                "optimizer_modes": [MODE], "chunk_iterations": 600, "chunk_evaluations": 15000,
                "gtol": 1e-6, "workers": 2,
                "dimension": "Known true source dimension supplied as an experimental condition",
                "optimizer_change": "Two-sided Bernoulli probabilities followed by one global label canonicalization across assays; no truth initialization or new data",
                "parameter_domain_caveat": "Canonical probabilities can extend below the primary lower bound (approximately 0.00669 rather than 0.05268); changes cannot be attributed solely to label traversal. Generating parameters lie in both classes.",
                "initialization": "Same physical parameter initialization as the primary optimizer, reparameterized without truth values",
                "selection": "Highest equal-assay per-feature calibration LL among gradient-converged finite starts; otherwise highest finite diagnostic. No finite start leaves explicit null selection. No truth-based or paired-assessment reselection.",
                "numerical_gates": base["numerical_gates"], "recovery_thresholds": base["recovery_thresholds"],
                "uncertainty": "Descriptive fixed-fit paired-row percentile95 intervals with1000resamples; no familywise guarantee and no fresh-test claim",
                "primary_run": str(PRIMARY_RUN.relative_to(REPO)),
                "primary_protocol_sha256": sha(PRIMARY_RUN / "protocol.json"),
                "primary_generation_seal_sha256": sha(PRIMARY_RUN / "generation_seal.json"),
                "primary_data_manifest_sha256": sha(PRIMARY_RUN / "data_manifest.json"),
                "primary_evaluation_seal_sha256": sha(PRIMARY_RUN / "evaluation_seal.json"),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    paths = [HERE / name for name in (
        "count_transport_label_run.py", "count_transport_label_optimizer.py",
        "tests/test_count_transport_label_run.py", "tests/test_count_transport_label_optimizer.py",
        "count_transport_model.py", "count_transport_converged.py", "count_transport_run.py",
        "count_transport_matched.py", "count_transport_matched_evaluate.py", "count_transport_evaluate.py",
        "tests/test_count_transport_matched_evaluate.py")]
    paths += [REPO / "project_resources/research/nonlinear_repair_2026_11_15/experiments/C01/model.py",
              REPO / "project_resources/research/sturma_pilot_2026_11_01/prepare_cached_data_pilot.py",
              run_dir / "PLAN.md"]
    protocol["sources"] = {str(path.relative_to(REPO)): sha(path) for path in paths}
    _primary(run_dir, protocol)
    write_exclusive(run_dir / "protocol.json", protocol)
    return protocol


def fit_one(primary: Path, protocol: dict, world: str, entry: dict, seed: int) -> dict:
    start = time.monotonic()
    result = {"world": world, "n_fit": N_FIT, "optimizer_mode": MODE,
              "dimension": entry["dimension"], "variant": "strict", "seed": seed}
    try:
        fit, calibration, scales = matched.training_arrays(primary, entry, protocol, N_FIT)
        with threadpool_limits(limits=1):
            fitted = fit_label_model(fit, entry["dimension"], seed,
                                     chunk_iterations=protocol["chunk_iterations"],
                                     chunk_evaluations=protocol["chunk_evaluations"], gtol=protocol["gtol"])
            model = fitted["model"]
            result.update(model=model, raw_optimizer=fitted.get("raw_optimizer"),
                          continuation=fitted.get("continuation"), error=fitted.get("error"), fit_scale=scales)
            if model is None:
                result.update(calibration_score=None, acceptance={"pass": False})
            else:
                scores = {a: float(score_rows(calibration[a], model["state_weights"], model["state_rates"][a])[0].mean() /
                                   protocol["widths"][a]) for a in protocol["assays"]}
                result.update(calibration_scores=scores, calibration_score=float(np.mean(list(scores.values()))),
                              acceptance=acceptance(model))
    except Exception as error:
        result.update(model=None, calibration_score=None, acceptance={"pass": False},
                      error=f"{type(error).__name__}: {error}")
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def select_group(candidates: dict) -> tuple[str | None, str]:
    finite = any(r.get("model") is not None and r.get("calibration_score") is not None
                 and np.isfinite(r["calibration_score"]) for r in candidates.values())
    return choose(candidates) if finite else (None, "no_finite_fit")


def fit_all(run_dir: Path) -> dict:
    if (run_dir / "fit_seal.json").exists() or (run_dir / "evaluation.json").exists():
        raise FileExistsError("Sealed followup cannot be fitted again")
    protocol = verify_sources(run_dir)
    primary, manifest, _ = _primary(run_dir, protocol)
    (run_dir / "fits").mkdir(exist_ok=False)
    artifacts, results = {}, {}
    tasks = list(itertools.product(protocol["worlds"], protocol["seeds"]))
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = {pool.submit(fit_one, primary, protocol, world, manifest["worlds"][world], seed): (world, seed)
                   for world, seed in tasks}
        for future in as_completed(futures):
            world, seed = futures[future]
            result = future.result()
            name = f"fits/{group_key(world)}__seed{seed}.json"
            write_exclusive(run_dir / name, result)
            artifacts[name], results[name] = sha(run_dir / name), result
            print(json.dumps({"completed": len(results), "total": len(tasks), "fit": name,
                              "calibration": result["calibration_score"], "accepted": result["acceptance"]["pass"],
                              "error": result.get("error")}), flush=True)
    selected, restarts, statuses = {}, {}, {}
    for world in protocol["worlds"]:
        candidates = {name: result for name, result in results.items() if result["world"] == world}
        group = group_key(world)
        selected[group], statuses[group] = select_group(candidates)
        restarts[group] = sorted(candidates, key=lambda name: candidates[name]["seed"])
    seal = {"protocol_sha256": sha(run_dir / "protocol.json"),
            "primary_generation_seal_sha256": protocol["primary_generation_seal_sha256"],
            "primary_data_manifest_sha256": protocol["primary_data_manifest_sha256"],
            "artifacts": artifacts, "selected": selected, "selected_dimension_restarts": restarts,
            "selection_status": statuses, "evaluator_read": False, "exploratory": True}
    write_exclusive(run_dir / "fit_seal.json", seal)
    return seal


def verify_frozen(run_dir: Path) -> tuple[dict, dict, Path, dict, dict, dict]:
    seal = json.loads((run_dir / "fit_seal.json").read_text())
    require_hash(run_dir / "protocol.json", seal["protocol_sha256"])
    if seal.get("evaluator_read") is not False or seal.get("exploratory") is not True:
        raise ValueError("Invalid exploratory fit seal or prior evaluator access")
    protocol = verify_sources(run_dir)
    for key in ("primary_generation_seal_sha256", "primary_data_manifest_sha256"):
        if seal[key] != protocol[key]:
            raise ValueError("Fit and protocol primary data pins differ")
    primary, manifest, input_hashes = _primary(run_dir, protocol, include_private=True)
    results = {}
    for name, digest in seal["artifacts"].items():
        path = within(run_dir, name)
        require_hash(path, digest)
        results[name] = json.loads(path.read_text())
    groups = {group_key(world) for world in protocol["worlds"]}
    if (len(results) != len(groups) * 4 or set(seal["selected"]) != groups
            or set(seal["selected_dimension_restarts"]) != groups or set(seal["selection_status"]) != groups):
        raise ValueError("Incomplete followup fit inventory")
    all_names = []
    for world in protocol["worlds"]:
        group = group_key(world)
        names = seal["selected_dimension_restarts"][group]
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError("Every world must retain all four starts")
        all_names.extend(names)
        seeds = []
        for name in names:
            if name not in results:
                raise ValueError("Unsealed restart artifact")
            result = results[name]
            if (result["world"] != world or result["n_fit"] != N_FIT or result["optimizer_mode"] != MODE
                    or result["dimension"] != manifest["worlds"][world]["dimension"]):
                raise ValueError("Fit identity or known dimension mismatch")
            seeds.append(result["seed"])
            if result.get("model") is not None:
                metrics.validate_model(result["model"], protocol["assays"], protocol["widths"])
        if set(seeds) != set(protocol["seeds"]):
            raise ValueError("Restart seed inventory differs")
        expected_name, expected_status = select_group({name: results[name] for name in names})
        if seal["selected"][group] != expected_name or seal["selection_status"][group] != expected_status:
            raise ValueError("Sealed selection differs from the declared calibration-only rule")
    if len(set(all_names)) != len(results) or set(all_names) != set(results):
        raise ValueError("Restart groups duplicate or omit artifacts")
    return protocol, seal, primary, manifest, results, input_hashes


def evaluate(run_dir: Path) -> dict:
    if any((run_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("Exploratory evaluation outputs already exist")
    protocol, seal, primary, manifest, artifacts, input_hashes = verify_frozen(run_dir)
    counts, truths, audits = metrics.load_evaluation_inputs(primary, protocol, manifest)
    oracle_losses = metrics.oracle_training_objectives(primary, protocol, manifest, truths)
    boot = {world: bootstrap_indices(len(next(iter(block.values()))), f"matched:{world}", 1000)
            for world, block in counts.items()}
    report = {"exploratory": True, "assessment_reuse": protocol["assessment_reuse"],
              "target": "Known-dimension count-law and log-rate transport on reused exact-model worlds; not causal direction recovery",
              "optimizer_change": protocol["optimizer_change"], "parameter_domain_caveat": protocol["parameter_domain_caveat"],
              "selection": protocol["selection"], "uncertainty": protocol["uncertainty"],
              "direction_interpretation": "Ordered source/target labels describe prediction. Same-model marginal per-cell gains are symmetric associations, not causal directions.",
              "optimization_interpretation": "Gradient convergence does not imply a global optimum. The generating-point training objective is evaluated only after seals; positive gaps certify suboptimality, negative gaps do not certify optimality.",
              "recovery_thresholds": protocol["recovery_thresholds"], "input_audits": audits,
              "selected": seal["selected"], "selection_status": seal["selection_status"],
              "fits": {}, "restart_stability": {}}
    rows = []
    for world in protocol["worlds"]:
        group = group_key(world)
        selected = seal["selected"][group]
        names = seal["selected_dimension_restarts"][group]
        for name in names:
            result, _ = metrics.evaluate_fit(artifacts[name], counts[world], truths[world], protocol["assays"], protocol["widths"],
                                             boot[world] if name == selected else None, protocol["recovery_thresholds"],
                                             oracle_losses[world][N_FIT])
            report["fits"][name] = result
        report["restart_stability"][group] = metrics.restart_summary(report["fits"], names)
        # An all-failed group remains visible, with no invented selected fit.
        summary = metrics.fit_summary(report["fits"][selected if selected is not None else names[0]])
        if selected is None:
            summary["seed"] = None
        summary.update(selection_status=seal["selection_status"][group], selected_artifact=selected, exploratory=True)
        rows.append(summary)
    report["aggregate"] = metrics.aggregate_rows(rows)
    for row in report["aggregate"]:
        subset = [r for r in rows if r["dimension"] == row["dimension"]]
        row["no_finite_fit_groups"] = sum(r["selection_status"] == "no_finite_fit" for r in subset)
        row["exploratory"] = True
    write_exclusive(run_dir / "evaluation.json", report)
    metrics.write_table(run_dir / "comparison.tsv", rows)
    metrics.write_table(run_dir / "aggregate.tsv", report["aggregate"])
    write_exclusive(run_dir / "evaluation_seal.json", {
        "protocol_sha256": sha(run_dir / "protocol.json"), "fit_seal_sha256": sha(run_dir / "fit_seal.json"),
        "primary_generation_seal_sha256": protocol["primary_generation_seal_sha256"],
        "primary_data_manifest_sha256": protocol["primary_data_manifest_sha256"],
        "primary_evaluation_seal_sha256": protocol["primary_evaluation_seal_sha256"],
        "primary_inputs": input_hashes,
        "outputs": {name: sha(run_dir / name) for name in OUTPUTS if name != "evaluation_seal.json"},
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
    print(json.dumps({"stage": args.stage, "run_dir": str(run_dir),
                      "selected_groups": len(result.get("selected", {}))}))


if __name__ == "__main__":
    main()
