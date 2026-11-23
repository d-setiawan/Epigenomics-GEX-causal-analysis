"""Exact E2 simulation and paired capped/convergence-driven fitting.

Generation knows the parameters. Fitting reads only public disjoint counts and
the explicitly supplied model dimension. Evaluation is a separate sealed stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from count_transport_converged import fit_convergence_pair
from count_transport_evaluate import sha, require_hash, within, write_exclusive
from count_transport_model import score_rows
from count_transport_run import acceptance, choose

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RUN = HERE / "runs" / "count_transport_matched_v1"
WIDTHS = {"RNA": 30, "H3K27ac": 24, "H3K4me3": 18}


def rng_for(*parts):
    seed = int.from_bytes(hashlib.sha256("|".join(map(str, ("matched-e2-v1", *parts))).encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def true_model(dimension, replicate, widths=WIDTHS):
    """Independent construction, without calling the estimator's decoder."""
    if dimension not in (1, 2, 3) or replicate not in (0, 1, 2):
        raise ValueError("The frozen design has dimensions 1..3 and replicates 0..2")
    p = np.array([.12, .27, .40])[:dimension]
    mu = np.array([.6, -.35, .25])[:dimension]
    bits = np.array(list(itertools.product((0., 1.), repeat=dimension)))
    z = (bits-p) / np.sqrt(p*(1-p))
    weights = np.prod(np.where(bits, p, 1-p), axis=1)
    loadings, log_rates, rates, ratios = {}, {}, {}, {}
    for assay, width in widths.items():
        q, _ = np.linalg.qr(rng_for("mechanism", dimension, replicate, assay).normal(size=(width, dimension)))
        matrix = q * np.sqrt(width) * .45
        if matrix.shape != (width, dimension) or np.max(np.abs(matrix)) >= 6:
            raise ValueError("True loading is outside the fitted full-rank parameter class")
        loadings[assay] = matrix.tolist()
        log_rates[assay] = ((z+mu) @ matrix.T).tolist()
        rates[assay] = np.exp(log_rates[assay]).tolist()
        singular = np.linalg.svd(matrix, compute_uv=False)
        ratios[assay] = float(singular[-1]/singular[0])
    np.testing.assert_allclose(weights @ z, 0, atol=1e-14)
    np.testing.assert_allclose(z.T @ (weights[:, None]*z), np.eye(dimension), atol=1e-14)
    raw_p = np.log((p-.05)/(.45-p))
    assert np.max(np.abs(raw_p)) < 5 and np.max(np.abs(mu)) < 20
    return {"dimension": dimension, "replicate": replicate, "variant": "strict",
            "probabilities": p.tolist(), "source_mean": mu.tolist(),
            "loadings": loadings, "state_bits": bits.tolist(), "state_sources": (z+mu).tolist(),
            "state_weights": weights.tolist(), "state_log_rates": log_rates, "state_rates": rates,
            "singular_value_ratios": ratios,
            "assumptions": {"independent_unit_variance_nongaussian_sources": True,
                            "distinct_probabilities_below_half": True,
                            "full_column_rank": True, "strict_log_linear_rates": True,
                            "conditionally_independent_poisson_noise_all_assays": True,
                            "private_factors": False, "extra_depth": False,
                            "true_parameters_strictly_inside_optimizer_bounds": True}}


def draw_sources(model, n, *key):
    p, mu = np.array(model["probabilities"]), np.array(model["source_mean"])
    bits = (rng_for(*key, "sources").random((n, len(p))) < p).astype(float)
    return (bits-p)/np.sqrt(p*(1-p)) + mu


def draw_counts(model, assay, source, *key):
    rate = np.exp(source @ np.asarray(model["loadings"][assay]).T)
    if not np.isfinite(rate).all() or np.any(rate <= 0):
        raise ValueError("Nonfinite generated rate; no clipping is permitted")
    return rng_for(*key, "count_noise", assay).poisson(rate)


def save_npz(path, **arrays):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def freeze(run_dir):
    sources = [HERE / name for name in (
        "count_transport_matched.py", "count_transport_converged.py", "count_transport_matched_evaluate.py",
        "count_transport_model.py", "count_transport_evaluate.py", "count_transport_run.py",
        "tests/test_count_transport_matched.py", "tests/test_count_transport_converged.py",
        "tests/test_count_transport_matched_evaluate.py")]
    sources += [REPO / "project_resources/research/nonlinear_repair_2026_11_15/experiments/C01/model.py",
                run_dir / "PLAN.md"]
    protocol = {
        "version": "count_transport_matched_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requested simulations without model mismatch and optimization budget limits; targeted new simulated counts are authorized",
        "worlds": [f"d{d}_rep{r}" for d in (1, 2, 3) for r in range(3)],
        "assays": list(WIDTHS), "widths": WIDTHS, "fit_sizes": [3000, 12000],
        "calibration_rows": 2000, "assessment_rows": 4000, "seeds": [0, 1, 2, 3],
        "optimizer_modes": ["capped600", "convergence"], "chunk_iterations": 600,
        "chunk_evaluations": 15000, "gtol": 1e-6, "workers": 2,
        "dimension": "Known and supplied as a controlled design condition; no dimension-discovery claim",
        "selection": "Highest equal-assay per-feature calibration LL among gradient-converged finite fits; otherwise highest finite diagnostic. Exact ties lower seed. No assessment-based reselection.",
        "numerical_gates": {"projected_gradient": 1e-6, "minimum_loading_singular_ratio": .001,
                            "minimum_probability_gap": .03},
        "recovery_thresholds": {"covariance_relative_error": .10, "transport_relative_error": .10,
                                "oracle_conditional_gap_per_feature": .02, "positive_nominal_gain_ci95": True},
        "uncertainty": "Nominal fixed-fit paired bootstrap95 intervals,1000replicates; no familywise or global-optimization guarantee",
        "sources": {str(path.relative_to(REPO)): sha(path) for path in sources},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    write_exclusive(run_dir / "protocol.json", protocol)
    return protocol


def verify_sources(run_dir):
    protocol = json.loads((run_dir / "protocol.json").read_text())
    for name, expected in protocol["sources"].items():
        require_hash(within(REPO, name), expected)
    return protocol


def symmetry_control():
    model = true_model(2, 0)
    p, mu = np.array([.27, .27]), np.array([.6, .6])
    bits = np.array(model["state_bits"])
    z = (bits-p)/np.sqrt(p*(1-p)) + mu
    weights = np.prod(np.where(bits, p, 1-p), axis=1)
    a, b = (np.asarray(model["loadings"][assay]) for assay in ("RNA", "H3K27ac"))
    ra, rb, swapped = np.exp(z @ a.T), np.exp(z @ b.T), np.exp(z @ b[:, ::-1].T)
    index = np.array([next(j for j, other in enumerate(bits) if np.array_equal(other, row[::-1])) for row in bits])
    np.testing.assert_allclose(swapped, rb[index], atol=1e-12)
    np.testing.assert_allclose(weights, weights[index], atol=1e-14)
    covariance = (ra-weights@ra).T @ (weights[:, None]*(rb-weights@rb))
    alternative = (ra-weights@ra).T @ (weights[:, None]*(swapped-weights@swapped))
    gap = float(np.linalg.norm(covariance-alternative)/np.linalg.norm(covariance))
    assert gap > .01
    return {"kind": "Exact analytic symmetry countercontrol; no simulated count sample or fitted result",
            "probabilities": p.tolist(), "source_mean": mu.tolist(),
            "source_swap_row_permutation": index.tolist(),
            "maximum_marginal_rate_atom_error": float(np.max(np.abs(swapped-rb[index]))),
            "relative_joint_crosscovariance_difference": gap,
            "interpretation": "Equal-probability/equal-mean source swap in one assay preserves every unpaired count law but changes joint covariance; violates the no-symmetry premise"}


def generate(run_dir):
    protocol = verify_sources(run_dir)
    root = run_dir / "data"
    root.mkdir(exist_ok=False)
    manifest = {"worlds": {}}
    artifacts = {}
    for world in protocol["worlds"]:
        dimension, replicate = int(world[1]), int(world[-1])
        model = true_model(dimension, replicate, protocol["widths"])
        directory = root / world
        directory.mkdir()
        truth_path = directory / "truth.json"
        write_exclusive(truth_path, model)
        entry = {"dimension": dimension, "replicate": replicate, "public": {},
                 "truth": {"path": str(truth_path.relative_to(run_dir)), "sha256": sha(truth_path)}}
        assessment = {"cell_ids": np.array([f"{world}:assessment:paired:{i}" for i in range(protocol["assessment_rows"])])}
        shared = draw_sources(model, protocol["assessment_rows"], world, "assessment")
        for assay in protocol["assays"]:
            features = np.array([f"{assay}:feature:{j}" for j in range(protocol["widths"][assay])])
            public = {"feature_ids": features}
            for stage, n in (("fit", max(protocol["fit_sizes"])), ("calibration", protocol["calibration_rows"])):
                source = draw_sources(model, n, world, stage, assay)
                public[f"{stage}_counts"] = draw_counts(model, assay, source, world, stage, assay)
                public[f"{stage}_ids"] = np.array([f"{world}:{stage}:{assay}:{i}" for i in range(n)])
            public_path = directory / f"public_{assay}.npz"
            save_npz(public_path, **public)
            entry["public"][assay] = {"path": str(public_path.relative_to(run_dir)), "sha256": sha(public_path)}
            assessment[f"counts_{assay}"] = draw_counts(model, assay, shared, world, "assessment")
            assessment[f"feature_ids_{assay}"] = features
        assessment_path = directory / "assessment.npz"
        save_npz(assessment_path, **assessment)
        entry["assessment"] = {"path": str(assessment_path.relative_to(run_dir)), "sha256": sha(assessment_path)}
        manifest["worlds"][world] = entry
        for item in [entry["truth"], entry["assessment"], *entry["public"].values()]:
            artifacts[item["path"]] = item["sha256"]
    write_exclusive(run_dir / "symmetry_control.json", symmetry_control())
    artifacts["symmetry_control.json"] = sha(run_dir / "symmetry_control.json")
    write_exclusive(run_dir / "data_manifest.json", manifest)
    write_exclusive(run_dir / "generation_seal.json", {
        "protocol_sha256": sha(run_dir / "protocol.json"),
        "data_manifest_sha256": sha(run_dir / "data_manifest.json"), "artifacts": artifacts,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    return manifest


def training_arrays(run_dir, entry, protocol, n_fit):
    fit, cal, scales = {}, {}, {}
    seen = set()
    for assay in protocol["assays"]:
        item = entry["public"][assay]
        path = within(run_dir, item["path"])
        require_hash(path, item["sha256"])
        with np.load(path, allow_pickle=False) as data:
            for stage, size in (("fit", max(protocol["fit_sizes"])), ("calibration", protocol["calibration_rows"])):
                x = data[f"{stage}_counts"]
                ids = data[f"{stage}_ids"].astype(str)
                if (x.shape != (size, protocol["widths"][assay]) or ids.shape != (size,)
                        or not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x))
                        or len(set(ids)) != size or seen.intersection(ids)):
                    raise ValueError("Invalid or paired training counts/identifiers")
                seen.update(ids)
            expected = [f"{assay}:feature:{j}" for j in range(protocol["widths"][assay])]
            if data["feature_ids"].astype(str).tolist() != expected:
                raise ValueError("Unexpected feature ordering")
            fit[assay] = data["fit_counts"][:n_fit].astype(float)
            cal[assay] = data["calibration_counts"].astype(float)
            scales[assay] = fit[assay].std(0).tolist()
    return fit, cal, scales


def fit_one(run_dir, protocol, world, entry, n_fit, seed):
    started = time.monotonic()
    common = {"world": world, "dimension": entry["dimension"], "n_fit": n_fit, "seed": seed}
    try:
        fit, cal, scales = training_arrays(run_dir, entry, protocol, n_fit)
        with threadpool_limits(limits=1):
            pair = fit_convergence_pair(fit, entry["dimension"], seed, protocol["chunk_iterations"],
                                        protocol["chunk_evaluations"], protocol["gtol"])
            results = {}
            for mode, key in (("capped600", "capped"), ("convergence", "uncapped")):
                model = pair[key]
                scores = {assay: float(score_rows(cal[assay], model["state_weights"], model["state_rates"][assay])[0].mean()/protocol["widths"][assay])
                          for assay in protocol["assays"]}
                results[mode] = {**common, "optimizer_mode": mode, "model": model,
                                 "fit_scale": scales, "calibration_scores": scores,
                                 "calibration_score": float(np.mean(list(scores.values()))),
                                 "acceptance": acceptance(model), "error": None}
            results["convergence"]["continuation"] = pair["continuation"]
    except Exception as exc:
        results = {mode: {**common, "optimizer_mode": mode, "model": None, "calibration_score": None,
                          "acceptance": {"pass": False}, "error": f"{type(exc).__name__}: {exc}"}
                   for mode in protocol["optimizer_modes"]}
    for result in results.values():
        result["paired_fit_elapsed_seconds"] = time.monotonic()-started
    return results


def fit_all(run_dir):
    protocol = verify_sources(run_dir)
    generation = json.loads((run_dir / "generation_seal.json").read_text())
    require_hash(run_dir / "protocol.json", generation["protocol_sha256"])
    require_hash(run_dir / "data_manifest.json", generation["data_manifest_sha256"])
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    (run_dir / "fits").mkdir(exist_ok=False)
    results = {}
    started = time.monotonic()
    tasks = list(itertools.product(protocol["worlds"], protocol["fit_sizes"], protocol["seeds"]))
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = {pool.submit(fit_one, run_dir, protocol, w, manifest["worlds"][w], n, s): (w, n, s)
                   for w, n, s in tasks}
        for future in as_completed(futures):
            world, size, seed = futures[future]
            pair = future.result()
            for mode, result in pair.items():
                name = f"fits/{world}__n{size}__{mode}__seed{seed}.json"
                write_exclusive(run_dir / name, result)
                results[name] = result
            print(json.dumps({"completed_streams": len(results)//2, "total_streams": len(tasks),
                              "world": world, "n_fit": size, "seed": seed,
                              "stop": pair["convergence"].get("model", {}).get("stop_reason") if pair["convergence"].get("model") else pair["convergence"]["error"],
                              "iterations": pair["convergence"]["model"]["iterations"] if pair["convergence"].get("model") else None}), flush=True)
    selected, restarts, statuses = {}, {}, {}
    for world, n_fit, mode in itertools.product(protocol["worlds"], protocol["fit_sizes"], protocol["optimizer_modes"]):
        key = f"{world}__n{n_fit}__{mode}"
        candidates = {name: result for name, result in results.items()
                      if (result["world"], result["n_fit"], result["optimizer_mode"]) == (world, n_fit, mode)}
        selected[key], statuses[key] = choose(candidates)
        restarts[key] = sorted(candidates)
    seal = {"protocol_sha256": sha(run_dir / "protocol.json"),
            "data_manifest_sha256": sha(run_dir / "data_manifest.json"),
            "generation_seal_sha256": sha(run_dir / "generation_seal.json"),
            "artifacts": {name: sha(run_dir / name) for name in sorted(results)},
            "selected": selected, "selected_dimension_restarts": restarts, "selection_status": statuses,
            "elapsed_seconds": time.monotonic()-started, "evaluator_read": False,
            "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_exclusive(run_dir / "fit_seal.json", seal)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "generate", "fit"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = {"freeze": freeze, "generate": generate, "fit": fit_all}[args.stage](run_dir)
    print(json.dumps({"stage": args.stage, "run_dir": str(run_dir), "entries": len(result)}))


if __name__ == "__main__":
    main()
