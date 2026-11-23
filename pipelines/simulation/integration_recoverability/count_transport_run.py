"""Freeze and fit a bounded count-transport experiment using cached counts only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from count_transport_evaluate import sha, require_hash, within, write_exclusive
from count_transport_model import fit_model, score_rows

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RUN = HERE / "runs" / "count_transport_demo_v1"
DATA_MANIFEST = "project_resources/research/count_extensions_2026_11_01/data_v1/manifest.json"
LEGACY_PATH = "project_resources/research/nonlinear_repair_2026_11_15/experiments/C01/model.py"
spec = importlib.util.spec_from_file_location("count_transport_split_helpers", REPO / LEGACY_PATH)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
WORLDS = ["linear_chain_0", "constrained_collider_0"]
WIDTHS = {"RNA": 30, "H3K27ac": 24, "H3K4me3": 18}


def freeze(run_dir: Path) -> dict:
    """Pin code and metadata before fitting; never read assessment arrays."""
    manifest = json.loads((REPO / DATA_MANIFEST).read_text())
    protocol = {
        "version": "count_transport_demo_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "worlds": WORLDS, "assays": list(WIDTHS), "widths": WIDTHS,
        "dimensions": [1, 2, 3], "seeds": [0, 1, 2, 3],
        "variants": ["strict", "affine"], "controls": ["original", "column_null"],
        "maxiter": 600, "workers": 2, "blas_threads_per_worker": 1,
        "data_manifest": DATA_MANIFEST, "data_manifest_sha256": sha(REPO / DATA_MANIFEST),
        "training_inputs": {}, "assessment_inputs": {}, "feature_ids": {}, "source_indices": {},
        "selection": "Highest equal-assay, per-feature calibration log likelihood among converged finite fits; fallback to best finite fit is diagnostic only. Exact ties: lower dimension then lower seed. No full-data refit.",
        "acceptance": {"converged": True, "finite": True, "minimum_loading_singular_ratio": .001,
                       "minimum_probability_gap": .03, "one_source_gap_exempt": True,
                       "bound_hits": "reported, not an exclusion", "poisson_calibration_gain": "diagnostic only"},
        "split": "C01 SHA256 count-dica-v1 identifiers, 75% fit / 25% calibration",
        "column_null_seed": "C01 key_seed(count-transport-v1,world,assay,column_null); permute independently inside fit and calibration",
        "scope": "Cached informative-count background panels only; no motif edges, new worlds, paired training, PCA, guidance, labels or depth adjustment",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    source_paths = [HERE / name for name in (
        "count_transport_model.py", "count_transport_run.py", "count_transport_evaluate.py",
        "tests/test_count_transport_model.py", "tests/test_count_transport_run.py",
        "tests/test_count_transport_evaluate.py")]
    source_paths += [REPO / LEGACY_PATH,
                     REPO / "project_resources/research/sturma_pilot_2026_11_01/prepare_cached_data_pilot.py",
                     run_dir / "PLAN.md"]
    for world in WORLDS:
        world_root = HERE / "runs" / "worlds" / world
        world_manifest = world_root / "manifest.json"
        source_paths.append(world_manifest)
        entries = json.loads(world_manifest.read_text())["files"]
        for key in ("training_inputs", "assessment_inputs", "feature_ids", "source_indices"):
            protocol[key][world] = {}
        for assay, width in WIDTHS.items():
            selected = manifest["datasets"][world]["assays"][assay]
            entry = selected["cohorts"]["train"]
            require_hash(REPO / entry["path"], entry["sha256"])
            protocol["training_inputs"][world][assay] = {k: entry[k] for k in ("path", "sha256")}
            protocol["feature_ids"][world][assay] = selected["selected_feature_ids"][:width]
            protocol["source_indices"][world][assay] = selected["selected_source_column_indices"][:width]
            assessment = f"private_inputs/informative/assessment/{assay}.h5ad"
            protocol["assessment_inputs"][world][assay] = {
                "path": str((world_root / assessment).relative_to(REPO)), "sha256": entries[assessment]}
    protocol["sources"] = {str(path.relative_to(REPO)): sha(path) for path in source_paths}
    write_exclusive(run_dir / "protocol.json", protocol)
    return protocol


def load_training(protocol: dict, world: str, control: str) -> tuple[dict, dict, dict]:
    manifest = json.loads(within(REPO, protocol["data_manifest"]).read_text())
    fit, calibration, scales = {}, {}, {}
    used_ids = set()
    for assay in protocol["assays"]:
        entry = protocol["training_inputs"][world][assay]
        path = within(REPO, entry["path"])
        require_hash(path, entry["sha256"])
        with np.load(path, allow_pickle=False) as data:
            ids = data["cell_ids"].astype(str)
            expected = manifest["datasets"][world]["assays"][assay]["selected_feature_ids"]
            x = legacy.validate(data["raw_counts"], ids, data["feature_ids"].astype(str), expected)
        if used_ids.intersection(ids):
            raise ValueError("Training assays share cell identifiers")
        used_ids.update(ids)
        x = x[:, :protocol["widths"][assay]]
        fit_rows, cal_rows = legacy.split(ids)
        scales[assay] = x[fit_rows].std(axis=0).tolist()
        if control == "column_null":
            x = legacy.permute(x, [fit_rows, cal_rows],
                               legacy.key_seed("count-transport-v1", world, assay, control))
        elif control != "original":
            raise ValueError("Unknown control")
        fit[assay], calibration[assay] = x[fit_rows], x[cal_rows]
    return fit, calibration, scales


def acceptance(model: dict) -> dict:
    checks = {"converged": bool(model["converged"]),
              "finite": bool(np.isfinite(model["objective"]) and all(
                  np.isfinite(r).all() and (np.asarray(r) > 0).all() for r in model["state_rates"].values())),
              "loading_rank": all(r >= .001 for r in model["singular_value_ratios"].values()),
              "source_separation": model["dimension"] == 1 or model["minimum_probability_gap"] >= .03}
    return {"pass": bool(all(checks.values())), "checks": checks,
            "interpretation": "Numerical and model-class checks only; not evidence the true data law satisfies E2"}


def fit_one(protocol: dict, world: str, control: str, variant: str, dimension: int, seed: int) -> dict:
    start = time.monotonic()
    result = {"world": world, "control": control, "variant": variant,
              "dimension": dimension, "seed": seed}
    try:
        fit, calibration, scales = load_training(protocol, world, control)
        with threadpool_limits(limits=1):
            model = fit_model(fit, dimension, variant, seed, maxiter=protocol["maxiter"])
            scores, baseline = {}, {}
            for assay, x in calibration.items():
                scores[assay] = float(score_rows(x, model["state_weights"], model["state_rates"][assay])[0].mean() / x.shape[1])
                # shortcut: a small floor defines the diagnostic independent-Poisson baseline for unseen positive counts.
                rates = np.maximum(fit[assay].mean(0, keepdims=True), 1e-8)
                baseline[assay] = float(score_rows(x, [1.], rates)[0].mean() / x.shape[1])
        result.update(model=model, fit_scale=scales, calibration_score=float(np.mean(list(scores.values()))),
                      calibration_scores=scores, independent_poisson_calibration_scores=baseline,
                      fit_rows={a: len(x) for a, x in fit.items()},
                      calibration_rows={a: len(x) for a, x in calibration.items()},
                      acceptance=acceptance(model), error=None)
    except Exception as exc:
        result.update(model=None, calibration_score=None, acceptance={"pass": False},
                      error=f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def choose(candidates: dict[str, dict]) -> tuple[str, str]:
    finite = {name: r for name, r in candidates.items() if r.get("model") is not None
              and r.get("calibration_score") is not None and np.isfinite(r["calibration_score"])}
    converged = {name: r for name, r in finite.items() if r["model"]["converged"]}
    pool = converged or finite
    if not pool:
        raise ValueError("No finite fitted candidate; preserve failures and stop before evaluation")
    selected = min(pool, key=lambda name: (-pool[name]["calibration_score"], pool[name]["dimension"], pool[name]["seed"]))
    return selected, "converged_calibration_selection" if converged else "finite_nonconverged_diagnostic"


def run_fits(run_dir: Path) -> dict:
    if (run_dir / "fit_seal.json").exists() or (run_dir / "evaluation.json").exists():
        raise FileExistsError("Sealed experiments cannot be fitted again")
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    require_hash(REPO / protocol["data_manifest"], protocol["data_manifest_sha256"])
    for path, expected in protocol["sources"].items():
        require_hash(within(REPO, path), expected)
    fit_dir = run_dir / "fits"
    fit_dir.mkdir(exist_ok=False)
    tasks = list(itertools.product(protocol["worlds"], protocol["controls"], protocol["variants"],
                                   protocol["dimensions"], protocol["seeds"]))
    results = {}
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = {pool.submit(fit_one, protocol, *task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            name = "fits/" + "__".join(map(str, futures[future])) + ".json"
            write_exclusive(run_dir / name, result)
            results[name] = result
            print(json.dumps({"completed": len(results), "total": len(tasks), "fit": name,
                              "calibration": result["calibration_score"], "accepted": result["acceptance"]["pass"],
                              "error": result["error"]}), flush=True)
    selected, restarts, status = {}, {}, {}
    for world, control, variant in itertools.product(protocol["worlds"], protocol["controls"], protocol["variants"]):
        key = "__".join((world, control, variant))
        candidates = {name: r for name, r in results.items()
                      if (r["world"], r["control"], r["variant"]) == (world, control, variant)}
        selected[key], status[key] = choose(candidates)
        dimension = results[selected[key]]["dimension"]
        restarts[key] = sorted(name for name, r in candidates.items() if r["dimension"] == dimension)
    seal = {"protocol_sha256": sha(protocol_path),
            "artifacts": {name: sha(run_dir / name) for name in sorted(results)},
            "selected": selected, "selected_dimension_restarts": restarts,
            "selection_status": status, "evaluator_read": False,
            "elapsed_seconds": time.monotonic() - start,
            "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_exclusive(run_dir / "fit_seal.json", seal)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.stage == "freeze":
        freeze(run_dir)
        print(f"Frozen protocol: {run_dir / 'protocol.json'}")
    else:
        seal = run_fits(run_dir)
        print(json.dumps({"sealed_fits": len(seal["artifacts"]), "selected_groups": len(seal["selected"])}))


if __name__ == "__main__":
    main()
