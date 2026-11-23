"""Independently check saved Ridge state probes with NumPy, without evaluator helpers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "pipelines/simulation/integration_recoverability"
REPORTS = BASE / "runs/integration_validation/20261101_scconfluence_state_v1/reports"
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(record):
    if digest(record["path"]) != record["sha256"]:
        raise ValueError(f"Changed source: {record['path']}")
    return Path(record["path"])


def rows(ids, selected):
    ids = np.asarray(ids, dtype=str)
    if len(set(ids)) != len(ids) or not set(selected) <= set(ids):
        raise ValueError("Duplicate or missing IDs")
    lookup = {v: i for i, v in enumerate(ids)}
    return np.asarray([lookup[v] for v in selected], dtype=int)


def review(world, seed):
    path = REPORTS / f"{world}_seed{seed}.json"
    report = read(path)
    spec = read(checked(report["state_protocol"]))
    if report["pools"] != spec["pools"][world] or report["seed"] != seed:
        raise ValueError("Report identity/pool changed")
    states, inputs = {}, {}
    for cohort in ("train", "val"):
        truth = BASE / "runs/worlds" / world / "truth" / f"{cohort}.npz"
        record = {"path": str(truth), "sha256": report["truth_hashes"][cohort]}
        checked(record)
        inputs[f"truth/{cohort}"] = record
        with np.load(truth, allow_pickle=False) as data:
            ids, values, assays = data["cell_ids"].astype(str), data["state"].astype(float), data["assigned_assay"].astype(str)
        states[cohort] = {}
        for assay in ASSAYS:
            idx = rows(ids, report["pools"][cohort][assay])
            if not np.all(assays[idx] == assay):
                raise ValueError("State pool has the wrong assay")
            states[cohort][assay] = values[idx]
    recomputed, max_error = {}, 0.0
    for label, result in report["results"].items():
        matrices = {}
        for cohort in ("train", "val"):
            proof = report["evidence"][label][cohort]
            if "encoding" in proof:
                encoding_path = checked(proof["encoding"])
                inputs[f"{label}/{cohort}/encoding"] = proof["encoding"]
                exports = read(encoding_path)["exports"]
            else:
                exports = proof["files"]
            matrices[cohort] = {}
            for assay in ASSAYS:
                entry = exports[assay]
                file = checked(entry)
                inputs[f"{label}/{cohort}/{assay}"] = {"path": str(file), "sha256": entry["sha256"]}
                with np.load(file, allow_pickle=False) as data:
                    ids, x = data["cell_ids"].astype(str), data["shared"]
                    # Native GLUE loader preserves its stored dtype; the new sealed loader casts to float64.
                    if label.startswith("scconfluence_"):
                        x = x.astype(np.float64)
                matrices[cohort][assay] = x[rows(ids, report["pools"][cohort][assay])]
        recomputed[label] = {}
        for source in ASSAYS:
            x, y = matrices["train"][source], states["train"][source]
            # StandardScaler estimates moments in float64, then transforms in the input dtype.
            mean, variance = x.astype(np.float64).mean(axis=0), x.astype(np.float64).var(axis=0, ddof=0)
            eps = np.finfo(np.float64).eps
            constant = variance <= len(x) * eps * variance + (len(x) * mean * eps) ** 2
            scale = np.sqrt(variance)
            scale[constant] = 1.0
            z = x.copy()
            z -= mean
            z /= scale
            y = y.astype(z.dtype)
            z_center, y_center = z.mean(axis=0), y.mean(axis=0)
            centered = z - z_center
            # alpha=1, fitted intercept; only the source training assay determines scaling.
            coefficients = np.linalg.solve(centered.T @ centered + np.eye(x.shape[1], dtype=z.dtype),
                                           centered.T @ (y - y_center))
            intercept = y_center - z_center @ coefficients
            for target in ASSAYS:
                expected = result["source_state_transfer"][f"{source}->{target}"]
                target_x = matrices["val"][target].copy()
                target_x -= mean
                target_x /= scale
                prediction = target_x @ coefficients + intercept
                observed = states["val"][target]
                total = np.square(observed - observed.mean(axis=0)).sum(axis=0)
                residual = np.square(observed - prediction).sum(axis=0)
                values = [float(1 - e / v) if v > 0 else None for e, v in zip(residual, total)]
                finite = [v for v in values if v is not None]
                mean_r2 = float(np.mean(finite)) if finite else None
                if ([v is None for v in values] != [v is None for v in expected["r2_by_state"]]
                        or expected["target_refit"] is not False):
                    raise ValueError("Undefined-coordinate or target-refit contract changed")
                errors = [abs(v - original) for v, original in zip(values, expected["r2_by_state"]) if v is not None]
                if mean_r2 is not None:
                    errors.append(abs(mean_r2 - expected["mean_r2"]))
                error = max(errors, default=0.0)
                tolerance = 1e-4 if z.dtype == np.float32 else 1e-7
                if error > tolerance:
                    raise ValueError(f"Independent Ridge result disagrees: {world}/{label}/{source}->{target}: {error}")
                max_error = max(max_error, error)
                recomputed[label][f"{source}->{target}"] = {
                    "r2_by_state": values, "mean_r2": mean_r2, "max_abs_error": error,
                    "fitting_dtype": str(z.dtype), "absolute_tolerance": tolerance,
                    "constant_training_coordinates": int(constant.sum())}
    receipt = {"status": "passed", "world": world, "seed": seed,
               "report": {"path": str(path), "sha256": digest(path)},
               "protocol": report["state_protocol"], "source": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
               "numpy_version": np.__version__, "inputs": inputs,
               "procedure": "NumPy closed-form Ridge alpha=1 with fitted intercept; source training mean and population variance; target-val variance R2 denominator",
               "tolerance": {"float32": 1e-4, "float64": 1e-7}, "maximum_absolute_error": max_error,
               "precision_note": "Existing GLUE arrays/fit are float32; sealed scConfluence loader casts to float64. Independent NumPy solve differs from sklearn's SciPy Cholesky rounding, so tolerances record each fit dtype; no saved results or evaluators changed.",
               "directions_checked": sum(len(x) for x in recomputed.values()), "recomputed": recomputed,
               "scope": "Independent arithmetic and input-identity check; no sklearn or benchmark evaluator helpers, fitting changes or causal claim"}
    destination = HERE / "reviews" / f"{world}_seed{seed}_state_transfer_audit.json"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"receipt": str(destination), "directions_checked": receipt["directions_checked"],
            "maximum_absolute_error": max_error, "status": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=("linear_chain_0", "constrained_collider_0"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.world, args.seed), sort_keys=True))
