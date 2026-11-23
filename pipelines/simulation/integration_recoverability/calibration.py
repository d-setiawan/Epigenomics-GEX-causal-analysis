"""Empirical count profiles and frozen observation-model calibration.

This module never looks at recovery results.  Source libraries are used as a
*working exposure distribution*, not as identified technical size factors.
Per-feature summaries are coherent: a focal profile always comes from one
deterministically selected real feature, never a mixture of convenient moments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
# Declared before any graph result.  These are engineering realism checks, not
# hypothesis tests or confidence intervals.
DEFAULT_TOLERANCES = {
    "detection_absolute": 0.002,
    "detection_relative": 0.25,
    "mean_relative": 0.10,
    "variance_relative": 0.40,
    "positive_mean_relative": 0.15,
    "library_quantile_relative": 0.35,
    "detected_features_quantile_relative": 0.35,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(float).tolist()


def summarize_count_matrix(matrix: Any, feature_ids: list[str] | np.ndarray,
                           chunk_size: int = 512) -> dict[str, Any]:
    """Read rows in chunks; never densify the full source count matrix."""
    n, p = matrix.shape
    if n < 2 or p < 1 or len(feature_ids) != p:
        raise ValueError("A count profile needs >=2 cells and matching feature IDs")
    sums = np.zeros(p, dtype=np.float64)
    squares = np.zeros(p, dtype=np.float64)
    detected = np.zeros(p, dtype=np.int64)
    ones = np.zeros(p, dtype=np.int64)
    libraries = np.zeros(n, dtype=np.float64)
    detected_per_cell = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        block = sparse.csr_matrix(matrix[start:start + chunk_size]).astype(np.float64)
        block.sum_duplicates()
        block.eliminate_zeros()
        if not np.isfinite(block.data).all() or (block.data < 0).any():
            raise ValueError("Calibration source contains negative/non-finite counts")
        if not np.equal(block.data, np.floor(block.data)).all():
            raise ValueError("Calibration requires integer counts, not normalized X")
        sums += np.asarray(block.sum(axis=0)).ravel()
        squares += np.asarray(block.power(2).sum(axis=0)).ravel()
        detected += np.bincount(block.indices, minlength=p)
        ones += np.bincount(block.indices[block.data == 1], minlength=p)
        libraries[start:start + block.shape[0]] = np.asarray(block.sum(axis=1)).ravel()
        detected_per_cell[start:start + block.shape[0]] = np.diff(block.indptr)
    mean = sums / n
    variance = np.maximum((squares - sums * sums / n) / (n - 1), 0)
    profiles = []
    for j, feature_id in enumerate(feature_ids):
        d = int(detected[j])
        profiles.append({
            "feature_id": str(feature_id), "n_detected": d,
            "detection": d / n, "mean": float(mean[j]),
            "variance": float(variance[j]),
            "positive_mean": float(sums[j] / d) if d else 0.0,
            "positive_one_fraction": float(ones[j] / d) if d else 0.0,
        })
    return {"n_cells": n, "n_features": p, "features": profiles,
            "library_totals": libraries.tolist(),
            "library_quantiles": _quantiles(libraries),
            "detected_features_quantiles": _quantiles(detected_per_cell),
            "zero_library_cells": int((libraries == 0).sum())}


def select_focal_profile(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Lower median by detection, then ID, among detected retained features."""
    ordered = sorted((x for x in profiles if x["n_detected"] > 0),
                     key=lambda x: (x["detection"], x["feature_id"]))
    if not ordered:
        raise ValueError("No detected features in retained source panel")
    return dict(ordered[(len(ordered) - 1) // 2])


def panel_exposures(assay_profile: dict[str, Any], panel: str) -> np.ndarray:
    """Use depth from the same encoder/retained universe being simulated."""
    totals = np.asarray(assay_profile[panel]["library_totals"], dtype=float)
    positive = totals[totals > 0]
    if not len(positive):
        raise ValueError("No positive library totals in selected empirical panel")
    return positive / positive.mean()


def default_source_paths(repo_root: str | Path) -> dict[str, dict[str, str]]:
    base = Path(repo_root) / "pipelines/empirical/integration/runs/scglue_joint_v1"
    return {assay: {
        "input": str(base / "preprocess" / (
            "rna_preprocessed.h5ad" if assay == "RNA" else f"chrom_{assay}_preprocessed.h5ad")),
        "retained": str(base / "train/modalities" / (
            "rna_with_glue.h5ad" if assay == "RNA" else f"chrom_{assay}_with_glue.h5ad")),
    } for assay in ASSAYS}


def calibrate_assay_profiles(source_paths: dict[str, dict[str, str]],
                             out_path: str | Path | None = None,
                             tolerances: dict[str, float] | None = None) -> dict[str, Any]:
    """Extract source summaries once, one assay at a time, with exact masks.

    AnnData may materialize layers even in backed mode; only one assay is open.
    No source file is modified.  Existing calibration artifacts are not replaced.
    """
    import anndata as ad

    if out_path is not None and Path(out_path).exists():
        raise FileExistsError(f"Refusing to replace frozen calibration: {out_path}")
    output: dict[str, Any] = {
        "schema_version": 1,
        "tolerances": dict(DEFAULT_TOLERANCES if tolerances is None else tolerances),
        "exposure_assumption": "Empirical positive library totals / their mean; all variation treated as working exposure, independently resampled across assays and cells. Not an identified technical decomposition.",
        "focal_selection": "lower median nonzero detection, tie by feature ID, in retained panel",
        "assays": {},
    }
    for assay in ASSAYS:
        paths = source_paths[assay]
        source = ad.read_h5ad(paths["input"], backed="r")
        retained = ad.read_h5ad(paths["retained"], backed="r")
        try:
            names = source.var_names.astype(str).tolist()
            retained_ids = retained.var_names.astype(str).tolist()
            retained_set = set(retained_ids)
            if not retained_set.issubset(set(names)):
                raise ValueError(f"{assay}: retained IDs are not a subset of source IDs")
            matrix = source.layers["counts"] if "counts" in source.layers else source.X
            stats = summarize_count_matrix(matrix, names)
            keep = np.array([x in retained_set for x in names])
            retained_stats = summarize_count_matrix(matrix[:, keep], np.asarray(names)[keep])
            positive = np.array(stats["library_totals"], dtype=float)
            positive = positive[positive > 0]
            if not len(positive):
                raise ValueError(f"{assay}: all source libraries are empty")
            output["assays"][assay] = {
                "source_path": str(Path(paths["input"]).resolve()),
                "source_sha256": file_sha256(paths["input"]),
                "retained_source_path": str(Path(paths["retained"]).resolve()),
                "retained_source_sha256": file_sha256(paths["retained"]),
                "count_layer": "counts" if "counts" in source.layers else "X",
                "encoder": stats, "retained": retained_stats,
                "retained_feature_ids": retained_ids,
                "focal_profile": select_focal_profile(retained_stats["features"]),
                "relative_exposures": (positive / positive.mean()).tolist(),
            }
        finally:
            source.file.close()
            retained.file.close()
    output["calibration_hash"] = canonical_hash(output)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as handle:
            json.dump(output, handle, sort_keys=True, indent=2, allow_nan=False)
    return output


def expected_count_moments(activity: np.ndarray, exposure: np.ndarray,
                           alpha: float, kappa: float, theta: float | None
                           ) -> dict[str, float]:
    mu = exposure * np.logaddexp(0.0, alpha + kappa * activity)
    conditional_variance = mu if theta is None else mu + mu * mu / theta
    pzero = np.exp(-mu) if theta is None else np.exp(-theta * np.log1p(mu / theta))
    mean = float(mu.mean())
    detection = float(np.mean(1 - pzero))
    return {"mean": mean, "variance": float(conditional_variance.mean() + mu.var()),
            "detection": detection,
            "positive_mean": mean / detection if detection > 0 else 0.0}


def fit_count_parameters(activity: np.ndarray, exposure: np.ndarray,
                         target: dict[str, float], distribution: str = "poisson",
                         kappa: float = 0.8) -> dict[str, Any]:
    """Freeze a mean-matched link and nonnegative NB residual overdispersion.

    Kappa is an a-priori measurement sensitivity, not recovery-tuned.  Alpha is
    fitted by bisection on an independent calibration draw.  NB dispersion uses
    the marginal variance after subtracting count-mean variation.  A Poisson
    lower-variance bound is retained and discrepancies are reported, not fixed
    by adding zero inflation or changing biology.
    """
    activity, exposure = np.asarray(activity, float), np.asarray(exposure, float)
    if activity.shape != exposure.shape or not np.isfinite(activity).all():
        raise ValueError("Calibration activity and exposure must be finite equal vectors")
    if (exposure <= 0).any() or not np.isfinite(exposure).all():
        raise ValueError("Exposures must be positive and finite")
    if distribution not in {"poisson", "nb"}:
        raise ValueError(f"Unknown distribution {distribution}")
    if target["mean"] <= 0:
        return {"alpha": -80.0, "kappa": float(kappa), "theta": None,
                "distribution": distribution, "structural_zero": True,
                "expected": {key: 0.0 for key in ("mean", "variance", "detection", "positive_mean")}}
    low = -80.0
    high = max(80.0, 2 * target["mean"] / float(exposure.mean()) + kappa * float(np.max(np.abs(activity))))
    for _ in range(48):
        middle = (low + high) / 2
        mean = np.mean(exposure * np.logaddexp(0, middle + kappa * activity))
        if mean < target["mean"]:
            low = middle
        else:
            high = middle
    alpha = (low + high) / 2
    mu = exposure * np.logaddexp(0, alpha + kappa * activity)
    theta = None
    if distribution == "nb":
        residual = target["variance"] - float(mu.mean() + mu.var())
        theta = float(max(np.mean(mu * mu) / residual, 1e-4)) if residual > 0 else None
    return {"alpha": float(alpha), "kappa": float(kappa), "theta": theta,
            "distribution": distribution, "structural_zero": False,
            "expected": expected_count_moments(activity, exposure, alpha, kappa, theta)}


def assess_profile_fit(target: dict[str, float], expected: dict[str, float],
                       tolerances: dict[str, float] | None = None) -> dict[str, Any]:
    tolerances = DEFAULT_TOLERANCES if tolerances is None else tolerances
    checks = {}
    for key in ("mean", "variance", "detection", "positive_mean"):
        allowed = tolerances[key + "_relative"] * max(abs(target[key]), 1e-12)
        if key == "detection":
            allowed = max(allowed, tolerances["detection_absolute"])
        checks[key] = {"target": target[key], "expected": expected[key],
                       "absolute_error": abs(expected[key] - target[key]),
                       "allowed_error": allowed,
                       "passed": bool(abs(expected[key] - target[key]) <= allowed)}
    return {"passed": all(x["passed"] for x in checks.values()), "checks": checks}


def sample_counts(rng: np.random.Generator, activity: np.ndarray,
                  exposure: np.ndarray, parameters: dict[str, Any],
                  multiplier: float = 1.0) -> np.ndarray:
    if parameters.get("structural_zero", False):
        return np.zeros(np.asarray(activity).shape, dtype=np.int32)
    mu = multiplier * exposure * np.logaddexp(
        0, parameters["alpha"] + parameters["kappa"] * activity)
    theta = parameters["theta"]
    if not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("Invalid count means")
    if theta is None:
        sampled = rng.poisson(mu)
    else:
        sampled = rng.negative_binomial(theta, theta / (theta + mu))
    if sampled.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("Counts exceed int32; do not silently truncate")
    return sampled.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_assay_profiles(default_source_paths(args.repo_root), args.out)
    print(json.dumps({"path": str(args.out), "hash": result["calibration_hash"],
                      "focal_profiles": {a: x["focal_profile"] for a, x in result["assays"].items()}}, indent=2))


if __name__ == "__main__":
    main()
