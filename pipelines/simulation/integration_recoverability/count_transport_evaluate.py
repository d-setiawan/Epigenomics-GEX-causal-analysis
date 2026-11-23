"""Evaluate sealed count-transport fits on existing, paired assessment counts.

Assessment pairing is used only here, after source and fit hashes are checked.
No fit, source alignment, feature selection, or model selection happens here.
The fitted joint count law is the target; these background panels do not contain
the planted causal motif and cannot demonstrate recovery of its DAG.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RUN = HERE / "runs" / "count_transport_demo_v1"
BOOTSTRAPS = 1000
FAMILY_COMPARISONS = 12
BASE_SEED = 71407
EXTRACTION_HELPER = REPO / "project_resources/research/sturma_pilot_2026_11_01/prepare_cached_data_pilot.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    return BASE_SEED + int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def within(root: Path, relative: str) -> Path:
    """Reject absolute/escaping paths before reading a sealed input."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes sealed root: {relative}")
    return path


def require_hash(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Hash changed: {path}")


def write_exclusive(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def count_array(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    if (x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all()
            or (x < 0).any() or (x != np.floor(x)).any()):
        raise ValueError("Expected finite nonnegative integer count matrix")
    return x


def validate_model(model: dict, assays: list[str], widths: dict) -> tuple[np.ndarray, dict]:
    weights = np.asarray(model["state_weights"], dtype=float)
    if (weights.ndim != 1 or len(weights) == 0 or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isclose(weights.sum(), 1, atol=1e-9, rtol=1e-9)):
        raise ValueError("Invalid finite-state probabilities")
    rates = {a: np.asarray(model["state_rates"][a], dtype=float) for a in assays}
    for assay, r in rates.items():
        if (r.shape != (len(weights), widths[assay]) or not np.isfinite(r).all()
                or (r < 0).any()):
            raise ValueError(f"Invalid state rates for {assay}")
    return weights, rates


def poisson_components(x: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Product-Poisson log probabilities, including valid zero-rate atoms."""
    x = count_array(x)
    rates = np.asarray(rates, dtype=float)
    if (rates.ndim != 2 or rates.shape[1] != x.shape[1]
            or not np.isfinite(rates).all() or (rates < 0).any()):
        raise ValueError("Invalid Poisson rates")
    # Component loops avoid an n x K x p allocation and handle 0*log(0).
    constant = gammaln(x + 1).sum(1)
    return np.column_stack([
        xlogy(x, r).sum(1) - r.sum() - constant for r in rates])


def prediction_rows(source: np.ndarray, target: np.ndarray, weights: np.ndarray,
                    source_rates: np.ndarray, target_rates: np.ndarray,
                    permutation: np.ndarray) -> dict[str, np.ndarray]:
    """Paired and source-shuffled scores with the identical target marginal."""
    n = len(source)
    permutation = np.asarray(permutation)
    if (len(target) != n or permutation.shape != (n,)
            or not np.issubdtype(permutation.dtype, np.integer)
            or not np.array_equal(np.sort(permutation), np.arange(n))):
        raise ValueError("Invalid source-only row permutation")
    source_component = poisson_components(source, source_rates)
    target_component = poisson_components(target, target_rates)
    source_joint = source_component + np.log(weights)[None, :]
    source_marginal = logsumexp(source_joint, axis=1)
    if not np.isfinite(source_marginal).all():
        raise ValueError("Model assigns zero probability to a source observation")
    posterior_log = source_joint - source_marginal[:, None]
    target_marginal = logsumexp(target_component + np.log(weights)[None, :], axis=1)
    conditional = logsumexp(posterior_log + target_component, axis=1)
    shuffled = logsumexp(posterior_log[permutation] + target_component, axis=1)
    if not all(np.isfinite(v).all() for v in (target_marginal, conditional, shuffled)):
        raise ValueError("Nonfinite heldout predictive score")
    return {
        "source_marginal_loglik": source_marginal,
        "target_marginal_loglik": target_marginal,
        "conditional_loglik": conditional,
        "shuffled_conditional_loglik": shuffled,
        "conditional_mean": np.exp(posterior_log) @ target_rates,
        "shuffled_conditional_mean": np.exp(posterior_log[permutation]) @ target_rates,
        "target_mean": weights @ target_rates,
    }


def cross_moments(weights: np.ndarray, left_rates: np.ndarray,
                  right_rates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = left_rates - weights @ left_rates
    right = right_rates - weights @ right_rates
    covariance = left.T @ (weights[:, None] * right)
    # Observation variance is needed for COUNT correlation. Cross-assay
    # observation noises are conditionally independent and add no covariance.
    left_variance = weights @ (left_rates + left ** 2)
    right_variance = weights @ (right_rates + right ** 2)
    return covariance, left_variance, right_variance


def covariance_metrics(source: np.ndarray, target: np.ndarray, weights: np.ndarray,
                       source_rates: np.ndarray, target_rates: np.ndarray,
                       source_scale: np.ndarray, target_scale: np.ndarray) -> dict:
    observed = (source - source.mean(0)).T @ (target - target.mean(0)) / len(source)
    predicted, left_variance, right_variance = cross_moments(weights, source_rates, target_rates)
    denominator = np.outer(source_scale, target_scale)
    valid = denominator > 0
    def rmse(value):
        return float(np.sqrt(np.mean((value[valid] / denominator[valid]) ** 2))) if valid.any() else None
    observed_denominator = np.sqrt(np.outer(source.var(0), target.var(0)))
    predicted_denominator = np.sqrt(np.outer(left_variance, right_variance))
    correlate = valid & (observed_denominator > 0) & (predicted_denominator > 0)
    cosine = None
    if correlate.any():
        a = observed[correlate] / observed_denominator[correlate]
        b = predicted[correlate] / predicted_denominator[correlate]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm > 0:
            cosine = float(np.clip(np.dot(a, b) / norm, -1, 1))
    return {"standardized_crosscovariance_rmse": rmse(predicted - observed),
            "zero_crosscovariance_rmse": rmse(observed),
            "count_crosscorrelation_cosine": cosine,
            "valid_covariance_pairs": int(valid.sum()),
            "unscorable_covariance_pairs": int((~valid).sum()),
            "valid_correlation_pairs": int(correlate.sum())}


def bootstrap_indices(n: int, world: str, repetitions: int = BOOTSTRAPS) -> np.ndarray:
    if n < 2 or repetitions < 2:
        raise ValueError("Bootstrap requires at least two cells and repetitions")
    return np.random.default_rng(stable_seed(world, "paired_rows")).integers(
        0, n, size=(repetitions, n), dtype=np.int32)


def interval(values: np.ndarray, indices: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid per-cell statistic")
    estimates = values[indices].mean(1)
    family_alpha = .05 / FAMILY_COMPARISONS
    return {"mean": float(values.mean()),
            "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "ci_family_adjusted": np.quantile(
                estimates, [family_alpha / 2, 1 - family_alpha / 2]).tolist()}


def per_feature(summary: dict, width: int) -> dict:
    return {"mean": summary["mean"] / width,
            "ci95": [value / width for value in summary["ci95"]],
            "ci_family_adjusted": [value / width for value in summary["ci_family_adjusted"]]}


def evaluate_fit(result: dict, counts: dict[str, np.ndarray], assays: list[str],
                 widths: dict, indices: np.ndarray, group_key: str) -> dict:
    output = {k: result[k] for k in ("world", "control", "variant", "dimension", "seed")}
    if result.get("model") is None:
        output.update({"evaluation_status": "fit_failed", "fit_accepted": False,
                       "fit_acceptance": result.get("acceptance", {"pass": False}),
                       "calibration_score": result.get("calibration_score"),
                       "error": result.get("error", "No finite fitted model"),
                       "pairs": {}, "marginal_loglik_per_cell": {},
                       "all_directions_positive_family_adjusted_gain": False})
        return output
    weights, rates = validate_model(result["model"], assays, widths)
    scales = {a: np.asarray(result["fit_scale"][a], dtype=float) for a in assays}
    for a, scale in scales.items():
        if scale.shape != (widths[a],) or not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError(f"Invalid fit-only scale for {a}")
        if counts[a].shape != (len(indices[0]), widths[a]):
            raise ValueError(f"Assessment shape differs from frozen panel for {a}")
    accepted = result["acceptance"]["pass"] is True
    output.update({"evaluation_status": "scored", "fit_accepted": accepted, "fit_acceptance": result["acceptance"],
                   "calibration_score": result["calibration_score"],
                   "pairs": {}, "marginal_loglik_per_cell": {}})
    for source, target in itertools.permutations(assays, 2):
        name = f"{source}__to__{target}"
        permutation = np.random.default_rng(stable_seed(group_key, name)).permutation(len(counts[source]))
        rows = prediction_rows(counts[source], counts[target], weights, rates[source], rates[target], permutation)
        gain = rows["conditional_loglik"] - rows["target_marginal_loglik"]
        shuffled_gain = rows["shuffled_conditional_loglik"] - rows["target_marginal_loglik"]
        squared_conditional = (counts[target] - rows["conditional_mean"]) ** 2
        squared_marginal = (counts[target] - rows["target_mean"]) ** 2
        conditional_mse = squared_conditional.mean(1)
        marginal_mse = squared_marginal.mean(1)
        valid = scales[target] > 0
        gain_summary = interval(gain, indices)
        shuffle_summary = interval(shuffled_gain, indices)
        gain_over_shuffle = interval(gain - shuffled_gain, indices)
        record = {
            "target_features": widths[target],
            "conditional_loglik_per_cell": float(rows["conditional_loglik"].mean()),
            "target_marginal_loglik_per_cell": float(rows["target_marginal_loglik"].mean()),
            "conditional_loglik_gain": gain_summary,
            "conditional_loglik_gain_per_target_feature": per_feature(gain_summary, widths[target]),
            "source_shuffle_loglik_gain": shuffle_summary,
            "source_shuffle_loglik_gain_per_target_feature": per_feature(shuffle_summary, widths[target]),
            "loglik_gain_over_source_shuffle": gain_over_shuffle,
            "loglik_gain_over_source_shuffle_per_target_feature": per_feature(gain_over_shuffle, widths[target]),
            "conditional_mean_mse": float(conditional_mse.mean()),
            "fitted_target_mean_mse": float(marginal_mse.mean()),
            "mean_mse_reduction": interval(marginal_mse - conditional_mse, indices),
            "standardized_mean_mse_reduction": None if not valid.any() else interval(
                ((squared_marginal[:, valid] - squared_conditional[:, valid]) /
                 scales[target][valid] ** 2).mean(1), indices),
            "source_shuffle_seed": stable_seed(group_key, name),
            "source_shuffle_permutation_sha256": hashlib.sha256(
                permutation.astype("<i8").tobytes()).hexdigest(),
            "accepted_fit_positive_family_adjusted_gain": bool(
                accepted and gain_summary["ci_family_adjusted"][0] > 0),
        }
        record.update(covariance_metrics(counts[source], counts[target], weights,
                                        rates[source], rates[target], scales[source], scales[target]))
        output["pairs"][name] = record
        output["marginal_loglik_per_cell"][target] = record["target_marginal_loglik_per_cell"]
    output["all_directions_positive_family_adjusted_gain"] = bool(
        accepted and all(p["accepted_fit_positive_family_adjusted_gain"] for p in output["pairs"].values()))
    return output


def verify_frozen(run_dir: Path, repo_root: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Complete all protocol/source/artifact checks before loading assessment."""
    protocol_path, seal_path = run_dir / "protocol.json", run_dir / "fit_seal.json"
    protocol = json.loads(protocol_path.read_text())
    seal = json.loads(seal_path.read_text())
    require_hash(protocol_path, seal["protocol_sha256"])
    if seal.get("evaluator_read", False) is not False:
        raise ValueError("Fit seal records prior evaluation access")
    for relative, expected in protocol["sources"].items():
        require_hash(within(repo_root, relative), expected)
    manifest_path = within(repo_root, protocol["data_manifest"])
    require_hash(manifest_path, protocol["data_manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    for name, expected in seal["artifacts"].items():
        path = within(run_dir, name)
        require_hash(path, expected)
        artifacts[name] = json.loads(path.read_text())
    selected, restarts = seal["selected"], seal["selected_dimension_restarts"]
    if not selected or set(selected) != set(restarts):
        raise ValueError("Selection and restart groups differ")
    for key, filename in selected.items():
        if filename not in artifacts or filename not in restarts[key]:
            raise ValueError("Selected fit is absent from sealed restart group")
        if len(restarts[key]) != 4 or len(set(restarts[key])) != 4:
            raise ValueError("Expected four sealed starts at the selected dimension")
        chosen = artifacts[filename]
        if chosen.get("model") is None:
            raise ValueError("Selected artifact has no finite fitted model")
        for name in restarts[key]:
            if name not in artifacts:
                raise ValueError("Restart artifact is not sealed")
            artifact = artifacts[name]
            actual_key = "__".join(artifact[k] for k in ("world", "control", "variant"))
            if actual_key != key or artifact["dimension"] != chosen["dimension"]:
                raise ValueError("Restart group identity or dimension mismatch")
            if artifact["world"] not in protocol["worlds"]:
                raise ValueError("Unknown world in fit artifact")
            if artifact.get("model") is not None:
                validate_model(artifact["model"], protocol["assays"], protocol["widths"])
    # Pin input bytes before H5 loading. Train IDs are used solely to verify
    # that assessment cells are disjoint from every allowed training assay.
    input_hashes, train_ids = {}, {}
    for world in protocol["worlds"]:
        train_ids[world] = set()
        for assay in protocol["assays"]:
            spec = manifest["datasets"][world]["assays"][assay]
            width = protocol["widths"][assay]
            feature_ids = protocol["feature_ids"][world][assay]
            source_indices = protocol["source_indices"][world][assay]
            if (feature_ids != spec["selected_feature_ids"][:width]
                    or source_indices != spec["selected_source_column_indices"][:width]):
                raise ValueError("Protocol feature panel differs from frozen prefix")
            entry = spec["cohorts"]["train"]
            path = within(repo_root, entry["path"])
            require_hash(path, entry["sha256"])
            input_hashes[entry["path"]] = entry["sha256"]
            with np.load(path, allow_pickle=False) as data:
                ids = data["cell_ids"].astype(str)
                if len(set(ids)) != len(ids) or train_ids[world].intersection(ids):
                    raise ValueError("Training identifiers are not unique and disjoint")
                if data["feature_ids"].astype(str).tolist() != spec["selected_feature_ids"]:
                    raise ValueError("Training feature identifiers changed")
                train_ids[world].update(ids)
            entry = protocol["assessment_inputs"][world][assay]
            require_hash(within(repo_root, entry["path"]), entry["sha256"])
            input_hashes[entry["path"]] = entry["sha256"]
    return protocol, seal, artifacts, train_ids, input_hashes


def read_assessment(protocol: dict, repo_root: Path, train_ids: dict) -> tuple[dict, dict]:
    """Slice cached CSR matrices; never regenerate counts or consult latent truth."""
    spec = importlib.util.spec_from_file_location("count_transport_frozen_count_helpers", EXTRACTION_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    counts, audits = {}, {}
    for world in protocol["worlds"]:
        counts[world], anchor_ids = {}, None
        for assay in protocol["assays"]:
            entry = protocol["assessment_inputs"][world][assay]
            path = within(repo_root, entry["path"])
            # Recheck each assessment file immediately before opening it.
            require_hash(path, entry["sha256"])
            matrix, ids, features = helper.h5_counts(path)
            if (len(set(ids)) != len(ids) or len(set(features)) != len(features)
                    or matrix.shape != (len(ids), len(features)) or train_ids[world].intersection(ids)):
                raise ValueError("Invalid or overlapping assessment identifiers")
            chosen = np.asarray(protocol["source_indices"][world][assay], dtype=int)
            if (len(chosen) != protocol["widths"][assay] or (chosen < 0).any()
                    or (chosen >= len(features)).any()
                    or features[chosen].tolist() != protocol["feature_ids"][world][assay]):
                raise ValueError("Assessment feature panel differs from frozen panel")
            if anchor_ids is None:
                anchor_ids = ids.copy()
            elif not np.array_equal(ids, anchor_ids):
                raise ValueError("Cached assessment assays are not in identical paired ID order")
            counts[world][assay] = count_array(matrix[:, chosen].toarray())
        audits[world] = {"paired_rows": len(anchor_ids), "train_assessment_disjoint": True,
                         "paired_ids_sha256": hashlib.sha256("\n".join(anchor_ids).encode()).hexdigest(),
                         "feature_order_verified": True}
    return counts, audits


def restart_summary(reports: dict[str, dict], names: list[str]) -> dict:
    def span(values):
        return [min(values), max(values)] if values else None
    scored = [name for name in names if reports[name]["evaluation_status"] == "scored"]
    directions = reports[scored[0]]["pairs"] if scored else {}
    return {"fits": len(names), "accepted_fits": sum(reports[n]["fit_accepted"] for n in names),
            "scored_fits": len(scored), "failed_fits": len(names) - len(scored),
            "interpretation": "All frozen starts at the calibration-selected dimension; no reselection. Ranges are not confidence intervals.",
            "pairs": {direction: {
                "conditional_loglik_gain_range": span([
                    reports[n]["pairs"][direction]["conditional_loglik_gain"]["mean"] for n in scored]),
                "conditional_loglik_gain_per_target_feature_range": span([
                    reports[n]["pairs"][direction]["conditional_loglik_gain_per_target_feature"]["mean"] for n in scored]),
                "mean_mse_reduction_range": span([
                    reports[n]["pairs"][direction]["mean_mse_reduction"]["mean"] for n in scored]),
                "accepted_positive_family_adjusted_fits": sum(
                    reports[n]["pairs"][direction]["accepted_fit_positive_family_adjusted_gain"] for n in scored),
            } for direction in directions}}


def evaluate_run(run_dir: Path, repo_root: Path = REPO) -> dict:
    run_dir, repo_root = Path(run_dir).resolve(), Path(repo_root).resolve()
    if any((run_dir / name).exists() for name in ("evaluation.json", "evaluation_seal.json")):
        raise FileExistsError("Evaluation outputs already exist; results cannot be overwritten")
    protocol, seal, artifacts, train_ids, input_hashes = verify_frozen(run_dir, repo_root)
    counts, audits = read_assessment(protocol, repo_root, train_ids)
    boot = {world: bootstrap_indices(len(next(iter(blocks.values()))), world)
            for world, blocks in counts.items()}
    report = {
        "target": "Heldout paired count-law transport on frozen background panels; not causal DAG recovery",
        "direction_interpretation": "Source-target labels describe prediction, not causal orientation. Paired per-cell log-score gains equal log joint minus both log marginals and are symmetric: six ordered directions contain three distinct association statistics per world. Per-target-feature gains can differ because panel widths differ.",
        "uncertainty": {"bootstrap_repetitions": BOOTSTRAPS,
                        "scheme": "Paired-cell rows; same resamples per world across directions and frozen starts",
                        "family_comparisons": FAMILY_COMPARISONS,
                        "family_scope": "Two worlds times six original-data directions, separately per frozen variant; controls and restart summaries are descriptive",
                        "family_interval_caveat": "Bonferroni-adjusted percentile intervals are approximate and exploratory, not guaranteed familywise coverage. With 1000 resamples, each adjusted tail contains about two bootstrap estimates; symmetric directions are conservatively counted twice.",
                        "shuffle_interval_caveat": "Fixed source permutations link observations, so paired-row intervals involving source-shuffled scores are descriptive comparisons, not independent-row confidence guarantees.",
                        "note": "Sampling uncertainty conditional on each fixed fit; no fit uncertainty or source-identifiability guarantee. Covariance diagnostics are point estimates."},
        "input_audits": audits, "selected": {}, "restart_stability": {}, "fits": {},
    }
    for key, filename in seal["selected"].items():
        names = seal["selected_dimension_restarts"][key]
        for name in names:
            result = artifacts[name]
            report["fits"][name] = evaluate_fit(result, counts[result["world"]], protocol["assays"],
                                                protocol["widths"], boot[result["world"]], key)
        report["selected"][key] = filename
        report["restart_stability"][key] = restart_summary(report["fits"], names)
    write_exclusive(run_dir / "evaluation.json", report)
    write_exclusive(run_dir / "evaluation_seal.json", {
        "protocol_sha256": sha(run_dir / "protocol.json"),
        "fit_seal_sha256": sha(run_dir / "fit_seal.json"),
        "inputs": input_hashes,
        "outputs": {"evaluation.json": sha(run_dir / "evaluation.json")},
        "no_fitting_or_reselection": True,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    report = evaluate_run(args.run_dir)
    print(json.dumps({"evaluation": str(args.run_dir / "evaluation.json"),
                      "selected_fits": len(report["selected"]),
                      "evaluated_fits": len(report["fits"])}))


if __name__ == "__main__":
    main()
