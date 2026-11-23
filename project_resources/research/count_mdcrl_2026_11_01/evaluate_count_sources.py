"""Independent, sealed-artifact evaluation of the fixed count-source pilot.

This executable does not import either source estimator or fitting runner. It
opens test counts and evaluator references only after verifying every fit seal.
All covariance estimates use the same ddof=0 moment convention as fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[2]
PUBLIC_KEYS = {"raw_counts", "log1p_cp10k", "library_depth", "cell_ids", "feature_ids"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(finite_json(value), indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def expected_arms(protocol):
    return {f"{dataset}__{exposure}__{method}__seed{seed}"
            + ("__independent_columns" if control != "original" else ""):
            (dataset, exposure, method, seed, control)
            for dataset in protocol["datasets"] for exposure in protocol["exposures"]
            for control, seeds in protocol["controls"].items()
            for method in protocol["methods"] for seed in seeds}


def verify_fit_seal(study=STUDY):
    """Finish all seal checks without opening any public test or reference NPZ."""
    protocol = json.loads((study / "protocol.json").read_text())
    runs = study / "runs_v1"
    seal = json.loads((runs / "provenance.json").read_text())
    require(seal["evaluator_read"] is False, "Training provenance does not exclude evaluator reads")
    require(seal["source_sha256"]["protocol.json"] == sha(study / "protocol.json"),
            "Frozen protocol hash changed")
    for name, digest in seal["source_sha256"].items():
        require(sha(study / name) == digest, f"Fit source hash changed: {name}")
    for name, digest in seal["support_sha256"].items():
        require(sha(study.parent / name) == digest, f"Supporting source hash changed: {name}")
    for name, digest in seal["artifacts"].items():
        require(sha(runs / name) == digest, f"Sealed fit artifact changed: {name}")
    require("summary.json" in seal["artifacts"], "Summary is not sealed")
    summary = json.loads((runs / "summary.json").read_text())
    expected = expected_arms(protocol)
    require(len(expected) == protocol["expected_arm_records"] == len(summary), "Incomplete arm block")
    require(len({a["path"] for a in summary}) == len(summary), "Duplicate arm in summary")
    require({a["path"] for a in summary} == set(expected), "Unexpected or missing arm")
    records = {}
    for entry in summary:
        name = entry["path"]
        require(f"{name}/result.json" in seal["artifacts"], f"Unsealed result: {name}")
        result = json.loads((runs / name / "result.json").read_text())
        actual = tuple(result[k] for k in ["dataset", "exposure", "method", "source_seed", "control"])
        require(actual == expected[name], f"Arm identity mismatch: {name}")
        require(entry["status"] == result["status"], f"Status mismatch: {name}")
        if result["status"] != "failed":
            require(f"{name}/arrays.npz" in seal["artifacts"], f"Unsealed arrays: {name}")
            require(result["arrays_sha256"] == seal["artifacts"][f"{name}/arrays.npz"],
                    f"Arrays hash mismatch: {name}")
        records[name] = result
    data = (study / protocol["data"]).resolve()
    digest = sha(data / "manifest.json")
    require(digest == protocol["data_manifest_sha256"] == seal["data_manifest_sha256"],
            "Frozen count manifest changed")
    manifest = json.loads((data / "manifest.json").read_text())
    return protocol, seal, records, data, manifest


def split_indices(ids):
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        ("count-dica-v1:" + str(ids[i])).encode()).hexdigest())
    cut = int(0.75 * len(ids))
    return np.array(order[:cut], dtype=int), np.array(order[cut:], dtype=int)


def exposure_from_original(counts, depth, mode, normalization):
    """The denominator is computed from original counts, before any permutation."""
    if mode == "unit":
        require(normalization == 1, "Unit exposure has a nonunit normalizer")
        return np.ones(len(counts))
    require(mode == "leave_panel_out", f"Unknown exposure: {mode}")
    outside = np.asarray(depth, float) - np.asarray(counts, float).sum(axis=1)
    require(np.isfinite(outside).all() and (outside > 0).all(), "Invalid outside-panel depth")
    require(np.isfinite(normalization) and normalization > 0, "Invalid exposure normalizer")
    return outside / normalization


def replay_training_counts(public, result, assay, protocol):
    """Replay only the frozen control randomization, never source fitting."""
    x = public["raw_counts"].astype(float).copy()
    fit, cal = split_indices(public["cell_ids"])
    normalization = result["exposure_normalization"][assay]
    outside = public["library_depth"].astype(float) - x.sum(axis=1)
    expected = 1.0 if result["exposure"] == "unit" else float(np.median(outside[fit]))
    require(normalization == expected, "Exposure normalization was not fit-split-only")
    exposure = exposure_from_original(x, public["library_depth"], result["exposure"], normalization)
    if result["control"] == "independent_columns":
        seed_text = ":".join(map(str, [protocol["control_seed"], result["dataset"],
                                      result["exposure"], assay]))
        rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16))
        for rows in [fit, cal]:
            block = x[rows].copy()
            local_exp = exposure[rows]
            strata = ([np.arange(len(rows))] if np.ptp(local_exp) == 0 else
                      np.array_split(np.argsort(local_exp, kind="stable"), 8))
            for stratum in strata:
                for column in range(x.shape[1]):
                    x[rows[stratum], column] = block[rng.permutation(stratum), column]
    else:
        require(result["control"] == "original", "Unknown control")
    return x, exposure, fit, cal


def covariance(a, b=None):
    a = np.asarray(a, float)
    b = a if b is None else np.asarray(b, float)
    require(a.ndim == b.ndim == 2 and len(a) == len(b) and len(a) > 1,
            "Covariance requires two aligned matrices with at least two rows")
    return (a - a.mean(axis=0)).T @ (b - b.mean(axis=0)) / len(a)


def correlation(a, b):
    cov = covariance(a, b)
    denom = np.outer(np.std(a, axis=0), np.std(b, axis=0))
    return np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)


def errors(prediction, observed, mask=None):
    prediction, observed = np.asarray(prediction), np.asarray(observed)
    require(prediction.shape == observed.shape, "Metric matrices have different shapes")
    use = np.ones(observed.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    count = int(use.sum())
    if not count or not np.isfinite(prediction[use]).all() or not np.isfinite(observed[use]).all():
        return {"status": "unscorable", "entries": count, "rmse": None, "mae": None}
    delta = prediction[use] - observed[use]
    return {"status": "ok", "entries": count, "rmse": float(np.sqrt(np.mean(delta**2))),
            "mae": float(np.mean(np.abs(delta)))}


def covariance_comparison(prediction, observed):
    model = errors(prediction, observed)
    zero = errors(np.zeros_like(observed), observed)
    improvement = None if model["rmse"] is None else zero["rmse"] - model["rmse"]
    return {"model": model, "zero": zero, "rmse_improvement_over_zero": improvement,
            "relative_rmse_improvement_over_zero":
                None if improvement is None or zero["rmse"] == 0 else improvement / zero["rmse"]}


def bootstrap_covariances(left, right, indices):
    require(np.shape(indices)[1] == len(left), "Bootstrap rows do not represent complete cell resamples")
    return np.stack([covariance(left[idx], right[idx]) for idx in indices])


def bootstrap_errors(prediction, covariances, scale):
    observed = covariances / scale
    model = np.sqrt(np.mean((observed - prediction)**2, axis=(1, 2)))
    zero = np.sqrt(np.mean(observed**2, axis=(1, 2)))
    difference = zero - model
    return {"model_rmse": model, "zero_rmse": zero, "rmse_improvement_over_zero": difference}, {
        "replicates": len(difference), "seed": 3000,
        "interval": "2.5% and 97.5% percentile; conditional on the frozen fit; descriptive",
        "model_rmse_percentiles": np.quantile(model, [0.025, 0.5, 0.975]),
        "zero_rmse_percentiles": np.quantile(zero, [0.025, 0.5, 0.975]),
        "rmse_improvement_percentiles": np.quantile(difference, [0.025, 0.5, 0.975]),
        "fraction_with_positive_improvement": float(np.mean(difference > 0)),
        "fraction_is_not_a_p_value": True}


def correlation_summary(matrix):
    matrix = np.asarray(matrix)
    return {"score_count": matrix.shape[0], "state_count": matrix.shape[1],
            "valid_entries": int(np.isfinite(matrix).sum()),
            "max_absolute_correlation_per_state": [
                float(np.max(np.abs(col[np.isfinite(col)]))) if np.isfinite(col).any() else None
                for col in matrix.T]}


def load_public(data, manifest, dataset, assay, partition):
    item = manifest["datasets"][dataset]["assays"][assay]["cohorts"][partition]
    path = data / "public" / dataset / partition / f"{assay}.npz"
    require(sha(path) == item["sha256"], f"Public source changed: {path}")
    z = read_npz(path)
    require(set(z) == PUBLIC_KEYS, f"Unexpected public fields: {path}")
    counts = z["raw_counts"]
    require(counts.ndim == 2 and counts.shape[1] == 30, "Panel feature count changed")
    require(np.isfinite(counts).all() and (counts >= 0).all() and np.equal(counts, np.floor(counts)).all(),
            "Invalid raw counts")
    require(z["library_depth"].shape == z["cell_ids"].shape == (len(counts),), "Invalid row metadata")
    require(z["feature_ids"].shape == (30,), "Invalid feature metadata")
    require(len(set(z["cell_ids"])) == len(counts) and len(set(z["feature_ids"])) == 30,
            "Duplicated IDs")
    require(np.isfinite(z["library_depth"]).all() and (z["library_depth"] > 0).all(), "Invalid library depths")
    require(np.array_equal(z["feature_ids"], manifest["datasets"][dataset]["assays"][assay]["selected_feature_ids"]),
            "Feature ordering changed")
    require(np.allclose(z["log1p_cp10k"], np.log1p(1e4 * counts / z["library_depth"][:, None]),
                        rtol=1e-12, atol=1e-12), "Frozen normalization is inconsistent")
    return z


def validate_arm_arrays(result, arrays, public, protocol):
    assays = protocol["assays"]
    records = {record["assay"]: record for record in result["assays"]}
    require(set(records) == set(assays), "Assay result set is incomplete")
    for assay in assays:
        train = public[assay]["train"]
        x, exposure, fit, cal = replay_training_counts(train, result, assay, protocol)
        require(np.array_equal(arrays[f"features_{assay}"], train["feature_ids"]), "Fit features changed")
        require(np.array_equal(arrays[f"fit_indices_{assay}"], fit), "Fit partition changed")
        require(np.array_equal(arrays[f"calibration_indices_{assay}"], cal), "Calibration partition changed")
        rate = x[fit] / exposure[fit, None]
        noise = np.mean(x[fit] / exposure[fit, None]**2, axis=0)
        expected_noise = noise if result["method"] == "poisson_dica" else np.zeros(30)
        for key, expected in [("mean", rate.mean(axis=0)), ("scale", rate.std(axis=0)),
                              ("noise_diagonal", expected_noise), ("poisson_noise_diagonal", noise)]:
            require(np.allclose(arrays[f"{key}_{assay}"], expected, rtol=1e-8, atol=1e-10),
                    f"Training-only {key} replay differs for {assay}")
        mixing, unmixing = arrays[f"mixing_{assay}"], arrays[f"unmixing_{assay}"]
        require(mixing.shape[0] == unmixing.shape[1] == 30 and mixing.shape[1] == unmixing.shape[0],
                "Invalid mixing/unmixing dimensions")
        within = arrays[f"predicted_observed_covariance_{assay}"]
        if records[assay]["status"] == "ok":
            require(mixing.shape[1] == records[assay]["k"], "Fitted rank does not match record")
            expected = mixing @ mixing.T + np.diag(expected_noise)
            require(np.allclose(within, expected, rtol=1e-8, atol=1e-10),
                    "Observed covariance has wrong measurement-noise semantics")
        else:
            require(np.isnan(within).all(), "Abstained model covariance is not explicitly undefined")
    for a, b in combinations(assays, 2):
        accepted = [m for m in result["pair_matches"] if (m["a"], m["b"]) == (a, b)]
        require(len({m["ia"] for m in accepted}) == len({m["ib"] for m in accepted}) == len(accepted),
                "Accepted sources are not one-to-one")
        expected = np.zeros((30, 30))
        for match in accepted:
            require(match["equivalent"] and match["source_gates_pass"] and match["sign"] in [-1, 1],
                    "Cross covariance includes an unaccepted source")
            require(records[a]["eligible_sources"][match["ia"]] and records[b]["eligible_sources"][match["ib"]],
                    "Cross covariance uses an ineligible source")
            expected += match["sign"] * np.outer(arrays[f"mixing_{a}"][:, match["ia"]],
                                                  arrays[f"mixing_{b}"][:, match["ib"]])
        require(np.allclose(arrays[f"predicted_cross_covariance_{a}_{b}"], expected, rtol=1e-10, atol=1e-12),
                "Predicted cross covariance does not match accepted source products")
    return records


def evaluate_within(result, arrays, public, assays, saved):
    output, rates = {}, {}
    for assay in assays:
        test = public[assay]["test"]
        exposure = exposure_from_original(test["raw_counts"], test["library_depth"], result["exposure"],
                                          result["exposure_normalization"][assay])
        rate = test["raw_counts"].astype(float) / exposure[:, None]
        rates[assay] = rate
        observed = covariance(rate)
        prediction = arrays[f"predicted_observed_covariance_{assay}"]
        scale = arrays[f"scale_{assay}"]
        saved[f"observed_covariance_{assay}"] = observed
        saved[f"observed_mean_{assay}"] = rate.mean(axis=0)
        record = {"test_cells": len(rate), "test_all_panel_zero_cells": int(np.sum(~np.any(rate, axis=1))),
                  "fit_mean_rate": arrays[f"mean_{assay}"], "test_mean_rate": rate.mean(axis=0),
                  "test_detected_cells_per_feature": np.sum(test["raw_counts"] > 0, axis=0),
                  "scale": "source-fit observed-rate SDs; covariance ddof=0"}
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            record.update(status="unscorable", reason="invalid fixed source-fit scale; no features removed")
        else:
            standardized = observed / np.outer(scale, scale)
            predicted = prediction / np.outer(scale, scale)
            baseline = np.eye(len(scale))
            off = ~np.eye(len(scale), dtype=bool)
            record.update(status="ok" if np.isfinite(prediction).all() else "model_abstained",
                          model_full=errors(predicted, standardized),
                          diagonal_training_baseline_full=errors(baseline, standardized),
                          model_offdiagonal=errors(predicted, standardized, off),
                          diagonal_training_baseline_offdiagonal=errors(baseline, standardized, off),
                          standardized_mean_rmse=float(np.sqrt(np.mean(
                              ((rate.mean(axis=0) - arrays[f"mean_{assay}"]) / scale)**2))))
            saved[f"observed_standardized_covariance_{assay}"] = standardized
            saved[f"predicted_standardized_covariance_{assay}"] = predicted
        output[assay] = record
    return output, rates


def verify_real_reference(reference, public, assays):
    n = len(reference["canonical_ids"])
    require(n == 595 and len(set(reference["canonical_ids"])) == n, "Real evaluation population changed")
    ordered, feature_ids = [], []
    for assay in assays:
        test = public[assay]["test"]
        indices = reference[f"paired_indices_{assay}"]
        require(indices.shape == (n,) and np.issubdtype(indices.dtype, np.integer), "Invalid paired indices")
        require(np.array_equal(np.sort(indices), np.arange(len(test["raw_counts"]))),
                "Paired reference does not use exactly the fixed test population")
        ordered.append(test["raw_counts"][indices])
        feature_ids.extend(test["feature_ids"])
    require(np.array_equal(np.concatenate(ordered, axis=1), reference["raw_counts"]), "Paired count join differs")
    require(np.array_equal(feature_ids, reference["column_ids"]), "Paired feature ordering differs")
    require(np.array_equal(np.repeat(assays, 30), reference["column_assays"]), "Paired assay ordering differs")


def evaluate_real(result, arrays, rates, reference, assays, saved, bootstrap_cache):
    endpoint_records = {record["assay"]: record for record in result["assays"]}
    key = (result["exposure"], tuple(result["exposure_normalization"][a] for a in assays))
    if key not in bootstrap_cache:
        indices = np.random.default_rng(3000).integers(0, 595, size=(200, 595))
        paired = {a: rates[a][reference[f"paired_indices_{a}"]] for a in assays}
        bootstrap_cache[key] = {(a, b): bootstrap_covariances(paired[a], paired[b], indices)
                                for a, b in combinations(assays, 2)}
    output = {}
    for a, b in combinations(assays, 2):
        label = f"{a}__{b}"
        left, right = rates[a][reference[f"paired_indices_{a}"]], rates[b][reference[f"paired_indices_{b}"]]
        observed = covariance(left, right)
        prediction = arrays[f"predicted_cross_covariance_{a}_{b}"]
        scale = np.outer(arrays[f"scale_{a}"], arrays[f"scale_{b}"])
        saved[f"paired_observed_cross_covariance_{a}_{b}"] = observed
        saved[f"predicted_cross_covariance_{a}_{b}"] = prediction
        record = {"paired_cells": len(left), "accepted_pair_sources": sum(
            (m["a"], m["b"]) == (a, b) for m in result["pair_matches"]),
            "interpretation": "observed joint moment agreement; not causal truth"}
        record["zero_predictor_due_to_no_accepted_sources"] = record["accepted_pair_sources"] == 0
        record["endpoint_source_fits"] = {
            assay: {"status": endpoint_records[assay]["status"],
                    "reason": endpoint_records[assay].get("reason"),
                    "requested_rank": endpoint_records[assay]["k"]}
            for assay in [a, b]}
        record["zero_predictor_meaning"] = (
            "Empty-candidate default; not an estimated independence conclusion"
            if record["zero_predictor_due_to_no_accepted_sources"] else None)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            record["primary"] = {"status": "unscorable", "reason": "invalid fixed fit scale; no pairs removed"}
        else:
            standardized, predicted = observed / scale, prediction / scale
            record["primary"] = covariance_comparison(predicted, standardized)
            bootstrap_arrays, intervals = bootstrap_errors(predicted, bootstrap_cache[key][a, b], scale)
            record["primary"]["paired_cell_bootstrap"] = intervals
            saved[f"paired_observed_standardized_cross_covariance_{a}_{b}"] = standardized
            saved[f"predicted_standardized_cross_covariance_{a}_{b}"] = predicted
            for name, value in bootstrap_arrays.items():
                saved[f"bootstrap_{name}_{a}_{b}"] = value
        var_a = np.diag(arrays[f"predicted_observed_covariance_{a}"])
        var_b = np.diag(arrays[f"predicted_observed_covariance_{b}"])
        obs_a, obs_b = np.var(left, axis=0), np.var(right, axis=0)
        good_a = np.isfinite(var_a) & (var_a > 0) & (obs_a > 0)
        good_b = np.isfinite(var_b) & (var_b > 0) & (obs_b > 0)
        mask = good_a[:, None] & good_b[None, :]
        expected = np.full((30, 30), np.nan)
        empirical = correlation(left, right)
        if mask.any():
            denominator = np.sqrt(np.maximum(var_a, 0)[:, None] * np.maximum(var_b, 0)[None, :])
            expected[mask] = prediction[mask] / denominator[mask]
        record["secondary_pearson"] = {
            "model": errors(expected, empirical, mask), "zero": errors(np.zeros_like(empirical), empirical, mask),
            "valid_pairs": int(mask.sum()), "total_pairs": 900,
            "undefined_model_variance_features": {a: int(np.sum(~np.isfinite(var_a) | (var_a <= 0))),
                                                   b: int(np.sum(~np.isfinite(var_b) | (var_b <= 0)))},
            "undefined_test_variance_features": {a: int(np.sum(obs_a <= 0)), b: int(np.sum(obs_b <= 0))},
            "mask_does_not_change_primary": True}
        saved[f"secondary_model_correlation_{a}_{b}"] = expected
        saved[f"secondary_test_correlation_{a}_{b}"] = empirical
        saved[f"secondary_valid_mask_{a}_{b}"] = mask
        output[label] = record
    return output


def evaluate_state(result, arrays, public, rates, reference, assays, saved):
    output = {}
    for position, assay in enumerate(assays):
        require(np.array_equal(reference[f"ids_{assay}"], public[assay]["test"]["cell_ids"]),
                "Simulation state IDs differ from public test row order")
        state = reference[f"state_{assay}"]
        require(state.shape == (len(rates[assay]), 2) and np.isfinite(state).all(), "Invalid state reference")
        scores = (rates[assay] - arrays[f"mean_{assay}"]) @ arrays[f"unmixing_{assay}"].T
        matrix = correlation(scores, state)
        matched = sorted({m["ia"] if m["a"] == assay else m["ib"] for m in result["pair_matches"]
                          if assay in [m["a"], m["b"]]})
        all_assay = sorted({m["indices"][position] for m in result["all_assay_matches"]})
        output[assay] = {"all_sources": correlation_summary(matrix),
                         "pair_matched_sources": correlation_summary(matrix[matched]),
                         "all_assay_matched_sources": correlation_summary(matrix[all_assay]),
                         "pair_matched_indices": matched, "all_assay_matched_indices": all_assay,
                         "score_interpretation": "frozen noisy count-rate projection, not posterior biological state",
                         "maximum_correlation_has_factor_selection_multiplicity": True}
        saved[f"state_correlation_all_sources_{assay}"] = matrix
        saved[f"state_correlation_pair_matched_{assay}"] = matrix[matched]
        saved[f"state_correlation_all_assay_matched_{assay}"] = matrix[all_assay]
    return output


def main(study=STUDY, output_name="evaluation_v1"):
    protocol, seal, records, data, manifest = verify_fit_seal(study)
    output = study / output_name
    require(not output.exists(), f"Refusing to overwrite evaluation: {output}")
    evaluator_hash = sha(Path(__file__))
    source_protocol_hash = sha(study / "protocol.json")
    public, references, consumed = {}, {}, {}
    # Every training run has been sealed before this first held-out array read.
    for dataset in protocol["datasets"]:
        public[dataset] = {}
        for assay in protocol["assays"]:
            public[dataset][assay] = {partition: load_public(data, manifest, dataset, assay, partition)
                                      for partition in ["train", "test"]}
            require(set(public[dataset][assay]["train"]["cell_ids"]).isdisjoint(
                public[dataset][assay]["test"]["cell_ids"]), "Train/test IDs overlap")
            for partition in ["train", "test"]:
                item = manifest["datasets"][dataset]["assays"][assay]["cohorts"][partition]
                consumed[item["path"]] = item["sha256"]
        ref = manifest["evaluator"][dataset]
        ref_path = REPO / ref["path"]
        require(sha(ref_path) == ref["sha256"], "Evaluator reference changed")
        references[dataset] = read_npz(ref_path)
        consumed[ref["path"]] = ref["sha256"]
        if dataset == "scmtr_seq_s1":
            verify_real_reference(references[dataset], public[dataset], protocol["assays"])
    output.mkdir(exist_ok=False)
    summary, bootstrap_cache = [], {}
    for name, result in records.items():
        target = output / name
        target.mkdir(exist_ok=False)
        report = {"arm": name, "dataset": result["dataset"], "exposure": result["exposure"],
                  "method": result["method"], "source_seed": result["source_seed"],
                  "control": result["control"], "fit_status": result["status"],
                  "fit_result_sha256": seal["artifacts"][f"{name}/result.json"],
                  "causal_recovery_claim": False}
        if result["status"] == "failed":
            report.update(evaluation_status="fit_failed", error=result.get("error"))
        else:
            cohort = public[result["dataset"]]
            for assay in protocol["assays"]:
                input_name = f"public/{result['dataset']}/train/{assay}.npz"
                require(result["input_hashes"][input_name] == manifest["datasets"][result["dataset"]]["assays"][assay]["cohorts"]["train"]["sha256"],
                        "Fit input hash is inconsistent with frozen count source")
            arrays = read_npz(study / "runs_v1" / name / "arrays.npz")
            assay_records = validate_arm_arrays(result, arrays, cohort, protocol)
            saved = {}
            within, rates = evaluate_within(result, arrays, cohort, protocol["assays"], saved)
            report.update(evaluation_status="complete", within_assay=within,
                          ranks={a: assay_records[a]["k"] for a in protocol["assays"]},
                          pair_match_count=len(result["pair_matches"]),
                          all_assay_match_count=len(result["all_assay_matches"]),
                          graph_status=result["graph_status"])
            if result["dataset"] == "scmtr_seq_s1":
                report["real_cross_assay"] = evaluate_real(result, arrays, rates, references[result["dataset"]],
                                                          protocol["assays"], saved, bootstrap_cache)
            else:
                report["simulation_state"] = evaluate_state(result, arrays, cohort, rates, references[result["dataset"]],
                                                             protocol["assays"], saved)
            np.savez_compressed(target / "arrays.npz", **saved)
            report["arrays_sha256"] = sha(target / "arrays.npz")
        write_json(target / "result.json", report)
        summary.append(report)
    write_json(output / "summary.json", summary)
    require(sha(Path(__file__)) == evaluator_hash, "Evaluator changed during evaluation")
    require(sha(study / "protocol.json") == source_protocol_hash, "Protocol changed during evaluation")
    # Recheck immutable training artifacts after evaluation; never rewrite them.
    verify_fit_seal(study)
    write_json(output / "provenance.json", {
        "status": "complete", "evaluator_sha256": evaluator_hash,
        "protocol_sha256": source_protocol_hash, "fit_provenance_sha256": sha(study / "runs_v1/provenance.json"),
        "data_manifest_sha256": sha(data / "manifest.json"), "consumed_input_sha256": consumed,
        "training_or_threshold_selection_performed": False, "private_or_new_cohorts_read": False,
        "covariance_convention": "finite-sample plug-in moments; ddof=0",
        "scope": "Previously inspected development panels; internal cell holdout; no real causal truth",
        "artifacts": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"status": "complete", "evaluated_arms": len(summary), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="evaluation_v1")
    args = parser.parse_args()
    main(output_name=args.output_name)
