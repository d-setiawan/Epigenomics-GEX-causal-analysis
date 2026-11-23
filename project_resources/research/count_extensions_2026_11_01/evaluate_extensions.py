"""Independent heldout scoring for frozen count-extension artifacts.

No estimator is imported here. Bootstrap moment errors use exact row-Gram
identities rather than materializing every feature-pair covariance resample.
The scientific fit schema is supplied by the separately frozen runner protocol.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[2]
UTILITY_SOURCE = STUDY.parent / "count_mdcrl_2026_11_01/evaluate_count_sources.py"
_spec = importlib.util.spec_from_file_location("prior_independent_evaluation_utilities", UTILITY_SOURCE)
utility = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utility)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariance(left, right=None):
    right = left if right is None else right
    require(left.ndim == right.ndim == 2 and len(left) == len(right) and len(left) > 1,
            "Covariance requires aligned rows")
    return (left - left.mean(0)).T @ (right - right.mean(0)) / len(left)


def bootstrap_multiplicities(n, repetitions=200, seed=3000):
    rows = np.random.default_rng(seed).integers(0, n, size=(repetitions, n))
    counts = np.array([np.bincount(row, minlength=n) for row in rows], dtype=float)
    return rows, counts


def _roundoff_nonnegative(values, reference):
    tolerance = 1e-9 * np.maximum(reference, 1.)
    require(np.all(values >= -tolerance), "Negative squared error exceeds floating-point tolerance")
    return np.maximum(values, 0.)


def bootstrap_cross_covariance_errors(left, right, prediction, left_scale, right_scale,
                                      multiplicities, observed_cache=None):
    """Exact fixed-scale paired-cell bootstrap RMSE using O(n^2) work buffers.

    Every row weight is shared across both assays. The predicted raw covariance
    and scales stay fixed. With A=diag(w/n)-ww'/n^2, C=X'AY; its Frobenius norm
    follows from the row Gram matrices. No feature pairs are removed.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    prediction = np.asarray(prediction, float)
    left_scale, right_scale = np.asarray(left_scale, float), np.asarray(right_scale, float)
    weights = np.asarray(multiplicities, float)
    n, p, q = len(left), left.shape[1], right.shape[1]
    require(len(right) == n and weights.ndim == 2 and weights.shape[1] == n, "Bootstrap row mismatch")
    require(np.all(weights >= 0) and np.all(weights == np.floor(weights))
            and np.all(weights.sum(1) == n), "Bootstrap multiplicities must describe n complete cells")
    require(left_scale.shape == (p,) and right_scale.shape == (q,), "Invalid fixed scales")
    require(np.isfinite(left_scale).all() and np.isfinite(right_scale).all()
            and np.all(left_scale > 0) and np.all(right_scale > 0), "Invalid fixed scales; no pairs removed")
    require(prediction.shape == (p, q) and np.isfinite(prediction).all(), "Invalid predicted covariance")
    x = (left - left.mean(0)) / left_scale
    y = (right - right.mean(0)) / right_scale
    predicted = prediction / np.outer(left_scale, right_scale)
    if observed_cache is None:
        gx, gy = x @ x.T, y @ y.T
        wx, wy = weights @ gx, weights @ gy
        term1 = np.einsum("bi,bi->b", weights @ (gx * gy), weights) / n**2
        term2 = 2 * np.sum(weights * wx * wy, axis=1) / n**3
        term3 = np.sum(weights * wx, axis=1) * np.sum(weights * wy, axis=1) / n**4
        squared_observed = _roundoff_nonnegative(term1 - term2 + term3,
                                                np.abs(term1) + np.abs(term2) + np.abs(term3))
    else:
        squared_observed = np.asarray(observed_cache, float)
        require(squared_observed.shape == (len(weights),), "Observed bootstrap cache shape changed")
    squared_prediction = float(np.sum(predicted**2))
    if np.any(predicted):
        diagonal_inner = np.sum((x @ predicted) * y, axis=1)
        mean_x, mean_y = weights @ x / n, weights @ y / n
        inner = weights @ diagonal_inner / n - np.einsum("bi,ij,bj->b", mean_x, predicted, mean_y, optimize=True)
        squared_error = _roundoff_nonnegative(squared_observed + squared_prediction - 2 * inner,
                                              squared_observed + squared_prediction + 2 * np.abs(inner))
    else:
        squared_error = squared_observed.copy()
    zero_rmse = np.sqrt(squared_observed / (p * q))
    model_rmse = np.sqrt(squared_error / (p * q))
    return {"zero_rmse": zero_rmse, "model_rmse": model_rmse,
            "rmse_improvement_over_zero": zero_rmse - model_rmse,
            "observed_squared_frobenius_cache": squared_observed}


def noise_diagonal_from_kernel(counts, exposure, theta=np.inf, retention=1., correction=True):
    """Independent NB/ZI factorial second-moment replay on the observed-rate scale."""
    counts, exposure = np.asarray(counts, float), np.asarray(exposure, float)
    if exposure.ndim == 1:
        exposure = exposure[:, None]
    require(exposure.shape in [counts.shape, (len(counts), 1)] and np.all(exposure > 0), "Invalid exposure")
    if not correction:
        return np.zeros(counts.shape[1])
    require(theta > 0 and 0 < retention <= 1, "Invalid kernel parameters")
    invtheta = 0. if np.isinf(theta) else 1 / theta
    factorial_second = retention * counts * (counts - 1) / ((1 + invtheta) * exposure**2)
    return np.mean((counts / exposure)**2 - factorial_second, axis=0)


def profile_beta(counts, outside_depth, fit_rows):
    """Independently solve the convex frozen nuisance profile; no source fit."""
    from scipy.optimize import brentq
    from scipy.special import logsumexp
    normalization = float(np.median(outside_depth[fit_rows]))
    z = np.log(outside_depth[fit_rows] / normalization)
    y = counts[fit_rows].sum(axis=1).astype(float)
    total, yz = float(y.sum()), float(y @ z)
    require(total > 0, "No fitting counts for depth profile")
    def derivative(beta):
        weights = np.exp(beta * z - logsumexp(beta * z))
        return total * (weights @ z) - yz
    if derivative(0.) >= 0:
        beta = 0.
    elif derivative(2.) <= 0:
        beta = 2.
    else:
        beta = float(brentq(derivative, 0., 2., xtol=1e-12))
    return beta, normalization, (z, total, yz)


def rates_exposure(public, parameters):
    require(public["raw_counts"].ndim == 2 and public["raw_counts"].shape[1] == 300,
            "Exposure requires the complete frozen 300-column panel")
    mode = parameters["mode"]
    if mode == "unit":
        return np.ones(len(public["raw_counts"]))
    outside = public["library_depth"].astype(float) - public["raw_counts"].sum(axis=1)
    require(np.all(outside > 0), "Nonpositive common outside300 depth")
    require(parameters["normalization"] > 0 and 0 <= parameters["beta"] <= 2, "Invalid fixed depth parameters")
    return (outside / parameters["normalization"]) ** parameters["beta"]


def expected_arm_names(protocol):
    return {f"{dataset}__{variant['name']}__seed{seed}"
            + ("__independent_columns" if control != "original" else ""):
            (dataset, variant, seed, control)
            for dataset in protocol["datasets"] for variant in protocol["variants"]
            for control, seeds in protocol["controls"].items() for seed in seeds}


def verify_seal(study=STUDY):
    """No heldout data are opened until every scientific arm is sealed."""
    protocol = json.loads((study / "protocol.json").read_text())
    runs = study / "runs_v1"
    seal = json.loads((runs / "provenance.json").read_text())
    require(seal["evaluator_read"] is False, "Fitting provenance permits evaluator reads")
    for name, digest in seal["source_sha256"].items():
        require(sha(study / name) == digest, f"Fit source changed: {name}")
    require(seal["source_sha256"]["protocol.json"] == sha(study / "protocol.json"), "Protocol unsealed")
    for name, digest in seal["support_sha256"].items():
        require(sha(study.parent / name) == digest, f"Supporting source changed: {name}")
    for name, digest in seal["artifacts"].items():
        require(sha(runs / name) == digest, f"Fit artifact changed: {name}")
    require("summary.json" in seal["artifacts"], "Fit summary is unsealed")
    summary = json.loads((runs / "summary.json").read_text())
    expected = expected_arm_names(protocol)
    require(len(expected) == len(summary) == protocol["expected_arm_records"], "Incomplete scientific fit block")
    require(len({r["path"] for r in summary}) == len(summary) and {r["path"] for r in summary} == set(expected),
            "Duplicate, missing, or unexpected scientific arms")
    records = {}
    for entry in summary:
        name = entry["path"]
        require(f"{name}/result.json" in seal["artifacts"], "Unsealed arm result")
        result = json.loads((runs / name / "result.json").read_text())
        require((result["dataset"], result["variant"], result["source_seed"], result["control"]) == expected[name],
                f"Arm specification differs: {name}")
        require(result["status"] == entry["status"], "Arm status differs from summary")
        if result["status"] != "failed":
            require(f"{name}/arrays.npz" in seal["artifacts"], "Unsealed model arrays")
            require(result["arrays_sha256"] == seal["artifacts"][f"{name}/arrays.npz"], "Arrays digest mismatch")
        records[name] = result
    data = (study / protocol["data"]).resolve()
    require(sha(data / "manifest.json") == protocol["data_manifest_sha256"] == seal["data_manifest_sha256"],
            "Data manifest differs from fit seal")
    return protocol, seal, records, data, json.loads((data / "manifest.json").read_text())


def load_data(protocol, manifest):
    public, references, hashes = {}, {}, {}
    for dataset in protocol["datasets"]:
        public[dataset] = {}
        for assay in protocol["assays"]:
            public[dataset][assay] = {}
            selected = manifest["datasets"][dataset]["assays"][assay]["selected_feature_ids"]
            for partition in ["train", "test"]:
                record = manifest["datasets"][dataset]["assays"][assay]["cohorts"][partition]
                path = REPO / record["path"]
                require(sha(path) == record["sha256"], "Public data hash mismatch")
                values = utility.read_npz(path)
                require(set(values) == utility.PUBLIC_KEYS, "Unexpected public learner fields")
                counts = values["raw_counts"]
                require(counts.ndim == 2 and counts.shape[1] == 300 and np.isfinite(counts).all()
                        and np.all(counts >= 0) and np.all(counts == np.floor(counts)), "Invalid raw counts")
                require(values["cell_ids"].shape == values["library_depth"].shape == (len(counts),), "Row metadata mismatch")
                require(len(set(values["cell_ids"])) == len(counts), "Duplicate cell IDs")
                require(np.array_equal(values["feature_ids"], selected), "Selected feature order mismatch")
                public[dataset][assay][partition] = values
                hashes[record["path"]] = record["sha256"]
            require(set(public[dataset][assay]["train"]["cell_ids"]).isdisjoint(public[dataset][assay]["test"]["cell_ids"]),
                    "Public train/test IDs overlap")
        item = manifest["evaluator"][dataset]
        path = REPO / item["path"]
        require(sha(path) == item["sha256"], "Evaluator reference changed")
        references[dataset] = utility.read_npz(path)
        hashes[item["path"]] = item["sha256"]
    real = references["scmtr_seq_s1"]
    require(len(set(real["canonical_ids"])) == len(real["canonical_ids"]) == 595, "Real evaluator population changed")
    blocks, columns = [], []
    for assay in protocol["assays"]:
        values = public["scmtr_seq_s1"][assay]["test"]
        rows = real[f"paired_indices_{assay}"]
        require(np.issubdtype(rows.dtype, np.integer) and np.array_equal(np.sort(rows), np.arange(595)),
                "Invalid paired reference indices")
        blocks.append(values["raw_counts"][rows])
        columns.extend(values["feature_ids"])
    require(np.array_equal(np.column_stack(blocks), real["raw_counts"]), "Real expanded paired join differs")
    require(np.array_equal(columns, real["column_ids"]), "Paired reference column order differs")
    require(np.array_equal(real["column_assays"], np.repeat(protocol["assays"], 300)), "Paired assay order differs")
    return public, references, hashes


def training_replay(public, result, assay, protocol, cache):
    from scipy.special import logsumexp
    parameters = result["exposure_parameters"][assay]
    mode = result["variant"]["depth"]
    require(parameters["mode"] == mode, "Exposure mode mismatch")
    key = (result["dataset"], mode, result["control"], assay)
    if key in cache:
        saved = cache[key]
        require(saved["parameters"] == parameters, "Exposure parameters changed across variants or widths")
        return saved
    original = public["raw_counts"].astype(float)
    fit, cal = utility.split_indices(public["cell_ids"])
    outside = public["library_depth"] - original.sum(axis=1)
    require(np.all(outside > 0), "Invalid original common training depth")
    normalizer = float(np.median(outside[fit]))
    if mode == "unit":
        require(parameters["beta"] == 0 and parameters["normalization"] in [1., normalizer], "Unit exposure changed")
    elif mode == "offset":
        require(parameters["beta"] == 1 and parameters["normalization"] == normalizer, "Offset exposure changed")
    else:
        require(mode == "elastic", "Unknown depth mode")
        beta, expected_normalizer, (z, total, yz) = profile_beta(original, outside, fit)
        require(parameters["normalization"] == expected_normalizer, "Depth median was not source-fit-only")
        require(abs(parameters["beta"] - beta) <= 1e-6, "Frozen beta differs from independent convex optimum")
        objective = lambda b: total * logsumexp(b * z) - b * yz
        require(objective(parameters["beta"]) - objective(beta) <= 1e-8 * max(total, 1), "Profile objective regret too large")
    exposure = rates_exposure(public, parameters)
    counts = original.copy()
    if result["control"] == "independent_columns":
        label = ":".join(map(str, [protocol["control_seed"], result["dataset"], mode, assay]))
        rng = np.random.default_rng(int(hashlib.sha256(label.encode()).hexdigest()[:8], 16))
        for rows in [fit, cal]:
            block = counts[rows].copy()
            strata = ([np.arange(len(rows))] if np.ptp(exposure[rows]) == 0 else
                      np.array_split(np.argsort(exposure[rows], kind="stable"), 8))
            for stratum in strata:
                for column in range(300):
                    counts[rows[stratum], column] = block[rng.permutation(stratum), column]
    else:
        require(result["control"] == "original", "Unknown control")
    saved = {"parameters": parameters, "counts": counts, "exposure": exposure, "fit": fit, "cal": cal}
    cache[key] = saved
    return saved


def validate_model(result, arrays, public, protocol, training_cache, anchor_scales):
    variant, assays = result["variant"], protocol["assays"]
    width = variant["width"]
    theta = np.inf if variant["theta"] is None else variant["theta"]
    metadata = {r["assay"]: r for r in result["assays"]}
    require(set(metadata) == set(assays), "Incomplete source fit metadata")
    for assay in assays:
        train = public[assay]["train"]
        replay = training_replay(train, result, assay, protocol, training_cache)
        fit, cal = replay["fit"], replay["cal"]
        x, t = replay["counts"][fit, :width], replay["exposure"][fit]
        rate = x / t[:, None]
        require(np.array_equal(arrays[f"fit_indices_{assay}"], fit)
                and np.array_equal(arrays[f"calibration_indices_{assay}"], cal), "Frozen row split changed")
        require(np.array_equal(arrays[f"features_{assay}"], train["feature_ids"][:width]), "Frozen feature prefix changed")
        noise = noise_diagonal_from_kernel(x, t, theta, variant["retention"], variant["correction"])
        for name, expected in [("mean", rate.mean(0)), ("scale", rate.std(0)), ("noise_diagonal", noise),
                               ("poisson_noise_diagonal", np.mean(x / t[:, None]**2, axis=0))]:
            require(np.allclose(arrays[f"{name}_{assay}"], expected, rtol=1e-8, atol=1e-10),
                    f"Independent training {name} replay differs")
        anchor_key = (result["dataset"], variant["depth"], result["control"], assay)
        scale = arrays[f"scale_{assay}"][:30]
        if anchor_key in anchor_scales:
            require(np.allclose(scale, anchor_scales[anchor_key], rtol=1e-12, atol=1e-12),
                    "Anchor scaling differs across widths or noise/orientation variants")
        else:
            anchor_scales[anchor_key] = scale.copy()
        mixing, unmixing = arrays[f"mixing_{assay}"], arrays[f"unmixing_{assay}"]
        require(mixing.shape[0] == unmixing.shape[1] == width and mixing.shape[1] == unmixing.shape[0],
                "Invalid source matrix dimensions")
        predicted = arrays[f"predicted_observed_covariance_{assay}"]
        if metadata[assay]["status"] == "ok":
            require(np.allclose(predicted, mixing @ mixing.T + np.diag(noise), rtol=1e-8, atol=1e-10),
                    "Predicted observed covariance has wrong kernel-noise restoration")
        else:
            require(np.isnan(predicted).all(), "Abstained within-assay covariance should be undefined")
    for level in ["moment", "confirmed"]:
        for a, b in combinations(assays, 2):
            matches = [m for m in result[f"{level}_pair_matches"] if (m["a"], m["b"]) == (a, b)]
            require(len({m["ia"] for m in matches}) == len({m["ib"] for m in matches}) == len(matches),
                    "Accepted source matching is not one-to-one")
            expected = np.zeros((width, width))
            for match in matches:
                require(match["sign"] in [-1, 1], "Invalid matching sign")
                require(match.get("source_gates_pass", True) and match.get("equivalent", True),
                        "Cross prediction uses rejected candidate")
                expected += match["sign"] * np.outer(arrays[f"mixing_{a}"][:, match["ia"]],
                                                      arrays[f"mixing_{b}"][:, match["ib"]])
            require(np.allclose(arrays[f"predicted_cross_covariance_{level}_{a}_{b}"], expected, rtol=1e-10, atol=1e-12),
                    "Cross prediction differs from accepted source products")
    return metadata


def metric_views(width):
    return [("full_panel", width), ("original30_anchor", 30)]


def within_scores(result, arrays, public, assays):
    rates, output = {}, {}
    for assay in assays:
        test = public[assay]["test"]
        exposure = rates_exposure(test, result["exposure_parameters"][assay])
        rate = test["raw_counts"][:, :result["variant"]["width"]].astype(float) / exposure[:, None]
        rates[assay] = rate
        observed = covariance(rate)
        predicted = arrays[f"predicted_observed_covariance_{assay}"]
        output[assay] = {"test_rows": len(rate), "views": {}}
        for view, width in metric_views(rate.shape[1]):
            scale = arrays[f"scale_{assay}"][:width]
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                output[assay]["views"][view] = {"status": "unscorable", "reason": "Invalid fixed scale; no features removed"}
                continue
            reference = observed[:width, :width] / np.outer(scale, scale)
            prediction = predicted[:width, :width] / np.outer(scale, scale)
            diagonal, off = np.eye(width), ~np.eye(width, dtype=bool)
            output[assay]["views"][view] = {
                "status": "ok" if np.isfinite(prediction).all() else "model_abstained",
                "features": width, "model_full": utility.errors(prediction, reference),
                "diagonal_training_baseline_full": utility.errors(diagonal, reference),
                "model_offdiagonal": utility.errors(prediction, reference, off),
                "diagonal_training_baseline_offdiagonal": utility.errors(diagonal, reference, off),
                "standardized_mean_rmse": float(np.sqrt(np.mean(((rate[:, :width].mean(0)
                    - arrays[f"mean_{assay}"][:width]) / scale)**2)))}
    return rates, output


def real_scores(result, arrays, rates, reference, assays, metadata, weights, observed_cache, saved):
    output = {}
    for a, b in combinations(assays, 2):
        left, right = rates[a][reference[f"paired_indices_{a}"]], rates[b][reference[f"paired_indices_{b}"]]
        observed = covariance(left, right)
        pair = {"paired_cells": 595, "endpoint_source_fits": {
            a: {k: metadata[a].get(k) for k in ["status", "reason", "k"]},
            b: {k: metadata[b].get(k) for k in ["status", "reason", "k"]}}, "levels": {}}
        for level in ["moment", "confirmed"]:
            matched = sum((m["a"], m["b"]) == (a, b) for m in result[f"{level}_pair_matches"])
            record = {"accepted_pair_sources": matched,
                      "zero_predictor_meaning": "Empty-candidate default, not inferred independence" if not matched else None,
                      "interpretation": "finite-moment candidate" if level == "moment" else "finite-grid CF-confirmed candidate",
                      "views": {}}
            for view, width in metric_views(left.shape[1]):
                sa, sb = arrays[f"scale_{a}"][:width], arrays[f"scale_{b}"][:width]
                if not np.isfinite(sa).all() or not np.isfinite(sb).all() or np.any(sa <= 0) or np.any(sb <= 0):
                    record["views"][view] = {"status": "unscorable", "reason": "Invalid fixed scale; no feature pairs removed"}
                    continue
                prediction = arrays[f"predicted_cross_covariance_{level}_{a}_{b}"][:width, :width]
                denominator = np.outer(sa, sb)
                metrics = utility.covariance_comparison(prediction / denominator, observed[:width, :width] / denominator)
                cache_key = (result["variant"]["depth"], result["control"], a, b, width,
                             hashlib.sha256(np.r_[sa, sb].tobytes()).hexdigest(),
                             json.dumps(result["exposure_parameters"], sort_keys=True))
                boot = bootstrap_cross_covariance_errors(left[:, :width], right[:, :width], prediction, sa, sb,
                                                         weights, observed_cache.get(cache_key))
                observed_cache[cache_key] = boot["observed_squared_frobenius_cache"]
                metrics.update(features_per_assay=width, covariance_convention="ddof=0; source-fit observed-rate SDs",
                               paired_cell_bootstrap={"replicates": len(weights),
                                   "interpretation": "2.5%,50%,97.5% descriptive percentiles; fixed fitted model",
                                   "model_rmse_percentiles": np.quantile(boot["model_rmse"], [.025, .5, .975]),
                                   "zero_rmse_percentiles": np.quantile(boot["zero_rmse"], [.025, .5, .975]),
                                   "rmse_improvement_percentiles": np.quantile(boot["rmse_improvement_over_zero"], [.025, .5, .975]),
                                   "fraction_positive_improvement": float(np.mean(boot["rmse_improvement_over_zero"] > 0)),
                                   "fraction_is_not_a_p_value": True})
                for key in ["model_rmse", "zero_rmse", "rmse_improvement_over_zero"]:
                    saved[f"bootstrap_{level}_{view}_{a}_{b}_{key}"] = boot[key]
                record["views"][view] = metrics
            pair["levels"][level] = record
        output[f"{a}__{b}"] = pair
    return output


def state_scores(result, arrays, public, rates, reference, assays, saved):
    output = {}
    for position, assay in enumerate(assays):
        require(np.array_equal(reference[f"ids_{assay}"], public[assay]["test"]["cell_ids"]),
                "State/public-discovery row order changed")
        state = reference[f"state_{assay}"]
        require(state.shape == (len(rates[assay]), 2) and np.isfinite(state).all(), "Invalid state reference")
        scores = (rates[assay] - arrays[f"mean_{assay}"]) @ arrays[f"unmixing_{assay}"].T
        matrix = utility.correlation(scores, state)
        record = {"all_sources": utility.correlation_summary(matrix), "levels": {},
                  "interpretation": "noisy frozen rate projections; not posterior biology or causal directions"}
        saved[f"state_correlations_all_{assay}"] = matrix
        for level in ["moment", "confirmed"]:
            matched = sorted({m["ia"] if m["a"] == assay else m["ib"] for m in result[f"{level}_pair_matches"]
                              if assay in [m["a"], m["b"]]})
            shared = sorted({m["indices"][position] for m in result[f"{level}_all_assay_matches"]})
            record["levels"][level] = {"pair_matched_indices": matched, "all_assay_matched_indices": shared,
                                         "pair_matched": utility.correlation_summary(matrix[matched]),
                                         "all_assay_matched": utility.correlation_summary(matrix[shared])}
            saved[f"state_correlations_{level}_pair_{assay}"] = matrix[matched]
            saved[f"state_correlations_{level}_all_assay_{assay}"] = matrix[shared]
        output[assay] = record
    return output


def main(study=STUDY):
    protocol, seal, records, data, manifest = verify_seal(study)
    output = study / "evaluation_v1"
    require(not output.exists(), "Refuse to overwrite evaluation outputs")
    initial = {"evaluator": sha(Path(__file__)), "utilities": sha(UTILITY_SOURCE),
               "protocol": sha(study / "protocol.json"), "fit_seal": sha(study / "runs_v1/provenance.json")}
    # All225 fits have been verified before the first heldout evaluation read.
    public, references, input_hashes = load_data(protocol, manifest)
    output.mkdir()
    training_cache, anchor_scales, observed_cache, summary = {}, {}, {}, []
    bootstrap = protocol["evaluation"]["paired_bootstrap"]
    _, weights = bootstrap_multiplicities(595, bootstrap["replicates"], bootstrap["seed"])
    for name, result in records.items():
        target = output / name
        target.mkdir()
        report = {"arm": name, "dataset": result["dataset"], "variant": result["variant"],
                  "source_seed": result["source_seed"], "control": result["control"],
                  "fit_status": result["status"], "fit_result_sha256": seal["artifacts"][f"{name}/result.json"],
                  "causal_recovery_claim": False,
                  "cross_prediction_scope": "Pairwise moment predictions; global shared-source consistency and a PSD joint distribution are not established"}
        if result["status"] == "failed":
            report.update(evaluation_status="fit_failed", error=result.get("error"))
        else:
            cohort = public[result["dataset"]]
            expected_inputs = {f"public/{result['dataset']}/train/{a}.npz":
                manifest["datasets"][result["dataset"]]["assays"][a]["cohorts"]["train"]["sha256"] for a in protocol["assays"]}
            require(result["input_hashes"] == expected_inputs, "Fitting input hashes differ from permitted training files")
            arrays = utility.read_npz(study / "runs_v1" / name / "arrays.npz")
            metadata = validate_model(result, arrays, cohort, protocol, training_cache, anchor_scales)
            rates, within = within_scores(result, arrays, cohort, protocol["assays"])
            saved = {}
            report.update(evaluation_status="complete", within_assay=within,
                          exposure_parameters=result["exposure_parameters"], ranks={a: metadata[a]["k"] for a in protocol["assays"]},
                          moment_pair_match_count=len(result["moment_pair_matches"]),
                          confirmed_pair_match_count=len(result["confirmed_pair_matches"]),
                          moment_all_assay_match_count=len(result["moment_all_assay_matches"]),
                          confirmed_all_assay_match_count=len(result["confirmed_all_assay_matches"]),
                          graph_status=result["graph_status"])
            if result["dataset"] == "scmtr_seq_s1":
                report["real_cross_assay"] = real_scores(result, arrays, rates, references[result["dataset"]],
                    protocol["assays"], metadata, weights, observed_cache, saved)
            else:
                report["simulation_state"] = state_scores(result, arrays, cohort, rates, references[result["dataset"]],
                                                          protocol["assays"], saved)
            np.savez_compressed(target / "arrays.npz", **saved)
            report["arrays_sha256"] = sha(target / "arrays.npz")
        utility.write_json(target / "result.json", report)
        summary.append(report)
    utility.write_json(output / "summary.json", summary)
    require(initial["evaluator"] == sha(Path(__file__)) and initial["utilities"] == sha(UTILITY_SOURCE),
            "Scoring source changed during evaluation")
    require(initial["protocol"] == sha(study / "protocol.json"), "Scientific protocol changed during evaluation")
    verify_seal(study)
    utility.write_json(output / "provenance.json", {
        "status": "complete", "source_sha256": initial, "data_manifest_sha256": sha(data / "manifest.json"),
        "consumed_input_sha256": input_hashes, "fit_arms_checked": len(records),
        "training_or_selection_performed": False,
        "nuisance_validation": "Independent fixed-profile replay only; frozen exported beta used for all predictions",
        "scientific_scope": "Previously inspected development cohorts; finite-fingerprint candidates, no real causal truth",
        "bootstrap_method": "Exact complete-cell row-Gram Frobenius identities; fixed model and source-fit scales",
        "artifacts": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"status": "complete", "evaluated_arms": len(summary), "output": str(output)}))


if __name__ == "__main__":
    main()
