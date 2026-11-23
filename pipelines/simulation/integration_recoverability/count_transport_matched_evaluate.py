"""Frozen evaluation of exact-model count transport, not causal directions.

Truth and paired counts are evaluator-only. Known source dimension is an
explicit experimental condition. Optimization convergence is not a certificate
of a global optimum. Existing count_transport_demo_v1 files remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RUN = HERE / "runs" / "count_transport_matched_v1"
SPEC = importlib.util.spec_from_file_location("matched_transport_metric_helpers", HERE / "count_transport_evaluate.py")
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)
SIZES = (3000, 12000)
MODES = ("capped600", "convergence")
BOOTSTRAPS = 1000
OUTPUTS = ("evaluation.json", "comparison.tsv", "aggregate.tsv", "mode_comparison.tsv", "evaluation_seal.json")
DEFAULT_THRESHOLDS = {"covariance_relative_error": .10, "transport_relative_error": .10,
                      "oracle_conditional_gap_per_feature": .02, "positive_nominal_gain_ci95": True}


def validate_model(model: dict, assays: list[str], widths: dict) -> dict:
    """Check the stored finite strict model without imposing rank as selection."""
    weights, rates = legacy.validate_model(model, assays, widths)
    d = int(model["dimension"])
    probabilities = np.asarray(model["probabilities"], dtype=float)
    mean = np.asarray(model["source_mean"], dtype=float)
    bits = np.asarray(model["state_bits"], dtype=float)
    matrices = {a: np.asarray(model["loadings"][a], dtype=float) for a in assays}
    if (d not in (1, 2, 3) or probabilities.shape != (d,) or mean.shape != (d,)
            or not np.isfinite(mean).all() or not np.isfinite(probabilities).all()
            or (probabilities <= 0).any() or (probabilities >= .5).any()
            or bits.shape != (2 ** d, d) or len(weights) != 2 ** d
            or not np.isin(bits, (0, 1)).all()
            or len({tuple(row) for row in bits}) != 2 ** d):
        raise ValueError("Invalid standardized independent-source model")
    expected_weights = np.prod(np.where(bits == 1, probabilities, 1 - probabilities), axis=1)
    if not np.allclose(weights, expected_weights, atol=1e-10, rtol=1e-8):
        raise ValueError("Finite weights do not equal independent-source probabilities")
    states = (bits - probabilities) / np.sqrt(probabilities * (1 - probabilities)) + mean
    eta = {}
    for assay in assays:
        if matrices[assay].shape != (widths[assay], d) or not np.isfinite(matrices[assay]).all():
            raise ValueError("Invalid loading dimensions or values")
        if (rates[assay] <= 0).any():
            raise ValueError("Strict log-rate model requires positive rates")
        eta[assay] = np.log(rates[assay])
        if not np.allclose(eta[assay], states @ matrices[assay].T, atol=1e-8, rtol=1e-8):
            raise ValueError("Rates do not satisfy the declared strict log-linear source model")
        if "state_log_rates" in model and not np.allclose(
                eta[assay], model["state_log_rates"][assay], atol=1e-8, rtol=1e-8):
            raise ValueError("Stored rates and log rates disagree")
    return {"dimension": d, "probabilities": probabilities, "mean": mean,
            "bits": bits, "weights": weights, "rates": rates, "eta": eta, "loadings": matrices}


def relative_error(estimated: np.ndarray, reference: np.ndarray) -> dict:
    absolute = float(np.linalg.norm(estimated - reference))
    denominator = float(np.linalg.norm(reference))
    relative = absolute / denominator if denominator > 0 else (0. if absolute == 0 else None)
    return {"absolute_frobenius_error": absolute, "reference_frobenius_norm": denominator,
            "relative_frobenius_error": relative}


def transport_error(estimated_source: np.ndarray, estimated_target: np.ndarray,
                    true_source: np.ndarray, true_target: np.ndarray) -> tuple[dict, np.ndarray]:
    """Evaluate learned transport only on the true source column space."""
    operator = estimated_target @ np.linalg.pinv(estimated_source)
    error = relative_error(operator @ true_source, true_target)
    return error, operator


def state_alignment(fitted: dict, truth: dict, assays: list[str]) -> dict:
    """One global source permutation, induced identically in every assay."""
    if fitted["dimension"] != truth["dimension"]:
        raise ValueError("Global source alignment requires the same known dimension")
    d = truth["dimension"]
    lookup = {tuple(row): i for i, row in enumerate(fitted["bits"])}
    candidates = []
    for permutation in itertools.permutations(range(d)):
        required = np.empty_like(truth["bits"])
        required[:, permutation] = truth["bits"]
        indices = np.array([lookup[tuple(row)] for row in required])
        loss = np.mean([np.sum(truth["weights"][:, None] *
                               (fitted["eta"][a][indices] - truth["eta"][a]) ** 2) /
                        truth["eta"][a].shape[1] for a in assays])
        candidates.append((float(loss), permutation, indices))
    loss, permutation, indices = min(candidates, key=lambda value: (value[0], value[1]))
    by_assay = {}
    for assay in assays:
        delta = fitted["eta"][assay][indices] - truth["eta"][assay]
        rmse = float(np.sqrt(np.sum(truth["weights"][:, None] * delta ** 2) / delta.shape[1]))
        by_assay[assay] = {
            "log_rate_atom_rmse": rmse,
            "loading_error": relative_error(fitted["loadings"][assay][:, permutation], truth["loadings"][assay]),
        }
    return {"source_permutation_true_to_fit": list(permutation),
            "induced_state_rows": indices.tolist(),
            "combined_log_rate_atom_rmse": float(np.sqrt(loss)),
            "probability_rmse": float(np.sqrt(np.mean(
                (fitted["probabilities"][list(permutation)] - truth["probabilities"]) ** 2))),
            "source_mean_rmse": float(np.sqrt(np.mean(
                (fitted["mean"][list(permutation)] - truth["mean"]) ** 2))),
            "assays": by_assay,
            "interpretation": "Secondary evaluator-only alignment: one shared permutation across all assays; never used to construct transport or predictions."}


def summarize_rows(values: np.ndarray, indices: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Nonfinite per-cell evaluation statistic")
    result = {"mean": float(values.mean())}
    if indices is not None:
        result["ci95"] = np.quantile(values[indices].mean(1), [.025, .975]).tolist()
    return result


def scaled_summary(summary: dict, width: int) -> dict:
    result = {"mean": summary["mean"] / width}
    if "ci95" in summary:
        result["ci95"] = [v / width for v in summary["ci95"]]
    return result


def predictive_rows(source: np.ndarray, target: np.ndarray, fitted: dict, truth: dict,
                    source_assay: str, target_assay: str) -> dict:
    identity = np.arange(len(source))
    args = (source, target)
    predicted = legacy.prediction_rows(*args, fitted["weights"], fitted["rates"][source_assay],
                                       fitted["rates"][target_assay], identity)
    oracle = legacy.prediction_rows(*args, truth["weights"], truth["rates"][source_assay],
                                    truth["rates"][target_assay], identity)
    return {
        "model_conditional_loglik": predicted["conditional_loglik"],
        "oracle_conditional_loglik": oracle["conditional_loglik"],
        "model_target_marginal_loglik": predicted["target_marginal_loglik"],
        "true_independent_target_loglik": oracle["target_marginal_loglik"],
        "gain_over_same_model_target_marginal": predicted["conditional_loglik"] - predicted["target_marginal_loglik"],
        "gain_over_true_independent_target": predicted["conditional_loglik"] - oracle["target_marginal_loglik"],
        "oracle_conditional_logscore_gap": oracle["conditional_loglik"] - predicted["conditional_loglik"],
        "target_marginal_logscore_gap": oracle["target_marginal_loglik"] - predicted["target_marginal_loglik"],
        "oracle_gain_over_true_independent_target": oracle["conditional_loglik"] - oracle["target_marginal_loglik"],
    }


def optimization_oracle_check(model: dict | None, true_objective: float | None) -> dict:
    if model is None or true_objective is None:
        return {"available": False, "true_feasible_training_objective": true_objective}
    fitted_objective = float(model["objective"])
    if not np.isfinite(fitted_objective) or not np.isfinite(true_objective):
        raise ValueError("Nonfinite training objective in oracle check")
    difference = fitted_objective - true_objective
    return {"available": True, "fitted_training_objective": fitted_objective,
            "true_feasible_training_objective": float(true_objective),
            "fitted_minus_true_training_objective": float(difference),
            "known_feasible_truth_better": bool(difference > 1e-8),
            "numerical_comparison_tolerance": 1e-8,
            "interpretation": ("The fitted objective exceeds a known feasible objective on the same training counts: a suboptimality certificate, regardless of local convergence."
                               if difference > 1e-8 else
                               "No suboptimality certificate from this comparison. A nonpositive difference does not certify a global optimum."),
            "information_boundary": "Generating parameters used only after fit seals; no initialization, fitting or reselection."}


def evaluate_fit(artifact: dict, counts: dict, true_model: dict, assays: list[str],
                 widths: dict, indices: np.ndarray | None = None,
                 thresholds: dict | None = None,
                 true_training_objective: float | None = None) -> tuple[dict, dict]:
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    output = {k: artifact[k] for k in ("world", "n_fit", "optimizer_mode", "dimension", "seed")}
    output.update({"calibration_score": artifact.get("calibration_score"),
                   "acceptance": artifact.get("acceptance", {"pass": False}),
                   "fit_accepted": False, "recovery_pass": False, "pairs": {},
                   "optimization_oracle_check": optimization_oracle_check(artifact.get("model"), true_training_objective)})
    if artifact.get("model") is None:
        output.update(evaluation_status="fit_failed", error=artifact.get("error"), gradient_converged=False)
        return output, {}
    fitted = validate_model(artifact["model"], assays, widths)
    truth = validate_model(true_model, assays, widths)
    if fitted["dimension"] != truth["dimension"] or fitted["dimension"] != artifact["dimension"]:
        raise ValueError("Fitted dimension differs from the preregistered known true dimension")
    gradient = artifact["model"].get("projected_gradient_inf_norm")
    tolerance = artifact["model"].get("convergence_tolerance")
    gradient_converged = bool(artifact["model"].get("converged") is True
                              and gradient is not None and tolerance is not None
                              and np.isfinite(gradient) and np.isfinite(tolerance)
                              and tolerance > 0 and 0 <= gradient <= tolerance)
    accepted = artifact["acceptance"]["pass"] is True and gradient_converged
    output.update(evaluation_status="scored", fit_accepted=accepted,
                  gradient_converged=gradient_converged,
                  projected_gradient_inf_norm=artifact["model"].get("projected_gradient_inf_norm"),
                  convergence_tolerance=artifact["model"].get("convergence_tolerance"),
                  stop_reason=artifact["model"].get("stop_reason"),
                  iterations=artifact["model"].get("iterations"),
                  global_source_alignment=state_alignment(fitted, truth, assays))
    row_scores = {}
    for source, target in itertools.permutations(assays, 2):
        key = f"{source}__to__{target}"
        true_covariance = legacy.cross_moments(truth["weights"], truth["rates"][source], truth["rates"][target])[0]
        estimated_covariance = legacy.cross_moments(fitted["weights"], fitted["rates"][source], fitted["rates"][target])[0]
        covariance = relative_error(estimated_covariance, true_covariance)
        transport, operator = transport_error(fitted["loadings"][source], fitted["loadings"][target],
                                               truth["loadings"][source], truth["loadings"][target])
        support_delta = truth["eta"][source] @ operator.T - truth["eta"][target]
        mean_transport = relative_error((truth["weights"] @ truth["eta"][source]) @ operator.T,
                                         truth["weights"] @ truth["eta"][target])
        rows = predictive_rows(counts[source], counts[target], fitted, truth, source, target)
        row_scores[key] = rows
        interval_names = {"gain_over_same_model_target_marginal", "gain_over_true_independent_target",
                          "oracle_conditional_logscore_gap", "target_marginal_logscore_gap",
                          "oracle_gain_over_true_independent_target"}
        statistics = {name: summarize_rows(values, indices if name in interval_names else None)
                      for name, values in rows.items()}
        statistics_per_feature = {name: scaled_summary(summary, widths[target]) for name, summary in statistics.items()}
        covariance_ok = covariance["relative_frobenius_error"] is not None and covariance["relative_frobenius_error"] <= thresholds["covariance_relative_error"]
        transport_ok = transport["relative_frobenius_error"] is not None and transport["relative_frobenius_error"] <= thresholds["transport_relative_error"]
        gain = statistics["gain_over_same_model_target_marginal"]
        criteria = {"fit_numerically_accepted": accepted,
                    "exact_covariance_relative_error_pass": bool(covariance_ok),
                    "restricted_transport_relative_error_pass": bool(transport_ok),
                    "positive_gain_ci95_lower": None if indices is None else bool(gain["ci95"][0] > 0),
                    "oracle_gap_per_feature_pass": bool(statistics_per_feature["oracle_conditional_logscore_gap"]["mean"] <= thresholds["oracle_conditional_gap_per_feature"])}
        if not thresholds["positive_nominal_gain_ci95"]:
            criteria["positive_gain_ci95_lower"] = True
        output["pairs"][key] = {
            "exact_count_covariance_error": covariance,
            "restricted_lograte_transport_error": transport,
            "true_support_lograte_transport_rmse": float(np.sqrt(
                np.sum(truth["weights"][:, None] * support_delta ** 2) / widths[target])),
            "mean_lograte_transport_error": mean_transport,
            "mean_lograte_transport_definition": "T_hat E_true[log R_source] versus E_true[log R_target]",
            "per_cell": statistics, "per_target_feature": statistics_per_feature,
            "target_features": widths[target], "criteria": criteria,
            "recovery_pass": None if indices is None else bool(all(criteria.values())),
        }
    output["recovery_pass"] = None if indices is None else bool(accepted and all(
        item["recovery_pass"] for item in output["pairs"].values()))
    return output, row_scores


def verify_frozen(run_dir: Path, repo_root: Path) -> tuple[dict, dict, dict, dict, dict]:
    protocol_path, fit_path = run_dir / "protocol.json", run_dir / "fit_seal.json"
    protocol, fit_seal = json.loads(protocol_path.read_text()), json.loads(fit_path.read_text())
    legacy.require_hash(protocol_path, fit_seal["protocol_sha256"])
    if fit_seal.get("evaluator_read", False) is not False:
        raise ValueError("Fit seal records prior evaluator access")
    for path, digest in protocol["sources"].items():
        legacy.require_hash(legacy.within(repo_root, path), digest)
    generation_path = run_dir / "generation_seal.json"
    legacy.require_hash(generation_path, fit_seal["generation_seal_sha256"])
    generation = json.loads(generation_path.read_text())
    if generation["protocol_sha256"] != fit_seal["protocol_sha256"]:
        raise ValueError("Generation and fit protocols differ")
    manifest_path = run_dir / "data_manifest.json"
    legacy.require_hash(manifest_path, fit_seal["data_manifest_sha256"])
    if generation["data_manifest_sha256"] != fit_seal["data_manifest_sha256"]:
        raise ValueError("Generation and fit data manifests differ")
    manifest = json.loads(manifest_path.read_text())
    if tuple(protocol["fit_sizes"]) != SIZES or tuple(protocol["optimizer_modes"]) != MODES:
        raise ValueError("Matched evaluator design differs from frozen fit sizes or modes")
    if set(protocol["recovery_thresholds"]) != set(DEFAULT_THRESHOLDS):
        raise ValueError("Incomplete recovery threshold specification")
    if set(manifest["worlds"]) != set(protocol["worlds"]):
        raise ValueError("Frozen world inventory differs")
    expected_inputs = {}
    for world in protocol["worlds"]:
        spec = manifest["worlds"][world]
        entries = list(spec["public"].values()) + [spec["assessment"], spec["truth"]]
        if set(spec["public"]) != set(protocol["assays"]):
            raise ValueError("Frozen assay inventory differs")
        for entry in entries:
            if entry["path"] in expected_inputs:
                raise ValueError("Distinct generated inputs alias the same path")
            expected_inputs[entry["path"]] = entry["sha256"]
    if "symmetry_control.json" not in generation["artifacts"]:
        raise ValueError("Generation seal omits the analytic symmetry control")
    expected_inputs["symmetry_control.json"] = generation["artifacts"]["symmetry_control.json"]
    if generation["artifacts"] != expected_inputs:
        raise ValueError("Generation seal does not exactly match the full input inventory")
    # All public, truth and assessment bytes are checked before interpretation
    # of any heldout values or true model parameters.
    for path, digest in expected_inputs.items():
        legacy.require_hash(legacy.within(run_dir, path), digest)
    artifacts = {}
    for name, digest in fit_seal["artifacts"].items():
        path = legacy.within(run_dir, name)
        legacy.require_hash(path, digest)
        artifacts[name] = json.loads(path.read_text())
    expected_groups = {f"{world}__n{size}__{mode}" for world in protocol["worlds"] for size in SIZES for mode in MODES}
    if (set(fit_seal["selected"]) != expected_groups
            or set(fit_seal["selected_dimension_restarts"]) != expected_groups
            or len(artifacts) != len(expected_groups) * 4):
        raise ValueError("Incomplete or unexpected fit selection inventory")
    all_restart_names = []
    for group, selected in fit_seal["selected"].items():
        names = fit_seal["selected_dimension_restarts"][group]
        if len(names) != 4 or len(set(names)) != 4 or selected not in names:
            raise ValueError("Every selected group requires all four original starts")
        all_restart_names.extend(names)
        seeds = []
        for name in names:
            if name not in artifacts:
                raise ValueError("Unsealed restart artifact")
            artifact = artifacts[name]
            actual = f"{artifact['world']}__n{artifact['n_fit']}__{artifact['optimizer_mode']}"
            if actual != group or artifact["dimension"] != manifest["worlds"][artifact["world"]]["dimension"]:
                raise ValueError("Fit identity or known dimension differs")
            seeds.append(artifact["seed"])
            if artifact.get("model") is not None:
                validate_model(artifact["model"], protocol["assays"], protocol["widths"])
        if len(set(seeds)) != 4 or artifacts[selected].get("model") is None:
            raise ValueError("Invalid restart seeds or missing selected fitted model")
    if len(set(all_restart_names)) != len(artifacts) or set(all_restart_names) != set(artifacts):
        raise ValueError("Restart groups duplicate or omit sealed artifacts")
    return protocol, fit_seal, manifest, artifacts, expected_inputs


def load_evaluation_inputs(run_dir: Path, protocol: dict, manifest: dict) -> tuple[dict, dict, dict]:
    counts, truths, audits = {}, {}, {}
    for world in protocol["worlds"]:
        spec = manifest["worlds"][world]
        used_ids, features = set(), {}
        for assay in protocol["assays"]:
            with np.load(legacy.within(run_dir, spec["public"][assay]["path"]), allow_pickle=False) as data:
                fit_ids, calibration_ids = data["fit_ids"].astype(str), data["calibration_ids"].astype(str)
                if fit_ids.shape != (max(protocol["fit_sizes"]),) or calibration_ids.shape != (protocol["calibration_rows"],):
                    raise ValueError("Public sample size differs from frozen design")
                ids = np.concatenate([fit_ids, calibration_ids])
                if len(set(ids)) != len(ids) or used_ids.intersection(ids):
                    raise ValueError("Public fit/calibration assays are not disjoint")
                used_ids.update(ids)
                features[assay] = data["feature_ids"].astype(str).tolist()
                if len(features[assay]) != protocol["widths"][assay] or len(set(features[assay])) != len(features[assay]):
                    raise ValueError("Invalid public feature panel")
        with np.load(legacy.within(run_dir, spec["assessment"]["path"]), allow_pickle=False) as data:
            ids = data["cell_ids"].astype(str)
            if ids.shape != (protocol["assessment_rows"],) or len(set(ids)) != len(ids) or used_ids.intersection(ids):
                raise ValueError("Assessment identities overlap training or calibration")
            counts[world] = {}
            for assay in protocol["assays"]:
                if data[f"feature_ids_{assay}"].astype(str).tolist() != features[assay]:
                    raise ValueError("Paired assessment feature order differs")
                counts[world][assay] = legacy.count_array(data[f"counts_{assay}"])
                if counts[world][assay].shape != (len(ids), protocol["widths"][assay]):
                    raise ValueError("Invalid paired assessment shape")
        truths[world] = json.loads(legacy.within(run_dir, spec["truth"]["path"]).read_text())
        parsed_truth = validate_model(truths[world], protocol["assays"], protocol["widths"])
        if parsed_truth["dimension"] != spec["dimension"]:
            raise ValueError("True model dimension differs from known-d protocol")
        audits[world] = {"paired_rows": len(ids), "public_rows_disjoint_from_assessment": True,
                         "assays_disjoint_in_public_data": True, "feature_order_verified": True,
                         "known_dimension": parsed_truth["dimension"]}
    return counts, truths, audits


def oracle_training_objectives(run_dir: Path, protocol: dict, manifest: dict, truths: dict) -> dict:
    """Evaluate the known feasible generator once per public training prefix."""
    losses = {}
    for world in protocol["worlds"]:
        truth = validate_model(truths[world], protocol["assays"], protocol["widths"])
        losses[world] = {size: 0. for size in protocol["fit_sizes"]}
        for assay in protocol["assays"]:
            entry = manifest["worlds"][world]["public"][assay]
            with np.load(legacy.within(run_dir, entry["path"]), allow_pickle=False) as data:
                counts = legacy.count_array(data["fit_counts"])
            if counts.shape != (max(protocol["fit_sizes"]), protocol["widths"][assay]):
                raise ValueError("Oracle objective training prefix shape differs")
            row_loglik = logsumexp(legacy.poisson_components(counts, truth["rates"][assay]) +
                                   np.log(truth["weights"])[None, :], axis=1)
            for size in protocol["fit_sizes"]:
                losses[world][size] -= float(row_loglik[:size].mean() / protocol["widths"][assay] / len(protocol["assays"]))
    return losses


def restart_summary(reports: dict, names: list[str]) -> dict:
    scored = [reports[name] for name in names if reports[name]["evaluation_status"] == "scored"]
    def span(values):
        return [min(values), max(values)] if values else None
    pairs = scored[0]["pairs"] if scored else {}
    return {"fits": len(names), "scored_fits": len(scored), "failed_fits": len(names) - len(scored),
            "gradient_converged_fits": sum(r["gradient_converged"] for r in scored),
            "accepted_fits": sum(r["fit_accepted"] for r in scored),
            "interpretation": "All four frozen starts; descriptive ranges, no truth-based reselection.",
            "pairs": {pair: {
                "covariance_relative_error_range": span([r["pairs"][pair]["exact_count_covariance_error"]["relative_frobenius_error"] for r in scored
                                                         if r["pairs"][pair]["exact_count_covariance_error"]["relative_frobenius_error"] is not None]),
                "restricted_transport_relative_error_range": span([r["pairs"][pair]["restricted_lograte_transport_error"]["relative_frobenius_error"] for r in scored
                                                                    if r["pairs"][pair]["restricted_lograte_transport_error"]["relative_frobenius_error"] is not None]),
                "oracle_gap_per_feature_range": span([r["pairs"][pair]["per_target_feature"]["oracle_conditional_logscore_gap"]["mean"] for r in scored]),
            } for pair in pairs}}


def fit_summary(report: dict) -> dict:
    row = {key: report[key] for key in ("world", "dimension", "n_fit", "optimizer_mode", "seed", "fit_accepted", "gradient_converged", "recovery_pass")}
    pairs = list(report["pairs"].values())
    def extrema(field, op):
        values = [item[field]["relative_frobenius_error"] for item in pairs]
        return op(values) if values and all(v is not None for v in values) else None
    row.update(max_covariance_relative_error=extrema("exact_count_covariance_error", max),
               max_restricted_transport_relative_error=extrema("restricted_lograte_transport_error", max),
               min_gain_per_feature=min((p["per_target_feature"]["gain_over_same_model_target_marginal"]["mean"] for p in pairs), default=None),
               max_oracle_gap_per_feature=max((p["per_target_feature"]["oracle_conditional_logscore_gap"]["mean"] for p in pairs), default=None),
               recovered_ordered_pairs=sum(p["recovery_pass"] is True for p in pairs),
               fitted_minus_true_training_objective=report["optimization_oracle_check"].get("fitted_minus_true_training_objective"),
               known_feasible_truth_better=report["optimization_oracle_check"].get("known_feasible_truth_better"),
               iterations=report.get("iterations"), stop_reason=report.get("stop_reason"))
    return row


def aggregate_rows(rows: list[dict]) -> list[dict]:
    result = []
    for dimension, size, mode in sorted({(r["dimension"], r["n_fit"], r["optimizer_mode"]) for r in rows}):
        selected = [r for r in rows if (r["dimension"], r["n_fit"], r["optimizer_mode"]) == (dimension, size, mode)]
        row = {"dimension": dimension, "n_fit": size, "optimizer_mode": mode, "worlds": len(selected),
               "gradient_converged": sum(r["gradient_converged"] for r in selected),
               "numerically_accepted": sum(r["fit_accepted"] for r in selected),
               "known_feasible_truth_better": sum(r["known_feasible_truth_better"] is True for r in selected),
               "all_pair_recovery_pass": sum(r["recovery_pass"] is True for r in selected)}
        for field in ("max_covariance_relative_error", "max_restricted_transport_relative_error", "min_gain_per_feature", "max_oracle_gap_per_feature"):
            values = [r[field] for r in selected if r[field] is not None]
            row[f"mean_{field}"] = float(np.mean(values)) if values else None
        result.append(row)
    return result


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(run_dir: Path, repo_root: Path = REPO) -> dict:
    run_dir, repo_root = Path(run_dir).resolve(), Path(repo_root).resolve()
    if any((run_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("Matched evaluation outputs already exist; cannot overwrite")
    protocol, seal, manifest, artifacts, input_hashes = verify_frozen(run_dir, repo_root)
    counts, truths, audits = load_evaluation_inputs(run_dir, protocol, manifest)
    true_training_objectives = oracle_training_objectives(run_dir, protocol, manifest, truths)
    bootstrap = {world: legacy.bootstrap_indices(len(next(iter(block.values()))), f"matched:{world}", BOOTSTRAPS)
                 for world, block in counts.items()}
    output = {
        "target": "Exact-model count-law and log-rate transport with known source dimension; not causal direction or global optimization recovery",
        "direction_interpretation": "Six ordered prediction problems per world contain only three unique per-cell gains over the same fitted target marginal; these gains are symmetric associations, not causal arrows.",
        "uncertainty": "1000 common paired-row percentile bootstrap resamples per world for selected fixed fits and mode comparisons only. Descriptive 95% intervals; no familywise coverage or fit uncertainty claim. Exact covariance and transport errors use the frozen true model and have no sampling interval.",
        "criteria_interpretation": "Descriptive preregistered tolerances are reported individually; passing them does not establish asymptotic consistency or success on realistic data. Oracle likelihood gaps use heldout Monte Carlo averages and can be slightly negative.",
        "recovery_thresholds": protocol["recovery_thresholds"],
        "analytic_symmetry_control": json.loads((run_dir / "symmetry_control.json").read_text()),
        "input_audits": audits, "selected": seal["selected"], "fits": {}, "restart_stability": {}, "mode_comparisons": {},
    }
    selected_rows = {}
    for group, selected in seal["selected"].items():
        names = seal["selected_dimension_restarts"][group]
        for name in names:
            artifact = artifacts[name]
            world = artifact["world"]
            report, rows = evaluate_fit(artifact, counts[world], truths[world], protocol["assays"], protocol["widths"],
                                        bootstrap[world] if name == selected else None, protocol["recovery_thresholds"],
                                        true_training_objectives[world][artifact["n_fit"]])
            output["fits"][name] = report
            if name == selected:
                selected_rows[group] = rows
        output["restart_stability"][group] = restart_summary(output["fits"], names)
    comparisons = []
    for world in protocol["worlds"]:
        for size in SIZES:
            capped_key, converged_key = (f"{world}__n{size}__{mode}" for mode in MODES)
            capped = output["fits"][seal["selected"][capped_key]]
            converged = output["fits"][seal["selected"][converged_key]]
            pairs = {}
            for pair in capped["pairs"]:
                width = capped["pairs"][pair]["target_features"]
                first, second = selected_rows[capped_key][pair], selected_rows[converged_key][pair]
                pairs[pair] = {
                    "conditional_logscore_improvement_per_target_feature": summarize_rows(
                        (second["model_conditional_loglik"] - first["model_conditional_loglik"]) / width, bootstrap[world]),
                    "same_model_marginal_gain_improvement_per_target_feature": summarize_rows(
                        (second["gain_over_same_model_target_marginal"] - first["gain_over_same_model_target_marginal"]) / width, bootstrap[world]),
                    "covariance_relative_error_reduction": capped["pairs"][pair]["exact_count_covariance_error"]["relative_frobenius_error"] -
                        converged["pairs"][pair]["exact_count_covariance_error"]["relative_frobenius_error"],
                    "restricted_transport_relative_error_reduction": capped["pairs"][pair]["restricted_lograte_transport_error"]["relative_frobenius_error"] -
                        converged["pairs"][pair]["restricted_lograte_transport_error"]["relative_frobenius_error"],
                }
            key = f"{world}__n{size}"
            output["mode_comparisons"][key] = {
                "capped_selected": seal["selected"][capped_key], "convergence_selected": seal["selected"][converged_key],
                "note": "Each mode independently selected by unpaired calibration; this compares the frozen selection procedures, not necessarily the same restart.",
                "pairs": pairs}
            first_summary, second_summary = fit_summary(capped), fit_summary(converged)
            comparisons.append({"world": world, "dimension": capped["dimension"], "n_fit": size,
                                "capped_accepted": capped["fit_accepted"], "convergence_accepted": converged["fit_accepted"],
                                "capped_recovery_pass": capped["recovery_pass"], "convergence_recovery_pass": converged["recovery_pass"],
                                "max_covariance_error_reduction": first_summary["max_covariance_relative_error"] - second_summary["max_covariance_relative_error"],
                                "max_transport_error_reduction": first_summary["max_restricted_transport_relative_error"] - second_summary["max_restricted_transport_relative_error"],
                                "mean_conditional_logscore_improvement_per_feature": float(np.mean([
                                    value["conditional_logscore_improvement_per_target_feature"]["mean"] for value in pairs.values()]))})
    rows = [fit_summary(output["fits"][name]) for _, name in sorted(seal["selected"].items())]
    output["aggregate"] = aggregate_rows(rows)
    legacy.write_exclusive(run_dir / "evaluation.json", output)
    write_table(run_dir / "comparison.tsv", rows)
    write_table(run_dir / "aggregate.tsv", output["aggregate"])
    write_table(run_dir / "mode_comparison.tsv", comparisons)
    legacy.write_exclusive(run_dir / "evaluation_seal.json", {
        "protocol_sha256": legacy.sha(run_dir / "protocol.json"),
        "fit_seal_sha256": legacy.sha(run_dir / "fit_seal.json"),
        "generation_seal_sha256": legacy.sha(run_dir / "generation_seal.json"),
        "data_manifest_sha256": legacy.sha(run_dir / "data_manifest.json"),
        "inputs": input_hashes,
        "outputs": {name: legacy.sha(run_dir / name) for name in OUTPUTS if name != "evaluation_seal.json"},
        "truth_used_only_for_evaluation": True, "no_fitting_or_reselection": True,
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    output = evaluate_run(args.run_dir)
    print(json.dumps({"evaluation": str(args.run_dir / "evaluation.json"),
                      "selected_fits": len(output["selected"]), "scored_artifacts": len(output["fits"])}))


if __name__ == "__main__":
    main()
