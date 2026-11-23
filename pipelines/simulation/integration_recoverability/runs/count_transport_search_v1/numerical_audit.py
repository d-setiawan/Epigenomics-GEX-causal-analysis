"""Independent post-seal numerical replay; never fit, initialize, or resample a world.

Adapts the previous matched/label audit algebra into a standalone artifact.
It does not import the optimizer, runner, or evaluator. Truth and paired counts
remain inaccessible until both experiment seals and every input hash pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import expit, logit, logsumexp
from scipy.stats import poisson
from threadpoolctl import threadpool_limits

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[4]
HERE = RUN.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.issues, self.counts, self.errors = [], Counter(), defaultdict(float)

    def check(self, condition, label):
        self.counts["boolean_checks"] += 1
        if not condition:
            self.issues.append(label)

    def pin(self, path, digest):
        self.counts["hash_comparisons"] += 1
        self.check(sha(path) == digest, "Hash mismatch: " + str(path))

    def compare(self, actual, expected, category, label, atol=1e-9, rtol=1e-9):
        self.counts["numeric_comparisons"] += 1
        if actual is None or expected is None:
            self.check(actual is expected, label + " null mismatch")
            return
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape:
            self.issues.append(label + " shape mismatch")
            return
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.errors[category] = max(self.errors[category], error)
        self.check(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=atol, rtol=rtol), label)


def component(counts, rates):
    # Independent SciPy probability implementation, not the fitted objective.
    return poisson.logpmf(counts[:, None, :], rates[None, :, :]).sum(2)


def loglik(counts, weights, rates):
    return logsumexp(component(counts, rates) + np.log(weights), axis=1)


def parse(model, assays, widths, audit, label):
    p, mu = np.asarray(model["probabilities"]), np.asarray(model["source_mean"])
    bits, weights = np.asarray(model["state_bits"]), np.asarray(model["state_weights"])
    d = model["dimension"]
    audit.check(p.shape == (d,) and mu.shape == (d,) and 1 <= d <= 3, label + " source dimension")
    audit.check(bits.shape == (2 ** d, d) and len({tuple(row) for row in bits}) == 2 ** d
                and np.isin(bits, (0, 1)).all(), label + " source support")
    audit.check(np.all(p > 0) and np.all(p < .5), label + " canonical probabilities")
    independent_weights = np.prod(np.where(bits, p, 1-p), axis=1)
    audit.compare(weights, independent_weights, "independent_source_weights", label + " source law")
    scale = np.sqrt(p*(1-p))
    state = (bits-p)/scale + mu
    matrices, rates, eta = {}, {}, {}
    for assay in assays:
        matrices[assay] = np.asarray(model["loadings"][assay])
        rates[assay] = np.asarray(model["state_rates"][assay])
        eta[assay] = np.log(rates[assay])
        audit.check(matrices[assay].shape == (widths[assay], d), label + "/" + assay + " loading shape")
        audit.compare(eta[assay], state @ matrices[assay].T, "strict_model_identity", label + "/" + assay + " strict model")
    return {"p": p, "mu": mu, "bits": bits, "weights": weights, "matrices": matrices, "rates": rates, "eta": eta, "dimension": d}


def decode_vertex(theta, assays, widths, d):
    theta = np.asarray(theta, dtype=float)
    p, h = expit(theta[:d]), theta[d:2*d]
    bits = np.array(list(itertools.product((0., 1.), repeat=d)))
    weights = np.prod(np.where(bits, p, 1-p), axis=1)
    cursor, matrices, eta = 2*d, {}, {}
    for assay in assays:
        matrices[assay] = theta[cursor:cursor+widths[assay]*d].reshape(widths[assay], d)
        cursor += widths[assay]*d
        eta[assay] = (bits+h) @ matrices[assay].T
    if cursor != len(theta):
        raise ValueError("Vertex parameter vector length differs")
    return {"p": p, "h": h, "bits": bits, "weights": weights, "matrices": matrices,
            "eta": eta, "rates": {a: np.exp(v) for a, v in eta.items()}}


def covariance(model, source, target):
    w, left, right = model["weights"], model["rates"][source], model["rates"][target]
    return left.T @ (w[:, None]*right) - np.outer(w @ left, w @ right)


def relative(estimate, reference):
    error, norm = float(np.linalg.norm(estimate-reference)), float(np.linalg.norm(reference))
    return {"absolute_frobenius_error": error, "reference_frobenius_norm": norm,
            "relative_frobenius_error": error/norm if norm else 0. if not error else None}


def compare_error(audit, actual, stored, category, label):
    for key, value in actual.items():
        audit.compare(value, stored[key], category, label + "/" + key)


def torch_gradient(theta, data, assays, widths, d):
    import torch
    torch.set_num_threads(1)
    point = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    p, h = torch.sigmoid(point[:d]), point[d:2*d]
    bits = torch.tensor(list(itertools.product((0., 1.), repeat=d)), dtype=torch.float64)
    log_weights = (bits*torch.log(p)+(1-bits)*torch.log1p(-p)).sum(1)
    loss, cursor = torch.tensor(0., dtype=torch.float64), 2*d
    for assay in assays:
        matrix = point[cursor:cursor+widths[assay]*d].reshape(widths[assay], d)
        cursor += widths[assay]*d
        x = torch.tensor(data[assay], dtype=torch.float64)
        eta = (bits+h) @ matrix.T
        joint = x @ eta.T - torch.exp(eta).sum(1) + log_weights - torch.lgamma(x+1).sum(1)[:, None]
        loss -= torch.logsumexp(joint, 1).mean()/widths[assay]/len(assays)
    loss.backward()
    return float(loss.detach()), point.grad.detach().numpy()


def global_state_permutation(raw_p, canonical_bits, raw_bits):
    flips = raw_p > .5
    order = np.argsort(np.where(flips, 1-raw_p, raw_p), kind="stable")
    wanted = np.empty_like(canonical_bits)
    wanted[:, order] = canonical_bits
    wanted[:, flips] = 1-wanted[:, flips]
    lookup = {tuple(row): i for i, row in enumerate(raw_bits)}
    return flips, order, np.array([lookup[tuple(row)] for row in wanted])


def self_checks():
    """Fixed arithmetic fixtures only; no world generation or experiment reads."""
    assays, widths, d = ["a", "b"], {"a": 3, "b": 4}, 2
    p, h = np.array([.73, .22]), np.array([.31, -.45])
    matrices = {"a": np.array([[.4, -.2], [.1, .6], [-.3, .2]]),
                "b": np.array([[.2, .5], [-.4, .1], [.5, -.1], [.2, .2]])}
    theta = np.concatenate([logit(p), h, *[matrices[a].ravel() for a in assays]])
    data = {"a": np.array([[0., 2., 1.], [1., 0., 4.], [2., 1., 0.]]),
            "b": np.array([[1., 0., 2., 1.], [3., 1., 0., 2.], [0., 2., 1., 1.]])}
    def objective(point):
        model = decode_vertex(point, assays, widths, d)
        return -float(np.mean([loglik(data[a], model["weights"], model["rates"][a]).mean()/widths[a]
                              for a in assays]))
    loss, gradient = torch_gradient(theta, data, assays, widths, d)
    np.testing.assert_allclose(loss, objective(theta), atol=1e-12, rtol=1e-12)
    finite_difference = np.empty_like(theta)
    for j in range(len(theta)):
        displacement = np.zeros_like(theta)
        displacement[j] = 1e-6
        finite_difference[j] = (objective(theta+displacement)-objective(theta-displacement))/2e-6
    np.testing.assert_allclose(gradient, finite_difference, atol=5e-7, rtol=5e-6)
    raw = decode_vertex(theta, assays, widths, d)
    flips, order, permutation = global_state_permutation(raw["p"], raw["bits"], raw["bits"])
    np.testing.assert_array_equal(permutation, [2, 0, 3, 1])
    signs = np.where(flips, -1., 1.)
    canonical_theta = np.concatenate([np.where(flips, -theta[:d], theta[:d])[order],
                                      np.where(flips, -h-1, h)[order],
                                      *[(matrices[a]*signs)[:, order].ravel() for a in assays]])
    canonical = decode_vertex(canonical_theta, assays, widths, d)
    np.testing.assert_allclose(raw["weights"][permutation], canonical["weights"], atol=1e-14)
    for assay in assays:
        np.testing.assert_allclose(raw["eta"][assay][permutation], canonical["eta"][assay], atol=1e-14)
    np.testing.assert_allclose(covariance(raw, "a", "b"), covariance(canonical, "a", "b"), atol=1e-14)
    np.testing.assert_allclose(objective(canonical_theta), loss, atol=1e-12)
    with tempfile.TemporaryDirectory() as directory:
        try:
            preflight(Path(directory), Audit())
        except RuntimeError as error:
            if "locked" not in str(error):
                raise
        else:
            raise AssertionError("Missing-seal guard did not stop the audit")
    return {"status": "pass", "fixture_gradient_maximum_absolute_difference": float(np.max(np.abs(gradient-finite_difference))),
            "checks": ["Torch float64 objective versus independent SciPy probabilities",
                       "Every raw vertex gradient coordinate versus centered finite differences",
                       "Nontrivial global complement plus axis permutation against fixed row mapping",
                       "Whole-joint rates, weights, covariance and objective invariant under canonicalization",
                       "Missing seals prevent all experiment access"]}


def preflight(run, audit):
    if not all((run/name).is_file() for name in ("fit_seal.json", "evaluation_seal.json")):
        raise RuntimeError("Audit locked until both fit and evaluation seals exist")
    protocol, fitseal, evalseal = [read(run/name) for name in ("protocol.json", "fit_seal.json", "evaluation_seal.json")]
    primary = REPO/protocol["primary_run"]
    audit.pin(run/"protocol.json", fitseal["protocol_sha256"])
    audit.pin(run/"protocol.json", evalseal["protocol_sha256"])
    audit.pin(run/"fit_seal.json", evalseal["fit_seal_sha256"])
    for filename, key in (("protocol.json", "primary_protocol_sha256"),
                          ("generation_seal.json", "primary_generation_seal_sha256"),
                          ("data_manifest.json", "primary_data_manifest_sha256"),
                          ("evaluation_seal.json", "primary_evaluation_seal_sha256")):
        audit.pin(primary/filename, protocol[key])
    for name, digest in protocol["sources"].items():
        audit.pin(REPO/name, digest)
    generation = read(primary/"generation_seal.json")
    audit.check(generation["protocol_sha256"] == protocol["primary_protocol_sha256"], "Primary generating protocol")
    audit.check(generation["data_manifest_sha256"] == protocol["primary_data_manifest_sha256"], "Primary data manifest")
    audit.check(evalseal["primary_inputs"] == generation["artifacts"], "Complete unchanged primary input inventory")
    for name, digest in generation["artifacts"].items():
        audit.pin(primary/name, digest)
    for name, digest in fitseal["artifacts"].items():
        audit.pin(run/name, digest)
    for name, digest in evalseal["outputs"].items():
        audit.pin(run/name, digest)
    audit.check(protocol["exploratory"] and fitseal["exploratory"] and evalseal["exploratory"], "Exploratory declarations")
    audit.check(fitseal["evaluator_read"] is False and evalseal["truth_used_only_after_fit_seal"] and evalseal["no_reselection"], "Information-boundary declarations")
    if audit.issues:
        raise RuntimeError("Hash or boundary audit failed before data access: " + "; ".join(audit.issues))
    return protocol, fitseal, evalseal, primary


def main(run):
    output_path = run/"NUMERICAL_REVIEW.json"
    if output_path.exists():
        raise FileExistsError("Numerical review already exists; preserve the completed audit")
    audit = Audit()
    protocol, seal, evalseal, primary = preflight(run, audit)
    print(json.dumps({"audit_stage": "all_seals_and_input_hashes_verified"}), flush=True)
    evaluation = read(run/"evaluation.json")
    manifest = read(primary/"data_manifest.json")
    artifacts = {name: read(run/name) for name in seal["artifacts"]}
    assays, widths = protocol["assays"], protocol["widths"]
    expected = set(itertools.product(protocol["worlds"], protocol["optimizer_modes"], protocol["seeds"]))
    actual = [(item["world"], item["optimizer_mode"], item["seed"]) for item in artifacts.values()]
    audit.check(len(actual) == len(expected) == 72 and set(actual) == expected, "All72outer outcomes retained")
    expected_groups = {f"{world}__n12000__{mode}" for world in protocol["worlds"] for mode in protocol["optimizer_modes"]}
    audit.check(set(seal["selected"]) == expected_groups and set(seal["selected_dimension_restarts"]) == expected_groups,
                "All18 world/mode selection groups retained")
    audit.check(all(item["n_fit"] == 12000 for item in artifacts.values()), "All training objectives use the declared12000cell prefix")
    audit.check(set(evaluation["fits"]) == set(artifacts) and evaluation["selected"] == seal["selected"], "Unchanged evaluated inventory and selections")
    public, assessment, truth = {}, {}, {}
    for world in protocol["worlds"]:
        entry, used, features = manifest["worlds"][world], set(), {}
        for assay in assays:
            with np.load(primary/entry["public"][assay]["path"], allow_pickle=False) as data:
                fit, cal = data["fit_counts"].astype(float), data["calibration_counts"].astype(float)
                ids = np.concatenate((data["fit_ids"].astype(str), data["calibration_ids"].astype(str)))
                features[assay] = data["feature_ids"].astype(str).tolist()
            audit.check(fit.shape == (12000, widths[assay]) and cal.shape == (2000, widths[assay]), world+"/"+assay+" public shape")
            audit.check(len(set(ids)) == len(ids) and not used.intersection(ids), world+"/"+assay+" disjoint public IDs")
            audit.check(np.isfinite(fit).all() and np.isfinite(cal).all() and (fit >= 0).all() and (cal >= 0).all()
                        and (fit % 1 == 0).all() and (cal % 1 == 0).all(), world+"/"+assay+" raw counts")
            used.update(ids)
            public[world, assay] = (fit, cal)
        with np.load(primary/entry["assessment"]["path"], allow_pickle=False) as data:
            ids = data["cell_ids"].astype(str)
            audit.check(len(ids) == 4000 and len(set(ids)) == 4000 and not used.intersection(ids), world+" assessment disjointness")
            for assay in assays:
                assessment[world, assay] = data["counts_"+assay].astype(float)
                audit.check(assessment[world, assay].shape == (4000, widths[assay]) and
                            data["feature_ids_"+assay].astype(str).tolist() == features[assay], world+"/"+assay+" paired panel")
        truth[world] = parse(read(primary/entry["truth"]["path"]), assays, widths, audit, "truth/"+world)
    scores, losses, parsed, selected_raw = {}, {}, {}, {}
    inner_counts, stops = Counter(), Counter()
    for name, item in artifacts.items():
        model, candidates = item.get("model"), item.get("candidates", [])
        diagnostics = item.get("search_diagnostics") or {}
        inner_counts["refinement_records"] += len(candidates)
        inner_counts["refinement_missing_models"] += sum(candidate.get("model") is None for candidate in candidates)
        inner_counts["mixture_records"] += sum(len(records) for records in diagnostics.get("mixtures", {}).values())
        valid = [i for i, candidate in enumerate(candidates) if candidate.get("model") is not None and np.isfinite(candidate["model"]["objective"])]
        converged = [i for i in valid if candidates[i]["model"]["converged"]]
        pool = converged or valid
        chosen = min(pool, key=lambda i: (candidates[i]["model"]["objective"], i)) if pool else None
        audit.check(diagnostics.get("selected_candidate_index") == chosen, name+" training-only inner selection")
        if model is None:
            audit.check(evaluation["fits"][name]["evaluation_status"] == "fit_failed", name+" explicit failed outcome")
            continue
        audit.check(chosen is not None and candidates[chosen]["model"] == model, name+" retained inner winner")
        candidate = candidates[chosen]
        raw = candidate["raw_optimizer"]
        selected_raw[name] = raw
        audit.check(model["assay_order"] == assays, name+" explicit parameter assay order")
        audit.check(model["statistical_parameter_bounds"] is None and not model["bound_hits"], name+" unbounded model declaration")
        fitted = parse(model, assays, widths, audit, name)
        parsed[name] = fitted
        raw_model = decode_vertex(raw["parameter_vector"], assays, widths, item["dimension"])
        canonical = decode_vertex(model["parameter_vector"], assays, widths, item["dimension"])
        audit.compare(canonical["p"], fitted["p"], "canonical_vertex_prior", name+" vertex probabilities")
        flips, order, permutation = global_state_permutation(raw_model["p"], fitted["bits"], raw_model["bits"])
        audit.check(candidate["canonicalization"]["flipped_coordinates"] == np.flatnonzero(flips).tolist()
                    and candidate["canonicalization"]["axis_order"] == order.tolist(), name+" global canonicalization metadata")
        audit.compare(fitted["weights"], raw_model["weights"][permutation], "global_canonical_weights", name+" global atom weights")
        for assay in assays:
            audit.compare(canonical["eta"][assay], fitted["eta"][assay], "canonical_vertex_rates", name+"/"+assay+" vertex rates")
            audit.compare(fitted["eta"][assay], raw_model["eta"][assay][permutation], "global_canonical_rates", name+"/"+assay+" global atom permutation")
        world = item["world"]
        cal_scores, train_losses = [], []
        for assay in assays:
            fit, cal = public[world, assay]
            value = float(loglik(cal, fitted["weights"], fitted["rates"][assay]).mean()/widths[assay])
            cal_scores.append(value)
            train_losses.append(-float(loglik(fit, fitted["weights"], fitted["rates"][assay]).mean()/widths[assay]))
            audit.compare(value, item["calibration_scores"][assay], "calibration_assay", name+"/"+assay+" calibration")
            audit.compare(fit.std(0), item["fit_scale"][assay], "training_scales", name+"/"+assay+" scale")
        scores[name], losses[name] = float(np.mean(cal_scores)), float(np.mean(train_losses))
        audit.compare(scores[name], item["calibration_score"], "calibration_total", name+" calibration total")
        audit.compare(losses[name], model["objective"], "training_objective", name+" training objective")
        audit.compare(losses[name], raw["objective"], "raw_training_objective", name+" raw objective")
        continuation = item["continuation"]
        audit.check(continuation["total_iteration_limit"] is None and continuation["total_evaluation_limit"] is None
                    and continuation["parameter_bounds"] is None, name+" no joint cap/boxes")
        audit.check(sum(chunk["iterations"] for chunk in continuation["chunks"]) == model["iterations"], name+" cumulative iterations")
        audit.check(sum(chunk["objective_evaluations"] for chunk in continuation["chunks"]) == model["objective_evaluations"], name+" cumulative evaluations")
        stops[item["optimizer_mode"]+"/"+model["stop_reason"]] += 1
        audit.counts["outer_objectives_recomputed"] += 1
    selected = {}
    for group, frozen in seal["selected"].items():
        names = [name for name in seal["selected_dimension_restarts"][group] if name in scores]
        converged = [name for name in names if artifacts[name]["model"]["converged"]]
        pool = converged or names
        chosen = min(pool, key=lambda name: (-scores[name], artifacts[name]["seed"])) if pool else None
        selected[group] = chosen
        audit.check(chosen == frozen, group+" independent calibration selection")
        status = "converged_calibration_selection" if converged else "finite_nonconverged_diagnostic" if pool else "no_finite_fit"
        audit.check(status == seal["selection_status"][group], group+" selection status")
    true_losses = {world: -float(np.mean([loglik(public[world, assay][0], truth[world]["weights"],
                                                 truth[world]["rates"][assay]).mean()/widths[assay] for assay in assays]))
                   for world in protocol["worlds"]}
    for name, loss in losses.items():
        world = artifacts[name]["world"]
        stored = evaluation["fits"][name]["optimization_oracle_check"]
        audit.compare(true_losses[world], stored["true_feasible_training_objective"], "true_feasible_objective", name+" true objective")
        audit.compare(loss-true_losses[world], stored["fitted_minus_true_training_objective"], "feasible_point_gap", name+" objective gap")
        audit.check((loss-true_losses[world] > 1e-8) == stored["known_feasible_truth_better"], name+" suboptimality witness")
    print(json.dumps({"audit_stage": "outer_objectives_calibration_selection_and_witness_replayed",
                      "finite_outer_models": len(losses), "selection_groups": len(selected), "issues_so_far": len(audit.issues)}), flush=True)
    autodiff, selected_results = {}, {}
    for group, name in selected.items():
        if name is None:
            selected_results[group] = {"selected": None}
            continue
        item, model, raw = artifacts[name], artifacts[name]["model"], selected_raw[name]
        world, d = item["world"], item["dimension"]
        objective, gradient = torch_gradient(raw["parameter_vector"], {a: public[world, a][0] for a in assays}, assays, widths, d)
        norm = float(np.max(np.abs(gradient)))
        audit.compare(objective, model["objective"], "torch_objective", group+" independent Torch objective")
        audit.compare(gradient, raw["gradient"], "torch_full_gradient", group+" independent Torch gradient vector")
        audit.compare(norm, model["gradient_inf_norm"], "torch_gradient_norm", group+" unbounded gradient norm")
        audit.compare(norm, model["projected_gradient_inf_norm"], "torch_projected_gradient_norm", group+" no projected-bound discount")
        strong = bool(model["status"] in (0, 1) and norm <= protocol["gtol"])
        audit.check(strong == model["converged"] == evaluation["fits"][name]["gradient_converged"], group+" independent convergence gate")
        fitted, target = parsed[name], truth[world]
        ratios = [np.linalg.svd(fitted["matrices"][a], compute_uv=False) for a in assays]
        rank = min(s[-1]/s[0] if s[0] else 0. for s in ratios)
        separation = d == 1 or np.min(np.diff(np.sort(fitted["p"]))) >= protocol["numerical_gates"]["minimum_probability_gap"]
        accepted = bool(strong and rank >= protocol["numerical_gates"]["minimum_loading_singular_ratio"] and separation)
        audit.check(accepted == item["acceptance"]["pass"] == evaluation["fits"][name]["fit_accepted"], group+" canonical M numerical acceptance")
        autodiff[group] = {"objective": objective, "gradient_inf_norm": norm, "gradient_converged": strong}
        components = {a: component(assessment[world, a], fitted["rates"][a]) for a in assays}
        oracle = {a: component(assessment[world, a], target["rates"][a]) for a in assays}
        seed = 71407+int.from_bytes(hashlib.sha256(f"matched:{world}|paired_rows".encode()).digest()[:4], "big")
        indices = np.random.default_rng(seed).integers(0, 4000, size=(1000, 4000), dtype=np.int32)
        directions = {}
        for source, destination in itertools.permutations(assays, 2):
            direction = source+"__to__"+destination
            stored = evaluation["fits"][name]["pairs"][direction]
            cov = relative(covariance(fitted, source, destination), covariance(target, source, destination))
            compare_error(audit, cov, stored["exact_count_covariance_error"], "count_covariance", group+"/"+direction)
            operator = np.linalg.lstsq(fitted["matrices"][source].T, fitted["matrices"][destination].T, rcond=1e-15)[0].T
            transport = relative(operator @ target["matrices"][source], target["matrices"][destination])
            compare_error(audit, transport, stored["restricted_lograte_transport_error"], "restricted_transport", group+"/"+direction)
            support = np.sqrt(np.sum(target["weights"][:, None]*(target["eta"][source] @ operator.T-target["eta"][destination])**2)/widths[destination])
            audit.compare(support, stored["true_support_lograte_transport_rmse"], "support_transport", group+"/"+direction)
            compare_error(audit, relative((target["weights"] @ target["eta"][source]) @ operator.T,
                                          target["weights"] @ target["eta"][destination]),
                          stored["mean_lograte_transport_error"], "mean_transport", group+"/"+direction)
            logw, true_logw = np.log(fitted["weights"]), np.log(target["weights"])
            conditional = logsumexp(components[source]+components[destination]+logw, 1)-logsumexp(components[source]+logw, 1)
            marginal = logsumexp(components[destination]+logw, 1)
            true_conditional = logsumexp(oracle[source]+oracle[destination]+true_logw, 1)-logsumexp(oracle[source]+true_logw, 1)
            true_marginal = logsumexp(oracle[destination]+true_logw, 1)
            rows = {"model_conditional_loglik": conditional, "oracle_conditional_loglik": true_conditional,
                    "model_target_marginal_loglik": marginal, "true_independent_target_loglik": true_marginal,
                    "gain_over_same_model_target_marginal": conditional-marginal,
                    "gain_over_true_independent_target": conditional-true_marginal,
                    "oracle_conditional_logscore_gap": true_conditional-conditional,
                    "target_marginal_logscore_gap": true_marginal-marginal,
                    "oracle_gain_over_true_independent_target": true_conditional-true_marginal}
            for key, values in rows.items():
                audit.compare(values.mean(), stored["per_cell"][key]["mean"], "predictive_mean", group+"/"+direction+"/"+key)
                audit.compare(values.mean()/widths[destination], stored["per_target_feature"][key]["mean"], "predictive_feature_mean", group+"/"+direction+"/"+key)
            intervals = {}
            for key in ("gain_over_same_model_target_marginal", "oracle_conditional_logscore_gap"):
                interval = np.quantile(rows[key][indices].mean(1), [.025, .975])
                intervals[key] = interval
                audit.compare(interval, stored["per_cell"][key]["ci95"], "bootstrap_ci95", group+"/"+direction+"/"+key)
                audit.compare(interval/widths[destination], stored["per_target_feature"][key]["ci95"], "bootstrap_feature_ci95", group+"/"+direction+"/"+key)
            thresholds = protocol["recovery_thresholds"]
            criteria = {"fit_numerically_accepted": accepted,
                        "exact_covariance_relative_error_pass": cov["relative_frobenius_error"] <= thresholds["covariance_relative_error"],
                        "restricted_transport_relative_error_pass": transport["relative_frobenius_error"] <= thresholds["transport_relative_error"],
                        "positive_gain_ci95_lower": bool(intervals["gain_over_same_model_target_marginal"][0] > 0),
                        "oracle_gap_per_feature_pass": bool(rows["oracle_conditional_logscore_gap"].mean()/widths[destination] <= thresholds["oracle_conditional_gap_per_feature"])}
            audit.check(criteria == stored["criteria"] and all(criteria.values()) == stored["recovery_pass"], group+"/"+direction+" independently replayed recovery")
            directions[direction] = {"covariance_relative_error": cov["relative_frobenius_error"],
                                     "transport_relative_error": transport["relative_frobenius_error"], "recovery_pass": bool(all(criteria.values()))}
            audit.counts["selected_directions_recomputed"] += 1
            audit.counts["main_bootstrap_intervals_recomputed"] += 2
        audit.check(all(v["recovery_pass"] for v in directions.values()) == evaluation["fits"][name]["recovery_pass"], group+" all-direction recovery")
        selected_results[group] = {"selected": name, "directions": directions}
    output = {"status": "pass" if not audit.issues else "issues", "created_utc": datetime.now(timezone.utc).isoformat(),
              "reviewer": "Runner/evaluator implementer; not the source-search optimizer author. Independent numerical replay without importing fitting or evaluation code.",
              "exploratory": True, "issues": audit.issues, "counts": dict(audit.counts),
              "maximum_absolute_errors": dict(audit.errors), "tolerances": {"absolute": 1e-9, "relative": 1e-9},
              "inner_inventory": dict(inner_counts), "convergence_outcomes": dict(stops),
              "independent_selected_autodiff": autodiff, "selected_results": selected_results,
              "sealed_hashes": {name: sha(run/name) for name in ("protocol.json", "fit_seal.json", "evaluation.json", "evaluation_seal.json")},
              "audit_script_sha256": sha(Path(__file__)),
              "methods": "Independent SciPy Poisson probabilities, NumPy moment/lstsq transport, Torch float64 gradients in raw unbounded vertex coordinates, calibration and inner-selection replay, and bootstrap quantile replay for selected predictive gains and oracle gaps.",
              "not_repeated": ["No fitting, count generation, EM optimization replay, or exhaustive initializer reconstruction.",
                               "Inner candidate selection replays stored training objectives; independent full-data objectives are recomputed for every returned outer model.",
                               "Exact recovery metrics and gradients are replayed for selected outer fits; unselected outer training/calibration objectives are also checked.",
                               "Bootstrap quantiles for other descriptive metrics are not replayed."],
              "limitations": ["Already assessed worlds remain exploratory; numerical correctness does not supply fresh validation.",
                              "Stationarity does not certify a global optimum; a negative generating-point objective gap is not an optimality certificate."]}
    with output_path.open("x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: output[key] for key in ("status", "issues", "counts", "maximum_absolute_errors", "inner_inventory")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--self-test", action="store_true", help="Check fixed arithmetic fixtures without reading experiment files")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.self_test:
            print(json.dumps(self_checks(), indent=2))
        else:
            main(args.run_dir.resolve())
