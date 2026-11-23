"""Frozen count-source pilot. Training and evaluation are separate executables."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

STUDY = Path(__file__).resolve().parent
NATIVE = STUDY.parent / "sturma_pilot_2026_11_01"
sys.path.insert(0, str(NATIVE))
from apply_native import RecordedMDCRL, save_json, sha
from count_sources import fit_counts, projected_factorial_moments

PROTOCOL = json.loads((STUDY / "protocol.json").read_text())
DATA = (STUDY / PROTOCOL["data"]).resolve()
ASSAYS = PROTOCOL["assays"]
GRID = np.asarray(PROTOCOL["fingerprint"]["characteristic_function_grid"])
MARGINS = np.asarray(PROTOCOL["fingerprint"]["moment_margins"] +
                     [PROTOCOL["fingerprint"]["cf_coordinate_margin"]] * (2 * len(GRID)))


def seed_for(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def split_rows(ids):
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        ("count-dica-v1:" + str(ids[i])).encode()).hexdigest())
    cut = int(0.75 * len(ids))
    return np.asarray(order[:cut]), np.asarray(order[cut:])


def read_training(dataset):
    out, hashes = {}, {}
    manifest = json.loads((DATA / "manifest.json").read_text())
    for assay in ASSAYS:
        path = DATA / "public" / dataset / "train" / f"{assay}.npz"
        actual_hash = sha(path)
        assert actual_hash == manifest["datasets"][dataset]["assays"][assay]["cohorts"]["train"]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {"raw_counts", "log1p_cp10k", "library_depth", "cell_ids", "feature_ids"}
            x = z["raw_counts"].astype(float)
            depth = z["library_depth"].astype(float)
            assert x.ndim == 2 and x.shape[1] == 30 and len(depth) == len(x)
            assert np.isfinite(x).all() and (x >= 0).all() and np.equal(x, np.floor(x)).all()
            assert np.isfinite(depth).all() and (depth - x.sum(axis=1) > 0).all()
            assert len(set(z["cell_ids"])) == len(x) and len(set(z["feature_ids"])) == 30
            fit, cal = split_rows(z["cell_ids"])
            out[assay] = dict(counts=x, depth=depth, features=z["feature_ids"].copy(), fit=fit, cal=cal)
        hashes[str(path.relative_to(DATA))] = actual_hash
    return out, hashes


def strata(exposure):
    if np.ptp(exposure) == 0:
        return [np.arange(len(exposure))]
    return np.array_split(np.argsort(exposure, kind="stable"), 8)


def permute_columns(x, exposure, rng):
    out = x.copy()
    for rows in strata(exposure):
        for j in range(x.shape[1]):
            out[rows, j] = x[rng.permutation(rows), j]
    return out


def corrected_covariance(x, exposure, scale):
    q = x / exposure[:, None]
    centered = q - q.mean(axis=0)
    cov = centered.T @ centered / len(q) - np.diag(np.mean(x / exposure[:, None]**2, axis=0))
    return cov / np.outer(scale, scale)


def select_rank(x, exposure, rng):
    scale = (x / exposure[:, None]).std(axis=0)
    if np.any(scale <= 0):
        return dict(k=0, uncapped=0, cap_binding=False, status="constant_fit_feature", null_maxima=[], eigenvalues=[], pvalues=[])
    eigenvalues = np.linalg.eigvalsh(corrected_covariance(x, exposure, scale))[::-1]
    nulls = np.asarray([np.linalg.eigvalsh(corrected_covariance(
        permute_columns(x, exposure, rng), exposure, scale))[-1]
        for _ in range(PROTOCOL["rank"]["null_replicates"])])
    pvalues = (1 + (nulls[:, None] >= eigenvalues).sum(axis=0)) / (len(nulls) + 1)
    uncapped = int(np.sum((eigenvalues > 0) & (pvalues <= PROTOCOL["rank"]["alpha"])))
    return dict(k=min(uncapped, PROTOCOL["rank"]["cap"]), uncapped=uncapped,
                cap_binding=uncapped > PROTOCOL["rank"]["cap"], status="ok",
                eigenvalues=eigenvalues, null_maxima=nulls, pvalues=pvalues)


def characteristic_fingerprint(x, exposure, unmixing, mean, correction):
    """Fixed-direction Poisson deconvolution; empirical variance gate is diagnostic."""
    k = len(unmixing)
    values = np.full((k, len(GRID)), complex(np.nan, np.nan))
    errors = np.full((k, len(GRID)), np.inf)
    max_log = np.full((k, len(GRID)), np.inf)
    for a, w in enumerate(unmixing):
        for b, frequency in enumerate(GRID):
            if correction:
                log_weight = np.sum(x * np.log1p(1j * frequency * w / exposure[:, None]), axis=1)
                log_weight -= 1j * frequency * (w @ mean)
            else:
                log_weight = 1j * frequency * ((x / exposure[:, None] - mean) @ w)
            max_log[a, b] = float(np.max(log_weight.real))
            if not np.isfinite(log_weight).all() or max_log[a, b] > 300:
                continue
            weights = np.exp(log_weight)
            values[a, b] = weights.mean()
            errors[a, b] = np.sqrt(np.mean(np.abs(weights - weights.mean())**2) / max(len(x) - 1, 1))
    valid = (np.isfinite(values).all(axis=1) & (errors <= 0.1).all(axis=1) &
             (np.abs(values) <= 1.1).all(axis=1))
    return values, errors, max_log, valid


def fingerprint(x, exposure, model, correction):
    moments = projected_factorial_moments(x, exposure, model["unmixing"],
                                         mean=model["mean"], correction=correction)
    cf, error, max_log, cf_valid = characteristic_fingerprint(
        x, exposure, model["unmixing"], model["mean"], correction)
    value = np.column_stack([moments["skewness"], moments["excess_kurtosis"], cf.real, cf.imag])
    valid = np.isfinite(value).all(axis=1) & (moments["variance"] > 0) & cf_valid
    return dict(value=value, valid=valid, variance=moments["variance"], cf_se=error, cf_max_log=max_log)


def align_columns(reference, candidate, scale):
    a, b = reference / scale[:, None], candidate / scale[:, None]
    cosine = (a.T @ b) / np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    row, col = linear_sum_assignment(-np.abs(cosine))
    assert np.array_equal(row, np.arange(reference.shape[1]))
    signs = np.where(cosine[row, col] >= 0, 1, -1)
    return col, signs, np.abs(cosine[row, col])


def fit_assay(x, exposure, fit_rows, cal_rows, rank, correction, source_seed, bootstrap_seed):
    xf, ef, xc, ec = x[fit_rows], exposure[fit_rows], x[cal_rows], exposure[cal_rows]
    k, repeats = rank["k"], PROTOCOL["bootstrap"]["replicates"]
    model = fit_counts(xf, ef, k, correction=correction, seed=source_seed)
    record = dict(k=k, fit_cells=len(xf), calibration_cells=len(xc), rank=rank,
                  status=model["status"], reason=model.get("reason"), warnings=[])
    if model["status"] != "ok":
        return record, model, None
    record.update(third_offdiagonal_residual=model["third_offdiagonal_residual"],
                  condition_number=model.get("condition_number"),
                  covariance_negative_eigenvalues=int(np.sum(model["eigenvalues"] < 0)))
    if k == 0:
        record.update(eligible_sources=[], bootstrap_failures=[], eligible_source_count=0)
        return record, model, dict(value=np.empty((0, len(MARGINS))), bootstrap=np.empty((repeats, 0, len(MARGINS))), eligible=np.zeros(0, dtype=bool))
    point = fingerprint(xc, ec, model, correction)
    boot = np.full((repeats, k, len(MARGINS)), np.nan)
    cosine = np.full((repeats, k), np.nan)
    boot_mix = np.full((repeats, xf.shape[1], k), np.nan)
    boot_unmix = np.full((repeats, k, xf.shape[1]), np.nan)
    boot_mean = np.full((repeats, xf.shape[1]), np.nan)
    failures = []
    for b in range(repeats):
        rng = np.random.default_rng(seed_for(bootstrap_seed, b))
        ii, jj = rng.integers(len(xf), size=len(xf)), rng.integers(len(xc), size=len(xc))
        try:
            candidate = fit_counts(xf[ii], ef[ii], k, correction=correction, seed=source_seed)
            if candidate["status"] != "ok":
                failures.append(dict(bootstrap=b, reason=candidate.get("reason", "abstained")))
                continue
            order, sign, cosine[b] = align_columns(model["mixing"], candidate["mixing"], model["scale"])
            candidate["mixing"] = candidate["mixing"][:, order] * sign
            candidate["unmixing"] = candidate["unmixing"][order] * sign[:, None]
            candidate_fp = fingerprint(xc[jj], ec[jj], candidate, correction)
            boot[b, candidate_fp["valid"]] = candidate_fp["value"][candidate_fp["valid"]]
            boot_mix[b], boot_unmix[b], boot_mean[b] = candidate["mixing"], candidate["unmixing"], candidate["mean"]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append(dict(bootstrap=b, reason=f"{type(exc).__name__}: {exc}"))
    valid_n = np.isfinite(boot).all(axis=2).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cosine_median = np.nanmedian(cosine, axis=0)
        cosine_low = np.nanquantile(cosine, 0.1, axis=0)
    eligible = (point["valid"] & (valid_n >= 26) & (cosine_median >= 0.9) & (cosine_low >= 0.7)
                & (model["third_offdiagonal_residual"] <= 0.35) & (not rank["cap_binding"]))
    record.update(eligible_sources=eligible, eligible_source_count=int(eligible.sum()),
                  point_fingerprint_valid=point["valid"], point_calibration_variance=point["variance"],
                  point_cf_standard_error=point["cf_se"], point_cf_max_log_weight=point["cf_max_log"],
                  valid_bootstrap_counts=valid_n, loading_cosine_median=cosine_median,
                  loading_cosine_10th=cosine_low, bootstrap_failures=failures)
    return record, model, dict(value=point["value"], bootstrap=boot, eligible=eligible, cosine=cosine,
                               bootstrap_mixing=boot_mix, bootstrap_unmixing=boot_unmix, bootstrap_mean=boot_mean)


def signed_fingerprint(value, sign):
    result = value.copy()
    result[..., 0] *= sign
    result[..., 2 + len(GRID):] *= sign
    return result


def match_pair(a, b, fa, fb):
    details, candidates = [], []
    if fa is None or fb is None:
        return [], [], []
    for ia, ib in np.ndindex(len(fa["value"]), len(fb["value"])):
        for sign in [1, -1]:
            difference = (fa["value"][ia] - signed_fingerprint(fb["value"][ib], sign)) / MARGINS
            bd = (fa["bootstrap"][:, ia] - signed_fingerprint(fb["bootstrap"][:, ib], sign)) / MARGINS
            valid = np.isfinite(bd).all(axis=1)
            width = float(np.quantile(np.max(np.abs(bd[valid] - difference), axis=1), 0.9, method="higher")) if valid.sum() >= 26 else np.inf
            distance = float(np.max(np.abs(difference)))
            gate = bool(fa["eligible"][ia] and fb["eligible"][ib])
            equivalent = bool(gate and distance + width <= 1)
            entry = dict(a=a, b=b, ia=ia, ib=ib, sign=sign, source_gates_pass=gate,
                         point_scaled_distance=distance, bootstrap_band_radius=width,
                         valid_bootstrap_pairs=int(valid.sum()), equivalent=equivalent,
                         not_excluded_by_band=bool(np.isfinite(width) and np.all(np.abs(difference) <= width)))
            details.append(entry)
            if equivalent:
                candidates.append(entry)
    matches = [d for d in candidates if sum(x["ia"] == d["ia"] for x in candidates) == 1
               and sum(x["ib"] == d["ib"] for x in candidates) == 1]
    ambiguity = [dict(a=a, b=b, ia=ia, accepted_candidates=[dict(ib=d["ib"], sign=d["sign"]) for d in candidates if d["ia"] == ia],
                      compatible_candidates=[dict(ib=d["ib"], sign=d["sign"]) for d in details if d["ia"] == ia and d["not_excluded_by_band"]])
                 for ia in range(len(fa["value"]))]
    return matches, details, ambiguity


def shared_cliques(matches):
    ab = [d for d in matches if (d["a"], d["b"]) == tuple(ASSAYS[:2])]
    ac = [d for d in matches if (d["a"], d["b"]) == (ASSAYS[0], ASSAYS[2])]
    bc = [d for d in matches if (d["a"], d["b"]) == tuple(ASSAYS[1:])]
    return [dict(indices=[x["ia"], x["ib"], y["ib"]], signs=[1, x["sign"], y["sign"]])
            for x in ab for y in ac for z in bc if x["ia"] == y["ia"] and x["ib"] == z["ia"]
            and y["ib"] == z["ib"] and x["sign"] * z["sign"] == y["sign"]]


def fit_arm(dataset, exposure_mode, method, seed, control, inputs, hashes, prepared, output):
    start = time.monotonic()
    record = dict(dataset=dataset, exposure=exposure_mode, method=method, source_seed=seed,
                  control=control, input_hashes=hashes, exposure_normalization={}, assays=[],
                  status="completed", interpretation="Exploratory finite-fingerprint candidates, not identified causal structure.")
    arrays, models, fps = {}, {}, {}
    correction = method == "poisson_dica"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for assay in ASSAYS:
            source, arm = inputs[assay], prepared[assay]
            ar, model, fp = fit_assay(arm["counts"], arm["exposure"], source["fit"], source["cal"], arm["rank"], correction, seed,
                                    seed_for(PROTOCOL["bootstrap"]["seed"], dataset, exposure_mode, control, assay, seed))
            ar["assay"] = assay
            record["assays"].append(ar)
            record["exposure_normalization"][assay] = arm["normalization"]
            models[assay], fps[assay] = model, fp
            for key in ["mean", "scale", "mixing", "unmixing", "covariance", "standardized_covariance", "noise_diagonal", "poisson_noise_diagonal", "third_tensor", "third_tensor_reconstruction"]:
                if key in model and model[key] is not None:
                    arrays[f"{key}_{assay}"] = model[key]
            arrays[f"features_{assay}"] = source["features"]
            arrays[f"fit_indices_{assay}"], arrays[f"calibration_indices_{assay}"] = source["fit"], source["cal"]
            if model["status"] == "ok":
                arrays[f"predicted_observed_covariance_{assay}"] = model["mixing"] @ model["mixing"].T + np.diag(model["noise_diagonal"])
            else:
                arrays[f"predicted_observed_covariance_{assay}"] = np.full((30, 30), np.nan)
                record["status"] = "completed_with_abstention"
            if fp is not None:
                for key, value in fp.items():
                    arrays[f"fingerprint_{key}_{assay}"] = value
        record["warnings"] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
    matches, pair_details, ambiguity = [], [], []
    for a, b in combinations(ASSAYS, 2):
        mm, dd, aa = match_pair(a, b, fps[a], fps[b])
        matches.extend(mm)
        pair_details.extend(dd)
        ambiguity.extend(aa)
        cross = np.zeros((30, 30))
        for d in mm:
            cross += d["sign"] * np.outer(models[a]["mixing"][:, d["ia"]], models[b]["mixing"][:, d["ib"]])
        arrays[f"predicted_cross_covariance_{a}_{b}"] = cross
    cliques = shared_cliques(matches)
    record.update(pair_matches=matches, pair_diagnostics=pair_details, ambiguity_sets=ambiguity, all_assay_matches=cliques,
                  graph_status="fewer_than_two_all_assay_candidates")
    if len(cliques) >= 2:
        native = RecordedMDCRL(alpha=0.05, gamma=0.2)
        native.nr_env, native.mixings = 3, [models[a]["mixing"] for a in ASSAYS]
        native.joint_factors = [d["indices"] for d in cliques]
        native.joint_signs = [d["signs"] for d in cliques]
        try:
            native.get_joint_mixing()
            native.get_joint_graph()
            arrays["exploratory_native_A"] = native.A
            record["graph_status"] = "exploratory_finite_coefficients" if np.isfinite(native.A).all() else "nonfinite_coefficients"
            record["graph_row_column_order"] = getattr(native, "recorded_order", None)
        except Exception as exc:
            record.update(graph_status="native_extraction_failed", graph_error=f"{type(exc).__name__}: {exc}")
    np.savez_compressed(output / "arrays.npz", **arrays)
    record.update(arrays_sha256=sha(output / "arrays.npz"), elapsed_seconds=time.monotonic() - start)
    save_json(output / "result.json", record)
    return record


def main():
    assert sha(DATA / "manifest.json") == PROTOCOL["data_manifest_sha256"]
    output = STUDY / "runs_v1"
    output.mkdir(exist_ok=False)
    records = []
    source_hashes = {p.name: sha(p) for p in [STUDY / "protocol.json", STUDY / "apply_counts.py", STUDY / "count_sources.py"]}
    support_hashes = {str(p.relative_to(STUDY.parent)): sha(p) for p in [NATIVE / "apply_native.py", NATIVE / "upstream/src/learner.py"]}
    with threadpool_limits(limits=1):
        for dataset in PROTOCOL["datasets"]:
            inputs, hashes = read_training(dataset)
            for exposure_mode in PROTOCOL["exposures"]:
                for control, seeds in PROTOCOL["controls"].items():
                    prepared = {}
                    for assay, source in inputs.items():
                        depth = source["depth"] - source["counts"].sum(axis=1)
                        normalization = 1.0 if exposure_mode == "unit" else float(np.median(depth[source["fit"]]))
                        exposure = np.ones(len(depth)) if exposure_mode == "unit" else depth / normalization
                        x = source["counts"].copy()
                        if control == "independent_columns":
                            rng = np.random.default_rng(seed_for(PROTOCOL["control_seed"], dataset, exposure_mode, assay))
                            for rows in [source["fit"], source["cal"]]:
                                x[rows] = permute_columns(x[rows], exposure[rows], rng)
                        rank = select_rank(x[source["fit"]], exposure[source["fit"]],
                                           np.random.default_rng(seed_for("rank", dataset, exposure_mode, control, assay)))
                        prepared[assay] = dict(counts=x, exposure=exposure, normalization=normalization, rank=rank)
                    for method in PROTOCOL["methods"]:
                        for seed in seeds:
                            name = f"{dataset}__{exposure_mode}__{method}__seed{seed}" + ("__independent_columns" if control != "original" else "")
                            arm = output / name
                            arm.mkdir(exist_ok=False)
                            try:
                                record = fit_arm(dataset, exposure_mode, method, seed, control, inputs, hashes, prepared, arm)
                            except Exception as exc:
                                record = dict(dataset=dataset, exposure=exposure_mode, method=method, source_seed=seed, control=control,
                                              status="failed", error=f"{type(exc).__name__}: {exc}")
                                save_json(arm / "result.json", record)
                            records.append(dict(path=name, status=record["status"], ranks=[x["k"] for x in record.get("assays", [])],
                                                eligible_sources=[x.get("eligible_source_count", 0) for x in record.get("assays", [])],
                                                pair_matches=len(record.get("pair_matches", [])), all_assay_matches=len(record.get("all_assay_matches", [])),
                                                graph_status=record.get("graph_status"), error=record.get("error")))
                            print(json.dumps(records[-1]), flush=True)
    assert len(records) == PROTOCOL["expected_arm_records"]
    save_json(output / "summary.json", records)
    assert source_hashes == {name: sha(STUDY / name) for name in source_hashes}, "Source changed during run"
    assert support_hashes == {name: sha(STUDY.parent / name) for name in support_hashes}, "Supporting source changed during run"
    save_json(output / "provenance.json", dict(source_sha256=source_hashes, support_sha256=support_hashes, data_manifest_sha256=sha(DATA / "manifest.json"),
               artifacts={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
               evaluator_read=False))


if __name__ == "__main__":
    main()
