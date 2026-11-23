"""Frozen six-family extension sweep; public training arrays only."""
from __future__ import annotations

import json
import multiprocessing
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

STUDY = Path(__file__).resolve().parent
PREVIOUS = STUDY.parent / "count_mdcrl_2026_11_01"
NATIVE = STUDY.parent / "sturma_pilot_2026_11_01"
sys.path.insert(0, str(PREVIOUS))
from apply_counts import align_columns, permute_columns, seed_for, shared_cliques, split_rows
from apply_native import RecordedMDCRL, save_json, sha
from extension_estimators import count_covariance, fit_counts, generalized_cf, projected_moments

PROTOCOL = json.loads((STUDY / "protocol.json").read_text())
ASSAYS = PROTOCOL["assays"]
DATA = STUDY / PROTOCOL["data"]
GRID = np.asarray(PROTOCOL["matching"]["cf_grid"])
MOMENT_MARGINS = np.asarray(PROTOCOL["matching"]["moment_margins"])
FULL_MARGINS = np.r_[MOMENT_MARGINS, np.repeat(PROTOCOL["matching"]["cf_margin"], 2 * len(GRID))]
TRAINING, INPUT_HASHES, PREPARED = {}, {}, {}


def kernel(variant):
    return dict(theta=np.inf if variant["theta"] is None else variant["theta"],
                retention=variant["retention"], correction=variant["correction"])


def read_training(dataset, manifest):
    out, hashes = {}, {}
    for assay in ASSAYS:
        path = DATA / "public" / dataset / "train" / f"{assay}.npz"
        digest = sha(path)
        assert digest == manifest["datasets"][dataset]["assays"][assay]["cohorts"]["train"]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {"raw_counts", "log1p_cp10k", "library_depth", "cell_ids", "feature_ids"}
            x, depth = z["raw_counts"].astype(float), z["library_depth"].astype(float)
            assert x.ndim == 2 and x.shape[1] == 300 and depth.shape == (len(x),)
            assert np.isfinite(x).all() and (x >= 0).all() and np.equal(x, np.floor(x)).all()
            assert np.isfinite(depth).all() and (depth - x.sum(axis=1) > 0).all()
            assert len(set(z["cell_ids"])) == len(x) and len(set(z["feature_ids"])) == 300
            fit, cal = split_rows(z["cell_ids"])
            out[assay] = dict(counts=x, depth=depth, features=z["feature_ids"].copy(), fit=fit, cal=cal)
        hashes[str(path.relative_to(DATA))] = digest
    return out, hashes


def depth_parameters(counts, depth, rows, mode):
    outside = depth - counts.sum(axis=1)
    if mode == "unit":
        return dict(mode=mode, normalization=1., beta=0., beta_objective=None)
    normalizer = float(np.median(outside[rows]))
    beta, objective = 1., None
    if mode == "elastic":
        z = np.log(outside[rows] / normalizer)
        y = counts[rows].sum(axis=1)
        assert y.sum() > 0
        def loss(b):
            return float(y.sum() * logsumexp(b * z) - b * (y @ z))
        fitted = minimize_scalar(loss, bounds=(0., 2.), method="bounded", options={"xatol": 1e-8})
        assert fitted.success and np.isfinite(fitted.fun)
        beta = min([0., float(fitted.x), 2.], key=loss)
        objective = loss(beta)
    else:
        assert mode == "offset"
    return dict(mode=mode, normalization=normalizer, beta=beta, beta_objective=objective)


def exposure_values(counts, depth, parameters):
    if parameters["mode"] == "unit":
        return np.ones(len(counts))
    return ((depth - counts.sum(axis=1)) / parameters["normalization"]) ** parameters["beta"]


def rank_selection(x, exposure, variant, rng):
    args = kernel(variant)
    if not args["correction"]:
        args = dict(theta=np.inf, retention=1., correction=True)
    original = count_covariance(x, exposure, **args)
    scale = original["working_scale"]
    values = np.linalg.eigvalsh(original["standardized_covariance"])[::-1]
    null = np.array([np.linalg.eigvalsh(count_covariance(
        permute_columns(x, exposure, rng), exposure, **args)["covariance"] / np.outer(scale, scale))[-1]
        for _ in range(PROTOCOL["rank"]["null_replicates"])])
    pvalues = (1 + (null[:, None] >= values).sum(axis=0)) / (len(null) + 1)
    uncapped = int(np.sum((values > 0) & (pvalues <= PROTOCOL["rank"]["alpha"])))
    return dict(status="ok", k=min(uncapped, PROTOCOL["rank"]["cap"]), uncapped=uncapped,
                cap_binding=uncapped > PROTOCOL["rank"]["cap"], eigenvalues=values,
                pvalues=pvalues, null_maxima=null)


def prepare(dataset, variants):
    source = TRAINING[dataset]
    transformed, ranks = {}, {}
    for variant in variants:
        for control in PROTOCOL["controls"]:
            for assay in ASSAYS:
                transform_key = variant["depth"], control, assay
                if transform_key not in transformed:
                    own = source[assay]
                    params = depth_parameters(own["counts"], own["depth"], own["fit"], variant["depth"])
                    exposure = exposure_values(own["counts"], own["depth"], params)
                    x = own["counts"].copy()
                    if control == "independent_columns":
                        rng = np.random.default_rng(seed_for(PROTOCOL["control_seed"], dataset, variant["depth"], assay))
                        for rows in [own["fit"], own["cal"]]:
                            x[rows] = permute_columns(x[rows], exposure[rows], rng)
                    transformed[transform_key] = dict(counts=x, exposure=exposure, parameters=params)
                block = transformed[transform_key]
                theta = variant["theta"] if variant["correction"] else None
                retention = variant["retention"] if variant["correction"] else 1.
                rank_key = variant["width"], variant["depth"], theta, retention, control, assay
                if rank_key not in ranks:
                    fit = source[assay]["fit"]
                    ranks[rank_key] = rank_selection(block["counts"][fit, :variant["width"]], block["exposure"][fit], variant,
                        np.random.default_rng(seed_for("extension_rank", dataset, *rank_key)))
                PREPARED[dataset, variant["name"], control, assay] = dict(block, rank=ranks[rank_key])
    return dict(dataset=dataset, depth_parameters={f"{a}:{d}": transformed[d, "original", a]["parameters"]
                for d in {v["depth"] for v in variants} for a in ASSAYS},
                rank_cache_entries=len(ranks), transformed_blocks=len(transformed))


def moments(x, exposure, model, variant):
    result = projected_moments(x, exposure, model["unmixing"], mean=model["mean"], **kernel(variant))
    value = np.column_stack([result["skewness"], result["excess_kurtosis"]])
    return value, np.isfinite(value).all(axis=1) & (result["variance"] > 0), result["variance"]


def effective_gate(model, n):
    if model["orientation"] != "pgf":
        return True
    if model["k_fitted"] == 0:
        return True
    return bool(np.min(model["pgf_effective_cells"]) >= max(
        PROTOCOL["orientation"]["pgf_min_effective_cells"], PROTOCOL["orientation"]["pgf_min_effective_fraction"] * n))


def bootstrap_rows(nfit, ncal, base_seed, b):
    rng = np.random.default_rng(seed_for(base_seed, b))
    return rng.integers(nfit, size=nfit), rng.integers(ncal, size=ncal)


def fit_assay(dataset, variant, seed, control, assay):
    source = TRAINING[dataset][assay]
    prep = PREPARED[dataset, variant["name"], control, assay]
    x, exposure = prep["counts"][:, :variant["width"]], prep["exposure"]
    fit, cal, rank = source["fit"], source["cal"], prep["rank"]
    xf, ef, xc, ec = x[fit], exposure[fit], x[cal], exposure[cal]
    k, repeats = rank["k"], PROTOCOL["bootstrap"]["replicates"]
    model = fit_counts(xf, ef, k, orientation=variant["orientation"], seed=seed, **kernel(variant))
    base_seed = seed_for(PROTOCOL["bootstrap"]["seed"], dataset, control, assay, seed)
    record = dict(assay=assay, k=k, status=model["status"], reason=model.get("reason"), rank=rank,
                  fit_cells=len(fit), calibration_cells=len(cal), bootstrap_seed=base_seed,
                  orientation_residual=model.get("orientation_residual"),
                  orientation_fit_residual=model.get("orientation_fit_residual"),
                  third_offdiagonal_residual=model.get("third_offdiagonal_residual"),
                  condition_number=model.get("condition_number"),
                  negative_eigenvalue_count=model.get("negative_eigenvalue_count"),
                  constant_feature_count=model["constant_feature_count"],
                  eligible_sources=[], eligible_source_count=0, bootstrap_failures=[])
    if model["status"] != "ok":
        return record, model, None
    if k == 0:
        return record, model, dict(value=np.empty((0, 2)), bootstrap=np.empty((repeats, 0, 2)), eligible=np.zeros(0, bool))
    point, valid_point, variance = moments(xc, ec, model, variant)
    boot = np.full((repeats, k, 2), np.nan)
    cosine = np.full((repeats, k), np.nan)
    boot_mix = np.full((repeats, variant["width"], k), np.nan)
    boot_unmix = np.full((repeats, k, variant["width"]), np.nan)
    boot_mean = np.full((repeats, variant["width"]), np.nan)
    boot_ess = np.zeros(repeats, dtype=bool)
    boot_min_ess = np.full(repeats, np.nan)
    boot_constants = np.full(repeats, -1, dtype=int)
    for b in range(repeats):
        ii, jj = bootstrap_rows(len(xf), len(xc), base_seed, b)
        try:
            other = fit_counts(xf[ii], ef[ii], k, orientation=variant["orientation"], seed=seed, **kernel(variant))
            boot_constants[b] = other["constant_feature_count"]
            if other["status"] != "ok":
                record["bootstrap_failures"].append(dict(bootstrap=b, reason=other.get("reason")))
                continue
            order, signs, cosine[b] = align_columns(model["mixing"], other["mixing"], model["working_scale"])
            other["mixing"] = other["mixing"][:, order] * signs
            other["unmixing"] = other["unmixing"][order] * signs[:, None]
            bm, valid, _ = moments(xc[jj], ec[jj], other, variant)
            boot_ess[b] = effective_gate(other, len(xf))
            if other["orientation"] == "pgf":
                boot_min_ess[b] = float(np.min(other["pgf_effective_cells"]))
            boot[b, valid & boot_ess[b]] = bm[valid & boot_ess[b]]
            boot_mix[b], boot_unmix[b], boot_mean[b] = other["mixing"], other["unmixing"], other["mean"]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record["bootstrap_failures"].append(dict(bootstrap=b, reason=f"{type(exc).__name__}: {exc}"))
    valid_n = np.isfinite(boot).all(axis=2).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(cosine, axis=0)
        lower = np.nanquantile(cosine, PROTOCOL["bootstrap"]["lower_cosine_quantile"], axis=0)
    eligible = (valid_point & (valid_n >= PROTOCOL["bootstrap"]["min_valid"])
                & (median >= PROTOCOL["bootstrap"]["median_cosine"])
                & (lower >= PROTOCOL["bootstrap"]["lower_cosine"])
                & (model["orientation_residual"] <= PROTOCOL["orientation"]["max_residual"])
                & effective_gate(model, len(xf)) & (not rank["cap_binding"]))
    record.update(eligible_sources=eligible, eligible_source_count=int(eligible.sum()), point_moment_valid=valid_point,
                  calibration_variance=variance, valid_bootstrap_counts=valid_n, loading_cosine_median=median,
                  loading_cosine_10th=lower, effective_cells_gate=effective_gate(model, len(xf)),
                  pgf_effective_cells=model.get("pgf_effective_cells"))
    return record, model, dict(value=point, bootstrap=boot, eligible=eligible, cosine=cosine,
                               bootstrap_mixing=boot_mix, bootstrap_unmixing=boot_unmix,
                               bootstrap_mean=boot_mean, bootstrap_effective_gate=boot_ess,
                               bootstrap_min_effective_cells=boot_min_ess, bootstrap_constant_features=boot_constants)


def sign_fingerprint(value, sign):
    result = value.copy()
    result[..., 0] *= sign
    if result.shape[-1] > 2:
        result[..., 2 + len(GRID):] *= sign
    return result


def equivalence(a, b, ia, ib, sign, fa, fb, margins):
    difference = (fa["value"][ia] - sign_fingerprint(fb["value"][ib], sign)) / margins
    bd = (fa["bootstrap"][:, ia] - sign_fingerprint(fb["bootstrap"][:, ib], sign)) / margins
    valid = np.isfinite(bd).all(axis=1)
    distance = float(np.max(np.abs(difference)))
    width = float(np.quantile(np.max(np.abs(bd[valid] - difference), axis=1),
        PROTOCOL["matching"]["band_quantile"], method="higher")) if valid.sum() >= PROTOCOL["bootstrap"]["min_valid"] else np.inf
    gate = bool(fa["eligible"][ia] and fb["eligible"][ib])
    return dict(a=a, b=b, ia=ia, ib=ib, sign=sign, source_gates_pass=gate,
                point_scaled_distance=distance, bootstrap_band_radius=width,
                valid_bootstrap_pairs=int(valid.sum()), equivalent=bool(gate and distance + width <= 1),
                not_excluded_by_band=bool(np.isfinite(width) and np.all(np.abs(difference) <= width)))


def unique_matches(candidates):
    return [d for d in candidates if sum((x["a"], x["b"], x["ia"]) == (d["a"], d["b"], d["ia"]) for x in candidates) == 1
            and sum((x["a"], x["b"], x["ib"]) == (d["a"], d["b"], d["ib"]) for x in candidates) == 1]


def empirical_cf(x, exposure, w, mean):
    scores = (x / exposure[:, None] - mean) @ w.T
    weights = np.exp(1j * scores[:, :, None] * GRID)
    value = weights.mean(axis=0)
    se = np.sqrt(np.mean(np.abs(weights - value)**2, axis=0) / (len(x) - 1))
    valid = np.isfinite(value).all(axis=1) & (se <= 0.1).all(axis=1) & (np.abs(value) <= 1.1).all(axis=1)
    return dict(value=value, se=se, valid=valid, max_logweight=np.zeros_like(se),
                mean_absolute_conditioning_proxy=np.zeros_like(se), numerical_valid=np.isfinite(value) & np.isfinite(se),
                reason=np.full(se.shape, "", dtype="<U80"))


def confirmation_fingerprints(dataset, variant, seed, control, assay, model, fp, endpoints):
    k, repeats = len(fp["value"]), PROTOCOL["bootstrap"]["replicates"]
    assessed = np.zeros(k, bool)
    assessed[endpoints] = True
    value = np.full((k, len(GRID)), complex(np.nan, np.nan))
    bootstrap = np.full((repeats, k, len(GRID)), complex(np.nan, np.nan))
    raw_bootstrap = bootstrap.copy()
    bootstrap_diagnostics = {key: np.full(bootstrap.shape, np.nan) for key in
                             ["se", "max_logweight", "mean_absolute_conditioning_proxy"]}
    bootstrap_numerical = np.zeros(bootstrap.shape, bool)
    point_valid = np.zeros(k, bool)
    diagnostics = {}
    if endpoints:
        prep, source = PREPARED[dataset, variant["name"], control, assay], TRAINING[dataset][assay]
        fit, cal = source["fit"], source["cal"]
        x, exposure = prep["counts"][cal, :variant["width"]], prep["exposure"][cal]
        def compute(xx, ee, ww, mm):
            if not variant["correction"]:
                return empirical_cf(xx, ee, ww, mm)
            return generalized_cf(xx, ee, ww, mm, theta=kernel(variant)["theta"], retention=variant["retention"], grid=GRID)
        point = compute(x, exposure, model["unmixing"][endpoints], model["mean"])
        value[endpoints], point_valid[endpoints] = point["value"], point["valid"]
        diagnostics = {key: result for key, result in point.items() if key != "value"}
        base_seed = seed_for(PROTOCOL["bootstrap"]["seed"], dataset, control, assay, seed)
        for b in range(repeats):
            if not np.isfinite(fp["bootstrap_unmixing"][b]).all():
                continue
            _, jj = bootstrap_rows(len(fit), len(cal), base_seed, b)
            got = compute(x[jj], exposure[jj], fp["bootstrap_unmixing"][b, endpoints], fp["bootstrap_mean"][b])
            raw_bootstrap[b, endpoints] = got["value"]
            bootstrap_numerical[b, endpoints] = got["numerical_valid"]
            for key in bootstrap_diagnostics:
                bootstrap_diagnostics[key][b, endpoints] = got[key]
            for j, index in enumerate(endpoints):
                if got["valid"][j]:
                    bootstrap[b, index] = got["value"][j]
    valid_n = np.isfinite(bootstrap).all(axis=2).sum(axis=0)
    eligible = fp["eligible"] & point_valid & (valid_n >= PROTOCOL["bootstrap"]["min_valid"])
    full = dict(value=np.c_[fp["value"], value.real, value.imag],
                bootstrap=np.concatenate([fp["bootstrap"], bootstrap.real, bootstrap.imag], axis=2), eligible=eligible)
    return full, dict(assessed=assessed, value=value, bootstrap=bootstrap, eligible=eligible,
                      point_valid=point_valid, valid_bootstrap_counts=valid_n,
                      bootstrap_raw_value=raw_bootstrap, bootstrap_numerical_valid=bootstrap_numerical,
                      **{f"bootstrap_{key}": val for key, val in bootstrap_diagnostics.items()}), diagnostics


def fit_task(task):
    dataset, variant, seed, control, name = task
    output = STUDY / "runs_v1" / name
    started = time.monotonic()
    record = dict(dataset=dataset, variant=variant, source_seed=seed, control=control, status="completed",
                  input_hashes=INPUT_HASHES[dataset], exposure_parameters={}, assays=[],
                  interpretation="Finite-fingerprint candidates; no identified causal structure claim.")
    try:
        arrays, models, fps = {}, {}, {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for assay in ASSAYS:
                ar, model, fp = fit_assay(dataset, variant, seed, control, assay)
                record["assays"].append(ar)
                record["exposure_parameters"][assay] = PREPARED[dataset, variant["name"], control, assay]["parameters"]
                models[assay], fps[assay] = model, fp
                for key in ["mean", "scale", "working_scale", "constant_mask", "mixing", "unmixing", "covariance", "noise_diagonal", "poisson_noise_diagonal", "whitening",
                            "white_third_tensor", "orientation_target_tensor", "shrink_variance_proxy", "shrink_group_tensors",
                            "shrink_group_sizes", "generalized_covariance_matrices", "pgf_processing_points", "pgf_effective_cells", "rotation"]:
                    if model.get(key) is not None:
                        arrays[f"{key}_{assay}"] = model[key]
                source = TRAINING[dataset][assay]
                arrays[f"features_{assay}"] = source["features"][:variant["width"]]
                arrays[f"fit_indices_{assay}"], arrays[f"calibration_indices_{assay}"] = source["fit"], source["cal"]
                if model["status"] == "ok":
                    arrays[f"predicted_observed_covariance_{assay}"] = model["mixing"] @ model["mixing"].T + np.diag(model["noise_diagonal"])
                else:
                    arrays[f"predicted_observed_covariance_{assay}"] = np.full((variant["width"], variant["width"]), np.nan)
                    record["status"] = "completed_with_abstention"
                if fp is not None:
                    for key, val in fp.items():
                        arrays[f"fingerprint_{key}_{assay}"] = val
            details = []
            for a, b in combinations(ASSAYS, 2):
                if fps[a] is None or fps[b] is None:
                    continue
                for ia, ib in np.ndindex(len(fps[a]["value"]), len(fps[b]["value"])):
                    for sign in [1, -1]:
                        details.append(equivalence(a, b, ia, ib, sign, fps[a], fps[b], MOMENT_MARGINS))
            candidates = [d for d in details if d["equivalent"]]
            moment_matches = unique_matches(candidates)
            full, cf_reports = {}, {}
            for assay in ASSAYS:
                if fps[assay] is None:
                    continue
                endpoints = sorted({d["ia"] if d["a"] == assay else d["ib"] for d in candidates if assay in [d["a"], d["b"]]})
                combined, values, diagnostics = confirmation_fingerprints(dataset, variant, seed, control, assay, models[assay], fps[assay], endpoints)
                full[assay] = combined
                cf_reports[assay] = dict(status="assessed_moment_candidates" if endpoints else "not_assessed_no_moment_candidates",
                                         endpoints=endpoints, valid_bootstrap_counts=values["valid_bootstrap_counts"],
                                         eligible=values["eligible"], point_diagnostics=diagnostics)
                for key, val in values.items():
                    arrays[f"cf_{key}_{assay}"] = val
            confirmed_details = [equivalence(d["a"], d["b"], d["ia"], d["ib"], d["sign"],
                full[d["a"]], full[d["b"]], FULL_MARGINS) for d in candidates]
            confirmed = unique_matches([d for d in confirmed_details if d["equivalent"]])
            record.update(moment_diagnostics=details, moment_equivalent_pairs=candidates,
                          moment_pair_matches=moment_matches, confirmation_diagnostics=confirmed_details,
                          confirmed_pair_matches=confirmed, cf_assessment=cf_reports,
                          moment_all_assay_matches=shared_cliques(moment_matches), confirmed_all_assay_matches=shared_cliques(confirmed))
            record["ambiguity_sets"] = [dict(a=a, b=b, ia=ia,
                moment_equivalent=[dict(ib=d["ib"], sign=d["sign"]) for d in candidates if (d["a"], d["b"], d["ia"]) == (a, b, ia)],
                cf_equivalent=[dict(ib=d["ib"], sign=d["sign"]) for d in confirmed_details if (d["a"], d["b"], d["ia"]) == (a, b, ia) and d["equivalent"]],
                not_excluded_by_moment_band=[dict(ib=d["ib"], sign=d["sign"]) for d in details if (d["a"], d["b"], d["ia"]) == (a, b, ia) and d["not_excluded_by_band"]])
                for a, b in combinations(ASSAYS, 2) if fps[a] is not None for ia in range(len(fps[a]["value"]))]
            for label, matches in [("moment", moment_matches), ("confirmed", confirmed)]:
                for a, b in combinations(ASSAYS, 2):
                    cross = np.zeros((variant["width"], variant["width"]))
                    for d in matches:
                        if (d["a"], d["b"]) == (a, b):
                            cross += d["sign"] * np.outer(models[a]["mixing"][:, d["ia"]], models[b]["mixing"][:, d["ib"]])
                    arrays[f"predicted_cross_covariance_{label}_{a}_{b}"] = cross
            cliques = record["confirmed_all_assay_matches"]
            record["graph_status"] = "fewer_than_two_confirmed_all_assay_candidates"
            if len(cliques) >= 2:
                native = RecordedMDCRL(alpha=0.05, gamma=0.2)
                native.nr_env, native.mixings = 3, [models[a]["mixing"] for a in ASSAYS]
                native.joint_factors, native.joint_signs = [c["indices"] for c in cliques], [c["signs"] for c in cliques]
                try:
                    native.get_joint_mixing()
                    native.get_joint_graph()
                    arrays["exploratory_native_A"] = native.A
                    record["graph_status"] = "exploratory_finite_coefficients" if np.isfinite(native.A).all() else "nonfinite_coefficients"
                except Exception as exc:
                    record.update(graph_status="native_extraction_failed", graph_error=f"{type(exc).__name__}: {exc}")
            record["warnings"] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
        np.savez_compressed(output / "arrays.npz", **arrays)
        record["arrays_sha256"] = sha(output / "arrays.npz")
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if "arrays" in locals() and arrays:
            np.savez_compressed(output / "partial_arrays.npz", **arrays)
            record["partial_arrays_sha256"] = sha(output / "partial_arrays.npz")
    record["elapsed_seconds"] = time.monotonic() - started
    save_json(output / "result.json", record)
    return dict(path=name, status=record["status"], ranks=[a["k"] for a in record["assays"]],
                eligible_sources=[a.get("eligible_source_count", 0) for a in record["assays"]],
                moment_equivalent_pairs=len(record.get("moment_equivalent_pairs", [])),
                moment_pair_matches=len(record.get("moment_pair_matches", [])),
                confirmed_pair_matches=len(record.get("confirmed_pair_matches", [])),
                confirmed_all_assay_matches=len(record.get("confirmed_all_assay_matches", [])),
                graph_status=record.get("graph_status"), error=record.get("error"), elapsed_seconds=record["elapsed_seconds"])


def main():
    global TRAINING, INPUT_HASHES
    assert sha(DATA / "manifest.json") == PROTOCOL["data_manifest_sha256"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    output = STUDY / "runs_v1"
    output.mkdir(exist_ok=False)
    local = [STUDY / name for name in ["protocol.json", "apply_extensions.py", "extension_estimators.py"]]
    support = [PREVIOUS / "apply_counts.py", PREVIOUS / "count_sources.py", PREVIOUS / "protocol.json", NATIVE / "apply_native.py", NATIVE / "upstream/src/learner.py"]
    hashes = {p.name: sha(p) for p in local}
    support_hashes = {str(p.relative_to(STUDY.parent)): sha(p) for p in support}
    start = time.monotonic()
    tasks, preparation = [], []
    with threadpool_limits(limits=1):
        for dataset in PROTOCOL["datasets"]:
            TRAINING[dataset], INPUT_HASHES[dataset] = read_training(dataset, manifest)
        # Dataset preparation writes disjoint cache keys and uses only local RNG streams.
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = [pool.submit(prepare, dataset, PROTOCOL["variants"]) for dataset in PROTOCOL["datasets"]]
            for future in as_completed(prepared):
                item = future.result()
                preparation.append(item)
                print(json.dumps(dict(prepared=item["dataset"], rank_cache_entries=item["rank_cache_entries"])), flush=True)
        preparation.sort(key=lambda item: item["dataset"])
        for dataset in PROTOCOL["datasets"]:
            for variant in PROTOCOL["variants"]:
                for control, seeds in PROTOCOL["controls"].items():
                    for seed in seeds:
                        name = f"{dataset}__{variant['name']}__seed{seed}" + ("__independent_columns" if control != "original" else "")
                        (output / name).mkdir(exist_ok=False)
                        tasks.append((dataset, variant, seed, control, name))
        save_json(output / "preparation.json", preparation)
        assert len(tasks) == PROTOCOL["expected_arm_records"]
        records = []
        # Fork shares the immutable cached training blocks without duplicating their preparation.
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
            for future in as_completed([pool.submit(fit_task, task) for task in tasks]):
                record = future.result()
                records.append(record)
                print(json.dumps(record), flush=True)
    records.sort(key=lambda r: r["path"])
    save_json(output / "summary.json", records)
    assert hashes == {p.name: sha(p) for p in local}, "Source changed during fitting"
    assert support_hashes == {str(p.relative_to(STUDY.parent)): sha(p) for p in support}, "Supporting source changed during fitting"
    save_json(output / "provenance.json", dict(source_sha256=hashes, support_sha256=support_hashes,
        data_manifest_sha256=sha(DATA / "manifest.json"), evaluator_read=False,
        elapsed_wall_seconds=time.monotonic() - start,
        artifacts={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}))


if __name__ == "__main__":
    main()
