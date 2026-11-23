"""Replay sealed count-source decisions without fitting or reading test data.

This audit does not import the training runner or source estimator. It checks
the saved numerical decisions, not the truth of the biological assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


STUDY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, name):
    """Compare JSON values, treating nonfinite predicted floats as saved null."""
    if isinstance(expected, np.ndarray):
        expected = expected.tolist()
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), name + ": keys")
        for key in expected:
            same(actual[key], expected[key], name + "/" + str(key))
    elif isinstance(expected, (list, tuple)):
        check(len(actual) == len(expected), name + ": length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, name + "/" + str(i))
    elif isinstance(expected, (float, np.floating)):
        if not np.isfinite(expected):
            check(actual is None, name + ": nonfinite must be null")
        else:
            check(actual is not None and np.isclose(actual, expected, rtol=1e-9, atol=1e-11), name + ": numerical value")
    else:
        check(actual == expected, name + ": value")


def close(actual, expected, name, rtol=1e-8, atol=1e-10):
    check(np.shape(actual) == np.shape(expected), name + ": shape")
    check(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), name + ": numerical value")


def sign_change(value, sign, ngrid):
    multipliers = np.ones(value.shape[-1])
    multipliers[0] = sign
    multipliers[2 + ngrid:] = sign
    return value * multipliers


def pair_decisions(a, b, fa, fb, margins, ngrid):
    if fa is None or fb is None:
        return [], [], []
    details, accepted = [], []
    for ia in range(len(fa["value"])):
        for ib in range(len(fb["value"])):
            for sign in (1, -1):
                difference = (fa["value"][ia] - sign_change(fb["value"][ib], sign, ngrid)) / margins
                delta = (fa["bootstrap"][:, ia] - sign_change(fb["bootstrap"][:, ib], sign, ngrid)) / margins
                valid = np.isfinite(delta).all(axis=1)
                n = int(valid.sum())
                radius = float("inf")
                if n >= 26:
                    maxima = np.sort(np.max(np.abs(delta[valid] - difference), axis=1))
                    radius = float(maxima[math.ceil(0.9 * (n - 1))])
                distance = float(np.max(np.abs(difference)))
                gates = bool(fa["eligible"][ia] and fb["eligible"][ib])
                equivalent = bool(gates and distance + radius <= 1)
                item = dict(a=a, b=b, ia=ia, ib=ib, sign=sign, source_gates_pass=gates,
                            point_scaled_distance=distance, bootstrap_band_radius=radius,
                            valid_bootstrap_pairs=n, equivalent=equivalent,
                            not_excluded_by_band=bool(np.isfinite(radius) and np.all(np.abs(difference) <= radius)))
                details.append(item)
                if equivalent:
                    accepted.append(item)
    matches = []
    for item in accepted:
        degree_a = sum(other["ia"] == item["ia"] for other in accepted)
        degree_b = sum(other["ib"] == item["ib"] for other in accepted)
        if degree_a == degree_b == 1:
            matches.append(item)
    ambiguity = []
    for ia in range(len(fa["value"])):
        ambiguity.append(dict(a=a, b=b, ia=ia,
                              accepted_candidates=[dict(ib=d["ib"], sign=d["sign"]) for d in accepted if d["ia"] == ia],
                              compatible_candidates=[dict(ib=d["ib"], sign=d["sign"]) for d in details if d["ia"] == ia and d["not_excluded_by_band"]]))
    return matches, details, ambiguity


def clique_decisions(matches, assays):
    edges = {(d["a"], d["ia"], d["b"], d["ib"]): d["sign"] for d in matches}
    out = []
    for ab in matches:
        if (ab["a"], ab["b"]) != tuple(assays[:2]):
            continue
        for ac in matches:
            if (ac["a"], ac["b"], ac["ia"]) != (assays[0], assays[2], ab["ia"]):
                continue
            bc = edges.get((assays[1], ab["ib"], assays[2], ac["ib"]))
            if bc is not None and ab["sign"] * bc == ac["sign"]:
                out.append(dict(indices=[ab["ia"], ab["ib"], ac["ib"]], signs=[1, ab["sign"], ac["sign"]]))
    return out


def rank_check(rank, protocol):
    if rank["status"] == "constant_fit_feature":
        check(rank["k"] == rank["uncapped"] == 0 and not rank["cap_binding"], "constant-feature rank")
        check(rank["null_maxima"] == rank["eigenvalues"] == rank["pvalues"] == [], "constant-feature rank arrays")
        return
    check(rank["status"] == "ok", "unknown rank status")
    nulls, eigen = np.asarray(rank["null_maxima"]), np.asarray(rank["eigenvalues"])
    check(len(nulls) == protocol["rank"]["null_replicates"], "rank null count")
    check(np.isfinite(nulls).all() and np.isfinite(eigen).all(), "rank finite inputs")
    pvalues = np.array([(1 + np.count_nonzero(nulls >= e)) / (len(nulls) + 1) for e in eigen])
    selected = int(np.count_nonzero((eigen > 0) & (pvalues <= protocol["rank"]["alpha"])))
    same(rank["pvalues"], pvalues, "rank p-values")
    check(rank["uncapped"] == selected, "uncapped rank")
    check(rank["k"] == min(selected, protocol["rank"]["cap"]), "capped rank")
    check(rank["cap_binding"] == (selected > protocol["rank"]["cap"]), "rank cap flag")


def audit_assay(ar, arrays, method, assay, protocol):
    rank_check(ar["rank"], protocol)
    k, correction = ar["k"], method == "poisson_dica"
    get = lambda key: arrays[key + "_" + assay]
    mixing, unmixing = get("mixing"), get("unmixing")
    scale, covariance = get("scale"), get("covariance")
    noise, poisson = get("noise_diagonal"), get("poisson_noise_diagonal")
    check(np.all(poisson >= 0), assay + ": negative Poisson noise")
    close(noise, poisson if correction else np.zeros_like(poisson), assay + ": noise convention")
    close(np.diag(covariance) + noise, scale**2, assay + ": observed rate variance", atol=1e-8)
    outer = np.outer(scale, scale)
    standardized = np.divide(covariance, outer, out=np.zeros_like(covariance), where=outer > 0)
    close(get("standardized_covariance"), standardized, assay + ": standardized covariance")
    if ar["rank"]["status"] == "ok":
        corrected = covariance if correction else covariance - np.diag(poisson)
        close(np.linalg.eigvalsh(corrected / outer)[::-1], ar["rank"]["eigenvalues"], assay + ": rank eigenvalues", atol=1e-7)
    fit_indices, cal_indices = get("fit_indices"), get("calibration_indices")
    check(len(fit_indices) == ar["fit_cells"] and len(cal_indices) == ar["calibration_cells"], assay + ": split sizes")
    check(not set(fit_indices) & set(cal_indices), assay + ": split overlap")
    check(set(fit_indices) | set(cal_indices) == set(range(len(fit_indices) + len(cal_indices))), assay + ": split coverage")
    if ar["status"] != "ok":
        check(ar["status"] == "abstained", assay + ": unexpected source status")
        check(mixing.shape == (30, 0) and unmixing.shape == (0, 30), assay + ": abstained source dimensions")
        check(np.isnan(get("predicted_observed_covariance")).all(), assay + ": abstained prediction")
        check("fingerprint_value_" + assay not in arrays, assay + ": abstained fingerprints")
        return None
    close(unmixing @ mixing, np.eye(k), assay + ": W B identity")
    close(unmixing @ covariance @ unmixing.T, np.eye(k), assay + ": unit source covariance")
    close(get("predicted_observed_covariance"), mixing @ mixing.T + np.diag(noise), assay + ": observed covariance prediction")
    fp = {key: get("fingerprint_" + key) for key in ["value", "bootstrap", "eligible"]}
    check(fp["value"].shape == (k, 2 + 2 * len(protocol["fingerprint"]["characteristic_function_grid"])), assay + ": fingerprint shape")
    check(fp["bootstrap"].shape == (protocol["bootstrap"]["replicates"],) + fp["value"].shape, assay + ": bootstrap shape")
    if k == 0:
        check(ar["eligible_source_count"] == 0 and not fp["eligible"].any(), assay + ": zero-source eligibility")
        return fp
    transformed = np.einsum("ai,bj,ck,ijk->abc", unmixing, unmixing, unmixing, get("third_tensor"), optimize=True)
    diagonal = np.array([transformed[i, i, i] for i in range(k)])
    offdiagonal = transformed.copy()
    for i in range(k):
        offdiagonal[i, i, i] = 0
    norm = np.linalg.norm(transformed)
    residual = float(np.linalg.norm(offdiagonal) / (norm if norm else 1.))
    same(ar["third_offdiagonal_residual"], residual, assay + ": third residual")
    reconstructed = np.einsum("ir,jr,kr,r->ijk", mixing, mixing, mixing, diagonal, optimize=True)
    close(get("third_tensor_reconstruction"), reconstructed, assay + ": tensor reconstruction", atol=1e-7)
    ngrid = len(protocol["fingerprint"]["characteristic_function_grid"])
    cf = fp["value"][:, 2:2 + ngrid] + 1j * fp["value"][:, 2 + ngrid:]
    point_valid = (np.isfinite(fp["value"]).all(axis=1) & (np.asarray(ar["point_calibration_variance"]) > 0)
                   & (np.asarray(ar["point_cf_standard_error"], dtype=float) <= 0.1).all(axis=1)
                   & (np.abs(cf) <= 1.1).all(axis=1))
    same(ar["point_fingerprint_valid"], point_valid, assay + ": point fingerprint gate")
    valid_n = np.isfinite(fp["bootstrap"]).all(axis=2).sum(axis=0)
    cosine = get("fingerprint_cosine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median, low = np.nanmedian(cosine, axis=0), np.nanquantile(cosine, 0.1, axis=0)
    same(ar["valid_bootstrap_counts"], valid_n, assay + ": bootstrap counts")
    same(ar["loading_cosine_median"], median, assay + ": loading median")
    same(ar["loading_cosine_10th"], low, assay + ": loading lower quantile")
    eligible = (point_valid & (valid_n >= 26) & (median >= 0.9) & (low >= 0.7)
                & (residual <= 0.35) & (not ar["rank"]["cap_binding"]))
    close(fp["eligible"], eligible, assay + ": saved source eligibility")
    same(ar["eligible_sources"], eligible, assay + ": recorded source eligibility")
    check(ar["eligible_source_count"] == int(eligible.sum()), assay + ": eligibility count")
    reference = mixing / scale[:, None]
    for b, candidate in enumerate(get("fingerprint_bootstrap_mixing")):
        if not np.isfinite(candidate).all():
            continue
        close(get("fingerprint_bootstrap_unmixing")[b] @ candidate, np.eye(k), assay + ": bootstrap W B")
        candidate = candidate / scale[:, None]
        cos = reference.T @ candidate / np.outer(np.linalg.norm(reference, axis=0), np.linalg.norm(candidate, axis=0))
        close(np.diag(cos), cosine[b], assay + ": aligned bootstrap cosines")
        row, col = linear_sum_assignment(-np.abs(cos))
        close(np.array([np.abs(cos[row, col]).sum()]), np.array([np.trace(cos)]), assay + ": optimal bootstrap permutation")
    return fp


def audit(run_dir):
    # The final provenance file is written only after all arms and source checks.
    seal_path, summary_path = run_dir / "provenance.json", run_dir / "summary.json"
    check(seal_path.is_file() and summary_path.is_file(), "Complete summary/provenance seal is required before audit")
    seal, summary = json.loads(seal_path.read_text()), json.loads(summary_path.read_text())
    check(seal["evaluator_read"] is False, "Training seal reports evaluator access")
    for name, expected in seal["source_sha256"].items():
        check(sha(STUDY / name) == expected, "Training source hash: " + name)
    for name, expected in seal["support_sha256"].items():
        check(sha(STUDY.parent / name) == expected, "Support source hash: " + name)
    for name, expected in seal["artifacts"].items():
        check(sha(run_dir / name) == expected, "Sealed artifact hash: " + name)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    data = (STUDY / protocol["data"]).resolve()
    check(sha(data / "manifest.json") == protocol["data_manifest_sha256"] == seal["data_manifest_sha256"], "Data manifest hash")
    check(len(summary) == protocol["expected_arm_records"], "Complete arm count")
    ngrid = len(protocol["fingerprint"]["characteristic_function_grid"])
    margins = np.array(protocol["fingerprint"]["moment_margins"] + [protocol["fingerprint"]["cf_coordinate_margin"]] * (2 * ngrid))
    results = []
    for entry in summary:
        directory = run_dir / entry["path"]
        record = json.loads((directory / "result.json").read_text())
        check(entry["status"] == record["status"], "Summary status")
        if record["status"] == "failed":
            check(record.get("error") is not None, "Failed arm reason")
            results.append(dict(path=entry["path"], status="recorded_failure", error=record["error"]))
            continue
        check(sha(directory / "arrays.npz") == record["arrays_sha256"], "Arm array hash")
        required_paths = {f"public/{record['dataset']}/train/{a}.npz" for a in protocol["assays"]}
        check(set(record["input_hashes"]) == required_paths, "Only permitted training inputs")
        for name, expected in record["input_hashes"].items():
            check(sha(data / name) == expected, "Training input hash: " + name)
        with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        check(all(a.dtype != object for a in arrays.values()), "Object array in model export")
        assay_records = {a["assay"]: a for a in record["assays"]}
        check(list(assay_records) == protocol["assays"], "Assay coverage/order")
        fps = {a: audit_assay(assay_records[a], arrays, record["method"], a, protocol) for a in protocol["assays"]}
        matches, details, ambiguity = [], [], []
        for a, b in combinations(protocol["assays"], 2):
            mm, dd, aa = pair_decisions(a, b, fps[a], fps[b], margins, ngrid)
            matches.extend(mm)
            details.extend(dd)
            ambiguity.extend(aa)
            cross = np.zeros((30, 30))
            for d in mm:
                cross += d["sign"] * arrays["mixing_" + a][:, d["ia"], None] * arrays["mixing_" + b][None, :, d["ib"]]
            close(arrays[f"predicted_cross_covariance_{a}_{b}"], cross, "Candidate cross covariance")
        cliques = clique_decisions(matches, protocol["assays"])
        for key, expected in [("pair_matches", matches), ("pair_diagnostics", details), ("ambiguity_sets", ambiguity), ("all_assay_matches", cliques)]:
            same(record[key], expected, entry["path"] + "/" + key)
        same(entry["ranks"], [a["k"] for a in record["assays"]], "Summary ranks")
        same(entry["eligible_sources"], [a.get("eligible_source_count", 0) for a in record["assays"]], "Summary eligible counts")
        check(entry["pair_matches"] == len(matches) and entry["all_assay_matches"] == len(cliques), "Summary match counts")
        check(entry["graph_status"] == record["graph_status"], "Summary graph status")
        if len(cliques) < 2:
            check(record["graph_status"] == "fewer_than_two_all_assay_candidates", "Graph extraction eligibility")
            check("exploratory_native_A" not in arrays, "Unexpected graph coefficients")
        results.append(dict(path=entry["path"], status="passed", ranks=entry["ranks"], eligible_sources=entry["eligible_sources"],
                            pair_matches=len(matches), all_assay_matches=len(cliques)))
    return dict(status="passed", source_sha256=sha(Path(__file__)), sealed_provenance_sha256=sha(seal_path),
                protocol_sha256=sha(STUDY / "protocol.json"), arms=len(results), results=results,
                method="Independent NumPy/SciPy replay of saved arrays and declared numerical gates; no runner/estimator imports, source refits, test files or evaluator references.",
                limitations="Checks execution and arithmetic only; does not establish biological assumptions, full source-law equality, or causal identifiability.")


def self_test():
    ngrid, margins = 3, np.array([0.25, 0.5] + [0.1] * 6)
    value = np.array([[1., 0., .98, .95, .9, .2, .15, .1]])
    fp = dict(value=value, bootstrap=np.repeat(value[None], 32, axis=0), eligible=np.array([True]))
    matches, _, _ = pair_decisions("A", "B", fp, fp, margins, ngrid)
    check(len(matches) == 1 and matches[0]["sign"] == 1, "Self-test positive sign")
    minus = dict(value=sign_change(value, -1, ngrid), bootstrap=sign_change(fp["bootstrap"], -1, ngrid), eligible=np.array([True]))
    check(pair_decisions("A", "B", fp, minus, margins, ngrid)[0][0]["sign"] == -1, "Self-test negative sign")
    duplicate = dict(value=np.repeat(value, 2, axis=0), bootstrap=np.repeat(fp["bootstrap"], 2, axis=1), eligible=np.ones(2, bool))
    check(pair_decisions("A", "B", fp, duplicate, margins, ngrid)[0] == [], "Self-test ambiguity")
    check(pair_decisions("A", "B", None, fp, margins, ngrid) == ([], [], []), "Self-test abstention")
    check(clique_decisions([dict(a="A", b="B", ia=0, ib=1, sign=-1), dict(a="A", b="C", ia=0, ib=2, sign=1), dict(a="B", b="C", ia=1, ib=2, sign=-1)], ["A", "B", "C"]) == [dict(indices=[0, 1, 2], signs=[1, -1, 1])], "Self-test clique")
    print("PASS: signed equivalence, mutual uniqueness, abstention and sign-consistent cliques")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=STUDY / "runs_v1")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("audit_count_fits.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = audit(args.runs.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "arms", "sealed_provenance_sha256"]}))


if __name__ == "__main__":
    main()
