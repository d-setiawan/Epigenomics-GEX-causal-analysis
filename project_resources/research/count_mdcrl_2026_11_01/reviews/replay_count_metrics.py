"""Independent heldout readout replay; imports no fitting or evaluation module."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
MAX_DIFFERENCE = 0.0


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    global MAX_DIFFERENCE
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11, equal_nan=True)
    valid = np.isfinite(actual) & np.isfinite(expected)
    if valid.any():
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, float(np.max(np.abs(actual[valid]-expected[valid]))))


def rmse(actual, expected, mask=None):
    delta = np.asarray(actual)-np.asarray(expected)
    if mask is not None:
        delta = delta[mask]
    return np.sqrt(np.mean(delta**2))


def check_errors(report, prediction, truth, mask=None):
    use = np.ones(np.shape(truth), dtype=bool) if mask is None else mask
    assert report["entries"] == int(use.sum())
    if not use.any():
        assert report["status"] == "unscorable" and report["rmse"] is None
        return
    assert report["status"] == "ok"
    close(report["rmse"], rmse(prediction, truth, use))
    close(report["mae"], np.mean(np.abs((prediction-truth)[use])))


def main():
    protocol = json.loads((STUDY/"protocol.json").read_text())
    data = (STUDY/protocol["data"]).resolve()
    manifest = json.loads((data/"manifest.json").read_text())
    assert sha(data/"manifest.json") == protocol["data_manifest_sha256"]
    for folder in ["runs_v1", "evaluation_v1"]:
        seal = json.loads((STUDY/folder/"provenance.json").read_text())
        for name, digest in seal["artifacts"].items():
            assert sha(STUDY/folder/name) == digest, (folder, name)
    fit_summary = json.loads((STUDY/"runs_v1/summary.json").read_text())
    reports = json.loads((STUDY/"evaluation_v1/summary.json").read_text())
    assert len(fit_summary) == len(reports) == 36
    assays = protocol["assays"]
    public, reference = {}, {}
    for dataset in protocol["datasets"]:
        for assay in assays:
            path = data/"public"/dataset/"test"/(assay+".npz")
            assert sha(path) == manifest["datasets"][dataset]["assays"][assay]["cohorts"]["test"]["sha256"]
            public[dataset,assay] = load(path)
        item = manifest["evaluator"][dataset]
        path = REPO/item["path"]
        assert sha(path) == item["sha256"]
        reference[dataset] = load(path)

    bootstrap_cache = {}
    counts = {"arms": 0, "within_assay_metrics": 0, "real_pair_metrics": 0,
              "state_correlation_matrices": 0, "real_bootstrap_replicates_checked": 0}
    for report in reports:
        name, dataset = report["arm"], report["dataset"]
        fit = json.loads((STUDY/"runs_v1"/name/"result.json").read_text())
        model = load(STUDY/"runs_v1"/name/"arrays.npz")
        measured = load(STUDY/"evaluation_v1"/name/"arrays.npz")
        assert fit["status"] == "completed" and report["evaluation_status"] == "complete"
        assert not fit["pair_matches"] and not fit["all_assay_matches"]
        assert report["pair_match_count"] == report["all_assay_match_count"] == 0
        assert report["causal_recovery_claim"] is False
        rates = {}
        for assay in assays:
            source = public[dataset,assay]
            y = source["raw_counts"].astype(float)
            normalizer = fit["exposure_normalization"][assay]
            exposure = (np.ones(len(y)) if fit["exposure"] == "unit" else
                        (source["library_depth"]-np.sum(y,axis=1))/normalizer)
            assert np.all(exposure > 0)
            rates[assay] = y/exposure[:,None]
            truth = np.cov(rates[assay], rowvar=False, bias=True)
            loading = model[f"mixing_{assay}"]
            predicted = loading@loading.T + np.diag(model[f"noise_diagonal_{assay}"])
            close(predicted, model[f"predicted_observed_covariance_{assay}"])
            close(truth, measured[f"observed_covariance_{assay}"])
            scale = model[f"scale_{assay}"]
            denominator = np.multiply.outer(scale, scale)
            truth, predicted = truth/denominator, predicted/denominator
            close(truth, measured[f"observed_standardized_covariance_{assay}"])
            close(predicted, measured[f"predicted_standardized_covariance_{assay}"])
            record = report["within_assay"][assay]
            off = ~np.eye(30,dtype=bool)
            check_errors(record["model_full"], predicted, truth)
            check_errors(record["model_offdiagonal"], predicted, truth, off)
            check_errors(record["diagonal_training_baseline_full"], np.eye(30), truth)
            check_errors(record["diagonal_training_baseline_offdiagonal"], np.eye(30), truth, off)
            close(record["standardized_mean_rmse"], np.sqrt(np.mean(
                ((rates[assay].mean(0)-model[f"mean_{assay}"])/scale)**2)))
            counts["within_assay_metrics"] += 1

        if dataset == "scmtr_seq_s1":
            ref = reference[dataset]
            paired = np.concatenate([rates[a][ref[f"paired_indices_{a}"]] for a in assays],axis=1)
            joint = np.cov(paired,rowvar=False,bias=True)
            key = (fit["exposure"], tuple(fit["exposure_normalization"][a] for a in assays))
            if key not in bootstrap_cache:
                indices = np.random.default_rng(3000).integers(0,595,size=(200,595))
                bootstrap_cache[key] = np.asarray([np.cov(paired[ix],rowvar=False,bias=True) for ix in indices])
            for ia, ib in itertools.combinations(range(3),2):
                a,b = assays[ia],assays[ib]
                sa,sb = slice(30*ia,30*(ia+1)),slice(30*ib,30*(ib+1))
                truth = joint[sa,sb]
                predicted = model[f"predicted_cross_covariance_{a}_{b}"]
                assert np.array_equal(predicted,np.zeros((30,30)))
                close(truth,measured[f"paired_observed_cross_covariance_{a}_{b}"])
                denominator = np.multiply.outer(model[f"scale_{a}"],model[f"scale_{b}"])
                standardized = truth/denominator
                pair = report["real_cross_assay"][f"{a}__{b}"]
                assert pair["zero_predictor_due_to_no_accepted_sources"]
                for arm in ["model","zero"]:
                    check_errors(pair["primary"][arm],np.zeros((30,30)),standardized)
                assert pair["primary"]["rmse_improvement_over_zero"] == 0
                samples = bootstrap_cache[key][:,sa,sb]/denominator
                bootstrap_rmse = np.sqrt(np.mean(samples**2,axis=(1,2)))
                close(bootstrap_rmse,measured[f"bootstrap_model_rmse_{a}_{b}"])
                close(bootstrap_rmse,measured[f"bootstrap_zero_rmse_{a}_{b}"])
                close(measured[f"bootstrap_rmse_improvement_over_zero_{a}_{b}"],np.zeros(200))
                for arm in ["model","zero"]:
                    close(pair["primary"]["paired_cell_bootstrap"][f"{arm}_rmse_percentiles"],
                          np.quantile(bootstrap_rmse,[.025,.5,.975]))
                var_a,var_b = np.diag(model[f"predicted_observed_covariance_{a}"]),np.diag(model[f"predicted_observed_covariance_{b}"])
                obs_a,obs_b = np.diag(joint)[sa],np.diag(joint)[sb]
                mask = ((var_a>0)&np.isfinite(var_a)&(obs_a>0))[:,None] & ((var_b>0)&np.isfinite(var_b)&(obs_b>0))[None,:]
                empirical = truth/np.sqrt(np.multiply.outer(obs_a,obs_b))
                close(empirical,measured[f"secondary_test_correlation_{a}_{b}"])
                np.testing.assert_array_equal(mask,measured[f"secondary_valid_mask_{a}_{b}"])
                for arm in ["model","zero"]:
                    check_errors(pair["secondary_pearson"][arm],np.zeros((30,30)),empirical,mask)
                counts["real_pair_metrics"] += 1
                counts["real_bootstrap_replicates_checked"] += 200
        else:
            for assay in assays:
                ref = reference[dataset]
                np.testing.assert_array_equal(public[dataset,assay]["cell_ids"],ref[f"ids_{assay}"])
                score = (rates[assay]-model[f"mean_{assay}"])@model[f"unmixing_{assay}"].T
                k = score.shape[1]
                correlation = (np.corrcoef(np.concatenate([score,ref[f"state_{assay}"]],axis=1),rowvar=False)[:k,k:]
                               if k else np.empty((0,2)))
                close(correlation,measured[f"state_correlation_all_sources_{assay}"])
                state_report = report["simulation_state"][assay]
                expected = np.max(np.abs(correlation),axis=0) if k else [None,None]
                if k:
                    close(expected,state_report["all_sources"]["max_absolute_correlation_per_state"])
                else:
                    assert state_report["all_sources"]["max_absolute_correlation_per_state"] == [None,None]
                assert state_report["all_sources"]["score_count"] == k
                assert state_report["pair_matched_sources"]["score_count"] == 0
                assert state_report["all_assay_matched_sources"]["score_count"] == 0
                counts["state_correlation_matrices"] += 1
        counts["arms"] += 1

    result = {"status":"passed","imports_fitter_or_evaluator":False,"new_fits":0,
              "new_datasets":0,"counts":counts,"maximum_absolute_numeric_difference":MAX_DIFFERENCE,
              "checks":["all sealed artifact hashes","within-assay same-scale full/offdiagonal moment errors and diagonal baseline",
                        "all real primary and secondary covariance/correlation metrics","all real paired-bootstrap errors and percentiles",
                        "all simulated frozen-score state correlations","zero matches and no causal-recovery claim"],
              "hashes":{p.name:sha(p) for p in [Path(__file__),STUDY/"protocol.json"]},
              "fit_provenance_sha256":sha(STUDY/"runs_v1/provenance.json"),
              "evaluation_provenance_sha256":sha(STUDY/"evaluation_v1/provenance.json")}
    path = Path(__file__).with_name("count_metric_audit.json")
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
