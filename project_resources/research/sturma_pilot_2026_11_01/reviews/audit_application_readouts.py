"""Replay sealed application readouts, without fitting or importing the runner."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]
ASSAYS = ["RNA", "H3K27ac", "H3K4me3"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True)


def corr(a, b):
    if a.shape[1] == 0:
        return np.empty((0, b.shape[1]))
    matrix = np.corrcoef(np.column_stack((a, b)), rowvar=False)
    return matrix[:a.shape[1], a.shape[1]:]


def main():
    app = BASE / "application_v1"
    summary = read(app / "summary.json")
    protocol = read(BASE / "application_protocol.json")
    if summary["completed_records"] != protocol["expected_fits"]:
        raise RuntimeError("Wait for every frozen application record before replay")
    provenance = read(app / "provenance.json")
    assert sha(BASE / "apply_native.py") == provenance["runner_sha256"]
    assert sha(BASE / "upstream/src/learner.py") == provenance["learner_sha256"]
    assert sha(BASE / "data_pilot_v1/manifest.json") == provenance["data_manifest_sha256"]
    assert sha(BASE / "application_protocol.json") == provenance["protocol_sha256"]
    data = read(BASE / "data_pilot_v1/manifest.json")
    results = []
    for path in sorted(app.glob("*/result.json")):
        record = read(path)
        name = record["dataset"]
        representation = record["representation"]
        arrays_path = path.parent / "arrays.npz"
        assert sha(arrays_path) == record["arrays_sha256"]
        for relative, expected in record["input_hashes"].items():
            assert sha(BASE / relative) == expected
        item = {"run": path.parent.name, "status": record["status"],
                "source_result_sha256": sha(path), "arrays_sha256": sha(arrays_path),
                "shared_source_count": record.get("shared_source_count"),
                "warning_count": len(record["warnings"]),
                "graph_status": record.get("graph_status")}
        if "evaluation" not in record:
            item["metric_replay"] = "unavailable; native failure retained"
            results.append(item)
            continue
        test = []
        for assay in ASSAYS:
            source = data["datasets"][name]["assays"][assay]["cohorts"]["test"]
            assert sha(REPO / source["path"]) == source["sha256"]
            with np.load(REPO / source["path"], allow_pickle=False) as z:
                test.append({k: z[k].copy() for k in z.files})
        with np.load(arrays_path, allow_pickle=False) as arrays:
            count = record["shared_source_count"]
            factors = np.asarray(record["joint_factors"], dtype=int).reshape(-1, 3)
            signs = np.asarray(record["joint_signs"], dtype=int).reshape(-1, 3)
            for i, assay in enumerate(ASSAYS):
                np.testing.assert_array_equal(arrays["features_" + assay], test[i]["feature_ids"])
                sources = np.linalg.lstsq(arrays["mixing_" + assay],
                    (test[i][representation] - arrays["train_mean_" + assay]).T,
                    rcond=None)[0].T
                expected = sources[:, factors[:, i]] * signs[:, i]
                close(arrays["heldout_shared_sources_" + assay], expected)
            ref_path = REPO / data["evaluator"][name]["path"]
            assert sha(ref_path) == record["evaluation"]["reference_sha256"]
            with np.load(ref_path, allow_pickle=False) as ref:
                if name == "scmtr_seq_s1":
                    # Construct all joint covariance blocks at once, including
                    # private variances, independently of the runner formula.
                    C = arrays["joint_mixing"] @ arrays["joint_mixing"].T
                    scales = np.sqrt(np.diag(C))
                    R = C / np.outer(scales, scales)
                    pairs = []
                    for k, (i, j) in enumerate(itertools.combinations(range(3), 2)):
                        prediction = R[30*i:30*(i+1), 30*j:30*(j+1)]
                        reference = corr(ref[representation][:, 30*i:30*(i+1)],
                                         ref[representation][:, 30*j:30*(j+1)])
                        close(prediction, arrays[f"predicted_correlation_{i}_{j}"])
                        close(reference, arrays[f"reference_correlation_{i}_{j}"])
                        close(C[30*i:30*(i+1),30*j:30*(j+1)], arrays[f"predicted_covariance_{i}_{j}"])
                        valid = np.isfinite(prediction) & np.isfinite(reference)
                        e = record["evaluation"]["pairs"][k]
                        assert int(valid.sum()) == e["valid_feature_pairs"]
                        rmse = float(np.sqrt(np.mean(np.square(prediction[valid] - reference[valid]))))
                        zero = float(np.sqrt(np.mean(np.square(reference[valid]))))
                        close(rmse, e["correlation_rmse"])
                        close(zero, e["zero_correlation_rmse"])
                        close(zero-rmse, e["rmse_improvement_over_zero"])
                        close(np.mean(np.abs(prediction[valid])), e["predicted_abs_mean"])
                        close(np.mean(np.abs(reference[valid])), e["reference_abs_mean"])
                        if np.std(prediction[valid]) > 0 and np.std(reference[valid]) > 0:
                            rho = np.corrcoef(rankdata(prediction[valid]), rankdata(reference[valid]))[0,1]
                            close(rho, e["flattened_spearman"])
                        else:
                            assert e["flattened_spearman"] is None
                        pairs.append({"assays": [ASSAYS[i], ASSAYS[j]],
                                      "correlation_rmse": rmse, "zero_correlation_rmse": zero})
                    item["pairs"] = pairs
                else:
                    for i, assay in enumerate(ASSAYS):
                        np.testing.assert_array_equal(test[i]["cell_ids"], ref["ids_" + assay])
                        correlation = corr(arrays["heldout_shared_sources_" + assay], ref["state_" + assay])
                        close(correlation, arrays["source_state_correlation_" + assay])
                        e = record["evaluation"]["assays"][i]
                        if count:
                            close(correlation, e["source_by_state_correlation"])
                            close(np.max(np.abs(correlation), axis=0), e["max_abs_source_correlation_per_state"])
                        else:
                            assert e["max_abs_source_correlation_per_state"] is None
            item["metric_replay"] = "passed"
        results.append(item)
    assert len(results) == protocol["expected_fits"]
    output = {"status": "passed", "reviewer": "scmtr_data_pilot",
              "imported_fit_runner": False, "new_fits": 0,
              "summary_sha256": sha(app / "summary.json"),
              "provenance_sha256": sha(app / "provenance.json"),
              "script_sha256": sha(Path(__file__)), "results": results}
    path = HERE / "application_readout_audit.json"
    with path.open("x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()
