"""Apply the pinned native estimator to frozen small count panels; score separately."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "upstream"))
from src.learner import LinearMDCRL

ASSAYS = ["RNA", "H3K27ac", "H3K4me3"]
DATA = STUDY / "data_pilot_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        return x
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


class RecordedMDCRL(LinearMDCRL):
    """Only records the native graph stage's intermediate matrices and ordering."""
    def get_pure_children(self, matrix):
        self.recorded_children = super().get_pure_children(matrix)
        return self.recorded_children

    def search_order_noisy(self, matrix):
        result = super().search_order_noisy(matrix)
        self.recorded_order = result
        if result[0] is not None:
            self.recorded_signs = np.sign(np.diag(matrix[result[0], :][:, result[1]]))
        return result


def encode(x, mean, mixing):
    return (x - mean) @ np.linalg.pinv(mixing).T


def cross_correlation(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    denominator = np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0))
    return np.divide(x.T @ y, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def model_cross_correlation(mixing_a, mixing_b, shared_a, shared_b):
    covariance = shared_a @ shared_b.T
    denominator = np.outer(np.linalg.norm(mixing_a, axis=1), np.linalg.norm(mixing_b, axis=1))
    correlation = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)
    return covariance, correlation


def read_inputs(dataset, representation):
    result = {"train": [], "test": [], "test_ids": [], "features": [], "hashes": {}}
    for assay in ASSAYS:
        feature_ids = None
        for cohort in ["train", "test"]:
            path = DATA / "public" / dataset / cohort / f"{assay}.npz"
            with np.load(path, allow_pickle=False) as z:
                assert set(z.files) == {"raw_counts", "log1p_cp10k", "library_depth", "cell_ids", "feature_ids"}
                x = z[representation].astype(float)
                assert x.ndim == 2 and x.shape[1] == 30 and np.isfinite(x).all()
                assert (z["raw_counts"] >= 0).all()
                assert np.equal(z["raw_counts"], np.floor(z["raw_counts"])).all()
                assert len(set(z["cell_ids"])) == len(x)
                if cohort == "train":
                    assert (x.std(axis=0) > 0).all()
                    feature_ids = z["feature_ids"].copy()
                    result["features"].append(feature_ids)
                else:
                    assert np.array_equal(feature_ids, z["feature_ids"])
                    result["test_ids"].append(z["cell_ids"].copy())
                result[cohort].append(x)
            result["hashes"][str(path.relative_to(STUDY))] = sha(path)
    return result


def evaluate(model, inputs, dataset, representation, arrays):
    shared = [m[:, model.joint_factors[:, e]] * model.joint_signs[:, e]
              for e, m in enumerate(model.mixings)] if model.nr_joint else [m[:, :0] for m in model.mixings]
    output = {"interpretation": "Joint-moment and source/state diagnostics; no original-feature causal truth score."}
    if dataset == "scmtr_seq_s1":
        ref_path = DATA / "evaluator" / dataset / "paired_reference.npz"
        with np.load(ref_path, allow_pickle=False) as ref:
            paired = []
            for i, assay in enumerate(ASSAYS):
                alias = {"RNA": "RNA", "H3K27ac": "ac", "H3K4me3": "me3"}[assay]
                key = f"paired_indices_{assay}" if f"paired_indices_{assay}" in ref else f"paired_indices_{alias}"
                idx = ref[key]
                assert np.issubdtype(idx.dtype, np.integer) and len(set(idx.tolist())) == len(idx)
                paired.append(inputs["test"][i][idx])
            assert len(set(len(x) for x in paired)) == 1
        output["reference_sha256"] = sha(ref_path)
        output["paired_test_cells"] = len(paired[0])
        output["pairs"] = []
        for a, b in combinations(range(3), 2):
            cov, pred = model_cross_correlation(model.mixings[a], model.mixings[b], shared[a], shared[b])
            reference = cross_correlation(paired[a], paired[b])
            valid = np.isfinite(pred) & np.isfinite(reference)
            arrays[f"predicted_correlation_{a}_{b}"] = pred
            arrays[f"reference_correlation_{a}_{b}"] = reference
            arrays[f"predicted_covariance_{a}_{b}"] = cov
            entry = {"assays": [ASSAYS[a], ASSAYS[b]], "valid_feature_pairs": int(valid.sum()), "total_feature_pairs": pred.size}
            if valid.any():
                entry.update(correlation_rmse=float(np.sqrt(np.mean((pred[valid] - reference[valid])**2))),
                             zero_correlation_rmse=float(np.sqrt(np.mean(reference[valid]**2))),
                             predicted_abs_mean=float(np.mean(np.abs(pred[valid]))),
                             reference_abs_mean=float(np.mean(np.abs(reference[valid]))))
                entry["rmse_improvement_over_zero"] = entry["zero_correlation_rmse"] - entry["correlation_rmse"]
                entry["flattened_spearman"] = float(spearmanr(pred[valid], reference[valid]).statistic) if np.std(pred[valid]) > 0 and np.std(reference[valid]) > 0 else None
            output["pairs"].append(entry)
    else:
        ref_path = DATA / "evaluator" / dataset / "state_reference.npz"
        output["reference_sha256"] = sha(ref_path)
        output["assays"] = []
        with np.load(ref_path, allow_pickle=False) as ref:
            for i, assay in enumerate(ASSAYS):
                assert np.array_equal(inputs["test_ids"][i], ref[f"ids_{assay}"])
                sources = arrays[f"heldout_shared_sources_{assay}"]
                state = ref[f"state_{assay}"]
                corr = cross_correlation(sources, state)
                arrays[f"source_state_correlation_{assay}"] = corr
                output["assays"].append({"assay": assay, "n_test": len(state), "matched_source_count": model.nr_joint,
                    "state_dimensions": state.shape[1], "source_by_state_correlation": corr,
                    "max_abs_source_correlation_per_state": np.max(np.abs(corr), axis=0) if model.nr_joint else None})
    return output


def fit_one(dataset, representation, seed, control, output):
    inputs = read_inputs(dataset, representation)
    train = [x.copy() for x in inputs["train"]]
    if control:
        rng = np.random.default_rng(1000)
        train = [np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])]) for x in train]
    random.seed(seed)
    np.random.seed(seed)
    model = RecordedMDCRL(measure="ks-test", alpha=0.05, gamma=0.2)
    model.data = train
    record = {"dataset": dataset, "representation": representation, "ica_seed": seed,
              "control": "independent_columns" if control else "original", "input_hashes": inputs["hashes"],
              "train_shapes": [list(x.shape) for x in train], "status": "started"}
    arrays = {}
    start = time.monotonic()
    stage = "get_sources"
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        try:
            model.get_sources()
            record["estimated_domain_ranks"] = [m.shape[1] for m in model.mixings]
            stage = "match"
            model.match()
            stage = "get_joint_mixing"
            model.get_joint_mixing()
            record.update(shared_source_count=model.nr_joint, adjusted_ks_alpha=model.adj_alpha,
                          joint_factors=model.joint_factors, joint_signs=model.joint_signs)
            stage = "get_joint_graph"
            if model.nr_joint:
                try:
                    model.get_joint_graph()
                    record["graph_status"] = "finite_native_coefficient_matrix" if np.isfinite(model.A).all() else "nonfinite_native_coefficients"
                    arrays["native_A"] = model.A
                    record["native_A"] = model.A
                    record["native_A_max_abs"] = float(np.max(np.abs(model.A)))
                except Exception as exc:
                    record["graph_status"] = "native_exception"
                    record["graph_error"] = f"{type(exc).__name__}: {exc}"
                if hasattr(model, "recorded_order"):
                    record["native_graph_row_column_order"] = model.recorded_order
                if hasattr(model, "recorded_signs"):
                    record["native_graph_column_signs"] = model.recorded_signs
                if hasattr(model, "recorded_children"):
                    record["native_selected_rows_condition_number"] = float(np.linalg.cond(model.recorded_children))
            else:
                record["graph_status"] = "no_shared_sources"
            arrays["joint_mixing"] = model.joint_mixing
            stage = "heldout_encoder"
            model.joint_factors = np.asarray(model.joint_factors, dtype=int).reshape(-1, 3)
            model.joint_signs = np.asarray(model.joint_signs, dtype=int).reshape(-1, 3)
            parity = []
            for i, assay in enumerate(ASSAYS):
                mean = train[i].mean(axis=0)
                inferred = encode(train[i], mean, model.mixings[i])
                delta = float(np.max(np.abs(inferred - model.indep_comps[i])))
                parity.append(delta)
                assert np.allclose(inferred, model.indep_comps[i], atol=1e-7, rtol=1e-7), (assay, delta)
                sources = encode(inputs["test"][i], mean, model.mixings[i])
                arrays[f"heldout_shared_sources_{assay}"] = sources[:, model.joint_factors[:, i]] * model.joint_signs[:, i]
                arrays[f"train_mean_{assay}"] = mean
                arrays[f"mixing_{assay}"] = model.mixings[i]
                arrays[f"features_{assay}"] = inputs["features"][i]
            record["train_encoder_max_abs_parity_errors"] = parity
            stage = "evaluation"
            record["evaluation"] = evaluate(model, inputs, dataset, representation, arrays)
            record["status"] = "completed_with_native_graph_failure" if record["graph_status"] == "native_exception" else "completed"
        except Exception as exc:
            record.update(status="failed", failure_stage=stage, error=f"{type(exc).__name__}: {exc}")
        record["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    record["elapsed_seconds"] = time.monotonic() - start
    np.savez_compressed(output / "arrays.npz", **arrays)
    record["arrays_sha256"] = sha(output / "arrays.npz")
    save_json(output / "result.json", record)
    return record


def main():
    protocol = json.loads((STUDY / "application_protocol.json").read_text())
    out = STUDY / "application_v1"
    out.mkdir(exist_ok=False)
    provenance = {"protocol_sha256": sha(STUDY / "application_protocol.json"), "runner_sha256": sha(Path(__file__)),
                  "data_manifest_sha256": sha(DATA / "manifest.json"),
                  "learner_sha256": sha(STUDY / "upstream/src/learner.py"), "upstream_commit": protocol["upstream_commit"]}
    save_json(out / "provenance.json", provenance)
    results = []
    for dataset in protocol["datasets"]:
        for representation in protocol["inputs"]:
            for seed, control in [(0, False), (1, False), (0, True)]:
                name = f"{dataset}__{representation}__seed{seed}" + ("__independent_columns" if control else "")
                run = out / name
                run.mkdir()
                record = fit_one(dataset, representation, seed, control, run)
                results.append(record)
                save_json(out / "summary.json", {"expected_fits": 18, "completed_records": len(results), "results": results})
                print(name, record["status"], "shared", record.get("shared_source_count"), "seconds", round(record["elapsed_seconds"], 2), flush=True)
    assert len(results) == protocol["expected_fits"]


if __name__ == "__main__":
    main()
