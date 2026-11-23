"""Explore all real assay pairs using the previously fitted native ICA sources."""
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from apply_native import ASSAYS, DATA, STUDY, RecordedMDCRL, cross_correlation, encode, model_cross_correlation, read_inputs, save_json, sha


def main():
    out = STUDY / "pairwise_v1"
    out.mkdir(exist_ok=False)
    parents = sorted((STUDY / "application_v1").glob("scmtr_seq_s1__*/result.json"))
    assert len(parents) == 6
    protocol = STUDY / "pairwise_protocol.json"
    save_json(out / "provenance.json", {"protocol_sha256": sha(protocol), "runner_sha256": sha(Path(__file__)),
        "shared_adapter_sha256": sha(STUDY / "apply_native.py"), "parents": {str(p.relative_to(STUDY)): sha(p) for p in parents}})
    results = []
    with np.load(DATA / "evaluator/scmtr_seq_s1/paired_reference.npz", allow_pickle=False) as ref:
        pair_ids = [ref[f"paired_indices_{a}"].copy() for a in ASSAYS]
    for parent_path in parents:
        parent = json.loads(parent_path.read_text())
        assert parent["status"] == "completed"
        parent_arrays = parent_path.parent / "arrays.npz"
        assert sha(parent_arrays) == parent["arrays_sha256"]
        inputs = read_inputs("scmtr_seq_s1", parent["representation"])
        assert inputs["hashes"] == parent["input_hashes"]
        train = inputs["train"]
        if parent["control"] == "independent_columns":
            rng = np.random.default_rng(1000)
            train = [np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])]) for x in train]
        with np.load(parent_arrays, allow_pickle=False) as z:
            mixings = [z[f"mixing_{a}"].copy() for a in ASSAYS]
            means = [z[f"train_mean_{a}"].copy() for a in ASSAYS]
        sources = [encode(x, mean, m) for x, mean, m in zip(train, means, mixings)]
        for source in sources:
            assert np.allclose(source.mean(axis=0), 0, atol=1e-7)
            assert np.allclose(source.std(axis=0), 1, atol=1e-7)
        for a, b in combinations(range(3), 2):
            run = out / f"{parent_path.parent.name}__{ASSAYS[a]}__{ASSAYS[b]}"
            run.mkdir()
            record = {"representation": parent["representation"], "ica_seed": parent["ica_seed"], "control": parent["control"],
                "assays": [ASSAYS[a], ASSAYS[b]], "parent_result": str(parent_path.relative_to(STUDY)),
                "original_three_assay_adjusted_alpha": parent["adjusted_ks_alpha"]}
            model = RecordedMDCRL(measure="ks-test", alpha=0.05, gamma=0.2)
            model.nr_env = 2
            model.mixings = [mixings[a], mixings[b]]
            model.indep_comps = [sources[a], sources[b]]
            arrays = {}
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                try:
                    model.match()
                    model.get_joint_mixing()
                    record.update(shared_source_count=model.nr_joint, pair_adjusted_alpha=model.adj_alpha,
                        joint_factors=model.joint_factors, joint_signs=model.joint_signs)
                    similarity = model.signed_similarity_matrix(*model.indep_comps)
                    record["source_count_at_original_global_alpha"] = len(model.maximum_matching(similarity, parent["adjusted_ks_alpha"]))
                    if model.nr_joint:
                        try:
                            model.get_joint_graph()
                            arrays["native_A"] = model.A
                            record["graph_status"] = "finite_native_coefficient_matrix" if np.isfinite(model.A).all() else "nonfinite_native_coefficients"
                            record["native_A"] = model.A
                            if hasattr(model, "recorded_order"):
                                record["native_graph_row_column_order"] = model.recorded_order
                            if hasattr(model, "recorded_signs"):
                                record["native_graph_column_signs"] = model.recorded_signs
                        except Exception as exc:
                            record.update(graph_status="native_exception", graph_error=f"{type(exc).__name__}: {exc}")
                    else:
                        record["graph_status"] = "no_shared_sources"
                    factors = np.asarray(model.joint_factors, dtype=int).reshape(-1, 2)
                    signs = np.asarray(model.joint_signs, dtype=int).reshape(-1, 2)
                    shared = [m[:, factors[:, i]] * signs[:, i] for i, m in enumerate(model.mixings)]
                    cov, pred = model_cross_correlation(*model.mixings, *shared)
                    reference = cross_correlation(inputs["test"][a][pair_ids[a]], inputs["test"][b][pair_ids[b]])
                    valid = np.isfinite(reference) & np.isfinite(pred)
                    assert valid.any()
                    record.update(valid_feature_pairs=int(valid.sum()), total_feature_pairs=pred.size,
                        correlation_rmse=float(np.sqrt(np.mean((pred[valid] - reference[valid])**2))),
                        zero_correlation_rmse=float(np.sqrt(np.mean(reference[valid]**2))))
                    record["rmse_improvement_over_zero"] = record["zero_correlation_rmse"] - record["correlation_rmse"]
                    record["status"] = "completed"
                    arrays.update(joint_mixing=model.joint_mixing, predicted_covariance=cov, predicted_correlation=pred, reference_correlation=reference)
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                record["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
            np.savez_compressed(run / "arrays.npz", **arrays)
            record["arrays_sha256"] = sha(run / "arrays.npz")
            save_json(run / "result.json", record)
            results.append(record)
            save_json(out / "summary.json", {"expected_comparisons": 18, "completed_records": len(results), "results": results})
            print(run.name, record["status"], record.get("shared_source_count"), flush=True)
    assert len(results) == 18


if __name__ == "__main__":
    main()
