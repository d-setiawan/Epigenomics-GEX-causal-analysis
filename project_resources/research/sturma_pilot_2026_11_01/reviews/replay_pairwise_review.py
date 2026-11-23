"""Independent bounded replay of saved Sturma pairwise readouts; no model fits."""
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ks_2samp

STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / "pairwise_v1"
ASSAYS = ["RNA", "H3K27ac", "H3K4me3"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def correlation(x, y):
    cov = np.cov(np.concatenate([x, y], axis=1), rowvar=False, bias=True)
    sd = np.sqrt(np.diag(cov))
    p = x.shape[1]
    denominator = np.outer(sd[:p], sd[p:])
    return np.divide(cov[:p, p:], denominator,
                     out=np.full((p, y.shape[1]), np.nan), where=denominator > 0)


def matching(similarity, alpha):
    row = np.argmax(similarity, axis=1)
    col = np.argmax(similarity, axis=0)
    return [(i, int(j)) for i, j in enumerate(row)
            if col[j] == i and similarity[i, j] >= alpha]


def replay(index):
    provenance = load_json(OUT / "provenance.json")
    assert sha(STUDY / "pairwise_protocol.json") == provenance["protocol_sha256"]
    assert sha(STUDY / "pairwise_followup.py") == provenance["runner_sha256"]
    assert sha(STUDY / "apply_native.py") == provenance["shared_adapter_sha256"]
    parents = sorted(provenance["parents"])
    assert len(parents) == 6
    parent_rel = parents[index]
    parent_path = STUDY / parent_rel
    assert sha(parent_path) == provenance["parents"][parent_rel]
    parent = load_json(parent_path)
    assert parent["status"] == "completed"
    assert sha(parent_path.parent / "arrays.npz") == parent["arrays_sha256"]
    ref_path = STUDY / "data_pilot_v1/evaluator/scmtr_seq_s1/paired_reference.npz"
    assert sha(ref_path) == parent["evaluation"]["reference_sha256"]
    with np.load(ref_path, allow_pickle=False) as z:
        ref = {k: z[k].copy() for k in z.files}
    trains, tests = [], []
    for assay in ASSAYS:
        arrays = []
        for cohort in ["train", "test"]:
            path = STUDY / "data_pilot_v1/public/scmtr_seq_s1" / cohort / (assay + ".npz")
            assert sha(path) == parent["input_hashes"][str(path.relative_to(STUDY))]
            with np.load(path, allow_pickle=False) as z:
                arrays.append(z[parent["representation"]].astype(float))
        trains.append(arrays[0])
        tests.append(arrays[1])
    if parent["control"] == "independent_columns":
        rng = np.random.default_rng(1000)
        trains = [np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
                  for x in trains]
    with np.load(parent_path.parent / "arrays.npz", allow_pickle=False) as z:
        mixings = [z["mixing_" + a].copy() for a in ASSAYS]
        means = [z["train_mean_" + a].copy() for a in ASSAYS]
    sources = []
    for x, mean, mixing in zip(trains, means, mixings):
        source = (x - mean) @ np.linalg.pinv(mixing).T
        least_squares = np.linalg.lstsq(mixing, (x - mean).T, rcond=None)[0].T
        np.testing.assert_allclose(source, least_squares, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(mean, x.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(source.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(source.std(axis=0), 1, atol=1e-7)
        sources.append(source)
    records = []
    for a, b in combinations(range(3), 2):
        run = OUT / (parent_path.parent.name + "__" + ASSAYS[a] + "__" + ASSAYS[b])
        r = load_json(run / "result.json")
        assert r["status"] == "completed"
        assert sha(run / "arrays.npz") == r["arrays_sha256"]
        n, m = sources[a].shape[1], sources[b].shape[1]
        positive, negative = np.empty((n, m)), np.empty((n, m))
        for i in range(n):
            for j in range(m):
                positive[i, j] = ks_2samp(sources[a][:, i], sources[b][:, j]).pvalue
                negative[i, j] = ks_2samp(sources[a][:, i], -sources[b][:, j]).pvalue
        similarity = np.maximum(positive, negative)
        alpha = .05 / (2 * n * m)
        global_alpha = parent["adjusted_ks_alpha"]
        factors = matching(similarity, alpha)
        global_factors = matching(similarity, global_alpha)
        assert r["pair_adjusted_alpha"] == alpha
        assert r["original_three_assay_adjusted_alpha"] == global_alpha
        assert r["joint_factors"] == [list(v) for v in factors]
        assert r["shared_source_count"] == len(factors)
        assert r["source_count_at_original_global_alpha"] == len(global_factors)
        signs = [[1, 1 if positive[i, j] >= negative[i, j] else -1] for i, j in factors]
        assert r["joint_signs"] == signs
        f = np.asarray(factors, dtype=int).reshape(-1, 2)
        s = np.asarray(signs, dtype=int).reshape(-1, 2)
        shared_a = mixings[a][:, f[:, 0]] * s[:, 0]
        shared_b = mixings[b][:, f[:, 1]] * s[:, 1]
        cov = shared_a @ shared_b.T
        denominator = np.sqrt(np.outer(np.sum(mixings[a] ** 2, axis=1),
                                       np.sum(mixings[b] ** 2, axis=1)))
        corr = np.divide(cov, denominator, out=np.full_like(cov, np.nan), where=denominator > 0)
        ia, ib = ref["paired_indices_" + ASSAYS[a]], ref["paired_indices_" + ASSAYS[b]]
        for idx, x in [(ia, tests[a]), (ib, tests[b])]:
            assert len(idx) == len(set(idx.tolist())) == 595
            assert idx.min() >= 0 and idx.max() < len(x)
        observed = correlation(tests[a][ia], tests[b][ib])
        valid = np.isfinite(corr) & np.isfinite(observed)
        with np.load(run / "arrays.npz", allow_pickle=False) as z:
            np.testing.assert_allclose(z["predicted_covariance"], cov, atol=1e-12)
            np.testing.assert_allclose(z["predicted_correlation"], corr, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(z["reference_correlation"], observed, atol=1e-12, equal_nan=True)
        rmse = float(np.sqrt(np.mean((corr[valid] - observed[valid]) ** 2)))
        zero = float(np.sqrt(np.mean(observed[valid] ** 2)))
        assert r["valid_feature_pairs"] == int(valid.sum()) and r["total_feature_pairs"] == corr.size
        assert np.isclose(r["correlation_rmse"], rmse)
        assert np.isclose(r["zero_correlation_rmse"], zero)
        assert np.isclose(r["rmse_improvement_over_zero"], zero - rmse)
        if not factors:
            assert r["graph_status"] == "no_shared_sources"
        records.append({"assays": [ASSAYS[a], ASSAYS[b]], "shared_source_count": len(factors),
                        "count_at_global_alpha": len(global_factors),
                        "max_signed_KS_pvalue": float(np.max(similarity)),
                        "correlation_rmse": rmse, "zero_correlation_rmse": zero})
    receipt = {"status": "pass", "reviewer": "causal_reliability_pilot", "parent": parent_rel,
               "protocol_sha256": provenance["protocol_sha256"], "helper_sha256": sha(Path(__file__)),
               "rows": records}
    partials = STUDY / "reviews/pairwise_replay_parts"
    partials.mkdir(exist_ok=True)
    (partials / (str(index) + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "pass", "parent": parent_path.parent.name, "comparisons": len(records)}), flush=True)


if __name__ == "__main__":
    replay(int(sys.argv[1]))
