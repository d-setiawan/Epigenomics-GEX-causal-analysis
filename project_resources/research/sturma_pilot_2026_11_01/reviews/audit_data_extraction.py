"""Independent read-only replay of data selection, normalization and joins.

Does not import the extraction helper and does not fit or calculate dependence.
"""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for b in iter(lambda: handle.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run():
    manifest_path = BASE / "data_pilot_v1/manifest.json"
    m = json.loads(manifest_path.read_text())
    p = json.loads((BASE / "data_extraction_protocol.json").read_text())
    for key in ("protocol", "helper"):
        assert sha(REPO / m[key]["path"]) == m[key]["sha256"]
    result = {"reviewer": "scmtr_data_pilot; separate replay implementation",
              "manifest_sha256": sha(manifest_path), "script_sha256": sha(Path(__file__)),
              "imported_extraction_helper": False, "fit_or_dependence_test_run": False,
              "public_archives": [], "evaluator_joins": {}}
    public = {}
    for name, d in m["datasets"].items():
        for assay, a in d["assays"].items():
            selected = None
            for cohort in ("train", "test"):
                item = a["cohorts"][cohort]
                source = REPO / item["source"]["path"]
                assert sha(source) == item["source"]["sha256"]
                with h5py.File(source) as h:
                    group = h["X"]
                    X = sparse.csr_matrix((group["data"][:], group["indices"][:],
                                           group["indptr"][:]), shape=group.attrs["shape"])
                    ids = h["obs"][h["obs"].attrs["_index"]].asstr()[:].astype(str)
                    names = h["var"][h["var"].attrs["_index"]].asstr()[:].astype(str)
                allowed = np.asarray([x not in p["datasets"][name]["exclusions"][assay]
                                      for x in names])
                if cohort == "train":
                    # Different criterion implementation: column sums and second
                    # moments detect exactly constant integer columns.
                    csc = X.tocsc()
                    hits = np.asarray((csc > 0).sum(axis=0)).ravel()
                    total = np.asarray(csc.sum(axis=0)).ravel().astype(float)
                    squares = np.asarray(csc.astype(float).power(2).sum(axis=0)).ravel()
                    nonconstant = X.shape[0] * squares > total ** 2
                    candidates = np.flatnonzero(allowed & (hits >= 10) & nonconstant)
                    ordered = sorted(candidates, key=lambda j: (
                        hashlib.sha256(names[j].encode()).hexdigest(), names[j]))
                    selected = np.asarray(ordered[:30])
                    assert len(candidates) == a["training_eligible_feature_count"]
                    assert names[selected].tolist() == a["selected_feature_ids"]
                    assert selected.tolist() == a["selected_source_column_indices"]
                path = REPO / item["path"]
                assert sha(path) == item["sha256"]
                with np.load(path, allow_pickle=False) as values:
                    assert set(values.files) == {"raw_counts", "log1p_cp10k", "library_depth",
                                                 "cell_ids", "feature_ids"}
                    np.testing.assert_array_equal(values["cell_ids"], ids)
                    np.testing.assert_array_equal(values["feature_ids"], names[selected])
                    raw = X.tocsc()[:, selected].toarray()
                    np.testing.assert_array_equal(values["raw_counts"], raw)
                    # Original whole-panel sum minus excluded columns, rather
                    # than the helper's allowed-submatrix summation.
                    depth = (np.asarray(X.sum(axis=1)).ravel() -
                             np.asarray(X[:, ~allowed].sum(axis=1)).ravel())
                    np.testing.assert_array_equal(values["library_depth"], depth)
                    norm = np.log1p((raw / np.maximum(depth, 1)[:, None]) * 1e4)
                    np.testing.assert_allclose(values["log1p_cp10k"], norm,
                                               atol=2e-14, rtol=2e-14)
                    if cohort == "test":
                        public[name, assay] = {k: values[k].copy() for k in values.files}
                result["public_archives"].append({"dataset": name, "assay": assay,
                    "cohort": cohort, "exact_counts_ids_features_depth": True,
                    "normalization_matches": True, "metadata_keys_allowlist_only": True,
                    "shape": list(raw.shape)})
    real = REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1/evaluator"
    with np.load(REPO / m["evaluator"]["scmtr_seq_s1"]["path"], allow_pickle=False) as ref:
        for i, assay in enumerate(p["assays"]):
            # Independent join using per-assay canonical maps, rather than the
            # helper's heldout_pairs columns.
            mapping = {r["canonical_id"]: r["cell_id"]
                       for r in table(real / (assay + "_test_map.tsv"))}
            values = public["scmtr_seq_s1", assay]
            row = {x: j for j, x in enumerate(values["cell_ids"])}
            order = np.asarray([row[mapping[x]] for x in ref["canonical_ids"]])
            np.testing.assert_array_equal(order, ref["paired_indices_" + assay])
            for key in ("raw_counts", "log1p_cp10k"):
                np.testing.assert_array_equal(values[key][order], ref[key][:, 30*i:30*(i+1)])
        result["evaluator_joins"]["scmtr_seq_s1"] = {
            "paired_reference_exact_from_independent_per_assay_maps": True,
            "n_paired_cells": len(ref["canonical_ids"])}
    for name in ("linear_chain_0", "constrained_collider_0"):
        record = m["evaluator"][name]
        with np.load(REPO / record["source"]["path"], allow_pickle=False) as truth:
            sorted_indices = np.argsort(truth["cell_ids"])
            sorted_ids = truth["cell_ids"][sorted_indices]
            with np.load(REPO / record["path"], allow_pickle=False) as ref:
                for assay in p["assays"]:
                    ids = public[name, assay]["cell_ids"]
                    positions = sorted_indices[np.searchsorted(sorted_ids, ids)]
                    np.testing.assert_array_equal(truth["cell_ids"][positions], ids)
                    np.testing.assert_array_equal(ref["ids_" + assay], ids)
                    np.testing.assert_array_equal(ref["state_" + assay], truth["state"][positions])
                    assert np.all(truth["assigned_assay"][positions] == assay)
        result["evaluator_joins"][name] = {"state_and_ID_joins_exact": True,
                                            "private_cohorts_read": False}
    result["status"] = "passed"
    output = HERE / "data_extraction_audit.json"
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output, sha(output))


if __name__ == "__main__":
    run()
