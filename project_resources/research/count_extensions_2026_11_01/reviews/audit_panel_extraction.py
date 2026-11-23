"""Independent source-H5 and immutable-prefix replay; no scientific readouts."""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
from scipy import sparse

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes(order="C") == b.tobytes(order="C")


def main():
    protocol = json.loads((STUDY / "data_extraction_protocol.json").read_text())
    manifest = json.loads((STUDY / "data_v1/manifest.json").read_text())
    previous = json.loads((REPO / protocol["old_data_manifest"]["path"]).read_text())
    assert sha(REPO / protocol["old_data_manifest"]["path"]) == protocol["old_data_manifest"]["sha256"]
    records, real_test = [], {}
    for dataset, config in protocol["datasets"].items():
        for assay in protocol["assays"]:
            item = manifest["datasets"][dataset]["assays"][assay]
            indices = np.asarray(item["selected_source_column_indices"])
            for partition in ["train", "test"]:
                artifact = item["cohorts"][partition]
                new_path = REPO / artifact["path"]
                assert sha(new_path) == artifact["sha256"]
                with np.load(new_path, allow_pickle=False) as f:
                    arrays = {k: f[k].copy() for k in f.files}
                assert set(arrays) == set(protocol["public_arrays"])
                old_info = previous["datasets"][dataset]["assays"][assay]["cohorts"][partition]
                assert sha(REPO / old_info["path"]) == old_info["sha256"]
                with np.load(REPO / old_info["path"], allow_pickle=False) as f:
                    for key in arrays:
                        got = arrays[key][:, :30] if key in ["raw_counts", "log1p_cp10k"] else arrays[key][:30] if key == "feature_ids" else arrays[key]
                        exact(got, f[key])
                source = config["cohorts"][partition][assay]
                source_path = REPO / source["path"]
                assert sha(source_path) == source["sha256"]
                with h5py.File(source_path, "r") as h:
                    group = h["X"]
                    matrix = sparse.csr_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=tuple(group.attrs["shape"]))
                    assert matrix.has_canonical_format
                    ids = h["obs"][h["obs"].attrs["_index"]].asstr()[:].astype(str)
                    features = h["var"][h["var"].attrs["_index"]].asstr()[:].astype(str)
                excluded = np.isin(features, config["exclusions"][assay])
                assert excluded.sum() == len(config["exclusions"][assay])
                if partition == "train":
                    detected = np.bincount(matrix.indices[matrix.data > 0], minlength=matrix.shape[1])
                    nonconstant = (detected > 0) & (detected < matrix.shape[0])
                    for j in np.flatnonzero(detected == matrix.shape[0]):
                        nonconstant[j] = np.ptp(matrix[:, j].toarray()) > 0
                    eligible = np.flatnonzero(~excluded & (detected >= 10) & nonconstant)
                    independently_ordered = sorted(eligible, key=lambda j: (hashlib.sha256(features[j].encode()).hexdigest(), features[j]))[:300]
                    assert len(eligible) == item["training_eligible_feature_count"]
                    np.testing.assert_array_equal(indices, independently_ordered)
                exact(matrix[:, indices].toarray(), arrays["raw_counts"])
                exact(ids, arrays["cell_ids"])
                exact(features[indices], arrays["feature_ids"])
                depth = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
                if excluded.any():
                    depth -= np.asarray(matrix[:, excluded].sum(axis=1)).ravel().astype(np.int64)
                exact(depth, arrays["library_depth"])
                np.testing.assert_array_equal(arrays["log1p_cp10k"], np.log1p(
                    1e4 * arrays["raw_counts"].astype(float) / np.maximum(depth, 1)[:, None]))
                records.append({"dataset": dataset, "assay": assay, "partition": partition,
                                "shape": list(arrays["raw_counts"].shape), "source_sha256": source["sha256"],
                                "public_sha256": artifact["sha256"], "source_and_30_prefix_replayed": True})
                if dataset == "scmtr_seq_s1" and partition == "test":
                    real_test[assay] = arrays
                del matrix
    ref_item = manifest["evaluator"]["scmtr_seq_s1"]
    ref_path = REPO / ref_item["path"]
    assert sha(ref_path) == ref_item["sha256"]
    old_ref = previous["evaluator"]["scmtr_seq_s1"]
    assert sha(REPO / old_ref["path"]) == old_ref["sha256"]
    with np.load(ref_path, allow_pickle=False) as ref, np.load(REPO / old_ref["path"], allow_pickle=False) as old_ref_arrays:
        exact(ref["canonical_ids"], old_ref_arrays["canonical_ids"])
        raw, normalized, features = [], [], []
        for assay in protocol["assays"]:
            idx = ref[f"paired_indices_{assay}"]
            exact(idx, old_ref_arrays[f"paired_indices_{assay}"])
            np.testing.assert_array_equal(np.sort(idx), np.arange(595))
            raw.append(real_test[assay]["raw_counts"][idx])
            normalized.append(real_test[assay]["log1p_cp10k"][idx])
            features.extend(real_test[assay]["feature_ids"])
        exact(np.column_stack(raw), ref["raw_counts"])
        exact(np.column_stack(normalized), ref["log1p_cp10k"])
        np.testing.assert_array_equal(features, ref["column_ids"])
        anchor = list(range(30)) + list(range(300, 330)) + list(range(600, 630))
        exact(ref["raw_counts"][:, anchor], old_ref_arrays["raw_counts"])
        exact(ref["log1p_cp10k"][:, anchor], old_ref_arrays["log1p_cp10k"])
    for dataset in ["linear_chain_0", "constrained_collider_0"]:
        assert sha(REPO / manifest["evaluator"][dataset]["path"]) == previous["evaluator"][dataset]["sha256"]
    result = {"status": "pass", "scope": "Full original public H5 replay and count/ID joins only; no model fitting or covariance/state performance metrics",
              "protocol_sha256": sha(STUDY / "data_extraction_protocol.json"),
              "manifest_sha256": sha(STUDY / "data_v1/manifest.json"), "audit_script_sha256": sha(Path(__file__)),
              "public_arrays_checked": 18, "training_selections_independently_replayed": 9,
              "old_30_prefix_byte_checks": 90, "real_paired_cells": 595, "real_paired_features": 900,
              "original90_paired_columns_exact": True, "simulation_state_files_byte_identical": True,
              "state_values_loaded": False, "canonical_training_pair_maps_read": False,
              "new_worlds_or_measurements": False, "records": records}
    out = STUDY / "reviews/data_extraction_audit.json"
    with out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": "pass", "artifact": str(out), "sha256": sha(out)}))


if __name__ == "__main__":
    main()
