"""Extract frozen, small count inputs; no fitting or data generation.

The existing source artifacts are read-only. Public outputs contain only allowed
counts, their deterministic normalization, and opaque row/feature identifiers.
Truth, pairing, and state joins are written to a separate evaluator directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = HERE / "data_extraction_protocol.json"
OUT = HERE / "data_pilot_v1"
KEYS = {"raw_counts", "log1p_cp10k", "library_depth", "cell_ids", "feature_ids"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strings_sha(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path.relative_to(REPO)), "sha256": sha(path),
            "bytes": path.stat().st_size}


def verify(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Changed source: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def h5_counts(path: Path) -> tuple:
    with h5py.File(path, "r") as h:
        x = h["X"]
        if x.attrs["encoding-type"] != "csr_matrix":
            raise ValueError(f"Expected raw CSR counts: {path}")
        X = sparse.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]),
                              shape=tuple(x.attrs["shape"]))
        ids = h["obs"][h["obs"].attrs["_index"]].asstr()[:].astype(str)
        features = h["var"][h["var"].attrs["_index"]].asstr()[:].astype(str)
    if not np.issubdtype(X.dtype, np.integer) or np.any(X.data < 0):
        raise ValueError("Input is not nonnegative integer count data")
    if len(set(ids)) != len(ids) or len(set(features)) != len(features):
        raise ValueError("Duplicate input identifiers")
    return X, ids, features


def quantiles(values: np.ndarray) -> list:
    return np.quantile(values, [0, .25, .5, .75, 1]).tolist()


def diagnostics(raw: np.ndarray, norm: np.ndarray, depth: np.ndarray) -> dict:
    result = {"n_cells": len(raw), "n_features": raw.shape[1],
              "selected_detected_cells_per_feature": (raw > 0).sum(0).tolist(),
              "selected_zero_row_count": int(np.sum(~np.any(raw, axis=1))),
              "full_allowed_depth_quantiles": quantiles(depth),
              "full_allowed_zero_depth_count": int(np.sum(depth == 0)),
              "selected_count_row_sum_quantiles": quantiles(raw.sum(1)),
              "unique_selected_count_columns": len({raw[:, j].tobytes()
                                                    for j in range(raw.shape[1])})}
    for key, values in (("raw_counts", raw), ("log1p_cp10k", norm)):
        centered = values.astype(np.float64) - values.mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        threshold = singular[0] * max(centered.shape) * np.finfo(np.float64).eps
        result[key + "_centered_rank"] = int(np.sum(singular > threshold))
        result[key + "_rank_tolerance"] = float(threshold)
        result[key + "_centered_singular_values"] = singular.tolist()
    return result


def save_npz(path: Path, **arrays: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays):
            raise AssertionError("Archive keys changed")
        for k, value in arrays.items():
            np.testing.assert_array_equal(loaded[k], value)
    return pin(path)


def public_data(protocol: dict) -> tuple:
    datasets = {}
    test_arrays = {}
    for name, spec in protocol["datasets"].items():
        verify(REPO / spec["source_manifest"], spec["source_manifest_sha256"])
        if "exclusion_source" in spec:
            verify(REPO / spec["exclusion_source"]["path"],
                   spec["exclusion_source"]["sha256"])
        record = {"assays": {}, "source_manifest": spec["source_manifest"],
                  "source_manifest_sha256": spec["source_manifest_sha256"]}
        population_ids = {}
        for assay in protocol["assays"]:
            frozen_features = None
            chosen = None
            eligible_n = None
            cohort_records = {}
            for cohort in ("train", "test"):
                source = spec["cohorts"][cohort][assay]
                source_path = REPO / source["path"]
                verify(source_path, source["sha256"])
                X, ids, features = h5_counts(source_path)
                if cohort == "train":
                    frozen_features = features.copy()
                else:
                    np.testing.assert_array_equal(features, frozen_features)
                excluded = np.asarray([x in set(spec["exclusions"][assay])
                                       for x in features])
                if int(excluded.sum()) != len(spec["exclusions"][assay]):
                    raise ValueError("A declared feature exclusion is absent")
                allowed = ~excluded
                depth = np.asarray(X[:, allowed].sum(axis=1)).ravel().astype(np.int64)
                if cohort == "train":
                    detected = np.asarray((X > 0).sum(axis=0)).ravel()
                    high = X.max(axis=0).toarray().ravel()
                    low = X.min(axis=0).toarray().ravel()
                    eligible = np.flatnonzero(allowed & (detected >= 10) & (high != low))
                    eligible_n = len(eligible)
                    ordered = sorted(eligible, key=lambda j: (
                        hashlib.sha256(features[j].encode()).hexdigest(), features[j]))
                    if len(ordered) < 30:
                        raise ValueError(f"Fewer than30 eligible features: {name}/{assay}")
                    chosen = np.asarray(ordered[:30], dtype=int)
                raw = X[:, chosen].toarray()
                norm = np.log1p(1e4 * raw.astype(np.float64) /
                                np.maximum(depth, 1)[:, None])
                if not np.all(np.isfinite(norm)):
                    raise ValueError("Nonfinite normalized values")
                arrays = dict(raw_counts=raw, log1p_cp10k=norm,
                              library_depth=depth, cell_ids=ids,
                              feature_ids=features[chosen])
                assert set(arrays) == KEYS
                path = OUT / "public" / name / cohort / (assay + ".npz")
                artifact = save_npz(path, **arrays)
                artifact.update(shape=list(raw.shape),
                                cell_ids_sha256=strings_sha(ids),
                                feature_ids_sha256=strings_sha(features[chosen]),
                                source=source, source_shape=list(X.shape),
                                allowed_depth_feature_count=int(allowed.sum()),
                                diagnostics=diagnostics(raw, norm, depth))
                cohort_records[cohort] = artifact
                population_ids[cohort, assay] = set(ids)
                if cohort == "test":
                    test_arrays[name, assay] = arrays
                del X
            record["assays"][assay] = {
                "training_eligible_feature_count": eligible_n,
                "selected_feature_ids": frozen_features[chosen].tolist(),
                "selected_source_column_indices": chosen.tolist(),
                "excluded_feature_ids": spec["exclusions"][assay],
                "cohorts": cohort_records}
        groups = list(population_ids.items())
        for i, (ka, va) in enumerate(groups):
            for kb, vb in groups[i + 1:]:
                if va & vb:
                    raise ValueError(f"Public identities overlap: {name}/{ka}/{kb}")
        record["all_source_assay_and_partition_ID_sets_disjoint"] = True
        datasets[name] = record
        print(name, {a: record["assays"][a]["cohorts"]["train"]["shape"]
                     for a in protocol["assays"]}, flush=True)
    return datasets, test_arrays


def table(path: Path) -> list:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def evaluator_data(protocol: dict, tests: dict) -> dict:
    result = {}
    real = REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1"
    summary = read_json(real / "summary.json")
    bridge = read_json(real / "model_inputs/manifest.json")
    verify(real / "summary.json", bridge["source_summary_sha256"])
    source = real / "evaluator/heldout_pairs.tsv"
    verify(source, summary["files"]["evaluator/heldout_pairs.tsv"]["sha256"])
    pairs = table(source)
    assert len(pairs) == 595
    arrays = {"canonical_ids": np.asarray([r["canonical_id"] for r in pairs])}
    blocks, normalized, columns = [], [], []
    train_sets = []
    for assay in protocol["assays"]:
        values = tests["scmtr_seq_s1", assay]
        mapping = {c: i for i, c in enumerate(values["cell_ids"])}
        order = np.asarray([mapping[r[assay + "_cell_id"]] for r in pairs])
        if len(set(order)) != 595:
            raise ValueError("Non-bijective heldout pairing")
        arrays["paired_indices_" + assay] = order
        blocks.append(values["raw_counts"][order])
        normalized.append(values["log1p_cp10k"][order])
        columns.extend(values["feature_ids"])
        path = real / "evaluator" / (assay + "_train_map.tsv")
        verify(path, summary["files"][str(path.relative_to(real))]["sha256"])
        train_sets.append({r["canonical_id"] for r in table(path)})
    for i, a in enumerate(train_sets):
        if a & set(arrays["canonical_ids"]):
            raise ValueError("Canonical train/test overlap")
        if any(a & b for b in train_sets[i + 1:]):
            raise ValueError("Canonical training-assay overlap")
    arrays.update(raw_counts=np.column_stack(blocks),
                  log1p_cp10k=np.column_stack(normalized),
                  column_ids=np.asarray(columns),
                  column_assays=np.repeat(protocol["assays"], 30))
    result["scmtr_seq_s1"] = save_npz(
        OUT / "evaluator/scmtr_seq_s1/paired_reference.npz", **arrays)
    result["scmtr_seq_s1"].update(source=pin(source), n_cells=595,
                                  n_features=90, target="Observed heldout joint; not causal truth",
                                  canonical_train_test_and_assay_disjoint=True)
    for name in ("linear_chain_0", "constrained_collider_0"):
        spec = protocol["datasets"][name]
        world = (REPO / spec["source_manifest"]).parent
        wm = read_json(world / "manifest.json")
        source = world / "truth/discovery.npz"
        verify(source, wm["files"]["truth/discovery.npz"])
        arrays = {}
        with np.load(source, allow_pickle=False) as truth:
            index = {x: i for i, x in enumerate(truth["cell_ids"])}
            for assay in protocol["assays"]:
                ids = tests[name, assay]["cell_ids"]
                rows = np.asarray([index[x] for x in ids])
                if not np.all(truth["assigned_assay"][rows] == assay):
                    raise ValueError("Wrong assay assignment in state join")
                arrays["ids_" + assay] = ids.copy()
                arrays["state_" + assay] = truth["state"][rows].copy()
        result[name] = save_npz(OUT / "evaluator" / name / "state_reference.npz",
                                **arrays)
        result[name].update(source=pin(source),
                            target="Heldout state association only; not a latent DAG oracle",
                            selected_background_cross_covariance_available=False,
                            private_cohorts_accessed=False)
    return result


def main() -> None:
    protocol = read_json(PROTOCOL)
    if OUT.exists():
        raise FileExistsError("Refuse to overwrite any existing extraction directory")
    OUT.mkdir()
    datasets, tests = public_data(protocol)
    evaluator = evaluator_data(protocol, tests)
    manifest = {"status": "complete", "protocol": pin(PROTOCOL),
                "helper": pin(Path(__file__).resolve()), "python": sys.executable,
                "versions": {"numpy": np.__version__, "h5py": h5py.__version__},
                "datasets": datasets, "evaluator": evaluator,
                "scope": "Existing count extraction only; no fitting, generated worlds, or locked-cohort reads"}
    path = OUT / "manifest.json"
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("manifest", path, sha(path), flush=True)


if __name__ == "__main__":
    main()
