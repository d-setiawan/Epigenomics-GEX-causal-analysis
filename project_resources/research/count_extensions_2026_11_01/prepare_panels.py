"""Materialize nested cached count panels without fitting or selecting outcomes.

The immutable previous extraction supplies source hashes, cell cohorts, target
exclusions, and heldout-only pair indices. Only 300 columns are made dense.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import h5py
import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[2]
OLD = STUDY.parent / "sturma_pilot_2026_11_01"
OUT = STUDY / "data_v1"
OLD_HELPER = OLD / "prepare_cached_data_pilot.py"
spec = importlib.util.spec_from_file_location("frozen_count_extraction_helpers", OLD_HELPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_bytes(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def validate_prefix(arrays, previous):
    checked = {}
    for key in old.KEYS:
        values = arrays[key]
        if key in ["raw_counts", "log1p_cp10k"]:
            values = values[:, :30]
        elif key == "feature_ids":
            values = values[:30]
        checked[key] = exact_bytes(values, previous[key])
        require(checked[key], f"Immutable 30-column prefix changed: {key}")
    return checked


def select_columns(matrix, features, allowed):
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    high = matrix.max(axis=0).toarray().ravel()
    low = matrix.min(axis=0).toarray().ravel()
    eligible = np.flatnonzero(allowed & (detected >= 10) & (high != low))
    ordered = sorted(eligible, key=lambda j: (hashlib.sha256(features[j].encode()).hexdigest(), features[j]))
    require(len(ordered) >= 300, "Fewer than 300 eligible features; thresholds will not be relaxed")
    return np.asarray(ordered[:300], dtype=int), len(eligible)


def h5_column(node):
    if isinstance(node, h5py.Group):
        categories = h5_column(node["categories"])
        codes = node["codes"][:]
        require(np.all(codes >= 0), "Missing selected feature annotation")
        return categories[codes]
    if node.dtype.kind in "OSU":
        return node.asstr()[:]
    return node[:]


def selected_overlap_diagnostic(path, chosen, feature_ids):
    """Describe pre-existing promoter overlaps; never change selected features."""
    with h5py.File(path, "r") as h:
        chrom = h5_column(h["var/chrom"])[chosen]
        start = h5_column(h["var/start"])[chosen]
        end = h5_column(h["var/end"])[chosen]
    overlaps = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if chrom[i] == chrom[j] and max(start[i], start[j]) < min(end[i], end[j]):
                overlaps.append({"left_index": i, "right_index": j,
                                 "left_feature": str(feature_ids[i]), "right_feature": str(feature_ids[j]),
                                 "overlap_bp": int(min(end[i], end[j]) - max(start[i], start[j]))})
    return {"overlap_pair_count": len(overlaps), "pairs": overlaps,
            "selection_changed": False,
            "interpretation": "Overlapping promoter counts can share fragments; featurewise independent Poisson noise may fail."}


def training_diagnostics(raw, depth):
    result = {}
    for width in [30, 300]:
        values = raw[:, :width]
        detected = np.sum(values > 0, axis=0)
        result[str(width)] = {"rows": len(values), "columns": width,
                             "detected_cells_per_feature": detected.tolist(),
                             "all_selected_zero_rows": int(np.sum(~np.any(values, axis=1))),
                             "selected_count_row_sum_quantiles": old.quantiles(values.sum(axis=1))}
    outside = depth - raw.sum(axis=1)
    result["common_leave300out_depth"] = {"minimum": int(np.min(outside)),
                                          "nonpositive_rows": int(np.sum(outside <= 0)),
                                          "quantiles": old.quantiles(outside)}
    return result


def public_panels(protocol, old_manifest):
    datasets, real_tests = {}, {}
    for dataset, config in protocol["datasets"].items():
        old.verify(REPO / config["source_manifest"], config["source_manifest_sha256"])
        if "exclusion_source" in config:
            old.verify(REPO / config["exclusion_source"]["path"], config["exclusion_source"]["sha256"])
        record, identities = {"assays": {}}, {}
        for assay in protocol["assays"]:
            previous_assay = old_manifest["datasets"][dataset]["assays"][assay]
            frozen_features, chosen, eligible_count, assay_record = None, None, None, {"cohorts": {}}
            for partition in ["train", "test"]:
                source = config["cohorts"][partition][assay]
                path = REPO / source["path"]
                old.verify(path, source["sha256"])
                matrix, ids, features = old.h5_counts(path)
                if frozen_features is None:
                    frozen_features = features.copy()
                else:
                    require(exact_bytes(features, frozen_features), "Source feature order changed between cohorts")
                excluded_ids = set(config["exclusions"][assay])
                excluded = np.array([feature in excluded_ids for feature in features])
                require(int(excluded.sum()) == len(excluded_ids), "A fixed target exclusion is missing")
                allowed = ~excluded
                depth = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
                if excluded.any():
                    depth -= np.asarray(matrix[:, excluded].sum(axis=1, dtype=np.int64)).ravel()
                if partition == "train":
                    chosen, eligible_count = select_columns(matrix, features, allowed)
                    require(eligible_count == previous_assay["training_eligible_feature_count"],
                            "Eligibility differs from immutable extraction")
                    require(np.array_equal(chosen[:30], previous_assay["selected_source_column_indices"]),
                            "Selection prefix differs from immutable source-column order")
                raw = matrix[:, chosen].toarray()
                normalized = np.log1p(1e4 * raw.astype(np.float64) / np.maximum(depth, 1)[:, None])
                arrays = {"raw_counts": raw, "log1p_cp10k": normalized, "library_depth": depth,
                          "cell_ids": ids, "feature_ids": features[chosen]}
                require(set(arrays) == set(protocol["public_arrays"]), "Public schema changed")
                previous_info = previous_assay["cohorts"][partition]
                previous_path = REPO / previous_info["path"]
                old.verify(previous_path, previous_info["sha256"])
                with np.load(previous_path, allow_pickle=False) as previous:
                    prefix_check = validate_prefix(arrays, previous)
                artifact = old.save_npz(OUT / "public" / dataset / partition / f"{assay}.npz", **arrays)
                artifact.update(shape=list(raw.shape), source=source, source_shape=list(matrix.shape),
                                cell_ids_sha256=old.strings_sha(ids), feature_ids_sha256=old.strings_sha(features[chosen]),
                                allowed_depth_feature_count=int(allowed.sum()),
                                immutable_30_prefix={"source": previous_info["path"],
                                                     "sha256": previous_info["sha256"],
                                                     "dtype_shape_and_bytes_equal": prefix_check})
                if partition == "train":
                    artifact["training_diagnostics"] = training_diagnostics(raw, depth)
                    if dataset == "scmtr_seq_s1" and assay != "RNA":
                        assay_record["annotation_overlap_diagnostic"] = selected_overlap_diagnostic(path, chosen, features[chosen])
                assay_record["cohorts"][partition] = artifact
                identities[assay, partition] = set(ids)
                if dataset == "scmtr_seq_s1" and partition == "test":
                    real_tests[assay] = arrays
                del matrix
            assay_record.update(selected_feature_ids=frozen_features[chosen].tolist(),
                                selected_source_column_indices=chosen.tolist(),
                                excluded_feature_ids=config["exclusions"][assay],
                                training_eligible_feature_count=eligible_count)
            record["assays"][assay] = assay_record
        groups = list(identities.items())
        for i, (left_name, left_ids) in enumerate(groups):
            for right_name, right_ids in groups[i + 1:]:
                require(left_ids.isdisjoint(right_ids), f"Opaque public IDs overlap: {dataset}/{left_name}/{right_name}")
        record["public_opaque_id_sets_disjoint"] = True
        record["canonical_disjointness"] = "Inherited from immutable source extraction audit; training pairing maps not reread"
        datasets[dataset] = record
        print(json.dumps({"dataset": dataset, "training_shapes": {
            a: record["assays"][a]["cohorts"]["train"]["shape"] for a in protocol["assays"]}}), flush=True)
    return datasets, real_tests


def evaluator_references(protocol, real_tests):
    records = {}
    old_reference = protocol["old_evaluator_artifacts"]["scmtr_seq_s1"]
    source = REPO / old_reference["path"]
    old.verify(source, old_reference["sha256"])
    arrays, raw, normalized, columns = {}, [], [], []
    with np.load(source, allow_pickle=False) as previous:
        arrays["canonical_ids"] = previous["canonical_ids"].copy()
        require(len(arrays["canonical_ids"]) == len(set(arrays["canonical_ids"])) == 595,
                "Held-out canonical evaluation population changed")
        for assay in protocol["assays"]:
            indices = previous[f"paired_indices_{assay}"].copy()
            test = real_tests[assay]
            require(np.array_equal(np.sort(indices), np.arange(len(test["cell_ids"]))),
                    "Held-out pairing is not a bijection of the fixed public test pool")
            arrays[f"paired_indices_{assay}"] = indices
            raw.append(test["raw_counts"][indices])
            normalized.append(test["log1p_cp10k"][indices])
            columns.extend(test["feature_ids"])
    arrays.update(raw_counts=np.column_stack(raw), log1p_cp10k=np.column_stack(normalized),
                  column_ids=np.array(columns), column_assays=np.repeat(protocol["assays"], 300))
    artifact = old.save_npz(OUT / "evaluator/scmtr_seq_s1/paired_reference.npz", **arrays)
    artifact.update(source=old_reference, n_cells=595, n_features=900,
                    target="Observed paired heldout moments, not causal truth",
                    canonical_training_pairs_read=False, outcome_metrics_computed=False)
    records["scmtr_seq_s1"] = artifact
    for dataset in ["linear_chain_0", "constrained_collider_0"]:
        source_record = protocol["old_evaluator_artifacts"][dataset]
        source = REPO / source_record["path"]
        old.verify(source, source_record["sha256"])
        destination = OUT / "evaluator" / dataset / "state_reference.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "Refuse to overwrite state reference")
        shutil.copyfile(source, destination)
        old.verify(destination, source_record["sha256"])
        records[dataset] = {**old.pin(destination), "source": source_record,
                            "copy_byte_identical": True, "state_values_loaded": False,
                            "target": "Existing public-discovery state association only; no latent DAG truth",
                            "selected_background_paired_covariance_available": False}
    return records


def main():
    started = time.monotonic()
    protocol_path = STUDY / "data_extraction_protocol.json"
    protocol = old.read_json(protocol_path)
    frozen = {"protocol": old.pin(protocol_path), "helper": old.pin(Path(__file__)),
              "reused_helpers": old.pin(OLD_HELPER)}
    for name in ["old_extraction_protocol", "old_data_manifest"]:
        old.verify(REPO / protocol[name]["path"], protocol[name]["sha256"])
    old_manifest = old.read_json(REPO / protocol["old_data_manifest"]["path"])
    require(not OUT.exists(), "Refuse to overwrite an existing extraction directory")
    OUT.mkdir()
    datasets, real_tests = public_panels(protocol, old_manifest)
    references = evaluator_references(protocol, real_tests)
    for record in frozen.values():
        old.verify(REPO / record["path"], record["sha256"])
    manifest = {"status": "complete", **frozen, "datasets": datasets, "evaluator": references,
                "panel_sizes": [30, 300], "materialized_feature_count": 300,
                "immutable_30_prefix_files_verified": 18,
                "python": sys.executable, "versions": {"numpy": np.__version__, "h5py": h5py.__version__},
                "elapsed_seconds": time.monotonic() - started,
                "maximum_resident_memory_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "output_bytes_excluding_manifest": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
                "no_new_worlds_or_measurements": True, "private_cohorts_read": False,
                "scope": "Counts and ID extraction only; pairing isolated to evaluator artifacts; no outcome fitting or metrics"}
    with (OUT / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "manifest": str((OUT / "manifest.json").relative_to(REPO)),
                      "sha256": old.sha(OUT / "manifest.json"), "elapsed_seconds": manifest["elapsed_seconds"],
                      "maximum_resident_memory_MiB": manifest["maximum_resident_memory_MiB"]}), flush=True)


if __name__ == "__main__":
    main()
