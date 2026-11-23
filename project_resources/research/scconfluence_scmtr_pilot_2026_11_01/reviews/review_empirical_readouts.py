"""Independent sealed retrieval and unchanged evaluator-decoder replay.

No integration fitting, tuning or new scientific comparison is performed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
DAYS = ("D0", "D1", "D2", "D3")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(actual, expected):
    assert np.isclose(float(actual), float(expected), atol=1e-12, rtol=1e-12), (actual, expected)


def review(seed):
    cohort = REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1"
    result_root = REPO / "pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_scmtr_reliability_v1"
    model = REPO / f"pipelines/empirical/integration/runs/20261101_scmtr_scconfluence_v1/seed{seed}"
    arm_path = result_root / f"arms/scconfluence_seed{seed}.json"
    arm = json.loads(arm_path.read_text())
    protocol_path = result_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert arm["protocol_sha256"] == sha256(protocol_path)
    assert arm["arm"] == f"scconfluence_seed{seed}"
    model_manifest = json.loads((model / "model_manifest.json").read_text())
    assert model_manifest["status"] == "complete" and model_manifest["configuration"]["seed"] == seed
    used = [Path(__file__).resolve(), arm_path, protocol_path, model / "model_manifest.json"]
    for path, expected in model_manifest["source_files"].items():
        assert sha256(Path(path)) == expected
        used.append(Path(path))
    for assay in ASSAYS:
        expected = protocol["inputs"][f"train/{assay}"]
        assert model_manifest["inputs"][assay] == expected
        assert sha256(Path(expected["path"])) == expected["sha256"]
        used.append(Path(expected["path"]))
    for key, name in (("checkpoint_sha256", "checkpoint.pt"), ("preprocessing_sha256", "preprocessing.json")):
        assert sha256(model / name) == model_manifest[key]
        used.append(model / name)
    views = {}
    for partition in ("train", "test"):
        manifest_path = model / partition / "encoding_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        used.append(manifest_path)
        assert manifest["status"] == "complete" and manifest["cohort"] == partition
        for key in ("checkpoint_sha256", "preprocessing_sha256"):
            assert manifest[key] == model_manifest[key]
        for assay in ASSAYS:
            expected_input = protocol["inputs"][f"{partition}/{assay}"]
            assert manifest["inputs"][assay] == expected_input
            assert sha256(Path(expected_input["path"])) == expected_input["sha256"]
            used.append(Path(expected_input["path"]))
            path = model / partition / f"{assay}.npz"
            item = manifest["exports"][assay]
            assert item["sha256"] == sha256(path) and not item["dropped_cell_ids"]
            with np.load(path, allow_pickle=False) as saved:
                assert set(saved.files) == {"cell_ids", "shared"}
                # The frozen evaluator's _matrix loader explicitly uses float64.
                # Reproduce that numerical input, not a float32 decoder variant.
                ids, values = saved["cell_ids"].astype(str), np.asarray(saved["shared"], dtype=float)
            cell_path = cohort / f"learner/{assay}_{partition}_cells.tsv"
            cells = pd.read_csv(cell_path, sep="\t").cell_id.tolist()
            assert len(ids) == len(set(ids)) and set(ids) == set(cells)
            assert values.shape == (len(ids), item["latent_dim"]) and np.isfinite(values).all()
            assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == item["cell_ids_sha256"]
            views[partition, assay] = (ids, values)
            used.extend([path, cell_path])
    pools = {assay: np.array(values) for assay, values in protocol["pools"].items()}
    aligned = {}
    for assay in ASSAYS:
        ids, values = views["test", assay]
        lookup = dict(zip(ids, range(len(ids))))
        aligned[assay] = values[[lookup[cell] for cell in pools[assay]]]
    assignment = {"RNA": pools["RNA"]}
    for assay in sorted(set(ASSAYS) - {"RNA"}):
        rows, columns = linear_sum_assignment(cdist(aligned["RNA"], aligned[assay]))
        assert np.array_equal(rows, np.arange(len(rows)))
        assignment[assay] = pools[assay][columns]
    # Pair labels are first loaded after independently reconstructing assignments.
    maps = {}
    for partition in ("train", "test"):
        for assay in ASSAYS:
            path = cohort / f"evaluator/{assay}_{partition}_map.tsv"
            frame = pd.read_csv(path, sep="\t").set_index("cell_id")
            maps[partition, assay] = frame.canonical_id.to_dict()
            used.append(path)
    full = arm["results"]["full"]
    report = {"seed": seed, "status": "pass", "integration_fits_performed": 0,
              "fixed_evaluator_decoder_replays": 3,
              "decoder_coordinate_dtype": "float64, matching the frozen evaluator's explicit loader conversion",
              "reviewer_precision_check": "An initial float32-only reviewer attempt disagreed on confusion matrices and wrote no audit; this replay uses the documented float64 evaluator inputs without changing scientific settings",
              "assignment": {}, "nearest_neighbor": {}, "day_transfer_replay": {},
              "limits": ["Retrieval is independently reconstructed from sealed embeddings and canonical evaluator maps",
                         "The three frozen source-only day decoders are replayed solely to verify all nine confusion matrices; this does not supply independent biological replication, causal truth or test-set model selection"]}
    canonical = {assay: np.array([maps["test", assay][cell] for cell in assignment[assay]]) for assay in ASSAYS}
    for assay in ASSAYS:
        assert assignment[assay].tolist() == full["assignment"]["cell_ids"][assay]
    for first, second in itertools.combinations(ASSAYS, 2):
        key = f"{first}--{second}"
        correct = int(np.count_nonzero(canonical[first] == canonical[second]))
        fraction = correct / len(canonical[first])
        close(fraction, full["retrieval"]["assignment"][key]["exact_pair_fraction"])
        close(1 / len(canonical[first]), full["retrieval"]["assignment"][key]["random_one_to_one_expectation"])
        report["assignment"][key] = {"correct": correct, "n": len(canonical[first]), "exact_pair_fraction": fraction}
    for first, second in itertools.permutations(ASSAYS, 2):
        first_ids, x = views["test", first]
        second_ids, y = views["test", second]
        target = {maps["test", second][cell]: i for i, cell in enumerate(second_ids)}
        expected = np.array([target[maps["test", first][cell]] for cell in first_ids])
        distances = cdist(x, y)
        true_distances = distances[np.arange(len(x)), expected]
        ranks = np.count_nonzero(distances <= true_distances[:, None], axis=1)
        values = {"mean_true_pair_rank": float(ranks.mean()), "recall_at_1": float(np.mean(ranks <= 1)), "recall_at_5": float(np.mean(ranks <= 5))}
        key = f"{first}-->{second}"
        for name, value in values.items():
            close(value, full["retrieval"]["nearest_neighbor"][key][name])
        report["nearest_neighbor"][key] = values
    source_path = cohort / "evaluator/source_cell_manifest.tsv"
    source = pd.read_csv(source_path, sep="\t", dtype={"dominant_RT_tag": str}).set_index("canonical_id")
    used.append(source_path)
    labels = {}
    mapping = {"01": "D0", "02": "D1", "03": "D2", "04": "D3"}
    for partition in ("train", "test"):
        for assay in ASSAYS:
            ids, _ = views[partition, assay]
            original = source.loc[[maps[partition, assay][cell] for cell in ids]]
            valid = original.dominant_RT_tag.isin(mapping) & (original.dominant_RT_fraction > .5)
            labels[partition, assay] = (original.dominant_RT_tag.map(mapping).to_numpy(), valid.to_numpy())
    assert protocol["day_decoder"]["C"] == 1.0
    assert protocol["day_decoder"]["max_iter"] == 1000 and protocol["day_decoder"]["seed"] == 0
    decoders = {}
    for assay in ASSAYS:
        train_labels, valid_train = labels["train", assay]
        scaler = StandardScaler()
        transformed = scaler.fit_transform(views["train", assay][1][valid_train])
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=0)
        classifier.fit(transformed, train_labels[valid_train])
        decoders[assay] = (scaler, classifier)
    for first, second in itertools.product(ASSAYS, repeat=2):
        key = f"{first}-->{second}"
        result = arm["day_transfer"][key]
        assert result["status"] == "ok" and result["confusion_order"] == list(DAYS)
        train_labels, valid_train = labels["train", first]
        test_labels, valid_test = labels["test", second]
        assert valid_test.all()
        confusion = np.asarray(result["confusion"])
        assert confusion.shape == (4, 4) and np.issubdtype(confusion.dtype, np.integer) and np.all(confusion >= 0)
        support = np.array([np.count_nonzero(test_labels == day) for day in DAYS])
        assert np.array_equal(confusion.sum(axis=1), support)
        assert result["n_train_labeled"] == int(valid_train.sum())
        assert result["n_train_invalid_label"] == int((~valid_train).sum())
        assert result["n_test"] == int(confusion.sum()) == len(test_labels)
        scaler, classifier = decoders[first]
        predicted = classifier.predict(scaler.transform(views["test", second][1]))
        fresh_confusion = np.zeros((4, 4), dtype=int)
        day_indices = {day: index for index, day in enumerate(DAYS)}
        for observed, prediction in zip(test_labels, predicted):
            fresh_confusion[day_indices[observed], day_indices[prediction]] += 1
        assert np.array_equal(fresh_confusion, confusion), key
        accuracy = float(np.trace(confusion) / confusion.sum())
        balanced = float(np.mean(np.diag(confusion) / support))
        close(accuracy, result["accuracy"]); close(balanced, result["balanced_accuracy"])
        report["day_transfer_replay"][key] = {"accuracy": accuracy, "balanced_accuracy": balanced,
                                                   "support": support.tolist(), "confusion": confusion.tolist(),
                                                   "n_train_labeled": int(valid_train.sum()), "n_train_invalid_label": int((~valid_train).sum()),
                                                   "per_cell_predictions_independently_recomputed": True,
                                                   "exact_confusion_replay_match": True,
                                                   "ordered_prediction_labels_sha256": hashlib.sha256("\n".join(predicted).encode()).hexdigest()}
    report["source_files"] = {str(path.relative_to(REPO)): sha256(path) for path in sorted(set(used))}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.seed)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "seed": args.seed, "audit": str(args.output), "sha256": sha256(args.output)}))
