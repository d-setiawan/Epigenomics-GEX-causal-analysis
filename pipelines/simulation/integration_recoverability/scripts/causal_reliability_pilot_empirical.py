"""Frozen empirical scMTR holdout diagnostics; paired graphs are not causal truth."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist

import causal_reliability_pilot as pilot

ev, ASSAYS, CONFIG = pilot.ev, pilot.ASSAYS, pilot.CONFIG
REPO = pilot.BASE.parents[2]
DATA = REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1"
ROOT = pilot.BASE / "runs/integration_validation/20261101_scmtr_reliability_v1"
ARMS = ("paired_reference", "random_reference", "scconfluence_seed0", "scconfluence_seed1")
DAYS = ("D0", "D1", "D2", "D3")
read, write, pin, verify, digest = pilot.read, pilot.write, pilot.pin, pilot.verify, pilot.digest


def table(relative):
    return pd.read_csv(DATA / relative, sep="\t", dtype=str, keep_default_na=False)


def freeze():
    """Read sealed provenance and public IDs, without reading paired outcomes."""
    summary, bridge = read(DATA / "summary.json"), read(DATA / "model_inputs/manifest.json")
    if bridge["source_summary_sha256"] != digest(DATA / "summary.json"):
        raise ValueError("Model inputs belong to a different cohort summary")
    files = {relative: {"path": str((DATA / relative).resolve()), "sha256": item["sha256"]}
             for relative, item in summary["files"].items()}
    inputs = {}
    for key, item in bridge["artifacts"].items():
        inputs[key] = {"path": str((DATA / "model_inputs" / item["path"]).resolve()), "sha256": item["sha256"]}
    for record in (*files.values(), *inputs.values()):
        verify(record)
    pools = {a: ev._ids(table(f"learner/{a}_test_cells.tsv").cell_id, a) for a in ASSAYS}
    if any(len(ids) != 595 for ids in pools.values()) or summary["paired_reference_n"] != 595:
        raise ValueError("Require all 595 predeclared complete-case test cells")
    protocol = {
        "status": "frozen", "arms": list(ARMS), "pools": ev._jsonable(pools), "files": files,
        "inputs": inputs, "summary": pin(DATA / "summary.json"),
        "cohort_protocol": pin(DATA / "frozen_protocol.json"), "model_input_manifest": pin(DATA / "model_inputs/manifest.json"),
        "sources": {str(Path(p).resolve()): digest(Path(p)) for p in (__file__, pilot.__file__, ev.__file__)},
        "discovery": asdict(CONFIG), "backend": ev.backend_provenance(CONFIG),
        "matching": "RNA-anchored Euclidean Hungarian assignment from exported coordinates only",
        "random_matching_seed": pilot.MATCHING_SEED,
        "subsamples": {"fraction": .8, "seeds": list(pilot.SUBSAMPLE_SEEDS),
                       "rule": "Independent per-assay sampling and rematching; paired control samples complete RNA-anchor cells"},
        "panel": "All nine frozen training-coverage-selected nodes, including all three assay pairs; no panel replacement",
        "graph_reference": "Full 595-cell paired-count FCI graph is an uncertain empirical agreement reference, never known causal truth",
        "primary": "Cross-assay adjacency overlap, extra/missing adjacencies and endpoint agreement with paired graph; undefined denominators remain null",
        "dependence": "All pairwise log1p-count Pearson and tied-rank Spearman errors; full pools only; within-assay blocks must remain exact",
        "retrieval": "Full-pool exact canonical pair fraction for every assay pair; Euclidean true-pair rank and recall@1/@5 in all six directions; ties use pessimistic rank",
        "day_decoder": {"method": "StandardScaler then balanced LogisticRegression", "C": 1.0, "max_iter": 1000, "seed": 0,
                        "training": "Each source assay's disjoint train embeddings and evaluator-only dominant RT labels; valid tag 01-04 and fraction >0.5 only",
                        "testing": "All nine source-train-assay to target-test-assay transfers on all595 paired holdout cells; no test fitting or selection",
                        "role": "Posthoc differentiation-day predictability; not cell identity, causal truth, or independent biological replication"},
        "limits": ["Single S1 pooled-day sublibrary and internal complete-case holdout; no donor/culture replication",
                   "Released DNA cohort inherits a six-target prefilter; new QC is assigned-assay-only",
                   "Sparse count ties, measurement noise and library depth can affect nominal RCIT results",
                   "Matching induces dependence; nominal CI tests have no established post-matching FDR calibration",
                   "Three subsamples with fixed encoders measure coarse conditional stability, not confidence intervals",
                   "Empty paired and predicted graphs cannot establish recovery; causal directions and false-positive rates are unknown"]}
    write(ROOT / "protocol.json", protocol)
    return protocol


def context():
    protocol = read(ROOT / "protocol.json")
    if protocol["discovery"] != asdict(CONFIG) or protocol["arms"] != list(ARMS):
        raise ValueError("Frozen configuration changed")
    for path, sha in protocol["sources"].items():
        verify({"path": path, "sha256": sha})
    for key in ("summary", "cohort_protocol", "model_input_manifest"):
        verify(protocol[key])
    for record in (*protocol["files"].values(), *protocol["inputs"].values()):
        verify(record)
    return protocol, {a: np.asarray(ids) for a, ids in protocol["pools"].items()}


def load_evaluator(pools):
    nodes = table("evaluator/panel_nodes.tsv")
    if len(nodes) != 9 or nodes.node_id.duplicated().any() or list(nodes.assay) != list(np.repeat(ASSAYS, 3)):
        raise ValueError("Unexpected frozen assay-major panel")
    candidates = [{"name": row.node_id, "assay": row.assay, "role": "heldout_panel"} for row in nodes.itertuples()]
    pairs = table("evaluator/heldout_pairs.tsv")
    if len(pairs) != 595 or pairs.canonical_id.duplicated().any():
        raise ValueError("Invalid paired-reference identities")
    blocks, ids, canonical = {}, {}, {}
    train_sets = []
    for assay in ASSAYS:
        rows = table(f"evaluator/{assay}_test_map.tsv")
        ids[assay] = ev._ids(rows.cell_id, assay)
        if list(ids[assay]) != table(f"learner/{assay}_test_cells.tsv").cell_id.tolist() or set(ids[assay]) != set(pools[assay]):
            raise ValueError("Evaluator IDs disagree with frozen learner rows")
        canonical[assay] = dict(zip(rows.cell_id, rows.canonical_id))
        if rows.canonical_id.duplicated().any() or set(rows.canonical_id) != set(pairs.canonical_id):
            raise ValueError("Assay does not cover complete paired reference")
        if any(canonical[assay][cell] != canon for cell, canon in zip(pairs[assay + "_cell_id"], pairs.canonical_id)):
            raise ValueError("Pair table disagrees with per-assay map")
        blocks[assay] = sparse.load_npz(DATA / f"evaluator/{assay}_test_panel_counts.npz").toarray()
        if blocks[assay].shape != (len(ids[assay]), 3) or np.any(blocks[assay] < 0) or np.any(blocks[assay] != np.floor(blocks[assay])):
            raise ValueError("Invalid original panel count block")
        training = table(f"evaluator/{assay}_train_map.tsv")
        if training.cell_id.tolist() != table(f"learner/{assay}_train_cells.tsv").cell_id.tolist():
            raise ValueError("Training evaluator IDs disagree with learner rows")
        train_sets.append(set(training.canonical_id))
    if any(s & set(pairs.canonical_id) for s in train_sets) or any(a & b for a, b in itertools.combinations(train_sets, 2)):
        raise ValueError("Canonical cells overlap between disjoint training assays or test")
    paired_rows = table("evaluator/paired_panel_rows.tsv").canonical_id.tolist()
    paired = sparse.load_npz(DATA / "evaluator/paired_panel_counts.npz").toarray()
    pair_lookup = pairs.set_index("canonical_id")
    for i, assay in enumerate(ASSAYS):
        ordered_ids = pair_lookup.loc[paired_rows, assay + "_cell_id"].to_numpy()
        if not np.array_equal(paired[:, i * 3:(i + 1) * 3], blocks[assay][ev._indices(ids[assay], ordered_ids)]):
            raise ValueError("Paired reference does not exactly reproduce original count blocks")
    return {"candidates": candidates, "pairs": pairs, "ids": ids, "canonical": canonical, "blocks": blocks}


def paired_assignment(rna_ids, data):
    lookup = data["pairs"].set_index("RNA_cell_id")
    return {"cell_ids": {a: np.asarray(rna_ids) if a == "RNA" else lookup.loc[rna_ids, a + "_cell_id"].to_numpy() for a in ASSAYS},
            "procedure": "observed_paired_evaluator_reference", "n_tuples": len(rna_ids)}


def assemble(assignment, data):
    columns = []
    for a in ASSAYS:
        chosen = ev._ids(assignment["cell_ids"][a], a)
        columns.append(data["blocks"][a][ev._indices(data["ids"][a], chosen)])
    return np.hstack(columns)


def graph_agreement(discovered, reference, candidates):
    if discovered["status"] != "ok" or reference["status"] != "ok":
        return {"status": "unavailable", "reason": "Discovery or paired-reference failure is not an empty graph"}
    pred, ref = ev._edge_map(discovered["edges"]), ev._edge_map(reference["edges"])
    assays = {r["name"]: r["assay"] for r in candidates}
    result = {"status": "ok", "reference": "Uncertain empirical paired graph; no causal truth or false-positive labels", "by_assay_pair": {}}
    groups = {"all": lambda pair: True, "cross_assay": lambda p: assays[p[0]] != assays[p[1]],
              "within_assay": lambda p: assays[p[0]] == assays[p[1]]}
    for a, b in itertools.combinations(ASSAYS, 2):
        groups[f"{a}--{b}"] = lambda p, a=a, b=b: {assays[p[0]], assays[p[1]]} == {a, b}
    for label, include in groups.items():
        p, r = {x for x in pred if include(x)}, {x for x in ref if include(x)}
        common = p & r
        endpoint_matches = sum(sum(x == y for x, y in zip(pred[pair], ref[pair])) for pair in common)
        result["by_assay_pair"][label] = {"predicted": len(p), "reference": len(r), "overlap": len(common),
                                            "extra_to_reference": len(p - r), "missing_from_reference": len(r - p),
                                            "prediction_overlap_fraction": len(common) / len(p) if p else None,
                                            "reference_coverage": len(common) / len(r) if r else None,
                                            "jaccard": len(common) / len(p | r) if p | r else None,
                                            "endpoint_agreement_on_overlap": endpoint_matches / (2 * len(common)) if common else None}
    result["assay_cross_adjacency_participation"] = {a: sum(assays[p[0]] != assays[p[1]] and a in (assays[p[0]], assays[p[1]]) for p in pred) for a in ASSAYS}
    return result


def retrieval(assignment, views, data):
    canonical = {a: np.asarray([data["canonical"][a][c] for c in assignment["cell_ids"][a]]) for a in ASSAYS}
    result = {"assignment": {f"{a}--{b}": {"exact_pair_fraction": float(np.mean(canonical[a] == canonical[b])),
                                           "random_one_to_one_expectation": 1 / len(canonical[a])}
                              for a, b in itertools.combinations(ASSAYS, 2)}, "nearest_neighbor": {}}
    if views is not None:
        for a, b in itertools.permutations(ASSAYS, 2):
            x, y = views[a], views[b]
            distances = cdist(x["shared"], y["shared"])
            target = {data["canonical"][b][c]: i for i, c in enumerate(y["cell_ids"])}
            true_indices = np.asarray([target[data["canonical"][a][c]] for c in x["cell_ids"]])
            true_dist = distances[np.arange(len(x["cell_ids"])), true_indices]
            # Counting all tied distances prevents a collapsed embedding from winning by row order.
            ranks = np.sum(distances <= true_dist[:, None], axis=1)
            result["nearest_neighbor"][f"{a}-->{b}"] = {"mean_true_pair_rank": float(ranks.mean()),
                                                        "recall_at_1": float(np.mean(ranks <= 1)), "recall_at_5": float(np.mean(ranks <= 5))}
    return result


def method_views(directory, seed, protocol, cohort):
    # Reuse the sealed loader without changing its already-frozen simulation source.
    with patch.object(pilot, "COHORT", cohort):
        views, provenance = pilot.external_views(directory, seed)
    if provenance["selection_references"]:
        raise ValueError("Real cohort must not use simulated training selection references")
    for assay in ASSAYS:
        if provenance["training_inputs"][assay] != protocol["inputs"][f"train/{assay}"]:
            raise ValueError("Model trained on different empirical inputs")
        if provenance["encoding_inputs"][assay] != protocol["inputs"][f"{cohort}/{assay}"]:
            raise ValueError("Model encoded different empirical inputs")
        expected = table(f"learner/{assay}_{cohort}_cells.tsv").cell_id.tolist()
        if set(views[assay]["cell_ids"]) != set(expected):
            raise ValueError("Require complete declared empirical cohort; do not shrink")
    return views, provenance


def day_transfer(train_views, test_views):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    source = table("evaluator/source_cell_manifest.tsv").set_index("canonical_id")
    label_map = {f"0{i + 1}": day for i, day in enumerate(DAYS)}
    labels = {}
    for cohort, views in (("train", train_views), ("test", test_views)):
        for assay in ASSAYS:
            mapping = table(f"evaluator/{assay}_{cohort}_map.tsv").set_index("cell_id")
            rows = source.loc[mapping.loc[views[assay]["cell_ids"], "canonical_id"]]
            valid = rows.dominant_RT_tag.isin(label_map) & (rows.dominant_RT_fraction.astype(float) > .5)
            values = rows.dominant_RT_tag.map(label_map).fillna("").to_numpy()
            labels[cohort, assay] = (values, valid.to_numpy())
    results = {}
    for a in ASSAYS:
        y, valid = labels["train", a]
        if len(set(y[valid])) != 4:
            results[a] = {"status": "unavailable", "reason": "Declared four-day source training labels incomplete"}
            continue
        decoder = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=0))
        decoder.fit(train_views[a]["shared"][valid], y[valid])
        for b in ASSAYS:
            expected, valid_test = labels["test", b]
            if not np.all(valid_test):
                raise ValueError("Frozen complete-case test includes invalid differentiation-day labels")
            predicted = decoder.predict(test_views[b]["shared"])
            results[f"{a}-->{b}"] = {"status": "ok", "n_train_labeled": int(valid.sum()), "n_train_invalid_label": int((~valid).sum()),
                                      "n_test": len(expected), "accuracy": float(accuracy_score(expected, predicted)),
                                      "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
                                      "confusion_order": list(DAYS), "confusion": confusion_matrix(expected, predicted, labels=list(DAYS)).tolist()}
    return results


def stage(arm, label, binding, values, names):
    path = ROOT / "stages" / f"{arm}_{label}.json"
    receipt = {"protocol_sha256": digest(ROOT / "protocol.json"), "inputs": binding}
    if path.exists():
        saved = read(path)
        if saved["binding"] != receipt:
            raise ValueError("Prior empirical graph belongs to different inputs")
        return saved["result"]
    result = ev.discover(values, names, method="fci", scale="counts", config=CONFIG)
    write(path, {"binding": receipt, "result": ev._jsonable(result)})
    return result


def run_arm(arm, directory=None):
    if arm not in ARMS:
        raise ValueError("Undeclared empirical arm")
    protocol, pools = context()
    views, provenance = (None, {}) if directory is None else method_views(directory, int(arm[-1]), protocol, "test")
    if arm.startswith("scconfluence") and views is None:
        raise ValueError("Model directory required")
    selections = {"full": pools, **{f"subsample_{s}": pilot.subsample(pools, s) for s in pilot.SUBSAMPLE_SEEDS}}
    assignments = {}
    if arm != "paired_reference":
        for key, selected in selections.items():
            assignments[key] = ev.match_one_to_one(selected, views) if views is not None else ev.match_one_to_one(selected, random_seed=pilot.MATCHING_SEED)
    # All learned/random couplings are chosen before loading paired mappings or panel values.
    data = load_evaluator(pools)
    names = [c["name"] for c in data["candidates"]]
    reference_assignment = paired_assignment(pools["RNA"], data)
    paired = assemble(reference_assignment, data)
    reference_binding = {"assignment": ev._jsonable(reference_assignment), "provenance": {}}
    reference = stage("paired_reference", "full", reference_binding, paired, names)
    results = {}
    for key, selected in selections.items():
        assignment = paired_assignment(selected["RNA"], data) if arm == "paired_reference" else assignments[key]
        values = assemble(assignment, data)
        if key == "full":
            for i, assay in enumerate(ASSAYS):
                ordered = ev._indices(assignment["cell_ids"][assay], data["ids"][assay])
                if not np.array_equal(values[ordered, i * 3:(i + 1) * 3], data["blocks"][assay]):
                    raise ValueError("Matching changed an original within-assay count block")
        binding = {"assignment": ev._jsonable(assignment), "provenance": provenance}
        print(f"scmtr/{arm}/{key}/counts", flush=True)
        graph = stage(arm, key, binding, values, names)
        readout = ROOT / "readouts" / f"{arm}_{key}.npz"
        ev._save_npz_new(readout, counts=values, candidate_names=np.asarray(names), **{f"cell_ids_{a}": ids for a, ids in assignment["cell_ids"].items()})
        results[key] = {"assignment": ev._jsonable(assignment), "readout": pin(readout), "discovery": graph,
                        "paired_graph_agreement": graph_agreement(graph, reference, data["candidates"])}
        if key == "full":
            results[key]["dependence"] = ev.dependence_metrics(np.log1p(values), np.log1p(paired), data["candidates"], scale="log1p_counts")
            results[key]["retrieval"] = retrieval(assignment, views, data)
    day, train_provenance = None, None
    if views is not None:
        train_views, train_provenance = method_views(directory, int(arm[-1]), protocol, "train")
        day = day_transfer(train_views, views)
    output = {"status": "complete" if all(r["discovery"]["status"] == "ok" for r in results.values()) else "complete_with_failures",
              "arm": arm, "protocol_sha256": digest(ROOT / "protocol.json"), "provenance": provenance,
              "train_export_provenance": train_provenance, "results": results, "day_transfer": day,
              "count_subsample_stability": pilot.edge_stability([results[f"subsample_{s}"]["discovery"] for s in pilot.SUBSAMPLE_SEEDS]),
              "paired_panel_detection": [{"node": name, "detected": int(np.count_nonzero(paired[:, i])), "n": len(paired)} for i, name in enumerate(names)],
              "causal_direction_recovery": {"status": "unavailable", "reason": "No causal direction ground truth in empirical S1 measurements"}}
    write(ROOT / "arms" / f"{arm}.json", output)
    return {"arm": arm, "status": output["status"], "paired_graph_agreement": results["full"]["paired_graph_agreement"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "baseline", "method"))
    parser.add_argument("--seed", type=int, choices=(0, 1))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
        result = pin(ROOT / "protocol.json")
    elif args.action == "baseline":
        result = [run_arm(a) for a in ARMS[:2]]
    else:
        if args.seed is None or args.directory is None:
            parser.error("method requires declared --seed and --directory")
        result = run_arm(f"scconfluence_seed{args.seed}", args.directory)
    print(json.dumps(ev._jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
