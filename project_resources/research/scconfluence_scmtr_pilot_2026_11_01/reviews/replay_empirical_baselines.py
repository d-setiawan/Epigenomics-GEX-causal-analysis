"""Independently assemble S1 baseline counts and replay the frozen CI backend.

Reads original prepared source-cell matrices, without importing the empirical
scorer's joining, matching, panel-selection or scoring functions. This checks
reproducibility; it is not an independent CI implementation or causal proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "pipelines/simulation/integration_recoverability"
sys.path.insert(0, str(BASE))
import evaluate as ev

ASSAYS = ("RNA", "H3K27ac", "H3K4me3")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit():
    root = BASE / "runs/integration_validation/20261101_scmtr_reliability_v1"
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for path, expected in protocol["sources"].items():
        assert sha256(Path(path)) == expected, path
    cohort = REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1"
    source = cohort.parent / "full_s1"
    used = [protocol_path, Path(__file__).resolve(), source / "features.tsv", source / "cells.tsv",
            cohort / "evaluator/panel_nodes.tsv", cohort / "evaluator/panel_genes.tsv"]
    nodes = pd.read_csv(cohort / "evaluator/panel_nodes.tsv", sep="\t")
    genes = pd.read_csv(cohort / "evaluator/panel_genes.tsv", sep="\t")
    features = pd.read_csv(source / "features.tsv", sep="\t")
    cells = pd.read_csv(source / "cells.tsv", sep="\t", dtype={"dominant_RT_tag": str})
    columns = pd.Index(features.gene_id).get_indexer(genes.gene_id)
    assert np.all(columns >= 0)
    row_lookup = dict(zip(cells.canonical_id, cells.index))
    original, maps = {}, {}
    for assay in ASSAYS:
        count_path = source / f"{assay}.npz"
        map_path = cohort / f"evaluator/{assay}_test_map.tsv"
        used.extend([count_path, map_path])
        original[assay] = sparse.load_npz(count_path)[:, columns].toarray()
        maps[assay] = pd.read_csv(map_path, sep="\t").set_index("cell_id").canonical_id.to_dict()
    pools = {assay: np.asarray(values) for assay, values in protocol["pools"].items()}
    selections = {"full": pools}
    for seed in (7000, 7001, 7002):
        selections[f"subsample_{seed}"] = {
            assay: np.sort(np.random.default_rng(np.random.SeedSequence([seed, index])).choice(ids, 476, replace=False))
            for index, (assay, ids) in enumerate(pools.items())}
    config = ev.DiscoveryConfig(**protocol["discovery"])
    report = {"status": "pass", "protocol_sha256": sha256(protocol_path), "reviewer": "scmtr_data_pilot",
              "review_scope": "Independent source-count reconstruction and fixed-backend replay, plus static data-join/leakage/normalization review; not independent CI software or biological validity proof",
              "arms": {}, "findings": [
                  {"level": "checked", "finding": "No blocking data-join, count-order, leakage or empty-reference denominator bug found", "source": "scripts/causal_reliability_pilot_empirical.py:88,140,205,255"},
                  {"level": "material_limit", "finding": "FCI uses log1p raw counts without library-depth adjustment; observed count dependence includes depth and measurement effects", "source": "evaluate.py:417"},
                  {"level": "material_limit", "finding": "Zero full paired cross-assay reference edges leaves cross-reference coverage undefined; an empty predicted graph cannot establish recovery", "source": "scripts/causal_reliability_pilot_empirical.py:154"},
                  {"level": "material_limit", "finding": "Sparse histone counts and cross edges in the random-coupling controls make biological interpretation and nominal CI calibration uncertain; three subsamples do not estimate FDR"},
                  {"level": "checked", "finding": "Learned/random assignments precede paired mappings and panel reads; day labels are used only by a posthoc decoder fitted on disjoint training embeddings", "source": "scripts/causal_reliability_pilot_empirical.py:205,255"}],
              "frozen_sources": protocol["sources"], "command": "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python project_resources/research/scconfluence_scmtr_pilot_2026_11_01/reviews/replay_empirical_baselines.py --output <new-audit.json>"}
    for arm in ("paired_reference", "random_reference"):
        arm_path = root / f"arms/{arm}.json"
        saved = json.loads(arm_path.read_text())
        used.append(arm_path)
        report["arms"][arm] = {}
        for label, selected in selections.items():
            if arm == "paired_reference":
                canonical = [maps["RNA"][cell] for cell in selected["RNA"]]
                values = np.hstack([original[assay][[row_lookup[cell] for cell in canonical]] for assay in ASSAYS])
            else:
                assigned = {"RNA": selected["RNA"]}
                for index, assay in enumerate(sorted(set(ASSAYS) - {"RNA"})):
                    assigned[assay] = selected[assay][np.random.default_rng(np.random.SeedSequence([3141, index])).permutation(len(selected[assay]))]
                values = np.hstack([original[assay][[row_lookup[maps[assay][cell]] for cell in assigned[assay]]] for assay in ASSAYS])
            readout_path = root / f"readouts/{arm}_{label}.npz"
            used.append(readout_path)
            with np.load(readout_path, allow_pickle=False) as readout:
                assert np.array_equal(values, readout["counts"])
            fresh = ev.discover(values, nodes.node_id.tolist(), method="fci", scale="counts", config=config)
            expected = saved["results"][label]["discovery"]
            assert fresh["status"] == "ok" and fresh["edges"] == expected["edges"], (arm, label)
            item = {"n": len(values), "total_edges": len(fresh["edges"]),
                    "cross_edges": sum(edge["node1"].split("|")[0] != edge["node2"].split("|")[0] for edge in fresh["edges"]),
                    "exact_source_count_match": True, "exact_replayed_edges_and_endpoints_match": True, "edges": fresh["edges"]}
            if label == "full":
                item["node_order"] = nodes.node_id.tolist()
                item["detected_per_node"] = np.count_nonzero(values, axis=0).tolist()
                item["nonzero_range_per_node"] = [[int(values[:, i][values[:, i] > 0].min()), int(values[:, i].max())] for i in range(9)]
            report["arms"][arm][label] = item
    report["source_files"] = {str(path.relative_to(REPO)): sha256(path) for path in sorted(set(used))}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output is None:
        print(json.dumps(result, indent=2))
    else:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"status": result["status"], "audit_path": str(args.output), "audit_sha256": sha256(args.output)}))
