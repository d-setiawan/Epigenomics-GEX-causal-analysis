"""Independent source-count replay of the fixed 50-seed exploratory null."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from replay_empirical_baselines import REPO, BASE, ASSAYS, ev, sha256
import replay_empirical_baselines as shared


def audit():
    root = BASE / "runs/integration_validation/20261101_scmtr_product_null_audit_v1"
    protocol_path, summary_path = root / "protocol.json", root / "summary.json"
    protocol, saved_summary = json.loads(protocol_path.read_text()), json.loads(summary_path.read_text())
    assert protocol["exploratory"] and protocol["seeds"] == list(range(10000, 10050))
    assert protocol["n_per_assay"] == 595
    for path, expected in protocol["sources"].items():
        assert sha256(Path(path)) == expected, path
    for key in ("primary_protocol", "motivation_primary_random_report"):
        record = protocol[key]
        assert sha256(Path(record["path"])) == record["sha256"]
    used = [protocol_path, summary_path, Path(__file__).resolve(), Path(shared.__file__).resolve()]
    cohort = REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1"
    source = cohort.parent / "full_s1"
    nodes = pd.read_csv(cohort / "evaluator/panel_nodes.tsv", sep="\t")
    genes = pd.read_csv(cohort / "evaluator/panel_genes.tsv", sep="\t")
    features = pd.read_csv(source / "features.tsv", sep="\t")
    cells = pd.read_csv(source / "cells.tsv", sep="\t")
    used.extend([cohort / "evaluator/panel_nodes.tsv", cohort / "evaluator/panel_genes.tsv", source / "features.tsv", source / "cells.tsv"])
    columns = pd.Index(features.gene_id).get_indexer(genes.gene_id)
    assert np.all(columns >= 0)
    rows = dict(zip(cells.canonical_id, cells.index))
    original, canonical = {}, {}
    for assay in ASSAYS:
        path, mapping = source / f"{assay}.npz", cohort / f"evaluator/{assay}_test_map.tsv"
        original[assay] = sparse.load_npz(path)[:, columns].toarray()
        canonical[assay] = pd.read_csv(mapping, sep="\t").set_index("cell_id").canonical_id.to_dict()
        used.extend([path, mapping])
    pools = {assay: np.asarray(values) for assay, values in protocol["pools"].items()}
    config = ev.DiscoveryConfig(**protocol["discovery"])
    outcomes, counts, recurrence, failed = [], [], Counter(), []
    for seed in protocol["seeds"]:
        assignment = {"RNA": pools["RNA"]}
        for index, assay in enumerate(sorted(set(ASSAYS) - {"RNA"})):
            permutation = np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(595)
            assignment[assay] = pools[assay][permutation]
        values = np.hstack([original[assay][[rows[canonical[assay][cell]] for cell in assignment[assay]]] for assay in ASSAYS])
        outcome_path = root / f"seeds/{seed}.json"
        outcome = json.loads(outcome_path.read_text())
        assert sha256(outcome_path) == saved_summary["source_outcomes"][str(outcome_path)]
        assert outcome["binding"]["protocol_sha256"] == sha256(protocol_path)
        for assay in ASSAYS:
            assert assignment[assay].tolist() == outcome["binding"]["assignment"]["cell_ids"][assay]
        fresh = ev.discover(values, nodes.node_id.tolist(), method="fci", scale="counts", config=config)
        assert fresh["status"] == outcome["discovery"]["status"]
        assert fresh["edges"] == outcome["discovery"]["edges"], seed
        assert fresh.get("backend") == protocol["backend"], seed
        used.append(outcome_path)
        if fresh["status"] != "ok":
            failed.append(seed)
            outcomes.append({"seed": seed, "status": fresh["status"], "cross_edges": None})
            continue
        pairs = {tuple(sorted((edge["node1"], edge["node2"]))) for edge in fresh["edges"] if edge["node1"].split("|")[0] != edge["node2"].split("|")[0]}
        counts.append(len(pairs)); recurrence.update(pairs)
        outcomes.append({"seed": seed, "status": "ok", "cross_edges": len(pairs)})
    n = len(counts)
    independent = {"n_attempted": 50, "n_successful": n, "n_failed": len(failed), "failed_seeds": failed,
                   "any_cross_edge_count": sum(value > 0 for value in counts),
                   "any_cross_edge_fraction": sum(value > 0 for value in counts) / n if n else None,
                   "cross_edge_count_distribution": {str(key): value for key, value in sorted(Counter(counts).items())},
                   "cross_edge_count_mean": float(np.mean(counts)) if n else None,
                   "cross_edge_count_quantiles": {str(q): float(np.quantile(counts, q)) for q in (0, .25, .5, .75, .95, 1)} if n else None,
                   "per_seed": outcomes,
                   "edge_recurrence": [{"node1": first, "node2": second, "count": count, "fraction": count / n} for (first, second), count in sorted(recurrence.items())]}
    for key, value in independent.items():
        assert value == saved_summary[key], key
    return {"status": "pass", "all_50_assignments_reconstructed_from_seed": True,
            "all_50_original_source_count_graphs_replayed_exactly": True, "all_summary_fields_recomputed_exactly": True,
            "exploratory_post_primary": True, "interpretation": "Conditional artificial product-coupling null on fixed empirical marginals; not biological causal truth or a biological FDR estimate",
            "independent_summary": independent, "frozen_sources": protocol["sources"],
            "source_files": {str(path.relative_to(REPO)): sha256(path) for path in sorted(set(used))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "audit": str(args.output), "sha256": sha256(args.output),
                      "any_cross_edge_fraction": report["independent_summary"]["any_cross_edge_fraction"],
                      "mean_cross_edges": report["independent_summary"]["cross_edge_count_mean"]}))
