"""Post-primary exploratory null: 50 fixed full-pool assay-row permutations."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

import causal_reliability_pilot_empirical as empirical

ev, CONFIG = empirical.ev, empirical.CONFIG
SEEDS = tuple(range(10000, 10050))
ROOT = empirical.pilot.BASE / "runs/integration_validation/20261101_scmtr_product_null_audit_v1"


def freeze():
    _, pools = empirical.context()
    protocol = {
        "status": "frozen", "exploratory": True,
        "timing": "Added after observing the primary paired and random controls; not a predeclared primary comparison",
        "reason": "A recurring cross-assay adjacency appeared under the primary random coupling; assess current test under repeated artificial null couplings",
        "primary_protocol": empirical.pin(empirical.ROOT / "protocol.json"),
        "motivation_primary_random_report": empirical.pin(empirical.ROOT / "arms/random_reference.json"),
        "sources": {str(Path(p).resolve()): empirical.digest(Path(p)) for p in (__file__, empirical.__file__, empirical.pilot.__file__, ev.__file__)},
        "seeds": list(SEEDS), "pools": ev._jsonable(pools), "n_per_assay": 595,
        "discovery": asdict(CONFIG), "backend": ev.backend_provenance(CONFIG),
        "null": "Fix RNA row order; independently permute the complete H3K27ac and H3K4me3 row blocks using SeedSequence([seed,assay_index]); preserve all within-assay values",
        "outcomes": ["All 50 graphs and failures", "Cross-edge count distribution", "Fraction of successful graphs with at least one cross edge", "Per-edge recurrence using successful graphs only"],
        "constraints": "No threshold, panel, method or model selection; no changes to primary protocol/arms; original count CI readout remains log1p without depth adjustment",
        "interpretation": "Conditional artificial product-coupling null on fixed empirical marginals; not biological causal truth, a biological FDR estimate, or independent biological replication"}
    empirical.write(ROOT / "protocol.json", protocol)
    return protocol


def summarize(outcomes, candidates):
    assays = {x["name"]: x["assay"] for x in candidates}
    successful, failed, counts, recurrence, by_seed = [], [], [], Counter(), []
    for row in outcomes:
        seed, graph = row["seed"], row["discovery"]
        if graph["status"] != "ok":
            failed.append(seed)
            by_seed.append({"seed": seed, "status": graph["status"], "cross_edges": None})
            continue
        successful.append(seed)
        pairs = {p for p in ev._edge_map(graph["edges"]) if assays[p[0]] != assays[p[1]]}
        counts.append(len(pairs))
        recurrence.update(pairs)
        by_seed.append({"seed": seed, "status": "ok", "cross_edges": len(pairs)})
    n = len(successful)
    return {
        "n_attempted": len(outcomes), "n_successful": n, "n_failed": len(failed), "failed_seeds": failed,
        "denominator": "Successful graphs only; failed graphs are not zero-edge outcomes",
        "any_cross_edge_count": sum(x > 0 for x in counts),
        "any_cross_edge_fraction": sum(x > 0 for x in counts) / n if n else None,
        "cross_edge_count_distribution": dict(sorted(Counter(counts).items())),
        "cross_edge_count_mean": float(np.mean(counts)) if n else None,
        "cross_edge_count_quantiles": {str(q): float(np.quantile(counts, q)) for q in (0, .25, .5, .75, .95, 1)} if n else None,
        "edge_recurrence": [{"node1": a, "node2": b, "count": count, "fraction": count / n} for (a, b), count in sorted(recurrence.items())],
        "per_seed": by_seed,
        "interpretation": "Post-primary exploratory artificial-null frequency, not biological FDR or causal truth"}


def run():
    spec = empirical.read(ROOT / "protocol.json")
    if spec["seeds"] != list(SEEDS) or spec["discovery"] != asdict(CONFIG):
        raise ValueError("Exploratory null settings changed")
    for key in ("primary_protocol", "motivation_primary_random_report"):
        empirical.verify(spec[key])
    for path, sha in spec["sources"].items():
        empirical.verify({"path": path, "sha256": sha})
    _, pools = empirical.context()
    if ev._jsonable(pools) != spec["pools"]:
        raise ValueError("Null assay pools changed")
    assignments = {seed: ev.match_one_to_one(pools, random_seed=seed) for seed in SEEDS}
    # Every random coupling is fixed before evaluator identities/counts are read.
    data = empirical.load_evaluator(pools)
    names = [r["name"] for r in data["candidates"]]
    outcomes, sources = [], {}
    for seed, assignment in assignments.items():
        values = empirical.assemble(assignment, data)
        for i, assay in enumerate(empirical.ASSAYS):
            rows = ev._indices(assignment["cell_ids"][assay], data["ids"][assay])
            if not np.array_equal(values[rows, i * 3:(i + 1) * 3], data["blocks"][assay]):
                raise ValueError("Null changed original within-assay observations")
        path = ROOT / "seeds" / f"{seed}.json"
        binding = {"protocol_sha256": empirical.digest(ROOT / "protocol.json"), "seed": seed, "assignment": ev._jsonable(assignment)}
        if path.exists():
            outcome = empirical.read(path)
            if outcome["binding"] != binding:
                raise ValueError("Existing seed outcome belongs to another null protocol")
        else:
            result = ev.discover(values, names, method="fci", scale="counts", config=CONFIG)
            outcome = {"binding": binding, "seed": seed, "discovery": result}
            empirical.write(path, ev._jsonable(outcome))
        outcomes.append(outcome)
        sources[str(path)] = empirical.digest(path)
        print(f"Product-coupling seed {seed}: {outcome['discovery']['status']}", flush=True)
    summary = summarize(outcomes, data["candidates"])
    summary.update(protocol=empirical.pin(ROOT / "protocol.json"), source_outcomes=sources)
    empirical.write(ROOT / "summary.json", summary)
    return {k: summary[k] for k in ("n_attempted", "n_successful", "n_failed", "any_cross_edge_fraction", "cross_edge_count_distribution", "cross_edge_count_mean")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
        result = empirical.pin(ROOT / "protocol.json")
    else:
        result = run()
    print(json.dumps(ev._jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
