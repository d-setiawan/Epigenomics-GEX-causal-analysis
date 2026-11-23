"""Guidance-only adjacency control, separate from data-driven discovery.

Only explicitly exposed feature-to-feature guidance links are returned. There
is no transitive closure, causal orientation, fitting, or access to truth edges
during extraction. Missing scored features are abstentions, not true negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_prior(graph, feature_map: Sequence[Mapping[str, str]], *,
                  excluded_feature_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Extract using only feature identifiers, readable names, and assay labels."""
    fields = {"feature_id", "name", "assay"}
    if any(set(row) != fields for row in feature_map):
        raise ValueError("Prior extraction takes only feature_id/name/assay, never truth roles or edges")
    by_id = {row["feature_id"]: dict(row) for row in feature_map}
    if len(by_id) != len(feature_map) or len({row["name"] for row in feature_map}) != len(feature_map):
        raise ValueError("Scored feature identifiers and names must be unique")
    available = sorted(set(by_id) & set(graph.nodes))
    unavailable = sorted(set(by_id) - set(graph.nodes))
    masked = set(excluded_feature_ids) & set(by_id)
    eligible = sorted(set(available) - masked)
    for node in set(available) & masked:
        neighbors = set(graph.neighbors(node))
        if graph.is_directed():
            neighbors.update(graph.predecessors(node))
        if neighbors - {node}:
            raise ValueError("Exposed graph contradicts its declared cross-feature guidance mask")
    edges = []
    for first, second in itertools.combinations(available, 2):
        evidence = []
        arcs = [(first, second)] + ([(second, first)] if graph.is_directed() else [])
        for start, end in arcs:
            if not graph.has_edge(start, end):
                continue
            attributes = graph.get_edge_data(start, end)
            attributes = list(attributes.values()) if graph.is_multigraph() else [attributes]
            for attrs in attributes:
                if "weight" not in attrs or "sign" not in attrs:
                    raise ValueError("Guidance edge lacks the exposed weight/sign attributes")
                weight, sign = float(attrs["weight"]), float(attrs["sign"])
                if not math.isfinite(weight) or weight <= 0 or sign not in {-1.0, 1.0}:
                    raise ValueError("Guidance edge weight/sign is invalid")
                evidence.append({"weight": weight, "association_sign": int(sign)})
        if evidence:
            # Feature order gives a stable serialization, not an orientation.
            edges.append({"node1": by_id[first]["name"], "node2": by_id[second]["name"],
                          "feature1": first, "feature2": second,
                          "edge_type": "undirected_annotation_prior", "guidance_evidence": evidence})
    reason = ("No scored features are vertices of the exposed guidance graph" if not available else
              "Fewer than two scored features are represented; no adjacency can be evaluated" if len(available) < 2 else
              "Scored vertices are explicitly masked from cross-feature guidance; no eligible scored pair" if len(eligible) < 2 else None)
    return {
        "status": "not_applicable" if reason else "ok", "reason": reason,
        "method": "guidance_only_undirected_prior", "edges": None if reason else edges,
        "available_features": [by_id[feature] for feature in available],
        "unavailable_features": [by_id[feature] for feature in unavailable],
        "excluded_cross_feature_ids": sorted(masked),
        "eligible_features": [by_id[feature] for feature in eligible],
        "n_direct_scored_links_in_exposed_graph": len(edges),
        "baseline_kind": "masked_scored_feature_negative_control" if masked else "exposed_annotation_prior",
        "mask_interpretation": "Missing links incident to explicitly masked features are imposed by construction, not successful null recovery",
        "n_graph_vertices": graph.number_of_nodes(), "n_graph_arcs": graph.number_of_edges(),
        "n_self_loops_ignored": nx.number_of_selfloops(graph),
        "extraction_rule": "Direct exposed links only; self-loops, arrow conventions, and background-mediated paths are not causal edges",
        "direction": {"status": "not_applicable", "reason": "Undirected annotation prior has no causal orientation output"},
        "hyperedges": {"status": "not_applicable", "reason": "No hyperedge output"},
        "equivalence_class": {"status": "not_applicable", "reason": "No justified ambiguity-set output"},
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _adjacency_scores(predicted: set, truth: set, universe: set) -> dict[str, Any]:
    if not universe:
        return {"status": "not_applicable", "reason": "No represented pair universe", "metrics": None}
    true_positive, false_positive, false_negative = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    return {"status": "ok", "n_possible_pairs": len(universe), "n_predicted": len(predicted), "n_true": len(truth),
            "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
            "precision": _ratio(true_positive, len(predicted)), "recall": _ratio(true_positive, len(truth)),
            "f1": _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
            "false_positive_rate": _ratio(false_positive, len(universe - truth))}


def score_prior(prior: Mapping[str, Any], candidates: Sequence[Mapping[str, str]],
                biological_edges: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Truth is first accessed here, after the prior graph has been extracted."""
    by_name = {row["name"]: row for row in candidates}
    represented = {row["name"] for row in prior["available_features"]}
    eligible = {row["name"] for row in prior["eligible_features"]}
    all_pairs = set(itertools.combinations(sorted(by_name), 2))
    represented_pairs = set(itertools.combinations(sorted(represented), 2))
    eligible_pairs = set(itertools.combinations(sorted(eligible), 2))
    true_pairs = {tuple(sorted(edge)) for edge in biological_edges if tuple(sorted(edge)) in all_pairs}
    coverage = {
        "n_scored_features": len(by_name), "n_represented_features": len(represented),
        "feature_coverage": _ratio(len(represented), len(by_name)),
        "n_scored_pairs": len(all_pairs), "n_represented_pairs": len(represented_pairs),
        "pair_coverage": _ratio(len(represented_pairs), len(all_pairs)),
        "n_link_eligible_features": len(eligible), "n_link_eligible_pairs": len(eligible_pairs),
        "link_eligible_pair_coverage": _ratio(len(eligible_pairs), len(all_pairs)),
        "n_true_biological_pairs_representable": len(true_pairs & represented_pairs),
        "n_true_biological_pairs_unrepresented": len(true_pairs - represented_pairs),
        "n_true_biological_pairs_link_eligible": len(true_pairs & eligible_pairs),
        "n_true_biological_pairs_masked": len((true_pairs & represented_pairs) - eligible_pairs),
        "by_assay": {assay: {"scored": sum(row["assay"] == assay for row in candidates),
                             "represented": sum(by_name[name]["assay"] == assay for name in represented)}
                     for assay in sorted({row["assay"] for row in candidates})},
        "interpretation": "Unrepresented or explicitly masked features/pairs are abstentions, not negative guidance evidence; link eligibility here means not explicitly masked, not a guarantee of an annotation link",
    }
    result = {"coverage": coverage,
              "target": "Biological direct-pair adjacency stress test, not causal discovery or count-PAG truth",
              "direction": {"status": "not_applicable", "reason": "No causal directions are predicted"}}
    if prior["status"] != "ok":
        return {**result, "status": "not_applicable", "reason": prior["reason"], "adjacency": None}
    predicted = {tuple(sorted((row["node1"], row["node2"]))) for row in prior["edges"]}
    if not predicted <= eligible_pairs:
        raise ValueError("Prior contains an edge involving an unrepresented or masked scored feature")
    cross = {pair for pair in eligible_pairs if by_name[pair[0]]["assay"] != by_name[pair[1]]["assay"]}
    core = {pair for pair in eligible_pairs if all(by_name[name]["role"] == "core" for name in pair)}
    sentinel = {pair for pair in eligible_pairs if any(by_name[name]["role"] == "sentinel" for name in pair)}
    result.update(status="ok", adjacency=_adjacency_scores(predicted, true_pairs & eligible_pairs, eligible_pairs),
                  cross_assay=_adjacency_scores(predicted & cross, true_pairs & cross, cross),
                  core_pairs=_adjacency_scores(predicted & core, true_pairs & core, core),
                  sentinel_incident_pairs=_adjacency_scores(predicted & sentinel, true_pairs & sentinel, sentinel),
                  whole_panel_comparison={"status": "ok" if eligible_pairs == all_pairs else "not_applicable",
                                          "reason": "Use coverage plus eligible-pair metrics; omitted or masked vertices are not absent causal edges"})
    return result


def run_baseline(preprocessing_dir: Path, truth_manifest: Path, output: Path) -> dict[str, Any]:
    preprocessing_dir, truth_manifest = Path(preprocessing_dir).resolve(), Path(truth_manifest).resolve()
    manifest_path = preprocessing_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Guidance baseline requires a complete frozen preprocessing manifest")
    graph_path = Path(manifest["guidance"]["path"])
    if file_hash(graph_path) != manifest["guidance"]["sha256"]:
        raise ValueError("Exposed guidance graph changed after preprocessing freeze")
    graph = nx.read_graphml(graph_path)
    exposed = {feature for item in manifest["assays"].values() for feature in item["feature_ids"]}
    if set(graph.nodes) != exposed:
        raise ValueError("Graph vertex set differs from exposed reconstruction features")
    world = truth_manifest.parent.parent
    world_manifest_path = world / "manifest.json"
    world_manifest = json.loads(world_manifest_path.read_text())
    if truth_manifest.parent.name != "truth" or world_manifest.get("status") != "complete":
        raise ValueError("Provide truth/manifest.json from a complete frozen world")
    if world_manifest["files"].get("truth/manifest.json") != file_hash(truth_manifest):
        raise ValueError("Scoring truth differs from the world's frozen truth manifest")
    matching_regimes = []
    for regime in world_manifest["regimes"]:
        matches = True
        for assay in manifest["assays"]:
            relative = f"observed/{regime}/train/{assay}.h5ad"
            expected = world_manifest["files"].get(relative)
            if expected != manifest["inputs"][assay]["sha256"] or expected is None:
                matches = False
                break
            if file_hash(world / relative) != expected:
                raise ValueError("Frozen world's observed training counts changed")
        if matches:
            matching_regimes.append(regime)
    if not matching_regimes:
        raise ValueError("Preprocessing training hashes do not belong to the scored world")
    scoring_manifest = json.loads(truth_manifest.read_text())
    feature_map = [{key: row[key] for key in ("feature_id", "name", "assay")} for row in scoring_manifest["candidates"]]
    prior = extract_prior(graph, feature_map,
                          excluded_feature_ids=manifest["guidance"].get("excluded_cross_feature_ids", []))
    result = {"status": prior["status"], "prior": prior,
              "scoring": score_prior(prior, scoring_manifest["candidates"], scoring_manifest["biological_dag_edges"]),
              "inputs": {"preprocessing_manifest": str(manifest_path), "preprocessing_manifest_sha256": file_hash(manifest_path),
                         "guidance_graph": str(graph_path), "guidance_sha256": file_hash(graph_path),
                         "truth_manifest": str(truth_manifest), "truth_manifest_sha256": file_hash(truth_manifest),
                         "world_manifest_sha256": file_hash(world_manifest_path), "matched_count_regimes": matching_regimes,
                         "source_sha256": file_hash(Path(__file__))},
              "interpretation": "Annotation-only baseline using the graph actually exposed to integration; no learned or oriented causal edges"}
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != encoded:
            raise FileExistsError("Existing guidance baseline has different inputs or source")
    else:
        with output.open("x") as handle:
            handle.write(encoded)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-dir", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_baseline(args.preprocessing_dir, args.truth_manifest, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "coverage": result["scoring"]["coverage"]["feature_coverage"],
                      "n_undirected_prior_edges": None if result["prior"]["edges"] is None else len(result["prior"]["edges"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
