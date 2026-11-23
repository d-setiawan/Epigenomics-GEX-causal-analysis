#!/usr/bin/env python3
"""Score an estimated focal graph under the simulation's attainable contract.

This scorer is intentionally separate from discovery.  Search implementations
must not read the semantics, DAG, membership, or ground-truth tables consumed
here.  The primary target contains recoverable direct/state adjacencies, two
shared-driver adjacencies, and eight sentinel-null pairs.  The residual-driven
cross-modality gene edge is reported only as an identifiability challenge.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def role_node(semantics: pd.DataFrame, role: str) -> str:
    matches = semantics.loc[semantics["role"] == role, "node"].astype(str).tolist()
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one node with role {role!r}, found {matches}")
    return matches[0]


def endpoint_map(edges: pd.DataFrame) -> dict[tuple[str, str], tuple[str, str]]:
    mapped: dict[tuple[str, str], tuple[str, str]] = {}
    for row in edges.to_dict(orient="records"):
        left, right = str(row["node1"]), str(row["node2"])
        if left.startswith("conf__") or right.startswith("conf__"):
            continue
        key = canonical_edge(left, right)
        endpoints = (
            (str(row["endpoint1"]), str(row["endpoint2"]))
            if (left, right) == key
            else (str(row["endpoint2"]), str(row["endpoint1"]))
        )
        mapped[key] = endpoints
    return mapped


def direction_status(
    estimated: dict[tuple[str, str], tuple[str, str]],
    parent: str,
    child: str,
) -> dict[str, int]:
    key = canonical_edge(parent, child)
    if key not in estimated:
        return {"present": 0, "strict_correct": 0, "pag_compatible": 0, "strict_reverse": 0}
    endpoints = estimated[key]
    parent_endpoint, child_endpoint = endpoints if key == (parent, child) else endpoints[::-1]
    return {
        "present": 1,
        "strict_correct": int(parent_endpoint == "TAIL" and child_endpoint == "ARROW"),
        "pag_compatible": int(child_endpoint == "ARROW" and parent_endpoint != "ARROW"),
        "strict_reverse": int(parent_endpoint == "ARROW" and child_endpoint == "TAIL"),
    }


def build_contract(semantics: pd.DataFrame) -> dict[str, Any]:
    gene = role_node(semantics, "target_gene")
    ac_cause = role_node(semantics, "slow_state_dominated_direct_gene_cause")
    # There are two nodes whose roles describe a same-mark child/cause, so use
    # mark-qualified lookup for the H3K4me3 pair.
    me_cause = role_node(semantics, "fast_residual_dominated_direct_gene_cause")
    child_candidates = semantics.loc[
        semantics["role"] == "same_mark_child_of_peak0", "node"
    ].astype(str).tolist()
    ac_children = [node for node in child_candidates if node.endswith("__H3K27ac")]
    me_children = [node for node in child_candidates if node.endswith("__H3K4me3")]
    if len(ac_children) != 1 or len(me_children) != 1:
        raise ValueError("Could not uniquely identify both same-mark child nodes")
    ac_child, me_child = ac_children[0], me_children[0]

    ac_shared = [
        ac_cause,
        role_node(semantics, "shared_driver_peer_no_direct_gene_edge"),
    ]
    me_shared = [
        role_node(semantics, "shared_driver_member_a"),
        role_node(semantics, "shared_driver_member_b"),
    ]
    null_nodes = semantics.loc[semantics["role"] == "null", "node"].astype(str).tolist()
    ac_null = [node for node in null_nodes if node.endswith("__H3K27ac")]
    me_null = [node for node in null_nodes if node.endswith("__H3K4me3")]
    if len(ac_null) != 2 or len(me_null) != 2:
        raise ValueError("Expected two sentinel nulls in each chromatin channel")

    # Attainable matched-state MAG.  After independently sampling modalities,
    # residual cross-channel effects are severed, but marginalizing latent state
    # creates inducing paths among every observed descendant of that state.  z0
    # also drives z1 in monocytes, so its descendant set includes the z1-loaded
    # H3K27ac variables.  z2 defines a second overlapping clique.
    z0_z1_group = {
        gene,
        ac_cause,
        ac_child,
        ac_shared[1],
        role_node(semantics, "state_associated_not_direct"),
    }
    z2_group = {
        gene,
        ac_child,
        me_cause,
        me_child,
        *me_shared,
    }
    matched_adjacencies = sorted(
        {
            canonical_edge(left, right)
            for group in (z0_z1_group, z2_group)
            for left, right in combinations(sorted(group), 2)
        }
    )

    direct = [
        (ac_cause, gene, "slow_gene"),
        (ac_cause, ac_child, "H3K27ac_direct"),
        (me_cause, me_child, "H3K4me3_direct"),
    ]
    shared = [
        (*ac_shared, "H3K27ac_shared"),
        (*me_shared, "H3K4me3_shared"),
    ]
    nulls = [
        *((node, gene, f"null_gene_{i}") for i, node in enumerate(ac_null + me_null)),
        *((left, right, f"null_cross_{i}_{j}") for (i, left), (j, right) in product(enumerate(ac_null), enumerate(me_null))),
    ]
    return {
        "direct": direct,
        "shared": shared,
        "null": nulls,
        "challenge": [(me_cause, gene, "fast_gene_unidentifiable")],
        "matched_adjacencies": matched_adjacencies,
        "focal_nodes": sorted(semantics["node"].astype(str).tolist()),
    }


def score(edges: pd.DataFrame, semantics: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    estimated = endpoint_map(edges)
    contract = build_contract(semantics)
    rows: list[dict[str, Any]] = []
    for truth_class in ("direct", "shared", "null", "challenge"):
        for left, right, label in contract[truth_class]:
            key = canonical_edge(left, right)
            endpoints = estimated.get(key, ("", ""))
            row: dict[str, Any] = {
                "truth_class": truth_class,
                "label": label,
                "node1": left,
                "node2": right,
                "present": int(key in estimated),
                "endpoint_node1": endpoints[0] if key == (left, right) else endpoints[1],
                "endpoint_node2": endpoints[1] if key == (left, right) else endpoints[0],
            }
            if truth_class in {"direct", "challenge"}:
                row.update(direction_status(estimated, left, right))
            if truth_class == "shared":
                row["false_one_way"] = int(set(endpoints) == {"TAIL", "ARROW"})
            rows.append(row)
    pair_scores = pd.DataFrame(rows)

    direct_rows = pair_scores[pair_scores["truth_class"] == "direct"]
    shared_rows = pair_scores[pair_scores["truth_class"] == "shared"]
    null_rows = pair_scores[pair_scores["truth_class"] == "null"]
    challenge_rows = pair_scores[pair_scores["truth_class"] == "challenge"]
    positive_hits = int(direct_rows["present"].sum() + shared_rows["present"].sum())
    null_hits = int(null_rows["present"].sum())
    sentinel_precision = positive_hits / max(positive_hits + null_hits, 1)
    sentinel_recall = positive_hits / 5
    sentinel_f1 = (
        2 * sentinel_precision * sentinel_recall / (sentinel_precision + sentinel_recall)
        if sentinel_precision + sentinel_recall
        else 0.0
    )
    classified = {
        canonical_edge(left, right)
        for truth_class in (contract["direct"], contract["shared"], contract["null"], contract["challenge"])
        for left, right, _ in truth_class
    }
    focal_nodes = set(contract["focal_nodes"])
    estimated_focal = {
        key for key in estimated if key[0] in focal_nodes and key[1] in focal_nodes
    }
    matched_truth = set(contract["matched_adjacencies"])
    matched_tp = len(estimated_focal & matched_truth)
    matched_fp = len(estimated_focal - matched_truth)
    matched_fn = len(matched_truth - estimated_focal)
    matched_precision = matched_tp / max(matched_tp + matched_fp, 1)
    matched_recall = matched_tp / max(len(matched_truth), 1)
    matched_f1 = (
        2 * matched_precision * matched_recall / (matched_precision + matched_recall)
        if matched_precision + matched_recall
        else 0.0
    )
    metrics: dict[str, Any] = {
        "n_estimated_biological_edges": int(len(estimated_focal)),
        "n_unclassified_edges": int(len(estimated_focal - classified)),
        "n_matched_truth_edges": int(len(matched_truth)),
        "matched_skeleton_true_positives": int(matched_tp),
        "matched_skeleton_false_positives": int(matched_fp),
        "matched_skeleton_false_negatives": int(matched_fn),
        "matched_skeleton_precision": float(matched_precision),
        "matched_skeleton_recall": float(matched_recall),
        "matched_skeleton_f1": float(matched_f1),
        "matched_skeleton_shd": int(matched_fp + matched_fn),
        "direct_adjacency_recall": float(direct_rows["present"].mean()),
        "shared_adjacency_recall": float(shared_rows["present"].mean()),
        "null_false_adjacency_rate": float(null_rows["present"].mean()),
        "sentinel_precision": float(sentinel_precision),
        "sentinel_recall": float(sentinel_recall),
        "sentinel_f1": float(sentinel_f1),
        "strict_direct_orientation_recall": float(direct_rows["strict_correct"].mean()),
        "pag_compatible_orientation_recall": float(direct_rows["pag_compatible"].mean()),
        "same_mark_wrong_direction_count": int(
            direct_rows.loc[direct_rows["label"] != "slow_gene", "strict_reverse"].sum()
        ),
        "shared_false_one_way_count": int(shared_rows["false_one_way"].sum()),
        "challenge_fast_gene_recovered": int(challenge_rows["present"].iloc[0]),
    }
    metrics["world_success"] = int(
        direct_rows["present"].sum() == 3
        and shared_rows["present"].sum() == 2
        and null_hits <= 1
        and metrics["same_mark_wrong_direction_count"] == 0
        and metrics["shared_false_one_way_count"] == 0
    )
    metrics["matched_world_success"] = int(
        metrics["matched_skeleton_precision"] >= 0.80
        and metrics["matched_skeleton_recall"] >= 0.80
        and metrics["same_mark_wrong_direction_count"] == 0
        and metrics["shared_false_one_way_count"] == 0
    )
    return metrics, pair_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--semantics", type=Path)
    parser.add_argument("--simulation-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    semantics_path = args.semantics
    if semantics_path is None and args.simulation_dir:
        semantics_path = args.simulation_dir / "focal_feature_semantics.tsv"
    if semantics_path is None:
        raise SystemExit("Provide --semantics or --simulation-dir")
    edges = pd.read_csv(args.edges, sep="\t")
    # ``null`` is a semantic role, not a missing-value token.
    semantics = pd.read_csv(semantics_path, sep="\t", keep_default_na=False)
    metrics, pair_scores = score(edges, semantics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_scores.to_csv(args.out_dir / "pair_scores.tsv", sep="\t", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
