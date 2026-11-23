#!/usr/bin/env python3
"""Score a CA-BOSS v4 graph against a separate typed attainable contract.

Discovery and scoring are intentionally separate processes.  This module reads
the scorer-only ``truth_typed_edges.tsv`` after edge selection and never runs a
search.  Cross-assay residual challenges and prior-constrained gene directions
are reported but excluded from the locked success gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_DIRECT = "within_assay_direct"
PRIMARY_SHARED = {"within_assay_shared_state", "cross_assay_shared_state"}
SLOW_GENE = "slow_peak_gene"
FAST_CHALLENGE = "fast_peak_gene_challenge"
SENTINEL_NULL = "sentinel_null"
MAX_FOCAL_EDGES_PER_WORLD = 30
MIN_PRIMARY_SKELETON_PRECISION = 0.80
EXPECTED_EDGE_TYPES = {
    PRIMARY_DIRECT: {"within_assay_direct_candidate"},
    "within_assay_shared_state": {"within_assay_shared_state"},
    "cross_assay_shared_state": {"cross_assay_shared_state"},
    SLOW_GENE: {"prior_constrained_peak_gene"},
}
ALLOWED_ESTIMATED_EDGE_TYPES = {
    "within_assay_direct_candidate",
    "within_assay_shared_state",
    "cross_assay_shared_state",
    "prior_constrained_peak_gene",
    "unresolved",
}
ALLOWED_ENDPOINTS = {"TAIL", "ARROW", "CIRCLE"}


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def estimated_map(edges: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in edges.to_dict(orient="records"):
        left, right = str(record["node1"]), str(record["node2"])
        if left.startswith("latent_state__") or right.startswith("latent_state__"):
            continue
        key = canonical_edge(left, right)
        if key == (left, right):
            endpoint_left = str(record.get("endpoint1", "CIRCLE"))
            endpoint_right = str(record.get("endpoint2", "CIRCLE"))
        else:
            endpoint_left = str(record.get("endpoint2", "CIRCLE"))
            endpoint_right = str(record.get("endpoint1", "CIRCLE"))
        mapped[key] = {
            **record,
            "canonical_endpoint1": endpoint_left,
            "canonical_endpoint2": endpoint_right,
        }
    return mapped


def _endpoint_at(record: dict[str, Any], node: str, key: tuple[str, str]) -> str:
    return str(
        record["canonical_endpoint1"] if node == key[0] else record["canonical_endpoint2"]
    )


def _truth_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _truth_node_set(record: dict[str, Any], column: str, fallback: str) -> tuple[str, ...]:
    raw = record.get(column, "")
    value = "" if raw is None or pd.isna(raw) else str(raw).strip()
    nodes = tuple(node for node in value.split(";") if node) if value else (fallback,)
    return tuple(dict.fromkeys(nodes))


def _truth_candidate_keys(record: dict[str, Any]) -> set[tuple[str, str]]:
    left = str(record["node1"])
    right = str(record["node2"])
    left_nodes = _truth_node_set(record, "node_set1", left)
    right_nodes = _truth_node_set(record, "node_set2", right)
    return {
        canonical_edge(node1, node2)
        for node1 in left_nodes
        for node2 in right_nodes
        if node1 != node2
    }


def score_v4(
    edges: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"node1", "node2", "truth_type", "label"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Typed truth is missing columns: {sorted(missing)}")
    edge_required = {"node1", "node2", "endpoint1", "endpoint2", "edge_type"}
    edge_missing = edge_required - set(edges.columns)
    if edge_missing:
        raise ValueError(f"Estimated graph is missing columns: {sorted(edge_missing)}")
    if truth["label"].astype(str).duplicated().any():
        raise ValueError("Typed truth labels must be unique")
    truth_keys = truth.apply(
        lambda row: canonical_edge(str(row["node1"]), str(row["node2"])), axis=1
    )
    if truth_keys.duplicated().any():
        raise ValueError("Typed truth pairs must be unique")
    biological = edges.loc[
        ~edges["node1"].astype(str).str.startswith("latent_state__")
        & ~edges["node2"].astype(str).str.startswith("latent_state__")
    ]
    invalid_types = sorted(
        set(biological["edge_type"].astype(str)) - ALLOWED_ESTIMATED_EDGE_TYPES
    )
    if invalid_types:
        raise ValueError(f"Unknown estimated edge types: {invalid_types}")
    invalid_endpoints = sorted(
        (
            set(biological["endpoint1"].astype(str))
            | set(biological["endpoint2"].astype(str))
        )
        - ALLOWED_ENDPOINTS
    )
    if invalid_endpoints:
        raise ValueError(f"Unknown estimated endpoints: {invalid_endpoints}")
    edge_keys = biological.apply(
        lambda row: canonical_edge(str(row["node1"]), str(row["node2"])), axis=1
    )
    if edge_keys.duplicated().any():
        raise ValueError("Estimated graph contains duplicate adjacencies")
    observed_nodes = set(truth["node1"].astype(str)) | set(truth["node2"].astype(str))
    unknown_nodes = (
        set(biological["node1"].astype(str))
        | set(biological["node2"].astype(str))
    ) - observed_nodes
    if unknown_nodes:
        raise ValueError(f"Estimated graph contains unknown observed nodes: {sorted(unknown_nodes)}")
    estimated = estimated_map(edges)
    pair_rows: list[dict[str, Any]] = []
    for record in truth.to_dict(orient="records"):
        left, right = str(record["node1"]), str(record["node2"])
        candidate_keys = _truth_candidate_keys(record)
        found_candidates = [
            (key, estimated[key]) for key in sorted(candidate_keys) if key in estimated
        ]
        adjacency_present = bool(found_candidates)
        truth_type = str(record["truth_type"])
        allowed_types = EXPECTED_EDGE_TYPES.get(truth_type)
        if allowed_types is None:
            # Nulls and the explicitly non-identifiable challenge are adjacency
            # diagnostics rather than typed recovery targets.
            typed_candidates = found_candidates
            present = adjacency_present
        else:
            typed_candidates = [
                (key, found)
                for key, found in found_candidates
                if str(found.get("edge_type", "")) in allowed_types
            ]
            present = bool(typed_candidates)
        chosen = (typed_candidates or found_candidates)[0] if adjacency_present else None
        key, found = chosen if chosen is not None else (canonical_edge(left, right), None)
        estimated_types = sorted(
            {str(item.get("edge_type", "")) for _, item in found_candidates}
        )
        estimated_type = ";".join(estimated_types)
        endpoint_left = _endpoint_at(found, left, key) if found and left in key else ""
        endpoint_right = _endpoint_at(found, right, key) if found and right in key else ""
        parent = str(record.get("parent", ""))
        child = str(record.get("child", ""))
        child_endpoint = _endpoint_at(found, child, key) if found and child in key else ""
        parent_endpoint = _endpoint_at(found, parent, key) if found and parent in key else ""
        orientation_credit = _truth_bool(record.get("orientation_credit", False))
        pair_rows.append(
            {
                **record,
                "adjacency_present": int(adjacency_present),
                "present": int(present),
                "estimated_edge_type": estimated_type,
                "wrong_layer": int(
                    adjacency_present and allowed_types is not None and not present
                ),
                "endpoint_node1": endpoint_left,
                "endpoint_node2": endpoint_right,
                "credited_child_arrowhead": int(
                    present and orientation_credit and child_endpoint == "ARROW"
                ),
                "strict_reverse_compelled": int(
                    present
                    and orientation_credit
                    and child_endpoint == "TAIL"
                    and parent_endpoint == "ARROW"
                ),
                "false_one_way_shared": int(
                    adjacency_present
                    and truth_type in PRIMARY_SHARED
                    and any(
                        {
                            str(item["canonical_endpoint1"]),
                            str(item["canonical_endpoint2"]),
                        }
                        == {"TAIL", "ARROW"}
                        for _, item in found_candidates
                    )
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    direct = pairs.loc[pairs["truth_type"] == PRIMARY_DIRECT]
    shared = pairs.loc[pairs["truth_type"].isin(PRIMARY_SHARED)]
    slow = pairs.loc[pairs["truth_type"] == SLOW_GENE]
    challenge = pairs.loc[pairs["truth_type"] == FAST_CHALLENGE]
    nulls = pairs.loc[pairs["truth_type"] == SENTINEL_NULL]
    if direct.empty or shared.empty or len(slow) != 1 or nulls.empty:
        raise ValueError(
            "Typed truth must contain direct, shared, one slow-gene, and sentinel-null rows"
        )
    mark_column = "mark" if "mark" in direct.columns else None
    if mark_column:
        per_mark_direct = direct.groupby(mark_column, dropna=False)["present"].mean()
        direct_macro_recall = float(per_mark_direct.mean())
        per_mark_raw_direct = direct.groupby(mark_column, dropna=False)[
            "adjacency_present"
        ].mean()
        direct_raw_adjacency_macro_recall = float(per_mark_raw_direct.mean())
    else:
        direct_macro_recall = float(direct["present"].mean())
        direct_raw_adjacency_macro_recall = float(
            direct["adjacency_present"].mean()
        )
    orientation_values = (
        direct["orientation_credit"]
        if "orientation_credit" in direct
        else pd.Series(False, index=direct.index)
    )
    credited = direct.loc[orientation_values.map(_truth_bool)]
    collider_recall = (
        float(credited["credited_child_arrowhead"].mean()) if not credited.empty else 0.0
    )
    shared_recall = float(shared["present"].mean())
    null_fpr = float(nulls["present"].mean())
    slow_present = int(slow["present"].iloc[0])
    reverse_count = int(direct["strict_reverse_compelled"].sum())
    false_shared_direction = int(shared["false_one_way_shared"].sum())
    truth_records = truth.to_dict(orient="records")
    focal_nodes = {
        node
        for record in truth_records
        for node in (
            *_truth_node_set(record, "node_set1", str(record["node1"])),
            *_truth_node_set(record, "node_set2", str(record["node2"])),
        )
    }
    estimated_focal = {
        key for key in estimated if key[0] in focal_nodes and key[1] in focal_nodes
    }
    positive_types = {PRIMARY_DIRECT, *PRIMARY_SHARED, SLOW_GENE}
    primary_truth_records = [
        row for row in truth_records if str(row["truth_type"]) in positive_types
    ]
    labeled_pairs = set().union(
        *(_truth_candidate_keys(row) for row in truth_records)
    )
    supported_estimated: set[tuple[str, str]] = set()
    wrong_layer_estimated: set[tuple[str, str]] = set()
    for row in primary_truth_records:
        allowed = EXPECTED_EDGE_TYPES[str(row["truth_type"])]
        for key in _truth_candidate_keys(row) & set(estimated):
            if str(estimated[key].get("edge_type", "")) in allowed:
                supported_estimated.add(key)
            else:
                wrong_layer_estimated.add(key)
    primary_rows = pairs.loc[pairs["truth_type"].isin(positive_types)]
    true_positives = int(primary_rows["present"].sum())
    sentinel_estimated = set().union(
        *(
            _truth_candidate_keys(row) & set(estimated)
            for row in truth_records
            if str(row["truth_type"]) == SENTINEL_NULL
        )
    )
    unsupported_estimated = estimated_focal - labeled_pairs
    false_estimated = (
        wrong_layer_estimated | sentinel_estimated | unsupported_estimated
    ) - supported_estimated
    wrong_layer_positives = int(len(wrong_layer_estimated - supported_estimated))
    sentinel_false_positives = int(len(sentinel_estimated - supported_estimated))
    precision = len(supported_estimated) / max(
        len(supported_estimated) + len(false_estimated), 1
    )
    recall = true_positives / max(len(primary_rows), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    direct_success = int(direct_macro_recall >= 0.90)
    collider_success = int(collider_recall >= 0.80 and reverse_count == 0)
    shared_success = int(shared_recall >= 0.80 and false_shared_direction == 0)
    slow_success = slow_present
    null_success = int(null_fpr <= 0.10)
    density_success = int(len(estimated_focal) <= MAX_FOCAL_EDGES_PER_WORLD)
    precision_success = int(precision >= MIN_PRIMARY_SKELETON_PRECISION)
    metrics: dict[str, Any] = {
        "n_estimated_focal_edges": int(len(estimated_focal)),
        "n_unlabeled_focal_edges": int(len(estimated_focal - labeled_pairs)),
        "n_unsupported_focal_edges": int(len(unsupported_estimated)),
        "n_false_typed_focal_edges": int(len(false_estimated)),
        "n_wrong_layer_primary_edges": wrong_layer_positives,
        "direct_raw_adjacency_macro_recall": direct_raw_adjacency_macro_recall,
        "direct_typed_candidate_macro_recall": direct_macro_recall,
        "direct_adjacency_macro_recall": direct_macro_recall,
        "collider_arrowhead_recall": collider_recall,
        "strict_reverse_compelled_count": reverse_count,
        "shared_state_adjacency_recall": shared_recall,
        "shared_state_false_one_way_count": false_shared_direction,
        "slow_peak_gene_present": slow_present,
        "fast_challenge_present": int(challenge["present"].mean() >= 0.5)
        if not challenge.empty
        else 0,
        "matched_null_false_positive_rate": null_fpr,
        "primary_skeleton_precision": float(precision),
        "sentinel_anchored_typed_precision": float(precision),
        "primary_skeleton_recall": float(recall),
        "primary_skeleton_f1": float(f1),
        "direct_class_success": direct_success,
        "collider_class_success": collider_success,
        "shared_class_success": shared_success,
        "slow_class_success": slow_success,
        "null_class_success": null_success,
        "density_class_success": density_success,
        "precision_class_success": precision_success,
        "minimum_primary_skeleton_precision": MIN_PRIMARY_SKELETON_PRECISION,
        "maximum_focal_edges_allowed": MAX_FOCAL_EDGES_PER_WORLD,
        "world_success": int(
            direct_success
            and collider_success
            and shared_success
            and slow_success
            and null_success
            and density_success
            and precision_success
        ),
        "gene_sink_orientations_credited": False,
        "fast_residual_edge_in_pass_fail": False,
        "direct_adjacency_gate_requires_direct_candidate_type": True,
    }
    return metrics, pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    edges = pd.read_csv(args.edges, sep="\t", keep_default_na=False)
    truth = pd.read_csv(args.truth, sep="\t", keep_default_na=False)
    metrics, pairs = score_v4(edges, truth)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / "typed_pair_scores.tsv", sep="\t", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
