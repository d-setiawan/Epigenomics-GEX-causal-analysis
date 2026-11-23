#!/usr/bin/env python3
"""Collapse nested CA-BOSS v4 runs into one typed graph per world."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONSENSUS_COLUMNS = [
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "edge_type",
    "interpretation",
    "orientation_source",
    "selection_count",
    "selection_frequency",
    "n_nested_runs",
    "edge_type_vote_count",
    "endpoint1_vote_count",
    "endpoint2_vote_count",
    "endpoint_vote_count",
]

_PROJECTABLE_GROUP_TYPES = {
    "within_assay_shared_state",
    "cross_assay_shared_state",
    "prior_constrained_peak_gene",
}

_PAIRWISE_STATE_TYPES = set(_PROJECTABLE_GROUP_TYPES)


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def canonical_record(record: dict[str, Any]) -> tuple[tuple[str, str], tuple[str, str]]:
    left, right = str(record["node1"]), str(record["node2"])
    key = canonical_edge(left, right)
    endpoints = (
        (str(record.get("endpoint1", "CIRCLE")), str(record.get("endpoint2", "CIRCLE")))
        if key == (left, right)
        else (str(record.get("endpoint2", "CIRCLE")), str(record.get("endpoint1", "CIRCLE")))
    )
    return key, endpoints


def consensus_typed_graph(
    graphs: list[pd.DataFrame],
    *,
    minimum_count: int = 4,
    endpoint_fraction: float = 2 / 3,
    type_fraction: float = 2 / 3,
) -> pd.DataFrame:
    if not graphs:
        raise ValueError("At least one nested graph is required")
    if minimum_count < 1 or minimum_count > len(graphs):
        raise ValueError("minimum_count must be between one and the number of graphs")
    counts: Counter[tuple[str, str]] = Counter()
    endpoint1_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    endpoint2_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    type_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    orientation_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    numeric: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for graph in graphs:
        seen: set[tuple[str, str]] = set()
        for record in graph.to_dict(orient="records"):
            key, endpoints = canonical_record(record)
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            endpoint1_votes[key][endpoints[0]] += 1
            endpoint2_votes[key][endpoints[1]] += 1
            type_votes[key][str(record.get("edge_type", "unresolved"))] += 1
            orientation_votes[key][str(record.get("orientation_source", "abstained"))] += 1
            for field, value in record.items():
                if field in {"node1", "node2", "endpoint1", "endpoint2"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    numeric[key][field].append(float(value))
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counts.items()):
        if count < minimum_count:
            continue
        endpoint1, endpoint1_count = endpoint1_votes[key].most_common(1)[0]
        endpoint2, endpoint2_count = endpoint2_votes[key].most_common(1)[0]
        if endpoint1_count / count < endpoint_fraction:
            endpoint1 = "CIRCLE"
        if endpoint2_count / count < endpoint_fraction:
            endpoint2 = "CIRCLE"
        endpoints = (endpoint1, endpoint2)
        edge_type, edge_type_count = type_votes[key].most_common(1)[0]
        if edge_type_count / count < type_fraction:
            edge_type = "unresolved"
            endpoints = ("CIRCLE", "CIRCLE")
        orientation_source, orientation_count = orientation_votes[key].most_common(1)[0]
        if orientation_count / count < endpoint_fraction or endpoints == ("CIRCLE", "CIRCLE"):
            orientation_source = "consensus_abstained"
        row: dict[str, Any] = {
            "node1": key[0],
            "endpoint1": endpoints[0],
            "node2": key[1],
            "endpoint2": endpoints[1],
            "edge_type": edge_type,
            "interpretation": edge_type,
            "orientation_source": orientation_source,
            "selection_count": int(count),
            "selection_frequency": float(count / len(graphs)),
            "n_nested_runs": int(len(graphs)),
            "edge_type_vote_count": int(edge_type_count),
            "endpoint1_vote_count": int(endpoint1_count),
            "endpoint2_vote_count": int(endpoint2_count),
            "endpoint_vote_count": int(min(endpoint1_count, endpoint2_count)),
        }
        row.update(
            {
                f"mean__{field}": float(np.mean(values))
                for field, values in numeric[key].items()
                if values
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=CONSENSUS_COLUMNS)
    return pd.DataFrame(rows)


def attach_group_projection_once(
    consensus: pd.DataFrame,
    sparse_projection: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one six-view loading-group projection after nested consensus.

    The group layer is already a consensus over the six independently fitted
    loading views.  Treating it as six additional graph votes would be
    pseudo-replication.  This function therefore annotates an existing edge or
    adds one separately, while preserving any direct edge and its orientation.
    Unresolved group matches never enter the ordinary skeleton.
    """

    output = consensus.copy()
    for column, default in (
        ("group_projection_selected", False),
        ("group_projection_edge_type", ""),
        ("group_projection_source", ""),
        ("group_projection_confidence", np.nan),
        ("group_projection_group1", ""),
        ("group_projection_group2", ""),
    ):
        if column not in output:
            output[column] = default
    if sparse_projection.empty:
        return output
    required = {
        "node1",
        "node2",
        "edge_type",
        "selection_status",
        "projection_source",
        "group1",
        "group2",
        "confidence",
    }
    missing = sorted(required - set(sparse_projection.columns))
    if missing:
        raise ValueError(f"Group projection is missing columns: {missing}")
    selected = sparse_projection.loc[
        sparse_projection["selection_status"].astype(str).eq("selected")
        & sparse_projection["edge_type"].astype(str).isin(_PROJECTABLE_GROUP_TYPES)
    ].copy()
    selected["__key"] = selected.apply(
        lambda row: canonical_edge(str(row["node1"]), str(row["node2"])), axis=1
    )
    if selected["__key"].duplicated().any():
        raise ValueError("Group projection contains duplicate selected adjacencies")
    existing = {
        canonical_edge(str(row.node1), str(row.node2)): index
        for index, row in output.iterrows()
    }
    additions: list[dict[str, Any]] = []
    for projection in selected.to_dict(orient="records"):
        key = projection["__key"]
        annotation = {
            "group_projection_selected": True,
            "group_projection_edge_type": str(projection["edge_type"]),
            "group_projection_source": str(projection["projection_source"]),
            "group_projection_confidence": float(projection["confidence"]),
            "group_projection_group1": str(projection["group1"]),
            "group_projection_group2": str(projection["group2"]),
        }
        if key in existing:
            index = existing[key]
            for column, value in annotation.items():
                output.at[index, column] = value
            # Group evidence is compatible with an already selected direct
            # adjacency, but can never retype or reorient it.
            continue
        edge_type = str(projection["edge_type"])
        left, right = key
        if edge_type == "prior_constrained_peak_gene":
            left_is_gene = left.startswith("expr__")
            right_is_gene = right.startswith("expr__")
            if left_is_gene == right_is_gene:
                raise ValueError("A peak-gene projection must contain exactly one expression node")
            endpoints = ("ARROW", "TAIL") if left_is_gene else ("TAIL", "ARROW")
            orientation_source = "group_projection_gene_sink_prior"
        else:
            endpoints = ("CIRCLE", "CIRCLE")
            orientation_source = "group_projection_abstained"
        row: dict[str, Any] = {
            "node1": left,
            "endpoint1": endpoints[0],
            "node2": right,
            "endpoint2": endpoints[1],
            "edge_type": edge_type,
            "interpretation": (
                "single post-consensus projection of a loading-group hyperedge"
            ),
            "orientation_source": orientation_source,
            "selection_count": 0,
            "selection_frequency": np.nan,
            "n_nested_runs": int(
                output["n_nested_runs"].max()
                if "n_nested_runs" in output and not output.empty
                else 6
            ),
            "edge_type_vote_count": 0,
            "endpoint1_vote_count": 0,
            "endpoint2_vote_count": 0,
            "endpoint_vote_count": 0,
            **annotation,
        }
        additions.append(row)
        existing[key] = len(output) + len(additions) - 1
    if additions:
        output = pd.concat([output, pd.DataFrame(additions)], ignore_index=True, sort=False)
    return output.sort_values(["node1", "node2"]).reset_index(drop=True)


def direct_collider_children(consensus: pd.DataFrame) -> tuple[str, ...]:
    """Find nodes with at least two incoming direct-candidate arrowheads."""

    arrowheads: Counter[str] = Counter()
    direct = consensus.loc[
        consensus.get("edge_type", pd.Series(dtype=str)).astype(str).eq(
            "within_assay_direct_candidate"
        )
    ]
    for record in direct.to_dict(orient="records"):
        if str(record.get("endpoint1", "CIRCLE")) == "ARROW":
            arrowheads[str(record["node1"])] += 1
        if str(record.get("endpoint2", "CIRCLE")) == "ARROW":
            arrowheads[str(record["node2"])] += 1
    return tuple(sorted(node for node, count in arrowheads.items() if count >= 2))


def replace_pairwise_state_layer(
    consensus: pd.DataFrame,
    sparse_projection: pd.DataFrame,
) -> pd.DataFrame:
    """Use group-consensus projections as the sole pairwise state layer.

    Marginal pairwise state matches are useful inputs to group construction,
    but retaining both layers creates duplicate and occasionally contradictory
    state claims.  Direct candidates and unresolved residual claims survive;
    selected group projections then provide the sparse state visualization.
    """

    retained = consensus.loc[
        ~consensus.get("edge_type", pd.Series(dtype=str)).astype(str).isin(
            _PAIRWISE_STATE_TYPES
        )
    ].copy()
    return attach_group_projection_once(retained, sparse_projection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, action="append", required=True)
    parser.add_argument("--minimum-count", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graphs = [pd.read_csv(path, sep="\t", keep_default_na=False) for path in args.edges]
    consensus = consensus_typed_graph(graphs, minimum_count=args.minimum_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(args.out, sep="\t", index=False)
    print(
        f"Wrote {len(consensus)} typed edges from {len(graphs)} nested runs "
        f"(required {args.minimum_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
