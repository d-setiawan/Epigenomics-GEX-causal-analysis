#!/usr/bin/env python3
"""Collapse nested metacell/search runs into one graph per biological world."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from score_multimodal_causal_graph import canonical_edge, score


def canonical_record(row: dict[str, Any]) -> tuple[tuple[str, str], tuple[str, str]]:
    left, right = str(row["node1"]), str(row["node2"])
    key = canonical_edge(left, right)
    endpoints = (
        (str(row["endpoint1"]), str(row["endpoint2"]))
        if (left, right) == key
        else (str(row["endpoint2"]), str(row["endpoint1"]))
    )
    return key, endpoints


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def consensus_graph(paths: list[Path], min_frequency: float) -> pd.DataFrame:
    adjacency_counts: Counter[tuple[str, str]] = Counter()
    endpoint_votes: dict[tuple[str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        seen: set[tuple[str, str]] = set()
        for row in frame.to_dict(orient="records"):
            key, endpoints = canonical_record(row)
            if key in seen:
                continue
            seen.add(key)
            adjacency_counts[key] += 1
            endpoint_votes[key][endpoints] += 1
            evidence[key].update(
                token
                for token in str(row.get("evidence_layers", "")).split(",")
                if token
            )

    required = max(1, int(math.ceil(min_frequency * len(paths) - 1e-12)))
    rows: list[dict[str, Any]] = []
    for key, count in sorted(adjacency_counts.items()):
        if count < required:
            continue
        votes = endpoint_votes[key]
        endpoints, endpoint_count = votes.most_common(1)[0]
        # Endpoint calls require a two-thirds majority among runs containing the
        # adjacency.  Otherwise preserve the adjacency and abstain on direction.
        if endpoint_count / count < 2 / 3:
            endpoints = ("CIRCLE", "CIRCLE")
        left, right = key
        symbol = (
            "o->"
            if endpoints == ("CIRCLE", "ARROW")
            else "<-o"
            if endpoints == ("ARROW", "CIRCLE")
            else "-->"
            if endpoints == ("TAIL", "ARROW")
            else "<--"
            if endpoints == ("ARROW", "TAIL")
            else "o-o"
        )
        rows.append(
            {
                "node1": left,
                "endpoint1": endpoints[0],
                "node2": right,
                "endpoint2": endpoints[1],
                "edge_text": f"{left} {symbol} {right}",
                "selection_count": count,
                "selection_frequency": count / len(paths),
                "n_nested_runs": len(paths),
                "evidence_layers": ",".join(sorted(evidence[key])),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-candidate-dir", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--world-regex", default=r"seed(\d+)")
    parser.add_argument("--min-frequency", type=float, default=2 / 3)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pattern = re.compile(args.world_regex)
    grouped: dict[str, list[Path]] = defaultdict(list)
    run_names: dict[str, list[str]] = defaultdict(list)
    for path in sorted(args.benchmark_candidate_dir.glob("*/edges.tsv")):
        match = pattern.search(path.parent.name)
        if not match:
            continue
        world = match.group(1)
        grouped[world].append(path)
        run_names[world].append(path.parent.name)
    if not grouped:
        raise SystemExit("No nested edge files matched the world regex")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    for world, paths in sorted(grouped.items(), key=lambda item: int(item[0])):
        edges = consensus_graph(paths, args.min_frequency)
        world_out = args.out_dir / f"world_{world}"
        world_out.mkdir(parents=True, exist_ok=True)
        edges.to_csv(world_out / "consensus_edges.tsv", sep="\t", index=False)

        simulation_run = args.simulation_root / run_names[world][0]
        if not simulation_run.exists():
            # The first benchmark run may carry an _mc suffix while another is
            # the base run; any nested run has the same world-level semantics.
            candidates = sorted(args.simulation_root.glob(f"*seed{world}*"))
            if not candidates:
                raise FileNotFoundError(f"No simulation directory for world {world}")
            simulation_run = candidates[0]
        semantics = pd.read_csv(
            simulation_run / "focal_feature_semantics.tsv",
            sep="\t",
            keep_default_na=False,
        )
        metrics, pair_scores = score(edges, semantics)
        pair_scores.to_csv(world_out / "pair_scores.tsv", sep="\t", index=False)
        (world_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        result_row = {
            "world": int(world),
            "n_nested_runs": len(paths),
            **metrics,
        }
        for pair in pair_scores.to_dict(orient="records"):
            result_row[f"present__{pair['label']}"] = int(pair["present"])
        result_rows.append(result_row)

    results = pd.DataFrame(result_rows)
    results.to_csv(args.out_dir / "world_consensus_results.tsv", sep="\t", index=False)
    pair_columns = sorted(column for column in results if column.startswith("present__"))
    pair_recovery = {
        column.removeprefix("present__"): {
            "count": int(results[column].sum()),
            "rate": float(results[column].mean()),
        }
        for column in pair_columns
    }
    n_worlds = int(len(results))
    successes = int(results["world_success"].sum())
    interval_low, interval_high = wilson_interval(successes, n_worlds)
    primary_labels = {
        "slow_gene",
        "H3K27ac_direct",
        "H3K4me3_direct",
        "H3K27ac_shared",
        "H3K4me3_shared",
    }
    primary_counts = [
        value["count"] for label, value in pair_recovery.items() if label in primary_labels
    ]
    null_counts = [
        value["count"] for label, value in pair_recovery.items() if label.startswith("null_")
    ]
    required_successes = int(math.ceil(0.85 * n_worlds - 1e-12))
    allowed_null_occurrences = int(math.floor(0.075 * n_worlds + 1e-12))
    reliability_gate = {
        "required_world_successes": required_successes,
        "observed_world_successes": successes,
        "minimum_primary_pair_count": int(min(primary_counts)),
        "maximum_null_pair_count": int(max(null_counts)),
        "allowed_null_pair_count": allowed_null_occurrences,
    }
    reliability_gate["passed"] = bool(
        successes >= required_successes
        and min(primary_counts) >= required_successes
        and max(null_counts) <= allowed_null_occurrences
    )
    summary = {
        "n_worlds": n_worlds,
        "successes": successes,
        "success_rate": float(results["world_success"].mean()),
        "success_rate_wilson_95": {
            "lower": interval_low,
            "upper": interval_high,
        },
        "matched_successes": int(results["matched_world_success"].sum()),
        "mean_sentinel_f1": float(results["sentinel_f1"].mean()),
        "mean_null_false_adjacency_rate": float(results["null_false_adjacency_rate"].mean()),
        "mean_matched_skeleton_f1": float(results["matched_skeleton_f1"].mean()),
        "mean_estimated_biological_edges": float(
            results["n_estimated_biological_edges"].mean()
        ),
        "pair_recovery": pair_recovery,
        "reliability_gate": reliability_gate,
        "min_frequency": args.min_frequency,
    }
    (args.out_dir / "consensus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
