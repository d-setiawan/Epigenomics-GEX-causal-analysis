#!/usr/bin/env python3
"""Repeated-world benchmark for Channel-Aware Multi-resolution BOSS.

The script keeps one JVM alive, evaluates a fixed candidate grid on cached
simulation worlds, and writes both per-world and aggregate results.  A candidate
receives only observed matrices, observed single-cell tables, and analysis
embeddings.  Truth is loaded only after discovery by the separate scorer.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from channel_aware_boss import ChannelAwareConfig, discover
from score_multimodal_causal_graph import score


def candidate_grid(seed: int) -> list[dict[str, Any]]:
    base = ChannelAwareConfig(seed=seed)
    candidates: list[dict[str, Any]] = [{"name": "production_fci", "kind": "existing"}]
    candidates.append(
        {
            "name": "channel_final_v1",
            "kind": "discover",
            "use_cells": True,
            "config": replace(
                base,
                within_sample_prior=5.0,
                sparse_alpha=0.20,
                state_boss_alpha=1.0,
            ),
        }
    )
    candidates.append(
        {
            "name": "channel_final_v2",
            "kind": "discover",
            "use_cells": True,
            "config": replace(base, within_sample_prior=5.0, sparse_alpha=0.20),
        }
    )
    candidates.append(
        {
            "name": "channel_final_v3",
            "kind": "discover",
            "use_cells": True,
            "config": replace(
                base,
                within_sample_prior=5.0,
                sparse_alpha=0.25,
                within_profile_clusters=20,
                within_profile_min_abs_r=0.60,
                within_profile_alpha=0.05,
            ),
        }
    )
    for penalty in (0.5, 1.0, 2.0, 4.0):
        candidates.append(
            {
                "name": f"boss_only_p{penalty:g}",
                "kind": "discover",
                "use_cells": False,
                "config": replace(
                    base,
                    state_penalty=penalty,
                    state_boss_min_abs_r=0.0,
                    state_marginal_min_abs_r=2.0,
                    state_marginal_alpha=0.0,
                    sparse_alpha=0.0,
                ),
            }
        )
    for minimum_r in (0.15, 0.20, 0.25):
        candidates.append(
            {
                "name": f"state_aug_r{minimum_r:g}",
                "kind": "discover",
                "use_cells": False,
                "config": replace(
                    base,
                    state_marginal_min_abs_r=minimum_r,
                    sparse_alpha=0.0,
                ),
            }
        )
    for prior in (1.0, 5.0, 10.0, 20.0):
        candidates.append(
            {
                "name": f"channel_bdeu_prior{prior:g}",
                "kind": "discover",
                "use_cells": True,
                "config": replace(base, within_sample_prior=prior, sparse_alpha=0.0),
            }
        )
    for prior in (5.0, 10.0):
        for alpha in (0.05, 0.10, 0.20):
            candidates.append(
                {
                    "name": f"channel_sparse_prior{prior:g}_a{alpha:g}",
                    "kind": "discover",
                    "use_cells": True,
                    "config": replace(
                        base,
                        within_sample_prior=prior,
                        sparse_alpha=alpha,
                    ),
                }
            )
    return candidates


def parse_world_seed(run_dir: Path, fallback: int) -> int:
    config_path = run_dir / "config.json"
    if config_path.exists():
        return int(json.loads(config_path.read_text()).get("seed", fallback))
    return fallback


def evaluate_candidate(
    candidate: dict[str, Any],
    run_dir: Path,
    *,
    arm: str,
    out_dir: Path,
    fallback_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate_name = str(candidate["name"])
    world_name = run_dir.name
    world_out = out_dir / candidate_name / world_name
    world_out.mkdir(parents=True, exist_ok=True)

    if candidate["kind"] == "existing":
        edges_path = run_dir / f"{arm}_fci_edges.tsv"
        edges = pd.read_csv(edges_path, sep="\t")
        search_summary: dict[str, Any] = {
            "method": "production_causal_learn_fci",
            "inputs": {"edges": str(edges_path)},
        }
    else:
        cfg: ChannelAwareConfig = replace(
            candidate["config"],
            seed=parse_world_seed(run_dir, fallback_seed),
        )
        metacell = run_dir / f"{arm}_observed_metacell_matrix.tsv"
        embedding = run_dir / f"{arm}_analysis_embeddings.tsv"
        cell_tables = (
            sorted(run_dir.glob("observed_cells__chrom_*.tsv"))
            if candidate.get("use_cells", True)
            else []
        )
        edges, search_summary = discover(
            metacell,
            cell_tables,
            embedding if embedding.exists() else None,
            cfg,
        )

    # Persist every estimate, including the frozen production baseline, so the
    # same nested-consensus implementation can be applied to every candidate.
    edges.to_csv(world_out / "edges.tsv", sep="\t", index=False)
    (world_out / "run_summary.json").write_text(
        json.dumps(search_summary, indent=2, allow_nan=True)
    )

    # Truth enters only here, after the estimate has been frozen on disk/in memory.
    semantics = pd.read_csv(
        run_dir / "focal_feature_semantics.tsv",
        sep="\t",
        keep_default_na=False,
    )
    metrics, pair_scores = score(edges, semantics)
    pair_scores.to_csv(world_out / "pair_scores.tsv", sep="\t", index=False)
    (world_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    row = {
        "candidate": candidate_name,
        "world": world_name,
        "world_seed": parse_world_seed(run_dir, fallback_seed),
        **metrics,
    }
    for pair in pair_scores.to_dict(orient="records"):
        row[f"present__{pair['label']}"] = int(pair["present"])
    return row, pair_scores


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "world_success",
        "matched_world_success",
        "matched_skeleton_precision",
        "matched_skeleton_recall",
        "matched_skeleton_f1",
        "matched_skeleton_shd",
        "direct_adjacency_recall",
        "shared_adjacency_recall",
        "null_false_adjacency_rate",
        "sentinel_precision",
        "sentinel_recall",
        "sentinel_f1",
        "n_estimated_biological_edges",
        "n_unclassified_edges",
        "same_mark_wrong_direction_count",
        "shared_false_one_way_count",
        "challenge_fast_gene_recovered",
    ]
    present_columns = [column for column in results.columns if column.startswith("present__")]
    summary = results.groupby("candidate")[metrics + present_columns].mean().reset_index()
    summary["n_worlds"] = results.groupby("candidate").size().to_numpy()
    summary["successes"] = (
        results.groupby("candidate")["world_success"].sum().astype(int).to_numpy()
    )
    summary["matched_successes"] = (
        results.groupby("candidate")["matched_world_success"].sum().astype(int).to_numpy()
    )
    return summary.sort_values(
        ["matched_world_success", "matched_skeleton_f1", "world_success", "n_estimated_biological_edges"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-glob", required=True)
    parser.add_argument("--arm", choices=["oracle", "scglue", "gapca"], default="oracle")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidates",
        default="all",
        help="Comma-separated candidate names after grid construction, or 'all'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path) for path in sorted(glob.glob(args.runs_glob))]
    if not run_dirs:
        raise SystemExit(f"No run directories matched: {args.runs_glob}")
    candidates = candidate_grid(args.seed)
    if args.candidates != "all":
        wanted = {token.strip() for token in args.candidates.split(",") if token.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in wanted]
        missing = wanted - {candidate["name"] for candidate in candidates}
        if missing:
            raise SystemExit(f"Unknown candidates: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate_i, candidate in enumerate(candidates):
        print(f"[{candidate_i + 1}/{len(candidates)}] {candidate['name']}")
        for world_i, run_dir in enumerate(run_dirs):
            row, _ = evaluate_candidate(
                candidate,
                run_dir,
                arm=args.arm,
                out_dir=args.out_dir,
                fallback_seed=args.seed + world_i,
            )
            rows.append(row)
            print(
                f"  {run_dir.name}: success={row['world_success']} "
                f"F1={row['sentinel_f1']:.3f} edges={row['n_estimated_biological_edges']}"
            )

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(args.out_dir / "world_results.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "candidate_summary.tsv", sep="\t", index=False)
    manifest = {
        "runs_glob": args.runs_glob,
        "arm": args.arm,
        "worlds": [str(path) for path in run_dirs],
        "candidates": [
            {
                "name": candidate["name"],
                "kind": candidate["kind"],
                "use_cells": candidate.get("use_cells"),
                "config": asdict(candidate["config"]) if "config" in candidate else None,
            }
            for candidate in candidates
        ],
    }
    (args.out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
