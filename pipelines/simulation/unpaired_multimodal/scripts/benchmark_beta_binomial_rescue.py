#!/usr/bin/env python3
"""Freeze and score the CA-BOSS v4 beta-binomial sparse-edge prototype.

The command has a hard phase boundary.  ``fit`` accepts only discovery-domain
artifacts, recomputes the background alignment, writes six independently seeded
evidence votes, and hashes every consensus before returning.  ``score`` first
verifies those hashes and only then opens the scorer-domain typed truth.

The benchmark is intentionally restricted to TRAIN worlds 30003, 30004, and
30008.  DEV and LOCKED paths are rejected.  The predeclared comparison includes
the rescue alone, the production posterior-positive hierarchical rule alone,
and their union; no threshold is changed during scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_ca_boss_v4_inputs import PrepareConfig, prepare_inputs
from stratified_beta_binomial_edges import (
    StratifiedBetaBinomialConfig,
    canonical_edge,
    estimate_from_observed_tables,
    frame_sha256,
)


ALLOWED_WORLD_NAMES = {"world_seed30003", "world_seed30004", "world_seed30008"}
N_VOTES = 6
MINIMUM_VOTES = 4
CURRENT_POSITIVE_PROBABILITY = 0.88
FROZEN_CONFIG = StratifiedBetaBinomialConfig()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_world(world: Path) -> None:
    if world.name not in ALLOWED_WORLD_NAMES:
        raise ValueError(f"Prototype permits only {sorted(ALLOWED_WORLD_NAMES)}")
    if any(token in str(world).lower() for token in ("dev", "locked")):
        raise ValueError("DEV and LOCKED worlds are forbidden")
    if not (world / "discovery").is_dir():
        raise ValueError(f"Missing discovery domain: {world}")


def _feature_columns(path: Path) -> list[str]:
    return [
        str(column)
        for column in pd.read_csv(path, sep="\t", nrows=0).columns
        if str(column).startswith("chr")
    ]


def _consensus(evidence: pd.DataFrame, selected_column: str = "selected") -> pd.DataFrame:
    frame = evidence.copy()
    frame["pair"] = frame.apply(
        lambda row: canonical_edge(str(row["node1"]), str(row["node2"])), axis=1
    )
    rows: list[dict[str, Any]] = []
    for (assay, pair), group in frame.groupby(["source_assay", "pair"], sort=True):
        votes = int(group[selected_column].astype(bool).sum())
        rows.append(
            {
                "source_assay": assay,
                "node1": pair[0],
                "node2": pair[1],
                "selected_vote_count": votes,
                "vote_count": int(group["vote"].nunique()),
                "selected": bool(votes >= MINIMUM_VOTES),
                "mean_empirical_p_positive": float(
                    group["empirical_p_positive"].mean()
                )
                if "empirical_p_positive" in group
                else float("nan"),
                "mean_pooled_log_odds": float(group["pooled_log_odds"].mean())
                if "pooled_log_odds" in group
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selected", "selected_vote_count", "source_assay", "node1", "node2"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def _current_hierarchy_consensus(world_out: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for vote, directory in enumerate(
        ("cells24_seed0", "cells24_seed1", "cells14_seed0", "cells14_seed1", "cells8_seed0", "cells8_seed1")
    ):
        path = world_out / directory / "hierarchical_sparse_edge_evidence.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing current hierarchy evidence: {path}")
        frame = pd.read_csv(path, sep="\t")
        frame["vote"] = vote
        frame["selected"] = (
            frame["posterior_probability_positive"] >= CURRENT_POSITIVE_PROBABILITY
        )
        frames.append(frame)
    current = pd.concat(frames, ignore_index=True)
    return _consensus(current)


def _selected_pairs(consensus: pd.DataFrame) -> set[tuple[str, str]]:
    selected = consensus.loc[consensus["selected"].astype(bool)]
    return {
        canonical_edge(str(row["node1"]), str(row["node2"]))
        for row in selected.to_dict(orient="records")
    }


def _pair_frame(pairs: set[tuple[str, str]], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node1": pair[0], "node2": pair[1], "selected": True, "source": source}
            for pair in sorted(pairs)
        ],
        columns=["node1", "node2", "selected", "source"],
    )


def fit_world(
    world: Path,
    out_root: Path,
    current_root: Path,
) -> dict[str, Any]:
    _validate_world(world)
    world_out = out_root / world.name
    world_out.mkdir(parents=True, exist_ok=True)
    prepared = world_out / "prepared_discovery"
    prepare_inputs(world / "discovery", prepared, PrepareConfig())
    embedding = prepared / "analysis_embeddings.tsv"
    cell_tables = sorted((world / "discovery").glob("observed_focal__chrom_*.tsv"))
    if len(cell_tables) != 6:
        raise ValueError(f"Expected six chromatin assays in {world}")

    vote_frames: list[pd.DataFrame] = []
    evidence_hashes: dict[str, str] = {}
    for vote in range(N_VOTES):
        assay_frames: list[pd.DataFrame] = []
        metadata: list[dict[str, Any]] = []
        for assay_i, table in enumerate(cell_tables):
            assay = table.stem.removeprefix("observed_focal__")
            config = replace(
                FROZEN_CONFIG,
                random_state=vote + 10_007 * (assay_i + 1),
            )
            result = estimate_from_observed_tables(
                table,
                embedding,
                feature_columns=_feature_columns(table),
                config=config,
            )
            evidence = result.edge_evidence.copy()
            evidence["source_assay"] = assay
            evidence["vote"] = vote
            assay_frames.append(evidence)
            metadata.append({"source_assay": assay, **result.metadata})
        combined = pd.concat(assay_frames, ignore_index=True).sort_values(
            ["source_assay", "node1", "node2"]
        ).reset_index(drop=True)
        path = world_out / f"rescue_evidence_vote{vote}.tsv"
        combined.to_csv(path, sep="\t", index=False)
        _write_json(world_out / f"rescue_metadata_vote{vote}.json", metadata)
        evidence_hashes[path.name] = _sha256(path)
        vote_frames.append(combined)

    all_evidence = pd.concat(vote_frames, ignore_index=True)
    consensus = _consensus(all_evidence)
    consensus_path = world_out / "rescue_consensus.tsv"
    consensus.to_csv(consensus_path, sep="\t", index=False)

    current_world_out = current_root / world.name
    current = _current_hierarchy_consensus(current_world_out)
    current_path = world_out / "current_hierarchy_consensus.tsv"
    current.to_csv(current_path, sep="\t", index=False)
    rescue_pairs = _selected_pairs(consensus)
    current_pairs = _selected_pairs(current)
    union_path = world_out / "union_consensus.tsv"
    _pair_frame(rescue_pairs | current_pairs, "current_or_rescue").to_csv(
        union_path, sep="\t", index=False
    )

    frozen = {
        "world": world.name,
        "fit_domain": "discovery_only",
        "scorer_opened": False,
        "frozen_config": asdict(FROZEN_CONFIG),
        "votes": N_VOTES,
        "minimum_votes": MINIMUM_VOTES,
        "current_positive_probability": CURRENT_POSITIVE_PROBABILITY,
        "predeclared_comparisons": ["rescue", "current_hierarchy", "union"],
        "hashes": {
            **evidence_hashes,
            consensus_path.name: _sha256(consensus_path),
            current_path.name: _sha256(current_path),
            union_path.name: _sha256(union_path),
        },
        "source_hashes": {
            "stratified_beta_binomial_edges.py": _sha256(
                Path(__file__).with_name("stratified_beta_binomial_edges.py")
            ),
            "benchmark_beta_binomial_rescue.py": _sha256(Path(__file__)),
        },
        "selected_counts": {
            "rescue": len(rescue_pairs),
            "current_hierarchy": len(current_pairs),
            "union": len(rescue_pairs | current_pairs),
        },
    }
    manifest_path = world_out / "frozen_pretruth_manifest.json"
    _write_json(manifest_path, frozen)
    return {"world": world.name, "manifest_sha256": _sha256(manifest_path), **frozen["selected_counts"]}


def _truth_candidate_pairs(record: dict[str, Any]) -> set[tuple[str, str]]:
    left = [value for value in str(record.get("node_set1", "") or record["node1"]).split(";") if value]
    right = [value for value in str(record.get("node_set2", "") or record["node2"]).split(";") if value]
    return {canonical_edge(a, b) for a in left for b in right if a != b}


def _score_pair_set(
    selected: set[tuple[str, str]],
    truth: pd.DataFrame,
    *,
    method: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records = truth.to_dict(orient="records")
    direct_records = [row for row in records if str(row["truth_type"]) == "within_assay_direct"]
    sentinel_records = [row for row in records if str(row["truth_type"]) == "sentinel_null"]
    within_sentinel = [row for row in sentinel_records if "|" not in str(row.get("mark", ""))]
    labeled = set().union(*(_truth_candidate_pairs(row) for row in records))
    rows: list[dict[str, Any]] = []
    for row in direct_records:
        candidates = _truth_candidate_pairs(row)
        rows.append(
            {
                "method": method,
                "label": row["label"],
                "truth_type": row["truth_type"],
                "present": int(bool(selected & candidates)),
            }
        )
    for row in sentinel_records:
        candidates = _truth_candidate_pairs(row)
        rows.append(
            {
                "method": method,
                "label": row["label"],
                "truth_type": row["truth_type"],
                "present": int(bool(selected & candidates)),
            }
        )
    direct_hits = sum(bool(selected & _truth_candidate_pairs(row)) for row in direct_records)
    sentinel_hits = sum(bool(selected & _truth_candidate_pairs(row)) for row in sentinel_records)
    within_sentinel_hits = sum(
        bool(selected & _truth_candidate_pairs(row)) for row in within_sentinel
    )
    metrics = {
        "method": method,
        "selected_edges": int(len(selected)),
        "direct_edges_recovered": int(direct_hits),
        "direct_edges_total": int(len(direct_records)),
        "direct_recall": float(direct_hits / max(len(direct_records), 1)),
        "sentinel_false_edges": int(sentinel_hits),
        "sentinel_edges_total": int(len(sentinel_records)),
        "sentinel_fpr": float(sentinel_hits / max(len(sentinel_records), 1)),
        "within_assay_sentinel_false_edges": int(within_sentinel_hits),
        "within_assay_sentinel_edges_total": int(len(within_sentinel)),
        "within_assay_sentinel_fpr": float(
            within_sentinel_hits / max(len(within_sentinel), 1)
        ),
        "unlabeled_edges": int(len(selected - labeled)),
    }
    return metrics, pd.DataFrame(rows)


def score_world(world: Path, out_root: Path) -> list[dict[str, Any]]:
    _validate_world(world)
    world_out = out_root / world.name
    manifest_path = world_out / "frozen_pretruth_manifest.json"
    frozen = json.loads(manifest_path.read_text())
    for filename, expected in frozen["hashes"].items():
        observed = _sha256(world_out / filename)
        if observed != expected:
            raise ValueError(f"Frozen evidence hash mismatch: {filename}")
    truth_path = world / "scorer" / "truth_typed_edges.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing scorer truth: {truth_path}")
    truth = pd.read_csv(truth_path, sep="\t", keep_default_na=False)
    frames = {
        "rescue": pd.read_csv(world_out / "rescue_consensus.tsv", sep="\t"),
        "current_hierarchy": pd.read_csv(
            world_out / "current_hierarchy_consensus.tsv", sep="\t"
        ),
        "union": pd.read_csv(world_out / "union_consensus.tsv", sep="\t"),
    }
    metrics: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for method, frame in frames.items():
        pairs = _selected_pairs(frame)
        summary, detail = _score_pair_set(pairs, truth, method=method)
        metrics.append({"world": world.name, **summary})
        details.append(detail)
    pd.concat(details, ignore_index=True).to_csv(
        world_out / "postfreeze_pair_scores.tsv", sep="\t", index=False
    )
    _write_json(world_out / "postfreeze_metrics.json", metrics)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    parser.add_argument("--world", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--current-root",
        type=Path,
        default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/train_hard_sparse8_v4"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "fit":
        summaries = [
            fit_world(world, args.out_dir, args.current_root) for world in args.world
        ]
        _write_json(
            args.out_dir / "frozen_train_manifest.json",
            {
                "phase": "fit_complete_scorer_unopened_by_command",
                "worlds": summaries,
                "frozen_config": asdict(FROZEN_CONFIG),
            },
        )
    else:
        summaries = [row for world in args.world for row in score_world(world, args.out_dir)]
        pd.DataFrame(summaries).to_csv(
            args.out_dir / "postfreeze_comparison.tsv", sep="\t", index=False
        )
        _write_json(args.out_dir / "postfreeze_comparison.json", summaries)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
