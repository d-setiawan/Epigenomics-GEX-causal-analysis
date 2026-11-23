#!/usr/bin/env python3
"""Replay the production observed-collider selector on old TRAIN 33001--33024.

All observed worlds and production decisions are frozen in Phase A.  Only
after that manifest is written does Phase B open old-TRAIN direct truth.  This
is development-history compatibility evidence, not formal certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from diagnose_formal_train_direct_failures_v4 import (
    _candidate_directory,
    _direct_metric_row,
    _directed_edges_from_frame,
    _label_wedges,
    _read_tsv,
    _replay_production_observed_refinement,
    _required,
    _sha256,
    _stack,
    _write_json,
)
from diagnose_rescue_only_completion_compatibility_v4 import (
    ARTIFACT_ROOT,
    SEEDS,
    _load_old_train_world,
)


WORLD_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_worlds_v1"
)


def _load_full_phase_a_world(seed: int) -> dict[str, Any]:
    world = _load_old_train_world(seed)
    world_artifact = ARTIFACT_ROOT / f"world_seed{seed}"
    candidate = _candidate_directory(world_artifact)
    current_path = _required(
        candidate / "collider_motif" / "direct_candidates.tsv"
    )
    raw_frames: list[pd.DataFrame] = []
    raw_paths: list[Path] = []
    views = sorted(
        path
        for path in world_artifact.glob("cells*_seed*")
        if path.is_dir() and not path.is_symlink()
    )
    if len(views) != 6:
        raise ValueError(f"Expected six old-TRAIN views for seed {seed}")
    for view in views:
        path = _required(view / "raw_pair_evidence.tsv")
        frame = _read_tsv(path)
        frame.insert(0, "view_id", view.name)
        raw_frames.append(frame)
        raw_paths.append(path)
    world["raw_pairs"] = pd.concat(
        raw_frames, ignore_index=True, sort=False
    )
    world["current_direct"] = _read_tsv(current_path)
    world["input_hashes"].update(
        {
            str(path.relative_to(ARTIFACT_ROOT)): _sha256(path)
            for path in [current_path, *raw_paths]
        }
    )
    return world


def _truth_after_barrier() -> tuple[pd.DataFrame, dict[int, set[tuple[str, str]]]]:
    frames: list[pd.DataFrame] = []
    truth_sets: dict[int, set[tuple[str, str]]] = {}
    for seed in SEEDS:
        path = _required(
            WORLD_ROOT
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        truth = _read_tsv(path)
        direct = truth.loc[
            truth["truth_type"].astype(str).eq("within_assay_direct")
        ].copy()
        direct.insert(0, "seed", seed)
        direct["source_assay"] = "chrom_" + direct["mark"].astype(str)
        frames.append(direct)
        truth_sets[seed] = {
            (str(row.parent), str(row.child))
            for row in direct.itertuples(index=False)
        }
    return pd.concat(frames, ignore_index=True, sort=False), truth_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: load all old worlds before replaying any production decision.
    held = [_load_full_phase_a_world(seed) for seed in SEEDS]
    if tuple(int(world["seed"]) for world in held) != SEEDS:
        raise RuntimeError("Old-TRAIN production-replay barrier is incomplete")
    for world in held:
        world.update(_replay_production_observed_refinement(world))

    phase_a_keys = (
        "current_direct",
        "production_direct",
        "production_unresolved",
        "production_components",
        "production_wedges",
        "production_repair_decisions",
        "production_wedge_view_evidence",
        "production_wedge_evidence",
        "production_completion_view_evidence",
        "production_completion_evidence",
    )
    phase_a_paths: list[Path] = []
    phase_a_frames: dict[str, pd.DataFrame] = {}
    for key in phase_a_keys:
        frame = _stack(held, key)
        path = args.out_dir / f"phase_a_{key}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        phase_a_paths.append(path)
        phase_a_frames[key] = frame
    input_hashes_path = args.out_dir / "phase_a_input_hashes.json"
    _write_json(
        input_hashes_path,
        {f"world_seed{world['seed']}": world["input_hashes"] for world in held},
    )
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("diagnose_formal_train_direct_failures_v4.py"),
        Path(__file__).with_name(
            "diagnose_rescue_only_completion_compatibility_v4.py"
        ),
        Path(__file__).with_name("observed_collider_structure.py"),
        Path(__file__).with_name("benchmark_ca_boss_v4.py"),
        Path(__file__).with_name("consensus_ca_boss_v4.py"),
        Path(__file__).with_name("structure_aware_collider_motif.py"),
        Path(__file__).with_name("structure_aware_collider_rescue.py"),
    ]
    manifest_path = args.out_dir / "phase_a_manifest.json"
    _write_json(
        manifest_path,
        {
            "block": "old_train_33001_33024_production_direct_replay",
            "seeds": list(SEEDS),
            "n_worlds": len(held),
            "all_worlds_loaded_before_any_production_replay": True,
            "all_production_replays_frozen_before_truth_open": True,
            "dev_locked_or_seed_ge_36001_opened": False,
            "phase_a_output_sha256": {
                path.name: _sha256(path)
                for path in [*phase_a_paths, input_hashes_path]
            },
            "source_sha256": {
                path.name: _sha256(path) for path in source_paths
            },
        },
    )

    # Phase B: direct truth opens only after the Phase-A manifest exists.
    truth_direct, truth_sets = _truth_after_barrier()
    metric_rows: list[dict[str, Any]] = []
    for world in held:
        seed = int(world["seed"])
        truth = truth_sets[seed]
        current = _directed_edges_from_frame(world["current_direct"])
        production = _directed_edges_from_frame(world["production_direct"])
        current_row = _direct_metric_row(seed, current, truth, "current")
        production_row = _direct_metric_row(
            seed, production, truth, "production"
        )
        metric_rows.append(
            {
                **current_row,
                **{
                    key: value
                    for key, value in production_row.items()
                    if key != "seed"
                },
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("seed").reset_index(drop=True)
    labeled_decisions = _label_wedges(
        phase_a_frames["production_repair_decisions"], truth_direct
    )
    metrics_path = args.out_dir / "phase_b_direct_metrics.tsv"
    decisions_path = args.out_dir / "phase_b_labeled_production_decisions.tsv"
    truth_path = args.out_dir / "phase_b_truth_direct.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    labeled_decisions.to_csv(decisions_path, sep="\t", index=False)
    truth_direct.to_csv(truth_path, sep="\t", index=False)
    summary = {
        "block": "old_train_33001_33024_development_history",
        "phase_a_manifest_sha256": _sha256(manifest_path),
        "truth_opened_only_after_phase_a_manifest": True,
        "n_worlds": len(held),
        "current_reliability_compatible_worlds": int(
            metrics["current_reliability_compatible"].sum()
        ),
        "production_reliability_compatible_worlds": int(
            metrics["production_reliability_compatible"].sum()
        ),
        "current_exact_directed_worlds": int(
            metrics["current_exact_directed"].sum()
        ),
        "production_exact_directed_worlds": int(
            metrics["production_exact_directed"].sum()
        ),
        "production_total_false_predicted": int(
            metrics["production_n_false_predicted"].sum()
        ),
        "production_total_strict_reverse": int(
            metrics["production_n_strict_reverse"].sum()
        ),
        "production_decision_action_counts": {
            str(action): int(count)
            for action, count in phase_a_frames[
                "production_repair_decisions"
            ]["action"].value_counts().items()
        },
        "diagnostic_status": "posthoc_old_train_not_certification",
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        args.out_dir / "phase_b_manifest.json",
        {
            "phase_a_manifest_sha256": _sha256(manifest_path),
            "phase_b_output_sha256": {
                path.name: _sha256(path)
                for path in (
                    metrics_path,
                    decisions_path,
                    truth_path,
                    summary_path,
                )
            },
            "truth_opened_only_after_phase_a_manifest": True,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
