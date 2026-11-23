#!/usr/bin/env python3
"""Freeze observed-only direct-structure evidence for retired TRAIN380.

This script is Phase A only.  It imports the existing direct-diagnostic
enumerators, points them at the already-completed formal TRAIN380 block, and
writes every observed table plus a prespecified family of candidate guard
rules.  It never resolves or reads ``pipelines/simulation/unpaired_multimodal/worlds/formal_worlds`` and therefore
cannot access scorer roles or truth.

The output is deliberately write-once.  Phase B must run as a separate
process after ``phase_a_manifest.json`` has been written and hashed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import diagnose_formal_train_direct_failures_v4 as diagnostic


SEEDS = tuple(range(38001, 38025))
RUN_ROOT = Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_38001_38024")


PHASE_A_KEYS = (
    "current_direct",
    "unresolved",
    "components",
    "wedges",
    "aggregated_votes",
    "observed_decisions",
    "observed_wedge_views",
    "observed_wedges",
    "observed_completion_views",
    "observed_completions",
    "subspace_features",
    "layered_features",
    "loading_feature_consensus",
    "structure_anchor_pairs",
    "structure_anchor_decisions",
    "raw_pairs",
    "cell_features",
    "axis_features",
    "cell_pairs",
    "axis_pairs",
    "hierarchical_pairs",
    "nested_edges",
    "beta_votes",
    "completion_repair_candidates",
    "completion_repair_views",
    "fixed_repair_evidence",
    "fixed_repair_decisions",
    "optional_repair_evidence",
    "optional_repair_decisions",
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _prespecified_rule_registry() -> dict[str, Any]:
    """Declare observed-only hypotheses before any TRAIN380 labels are read."""

    return {
        "schema_version": "ca_boss_v4.train380_direct_rule_registry.phase_a.v1",
        "status": "prespecified_before_train380_truth_access",
        "estimand": "within_assay two-parent collider structure",
        "selection_policy": (
            "Rules are diagnostic candidates only; no rule may be called "
            "validated on TRAIN380. Any selected rule requires compatibility "
            "checks on retired TRAIN330/360/370 and a fresh disjoint TRAIN."
        ),
        "candidate_guard_families": [
            {
                "family": "production_replay",
                "operation": "retain every production observed-structure decision",
                "thresholds": {},
            },
            {
                "family": "completion_state_floor_guard",
                "applies_to_action": "complete_dual_supported_unshielded_anchor",
                "operation": (
                    "retain completion only when its selected candidate has "
                    "outer_state_support_floor strictly above threshold"
                ),
                "thresholds": [0.0, 0.0025, 0.005, 0.01, 0.02],
            },
            {
                "family": "completion_candidate_state_gate",
                "applies_to_action": "complete_dual_supported_unshielded_anchor",
                "operation": (
                    "retain completion only when the candidate new parent is "
                    "formally selected in the held-out shared-state layer"
                ),
                "thresholds": {"selection_status": "selected"},
            },
            {
                "family": "completion_log_odds_guard",
                "applies_to_action": "complete_dual_supported_unshielded_anchor",
                "operation": (
                    "retain completion only when selected mean pooled log odds "
                    "and the selected-minus-runner margin exceed thresholds"
                ),
                "minimum_selected_log_odds": [1.0, 1.25, 1.5, 1.75],
                "minimum_runner_margin": [0.0, 0.1, 0.25, 0.5],
            },
            {
                "family": "repair_action_abstention",
                "applies_to_action": [
                    "complete_dual_supported_unshielded_anchor",
                    "resolve_competing_multicenter_path",
                    "reselect_overcomplete_same_center_wedge",
                ],
                "operation": (
                    "abstain from one production repair family and retain its "
                    "pre-repair component output"
                ),
            },
            {
                "family": "component_ambiguity_abstention",
                "operation": (
                    "retain a production repair only if exactly one candidate "
                    "wins the applicable six-view comparison and the runner "
                    "does not tie on the primary score"
                ),
                "minimum_winner_views": [4, 5, 6],
            },
        ],
        "privileged_fields_used": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_root = diagnostic._required(RUN_ROOT / "global_pretruth_freeze.json").parent

    # Override only the retired block allow-list used by the imported loader.
    diagnostic.SEEDS = SEEDS
    held = [diagnostic._load_phase_a_world(seed, run_root) for seed in SEEDS]
    if tuple(int(world["seed"]) for world in held) != SEEDS:
        raise RuntimeError("Global TRAIN380 Phase-A barrier is incomplete")

    for world in held:
        repair_summary, repair_views = diagnostic._enumerate_completion_repairs(world)
        world["completion_repair_candidates"] = repair_summary
        world["completion_repair_views"] = repair_views
        fixed_evidence, fixed_decisions = diagnostic._fixed_repair_proposals(world)
        world["fixed_repair_evidence"] = fixed_evidence
        world["fixed_repair_decisions"] = fixed_decisions
        optional_evidence, optional_decisions = (
            diagnostic._optional_rescue_only_completion_proposals(world)
        )
        world["optional_repair_evidence"] = optional_evidence
        world["optional_repair_decisions"] = optional_decisions
        world.update(diagnostic._replay_production_observed_refinement(world))

    output_paths: list[Path] = []
    for key in PHASE_A_KEYS:
        frame = diagnostic._stack(held, key)
        path = args.out_dir / f"phase_a_{key}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        output_paths.append(path)

    input_hash_path = args.out_dir / "phase_a_input_hashes.json"
    _write_json(
        input_hash_path,
        {
            f"world_seed{world['seed']}": world["input_hashes"]
            for world in held
        },
    )
    output_paths.append(input_hash_path)

    rule_path = args.out_dir / "phase_a_rule_registry.json"
    _write_json(rule_path, _prespecified_rule_registry())
    output_paths.append(rule_path)

    source_dependencies = [
        Path(__file__),
        Path(diagnostic.__file__),
        Path(diagnostic.__file__).with_name("observed_collider_structure.py"),
        Path(diagnostic.__file__).with_name("benchmark_ca_boss_v4.py"),
        Path(diagnostic.__file__).with_name("consensus_ca_boss_v4.py"),
        Path(diagnostic.__file__).with_name("structure_aware_collider_motif.py"),
        Path(diagnostic.__file__).with_name("structure_aware_collider_rescue.py"),
    ]
    manifest = {
        "schema_version": "ca_boss_v4.train380_direct_evidence.phase_a.v1",
        "block": "retired_formal_train_38001_38024",
        "seeds": list(SEEDS),
        "n_worlds": len(held),
        "all_observed_candidates_loaded_before_truth_access": True,
        "formal_world_root_or_scorer_opened": False,
        "dev_or_locked_opened": False,
        "phase_a_output_sha256": {
            path.name: diagnostic._sha256(path) for path in output_paths
        },
        "source_sha256": {
            str(path): diagnostic._sha256(path) for path in source_dependencies
        },
    }
    diagnostic._write_json(args.out_dir / "phase_a_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
