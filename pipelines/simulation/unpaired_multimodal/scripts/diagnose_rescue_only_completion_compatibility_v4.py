#!/usr/bin/env python3
"""Audit the optional rescue-only completion rule on old TRAIN 33001--33024.

This compatibility diagnostic is observed-only.  It loads every required
candidate table for all 24 old TRAIN worlds before selecting any completion.
It never opens truth, role, DEV, LOCKED, or proposed formal-world files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from diagnose_formal_train_direct_failures_v4 import (
    _candidate_directory,
    _optional_rescue_only_completion_proposals,
    _read_tsv,
    _required,
    _sha256,
    _stack,
    _write_json,
)


SEEDS = tuple(range(33001, 33025))
ARTIFACT_ROOT = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_v1/artifacts"
)


def _load_old_train_world(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("Compatibility audit is restricted to TRAIN 33001-33024")
    world_artifact = _required(
        ARTIFACT_ROOT / f"world_seed{seed}" / "pretruth_consensus_manifest.json"
    ).parent
    candidate = _candidate_directory(world_artifact)
    motif_root = candidate / "collider_motif"
    paths = {
        "components": _required(motif_root / "component_diagnostics.tsv"),
        "unresolved": _required(motif_root / "demoted_or_unresolved.tsv"),
        "aggregated_votes": _required(
            motif_root / "aggregated_vote_evidence.tsv"
        ),
        "subspace_features": _required(
            world_artifact
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        ),
    }
    frames = {key: _read_tsv(path) for key, path in paths.items()}
    input_paths = list(paths.values())

    views = sorted(
        path
        for path in world_artifact.glob("cells*_seed*")
        if path.is_dir() and not path.is_symlink()
    )
    if len(views) != 6:
        raise ValueError(f"Expected six evidence views for seed {seed}")
    cell_features: list[pd.DataFrame] = []
    cell_pairs: list[pd.DataFrame] = []
    nested_edges: list[pd.DataFrame] = []
    for index, view in enumerate(views):
        feature_path = _required(view / "cell_loading_feature_models.tsv")
        pair_path = _required(view / "cell_loading_pair_evidence.tsv")
        nested_path = _required(candidate / f"nested_{index}_edges.tsv")
        feature_frame = _read_tsv(feature_path)
        pair_frame = _read_tsv(pair_path)
        nested_frame = _read_tsv(nested_path)
        for frame in (feature_frame, pair_frame, nested_frame):
            frame.insert(0, "view_id", view.name)
        cell_features.append(feature_frame)
        cell_pairs.append(pair_frame)
        nested_edges.append(nested_frame)
        input_paths.extend((feature_path, pair_path, nested_path))

    vote_paths = [
        _required(
            world_artifact
            / "collider_motif_evidence"
            / f"beta_binomial_vote{vote}.tsv"
        )
        for vote in range(6)
    ]
    input_paths.extend(vote_paths)
    return {
        "seed": seed,
        **frames,
        "cell_features": pd.concat(
            cell_features, ignore_index=True, sort=False
        ),
        "cell_pairs": pd.concat(cell_pairs, ignore_index=True, sort=False),
        "nested_edges": pd.concat(
            nested_edges, ignore_index=True, sort=False
        ),
        "beta_votes": pd.concat(
            [_read_tsv(path) for path in vote_paths],
            ignore_index=True,
            sort=False,
        ),
        "input_hashes": {
            str(path.relative_to(ARTIFACT_ROOT)): _sha256(path)
            for path in sorted(input_paths)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Global Phase-A barrier: all observed worlds are resident before selection.
    held = [_load_old_train_world(seed) for seed in SEEDS]
    if tuple(int(world["seed"]) for world in held) != SEEDS:
        raise RuntimeError("Old-TRAIN global candidate barrier is incomplete")
    for world in held:
        evidence, decisions = _optional_rescue_only_completion_proposals(world)
        world["optional_repair_evidence"] = evidence
        world["optional_repair_decisions"] = decisions

    evidence = _stack(held, "optional_repair_evidence")
    decisions = _stack(held, "optional_repair_decisions")
    evidence_path = args.out_dir / "phase_a_optional_repair_evidence.tsv"
    decisions_path = args.out_dir / "phase_a_optional_repair_decisions.tsv"
    hashes_path = args.out_dir / "phase_a_input_hashes.json"
    evidence.to_csv(evidence_path, sep="\t", index=False)
    decisions.to_csv(decisions_path, sep="\t", index=False)
    _write_json(
        hashes_path,
        {f"world_seed{world['seed']}": world["input_hashes"] for world in held},
    )

    component_summary = (
        evidence.groupby(
            ["seed", "source_assay", "component_id"], sort=True
        )
        .agg(
            rescue_only_component=("rescue_only_component", "max"),
            n_gate_passing_candidates=(
                "passes_all_candidate_gates",
                "sum",
            ),
        )
        .reset_index()
    )
    component_summary_path = args.out_dir / "phase_a_component_summary.tsv"
    component_summary.to_csv(component_summary_path, sep="\t", index=False)
    rescue_only = component_summary.loc[
        component_summary["rescue_only_component"].astype(bool)
    ]
    summary_path = args.out_dir / "summary.json"
    summary = {
        "block": "old_train_33001_33024_observed_only_compatibility",
        "n_worlds": len(held),
        "all_worlds_loaded_before_any_candidate_selection": True,
        "truth_or_roles_opened": False,
        "dev_locked_or_seed_ge_37001_opened": False,
        "n_rescue_only_components": len(rescue_only),
        "rescue_only_component_candidate_counts": [
            {
                "seed": int(row.seed),
                "source_assay": str(row.source_assay),
                "component_id": str(row.component_id),
                "n_gate_passing_candidates": int(
                    row.n_gate_passing_candidates
                ),
            }
            for row in rescue_only.itertuples(index=False)
        ],
        "n_optional_repair_decisions": len(decisions),
        "fixed_rule": {
            "anchor_selected_vote_fraction_minimum": 5 / 6,
            "anchor_mean_empirical_p_positive_maximum": 0.020,
            "anchor_mean_pooled_log_odds_minimum": 1.0,
            "anchor_base_view_count_maximum_exclusive": 4,
            "new_parent_formal_state_selection_status": "selected",
            "center_new_loading_empirical_null_p_maximum": 0.125,
            "required_loading_view_support": "6/6",
            "direct_vs_outer_lor_contrast_formula": (
                "pooled_log_odds(center,new_parent) - "
                "pooled_log_odds(anchor_outer,new_parent)"
            ),
            "required_positive_contrast_view_support": "6/6",
            "required_component_candidate_count": 1,
        },
        "phase_a_output_sha256": {
            path.name: _sha256(path)
            for path in (
                evidence_path,
                decisions_path,
                hashes_path,
                component_summary_path,
            )
        },
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
