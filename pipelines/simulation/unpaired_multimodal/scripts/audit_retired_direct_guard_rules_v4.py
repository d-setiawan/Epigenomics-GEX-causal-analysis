#!/usr/bin/env python3
"""Audit observed-only direct-repair guard rules on retired TRAIN blocks.

TRAIN330/360/370/380 are development history here.  TRAIN380 may be read only
after its independent Phase-A evidence manifest exists and verifies.  The
script replays the frozen production selector, filters only its observed-only
decisions, and reports both decision false positives and resulting graph
metrics.  It never accepts DEV or LOCKED paths.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import diagnose_formal_train_direct_failures_v4 as diagnostic
from benchmark_ca_boss_v4 import _direct_consensus_for_motif
from consensus_ca_boss_v4 import consensus_typed_graph
from diagnose_rescue_only_completion_compatibility_v4 import SEEDS as OLD_SEEDS
from observed_collider_structure import (
    ObservedColliderStructureConfig,
    apply_observed_collider_refinements,
    select_observed_collider_refinements,
)
from replay_production_direct_repairs_old_train_v4 import _load_full_phase_a_world
from structure_aware_collider_motif import select_unique_collider_motifs


BLOCKS = {
    "TRAIN330": {
        "seeds": tuple(int(seed) for seed in OLD_SEEDS),
        "world_root": Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_worlds_v1"
        ),
    },
    "TRAIN360": {
        "seeds": tuple(range(36001, 36025)),
        "run_root": Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_36001_36024"),
        "world_root": Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_36001_36024"),
    },
    "TRAIN370": {
        "seeds": tuple(range(37001, 37025)),
        "run_root": Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_37001_37024"),
        "world_root": Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_37001_37024"),
    },
    "TRAIN380": {
        "seeds": tuple(range(38001, 38025)),
        "run_root": Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_38001_38024"),
        "world_root": Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_38001_38024"),
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _verify_train380_barrier(phase_a: Path) -> str:
    path = phase_a / "phase_a_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("seeds") != list(BLOCKS["TRAIN380"]["seeds"]):
        raise ValueError("TRAIN380 Phase A has the wrong seed registry")
    if not manifest.get("all_observed_candidates_loaded_before_truth_access"):
        raise ValueError("TRAIN380 Phase-A barrier is absent")
    if manifest.get("formal_world_root_or_scorer_opened") is not False:
        raise ValueError("TRAIN380 Phase A reports scorer access")
    for name, expected in manifest["phase_a_output_sha256"].items():
        if diagnostic._sha256(phase_a / name) != expected:
            raise ValueError(f"TRAIN380 Phase-A drift: {name}")
    return diagnostic._sha256(path)


def _load_block_worlds(block: str) -> list[dict[str, Any]]:
    spec = BLOCKS[block]
    seeds = tuple(spec["seeds"])
    if block == "TRAIN330":
        worlds = [_load_full_phase_a_world(seed) for seed in seeds]
    else:
        diagnostic.SEEDS = seeds
        worlds = [
            diagnostic._load_phase_a_world(seed, Path(spec["run_root"]))
            for seed in seeds
        ]
    if tuple(int(world["seed"]) for world in worlds) != seeds:
        raise RuntimeError(f"Incomplete observed block load: {block}")
    return worlds


def _selector_inputs(world: dict[str, Any]) -> tuple[Any, Any]:
    nested_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["nested_edges"].groupby("view_id", sort=True)
    }
    preliminary = consensus_typed_graph(list(nested_views.values()), minimum_count=4)
    motif = select_unique_collider_motifs(
        _direct_consensus_for_motif(preliminary), world["beta_votes"]
    )
    residual_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["raw_pairs"].groupby("view_id", sort=True)
    }
    feature_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["cell_features"].groupby("view_id", sort=True)
    }
    loading_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["cell_pairs"].groupby("view_id", sort=True)
    }
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        world["beta_votes"],
        nested_views,
        residual_views,
        feature_views,
        ObservedColliderStructureConfig(),
        loading_pair_views=loading_views,
        state_membership=world["subspace_features"],
    )
    return motif, refinement


def _same_outer_set(row: pd.Series, decision: pd.Series, columns: tuple[str, str]) -> bool:
    if any(column not in row.index or pd.isna(row[column]) for column in columns):
        return False
    return frozenset((str(row[columns[0]]), str(row[columns[1]]))) == frozenset(
        (str(decision["outer1"]), str(decision["outer2"]))
    )


def _decision_features(
    decision: pd.Series,
    refinement: Any,
    state_membership: pd.DataFrame,
) -> dict[str, Any]:
    component_id = str(decision["component_id"])
    center = str(decision["center"])
    action = str(decision["action"])
    values: dict[str, Any] = {
        "winner_view_count": int(decision["winner_view_count"]),
        "n_views": int(decision["n_views"]),
        "median_collider_separation_score": np.nan,
        "median_collider_opening_delta": np.nan,
        "median_outer_state_support_floor": np.nan,
        "pairwise_wedge_score": np.nan,
        "anchor_base_view_count": np.nan,
        "selected_arm_mean_pooled_log_odds": np.nan,
        "selected_arm_outer_state_support_floor": np.nan,
        "selected_arm_runner_log_odds_margin": np.nan,
        "candidate_new_parent_formal_state_selected": np.nan,
        "maximum_loading_empirical_null_p": np.nan,
        "minimum_absolute_loading_cosine": np.nan,
        "minimum_direct_vs_outer_lor_contrast": np.nan,
    }

    wedges = refinement.wedge_evidence
    if not wedges.empty:
        candidates = wedges.loc[
            wedges["component_id"].astype(str).eq(component_id)
            & wedges["center"].astype(str).eq(center)
        ]
        for _, row in candidates.iterrows():
            if _same_outer_set(row, decision, ("outer1", "outer2")):
                for column in (
                    "median_collider_separation_score",
                    "median_collider_opening_delta",
                    "median_outer_state_support_floor",
                    "pairwise_wedge_score",
                ):
                    if column in row.index:
                        values[column] = float(row[column])
                break

    completion = refinement.completion_evidence
    selected_row: pd.Series | None = None
    if not completion.empty:
        candidates = completion.loc[
            completion["component_id"].astype(str).eq(component_id)
        ]
        for _, row in candidates.iterrows():
            candidate_center = str(
                row.get("selected_center")
                if pd.notna(row.get("selected_center"))
                else row.get("center", "")
            )
            if candidate_center != center:
                continue
            if action == "complete_unique_rescue_only_state_loading_contrast":
                if _same_outer_set(
                    row, decision, ("anchor_outer", "candidate_new_parent")
                ) and bool(row.get("passes_all_candidate_gates", False)):
                    selected_row = row
                    break
            elif _same_outer_set(row, decision, ("anchor_outer", "new_outer")):
                if bool(row.get("stable_view_winner", False)):
                    selected_row = row
                    break
            elif _same_outer_set(
                row, decision, ("anchor_outer", "selected_new_parent")
            ):
                selected_row = row
                break
    if selected_row is not None:
        mappings = {
            "anchor_base_view_count": "anchor_base_view_count",
            "selected_arm_mean_pooled_log_odds": "mean_pooled_log_odds",
            "selected_arm_outer_state_support_floor": "outer_state_support_floor",
            "maximum_loading_empirical_null_p": "maximum_loading_empirical_null_p",
            "minimum_absolute_loading_cosine": "minimum_absolute_loading_cosine",
            "minimum_direct_vs_outer_lor_contrast": (
                "minimum_direct_vs_outer_lor_contrast"
            ),
        }
        for target, source in mappings.items():
            if source in selected_row.index and pd.notna(selected_row[source]):
                values[target] = float(selected_row[source])
        new_parent = selected_row.get("candidate_new_parent")
        if pd.isna(new_parent):
            new_parent = selected_row.get("new_outer")
        if pd.notna(new_parent):
            membership = state_membership.loc[
                state_membership["feature"].astype(str).eq(str(new_parent))
            ]
            if len(membership) == 1:
                values["candidate_new_parent_formal_state_selected"] = bool(
                    str(membership.iloc[0]["selection_status"]) == "selected"
                )
        if (
            "mean_pooled_log_odds" in candidates.columns
            and pd.notna(values["selected_arm_mean_pooled_log_odds"])
        ):
            peers = candidates.loc[
                candidates.index != selected_row.name, "mean_pooled_log_odds"
            ].dropna()
            if len(peers):
                values["selected_arm_runner_log_odds_margin"] = float(
                    values["selected_arm_mean_pooled_log_odds"] - peers.max()
                )
    return values


def _rules() -> dict[str, Callable[[pd.Series], bool]]:
    completion = "complete_dual_supported_unshielded_anchor"
    rescue = "complete_unique_rescue_only_state_loading_contrast"
    multicenter = "resolve_competing_multicenter_path"

    rules: dict[str, Callable[[pd.Series], bool]] = {
        "production": lambda row: True,
        "abstain_rescue_only": lambda row: str(row.action) != rescue,
        "rescue_anchor_base_ge_1": lambda row: (
            str(row.action) != rescue
            or float(row.anchor_base_view_count) >= 1
        ),
        "rescue_anchor_base_ge_2": lambda row: (
            str(row.action) != rescue
            or float(row.anchor_base_view_count) >= 2
        ),
        "rescue_anchor_base_ge_3": lambda row: (
            str(row.action) != rescue
            or float(row.anchor_base_view_count) >= 3
        ),
        "multicenter_positive_separation": lambda row: (
            str(row.action) != multicenter
            or float(row.median_collider_separation_score) > 0
        ),
        "multicenter_six_of_six": lambda row: (
            str(row.action) != multicenter or int(row.winner_view_count) == 6
        ),
        "multicenter_positive_separation_and_six": lambda row: (
            str(row.action) != multicenter
            or (
                float(row.median_collider_separation_score) > 0
                and int(row.winner_view_count) == 6
            )
        ),
    }
    for threshold in (0.0, 0.0025, 0.005, 0.01, 0.02):
        rules[f"completion_state_floor_gt_{threshold:g}"] = (
            lambda row, threshold=threshold: (
                str(row.action) != completion
                or float(row.selected_arm_outer_state_support_floor) > threshold
            )
        )
    for threshold in (0.5, 1.0, 1.25, 1.5, 1.75):
        rules[f"completion_selected_lor_ge_{threshold:g}"] = (
            lambda row, threshold=threshold: (
                str(row.action) != completion
                or float(row.selected_arm_mean_pooled_log_odds) >= threshold
            )
        )
    rules["completion_candidate_formal_state_selected"] = lambda row: (
        str(row.action) != completion
        or bool(row.candidate_new_parent_formal_state_selected)
    )
    rules["composite_guard_v1"] = lambda row: bool(
        (
            str(row.action) != rescue
            or float(row.anchor_base_view_count) >= 1
        )
        and (
            str(row.action) != completion
            or float(row.selected_arm_outer_state_support_floor) > 0
        )
        and (
            str(row.action) != multicenter
            or float(row.median_collider_separation_score) > 0
        )
    )
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train380-phase-a", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train380_manifest_sha = _verify_train380_barrier(args.train380_phase_a)

    rules = _rules()
    decision_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for block, spec in BLOCKS.items():
        worlds = _load_block_worlds(block)
        truth_direct, truth_sets = diagnostic._load_direct_truth_after_barrier(
            worlds, Path(spec["world_root"])
        )
        truth_map = diagnostic._truth_wedge_map(truth_direct)
        for world in worlds:
            seed = int(world["seed"])
            motif, refinement = _selector_inputs(world)
            decisions = refinement.decisions.copy()
            enriched: list[dict[str, Any]] = []
            for _, decision in decisions.iterrows():
                assay = str(decision["source_assay"])
                true_center, true_parents = truth_map[(seed, assay)]
                exact = bool(
                    str(decision["center"]) == true_center
                    and frozenset(
                        (str(decision["outer1"]), str(decision["outer2"]))
                    )
                    == true_parents
                )
                record = {
                    "block": block,
                    "seed": seed,
                    **decision.to_dict(),
                    **_decision_features(
                        decision, refinement, world["subspace_features"]
                    ),
                    "decision_exact": exact,
                }
                enriched.append(record)
                decision_rows.append(record)
            enriched_frame = pd.DataFrame(enriched)
            truth = truth_sets[seed]
            for rule_name, predicate in rules.items():
                if decisions.empty:
                    kept = decisions.copy()
                else:
                    keep_mask = [
                        bool(predicate(row))
                        for _, row in enriched_frame.iterrows()
                    ]
                    kept = decisions.loc[keep_mask].copy().reset_index(drop=True)
                filtered = replace(refinement, decisions=kept)
                graph = apply_observed_collider_refinements(motif, filtered)
                predicted = diagnostic._directed_edges_from_frame(
                    graph.direct_candidates
                )
                metric = diagnostic._direct_metric_row(
                    seed, predicted, truth, "graph"
                )
                graph_rows.append(
                    {
                        "block": block,
                        "rule": rule_name,
                        "n_production_decisions": len(decisions),
                        "n_retained_decisions": len(kept),
                        **metric,
                    }
                )
        if block != "TRAIN330":
            freeze_path = Path(spec["run_root"]) / "global_pretruth_freeze.json"
            input_hashes[str(freeze_path)] = diagnostic._sha256(freeze_path)

    decisions = pd.DataFrame(decision_rows)
    graphs = pd.DataFrame(graph_rows)
    decision_audit_rows: list[dict[str, Any]] = []
    for scope, scoped in [
        ("ALL_RETIRED", decisions),
        *[(block, decisions.loc[decisions["block"].eq(block)]) for block in BLOCKS],
    ]:
        for rule_name, predicate in rules.items():
            retained = scoped.loc[
                [bool(predicate(row)) for _, row in scoped.iterrows()]
            ] if not scoped.empty else scoped
            removed = scoped.loc[~scoped.index.isin(retained.index)]
            decision_audit_rows.append(
                {
                    "scope": scope,
                    "rule": rule_name,
                    "n_decisions": len(scoped),
                    "n_retained": len(retained),
                    "retained_exact": int(retained["decision_exact"].sum()),
                    "retained_false": int((~retained["decision_exact"].astype(bool)).sum()),
                    "removed_exact": int(removed["decision_exact"].sum()),
                    "removed_false": int((~removed["decision_exact"].astype(bool)).sum()),
                    "retained_precision": (
                        float(retained["decision_exact"].mean())
                        if len(retained)
                        else np.nan
                    ),
                }
            )
    decision_audit = pd.DataFrame(decision_audit_rows)

    graph_audit = (
        graphs.groupby(["block", "rule"], sort=True)
        .agg(
            n_worlds=("seed", "size"),
            reliable_worlds=("graph_reliability_compatible", "sum"),
            exact_worlds=("graph_exact_directed", "sum"),
            total_false_predicted=("graph_n_false_predicted", "sum"),
            total_strict_reverse=("graph_n_strict_reverse", "sum"),
            total_true_predicted=("graph_n_true_predicted", "sum"),
            total_correctly_oriented=("graph_n_correctly_oriented", "sum"),
        )
        .reset_index()
    )
    all_graph = (
        graphs.groupby("rule", sort=True)
        .agg(
            n_worlds=("seed", "size"),
            reliable_worlds=("graph_reliability_compatible", "sum"),
            exact_worlds=("graph_exact_directed", "sum"),
            total_false_predicted=("graph_n_false_predicted", "sum"),
            total_strict_reverse=("graph_n_strict_reverse", "sum"),
            total_true_predicted=("graph_n_true_predicted", "sum"),
            total_correctly_oriented=("graph_n_correctly_oriented", "sum"),
        )
        .reset_index()
    )
    all_graph.insert(0, "block", "ALL_RETIRED")
    graph_audit = pd.concat([all_graph, graph_audit], ignore_index=True)

    outputs = {
        "decision_features.tsv": decisions,
        "world_rule_metrics.tsv": graphs,
        "decision_guard_audit.tsv": decision_audit,
        "graph_guard_audit.tsv": graph_audit,
    }
    output_paths: list[Path] = []
    for name, frame in outputs.items():
        path = args.out_dir / name
        frame.to_csv(path, sep="\t", index=False)
        output_paths.append(path)

    summary = {
        "schema_version": "ca_boss_v4.retired_direct_guard_audit.v1",
        "blocks": list(BLOCKS),
        "n_worlds": int(graphs.loc[graphs["rule"].eq("production")].shape[0]),
        "train380_phase_a_manifest_sha256": train380_manifest_sha,
        "train380_phase_a_verified_before_train380_truth_access": True,
        "all_blocks_are_retired_development_history": True,
        "dev_or_locked_opened": False,
        "rules": list(rules),
        "input_global_pretruth_freeze_sha256": input_hashes,
        "status": "posthoc_rule_audit_not_certification",
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    output_paths.append(summary_path)
    _write_json(
        args.out_dir / "manifest.json",
        {
            "summary_sha256": diagnostic._sha256(summary_path),
            "output_sha256": {
                path.name: diagnostic._sha256(path) for path in output_paths
            },
            "source_sha256": diagnostic._sha256(Path(__file__)),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
