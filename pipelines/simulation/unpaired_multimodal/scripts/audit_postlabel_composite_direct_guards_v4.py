#!/usr/bin/env python3
"""Post-label composite guard audit over immutable retired-TRAIN evidence.

The elementary evidence and threshold families were frozen before TRAIN380
truth access.  The conjunctions in this script were assembled after labels
were opened, so the output is explicitly exploratory and cannot certify a
production change.  Its purpose is to reject or retain candidate mechanisms
before a fresh disjoint TRAIN block is registered.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import audit_retired_direct_guard_rules_v4 as base
import diagnose_formal_train_direct_failures_v4 as diagnostic
from observed_collider_structure import apply_observed_collider_refinements


COMPLETION = "complete_dual_supported_unshielded_anchor"
RESCUE_ONLY = "complete_unique_rescue_only_state_loading_contrast"
MULTICENTER = "resolve_competing_multicenter_path"


def _completion_not_double_negative(row: pd.Series) -> bool:
    if str(row.action) != COMPLETION:
        return True
    state_selected = bool(row.candidate_new_parent_formal_state_selected)
    state_floor = float(row.selected_arm_outer_state_support_floor)
    return state_selected or state_floor > 0.0


def _rescue_has_any_base_anchor(row: pd.Series) -> bool:
    return bool(
        str(row.action) != RESCUE_ONLY
        or float(row.anchor_base_view_count) >= 1.0
    )


def _multicenter_not_jointly_contradicted(row: pd.Series) -> bool:
    if str(row.action) != MULTICENTER:
        return True
    separation = float(row.median_collider_separation_score)
    winner_views = int(row.winner_view_count)
    state_floor = float(row.median_outer_state_support_floor)
    return bool(
        separation > 0.0
        or (winner_views == 6 and state_floor > 0.0025)
    )


def _rules() -> dict[str, Callable[[pd.Series], bool]]:
    return {
        "production": lambda row: True,
        "completion_double_negative_abstention": _completion_not_double_negative,
        "rescue_any_base_anchor": _rescue_has_any_base_anchor,
        "completion_plus_rescue": lambda row: bool(
            _completion_not_double_negative(row)
            and _rescue_has_any_base_anchor(row)
        ),
        "multicenter_disjunctive_guard": _multicenter_not_jointly_contradicted,
        "completion_rescue_multicenter_composite": lambda row: bool(
            _completion_not_double_negative(row)
            and _rescue_has_any_base_anchor(row)
            and _multicenter_not_jointly_contradicted(row)
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train380-phase-a", required=True, type=Path)
    parser.add_argument("--elementary-audit", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phase_a_sha = base._verify_train380_barrier(args.train380_phase_a)
    elementary_manifest = args.elementary_audit / "manifest.json"
    elementary_sha = diagnostic._sha256(elementary_manifest)
    rules = _rules()

    decision_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for block, spec in base.BLOCKS.items():
        worlds = base._load_block_worlds(block)
        truth_direct, truth_sets = diagnostic._load_direct_truth_after_barrier(
            worlds, Path(spec["world_root"])
        )
        truth_map = diagnostic._truth_wedge_map(truth_direct)
        for world in worlds:
            seed = int(world["seed"])
            motif, refinement = base._selector_inputs(world)
            decisions = refinement.decisions.copy()
            enriched_records: list[dict[str, Any]] = []
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
                enriched_records.append(
                    {
                        "block": block,
                        "seed": seed,
                        **decision.to_dict(),
                        **base._decision_features(
                            decision, refinement, world["subspace_features"]
                        ),
                        "decision_exact": exact,
                    }
                )
            enriched = pd.DataFrame(enriched_records)
            truth = truth_sets[seed]
            for rule_name, predicate in rules.items():
                if decisions.empty:
                    keep_mask: list[bool] = []
                    retained = decisions.copy()
                else:
                    keep_mask = [
                        bool(predicate(row)) for _, row in enriched.iterrows()
                    ]
                    retained = decisions.loc[keep_mask].reset_index(drop=True)
                    for record, keep in zip(enriched_records, keep_mask):
                        decision_rows.append(
                            {**record, "rule": rule_name, "retained": keep}
                        )
                filtered = replace(refinement, decisions=retained)
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
                        "seed": seed,
                        "rule": rule_name,
                        "n_production_decisions": len(decisions),
                        "n_retained_decisions": len(retained),
                        **metric,
                    }
                )

    decision_rules = pd.DataFrame(decision_rows)
    graphs = pd.DataFrame(graph_rows)
    decision_summary = (
        decision_rules.groupby(["block", "rule"], sort=True)
        .agg(
            n_decisions=("decision_exact", "size"),
            retained_decisions=("retained", "sum"),
            retained_exact=(
                "decision_exact",
                lambda values: int(
                    values[
                        decision_rules.loc[values.index, "retained"].astype(bool)
                    ].sum()
                ),
            ),
            retained_false=(
                "decision_exact",
                lambda values: int(
                    (~values.astype(bool))[
                        decision_rules.loc[values.index, "retained"].astype(bool)
                    ].sum()
                ),
            ),
            removed_exact=(
                "decision_exact",
                lambda values: int(
                    values[
                        ~decision_rules.loc[values.index, "retained"].astype(bool)
                    ].sum()
                ),
            ),
            removed_false=(
                "decision_exact",
                lambda values: int(
                    (~values.astype(bool))[
                        ~decision_rules.loc[values.index, "retained"].astype(bool)
                    ].sum()
                ),
            ),
        )
        .reset_index()
    )
    all_decisions = (
        decision_rules.groupby("rule", sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "n_decisions": len(group),
                    "retained_decisions": int(group["retained"].sum()),
                    "retained_exact": int(
                        group.loc[group["retained"], "decision_exact"].sum()
                    ),
                    "retained_false": int(
                        (~group.loc[group["retained"], "decision_exact"].astype(bool)).sum()
                    ),
                    "removed_exact": int(
                        group.loc[~group["retained"], "decision_exact"].sum()
                    ),
                    "removed_false": int(
                        (~group.loc[~group["retained"], "decision_exact"].astype(bool)).sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    all_decisions.insert(0, "block", "ALL_RETIRED")
    decision_summary = pd.concat(
        [all_decisions, decision_summary], ignore_index=True
    )

    graph_summary = (
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
    graph_summary = pd.concat([all_graph, graph_summary], ignore_index=True)

    outputs = {
        "decision_rule_rows.tsv": decision_rules,
        "decision_rule_summary.tsv": decision_summary,
        "world_rule_metrics.tsv": graphs,
        "graph_rule_summary.tsv": graph_summary,
    }
    paths: list[Path] = []
    for name, frame in outputs.items():
        path = args.out_dir / name
        frame.to_csv(path, sep="\t", index=False)
        paths.append(path)
    summary = {
        "schema_version": "ca_boss_v4.postlabel_composite_direct_guard_audit.v1",
        "status": "postlabel_exploration_not_certification",
        "elementary_evidence_was_frozen_before_train380_labels": True,
        "composite_rules_were_assembled_after_train380_labels": True,
        "train380_phase_a_manifest_sha256": phase_a_sha,
        "elementary_audit_manifest_sha256": elementary_sha,
        "blocks": list(base.BLOCKS),
        "rules": list(rules),
        "dev_or_locked_opened": False,
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    paths.append(summary_path)
    _write_json(
        args.out_dir / "manifest.json",
        {
            "output_sha256": {
                path.name: diagnostic._sha256(path) for path in paths
            },
            "source_sha256": diagnostic._sha256(Path(__file__)),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
