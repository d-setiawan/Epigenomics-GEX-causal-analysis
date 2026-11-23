#!/usr/bin/env python3
"""Synthesize the TRAIN-370 loading recommendation without production edits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd


RULE_ID = "p50_1.00__s0.00__uinf__c0.00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _formal_p50_additions(
    candidates: pd.DataFrame,
    production: pd.DataFrame,
    state_column: str,
) -> pd.DataFrame:
    production_assays = set(
        map(tuple, production[["seed", "modality"]].to_records(index=False))
    )
    pool = candidates.loc[
        [
            (int(row.seed), str(row.modality)) not in production_assays
            for row in candidates.itertuples(index=False)
        ]
    ].copy()
    mask = (
        pool[state_column].astype(bool)
        & pool["relative_median_parent_margin"].ge(0.10)
        & pool["parent_win_view_support"].ge(1.0)
        & pool["empirical_null_support_p_le_0_50"].ge(1.0)
    )
    eligible = pool.loc[mask].copy()
    counts = eligible.groupby(["seed", "modality"], sort=True).size()
    unique = set(counts.loc[counts.eq(1)].index)
    return eligible.loc[
        [
            (int(row.seed), str(row.modality)) in unique
            for row in eligible.itertuples(index=False)
        ]
    ].copy()


def _metric(block: str, production: pd.DataFrame, additions: pd.DataFrame) -> dict[str, Any]:
    production_tp = int(production["exact_ps_pair"].astype(bool).sum())
    production_fp = len(production) - production_tp
    addition_tp = int(additions["exact_ps_pair"].astype(bool).sum())
    addition_fp = len(additions) - addition_tp
    tp = production_tp + addition_tp
    fp = production_fp + addition_fp
    return {
        "block": block,
        "production_true_positive": production_tp,
        "production_false_positive": production_fp,
        "addition_true_positive": addition_tp,
        "addition_false_positive": addition_fp,
        "n_additions": len(additions),
        "combined_true_positive": tp,
        "combined_false_positive": fp,
        "combined_precision": tp / (tp + fp),
        "combined_recall": tp / 144.0,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[4]
    diagnostic = repo / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/formal_train_37001_37024_loading_recall_diagnostic_v1"
    score_root = diagnostic / "phase_b_score"
    grid_root = diagnostic / "prelabel_rule_grid"
    out_dir = diagnostic / "recommendation_synthesis"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    grid_freeze_path = grid_root / "prelabel_rule_grid_freeze.json"
    grid_freeze = json.loads(grid_freeze_path.read_text())
    rules = {row["rule_id"]: row for row in grid_freeze["rules"]}
    if RULE_ID not in rules:
        raise ValueError("Recommended rule was not in the prelabel-frozen grid")

    candidates370_path = score_root / "labeled_candidates.tsv"
    scored370_path = score_root / "all_scored_rule_selections.tsv"
    candidates370 = pd.read_csv(candidates370_path, sep="\t")
    scored370 = pd.read_csv(scored370_path, sep="\t")
    selected370 = scored370.loc[scored370["rule_id"].eq(RULE_ID)].copy()
    production370 = candidates370.loc[candidates370["selected"].astype(bool)].copy()
    production_keys = set(production370["pair_key"].astype(str))
    additions370 = candidates370.merge(
        selected370[["seed", "modality", "pair_key"]],
        on=["seed", "modality", "pair_key"],
        how="inner",
        validate="one_to_one",
    )
    additions370 = additions370.loc[
        ~additions370["pair_key"].astype(str).isin(production_keys)
    ].copy()

    root360 = repo / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/formal_train_36001_36024_loading_recall_diagnostic_v1"
    candidates360_path = root360 / "phase_b_empirical_null_rescue/labeled_candidates.tsv"
    production360_path = root360 / "phase_b_empirical_null_rescue/proposed_selected_groups.tsv"
    candidates360 = pd.read_csv(candidates360_path, sep="\t")
    production360 = pd.read_csv(production360_path, sep="\t")
    additions360 = _formal_p50_additions(
        candidates360, production360, "passes_state_membership_gate"
    )

    candidates330_path = repo / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/structure_anchored_loading_train_diagnostic_v2/train_labels/labeled_structural_loading_candidates.tsv"
    production330_path = root360 / "old_train_33001_33024_crosscheck/scored_selected.tsv"
    candidates330 = pd.read_csv(candidates330_path, sep="\t")
    production330 = pd.read_csv(production330_path, sep="\t")
    additions330 = _formal_p50_additions(
        candidates330, production330, "sibling_formal_state_selected"
    )

    metrics = pd.DataFrame(
        [
            _metric("TRAIN_33001_33024", production330, additions330),
            _metric("TRAIN_36001_36024", production360, additions360),
            _metric("TRAIN_37001_37024", production370, additions370),
        ]
    )
    additions330.insert(0, "block", "TRAIN_33001_33024")
    additions360.insert(0, "block", "TRAIN_36001_36024")
    additions370.insert(0, "block", "TRAIN_37001_37024")
    additions = pd.concat([additions330, additions360, additions370], ignore_index=True)

    production_world = (
        production370.groupby("seed", sort=True)["exact_ps_pair"]
        .agg(production_true_positive="sum", production_n_selected="size")
        .reindex(range(37_001, 37_025), fill_value=0)
    )
    addition_world = (
        additions370.groupby("seed", sort=True)["exact_ps_pair"]
        .agg(addition_true_positive="sum", addition_n_selected="size")
        .reindex(range(37_001, 37_025), fill_value=0)
    )
    per_world = production_world.join(addition_world).reset_index()
    per_world["production_false_positive"] = (
        per_world["production_n_selected"] - per_world["production_true_positive"]
    )
    per_world["addition_false_positive"] = (
        per_world["addition_n_selected"] - per_world["addition_true_positive"]
    )
    per_world["combined_true_positive"] = (
        per_world["production_true_positive"] + per_world["addition_true_positive"]
    )
    per_world["combined_false_positive"] = (
        per_world["production_false_positive"] + per_world["addition_false_positive"]
    )
    per_world["combined_recall"] = per_world["combined_true_positive"] / 6.0

    if tuple(metrics.loc[metrics["block"].eq("TRAIN_37001_37024"), ["combined_true_positive", "combined_false_positive"]].iloc[0]) != (126, 1):
        raise ValueError("Did not reproduce the frozen-grid 126 TP / 1 FP result")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        outputs = {
            "formal_p50_cross_block_metrics.tsv": metrics,
            "formal_p50_cross_block_additions.tsv": additions,
            "formal_p50_train_370_per_world.tsv": per_world,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, sep="\t", index=False)
        summary = {
            "schema_version": "ca_boss_v4.loading_recommendation_synthesis.v1",
            "status": "development_recommendation_requires_new_disjoint_train",
            "recommended_prelabel_frozen_rule": {
                **rules[RULE_ID],
                "production_retained_without_replacement": True,
                "applies_only_to_production_abstentions": True,
                "unique_candidate_within_assay": True,
                "direct_parent_state_membership_gate": False,
                "interpretation": (
                    "candidate must beat the empirical-null median in all six loading views; "
                    "p<=0.50 is a stability sign criterion, not a single-test significance claim"
                ),
            },
            "formal_p50_cross_block_metrics": metrics.to_dict(orient="records"),
            "postlabel_hierarchical_candidate": {
                "aggregate_path": str(
                    (
                        diagnostic
                        / "postlabel_hierarchical_state_exploration/aggregate.json"
                    ).relative_to(repo)
                ),
                "strictly_exploratory": True,
                "may_be_frozen_for_the_same_next_disjoint_block_as_a_prespecified_comparator": True,
            },
            "prelabel_rule_grid_freeze_sha256": _sha256(grid_freeze_path),
            "input_sha256": {
                str(candidates370_path.relative_to(repo)): _sha256(candidates370_path),
                str(scored370_path.relative_to(repo)): _sha256(scored370_path),
                str(candidates360_path.relative_to(repo)): _sha256(candidates360_path),
                str(production360_path.relative_to(repo)): _sha256(production360_path),
                str(candidates330_path.relative_to(repo)): _sha256(candidates330_path),
                str(production330_path.relative_to(repo)): _sha256(production330_path),
            },
            "output_sha256": {
                filename: _sha256(temporary / filename) for filename in outputs
            },
            "certification_boundary": {
                "370_is_development_history": True,
                "recommendation_selected_after_labels": True,
                "requires_new_disjoint_train": True,
            },
        }
        _write_json(temporary / "aggregate.json", summary)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.loading_recommendation_synthesis.run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "aggregate_sha256": _sha256(temporary / "aggregate.json"),
            },
        )
        os.replace(temporary, out_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
