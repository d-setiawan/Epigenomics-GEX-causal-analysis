#!/usr/bin/env python3
"""Post-label exploration of a hierarchical state/loading rescue.

This diagnostic is deliberately outside the prelabel rule grid.  It evaluates
a candidate for a *future disjoint* TRAIN block: retain production; among each
abstained assay's p50/parent-qualified candidates, first require the full
formal sibling-state gate and select only a singleton; only when that tier is
empty, fall back to assay-global maxT admission plus conditional-Wald support
in all views, again requiring a singleton.  No current result is certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _select(
    candidates: pd.DataFrame,
    production: pd.DataFrame,
    *,
    formal_state_column: str,
    assay_state_column: str,
    wald_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    production_assays = set(
        map(tuple, production[["seed", "modality"]].to_records(index=False))
    )
    pool = candidates.loc[
        [
            (int(row.seed), str(row.modality)) not in production_assays
            for row in candidates.itertuples(index=False)
        ]
    ].copy()
    common = (
        pool["relative_median_parent_margin"].ge(0.10)
        & pool["parent_win_view_support"].ge(1.0)
        & pool["empirical_null_support_p_le_0_50"].ge(1.0)
    )
    selected_frames: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    for (seed, modality), assay in pool.groupby(["seed", "modality"], sort=True):
        assay_common = common.loc[assay.index]
        formal = assay.loc[assay_common & assay[formal_state_column].astype(bool)]
        relaxed = assay.loc[
            assay_common
            & assay[assay_state_column].astype(bool)
            & assay[wald_column].ge(1.0)
        ]
        if len(formal) == 1:
            choice = formal.copy()
            stage = "formal_state_singleton_tier"
            reason = ""
        elif len(formal) > 1:
            choice = formal.iloc[0:0].copy()
            stage = "none"
            reason = "ambiguous_formal_state_tier"
        elif len(relaxed) == 1:
            choice = relaxed.copy()
            stage = "assay_global_wald_singleton_fallback"
            reason = ""
        elif len(relaxed) > 1:
            choice = relaxed.iloc[0:0].copy()
            stage = "none"
            reason = "ambiguous_assay_global_wald_fallback"
        else:
            choice = relaxed.iloc[0:0].copy()
            stage = "none"
            reason = "no_candidate_in_either_tier"
        if len(choice):
            choice.insert(0, "hierarchical_selection_stage", stage)
            selected_frames.append(choice)
            pair_key = str(choice.iloc[0]["pair_key"])
        else:
            pair_key = ""
        decisions.append(
            {
                "seed": int(seed),
                "modality": str(modality),
                "selection_status": "selected" if len(choice) else "unresolved",
                "selection_stage": stage,
                "selected_pair_key": pair_key,
                "n_formal_tier_candidates": len(formal),
                "n_relaxed_tier_candidates": len(relaxed),
                "abstention_reason": reason,
            }
        )
    selected = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else candidates.iloc[0:0].copy()
    )
    return selected, pd.DataFrame(decisions)


def _metrics(label: str, production: pd.DataFrame, additions: pd.DataFrame) -> dict[str, Any]:
    production_tp = int(production["exact_ps_pair"].astype(bool).sum())
    production_fp = len(production) - production_tp
    addition_tp = int(additions["exact_ps_pair"].astype(bool).sum())
    addition_fp = len(additions) - addition_tp
    total_tp = production_tp + addition_tp
    total_fp = production_fp + addition_fp
    return {
        "block": label,
        "production_true_positive": production_tp,
        "production_false_positive": production_fp,
        "addition_true_positive": addition_tp,
        "addition_false_positive": addition_fp,
        "n_additions": len(additions),
        "combined_true_positive": total_tp,
        "combined_false_positive": total_fp,
        "combined_precision": total_tp / (total_tp + total_fp),
        "combined_recall": total_tp / 144.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_37001_37024_loading_recall_diagnostic_v1/"
            "postlabel_hierarchical_state_exploration"
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    score_root = repo_root / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/formal_train_37001_37024_loading_recall_diagnostic_v1/phase_b_score"
    candidates370_path = score_root / "labeled_candidates.tsv"
    candidates370 = pd.read_csv(candidates370_path, sep="\t")
    production370 = candidates370.loc[candidates370["selected"].astype(bool)].copy()
    additions370, decisions370 = _select(
        candidates370,
        production370,
        formal_state_column="passes_state_membership_gate",
        assay_state_column="sibling_assay_global_maxT_admission_gate",
        wald_column="sibling_conditional_wald_view_support",
    )

    root360 = repo_root / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/formal_train_36001_36024_loading_recall_diagnostic_v1"
    candidates360_path = root360 / "phase_b_empirical_null_rescue/labeled_candidates.tsv"
    production360_path = root360 / "phase_b_empirical_null_rescue/proposed_selected_groups.tsv"
    candidates360 = pd.read_csv(candidates360_path, sep="\t")
    production360 = pd.read_csv(production360_path, sep="\t")
    additions360, decisions360 = _select(
        candidates360,
        production360,
        formal_state_column="passes_state_membership_gate",
        assay_state_column="sibling_assay_global_maxT_admission_gate",
        wald_column="sibling_conditional_wald_view_support",
    )

    candidates330_path = repo_root / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/structure_anchored_loading_train_diagnostic_v2/train_labels/labeled_structural_loading_candidates.tsv"
    production330_path = root360 / "old_train_33001_33024_crosscheck/scored_selected.tsv"
    candidates330 = pd.read_csv(candidates330_path, sep="\t")
    production330 = pd.read_csv(production330_path, sep="\t")
    additions330, decisions330 = _select(
        candidates330,
        production330,
        formal_state_column="sibling_formal_state_selected",
        assay_state_column="sibling_assay_global_maxT_admission_gate",
        wald_column="sibling_conditional_wald_view_support",
    )

    metrics = pd.DataFrame(
        [
            _metrics("TRAIN_33001_33024", production330, additions330),
            _metrics("TRAIN_36001_36024", production360, additions360),
            _metrics("TRAIN_37001_37024", production370, additions370),
        ]
    )
    additions330.insert(0, "block", "TRAIN_33001_33024")
    additions360.insert(0, "block", "TRAIN_36001_36024")
    additions370.insert(0, "block", "TRAIN_37001_37024")
    decisions330.insert(0, "block", "TRAIN_33001_33024")
    decisions360.insert(0, "block", "TRAIN_36001_36024")
    decisions370.insert(0, "block", "TRAIN_37001_37024")
    additions = pd.concat([additions330, additions360, additions370], ignore_index=True)
    decisions = pd.concat([decisions330, decisions360, decisions370], ignore_index=True)

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

    if tuple(metrics.loc[metrics["block"].eq("TRAIN_37001_37024"), ["addition_true_positive", "addition_false_positive"]].iloc[0]) != (22, 0):
        raise ValueError("Canonical 370 hierarchical result is not 22 TP / 0 FP additions")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        outputs = {
            "cross_block_metrics.tsv": metrics,
            "cross_block_additions.tsv": additions,
            "cross_block_assay_decisions.tsv": decisions,
            "train_370_per_world.tsv": per_world,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, sep="\t", index=False)
        summary = {
            "schema_version": "ca_boss_v4.postlabel_hierarchical_state_loading.v1",
            "status": "postlabel_exploratory_not_preregistered",
            "eligible_for_current_block_certification": False,
            "requires_new_disjoint_train_block": True,
            "rule": {
                "production_retained_without_replacement": True,
                "applies_only_to_production_abstentions": True,
                "minimum_relative_parent_margin": 0.10,
                "parent_win_support": "6/6",
                "loading_empirical_null_p_at_most_0_50_support": "6/6",
                "first_tier": "unique full-formal-state sibling candidate",
                "fallback_only_if_first_tier_empty": (
                    "unique assay-global-maxT plus conditional-Wald-6/6 sibling candidate"
                ),
                "cosine_stability_or_uncertainty_gate": False,
                "direct_parent_state_membership_gate": False,
            },
            "metrics": metrics.to_dict(orient="records"),
            "input_sha256": {
                str(candidates370_path.relative_to(repo_root)): _sha256(candidates370_path),
                str(candidates360_path.relative_to(repo_root)): _sha256(candidates360_path),
                str(production360_path.relative_to(repo_root)): _sha256(production360_path),
                str(candidates330_path.relative_to(repo_root)): _sha256(candidates330_path),
                str(production330_path.relative_to(repo_root)): _sha256(production330_path),
            },
            "output_sha256": {
                filename: _sha256(temporary / filename) for filename in outputs
            },
            "semantics": {
                "within_assay_shared_state_support_only": True,
                "direct_causal_claimed": False,
                "cross_assay_pair_claimed": False,
                "cross_assay_direction_claimed": False,
            },
        }
        _write_json(temporary / "aggregate.json", summary)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.postlabel_hierarchical_state_loading.run.v1",
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
