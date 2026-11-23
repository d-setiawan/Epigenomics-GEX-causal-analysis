#!/usr/bin/env python3
"""Label and score the frozen TRAIN-370 loading-rescue rule grid.

This Phase-B process refuses to proceed unless both the observed-only Phase-A
freeze and the separately published prelabel rule-grid freeze validate.  It
then opens TRAIN roles, quantifies production and candidate-rule errors, and
performs retrospective compatibility checks on the already-retired 330 and
360 development blocks.  It never reads DEV or LOCKED artifacts.
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

import numpy as np
import pandas as pd


FORMAL_TRAIN_SEEDS = tuple(range(37_001, 37_025))
PRODUCTION_RULE = "production_only"
RECOMMENDED_RULE = "p50_1.00__s0.70__u0.50__c0.70"
TIERED_RULE = "recommended_tiered__p25_half_or_p50_all__u0.50_c0.70_s0.70"
N_TRUE_ASSAY_PAIRS = 144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_freezes(
    phase_a: Path, rule_grid: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    phase_freeze_path = phase_a / "global_phase_a_freeze.json"
    rule_freeze_path = rule_grid / "prelabel_rule_grid_freeze.json"
    phase_freeze = json.loads(phase_freeze_path.read_text())
    rule_freeze = json.loads(rule_freeze_path.read_text())
    if tuple(phase_freeze.get("formal_train_seeds", ())) != FORMAL_TRAIN_SEEDS:
        raise ValueError("Phase A seed registration mismatch")
    if phase_freeze.get("role_truth_or_score_domain_read_by_this_process"):
        raise ValueError("Phase A reports privileged access")
    if not rule_freeze.get(
        "all_rule_selections_frozen_without_roles_truth_or_scores"
    ):
        raise ValueError("Rule grid lacks a strict prelabel freeze")
    if rule_freeze.get("world_role_truth_or_score_paths_opened"):
        raise ValueError("Rule-grid process reports privileged access")
    if rule_freeze.get("phase_a_freeze_sha256") != _sha256(phase_freeze_path):
        raise ValueError("Rule grid is not bound to this Phase-A freeze")
    candidates_path = phase_a / "observable_candidates.tsv"
    views_path = phase_a / "observable_pair_views.tsv"
    selections_path = rule_grid / "prelabel_rule_selections.tsv"
    decisions_path = rule_grid / "prelabel_rule_decisions.tsv"
    expected = phase_freeze["output_sha256"]
    if _sha256(candidates_path) != expected["observable_candidates.tsv"]:
        raise ValueError("Candidate table hash mismatch")
    if _sha256(views_path) != expected["observable_pair_views.tsv"]:
        raise ValueError("View table hash mismatch")
    if _sha256(selections_path) != rule_freeze["prelabel_selection_sha256"]:
        raise ValueError("Prelabel selections hash mismatch")
    if _sha256(decisions_path) != rule_freeze["prelabel_decision_sha256"]:
        raise ValueError("Prelabel decisions hash mismatch")
    candidates = pd.read_csv(candidates_path, sep="\t")
    views = pd.read_csv(views_path, sep="\t")
    selections = pd.read_csv(selections_path, sep="\t")
    decisions = pd.read_csv(decisions_path, sep="\t")
    return candidates, views, selections, decisions, phase_freeze, rule_freeze


def _load_roles(world_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if world_root.name != "train_37001_37024":
        raise ValueError("Only retired TRAIN 37001--37024 is allowed")
    if any(token in str(world_root).casefold() for token in ("locked", "dev")):
        raise ValueError("DEV/LOCKED roots are forbidden")
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for seed in FORMAL_TRAIN_SEEDS:
        path = world_root / f"world_seed{seed}" / "scorer" / "role_assignments.tsv"
        frame = pd.read_csv(path, sep="\t")[["node", "role"]].copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)
        hashes[str(seed)] = _sha256(path)
    roles = pd.concat(frames, ignore_index=True)
    if roles.duplicated(["seed", "node"]).any():
        raise ValueError("Role table duplicates world/node")
    return roles, hashes


def _label_candidates(candidates: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    lookup = roles.set_index(["seed", "node"])["role"]
    output = candidates.copy()
    for target, feature in (
        ("parent_role", "parent_candidate"),
        ("sibling_role", "noncollider_sibling_candidate"),
        ("other_parent_role", "other_parent_candidate"),
        ("center_role", "collider_center"),
    ):
        output[target] = [
            str(lookup.loc[(int(row.seed), str(getattr(row, feature)))])
            for row in output.itertuples(index=False)
        ]
    output["exact_ps_pair"] = [
        {str(row.parent_role), str(row.sibling_role)} == {"P", "S"}
        for row in output.itertuples(index=False)
    ]
    output["pair_role_type"] = output["parent_role"] + "-" + output["sibling_role"]
    return output


def _score_rules(
    labeled: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labeled[
        [
            "seed",
            "modality",
            "pair_key",
            "exact_ps_pair",
            "pair_role_type",
            "parent_role",
            "sibling_role",
        ]
    ]
    scored = selections.merge(
        labels,
        on=["seed", "modality", "pair_key"],
        how="left",
        validate="many_to_one",
    )
    if scored["exact_ps_pair"].isna().any():
        raise ValueError("A frozen selection lacks a label")
    scored["exact_ps_pair"] = scored["exact_ps_pair"].astype(bool)
    rows: list[dict[str, Any]] = []
    for rule_id, subset in scored.groupby("rule_id", sort=True):
        rescue = subset.loc[
            subset["diagnostic_selection_stage"].eq("abstention_only_rescue")
        ]
        tp = int(subset["exact_ps_pair"].sum())
        fp = int((~subset["exact_ps_pair"]).sum())
        rows.append(
            {
                "rule_id": rule_id,
                "n_selected": len(subset),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": N_TRUE_ASSAY_PAIRS - tp,
                "precision": tp / len(subset) if len(subset) else 1.0,
                "recall": tp / N_TRUE_ASSAY_PAIRS,
                "n_rescued": len(rescue),
                "rescue_true_positive": int(rescue["exact_ps_pair"].sum()),
                "rescue_false_positive": int((~rescue["exact_ps_pair"]).sum()),
            }
        )
    return scored, pd.DataFrame(rows)


def _per_assay(
    labeled: pd.DataFrame,
    scored: pd.DataFrame,
    formal_decisions: pd.DataFrame,
    rule_ids: tuple[str, ...],
) -> pd.DataFrame:
    formal_decisions = formal_decisions.rename(
        columns={
            "selected_pair_key": "formal_selected_pair_key",
            "selection_status": "formal_selection_status",
            "selection_stage": "formal_selection_stage",
        }
    )
    modalities = sorted(formal_decisions["modality"].unique())
    assays = pd.MultiIndex.from_product(
        [FORMAL_TRAIN_SEEDS, modalities], names=["seed", "modality"]
    ).to_frame(index=False)
    exact = labeled.loc[labeled["exact_ps_pair"].astype(bool)].copy()
    exact_counts = exact.groupby(["seed", "modality"]).size()
    if (exact_counts > 1).any():
        raise ValueError("More than one exact P/S structural candidate in an assay")
    exact = exact[
        ["seed", "modality", "pair_key", "parent_candidate", "noncollider_sibling_candidate"]
    ].rename(
        columns={
            "pair_key": "exact_pair_key",
            "parent_candidate": "exact_parent_candidate",
            "noncollider_sibling_candidate": "exact_sibling_candidate",
        }
    )
    frames: list[pd.DataFrame] = []
    for rule_id in rule_ids:
        selected = scored.loc[
            scored["rule_id"].eq(rule_id),
            [
                "seed",
                "modality",
                "pair_key",
                "diagnostic_selection_stage",
                "exact_ps_pair",
                "pair_role_type",
            ],
        ].rename(
            columns={
                "pair_key": "selected_pair_key",
                "exact_ps_pair": "selected_exact_ps_pair",
                "pair_role_type": "selected_pair_role_type",
            }
        )
        frame = (
            assays.merge(formal_decisions, on=["seed", "modality"], how="left")
            .merge(exact, on=["seed", "modality"], how="left")
            .merge(selected, on=["seed", "modality"], how="left")
        )
        frame.insert(0, "rule_id", rule_id)
        frame["exact_candidate_available"] = frame["exact_pair_key"].notna()
        frame["predicted"] = frame["selected_pair_key"].notna()
        frame["true_positive"] = frame["selected_exact_ps_pair"].fillna(False).astype(bool)
        frame["false_positive"] = frame["predicted"] & ~frame["true_positive"]
        frame["false_negative"] = ~frame["true_positive"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fn_gate_audit(
    labeled: pd.DataFrame, production_assays: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    misses = production_assays.loc[production_assays["false_negative"]].copy()
    exact = labeled.loc[labeled["exact_ps_pair"].astype(bool)].copy()
    fields = [
        "seed",
        "modality",
        "pair_key",
        "passes_state_membership_gate",
        "passes_cosine_gate",
        "passes_relative_parent_margin_gate",
        "passes_all_selection_gates",
        "passes_rescue_parent_win_gate",
        "passes_rescue_empirical_null_gate",
        "relative_median_parent_margin",
        "parent_win_view_support",
        "empirical_null_median_p",
        "empirical_null_support_p_le_0_125",
        "empirical_null_support_p_le_0_25",
        "empirical_null_support_p_le_0_50",
        "lower_absolute_loading_cosine",
        "median_absolute_loading_cosine",
        "median_minimum_loading_stability",
        "view_q90__candidate_pair_max_relative_uncertainty",
        "view_median__candidate_loading_empirical_null_p",
        "view_q90__candidate_loading_empirical_null_p",
        "view_max__candidate_loading_empirical_null_p",
    ]
    exact = exact[fields].rename(
        columns={column: f"exact__{column}" for column in fields if column not in {"seed", "modality"}}
    )
    audit = misses.merge(exact, on=["seed", "modality"], how="left", validate="one_to_one")
    available = audit["exact__pair_key"].notna()
    boolean_failures = {
        "fails_structural_candidate_availability": ~available,
        "fails_formal_sibling_state_gate": available
        & ~audit["exact__passes_state_membership_gate"].fillna(False).astype(bool),
        "fails_baseline_cosine_gate": available
        & ~audit["exact__passes_cosine_gate"].fillna(False).astype(bool),
        "fails_relative_parent_margin_gate": available
        & ~audit["exact__passes_relative_parent_margin_gate"].fillna(False).astype(bool),
        "fails_parent_win_unanimity": available
        & audit["exact__parent_win_view_support"].fillna(-np.inf).lt(1.0),
        "fails_empirical_null_p125_unanimity": available
        & audit["exact__empirical_null_support_p_le_0_125"].fillna(-np.inf).lt(1.0),
        "fails_empirical_null_p25_half_support": available
        & audit["exact__empirical_null_support_p_le_0_25"].fillna(-np.inf).lt(0.50),
        "fails_empirical_null_p50_unanimity": available
        & audit["exact__empirical_null_support_p_le_0_50"].fillna(-np.inf).lt(1.0),
        "fails_recommended_stability": available
        & audit["exact__median_minimum_loading_stability"].fillna(-np.inf).lt(0.70),
        "fails_recommended_uncertainty": available
        & audit["exact__view_q90__candidate_pair_max_relative_uncertainty"].fillna(np.inf).gt(0.50),
        "fails_recommended_lower_cosine": available
        & audit["exact__lower_absolute_loading_cosine"].fillna(-np.inf).lt(0.70),
    }
    for column, values in boolean_failures.items():
        audit[column] = values.astype(bool)
    primary: list[str] = []
    for row in audit.itertuples(index=False):
        if bool(row.fails_structural_candidate_availability):
            primary.append("exact_ps_absent_from_structural_candidate_set")
        elif bool(row.fails_formal_sibling_state_gate):
            primary.append("exact_ps_failed_formal_sibling_state_gate")
        elif bool(row.fails_relative_parent_margin_gate) or bool(row.fails_parent_win_unanimity):
            primary.append("exact_ps_failed_parent_comparison")
        elif bool(row.fails_empirical_null_p125_unanimity):
            primary.append("exact_ps_failed_p125_empirical_null_unanimity")
        else:
            primary.append("production_selected_wrong_candidate")
    audit["primary_production_failure"] = primary
    failure_columns = list(boolean_failures)
    summary = pd.DataFrame(
        {
            "gate_failure": failure_columns,
            "n_production_false_negative_assays": [
                int(audit[column].sum()) for column in failure_columns
            ],
            "fraction_of_35_production_false_negatives": [
                float(audit[column].mean()) for column in failure_columns
            ],
        }
    )
    primary_summary = (
        audit.groupby("primary_production_failure", sort=True)
        .size()
        .rename("n_assays")
        .reset_index()
    )
    return audit, summary, primary_summary


def _group_metrics(per_assay: pd.DataFrame, group: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (rule_id, value), frame in per_assay.groupby(["rule_id", group], sort=True):
        tp = int(frame["true_positive"].sum())
        fp = int(frame["false_positive"].sum())
        predicted = int(frame["predicted"].sum())
        rows.append(
            {
                "rule_id": rule_id,
                group: value,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": len(frame) - tp,
                "n_selected": predicted,
                "precision": tp / predicted if predicted else 1.0,
                "recall": tp / len(frame),
            }
        )
    return pd.DataFrame(rows)


def _apply_fixed_rule(
    candidates: pd.DataFrame,
    production_pairs: pd.DataFrame,
    *,
    state_column: str,
    uncertainty_column: str,
    sibling_column: str,
) -> pd.DataFrame:
    production_assays = set(
        map(tuple, production_pairs[["seed", "modality"]].to_records(index=False))
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
        & pool["median_minimum_loading_stability"].ge(0.70)
        & pool[uncertainty_column].le(0.50)
        & pool["lower_absolute_loading_cosine"].ge(0.70)
    )
    eligible = pool.loc[mask].copy()
    counts = eligible.groupby(["seed", "modality"], sort=True).size()
    unique_assays = set(counts.loc[counts.eq(1)].index)
    rescued = eligible.loc[
        [
            (int(row.seed), str(row.modality)) in unique_assays
            for row in eligible.itertuples(index=False)
        ]
    ].copy()
    if sibling_column != "noncollider_sibling_candidate":
        rescued = rescued.rename(columns={sibling_column: "noncollider_sibling_candidate"})
    return rescued


def _historical_crosscheck(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    additions: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}

    root360 = repo_root / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/formal_train_36001_36024_loading_recall_diagnostic_v1"
    cand360_path = root360 / "phase_b_empirical_null_rescue/labeled_candidates.tsv"
    prod360_path = root360 / "phase_b_empirical_null_rescue/proposed_selected_groups.tsv"
    cand360 = pd.read_csv(cand360_path, sep="\t")
    prod360 = pd.read_csv(prod360_path, sep="\t")
    add360 = _apply_fixed_rule(
        cand360,
        prod360,
        state_column="passes_state_membership_gate",
        uncertainty_column="view_q90__candidate_pair_max_relative_uncertainty",
        sibling_column="noncollider_sibling_candidate",
    )
    add360.insert(0, "historical_block", "TRAIN_36001_36024")
    additions.append(add360)
    hashes[str(cand360_path.relative_to(repo_root))] = _sha256(cand360_path)
    hashes[str(prod360_path.relative_to(repo_root))] = _sha256(prod360_path)
    prod360_tp = int(prod360["exact_ps_pair"].astype(bool).sum())
    prod360_fp = len(prod360) - prod360_tp
    rows.append(
        {
            "historical_block": "TRAIN_36001_36024",
            "production_true_positive": prod360_tp,
            "production_false_positive": prod360_fp,
            "recommended_additions": len(add360),
            "recommended_addition_true_positive": int(add360["exact_ps_pair"].sum()),
            "recommended_addition_false_positive": int((~add360["exact_ps_pair"]).sum()),
        }
    )

    old_root = root360 / "old_train_33001_33024_crosscheck"
    cand330_path = repo_root / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/structure_anchored_loading_train_diagnostic_v2/train_labels/labeled_structural_loading_candidates.tsv"
    prod330_path = old_root / "scored_selected.tsv"
    cand330 = pd.read_csv(cand330_path, sep="\t")
    prod330 = pd.read_csv(prod330_path, sep="\t")
    add330 = _apply_fixed_rule(
        cand330,
        prod330,
        state_column="sibling_formal_state_selected",
        uncertainty_column="upper_maximum_relative_loading_uncertainty",
        sibling_column="nonmotif_sibling_candidate",
    )
    add330.insert(0, "historical_block", "TRAIN_33001_33024")
    additions.append(add330)
    hashes[str(cand330_path.relative_to(repo_root))] = _sha256(cand330_path)
    hashes[str(prod330_path.relative_to(repo_root))] = _sha256(prod330_path)
    prod330_tp = int(prod330["exact_ps_pair"].astype(bool).sum())
    prod330_fp = len(prod330) - prod330_tp
    rows.append(
        {
            "historical_block": "TRAIN_33001_33024",
            "production_true_positive": prod330_tp,
            "production_false_positive": prod330_fp,
            "recommended_additions": len(add330),
            "recommended_addition_true_positive": int(add330["exact_ps_pair"].sum()),
            "recommended_addition_false_positive": int((~add330["exact_ps_pair"]).sum()),
        }
    )
    summary = pd.DataFrame(rows)
    summary["combined_true_positive"] = (
        summary["production_true_positive"]
        + summary["recommended_addition_true_positive"]
    )
    summary["combined_false_positive"] = (
        summary["production_false_positive"]
        + summary["recommended_addition_false_positive"]
    )
    summary["combined_precision"] = summary["combined_true_positive"] / (
        summary["combined_true_positive"] + summary["combined_false_positive"]
    )
    summary["combined_recall"] = summary["combined_true_positive"] / N_TRUE_ASSAY_PAIRS
    return summary, pd.concat(additions, ignore_index=True), hashes


def _postlabel_relaxed_state_exploration(
    labeled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the explicitly post-label assay-global/Wald rule requested later."""
    production = labeled.loc[labeled["selected"].astype(bool)].copy()
    production_assays = set(
        map(tuple, production[["seed", "modality"]].to_records(index=False))
    )
    pool = labeled.loc[
        [
            (int(row.seed), str(row.modality)) not in production_assays
            for row in labeled.itertuples(index=False)
        ]
    ].copy()
    eligible_mask = (
        pool["sibling_assay_global_maxT_admission_gate"].astype(bool)
        & pool["sibling_conditional_wald_view_support"].ge(1.0)
        & pool["relative_median_parent_margin"].ge(0.10)
        & pool["parent_win_view_support"].ge(1.0)
        & pool["empirical_null_support_p_le_0_50"].ge(1.0)
    )
    eligible = pool.loc[eligible_mask].copy()
    counts = eligible.groupby(["seed", "modality"], sort=True).size()
    unique_assays = set(counts.loc[counts.eq(1)].index)
    additions = eligible.loc[
        [
            (int(row.seed), str(row.modality)) in unique_assays
            for row in eligible.itertuples(index=False)
        ]
    ].copy()
    additions.insert(0, "rule_status", "postlabel_exploratory_not_preregistered")
    ambiguous_assays = set(counts.loc[counts.gt(1)].index)
    ambiguous = eligible.loc[
        [
            (int(row.seed), str(row.modality)) in ambiguous_assays
            for row in eligible.itertuples(index=False)
        ]
    ].copy()
    ambiguous.insert(0, "rule_status", "abstained_due_to_nonunique_candidate")
    production_world = (
        production.groupby("seed", sort=True)["exact_ps_pair"]
        .agg(production_true_positive="sum", production_n_selected="size")
        .reindex(FORMAL_TRAIN_SEEDS, fill_value=0)
    )
    addition_world = (
        additions.groupby("seed", sort=True)["exact_ps_pair"]
        .agg(addition_true_positive="sum", addition_n_selected="size")
        .reindex(FORMAL_TRAIN_SEEDS, fill_value=0)
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
    return additions, ambiguous, per_world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_37001_37024_loading_recall_diagnostic_v1"
        ),
    )
    parser.add_argument(
        "--world-root", default="pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_37001_37024"
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_37001_37024_loading_recall_diagnostic_v1/phase_b_score"
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    diagnostic_root = Path(args.diagnostic_root).resolve()
    phase_a = diagnostic_root / "phase_a"
    rule_grid = diagnostic_root / "prelabel_rule_grid"
    world_root = Path(args.world_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    candidates, views, selections, frozen_decisions, phase_freeze, rule_freeze = _validate_freezes(
        phase_a, rule_grid
    )

    # This is the first world role/truth-domain access in this process.
    roles, role_hashes = _load_roles(world_root)
    labeled = _label_candidates(candidates, roles)
    scored, rule_metrics = _score_rules(labeled, selections)
    requested_rules = (PRODUCTION_RULE, RECOMMENDED_RULE, TIERED_RULE)
    per_assay = _per_assay(labeled, scored, pd.read_csv(phase_a / "formal_assay_decisions.tsv", sep="\t"), requested_rules)
    production_assays = per_assay.loc[per_assay["rule_id"].eq(PRODUCTION_RULE)]
    fn_audit, gate_summary, primary_summary = _fn_gate_audit(labeled, production_assays)
    world_metrics = _group_metrics(per_assay, "seed")
    modality_metrics = _group_metrics(per_assay, "modality")
    recommended_additions = scored.loc[
        scored["rule_id"].eq(RECOMMENDED_RULE)
        & scored["diagnostic_selection_stage"].eq("abstention_only_rescue")
    ].copy()
    false_selections = scored.loc[
        scored["rule_id"].isin(requested_rules) & ~scored["exact_ps_pair"].astype(bool)
    ].copy()
    miss_keys = fn_audit.loc[fn_audit["exact__pair_key"].notna(), ["seed", "exact__pair_key"]].rename(
        columns={"exact__pair_key": "pair_key"}
    )
    miss_views = views.merge(miss_keys, on=["seed", "pair_key"], how="inner", validate="many_to_one")
    miss_views = miss_views.merge(
        labeled[["seed", "pair_key", "parent_role", "sibling_role", "exact_ps_pair"]],
        on=["seed", "pair_key"],
        how="left",
        validate="many_to_one",
    )
    historical_summary, historical_additions, historical_hashes = _historical_crosscheck(repo_root)
    relaxed_additions, relaxed_ambiguous, relaxed_per_world = (
        _postlabel_relaxed_state_exploration(labeled)
    )

    production_metric = rule_metrics.loc[rule_metrics["rule_id"].eq(PRODUCTION_RULE)].iloc[0]
    recommended_metric = rule_metrics.loc[rule_metrics["rule_id"].eq(RECOMMENDED_RULE)].iloc[0]
    tiered_metric = rule_metrics.loc[rule_metrics["rule_id"].eq(TIERED_RULE)].iloc[0]
    if (int(production_metric["true_positive"]), int(production_metric["false_positive"])) != (109, 1):
        raise ValueError("Did not reproduce production 109 TP / 1 FP")
    if (int(recommended_metric["true_positive"]), int(recommended_metric["false_positive"])) != (116, 1):
        raise ValueError("Frozen narrow rule did not yield 116 TP / 1 FP")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        outputs = {
            "labeled_candidates.tsv": labeled,
            "all_scored_rule_selections.tsv": scored,
            "rule_metrics.tsv": rule_metrics,
            "production_recommended_tiered_per_assay.tsv": per_assay,
            "production_recommended_tiered_per_world.tsv": world_metrics,
            "production_recommended_tiered_per_modality.tsv": modality_metrics,
            "production_false_negative_gate_audit.tsv": fn_audit,
            "production_false_negative_gate_summary.tsv": gate_summary,
            "production_false_negative_primary_summary.tsv": primary_summary,
            "production_false_negative_exact_candidate_views.tsv": miss_views,
            "recommended_rescue_additions.tsv": recommended_additions,
            "requested_rule_false_selections.tsv": false_selections,
            "historical_330_360_crosscheck_summary.tsv": historical_summary,
            "historical_330_360_recommended_additions.tsv": historical_additions,
            "postlabel_relaxed_state_additions.tsv": relaxed_additions,
            "postlabel_relaxed_state_ambiguous_candidates.tsv": relaxed_ambiguous,
            "postlabel_relaxed_state_per_world.tsv": relaxed_per_world,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, sep="\t", index=False)
        output_hashes = {filename: _sha256(temporary / filename) for filename in outputs}
        summary = {
            "schema_version": "ca_boss_v4.formal_train_370_loading_recall.score.v1",
            "block_status": "immutable_development_history",
            "strict_phase_a_then_prelabel_rule_freeze_before_roles": True,
            "dev_or_locked_read": False,
            "phase_a_freeze_sha256": _sha256(phase_a / "global_phase_a_freeze.json"),
            "prelabel_rule_grid_freeze_sha256": _sha256(
                rule_grid / "prelabel_rule_grid_freeze.json"
            ),
            "prelabel_selection_sha256": rule_freeze["prelabel_selection_sha256"],
            "production": {
                "true_positive": int(production_metric["true_positive"]),
                "false_positive": int(production_metric["false_positive"]),
                "false_negative": int(production_metric["false_negative"]),
                "precision": float(production_metric["precision"]),
                "recall": float(production_metric["recall"]),
                "formal_baseline_true_positive": 108,
                "formal_baseline_false_positive": 1,
                "p125_unanimity_rescue_true_positive": 1,
                "p125_unanimity_rescue_false_positive": 0,
            },
            "recommended_narrow_rule": {
                "rule_id": RECOMMENDED_RULE,
                "abstention_only": True,
                "production_never_replaced": True,
                "formal_sibling_state_gate": True,
                "minimum_relative_parent_margin": 0.10,
                "parent_win_support": "6/6",
                "loading_empirical_null_p_at_most_0_50_support": "6/6",
                "minimum_median_minimum_loading_stability": 0.70,
                "maximum_q90_pair_relative_uncertainty": 0.50,
                "minimum_lower_absolute_loading_cosine": 0.70,
                "unique_candidate_within_assay": True,
                "true_positive": int(recommended_metric["true_positive"]),
                "false_positive": int(recommended_metric["false_positive"]),
                "false_negative": int(recommended_metric["false_negative"]),
                "precision": float(recommended_metric["precision"]),
                "recall": float(recommended_metric["recall"]),
                "n_rescued": int(recommended_metric["n_rescued"]),
                "rescue_true_positive": int(recommended_metric["rescue_true_positive"]),
                "rescue_false_positive": int(recommended_metric["rescue_false_positive"]),
            },
            "exploratory_tiered_rule": {
                "rule_id": TIERED_RULE,
                "true_positive": int(tiered_metric["true_positive"]),
                "false_positive": int(tiered_metric["false_positive"]),
                "precision": float(tiered_metric["precision"]),
                "recall": float(tiered_metric["recall"]),
                "n_rescued": int(tiered_metric["n_rescued"]),
                "not_recommended_due_to_greater_complexity": True,
            },
            "historical_compatibility": historical_summary.to_dict(orient="records"),
            "postlabel_exploratory_relaxed_state_rule": {
                "not_in_prelabel_rule_grid": True,
                "not_eligible_for_rule_selection_or_certification": True,
                "baseline_abstention_only": True,
                "sibling_assay_global_maxT_admission_required": True,
                "sibling_conditional_wald_view_support": "6/6",
                "feature_global_maxT_or_formal_selection_required": False,
                "minimum_relative_parent_margin": 0.10,
                "parent_win_support": "6/6",
                "loading_empirical_null_p_at_most_0_50_support": "6/6",
                "cosine_stability_or_uncertainty_gate": False,
                "unique_candidate_within_assay": True,
                "n_raw_eligible_candidates": len(relaxed_additions) + len(relaxed_ambiguous),
                "n_unique_additions": len(relaxed_additions),
                "addition_true_positive": int(relaxed_additions["exact_ps_pair"].sum()),
                "addition_false_positive": int((~relaxed_additions["exact_ps_pair"]).sum()),
                "n_ambiguous_assays": int(
                    relaxed_ambiguous[["seed", "modality"]].drop_duplicates().shape[0]
                ),
                "combined_true_positive": 109
                + int(relaxed_additions["exact_ps_pair"].sum()),
                "combined_false_positive": 1
                + int((~relaxed_additions["exact_ps_pair"]).sum()),
            },
            "role_assignment_sha256": role_hashes,
            "historical_input_sha256": historical_hashes,
            "output_sha256": output_hashes,
            "semantics": {
                "within_assay_shared_state_support_only": True,
                "direct_causal_claimed": False,
                "cross_assay_pair_claimed": False,
                "cross_assay_direction_claimed": False,
            },
            "certification_boundary": {
                "rule_selected_after_370_became_development_history": True,
                "370_result_is_not_confirmation": True,
                "requires_new_disjoint_train_block": True,
            },
        }
        _write_json(temporary / "aggregate.json", summary)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.formal_train_370_loading_recall.run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "phase_a_freeze_sha256": summary["phase_a_freeze_sha256"],
                "prelabel_rule_grid_freeze_sha256": summary[
                    "prelabel_rule_grid_freeze_sha256"
                ],
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
