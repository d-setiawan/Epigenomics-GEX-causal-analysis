#!/usr/bin/env python3
"""Freeze label-free loading-rescue rules for retired TRAIN 37001--37024.

The input is the immutable Phase-A table produced by
``freeze_formal_loading_recall_candidates_370_train_v4.py``.  This process
never opens worlds, roles, truth, scores, DEV, or LOCKED artifacts.  Existing
production selections are retained exactly; every candidate rule is an
abstention-only addition and requires a unique assay-local candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class RescueRule:
    rule_id: str
    null_support_column: str = "empirical_null_support_p_le_0_50"
    minimum_null_support: float = 1.0
    minimum_relative_parent_margin: float = 0.10
    minimum_parent_win_support: float = 1.0
    minimum_loading_stability: float = 0.70
    maximum_q90_pair_relative_uncertainty: float = 0.50
    minimum_lower_absolute_loading_cosine: float = 0.70
    tiered_p25_half_or_configured_gate: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rules() -> list[RescueRule]:
    rules: list[RescueRule] = [
        RescueRule(
            "production_only",
            minimum_null_support=2.0,
            minimum_loading_stability=2.0,
        ),
        RescueRule(
            "recommended_tiered__p25_half_or_p50_all__u0.50_c0.70_s0.70",
            tiered_p25_half_or_configured_gate=True,
        ),
    ]
    for column, family, supports in (
        ("empirical_null_support_p_le_0_25", "p25", (0.50, 1.0)),
        ("empirical_null_support_p_le_0_50", "p50", (1.0,)),
    ):
        for support in supports:
            for stability in (0.0, 0.60, 0.70, 0.80):
                for uncertainty in (0.50, 0.75, 1.0, float("inf")):
                    for cosine in (0.0, 0.60, 0.70, 0.75, 0.80):
                        rules.append(
                            RescueRule(
                                rule_id=(
                                    f"{family}_{support:.2f}__s{stability:.2f}__"
                                    f"u{uncertainty:.2f}__c{cosine:.2f}"
                                ),
                                null_support_column=column,
                                minimum_null_support=support,
                                minimum_loading_stability=stability,
                                maximum_q90_pair_relative_uncertainty=uncertainty,
                                minimum_lower_absolute_loading_cosine=cosine,
                            )
                        )
    unique = {rule.rule_id: rule for rule in rules}
    return [unique[key] for key in sorted(unique)]


def _load_phase_a(phase_a: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    freeze_path = phase_a / "global_phase_a_freeze.json"
    candidates_path = phase_a / "observable_candidates.tsv"
    decisions_path = phase_a / "formal_assay_decisions.tsv"
    freeze = json.loads(freeze_path.read_text())
    if tuple(freeze.get("formal_train_seeds", ())) != FORMAL_TRAIN_SEEDS:
        raise ValueError("Phase A is not TRAIN 37001--37024")
    if not freeze.get(
        "all_24_worlds_loaded_before_any_role_truth_or_score_open_by_this_process"
    ):
        raise ValueError("Phase A did not enforce the information barrier")
    if freeze.get("role_truth_or_score_domain_read_by_this_process"):
        raise ValueError("Phase A reports privileged-domain access")
    for filename, path in (
        ("observable_candidates.tsv", candidates_path),
        ("formal_assay_decisions.tsv", decisions_path),
    ):
        if _sha256(path) != freeze["output_sha256"][filename]:
            raise ValueError(f"Phase-A hash mismatch: {filename}")
    candidates = pd.read_csv(candidates_path, sep="\t")
    decisions = pd.read_csv(decisions_path, sep="\t")
    forbidden_tokens = ("truth", "role", "scorer", "oracle", "generative")
    forbidden = sorted(
        str(column)
        for column in candidates.columns
        if any(token in str(column).casefold() for token in forbidden_tokens)
    )
    if forbidden:
        raise ValueError(f"Observable candidates contain privileged columns: {forbidden}")
    return candidates, decisions, freeze


def _production(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates.loc[candidates["selected"].astype(bool)].copy()
    if selected.duplicated(["seed", "modality"]).any():
        raise ValueError("Production selected multiple loading pairs in an assay")
    selected["diagnostic_selection_stage"] = "production_retained"
    return selected


def _rescue_mask(candidates: pd.DataFrame, rule: RescueRule) -> pd.Series:
    common = (
        candidates["passes_state_membership_gate"].astype(bool)
        & candidates["relative_median_parent_margin"].ge(
            rule.minimum_relative_parent_margin
        )
        & candidates["parent_win_view_support"].ge(rule.minimum_parent_win_support)
    )
    configured = (
        candidates[rule.null_support_column].ge(rule.minimum_null_support)
        & candidates["median_minimum_loading_stability"].ge(
            rule.minimum_loading_stability
        )
        & candidates["view_q90__candidate_pair_max_relative_uncertainty"].le(
            rule.maximum_q90_pair_relative_uncertainty
        )
        & candidates["lower_absolute_loading_cosine"].ge(
            rule.minimum_lower_absolute_loading_cosine
        )
    )
    if rule.tiered_p25_half_or_configured_gate:
        configured = (
            candidates["empirical_null_support_p_le_0_25"].ge(0.50)
            | configured
        )
    return common & configured


def _select(
    candidates: pd.DataFrame,
    production: pd.DataFrame,
    rule: RescueRule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    production_assays = set(
        map(tuple, production[["seed", "modality"]].to_records(index=False))
    )
    abstained = candidates.loc[
        [
            (int(row.seed), str(row.modality)) not in production_assays
            for row in candidates.itertuples(index=False)
        ]
    ]
    eligible = abstained.loc[_rescue_mask(abstained, rule)].copy()
    rescue_counts = eligible.groupby(["seed", "modality"], sort=True).size()
    unique_assays = set(rescue_counts.loc[rescue_counts.eq(1)].index)
    rescued = eligible.loc[
        [
            (int(row.seed), str(row.modality)) in unique_assays
            for row in eligible.itertuples(index=False)
        ]
    ].copy()
    rescued["diagnostic_selection_stage"] = "abstention_only_rescue"
    selected = pd.concat([production, rescued], ignore_index=True)
    selected.insert(0, "rule_id", rule.rule_id)

    assay_rows: list[dict[str, Any]] = []
    modalities = sorted(candidates["modality"].unique())
    selected_lookup = selected.set_index(["seed", "modality"])
    for seed in FORMAL_TRAIN_SEEDS:
        for modality in modalities:
            key = (seed, modality)
            count = int(rescue_counts.get(key, 0))
            if key in selected_lookup.index:
                row = selected_lookup.loc[key]
                assay_rows.append(
                    {
                        "rule_id": rule.rule_id,
                        "seed": seed,
                        "modality": modality,
                        "selection_status": "selected",
                        "diagnostic_selection_stage": str(
                            row["diagnostic_selection_stage"]
                        ),
                        "selected_pair_key": str(row["pair_key"]),
                        "n_rescue_eligible_candidates": count,
                        "abstention_reason": "",
                    }
                )
            else:
                assay_rows.append(
                    {
                        "rule_id": rule.rule_id,
                        "seed": seed,
                        "modality": modality,
                        "selection_status": "unresolved",
                        "diagnostic_selection_stage": "none",
                        "selected_pair_key": "",
                        "n_rescue_eligible_candidates": count,
                        "abstention_reason": (
                            "ambiguous_rescue_candidates"
                            if count > 1
                            else "no_unique_candidate_passed_rule"
                        ),
                    }
                )
    return selected, pd.DataFrame(assay_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-a",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_37001_37024_loading_recall_diagnostic_v1/phase_a"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_37001_37024_loading_recall_diagnostic_v1/"
            "prelabel_rule_grid"
        ),
    )
    args = parser.parse_args()
    phase_a = Path(args.phase_a).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    candidates, formal_decisions, phase_a_freeze = _load_phase_a(phase_a)
    rules = _rules()
    production = _production(candidates)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        selected_frames: list[pd.DataFrame] = []
        decision_frames: list[pd.DataFrame] = []
        for rule in rules:
            selected, decisions = _select(candidates, production, rule)
            selected_frames.append(selected)
            decision_frames.append(decisions)
        all_selected = pd.concat(selected_frames, ignore_index=True)
        all_decisions = pd.concat(decision_frames, ignore_index=True)
        selection_columns = (
            "rule_id",
            "seed",
            "modality",
            "pair_key",
            "diagnostic_selection_stage",
        )
        all_selected.loc[:, selection_columns].to_csv(
            temporary / "prelabel_rule_selections.tsv", sep="\t", index=False
        )
        all_decisions.to_csv(
            temporary / "prelabel_rule_decisions.tsv", sep="\t", index=False
        )
        serializable_rules = []
        for rule in rules:
            payload = asdict(rule)
            if not np.isfinite(payload["maximum_q90_pair_relative_uncertainty"]):
                payload["maximum_q90_pair_relative_uncertainty"] = "inf"
            serializable_rules.append(payload)
        freeze = {
            "schema_version": "ca_boss_v4.formal_train_370_loading_rule_grid.v1",
            "formal_train_seeds": list(FORMAL_TRAIN_SEEDS),
            "phase_a_freeze_sha256": _sha256(phase_a / "global_phase_a_freeze.json"),
            "phase_a_candidate_sha256": phase_a_freeze["output_sha256"][
                "observable_candidates.tsv"
            ],
            "all_rule_selections_frozen_without_roles_truth_or_scores": True,
            "world_role_truth_or_score_paths_opened": False,
            "production_selections_retained_without_replacement": True,
            "rescue_applies_only_to_production_abstentions": True,
            "formal_sibling_state_gate_required": True,
            "direct_parent_state_membership_gate_used": False,
            "unique_rescue_candidate_required": True,
            "rules": serializable_rules,
            "n_rules": len(rules),
            "n_production_selected": len(production),
            "prelabel_selection_sha256": _sha256(
                temporary / "prelabel_rule_selections.tsv"
            ),
            "prelabel_decision_sha256": _sha256(
                temporary / "prelabel_rule_decisions.tsv"
            ),
        }
        _write_json(temporary / "prelabel_rule_grid_freeze.json", freeze)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.formal_train_370_loading_rule_grid_run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "phase_a_freeze_sha256": freeze["phase_a_freeze_sha256"],
                "prelabel_rule_grid_freeze_sha256": _sha256(
                    temporary / "prelabel_rule_grid_freeze.json"
                ),
            },
        )
        os.replace(temporary, out_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
