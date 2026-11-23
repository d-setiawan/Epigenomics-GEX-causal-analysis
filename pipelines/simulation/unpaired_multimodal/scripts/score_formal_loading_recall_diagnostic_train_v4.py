#!/usr/bin/env python3
"""Score a frozen formal-TRAIN loading recall diagnostic after Phase A.

The script first validates the global 36001--36024 observable freeze, builds
and persists every rule-grid selection without labels, and only then opens
the corresponding TRAIN role assignments.  DEV, LOCKED, score outputs, and
all seeds at or above 37001 are outside this diagnostic's information domain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd


FORMAL_TRAIN_SEEDS = tuple(range(36_001, 36_025))
BASELINE_MINIMUM_LOWER_COSINE = 0.85
BASELINE_MINIMUM_RELATIVE_PARENT_MARGIN = 0.10


@dataclass(frozen=True)
class RescueConfig:
    config_id: str
    empirical_null_support_column: str = "empirical_null_support_p_le_0_125"
    minimum_empirical_null_view_support: float = 1.0
    minimum_parent_win_view_support: float = 1.0
    minimum_relative_parent_margin: float = 0.10
    minimum_loading_stability: float = 0.0
    maximum_q90_pair_relative_uncertainty: float = float("inf")
    minimum_lower_absolute_loading_cosine: float = 0.0
    require_unique_rescue_candidate: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _configs() -> list[RescueConfig]:
    configs = [RescueConfig("proposed_empirical_null_unanimity")]
    for column, family in (
        ("empirical_null_support_p_le_0_125", "p125"),
        ("empirical_null_support_p_le_0_25", "p25"),
        ("empirical_null_support_p_le_0_50", "p50"),
    ):
        for support in (4.0 / 6.0, 5.0 / 6.0, 1.0):
            configs.append(
                RescueConfig(
                    f"null_{family}__support_{support:.6f}",
                    empirical_null_support_column=column,
                    minimum_empirical_null_view_support=support,
                )
            )
    for margin in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.425, 0.45, 0.475, 0.50):
        configs.append(
            RescueConfig(
                f"parent_margin_{margin:.3f}",
                minimum_relative_parent_margin=margin,
            )
        )
    for support in (0.50, 4.0 / 6.0, 5.0 / 6.0, 1.0):
        configs.append(
            RescueConfig(
                f"parent_win_support_{support:.6f}",
                minimum_parent_win_view_support=support,
            )
        )
    for stability in (0.0, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90):
        configs.append(
            RescueConfig(
                f"loading_stability_{stability:.3f}",
                minimum_loading_stability=stability,
            )
        )
    for uncertainty in (0.35, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25):
        configs.append(
            RescueConfig(
                f"q90_pair_uncertainty_{uncertainty:.3f}",
                maximum_q90_pair_relative_uncertainty=uncertainty,
            )
        )
    for cosine in (0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.83, 0.84):
        configs.append(
            RescueConfig(
                f"rescue_lower_cosine_{cosine:.3f}",
                minimum_lower_absolute_loading_cosine=cosine,
            )
        )
    unique = {config.config_id: config for config in configs}
    return [unique[key] for key in sorted(unique)]


def _load_phase_a(phase_a: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    freeze_path = phase_a / "global_phase_a_freeze.json"
    candidates_path = phase_a / "observable_candidates.tsv"
    decisions_path = phase_a / "formal_assay_decisions.tsv"
    freeze = json.loads(freeze_path.read_text())
    if tuple(freeze.get("formal_train_seeds", ())) != FORMAL_TRAIN_SEEDS:
        raise ValueError("Phase A is not the formal TRAIN 36001--36024 freeze")
    if not freeze.get("all_24_worlds_loaded_before_any_role_or_scorer_open"):
        raise ValueError("Phase A did not satisfy the global information barrier")
    if freeze.get("role_or_scorer_domain_read"):
        raise ValueError("Phase A reports scorer-domain access")
    for filename, path in (
        ("observable_candidates.tsv", candidates_path),
        ("formal_assay_decisions.tsv", decisions_path),
    ):
        expected = freeze["output_sha256"][filename]
        if _sha256(path) != expected:
            raise ValueError(f"Phase-A hash mismatch for {filename}")
    candidates = pd.read_csv(candidates_path, sep="\t")
    decisions = pd.read_csv(decisions_path, sep="\t")
    forbidden = sorted(
        str(column)
        for column in candidates.columns
        if any(
            token in str(column).casefold()
            for token in ("truth", "role", "scorer", "oracle", "generative")
        )
    )
    if forbidden:
        raise ValueError(f"Frozen observable candidates contain labels: {forbidden}")
    if set(candidates["seed"].astype(int)) != set(FORMAL_TRAIN_SEEDS):
        raise ValueError("Frozen candidates contain an unexpected seed")
    return candidates, decisions, freeze


def _baseline_mask(candidates: pd.DataFrame) -> pd.Series:
    return (
        candidates["passes_state_membership_gate"].astype(bool)
        & (
            candidates["lower_absolute_loading_cosine"]
            >= BASELINE_MINIMUM_LOWER_COSINE
        )
        & (
            candidates["relative_median_parent_margin"]
            >= BASELINE_MINIMUM_RELATIVE_PARENT_MARGIN
        )
    )


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "seed",
            "modality",
            "median_coherence_group_score",
            "median_absolute_loading_cosine",
            "pair_key",
        ],
        ascending=[True, True, False, False, True],
    )


def _select(candidates: pd.DataFrame, config: RescueConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _rank(candidates.loc[_baseline_mask(candidates)]).groupby(
        ["seed", "modality"], sort=True
    ).head(1)
    baseline = baseline.copy()
    baseline["selection_stage"] = "formal_baseline"
    baseline_assays = set(
        map(tuple, baseline[["seed", "modality"]].to_records(index=False))
    )
    not_baseline = pd.Series(
        [
            (int(row.seed), str(row.modality)) not in baseline_assays
            for row in candidates.itertuples(index=False)
        ],
        index=candidates.index,
    )
    rescue_mask = (
        not_baseline
        & candidates["passes_state_membership_gate"].astype(bool)
        & (
            candidates["relative_median_parent_margin"]
            >= config.minimum_relative_parent_margin
        )
        & (
            candidates["parent_win_view_support"]
            >= config.minimum_parent_win_view_support
        )
        & (
            candidates[config.empirical_null_support_column]
            >= config.minimum_empirical_null_view_support
        )
        & (
            candidates["median_minimum_loading_stability"]
            >= config.minimum_loading_stability
        )
        & (
            candidates["view_q90__candidate_pair_max_relative_uncertainty"]
            <= config.maximum_q90_pair_relative_uncertainty
        )
        & (
            candidates["lower_absolute_loading_cosine"]
            >= config.minimum_lower_absolute_loading_cosine
        )
    )
    rescue_pool = _rank(candidates.loc[rescue_mask]).copy()
    pool_counts = rescue_pool.groupby(["seed", "modality"]).size()
    if config.require_unique_rescue_candidate:
        unique_assays = set(pool_counts.loc[pool_counts.eq(1)].index)
        rescue_pool = rescue_pool.loc[
            [
                (int(row.seed), str(row.modality)) in unique_assays
                for row in rescue_pool.itertuples(index=False)
            ]
        ]
    rescue = rescue_pool.groupby(["seed", "modality"], sort=True).head(1).copy()
    rescue["selection_stage"] = "empirical_null_unanimity_rescue"
    selected = pd.concat([baseline, rescue], ignore_index=True)
    selected.insert(1, "config_id", config.config_id)
    decision_rows: list[dict[str, Any]] = []
    all_assays = pd.MultiIndex.from_product(
        [FORMAL_TRAIN_SEEDS, sorted(candidates["modality"].unique())],
        names=["seed", "modality"],
    )
    selected_lookup = selected.set_index(["seed", "modality"])
    rescue_counts = pool_counts.to_dict()
    for seed, modality in all_assays:
        key = (seed, modality)
        if key in selected_lookup.index:
            row = selected_lookup.loc[key]
            decision_rows.append(
                {
                    "config_id": config.config_id,
                    "seed": seed,
                    "modality": modality,
                    "selection_status": "selected",
                    "selection_stage": str(row["selection_stage"]),
                    "selected_pair_key": str(row["pair_key"]),
                    "n_rescue_candidates": int(rescue_counts.get(key, 0)),
                    "abstention_reason": "",
                }
            )
        else:
            count = int(rescue_counts.get(key, 0))
            reason = (
                "ambiguous_rescue_candidates"
                if config.require_unique_rescue_candidate and count > 1
                else "no_candidate_passed_baseline_or_rescue"
            )
            decision_rows.append(
                {
                    "config_id": config.config_id,
                    "seed": seed,
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selection_stage": "none",
                    "selected_pair_key": "",
                    "n_rescue_candidates": count,
                    "abstention_reason": reason,
                }
            )
    return selected, pd.DataFrame(decision_rows)


def _load_roles_after_rule_freeze(world_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    roles: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for seed in FORMAL_TRAIN_SEEDS:
        if seed >= 37_001:
            raise AssertionError("Refusing future formal seeds")
        path = world_root / f"world_seed{seed}" / "scorer" / "role_assignments.tsv"
        frame = pd.read_csv(path, sep="\t")
        if not {"node", "role"}.issubset(frame.columns):
            raise ValueError(f"Role assignment schema is incomplete for seed {seed}")
        frame = frame[["node", "role"]].copy()
        frame.insert(0, "seed", seed)
        roles.append(frame)
        hashes[str(seed)] = _sha256(path)
    output = pd.concat(roles, ignore_index=True)
    if set(output["seed"].astype(int)) != set(FORMAL_TRAIN_SEEDS):
        raise ValueError("Role assignments contain an unexpected seed")
    if output.duplicated(["seed", "node"]).any():
        raise ValueError("Role assignments duplicate a node within a world")
    return output, hashes


def _label_candidates(candidates: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    lookup = roles.set_index(["seed", "node"])["role"]
    output = candidates.copy()
    mappings = (
        ("parent_role", "parent_candidate"),
        ("sibling_role", "noncollider_sibling_candidate"),
        ("other_parent_role", "other_parent_candidate"),
        ("center_role", "collider_center"),
    )
    for output_column, feature_column in mappings:
        output[output_column] = [
            str(lookup.loc[(int(row.seed), str(getattr(row, feature_column)))])
            for row in output.itertuples(index=False)
        ]
    output["exact_ps_pair"] = [
        {str(row.parent_role), str(row.sibling_role)} == {"P", "S"}
        for row in output.itertuples(index=False)
    ]
    output["pair_role_type"] = output["parent_role"] + "-" + output["sibling_role"]
    return output


def _score(
    labeled: pd.DataFrame,
    selections: pd.DataFrame,
    decisions: pd.DataFrame,
    configs: list[RescueConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labeled[
        ["seed", "modality", "pair_key", "exact_ps_pair", "pair_role_type"]
    ]
    scored = selections.merge(
        labels,
        on=["seed", "modality", "pair_key"],
        how="left",
        validate="many_to_one",
    )
    if scored["exact_ps_pair"].isna().any():
        raise ValueError("A prelabel selection lacks a role audit")
    scored["exact_ps_pair"] = scored["exact_ps_pair"].astype(bool)
    rows: list[dict[str, Any]] = []
    for config in configs:
        subset = scored.loc[scored["config_id"].eq(config.config_id)]
        tp = int(subset["exact_ps_pair"].sum())
        fp = int((~subset["exact_ps_pair"]).sum())
        rescue = subset.loc[subset["selection_stage"].eq(
            "empirical_null_unanimity_rescue"
        )]
        rows.append(
            {
                **asdict(config),
                "maximum_q90_pair_relative_uncertainty": (
                    config.maximum_q90_pair_relative_uncertainty
                    if np.isfinite(config.maximum_q90_pair_relative_uncertainty)
                    else "inf"
                ),
                "n_selected": len(subset),
                "true_positive": tp,
                "false_positive": fp,
                "precision": tp / len(subset) if len(subset) else 1.0,
                "recall": tp / 144.0,
                "n_rescued": len(rescue),
                "n_true_rescued": int(rescue["exact_ps_pair"].sum()),
                "n_false_rescued": int((~rescue["exact_ps_pair"]).sum()),
                "n_ambiguous_rescue_assays": int(
                    decisions.loc[
                        decisions["config_id"].eq(config.config_id),
                        "abstention_reason",
                    ].eq("ambiguous_rescue_candidates").sum()
                ),
            }
        )
    return scored, pd.DataFrame(rows)


def _abstention_audit(
    labeled: pd.DataFrame,
    scored: pd.DataFrame,
    formal_decisions: pd.DataFrame,
    config_id: str,
) -> pd.DataFrame:
    modalities = sorted(formal_decisions["modality"].unique())
    assays = pd.MultiIndex.from_product(
        [FORMAL_TRAIN_SEEDS, modalities], names=["seed", "modality"]
    ).to_frame(index=False)
    selected = scored.loc[scored["config_id"].eq(config_id), [
        "seed", "modality", "pair_key", "selection_stage", "exact_ps_pair"
    ]]
    exact = labeled.loc[labeled["exact_ps_pair"].astype(bool)].copy()
    exact = exact[
        [
            "seed",
            "modality",
            "pair_key",
            "passes_state_membership_gate",
            "passes_cosine_gate",
            "passes_relative_parent_margin_gate",
            "lower_absolute_loading_cosine",
            "relative_median_parent_margin",
            "parent_win_view_support",
            "empirical_null_support_p_le_0_125",
            "median_minimum_loading_stability",
            "view_q90__candidate_pair_max_relative_uncertainty",
        ]
    ].rename(columns={column: f"exact__{column}" for column in exact.columns if column not in {"seed", "modality"}})
    output = (
        assays.merge(formal_decisions, on=["seed", "modality"], how="left")
        .merge(selected, on=["seed", "modality"], how="left")
        .merge(exact, on=["seed", "modality"], how="left")
    )
    output["proposed_selected"] = output["pair_key"].notna()
    output["exact_candidate_available"] = output["exact__pair_key"].notna()
    reasons: list[str] = []
    for row in output.itertuples(index=False):
        if bool(row.proposed_selected):
            reasons.append("")
        elif not bool(row.exact_candidate_available):
            reasons.append("exact_ps_absent_from_structural_candidate_set")
        elif not bool(row.exact__passes_state_membership_gate):
            reasons.append("exact_ps_failed_formal_sibling_state_gate")
        else:
            failures: list[str] = []
            if float(row.exact__relative_median_parent_margin) < 0.10:
                failures.append("parent_margin")
            if float(row.exact__parent_win_view_support) < 1.0:
                failures.append("parent_win_unanimity")
            if float(row.exact__empirical_null_support_p_le_0_125) < 1.0:
                failures.append("empirical_null_unanimity")
            reasons.append("exact_ps_failed_rescue__" + ";".join(failures))
    output["postproposal_abstention_reason"] = reasons
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-a",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_36001_36024_loading_recall_diagnostic_v1/phase_a"
        ),
    )
    parser.add_argument(
        "--world-root",
        default="pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_36001_36024",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_36001_36024_loading_recall_diagnostic_v1/"
            "phase_b_empirical_null_rescue"
        ),
    )
    args = parser.parse_args()
    phase_a = Path(args.phase_a).resolve()
    world_root = Path(args.world_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if world_root.name != "train_36001_36024":
        raise ValueError("World root must be formal TRAIN 36001--36024")
    if any(token in str(world_root).casefold() for token in ("locked", "dev")):
        raise ValueError("DEV/LOCKED roots are forbidden")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    candidates, formal_decisions, phase_a_freeze = _load_phase_a(phase_a)
    configs = _configs()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        # Second barrier: construct and freeze every rule selection before labels.
        selections: list[pd.DataFrame] = []
        decisions: list[pd.DataFrame] = []
        for config in configs:
            selected, assay_decisions = _select(candidates, config)
            selections.append(selected)
            decisions.append(assay_decisions)
        all_selections = pd.concat(selections, ignore_index=True)
        all_decisions = pd.concat(decisions, ignore_index=True)
        selection_columns = [
            "config_id",
            "seed",
            "modality",
            "pair_key",
            "selection_stage",
        ]
        all_selections[selection_columns].to_csv(
            temporary / "prelabel_rule_selections.tsv", sep="\t", index=False
        )
        all_decisions.to_csv(
            temporary / "prelabel_rule_decisions.tsv", sep="\t", index=False
        )
        rule_freeze = {
            "schema_version": "ca_boss_v4.formal_train_loading_recall.rule_grid.v1",
            "formal_train_seeds": list(FORMAL_TRAIN_SEEDS),
            "phase_a_freeze_sha256": _sha256(phase_a / "global_phase_a_freeze.json"),
            "all_rule_selections_built_before_roles_open": True,
            "roles_read": False,
            "baseline": {
                "minimum_lower_absolute_loading_cosine": BASELINE_MINIMUM_LOWER_COSINE,
                "minimum_relative_parent_margin": BASELINE_MINIMUM_RELATIVE_PARENT_MARGIN,
            },
            "configs": [asdict(config) for config in configs],
            "prelabel_selection_sha256": _sha256(
                temporary / "prelabel_rule_selections.tsv"
            ),
            "prelabel_decision_sha256": _sha256(
                temporary / "prelabel_rule_decisions.tsv"
            ),
        }
        _write_json(temporary / "prelabel_rule_grid_freeze.json", rule_freeze)

        # First scorer-domain access in this process occurs here.
        roles, role_hashes = _load_roles_after_rule_freeze(world_root)
        labeled = _label_candidates(candidates, roles)
        scored, sensitivity = _score(labeled, all_selections, all_decisions, configs)
        proposed_id = "proposed_empirical_null_unanimity"
        proposed = scored.loc[scored["config_id"].eq(proposed_id)].copy()
        baseline = proposed.loc[proposed["selection_stage"].eq("formal_baseline")]
        if len(baseline) != 114 or not baseline["exact_ps_pair"].astype(bool).all():
            raise ValueError("Diagnostic did not reproduce the formal 114/144 loading result")
        if len(proposed) != 116 or not proposed["exact_ps_pair"].astype(bool).all():
            raise ValueError("Proposed rescue did not reproduce 116 TP and zero FP")
        labeled.to_csv(temporary / "labeled_candidates.tsv", sep="\t", index=False)
        scored.to_csv(temporary / "all_scored_selections.tsv", sep="\t", index=False)
        sensitivity.to_csv(temporary / "rule_sensitivity.tsv", sep="\t", index=False)
        proposed.to_csv(temporary / "proposed_selected_groups.tsv", sep="\t", index=False)
        rescued = proposed.loc[
            proposed["selection_stage"].eq("empirical_null_unanimity_rescue")
        ].copy()
        rescued.to_csv(temporary / "proposed_rescued_groups.tsv", sep="\t", index=False)
        abstentions = _abstention_audit(
            labeled, scored, formal_decisions, proposed_id
        )
        abstentions.to_csv(
            temporary / "proposed_assay_audit.tsv", sep="\t", index=False
        )
        per_world = (
            proposed.groupby("seed", sort=True)["exact_ps_pair"]
            .agg(true_positive="sum", n_selected="size")
            .reindex(FORMAL_TRAIN_SEEDS, fill_value=0)
            .reset_index()
        )
        per_world["false_positive"] = per_world["n_selected"] - per_world["true_positive"]
        per_world["precision"] = np.where(
            per_world["n_selected"].gt(0),
            per_world["true_positive"] / per_world["n_selected"],
            1.0,
        )
        per_world["recall"] = per_world["true_positive"] / 6.0
        per_world["n_abstained"] = 6 - per_world["n_selected"]
        per_world.to_csv(temporary / "proposed_per_world_metrics.tsv", sep="\t", index=False)

        false_in_baseline_abstained = labeled.loc[
            ~labeled["exact_ps_pair"].astype(bool)
            & labeled["passes_state_membership_gate"].astype(bool)
            & labeled["relative_median_parent_margin"].ge(0.10)
            & ~pd.MultiIndex.from_frame(labeled[["seed", "modality"]]).isin(
                pd.MultiIndex.from_frame(baseline[["seed", "modality"]])
            )
        ]
        proposed_metrics = sensitivity.loc[
            sensitivity["config_id"].eq(proposed_id)
        ].iloc[0]
        summary = {
            "schema_version": "ca_boss_v4.formal_train_loading_recall.score.v1",
            "formal_train_seeds": list(FORMAL_TRAIN_SEEDS),
            "strict_phase_a_before_role_barrier": True,
            "future_seeds_37001_or_above_read": False,
            "dev_or_locked_read": False,
            "formal_baseline": {
                "true_positive": 114,
                "false_positive": 0,
                "precision": 1.0,
                "recall": 114.0 / 144.0,
                "n_abstained": 30,
            },
            "proposed_rule": {
                "name": "abstention_only_empirical_null_unanimity_rescue",
                "baseline_is_never_replaced": True,
                "rescue_only_when_baseline_abstains": True,
                "sibling_must_pass_existing_formal_state_gate": True,
                "minimum_relative_parent_margin": 0.10,
                "parent_must_win_all_six_loading_views": True,
                "loading_empirical_null_p_at_most_0_125_in_all_six_views": True,
                "rescue_candidate_must_be_unique_within_assay": True,
                "fixed_absolute_cosine_threshold_in_rescue": False,
                "direct_parent_membership_used_as_gate": False,
            },
            "proposed_metrics": {
                "true_positive": int(proposed_metrics["true_positive"]),
                "false_positive": int(proposed_metrics["false_positive"]),
                "precision": float(proposed_metrics["precision"]),
                "recall": float(proposed_metrics["recall"]),
                "n_rescued": int(proposed_metrics["n_rescued"]),
                "n_abstained": 144 - int(proposed_metrics["n_selected"]),
                "recall_gain_groups": int(proposed_metrics["true_positive"]) - 114,
            },
            "abstention_decomposition_before_rescue": {
                "exact_ps_absent_from_structural_candidates": 4,
                "exact_ps_failed_formal_state_gate": 5,
                "exact_ps_passed_state_but_failed_loading_geometry": 21,
            },
            "observable_separation": {
                "rescued_empirical_null_view_support_minimum": float(
                    rescued["empirical_null_support_p_le_0_125"].min()
                ),
                "rescued_relative_parent_margin_minimum": float(
                    rescued["relative_median_parent_margin"].min()
                ),
                "rescued_parent_win_support_minimum": float(
                    rescued["parent_win_view_support"].min()
                ),
                "false_candidate_empirical_null_view_support_maximum": float(
                    false_in_baseline_abstained[
                        "empirical_null_support_p_le_0_125"
                    ].max()
                ),
            },
            "role_assignment_sha256": role_hashes,
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
                "schema_version": "ca_boss_v4.formal_train_loading_recall.run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "phase_a_freeze_sha256": _sha256(
                    phase_a / "global_phase_a_freeze.json"
                ),
                "prelabel_rule_grid_freeze_sha256": _sha256(
                    temporary / "prelabel_rule_grid_freeze.json"
                ),
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
