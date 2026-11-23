#!/usr/bin/env python3
"""TRAIN-only audit of the structure-anchored state-reporter gate.

This diagnostic deliberately uses only legacy TRAIN seeds 33001--33024.  It
loads every pretruth structural candidate before opening the previously
frozen TRAIN role audit, freezes all threshold-grid selections, and only then
scores exact P--S recovery.  It must never be pointed at the untouched formal
TRAIN block or at DEV/LOCKED data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGACY_TRAIN_SEEDS = tuple(range(33001, 33025))
COSINE_GRID = (0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90)
RELATIVE_PARENT_MARGIN_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
FIXED_COSINE = 0.85
FIXED_RELATIVE_PARENT_MARGIN = 0.10
FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "role",
    "scorer",
    "oracle",
    "generative",
    "counterfactual",
)


@dataclass(frozen=True)
class GateConfig:
    minimum_lower_absolute_loading_cosine: float
    minimum_relative_median_parent_margin: float
    minimum_best_coherence_margin: float = 0.0

    @property
    def config_id(self) -> str:
        return (
            f"cosine_{self.minimum_lower_absolute_loading_cosine:.3f}__"
            f"parent_margin_{self.minimum_relative_median_parent_margin:.3f}__"
            f"best_margin_{self.minimum_best_coherence_margin:.3f}"
        ).replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_pretruth_candidates(path: Path, seed: int) -> pd.DataFrame:
    if seed not in LEGACY_TRAIN_SEEDS:
        raise ValueError(f"Refusing non-legacy-TRAIN seed {seed}")
    frame = pd.read_csv(path, sep="\t")
    forbidden = sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Pretruth candidates contain privileged columns: {forbidden}")
    required = {
        "pair_key",
        "modality",
        "collider_center",
        "parent_candidate",
        "other_parent_candidate",
        "nonmotif_sibling_candidate",
        "median_coherence_group_score",
        "lower_absolute_loading_cosine",
        "relative_median_parent_margin",
        "sibling_formal_state_selected",
        "parent_membership_used_as_gate",
        "center_excluded",
        "parent_parent_pair_forbidden",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pretruth candidates are missing columns: {missing}")
    if len(frame) != 36 or frame["modality"].nunique() != 6:
        raise ValueError(f"Seed {seed} must contain six candidates in each of six assays")
    if frame["pair_key"].duplicated().any():
        raise ValueError(f"Seed {seed} contains duplicate pair keys")
    if frame["parent_membership_used_as_gate"].astype(bool).any():
        raise ValueError("Direct-parent membership was used as a gate")
    if not frame["center_excluded"].astype(bool).all():
        raise ValueError("Collider centers were not excluded")
    if not frame["parent_parent_pair_forbidden"].astype(bool).all():
        raise ValueError("A direct-parent--direct-parent pair was admitted")
    output = frame.copy()
    output.insert(0, "seed", seed)
    return output


def _select(frame: pd.DataFrame, config: GateConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = frame.copy()
    evidence["passes_sibling_formal_state_membership"] = evidence[
        "sibling_formal_state_selected"
    ].astype(bool)
    evidence["passes_lower_loading_cosine"] = (
        evidence["lower_absolute_loading_cosine"]
        >= config.minimum_lower_absolute_loading_cosine
    )
    evidence["passes_relative_parent_margin"] = (
        evidence["relative_median_parent_margin"]
        >= config.minimum_relative_median_parent_margin
    )
    evidence["passes_pre_rank_gates"] = evidence[
        [
            "passes_sibling_formal_state_membership",
            "passes_lower_loading_cosine",
            "passes_relative_parent_margin",
        ]
    ].all(axis=1)
    evidence["selected"] = False
    evidence["relative_best_coherence_margin"] = 0.0
    decisions: list[dict[str, Any]] = []
    for modality, indices in evidence.groupby("modality", sort=True).groups.items():
        eligible = evidence.loc[indices].loc[
            evidence.loc[indices, "passes_pre_rank_gates"]
        ].sort_values(
            ["median_coherence_group_score", "pair_key"],
            ascending=[False, True],
        )
        if eligible.empty:
            decisions.append(
                {
                    "seed": int(evidence.loc[indices, "seed"].iloc[0]),
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selected_pair_key": "",
                    "relative_best_coherence_margin": 0.0,
                    "abstention_reason": "no_candidate_passed_state_cosine_parent_margin",
                }
            )
            continue
        best_index = eligible.index[0]
        best_score = float(eligible.iloc[0]["median_coherence_group_score"])
        runner_score = (
            float(eligible.iloc[1]["median_coherence_group_score"])
            if len(eligible) > 1
            else 0.0
        )
        margin = max((best_score - runner_score) / max(best_score, 1e-12), 0.0)
        evidence.loc[best_index, "relative_best_coherence_margin"] = margin
        if margin < config.minimum_best_coherence_margin:
            decisions.append(
                {
                    "seed": int(evidence.loc[best_index, "seed"]),
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selected_pair_key": "",
                    "relative_best_coherence_margin": margin,
                    "abstention_reason": "ambiguous_best_coherence_margin",
                }
            )
            continue
        evidence.loc[best_index, "selected"] = True
        decisions.append(
            {
                "seed": int(evidence.loc[best_index, "seed"]),
                "modality": modality,
                "selection_status": "selected",
                "selected_pair_key": str(evidence.loc[best_index, "pair_key"]),
                "relative_best_coherence_margin": margin,
                "abstention_reason": "",
            }
        )
    selected = evidence.loc[evidence["selected"]].copy()
    selected.insert(1, "config_id", config.config_id)
    decisions_frame = pd.DataFrame(decisions)
    decisions_frame.insert(1, "config_id", config.config_id)
    return selected, decisions_frame


def _configs() -> list[GateConfig]:
    configs = [
        GateConfig(cosine, margin)
        for cosine in COSINE_GRID
        for margin in RELATIVE_PARENT_MARGIN_GRID
    ]
    configs.append(
        GateConfig(
            FIXED_COSINE,
            FIXED_RELATIVE_PARENT_MARGIN,
            minimum_best_coherence_margin=0.10,
        )
    )
    unique = {config.config_id: config for config in configs}
    return [unique[key] for key in sorted(unique)]


def _phase_a(
    candidate_root: Path,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, GateConfig], dict[str, str]]:
    configs = _configs()
    by_config = {config.config_id: config for config in configs}
    worlds: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    # The global barrier: load every world before constructing any score table.
    for seed in LEGACY_TRAIN_SEEDS:
        path = candidate_root / f"world_seed{seed}" / "structural_loading_candidates.tsv"
        hashes[str(seed)] = _sha256(path)
        worlds.append(_read_pretruth_candidates(path, seed))
    selected_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    for config in configs:
        for world in worlds:
            selected, decisions = _select(world, config)
            selected_frames.append(selected)
            decision_frames.append(decisions)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    decisions_all = pd.concat(decision_frames, ignore_index=True)
    selected_all.to_csv(output / "phase_a_selected_pretruth.tsv", sep="\t", index=False)
    decisions_all.to_csv(output / "phase_a_assay_decisions.tsv", sep="\t", index=False)
    freeze = {
        "schema_version": "ca_boss_v4.structure_state_reporter_gate.phase_a.v1",
        "legacy_train_seeds": list(LEGACY_TRAIN_SEEDS),
        "all_24_worlds_loaded_before_any_roles_open": True,
        "roles_or_scorer_read": False,
        "cross_assay_matching_used": False,
        "direct_parent_membership_used_as_gate": False,
        "n_configs": len(configs),
        "configs": [asdict(config) | {"config_id": config.config_id} for config in configs],
        "candidate_sha256": hashes,
    }
    _write_json(output / "phase_a_global_freeze.json", freeze)
    return selected_all, decisions_all, by_config, hashes


def _score(
    selected: pd.DataFrame,
    decisions: pd.DataFrame,
    configs: dict[str, GateConfig],
    labels_path: Path,
    output: Path,
) -> dict[str, Any]:
    # This is the first role-bearing file opened by this process.
    labels = pd.read_csv(labels_path, sep="\t")
    if set(labels["seed"].astype(int)) != set(LEGACY_TRAIN_SEEDS):
        raise ValueError("Role audit must contain exactly legacy TRAIN seeds 33001--33024")
    label_columns = [
        "seed",
        "pair_key",
        "modality",
        "parent_role",
        "sibling_role",
        "exact_ps_pair",
        "false_pair_type",
    ]
    scored = selected.merge(
        labels[label_columns],
        on=["seed", "pair_key", "modality"],
        how="left",
        validate="many_to_one",
    )
    if scored["exact_ps_pair"].isna().any():
        raise ValueError("A frozen selection was absent from the TRAIN role audit")
    scored["exact_ps_pair"] = scored["exact_ps_pair"].astype(bool)
    scored.to_csv(output / "phase_b_selected_role_audit.tsv", sep="\t", index=False)

    aggregate_rows: list[dict[str, Any]] = []
    per_world_rows: list[dict[str, Any]] = []
    for config_id, config in configs.items():
        subset = scored.loc[scored["config_id"].eq(config_id)]
        true_positive = int(subset["exact_ps_pair"].sum())
        false_positive = int((~subset["exact_ps_pair"]).sum())
        aggregate_rows.append(
            {
                "config_id": config_id,
                **asdict(config),
                "n_selected": len(subset),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "micro_precision": (
                    true_positive / len(subset) if len(subset) else 1.0
                ),
                "exact_ps_recall": true_positive / 144.0,
            }
        )
        for seed in LEGACY_TRAIN_SEEDS:
            world = subset.loc[subset["seed"].eq(seed)]
            tp = int(world["exact_ps_pair"].sum())
            fp = int((~world["exact_ps_pair"]).sum())
            per_world_rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "true_positive": tp,
                    "false_positive": fp,
                    "n_selected": len(world),
                    "precision": tp / len(world) if len(world) else 1.0,
                    "recall": tp / 6.0,
                    "n_abstained": 6 - len(world),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        [
            "minimum_best_coherence_margin",
            "minimum_lower_absolute_loading_cosine",
            "minimum_relative_median_parent_margin",
        ]
    )
    per_world = pd.DataFrame(per_world_rows)
    aggregate.to_csv(output / "threshold_sensitivity.tsv", sep="\t", index=False)
    per_world.to_csv(output / "per_world_metrics.tsv", sep="\t", index=False)

    fixed_config = GateConfig(FIXED_COSINE, FIXED_RELATIVE_PARENT_MARGIN)
    fixed_id = fixed_config.config_id
    fixed = scored.loc[scored["config_id"].eq(fixed_id)].copy()
    fixed.to_csv(output / "fixed_selected_groups.tsv", sep="\t", index=False)
    per_world.loc[per_world["config_id"].eq(fixed_id)].to_csv(
        output / "fixed_per_world_metrics.tsv", sep="\t", index=False
    )
    exact = labels.loc[labels["exact_ps_pair"].astype(bool)].copy()
    if len(exact) != 144:
        raise ValueError("Expected exactly one P--S candidate in each of 144 assays")
    fixed_keys = set(zip(fixed["seed"].astype(int), fixed["pair_key"].astype(str)))
    exact["selected"] = [
        (int(row.seed), str(row.pair_key)) in fixed_keys for row in exact.itertuples()
    ]
    abstained = exact.loc[~exact["selected"]].copy()
    abstained["fails_sibling_state_membership"] = ~abstained[
        "sibling_formal_state_selected"
    ].astype(bool)
    abstained["fails_lower_cosine"] = (
        abstained["lower_absolute_loading_cosine"] < FIXED_COSINE
    )
    abstained["fails_relative_parent_margin"] = (
        abstained["relative_median_parent_margin"] < FIXED_RELATIVE_PARENT_MARGIN
    )
    abstained["failed_gates"] = abstained.apply(
        lambda row: ";".join(
            name
            for name, failed in (
                ("sibling_state_membership", row["fails_sibling_state_membership"]),
                ("lower_cosine", row["fails_lower_cosine"]),
                ("relative_parent_margin", row["fails_relative_parent_margin"]),
            )
            if bool(failed)
        ),
        axis=1,
    )
    abstained.to_csv(output / "fixed_abstentions.tsv", sep="\t", index=False)

    fixed_aggregate = aggregate.loc[aggregate["config_id"].eq(fixed_id)].iloc[0]
    requested_margin_config = GateConfig(
        FIXED_COSINE,
        FIXED_RELATIVE_PARENT_MARGIN,
        minimum_best_coherence_margin=0.10,
    )
    requested_margin = aggregate.loc[
        aggregate["config_id"].eq(requested_margin_config.config_id)
    ].iloc[0]
    exact_s = exact
    state_count = int(exact_s["sibling_formal_state_selected"].astype(bool).sum())
    feature_max_t_count = int(
        exact_s["sibling_feature_global_maxT_confirmation_gate"].astype(bool).sum()
    )
    assay_max_t_count = int(
        exact_s["sibling_assay_global_maxT_admission_gate"].astype(bool).sum()
    )
    summary = {
        "schema_version": "ca_boss_v4.structure_state_reporter_gate.train_score.v1",
        "train_only": True,
        "legacy_train_seeds": list(LEGACY_TRAIN_SEEDS),
        "all_phase_a_selections_frozen_before_roles_open": True,
        "fixed_config": asdict(fixed_config) | {"config_id": fixed_id},
        "fixed_metrics": {
            "n_selected": int(fixed_aggregate["n_selected"]),
            "true_positive": int(fixed_aggregate["true_positive"]),
            "false_positive": int(fixed_aggregate["false_positive"]),
            "micro_precision": float(fixed_aggregate["micro_precision"]),
            "exact_ps_recall": float(fixed_aggregate["exact_ps_recall"]),
            "n_abstained": len(abstained),
        },
        "fixed_plus_best_coherence_margin_0_10_metrics": {
            "n_selected": int(requested_margin["n_selected"]),
            "true_positive": int(requested_margin["true_positive"]),
            "false_positive": int(requested_margin["false_positive"]),
            "micro_precision": float(requested_margin["micro_precision"]),
            "exact_ps_recall": float(requested_margin["exact_ps_recall"]),
        },
        "state_membership_audit": {
            "exact_s_selected": state_count,
            "exact_s_feature_global_maxT_confirmed": feature_max_t_count,
            "exact_s_assay_global_maxT_admitted": assay_max_t_count,
            "exact_s_not_selected": 144 - state_count,
        },
        "role_audit_sha256": _sha256(labels_path),
        "semantics": {
            "output_relation": "within_assay_shared_state_support",
            "direct_causal_claimed": False,
            "cross_assay_pair_claimed": False,
            "cross_assay_direction_claimed": False,
        },
    }
    _write_json(output / "aggregate.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "structure_anchored_loading_train_diagnostic_v2/phase_a_candidates"
        ),
    )
    parser.add_argument(
        "--role-audit",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "structure_anchored_loading_train_diagnostic_v2/train_labels/"
            "labeled_structural_loading_candidates.tsv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "structure_state_reporter_gate_train_diagnostic_v1"
        ),
    )
    args = parser.parse_args()
    candidate_root = Path(args.candidate_root)
    labels_path = Path(args.role_audit)
    out_dir = Path(args.out_dir)
    if "330" not in str(candidate_root) and "structure_anchored_loading_train_diagnostic_v2" not in str(candidate_root):
        raise ValueError("Candidate root is not the pinned legacy TRAIN diagnostic")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent)
    )
    try:
        selected, decisions, configs, _ = _phase_a(candidate_root, temporary)
        summary = _score(selected, decisions, configs, labels_path, temporary)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.structure_state_reporter_gate.run.v1",
                "script_sha256": _sha256(Path(__file__)),
                "phase_a_freeze_sha256": _sha256(
                    temporary / "phase_a_global_freeze.json"
                ),
                "aggregate_sha256": _sha256(temporary / "aggregate.json"),
                "fixed_metrics": summary["fixed_metrics"],
            },
        )
        os.replace(temporary, out_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
