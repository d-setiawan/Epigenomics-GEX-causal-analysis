#!/usr/bin/env python3
"""Replay the posthoc raw-evidence collider replacement gate on retired TRAIN.

This diagnostic consumes only the frozen observed-only candidate table while
making replacement decisions.  Retired truth is opened only after those
decisions have been serialized and hashed.  The three-tier thresholds were
assembled after retired labels were available, so the result is development
evidence for a future untouched block rather than certification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rescore_retired_crossfit_hierarchical_collider_v1 import (
    _aggregate,
    _edges_from_stars,
    _score_edges,
    _truth_edges,
)


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
SCHEMA = "ca_boss_v4.retired_raw_evidence_collider_gate.v1"
KEYS = ("block", "seed", "source_assay")
STAR_KEYS = (*KEYS, "center", "outer1", "outer2")
COUNTEREXAMPLE_ASSAYS = {
    ("TRAIN360", 36011, "chrom_H3K4me3"),
    ("TRAIN360", 36016, "chrom_H3K4me1"),
    ("TRAIN370", 37019, "chrom_H3K4me2"),
    ("TRAIN370", 37019, "chrom_H3K9me3"),
    ("TRAIN380", 38011, "chrom_H3K27ac"),
    ("TRAIN380", 38011, "chrom_H3K27me3"),
    ("TRAIN380", 38020, "chrom_H3K27me3"),
    ("TRAIN380", 38020, "chrom_H3K9me3"),
}

THRESHOLDS: dict[str, dict[str, float]] = {
    "tier1": {
        "minimum_parent_positive_posterior": 0.98,
        "minimum_joint_gain_exclusive": 0.005,
        "minimum_drop_one_gain_exclusive": 0.0,
        "minimum_both_positive_fold_support": 0.80,
        "minimum_reverse_center_gain_margin_exclusive": 0.0,
        "maximum_arm_empirical_positive_p": 0.125,
        "minimum_eligible_runner_score_margin": 0.003,
        "minimum_exact_production_score_improvement": 0.003,
        "minimum_nonexact_score": 0.040,
    },
    "tier2": {
        "minimum_parent_positive_posterior": 0.90,
        "minimum_joint_gain": 0.009,
        "minimum_drop_one_gain": 0.00025,
        "minimum_both_positive_fold_support": 1.0,
        "minimum_reverse_center_gain_margin": 0.0001,
        "maximum_arm_empirical_positive_p": 0.125,
        "minimum_outer_empirical_null_compatibility_p": 0.125,
        "minimum_collider_separation": 0.040,
        "minimum_score": 0.040,
        "minimum_all_candidate_score_margin": 0.010,
    },
    "tier3": {
        "minimum_parent_positive_posterior": 0.90,
        "minimum_joint_gain": 0.005,
        "minimum_drop_one_gain": -0.0001,
        "minimum_both_positive_fold_support": 0.80,
        "maximum_arm_empirical_positive_p": 0.125,
        "minimum_outer_empirical_null_compatibility_p": 0.125,
        "maximum_joint_gain_deficit_from_production": 0.0001,
        "maximum_score_deficit_from_production": 0.001,
        "production_minimum_arm_empirical_positive_p_exclusive": 0.125,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _verify_phase_a(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "phase_a_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Observed-only Phase-A barrier is absent")
    if manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Phase A reports truth/scorer access")
    if manifest.get("dev_or_locked_opened") is not False:
        raise ValueError("Phase A reports DEV/LOCKED access")
    for name, expected in manifest["output_sha256"].items():
        if _sha256(input_dir / name) != expected:
            raise ValueError(f"Frozen Phase-A drift: {name}")
    return {
        "phase_a_manifest_sha256": _sha256(path),
        "phase_a_candidate_sha256": manifest["output_sha256"][
            "phase_a_crossfit_star_candidates.tsv"
        ],
        "phase_a_source_sha256": manifest["source_sha256"],
    }


def _score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["crossfit_joint_gain_over_baseline"]
        + frame["crossfit_min_drop_one_gain"]
        + 0.02 * frame["laplace_min_parent_positive_posterior"]
        + 0.25 * frame["residual_star_separation"]
    )


def _edge_set(row: Any) -> set[tuple[str, str]]:
    return {(str(row.outer1), str(row.center)), (str(row.outer2), str(row.center))}


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["raw_generative_score", "center", "outer1", "outer2"],
        ascending=[False, True, True, True],
    )


def _select(
    candidates: pd.DataFrame,
    production_edges: pd.DataFrame,
    production_status: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select replacements without consulting any truth-labelled column."""
    if "truth_exact_star" in candidates:
        raise ValueError("Selection must consume the observed-only Phase-A table")
    frame = candidates.copy()
    frame["raw_generative_score"] = _score(frame)
    status = {
        (str(row.block), int(row.seed), str(row.source_assay)): str(row.risk_reasons)
        if pd.notna(row.risk_reasons)
        else ""
        for row in production_status.itertuples(index=False)
    }
    production = {
        (str(block), int(seed), str(assay)): {
            (str(row.parent), str(row.child)) for row in group.itertuples(index=False)
        }
        for (block, seed, assay), group in production_edges.groupby(list(KEYS), sort=True)
    }

    selected: list[pd.Series] = []
    audit: list[dict[str, Any]] = []
    for key_value, group in frame.groupby(list(KEYS), sort=True):
        key = (str(key_value[0]), int(key_value[1]), str(key_value[2]))
        group = group.copy()
        prod_edges = production.get(key, set())
        production_matches = group.loc[
            [_edge_set(row) == prod_edges for row in group.itertuples(index=False)]
        ]
        production_exact = bool(len(prod_edges) == 2 and len(production_matches) == 1)
        production_row = production_matches.iloc[0] if production_exact else None
        chosen: pd.Series | None = None
        tier = "retain_production"
        runner_margin = math.nan
        candidate_vs_production = math.nan

        t1 = thresholds["tier1"]
        eligible1 = _sorted(
            group.loc[
                group["laplace_min_parent_positive_posterior"].ge(
                    t1["minimum_parent_positive_posterior"]
                )
                & group["crossfit_joint_gain_over_baseline"].gt(
                    t1["minimum_joint_gain_exclusive"]
                )
                & group["crossfit_min_drop_one_gain"].gt(
                    t1["minimum_drop_one_gain_exclusive"]
                )
                & group["laplace_both_positive_fold_support"].ge(
                    t1["minimum_both_positive_fold_support"]
                )
                & group["crossfit_joint_gain_over_baseline_reverse_center_margin"].gt(
                    t1["minimum_reverse_center_gain_margin_exclusive"]
                )
                & group["maximum_arm_empirical_positive_p"].le(
                    t1["maximum_arm_empirical_positive_p"]
                )
            ]
        )
        if len(eligible1):
            top = eligible1.iloc[0]
            runner_margin = float(
                top.raw_generative_score - eligible1.iloc[1].raw_generative_score
            ) if len(eligible1) > 1 else math.inf
            unique = runner_margin >= t1["minimum_eligible_runner_score_margin"]
            if production_exact:
                candidate_vs_production = float(
                    top.raw_generative_score - production_row.raw_generative_score
                )
                if (
                    unique
                    and _edge_set(top) != prod_edges
                    and bool(top.two_parent_positive_state_wald_consistency)
                    and candidate_vs_production
                    >= t1["minimum_exact_production_score_improvement"]
                ):
                    chosen, tier = top, "tier1_exact_production_disagreement"
            elif (
                unique
                and top.raw_generative_score >= t1["minimum_nonexact_score"]
                and (
                    bool(top.two_parent_positive_state_wald_consistency)
                    or top.minimum_collider_separation > 0.0
                )
            ):
                chosen, tier = top, "tier1_nonexact_production_completion"

        t2 = thresholds["tier2"]
        eligible2 = pd.DataFrame()
        if chosen is None and not production_exact:
            eligible2 = _sorted(
                group.loc[
                    group["laplace_min_parent_positive_posterior"].ge(
                        t2["minimum_parent_positive_posterior"]
                    )
                    & group["crossfit_joint_gain_over_baseline"].ge(
                        t2["minimum_joint_gain"]
                    )
                    & group["crossfit_min_drop_one_gain"].ge(
                        t2["minimum_drop_one_gain"]
                    )
                    & group["laplace_both_positive_fold_support"].ge(
                        t2["minimum_both_positive_fold_support"]
                    )
                    & group["crossfit_joint_gain_over_baseline_reverse_center_margin"].ge(
                        t2["minimum_reverse_center_gain_margin"]
                    )
                    & group["maximum_arm_empirical_positive_p"].le(
                        t2["maximum_arm_empirical_positive_p"]
                    )
                    & group["outer_empirical_null_compatibility_p"].ge(
                        t2["minimum_outer_empirical_null_compatibility_p"]
                    )
                    & group["minimum_collider_separation"].ge(
                        t2["minimum_collider_separation"]
                    )
                    & group["raw_generative_score"].ge(t2["minimum_score"])
                ]
            )
            if len(eligible2):
                top = eligible2.iloc[0]
                other_max = group.drop(index=top.name)["raw_generative_score"].max()
                runner_margin = float(top.raw_generative_score - other_max)
                if runner_margin >= t2["minimum_all_candidate_score_margin"]:
                    chosen, tier = top, "tier2_positive_collider_incomplete_rescue"

        t3 = thresholds["tier3"]
        eligible3 = pd.DataFrame()
        reasons = status.get(key, "")
        if (
            chosen is None
            and production_exact
            and "production_parent_redundancy_demoted" in reasons
            and not bool(production_row.two_parent_positive_state_wald_consistency)
            and production_row.maximum_arm_empirical_positive_p
            > t3["production_minimum_arm_empirical_positive_p_exclusive"]
        ):
            prod_parents = {str(production_row.outer1), str(production_row.outer2)}
            same_center_one_parent = group.apply(
                lambda row: str(row.center) == str(production_row.center)
                and len({str(row.outer1), str(row.outer2)} & prod_parents) == 1,
                axis=1,
            )
            eligible3 = group.loc[
                same_center_one_parent
                & group["two_parent_positive_state_wald_consistency"].astype(bool)
                & group["laplace_min_parent_positive_posterior"].ge(
                    t3["minimum_parent_positive_posterior"]
                )
                & group["crossfit_joint_gain_over_baseline"].ge(
                    t3["minimum_joint_gain"]
                )
                & group["crossfit_min_drop_one_gain"].ge(
                    t3["minimum_drop_one_gain"]
                )
                & group["laplace_both_positive_fold_support"].ge(
                    t3["minimum_both_positive_fold_support"]
                )
                & group["maximum_arm_empirical_positive_p"].le(
                    t3["maximum_arm_empirical_positive_p"]
                )
                & group["outer_empirical_null_compatibility_p"].ge(
                    t3["minimum_outer_empirical_null_compatibility_p"]
                )
                & group["crossfit_joint_gain_over_baseline"].ge(
                    production_row.crossfit_joint_gain_over_baseline
                    - t3["maximum_joint_gain_deficit_from_production"]
                )
                & group["raw_generative_score"].ge(
                    production_row.raw_generative_score
                    - t3["maximum_score_deficit_from_production"]
                )
            ]
            if len(eligible3) == 1:
                chosen, tier = eligible3.iloc[0], "tier3_state_consistent_one_parent_swap"
                candidate_vs_production = float(
                    chosen.raw_generative_score - production_row.raw_generative_score
                )

        audit.append(
            {
                "block": key[0],
                "seed": key[1],
                "source_assay": key[2],
                "production_exact_two_arm_star": production_exact,
                "production_risk_reasons": reasons,
                "tier1_eligible_candidates": len(eligible1),
                "tier2_eligible_candidates": len(eligible2),
                "tier3_eligible_candidates": len(eligible3),
                "decision": tier,
                "selected": chosen is not None,
                "selected_center": str(chosen.center) if chosen is not None else "",
                "selected_outer1": str(chosen.outer1) if chosen is not None else "",
                "selected_outer2": str(chosen.outer2) if chosen is not None else "",
                "selected_raw_generative_score": float(chosen.raw_generative_score)
                if chosen is not None
                else math.nan,
                "selection_runner_margin": runner_margin,
                "selected_score_minus_production_score": candidate_vs_production,
            }
        )
        if chosen is not None:
            chosen = chosen.copy()
            chosen["selection_tier"] = tier
            chosen["production_exact_two_arm_star"] = production_exact
            chosen["selection_runner_margin"] = runner_margin
            chosen["selected_score_minus_production_score"] = candidate_vs_production
            selected.append(chosen)
    return pd.DataFrame(selected), pd.DataFrame(audit)


def _apply_replacements(production: pd.DataFrame, replacements: pd.DataFrame) -> pd.DataFrame:
    replacement_edges = _edges_from_stars(replacements)
    replaced = set(
        map(
            tuple,
            replacement_edges[list(KEYS)].drop_duplicates().itertuples(
                index=False, name=None
            ),
        )
    )
    retained = production.loc[
        [
            (str(row.block), int(row.seed), str(row.source_assay)) not in replaced
            for row in production.itertuples(index=False)
        ]
    ]
    return pd.concat([retained, replacement_edges], ignore_index=True, sort=False)


def _sensitivity_specs() -> list[tuple[str, str, float]]:
    return [
        ("tier1", "minimum_parent_positive_posterior", value)
        for value in (0.975, 0.980, 0.985)
    ] + [
        ("tier1", "minimum_joint_gain_exclusive", value)
        for value in (0.004, 0.005, 0.006)
    ] + [
        ("tier1", "minimum_eligible_runner_score_margin", value)
        for value in (0.002, 0.003, 0.0035)
    ] + [
        ("tier2", "minimum_parent_positive_posterior", value)
        for value in (0.90, 0.915, 0.93)
    ] + [
        ("tier2", "minimum_joint_gain", value)
        for value in (0.009, 0.0095, 0.010)
    ] + [
        ("tier2", "minimum_drop_one_gain", value)
        for value in (0.00025, 0.000275, 0.00030)
    ] + [
        ("tier2", "minimum_collider_separation", value)
        for value in (0.040, 0.0425, 0.045)
    ] + [
        ("tier2", "minimum_all_candidate_score_margin", value)
        for value in (0.010, 0.0125, 0.015)
    ] + [
        ("tier3", "minimum_parent_positive_posterior", value)
        for value in (0.90, 0.92, 0.94)
    ] + [
        ("tier3", "minimum_drop_one_gain", value)
        for value in (-0.00010, -0.00008, -0.00006)
    ] + [
        ("tier3", "maximum_joint_gain_deficit_from_production", value)
        for value in (0.00005, 0.000075, 0.00010)
    ] + [
        ("tier3", "maximum_score_deficit_from_production", value)
        for value in (0.0005, 0.00075, 0.001)
    ] + [
        ("tier3", "minimum_outer_empirical_null_compatibility_p", value)
        for value in (0.125, 0.15625, 0.1875)
    ]


def _sensitivity(
    candidates: pd.DataFrame,
    production: pd.DataFrame,
    status: pd.DataFrame,
    truth: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    exact = labels.set_index(list(STAR_KEYS))["truth_exact_star"].astype(bool)
    rows: list[dict[str, Any]] = []
    for index, (tier, field, value) in enumerate(_sensitivity_specs()):
        thresholds = copy.deepcopy(THRESHOLDS)
        thresholds[tier][field] = value
        replacements, _ = _select(candidates, production, status, thresholds)
        replacement_exact = [
            bool(exact.get(tuple(row), False))
            for row in replacements[list(STAR_KEYS)].itertuples(index=False, name=None)
        ]
        world = _score_edges(
            f"one_at_a_time_{index:02d}",
            _apply_replacements(production, replacements),
            truth,
        )
        reliable = world.groupby("block")["direct_reliable_precision99"].sum()
        rows.append(
            {
                "sensitivity_id": f"one_at_a_time_{index:02d}",
                "tier": tier,
                "field": field,
                "value": value,
                "n_replacements": len(replacements),
                "n_exact_replacements": int(sum(replacement_exact)),
                "n_false_replacements": int(len(replacement_exact) - sum(replacement_exact)),
                "replacement_identity": ";".join(
                    f"{row.block}:{int(row.seed)}:{row.source_assay}"
                    for row in replacements.itertuples(index=False)
                ),
                **{
                    f"{block}_reliable_worlds": int(reliable.get(block, 0))
                    for block in BLOCKS
                },
                "all_blocks_meet_22_of_24": bool(
                    all(int(reliable.get(block, 0)) >= 22 for block in BLOCKS)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_crossfit_hierarchical_collider_audit_v1"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_raw_evidence_collider_gate_audit_v1_complete"
        ),
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    input_identity = _verify_phase_a(args.input_dir)

    candidates = pd.read_csv(
        args.input_dir / "phase_a_crossfit_star_candidates.tsv", sep="\t"
    )
    production = pd.read_csv(
        args.input_dir / "phase_a_production_direct_edges.tsv", sep="\t"
    )
    status = pd.read_csv(
        args.input_dir / "phase_a_production_assay_status.tsv", sep="\t"
    )
    replacements, assay_audit = _select(candidates, production, status, THRESHOLDS)
    replacements = replacements.sort_values(list(STAR_KEYS)).reset_index(drop=True)
    assay_audit = assay_audit.sort_values(list(KEYS)).reset_index(drop=True)
    replacement_path = args.out_dir / "observed_only_selected_replacements.tsv"
    assay_path = args.out_dir / "observed_only_assay_decisions.tsv"
    threshold_path = args.out_dir / "posthoc_threshold_specification.json"
    replacements.to_csv(replacement_path, sep="\t", index=False)
    assay_audit.to_csv(assay_path, sep="\t", index=False)
    _write_json(threshold_path, THRESHOLDS)
    observed_decision_hash = _sha256(replacement_path)

    # Retired truth is consulted only below this serialized decision barrier.
    labels = pd.read_csv(
        args.input_dir / "phase_b_labeled_crossfit_star_candidates.tsv", sep="\t"
    )
    truth = _truth_edges(labels)
    exact_lookup = labels.set_index(list(STAR_KEYS))["truth_exact_star"].astype(bool)
    replacements["truth_exact_star"] = [
        bool(exact_lookup.get(tuple(row), False))
        for row in replacements[list(STAR_KEYS)].itertuples(index=False, name=None)
    ]
    evaluated_replacement_path = args.out_dir / "evaluated_selected_replacements.tsv"
    replacements.to_csv(evaluated_replacement_path, sep="\t", index=False)
    predicted = _apply_replacements(production, replacements)
    world = _score_edges("posthoc_raw_evidence_three_tier_gate", predicted, truth)
    aggregate = _aggregate(world)
    world_path = args.out_dir / "world_direct_metrics.tsv"
    aggregate_path = args.out_dir / "decoder_metrics.tsv"
    world.to_csv(world_path, sep="\t", index=False)
    aggregate.to_csv(aggregate_path, sep="\t", index=False)

    counterexample_mask = [
        (str(row.block), int(row.seed), str(row.source_assay)) in COUNTEREXAMPLE_ASSAYS
        for row in candidates.itertuples(index=False)
    ]
    counterexamples = candidates.loc[counterexample_mask].copy()
    counterexamples["raw_generative_score"] = _score(counterexamples)
    counterexamples = counterexamples.merge(
        labels[list(STAR_KEYS) + ["truth_exact_star"]],
        on=list(STAR_KEYS),
        how="left",
        validate="one_to_one",
    ).sort_values(
        [*KEYS, "raw_generative_score"], ascending=[True, True, True, False]
    )
    counterexample_path = args.out_dir / "counterexample_candidate_audit.tsv"
    counterexamples.to_csv(counterexample_path, sep="\t", index=False)

    sensitivity = _sensitivity(candidates, production, status, truth, labels)
    sensitivity_path = args.out_dir / "one_at_a_time_threshold_sensitivity.tsv"
    sensitivity.to_csv(sensitivity_path, sep="\t", index=False)

    by_block = aggregate.loc[aggregate["scope"].isin(BLOCKS)].set_index("scope")
    all_row = aggregate.loc[aggregate["scope"].eq("ALL_RETIRED")].iloc[0]
    summary = {
        "schema_version": SCHEMA,
        "status": "posthoc_retired_development_not_production_certification",
        "thresholds_assembled_after_retired_labels": True,
        "selection_consumed_observed_only_phase_a_table": True,
        "observed_decisions_serialized_before_truth_read": True,
        "dev_or_locked_opened": False,
        "n_replacements": len(replacements),
        "n_exact_replacements": int(replacements["truth_exact_star"].sum()),
        "n_false_replacements": int((~replacements["truth_exact_star"]).sum()),
        "block_reliable_worlds_precision99": {
            block: int(by_block.loc[block, "reliable_worlds_precision99"])
            for block in BLOCKS
        },
        "all_blocks_meet_22_of_24": bool(
            all(int(by_block.loc[block, "reliable_worlds_precision99"]) >= 22 for block in BLOCKS)
        ),
        "all_tp": int(all_row["tp"]),
        "all_fp": int(all_row["fp"]),
        "all_fn": int(all_row["fn"]),
        "all_strict_reversals": int(all_row["strict_reverse"]),
        "all_precision": float(all_row["precision"]),
        "all_recall": float(all_row["recall"]),
        "sensitivity_runs": len(sensitivity),
        "sensitivity_runs_with_false_replacement": int(
            sensitivity["n_false_replacements"].gt(0).sum()
        ),
        "sensitivity_runs_meeting_all_block_gate": int(
            sensitivity["all_blocks_meet_22_of_24"].sum()
        ),
        "observed_decision_sha256_before_truth": observed_decision_hash,
        "input_identity": input_identity,
        "limitation": (
            "Each rescue-specific tier has few positive retired events; local "
            "threshold stability is not independent validation. Freeze before TRAIN390."
        ),
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    artifacts = [
        replacement_path,
        evaluated_replacement_path,
        assay_path,
        threshold_path,
        world_path,
        aggregate_path,
        counterexample_path,
        sensitivity_path,
        summary_path,
    ]
    manifest = {
        "schema_version": SCHEMA,
        "source_sha256": _sha256(Path(__file__)),
        "input_identity": input_identity,
        "output_sha256": {path.name: _sha256(path) for path in artifacts},
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
