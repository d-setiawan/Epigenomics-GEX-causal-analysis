#!/usr/bin/env python3
"""Phase-separated retired-TRAIN hybrid direct-decoder audit.

The audit combines replayed observed-only production calls with the frozen
Phase-A candidate universe of ``retired_exhaustive_state_star_decoder_audit``.
It freezes observable risk flags and a small hybrid family before any direct
truth file is opened, then evaluates the frozen family on retired blocks.

This is diagnostic evidence only: no production modules are imported for
modification, no worlds are generated, and DEV/LOCKED paths are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import audit_retired_direct_guard_rules_v4 as retired
import diagnose_formal_train_direct_failures_v4 as diagnostic
from observed_collider_structure import apply_observed_collider_refinements


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
SCHEMA = "ca_boss_v4.retired_hybrid_direct_decoder.v1"
DECODER_NAME = "beta_rank"
FORMULAS = {
    "production_only": frozenset(),
    "hybrid_incomplete_beta": frozenset({"incomplete_or_non_two_arm_motif"}),
    "hybrid_guard_failure_beta": frozenset({"conservative_repair_guard_failure"}),
    "hybrid_incomplete_or_guard_beta": frozenset(
        {"incomplete_or_non_two_arm_motif", "conservative_repair_guard_failure"}
    ),
    "hybrid_any_risk_beta": frozenset(
        {
            "incomplete_or_non_two_arm_motif",
            "predicted_parent_not_formal_state_confirmed",
            "conservative_repair_guard_failure",
        }
    ),
    "hybrid_any_risk_abstain": frozenset(
        {
            "incomplete_or_non_two_arm_motif",
            "predicted_parent_not_formal_state_confirmed",
            "conservative_repair_guard_failure",
        }
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _verify_decoder_phase_a(path: Path) -> pd.DataFrame:
    manifest = json.loads((path / "phase_a_manifest.json").read_text())
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Decoder input lacks an observed-only Phase-A barrier")
    if manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Decoder Phase A reports scorer access")
    for name, expected in manifest["output_sha256"].items():
        if _sha256(path / name) != expected:
            raise ValueError(f"Decoder Phase-A drift: {name}")
    return pd.read_csv(path / "phase_a_star_candidates.tsv", sep="\t")


def _decoder_stars(candidates: pd.DataFrame) -> dict[tuple[int, str], set[tuple[str, str]]]:
    """Freeze the pre-registered high-precision beta ranker from Phase-A fields."""

    eligible = candidates.loc[
        candidates["min_arm_vote_fraction"].ge(5.0 / 6.0)
        & candidates["max_arm_empirical_p"].le(0.02)
        & candidates["min_arm_lor"].ge(1.0)
    ].copy()
    eligible["rank_score"] = (
        eligible["min_arm_lor"]
        + eligible["median_collider_separation"]
        + 0.25 * eligible["min_arm_vs_outer_lor"]
    )
    result: dict[tuple[int, str], set[tuple[str, str]]] = {}
    for (seed, assay), group in eligible.groupby(["seed", "source_assay"], sort=True):
        winner = group.sort_values(
            ["rank_score", "minimum_collider_separation", "center", "outer1", "outer2"],
            ascending=[False, False, True, True, True],
        ).iloc[0]
        result[(int(seed), str(assay))] = {
            (str(winner.outer1), str(winner.center)),
            (str(winner.outer2), str(winner.center)),
        }
    return result


def _state_selected(world: dict[str, Any]) -> dict[str, bool]:
    frame = world["subspace_features"]
    return {
        str(row.feature): str(row.selection_status) == "selected"
        for row in frame[["feature", "selection_status"]].itertuples(index=False)
    }


def _production_edges(frame: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    output: dict[str, set[tuple[str, str]]] = {}
    if frame.empty:
        return output
    for row in frame.itertuples(index=False):
        center = str(row.collider_center)
        left, right = str(row.node1), str(row.node2)
        if center == left:
            parent = right
        elif center == right:
            parent = left
        else:
            raise ValueError("Production direct candidate has no endpoint center")
        output.setdefault(str(row.source_assay), set()).add((parent, center))
    return output


def _guard_failure_rows(world: dict[str, Any], refinement: Any) -> dict[str, bool]:
    """Observable conservative guard, fixed before labels in this audit."""

    state = _state_selected(world)
    result: dict[str, bool] = {}
    for _, decision in refinement.decisions.iterrows():
        values = retired._decision_features(decision, refinement, world["subspace_features"])
        action = str(decision.action)
        failure = False
        if action == "complete_dual_supported_unshielded_anchor":
            failure = (
                not bool(values["candidate_new_parent_formal_state_selected"])
                and float(values["selected_arm_outer_state_support_floor"]) <= 0.0
            )
        elif action == "complete_unique_rescue_only_state_loading_contrast":
            failure = float(values["anchor_base_view_count"]) < 1.0
        elif action == "resolve_competing_multicenter_path":
            failure = (
                float(values["median_collider_separation_score"]) <= 0.0
                and not (
                    int(values["winner_view_count"]) == 6
                    and float(values["median_outer_state_support_floor"]) > 0.0025
                )
            )
        assay = str(decision.source_assay)
        result[assay] = bool(result.get(assay, False) or failure)
    return result


def _phase_a_world(
    block: str,
    world: dict[str, Any],
    decoder: dict[tuple[int, str], set[tuple[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[str, str]]]]:
    seed = int(world["seed"])
    motif, refinement = retired._selector_inputs(world)
    applied = apply_observed_collider_refinements(motif, refinement)
    production = _production_edges(applied.direct_candidates)
    state = _state_selected(world)
    guard_failure = _guard_failure_rows(world, refinement)
    all_assays = sorted({str(value) for value in world["cell_features"]["modality"].unique() if str(value).startswith("chrom_")})
    risk_rows: list[dict[str, Any]] = []
    formula_edges: dict[str, set[tuple[str, str]]] = {name: set() for name in FORMULAS}
    for assay in all_assays:
        production_assay = production.get(assay, set())
        centers = {child for _, child in production_assay}
        incomplete = len(production_assay) != 2 or len(centers) != 1
        unconfirmed = any(not state.get(parent, False) for parent, _ in production_assay)
        failed_guard = bool(guard_failure.get(assay, False))
        high_confidence = not incomplete and not unconfirmed and not failed_guard
        triggers = {
            "incomplete_or_non_two_arm_motif": incomplete,
            "predicted_parent_not_formal_state_confirmed": unconfirmed,
            "conservative_repair_guard_failure": failed_guard,
        }
        decoder_assay = decoder.get((seed, assay), set())
        risk_rows.append(
            {
                "block": block,
                "seed": seed,
                "source_assay": assay,
                **triggers,
                "any_risk": any(triggers.values()),
                "complete_high_confidence_production": high_confidence,
                "n_production_direct_edges": len(production_assay),
                "n_production_centers": len(centers),
                "n_decoder_direct_edges": len(decoder_assay),
            }
        )
        for formula, replacement_triggers in FORMULAS.items():
            use_hybrid = any(triggers[name] for name in replacement_triggers)
            if formula == "production_only" or not use_hybrid:
                selected = production_assay
            elif formula == "hybrid_any_risk_abstain":
                selected = set()
            else:
                selected = decoder_assay
            formula_edges[formula].update(selected)
    return risk_rows, formula_edges


def _verify_phase_a(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "phase_a_manifest.json").read_text())
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Hybrid Phase A has wrong phase")
    for name, expected in manifest["output_sha256"].items():
        if _sha256(path / name) != expected:
            raise ValueError(f"Hybrid Phase-A drift: {name}")
    return manifest


def _metric(predicted: set[tuple[str, str]], truth: set[tuple[str, str]]) -> dict[str, Any]:
    tp = len(predicted & truth)
    strict_reverse = sum(edge not in truth and (edge[1], edge[0]) in truth for edge in predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else 0.0
    reliable = bool(precision == precision and precision >= 0.99 and recall >= 0.90 and strict_reverse == 0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "strict_reverse": strict_reverse,
        "precision": precision,
        "recall": recall,
        "reliable": reliable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder-phase-a",
        type=Path,
        default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/retired_exhaustive_state_star_decoder_audit_v3"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/retired_hybrid_direct_decoder_audit_v1"),
    )
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    # Phase A: decoder candidate loading and complete hybrid prediction freeze.
    candidates = _verify_decoder_phase_a(args.decoder_phase_a)
    decoder = _decoder_stars(candidates)
    held: dict[str, list[dict[str, Any]]] = {}
    risk_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    frozen_edges: dict[tuple[str, int, str], set[tuple[str, str]]] = {}
    for block in BLOCKS:
        worlds = retired._load_block_worlds(block)
        held[block] = worlds
        for world in worlds:
            local_risk, edge_sets = _phase_a_world(block, world, decoder)
            risk_rows.extend(local_risk)
            seed = int(world["seed"])
            for formula, edges in edge_sets.items():
                frozen_edges[(block, seed, formula)] = edges
                for parent, child in sorted(edges):
                    prediction_rows.append(
                        {"block": block, "seed": seed, "formula": formula, "parent": parent, "child": child}
                    )
    risks = pd.DataFrame(risk_rows).sort_values(["block", "seed", "source_assay"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(["block", "seed", "formula", "parent", "child"]).reset_index(drop=True)
    risk_path = args.out_dir / "phase_a_risk_triggers.tsv"
    prediction_path = args.out_dir / "phase_a_hybrid_predictions.tsv"
    registry_path = args.out_dir / "phase_a_formula_registry.json"
    risks.to_csv(risk_path, sep="\t", index=False)
    predictions.to_csv(prediction_path, sep="\t", index=False)
    _write_json(registry_path, {
        "decoder": DECODER_NAME,
        "risk_triggers": {
            "incomplete_or_non_two_arm_motif": "production assay has not exactly two calls to one center",
            "predicted_parent_not_formal_state_confirmed": "any production parent has frozen selection_status other than selected",
            "conservative_repair_guard_failure": "fixed observed completion/rescue/multicenter abstention guard fails",
        },
        "formulas": {name: sorted(triggers) for name, triggers in FORMULAS.items()},
        "reliability_rule": "per-world direct precision >=0.99, recall >=0.90, and zero strict reversals",
    })
    phase_a = {
        "schema_version": SCHEMA,
        "phase": "observed_only_before_truth",
        "blocks": list(BLOCKS),
        "decoder_phase_a_manifest_sha256": _sha256(args.decoder_phase_a / "phase_a_manifest.json"),
        "n_worlds": sum(len(worlds) for worlds in held.values()),
        "n_risk_rows": int(len(risks)),
        "n_prediction_rows": int(len(predictions)),
        "truth_or_scorer_opened": False,
        "dev_or_locked_opened": False,
        "output_sha256": {path.name: _sha256(path) for path in (risk_path, prediction_path, registry_path)},
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.out_dir / "phase_a_manifest.json", phase_a)

    # Phase B: verify all observed artifacts first, then load retired direct truth.
    _verify_phase_a(args.out_dir)
    world_truth: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for block in BLOCKS:
        direct_truth, truth_sets = diagnostic._load_direct_truth_after_barrier(
            held[block], Path(retired.BLOCKS[block]["world_root"])
        )
        del direct_truth
        world_truth.update({(block, int(seed)): edges for seed, edges in truth_sets.items()})
    metric_rows: list[dict[str, Any]] = []
    for (block, seed, formula), predicted in sorted(frozen_edges.items()):
        high_conf = risks.loc[
            risks["block"].eq(block) & risks["seed"].eq(seed) & risks["complete_high_confidence_production"]
        ]
        production_edges = frozen_edges[(block, seed, "production_only")]
        # High-confidence assays are unchanged iff no production edge with a high-confidence child is altered.
        high_assays = set(high_conf["source_assay"].astype(str))
        assay_map = {str(row.feature): str(row.modality) for row in held[block][seed - int(held[block][0]["seed"])]["cell_features"][["feature", "modality"]].drop_duplicates().itertuples(index=False)}
        unchanged = all(
            ((parent, child) in predicted) == ((parent, child) in production_edges)
            for parent, child in production_edges | predicted
            if assay_map.get(child, "") in high_assays
        )
        metric_rows.append({
            "block": block,
            "seed": seed,
            "formula": formula,
            "high_confidence_calls_unchanged": unchanged,
            **_metric(predicted, world_truth[(block, seed)]),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics_path = args.out_dir / "phase_b_world_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary = (
        metrics.groupby(["block", "formula"], sort=True)
        .agg(
            n_worlds=("seed", "size"),
            reliable_worlds=("reliable", "sum"),
            total_tp=("tp", "sum"), total_fp=("fp", "sum"), total_fn=("fn", "sum"),
            total_strict_reverse=("strict_reverse", "sum"),
            all_high_confidence_calls_unchanged=("high_confidence_calls_unchanged", "all"),
        ).reset_index()
    )
    summary["direct_precision"] = summary["total_tp"] / (summary["total_tp"] + summary["total_fp"])
    summary["direct_recall"] = summary["total_tp"] / (summary["total_tp"] + summary["total_fn"])
    summary["meets_block_gate"] = (
        summary["reliable_worlds"].ge(22)
        & summary["direct_precision"].ge(0.99)
        & summary["total_strict_reverse"].eq(0)
        & summary["all_high_confidence_calls_unchanged"]
    )
    summary_path = args.out_dir / "phase_b_block_formula_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    qualification = summary.groupby("formula", sort=True).agg(
        all_retired_blocks_meet_gate=("meets_block_gate", "all"),
        minimum_reliable_worlds=("reliable_worlds", "min"),
        minimum_direct_precision=("direct_precision", "min"),
        total_strict_reverse=("total_strict_reverse", "sum"),
        all_high_confidence_calls_unchanged=("all_high_confidence_calls_unchanged", "all"),
    ).reset_index()
    qualification_path = args.out_dir / "phase_b_qualification.tsv"
    qualification.to_csv(qualification_path, sep="\t", index=False)

    loo_rows: list[dict[str, Any]] = []
    for held_block in BLOCKS:
        train = summary.loc[summary["block"].ne(held_block)].groupby("formula", sort=True).agg(
            reliable_worlds=("reliable_worlds", "sum"), total_tp=("total_tp", "sum"), total_fp=("total_fp", "sum"), total_fn=("total_fn", "sum"), total_reverse=("total_strict_reverse", "sum")
        ).reset_index()
        train["precision"] = train["total_tp"] / (train["total_tp"] + train["total_fp"])
        train["recall"] = train["total_tp"] / (train["total_tp"] + train["total_fn"])
        choice = train.sort_values(["reliable_worlds", "precision", "recall", "total_reverse", "formula"], ascending=[False, False, False, True, True]).iloc[0]
        held_rows = summary.loc[(summary["block"].eq(held_block)) & (summary["formula"].eq(choice.formula))]
        loo_rows.append({"held_out_block": held_block, "selected_formula": choice.formula, "training_reliable_worlds": int(choice.reliable_worlds), "training_precision": float(choice.precision), "training_recall": float(choice.recall), **held_rows.iloc[0].to_dict()})
    loo = pd.DataFrame(loo_rows)
    loo_path = args.out_dir / "phase_b_leave_one_block_out.tsv"
    loo.to_csv(loo_path, sep="\t", index=False)

    final = {
        "schema_version": SCHEMA,
        "status": "posthoc_retired_train_diagnostic_not_production_certification",
        "phase_a_verified_before_truth_access": True,
        "dev_or_locked_opened": False,
        "qualification_formula_count": int(qualification["all_retired_blocks_meet_gate"].sum()),
        "qualified_formulas": qualification.loc[qualification["all_retired_blocks_meet_gate"], "formula"].tolist(),
    }
    final_path = args.out_dir / "summary.json"
    _write_json(final_path, final)
    _write_json(args.out_dir / "manifest.json", {
        "phase_a_manifest_sha256": _sha256(args.out_dir / "phase_a_manifest.json"),
        "output_sha256": {path.name: _sha256(path) for path in (risk_path, prediction_path, registry_path, metrics_path, summary_path, qualification_path, loo_path, final_path)},
        "source_sha256": _sha256(Path(__file__)),
    })
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
