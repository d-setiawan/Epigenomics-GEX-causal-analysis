#!/usr/bin/env python3
"""Observed-only near-boundary parent-swap audit over retired TRAIN blocks.

Phase A freezes alternative stars for only observed multicenter/completion
decisions, with frozen state and beta evidence.  Phase B opens retired truth
only after the candidate and formula manifests verify.  This diagnostic makes
no production changes and never reads DEV/LOCKED paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import audit_retired_direct_guard_rules_v4 as retired
import diagnose_formal_train_direct_failures_v4 as diagnostic
from observed_collider_structure import apply_observed_collider_refinements


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
SWAP_ACTIONS = {"resolve_competing_multicenter_path", "complete_dual_supported_unshielded_anchor"}
SCHEMA = "ca_boss_v4.retired_parent_swap_decoder.v1"
FORMULAS = ("production_only", "abstain_state_conflict", "swap_strict_state", "swap_comparable_beta")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _verify_decoder(path: Path) -> pd.DataFrame:
    manifest = json.loads((path / "phase_a_manifest.json").read_text())
    if manifest.get("phase") != "observed_only_before_truth" or manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Decoder source lacks a valid Phase-A barrier")
    for name, expected in manifest["output_sha256"].items():
        if _sha(path / name) != expected:
            raise ValueError(f"Decoder Phase-A drift: {name}")
    return pd.read_csv(path / "phase_a_star_candidates.tsv", sep="\t")


def _edge_sets(frame: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {}
    for row in frame.itertuples(index=False):
        center, left, right = str(row.collider_center), str(row.node1), str(row.node2)
        parent = right if center == left else left if center == right else None
        if parent is None:
            raise ValueError("Center not an endpoint")
        result.setdefault(str(row.source_assay), set()).add((parent, center))
    return result


def _structural_centers(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Observed repair proxy: stars containing a structural-consensus arm."""
    result: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        if str(getattr(row, "orientation_source", "")) != "observed_structural_view_consensus":
            continue
        center = str(row.collider_center)
        result.add((str(row.source_assay), center))
    return result


def _state_map(world: dict[str, Any]) -> dict[str, dict[str, float]]:
    f = world["subspace_features"]
    return {
        str(row.feature): {
            "wald_support": float(row.conditional_wald_view_support),
            "partial_fit": float(row.oof_partial_state_fit),
            "direction_stability": float(row.median_conditional_score_direction_stability),
        }
        for row in f[["feature", "conditional_wald_view_support", "oof_partial_state_fit", "median_conditional_score_direction_stability"]].itertuples(index=False)
    }


def _rows_for_world(block: str, world: dict[str, Any], candidates: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, set[tuple[str, str]]]]:
    seed = int(world["seed"])
    production = _edge_sets(world["current_direct"])
    structural_centers = _structural_centers(world["current_direct"])
    state = _state_map(world)
    alternatives: list[dict[str, Any]] = []
    swaps: dict[str, set[tuple[str, str]]] = {}
    phase = candidates.loc[candidates["seed"].eq(seed)]
    decision_index = 0
    for assay, assay_edges in production.items():
        for center in sorted({child for _, child in assay_edges}):
            if (assay, center) not in structural_centers:
                continue
            chosen = {parent for parent, child in assay_edges if child == center}
            if len(chosen) != 2:
                continue
            base = phase.loc[
                phase["source_assay"].astype(str).eq(assay)
                & phase["center"].astype(str).eq(center)
            ]
            chosen_rows = base.loc[
                base.apply(lambda r: {str(r.outer1), str(r.outer2)} == chosen, axis=1)
            ]
            if len(chosen_rows) != 1:
                continue
            chosen_row = chosen_rows.iloc[0]
            for _, alternative in base.iterrows():
                outer = {str(alternative.outer1), str(alternative.outer2)}
                common = chosen & outer
                if len(common) != 1 or outer == chosen:
                    continue
                old_parent = next(iter(chosen - common))
                new_parent = next(iter(outer - common))
                old_state, new_state = state[old_parent], state[new_parent]
                record = {
                    "block": block, "seed": seed, "decision_index": decision_index,
                    "action": "observed_structural_repair_star_proxy", "source_assay": assay, "center": center,
                    "anchor_parent": next(iter(common)), "chosen_parent": old_parent, "alternative_parent": new_parent,
                    "chosen_wald_view_support": old_state["wald_support"], "chosen_partial_state_fit": old_state["partial_fit"],
                    "chosen_direction_stability": old_state["direction_stability"],
                    "alternative_wald_view_support": new_state["wald_support"], "alternative_partial_state_fit": new_state["partial_fit"],
                    "alternative_direction_stability": new_state["direction_stability"],
                    "chosen_max_arm_empirical_p": float(chosen_row.max_arm_empirical_p), "alternative_max_arm_empirical_p": float(alternative.max_arm_empirical_p),
                    "chosen_min_arm_lor": float(chosen_row.min_arm_lor), "alternative_min_arm_lor": float(alternative.min_arm_lor),
                    "alternative_beta_p_margin": float(chosen_row.max_arm_empirical_p - alternative.max_arm_empirical_p),
                    "alternative_lor_margin": float(alternative.min_arm_lor - chosen_row.min_arm_lor),
                    "alternative_min_vote_fraction": float(alternative.min_arm_vote_fraction),
                    "alternative_minimum_separation": float(alternative.minimum_collider_separation),
                }
                record["chosen_state_conflict"] = bool(old_state["wald_support"] <= 0.0 and old_state["partial_fit"] < 0.0)
                record["alternative_full_positive_state"] = bool(new_state["wald_support"] >= 1.0 and new_state["partial_fit"] > 0.0)
                record["strict_swap_gate"] = bool(record["chosen_state_conflict"] and record["alternative_full_positive_state"] and float(alternative.max_arm_empirical_p) <= 0.03 and float(alternative.min_arm_lor) >= float(chosen_row.min_arm_lor))
                record["comparable_beta_gate"] = bool(record["chosen_state_conflict"] and record["alternative_full_positive_state"] and (float(alternative.max_arm_empirical_p) <= 0.03 or float(alternative.min_arm_lor) > float(chosen_row.min_arm_lor)))
                alternatives.append(record)
            decision_index += 1
    table = pd.DataFrame(alternatives)
    for formula in FORMULAS:
        edges = set().union(*production.values()) if production else set()
        if formula != "production_only" and not table.empty:
            for key, group in table.groupby("decision_index", sort=True):
                gate = "strict_swap_gate" if formula == "swap_strict_state" else "comparable_beta_gate"
                passing = group.loc[group[gate]] if formula != "abstain_state_conflict" else group.loc[group["chosen_state_conflict"] & group["alternative_full_positive_state"]]
                if len(passing) != 1:
                    continue
                row = passing.iloc[0]
                old, new, center = (str(row.chosen_parent), str(row.center)), (str(row.alternative_parent), str(row.center)), str(row.center)
                if formula == "abstain_state_conflict":
                    edges.discard(old)
                else:
                    edges.discard(old); edges.add(new)
        swaps[formula] = edges
    return alternatives, swaps


def _metric(pred: set[tuple[str, str]], truth: set[tuple[str, str]]) -> dict[str, Any]:
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    reverse = sum(e not in truth and (e[1], e[0]) in truth for e in pred)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "strict_reverse": reverse, "precision": precision, "recall": recall, "reliable": bool(precision == precision and precision >= .99 and recall >= .90 and reverse == 0)}


def _verify_phase_a(out: Path) -> None:
    m = json.loads((out / "phase_a_manifest.json").read_text())
    if m.get("phase") != "observed_only_before_truth":
        raise ValueError("bad Phase A")
    for n, h in m["output_sha256"].items():
        if _sha(out / n) != h:
            raise ValueError(f"Phase-A drift: {n}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoder-phase-a", type=Path, default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/retired_exhaustive_state_star_decoder_audit_v3"))
    p.add_argument("--out-dir", type=Path, default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/retired_parent_swap_decoder_audit_v1"))
    a = p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError(a.out_dir)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    candidates = _verify_decoder(a.decoder_phase_a)
    held: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    frozen: dict[tuple[str, int, str], set[tuple[str, str]]] = {}
    for block in BLOCKS:
        held[block] = retired._load_block_worlds(block)
        for world in held[block]:
            observed, variants = _rows_for_world(block, world, candidates)
            rows.extend(observed)
            seed = int(world["seed"])
            for formula, edges in variants.items():
                frozen[(block, seed, formula)] = edges
                predictions.extend({"block": block, "seed": seed, "formula": formula, "parent": parent, "child": child} for parent, child in sorted(edges))
    alt = pd.DataFrame(rows)
    pred = pd.DataFrame(predictions)
    alt_path, pred_path = a.out_dir / "phase_a_parent_swap_alternatives.tsv", a.out_dir / "phase_a_formula_predictions.tsv"
    alt.to_csv(alt_path, sep="\t", index=False); pred.to_csv(pred_path, sep="\t", index=False)
    registry_path = a.out_dir / "phase_a_rule_registry.json"
    _json(registry_path, {"scope_actions": sorted(SWAP_ACTIONS), "scope_proxy": "complete production stars containing at least one observed_structural_view_consensus arm", "formulas": list(FORMULAS), "strict_rule": "chosen Wald support=0 and signed partial fit<0; unique alternative Wald support=1 and positive fit; alternative max beta p<=.03 and min-arm LOR no weaker", "comparable_rule": "same state conflict, but beta criterion is p<=.03 OR stronger min-arm LOR", "reliability": "per-world P>=.99, R>=.90, zero strict reversals"})
    phase = {"schema_version": SCHEMA, "phase": "observed_only_before_truth", "blocks": list(BLOCKS), "decoder_phase_a_manifest_sha256": _sha(a.decoder_phase_a / "phase_a_manifest.json"), "truth_or_scorer_opened": False, "dev_or_locked_opened": False, "n_alternative_rows": int(len(alt)), "output_sha256": {q.name: _sha(q) for q in (alt_path, pred_path, registry_path)}, "source_sha256": _sha(Path(__file__))}
    _json(a.out_dir / "phase_a_manifest.json", phase)
    _verify_phase_a(a.out_dir)
    truths: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for block in BLOCKS:
        _, sets = diagnostic._load_direct_truth_after_barrier(held[block], Path(retired.BLOCKS[block]["world_root"]))
        truths.update({(block, int(seed)): value for seed, value in sets.items()})
    metrics = pd.DataFrame([{"block": b, "seed": s, "formula": f, **_metric(edges, truths[(b, s)])} for (b, s, f), edges in frozen.items()])
    metric_path = a.out_dir / "phase_b_world_metrics.tsv"; metrics.to_csv(metric_path, sep="\t", index=False)
    summary = metrics.groupby(["block", "formula"], sort=True).agg(n_worlds=("seed", "size"), reliable_worlds=("reliable", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), strict_reverse=("strict_reverse", "sum")).reset_index()
    summary["precision"] = summary.tp / (summary.tp + summary.fp); summary["recall"] = summary.tp / (summary.tp + summary.fn)
    summary["meets_block_gate"] = summary.reliable_worlds.ge(22) & summary.precision.ge(.99) & summary.strict_reverse.eq(0)
    summary_path = a.out_dir / "phase_b_block_summary.tsv"; summary.to_csv(summary_path, sep="\t", index=False)
    qualify = summary.groupby("formula", sort=True).agg(all_blocks_qualify=("meets_block_gate", "all"), minimum_reliable=("reliable_worlds", "min"), minimum_precision=("precision", "min"), total_reverse=("strict_reverse", "sum")).reset_index()
    qualify_path = a.out_dir / "phase_b_qualification.tsv"; qualify.to_csv(qualify_path, sep="\t", index=False)
    loo_rows=[]
    for hold in BLOCKS:
        train=summary.loc[summary.block.ne(hold)].groupby("formula", sort=True).agg(reliable=("reliable_worlds", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), rev=("strict_reverse", "sum")).reset_index(); train["precision"]=train.tp/(train.tp+train.fp); train["recall"]=train.tp/(train.tp+train.fn)
        chosen=train.sort_values(["reliable","precision","recall","rev","formula"],ascending=[False,False,False,True,True]).iloc[0]
        held_row=summary.loc[(summary.block==hold)&(summary.formula==chosen.formula)].iloc[0].to_dict(); loo_rows.append({"held_out_block":hold,"selected_formula":chosen.formula,"training_reliable":int(chosen.reliable),"training_precision":float(chosen.precision),**held_row})
    loo=pd.DataFrame(loo_rows); loo_path=a.out_dir / "phase_b_leave_one_block_out.tsv"; loo.to_csv(loo_path,sep="\t",index=False)
    final={"schema_version":SCHEMA,"status":"posthoc_retired_train_diagnostic_not_production_certification","phase_a_verified_before_truth_access":True,"dev_or_locked_opened":False,"qualified_formulas":qualify.loc[qualify.all_blocks_qualify,"formula"].tolist()}
    final_path=a.out_dir / "summary.json"; _json(final_path,final)
    _json(a.out_dir / "manifest.json", {"phase_a_manifest_sha256":_sha(a.out_dir / "phase_a_manifest.json"),"output_sha256":{q.name:_sha(q) for q in (alt_path,pred_path,registry_path,metric_path,summary_path,qualify_path,loo_path,final_path)},"source_sha256":_sha(Path(__file__))})
    print(json.dumps(final,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
