#!/usr/bin/env python3
"""Exhaustive observed-only star-decoder audit over retired TRAIN blocks.

Phase A enumerates every within-assay ``outer1 -> center <- outer2`` triple
whose three pairs occur in the six frozen beta and residual views.  It records
only discovery artifacts (beta, residual, frozen state, nested-graph and
loading evidence).  Phase B is deliberately unreachable until the Phase-A
manifest has been written and re-verified; only then are retired scorer truth
files opened to label candidate stars and evaluate fixed simple decoders.

This is a diagnostic, not a production selector.  It never reads DEV/LOCKED,
never generates worlds, and refuses to overwrite an existing output folder.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import audit_retired_direct_guard_rules_v4 as retired
import diagnose_formal_train_direct_failures_v4 as diagnostic
from observed_collider_structure import conditional_outer_correlation


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
TARGET_SEEDS = {38011, 38012, 38014, 38020}
OUT_SCHEMA = "ca_boss_v4.retired_exhaustive_state_star_decoder.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(float(value))]


def _pair_maps(frame: pd.DataFrame, column: str) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in frame[["node1", "node2", column]].itertuples(index=False):
        key = _edge(str(row[0]), str(row[1]))
        if key in result:
            raise ValueError(f"Duplicate pair in observed view: {key}")
        result[key] = float(row[2])
    return result


def _base_counts(nested_edges: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for _, view in nested_edges.groupby("view_id", sort=True):
        selected = view.copy()
        if "selected" in selected:
            selected = selected.loc[selected["selected"].astype(bool)]
        if "edge_type" in selected:
            selected = selected.loc[
                selected["edge_type"].astype(str).eq("within_assay_direct_candidate")
            ]
        for row in selected[["node1", "node2"]].itertuples(index=False):
            key = _edge(str(row.node1), str(row.node2))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _state_views(cell_features: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for view_id, view in cell_features.groupby("view_id", sort=True):
        values: dict[str, float] = {}
        for row in view[["feature", "oof_partial_state_fit", "loading_direction_stability"]].itertuples(index=False):
            values[str(row.feature)] = max(float(row.oof_partial_state_fit), 0.0) * max(
                float(row.loading_direction_stability), 0.0
            )
        result[str(view_id)] = values
    return result


def _current_arm_counts(current_direct: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    if current_direct.empty:
        return counts
    for row in current_direct[["node1", "node2"]].itertuples(index=False):
        key = _edge(str(row.node1), str(row.node2))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _aggregate_beta(beta_votes: pd.DataFrame) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    output: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for assay, frame in beta_votes.groupby("source_assay", sort=True):
        assay_map: dict[tuple[str, str], dict[str, float]] = {}
        for key, group in frame.groupby(["node1", "node2"], sort=True):
            pair = _edge(str(key[0]), str(key[1]))
            assay_map[pair] = {
                "vote_fraction": float(group["selected"].astype(bool).mean()),
                "mean_empirical_p": float(group["empirical_p_positive"].mean()),
                "mean_lor": float(group["pooled_log_odds"].mean()),
                "n_votes": float(group["vote"].nunique()),
            }
        output[str(assay)] = assay_map
    return output


def _state_membership(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame["feature"].astype(str).duplicated().any():
        raise ValueError("Frozen state membership contains duplicate feature rows")
    return {
        str(row.feature): {
            "selected": str(row.selection_status) == "selected",
            "maxT_p": float(row.feature_global_maxT_p),
            "partial_fit": float(row.oof_partial_state_fit),
        }
        for row in frame[["feature", "selection_status", "feature_global_maxT_p", "oof_partial_state_fit"]].itertuples(index=False)
    }


def _phase_a_world(block: str, world: dict[str, Any]) -> list[dict[str, Any]]:
    """Build every assay-local triple from observed inputs only."""

    seed = int(world["seed"])
    beta = _aggregate_beta(world["beta_votes"])
    base = _base_counts(world["nested_edges"])
    current = _current_arm_counts(world["current_direct"])
    states = _state_membership(world["subspace_features"])
    residual_maps = {
        str(view_id): _pair_maps(view, "residual_r")
        for view_id, view in world["raw_pairs"].groupby("view_id", sort=True)
    }
    state_views = _state_views(world["cell_features"])
    loading_p = {
        str(view_id): _pair_maps(view, "loading_empirical_null_p")
        for view_id, view in world["cell_pairs"].groupby("view_id", sort=True)
    }
    loading_cosine = {
        str(view_id): _pair_maps(view, "absolute_loading_cosine")
        for view_id, view in world["cell_pairs"].groupby("view_id", sort=True)
    }
    if set(residual_maps) != set(state_views) or set(residual_maps) != set(loading_p):
        raise ValueError(f"View mismatch in seed {seed}")

    rows: list[dict[str, Any]] = []
    for assay, evidence in beta.items():
        nodes = sorted({node for edge in evidence for node in edge})
        for trio in itertools.combinations(nodes, 3):
            for center in trio:
                outer1, outer2 = sorted(node for node in trio if node != center)
                arm1, arm2, outer = _edge(center, outer1), _edge(center, outer2), _edge(outer1, outer2)
                if not all(edge in evidence for edge in (arm1, arm2, outer)):
                    continue
                arm_info = (evidence[arm1], evidence[arm2])
                view_separation: list[float] = []
                view_opening: list[float] = []
                view_state_floor: list[float] = []
                arm_loading_p: list[float] = []
                arm_loading_cosine: list[float] = []
                for view_id in sorted(residual_maps):
                    residual = residual_maps[view_id]
                    if not all(edge in residual for edge in (arm1, arm2, outer)):
                        continue
                    conditional = conditional_outer_correlation(
                        residual[outer], residual[arm1], residual[arm2]
                    )
                    view_separation.append(-conditional)
                    view_opening.append(residual[outer] - conditional)
                    view_state_floor.append(min(state_views[view_id].get(outer1, 0.0), state_views[view_id].get(outer2, 0.0)))
                    for arm in (arm1, arm2):
                        if arm in loading_p[view_id]:
                            arm_loading_p.append(loading_p[view_id][arm])
                            arm_loading_cosine.append(loading_cosine[view_id][arm])
                if len(view_separation) != 6:
                    raise ValueError(f"Expected six observed residual views for seed {seed}")
                values = [info["mean_lor"] for info in arm_info]
                vote_fractions = [info["vote_fraction"] for info in arm_info]
                empirical_p = [info["mean_empirical_p"] for info in arm_info]
                base_support = [base.get(arm, 0) for arm in (arm1, arm2)]
                current_support = [current.get(arm, 0) for arm in (arm1, arm2)]
                rescue_support = [
                    int(info["vote_fraction"] >= 5.0 / 6.0 and info["mean_empirical_p"] <= 0.02 and info["mean_lor"] >= 1.0)
                    for info in arm_info
                ]
                state_outer = [states.get(outer1, {}), states.get(outer2, {})]
                center_state = states.get(center, {})
                rows.append(
                    {
                        "block": block,
                        "seed": seed,
                        "source_assay": assay,
                        "center": center,
                        "outer1": outer1,
                        "outer2": outer2,
                        "arm1": "|".join(arm1),
                        "arm2": "|".join(arm2),
                        "outer_pair": "|".join(outer),
                        "median_collider_separation": float(np.median(view_separation)),
                        "minimum_collider_separation": float(np.min(view_separation)),
                        "median_collider_opening": float(np.median(view_opening)),
                        "minimum_collider_opening": float(np.min(view_opening)),
                        "median_outer_state_support_floor": float(np.median(view_state_floor)),
                        "minimum_outer_state_support_floor": float(np.min(view_state_floor)),
                        "min_arm_vote_fraction": min(vote_fractions),
                        "min_arm_lor": min(values),
                        "mean_arm_lor": float(np.mean(values)),
                        "max_arm_empirical_p": max(empirical_p),
                        "outer_lor": float(evidence[outer]["mean_lor"]),
                        "min_arm_vs_outer_lor": min(values) - float(evidence[outer]["mean_lor"]),
                        "base_arm_view_support_min": min(base_support),
                        "base_arm_view_support_sum": sum(base_support),
                        "current_arm_support": sum(current_support),
                        "rescue_arm_support": sum(rescue_support),
                        "center_state_selected": bool(center_state.get("selected", False)),
                        "outer_state_selected_count": int(sum(bool(item.get("selected", False)) for item in state_outer)),
                        "center_state_maxT_p": float(center_state.get("maxT_p", math.nan)),
                        "outer_max_state_maxT_p": float(max(item.get("maxT_p", math.nan) for item in state_outer)),
                        "center_partial_state_fit": float(center_state.get("partial_fit", math.nan)),
                        "max_loading_empirical_null_p": float(max(arm_loading_p)) if arm_loading_p else math.nan,
                        "minimum_absolute_loading_cosine": float(min(arm_loading_cosine)) if arm_loading_cosine else math.nan,
                        "loading_p_le_0_125_support": int(sum(value <= 0.125 for value in arm_loading_p)),
                        "n_loading_arm_views": len(arm_loading_p),
                        "risky_component": bool(sum(rescue_support) or sum(base_support) or sum(current_support)),
                    }
                )
    return rows


def _formula_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    beta = (
        frame["min_arm_vote_fraction"].ge(5.0 / 6.0)
        & frame["max_arm_empirical_p"].le(0.02)
        & frame["min_arm_lor"].ge(1.0)
    )
    separation = frame["median_collider_separation"].gt(0.0)
    state = frame["median_outer_state_support_floor"].gt(0.0)
    loading = frame["max_loading_empirical_null_p"].le(0.125)
    return {
        "all_observed_rank_no_abstention": pd.Series(True, index=frame.index),
        "beta_rank": beta,
        "beta_positive_separation": beta & separation,
        "beta_separation_state": beta & separation & state,
        "beta_separation_state_loading": beta & separation & state & loading,
    }


def _selected_stars(frame: pd.DataFrame, formula: str) -> pd.DataFrame:
    mask = _formula_masks(frame)[formula]
    eligible = frame.loc[mask].copy()
    if eligible.empty:
        return eligible
    eligible["rank_score"] = (
        eligible["min_arm_lor"]
        + eligible["median_collider_separation"]
        + 0.25 * eligible["min_arm_vs_outer_lor"]
    )
    rows: list[pd.Series] = []
    for (_, seed, assay), group in eligible.groupby(["block", "seed", "source_assay"], sort=True):
        ordered = group.sort_values(
            ["rank_score", "minimum_collider_separation", "center", "outer1", "outer2"],
            ascending=[False, False, True, True, True],
        )
        rows.append(ordered.iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected.insert(0, "formula", formula)
    return selected


def _decode(frame: pd.DataFrame, formula: str) -> set[tuple[int, str, str, str]]:
    selected = _selected_stars(frame, formula)
    return {
        (int(row.seed), str(row.center), str(row.outer1), str(row.outer2))
        for row in selected.itertuples(index=False)
    }


def _truth_stars(truth: pd.DataFrame) -> dict[int, set[tuple[str, str, str]]]:
    stars: dict[int, set[tuple[str, str, str]]] = {}
    for (seed, child), group in truth.groupby(["seed", "child"], sort=True):
        parents = sorted(str(value) for value in group["parent"])
        if len(parents) == 2:
            stars.setdefault(int(seed), set()).add((str(child), parents[0], parents[1]))
    return stars


def _score_predictions(predicted: set[tuple[int, str, str, str]], truth: dict[int, set[tuple[str, str, str]]]) -> dict[str, float]:
    truth_set = {(seed, center, outer1, outer2) for seed, rows in truth.items() for center, outer1, outer2 in rows}
    tp = len(predicted & truth_set)
    fp = len(predicted - truth_set)
    fn = len(truth_set - predicted)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision == precision and precision + recall else 0.0}


def _verify_phase_a(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "phase_a_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Phase-A manifest has wrong phase")
    for name, expected in manifest["output_sha256"].items():
        if _sha256(out_dir / name) != expected:
            raise ValueError(f"Phase-A output drift: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/retired_exhaustive_state_star_decoder_audit_v3"),
    )
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    # Phase A: all blocks and all observed candidates are materialized before truth access.
    held: dict[str, list[dict[str, Any]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    freezes: dict[str, str] = {}
    for block in BLOCKS:
        worlds = retired._load_block_worlds(block)
        held[block] = worlds
        candidate_rows.extend(row for world in worlds for row in _phase_a_world(block, world))
        if block != "TRAIN330":
            freeze = Path(retired.BLOCKS[block]["run_root"]) / "global_pretruth_freeze.json"
            freezes[str(freeze)] = _sha256(freeze)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["block", "seed", "source_assay", "center", "outer1", "outer2"]
    ).reset_index(drop=True)
    phase_a_candidates = args.out_dir / "phase_a_star_candidates.tsv"
    candidates.to_csv(phase_a_candidates, sep="\t", index=False)
    phase_a_manifest = {
        "schema_version": OUT_SCHEMA,
        "phase": "observed_only_before_truth",
        "blocks": list(BLOCKS),
        "n_worlds": sum(len(worlds) for worlds in held.values()),
        "n_candidates": int(len(candidates)),
        "candidate_semantics": "all observed center+two-parent triples from complete beta/residual assay views; risky_component is a descriptive flag only",
        "dev_or_locked_opened": False,
        "truth_or_scorer_opened": False,
        "input_global_pretruth_freeze_sha256": freezes,
        "output_sha256": {phase_a_candidates.name: _sha256(phase_a_candidates)},
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.out_dir / "phase_a_manifest.json", phase_a_manifest)

    # Phase B begins only after verifying the frozen observed candidate universe.
    _verify_phase_a(args.out_dir)
    truth_frames: list[pd.DataFrame] = []
    truth_by_seed: dict[int, set[tuple[str, str, str]]] = {}
    for block in BLOCKS:
        world_root = Path(retired.BLOCKS[block]["world_root"])
        direct_truth, _ = diagnostic._load_direct_truth_after_barrier(held[block], world_root)
        truth_frames.append(direct_truth)
    truth = pd.concat(truth_frames, ignore_index=True, sort=False)
    truth_by_seed = _truth_stars(truth)
    labels = candidates.copy()
    labels["truth_exact_star"] = [
        (str(row.center), str(row.outer1), str(row.outer2)) in truth_by_seed.get(int(row.seed), set())
        for row in labels.itertuples(index=False)
    ]
    labels["target_train380_failure"] = labels["seed"].isin(TARGET_SEEDS)
    label_path = args.out_dir / "phase_b_labeled_star_candidates.tsv"
    labels.to_csv(label_path, sep="\t", index=False)

    formulas = list(_formula_masks(labels).keys())
    selected = pd.concat(
        [_selected_stars(labels, formula) for formula in formulas],
        ignore_index=True,
        sort=False,
    )
    selected_path = args.out_dir / "phase_b_formula_selected_stars.tsv"
    selected.to_csv(selected_path, sep="\t", index=False)
    score_rows: list[dict[str, Any]] = []
    for formula in formulas:
        predicted = _decode(labels, formula)
        score_rows.append({"scope": "ALL_RETIRED", "formula": formula, **_score_predictions(predicted, truth_by_seed)})
        for block in BLOCKS:
            seeds = set(labels.loc[labels["block"].eq(block), "seed"].astype(int))
            score_rows.append({"scope": block, "formula": formula, **_score_predictions({item for item in predicted if item[0] in seeds}, {seed: rows for seed, rows in truth_by_seed.items() if seed in seeds})})
    scores = pd.DataFrame(score_rows)
    score_path = args.out_dir / "phase_b_formula_scores.tsv"
    scores.to_csv(score_path, sep="\t", index=False)

    loo_rows: list[dict[str, Any]] = []
    for held_block in BLOCKS:
        train_scores = scores.loc[(scores["scope"].isin([block for block in BLOCKS if block != held_block]))]
        aggregate = train_scores.groupby("formula", sort=True)[["tp", "fp", "fn"]].sum()
        aggregate["precision"] = aggregate["tp"] / (aggregate["tp"] + aggregate["fp"])
        aggregate["recall"] = aggregate["tp"] / (aggregate["tp"] + aggregate["fn"])
        aggregate["f1"] = 2 * aggregate["precision"] * aggregate["recall"] / (aggregate["precision"] + aggregate["recall"])
        selected = aggregate.reset_index().sort_values(["f1", "precision", "recall", "fp", "formula"], ascending=[False, False, False, True, True]).iloc[0]
        held_row = scores.loc[(scores["scope"].eq(held_block)) & (scores["formula"].eq(selected.formula))].iloc[0]
        loo_rows.append({"held_out_block": held_block, "selected_formula": selected.formula, "training_precision": float(selected.precision), "training_recall": float(selected.recall), "training_f1": float(selected.f1), "held_precision": float(held_row.precision), "held_recall": float(held_row.recall), "held_tp": int(held_row.tp), "held_fp": int(held_row.fp), "held_fn": int(held_row.fn)})
    loo = pd.DataFrame(loo_rows)
    loo_path = args.out_dir / "phase_b_leave_one_block_out.tsv"
    loo.to_csv(loo_path, sep="\t", index=False)

    target_rows: list[dict[str, Any]] = []
    for formula in formulas:
        predicted = _decode(labels, formula)
        for seed in sorted(TARGET_SEEDS):
            target_truth = {seed: truth_by_seed.get(seed, set())}
            target_rows.append({"formula": formula, "seed": seed, **_score_predictions({item for item in predicted if item[0] == seed}, target_truth)})
    targets = pd.DataFrame(target_rows)
    target_path = args.out_dir / "phase_b_train380_target_recovery.tsv"
    targets.to_csv(target_path, sep="\t", index=False)

    summary = {
        "schema_version": OUT_SCHEMA,
        "status": "posthoc_retired_train_diagnostic_not_production_certification",
        "phase_a_verified_before_truth_access": True,
        "dev_or_locked_opened": False,
        "n_phase_a_candidates": int(len(candidates)),
        "n_labeled_true_stars": int(labels["truth_exact_star"].sum()),
        "formulas": formulas,
        "target_seeds": sorted(TARGET_SEEDS),
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "phase_a_manifest_sha256": _sha256(args.out_dir / "phase_a_manifest.json"),
        "output_sha256": {path.name: _sha256(path) for path in (phase_a_candidates, label_path, selected_path, score_path, loo_path, target_path, summary_path)},
        "source_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
