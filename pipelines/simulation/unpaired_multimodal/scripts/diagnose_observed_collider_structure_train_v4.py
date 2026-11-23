#!/usr/bin/env python3
"""Reproduce the observed-only collider refinement on an already-open TRAIN block.

This is a post-hoc TRAIN diagnostic, not a locked-block certification.  Every
candidate table for every requested world is loaded, refined, and held in
memory before the script opens the first ``scorer/truth_typed_edges.tsv``.
No role table, simulator parameter, DEV world, or LOCKED world is read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from observed_collider_structure import (
    ObservedColliderStructureConfig,
    apply_observed_collider_refinements,
    select_observed_collider_refinements,
)
from structure_aware_collider_motif import ColliderMotifResult


FRESH_TRAIN_SEED_START = 33001
FRESH_TRAIN_SEED_END = 33024


def _validated_train_seeds(seed_start: int, seed_end: int) -> list[int]:
    """Return a nonempty fresh-TRAIN subset and reject every other block."""

    if seed_end < seed_start:
        raise ValueError("Seed range is empty")
    if (
        seed_start < FRESH_TRAIN_SEED_START
        or seed_end > FRESH_TRAIN_SEED_END
    ):
        raise ValueError(
            "This post-hoc diagnostic is restricted to fresh TRAIN seeds "
            f"{FRESH_TRAIN_SEED_START}-{FRESH_TRAIN_SEED_END}"
        )
    return list(range(seed_start, seed_end + 1))


def _read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except EmptyDataError:
        return pd.DataFrame()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _candidate_directory(world_artifact: Path) -> Path:
    candidates = sorted(
        path
        for path in world_artifact.glob("v4_*")
        if path.is_dir() and (path / "collider_motif").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one saved v4 candidate in {world_artifact}, found {len(candidates)}"
        )
    return candidates[0]


def _load_candidate(
    seed: int,
    artifact_root: Path,
    config: ObservedColliderStructureConfig,
) -> dict[str, Any]:
    world_artifact = artifact_root / f"world_seed{seed}"
    candidate = _candidate_directory(world_artifact)
    motif_root = candidate / "collider_motif"
    motif_paths = {
        "direct": motif_root / "direct_candidates.tsv",
        "unresolved": motif_root / "demoted_or_unresolved.tsv",
        "components": motif_root / "component_diagnostics.tsv",
        "wedges": motif_root / "wedge_diagnostics.tsv",
        "aggregated": motif_root / "aggregated_vote_evidence.tsv",
        "metadata": motif_root / "motif_provenance.json",
    }
    for path in motif_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    motif = ColliderMotifResult(
        direct_candidates=_read_tsv(motif_paths["direct"]),
        demoted_or_unresolved=_read_tsv(motif_paths["unresolved"]),
        component_diagnostics=_read_tsv(motif_paths["components"]),
        wedge_diagnostics=_read_tsv(motif_paths["wedges"]),
        aggregated_evidence=_read_tsv(motif_paths["aggregated"]),
        metadata=json.loads(motif_paths["metadata"].read_text()),
    )
    views = sorted(
        path
        for path in world_artifact.glob("cells*_seed*")
        if path.is_dir()
    )
    if len(views) != 6:
        raise ValueError(f"Expected six frozen evidence views for seed {seed}")
    base_views: dict[str, pd.DataFrame] = {}
    residual_views: dict[str, pd.DataFrame] = {}
    feature_views: dict[str, pd.DataFrame] = {}
    input_paths = list(motif_paths.values())
    for index, view in enumerate(views):
        base_path = candidate / f"nested_{index}_edges.tsv"
        residual_path = view / "raw_pair_evidence.tsv"
        feature_path = view / "cell_loading_feature_models.tsv"
        for path in (base_path, residual_path, feature_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        base_views[view.name] = _read_tsv(base_path)
        residual_views[view.name] = _read_tsv(residual_path)
        feature_views[view.name] = _read_tsv(feature_path)
        input_paths.extend((base_path, residual_path, feature_path))
    vote_paths = sorted(
        (world_artifact / "collider_motif_evidence").glob(
            "beta_binomial_vote[0-9].tsv"
        )
    )
    if len(vote_paths) != 6:
        raise ValueError(f"Expected six beta-binomial votes for seed {seed}")
    votes = pd.concat([_read_tsv(path) for path in vote_paths], ignore_index=True)
    input_paths.extend(vote_paths)
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        votes,
        base_views,
        residual_views,
        feature_views,
        config,
    )
    applied = apply_observed_collider_refinements(motif, refinement)
    return {
        "seed": seed,
        "motif": motif,
        "refinement": refinement,
        "applied": applied,
        "input_hashes": {
            str(path.relative_to(artifact_root)): _sha256(path)
            for path in sorted(input_paths)
        },
    }


def _selected_structure_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(candidate["seed"])
    applied = candidate["applied"]
    decisions = candidate["refinement"].decisions
    decision_map = {
        str(record["component_id"]): str(record["action"])
        for record in decisions.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    if applied.direct_candidates.empty:
        return rows
    for (assay, component_id, center), group in applied.direct_candidates.groupby(
        ["source_assay", "component_id", "collider_center"], sort=True
    ):
        outers = sorted(
            str(record["node2"])
            if str(record["node1"]) == str(center)
            else str(record["node1"])
            for record in group.to_dict(orient="records")
        )
        if len(outers) != 2:
            raise ValueError(
                f"Selected component {component_id!r} does not have exactly two arms"
            )
        rows.append(
            {
                "seed": seed,
                "source_assay": str(assay),
                "component_id": str(component_id),
                "center": str(center),
                "outer1": outers[0],
                "outer2": outers[1],
                "selection_action": decision_map.get(
                    str(component_id), "unchanged_pairwise_motif"
                ),
            }
        )
    return rows


def _score_candidate(candidate: dict[str, Any], truth_path: Path) -> dict[str, Any]:
    seed = int(candidate["seed"])
    truth = _read_tsv(truth_path)
    direct_truth = truth.loc[
        truth["truth_type"].astype(str) == "within_assay_direct"
    ]
    true_pairs = {
        _canonical(str(record["parent"]), str(record["child"]))
        for record in direct_truth.to_dict(orient="records")
    }
    true_orientations = {
        _canonical(str(record["parent"]), str(record["child"])): str(record["child"])
        for record in direct_truth.to_dict(orient="records")
    }

    def selected(frame: pd.DataFrame) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
        pairs: set[tuple[str, str]] = set()
        orientations: dict[tuple[str, str], str] = {}
        for record in frame.to_dict(orient="records"):
            edge = _canonical(str(record["node1"]), str(record["node2"]))
            pairs.add(edge)
            orientations[edge] = str(record["collider_center"])
        return pairs, orientations

    current_pairs, current_orientations = selected(
        candidate["motif"].direct_candidates
    )
    applied_pairs, applied_orientations = selected(
        candidate["applied"].direct_candidates
    )
    current_tp = len(current_pairs & true_pairs)
    applied_tp = len(applied_pairs & true_pairs)
    applied_orientation_tp = sum(
        applied_orientations.get(edge) == center
        for edge, center in true_orientations.items()
    )
    return {
        "seed": seed,
        "n_truth_direct_edges": len(true_pairs),
        "current_n_edges": len(current_pairs),
        "current_tp": current_tp,
        "current_fp": len(current_pairs - true_pairs),
        "current_fn": len(true_pairs - current_pairs),
        "current_recall": current_tp / max(len(true_pairs), 1),
        "current_precision": current_tp / max(len(current_pairs), 1),
        "applied_n_edges": len(applied_pairs),
        "applied_tp": applied_tp,
        "applied_fp": len(applied_pairs - true_pairs),
        "applied_fn": len(true_pairs - applied_pairs),
        "applied_recall": applied_tp / max(len(true_pairs), 1),
        "applied_precision": applied_tp / max(len(applied_pairs), 1),
        "applied_orientation_recall": applied_orientation_tp
        / max(len(true_orientations), 1),
        "applied_exact": bool(
            applied_pairs == true_pairs
            and applied_orientation_tp == len(true_orientations)
        ),
        "n_refinement_decisions": len(candidate["refinement"].decisions),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _stack_refinement_evidence(
    held_candidates: list[dict[str, Any]],
    attribute: str,
) -> pd.DataFrame:
    frames = []
    for candidate in held_candidates:
        frame = getattr(candidate["refinement"], attribute).copy()
        frame.insert(0, "seed", int(candidate["seed"]))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        return combined
    sort_columns = [
        column
        for column in (
            "seed",
            "source_assay",
            "component_id",
            "center",
            "outer1",
            "outer2",
            "anchor_outer",
            "new_outer",
            "view_id",
            "vote",
        )
        if column in combined.columns
    ]
    return combined.sort_values(sort_columns).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=FRESH_TRAIN_SEED_START)
    parser.add_argument("--seed-end", type=int, default=FRESH_TRAIN_SEED_END)
    args = parser.parse_args()
    seeds = _validated_train_seeds(args.seed_start, args.seed_end)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = ObservedColliderStructureConfig()

    # Phase A: every candidate is loaded and refined before the scorer loop.
    held_candidates = [
        _load_candidate(seed, args.artifact_root, config) for seed in seeds
    ]
    if len(held_candidates) != len(seeds):
        raise RuntimeError("Candidate barrier is incomplete")

    # Phase B: TRAIN truth is opened only after Phase A succeeded globally.
    metric_rows = []
    for candidate in held_candidates:
        seed = int(candidate["seed"])
        truth_path = (
            args.world_root
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        if not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        metric_rows.append(_score_candidate(candidate, truth_path))
    metrics = pd.DataFrame(metric_rows).sort_values("seed").reset_index(drop=True)
    decisions = pd.concat(
        [
            candidate["refinement"].decisions.assign(seed=int(candidate["seed"]))
            for candidate in held_candidates
            if not candidate["refinement"].decisions.empty
        ],
        ignore_index=True,
    )
    structures = pd.DataFrame(
        [
            row
            for candidate in held_candidates
            for row in _selected_structure_rows(candidate)
        ]
    ).sort_values(["seed", "source_assay", "component_id"])
    wedge_view_evidence = _stack_refinement_evidence(
        held_candidates, "wedge_view_evidence"
    )
    wedge_evidence = _stack_refinement_evidence(
        held_candidates, "wedge_evidence"
    )
    completion_view_evidence = _stack_refinement_evidence(
        held_candidates, "completion_view_evidence"
    )
    completion_evidence = _stack_refinement_evidence(
        held_candidates, "completion_evidence"
    )
    aggregate = {
        "diagnostic_status": "posthoc_fresh_train_not_certification",
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "n_worlds": len(seeds),
        "all_candidates_loaded_before_any_scorer_open": True,
        "current_mean_recall": float(metrics["current_recall"].mean()),
        "current_mean_precision": float(metrics["current_precision"].mean()),
        "current_exact_worlds": int(
            (
                (metrics["current_recall"] == 1.0)
                & (metrics["current_precision"] == 1.0)
            ).sum()
        ),
        "applied_mean_recall": float(metrics["applied_recall"].mean()),
        "applied_mean_precision": float(metrics["applied_precision"].mean()),
        "applied_mean_orientation_recall": float(
            metrics["applied_orientation_recall"].mean()
        ),
        "applied_total_false_edges": int(metrics["applied_fp"].sum()),
        "applied_exact_worlds": int(metrics["applied_exact"].sum()),
        "n_refinement_decisions": int(metrics["n_refinement_decisions"].sum()),
        "n_wedge_view_evidence_rows": int(len(wedge_view_evidence)),
        "n_wedge_evidence_rows": int(len(wedge_evidence)),
        "n_completion_view_evidence_rows": int(len(completion_view_evidence)),
        "n_completion_evidence_rows": int(len(completion_evidence)),
        "fixed_config": asdict(config),
        "formulas": {
            "conditional_outer_residual_correlation": (
                "(r_outer-r_arm1*r_arm2)/sqrt((1-r_arm1^2)*(1-r_arm2^2))"
            ),
            "collider_opening_delta": (
                "r_outer-conditional_outer_residual_correlation"
            ),
            "feature_state_support": (
                "max(oof_partial_state_fit,0)*max(loading_direction_stability,0)"
            ),
            "same_center_star_view_rank": (
                "collider_separation=-conditional_r, outer_state_floor, pairwise_wedge_score"
            ),
            "competing_path_view_rank": (
                "base_arm_count, outer_state_floor, collider_separation, pairwise_wedge_score"
            ),
            "completion_vote_rank": (
                "vote_selected, pooled_log_odds, median_outer_state_floor"
            ),
        },
        "information_limits": {
            "cross_assay_paired_cells_used": False,
            "feature_roles_used_for_selection": False,
            "external_priors_used": False,
            "truth_used_only_after_global_candidate_barrier_for_posthoc_scoring": True,
        },
    }
    metrics_path = args.out_dir / "per_world_direct_metrics.tsv"
    decisions_path = args.out_dir / "refinement_decisions.tsv"
    structures_path = args.out_dir / "selected_collider_structures.tsv"
    aggregate_path = args.out_dir / "aggregate.json"
    audit_path = args.out_dir / "candidate_input_hashes.json"
    evidence_paths = {
        "wedge_view_evidence.tsv": wedge_view_evidence,
        "wedge_evidence.tsv": wedge_evidence,
        "completion_view_evidence.tsv": completion_view_evidence,
        "completion_evidence.tsv": completion_evidence,
    }
    metrics.to_csv(metrics_path, sep="\t", index=False)
    decisions.to_csv(decisions_path, sep="\t", index=False)
    structures.to_csv(structures_path, sep="\t", index=False)
    for name, frame in evidence_paths.items():
        frame.to_csv(args.out_dir / name, sep="\t", index=False)
    _write_json(aggregate_path, aggregate)
    _write_json(
        audit_path,
        {
            f"world_seed{candidate['seed']}": candidate["input_hashes"]
            for candidate in held_candidates
        },
    )
    output_hashes = {
        path.name: _sha256(path)
        for path in (
            metrics_path,
            decisions_path,
            structures_path,
            aggregate_path,
            audit_path,
            *(args.out_dir / name for name in evidence_paths),
        )
    }
    _write_json(
        args.out_dir / "diagnostic_manifest.json",
        {
            "source_sha256": {
                Path(__file__).name: _sha256(Path(__file__)),
                "observed_collider_structure.py": _sha256(
                    Path(__file__).with_name("observed_collider_structure.py")
                ),
            },
            "output_sha256": output_hashes,
            "all_candidates_loaded_before_any_scorer_open": True,
        },
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
