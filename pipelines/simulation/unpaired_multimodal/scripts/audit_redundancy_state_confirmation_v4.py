#!/usr/bin/env python3
"""Audit a subtractive within-assay state-confirmation filter on retired TRAIN.

The audit has an explicit global barrier.  Phase A loads every already-frozen
discovery table from TRAIN360, TRAIN370, and TRAIN380, applies one fixed rule,
and content-addresses the resulting selections before any role or scorer file
is opened.  Phase B then labels those frozen selections.  DEV and LOCKED paths
are neither accepted nor discovered.

The rule does not add state members.  Within each assay it retains preliminary
maxT-selected features whose existing direction-stability diagnostic passes.
If none passes, it retains exactly one preliminary member: the smallest
feature-global-maxT p-value, with reference Wald and feature name as fixed
tie-breakers.  Thus an admitted assay cannot disappear merely because every
member is near the direction-stability boundary, while redundant unstable
members are demoted whenever a stable alternative exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


BLOCKS = {
    "TRAIN360": (36001, 36024),
    "TRAIN370": (37001, 37024),
    "TRAIN380": (38001, 38024),
}
STATE_ROLES = {"P", "Q", "S", "C"}
PHASE_A_COLUMNS = (
    "feature",
    "modality",
    "selection_status",
    "feature_global_maxT_p",
    "reference_conditional_wald",
    "median_conditional_score_direction_stability",
    "direction_stability_diagnostic",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _roots(block: str) -> tuple[Path, Path]:
    start, end = BLOCKS[block]
    run = Path(f"pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_{start}_{end}")
    world = Path(f"pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_{start}_{end}")
    return run, world


def _state_path(run_root: Path, seed: int) -> Path:
    return (
        run_root
        / "artifacts"
        / f"world_seed{seed}"
        / "shared_state_subspace_v4"
        / "subspace_feature_membership.tsv"
    )


def _apply_fixed_filter(frame: pd.DataFrame) -> pd.DataFrame:
    preliminary = frame.loc[frame["selection_status"].eq("selected")].copy()
    selected_indices: set[int] = set()
    fallback_indices: set[int] = set()
    for _, group in preliminary.groupby("modality", sort=True):
        stable = group.loc[group["direction_stability_diagnostic"].astype(bool)]
        if not stable.empty:
            selected_indices.update(int(index) for index in stable.index)
            continue
        winner = group.sort_values(
            [
                "feature_global_maxT_p",
                "reference_conditional_wald",
                "feature",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        ).index[0]
        selected_indices.add(int(winner))
        fallback_indices.add(int(winner))
    output = frame.copy()
    output["preliminary_maxT_selected"] = output["selection_status"].eq(
        "selected"
    )
    output["redundancy_confirmed"] = output.index.isin(selected_indices)
    output["redundancy_fallback_selected"] = output.index.isin(
        fallback_indices
    )
    output["redundancy_demoted"] = (
        output["preliminary_maxT_selected"]
        & ~output["redundancy_confirmed"]
    )
    return output


def _phase_a(out_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for block, (start, end) in BLOCKS.items():
        run_root, _ = _roots(block)
        freeze = run_root / "global_pretruth_freeze.json"
        if not freeze.is_file():
            raise FileNotFoundError(freeze)
        hashes[str(freeze)] = _sha256(freeze)
        for seed in range(start, end + 1):
            path = _state_path(run_root, seed)
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, sep="\t")
            missing = sorted(set(PHASE_A_COLUMNS) - set(frame.columns))
            if missing:
                raise ValueError(f"{path} lacks columns: {missing}")
            if frame["feature"].astype(str).duplicated().any():
                raise ValueError(f"{path} contains duplicate features")
            observed = frame.loc[:, PHASE_A_COLUMNS].copy()
            observed.insert(0, "seed", seed)
            observed.insert(0, "block", block)
            frames.append(_apply_fixed_filter(observed))
            hashes[str(path)] = _sha256(path)
    combined = pd.concat(frames, ignore_index=True)
    phase_a_path = out_dir / "phase_a_fixed_state_filter.tsv"
    combined.to_csv(phase_a_path, sep="\t", index=False)
    manifest = {
        "schema_version": (
            "ca_boss_v4.redundancy_state_confirmation.phase_a.v1"
        ),
        "blocks": BLOCKS,
        "n_worlds": 72,
        "all_worlds_loaded_before_any_truth_open": True,
        "dev_or_locked_opened": False,
        "filter_is_strictly_subtractive": True,
        "stable_alternative_uses_existing_direction_stability_diagnostic": True,
        "fallback": (
            "one_preliminary_member_by_min_feature_global_maxT_p_then_"
            "max_reference_wald_then_lexical_feature"
        ),
        "input_sha256": hashes,
        "phase_a_table_sha256": _sha256(phase_a_path),
        "source_sha256": _sha256(Path(__file__)),
    }
    manifest_path = out_dir / "phase_a_manifest.json"
    _write_json(manifest_path, manifest)
    return combined, hashes


def _phase_b(phase_a: pd.DataFrame, out_dir: Path) -> None:
    labeled_frames: list[pd.DataFrame] = []
    phase_a_manifest = out_dir / "phase_a_manifest.json"
    for block, (start, end) in BLOCKS.items():
        _, world_root = _roots(block)
        for seed in range(start, end + 1):
            roles_path = (
                world_root
                / f"world_seed{seed}"
                / "scorer"
                / "role_assignments.tsv"
            )
            roles = pd.read_csv(roles_path, sep="\t").loc[
                :, ["node", "role"]
            ]
            world = phase_a.loc[
                phase_a["block"].eq(block) & phase_a["seed"].eq(seed)
            ].merge(
                roles,
                left_on="feature",
                right_on="node",
                how="left",
                validate="one_to_one",
            )
            world["role"] = world["role"].fillna(
                world["modality"].map(lambda value: "GENE" if value == "rna" else "")
            )
            if world["role"].eq("").any():
                raise ValueError(f"Unlabeled focal feature in seed {seed}")
            world["truth_state_member"] = world["role"].isin(
                STATE_ROLES | {"GENE"}
            )
            world["truth_sentinel"] = world["role"].astype(str).str.startswith(
                "N"
            )
            labeled_frames.append(world)
    labeled = pd.concat(labeled_frames, ignore_index=True)
    labeled_path = out_dir / "phase_b_labeled_fixed_state_filter.tsv"
    labeled.to_csv(labeled_path, sep="\t", index=False)

    rows: list[dict[str, Any]] = []
    for (block, seed), world in labeled.groupby(["block", "seed"], sort=True):
        selected = world.loc[world["redundancy_confirmed"].astype(bool)]
        tp = int(selected["truth_state_member"].sum())
        fp = int(selected["truth_sentinel"].sum())
        all_assays = int(selected["modality"].nunique()) == 7
        rows.append(
            {
                "block": block,
                "seed": int(seed),
                "n_selected": len(selected),
                "true_selected": tp,
                "sentinel_selected": fp,
                "all_7_assays": all_assays,
                "confirmed_subset_reliable": bool(all_assays and fp == 0),
                "n_demoted": int(world["redundancy_demoted"].sum()),
                "n_fallback_selected": int(
                    world["redundancy_fallback_selected"].sum()
                ),
            }
        )
    world_metrics = pd.DataFrame(rows)
    world_path = out_dir / "phase_b_world_metrics.tsv"
    world_metrics.to_csv(world_path, sep="\t", index=False)
    block_rows: list[dict[str, Any]] = []
    for block, world_group in world_metrics.groupby("block", sort=True):
        feature_group = labeled.loc[labeled["block"].eq(block)]
        selected = feature_group.loc[
            feature_group["redundancy_confirmed"].astype(bool)
        ]
        tp = int(selected["truth_state_member"].sum())
        fp = int(selected["truth_sentinel"].sum())
        block_rows.append(
            {
                "block": block,
                "n_worlds": len(world_group),
                "reliable_worlds": int(
                    world_group["confirmed_subset_reliable"].sum()
                ),
                "true_selected": tp,
                "sentinel_selected": fp,
                "precision": tp / max(tp + fp, 1),
                "total_demoted": int(world_group["n_demoted"].sum()),
                "total_fallback_selected": int(
                    world_group["n_fallback_selected"].sum()
                ),
            }
        )
    block_summary = pd.DataFrame(block_rows)
    summary_path = out_dir / "phase_b_block_summary.tsv"
    block_summary.to_csv(summary_path, sep="\t", index=False)
    _write_json(
        out_dir / "phase_b_manifest.json",
        {
            "schema_version": (
                "ca_boss_v4.redundancy_state_confirmation.phase_b.v1"
            ),
            "phase_a_manifest_sha256": _sha256(phase_a_manifest),
            "truth_opened_only_after_phase_a_manifest": True,
            "dev_or_locked_opened": False,
            "labeled_table_sha256": _sha256(labeled_path),
            "world_metrics_sha256": _sha256(world_path),
            "block_summary_sha256": _sha256(summary_path),
            "status": "retired_train_posthoc_audit_not_certification",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase_a, _ = _phase_a(args.out_dir)
    _phase_b(phase_a, args.out_dir)


if __name__ == "__main__":
    main()
