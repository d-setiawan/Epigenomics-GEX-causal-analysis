#!/usr/bin/env python3
"""Cross-check the frozen empirical-null rescue on old TRAIN 33001--33024."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd


OLD_TRAIN_SEEDS = tuple(range(33_001, 33_025))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-phase-a-root",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "structure_anchored_loading_train_diagnostic_v2/phase_a_candidates"
        ),
    )
    parser.add_argument(
        "--old-global-freeze",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "structure_anchored_loading_train_diagnostic_v2/global_phase_a_freeze.json"
        ),
    )
    parser.add_argument(
        "--old-label-audit",
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
            "formal_train_36001_36024_loading_recall_diagnostic_v1/"
            "old_train_33001_33024_crosscheck"
        ),
    )
    args = parser.parse_args()
    phase_root = Path(args.old_phase_a_root).resolve()
    freeze_path = Path(args.old_global_freeze).resolve()
    label_path = Path(args.old_label_audit).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("all_candidates_frozen_before_roles_open"):
        raise ValueError("Old TRAIN candidates did not satisfy the role barrier")
    frames: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    for seed in OLD_TRAIN_SEEDS:
        path = phase_root / f"world_seed{seed}" / "structural_loading_candidates.tsv"
        frame = pd.read_csv(path, sep="\t")
        if any(
            token in str(column).casefold()
            for column in frame.columns
            for token in ("truth", "role", "scorer", "oracle")
        ):
            raise ValueError("Old prelabel candidate contains privileged columns")
        frame.insert(0, "seed", seed)
        frames.append(frame)
        input_hashes[str(seed)] = _sha256(path)
    candidates = pd.concat(frames, ignore_index=True)
    if len(candidates) != 864:
        raise ValueError("Old TRAIN must contain 864 structural candidates")
    state_gate = (
        candidates["sibling_formal_state_selected"].astype(bool)
        & candidates["sibling_feature_global_maxT_confirmation_gate"].astype(bool)
        & candidates["sibling_assay_global_maxT_admission_gate"].astype(bool)
        & candidates["sibling_conditional_wald_view_support"].ge(2.0 / 3.0)
    )
    baseline_mask = (
        state_gate
        & candidates["lower_absolute_loading_cosine"].ge(0.85)
        & candidates["relative_median_parent_margin"].ge(0.10)
    )
    baseline = _rank(candidates.loc[baseline_mask]).groupby(
        ["seed", "modality"], sort=True
    ).head(1).copy()
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
        & state_gate
        & candidates["relative_median_parent_margin"].ge(0.10)
        & candidates["parent_win_view_support"].ge(1.0)
        & candidates["empirical_null_support_p_le_0_125"].ge(1.0)
    )
    rescue_pool = _rank(candidates.loc[rescue_mask])
    pool_counts = rescue_pool.groupby(["seed", "modality"], sort=True).size()
    unique_assays = set(pool_counts.loc[pool_counts.eq(1)].index)
    rescue = rescue_pool.loc[
        [
            (int(row.seed), str(row.modality)) in unique_assays
            for row in rescue_pool.itertuples(index=False)
        ]
    ].copy()
    rescue["selection_stage"] = "empirical_null_unanimity_rescue"
    selected = pd.concat([baseline, rescue], ignore_index=True)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        selected.to_csv(temporary / "prelabel_selected.tsv", sep="\t", index=False)
        prelabel_freeze = {
            "schema_version": "ca_boss_v4.empirical_null_rescue.old_train_prelabel.v1",
            "old_train_seeds": list(OLD_TRAIN_SEEDS),
            "all_candidates_and_selections_built_before_label_audit_open": True,
            "label_audit_read": False,
            "rule": {
                "baseline_lower_cosine": 0.85,
                "baseline_and_rescue_relative_parent_margin": 0.10,
                "rescue_parent_win_view_support": 1.0,
                "rescue_empirical_null_p_le_0_125_view_support": 1.0,
                "unique_rescue_candidate_required": True,
            },
            "input_sha256": input_hashes,
            "prelabel_selected_sha256": _sha256(temporary / "prelabel_selected.tsv"),
        }
        _write_json(temporary / "prelabel_freeze.json", prelabel_freeze)

        # First label-bearing input is opened only after the selection freeze.
        labels = pd.read_csv(label_path, sep="\t")
        if set(labels["seed"].astype(int)) != set(OLD_TRAIN_SEEDS):
            raise ValueError("Old label audit contains an unexpected seed")
        scored = selected.merge(
            labels[
                ["seed", "pair_key", "parent_role", "sibling_role", "exact_ps_pair"]
            ],
            on=["seed", "pair_key"],
            how="left",
            validate="one_to_one",
        )
        if scored["exact_ps_pair"].isna().any():
            raise ValueError("Old selection is absent from the label audit")
        scored["exact_ps_pair"] = scored["exact_ps_pair"].astype(bool)
        if len(baseline) != 119 or len(scored) != 119:
            raise ValueError("Old TRAIN baseline or proposed count changed unexpectedly")
        if not scored["exact_ps_pair"].all():
            raise ValueError("Frozen rescue introduced an old-TRAIN false pair")
        scored.to_csv(temporary / "scored_selected.tsv", sep="\t", index=False)
        summary = {
            "schema_version": "ca_boss_v4.empirical_null_rescue.old_train_score.v1",
            "old_train_seeds": list(OLD_TRAIN_SEEDS),
            "strict_prelabel_barrier": True,
            "baseline": {
                "true_positive": 119,
                "false_positive": 0,
                "precision": 1.0,
                "recall": 119.0 / 144.0,
            },
            "proposed": {
                "true_positive": int(scored["exact_ps_pair"].sum()),
                "false_positive": int((~scored["exact_ps_pair"]).sum()),
                "precision": 1.0,
                "recall": float(scored["exact_ps_pair"].sum() / 144.0),
                "n_rescued": len(rescue),
            },
            "old_label_audit_sha256": _sha256(label_path),
        }
        _write_json(temporary / "aggregate.json", summary)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.empirical_null_rescue.old_train_run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "prelabel_freeze_sha256": _sha256(temporary / "prelabel_freeze.json"),
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
