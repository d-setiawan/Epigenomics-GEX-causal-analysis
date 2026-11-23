#!/usr/bin/env python3
"""Freeze observed-only loading candidates for retired TRAIN 37001--37024.

This is a one-way Phase-A collector.  It imports the already-audited feature
extraction helpers used for the 360 block, but it never opens scorer, score,
role, truth, DEV, or LOCKED artifacts.  All 24 worlds are loaded before the
global freeze is atomically published.  The completed 370 block is treated as
immutable development history; the collector refuses to overwrite an output.
"""

from __future__ import annotations

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

from freeze_formal_loading_recall_candidates_train_v4 import (
    _phase_a_world,
    _sha256,
)


FORMAL_TRAIN_SEEDS = tuple(range(37_001, 37_025))
EXPECTED_ASSAYS_PER_WORLD = 6
EXPECTED_VIEWS_PER_CANDIDATE = 6


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_closure(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "pipelines/simulation/unpaired_multimodal/scripts/freeze_formal_loading_recall_candidates_370_train_v4.py",
        "pipelines/simulation/unpaired_multimodal/scripts/freeze_formal_loading_recall_candidates_train_v4.py",
        "pipelines/simulation/unpaired_multimodal/scripts/structure_anchored_loading_groups_v4.py",
        "pipelines/simulation/unpaired_multimodal/scripts/cell_state_loading_groups.py",
        "pipelines/simulation/unpaired_multimodal/scripts/shared_state_subspace_v4.py",
        "pipelines/simulation/unpaired_multimodal/scripts/benchmark_ca_boss_v4.py",
    )
    closure: dict[str, str] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        closure[relative] = _sha256(path)
    return closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_37001_37024/artifacts",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_37001_37024_loading_recall_diagnostic_v1/phase_a"
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    artifact_root = Path(args.artifact_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    expected_root = (repo_root / "pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_37001_37024/artifacts").resolve()
    if artifact_root != expected_root:
        raise ValueError("Artifact root must be the immutable TRAIN 37001--37024 artifact root")
    if any(token in str(artifact_root).casefold() for token in ("locked", "dev")):
        raise ValueError("DEV/LOCKED roots are forbidden")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    global_pretruth = artifact_root.parent / "global_pretruth_freeze.json"
    if not global_pretruth.is_file():
        raise FileNotFoundError(global_pretruth)
    source_closure = _source_closure(repo_root)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        candidate_frames: list[pd.DataFrame] = []
        view_frames: list[pd.DataFrame] = []
        decision_frames: list[pd.DataFrame] = []
        input_hashes: dict[str, dict[str, str]] = {}
        pretruth_manifest_hashes: dict[str, str] = {}
        candidate_counts: dict[str, int] = {}
        for seed in FORMAL_TRAIN_SEEDS:
            world = artifact_root / f"world_seed{seed}"
            candidates, views, decisions, hashes = _phase_a_world(world, seed)
            if len(candidates) != int(decisions["n_structural_candidates"].sum()):
                raise ValueError(f"World {seed} candidate count disagrees with decisions")
            if len(candidates) > EXPECTED_ASSAYS_PER_WORLD * EXPECTED_ASSAYS_PER_WORLD:
                raise ValueError(f"World {seed} has too many structural candidates")
            if len(views) != EXPECTED_VIEWS_PER_CANDIDATE * len(candidates):
                raise ValueError(f"World {seed} does not have six rows per candidate")
            if len(decisions) != EXPECTED_ASSAYS_PER_WORLD:
                raise ValueError(f"World {seed} does not have six assay decisions")
            manifest = world / "pretruth_consensus_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            candidate_frames.append(candidates)
            view_frames.append(views)
            decision_frames.append(decisions)
            input_hashes[str(seed)] = hashes
            pretruth_manifest_hashes[str(seed)] = _sha256(manifest)
            candidate_counts[str(seed)] = len(candidates)

        candidates = pd.concat(candidate_frames, ignore_index=True)
        views = pd.concat(view_frames, ignore_index=True)
        decisions = pd.concat(decision_frames, ignore_index=True)
        expected_candidates = sum(candidate_counts.values())
        if len(candidates) != expected_candidates:
            raise ValueError("Global candidate count mismatch")
        if len(views) != expected_candidates * EXPECTED_VIEWS_PER_CANDIDATE:
            raise ValueError("Global candidate-view count mismatch")
        if len(decisions) != len(FORMAL_TRAIN_SEEDS) * EXPECTED_ASSAYS_PER_WORLD:
            raise ValueError("Global assay-decision count mismatch")
        if candidates.duplicated(["seed", "pair_key"]).any():
            raise ValueError("Frozen candidates duplicate seed/pair_key")
        if views.duplicated(["seed", "view_id", "pair_key"]).any():
            raise ValueError("Frozen views duplicate seed/view/pair_key")
        forbidden_tokens = ("truth", "role", "scorer", "oracle", "generative")
        for label, frame in (
            ("candidates", candidates),
            ("views", views),
            ("decisions", decisions),
        ):
            forbidden = sorted(
                str(column)
                for column in frame.columns
                if any(token in str(column).casefold() for token in forbidden_tokens)
            )
            if forbidden:
                raise ValueError(f"{label} contains privileged columns: {forbidden}")

        candidates.to_csv(temporary / "observable_candidates.tsv", sep="\t", index=False)
        views.to_csv(temporary / "observable_pair_views.tsv", sep="\t", index=False)
        decisions.to_csv(temporary / "formal_assay_decisions.tsv", sep="\t", index=False)
        freeze = {
            "schema_version": "ca_boss_v4.formal_train_370_loading_recall.phase_a.v1",
            "formal_train_seeds": list(FORMAL_TRAIN_SEEDS),
            "block_status": "immutable_development_history",
            "all_24_worlds_loaded_before_any_role_truth_or_score_open_by_this_process": True,
            "role_truth_or_score_domain_read_by_this_process": False,
            "dev_or_locked_read": False,
            "n_candidates": len(candidates),
            "n_pair_views": len(views),
            "n_assay_decisions": len(decisions),
            "candidate_counts_by_world": candidate_counts,
            "source_sha256": source_closure,
            "global_pretruth_freeze_sha256": _sha256(global_pretruth),
            "pretruth_consensus_manifest_sha256": pretruth_manifest_hashes,
            "observable_input_sha256": input_hashes,
            "runtime": {
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
            "information_boundary": {
                "formal_artifacts_only": True,
                "state_membership_observable_only": True,
                "assay_local_loading_views_only": True,
                "direct_parent_membership_used_as_gate": False,
                "cross_assay_matching_used": False,
                "scorer_or_role_paths_opened": False,
            },
            "output_sha256": {
                "observable_candidates.tsv": _sha256(temporary / "observable_candidates.tsv"),
                "observable_pair_views.tsv": _sha256(temporary / "observable_pair_views.tsv"),
                "formal_assay_decisions.tsv": _sha256(temporary / "formal_assay_decisions.tsv"),
            },
        }
        _write_json(temporary / "global_phase_a_freeze.json", freeze)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.formal_train_370_loading_recall.phase_a_run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "global_phase_a_freeze_sha256": _sha256(
                    temporary / "global_phase_a_freeze.json"
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
