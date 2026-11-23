#!/usr/bin/env python3
"""Confirm a predeclared two-view joint-geometry consensus on cached worlds.

This CA-BOSS v4.56 diagnostic consumes two disjoint 40-world screens and tests
their posthoc hypothesis on the 23 cached worlds used by neither screen.  The
rule is frozen before opening confirmation-world truth: a pair must be a new
nominal proposal and reach the fixed stability threshold independently under
both standardized joint sliced-W2 geometry and rank-copula sliced-W2 geometry.

The 80 prior worlds provide empirical-rank training distributions and labelled
null calibration pairs.  No positive label fits a score.  A target block never
contributes truth to its own prediction.  Existing held-out-background scores
are the only inputs; no cells are paired and no transport plan is constructed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_joint_sliced_geometry_rescue_v4 as joint  # noqa: E402


SCHEMA = "ca_boss_v4.joint_sliced_geometry_consensus_confirmation.v1"
ALGORITHM = "CA-BOSS v4.56 two-view joint sliced-Wasserstein consensus"
VIEW_NAMES = ("joint_global_mirror", "copula_global_mirror")
DEFAULT_BOOTSTRAP_REPLICATES = 50
DEFAULT_BOOTSTRAP_FREQUENCY = 0.80
DEFAULT_RANDOM_SEED = 456002
FIXED_SELECTION_RULE = (
    "new_relative_to_target_safe_v455_and_nominally_selected_independently_by_"
    "both_the_standardized_joint_and_rank_copula_global_sliced_w2_views_then_"
    "selected_in_at_least_0_80_of_50_whole_world_bootstrap_refits_in_each_view_"
    "with_other_block_truth_used_only_to_identify_empirical_null_pairs"
)
IDENTITY_COLUMNS = ["world_key", "left_modality", "right_modality"]


def _verified_screen(directory: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = joint.geometry._verified_manifest(
        resolved / "diagnostic_manifest.json", joint.SCHEMA
    )
    if int(manifest.get("sampled_world_count", -1)) != 40:
        raise ValueError("Each prior screen must contain 40 worlds")
    if int(manifest.get("bootstrap_replicates", -1)) != 50:
        raise ValueError("Prior screen bootstrap count changed")
    if float(manifest.get("bootstrap_frequency_threshold", -1.0)) != 0.80:
        raise ValueError("Prior screen stability threshold changed")
    boundary = manifest.get("information_boundary", {})
    for field, expected in {
        "existing_cached_worlds_only": True,
        "new_worlds_or_datasets_created": False,
        "heldout_background_component_scores_only_for_new_features": True,
        "paired_cells_used": False,
        "cross_assay_cell_correspondence_used": False,
        "transport_plan_constructed": False,
        "simulator_latent_state_used": False,
        "positive_labels_used_for_score_or_threshold_fit": False,
    }.items():
        if boundary.get(field) is not expected:
            raise ValueError(f"Prior screen boundary changed: {field}")
    source = Path(joint.__file__).resolve()
    if manifest.get("source_sha256", {}).get(source.name) != joint.geometry._sha256(
        source
    ):
        raise ValueError("Prior joint-geometry source changed")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if joint.geometry._sha256(path) != str(expected):
            raise ValueError(f"Prior screen output changed: {name}")
    table = pd.read_csv(
        resolved / "joint_sliced_geometry_pair_features.tsv",
        sep="\t",
        keep_default_na=False,
    ).drop(columns=["truth_pair", "selected_by_v455_baseline"], errors="ignore")
    return manifest, table


def _stable_mask(
    proposal: np.ndarray,
    counts: np.ndarray,
    *,
    replicates: int,
    threshold: float,
) -> np.ndarray:
    if proposal.dtype != bool or counts.shape != proposal.shape:
        raise ValueError("Proposal and bootstrap arrays are incompatible")
    if np.any(counts < 0) or np.any(counts > replicates):
        raise ValueError("Bootstrap count is outside its valid range")
    return proposal & (counts / float(replicates) >= threshold)


def _view_selection(
    *,
    target: pd.DataFrame,
    target_block: str,
    calibration_frames: Mapping[str, pd.DataFrame],
    block_keys: Mapping[str, Sequence[str]],
    world_frames: Mapping[str, pd.DataFrame],
    calibration_truth: Mapping[str, set[frozenset[str]]],
    baseline: Mapping[str, set[frozenset[str]]],
    variant: str,
    variant_index: int,
    target_index: int,
    bootstrap_replicates: int,
    bootstrap_frequency: float,
    random_seed: int,
) -> dict[str, Any]:
    columns, directions = joint.VARIANTS[variant]
    nominal, margins, null_maxima = joint.geometry._nested_selection(
        target=target,
        target_block=target_block,
        frames=calibration_frames,
        block_keys=block_keys,
        truth=calibration_truth,
        columns=columns,
        directions=directions,
        world_frames=world_frames,
    )
    baseline_mask = np.asarray(
        [
            joint.geometry._pair(row.left_modality, row.right_modality)
            in baseline.get(str(row.world_key), set())
            for row in target.itertuples(index=False)
        ],
        dtype=bool,
    )
    proposal = nominal & ~baseline_mask
    counts = np.zeros(len(target), dtype=np.int64)
    calibration_blocks = [
        block for block in sorted(calibration_frames) if block != target_block
    ]
    for replicate in range(bootstrap_replicates):
        sampled_training: dict[tuple[str, str], list[str]] = {}
        for calibration_index, calibration_block in enumerate(calibration_blocks):
            training_blocks = [
                block for block in calibration_blocks if block != calibration_block
            ]
            for training_index, training_block in enumerate(training_blocks):
                keys = list(block_keys[training_block])
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            random_seed,
                            variant_index,
                            target_index,
                            replicate,
                            calibration_index,
                            training_index,
                        ]
                    )
                )
                sampled_training[(calibration_block, training_block)] = [
                    keys[int(index)]
                    for index in rng.integers(0, len(keys), size=len(keys))
                ]
        selected, _, _ = joint.geometry._nested_selection(
            target=target,
            target_block=target_block,
            frames=calibration_frames,
            block_keys=block_keys,
            truth=calibration_truth,
            columns=columns,
            directions=directions,
            world_frames=world_frames,
            sampled_training_keys=sampled_training,
        )
        counts += selected.astype(np.int64)
    stable = _stable_mask(
        proposal,
        counts,
        replicates=bootstrap_replicates,
        threshold=bootstrap_frequency,
    )
    return {
        "nominal": nominal,
        "baseline": baseline_mask,
        "proposal": proposal,
        "counts": counts,
        "stable": stable,
        "margins": margins,
        "null_maximum": float(max(null_maxima.values())),
    }


def run_confirmation(
    *,
    output_root: Path,
    formal_worlds_root: Path,
    prior_screen_dirs: Sequence[Path],
    out_dir: Path,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_frequency: float = DEFAULT_BOOTSTRAP_FREQUENCY,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_workers: int = 10,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    if len(prior_screen_dirs) != 2:
        raise ValueError("Exactly two prior 40-world screens are required")
    if bootstrap_replicates < 1 or not 0.0 < bootstrap_frequency <= 1.0:
        raise ValueError("Bootstrap settings are invalid")
    output_root = output_root.expanduser().absolute().resolve(strict=True)
    formal_worlds_root = formal_worlds_root.expanduser().absolute().resolve(
        strict=True
    )
    verified = [_verified_screen(path) for path in prior_screen_dirs]
    manifests = [value[0] for value in verified]
    prior_tables = [value[1] for value in verified]
    prior_keys_by_block: dict[str, list[str]] = {}
    prior_key_sets: list[set[str]] = []
    for manifest in manifests:
        keys = {
            str(key)
            for values in manifest["sampled_world_keys_by_block"].values()
            for key in values
        }
        prior_key_sets.append(keys)
        for block, values in manifest["sampled_world_keys_by_block"].items():
            prior_keys_by_block.setdefault(str(block), []).extend(map(str, values))
    if prior_key_sets[0] & prior_key_sets[1]:
        raise ValueError("Prior screens are not disjoint")
    if len(prior_key_sets[0] | prior_key_sets[1]) != 80:
        raise ValueError("Prior screens do not cover 80 unique worlds")

    raw_frames, _, v453_manifests = joint.geometry._discover_v453(output_root)
    baseline, v455_manifests = joint._discover_v455_baseline(output_root)
    raw_worlds = joint.geometry.frames_by_world(raw_frames)
    all_keys_by_block = {
        block: sorted(frame["world_key"].astype(str).unique())
        for block, frame in raw_frames.items()
    }
    remaining = {
        block: [
            key
            for key in all_keys_by_block[block]
            if key not in set(prior_keys_by_block.get(block, []))
        ]
        for block in sorted(all_keys_by_block)
    }
    remaining_keys = sorted(key for values in remaining.values() for key in values)
    if len(remaining_keys) != 23:
        raise ValueError(f"Expected 23 untouched cached worlds, found {len(remaining_keys)}")

    calibration_table = pd.concat(prior_tables, ignore_index=True)
    if calibration_table.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Prior feature screens overlap on pair identities")
    calibration_frames = {
        str(block): group.reset_index(drop=True)
        for block, group in calibration_table.groupby("block", sort=True)
    }
    if set(calibration_frames) != set(raw_frames):
        raise ValueError("Prior screens do not represent all five blocks")
    calibration_worlds = joint.geometry.frames_by_world(calibration_frames)
    block_keys = {
        block: sorted(frame["world_key"].astype(str).unique())
        for block, frame in calibration_frames.items()
    }
    if any(len(keys) != 16 for keys in block_keys.values()):
        raise ValueError("Each calibration block must contain 16 prior worlds")

    artifacts = joint.geometry._resolve_artifacts(output_root, remaining_keys)
    completed: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                joint._compute_world, key, raw_worlds[key], artifacts[key]
            ): key
            for key in remaining_keys
        }
        for future in as_completed(futures):
            key, frame = future.result()
            completed[key] = frame
    if set(completed) != set(remaining_keys):
        raise RuntimeError("Confirmation geometry did not return every world")
    target_frames = {
        block: pd.concat(
            [completed[key] for key in remaining[block]], ignore_index=True
        )
        for block in sorted(remaining)
        if remaining[block]
    }

    # Only the 80 prior-screen worlds are opened here to identify null pairs.
    calibration_truth = joint.geometry._load_truth(
        formal_worlds_root,
        {block: block_keys[block] for block in sorted(block_keys)},
    )
    view_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    for target_index, (target_block, target) in enumerate(target_frames.items()):
        results = {
            variant: _view_selection(
                target=target,
                target_block=target_block,
                calibration_frames=calibration_frames,
                block_keys=block_keys,
                world_frames=calibration_worlds,
                calibration_truth=calibration_truth,
                baseline=baseline,
                variant=variant,
                variant_index=variant_index,
                target_index=target_index,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_frequency=bootstrap_frequency,
                random_seed=random_seed,
            )
            for variant_index, variant in enumerate(VIEW_NAMES)
        }
        consensus = np.logical_and.reduce(
            [results[variant]["stable"] for variant in VIEW_NAMES]
        )
        for index, row in enumerate(target.itertuples(index=False)):
            for variant in VIEW_NAMES:
                result = results[variant]
                if result["proposal"][index] or result["stable"][index]:
                    view_rows.append(
                        {
                            "variant": variant,
                            "target_block": target_block,
                            "world_key": str(row.world_key),
                            "left_modality": str(row.left_modality),
                            "right_modality": str(row.right_modality),
                            "nominal_new_proposal": bool(result["proposal"][index]),
                            "minimum_nested_absolute_margin": float(
                                result["margins"][index].min()
                            ),
                            "null_margin_upper_envelope": result["null_maximum"],
                            "bootstrap_selection_count": int(result["counts"][index]),
                            "bootstrap_replicates": bootstrap_replicates,
                            "bootstrap_selection_frequency": float(
                                result["counts"][index] / bootstrap_replicates
                            ),
                            "stable_in_view": bool(result["stable"][index]),
                        }
                    )
            if consensus[index]:
                consensus_rows.append(
                    {
                        "world_key": str(row.world_key),
                        "block": target_block,
                        "seed": int(row.seed),
                        "left_modality": str(row.left_modality),
                        "right_modality": str(row.right_modality),
                        "candidate_cross_assay_feature_pairs_json": str(
                            row.candidate_cross_assay_feature_pairs_json
                        ) if hasattr(row, "candidate_cross_assay_feature_pairs_json") else "[]",
                        "joint_bootstrap_selection_frequency": float(
                            results["joint_global_mirror"]["counts"][index]
                            / bootstrap_replicates
                        ),
                        "copula_bootstrap_selection_frequency": float(
                            results["copula_global_mirror"]["counts"][index]
                            / bootstrap_replicates
                        ),
                        "minimum_view_bootstrap_selection_frequency": float(
                            min(
                                results[variant]["counts"][index]
                                / bootstrap_replicates
                                for variant in VIEW_NAMES
                            )
                        ),
                        "selected_by_fixed_two_view_consensus": True,
                        "fixed_selection_rule": FIXED_SELECTION_RULE,
                        "candidate_pairs_may_be_expanded_as_edges": False,
                        "within_assay_direct_structure_claimed": False,
                        "exact_cross_assay_pair_claimed": False,
                        "cross_assay_direction_claimed": False,
                    }
                )

    out_dir.mkdir(parents=True)
    view_audit = pd.DataFrame(view_rows)
    predictions = pd.DataFrame(consensus_rows)
    view_path = out_dir / "confirmation_view_audit_pretruth.tsv"
    prediction_path = out_dir / "confirmation_consensus_predictions_pretruth.tsv"
    view_audit.to_csv(view_path, sep="\t", index=False)
    predictions.to_csv(prediction_path, sep="\t", index=False)
    pretruth_hashes = {
        view_path.name: joint.geometry._sha256(view_path),
        prediction_path.name: joint.geometry._sha256(prediction_path),
    }

    # Confirmation-world truth is opened only after both prediction files exist.
    target_truth = joint.geometry._load_truth(formal_worlds_root, remaining)
    if predictions.empty:
        scored = predictions.assign(truth_pair=pd.Series(dtype=bool))
    else:
        scored = predictions.copy()
        scored["truth_pair"] = joint.geometry._truth_labels(scored, target_truth)
    score_path = out_dir / "confirmation_consensus_scores.tsv"
    scored.to_csv(score_path, sep="\t", index=False)
    true_count = int(scored["truth_pair"].sum()) if len(scored) else 0
    false_count = int(len(scored) - true_count)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "predeclared_remaining23_confirmation",
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "random_seed": random_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_frequency_threshold": bootstrap_frequency,
        "prior_screen_world_count": 80,
        "confirmation_world_count": len(remaining_keys),
        "confirmation_world_keys_by_block": remaining,
        "selected_confirmation_pair_count": len(scored),
        "selected_confirmation_true_pair_count": true_count,
        "selected_confirmation_false_pair_count": false_count,
        "selected_confirmation_precision": (
            float(true_count / len(scored)) if len(scored) else None
        ),
        "information_boundary": {
            "existing_cached_worlds_only": True,
            "new_worlds_or_datasets_created": False,
            "heldout_background_component_scores_only_for_new_features": True,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "transport_plan_constructed": False,
            "simulator_latent_state_used": False,
            "positive_labels_used_for_score_or_threshold_fit": False,
            "prior_screen_truth_used_only_to_identify_null_pairs": True,
            "target_block_truth_used_for_fit_or_selection": False,
            "confirmation_truth_opened_only_after_pretruth_files_written": True,
        },
        "prior_screen_manifest_sha256": {
            str(Path(path).expanduser().absolute().resolve(strict=True)): (
                joint.geometry._sha256(
                    Path(path).expanduser().absolute().resolve(strict=True)
                    / "diagnostic_manifest.json"
                )
            )
            for path in prior_screen_dirs
        },
        "v453_manifest_payload_sha256_by_block": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v453_manifests.items()
        },
        "v455_manifest_payload_sha256_by_block": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v455_manifests.items()
        },
        "source_sha256": {
            Path(__file__).name: joint.geometry._sha256(Path(__file__).resolve()),
            Path(joint.__file__).name: joint.geometry._sha256(
                Path(joint.__file__).resolve()
            ),
        },
        "pretruth_output_sha256": pretruth_hashes,
        "output_sha256": {
            score_path.name: joint.geometry._sha256(score_path)
        },
        "retrospective_cached_world_diagnostic": True,
        "promoted_to_checkpoint": False,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
    }
    manifest["manifest_payload_sha256"] = joint.geometry._canonical_sha256(manifest)
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--prior-screen-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-frequency", type=float, default=DEFAULT_BOOTSTRAP_FREQUENCY)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-workers", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = run_confirmation(
        output_root=args.output_root,
        formal_worlds_root=args.formal_worlds_root,
        prior_screen_dirs=args.prior_screen_dir,
        out_dir=args.out_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_frequency=args.bootstrap_frequency,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
