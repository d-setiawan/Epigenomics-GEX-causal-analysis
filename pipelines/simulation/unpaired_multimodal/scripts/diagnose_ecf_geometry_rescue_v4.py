#!/usr/bin/env python3
"""Diagnose characteristic-function component alignment on cached worlds.

Wasserstein quantiles emphasize displacement of probability mass.  This
alternative represents each unpaired assay-local ICA component by its
empirical characteristic function (ECF), a finite kernel-mean fingerprint of
the entire one-dimensional distribution.  Component costs are invariant to
the arbitrary ICA sign and feed the existing local mutual-nearest and exact
multi-assay permutation-synchronization machinery in the full fit and both
disjoint background-feature folds.

Only existing held-out-background component scores and reporter assignments
are used.  No cell coupling, paired cell, new world, simulator state, feature
identity, modality identity, or external prior enters the geometry.  Target
truth is opened only after the evidence freeze to score this retrospective
diagnostic.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_depth_conditioned_geometry_rescue_v4 as depth_diagnostic  # noqa: E402
import diagnose_joint_sliced_geometry_rescue_v4 as joint  # noqa: E402
import diagnose_quantile_geometry_rescue_v4 as geometry  # noqa: E402


SCHEMA = "ca_boss_v4.ecf_geometry_diagnostic.v1"
ALGORITHM = "CA-BOSS v4 empirical-characteristic-function geometry diagnostic"
FREQUENCIES = np.linspace(0.20, 4.0, 32, dtype=np.float64)
FREQUENCY_WEIGHTS = np.exp(-0.125 * np.square(FREQUENCIES))
METRICS = ("ecf_raw", "ecf_z")
FEATURE_SUFFIXES = joint.FEATURE_SUFFIXES
VARIANTS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "ecf_raw_global_mirror": (
        (
            "statistical_context_support_count",
            "ecf_raw_global_coassignment_context_count",
            "ecf_raw_maximum_reporter_cost_ratio",
            "ecf_raw_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "ecf_z_global_mirror": (
        (
            "statistical_context_support_count",
            "ecf_z_global_coassignment_context_count",
            "ecf_z_maximum_reporter_cost_ratio",
            "ecf_z_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "joint_plus_ecf_confirmation": (
        (
            "statistical_context_support_count",
            "joint_sliced_w2_global_coassignment_context_count",
            "ecf_z_global_coassignment_context_count",
            "joint_sliced_w2_maximum_reporter_cost_ratio",
            "ecf_z_maximum_reporter_cost_ratio",
            "joint_sliced_w2_mean_coassigned_exclusion_margin",
            "ecf_z_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0),
    ),
}


def _ecf_fingerprint(values: np.ndarray, *, standardize: bool) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 8:
        raise ValueError("ECF geometry requires at least eight rank-three scores")
    if not np.isfinite(samples).all():
        raise ValueError("Component scores must be finite")
    if standardize:
        scale = samples.std(axis=0)
        if np.any(scale <= 1e-12):
            raise ValueError("A component has degenerate variance")
        samples = (samples - samples.mean(axis=0)) / scale
    # Shape: component x frequency.  Complex means are deterministic and are
    # equivalent to a fixed random-Fourier/kernel-mean embedding.
    return np.asarray(
        [
            np.exp(1j * np.outer(samples[:, component], FREQUENCIES)).mean(axis=0)
            for component in range(3)
        ],
        dtype=np.complex128,
    )


def _ecf_cost_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.shape != (3, len(FREQUENCIES)):
        raise ValueError("ECF fingerprints have incompatible shapes")
    weights = FREQUENCY_WEIGHTS / FREQUENCY_WEIGHTS.sum()
    matrix = np.empty((3, 3), dtype=np.float64)
    for left_component in range(3):
        for right_component in range(3):
            same = np.abs(left[left_component] - right[right_component]) ** 2
            flipped = (
                np.abs(left[left_component] - np.conjugate(right[right_component]))
                ** 2
            )
            matrix[left_component, right_component] = float(
                np.sqrt(min(np.sum(weights * same), np.sum(weights * flipped)))
            )
    return matrix


def _world_features(
    pair_frame: pd.DataFrame, artifacts: geometry._WorldArtifacts
) -> pd.DataFrame:
    context_labels, contexts = geometry._score_contexts(artifacts.mdcr_dir)
    components, supports = geometry._reporter_contexts(
        artifacts.assignment_path, context_labels
    )
    matrices: dict[str, list[dict[tuple[str, str], np.ndarray]]] = {
        metric: [] for metric in METRICS
    }
    solutions: dict[str, list[Any | None]] = {metric: [] for metric in METRICS}
    tables: dict[str, list[pd.DataFrame | None]] = {metric: [] for metric in METRICS}
    for context_index, label in enumerate(context_labels):
        eligible = sorted(
            modality
            for modality, flags in supports.items()
            if flags[context_index] and modality in contexts[label]
        )
        for metric, standardize in (("ecf_raw", False), ("ecf_z", True)):
            fingerprints = {
                modality: _ecf_fingerprint(scores, standardize=standardize)
                for modality, scores in contexts[label].items()
            }
            pair_matrices = {
                (left, right): _ecf_cost_matrix(
                    fingerprints[left], fingerprints[right]
                )
                for left, right in combinations(sorted(fingerprints), 2)
            }
            matrices[metric].append(pair_matrices)
            if len(eligible) >= 2:
                table = geometry._normalized_cost_table(
                    {modality: contexts[label][modality] for modality in eligible},
                    pair_matrices,
                )
                solution = geometry.synchronization.solve_multiway_permutation_synchronization(
                    table, eligible
                )
            else:
                table = None
                solution = None
            tables[metric].append(table)
            solutions[metric].append(solution)

    enriched = pair_frame.copy()
    for metric in METRICS:
        aggregates: dict[str, list[Any]] = {
            f"{metric}_{suffix}": [] for suffix in FEATURE_SUFFIXES
        }
        for row in pair_frame.itertuples(index=False):
            left = str(row.left_modality)
            right = str(row.right_modality)
            local_mutual: list[bool] = []
            local_margins: list[float] = []
            reporter_costs: list[float] = []
            global_coassigned: list[bool] = []
            global_margins: list[float] = []
            for context_index, _ in enumerate(context_labels):
                left_component = components.get(left, [None] * 3)[context_index]
                right_component = components.get(right, [None] * 3)[context_index]
                if left_component is None or right_component is None:
                    local_mutual.append(False)
                    local_margins.append(0.0)
                    reporter_costs.append(float("nan"))
                    global_coassigned.append(False)
                    continue
                matrix = geometry._oriented_matrix(
                    matrices[metric][context_index], left, right
                )
                cost, margin, mutual = geometry._local_diagnostics(
                    matrix, int(left_component), int(right_component)
                )
                reporter_costs.append(cost)
                local_margins.append(margin)
                local_mutual.append(mutual)
                solution = solutions[metric][context_index]
                table = tables[metric][context_index]
                if (
                    solution is None
                    or table is None
                    or left not in solution.modalities
                    or right not in solution.modalities
                ):
                    global_coassigned.append(False)
                    continue
                diagnostic = geometry.synchronization.reporter_pair_diagnostics(
                    solution,
                    table,
                    left,
                    right,
                    int(left_component),
                    int(right_component),
                )
                coassigned = bool(diagnostic["coassigned_in_best_solution"])
                global_coassigned.append(coassigned)
                if coassigned:
                    global_margins.append(
                        float(diagnostic["pair_exclusion_relative_margin"])
                    )
            finite_costs = [value for value in reporter_costs if np.isfinite(value)]
            aggregates[f"{metric}_local_mutual_context_count"].append(
                int(sum(local_mutual))
            )
            aggregates[f"{metric}_heldout_local_mutual_context_count"].append(
                int(sum(local_mutual[1:]))
            )
            aggregates[f"{metric}_mean_local_relative_margin"].append(
                float(np.mean(local_margins)) if local_margins else 0.0
            )
            aggregates[f"{metric}_minimum_local_relative_margin"].append(
                float(min(local_margins)) if local_margins else 0.0
            )
            aggregates[f"{metric}_maximum_reporter_cost_ratio"].append(
                float(max(finite_costs)) if finite_costs else float("nan")
            )
            aggregates[f"{metric}_mean_reporter_cost_ratio"].append(
                float(np.mean(finite_costs)) if finite_costs else float("nan")
            )
            aggregates[f"{metric}_global_coassignment_context_count"].append(
                int(sum(global_coassigned))
            )
            aggregates[f"{metric}_heldout_global_coassignment_context_count"].append(
                int(sum(global_coassigned[1:]))
            )
            aggregates[f"{metric}_mean_coassigned_exclusion_margin"].append(
                float(np.mean(global_margins)) if global_margins else 0.0
            )
        for column, values in aggregates.items():
            enriched[column] = values
    return enriched


def _compute_world(
    key: str,
    frame: pd.DataFrame,
    artifacts: geometry._WorldArtifacts,
) -> tuple[str, pd.DataFrame]:
    return key, _world_features(frame.reset_index(drop=True), artifacts)


def run_diagnostic(
    *,
    output_root: Path,
    formal_root: Path,
    out_dir: Path,
    worlds_per_block: int,
    random_seed: int,
    bootstrap_replicates: int,
    bootstrap_frequency: float,
    max_workers: int,
    exclude_manifest: Path | None = None,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    output_root = output_root.expanduser().absolute().resolve(strict=True)
    formal_root = formal_root.expanduser().absolute().resolve(strict=True)
    frames, _, v453_manifests = geometry._discover_v453(output_root)
    excluded: set[str] = set()
    if exclude_manifest is not None:
        prior = json.loads(exclude_manifest.read_text(encoding="utf-8"))
        excluded.update(map(str, prior["selected_world_keys"]))
    block_keys = depth_diagnostic._sample_keys(
        frames,
        worlds_per_block=worlds_per_block,
        random_seed=random_seed,
        excluded=excluded,
    )
    selected_keys = sorted(key for keys in block_keys.values() for key in keys)
    selected_frames = {
        block: frame[frame["world_key"].astype(str).isin(block_keys[block])]
        .copy()
        .reset_index(drop=True)
        for block, frame in frames.items()
    }
    artifacts = geometry._resolve_artifacts(output_root, selected_keys)
    world_frames = geometry.frames_by_world(selected_frames)
    completed: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_world, key, world_frames[key], artifacts[key]): key
            for key in selected_keys
        }
        for future in as_completed(futures):
            key, frame = future.result()
            completed[key] = frame
    if set(completed) != set(selected_keys):
        raise RuntimeError("Parallel ECF computation lost a world")
    enriched = pd.concat([completed[key] for key in selected_keys], ignore_index=True)

    joint_directories = [
        path
        for path in output_root.glob(
            "joint_sliced_geometry_synchronization_evidence_v457_source_current_all103_pretruth"
        )
        if path.is_dir()
    ]
    if len(joint_directories) != 1:
        raise ValueError("Expected one frozen all-103 v4.57 joint-evidence directory")
    _, joint_manifest, joint_evidence = (
        depth_diagnostic.v458.diagnostic.v457._verified_joint_evidence(
            joint_directories[0]
        )
    )
    identity = ["world_key", "left_modality", "right_modality"]
    joint_columns = sorted(
        {
            column
            for columns, _ in VARIANTS.values()
            for column in columns
            if column.startswith("joint_sliced_w2_")
        }
    )
    enriched = enriched.merge(
        joint_evidence.loc[:, identity + joint_columns],
        on=identity,
        how="left",
        validate="one_to_one",
    )
    if enriched[joint_columns].isna().any().any():
        raise ValueError("ECF evidence lacks a frozen joint-geometry parent row")
    enriched_frames = {
        block: enriched[enriched["block"].astype(str).eq(block)]
        .copy()
        .reset_index(drop=True)
        for block in sorted(selected_frames)
    }

    out_dir.mkdir(parents=True)
    evidence_path = out_dir / "ecf_geometry_pair_evidence.tsv"
    enriched.to_csv(evidence_path, sep="\t", index=False)
    evidence_manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "pretruth_evidence",
        "selected_world_keys": selected_keys,
        "worlds_per_block": worlds_per_block,
        "random_seed": random_seed,
        "excluded_world_keys": sorted(excluded),
        "frequency_grid": FREQUENCIES.tolist(),
        "frequency_weight": "exp(-t^2/8)",
        "context_definition": "full_fit_plus_two_disjoint_background_feature_folds",
        "information_boundary": {
            "truth_or_scorer_opened_before_evidence_freeze": False,
            "existing_cached_worlds_only": True,
            "new_worlds_or_datasets_created": False,
            "heldout_background_component_scores_only": True,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "transport_plan_constructed": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
        },
        "published_method_inspiration": {
            "mmd_ma": {
                "citation": "Liu et al., WABI 2019",
                "doi": "10.4230/LIPIcs.WABI.2019.10",
            },
            "interpretation": (
                "the finite ECF is a deterministic kernel-mean distribution fingerprint; MMD-MA contributes the distribution-alignment principle, not data or correspondence"
            ),
        },
        "source_sha256": {
            Path(__file__).name: geometry._sha256(Path(__file__).resolve())
        },
        "parent_v453_manifest_payload_sha256": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v453_manifests.items()
        },
        "parent_joint_evidence_manifest_payload_sha256": joint_manifest[
            "manifest_payload_sha256"
        ],
        "output_sha256": {evidence_path.name: geometry._sha256(evidence_path)},
        "formal_pretruth_barrier_satisfied": True,
        "fresh_locked_certification": False,
    }
    evidence_manifest["manifest_payload_sha256"] = geometry._canonical_sha256(
        evidence_manifest
    )
    evidence_manifest_path = out_dir / "evidence_manifest.json"
    evidence_manifest_path.write_text(
        json.dumps(evidence_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    truth = depth_diagnostic._load_truth(formal_root, selected_keys)
    baseline, v458_payloads = depth_diagnostic._v458_baseline(output_root)
    original_variants = joint.VARIANTS
    try:
        joint.VARIANTS = VARIANTS
        summary, proposals = joint._evaluate_variants(
            enriched_frames,
            block_keys,
            truth,
            baseline,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_frequency=bootstrap_frequency,
            random_seed=random_seed,
        )
    finally:
        joint.VARIANTS = original_variants
    summary_path = out_dir / "diagnostic_variant_summary.tsv"
    proposal_path = out_dir / "diagnostic_pair_proposals.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    proposals.to_csv(proposal_path, sep="\t", index=False)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "retrospective_truth_separated_diagnostic",
        "selected_world_keys": selected_keys,
        "worlds_per_block": worlds_per_block,
        "random_seed": random_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_frequency": bootstrap_frequency,
        "variant_count": len(VARIANTS),
        "evidence_manifest_payload_sha256": evidence_manifest[
            "manifest_payload_sha256"
        ],
        "evidence_manifest_file_sha256": geometry._sha256(evidence_manifest_path),
        "v458_manifest_payload_sha256_by_block": v458_payloads,
        "target_truth_used_only_to_score_frozen_diagnostic_proposals": True,
        "other_block_truth_used_only_to_identify_empirical_null_pairs": True,
        "new_worlds_or_datasets_created": False,
        "retrospective_cached_world_exploration": True,
        "fresh_locked_certification": False,
        "output_sha256": {
            summary_path.name: geometry._sha256(summary_path),
            proposal_path.name: geometry._sha256(proposal_path),
        },
        "variant_results": summary.to_dict(orient="records"),
    }
    result["summary_payload_sha256"] = geometry._canonical_sha256(result)
    (out_dir / "diagnostic_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worlds-per-block", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=461000)
    parser.add_argument("--bootstrap-replicates", type=int, default=30)
    parser.add_argument("--bootstrap-frequency", type=float, default=0.80)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--exclude-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_diagnostic(
        output_root=args.output_root,
        formal_root=args.formal_root,
        out_dir=args.out_dir,
        worlds_per_block=args.worlds_per_block,
        random_seed=args.random_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_frequency=args.bootstrap_frequency,
        max_workers=args.max_workers,
        exclude_manifest=args.exclude_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
