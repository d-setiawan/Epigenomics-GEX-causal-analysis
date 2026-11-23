#!/usr/bin/env python3
"""Diagnose depth-conditioned latent-component geometry on cached worlds.

Unpaired assays can share a biological-state distribution while differing in
library depth.  This diagnostic fingerprints each assay-local ICA component
within four rank-balanced strata of *observed background library depth*, then
compares those conditional quantile distributions with a sign-invariant W2
cost.  The resulting component cost matrices enter the existing exact
multi-assay permutation synchronizer in the full fit and both disjoint
background-feature folds.

The pretruth evidence phase reads only existing hash-bound component scores,
reporter assignments, and observed depth.  It creates no worlds, cell
couplings, or external features.  A later diagnostic phase uses non-target
truth only to identify empirical-null assay pairs and opens target truth only
to score already-selected proposals.  Results are retrospective diagnostics,
not fresh certification or cross-assay causal directions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_joint_context_stratified_stability_rescue_v4 as v458  # noqa: E402
import diagnose_joint_sliced_geometry_rescue_v4 as joint  # noqa: E402
import diagnose_quantile_geometry_rescue_v4 as geometry  # noqa: E402


SCHEMA = "ca_boss_v4.depth_conditioned_geometry_diagnostic.v1"
ALGORITHM = "CA-BOSS v4 depth-conditioned component-distribution diagnostic"
DEPTH_BINS = 4
QUANTILE_GRID = geometry.QUANTILE_GRID
METRICS = ("depth_conditional_w2", "depth_conditional_z_w2")
FEATURE_SUFFIXES = joint.FEATURE_SUFFIXES
VARIANTS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "depth_conditional_global_mirror": (
        (
            "statistical_context_support_count",
            "depth_conditional_w2_global_coassignment_context_count",
            "depth_conditional_w2_maximum_reporter_cost_ratio",
            "depth_conditional_w2_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "depth_conditional_z_global_mirror": (
        (
            "statistical_context_support_count",
            "depth_conditional_z_w2_global_coassignment_context_count",
            "depth_conditional_z_w2_maximum_reporter_cost_ratio",
            "depth_conditional_z_w2_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, -1.0, 1.0),
    ),
    "joint_plus_depth_confirmation": (
        (
            "statistical_context_support_count",
            "joint_sliced_w2_global_coassignment_context_count",
            "depth_conditional_w2_global_coassignment_context_count",
            "joint_sliced_w2_maximum_reporter_cost_ratio",
            "depth_conditional_w2_maximum_reporter_cost_ratio",
            "joint_sliced_w2_mean_coassigned_exclusion_margin",
            "depth_conditional_w2_mean_coassigned_exclusion_margin",
        ),
        (1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0),
    ),
}


def _balanced_depth_bins(depth: np.ndarray, cells: Sequence[str]) -> np.ndarray:
    """Assign deterministic, equally sized within-assay depth ranks."""

    values = np.asarray(depth, dtype=np.float64)
    labels = np.asarray(list(map(str, cells)), dtype=object)
    if values.ndim != 1 or len(values) != len(labels) or len(values) < 8:
        raise ValueError("Depth strata require at least eight aligned cells")
    if not np.isfinite(values).all():
        raise ValueError("Observed background depth must be finite")
    order = np.lexsort((labels, values))
    bins = np.empty(len(values), dtype=np.int64)
    bins[order] = np.minimum(
        (np.arange(len(values), dtype=np.int64) * DEPTH_BINS) // len(values),
        DEPTH_BINS - 1,
    )
    return bins


def _depth_fingerprint(
    scores: np.ndarray,
    depth: np.ndarray,
    cells: Sequence[str],
    *,
    standardize: bool,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(depth):
        raise ValueError("Expected aligned rank-three component scores")
    if not np.isfinite(values).all():
        raise ValueError("Component scores must be finite")
    if standardize:
        scale = values.std(axis=0)
        if np.any(scale <= 1e-12):
            raise ValueError("A component has degenerate variance")
        values = (values - values.mean(axis=0)) / scale
    strata = _balanced_depth_bins(depth, cells)
    fingerprints = np.empty(
        (3, DEPTH_BINS, len(QUANTILE_GRID)), dtype=np.float64
    )
    for component in range(3):
        for depth_bin in range(DEPTH_BINS):
            selected = values[strata == depth_bin, component]
            if len(selected) < 2:
                raise ValueError("A balanced depth stratum has fewer than two cells")
            fingerprints[component, depth_bin] = np.quantile(
                selected, QUANTILE_GRID
            )
    return fingerprints


def _fingerprint_cost_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.shape[0:2] != (3, DEPTH_BINS):
        raise ValueError("Depth fingerprints have incompatible shapes")
    matrix = np.empty((3, 3), dtype=np.float64)
    for left_component in range(3):
        for right_component in range(3):
            lhs = left[left_component]
            rhs = right[right_component]
            same = float(np.sqrt(np.mean(np.square(lhs - rhs))))
            flipped = float(
                np.sqrt(np.mean(np.square(lhs + rhs[:, ::-1])))
            )
            matrix[left_component, right_component] = min(same, flipped)
    return matrix


def _depth_table(discovery_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(discovery_dir.glob("observed_focal__*.tsv")):
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=["cell", "modality", "observed_background_library_size"],
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No observed focal tables in {discovery_dir}")
    result = pd.concat(frames, ignore_index=True)
    if result[["modality", "cell"]].duplicated().any():
        raise ValueError("Observed depth contains duplicate modality/cell rows")
    return result


def _context_tables(
    mdcr_dir: Path,
    discovery_dir: Path,
) -> tuple[list[str], dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]]]:
    depth = _depth_table(discovery_dir)
    full = pd.read_csv(mdcr_dir / "component_scores.tsv", sep="\t")
    folds = pd.read_csv(mdcr_dir / "fold_component_scores.tsv", sep="\t")
    component_columns = [f"component_{index}" for index in range(3)]
    contexts: list[tuple[str, pd.DataFrame]] = [("full_fit", full)]
    fold_ids = sorted(
        pd.to_numeric(folds["feature_fold"], errors="raise").astype(int).unique()
    )
    if fold_ids != [0, 1]:
        raise ValueError("Expected exactly feature folds 0 and 1")
    for fold_id in fold_ids:
        contexts.append(
            (
                f"feature_fold_{fold_id}",
                folds[
                    pd.to_numeric(folds["feature_fold"], errors="raise")
                    .astype(int)
                    .eq(fold_id)
                ],
            )
        )
    output: dict[
        str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]
    ] = {}
    for label, frame in contexts:
        by_modality: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        for modality, group in frame.groupby("modality", sort=True):
            merged = group.loc[:, ["cell_id", *component_columns]].merge(
                depth[depth["modality"].astype(str).eq(str(modality))],
                left_on="cell_id",
                right_on="cell",
                how="left",
                validate="one_to_one",
            )
            if merged["observed_background_library_size"].isna().any():
                raise ValueError(f"Component cells lack observed depth: {modality}")
            by_modality[str(modality)] = (
                merged[component_columns].to_numpy(dtype=np.float64),
                merged["observed_background_library_size"].to_numpy(
                    dtype=np.float64
                ),
                merged["cell_id"].astype(str).tolist(),
            )
        output[label] = by_modality
    return [label for label, _ in contexts], output


def _formal_discovery(formal_root: Path, world_key: str) -> Path:
    marker = "__world_seed"
    if marker not in world_key:
        raise ValueError(f"Invalid world key: {world_key}")
    block, seed = world_key.split(marker, 1)
    return formal_root / block / f"world_seed{int(seed)}" / "discovery"


def _world_features(
    pair_frame: pd.DataFrame,
    artifacts: geometry._WorldArtifacts,
    formal_root: Path,
    world_key: str,
) -> pd.DataFrame:
    context_labels, contexts = _context_tables(
        artifacts.mdcr_dir,
        _formal_discovery(formal_root, world_key),
    )
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
        for metric, standardize in (
            ("depth_conditional_w2", False),
            ("depth_conditional_z_w2", True),
        ):
            fingerprints = {
                modality: _depth_fingerprint(
                    scores, depth, cells, standardize=standardize
                )
                for modality, (scores, depth, cells) in contexts[label].items()
            }
            pair_matrices = {
                (left, right): _fingerprint_cost_matrix(
                    fingerprints[left], fingerprints[right]
                )
                for left, right in combinations(sorted(fingerprints), 2)
            }
            matrices[metric].append(pair_matrices)
            if len(eligible) >= 2:
                score_arrays = {
                    modality: contexts[label][modality][0]
                    for modality in eligible
                }
                table = geometry._normalized_cost_table(
                    score_arrays, pair_matrices
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
    formal_root: Path,
) -> tuple[str, pd.DataFrame]:
    return key, _world_features(
        frame.reset_index(drop=True), artifacts, formal_root, key
    )


def _sample_keys(
    frames: Mapping[str, pd.DataFrame],
    *,
    worlds_per_block: int,
    random_seed: int,
    excluded: set[str],
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for block_index, block in enumerate(sorted(frames)):
        keys = sorted(set(frames[block]["world_key"].astype(str)) - excluded)
        if len(keys) < worlds_per_block:
            raise ValueError(f"Too few unexcluded cached worlds in {block}")
        rng = np.random.default_rng(
            np.random.SeedSequence([random_seed, block_index])
        )
        indexes = rng.choice(len(keys), size=worlds_per_block, replace=False)
        selected[block] = sorted(keys[int(index)] for index in indexes)
    return selected


def _load_truth(
    formal_root: Path, world_keys: Sequence[str]
) -> dict[str, set[frozenset[str]]]:
    truth: dict[str, set[frozenset[str]]] = {}
    for key in world_keys:
        marker = "__world_seed"
        block, seed = key.split(marker, 1)
        path = formal_root / block / f"world_seed{int(seed)}" / "scorer/truth_typed_edges.tsv"
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        cross = frame[frame["truth_type"].astype(str).eq("cross_assay_shared_state")]
        truth[key] = {
            geometry._pair(f"chrom_{left}", f"chrom_{right}")
            for marks in cross["mark"].astype(str)
            for left, right in [marks.split("|")]
        }
    return truth


def _v458_baseline(output_root: Path) -> tuple[dict[str, set[frozenset[str]]], dict[str, str]]:
    result: dict[str, set[frozenset[str]]] = {}
    payloads: dict[str, str] = {}
    pattern = "bootstrap_joint_context_stratified_stability_rescue_v458_source_current_target_*"
    for directory in sorted(output_root.glob(pattern)):
        manifest = geometry._verified_manifest(
            directory / "calibration_manifest.json", v458.SCHEMA
        )
        block = str(manifest["target_block"])
        if block in payloads:
            raise ValueError(f"Duplicate v4.58 block: {block}")
        path = directory / "bootstrap_joint_context_stability_candidates.tsv"
        if geometry._sha256(path) != str(manifest["output_sha256"][path.name]):
            raise ValueError(f"v4.58 candidate hash mismatch: {block}")
        candidates = pd.read_csv(path, sep="\t", keep_default_na=False)
        for row in candidates.itertuples(index=False):
            result.setdefault(str(row.world_key), set()).update(
                geometry._parse_pairs(row.candidate_assay_pairs_json)
            )
        payloads[block] = str(manifest["manifest_payload_sha256"])
    if len(payloads) != 5:
        raise ValueError(f"Expected five v4.58 blocks, found {len(payloads)}")
    return result, payloads


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
    block_keys = _sample_keys(
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
            executor.submit(
                _compute_world,
                key,
                world_frames[key],
                artifacts[key],
                formal_root,
            ): key
            for key in selected_keys
        }
        for future in as_completed(futures):
            key, frame = future.result()
            completed[key] = frame
    if set(completed) != set(selected_keys):
        raise RuntimeError("Parallel depth-geometry computation lost a world")
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
        v458.diagnostic.v457._verified_joint_evidence(joint_directories[0])
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
    joint_payload = joint_evidence.loc[:, identity + joint_columns]
    if joint_payload.duplicated(identity).any():
        raise ValueError("Frozen joint evidence contains duplicate pair identities")
    enriched = enriched.merge(
        joint_payload,
        on=identity,
        how="left",
        validate="one_to_one",
    )
    if enriched[joint_columns].isna().any().any():
        raise ValueError("Depth evidence lacks a frozen joint-geometry parent row")
    enriched_frames = {
        block: enriched[enriched["block"].astype(str).eq(block)]
        .copy()
        .reset_index(drop=True)
        for block in sorted(selected_frames)
    }

    out_dir.mkdir(parents=True)
    evidence_path = out_dir / "depth_conditioned_pair_evidence.tsv"
    enriched.to_csv(evidence_path, sep="\t", index=False)
    evidence_manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "pretruth_evidence",
        "selected_world_keys": selected_keys,
        "worlds_per_block": worlds_per_block,
        "random_seed": random_seed,
        "excluded_world_keys": sorted(excluded),
        "depth_variable": "observed_background_library_size",
        "depth_bin_count": DEPTH_BINS,
        "depth_bin_rule": "within_assay_rank_balanced_with_cell_id_tie_break",
        "context_definition": "full_fit_plus_two_disjoint_background_feature_folds",
        "quantile_count": len(QUANTILE_GRID),
        "information_boundary": {
            "truth_or_scorer_opened_before_evidence_freeze": False,
            "existing_cached_worlds_only": True,
            "new_worlds_or_datasets_created": False,
            "observed_depth_only": True,
            "heldout_background_component_scores_only": True,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "transport_plan_constructed": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
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

    # Target truth is opened only after the observable evidence table and its
    # content-addressed manifest have been written.
    truth = _load_truth(formal_root, selected_keys)
    baseline, v458_payloads = _v458_baseline(output_root)
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
    parser.add_argument("--random-seed", type=int, default=459000)
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
