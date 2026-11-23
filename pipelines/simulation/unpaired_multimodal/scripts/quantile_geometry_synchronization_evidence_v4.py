#!/usr/bin/env python3
"""Freeze truth-blind Wasserstein synchronization evidence for CA-BOSS v4.54.

The stage reuses only hash-bound, held-out-background ICA scores and the
reporter-component assignments already available to CA-BOSS v4.53.  In the
full fit and two disjoint feature folds, sign-invariant empirical quantile
functions define one- and two-Wasserstein component costs.  Median-normalized
cost matrices are synchronized across all eligible assays by the existing
exact rank-three permutation solver.  No cell coupling is constructed.

This file deliberately freezes evidence only.  It does not open truth, fit a
threshold, or select a hypothesis.  The downstream target-block calibrator
owns empirical-null selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_quantile_geometry_rescue_v4 as diagnostic  # noqa: E402


SCHEMA = "ca_boss_v4.quantile_geometry_synchronization_evidence.v1"
ALGORITHM = "CA-BOSS v4.54 truth-blind Wasserstein quantile synchronization evidence"
DEFAULT_MAX_WORKERS = 10
IDENTITY_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "left_modality",
    "right_modality",
    "candidate_cross_assay_feature_pairs_json",
]
EVIDENCE_COLUMNS = [
    *IDENTITY_COLUMNS,
    *diagnostic.BASE_COLUMNS,
    *[
        f"{metric}_{suffix}"
        for metric in diagnostic.METRICS
        for suffix in (
            "local_mutual_context_count",
            "heldout_local_mutual_context_count",
            "mean_local_relative_margin",
            "minimum_local_relative_margin",
            "maximum_reporter_cost_ratio",
            "mean_reporter_cost_ratio",
            "global_coassignment_context_count",
            "heldout_global_coassignment_context_count",
            "mean_coassigned_exclusion_margin",
        )
    ],
]


def _add_feature_payload(
    frame: pd.DataFrame,
    artifacts: diagnostic._WorldArtifacts,
) -> pd.DataFrame:
    relaxed_path = (
        artifacts.relaxed_dir / "relaxed_reporter_synchronization_pairs.tsv"
    )
    relaxed = pd.read_csv(relaxed_path, sep="\t", keep_default_na=False)
    lookup: dict[frozenset[str], str] = {}
    for row in relaxed.itertuples(index=False):
        pair = diagnostic._pair(row.left_modality, row.right_modality)
        if pair in lookup:
            raise ValueError(f"Duplicate relaxed assay pair in {relaxed_path}: {sorted(pair)}")
        lookup[pair] = str(row.candidate_cross_assay_feature_pairs_json)
    enriched = frame.copy()
    frame_pairs = [
        diagnostic._pair(row.left_modality, row.right_modality)
        for row in enriched.itertuples(index=False)
    ]
    missing = sorted(
        (pair for pair in set(frame_pairs) if pair not in lookup),
        key=diagnostic._pair_tuple,
    )
    if missing:
        raise ValueError(
            f"Relaxed evidence lacks assay pairs required by {artifacts.relaxed_dir}: "
            f"{[diagnostic._pair_tuple(pair) for pair in missing]}"
        )
    enriched["candidate_cross_assay_feature_pairs_json"] = [lookup[pair] for pair in frame_pairs]
    return enriched


def _compute_world(
    key: str,
    frame: pd.DataFrame,
    artifacts: diagnostic._WorldArtifacts,
) -> tuple[str, pd.DataFrame]:
    prepared = _add_feature_payload(frame, artifacts)
    evidence = diagnostic._world_features(prepared, artifacts)
    missing = [column for column in EVIDENCE_COLUMNS if column not in evidence]
    if missing:
        raise ValueError(f"Computed evidence for {key} lacks columns: {missing}")
    return key, evidence.loc[:, EVIDENCE_COLUMNS]


def freeze_evidence(
    *,
    output_root: Path,
    out_dir: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    output_root = output_root.expanduser().absolute().resolve(strict=True)
    frames_by_block, _, v453_manifests = diagnostic._discover_v453(output_root)
    frames_by_world = diagnostic.frames_by_world(frames_by_block)
    world_keys = sorted(frames_by_world)
    artifacts = diagnostic._resolve_artifacts(output_root, world_keys)

    completed: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_world,
                key,
                frames_by_world[key],
                artifacts[key],
            ): key
            for key in world_keys
        }
        for future in as_completed(futures):
            key, frame = future.result()
            completed[key] = frame
    if set(completed) != set(world_keys):
        raise RuntimeError("Parallel evidence freeze did not return every world")
    table = pd.concat(
        [completed[key] for key in world_keys], ignore_index=True
    )
    identities = list(
        zip(
            table["world_key"].astype(str),
            table["left_modality"].astype(str),
            table["right_modality"].astype(str),
            strict=True,
        )
    )
    if len(set(identities)) != len(identities):
        raise ValueError("Quantile evidence contains duplicate world/assay pairs")
    forbidden = {
        "truth_pair",
        "matched_to_unique_truth_pair",
        "truth_type",
        "label",
    }
    if forbidden.intersection(table.columns):
        raise RuntimeError("Truth-bearing columns entered pretruth evidence")

    out_dir.mkdir(parents=True)
    evidence_path = out_dir / "quantile_geometry_pair_evidence.tsv"
    table.to_csv(evidence_path, sep="\t", index=False)
    input_hashes = {
        key: {
            "v453_calibration_manifest_payload_sha256": str(
                v453_manifests[str(frames_by_world[key]["block"].iloc[0])][
                    "manifest_payload_sha256"
                ]
            ),
            "bootstrap_mnn_pair_audit.tsv": str(
                v453_manifests[str(frames_by_world[key]["block"].iloc[0])][
                    "output_sha256"
                ]["bootstrap_mnn_pair_audit.tsv"]
            ),
            "relaxed_evidence_manifest.json": diagnostic._sha256(
                artifacts[key].relaxed_dir / "evidence_manifest.json"
            ),
            "relaxed_reporter_synchronization_pairs.tsv": diagnostic._sha256(
                artifacts[key].relaxed_dir
                / "relaxed_reporter_synchronization_pairs.tsv"
            ),
            "component_scores.tsv": diagnostic._sha256(
                artifacts[key].mdcr_dir / "component_scores.tsv"
            ),
            "fold_component_scores.tsv": diagnostic._sha256(
                artifacts[key].mdcr_dir / "fold_component_scores.tsv"
            ),
            "assay_axis_assignments.tsv": diagnostic._sha256(
                artifacts[key].assignment_path
            ),
        }
        for key in world_keys
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "pretruth_evidence_only",
        "world_count": len(world_keys),
        "world_keys": world_keys,
        "blocks": sorted(frames_by_block),
        "row_count": len(table),
        "component_rank": 3,
        "context_count": 3,
        "context_definition": "full_fit_plus_two_disjoint_background_feature_folds",
        "quantile_grid": {
            "minimum": float(diagnostic.QUANTILE_GRID.min()),
            "maximum": float(diagnostic.QUANTILE_GRID.max()),
            "count": len(diagnostic.QUANTILE_GRID),
        },
        "component_costs": {
            "wasserstein1": "mean_absolute_empirical_quantile_difference",
            "wasserstein2": "root_mean_squared_empirical_quantile_difference",
            "sign_handling": "minimum_of_same_and_reflected_right_distribution",
            "pair_normalization": "median_positive_rank_three_cost_matrix",
        },
        "global_alignment": (
            "exact_multi_assay_rank_three_permutation_synchronization_per_context"
        ),
        "new_features": [
            column for column in EVIDENCE_COLUMNS if column not in IDENTITY_COLUMNS + list(diagnostic.BASE_COLUMNS)
        ],
        "information_boundary": {
            "truth_or_scorer_opened": False,
            "hypothesis_selected": False,
            "existing_cached_worlds_only": True,
            "new_worlds_or_datasets_created": False,
            "heldout_background_features_only": True,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "cell_level_transport_coupling_constructed": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
        },
        "published_method_inspiration": {
            "unpaired_multi_domain_causal_representation": {
                "citation": "Sturma, Squires, Drton, and Uhler, NeurIPS 2023",
                "doi": "10.52202/075280-1496",
            },
            "unpaired_multiomics_optimal_transport": {
                "citation": "Demetci et al., Journal of Computational Biology 2022",
                "doi": "10.1089/cmb.2021.0446",
                "relationship": "distribution_geometry_inspiration_without_cell_coupling",
            },
        },
        "v453_manifest_payload_sha256_by_block": {
            block: str(value["manifest_payload_sha256"])
            for block, value in v453_manifests.items()
        },
        "source_sha256": {
            Path(__file__).name: diagnostic._sha256(Path(__file__).resolve()),
            Path(diagnostic.__file__).name: diagnostic._sha256(
                Path(diagnostic.__file__).resolve()
            ),
            Path(diagnostic.synchronization.__file__).name: diagnostic._sha256(
                Path(diagnostic.synchronization.__file__).resolve()
            ),
            Path(diagnostic.relaxed.__file__).name: diagnostic._sha256(
                Path(diagnostic.relaxed.__file__).resolve()
            ),
        },
        "input_sha256": input_hashes,
        "output_sha256": {
            evidence_path.name: diagnostic._sha256(evidence_path)
        },
        "formal_pretruth_barrier_satisfied": True,
        "new_worlds_or_datasets_created": False,
    }
    manifest["manifest_payload_sha256"] = diagnostic._canonical_sha256(manifest)
    (out_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = freeze_evidence(
        output_root=args.output_root,
        out_dir=args.out_dir,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
