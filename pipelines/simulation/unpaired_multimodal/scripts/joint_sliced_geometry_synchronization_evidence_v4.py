#!/usr/bin/env python3
"""Freeze truth-blind joint sliced-Wasserstein evidence for CA-BOSS v4.57."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_joint_sliced_geometry_rescue_v4 as joint  # noqa: E402


SCHEMA = "ca_boss_v4.joint_sliced_geometry_synchronization_evidence.v1"
ALGORITHM = "CA-BOSS v4.57 truth-blind joint sliced-Wasserstein evidence"
DEFAULT_MAX_WORKERS = 10
IDENTITY_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "left_modality",
    "right_modality",
    "candidate_cross_assay_feature_pairs_json",
]
JOINT_COLUMNS = [
    f"{metric}_{suffix}"
    for metric in joint.METRICS
    for suffix in joint.FEATURE_SUFFIXES
]
EVIDENCE_COLUMNS = [
    *IDENTITY_COLUMNS,
    *joint.geometry.BASE_COLUMNS,
    *JOINT_COLUMNS,
]


def _compute_world(
    key: str,
    frame: pd.DataFrame,
    artifacts: joint.geometry._WorldArtifacts,
) -> tuple[str, pd.DataFrame]:
    return key, joint._world_features(frame.reset_index(drop=True), artifacts)


def freeze_evidence(
    *,
    output_root: Path,
    v454_evidence_dir: Path,
    out_dir: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    output_root = output_root.expanduser().absolute().resolve(strict=True)
    v454_path, v454_manifest, v454_table = joint.v455.v454._verified_evidence(
        v454_evidence_dir
    )
    raw_frames, _, v453_manifests = joint.geometry._discover_v453(output_root)
    world_frames = joint.geometry.frames_by_world(raw_frames)
    world_keys = sorted(world_frames)
    artifacts = joint.geometry._resolve_artifacts(output_root, world_keys)

    completed: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_world, key, world_frames[key], artifacts[key]
            ): key
            for key in world_keys
        }
        for future in as_completed(futures):
            key, frame = future.result()
            completed[key] = frame
    if set(completed) != set(world_keys):
        raise RuntimeError("Parallel joint evidence freeze lost a world")
    table = pd.concat([completed[key] for key in world_keys], ignore_index=True)

    pair_identity = ["world_key", "left_modality", "right_modality"]
    payload = v454_table.loc[
        :, pair_identity + ["candidate_cross_assay_feature_pairs_json"]
    ]
    if payload.duplicated(pair_identity).any():
        raise ValueError("V4.54 feature-pair evidence contains duplicate identities")
    table = table.merge(payload, on=pair_identity, how="left", validate="one_to_one")
    if table["candidate_cross_assay_feature_pairs_json"].isna().any():
        raise ValueError("Joint evidence lacks a v4.54 feature-pair payload")
    for column in joint.geometry.BASE_COLUMNS:
        comparison = v454_table.loc[:, pair_identity + [column]]
        merged = table.loc[:, pair_identity + [column]].merge(
            comparison,
            on=pair_identity,
            suffixes=("_joint", "_v454"),
            validate="one_to_one",
        )
        if not np.allclose(
            pd.to_numeric(merged[f"{column}_joint"], errors="raise"),
            pd.to_numeric(merged[f"{column}_v454"], errors="raise"),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"Joint and v4.54 base evidence disagree: {column}")
    table = table.loc[:, EVIDENCE_COLUMNS]
    identities = list(
        zip(
            table["world_key"].astype(str),
            table["left_modality"].astype(str),
            table["right_modality"].astype(str),
            strict=True,
        )
    )
    if len(identities) != len(set(identities)):
        raise ValueError("Joint evidence contains duplicate world/assay pairs")
    forbidden = {"truth_pair", "truth_type", "role", "label"}
    if forbidden.intersection(table.columns):
        raise RuntimeError("Truth-bearing columns entered joint evidence")

    out_dir.mkdir(parents=True)
    evidence_path = out_dir / "joint_sliced_geometry_pair_evidence.tsv"
    table.to_csv(evidence_path, sep="\t", index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "pretruth_evidence_only",
        "world_count": len(world_keys),
        "world_keys": world_keys,
        "blocks": sorted(raw_frames),
        "row_count": len(table),
        "component_rank": 3,
        "context_count": 3,
        "context_definition": "full_fit_plus_two_disjoint_background_feature_folds",
        "signed_permutation_count": len(joint.SIGNED_PERMUTATIONS),
        "projection_direction_count": len(joint.INTEGER_DIRECTIONS),
        "quantile_grid": {
            "minimum": float(joint.QUANTILE_GRID.min()),
            "maximum": float(joint.QUANTILE_GRID.max()),
            "count": len(joint.QUANTILE_GRID),
        },
        "geometry_views": {
            "joint_sliced_w2": "column-standardized joint rank-three distribution",
            "copula_sliced_w2": "componentwise rank-normalized joint distribution",
            "conditional_component_cost": (
                "minimum sliced-W2 over signed permutations constrained to map the reporter components"
            ),
            "global_alignment": "exact multi-assay rank-three permutation synchronization",
        },
        "information_boundary": {
            "truth_or_scorer_opened": False,
            "hypothesis_selected": False,
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
            "scot": {
                "citation": "Demetci et al., Journal of Computational Biology 2022",
                "doi": "10.1089/cmb.2021.0446",
            },
            "sliced_wasserstein": {
                "citation": "Yi and Liu, PMLR 2023",
                "url": "https://proceedings.mlr.press/v189/yi23a.html",
            },
        },
        "v454_evidence_manifest_payload_sha256": v454_manifest[
            "manifest_payload_sha256"
        ],
        "v453_manifest_payload_sha256_by_block": {
            block: str(manifest["manifest_payload_sha256"])
            for block, manifest in v453_manifests.items()
        },
        "source_sha256": {
            Path(__file__).name: joint.geometry._sha256(Path(__file__).resolve()),
            Path(joint.__file__).name: joint.geometry._sha256(
                Path(joint.__file__).resolve()
            ),
        },
        "input_sha256": {
            "v454_evidence": {
                "evidence_manifest.json": joint.geometry._sha256(
                    v454_path / "evidence_manifest.json"
                ),
                "quantile_geometry_pair_evidence.tsv": joint.geometry._sha256(
                    v454_path / "quantile_geometry_pair_evidence.tsv"
                ),
            },
            "world_artifacts": {
                key: {
                    "component_scores.tsv": joint.geometry._sha256(
                        artifacts[key].mdcr_dir / "component_scores.tsv"
                    ),
                    "fold_component_scores.tsv": joint.geometry._sha256(
                        artifacts[key].mdcr_dir / "fold_component_scores.tsv"
                    ),
                    "assay_axis_assignments.tsv": joint.geometry._sha256(
                        artifacts[key].assignment_path
                    ),
                }
                for key in world_keys
            },
        },
        "output_sha256": {
            evidence_path.name: joint.geometry._sha256(evidence_path)
        },
        "formal_pretruth_barrier_satisfied": True,
        "new_worlds_or_datasets_created": False,
    }
    manifest["manifest_payload_sha256"] = joint.geometry._canonical_sha256(manifest)
    (out_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v454-evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = freeze_evidence(
        output_root=args.output_root,
        v454_evidence_dir=args.v454_evidence_dir,
        out_dir=args.out_dir,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
