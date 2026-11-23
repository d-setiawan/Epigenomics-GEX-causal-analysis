#!/usr/bin/env python3
"""Discovery-only equivalence audit for the production crossfit collider fit.

The audit refits all retired TRAIN330/360/370/380 worlds from their six
observed chromatin cell tables and frozen same-world assay-local background
embedding.  It compares only observed generative columns against the frozen
Phase-A candidate table.  It never opens Phase-B labels, scorer files, DEV, or
LOCKED artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from crossfit_hierarchical_collider_v4 import (
    deterministic_frame_sha256,
    fit_crossfit_collider_world_evidence,
)


SCHEMA = "ca_boss_v4.retired_crossfit_raw_fit_equivalence.v1"
TOLERANCE = 1e-12
BLOCKS: dict[str, dict[str, Any]] = {
    "TRAIN330": {
        "seeds": tuple(range(33001, 33025)),
        "world_root": Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_worlds_v1"
        ),
        "embedding_root": Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/layered_fresh_train_v1/artifacts"
        ),
    },
    "TRAIN360": {
        "seeds": tuple(range(36001, 36025)),
        "world_root": Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_36001_36024"),
        "embedding_root": Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_36001_36024/artifacts"),
    },
    "TRAIN370": {
        "seeds": tuple(range(37001, 37025)),
        "world_root": Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_37001_37024"),
        "embedding_root": Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_37001_37024/artifacts"),
    },
    "TRAIN380": {
        "seeds": tuple(range(38001, 38025)),
        "world_root": Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_38001_38024"),
        "embedding_root": Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_38001_38024/artifacts"),
    },
}
IDENTITY_COLUMNS = ("source_assay", "center", "outer1", "outer2")
METRIC_COLUMNS = (
    "crossfit_joint_gain_over_baseline",
    "crossfit_drop_parent1_gain",
    "crossfit_drop_parent2_gain",
    "crossfit_min_drop_one_gain",
    "crossfit_joint_gain_fold_min",
    "laplace_mean_parent_coefficient_min",
    "laplace_fold_parent_coefficient_min",
    "laplace_min_parent_positive_posterior",
    "laplace_fold_min_parent_positive_posterior",
    "laplace_both_positive_fold_support",
    "laplace_both_practical_fold_support",
    "center_detection_rate",
    "minimum_parent_detection_rate",
    "maximum_parent_detection_rate",
    "minimum_arm_residual_correlation",
    "outer_absolute_residual_correlation",
    "residual_star_separation",
    "maximum_arm_empirical_positive_p",
    "outer_empirical_null_compatibility_p",
    "fold_minimum_arm_residual_correlation",
    "fold_median_arm_residual_correlation",
    "fold_maximum_outer_absolute_correlation",
    "crossfit_joint_gain_over_baseline_reverse_center_margin",
    "crossfit_min_drop_one_gain_reverse_center_margin",
    "laplace_min_parent_positive_posterior_reverse_center_margin",
    "residual_star_separation_reverse_center_margin",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _combined_input_hash(
    tables: dict[str, pd.DataFrame], embedding: pd.DataFrame
) -> str:
    digest = hashlib.sha256()
    for assay, frame in sorted(tables.items()):
        digest.update(assay.encode("utf-8"))
        digest.update(deterministic_frame_sha256(frame).encode("ascii"))
    digest.update(b"assay_local_embedding")
    digest.update(deterministic_frame_sha256(embedding).encode("ascii"))
    return digest.hexdigest()


def _audit_world(task: tuple[str, int, str, str, pd.DataFrame]) -> dict[str, Any]:
    block, seed, world_root_raw, embedding_root_raw, expected = task
    started = time.monotonic()
    world_root = Path(world_root_raw)
    embedding_root = Path(embedding_root_raw)
    discovery = world_root / f"world_seed{seed}" / "discovery"
    embedding_path = (
        embedding_root
        / f"world_seed{seed}"
        / "assay_local_state_embedding_v4"
        / "analysis_embeddings.tsv"
    )
    cell_paths = sorted(discovery.glob("observed_focal__chrom_*.tsv"))
    if len(cell_paths) != 6:
        raise ValueError(f"{block} {seed} does not have six chromatin tables")
    tables: dict[str, pd.DataFrame] = {}
    for path in cell_paths:
        frame = pd.read_csv(
            path,
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        modalities = set(frame["modality"].astype(str))
        if len(modalities) != 1:
            raise ValueError(f"{path} mixes modalities")
        assay = next(iter(modalities))
        if assay in tables:
            raise ValueError(f"{block} {seed} repeats assay {assay}")
        tables[assay] = frame
    embedding = pd.read_csv(
        embedding_path,
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )
    observed = fit_crossfit_collider_world_evidence(
        assay_cell_tables=tables,
        assay_local_embedding=embedding,
        randomization_key=seed,
    ).generative_candidates
    left = observed.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)
    right = expected.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)
    identities_match = bool(
        len(left) == len(right)
        and left[list(IDENTITY_COLUMNS)].equals(right[list(IDENTITY_COLUMNS)])
    )
    maximum = -1.0
    worst = ""
    nan_patterns_match = True
    per_column: dict[str, float] = {}
    if identities_match:
        for column in METRIC_COLUMNS:
            observed_values = left[column].to_numpy(dtype=np.float64)
            expected_values = right[column].to_numpy(dtype=np.float64)
            same_nan = np.array_equal(
                np.isnan(observed_values), np.isnan(expected_values)
            )
            nan_patterns_match = nan_patterns_match and bool(same_nan)
            finite = np.isfinite(observed_values) & np.isfinite(expected_values)
            delta = float(
                np.max(np.abs(observed_values[finite] - expected_values[finite]))
            ) if finite.any() else 0.0
            per_column[column] = delta
            if delta > maximum:
                maximum, worst = delta, column
    else:
        maximum, worst = float("inf"), "candidate_identity"
    expected_slice = right[[*IDENTITY_COLUMNS, *METRIC_COLUMNS]]
    observed_slice = left[[*IDENTITY_COLUMNS, *METRIC_COLUMNS]]
    return {
        "block": block,
        "seed": seed,
        "n_expected_candidates": len(right),
        "n_observed_candidates": len(left),
        "candidate_identities_match": identities_match,
        "nan_patterns_match": nan_patterns_match,
        "n_metric_columns": len(METRIC_COLUMNS),
        "maximum_absolute_delta": maximum,
        "worst_metric_column": worst,
        "within_1e_12": bool(
            identities_match and nan_patterns_match and maximum <= TOLERANCE
        ),
        "expected_generative_slice_sha256": deterministic_frame_sha256(
            expected_slice
        ),
        "observed_generative_slice_sha256": deterministic_frame_sha256(
            observed_slice
        ),
        "raw_discovery_input_sha256": _combined_input_hash(tables, embedding),
        "fit_seconds": time.monotonic() - started,
        "per_column_maximum_absolute_delta": per_column,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_crossfit_hierarchical_collider_audit_v1"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_crossfit_raw_fit_equivalence_audit_v1"
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("workers must be one or two")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")
    phase_a_manifest_path = args.input_dir / "phase_a_manifest.json"
    phase_a_manifest = json.loads(
        phase_a_manifest_path.read_text(encoding="utf-8")
    )
    if (
        phase_a_manifest.get("phase") != "observed_only_before_truth"
        or phase_a_manifest.get("truth_or_scorer_opened") is not False
        or phase_a_manifest.get("dev_or_locked_opened") is not False
    ):
        raise ValueError("Frozen Phase A lacks its observed-only barrier")
    candidate_path = args.input_dir / "phase_a_crossfit_star_candidates.tsv"
    expected_candidate_hash = phase_a_manifest["output_sha256"][candidate_path.name]
    if _sha256(candidate_path) != expected_candidate_hash:
        raise ValueError("Frozen Phase-A candidate table drifted")
    frozen = pd.read_csv(
        candidate_path, sep="\t", float_precision="round_trip"
    )
    required = {"block", "seed", *IDENTITY_COLUMNS, *METRIC_COLUMNS}
    missing = required - set(frozen.columns)
    if missing:
        raise ValueError(f"Frozen candidates lack columns: {sorted(missing)}")

    tasks: list[tuple[str, int, str, str, pd.DataFrame]] = []
    for block, specification in BLOCKS.items():
        for seed in specification["seeds"]:
            expected = frozen.loc[
                frozen["block"].astype(str).eq(block) & frozen["seed"].eq(seed),
                [*IDENTITY_COLUMNS, *METRIC_COLUMNS],
            ].copy()
            if len(expected) != 360:
                raise ValueError(f"Expected 360 frozen candidates for {block} {seed}")
            tasks.append(
                (
                    block,
                    seed,
                    str(specification["world_root"]),
                    str(specification["embedding_root"]),
                    expected,
                )
            )
    args.out_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_audit_world, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['block']} {row['seed']} delta="
                f"{row['maximum_absolute_delta']:.3g} ok={row['within_1e_12']}",
                flush=True,
            )
            if not row["within_1e_12"]:
                print("MISMATCH_EXCEEDS_TOLERANCE", flush=True)

    detailed_path = args.out_dir / "per_world_equivalence_details.json"
    _write_json(
        detailed_path,
        sorted(rows, key=lambda row: (row["block"], row["seed"])),
    )
    compact_rows = [
        {key: value for key, value in row.items() if key != "per_column_maximum_absolute_delta"}
        for row in rows
    ]
    report = pd.DataFrame(compact_rows).sort_values(["block", "seed"])
    report_path = args.out_dir / "per_world_equivalence.tsv"
    report.to_csv(report_path, sep="\t", index=False)
    summary = {
        "schema_version": SCHEMA,
        "status": "discovery_only_retired_raw_fit_equivalence",
        "n_worlds": len(report),
        "n_worlds_with_matching_identities": int(
            report["candidate_identities_match"].sum()
        ),
        "n_worlds_with_matching_nan_patterns": int(report["nan_patterns_match"].sum()),
        "n_worlds_within_1e_12": int(report["within_1e_12"].sum()),
        "maximum_absolute_delta": float(report["maximum_absolute_delta"].max()),
        "all_worlds_equivalent": bool(report["within_1e_12"].all()),
        "workers": args.workers,
        "tolerance": TOLERANCE,
        "phase_a_truth_or_scorer_opened": False,
        "dev_or_locked_opened": False,
        "raw_inputs": (
            "six observed chromatin focal tables plus same-world frozen "
            "assay-local background embedding"
        ),
        "frozen_phase_a_manifest_sha256": _sha256(phase_a_manifest_path),
        "frozen_phase_a_candidate_sha256": expected_candidate_hash,
        "production_source_sha256": _sha256(
            Path(__file__).with_name("crossfit_hierarchical_collider_v4.py")
        ),
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    outputs = (report_path, detailed_path, summary_path)
    manifest = {
        "schema_version": SCHEMA,
        "source_sha256": _sha256(Path(__file__)),
        "production_source_sha256": summary["production_source_sha256"],
        "input_phase_a_manifest_sha256": summary[
            "frozen_phase_a_manifest_sha256"
        ],
        "input_phase_a_candidate_sha256": expected_candidate_hash,
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "truth_or_scorer_opened": False,
        "dev_or_locked_opened": False,
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["all_worlds_equivalent"]:
        raise SystemExit("At least one raw production fit differs from frozen Phase A")


if __name__ == "__main__":
    main()
