#!/usr/bin/env python3
"""Scorer-only exact-test ceiling for shared-state-subspace v4.

This diagnostic substitutes the simulator's true per-assay latent state for
the discovery embedding, but otherwise calls the unchanged v4 feature-score,
B=999 synchronized wild null, assay admission, within-assay BH, view-support,
and order-2 conjunction code.  It is deliberately scorer-only and every
artifact it writes is forbidden as a discovery input.

The five-world runner enforces a phase barrier: all state-score candidates are
written, hashed, and reloaded before role assignments or world parameters are
opened for scoring.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import cell_state_loading_groups as loading_groups
import score_shared_state_subspace_v4 as scorer
import shared_state_subspace_v4 as discovery


SCHEMA_VERSION = "shared_state_subspace_v4.true_state_ceiling.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _world_seed(world: Path) -> int:
    try:
        return int(world.name.removeprefix("world_seed"))
    except ValueError as exc:
        raise ValueError(f"World path lacks a numeric world_seed suffix: {world}") from exc


def _validate_world(world_path: str | Path) -> tuple[Path, Path, Path]:
    lexical = Path(world_path).absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"World must be a direct directory: {lexical}")
    world = lexical.resolve(strict=True)
    discovery_root = world / "discovery"
    scorer_root = world / "scorer"
    if (
        discovery_root.is_symlink()
        or scorer_root.is_symlink()
        or not discovery_root.is_dir()
        or not scorer_root.is_dir()
    ):
        raise ValueError("World must contain direct discovery/ and scorer/ domains")
    manifest = json.loads((scorer_root / "manifest.json").read_text())
    if manifest.get("scorer_only") is not True or (
        manifest.get("must_not_be_read_by_discovery") is not True
    ):
        raise ValueError("Scorer manifest does not enforce the information firewall")
    return world, discovery_root, scorer_root


def _write_true_state_embedding(
    discovery_root: Path, scorer_root: Path, embedding_path: Path
) -> dict[str, str]:
    rows: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    focal_paths = sorted(discovery_root.glob("observed_focal__*.tsv"))
    if len(focal_paths) != 7:
        raise ValueError("Exact ceiling requires RNA and six chromatin focal tables")
    for focal_path in focal_paths:
        modality = focal_path.name.removeprefix("observed_focal__").removesuffix(
            ".tsv"
        )
        focal = pd.read_csv(focal_path, sep="\t", keep_default_na=False)
        if set(focal["modality"].astype(str)) != {modality}:
            raise ValueError(f"Focal modality mismatch for {modality}")
        state_path = scorer_root / f"cell_ground_truth__{modality}.npz"
        if state_path.is_symlink() or not state_path.is_file():
            raise ValueError(f"Missing direct true-state archive for {modality}")
        with np.load(state_path, allow_pickle=False) as payload:
            state_ids = np.asarray(payload["cell_ids"], dtype=str)
            state = np.asarray(payload["latent_state"], dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 3 or not np.isfinite(state).all():
            raise ValueError(f"True state must be finite n-by-3 for {modality}")
        if len(np.unique(state_ids)) != len(state_ids):
            raise ValueError(f"True-state cell IDs repeat for {modality}")
        focal_ids = focal["cell"].astype(str).to_numpy()
        index = {cell: row for row, cell in enumerate(state_ids)}
        if set(focal_ids) != set(state_ids):
            raise ValueError(f"Focal/true-state cell sets differ for {modality}")
        aligned = state[np.asarray([index[cell] for cell in focal_ids], dtype=np.int64)]
        frame = pd.DataFrame(
            {
                "modality": modality,
                "cell": focal_ids,
                "integration__dim0": aligned[:, 0],
                "integration__dim1": aligned[:, 1],
                "integration__dim2": aligned[:, 2],
                "integration_uncertainty": np.zeros(len(aligned), dtype=np.float64),
            }
        )
        rows.append(frame)
        input_hashes[f"discovery/{focal_path.name}"] = _sha256(focal_path)
        input_hashes[f"scorer/{state_path.name}"] = _sha256(state_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=False)
    pd.concat(rows, ignore_index=True).to_csv(
        embedding_path, sep="\t", index=False
    )
    return dict(sorted(input_hashes.items()))


def _phase_a_one(world_path: str, output_root: str) -> dict[str, Any]:
    world, discovery_root, scorer_root = _validate_world(world_path)
    seed = _world_seed(world)
    world_output = Path(output_root) / "phase_a" / f"world_seed{seed}"
    if world_output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-A world {world_output}")
    world_output.mkdir(parents=True)
    embedding_path = (
        world_output / "privileged_state_coordinates" / "analysis_embeddings.tsv"
    )
    input_hashes = _write_true_state_embedding(
        discovery_root, scorer_root, embedding_path
    )
    config = discovery.SharedStateSubspaceConfig()
    raw_inputs = loading_groups.load_loading_inputs(discovery_root, embedding_path)
    inputs, assay_scales = discovery._rotation_invariant_local_inputs(raw_inputs)
    observed = discovery._fit_observed_evidence(inputs, config)
    reference, null = discovery.estimate_conditional_wild_null(inputs, config=config)
    result = discovery.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership_path = world_output / "subspace_feature_membership.tsv"
    hyperedge_path = world_output / "typed_subspace_hyperedges.tsv"
    provenance_path = world_output / "ceiling_provenance.json"
    result.feature_membership.to_csv(membership_path, sep="\t", index=False)
    result.typed_subspace_hyperedges.to_csv(hyperedge_path, sep="\t", index=False)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "information_domain": "scorer_only_true_state_ceiling",
        "discovery_eligible": False,
        "must_not_be_consumed_by_discovery": True,
        "true_latent_state_used": True,
        "simulator_roles_used_in_phase_a": False,
        "truth_graph_used_in_phase_a": False,
        "exact_v4_score_and_selection_code_used": True,
        "cross_assay_rows_paired": False,
        "config": asdict(config),
        "assay_coordinate_scales": assay_scales,
        "input_hashes": input_hashes,
        "embedded_discovery_provenance": result.provenance,
    }
    _write_json(provenance_path, provenance)
    artifacts = {
        str(path.relative_to(world_output)): _sha256(path)
        for path in sorted(world_output.rglob("*"))
        if path.is_file()
    }
    phase_manifest = {
        "schema_version": f"{SCHEMA_VERSION}.phase_a",
        "world_seed": seed,
        "world_path": str(world),
        "candidate_frozen_before_role_read": True,
        "artifacts": artifacts,
        "config_sha256": _canonical_sha256(asdict(config)),
        "discovery_source_sha256": _sha256(Path(discovery.__file__)),
        "loading_source_sha256": _sha256(Path(loading_groups.__file__)),
    }
    phase_manifest["manifest_payload_sha256"] = _canonical_sha256(phase_manifest)
    _write_json(world_output / "phase_a_manifest.json", phase_manifest)
    return phase_manifest


def _aggregate_metrics(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    numeric_mean_fields = (
        "subspace_member_precision",
        "subspace_member_recall",
        "chromatin_state_responsive_member_recall",
        "P_member_recall",
        "Q_member_recall",
        "S_member_recall",
        "C_state_responsive_member_recall",
        "sentinel_null_inclusion_rate",
        "minimum_chromatin_assay_member_recall",
    )
    return {
        "n_worlds": len(metrics),
        "membership_world_successes": int(
            sum(int(row["membership_class_success"]) for row in metrics)
        ),
        "hyperedge_world_successes": int(
            sum(int(row["hyperedge_class_success"]) for row in metrics)
        ),
        "subspace_world_successes": int(
            sum(int(row["subspace_world_success"]) for row in metrics)
        ),
        "total_selected_true_members": int(
            sum(int(row["n_selected_true_subspace_features"]) for row in metrics)
        ),
        "total_true_members": int(
            sum(int(row["n_truth_state_responsive_features"]) for row in metrics)
        ),
        "total_selected_sentinels": int(
            sum(int(row["n_selected_sentinel_null_features"]) for row in metrics)
        ),
        "total_sentinels": int(
            sum(int(row["n_truth_sentinel_null_features"]) for row in metrics)
        ),
        "means": {
            field: float(np.mean([float(row[field]) for row in metrics]))
            for field in numeric_mean_fields
        },
    }


def run_ceiling(
    worlds: Sequence[Path], output_root: Path, *, workers: int = 1
) -> dict[str, Any]:
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    world_strings = [str(Path(world).absolute()) for world in worlds]
    if len({_world_seed(Path(world)) for world in worlds}) != len(worlds):
        raise ValueError("World seeds must be unique")
    phase_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_phase_a_one, world, str(output_root)): world
            for world in world_strings
        }
        for future in as_completed(futures):
            phase_records.append(future.result())
    phase_records.sort(key=lambda row: int(row["world_seed"]))
    freeze = {
        "schema_version": f"{SCHEMA_VERSION}.global_pre_role_freeze",
        "phase_a_complete_for_all_worlds": True,
        "roles_or_truth_graph_read_before_freeze": False,
        "worlds": phase_records,
        "runner_source_sha256": _sha256(Path(__file__)),
    }
    freeze["freeze_payload_sha256"] = _canonical_sha256(freeze)
    freeze_path = output_root / "global_pre_role_freeze.json"
    _write_json(freeze_path, freeze)

    # Reload and hash-check every candidate before any role table is opened.
    loaded: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for record in phase_records:
        seed = int(record["world_seed"])
        world_output = output_root / "phase_a" / f"world_seed{seed}"
        for relative, digest in record["artifacts"].items():
            if _sha256(world_output / relative) != digest:
                raise ValueError(f"Frozen Phase-A bytes changed for seed {seed}: {relative}")
        membership = pd.read_csv(
            world_output / "subspace_feature_membership.tsv",
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        hyperedge = pd.read_csv(
            world_output / "typed_subspace_hyperedges.tsv",
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        loaded[seed] = (membership, hyperedge)

    metrics_rows: list[dict[str, Any]] = []
    for world_path in worlds:
        world, _, scorer_root = _validate_world(world_path)
        seed = _world_seed(world)
        world_parameters = json.loads((scorer_root / "world_parameters.json").read_text())
        roles = pd.read_csv(
            scorer_root / "role_assignments.tsv", sep="\t", keep_default_na=False
        )
        membership, hyperedge = loaded[seed]
        score = scorer.score_shared_state_subspace(
            world_parameters=world_parameters,
            role_assignments=roles,
            feature_membership=membership,
            typed_subspace_hyperedges=hyperedge,
        )
        score_root = output_root / "scores" / f"world_seed{seed}"
        score_root.mkdir(parents=True)
        _write_json(score_root / "metrics.json", score.metrics)
        score.feature_scores.to_csv(
            score_root / "feature_scores.tsv", sep="\t", index=False
        )
        score.assay_scores.to_csv(
            score_root / "assay_scores.tsv", sep="\t", index=False
        )
        score.hyperedge_scores.to_csv(
            score_root / "hyperedge_scores.tsv", sep="\t", index=False
        )
        metrics_rows.append(score.metrics)
    metrics_rows.sort(key=lambda row: int(row["world_seed"]))
    summary = {
        "schema_version": f"{SCHEMA_VERSION}.summary",
        "information_domain": "scorer_only_true_state_ceiling",
        "discovery_eligible": False,
        "global_pre_role_freeze_sha256": _sha256(freeze_path),
        "per_world": metrics_rows,
        "aggregate": _aggregate_metrics(metrics_rows),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_ceiling(args.world, args.out_root, workers=args.workers)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
