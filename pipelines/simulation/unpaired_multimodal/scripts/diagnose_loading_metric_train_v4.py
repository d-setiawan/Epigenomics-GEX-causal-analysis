#!/usr/bin/env python3
"""TRAIN-only loading-geometry and observable-exclusion diagnostic.

Phase A refits all six deterministic cell-loading views against the already
frozen assay-local background embedding.  It then freezes four candidate
projections before opening any scorer artifact:

* covariance-response geometry with the existing observable exclusions;
* covariance-response geometry with selected collider centers only;
* raw coefficient geometry with the existing observable exclusions; and
* raw coefficient geometry with selected collider centers only.

Cross-assay loading-group matching is disabled in every projection.  Phase B
starts only after every requested world has a complete, content-addressed
candidate manifest.  It reads TRAIN role assignments to score exact assay-local
P--S pairs and projects the already-recorded per-world reliability gates after
replacing only the loading-pair gate.

This is a diagnostic, not a fresh validation block.  It must never be pointed
at DEV or LOCKED worlds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmark_ca_boss_v4 import state_group_exclusions
from cell_state_loading_groups import (
    LoadingGroupConfig,
    estimate_loading_groups,
    load_loading_inputs,
)
from consensus_ca_boss_v4 import direct_collider_children
from loading_group_projection_v4 import (
    LoadingEvidenceView,
    LoadingProjectionConfig,
    estimate_loading_group_projection,
    make_evidence_view,
)
from structure_aware_collider_motif import ColliderMotifResult


TRAIN_SEEDS = tuple(range(33_001, 33_025))
VIEW_COUNT = 6
METRICS = ("covariance_response", "raw_coefficient")
POLICIES = ("existing_observable", "selected_centers_only")
VARIANTS = tuple(
    f"{metric}__{policy}" for metric in METRICS for policy in POLICIES
)
ALGORITHM_SOURCES = (
    "benchmark_ca_boss_v4.py",
    "cell_state_loading_groups.py",
    "consensus_ca_boss_v4.py",
    "loading_group_projection_v4.py",
    "structure_aware_collider_motif.py",
)


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _sha256(root / name) for name in ALGORITHM_SOURCES}


def _payload_hashes(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and str(path.relative_to(root)) not in excluded
    }


def _assert_train_only_path(path: Path, label: str) -> None:
    folded = str(path.resolve(strict=False)).casefold()
    if "dev" in folded or "locked" in folded:
        raise ValueError(f"Refusing {label} path containing DEV/LOCKED: {path}")


def _candidate_root(artifact_world: Path) -> Path:
    candidates = sorted(
        path
        for path in artifact_world.glob("v4_*")
        if path.is_dir() and (path / "collider_motif").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one frozen v4 candidate in {artifact_world}, found {len(candidates)}"
        )
    return candidates[0]


def _read_tsv_allow_empty(path: Path, columns: Sequence[str] = ()) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def _frozen_motif(candidate: Path) -> ColliderMotifResult:
    root = candidate / "collider_motif"
    return ColliderMotifResult(
        direct_candidates=_read_tsv_allow_empty(root / "direct_candidates.tsv"),
        demoted_or_unresolved=_read_tsv_allow_empty(
            root / "demoted_or_unresolved.tsv",
            (
                "source_assay",
                "node1",
                "node2",
                "component_id",
                "was_base_selected",
                "reason",
            ),
        ),
        component_diagnostics=_read_tsv_allow_empty(
            root / "component_diagnostics.tsv"
        ),
        wedge_diagnostics=_read_tsv_allow_empty(root / "wedge_diagnostics.tsv"),
        aggregated_evidence=_read_tsv_allow_empty(
            root / "aggregated_vote_evidence.tsv"
        ),
        metadata=json.loads((root / "motif_provenance.json").read_text()),
    )


def _observable_exclusion_policies(candidate: Path) -> dict[str, tuple[str, ...]]:
    consensus = pd.read_csv(candidate / "consensus_edges.tsv", sep="\t")
    motif = _frozen_motif(candidate)
    centers = direct_collider_children(consensus)
    existing = state_group_exclusions(consensus, motif)
    if not set(centers).issubset(existing):
        raise ValueError("Existing observable exclusions omitted a selected collider center")
    return {
        "existing_observable": tuple(sorted(existing)),
        "selected_centers_only": tuple(sorted(centers)),
    }


def _loading_config(view_dir: Path) -> LoadingGroupConfig:
    payload = json.loads((view_dir / "cell_loading_provenance.json").read_text())
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing frozen loading config in {view_dir}")
    allowed = {field.name for field in fields(LoadingGroupConfig)}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown loading config fields in {view_dir}: {unknown}")
    return LoadingGroupConfig(**config)


def _projection_config(candidate: Path) -> LoadingProjectionConfig:
    payload = json.loads(
        (
            candidate
            / "loading_group_projection"
            / "projection_provenance.json"
        ).read_text()
    )
    frozen = payload.get("config")
    if not isinstance(frozen, dict):
        raise ValueError(f"Missing frozen projection config in {candidate}")
    allowed = {field.name for field in fields(LoadingProjectionConfig)}
    values = {key: value for key, value in frozen.items() if key in allowed}
    values["allow_cross_assay_group_matching"] = False
    return LoadingProjectionConfig(**values)


def _raw_geometry_view(view: LoadingEvidenceView) -> LoadingEvidenceView:
    pairs = view.pair_evidence.copy()
    required = {
        "raw_coefficient_loading_cosine",
        "absolute_raw_coefficient_loading_cosine",
        "loading_stability_support",
        "geometric_mean_partial_state_fit",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Current loading evidence lacks raw-geometry columns: {missing}")
    pairs["loading_cosine"] = pairs["raw_coefficient_loading_cosine"]
    pairs["absolute_loading_cosine"] = pairs[
        "absolute_raw_coefficient_loading_cosine"
    ]
    pairs["pair_state_score"] = (
        pairs["absolute_loading_cosine"].clip(0.0, 1.0)
        * pairs["loading_stability_support"].clip(lower=0.0)
        * pairs["geometric_mean_partial_state_fit"].clip(lower=0.0)
    )
    loadings = view.loadings.drop(
        columns=[
            column
            for column in (
                "response_metric_coefficient",
                "marginal_response_metric_coefficient",
            )
            if column in view.loadings
        ]
    )
    return make_evidence_view(
        view_id=view.view_id,
        feature_models=view.feature_models,
        loadings=loadings,
        pair_evidence=pairs,
        input_hashes=view.input_hashes,
    )


def _write_projection(root: Path, result: Any) -> None:
    root.mkdir(parents=True, exist_ok=False)
    outputs = {
        "feature_consensus.tsv": result.feature_consensus,
        "within_group_candidates.tsv": result.within_group_candidates,
        "loading_groups.tsv": result.groups,
        "loading_group_membership.tsv": result.group_membership,
        "group_match_evidence.tsv": result.group_match_evidence,
        "typed_group_hyperedges.tsv": result.typed_hyperedges,
        "sparse_pair_projection.tsv": result.sparse_pair_projection,
    }
    for filename, frame in outputs.items():
        frame.to_csv(root / filename, sep="\t", index=False)
    _write_json(root / "projection_provenance.json", result.provenance)


def _phase_a_world(
    seed: int,
    worlds_root: str,
    artifacts_root: str,
    candidates_root: str,
) -> dict[str, Any]:
    worlds = Path(worlds_root)
    artifacts = Path(artifacts_root)
    output_root = Path(candidates_root)
    world = worlds / f"world_seed{seed}"
    artifact_world = artifacts / f"world_seed{seed}"
    discovery = world / "discovery"
    embedding = (
        artifact_world
        / "assay_local_state_embedding_v4"
        / "analysis_embeddings.tsv"
    )
    if not discovery.is_dir() or not embedding.is_file() or embedding.is_symlink():
        raise FileNotFoundError(f"Incomplete discovery/local embedding for seed {seed}")
    candidate = _candidate_root(artifact_world)
    policies = _observable_exclusion_policies(candidate)
    projection_config = _projection_config(candidate)
    view_dirs = sorted(
        path
        for path in artifact_world.glob("cells*_seed*")
        if path.is_dir() and (path / "cell_loading_provenance.json").is_file()
    )
    if len(view_dirs) != VIEW_COUNT:
        raise ValueError(f"Seed {seed} has {len(view_dirs)} loading views, expected six")

    inputs = load_loading_inputs(discovery, embedding)
    covariance_views: list[LoadingEvidenceView] = []
    raw_views: list[LoadingEvidenceView] = []
    source_hashes = _source_hashes()
    temp_parent = output_root / ".phase_a_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"world_seed{seed}__", dir=temp_parent))
    final = output_root / f"world_seed{seed}"
    try:
        views_out = temporary / "views"
        views_out.mkdir()
        for view_dir in view_dirs:
            config = _loading_config(view_dir)
            fitted = estimate_loading_groups(inputs, config)
            covariance = make_evidence_view(
                view_id=view_dir.name,
                feature_models=fitted.feature_models,
                loadings=fitted.loadings,
                pair_evidence=fitted.pair_evidence,
            )
            raw = _raw_geometry_view(covariance)
            covariance_views.append(covariance)
            raw_views.append(raw)
            target = views_out / view_dir.name
            target.mkdir()
            fitted.feature_models.to_csv(
                target / "feature_models.tsv", sep="\t", index=False
            )
            fitted.loadings.to_csv(target / "loadings.tsv", sep="\t", index=False)
            fitted.pair_evidence.to_csv(
                target / "pair_evidence.tsv", sep="\t", index=False
            )
            _write_json(
                target / "fit_provenance.json",
                {
                    **fitted.provenance,
                    "source_frozen_view": str(view_dir.resolve(strict=True)),
                    "source_frozen_view_config_sha256": _sha256(
                        view_dir / "cell_loading_provenance.json"
                    ),
                },
            )

        metric_views = {
            "covariance_response": covariance_views,
            "raw_coefficient": raw_views,
        }
        variant_summaries: dict[str, Any] = {}
        for metric, views in metric_views.items():
            for policy, excluded in policies.items():
                variant = f"{metric}__{policy}"
                result = estimate_loading_group_projection(
                    views,
                    projection_config,
                    excluded_features=excluded,
                )
                if not result.group_match_evidence.empty:
                    raise RuntimeError(
                        f"Cross-assay group matching was not suppressed for {variant}"
                    )
                _write_projection(temporary / "variants" / variant, result)
                selected = result.groups.loc[
                    result.groups["selection_status"].astype(str).eq("selected")
                    & result.groups["group_kind"]
                    .astype(str)
                    .eq("within_assay_coloading")
                ]
                variant_summaries[variant] = {
                    "n_selected_within_assay_groups": int(len(selected)),
                    "selected_modalities": sorted(selected["modality"].astype(str)),
                    "cross_assay_group_matching_enabled": False,
                }

        manifest = {
            "schema_version": "ca_boss_v4.loading_metric_train_diagnostic.phase_a_world.v1",
            "seed": seed,
            "train_only": True,
            "scorer_or_roles_read": False,
            "all_four_variants_built_before_truth_open": True,
            "n_refit_views": len(covariance_views),
            "view_ids": [view.view_id for view in covariance_views],
            "assay_local_embedding": str(embedding.resolve(strict=True)),
            "assay_local_embedding_sha256": _sha256(embedding),
            "focal_discovery": str(discovery.resolve(strict=True)),
            "focal_input_sha256": dict(inputs.input_hashes),
            "frozen_observable_candidate": str(candidate.resolve(strict=True)),
            "exclusion_policies": policies,
            "projection_config": asdict(projection_config),
            "variant_summaries": variant_summaries,
            "algorithm_source_sha256": source_hashes,
            "payload_sha256": _payload_hashes(temporary),
        }
        _write_json(temporary / "candidate_manifest.json", manifest)
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite Phase-A candidate {final}")
        os.replace(temporary, final)
        return {
            "seed": seed,
            "candidate_manifest_sha256": _sha256(final / "candidate_manifest.json"),
            "variant_summaries": variant_summaries,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_world_candidate(path: Path, seed: int) -> dict[str, Any]:
    manifest_path = path / "candidate_manifest.json"
    if path.is_symlink() or not path.is_dir() or not manifest_path.is_file():
        raise ValueError(f"Seed {seed} lacks a direct complete candidate directory")
    payload = json.loads(manifest_path.read_text())
    if payload.get("seed") != seed or payload.get("scorer_or_roles_read") is not False:
        raise ValueError(f"Seed {seed} has an invalid Phase-A manifest")
    expected = payload.get("payload_sha256")
    observed = _payload_hashes(path, exclude={"candidate_manifest.json"})
    if expected != observed:
        raise ValueError(f"Seed {seed} Phase-A candidate payload changed")
    return payload


def _selected_pairs(variant_root: Path) -> list[tuple[str, str, str]]:
    groups = pd.read_csv(variant_root / "loading_groups.tsv", sep="\t")
    membership = pd.read_csv(
        variant_root / "loading_group_membership.tsv", sep="\t"
    )
    selected = groups.loc[
        groups["selection_status"].astype(str).eq("selected")
        & groups["group_kind"].astype(str).eq("within_assay_coloading")
    ]
    rows: list[tuple[str, str, str]] = []
    for group in selected.itertuples(index=False):
        members = sorted(
            membership.loc[
                membership["loading_group"].astype(str).eq(str(group.loading_group)),
                "feature",
            ].astype(str)
        )
        if len(members) != 2:
            raise ValueError(
                f"Selected within-assay group {group.loading_group} is not a pair"
            )
        rows.append((str(group.modality), members[0], members[1]))
    return rows


def _score_variant(
    seed: int,
    variant: str,
    variant_root: Path,
    roles: pd.DataFrame,
    existing_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    role_lookup = roles.set_index("node")["role"].astype(str).to_dict()
    mark_lookup = roles.set_index("node")["mark"].astype(str).to_dict()
    pairs = _selected_pairs(variant_root)
    pair_rows: list[dict[str, Any]] = []
    true_count = 0
    sentinel_members = 0
    for modality, left, right in pairs:
        left_role = role_lookup.get(left, "UNKNOWN")
        right_role = role_lookup.get(right, "UNKNOWN")
        exact = {left_role, right_role} == {"P", "S"}
        true_count += int(exact)
        sentinel_members += int(left_role in {"N1", "N2"}) + int(
            right_role in {"N1", "N2"}
        )
        pair_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "modality": modality,
                "node1": left,
                "role1": left_role,
                "node2": right,
                "role2": right_role,
                "mark1": mark_lookup.get(left, "UNKNOWN"),
                "mark2": mark_lookup.get(right, "UNKNOWN"),
                "exact_ps_pair": exact,
            }
        )
    n_selected = len(pairs)
    precision = true_count / n_selected if n_selected else 1.0
    recall = true_count / 6.0
    zero_conflation = bool(true_count == n_selected and sentinel_members == 0)
    gate_checks = dict(
        existing_metrics["preregistered_validation"]["gate_checks"]
    )
    gate_checks["loading_pair_precision_at_least_0_80"] = precision >= 0.80
    gate_checks["loading_zero_sentinel_or_member_conflation"] = zero_conflation
    direct_count = int(existing_metrics["direct_score"]["n_predicted"])
    cross_count = int(existing_metrics["cross_assay_state_hyperedge"]["n_selected"])
    projected_density = direct_count + n_selected + cross_count
    gate_checks["expanded_focal_density_at_most_20"] = projected_density <= 20
    projected_reliable = bool(all(bool(value) for value in gate_checks.values()))
    return (
        {
            "seed": seed,
            "variant": variant,
            "n_selected_pairs": n_selected,
            "n_true_ps_pairs": true_count,
            "n_false_pairs": n_selected - true_count,
            "n_sentinel_members": sentinel_members,
            "pair_precision": precision,
            "pair_recall": recall,
            "chromatin_assay_coverage": n_selected / 6.0,
            "zero_sentinel_or_member_conflation": zero_conflation,
            "projected_expanded_focal_density": projected_density,
            "projected_world_reliability_success": projected_reliable,
            "original_world_reliability_success": bool(
                existing_metrics["preregistered_validation"][
                    "world_reliability_success"
                ]
            ),
        },
        pair_rows,
    )


def _pair_metric_rows(
    seed: int, candidate_world: Path, roles: pd.DataFrame
) -> list[dict[str, Any]]:
    role_lookup = roles.set_index("node")["role"].astype(str).to_dict()
    rows: list[dict[str, Any]] = []
    for path in sorted((candidate_world / "views").glob("*/pair_evidence.tsv")):
        frame = pd.read_csv(path, sep="\t")
        frame = frame.loc[frame["same_assay"].astype(bool)]
        frame = frame.loc[frame["modality1"].astype(str).ne("rna")]
        for row in frame.itertuples(index=False):
            left_role = role_lookup.get(str(row.node1), "UNKNOWN")
            right_role = role_lookup.get(str(row.node2), "UNKNOWN")
            rows.append(
                {
                    "seed": seed,
                    "view_id": path.parent.name,
                    "modality": str(row.modality1),
                    "pair_key": str(row.pair_key),
                    "node1": str(row.node1),
                    "role1": left_role,
                    "node2": str(row.node2),
                    "role2": right_role,
                    "exact_ps_pair": {left_role, right_role} == {"P", "S"},
                    "covariance_absolute_similarity": float(
                        row.absolute_loading_cosine
                    ),
                    "raw_coefficient_absolute_similarity": float(
                        row.absolute_raw_coefficient_loading_cosine
                    ),
                    "covariance_minus_raw": float(
                        row.absolute_loading_cosine
                        - row.absolute_raw_coefficient_loading_cosine
                    ),
                }
            )
    return rows


def _aggregate(
    per_world: pd.DataFrame, pair_metrics: pd.DataFrame
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": "ca_boss_v4.loading_metric_train_diagnostic.aggregate.v1",
        "train_only": True,
        "diagnostic_block_reused_for_algorithm_development": True,
        "fresh_validation_claimed": False,
        "n_worlds": int(per_world["seed"].nunique()),
        "variants": {},
        "pair_geometry": {},
    }
    for variant, frame in per_world.groupby("variant", sort=True):
        total_selected = int(frame["n_selected_pairs"].sum())
        total_true = int(frame["n_true_ps_pairs"].sum())
        mean_precision = float(frame["pair_precision"].mean())
        mean_recall = float(frame["pair_recall"].mean())
        reliable = int(frame["projected_world_reliability_success"].sum())
        output["variants"][str(variant)] = {
            "total_selected_pairs": total_selected,
            "total_true_ps_pairs": total_true,
            "micro_pair_precision": (
                total_true / total_selected if total_selected else 1.0
            ),
            "micro_pair_recall": total_true / (6.0 * len(frame)),
            "mean_world_pair_precision": mean_precision,
            "mean_world_pair_recall": mean_recall,
            "n_zero_conflation_worlds": int(
                frame["zero_sentinel_or_member_conflation"].sum()
            ),
            "n_projected_reliable_worlds": reliable,
            "mean_loading_precision_at_least_0_90": mean_precision >= 0.90,
            "mean_loading_recall_at_least_0_80": mean_recall >= 0.80,
            "at_least_22_of_24_projected_reliable": reliable >= 22,
        }
    labels = pair_metrics["exact_ps_pair"].astype(int).to_numpy()
    for metric in (
        "covariance_absolute_similarity",
        "raw_coefficient_absolute_similarity",
    ):
        values = pair_metrics[metric].to_numpy(dtype=float)
        true_values = values[labels == 1]
        false_values = values[labels == 0]
        output["pair_geometry"][metric] = {
            "roc_auc_all_views": float(roc_auc_score(labels, values)),
            "true_ps_median": float(np.median(true_values)),
            "true_ps_q10": float(np.quantile(true_values, 0.10)),
            "false_pair_median": float(np.median(false_values)),
            "false_pair_q90": float(np.quantile(false_values, 0.90)),
        }
    return output


def _phase_b_score(
    seeds: Sequence[int],
    worlds_root: Path,
    candidates_root: Path,
    scores_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    per_world_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        candidate_world = candidates_root / f"world_seed{seed}"
        _verify_world_candidate(candidate_world, seed)
        roles_path = worlds_root / f"world_seed{seed}" / "scorer" / "role_assignments.tsv"
        metrics_path = scores_root / f"world_seed{seed}" / "metrics.json"
        if not roles_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Seed {seed} lacks TRAIN scoring artifacts")
        roles = pd.read_csv(roles_path, sep="\t")
        existing_metrics = json.loads(metrics_path.read_text())
        for variant in VARIANTS:
            row, pairs = _score_variant(
                seed,
                variant,
                candidate_world / "variants" / variant,
                roles,
                existing_metrics,
            )
            per_world_rows.append(row)
            selected_rows.extend(pairs)
        pair_metric_rows.extend(_pair_metric_rows(seed, candidate_world, roles))

    score_dir = output_root / "train_truth_scores"
    score_dir.mkdir(exist_ok=False)
    per_world = pd.DataFrame(per_world_rows).sort_values(["seed", "variant"])
    selected = pd.DataFrame(selected_rows).sort_values(
        ["seed", "variant", "modality"]
    )
    pair_metrics = pd.DataFrame(pair_metric_rows).sort_values(
        ["seed", "view_id", "modality", "pair_key"]
    )
    per_world.to_csv(score_dir / "per_world_variant_metrics.tsv", sep="\t", index=False)
    selected.to_csv(score_dir / "selected_pair_roles.tsv", sep="\t", index=False)
    pair_metrics.to_csv(score_dir / "pair_geometry_by_view.tsv", sep="\t", index=False)
    aggregate = _aggregate(per_world, pair_metrics)
    aggregate["phase_a_candidates_unchanged_after_scoring"] = all(
        _verify_world_candidate(candidates_root / f"world_seed{seed}", seed)
        for seed in seeds
    )
    aggregate["scorer_files_read_only_after_global_phase_a_freeze"] = True
    _write_json(score_dir / "aggregate.json", aggregate)
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Any]:
    worlds_root = args.worlds_root.resolve(strict=True)
    artifacts_root = args.artifacts_root.resolve(strict=True)
    scores_root = args.scores_root.resolve(strict=True)
    output_root = args.out_root.absolute()
    for path, label in (
        (worlds_root, "worlds"),
        (artifacts_root, "artifacts"),
        (scores_root, "scores"),
        (output_root, "output"),
    ):
        _assert_train_only_path(path, label)
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and non-empty")
    if any(seed not in TRAIN_SEEDS for seed in seeds):
        raise ValueError("This diagnostic is restricted to TRAIN seeds 33001--33024")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output {output_root}")
    output_root.mkdir(parents=True)
    candidates_root = output_root / "phase_a_candidates"
    candidates_root.mkdir()
    source_hashes_before = _source_hashes()

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _phase_a_world,
                seed,
                str(worlds_root),
                str(artifacts_root),
                str(candidates_root),
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            results.append(result)
            print(f"Phase A complete: seed {seed}", flush=True)

    manifests: dict[str, str] = {}
    for seed in seeds:
        candidate_world = candidates_root / f"world_seed{seed}"
        payload = _verify_world_candidate(candidate_world, seed)
        if payload.get("algorithm_source_sha256") != source_hashes_before:
            raise RuntimeError(f"Algorithm source drifted during Phase A for seed {seed}")
        manifests[f"world_seed{seed}"] = _sha256(
            candidate_world / "candidate_manifest.json"
        )
    if _source_hashes() != source_hashes_before:
        raise RuntimeError("Algorithm source changed during Phase A; refusing truth access")
    freeze = {
        "schema_version": "ca_boss_v4.loading_metric_train_diagnostic.global_freeze.v1",
        "train_only": True,
        "requested_seeds": list(seeds),
        "all_requested_candidates_complete_before_truth_open": True,
        "candidate_manifest_sha256": manifests,
        "algorithm_source_sha256": source_hashes_before,
        "variants": list(VARIANTS),
        "cross_assay_group_matching_suppressed": True,
    }
    _write_json(output_root / "global_phase_a_freeze.json", freeze)

    aggregate = _phase_b_score(
        seeds,
        worlds_root,
        candidates_root,
        scores_root,
        output_root,
    )
    run_summary = {
        "schema_version": "ca_boss_v4.loading_metric_train_diagnostic.run.v1",
        "train_only": True,
        "requested_seeds": list(seeds),
        "n_worlds": len(seeds),
        "workers": args.workers,
        "phase_a_global_freeze_sha256": _sha256(
            output_root / "global_phase_a_freeze.json"
        ),
        "aggregate_sha256": _sha256(
            output_root / "train_truth_scores" / "aggregate.json"
        ),
        "aggregate": aggregate,
    }
    _write_json(output_root / "run_summary.json", run_summary)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 2:
        parser.error("--workers must be 1 or 2 on the available host")
    return args


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps(_json_ready(summary["aggregate"]), indent=2, sort_keys=True))
