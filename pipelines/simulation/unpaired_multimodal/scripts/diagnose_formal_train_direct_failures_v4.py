#!/usr/bin/env python3
"""Freeze and label direct-layer evidence for formal TRAIN 36001--36024.

This diagnostic is deliberately restricted to the now-open formal TRAIN block.
It first loads and writes every observed candidate/evidence table for all 24
worlds.  Only after the Phase-A manifest is complete does it open direct truth
to label motifs.  It cannot inspect DEV, LOCKED, or proposed seeds >=37001.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from benchmark_ca_boss_v4 import _direct_consensus_for_motif
from consensus_ca_boss_v4 import consensus_typed_graph
from observed_collider_structure import (
    ObservedColliderStructureConfig,
    apply_observed_collider_refinements,
    select_observed_collider_refinements,
)
from structure_aware_collider_motif import select_unique_collider_motifs


TRAIN_START = 36001
TRAIN_END = 36024
SEEDS = tuple(range(TRAIN_START, TRAIN_END + 1))
RUN_ROOT = Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_36001_36024")
WORLD_ROOT = Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_36001_36024")


def _read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except EmptyDataError:
        return pd.DataFrame()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _canonical(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _candidate_directory(world_artifact: Path) -> Path:
    candidates = sorted(
        path
        for path in world_artifact.glob("v4_*")
        if path.is_dir() and (path / "collider_motif").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one formal candidate in {world_artifact}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _required(path: Path) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _load_phase_a_world(seed: int, run_root: Path) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("Diagnostic is restricted to TRAIN seeds 36001-36024")
    world_artifact = run_root / "artifacts" / f"world_seed{seed}"
    candidate = _candidate_directory(world_artifact)
    motif = candidate / "collider_motif"
    motif_names = {
        "current_direct": "direct_candidates.tsv",
        "unresolved": "demoted_or_unresolved.tsv",
        "components": "component_diagnostics.tsv",
        "wedges": "wedge_diagnostics.tsv",
        "aggregated_votes": "aggregated_vote_evidence.tsv",
        "observed_decisions": "observed_structure_decisions.tsv",
        "observed_wedge_views": "observed_structure_wedge_view_evidence.tsv",
        "observed_wedges": "observed_structure_wedge_evidence.tsv",
        "observed_completion_views": (
            "observed_structure_completion_view_evidence.tsv"
        ),
        "observed_completions": "observed_structure_completion_evidence.tsv",
    }
    frames: dict[str, pd.DataFrame] = {}
    input_paths: list[Path] = []
    for key, name in motif_names.items():
        path = _required(motif / name)
        frames[key] = _read_tsv(path)
        input_paths.append(path)

    extra_paths = {
        "subspace_features": (
            world_artifact
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        ),
        "layered_features": (
            world_artifact
            / "layered_state_evidence_v4"
            / "layered_feature_evidence.tsv"
        ),
        "loading_feature_consensus": (
            candidate / "loading_group_projection" / "feature_consensus.tsv"
        ),
        "structure_anchor_pairs": (
            candidate
            / "structure_anchored_loading_groups"
            / "structure_anchor_pair_evidence.tsv"
        ),
        "structure_anchor_decisions": (
            candidate
            / "structure_anchored_loading_groups"
            / "assay_decisions.tsv"
        ),
    }
    for key, path in extra_paths.items():
        path = _required(path)
        frames[key] = _read_tsv(path)
        input_paths.append(path)

    views = sorted(
        path
        for path in world_artifact.glob("cells*_seed*")
        if path.is_dir() and not path.is_symlink()
    )
    if len(views) != 6:
        raise ValueError(f"Expected six evidence views for seed {seed}")
    view_names = {
        "raw_pairs": "raw_pair_evidence.tsv",
        "cell_features": "cell_loading_feature_models.tsv",
        "axis_features": "axis_loading_feature_models.tsv",
        "cell_pairs": "cell_loading_pair_evidence.tsv",
        "axis_pairs": "axis_loading_pair_evidence.tsv",
        "hierarchical_pairs": "hierarchical_sparse_edge_evidence.tsv",
    }
    view_frames: dict[str, list[pd.DataFrame]] = {
        key: [] for key in view_names
    }
    nested_frames: list[pd.DataFrame] = []
    for index, view in enumerate(views):
        for key, name in view_names.items():
            path = _required(view / name)
            frame = _read_tsv(path)
            frame.insert(0, "view_id", view.name)
            view_frames[key].append(frame)
            input_paths.append(path)
        nested_path = _required(candidate / f"nested_{index}_edges.tsv")
        nested = _read_tsv(nested_path)
        nested.insert(0, "view_id", view.name)
        nested_frames.append(nested)
        input_paths.append(nested_path)

    vote_paths = [
        _required(
            world_artifact
            / "collider_motif_evidence"
            / f"beta_binomial_vote{vote}.tsv"
        )
        for vote in range(6)
    ]
    beta_votes = pd.concat(
        [_read_tsv(path) for path in vote_paths], ignore_index=True, sort=False
    )
    input_paths.extend(vote_paths)

    return {
        "seed": seed,
        "candidate_name": candidate.name,
        **frames,
        **{
            key: pd.concat(value, ignore_index=True, sort=False)
            for key, value in view_frames.items()
        },
        "nested_edges": pd.concat(nested_frames, ignore_index=True, sort=False),
        "beta_votes": beta_votes,
        "input_hashes": {
            str(path.relative_to(run_root)): _sha256(path)
            for path in sorted(input_paths)
        },
    }


def _stack(held: list[dict[str, Any]], key: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for world in held:
        frame = world[key].copy()
        seed = int(world["seed"])
        if "seed" in frame.columns:
            if not frame.empty and not frame["seed"].astype(int).eq(seed).all():
                raise ValueError(f"Phase-A frame {key!r} has a seed mismatch")
        else:
            frame.insert(0, "seed", seed)
        frames.append(frame)
    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True, sort=False)
    return frames[0].iloc[0:0].copy()


def _feature_assay_map(world: dict[str, Any]) -> dict[str, str]:
    features = world["cell_features"]
    if features.empty or not {"feature", "modality"}.issubset(features.columns):
        raise ValueError(f"Feature evidence missing for seed {world['seed']}")
    pairs = features[["feature", "modality"]].drop_duplicates()
    if pairs["feature"].astype(str).duplicated().any():
        raise ValueError(f"Feature changes modality for seed {world['seed']}")
    return {
        str(row.feature): str(row.modality)
        for row in pairs.itertuples(index=False)
    }


def _pair_records(
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, Any]]:
    required = {"node1", "node2", *value_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"Pair evidence is missing {sorted(required - set(frame.columns))}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        edge = _canonical(str(record["node1"]), str(record["node2"]))
        if edge in result:
            raise ValueError(f"Duplicate pair evidence for {edge}")
        result[edge] = record
    return result


def _selected_nested_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    selected = frame.copy()
    if "selected" in selected.columns:
        selected = selected.loc[selected["selected"].astype(bool)]
    if "edge_type" in selected.columns:
        selected = selected.loc[
            selected["edge_type"].astype(str).eq("within_assay_direct_candidate")
        ]
    if not {"node1", "node2"}.issubset(selected.columns):
        raise ValueError("Nested direct evidence lacks endpoints")
    return {
        _canonical(str(row.node1), str(row.node2))
        for row in selected.itertuples(index=False)
    }


def _conditional_outer_r(outer: float, arm1: float, arm2: float) -> float:
    values = np.clip(np.asarray([outer, arm1, arm2], dtype=float), -1 + 1e-9, 1 - 1e-9)
    denominator = np.sqrt(max((1 - values[1] ** 2) * (1 - values[2] ** 2), 1e-18))
    return float(np.clip((values[0] - values[1] * values[2]) / denominator, -1, 1))


def _completion_targets(world: dict[str, Any]) -> list[dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    decisions = world["observed_decisions"]
    if not decisions.empty and {"action", "component_id"}.issubset(decisions.columns):
        for row in decisions.loc[
            decisions["action"].astype(str).eq(
                "complete_dual_supported_unshielded_anchor"
            )
        ].to_dict(orient="records"):
            targets[str(row["component_id"])] = {
                "source_assay": str(row["source_assay"]),
                "component_id": str(row["component_id"]),
                "anchor_node1": str(row["anchor_node1"]),
                "anchor_node2": str(row["anchor_node2"]),
                "target_source": "existing_completion_decision",
            }
    components = world["components"]
    unresolved = world["unresolved"]
    for component in components.loc[
        components["status"].astype(str).eq("unresolved_no_complete_wedge")
    ].to_dict(orient="records"):
        component_id = str(component["component_id"])
        edges = unresolved.loc[
            unresolved["component_id"].astype(str).eq(component_id)
        ]
        if len(edges) != 1:
            continue
        edge = edges.iloc[0]
        targets[component_id] = {
            "source_assay": str(component["source_assay"]),
            "component_id": component_id,
            "anchor_node1": str(edge["node1"]),
            "anchor_node2": str(edge["node2"]),
            "target_source": "unresolved_one_arm",
        }
    return [targets[key] for key in sorted(targets)]


def _enumerate_completion_repairs(
    world: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(world["seed"])
    assay_map = _feature_assay_map(world)
    features_by_assay: dict[str, list[str]] = {}
    for feature, assay in assay_map.items():
        features_by_assay.setdefault(assay, []).append(feature)
    features_by_assay = {
        assay: sorted(set(features)) for assay, features in features_by_assay.items()
    }
    aggregate_map = _pair_records(
        world["aggregated_votes"],
        value_columns=(
            "selected_vote_fraction",
            "mean_empirical_p_positive",
            "mean_pooled_log_odds",
        ),
    )
    base_views = {
        str(view_id): _selected_nested_pairs(frame)
        for view_id, frame in world["nested_edges"].groupby("view_id", sort=True)
    }
    base_counts: dict[tuple[str, str], int] = {}
    for pairs in base_views.values():
        for edge in pairs:
            base_counts[edge] = base_counts.get(edge, 0) + 1
    residual_maps = {
        str(view_id): _pair_records(frame, value_columns=("residual_r",))
        for view_id, frame in world["raw_pairs"].groupby("view_id", sort=True)
    }
    loading_maps = {
        str(view_id): _pair_records(
            frame,
            value_columns=(
                "absolute_loading_cosine",
                "absolute_state_response_correlation",
                "minimum_loading_norm",
                "loading_stability_support",
                "loading_empirical_null_p",
            ),
        )
        for view_id, frame in world["cell_pairs"].groupby("view_id", sort=True)
    }
    if set(residual_maps) != set(loading_maps):
        raise ValueError(f"Residual/loading view mismatch for seed {seed}")
    subspace = world["subspace_features"].set_index("feature", drop=False)
    summary_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    for target in _completion_targets(world):
        assay = target["source_assay"]
        anchor = _canonical(target["anchor_node1"], target["anchor_node2"])
        anchor_evidence = aggregate_map[anchor]
        anchor_rescue = bool(
            float(anchor_evidence["selected_vote_fraction"]) >= 5 / 6
            and float(anchor_evidence["mean_empirical_p_positive"]) <= 0.020
            and float(anchor_evidence["mean_pooled_log_odds"]) >= 1.0
        )
        for center, anchor_outer in (anchor, anchor[::-1]):
            for new_parent in features_by_assay[assay]:
                if new_parent in anchor:
                    continue
                new_arm = _canonical(center, new_parent)
                outer_pair = _canonical(anchor_outer, new_parent)
                if new_arm not in aggregate_map or outer_pair not in aggregate_map:
                    continue
                residual_records: list[dict[str, Any]] = []
                for view_id in sorted(residual_maps):
                    pairs = residual_maps[view_id]
                    required_edges = (anchor, new_arm, outer_pair)
                    if any(edge not in pairs for edge in required_edges):
                        raise ValueError(
                            f"Residual view {view_id} lacks a completion triple"
                        )
                    anchor_r = float(pairs[anchor]["residual_r"])
                    new_arm_r = float(pairs[new_arm]["residual_r"])
                    outer_r = float(pairs[outer_pair]["residual_r"])
                    conditional_r = _conditional_outer_r(
                        outer_r, anchor_r, new_arm_r
                    )
                    record = {
                        "seed": seed,
                        **target,
                        "center": center,
                        "anchor_outer": anchor_outer,
                        "new_parent": new_parent,
                        "view_id": view_id,
                        "anchor_residual_r": anchor_r,
                        "new_arm_residual_r": new_arm_r,
                        "outer_pair_residual_r": outer_r,
                        "conditional_outer_residual_r": conditional_r,
                        "collider_separation_score": -conditional_r,
                        "collider_opening_delta": outer_r - conditional_r,
                    }
                    residual_records.append(record)
                    view_rows.append(record)
                separation = [
                    float(record["collider_separation_score"])
                    for record in residual_records
                ]
                opening = [
                    float(record["collider_opening_delta"])
                    for record in residual_records
                ]
                for sibling in features_by_assay[assay]:
                    if sibling in {center, anchor_outer, new_parent}:
                        continue
                    sibling_parent_a = _canonical(sibling, anchor_outer)
                    sibling_parent_b = _canonical(sibling, new_parent)
                    center_sibling = _canonical(center, sibling)
                    cosine_a: list[float] = []
                    cosine_b: list[float] = []
                    state_correlation_a: list[float] = []
                    state_correlation_b: list[float] = []
                    minimum_norm_a: list[float] = []
                    minimum_norm_b: list[float] = []
                    for view_id in sorted(loading_maps):
                        pairs = loading_maps[view_id]
                        if sibling_parent_a not in pairs or sibling_parent_b not in pairs:
                            raise ValueError(
                                f"Loading view {view_id} lacks sibling pairs"
                            )
                        row_a = pairs[sibling_parent_a]
                        row_b = pairs[sibling_parent_b]
                        cosine_a.append(float(row_a["absolute_loading_cosine"]))
                        cosine_b.append(float(row_b["absolute_loading_cosine"]))
                        state_correlation_a.append(
                            float(row_a["absolute_state_response_correlation"])
                        )
                        state_correlation_b.append(
                            float(row_b["absolute_state_response_correlation"])
                        )
                        minimum_norm_a.append(float(row_a["minimum_loading_norm"]))
                        minimum_norm_b.append(float(row_b["minimum_loading_norm"]))
                    median_a = float(np.median(cosine_a))
                    median_b = float(np.median(cosine_b))
                    if (median_a, anchor_outer) >= (median_b, new_parent):
                        linked_parent = anchor_outer
                        linked_cosine = median_a
                        other_cosine = median_b
                        linked_minimum_norm = float(np.median(minimum_norm_a))
                        linked_state_correlation = float(
                            np.median(state_correlation_a)
                        )
                    else:
                        linked_parent = new_parent
                        linked_cosine = median_b
                        other_cosine = median_a
                        linked_minimum_norm = float(np.median(minimum_norm_b))
                        linked_state_correlation = float(
                            np.median(state_correlation_b)
                        )
                    relative_margin = (
                        (linked_cosine - other_cosine) / max(linked_cosine, 1e-12)
                    )
                    sibling_state = subspace.loc[sibling]
                    new_evidence = aggregate_map[new_arm]
                    outer_evidence = aggregate_map[outer_pair]
                    center_sibling_evidence = aggregate_map[center_sibling]
                    direct_min_lor = min(
                        float(anchor_evidence["mean_pooled_log_odds"]),
                        float(new_evidence["mean_pooled_log_odds"]),
                    )
                    direct_sum_lor = float(
                        anchor_evidence["mean_pooled_log_odds"]
                    ) + float(new_evidence["mean_pooled_log_odds"])
                    forbidden_positive_lor = max(
                        0.0,
                        float(outer_evidence["mean_pooled_log_odds"]),
                        float(center_sibling_evidence["mean_pooled_log_odds"]),
                    )
                    summary_rows.append(
                        {
                            "seed": seed,
                            **target,
                            "center": center,
                            "anchor_outer": anchor_outer,
                            "new_parent": new_parent,
                            "sibling": sibling,
                            "anchor_base_view_count": base_counts.get(anchor, 0),
                            "new_arm_base_view_count": base_counts.get(new_arm, 0),
                            "outer_pair_base_view_count": base_counts.get(
                                outer_pair, 0
                            ),
                            "anchor_rescue_selected": anchor_rescue,
                            "anchor_mean_lor": float(
                                anchor_evidence["mean_pooled_log_odds"]
                            ),
                            "new_arm_mean_lor": float(
                                new_evidence["mean_pooled_log_odds"]
                            ),
                            "new_arm_vote_fraction": float(
                                new_evidence["selected_vote_fraction"]
                            ),
                            "outer_pair_mean_lor": float(
                                outer_evidence["mean_pooled_log_odds"]
                            ),
                            "center_sibling_mean_lor": float(
                                center_sibling_evidence["mean_pooled_log_odds"]
                            ),
                            "direct_min_lor": direct_min_lor,
                            "direct_sum_lor": direct_sum_lor,
                            "forbidden_positive_lor": forbidden_positive_lor,
                            "joint_net_lor": direct_sum_lor
                            - forbidden_positive_lor,
                            "median_collider_separation": float(
                                np.median(separation)
                            ),
                            "minimum_collider_separation": float(
                                np.min(separation)
                            ),
                            "positive_separation_view_support": float(
                                np.mean(np.asarray(separation) > 0)
                            ),
                            "median_collider_opening_delta": float(
                                np.median(opening)
                            ),
                            "positive_opening_view_support": float(
                                np.mean(np.asarray(opening) > 0)
                            ),
                            "linked_parent": linked_parent,
                            "median_linked_parent_sibling_cosine": linked_cosine,
                            "median_other_parent_sibling_cosine": other_cosine,
                            "relative_linked_parent_margin": relative_margin,
                            "median_linked_pair_minimum_loading_norm": (
                                linked_minimum_norm
                            ),
                            "median_linked_pair_state_response_correlation": (
                                linked_state_correlation
                            ),
                            "sibling_formal_state_selected": str(
                                sibling_state["selection_status"]
                            )
                            == "selected",
                            "sibling_conditional_wald_view_support": float(
                                sibling_state["conditional_wald_view_support"]
                            ),
                            "sibling_feature_global_maxT_gate": bool(
                                sibling_state[
                                    "feature_global_maxT_confirmation_gate"
                                ]
                            ),
                            "sibling_assay_global_maxT_gate": bool(
                                sibling_state["assay_global_maxT_admission_gate"]
                            ),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(view_rows)


def _fixed_repair_proposals(
    world: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the frozen beta-centrality and half-view dominance proposals."""

    seed = int(world["seed"])
    assay_map = _feature_assay_map(world)
    features_by_assay: dict[str, list[str]] = {}
    for feature, assay in assay_map.items():
        features_by_assay.setdefault(assay, []).append(feature)
    features_by_assay = {
        assay: sorted(set(features)) for assay, features in features_by_assay.items()
    }
    evidence_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    aggregated = world["aggregated_votes"]
    beta_votes = world["beta_votes"]
    decisions = world["observed_decisions"]
    completion = (
        decisions.loc[
            decisions["action"].astype(str).eq(
                "complete_dual_supported_unshielded_anchor"
            )
        ]
        if not decisions.empty and "action" in decisions.columns
        else pd.DataFrame()
    )
    for row in completion.to_dict(orient="records"):
        assay = str(row["source_assay"])
        component_id = str(row["component_id"])
        anchor = _canonical(str(row["anchor_node1"]), str(row["anchor_node2"]))
        assay_evidence = aggregated.loc[
            aggregated["source_assay"].astype(str).eq(assay)
        ]
        centrality: dict[str, tuple[float, float]] = {}
        for endpoint in anchor:
            incident = assay_evidence.loc[
                assay_evidence["node1"].astype(str).eq(endpoint)
                | assay_evidence["node2"].astype(str).eq(endpoint)
            ]
            centrality[endpoint] = (
                float(incident["selected_vote_fraction"].sum()),
                float(incident["mean_pooled_log_odds"].sum()),
            )
        selected_center = max(
            anchor,
            key=lambda endpoint: (
                centrality[endpoint][0],
                centrality[endpoint][1],
                endpoint,
            ),
        )
        anchor_outer = anchor[1] if selected_center == anchor[0] else anchor[0]
        current_center = str(row["center"])
        candidate_features = [
            feature for feature in features_by_assay[assay] if feature not in anchor
        ]
        winner_counts = {feature: 0 for feature in candidate_features}
        contrasts = {feature: [] for feature in candidate_features}
        assay_votes = beta_votes.loc[
            beta_votes["source_assay"].astype(str).eq(assay)
        ]
        for vote, vote_frame in assay_votes.groupby("vote", sort=True):
            vote_map = {
                _canonical(str(item["node1"]), str(item["node2"])): float(
                    item["pooled_log_odds"]
                )
                for item in vote_frame.to_dict(orient="records")
            }
            vote_contrasts = {
                feature: vote_map[_canonical(selected_center, feature)]
                - vote_map[_canonical(anchor_outer, feature)]
                for feature in candidate_features
            }
            winner = max(
                candidate_features,
                key=lambda feature: (vote_contrasts[feature], feature),
            )
            winner_counts[winner] += 1
            for feature, contrast in vote_contrasts.items():
                contrasts[feature].append(float(contrast))
        selected_new_parent = max(
            candidate_features,
            key=lambda feature: (
                winner_counts[feature],
                float(np.mean(contrasts[feature])),
                feature,
            ),
        )
        for feature in candidate_features:
            evidence_rows.append(
                {
                    "seed": seed,
                    "source_assay": assay,
                    "component_id": component_id,
                    "proposal_family": "completion_beta_centrality",
                    "current_center": current_center,
                    "selected_center": selected_center,
                    "anchor_outer": anchor_outer,
                    "candidate_new_parent": feature,
                    "anchor_node1": anchor[0],
                    "anchor_node2": anchor[1],
                    "anchor_node1_selected_fraction_centrality": centrality[
                        anchor[0]
                    ][0],
                    "anchor_node1_lor_centrality": centrality[anchor[0]][1],
                    "anchor_node2_selected_fraction_centrality": centrality[
                        anchor[1]
                    ][0],
                    "anchor_node2_lor_centrality": centrality[anchor[1]][1],
                    "center_changes": selected_center != current_center,
                    "contrast_winner_view_count": winner_counts[feature],
                    "n_contrast_views": len(contrasts[feature]),
                    "mean_center_vs_outer_lor_contrast": float(
                        np.mean(contrasts[feature])
                    ),
                    "minimum_center_vs_outer_lor_contrast": float(
                        np.min(contrasts[feature])
                    ),
                    "selected_new_parent": feature == selected_new_parent,
                }
            )
        if (
            selected_center != current_center
            and winner_counts[selected_new_parent] >= 4
        ):
            decision_rows.append(
                {
                    "seed": seed,
                    "source_assay": assay,
                    "component_id": component_id,
                    "action": "reorient_completion_by_assay_beta_centrality",
                    "center": selected_center,
                    "outer1": anchor_outer,
                    "outer2": selected_new_parent,
                    "winner_view_count": winner_counts[selected_new_parent],
                    "n_views": len(contrasts[selected_new_parent]),
                }
            )

    components = world["components"]
    observed_wedges = world["observed_wedges"]
    for component in components.loc[~components["accepted"].astype(bool)].to_dict(
        orient="records"
    ):
        component_id = str(component["component_id"])
        candidates = observed_wedges.loc[
            observed_wedges["component_id"].astype(str).eq(component_id)
        ]
        if len(candidates) <= 1 or candidates["center"].astype(str).nunique() != 1:
            continue
        maximum_base_arms = int(candidates["base_arm_count"].max())
        leaders = candidates.loc[
            candidates["base_arm_count"].astype(int).eq(maximum_base_arms)
        ]
        if maximum_base_arms != 2 or len(leaders) != 1:
            continue
        leader = leaders.iloc[0]
        runners = candidates.drop(index=leaders.index)
        runners_at_most_one_base = bool(
            (runners["base_arm_count"].astype(int) <= 1).all()
        )
        half_view_support = bool(
            int(leader["winner_view_count"]) * 2 >= int(leader["n_views"])
        )
        positive_pairwise_margin = bool(
            float(leader["pairwise_wedge_score"])
            > float(runners["pairwise_wedge_score"].max())
        )
        evidence_rows.append(
            {
                "seed": seed,
                "source_assay": str(component["source_assay"]),
                "component_id": component_id,
                "proposal_family": "rejected_same_center_dual_base_dominance",
                "current_center": component.get("best_center", ""),
                "selected_center": str(leader["center"]),
                "anchor_outer": str(leader["outer1"]),
                "candidate_new_parent": str(leader["outer2"]),
                "center_changes": False,
                "contrast_winner_view_count": int(leader["winner_view_count"]),
                "n_contrast_views": int(leader["n_views"]),
                "unique_two_base_arm_wedge": True,
                "runners_at_most_one_base_arm": runners_at_most_one_base,
                "half_view_support": half_view_support,
                "positive_pairwise_margin": positive_pairwise_margin,
                "selected_new_parent": True,
            }
        )
        if (
            runners_at_most_one_base
            and half_view_support
            and positive_pairwise_margin
        ):
            decision_rows.append(
                {
                    "seed": seed,
                    "source_assay": str(component["source_assay"]),
                    "component_id": component_id,
                    "action": "accept_rejected_unique_dual_base_half_view_wedge",
                    "center": str(leader["center"]),
                    "outer1": str(leader["outer1"]),
                    "outer2": str(leader["outer2"]),
                    "winner_view_count": int(leader["winner_view_count"]),
                    "n_views": int(leader["n_views"]),
                }
            )
    return pd.DataFrame(evidence_rows), pd.DataFrame(decision_rows)


def _optional_rescue_only_completion_proposals(
    world: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a unique rescue-only completion with unanimous observed support."""

    seed = int(world["seed"])
    assay_map = _feature_assay_map(world)
    features_by_assay: dict[str, list[str]] = {}
    for feature, assay in assay_map.items():
        features_by_assay.setdefault(assay, []).append(feature)
    features_by_assay = {
        assay: sorted(set(features)) for assay, features in features_by_assay.items()
    }
    aggregated_map = _pair_records(
        world["aggregated_votes"],
        value_columns=(
            "selected_vote_fraction",
            "mean_empirical_p_positive",
            "mean_pooled_log_odds",
        ),
    )
    base_views = {
        str(view_id): _selected_nested_pairs(frame)
        for view_id, frame in world["nested_edges"].groupby("view_id", sort=True)
    }
    loading_maps = {
        str(view_id): _pair_records(
            frame,
            value_columns=("loading_empirical_null_p", "absolute_loading_cosine"),
        )
        for view_id, frame in world["cell_pairs"].groupby("view_id", sort=True)
    }
    subspace = world["subspace_features"].set_index("feature", drop=False)
    beta_votes = world["beta_votes"]
    evidence_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    components = world["components"]
    unresolved = world["unresolved"]
    for component in components.loc[
        components["status"].astype(str).eq("unresolved_no_complete_wedge")
    ].to_dict(orient="records"):
        component_id = str(component["component_id"])
        assay = str(component["source_assay"])
        component_edges = unresolved.loc[
            unresolved["component_id"].astype(str).eq(component_id)
        ]
        if len(component_edges) != 1:
            continue
        edge_row = component_edges.iloc[0]
        anchor = _canonical(str(edge_row["node1"]), str(edge_row["node2"]))
        anchor_base_view_count = sum(
            anchor in pairs for pairs in base_views.values()
        )
        anchor_evidence = aggregated_map[anchor]
        anchor_rescue_selected = bool(
            float(anchor_evidence["selected_vote_fraction"]) >= 5 / 6
            and float(anchor_evidence["mean_empirical_p_positive"]) <= 0.020
            and float(anchor_evidence["mean_pooled_log_odds"]) >= 1.0
        )
        rescue_only_component = bool(
            anchor_rescue_selected and anchor_base_view_count < 4
        )
        passing: list[dict[str, Any]] = []
        for center, anchor_outer in (anchor, anchor[::-1]):
            for new_parent in features_by_assay[assay]:
                if new_parent in anchor:
                    continue
                new_arm = _canonical(center, new_parent)
                state_selected = bool(
                    str(subspace.loc[new_parent, "selection_status"]) == "selected"
                )
                loading_p_values = [
                    float(loading_maps[view_id][new_arm]["loading_empirical_null_p"])
                    for view_id in sorted(loading_maps)
                ]
                unanimous_loading_support = bool(
                    len(loading_p_values) == 6
                    and all(value <= 0.125 for value in loading_p_values)
                )
                assay_votes = beta_votes.loc[
                    beta_votes["source_assay"].astype(str).eq(assay)
                ]
                contrasts: list[float] = []
                for _, vote_frame in assay_votes.groupby("vote", sort=True):
                    vote_map = {
                        _canonical(str(item["node1"]), str(item["node2"])): float(
                            item["pooled_log_odds"]
                        )
                        for item in vote_frame.to_dict(orient="records")
                    }
                    contrasts.append(
                        vote_map[new_arm]
                        - vote_map[_canonical(anchor_outer, new_parent)]
                    )
                unanimous_positive_contrast = bool(
                    len(contrasts) == 6 and all(value > 0 for value in contrasts)
                )
                passes = bool(
                    rescue_only_component
                    and state_selected
                    and unanimous_loading_support
                    and unanimous_positive_contrast
                )
                record = {
                    "seed": seed,
                    "source_assay": assay,
                    "component_id": component_id,
                    "center": center,
                    "anchor_outer": anchor_outer,
                    "new_parent": new_parent,
                    "anchor_node1": anchor[0],
                    "anchor_node2": anchor[1],
                    "anchor_base_view_count": anchor_base_view_count,
                    "anchor_rescue_selected": anchor_rescue_selected,
                    "rescue_only_component": rescue_only_component,
                    "new_parent_formal_state_selected": state_selected,
                    "loading_p_le_0_125_view_support": int(
                        sum(value <= 0.125 for value in loading_p_values)
                    ),
                    "maximum_loading_empirical_null_p": max(loading_p_values),
                    "minimum_absolute_loading_cosine": min(
                        float(
                            loading_maps[view_id][new_arm][
                                "absolute_loading_cosine"
                            ]
                        )
                        for view_id in sorted(loading_maps)
                    ),
                    "positive_contrast_view_support": int(
                        sum(value > 0 for value in contrasts)
                    ),
                    "mean_direct_vs_outer_lor_contrast": float(
                        np.mean(contrasts)
                    ),
                    "minimum_direct_vs_outer_lor_contrast": float(
                        np.min(contrasts)
                    ),
                    "passes_all_candidate_gates": passes,
                }
                evidence_rows.append(record)
                if passes:
                    passing.append(record)
        for record in evidence_rows:
            if (
                record["seed"] == seed
                and record["component_id"] == component_id
            ):
                record["n_component_gate_passing_candidates"] = len(passing)
                record["unique_component_candidate"] = len(passing) == 1
        if len(passing) == 1:
            winner = passing[0]
            decision_rows.append(
                {
                    "seed": seed,
                    "source_assay": assay,
                    "component_id": component_id,
                    "action": "complete_unique_rescue_only_state_loading_contrast",
                    "center": str(winner["center"]),
                    "outer1": str(winner["anchor_outer"]),
                    "outer2": str(winner["new_parent"]),
                    "winner_view_count": int(
                        winner["positive_contrast_view_support"]
                    ),
                    "n_views": 6,
                }
            )
    return pd.DataFrame(evidence_rows), pd.DataFrame(decision_rows)


def _replay_production_observed_refinement(
    world: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Reconstruct and apply the production selector from frozen Phase-A inputs."""

    nested_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["nested_edges"].groupby(
            "view_id", sort=True
        )
    }
    preliminary = consensus_typed_graph(
        list(nested_views.values()), minimum_count=4
    )
    motif = select_unique_collider_motifs(
        _direct_consensus_for_motif(preliminary),
        world["beta_votes"],
    )
    residual_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["raw_pairs"].groupby("view_id", sort=True)
    }
    feature_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["cell_features"].groupby(
            "view_id", sort=True
        )
    }
    loading_pair_views = {
        str(view_id): frame.drop(columns="view_id").reset_index(drop=True)
        for view_id, frame in world["cell_pairs"].groupby("view_id", sort=True)
    }
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        world["beta_votes"],
        nested_views,
        residual_views,
        feature_views,
        ObservedColliderStructureConfig(),
        loading_pair_views=loading_pair_views,
        state_membership=world["subspace_features"],
    )
    applied = apply_observed_collider_refinements(motif, refinement)
    return {
        "production_direct": applied.direct_candidates,
        "production_unresolved": applied.demoted_or_unresolved,
        "production_components": applied.component_diagnostics,
        "production_wedges": applied.wedge_diagnostics,
        "production_repair_decisions": refinement.decisions,
        "production_wedge_view_evidence": refinement.wedge_view_evidence,
        "production_wedge_evidence": refinement.wedge_evidence,
        "production_completion_view_evidence": (
            refinement.completion_view_evidence
        ),
        "production_completion_evidence": refinement.completion_evidence,
    }


def _load_direct_truth_after_barrier(
    held: list[dict[str, Any]], world_root: Path
) -> tuple[pd.DataFrame, dict[int, set[tuple[str, str]]]]:
    rows: list[pd.DataFrame] = []
    truth_sets: dict[int, set[tuple[str, str]]] = {}
    for world in held:
        seed = int(world["seed"])
        path = _required(
            world_root / f"world_seed{seed}" / "scorer" / "truth_typed_edges.tsv"
        )
        truth = _read_tsv(path)
        required = {"truth_type", "parent", "child"}
        if not required.issubset(truth.columns):
            raise ValueError(f"Truth schema mismatch for seed {seed}")
        direct = truth.loc[
            truth["truth_type"].astype(str).eq("within_assay_direct")
        ].copy()
        direct.insert(0, "seed", seed)
        assay_map = _feature_assay_map(world)
        direct["source_assay"] = direct["child"].astype(str).map(assay_map)
        if direct["source_assay"].isna().any():
            raise ValueError(f"Could not assign truth assay for seed {seed}")
        rows.append(direct)
        truth_sets[seed] = {
            (str(row.parent), str(row.child))
            for row in direct.itertuples(index=False)
        }
    return pd.concat(rows, ignore_index=True, sort=False), truth_sets


def _load_roles_after_barrier(
    held: list[dict[str, Any]], world_root: Path
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for world in held:
        seed = int(world["seed"])
        path = _required(
            world_root / f"world_seed{seed}" / "scorer" / "role_assignments.tsv"
        )
        roles = _read_tsv(path)
        if not {"node", "mark", "role"}.issubset(roles.columns):
            raise ValueError(f"Role schema mismatch for seed {seed}")
        roles.insert(0, "seed", seed)
        roles["source_assay"] = "chrom_" + roles["mark"].astype(str)
        rows.append(roles)
    return pd.concat(rows, ignore_index=True, sort=False)


def _directed_edges_from_frame(frame: pd.DataFrame) -> set[tuple[str, str]]:
    directed: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        center = str(row.collider_center)
        left, right = str(row.node1), str(row.node2)
        if center == left:
            directed.add((right, left))
        elif center == right:
            directed.add((left, right))
        else:
            raise ValueError(f"Collider center is not an endpoint: {row}")
    return directed


def _current_directed_edges(world: dict[str, Any]) -> set[tuple[str, str]]:
    return _directed_edges_from_frame(world["current_direct"])


def _direct_edge_outcomes(
    held: list[dict[str, Any]],
    truth_sets: dict[int, set[tuple[str, str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_rows: list[dict[str, Any]] = []
    assay_rows: list[dict[str, Any]] = []
    for world in held:
        seed = int(world["seed"])
        assay_map = _feature_assay_map(world)
        predicted = _current_directed_edges(world)
        truth = truth_sets[seed]
        all_assays = sorted(set(assay_map.values()))
        for parent, child in sorted(predicted | truth):
            if (parent, child) in predicted and (parent, child) in truth:
                outcome = "true_positive"
            elif (parent, child) in predicted and (child, parent) in truth:
                outcome = "strict_reverse_prediction"
            elif (parent, child) in predicted:
                outcome = "false_prediction"
            elif (child, parent) in predicted:
                outcome = "truth_missed_by_strict_reverse"
            else:
                outcome = "missing_truth_edge"
            edge_rows.append(
                {
                    "seed": seed,
                    "source_assay": assay_map.get(child, assay_map.get(parent, "")),
                    "parent": parent,
                    "child": child,
                    "predicted": (parent, child) in predicted,
                    "truth": (parent, child) in truth,
                    "outcome": outcome,
                }
            )
        for assay in all_assays:
            assay_features = {
                feature for feature, value in assay_map.items() if value == assay
            }
            assay_predicted = {
                edge for edge in predicted if set(edge).issubset(assay_features)
            }
            assay_truth = {
                edge for edge in truth if set(edge).issubset(assay_features)
            }
            reverse_count = sum(
                edge not in assay_truth and (edge[1], edge[0]) in assay_truth
                for edge in assay_predicted
            )
            tp = len(assay_predicted & assay_truth)
            assay_rows.append(
                {
                    "seed": seed,
                    "source_assay": assay,
                    "n_truth": len(assay_truth),
                    "n_predicted": len(assay_predicted),
                    "tp": tp,
                    "strict_reverse_predictions": reverse_count,
                    "other_false_predictions": len(assay_predicted - assay_truth)
                    - reverse_count,
                    "fn": len(assay_truth - assay_predicted),
                    "exact": assay_predicted == assay_truth,
                }
            )
    return pd.DataFrame(edge_rows), pd.DataFrame(assay_rows)


def _directed_edges_after_decision_frames(
    world: dict[str, Any],
    decision_keys: tuple[str, ...],
) -> set[tuple[str, str]]:
    direct = world["current_direct"].copy()
    nonempty = [
        world[key]
        for key in decision_keys
        if key in world and not world[key].empty
    ]
    decisions = (
        pd.concat(nonempty, ignore_index=True, sort=False)
        if nonempty
        else pd.DataFrame()
    )
    if decisions.empty:
        return _current_directed_edges(world)
    if decisions["component_id"].astype(str).duplicated().any():
        duplicate_ids = sorted(
            decisions.loc[
                decisions["component_id"].astype(str).duplicated(keep=False),
                "component_id",
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "Multiple repair decisions target the same component: "
            f"{duplicate_ids}"
        )
    for decision in decisions.to_dict(orient="records"):
        component_id = str(decision["component_id"])
        direct = direct.loc[
            ~direct["component_id"].astype(str).eq(component_id)
        ].copy()
    repaired: set[tuple[str, str]] = set()
    for row in direct.itertuples(index=False):
        center = str(row.collider_center)
        left, right = str(row.node1), str(row.node2)
        parent = right if center == left else left
        repaired.add((parent, center))
    for decision in decisions.to_dict(orient="records"):
        center = str(decision["center"])
        repaired.add((str(decision["outer1"]), center))
        repaired.add((str(decision["outer2"]), center))
    return repaired


def _repaired_directed_edges(world: dict[str, Any]) -> set[tuple[str, str]]:
    """Apply only the frozen primary three-decision repair."""

    return _directed_edges_after_decision_frames(
        world, ("fixed_repair_decisions",)
    )


def _primary_plus_optional_directed_edges(
    world: dict[str, Any],
) -> set[tuple[str, str]]:
    """Apply the primary repair plus the separately audited optional rescue."""

    return _directed_edges_after_decision_frames(
        world,
        ("fixed_repair_decisions", "optional_repair_decisions"),
    )


def _direct_metric_row(
    seed: int,
    predicted: set[tuple[str, str]],
    truth: set[tuple[str, str]],
    prefix: str,
) -> dict[str, Any]:
    predicted_skeleton = {_canonical(*edge) for edge in predicted}
    truth_skeleton = {_canonical(*edge) for edge in truth}
    skeleton_tp = len(predicted_skeleton & truth_skeleton)
    strict_reverse = sum(
        edge not in truth and (edge[1], edge[0]) in truth for edge in predicted
    )
    skeleton_precision = skeleton_tp / max(len(predicted_skeleton), 1)
    skeleton_recall = skeleton_tp / max(len(truth_skeleton), 1)
    reliability = bool(
        skeleton_precision >= 0.80
        and skeleton_recall >= 0.90
        and strict_reverse == 0
    )
    return {
        "seed": seed,
        f"{prefix}_n_predicted": len(predicted_skeleton),
        f"{prefix}_n_true_predicted": skeleton_tp,
        f"{prefix}_n_false_predicted": len(predicted_skeleton - truth_skeleton),
        f"{prefix}_n_correctly_oriented": len(predicted & truth),
        f"{prefix}_n_strict_reverse": strict_reverse,
        f"{prefix}_skeleton_precision": skeleton_precision,
        f"{prefix}_skeleton_recall": skeleton_recall,
        f"{prefix}_exact_directed": predicted == truth,
        f"{prefix}_reliability_compatible": reliability,
    }


def _repair_metric_rows(
    held: list[dict[str, Any]],
    truth_sets: dict[int, set[tuple[str, str]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for world in held:
        seed = int(world["seed"])
        truth = truth_sets[seed]
        current = _current_directed_edges(world)
        repaired = _repaired_directed_edges(world)
        primary_plus_optional = _primary_plus_optional_directed_edges(world)
        production = _directed_edges_from_frame(world["production_direct"])
        current_row = _direct_metric_row(seed, current, truth, "current")
        repaired_row = _direct_metric_row(seed, repaired, truth, "repaired")
        optional_row = _direct_metric_row(
            seed,
            primary_plus_optional,
            truth,
            "primary_plus_optional",
        )
        production_row = _direct_metric_row(
            seed,
            production,
            truth,
            "production_replay",
        )
        rows.append(
            {
                **current_row,
                **{key: value for key, value in repaired_row.items() if key != "seed"},
                **{key: value for key, value in optional_row.items() if key != "seed"},
                **{
                    key: value
                    for key, value in production_row.items()
                    if key != "seed"
                },
                "n_fixed_repair_decisions": len(world["fixed_repair_decisions"]),
                "n_optional_repair_decisions": len(
                    world["optional_repair_decisions"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _truth_wedge_map(
    truth_direct: pd.DataFrame,
) -> dict[tuple[int, str], tuple[str, frozenset[str]]]:
    result: dict[tuple[int, str], tuple[str, frozenset[str]]] = {}
    for (seed, assay), group in truth_direct.groupby(
        ["seed", "source_assay"], sort=True
    ):
        children = set(group["child"].astype(str))
        if len(children) != 1 or len(group) != 2:
            raise ValueError(f"Truth is not a two-arm collider for {seed}, {assay}")
        result[(int(seed), str(assay))] = (
            next(iter(children)),
            frozenset(group["parent"].astype(str)),
        )
    return result


def _label_wedges(frame: pd.DataFrame, truth_direct: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    truth_map = _truth_wedge_map(truth_direct)
    labeled = frame.copy()
    records: list[dict[str, Any]] = []
    for row in labeled.to_dict(orient="records"):
        assay = row.get("source_assay")
        if assay is None or pd.isna(assay):
            component_id = str(row.get("component_id", ""))
            if "__component" not in component_id:
                raise ValueError("Wedge evidence lacks both assay and component assay")
            assay = component_id.split("__component", 1)[0]
        key = (int(row["seed"]), str(assay))
        true_center, true_parents = truth_map[key]
        if "outer1" in row:
            candidate_outers = frozenset(
                (str(row["outer1"]), str(row["outer2"]))
            )
        else:
            candidate_outers = frozenset(
                (str(row["anchor_outer"]), str(row["new_outer"]))
            )
        candidate_center = str(row["center"])
        records.append(
            {
                **row,
                "true_center": true_center,
                "true_parent1": sorted(true_parents)[0],
                "true_parent2": sorted(true_parents)[1],
                "center_is_true": candidate_center == true_center,
                "outer_set_is_true": candidate_outers == true_parents,
                "candidate_is_exact_true_wedge": (
                    candidate_center == true_center
                    and candidate_outers == true_parents
                ),
                "candidate_node_overlap": len(
                    ({candidate_center} | set(candidate_outers))
                    & ({true_center} | set(true_parents))
                ),
            }
        )
    return pd.DataFrame(records)


def _label_features(frame: pd.DataFrame, truth_direct: pd.DataFrame) -> pd.DataFrame:
    truth_map = _truth_wedge_map(truth_direct)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        key = (int(row["seed"]), str(row["modality"]))
        feature = str(row["feature"])
        if key not in truth_map:
            role = "not_applicable"
        else:
            true_center, true_parents = truth_map[key]
            role = (
                "center"
                if feature == true_center
                else "parent"
                if feature in true_parents
                else "null"
            )
        records.append({**row, "direct_truth_role": role})
    return pd.DataFrame(records)


def _label_completion_repairs(
    frame: pd.DataFrame,
    truth_direct: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    truth_map = _truth_wedge_map(truth_direct)
    role_map = {
        (int(row.seed), str(row.node)): str(row.role)
        for row in roles.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        seed = int(row["seed"])
        assay = str(row["source_assay"])
        true_center, true_parents = truth_map[(seed, assay)]
        candidate_parents = frozenset(
            (str(row["anchor_outer"]), str(row["new_parent"]))
        )
        sibling_role = role_map[(seed, str(row["sibling"]))]
        records.append(
            {
                **row,
                "true_center": true_center,
                "true_parent1": sorted(true_parents)[0],
                "true_parent2": sorted(true_parents)[1],
                "center_is_true": str(row["center"]) == true_center,
                "parent_set_is_true": candidate_parents == true_parents,
                "direct_wedge_is_exact": (
                    str(row["center"]) == true_center
                    and candidate_parents == true_parents
                ),
                "sibling_truth_role": sibling_role,
                "sibling_is_true_S": sibling_role == "S",
                "linked_parent_truth_role": role_map[
                    (seed, str(row["linked_parent"]))
                ],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_root = _required(RUN_ROOT / "global_pretruth_freeze.json").parent
    world_root = WORLD_ROOT

    # Phase A: load every observed table from every world before any truth path.
    held = [_load_phase_a_world(seed, run_root) for seed in SEEDS]
    if tuple(int(world["seed"]) for world in held) != SEEDS:
        raise RuntimeError("Global Phase-A candidate barrier is incomplete")
    for world in held:
        repair_summary, repair_views = _enumerate_completion_repairs(world)
        world["completion_repair_candidates"] = repair_summary
        world["completion_repair_views"] = repair_views
        fixed_evidence, fixed_decisions = _fixed_repair_proposals(world)
        world["fixed_repair_evidence"] = fixed_evidence
        world["fixed_repair_decisions"] = fixed_decisions
        optional_evidence, optional_decisions = (
            _optional_rescue_only_completion_proposals(world)
        )
        world["optional_repair_evidence"] = optional_evidence
        world["optional_repair_decisions"] = optional_decisions
        world.update(_replay_production_observed_refinement(world))

    phase_a_keys = (
        "current_direct",
        "unresolved",
        "components",
        "wedges",
        "aggregated_votes",
        "observed_decisions",
        "observed_wedge_views",
        "observed_wedges",
        "observed_completion_views",
        "observed_completions",
        "subspace_features",
        "layered_features",
        "loading_feature_consensus",
        "structure_anchor_pairs",
        "structure_anchor_decisions",
        "raw_pairs",
        "cell_features",
        "axis_features",
        "cell_pairs",
        "axis_pairs",
        "hierarchical_pairs",
        "nested_edges",
        "beta_votes",
        "completion_repair_candidates",
        "completion_repair_views",
        "fixed_repair_evidence",
        "fixed_repair_decisions",
        "optional_repair_evidence",
        "optional_repair_decisions",
        "production_direct",
        "production_unresolved",
        "production_components",
        "production_wedges",
        "production_repair_decisions",
        "production_wedge_view_evidence",
        "production_wedge_evidence",
        "production_completion_view_evidence",
        "production_completion_evidence",
    )
    phase_a_outputs: dict[str, Path] = {}
    phase_a_frames: dict[str, pd.DataFrame] = {}
    for key in phase_a_keys:
        frame = _stack(held, key)
        path = args.out_dir / f"phase_a_{key}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        phase_a_outputs[key] = path
        phase_a_frames[key] = frame
    input_hash_path = args.out_dir / "phase_a_input_hashes.json"
    _write_json(
        input_hash_path,
        {
            f"world_seed{world['seed']}": world["input_hashes"]
            for world in held
        },
    )
    phase_a_manifest_path = args.out_dir / "phase_a_manifest.json"
    executable_dependencies = [
        Path(__file__),
        Path(__file__).with_name("observed_collider_structure.py"),
        Path(__file__).with_name("benchmark_ca_boss_v4.py"),
        Path(__file__).with_name("consensus_ca_boss_v4.py"),
        Path(__file__).with_name("structure_aware_collider_motif.py"),
        Path(__file__).with_name("structure_aware_collider_rescue.py"),
    ]
    phase_a_manifest = {
        "block": "formal_train_36001_36024",
        "seeds": list(SEEDS),
        "n_worlds": len(held),
        "all_candidates_loaded_before_any_truth_open": True,
        "dev_locked_or_seed_ge_37001_opened": False,
        "phase_a_output_sha256": {
            path.name: _sha256(path)
            for path in [*phase_a_outputs.values(), input_hash_path]
        },
        "source_sha256": _sha256(Path(__file__)),
        "executable_dependency_sha256": {
            path.name: _sha256(path) for path in executable_dependencies
        },
    }
    _write_json(phase_a_manifest_path, phase_a_manifest)

    # Phase B: only now open direct truth and label the already-frozen tables.
    truth_direct, truth_sets = _load_direct_truth_after_barrier(held, world_root)
    roles = _load_roles_after_barrier(held, world_root)
    edge_outcomes, assay_outcomes = _direct_edge_outcomes(held, truth_sets)
    repair_metrics = _repair_metric_rows(held, truth_sets)
    labeled_wedges = _label_wedges(phase_a_frames["wedges"], truth_direct)
    labeled_observed_wedges = _label_wedges(
        phase_a_frames["observed_wedges"], truth_direct
    )
    labeled_observed_completions = _label_wedges(
        phase_a_frames["observed_completions"], truth_direct
    )
    labeled_cell_features = _label_features(
        phase_a_frames["cell_features"], truth_direct
    )
    labeled_axis_features = _label_features(
        phase_a_frames["axis_features"], truth_direct
    )
    labeled_completion_repairs = _label_completion_repairs(
        phase_a_frames["completion_repair_candidates"], truth_direct, roles
    )
    labeled_fixed_repair_decisions = _label_wedges(
        phase_a_frames["fixed_repair_decisions"], truth_direct
    )
    labeled_optional_repair_decisions = _label_wedges(
        phase_a_frames["optional_repair_decisions"], truth_direct
    )
    labeled_production_repair_decisions = _label_wedges(
        phase_a_frames["production_repair_decisions"], truth_direct
    )
    labeled = {
        "truth_direct": truth_direct,
        "role_assignments": roles,
        "direct_edge_outcomes": edge_outcomes,
        "assay_outcomes": assay_outcomes,
        "fixed_repair_metrics": repair_metrics,
        "labeled_wedges": labeled_wedges,
        "labeled_observed_wedges": labeled_observed_wedges,
        "labeled_observed_completions": labeled_observed_completions,
        "labeled_cell_features": labeled_cell_features,
        "labeled_axis_features": labeled_axis_features,
        "labeled_completion_repairs": labeled_completion_repairs,
        "labeled_fixed_repair_decisions": labeled_fixed_repair_decisions,
        "labeled_optional_repair_decisions": (
            labeled_optional_repair_decisions
        ),
        "labeled_production_repair_decisions": (
            labeled_production_repair_decisions
        ),
    }
    labeled_paths: list[Path] = []
    for key, frame in labeled.items():
        path = args.out_dir / f"phase_b_{key}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        labeled_paths.append(path)
    failed_assays = assay_outcomes.loc[~assay_outcomes["exact"]]
    summary = {
        "block": "formal_train_36001_36024_development_history",
        "all_candidates_loaded_and_frozen_before_any_truth_open": True,
        "n_worlds": len(SEEDS),
        "n_exact_worlds": int(
            assay_outcomes.groupby("seed")["exact"].all().sum()
        ),
        "failed_worlds": sorted(failed_assays["seed"].astype(int).unique()),
        "failed_assays": failed_assays.to_dict(orient="records"),
        "total_strict_reverse_predictions": int(
            assay_outcomes["strict_reverse_predictions"].sum()
        ),
        "total_other_false_predictions": int(
            assay_outcomes["other_false_predictions"].sum()
        ),
        "total_false_negatives": int(assay_outcomes["fn"].sum()),
        "current_reliability_compatible_worlds": int(
            repair_metrics["current_reliability_compatible"].sum()
        ),
        "repaired_reliability_compatible_worlds": int(
            repair_metrics["repaired_reliability_compatible"].sum()
        ),
        "current_exact_directed_worlds": int(
            repair_metrics["current_exact_directed"].sum()
        ),
        "repaired_exact_directed_worlds": int(
            repair_metrics["repaired_exact_directed"].sum()
        ),
        "repaired_total_false_predicted": int(
            repair_metrics["repaired_n_false_predicted"].sum()
        ),
        "repaired_total_strict_reverse": int(
            repair_metrics["repaired_n_strict_reverse"].sum()
        ),
        "primary_plus_optional_reliability_compatible_worlds": int(
            repair_metrics[
                "primary_plus_optional_reliability_compatible"
            ].sum()
        ),
        "primary_plus_optional_exact_directed_worlds": int(
            repair_metrics["primary_plus_optional_exact_directed"].sum()
        ),
        "primary_plus_optional_total_false_predicted": int(
            repair_metrics["primary_plus_optional_n_false_predicted"].sum()
        ),
        "primary_plus_optional_total_strict_reverse": int(
            repair_metrics["primary_plus_optional_n_strict_reverse"].sum()
        ),
        "production_replay_reliability_compatible_worlds": int(
            repair_metrics[
                "production_replay_reliability_compatible"
            ].sum()
        ),
        "production_replay_exact_directed_worlds": int(
            repair_metrics["production_replay_exact_directed"].sum()
        ),
        "production_replay_total_false_predicted": int(
            repair_metrics["production_replay_n_false_predicted"].sum()
        ),
        "production_replay_total_strict_reverse": int(
            repair_metrics["production_replay_n_strict_reverse"].sum()
        ),
        "diagnostic_status": "posthoc_formal_train_not_certification",
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        args.out_dir / "phase_b_manifest.json",
        {
            "phase_a_manifest_sha256": _sha256(phase_a_manifest_path),
            "labeled_output_sha256": {
                path.name: _sha256(path)
                for path in [*labeled_paths, summary_path]
            },
            "truth_opened_only_after_phase_a_manifest": True,
            "source_sha256": _sha256(Path(__file__)),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
