#!/usr/bin/env python3
"""Freeze truth-blind sparse-reporter completion evidence for CA-BOSS v4.49.

This stage targets the small set of assays omitted by the v4.33 reporter gate.
For every assay it identifies the three focal features outside the observed
two-parent/one-child collider and records their full/fold component strength,
stability, detection, and agreement with either direct parent.  For an assay
whose v4.33 hypothesis is unresolved, it also evaluates every external
candidate against each already-supported reporter in an exact all-assay
permutation synchronization.

All assays' background-derived components participate in synchronization, but
an unresolved assay is never assigned a reporter merely because one component
aligns.  Reporter selection is deferred to a separate cross-fit calibration.
The stage opens no scorer or truth resource, creates no simulation data, and
makes no edge, pair, or direction claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import score_unpaired_mdcr_ks_baseline_v4 as mdcr_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.sparse_reporter_completion_evidence.v1"
ALGORITHM = "CA-BOSS v4.49 truth-blind sparse reporter completion evidence"
FEATURE_COLUMNS = (
    "detected_fraction",
    "full_strength",
    "full_margin",
    "fold_min_strength",
    "fold_min_margin",
    "component_stable",
    "parent_component_match_count",
)
CANDIDATE_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "modality",
    "feature",
    "assay_upstream_selected",
    "feature_is_upstream_selected_reporter",
    "direct_parent_members",
    "direct_collider_child",
    *FEATURE_COLUMNS,
    "fold_mean_strength",
    "fold_mean_margin",
    "component_mode_fraction",
    "parent_component_match_full",
    "parent_component_match_folds",
    "max_parent_loading_cosine_full",
    "min_max_parent_loading_cosine_fold",
    "mean_max_parent_loading_cosine_fold",
    "full_component",
    "fold_components_json",
    "hypothesis_selected",
    "hypothesis_type",
]
PAIR_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "unresolved_modality",
    "candidate_reporter_feature",
    "partner_modality",
    "partner_reporter_feature",
    "candidate_assay_pair_json",
    "candidate_cross_assay_feature_pair_json",
    "global_coassignment_context_pattern",
    "global_coassignment_context_count",
    "heldout_global_coassignment_count",
    "pair_exclusion_relative_margin_json",
    "minimum_pair_exclusion_relative_margin",
    "mean_pair_exclusion_relative_margin",
    "minimum_heldout_pair_exclusion_relative_margin",
    "mean_heldout_pair_exclusion_relative_margin",
    "reporter_pair_scaled_ks_ratio_json",
    "minimum_reporter_pair_scaled_ks_ratio",
    "mean_reporter_pair_scaled_ks_ratio",
    "maximum_reporter_pair_scaled_ks_ratio",
    "all_contexts_globally_coassigned",
    "all_context_exclusion_margins_positive",
    "all_context_reporter_pair_ks_within_bound",
    "hypothesis_selected",
    "hypothesis_type",
]
CONTEXT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "context_index",
    "context_label",
    "modalities",
    "n_modalities",
    "rank",
    "assignments_evaluated",
    "best_total_scaled_ks_ratio",
    "second_best_total_scaled_ks_ratio",
    "global_assignment_relative_margin",
    "best_global_to_local_components_json",
]


def _as_bool(value: Any) -> bool:
    return synchronization._as_bool(value)


def _verified_json(path: Path, hash_field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = str(value.get(hash_field, ""))
    unsigned = dict(value)
    unsigned.pop(hash_field, None)
    if observed != synchronization._canonical_sha256(unsigned):
        raise ValueError(f"Payload hash mismatch: {path}")
    return value


def _verify_phase_a(
    phase_a_dir: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], pd.DataFrame]:
    directory = synchronization._forbid_truth_path(phase_a_dir)
    manifest_path = directory / "phase_a_manifest.json"
    selection_path = directory / "phase_a_selection.json"
    manifest = _verified_json(manifest_path, "manifest_payload_sha256")
    selection = _verified_json(selection_path, "selection_payload_sha256")
    if manifest.get("phase") != "a":
        raise ValueError("Expected a Phase-A input")
    if manifest.get("scorer_or_truth_opened") is not False:
        raise ValueError("Phase A did not certify truth separation")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Phase A created a new world or dataset")
    if manifest.get("selection_sha256") != synchronization._sha256(selection_path):
        raise ValueError("Phase-A selection hash mismatch")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"Phase-A output hash mismatch: {name}")
    direct_path = directory / "phase_a_direct_candidates.tsv"
    direct = pd.read_csv(direct_path, sep="\t", keep_default_na=False)
    return directory, manifest, selection, direct


def _world_key(block: str, seed: int) -> str:
    return f"{block}__world_seed{int(seed)}"


def _selected_world_keys(selection: Mapping[str, Any]) -> set[str]:
    return {
        _world_key(str(row["block"]), int(row["seed"]))
        for row in selection.get("selected_worlds", [])
    }


def _component_vector(row: pd.Series) -> np.ndarray:
    columns = sorted(
        [column for column in row.index if column.startswith("component_") and column.endswith("_partial_correlation")],
        key=lambda value: int(value.split("_")[1]),
    )
    if not columns:
        raise ValueError("Focal association lacks component correlations")
    return pd.to_numeric(row[columns], errors="raise").to_numpy(dtype=float)


def _absolute_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return abs(float(left @ right / denominator)) if denominator > 1e-12 else 0.0


def _motif_by_assay(
    direct: pd.DataFrame, *, block: str, seed: int
) -> dict[str, tuple[tuple[str, str], str]]:
    required = {
        "block",
        "seed",
        "source_assay",
        "node1",
        "endpoint1",
        "node2",
        "endpoint2",
    }
    if missing := sorted(required.difference(direct.columns)):
        raise ValueError(f"Direct table lacks columns: {missing}")
    world = direct.loc[
        direct["block"].astype(str).eq(str(block))
        & pd.to_numeric(direct["seed"], errors="raise").eq(int(seed))
    ]
    motifs: dict[str, tuple[tuple[str, str], str]] = {}
    for assay, arms in world.groupby("source_assay", sort=True):
        if len(arms) != 2:
            raise ValueError(f"{assay} does not have two direct arms")
        if not arms["endpoint1"].astype(str).eq("TAIL").all():
            raise ValueError("Direct parent endpoint is not TAIL")
        if not arms["endpoint2"].astype(str).eq("ARROW").all():
            raise ValueError("Direct child endpoint is not ARROW")
        parents = tuple(sorted(arms["node1"].astype(str).unique()))
        children = tuple(sorted(arms["node2"].astype(str).unique()))
        if len(parents) != 2 or len(children) != 1:
            raise ValueError(f"Malformed direct collider for {assay}")
        motifs[str(assay)] = ((parents[0], parents[1]), children[0])
    return motifs


def _feature_candidates(
    *,
    world_key: str,
    block: str,
    seed: int,
    associations: pd.DataFrame,
    fold_associations: pd.DataFrame,
    upstream: pd.DataFrame,
    motifs: Mapping[str, tuple[tuple[str, str], str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    upstream_by_assay = upstream.set_index("modality", verify_integrity=True)
    for modality, assay in associations.groupby("modality", sort=True):
        modality = str(modality)
        if modality not in motifs:
            raise ValueError(f"Direct motif missing for {modality}")
        parents, child = motifs[modality]
        motif = {*parents, child}
        external = sorted(set(assay["feature"].astype(str)).difference(motif))
        if len(external) != 3:
            raise ValueError(
                f"Expected three external focal features for {modality}"
            )
        upstream_row = upstream_by_assay.loc[modality]
        upstream_selected = _as_bool(upstream_row["hypothesis_selected"])
        selected_reporters = {
            value
            for value in str(
                upstream_row["external_state_reporter_members"]
            ).split(";")
            if value
        }
        parent_full = assay.loc[assay["feature"].astype(str).isin(parents)]
        if len(parent_full) != 2:
            raise ValueError("Direct parents are absent from focal associations")
        assay_folds = fold_associations.loc[
            fold_associations["modality"].astype(str).eq(modality)
        ]
        fold_ids = sorted(
            pd.to_numeric(
                assay_folds["feature_fold"], errors="raise"
            ).astype(int).unique()
        )
        if len(fold_ids) != 2:
            raise ValueError("Expected two held-out feature folds")
        for feature in external:
            full_row = assay.loc[assay["feature"].astype(str).eq(feature)]
            if len(full_row) != 1:
                raise ValueError("Focal feature is not unique")
            full_row = full_row.iloc[0]
            fold_rows = assay_folds.loc[
                assay_folds["feature"].astype(str).eq(feature)
            ].sort_values("feature_fold")
            if len(fold_rows) != len(fold_ids):
                raise ValueError("Focal feature lacks a held-out fold")
            full_component = int(full_row["dominant_component"])
            fold_components = [
                int(value) for value in fold_rows["dominant_component"]
            ]
            components = [full_component, *fold_components]
            full_parent_components = set(
                pd.to_numeric(
                    parent_full["dominant_component"], errors="raise"
                ).astype(int)
            )
            match_full = int(full_component in full_parent_components)
            match_folds: list[int] = []
            fold_cosines: list[float] = []
            for fold_id, (_, candidate_row) in zip(
                fold_ids, fold_rows.iterrows(), strict=True
            ):
                fold_parents = assay_folds.loc[
                    pd.to_numeric(
                        assay_folds["feature_fold"], errors="raise"
                    ).eq(fold_id)
                    & assay_folds["feature"].astype(str).isin(parents)
                ]
                match_folds.append(
                    int(
                        int(candidate_row["dominant_component"])
                        in set(
                            pd.to_numeric(
                                fold_parents["dominant_component"],
                                errors="raise",
                            ).astype(int)
                        )
                    )
                )
                candidate_vector = _component_vector(candidate_row)
                fold_cosines.append(
                    max(
                        _absolute_cosine(
                            candidate_vector, _component_vector(parent_row)
                        )
                        for _, parent_row in fold_parents.iterrows()
                    )
                )
            full_vector = _component_vector(full_row)
            full_cosine = max(
                _absolute_cosine(full_vector, _component_vector(parent_row))
                for _, parent_row in parent_full.iterrows()
            )
            counts = {component: components.count(component) for component in set(components)}
            rows.append(
                {
                    "world_key": world_key,
                    "block": block,
                    "seed": seed,
                    "modality": modality,
                    "feature": feature,
                    "assay_upstream_selected": upstream_selected,
                    "feature_is_upstream_selected_reporter": (
                        feature in selected_reporters
                    ),
                    "direct_parent_members": ";".join(parents),
                    "direct_collider_child": child,
                    "detected_fraction": float(full_row["detected_cells"])
                    / float(full_row["cells"]),
                    "full_strength": float(
                        full_row["dominant_absolute_partial_correlation"]
                    ),
                    "full_margin": float(
                        full_row["component_relative_margin"]
                    ),
                    "fold_min_strength": float(
                        fold_rows[
                            "dominant_absolute_partial_correlation"
                        ].min()
                    ),
                    "fold_min_margin": float(
                        fold_rows["component_relative_margin"].min()
                    ),
                    "component_stable": len(set(components)) == 1,
                    "parent_component_match_count": match_full
                    + sum(match_folds),
                    "fold_mean_strength": float(
                        fold_rows[
                            "dominant_absolute_partial_correlation"
                        ].mean()
                    ),
                    "fold_mean_margin": float(
                        fold_rows["component_relative_margin"].mean()
                    ),
                    "component_mode_fraction": max(counts.values())
                    / len(components),
                    "parent_component_match_full": match_full,
                    "parent_component_match_folds": sum(match_folds),
                    "max_parent_loading_cosine_full": full_cosine,
                    "min_max_parent_loading_cosine_fold": min(fold_cosines),
                    "mean_max_parent_loading_cosine_fold": float(
                        np.mean(fold_cosines)
                    ),
                    "full_component": full_component,
                    "fold_components_json": json.dumps(
                        dict(zip(map(str, fold_ids), fold_components)),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "hypothesis_selected": False,
                    "hypothesis_type": "unresolved_reporter_candidate",
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _pair_evidence(
    *,
    world_key: str,
    block: str,
    seed: int,
    candidates: pd.DataFrame,
    upstream: pd.DataFrame,
    context_tables: Sequence[pd.DataFrame],
    context_associations: Sequence[pd.DataFrame],
    solutions: Sequence[synchronization.MultiwayPermutationSolution],
) -> pd.DataFrame:
    unresolved_assays = sorted(
        candidates.loc[
            ~candidates["assay_upstream_selected"].astype(bool), "modality"
        ].astype(str).unique()
    )
    if len(unresolved_assays) > 1:
        raise ValueError(
            "Current sparse completion contract permits at most one unresolved assay per world"
        )
    if not unresolved_assays:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    unresolved = unresolved_assays[0]
    selected_upstream = upstream.loc[
        upstream["hypothesis_selected"].map(_as_bool)
    ].copy()
    partner_reporters: dict[str, str] = {}
    for row in selected_upstream.itertuples(index=False):
        reporters = [
            value
            for value in str(row.external_state_reporter_members).split(";")
            if value
        ]
        if len(reporters) != 1:
            raise ValueError("Sparse completion requires one supported reporter")
        partner_reporters[str(row.modality)] = reporters[0]
    rows: list[dict[str, Any]] = []
    unresolved_candidates = candidates.loc[
        candidates["modality"].astype(str).eq(unresolved)
    ]
    for candidate in unresolved_candidates.itertuples(index=False):
        for partner, partner_feature in sorted(partner_reporters.items()):
            coassigned: list[bool] = []
            margins: list[float] = []
            ratios: list[float] = []
            for context_index, (table, associations, solution) in enumerate(
                zip(
                    context_tables,
                    context_associations,
                    solutions,
                    strict=True,
                )
            ):
                left, right = sorted((unresolved, partner))
                left_feature = (
                    str(candidate.feature)
                    if left == unresolved
                    else partner_feature
                )
                right_feature = (
                    str(candidate.feature)
                    if right == unresolved
                    else partner_feature
                )
                left_rows = associations.loc[
                    associations["modality"].astype(str).eq(left)
                    & associations["feature"].astype(str).eq(left_feature)
                ]
                right_rows = associations.loc[
                    associations["modality"].astype(str).eq(right)
                    & associations["feature"].astype(str).eq(right_feature)
                ]
                if len(left_rows) != 1 or len(right_rows) != 1:
                    raise ValueError("Reporter association is not unique")
                diagnostic = synchronization.reporter_pair_diagnostics(
                    solution,
                    table,
                    left,
                    right,
                    int(left_rows.iloc[0]["dominant_component"]),
                    int(right_rows.iloc[0]["dominant_component"]),
                )
                coassigned.append(
                    bool(diagnostic["coassigned_in_best_solution"])
                )
                margins.append(
                    float(diagnostic["pair_exclusion_relative_margin"])
                )
                ratios.append(
                    float(diagnostic["reporter_pair_scaled_ks_ratio"])
                )
            assay_pair = sorted((unresolved, partner))
            feature_pair = sorted((str(candidate.feature), partner_feature))
            heldout_margins = margins[1:]
            rows.append(
                {
                    "world_key": world_key,
                    "block": block,
                    "seed": seed,
                    "unresolved_modality": unresolved,
                    "candidate_reporter_feature": str(candidate.feature),
                    "partner_modality": partner,
                    "partner_reporter_feature": partner_feature,
                    "candidate_assay_pair_json": json.dumps(
                        assay_pair, separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_json": json.dumps(
                        feature_pair, separators=(",", ":")
                    ),
                    "global_coassignment_context_pattern": "".join(
                        "1" if value else "0" for value in coassigned
                    ),
                    "global_coassignment_context_count": sum(coassigned),
                    "heldout_global_coassignment_count": sum(coassigned[1:]),
                    "pair_exclusion_relative_margin_json": json.dumps(
                        margins, separators=(",", ":")
                    ),
                    "minimum_pair_exclusion_relative_margin": min(margins),
                    "mean_pair_exclusion_relative_margin": float(
                        np.mean(margins)
                    ),
                    "minimum_heldout_pair_exclusion_relative_margin": min(
                        heldout_margins
                    ),
                    "mean_heldout_pair_exclusion_relative_margin": float(
                        np.mean(heldout_margins)
                    ),
                    "reporter_pair_scaled_ks_ratio_json": json.dumps(
                        ratios, separators=(",", ":")
                    ),
                    "minimum_reporter_pair_scaled_ks_ratio": min(ratios),
                    "mean_reporter_pair_scaled_ks_ratio": float(
                        np.mean(ratios)
                    ),
                    "maximum_reporter_pair_scaled_ks_ratio": max(ratios),
                    "all_contexts_globally_coassigned": all(coassigned),
                    "all_context_exclusion_margins_positive": all(
                        margin > 0.0 for margin in margins
                    ),
                    "all_context_reporter_pair_ks_within_bound": all(
                        ratio <= 1.0 for ratio in ratios
                    ),
                    "hypothesis_selected": False,
                    "hypothesis_type": "unresolved_cross_assay_reporter_pair",
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def freeze_panel(
    *,
    bridge_root: Path,
    evidence_root: Path,
    phase_a_dir: Path,
    out_root: Path,
) -> dict[str, Any]:
    """Freeze one previously selected cached panel without opening truth."""

    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_root}")
    bridge_root = synchronization._forbid_truth_path(bridge_root)
    evidence_root = synchronization._forbid_truth_path(evidence_root)
    phase_a, phase_manifest, selection, direct = _verify_phase_a(phase_a_dir)
    expected_keys = _selected_world_keys(selection)
    bridge_dirs = {
        path.name: path
        for path in bridge_root.iterdir()
        if path.is_dir()
    }
    evidence_dirs = {
        path.name: path
        for path in evidence_root.iterdir()
        if path.is_dir()
    }
    if set(bridge_dirs) != expected_keys or set(evidence_dirs) != expected_keys:
        raise ValueError("Bridge/evidence worlds differ from Phase-A selection")

    out_root.mkdir(parents=True)
    world_manifest_hashes: dict[str, str] = {}
    for key in sorted(expected_keys):
        bridge, bridge_manifest = synchronization.verify_bridge(
            bridge_dirs[key]
        )
        evidence = synchronization._forbid_truth_path(evidence_dirs[key])
        evidence_manifest = mdcr_scorer.verify_frozen_evidence(evidence)
        block = str(bridge_manifest["block"])
        seed = int(bridge_manifest["seed"])
        if key != _world_key(block, seed):
            raise ValueError("Bridge world identity mismatch")
        if str(bridge_manifest.get("mdcr_manifest_payload_sha256")) != str(
            evidence_manifest.get("manifest_payload_sha256")
        ):
            raise ValueError("Bridge and MDCR evidence manifests disagree")
        if str(
            bridge_manifest.get(
                "clean_upstream_phase_a_manifest_payload_sha256"
            )
        ) != str(phase_manifest["manifest_payload_sha256"]):
            raise ValueError("Bridge and Phase-A manifest payloads disagree")
        if str(
            bridge_manifest.get("input_sha256", {}).get(
                "clean_upstream_phase_a_manifest"
            )
        ) != synchronization._sha256(phase_a / "phase_a_manifest.json"):
            raise ValueError("Bridge and Phase-A manifest files disagree")

        associations = pd.read_csv(
            bridge / "focal_component_associations.tsv",
            sep="\t",
            keep_default_na=False,
        )
        fold_associations = pd.read_csv(
            bridge / "fold_focal_component_associations.tsv",
            sep="\t",
            keep_default_na=False,
        )
        upstream = pd.read_csv(
            bridge / "upstream_assay_local_hypotheses.tsv",
            sep="\t",
            keep_default_na=False,
        )
        motifs = _motif_by_assay(direct, block=block, seed=seed)
        candidates = _feature_candidates(
            world_key=key,
            block=block,
            seed=seed,
            associations=associations,
            fold_associations=fold_associations,
            upstream=upstream,
            motifs=motifs,
        )

        full_pairwise = pd.read_csv(
            evidence / "pairwise_component_matches.tsv",
            sep="\t",
            keep_default_na=False,
        )
        fold_pairwise = pd.read_csv(
            evidence / "fold_pairwise_component_matches.tsv",
            sep="\t",
            keep_default_na=False,
        )
        fold_pairwise["feature_fold"] = pd.to_numeric(
            fold_pairwise["feature_fold"], errors="raise"
        ).astype(int)
        fold_ids = sorted(fold_pairwise["feature_fold"].unique())
        if len(fold_ids) != 2:
            raise ValueError("Expected exactly two held-out feature folds")
        context_tables = [full_pairwise] + [
            frame.reset_index(drop=True)
            for _, frame in fold_pairwise.groupby("feature_fold", sort=True)
        ]
        context_associations = [associations] + [
            fold_associations.loc[
                pd.to_numeric(
                    fold_associations["feature_fold"], errors="raise"
                ).eq(fold)
            ].reset_index(drop=True)
            for fold in fold_ids
        ]
        modalities = sorted(associations["modality"].astype(str).unique())
        solutions = [
            synchronization.solve_multiway_permutation_synchronization(
                table, modalities
            )
            for table in context_tables
        ]
        pair_evidence = _pair_evidence(
            world_key=key,
            block=block,
            seed=seed,
            candidates=candidates,
            upstream=upstream,
            context_tables=context_tables,
            context_associations=context_associations,
            solutions=solutions,
        )
        context_rows: list[dict[str, Any]] = []
        labels = ["full_fit", *[f"feature_fold_{fold}" for fold in fold_ids]]
        for index, (label, solution) in enumerate(
            zip(labels, solutions, strict=True)
        ):
            assignment = {
                modality: solution.best_orders[modality_index].tolist()
                for modality_index, modality in enumerate(solution.modalities)
            }
            context_rows.append(
                {
                    "world_key": key,
                    "block": block,
                    "seed": seed,
                    "context_index": index,
                    "context_label": label,
                    "modalities": ";".join(solution.modalities),
                    "n_modalities": len(solution.modalities),
                    "rank": int(solution.orders.shape[2]),
                    "assignments_evaluated": len(solution.objectives),
                    "best_total_scaled_ks_ratio": solution.best_objective,
                    "second_best_total_scaled_ks_ratio": (
                        solution.second_best_objective
                    ),
                    "global_assignment_relative_margin": (
                        solution.global_relative_margin
                    ),
                    "best_global_to_local_components_json": json.dumps(
                        assignment,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        contexts = pd.DataFrame(context_rows, columns=CONTEXT_COLUMNS)

        world_out = out_root / key
        world_out.mkdir()
        outputs = {
            "external_reporter_candidates.tsv": candidates,
            "unresolved_reporter_pair_evidence.tsv": pair_evidence,
            "all_assay_synchronization_contexts.tsv": contexts,
        }
        for name, frame in outputs.items():
            frame.to_csv(world_out / name, sep="\t", index=False)
        manifest = {
            "schema_version": SCHEMA,
            "algorithm": ALGORITHM,
            "world_key": key,
            "block": block,
            "seed": seed,
            "model_feature_columns": list(FEATURE_COLUMNS),
            "external_candidate_count": len(candidates),
            "upstream_unresolved_assay_count": int(
                (~candidates["assay_upstream_selected"].astype(bool))
                .groupby(candidates["modality"])
                .any()
                .sum()
            ),
            "unresolved_reporter_pair_evidence_count": len(pair_evidence),
            "all_assay_synchronization_context_count": len(contexts),
            "information_boundary": {
                "scorer_truth_used": False,
                "simulator_latent_state_used": False,
                "paired_cells_used": False,
                "cross_assay_cell_correspondence_used": False,
                "external_biological_features_or_priors_used": False,
                "heldout_background_components_used": True,
                "direct_structure_used_only_to_define_external_candidates": True,
                "all_assay_components_used_as_alignment_anchors_only": True,
                "unresolved_assay_reporter_selected": False,
            },
            "formal_pretruth_barrier_satisfied": True,
            "new_worlds_or_datasets_created": False,
            "hypothesis_selected": False,
            "within_assay_direct_structure_claimed": False,
            "exact_cross_assay_pair_claimed": False,
            "cross_assay_direction_claimed": False,
            "candidate_pairs_may_be_expanded_as_edges": False,
            "bridge_provenance_payload_sha256": bridge_manifest[
                "provenance_payload_sha256"
            ],
            "mdcr_manifest_payload_sha256": evidence_manifest[
                "manifest_payload_sha256"
            ],
            "phase_a_manifest_payload_sha256": phase_manifest[
                "manifest_payload_sha256"
            ],
            "source_sha256": {
                Path(__file__).name: synchronization._sha256(
                    Path(__file__).resolve()
                )
            },
            "input_sha256": {
                "bridge_provenance.json": synchronization._sha256(
                    bridge / "provenance.json"
                ),
                "focal_component_associations.tsv": synchronization._sha256(
                    bridge / "focal_component_associations.tsv"
                ),
                "fold_focal_component_associations.tsv": synchronization._sha256(
                    bridge / "fold_focal_component_associations.tsv"
                ),
                "upstream_assay_local_hypotheses.tsv": synchronization._sha256(
                    bridge / "upstream_assay_local_hypotheses.tsv"
                ),
                "frozen_evidence_manifest.json": synchronization._sha256(
                    evidence / "frozen_evidence_manifest.json"
                ),
                "pairwise_component_matches.tsv": synchronization._sha256(
                    evidence / "pairwise_component_matches.tsv"
                ),
                "fold_pairwise_component_matches.tsv": synchronization._sha256(
                    evidence / "fold_pairwise_component_matches.tsv"
                ),
                "phase_a_manifest.json": synchronization._sha256(
                    phase_a / "phase_a_manifest.json"
                ),
                "phase_a_direct_candidates.tsv": synchronization._sha256(
                    phase_a / "phase_a_direct_candidates.tsv"
                ),
            },
            "output_sha256": {
                name: synchronization._sha256(world_out / name)
                for name in outputs
            },
        }
        manifest["manifest_payload_sha256"] = synchronization._canonical_sha256(
            manifest
        )
        manifest_path = world_out / "reporter_completion_evidence_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        world_manifest_hashes[key] = synchronization._sha256(manifest_path)

    panel_manifest = {
        "schema_version": "ca_boss_v4.sparse_reporter_completion_evidence_panel.v1",
        "algorithm": ALGORITHM,
        "world_count": len(expected_keys),
        "world_keys": sorted(expected_keys),
        "new_worlds_or_datasets_created": False,
        "scorer_truth_used": False,
        "phase_a_manifest_payload_sha256": phase_manifest[
            "manifest_payload_sha256"
        ],
        "world_manifest_file_sha256": world_manifest_hashes,
        "source_sha256": {
            Path(__file__).name: synchronization._sha256(Path(__file__).resolve())
        },
    }
    panel_manifest["manifest_payload_sha256"] = synchronization._canonical_sha256(
        panel_manifest
    )
    (out_root / "panel_evidence_manifest.json").write_text(
        json.dumps(panel_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return panel_manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = freeze_panel(
        bridge_root=args.bridge_root,
        evidence_root=args.evidence_root,
        phase_a_dir=args.phase_a_dir,
        out_root=args.out_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
