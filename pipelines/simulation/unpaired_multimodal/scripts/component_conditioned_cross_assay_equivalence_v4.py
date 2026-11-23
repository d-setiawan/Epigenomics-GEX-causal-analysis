#!/usr/bin/env python3
"""Component-conditioned bridge from assay-local to cross-assay structure.

This exploratory CA-BOSS v4 bridge uses only within-assay cell rows.  Focal
features are associated with background-only ICA components after linear
residualization of log background depth.  Assay-local v4.33 reporter sets are
then mapped through full and disjoint-background-feature-fold component
alignments.  Exact two-assay participation requires either direct full-context
agreement or a unique all-context global component bijection after the bounded
cross-context alternative-set route has abstained.

The bridge returns shared-state equivalence hyperedges.  It does not infer a
cross-assay causal direction and never expands assay-local ambiguity sets into
ordinary causal edges.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.stats import binomtest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import direct_parent_state_reporter_equivalence_v4 as equivalence  # noqa: E402
import score_unpaired_mdcr_ks_baseline_v4 as mdcr_scorer  # noqa: E402


FloatArray = NDArray[np.float64]
SCHEMA = "ca_boss_v4.component_conditioned_cross_assay_equivalence.v15"
ALGORITHM = (
    "CA-BOSS v4.44 global-bijection-completed component ambiguity graph "
    "equivalence"
)
REPORTER_BOOTSTRAP_REPLICATES = 128
REPORTER_BOOTSTRAP_ALPHA = 0.05
MINIMUM_CONTEXT_CLIQUE_FRACTION = 0.50
FIXED_SELECTION_RULE = (
    "use_each_reporters_context_specific_component_in_the_full_fit_and_two_"
    "disjoint_background_feature_folds_connect_two_component_resolved_assays_"
    "when_their_reporters_share_a_consensus_axis_and_are_mutual_minimum_"
    "bonferroni_ks_matches_with_positive_runner_up_margin_in_a_strict_"
    "majority_of_contexts_then_return_multi_assay_connected_components_as_"
    "unexpanded_participation_ambiguity_hyperedges_but_require_a_two_assay_"
    "component_to_match_in_every_context_and_both_reporters_to_pass_a_128_"
    "resample_bonferroni_binomial_assignment_test_in_the_full_fit_and_each_"
    "disjoint_background_feature_fold_then_when_an_exact_two_assay_component_"
    "has_exactly_one_statistically_unsupported_context_despite_all_context_"
    "reporter_bootstrap_support_return_the_focal_pair_together_with_directly_"
    "matched_partners_from_the_disagreeing_context_as_one_unexpanded_"
    "assay_pair_ambiguity_hyperedge_only_when_at_least_two_noncore_candidate_"
    "pairs_exist_and_allow_a_component_identified_all_context_bootstrap_"
    "supported_assay_with_unresolved_within_assay_parent_pairing_only_as_a_"
    "nonpoint_ambiguity_alternative_never_as_a_primary_stable_graph_node_"
    "then_for_a_still_unresolved_primary_two_assay_majority_pair_with_no_"
    "ambiguity_rescue_require_the_minimum_total_ks_cost_component_bijection_"
    "to_map_the_reporters_with_positive_global_runner_up_margin_and_"
    "bonferroni_ks_fit_in_every_context_before_exact_participation_completion"
)
HYPOTHESIS_COLUMNS = [
    "block",
    "seed",
    "hypothesis_id",
    "modality",
    "direct_parent_members",
    "external_state_reporter_members",
    "candidate_pairs_json",
    "candidate_pair_count",
    "hypothesis_selected",
]
ASSAY_ASSIGNMENT_COLUMNS = [
    "hypothesis_id",
    "modality",
    "external_state_reporter_members",
    "candidate_pairs_json",
    "candidate_pair_count",
    "component_compatible_candidate_pairs_json",
    "component_compatible_candidate_pair_count",
    "within_assay_pair_resolved_by_component",
    "reporter_component_identified",
    "reporter_local_component",
    "reporter_minimum_absolute_partial_correlation",
    "reporter_minimum_component_relative_margin",
    "reporter_feature_fold_components_json",
    "reporter_context_components_identified",
    "reporter_components_stable_across_full_and_feature_folds",
    "reporter_minimum_feature_fold_absolute_partial_correlation",
    "reporter_minimum_feature_fold_component_relative_margin",
    "reporter_bootstrap_component_assignment_supported",
    "reporter_maximum_bootstrap_bonferroni_p",
    "reporter_minimum_bootstrap_selected_fraction",
    "reporter_feature_fold_bootstrap_supported_json",
    "reporter_all_feature_fold_bootstrap_component_assignments_supported",
    "reporter_all_context_bootstrap_component_assignments_supported",
    "reporter_maximum_feature_fold_bootstrap_bonferroni_p",
    "reporter_minimum_feature_fold_bootstrap_selected_fraction",
    "full_consensus_axis",
    "feature_fold_axis_json",
    "axis_signature",
]
HYPEREDGE_COLUMNS = [
    "hyperedge_id",
    "axis_signature",
    "modalities",
    "n_modalities",
    "candidate_assay_pairs_json",
    "candidate_assay_pair_count",
    "ambiguity_source_assay_pair_json",
    "ambiguity_context_indices_json",
    "ambiguity_candidate_generation_rule",
    "ambiguity_source_pair_all_context_reporter_bootstrap_supported",
    "ambiguity_alternatives_supported_in_disagreeing_context",
    "ambiguity_candidate_pairs_exclude_retained_core_pairs",
    "ambiguity_alternative_pool_rule",
    "ambiguity_unresolved_local_alternative_modalities_json",
    "ambiguity_all_focal_assays_within_assay_pair_resolved",
    "ambiguity_all_candidate_modalities_within_assay_pair_resolved",
    "selection_route",
    "global_bijection_context_support_pattern",
    "all_context_global_bijection_supported",
    "minimum_context_global_bijection_relative_margin",
    "maximum_context_global_bijection_assigned_scaled_ks",
    "assay_participation_is_set_valued",
    "exact_assay_participation_claimed",
    "assay_local_hypothesis_ids",
    "external_state_reporter_members",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "minimum_reporter_absolute_partial_correlation",
    "minimum_reporter_component_relative_margin",
    "minimum_feature_fold_reporter_absolute_partial_correlation",
    "minimum_feature_fold_reporter_component_relative_margin",
    "all_reporter_components_stable_across_full_and_feature_folds",
    "all_reporter_bootstrap_component_assignments_supported",
    "maximum_reporter_bootstrap_bonferroni_p",
    "minimum_reporter_bootstrap_selected_fraction",
    "all_reporter_feature_fold_bootstrap_component_assignments_supported",
    "all_reporter_context_bootstrap_component_assignments_supported",
    "maximum_reporter_feature_fold_bootstrap_bonferroni_p",
    "minimum_reporter_feature_fold_bootstrap_selected_fraction",
    "context_count",
    "minimum_stable_pair_graph_context_support_count",
    "minimum_stable_pair_graph_context_support_fraction",
    "all_stable_pair_graph_edges_have_strict_context_majority",
    "minimum_context_pairwise_match_support",
    "minimum_context_maximum_clique_fraction",
    "minimum_context_direct_component_assignment_margin",
    "all_context_direct_component_assignments_unique",
    "full_fit_direct_component_pair_matched",
    "all_feature_fold_direct_component_pairs_matched",
    "full_and_all_fold_direct_component_pairs_matched",
    "all_modalities_in_full_and_fold_maximum_cliques",
    "maximum_clique_stability_guard_passed",
    "half_assay_clique_support_guard_passed",
    "all_within_assay_pairs_component_resolved",
    "local_structure_or_all_context_clique_guard_passed",
    "stable_pair_graph_edge_count",
    "stable_pair_graph_possible_edge_count",
    "stable_pair_graph_is_complete",
    "stable_pair_graph_connected",
    "stable_pair_graph_edges_pass_local_structure_or_pair_clique_guard",
    "hypothesis_selected",
    "hypothesis_status",
    "fixed_selection_rule",
    "full_and_all_feature_folds_co_cluster",
    "hypothesis_type",
    "exact_cross_assay_pair_claimed",
    "cross_assay_direction_claimed",
    "candidate_pairs_may_be_expanded_as_edges",
]
PAIR_DECISION_COLUMNS = [
    "left_modality",
    "right_modality",
    "statistical_context_support_pattern",
    "reporter_bootstrap_context_support_pattern",
    "context_support_pattern",
    "context_count",
    "context_support_count",
    "context_support_fraction",
    "strict_context_majority_passed",
    "all_contexts_supported",
    "left_reporter_bootstrap_supported",
    "right_reporter_bootstrap_supported",
    "both_reporters_bootstrap_supported",
    "left_reporter_all_feature_fold_bootstrap_supported",
    "right_reporter_all_feature_fold_bootstrap_supported",
    "both_reporters_all_context_bootstrap_supported",
    "left_within_assay_pair_resolved_by_component",
    "right_within_assay_pair_resolved_by_component",
    "eligible_for_primary_stable_pair_graph",
    "eligible_as_cross_context_ambiguity_alternative",
    "global_bijection_context_support_pattern",
    "all_context_global_bijection_supported",
    "minimum_context_global_bijection_relative_margin",
    "maximum_context_global_bijection_assigned_scaled_ks",
    "retained_by_global_bijection_completion",
    "included_in_output_ambiguity_set",
    "ambiguity_hyperedge_ids",
    "ambiguity_candidate_assay_pair_count",
    "ambiguity_candidate_assay_pairs_json",
    "majority_graph_edge_selected",
    "connected_component_id",
    "connected_component_size",
    "retained_in_output_hyperedge",
    "selection_status",
    "edge_type",
    "candidate_pair_may_be_expanded_as_edge",
]
BOOTSTRAP_STABILITY_COLUMNS = [
    "modality",
    "feature",
    "selected_component",
    "bootstrap_replicates",
    "component_0_assignment_count",
    "component_1_assignment_count",
    "component_2_assignment_count",
    "selected_component_assignment_count",
    "selected_component_assignment_fraction",
    "maximum_competitor_assignment_count",
    "maximum_one_sided_binomial_p",
    "bonferroni_component_assignment_p",
    "component_assignment_bootstrap_supported",
]
FOLD_BOOTSTRAP_STABILITY_COLUMNS = [
    "feature_fold",
    *BOOTSTRAP_STABILITY_COLUMNS,
]
FOCAL_METADATA = {
    "cell",
    "modality",
    "observed_background_library_size",
    "observed_library_size",
}


def local_structure_or_all_context_clique_guard(
    *,
    half_assay_clique_support: bool,
    all_within_assay_pairs_component_resolved: bool,
    all_modalities_in_every_context_maximum_clique: bool,
) -> bool:
    """Require local structural resolution or paper-like clique consistency."""

    return bool(
        half_assay_clique_support
        and (
            all_within_assay_pairs_component_resolved
            or all_modalities_in_every_context_maximum_clique
        )
    )


def stable_pair_graph_components(
    modalities: Sequence[str], stable_edges: set[frozenset[str]]
) -> list[list[str]]:
    """Return deterministic non-singleton connected components."""

    nodes = sorted({str(value) for value in modalities})
    adjacency = {node: set() for node in nodes}
    for edge in stable_edges:
        if len(edge) != 2:
            raise ValueError("Stable pair-graph edges must contain two assays")
        left, right = sorted(edge)
        if left not in adjacency or right not in adjacency:
            raise ValueError("Stable pair-graph edge references an unknown assay")
        adjacency[left].add(right)
        adjacency[right].add(left)

    components: list[list[str]] = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node].difference(component)))
        unseen.difference_update(component)
        if len(component) >= 2:
            components.append(sorted(component))
    return components


def direct_component_assignment_margin(
    table: pd.DataFrame, evidence: Mapping[str, Any]
) -> float:
    """Return the selected KS match's nearest row/column runner-up gap."""

    left_modality = str(evidence["left_modality"])
    right_modality = str(evidence["right_modality"])
    left_component = int(evidence["left_component"])
    right_component = int(evidence["right_component"])
    selected_statistic = float(evidence["scaled_ks_statistic"])
    pair_table = table.loc[
        table["left_modality"].astype(str).eq(left_modality)
        & table["right_modality"].astype(str).eq(right_modality)
    ]
    row_statistics = sorted(
        pd.to_numeric(
            pair_table.loc[
                pd.to_numeric(
                    pair_table["left_component"], errors="raise"
                ).eq(left_component),
                "scaled_ks_statistic",
            ],
            errors="raise",
        ).astype(float)
    )
    column_statistics = sorted(
        pd.to_numeric(
            pair_table.loc[
                pd.to_numeric(
                    pair_table["right_component"], errors="raise"
                ).eq(right_component),
                "scaled_ks_statistic",
            ],
            errors="raise",
        ).astype(float)
    )
    if len(row_statistics) < 2 or len(column_statistics) < 2:
        raise ValueError("Component assignment margin needs competitors")
    return float(
        min(
            row_statistics[1] - selected_statistic,
            column_statistics[1] - selected_statistic,
        )
    )


def minimum_cost_component_bijection(
    table: pd.DataFrame, left_modality: str, right_modality: str
) -> tuple[dict[int, int], float, float, float]:
    """Return the globally minimum KS-cost component permutation.

    The fourth return value is the maximum scaled KS statistic among pairs in
    the selected bijection.  The relative runner-up margin is zero for a tied
    optimum, which makes the downstream completion route abstain.
    """

    forward = table.loc[
        table["left_modality"].astype(str).eq(str(left_modality))
        & table["right_modality"].astype(str).eq(str(right_modality))
    ].copy()
    reverse = table.loc[
        table["left_modality"].astype(str).eq(str(right_modality))
        & table["right_modality"].astype(str).eq(str(left_modality))
    ].copy()
    if not forward.empty and not reverse.empty:
        raise ValueError("Component table contains both pair orientations")
    if forward.empty and reverse.empty:
        raise ValueError("Component table lacks the requested assay pair")
    if not reverse.empty:
        pair_table = reverse.rename(
            columns={
                "left_component": "right_component",
                "right_component": "left_component",
            }
        )
    else:
        pair_table = forward
    left_components = sorted(
        pd.to_numeric(
            pair_table["left_component"], errors="raise"
        ).astype(int).unique()
    )
    right_components = sorted(
        pd.to_numeric(
            pair_table["right_component"], errors="raise"
        ).astype(int).unique()
    )
    if len(left_components) != len(right_components) or len(left_components) < 2:
        raise ValueError("Component bijection requires equal nontrivial ranks")
    costs: dict[tuple[int, int], float] = {}
    for row in pair_table.itertuples(index=False):
        key = (int(row.left_component), int(row.right_component))
        if key in costs:
            raise ValueError("Component table contains a duplicate pair")
        costs[key] = float(row.scaled_ks_statistic)
    expected = set(product(left_components, right_components))
    if set(costs) != expected:
        raise ValueError("Component table is not a complete cost matrix")
    assignments: list[tuple[float, tuple[int, ...]]] = []
    for right_order in permutations(right_components):
        total = float(
            sum(
                costs[(left, right)]
                for left, right in zip(
                    left_components, right_order, strict=True
                )
            )
        )
        assignments.append((total, tuple(int(value) for value in right_order)))
    assignments.sort(key=lambda value: (value[0], value[1]))
    best_cost, best_order = assignments[0]
    second_cost = assignments[1][0]
    mapping = {
        left: right
        for left, right in zip(left_components, best_order, strict=True)
    }
    relative_margin = float(
        (second_cost - best_cost) / max(abs(best_cost), 1e-12)
    )
    maximum_assigned = float(
        max(costs[(left, mapping[left])] for left in left_components)
    )
    return mapping, float(best_cost), relative_margin, maximum_assigned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_seed(base: int, token: str) -> int:
    digest = hashlib.sha256(f"{base}::{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _verify_clean_layered_phase_a(
    upstream_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    directory = upstream_dir.expanduser().absolute().resolve(strict=True)
    manifest_path = directory / "phase_a_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError("Clean upstream Phase-A manifest hash mismatch")
    if manifest.get("phase") != "a":
        raise ValueError("Clean upstream artifact is not Phase A")
    if manifest.get("scorer_or_truth_opened") is not False:
        raise ValueError("Clean upstream Phase A opened scorer or truth")
    if manifest.get("all_observable_inputs_loaded_before_truth") is not True:
        raise ValueError("Clean upstream did not certify the pretruth barrier")
    if int(manifest.get("selected_world_count", -1)) != int(
        manifest.get("successful_world_count", -2)
    ):
        raise ValueError("Clean upstream Phase A has failed worlds")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"Clean upstream output hash mismatch: {name}")
    return directory, manifest


def build_assay_local_hypotheses_from_layered(
    feature_evidence: pd.DataFrame,
    direct_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct v4.33 from a clean production Phase-A projection."""

    feature_columns = {
        "feature",
        "modality",
        "state_v427_supported_member",
    }
    direct_columns = {
        "source_assay",
        "node1",
        "endpoint1",
        "node2",
        "endpoint2",
    }
    if missing := sorted(feature_columns.difference(feature_evidence.columns)):
        raise ValueError(f"Layered feature table is missing columns: {missing}")
    if missing := sorted(direct_columns.difference(direct_candidates.columns)):
        raise ValueError(f"Layered direct table is missing columns: {missing}")
    normalized_rows: list[dict[str, str]] = []
    for row in direct_candidates.itertuples(index=False):
        endpoint1 = str(row.endpoint1).strip().casefold()
        endpoint2 = str(row.endpoint2).strip().casefold()
        if (endpoint1, endpoint2) == ("tail", "arrow"):
            parent, child = str(row.node1), str(row.node2)
        elif (endpoint1, endpoint2) == ("arrow", "tail"):
            parent, child = str(row.node2), str(row.node1)
        else:
            raise ValueError("Clean direct candidate is not oriented tail-to-arrow")
        normalized_rows.append(
            {
                "source_assay": str(row.source_assay),
                "parent": parent,
                "child": child,
            }
        )
    state = feature_evidence.loc[:, sorted(feature_columns)].copy()
    direct = pd.DataFrame(normalized_rows)
    return equivalence.build_direct_parent_state_reporter_equivalence(
        direct, state
    ).decisions


def _residualize(values: FloatArray, covariate: FloatArray) -> FloatArray:
    design = np.column_stack([np.ones(len(covariate)), covariate])
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return np.asarray(values - design @ coefficients, dtype=float)


def focal_component_associations(
    focal: pd.DataFrame,
    component_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compute depth-adjusted within-assay focal/component correlations."""

    required_focal = {
        "cell",
        "modality",
        "observed_background_library_size",
    }
    if missing := sorted(required_focal.difference(focal.columns)):
        raise ValueError(f"Focal table is missing columns: {missing}")
    required_scores = {"cell_id", "modality"}
    if missing := sorted(required_scores.difference(component_scores.columns)):
        raise ValueError(f"Component table is missing columns: {missing}")
    modalities = set(focal["modality"].astype(str))
    if len(modalities) != 1:
        raise ValueError("Each focal table must contain exactly one modality")
    modality = modalities.pop()
    scores = component_scores.loc[
        component_scores["modality"].astype(str).eq(modality)
    ].copy()
    merged = focal.merge(
        scores,
        left_on=["modality", "cell"],
        right_on=["modality", "cell_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(focal) or len(merged) != len(scores):
        raise ValueError(f"Focal/component cell rows do not align for {modality}")
    component_columns = sorted(
        [column for column in scores if column.startswith("component_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if not component_columns:
        raise ValueError("No component score columns were found")
    feature_columns = [
        column for column in focal.columns if column not in FOCAL_METADATA
    ]
    depth = np.log1p(
        pd.to_numeric(
            merged["observed_background_library_size"], errors="raise"
        ).to_numpy(dtype=float)
    )
    component_residuals = {
        column: _residualize(
            pd.to_numeric(merged[column], errors="raise").to_numpy(dtype=float),
            depth,
        )
        for column in component_columns
    }
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        observed = pd.to_numeric(merged[feature], errors="raise").to_numpy(
            dtype=float
        )
        residual = _residualize(np.log1p(observed), depth)
        correlations: list[float] = []
        for column in component_columns:
            component = component_residuals[column]
            denominator = float(
                np.linalg.norm(residual) * np.linalg.norm(component)
            )
            correlation = (
                float(np.dot(residual, component) / denominator)
                if denominator > 1e-12
                else 0.0
            )
            correlations.append(correlation)
        absolute = np.abs(correlations)
        order = np.argsort(-absolute, kind="stable")
        best = float(absolute[order[0]])
        second = float(absolute[order[1]]) if len(order) > 1 else 0.0
        rows.append(
            {
                "modality": modality,
                "feature": feature,
                "cells": len(observed),
                "detected_cells": int(np.sum(observed > 0.0)),
                "dominant_component": int(order[0]),
                "dominant_absolute_partial_correlation": best,
                "component_relative_margin": float(
                    (best - second) / max(best, 1e-12)
                ),
                **{
                    f"component_{index}_partial_correlation": float(value)
                    for index, value in enumerate(correlations)
                },
            }
        )
    return pd.DataFrame(rows)


def bootstrap_focal_component_assignments(
    focal: pd.DataFrame,
    component_scores: pd.DataFrame,
    features: Sequence[str],
    *,
    replicates: int = REPORTER_BOOTSTRAP_REPLICATES,
    alpha: float = REPORTER_BOOTSTRAP_ALPHA,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Test reporter-component uniqueness by resampling within-assay cells."""

    if replicates < 32:
        raise ValueError("Reporter bootstrap needs at least 32 resamples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Reporter bootstrap alpha must lie in (0, 1)")
    requested = sorted({str(feature) for feature in features if str(feature)})
    associations = focal_component_associations(focal, component_scores)
    if missing := sorted(set(requested).difference(associations["feature"])):
        raise ValueError(f"Reporter bootstrap features are missing: {missing}")
    modalities = set(focal["modality"].astype(str))
    if len(modalities) != 1:
        raise ValueError("Reporter bootstrap accepts one assay at a time")
    modality = modalities.pop()
    scores = component_scores.loc[
        component_scores["modality"].astype(str).eq(modality)
    ].copy()
    merged = focal.merge(
        scores,
        left_on=["modality", "cell"],
        right_on=["modality", "cell_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(focal) or len(merged) != len(scores):
        raise ValueError("Reporter bootstrap cell rows do not align")
    component_columns = sorted(
        [column for column in scores if column.startswith("component_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    n_components = len(component_columns)
    if n_components < 2:
        raise ValueError("Reporter bootstrap needs competing components")
    depth = np.log1p(
        pd.to_numeric(
            merged["observed_background_library_size"], errors="raise"
        ).to_numpy(dtype=float)
    )
    components = merged[component_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    association_lookup = associations.set_index("feature", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for feature in requested:
        selected_component = int(
            association_lookup.loc[feature, "dominant_component"]
        )
        observed = np.log1p(
            pd.to_numeric(merged[feature], errors="raise").to_numpy(dtype=float)
        )
        rng = np.random.default_rng(
            _stable_seed(random_seed, f"{modality}::{feature}")
        )
        counts = np.zeros(n_components, dtype=int)
        for _ in range(replicates):
            indices = rng.integers(0, len(merged), size=len(merged))
            local_depth = depth[indices]
            response = _residualize(observed[indices], local_depth)
            local_components = _residualize(
                components[indices], local_depth
            )
            denominator = np.linalg.norm(response) * np.linalg.norm(
                local_components, axis=0
            )
            correlations = np.divide(
                response @ local_components,
                denominator,
                out=np.zeros(n_components, dtype=float),
                where=denominator > 1e-12,
            )
            counts[int(np.argmax(np.abs(correlations)))] += 1
        selected_count = int(counts[selected_component])
        competitor_counts = [
            int(counts[index])
            for index in range(n_components)
            if index != selected_component
        ]
        raw_p_values = [
            (
                float(
                    binomtest(
                        selected_count,
                        n=selected_count + competitor_count,
                        p=0.5,
                        alternative="greater",
                    ).pvalue
                )
                if selected_count + competitor_count > 0
                else 1.0
            )
            for competitor_count in competitor_counts
        ]
        maximum_raw_p = max(raw_p_values)
        bonferroni_p = min(1.0, (n_components - 1) * maximum_raw_p)
        rows.append(
            {
                "modality": modality,
                "feature": feature,
                "selected_component": selected_component,
                "bootstrap_replicates": replicates,
                **{
                    f"component_{index}_assignment_count": int(count)
                    for index, count in enumerate(counts)
                },
                "selected_component_assignment_count": selected_count,
                "selected_component_assignment_fraction": float(
                    selected_count / replicates
                ),
                "maximum_competitor_assignment_count": max(
                    competitor_counts
                ),
                "maximum_one_sided_binomial_p": maximum_raw_p,
                "bonferroni_component_assignment_p": bonferroni_p,
                "component_assignment_bootstrap_supported": bool(
                    bonferroni_p <= alpha
                ),
            }
        )
    return pd.DataFrame(rows, columns=BOOTSTRAP_STABILITY_COLUMNS)


def _component_axis_lookup(
    consensus: pd.DataFrame,
) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for axis, row in enumerate(consensus.itertuples(index=False)):
        assignment = json.loads(str(row.component_assignment_json))
        for modality, component in assignment.items():
            key = (str(modality), int(component))
            if key in lookup:
                raise ValueError("Consensus assignment is not a permutation")
            lookup[key] = int(axis)
    return lookup


def _build_v438_component_conditioned_equivalence(
    hypotheses: pd.DataFrame,
    associations: pd.DataFrame,
    consensus: pd.DataFrame,
    fold_consensus: pd.DataFrame,
    pairwise_matches: pd.DataFrame | None = None,
    fold_pairwise_matches: pd.DataFrame | None = None,
    fold_associations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach local components and build full-plus-fold-stable hyperedges."""

    if missing := sorted(set(HYPOTHESIS_COLUMNS).difference(hypotheses.columns)):
        raise ValueError(f"Hypothesis table is missing columns: {missing}")
    selected = hypotheses.loc[
        hypotheses["hypothesis_selected"].astype(bool)
    ].copy()
    if selected["modality"].duplicated().any():
        raise ValueError("Expected at most one selected hypothesis per modality")
    association_lookup = associations.set_index(
        ["modality", "feature"], verify_integrity=True
    )
    full_lookup = _component_axis_lookup(consensus)
    fold_lookups = {
        int(fold): _component_axis_lookup(frame)
        for fold, frame in fold_consensus.groupby("feature_fold", sort=True)
    }
    fold_frames = {
        int(fold): frame.reset_index(drop=True)
        for fold, frame in fold_consensus.groupby("feature_fold", sort=True)
    }
    if len(fold_lookups) < 2:
        raise ValueError("At least two feature-fold consensus fits are required")
    fold_association_columns = {
        "feature_fold",
        "modality",
        "feature",
        "dominant_component",
        "dominant_absolute_partial_correlation",
        "component_relative_margin",
    }
    if fold_associations is None:
        if not selected.empty:
            raise ValueError("Feature-fold focal associations are required")
        fold_associations = pd.DataFrame(columns=sorted(fold_association_columns))
    if missing := sorted(
        fold_association_columns.difference(fold_associations.columns)
    ):
        raise ValueError(f"Feature-fold associations are missing columns: {missing}")
    association_folds = {
        int(value)
        for value in pd.to_numeric(
            fold_associations["feature_fold"], errors="raise"
        ).tolist()
    }
    if not selected.empty and association_folds != set(fold_lookups):
        raise ValueError(
            "Feature-fold focal associations and consensus fits disagree"
        )
    fold_association_lookup = fold_associations.set_index(
        ["feature_fold", "modality", "feature"], verify_integrity=True
    )

    assay_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        reporters = [
            value
            for value in str(row.external_state_reporter_members).split(";")
            if value
        ]
        reporter_components = {
            int(
                association_lookup.loc[
                    (str(row.modality), reporter), "dominant_component"
                ]
            )
            for reporter in reporters
        }
        reporter_component = (
            next(iter(reporter_components))
            if len(reporter_components) == 1
            else None
        )
        reporter_fold_components = {
            reporter: {
                fold: int(
                    fold_association_lookup.loc[
                        (fold, str(row.modality), reporter),
                        "dominant_component",
                    ]
                )
                for fold in sorted(fold_lookups)
            }
            for reporter in reporters
        }
        reporter_components_stable = bool(
            reporter_component is not None
            and reporters
            and all(
                component == reporter_component
                for values in reporter_fold_components.values()
                for component in values.values()
            )
        )
        fold_reporter_components = {
            fold: (
                next(iter(components)) if len(components) == 1 else None
            )
            for fold in sorted(fold_lookups)
            for components in [
                {
                    values[fold]
                    for values in reporter_fold_components.values()
                }
            ]
        }
        reporter_context_components_identified = bool(
            reporter_component is not None
            and reporters
            and all(
                component is not None
                for component in fold_reporter_components.values()
            )
        )
        parsed_pairs = equivalence.parse_candidate_pairs(
            str(row.candidate_pairs_json)
        )
        compatible: list[tuple[str, str]] = []
        if reporter_component is not None:
            for left, right in parsed_pairs:
                parent = right if left in reporters else left
                parent_component = int(
                    association_lookup.loc[
                        (str(row.modality), parent), "dominant_component"
                    ]
                )
                if parent_component == reporter_component:
                    compatible.append(tuple(sorted((left, right))))
        fold_axes = {
            fold: (
                lookup[
                    (
                        str(row.modality),
                        int(fold_reporter_components[fold]),
                    )
                ]
                if fold_reporter_components[fold] is not None
                else None
            )
            for fold, lookup in fold_lookups.items()
        }
        reporter_strengths = [
            float(
                association_lookup.loc[
                    (str(row.modality), reporter),
                    "dominant_absolute_partial_correlation",
                ]
            )
            for reporter in reporters
        ]
        reporter_margins = [
            float(
                association_lookup.loc[
                    (str(row.modality), reporter),
                    "component_relative_margin",
                ]
            )
            for reporter in reporters
        ]
        fold_reporter_strengths = [
            float(
                fold_association_lookup.loc[
                    (fold, str(row.modality), reporter),
                    "dominant_absolute_partial_correlation",
                ]
            )
            for reporter in reporters
            for fold in sorted(fold_lookups)
        ]
        fold_reporter_margins = [
            float(
                fold_association_lookup.loc[
                    (fold, str(row.modality), reporter),
                    "component_relative_margin",
                ]
            )
            for reporter in reporters
            for fold in sorted(fold_lookups)
        ]
        assay_rows.append(
            {
                "hypothesis_id": str(row.hypothesis_id),
                "modality": str(row.modality),
                "external_state_reporter_members": ";".join(reporters),
                "candidate_pairs_json": str(row.candidate_pairs_json),
                "candidate_pair_count": int(row.candidate_pair_count),
                "component_compatible_candidate_pairs_json": json.dumps(
                    compatible, separators=(",", ":")
                ),
                "component_compatible_candidate_pair_count": len(compatible),
                "within_assay_pair_resolved_by_component": len(compatible) == 1,
                "reporter_component_identified": reporter_component is not None,
                "reporter_local_component": reporter_component,
                "reporter_minimum_absolute_partial_correlation": (
                    min(reporter_strengths) if reporter_strengths else 0.0
                ),
                "reporter_minimum_component_relative_margin": (
                    min(reporter_margins) if reporter_margins else 0.0
                ),
                "reporter_feature_fold_components_json": json.dumps(
                    reporter_fold_components,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "reporter_context_components_identified": (
                    reporter_context_components_identified
                ),
                "reporter_components_stable_across_full_and_feature_folds": (
                    reporter_components_stable
                ),
                "reporter_minimum_feature_fold_absolute_partial_correlation": (
                    min(fold_reporter_strengths)
                    if fold_reporter_strengths
                    else 0.0
                ),
                "reporter_minimum_feature_fold_component_relative_margin": (
                    min(fold_reporter_margins)
                    if fold_reporter_margins
                    else 0.0
                ),
                "full_consensus_axis": (
                    full_lookup[(str(row.modality), int(reporter_component))]
                    if reporter_component is not None
                    else None
                ),
                "feature_fold_axis_json": json.dumps(
                    fold_axes, sort_keys=True, separators=(",", ":")
                ),
                "axis_signature": (
                    json.dumps(
                        [
                            full_lookup[
                                (str(row.modality), int(reporter_component))
                            ],
                            *[fold_axes[key] for key in sorted(fold_axes)],
                        ],
                        separators=(",", ":"),
                    )
                    if reporter_context_components_identified
                    else ""
                ),
            }
        )
    assay = pd.DataFrame(assay_rows, columns=ASSAY_ASSIGNMENT_COLUMNS)
    hyperedge_rows: list[dict[str, Any]] = []
    eligible = assay.loc[
        assay["reporter_component_identified"].astype(bool)
        & assay[
            "reporter_components_stable_across_full_and_feature_folds"
        ].astype(bool)
    ]
    hyperedge_index = 0
    for signature, frame in eligible.groupby("axis_signature", sort=True):
        if len(frame) < 2:
            continue
        if frame["modality"].astype(str).duplicated().any():
            raise ValueError("An axis signature contains duplicate assays")
        modalities = sorted(frame["modality"].astype(str))
        signature_axes = [int(value) for value in json.loads(str(signature))]
        context_rows = [consensus.reset_index(drop=True).iloc[signature_axes[0]]]
        context_rows.extend(
            fold_frames[fold].iloc[axis]
            for fold, axis in zip(
                sorted(fold_frames), signature_axes[1:], strict=True
            )
        )
        context_cliques = [
            {
                value
                for value in str(row["maximum_match_clique_modalities"]).split(
                    ";"
                )
                if value
            }
            for row in context_rows
        ]
        full_components = {
            str(row.modality): int(row.reporter_local_component)
            for row in frame.itertuples(index=False)
        }
        resolved_by_modality = {
            str(row.modality): bool(
                row.within_assay_pair_resolved_by_component
            )
            for row in frame.itertuples(index=False)
        }

        def direct_pair_evidence(
            table: pd.DataFrame, left: str, right: str
        ) -> pd.Series:
            forward = table.loc[
                table["left_modality"].astype(str).eq(left)
                & table["right_modality"].astype(str).eq(right)
                & pd.to_numeric(
                    table["left_component"], errors="raise"
                ).eq(full_components[left])
                & pd.to_numeric(
                    table["right_component"], errors="raise"
                ).eq(full_components[right])
            ]
            reverse = table.loc[
                table["left_modality"].astype(str).eq(right)
                & table["right_modality"].astype(str).eq(left)
                & pd.to_numeric(
                    table["left_component"], errors="raise"
                ).eq(full_components[right])
                & pd.to_numeric(
                    table["right_component"], errors="raise"
                ).eq(full_components[left])
            ]
            evidence = pd.concat([forward, reverse], ignore_index=True)
            if len(evidence) != 1:
                raise ValueError("Missing pairwise component evidence")
            return evidence.iloc[0]

        minimum_context_clique_fraction = min(
            float(row["maximum_match_clique_fraction"])
            for row in context_rows
        )
        fold_tables = (
            [
                table
                for _, table in fold_pairwise_matches.groupby(
                    "feature_fold", sort=True
                )
            ]
            if fold_pairwise_matches is not None
            else []
        )
        pair_evidence: dict[frozenset[str], dict[str, Any]] = {}
        stable_edges: set[frozenset[str]] = set()
        for left, right in combinations(modalities, 2):
            full_evidence = (
                direct_pair_evidence(pairwise_matches, left, right)
                if pairwise_matches is not None
                else None
            )
            fold_evidence = [
                direct_pair_evidence(table, left, right)
                for table in fold_tables
            ]
            full_match = (
                bool(full_evidence["matched"])
                if full_evidence is not None
                else True
            )
            fold_match = all(bool(row["matched"]) for row in fold_evidence)
            assignment_margins = [
                *(
                    [
                        direct_component_assignment_margin(
                            pairwise_matches, full_evidence
                        )
                    ]
                    if pairwise_matches is not None
                    and full_evidence is not None
                    else []
                ),
                *[
                    direct_component_assignment_margin(table, row)
                    for table, row in zip(
                        fold_tables, fold_evidence, strict=True
                    )
                ],
            ]
            minimum_assignment_margin = (
                min(assignment_margins) if assignment_margins else float("inf")
            )
            assignments_unique = bool(minimum_assignment_margin > 0.0)
            local_resolved = bool(
                resolved_by_modality[left] and resolved_by_modality[right]
            )
            pair_in_all_context_cliques = all(
                {left, right}.issubset(clique) for clique in context_cliques
            )
            local_or_clique = bool(
                local_resolved or pair_in_all_context_cliques
            )
            edge_selected = bool(
                full_match
                and fold_match
                and assignments_unique
                and minimum_context_clique_fraction
                >= MINIMUM_CONTEXT_CLIQUE_FRACTION
                and local_or_clique
            )
            key = frozenset((left, right))
            pair_evidence[key] = {
                "full_match": bool(full_match),
                "fold_match": bool(fold_match),
                "assignments_unique": assignments_unique,
                "local_or_clique": local_or_clique,
                "edge_selected": edge_selected,
                "minimum_assignment_margin": minimum_assignment_margin,
            }
            if edge_selected:
                stable_edges.add(key)

        row_by_modality = {
            str(row.modality): row for row in frame.itertuples(index=False)
        }
        for component_modalities in stable_pair_graph_components(
            modalities, stable_edges
        ):
            hyperedge_index += 1
            component_frame = frame.loc[
                frame["modality"].astype(str).isin(component_modalities)
            ]
            reporters_by_modality = {
                modality: [
                    value
                    for value in str(
                        row_by_modality[
                            modality
                        ].external_state_reporter_members
                    ).split(";")
                    if value
                ]
                for modality in component_modalities
            }
            component_pairs = [
                frozenset(pair)
                for pair in combinations(component_modalities, 2)
            ]
            candidate_assay_pairs = [
                list(pair) for pair in combinations(component_modalities, 2)
            ]
            candidate_pairs = sorted(
                {
                    tuple(sorted((left, right)))
                    for left_modality, right_modality in combinations(
                        component_modalities, 2
                    )
                    for left, right in product(
                        reporters_by_modality[left_modality],
                        reporters_by_modality[right_modality],
                    )
                }
            )
            full_pairwise_match = all(
                pair_evidence[pair]["full_match"]
                for pair in component_pairs
            )
            fold_pairwise_match = all(
                pair_evidence[pair]["fold_match"]
                for pair in component_pairs
            )
            all_modalities_in_each_context_clique = all(
                set(component_modalities).issubset(clique)
                for clique in context_cliques
            )
            all_within_assay_resolved = bool(
                component_frame[
                    "within_assay_pair_resolved_by_component"
                ]
                .astype(bool)
                .all()
            )
            old_half_clique_guard = bool(
                full_pairwise_match
                and fold_pairwise_match
                and minimum_context_clique_fraction
                >= MINIMUM_CONTEXT_CLIQUE_FRACTION
            )
            old_local_or_clique_guard = (
                local_structure_or_all_context_clique_guard(
                    half_assay_clique_support=old_half_clique_guard,
                    all_within_assay_pairs_component_resolved=(
                        all_within_assay_resolved
                    ),
                    all_modalities_in_every_context_maximum_clique=(
                        all_modalities_in_each_context_clique
                    ),
                )
            )
            component_edges = [
                pair for pair in component_pairs if pair in stable_edges
            ]
            all_stable_edges_guarded = all(
                pair_evidence[pair]["local_or_clique"]
                for pair in component_edges
            )
            minimum_component_assignment_margin = min(
                float(pair_evidence[pair]["minimum_assignment_margin"])
                for pair in component_edges
            )
            all_component_assignments_unique = all(
                pair_evidence[pair]["assignments_unique"]
                for pair in component_edges
            )
            hyperedge_rows.append(
                {
                    "hyperedge_id": (
                        "component_conditioned_shared_state_"
                        f"{hyperedge_index:03d}"
                    ),
                    "axis_signature": str(signature),
                    "modalities": ";".join(component_modalities),
                    "n_modalities": len(component_modalities),
                    "candidate_assay_pairs_json": json.dumps(
                        candidate_assay_pairs, separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": len(candidate_assay_pairs),
                    "assay_participation_is_set_valued": (
                        len(component_modalities) > 2
                    ),
                    "exact_assay_participation_claimed": (
                        len(component_modalities) == 2
                    ),
                    "assay_local_hypothesis_ids": ";".join(
                        sorted(component_frame["hypothesis_id"].astype(str))
                    ),
                    "external_state_reporter_members": ";".join(
                        sorted(
                            value
                            for values in reporters_by_modality.values()
                            for value in values
                        )
                    ),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        candidate_pairs, separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(
                        candidate_pairs
                    ),
                    "minimum_reporter_absolute_partial_correlation": float(
                        component_frame[
                            "reporter_minimum_absolute_partial_correlation"
                        ].min()
                    ),
                    "minimum_reporter_component_relative_margin": float(
                        component_frame[
                            "reporter_minimum_component_relative_margin"
                        ].min()
                    ),
                    "minimum_feature_fold_reporter_absolute_partial_"
                    "correlation": float(
                        component_frame[
                            "reporter_minimum_feature_fold_absolute_"
                            "partial_correlation"
                        ].min()
                    ),
                    "minimum_feature_fold_reporter_component_relative_"
                    "margin": float(
                        component_frame[
                            "reporter_minimum_feature_fold_component_"
                            "relative_margin"
                        ].min()
                    ),
                    "all_reporter_components_stable_across_full_and_"
                    "feature_folds": bool(
                        component_frame[
                            "reporter_components_stable_across_full_and_"
                            "feature_folds"
                        ]
                        .astype(bool)
                        .all()
                    ),
                    "minimum_context_pairwise_match_support": float(
                        min(
                            float(row["pairwise_match_support"])
                            for row in context_rows
                        )
                    ),
                    "minimum_context_maximum_clique_fraction": float(
                        minimum_context_clique_fraction
                    ),
                    "minimum_context_direct_component_assignment_margin": (
                        minimum_component_assignment_margin
                    ),
                    "all_context_direct_component_assignments_unique": (
                        all_component_assignments_unique
                    ),
                    "full_fit_direct_component_pair_matched": bool(
                        full_pairwise_match
                    ),
                    "all_feature_fold_direct_component_pairs_matched": bool(
                        fold_pairwise_match
                    ),
                    "full_and_all_fold_direct_component_pairs_matched": bool(
                        full_pairwise_match and fold_pairwise_match
                    ),
                    "all_modalities_in_full_and_fold_maximum_cliques": bool(
                        all_modalities_in_each_context_clique
                    ),
                    "maximum_clique_stability_guard_passed": bool(
                        full_pairwise_match
                        and all_modalities_in_each_context_clique
                    ),
                    "half_assay_clique_support_guard_passed": (
                        old_half_clique_guard
                    ),
                    "all_within_assay_pairs_component_resolved": (
                        all_within_assay_resolved
                    ),
                    "local_structure_or_all_context_clique_guard_passed": (
                        old_local_or_clique_guard
                    ),
                    "stable_pair_graph_edge_count": len(component_edges),
                    "stable_pair_graph_possible_edge_count": len(
                        component_pairs
                    ),
                    "stable_pair_graph_is_complete": (
                        len(component_edges) == len(component_pairs)
                    ),
                    "stable_pair_graph_connected": True,
                    "stable_pair_graph_edges_pass_local_structure_or_pair_"
                    "clique_guard": all_stable_edges_guarded,
                    "hypothesis_selected": True,
                    "hypothesis_status": (
                        "retained_as_connected_cross_assay_shared_state_"
                        "equivalence"
                    ),
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                    "full_and_all_feature_folds_co_cluster": True,
                    "hypothesis_type": (
                        "cross_assay_shared_state_equivalence"
                    ),
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                }
            )
    return assay, pd.DataFrame(hyperedge_rows, columns=HYPEREDGE_COLUMNS)


def build_component_conditioned_equivalence(
    hypotheses: pd.DataFrame,
    associations: pd.DataFrame,
    consensus: pd.DataFrame,
    fold_consensus: pd.DataFrame,
    pairwise_matches: pd.DataFrame | None = None,
    fold_pairwise_matches: pd.DataFrame | None = None,
    fold_associations: pd.DataFrame | None = None,
    reporter_bootstrap_stability: pd.DataFrame | None = None,
    fold_reporter_bootstrap_stability: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build bootstrap-calibrated, set-valued shared-state hyperedges."""

    assay, _ = _build_v438_component_conditioned_equivalence(
        hypotheses,
        associations,
        consensus,
        fold_consensus,
        pairwise_matches=pairwise_matches,
        fold_pairwise_matches=fold_pairwise_matches,
        fold_associations=fold_associations,
    )
    if assay.empty:
        return (
            assay,
            pd.DataFrame(columns=HYPEREDGE_COLUMNS),
            pd.DataFrame(columns=PAIR_DECISION_COLUMNS),
        )
    bootstrap_columns = {
        "modality",
        "feature",
        "selected_component",
        "selected_component_assignment_fraction",
        "bonferroni_component_assignment_p",
        "component_assignment_bootstrap_supported",
    }
    if reporter_bootstrap_stability is None:
        raise ValueError("Full-fit reporter bootstrap evidence is required")
    if missing := sorted(
        bootstrap_columns.difference(reporter_bootstrap_stability.columns)
    ):
        raise ValueError(
            f"Full-fit reporter bootstrap evidence is missing: {missing}"
        )
    fold_bootstrap_columns = {"feature_fold", *bootstrap_columns}
    if fold_reporter_bootstrap_stability is None:
        raise ValueError("Feature-fold reporter bootstrap evidence is required")
    if missing := sorted(
        fold_bootstrap_columns.difference(
            fold_reporter_bootstrap_stability.columns
        )
    ):
        raise ValueError(
            f"Feature-fold reporter bootstrap evidence is missing: {missing}"
        )
    bootstrap_lookup = reporter_bootstrap_stability.set_index(
        ["modality", "feature"], verify_integrity=True
    )
    fold_bootstrap = fold_reporter_bootstrap_stability.copy()
    fold_bootstrap["feature_fold"] = pd.to_numeric(
        fold_bootstrap["feature_fold"], errors="raise"
    ).astype(int)
    fold_bootstrap_lookup = fold_bootstrap.set_index(
        ["feature_fold", "modality", "feature"], verify_integrity=True
    )
    fold_ids = sorted(
        pd.to_numeric(
            fold_consensus["feature_fold"], errors="raise"
        ).astype(int).unique()
    )
    if len(fold_ids) < 2:
        raise ValueError("At least two feature-fold bootstraps are required")
    observed_fold_ids = sorted(
        fold_bootstrap["feature_fold"].astype(int).unique()
    )
    if observed_fold_ids != fold_ids:
        raise ValueError(
            "Feature-fold bootstrap and consensus contexts disagree"
        )
    bootstrap_supported: list[bool] = []
    bootstrap_p_values: list[float] = []
    bootstrap_fractions: list[float] = []
    fold_bootstrap_support_json: list[str] = []
    fold_bootstrap_supported: list[bool] = []
    context_bootstrap_supported: list[bool] = []
    fold_bootstrap_p_values: list[float] = []
    fold_bootstrap_fractions: list[float] = []
    for row in assay.itertuples(index=False):
        if not bool(row.reporter_component_identified):
            bootstrap_supported.append(False)
            bootstrap_p_values.append(1.0)
            bootstrap_fractions.append(0.0)
            fold_bootstrap_support_json.append("{}")
            fold_bootstrap_supported.append(False)
            context_bootstrap_supported.append(False)
            fold_bootstrap_p_values.append(1.0)
            fold_bootstrap_fractions.append(0.0)
            continue
        reporters = [
            value
            for value in str(row.external_state_reporter_members).split(";")
            if value
        ]
        records = [
            bootstrap_lookup.loc[(str(row.modality), reporter)]
            for reporter in reporters
        ]
        if any(
            int(record["selected_component"])
            != int(row.reporter_local_component)
            for record in records
        ):
            raise ValueError("Bootstrap and full reporter components disagree")
        full_supported = bool(
            records
            and all(
                bool(record["component_assignment_bootstrap_supported"])
                for record in records
            )
        )
        bootstrap_supported.append(full_supported)
        bootstrap_p_values.append(
            max(
                (
                    float(record["bonferroni_component_assignment_p"])
                    for record in records
                ),
                default=1.0,
            )
        )
        bootstrap_fractions.append(
            min(
                (
                    float(record["selected_component_assignment_fraction"])
                    for record in records
                ),
                default=0.0,
            )
        )
        if not bool(row.reporter_context_components_identified):
            fold_bootstrap_support_json.append("{}")
            fold_bootstrap_supported.append(False)
            context_bootstrap_supported.append(False)
            fold_bootstrap_p_values.append(1.0)
            fold_bootstrap_fractions.append(0.0)
            continue
        fold_component_payload = json.loads(
            str(row.reporter_feature_fold_components_json)
        )
        fold_records: list[pd.Series] = []
        support_by_fold: dict[str, bool] = {}
        for fold in fold_ids:
            current_records = [
                fold_bootstrap_lookup.loc[
                    (fold, str(row.modality), reporter)
                ]
                for reporter in reporters
            ]
            for reporter, record in zip(
                reporters, current_records, strict=True
            ):
                expected_component = int(
                    fold_component_payload[reporter][str(fold)]
                )
                if int(record["selected_component"]) != expected_component:
                    raise ValueError(
                        "Bootstrap and feature-fold reporter components "
                        "disagree"
                    )
            fold_records.extend(current_records)
            support_by_fold[str(fold)] = bool(
                current_records
                and all(
                    bool(
                        record[
                            "component_assignment_bootstrap_supported"
                        ]
                    )
                    for record in current_records
                )
            )
        all_folds_supported = bool(
            support_by_fold
            and all(support_by_fold.values())
        )
        fold_bootstrap_support_json.append(
            json.dumps(
                support_by_fold, sort_keys=True, separators=(",", ":")
            )
        )
        fold_bootstrap_supported.append(all_folds_supported)
        context_bootstrap_supported.append(
            bool(full_supported and all_folds_supported)
        )
        fold_bootstrap_p_values.append(
            max(
                (
                    float(record["bonferroni_component_assignment_p"])
                    for record in fold_records
                ),
                default=1.0,
            )
        )
        fold_bootstrap_fractions.append(
            min(
                (
                    float(record["selected_component_assignment_fraction"])
                    for record in fold_records
                ),
                default=0.0,
            )
        )
    assay["reporter_bootstrap_component_assignment_supported"] = pd.Series(
        bootstrap_supported, index=assay.index, dtype=bool
    )
    assay["reporter_maximum_bootstrap_bonferroni_p"] = bootstrap_p_values
    assay["reporter_minimum_bootstrap_selected_fraction"] = (
        bootstrap_fractions
    )
    assay["reporter_feature_fold_bootstrap_supported_json"] = (
        fold_bootstrap_support_json
    )
    assay[
        "reporter_all_feature_fold_bootstrap_component_assignments_supported"
    ] = pd.Series(fold_bootstrap_supported, index=assay.index, dtype=bool)
    assay[
        "reporter_all_context_bootstrap_component_assignments_supported"
    ] = pd.Series(context_bootstrap_supported, index=assay.index, dtype=bool)
    assay[
        "reporter_maximum_feature_fold_bootstrap_bonferroni_p"
    ] = fold_bootstrap_p_values
    assay[
        "reporter_minimum_feature_fold_bootstrap_selected_fraction"
    ] = fold_bootstrap_fractions
    component_identified = (
        assay["reporter_component_identified"].astype(bool)
        & assay["reporter_context_components_identified"].astype(bool)
    )
    eligible = assay.loc[
        component_identified
        & assay["within_assay_pair_resolved_by_component"].astype(bool)
    ].copy()
    if eligible.empty:
        return (
            assay,
            pd.DataFrame(columns=HYPEREDGE_COLUMNS),
            pd.DataFrame(columns=PAIR_DECISION_COLUMNS),
        )
    if pairwise_matches is None or fold_pairwise_matches is None:
        raise ValueError("Context-majority matching requires pairwise evidence")
    ambiguity_alternative_pool = assay.loc[
        component_identified
        & assay[
            "reporter_all_context_bootstrap_component_assignments_supported"
        ].astype(bool)
    ].copy()

    fold_frames = {
        int(fold): frame.reset_index(drop=True)
        for fold, frame in fold_consensus.groupby("feature_fold", sort=True)
    }
    fold_tables = {
        int(fold): frame.reset_index(drop=True)
        for fold, frame in fold_pairwise_matches.groupby(
            "feature_fold", sort=True
        )
    }
    if set(fold_frames) != set(fold_tables) or len(fold_frames) < 2:
        raise ValueError("Feature-fold consensus and pairwise contexts disagree")
    context_frames = [consensus.reset_index(drop=True)] + [
        fold_frames[fold] for fold in sorted(fold_frames)
    ]
    context_tables = [pairwise_matches.reset_index(drop=True)] + [
        fold_tables[fold] for fold in sorted(fold_tables)
    ]
    context_count = len(context_frames)
    strict_majority = context_count // 2 + 1

    primary_modalities = sorted(eligible["modality"].astype(str).unique())
    alternative_modalities = sorted(
        ambiguity_alternative_pool["modality"].astype(str).unique()
    )
    analysis_modalities = sorted(
        set(primary_modalities).union(alternative_modalities)
    )
    analysis_pool = assay.loc[
        assay["modality"].astype(str).isin(analysis_modalities)
    ]
    row_by_modality = {
        str(row.modality): row
        for row in analysis_pool.itertuples(index=False)
    }
    context_axes: dict[str, list[int]] = {}
    context_components: dict[str, list[int]] = {}
    context_reporter_bootstrap_support: dict[str, list[bool]] = {}
    for modality, row in row_by_modality.items():
        axes = [int(value) for value in json.loads(str(row.axis_signature))]
        fold_payload = json.loads(
            str(row.reporter_feature_fold_components_json)
        )
        fold_components: list[int] = []
        for fold in sorted(fold_frames):
            values = {
                int(components[str(fold)])
                for components in fold_payload.values()
            }
            if len(values) != 1:
                raise ValueError("Reporter fold components are not identified")
            fold_components.append(next(iter(values)))
        components = [int(row.reporter_local_component), *fold_components]
        if len(axes) != context_count or len(components) != context_count:
            raise ValueError("Reporter context signature has the wrong length")
        context_axes[modality] = axes
        context_components[modality] = components
        fold_bootstrap_payload = json.loads(
            str(row.reporter_feature_fold_bootstrap_supported_json)
        )
        bootstrap_contexts = [
            bool(row.reporter_bootstrap_component_assignment_supported),
            *[
                bool(fold_bootstrap_payload[str(fold)])
                for fold in sorted(fold_frames)
            ],
        ]
        if len(bootstrap_contexts) != context_count:
            raise ValueError(
                "Reporter bootstrap context signature has the wrong length"
            )
        context_reporter_bootstrap_support[modality] = bootstrap_contexts

    def direct_pair_evidence(
        table: pd.DataFrame,
        left: str,
        right: str,
        left_component: int,
        right_component: int,
    ) -> pd.Series:
        forward = table.loc[
            table["left_modality"].astype(str).eq(left)
            & table["right_modality"].astype(str).eq(right)
            & pd.to_numeric(
                table["left_component"], errors="raise"
            ).eq(left_component)
            & pd.to_numeric(
                table["right_component"], errors="raise"
            ).eq(right_component)
        ]
        reverse = table.loc[
            table["left_modality"].astype(str).eq(right)
            & table["right_modality"].astype(str).eq(left)
            & pd.to_numeric(
                table["left_component"], errors="raise"
            ).eq(right_component)
            & pd.to_numeric(
                table["right_component"], errors="raise"
            ).eq(left_component)
        ]
        evidence = pd.concat([forward, reverse], ignore_index=True)
        if len(evidence) != 1:
            raise ValueError("Missing context-specific pairwise evidence")
        return evidence.iloc[0]

    pair_evidence: dict[frozenset[str], dict[str, Any]] = {}
    stable_edges: set[frozenset[str]] = set()
    primary_modality_set = set(primary_modalities)
    for left, right in combinations(analysis_modalities, 2):
        supports: list[bool] = []
        statistical_supports: list[bool] = []
        reporter_bootstrap_supports: list[bool] = []
        global_bijection_supports: list[bool] = []
        global_bijection_relative_margins: list[float] = []
        global_bijection_maximum_assigned_scaled_ks: list[float] = []
        matches: list[bool] = []
        unique_assignments: list[bool] = []
        clique_memberships: list[bool] = []
        assignment_margins: list[float] = []
        supported_axis_rows: list[pd.Series] = []
        for context_index, (frame, table) in enumerate(
            zip(context_frames, context_tables, strict=True)
        ):
            left_component = context_components[left][context_index]
            right_component = context_components[right][context_index]
            evidence = direct_pair_evidence(
                table,
                left,
                right,
                left_component,
                right_component,
            )
            margin = direct_component_assignment_margin(table, evidence)
            (
                global_mapping,
                _,
                global_relative_margin,
                global_maximum_assigned_scaled_ks,
            ) = minimum_cost_component_bijection(table, left, right)
            if "asymptotic_critical_value" not in table.columns:
                raise ValueError(
                    "Global-bijection completion requires the pairwise "
                    "asymptotic KS critical value"
                )
            pair_rows = table.loc[
                (
                    table["left_modality"].astype(str).eq(left)
                    & table["right_modality"].astype(str).eq(right)
                )
                | (
                    table["left_modality"].astype(str).eq(right)
                    & table["right_modality"].astype(str).eq(left)
                )
            ]
            pair_critical_value = float(
                pd.to_numeric(
                    pair_rows["asymptotic_critical_value"], errors="raise"
                ).min()
            )
            global_bijection_supported = bool(
                global_mapping[left_component] == right_component
                and global_relative_margin > 0.0
                and global_maximum_assigned_scaled_ks
                <= pair_critical_value
            )
            matched = bool(evidence["matched"])
            same_axis = bool(
                context_axes[left][context_index]
                == context_axes[right][context_index]
            )
            axis = context_axes[left][context_index]
            axis_row = frame.iloc[axis]
            clique = {
                value
                for value in str(
                    axis_row["maximum_match_clique_modalities"]
                ).split(";")
                if value
            }
            statistical_supported = bool(
                same_axis and matched and margin > 0.0
            )
            bootstrap_supported = bool(
                context_reporter_bootstrap_support[left][context_index]
                and context_reporter_bootstrap_support[right][context_index]
            )
            supported = bool(
                statistical_supported and bootstrap_supported
            )
            supports.append(supported)
            statistical_supports.append(statistical_supported)
            reporter_bootstrap_supports.append(bootstrap_supported)
            global_bijection_supports.append(global_bijection_supported)
            global_bijection_relative_margins.append(
                float(global_relative_margin)
            )
            global_bijection_maximum_assigned_scaled_ks.append(
                float(global_maximum_assigned_scaled_ks)
            )
            matches.append(matched)
            unique_assignments.append(margin > 0.0)
            clique_memberships.append({left, right}.issubset(clique))
            assignment_margins.append(float(margin))
            if supported:
                supported_axis_rows.append(axis_row)
        support_count = int(sum(supports))
        key = frozenset((left, right))
        pair_evidence[key] = {
            "supports": supports,
            "statistical_supports": statistical_supports,
            "reporter_bootstrap_supports": reporter_bootstrap_supports,
            "global_bijection_supports": global_bijection_supports,
            "global_bijection_relative_margins": (
                global_bijection_relative_margins
            ),
            "global_bijection_maximum_assigned_scaled_ks": (
                global_bijection_maximum_assigned_scaled_ks
            ),
            "matches": matches,
            "unique_assignments": unique_assignments,
            "clique_memberships": clique_memberships,
            "assignment_margins": assignment_margins,
            "supported_axis_rows": supported_axis_rows,
            "support_count": support_count,
            "support_fraction": float(support_count / context_count),
            "edge_selected": support_count >= strict_majority,
        }
        if (
            support_count >= strict_majority
            and {left, right}.issubset(primary_modality_set)
        ):
            stable_edges.add(key)

    graph_components = stable_pair_graph_components(
        primary_modalities, stable_edges
    )
    component_ids: dict[str, tuple[int, list[str]]] = {}
    component_retained: dict[int, bool] = {}
    component_selection_route: dict[int, str] = {}
    for component_index, component_modalities in enumerate(
        graph_components, start=1
    ):
        for modality in component_modalities:
            component_ids[modality] = (component_index, component_modalities)
        if len(component_modalities) > 2:
            component_retained[component_index] = True
            component_selection_route[component_index] = (
                "context_majority_multi_assay_component"
            )
        else:
            pair = frozenset(component_modalities)
            component_retained[component_index] = bool(
                pair_evidence[pair]["support_count"] == context_count
            )
            component_selection_route[component_index] = (
                "all_context_pairwise_core"
                if component_retained[component_index]
                else "unresolved_context_majority_pair"
            )

    retained_core_pairs = {
        frozenset(pair)
        for component_index, component_modalities in enumerate(
            graph_components, start=1
        )
        if component_retained[component_index]
        for pair in combinations(component_modalities, 2)
    }
    ambiguity_rescues: dict[int, dict[str, Any]] = {}
    for component_index, component_modalities in enumerate(
        graph_components, start=1
    ):
        if component_retained[component_index] or len(component_modalities) != 2:
            continue
        focal_pair = frozenset(component_modalities)
        focal_evidence = pair_evidence[focal_pair]
        statistical_supports = focal_evidence["statistical_supports"]
        bootstrap_supports = focal_evidence["reporter_bootstrap_supports"]
        if not (
            bool(focal_evidence["edge_selected"])
            and int(sum(statistical_supports)) == context_count - 1
            and bool(all(bootstrap_supports))
        ):
            continue
        missing_contexts = [
            index
            for index, supported in enumerate(statistical_supports)
            if not supported
        ]
        if len(missing_contexts) != 1:
            continue
        candidate_pairs = {focal_pair}
        for context_index in missing_contexts:
            for endpoint in sorted(focal_pair):
                for other in analysis_modalities:
                    if other == endpoint:
                        continue
                    alternative = frozenset((endpoint, other))
                    alternative_evidence = pair_evidence.get(alternative)
                    if alternative_evidence is None:
                        continue
                    if (
                        alternative_evidence["statistical_supports"][
                            context_index
                        ]
                        and alternative_evidence[
                            "reporter_bootstrap_supports"
                        ][context_index]
                    ):
                        candidate_pairs.add(alternative)
        candidate_pairs = {
            pair
            for pair in candidate_pairs
            if pair == focal_pair or pair not in retained_core_pairs
        }
        if len(candidate_pairs) < 2:
            continue
        candidate_modalities = {
            modality for pair in candidate_pairs for modality in pair
        }
        ambiguity_rescues[component_index] = {
            "focal_pair": focal_pair,
            "candidate_pairs": candidate_pairs,
            "missing_contexts": missing_contexts,
            "unresolved_local_alternative_modalities": sorted(
                candidate_modalities.difference(primary_modality_set)
            ),
        }

    global_bijection_completions: set[int] = set()
    for component_index, component_modalities in enumerate(
        graph_components, start=1
    ):
        if (
            component_retained[component_index]
            or component_index in ambiguity_rescues
            or len(component_modalities) != 2
        ):
            continue
        focal_pair = frozenset(component_modalities)
        focal_evidence = pair_evidence[focal_pair]
        if not (
            bool(focal_evidence["edge_selected"])
            and bool(all(focal_evidence["reporter_bootstrap_supports"]))
            and bool(all(focal_evidence["global_bijection_supports"]))
        ):
            continue
        component_retained[component_index] = True
        component_selection_route[component_index] = (
            "global_bijection_completion"
        )
        global_bijection_completions.add(component_index)
        retained_core_pairs.add(focal_pair)

    core_hyperedge_count = int(sum(component_retained.values()))
    for offset, component_index in enumerate(
        sorted(ambiguity_rescues), start=1
    ):
        ambiguity_rescues[component_index]["hyperedge_id"] = (
            "component_conditioned_shared_state_"
            f"{core_hyperedge_count + offset:03d}"
        )

    ambiguity_memberships: dict[frozenset[str], list[int]] = {}
    for component_index, rescue in ambiguity_rescues.items():
        for candidate_pair in rescue["candidate_pairs"]:
            ambiguity_memberships.setdefault(candidate_pair, []).append(
                component_index
            )

    pair_decision_rows: list[dict[str, Any]] = []
    alternative_modality_set = set(analysis_modalities)
    for left, right in combinations(analysis_modalities, 2):
        key = frozenset((left, right))
        evidence = pair_evidence[key]
        component_record = component_ids.get(left)
        same_component = bool(
            component_record is not None
            and component_ids.get(right) is not None
            and component_record[0] == component_ids[right][0]
        )
        component_index = component_record[0] if same_component else 0
        component_modalities = component_record[1] if same_component else []
        retained = bool(
            same_component and component_retained[component_index]
        )
        ambiguity_source_retained = bool(
            same_component and component_index in ambiguity_rescues
        )
        membership_indices = sorted(ambiguity_memberships.get(key, []))
        membership_candidate_pairs = {
            candidate_pair
            for index in membership_indices
            for candidate_pair in ambiguity_rescues[index]["candidate_pairs"]
        }
        raw_context_majority = bool(evidence["edge_selected"])
        graph_edge = key in stable_edges
        left_resolved = bool(
            row_by_modality[left]
            .within_assay_pair_resolved_by_component
        )
        right_resolved = bool(
            row_by_modality[right]
            .within_assay_pair_resolved_by_component
        )
        eligible_primary_pair = bool(
            {left, right}.issubset(primary_modality_set)
        )
        eligible_ambiguity_alternative = bool(
            {left, right}.issubset(alternative_modality_set)
        )
        left_bootstrap = bool(
            row_by_modality[left]
            .reporter_bootstrap_component_assignment_supported
        )
        right_bootstrap = bool(
            row_by_modality[right]
            .reporter_bootstrap_component_assignment_supported
        )
        left_fold_bootstrap = bool(
            row_by_modality[left]
            .reporter_all_feature_fold_bootstrap_component_assignments_supported
        )
        right_fold_bootstrap = bool(
            row_by_modality[right]
            .reporter_all_feature_fold_bootstrap_component_assignments_supported
        )
        both_all_context_bootstrap = bool(
            row_by_modality[left]
            .reporter_all_context_bootstrap_component_assignments_supported
            and row_by_modality[right]
            .reporter_all_context_bootstrap_component_assignments_supported
        )
        statistical_majority = bool(
            sum(evidence["statistical_supports"]) >= strict_majority
        )
        global_bijection_completed = bool(
            same_component
            and component_index in global_bijection_completions
        )
        if graph_edge and retained and len(component_modalities) > 2:
            status = "retained_in_multi_assay_participation_ambiguity"
        elif graph_edge and global_bijection_completed:
            status = (
                "retained_as_global_bijection_completed_assay_"
                "participation_equivalence"
            )
        elif graph_edge and retained:
            status = "retained_as_exact_assay_participation_equivalence"
        elif graph_edge and ambiguity_source_retained:
            status = "retained_in_cross_context_assay_pair_ambiguity"
        elif statistical_majority and not (
            left_bootstrap and right_bootstrap
        ):
            status = "unresolved_bootstrap_component_assignment"
        elif statistical_majority and not both_all_context_bootstrap:
            status = (
                "unresolved_feature_fold_bootstrap_component_assignment"
            )
        elif graph_edge and evidence["support_count"] < context_count:
            status = "unresolved_not_all_contexts_supported"
        elif same_component and retained:
            status = "included_only_as_unexpanded_transitive_ambiguity"
        elif membership_indices:
            status = (
                "included_only_as_unexpanded_cross_context_ambiguity_"
                "alternative"
            )
        else:
            status = "unresolved_insufficient_context_support"
        pair_decision_rows.append(
            {
                "left_modality": left,
                "right_modality": right,
                "statistical_context_support_pattern": "".join(
                    "1" if value else "0"
                    for value in evidence["statistical_supports"]
                ),
                "reporter_bootstrap_context_support_pattern": "".join(
                    "1" if value else "0"
                    for value in evidence["reporter_bootstrap_supports"]
                ),
                "context_support_pattern": "".join(
                    "1" if value else "0" for value in evidence["supports"]
                ),
                "context_count": context_count,
                "context_support_count": int(evidence["support_count"]),
                "context_support_fraction": float(
                    evidence["support_fraction"]
                ),
                "strict_context_majority_passed": raw_context_majority,
                "all_contexts_supported": bool(
                    evidence["support_count"] == context_count
                ),
                "left_reporter_bootstrap_supported": left_bootstrap,
                "right_reporter_bootstrap_supported": right_bootstrap,
                "both_reporters_bootstrap_supported": bool(
                    left_bootstrap and right_bootstrap
                ),
                "left_reporter_all_feature_fold_bootstrap_supported": (
                    left_fold_bootstrap
                ),
                "right_reporter_all_feature_fold_bootstrap_supported": (
                    right_fold_bootstrap
                ),
                "both_reporters_all_context_bootstrap_supported": (
                    both_all_context_bootstrap
                ),
                "left_within_assay_pair_resolved_by_component": (
                    left_resolved
                ),
                "right_within_assay_pair_resolved_by_component": (
                    right_resolved
                ),
                "eligible_for_primary_stable_pair_graph": (
                    eligible_primary_pair
                ),
                "eligible_as_cross_context_ambiguity_alternative": (
                    eligible_ambiguity_alternative
                ),
                "global_bijection_context_support_pattern": "".join(
                    "1" if value else "0"
                    for value in evidence["global_bijection_supports"]
                ),
                "all_context_global_bijection_supported": bool(
                    all(evidence["global_bijection_supports"])
                ),
                "minimum_context_global_bijection_relative_margin": float(
                    min(evidence["global_bijection_relative_margins"])
                ),
                "maximum_context_global_bijection_assigned_scaled_ks": float(
                    max(
                        evidence[
                            "global_bijection_maximum_assigned_scaled_ks"
                        ]
                    )
                ),
                "retained_by_global_bijection_completion": (
                    global_bijection_completed
                ),
                "included_in_output_ambiguity_set": bool(
                    membership_indices
                ),
                "ambiguity_hyperedge_ids": ";".join(
                    str(ambiguity_rescues[index]["hyperedge_id"])
                    for index in membership_indices
                ),
                "ambiguity_candidate_assay_pair_count": len(
                    membership_candidate_pairs
                ),
                "ambiguity_candidate_assay_pairs_json": json.dumps(
                    [
                        sorted(pair)
                        for pair in sorted(
                            membership_candidate_pairs,
                            key=lambda value: tuple(sorted(value)),
                        )
                    ],
                    separators=(",", ":"),
                ),
                "majority_graph_edge_selected": graph_edge,
                "connected_component_id": (
                    f"context_majority_component_{component_index:03d}"
                    if same_component
                    else ""
                ),
                "connected_component_size": len(component_modalities),
                "retained_in_output_hyperedge": bool(
                    retained or ambiguity_source_retained
                ),
                "selection_status": status,
                "edge_type": (
                    "cross_assay_shared_state"
                    if graph_edge
                    and (retained or ambiguity_source_retained)
                    else "unresolved"
                ),
                "candidate_pair_may_be_expanded_as_edge": False,
            }
        )

    hyperedge_rows: list[dict[str, Any]] = []
    hyperedge_index = 0
    for component_index, component_modalities in enumerate(
        graph_components, start=1
    ):
        if not component_retained[component_index]:
            continue
        hyperedge_index += 1
        component_frame = eligible.loc[
            eligible["modality"].astype(str).isin(component_modalities)
        ]
        component_pairs = [
            frozenset(pair) for pair in combinations(component_modalities, 2)
        ]
        component_edges = [
            pair for pair in component_pairs if pair in stable_edges
        ]
        reporters_by_modality = {
            modality: [
                value
                for value in str(
                    row_by_modality[modality].external_state_reporter_members
                ).split(";")
                if value
            ]
            for modality in component_modalities
        }
        candidate_pairs = sorted(
            {
                tuple(sorted((left, right)))
                for left_modality, right_modality in combinations(
                    component_modalities, 2
                )
                for left, right in product(
                    reporters_by_modality[left_modality],
                    reporters_by_modality[right_modality],
                )
            }
        )
        supported_rows = [
            row
            for pair in component_edges
            for row in pair_evidence[pair]["supported_axis_rows"]
        ]
        supported_margins = [
            margin
            for pair in component_edges
            for margin, supported in zip(
                pair_evidence[pair]["assignment_margins"],
                pair_evidence[pair]["supports"],
                strict=True,
            )
            if supported
        ]
        edge_support_counts = [
            int(pair_evidence[pair]["support_count"])
            for pair in component_edges
        ]
        all_context_clique = all(
            all(pair_evidence[pair]["clique_memberships"])
            for pair in component_edges
        )
        global_bijection_context_support = [
            all(
                pair_evidence[pair]["global_bijection_supports"][
                    context_index
                ]
                for pair in component_edges
            )
            for context_index in range(context_count)
        ]
        minimum_global_bijection_margin = min(
            margin
            for pair in component_edges
            for margin in pair_evidence[pair][
                "global_bijection_relative_margins"
            ]
        )
        maximum_global_bijection_assigned_scaled_ks = max(
            value
            for pair in component_edges
            for value in pair_evidence[pair][
                "global_bijection_maximum_assigned_scaled_ks"
            ]
        )
        candidate_assay_pairs = [
            list(pair) for pair in combinations(component_modalities, 2)
        ]
        hyperedge_rows.append(
            {
                "hyperedge_id": (
                    "component_conditioned_shared_state_"
                    f"{hyperedge_index:03d}"
                ),
                "axis_signature": json.dumps(
                    {
                        modality: context_axes[modality]
                        for modality in component_modalities
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "modalities": ";".join(component_modalities),
                "n_modalities": len(component_modalities),
                "candidate_assay_pairs_json": json.dumps(
                    candidate_assay_pairs, separators=(",", ":")
                ),
                "candidate_assay_pair_count": len(candidate_assay_pairs),
                "ambiguity_source_assay_pair_json": "[]",
                "ambiguity_context_indices_json": "[]",
                "ambiguity_candidate_generation_rule": (
                    "not_applicable_to_context_majority_core"
                ),
                "ambiguity_source_pair_all_context_reporter_bootstrap_"
                "supported": False,
                "ambiguity_alternatives_supported_in_disagreeing_context": (
                    False
                ),
                "ambiguity_candidate_pairs_exclude_retained_core_pairs": (
                    False
                ),
                "ambiguity_alternative_pool_rule": (
                    "not_applicable_to_context_majority_core"
                ),
                "ambiguity_unresolved_local_alternative_modalities_json": (
                    "[]"
                ),
                "ambiguity_all_focal_assays_within_assay_pair_resolved": (
                    True
                ),
                "ambiguity_all_candidate_modalities_within_assay_pair_"
                "resolved": True,
                "selection_route": component_selection_route[
                    component_index
                ],
                "global_bijection_context_support_pattern": "".join(
                    "1" if value else "0"
                    for value in global_bijection_context_support
                ),
                "all_context_global_bijection_supported": bool(
                    all(global_bijection_context_support)
                ),
                "minimum_context_global_bijection_relative_margin": float(
                    minimum_global_bijection_margin
                ),
                "maximum_context_global_bijection_assigned_scaled_ks": float(
                    maximum_global_bijection_assigned_scaled_ks
                ),
                "assay_participation_is_set_valued": (
                    len(component_modalities) > 2
                ),
                "exact_assay_participation_claimed": (
                    len(component_modalities) == 2
                ),
                "assay_local_hypothesis_ids": ";".join(
                    sorted(component_frame["hypothesis_id"].astype(str))
                ),
                "external_state_reporter_members": ";".join(
                    sorted(
                        value
                        for values in reporters_by_modality.values()
                        for value in values
                    )
                ),
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    candidate_pairs, separators=(",", ":")
                ),
                "candidate_cross_assay_feature_pair_count": len(
                    candidate_pairs
                ),
                "minimum_reporter_absolute_partial_correlation": float(
                    component_frame[
                        "reporter_minimum_absolute_partial_correlation"
                    ].min()
                ),
                "minimum_reporter_component_relative_margin": float(
                    component_frame[
                        "reporter_minimum_component_relative_margin"
                    ].min()
                ),
                "minimum_feature_fold_reporter_absolute_partial_"
                "correlation": float(
                    component_frame[
                        "reporter_minimum_feature_fold_absolute_"
                        "partial_correlation"
                    ].min()
                ),
                "minimum_feature_fold_reporter_component_relative_"
                "margin": float(
                    component_frame[
                        "reporter_minimum_feature_fold_component_"
                        "relative_margin"
                    ].min()
                ),
                "all_reporter_components_stable_across_full_and_"
                "feature_folds": bool(
                    component_frame[
                        "reporter_components_stable_across_full_and_"
                        "feature_folds"
                    ]
                    .astype(bool)
                    .all()
                ),
                "all_reporter_bootstrap_component_assignments_supported": (
                    bool(
                        component_frame[
                            "reporter_bootstrap_component_assignment_supported"
                        ]
                        .astype(bool)
                        .all()
                    )
                ),
                "maximum_reporter_bootstrap_bonferroni_p": float(
                    component_frame[
                        "reporter_maximum_bootstrap_bonferroni_p"
                    ].max()
                ),
                "minimum_reporter_bootstrap_selected_fraction": float(
                    component_frame[
                        "reporter_minimum_bootstrap_selected_fraction"
                    ].min()
                ),
                "all_reporter_feature_fold_bootstrap_component_assignments_"
                "supported": bool(
                    component_frame[
                        "reporter_all_feature_fold_bootstrap_component_"
                        "assignments_supported"
                    ]
                    .astype(bool)
                    .all()
                ),
                "all_reporter_context_bootstrap_component_assignments_"
                "supported": bool(
                    component_frame[
                        "reporter_all_context_bootstrap_component_"
                        "assignments_supported"
                    ]
                    .astype(bool)
                    .all()
                ),
                "maximum_reporter_feature_fold_bootstrap_bonferroni_p": (
                    float(
                        component_frame[
                            "reporter_maximum_feature_fold_bootstrap_"
                            "bonferroni_p"
                        ].max()
                    )
                ),
                "minimum_reporter_feature_fold_bootstrap_selected_"
                "fraction": float(
                    component_frame[
                        "reporter_minimum_feature_fold_bootstrap_selected_"
                        "fraction"
                    ].min()
                ),
                "context_count": context_count,
                "minimum_stable_pair_graph_context_support_count": min(
                    edge_support_counts
                ),
                "minimum_stable_pair_graph_context_support_fraction": float(
                    min(edge_support_counts) / context_count
                ),
                "all_stable_pair_graph_edges_have_strict_context_majority": (
                    all(value >= strict_majority for value in edge_support_counts)
                ),
                "minimum_context_pairwise_match_support": float(
                    min(
                        float(row["pairwise_match_support"])
                        for row in supported_rows
                    )
                ),
                "minimum_context_maximum_clique_fraction": float(
                    min(
                        float(row["maximum_match_clique_fraction"])
                        for row in supported_rows
                    )
                ),
                "minimum_context_direct_component_assignment_margin": min(
                    supported_margins
                ),
                "all_context_direct_component_assignments_unique": all(
                    all(pair_evidence[pair]["unique_assignments"])
                    for pair in component_edges
                ),
                "full_fit_direct_component_pair_matched": all(
                    pair_evidence[pair]["statistical_supports"][0]
                    for pair in component_edges
                ),
                "all_feature_fold_direct_component_pairs_matched": all(
                    all(pair_evidence[pair]["statistical_supports"][1:])
                    for pair in component_edges
                ),
                "full_and_all_fold_direct_component_pairs_matched": (
                    all(
                        all(pair_evidence[pair]["statistical_supports"])
                        for pair in component_edges
                    )
                ),
                "all_modalities_in_full_and_fold_maximum_cliques": (
                    all_context_clique
                ),
                "maximum_clique_stability_guard_passed": all_context_clique,
                "half_assay_clique_support_guard_passed": bool(
                    min(
                        float(row["maximum_match_clique_fraction"])
                        for row in supported_rows
                    )
                    >= MINIMUM_CONTEXT_CLIQUE_FRACTION
                ),
                "all_within_assay_pairs_component_resolved": True,
                "local_structure_or_all_context_clique_guard_passed": True,
                "stable_pair_graph_edge_count": len(component_edges),
                "stable_pair_graph_possible_edge_count": len(component_pairs),
                "stable_pair_graph_is_complete": (
                    len(component_edges) == len(component_pairs)
                ),
                "stable_pair_graph_connected": True,
                "stable_pair_graph_edges_pass_local_structure_or_pair_"
                "clique_guard": True,
                "hypothesis_selected": True,
                "hypothesis_status": (
                    "retained_as_context_majority_assay_participation_"
                    "ambiguity"
                    if len(component_modalities) > 2
                    else (
                        "retained_as_global_bijection_completed_shared_"
                        "state_equivalence"
                        if component_index in global_bijection_completions
                        else "retained_as_context_majority_shared_state_"
                        "equivalence"
                    )
                ),
                "fixed_selection_rule": FIXED_SELECTION_RULE,
                "full_and_all_feature_folds_co_cluster": (
                    all(
                        all(pair_evidence[pair]["statistical_supports"])
                        for pair in component_edges
                    )
                ),
                "hypothesis_type": "cross_assay_shared_state_equivalence",
                "exact_cross_assay_pair_claimed": False,
                "cross_assay_direction_claimed": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        )
    for component_index in sorted(ambiguity_rescues):
        rescue = ambiguity_rescues[component_index]
        focal_pair = rescue["focal_pair"]
        focal_evidence = pair_evidence[focal_pair]
        candidate_assay_pair_tuples = sorted(
            (tuple(sorted(pair)) for pair in rescue["candidate_pairs"]),
            key=lambda value: value,
        )
        ambiguity_modalities = sorted(
            {
                modality
                for pair in candidate_assay_pair_tuples
                for modality in pair
            }
        )
        component_frame = analysis_pool.loc[
            analysis_pool["modality"].astype(str).isin(
                ambiguity_modalities
            )
        ]
        reporters_by_modality = {
            modality: [
                value
                for value in str(
                    row_by_modality[modality]
                    .external_state_reporter_members
                ).split(";")
                if value
            ]
            for modality in ambiguity_modalities
        }
        candidate_feature_pairs = sorted(
            {
                tuple(sorted((left, right)))
                for left_modality, right_modality in candidate_assay_pair_tuples
                for left, right in product(
                    reporters_by_modality[left_modality],
                    reporters_by_modality[right_modality],
                )
            }
        )
        supported_rows = focal_evidence["supported_axis_rows"]
        supported_margins = [
            margin
            for margin, supported in zip(
                focal_evidence["assignment_margins"],
                focal_evidence["supports"],
                strict=True,
            )
            if supported
        ]
        all_context_clique = bool(
            all(focal_evidence["clique_memberships"])
        )
        hyperedge_rows.append(
            {
                "hyperedge_id": str(rescue["hyperedge_id"]),
                "axis_signature": json.dumps(
                    {
                        modality: context_axes[modality]
                        for modality in ambiguity_modalities
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "modalities": ";".join(ambiguity_modalities),
                "n_modalities": len(ambiguity_modalities),
                "candidate_assay_pairs_json": json.dumps(
                    candidate_assay_pair_tuples, separators=(",", ":")
                ),
                "candidate_assay_pair_count": len(
                    candidate_assay_pair_tuples
                ),
                "ambiguity_source_assay_pair_json": json.dumps(
                    sorted(focal_pair), separators=(",", ":")
                ),
                "ambiguity_context_indices_json": json.dumps(
                    rescue["missing_contexts"], separators=(",", ":")
                ),
                "ambiguity_candidate_generation_rule": (
                    "focal_context_majority_pair_plus_noncore_directly_"
                    "matched_bootstrap_supported_partners_in_the_single_"
                    "statistically_disagreeing_context"
                ),
                "ambiguity_source_pair_all_context_reporter_bootstrap_"
                "supported": bool(
                    all(focal_evidence["reporter_bootstrap_supports"])
                ),
                "ambiguity_alternatives_supported_in_disagreeing_context": (
                    True
                ),
                "ambiguity_candidate_pairs_exclude_retained_core_pairs": (
                    True
                ),
                "ambiguity_alternative_pool_rule": (
                    "primary_modalities_preserve_the_context_specific_"
                    "bootstrap_alternative_rule_while_additional_nonpoint_"
                    "modalities_require_identified_context_components_and_"
                    "all_context_reporter_bootstrap_support"
                ),
                "ambiguity_unresolved_local_alternative_modalities_json": (
                    json.dumps(
                        rescue[
                            "unresolved_local_alternative_modalities"
                        ],
                        separators=(",", ":"),
                    )
                ),
                "ambiguity_all_focal_assays_within_assay_pair_resolved": (
                    True
                ),
                "ambiguity_all_candidate_modalities_within_assay_pair_"
                "resolved": bool(
                    component_frame[
                        "within_assay_pair_resolved_by_component"
                    ]
                    .astype(bool)
                    .all()
                ),
                "selection_route": "cross_context_alternative_set",
                "global_bijection_context_support_pattern": "".join(
                    "1" if value else "0"
                    for value in focal_evidence[
                        "global_bijection_supports"
                    ]
                ),
                "all_context_global_bijection_supported": bool(
                    all(focal_evidence["global_bijection_supports"])
                ),
                "minimum_context_global_bijection_relative_margin": float(
                    min(
                        focal_evidence[
                            "global_bijection_relative_margins"
                        ]
                    )
                ),
                "maximum_context_global_bijection_assigned_scaled_ks": float(
                    max(
                        focal_evidence[
                            "global_bijection_maximum_assigned_scaled_ks"
                        ]
                    )
                ),
                "assay_participation_is_set_valued": True,
                "exact_assay_participation_claimed": False,
                "assay_local_hypothesis_ids": ";".join(
                    sorted(component_frame["hypothesis_id"].astype(str))
                ),
                "external_state_reporter_members": ";".join(
                    sorted(
                        value
                        for values in reporters_by_modality.values()
                        for value in values
                    )
                ),
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    candidate_feature_pairs, separators=(",", ":")
                ),
                "candidate_cross_assay_feature_pair_count": len(
                    candidate_feature_pairs
                ),
                "minimum_reporter_absolute_partial_correlation": float(
                    component_frame[
                        "reporter_minimum_absolute_partial_correlation"
                    ].min()
                ),
                "minimum_reporter_component_relative_margin": float(
                    component_frame[
                        "reporter_minimum_component_relative_margin"
                    ].min()
                ),
                "minimum_feature_fold_reporter_absolute_partial_"
                "correlation": float(
                    component_frame[
                        "reporter_minimum_feature_fold_absolute_"
                        "partial_correlation"
                    ].min()
                ),
                "minimum_feature_fold_reporter_component_relative_"
                "margin": float(
                    component_frame[
                        "reporter_minimum_feature_fold_component_"
                        "relative_margin"
                    ].min()
                ),
                "all_reporter_components_stable_across_full_and_"
                "feature_folds": bool(
                    component_frame[
                        "reporter_components_stable_across_full_and_"
                        "feature_folds"
                    ]
                    .astype(bool)
                    .all()
                ),
                "all_reporter_bootstrap_component_assignments_supported": (
                    bool(
                        component_frame[
                            "reporter_bootstrap_component_assignment_supported"
                        ]
                        .astype(bool)
                        .all()
                    )
                ),
                "maximum_reporter_bootstrap_bonferroni_p": float(
                    component_frame[
                        "reporter_maximum_bootstrap_bonferroni_p"
                    ].max()
                ),
                "minimum_reporter_bootstrap_selected_fraction": float(
                    component_frame[
                        "reporter_minimum_bootstrap_selected_fraction"
                    ].min()
                ),
                "all_reporter_feature_fold_bootstrap_component_assignments_"
                "supported": bool(
                    component_frame[
                        "reporter_all_feature_fold_bootstrap_component_"
                        "assignments_supported"
                    ]
                    .astype(bool)
                    .all()
                ),
                "all_reporter_context_bootstrap_component_assignments_"
                "supported": bool(
                    component_frame[
                        "reporter_all_context_bootstrap_component_"
                        "assignments_supported"
                    ]
                    .astype(bool)
                    .all()
                ),
                "maximum_reporter_feature_fold_bootstrap_bonferroni_p": (
                    float(
                        component_frame[
                            "reporter_maximum_feature_fold_bootstrap_"
                            "bonferroni_p"
                        ].max()
                    )
                ),
                "minimum_reporter_feature_fold_bootstrap_selected_"
                "fraction": float(
                    component_frame[
                        "reporter_minimum_feature_fold_bootstrap_selected_"
                        "fraction"
                    ].min()
                ),
                "context_count": context_count,
                "minimum_stable_pair_graph_context_support_count": int(
                    focal_evidence["support_count"]
                ),
                "minimum_stable_pair_graph_context_support_fraction": float(
                    focal_evidence["support_fraction"]
                ),
                "all_stable_pair_graph_edges_have_strict_context_majority": (
                    True
                ),
                "minimum_context_pairwise_match_support": float(
                    min(
                        float(row["pairwise_match_support"])
                        for row in supported_rows
                    )
                ),
                "minimum_context_maximum_clique_fraction": float(
                    min(
                        float(row["maximum_match_clique_fraction"])
                        for row in supported_rows
                    )
                ),
                "minimum_context_direct_component_assignment_margin": float(
                    min(supported_margins)
                ),
                "all_context_direct_component_assignments_unique": bool(
                    all(focal_evidence["unique_assignments"])
                ),
                "full_fit_direct_component_pair_matched": bool(
                    focal_evidence["statistical_supports"][0]
                ),
                "all_feature_fold_direct_component_pairs_matched": bool(
                    all(focal_evidence["statistical_supports"][1:])
                ),
                "full_and_all_fold_direct_component_pairs_matched": False,
                "all_modalities_in_full_and_fold_maximum_cliques": (
                    all_context_clique
                ),
                "maximum_clique_stability_guard_passed": all_context_clique,
                "half_assay_clique_support_guard_passed": bool(
                    min(
                        float(row["maximum_match_clique_fraction"])
                        for row in supported_rows
                    )
                    >= MINIMUM_CONTEXT_CLIQUE_FRACTION
                ),
                "all_within_assay_pairs_component_resolved": bool(
                    component_frame[
                        "within_assay_pair_resolved_by_component"
                    ]
                    .astype(bool)
                    .all()
                ),
                "local_structure_or_all_context_clique_guard_passed": True,
                "stable_pair_graph_edge_count": 1,
                "stable_pair_graph_possible_edge_count": len(
                    candidate_assay_pair_tuples
                ),
                "stable_pair_graph_is_complete": False,
                "stable_pair_graph_connected": True,
                "stable_pair_graph_edges_pass_local_structure_or_pair_"
                "clique_guard": True,
                "hypothesis_selected": True,
                "hypothesis_status": (
                    "retained_as_cross_context_assay_pair_ambiguity"
                ),
                "fixed_selection_rule": FIXED_SELECTION_RULE,
                "full_and_all_feature_folds_co_cluster": False,
                "hypothesis_type": "cross_assay_shared_state_ambiguity",
                "exact_cross_assay_pair_claimed": False,
                "cross_assay_direction_claimed": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        )
    return (
        assay,
        pd.DataFrame(hyperedge_rows, columns=HYPEREDGE_COLUMNS),
        pd.DataFrame(pair_decision_rows, columns=PAIR_DECISION_COLUMNS),
    )


def freeze_component_conditioned_bridge(
    *,
    discovery_dir: Path,
    evidence_dir: Path,
    hypothesis_table: Path | None,
    upstream_phase_a_dir: Path | None,
    block: str,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Freeze one exploratory cached-world bridge without opening scorer files."""

    discovery = discovery_dir.expanduser().absolute().resolve(strict=True)
    evidence = evidence_dir.expanduser().absolute().resolve(strict=True)
    if (hypothesis_table is None) == (upstream_phase_a_dir is None):
        raise ValueError(
            "Provide exactly one of hypothesis_table or upstream_phase_a_dir"
        )
    if any(part.casefold() in {"scorer", "truth"} for part in discovery.parts):
        raise ValueError("Discovery path crosses scorer/truth boundary")
    manifest = mdcr_scorer.verify_frozen_evidence(evidence)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    upstream_manifest: dict[str, Any] | None = None
    hypothesis_path: Path | None = None
    if upstream_phase_a_dir is not None:
        upstream, upstream_manifest = _verify_clean_layered_phase_a(
            upstream_phase_a_dir
        )
        feature_evidence = pd.read_csv(
            upstream / "phase_a_feature_evidence.tsv",
            sep="\t",
            usecols=[
                "block",
                "seed",
                "feature",
                "modality",
                "state_v427_supported_member",
            ],
            keep_default_na=False,
        )
        direct_candidates = pd.read_csv(
            upstream / "phase_a_direct_candidates.tsv",
            sep="\t",
            usecols=[
                "block",
                "seed",
                "source_assay",
                "node1",
                "endpoint1",
                "node2",
                "endpoint2",
            ],
            keep_default_na=False,
        )
        feature_world = feature_evidence.loc[
            feature_evidence["block"].astype(str).eq(str(block))
            & pd.to_numeric(feature_evidence["seed"], errors="raise").eq(
                int(seed)
            )
        ].copy()
        direct_world = direct_candidates.loc[
            direct_candidates["block"].astype(str).eq(str(block))
            & pd.to_numeric(direct_candidates["seed"], errors="raise").eq(
                int(seed)
            )
        ].copy()
        decisions = build_assay_local_hypotheses_from_layered(
            feature_world, direct_world
        )
        decisions.insert(0, "seed", int(seed))
        decisions.insert(0, "block", str(block))
        hypotheses = decisions.loc[:, HYPOTHESIS_COLUMNS].copy()
    else:
        assert hypothesis_table is not None
        hypothesis_path = (
            hypothesis_table.expanduser().absolute().resolve(strict=True)
        )
        hypotheses = pd.read_csv(
            hypothesis_path,
            sep="\t",
            usecols=HYPOTHESIS_COLUMNS,
            keep_default_na=False,
        )
        hypotheses = hypotheses.loc[
            hypotheses["block"].astype(str).eq(str(block))
            & pd.to_numeric(hypotheses["seed"], errors="raise").eq(int(seed))
        ].copy()
    if hypotheses.empty:
        raise ValueError("No upstream hypotheses match the requested world")
    component_scores = pd.read_csv(evidence / "component_scores.tsv", sep="\t")
    fold_component_scores = pd.read_csv(
        evidence / "fold_component_scores.tsv", sep="\t"
    )
    selected_hypotheses = hypotheses.loc[
        hypotheses["hypothesis_selected"].astype(bool)
    ]
    reporter_features_by_modality: dict[str, list[str]] = {}
    for row in selected_hypotheses.itertuples(index=False):
        reporter_features_by_modality.setdefault(
            str(row.modality), []
        ).extend(
            value
            for value in str(row.external_state_reporter_members).split(";")
            if value
        )
    association_frames: list[pd.DataFrame] = []
    fold_association_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    fold_bootstrap_frames: list[pd.DataFrame] = []
    for modality in sorted(hypotheses["modality"].astype(str).unique()):
        path = discovery / f"observed_focal__{modality}.tsv"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        focal = pd.read_csv(path, sep="\t")
        association_frames.append(
            focal_component_associations(focal, component_scores)
        )
        reporters = reporter_features_by_modality.get(modality, [])
        if reporters:
            bootstrap_frames.append(
                bootstrap_focal_component_assignments(
                    focal,
                    component_scores,
                    reporters,
                )
            )
        assay_fold_scores = fold_component_scores.loc[
            fold_component_scores["modality"].astype(str).eq(modality)
        ]
        for fold, scores in assay_fold_scores.groupby(
            "feature_fold", sort=True
        ):
            current = focal_component_associations(focal, scores)
            current.insert(0, "feature_fold", int(fold))
            fold_association_frames.append(current)
            if reporters:
                current_bootstrap = bootstrap_focal_component_assignments(
                    focal,
                    scores,
                    reporters,
                )
                current_bootstrap.insert(0, "feature_fold", int(fold))
                fold_bootstrap_frames.append(current_bootstrap)
    associations = pd.concat(association_frames, ignore_index=True)
    fold_associations = pd.concat(fold_association_frames, ignore_index=True)
    bootstrap_stability = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame(columns=BOOTSTRAP_STABILITY_COLUMNS)
    )
    fold_bootstrap_stability = (
        pd.concat(fold_bootstrap_frames, ignore_index=True)
        if fold_bootstrap_frames
        else pd.DataFrame(columns=FOLD_BOOTSTRAP_STABILITY_COLUMNS)
    )
    consensus = pd.read_csv(
        evidence / "consensus_component_tuples.tsv", sep="\t"
    )
    fold_consensus = pd.read_csv(
        evidence / "fold_consensus_component_tuples.tsv", sep="\t"
    )
    pairwise_matches = pd.read_csv(
        evidence / "pairwise_component_matches.tsv", sep="\t"
    )
    fold_pairwise_matches = pd.read_csv(
        evidence / "fold_pairwise_component_matches.tsv", sep="\t"
    )
    assay, hyperedges, pair_decisions = build_component_conditioned_equivalence(
        hypotheses,
        associations,
        consensus,
        fold_consensus,
        pairwise_matches=pairwise_matches,
        fold_pairwise_matches=fold_pairwise_matches,
        fold_associations=fold_associations,
        reporter_bootstrap_stability=bootstrap_stability,
        fold_reporter_bootstrap_stability=fold_bootstrap_stability,
    )
    out_dir.mkdir(parents=True)
    outputs = {
        "upstream_assay_local_hypotheses.tsv": hypotheses,
        "focal_component_associations.tsv": associations,
        "fold_focal_component_associations.tsv": fold_associations,
        "reporter_bootstrap_component_stability.tsv": bootstrap_stability,
        "fold_reporter_bootstrap_component_stability.tsv": (
            fold_bootstrap_stability
        ),
        "assay_axis_assignments.tsv": assay,
        "context_majority_pair_decisions.tsv": pair_decisions,
        "stable_cross_assay_hyperedges.tsv": hyperedges,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    provenance = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "block": str(block),
        "seed": int(seed),
        "information_boundary": {
            "held_out_background_component_scores_used": True,
            "disjoint_background_feature_fold_component_scores_used": True,
            "within_assay_focal_rows_used": True,
            "within_assay_cell_bootstrap_used": True,
            "within_assay_cell_bootstrap_repeated_in_each_background_"
            "feature_fold": True,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "scorer_truth_used": False,
            "external_features_or_priors_used": False,
        },
        "upstream_hypothesis_container_also_has_phase_b_scorer_columns": (
            hypothesis_path is not None
        ),
        "formal_pretruth_barrier_satisfied": upstream_manifest is not None,
        "interpretation": (
            "clean_production_phase_a_projection"
            if upstream_manifest is not None
            else "exploratory_phase_b_container_projection"
        ),
        "new_worlds_or_datasets_created": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "multi_assay_groups_are_assay_pair_ambiguity_sets": True,
        "assay_pair_alternatives_may_be_expanded_as_predictions": False,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "stable_pair_graph_connected_components_are_single_hyperedges": True,
        "transitive_component_alternatives_may_be_expanded_as_edges": False,
        "every_component_match_requires_strictly_positive_empirical_runner_"
        "up_assignment_margin": True,
        "reporter_component_instability_is_retained_as_set_valued_"
        "uncertainty": True,
        "each_stable_pair_requires_assay_local_component_resolution": True,
        "primary_stable_pair_graph_requires_within_assay_pair_resolution": (
            True
        ),
        "unresolved_within_assay_pair_modalities_may_enter_primary_graph": (
            False
        ),
        "unresolved_within_assay_pair_modalities_may_enter_only_as_"
        "ambiguity_alternatives": True,
        "ambiguity_alternative_requires_all_context_reporter_bootstrap_"
        "support": True,
        "each_stable_pair_requires_strict_context_majority": True,
        "two_assay_hyperedges_require_all_contexts_supported": False,
        "two_assay_hyperedges_require_all_context_pairwise_or_global_"
        "bijection_support": True,
        "global_bijection_completion_requires_prior_strict_context_"
        "majority": True,
        "global_bijection_completion_requires_all_context_reporter_"
        "bootstrap_support": True,
        "global_bijection_completion_requires_positive_global_runner_up_"
        "margin": True,
        "global_bijection_completion_requires_all_assigned_pairs_within_"
        "bonferroni_ks_bound": True,
        "global_bijection_completion_runs_only_after_ambiguity_abstention": (
            True
        ),
        "two_assay_hyperedges_require_reporter_bootstrap_support": True,
        "two_assay_hyperedges_require_reporter_bootstrap_support_in_full_"
        "and_every_feature_fold": True,
        "reporter_bootstrap_is_part_of_each_context_support_decision": True,
        "reporter_bootstrap_replicates": REPORTER_BOOTSTRAP_REPLICATES,
        "reporter_bootstrap_alpha": REPORTER_BOOTSTRAP_ALPHA,
        "reporter_bootstrap_runner_up_tests_bonferroni_corrected": True,
        "maximum_clique_evidence_is_diagnostic_not_a_selection_bypass": True,
        "minimum_context_clique_fraction": (
            MINIMUM_CONTEXT_CLIQUE_FRACTION
        ),
        "mdcr_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "clean_upstream_phase_a_manifest_payload_sha256": (
            upstream_manifest["manifest_payload_sha256"]
            if upstream_manifest is not None
            else None
        ),
        "source_sha256": {
            Path(__file__).name: _sha256(Path(__file__).resolve(strict=True)),
            "direct_parent_state_reporter_equivalence_v4.py": _sha256(
                SCRIPT_DIR / "direct_parent_state_reporter_equivalence_v4.py"
            ),
            "score_unpaired_mdcr_ks_baseline_v4.py": _sha256(
                SCRIPT_DIR / "score_unpaired_mdcr_ks_baseline_v4.py"
            ),
        },
        "input_sha256": {
            **(
                {"hypothesis_table": _sha256(hypothesis_path)}
                if hypothesis_path is not None
                else {
                    "clean_upstream_phase_a_manifest": _sha256(
                        upstream_phase_a_dir
                        / "phase_a_manifest.json"
                    )
                }
            ),
            **{
                f"observed_focal__{modality}.tsv": _sha256(
                    discovery / f"observed_focal__{modality}.tsv"
                )
                for modality in sorted(hypotheses["modality"].astype(str).unique())
            },
        },
        "output_sha256": {
            name: _sha256(out_dir / name) for name in outputs
        },
    }
    provenance["provenance_payload_sha256"] = _canonical_sha256(provenance)
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    upstream = parser.add_mutually_exclusive_group(required=True)
    upstream.add_argument("--hypothesis-table", type=Path)
    upstream.add_argument("--upstream-phase-a-dir", type=Path)
    parser.add_argument("--block", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = freeze_component_conditioned_bridge(
        discovery_dir=args.discovery_dir,
        evidence_dir=args.evidence_dir,
        hypothesis_table=args.hypothesis_table,
        upstream_phase_a_dir=args.upstream_phase_a_dir,
        block=args.block,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
