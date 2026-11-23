#!/usr/bin/env python3
"""Observed-only hierarchical collider replacement for six disjoint assays.

The module estimates assay-local latent-state/depth nuisance from a frozen
background-only embedding, fits cross-fitted sparse Bernoulli models for every
directed two-parent star, and calibrates residual arms against state/depth
preserving empirical nulls.  A conservative gate, a fold-stable uncertainty
tier, and a frozen state-gated rank-fusion fallback may replace one assay's
current direct motif.  No feature roles, truth, simulator parameters, paired
cells, or cross-assay coordinate registration are accepted.

Displaced edges remain in ``demoted_or_unresolved`` for provenance.  Their
``was_base_selected`` flag is inherited from the existing six base graph
views; downstream graph code should retain only base-selected rows as ordinary
unresolved adjacencies.  Rescue-only displaced rows remain diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import expit, xlogy
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from observed_collider_structure import conditional_outer_correlation
from structure_aware_collider_rescue import aggregate_vote_evidence, canonical_edge


FloatArray = NDArray[np.float64]
SCHEMA_VERSION = "crossfit_hierarchical_collider.v4.22"
DIRECT_EDGE_TYPE = "within_assay_direct_candidate"
UNRESOLVED_EDGE_TYPE = "unresolved"
RESIDUAL_FIRST_REPLACEMENT_ACTION = (
    "tier1r_residual_first_exact_production_disagreement"
)
UNORIENTED_RESCUE_OVERLAP_ACTION = (
    "zero_edge_unique_rescue_rank_overlap_unoriented_adjacency"
)
UNORIENTED_RESCUE_MIN_SELECTED_VOTE_FRACTION = 5.0 / 6.0
UNORIENTED_RESCUE_MAX_MEAN_EMPIRICAL_P_POSITIVE = 0.020
UNORIENTED_RESCUE_MIN_MEAN_POOLED_LOG_ODDS = 1.00
RETAINED_ORIENTATION_DEMOTION_ACTION = (
    "demote_retained_orientation_ambiguous_production_motif"
)
MULTI_COMPONENT_COLLAPSE_ACTION = (
    "collapse_multi_component_production_to_unique_crossfit_star"
)
ZERO_EDGE_STATE_CONSISTENT_RUNNER_ACTION = (
    "zero_edge_unique_state_consistent_runner_completion"
)
ZERO_EDGE_RESCUE_ANCHORED_COMPLETION_ACTION = (
    "zero_edge_unique_rescue_anchored_state_consistent_completion"
)
ZERO_EDGE_RESCUE_ORIENTATION_COMPLETION_ACTION = (
    "zero_edge_unanimous_rescue_orientation_sparse_completion"
)
ZERO_EDGE_DOMINANT_RESCUE_COMPLETION_ACTION = (
    "zero_edge_dominant_competing_rescue_sparse_completion"
)
ZERO_EDGE_ALL_STATE_TOPOLOGY_COMPLETION_ACTION = (
    "zero_edge_unique_all_state_topology_completion"
)
WEAK_MULTICENTER_REPLACEMENT_ACTION = (
    "replace_weak_observed_multicenter_with_unique_state_consistent_star"
)
WEAK_MULTICENTER_DEMOTION_ACTION = (
    "demote_weak_observed_multicenter_without_unique_validated_star"
)
CENTRALITY_SAME_SKELETON_REPLACEMENT_ACTION = (
    "constrain_weak_centrality_reorientation_to_state_consistent_skeleton"
)
CENTRALITY_PRECOMPLETION_REVERSION_ACTION = (
    "revert_weak_centrality_reorientation_to_stable_precompletion"
)
CENTRALITY_UNRECOVERABLE_DEMOTION_ACTION = (
    "demote_weak_centrality_reorientation_without_recoverable_completion"
)
DUAL_ANCHOR_RESIDUAL_REPLACEMENT_ACTION = (
    "replace_residually_inconsistent_dual_anchor_completion"
)
STATE_INCONSISTENT_NEIGHBOR_REPLACEMENT_ACTION = (
    "replace_state_inconsistent_production_with_unique_supported_neighbor_star"
)
STATE_LOADED_PARENT_ROTATION_ACTION = (
    "rotate_state_loaded_production_parent_to_residual_center_star"
)
STATE_CENTER_SIBLING_ROTATION_ACTION = (
    "replace_state_loaded_center_with_loading_equivalent_sibling_rotation"
)
STATE_INCONSISTENT_SAME_CENTER_SUBSTITUTION_ACTION = (
    "replace_state_inconsistent_parent_with_stable_same_center_candidate"
)
ZERO_EDGE_SPOKE_CONSENSUS_ACTION = (
    "replace_zero_edge_overcomplete_spokes_with_beta_consensus_star"
)
AMBIGUOUS_SKELETON_REORIENTATION_ACTION = (
    "reorient_ambiguous_production_on_residually_supported_skeleton"
)
OBSERVED_CENTRALITY_REORIENTATION_ACTION = (
    "reorient_completion_by_assay_beta_centrality"
)
OBSERVED_DUAL_ANCHOR_COMPLETION_ACTION = (
    "complete_dual_supported_unshielded_anchor"
)
FORBIDDEN_INPUT_TOKENS = (
    "truth",
    "role",
    "oracle",
    "simulator",
    "paired_cell",
    "counterfactual",
)


# The zero-edge fallback is intentionally small and frozen.  It was fitted once
# on the retired TRAIN330/360/370/380 candidate tables, with entire TRAIN blocks
# held out while choosing the sparse model.  The rank-fusion runtime path only
# evaluates these nine observed inputs; it invokes no estimator fit and never
# updates the frozen weights.  Other evidence-building paths in this module do
# use sklearn for deterministic folds and microstate clustering.
ZERO_EDGE_RANK_FUSION_MODEL_VERSION = "retired_train_330_380_sparse_l1_v1"
ZERO_EDGE_RANK_FUSION_DERIVATION_CONTRACT_SCHEMA_VERSION = (
    "ca_boss_v4.zero_edge_rank_fusion_derivation_contract.v1"
)
ZERO_EDGE_RANK_FUSION_DERIVATION_CONTRACT_SHA256 = (
    "130a4c7d5c2236d293850c95c60388981cdfc87e0672ed8b33c61843542d233a"
)
ZERO_EDGE_RANK_FUSION_RETIRED_PHASE_B_MANIFEST_SHA256 = (
    "71ac33cce950aa26194e35a45ebd51fac383fb0fa7c3ba659129031835b2d7da"
)
ZERO_EDGE_RANK_FUSION_COEFFICIENT_DECIMAL_PLACES = 9
ZERO_EDGE_RANK_FUSION_REQUIRED_FEATURES = 6
ZERO_EDGE_RANK_FUSION_SCHEMA: tuple[tuple[str, str], ...] = (
    ("direct", "crossfit_min_drop_one_gain"),
    ("direct", "laplace_mean_parent_coefficient_min"),
    ("direct", "laplace_both_practical_fold_support"),
    ("direct", "center_detection_rate"),
    ("direct", "residual_star_separation_reverse_center_margin"),
    ("reverse", "crossfit_min_drop_one_gain"),
    ("outer", "minimum_outer_conditional_wald_view_support"),
    ("outer", "minimum_parent_detection_rate"),
    ("outer", "fold_maximum_outer_absolute_correlation"),
)
ZERO_EDGE_RANK_FUSION_MEDIANS: tuple[float, ...] = (
    -0.00045129999314269997,
    -0.0153146600668196,
    0.0,
    0.0782051282051282,
    0.0,
    -0.00045129999314269997,
    0.0,
    0.0724358974358974,
    0.08095664113933485,
)
ZERO_EDGE_RANK_FUSION_WEIGHTS: tuple[float, ...] = (
    282.903889009,
    1.439185288,
    0.764827299,
    -0.800295987,
    8.426512962,
    -1.526146021,
    0.031428019,
    -3.168528055,
    1.306819242,
)
ZERO_EDGE_RANK_FUSION_INTERCEPT = -2.455486221


def _zero_edge_rank_fusion_model_payload() -> dict[str, Any]:
    return {
        "model_version": ZERO_EDGE_RANK_FUSION_MODEL_VERSION,
        "retired_training_blocks": ["TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380"],
        "selection_rule": (
            "sparsest_l1_within_one_exact_assay_of_best_retired_block_oof"
        ),
        "fit": {
            "penalty": "l1",
            "C": 0.003,
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": 19,
        },
        "context_aggregation": "highest_raw_generative_score_star",
        "derivation_contract": {
            "schema_version": (
                ZERO_EDGE_RANK_FUSION_DERIVATION_CONTRACT_SCHEMA_VERSION
            ),
            "sha256": ZERO_EDGE_RANK_FUSION_DERIVATION_CONTRACT_SHA256,
            "retired_phase_b_manifest_sha256": (
                ZERO_EDGE_RANK_FUSION_RETIRED_PHASE_B_MANIFEST_SHA256
            ),
        },
        "feature_schema": [
            f"{context}__{feature}"
            for context, feature in ZERO_EDGE_RANK_FUSION_SCHEMA
        ],
        "imputation_medians": list(ZERO_EDGE_RANK_FUSION_MEDIANS),
        "raw_scale_weights": list(ZERO_EDGE_RANK_FUSION_WEIGHTS),
        "raw_scale_intercept": ZERO_EDGE_RANK_FUSION_INTERCEPT,
        "coefficient_canonical_decimal_places": (
            ZERO_EDGE_RANK_FUSION_COEFFICIENT_DECIMAL_PLACES
        ),
        "runtime_refit": False,
    }


def _zero_edge_rank_fusion_model_sha256() -> str:
    payload = json.dumps(
        _zero_edge_rank_fusion_model_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CrossfitHierarchicalColliderConfig:
    """Frozen statistical and conservative selection settings."""

    n_folds: int = 5
    parent_prior_sd: float = 2.5
    nuisance_prior_sd: float = 5.0
    microstate_resolutions: tuple[int, ...] = (8, 16)
    null_permutations: int = 31
    required_assays: int = 6
    required_features_per_assay: int = 6
    required_evidence_views: int = 6
    minimum_base_view_count: int = 4

    tier1_minimum_parent_positive_posterior: float = 0.98
    tier1_minimum_joint_gain_exclusive: float = 0.005
    tier1_minimum_drop_one_gain_exclusive: float = 0.0
    tier1_minimum_both_positive_fold_support: float = 0.80
    tier1_minimum_reverse_center_gain_margin_exclusive: float = 0.0
    tier1_maximum_arm_empirical_positive_p: float = 0.125
    tier1_minimum_eligible_runner_score_margin: float = 0.003
    tier1_minimum_exact_production_score_improvement: float = 0.003
    tier1_minimum_nonexact_score: float = 0.040

    tier1u_minimum_crossfit_joint_gain_fold_min_exclusive: float = 0.0
    tier1u_minimum_laplace_fold_parent_positive_posterior: float = 0.99
    tier1u_minimum_both_positive_fold_support: float = 1.0
    tier1u_minimum_both_practical_fold_support: float = 1.0
    tier1u_maximum_arm_empirical_positive_p: float = 1.0 / 32.0
    tier1u_minimum_outer_empirical_null_compatibility_p: float = 0.50
    tier1u_minimum_eligible_runner_score_margin: float = 0.010
    tier1u_minimum_exact_production_score_improvement: float = 0.020

    tier1r_minimum_crossfit_joint_gain_fold_min_exclusive: float = 0.0
    tier1r_minimum_crossfit_min_drop_one_gain_exclusive: float = 0.0
    tier1r_minimum_laplace_mean_parent_positive_posterior: float = 0.90
    tier1r_minimum_laplace_fold_parent_positive_posterior: float = 0.70
    tier1r_minimum_both_positive_fold_support: float = 1.0
    tier1r_minimum_both_practical_fold_support: float = 1.0
    tier1r_maximum_arm_empirical_positive_p: float = 1.0 / 16.0
    tier1r_minimum_outer_empirical_null_compatibility_p: float = 0.90
    tier1r_minimum_global_runner_score_margin: float = 0.003
    tier1r_minimum_exact_production_score_improvement: float = 0.003
    tier1r_require_center_change: bool = True

    multicenter_minimum_production_fold_parent_posterior: float = 0.85
    multicenter_minimum_production_drop_one_gain_exclusive: float = 0.0
    multicenter_alternative_minimum_parent_posterior: float = 0.90
    multicenter_alternative_minimum_fold_parent_posterior: float = 0.85
    multicenter_alternative_minimum_both_positive_fold_support: float = 1.0
    multicenter_alternative_minimum_both_practical_fold_support: float = 1.0
    multicenter_alternative_maximum_arm_empirical_positive_p: float = 0.25
    multicenter_alternative_minimum_outer_empirical_null_compatibility_p: float = 0.90
    multicenter_alternative_minimum_outer_wald_view_support: float = 1.0
    multicenter_alternative_minimum_raw_generative_score: float = 0.020
    multicenter_alternative_minimum_drop_one_gain_exclusive: float = 0.0
    multicenter_alternative_minimum_current_skeleton_arm_support: int = 1

    centrality_full_minimum_contrast: float = 1.50
    centrality_full_minimum_parent_posterior: float = 0.90
    centrality_full_minimum_fold_parent_posterior: float = 0.85
    centrality_full_minimum_both_positive_fold_support: float = 1.0
    centrality_full_minimum_both_practical_fold_support: float = 1.0
    centrality_full_maximum_arm_empirical_positive_p: float = 0.125
    centrality_full_minimum_drop_one_gain_exclusive: float = 0.0
    centrality_skeleton_minimum_parent_posterior: float = 0.85
    centrality_skeleton_minimum_fold_parent_posterior: float = 0.65
    centrality_skeleton_minimum_both_positive_fold_support: float = 1.0
    centrality_skeleton_minimum_both_practical_fold_support: float = 0.80
    centrality_skeleton_maximum_arm_empirical_positive_p: float = 0.25
    centrality_skeleton_minimum_outer_wald_view_support: float = 1.0
    centrality_skeleton_minimum_raw_generative_score: float = 0.020

    dual_anchor_maximum_production_collider_separation_exclusive: float = -0.20
    dual_anchor_minimum_observed_winner_fraction: float = 1.0
    dual_anchor_alternative_minimum_parent_posterior: float = 0.75
    dual_anchor_alternative_minimum_fold_parent_posterior: float = 0.65
    dual_anchor_alternative_minimum_both_positive_fold_support: float = 1.0
    dual_anchor_alternative_minimum_both_practical_fold_support: float = 1.0
    dual_anchor_alternative_maximum_arm_empirical_positive_p: float = 0.25
    dual_anchor_alternative_minimum_outer_wald_view_support: float = 1.0
    dual_anchor_alternative_minimum_raw_generative_score: float = 0.018
    dual_anchor_alternative_minimum_joint_gain: float = 0.004
    dual_anchor_alternative_minimum_collider_separation: float = -0.05
    dual_anchor_alternative_minimum_collider_separation_improvement: float = 0.20
    dual_anchor_structural_minimum_parent_posterior: float = 0.70
    dual_anchor_structural_minimum_fold_parent_posterior: float = 0.60
    dual_anchor_structural_minimum_both_positive_fold_support: float = 1.0
    dual_anchor_structural_minimum_both_practical_fold_support: float = 0.80
    dual_anchor_structural_minimum_raw_generative_score: float = 0.005
    dual_anchor_structural_minimum_joint_gain: float = 0.001
    dual_anchor_structural_minimum_collider_separation_exclusive: float = 0.0
    dual_anchor_structural_minimum_collider_separation_improvement: float = 0.05
    dual_anchor_state_completion_minimum_raw_generative_score: float = 0.024
    dual_anchor_state_completion_minimum_joint_gain: float = 0.010
    dual_anchor_state_completion_minimum_fold_joint_gain: float = 0.0015
    dual_anchor_state_completion_minimum_parent_posterior: float = 0.75
    dual_anchor_state_completion_minimum_fold_parent_posterior: float = 0.55
    dual_anchor_state_completion_minimum_both_positive_fold_support: float = 1.0
    dual_anchor_state_completion_minimum_both_practical_fold_support: float = 0.80
    dual_anchor_state_completion_maximum_arm_empirical_positive_p: float = 0.25
    dual_anchor_state_completion_minimum_collider_separation_improvement: float = 0.03

    neighbor_star_maximum_production_outer_wald_support_exclusive: float = 1.0
    neighbor_star_maximum_production_raw_generative_score_exclusive: float = 0.050
    neighbor_star_maximum_production_joint_gain: float = 0.010
    neighbor_star_maximum_production_drop_one_gain: float = 0.002
    neighbor_star_minimum_raw_generative_score: float = 0.020
    neighbor_star_minimum_raw_score_fraction_of_production: float = 0.50
    neighbor_star_minimum_joint_gain: float = 0.004
    neighbor_star_minimum_drop_one_gain: float = -0.0005
    neighbor_star_minimum_fold_joint_gain: float = -0.005
    neighbor_star_minimum_parent_posterior: float = 0.90
    neighbor_star_minimum_fold_parent_posterior: float = 0.75
    neighbor_star_minimum_both_positive_fold_support: float = 1.0
    neighbor_star_minimum_both_practical_fold_support: float = 1.0
    neighbor_star_maximum_arm_empirical_positive_p: float = 13.0 / 32.0
    neighbor_star_minimum_outer_wald_view_support: float = 1.0
    neighbor_star_minimum_reverse_center_parent_posterior_margin: float = 0.35

    state_parent_rotation_maximum_production_outer_wald_support_exclusive: float = 1.0
    state_parent_rotation_minimum_raw_generative_score: float = 0.024
    state_parent_rotation_minimum_raw_score_fraction_of_production: float = 0.75
    state_parent_rotation_minimum_joint_gain: float = 0.005
    state_parent_rotation_minimum_drop_one_gain: float = 0.0005
    state_parent_rotation_minimum_fold_joint_gain: float = -0.005
    state_parent_rotation_minimum_parent_posterior: float = 0.93
    state_parent_rotation_minimum_fold_parent_posterior: float = 0.80
    state_parent_rotation_minimum_both_positive_fold_support: float = 1.0
    state_parent_rotation_minimum_both_practical_fold_support: float = 1.0
    state_parent_rotation_maximum_arm_empirical_positive_p: float = 11.0 / 32.0
    state_parent_rotation_minimum_reverse_center_joint_gain_margin: float = 0.0015
    state_parent_rotation_minimum_reverse_center_parent_posterior_margin: float = 0.25
    state_parent_rotation_topology_maximum_production_raw_generative_score_exclusive: float = 0.050
    state_parent_rotation_topology_minimum_raw_generative_score: float = 0.025
    state_parent_rotation_topology_minimum_joint_gain: float = 0.005
    state_parent_rotation_topology_minimum_drop_one_gain: float = -0.0025
    state_parent_rotation_topology_minimum_fold_joint_gain: float = 0.001
    state_parent_rotation_topology_minimum_parent_posterior: float = 0.45
    state_parent_rotation_topology_minimum_fold_parent_posterior: float = 0.15
    state_parent_rotation_topology_minimum_both_positive_fold_support: float = 0.40
    state_parent_rotation_topology_minimum_both_practical_fold_support: float = 0.20
    state_parent_rotation_topology_maximum_arm_empirical_positive_p: float = 0.0625
    state_parent_rotation_topology_minimum_outer_empirical_null_compatibility_p: float = 0.90
    state_parent_rotation_topology_minimum_outer_wald_view_support: float = 1.0
    state_parent_rotation_topology_minimum_collider_separation: float = -0.07
    state_parent_rotation_topology_minimum_median_collider_separation: float = 0.14
    state_parent_rotation_topology_minimum_collider_separation_improvement: float = 0.06
    state_parent_rotation_topology_minimum_median_collider_separation_improvement: float = 0.17
    state_parent_rotation_topology_minimum_median_collider_separation_runner_margin: float = 0.20

    state_center_sibling_rotation_minimum_loading_cosine: float = 0.95
    state_center_sibling_rotation_maximum_loading_empirical_p: float = 0.125
    state_center_sibling_rotation_minimum_log_odds_gain: float = 0.25
    state_center_sibling_rotation_minimum_empirical_p_improvement: float = 0.25
    state_center_sibling_rotation_minimum_raw_generative_score: float = 0.010
    state_center_sibling_rotation_minimum_joint_gain: float = 0.010
    state_center_sibling_rotation_minimum_drop_one_gain: float = -0.0015
    state_center_sibling_rotation_minimum_fold_joint_gain: float = 0.005
    state_center_sibling_rotation_minimum_parent_posterior: float = 0.70
    state_center_sibling_rotation_minimum_fold_parent_posterior: float = 0.45
    state_center_sibling_rotation_minimum_both_positive_fold_support: float = 0.80
    state_center_sibling_rotation_minimum_both_practical_fold_support: float = 0.60
    state_center_sibling_rotation_maximum_arm_empirical_positive_p: float = 0.50
    state_center_sibling_rotation_minimum_outer_empirical_null_compatibility_p: float = 0.09
    state_center_sibling_rotation_minimum_outer_wald_view_support: float = 1.0
    state_center_sibling_rotation_minimum_reverse_center_parent_posterior_margin: float = 0.50
    state_center_sibling_rotation_minimum_median_collider_separation: float = 0.040
    state_center_sibling_rotation_minimum_median_collider_separation_improvement: float = 0.15

    same_center_substitution_maximum_production_raw_generative_score_exclusive: float = 0.050
    same_center_substitution_minimum_raw_generative_score: float = 0.005
    same_center_substitution_minimum_joint_gain: float = 0.005
    same_center_substitution_minimum_drop_one_gain: float = -0.0015
    same_center_substitution_minimum_fold_joint_gain: float = -0.002
    same_center_substitution_minimum_parent_posterior: float = 0.50
    same_center_substitution_minimum_fold_parent_posterior: float = 0.30
    same_center_substitution_minimum_both_positive_fold_support: float = 0.60
    same_center_substitution_minimum_both_practical_fold_support: float = 0.20
    same_center_substitution_maximum_arm_empirical_positive_p: float = 17.0 / 32.0
    same_center_substitution_minimum_outer_empirical_null_compatibility_p: float = 0.50
    same_center_substitution_minimum_outer_wald_view_support: float = 1.0
    same_center_substitution_minimum_collider_separation: float = -0.05
    same_center_substitution_minimum_median_collider_separation: float = 0.08
    same_center_substitution_minimum_collider_separation_improvement: float = 0.10
    same_center_substitution_minimum_collider_separation_runner_margin: float = 0.10

    tier2_minimum_parent_positive_posterior: float = 0.90
    tier2_minimum_joint_gain: float = 0.009
    tier2_minimum_drop_one_gain: float = 0.00025
    tier2_minimum_both_positive_fold_support: float = 1.0
    tier2_minimum_reverse_center_gain_margin: float = 0.0001
    tier2_maximum_arm_empirical_positive_p: float = 0.125
    tier2_minimum_outer_empirical_null_compatibility_p: float = 0.125
    tier2_minimum_collider_separation: float = 0.040
    tier2_minimum_score: float = 0.040
    tier2_minimum_all_candidate_score_margin: float = 0.010

    tier3_minimum_parent_positive_posterior: float = 0.90
    tier3_minimum_joint_gain: float = 0.005
    tier3_minimum_drop_one_gain: float = -0.0001
    tier3_minimum_both_positive_fold_support: float = 0.80
    tier3_maximum_arm_empirical_positive_p: float = 0.125
    tier3_minimum_outer_empirical_null_compatibility_p: float = 0.125
    tier3_maximum_joint_gain_deficit_from_production: float = 0.0001
    tier3_maximum_score_deficit_from_production: float = 0.001
    tier3_production_arm_p_exclusive: float = 0.125

    multi_component_minimum_directed_edge_count: int = 4
    multi_component_minimum_current_arm_support: int = 2
    multi_component_minimum_raw_generative_score: float = 0.060
    multi_component_minimum_global_runner_score_margin: float = 0.020
    multi_component_minimum_joint_gain: float = 0.015
    multi_component_minimum_drop_one_gain: float = 0.001
    multi_component_minimum_parent_positive_posterior: float = 0.970
    multi_component_minimum_fold_parent_positive_posterior: float = 0.900
    multi_component_minimum_both_positive_fold_support: float = 1.0
    multi_component_minimum_both_practical_fold_support: float = 1.0
    multi_component_maximum_arm_empirical_positive_p: float = 1.0 / 32.0
    multi_component_minimum_outer_empirical_null_compatibility_p: float = 0.40

    zero_edge_minimum_outer_conditional_wald_view_support: float = 1.0
    zero_edge_minimum_selected_practical_fold_support: float = 1.0
    zero_edge_minimum_center_score_margin_exclusive: float = 0.0
    zero_edge_runner_minimum_parent_positive_posterior: float = 0.99
    zero_edge_runner_minimum_fold_parent_positive_posterior: float = 0.99
    zero_edge_runner_minimum_both_positive_fold_support: float = 1.0
    zero_edge_runner_minimum_both_practical_fold_support: float = 1.0
    zero_edge_runner_maximum_arm_empirical_positive_p: float = 1.0 / 32.0
    zero_edge_runner_minimum_joint_gain: float = 0.005
    zero_edge_runner_minimum_drop_one_gain: float = 0.003
    zero_edge_runner_minimum_raw_generative_score: float = 0.040
    zero_edge_runner_minimum_eligible_score_margin: float = 0.010
    zero_edge_all_state_topology_minimum_collider_separation: float = -0.020
    zero_edge_all_state_topology_minimum_median_collider_separation: float = 0.060
    zero_edge_all_state_topology_minimum_raw_generative_score: float = 0.0
    zero_edge_all_state_topology_minimum_joint_gain: float = -0.003
    zero_edge_all_state_topology_minimum_drop_one_gain: float = -0.002
    zero_edge_all_state_topology_minimum_fold_joint_gain: float = -0.006
    zero_edge_all_state_topology_minimum_parent_posterior: float = 0.60
    zero_edge_all_state_topology_minimum_fold_parent_posterior: float = 0.25
    zero_edge_all_state_topology_minimum_both_positive_fold_support: float = 0.80
    zero_edge_all_state_topology_minimum_both_practical_fold_support: float = 0.60
    zero_edge_all_state_topology_maximum_hierarchical_arm_empirical_p: float = (
        13.0 / 32.0
    )
    zero_edge_all_state_topology_minimum_outer_empirical_null_compatibility_p: float = (
        0.25
    )
    zero_edge_all_state_topology_minimum_outer_wald_view_support: float = 1.0
    zero_edge_all_state_topology_minimum_beta_arm_log_odds: float = 0.90
    zero_edge_all_state_topology_maximum_beta_arm_empirical_p: float = 0.425
    zero_edge_rescue_completion_minimum_parent_posterior: float = 0.90
    zero_edge_rescue_completion_minimum_fold_parent_posterior: float = 0.90
    zero_edge_rescue_completion_minimum_both_positive_fold_support: float = 1.0
    zero_edge_rescue_completion_minimum_both_practical_fold_support: float = 1.0
    zero_edge_rescue_completion_maximum_arm_empirical_positive_p: float = 5.0 / 32.0
    zero_edge_rescue_completion_minimum_outer_empirical_null_compatibility_p: float = 0.25
    zero_edge_rescue_completion_minimum_outer_wald_view_support: float = 1.0
    zero_edge_rescue_completion_minimum_joint_gain: float = 0.004
    zero_edge_rescue_completion_minimum_drop_one_gain_exclusive: float = 0.0
    zero_edge_rescue_completion_minimum_raw_generative_score: float = 0.020
    zero_edge_rescue_orientation_minimum_reverse_center_gain_margin: float = 0.0005
    zero_edge_rescue_orientation_minimum_raw_generative_score: float = 0.010
    zero_edge_rescue_orientation_minimum_joint_gain: float = 0.003
    zero_edge_rescue_orientation_minimum_parent_posterior: float = 0.60
    zero_edge_rescue_orientation_minimum_fold_parent_posterior: float = 0.40
    zero_edge_rescue_orientation_minimum_both_positive_fold_support: float = 0.60
    zero_edge_rescue_orientation_minimum_both_practical_fold_support: float = 0.60
    zero_edge_rescue_orientation_maximum_arm_empirical_positive_p: float = 5.0 / 32.0
    zero_edge_rescue_orientation_minimum_outer_empirical_null_compatibility_p: float = 0.09
    zero_edge_rescue_orientation_minimum_outer_wald_view_support: float = 1.0 / 3.0
    zero_edge_rescue_orientation_minimum_same_center_runner_margin: float = 0.0015
    zero_edge_dominant_rescue_required_count: int = 2
    zero_edge_dominant_rescue_minimum_leading_log_odds: float = 2.0
    zero_edge_dominant_rescue_minimum_log_odds_margin: float = 0.50
    zero_edge_dominant_rescue_maximum_leading_empirical_p: float = 0.01
    zero_edge_dominant_rescue_minimum_direction_probability: float = 0.15
    zero_edge_dominant_rescue_minimum_direction_probability_margin: float = 0.05
    zero_edge_dominant_rescue_minimum_raw_generative_score: float = 0.018
    zero_edge_dominant_rescue_minimum_same_center_runner_margin: float = 0.003
    zero_edge_dominant_rescue_minimum_joint_gain: float = 0.004
    zero_edge_dominant_rescue_minimum_parent_posterior: float = 0.70
    zero_edge_dominant_rescue_minimum_fold_parent_posterior: float = 0.30
    zero_edge_dominant_rescue_minimum_both_positive_fold_support: float = 0.80
    zero_edge_dominant_rescue_minimum_both_practical_fold_support: float = 0.60
    zero_edge_dominant_rescue_maximum_arm_empirical_positive_p: float = 10.0 / 32.0
    zero_edge_dominant_rescue_minimum_outer_empirical_null_compatibility_p: float = 0.50
    zero_edge_spoke_consensus_required_unresolved_spokes: int = 3
    zero_edge_spoke_consensus_required_base_spokes: int = 2
    zero_edge_spoke_consensus_minimum_second_arm_log_odds: float = 1.0
    zero_edge_spoke_consensus_minimum_second_vs_third_log_odds_margin: float = 0.10
    zero_edge_spoke_consensus_maximum_second_arm_empirical_positive_p: float = 0.01
    zero_edge_spoke_consensus_minimum_parent_posterior: float = 0.99
    zero_edge_spoke_consensus_minimum_fold_parent_posterior: float = 0.98
    zero_edge_spoke_consensus_minimum_both_positive_fold_support: float = 1.0
    zero_edge_spoke_consensus_minimum_both_practical_fold_support: float = 1.0
    zero_edge_spoke_consensus_maximum_arm_empirical_positive_p: float = 1.0 / 32.0
    zero_edge_spoke_consensus_minimum_outer_wald_view_support: float = 1.0
    zero_edge_spoke_consensus_minimum_joint_gain: float = 0.008
    zero_edge_spoke_consensus_minimum_drop_one_gain: float = 0.002
    zero_edge_spoke_consensus_minimum_raw_generative_score: float = 0.040

    retained_orientation_demotion_minimum_parent_posterior: float = 0.90
    retained_orientation_demotion_maximum_arm_empirical_positive_p: float = 0.125
    retained_orientation_demotion_maximum_reverse_center_posterior_margin: float = 0.10
    ambiguous_skeleton_minimum_parent_posterior: float = 0.75
    ambiguous_skeleton_minimum_both_positive_fold_support: float = 0.80
    ambiguous_skeleton_minimum_both_practical_fold_support: float = 0.80
    ambiguous_skeleton_minimum_outer_wald_view_support: float = 1.0
    ambiguous_skeleton_minimum_joint_gain: float = 0.005
    ambiguous_skeleton_minimum_raw_generative_score: float = 0.010
    ambiguous_skeleton_minimum_collider_separation_improvement: float = 0.10
    ambiguous_skeleton_minimum_collider_separation_runner_margin: float = 0.10

    def validate(self) -> None:
        for name in (
            "n_folds",
            "null_permutations",
            "required_assays",
            "required_features_per_assay",
            "required_evidence_views",
            "minimum_base_view_count",
            "multi_component_minimum_directed_edge_count",
            "multi_component_minimum_current_arm_support",
            "multicenter_alternative_minimum_current_skeleton_arm_support",
            "zero_edge_spoke_consensus_required_unresolved_spokes",
            "zero_edge_spoke_consensus_required_base_spokes",
            "zero_edge_dominant_rescue_required_count",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if not self.microstate_resolutions or any(
            int(value) < 2 for value in self.microstate_resolutions
        ):
            raise ValueError("microstate_resolutions must contain values >= 2")
        if tuple(sorted(set(self.microstate_resolutions))) != self.microstate_resolutions:
            raise ValueError("microstate_resolutions must be sorted and unique")
        for name in ("parent_prior_sd", "nuisance_prior_sd"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            is_probability = (
                "posterior" in name
                or "support" in name
                or "empirical" in name
                or name.endswith("arm_p_exclusive")
            )
            if is_probability:
                if isinstance(value, float) and (
                    not np.isfinite(value) or not 0.0 <= value <= 1.0
                ):
                    raise ValueError(f"{name} must lie in [0, 1]")


@dataclass
class CrossfitHierarchicalColliderResult:
    """Complete deterministic audit trail and revised direct layer."""

    candidates: pd.DataFrame
    decisions: pd.DataFrame
    direct_candidates: pd.DataFrame
    demoted_or_unresolved: pd.DataFrame
    provenance: dict[str, Any]


@dataclass
class CrossfitColliderWorldEvidence:
    """Expensive assay-cell evidence fitted exactly once per world."""

    generative_candidates: pd.DataFrame
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _MapFit:
    coefficients: FloatArray
    covariance: FloatArray


def _assert_observed_only(frame: pd.DataFrame, source: str) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"{source} contains duplicate column names")
    forbidden = sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN_INPUT_TOKENS)
    )
    if forbidden:
        raise ValueError(f"{source} contains privileged columns: {forbidden}")


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def _require_strict_boolean_column(
    frame: pd.DataFrame,
    column: str,
    source: str,
) -> None:
    """Reject missing, numeric, or string stand-ins before Boolean gating."""

    _require_columns(frame, {column}, source)
    valid = frame[column].map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(valid.all()):
        raise ValueError(
            f"{source}.{column} must contain only Boolean values; "
            "missing, numeric, and string stand-ins are not allowed"
        )


def _stable_seed(*values: object) -> int:
    payload = "||".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _normalise_hash_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isposinf(value):
            return "+Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return format(float(value), ".17g")
    return str(value)


def deterministic_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a frame independently of its row and column order."""

    columns = sorted(map(str, frame.columns))
    records = []
    for record in frame[columns].to_dict(orient="records"):
        records.append(
            json.dumps(
                {key: _normalise_hash_value(record[key]) for key in columns},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    payload = "\n".join(sorted(records)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_correlation(left: FloatArray, right: FloatArray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _log_likelihood(y: FloatArray, probability: FloatArray) -> FloatArray:
    p = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return xlogy(y, p) + xlogy(1.0 - y, 1.0 - p)


def _fit_map_logistic(
    design: FloatArray,
    y: FloatArray,
    prior_precision: FloatArray,
) -> _MapFit:
    x = np.column_stack([np.ones(len(design)), design]).astype(np.float64)
    beta = np.zeros(x.shape[1], dtype=np.float64)
    precision = np.r_[1e-8, np.asarray(prior_precision, dtype=np.float64)]

    def objective(value: FloatArray) -> float:
        probability = expit(x @ value)
        return float(
            _log_likelihood(y, probability).sum()
            - 0.5 * np.dot(precision * value, value)
        )

    current = objective(beta)
    for _ in range(40):
        probability = expit(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = x.T @ (y - probability) - precision * beta
        information = x.T @ (weight[:, None] * x) + np.diag(precision)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        if float(np.max(np.abs(step))) < 1e-7:
            break
        scale = 1.0
        accepted = False
        while scale >= 1.0 / 1024.0:
            proposal = beta + scale * step
            proposed = objective(proposal)
            if proposed >= current - 1e-10:
                beta = proposal
                current = proposed
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    probability = expit(x @ beta)
    weight = np.maximum(probability * (1.0 - probability), 1e-7)
    information = x.T @ (weight[:, None] * x) + np.diag(precision)
    return _MapFit(beta, np.linalg.pinv(information))


def _nuisance_design(frame: pd.DataFrame) -> FloatArray:
    state_columns = sorted(
        column for column in frame.columns if str(column).startswith("integration__dim")
    )
    if not state_columns:
        raise ValueError("assay_local_embedding has no integration__dim columns")
    state = frame[state_columns].to_numpy(dtype=np.float64)
    depth = np.log1p(
        frame["observed_background_library_size"].to_numpy(dtype=np.float64)
    )[:, None]
    pieces = [state, depth, state * state, depth * depth, state * depth]
    if "integration_uncertainty" in frame:
        pieces.append(frame[["integration_uncertainty"]].to_numpy(dtype=np.float64))
    design = np.column_stack(pieces)
    if not np.isfinite(design).all():
        raise ValueError("Nuisance design contains non-finite values")
    return design


def _scale_train_test(train: FloatArray, test: FloatArray) -> tuple[FloatArray, FloatArray]:
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def _invalid_center_rows(
    feature_names: tuple[str, ...], center_index: int
) -> list[dict[str, Any]]:
    parents = [index for index in range(len(feature_names)) if index != center_index]
    return [
        {
            "center": feature_names[center_index],
            "outer1": min(feature_names[first], feature_names[second]),
            "outer2": max(feature_names[first], feature_names[second]),
            "crossfit_status": "insufficient_detection_for_crossfit",
            "crossfit_joint_gain_over_baseline": math.nan,
            "crossfit_drop_parent1_gain": math.nan,
            "crossfit_drop_parent2_gain": math.nan,
            "crossfit_min_drop_one_gain": math.nan,
            "crossfit_joint_gain_fold_min": math.nan,
            "laplace_mean_parent_coefficient_min": math.nan,
            "laplace_fold_parent_coefficient_min": math.nan,
            "laplace_min_parent_positive_posterior": math.nan,
            "laplace_fold_min_parent_positive_posterior": math.nan,
            "laplace_both_positive_fold_support": math.nan,
            "laplace_both_practical_fold_support": math.nan,
        }
        for first, second in itertools.combinations(parents, 2)
    ]


def _crossfit_response(
    detections: FloatArray,
    nuisance: FloatArray,
    center_index: int,
    feature_names: tuple[str, ...],
    assay: str,
    randomization_key: str,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    y = detections[:, center_index]
    minimum_class = int(min(y.sum(), len(y) - y.sum()))
    if minimum_class < 2:
        mean = float((y.sum() + 0.5) / (len(y) + 1.0))
        return np.repeat(mean, len(y)), _invalid_center_rows(feature_names, center_index)
    n_folds = min(config.n_folds, minimum_class)
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=_stable_seed(
            "fold", randomization_key, assay, feature_names[center_index]
        ),
    )
    splits = list(splitter.split(nuisance, y))
    parents = [index for index in range(len(feature_names)) if index != center_index]
    pairs = list(itertools.combinations(parents, 2))
    baseline_ll = np.zeros(len(y), dtype=np.float64)
    baseline_probability = np.zeros(len(y), dtype=np.float64)
    single_ll = {parent: np.zeros(len(y), dtype=np.float64) for parent in parents}
    joint_ll = {pair: np.zeros(len(y), dtype=np.float64) for pair in pairs}
    fold_metrics: dict[tuple[int, int], list[dict[str, float]]] = {
        pair: [] for pair in pairs
    }
    nuisance_precision = np.repeat(
        1.0 / config.nuisance_prior_sd**2, nuisance.shape[1]
    )
    parent_precision = 1.0 / config.parent_prior_sd**2
    for fold, (train, test) in enumerate(splits):
        train_nuisance, test_nuisance = _scale_train_test(
            nuisance[train], nuisance[test]
        )
        baseline = _fit_map_logistic(train_nuisance, y[train], nuisance_precision)
        probability0 = expit(
            np.column_stack([np.ones(len(test)), test_nuisance])
            @ baseline.coefficients
        )
        baseline_probability[test] = probability0
        baseline_ll[test] = _log_likelihood(y[test], probability0)
        for parent in parents:
            train_design = np.column_stack([train_nuisance, detections[train, parent]])
            test_design = np.column_stack([test_nuisance, detections[test, parent]])
            fit = _fit_map_logistic(
                train_design,
                y[train],
                np.r_[nuisance_precision, parent_precision],
            )
            probability = expit(
                np.column_stack([np.ones(len(test)), test_design]) @ fit.coefficients
            )
            single_ll[parent][test] = _log_likelihood(y[test], probability)
        for parent1, parent2 in pairs:
            train_design = np.column_stack(
                [train_nuisance, detections[train, parent1], detections[train, parent2]]
            )
            test_design = np.column_stack(
                [test_nuisance, detections[test, parent1], detections[test, parent2]]
            )
            fit = _fit_map_logistic(
                train_design,
                y[train],
                np.r_[nuisance_precision, parent_precision, parent_precision],
            )
            probability = expit(
                np.column_stack([np.ones(len(test)), test_design]) @ fit.coefficients
            )
            joint_ll[(parent1, parent2)][test] = _log_likelihood(y[test], probability)
            coefficients = fit.coefficients[-2:]
            standard_errors = np.sqrt(
                np.maximum(np.diag(fit.covariance)[-2:], 1e-12)
            )
            posterior = norm.cdf(coefficients / standard_errors)
            fold_metrics[(parent1, parent2)].append(
                {
                    "fold": float(fold),
                    "coefficient1": float(coefficients[0]),
                    "coefficient2": float(coefficients[1]),
                    "posterior1": float(posterior[0]),
                    "posterior2": float(posterior[1]),
                }
            )

    rows: list[dict[str, Any]] = []
    for parent1, parent2 in pairs:
        metrics = fold_metrics[(parent1, parent2)]
        coefficients = np.asarray(
            [[row["coefficient1"], row["coefficient2"]] for row in metrics]
        )
        posterior = np.asarray(
            [[row["posterior1"], row["posterior2"]] for row in metrics]
        )
        joint = joint_ll[(parent1, parent2)]
        drop1 = joint - single_ll[parent2]
        drop2 = joint - single_ll[parent1]
        rows.append(
            {
                "center": feature_names[center_index],
                "outer1": min(feature_names[parent1], feature_names[parent2]),
                "outer2": max(feature_names[parent1], feature_names[parent2]),
                "crossfit_status": "fitted",
                "crossfit_joint_gain_over_baseline": float(np.mean(joint - baseline_ll)),
                "crossfit_drop_parent1_gain": float(np.mean(drop1)),
                "crossfit_drop_parent2_gain": float(np.mean(drop2)),
                "crossfit_min_drop_one_gain": float(min(np.mean(drop1), np.mean(drop2))),
                "crossfit_joint_gain_fold_min": float(
                    min(np.mean((joint - baseline_ll)[test]) for _, test in splits)
                ),
                "laplace_mean_parent_coefficient_min": float(
                    coefficients.mean(axis=0).min()
                ),
                "laplace_fold_parent_coefficient_min": float(coefficients.min()),
                "laplace_min_parent_positive_posterior": float(
                    posterior.mean(axis=0).min()
                ),
                "laplace_fold_min_parent_positive_posterior": float(posterior.min()),
                "laplace_both_positive_fold_support": float(
                    np.mean(np.all(coefficients > 0.0, axis=1))
                ),
                "laplace_both_practical_fold_support": float(
                    np.mean(np.all(coefficients >= 0.30, axis=1))
                ),
            }
        )
    return baseline_probability, rows


def _empirical_null_pair_metrics(
    residuals: FloatArray,
    nuisance: FloatArray,
    feature_names: tuple[str, ...],
    assay: str,
    randomization_key: str,
    config: CrossfitHierarchicalColliderConfig,
) -> dict[tuple[str, str], dict[str, float]]:
    if len(residuals) < max(config.microstate_resolutions):
        raise ValueError(
            f"{assay} has fewer cells than the largest microstate resolution"
        )
    observed = {
        canonical_edge(feature_names[left], feature_names[right]): _finite_correlation(
            residuals[:, left], residuals[:, right]
        )
        for left, right in itertools.combinations(range(len(feature_names)), 2)
    }
    values: dict[tuple[str, str], list[tuple[float, float]]] = {
        pair: [] for pair in observed
    }
    scale = nuisance.std(axis=0)
    scaled = (nuisance - nuisance.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    feature_index = {feature: index for index, feature in enumerate(feature_names)}
    for resolution in config.microstate_resolutions:
        labels = KMeans(
            n_clusters=resolution,
            n_init=10,
            random_state=_stable_seed(
                "microstate", randomization_key, assay, resolution
            ),
        ).fit_predict(scaled)
        groups = [np.flatnonzero(labels == label) for label in range(resolution)]
        for pair, observed_value in observed.items():
            left, right = map(feature_index.__getitem__, pair)
            rng = np.random.default_rng(
                _stable_seed(
                    "null",
                    randomization_key,
                    assay,
                    resolution,
                    pair[0],
                    pair[1],
                )
            )
            null = np.zeros(config.null_permutations, dtype=np.float64)
            for permutation in range(config.null_permutations):
                order = np.arange(len(residuals))
                for group in groups:
                    order[group] = rng.permutation(group)
                null[permutation] = _finite_correlation(
                    residuals[:, left], residuals[order, right]
                )
            values[pair].append(
                (
                    float((1 + np.sum(null >= observed_value)) / (len(null) + 1)),
                    float(
                        (1 + np.sum(np.abs(null) >= abs(observed_value)))
                        / (len(null) + 1)
                    ),
                )
            )
    return {
        pair: {
            "residual_correlation": observed[pair],
            "empirical_positive_p_worst_resolution": max(item[0] for item in pair_values),
            "empirical_absolute_p_worst_resolution": min(item[1] for item in pair_values),
        }
        for pair, pair_values in values.items()
    }


def _validate_assay_cell_inputs(
    assay_cell_tables: Mapping[str, pd.DataFrame],
    assay_local_embedding: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig,
) -> dict[str, tuple[str, ...]]:
    if len(assay_cell_tables) != config.required_assays:
        raise ValueError(
            f"Expected {config.required_assays} assay cell tables; "
            f"received {len(assay_cell_tables)}"
        )
    _assert_observed_only(assay_local_embedding, "assay_local_embedding")
    _require_columns(
        assay_local_embedding, {"cell", "modality"}, "assay_local_embedding"
    )
    if assay_local_embedding[["cell", "modality"]].duplicated().any():
        raise ValueError("assay_local_embedding contains duplicate cell/modality rows")
    state_columns = [
        column
        for column in assay_local_embedding
        if str(column).startswith("integration__dim")
    ]
    if not state_columns:
        raise ValueError("assay_local_embedding has no integration coordinates")
    modalities = set(assay_local_embedding["modality"].astype(str))
    if set(map(str, assay_cell_tables)) - modalities:
        raise ValueError("assay_local_embedding is missing an assay modality")

    metadata = {
        "cell",
        "modality",
        "observed_background_library_size",
        "observed_library_size",
    }
    features: dict[str, tuple[str, ...]] = {}
    for assay, raw in sorted(assay_cell_tables.items()):
        assay = str(assay)
        _assert_observed_only(raw, f"assay_cell_tables[{assay}]")
        _require_columns(
            raw,
            {"cell", "modality", "observed_background_library_size"},
            f"assay_cell_tables[{assay}]",
        )
        if raw[["cell", "modality"]].duplicated().any():
            raise ValueError(f"{assay} contains duplicate cell/modality rows")
        if set(raw["modality"].astype(str)) != {assay}:
            raise ValueError(f"{assay} table has inconsistent modality labels")
        focal = tuple(sorted(str(column) for column in raw if column not in metadata))
        if len(focal) != config.required_features_per_assay:
            raise ValueError(
                f"{assay} must contain exactly "
                f"{config.required_features_per_assay} focal features"
            )
        values = raw[list(focal)].apply(pd.to_numeric, errors="raise").to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{assay} focal counts must be finite and non-negative")
        features[assay] = focal
    if len({feature for focal in features.values() for feature in focal}) != sum(
        map(len, features.values())
    ):
        raise ValueError("Focal feature identifiers must be unique across assays")
    return features


def _fit_assay_generative_candidates(
    assay: str,
    cells: pd.DataFrame,
    embedding: pd.DataFrame,
    feature_names: tuple[str, ...],
    randomization_key: str,
    config: CrossfitHierarchicalColliderConfig,
) -> pd.DataFrame:
    embedding_columns = [
        "cell",
        "modality",
        *sorted(
            column
            for column in embedding
            if str(column).startswith("integration__dim")
        ),
    ]
    if "integration_uncertainty" in embedding:
        embedding_columns.append("integration_uncertainty")
    assay_embedding = embedding.loc[
        embedding["modality"].astype(str).eq(assay), embedding_columns
    ].copy()
    merged = cells.merge(
        assay_embedding, on=["cell", "modality"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(cells) or len(assay_embedding) != len(cells):
        raise ValueError(f"{assay} cell table and assay-local embedding do not align")
    merged = merged.sort_values("cell", kind="mergesort").reset_index(drop=True)
    detections = (
        merged[list(feature_names)].to_numpy(dtype=np.float64) > 0.0
    ).astype(np.float64)
    nuisance = _nuisance_design(merged)
    baseline_probability = np.zeros_like(detections)
    likelihood_rows: list[dict[str, Any]] = []
    for center_index in range(len(feature_names)):
        probability, rows = _crossfit_response(
            detections,
            nuisance,
            center_index,
            feature_names,
            assay,
            randomization_key,
            config,
        )
        baseline_probability[:, center_index] = probability
        likelihood_rows.extend(rows)
    denominator = np.sqrt(
        np.maximum(baseline_probability * (1.0 - baseline_probability), 0.01)
    )
    residuals = (detections - baseline_probability) / denominator
    pair_metrics = _empirical_null_pair_metrics(
        residuals, nuisance, feature_names, assay, randomization_key, config
    )
    global_folds = np.asarray(
        [
            _stable_seed("stability-fold", randomization_key, assay, cell)
            % config.n_folds
            for cell in merged["cell"]
        ],
        dtype=np.int64,
    )
    index = {feature: position for position, feature in enumerate(feature_names)}
    rows: list[dict[str, Any]] = []
    for likelihood in likelihood_rows:
        center = str(likelihood["center"])
        outer1 = str(likelihood["outer1"])
        outer2 = str(likelihood["outer2"])
        arm1 = pair_metrics[canonical_edge(center, outer1)]
        arm2 = pair_metrics[canonical_edge(center, outer2)]
        outer = pair_metrics[canonical_edge(outer1, outer2)]
        fold_arm_floor: list[float] = []
        fold_outer_absolute: list[float] = []
        for fold in range(config.n_folds):
            selected = global_folds == fold
            if selected.sum() < 3:
                continue
            corr1 = _finite_correlation(
                residuals[selected, index[center]], residuals[selected, index[outer1]]
            )
            corr2 = _finite_correlation(
                residuals[selected, index[center]], residuals[selected, index[outer2]]
            )
            corr_outer = _finite_correlation(
                residuals[selected, index[outer1]], residuals[selected, index[outer2]]
            )
            fold_arm_floor.append(min(corr1, corr2))
            fold_outer_absolute.append(abs(corr_outer))
        rows.append(
            {
                "source_assay": assay,
                **likelihood,
                "center_detection_rate": float(detections[:, index[center]].mean()),
                "minimum_parent_detection_rate": float(
                    min(
                        detections[:, index[outer1]].mean(),
                        detections[:, index[outer2]].mean(),
                    )
                ),
                "maximum_parent_detection_rate": float(
                    max(
                        detections[:, index[outer1]].mean(),
                        detections[:, index[outer2]].mean(),
                    )
                ),
                "minimum_arm_residual_correlation": float(
                    min(arm1["residual_correlation"], arm2["residual_correlation"])
                ),
                "outer_absolute_residual_correlation": float(
                    abs(outer["residual_correlation"])
                ),
                "residual_star_separation": float(
                    min(arm1["residual_correlation"], arm2["residual_correlation"])
                    - abs(outer["residual_correlation"])
                ),
                "maximum_arm_empirical_positive_p": float(
                    max(
                        arm1["empirical_positive_p_worst_resolution"],
                        arm2["empirical_positive_p_worst_resolution"],
                    )
                ),
                "outer_empirical_null_compatibility_p": float(
                    outer["empirical_absolute_p_worst_resolution"]
                ),
                "fold_minimum_arm_residual_correlation": float(
                    min(fold_arm_floor) if fold_arm_floor else math.nan
                ),
                "fold_median_arm_residual_correlation": float(
                    np.median(fold_arm_floor) if fold_arm_floor else math.nan
                ),
                "fold_maximum_outer_absolute_correlation": float(
                    max(fold_outer_absolute) if fold_outer_absolute else math.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["trio_key"] = [
        "||".join(sorted((row.center, row.outer1, row.outer2)))
        for row in frame.itertuples(index=False)
    ]
    for metric in (
        "crossfit_joint_gain_over_baseline",
        "crossfit_min_drop_one_gain",
        "laplace_min_parent_positive_posterior",
        "residual_star_separation",
    ):
        competitor = pd.Series(math.nan, index=frame.index, dtype=float)
        for _, group in frame.groupby("trio_key", sort=False):
            for row_index in group.index:
                peers = group.loc[group.index != row_index, metric]
                competitor.loc[row_index] = float(peers.max())
        frame[f"{metric}_reverse_center_margin"] = frame[metric] - competitor
    return frame.drop(columns="trio_key").sort_values(
        ["source_assay", "center", "outer1", "outer2"]
    ).reset_index(drop=True)


def fit_crossfit_collider_world_evidence(
    *,
    assay_cell_tables: Mapping[str, pd.DataFrame],
    assay_local_embedding: pd.DataFrame,
    randomization_key: str | int,
    config: CrossfitHierarchicalColliderConfig | None = None,
) -> CrossfitColliderWorldEvidence:
    """Fit expensive Bernoulli and empirical-null evidence once per world."""

    cfg = config or CrossfitHierarchicalColliderConfig()
    cfg.validate()
    if isinstance(randomization_key, bool) or not isinstance(
        randomization_key, (str, int, np.integer)
    ):
        raise TypeError("randomization_key must be a nonempty string or integer")
    key = str(randomization_key)
    if not key:
        raise ValueError("randomization_key must be nonempty")
    feature_map = _validate_assay_cell_inputs(
        assay_cell_tables, assay_local_embedding, cfg
    )
    frames = [
        _fit_assay_generative_candidates(
            assay,
            assay_cell_tables[assay],
            assay_local_embedding,
            feature_map[assay],
            key,
            cfg,
        )
        for assay in sorted(feature_map)
    ]
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    expected = (
        cfg.required_assays
        * cfg.required_features_per_assay
        * math.comb(cfg.required_features_per_assay - 1, 2)
    )
    if len(candidates) != expected:
        raise RuntimeError("World evidence did not enumerate every directed two-star")
    keys = ["source_assay", "center", "outer1", "outer2"]
    if candidates.duplicated(keys).any():
        raise RuntimeError("World evidence contains duplicate candidate identities")
    provenance = {
        "schema_version": f"{SCHEMA_VERSION}.world_evidence",
        "method": "cross_fitted_sparse_bernoulli_plus_state_depth_empirical_null",
        "config": asdict(cfg),
        "randomization_key": key,
        "n_assays": len(feature_map),
        "n_candidates": len(candidates),
        "assay_cell_table_sha256": {
            assay: deterministic_frame_sha256(assay_cell_tables[assay])
            for assay in sorted(feature_map)
        },
        "assay_local_embedding_sha256": deterministic_frame_sha256(
            assay_local_embedding
        ),
        "generative_candidates_sha256": deterministic_frame_sha256(candidates),
        "fit_once_per_world": True,
        "background_only_assay_local_embedding": True,
        "cross_assay_coordinate_registration_required": False,
        "paired_cells_used": False,
        "scorer_information_used": False,
    }
    return CrossfitColliderWorldEvidence(candidates, provenance)


def _pair_map(frame: pd.DataFrame, column: str, source: str) -> dict[tuple[str, str], float]:
    _require_columns(frame, {"node1", "node2", column}, source)
    output: dict[tuple[str, str], float] = {}
    for row in frame[["node1", "node2", column]].itertuples(index=False):
        pair = canonical_edge(str(row.node1), str(row.node2))
        if pair in output:
            raise ValueError(f"{source} contains duplicate pair {pair}")
        value = float(getattr(row, column))
        if not np.isfinite(value):
            raise ValueError(f"{source}.{column} must be finite")
        output[pair] = value
    return output


def _validate_view_mappings(
    base_edge_views: Mapping[str, pd.DataFrame],
    residual_views: Mapping[str, pd.DataFrame],
    feature_model_views: Mapping[str, pd.DataFrame],
    loading_pair_views: Mapping[str, pd.DataFrame],
    config: CrossfitHierarchicalColliderConfig,
) -> None:
    mappings = {
        "base_edge_views": base_edge_views,
        "residual_views": residual_views,
        "feature_model_views": feature_model_views,
        "loading_pair_views": loading_pair_views,
    }
    expected: set[str] | None = None
    for name, mapping in mappings.items():
        if len(mapping) != config.required_evidence_views:
            raise ValueError(
                f"{name} must contain exactly {config.required_evidence_views} views"
            )
        keys = set(map(str, mapping))
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError("Observed evidence view identifiers do not match")
        for view_id, frame in mapping.items():
            _assert_observed_only(frame, f"{name}[{view_id}]")


def _base_support(base_edge_views: Mapping[str, pd.DataFrame]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for _, raw in sorted(base_edge_views.items()):
        _require_columns(raw, {"node1", "node2"}, "base_edge_views")
        frame = raw.copy()
        if "selected" in frame:
            _require_strict_boolean_column(frame, "selected", "base_edge_views")
            frame = frame.loc[frame["selected"].eq(True)]
        if "edge_type" in frame:
            frame = frame.loc[frame["edge_type"].astype(str).eq(DIRECT_EDGE_TYPE)]
        for row in frame[["node1", "node2"]].itertuples(index=False):
            pair = canonical_edge(str(row.node1), str(row.node2))
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _state_view_maps(
    feature_model_views: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for view_id, frame in sorted(feature_model_views.items()):
        _require_columns(
            frame,
            {"feature", "oof_partial_state_fit", "loading_direction_stability"},
            f"feature_model_views[{view_id}]",
        )
        if frame["feature"].astype(str).duplicated().any():
            raise ValueError(f"feature_model_views[{view_id}] has duplicate features")
        output[str(view_id)] = {
            str(row.feature): max(float(row.oof_partial_state_fit), 0.0)
            * max(float(row.loading_direction_stability), 0.0)
            for row in frame.itertuples(index=False)
        }
    return output


def _state_membership_map(state_membership: pd.DataFrame) -> dict[str, Any]:
    required = {
        "feature",
        "modality",
        "selection_status",
        "feature_global_maxT_p",
        "oof_partial_state_fit",
        "conditional_wald_view_support",
        "conditional_wald_above_null_95pct",
        "reference_conditional_wald",
        "feature_global_maxT_confirmation_gate",
        "direction_stability_diagnostic",
    }
    _require_columns(state_membership, required, "state_membership")
    for column in (
        "feature_global_maxT_confirmation_gate",
        "direction_stability_diagnostic",
    ):
        _require_strict_boolean_column(state_membership, column, "state_membership")
    if state_membership["feature"].astype(str).duplicated().any():
        raise ValueError("state_membership contains duplicate feature rows")
    return {
        str(row.feature): row
        for row in state_membership.itertuples(index=False)
    }


def _build_exhaustive_base_candidates(
    *,
    beta_vote_evidence: pd.DataFrame,
    base_edge_views: Mapping[str, pd.DataFrame],
    residual_views: Mapping[str, pd.DataFrame],
    feature_model_views: Mapping[str, pd.DataFrame],
    loading_pair_views: Mapping[str, pd.DataFrame],
    state_membership: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig,
) -> pd.DataFrame:
    _assert_observed_only(beta_vote_evidence, "beta_vote_evidence")
    _require_columns(
        beta_vote_evidence,
        {
            "source_assay",
            "node1",
            "node2",
            "vote",
            "selected",
            "empirical_p_positive",
            "pooled_log_odds",
        },
        "beta_vote_evidence",
    )
    _require_strict_boolean_column(
        beta_vote_evidence, "selected", "beta_vote_evidence"
    )
    if beta_vote_evidence.duplicated(
        ["source_assay", "node1", "node2", "vote"]
    ).any():
        raise ValueError("beta_vote_evidence contains a duplicate assay/pair/vote row")
    _validate_view_mappings(
        base_edge_views,
        residual_views,
        feature_model_views,
        loading_pair_views,
        config,
    )
    state = _state_membership_map(state_membership)
    state_views = _state_view_maps(feature_model_views)
    residual_maps = {
        str(view_id): _pair_map(frame, "residual_r", f"residual_views[{view_id}]")
        for view_id, frame in sorted(residual_views.items())
    }
    loading_p = {
        str(view_id): _pair_map(
            frame,
            "loading_empirical_null_p",
            f"loading_pair_views[{view_id}]",
        )
        for view_id, frame in sorted(loading_pair_views.items())
    }
    loading_cosine = {
        str(view_id): _pair_map(
            frame,
            "absolute_loading_cosine",
            f"loading_pair_views[{view_id}]",
        )
        for view_id, frame in sorted(loading_pair_views.items())
    }
    base_counts = _base_support(base_edge_views)
    current_counts: dict[tuple[str, str], int] = {}
    if not current_direct_candidates.empty:
        _require_columns(
            current_direct_candidates,
            {"source_assay", "node1", "node2", "endpoint1", "endpoint2"},
            "current_direct_candidates",
        )
        for record in current_direct_candidates.to_dict(orient="records"):
            if _is_circle_only_direct_record(record):
                continue
            pair = canonical_edge(str(record["node1"]), str(record["node2"]))
            current_counts[pair] = current_counts.get(pair, 0) + 1

    aggregated = aggregate_vote_evidence(beta_vote_evidence)
    evidence_by_assay: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for assay, frame in aggregated.groupby("source_assay", sort=True):
        evidence_by_assay[str(assay)] = {
            canonical_edge(str(row["node1"]), str(row["node2"])): row
            for row in frame.to_dict(orient="records")
        }
    rows: list[dict[str, Any]] = []
    for assay, evidence in sorted(evidence_by_assay.items()):
        nodes = sorted({node for pair in evidence for node in pair})
        if len(nodes) != config.required_features_per_assay:
            raise ValueError(f"{assay} beta evidence must cover exactly six features")
        for pair, group in beta_vote_evidence.loc[
            beta_vote_evidence["source_assay"].astype(str).eq(assay)
        ].groupby(["node1", "node2"], sort=True):
            if group["vote"].nunique() != config.required_evidence_views:
                raise ValueError(f"{assay} beta pair {pair} lacks six vote views")
        for trio in itertools.combinations(nodes, 3):
            for center in trio:
                outer1, outer2 = sorted(node for node in trio if node != center)
                arm1 = canonical_edge(center, outer1)
                arm2 = canonical_edge(center, outer2)
                outer = canonical_edge(outer1, outer2)
                if any(pair not in evidence for pair in (arm1, arm2, outer)):
                    raise ValueError(f"{assay} beta evidence is not pair-complete")
                separations: list[float] = []
                openings: list[float] = []
                state_floor: list[float] = []
                arm_loading_p: list[float] = []
                arm_loading_cosine: list[float] = []
                for view_id in sorted(residual_maps):
                    residual = residual_maps[view_id]
                    if any(pair not in residual for pair in (arm1, arm2, outer)):
                        raise ValueError(f"{view_id} residual evidence is not pair-complete")
                    conditional = conditional_outer_correlation(
                        residual[outer], residual[arm1], residual[arm2]
                    )
                    separations.append(-conditional)
                    openings.append(residual[outer] - conditional)
                    state_floor.append(
                        min(
                            state_views[view_id].get(outer1, 0.0),
                            state_views[view_id].get(outer2, 0.0),
                        )
                    )
                    for arm in (arm1, arm2):
                        if arm not in loading_p[view_id]:
                            raise ValueError(f"{view_id} loading evidence is not pair-complete")
                        arm_loading_p.append(loading_p[view_id][arm])
                        arm_loading_cosine.append(loading_cosine[view_id][arm])
                arm_evidence = (evidence[arm1], evidence[arm2])
                parent_state = [state.get(outer1), state.get(outer2)]
                if any(item is None for item in parent_state) or center not in state:
                    raise ValueError(f"state_membership is missing a focal feature in {assay}")
                partial = [float(item.oof_partial_state_fit) for item in parent_state]
                wald_support = [
                    float(item.conditional_wald_view_support) for item in parent_state
                ]
                rows.append(
                    {
                        "source_assay": assay,
                        "center": center,
                        "outer1": outer1,
                        "outer2": outer2,
                        "arm1": "|".join(arm1),
                        "arm2": "|".join(arm2),
                        "outer_pair": "|".join(outer),
                        "median_collider_separation": float(np.median(separations)),
                        "minimum_collider_separation": float(np.min(separations)),
                        "median_collider_opening": float(np.median(openings)),
                        "minimum_collider_opening": float(np.min(openings)),
                        "median_outer_state_support_floor": float(np.median(state_floor)),
                        "minimum_outer_state_support_floor": float(np.min(state_floor)),
                        "min_arm_vote_fraction": float(
                            min(item["selected_vote_fraction"] for item in arm_evidence)
                        ),
                        "min_arm_lor": float(
                            min(item["mean_pooled_log_odds"] for item in arm_evidence)
                        ),
                        "mean_arm_lor": float(
                            np.mean([item["mean_pooled_log_odds"] for item in arm_evidence])
                        ),
                        "max_arm_empirical_p": float(
                            max(item["mean_empirical_p_positive"] for item in arm_evidence)
                        ),
                        "outer_lor": float(evidence[outer]["mean_pooled_log_odds"]),
                        "base_arm_view_support_min": min(
                            base_counts.get(arm1, 0), base_counts.get(arm2, 0)
                        ),
                        "base_arm_view_support_sum": base_counts.get(arm1, 0)
                        + base_counts.get(arm2, 0),
                        "current_arm_support": current_counts.get(arm1, 0)
                        + current_counts.get(arm2, 0),
                        "max_loading_empirical_null_p": float(max(arm_loading_p)),
                        "minimum_absolute_loading_cosine": float(
                            min(arm_loading_cosine)
                        ),
                        "loading_p_le_0_125_support": int(
                            sum(value <= 0.125 for value in arm_loading_p)
                        ),
                        "n_loading_arm_views": len(arm_loading_p),
                        "outer_positive_partial_state_count": int(
                            sum(value > 0.0 for value in partial)
                        ),
                        "minimum_outer_partial_state_fit": float(min(partial)),
                        "minimum_outer_conditional_wald_view_support": float(
                            min(wald_support)
                        ),
                        "minimum_outer_conditional_wald_above_null_95pct": float(
                            min(
                                float(item.conditional_wald_above_null_95pct)
                                for item in parent_state
                            )
                        ),
                        "minimum_outer_reference_conditional_wald": float(
                            min(float(item.reference_conditional_wald) for item in parent_state)
                        ),
                        "outer_feature_maxT_confirmation_count": int(
                            sum(
                                bool(item.feature_global_maxT_confirmation_gate)
                                for item in parent_state
                            )
                        ),
                        "two_parent_positive_state_wald_consistency": bool(
                            min(partial) > 0.0 and min(wald_support) >= 1.0
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["source_assay", "center", "outer1", "outer2"]
    ).reset_index(drop=True)


def _annotate_state_center_sibling_rotation_evidence(
    *,
    candidates: pd.DataFrame,
    beta_vote_evidence: pd.DataFrame,
    loading_pair_views: Mapping[str, pd.DataFrame],
    current_direct_candidates: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig,
) -> pd.DataFrame:
    """Attach observed evidence for one local state-surrogate rotation.

    The candidate must retain one production adjacency while rotating its
    center onto a production parent.  Its added parent replaces the other
    production parent only when those two displaced/new nodes have stable,
    sign-invariant loading equivalence.  This exposes a narrowly defined
    shared-state-versus-direct-structure comparison without using feature
    identities or labels.
    """

    for source, frame in (
        ("candidates", candidates),
        ("beta_vote_evidence", beta_vote_evidence),
        ("current_direct_candidates", current_direct_candidates),
    ):
        _assert_observed_only(frame, source)
    if len(loading_pair_views) != config.required_evidence_views:
        raise ValueError(
            "loading_pair_views must contain every aggregation/seed view"
        )
    output = candidates.copy()
    defaults: dict[str, Any] = {
        "state_center_sibling_rotation_structural_candidate": False,
        "state_center_sibling_rotation_displaced_node": "",
        "state_center_sibling_rotation_new_node": "",
        "state_center_sibling_rotation_minimum_loading_cosine": math.nan,
        "state_center_sibling_rotation_maximum_loading_empirical_p": math.nan,
        "state_center_sibling_rotation_new_edge_log_odds": math.nan,
        "state_center_sibling_rotation_displaced_edge_log_odds": math.nan,
        "state_center_sibling_rotation_log_odds_gain": math.nan,
        "state_center_sibling_rotation_new_edge_empirical_p": math.nan,
        "state_center_sibling_rotation_displaced_edge_empirical_p": math.nan,
        "state_center_sibling_rotation_empirical_p_improvement": math.nan,
    }
    for column, value in defaults.items():
        output[column] = value
    if current_direct_candidates.empty:
        return output

    directed = _current_directed_edges(current_direct_candidates)
    aggregated = aggregate_vote_evidence(beta_vote_evidence)
    beta_map = {
        (
            str(row.source_assay),
            canonical_edge(str(row.node1), str(row.node2)),
        ): (
            float(row.mean_pooled_log_odds),
            float(row.mean_empirical_p_positive),
        )
        for row in aggregated.itertuples(index=False)
    }
    loading_cosine = {
        str(view_id): _pair_map(
            frame,
            "absolute_loading_cosine",
            f"loading_pair_views[{view_id}]",
        )
        for view_id, frame in sorted(loading_pair_views.items())
    }
    loading_p = {
        str(view_id): _pair_map(
            frame,
            "loading_empirical_null_p",
            f"loading_pair_views[{view_id}]",
        )
        for view_id, frame in sorted(loading_pair_views.items())
    }

    for assay, assay_directed in directed.groupby("source_assay", sort=True):
        assay = str(assay)
        if len(assay_directed) != 2 or assay_directed["child"].nunique() != 1:
            continue
        production_center = str(assay_directed.iloc[0]["child"])
        production_parents = set(assay_directed["parent"].astype(str))
        if len(production_parents) != 2:
            continue
        for index, row in output.loc[
            output["source_assay"].astype(str).eq(assay)
        ].iterrows():
            candidate_center = str(row["center"])
            candidate_parents = {str(row["outer1"]), str(row["outer2"])}
            if (
                candidate_center not in production_parents
                or production_center not in candidate_parents
            ):
                continue
            new_nodes = candidate_parents - {production_center}
            displaced_nodes = production_parents - {candidate_center}
            if len(new_nodes) != 1 or len(displaced_nodes) != 1:
                continue
            new_node = next(iter(new_nodes))
            displaced_node = next(iter(displaced_nodes))
            if new_node == displaced_node:
                continue
            equivalent_pair = canonical_edge(displaced_node, new_node)
            if any(
                equivalent_pair not in loading_cosine[view_id]
                or equivalent_pair not in loading_p[view_id]
                for view_id in sorted(loading_cosine)
            ):
                raise ValueError(
                    f"{assay} loading evidence lacks displaced/new pair "
                    f"{equivalent_pair}"
                )
            new_edge = canonical_edge(candidate_center, new_node)
            displaced_edge = canonical_edge(candidate_center, displaced_node)
            new_key = (assay, new_edge)
            displaced_key = (assay, displaced_edge)
            if new_key not in beta_map or displaced_key not in beta_map:
                raise ValueError(
                    f"{assay} beta evidence lacks a sibling-rotation edge"
                )
            minimum_cosine = min(
                loading_cosine[view_id][equivalent_pair]
                for view_id in sorted(loading_cosine)
            )
            maximum_loading_p = max(
                loading_p[view_id][equivalent_pair]
                for view_id in sorted(loading_p)
            )
            new_log_odds, new_empirical_p = beta_map[new_key]
            displaced_log_odds, displaced_empirical_p = beta_map[displaced_key]
            output.loc[index, list(defaults)] = [
                True,
                displaced_node,
                new_node,
                float(minimum_cosine),
                float(maximum_loading_p),
                float(new_log_odds),
                float(displaced_log_odds),
                float(new_log_odds - displaced_log_odds),
                float(new_empirical_p),
                float(displaced_empirical_p),
                float(displaced_empirical_p - new_empirical_p),
            ]
    return output


def _raw_score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["crossfit_joint_gain_over_baseline"]
        + frame["crossfit_min_drop_one_gain"]
        + 0.02 * frame["laplace_min_parent_positive_posterior"]
        + 0.25 * frame["residual_star_separation"]
    )


def _star_edges(row: Any) -> set[tuple[str, str]]:
    return {
        (str(row.outer1), str(row.center)),
        (str(row.outer2), str(row.center)),
    }


def _is_circle_only_direct_record(record: Mapping[str, Any]) -> bool:
    endpoint1 = str(record.get("endpoint1", "")).upper()
    endpoint2 = str(record.get("endpoint2", "")).upper()
    if endpoint1 == endpoint2 == "CIRCLE":
        center = str(record.get("collider_center", "")).strip().lower()
        if center not in {"", "nan", "none"}:
            raise ValueError(
                "A CIRCLE--CIRCLE direct adjacency cannot claim a collider center"
            )
        return True
    if "CIRCLE" in {endpoint1, endpoint2}:
        raise ValueError(
            "A direct candidate must be either TAIL--ARROW or CIRCLE--CIRCLE"
        )
    return False


def _current_directed_edges(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["source_assay", "parent", "child", "node1", "node2"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        frame,
        {"source_assay", "node1", "node2", "endpoint1", "endpoint2"},
        "current_direct_candidates",
    )
    rows: list[dict[str, str]] = []
    for record in frame.to_dict(orient="records"):
        if _is_circle_only_direct_record(record):
            continue
        left, right = str(record["node1"]), str(record["node2"])
        center = str(record.get("collider_center", ""))
        if center not in {left, right}:
            endpoint1 = str(record["endpoint1"])
            endpoint2 = str(record["endpoint2"])
            if endpoint1 == "ARROW" and endpoint2 == "TAIL":
                center = left
            elif endpoint2 == "ARROW" and endpoint1 == "TAIL":
                center = right
            else:
                raise ValueError(
                    "Current direct candidates must identify one oriented collider center"
                )
        parent = right if center == left else left
        rows.append(
            {
                "source_assay": str(record["source_assay"]),
                "parent": parent,
                "child": center,
                "node1": min(left, right),
                "node2": max(left, right),
            }
        )
    output = pd.DataFrame(rows)
    if output.duplicated(["source_assay", "parent", "child"]).any():
        raise ValueError("current_direct_candidates contains duplicate directed edges")
    return output


def _redundancy_filtered_state_members(state_membership: pd.DataFrame) -> set[str]:
    selected = state_membership.loc[
        state_membership["selection_status"].astype(str).eq("selected")
    ].copy()
    retained: set[str] = set()
    for _, assay in selected.groupby("modality", sort=True):
        stable = assay.loc[assay["direction_stability_diagnostic"].astype(bool)]
        if len(stable):
            retained.update(stable["feature"].astype(str))
            continue
        winner = assay.sort_values(
            ["feature_global_maxT_p", "reference_conditional_wald", "feature"],
            ascending=[True, False, True],
        ).iloc[0]
        retained.add(str(winner["feature"]))
    return retained


def _ordered_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["raw_generative_score", "center", "outer1", "outer2"],
        ascending=[False, True, True, True],
    )


def _star_identity(row: Any) -> dict[str, str]:
    return {
        "center": str(row.center),
        "outer1": str(row.outer1),
        "outer2": str(row.outer2),
    }


def _zero_edge_rank_fusion_audit(
    group: pd.DataFrame,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Score a complete six-feature assay without using runtime fitting."""

    model_hash = _zero_edge_rank_fusion_model_sha256()
    base: dict[str, Any] = {
        "zero_edge_rank_fusion_model_version": ZERO_EDGE_RANK_FUSION_MODEL_VERSION,
        "zero_edge_rank_fusion_model_sha256": model_hash,
        "zero_edge_rank_fusion_runtime_refit": False,
        "zero_edge_rank_fusion_status": "not_scored",
        "zero_edge_rank_fusion_abstention_reason": "",
        "zero_edge_rank_fusion_selected_center": "",
        "zero_edge_rank_fusion_selected_outer1": "",
        "zero_edge_rank_fusion_selected_outer2": "",
        "zero_edge_rank_fusion_arm1_logit": math.nan,
        "zero_edge_rank_fusion_arm2_logit": math.nan,
        "zero_edge_rank_fusion_arm1_probability": math.nan,
        "zero_edge_rank_fusion_arm2_probability": math.nan,
        "zero_edge_rank_fusion_selected_center_score": math.nan,
        "zero_edge_rank_fusion_runner_center_score": math.nan,
        "zero_edge_rank_fusion_center_score_margin": math.nan,
        "zero_edge_rank_fusion_selected_state_wald_consistency": False,
        "zero_edge_rank_fusion_selected_outer_wald_view_support": math.nan,
        "zero_edge_rank_fusion_selected_practical_fold_support": math.nan,
        "zero_edge_rank_fusion_practical_fold_support_gate": False,
        "zero_edge_rank_fusion_arm1_evidence_json": "",
        "zero_edge_rank_fusion_arm2_evidence_json": "",
        "zero_edge_rank_fusion_all_edge_scores_json": "",
        "zero_edge_rank_fusion_all_center_scores_json": "",
    }
    nodes = sorted(
        set(group["center"].astype(str))
        | set(group["outer1"].astype(str))
        | set(group["outer2"].astype(str))
    )
    required = ZERO_EDGE_RANK_FUSION_REQUIRED_FEATURES
    expected_stars = required * math.comb(required - 1, 2)
    if len(nodes) != required or len(group) != expected_stars:
        base["zero_edge_rank_fusion_status"] = "abstained"
        base["zero_edge_rank_fusion_abstention_reason"] = (
            "model_domain_requires_complete_six_feature_candidate_universe"
        )
        return base, None

    edge_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    center_values = group["center"].astype(str)
    outer1_values = group["outer1"].astype(str)
    outer2_values = group["outer2"].astype(str)
    expected_context_stars = required - 2
    for parent in nodes:
        for child in nodes:
            if parent == child:
                continue
            contexts = {
                "direct": group.loc[
                    center_values.eq(child)
                    & (outer1_values.eq(parent) | outer2_values.eq(parent))
                ],
                "reverse": group.loc[
                    center_values.eq(parent)
                    & (outer1_values.eq(child) | outer2_values.eq(child))
                ],
                "outer": group.loc[
                    (outer1_values.eq(parent) & outer2_values.eq(child))
                    | (outer1_values.eq(child) & outer2_values.eq(parent))
                ],
            }
            if any(
                len(context) != expected_context_stars
                for context in contexts.values()
            ):
                base["zero_edge_rank_fusion_status"] = "abstained"
                base["zero_edge_rank_fusion_abstention_reason"] = (
                    "incomplete_directional_context"
                )
                return base, None
            top_context = {
                name: _ordered_candidates(context).iloc[0]
                for name, context in contexts.items()
            }
            logit = ZERO_EDGE_RANK_FUSION_INTERCEPT
            feature_audit: list[dict[str, Any]] = []
            for (context_name, feature), median, weight in zip(
                ZERO_EDGE_RANK_FUSION_SCHEMA,
                ZERO_EDGE_RANK_FUSION_MEDIANS,
                ZERO_EDGE_RANK_FUSION_WEIGHTS,
                strict=True,
            ):
                top = top_context[context_name]
                observed = pd.to_numeric(
                    pd.Series([top[feature]]), errors="coerce"
                ).iloc[0]
                imputed = float(observed) if np.isfinite(observed) else float(median)
                logit += float(weight) * imputed
                feature_audit.append(
                    {
                        "model_feature": f"{context_name}__{feature}",
                        "observed_value": float(observed)
                        if np.isfinite(observed)
                        else None,
                        "imputed_value": imputed,
                        "weight": float(weight),
                        "top_context_star": _star_identity(top),
                    }
                )
            edge_evidence[(parent, child)] = {
                "parent": parent,
                "child": child,
                "logit": float(logit),
                "probability": float(expit(logit)),
                "feature_evidence": feature_audit,
            }

    center_scores: list[dict[str, Any]] = []
    for child in nodes:
        ranked_parents = sorted(
            (
                edge_evidence[(parent, child)]
                for parent in nodes
                if parent != child
            ),
            key=lambda record: (-float(record["probability"]), str(record["parent"])),
        )
        selected_parents = ranked_parents[:2]
        center_scores.append(
            {
                "center": child,
                "outer1": min(
                    str(selected_parents[0]["parent"]),
                    str(selected_parents[1]["parent"]),
                ),
                "outer2": max(
                    str(selected_parents[0]["parent"]),
                    str(selected_parents[1]["parent"]),
                ),
                "score": float(
                    selected_parents[0]["probability"]
                    + selected_parents[1]["probability"]
                ),
            }
        )
    center_scores.sort(
        key=lambda record: (
            -float(record["score"]),
            str(record["center"]),
            str(record["outer1"]),
            str(record["outer2"]),
        )
    )
    winner, runner = center_scores[:2]
    matches = group.loc[
        center_values.eq(str(winner["center"]))
        & outer1_values.eq(str(winner["outer1"]))
        & outer2_values.eq(str(winner["outer2"]))
    ]
    if len(matches) != 1:
        base["zero_edge_rank_fusion_status"] = "abstained"
        base["zero_edge_rank_fusion_abstention_reason"] = (
            "selected_star_identity_not_unique"
        )
        return base, None
    selected = matches.iloc[0]
    arm1 = edge_evidence[(str(winner["outer1"]), str(winner["center"]))]
    arm2 = edge_evidence[(str(winner["outer2"]), str(winner["center"]))]
    selected_state_wald = selected["two_parent_positive_state_wald_consistency"]
    selected_state_wald_consistency = isinstance(
        selected_state_wald, (bool, np.bool_)
    ) and bool(selected_state_wald)
    base.update(
        {
            "zero_edge_rank_fusion_status": "scored",
            "zero_edge_rank_fusion_selected_center": str(winner["center"]),
            "zero_edge_rank_fusion_selected_outer1": str(winner["outer1"]),
            "zero_edge_rank_fusion_selected_outer2": str(winner["outer2"]),
            "zero_edge_rank_fusion_arm1_logit": float(arm1["logit"]),
            "zero_edge_rank_fusion_arm2_logit": float(arm2["logit"]),
            "zero_edge_rank_fusion_arm1_probability": float(arm1["probability"]),
            "zero_edge_rank_fusion_arm2_probability": float(arm2["probability"]),
            "zero_edge_rank_fusion_selected_center_score": float(winner["score"]),
            "zero_edge_rank_fusion_runner_center_score": float(runner["score"]),
            "zero_edge_rank_fusion_center_score_margin": float(
                winner["score"] - runner["score"]
            ),
            "zero_edge_rank_fusion_selected_state_wald_consistency": (
                selected_state_wald_consistency
            ),
            "zero_edge_rank_fusion_selected_outer_wald_view_support": float(
                selected["minimum_outer_conditional_wald_view_support"]
            ),
            "zero_edge_rank_fusion_selected_practical_fold_support": float(
                selected["laplace_both_practical_fold_support"]
            ),
            "zero_edge_rank_fusion_arm1_evidence_json": json.dumps(
                arm1, sort_keys=True, separators=(",", ":")
            ),
            "zero_edge_rank_fusion_arm2_evidence_json": json.dumps(
                arm2, sort_keys=True, separators=(",", ":")
            ),
            "zero_edge_rank_fusion_all_edge_scores_json": json.dumps(
                [
                    {
                        "parent": parent,
                        "child": child,
                        "logit": float(edge_evidence[(parent, child)]["logit"]),
                        "probability": float(
                            edge_evidence[(parent, child)]["probability"]
                        ),
                    }
                    for parent, child in sorted(edge_evidence)
                ],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "zero_edge_rank_fusion_all_center_scores_json": json.dumps(
                center_scores, sort_keys=True, separators=(",", ":")
            ),
        }
    )
    return base, selected


def _tier1u_gate_failures(
    *,
    candidate: pd.Series,
    runner_margin: float,
    score_improvement: float,
    config: CrossfitHierarchicalColliderConfig,
) -> list[str]:
    checks = (
        (
            "crossfit_joint_gain_fold_min_not_positive",
            float(candidate.crossfit_joint_gain_fold_min)
            > config.tier1u_minimum_crossfit_joint_gain_fold_min_exclusive,
        ),
        (
            "laplace_fold_parent_posterior_below_threshold",
            float(candidate.laplace_fold_min_parent_positive_posterior)
            >= config.tier1u_minimum_laplace_fold_parent_positive_posterior,
        ),
        (
            "both_positive_fold_support_below_threshold",
            float(candidate.laplace_both_positive_fold_support)
            >= config.tier1u_minimum_both_positive_fold_support,
        ),
        (
            "both_practical_fold_support_below_threshold",
            float(candidate.laplace_both_practical_fold_support)
            >= config.tier1u_minimum_both_practical_fold_support,
        ),
        (
            "arm_empirical_positive_p_above_threshold",
            float(candidate.maximum_arm_empirical_positive_p)
            <= config.tier1u_maximum_arm_empirical_positive_p,
        ),
        (
            "outer_empirical_null_compatibility_below_threshold",
            float(candidate.outer_empirical_null_compatibility_p)
            >= config.tier1u_minimum_outer_empirical_null_compatibility_p,
        ),
        (
            "eligible_runner_margin_below_threshold",
            float(runner_margin)
            >= config.tier1u_minimum_eligible_runner_score_margin,
        ),
        (
            "production_score_improvement_below_threshold",
            float(score_improvement)
            >= config.tier1u_minimum_exact_production_score_improvement,
        ),
    )
    return [reason for reason, passed in checks if not passed]


def _tier1r_gate_failures(
    *,
    candidate: pd.Series,
    production_candidate: pd.Series,
    global_runner_margin: float,
    score_improvement: float,
    config: CrossfitHierarchicalColliderConfig,
) -> list[str]:
    """Audit the all-fold residual center-reorientation gate."""

    checks = (
        (
            "crossfit_joint_gain_fold_min_not_positive",
            float(candidate.crossfit_joint_gain_fold_min)
            > config.tier1r_minimum_crossfit_joint_gain_fold_min_exclusive,
        ),
        (
            "crossfit_min_drop_one_gain_not_positive",
            float(candidate.crossfit_min_drop_one_gain)
            > config.tier1r_minimum_crossfit_min_drop_one_gain_exclusive,
        ),
        (
            "laplace_mean_parent_posterior_below_threshold",
            float(candidate.laplace_min_parent_positive_posterior)
            >= config.tier1r_minimum_laplace_mean_parent_positive_posterior,
        ),
        (
            "laplace_fold_parent_posterior_below_threshold",
            float(candidate.laplace_fold_min_parent_positive_posterior)
            >= config.tier1r_minimum_laplace_fold_parent_positive_posterior,
        ),
        (
            "both_positive_fold_support_below_threshold",
            float(candidate.laplace_both_positive_fold_support)
            >= config.tier1r_minimum_both_positive_fold_support,
        ),
        (
            "both_practical_fold_support_below_threshold",
            float(candidate.laplace_both_practical_fold_support)
            >= config.tier1r_minimum_both_practical_fold_support,
        ),
        (
            "arm_empirical_positive_p_above_threshold",
            float(candidate.maximum_arm_empirical_positive_p)
            <= config.tier1r_maximum_arm_empirical_positive_p,
        ),
        (
            "outer_empirical_null_compatibility_below_threshold",
            float(candidate.outer_empirical_null_compatibility_p)
            >= config.tier1r_minimum_outer_empirical_null_compatibility_p,
        ),
        (
            "global_runner_margin_below_threshold",
            float(global_runner_margin)
            >= config.tier1r_minimum_global_runner_score_margin,
        ),
        (
            "production_score_improvement_below_threshold",
            float(score_improvement)
            >= config.tier1r_minimum_exact_production_score_improvement,
        ),
        (
            "replacement_does_not_change_center",
            not config.tier1r_require_center_change
            or str(candidate.center) != str(production_candidate.center),
        ),
    )
    return [reason for reason, passed in checks if not passed]


def _weak_multicenter_refinement_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    observed_refinement_action: str,
    higher_priority_replacement_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None, bool]:
    """Validate an observed multicenter resolution or conservatively demote it."""

    audit: dict[str, Any] = {
        "multicenter_validation_considered": False,
        "multicenter_validation_status": "not_applicable",
        "multicenter_validation_abstention_reason": "",
        "multicenter_validation_production_fold_parent_posterior": math.nan,
        "multicenter_validation_production_drop_one_gain": math.nan,
        "multicenter_validation_production_is_weak": False,
        "multicenter_validation_eligible_alternatives": 0,
        "multicenter_validation_selected_replacement": False,
        "multicenter_validation_selected_demotion": False,
        "multicenter_validation_candidate_center": "",
        "multicenter_validation_candidate_outer1": "",
        "multicenter_validation_candidate_outer2": "",
        "multicenter_validation_candidate_raw_generative_score": math.nan,
        "multicenter_validation_candidate_current_skeleton_arm_support": 0,
    }
    if observed_refinement_action != "resolve_competing_multicenter_path":
        audit["multicenter_validation_abstention_reason"] = (
            "production_is_not_an_observed_multicenter_resolution"
        )
        return audit, None, False
    if production is None:
        audit["multicenter_validation_abstention_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return audit, None, False

    audit["multicenter_validation_considered"] = True
    fold_posterior = float(
        production.laplace_fold_min_parent_positive_posterior
    )
    drop_one_gain = float(production.crossfit_min_drop_one_gain)
    production_is_weak = bool(
        fold_posterior
        < config.multicenter_minimum_production_fold_parent_posterior
        or drop_one_gain
        <= config.multicenter_minimum_production_drop_one_gain_exclusive
    )
    audit.update(
        {
            "multicenter_validation_production_fold_parent_posterior": (
                fold_posterior
            ),
            "multicenter_validation_production_drop_one_gain": drop_one_gain,
            "multicenter_validation_production_is_weak": production_is_weak,
        }
    )
    if higher_priority_replacement_selected:
        audit["multicenter_validation_status"] = "not_needed_higher_priority_selected"
        audit["multicenter_validation_abstention_reason"] = (
            "a_higher_priority_replacement_already_selected_a_star"
        )
        return audit, None, False
    if not production_is_weak:
        audit["multicenter_validation_status"] = "retained"
        audit["multicenter_validation_abstention_reason"] = (
            "production_passed_fold_posterior_and_drop_one_gates"
        )
        return audit, None, False

    production_skeleton = {
        canonical_edge(parent, child) for parent, child in production_edges
    }
    current_skeleton_support = group.apply(
        lambda row: len(
            {
                canonical_edge(str(row.center), str(row.outer1)),
                canonical_edge(str(row.center), str(row.outer2)),
            }
            & production_skeleton
        ),
        axis=1,
    )
    alternative = group.loc[
        group.apply(lambda row: _star_edges(row) != production_edges, axis=1)
        & group["two_parent_positive_state_wald_consistency"].astype(bool)
        & group["minimum_outer_conditional_wald_view_support"].ge(
            config.multicenter_alternative_minimum_outer_wald_view_support
        )
        & group["laplace_min_parent_positive_posterior"].ge(
            config.multicenter_alternative_minimum_parent_posterior
        )
        & group["laplace_fold_min_parent_positive_posterior"].ge(
            config.multicenter_alternative_minimum_fold_parent_posterior
        )
        & group["laplace_both_positive_fold_support"].ge(
            config.multicenter_alternative_minimum_both_positive_fold_support
        )
        & group["laplace_both_practical_fold_support"].ge(
            config.multicenter_alternative_minimum_both_practical_fold_support
        )
        & group["maximum_arm_empirical_positive_p"].le(
            config.multicenter_alternative_maximum_arm_empirical_positive_p
        )
        & group["outer_empirical_null_compatibility_p"].ge(
            config.multicenter_alternative_minimum_outer_empirical_null_compatibility_p
        )
        & group["raw_generative_score"].ge(
            config.multicenter_alternative_minimum_raw_generative_score
        )
        & group["crossfit_min_drop_one_gain"].gt(
            config.multicenter_alternative_minimum_drop_one_gain_exclusive
        )
        & current_skeleton_support.ge(
            config.multicenter_alternative_minimum_current_skeleton_arm_support
        )
    ].copy()
    alternative["_current_skeleton_arm_support"] = current_skeleton_support.loc[
        alternative.index
    ]
    alternative = _ordered_candidates(alternative)
    audit["multicenter_validation_eligible_alternatives"] = int(len(alternative))
    if len(alternative) != 1:
        audit["multicenter_validation_status"] = "demoted"
        audit["multicenter_validation_selected_demotion"] = True
        audit["multicenter_validation_abstention_reason"] = (
            "weak_production_lacked_one_unique_validated_alternative"
        )
        return audit, None, True

    candidate = alternative.iloc[0]
    audit.update(
        {
            "multicenter_validation_status": "replaced",
            "multicenter_validation_selected_replacement": True,
            "multicenter_validation_candidate_center": str(candidate.center),
            "multicenter_validation_candidate_outer1": str(candidate.outer1),
            "multicenter_validation_candidate_outer2": str(candidate.outer2),
            "multicenter_validation_candidate_raw_generative_score": float(
                candidate.raw_generative_score
            ),
            "multicenter_validation_candidate_current_skeleton_arm_support": int(
                candidate["_current_skeleton_arm_support"]
            ),
        }
    )
    return audit, candidate, False


def _centrality_provenance_by_assay(
    current_direct_candidates: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Extract one internally consistent observed centrality record per assay."""

    if (
        current_direct_candidates.empty
        or "observed_refinement_action" not in current_direct_candidates.columns
    ):
        return {}
    centrality = current_direct_candidates.loc[
        current_direct_candidates["observed_refinement_action"]
        .fillna("")
        .astype(str)
        .eq(OBSERVED_CENTRALITY_REORIENTATION_ACTION)
    ].copy()
    if centrality.empty:
        return {}
    string_fields = (
        "precentrality_center",
        "precentrality_outer1",
        "precentrality_outer2",
        "centrality_selected_new_parent",
    )
    numeric_fields = (
        "centrality_mean_center_vs_outer_lor_contrast",
        "centrality_minimum_center_vs_outer_lor_contrast",
    )
    _require_columns(
        centrality,
        {"source_assay", *string_fields, *numeric_fields},
        "centrality-refined current_direct_candidates",
    )
    output: dict[str, dict[str, Any]] = {}
    for assay, assay_rows in centrality.groupby("source_assay", sort=True):
        record: dict[str, Any] = {}
        for field in string_fields:
            values = {
                str(value).strip()
                for value in assay_rows[field]
                if str(value).strip()
            }
            if len(values) != 1:
                raise ValueError(
                    f"Centrality provenance for {assay}.{field} must contain "
                    "one nonempty value shared by every direct arm"
                )
            record[field] = next(iter(values))
        for field in numeric_fields:
            values = pd.to_numeric(assay_rows[field], errors="coerce")
            if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
                raise ValueError(
                    f"Centrality provenance for {assay}.{field} must be finite"
                )
            unique = np.unique(values.to_numpy(dtype=float))
            if len(unique) != 1:
                raise ValueError(
                    f"Centrality provenance for {assay}.{field} must be identical "
                    "on every direct arm"
                )
            record[field] = float(unique[0])
        precentrality_nodes = {
            str(record["precentrality_center"]),
            str(record["precentrality_outer1"]),
            str(record["precentrality_outer2"]),
        }
        if len(precentrality_nodes) != 3:
            raise ValueError(
                f"Centrality provenance for {assay} must identify three distinct "
                "pre-centrality nodes"
            )
        output[str(assay)] = record
    return output


def _candidate_with_identity(
    group: pd.DataFrame,
    *,
    center: str,
    outer1: str,
    outer2: str,
) -> pd.Series | None:
    """Return one exhaustive candidate with the requested canonical identity."""

    left, right = sorted((str(outer1), str(outer2)))
    matches = group.loc[
        group["center"].astype(str).eq(str(center))
        & group["outer1"].astype(str).eq(left)
        & group["outer2"].astype(str).eq(right)
    ]
    if len(matches) > 1:
        raise ValueError("Candidate universe contains a duplicate star identity")
    return matches.iloc[0] if len(matches) == 1 else None


def _centrality_completion_validation_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    observed_refinement_action: str,
    provenance: Mapping[str, Any] | None,
    higher_priority_replacement_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None, bool]:
    """Treat beta-centrality completion as provisional crossfit structure.

    A strong full proposal is retained.  A weak proposal may only keep its
    proposed center on the original three-node skeleton when that exact star
    passes state, fold, posterior, and empirical-null gates.  Otherwise the
    method reverts to the stable observed pre-centrality completion; if that
    candidate is unavailable, it removes the unsupported orientation.
    """

    audit: dict[str, Any] = {
        "centrality_validation_considered": False,
        "centrality_validation_status": "not_applicable",
        "centrality_validation_reason": "",
        "centrality_validation_provenance_present": provenance is not None,
        "centrality_validation_full_minimum_contrast": math.nan,
        "centrality_validation_full_parent_posterior": math.nan,
        "centrality_validation_full_fold_parent_posterior": math.nan,
        "centrality_validation_full_positive_fold_support": math.nan,
        "centrality_validation_full_practical_fold_support": math.nan,
        "centrality_validation_full_arm_empirical_positive_p": math.nan,
        "centrality_validation_full_drop_one_gain": math.nan,
        "centrality_validation_full_is_strong": False,
        "centrality_validation_precompletion_center": "",
        "centrality_validation_precompletion_outer1": "",
        "centrality_validation_precompletion_outer2": "",
        "centrality_validation_precompletion_candidate_available": False,
        "centrality_validation_constrained_center": "",
        "centrality_validation_constrained_outer1": "",
        "centrality_validation_constrained_outer2": "",
        "centrality_validation_constrained_candidate_available": False,
        "centrality_validation_constrained_state_wald_consistency": False,
        "centrality_validation_constrained_outer_wald_view_support": math.nan,
        "centrality_validation_constrained_raw_generative_score": math.nan,
        "centrality_validation_constrained_parent_posterior": math.nan,
        "centrality_validation_constrained_fold_parent_posterior": math.nan,
        "centrality_validation_constrained_positive_fold_support": math.nan,
        "centrality_validation_constrained_practical_fold_support": math.nan,
        "centrality_validation_constrained_arm_empirical_positive_p": math.nan,
        "centrality_validation_constrained_passed": False,
        "centrality_validation_selected_replacement": False,
        "centrality_validation_selected_demotion": False,
    }
    if observed_refinement_action != OBSERVED_CENTRALITY_REORIENTATION_ACTION:
        audit["centrality_validation_reason"] = (
            "production_is_not_an_observed_beta_centrality_reorientation"
        )
        return audit, None, False

    audit["centrality_validation_considered"] = True
    if production is None:
        audit["centrality_validation_status"] = "abstained"
        audit["centrality_validation_reason"] = (
            "centrality_production_is_not_an_exact_two_arm_star"
        )
        return audit, None, False

    full_metrics = {
        "centrality_validation_full_parent_posterior": float(
            production.laplace_min_parent_positive_posterior
        ),
        "centrality_validation_full_fold_parent_posterior": float(
            production.laplace_fold_min_parent_positive_posterior
        ),
        "centrality_validation_full_positive_fold_support": float(
            production.laplace_both_positive_fold_support
        ),
        "centrality_validation_full_practical_fold_support": float(
            production.laplace_both_practical_fold_support
        ),
        "centrality_validation_full_arm_empirical_positive_p": float(
            production.maximum_arm_empirical_positive_p
        ),
        "centrality_validation_full_drop_one_gain": float(
            production.crossfit_min_drop_one_gain
        ),
    }
    audit.update(full_metrics)
    if higher_priority_replacement_selected:
        audit["centrality_validation_status"] = "not_needed_higher_priority_selected"
        audit["centrality_validation_reason"] = (
            "a_higher_priority_replacement_already_selected_a_star"
        )
        return audit, None, False
    if provenance is None:
        audit["centrality_validation_status"] = "demoted"
        audit["centrality_validation_reason"] = (
            "centrality_provenance_is_missing"
        )
        audit["centrality_validation_selected_demotion"] = True
        return audit, None, True

    contrast = float(provenance["centrality_minimum_center_vs_outer_lor_contrast"])
    audit["centrality_validation_full_minimum_contrast"] = contrast
    full_is_strong = bool(
        contrast >= config.centrality_full_minimum_contrast
        and full_metrics["centrality_validation_full_parent_posterior"]
        >= config.centrality_full_minimum_parent_posterior
        and full_metrics["centrality_validation_full_fold_parent_posterior"]
        >= config.centrality_full_minimum_fold_parent_posterior
        and full_metrics["centrality_validation_full_positive_fold_support"]
        >= config.centrality_full_minimum_both_positive_fold_support
        and full_metrics["centrality_validation_full_practical_fold_support"]
        >= config.centrality_full_minimum_both_practical_fold_support
        and full_metrics["centrality_validation_full_arm_empirical_positive_p"]
        <= config.centrality_full_maximum_arm_empirical_positive_p
        and full_metrics["centrality_validation_full_drop_one_gain"]
        > config.centrality_full_minimum_drop_one_gain_exclusive
    )
    audit["centrality_validation_full_is_strong"] = full_is_strong
    if full_is_strong:
        audit["centrality_validation_status"] = "retained"
        audit["centrality_validation_reason"] = (
            "full_centrality_proposal_passed_contrast_and_uncertainty_gates"
        )
        return audit, None, False

    precenter = str(provenance["precentrality_center"])
    preouter1, preouter2 = sorted(
        (
            str(provenance["precentrality_outer1"]),
            str(provenance["precentrality_outer2"]),
        )
    )
    audit.update(
        {
            "centrality_validation_precompletion_center": precenter,
            "centrality_validation_precompletion_outer1": preouter1,
            "centrality_validation_precompletion_outer2": preouter2,
        }
    )
    precompletion = _candidate_with_identity(
        group,
        center=precenter,
        outer1=preouter1,
        outer2=preouter2,
    )
    audit["centrality_validation_precompletion_candidate_available"] = (
        precompletion is not None
    )

    original_nodes = {precenter, preouter1, preouter2}
    production_center = str(production.center)
    constrained: pd.Series | None = None
    if production_center in original_nodes:
        constrained_outers = sorted(original_nodes - {production_center})
        audit.update(
            {
                "centrality_validation_constrained_center": production_center,
                "centrality_validation_constrained_outer1": constrained_outers[0],
                "centrality_validation_constrained_outer2": constrained_outers[1],
            }
        )
        constrained = _candidate_with_identity(
            group,
            center=production_center,
            outer1=constrained_outers[0],
            outer2=constrained_outers[1],
        )
    audit["centrality_validation_constrained_candidate_available"] = (
        constrained is not None
    )
    constrained_passed = False
    if constrained is not None and _star_edges(constrained) != production_edges:
        constrained_metrics = {
            "centrality_validation_constrained_state_wald_consistency": bool(
                constrained.two_parent_positive_state_wald_consistency
            ),
            "centrality_validation_constrained_outer_wald_view_support": float(
                constrained.minimum_outer_conditional_wald_view_support
            ),
            "centrality_validation_constrained_raw_generative_score": float(
                constrained.raw_generative_score
            ),
            "centrality_validation_constrained_parent_posterior": float(
                constrained.laplace_min_parent_positive_posterior
            ),
            "centrality_validation_constrained_fold_parent_posterior": float(
                constrained.laplace_fold_min_parent_positive_posterior
            ),
            "centrality_validation_constrained_positive_fold_support": float(
                constrained.laplace_both_positive_fold_support
            ),
            "centrality_validation_constrained_practical_fold_support": float(
                constrained.laplace_both_practical_fold_support
            ),
            "centrality_validation_constrained_arm_empirical_positive_p": float(
                constrained.maximum_arm_empirical_positive_p
            ),
        }
        audit.update(constrained_metrics)
        constrained_passed = bool(
            constrained_metrics[
                "centrality_validation_constrained_state_wald_consistency"
            ]
            and constrained_metrics[
                "centrality_validation_constrained_outer_wald_view_support"
            ]
            >= config.centrality_skeleton_minimum_outer_wald_view_support
            and constrained_metrics[
                "centrality_validation_constrained_raw_generative_score"
            ]
            >= config.centrality_skeleton_minimum_raw_generative_score
            and constrained_metrics[
                "centrality_validation_constrained_parent_posterior"
            ]
            >= config.centrality_skeleton_minimum_parent_posterior
            and constrained_metrics[
                "centrality_validation_constrained_fold_parent_posterior"
            ]
            >= config.centrality_skeleton_minimum_fold_parent_posterior
            and constrained_metrics[
                "centrality_validation_constrained_positive_fold_support"
            ]
            >= config.centrality_skeleton_minimum_both_positive_fold_support
            and constrained_metrics[
                "centrality_validation_constrained_practical_fold_support"
            ]
            >= config.centrality_skeleton_minimum_both_practical_fold_support
            and constrained_metrics[
                "centrality_validation_constrained_arm_empirical_positive_p"
            ]
            <= config.centrality_skeleton_maximum_arm_empirical_positive_p
        )
    audit["centrality_validation_constrained_passed"] = constrained_passed
    if constrained_passed:
        audit["centrality_validation_status"] = "constrained_to_original_skeleton"
        audit["centrality_validation_reason"] = (
            "weak_full_proposal_but_selected_center_passed_on_original_skeleton"
        )
        audit["centrality_validation_selected_replacement"] = True
        return audit, constrained, False
    if precompletion is not None and _star_edges(precompletion) != production_edges:
        audit["centrality_validation_status"] = "reverted_to_precompletion"
        audit["centrality_validation_reason"] = (
            "weak_full_and_constrained_proposals_reverted_to_stable_precompletion"
        )
        audit["centrality_validation_selected_replacement"] = True
        return audit, precompletion, False

    audit["centrality_validation_status"] = "demoted"
    audit["centrality_validation_reason"] = (
        "weak_centrality_proposal_lacked_a_recoverable_completion"
    )
    audit["centrality_validation_selected_demotion"] = True
    return audit, None, True


def _dual_anchor_provenance_by_assay(
    current_direct_candidates: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Extract the stable observed anchor behind each dual completion."""

    if (
        current_direct_candidates.empty
        or "observed_refinement_action" not in current_direct_candidates.columns
    ):
        return {}
    rows = current_direct_candidates.loc[
        current_direct_candidates["observed_refinement_action"]
        .fillna("")
        .astype(str)
        .eq(OBSERVED_DUAL_ANCHOR_COMPLETION_ACTION)
    ].copy()
    if rows.empty:
        return {}
    _require_columns(
        rows,
        {
            "source_assay",
            "completion_anchor_node1",
            "completion_anchor_node2",
            "completion_winner_view_count",
            "completion_n_views",
        },
        "dual-anchor current_direct_candidates",
    )
    output: dict[str, dict[str, Any]] = {}
    for assay, assay_rows in rows.groupby("source_assay", sort=True):
        record: dict[str, Any] = {}
        for field in ("completion_anchor_node1", "completion_anchor_node2"):
            values = {
                str(value).strip()
                for value in assay_rows[field]
                if str(value).strip()
            }
            if len(values) != 1:
                raise ValueError(
                    f"Dual-anchor provenance for {assay}.{field} must contain "
                    "one nonempty value shared by every direct arm"
                )
            record[field] = next(iter(values))
        for field in ("completion_winner_view_count", "completion_n_views"):
            values = pd.to_numeric(assay_rows[field], errors="coerce")
            if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
                raise ValueError(
                    f"Dual-anchor provenance for {assay}.{field} must be finite"
                )
            unique = np.unique(values.to_numpy(dtype=float))
            if len(unique) != 1 or not float(unique[0]).is_integer():
                raise ValueError(
                    f"Dual-anchor provenance for {assay}.{field} must be one "
                    "shared integer"
                )
            record[field] = int(unique[0])
        if (
            record["completion_anchor_node1"]
            == record["completion_anchor_node2"]
        ):
            raise ValueError(f"Dual-anchor provenance for {assay} is a self-edge")
        winner = int(record["completion_winner_view_count"])
        total = int(record["completion_n_views"])
        if total < 1 or winner < 0 or winner > total:
            raise ValueError(
                f"Dual-anchor provenance for {assay} has invalid view counts"
            )
        output[str(assay)] = record
    return output


def _dual_anchor_completion_validation_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    observed_refinement_action: str,
    provenance: Mapping[str, Any] | None,
    retained_state: set[str],
    higher_priority_replacement_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Replace only a residually contradicted observed anchor completion."""

    audit: dict[str, Any] = {
        "dual_anchor_validation_considered": False,
        "dual_anchor_validation_status": "not_applicable",
        "dual_anchor_validation_reason": "",
        "dual_anchor_validation_provenance_present": provenance is not None,
        "dual_anchor_validation_observed_winner_fraction": math.nan,
        "dual_anchor_validation_production_collider_separation": math.nan,
        "dual_anchor_validation_production_is_residually_inconsistent": False,
        "dual_anchor_validation_production_has_nonpositive_collider_separation": False,
        "dual_anchor_validation_strict_eligible_alternatives": 0,
        "dual_anchor_validation_structural_eligible_alternatives": 0,
        "dual_anchor_validation_state_completion_eligible_alternatives": 0,
        "dual_anchor_validation_eligible_alternatives": 0,
        "dual_anchor_validation_selected_mode": "",
        "dual_anchor_validation_selected_replacement": False,
        "dual_anchor_validation_selected_center": "",
        "dual_anchor_validation_selected_outer1": "",
        "dual_anchor_validation_selected_outer2": "",
        "dual_anchor_validation_selected_collider_separation": math.nan,
        "dual_anchor_validation_selected_collider_separation_improvement": math.nan,
        "dual_anchor_validation_selected_raw_generative_score": math.nan,
        "dual_anchor_validation_selected_parent_posterior": math.nan,
        "dual_anchor_validation_selected_fold_parent_posterior": math.nan,
    }
    if observed_refinement_action != OBSERVED_DUAL_ANCHOR_COMPLETION_ACTION:
        audit["dual_anchor_validation_reason"] = (
            "production_is_not_an_observed_dual_anchor_completion"
        )
        return audit, None
    audit["dual_anchor_validation_considered"] = True
    if production is None:
        audit["dual_anchor_validation_status"] = "abstained"
        audit["dual_anchor_validation_reason"] = (
            "dual_anchor_production_is_not_an_exact_two_arm_star"
        )
        return audit, None
    production_separation = float(production.minimum_collider_separation)
    audit["dual_anchor_validation_production_collider_separation"] = (
        production_separation
    )
    if higher_priority_replacement_selected:
        audit["dual_anchor_validation_status"] = (
            "not_needed_higher_priority_selected"
        )
        audit["dual_anchor_validation_reason"] = (
            "a_higher_priority_replacement_already_selected_a_star"
        )
        return audit, None
    if provenance is None:
        audit["dual_anchor_validation_status"] = "abstained"
        audit["dual_anchor_validation_reason"] = "dual_anchor_provenance_is_missing"
        return audit, None

    winner_fraction = float(provenance["completion_winner_view_count"]) / float(
        provenance["completion_n_views"]
    )
    production_is_inconsistent = bool(
        production_separation
        < config.dual_anchor_maximum_production_collider_separation_exclusive
    )
    audit.update(
        {
            "dual_anchor_validation_observed_winner_fraction": winner_fraction,
            "dual_anchor_validation_production_is_residually_inconsistent": (
                production_is_inconsistent
            ),
            "dual_anchor_validation_production_has_nonpositive_collider_separation": (
                production_separation <= 0.0
            ),
        }
    )
    if (
        winner_fraction
        < config.dual_anchor_minimum_observed_winner_fraction
    ):
        audit["dual_anchor_validation_status"] = "retained"
        audit["dual_anchor_validation_reason"] = (
            "observed_completion_lacked_full_view_winner_stability_for_refinement"
        )
        return audit, None
    if production_separation > 0.0:
        audit["dual_anchor_validation_status"] = "retained"
        audit["dual_anchor_validation_reason"] = (
            "production_already_has_positive_conditional_collider_separation"
        )
        return audit, None

    anchor = canonical_edge(
        str(provenance["completion_anchor_node1"]),
        str(provenance["completion_anchor_node2"]),
    )
    production_center = str(production.center)
    if production_center not in anchor or anchor not in {
        canonical_edge(production_center, str(production.outer1)),
        canonical_edge(production_center, str(production.outer2)),
    }:
        audit["dual_anchor_validation_status"] = "abstained"
        audit["dual_anchor_validation_reason"] = (
            "production_does_not_preserve_the_recorded_anchor"
        )
        return audit, None

    contains_anchor = group.apply(
        lambda row: anchor
        in {
            canonical_edge(str(row.center), str(row.outer1)),
            canonical_edge(str(row.center), str(row.outer2)),
        },
        axis=1,
    )
    separation_improvement = (
        group["minimum_collider_separation"] - production_separation
    )
    strict_alternatives = _ordered_candidates(
        group.loc[
            production_is_inconsistent
            & group["center"].astype(str).eq(production_center)
            & contains_anchor
            & group.apply(lambda row: _star_edges(row) != production_edges, axis=1)
            & group["two_parent_positive_state_wald_consistency"].astype(bool)
            & group["minimum_outer_conditional_wald_view_support"].ge(
                config.dual_anchor_alternative_minimum_outer_wald_view_support
            )
            & group["raw_generative_score"].ge(
                config.dual_anchor_alternative_minimum_raw_generative_score
            )
            & group["crossfit_joint_gain_over_baseline"].ge(
                config.dual_anchor_alternative_minimum_joint_gain
            )
            & group["laplace_min_parent_positive_posterior"].ge(
                config.dual_anchor_alternative_minimum_parent_posterior
            )
            & group["laplace_fold_min_parent_positive_posterior"].ge(
                config.dual_anchor_alternative_minimum_fold_parent_posterior
            )
            & group["laplace_both_positive_fold_support"].ge(
                config.dual_anchor_alternative_minimum_both_positive_fold_support
            )
            & group["laplace_both_practical_fold_support"].ge(
                config.dual_anchor_alternative_minimum_both_practical_fold_support
            )
            & group["maximum_arm_empirical_positive_p"].le(
                config.dual_anchor_alternative_maximum_arm_empirical_positive_p
            )
            & group["minimum_collider_separation"].ge(
                config.dual_anchor_alternative_minimum_collider_separation
            )
            & separation_improvement.ge(
                config.dual_anchor_alternative_minimum_collider_separation_improvement
            )
        ]
    )
    structural_alternatives = _ordered_candidates(
        group.loc[
            group["center"].astype(str).eq(production_center)
            & contains_anchor
            & group.apply(lambda row: _star_edges(row) != production_edges, axis=1)
            & group["raw_generative_score"].ge(
                config.dual_anchor_structural_minimum_raw_generative_score
            )
            & group["crossfit_joint_gain_over_baseline"].ge(
                config.dual_anchor_structural_minimum_joint_gain
            )
            & group["laplace_min_parent_positive_posterior"].ge(
                config.dual_anchor_structural_minimum_parent_posterior
            )
            & group["laplace_fold_min_parent_positive_posterior"].ge(
                config.dual_anchor_structural_minimum_fold_parent_posterior
            )
            & group["laplace_both_positive_fold_support"].ge(
                config.dual_anchor_structural_minimum_both_positive_fold_support
            )
            & group["laplace_both_practical_fold_support"].ge(
                config.dual_anchor_structural_minimum_both_practical_fold_support
            )
            & group["minimum_collider_separation"].gt(
                config.dual_anchor_structural_minimum_collider_separation_exclusive
            )
            & separation_improvement.ge(
                config.dual_anchor_structural_minimum_collider_separation_improvement
            )
        ]
    )
    anchor_parent = next(node for node in anchor if node != production_center)
    production_parents = {str(production.outer1), str(production.outer2)}
    completion_parents = production_parents - {anchor_parent}
    state_completion_alternatives = pd.DataFrame()
    if len(completion_parents) == 1:
        completion_parent = next(iter(completion_parents))
        added_state_parent = group.apply(
            lambda row: (
                len(
                    {str(row.outer1), str(row.outer2)} - production_parents
                )
                == 1
                and next(
                    iter(
                        {str(row.outer1), str(row.outer2)}
                        - production_parents
                    )
                )
                in retained_state
            ),
            axis=1,
        )
        state_completion_alternatives = _ordered_candidates(
            group.loc[
                production_is_inconsistent
                & (completion_parent not in retained_state)
                & group["center"].astype(str).eq(production_center)
                & contains_anchor
                & group.apply(
                    lambda row: _star_edges(row) != production_edges,
                    axis=1,
                )
                & added_state_parent
                & group["raw_generative_score"].ge(
                    config.dual_anchor_state_completion_minimum_raw_generative_score
                )
                & group["crossfit_joint_gain_over_baseline"].ge(
                    config.dual_anchor_state_completion_minimum_joint_gain
                )
                & group["crossfit_joint_gain_fold_min"].ge(
                    config.dual_anchor_state_completion_minimum_fold_joint_gain
                )
                & group["laplace_min_parent_positive_posterior"].ge(
                    config.dual_anchor_state_completion_minimum_parent_posterior
                )
                & group["laplace_fold_min_parent_positive_posterior"].ge(
                    config.dual_anchor_state_completion_minimum_fold_parent_posterior
                )
                & group["laplace_both_positive_fold_support"].ge(
                    config.dual_anchor_state_completion_minimum_both_positive_fold_support
                )
                & group["laplace_both_practical_fold_support"].ge(
                    config.dual_anchor_state_completion_minimum_both_practical_fold_support
                )
                & group["maximum_arm_empirical_positive_p"].le(
                    config.dual_anchor_state_completion_maximum_arm_empirical_positive_p
                )
                & separation_improvement.ge(
                    config.dual_anchor_state_completion_minimum_collider_separation_improvement
                )
            ]
        )
    audit["dual_anchor_validation_strict_eligible_alternatives"] = int(
        len(strict_alternatives)
    )
    audit["dual_anchor_validation_structural_eligible_alternatives"] = int(
        len(structural_alternatives)
    )
    audit[
        "dual_anchor_validation_state_completion_eligible_alternatives"
    ] = int(len(state_completion_alternatives))
    selected_mode = ""
    selected: pd.Series | None = None
    if len(strict_alternatives) == 1:
        selected = strict_alternatives.iloc[0]
        selected_mode = "state_uncertainty_and_residual"
    elif len(strict_alternatives) == 0 and len(structural_alternatives) == 1:
        selected = structural_alternatives.iloc[0]
        selected_mode = "unique_positive_conditional_collider_separation"
    elif (
        len(strict_alternatives) == 0
        and len(structural_alternatives) == 0
        and len(state_completion_alternatives) == 1
    ):
        selected = state_completion_alternatives.iloc[0]
        selected_mode = "unique_state_member_fold_residual_completion"
    audit["dual_anchor_validation_eligible_alternatives"] = int(
        len(strict_alternatives)
        if len(strict_alternatives)
        else (
            len(structural_alternatives)
            if len(structural_alternatives)
            else len(state_completion_alternatives)
        )
    )
    if selected is None:
        audit["dual_anchor_validation_status"] = "retained"
        audit["dual_anchor_validation_reason"] = (
            "nonpositive_separation_production_lacked_one_unique_validated_"
            "anchor_preserving_alternative"
        )
        return audit, None

    selected_separation = float(selected.minimum_collider_separation)
    audit.update(
        {
            "dual_anchor_validation_status": "replaced",
            "dual_anchor_validation_reason": (
                "one_unique_anchor_preserving_alternative_resolved_the_"
                "production_residual_contradiction"
            ),
            "dual_anchor_validation_selected_replacement": True,
            "dual_anchor_validation_selected_mode": selected_mode,
            "dual_anchor_validation_selected_center": str(selected.center),
            "dual_anchor_validation_selected_outer1": str(selected.outer1),
            "dual_anchor_validation_selected_outer2": str(selected.outer2),
            "dual_anchor_validation_selected_collider_separation": (
                selected_separation
            ),
            "dual_anchor_validation_selected_collider_separation_improvement": (
                selected_separation - production_separation
            ),
            "dual_anchor_validation_selected_raw_generative_score": float(
                selected.raw_generative_score
            ),
            "dual_anchor_validation_selected_parent_posterior": float(
                selected.laplace_min_parent_positive_posterior
            ),
            "dual_anchor_validation_selected_fold_parent_posterior": float(
                selected.laplace_fold_min_parent_positive_posterior
            ),
        }
    )
    return audit, selected


def _state_inconsistent_neighbor_star_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    higher_priority_replacement_selected: bool,
    higher_priority_demotion_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Replace one state-inconsistent star only with a unique adjacent star.

    This validator is deliberately local to the production skeleton: a
    candidate must retain exactly one undirected production adjacency.  The
    retained adjacency may reverse when the supported candidate changes the
    collider center, but wholesale graph rewrites are not allowed.
    """

    audit: dict[str, Any] = {
        "neighbor_star_validation_considered": False,
        "neighbor_star_validation_status": "not_applicable",
        "neighbor_star_validation_reason": "",
        "neighbor_star_validation_production_state_wald_consistency": False,
        "neighbor_star_validation_production_outer_wald_view_support": math.nan,
        "neighbor_star_validation_production_raw_generative_score": math.nan,
        "neighbor_star_validation_production_joint_gain": math.nan,
        "neighbor_star_validation_production_drop_one_gain": math.nan,
        "neighbor_star_validation_eligible_candidates": 0,
        "neighbor_star_validation_selected": False,
        "neighbor_star_validation_selected_center": "",
        "neighbor_star_validation_selected_outer1": "",
        "neighbor_star_validation_selected_outer2": "",
        "neighbor_star_validation_selected_raw_generative_score": math.nan,
        "neighbor_star_validation_selected_raw_fraction_of_production": math.nan,
        "neighbor_star_validation_selected_parent_posterior": math.nan,
        "neighbor_star_validation_selected_fold_parent_posterior": math.nan,
        "neighbor_star_validation_selected_outer_wald_view_support": math.nan,
        "neighbor_star_validation_selected_shared_adjacencies": 0,
    }
    if production is None or len(production_edges) != 2:
        audit["neighbor_star_validation_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return audit, None
    audit["neighbor_star_validation_considered"] = True
    production_state_consistent = bool(
        production.two_parent_positive_state_wald_consistency
    )
    production_outer_support = float(
        production.minimum_outer_conditional_wald_view_support
    )
    production_raw_score = float(production.raw_generative_score)
    production_joint_gain = float(production.crossfit_joint_gain_over_baseline)
    production_drop_one_gain = float(production.crossfit_min_drop_one_gain)
    audit.update(
        {
            "neighbor_star_validation_production_state_wald_consistency": (
                production_state_consistent
            ),
            "neighbor_star_validation_production_outer_wald_view_support": (
                production_outer_support
            ),
            "neighbor_star_validation_production_raw_generative_score": (
                production_raw_score
            ),
            "neighbor_star_validation_production_joint_gain": (
                production_joint_gain
            ),
            "neighbor_star_validation_production_drop_one_gain": (
                production_drop_one_gain
            ),
        }
    )
    if higher_priority_replacement_selected:
        audit["neighbor_star_validation_status"] = (
            "not_needed_higher_priority_replacement_selected"
        )
        audit["neighbor_star_validation_reason"] = (
            "a_higher_priority_validator_already_selected_a_star"
        )
        return audit, None
    if higher_priority_demotion_selected:
        audit["neighbor_star_validation_status"] = (
            "not_needed_higher_priority_demotion_selected"
        )
        audit["neighbor_star_validation_reason"] = (
            "a_higher_priority_validator_selected_subtractive_demotion"
        )
        return audit, None
    if production_state_consistent:
        audit["neighbor_star_validation_status"] = "retained"
        audit["neighbor_star_validation_reason"] = (
            "production_already_has_positive_state_wald_consistency"
        )
        return audit, None
    if (
        production_outer_support
        >= config.neighbor_star_maximum_production_outer_wald_support_exclusive
    ):
        audit["neighbor_star_validation_status"] = "retained"
        audit["neighbor_star_validation_reason"] = (
            "production_already_has_outer_wald_view_support"
        )
        return audit, None
    if (
        production_raw_score
        >= config.neighbor_star_maximum_production_raw_generative_score_exclusive
        or production_joint_gain
        > config.neighbor_star_maximum_production_joint_gain
        or production_drop_one_gain
        > config.neighbor_star_maximum_production_drop_one_gain
    ):
        audit["neighbor_star_validation_status"] = "retained"
        audit["neighbor_star_validation_reason"] = (
            "production_residual_evidence_is_too_strong_for_state_based_replacement"
        )
        return audit, None

    production_adjacencies = {
        canonical_edge(parent, child) for parent, child in production_edges
    }
    shared_adjacencies = group.apply(
        lambda row: len(
            production_adjacencies
            & {
                canonical_edge(str(row.center), str(row.outer1)),
                canonical_edge(str(row.center), str(row.outer2)),
            }
        ),
        axis=1,
    )
    different_star = group.apply(
        lambda row: _star_edges(row) != production_edges,
        axis=1,
    )
    minimum_relative_score = (
        config.neighbor_star_minimum_raw_score_fraction_of_production
        * production_raw_score
    )
    eligible = _ordered_candidates(
        group.loc[
            different_star
            & shared_adjacencies.eq(1)
            & group["two_parent_positive_state_wald_consistency"].astype(bool)
            & group["raw_generative_score"].ge(
                config.neighbor_star_minimum_raw_generative_score
            )
            & group["raw_generative_score"].ge(minimum_relative_score)
            & group["crossfit_joint_gain_over_baseline"].ge(
                config.neighbor_star_minimum_joint_gain
            )
            & group["crossfit_min_drop_one_gain"].ge(
                config.neighbor_star_minimum_drop_one_gain
            )
            & group["crossfit_joint_gain_fold_min"].ge(
                config.neighbor_star_minimum_fold_joint_gain
            )
            & group["laplace_min_parent_positive_posterior"].ge(
                config.neighbor_star_minimum_parent_posterior
            )
            & group["laplace_fold_min_parent_positive_posterior"].ge(
                config.neighbor_star_minimum_fold_parent_posterior
            )
            & group["laplace_both_positive_fold_support"].ge(
                config.neighbor_star_minimum_both_positive_fold_support
            )
            & group["laplace_both_practical_fold_support"].ge(
                config.neighbor_star_minimum_both_practical_fold_support
            )
            & group["maximum_arm_empirical_positive_p"].le(
                config.neighbor_star_maximum_arm_empirical_positive_p
            )
            & group["minimum_outer_conditional_wald_view_support"].ge(
                config.neighbor_star_minimum_outer_wald_view_support
            )
            & group[
                "laplace_min_parent_positive_posterior_reverse_center_margin"
            ].ge(
                config.neighbor_star_minimum_reverse_center_parent_posterior_margin
            )
        ]
    )
    audit["neighbor_star_validation_eligible_candidates"] = int(len(eligible))
    if len(eligible) != 1:
        audit["neighbor_star_validation_status"] = "retained"
        audit["neighbor_star_validation_reason"] = (
            "eligible_adjacent_state_supported_candidate_count_is_not_exactly_one"
        )
        return audit, None

    selected = eligible.iloc[0]
    selected_raw_score = float(selected.raw_generative_score)
    selected_fraction = (
        selected_raw_score / production_raw_score
        if production_raw_score > 0.0
        else math.inf
    )
    audit.update(
        {
            "neighbor_star_validation_status": "replaced",
            "neighbor_star_validation_reason": (
                "one_adjacent_star_uniquely_restored_state_and_uncertainty_support"
            ),
            "neighbor_star_validation_selected": True,
            "neighbor_star_validation_selected_center": str(selected.center),
            "neighbor_star_validation_selected_outer1": str(selected.outer1),
            "neighbor_star_validation_selected_outer2": str(selected.outer2),
            "neighbor_star_validation_selected_raw_generative_score": (
                selected_raw_score
            ),
            "neighbor_star_validation_selected_raw_fraction_of_production": (
                selected_fraction
            ),
            "neighbor_star_validation_selected_parent_posterior": float(
                selected.laplace_min_parent_positive_posterior
            ),
            "neighbor_star_validation_selected_fold_parent_posterior": float(
                selected.laplace_fold_min_parent_positive_posterior
            ),
            "neighbor_star_validation_selected_outer_wald_view_support": float(
                selected.minimum_outer_conditional_wald_view_support
            ),
            "neighbor_star_validation_selected_shared_adjacencies": 1,
        }
    )
    return audit, selected


def _state_inconsistent_same_center_substitution_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    retained_state: set[str],
    higher_priority_replacement_selected: bool,
    higher_priority_demotion_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Replace one state-inconsistent parent without changing the center."""

    audit: dict[str, Any] = {
        "same_center_substitution_considered": False,
        "same_center_substitution_status": "not_applicable",
        "same_center_substitution_reason": "",
        "same_center_substitution_production_center_is_state_member": False,
        "same_center_substitution_production_state_parent_count": 0,
        "same_center_substitution_production_state_wald_consistency": False,
        "same_center_substitution_production_outer_wald_view_support": math.nan,
        "same_center_substitution_production_raw_generative_score": math.nan,
        "same_center_substitution_production_minimum_collider_separation": math.nan,
        "same_center_substitution_production_median_collider_separation": math.nan,
        "same_center_substitution_structural_candidates": 0,
        "same_center_substitution_eligible_candidates": 0,
        "same_center_substitution_selected": False,
        "same_center_substitution_selected_center": "",
        "same_center_substitution_selected_outer1": "",
        "same_center_substitution_selected_outer2": "",
        "same_center_substitution_selected_raw_generative_score": math.nan,
        "same_center_substitution_selected_joint_gain": math.nan,
        "same_center_substitution_selected_drop_one_gain": math.nan,
        "same_center_substitution_selected_fold_joint_gain": math.nan,
        "same_center_substitution_selected_parent_posterior": math.nan,
        "same_center_substitution_selected_fold_parent_posterior": math.nan,
        "same_center_substitution_selected_minimum_collider_separation": math.nan,
        "same_center_substitution_selected_median_collider_separation": math.nan,
        "same_center_substitution_selected_collider_separation_improvement": math.nan,
        "same_center_substitution_selected_collider_separation_runner_margin": math.nan,
    }
    if production is None or len(production_edges) != 2:
        audit["same_center_substitution_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return audit, None
    if "median_collider_separation" not in group.columns:
        audit["same_center_substitution_reason"] = (
            "median_collider_separation_evidence_is_missing"
        )
        return audit, None

    audit["same_center_substitution_considered"] = True
    production_center = str(production.center)
    production_parents = {str(production.outer1), str(production.outer2)}
    retained_production_parents = production_parents & retained_state
    center_is_state = production_center in retained_state
    state_parent_count = len(retained_production_parents)
    production_state_consistent = bool(
        production.two_parent_positive_state_wald_consistency
    )
    production_outer_support = float(
        production.minimum_outer_conditional_wald_view_support
    )
    production_raw_score = float(production.raw_generative_score)
    production_minimum_separation = float(production.minimum_collider_separation)
    production_median_separation = float(production.median_collider_separation)
    audit.update(
        {
            "same_center_substitution_production_center_is_state_member": (
                center_is_state
            ),
            "same_center_substitution_production_state_parent_count": (
                state_parent_count
            ),
            "same_center_substitution_production_state_wald_consistency": (
                production_state_consistent
            ),
            "same_center_substitution_production_outer_wald_view_support": (
                production_outer_support
            ),
            "same_center_substitution_production_raw_generative_score": (
                production_raw_score
            ),
            "same_center_substitution_production_minimum_collider_separation": (
                production_minimum_separation
            ),
            "same_center_substitution_production_median_collider_separation": (
                production_median_separation
            ),
        }
    )
    if higher_priority_replacement_selected:
        audit["same_center_substitution_status"] = (
            "not_needed_higher_priority_replacement_selected"
        )
        audit["same_center_substitution_reason"] = (
            "a_higher_priority_validator_already_selected_a_star"
        )
        return audit, None
    if higher_priority_demotion_selected:
        audit["same_center_substitution_status"] = (
            "not_needed_higher_priority_demotion_selected"
        )
        audit["same_center_substitution_reason"] = (
            "a_higher_priority_validator_selected_subtractive_demotion"
        )
        return audit, None
    if center_is_state or state_parent_count != 1:
        audit["same_center_substitution_status"] = "retained"
        audit["same_center_substitution_reason"] = (
            "production_does_not_have_one_retained_state_parent_above_a_nonstate_center"
        )
        return audit, None
    if production_state_consistent or production_outer_support >= 1.0:
        audit["same_center_substitution_status"] = "retained"
        audit["same_center_substitution_reason"] = (
            "production_parents_are_not_state_inconsistent"
        )
        return audit, None
    if (
        production_raw_score
        >= config.same_center_substitution_maximum_production_raw_generative_score_exclusive
    ):
        audit["same_center_substitution_status"] = "retained"
        audit["same_center_substitution_reason"] = (
            "production_residual_score_is_too_strong_for_parent_substitution"
        )
        return audit, None

    retained_parent = next(iter(retained_production_parents))
    production_adjacencies = {
        canonical_edge(parent, child) for parent, child in production_edges
    }
    shared_adjacencies = group.apply(
        lambda row: len(
            production_adjacencies
            & {
                canonical_edge(str(row.center), str(row.outer1)),
                canonical_edge(str(row.center), str(row.outer2)),
            }
        ),
        axis=1,
    )
    structural = group.loc[
        group["center"].astype(str).eq(production_center)
        & group.apply(lambda row: _star_edges(row) != production_edges, axis=1)
        & shared_adjacencies.eq(1)
        & group.apply(
            lambda row: retained_parent in {str(row.outer1), str(row.outer2)},
            axis=1,
        )
        & group["two_parent_positive_state_wald_consistency"].astype(bool)
    ].copy()
    audit["same_center_substitution_structural_candidates"] = int(len(structural))
    if structural.empty:
        audit["same_center_substitution_status"] = "retained"
        audit["same_center_substitution_reason"] = (
            "no_state_consistent_same_center_parent_substitution_exists"
        )
        return audit, None

    separation_runner_margins = pd.Series(
        math.inf, index=structural.index, dtype=float
    )
    if len(structural) > 1:
        for index in structural.index:
            separation_runner_margins.loc[index] = float(
                structural.loc[index, "minimum_collider_separation"]
                - structural.loc[
                    structural.index != index, "minimum_collider_separation"
                ].max()
            )
    separation_improvement = (
        structural["minimum_collider_separation"]
        - production_minimum_separation
    )
    eligible = _ordered_candidates(
        structural.loc[
            structural["raw_generative_score"].ge(
                config.same_center_substitution_minimum_raw_generative_score
            )
            & structural["crossfit_joint_gain_over_baseline"].ge(
                config.same_center_substitution_minimum_joint_gain
            )
            & structural["crossfit_min_drop_one_gain"].ge(
                config.same_center_substitution_minimum_drop_one_gain
            )
            & structural["crossfit_joint_gain_fold_min"].ge(
                config.same_center_substitution_minimum_fold_joint_gain
            )
            & structural["laplace_min_parent_positive_posterior"].ge(
                config.same_center_substitution_minimum_parent_posterior
            )
            & structural["laplace_fold_min_parent_positive_posterior"].ge(
                config.same_center_substitution_minimum_fold_parent_posterior
            )
            & structural["laplace_both_positive_fold_support"].ge(
                config.same_center_substitution_minimum_both_positive_fold_support
            )
            & structural["laplace_both_practical_fold_support"].ge(
                config.same_center_substitution_minimum_both_practical_fold_support
            )
            & structural["maximum_arm_empirical_positive_p"].le(
                config.same_center_substitution_maximum_arm_empirical_positive_p
            )
            & structural["outer_empirical_null_compatibility_p"].ge(
                config.same_center_substitution_minimum_outer_empirical_null_compatibility_p
            )
            & structural["minimum_outer_conditional_wald_view_support"].ge(
                config.same_center_substitution_minimum_outer_wald_view_support
            )
            & structural["minimum_collider_separation"].ge(
                config.same_center_substitution_minimum_collider_separation
            )
            & structural["median_collider_separation"].ge(
                config.same_center_substitution_minimum_median_collider_separation
            )
            & separation_improvement.ge(
                config.same_center_substitution_minimum_collider_separation_improvement
            )
            & separation_runner_margins.ge(
                config.same_center_substitution_minimum_collider_separation_runner_margin
            )
        ]
    )
    audit["same_center_substitution_eligible_candidates"] = int(len(eligible))
    if len(eligible) != 1:
        audit["same_center_substitution_status"] = "retained"
        audit["same_center_substitution_reason"] = (
            "eligible_same_center_parent_substitution_count_is_not_exactly_one"
        )
        return audit, None

    selected = eligible.iloc[0]
    selected_minimum_separation = float(selected.minimum_collider_separation)
    selected_median_separation = float(selected.median_collider_separation)
    selected_runner_margin = float(separation_runner_margins.loc[selected.name])
    audit.update(
        {
            "same_center_substitution_status": "replaced",
            "same_center_substitution_reason": (
                "one_state_consistent_parent_substitution_uniquely_stabilized_"
                "the_same_center_across_aggregation_views"
            ),
            "same_center_substitution_selected": True,
            "same_center_substitution_selected_center": str(selected.center),
            "same_center_substitution_selected_outer1": str(selected.outer1),
            "same_center_substitution_selected_outer2": str(selected.outer2),
            "same_center_substitution_selected_raw_generative_score": float(
                selected.raw_generative_score
            ),
            "same_center_substitution_selected_joint_gain": float(
                selected.crossfit_joint_gain_over_baseline
            ),
            "same_center_substitution_selected_drop_one_gain": float(
                selected.crossfit_min_drop_one_gain
            ),
            "same_center_substitution_selected_fold_joint_gain": float(
                selected.crossfit_joint_gain_fold_min
            ),
            "same_center_substitution_selected_parent_posterior": float(
                selected.laplace_min_parent_positive_posterior
            ),
            "same_center_substitution_selected_fold_parent_posterior": float(
                selected.laplace_fold_min_parent_positive_posterior
            ),
            "same_center_substitution_selected_minimum_collider_separation": (
                selected_minimum_separation
            ),
            "same_center_substitution_selected_median_collider_separation": (
                selected_median_separation
            ),
            "same_center_substitution_selected_collider_separation_improvement": (
                selected_minimum_separation - production_minimum_separation
            ),
            "same_center_substitution_selected_collider_separation_runner_margin": (
                selected_runner_margin
            ),
        }
    )
    return audit, selected


def _state_center_sibling_rotation_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    retained_state: set[str],
    higher_priority_replacement_selected: bool,
    higher_priority_demotion_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Rotate a state-loaded center through one loading-equivalent sibling.

    This route addresses a specific shared-state failure mode: a direct star
    can use a pure state-associated feature as its center while a production
    parent has a stable loading-equivalent sibling.  Replacement is allowed
    only when the sibling swap improves both sparse-count and conditional
    collider evidence, and exactly one adjacent candidate passes every gate.
    """

    audit: dict[str, Any] = {
        "state_center_sibling_rotation_considered": False,
        "state_center_sibling_rotation_status": "not_applicable",
        "state_center_sibling_rotation_reason": "",
        "state_center_sibling_rotation_production_center_is_state_member": False,
        "state_center_sibling_rotation_production_state_parent_count": 0,
        "state_center_sibling_rotation_production_state_wald_consistency": False,
        "state_center_sibling_rotation_production_median_collider_separation": math.nan,
        "state_center_sibling_rotation_structural_candidates": 0,
        "state_center_sibling_rotation_eligible_candidates": 0,
        "state_center_sibling_rotation_selected": False,
        "state_center_sibling_rotation_selected_center": "",
        "state_center_sibling_rotation_selected_outer1": "",
        "state_center_sibling_rotation_selected_outer2": "",
        "state_center_sibling_rotation_selected_displaced_node": "",
        "state_center_sibling_rotation_selected_new_node": "",
        "state_center_sibling_rotation_selected_minimum_loading_cosine": math.nan,
        "state_center_sibling_rotation_selected_maximum_loading_empirical_p": math.nan,
        "state_center_sibling_rotation_selected_log_odds_gain": math.nan,
        "state_center_sibling_rotation_selected_empirical_p_improvement": math.nan,
        "state_center_sibling_rotation_selected_raw_generative_score": math.nan,
        "state_center_sibling_rotation_selected_joint_gain": math.nan,
        "state_center_sibling_rotation_selected_drop_one_gain": math.nan,
        "state_center_sibling_rotation_selected_fold_joint_gain": math.nan,
        "state_center_sibling_rotation_selected_parent_posterior": math.nan,
        "state_center_sibling_rotation_selected_fold_parent_posterior": math.nan,
        "state_center_sibling_rotation_selected_median_collider_separation": math.nan,
        "state_center_sibling_rotation_selected_collider_separation_improvement": math.nan,
    }
    if production is None or len(production_edges) != 2:
        audit["state_center_sibling_rotation_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return audit, None
    required_annotations = {
        "median_collider_separation",
        "state_center_sibling_rotation_structural_candidate",
        "state_center_sibling_rotation_displaced_node",
        "state_center_sibling_rotation_new_node",
        "state_center_sibling_rotation_minimum_loading_cosine",
        "state_center_sibling_rotation_maximum_loading_empirical_p",
        "state_center_sibling_rotation_log_odds_gain",
        "state_center_sibling_rotation_empirical_p_improvement",
    }
    if not required_annotations.issubset(group.columns):
        audit["state_center_sibling_rotation_reason"] = (
            "loading_sibling_rotation_evidence_was_not_annotated"
        )
        return audit, None

    audit["state_center_sibling_rotation_considered"] = True
    production_center = str(production.center)
    production_parents = {str(production.outer1), str(production.outer2)}
    center_is_state = production_center in retained_state
    state_parent_count = len(production_parents & retained_state)
    production_state_consistent = bool(
        production.two_parent_positive_state_wald_consistency
    )
    production_separation = float(production.median_collider_separation)
    audit.update(
        {
            "state_center_sibling_rotation_production_center_is_state_member": (
                center_is_state
            ),
            "state_center_sibling_rotation_production_state_parent_count": (
                state_parent_count
            ),
            "state_center_sibling_rotation_production_state_wald_consistency": (
                production_state_consistent
            ),
            "state_center_sibling_rotation_production_median_collider_separation": (
                production_separation
            ),
        }
    )
    if higher_priority_replacement_selected:
        audit["state_center_sibling_rotation_status"] = (
            "not_needed_higher_priority_replacement_selected"
        )
        audit["state_center_sibling_rotation_reason"] = (
            "a_higher_priority_validator_already_selected_a_star"
        )
        return audit, None
    if higher_priority_demotion_selected:
        audit["state_center_sibling_rotation_status"] = (
            "not_needed_higher_priority_demotion_selected"
        )
        audit["state_center_sibling_rotation_reason"] = (
            "a_higher_priority_validator_selected_subtractive_demotion"
        )
        return audit, None
    if not center_is_state or state_parent_count != 0:
        audit["state_center_sibling_rotation_status"] = "retained"
        audit["state_center_sibling_rotation_reason"] = (
            "production_does_not_isolate_state_membership_at_its_center"
        )
        return audit, None
    if not production_state_consistent:
        audit["state_center_sibling_rotation_status"] = "retained"
        audit["state_center_sibling_rotation_reason"] = (
            "production_parents_lack_positive_state_wald_consistency"
        )
        return audit, None

    structural = group.loc[
        group["state_center_sibling_rotation_structural_candidate"].astype(bool)
        & ~group["center"].astype(str).isin(retained_state)
    ].copy()
    audit["state_center_sibling_rotation_structural_candidates"] = int(
        len(structural)
    )
    separation_improvement = (
        structural["median_collider_separation"] - production_separation
    )
    eligible = _ordered_candidates(
        structural.loc[
            structural["two_parent_positive_state_wald_consistency"].astype(bool)
            & structural["state_center_sibling_rotation_minimum_loading_cosine"].ge(
                config.state_center_sibling_rotation_minimum_loading_cosine
            )
            & structural[
                "state_center_sibling_rotation_maximum_loading_empirical_p"
            ].le(config.state_center_sibling_rotation_maximum_loading_empirical_p)
            & structural["state_center_sibling_rotation_log_odds_gain"].ge(
                config.state_center_sibling_rotation_minimum_log_odds_gain
            )
            & structural[
                "state_center_sibling_rotation_empirical_p_improvement"
            ].ge(
                config.state_center_sibling_rotation_minimum_empirical_p_improvement
            )
            & structural["raw_generative_score"].ge(
                config.state_center_sibling_rotation_minimum_raw_generative_score
            )
            & structural["crossfit_joint_gain_over_baseline"].ge(
                config.state_center_sibling_rotation_minimum_joint_gain
            )
            & structural["crossfit_min_drop_one_gain"].ge(
                config.state_center_sibling_rotation_minimum_drop_one_gain
            )
            & structural["crossfit_joint_gain_fold_min"].ge(
                config.state_center_sibling_rotation_minimum_fold_joint_gain
            )
            & structural["laplace_min_parent_positive_posterior"].ge(
                config.state_center_sibling_rotation_minimum_parent_posterior
            )
            & structural["laplace_fold_min_parent_positive_posterior"].ge(
                config.state_center_sibling_rotation_minimum_fold_parent_posterior
            )
            & structural["laplace_both_positive_fold_support"].ge(
                config.state_center_sibling_rotation_minimum_both_positive_fold_support
            )
            & structural["laplace_both_practical_fold_support"].ge(
                config.state_center_sibling_rotation_minimum_both_practical_fold_support
            )
            & structural["maximum_arm_empirical_positive_p"].le(
                config.state_center_sibling_rotation_maximum_arm_empirical_positive_p
            )
            & structural["outer_empirical_null_compatibility_p"].ge(
                config.state_center_sibling_rotation_minimum_outer_empirical_null_compatibility_p
            )
            & structural["minimum_outer_conditional_wald_view_support"].ge(
                config.state_center_sibling_rotation_minimum_outer_wald_view_support
            )
            & structural[
                "laplace_min_parent_positive_posterior_reverse_center_margin"
            ].ge(
                config.state_center_sibling_rotation_minimum_reverse_center_parent_posterior_margin
            )
            & structural["median_collider_separation"].ge(
                config.state_center_sibling_rotation_minimum_median_collider_separation
            )
            & separation_improvement.ge(
                config.state_center_sibling_rotation_minimum_median_collider_separation_improvement
            )
        ]
    )
    audit["state_center_sibling_rotation_eligible_candidates"] = int(len(eligible))
    if len(eligible) != 1:
        audit["state_center_sibling_rotation_status"] = "retained"
        audit["state_center_sibling_rotation_reason"] = (
            "eligible_loading_sibling_rotation_count_is_not_exactly_one"
        )
        return audit, None

    selected = eligible.iloc[0]
    selected_separation = float(selected.median_collider_separation)
    audit.update(
        {
            "state_center_sibling_rotation_status": "replaced",
            "state_center_sibling_rotation_reason": (
                "one_loading_equivalent_sibling_rotation_uniquely_separated_"
                "shared_state_from_direct_structure"
            ),
            "state_center_sibling_rotation_selected": True,
            "state_center_sibling_rotation_selected_center": str(selected.center),
            "state_center_sibling_rotation_selected_outer1": str(selected.outer1),
            "state_center_sibling_rotation_selected_outer2": str(selected.outer2),
            "state_center_sibling_rotation_selected_displaced_node": str(
                selected.state_center_sibling_rotation_displaced_node
            ),
            "state_center_sibling_rotation_selected_new_node": str(
                selected.state_center_sibling_rotation_new_node
            ),
            "state_center_sibling_rotation_selected_minimum_loading_cosine": float(
                selected.state_center_sibling_rotation_minimum_loading_cosine
            ),
            "state_center_sibling_rotation_selected_maximum_loading_empirical_p": float(
                selected.state_center_sibling_rotation_maximum_loading_empirical_p
            ),
            "state_center_sibling_rotation_selected_log_odds_gain": float(
                selected.state_center_sibling_rotation_log_odds_gain
            ),
            "state_center_sibling_rotation_selected_empirical_p_improvement": float(
                selected.state_center_sibling_rotation_empirical_p_improvement
            ),
            "state_center_sibling_rotation_selected_raw_generative_score": float(
                selected.raw_generative_score
            ),
            "state_center_sibling_rotation_selected_joint_gain": float(
                selected.crossfit_joint_gain_over_baseline
            ),
            "state_center_sibling_rotation_selected_drop_one_gain": float(
                selected.crossfit_min_drop_one_gain
            ),
            "state_center_sibling_rotation_selected_fold_joint_gain": float(
                selected.crossfit_joint_gain_fold_min
            ),
            "state_center_sibling_rotation_selected_parent_posterior": float(
                selected.laplace_min_parent_positive_posterior
            ),
            "state_center_sibling_rotation_selected_fold_parent_posterior": float(
                selected.laplace_fold_min_parent_positive_posterior
            ),
            "state_center_sibling_rotation_selected_median_collider_separation": (
                selected_separation
            ),
            "state_center_sibling_rotation_selected_collider_separation_improvement": (
                selected_separation - production_separation
            ),
        }
    )
    return audit, selected


def _state_loaded_parent_rotation_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    production_edges: set[tuple[str, str]],
    retained_state: set[str],
    higher_priority_replacement_selected: bool,
    higher_priority_demotion_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Rotate one state-loaded production parent into a residual-supported center."""

    audit: dict[str, Any] = {
        "state_parent_rotation_considered": False,
        "state_parent_rotation_status": "not_applicable",
        "state_parent_rotation_reason": "",
        "state_parent_rotation_production_center_is_state_member": False,
        "state_parent_rotation_production_state_parent_count": 0,
        "state_parent_rotation_production_state_wald_consistency": False,
        "state_parent_rotation_production_outer_wald_view_support": math.nan,
        "state_parent_rotation_production_raw_generative_score": math.nan,
        "state_parent_rotation_production_minimum_collider_separation": math.nan,
        "state_parent_rotation_production_median_collider_separation": math.nan,
        "state_parent_rotation_strict_eligible_candidates": 0,
        "state_parent_rotation_topology_eligible_candidates": 0,
        "state_parent_rotation_eligible_candidates": 0,
        "state_parent_rotation_selected_mode": "",
        "state_parent_rotation_selected": False,
        "state_parent_rotation_selected_center": "",
        "state_parent_rotation_selected_outer1": "",
        "state_parent_rotation_selected_outer2": "",
        "state_parent_rotation_selected_raw_generative_score": math.nan,
        "state_parent_rotation_selected_raw_fraction_of_production": math.nan,
        "state_parent_rotation_selected_parent_posterior": math.nan,
        "state_parent_rotation_selected_fold_parent_posterior": math.nan,
        "state_parent_rotation_selected_reverse_center_joint_gain_margin": math.nan,
        "state_parent_rotation_selected_reverse_center_parent_posterior_margin": math.nan,
        "state_parent_rotation_selected_minimum_collider_separation": math.nan,
        "state_parent_rotation_selected_median_collider_separation": math.nan,
        "state_parent_rotation_selected_minimum_collider_separation_improvement": math.nan,
        "state_parent_rotation_selected_median_collider_separation_improvement": math.nan,
        "state_parent_rotation_selected_median_collider_separation_runner_margin": math.nan,
    }
    if production is None or len(production_edges) != 2:
        audit["state_parent_rotation_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return audit, None
    audit["state_parent_rotation_considered"] = True
    production_center = str(production.center)
    production_parents = {str(production.outer1), str(production.outer2)}
    production_state_parent_count = len(production_parents & retained_state)
    production_center_is_state_member = production_center in retained_state
    production_state_consistent = bool(
        production.two_parent_positive_state_wald_consistency
    )
    production_outer_support = float(
        production.minimum_outer_conditional_wald_view_support
    )
    production_raw_score = float(production.raw_generative_score)
    production_minimum_separation = float(production.minimum_collider_separation)
    production_median_separation = float(production.median_collider_separation)
    audit.update(
        {
            "state_parent_rotation_production_center_is_state_member": (
                production_center_is_state_member
            ),
            "state_parent_rotation_production_state_parent_count": (
                production_state_parent_count
            ),
            "state_parent_rotation_production_state_wald_consistency": (
                production_state_consistent
            ),
            "state_parent_rotation_production_outer_wald_view_support": (
                production_outer_support
            ),
            "state_parent_rotation_production_raw_generative_score": (
                production_raw_score
            ),
            "state_parent_rotation_production_minimum_collider_separation": (
                production_minimum_separation
            ),
            "state_parent_rotation_production_median_collider_separation": (
                production_median_separation
            ),
        }
    )
    if higher_priority_replacement_selected:
        audit["state_parent_rotation_status"] = (
            "not_needed_higher_priority_replacement_selected"
        )
        audit["state_parent_rotation_reason"] = (
            "a_higher_priority_validator_already_selected_a_star"
        )
        return audit, None
    if higher_priority_demotion_selected:
        audit["state_parent_rotation_status"] = (
            "not_needed_higher_priority_demotion_selected"
        )
        audit["state_parent_rotation_reason"] = (
            "a_higher_priority_validator_selected_subtractive_demotion"
        )
        return audit, None
    if production_center_is_state_member or production_state_parent_count < 1:
        audit["state_parent_rotation_status"] = "retained"
        audit["state_parent_rotation_reason"] = (
            "production_does_not_place_a_state_member_above_a_nonstate_center"
        )
        return audit, None
    if production_state_consistent:
        audit["state_parent_rotation_status"] = "retained"
        audit["state_parent_rotation_reason"] = (
            "production_already_has_positive_state_wald_consistency"
        )
        return audit, None
    if (
        production_outer_support
        >= config.state_parent_rotation_maximum_production_outer_wald_support_exclusive
    ):
        audit["state_parent_rotation_status"] = "retained"
        audit["state_parent_rotation_reason"] = (
            "production_already_has_outer_wald_view_support"
        )
        return audit, None

    production_adjacencies = {
        canonical_edge(parent, child) for parent, child in production_edges
    }
    shared_adjacencies = group.apply(
        lambda row: len(
            production_adjacencies
            & {
                canonical_edge(str(row.center), str(row.outer1)),
                canonical_edge(str(row.center), str(row.outer2)),
            }
        ),
        axis=1,
    )
    rotates_state_parent = group.apply(
        lambda row: (
            str(row.center) in production_parents
            and str(row.center) in retained_state
            and production_center in {str(row.outer1), str(row.outer2)}
            and not ({str(row.outer1), str(row.outer2)} & retained_state)
        ),
        axis=1,
    )
    minimum_relative_score = (
        config.state_parent_rotation_minimum_raw_score_fraction_of_production
        * production_raw_score
    )
    structural_mask = (
        rotates_state_parent
        & shared_adjacencies.eq(1)
        & group.apply(lambda row: _star_edges(row) != production_edges, axis=1)
    )
    strict_eligible = _ordered_candidates(
        group.loc[
            structural_mask
            & group["raw_generative_score"].ge(
                config.state_parent_rotation_minimum_raw_generative_score
            )
            & group["raw_generative_score"].ge(minimum_relative_score)
            & group["crossfit_joint_gain_over_baseline"].ge(
                config.state_parent_rotation_minimum_joint_gain
            )
            & group["crossfit_min_drop_one_gain"].ge(
                config.state_parent_rotation_minimum_drop_one_gain
            )
            & group["crossfit_joint_gain_fold_min"].ge(
                config.state_parent_rotation_minimum_fold_joint_gain
            )
            & group["laplace_min_parent_positive_posterior"].ge(
                config.state_parent_rotation_minimum_parent_posterior
            )
            & group["laplace_fold_min_parent_positive_posterior"].ge(
                config.state_parent_rotation_minimum_fold_parent_posterior
            )
            & group["laplace_both_positive_fold_support"].ge(
                config.state_parent_rotation_minimum_both_positive_fold_support
            )
            & group["laplace_both_practical_fold_support"].ge(
                config.state_parent_rotation_minimum_both_practical_fold_support
            )
            & group["maximum_arm_empirical_positive_p"].le(
                config.state_parent_rotation_maximum_arm_empirical_positive_p
            )
            & group[
                "crossfit_joint_gain_over_baseline_reverse_center_margin"
            ].ge(
                config.state_parent_rotation_minimum_reverse_center_joint_gain_margin
            )
            & group[
                "laplace_min_parent_positive_posterior_reverse_center_margin"
            ].ge(
                config.state_parent_rotation_minimum_reverse_center_parent_posterior_margin
            )
        ]
    )
    topology_structural = group.loc[
        structural_mask
        & group["two_parent_positive_state_wald_consistency"].astype(bool)
    ].copy()
    topology_runner_margins = pd.Series(
        math.inf, index=topology_structural.index, dtype=float
    )
    if len(topology_structural) > 1:
        for index in topology_structural.index:
            topology_runner_margins.loc[index] = float(
                topology_structural.loc[index, "median_collider_separation"]
                - topology_structural.loc[
                    topology_structural.index != index,
                    "median_collider_separation",
                ].max()
            )
    topology_minimum_improvement = (
        topology_structural["minimum_collider_separation"]
        - production_minimum_separation
    )
    topology_median_improvement = (
        topology_structural["median_collider_separation"]
        - production_median_separation
    )
    topology_eligible = _ordered_candidates(
        topology_structural.loc[
            (production_raw_score
             < config.state_parent_rotation_topology_maximum_production_raw_generative_score_exclusive)
            & topology_structural["raw_generative_score"].ge(
                config.state_parent_rotation_topology_minimum_raw_generative_score
            )
            & topology_structural["crossfit_joint_gain_over_baseline"].ge(
                config.state_parent_rotation_topology_minimum_joint_gain
            )
            & topology_structural["crossfit_min_drop_one_gain"].ge(
                config.state_parent_rotation_topology_minimum_drop_one_gain
            )
            & topology_structural["crossfit_joint_gain_fold_min"].ge(
                config.state_parent_rotation_topology_minimum_fold_joint_gain
            )
            & topology_structural["laplace_min_parent_positive_posterior"].ge(
                config.state_parent_rotation_topology_minimum_parent_posterior
            )
            & topology_structural[
                "laplace_fold_min_parent_positive_posterior"
            ].ge(
                config.state_parent_rotation_topology_minimum_fold_parent_posterior
            )
            & topology_structural["laplace_both_positive_fold_support"].ge(
                config.state_parent_rotation_topology_minimum_both_positive_fold_support
            )
            & topology_structural["laplace_both_practical_fold_support"].ge(
                config.state_parent_rotation_topology_minimum_both_practical_fold_support
            )
            & topology_structural["maximum_arm_empirical_positive_p"].le(
                config.state_parent_rotation_topology_maximum_arm_empirical_positive_p
            )
            & topology_structural["outer_empirical_null_compatibility_p"].ge(
                config.state_parent_rotation_topology_minimum_outer_empirical_null_compatibility_p
            )
            & topology_structural[
                "minimum_outer_conditional_wald_view_support"
            ].ge(
                config.state_parent_rotation_topology_minimum_outer_wald_view_support
            )
            & topology_structural["minimum_collider_separation"].ge(
                config.state_parent_rotation_topology_minimum_collider_separation
            )
            & topology_structural["median_collider_separation"].ge(
                config.state_parent_rotation_topology_minimum_median_collider_separation
            )
            & topology_minimum_improvement.ge(
                config.state_parent_rotation_topology_minimum_collider_separation_improvement
            )
            & topology_median_improvement.ge(
                config.state_parent_rotation_topology_minimum_median_collider_separation_improvement
            )
            & topology_runner_margins.ge(
                config.state_parent_rotation_topology_minimum_median_collider_separation_runner_margin
            )
        ]
    )
    audit["state_parent_rotation_strict_eligible_candidates"] = int(
        len(strict_eligible)
    )
    audit["state_parent_rotation_topology_eligible_candidates"] = int(
        len(topology_eligible)
    )
    selected_mode = ""
    if len(strict_eligible) == 1:
        eligible = strict_eligible
        selected_mode = "strict_residual_uncertainty_rotation"
    elif len(strict_eligible) == 0 and len(topology_eligible) == 1:
        eligible = topology_eligible
        selected_mode = "unique_cross_resolution_topology_rotation"
    else:
        eligible = pd.DataFrame()
    audit["state_parent_rotation_eligible_candidates"] = int(len(eligible))
    if len(eligible) != 1:
        audit["state_parent_rotation_status"] = "retained"
        audit["state_parent_rotation_reason"] = (
            "neither_strict_nor_topology_state_parent_rotation_is_unique"
        )
        return audit, None

    selected = eligible.iloc[0]
    selected_raw_score = float(selected.raw_generative_score)
    selected_minimum_separation = float(selected.minimum_collider_separation)
    selected_median_separation = float(selected.median_collider_separation)
    selected_topology_runner_margin = float(
        topology_runner_margins.get(selected.name, math.nan)
    )
    audit.update(
        {
            "state_parent_rotation_status": "replaced",
            "state_parent_rotation_reason": (
                "one_state_loaded_parent_uniquely_became_the_supported_center"
            ),
            "state_parent_rotation_selected_mode": selected_mode,
            "state_parent_rotation_selected": True,
            "state_parent_rotation_selected_center": str(selected.center),
            "state_parent_rotation_selected_outer1": str(selected.outer1),
            "state_parent_rotation_selected_outer2": str(selected.outer2),
            "state_parent_rotation_selected_raw_generative_score": selected_raw_score,
            "state_parent_rotation_selected_raw_fraction_of_production": (
                selected_raw_score / production_raw_score
                if production_raw_score > 0.0
                else math.inf
            ),
            "state_parent_rotation_selected_parent_posterior": float(
                selected.laplace_min_parent_positive_posterior
            ),
            "state_parent_rotation_selected_fold_parent_posterior": float(
                selected.laplace_fold_min_parent_positive_posterior
            ),
            "state_parent_rotation_selected_reverse_center_joint_gain_margin": float(
                selected.crossfit_joint_gain_over_baseline_reverse_center_margin
            ),
            "state_parent_rotation_selected_reverse_center_parent_posterior_margin": float(
                selected.laplace_min_parent_positive_posterior_reverse_center_margin
            ),
            "state_parent_rotation_selected_minimum_collider_separation": (
                selected_minimum_separation
            ),
            "state_parent_rotation_selected_median_collider_separation": (
                selected_median_separation
            ),
            "state_parent_rotation_selected_minimum_collider_separation_improvement": (
                selected_minimum_separation - production_minimum_separation
            ),
            "state_parent_rotation_selected_median_collider_separation_improvement": (
                selected_median_separation - production_median_separation
            ),
            "state_parent_rotation_selected_median_collider_separation_runner_margin": (
                selected_topology_runner_margin
            ),
        }
    )
    return audit, selected


def _multi_component_collapse_audit(
    *,
    group: pd.DataFrame,
    production_edges: set[tuple[str, str]],
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Select one existing star only when multi-component evidence is decisive."""

    audit: dict[str, Any] = {
        "multi_component_collapse_considered": False,
        "multi_component_collapse_selected": False,
        "multi_component_collapse_status": "not_applicable",
        "multi_component_collapse_abstention_reason": "",
        "multi_component_collapse_production_edge_count": len(production_edges),
        "multi_component_collapse_candidate_center": "",
        "multi_component_collapse_candidate_outer1": "",
        "multi_component_collapse_candidate_outer2": "",
        "multi_component_collapse_raw_generative_score": math.nan,
        "multi_component_collapse_global_runner_score_margin": math.nan,
        "multi_component_collapse_current_arm_support": math.nan,
        "multi_component_collapse_star_is_production_subset": False,
    }
    if len(production_edges) < config.multi_component_minimum_directed_edge_count:
        audit["multi_component_collapse_abstention_reason"] = (
            "production_does_not_contain_multiple_direct_stars"
        )
        return audit, None

    audit["multi_component_collapse_considered"] = True
    ordered = _ordered_candidates(group)
    if ordered.empty:
        audit["multi_component_collapse_status"] = "abstained"
        audit["multi_component_collapse_abstention_reason"] = (
            "no_crossfit_star_candidates"
        )
        return audit, None
    candidate = ordered.iloc[0]
    runner_margin = (
        float(candidate.raw_generative_score - ordered.iloc[1].raw_generative_score)
        if len(ordered) > 1
        else math.inf
    )
    candidate_edges = _star_edges(candidate)
    star_is_subset = candidate_edges.issubset(production_edges)
    audit.update(
        {
            "multi_component_collapse_candidate_center": str(candidate.center),
            "multi_component_collapse_candidate_outer1": str(candidate.outer1),
            "multi_component_collapse_candidate_outer2": str(candidate.outer2),
            "multi_component_collapse_raw_generative_score": float(
                candidate.raw_generative_score
            ),
            "multi_component_collapse_global_runner_score_margin": runner_margin,
            "multi_component_collapse_current_arm_support": int(
                candidate.current_arm_support
            ),
            "multi_component_collapse_star_is_production_subset": star_is_subset,
        }
    )
    checks = (
        (
            "candidate_is_not_a_subset_of_current_direct_arms",
            star_is_subset,
        ),
        (
            "candidate_lacks_two_current_arms",
            int(candidate.current_arm_support)
            >= config.multi_component_minimum_current_arm_support,
        ),
        (
            "raw_generative_score_below_threshold",
            float(candidate.raw_generative_score)
            >= config.multi_component_minimum_raw_generative_score,
        ),
        (
            "global_runner_margin_below_threshold",
            runner_margin
            >= config.multi_component_minimum_global_runner_score_margin,
        ),
        (
            "joint_gain_below_threshold",
            float(candidate.crossfit_joint_gain_over_baseline)
            >= config.multi_component_minimum_joint_gain,
        ),
        (
            "drop_one_gain_below_threshold",
            float(candidate.crossfit_min_drop_one_gain)
            >= config.multi_component_minimum_drop_one_gain,
        ),
        (
            "parent_posterior_below_threshold",
            float(candidate.laplace_min_parent_positive_posterior)
            >= config.multi_component_minimum_parent_positive_posterior,
        ),
        (
            "fold_parent_posterior_below_threshold",
            float(candidate.laplace_fold_min_parent_positive_posterior)
            >= config.multi_component_minimum_fold_parent_positive_posterior,
        ),
        (
            "positive_fold_support_below_threshold",
            float(candidate.laplace_both_positive_fold_support)
            >= config.multi_component_minimum_both_positive_fold_support,
        ),
        (
            "practical_fold_support_below_threshold",
            float(candidate.laplace_both_practical_fold_support)
            >= config.multi_component_minimum_both_practical_fold_support,
        ),
        (
            "arm_empirical_p_above_threshold",
            float(candidate.maximum_arm_empirical_positive_p)
            <= config.multi_component_maximum_arm_empirical_positive_p,
        ),
        (
            "outer_empirical_null_compatibility_below_threshold",
            float(candidate.outer_empirical_null_compatibility_p)
            >= config.multi_component_minimum_outer_empirical_null_compatibility_p,
        ),
    )
    failures = [reason for reason, passed in checks if not passed]
    if failures:
        audit["multi_component_collapse_status"] = "abstained"
        audit["multi_component_collapse_abstention_reason"] = ";".join(failures)
        return audit, None
    audit["multi_component_collapse_selected"] = True
    audit["multi_component_collapse_status"] = "selected"
    return audit, candidate


def _zero_edge_state_consistent_runner_audit(
    *,
    group: pd.DataFrame,
    selected_center: str,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Find a unique strong runner when the pairwise rank winner fails gates."""

    audit: dict[str, Any] = {
        "zero_edge_runner_considered": True,
        "zero_edge_runner_selected": False,
        "zero_edge_runner_status": "abstained",
        "zero_edge_runner_abstention_reason": "",
        "zero_edge_runner_eligible_candidates": 0,
        "zero_edge_runner_candidate_center": "",
        "zero_edge_runner_candidate_outer1": "",
        "zero_edge_runner_candidate_outer2": "",
        "zero_edge_runner_raw_generative_score": math.nan,
        "zero_edge_runner_eligible_score_margin": math.nan,
    }
    if not selected_center:
        audit["zero_edge_runner_abstention_reason"] = "rank_fusion_center_is_empty"
        return audit, None
    eligible = _ordered_candidates(
        group.loc[
            group["center"].astype(str).eq(selected_center)
            & group["two_parent_positive_state_wald_consistency"].astype(bool)
            & group["minimum_outer_conditional_wald_view_support"].ge(
                config.zero_edge_minimum_outer_conditional_wald_view_support
            )
            & group["laplace_min_parent_positive_posterior"].ge(
                config.zero_edge_runner_minimum_parent_positive_posterior
            )
            & group["laplace_fold_min_parent_positive_posterior"].ge(
                config.zero_edge_runner_minimum_fold_parent_positive_posterior
            )
            & group["laplace_both_positive_fold_support"].ge(
                config.zero_edge_runner_minimum_both_positive_fold_support
            )
            & group["laplace_both_practical_fold_support"].ge(
                config.zero_edge_runner_minimum_both_practical_fold_support
            )
            & group["maximum_arm_empirical_positive_p"].le(
                config.zero_edge_runner_maximum_arm_empirical_positive_p
            )
            & group["crossfit_joint_gain_over_baseline"].ge(
                config.zero_edge_runner_minimum_joint_gain
            )
            & group["crossfit_min_drop_one_gain"].ge(
                config.zero_edge_runner_minimum_drop_one_gain
            )
            & group["raw_generative_score"].ge(
                config.zero_edge_runner_minimum_raw_generative_score
            )
        ]
    )
    audit["zero_edge_runner_eligible_candidates"] = len(eligible)
    if eligible.empty:
        audit["zero_edge_runner_abstention_reason"] = (
            "no_same_center_candidate_passed_all_uncertainty_gates"
        )
        return audit, None
    candidate = eligible.iloc[0]
    runner_margin = (
        float(candidate.raw_generative_score - eligible.iloc[1].raw_generative_score)
        if len(eligible) > 1
        else math.inf
    )
    audit.update(
        {
            "zero_edge_runner_candidate_center": str(candidate.center),
            "zero_edge_runner_candidate_outer1": str(candidate.outer1),
            "zero_edge_runner_candidate_outer2": str(candidate.outer2),
            "zero_edge_runner_raw_generative_score": float(
                candidate.raw_generative_score
            ),
            "zero_edge_runner_eligible_score_margin": runner_margin,
        }
    )
    if runner_margin < config.zero_edge_runner_minimum_eligible_score_margin:
        audit["zero_edge_runner_abstention_reason"] = (
            "eligible_runner_margin_below_threshold"
        )
        return audit, None
    audit["zero_edge_runner_selected"] = True
    audit["zero_edge_runner_status"] = "selected"
    return audit, candidate


def _retained_orientation_demotion_audit(
    *,
    production: pd.Series | None,
    parent_redundancy_demoted: bool,
    replacement_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> dict[str, Any]:
    """Audit a subtractive demotion for an uncertain retained orientation."""

    base: dict[str, Any] = {
        "retained_orientation_demotion_considered": False,
        "retained_orientation_demotion_selected": False,
        "retained_orientation_demotion_status": "not_applicable",
        "retained_orientation_demotion_reason": "",
        "retained_orientation_demotion_evidence_complete": False,
        "retained_orientation_demotion_parent_posterior": math.nan,
        "retained_orientation_demotion_maximum_arm_empirical_positive_p": math.nan,
        "retained_orientation_demotion_reverse_center_posterior_margin": math.nan,
        "retained_orientation_demotion_parent_posterior_below_threshold": False,
        "retained_orientation_demotion_arm_empirical_p_above_threshold": False,
        "retained_orientation_demotion_reverse_margin_below_threshold": False,
    }
    if production is None:
        base["retained_orientation_demotion_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return base
    if not parent_redundancy_demoted:
        base["retained_orientation_demotion_status"] = "not_considered"
        base["retained_orientation_demotion_reason"] = (
            "both_production_parents_retained_state_support"
        )
        return base

    base["retained_orientation_demotion_considered"] = True
    posterior = float(production.laplace_min_parent_positive_posterior)
    arm_p = float(production.maximum_arm_empirical_positive_p)
    reverse_margin = float(
        production.laplace_min_parent_positive_posterior_reverse_center_margin
    )
    base.update(
        {
            "retained_orientation_demotion_parent_posterior": posterior,
            "retained_orientation_demotion_maximum_arm_empirical_positive_p": arm_p,
            "retained_orientation_demotion_reverse_center_posterior_margin": (
                reverse_margin
            ),
        }
    )
    evidence = {
        "parent_posterior": posterior,
        "maximum_arm_empirical_positive_p": arm_p,
        "reverse_center_posterior_margin": reverse_margin,
    }
    missing = sorted(name for name, value in evidence.items() if not np.isfinite(value))
    if missing:
        base["retained_orientation_demotion_status"] = "abstained"
        base["retained_orientation_demotion_reason"] = (
            "missing_or_nonfinite_observed_evidence:" + ",".join(missing)
        )
        return base
    base["retained_orientation_demotion_evidence_complete"] = True

    posterior_uncertain = (
        posterior
        < config.retained_orientation_demotion_minimum_parent_posterior
    )
    empirical_null_unsupported = (
        arm_p
        > config.retained_orientation_demotion_maximum_arm_empirical_positive_p
    )
    orientation_ambiguous = (
        reverse_margin
        < config.retained_orientation_demotion_maximum_reverse_center_posterior_margin
    )
    base.update(
        {
            "retained_orientation_demotion_parent_posterior_below_threshold": (
                posterior_uncertain
            ),
            "retained_orientation_demotion_arm_empirical_p_above_threshold": (
                empirical_null_unsupported
            ),
            "retained_orientation_demotion_reverse_margin_below_threshold": (
                orientation_ambiguous
            ),
        }
    )
    if replacement_selected:
        base["retained_orientation_demotion_status"] = (
            "not_needed_replacement_selected"
        )
        base["retained_orientation_demotion_reason"] = (
            "an_observed_replacement_gate_selected_a_motif"
        )
        return base
    failures: list[str] = []
    if not posterior_uncertain:
        failures.append("production_parent_posterior_not_below_threshold")
    if not empirical_null_unsupported:
        failures.append("production_arm_empirical_p_not_above_threshold")
    if not orientation_ambiguous:
        failures.append("reverse_center_posterior_margin_not_below_threshold")
    if failures:
        base["retained_orientation_demotion_status"] = "not_selected"
        base["retained_orientation_demotion_reason"] = ";".join(failures)
        return base
    base["retained_orientation_demotion_selected"] = True
    base["retained_orientation_demotion_status"] = "selected"
    base["retained_orientation_demotion_reason"] = (
        "redundancy_demoted_parent_with_unsupported_orientation"
    )
    return base


def _ambiguous_skeleton_reorientation_audit(
    *,
    group: pd.DataFrame,
    production: pd.Series | None,
    orientation_demotion_selected: bool,
    higher_priority_demotion_selected: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None]:
    """Orient an otherwise demoted three-node skeleton only when unique."""

    audit: dict[str, Any] = {
        "ambiguous_skeleton_reorientation_considered": False,
        "ambiguous_skeleton_reorientation_status": "not_applicable",
        "ambiguous_skeleton_reorientation_reason": "",
        "ambiguous_skeleton_reorientation_eligible_candidates": 0,
        "ambiguous_skeleton_reorientation_selected": False,
        "ambiguous_skeleton_reorientation_selected_center": "",
        "ambiguous_skeleton_reorientation_selected_outer1": "",
        "ambiguous_skeleton_reorientation_selected_outer2": "",
        "ambiguous_skeleton_reorientation_selected_collider_separation": math.nan,
        "ambiguous_skeleton_reorientation_collider_separation_improvement": math.nan,
        "ambiguous_skeleton_reorientation_collider_separation_runner_margin": math.nan,
        "ambiguous_skeleton_reorientation_selected_raw_generative_score": math.nan,
    }
    if production is None:
        audit["ambiguous_skeleton_reorientation_reason"] = (
            "production_is_not_an_exact_two_arm_star"
        )
        return audit, None
    if higher_priority_demotion_selected:
        audit["ambiguous_skeleton_reorientation_reason"] = (
            "higher_priority_subtractive_validation_selected"
        )
        return audit, None
    if not orientation_demotion_selected:
        audit["ambiguous_skeleton_reorientation_reason"] = (
            "generic_orientation_demotion_was_not_selected"
        )
        return audit, None

    audit["ambiguous_skeleton_reorientation_considered"] = True
    skeleton = {
        str(production.center),
        str(production.outer1),
        str(production.outer2),
    }
    same_skeleton = group.loc[
        group.apply(
            lambda row: {str(row.center), str(row.outer1), str(row.outer2)}
            == skeleton,
            axis=1,
        )
    ].copy()
    if len(same_skeleton) != 3:
        audit["ambiguous_skeleton_reorientation_status"] = "abstained"
        audit["ambiguous_skeleton_reorientation_reason"] = (
            "candidate_universe_lacks_three_exact_skeleton_orientations"
        )
        return audit, None

    production_separation = float(production.minimum_collider_separation)
    improvements = (
        same_skeleton["minimum_collider_separation"] - production_separation
    )
    eligible = _ordered_candidates(
        same_skeleton.loc[
            same_skeleton["center"].astype(str).ne(str(production.center))
            & same_skeleton["two_parent_positive_state_wald_consistency"].astype(bool)
            & same_skeleton["minimum_outer_conditional_wald_view_support"].ge(
                config.ambiguous_skeleton_minimum_outer_wald_view_support
            )
            & same_skeleton["crossfit_joint_gain_over_baseline"].ge(
                config.ambiguous_skeleton_minimum_joint_gain
            )
            & same_skeleton["raw_generative_score"].ge(
                config.ambiguous_skeleton_minimum_raw_generative_score
            )
            & same_skeleton["laplace_min_parent_positive_posterior"].ge(
                config.ambiguous_skeleton_minimum_parent_posterior
            )
            & same_skeleton["laplace_both_positive_fold_support"].ge(
                config.ambiguous_skeleton_minimum_both_positive_fold_support
            )
            & same_skeleton["laplace_both_practical_fold_support"].ge(
                config.ambiguous_skeleton_minimum_both_practical_fold_support
            )
            & improvements.ge(
                config.ambiguous_skeleton_minimum_collider_separation_improvement
            )
        ]
    )
    audit["ambiguous_skeleton_reorientation_eligible_candidates"] = int(
        len(eligible)
    )
    if len(eligible) != 1:
        audit["ambiguous_skeleton_reorientation_status"] = "abstained"
        audit["ambiguous_skeleton_reorientation_reason"] = (
            "eligible_same_skeleton_orientation_count_is_not_exactly_one"
        )
        return audit, None

    selected = eligible.iloc[0]
    selected_separation = float(selected.minimum_collider_separation)
    other_orientations = same_skeleton.drop(index=selected.name)
    separation_runner_margin = float(
        selected_separation
        - other_orientations["minimum_collider_separation"].max()
    )
    audit.update(
        {
            "ambiguous_skeleton_reorientation_selected_center": str(
                selected.center
            ),
            "ambiguous_skeleton_reorientation_selected_outer1": str(
                selected.outer1
            ),
            "ambiguous_skeleton_reorientation_selected_outer2": str(
                selected.outer2
            ),
            "ambiguous_skeleton_reorientation_selected_collider_separation": (
                selected_separation
            ),
            "ambiguous_skeleton_reorientation_collider_separation_improvement": (
                selected_separation - production_separation
            ),
            "ambiguous_skeleton_reorientation_collider_separation_runner_margin": (
                separation_runner_margin
            ),
            "ambiguous_skeleton_reorientation_selected_raw_generative_score": float(
                selected.raw_generative_score
            ),
        }
    )
    if (
        separation_runner_margin
        < config.ambiguous_skeleton_minimum_collider_separation_runner_margin
    ):
        audit["ambiguous_skeleton_reorientation_status"] = "abstained"
        audit["ambiguous_skeleton_reorientation_reason"] = (
            "best_orientation_lacks_collider_separation_runner_margin"
        )
        return audit, None

    audit["ambiguous_skeleton_reorientation_status"] = "selected"
    audit["ambiguous_skeleton_reorientation_reason"] = (
        "one_state_consistent_orientation_uniquely_improved_the_same_skeleton"
    )
    audit["ambiguous_skeleton_reorientation_selected"] = True
    return audit, selected


def select_crossfit_hierarchical_collider_replacements(
    *,
    candidates: pd.DataFrame,
    state_membership: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig | None = None,
) -> pd.DataFrame:
    """Apply frozen raw, uncertainty, and zero-edge gates to observed candidates."""

    cfg = config or CrossfitHierarchicalColliderConfig()
    cfg.validate()
    _assert_observed_only(candidates, "candidates")
    _assert_observed_only(state_membership, "state_membership")
    _assert_observed_only(current_direct_candidates, "current_direct_candidates")
    candidate_required = {
        "source_assay",
        "center",
        "outer1",
        "outer2",
        "crossfit_joint_gain_over_baseline",
        "crossfit_min_drop_one_gain",
        "crossfit_joint_gain_fold_min",
        "laplace_mean_parent_coefficient_min",
        "laplace_min_parent_positive_posterior",
        "laplace_fold_min_parent_positive_posterior",
        "laplace_both_positive_fold_support",
        "laplace_both_practical_fold_support",
        "center_detection_rate",
        "minimum_parent_detection_rate",
        "crossfit_joint_gain_over_baseline_reverse_center_margin",
        "laplace_min_parent_positive_posterior_reverse_center_margin",
        "residual_star_separation_reverse_center_margin",
        "maximum_arm_empirical_positive_p",
        "outer_empirical_null_compatibility_p",
        "fold_maximum_outer_absolute_correlation",
        "residual_star_separation",
        "minimum_collider_separation",
        "minimum_outer_conditional_wald_view_support",
        "two_parent_positive_state_wald_consistency",
    }
    _require_columns(candidates, candidate_required, "candidates")
    _require_strict_boolean_column(
        candidates,
        "two_parent_positive_state_wald_consistency",
        "candidates",
    )
    candidate_identity_columns = {"source_assay", "center", "outer1", "outer2"}
    candidate_boolean_columns = {"two_parent_positive_state_wald_consistency"}
    candidate_numeric_columns = sorted(
        candidate_required - candidate_identity_columns - candidate_boolean_columns
    )
    numeric_candidates = candidates[candidate_numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    nonnumeric = candidates[candidate_numeric_columns].notna() & numeric_candidates.isna()
    if nonnumeric.any().any():
        invalid_columns = sorted(nonnumeric.columns[nonnumeric.any()].astype(str))
        raise ValueError(f"candidates contains nonnumeric evidence: {invalid_columns}")
    infinite = np.isinf(numeric_candidates.to_numpy(dtype=float))
    if infinite.any():
        invalid_columns = sorted(
            numeric_candidates.columns[np.any(infinite, axis=0)].astype(str)
        )
        raise ValueError(f"candidates contains infinite evidence: {invalid_columns}")
    _state_membership_map(state_membership)
    keys = ["source_assay", "center", "outer1", "outer2"]
    if candidates.duplicated(keys).any():
        raise ValueError("candidates contains duplicate star identities")
    frame = candidates.copy()
    frame[candidate_numeric_columns] = numeric_candidates
    frame["raw_generative_score"] = _raw_score(frame)
    directed = _current_directed_edges(current_direct_candidates)
    production = {
        str(assay): {
            (str(row.parent), str(row.child)) for row in group.itertuples(index=False)
        }
        for assay, group in directed.groupby("source_assay", sort=True)
    }
    observed_refinement_actions: dict[str, str] = {}
    if (
        not current_direct_candidates.empty
        and "observed_refinement_action" in current_direct_candidates.columns
    ):
        for assay, assay_rows in current_direct_candidates.groupby(
            "source_assay", sort=True
        ):
            actions = {
                str(value).strip()
                for value in assay_rows["observed_refinement_action"].fillna("")
                if str(value).strip()
            }
            if len(actions) == 1:
                observed_refinement_actions[str(assay)] = next(iter(actions))
    centrality_provenance = _centrality_provenance_by_assay(
        current_direct_candidates
    )
    dual_anchor_provenance = _dual_anchor_provenance_by_assay(
        current_direct_candidates
    )
    retained_state = _redundancy_filtered_state_members(state_membership)
    decisions: list[dict[str, Any]] = []
    for assay, group in frame.groupby("source_assay", sort=True):
        assay = str(assay)
        group = group.copy()
        production_edges = production.get(assay, set())
        matches = group.loc[
            [_star_edges(row) == production_edges for row in group.itertuples(index=False)]
        ]
        production_exact = bool(len(production_edges) == 2 and len(matches) == 1)
        production_row = matches.iloc[0] if production_exact else None
        parent_redundancy_demoted = any(
            parent not in retained_state for parent, _ in production_edges
        )
        chosen: pd.Series | None = None
        action = "retain_current_direct_motif"
        demote_production_motif = False
        runner_margin = math.nan
        candidate_vs_production = math.nan
        tier1u_considered = False
        tier1u_selected = False
        tier1u_failures: list[str] = []
        tier1u_candidate: pd.Series | None = None
        tier1r_considered = False
        tier1r_selected = False
        tier1r_failures: list[str] = []
        tier1r_candidate: pd.Series | None = None
        tier1r_global_runner_margin = math.nan
        tier1r_score_improvement = math.nan
        tier1r_laplace_parent_posterior_improvement = math.nan
        tier1r_laplace_fold_parent_posterior_improvement = math.nan
        multicenter_validation_audit: dict[str, Any] = {}
        multicenter_validation_demote = False
        centrality_validation_audit: dict[str, Any] = {}
        centrality_validation_demote = False
        dual_anchor_validation_audit: dict[str, Any] = {}
        neighbor_star_validation_audit: dict[str, Any] = {}
        same_center_substitution_audit: dict[str, Any] = {}
        state_center_sibling_rotation_audit: dict[str, Any] = {}
        state_parent_rotation_audit: dict[str, Any] = {}
        zero_edge_audit: dict[str, Any] = {
            "zero_edge_rank_fusion_model_version": ZERO_EDGE_RANK_FUSION_MODEL_VERSION,
            "zero_edge_rank_fusion_model_sha256": _zero_edge_rank_fusion_model_sha256(),
            "zero_edge_rank_fusion_runtime_refit": False,
            "zero_edge_rank_fusion_status": "not_applicable_nonzero_direct_motif",
            "zero_edge_rank_fusion_abstention_reason": "",
            "zero_edge_rank_fusion_selected_center": "",
            "zero_edge_rank_fusion_selected_outer1": "",
            "zero_edge_rank_fusion_selected_outer2": "",
            "zero_edge_rank_fusion_arm1_logit": math.nan,
            "zero_edge_rank_fusion_arm2_logit": math.nan,
            "zero_edge_rank_fusion_arm1_probability": math.nan,
            "zero_edge_rank_fusion_arm2_probability": math.nan,
            "zero_edge_rank_fusion_selected_center_score": math.nan,
            "zero_edge_rank_fusion_runner_center_score": math.nan,
            "zero_edge_rank_fusion_center_score_margin": math.nan,
            "zero_edge_rank_fusion_selected_state_wald_consistency": False,
            "zero_edge_rank_fusion_selected_outer_wald_view_support": math.nan,
            "zero_edge_rank_fusion_selected_practical_fold_support": math.nan,
            "zero_edge_rank_fusion_practical_fold_support_gate": False,
            "zero_edge_rank_fusion_arm1_evidence_json": "",
            "zero_edge_rank_fusion_arm2_evidence_json": "",
            "zero_edge_rank_fusion_all_edge_scores_json": "",
            "zero_edge_rank_fusion_all_center_scores_json": "",
        }
        zero_edge_runner_audit: dict[str, Any] = {
            "zero_edge_runner_considered": False,
            "zero_edge_runner_selected": False,
            "zero_edge_runner_status": "not_applicable",
            "zero_edge_runner_abstention_reason": "",
            "zero_edge_runner_eligible_candidates": 0,
            "zero_edge_runner_candidate_center": "",
            "zero_edge_runner_candidate_outer1": "",
            "zero_edge_runner_candidate_outer2": "",
            "zero_edge_runner_raw_generative_score": math.nan,
            "zero_edge_runner_eligible_score_margin": math.nan,
        }
        multi_component_audit, multi_component_candidate = (
            _multi_component_collapse_audit(
                group=group,
                production_edges=production_edges,
                config=cfg,
            )
        )
        if multi_component_candidate is not None:
            chosen = multi_component_candidate
            action = MULTI_COMPONENT_COLLAPSE_ACTION
            runner_margin = float(
                multi_component_audit[
                    "multi_component_collapse_global_runner_score_margin"
                ]
            )

        eligible1 = _ordered_candidates(
            group.loc[
                group["laplace_min_parent_positive_posterior"].ge(
                    cfg.tier1_minimum_parent_positive_posterior
                )
                & group["crossfit_joint_gain_over_baseline"].gt(
                    cfg.tier1_minimum_joint_gain_exclusive
                )
                & group["crossfit_min_drop_one_gain"].gt(
                    cfg.tier1_minimum_drop_one_gain_exclusive
                )
                & group["laplace_both_positive_fold_support"].ge(
                    cfg.tier1_minimum_both_positive_fold_support
                )
                & group["crossfit_joint_gain_over_baseline_reverse_center_margin"].gt(
                    cfg.tier1_minimum_reverse_center_gain_margin_exclusive
                )
                & group["maximum_arm_empirical_positive_p"].le(
                    cfg.tier1_maximum_arm_empirical_positive_p
                )
            ]
        )
        if chosen is None and len(eligible1):
            top = eligible1.iloc[0]
            runner_margin = float(
                top.raw_generative_score - eligible1.iloc[1].raw_generative_score
            ) if len(eligible1) > 1 else math.inf
            unique = runner_margin >= cfg.tier1_minimum_eligible_runner_score_margin
            if production_exact:
                candidate_vs_production = float(
                    top.raw_generative_score - production_row.raw_generative_score
                )
                if (
                    unique
                    and _star_edges(top) != production_edges
                    and bool(top.two_parent_positive_state_wald_consistency)
                    and candidate_vs_production
                    >= cfg.tier1_minimum_exact_production_score_improvement
                ):
                    chosen = top
                    action = "tier1_exact_production_disagreement"
                elif (
                    _star_edges(top) != production_edges
                    and not bool(top.two_parent_positive_state_wald_consistency)
                ):
                    tier1u_considered = True
                    tier1u_candidate = top
                    tier1u_failures = _tier1u_gate_failures(
                        candidate=top,
                        runner_margin=runner_margin,
                        score_improvement=candidate_vs_production,
                        config=cfg,
                    )
                    if not tier1u_failures:
                        chosen = top
                        tier1u_selected = True
                        action = "tier1u_fold_stable_exact_production_disagreement"
            elif (
                unique
                and top.raw_generative_score >= cfg.tier1_minimum_nonexact_score
                and (
                    bool(top.two_parent_positive_state_wald_consistency)
                    or top.minimum_collider_separation > 0.0
                )
            ):
                chosen = top
                action = "tier1_nonexact_production_completion"

        if (
            chosen is None
            and production_exact
            and parent_redundancy_demoted
            and len(group) >= 2
        ):
            globally_ordered = _ordered_candidates(group)
            residual_top = globally_ordered.iloc[0]
            if _star_edges(residual_top) != production_edges:
                tier1r_considered = True
                tier1r_candidate = residual_top
                tier1r_global_runner_margin = float(
                    residual_top.raw_generative_score
                    - globally_ordered.iloc[1].raw_generative_score
                )
                tier1r_score_improvement = float(
                    residual_top.raw_generative_score
                    - production_row.raw_generative_score
                )
                tier1r_laplace_parent_posterior_improvement = float(
                    residual_top.laplace_min_parent_positive_posterior
                    - production_row.laplace_min_parent_positive_posterior
                )
                tier1r_laplace_fold_parent_posterior_improvement = float(
                    residual_top.laplace_fold_min_parent_positive_posterior
                    - production_row.laplace_fold_min_parent_positive_posterior
                )
                tier1r_failures = _tier1r_gate_failures(
                    candidate=residual_top,
                    production_candidate=production_row,
                    global_runner_margin=tier1r_global_runner_margin,
                    score_improvement=tier1r_score_improvement,
                    config=cfg,
                )
                if not tier1r_failures:
                    chosen = residual_top
                    tier1r_selected = True
                    action = RESIDUAL_FIRST_REPLACEMENT_ACTION
                    runner_margin = tier1r_global_runner_margin
                    candidate_vs_production = tier1r_score_improvement

        (
            multicenter_validation_audit,
            multicenter_validation_candidate,
            multicenter_validation_demote,
        ) = _weak_multicenter_refinement_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            observed_refinement_action=observed_refinement_actions.get(assay, ""),
            higher_priority_replacement_selected=chosen is not None,
            config=cfg,
        )
        if multicenter_validation_candidate is not None:
            chosen = multicenter_validation_candidate
            action = WEAK_MULTICENTER_REPLACEMENT_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        (
            centrality_validation_audit,
            centrality_validation_candidate,
            centrality_validation_demote,
        ) = _centrality_completion_validation_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            observed_refinement_action=observed_refinement_actions.get(assay, ""),
            provenance=centrality_provenance.get(assay),
            higher_priority_replacement_selected=chosen is not None,
            config=cfg,
        )
        if centrality_validation_candidate is not None:
            chosen = centrality_validation_candidate
            if (
                centrality_validation_audit["centrality_validation_status"]
                == "constrained_to_original_skeleton"
            ):
                action = CENTRALITY_SAME_SKELETON_REPLACEMENT_ACTION
            else:
                action = CENTRALITY_PRECOMPLETION_REVERSION_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        (
            dual_anchor_validation_audit,
            dual_anchor_validation_candidate,
        ) = _dual_anchor_completion_validation_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            observed_refinement_action=observed_refinement_actions.get(assay, ""),
            provenance=dual_anchor_provenance.get(assay),
            retained_state=retained_state,
            higher_priority_replacement_selected=chosen is not None,
            config=cfg,
        )
        if dual_anchor_validation_candidate is not None:
            chosen = dual_anchor_validation_candidate
            action = DUAL_ANCHOR_RESIDUAL_REPLACEMENT_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        (
            neighbor_star_validation_audit,
            neighbor_star_validation_candidate,
        ) = _state_inconsistent_neighbor_star_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            higher_priority_replacement_selected=chosen is not None,
            higher_priority_demotion_selected=(
                multicenter_validation_demote or centrality_validation_demote
            ),
            config=cfg,
        )
        if neighbor_star_validation_candidate is not None:
            chosen = neighbor_star_validation_candidate
            action = STATE_INCONSISTENT_NEIGHBOR_REPLACEMENT_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        (
            same_center_substitution_audit,
            same_center_substitution_candidate,
        ) = _state_inconsistent_same_center_substitution_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            retained_state=retained_state,
            higher_priority_replacement_selected=chosen is not None,
            higher_priority_demotion_selected=(
                multicenter_validation_demote or centrality_validation_demote
            ),
            config=cfg,
        )
        if same_center_substitution_candidate is not None:
            chosen = same_center_substitution_candidate
            action = STATE_INCONSISTENT_SAME_CENTER_SUBSTITUTION_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        (
            state_center_sibling_rotation_audit,
            state_center_sibling_rotation_candidate,
        ) = _state_center_sibling_rotation_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            retained_state=retained_state,
            higher_priority_replacement_selected=chosen is not None,
            higher_priority_demotion_selected=(
                multicenter_validation_demote or centrality_validation_demote
            ),
            config=cfg,
        )
        if state_center_sibling_rotation_candidate is not None:
            chosen = state_center_sibling_rotation_candidate
            action = STATE_CENTER_SIBLING_ROTATION_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        (
            state_parent_rotation_audit,
            state_parent_rotation_candidate,
        ) = _state_loaded_parent_rotation_audit(
            group=group,
            production=production_row,
            production_edges=production_edges,
            retained_state=retained_state,
            higher_priority_replacement_selected=chosen is not None,
            higher_priority_demotion_selected=(
                multicenter_validation_demote or centrality_validation_demote
            ),
            config=cfg,
        )
        if state_parent_rotation_candidate is not None:
            chosen = state_parent_rotation_candidate
            action = STATE_LOADED_PARENT_ROTATION_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
        eligible2 = pd.DataFrame()
        if chosen is None and not production_exact:
            eligible2 = _ordered_candidates(
                group.loc[
                    group["laplace_min_parent_positive_posterior"].ge(
                        cfg.tier2_minimum_parent_positive_posterior
                    )
                    & group["crossfit_joint_gain_over_baseline"].ge(
                        cfg.tier2_minimum_joint_gain
                    )
                    & group["crossfit_min_drop_one_gain"].ge(
                        cfg.tier2_minimum_drop_one_gain
                    )
                    & group["laplace_both_positive_fold_support"].ge(
                        cfg.tier2_minimum_both_positive_fold_support
                    )
                    & group["crossfit_joint_gain_over_baseline_reverse_center_margin"].ge(
                        cfg.tier2_minimum_reverse_center_gain_margin
                    )
                    & group["maximum_arm_empirical_positive_p"].le(
                        cfg.tier2_maximum_arm_empirical_positive_p
                    )
                    & group["outer_empirical_null_compatibility_p"].ge(
                        cfg.tier2_minimum_outer_empirical_null_compatibility_p
                    )
                    & group["minimum_collider_separation"].ge(
                        cfg.tier2_minimum_collider_separation
                    )
                    & group["raw_generative_score"].ge(cfg.tier2_minimum_score)
                ]
            )
            if len(eligible2):
                top = eligible2.iloc[0]
                runner_margin = float(
                    top.raw_generative_score
                    - group.drop(index=top.name)["raw_generative_score"].max()
                )
                if runner_margin >= cfg.tier2_minimum_all_candidate_score_margin:
                    chosen = top
                    action = "tier2_positive_collider_incomplete_rescue"

        eligible3 = pd.DataFrame()
        if (
            chosen is None
            and production_exact
            and parent_redundancy_demoted
            and not bool(production_row.two_parent_positive_state_wald_consistency)
            and production_row.maximum_arm_empirical_positive_p
            > cfg.tier3_production_arm_p_exclusive
        ):
            parents = {str(production_row.outer1), str(production_row.outer2)}
            same_center_one_parent = group.apply(
                lambda row: str(row.center) == str(production_row.center)
                and len({str(row.outer1), str(row.outer2)} & parents) == 1,
                axis=1,
            )
            eligible3 = group.loc[
                same_center_one_parent
                & group["two_parent_positive_state_wald_consistency"].astype(bool)
                & group["laplace_min_parent_positive_posterior"].ge(
                    cfg.tier3_minimum_parent_positive_posterior
                )
                & group["crossfit_joint_gain_over_baseline"].ge(
                    cfg.tier3_minimum_joint_gain
                )
                & group["crossfit_min_drop_one_gain"].ge(
                    cfg.tier3_minimum_drop_one_gain
                )
                & group["laplace_both_positive_fold_support"].ge(
                    cfg.tier3_minimum_both_positive_fold_support
                )
                & group["maximum_arm_empirical_positive_p"].le(
                    cfg.tier3_maximum_arm_empirical_positive_p
                )
                & group["outer_empirical_null_compatibility_p"].ge(
                    cfg.tier3_minimum_outer_empirical_null_compatibility_p
                )
                & group["crossfit_joint_gain_over_baseline"].ge(
                    production_row.crossfit_joint_gain_over_baseline
                    - cfg.tier3_maximum_joint_gain_deficit_from_production
                )
                & group["raw_generative_score"].ge(
                    production_row.raw_generative_score
                    - cfg.tier3_maximum_score_deficit_from_production
                )
            ]
            if len(eligible3) == 1:
                chosen = eligible3.iloc[0]
                action = "tier3_state_consistent_one_parent_swap"
                candidate_vs_production = float(
                    chosen.raw_generative_score - production_row.raw_generative_score
                )

        if not production_edges:
            zero_edge_audit, zero_edge_candidate = _zero_edge_rank_fusion_audit(group)
            if chosen is not None:
                zero_edge_audit["zero_edge_rank_fusion_status"] = (
                    "not_needed_higher_tier_selected"
                )
                zero_edge_audit["zero_edge_rank_fusion_abstention_reason"] = (
                    "ordinary_observed_gate_selected_a_motif"
                )
            elif zero_edge_candidate is not None:
                zero_gate_failures: list[str] = []
                if not bool(
                    zero_edge_audit[
                        "zero_edge_rank_fusion_selected_state_wald_consistency"
                    ]
                ):
                    zero_gate_failures.append(
                        "selected_star_failed_state_wald_consistency"
                    )
                outer_wald_support = float(
                    zero_edge_audit[
                        "zero_edge_rank_fusion_selected_outer_wald_view_support"
                    ]
                )
                if (
                    not np.isfinite(outer_wald_support)
                    or outer_wald_support
                    < cfg.zero_edge_minimum_outer_conditional_wald_view_support
                ):
                    zero_gate_failures.append(
                        "selected_star_lacked_outer_wald_view_support"
                    )
                practical_fold_support = float(
                    zero_edge_audit[
                        "zero_edge_rank_fusion_selected_practical_fold_support"
                    ]
                )
                practical_support_gate = bool(
                    np.isfinite(practical_fold_support)
                    and practical_fold_support
                    >= cfg.zero_edge_minimum_selected_practical_fold_support
                )
                zero_edge_audit[
                    "zero_edge_rank_fusion_practical_fold_support_gate"
                ] = practical_support_gate
                if not practical_support_gate:
                    zero_gate_failures.append(
                        "selected_star_lacked_all_fold_practical_support"
                    )
                if float(
                    zero_edge_audit["zero_edge_rank_fusion_center_score_margin"]
                ) <= cfg.zero_edge_minimum_center_score_margin_exclusive:
                    zero_gate_failures.append("center_score_margin_not_positive")
                if zero_gate_failures:
                    zero_edge_runner_audit, zero_edge_runner = (
                        _zero_edge_state_consistent_runner_audit(
                            group=group,
                            selected_center=str(
                                zero_edge_audit[
                                    "zero_edge_rank_fusion_selected_center"
                                ]
                            ),
                            config=cfg,
                        )
                    )
                    zero_edge_audit["zero_edge_rank_fusion_status"] = "abstained"
                    zero_edge_audit[
                        "zero_edge_rank_fusion_abstention_reason"
                    ] = ";".join(zero_gate_failures)
                    if zero_edge_runner is not None:
                        chosen = zero_edge_runner
                        action = ZERO_EDGE_STATE_CONSISTENT_RUNNER_ACTION
                        runner_margin = float(
                            zero_edge_runner_audit[
                                "zero_edge_runner_eligible_score_margin"
                            ]
                        )
                        zero_edge_audit["zero_edge_rank_fusion_status"] = (
                            "top_abstained_state_consistent_runner_selected"
                        )
                else:
                    chosen = zero_edge_candidate
                    action = "zero_edge_state_gated_rank_fusion_completion"
                    runner_margin = float(
                        zero_edge_audit["zero_edge_rank_fusion_center_score_margin"]
                    )
                    zero_edge_audit["zero_edge_rank_fusion_status"] = "selected"

        orientation_demotion_audit = _retained_orientation_demotion_audit(
            production=production_row,
            parent_redundancy_demoted=parent_redundancy_demoted,
            replacement_selected=chosen is not None,
            config=cfg,
        )
        (
            ambiguous_skeleton_reorientation_audit,
            ambiguous_skeleton_reorientation_candidate,
        ) = _ambiguous_skeleton_reorientation_audit(
            group=group,
            production=production_row,
            orientation_demotion_selected=bool(
                orientation_demotion_audit[
                    "retained_orientation_demotion_selected"
                ]
            ),
            higher_priority_demotion_selected=(
                multicenter_validation_demote or centrality_validation_demote
            ),
            config=cfg,
        )
        if ambiguous_skeleton_reorientation_candidate is not None:
            chosen = ambiguous_skeleton_reorientation_candidate
            action = AMBIGUOUS_SKELETON_REORIENTATION_ACTION
            candidate_vs_production = float(
                chosen.raw_generative_score - production_row.raw_generative_score
            )
            other_alternatives = group.drop(index=chosen.name)
            runner_margin = (
                float(
                    chosen.raw_generative_score
                    - other_alternatives.raw_generative_score.max()
                )
                if len(other_alternatives)
                else math.inf
            )
            orientation_demotion_audit[
                "retained_orientation_demotion_selected"
            ] = False
            orientation_demotion_audit[
                "retained_orientation_demotion_status"
            ] = "replaced_by_same_skeleton_residual_orientation"
            orientation_demotion_audit[
                "retained_orientation_demotion_reason"
            ] = (
                "unique_same_skeleton_orientation_resolved_the_ambiguity"
            )
        if multicenter_validation_demote:
            demote_production_motif = True
            action = WEAK_MULTICENTER_DEMOTION_ACTION
            orientation_demotion_audit[
                "retained_orientation_demotion_selected"
            ] = False
            orientation_demotion_audit[
                "retained_orientation_demotion_status"
            ] = "not_needed_weak_multicenter_demotion_selected"
            orientation_demotion_audit[
                "retained_orientation_demotion_reason"
            ] = "weak_multicenter_validation_selected_a_subtractive_demotion"
        elif centrality_validation_demote:
            demote_production_motif = True
            action = CENTRALITY_UNRECOVERABLE_DEMOTION_ACTION
            orientation_demotion_audit[
                "retained_orientation_demotion_selected"
            ] = False
            orientation_demotion_audit[
                "retained_orientation_demotion_status"
            ] = "not_needed_weak_centrality_demotion_selected"
            orientation_demotion_audit[
                "retained_orientation_demotion_reason"
            ] = "weak_centrality_validation_selected_a_subtractive_demotion"
        elif bool(
            orientation_demotion_audit["retained_orientation_demotion_selected"]
        ):
            demote_production_motif = True
            action = RETAINED_ORIENTATION_DEMOTION_ACTION

        decisions.append(
            {
                "source_assay": assay,
                "action": action,
                "selected_replacement": chosen is not None,
                "demote_production_motif": demote_production_motif,
                "production_exact_two_arm_star": production_exact,
                "production_parent_redundancy_demoted": parent_redundancy_demoted,
                "production_center": str(production_row.center)
                if production_exact
                else "",
                "production_outer1": str(production_row.outer1)
                if production_exact
                else "",
                "production_outer2": str(production_row.outer2)
                if production_exact
                else "",
                "selected_center": str(chosen.center) if chosen is not None else "",
                "selected_outer1": str(chosen.outer1) if chosen is not None else "",
                "selected_outer2": str(chosen.outer2) if chosen is not None else "",
                "selected_raw_generative_score": float(chosen.raw_generative_score)
                if chosen is not None
                else math.nan,
                "selection_runner_margin": runner_margin,
                "selected_score_minus_production_score": candidate_vs_production,
                "tier1_eligible_candidates": len(eligible1),
                "tier1u_considered": tier1u_considered,
                "tier1u_selected": tier1u_selected,
                "tier1u_abstention_reason": ";".join(tier1u_failures),
                "tier1u_candidate_center": str(tier1u_candidate.center)
                if tier1u_candidate is not None
                else "",
                "tier1u_candidate_outer1": str(tier1u_candidate.outer1)
                if tier1u_candidate is not None
                else "",
                "tier1u_candidate_outer2": str(tier1u_candidate.outer2)
                if tier1u_candidate is not None
                else "",
                "tier1u_crossfit_joint_gain_fold_min": float(
                    tier1u_candidate.crossfit_joint_gain_fold_min
                )
                if tier1u_candidate is not None
                else math.nan,
                "tier1u_laplace_fold_min_parent_positive_posterior": float(
                    tier1u_candidate.laplace_fold_min_parent_positive_posterior
                )
                if tier1u_candidate is not None
                else math.nan,
                "tier1u_both_positive_fold_support": float(
                    tier1u_candidate.laplace_both_positive_fold_support
                )
                if tier1u_candidate is not None
                else math.nan,
                "tier1u_both_practical_fold_support": float(
                    tier1u_candidate.laplace_both_practical_fold_support
                )
                if tier1u_candidate is not None
                else math.nan,
                "tier1u_maximum_arm_empirical_positive_p": float(
                    tier1u_candidate.maximum_arm_empirical_positive_p
                )
                if tier1u_candidate is not None
                else math.nan,
                "tier1u_outer_empirical_null_compatibility_p": float(
                    tier1u_candidate.outer_empirical_null_compatibility_p
                )
                if tier1u_candidate is not None
                else math.nan,
                "tier1r_considered": tier1r_considered,
                "tier1r_selected": tier1r_selected,
                "tier1r_abstention_reason": ";".join(tier1r_failures),
                "tier1r_candidate_center": str(tier1r_candidate.center)
                if tier1r_candidate is not None
                else "",
                "tier1r_candidate_outer1": str(tier1r_candidate.outer1)
                if tier1r_candidate is not None
                else "",
                "tier1r_candidate_outer2": str(tier1r_candidate.outer2)
                if tier1r_candidate is not None
                else "",
                "tier1r_global_runner_score_margin": tier1r_global_runner_margin,
                "tier1r_score_minus_production_score": tier1r_score_improvement,
                "tier1r_laplace_parent_posterior_improvement": (
                    tier1r_laplace_parent_posterior_improvement
                ),
                "tier1r_laplace_fold_parent_posterior_improvement": (
                    tier1r_laplace_fold_parent_posterior_improvement
                ),
                "tier1r_crossfit_joint_gain_fold_min": float(
                    tier1r_candidate.crossfit_joint_gain_fold_min
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_crossfit_min_drop_one_gain": float(
                    tier1r_candidate.crossfit_min_drop_one_gain
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_laplace_min_parent_positive_posterior": float(
                    tier1r_candidate.laplace_min_parent_positive_posterior
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_laplace_fold_min_parent_positive_posterior": float(
                    tier1r_candidate.laplace_fold_min_parent_positive_posterior
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_both_positive_fold_support": float(
                    tier1r_candidate.laplace_both_positive_fold_support
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_both_practical_fold_support": float(
                    tier1r_candidate.laplace_both_practical_fold_support
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_maximum_arm_empirical_positive_p": float(
                    tier1r_candidate.maximum_arm_empirical_positive_p
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier1r_outer_empirical_null_compatibility_p": float(
                    tier1r_candidate.outer_empirical_null_compatibility_p
                )
                if tier1r_candidate is not None
                else math.nan,
                "tier2_eligible_candidates": len(eligible2),
                "tier3_eligible_candidates": len(eligible3),
                **multicenter_validation_audit,
                **centrality_validation_audit,
                **dual_anchor_validation_audit,
                **neighbor_star_validation_audit,
                **same_center_substitution_audit,
                **state_center_sibling_rotation_audit,
                **state_parent_rotation_audit,
                **ambiguous_skeleton_reorientation_audit,
                **multi_component_audit,
                **orientation_demotion_audit,
                **zero_edge_audit,
                **zero_edge_runner_audit,
            }
        )
    return pd.DataFrame(decisions).sort_values("source_assay").reset_index(drop=True)


def _annotate_zero_edge_spoke_consensus(
    *,
    decisions: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    current_demoted_or_unresolved: pd.DataFrame,
    candidates: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig,
) -> pd.DataFrame:
    """Resolve one strict three-spoke zero-production component by arm consensus."""

    for source, frame in (
        ("decisions", decisions),
        ("current_direct_candidates", current_direct_candidates),
        ("current_demoted_or_unresolved", current_demoted_or_unresolved),
        ("candidates", candidates),
    ):
        _assert_observed_only(frame, source)
    _require_columns(
        decisions,
        {
            "source_assay",
            "action",
            "selected_replacement",
            "demote_production_motif",
            "production_exact_two_arm_star",
        },
        "decisions",
    )
    _require_strict_boolean_column(decisions, "selected_replacement", "decisions")
    _require_strict_boolean_column(decisions, "demote_production_motif", "decisions")
    _require_strict_boolean_column(
        decisions,
        "production_exact_two_arm_star",
        "decisions",
    )
    _require_columns(
        candidates,
        {
            "source_assay",
            "center",
            "outer1",
            "outer2",
            "raw_generative_score",
            "crossfit_joint_gain_over_baseline",
            "crossfit_min_drop_one_gain",
            "laplace_min_parent_positive_posterior",
            "laplace_fold_min_parent_positive_posterior",
            "laplace_both_positive_fold_support",
            "laplace_both_practical_fold_support",
            "maximum_arm_empirical_positive_p",
            "minimum_outer_conditional_wald_view_support",
            "two_parent_positive_state_wald_consistency",
        },
        "candidates",
    )
    _require_strict_boolean_column(
        candidates,
        "two_parent_positive_state_wald_consistency",
        "candidates",
    )
    if not current_direct_candidates.empty:
        _require_columns(
            current_direct_candidates,
            {"source_assay"},
            "current_direct_candidates",
        )
    if not current_demoted_or_unresolved.empty:
        _require_columns(
            current_demoted_or_unresolved,
            {
                "source_assay",
                "node1",
                "node2",
                "edge_type",
                "component_id",
                "was_base_selected",
                "was_rescue_selected",
                "mean_empirical_p_positive",
                "mean_pooled_log_odds",
            },
            "current_demoted_or_unresolved",
        )
        _require_strict_boolean_column(
            current_demoted_or_unresolved,
            "was_base_selected",
            "current_demoted_or_unresolved",
        )
        _require_strict_boolean_column(
            current_demoted_or_unresolved,
            "was_rescue_selected",
            "current_demoted_or_unresolved",
        )

    output = decisions.copy()
    defaults: dict[str, Any] = {
        "zero_edge_spoke_consensus_considered": False,
        "zero_edge_spoke_consensus_status": "not_applicable",
        "zero_edge_spoke_consensus_reason": "",
        "zero_edge_spoke_consensus_prior_action": "",
        "zero_edge_spoke_consensus_unresolved_spokes": 0,
        "zero_edge_spoke_consensus_base_spokes": 0,
        "zero_edge_spoke_consensus_all_rescue_selected": False,
        "zero_edge_spoke_consensus_shared_center": "",
        "zero_edge_spoke_consensus_second_arm_log_odds": math.nan,
        "zero_edge_spoke_consensus_second_vs_third_log_odds_margin": math.nan,
        "zero_edge_spoke_consensus_maximum_top_arm_empirical_positive_p": math.nan,
        "zero_edge_spoke_consensus_candidate_universe_complete": False,
        "zero_edge_spoke_consensus_candidate_available": False,
        "zero_edge_spoke_consensus_candidate_passed": False,
        "zero_edge_spoke_consensus_selected": False,
        "zero_edge_spoke_consensus_selected_center": "",
        "zero_edge_spoke_consensus_selected_outer1": "",
        "zero_edge_spoke_consensus_selected_outer2": "",
        "zero_edge_spoke_consensus_selected_raw_generative_score": math.nan,
        "zero_edge_spoke_consensus_selected_joint_gain": math.nan,
        "zero_edge_spoke_consensus_selected_drop_one_gain": math.nan,
        "zero_edge_spoke_consensus_selected_parent_posterior": math.nan,
        "zero_edge_spoke_consensus_selected_fold_parent_posterior": math.nan,
    }
    for column, value in defaults.items():
        output[column] = value

    allowed_prior_actions = {
        "retain_current_direct_motif",
        "tier1_nonexact_production_completion",
        "tier2_positive_collider_incomplete_rescue",
        "zero_edge_state_gated_rank_fusion_completion",
        ZERO_EDGE_STATE_CONSISTENT_RUNNER_ACTION,
    }
    for index, decision in output.iterrows():
        assay = str(decision["source_assay"])
        prior_action = str(decision["action"])
        output.at[index, "zero_edge_spoke_consensus_prior_action"] = prior_action
        reason = ""
        if bool(decision["demote_production_motif"]):
            reason = "prior_demotion_selected"
        elif bool(decision["production_exact_two_arm_star"]):
            reason = "production_is_already_an_exact_two_arm_star"
        elif prior_action not in allowed_prior_actions:
            reason = "prior_action_is_outside_zero_edge_completion_scope"
        elif (
            not current_direct_candidates.empty
            and current_direct_candidates["source_assay"]
            .astype(str)
            .eq(assay)
            .any()
        ):
            reason = "assay_has_nonempty_direct_production"
        else:
            assay_unresolved = (
                current_demoted_or_unresolved.loc[
                    current_demoted_or_unresolved["source_assay"]
                    .astype(str)
                    .eq(assay)
                ].copy()
                if not current_demoted_or_unresolved.empty
                else pd.DataFrame()
            )
            output.at[
                index, "zero_edge_spoke_consensus_unresolved_spokes"
            ] = int(len(assay_unresolved))
            if (
                len(assay_unresolved)
                != config.zero_edge_spoke_consensus_required_unresolved_spokes
            ):
                reason = "unresolved_spoke_count_does_not_match_frozen_domain"
            elif not assay_unresolved["edge_type"].astype(str).eq(
                UNRESOLVED_EDGE_TYPE
            ).all():
                reason = "component_contains_a_nonunresolved_edge"
            elif assay_unresolved["component_id"].astype(str).nunique() != 1:
                reason = "unresolved_spokes_do_not_share_one_component"
            else:
                output.at[index, "zero_edge_spoke_consensus_considered"] = True
                base_count = int(
                    assay_unresolved["was_base_selected"].astype(bool).sum()
                )
                all_rescue = bool(
                    assay_unresolved["was_rescue_selected"].astype(bool).all()
                )
                output.at[index, "zero_edge_spoke_consensus_base_spokes"] = (
                    base_count
                )
                output.at[
                    index, "zero_edge_spoke_consensus_all_rescue_selected"
                ] = all_rescue
                if (
                    base_count
                    != config.zero_edge_spoke_consensus_required_base_spokes
                ):
                    reason = "base_spoke_count_does_not_match_frozen_domain"
                elif not all_rescue:
                    reason = "not_every_spoke_passed_rescue_selection"
                else:
                    degrees: dict[str, int] = {}
                    pairs: list[tuple[str, str]] = []
                    for row in assay_unresolved.itertuples(index=False):
                        pair = canonical_edge(str(row.node1), str(row.node2))
                        pairs.append(pair)
                        for node in pair:
                            degrees[node] = degrees.get(node, 0) + 1
                    centers = [
                        node
                        for node, degree in degrees.items()
                        if degree
                        == config.zero_edge_spoke_consensus_required_unresolved_spokes
                    ]
                    if len(set(pairs)) != len(pairs) or len(degrees) != 4:
                        reason = "unresolved_component_is_not_a_simple_three_spoke_star"
                    elif len(centers) != 1:
                        reason = "unresolved_component_lacks_one_shared_center"
                    else:
                        center = centers[0]
                        output.at[
                            index, "zero_edge_spoke_consensus_shared_center"
                        ] = center
                        ranked_spokes = assay_unresolved.copy()
                        ranked_spokes["_empirical_p"] = pd.to_numeric(
                            ranked_spokes["mean_empirical_p_positive"],
                            errors="coerce",
                        )
                        ranked_spokes["_log_odds"] = pd.to_numeric(
                            ranked_spokes["mean_pooled_log_odds"],
                            errors="coerce",
                        )
                        if not bool(
                            np.isfinite(
                                ranked_spokes[["_empirical_p", "_log_odds"]]
                                .to_numpy(dtype=float)
                            ).all()
                        ):
                            raise ValueError(
                                "Three-spoke consensus requires finite beta evidence"
                            )
                        ranked_spokes["_pair"] = ranked_spokes.apply(
                            lambda row: "||".join(
                                canonical_edge(str(row.node1), str(row.node2))
                            ),
                            axis=1,
                        )
                        ranked_spokes = ranked_spokes.sort_values(
                            ["_log_odds", "_empirical_p", "_pair"],
                            ascending=[False, True, True],
                        )
                        top_two = ranked_spokes.iloc[:2]
                        second_log_odds = float(top_two.iloc[1]["_log_odds"])
                        log_odds_margin = float(
                            second_log_odds
                            - ranked_spokes.iloc[2]["_log_odds"]
                        )
                        max_top_p = float(top_two["_empirical_p"].max())
                        output.at[
                            index, "zero_edge_spoke_consensus_second_arm_log_odds"
                        ] = second_log_odds
                        output.at[
                            index,
                            "zero_edge_spoke_consensus_second_vs_third_log_odds_margin",
                        ] = log_odds_margin
                        output.at[
                            index,
                            "zero_edge_spoke_consensus_maximum_top_arm_empirical_positive_p",
                        ] = max_top_p
                        if (
                            second_log_odds
                            < config.zero_edge_spoke_consensus_minimum_second_arm_log_odds
                        ):
                            reason = "second_beta_spoke_log_odds_below_threshold"
                        elif (
                            log_odds_margin
                            < config.zero_edge_spoke_consensus_minimum_second_vs_third_log_odds_margin
                        ):
                            reason = "second_beta_spoke_lacks_margin_over_third"
                        elif (
                            max_top_p
                            > config.zero_edge_spoke_consensus_maximum_second_arm_empirical_positive_p
                        ):
                            reason = "top_beta_spoke_empirical_p_above_threshold"
                        else:
                            outers = sorted(
                                next(node for node in pair if node != center)
                                for pair in (
                                    canonical_edge(str(row.node1), str(row.node2))
                                    for row in top_two.itertuples(index=False)
                                )
                            )
                            assay_candidates = candidates.loc[
                                candidates["source_assay"].astype(str).eq(assay)
                            ].copy()
                            candidate_nodes = sorted(
                                set(assay_candidates["center"].astype(str))
                                | set(assay_candidates["outer1"].astype(str))
                                | set(assay_candidates["outer2"].astype(str))
                            )
                            universe_complete = bool(
                                len(candidate_nodes) == 6
                                and len(assay_candidates) == 6 * math.comb(5, 2)
                                and not assay_candidates.duplicated(
                                    ["center", "outer1", "outer2"]
                                ).any()
                            )
                            output.at[
                                index,
                                "zero_edge_spoke_consensus_candidate_universe_complete",
                            ] = universe_complete
                            selected = (
                                _candidate_with_identity(
                                    assay_candidates,
                                    center=center,
                                    outer1=outers[0],
                                    outer2=outers[1],
                                )
                                if universe_complete
                                else None
                            )
                            output.at[
                                index,
                                "zero_edge_spoke_consensus_candidate_available",
                            ] = selected is not None
                            if selected is None:
                                reason = "beta_consensus_candidate_is_unavailable"
                            else:
                                candidate_passed = bool(
                                    selected.two_parent_positive_state_wald_consistency
                                    and float(
                                        selected.minimum_outer_conditional_wald_view_support
                                    )
                                    >= config.zero_edge_spoke_consensus_minimum_outer_wald_view_support
                                    and float(selected.raw_generative_score)
                                    >= config.zero_edge_spoke_consensus_minimum_raw_generative_score
                                    and float(
                                        selected.crossfit_joint_gain_over_baseline
                                    )
                                    >= config.zero_edge_spoke_consensus_minimum_joint_gain
                                    and float(selected.crossfit_min_drop_one_gain)
                                    >= config.zero_edge_spoke_consensus_minimum_drop_one_gain
                                    and float(
                                        selected.laplace_min_parent_positive_posterior
                                    )
                                    >= config.zero_edge_spoke_consensus_minimum_parent_posterior
                                    and float(
                                        selected.laplace_fold_min_parent_positive_posterior
                                    )
                                    >= config.zero_edge_spoke_consensus_minimum_fold_parent_posterior
                                    and float(
                                        selected.laplace_both_positive_fold_support
                                    )
                                    >= config.zero_edge_spoke_consensus_minimum_both_positive_fold_support
                                    and float(
                                        selected.laplace_both_practical_fold_support
                                    )
                                    >= config.zero_edge_spoke_consensus_minimum_both_practical_fold_support
                                    and float(
                                        selected.maximum_arm_empirical_positive_p
                                    )
                                    <= config.zero_edge_spoke_consensus_maximum_arm_empirical_positive_p
                                )
                                output.at[
                                    index,
                                    "zero_edge_spoke_consensus_candidate_passed",
                                ] = candidate_passed
                                if not candidate_passed:
                                    reason = (
                                        "beta_consensus_candidate_failed_state_or_"
                                        "uncertainty_gates"
                                    )
                                else:
                                    other_candidates = assay_candidates.drop(
                                        index=selected.name
                                    )
                                    runner_margin = (
                                        float(
                                            selected.raw_generative_score
                                            - other_candidates[
                                                "raw_generative_score"
                                            ].max()
                                        )
                                        if len(other_candidates)
                                        else math.inf
                                    )
                                    current_identity = (
                                        str(decision.get("selected_center", "")),
                                        str(decision.get("selected_outer1", "")),
                                        str(decision.get("selected_outer2", "")),
                                    )
                                    consensus_identity = (
                                        str(selected.center),
                                        str(selected.outer1),
                                        str(selected.outer2),
                                    )
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_center",
                                    ] = consensus_identity[0]
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_outer1",
                                    ] = consensus_identity[1]
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_outer2",
                                    ] = consensus_identity[2]
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_raw_generative_score",
                                    ] = float(selected.raw_generative_score)
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_joint_gain",
                                    ] = float(
                                        selected.crossfit_joint_gain_over_baseline
                                    )
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_drop_one_gain",
                                    ] = float(selected.crossfit_min_drop_one_gain)
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_parent_posterior",
                                    ] = float(
                                        selected.laplace_min_parent_positive_posterior
                                    )
                                    output.at[
                                        index,
                                        "zero_edge_spoke_consensus_selected_fold_parent_posterior",
                                    ] = float(
                                        selected.laplace_fold_min_parent_positive_posterior
                                    )
                                    if (
                                        bool(decision["selected_replacement"])
                                        and current_identity == consensus_identity
                                    ):
                                        output.at[
                                            index, "zero_edge_spoke_consensus_status"
                                        ] = "agreed_with_prior_selection"
                                        output.at[
                                            index, "zero_edge_spoke_consensus_reason"
                                        ] = (
                                            "beta_spoke_consensus_matches_prior_"
                                            "zero_edge_completion"
                                        )
                                        continue
                                    output.at[index, "selected_replacement"] = True
                                    output.at[index, "demote_production_motif"] = False
                                    output.at[index, "selected_center"] = (
                                        consensus_identity[0]
                                    )
                                    output.at[index, "selected_outer1"] = (
                                        consensus_identity[1]
                                    )
                                    output.at[index, "selected_outer2"] = (
                                        consensus_identity[2]
                                    )
                                    output.at[
                                        index, "selected_raw_generative_score"
                                    ] = float(selected.raw_generative_score)
                                    output.at[index, "selection_runner_margin"] = (
                                        runner_margin
                                    )
                                    output.at[
                                        index,
                                        "selected_score_minus_production_score",
                                    ] = math.nan
                                    output.at[index, "action"] = (
                                        ZERO_EDGE_SPOKE_CONSENSUS_ACTION
                                    )
                                    output.at[
                                        index, "zero_edge_spoke_consensus_selected"
                                    ] = True
                                    output.at[
                                        index, "zero_edge_spoke_consensus_status"
                                    ] = "selected"
                                    output.at[
                                        index, "zero_edge_spoke_consensus_reason"
                                    ] = (
                                        "two_strongest_beta_spokes_defined_one_"
                                        "state_stable_crossfit_star"
                                    )
                                    continue
        output.at[index, "zero_edge_spoke_consensus_status"] = (
            "abstained" if bool(output.at[index, "zero_edge_spoke_consensus_considered"])
            else "not_applicable"
        )
        output.at[index, "zero_edge_spoke_consensus_reason"] = reason

    for column in (
        "zero_edge_spoke_consensus_considered",
        "zero_edge_spoke_consensus_all_rescue_selected",
        "zero_edge_spoke_consensus_candidate_universe_complete",
        "zero_edge_spoke_consensus_candidate_available",
        "zero_edge_spoke_consensus_candidate_passed",
        "zero_edge_spoke_consensus_selected",
    ):
        _require_strict_boolean_column(output, column, "spoke-consensus decisions")
    _require_strict_boolean_column(
        output, "selected_replacement", "spoke-consensus decisions"
    )
    return output


def _rescue_orientation_completion_audit(
    *,
    assay_candidates: pd.DataFrame,
    rescue_pair: tuple[str, str],
    candidate_universe_complete: bool,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None, float]:
    """Use unanimous reverse-center evidence only after the strong rescue gate."""

    audit: dict[str, Any] = {
        "rescue_orientation_completion_considered": False,
        "rescue_orientation_completion_status": "not_applicable",
        "rescue_orientation_completion_reason": "",
        "rescue_orientation_completion_passing_endpoints": 0,
        "rescue_orientation_completion_selected_center": "",
        "rescue_orientation_completion_minimum_reverse_center_gain_margin": math.nan,
        "rescue_orientation_completion_eligible_candidates": 0,
        "rescue_orientation_completion_selected": False,
        "rescue_orientation_completion_selected_outer1": "",
        "rescue_orientation_completion_selected_outer2": "",
        "rescue_orientation_completion_selected_raw_generative_score": math.nan,
        "rescue_orientation_completion_same_center_runner_margin": math.nan,
    }
    if not candidate_universe_complete:
        audit["rescue_orientation_completion_reason"] = (
            "candidate_universe_incomplete"
        )
        return audit, None, math.nan
    audit["rescue_orientation_completion_considered"] = True
    nodes = sorted(
        set(assay_candidates["center"].astype(str))
        | set(assay_candidates["outer1"].astype(str))
        | set(assay_candidates["outer2"].astype(str))
    )
    third_features = sorted(set(nodes) - set(rescue_pair))
    endpoint_rows: dict[str, pd.DataFrame] = {}
    endpoint_minimum_margin: dict[str, float] = {}
    for endpoint in rescue_pair:
        rows = assay_candidates.loc[
            assay_candidates["center"].astype(str).eq(endpoint)
            & assay_candidates.apply(
                lambda row: rescue_pair
                in {
                    canonical_edge(str(row.center), str(row.outer1)),
                    canonical_edge(str(row.center), str(row.outer2)),
                },
                axis=1,
            )
        ].copy()
        endpoint_rows[endpoint] = rows
        margins = pd.to_numeric(
            rows["crossfit_joint_gain_over_baseline_reverse_center_margin"],
            errors="coerce",
        )
        if len(rows) != len(third_features) or not bool(
            np.isfinite(margins.to_numpy(dtype=float)).all()
        ):
            audit["rescue_orientation_completion_status"] = "abstained"
            audit["rescue_orientation_completion_reason"] = (
                "endpoint_orientation_context_is_incomplete"
            )
            return audit, None, math.nan
        endpoint_minimum_margin[endpoint] = float(margins.min())

    passing_endpoints = [
        endpoint
        for endpoint, margin in endpoint_minimum_margin.items()
        if margin
        >= config.zero_edge_rescue_orientation_minimum_reverse_center_gain_margin
    ]
    audit["rescue_orientation_completion_passing_endpoints"] = len(
        passing_endpoints
    )
    if len(passing_endpoints) != 1:
        audit["rescue_orientation_completion_status"] = "abstained"
        audit["rescue_orientation_completion_reason"] = (
            "unanimous_orientation_endpoint_count_is_not_exactly_one"
        )
        return audit, None, math.nan

    selected_center = passing_endpoints[0]
    center_rows = endpoint_rows[selected_center]
    audit["rescue_orientation_completion_selected_center"] = selected_center
    audit[
        "rescue_orientation_completion_minimum_reverse_center_gain_margin"
    ] = endpoint_minimum_margin[selected_center]
    eligible = _ordered_candidates(
        center_rows.loc[
            center_rows["raw_generative_score"].ge(
                config.zero_edge_rescue_orientation_minimum_raw_generative_score
            )
            & center_rows["crossfit_joint_gain_over_baseline"].ge(
                config.zero_edge_rescue_orientation_minimum_joint_gain
            )
            & center_rows["laplace_min_parent_positive_posterior"].ge(
                config.zero_edge_rescue_orientation_minimum_parent_posterior
            )
            & center_rows["laplace_fold_min_parent_positive_posterior"].ge(
                config.zero_edge_rescue_orientation_minimum_fold_parent_posterior
            )
            & center_rows["laplace_both_positive_fold_support"].ge(
                config.zero_edge_rescue_orientation_minimum_both_positive_fold_support
            )
            & center_rows["laplace_both_practical_fold_support"].ge(
                config.zero_edge_rescue_orientation_minimum_both_practical_fold_support
            )
            & center_rows["maximum_arm_empirical_positive_p"].le(
                config.zero_edge_rescue_orientation_maximum_arm_empirical_positive_p
            )
            & center_rows["outer_empirical_null_compatibility_p"].ge(
                config.zero_edge_rescue_orientation_minimum_outer_empirical_null_compatibility_p
            )
            & center_rows["minimum_outer_conditional_wald_view_support"].ge(
                config.zero_edge_rescue_orientation_minimum_outer_wald_view_support
            )
        ]
    )
    audit["rescue_orientation_completion_eligible_candidates"] = int(len(eligible))
    if len(eligible) != 1:
        audit["rescue_orientation_completion_status"] = "abstained"
        audit["rescue_orientation_completion_reason"] = (
            "sparse_orientation_candidate_count_is_not_exactly_one"
        )
        return audit, None, math.nan

    selected = eligible.iloc[0]
    other_center_rows = center_rows.drop(index=selected.name)
    runner_margin = (
        float(
            selected.raw_generative_score
            - other_center_rows["raw_generative_score"].max()
        )
        if len(other_center_rows)
        else math.inf
    )
    audit.update(
        {
            "rescue_orientation_completion_selected_outer1": str(
                selected.outer1
            ),
            "rescue_orientation_completion_selected_outer2": str(
                selected.outer2
            ),
            "rescue_orientation_completion_selected_raw_generative_score": float(
                selected.raw_generative_score
            ),
            "rescue_orientation_completion_same_center_runner_margin": (
                runner_margin
            ),
        }
    )
    if (
        runner_margin
        < config.zero_edge_rescue_orientation_minimum_same_center_runner_margin
    ):
        audit["rescue_orientation_completion_status"] = "abstained"
        audit["rescue_orientation_completion_reason"] = (
            "sparse_orientation_candidate_lacks_same_center_runner_margin"
        )
        return audit, None, runner_margin

    audit["rescue_orientation_completion_status"] = "selected"
    audit["rescue_orientation_completion_reason"] = (
        "one_endpoint_won_every_reverse_center_context_and_one_sparse_star_"
        "passed_uncertainty_gates"
    )
    audit["rescue_orientation_completion_selected"] = True
    return audit, selected, runner_margin


def _dominant_rescue_completion_audit(
    *,
    assay_candidates: pd.DataFrame,
    rescue_only: pd.DataFrame,
    rank_edge_scores_json: str,
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[dict[str, Any], pd.Series | None, float]:
    """Resolve two competing rescue arms only when one is decisively stronger."""

    audit: dict[str, Any] = {
        "dominant_rescue_completion_considered": False,
        "dominant_rescue_completion_status": "not_applicable",
        "dominant_rescue_completion_reason": "",
        "dominant_rescue_completion_candidate_universe_complete": False,
        "dominant_rescue_completion_rescue_count": int(len(rescue_only)),
        "dominant_rescue_completion_selected_node1": "",
        "dominant_rescue_completion_selected_node2": "",
        "dominant_rescue_completion_leading_log_odds": math.nan,
        "dominant_rescue_completion_runner_log_odds": math.nan,
        "dominant_rescue_completion_log_odds_margin": math.nan,
        "dominant_rescue_completion_leading_empirical_p": math.nan,
        "dominant_rescue_completion_direction_parent": "",
        "dominant_rescue_completion_direction_center": "",
        "dominant_rescue_completion_direction_probability": math.nan,
        "dominant_rescue_completion_reverse_probability": math.nan,
        "dominant_rescue_completion_direction_probability_margin": math.nan,
        "dominant_rescue_completion_eligible_candidates": 0,
        "dominant_rescue_completion_selected": False,
        "dominant_rescue_completion_selected_center": "",
        "dominant_rescue_completion_selected_outer1": "",
        "dominant_rescue_completion_selected_outer2": "",
        "dominant_rescue_completion_selected_raw_generative_score": math.nan,
        "dominant_rescue_completion_same_center_runner_margin": math.nan,
    }
    if len(rescue_only) != config.zero_edge_dominant_rescue_required_count:
        audit["dominant_rescue_completion_reason"] = (
            "rescue_only_count_does_not_match_competing_rescue_contract"
        )
        return audit, None, math.nan
    audit["dominant_rescue_completion_considered"] = True

    evidence = rescue_only.copy()
    for column in ("mean_pooled_log_odds", "mean_empirical_p_positive"):
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    if not bool(
        np.isfinite(
            evidence[
                ["mean_pooled_log_odds", "mean_empirical_p_positive"]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("Competing rescue evidence must be finite")
    evidence["_pair"] = evidence.apply(
        lambda row: canonical_edge(str(row["node1"]), str(row["node2"])),
        axis=1,
    )
    if evidence["_pair"].duplicated().any():
        raise ValueError("Competing rescue evidence contains duplicate adjacencies")
    evidence = evidence.sort_values(
        ["mean_pooled_log_odds", "mean_empirical_p_positive", "node1", "node2"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    leading = evidence.iloc[0]
    runner = evidence.iloc[1]
    pair = leading["_pair"]
    leading_log_odds = float(leading["mean_pooled_log_odds"])
    runner_log_odds = float(runner["mean_pooled_log_odds"])
    log_odds_margin = leading_log_odds - runner_log_odds
    leading_empirical_p = float(leading["mean_empirical_p_positive"])
    audit.update(
        {
            "dominant_rescue_completion_selected_node1": pair[0],
            "dominant_rescue_completion_selected_node2": pair[1],
            "dominant_rescue_completion_leading_log_odds": leading_log_odds,
            "dominant_rescue_completion_runner_log_odds": runner_log_odds,
            "dominant_rescue_completion_log_odds_margin": log_odds_margin,
            "dominant_rescue_completion_leading_empirical_p": leading_empirical_p,
        }
    )
    if (
        leading_log_odds
        < config.zero_edge_dominant_rescue_minimum_leading_log_odds
        or log_odds_margin
        < config.zero_edge_dominant_rescue_minimum_log_odds_margin
        or leading_empirical_p
        > config.zero_edge_dominant_rescue_maximum_leading_empirical_p
    ):
        audit["dominant_rescue_completion_status"] = "abstained"
        audit["dominant_rescue_completion_reason"] = (
            "leading_rescue_lacked_strength_or_margin"
        )
        return audit, None, math.nan

    try:
        edge_score_records = json.loads(rank_edge_scores_json)
    except (TypeError, json.JSONDecodeError):
        edge_score_records = []
    if not isinstance(edge_score_records, list):
        edge_score_records = []
    direction_probabilities: dict[tuple[str, str], float] = {}
    for record in edge_score_records:
        if not isinstance(record, Mapping):
            continue
        parent = str(record.get("parent", "")).strip()
        child = str(record.get("child", "")).strip()
        probability = pd.to_numeric(
            pd.Series([record.get("probability")]), errors="coerce"
        ).iloc[0]
        if not parent or not child or not np.isfinite(probability):
            continue
        key = (parent, child)
        if key in direction_probabilities:
            raise ValueError("Frozen rank edge scores contain a duplicate direction")
        direction_probabilities[key] = float(probability)
    forward = direction_probabilities.get((pair[0], pair[1]), math.nan)
    reverse = direction_probabilities.get((pair[1], pair[0]), math.nan)
    if not np.isfinite(forward) or not np.isfinite(reverse):
        audit["dominant_rescue_completion_status"] = "abstained"
        audit["dominant_rescue_completion_reason"] = (
            "dominant_rescue_lacked_both_frozen_direction_scores"
        )
        return audit, None, math.nan
    if forward >= reverse:
        parent, center = pair[0], pair[1]
        direction_probability, reverse_probability = forward, reverse
    else:
        parent, center = pair[1], pair[0]
        direction_probability, reverse_probability = reverse, forward
    direction_margin = direction_probability - reverse_probability
    audit.update(
        {
            "dominant_rescue_completion_direction_parent": parent,
            "dominant_rescue_completion_direction_center": center,
            "dominant_rescue_completion_direction_probability": (
                direction_probability
            ),
            "dominant_rescue_completion_reverse_probability": reverse_probability,
            "dominant_rescue_completion_direction_probability_margin": (
                direction_margin
            ),
        }
    )
    if (
        direction_probability
        < config.zero_edge_dominant_rescue_minimum_direction_probability
        or direction_margin
        < config.zero_edge_dominant_rescue_minimum_direction_probability_margin
    ):
        audit["dominant_rescue_completion_status"] = "abstained"
        audit["dominant_rescue_completion_reason"] = (
            "dominant_rescue_lacked_frozen_direction_margin"
        )
        return audit, None, math.nan

    candidate_nodes = sorted(
        set(assay_candidates["center"].astype(str))
        | set(assay_candidates["outer1"].astype(str))
        | set(assay_candidates["outer2"].astype(str))
    )
    candidate_universe_complete = bool(
        len(candidate_nodes) == 6
        and len(assay_candidates) == 6 * math.comb(5, 2)
        and not assay_candidates.duplicated(
            ["center", "outer1", "outer2"]
        ).any()
    )
    audit[
        "dominant_rescue_completion_candidate_universe_complete"
    ] = candidate_universe_complete
    if not candidate_universe_complete:
        audit["dominant_rescue_completion_status"] = "abstained"
        audit["dominant_rescue_completion_reason"] = (
            "candidate_universe_incomplete"
        )
        return audit, None, math.nan

    center_rows = assay_candidates.loc[
        assay_candidates["center"].astype(str).eq(center)
        & assay_candidates.apply(
            lambda row: pair
            in {
                canonical_edge(str(row.center), str(row.outer1)),
                canonical_edge(str(row.center), str(row.outer2)),
            },
            axis=1,
        )
    ].copy()
    eligible = _ordered_candidates(
        center_rows.loc[
            center_rows["raw_generative_score"].ge(
                config.zero_edge_dominant_rescue_minimum_raw_generative_score
            )
            & center_rows["crossfit_joint_gain_over_baseline"].ge(
                config.zero_edge_dominant_rescue_minimum_joint_gain
            )
            & center_rows["laplace_min_parent_positive_posterior"].ge(
                config.zero_edge_dominant_rescue_minimum_parent_posterior
            )
            & center_rows["laplace_fold_min_parent_positive_posterior"].ge(
                config.zero_edge_dominant_rescue_minimum_fold_parent_posterior
            )
            & center_rows["laplace_both_positive_fold_support"].ge(
                config.zero_edge_dominant_rescue_minimum_both_positive_fold_support
            )
            & center_rows["laplace_both_practical_fold_support"].ge(
                config.zero_edge_dominant_rescue_minimum_both_practical_fold_support
            )
            & center_rows["maximum_arm_empirical_positive_p"].le(
                config.zero_edge_dominant_rescue_maximum_arm_empirical_positive_p
            )
            & center_rows["outer_empirical_null_compatibility_p"].ge(
                config.zero_edge_dominant_rescue_minimum_outer_empirical_null_compatibility_p
            )
        ]
    )
    audit["dominant_rescue_completion_eligible_candidates"] = int(len(eligible))
    if len(eligible) != 1:
        audit["dominant_rescue_completion_status"] = "abstained"
        audit["dominant_rescue_completion_reason"] = (
            "dominant_rescue_sparse_candidate_count_is_not_exactly_one"
        )
        return audit, None, math.nan

    selected = eligible.iloc[0]
    other_center_rows = center_rows.drop(index=selected.name)
    runner_margin = (
        float(
            selected.raw_generative_score
            - other_center_rows["raw_generative_score"].max()
        )
        if len(other_center_rows)
        else math.inf
    )
    audit.update(
        {
            "dominant_rescue_completion_selected_center": str(selected.center),
            "dominant_rescue_completion_selected_outer1": str(selected.outer1),
            "dominant_rescue_completion_selected_outer2": str(selected.outer2),
            "dominant_rescue_completion_selected_raw_generative_score": float(
                selected.raw_generative_score
            ),
            "dominant_rescue_completion_same_center_runner_margin": runner_margin,
        }
    )
    if (
        runner_margin
        < config.zero_edge_dominant_rescue_minimum_same_center_runner_margin
    ):
        audit["dominant_rescue_completion_status"] = "abstained"
        audit["dominant_rescue_completion_reason"] = (
            "dominant_rescue_candidate_lacked_same_center_runner_margin"
        )
        return audit, None, runner_margin

    audit["dominant_rescue_completion_status"] = "selected"
    audit["dominant_rescue_completion_reason"] = (
        "one_rescue_arm_dominated_the_competitor_and_one_directional_sparse_"
        "completion_passed_uncertainty_gates"
    )
    audit["dominant_rescue_completion_selected"] = True
    return audit, selected, runner_margin


def _annotate_unique_rescue_rank_overlap_adjacencies(
    *,
    decisions: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    current_demoted_or_unresolved: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    config: CrossfitHierarchicalColliderConfig | None = None,
) -> pd.DataFrame:
    """Validate a rescue-anchored star or retain adjacency-only semantics.

    This fallback is intentionally downstream of every oriented-star decision.
    It can act only when frozen rank fusion abstained on an assay with no direct
    production edge and exactly one rescue-only unresolved adjacency.  The
    adjacency must also be an arm of rank fusion's frozen winning star.  One
    and only one complete star containing that adjacency may be oriented when
    it independently passes the state, fold, posterior, and empirical-null
    gates.  Otherwise the overlap supplies adjacency evidence only.
    """

    cfg = config or CrossfitHierarchicalColliderConfig()
    cfg.validate()
    _assert_observed_only(decisions, "decisions")
    _assert_observed_only(current_direct_candidates, "current_direct_candidates")
    _assert_observed_only(
        current_demoted_or_unresolved,
        "current_demoted_or_unresolved",
    )
    candidate_frame = pd.DataFrame() if candidates is None else candidates.copy()
    if not candidate_frame.empty:
        _assert_observed_only(candidate_frame, "candidates")
        _require_columns(
            candidate_frame,
            {
                "source_assay",
                "center",
                "outer1",
                "outer2",
                "raw_generative_score",
                "crossfit_joint_gain_over_baseline",
                "crossfit_joint_gain_over_baseline_reverse_center_margin",
                "crossfit_min_drop_one_gain",
                "laplace_min_parent_positive_posterior",
                "laplace_fold_min_parent_positive_posterior",
                "laplace_both_positive_fold_support",
                "laplace_both_practical_fold_support",
                "maximum_arm_empirical_positive_p",
                "outer_empirical_null_compatibility_p",
                "minimum_outer_conditional_wald_view_support",
                "two_parent_positive_state_wald_consistency",
            },
            "candidates",
        )
        _require_strict_boolean_column(
            candidate_frame,
            "two_parent_positive_state_wald_consistency",
            "candidates",
        )
    _require_columns(
        decisions,
        {
            "source_assay",
            "action",
            "selected_replacement",
            "demote_production_motif",
            "zero_edge_rank_fusion_status",
            "zero_edge_rank_fusion_selected_center",
            "zero_edge_rank_fusion_selected_outer1",
            "zero_edge_rank_fusion_selected_outer2",
        },
        "decisions",
    )
    _require_strict_boolean_column(decisions, "selected_replacement", "decisions")
    _require_strict_boolean_column(decisions, "demote_production_motif", "decisions")
    if decisions["source_assay"].astype(str).duplicated().any():
        raise ValueError("decisions contains duplicate assay rows")
    if not current_direct_candidates.empty:
        _require_columns(
            current_direct_candidates,
            {"source_assay"},
            "current_direct_candidates",
        )
    if not current_demoted_or_unresolved.empty:
        _require_columns(
            current_demoted_or_unresolved,
            {
                "source_assay",
                "node1",
                "node2",
                "edge_type",
                "was_base_selected",
                "was_rescue_selected",
                "mean_empirical_p_positive",
                "mean_pooled_log_odds",
            },
            "current_demoted_or_unresolved",
        )
        _require_strict_boolean_column(
            current_demoted_or_unresolved,
            "was_base_selected",
            "current_demoted_or_unresolved",
        )
        _require_strict_boolean_column(
            current_demoted_or_unresolved,
            "was_rescue_selected",
            "current_demoted_or_unresolved",
        )

    output = decisions.copy()
    defaults: dict[str, Any] = {
        "selected_unoriented_adjacency": False,
        "unoriented_adjacency_node1": "",
        "unoriented_adjacency_node2": "",
        "unoriented_adjacency_status": "not_applicable",
        "unoriented_adjacency_reason": "",
        "unoriented_adjacency_rescue_only_count": 0,
        "unoriented_adjacency_rank_winner_arm_overlap": False,
        "unoriented_adjacency_mean_empirical_p_positive": math.nan,
        "unoriented_adjacency_mean_pooled_log_odds": math.nan,
        "rescue_anchored_completion_considered": False,
        "rescue_anchored_completion_status": "not_applicable",
        "rescue_anchored_completion_reason": "",
        "rescue_anchored_completion_candidate_universe_complete": False,
        "rescue_anchored_completion_eligible_candidates": 0,
        "rescue_anchored_completion_selected": False,
        "rescue_anchored_completion_selected_center": "",
        "rescue_anchored_completion_selected_outer1": "",
        "rescue_anchored_completion_selected_outer2": "",
        "rescue_anchored_completion_selected_raw_generative_score": math.nan,
        "rescue_anchored_completion_selected_joint_gain": math.nan,
        "rescue_anchored_completion_selected_drop_one_gain": math.nan,
        "rescue_anchored_completion_selected_parent_posterior": math.nan,
        "rescue_anchored_completion_selected_fold_parent_posterior": math.nan,
        "rescue_anchored_completion_selected_arm_empirical_positive_p": math.nan,
        "rescue_anchored_completion_selected_outer_empirical_null_compatibility_p": math.nan,
        "rescue_anchored_completion_selected_outer_wald_view_support": math.nan,
        "rescue_orientation_completion_considered": False,
        "rescue_orientation_completion_status": "not_applicable",
        "rescue_orientation_completion_reason": "",
        "rescue_orientation_completion_passing_endpoints": 0,
        "rescue_orientation_completion_selected_center": "",
        "rescue_orientation_completion_minimum_reverse_center_gain_margin": math.nan,
        "rescue_orientation_completion_eligible_candidates": 0,
        "rescue_orientation_completion_selected": False,
        "rescue_orientation_completion_selected_outer1": "",
        "rescue_orientation_completion_selected_outer2": "",
        "rescue_orientation_completion_selected_raw_generative_score": math.nan,
        "rescue_orientation_completion_same_center_runner_margin": math.nan,
        "dominant_rescue_completion_considered": False,
        "dominant_rescue_completion_status": "not_applicable",
        "dominant_rescue_completion_reason": "",
        "dominant_rescue_completion_candidate_universe_complete": False,
        "dominant_rescue_completion_rescue_count": 0,
        "dominant_rescue_completion_selected_node1": "",
        "dominant_rescue_completion_selected_node2": "",
        "dominant_rescue_completion_leading_log_odds": math.nan,
        "dominant_rescue_completion_runner_log_odds": math.nan,
        "dominant_rescue_completion_log_odds_margin": math.nan,
        "dominant_rescue_completion_leading_empirical_p": math.nan,
        "dominant_rescue_completion_direction_parent": "",
        "dominant_rescue_completion_direction_center": "",
        "dominant_rescue_completion_direction_probability": math.nan,
        "dominant_rescue_completion_reverse_probability": math.nan,
        "dominant_rescue_completion_direction_probability_margin": math.nan,
        "dominant_rescue_completion_eligible_candidates": 0,
        "dominant_rescue_completion_selected": False,
        "dominant_rescue_completion_selected_center": "",
        "dominant_rescue_completion_selected_outer1": "",
        "dominant_rescue_completion_selected_outer2": "",
        "dominant_rescue_completion_selected_raw_generative_score": math.nan,
        "dominant_rescue_completion_same_center_runner_margin": math.nan,
    }
    for column, value in defaults.items():
        output[column] = value

    for index, decision in output.iterrows():
        assay = str(decision["source_assay"])
        reason = ""
        if bool(decision["selected_replacement"]) or bool(
            decision["demote_production_motif"]
        ):
            reason = "higher_priority_oriented_action_selected"
        elif str(decision["zero_edge_rank_fusion_status"]) != "abstained":
            reason = "rank_fusion_did_not_abstain"
        elif str(decision["action"]) != "retain_current_direct_motif":
            reason = "prior_action_is_not_retention"
        elif (
            not current_direct_candidates.empty
            and bool(
                current_direct_candidates["source_assay"]
                .astype(str)
                .eq(assay)
                .any()
            )
        ):
            reason = "assay_has_nonempty_direct_production"
        else:
            assay_unresolved = current_demoted_or_unresolved.loc[
                current_demoted_or_unresolved["source_assay"]
                .astype(str)
                .eq(assay)
            ].copy() if not current_demoted_or_unresolved.empty else pd.DataFrame()
            rescue_only = assay_unresolved.loc[
                assay_unresolved["edge_type"].astype(str).eq(UNRESOLVED_EDGE_TYPE)
                & assay_unresolved["was_rescue_selected"].astype(bool)
                & ~assay_unresolved["was_base_selected"].astype(bool)
            ].copy() if not assay_unresolved.empty else pd.DataFrame()
            output.at[
                index, "unoriented_adjacency_rescue_only_count"
            ] = int(len(rescue_only))
            if len(rescue_only) == cfg.zero_edge_dominant_rescue_required_count:
                assay_candidates = candidate_frame.loc[
                    candidate_frame["source_assay"].astype(str).eq(assay)
                ].copy() if not candidate_frame.empty else pd.DataFrame()
                (
                    dominant_audit,
                    dominant_candidate,
                    dominant_runner_margin,
                ) = _dominant_rescue_completion_audit(
                    assay_candidates=assay_candidates,
                    rescue_only=rescue_only,
                    rank_edge_scores_json=str(
                        decision.get("zero_edge_rank_fusion_all_edge_scores_json", "")
                    ),
                    config=cfg,
                )
                for column, value in dominant_audit.items():
                    output.at[index, column] = value
                if dominant_candidate is not None:
                    output.at[index, "selected_replacement"] = True
                    output.at[index, "selected_center"] = str(
                        dominant_candidate.center
                    )
                    output.at[index, "selected_outer1"] = str(
                        dominant_candidate.outer1
                    )
                    output.at[index, "selected_outer2"] = str(
                        dominant_candidate.outer2
                    )
                    output.at[index, "selected_raw_generative_score"] = float(
                        dominant_candidate.raw_generative_score
                    )
                    output.at[index, "selection_runner_margin"] = (
                        dominant_runner_margin
                    )
                    output.at[
                        index, "selected_score_minus_production_score"
                    ] = math.nan
                    output.at[index, "action"] = (
                        ZERO_EDGE_DOMINANT_RESCUE_COMPLETION_ACTION
                    )
                    output.at[index, "unoriented_adjacency_status"] = (
                        "not_needed_oriented_completion_selected"
                    )
                    output.at[index, "unoriented_adjacency_reason"] = (
                        "dominant_competing_rescue_selected_a_sparse_star"
                    )
                    continue
                reason = str(dominant_audit["dominant_rescue_completion_reason"])
            elif len(rescue_only) != 1:
                reason = "rescue_only_unresolved_count_is_not_exactly_one"
            else:
                rescue = rescue_only.iloc[0]
                empirical_p = pd.to_numeric(
                    pd.Series([rescue["mean_empirical_p_positive"]]),
                    errors="coerce",
                ).iloc[0]
                pooled_log_odds = pd.to_numeric(
                    pd.Series([rescue["mean_pooled_log_odds"]]),
                    errors="coerce",
                ).iloc[0]
                if not np.isfinite(empirical_p) or not np.isfinite(pooled_log_odds):
                    raise ValueError(
                        "The unique rescue-only adjacency requires finite empirical "
                        "p-value and pooled log-odds evidence"
                    )
                pair = canonical_edge(str(rescue["node1"]), str(rescue["node2"]))
                center = str(
                    decision["zero_edge_rank_fusion_selected_center"]
                ).strip()
                outer1 = str(
                    decision["zero_edge_rank_fusion_selected_outer1"]
                ).strip()
                outer2 = str(
                    decision["zero_edge_rank_fusion_selected_outer2"]
                ).strip()
                winner_arms = (
                    {
                        canonical_edge(center, outer1),
                        canonical_edge(center, outer2),
                    }
                    if center and outer1 and outer2
                    else set()
                )
                overlap = pair in winner_arms
                output.at[
                    index, "unoriented_adjacency_rank_winner_arm_overlap"
                ] = bool(overlap)
                output.at[
                    index, "unoriented_adjacency_mean_empirical_p_positive"
                ] = float(empirical_p)
                output.at[
                    index, "unoriented_adjacency_mean_pooled_log_odds"
                ] = float(pooled_log_odds)
                if (
                    empirical_p
                    > UNORIENTED_RESCUE_MAX_MEAN_EMPIRICAL_P_POSITIVE
                ):
                    reason = "unique_rescue_empirical_p_exceeds_frozen_limit"
                elif (
                    pooled_log_odds
                    < UNORIENTED_RESCUE_MIN_MEAN_POOLED_LOG_ODDS
                ):
                    reason = "unique_rescue_log_odds_below_frozen_minimum"
                elif not overlap:
                    reason = "unique_rescue_does_not_overlap_rank_winner_arm"
                else:
                    output.at[
                        index, "rescue_anchored_completion_considered"
                    ] = True
                    assay_candidates = candidate_frame.loc[
                        candidate_frame["source_assay"].astype(str).eq(assay)
                    ].copy() if not candidate_frame.empty else pd.DataFrame()
                    candidate_nodes = (
                        sorted(
                            set(assay_candidates["center"].astype(str))
                            | set(assay_candidates["outer1"].astype(str))
                            | set(assay_candidates["outer2"].astype(str))
                        )
                        if not assay_candidates.empty
                        else []
                    )
                    expected_candidate_count = 6 * math.comb(5, 2)
                    candidate_universe_complete = bool(
                        len(candidate_nodes) == 6
                        and len(assay_candidates) == expected_candidate_count
                        and not assay_candidates.duplicated(
                            ["center", "outer1", "outer2"]
                        ).any()
                    )
                    output.at[
                        index,
                        "rescue_anchored_completion_candidate_universe_complete",
                    ] = candidate_universe_complete
                    eligible = pd.DataFrame()
                    rescue_contains = pd.Series(
                        False,
                        index=assay_candidates.index,
                        dtype=bool,
                    )
                    if candidate_universe_complete:
                        rescue_contains = assay_candidates.apply(
                            lambda row: pair
                            in {
                                canonical_edge(
                                    str(row["center"]), str(row["outer1"])
                                ),
                                canonical_edge(
                                    str(row["center"]), str(row["outer2"])
                                ),
                            },
                            axis=1,
                        )
                        eligible = _ordered_candidates(
                            assay_candidates.loc[
                                rescue_contains
                                & assay_candidates[
                                    "two_parent_positive_state_wald_consistency"
                                ].astype(bool)
                                & assay_candidates[
                                    "minimum_outer_conditional_wald_view_support"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_outer_wald_view_support
                                )
                                & assay_candidates[
                                    "raw_generative_score"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_raw_generative_score
                                )
                                & assay_candidates[
                                    "crossfit_joint_gain_over_baseline"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_joint_gain
                                )
                                & assay_candidates[
                                    "crossfit_min_drop_one_gain"
                                ].gt(
                                    cfg.zero_edge_rescue_completion_minimum_drop_one_gain_exclusive
                                )
                                & assay_candidates[
                                    "laplace_min_parent_positive_posterior"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_parent_posterior
                                )
                                & assay_candidates[
                                    "laplace_fold_min_parent_positive_posterior"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_fold_parent_posterior
                                )
                                & assay_candidates[
                                    "laplace_both_positive_fold_support"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_both_positive_fold_support
                                )
                                & assay_candidates[
                                    "laplace_both_practical_fold_support"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_both_practical_fold_support
                                )
                                & assay_candidates[
                                    "maximum_arm_empirical_positive_p"
                                ].le(
                                    cfg.zero_edge_rescue_completion_maximum_arm_empirical_positive_p
                                )
                                & assay_candidates[
                                    "outer_empirical_null_compatibility_p"
                                ].ge(
                                    cfg.zero_edge_rescue_completion_minimum_outer_empirical_null_compatibility_p
                                )
                            ]
                        )
                    output.at[
                        index, "rescue_anchored_completion_eligible_candidates"
                    ] = int(len(eligible))
                    if len(eligible) == 1:
                        selected = eligible.iloc[0]
                        other_rescue_candidates = assay_candidates.loc[
                            rescue_contains & (assay_candidates.index != selected.name)
                        ]
                        selection_runner_margin = (
                            float(
                                selected.raw_generative_score
                                - other_rescue_candidates[
                                    "raw_generative_score"
                                ].max()
                            )
                            if len(other_rescue_candidates)
                            else math.inf
                        )
                        output.at[index, "selected_replacement"] = True
                        output.at[index, "selected_center"] = str(selected.center)
                        output.at[index, "selected_outer1"] = str(selected.outer1)
                        output.at[index, "selected_outer2"] = str(selected.outer2)
                        output.at[index, "selected_raw_generative_score"] = float(
                            selected.raw_generative_score
                        )
                        output.at[index, "selection_runner_margin"] = (
                            selection_runner_margin
                        )
                        output.at[
                            index, "selected_score_minus_production_score"
                        ] = math.nan
                        output.at[index, "action"] = (
                            ZERO_EDGE_RESCUE_ANCHORED_COMPLETION_ACTION
                        )
                        output.at[
                            index, "rescue_anchored_completion_selected"
                        ] = True
                        output.at[
                            index, "rescue_anchored_completion_status"
                        ] = "selected"
                        output.at[
                            index, "rescue_anchored_completion_reason"
                        ] = (
                            "one_unique_rescue_anchored_star_passed_all_"
                            "state_and_uncertainty_gates"
                        )
                        for column, value in (
                            (
                                "rescue_anchored_completion_selected_center",
                                str(selected.center),
                            ),
                            (
                                "rescue_anchored_completion_selected_outer1",
                                str(selected.outer1),
                            ),
                            (
                                "rescue_anchored_completion_selected_outer2",
                                str(selected.outer2),
                            ),
                            (
                                "rescue_anchored_completion_selected_raw_generative_score",
                                float(selected.raw_generative_score),
                            ),
                            (
                                "rescue_anchored_completion_selected_joint_gain",
                                float(selected.crossfit_joint_gain_over_baseline),
                            ),
                            (
                                "rescue_anchored_completion_selected_drop_one_gain",
                                float(selected.crossfit_min_drop_one_gain),
                            ),
                            (
                                "rescue_anchored_completion_selected_parent_posterior",
                                float(
                                    selected.laplace_min_parent_positive_posterior
                                ),
                            ),
                            (
                                "rescue_anchored_completion_selected_fold_parent_posterior",
                                float(
                                    selected.laplace_fold_min_parent_positive_posterior
                                ),
                            ),
                            (
                                "rescue_anchored_completion_selected_arm_empirical_positive_p",
                                float(selected.maximum_arm_empirical_positive_p),
                            ),
                            (
                                "rescue_anchored_completion_selected_outer_empirical_null_compatibility_p",
                                float(
                                    selected.outer_empirical_null_compatibility_p
                                ),
                            ),
                            (
                                "rescue_anchored_completion_selected_outer_wald_view_support",
                                float(
                                    selected.minimum_outer_conditional_wald_view_support
                                ),
                            ),
                        ):
                            output.at[index, column] = value
                        output.at[index, "unoriented_adjacency_status"] = (
                            "not_needed_oriented_completion_selected"
                        )
                        output.at[index, "unoriented_adjacency_reason"] = (
                            "unique_rescue_anchored_star_passed_all_orientation_gates"
                        )
                        continue
                    output.at[index, "rescue_anchored_completion_status"] = (
                        "abstained"
                    )
                    output.at[index, "rescue_anchored_completion_reason"] = (
                        "candidate_universe_incomplete"
                        if not candidate_universe_complete
                        else "eligible_rescue_anchored_star_count_is_not_exactly_one"
                    )
                    (
                        orientation_audit,
                        orientation_candidate,
                        orientation_runner_margin,
                    ) = _rescue_orientation_completion_audit(
                        assay_candidates=assay_candidates,
                        rescue_pair=pair,
                        candidate_universe_complete=candidate_universe_complete,
                        config=cfg,
                    )
                    for column, value in orientation_audit.items():
                        output.at[index, column] = value
                    if orientation_candidate is not None:
                        output.at[index, "selected_replacement"] = True
                        output.at[index, "selected_center"] = str(
                            orientation_candidate.center
                        )
                        output.at[index, "selected_outer1"] = str(
                            orientation_candidate.outer1
                        )
                        output.at[index, "selected_outer2"] = str(
                            orientation_candidate.outer2
                        )
                        output.at[index, "selected_raw_generative_score"] = float(
                            orientation_candidate.raw_generative_score
                        )
                        output.at[index, "selection_runner_margin"] = (
                            orientation_runner_margin
                        )
                        output.at[
                            index, "selected_score_minus_production_score"
                        ] = math.nan
                        output.at[index, "action"] = (
                            ZERO_EDGE_RESCUE_ORIENTATION_COMPLETION_ACTION
                        )
                        output.at[index, "unoriented_adjacency_status"] = (
                            "not_needed_oriented_completion_selected"
                        )
                        output.at[index, "unoriented_adjacency_reason"] = (
                            "unanimous_rescue_orientation_selected_a_sparse_star"
                        )
                        continue
                    output.at[index, "selected_unoriented_adjacency"] = True
                    output.at[index, "unoriented_adjacency_node1"] = pair[0]
                    output.at[index, "unoriented_adjacency_node2"] = pair[1]
                    output.at[index, "unoriented_adjacency_status"] = "selected"
                    output.at[index, "unoriented_adjacency_reason"] = (
                        "unique_rescue_only_unresolved_overlaps_frozen_rank_winner_arm"
                    )
                    output.at[index, "action"] = UNORIENTED_RESCUE_OVERLAP_ACTION
                    continue
        output.at[index, "unoriented_adjacency_status"] = (
            "abstained" if reason else "not_applicable"
        )
        output.at[index, "unoriented_adjacency_reason"] = reason

    _require_strict_boolean_column(
        output,
        "selected_unoriented_adjacency",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "unoriented_adjacency_rank_winner_arm_overlap",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "rescue_anchored_completion_considered",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "rescue_anchored_completion_candidate_universe_complete",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "rescue_anchored_completion_selected",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "rescue_orientation_completion_considered",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "rescue_orientation_completion_selected",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "dominant_rescue_completion_considered",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "dominant_rescue_completion_candidate_universe_complete",
        "annotated decisions",
    )
    _require_strict_boolean_column(
        output,
        "dominant_rescue_completion_selected",
        "annotated decisions",
    )
    return output


def _annotate_zero_edge_all_state_topology_completion(
    *,
    decisions: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    current_demoted_or_unresolved: pd.DataFrame,
    candidates: pd.DataFrame,
    state_membership: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig | None = None,
) -> pd.DataFrame:
    """Complete one otherwise empty assay from unique all-state topology.

    This is the final zero-edge fallback.  It cannot compete with an existing
    direct or unresolved adjacency, and it runs only after frozen rank fusion
    abstained because its winner lacked all-fold practical support.  The
    selected star must be unique in the complete six-feature candidate
    universe and must combine held-out state/Wald support for all three nodes
    with stable collider topology, cross-fitted Bernoulli evidence, and two
    independently calibrated arm tests.
    """

    cfg = config or CrossfitHierarchicalColliderConfig()
    cfg.validate()
    for source, frame in (
        ("decisions", decisions),
        ("current_direct_candidates", current_direct_candidates),
        ("current_demoted_or_unresolved", current_demoted_or_unresolved),
        ("candidates", candidates),
        ("state_membership", state_membership),
    ):
        _assert_observed_only(frame, source)
    _require_columns(
        decisions,
        {
            "source_assay",
            "action",
            "selected_replacement",
            "demote_production_motif",
            "selected_unoriented_adjacency",
            "zero_edge_rank_fusion_status",
            "zero_edge_rank_fusion_abstention_reason",
        },
        "decisions",
    )
    for column in (
        "selected_replacement",
        "demote_production_motif",
        "selected_unoriented_adjacency",
    ):
        _require_strict_boolean_column(decisions, column, "decisions")
    if decisions["source_assay"].astype(str).duplicated().any():
        raise ValueError("decisions contains duplicate assay rows")
    candidate_required = {
        "source_assay",
        "center",
        "outer1",
        "outer2",
        "raw_generative_score",
        "minimum_collider_separation",
        "median_collider_separation",
        "crossfit_joint_gain_over_baseline",
        "crossfit_min_drop_one_gain",
        "crossfit_joint_gain_fold_min",
        "laplace_min_parent_positive_posterior",
        "laplace_fold_min_parent_positive_posterior",
        "laplace_both_positive_fold_support",
        "laplace_both_practical_fold_support",
        "maximum_arm_empirical_positive_p",
        "outer_empirical_null_compatibility_p",
        "minimum_outer_conditional_wald_view_support",
        "two_parent_positive_state_wald_consistency",
        "min_arm_lor",
        "max_arm_empirical_p",
    }
    _require_columns(candidates, candidate_required, "candidates")
    _require_strict_boolean_column(
        candidates,
        "two_parent_positive_state_wald_consistency",
        "candidates",
    )
    candidate_keys = ["source_assay", "center", "outer1", "outer2"]
    if candidates.duplicated(candidate_keys).any():
        raise ValueError("candidates contains duplicate star identities")
    if not current_direct_candidates.empty:
        _require_columns(
            current_direct_candidates,
            {"source_assay"},
            "current_direct_candidates",
        )
    if not current_demoted_or_unresolved.empty:
        _require_columns(
            current_demoted_or_unresolved,
            {"source_assay"},
            "current_demoted_or_unresolved",
        )
    state = _state_membership_map(state_membership)
    output = decisions.copy()
    defaults: dict[str, Any] = {
        "zero_edge_all_state_topology_considered": False,
        "zero_edge_all_state_topology_status": "not_applicable",
        "zero_edge_all_state_topology_reason": "",
        "zero_edge_all_state_topology_candidate_universe_complete": False,
        "zero_edge_all_state_topology_center_state_wald_candidates": 0,
        "zero_edge_all_state_topology_eligible_candidates": 0,
        "zero_edge_all_state_topology_selected": False,
        "zero_edge_all_state_topology_selected_center": "",
        "zero_edge_all_state_topology_selected_outer1": "",
        "zero_edge_all_state_topology_selected_outer2": "",
        "zero_edge_all_state_topology_selected_raw_generative_score": math.nan,
        "zero_edge_all_state_topology_selected_minimum_collider_separation": math.nan,
        "zero_edge_all_state_topology_selected_median_collider_separation": math.nan,
        "zero_edge_all_state_topology_selected_joint_gain": math.nan,
        "zero_edge_all_state_topology_selected_drop_one_gain": math.nan,
        "zero_edge_all_state_topology_selected_fold_joint_gain": math.nan,
        "zero_edge_all_state_topology_selected_parent_posterior": math.nan,
        "zero_edge_all_state_topology_selected_fold_parent_posterior": math.nan,
        "zero_edge_all_state_topology_selected_hierarchical_arm_empirical_p": math.nan,
        "zero_edge_all_state_topology_selected_outer_empirical_null_p": math.nan,
        "zero_edge_all_state_topology_selected_beta_arm_log_odds": math.nan,
        "zero_edge_all_state_topology_selected_beta_arm_empirical_p": math.nan,
    }
    for column, value in defaults.items():
        output[column] = value

    expected_candidate_count = cfg.required_features_per_assay * math.comb(
        cfg.required_features_per_assay - 1,
        2,
    )
    required_abstention_reason = (
        "selected_star_lacked_all_fold_practical_support"
    )
    for index, decision in output.iterrows():
        assay = str(decision["source_assay"])
        reason = ""
        if (
            bool(decision["selected_replacement"])
            or bool(decision["demote_production_motif"])
            or bool(decision["selected_unoriented_adjacency"])
        ):
            reason = "higher_priority_action_selected"
        elif str(decision["action"]) != "retain_current_direct_motif":
            reason = "prior_action_is_not_retention"
        elif str(decision["zero_edge_rank_fusion_status"]) != "abstained":
            reason = "rank_fusion_did_not_abstain"
        elif (
            str(decision["zero_edge_rank_fusion_abstention_reason"])
            != required_abstention_reason
        ):
            reason = "rank_fusion_abstained_for_a_different_reason"
        elif (
            not current_direct_candidates.empty
            and current_direct_candidates["source_assay"].astype(str).eq(assay).any()
        ):
            reason = "assay_has_nonempty_direct_production"
        elif (
            not current_demoted_or_unresolved.empty
            and current_demoted_or_unresolved["source_assay"]
            .astype(str)
            .eq(assay)
            .any()
        ):
            reason = "assay_has_existing_unresolved_adjacency"
        else:
            output.at[index, "zero_edge_all_state_topology_considered"] = True
            group = candidates.loc[
                candidates["source_assay"].astype(str).eq(assay)
            ].copy()
            nodes = set(group["center"].astype(str))
            nodes.update(group["outer1"].astype(str))
            nodes.update(group["outer2"].astype(str))
            candidate_universe_complete = (
                len(group) == expected_candidate_count
                and len(nodes) == cfg.required_features_per_assay
            )
            output.at[
                index,
                "zero_edge_all_state_topology_candidate_universe_complete",
            ] = candidate_universe_complete
            missing_state = sorted(node for node in nodes if node not in state)
            if missing_state:
                raise ValueError(
                    f"state_membership is missing candidate nodes in {assay}: "
                    f"{missing_state}"
                )
            center_state_wald = group["center"].astype(str).map(
                lambda node: (
                    float(state[node].oof_partial_state_fit) > 0.0
                    and float(state[node].conditional_wald_view_support) >= 1.0
                )
            )
            output.at[
                index,
                "zero_edge_all_state_topology_center_state_wald_candidates",
            ] = int(center_state_wald.sum())
            eligible = _ordered_candidates(
                group.loc[
                    center_state_wald.astype(bool)
                    & group["two_parent_positive_state_wald_consistency"].astype(bool)
                    & group["minimum_collider_separation"].ge(
                        cfg.zero_edge_all_state_topology_minimum_collider_separation
                    )
                    & group["median_collider_separation"].ge(
                        cfg.zero_edge_all_state_topology_minimum_median_collider_separation
                    )
                    & group["raw_generative_score"].ge(
                        cfg.zero_edge_all_state_topology_minimum_raw_generative_score
                    )
                    & group["crossfit_joint_gain_over_baseline"].ge(
                        cfg.zero_edge_all_state_topology_minimum_joint_gain
                    )
                    & group["crossfit_min_drop_one_gain"].ge(
                        cfg.zero_edge_all_state_topology_minimum_drop_one_gain
                    )
                    & group["crossfit_joint_gain_fold_min"].ge(
                        cfg.zero_edge_all_state_topology_minimum_fold_joint_gain
                    )
                    & group["laplace_min_parent_positive_posterior"].ge(
                        cfg.zero_edge_all_state_topology_minimum_parent_posterior
                    )
                    & group["laplace_fold_min_parent_positive_posterior"].ge(
                        cfg.zero_edge_all_state_topology_minimum_fold_parent_posterior
                    )
                    & group["laplace_both_positive_fold_support"].ge(
                        cfg.zero_edge_all_state_topology_minimum_both_positive_fold_support
                    )
                    & group["laplace_both_practical_fold_support"].ge(
                        cfg.zero_edge_all_state_topology_minimum_both_practical_fold_support
                    )
                    & group["maximum_arm_empirical_positive_p"].le(
                        cfg.zero_edge_all_state_topology_maximum_hierarchical_arm_empirical_p
                    )
                    & group["outer_empirical_null_compatibility_p"].ge(
                        cfg.zero_edge_all_state_topology_minimum_outer_empirical_null_compatibility_p
                    )
                    & group["minimum_outer_conditional_wald_view_support"].ge(
                        cfg.zero_edge_all_state_topology_minimum_outer_wald_view_support
                    )
                    & group["min_arm_lor"].ge(
                        cfg.zero_edge_all_state_topology_minimum_beta_arm_log_odds
                    )
                    & group["max_arm_empirical_p"].le(
                        cfg.zero_edge_all_state_topology_maximum_beta_arm_empirical_p
                    )
                ]
            ) if candidate_universe_complete else pd.DataFrame()
            output.at[
                index, "zero_edge_all_state_topology_eligible_candidates"
            ] = int(len(eligible))
            if not candidate_universe_complete:
                reason = "candidate_universe_incomplete"
            elif len(eligible) != 1:
                reason = "eligible_all_state_topology_star_count_is_not_exactly_one"
            else:
                selected = eligible.iloc[0]
                output.at[index, "selected_replacement"] = True
                output.at[index, "selected_center"] = str(selected.center)
                output.at[index, "selected_outer1"] = str(selected.outer1)
                output.at[index, "selected_outer2"] = str(selected.outer2)
                output.at[index, "selected_raw_generative_score"] = float(
                    selected.raw_generative_score
                )
                output.at[index, "selection_runner_margin"] = math.inf
                output.at[index, "selected_score_minus_production_score"] = math.nan
                output.at[index, "action"] = (
                    ZERO_EDGE_ALL_STATE_TOPOLOGY_COMPLETION_ACTION
                )
                output.at[index, "zero_edge_all_state_topology_selected"] = True
                output.at[index, "zero_edge_all_state_topology_status"] = "selected"
                output.at[index, "zero_edge_all_state_topology_reason"] = (
                    "one_complete_all_state_star_passed_cross_resolution_topology_"
                    "and_empirical_null_gates"
                )
                for column, value in (
                    ("zero_edge_all_state_topology_selected_center", str(selected.center)),
                    ("zero_edge_all_state_topology_selected_outer1", str(selected.outer1)),
                    ("zero_edge_all_state_topology_selected_outer2", str(selected.outer2)),
                    (
                        "zero_edge_all_state_topology_selected_raw_generative_score",
                        float(selected.raw_generative_score),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_minimum_collider_separation",
                        float(selected.minimum_collider_separation),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_median_collider_separation",
                        float(selected.median_collider_separation),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_joint_gain",
                        float(selected.crossfit_joint_gain_over_baseline),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_drop_one_gain",
                        float(selected.crossfit_min_drop_one_gain),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_fold_joint_gain",
                        float(selected.crossfit_joint_gain_fold_min),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_parent_posterior",
                        float(selected.laplace_min_parent_positive_posterior),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_fold_parent_posterior",
                        float(selected.laplace_fold_min_parent_positive_posterior),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_hierarchical_arm_empirical_p",
                        float(selected.maximum_arm_empirical_positive_p),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_outer_empirical_null_p",
                        float(selected.outer_empirical_null_compatibility_p),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_beta_arm_log_odds",
                        float(selected.min_arm_lor),
                    ),
                    (
                        "zero_edge_all_state_topology_selected_beta_arm_empirical_p",
                        float(selected.max_arm_empirical_p),
                    ),
                ):
                    output.at[index, column] = value
                continue
        output.at[index, "zero_edge_all_state_topology_status"] = (
            "abstained"
            if bool(output.at[index, "zero_edge_all_state_topology_considered"])
            else "not_applicable"
        )
        output.at[index, "zero_edge_all_state_topology_reason"] = reason

    for column in (
        "zero_edge_all_state_topology_considered",
        "zero_edge_all_state_topology_candidate_universe_complete",
        "zero_edge_all_state_topology_selected",
    ):
        _require_strict_boolean_column(output, column, "annotated decisions")
    return output


def _aggregated_beta_map(
    beta_vote_evidence: pd.DataFrame,
) -> dict[tuple[str, tuple[str, str]], dict[str, Any]]:
    _require_strict_boolean_column(
        beta_vote_evidence, "selected", "beta_vote_evidence"
    )
    aggregated = aggregate_vote_evidence(beta_vote_evidence)
    return {
        (
            str(row["source_assay"]),
            canonical_edge(str(row["node1"]), str(row["node2"])),
        ): row
        for row in aggregated.to_dict(orient="records")
    }


def _new_direct_row(
    assay: str,
    pair: tuple[str, str],
    center: str,
    decision: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    rank_fusion = str(decision.action) == (
        "zero_edge_state_gated_rank_fusion_completion"
    )
    if rank_fusion:
        wedge_score = float(decision.zero_edge_rank_fusion_selected_center_score)
        runner_score = float(decision.zero_edge_rank_fusion_runner_center_score)
        wedge_margin = float(decision.zero_edge_rank_fusion_center_score_margin)
        orientation_source = "crossfit_hierarchical_zero_edge_state_gated_rank_fusion"
        interpretation = (
            "cross-fitted within-assay zero-edge state-gated rank-fusion candidate"
        )
    elif str(decision.action) == RESIDUAL_FIRST_REPLACEMENT_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = float(
            decision.selected_raw_generative_score
            - decision.selection_runner_margin
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = "crossfit_hierarchical_residual_first_gate"
        interpretation = (
            "cross-fitted within-assay all-fold residual-first collider candidate"
        )
    elif str(decision.action) == MULTI_COMPONENT_COLLAPSE_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = float(
            decision.selected_raw_generative_score - decision.selection_runner_margin
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_multi_component_sparsification"
        )
        interpretation = (
            "cross-fitted within-assay unique existing star retained from a "
            "multi-component direct production"
        )
    elif str(decision.action) == ZERO_EDGE_STATE_CONSISTENT_RUNNER_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_zero_edge_state_consistent_runner"
        )
        interpretation = (
            "cross-fitted within-assay zero-edge same-center runner passing "
            "state, fold, posterior, and empirical-null gates"
        )
    elif str(decision.action) == ZERO_EDGE_RESCUE_ANCHORED_COMPLETION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_zero_edge_rescue_anchored_completion"
        )
        interpretation = (
            "unique rescue-anchored within-assay star passing state, fold, "
            "posterior, and empirical-null gates"
        )
    elif str(decision.action) == ZERO_EDGE_RESCUE_ORIENTATION_COMPLETION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_zero_edge_unanimous_rescue_orientation"
        )
        interpretation = (
            "sparse within-assay rescue completion whose center won every "
            "reverse-center context and whose unique star passed uncertainty gates"
        )
    elif str(decision.action) == ZERO_EDGE_DOMINANT_RESCUE_COMPLETION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_zero_edge_dominant_rescue_completion"
        )
        interpretation = (
            "sparse within-assay completion anchored by a beta-binomial rescue "
            "that decisively exceeded its competing rescue arm"
        )
    elif str(decision.action) == ZERO_EDGE_ALL_STATE_TOPOLOGY_COMPLETION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = math.nan
        wedge_margin = math.inf
        orientation_source = (
            "crossfit_hierarchical_zero_edge_all_state_topology_completion"
        )
        interpretation = (
            "unique zero-edge within-assay collider candidate supported by "
            "held-out state/Wald evidence for all three nodes, cross-resolution "
            "residual topology, and empirical-null-calibrated sparse arms"
        )
    elif str(decision.action) == ZERO_EDGE_SPOKE_CONSENSUS_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_zero_edge_beta_spoke_consensus"
        )
        interpretation = (
            "state-stable within-assay star defined by the two strongest arms "
            "of a strict unresolved three-spoke component"
        )
    elif str(decision.action) == WEAK_MULTICENTER_REPLACEMENT_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_multicenter_uncertainty_validation"
        )
        interpretation = (
            "unique state-consistent replacement for a weak observed-only "
            "multicenter resolution"
        )
    elif str(decision.action) == CENTRALITY_SAME_SKELETON_REPLACEMENT_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_centrality_original_skeleton_validation"
        )
        interpretation = (
            "state-consistent orientation of the stable pre-centrality "
            "three-node skeleton"
        )
    elif str(decision.action) == CENTRALITY_PRECOMPLETION_REVERSION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_centrality_precompletion_reversion"
        )
        interpretation = (
            "stable observed pre-centrality completion retained after the "
            "provisional centrality proposal failed uncertainty validation"
        )
    elif str(decision.action) == DUAL_ANCHOR_RESIDUAL_REPLACEMENT_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_dual_anchor_residual_validation"
        )
        interpretation = (
            "unique anchor-preserving replacement for an observed completion "
            "contradicted by within-assay residual collider topology"
        )
    elif str(decision.action) == STATE_INCONSISTENT_NEIGHBOR_REPLACEMENT_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_state_consistent_neighbor_validation"
        )
        interpretation = (
            "unique adjacent within-assay star replacing a production motif "
            "that lacked state-Wald and outer-pair support"
        )
    elif str(decision.action) == STATE_INCONSISTENT_SAME_CENTER_SUBSTITUTION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_same_center_state_parent_substitution"
        )
        interpretation = (
            "unique state-consistent parent substitution retaining the "
            "production center and stable arm across aggregation views"
        )
    elif str(decision.action) == STATE_CENTER_SIBLING_ROTATION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_state_center_loading_sibling_rotation"
        )
        interpretation = (
            "unique adjacent within-assay star separating shared-state "
            "surrogacy from direct structure through a stable loading-"
            "equivalent sibling substitution"
        )
    elif str(decision.action) == STATE_LOADED_PARENT_ROTATION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_state_loaded_parent_rotation"
        )
        interpretation = (
            "unique residual-supported within-assay star that rotates a "
            "state-loaded production parent into the collider center"
        )
    elif str(decision.action) == AMBIGUOUS_SKELETON_REORIENTATION_ACTION:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = (
            "crossfit_hierarchical_same_skeleton_residual_orientation"
        )
        interpretation = (
            "unique state-consistent orientation on the retained three-node "
            "skeleton selected by conditional collider separation"
        )
    else:
        wedge_score = float(decision.selected_raw_generative_score)
        runner_score = (
            float(
                decision.selected_raw_generative_score
                - decision.selection_runner_margin
            )
            if np.isfinite(decision.selection_runner_margin)
            else math.nan
        )
        wedge_margin = float(decision.selection_runner_margin)
        orientation_source = "crossfit_hierarchical_raw_evidence_gate"
        interpretation = "cross-fitted within-assay two-parent collider candidate"
    return {
        "source_assay": assay,
        "node1": pair[0],
        "endpoint1": "ARROW" if pair[0] == center else "TAIL",
        "node2": pair[1],
        "endpoint2": "ARROW" if pair[1] == center else "TAIL",
        "edge_type": DIRECT_EDGE_TYPE,
        "orientation_source": orientation_source,
        "collider_center": center,
        "component_id": f"crossfit_hierarchical__{assay}",
        "wedge_score": wedge_score,
        "runner_up_wedge_score": runner_score,
        "wedge_margin": wedge_margin,
        "selected_vote_fraction": float(evidence["selected_vote_fraction"]),
        "mean_empirical_p_positive": float(evidence["mean_empirical_p_positive"]),
        "mean_pooled_log_odds": float(evidence["mean_pooled_log_odds"]),
        "interpretation": interpretation,
        # These rows are produced by the crossfit layer, not by the earlier
        # observed-structure refinement.  The explicit empty marker prevents a
        # stale refinement action from being copied across a replacement.
        "observed_refinement_action": "",
        "crossfit_hierarchical_action": str(decision.action),
    }


def _new_unoriented_direct_row(
    assay: str,
    pair: tuple[str, str],
    decision: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_assay": assay,
        "node1": pair[0],
        "endpoint1": "CIRCLE",
        "node2": pair[1],
        "endpoint2": "CIRCLE",
        "edge_type": DIRECT_EDGE_TYPE,
        "orientation_source": "crossfit_hierarchical_unique_rescue_rank_overlap",
        "collider_center": "",
        "component_id": f"crossfit_hierarchical__{assay}",
        "wedge_score": float(
            decision.zero_edge_rank_fusion_selected_center_score
        ),
        "runner_up_wedge_score": float(
            decision.zero_edge_rank_fusion_runner_center_score
        ),
        "wedge_margin": float(
            decision.zero_edge_rank_fusion_center_score_margin
        ),
        "selected_vote_fraction": float(evidence["selected_vote_fraction"]),
        "mean_empirical_p_positive": float(evidence["mean_empirical_p_positive"]),
        "mean_pooled_log_odds": float(evidence["mean_pooled_log_odds"]),
        "interpretation": (
            "cross-fitted within-assay adjacency candidate only; collider center "
            "and direction unresolved"
        ),
        "observed_refinement_action": "",
        "crossfit_hierarchical_action": str(decision.action),
        "unoriented_adjacency_only": True,
        "direction_claimed": False,
        "collider_center_claimed": False,
    }


def _apply_decisions(
    *,
    decisions: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    current_demoted_or_unresolved: pd.DataFrame,
    beta_vote_evidence: pd.DataFrame,
    base_edge_views: Mapping[str, pd.DataFrame],
    config: CrossfitHierarchicalColliderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_strict_boolean_column(decisions, "selected_replacement", "decisions")
    _require_strict_boolean_column(decisions, "demote_production_motif", "decisions")
    _require_strict_boolean_column(
        decisions,
        "selected_unoriented_adjacency",
        "decisions",
    )
    action_count = (
        decisions[
            [
                "selected_replacement",
                "demote_production_motif",
                "selected_unoriented_adjacency",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    if bool(action_count.gt(1).any()):
        raise ValueError(
            "A decision cannot replace and demote or select an unoriented adjacency "
            "at the same time"
        )
    acted = decisions.loc[
        decisions["selected_replacement"]
        | decisions["demote_production_motif"]
        | decisions["selected_unoriented_adjacency"]
    ].copy()
    if acted.empty:
        return (
            current_direct_candidates.copy().reset_index(drop=True),
            current_demoted_or_unresolved.copy().reset_index(drop=True),
        )
    _require_columns(acted, {"source_assay", "action"}, "decisions")
    demotions = acted.loc[acted["demote_production_motif"]]
    if not demotions.empty:
        _require_columns(
            demotions,
            {"production_center", "production_outer1", "production_outer2"},
            "demotion decisions",
        )
        invalid_action = ~demotions["action"].astype(str).isin(
            {
                RETAINED_ORIENTATION_DEMOTION_ACTION,
                WEAK_MULTICENTER_DEMOTION_ACTION,
                CENTRALITY_UNRECOVERABLE_DEMOTION_ACTION,
            }
        )
        if bool(invalid_action.any()):
            raise ValueError(
                "Orientation demotion decisions must carry the explicit demotion action"
            )
    replacements = acted.loc[acted["selected_replacement"]]
    if not replacements.empty:
        _require_columns(
            replacements,
            {"selected_center", "selected_outer1", "selected_outer2"},
            "replacement decisions",
        )
        if bool(
            replacements["action"]
            .astype(str)
            .eq(RETAINED_ORIENTATION_DEMOTION_ACTION)
            .any()
        ):
            raise ValueError("Replacement decisions cannot carry the demotion action")
    unoriented_selections = acted.loc[acted["selected_unoriented_adjacency"]]
    if not unoriented_selections.empty:
        _require_columns(
            unoriented_selections,
            {"unoriented_adjacency_node1", "unoriented_adjacency_node2"},
            "unoriented adjacency decisions",
        )
        invalid_action = ~unoriented_selections["action"].astype(str).eq(
            UNORIENTED_RESCUE_OVERLAP_ACTION
        )
        if bool(invalid_action.any()):
            raise ValueError(
                "Unoriented adjacency decisions must carry the explicit action"
            )
    rewritten_assays = set(
        acted.loc[
            acted["selected_replacement"] | acted["demote_production_motif"],
            "source_assay",
        ].astype(str)
    )
    direct = current_direct_candidates.loc[
        ~current_direct_candidates["source_assay"].astype(str).isin(rewritten_assays)
    ].copy() if not current_direct_candidates.empty else pd.DataFrame()
    unresolved = current_demoted_or_unresolved.copy()
    beta = _aggregated_beta_map(beta_vote_evidence)
    base_counts = _base_support(base_edge_views)
    new_direct: list[dict[str, Any]] = []
    displaced: list[dict[str, Any]] = []
    for decision in acted.itertuples(index=False):
        assay = str(decision.source_assay)
        is_replacement = bool(decision.selected_replacement)
        is_demotion = bool(decision.demote_production_motif)
        is_unoriented_selection = bool(decision.selected_unoriented_adjacency)
        selected_pairs: set[tuple[str, str]] = set()
        if is_replacement:
            center = str(decision.selected_center)
            selected_pairs = {
                canonical_edge(center, str(decision.selected_outer1)),
                canonical_edge(center, str(decision.selected_outer2)),
            }
            for pair in sorted(selected_pairs):
                evidence = beta.get((assay, pair))
                if evidence is None:
                    raise ValueError("Selected direct arm lacks beta evidence")
                new_direct.append(
                    _new_direct_row(assay, pair, center, decision, evidence)
                )
        elif is_unoriented_selection:
            pair = canonical_edge(
                str(decision.unoriented_adjacency_node1),
                str(decision.unoriented_adjacency_node2),
            )
            evidence = beta.get((assay, pair))
            if evidence is None:
                raise ValueError("Selected unoriented adjacency lacks beta evidence")
            selected_pairs = {pair}
            new_direct.append(
                _new_unoriented_direct_row(assay, pair, decision, evidence)
            )
        if selected_pairs and not unresolved.empty:
            _require_columns(
                unresolved,
                {"source_assay", "node1", "node2"},
                "current_demoted_or_unresolved",
            )
            unresolved = unresolved.loc[
                ~unresolved.apply(
                    lambda row: str(row["source_assay"]) == assay
                    and canonical_edge(str(row["node1"]), str(row["node2"]))
                    in selected_pairs,
                    axis=1,
                )
            ].copy()
        assay_current = current_direct_candidates.loc[
            current_direct_candidates["source_assay"].astype(str).eq(assay)
        ] if not current_direct_candidates.empty else pd.DataFrame()
        if is_demotion:
            expected_pairs = {
                canonical_edge(
                    str(decision.production_center),
                    str(decision.production_outer1),
                ),
                canonical_edge(
                    str(decision.production_center),
                    str(decision.production_outer2),
                ),
            }
            observed_pairs = {
                canonical_edge(str(row.node1), str(row.node2))
                for row in assay_current.itertuples(index=False)
            }
            if observed_pairs != expected_pairs:
                raise ValueError(
                    "Orientation demotion does not match the exact production motif"
                )
        for row in (
            assay_current.to_dict(orient="records")
            if is_replacement or is_demotion
            else []
        ):
            pair = canonical_edge(str(row["node1"]), str(row["node2"]))
            if pair in selected_pairs:
                continue
            evidence = beta.get((assay, pair), {})
            original_observed_action = str(
                row.get("observed_refinement_action", "")
            ).strip()
            displaced.append(
                {
                    "source_assay": assay,
                    "node1": pair[0],
                    "endpoint1": "CIRCLE",
                    "node2": pair[1],
                    "endpoint2": "CIRCLE",
                    "edge_type": UNRESOLVED_EDGE_TYPE,
                    "component_id": f"crossfit_hierarchical__{assay}",
                    "was_base_selected": base_counts.get(pair, 0)
                    >= config.minimum_base_view_count,
                    "was_rescue_selected": bool(
                        evidence
                        and float(evidence["selected_vote_fraction"])
                        >= UNORIENTED_RESCUE_MIN_SELECTED_VOTE_FRACTION
                        and float(evidence["mean_empirical_p_positive"])
                        <= UNORIENTED_RESCUE_MAX_MEAN_EMPIRICAL_P_POSITIVE
                        and float(evidence["mean_pooled_log_odds"])
                        >= UNORIENTED_RESCUE_MIN_MEAN_POOLED_LOG_ODDS
                    ),
                    "mean_empirical_p_positive": float(
                        evidence.get("mean_empirical_p_positive", math.nan)
                    ),
                    "mean_pooled_log_odds": float(
                        evidence.get("mean_pooled_log_odds", math.nan)
                    ),
                    "reason": (
                        "weak_multicenter_resolution_without_validated_star"
                        if is_demotion
                        and str(decision.action) == WEAK_MULTICENTER_DEMOTION_ACTION
                        else (
                            "weak_centrality_reorientation_without_recoverable_completion"
                            if is_demotion
                            and str(decision.action)
                            == CENTRALITY_UNRECOVERABLE_DEMOTION_ACTION
                            else (
                                "retained_orientation_ambiguous_production_motif"
                                if is_demotion
                                else "displaced_by_crossfit_hierarchical_collider"
                            )
                        )
                    ),
                    "observed_refinement_action": "",
                    "displaced_observed_refinement_action": (
                        original_observed_action
                    ),
                    "crossfit_hierarchical_action": str(decision.action),
                }
            )
    if new_direct:
        direct = pd.concat([direct, pd.DataFrame(new_direct)], ignore_index=True, sort=False)
    if displaced:
        additions = pd.DataFrame(displaced)
        if not unresolved.empty:
            existing = {
                (str(row.source_assay), canonical_edge(str(row.node1), str(row.node2)))
                for row in unresolved.itertuples(index=False)
            }
            additions = additions.loc[
                [
                    (str(row.source_assay), canonical_edge(str(row.node1), str(row.node2)))
                    not in existing
                    for row in additions.itertuples(index=False)
                ]
            ]
        unresolved = pd.concat([unresolved, additions], ignore_index=True, sort=False)
    if not direct.empty:
        pairs = direct.apply(
            lambda row: (
                str(row["source_assay"]),
                canonical_edge(str(row["node1"]), str(row["node2"])),
            ),
            axis=1,
        )
        if pairs.duplicated().any():
            raise ValueError("Resolved direct candidates contain duplicate adjacencies")
        direct = direct.sort_values(["source_assay", "node1", "node2"]).reset_index(drop=True)
    if not unresolved.empty:
        unresolved_pairs = unresolved.apply(
            lambda row: (
                str(row["source_assay"]),
                canonical_edge(str(row["node1"]), str(row["node2"])),
            ),
            axis=1,
        )
        if unresolved_pairs.duplicated().any():
            raise ValueError("Resolved unresolved table contains duplicate adjacencies")
        unresolved = unresolved.sort_values(
            ["source_assay", "node1", "node2"]
        ).reset_index(drop=True)
    return direct, unresolved


def _mapping_hashes(mapping: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return {
        str(view_id): deterministic_frame_sha256(frame)
        for view_id, frame in sorted(mapping.items())
    }


def resolve_crossfit_hierarchical_colliders_v4(
    *,
    world_evidence: CrossfitColliderWorldEvidence,
    beta_vote_evidence: pd.DataFrame,
    base_edge_views: Mapping[str, pd.DataFrame],
    residual_views: Mapping[str, pd.DataFrame],
    feature_model_views: Mapping[str, pd.DataFrame],
    loading_pair_views: Mapping[str, pd.DataFrame],
    state_membership: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    current_demoted_or_unresolved: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig | None = None,
) -> CrossfitHierarchicalColliderResult:
    """Resolve one current candidate using reusable once-per-world evidence."""

    cfg = config or CrossfitHierarchicalColliderConfig()
    cfg.validate()
    for source, frame in (
        ("beta_vote_evidence", beta_vote_evidence),
        ("state_membership", state_membership),
        ("current_direct_candidates", current_direct_candidates),
        ("current_demoted_or_unresolved", current_demoted_or_unresolved),
        ("world_evidence.generative_candidates", world_evidence.generative_candidates),
    ):
        _assert_observed_only(frame, source)
    if world_evidence.provenance.get("config") != asdict(cfg):
        raise ValueError("world_evidence was fitted with a different config")
    expected_world_hash = world_evidence.provenance.get("generative_candidates_sha256")
    if deterministic_frame_sha256(world_evidence.generative_candidates) != expected_world_hash:
        raise ValueError("world_evidence candidate hash does not match its provenance")
    base = _build_exhaustive_base_candidates(
        beta_vote_evidence=beta_vote_evidence,
        base_edge_views=base_edge_views,
        residual_views=residual_views,
        feature_model_views=feature_model_views,
        loading_pair_views=loading_pair_views,
        state_membership=state_membership,
        current_direct_candidates=current_direct_candidates,
        config=cfg,
    )
    keys = ["source_assay", "center", "outer1", "outer2"]
    candidates = base.merge(
        world_evidence.generative_candidates,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).sort_values(keys).reset_index(drop=True)
    if len(candidates) != len(base) or len(candidates) != len(
        world_evidence.generative_candidates
    ):
        raise ValueError("Base and fitted candidate universes do not match")
    candidates = _annotate_state_center_sibling_rotation_evidence(
        candidates=candidates,
        beta_vote_evidence=beta_vote_evidence,
        loading_pair_views=loading_pair_views,
        current_direct_candidates=current_direct_candidates,
        config=cfg,
    )
    candidates["raw_generative_score"] = _raw_score(candidates)
    decisions = select_crossfit_hierarchical_collider_replacements(
        candidates=candidates,
        state_membership=state_membership,
        current_direct_candidates=current_direct_candidates,
        config=cfg,
    )
    decisions = _annotate_zero_edge_spoke_consensus(
        decisions=decisions,
        current_direct_candidates=current_direct_candidates,
        current_demoted_or_unresolved=current_demoted_or_unresolved,
        candidates=candidates,
        config=cfg,
    )
    decisions = _annotate_unique_rescue_rank_overlap_adjacencies(
        decisions=decisions,
        current_direct_candidates=current_direct_candidates,
        current_demoted_or_unresolved=current_demoted_or_unresolved,
        candidates=candidates,
        config=cfg,
    )
    decisions = _annotate_zero_edge_all_state_topology_completion(
        decisions=decisions,
        current_direct_candidates=current_direct_candidates,
        current_demoted_or_unresolved=current_demoted_or_unresolved,
        candidates=candidates,
        state_membership=state_membership,
        config=cfg,
    )
    direct, unresolved = _apply_decisions(
        decisions=decisions,
        current_direct_candidates=current_direct_candidates,
        current_demoted_or_unresolved=current_demoted_or_unresolved,
        beta_vote_evidence=beta_vote_evidence,
        base_edge_views=base_edge_views,
        config=cfg,
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "raw_evidence_tier1u_tier1r_residual_first_zero_edge_all_fold_"
            "rank_fusion_beta_spoke_consensus_dominant_or_unique_rescue_anchored_or_"
            "unanimously_oriented_sparse_completion_rank_overlap_unoriented_"
            "adjacency_all_state_topology_completion_dual_anchor_state_consistent_"
            "neighbor_same_center_parent_"
            "substitution_state_center_sibling_rotation_state_parent_rotation_"
            "and_retained_"
            "orientation_uncertainty_and_same_skeleton_residual_validation"
        ),
        "config": asdict(cfg),
        "zero_edge_rank_fusion_model": {
            **_zero_edge_rank_fusion_model_payload(),
            "sha256": _zero_edge_rank_fusion_model_sha256(),
        },
        "world_evidence_sha256": expected_world_hash,
        "world_evidence_reused_without_refit": True,
        "input_sha256": {
            "beta_vote_evidence": deterministic_frame_sha256(beta_vote_evidence),
            "base_edge_views": _mapping_hashes(base_edge_views),
            "residual_views": _mapping_hashes(residual_views),
            "feature_model_views": _mapping_hashes(feature_model_views),
            "loading_pair_views": _mapping_hashes(loading_pair_views),
            "state_membership": deterministic_frame_sha256(state_membership),
            "current_direct_candidates": deterministic_frame_sha256(
                current_direct_candidates
            ),
            "current_demoted_or_unresolved": deterministic_frame_sha256(
                current_demoted_or_unresolved
            ),
        },
        "output_sha256": {
            "candidates": deterministic_frame_sha256(candidates),
            "decisions": deterministic_frame_sha256(decisions),
            "direct_candidates": deterministic_frame_sha256(direct),
            "demoted_or_unresolved": deterministic_frame_sha256(unresolved),
        },
        "n_selected_replacements": int(
            decisions["selected_replacement"].astype(bool).sum()
        ),
        "n_tier1r_residual_first_replacements": int(
            decisions["tier1r_selected"].astype(bool).sum()
        ),
        "n_selected_unoriented_adjacencies": int(
            decisions["selected_unoriented_adjacency"].astype(bool).sum()
        ),
        "n_zero_edge_rescue_anchored_completions": int(
            decisions["rescue_anchored_completion_selected"].astype(bool).sum()
        ),
        "n_zero_edge_rescue_orientation_completions": int(
            decisions["rescue_orientation_completion_selected"]
            .astype(bool)
            .sum()
        ),
        "n_zero_edge_dominant_rescue_completions": int(
            decisions["dominant_rescue_completion_selected"].astype(bool).sum()
        ),
        "n_zero_edge_spoke_consensus_completions": int(
            decisions["zero_edge_spoke_consensus_selected"].astype(bool).sum()
        ),
        "n_zero_edge_all_state_topology_completions": int(
            decisions["zero_edge_all_state_topology_selected"].astype(bool).sum()
        ),
        "n_ambiguous_skeleton_reorientations": int(
            decisions["ambiguous_skeleton_reorientation_selected"]
            .astype(bool)
            .sum()
        ),
        "n_state_inconsistent_neighbor_replacements": int(
            decisions["neighbor_star_validation_selected"].astype(bool).sum()
        ),
        "n_state_inconsistent_same_center_substitutions": int(
            decisions["same_center_substitution_selected"].astype(bool).sum()
        ),
        "n_state_center_sibling_rotations": int(
            decisions["state_center_sibling_rotation_selected"]
            .astype(bool)
            .sum()
        ),
        "n_state_loaded_parent_rotations": int(
            decisions["state_parent_rotation_selected"].astype(bool).sum()
        ),
        "n_retained_orientation_demotions": int(
            decisions["demote_production_motif"].astype(bool).sum()
        ),
        "displaced_base_edges_remain_unresolved": True,
        "nonselected_displaced_rescue_only_edges_are_diagnostic": True,
        "orientation_demotions_preserve_unresolved_adjacencies": True,
        "unoriented_adjacency_requires_empty_production": True,
        "unoriented_adjacency_requires_exactly_one_rescue_only_unresolved": True,
        "unoriented_adjacency_requires_upstream_rescue_selection": True,
        "unoriented_adjacency_min_selected_vote_fraction": (
            UNORIENTED_RESCUE_MIN_SELECTED_VOTE_FRACTION
        ),
        "unoriented_adjacency_requires_frozen_rank_winner_arm_overlap": True,
        "unoriented_adjacency_max_mean_empirical_p_positive": (
            UNORIENTED_RESCUE_MAX_MEAN_EMPIRICAL_P_POSITIVE
        ),
        "unoriented_adjacency_min_mean_pooled_log_odds": (
            UNORIENTED_RESCUE_MIN_MEAN_POOLED_LOG_ODDS
        ),
        "unoriented_adjacency_direction_claimed": False,
        "unoriented_adjacency_collider_center_claimed": False,
        "unoriented_adjacency_runtime_refit": False,
        "rescue_anchored_completion_requires_complete_candidate_universe": True,
        "rescue_anchored_completion_requires_unique_eligible_star": True,
        "rescue_anchored_completion_requires_state_and_uncertainty_gates": True,
        "rescue_orientation_completion_requires_unanimous_reverse_center_gain": True,
        "rescue_orientation_completion_requires_unique_sparse_candidate": True,
        "dominant_rescue_completion_requires_two_competing_rescue_arms": True,
        "dominant_rescue_completion_requires_beta_and_direction_margins": True,
        "dominant_rescue_completion_requires_unique_sparse_candidate": True,
        "zero_edge_spoke_consensus_requires_exact_three_spoke_component": True,
        "zero_edge_spoke_consensus_requires_beta_arm_margin": True,
        "zero_edge_spoke_consensus_requires_state_and_uncertainty_gates": True,
        "zero_edge_all_state_topology_requires_empty_direct_production": True,
        "zero_edge_all_state_topology_requires_empty_unresolved_layer": True,
        "zero_edge_all_state_topology_requires_complete_candidate_universe": True,
        "zero_edge_all_state_topology_requires_all_three_state_wald_positive": True,
        "zero_edge_all_state_topology_requires_unique_empirical_null_candidate": True,
        "preserve_in_consensus_override_used": False,
        "paired_cells_used": False,
        "cross_assay_coordinate_registration_required": False,
        "scorer_information_used": False,
    }
    return CrossfitHierarchicalColliderResult(
        candidates=candidates,
        decisions=decisions,
        direct_candidates=direct,
        demoted_or_unresolved=unresolved,
        provenance=provenance,
    )


def infer_crossfit_hierarchical_colliders_v4(
    *,
    assay_cell_tables: Mapping[str, pd.DataFrame],
    assay_local_embedding: pd.DataFrame,
    randomization_key: str | int,
    beta_vote_evidence: pd.DataFrame,
    base_edge_views: Mapping[str, pd.DataFrame],
    residual_views: Mapping[str, pd.DataFrame],
    feature_model_views: Mapping[str, pd.DataFrame],
    loading_pair_views: Mapping[str, pd.DataFrame],
    state_membership: pd.DataFrame,
    current_direct_candidates: pd.DataFrame,
    current_demoted_or_unresolved: pd.DataFrame,
    config: CrossfitHierarchicalColliderConfig | None = None,
) -> CrossfitHierarchicalColliderResult:
    """Convenience wrapper; formal benchmarks should fit world evidence once."""

    cfg = config or CrossfitHierarchicalColliderConfig()
    evidence = fit_crossfit_collider_world_evidence(
        assay_cell_tables=assay_cell_tables,
        assay_local_embedding=assay_local_embedding,
        randomization_key=randomization_key,
        config=cfg,
    )
    return resolve_crossfit_hierarchical_colliders_v4(
        world_evidence=evidence,
        beta_vote_evidence=beta_vote_evidence,
        base_edge_views=base_edge_views,
        residual_views=residual_views,
        feature_model_views=feature_model_views,
        loading_pair_views=loading_pair_views,
        state_membership=state_membership,
        current_direct_candidates=current_direct_candidates,
        current_demoted_or_unresolved=current_demoted_or_unresolved,
        config=cfg,
    )
