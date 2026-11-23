#!/usr/bin/env python3
"""Select conservative assay-local loading pairs anchored by a collider motif.

For each assay, a frozen typed direct graph must contain exactly two directed
candidate edges with one common collider center and two distinct parents.  A
candidate shared-state group may then contain one collider parent and one
feature outside that three-node motif.  The two parents are compared on the
same loading views.  The frozen baseline requires the outside feature to pass
the independently frozen cross-assay state-membership model, a strong lower
loading cosine, and a relative parent margin.  When that baseline abstains, the
existing formal-state rescue is applied unchanged: p <= 0.125, the same parent
winning all six views, and a unique candidate.  Only after the complete current
production rule still abstains may a new p <= 0.50 tiered rescue run.  Formal
cross-assay state candidates form its first tier.  Only when that tier is empty
does a second tier admit siblings with assay-global maxT and six-of-six
conditional-Wald support but without feature-global maxT confirmation: the
latter belongs to the separate cross-assay membership layer.  Neither rescue
has an absolute cosine cutoff.  The tiered p <= 0.50 condition is interpreted
only as repeated non-null-domination, not as a per-view significance threshold.
The relaxed assay-local tier additionally abstains when both active direct arms
carry structured provenance showing that they were produced by an observed
over-complete same-center-star reselection.  Baseline, legacy rescue, and the
formal-state tier are unchanged by that composition guard.
At most one pair is emitted per assay; ambiguity or an incomplete collider
motif causes abstention.

The output is strictly ``within_assay_shared_state_support``.  It never adds a
feature to a cross-assay state hyperedge and never treats the loading relation
as direct causation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructureAnchoredGroupConfig:
    minimum_lower_absolute_loading_cosine: float = 0.85
    minimum_relative_parent_margin: float = 0.10
    minimum_sibling_state_view_support: float = 2.0 / 3.0
    enable_abstention_only_empirical_null_rescue: bool = True
    rescue_empirical_null_p_threshold: float = 0.125
    rescue_minimum_empirical_null_view_support: float = 1.0
    rescue_minimum_parent_win_view_support: float = 1.0
    rescue_required_view_count: int = 6
    rescue_requires_unique_candidate: bool = True
    enable_layer_separated_tiered_rescue: bool = True
    tiered_rescue_empirical_null_p_threshold: float = 0.50
    tiered_rescue_minimum_empirical_null_view_support: float = 1.0
    tiered_rescue_minimum_parent_win_view_support: float = 1.0
    tiered_rescue_requires_sibling_assay_global_maxT_admission: bool = True
    tiered_rescue_minimum_sibling_conditional_wald_view_support: float = 1.0
    tiered_rescue_required_view_count: int = 6
    tiered_rescue_requires_unique_candidate: bool = True
    tiered_rescue_assay_local_blocked_observed_refinement_actions: tuple[
        str, ...
    ] = ("reselect_overcomplete_same_center_star",)

    def __post_init__(self) -> None:
        bounded = (
            self.minimum_lower_absolute_loading_cosine,
            self.minimum_relative_parent_margin,
            self.minimum_sibling_state_view_support,
            self.rescue_empirical_null_p_threshold,
            self.rescue_minimum_empirical_null_view_support,
            self.rescue_minimum_parent_win_view_support,
            self.tiered_rescue_empirical_null_p_threshold,
            self.tiered_rescue_minimum_empirical_null_view_support,
            self.tiered_rescue_minimum_parent_win_view_support,
            self.tiered_rescue_minimum_sibling_conditional_wald_view_support,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("Cosine, probability, support, and margin must lie in [0, 1]")
        if not isinstance(
            self.enable_abstention_only_empirical_null_rescue, bool
        ):
            raise ValueError(
                "enable_abstention_only_empirical_null_rescue must be Boolean"
            )
        if not isinstance(self.rescue_requires_unique_candidate, bool):
            raise ValueError("rescue_requires_unique_candidate must be Boolean")
        if not isinstance(
            self.tiered_rescue_requires_sibling_assay_global_maxT_admission, bool
        ):
            raise ValueError(
                "tiered_rescue_requires_sibling_assay_global_maxT_admission must be Boolean"
            )
        if not isinstance(self.enable_layer_separated_tiered_rescue, bool):
            raise ValueError("enable_layer_separated_tiered_rescue must be Boolean")
        if not isinstance(self.tiered_rescue_requires_unique_candidate, bool):
            raise ValueError("tiered_rescue_requires_unique_candidate must be Boolean")
        blocked_actions = (
            self.tiered_rescue_assay_local_blocked_observed_refinement_actions
        )
        if (
            not isinstance(blocked_actions, tuple)
            or not blocked_actions
            or any(
                not isinstance(action, str) or not action.strip()
                for action in blocked_actions
            )
            or len(set(blocked_actions)) != len(blocked_actions)
        ):
            raise ValueError(
                "tiered_rescue_assay_local_blocked_observed_refinement_actions "
                "must be a non-empty tuple of unique non-empty strings"
            )
        if (
            isinstance(self.rescue_required_view_count, bool)
            or not isinstance(self.rescue_required_view_count, int)
            or self.rescue_required_view_count < 1
        ):
            raise ValueError("rescue_required_view_count must be a positive integer")
        if (
            isinstance(self.tiered_rescue_required_view_count, bool)
            or not isinstance(self.tiered_rescue_required_view_count, int)
            or self.tiered_rescue_required_view_count < 1
        ):
            raise ValueError(
                "tiered_rescue_required_view_count must be a positive integer"
            )


@dataclass
class StructureAnchoredGroupResult:
    pair_evidence: pd.DataFrame
    typed_groups: pd.DataFrame
    assay_decisions: pd.DataFrame
    provenance: dict[str, Any]


_CANDIDATE_COLUMNS = {
    "pair_key",
    "node1",
    "node2",
    "modality",
    "n_views",
    "median_absolute_loading_cosine",
    "lower_absolute_loading_cosine",
    "median_minimum_partial_state_fit",
    "median_minimum_loading_stability",
    "eligible_view_support",
}
_DIRECT_COLUMNS = {
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "edge_type",
    "observed_refinement_action",
}
_NULL_COLUMNS = {
    "pair_key",
    "absolute_loading_cosine",
    "loading_empirical_null_p",
    "view_id",
}
_STATE_COLUMNS = {
    "feature",
    "modality",
    "selection_status",
    "feature_global_maxT_confirmation_gate",
    "assay_global_maxT_admission_gate",
    "conditional_wald_view_support",
}


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _coerce_boolean(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    values: list[bool] = []
    for value in series:
        key: Any = value
        if isinstance(value, str):
            key = value.strip().casefold()
        if key not in mapping:
            raise ValueError(f"{label} contains a non-Boolean value: {value!r}")
        values.append(mapping[key])
    return pd.Series(values, index=series.index, dtype=bool)


def _active_observed_refinement_action(assay_direct: pd.DataFrame) -> str:
    """Return one action only when it provenance-binds both active motif arms."""

    actions = []
    for value in assay_direct["observed_refinement_action"]:
        action = "" if pd.isna(value) else str(value).strip()
        actions.append(action)
    nonempty = {action for action in actions if action}
    if not nonempty:
        return ""
    if len(nonempty) != 1 or any(not action for action in actions):
        raise ValueError(
            "Active direct motif has inconsistent observed refinement action provenance"
        )
    if "component_id" not in assay_direct.columns:
        raise ValueError(
            "Action-bound active direct motif is missing component_id provenance"
        )
    component_ids = {
        str(value).strip()
        for value in assay_direct["component_id"]
        if not pd.isna(value) and str(value).strip()
    }
    if len(component_ids) != 1:
        raise ValueError(
            "Action-bound active direct motif must have exactly one component_id"
        )
    return next(iter(nonempty))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _center_from_edge(row: Any) -> str:
    explicit = str(getattr(row, "collider_motif_center", "")).strip()
    if explicit and explicit.lower() != "nan":
        return explicit
    arrow_nodes: list[str] = []
    if str(row.endpoint1).upper() == "ARROW":
        arrow_nodes.append(str(row.node1))
    if str(row.endpoint2).upper() == "ARROW":
        arrow_nodes.append(str(row.node2))
    return arrow_nodes[0] if len(arrow_nodes) == 1 else ""


def estimate_structure_anchored_groups(
    consensus_edges: pd.DataFrame,
    within_group_candidates: pd.DataFrame,
    empirical_null_views: pd.DataFrame,
    state_membership: pd.DataFrame,
    config: StructureAnchoredGroupConfig | None = None,
) -> StructureAnchoredGroupResult:
    """Estimate at most one structure-anchored loading group per assay."""

    config = config or StructureAnchoredGroupConfig()
    _require(consensus_edges, _DIRECT_COLUMNS, "consensus_edges")
    _require(within_group_candidates, _CANDIDATE_COLUMNS, "within_group_candidates")
    _require(empirical_null_views, _NULL_COLUMNS, "empirical_null_views")
    _require(state_membership, _STATE_COLUMNS, "state_membership")

    membership = state_membership.copy()
    membership["feature"] = membership["feature"].astype(str)
    membership["modality"] = membership["modality"].astype(str)
    if membership["feature"].duplicated().any():
        raise ValueError("state_membership contains duplicate features")
    membership["conditional_wald_view_support"] = pd.to_numeric(
        membership["conditional_wald_view_support"], errors="raise"
    )
    for column in (
        "feature_global_maxT_confirmation_gate",
        "assay_global_maxT_admission_gate",
    ):
        membership[column] = _coerce_boolean(
            membership[column], f"state_membership.{column}"
        )
    if not membership["conditional_wald_view_support"].between(0.0, 1.0).all():
        raise ValueError("conditional_wald_view_support must lie in [0, 1]")
    membership["formal_sibling_eligible"] = (
        membership["selection_status"].astype(str).eq("selected")
        & membership["feature_global_maxT_confirmation_gate"]
        & membership["assay_global_maxT_admission_gate"]
        & (
            membership["conditional_wald_view_support"]
            >= config.minimum_sibling_state_view_support
        )
    )
    membership_lookup = membership.set_index("feature", drop=False)

    candidates = within_group_candidates.copy()
    candidates["pair_key"] = candidates["pair_key"].astype(str)
    candidates["node1"] = candidates["node1"].astype(str)
    candidates["node2"] = candidates["node2"].astype(str)
    candidates["modality"] = candidates["modality"].astype(str)
    if candidates["pair_key"].duplicated().any():
        raise ValueError("within_group_candidates contains duplicate pair keys")
    feature_modality: dict[str, str] = {}
    for row in candidates.itertuples(index=False):
        for feature in (str(row.node1), str(row.node2)):
            previous = feature_modality.setdefault(feature, str(row.modality))
            if previous != str(row.modality):
                raise ValueError(f"Feature occurs in multiple modalities: {feature}")

    null_views = empirical_null_views.copy()
    null_views["pair_key"] = null_views["pair_key"].astype(str)
    null_views["view_id"] = null_views["view_id"].astype(str)
    null_views["absolute_loading_cosine"] = pd.to_numeric(
        null_views["absolute_loading_cosine"], errors="raise"
    )
    null_views["loading_empirical_null_p"] = pd.to_numeric(
        null_views["loading_empirical_null_p"], errors="raise"
    )
    if null_views.duplicated(["view_id", "pair_key"]).any():
        raise ValueError("Empirical-null evidence duplicates a pair within a view")
    if not null_views["loading_empirical_null_p"].between(0.0, 1.0).all():
        raise ValueError("loading_empirical_null_p must lie in [0, 1]")
    if not null_views["absolute_loading_cosine"].between(0.0, 1.0).all():
        raise ValueError("absolute_loading_cosine must lie in [0, 1]")
    pair_view_lookup = {
        str(pair_key): frame.sort_values("view_id").reset_index(drop=True)
        for pair_key, frame in null_views.groupby("pair_key", sort=True)
    }
    null_summary = (
        null_views.groupby("pair_key", sort=True)["loading_empirical_null_p"]
        .agg(
            empirical_null_median_p="median",
            empirical_null_n_views="size",
            empirical_null_support_p_le_0_125=lambda values: float(
                np.mean(np.asarray(values, dtype=np.float64) <= 0.125)
            ),
            empirical_null_support_p_le_0_25=lambda values: float(
                np.mean(np.asarray(values, dtype=np.float64) <= 0.25)
            ),
            empirical_null_support_p_le_0_50=lambda values: float(
                np.mean(np.asarray(values, dtype=np.float64) <= 0.50)
            ),
        )
        .reset_index()
    )
    candidates = candidates.merge(
        null_summary, on="pair_key", how="left", validate="one_to_one"
    )
    if candidates[
        ["empirical_null_median_p", "empirical_null_n_views"]
    ].isna().any().any():
        raise ValueError("Every candidate pair must have empirical-null view evidence")
    if not (
        candidates["empirical_null_n_views"].astype(int)
        == candidates["n_views"].astype(int)
    ).all():
        raise ValueError("Candidate and empirical-null view counts disagree")

    direct = consensus_edges.loc[
        consensus_edges["edge_type"].astype(str).eq(
            "within_assay_direct_candidate"
        )
    ].copy()
    direct["node1"] = direct["node1"].astype(str)
    direct["node2"] = direct["node2"].astype(str)
    unknown_direct = sorted(
        (set(direct["node1"]) | set(direct["node2"])).difference(feature_modality)
    )
    if unknown_direct:
        raise ValueError(f"Direct graph contains unknown features: {unknown_direct}")
    direct["modality"] = direct["node1"].map(feature_modality)
    if not direct["node2"].map(feature_modality).eq(direct["modality"]).all():
        raise ValueError("A within-assay direct edge joins different modalities")

    evidence_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    typed_rows: list[dict[str, Any]] = []
    for modality in sorted(candidates["modality"].unique()):
        assay_direct = direct.loc[direct["modality"].eq(modality)]
        if len(assay_direct) != 2:
            decision_rows.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selection_stage": "none",
                    "abstention_reason": "direct_motif_not_exactly_two_edges",
                    "n_direct_edges": len(assay_direct),
                    "n_structural_candidates": 0,
                    "n_gate_eligible_candidates": 0,
                    "n_baseline_eligible_candidates": 0,
                    "n_rescue_eligible_candidates": 0,
                    "n_tiered_rescue_eligible_candidates": 0,
                    "active_observed_refinement_action": "",
                    "relative_best_cosine_margin": 0.0,
                    "selected_pair_key": "",
                }
            )
            continue
        centers = [_center_from_edge(row) for row in assay_direct.itertuples(index=False)]
        if not centers[0] or len(set(centers)) != 1:
            decision_rows.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selection_stage": "none",
                    "abstention_reason": "direct_motif_center_ambiguous",
                    "n_direct_edges": len(assay_direct),
                    "n_structural_candidates": 0,
                    "n_gate_eligible_candidates": 0,
                    "n_baseline_eligible_candidates": 0,
                    "n_rescue_eligible_candidates": 0,
                    "n_tiered_rescue_eligible_candidates": 0,
                    "active_observed_refinement_action": "",
                    "relative_best_cosine_margin": 0.0,
                    "selected_pair_key": "",
                }
            )
            continue
        center = centers[0]
        motif_nodes = set(assay_direct["node1"]) | set(assay_direct["node2"])
        parents = motif_nodes.difference({center})
        if len(parents) != 2:
            decision_rows.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selection_stage": "none",
                    "abstention_reason": "direct_motif_not_two_distinct_parents",
                    "n_direct_edges": len(assay_direct),
                    "n_structural_candidates": 0,
                    "n_gate_eligible_candidates": 0,
                    "n_baseline_eligible_candidates": 0,
                    "n_rescue_eligible_candidates": 0,
                    "n_tiered_rescue_eligible_candidates": 0,
                    "active_observed_refinement_action": "",
                    "relative_best_cosine_margin": 0.0,
                    "selected_pair_key": "",
                }
            )
            continue
        active_observed_refinement_action = _active_observed_refinement_action(
            assay_direct
        )
        assay_local_action_gate = bool(
            active_observed_refinement_action
            not in config.tiered_rescue_assay_local_blocked_observed_refinement_actions
        )

        structural: list[tuple[int, str, str]] = []
        assay_candidates = candidates.loc[candidates["modality"].eq(modality)]
        for index, row in assay_candidates.iterrows():
            nodes = {str(row["node1"]), str(row["node2"])}
            parent_members = nodes & parents
            outside_members = nodes.difference(motif_nodes)
            if len(parent_members) == 1 and len(outside_members) == 1:
                structural.append(
                    (index, next(iter(parent_members)), next(iter(outside_members)))
                )
        if not structural:
            decision_rows.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selection_stage": "none",
                    "abstention_reason": "no_parent_to_noncollider_candidate",
                    "n_direct_edges": len(assay_direct),
                    "n_structural_candidates": 0,
                    "n_gate_eligible_candidates": 0,
                    "n_baseline_eligible_candidates": 0,
                    "n_rescue_eligible_candidates": 0,
                    "n_tiered_rescue_eligible_candidates": 0,
                    "active_observed_refinement_action": (
                        active_observed_refinement_action
                    ),
                    "relative_best_cosine_margin": 0.0,
                    "selected_pair_key": "",
                }
            )
            continue

        assay_rows: list[dict[str, Any]] = []
        for index, parent, sibling in structural:
            row = candidates.loc[index]
            if sibling not in membership_lookup.index:
                raise ValueError(
                    f"Structural sibling is absent from state membership: {sibling}"
                )
            sibling_membership = membership_lookup.loc[sibling]
            if str(sibling_membership["modality"]) != modality:
                raise ValueError(
                    f"State membership modality disagrees for sibling {sibling}"
                )
            other_parent = next(iter(parents.difference({parent})))
            opposite_pair_key = "|||".join(sorted((other_parent, sibling)))
            pair_key = str(row["pair_key"])
            if pair_key not in pair_view_lookup or opposite_pair_key not in pair_view_lookup:
                raise ValueError(
                    "Both parent-to-sibling pairs require the same per-view evidence"
                )
            pair_views = pair_view_lookup[pair_key]
            opposite_views = pair_view_lookup[opposite_pair_key]
            if list(pair_views["view_id"]) != list(opposite_views["view_id"]):
                raise ValueError("Parent-comparison loading views are misaligned")
            if len(pair_views) != int(row["n_views"]):
                raise ValueError("Candidate and parent-comparison view counts disagree")
            similarity = pair_views["absolute_loading_cosine"].to_numpy(
                dtype=np.float64
            )
            opposite_similarity = opposite_views[
                "absolute_loading_cosine"
            ].to_numpy(dtype=np.float64)
            advantage = similarity - opposite_similarity
            median_similarity = float(np.median(similarity))
            relative_parent_margin = float(
                np.median(advantage) / max(median_similarity, 1e-12)
            )
            parent_win_view_support = float(np.mean(advantage > 0.0))
            rescue_empirical_null_view_support = float(
                np.mean(
                    pair_views["loading_empirical_null_p"].to_numpy(
                        dtype=np.float64
                    )
                    <= config.rescue_empirical_null_p_threshold
                )
            )
            tiered_rescue_empirical_null_view_support = float(
                np.mean(
                    pair_views["loading_empirical_null_p"].to_numpy(
                        dtype=np.float64
                    )
                    <= config.tiered_rescue_empirical_null_p_threshold
                )
            )
            state_gate = bool(sibling_membership["formal_sibling_eligible"])
            tiered_rescue_assay_global_state_gate = bool(
                not config.tiered_rescue_requires_sibling_assay_global_maxT_admission
                or sibling_membership["assay_global_maxT_admission_gate"]
            )
            tiered_rescue_conditional_state_gate = bool(
                float(sibling_membership["conditional_wald_view_support"])
                >= config.tiered_rescue_minimum_sibling_conditional_wald_view_support
            )
            tiered_rescue_assay_local_state_gate = bool(
                tiered_rescue_assay_global_state_gate
                and tiered_rescue_conditional_state_gate
            )
            cosine_gate = bool(
                float(row["lower_absolute_loading_cosine"])
                >= config.minimum_lower_absolute_loading_cosine
            )
            parent_margin_gate = bool(
                relative_parent_margin >= config.minimum_relative_parent_margin
            )
            rescue_view_count_gate = bool(
                len(pair_views) == config.rescue_required_view_count
            )
            rescue_parent_win_gate = bool(
                parent_win_view_support
                >= config.rescue_minimum_parent_win_view_support
            )
            rescue_empirical_null_gate = bool(
                rescue_empirical_null_view_support
                >= config.rescue_minimum_empirical_null_view_support
            )
            rescue_all_gates = bool(
                config.enable_abstention_only_empirical_null_rescue
                and state_gate
                and parent_margin_gate
                and rescue_view_count_gate
                and rescue_parent_win_gate
                and rescue_empirical_null_gate
            )
            tiered_rescue_view_count_gate = bool(
                len(pair_views) == config.tiered_rescue_required_view_count
            )
            tiered_rescue_parent_win_gate = bool(
                parent_win_view_support
                >= config.tiered_rescue_minimum_parent_win_view_support
            )
            tiered_rescue_empirical_null_gate = bool(
                tiered_rescue_empirical_null_view_support
                >= config.tiered_rescue_minimum_empirical_null_view_support
            )
            tiered_rescue_common_gates = bool(
                config.enable_layer_separated_tiered_rescue
                and parent_margin_gate
                and tiered_rescue_view_count_gate
                and tiered_rescue_parent_win_gate
                and tiered_rescue_empirical_null_gate
            )
            tiered_rescue_strict_state_all_gates = bool(
                tiered_rescue_common_gates and state_gate
            )
            tiered_rescue_assay_local_state_pre_action_gates = bool(
                tiered_rescue_common_gates
                and tiered_rescue_assay_local_state_gate
            )
            tiered_rescue_assay_local_state_all_gates = bool(
                tiered_rescue_assay_local_state_pre_action_gates
                and assay_local_action_gate
            )
            blocked_relaxed_tier_by_observed_refinement_action = bool(
                tiered_rescue_assay_local_state_pre_action_gates
                and not tiered_rescue_strict_state_all_gates
                and not assay_local_action_gate
            )
            record = {
                "pair_key": pair_key,
                "opposite_parent_pair_key": opposite_pair_key,
                "modality": modality,
                "parent_candidate": parent,
                "noncollider_sibling_candidate": sibling,
                "collider_center": center,
                "other_parent_candidate": other_parent,
                "active_observed_refinement_action": (
                    active_observed_refinement_action
                ),
                "median_absolute_loading_cosine": float(
                    row["median_absolute_loading_cosine"]
                ),
                "lower_absolute_loading_cosine": float(
                    row["lower_absolute_loading_cosine"]
                ),
                "median_coherence_group_score": float(
                    row.get("median_coherence_group_score", median_similarity)
                ),
                "median_minimum_partial_state_fit": float(
                    row["median_minimum_partial_state_fit"]
                ),
                "median_minimum_loading_stability": float(
                    row["median_minimum_loading_stability"]
                ),
                "eligible_view_support": float(row["eligible_view_support"]),
                "parent_win_view_support": parent_win_view_support,
                "parent_nonloss_view_support": float(np.mean(advantage >= 0.0)),
                "median_parent_similarity_advantage": float(
                    np.median(advantage)
                ),
                "lower_parent_similarity_advantage": float(
                    np.quantile(advantage, 0.10)
                ),
                "relative_median_parent_margin": relative_parent_margin,
                "empirical_null_median_p": float(row["empirical_null_median_p"]),
                "empirical_null_support_p_le_0_125": float(
                    row["empirical_null_support_p_le_0_125"]
                ),
                "empirical_null_support_p_le_0_25": float(
                    row["empirical_null_support_p_le_0_25"]
                ),
                "empirical_null_support_p_le_0_50": float(
                    row["empirical_null_support_p_le_0_50"]
                ),
                "rescue_empirical_null_view_support": (
                    rescue_empirical_null_view_support
                ),
                "tiered_rescue_empirical_null_view_support": (
                    tiered_rescue_empirical_null_view_support
                ),
                "sibling_formal_state_selected": bool(
                    str(sibling_membership["selection_status"]) == "selected"
                ),
                "sibling_feature_global_maxT_confirmation_gate": bool(
                    sibling_membership["feature_global_maxT_confirmation_gate"]
                ),
                "sibling_assay_global_maxT_admission_gate": bool(
                    sibling_membership["assay_global_maxT_admission_gate"]
                ),
                "sibling_conditional_wald_view_support": float(
                    sibling_membership["conditional_wald_view_support"]
                ),
                "parent_membership_used_as_gate": False,
                "passes_state_membership_gate": state_gate,
                "passes_cosine_gate": cosine_gate,
                "passes_relative_parent_margin_gate": parent_margin_gate,
                "passes_all_selection_gates": bool(
                    state_gate and cosine_gate and parent_margin_gate
                ),
                "passes_tiered_rescue_assay_global_state_gate": (
                    tiered_rescue_assay_global_state_gate
                ),
                "passes_tiered_rescue_conditional_state_gate": (
                    tiered_rescue_conditional_state_gate
                ),
                "passes_tiered_rescue_assay_local_state_gate": (
                    tiered_rescue_assay_local_state_gate
                ),
                "passes_tiered_rescue_assay_local_action_gate": (
                    assay_local_action_gate
                ),
                "passes_tiered_rescue_strict_state_all_gates": (
                    tiered_rescue_strict_state_all_gates
                ),
                "passes_tiered_rescue_assay_local_state_all_gates": (
                    tiered_rescue_assay_local_state_all_gates
                ),
                "blocked_relaxed_tier_by_observed_refinement_action": (
                    blocked_relaxed_tier_by_observed_refinement_action
                ),
                "passes_rescue_relative_parent_margin_gate": parent_margin_gate,
                "passes_rescue_required_view_count_gate": rescue_view_count_gate,
                "passes_rescue_parent_win_gate": rescue_parent_win_gate,
                "passes_rescue_empirical_null_gate": rescue_empirical_null_gate,
                "passes_all_rescue_gates": rescue_all_gates,
                "passes_tiered_rescue_required_view_count_gate": (
                    tiered_rescue_view_count_gate
                ),
                "passes_tiered_rescue_parent_win_gate": (
                    tiered_rescue_parent_win_gate
                ),
                "passes_tiered_rescue_empirical_null_gate": (
                    tiered_rescue_empirical_null_gate
                ),
                "selected": False,
                "selection_stage": "none",
            }
            assay_rows.append(record)

        baseline_eligible = sorted(
            (row for row in assay_rows if row["passes_all_selection_gates"]),
            key=lambda row: (
                -row["median_coherence_group_score"],
                -row["median_absolute_loading_cosine"],
                row["pair_key"],
            ),
        )
        legacy_rescue_gate_eligible = sorted(
            (row for row in assay_rows if row["passes_all_rescue_gates"]),
            key=lambda row: (
                -row["median_coherence_group_score"],
                -row["median_absolute_loading_cosine"],
                row["pair_key"],
            ),
        )
        strict_tiered_rescue_gate_eligible = sorted(
            (
                row
                for row in assay_rows
                if row["passes_tiered_rescue_strict_state_all_gates"]
            ),
            key=lambda row: (
                -row["median_coherence_group_score"],
                -row["median_absolute_loading_cosine"],
                row["pair_key"],
            ),
        )
        relaxed_tiered_rescue_gate_eligible = sorted(
            (
                row
                for row in assay_rows
                if row["passes_tiered_rescue_assay_local_state_all_gates"]
                and not row["passes_tiered_rescue_strict_state_all_gates"]
            ),
            key=lambda row: (
                -row["median_coherence_group_score"],
                -row["median_absolute_loading_cosine"],
                row["pair_key"],
            ),
        )
        # Prefer the stronger cross-assay/formal state tier whenever it has a
        # candidate.  The assay-local tier is considered only after the
        # stronger tier abstains entirely; ambiguity within the active tier is
        # never resolved by score ranking.
        tiered_rescue_gate_eligible = (
            strict_tiered_rescue_gate_eligible
            if strict_tiered_rescue_gate_eligible
            else relaxed_tiered_rescue_gate_eligible
        )
        # Both rescue stages are deliberately invisible to assays selected by
        # the formal baseline.  The new p50 tier is additionally invisible to
        # the legacy p125 rescue, preserving every current production choice.
        legacy_rescue_eligible = (
            legacy_rescue_gate_eligible if not baseline_eligible else []
        )
        legacy_rescue_selectable = bool(
            legacy_rescue_eligible
            and (
                not config.rescue_requires_unique_candidate
                or len(legacy_rescue_eligible) == 1
            )
        )
        tiered_rescue_eligible = (
            tiered_rescue_gate_eligible
            if not baseline_eligible and not legacy_rescue_selectable
            else []
        )
        selected_record: dict[str, Any] | None = None
        selection_stage = "none"
        selected_rescue_state_tier = "not_applicable"
        selected_pool: list[dict[str, Any]] = []
        if baseline_eligible:
            selected_record = baseline_eligible[0]
            selected_pool = baseline_eligible
            selection_stage = "formal_baseline"
        elif legacy_rescue_selectable:
            selected_record = legacy_rescue_eligible[0]
            selected_pool = legacy_rescue_eligible
            selection_stage = "empirical_null_unanimity_rescue"
            selected_rescue_state_tier = "legacy_formal_state"
        elif tiered_rescue_eligible and (
            not config.tiered_rescue_requires_unique_candidate
            or len(tiered_rescue_eligible) == 1
        ):
            selected_record = tiered_rescue_eligible[0]
            selected_pool = tiered_rescue_eligible
            selection_stage = "tiered_empirical_null_non_domination_rescue"
            selected_rescue_state_tier = (
                "formal_cross_assay_state"
                if bool(
                    selected_record[
                        "passes_tiered_rescue_strict_state_all_gates"
                    ]
                )
                else "assay_local_state"
            )
        if selected_record is not None:
            selected_record["selected"] = True
            selected_record["selection_stage"] = selection_stage
        evidence_rows.extend(assay_rows)

        if selected_record is None:
            state_nominated = sum(
                bool(row["passes_state_membership_gate"]) for row in assay_rows
            )
            if (
                config.rescue_requires_unique_candidate
                and len(legacy_rescue_eligible) > 1
            ):
                reason = "empirical_null_rescue_ambiguous"
            elif (
                config.tiered_rescue_requires_unique_candidate
                and len(tiered_rescue_eligible) > 1
            ):
                reason = "tiered_empirical_null_rescue_ambiguous"
            elif any(
                bool(
                    row[
                        "blocked_relaxed_tier_by_observed_refinement_action"
                    ]
                )
                for row in assay_rows
            ):
                reason = (
                    "tiered_assay_local_rescue_blocked_by_observed_"
                    "refinement_action"
                )
            else:
                reason = (
                    "no_formal_state_selected_noncollider_sibling"
                    if state_nominated == 0
                    else "state_sibling_failed_loading_geometry"
                )
            decision_rows.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "selection_stage": "none",
                    "abstention_reason": reason,
                    "n_direct_edges": len(assay_direct),
                    "n_structural_candidates": len(assay_rows),
                    "n_gate_eligible_candidates": len(baseline_eligible),
                    "n_baseline_eligible_candidates": len(baseline_eligible),
                    "n_rescue_eligible_candidates": len(legacy_rescue_eligible),
                    "n_tiered_rescue_eligible_candidates": len(
                        tiered_rescue_eligible
                    ),
                    "active_observed_refinement_action": (
                        active_observed_refinement_action
                    ),
                    "relative_best_cosine_margin": 0.0,
                    "selected_pair_key": "",
                }
            )
        else:
            pair_key = selected_record["pair_key"]
            runner_up = (
                float(selected_pool[1]["median_coherence_group_score"])
                if len(selected_pool) > 1
                else 0.0
            )
            best_margin = float(
                (
                    selected_record["median_coherence_group_score"]
                    - runner_up
                )
                / max(selected_record["median_coherence_group_score"], 1e-12)
            )
            decision_rows.append(
                {
                    "modality": modality,
                    "selection_status": "selected",
                    "selection_stage": selection_stage,
                    "abstention_reason": "",
                    "n_direct_edges": len(assay_direct),
                    "n_structural_candidates": len(assay_rows),
                    "n_gate_eligible_candidates": len(baseline_eligible),
                    "n_baseline_eligible_candidates": len(baseline_eligible),
                    "n_rescue_eligible_candidates": len(legacy_rescue_eligible),
                    "n_tiered_rescue_eligible_candidates": len(
                        tiered_rescue_eligible
                    ),
                    "active_observed_refinement_action": (
                        active_observed_refinement_action
                    ),
                    "relative_best_cosine_margin": best_margin,
                    "selected_pair_key": pair_key,
                }
            )
            members = sorted(
                [
                    selected_record["parent_candidate"],
                    selected_record["noncollider_sibling_candidate"],
                ]
            )
            typed_rows.append(
                {
                    "relation_id": f"structure_anchored__{modality}",
                    "edge_type": "within_assay_shared_state_support",
                    "selection_status": "selected",
                    "selection_stage": selection_stage,
                    "empirical_null_rescue_used": bool(
                        selection_stage
                        in {
                            "empirical_null_unanimity_rescue",
                            "tiered_empirical_null_non_domination_rescue",
                        }
                    ),
                    "legacy_empirical_null_rescue_used": bool(
                        selection_stage == "empirical_null_unanimity_rescue"
                    ),
                    "tiered_empirical_null_rescue_used": bool(
                        selection_stage
                        == "tiered_empirical_null_non_domination_rescue"
                    ),
                    "rescue_state_tier": selected_rescue_state_tier,
                    "modality": modality,
                    "members": ";".join(members),
                    "n_members": 2,
                    "collider_parent_anchor": selected_record["parent_candidate"],
                    "noncollider_sibling": selected_record[
                        "noncollider_sibling_candidate"
                    ],
                    "collider_center_excluded": center,
                    "active_observed_refinement_action": (
                        active_observed_refinement_action
                    ),
                    "relative_best_cosine_margin": best_margin,
                    "relative_parent_margin": selected_record[
                        "relative_median_parent_margin"
                    ],
                    "parent_win_view_support": selected_record[
                        "parent_win_view_support"
                    ],
                    "rescue_empirical_null_view_support": selected_record[
                        "rescue_empirical_null_view_support"
                    ],
                    "tiered_rescue_empirical_null_view_support": selected_record[
                        "tiered_rescue_empirical_null_view_support"
                    ],
                    "evidence_basis": (
                        "formal_state_sibling_plus_collider_parent_loading_choice"
                        if selection_stage == "formal_baseline"
                        else (
                            "formal_state_sibling_plus_parent_win_and_legacy_"
                            "empirical_null_unanimity_rescue"
                            if selection_stage
                            == "empirical_null_unanimity_rescue"
                            else (
                                "formal_state_sibling_plus_parent_win_and_"
                                "empirical_null_non_domination_rescue"
                                if selected_rescue_state_tier
                                == "formal_cross_assay_state"
                                else (
                                    "assay_local_state_unanimity_plus_parent_win_"
                                    "and_empirical_null_non_domination_rescue"
                                )
                            )
                        )
                    ),
                    "interpretation": (
                        "assay-local shared-state support only; not direct causation "
                        "or cross-assay membership"
                    ),
                    "direct_causal_claimed": False,
                    "cross_assay_membership_claimed": False,
                    "exact_axis_pairing_claimed": False,
                }
            )

    pair_columns = [
        "pair_key",
        "opposite_parent_pair_key",
        "modality",
        "parent_candidate",
        "noncollider_sibling_candidate",
        "collider_center",
        "other_parent_candidate",
        "active_observed_refinement_action",
        "median_absolute_loading_cosine",
        "lower_absolute_loading_cosine",
        "median_coherence_group_score",
        "median_minimum_partial_state_fit",
        "median_minimum_loading_stability",
        "eligible_view_support",
        "parent_win_view_support",
        "parent_nonloss_view_support",
        "median_parent_similarity_advantage",
        "lower_parent_similarity_advantage",
        "relative_median_parent_margin",
        "empirical_null_median_p",
        "empirical_null_support_p_le_0_125",
        "empirical_null_support_p_le_0_25",
        "empirical_null_support_p_le_0_50",
        "rescue_empirical_null_view_support",
        "tiered_rescue_empirical_null_view_support",
        "sibling_formal_state_selected",
        "sibling_feature_global_maxT_confirmation_gate",
        "sibling_assay_global_maxT_admission_gate",
        "sibling_conditional_wald_view_support",
        "parent_membership_used_as_gate",
        "passes_state_membership_gate",
        "passes_cosine_gate",
        "passes_relative_parent_margin_gate",
        "passes_all_selection_gates",
        "passes_tiered_rescue_assay_global_state_gate",
        "passes_tiered_rescue_conditional_state_gate",
        "passes_tiered_rescue_assay_local_state_gate",
        "passes_tiered_rescue_assay_local_action_gate",
        "passes_tiered_rescue_strict_state_all_gates",
        "passes_tiered_rescue_assay_local_state_all_gates",
        "blocked_relaxed_tier_by_observed_refinement_action",
        "passes_rescue_relative_parent_margin_gate",
        "passes_rescue_required_view_count_gate",
        "passes_rescue_parent_win_gate",
        "passes_rescue_empirical_null_gate",
        "passes_all_rescue_gates",
        "passes_tiered_rescue_required_view_count_gate",
        "passes_tiered_rescue_parent_win_gate",
        "passes_tiered_rescue_empirical_null_gate",
        "selected",
        "selection_stage",
    ]
    typed_columns = [
        "relation_id",
        "edge_type",
        "selection_status",
        "selection_stage",
        "empirical_null_rescue_used",
        "legacy_empirical_null_rescue_used",
        "tiered_empirical_null_rescue_used",
        "rescue_state_tier",
        "modality",
        "members",
        "n_members",
        "collider_parent_anchor",
        "noncollider_sibling",
        "collider_center_excluded",
        "active_observed_refinement_action",
        "relative_best_cosine_margin",
        "relative_parent_margin",
        "parent_win_view_support",
        "rescue_empirical_null_view_support",
        "tiered_rescue_empirical_null_view_support",
        "evidence_basis",
        "interpretation",
        "direct_causal_claimed",
        "cross_assay_membership_claimed",
        "exact_axis_pairing_claimed",
    ]
    decisions = pd.DataFrame(decision_rows).sort_values("modality").reset_index(drop=True)
    pairs = pd.DataFrame(evidence_rows, columns=pair_columns)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["modality", "selected", "median_absolute_loading_cosine", "pair_key"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
    typed = pd.DataFrame(typed_rows, columns=typed_columns)
    if not typed.empty:
        typed = typed.sort_values("modality").reset_index(drop=True)
    provenance = {
        "schema_version": "structure_anchored_loading_groups_v4.v5",
        "config": asdict(config),
        "method": (
            "layer_separated_state_sibling_gated_collider_parent_choice_with_"
            "production_preserving_hierarchical_abstention_rescue"
        ),
        "selection_rule": {
            "baseline_noncollider_sibling_must_be_formal_state_selected": True,
            "direct_parent_state_membership_is_diagnostic_only": True,
            "baseline_minimum_lower_absolute_loading_cosine": (
                config.minimum_lower_absolute_loading_cosine
            ),
            "baseline_minimum_relative_parent_margin": (
                config.minimum_relative_parent_margin
            ),
            "baseline_selections_preserved_before_rescue": True,
            "rescue_only_after_baseline_abstention": True,
            "legacy_rescue_enabled": (
                config.enable_abstention_only_empirical_null_rescue
            ),
            "legacy_rescue_minimum_relative_parent_margin": (
                config.minimum_relative_parent_margin
            ),
            "legacy_rescue_required_view_count": config.rescue_required_view_count,
            "legacy_rescue_minimum_parent_win_view_support": (
                config.rescue_minimum_parent_win_view_support
            ),
            "legacy_rescue_requires_formal_state_sibling": True,
            "legacy_rescue_empirical_null_p_threshold": (
                config.rescue_empirical_null_p_threshold
            ),
            "legacy_rescue_minimum_empirical_null_view_support": (
                config.rescue_minimum_empirical_null_view_support
            ),
            "legacy_rescue_requires_unique_candidate": (
                config.rescue_requires_unique_candidate
            ),
            "current_production_selections_preserved_before_tiered_rescue": True,
            "tiered_rescue_only_after_current_production_abstention": True,
            "tiered_rescue_enabled": config.enable_layer_separated_tiered_rescue,
            "tiered_rescue_minimum_relative_parent_margin": (
                config.minimum_relative_parent_margin
            ),
            "tiered_rescue_required_view_count": (
                config.tiered_rescue_required_view_count
            ),
            "tiered_rescue_minimum_parent_win_view_support": (
                config.tiered_rescue_minimum_parent_win_view_support
            ),
            "tiered_rescue_empirical_null_p_threshold": (
                config.tiered_rescue_empirical_null_p_threshold
            ),
            "tiered_rescue_minimum_empirical_null_view_support": (
                config.tiered_rescue_minimum_empirical_null_view_support
            ),
            "tiered_rescue_requires_sibling_assay_global_maxT_admission": (
                config.tiered_rescue_requires_sibling_assay_global_maxT_admission
            ),
            "tiered_rescue_minimum_sibling_conditional_wald_view_support": (
                config.tiered_rescue_minimum_sibling_conditional_wald_view_support
            ),
            "tiered_rescue_requires_unique_candidate": (
                config.tiered_rescue_requires_unique_candidate
            ),
            "tiered_rescue_sibling_feature_global_maxT_used_in_strict_tier": True,
            "tiered_rescue_sibling_feature_global_maxT_used_in_relaxed_tier": False,
            "tiered_rescue_uses_hierarchical_state_tiers": True,
            "tiered_rescue_prefers_formal_cross_assay_state_tier": True,
            "tiered_rescue_uses_assay_local_state_only_when_formal_tier_empty": True,
            "tiered_rescue_relaxed_tier_excludes_formal_candidates": True,
            "tiered_rescue_assay_local_blocked_observed_refinement_actions": (
                list(
                    config.tiered_rescue_assay_local_blocked_observed_refinement_actions
                )
            ),
            "tiered_rescue_observed_refinement_action_guard_applies_only_to_"
            "relaxed_assay_local_tier": True,
            "tiered_rescue_ambiguity_within_active_tier_forces_abstention": True,
            "legacy_and_tiered_rescues_have_absolute_cosine_cutoff": False,
            "empirical_loading_null_used_only_for_abstention_rescues": True,
            "tiered_empirical_null_p_le_0_50_means_not_null_dominated": True,
            "tiered_empirical_null_p_le_0_50_is_not_per_view_significance": True,
            "six_views_are_correlated_stability_repeats": True,
            "six_views_are_not_claimed_as_independent_replicates": True,
            "at_most_one_group_per_assay": True,
        },
        "semantics": {
            "within_assay_shared_state_support_only": True,
            "direct_causal_claimed": False,
            "cross_assay_membership_claimed": False,
            "layer_union_is_state_membership": False,
        },
        "counts": {
            "n_assays": len(decisions),
            "n_selected_groups": len(typed),
            "n_baseline_selected_groups": int(
                typed["selection_stage"].eq("formal_baseline").sum()
            ),
            "n_empirical_null_rescued_groups": int(
                typed["selection_stage"]
                .eq("empirical_null_unanimity_rescue")
                .sum()
            ),
            "n_tiered_empirical_null_rescued_groups": int(
                typed["selection_stage"]
                .eq("tiered_empirical_null_non_domination_rescue")
                .sum()
            ),
            "n_formal_state_tier_rescued_groups": int(
                typed["rescue_state_tier"].eq("formal_cross_assay_state").sum()
            ),
            "n_assay_local_state_tier_rescued_groups": int(
                typed["rescue_state_tier"].eq("assay_local_state").sum()
            ),
            "n_relaxed_tier_candidates_blocked_by_observed_refinement_action": int(
                pairs[
                    "blocked_relaxed_tier_by_observed_refinement_action"
                ].astype(bool).sum()
                if not pairs.empty
                else 0
            ),
            "n_abstained_assays": int(decisions["selection_status"].ne("selected").sum()),
        },
    }
    return StructureAnchoredGroupResult(pairs, typed, decisions, provenance)


def run_structure_anchored_groups(
    consensus_edges_path: Path,
    within_group_candidates_path: Path,
    empirical_null_paths: Sequence[Path],
    state_membership_path: Path,
    out_dir: Path,
    config: StructureAnchoredGroupConfig | None = None,
) -> dict[str, Any]:
    config = config or StructureAnchoredGroupConfig()
    consensus_path = consensus_edges_path.resolve(strict=True)
    candidate_path = within_group_candidates_path.resolve(strict=True)
    membership_path = state_membership_path.resolve(strict=True)
    null_paths = [path.resolve(strict=True) for path in empirical_null_paths]
    if len(null_paths) < 2:
        raise ValueError("At least two empirical-null views are required")
    output = out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    null_frames: list[pd.DataFrame] = []
    for index, path in enumerate(null_paths):
        frame = pd.read_csv(path, sep="\t")
        frame["view_id"] = path.parent.name or f"view_{index}"
        null_frames.append(frame)
    result = estimate_structure_anchored_groups(
        pd.read_csv(consensus_path, sep="\t"),
        pd.read_csv(candidate_path, sep="\t"),
        pd.concat(null_frames, ignore_index=True),
        pd.read_csv(membership_path, sep="\t"),
        config,
    )
    paths = {
        "structure_anchor_pair_evidence.tsv": output / "structure_anchor_pair_evidence.tsv",
        "typed_structure_anchored_groups.tsv": output / "typed_structure_anchored_groups.tsv",
        "assay_decisions.tsv": output / "assay_decisions.tsv",
        "structure_anchor_provenance.json": output / "structure_anchor_provenance.json",
    }
    result.pair_evidence.to_csv(paths["structure_anchor_pair_evidence.tsv"], sep="\t", index=False)
    result.typed_groups.to_csv(paths["typed_structure_anchored_groups.tsv"], sep="\t", index=False)
    result.assay_decisions.to_csv(paths["assay_decisions.tsv"], sep="\t", index=False)
    paths["structure_anchor_provenance.json"].write_text(
        json.dumps(result.provenance, indent=2, sort_keys=True) + "\n"
    )
    source = Path(__file__).resolve(strict=True)
    inputs = {
        "consensus_edges.tsv": consensus_path,
        "within_group_candidates.tsv": candidate_path,
        "subspace_feature_membership.tsv": membership_path,
        **{f"empirical_null_view_{index:02d}": path for index, path in enumerate(null_paths)},
    }
    manifest = {
        "schema_version": "structure_anchored_loading_groups_v4.frozen_candidate.v5",
        "algorithm_source_sha256": _sha256(source),
        "config": asdict(config),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_sha256": {label: _sha256(path) for label, path in sorted(inputs.items())},
        "output_sha256": {name: _sha256(path) for name, path in sorted(paths.items())},
        "candidate_frozen": True,
        "truth_or_scorer_consumed_by_algorithm": False,
    }
    manifest_path = output / "frozen_structure_anchor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-edges", type=Path, required=True)
    parser.add_argument("--within-group-candidates", type=Path, required=True)
    parser.add_argument("--empirical-null-view", type=Path, action="append", required=True)
    parser.add_argument("--state-membership", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_structure_anchored_groups(
        args.consensus_edges,
        args.within_group_candidates,
        args.empirical_null_view,
        args.state_membership,
        args.out_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "StructureAnchoredGroupConfig",
    "StructureAnchoredGroupResult",
    "estimate_structure_anchored_groups",
    "run_structure_anchored_groups",
]
