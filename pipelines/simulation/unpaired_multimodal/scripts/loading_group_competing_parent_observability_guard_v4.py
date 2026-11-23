#!/usr/bin/env python3
"""CA-BOSS v4.31 guard against biased collider-parent competition.

The v4.30 loading-group guard can retain a state-route group containing one
parent of a two-parent collider and one non-collider state reporter.  Choosing
between those two collider parents is not a fair comparison when the omitted
parent is not itself observable as a depth-orthogonal state-responsive direct
candidate: the detectable parent can win merely because its competitor is
below the assay's observation floor.

This post-v4.30 guard therefore requires the *competing* (unselected) parent
to pass the already frozen v4.27 observable conjunction:

* the focal feature is formally testable;
* marginal and depth-conditional state fits are positive; and
* the feature participates in the within-assay direct-candidate layer.

The requirement applies only to v4.30 groups retained by the all-members-state
route.  A group retained through independent leave-one-group-out cross-assay
axis evidence bypasses it.  Failure is reported as ``unresolved``; it is not
evidence that the biological association is absent.  The rule uses no roles,
truth, paired cells, external data, or exact cross-assay feature identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import loading_group_state_guard_v4 as state_axis_guard


SCHEMA = "assay_local_loading_group_competing_parent_observability_guard.v4.31"
STATE_ROUTE = state_axis_guard.STATE_ROUTE
AXIS_ROUTE = state_axis_guard.AXIS_ROUTE
FIXED_RULE = (
    "v430_retained_axis_route_bypasses_competitor_gate_or_v430_retained_state_"
    "route_requires_complete_two_parent_collider_and_unselected_parent_passes_"
    "frozen_v427_topology_consensus_fixed_rule_eligible"
)

STATE_RETAINED_STATUS = (
    "retained_state_route_after_competing_parent_observability_guard"
)
AXIS_RETAINED_STATUS = (
    "retained_axis_route_after_competing_parent_observability_guard"
)
UPSTREAM_REJECTED_STATUS = "rejected_by_v430_structure_uncertainty_guard"
COMPETITOR_UNOBSERVABLE_STATUS = (
    "unresolved_competing_parent_not_formally_observable"
)
COMPARISON_UNAVAILABLE_STATUS = (
    "unresolved_complete_two_parent_comparison_unavailable"
)

_DECISION_COLUMNS = {
    "loading_group",
    "modality",
    "members",
    "retained",
    "guard_route",
    "guard_status",
}
_STRUCTURE_COLUMNS = {
    "pair_key",
    "opposite_parent_pair_key",
    "modality",
    "parent_candidate",
    "noncollider_sibling_candidate",
    "collider_center",
    "other_parent_candidate",
    "selected",
    "selection_stage",
}
_FEATURE_COLUMNS = {
    "feature",
    "modality",
    "formal_test_eligible",
    "oof_marginal_state_fit",
    "oof_partial_state_fit",
    "within_assay_direct_candidate_degree",
    "topology_consensus_state_specific_gate",
    "topology_consensus_residual_structure_support",
    "topology_consensus_fixed_rule_eligible",
}


@dataclass(frozen=True)
class LoadingGroupCompetingParentObservabilityGuardResult:
    decisions: pd.DataFrame

    @property
    def retained_group_ids(self) -> set[str]:
        return set(
            self.decisions.loc[
                self.decisions["retained"].astype(bool), "loading_group"
            ].astype(str)
        )

    @property
    def rejected_group_ids(self) -> set[str]:
        return set(
            self.decisions.loc[
                ~self.decisions["retained"].astype(bool), "loading_group"
            ].astype(str)
        )


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _boolean(values: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{label} must contain Boolean values")
    return normalized.map(mapping).astype(bool)


def _canonical_members(values: Iterable[str]) -> tuple[str, ...]:
    members = tuple(
        sorted({str(value).strip() for value in values if str(value).strip()})
    )
    if len(members) < 2:
        raise ValueError("A guarded loading group must contain at least two members")
    return members


def _pair_key(values: Iterable[str]) -> str:
    return "|||".join(_canonical_members(values))


def apply_loading_group_competing_parent_observability_guard(
    v430_decisions: pd.DataFrame,
    structure_anchor_pair_evidence: pd.DataFrame,
    feature_evidence: pd.DataFrame,
) -> LoadingGroupCompetingParentObservabilityGuardResult:
    """Apply the fixed v4.31 fair-comparison prerequisite."""

    _require(v430_decisions, _DECISION_COLUMNS, "v4.30 decisions")
    _require(
        structure_anchor_pair_evidence,
        _STRUCTURE_COLUMNS,
        "structure-anchor pair evidence",
    )
    _require(feature_evidence, _FEATURE_COLUMNS, "feature evidence")

    decisions = v430_decisions.copy()
    structure = structure_anchor_pair_evidence.copy()
    features = feature_evidence.copy()
    for frame, columns in (
        (decisions, ("loading_group", "modality", "members", "guard_route")),
        (
            structure,
            (
                "pair_key",
                "opposite_parent_pair_key",
                "modality",
                "parent_candidate",
                "noncollider_sibling_candidate",
                "collider_center",
                "other_parent_candidate",
                "selection_stage",
            ),
        ),
        (features, ("feature", "modality")),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)

    if decisions["loading_group"].duplicated().any():
        raise ValueError("v4.30 decisions contain duplicate loading groups")
    if features["feature"].duplicated().any():
        raise ValueError("feature evidence contains duplicate features")
    for column in (
        "formal_test_eligible",
        "topology_consensus_state_specific_gate",
        "topology_consensus_residual_structure_support",
        "topology_consensus_fixed_rule_eligible",
    ):
        features[column] = _boolean(features[column], label=f"features.{column}")
    for column in ("oof_marginal_state_fit", "oof_partial_state_fit"):
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"features.{column} must be finite")
    degree = pd.to_numeric(
        features["within_assay_direct_candidate_degree"], errors="raise"
    )
    if (
        not np.isfinite(degree.to_numpy(dtype=float)).all()
        or (degree < 0).any()
        or not np.equal(degree, np.floor(degree)).all()
    ):
        raise ValueError(
            "within_assay_direct_candidate_degree must contain nonnegative integers"
        )
    features["within_assay_direct_candidate_degree"] = degree.astype(int)
    reconstructed_state_gate = (
        features["oof_marginal_state_fit"].gt(0.0)
        & features["oof_partial_state_fit"].gt(0.0)
    )
    reconstructed_residual_gate = features[
        "within_assay_direct_candidate_degree"
    ].ge(1)
    reconstructed_fixed = (
        features["formal_test_eligible"]
        & reconstructed_state_gate
        & reconstructed_residual_gate
    )
    if not reconstructed_state_gate.equals(
        features["topology_consensus_state_specific_gate"]
    ):
        raise ValueError("feature state-specific gate disagrees with its fixed rule")
    if not reconstructed_residual_gate.equals(
        features["topology_consensus_residual_structure_support"]
    ):
        raise ValueError("feature residual-structure gate disagrees with its fixed rule")
    if not reconstructed_fixed.equals(
        features["topology_consensus_fixed_rule_eligible"]
    ):
        raise ValueError("feature fixed-rule eligibility cannot be reconstructed")

    structure["selected"] = _boolean(
        structure["selected"], label="structure.selected"
    )
    selected_structure = structure.loc[structure["selected"]].copy()
    if selected_structure.duplicated(["modality", "pair_key"]).any():
        raise ValueError("selected structure evidence contains duplicate pair keys")
    for row in selected_structure.itertuples(index=False):
        parent = str(row.parent_candidate)
        sibling = str(row.noncollider_sibling_candidate)
        competitor = str(row.other_parent_candidate)
        center = str(row.collider_center)
        if len({parent, sibling, competitor, center}) != 4:
            raise ValueError("selected structure competition must contain four nodes")
        if str(row.pair_key) != _pair_key((parent, sibling)):
            raise ValueError("selected structure pair key cannot be reconstructed")
        if str(row.opposite_parent_pair_key) != _pair_key((competitor, sibling)):
            raise ValueError("opposite-parent pair key cannot be reconstructed")

    feature_lookup = features.set_index("feature", drop=False)
    rows: list[dict[str, object]] = []
    for upstream in decisions.sort_values(
        ["modality", "loading_group"]
    ).to_dict(orient="records"):
        group_id = str(upstream["loading_group"])
        modality = str(upstream["modality"])
        members = _canonical_members(str(upstream["members"]).split(";"))
        member_set = set(members)
        upstream_retained = bool(upstream["retained"])
        upstream_route = str(upstream["guard_route"])
        if upstream_retained and upstream_route not in {STATE_ROUTE, AXIS_ROUTE}:
            raise ValueError(
                f"Retained v4.30 group has an unknown route: {upstream_route}"
            )
        if not upstream_retained and upstream_route != "unresolved":
            raise ValueError(
                f"Rejected v4.30 group has a non-unresolved route: {upstream_route}"
            )

        group_pair_key = _pair_key(members)
        comparisons = selected_structure.loc[
            selected_structure["modality"].eq(modality)
            & selected_structure["pair_key"].eq(group_pair_key)
        ]
        comparison_available = len(comparisons) == 1
        selected_parent = competing_parent = collider_child = sibling = ""
        competition_selection_stage = ""
        competitor_formal = False
        competitor_state_gate = False
        competitor_residual_gate = False
        competitor_fixed = False
        competitor_marginal_fit = float("nan")
        competitor_partial_fit = float("nan")
        competitor_degree = 0
        if comparison_available:
            comparison = comparisons.iloc[0]
            selected_parent = str(comparison["parent_candidate"])
            competing_parent = str(comparison["other_parent_candidate"])
            collider_child = str(comparison["collider_center"])
            sibling = str(comparison["noncollider_sibling_candidate"])
            competition_selection_stage = str(comparison["selection_stage"])
            if {selected_parent, sibling} != member_set:
                raise ValueError(
                    f"Selected structure pair disagrees with group {group_id}"
                )
            if competing_parent not in feature_lookup.index:
                raise ValueError(
                    f"Competing parent lacks feature evidence: {competing_parent}"
                )
            competitor = feature_lookup.loc[competing_parent]
            if str(competitor["modality"]) != modality:
                raise ValueError(
                    f"Competing-parent modality disagrees for {group_id}"
                )
            competitor_formal = bool(competitor["formal_test_eligible"])
            competitor_state_gate = bool(
                competitor["topology_consensus_state_specific_gate"]
            )
            competitor_residual_gate = bool(
                competitor["topology_consensus_residual_structure_support"]
            )
            competitor_fixed = bool(
                competitor["topology_consensus_fixed_rule_eligible"]
            )
            competitor_marginal_fit = float(
                competitor["oof_marginal_state_fit"]
            )
            competitor_partial_fit = float(competitor["oof_partial_state_fit"])
            competitor_degree = int(
                competitor["within_assay_direct_candidate_degree"]
            )

        requirement_active = bool(
            upstream_retained and upstream_route == STATE_ROUTE
        )
        passes_requirement = bool(
            not requirement_active
            or (comparison_available and competitor_fixed)
        )
        retained = bool(upstream_retained and passes_requirement)
        if retained and upstream_route == STATE_ROUTE:
            guard_status = STATE_RETAINED_STATUS
            guard_route = STATE_ROUTE
        elif retained and upstream_route == AXIS_ROUTE:
            guard_status = AXIS_RETAINED_STATUS
            guard_route = AXIS_ROUTE
        elif not upstream_retained:
            guard_status = UPSTREAM_REJECTED_STATUS
            guard_route = "unresolved"
        elif not comparison_available:
            guard_status = COMPARISON_UNAVAILABLE_STATUS
            guard_route = "unresolved"
        else:
            guard_status = COMPETITOR_UNOBSERVABLE_STATUS
            guard_route = "unresolved"

        output = dict(upstream)
        output["v430_retained"] = upstream_retained
        output["v430_guard_route"] = upstream_route
        output["v430_guard_status"] = str(upstream["guard_status"])
        output["competing_parent_observability_required"] = requirement_active
        output["complete_two_parent_comparison_available"] = comparison_available
        output["selected_parent_candidate"] = selected_parent
        output["competing_parent_candidate"] = competing_parent
        output["noncollider_sibling_candidate"] = sibling
        output["collider_child_candidate"] = collider_child
        output["competition_selection_stage"] = competition_selection_stage
        output["competing_parent_formal_test_eligible"] = competitor_formal
        output["competing_parent_oof_marginal_state_fit"] = competitor_marginal_fit
        output["competing_parent_oof_partial_state_fit"] = competitor_partial_fit
        output["competing_parent_direct_candidate_degree"] = competitor_degree
        output["competing_parent_state_specific_gate"] = competitor_state_gate
        output["competing_parent_residual_structure_support"] = (
            competitor_residual_gate
        )
        output["competing_parent_fixed_rule_eligible"] = competitor_fixed
        output["passes_competing_parent_observability_guard"] = passes_requirement
        output["retained"] = retained
        output["guard_route"] = guard_route
        output["guard_status"] = guard_status
        rows.append(output)

    return LoadingGroupCompetingParentObservabilityGuardResult(
        decisions=pd.DataFrame(rows).reset_index(drop=True)
    )


__all__ = [
    "AXIS_RETAINED_STATUS",
    "AXIS_ROUTE",
    "COMPARISON_UNAVAILABLE_STATUS",
    "COMPETITOR_UNOBSERVABLE_STATUS",
    "FIXED_RULE",
    "LoadingGroupCompetingParentObservabilityGuardResult",
    "SCHEMA",
    "STATE_RETAINED_STATUS",
    "STATE_ROUTE",
    "UPSTREAM_REJECTED_STATUS",
    "apply_loading_group_competing_parent_observability_guard",
]
