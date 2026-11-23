#!/usr/bin/env python3
"""Conservative v4.30 guard for assay-local shared-state loading groups.

This guard is applied *after* the v4.29 state/axis guard.  It separates two
remaining sources of ambiguity without changing feature-level state
membership or the within-assay direct graph:

* a v4.29 all-members-state group must retain a frozen minimum
  member-to-centroid cosine of at least 0.90; and
* a group that overlaps the child of a complete directed collider candidate,
  or contains at least two parents of that same child, is downgraded to
  unresolved because direct and shared-state explanations are entangled.

The v4.29 cross-assay-axis route is not subjected to the cosine floor, because
it already uses independent leave-one-group-out cross-assay evidence.  Both
routes are subject to the direct-layer overlap downgrade.  A downgrade is an
uncertainty label, not evidence that the biological association is absent.

The rule consumes observable, pretruth artifacts only.  It creates no edge,
does not alter the direct candidates, and makes no exact cross-assay pairing
or direction claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import loading_group_state_guard_v4 as state_axis_guard


SCHEMA = "assay_local_loading_group_structure_uncertainty_guard.v4.30"
MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE = 0.90
STATE_ROUTE = state_axis_guard.STATE_ROUTE
AXIS_ROUTE = state_axis_guard.AXIS_ROUTE
FIXED_RULE = (
    "retain_v429_axis_route_or_v429_state_route_with_minimum_member_centroid_"
    "cosine_at_least_0.90_then_downgrade_complete_direct_collider_overlap"
)

STATE_RETAINED_STATUS = (
    "retained_state_route_after_structure_uncertainty_guard"
)
AXIS_RETAINED_STATUS = (
    "retained_axis_route_after_structure_uncertainty_guard"
)
UPSTREAM_REJECTED_STATUS = "rejected_by_v429_state_axis_guard"
DIRECT_OVERLAP_STATUS = "unresolved_direct_state_layer_overlap"
LOW_CONFIDENCE_STATUS = "unresolved_below_loading_uncertainty_floor"

_DECISION_COLUMNS = {
    "loading_group",
    "modality",
    "members",
    "retained",
    "guard_route",
    "guard_status",
}
_GROUP_COLUMNS = {
    "loading_group",
    "modality",
    "members",
    "minimum_member_centroid_cosine",
    "selection_status",
}
_DIRECT_COLUMNS = {
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "edge_type",
    "source_assay",
}


@dataclass(frozen=True)
class LoadingGroupStructureUncertaintyGuardResult:
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


def _canonical_members(values: Iterable[str]) -> tuple[str, ...]:
    members = tuple(
        sorted({str(value).strip() for value in values if str(value).strip()})
    )
    if len(members) < 2:
        raise ValueError("A guarded loading group must contain at least two members")
    return members


def _directed_parent_child(row: object) -> tuple[str, str]:
    endpoint1 = str(getattr(row, "endpoint1")).upper()
    endpoint2 = str(getattr(row, "endpoint2")).upper()
    node1 = str(getattr(row, "node1"))
    node2 = str(getattr(row, "node2"))
    if endpoint1 == "TAIL" and endpoint2 == "ARROW":
        return node1, node2
    if endpoint1 == "ARROW" and endpoint2 == "TAIL":
        return node2, node1
    raise ValueError(
        "within-assay direct candidates must have one TAIL and one ARROW endpoint"
    )


def apply_loading_group_structure_uncertainty_guard(
    v429_decisions: pd.DataFrame,
    groups: pd.DataFrame,
    direct_candidates: pd.DataFrame,
) -> LoadingGroupStructureUncertaintyGuardResult:
    """Apply the fixed v4.30 post-v4.29 uncertainty rule."""

    _require(v429_decisions, _DECISION_COLUMNS, "v4.29 decisions")
    _require(groups, _GROUP_COLUMNS, "loading groups")
    _require(direct_candidates, _DIRECT_COLUMNS, "direct candidates")

    decisions = v429_decisions.copy()
    loading_groups = groups.copy()
    direct = direct_candidates.copy()

    for frame, columns in (
        (decisions, ("loading_group", "modality", "members", "guard_route")),
        (loading_groups, ("loading_group", "modality", "members")),
        (direct, ("node1", "endpoint1", "node2", "endpoint2", "edge_type", "source_assay")),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)

    if decisions["loading_group"].duplicated().any():
        raise ValueError("v4.29 decisions contain duplicate loading groups")
    if loading_groups["loading_group"].duplicated().any():
        raise ValueError("loading groups contain duplicate identifiers")

    selected_groups = loading_groups.loc[
        loading_groups["selection_status"].astype(str).eq("selected")
    ].copy()
    selected_ids = set(selected_groups["loading_group"].astype(str))
    decision_ids = set(decisions["loading_group"].astype(str))
    if not decision_ids.issubset(selected_ids):
        unknown = sorted(decision_ids.difference(selected_ids))
        raise ValueError(f"v4.29 decisions contain unknown selected groups: {unknown}")

    selected_groups["minimum_member_centroid_cosine"] = pd.to_numeric(
        selected_groups["minimum_member_centroid_cosine"], errors="raise"
    )
    cosine = selected_groups["minimum_member_centroid_cosine"].to_numpy(
        dtype=float
    )
    if not np.isfinite(cosine).all() or not selected_groups[
        "minimum_member_centroid_cosine"
    ].between(0.0, 1.0).all():
        raise ValueError("minimum_member_centroid_cosine must lie in [0, 1]")

    group_lookup = selected_groups.set_index("loading_group", drop=False)
    direct = direct.loc[
        direct["edge_type"].eq("within_assay_direct_candidate")
    ].copy()
    if direct.duplicated(["source_assay", "node1", "endpoint1", "node2", "endpoint2"]).any():
        raise ValueError("direct candidates contain duplicate directed edges")

    parents_by_assay_child: dict[tuple[str, str], set[str]] = {}
    for row in direct.itertuples(index=False):
        parent, child = _directed_parent_child(row)
        if parent == child:
            raise ValueError("A direct candidate cannot be a self-edge")
        key = (str(row.source_assay), child)
        parents_by_assay_child.setdefault(key, set()).add(parent)
    complete_colliders = {
        key: parents
        for key, parents in parents_by_assay_child.items()
        if len(parents) >= 2
    }

    rows: list[dict[str, object]] = []
    for upstream in decisions.sort_values(
        ["modality", "loading_group"]
    ).to_dict(orient="records"):
        group_id = str(upstream["loading_group"])
        group = group_lookup.loc[group_id]
        members = _canonical_members(str(upstream["members"]).split(";"))
        declared = _canonical_members(str(group["members"]).split(";"))
        if members != declared:
            raise ValueError(f"Loading-group members disagree for {group_id}")
        modality = str(upstream["modality"])
        if str(group["modality"]) != modality:
            raise ValueError(f"Loading-group modality disagrees for {group_id}")

        member_set = set(members)
        child_overlap: set[str] = set()
        shared_parent_children: set[str] = set()
        for (assay, child), parents in complete_colliders.items():
            if assay != modality:
                continue
            if child in member_set:
                child_overlap.add(child)
            if len(member_set.intersection(parents)) >= 2:
                shared_parent_children.add(child)
        direct_layer_conflict = bool(child_overlap or shared_parent_children)

        upstream_retained = bool(upstream["retained"])
        upstream_route = str(upstream["guard_route"])
        if upstream_retained and upstream_route not in {STATE_ROUTE, AXIS_ROUTE}:
            raise ValueError(
                f"Retained v4.29 group has an unknown route: {upstream_route}"
            )
        if not upstream_retained and upstream_route != "unresolved":
            raise ValueError(
                f"Rejected v4.29 group has a non-unresolved route: {upstream_route}"
            )

        group_cosine = float(group["minimum_member_centroid_cosine"])
        passes_uncertainty_floor = bool(
            group_cosine >= MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        )
        state_route_allowed = bool(
            upstream_route == STATE_ROUTE and passes_uncertainty_floor
        )
        axis_route_allowed = bool(upstream_route == AXIS_ROUTE)
        retained = bool(
            upstream_retained
            and not direct_layer_conflict
            and (state_route_allowed or axis_route_allowed)
        )

        if retained and upstream_route == STATE_ROUTE:
            guard_status = STATE_RETAINED_STATUS
            guard_route = STATE_ROUTE
        elif retained and upstream_route == AXIS_ROUTE:
            guard_status = AXIS_RETAINED_STATUS
            guard_route = AXIS_ROUTE
        elif not upstream_retained:
            guard_status = UPSTREAM_REJECTED_STATUS
            guard_route = "unresolved"
        elif direct_layer_conflict:
            guard_status = DIRECT_OVERLAP_STATUS
            guard_route = "unresolved"
        else:
            guard_status = LOW_CONFIDENCE_STATUS
            guard_route = "unresolved"

        output = dict(upstream)
        output["v429_retained"] = upstream_retained
        output["v429_guard_route"] = upstream_route
        output["v429_guard_status"] = str(upstream["guard_status"])
        output["minimum_member_centroid_cosine"] = group_cosine
        output["minimum_state_route_member_centroid_cosine"] = (
            MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        )
        output["passes_loading_uncertainty_floor"] = passes_uncertainty_floor
        output["complete_direct_collider_overlap"] = direct_layer_conflict
        output["direct_collider_child_overlap"] = ";".join(sorted(child_overlap))
        output["shared_direct_parent_children"] = ";".join(
            sorted(shared_parent_children)
        )
        output["retained"] = retained
        output["guard_route"] = guard_route
        output["guard_status"] = guard_status
        rows.append(output)

    return LoadingGroupStructureUncertaintyGuardResult(
        decisions=pd.DataFrame(rows).reset_index(drop=True)
    )


__all__ = [
    "AXIS_RETAINED_STATUS",
    "AXIS_ROUTE",
    "DIRECT_OVERLAP_STATUS",
    "FIXED_RULE",
    "LOW_CONFIDENCE_STATUS",
    "LoadingGroupStructureUncertaintyGuardResult",
    "MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE",
    "SCHEMA",
    "STATE_RETAINED_STATUS",
    "STATE_ROUTE",
    "UPSTREAM_REJECTED_STATUS",
    "apply_loading_group_structure_uncertainty_guard",
]
