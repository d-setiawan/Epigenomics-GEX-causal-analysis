#!/usr/bin/env python3
"""Truth-free certification for assay-local loading groups.

CA-BOSS v4.29 keeps the v4.28 all-members-state route and adds one bounded
route for groups that v4.28 would otherwise reject.  A group may also be
retained when it has at least one v4.27 state-supported member and independently
joins a *selected cross-assay latent axis* under leave-one-group-out evidence.
The axis must contain at least two groups and two modalities; the candidate
group is compared with an axis estimated without that group.  This is a
group-level cross-assay consensus claim, not an exact feature correspondence.

The dependency remains one-way: v4.27 membership and an already frozen axis
may certify a loading group, while the loading group never changes v4.27
membership or direct structure.  A structure-permitted pair alone therefore
remains insufficient.  The guard uses no scorer, simulator role, paired cells,
cross-assay feature pairing, or direct-edge truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


SCHEMA = "assay_local_loading_group_state_guard.v4.29"
STATE_ROUTE = "all_loading_group_members_are_state_v427_supported"
AXIS_ROUTE = (
    "selected_leave_one_group_out_cross_assay_axis_consensus_with_at_least_"
    "one_state_v427_supported_member"
)
FIXED_RULE = f"{STATE_ROUTE}_or_{AXIS_ROUTE}"
RETAINED_STATUS = "retained_all_members_state_v427_supported"
AXIS_RETAINED_STATUS = "retained_cross_assay_axis_consensus"
REJECTED_STATUS = "rejected_no_state_or_cross_assay_axis_consensus"

_GROUP_COLUMNS = {
    "loading_group",
    "modality",
    "members",
    "selection_status",
}
_MEMBERSHIP_COLUMNS = {"loading_group", "feature", "modality"}
_FEATURE_COLUMNS = {"feature", "modality", "state_v427_supported_member"}
_GROUP_AXIS_COLUMNS = {
    "loading_group",
    "group_modality",
    "axis_id",
    "is_cluster_member",
    "leave_one_group_out_for_member",
    "modality_conflict_with_axis",
    "lower_absolute_group_axis_cosine",
    "group_axis_view_support",
    "relative_axis_margin",
    "passes_member_lower_cosine",
    "passes_member_view_support",
    "passes_relative_axis_margin",
    "selection_status",
}
_AXIS_COLUMNS = {
    "axis_id",
    "loading_groups",
    "modalities",
    "n_groups",
    "n_modalities",
    "singleton_only_axis",
    "axis_confidence",
    "selection_status",
}


@dataclass(frozen=True)
class LoadingGroupStateGuardResult:
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
    members = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    if len(members) < 2:
        raise ValueError("A guarded loading group must contain at least two members")
    return members


def apply_loading_group_state_guard(
    groups: pd.DataFrame,
    group_membership: pd.DataFrame,
    feature_evidence: pd.DataFrame,
    group_axis_evidence: pd.DataFrame | None = None,
    axis_candidates: pd.DataFrame | None = None,
) -> LoadingGroupStateGuardResult:
    """Return fixed v4.29 decisions for preselected multi-feature groups.

    Omitting both axis tables is a safe compatibility mode: only the v4.28
    all-members-state route can retain a group.  Supplying exactly one table is
    rejected because partial axis provenance must never activate the new route.
    """

    _require(groups, _GROUP_COLUMNS, "loading groups")
    _require(group_membership, _MEMBERSHIP_COLUMNS, "loading-group membership")
    _require(feature_evidence, _FEATURE_COLUMNS, "feature evidence")
    if (group_axis_evidence is None) != (axis_candidates is None):
        raise ValueError(
            "group-axis evidence and axis candidates must be supplied together"
        )
    axis_route_active = group_axis_evidence is not None
    if axis_route_active:
        assert group_axis_evidence is not None
        assert axis_candidates is not None
        _require(
            group_axis_evidence,
            _GROUP_AXIS_COLUMNS,
            "group-axis evidence",
        )
        _require(axis_candidates, _AXIS_COLUMNS, "axis candidates")

    groups = groups.copy()
    membership = group_membership.copy()
    features = feature_evidence.copy()
    for frame, columns in (
        (groups, ("loading_group", "modality", "members")),
        (membership, ("loading_group", "feature", "modality")),
        (features, ("feature", "modality")),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)

    if groups["loading_group"].duplicated().any():
        raise ValueError("loading groups contain duplicate identifiers")
    if membership.duplicated(["loading_group", "feature"]).any():
        raise ValueError("loading-group membership contains duplicate rows")
    if features["feature"].duplicated().any():
        raise ValueError("feature evidence contains duplicate features")

    selected_axis_rows: dict[str, dict[str, object]] = {}
    if axis_route_active:
        assert group_axis_evidence is not None
        assert axis_candidates is not None
        group_axes = group_axis_evidence.copy()
        axes = axis_candidates.copy()
        for frame, columns in (
            (group_axes, ("loading_group", "group_modality", "axis_id")),
            (axes, ("axis_id", "loading_groups", "modalities")),
        ):
            for column in columns:
                frame[column] = frame[column].astype(str)
        if axes["axis_id"].duplicated().any():
            raise ValueError("axis candidates contain duplicate identifiers")
        if group_axes.duplicated(["loading_group", "axis_id"]).any():
            raise ValueError("group-axis evidence contains duplicate rows")

        for column in (
            "is_cluster_member",
            "leave_one_group_out_for_member",
            "modality_conflict_with_axis",
            "passes_member_lower_cosine",
            "passes_member_view_support",
            "passes_relative_axis_margin",
        ):
            group_axes[column] = group_axes[column].astype(bool)
        axes["singleton_only_axis"] = axes["singleton_only_axis"].astype(bool)
        for column in ("n_groups", "n_modalities"):
            numeric = pd.to_numeric(axes[column], errors="raise")
            if (numeric < 0).any() or not numeric.eq(numeric.round()).all():
                raise ValueError(f"{column} must contain nonnegative integers")
            axes[column] = numeric.astype(int)
        for frame, columns in (
            (
                group_axes,
                (
                    "lower_absolute_group_axis_cosine",
                    "group_axis_view_support",
                    "relative_axis_margin",
                ),
            ),
            (axes, ("axis_confidence",)),
        ):
            for column in columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise")

        selected_axes = axes.loc[
            axes["selection_status"].astype(str).eq("selected")
            & axes["n_groups"].ge(2)
            & axes["n_modalities"].ge(2)
            & ~axes["singleton_only_axis"]
        ].set_index("axis_id", drop=False)
        qualifying = group_axes.loc[
            group_axes["selection_status"].astype(str).eq("selected")
            & group_axes["is_cluster_member"]
            & group_axes["leave_one_group_out_for_member"]
            & ~group_axes["modality_conflict_with_axis"]
            & group_axes["passes_member_lower_cosine"]
            & group_axes["passes_member_view_support"]
            & group_axes["passes_relative_axis_margin"]
            & group_axes["axis_id"].isin(selected_axes.index)
        ]
        for group_id, frame in qualifying.groupby("loading_group", sort=True):
            if len(frame) != 1:
                raise ValueError(
                    f"Loading group {group_id} joins multiple selected axes"
                )
            row = frame.iloc[0]
            axis = selected_axes.loc[str(row["axis_id"])]
            declared_groups = {
                value.strip()
                for value in str(axis["loading_groups"]).split(";")
                if value.strip()
            }
            declared_modalities = {
                value.strip()
                for value in str(axis["modalities"]).split(";")
                if value.strip()
            }
            if str(group_id) not in declared_groups:
                raise ValueError(
                    f"Axis membership does not list loading group {group_id}"
                )
            if str(row["group_modality"]) not in declared_modalities:
                raise ValueError(
                    f"Axis modalities do not list {row['group_modality']}"
                )
            selected_axis_rows[str(group_id)] = {
                "axis_id": str(row["axis_id"]),
                "axis_modalities": ";".join(sorted(declared_modalities)),
                "axis_n_groups": int(axis["n_groups"]),
                "axis_n_modalities": int(axis["n_modalities"]),
                "axis_confidence": float(axis["axis_confidence"]),
                "group_axis_lower_cosine": float(
                    row["lower_absolute_group_axis_cosine"]
                ),
                "group_axis_view_support": float(
                    row["group_axis_view_support"]
                ),
                "group_axis_relative_margin": float(
                    row["relative_axis_margin"]
                ),
            }

    feature_modality = features.set_index("feature")["modality"].to_dict()
    state_supported = set(
        features.loc[
            features["state_v427_supported_member"].astype(bool), "feature"
        ].astype(str)
    )
    unknown = sorted(set(membership["feature"]).difference(feature_modality))
    if unknown:
        raise ValueError(f"loading-group membership contains unknown features: {unknown}")
    for row in membership.itertuples(index=False):
        if feature_modality[str(row.feature)] != str(row.modality):
            raise ValueError(
                f"Loading-group modality disagrees for feature {row.feature}"
            )

    member_sets = {
        str(group_id): _canonical_members(frame["feature"].astype(str))
        for group_id, frame in membership.groupby("loading_group", sort=True)
        if frame["feature"].nunique() >= 2
    }
    preselected = groups.loc[
        groups["selection_status"].astype(str).eq("selected")
        & groups["loading_group"].isin(member_sets)
    ].sort_values(["modality", "loading_group"])

    rows: list[dict[str, object]] = []
    for group in preselected.itertuples(index=False):
        group_id = str(group.loading_group)
        members = member_sets[group_id]
        declared = _canonical_members(str(group.members).split(";"))
        if declared != members:
            raise ValueError(
                f"Loading-group member table disagrees for {group_id}"
            )
        modalities = {feature_modality[member] for member in members}
        if modalities != {str(group.modality)}:
            raise ValueError(f"Loading group {group_id} crosses modalities")
        supported_members = tuple(
            member for member in members if member in state_supported
        )
        all_members_supported = len(supported_members) == len(members)
        axis_payload = selected_axis_rows.get(group_id)
        cross_assay_axis_consensus = bool(
            axis_payload is not None
            and 0 < len(supported_members) < len(members)
        )
        retained = all_members_supported or cross_assay_axis_consensus
        if all_members_supported:
            guard_route = STATE_ROUTE
            guard_status = RETAINED_STATUS
        elif cross_assay_axis_consensus:
            guard_route = AXIS_ROUTE
            guard_status = AXIS_RETAINED_STATUS
        else:
            guard_route = "unresolved"
            guard_status = REJECTED_STATUS
        axis_payload = axis_payload if cross_assay_axis_consensus else {}
        rows.append(
            {
                "loading_group": group_id,
                "modality": str(group.modality),
                "members": ";".join(members),
                "n_members": len(members),
                "n_state_v427_supported_members": len(supported_members),
                "unsupported_members": ";".join(
                    member for member in members if member not in state_supported
                ),
                "pre_v428_selected": True,
                "all_members_state_v427_supported": all_members_supported,
                "cross_assay_axis_consensus": cross_assay_axis_consensus,
                "cross_assay_axis_id": axis_payload.get("axis_id", ""),
                "cross_assay_axis_modalities": axis_payload.get(
                    "axis_modalities", ""
                ),
                "cross_assay_axis_n_groups": axis_payload.get(
                    "axis_n_groups", 0
                ),
                "cross_assay_axis_n_modalities": axis_payload.get(
                    "axis_n_modalities", 0
                ),
                "cross_assay_axis_confidence": axis_payload.get(
                    "axis_confidence", 0.0
                ),
                "group_axis_lower_cosine": axis_payload.get(
                    "group_axis_lower_cosine", 0.0
                ),
                "group_axis_view_support": axis_payload.get(
                    "group_axis_view_support", 0.0
                ),
                "group_axis_relative_margin": axis_payload.get(
                    "group_axis_relative_margin", 0.0
                ),
                "guard_route": guard_route,
                "retained": retained,
                "guard_status": guard_status,
                "fixed_rule": FIXED_RULE,
                "guard_schema": SCHEMA,
                "direct_structure_alone_is_insufficient": True,
                "changes_state_v427_membership": False,
                "exact_cross_assay_pair_claimed": False,
                "truth_used": False,
            }
        )

    decisions = pd.DataFrame(
        rows,
        columns=[
            "loading_group",
            "modality",
            "members",
            "n_members",
            "n_state_v427_supported_members",
            "unsupported_members",
            "pre_v428_selected",
            "all_members_state_v427_supported",
            "cross_assay_axis_consensus",
            "cross_assay_axis_id",
            "cross_assay_axis_modalities",
            "cross_assay_axis_n_groups",
            "cross_assay_axis_n_modalities",
            "cross_assay_axis_confidence",
            "group_axis_lower_cosine",
            "group_axis_view_support",
            "group_axis_relative_margin",
            "guard_route",
            "retained",
            "guard_status",
            "fixed_rule",
            "guard_schema",
            "direct_structure_alone_is_insufficient",
            "changes_state_v427_membership",
            "exact_cross_assay_pair_claimed",
            "truth_used",
        ],
    )
    return LoadingGroupStateGuardResult(decisions=decisions)


__all__ = [
    "AXIS_RETAINED_STATUS",
    "AXIS_ROUTE",
    "FIXED_RULE",
    "LoadingGroupStateGuardResult",
    "REJECTED_STATUS",
    "RETAINED_STATUS",
    "SCHEMA",
    "STATE_ROUTE",
    "apply_loading_group_state_guard",
]
