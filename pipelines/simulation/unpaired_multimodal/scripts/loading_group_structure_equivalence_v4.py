#!/usr/bin/env python3
"""CA-BOSS v4.32 set-valued loading/structure equivalence layer.

Disjoint-cell assays can support an assay-local shared-state hypothesis while
leaving the exact roles inside a two-parent collider motif unidentified.  A
v4.31 state-route loading group contains the locally selected parent and the
non-collider sibling.  The same frozen structure record also exposes the
competing parent and the collider child.  When both parents pass the v4.31
fair-comparison prerequisite, the observed data support the Cartesian product

    {selected parent, competing parent}
        x {non-collider sibling, collider child}

as a four-pair structural equivalence class.  V4.32 emits that object as one
typed hypothesis.  It does *not* expand it into four edges, select a causal
direction, or claim four biological relationships.  Loading groups retained
by the independent leave-one-group-out cross-assay-axis route remain exact
assay-local groups because that route does not depend on the collider-parent
competition.

The transformation is truth-free and has no fitted numeric threshold.  It
uses only fields already produced by the v4.31 guard.  Upstream abstentions
remain unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import pandas as pd

import loading_group_competing_parent_observability_guard_v4 as v431


SCHEMA = "assay_local_loading_group_structure_equivalence.v4.32"
ALGORITHM = "CA-BOSS v4.32"
STATE_ROUTE = v431.STATE_ROUTE
AXIS_ROUTE = v431.AXIS_ROUTE
FIXED_RULE = (
    "v431_retained_axis_route_remains_exact_or_v431_retained_state_route_is_"
    "represented_as_two_by_two_selected_competing_parent_cross_"
    "sibling_collider_structure_equivalence"
)

STATE_EQUIVALENCE_TYPE = "within_assay_state_structure_equivalence"
AXIS_EXACT_TYPE = "within_assay_shared_state_loading_group"
UNRESOLVED_TYPE = "unresolved"
STATE_EQUIVALENCE_STATUS = (
    "retained_as_two_by_two_state_structure_equivalence"
)
AXIS_EXACT_STATUS = "retained_exact_assay_local_group_by_cross_assay_axis"
UPSTREAM_UNRESOLVED_STATUS = "unresolved_by_v431_guard"

REQUIRED_COLUMNS = {
    "loading_group",
    "modality",
    "members",
    "retained",
    "guard_route",
    "guard_status",
    "complete_two_parent_comparison_available",
    "selected_parent_candidate",
    "competing_parent_candidate",
    "noncollider_sibling_candidate",
    "collider_child_candidate",
    "passes_competing_parent_observability_guard",
}
_FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "oracle",
    "scorer",
    "ground_truth",
    "true_role",
    "sentinel",
)


@dataclass(frozen=True)
class LoadingGroupStructureEquivalenceResult:
    decisions: pd.DataFrame

    @property
    def selected_hypotheses(self) -> pd.DataFrame:
        return self.decisions.loc[
            self.decisions["hypothesis_selected"].astype(bool)
        ].copy()

    @property
    def structure_equivalence_hyperedges(self) -> pd.DataFrame:
        return self.decisions.loc[
            self.decisions["structure_equivalence_selected"].astype(bool)
        ].copy()

    @property
    def exact_loading_groups(self) -> pd.DataFrame:
        return self.decisions.loc[
            self.decisions["exact_loading_group_selected"].astype(bool)
        ].copy()


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _boolean(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be Boolean")


def _canonical_members(values: Iterable[str], *, expected: int | None = None) -> tuple[str, ...]:
    members = tuple(
        sorted({str(value).strip() for value in values if str(value).strip()})
    )
    if expected is not None and len(members) != expected:
        raise ValueError(f"Expected exactly {expected} distinct members, found {len(members)}")
    if len(members) < 2:
        raise ValueError("A loading hypothesis must contain at least two members")
    return members


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return _canonical_members((left, right), expected=2)


def pair_key(left: str, right: str) -> str:
    return "|||".join(canonical_pair(left, right))


def _candidate_pairs_json(pairs: Iterable[tuple[str, str]]) -> str:
    canonical = sorted({canonical_pair(*pair) for pair in pairs})
    return json.dumps([list(pair) for pair in canonical], separators=(",", ":"))


def parse_candidate_pairs(value: str) -> tuple[tuple[str, str], ...]:
    """Parse and validate a serialized v4.32 candidate-pair set."""

    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("candidate_pairs_json must encode a list")
    pairs = tuple(canonical_pair(*pair) for pair in payload)
    if tuple(sorted(set(pairs))) != pairs:
        raise ValueError("candidate_pairs_json must be unique and canonically sorted")
    return pairs


def build_loading_group_structure_equivalence(
    v431_decisions: pd.DataFrame,
) -> LoadingGroupStructureEquivalenceResult:
    """Convert v4.31 point groups into exact or set-valued typed hypotheses."""

    _require(v431_decisions, REQUIRED_COLUMNS, "v4.31 decisions")
    forbidden = sorted(
        column
        for column in v431_decisions.columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Truth/scorer columns are forbidden: {forbidden}")

    decisions = v431_decisions.copy()
    if decisions["loading_group"].astype(str).duplicated().any():
        raise ValueError("v4.31 decisions contain duplicate loading groups")

    rows: list[dict[str, object]] = []
    for upstream in decisions.sort_values(
        ["modality", "loading_group"]
    ).to_dict(orient="records"):
        group_id = str(upstream["loading_group"])
        modality = str(upstream["modality"])
        nominal_members = _canonical_members(
            str(upstream["members"]).split(";"), expected=2
        )
        retained = _boolean(
            upstream["retained"], label=f"{group_id}.retained"
        )
        route = str(upstream["guard_route"])
        comparison_available = _boolean(
            upstream["complete_two_parent_comparison_available"],
            label=f"{group_id}.complete_two_parent_comparison_available",
        )
        passes_guard = _boolean(
            upstream["passes_competing_parent_observability_guard"],
            label=f"{group_id}.passes_competing_parent_observability_guard",
        )

        hypothesis_selected = retained
        exact_selected = False
        equivalence_selected = False
        exact_identified = False
        hypothesis_type = UNRESOLVED_TYPE
        status = UPSTREAM_UNRESOLVED_STATUS
        parent_members: tuple[str, ...] = ()
        counterpart_members: tuple[str, ...] = ()
        candidate_pairs: tuple[tuple[str, str], ...] = ()

        if retained and route == AXIS_ROUTE:
            exact_selected = True
            exact_identified = True
            hypothesis_type = AXIS_EXACT_TYPE
            status = AXIS_EXACT_STATUS
            candidate_pairs = (canonical_pair(*nominal_members),)
        elif retained and route == STATE_ROUTE:
            if not comparison_available or not passes_guard:
                raise ValueError(
                    f"Retained v4.31 state route lacks its fair comparison: {group_id}"
                )
            selected_parent = str(upstream["selected_parent_candidate"]).strip()
            competing_parent = str(upstream["competing_parent_candidate"]).strip()
            sibling = str(upstream["noncollider_sibling_candidate"]).strip()
            collider = str(upstream["collider_child_candidate"]).strip()
            if len({selected_parent, competing_parent, sibling, collider}) != 4:
                raise ValueError(
                    f"State-route structure equivalence must contain four nodes: {group_id}"
                )
            if set(nominal_members) != {selected_parent, sibling}:
                raise ValueError(
                    f"Nominal v4.31 group disagrees with selected parent/sibling: {group_id}"
                )
            parent_members = _canonical_members(
                (selected_parent, competing_parent), expected=2
            )
            counterpart_members = _canonical_members(
                (sibling, collider), expected=2
            )
            candidate_pairs = tuple(
                sorted(
                    canonical_pair(parent, counterpart)
                    for parent in parent_members
                    for counterpart in counterpart_members
                )
            )
            if len(set(candidate_pairs)) != 4:
                raise ValueError(
                    f"State-route equivalence did not produce four pairs: {group_id}"
                )
            equivalence_selected = True
            hypothesis_type = STATE_EQUIVALENCE_TYPE
            status = STATE_EQUIVALENCE_STATUS
        elif retained:
            raise ValueError(f"Retained v4.31 group has unknown route: {route}")
        elif route != "unresolved":
            raise ValueError(
                f"Rejected v4.31 group has non-unresolved route: {route}"
            )

        output = dict(upstream)
        output["v431_retained"] = retained
        output["v431_guard_route"] = route
        output["v431_guard_status"] = str(upstream["guard_status"])
        output["hypothesis_selected"] = hypothesis_selected
        output["hypothesis_type"] = hypothesis_type
        output["hypothesis_status"] = status
        output["nominal_pair_key"] = pair_key(*nominal_members)
        output["parent_equivalence_members"] = ";".join(parent_members)
        output["counterpart_equivalence_members"] = ";".join(counterpart_members)
        output["candidate_pairs_json"] = _candidate_pairs_json(candidate_pairs)
        output["candidate_pair_count"] = len(candidate_pairs)
        output["exact_loading_group_selected"] = exact_selected
        output["structure_equivalence_selected"] = equivalence_selected
        output["exact_pair_identified"] = exact_identified
        output["shared_state_hypothesis_supported"] = hypothesis_selected
        output["candidate_pairs_may_be_expanded_as_edges"] = False
        output["direct_causal_relation_claimed"] = False
        output["exact_cross_assay_pair_claimed"] = False
        output["fixed_rule"] = FIXED_RULE
        output["structure_equivalence_schema"] = SCHEMA
        rows.append(output)

    return LoadingGroupStructureEquivalenceResult(
        decisions=pd.DataFrame(rows).reset_index(drop=True)
    )


__all__ = [
    "ALGORITHM",
    "AXIS_EXACT_STATUS",
    "AXIS_EXACT_TYPE",
    "AXIS_ROUTE",
    "FIXED_RULE",
    "LoadingGroupStructureEquivalenceResult",
    "REQUIRED_COLUMNS",
    "SCHEMA",
    "STATE_EQUIVALENCE_STATUS",
    "STATE_EQUIVALENCE_TYPE",
    "STATE_ROUTE",
    "UNRESOLVED_TYPE",
    "UPSTREAM_UNRESOLVED_STATUS",
    "build_loading_group_structure_equivalence",
    "canonical_pair",
    "pair_key",
    "parse_candidate_pairs",
]
