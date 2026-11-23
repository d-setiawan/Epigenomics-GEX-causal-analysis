#!/usr/bin/env python3
"""CA-BOSS v4.33 direct-parent/state-reporter equivalence hypotheses.

The disjoint-cell observation model does not identify which parent of an
assay-local collider shares a latent-state axis with an assay-local reporter.
It can nevertheless support a strictly coarser statement when two independent
observable channels agree:

* the frozen direct layer identifies two assay-local parents of one child; and
* the frozen state layer supports at least one parent plus at least one feature
  outside that three-node collider motif.

For each assay, v4.33 returns one set-valued hypothesis

    {direct parent 1, direct parent 2}
        x {state-supported reporters outside the collider motif}.

The Cartesian-product members are alternatives inside one partially identified
hypothesis.  They must never be expanded into several discovered edges.  The
rule does not fit a numeric threshold, use labels, infer a cross-assay pairing,
or add a new directed relation.  An assay is left unresolved when either state
evidence channel is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import pandas as pd


SCHEMA = "assay_local_direct_parent_state_reporter_equivalence.v4.33"
ALGORITHM = "CA-BOSS v4.33"
FIXED_RULE = (
    "exactly_two_frozen_direct_parents_share_one_assay_local_child_and_at_"
    "least_one_parent_is_state_v427_supported_and_at_least_one_state_v427_"
    "supported_reporter_lies_outside_the_direct_collider_motif"
)
HYPOTHESIS_TYPE = "within_assay_direct_parent_state_reporter_equivalence"
UNRESOLVED_TYPE = "unresolved"
SELECTED_STATUS = "retained_as_direct_parent_by_external_state_reporter_equivalence"
NO_PARENT_STATE_STATUS = "unresolved_no_state_supported_direct_parent"
NO_EXTERNAL_REPORTER_STATUS = "unresolved_no_external_state_supported_reporter"

DIRECT_COLUMNS = {"source_assay", "parent", "child"}
STATE_COLUMNS = {"feature", "modality", "state_v427_supported_member"}
_FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "oracle",
    "scorer",
    "ground_truth",
    "true_role",
    "sentinel",
)


@dataclass(frozen=True)
class DirectParentStateReporterResult:
    decisions: pd.DataFrame

    @property
    def selected_hypotheses(self) -> pd.DataFrame:
        return self.decisions.loc[
            self.decisions["hypothesis_selected"].astype(bool)
        ].copy()


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _reject_truth_columns(frame: pd.DataFrame, *, label: str) -> None:
    forbidden = sorted(
        column
        for column in frame.columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"{label} contains truth/scorer columns: {forbidden}")


def _boolean(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be Boolean")


def _canonical_members(
    values: Iterable[str], *, expected: int | None = None
) -> tuple[str, ...]:
    members = tuple(
        sorted({str(value).strip() for value in values if str(value).strip()})
    )
    if expected is not None and len(members) != expected:
        raise ValueError(
            f"Expected exactly {expected} distinct members, found {len(members)}"
        )
    return members


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    pair = _canonical_members((left, right), expected=2)
    return pair


def pair_key(left: str, right: str) -> str:
    return "|||".join(canonical_pair(left, right))


def _pairs_json(pairs: Iterable[tuple[str, str]]) -> str:
    canonical = tuple(sorted({canonical_pair(*pair) for pair in pairs}))
    return json.dumps([list(pair) for pair in canonical], separators=(",", ":"))


def parse_candidate_pairs(value: str) -> tuple[tuple[str, str], ...]:
    """Parse and validate a serialized v4.33 alternative-pair set."""

    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("candidate_pairs_json must encode a list")
    pairs = tuple(canonical_pair(*pair) for pair in payload)
    if tuple(sorted(set(pairs))) != pairs:
        raise ValueError("candidate_pairs_json must be unique and canonically sorted")
    return pairs


def build_direct_parent_state_reporter_equivalence(
    direct_candidates: pd.DataFrame,
    state_membership: pd.DataFrame,
) -> DirectParentStateReporterResult:
    """Build one truth-free partial-identification hypothesis per assay."""

    _require(direct_candidates, DIRECT_COLUMNS, "direct candidates")
    _require(state_membership, STATE_COLUMNS, "state membership")
    _reject_truth_columns(direct_candidates, label="direct candidates")
    _reject_truth_columns(state_membership, label="state membership")

    direct = direct_candidates.loc[:, sorted(DIRECT_COLUMNS)].copy()
    state = state_membership.loc[:, sorted(STATE_COLUMNS)].copy()
    if direct.duplicated(["source_assay", "parent", "child"]).any():
        raise ValueError("Direct candidates contain duplicate directed arms")
    if state.duplicated(["feature"]).any():
        raise ValueError("State membership contains duplicate features")
    for column in ("source_assay", "parent", "child"):
        if (
            direct[column].isna().any()
            or direct[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Direct candidate column {column} contains empty values")
    for column in ("feature", "modality"):
        if (
            state[column].isna().any()
            or state[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"State membership column {column} contains empty values")
    state["state_v427_supported_member"] = state[
        "state_v427_supported_member"
    ].map(lambda value: _boolean(value, label="state_v427_supported_member"))

    endpoint_rows = pd.concat(
        [
            direct.loc[:, ["source_assay", "parent"]].rename(
                columns={"parent": "feature"}
            ),
            direct.loc[:, ["source_assay", "child"]].rename(
                columns={"child": "feature"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    endpoint_check = endpoint_rows.merge(
        state.loc[:, ["feature", "modality"]],
        on="feature",
        how="left",
        validate="many_to_one",
    )
    if endpoint_check["modality"].isna().any():
        missing = sorted(
            endpoint_check.loc[endpoint_check["modality"].isna(), "feature"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Direct endpoints are absent from state membership: {missing}")
    if not endpoint_check["source_assay"].astype(str).equals(
        endpoint_check["modality"].astype(str)
    ):
        raise ValueError("Direct endpoints and state membership disagree on modality")

    rows: list[dict[str, object]] = []
    for modality, arms in direct.groupby("source_assay", sort=True):
        parents = _canonical_members(arms["parent"])
        children = _canonical_members(arms["child"])
        if len(arms) != 2 or len(parents) != 2 or len(children) != 1:
            raise ValueError(
                f"{modality} must contain exactly two distinct parents of one child"
            )
        child = children[0]
        if child in parents:
            raise ValueError(
                f"{modality} must contain exactly two distinct parents of one child"
            )
        if (
            set(arms["parent"].astype(str)) != set(parents)
            or set(arms["child"].astype(str)) != {child}
        ):
            raise ValueError(f"Malformed direct collider motif for {modality}")

        assay_state = state.loc[state["modality"].astype(str).eq(str(modality))]
        if assay_state.empty:
            raise ValueError(f"No state-membership rows exist for {modality}")
        supported = set(
            assay_state.loc[
                assay_state["state_v427_supported_member"].astype(bool), "feature"
            ].astype(str)
        )
        supported_parents = tuple(parent for parent in parents if parent in supported)
        motif = {*parents, child}
        reporters = _canonical_members(supported.difference(motif))

        parent_gate = len(supported_parents) >= 1
        reporter_gate = len(reporters) >= 1
        selected = parent_gate and reporter_gate
        if not parent_gate:
            status = NO_PARENT_STATE_STATUS
        elif not reporter_gate:
            status = NO_EXTERNAL_REPORTER_STATUS
        else:
            status = SELECTED_STATUS
        pairs = (
            tuple(
                sorted(
                    canonical_pair(parent, reporter)
                    for parent in parents
                    for reporter in reporters
                )
            )
            if selected
            else ()
        )
        hypothesis_id = f"direct_parent_state_reporter_equivalence__{modality}"
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "modality": str(modality),
                "direct_parent_members": ";".join(parents),
                "direct_collider_child": child,
                "state_supported_direct_parent_members": ";".join(
                    supported_parents
                ),
                "external_state_reporter_members": ";".join(reporters),
                "state_supported_direct_parent_count": len(supported_parents),
                "external_state_reporter_count": len(reporters),
                "candidate_pairs_json": _pairs_json(pairs),
                "candidate_pair_count": len(pairs),
                "hypothesis_selected": selected,
                "hypothesis_type": (
                    HYPOTHESIS_TYPE if selected else UNRESOLVED_TYPE
                ),
                "hypothesis_status": status,
                "direct_collider_candidate_used": True,
                "parent_state_support_gate_passed": parent_gate,
                "external_state_reporter_gate_passed": reporter_gate,
                "shared_state_partial_identification_supported": selected,
                "exact_parent_reporter_pair_identified": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
                "new_direct_causal_relation_claimed": False,
                "exact_cross_assay_pair_claimed": False,
                "cross_assay_relation_created": False,
                "fixed_rule": FIXED_RULE,
                "structure_equivalence_schema": SCHEMA,
            }
        )

    return DirectParentStateReporterResult(
        decisions=pd.DataFrame(rows).reset_index(drop=True)
    )


__all__ = [
    "ALGORITHM",
    "DIRECT_COLUMNS",
    "DirectParentStateReporterResult",
    "FIXED_RULE",
    "HYPOTHESIS_TYPE",
    "NO_EXTERNAL_REPORTER_STATUS",
    "NO_PARENT_STATE_STATUS",
    "SCHEMA",
    "SELECTED_STATUS",
    "STATE_COLUMNS",
    "UNRESOLVED_TYPE",
    "build_direct_parent_state_reporter_equivalence",
    "canonical_pair",
    "pair_key",
    "parse_candidate_pairs",
]
