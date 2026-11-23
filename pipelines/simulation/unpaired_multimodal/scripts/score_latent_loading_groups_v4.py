#!/usr/bin/env python3
"""Score CA-BOSS v4 loading groups against scorer-only latent-axis metadata.

This scorer evaluates the attainable *shared-state* estimand.  It does not
score direct causal direction.  P and S inherit their mark's primary state
dimension, Q inherits its separately randomized Q dimension, and signs are
discarded.  C and sentinel features are observed nodes but are never members
of a pure state-loading group.  The simulated gene is a support target only
for the slow mark's state dimension.

Discovery remains isolated from these inputs: ``world_parameters.json`` and
``role_assignments.tsv`` are read only by this post-discovery scorer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PURE_ROLES = {"P", "Q", "S"}
ALLOWED_ROLES = {"P", "Q", "C", "S", "N1", "N2"}
WITHIN_GROUP_TYPE = "within_assay_shared_state_group"
WITHIN_EDGE_TYPE = "within_assay_shared_state"
CROSS_TYPE = "cross_assay_shared_state"
GENE_TYPE = "prior_constrained_peak_gene"
RELATIONAL_EDGE_TYPES = {WITHIN_EDGE_TYPE, CROSS_TYPE, GENE_TYPE}
CHROMATIN_AXIS_TYPE = "cross_assay_shared_state_axis"
RNA_AXIS_TYPE = "cross_assay_shared_state_axis_with_rna"
LATENT_AXIS_TYPES = {CHROMATIN_AXIS_TYPE, RNA_AXIS_TYPE}


@dataclass(frozen=True)
class LatentLoadingTruth:
    """Scorer-only, sign-invariant feature-to-axis contract."""

    features: pd.DataFrame
    expected_assay_groups: pd.DataFrame
    gene: str
    slow_state_dimension: int
    eligible_cross_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class LatentLoadingGroupScore:
    """Metrics and auditable row-level decisions for both input layers."""

    metrics: dict[str, Any]
    hyperedge_scores: pd.DataFrame
    sparse_edge_scores: pd.DataFrame
    truth: LatentLoadingTruth
    axis_hyperedge_scores: pd.DataFrame


def _as_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _members(value: Any) -> tuple[str, ...]:
    text = _as_text(value)
    return tuple(dict.fromkeys(part.strip() for part in text.split(";") if part.strip()))


def _selected(record: Mapping[str, Any]) -> bool:
    status = _as_text(record.get("selection_status", "selected")).casefold()
    return status in {"selected", "select", "true", "1"}


def _require_mapping(mapping: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")


def build_latent_loading_truth(
    world_parameters: Mapping[str, Any],
    role_assignments: pd.DataFrame,
) -> LatentLoadingTruth:
    """Build the exact sign-invariant loading groups from scorer metadata.

    The function deliberately cross-checks the redundant dimension stored in
    ``role_assignments.tsv`` against ``world_parameters.json``.  This makes a
    stale or accidentally mixed scorer bundle fail closed.
    """

    _require_mapping(
        world_parameters,
        ("marks", "state_dimension", "q_dimension", "slow_mark", "gene"),
        "world parameters",
    )
    roles_required = {"node", "mark", "role", "state_dimension"}
    missing = sorted(roles_required - set(role_assignments.columns))
    if missing:
        raise ValueError(f"Role assignments are missing columns: {missing}")
    roles = role_assignments.copy()
    if roles["node"].astype(str).duplicated().any():
        raise ValueError("Role-assignment nodes must be unique")

    marks = tuple(str(mark) for mark in world_parameters["marks"])
    state_dimension = {
        str(mark): int(dimension)
        for mark, dimension in dict(world_parameters["state_dimension"]).items()
    }
    q_dimension = {
        str(mark): int(dimension)
        for mark, dimension in dict(world_parameters["q_dimension"]).items()
    }
    if set(marks) != set(state_dimension) or set(marks) != set(q_dimension):
        raise ValueError("Mark and state-dimension maps do not have the same keys")
    if set(roles["mark"].astype(str)) != set(marks):
        raise ValueError("Role-assignment marks do not match world parameters")
    unknown_roles = sorted(set(roles["role"].astype(str)) - ALLOWED_ROLES)
    if unknown_roles:
        raise ValueError(f"Unknown feature roles: {unknown_roles}")
    for mark, frame in roles.groupby(roles["mark"].astype(str)):
        counts = frame["role"].astype(str).value_counts().to_dict()
        if any(counts.get(role, 0) != 1 for role in ALLOWED_ROLES):
            raise ValueError(f"Mark {mark} must contain exactly one feature per role")

    feature_rows: list[dict[str, Any]] = []
    for record in roles.to_dict(orient="records"):
        node = str(record["node"])
        mark = str(record["mark"])
        role = str(record["role"])
        pure = role in PURE_ROLES
        expected_dimension: int | None
        if role in {"P", "S"}:
            expected_dimension = state_dimension[mark]
        elif role == "Q":
            expected_dimension = q_dimension[mark]
        else:
            expected_dimension = None
        recorded = pd.to_numeric(pd.Series([record["state_dimension"]]), errors="coerce").iloc[0]
        if expected_dimension is None:
            if pd.notna(recorded):
                raise ValueError(f"Non-pure feature {node} unexpectedly has a state dimension")
        elif pd.isna(recorded) or int(recorded) != expected_dimension:
            raise ValueError(f"State dimension mismatch for {node}")
        feature_rows.append(
            {
                "feature": node,
                "modality": f"chrom_{mark}",
                "mark": mark,
                "role": role,
                "pure_state_feature": bool(pure),
                "state_dimension": expected_dimension,
                "support_target": False,
            }
        )

    slow_mark = str(world_parameters["slow_mark"])
    if slow_mark not in state_dimension:
        raise ValueError("slow_mark is not present in state_dimension")
    gene = str(world_parameters["gene"])
    if not gene or gene in set(roles["node"].astype(str)):
        raise ValueError("The scorer gene must be a distinct non-empty node")
    slow_dimension = state_dimension[slow_mark]
    feature_rows.append(
        {
            "feature": gene,
            "modality": "rna",
            "mark": "",
            "role": "GENE_SUPPORT_TARGET",
            "pure_state_feature": False,
            "state_dimension": slow_dimension,
            "support_target": True,
        }
    )
    features = pd.DataFrame(feature_rows).sort_values("feature").reset_index(drop=True)

    group_rows: list[dict[str, Any]] = []
    for mark in marks:
        frame = features.loc[
            (features["mark"] == mark) & features["role"].isin(["P", "S"])
        ]
        if set(frame["role"]) != {"P", "S"} or len(frame) != 2:
            raise ValueError(f"Mark {mark} must have exactly one P and one S feature")
        members = tuple(sorted(frame["feature"].astype(str)))
        group_rows.append(
            {
                "target_group": f"primary_state_group__{mark}",
                "modality": f"chrom_{mark}",
                "mark": mark,
                "state_dimension": state_dimension[mark],
                "members": ";".join(members),
            }
        )
    expected_groups = pd.DataFrame(group_rows)

    pure = features.loc[features["pure_state_feature"]]
    eligible_dimensions: list[int] = []
    for dimension, frame in pure.groupby("state_dimension"):
        if frame["modality"].nunique() >= 2:
            eligible_dimensions.append(int(dimension))
    return LatentLoadingTruth(
        features=features,
        expected_assay_groups=expected_groups,
        gene=gene,
        slow_state_dimension=slow_dimension,
        eligible_cross_dimensions=tuple(sorted(eligible_dimensions)),
    )


def _feature_lookup(truth: LatentLoadingTruth) -> dict[str, dict[str, Any]]:
    return {
        str(record["feature"]): record
        for record in truth.features.to_dict(orient="records")
    }


def _describe_members(
    members: Sequence[str], truth: LatentLoadingTruth
) -> dict[str, Any]:
    lookup = _feature_lookup(truth)
    unknown = sorted(set(members) - set(lookup))
    if unknown:
        raise ValueError(f"Discovery output contains unknown scorer nodes: {unknown}")
    records = [lookup[member] for member in members]
    peak_records = [record for record in records if record["modality"] != "rna"]
    pure_records = [record for record in peak_records if bool(record["pure_state_feature"])]
    dimensions = sorted(
        {
            int(record["state_dimension"])
            for record in records
            if pd.notna(record["state_dimension"])
        }
    )
    modalities = sorted({str(record["modality"]) for record in records})
    return {
        "n_members": len(records),
        "n_pure_peaks": len(pure_records),
        "n_nonpure_peaks": len(peak_records) - len(pure_records),
        "contains_gene": any(record["feature"] == truth.gene for record in records),
        "dimensions": dimensions,
        "modalities": modalities,
        "all_peaks_pure": len(peak_records) == len(pure_records),
    }


def _expected_relation(
    side1: Sequence[str], side2: Sequence[str], truth: LatentLoadingTruth
) -> tuple[str, int | None, str]:
    """Return expected type, common axis, and a reason for a two-sided claim."""

    left = _describe_members(side1, truth)
    right = _describe_members(side2, truth)
    combined = _describe_members(tuple(side1) + tuple(side2), truth)
    if not side1 or not side2:
        return "", None, "empty_side"
    if combined["n_nonpure_peaks"]:
        return "", None, "nonpure_peak_member"
    if len(left["dimensions"]) != 1 or len(right["dimensions"]) != 1:
        return "", None, "mixed_axis_within_side"
    if left["dimensions"][0] != right["dimensions"][0]:
        return "", None, "mismatched_state_dimensions"
    dimension = int(left["dimensions"][0])
    left_gene = bool(left["contains_gene"])
    right_gene = bool(right["contains_gene"])
    if left_gene or right_gene:
        gene_side = side1 if left_gene else side2
        peak_side = side2 if left_gene else side1
        peak = _describe_members(peak_side, truth)
        if left_gene and right_gene:
            return "", None, "gene_on_both_sides"
        if tuple(gene_side) != (truth.gene,):
            return "", None, "gene_support_side_not_singleton"
        if not peak_side or peak["contains_gene"] or len(peak["modalities"]) != 1:
            return "", None, "invalid_peak_support_group"
        if dimension != truth.slow_state_dimension:
            return "", None, "gene_support_on_non_slow_axis"
        return GENE_TYPE, dimension, "slow_axis_peak_gene_support"
    if len(left["modalities"]) != 1 or len(right["modalities"]) != 1:
        return "", None, "mixed_modality_within_side"
    if left["modalities"][0] == right["modalities"][0]:
        return WITHIN_EDGE_TYPE, dimension, "within_assay_same_axis"
    return CROSS_TYPE, dimension, "cross_assay_same_axis"


def _empty_hyperedge_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "hyperedge_id", "edge_type", "selection_status", "selected",
            "claim_kind", "members1", "members2", "expected_edge_type",
            "state_dimension", "axis_compatible", "type_correct", "typed_valid",
            "failure_reason",
        ]
    )


def score_typed_hyperedges(
    typed_hyperedges: pd.DataFrame | None, truth: LatentLoadingTruth
) -> pd.DataFrame:
    """Score group definitions and cross-group claims from discovery."""

    if typed_hyperedges is None:
        return _empty_hyperedge_scores()
    required = {"hyperedge_id", "edge_type", "selection_status", "group1", "group2", "members"}
    missing = sorted(required - set(typed_hyperedges.columns))
    if missing:
        raise ValueError(f"Typed hyperedges are missing columns: {missing}")
    if typed_hyperedges["hyperedge_id"].astype(str).duplicated().any():
        raise ValueError("Hyperedge identifiers must be unique")

    group_members: dict[str, tuple[str, ...]] = {}
    for record in typed_hyperedges.to_dict(orient="records"):
        group1 = _as_text(record.get("group1"))
        group2 = _as_text(record.get("group2"))
        if group1 and not group2:
            parsed = _members(record.get("members"))
            if group1 in group_members and group_members[group1] != parsed:
                raise ValueError(f"Inconsistent members for loading group {group1}")
            group_members[group1] = parsed

    rows: list[dict[str, Any]] = []
    for record in typed_hyperedges.to_dict(orient="records"):
        group1 = _as_text(record.get("group1"))
        group2 = _as_text(record.get("group2"))
        selected = _selected(record)
        edge_type = _as_text(record["edge_type"])
        if not group2:
            side1 = _members(record.get("members"))
            description = _describe_members(side1, truth)
            relational = len(side1) >= 2
            if not relational:
                expected = ""
                dimension = description["dimensions"][0] if len(description["dimensions"]) == 1 else None
                compatible = bool(
                    len(description["dimensions"]) == 1
                    and description["n_nonpure_peaks"] == 0
                )
                type_correct = True
                reason = "singleton_group_definition"
                claim_kind = "group_definition"
            else:
                expected = WITHIN_GROUP_TYPE
                dimension = description["dimensions"][0] if len(description["dimensions"]) == 1 else None
                compatible = bool(
                    description["n_nonpure_peaks"] == 0
                    and not description["contains_gene"]
                    and len(description["dimensions"]) == 1
                    and len(description["modalities"]) == 1
                )
                type_correct = edge_type == expected
                if description["n_nonpure_peaks"]:
                    reason = "nonpure_peak_member"
                elif len(description["dimensions"]) != 1:
                    reason = "mixed_state_dimensions"
                elif len(description["modalities"]) != 1:
                    reason = "mixed_modalities"
                elif description["contains_gene"]:
                    reason = "gene_in_within_assay_relation"
                else:
                    reason = "within_assay_same_axis"
                claim_kind = "within_assay_group"
            side2: tuple[str, ...] = ()
        else:
            side1 = _members(record.get("members1")) if "members1" in record else ()
            side2 = _members(record.get("members2")) if "members2" in record else ()
            if not side1:
                if group1 not in group_members:
                    raise ValueError(f"No membership definition for loading group {group1}")
                side1 = group_members[group1]
            if not side2:
                if group2 not in group_members:
                    raise ValueError(f"No membership definition for loading group {group2}")
                side2 = group_members[group2]
            expected, dimension, reason = _expected_relation(side1, side2, truth)
            compatible = bool(expected)
            type_correct = edge_type == expected if expected else False
            relational = True
            claim_kind = "cross_group_relation"
        typed_valid = bool(selected and relational and compatible and type_correct)
        rows.append(
            {
                "hyperedge_id": str(record["hyperedge_id"]),
                "edge_type": edge_type,
                "selection_status": _as_text(record["selection_status"]),
                "selected": int(selected),
                "claim_kind": claim_kind,
                "members1": ";".join(side1),
                "members2": ";".join(side2),
                "expected_edge_type": expected,
                "state_dimension": dimension if dimension is not None else "",
                "axis_compatible": int(compatible),
                "type_correct": int(type_correct),
                "typed_valid": int(typed_valid),
                "failure_reason": "" if typed_valid else reason,
            }
        )
    return pd.DataFrame(rows)


def _empty_axis_hyperedge_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "hyperedge_id",
            "axis_id",
            "edge_type",
            "selection_status",
            "selected",
            "members",
            "n_members",
            "n_modalities",
            "contains_gene",
            "pure_member_set",
            "expected_edge_type",
            "state_dimension",
            "axis_compatible",
            "type_correct",
            "typed_valid",
            "failure_reason",
        ]
    )


def score_typed_axis_hyperedges(
    typed_axis_hyperedges: pd.DataFrame | None,
    truth: LatentLoadingTruth,
) -> pd.DataFrame:
    """Score n-ary assay-spanning latent-axis claims exactly once.

    These rows are intentionally distinct from pairwise group relations.  An
    axis is compatible when all peak members are pure state-response features,
    every member maps to one sign-invariant state dimension, and at least two
    observed modalities participate.  An RNA-containing axis is valid only on
    the simulated slow dimension and must use the scorer gene as its sole RNA
    member.  The discovery layer never sees these scorer-only annotations.
    """

    if typed_axis_hyperedges is None:
        return _empty_axis_hyperedge_scores()
    required = {
        "hyperedge_id",
        "axis_id",
        "edge_type",
        "selection_status",
        "features",
    }
    missing = sorted(required - set(typed_axis_hyperedges.columns))
    if missing:
        raise ValueError(f"Typed axis hyperedges are missing columns: {missing}")
    if typed_axis_hyperedges["hyperedge_id"].astype(str).duplicated().any():
        raise ValueError("Typed axis hyperedge identifiers must be unique")
    if typed_axis_hyperedges["axis_id"].astype(str).duplicated().any():
        raise ValueError("Typed latent-axis identifiers must be unique")

    lookup = _feature_lookup(truth)
    rows: list[dict[str, Any]] = []
    for record in typed_axis_hyperedges.to_dict(orient="records"):
        members = _members(record["features"])
        unknown = sorted(set(members) - set(lookup))
        if unknown:
            raise ValueError(
                f"Discovery output contains unknown scorer nodes: {unknown}"
            )
        description = _describe_members(members, truth) if members else {
            "n_members": 0,
            "n_nonpure_peaks": 0,
            "contains_gene": False,
            "dimensions": [],
            "modalities": [],
        }
        selected = _selected(record)
        edge_type = _as_text(record["edge_type"])
        dimensions = list(description["dimensions"])
        modalities = list(description["modalities"])
        contains_gene = bool(description["contains_gene"])
        pure_members = bool(
            members
            and description["n_nonpure_peaks"] == 0
            and all(
                bool(lookup[member]["pure_state_feature"])
                or member == truth.gene
                for member in members
            )
        )
        one_dimension = len(dimensions) == 1
        dimension = int(dimensions[0]) if one_dimension else None
        cross_assay = len(modalities) >= 2
        rna_members = [
            member for member in members if str(lookup[member]["modality"]) == "rna"
        ]
        rna_contract = bool(
            not contains_gene
            or (
                rna_members == [truth.gene]
                and dimension == truth.slow_state_dimension
                and description["n_pure_peaks"] >= 1
            )
        )
        compatible = bool(
            pure_members and one_dimension and cross_assay and rna_contract
        )
        expected = (
            RNA_AXIS_TYPE
            if compatible and contains_gene
            else CHROMATIN_AXIS_TYPE
            if compatible
            else ""
        )
        type_correct = bool(expected and edge_type == expected)
        typed_valid = bool(selected and compatible and type_correct)
        if not members:
            reason = "empty_axis"
        elif not pure_members:
            reason = "nonpure_peak_member"
        elif not one_dimension:
            reason = "mixed_state_dimensions"
        elif not cross_assay:
            reason = "single_modality_axis"
        elif not rna_contract:
            reason = "rna_axis_not_slow_gene_support"
        elif not type_correct:
            reason = "wrong_axis_layer"
        elif not selected:
            reason = "not_selected"
        else:
            reason = ""
        rows.append(
            {
                "hyperedge_id": str(record["hyperedge_id"]),
                "axis_id": str(record["axis_id"]),
                "edge_type": edge_type,
                "selection_status": _as_text(record["selection_status"]),
                "selected": int(selected),
                "members": ";".join(members),
                "n_members": len(members),
                "n_modalities": len(modalities),
                "contains_gene": int(contains_gene),
                "pure_member_set": int(pure_members),
                "expected_edge_type": expected,
                "state_dimension": dimension if dimension is not None else "",
                "axis_compatible": int(compatible),
                "type_correct": int(type_correct),
                "typed_valid": int(typed_valid),
                "failure_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _empty_sparse_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "node1", "node2", "edge_type", "selection_status", "selected",
            "in_loading_scope", "expected_edge_type", "state_dimension",
            "axis_compatible", "type_correct", "typed_valid", "failure_reason",
        ]
    )


def score_sparse_edges(
    sparse_edges: pd.DataFrame | None, truth: LatentLoadingTruth
) -> pd.DataFrame:
    """Score pair projections or a full typed graph at the shared-state layer."""

    if sparse_edges is None:
        return _empty_sparse_scores()
    required = {"node1", "node2", "edge_type"}
    missing = sorted(required - set(sparse_edges.columns))
    if missing:
        raise ValueError(f"Sparse edges are missing columns: {missing}")
    pairs = sparse_edges.apply(
        lambda row: tuple(sorted((str(row["node1"]), str(row["node2"])))), axis=1
    )
    if pairs.duplicated().any():
        raise ValueError("Sparse edges contain duplicate adjacencies")
    lookup = _feature_lookup(truth)
    unknown = sorted(
        (set(sparse_edges["node1"].astype(str)) | set(sparse_edges["node2"].astype(str)))
        - set(lookup)
    )
    if unknown:
        raise ValueError(f"Discovery output contains unknown scorer nodes: {unknown}")

    rows: list[dict[str, Any]] = []
    for record in sparse_edges.to_dict(orient="records"):
        node1, node2 = str(record["node1"]), str(record["node2"])
        edge_type = _as_text(record["edge_type"])
        selected = _selected(record)
        in_scope = edge_type in RELATIONAL_EDGE_TYPES
        expected, dimension, reason = _expected_relation((node1,), (node2,), truth)
        compatible = bool(expected)
        type_correct = edge_type == expected if expected else False
        typed_valid = bool(selected and in_scope and compatible and type_correct)
        rows.append(
            {
                "node1": node1,
                "node2": node2,
                "edge_type": edge_type,
                "selection_status": _as_text(record.get("selection_status", "selected")),
                "selected": int(selected),
                "in_loading_scope": int(in_scope),
                "expected_edge_type": expected,
                "state_dimension": dimension if dimension is not None else "",
                "axis_compatible": int(compatible),
                "type_correct": int(type_correct),
                "typed_valid": int(typed_valid),
                "failure_reason": "" if typed_valid else reason,
            }
        )
    return pd.DataFrame(rows)


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _recovered_local_targets(
    truth: LatentLoadingTruth,
    hyperedge_scores: pd.DataFrame,
    edge_scores: pd.DataFrame,
) -> set[str]:
    recovered: set[str] = set()
    valid_groups = hyperedge_scores.loc[
        (hyperedge_scores["claim_kind"] == "within_assay_group")
        & (hyperedge_scores["typed_valid"] == 1)
    ]
    valid_edges = edge_scores.loc[
        (edge_scores["expected_edge_type"] == WITHIN_EDGE_TYPE)
        & (edge_scores["typed_valid"] == 1)
    ]
    group_member_sets = [set(_members(value)) for value in valid_groups["members1"]]
    edge_member_sets = [
        {str(row.node1), str(row.node2)} for row in valid_edges.itertuples(index=False)
    ]
    for record in truth.expected_assay_groups.to_dict(orient="records"):
        target = set(_members(record["members"]))
        if any(target <= members for members in group_member_sets + edge_member_sets):
            recovered.add(str(record["target_group"]))
    return recovered


def _valid_cross_dimensions(frame: pd.DataFrame) -> set[int]:
    if frame.empty:
        return set()
    selected = frame.loc[
        (frame["expected_edge_type"] == CROSS_TYPE) & (frame["typed_valid"] == 1),
        "state_dimension",
    ]
    return {int(value) for value in selected if _as_text(value)}


def _slow_support(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    return bool(
        (
            (frame["expected_edge_type"] == GENE_TYPE)
            & (frame["typed_valid"] == 1)
        ).any()
    )


def _valid_native_axis_dimensions(frame: pd.DataFrame) -> set[int]:
    if frame.empty:
        return set()
    selected = frame.loc[frame["typed_valid"] == 1, "state_dimension"]
    return {int(value) for value in selected if _as_text(value)}


def _native_slow_rna_support(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    return bool(
        (
            (frame["expected_edge_type"] == RNA_AXIS_TYPE)
            & (frame["typed_valid"] == 1)
        ).any()
    )


def score_latent_loading_groups(
    *,
    world_parameters: Mapping[str, Any],
    role_assignments: pd.DataFrame,
    typed_hyperedges: pd.DataFrame | None = None,
    typed_axis_hyperedges: pd.DataFrame | None = None,
    sparse_edges: pd.DataFrame | None = None,
) -> LatentLoadingGroupScore:
    """Score one or both discovery representations against latent axes."""

    if (
        typed_hyperedges is None
        and typed_axis_hyperedges is None
        and sparse_edges is None
    ):
        raise ValueError("At least one discovery representation must be provided")
    truth = build_latent_loading_truth(world_parameters, role_assignments)
    hyper_scores = score_typed_hyperedges(typed_hyperedges, truth)
    axis_scores = score_typed_axis_hyperedges(typed_axis_hyperedges, truth)
    edge_scores = score_sparse_edges(sparse_edges, truth)

    hyper_rel = hyper_scores.loc[
        hyper_scores["claim_kind"].isin(["within_assay_group", "cross_group_relation"])
        & (hyper_scores["selected"] == 1)
    ]
    group_definitions = hyper_scores.loc[
        hyper_scores["claim_kind"].isin(["within_assay_group", "group_definition"])
        & (hyper_scores["selected"] == 1)
    ]
    edge_rel = edge_scores.loc[
        (edge_scores["in_loading_scope"] == 1) & (edge_scores["selected"] == 1)
    ]
    recovered = _recovered_local_targets(truth, hyper_scores, edge_scores)
    native_axis_dimensions = _valid_native_axis_dimensions(axis_scores)
    cross_dimensions = (
        _valid_cross_dimensions(hyper_scores)
        | _valid_cross_dimensions(edge_scores)
        | native_axis_dimensions
    )
    native_slow_axis_supported = _native_slow_rna_support(axis_scores)
    slow_axis_supported = (
        _slow_support(hyper_scores)
        or _slow_support(edge_scores)
        or native_slow_axis_supported
    )
    # A correctly typed peak--gene group is itself a cross-assay observation of
    # the slow latent axis.  Count that axis as covered while retaining the
    # separate slow-support metric; this avoids requiring an arbitrary second
    # chromatin pair for the same state coordinate.
    if slow_axis_supported:
        cross_dimensions.add(int(truth.slow_state_dimension))
    eligible_dimensions = set(truth.eligible_cross_dimensions)
    selected_axes = axis_scores.loc[axis_scores["selected"] == 1]

    metrics: dict[str, Any] = {
        "scoring_contract": "sign_invariant_latent_loading_groups_v2",
        "direct_causal_direction_credited": False,
        "cell_correspondence_used": False,
        "n_truth_pure_peak_features": int(truth.features["pure_state_feature"].sum()),
        "n_truth_primary_assay_groups": len(truth.expected_assay_groups),
        "n_truth_cross_eligible_dimensions": len(eligible_dimensions),
        "slow_state_dimension": truth.slow_state_dimension,
        "n_selected_group_definitions": len(group_definitions),
        "n_axis_compatible_selected_group_definitions": int(
            group_definitions["axis_compatible"].sum()
        ),
        "selected_group_definition_axis_precision": _safe_rate(
            int(group_definitions["axis_compatible"].sum()), len(group_definitions)
        ),
        "n_selected_relational_hyperedges": len(hyper_rel),
        "n_axis_compatible_selected_hyperedges": int(hyper_rel["axis_compatible"].sum()),
        "n_typed_valid_selected_hyperedges": int(hyper_rel["typed_valid"].sum()),
        "selected_hyperedge_axis_precision": _safe_rate(
            int(hyper_rel["axis_compatible"].sum()), len(hyper_rel)
        ),
        "selected_hyperedge_typed_precision": _safe_rate(
            int(hyper_rel["typed_valid"].sum()), len(hyper_rel)
        ),
        "n_selected_latent_axis_hyperedges": len(selected_axes),
        "n_pure_selected_latent_axis_hyperedges": int(
            selected_axes["pure_member_set"].sum()
        ),
        "n_axis_compatible_selected_latent_axis_hyperedges": int(
            selected_axes["axis_compatible"].sum()
        ),
        "n_typed_valid_selected_latent_axis_hyperedges": int(
            selected_axes["typed_valid"].sum()
        ),
        "selected_latent_axis_purity_precision": _safe_rate(
            int(selected_axes["pure_member_set"].sum()), len(selected_axes)
        ),
        "selected_latent_axis_axis_precision": _safe_rate(
            int(selected_axes["axis_compatible"].sum()), len(selected_axes)
        ),
        "selected_latent_axis_typed_precision": _safe_rate(
            int(selected_axes["typed_valid"].sum()), len(selected_axes)
        ),
        "native_recovered_latent_axis_dimensions": sorted(native_axis_dimensions),
        "native_latent_axis_dimension_coverage": _safe_rate(
            len(native_axis_dimensions & eligible_dimensions),
            len(eligible_dimensions),
        ),
        "native_slow_rna_axis_support": int(native_slow_axis_supported),
        "n_selected_latent_axis_mismatch_claims": int(
            (selected_axes["axis_compatible"] == 0).sum()
        ),
        "n_selected_latent_axis_wrong_layer_claims": int(
            (
                (selected_axes["axis_compatible"] == 1)
                & (selected_axes["type_correct"] == 0)
            ).sum()
        ),
        "n_selected_loading_edges": len(edge_rel),
        "n_axis_compatible_selected_loading_edges": int(edge_rel["axis_compatible"].sum()),
        "n_typed_valid_selected_loading_edges": int(edge_rel["typed_valid"].sum()),
        "selected_loading_edge_axis_precision": _safe_rate(
            int(edge_rel["axis_compatible"].sum()), len(edge_rel)
        ),
        "selected_loading_edge_typed_precision": _safe_rate(
            int(edge_rel["typed_valid"].sum()), len(edge_rel)
        ),
        "n_recovered_primary_assay_groups": len(recovered),
        "combined_within_assay_ps_group_recall": _safe_rate(
            len(recovered), len(truth.expected_assay_groups)
        ),
        "recovered_cross_state_dimensions": sorted(cross_dimensions),
        "combined_cross_assay_latent_axis_coverage": _safe_rate(
            len(cross_dimensions & eligible_dimensions), len(eligible_dimensions)
        ),
        "combined_slow_gene_axis_support": int(
            slow_axis_supported
        ),
        "slow_gene_support_counts_as_cross_assay_axis_coverage": True,
        "n_selected_axis_mismatch_claims": int(
            ((hyper_rel["axis_compatible"] == 0)).sum()
            + ((edge_rel["axis_compatible"] == 0)).sum()
            + ((selected_axes["axis_compatible"] == 0)).sum()
        ),
        "n_selected_wrong_layer_claims": int(
            (
                (hyper_rel["axis_compatible"] == 1)
                & (hyper_rel["type_correct"] == 0)
            ).sum()
            + (
                (edge_rel["axis_compatible"] == 1)
                & (edge_rel["type_correct"] == 0)
            ).sum()
            + (
                (selected_axes["axis_compatible"] == 1)
                & (selected_axes["type_correct"] == 0)
            ).sum()
        ),
        "n_out_of_scope_sparse_edges": int((edge_scores["in_loading_scope"] == 0).sum()),
    }
    return LatentLoadingGroupScore(
        metrics=metrics,
        hyperedge_scores=hyper_scores,
        sparse_edge_scores=edge_scores,
        truth=truth,
        axis_hyperedge_scores=axis_scores,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-parameters", type=Path, required=True)
    parser.add_argument("--role-assignments", type=Path, required=True)
    parser.add_argument("--typed-hyperedges", type=Path)
    parser.add_argument("--typed-axis-hyperedges", type=Path)
    parser.add_argument("--sparse-edges", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.typed_hyperedges is None
        and args.typed_axis_hyperedges is None
        and args.sparse_edges is None
    ):
        raise SystemExit(
            "Provide --typed-hyperedges, --typed-axis-hyperedges, and/or --sparse-edges"
        )
    parameters = json.loads(args.world_parameters.read_text())
    roles = pd.read_csv(args.role_assignments, sep="\t", keep_default_na=False)
    hyperedges = (
        pd.read_csv(args.typed_hyperedges, sep="\t", keep_default_na=False)
        if args.typed_hyperedges is not None
        else None
    )
    axis_hyperedges = (
        pd.read_csv(args.typed_axis_hyperedges, sep="\t", keep_default_na=False)
        if args.typed_axis_hyperedges is not None
        else None
    )
    edges = (
        pd.read_csv(args.sparse_edges, sep="\t", keep_default_na=False)
        if args.sparse_edges is not None
        else None
    )
    result = score_latent_loading_groups(
        world_parameters=parameters,
        role_assignments=roles,
        typed_hyperedges=hyperedges,
        typed_axis_hyperedges=axis_hyperedges,
        sparse_edges=edges,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.truth.features.to_csv(args.out_dir / "latent_loading_truth_features.tsv", sep="\t", index=False)
    result.truth.expected_assay_groups.to_csv(
        args.out_dir / "latent_loading_truth_groups.tsv", sep="\t", index=False
    )
    result.hyperedge_scores.to_csv(args.out_dir / "latent_group_hyperedge_scores.tsv", sep="\t", index=False)
    result.sparse_edge_scores.to_csv(args.out_dir / "latent_group_edge_scores.tsv", sep="\t", index=False)
    result.axis_hyperedge_scores.to_csv(
        args.out_dir / "latent_axis_hyperedge_scores.tsv", sep="\t", index=False
    )
    (args.out_dir / "latent_group_metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n"
    )
    print(json.dumps(result.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
