from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loading_group_state_guard_v4 as guard  # noqa: E402


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = pd.DataFrame(
        [
            {
                "loading_group": "group_keep",
                "modality": "chrom_A",
                "members": "a;b",
                "selection_status": "selected",
            },
            {
                "loading_group": "group_reject",
                "modality": "chrom_B",
                "members": "c;null",
                "selection_status": "selected",
            },
            {
                "loading_group": "singleton",
                "modality": "chrom_B",
                "members": "c",
                "selection_status": "singleton_candidate",
            },
        ]
    )
    membership = pd.DataFrame(
        [
            {"loading_group": "group_keep", "feature": "a", "modality": "chrom_A"},
            {"loading_group": "group_keep", "feature": "b", "modality": "chrom_A"},
            {"loading_group": "group_reject", "feature": "c", "modality": "chrom_B"},
            {
                "loading_group": "group_reject",
                "feature": "null",
                "modality": "chrom_B",
            },
            {"loading_group": "singleton", "feature": "c", "modality": "chrom_B"},
        ]
    )
    features = pd.DataFrame(
        [
            {"feature": "a", "modality": "chrom_A", "state_v427_supported_member": True},
            {"feature": "b", "modality": "chrom_A", "state_v427_supported_member": True},
            {"feature": "c", "modality": "chrom_B", "state_v427_supported_member": True},
            {
                "feature": "null",
                "modality": "chrom_B",
                "state_v427_supported_member": False,
            },
        ]
    )
    return groups, membership, features


def _axis_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    group_axes = pd.DataFrame(
        [
            {
                "loading_group": "group_reject",
                "group_modality": "chrom_B",
                "axis_id": "axis_1",
                "is_cluster_member": True,
                "leave_one_group_out_for_member": True,
                "modality_conflict_with_axis": False,
                "lower_absolute_group_axis_cosine": 0.91,
                "group_axis_view_support": 1.0,
                "relative_axis_margin": 0.20,
                "passes_member_lower_cosine": True,
                "passes_member_view_support": True,
                "passes_relative_axis_margin": True,
                "selection_status": "selected",
            }
        ]
    )
    axes = pd.DataFrame(
        [
            {
                "axis_id": "axis_1",
                "loading_groups": "group_keep;group_reject",
                "modalities": "chrom_A;chrom_B",
                "n_groups": 2,
                "n_modalities": 2,
                "singleton_only_axis": False,
                "axis_confidence": 0.89,
                "selection_status": "selected",
            }
        ]
    )
    return group_axes, axes


def test_guard_retains_only_groups_whose_members_are_v427_supported() -> None:
    result = guard.apply_loading_group_state_guard(*_inputs())

    assert result.retained_group_ids == {"group_keep"}
    assert result.rejected_group_ids == {"group_reject"}
    decisions = result.decisions.set_index("loading_group")
    assert decisions.loc["group_keep", "guard_status"] == guard.RETAINED_STATUS
    assert decisions.loc["group_reject", "guard_status"] == guard.REJECTED_STATUS
    assert decisions.loc["group_reject", "unsupported_members"] == "null"
    assert decisions["direct_structure_alone_is_insufficient"].all()
    assert not decisions["truth_used"].any()
    assert set(decisions["guard_schema"]) == {guard.SCHEMA}


def test_guard_is_one_way_and_ignores_unregistered_direct_annotations() -> None:
    groups, membership, features = _inputs()
    features["within_assay_direct_candidate_overlap"] = [False, False, True, True]

    result = guard.apply_loading_group_state_guard(groups, membership, features)

    assert result.retained_group_ids == {"group_keep"}
    assert result.rejected_group_ids == {"group_reject"}


def test_guard_retains_one_unresolved_member_with_leave_one_group_out_axis() -> None:
    groups, membership, features = _inputs()
    group_axes, axes = _axis_inputs()

    result = guard.apply_loading_group_state_guard(
        groups, membership, features, group_axes, axes
    )

    assert result.retained_group_ids == {"group_keep", "group_reject"}
    decisions = result.decisions.set_index("loading_group")
    rescued = decisions.loc["group_reject"]
    assert rescued["guard_route"] == guard.AXIS_ROUTE
    assert rescued["guard_status"] == guard.AXIS_RETAINED_STATUS
    assert rescued["cross_assay_axis_consensus"]
    assert rescued["cross_assay_axis_id"] == "axis_1"
    assert rescued["cross_assay_axis_n_modalities"] == 2
    assert rescued["unsupported_members"] == "null"
    assert not rescued["changes_state_v427_membership"]
    assert not rescued["exact_cross_assay_pair_claimed"]


def test_axis_route_fails_closed_without_leave_one_group_out_evidence() -> None:
    groups, membership, features = _inputs()
    group_axes, axes = _axis_inputs()
    group_axes["leave_one_group_out_for_member"] = False

    result = guard.apply_loading_group_state_guard(
        groups, membership, features, group_axes, axes
    )

    assert result.retained_group_ids == {"group_keep"}
    assert result.rejected_group_ids == {"group_reject"}


def test_axis_route_rejects_partial_provenance() -> None:
    groups, membership, features = _inputs()
    group_axes, _ = _axis_inputs()

    with pytest.raises(ValueError, match="must be supplied together"):
        guard.apply_loading_group_state_guard(
            groups, membership, features, group_axes, None
        )


def test_guard_rejects_cross_assay_or_inconsistent_membership() -> None:
    groups, membership, features = _inputs()
    membership.loc[membership["feature"].eq("b"), "modality"] = "chrom_B"

    with pytest.raises(ValueError, match="modality disagrees"):
        guard.apply_loading_group_state_guard(groups, membership, features)


def test_guard_rejects_duplicate_group_membership() -> None:
    groups, membership, features = _inputs()
    membership = pd.concat([membership, membership.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate rows"):
        guard.apply_loading_group_state_guard(groups, membership, features)
