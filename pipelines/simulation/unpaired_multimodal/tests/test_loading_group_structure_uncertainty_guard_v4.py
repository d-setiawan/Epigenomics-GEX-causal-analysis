from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loading_group_state_guard_v4 as v429  # noqa: E402
import loading_group_structure_uncertainty_guard_v4 as guard  # noqa: E402


def _decision(
    group_id: str,
    modality: str,
    members: str,
    route: str,
) -> dict[str, object]:
    retained = route in {v429.STATE_ROUTE, v429.AXIS_ROUTE}
    return {
        "loading_group": group_id,
        "modality": modality,
        "members": members,
        "retained": retained,
        "guard_route": route,
        "guard_status": (
            v429.RETAINED_STATUS
            if route == v429.STATE_ROUTE
            else (
                v429.AXIS_RETAINED_STATUS
                if route == v429.AXIS_ROUTE
                else v429.REJECTED_STATUS
            )
        ),
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = pd.DataFrame(
        [
            _decision("high_state", "chrom_A", "p;s", v429.STATE_ROUTE),
            _decision("low_state", "chrom_B", "u;v", v429.STATE_ROUTE),
            _decision("low_axis", "chrom_C", "x;y", v429.AXIS_ROUTE),
            _decision("child_overlap", "chrom_D", "c;s2", v429.STATE_ROUTE),
            _decision("two_parents", "chrom_D", "p1;p2", v429.STATE_ROUTE),
            _decision("one_parent", "chrom_D", "p1;s3", v429.STATE_ROUTE),
            _decision("upstream_rejected", "chrom_E", "n;s4", "unresolved"),
        ]
    )
    cosine = {
        "high_state": 0.95,
        "low_state": 0.89,
        "low_axis": 0.20,
        "child_overlap": 0.98,
        "two_parents": 0.99,
        "one_parent": 0.96,
        "upstream_rejected": 0.99,
    }
    groups = decisions.loc[:, ["loading_group", "modality", "members"]].copy()
    groups["minimum_member_centroid_cosine"] = groups["loading_group"].map(cosine)
    groups["selection_status"] = "selected"
    direct = pd.DataFrame(
        [
            {
                "node1": "p1",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "source_assay": "chrom_D",
            },
            {
                "node1": "p2",
                "endpoint1": "TAIL",
                "node2": "c",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "source_assay": "chrom_D",
            },
        ]
    )
    return decisions, groups, direct


def test_guard_separates_uncertainty_and_direct_overlap_from_state_support() -> None:
    result = guard.apply_loading_group_structure_uncertainty_guard(*_inputs())
    rows = result.decisions.set_index("loading_group")

    assert result.retained_group_ids == {"high_state", "low_axis", "one_parent"}
    assert rows.loc["high_state", "guard_status"] == guard.STATE_RETAINED_STATUS
    assert rows.loc["low_state", "guard_status"] == guard.LOW_CONFIDENCE_STATUS
    assert rows.loc["low_axis", "guard_status"] == guard.AXIS_RETAINED_STATUS
    assert rows.loc["child_overlap", "guard_status"] == guard.DIRECT_OVERLAP_STATUS
    assert rows.loc["two_parents", "guard_status"] == guard.DIRECT_OVERLAP_STATUS
    assert rows.loc["upstream_rejected", "guard_status"] == guard.UPSTREAM_REJECTED_STATUS
    assert rows.loc["child_overlap", "direct_collider_child_overlap"] == "c"
    assert rows.loc["two_parents", "shared_direct_parent_children"] == "c"
    assert rows.loc["one_parent", "shared_direct_parent_children"] == ""
    assert rows.loc["low_axis", "minimum_member_centroid_cosine"] == 0.20
    assert not rows.loc["low_axis", "passes_loading_uncertainty_floor"]
    assert rows.loc["low_axis", "v429_guard_route"] == v429.AXIS_ROUTE


def test_incomplete_direct_fork_does_not_activate_collider_overlap() -> None:
    decisions, groups, direct = _inputs()
    direct = direct.iloc[[0]].copy()

    rows = guard.apply_loading_group_structure_uncertainty_guard(
        decisions, groups, direct
    ).decisions.set_index("loading_group")

    assert rows.loc["child_overlap", "retained"]
    assert rows.loc["two_parents", "retained"]
    assert not rows["complete_direct_collider_overlap"].any()


def test_reversed_direct_endpoint_order_is_normalized() -> None:
    decisions, groups, direct = _inputs()
    direct.loc[0, ["node1", "endpoint1", "node2", "endpoint2"]] = [
        "c",
        "ARROW",
        "p1",
        "TAIL",
    ]

    rows = guard.apply_loading_group_structure_uncertainty_guard(
        decisions, groups, direct
    ).decisions.set_index("loading_group")

    assert rows.loc["child_overlap", "complete_direct_collider_overlap"]
    assert rows.loc["two_parents", "complete_direct_collider_overlap"]


def test_guard_rejects_invalid_cosine_or_direct_endpoint() -> None:
    decisions, groups, direct = _inputs()
    invalid_groups = groups.copy()
    invalid_groups.loc[
        invalid_groups["loading_group"].eq("high_state"),
        "minimum_member_centroid_cosine",
    ] = 1.01
    with pytest.raises(ValueError, match="must lie in"):
        guard.apply_loading_group_structure_uncertainty_guard(
            decisions, invalid_groups, direct
        )

    invalid_direct = direct.copy()
    invalid_direct.loc[0, "endpoint1"] = "CIRCLE"
    with pytest.raises(ValueError, match="one TAIL and one ARROW"):
        guard.apply_loading_group_structure_uncertainty_guard(
            decisions, groups, invalid_direct
        )
