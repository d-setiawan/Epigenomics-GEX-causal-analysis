from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loading_group_structure_equivalence_v4 as equivalence  # noqa: E402


def _row(
    group: str,
    modality: str,
    members: str,
    *,
    retained: bool,
    route: str,
) -> dict[str, object]:
    state = retained and route == equivalence.STATE_ROUTE
    return {
        "loading_group": group,
        "modality": modality,
        "members": members,
        "retained": retained,
        "guard_route": route,
        "guard_status": f"v431_{route}",
        "complete_two_parent_comparison_available": state,
        "selected_parent_candidate": "p" if state else "",
        "competing_parent_candidate": "q" if state else "",
        "noncollider_sibling_candidate": "s" if state else "",
        "collider_child_candidate": "c" if state else "",
        "passes_competing_parent_observability_guard": retained,
    }


def _inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                "state",
                "chrom_A",
                "p;s",
                retained=True,
                route=equivalence.STATE_ROUTE,
            ),
            _row(
                "axis",
                "chrom_B",
                "x;y",
                retained=True,
                route=equivalence.AXIS_ROUTE,
            ),
            _row(
                "rejected",
                "chrom_C",
                "u;v",
                retained=False,
                route="unresolved",
            ),
        ]
    )


def test_state_route_becomes_one_four_pair_equivalence_hyperedge() -> None:
    rows = equivalence.build_loading_group_structure_equivalence(
        _inputs()
    ).decisions.set_index("loading_group")
    state = rows.loc["state"]

    assert state["hypothesis_selected"]
    assert state["structure_equivalence_selected"]
    assert not state["exact_loading_group_selected"]
    assert not state["exact_pair_identified"]
    assert state["hypothesis_type"] == equivalence.STATE_EQUIVALENCE_TYPE
    assert state["parent_equivalence_members"] == "p;q"
    assert state["counterpart_equivalence_members"] == "c;s"
    assert equivalence.parse_candidate_pairs(state["candidate_pairs_json"]) == (
        ("c", "p"),
        ("c", "q"),
        ("p", "s"),
        ("q", "s"),
    )
    assert state["candidate_pair_count"] == 4
    assert not state["candidate_pairs_may_be_expanded_as_edges"]
    assert not state["direct_causal_relation_claimed"]


def test_axis_route_remains_one_exact_assay_local_group() -> None:
    result = equivalence.build_loading_group_structure_equivalence(_inputs())
    axis = result.decisions.set_index("loading_group").loc["axis"]

    assert axis["hypothesis_selected"]
    assert axis["exact_loading_group_selected"]
    assert not axis["structure_equivalence_selected"]
    assert axis["exact_pair_identified"]
    assert axis["hypothesis_type"] == equivalence.AXIS_EXACT_TYPE
    assert equivalence.parse_candidate_pairs(axis["candidate_pairs_json"]) == (
        ("x", "y"),
    )
    assert len(result.structure_equivalence_hyperedges) == 1
    assert len(result.exact_loading_groups) == 1


def test_upstream_rejection_remains_unresolved_without_candidate_pairs() -> None:
    rejected = equivalence.build_loading_group_structure_equivalence(
        _inputs()
    ).decisions.set_index("loading_group").loc["rejected"]

    assert not rejected["hypothesis_selected"]
    assert rejected["hypothesis_type"] == equivalence.UNRESOLVED_TYPE
    assert equivalence.parse_candidate_pairs(rejected["candidate_pairs_json"]) == ()
    assert rejected["candidate_pair_count"] == 0


def test_state_route_requires_the_exact_four_node_v431_comparison() -> None:
    frame = _inputs()
    frame.loc[frame["loading_group"].eq("state"), "collider_child_candidate"] = "s"

    with pytest.raises(ValueError, match="must contain four nodes"):
        equivalence.build_loading_group_structure_equivalence(frame)


def test_state_nominal_pair_must_match_selected_parent_and_sibling() -> None:
    frame = _inputs()
    frame.loc[frame["loading_group"].eq("state"), "members"] = "q;s"

    with pytest.raises(ValueError, match="disagrees with selected parent"):
        equivalence.build_loading_group_structure_equivalence(frame)


def test_core_transform_rejects_truth_or_scorer_columns() -> None:
    frame = _inputs().assign(truth_pair=False)

    with pytest.raises(ValueError, match="Truth/scorer columns are forbidden"):
        equivalence.build_loading_group_structure_equivalence(frame)


def test_candidate_pair_parser_rejects_noncanonical_serialization() -> None:
    with pytest.raises(ValueError, match="unique and canonically sorted"):
        equivalence.parse_candidate_pairs('[["q","p"],["p","q"]]')
