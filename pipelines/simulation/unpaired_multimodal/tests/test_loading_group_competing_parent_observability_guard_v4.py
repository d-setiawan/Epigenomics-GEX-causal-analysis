from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loading_group_competing_parent_observability_guard_v4 as guard  # noqa: E402
import loading_group_state_guard_v4 as v429  # noqa: E402


def _decision(
    group_id: str, modality: str, members: str, route: str
) -> dict[str, object]:
    return {
        "loading_group": group_id,
        "modality": modality,
        "members": members,
        "retained": route in {v429.STATE_ROUTE, v429.AXIS_ROUTE},
        "guard_route": route,
        "guard_status": f"v430_{route}",
    }


def _feature(
    name: str,
    modality: str,
    *,
    marginal: float = 0.02,
    partial: float = 0.01,
    degree: int = 1,
) -> dict[str, object]:
    state_gate = marginal > 0.0 and partial > 0.0
    residual = degree >= 1
    return {
        "feature": name,
        "modality": modality,
        "formal_test_eligible": True,
        "oof_marginal_state_fit": marginal,
        "oof_partial_state_fit": partial,
        "within_assay_direct_candidate_degree": degree,
        "topology_consensus_state_specific_gate": state_gate,
        "topology_consensus_residual_structure_support": residual,
        "topology_consensus_fixed_rule_eligible": state_gate and residual,
    }


def _structure(
    parent: str,
    sibling: str,
    center: str,
    competitor: str,
    modality: str,
) -> dict[str, object]:
    return {
        "pair_key": "|||".join(sorted((parent, sibling))),
        "opposite_parent_pair_key": "|||".join(sorted((competitor, sibling))),
        "modality": modality,
        "parent_candidate": parent,
        "noncollider_sibling_candidate": sibling,
        "collider_center": center,
        "other_parent_candidate": competitor,
        "selected": True,
        "selection_stage": "formal_baseline",
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = pd.DataFrame(
        [
            _decision("eligible", "chrom_A", "p;s", v429.STATE_ROUTE),
            _decision("unobservable", "chrom_B", "p2;s2", v429.STATE_ROUTE),
            _decision("axis", "chrom_C", "x;y", v429.AXIS_ROUTE),
            _decision("no_comparison", "chrom_D", "p4;s4", v429.STATE_ROUTE),
            _decision("upstream_rejected", "chrom_E", "u;v", "unresolved"),
        ]
    )
    structure = pd.DataFrame(
        [
            _structure("p", "s", "c", "q", "chrom_A"),
            _structure("p2", "s2", "c2", "q2", "chrom_B"),
        ]
    )
    features = pd.DataFrame(
        [
            _feature("q", "chrom_A"),
            _feature("q2", "chrom_B", partial=-0.01),
            _feature("p4", "chrom_D"),
        ]
    )
    return decisions, structure, features


def test_guard_requires_an_observable_competing_parent_only_for_state_route() -> None:
    rows = guard.apply_loading_group_competing_parent_observability_guard(
        *_inputs()
    ).decisions.set_index("loading_group")

    assert rows.loc["eligible", "retained"]
    assert rows.loc["eligible", "guard_status"] == guard.STATE_RETAINED_STATUS
    assert rows.loc["eligible", "selected_parent_candidate"] == "p"
    assert rows.loc["eligible", "competing_parent_candidate"] == "q"
    assert rows.loc["eligible", "noncollider_sibling_candidate"] == "s"
    assert rows.loc["eligible", "complete_two_parent_comparison_available"]
    assert rows.loc["eligible", "competing_parent_fixed_rule_eligible"]

    assert not rows.loc["unobservable", "retained"]
    assert (
        rows.loc["unobservable", "guard_status"]
        == guard.COMPETITOR_UNOBSERVABLE_STATUS
    )
    assert rows.loc["unobservable", "competing_parent_candidate"] == "q2"
    assert not rows.loc["unobservable", "competing_parent_state_specific_gate"]

    assert rows.loc["axis", "retained"]
    assert rows.loc["axis", "guard_status"] == guard.AXIS_RETAINED_STATUS
    assert not rows.loc["axis", "competing_parent_observability_required"]
    assert rows.loc["axis", "passes_competing_parent_observability_guard"]

    assert not rows.loc["no_comparison", "retained"]
    assert (
        rows.loc["no_comparison", "guard_status"]
        == guard.COMPARISON_UNAVAILABLE_STATUS
    )
    assert not rows.loc["upstream_rejected", "retained"]
    assert (
        rows.loc["upstream_rejected", "guard_status"]
        == guard.UPSTREAM_REJECTED_STATUS
    )


def test_structure_competition_pair_keys_are_reconstructed() -> None:
    decisions, structure, features = _inputs()
    structure.loc[0, "opposite_parent_pair_key"] = "q|||wrong"

    with pytest.raises(ValueError, match="opposite-parent pair key"):
        guard.apply_loading_group_competing_parent_observability_guard(
            decisions, structure, features
        )


def test_guard_reconstructs_the_frozen_v427_feature_conjunction() -> None:
    decisions, structure, features = _inputs()
    features.loc[features["feature"].eq("q"), "topology_consensus_fixed_rule_eligible"] = False

    with pytest.raises(ValueError, match="cannot be reconstructed"):
        guard.apply_loading_group_competing_parent_observability_guard(
            decisions, structure, features
        )


def test_guard_fails_if_a_resolved_competitor_lacks_feature_evidence() -> None:
    decisions, structure, features = _inputs()
    features = features.loc[~features["feature"].eq("q")].copy()

    with pytest.raises(ValueError, match="lacks feature evidence"):
        guard.apply_loading_group_competing_parent_observability_guard(
            decisions, structure, features
        )
