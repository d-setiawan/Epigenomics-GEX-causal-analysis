from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import state_equivalence_membership_v4 as equivalence  # noqa: E402


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [
                "strict",
                "strong_redundant",
                "weak_redundant",
                "not_redundant",
                "failed_before_redundancy",
            ],
            "modality": ["chrom_A"] * 5,
            "audit_selected": [True, False, False, False, False],
            "pre_redundancy_selection_status": [
                "selected",
                "selected",
                "selected",
                "selected",
                "unresolved",
            ],
            "redundancy_demoted": [False, True, True, False, True],
            "feature_global_maxT_p": [0.001, 0.01, 0.011, 0.001, 0.001],
        }
    )


def test_equivalence_tier_adds_only_strong_redundancy_demotions() -> None:
    annotated = equivalence.annotate_state_equivalence_membership(
        _membership()
    ).set_index("feature")

    assert annotated.loc["strict", "state_membership_tier"] == (
        equivalence.STRICT_TIER
    )
    assert annotated.loc["strong_redundant", "state_membership_tier"] == (
        equivalence.EQUIVALENCE_TIER
    )
    assert annotated.loc["weak_redundant", "state_membership_tier"] == (
        equivalence.UNRESOLVED_TIER
    )
    assert annotated.loc["not_redundant", "state_membership_tier"] == (
        equivalence.UNRESOLVED_TIER
    )
    assert annotated.loc[
        "failed_before_redundancy", "state_membership_tier"
    ] == equivalence.UNRESOLVED_TIER
    assert annotated["strict_membership_changed"].eq(False).all()
    assert annotated["exact_cross_assay_pair_claimed"].eq(False).all()
    assert annotated["direct_causal_claimed"].eq(False).all()


def test_typed_groups_are_sets_and_forbid_pairwise_expansion() -> None:
    annotated = equivalence.annotate_state_equivalence_membership(_membership())
    groups = equivalence.build_state_equivalence_groups(annotated)
    group = groups.iloc[0]

    assert group["edge_type"] == equivalence.GROUP_EDGE_TYPE
    assert group["strict_members"] == "strict"
    assert group["equivalence_only_members"] == "strong_redundant"
    assert group["supported_members"] == "strict;strong_redundant"
    assert group["n_supported_members"] == 2
    assert not group["pairwise_expansion_permitted"]
    assert not group["exact_cross_assay_pair_claimed"]
    assert not group["direct_causal_claimed"]


def test_boolean_parser_fails_closed() -> None:
    membership = _membership()
    membership["redundancy_demoted"] = membership[
        "redundancy_demoted"
    ].astype(object)
    membership.loc[0, "redundancy_demoted"] = "maybe"

    with pytest.raises(ValueError, match="non-boolean"):
        equivalence.annotate_state_equivalence_membership(membership)


def test_layered_producer_can_supply_frozen_strict_membership() -> None:
    membership = _membership().drop(columns="audit_selected")
    strict = pd.Series(
        [False, True, False, False, False], index=membership.index
    )
    annotated = equivalence.annotate_state_equivalence_membership(
        membership, strict_membership=strict
    ).set_index("feature")

    assert annotated.loc["strong_redundant", "strict_state_member"]
    assert not annotated.loc[
        "strong_redundant", "state_equivalence_only_member"
    ]
    assert annotated.loc["strict", "state_membership_tier"] == (
        equivalence.UNRESOLVED_TIER
    )

    misaligned = strict.copy()
    misaligned.index = range(10, 15)
    with pytest.raises(ValueError, match="index must exactly match"):
        equivalence.annotate_state_equivalence_membership(
            membership, strict_membership=misaligned
        )
