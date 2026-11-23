from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_joint_context_stratified_stability_v4 as diagnostic  # noqa: E402


def test_maximum_context_stratum_is_exactly_three_of_three() -> None:
    observed = diagnostic._is_maximum_local_consensus(
        pd.Series([0, 1, 2, 3])
    )
    assert observed.tolist() == [False, False, False, True]
    assert diagnostic.REQUIRED_HELDOUT_LOCAL_CONTEXT_COUNT == 2


def test_context_stratified_selection_retains_all_strict_guards() -> None:
    selected, _ = diagnostic._selection_decision(
        nominal_new_proposal=True,
        maximum_local_consensus=True,
        candidate_count=99,
        null_count=78,
    )
    assert selected
    no_consensus, _ = diagnostic._selection_decision(
        nominal_new_proposal=True,
        maximum_local_consensus=False,
        candidate_count=100,
        null_count=0,
    )
    assert not no_consensus
    tied, _ = diagnostic._selection_decision(
        nominal_new_proposal=True,
        maximum_local_consensus=True,
        candidate_count=78,
        null_count=78,
    )
    assert not tied
    no_majority, _ = diagnostic._selection_decision(
        nominal_new_proposal=True,
        maximum_local_consensus=True,
        candidate_count=50,
        null_count=0,
    )
    assert not no_majority


def test_local_context_validation_rejects_fractional_or_out_of_range() -> None:
    for values in ([3.5], [4], [-1]):
        try:
            diagnostic._is_maximum_local_consensus(pd.Series(values))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid local-context count: {values}")
