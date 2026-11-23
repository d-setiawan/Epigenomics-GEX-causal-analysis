from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_dual_null_disagreement_stability_v4 as diagnostic


def test_dual_null_accepts_only_two_context_candidate_above_both_references() -> None:
    selected, probability = diagnostic._selection_decision(
        nominal_new_proposal=True,
        local_context_count=2,
        candidate_count=100,
        stratum_null_count=88,
        global_null_count=96,
    )
    assert selected
    assert probability >= 0.975


def test_global_safety_ceiling_rejects_otherwise_strong_two_context_pair() -> None:
    selected, probability = diagnostic._selection_decision(
        nominal_new_proposal=True,
        local_context_count=2,
        candidate_count=86,
        stratum_null_count=0,
        global_null_count=97,
    )
    assert not selected
    assert probability >= 0.975


def test_route_does_not_reselect_maximum_context_or_nonproposal() -> None:
    maximum_context, _ = diagnostic._selection_decision(
        nominal_new_proposal=True,
        local_context_count=3,
        candidate_count=100,
        stratum_null_count=0,
        global_null_count=90,
    )
    nonproposal, _ = diagnostic._selection_decision(
        nominal_new_proposal=False,
        local_context_count=2,
        candidate_count=100,
        stratum_null_count=0,
        global_null_count=90,
    )
    assert not maximum_context
    assert not nonproposal
