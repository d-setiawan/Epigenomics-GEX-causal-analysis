from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_soft_reporter_representation_v4 as diagnostic


def _table():
    return pd.DataFrame([
        {"block": "target", "mode": "both", "expected_assay_count": 6,
         "truth_pair": True, diagnostic.PRIMARY_SCORE: 0.12},
        {"block": "train", "mode": "both", "expected_assay_count": 6,
         "truth_pair": False, diagnostic.PRIMARY_SCORE: 0.10},
        {"block": "train", "mode": "both", "expected_assay_count": 5,
         "truth_pair": False, diagnostic.PRIMARY_SCORE: 0.9},
        {"block": "train", "mode": "both", "expected_assay_count": 6,
         "truth_pair": True, diagnostic.PRIMARY_SCORE: 0.99},
    ])


def test_target_labels_do_not_change_target_decision():
    frame = _table()
    original = diagnostic.select_diagnostic(frame).iloc[0]
    frame.loc[0, "truth_pair"] = False
    changed = diagnostic.select_diagnostic(frame).iloc[0]
    assert bool(original.selected_diagnostic_proposal)
    assert original.selected_diagnostic_proposal == changed.selected_diagnostic_proposal
    assert original.null_upper_envelope == changed.null_upper_envelope == 0.1


def test_missing_matching_null_stratum_abstains():
    frame = _table()
    frame.loc[0, "expected_assay_count"] = 4
    result = diagnostic.select_diagnostic(frame).iloc[0]
    assert not result.null_reference_available
    assert not result.selected_diagnostic_proposal


def test_strict_null_tie_and_negative_scores_abstain():
    frame = _table()
    frame.loc[0, diagnostic.PRIMARY_SCORE] = 0.10
    assert not diagnostic.select_diagnostic(frame).iloc[0].selected_diagnostic_proposal
    frame.loc[0, diagnostic.PRIMARY_SCORE] = -0.05
    frame.loc[1, diagnostic.PRIMARY_SCORE] = -0.10
    assert not diagnostic.select_diagnostic(frame).iloc[0].selected_diagnostic_proposal
