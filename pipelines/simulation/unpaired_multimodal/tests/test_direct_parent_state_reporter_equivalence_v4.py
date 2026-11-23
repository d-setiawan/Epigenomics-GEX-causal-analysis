from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import direct_parent_state_reporter_equivalence_v4 as equivalence  # noqa: E402


def _direct() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_assay": "chrom_A", "parent": "p", "child": "c"},
            {"source_assay": "chrom_A", "parent": "q", "child": "c"},
        ]
    )


def _state(*, p: bool = True, q: bool = False, s: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": "p", "modality": "chrom_A", "state_v427_supported_member": p},
            {"feature": "q", "modality": "chrom_A", "state_v427_supported_member": q},
            {"feature": "c", "modality": "chrom_A", "state_v427_supported_member": True},
            {"feature": "s", "modality": "chrom_A", "state_v427_supported_member": s},
            {"feature": "n1", "modality": "chrom_A", "state_v427_supported_member": False},
        ]
    )


def test_selected_hypothesis_is_one_unexpanded_two_pair_equivalence() -> None:
    result = equivalence.build_direct_parent_state_reporter_equivalence(
        _direct(), _state()
    )

    assert len(result.decisions) == 1
    row = result.decisions.iloc[0]
    assert bool(row["hypothesis_selected"])
    assert row["direct_parent_members"] == "p;q"
    assert row["external_state_reporter_members"] == "s"
    assert equivalence.parse_candidate_pairs(row["candidate_pairs_json"]) == (
        ("p", "s"),
        ("q", "s"),
    )
    assert int(row["candidate_pair_count"]) == 2
    assert not bool(row["exact_parent_reporter_pair_identified"])
    assert not bool(row["candidate_pairs_may_be_expanded_as_edges"])
    assert not bool(row["exact_cross_assay_pair_claimed"])


def test_no_supported_parent_is_an_explicit_abstention() -> None:
    row = equivalence.build_direct_parent_state_reporter_equivalence(
        _direct(), _state(p=False, q=False)
    ).decisions.iloc[0]

    assert not bool(row["hypothesis_selected"])
    assert row["hypothesis_status"] == equivalence.NO_PARENT_STATE_STATUS
    assert row["candidate_pairs_json"] == "[]"


def test_no_external_reporter_is_an_explicit_abstention() -> None:
    row = equivalence.build_direct_parent_state_reporter_equivalence(
        _direct(), _state(s=False)
    ).decisions.iloc[0]

    assert not bool(row["hypothesis_selected"])
    assert row["hypothesis_status"] == equivalence.NO_EXTERNAL_REPORTER_STATUS
    assert row["candidate_pairs_json"] == "[]"


def test_multiple_external_reporters_enlarge_one_ambiguity_set() -> None:
    state = pd.concat(
        [
            _state(),
            pd.DataFrame(
                [
                    {
                        "feature": "s2",
                        "modality": "chrom_A",
                        "state_v427_supported_member": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    row = equivalence.build_direct_parent_state_reporter_equivalence(
        _direct(), state
    ).decisions.iloc[0]

    assert int(row["candidate_pair_count"]) == 4
    assert len(equivalence.parse_candidate_pairs(row["candidate_pairs_json"])) == 4
    assert len(result := equivalence.DirectParentStateReporterResult(pd.DataFrame([row])).selected_hypotheses) == 1


def test_truth_columns_are_rejected_at_application_boundary() -> None:
    state = _state().assign(truth_role=["P", "Q", "C", "S", "N1"])

    with pytest.raises(ValueError, match="truth/scorer"):
        equivalence.build_direct_parent_state_reporter_equivalence(
            _direct(), state
        )


def test_malformed_direct_motif_is_rejected() -> None:
    malformed = _direct().copy()
    malformed.loc[1, "child"] = "other_child"
    state = pd.concat(
        [
            _state(),
            pd.DataFrame(
                [
                    {
                        "feature": "other_child",
                        "modality": "chrom_A",
                        "state_v427_supported_member": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly two distinct parents"):
        equivalence.build_direct_parent_state_reporter_equivalence(
            malformed, state
        )
