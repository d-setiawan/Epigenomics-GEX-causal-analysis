from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import heldout_synchronized_majority_rescue_v4 as rescue  # noqa: E402


def _assays() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "modality": modality,
                "external_state_reporter_members": f"{modality}_s",
                "within_assay_pair_resolved_by_component": True,
                "reporter_context_components_identified": True,
                "reporter_all_context_bootstrap_component_assignments_supported": True,
            }
            for modality in ("a", "b", "c")
        ]
    )


def _contexts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "context_index": 2,
                "context_label": "feature_fold_1",
                "reporter_groups_json": json.dumps([["b", "c"]]),
            },
            {
                "context_index": 0,
                "context_label": "full_fit",
                "reporter_groups_json": "[]",
            },
            {
                "context_index": 1,
                "context_label": "feature_fold_0",
                "reporter_groups_json": json.dumps([["a", "b"]]),
            },
        ]
    )


def test_majority_rescue_requires_heldout_coassignment_and_novel_pair() -> None:
    decisions = pd.DataFrame(
        [
            {
                "left_modality": "a",
                "right_modality": "b",
                "statistical_context_support_pattern": 110,
            },
            {
                "left_modality": "a",
                "right_modality": "c",
                "statistical_context_support_pattern": 111,
            },
            {
                "left_modality": "b",
                "right_modality": "c",
                "statistical_context_support_pattern": 11,
            },
        ]
    )
    represented = pd.DataFrame(
        [{"candidate_assay_pairs_json": json.dumps([["b", "c"]])}]
    )

    candidates, audit = rescue.build_majority_rescues(
        _assays(), represented, decisions, _contexts()
    )

    assert len(candidates) == 1
    candidate = candidates.iloc[0]
    assert candidate["modalities"] == "a;b"
    assert candidate["statistical_context_support_pattern"] == "110"
    assert candidate["heldout_global_reporter_coassignment_pattern"] == "10"
    assert candidate["heldout_global_reporter_coassignment_count"] == 1
    assert candidate["hypothesis_selected"]
    assert candidate["hypothesis_type"] == "cross_assay_shared_state_equivalence"
    assert not candidate["within_assay_direct_structure_claimed"]
    assert not candidate["exact_cross_assay_pair_claimed"]
    assert not candidate["cross_assay_direction_claimed"]
    assert not candidate["candidate_pairs_may_be_expanded_as_edges"]

    status = {
        frozenset((row.left_modality, row.right_modality)): row.selection_status
        for row in audit.itertuples(index=False)
    }
    assert status[frozenset(("a", "c"))] == (
        "unresolved_without_heldout_global_coassignment"
    )
    assert status[frozenset(("b", "c"))] == (
        "diagnostic_duplicate_of_v444_output"
    )


def test_context_parser_rejects_missing_heldout_fold() -> None:
    with pytest.raises(ValueError, match="context indices|full fit"):
        rescue._context_pair_patterns(_contexts().iloc[:2])
