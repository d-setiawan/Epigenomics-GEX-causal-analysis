from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_joint_sliced_consensus_v4 as diagnostic  # noqa: E402


def test_stable_mask_requires_proposal_and_threshold() -> None:
    proposal = np.asarray([True, True, False, True], dtype=bool)
    counts = np.asarray([40, 39, 50, 45], dtype=np.int64)

    selected = diagnostic._stable_mask(
        proposal, counts, replicates=50, threshold=0.80
    )

    assert selected.tolist() == [True, False, False, True]


def test_stable_mask_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="valid range"):
        diagnostic._stable_mask(
            np.asarray([True], dtype=bool),
            np.asarray([51], dtype=np.int64),
            replicates=50,
            threshold=0.80,
        )


def test_fixed_rule_names_both_views_and_no_direction() -> None:
    assert "standardized_joint" in diagnostic.FIXED_SELECTION_RULE
    assert "rank_copula" in diagnostic.FIXED_SELECTION_RULE
    assert diagnostic.VIEW_NAMES == (
        "joint_global_mirror",
        "copula_global_mirror",
    )
