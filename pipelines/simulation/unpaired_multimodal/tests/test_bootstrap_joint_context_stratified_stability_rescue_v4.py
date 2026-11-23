from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_joint_context_stratified_stability_rescue_v4 as calibration  # noqa: E402


def test_candidate_contract_never_claims_direct_or_directional_structure() -> None:
    assert "candidate_pairs_may_be_expanded_as_edges" in calibration.CANDIDATE_COLUMNS
    assert "exact_cross_assay_pair_claimed" in calibration.CANDIDATE_COLUMNS
    assert "cross_assay_direction_claimed" in calibration.CANDIDATE_COLUMNS
    assert calibration.FIXED_SELECTION_RULE.startswith(
        calibration.diagnostic.FIXED_SELECTION_RULE
    )


def test_empty_union_has_stable_candidate_schema() -> None:
    frame, ids = calibration._union_candidates(
        target_keys=["block__world_seed1"],
        world_frames={
            "block__world_seed1": pd.DataFrame(
                {"block": ["block"], "seed": [1]}
            )
        },
        baseline={},
        selected_rows={},
    )
    assert frame.empty
    assert list(frame.columns) == calibration.CANDIDATE_COLUMNS
    assert ids == {}
