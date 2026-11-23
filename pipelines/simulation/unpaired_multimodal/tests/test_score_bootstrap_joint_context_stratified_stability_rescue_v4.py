from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_bootstrap_joint_context_stratified_stability_rescue_v4 as scorer  # noqa: E402


def test_scorer_schema_and_interpretation_are_explicit() -> None:
    assert scorer.SCHEMA.endswith(".scorer.v1")
    assert "context" in scorer.ALGORITHM.lower()
    assert scorer.benchmark.STRESS_LABEL == (
        "identifiability_stress_test_not_expected_success_criterion"
    )
