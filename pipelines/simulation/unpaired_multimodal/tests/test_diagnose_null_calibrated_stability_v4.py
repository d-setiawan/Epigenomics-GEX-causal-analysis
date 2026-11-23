from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_null_calibrated_stability_v4 as diagnostic  # noqa: E402


def test_posterior_dominance_is_monotone_and_symmetric_at_equal_counts() -> None:
    tied = diagnostic._posterior_dominance_probability(60, 60)
    separated = diagnostic._posterior_dominance_probability(75, 50)
    reversed_probability = diagnostic._posterior_dominance_probability(50, 75)

    assert abs(tied - 0.5) < 1e-10
    assert separated > diagnostic.POSTERIOR_DOMINANCE_THRESHOLD
    assert reversed_probability < 1.0 - diagnostic.POSTERIOR_DOMINANCE_THRESHOLD
    assert abs(separated + reversed_probability - 1.0) < 1e-10


def test_null_calibrated_rule_keeps_a_majority_floor() -> None:
    assert diagnostic.MINIMUM_SELECTION_FREQUENCY == 0.50
    assert diagnostic.BOOTSTRAP_REPLICATES == 100
    assert diagnostic.POSTERIOR_DOMINANCE_THRESHOLD == 0.975
    assert "strictly_more_than_50" in diagnostic.FIXED_SELECTION_RULE
