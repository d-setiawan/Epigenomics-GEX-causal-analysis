"""Selection checks require no benchmark inputs or generated worlds."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from count_transport_run import acceptance, choose


def candidate(score, dimension=1, seed=0, converged=True):
    return {"calibration_score": score, "dimension": dimension, "seed": seed,
            "model": {"converged": converged}}


def test_selection_is_calibration_only_and_prefers_convergence():
    candidates = {"nonconverged": candidate(10, converged=False), "best": candidate(2), "other": candidate(1)}
    assert choose(candidates) == ("best", "converged_calibration_selection")


def test_selection_ties_and_diagnostic_fallback():
    assert choose({"d2": candidate(2, 2), "seed1": candidate(2, 1, 1), "d1": candidate(2)})[0] == "d1"
    assert choose({"only": candidate(2, converged=False)})[1] == "finite_nonconverged_diagnostic"
    with pytest.raises(ValueError):
        choose({"failed": {"model": None, "calibration_score": None}})


def test_acceptance_requires_each_numerical_gate():
    model = {"converged": True, "dimension": 2, "objective": 1., "state_rates": {"A": [[1., 2.]]},
             "singular_value_ratios": {"A": .2}, "minimum_probability_gap": .1}
    assert acceptance(model)["pass"]
    for key, value in (("converged", False), ("minimum_probability_gap", .01),
                       ("singular_value_ratios", {"A": .0001}), ("objective", float("inf"))):
        assert not acceptance({**model, key: value})["pass"]
    assert acceptance({**model, "dimension": 1, "minimum_probability_gap": None})["pass"]
