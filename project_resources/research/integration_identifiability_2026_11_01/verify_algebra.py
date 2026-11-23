"""Check exact identities in theory notes 07/08; no sampled cells or fitting."""

import hashlib
import json
import platform
from pathlib import Path

import sympy as sp


def main():
    rho, state, dose = sp.symbols("rho state dose", real=True)
    depth_x, depth_y, n = sp.symbols("depth_x depth_y n", positive=True)
    noise = sp.sqrt(1 - rho**2)
    forward = sp.Matrix([[1, 0, 0], [1, 1, 0], [1, rho, noise]])
    reverse = sp.Matrix([[1, 0, 0], [1, noise, rho], [1, 0, 1]])
    covariance = forward * forward.T
    expected = sp.Matrix([[1, 1, 1], [1, 2, 1 + rho], [1, 1 + rho, 2]])
    residual_cov = covariance[1, 2] - covariance[1, 0] * covariance[0, 2]
    conditional_sx = covariance[0, 1] - covariance[0, 2] * covariance[2, 1] / 2
    conditional_sy = covariance[0, 2] - covariance[0, 1] * covariance[1, 2] / 2
    xy_exponential_moment = sp.exp(2 * state + (1 + 1 + 2 * residual_cov) / 2)
    count_cov = depth_x * depth_y * (xy_exponential_moment - sp.exp(2 * state + 1))
    mean_intervention = (1 - rho) * state + rho * dose
    block = sp.Matrix([[1, 1], [1, 2]])
    coupling = sp.symbols("coupling", real=True)
    penalty = sp.symbols("penalty", positive=True)
    transport_objective = 2 - 2 * coupling - penalty * sp.log(1 - coupling**2) / 2
    transport_first = sp.diff(transport_objective, coupling)
    optimum = (sp.sqrt(penalty**2 + 16) - penalty) / 4
    reflection = 2 * state - dose
    other_dose = sp.symbols("other_dose", real=True)
    identities = {
        "joint_covariance": covariance - expected,
        "assay_A_law_covariance_invariant": covariance.extract([0, 1], [0, 1]) - block,
        "assay_B_law_covariance_invariant": covariance.extract([0, 2], [0, 2]) - block,
        "first_leading_minor": covariance[0, 0] - 1,
        "second_leading_minor": covariance[:2, :2].det() - 1,
        "positive_definiteness_determinant": covariance.det() - (1 - rho**2),
        "conditional_XY": residual_cov - rho,
        "conditional_SX": conditional_sx - (1 - rho) / 2,
        "conditional_SY": conditional_sy - (1 - rho) / 2,
        "intervention_derivative": sp.diff(mean_intervention, dose) - rho,
        "reverse_model_same_joint": reverse * reverse.T - covariance,
        "poisson_count_covariance": count_cov - depth_x * depth_y * sp.exp(2 * state + 1) * (sp.exp(rho) - 1),
        "paired_shuffled_mean_covariance": n * covariance[1, 2] / n**2 - (1 + rho) / n,
        "paired_shuffled_residual_mean_covariance": n * residual_cov / n**2 - rho / n,
        "transport_first_derivative": transport_first - (-2 + penalty * coupling / (1 - coupling**2)),
        "transport_second_derivative": sp.diff(transport_first, coupling) - penalty * (1 + coupling**2) / (1 - coupling**2)**2,
        "transport_optimal_stationarity": transport_first.subs(coupling, optimum),
        "transport_penalty_three": optimum.subs(penalty, 3) - sp.Rational(1, 2),
        "transport_zero_penalty_limit": sp.limit(optimum, penalty, 0, dir="+") - 1,
        "transport_infinite_penalty_limit": sp.limit(optimum, penalty, sp.oo),
        "gaussian_log_density_ratio_expectation": -(2 - 2 * coupling**2) / (2 * (1 - coupling**2)) + 1,
        "reflection_cycle": reflection.subs(dose, reflection) - dose,
        "reflection_preserves_distances": (reflection - reflection.subs(dose, other_dose))**2 - (dose - other_dose)**2,
    }
    for label, difference in identities.items():
        entries = list(difference) if isinstance(difference, sp.MatrixBase) else [difference]
        assert all(sp.simplify(value) == 0 for value in entries), label
    # Concrete model checks supplement symbolic equalities; domain inequalities are in the note.
    for value in (sp.S.Zero, sp.Rational(1, 2)):
        matrix = covariance.subs(rho, value)
        assert all(matrix[:k, :k].det() > 0 for k in (1, 2, 3))
        assert conditional_sx.subs(rho, value) > 0
        assert conditional_sy.subs(rho, value) > 0
    assert residual_cov.subs(rho, 0) == 0
    assert residual_cov.subs(rho, sp.Rational(1, 2)) > 0
    source = Path(__file__).resolve()
    report = {
        "status": "passed",
        "scope": "Exact algebra identities; the written observation-law argument supplies the proof.",
        "domain": "0 <= rho < 1; positive depths; positive integer n; positive OT penalty and -1 < coupling < 1",
        "symbolic_identities": list(identities),
        "concrete_models": ["rho=0", "rho=1/2"],
        "python": platform.python_version(),
        "sympy": sp.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sampled_cells": 0,
        "fitted_models": 0,
    }
    source.with_name("symbolic_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(identities)} symbolic identities and both concrete model checks.")


if __name__ == "__main__":
    main()
