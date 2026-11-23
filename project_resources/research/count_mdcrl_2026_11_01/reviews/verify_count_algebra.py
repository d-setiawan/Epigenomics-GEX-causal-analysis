"""Independent symbolic/code checks; no project data reads or model-data fits."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp
from sympy.functions.combinatorial.numbers import stirling

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
import count_sources as cs
import apply_counts as runner


def main():
    checks = []
    y = sp.symbols("y0:3")
    rate = sp.symbols("l0:3", positive=True)
    t = sp.symbols("t", positive=True)
    w = sp.symbols("w0:3", real=True)
    center = sp.symbols("c", real=True)

    def expectation(expression):
        out = 0
        for powers, coefficient in sp.Poly(sp.expand(expression), *y).terms():
            term = coefficient
            for index, degree in enumerate(powers):
                term *= sum(stirling(degree, k, kind=2) * (t * rate[index]) ** k
                            for k in range(degree + 1))
            out += term
        return sp.expand(out)

    for i, j in itertools.product(range(3), repeat=2):
        raw = y[i] * y[j] / t**2 - (y[i] / t**2 if i == j else 0)
        assert sp.simplify(expectation(raw) - rate[i] * rate[j]) == 0
    checks.append("all_nine_conditional_second_factorial_moments")

    for i, j, k in itertools.product(range(3), repeat=3):
        raw = y[i] * y[j] * y[k] / t**3
        if i == j:
            raw -= y[i] * y[k] / t**3
        if i == k:
            raw -= y[i] * y[j] / t**3
        if j == k:
            raw -= y[i] * y[j] / t**3
        if i == j == k:
            raw += 2 * y[i] / t**3
        assert sp.simplify(expectation(raw) - rate[i] * rate[j] * rate[k]) == 0
    checks.append("all_27_conditional_third_factorial_moments")

    power = {r: sum(y[j] * (w[j] / t) ** r for j in range(3)) for r in range(1, 5)}
    z = power[1] - center
    factorial = [z, z**2 - power[2], z**3 - 3*z*power[2] + 2*power[3],
                 z**4 - 6*z**2*power[2] + 3*power[2]**2 + 8*z*power[3] - 6*power[4]]
    projected_rate = sum(w[j] * rate[j] for j in range(3)) - center
    for degree, expression in enumerate(factorial, 1):
        assert sp.simplify(expectation(expression) - projected_rate**degree) == 0
    checks.append("four_signed_centered_projected_factorial_moment_identities")

    omega = sp.symbols("omega", real=True)
    first_log = -sp.I * omega * center + sum(
        t * rate[j] * ((1 + sp.I * omega * w[j] / t) - 1) for j in range(3))
    second_log = sum(t * rate[j] * (
        (1 + sp.I * omega * w[j] / t) * (1 - sp.I * omega * w[j] / t) - 1)
        for j in range(3))
    assert sp.simplify(first_log - sp.I * omega * projected_rate) == 0
    assert sp.simplify(second_log - omega**2 / t * sum(rate[j]*w[j]**2 for j in range(3))) == 0
    checks.append("conditional_deconvolved_cf_mean_and_absolute_second_moment")

    # Compare the implementation with the alternative central-cumulant identity.
    counts = np.array([[0,2,1], [4,0,2], [1,1,0], [2,3,1], [0,1,4], [3,2,0]], dtype=float)
    exposures = np.array([.4, 1.2, .75, 2., .6, 1.5])
    u, q = counts / exposures[:, None], counts / exposures[:, None]**2
    mu, _, covariance, third, _, _ = cs._cumulants(counts, exposures, True)
    uc, qc = u-u.mean(0), q-q.mean(0)
    independently_centered = np.einsum("ni,nj,nk->ijk", uc, uc, uc) / len(counts)
    for i, j, k in itertools.product(range(3), repeat=3):
        if j == k:
            independently_centered[i,j,k] -= np.mean(uc[:,i] * qc[:,j])
        if i == k:
            independently_centered[i,j,k] -= np.mean(uc[:,j] * qc[:,i])
        if i == j:
            independently_centered[i,j,k] -= np.mean(uc[:,k] * qc[:,i])
        if i == j == k:
            independently_centered[i,j,k] += 2*np.mean(counts[:,i]/exposures**3)
    np.testing.assert_allclose(third, independently_centered, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(covariance, uc.T@uc/len(counts)-np.diag(q.mean(0)), atol=1e-12)
    checks.append("implemented_exposure_aware_cumulants_equal_independent_central_formula")

    direction = np.array([[.3, -.2, .4]])
    fixed_center = np.array([.2, .7, .1])
    got = cs.projected_factorial_moments(counts, exposures, direction, fixed_center, True)
    evaluate = sp.lambdify((*y, t, *w, center), factorial, "numpy")
    expected = np.asarray([evaluate(*row, exp, *direction[0], direction[0]@fixed_center)
                           for row, exp in zip(counts, exposures)])
    np.testing.assert_allclose(got["moment_contributions"][:,0,:], expected, atol=1e-12)
    checks.append("implemented_scalar_contributions_equal_symbolically_verified_polynomials")

    actual, errors, _, valid = runner.characteristic_fingerprint(
        counts, exposures, direction, fixed_center, True)
    expected_cf, expected_se = [], []
    for frequency in runner.GRID:
        weights = np.prod((1+1j*frequency*direction[0]/exposures[:,None])**counts, axis=1)
        weights *= np.exp(-1j*frequency*(direction[0]@fixed_center))
        expected_cf.append(weights.mean())
        expected_se.append(np.sqrt(np.sum(np.abs(weights-weights.mean())**2)/(len(weights)*(len(weights)-1))))
    np.testing.assert_allclose(actual[0], expected_cf, atol=1e-12)
    np.testing.assert_allclose(errors[0], expected_se, atol=1e-12)
    checks.append("implemented_cf_and_complex_mean_standard_error_equal_direct_product")

    # Exact tensors of bounded independent Bernoulli sources; no sampled world.
    mixing = np.array([[1., -.2], [.3, .8], [0., 1.1], [.9, 0.]])
    probabilities = np.array([.2, .35])
    skewness = (1-2*probabilities)/np.sqrt(probabilities*(1-probabilities))
    cov = mixing@mixing.T
    tensor = np.einsum("ir,jr,kr,r->ijk", mixing, mixing, mixing, skewness)
    scale = np.sqrt(np.diag(cov) + [1., 2., 3., 4.])
    fitted = cs._spectral_sources(cov, tensor, scale, 2, 11)
    assert fitted["status"] == "ok"
    np.testing.assert_allclose(fitted["unmixing"]@cov@fitted["unmixing"].T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(fitted["unmixing"]@fitted["mixing"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(cs.align_mixing(mixing, fitted["mixing"])["aligned_mixing"], mixing, atol=1e-12)
    checks.append("exact_bounded_source_tensor_recovers_units_up_to_signed_permutation")

    # With 49 nulls, <=.05 accepts at most one null maximum above the statistic.
    nulls = np.arange(49.)
    for observed, expected_p, expected_pass in [(48.5,.02,True),(47.5,.04,True),(46.5,.06,False),(47.,.06,False)]:
        p = (1 + np.sum(nulls >= observed))/50
        assert p == expected_p and (p <= .05) == expected_pass
    checks.append("49_permutation_plus_one_boundary_and_conservative_ties")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"status": "passed", "check_groups": checks, "project_data_read": False,
              "model_data_fits": 0, "new_benchmark_worlds": 0,
              "source_sha256": {p.name: sha(p) for p in [STUDY/"protocol.json", STUDY/"count_sources.py", STUDY/"apply_counts.py", Path(__file__)]}}
    destination = Path(__file__).with_name("count_algebra_audit.json")
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
