"""Exact finite independent-source Poisson model; no data or truth loading.

``strict`` is a finite-source submodel of E2: log R_c = M_c (Z + mu).
``affine`` is a sensitivity model: log R_c = b_c + M_c Z. Its identification
requires the stronger absence of symmetries of the *centered* source law;
it is not an unqualified extension of E2 to arbitrary intercepts.

Each coordinate of Z is a standardized Bernoulli variable with a common,
learned probability below 1/2. Distinct probabilities and full-rank loadings
must be checked separately; finite likelihood alone does not establish them.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logsumexp
from threadpoolctl import threadpool_limits


def _counts(x):
    x = np.asarray(x, dtype=np.float64)
    if (x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all()
            or np.any(x < 0) or np.any(x != np.floor(x))):
        raise ValueError("counts must be a nonempty finite nonnegative integer matrix")
    return x


def source_grid(probabilities):
    """Enumerate independent unit-variance, zero-mean Bernoulli sources."""
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 1 or not 1 <= p.size <= 3 or not np.isfinite(p).all()
            or np.any(p <= 0) or np.any(p >= .5)):
        raise ValueError("one to three probabilities strictly between zero and 1/2 required")
    bits = np.asarray(list(itertools.product((0., 1.), repeat=len(p))))
    states = (bits - p) / np.sqrt(p * (1 - p))
    log_weights = (bits * np.log(p) + (1 - bits) * np.log1p(-p)).sum(axis=1)
    return bits, states, np.exp(log_weights)


def _mixture(weights, rates):
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if (r.ndim != 2 or min(r.shape) == 0 or w.shape != (len(r),)
            or not np.isfinite(r).all() or np.any(r <= 0)
            or not np.isfinite(w).all() or np.any(w <= 0)
            or not np.isclose(w.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("invalid positive-rate Poisson mixture")
    return w, r


def score_rows(x, weights, rates):
    """Return row log likelihoods and log joint component probabilities."""
    x = _counts(x)
    w, r = _mixture(weights, rates)
    if x.shape[1] != r.shape[1]:
        raise ValueError("count and rate feature widths differ")
    joint = (x @ np.log(r).T - r.sum(axis=1) + np.log(w)
             - gammaln(x + 1).sum(axis=1)[:, None])
    return logsumexp(joint, axis=1), joint


def posterior(x, weights, rates):
    _, joint = score_rows(x, weights, rates)
    shifted = joint - joint.max(axis=1, keepdims=True)
    return np.exp(shifted - logsumexp(shifted, axis=1, keepdims=True))


def cross_covariance(weights, rates_a, rates_b):
    """Cross-assay count covariance under conditional assay-noise independence."""
    w, a = _mixture(weights, rates_a)
    _, b = _mixture(weights, rates_b)
    return (a - w @ a).T @ (w[:, None] * (b - w @ b))


def _setup(counts, dimension, variant, seed):
    if not isinstance(dimension, int) or not 1 <= dimension <= 3:
        raise ValueError("dimension must be an integer from one to three")
    if variant not in ("strict", "affine"):
        raise ValueError("variant must be strict or affine")
    if not isinstance(counts, dict) or not counts or any(not isinstance(a, str) for a in counts):
        raise ValueError("counts must map assay names to matrices")
    data = {a: _counts(x) for a, x in counts.items()}
    if any(x.shape[1] < dimension for x in data.values()):
        raise ValueError("each assay must have at least dimension features")
    rng = np.random.default_rng(seed)
    initial = list(rng.normal(0, .8, dimension))
    bounds = [(-5., 5.)] * dimension
    names = [f"raw_p[{j}]" for j in range(dimension)]
    if variant == "strict":
        mean_first = 8. if seed % 2 == 0 else -8.
        initial += [mean_first] + [0.] * (dimension - 1)
        bounds += [(-20., 20.)] * dimension
        names += [f"mu[{j}]" for j in range(dimension)]
    for assay, x in data.items():
        logmean = np.log(np.maximum(x.mean(axis=0), 1e-4))
        matrix = rng.normal(0, .12, (x.shape[1], dimension))
        if variant == "strict":
            matrix[:, 0] = logmean / mean_first
        initial += matrix.ravel().tolist()
        bounds += [(-6., 6.)] * matrix.size
        names += [f"M/{assay}[{i},{j}]" for i in range(x.shape[1]) for j in range(dimension)]
        if variant == "affine":
            initial += np.clip(logmean, -12., 8.).tolist()
            bounds += [(-12., 8.)] * x.shape[1]
            names += [f"b/{assay}[{j}]" for j in range(x.shape[1])]
    return data, np.asarray(initial), bounds, names


def _decode(theta, widths, dimension, variant):
    p = .05 + .4 * expit(theta[:dimension])
    bits, centered, weights = source_grid(p)
    cursor = dimension
    mu = np.zeros(dimension)
    if variant == "strict":
        mu = theta[cursor:cursor + dimension]
        cursor += dimension
    states = centered + mu
    matrices, intercepts, log_rates = {}, {}, {}
    for assay, width in widths.items():
        matrices[assay] = theta[cursor:cursor + width * dimension].reshape(width, dimension)
        cursor += width * dimension
        intercepts[assay] = np.zeros(width)
        if variant == "affine":
            intercepts[assay] = theta[cursor:cursor + width]
            cursor += width
        log_rates[assay] = states @ matrices[assay].T + intercepts[assay]
    if cursor != len(theta):
        raise ValueError("parameter vector length mismatch")
    return p, bits, centered, states, weights, mu, matrices, intercepts, log_rates


def _objective(theta, data, dimension, variant, constants=None):
    """Exact normalized negative log likelihood and its analytic gradient."""
    widths = {a: x.shape[1] for a, x in data.items()}
    p, bits, centered, states, w, _, matrices, _, eta = _decode(theta, widths, dimension, variant)
    if constants is None:
        constants = {a: gammaln(x + 1).sum(axis=1) for a, x in data.items()}
    grad_states = np.zeros_like(states)
    grad_logw = np.zeros(len(w))
    matrix_grads, intercept_grads = {}, {}
    loss = 0.
    for assay, x in data.items():
        rates = np.exp(eta[assay])
        joint = x @ eta[assay].T - rates.sum(axis=1) + np.log(w) - constants[assay][:, None]
        scores = logsumexp(joint, axis=1)
        shifted = joint - joint.max(axis=1, keepdims=True)
        responsibility = np.exp(shifted - logsumexp(shifted, axis=1, keepdims=True))
        scale = 1. / (len(data) * x.shape[0] * x.shape[1])
        loss -= scores.sum() * scale
        mass = responsibility.sum(axis=0)
        grad_eta = (mass[:, None] * rates - responsibility.T @ x) * scale
        matrix_grads[assay] = grad_eta.T @ states
        intercept_grads[assay] = grad_eta.sum(axis=0)
        grad_states += grad_eta @ matrices[assay]
        grad_logw -= mass * scale
    variance = p * (1 - p)
    dz_dp = -1 / np.sqrt(variance) - centered * (1 - 2 * p) / (2 * variance)
    dlogw_dp = bits / p - (1 - bits) / (1 - p)
    sigmoid = expit(theta[:dimension])
    grad_p = ((grad_states * dz_dp).sum(axis=0) + grad_logw @ dlogw_dp)
    gradient = list(grad_p * .4 * sigmoid * (1 - sigmoid))
    if variant == "strict":
        gradient += grad_states.sum(axis=0).tolist()
    for assay in data:
        gradient += matrix_grads[assay].ravel().tolist()
        if variant == "affine":
            gradient += intercept_grads[assay].tolist()
    gradient = np.asarray(gradient)
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite exact objective or gradient")
    return float(loss), gradient


def fit_model(counts, dimension, variant, seed, maxiter=600, initial=None):
    """Fit marginal counts only; caller owns selection, checks and evaluation."""
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be positive")
    data, theta, bounds, names = _setup(counts, dimension, variant, seed)
    if initial is not None:
        if isinstance(initial, dict):
            if (initial.get("dimension") != dimension or initial.get("variant") != variant
                    or initial.get("assay_order") != list(data)
                    or initial.get("feature_counts") != {a: x.shape[1] for a, x in data.items()}):
                raise ValueError("warm-start model schema differs")
            initial = initial["parameter_vector"]
        proposed = np.asarray(initial, dtype=np.float64)
        limits = np.asarray(bounds)
        if (proposed.shape != theta.shape or not np.isfinite(proposed).all()
                or np.any(proposed < limits[:, 0]) or np.any(proposed > limits[:, 1])):
            raise ValueError("invalid or out-of-bounds warm-start parameters")
        theta = proposed.copy()
    constants = {a: gammaln(x + 1).sum(axis=1) for a, x in data.items()}
    with threadpool_limits(limits=1):
        start_loss, _ = _objective(theta, data, dimension, variant, constants)
        result = minimize(_objective, theta, args=(data, dimension, variant, constants),
                          method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 40})
        loss, gradient = _objective(result.x, data, dimension, variant, constants)
    widths = {a: x.shape[1] for a, x in data.items()}
    p, bits, _, states, w, mu, matrices, intercepts, eta = _decode(result.x, widths, dimension, variant)
    singular = {a: np.linalg.svd(m, compute_uv=False) for a, m in matrices.items()}
    limits = np.asarray(bounds)
    lower = np.isclose(result.x, limits[:, 0], atol=1e-6, rtol=0)
    upper = np.isclose(result.x, limits[:, 1], atol=1e-6, rtol=0)
    projected = gradient.copy()
    projected[lower & (gradient > 0)] = 0
    projected[upper & (gradient < 0)] = 0
    return {
        "variant": variant, "dimension": dimension, "seed": int(seed),
        "assay_order": list(data), "feature_counts": widths,
        "parameter_vector": result.x.tolist(), "p": p.tolist(), "mu": mu.tolist(),
        "probabilities": p.tolist(), "source_mean": mu.tolist() if variant == "strict" else None,
        "loadings": {a: m.tolist() for a, m in matrices.items()},
        "intercepts": {a: b.tolist() for a, b in intercepts.items()},
        "state_bits": bits.tolist(), "state_sources": states.tolist(),
        "state_weights": w.tolist(), "state_log_rates": {a: e.tolist() for a, e in eta.items()},
        "state_rates": {a: np.exp(e).tolist() for a, e in eta.items()},
        "initial_loss": start_loss, "loss": loss, "objective": loss, "converged": bool(result.success),
        "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "objective_evaluations": int(result.nfev),
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "projected_gradient_inf_norm": float(np.max(np.abs(projected))),
        "bound_hits": [names[i] for i in np.flatnonzero(lower | upper)],
        "singular_values": {a: s.tolist() for a, s in singular.items()},
        "singular_value_ratios": {a: float(s[-1] / s[0]) if s[0] > 0 else 0. for a, s in singular.items()},
        "minimum_probability_gap": float(np.min(np.diff(np.sort(p)))) if dimension > 1 else None,
        "objective_definition": "mean assay marginal NLL per cell per feature; assays equally weighted",
        "interpretation": "algebra and numerical diagnostics do not verify biological identification assumptions",
    }
