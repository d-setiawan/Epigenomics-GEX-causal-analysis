"""Exploratory label-symmetric Bernoulli optimization of the strict count law.

Optimization permits probabilities on either side of one half. Afterwards,
every probability above one half is complemented, with the matching source
mean and loading columns negated in every assay. This is a global relabeling
of the identical joint law, never a pairing- or truth-based source alignment.

The logistic probability range also extends the old canonical probability
domain. Consequently this experiment changes both the parametrization and
the feasible domain; it does not isolate an optimizer-only effect.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import expit, gammaln, logit, logsumexp
from threadpoolctl import threadpool_limits

import count_transport_converged as convergence
import count_transport_model as legacy

PARAMETER_SCHEMA = "logistic_bernoulli_strict_v1: logit_p, source_mean, assay_order loading rows"


def _setup(counts, dimension, seed):
    """Preserve the old initializer's physical count distribution exactly."""
    data, theta, bounds, names = legacy._setup(counts, dimension, "strict", seed)
    theta[:dimension] = logit(.05 + .4 * expit(theta[:dimension]))
    names[:dimension] = [f"logit_p[{j}]" for j in range(dimension)]
    return data, theta, bounds, names


def _decode(theta, widths, dimension):
    p = expit(theta[:dimension])
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("Logistic Bernoulli probabilities must lie strictly inside (0,1)")
    bits = np.asarray(list(itertools.product((0., 1.), repeat=dimension)))
    centered = (bits-p) / np.sqrt(p*(1-p))
    mu = theta[dimension:2*dimension]
    states = centered+mu
    logweights = (bits*np.log(p) + (1-bits)*np.log1p(-p)).sum(axis=1)
    cursor = 2*dimension
    matrices, eta = {}, {}
    for assay, width in widths.items():
        matrices[assay] = theta[cursor:cursor+width*dimension].reshape(width, dimension)
        cursor += width*dimension
        eta[assay] = states @ matrices[assay].T
    if cursor != len(theta):
        raise ValueError("Parameter vector length mismatch")
    return p, bits, centered, states, logweights, mu, matrices, eta


def _objective(theta, data, dimension, constants=None):
    """Exact marginal count NLL; analytic gradient through logistic sources."""
    widths = {a:x.shape[1] for a,x in data.items()}
    p, bits, centered, states, logweights, _, matrices, eta = _decode(theta, widths, dimension)
    if constants is None:
        constants = {a:gammaln(x+1).sum(axis=1) for a,x in data.items()}
    state_gradient = np.zeros_like(states)
    weight_gradient = np.zeros(len(bits))
    matrix_gradients = {}
    loss = 0.
    for assay, x in data.items():
        rates = np.exp(eta[assay])
        joint = x @ eta[assay].T - rates.sum(axis=1) + logweights - constants[assay][:,None]
        scores = logsumexp(joint, axis=1)
        shifted = joint-joint.max(axis=1, keepdims=True)
        posterior = np.exp(shifted-logsumexp(shifted, axis=1, keepdims=True))
        scale = 1. / (len(data)*x.shape[0]*x.shape[1])
        loss -= scores.sum()*scale
        mass = posterior.sum(axis=0)
        eta_gradient = (mass[:,None]*rates - posterior.T @ x)*scale
        matrix_gradients[assay] = eta_gradient.T @ states
        state_gradient += eta_gradient @ matrices[assay]
        weight_gradient -= mass*scale
    variance = p*(1-p)
    centered_derivative = -1/np.sqrt(variance) - centered*(1-2*p)/(2*variance)
    logweight_derivative = bits/p - (1-bits)/(1-p)
    probability_gradient = (state_gradient*centered_derivative).sum(axis=0) + weight_gradient @ logweight_derivative
    gradient = np.concatenate([probability_gradient*variance, state_gradient.sum(axis=0),
                               *[matrix_gradients[a].ravel() for a in data]])
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite label-optimizer objective or gradient")
    return float(loss), gradient


def canonicalize(theta, widths, dimension):
    """Globally complement Bernoulli labels; preserve the full joint law."""
    p = expit(theta[:dimension])
    if np.any(p == .5):
        raise ValueError("Exactly 0.5 source probability has no asymmetric canonical representation")
    flips = p > .5
    signs = np.where(flips, -1., 1.)
    canonical = np.asarray(theta, dtype=float).copy()
    canonical[:dimension] *= signs
    canonical[dimension:2*dimension] *= signs
    cursor = 2*dimension
    for width in widths.values():
        matrix = canonical[cursor:cursor+width*dimension].reshape(width, dimension)
        matrix *= signs
        cursor += width*dimension
    if cursor != len(theta):
        raise ValueError("Parameter vector length mismatch")
    return canonical, np.flatnonzero(flips).tolist()


def _pack(theta, result, data, dimension, seed, bounds, names, initial_loss, gtol, continuation):
    widths = {a:x.shape[1] for a,x in data.items()}
    p, bits, _, states, logweights, mu, matrices, eta = _decode(theta, widths, dimension)
    singular = {a:np.linalg.svd(m, compute_uv=False) for a,m in matrices.items()}
    limits = np.asarray(bounds)
    hits = np.isclose(theta,limits[:,0],atol=1e-6,rtol=0) | np.isclose(theta,limits[:,1],atol=1e-6,rtol=0)
    old_lower, old_upper = .05+.4*expit(-5), .05+.4*expit(5)
    return {
        "variant":"strict", "dimension":dimension, "seed":int(seed),
        "assay_order":list(data), "feature_counts":widths,
        "parameter_vector":theta.tolist(), "parameter_schema":PARAMETER_SCHEMA,
        "p":p.tolist(), "mu":mu.tolist(), "probabilities":p.tolist(), "source_mean":mu.tolist(),
        "loadings":{a:m.tolist() for a,m in matrices.items()},
        "intercepts":{a:np.zeros(width).tolist() for a,width in widths.items()},
        "state_bits":bits.tolist(), "state_sources":states.tolist(), "state_weights":np.exp(logweights).tolist(),
        "state_log_rates":{a:e.tolist() for a,e in eta.items()},
        "state_rates":{a:np.exp(e).tolist() for a,e in eta.items()},
        "initial_loss":initial_loss, "objective":float(result.fun), "loss":float(result.fun),
        "converged":result.strongly_converged, "optimizer_success":bool(result.success),
        "status":int(result.status), "message":str(result.message), "stop_reason":continuation["stop_reason"],
        "iterations":continuation["iterations"], "objective_evaluations":continuation["objective_evaluations"],
        "gradient_inf_norm":float(np.max(np.abs(result.jac))),
        "projected_gradient_inf_norm":result.projected_gradient_inf_norm,
        "convergence_tolerance":gtol,
        "convergence_definition":"finite objective and unit-step box-gradient mapping infinity norm <= tolerance",
        "bound_hits":[names[i] for i in np.flatnonzero(hits)],
        "singular_values":{a:s.tolist() for a,s in singular.items()},
        "singular_value_ratios":{a:float(s[-1]/s[0]) if s[0]>0 else 0. for a,s in singular.items()},
        "minimum_probability_gap":float(np.min(np.diff(np.sort(p)))) if dimension>1 else None,
        "canonical_outside_legacy_probability_range":bool(np.any(p<old_lower) or np.any(p>old_upper)),
        "canonical_probability_domain":{"minimum":float(expit(-5)), "maximum_exclusive":.5},
        "objective_definition":"mean assay marginal NLL per cell per feature; assays equally weighted",
        "interpretation":"Exploratory reparametrization with a wider probability domain; numerical stationarity is not global optimality",
    }


def fit_label_model(counts, dimension, seed, chunk_iterations=600, chunk_evaluations=15000, gtol=1e-6):
    """Fit public marginal counts only, then canonicalize every assay together."""
    if (not isinstance(chunk_iterations,int) or chunk_iterations<1
            or not isinstance(chunk_evaluations,int) or chunk_evaluations<1
            or not np.isfinite(gtol) or gtol<=0):
        raise ValueError("Positive integer chunk sizes and a positive finite gradient tolerance required")
    data, initial, bounds, names = _setup(counts,dimension,seed)
    constants = {a:gammaln(x+1).sum(axis=1) for a,x in data.items()}
    widths = {a:x.shape[1] for a,x in data.items()}
    def objective(theta):
        return _objective(theta,data,dimension,constants)
    with threadpool_limits(limits=1):
        initial_loss = objective(initial)[0]
        _, result, continuation = convergence._drive(objective,initial,bounds,chunk_iterations,chunk_evaluations,gtol)
    raw = {"parameter_vector":result.x.tolist(), "parameter_schema":PARAMETER_SCHEMA,
           "initial_parameter_vector":initial.tolist(), "initial_probabilities":expit(initial[:dimension]).tolist(),
           "probabilities":expit(result.x[:dimension]).tolist(),
           "objective":float(result.fun), "gradient_inf_norm":float(np.max(np.abs(result.jac))),
           "projected_gradient_inf_norm":result.projected_gradient_inf_norm,
           "converged":result.strongly_converged, "status":int(result.status), "message":str(result.message)}
    try:
        canonical, flips = canonicalize(result.x,widths,dimension)
    except ValueError as exc:
        raw["flipped_coordinates"] = None
        return {"model":None, "raw_optimizer":raw, "continuation":continuation, "error":str(exc)}
    raw["flipped_coordinates"] = flips
    model = _pack(canonical,result,data,dimension,seed,bounds,names,initial_loss,gtol,continuation)
    return {"model":model, "raw_optimizer":raw, "continuation":continuation, "error":None}
