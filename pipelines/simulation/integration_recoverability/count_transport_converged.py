"""Continue the sealed strict count model to numerical stationarity.

The legacy model is imported without modification. A capped fit is retained as
a checkpoint, then optimization continues from its exact parameter vector.
Iteration/evaluation limits bound individual SciPy calls only: they never end
the uncapped fit. Each continuation resets L-BFGS curvature history. Numerical
failure/stagnation terminates explicitly; convergence means approximate KKT
stationarity in the unchanged bounded parameter domain, not a global optimum.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

import count_transport_model as legacy


def projected_gradient(theta, gradient, bounds):
    """Unit-step gradient mapping for the unchanged box-constrained domain."""
    limits = np.asarray(bounds, dtype=float)
    return theta - np.clip(theta - gradient, limits[:, 0], limits[:, 1])


def _chunk(objective, initial, bounds, iterations, evaluations, gtol, ftol):
    """Run one public SciPy call and preserve its last accepted finite iterate."""
    x = np.asarray(initial, dtype=float).copy()
    loss, gradient = objective(x)
    last = {"x": x.copy(), "fun": float(loss), "jac": gradient.copy()}
    calls, accepted = 0, 0
    cache = dict(last)

    def checked(point):
        nonlocal calls, cache
        calls += 1
        value, grad = objective(point)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite objective or gradient")
        cache = {"x": point.copy(), "fun": float(value), "jac": grad.copy()}
        return value, grad

    def remember(point):
        nonlocal last, accepted
        accepted += 1
        if not np.array_equal(point, cache["x"]):
            checked(point)
        last = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in cache.items()}

    try:
        result = minimize(checked, x, method="L-BFGS-B", jac=True, bounds=bounds,
                          callback=remember, options={"maxiter": iterations, "maxfun": evaluations,
                          "ftol": ftol, "gtol": gtol, "maxls": 40})
    except (FloatingPointError, OverflowError) as exc:
        result = OptimizeResult(**last, nit=accepted, nfev=calls, status=2, success=False,
                                message=f"NUMERICAL_FAILURE: {type(exc).__name__}: {exc}")
    result.projected_gradient_inf_norm = float(np.max(np.abs(
        projected_gradient(result.x, result.jac, bounds))))
    result.strongly_converged = bool(result.status in (0, 1)
                                     and np.isfinite(result.fun) and np.isfinite(result.jac).all()
                                     and result.projected_gradient_inf_norm <= gtol)
    return result


def _drive(objective, initial, bounds, chunk_iterations, chunk_evaluations, gtol):
    """A numerical stopping rule with no total iteration/evaluation budget."""
    capped = _chunk(objective, initial, bounds, chunk_iterations, chunk_evaluations, gtol, 1e-9)
    trace = []
    current = capped
    previous_x = np.asarray(initial, dtype=float)
    previous_loss = float(objective(previous_x)[0])
    total_iterations = total_evaluations = 0
    while True:
        total_iterations += int(current.nit)
        total_evaluations += int(current.nfev)
        trace.append({"chunk": len(trace), "iterations": int(current.nit),
                      "objective_evaluations": int(current.nfev), "objective": float(current.fun),
                      "start_objective": previous_loss,
                      "projected_gradient_inf_norm": current.projected_gradient_inf_norm,
                      "optimizer_status": int(current.status), "optimizer_success": bool(current.success),
                      "budget_hit": bool(current.status == 1),
                      "optimizer_message": str(current.message), "ftol": 1e-9 if not trace else 0.})
        if current.strongly_converged:
            reason = "projected_gradient_convergence"
            break
        message = str(current.message)
        if current.status not in (0, 1):
            reason = "line_search_failure" if "LNSRCH" in message or "LINE SEARCH" in message.upper() else "numerical_failure"
            break
        # The capped arm may stop on the old loose function tolerance. Continue
        # it once with ftol=0; a later function-tolerance stop is stagnation.
        if current.status == 0 and len(trace) > 1:
            reason = "floating_point_stagnation"
            break
        changed = np.max(np.abs(current.x - previous_x))
        step_floor = 8 * np.finfo(float).eps * max(1., float(np.max(np.abs(previous_x))))
        loss_floor = np.finfo(float).eps * max(1., abs(previous_loss), abs(float(current.fun)))
        if current.status == 1 and changed <= step_floor and previous_loss - current.fun <= loss_floor:
            reason = "floating_point_stagnation"
            break
        if current.fun > previous_loss + 64 * loss_floor:
            reason = "nonmonotone_objective_failure"
            break
        previous_x, previous_loss = current.x.copy(), float(current.fun)
        current = _chunk(objective, previous_x, bounds, chunk_iterations, chunk_evaluations, gtol, 0.)
    return capped, current, {"stop_reason": reason, "chunks": trace,
        "iterations": total_iterations, "objective_evaluations": total_evaluations,
        "resumed_chunks": len(trace)-1, "total_iteration_limit": None, "total_evaluation_limit": None,
        "budget_limit_resumptions": sum(t["budget_hit"] for t in trace[:-1]),
        "terminal_status": "converged" if current.strongly_converged else "numerical_failure",
        "curvature_history": "Reset at each chunk; parameters continue exactly from the preceding chunk",
        "global_optimum_claim": False}


def _pack(result, data, dimension, seed, bounds, names, initial_loss, gtol,
          iterations, evaluations, stop_reason):
    widths = {a: x.shape[1] for a, x in data.items()}
    p, bits, _, states, w, mu, matrices, intercepts, eta = legacy._decode(
        result.x, widths, dimension, "strict")
    singular = {a: np.linalg.svd(m, compute_uv=False) for a, m in matrices.items()}
    limits = np.asarray(bounds)
    hit = (np.isclose(result.x, limits[:, 0], atol=1e-6, rtol=0)
           | np.isclose(result.x, limits[:, 1], atol=1e-6, rtol=0))
    return {
        "variant": "strict", "dimension": dimension, "seed": int(seed),
        "assay_order": list(data), "feature_counts": widths,
        "parameter_vector": result.x.tolist(), "p": p.tolist(), "mu": mu.tolist(),
        "probabilities": p.tolist(), "source_mean": mu.tolist(),
        "loadings": {a: m.tolist() for a, m in matrices.items()},
        "intercepts": {a: b.tolist() for a, b in intercepts.items()},
        "state_bits": bits.tolist(), "state_sources": states.tolist(), "state_weights": w.tolist(),
        "state_log_rates": {a: e.tolist() for a, e in eta.items()},
        "state_rates": {a: np.exp(e).tolist() for a, e in eta.items()},
        "initial_loss": initial_loss, "loss": float(result.fun), "objective": float(result.fun),
        "converged": result.strongly_converged, "status": int(result.status), "message": str(result.message),
        "optimizer_success": bool(result.success), "stop_reason": stop_reason,
        "iterations": int(iterations), "objective_evaluations": int(evaluations),
        "gradient_inf_norm": float(np.max(np.abs(result.jac))),
        "projected_gradient_inf_norm": result.projected_gradient_inf_norm,
        "convergence_tolerance": gtol, "convergence_definition": "finite objective and unit-step box-gradient mapping infinity norm <= tolerance",
        "bound_hits": [names[i] for i in np.flatnonzero(hit)],
        "singular_values": {a: s.tolist() for a, s in singular.items()},
        "singular_value_ratios": {a: float(s[-1]/s[0]) if s[0]>0 else 0. for a,s in singular.items()},
        "minimum_probability_gap": float(np.min(np.diff(np.sort(p)))) if dimension>1 else None,
        "objective_definition": "mean assay marginal NLL per cell per feature; assays equally weighted",
        "interpretation": "Constrained numerical stationarity is not global optimality or verification of biological assumptions",
    }


def fit_convergence_pair(counts, dimension, seed, chunk_iterations=600,
                         chunk_evaluations=15000, gtol=1e-6):
    """Return capped and uncapped strict fits from the identical initial start.

    ``chunk_iterations`` is the capped-arm budget, normally 600. Neither it nor
    ``chunk_evaluations`` limits total continuation work. Lack of numerical
    progress is reported as failure; no finite termination theorem is claimed.
    """
    if (not isinstance(chunk_iterations, int) or chunk_iterations < 1
            or not isinstance(chunk_evaluations, int) or chunk_evaluations < 1
            or not np.isfinite(gtol) or gtol <= 0):
        raise ValueError("positive integer chunk sizes and a positive finite gradient tolerance required")
    data, initial, bounds, names = legacy._setup(counts, dimension, "strict", seed)
    constants = {a: gammaln(x+1).sum(axis=1) for a,x in data.items()}

    def objective(theta):
        return legacy._objective(theta, data, dimension, "strict", constants)

    with threadpool_limits(limits=1):
        initial_loss = objective(initial)[0]
        cap, final, continuation = _drive(objective, initial, bounds, chunk_iterations, chunk_evaluations, gtol)
    cap_reason = "projected_gradient_convergence" if cap.strongly_converged else (
        "capped_chunk_budget" if cap.status == 1 else "relative_objective_tolerance" if cap.status == 0 else "numerical_failure")
    return {"capped": _pack(cap, data, dimension, seed, bounds, names, initial_loss, gtol,
                            cap.nit, cap.nfev, cap_reason),
            "uncapped": _pack(final, data, dimension, seed, bounds, names, initial_loss, gtol,
                              continuation["iterations"], continuation["objective_evaluations"], continuation["stop_reason"]),
            "continuation": continuation, "capped_iteration_budget": chunk_iterations,
            "chunk_evaluation_budget": chunk_evaluations,
            "parameter_domain": "Unchanged legacy bounds; no new clipping or widening"}
