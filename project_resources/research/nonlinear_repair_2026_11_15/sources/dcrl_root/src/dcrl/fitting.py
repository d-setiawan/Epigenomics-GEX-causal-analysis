"""Fit observed data from spectral or random starts and record diagnostics."""

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter
import warnings

import numpy as np
from scipy.special import gammaln,logsumexp,xlogy

from .estimator import DAGEstimator
from .initialization import initialize_parameters,_empirical_latent_probabilities
from .latent_law import SparseLatentLaw
from .orientation import orient_columns
from .utils import TLP


class LikelihoodLimitError(ValueError):
    """Exact scoring would exceed the requested state-enumeration limit."""


def coefficient_bounds(K,distribution,scale=1.0):
    """Raw optimization bounds; the Lognormal variance box is not scaled."""
    if distribution=="Poisson":
        lower=np.zeros(K+1)
        upper=np.concatenate(([2],3*np.ones(K)))
    elif distribution=="Lognormal":
        lower=np.concatenate(([-2],np.zeros(K)))
        upper=4*np.ones(K+1)
    elif distribution=="Bernoulli":
        lower=np.concatenate(([-5],-2.5*np.ones(K)))
        upper=5*np.ones(K+1)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return lower*scale,upper*scale


def _validate_X(X,distribution):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or not X.size or not np.all(np.isfinite(X)):
        raise ValueError("X must be a nonempty finite two-dimensional array")
    if distribution=="Bernoulli" and not np.all((X==0)|(X==1)):
        raise ValueError("Bernoulli X must contain only 0 and 1")
    if distribution=="Poisson" and (np.any(X<0) or np.any(X!=np.floor(X))):
        raise ValueError("Poisson X must contain nonnegative integer counts")
    if distribution=="Lognormal" and np.any(X<=0):
        raise ValueError("Lognormal X must be strictly positive")
    if distribution not in {"Bernoulli","Poisson","Lognormal"}:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return X


def observed_loglikelihood(X,B,p,gamma,distribution,*,max_states=65536,batch_size=256):
    """Exact observed log likelihood, including data-dependent constants.

    With no uniform component only stored positive-mass states are evaluated;
    with a uniform component all 2**K states are required, and a hard explicit
    cap applies. Observation and state batching bounds temporary storage. No
    probability truncation or approximation is used for start selection.
    """
    X=_validate_X(X,distribution)
    B=np.asarray(B,dtype=float)
    if not isinstance(p,SparseLatentLaw):
        raise TypeError("p must be a SparseLatentLaw")
    if B.shape!=(X.shape[1],p.K+1) or not np.all(np.isfinite(B)):
        raise ValueError("B has invalid shape or nonfinite entries")
    if not isinstance(max_states,int) or max_states<1 or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError("max_states and batch_size must be positive integers")
    count=(1<<p.K) if p.uniform_mass else len(p.masses)
    if count>max_states:
        kind="full state space for nonzero uniform mass" if p.uniform_mass else "stored sparse support"
        raise LikelihoodLimitError(f"Exact likelihood requires {count} states ({kind}); limit is {max_states}")
    states=range(count) if p.uniform_mass else sorted(p.masses)
    if distribution=="Lognormal":
        variance=np.asarray(gamma,dtype=float).reshape(-1)
        if variance.shape!=(X.shape[1],) or not np.all(np.isfinite(variance)) or np.any(variance<=0):
            raise ValueError("Lognormal gamma must contain positive finite variances")
        transformed=np.log(X)
    else:
        transformed=X
    row_loglik=np.full(X.shape[0],-np.inf)
    for start in range(0,count,batch_size):
        selected=states[start:start+batch_size]
        A=np.array([[(int(s)>>k)&1 for k in range(p.K-1,-1,-1)] for s in selected],dtype=float)
        means=A@B[:,1:].T+B[:,0]
        if distribution=="Poisson" and np.any(means<0):
            raise ValueError("Poisson fit has negative conditional means on positive-probability states")
        log_weights=np.log([p.probability(s) for s in selected])
        for offset in range(0,len(X),batch_size):
            xb=transformed[offset:offset+batch_size]
            if distribution=="Bernoulli":
                conditional=xb@means.T-np.logaddexp(0,means).sum(axis=1)[None,:]
            else:
                conditional=np.zeros((len(xb),len(selected)))
                for j in range(X.shape[1]):
                    if distribution=="Poisson":
                        conditional+=xlogy(xb[:,j,None],means[None,:,j])-means[None,:,j]-gammaln(xb[:,j,None]+1)
                    else:
                        conditional+=-0.5*((xb[:,j,None]-means[None,:,j])**2/variance[j]+np.log(2*np.pi*variance[j]))-xb[:,j,None]
            chunk=logsumexp(conditional+log_weights[None,:],axis=1)
            row_loglik[offset:offset+len(xb)]=np.logaddexp(row_loglik[offset:offset+len(xb)],chunk)
    return float(np.sum(row_loglik))


def _initial_parameters(X,cfg,spec,known_Q):
    kind=spec["kind"]
    seed=spec.get("init_seed",spec["seed"])
    rng=np.random.RandomState(seed)
    lower,upper=coefficient_bounds(cfg.K,cfg.distribution,cfg.coefficient_bound_scale)
    if kind in {"spectral","spectral_jitter"}:
        if cfg.J<3*cfg.K or min(X.shape)<cfg.K:
            raise ValueError("The original spectral initializer requires J >= 3*K and min(N,J) >= K")
        p,B,gamma,A=initialize_parameters(X,cfg.J,cfg.K,cfg.distribution,cfg.epsilon,
                                               sparse=True,cutoff_c=cfg.init_cutoff_c)
        if kind=="spectral_jitter":
            jitter=float(spec.get("jitter",0.1))
            if not np.isfinite(jitter) or jitter<=0:
                raise ValueError("spectral_jitter requires a finite positive jitter")
            B=np.clip(B+rng.normal(0,jitter,B.shape),lower,upper)
            if cfg.distribution=="Lognormal":
                gamma=np.clip(gamma+rng.normal(0,jitter,gamma.shape),1e-6,2)
    elif kind=="random":
        # Draw independent starts within the coefficient bounds.
        B=rng.uniform(lower,upper,size=(cfg.J,cfg.K+1))
        A=rng.binomial(1,0.5,size=(cfg.N,cfg.K)).astype(float)
        p=_empirical_latent_probabilities(A,cfg.K,sparse=True)
        gamma=rng.uniform(0.5,1.5,size=(cfg.J,1))
    else:
        raise ValueError("start kind must be 'spectral', 'spectral_jitter', or 'random'")
    if known_Q is not None:
        # For the known-Q oracle, zero excluded slopes before the first sweep.
        B=np.clip(B,lower,upper)
        B[:,1:]*=known_Q
        if cfg.distribution=="Lognormal":
            gamma=np.clip(gamma,1e-6,2)
    return p,B,gamma,A


def _make_estimator(X,cfg,initial,known_Q,backend="standard"):
    names=("N","J","K","Q_type","DAG_type","distribution","algorithm","tau","pen",
             "max_iter","tol","C","epsilon","Q_N","kappa","L_final","init_cutoff_c",
             "inner_max_iter","gain_numerator","gain_offset","min_outer_iter","coefficient_bound_scale")
    kwargs={name: getattr(cfg,name) for name in names}
    kwargs.update(asdict(cfg.bounds))
    cls=DAGEstimator
    if backend=="section5_validated":
        from .section5_backend import estimator_class
        cls=estimator_class()
    elif backend!="standard":
        raise ValueError("Unknown fitting backend: "+str(backend))
    return cls(**kwargs,observed_X=X,initial_parameters=initial,known_Q=known_Q)


def fit_observed(X,config,start_specs=None,known_Q=None,orientation="column_sum",backend="standard"):
    """Fit observed X and retain every attempted start, including failures.

    Start dictionaries accept ``kind``, ``seed`` (SAEM seed), ``init_seed``
    (defaults to seed), optional ``id``, and ``jitter`` for spectral_jitter.
    Repeated spectral starts share the deterministic initialization and differ
    only in their stochastic SAEM paths. Each start's arrays are raw fitted
    coordinates. Only the selected top-level B/p can be finally oriented.

    Selection maximizes observed_loglik - pen*TLP(slopes,tau), evaluated on the
    same observed data and with the same penalty for all starts. Known-Q fits
    instead use pen=0 and do not threshold coefficients on the allowed support.
    The result is an attained finite fit, never a global-MLE certificate.
    All failed starts are returned even if no scorable candidate remains.

    ``section5_validated`` selects the frozen analytic-objective/vectorized-Gibbs
    backend used by the three Section 5 validation groups; ``standard`` retains
    the original estimator. Backend selection is local to this call.

    PSAEM uses NumPy's legacy global RNG. Its state is restored on return, but
    concurrent calls in threads are unsupported; use separate processes.
    """
    if backend not in {"standard","section5_validated"}:
        raise ValueError("Unknown fitting backend: "+str(backend))
    cfg=deepcopy(config).finalize()
    X=_validate_X(X,cfg.distribution)
    if X.shape!=(cfg.N,cfg.J):
        raise ValueError("X shape must agree with config.N and config.J")
    if cfg.algorithm!="PSAEM":
        raise ValueError("fit_observed currently supports algorithm='PSAEM'")
    if orientation!="column_sum":
        raise ValueError("All final fits must use column_sum orientation")
    if known_Q is not None:
        known_Q=np.asarray(known_Q)
        if known_Q.shape!=(cfg.J,cfg.K) or not np.all((known_Q==0)|(known_Q==1)):
            raise ValueError("known_Q must be binary with shape (J,K)")
        known_Q=known_Q.astype(bool)
    if start_specs is None:
        start_specs=[{"kind": "spectral","seed": 0}]
    if not start_specs:
        raise ValueError("At least one start is required")
    normalized_specs=[]
    for i,original_spec in enumerate(start_specs):
        spec=dict(original_spec)
        spec.setdefault("kind","spectral")
        spec.setdefault("seed",i)
        spec.setdefault("id",f"{spec['kind']}_{i}")
        if not isinstance(spec["id"],str) or not spec["id"]:
            raise ValueError("Every start id must be a nonempty string")
        normalized_specs.append(spec)
    if len({spec["id"] for spec in normalized_specs})!=len(normalized_specs):
        raise ValueError("Start ids must be unique")
    records=[]
    original_rng=np.random.get_state()
    started=perf_counter()
    try:
        for i,spec in enumerate(normalized_specs):
            record={"id": spec["id"],"spec": spec,"status": "failed","fit_valid": False,
                      "scorable": False,"initialization_kind": spec["kind"],
                      "initialization_deterministic": spec["kind"]=="spectral"}
            records.append(record)
            est=None
            start_time=perf_counter()
            caught=[]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    initial=_initial_parameters(X,cfg,spec,known_Q)
                    p0,B0,gamma0,A0=initial
                    record.update(initial_p=SparseLatentLaw(cfg.K,dict(p0)),initial_B=B0.copy(),
                                  initial_gamma=gamma0.copy(),initial_A=A0.copy())
                    est=_make_estimator(X,cfg,initial,known_Q,backend=backend)
                    np.random.seed(spec["seed"])
                    est.PSAEM(i)
                    record.update(B=est.B_hat.copy(),p=est.p_hat,
                                  gamma=est.gamma_hat.copy(),final_sample_A=est.last_A.copy())
                    finite=np.all(np.isfinite(est.B_hat)) and np.all(np.isfinite(est.gamma_hat))
                    if not finite:
                        raise ValueError("Nonfinite fitted parameters")
                    if cfg.distribution=="Lognormal" and np.any(est.gamma_hat<=0):
                        raise ValueError("Nonpositive fitted Lognormal variance")
                    if known_Q is not None and np.any(est.B_hat[:,1:][~known_Q]!=0):
                        raise AssertionError("Fixed-support fit violated known_Q")
                    record["fit_valid"]=True
                    ll=observed_loglikelihood(X,est.B_hat,est.p_hat,est.gamma_hat,cfg.distribution,
                                               max_states=cfg.likelihood_max_states,batch_size=cfg.likelihood_batch_size)
                    if not np.isfinite(ll):
                        raise ValueError("Nonfinite attained observed likelihood")
                    penalty=float(est.pen*TLP(est.B_hat[:,1:],cfg.tau))
                    record.update(status="ok",scorable=True,observed_loglik=ll,
                                  penalty=penalty,penalized_objective=ll-penalty)
            except Exception as exc:
                record["error"]=f"{type(exc).__name__}: {exc}"
                record["failure_stage"]="scoring" if record["fit_valid"] else "initialization" if est is None else "fitting"
            finally:
                optimizer=[] if est is None else est.optimizer_diagnostics
                record["diagnostics"]={
                    "iterations": [] if est is None else est.iteration_diagnostics,
                    "optimizer": optimizer,
                    "iteration_count": 0 if est is None else len(est.iteration_diagnostics),
                    "optimizer_calls": len(optimizer),
                    "optimizer_failures": sum(not r["success"] for r in optimizer),
                    "stopping_reason": None if est is None else est.stopping_reason,
                    "elapsed_seconds": perf_counter()-start_time,
                    "warnings": [{"category": type(w.message).__name__,"message": str(w.message)} for w in caught],
                }
    finally:
        np.random.set_state(original_rng)
    lower,upper=coefficient_bounds(cfg.K,cfg.distribution,cfg.coefficient_bound_scale)
    output={
        "status": "failed","starts": records,"selected_start": None,
        "config": asdict(cfg),"orientation": {"mode": orientation},
        "fit_kind": "fixed_support_nuisance" if known_Q is not None else "support_estimation",
        "effective_penalty": 0.0 if known_Q is not None else cfg.pen,
        "selection_criterion": "largest attained observed_loglik - penalty*TLP(slopes,tau)",
        "global_optimum_certified": False,
        "raw_bounds": {"coefficient_lower": lower.tolist(),"coefficient_upper": upper.tolist(),
                       "gamma_lower": 0.0 if cfg.distribution=="Lognormal" else None,
                       "gamma_upper": 2.0 if cfg.distribution=="Lognormal" else None,
                       "known_Q": None if known_Q is None else known_Q.astype(int).tolist()},
        "elapsed_seconds": perf_counter()-started,
    }
    eligible=[r for r in records if r["scorable"]]
    if not eligible:
        output["error"]="No finite, exactly scorable candidate; inspect every start record"
        return output
    selected=max(eligible,key=lambda r: r["penalized_objective"])
    output.update(status="ok",selected_start=selected["id"],B=selected["B"].copy(),p=selected["p"],
                  gamma=selected["gamma"].copy(),raw_B=selected["B"].copy(),raw_p=selected["p"],
                  final_sample_A=selected["final_sample_A"].copy(),
                  raw_final_sample_A=selected["final_sample_A"].copy(),
                  initial_B=selected["initial_B"].copy(),initial_p=selected["initial_p"],
                  initial_gamma=selected["initial_gamma"].copy(),initial_A=selected["initial_A"].copy(),
                  observed_loglik=selected["observed_loglik"],penalized_objective=selected["penalized_objective"],
                  penalty=selected["penalty"],diagnostics=selected["diagnostics"])
    if orientation=="column_sum":
        oriented=orient_columns(selected["B"],selected["p"],selected["gamma"])
        output["B"],output["p"]=oriented.pop("B"),oriented.pop("p")
        oriented.pop("gamma")
        output["orientation"].update(oriented)
        flips=oriented["flipped_columns"]
        output["final_sample_A"][:,flips]=1-output["final_sample_A"][:,flips]
    return output
