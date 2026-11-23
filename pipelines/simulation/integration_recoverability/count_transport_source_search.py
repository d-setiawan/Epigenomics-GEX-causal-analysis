"""Exploratory raw-count source search without finite parameter boxes.

The strict model is log R_c = (B+h) A_c.T with shared independent Bernoulli
coordinates. Its usual standardized parameters are M=A sqrt(p(1-p)) and
mu=(h+p)/sqrt(p(1-p)). Only public marginal counts enter initialization,
candidate ranking, and fitting. Finite start/refinement inventories remain;
exhaustive cube labeling of fitted atoms is not exhaustive likelihood search.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from functools import lru_cache

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, gammaln, logit, logsumexp, xlogy
from threadpoolctl import threadpool_limits

import count_transport_model as legacy

SCHEMA = "bernoulli_vertex_lograte_v1: logit_p, shared_h, assay_order A rows"


def _seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"big")


@lru_cache(None)
def cube_bits(dimension):
    return np.asarray(list(itertools.product((0.,1.),repeat=dimension)))


@lru_cache(None)
def cube_orbits(dimension):
    """All cube labelings modulo bit complements and coordinate permutations."""
    if dimension not in (1,2,3):
        raise ValueError("Cube dimension must be 1, 2, or 3")
    k=2**dimension
    neighbors=[2**(dimension-1-j) for j in range(dimension)]
    representatives=[]
    for tail in itertools.permutations(range(1,k)):
        labeling=(0,*tail)
        values=[labeling[j] for j in neighbors]
        if values==sorted(values):
            representatives.append(labeling)
    return np.asarray(representatives,dtype=int)


def _decode(theta,widths,dimension):
    if not np.isfinite(theta).all():
        raise FloatingPointError("Nonfinite unconstrained parameter")
    p=expit(theta[:dimension])
    if (p<=0).any() or (p>=1).any():
        raise FloatingPointError("Bernoulli probability rounded to a numerical endpoint")
    h=theta[dimension:2*dimension]
    bits=cube_bits(dimension)
    logweights=(bits*np.log(p)+(1-bits)*np.log1p(-p)).sum(1)
    cursor=2*dimension;matrices={};eta={}
    for assay,width in widths.items():
        matrices[assay]=theta[cursor:cursor+width*dimension].reshape(width,dimension)
        cursor+=width*dimension
        eta[assay]=(bits+h)@matrices[assay].T
    if cursor!=len(theta):
        raise ValueError("Parameter vector length mismatch")
    return p,h,bits,logweights,matrices,eta


def _objective(theta,data,dimension,constants=None):
    widths={a:x.shape[1] for a,x in data.items()}
    p,h,bits,logweights,matrices,eta=_decode(theta,widths,dimension)
    if constants is None:
        constants={a:gammaln(x+1).sum(1) for a,x in data.items()}
    loss=0.;grad_h=np.zeros(dimension);grad_logw=np.zeros(len(bits));grad_a={}
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        for assay,x in data.items():
            rates=np.exp(eta[assay])
            joint=x@eta[assay].T-rates.sum(1)+logweights-constants[assay][:,None]
            scores=logsumexp(joint,axis=1)
            shifted=joint-joint.max(1,keepdims=True)
            posterior=np.exp(shifted-logsumexp(shifted,axis=1,keepdims=True))
            scale=1./(len(data)*len(x)*x.shape[1]);mass=posterior.sum(0)
            loss-=scores.sum()*scale
            grad_eta=(mass[:,None]*rates-posterior.T@x)*scale
            grad_a[assay]=grad_eta.T@(bits+h)
            grad_h+=(grad_eta@matrices[assay]).sum(0)
            grad_logw-=mass*scale
        gradient=np.concatenate([grad_logw@(bits-p),grad_h,*[grad_a[a].ravel() for a in data]])
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite exact objective or gradient")
    return float(loss),gradient


def physical_legacy_start(data,dimension,seed):
    """Convert the identical old physical initializer; no truth parameters."""
    _,old,_,_=legacy._setup(data,dimension,"strict",seed)
    widths={a:x.shape[1] for a,x in data.items()}
    p,_,_,_,_,mu,matrices,_,_=legacy._decode(old,widths,dimension,"strict")
    scale=np.sqrt(p*(1-p))
    return np.concatenate([logit(p),mu*scale-p,*[(matrices[a]/scale).ravel() for a in data]])


def _optimize(objective,initial,chunk_iterations,chunk_evaluations,gtol):
    """No parameter boxes or total budgets; reject nonfinite trial proposals."""
    initial=np.asarray(initial,dtype=float)
    start_loss,start_gradient=objective(initial)
    current=OptimizeResult(x=initial.copy(),fun=start_loss,jac=start_gradient,nit=0,nfev=0,
                          status=1,success=False,message="initial state")
    trace=[];total_iterations=total_evaluations=total_rejections=0
    while True:
        start_x=current.x.copy();start_value=float(current.fun)
        last={"x":current.x.copy(),"fun":float(current.fun),"jac":current.jac.copy()}
        cache=dict(last);calls=accepted=rejections=0
        def checked(point):
            nonlocal cache,calls,rejections
            calls+=1
            try:
                value,gradient=objective(point)
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite trial")
            except (FloatingPointError,OverflowError):
                rejections+=1
                return np.inf,np.zeros_like(point)
            cache={"x":point.copy(),"fun":float(value),"jac":gradient.copy()}
            return value,gradient
        def remember(point):
            nonlocal last,accepted
            if not np.array_equal(point,cache["x"]):
                value,_=checked(point)
                if not np.isfinite(value):
                    raise FloatingPointError("Optimizer accepted a rejected numerical trial")
            accepted+=1
            last={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in cache.items()}
        try:
            current=minimize(checked,start_x,method="L-BFGS-B",jac=True,callback=remember,
                             options={"maxiter":chunk_iterations,"maxfun":chunk_evaluations,
                                      "ftol":0.,"gtol":gtol,"maxls":40})
            # Never certify a placeholder gradient used to reject a trial.
            # Validate the returned parameter point with the real objective.
            current.fun,current.jac=objective(current.x)
            if not np.isfinite(current.fun) or not np.isfinite(current.jac).all():
                raise FloatingPointError("Nonfinite accepted result")
        except (FloatingPointError,OverflowError) as exc:
            current=OptimizeResult(**last,nit=accepted,nfev=calls,status=2,success=False,
                                   message=f"NUMERICAL_FAILURE: {exc}")
        gradient_norm=float(np.max(np.abs(current.jac)))
        current.strongly_converged=bool(current.status in (0,1) and gradient_norm<=gtol)
        current.projected_gradient_inf_norm=gradient_norm
        total_iterations+=int(current.nit);total_evaluations+=int(current.nfev);total_rejections+=rejections
        trace.append({"chunk":len(trace),"start_objective":start_value,"objective":float(current.fun),
                      "iterations":int(current.nit),"objective_evaluations":int(current.nfev),
                      "gradient_inf_norm":gradient_norm,"optimizer_status":int(current.status),
                      "optimizer_message":str(current.message),"budget_hit":bool(current.status==1),
                      "rejected_nonfinite_trials":rejections,"ftol":0.})
        if current.strongly_converged:
            reason="gradient_convergence";break
        if current.status not in (0,1):
            reason="line_search_failure" if "LNSRCH" in str(current.message) else "numerical_failure";break
        if current.status==0:
            reason="floating_point_stagnation";break
        eps=np.finfo(float).eps
        if (np.max(np.abs(current.x-start_x))<=8*eps*max(1.,float(np.max(np.abs(start_x))))
                and start_value-current.fun<=eps*max(1.,abs(start_value))):
            reason="floating_point_stagnation";break
        if current.fun>start_value+64*eps*max(1.,abs(start_value)):
            reason="nonmonotone_objective_failure";break
    return current,{"stop_reason":reason,"iterations":total_iterations,
        "objective_evaluations":total_evaluations,"chunks":trace,"resumed_chunks":len(trace)-1,
        "budget_limit_resumptions":sum(c["budget_hit"] for c in trace[:-1]),
        "total_iteration_limit":None,"total_evaluation_limit":None,"parameter_bounds":None,
        "rejected_nonfinite_trials":total_rejections,"line_search_trials_per_iteration":40,
        "terminal_status":"converged" if current.strongly_converged else "numerical_failure",
        "curvature_history":"Reset at chunk boundaries; accepted parameters continue unchanged",
        "global_optimum_claim":False}


def poisson_mixture(x,components,seed,gtol=1e-6):
    """Uncapped exact EM from count-space Poisson-deviance row seeds."""
    x=legacy._counts(x);n,width=x.shape
    if components<1 or components>n:
        raise ValueError("Invalid number of mixture components")
    rng=np.random.default_rng(seed);mean=np.maximum(x.mean(0),.1)
    constant=gammaln(x+1).sum(1);deviance_constant=xlogy(x,x).sum(1)-x.sum(1)
    chosen=[];rates=[];distance=np.full(n,np.inf)
    for _ in range(components):
        if not chosen:
            row=int(rng.integers(n))
        else:
            available=np.ones(n,dtype=bool);available[chosen]=False
            probabilities=np.where(available,np.maximum(distance,0),0.)
            row=int(rng.choice(n,p=probabilities/probabilities.sum())) if probabilities.sum()>0 else int(rng.choice(np.flatnonzero(available)))
        chosen.append(row);rate=.8*x[row]+.2*mean;rates.append(rate)
        deviance=deviance_constant-x@np.log(rate)+rate.sum()
        distance=np.minimum(distance,deviance)
    rates=np.asarray(rates);weights=np.ones(components)/components
    iterations=0;trace=[];previous=None;error=None;converged=False
    while True:
        if (not np.isfinite(rates).all() or not np.isfinite(weights).all()
                or np.any(rates<=0) or np.any(weights<=0)):
            error="Zero/dead/nonfinite mixture parameter";break
        joint=x@np.log(rates).T-rates.sum(1)+np.log(weights)-constant[:,None]
        scores=logsumexp(joint,axis=1);loss=-float(scores.mean()/width)
        shifted=joint-joint.max(1,keepdims=True)
        posterior=np.exp(shifted-logsumexp(shifted,axis=1,keepdims=True))
        mass=posterior.sum(0);sufficient=posterior.T@x
        gradient=max(float(np.max(np.abs((mass[:,None]*rates-sufficient)/(n*width)))),
                     float(np.max(np.abs((weights-mass/n)/width))))
        if not np.isfinite(loss) or not np.isfinite(gradient):
            error="Nonfinite EM objective or gradient";break
        if iterations%100==0:
            trace.append({"iteration":iterations,"objective":loss,"gradient_inf_norm":gradient})
        if previous is not None and loss>previous+64*np.finfo(float).eps*max(1.,abs(previous)):
            error="Nonmonotone EM objective";break
        if gradient<=gtol:
            converged=True;break
        if np.any(mass<=0) or np.any(sufficient<=0):
            error="Zero/dead mixture component or zero rate update";break
        new_weights=mass/n;new_rates=sufficient/mass[:,None]
        step=max(float(np.max(np.abs(new_weights-weights))),float(np.max(np.abs(new_rates-rates))))
        if (previous is not None and previous-loss<=np.finfo(float).eps*max(1.,abs(previous))
                and step<=8*np.finfo(float).eps*max(1.,float(np.max(rates)))):
            error="Floating-point EM stagnation";break
        weights,rates=new_weights,new_rates;previous=loss;iterations+=1
    trace.append({"iteration":iterations,"objective":loss,"gradient_inf_norm":gradient})
    return {"seed":int(seed),"initial_row_indices":chosen,"weights":weights.tolist(),"rates":rates.tolist(),
            "objective":loss,"gradient_inf_norm":gradient,"converged":converged,"iterations":iterations,
            "total_iteration_limit":None,"trace_stride":100,"trace":trace,"error":error,
            "initializer":"Poisson-deviance row seeding; centers 0.8*row+0.2*max(feature_mean,0.1); initial floor only"}


def cube_geometry(weights,rates,dimension):
    """Enumerate every cube-label orbit; free affine intercept only here."""
    weights=np.asarray(weights);rates=np.asarray(rates);bits=cube_bits(dimension)
    if rates.shape[0]!=len(bits) or weights.shape!=(len(bits),) or np.any(rates<=0) or np.any(weights<=0):
        raise ValueError("Invalid positive mixture for cube labeling")
    permutations=cube_orbits(dimension);design=np.column_stack((np.ones(len(bits)),bits))
    ordered_rates=rates[permutations];ordered_log=np.log(ordered_rates);ordered_weights=weights[permutations]
    coefficients=np.einsum("ak,mkp->map",np.linalg.pinv(design),ordered_log)
    predicted=np.einsum("ka,map->mkp",design,coefficients)
    probabilities=ordered_weights@bits
    if np.any(probabilities<=0) or np.any(probabilities>=1):
        raise FloatingPointError("Degenerate cube-coordinate probability")
    log_product=np.einsum("kj,mj->mk",bits,np.log(probabilities))+np.einsum("kj,mj->mk",1-bits,np.log1p(-probabilities))
    with np.errstate(over="raise",invalid="raise"):
        rate_kl=(ordered_rates*(ordered_log-predicted)+np.exp(predicted)-ordered_rates).sum(2)
        weight_kl=(ordered_weights*(np.log(ordered_weights)-log_product)).sum(1)
        scores=((ordered_weights*rate_kl).sum(1)+weight_kl)/rates.shape[1]
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite cube projection score")
    best=int(np.argmin(scores));p=probabilities[best].copy()
    intercept=coefficients[best,0].copy();matrix=coefficients[best,1:].T.copy()
    flips=p>.5
    intercept+=matrix[:,flips].sum(1);matrix[:,flips]*=-1;p[flips]=1-p[flips]
    order=np.argsort(p,kind="stable");p=p[order];matrix=matrix[:,order]
    return {"probabilities":p.tolist(),"A":matrix.tolist(),"affine_intercept":intercept.tolist(),
            "score":float(scores[best]),"best_vertex_to_component":permutations[best].tolist(),
            "canonical_flips":np.flatnonzero(flips).tolist(),"canonical_axis_order":order.tolist(),
            "orbit_representatives":len(permutations),"covered_vertex_assignments":math.factorial(len(bits)),
            "cube_automorphisms_per_orbit":2**dimension*math.factorial(dimension),
            "all_orbit_scores":scores.tolist(),
            "score_definition":"(product-weight KL + weighted product-Poisson KL to free affine log-rate projection)/features"}


def joint_initializer(cubes,assay_order,widths,dimension):
    p=np.mean([cubes[a]["probabilities"] for a in assay_order],axis=0)
    matrices={a:np.asarray(cubes[a]["A"]) for a in assay_order}
    design=np.vstack([matrices[a]/np.sqrt(widths[a]) for a in assay_order])
    target=np.concatenate([np.asarray(cubes[a]["affine_intercept"])/np.sqrt(widths[a]) for a in assay_order])
    h=np.linalg.lstsq(design,target,rcond=None)[0]
    return np.concatenate([logit(p),h,*[matrices[a].ravel() for a in assay_order]])


def canonicalize(theta,widths,dimension):
    p,h,_,_,matrices,_=_decode(theta,widths,dimension)
    if np.any(p==.5):
        raise ValueError("Exactly symmetric p=0.5 has no asymmetric canonical representation")
    flips=p>.5;canonical_p=np.where(flips,1-p,p)
    new_h=np.where(flips,-h-1,h);signs=np.where(flips,-1.,1.)
    order=np.argsort(canonical_p,kind="stable")
    result=np.concatenate([np.where(flips,-theta[:dimension],theta[:dimension])[order],new_h[order],
                           *[(matrices[a]*signs)[:,order].ravel() for a in widths]])
    return result,{"flipped_coordinates":np.flatnonzero(flips).tolist(),"axis_order":order.tolist()}


def _pack(theta,result,data,dimension,seed,initial_loss,gtol,continuation):
    widths={a:x.shape[1] for a,x in data.items()}
    p,h,bits,logw,matrices,eta=_decode(theta,widths,dimension)
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        scale=np.sqrt(p*(1-p));mu=(h+p)/scale
        sources=(bits-p)/scale+mu
        loadings={a:A*scale for a,A in matrices.items()}
    singular={a:np.linalg.svd(M,compute_uv=False) for a,M in loadings.items()}
    with np.errstate(over="raise",invalid="raise"):
        rates={a:np.exp(v) for a,v in eta.items()}
    weights=np.exp(logw)
    if np.any(weights<=0) or not np.isfinite(weights).all():
        raise FloatingPointError("Canonical state probabilities underflowed; no clipping")
    if any(np.any(v<=0) or not np.isfinite(v).all() for v in rates.values()):
        raise FloatingPointError("Canonical rates underflowed or overflowed; no clipping")
    return {"variant":"strict","dimension":dimension,"seed":int(seed),"assay_order":list(data),"feature_counts":widths,
        "parameter_vector":theta.tolist(),"parameter_schema":SCHEMA,"vertex_offsets":h.tolist(),
        "vertex_loadings":{a:A.tolist() for a,A in matrices.items()},"probabilities":p.tolist(),"p":p.tolist(),
        "source_mean":mu.tolist(),"mu":mu.tolist(),"loadings":{a:M.tolist() for a,M in loadings.items()},
        "intercepts":{a:np.zeros(widths[a]).tolist() for a in data},"state_bits":bits.tolist(),
        "state_sources":sources.tolist(),"state_weights":weights.tolist(),
        "state_log_rates":{a:v.tolist() for a,v in eta.items()},"state_rates":{a:v.tolist() for a,v in rates.items()},
        "initial_loss":initial_loss,"objective":float(result.fun),"loss":float(result.fun),
        "converged":result.strongly_converged,"status":int(result.status),"message":str(result.message),
        "optimizer_success":bool(result.success),"stop_reason":continuation["stop_reason"],
        "iterations":continuation["iterations"],"objective_evaluations":continuation["objective_evaluations"],
        "gradient_inf_norm":float(np.max(np.abs(result.jac))),"projected_gradient_inf_norm":float(np.max(np.abs(result.jac))),
        "convergence_definition":"Ordinary raw-gradient infinity norm <= tolerance; projected_gradient field is evaluator compatibility only, since no parameter boxes exist",
        "convergence_tolerance":gtol,"bound_hits":[],"statistical_parameter_bounds":None,
        "singular_values":{a:s.tolist() for a,s in singular.items()},
        "singular_value_ratios":{a:float(s[-1]/s[0]) if s[0]>0 else 0. for a,s in singular.items()},
        "minimum_probability_gap":float(np.min(np.diff(np.sort(p)))) if dimension>1 else None,
        "objective_definition":"mean assay marginal NLL per cell per feature; assays equally weighted",
        "interpretation":"Unbounded parameterization; numerical stationarity and finite source search do not certify a global optimum"}


def fit_source_search(counts,dimension,seed,mode="source_search",mixture_starts=4,refine_candidates=4,
                      chunk_iterations=600,chunk_evaluations=15000,gtol=1e-6):
    if mode not in ("source_search","plain_unbounded"):
        raise ValueError("Unknown source-search mode")
    if any(not isinstance(v,int) or v<1 for v in (mixture_starts,refine_candidates,chunk_iterations,chunk_evaluations)) or not np.isfinite(gtol) or gtol<=0:
        raise ValueError("Positive inventory/chunk sizes and finite positive tolerance required")
    data,_,_,_=legacy._setup(counts,dimension,"strict",seed)
    widths={a:x.shape[1] for a,x in data.items()};assays=list(data)
    constants={a:gammaln(x+1).sum(1) for a,x in data.items()}
    def objective(theta):
        return _objective(theta,data,dimension,constants)
    diagnostics={"mode":mode,"mixtures":{},"joint_initializers":[],
        "finite_search_inventory":{"mixture_starts_per_assay":mixture_starts if mode=="source_search" else 0,
            "refinement_candidates":refine_candidates if mode=="source_search" else 1},
        "initialization_information":"Public raw counts only; no PCA, pairing, truth, or calibration",
        "axis_matching":"Sort inferred asymmetric Bernoulli probabilities within each assay",
        "statistical_parameter_bounds":None,"em_total_iteration_limit":None,
        "joint_total_iteration_limit":None,"joint_total_evaluation_limit":None,
        "line_search_trials_per_iteration":40,"no_global_search_completeness_claim":True}
    starts=[];candidates=[]
    with threadpool_limits(limits=1):
        if mode=="plain_unbounded":
            starts=[("legacy_physical",physical_legacy_start(data,dimension,seed))]
        else:
            available={}
            for assay,x in data.items():
                records=[];available[assay]=[]
                for index in range(mixture_starts):
                    entry={"mixture_start":index}
                    try:
                        mixture=poisson_mixture(x,2**dimension,_seed("count-source-search",seed,assay,index),gtol)
                        entry["mixture"]=mixture
                        if not mixture["converged"]:
                            raise ValueError(mixture["error"] or "Mixture did not converge")
                        cube=cube_geometry(mixture["weights"],mixture["rates"],dimension)
                        entry["cube"]=cube;entry["error"]=None;available[assay].append((index,cube))
                    except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
                        entry["error"]=f"{type(exc).__name__}: {exc}"
                    records.append(entry)
                diagnostics["mixtures"][assay]=records
            if all(available.values()):
                for combination in itertools.product(*[available[a] for a in assays]):
                    identity=tuple(item[0] for item in combination);record={"mixture_start_indices":list(identity)}
                    try:
                        theta=joint_initializer({a:item[1] for a,item in zip(assays,combination)},assays,widths,dimension)
                        value,_=objective(theta);record.update(objective=value,error=None)
                        starts.append((identity,theta))
                    except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
                        record.update(objective=None,error=f"{type(exc).__name__}: {exc}")
                    diagnostics["joint_initializers"].append(record)
                losses={tuple(r["mixture_start_indices"]):r["objective"] for r in diagnostics["joint_initializers"] if r["objective"] is not None}
                starts=sorted(starts,key=lambda item:(losses[item[0]],item[0]))[:refine_candidates]
        for identity,theta in starts:
            candidate={"initializer_id":list(identity) if isinstance(identity,tuple) else identity,
                       "initial_parameter_vector":theta.tolist(),"model":None,"error":None}
            try:
                initial_loss=objective(theta)[0]
                result,continuation=_optimize(objective,theta,chunk_iterations,chunk_evaluations,gtol)
                candidate.update(raw_optimizer={"parameter_vector":result.x.tolist(),"parameter_schema":SCHEMA,
                    "gradient":result.jac.tolist(),"objective":float(result.fun),"converged":result.strongly_converged},
                    continuation=continuation)
                canonical,labels=canonicalize(result.x,widths,dimension)
                candidate["canonicalization"]=labels
                candidate["model"]=_pack(canonical,result,data,dimension,seed,initial_loss,gtol,continuation)
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
                candidate["error"]=f"{type(exc).__name__}: {exc}"
            candidates.append(candidate)
    finite=[i for i,c in enumerate(candidates) if c["model"] is not None and np.isfinite(c["model"]["objective"])]
    converged=[i for i in finite if candidates[i]["model"]["converged"]]
    pool=converged or finite
    selected=min(pool,key=lambda i:(candidates[i]["model"]["objective"],i)) if pool else None
    diagnostics["selected_candidate_index"]=selected
    diagnostics["internal_selection"]="Smallest training NLL among gradient-converged finite refinements, otherwise finite diagnostic; no calibration used"
    if selected is None:
        return {"model":None,"candidates":candidates,"search_diagnostics":diagnostics,"continuation":None,
                "error":"No finite refined model; all attempted mixture/refinement diagnostics retained"}
    chosen=candidates[selected]
    return {"model":chosen["model"],"candidates":candidates,"search_diagnostics":diagnostics,
            "continuation":chosen["continuation"],"error":None}
