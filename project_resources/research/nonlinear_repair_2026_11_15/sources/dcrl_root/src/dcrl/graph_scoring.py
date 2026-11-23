"""BDeu/GES using resamples or fractional counts ``effective_n*p_hat``."""
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from importlib.metadata import version
import math
from numbers import Integral
import time
from types import FunctionType

import numpy as np
from scipy.special import gammaln

from .latent_law import SparseLatentLaw


def _dimension(K):
    if not isinstance(K,Integral) or isinstance(K,bool) or K<1:
        raise ValueError("K must be a positive integer")
    return int(K)


def _hyperparameters(K,ess,structure_prior):
    ess=1.0 if ess is None else float(ess)
    if not math.isfinite(ess) or ess<=0:
        raise ValueError("ess must be finite and positive")
    # The library's default 1/(K-1) prior is undefined for K=1 and
    # degenerates for K=2. Use a neutral edge prior for that small case.
    prior=(0.0 if K==1 else .5 if K==2 else 1.0) if structure_prior is None else float(structure_prior)
    if not math.isfinite(prior) or (K==1 and prior!=0) or (K>1 and not 0<prior<K-1):
        raise ValueError("structure_prior must be 0 for K=1, otherwise strictly between 0 and K-1")
    return ess,prior


def _effective_size(effective_n,*,integer=False):
    if isinstance(effective_n,bool):
        raise ValueError("effective_n must be positive")
    value=float(effective_n)
    if not math.isfinite(value) or value<=0 or (integer and value!=int(value)):
        raise ValueError("effective_n must be finite and positive (an integer for resampling)")
    return int(value) if integer else value


def _law(p_hat,K):
    if isinstance(p_hat,SparseLatentLaw):
        if p_hat.K!=K:
            raise ValueError("K differs from the latent law")
        return p_hat
    if isinstance(p_hat,Mapping):
        if not p_hat or math.fsum(float(v) for v in p_hat.values())<=0:
            raise ValueError("Latent weights must have positive total mass")
        return SparseLatentLaw.from_counts(K,p_hat)
    p=np.asarray(p_hat,dtype=float).reshape(-1)
    if p.size!=1<<K or not np.all(np.isfinite(p)) or np.any(p<0) or p.sum()<=0:
        raise ValueError("Require finite nonnegative probabilities for all 2**K states")
    return SparseLatentLaw.from_counts(K,{int(state): float(p[state]) for state in np.flatnonzero(p)})


@dataclass
class _ExpectedCounts:
    law: SparseLatentLaw
    effective_n: float

    def __post_init__(self):
        self.K=self.law.K
        # Match causal-learn's cardinalities on positive-mass support.
        # A positive uniform component analytically supplies both levels.
        self.cardinalities=[]
        for node in range(self.K):
            levels={int(state>>(self.K-1-node))&1 for state in self.law.masses}
            self.cardinalities.append(2 if self.law.uniform_mass else len(levels))

    def local_score(self,node,parents,ess,structure_prior):
        if not isinstance(node,Integral) or not 0<=node<self.K:
            raise ValueError("node index is outside 0..K-1")
        parents=tuple(sorted(parents))
        if len(set(parents))!=len(parents) or node in parents or any(
            not isinstance(p,Integral) or not 0<=p<self.K for p in parents
        ):
            raise ValueError("parents must be distinct valid indices excluding node")
        varying=[p for p in parents if self.cardinalities[p]==2]
        q=1<<len(varying)
        r=self.cardinalities[node]
        alpha_parent=math.ldexp(ess,-len(varying))
        alpha_cell=alpha_parent/r
        if alpha_cell==0:
            raise ValueError("Parent set is too large for floating-point BDeu hyperparameters")
        # Only occupied parent configurations are stored; no 2**K or 2**|Pa|
        # array is constructed, including when the law has uniform mass.
        rows={}
        for state,mass in self.law.masses.items():
            code=0
            for parent in varying:
                code=(code<<1)|((state>>(self.K-1-parent))&1)
            row=rows.setdefault(code,[0.0,0.0])
            row[(state>>(self.K-1-node))&1]+=self.effective_n*mass
        uniform_cell=math.ldexp(self.effective_n*self.law.uniform_mass,-len(varying)-1)

        def contribution(counts):
            counts=np.asarray(counts,dtype=float)
            return float(gammaln(alpha_parent)-gammaln(counts.sum()+alpha_parent)
                         +np.sum(gammaln(counts+alpha_cell)-gammaln(alpha_cell)))

        value=0.0
        if self.K>1:
            probability=structure_prior/(self.K-1)
            value=len(parents)*math.log(probability)+(self.K-1-len(parents))*math.log1p(-probability)
        for counts in rows.values():
            value+=contribution(np.asarray(counts)+uniform_cell)
        if uniform_cell:
            value+=(q-len(rows))*contribution([uniform_cell,uniform_cell])
        return float(value)


def expected_count_local_bdeu(p_hat,K,effective_n,node,parents,ess=None,structure_prior=None):
    """Local BDeu score at fractional counts, using the pinned-library prior.

    Mappings are nonnegative encoded-state weights and are normalized.
    SparseLatentLaw uniform mass is marginalized analytically. Empty cells
    contribute zero and need not be materialized. Larger scores are better.
    """
    K=_dimension(K)
    ess,prior=_hyperparameters(K,ess,structure_prior)
    counts=_ExpectedCounts(_law(p_hat,K),_effective_size(effective_n))
    return counts.local_score(node,parents,ess,prior)


class _VariableDimensions:
    """Shape metadata consumed by GES; intentionally contains no observations."""
    def __init__(self,K):
        self.shape=(K,K)


def _isolated_ges(data,score,K,max_parents=None):
    module=importlib.import_module("causallearn.search.ScoreBased.GES")
    if max_parents is not None and (not isinstance(max_parents,Integral) or not 0<=max_parents<K):
        raise ValueError("max_parents must be an integer between 0 and K-1")

    def local_score(data,node,parents,parameters=None):
        if max_parents is not None and len(parents)>max_parents:
            return -math.inf
        return score(node,parents)

    # GES currently ignores supplied BDeu parameters. Bind the local-score
    # callback in a private globals dictionary instead of monkeypatching the
    # imported module (safe for concurrent calls with different priors).
    namespace=dict(module.ges.__globals__)
    namespace["local_score_BDeu"]=local_score
    isolated=FunctionType(module.ges.__code__,namespace,module.ges.__name__,
                            module.ges.__defaults__,module.ges.__closure__)
    isolated.__kwdefaults__=module.ges.__kwdefaults__
    return isolated(data,score_func="local_score_BDeu",maxP=max_parents)


def _encode_counts(samples):
    counts=Counter()
    for row in samples:
        state=0
        for bit in row:
            state=(state<<1)|int(bit)
        counts[state]+=1
    return dict(counts)


def _sample(p_hat,K,n,seed):
    # Seed=None preserves the legacy global RNG path exactly.
    if seed is None:
        from .evaluation import sample_latent_states_from_p
        return sample_latent_states_from_p(p_hat,K,n)
    rng=np.random.RandomState(seed)
    if isinstance(p_hat,(SparseLatentLaw,Mapping)):
        law=_law(p_hat,K)
        states=list(law.masses)
        weights=[law.masses[state] for state in states]
        if law.uniform_mass:
            weights.append(law.uniform_mass)
        weights=np.asarray(weights,float)
        weights/=weights.sum()
        counts=rng.multinomial(n,weights)
        samples=np.empty((n,K),dtype=float)
        start=0
        for state,count in zip(states,counts):
            if count:
                samples[start:start+count]=[(state>>shift)&1 for shift in range(K-1,-1,-1)]
                start+=count
        if law.uniform_mass and counts[-1]:
            samples[start:]=rng.randint(0,2,size=(int(counts[-1]),K))
        return samples
    p=np.asarray(p_hat).reshape(-1)
    counts=rng.multinomial(n,p)
    samples=np.empty((n,K),dtype=float)
    start=0
    for state in np.flatnonzero(counts):
        count=int(counts[state])
        samples[start:start+count]=[(int(state)>>shift)&1 for shift in range(K-1,-1,-1)]
        start+=count
    return samples


def learn_graph_from_samples(samples,ess=None,structure_prior=None,max_parents=None):
    """Run BDeu/GES on an actual binary sample matrix.

    ``ess`` is the Dirichlet equivalent sample size, distinct from the number
    of rows. The returned graph uses causal-learn endpoint encoding.
    """
    started=time.perf_counter()
    samples=np.asarray(samples)
    if samples.ndim!=2 or not len(samples) or not np.all(np.isin(samples,[0,1])):
        raise ValueError("samples must be a nonempty binary matrix")
    K=_dimension(samples.shape[1])
    resolved_ess,prior=_hyperparameters(K,ess,structure_prior)
    module=importlib.import_module("causallearn.search.ScoreBased.GES")
    if ess is None and structure_prior is None and max_parents is None and K>=3:
        record=module.ges(samples,score_func="local_score_BDeu")
    elif K>=2:
        # Explicitly forward ESS and structure prior despite GES's discarded
        # parameters. Cardinalities match the same pinned-library convention.
        parameters=dict(sample_prior=resolved_ess,structure_prior=prior,
                          r_i_map={node: len(np.unique(samples[:,node])) for node in range(K)})
        original_score=module.local_score_BDeu
        record=_isolated_ges(samples,lambda node,parents: original_score(samples,node,parents,parameters),K,max_parents)
    else:
        counts=_ExpectedCounts(_law(_encode_counts(samples),K),float(len(samples)))
        record=_isolated_ges(samples,lambda node,parents: counts.local_score(node,parents,resolved_ess,prior),K,max_parents)
    return dict(graph=record["G"].graph.copy(),score=float(record["score"]),
                method="empirical",K=K,effective_n=len(samples),ess=resolved_ess,
                structure_prior=prior,sample_counts=_encode_counts(samples),
                seed=None,max_parents=max_parents,
                provenance=_provenance("observed integer contingency counts","levels observed in input samples"),
                elapsed_seconds=time.perf_counter()-started)


def _provenance(counts,cardinalities="levels in positive-mass support of the latent law"):
    return dict(score="BDeu",search="GES",causal_learn_version=version("causal-learn"),
                counts=counts,cardinalities=cardinalities,
                structure_prior_convention="expected parents; per-edge prior = structure_prior/(K-1)",
                small_K_default="K=1: no-edge prior; K=2: edge probability 0.5")


def learn_latent_graph(p_hat,K,effective_n,method="resampling",ess=None,
                       structure_prior=None,seed=None,max_parents=None):
    """Return graph, BDeu score, counts and reproducibility metadata.

    ``resampling`` retains the public multinomial sampling path, including
    dense zero categories and sparse insertion order. ``seed=None`` consumes
    the global NumPy RNG as before; a seed uses an isolated RandomState.
    ``plugin`` evaluates fractional counts analytically and consumes no RNG.
    A plugin seed is rejected to avoid implying that it affects the result.

    ``effective_n`` scales contingency counts; ``ess`` sets the BDeu
    Dirichlet prior and defaults to one. Mappings and plugin dense inputs
    are normalized; dense resampling inputs must already sum to one.
    The default structure prior is causal-learn's one expected parent for
    K>=3, neutral edge probability 0.5 for K=2, and no edges for K=1.
    """
    started=time.perf_counter()
    K=_dimension(K)
    if method not in {"resampling","plugin"}:
        raise ValueError("method must be 'resampling' or 'plugin'")
    if method=="plugin" and seed is not None:
        raise ValueError("The deterministic plugin method does not accept a random seed")
    n=_effective_size(effective_n,integer=method=="resampling")
    resolved_ess,prior=_hyperparameters(K,ess,structure_prior)
    law=_law(p_hat,K)
    if method=="resampling":
        # Validate normalized dense inputs without renormalizing and changing
        # their historical fixed-seed realization.
        if not isinstance(p_hat,(SparseLatentLaw,Mapping)) and not np.isclose(np.asarray(p_hat).sum(),1.,rtol=0,atol=1e-12):
            raise ValueError("Dense resampling probabilities must sum to one")
        sample=_sample(p_hat,K,n,seed)
        result=learn_graph_from_samples(sample,ess,structure_prior,max_parents)
        result.update(method=method,seed=seed,effective_n=n)
        result["provenance"]["counts"]="multinomial integer counts"
        result["provenance"]["cardinalities"]="levels observed in the resampled data"
    else:
        counts=_ExpectedCounts(law,n)
        record=_isolated_ges(_VariableDimensions(K),
                               lambda node,parents: counts.local_score(node,parents,resolved_ess,prior),K,max_parents)
        result=dict(graph=record["G"].graph.copy(),score=float(record["score"]),
                      method=method,K=K,effective_n=n,ess=resolved_ess,structure_prior=prior,
                      seed=None,max_parents=max_parents,sample_counts=None,
                      sufficient_statistics=dict(stored_state_expected_counts={state: n*mass for state,mass in law.masses.items()},
                                                 uniform_component_total=n*law.uniform_mass),
                      provenance=_provenance("fractional expected contingency counts; no rounding or sampling"))
    result["elapsed_seconds"]=time.perf_counter()-started
    return result
