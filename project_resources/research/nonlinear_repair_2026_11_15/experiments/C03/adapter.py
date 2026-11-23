"""Narrow compatibility and exact r=4 weight QP for the pinned MEITD source."""
import itertools
from contextlib import contextmanager
import numpy as np
import scipy.linalg as la

QP_TOL=1e-8
BOX_TOL=1e-10

def simplex_qp(b,W,lb):
    b=np.asarray(b,float);W=np.asarray(W,float);n=len(b)
    if n!=4 or W.shape!=(n,n) or not np.isfinite(W).all() or not np.isfinite(b).all() or not np.allclose(W,W.T,rtol=1e-10,atol=1e-10) or not 0<=lb<1/n:raise ValueError('invalid QP')
    np.linalg.cholesky(W)
    target=W@b;best=None
    for k in range(1,n+1):
        for free_tuple in itertools.combinations(range(n),k):
            free=np.array(free_tuple,dtype=int);fixed=np.array([i for i in range(n) if i not in free_tuple],dtype=int)
            x=np.full(n,lb);rhs=target[free]-(W[np.ix_(free,fixed)]@x[fixed] if len(fixed) else 0)
            K=np.block([[W[np.ix_(free,free)],np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
            try:sol=np.linalg.solve(K,np.r_[rhs,1-lb*len(fixed)])
            except np.linalg.LinAlgError:continue
            x[free]=sol[:k];grad=W@(x-b);nu=sol[-1];tol=QP_TOL*max(1,float(np.max(np.abs(grad))),float(np.max(np.abs(target))))
            if abs(x.sum()-1)>QP_TOL or np.min(x)<lb-QP_TOL or np.max(np.abs(grad[free]+nu))>tol or (len(fixed) and np.min(grad[fixed]+nu)<-tol):continue
            objective=float((x-b)@W@(x-b))
            if best is None or objective<best[0]:best=(objective,x)
    if best is None:raise ValueError('no QP solution satisfying primal/KKT conditions')
    return best[1]

@contextmanager
def scipy_compatibility():
    original=la.solve
    def solve(*args,**kwargs):
        if 'sym_pos' in kwargs:
            value=kwargs.pop('sym_pos')
            if value:kwargs['assume_a']='pos'
        return original(*args,**kwargs)
    la.solve=solve
    had_alias='int' in np.__dict__;old_alias=np.__dict__.get('int');np.int=int
    rng=np.random.get_state()
    try:yield
    finally:
        la.solve=original
        if had_alias:np.int=old_alias
        else:delattr(np,'int')
        np.random.set_state(rng)

def stable_inverse(meitd,raw):
    lo=meitd.Mlb;hi=meitd.Mub;M=meitd.M
    tol=BOX_TOL*np.maximum(1,np.maximum(np.abs(lo),np.abs(hi)))
    if not np.isfinite(M).all() or np.any(M<lo-tol) or np.any(M>hi+tol):raise ValueError('source standardized means out of box')
    result=raw.min(axis=1,keepdims=True)+(M-lo)/meitd.gamma
    direct=meitd.get_soln()[0]
    if np.any(result<0) or not np.isfinite(result).all() or not np.allclose(result,direct,rtol=1e-9,atol=1e-10):raise ValueError('inverse box or direct discrepancy')
    return result,float(np.max(np.abs(result-direct)))
