"""Count mixture likelihood, split and moments from observed data only."""
import hashlib
import numpy as np
from scipy.special import xlogy,gammaln,logsumexp

def seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'big')
def split(ids):
    if len(set(ids.tolist()))!=len(ids):raise ValueError('duplicate IDs')
    order=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()));n=int(.75*len(ids));return order[:n],order[n:]
def control(x,groups,key):
    y=x.copy();rng=np.random.default_rng(key)
    for idx in groups:
        for j in range(x.shape[1]):y[idx,j]=x[rng.permutation(idx),j]
    return y
def score_rows(x,w,r):
    x=np.asarray(x,float);w=np.asarray(w,float);r=np.asarray(r,float)
    if x.ndim!=2 or r.ndim!=2 or r.shape[1]!=x.shape[1] or w.shape!=(len(r),) or not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)) or not np.isfinite(w).all() or np.any(w<=0) or not np.isclose(w.sum(),1,atol=1e-8,rtol=0) or not np.isfinite(r).all() or np.any(r<0):raise ValueError('invalid count mixture')
    lp=np.sum(xlogy(x[:,None,:],r[None,:,:])-r[None,:,:]-gammaln(x[:,None,:]+1),axis=2)+np.log(w)
    return logsumexp(lp,axis=1),lp

def score_summary(values):
    values=np.asarray(values,float)
    if values.ndim!=1 or values.size==0 or np.isnan(values).any() or np.isposinf(values).any():raise ValueError('invalid score')
    zeros=int(np.isneginf(values).sum());return {'log_score':None if zeros else float(values.mean()),'log_score_status':'negative_infinity' if zeros else 'finite','zero_probability_rows':zeros,'rows':len(values)}

def moments(w,r):
    mean=w@r;d=r-mean;return mean,d.T@(w[:,None]*d)+np.diag(mean),w@np.exp(-r)


def active_status(n):
    return 'theorem_dimension_abstention' if n<9 else 'source_design_unsupported_n9' if n==9 else 'eligible'
