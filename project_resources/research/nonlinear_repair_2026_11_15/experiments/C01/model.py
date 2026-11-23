"""C01 count-mixture model and deterministic training-only operations."""
import hashlib
import itertools
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

FLOOR=1e-8
ASSAYS=('RNA','H3K27ac','H3K4me3')

def key_seed(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,args)).encode()).digest()[:4], 'big')

def validate(x, ids, features, expected):
    x=np.asarray(x); ids=np.asarray(ids); features=np.asarray(features)
    if x.ndim!=2 or x.shape[0]<4 or x.shape[1]!=300 or ids.shape!=(len(x),) or features.shape!=(300,): raise ValueError('invalid shapes')
    if not np.issubdtype(x.dtype,np.number) or not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)): raise ValueError('invalid counts')
    if len(set(ids.tolist()))!=len(ids) or len(set(features.tolist()))!=300 or features.tolist()!=expected: raise ValueError('invalid identity or feature order')
    return x.astype(float)

def split(ids):
    order=np.array(sorted(range(len(ids)),key=lambda i: hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()))
    n=int(.75*len(ids)); return order[:n],order[n:]

def permute(x, rows, seed):
    rng=np.random.default_rng(seed); y=x.copy()
    for ix in rows:
        for j in range(x.shape[1]): y[ix,j]=x[rng.permutation(ix),j]
    return y

def log_components(x, rates, kernel, constant=None):
    x=np.asarray(x,float); rates=np.asarray(rates,float)
    if x.ndim!=2 or rates.ndim!=2 or x.shape[1]!=rates.shape[1] or np.any(rates<=0) or not np.isfinite(rates).all(): raise ValueError('invalid rates')
    if kernel=='poisson':
        return x@np.log(rates).T-rates.sum(axis=1)[None,:]+((-gammaln(x+1).sum(axis=1)) if constant is None else constant)[:,None]
    if kernel=='nb2':
        # shape 2: mean mu and variance mu + mu^2 / 2.
        return x@(np.log(rates)-np.log(2+rates)).T + (2*np.log(2/(2+rates))).sum(axis=1)[None,:] + ((np.log(x+1)).sum(axis=1) if constant is None else constant)[:,None]
    raise ValueError('unknown kernel')

def logprob(x,weights,rates,kernel):
    weights=np.asarray(weights,float)
    if weights.ndim!=1 or len(weights)!=len(rates) or np.any(weights<=0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(),1): raise ValueError('invalid weights')
    return logsumexp(log_components(x,rates,kernel)+np.log(weights),axis=1)

def initialize(x,k,seed):
    if k==1:return np.ones(1),np.maximum(x.mean(0,keepdims=True),FLOOR)
    labels=KMeans(n_clusters=k,n_init=1,random_state=seed).fit_predict(np.log1p(x))
    counts=np.bincount(labels,minlength=k)
    if np.any(counts==0): raise ValueError('empty initialization')
    return counts/len(x),np.maximum(np.stack([x[labels==j].mean(0) for j in range(k)]),FLOOR)

def em(x,k,kernel,seed,initial=None):
    try:
        w,r=initialize(x,k,seed) if initial is None else (initial[0].copy(),initial[1].copy())
        trace=[]; converged=False
        constant=-gammaln(x+1).sum(axis=1) if kernel=='poisson' else np.log1p(x).sum(axis=1)
        for iteration in range(301):
            lj=log_components(x,r,kernel,constant)+np.log(w); norm=logsumexp(lj,axis=1); ll=float(norm.mean())
            if not np.isfinite(ll): raise ValueError('nonfinite objective')
            if trace and ll < trace[-1]-1e-8*max(1,abs(trace[-1])): raise ValueError('nonmonotone objective')
            trace.append(ll)
            if iteration and ll-trace[-2]<=1e-7*max(1,abs(trace[-2])):
                converged=True; break
            if iteration==300: break
            resp=np.exp(lj-norm[:,None]); eff=resp.sum(0)
            if np.any(eff<=1e-12): raise ValueError('dead component')
            w=eff/len(x); r=np.maximum((resp.T@x)/eff[:,None],FLOOR)
        lj=log_components(x,r,kernel,constant)+np.log(w); resp=np.exp(lj-logsumexp(lj,axis=1)[:,None])
        return {'converged':converged,'error':None,'weights':w,'rates':r,'effective':resp.sum(0),'trace':trace,'floor_count':int((r==FLOOR).sum()),'iterations':len(trace)-1}
    except Exception as e:
        return {'converged':False,'error':f'{type(e).__name__}: {e}','trace':locals().get('trace',[])}

def valid(m,n):
    return m['converged'] and np.min(m['effective'])>=max(20,.02*n)

def assignment(point,boot):
    dist=np.linalg.norm(np.sqrt(point['rates'])[:,None,:]-np.sqrt(boot['rates'])[None,:,:],axis=2)
    _,idx=linear_sum_assignment(dist**2)
    return idx,dist[np.arange(len(idx)),idx]

def stability(point,boots,n):
    k=len(point['weights']); good=[b for b in boots if valid(b,n)]
    out={'valid_bootstraps':len(good),'point_membership_ok':bool(valid(point,n)),'pass':False}
    if len(good)<26 or not valid(point,n):return out
    dis=[]; weights=[]
    for b in good:
        idx,d=assignment(point,b); b['assignment']=idx.tolist(); b['displacement']=d.tolist(); dis.append(d); weights.append(b['weights'][idx])
    rates=np.sqrt(point['rates']); ds=np.linalg.norm(rates[:,None,:]-rates[None,:,:],axis=2); np.fill_diagonal(ds,np.inf); nearest=ds.min(1)
    displacement=np.quantile(np.asarray(dis),.9,axis=0); eps=float(np.quantile(np.max(np.abs(np.asarray(weights)-point['weights']),axis=1),.9))
    w=point['weights']; separated=all(abs(w[i]-w[j])>2*eps for i in range(k) for j in range(i))
    out.update(nearest=nearest.tolist(),displacement90=displacement.tolist(),epsilon=eps,weight_intervals=[[float(z-eps),float(z+eps)] for z in w],disjoint_weights=separated)
    out['pass']=bool(np.all(nearest>0) and np.all(displacement<nearest/2) and eps<=.10)
    return out

def compatible(points,stabilities):
    k=len(points[0]['weights'])
    if k==1 or not all(s['pass'] for s in stabilities):return []
    intervals=[s['weight_intervals'] for s in stabilities]; result=[]
    for p1 in itertools.permutations(range(k)):
        for p2 in itertools.permutations(range(k)):
            choices=[tuple(range(k)),p1,p2]
            if all(max(intervals[a][choices[a][i]][0] for a in range(3)) <= min(intervals[a][choices[a][i]][1] for a in range(3)) for i in range(k)):
                result.append([list(c) for c in choices])
    return result

def completion(points,perm):
    weights=np.mean([p['weights'][perm[a]] for a,p in enumerate(points)],axis=0); weights/=weights.sum()
    rates=[p['rates'][perm[a]] for a,p in enumerate(points)]
    means=[weights@r for r in rates]
    cov={(a,b): (rates[a]-means[a]).T@((rates[b]-means[b])*weights[:,None]) for a in range(3) for b in range(a+1,3)}
    return weights,rates,cov
