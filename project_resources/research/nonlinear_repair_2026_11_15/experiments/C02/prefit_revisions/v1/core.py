"""C02 pure count and design checks; deterministic fixtures are not data."""
import hashlib,itertools
import numpy as np
from scipy.special import gammaln,logsumexp,xlogy

def seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'big')
def split(ids):
    ids=np.asarray(ids)
    if ids.ndim!=1 or len(set(ids.tolist()))!=len(ids):raise ValueError('bad IDs')
    order=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()))
    n=int(.75*len(ids));return order[:n],order[n:]
def control(x,groups,key):
    y=x.copy();rng=np.random.default_rng(key)
    for ids in groups:
        for j in range(x.shape[1]):y[ids,j]=x[rng.permutation(ids),j]
    return y

def q_certificate(q):
    q=np.asarray(q)
    if q.ndim!=2 or q.shape[1]!=2 or not np.all((q==0)|(q==1)):raise ValueError('invalid Q')
    types=[(0,0),(1,0),(0,1),(1,1)]; count={t:int(np.all(q==t,axis=1).sum()) for t in types}
    noncontainment=count[1,0]>0 and count[0,1]>0
    for slots in itertools.product(((1,0),(1,1)),((0,1),(1,1)),((1,0),(1,1)),((0,1),(1,1))):
        used={t:slots.count(t) for t in types}
        if any(used[t]>count[t] for t in types):continue
        remaining={t:count[t]-used[t] for t in types}
        cover=(remaining[1,0]+remaining[1,1]>0 and remaining[0,1]+remaining[1,1]>0)
        if cover and noncontainment:
            indices=[]
            for t in slots:indices.append(next(i for i,row in enumerate(q) if tuple(row)==t and i not in indices))
            return {'pass':True,'block_rows':[indices[:2],indices[2:]],'remaining_coverage':True,'column_noncontainment':True,'row_type_counts':{''.join(map(str,t)):count[t] for t in types}}
    return {'pass':False,'block_rows':None,'remaining_coverage':False,'column_noncontainment':noncontainment,'row_type_counts':{''.join(map(str,t)):count[t] for t in types}}

def exact_logprob(x,B,p):
    x=np.asarray(x,float);B=np.asarray(B,float);p=np.asarray(p,float)
    if x.ndim!=2 or B.shape!=(x.shape[1],3) or p.shape!=(4,) or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any() or not np.isfinite(B).all() or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('invalid count parameters')
    z=np.array([[0,0],[0,1],[1,0],[1,1]],float); rate=B[:,0]+z@B[:,1:].T
    if (rate<0).any():raise ValueError('negative rate')
    components=np.sum(xlogy(x[:,None,:],rate[None,:,:])-rate[None,:,:]-gammaln(x[:,None,:]+1),axis=2)
    lp=np.full(4,-np.inf);lp[p>0]=np.log(p[p>0]); return logsumexp(components+lp,axis=1)

def moments(B,p):
    z=np.array([[0,0],[0,1],[1,0],[1,1]],float);r=B[:,0]+z@B[:,1:].T;mean=p@r;dc=r-mean
    cov=dc.T@(p[:,None]*dc)+np.diag(mean);zero=p@np.exp(-r)
    return mean,cov,zero


def select_start(records):
    eligible=[r for r in records if r.get('status')=='ok' and np.isfinite(r.get('penalized_objective',np.nan))]
    return max(eligible,key=lambda r:r['penalized_objective']) if eligible else None
