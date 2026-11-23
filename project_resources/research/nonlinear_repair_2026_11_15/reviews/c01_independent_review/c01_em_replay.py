import sys,json
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from sklearn.cluster import KMeans
sys.path.insert(0,sys.argv[1]);import train,model
H=Path(sys.argv[1]); M=json.loads((train.DATA/'manifest.json').read_text())
def em(x,k,kernel,seed,initial=None):
 if initial is None:
  if k==1:w=np.ones(1);r=np.maximum(x.mean(0,keepdims=True),1e-8)
  else:
   labs=KMeans(n_clusters=k,n_init=1,random_state=seed).fit_predict(np.log1p(x));counts=np.bincount(labs,minlength=k);w=counts/len(x);r=np.maximum(np.array([x[labs==j].mean(0) for j in range(k)]),1e-8)
 else:w=initial[0].copy();r=initial[1].copy()
 trace=[]
 for it in range(301):
  if kernel=='poisson': lp=x@np.log(r).T-r.sum(1)-gammaln(x+1).sum(1)[:,None]
  else:lp=x@(np.log(r)-np.log(2+r)).T+(2*np.log(2/(2+r))).sum(1)+np.log1p(x).sum(1)[:,None]
  lp+=np.log(w);s=logsumexp(lp,axis=1);ll=s.mean();trace.append(ll)
  if it and ll-trace[-2]<=1e-7*max(1,abs(trace[-2])):break
  if it==300:break
  q=np.exp(lp-s[:,None]);n=q.sum(0);w=n/len(x);r=np.maximum(q.T@x/n[:,None],1e-8)
 return w,r,trace
for filename in ['linear_chain_0__30__poisson__original.json','scmtr_seq_s1__30__nb2__original.json','constrained_collider_0__30__poisson__column_null.json']:
 R=json.loads((H/'fits'/filename).read_text());d=R['dataset']; width=R['panel'];kernel=R['kernel'];control=R['control'];k=R['selected_K'];a='RNA'
 X,fi,*_=train.read_training(d,a,control,M);x=X[fi,:width];record=R['fits'][a][str(k)];idx=record['best_restart'];m=record['restarts'][idx];w,r,t=em(x,k,kernel,idx)
 np.testing.assert_allclose(w,m['weights'],rtol=2e-9,atol=2e-9);np.testing.assert_allclose(r,m['rates'],rtol=2e-9,atol=2e-9);np.testing.assert_allclose(t,m['trace'],rtol=2e-9,atol=2e-9)
 for rep in [0,1]:
  rng=np.random.default_rng(model.key_seed('bootstrap',d,width,kernel,control,a,rep));xx=X[rng.choice(fi,len(fi),replace=True),:width];stored=R['bootstrap'][a][rep]['candidates'][0];wb,rb,tb=em(xx,k,kernel,0,(w,r));np.testing.assert_allclose(wb,stored['weights'],rtol=2e-8,atol=2e-8);np.testing.assert_allclose(rb,stored['rates'],rtol=2e-8,atol=2e-8);np.testing.assert_allclose(tb,stored['trace'],rtol=2e-8,atol=2e-8)
 print('PASS',filename,'RNA best restart',idx,'and bootstrap 0,1 warm starts')
