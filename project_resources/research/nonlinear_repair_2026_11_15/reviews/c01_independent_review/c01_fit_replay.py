import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,gammaln
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,sys.argv[1]); import train,model
H=Path(sys.argv[1]); seal=json.loads((H/'fit_seal.json').read_text()); manifest=json.loads((train.DATA/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(seal['artifacts'])==24
assert sha(H/'protocol.json')==seal['protocol_sha256']
assert sha(H/'source_approval.json')==seal['source_approval_sha256']
for f,h in seal['artifacts'].items(): assert sha(H/'fits'/f)==h
counts={}; selected={}; gate_checks=0; calibration_checks=0
for filename in seal['artifacts']:
 r=json.loads((H/'fits'/filename).read_text()); dataset=r['dataset']; panel=r['panel']; kernel=r['kernel']; control=r['control']; A=model.ASSAYS
 assert filename=='__'.join(map(str,[dataset,panel,kernel,control]))+'.json'
 scores={}; point=[]
 for a in A:
  x,fi,ca,*_=train.read_training(dataset,a,control,manifest); x=x[:,:panel]
  for k in range(1,5):
   f=r['fits'][a][str(k)]; runs=f['restarts']; ok=[i for i,m in enumerate(runs) if m['converged']]
   best=max(ok,key=lambda i:runs[i]['trace'][-1]) if ok else None
   assert f['best_restart']==best
   if best is not None:
    m=runs[best]; W=np.asarray(m['weights']); R=np.asarray(m['rates']); X=x[ca]
    if kernel=='poisson': lp=X@np.log(R).T-R.sum(1)-gammaln(X+1).sum(1)[:,None]
    else: lp=X@(np.log(R)-np.log(2+R)).T+(2*np.log(2/(2+R))).sum(1)+np.log1p(X).sum(1)[:,None]
    score=logsumexp(lp+np.log(W),axis=1).mean(); np.testing.assert_allclose(score,f['calibration_score'],rtol=1e-11,atol=1e-11); calibration_checks+=1
   for m in runs:
    trace=np.asarray(m['trace']); assert all(np.diff(trace)>=-1e-8*np.maximum(1,np.abs(trace[:-1])))
  scores[a]={k:r['fits'][a][str(k)]['calibration_score'] for k in range(1,5)}
 common={k:sum(scores[a][k] for a in A) if all(scores[a][k] is not None for a in A) else None for k in range(1,5)}
 assert all((v is None and r['scores'][str(k)] is None) or np.isclose(v,r['scores'][str(k)]) for k,v in common.items())
 def choose(d):
  finite={k:v for k,v in d.items() if v is not None}
  return min((k for k,v in finite.items() if v>=max(finite.values())-1e-8),default=None)
 assert r['selected_K']==choose(common)
 for a in A: assert r['preferred_K'][a]==choose(scores[a])
 K=r['selected_K']; selected[str(K)]=selected.get(str(K),0)+1
 if K and K>1:
  st=[]
  for a in A:
   p=r['fits'][a][str(K)]; p=p['restarts'][p['best_restart']]; n=r['assays'][a]['fit_rows']; pw=np.array(p['weights']); pr=np.array(p['rates']); dis=[]; ws=[]
   valid=lambda m:m['converged'] and min(m['effective'])>=max(20,.02*n)
   for boot in r['bootstrap'][a]:
    cand=boot['candidates']; good=[i for i,m in enumerate(cand) if m['converged']]; b=max(good,key=lambda i:cand[i]['trace'][-1]) if good else None
    assert boot['best']==b
    if b is not None and valid(cand[b]):
     bm=cand[b]; br=np.array(bm['rates']); bw=np.array(bm['weights']); D=np.linalg.norm(np.sqrt(pr)[:,None,:]-np.sqrt(br)[None,:,:],axis=2); ii,jj=linear_sum_assignment(D**2); assert np.array_equal(ii,np.arange(K)); dis.append(D[ii,jj]); ws.append(bw[jj])
   stored=r['stability'][a]; assert len(dis)==stored['valid_bootstraps']; assert valid(p)==stored['point_membership_ok']
   if len(dis)>=26 and valid(p):
    D=np.linalg.norm(np.sqrt(pr)[:,None,:]-np.sqrt(pr)[None,:,:],axis=2); np.fill_diagonal(D,np.inf); near=D.min(1); disp=np.quantile(dis,.9,axis=0); eps=float(np.quantile(np.max(np.abs(np.array(ws)-pw),axis=1),.9)); ints=np.array([[v-eps,v+eps] for v in pw]); sep=all(abs(pw[i]-pw[j])>2*eps for i in range(K) for j in range(i)); passed=bool(np.all(near>0) and np.all(disp<near/2) and eps<=.1)
    np.testing.assert_allclose(stored['nearest'],near);np.testing.assert_allclose(stored['displacement90'],disp);np.testing.assert_allclose(stored['epsilon'],eps);np.testing.assert_allclose(stored['weight_intervals'],ints);assert stored['pass']==passed and stored['disjoint_weights']==sep
    st.append((passed,sep,ints))
   else: assert stored['pass'] is False;st.append((False,False,None))
  assignments=[]
  if all(s[0] for s in st):
   for p1 in itertools.permutations(range(K)):
    for p2 in itertools.permutations(range(K)):
     P=[list(range(K)),list(p1),list(p2)]
     if all(max(st[a][2][P[a][i],0] for a in range(3))<=min(st[a][2][P[a][i],1] for a in range(3)) for i in range(K)):assignments.append(P)
  assert assignments==r['assignments']
  status='unique' if len(assignments)==1 and all(s[1] for s in st) else 'ambiguous' if len(assignments)>1 else 'near_tie_abstain' if assignments else 'abstain'
  assert status==r['alignment_status'];gate_checks+=1
 counts[r['alignment_status']]=counts.get(r['alignment_status'],0)+1
print(json.dumps({'sealed_fits':len(seal['artifacts']),'calibration_scores_replayed':calibration_checks,'K_gt_1_gates_replayed':gate_checks,'selected_K_counts':selected,'alignment_status_counts':counts},sort_keys=True))
