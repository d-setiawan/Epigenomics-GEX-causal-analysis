import hashlib,itertools,json,sys,collections,math
from pathlib import Path
import numpy as np
from scipy.special import xlogy,gammaln,logsumexp
H=Path(sys.argv[1]);repo=H.parents[4];data=repo/'project_resources/research/count_extensions_2026_11_01/data_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prot=json.loads((H/'protocol.json').read_text());seal=json.loads((H/'fit_seal.json').read_text());manifest=json.loads((data/'manifest.json').read_text());assert sha(H/'protocol.json')==seal['protocol_sha256']=='49c0f068cb213fefc555824bbbfeee36b55c12ef9fef6b8c1e0ee733b6ef8f68';assert sha(H/'source_approval.json')==seal['source_approval_sha256'];assert seal['applications']==36
for f,h in seal['artifacts'].items():assert sha(H/'fits'/f)==h
for f,h in prot['source_sha256'].items():assert sha(H/f)==h
for f,h in prot['inputs'].items():assert sha(repo/f)==h
assert sha(data/'manifest.json')==prot['manifest_sha256']
def split(ids):
 ix=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()));n=int(.75*len(ids));return ix[:n],ix[n:]
def score(X,w,r):
 lp=np.sum(xlogy(X[:,None,:],r[None,:,:])-r[None,:,:]-gammaln(X[:,None,:]+1),axis=2)+np.log(w);return logsumexp(lp,axis=1)
def check_score(v,s):
 n=int(np.isneginf(v).sum());assert s['rows']==len(v) and s['zero_probability_rows']==n and s['log_score_status']==('negative_infinity' if n else 'finite');assert not np.isnan(v).any() and not np.isposinf(v).any()
 if n:assert s['log_score'] is None
 else:np.testing.assert_allclose(s['log_score'],v.mean(),rtol=1e-9,atol=1e-9)
def esym(t):
 e=[np.ones(t.shape[1:])]+[np.zeros(t.shape[1:]) for _ in range(4)]
 for row in t:
  for k in range(4,0,-1):e[k]+=row*e[k-1]
 return e[1:]
def cost(X,r,w):
 mu=X.mean(0);sd=X.std(0,ddof=1);V=((X-mu)/sd).T;M=((r-mu)/sd).T;n=len(mu)
 a=np.zeros((n,4,4));a[:]=M[:,:,None]*M[:,None,:];b=V[:,:,None]*M[:,None,:];ea,eb=esym(a),esym(b)
 return sum(math.factorial(k)*(1/np.prod(np.arange(n-k+1,n)) if k>1 else 1)*((ea[k-1]*np.outer(w,w)).sum()-2*(eb[k-1]*w[None,:]).sum()/len(X)) for k in range(1,5))
counts=collections.Counter();checked=collections.Counter()
for f in sorted((H/'fits').iterdir()):
 ex=json.loads((f/'execution.json').read_text());counts['execution_'+ex['status']]+=1
 if not (f/'application.json').exists():counts['no_application']+=1;continue
 app=json.loads((f/'application.json').read_text());counts['app_'+app['status']]+=1;d,a,panel,ctl=app['dataset'],app['assay'],app['panel'],app['control'];spec=manifest['datasets'][d]['assays'][a]
 with np.load(data/'public'/d/'train'/f'{a}.npz',allow_pickle=False) as z:X=z['raw_counts'].astype(float);ids=z['cell_ids'];features=z['feature_ids']
 fi,ca=split(ids)
 if ctl=='column_null':
  seed=int.from_bytes(hashlib.sha256(f'1000|{d}|{a}'.encode()).digest()[:4],'big');rng=np.random.default_rng(seed)
  for group in [fi,ca]:
   for j in range(300):X[group,j]=X[rng.permutation(group),j]
 X=X[:,:panel];sd=X[fi].std(0,ddof=1);active=np.flatnonzero(sd>0);inactive=np.flatnonzero(sd==0);assert np.array_equal(active,app['active_indices']) and np.array_equal(inactive,app['inactive_indices']);np.testing.assert_allclose(app['inactive_rates'],X[fi][:,inactive].mean(0));np.testing.assert_allclose(app['fit_scales'],X[fi].std(0));assert app['feature_ids']==features[:panel].tolist();checked['active_and_scales']+=1
 if len(active)<10:
  assert app['status']==('theorem_dimension_abstention' if len(active)<9 else 'source_design_unsupported_n9');continue
 starts=[]
 for i in range(3):
  p=f/f'start_{i}.json'
  if not p.exists():counts['missing_start']+=1;continue
  r=json.loads(p.read_text());starts.append(r);counts['start_'+r['status']]+=1
  if r['status'] in ('finite_converged','finite_nonconverged'):
   w=np.array(r['weights']);rates=np.array(r['rates']);assert np.isfinite(w).all() and (w>=.025-1e-8).all() and abs(w.sum()-1)<1e-8;assert np.isfinite(rates).all() and (rates>=0).all();assert (rates[:,active]>=X[fi][:,active].min(0)-1e-9).all() and (rates[:,active]<=X[fi][:,active].max(0)+1e-9).all();tau=[1.0]+[1/np.prod(np.arange(len(active)-k+1,len(active))) for k in range(2,5)];np.testing.assert_allclose(r['source_tau'],tau);np.testing.assert_allclose(rates[:,inactive],np.broadcast_to(X[fi][:,inactive].mean(0),rates[:,inactive].shape));assert r['status']==('finite_converged' if r['Mconv'] and r['wconv'] else 'finite_nonconverged')
   truecost=cost(X[fi][:,active],rates[:,active],w);np.testing.assert_allclose(truecost,r['final_objective'],rtol=2e-7,atol=1e-6);checked['final_moment_objective']+=1
 valid=[r for r in starts if r['status'] in ('finite_converged','finite_nonconverged')]
 if valid:
  best=min(valid,key=lambda r:r['final_objective']);assert app['selected_seed']==best['seed'] and ex['best_completed_seed']==best['seed'];np.testing.assert_allclose(app['weights'],best['weights']);np.testing.assert_allclose(app['rates'],best['rates']);w=np.array(app['weights']);rates=np.array(app['rates']);check_score(score(X[fi],w,rates),app['fit_score']);check_score(score(X[ca],w,rates),app['calibration_score']);checked['selected_scores']+=1;counts['fit_'+app['fit_score']['log_score_status']]+=1;counts['cal_'+app['calibration_score']['log_score_status']]+=1
print(json.dumps({'seal':sha(H/'fit_seal.json'),'counts':counts,'checked':checked},sort_keys=True))
