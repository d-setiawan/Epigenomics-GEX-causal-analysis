import json,hashlib,itertools,sys,collections
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp,xlogy
H=Path(sys.argv[1]);repo=H.parents[4];data=repo/'project_resources/research/count_extensions_2026_11_01/data_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prot=json.loads((H/'protocol.json').read_text());seal=json.loads((H/'fit_seal.json').read_text());manifest=json.loads((data/'manifest.json').read_text())
assert sha(H/'protocol.json')==seal['protocol_sha256']=='6f39a86e041c00cc3272ae2958a38a8b0a7b13b981ce7d8d27a423bc1dd4c7a5'
assert sha(H/'source_approval.json')==seal['source_approval_sha256'];assert seal['applications']==36
for name,h in seal['artifacts'].items():assert sha(H/'fits'/name)==h
for name,h in prot['source_sha256'].items():assert sha(H/name)==h
for name,h in prot['inputs'].items():assert sha(repo/name)==h
assert sha(data/'manifest.json')==prot['manifest_sha256']
z=np.array([[0,0],[0,1],[1,0],[1,1]]);counts=collections.Counter();check=collections.Counter();possible=[]
def split(ids):
 ix=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()));n=int(.75*len(ids));return ix[:n],ix[n:]
def logp(X,B,p):
 rate=B[:,0]+z@B[:,1:].T
 assert (rate>=0).all() and np.isfinite(rate).all()
 lp=np.full(4,-np.inf);lp[p>0]=np.log(p[p>0]);parts=np.sum(xlogy(X[:,None,:],rate[None,:,:])-rate[None,:,:]-gammaln(X[:,None,:]+1),axis=2);return logsumexp(parts+lp,axis=1)
def law(x):return np.array([x['masses'].get(str(i),0)+x['uniform_mass']/4 for i in range(4)])
def support(q):
 n1=int(np.sum((q[:,0]==1)&(q[:,1]==0)));n2=int(np.sum((q[:,0]==0)&(q[:,1]==1)));n3=int(np.sum((q[:,0]==1)&(q[:,1]==1)))
 if not(n1 and n2):return False
 for a,b,c,d in itertools.product([0,1],repeat=4):
  needed=[sum([a==0,c==0]),sum([b==0,d==0]),sum([a==1,b==1,c==1,d==1])]
  if needed[0]<=n1 and needed[1]<=n2 and needed[2]<=n3 and n1-needed[0]+n3-needed[2]>0 and n2-needed[1]+n3-needed[2]>0:return True
 return False
for folder in sorted((H/'fits').iterdir()):
 exe=json.loads((folder/'execution.json').read_text());counts['exec_'+exe['status']]+=1; assert exe['start_states']==['completed']*3
 rows=[json.loads((folder/f'start_{s}.json').read_text()) for s in range(3)];check['starts']+=len(rows)
 for m in rows:
  counts['start_'+m['status']]+=1
  if m['status']=='ok':
   assert m['selected_start']==f"spectral_{m['start_seed']}";assert np.isfinite(m['penalized_objective']);np.testing.assert_allclose(m['penalized_objective'],m['observed_loglik']-m['penalty'],rtol=1e-11)
   cfg=m['config'];assert cfg['K']==2 and cfg['max_iter']==20 and cfg['tol']==.05 and cfg['epsilon']==1e-50 and cfg['kappa']==1e-200 and cfg['distribution']=='Poisson' and cfg['algorithm']=='PSAEM';B=np.array(m['raw_B']);assert np.isfinite(B).all() and (B>=-1e-12).all() and (B<=np.array([2,3,3])+1e-12).all();pen=cfg['pen']*np.minimum(abs(B[:,1:]),cfg['tau']).sum();np.testing.assert_allclose(pen,m['penalty'],rtol=1e-10,atol=1e-7)
  else:check['failed_start_records']+=1
 good=[r for r in rows if r['status']=='ok' and np.isfinite(r['penalized_objective'])];best=max(good,key=lambda r:r['penalized_objective']) if good else None
 assert exe['best_completed_start']==(best['selected_start'] if best else None)
 app=json.loads((folder/'application.json').read_text());counts['app_'+app['status']]+=1
 if best is None:continue
 assert app['selected_start']==best['selected_start'];dg=best['diagnostics'];expected_status='finite_nonconverged' if dg['stopping_reason']!='tolerance' or dg['optimizer_failures']>0 else 'finite_converged';assert app['status']==expected_status;assert app['optimizer_failures']==dg['optimizer_failures'];d,a,width,control=app['dataset'],app['assay'],app['panel'],app['control'];spec=manifest['datasets'][d]['assays'][a]
 with np.load(data/'public'/d/'train'/f'{a}.npz',allow_pickle=False) as f:X=f['raw_counts'].astype(float);ids=f['cell_ids'];features=f['feature_ids'];depth=f['library_depth'].astype(float)
 outside=depth-X.sum(1);fi,ca=split(ids)
 if control=='column_null':
  seed=int.from_bytes(hashlib.sha256(f'1000|{d}|{a}'.encode()).digest()[:4],'big');rng=np.random.default_rng(seed)
  for group in [fi,ca]:
   for j in range(300):X[group,j]=X[rng.permutation(group),j]
 X=X[:,:width];assert app['feature_ids']==features[:width].tolist();assert app['fit_rows']==len(fi) and app['cal_rows']==len(ca)
 for m in good:
  B=np.array(m['B']);p=law(m['p']); v=logp(X[fi],B,p);np.testing.assert_allclose(v.sum(),m['observed_loglik'],rtol=1e-9,atol=1e-6);check['start_train_likelihoods']+=1
 if app['status'] not in ['finite_converged','finite_nonconverged']:continue
 B=np.array(app['B']);p=np.array(app['state_masses']);np.testing.assert_allclose(B,best['B']);np.testing.assert_allclose(p,law(best['p']));np.testing.assert_allclose(logp(X[fi],B,p).mean(),app['fit_average_loglik'],rtol=1e-9)
 ca_lp=logp(X[ca],B,p);summary=app['calibration_score'];assert summary['rows']==len(ca) and summary['zero_probability_rows']==int(np.isneginf(ca_lp).sum());assert summary['log_score_status']==('negative_infinity' if np.isneginf(ca_lp).any() else 'finite')
 if summary['log_score_status']=='finite':np.testing.assert_allclose(summary['log_score'],ca_lp.mean(),rtol=1e-9)
 else:assert summary['log_score'] is None
 q=(B[:,1:]!=0).astype(int);assert np.array_equal(q,app['Q']);assert support(q)==app['q_certificate']['pass'];assert int((p>0).sum())==app['positive_state_count'];np.testing.assert_allclose(np.std(X[fi],axis=0),app['fit_scales']);check['apps_score_Q']+=1
 counts['Q_pass' if support(q) else 'Q_fail']+=1;counts['stopping_'+str(app['stopping_reason'])]+=1;counts['positive_states_'+str(app['positive_state_count'])]+=1;counts['cal_'+summary['log_score_status']]+=1
 print(folder.name,app['status'],'Q',app['q_certificate']['pass'],'states',app['positive_state_count'],'optfail',app['optimizer_failures'],'cal',summary['log_score_status'])
print('SUMMARY',json.dumps({'counts':counts,'checks':check,'fit_seal_sha256':sha(H/'fit_seal.json')},sort_keys=True))
