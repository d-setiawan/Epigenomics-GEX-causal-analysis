import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
sys.path.insert(0,sys.argv[1]); import train,model
H=Path(sys.argv[1]); fs=json.loads((H/'fit_seal.json').read_text()); es=json.loads((H/'evaluation_seal.json').read_text()); manifest=json.loads((train.DATA/'manifest.json').read_text()); repo=H.parents[4];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert es['fit_seal_sha256']==sha(H/'fit_seal.json') and len(es['outputs'])==24
for f,h in es['outputs'].items(): assert sha(H/'evaluation'/f)==h
for f,h in es['inputs'].items(): assert sha(repo/f)==h
public={};refs={}
for d in train.DATASETS:
 public[d]={}
 for a in model.ASSAYS:
  with np.load(repo/manifest['datasets'][d]['assays'][a]['cohorts']['test']['path'],allow_pickle=False) as f:public[d][a]={k:f[k] for k in ('raw_counts','cell_ids','feature_ids')}
 with np.load(repo/manifest['evaluator'][d]['path'],allow_pickle=False) as f:refs[d]={k:f[k] for k in f.files}
def expected_metrics(x,w,r,kernel,scale,width):
 x=x[:,:width];r=r[:,:width];scale=np.array(scale[:width]);mu=w@r; dr=r-mu;C=(dr*w[:,None]).T@dr;V=r if kernel=='poisson' else r+r*r/2;C[np.diag_indices(width)]+=w@V
 z=w@(np.exp(-r) if kernel=='poisson' else (2/(2+r))**2);o=np.cov(x,rowvar=False,bias=True);mask=(scale>0)[:,None]&(scale>0)[None,:];den=np.outer(scale,scale);v=np.divide(C-o,den,out=np.zeros_like(C),where=mask); diag=np.eye(width,dtype=bool)
 rms=lambda y,m:float(np.sqrt(np.mean(y[m]**2))) if np.any(m) else None
 if kernel=='poisson':lp=x@np.log(r).T-r.sum(1)-gammaln(x+1).sum(1)[:,None]
 else:lp=x@(np.log(r)-np.log(2+r)).T+(2*np.log(2/(2+r))).sum(1)+np.log1p(x).sum(1)[:,None]
 return {'rows':len(x),'features':width,'zero_scale_features':int((scale==0).sum()),'log_score':float(logsumexp(lp+np.log(w),axis=1).mean()),'zero_frequency_rmse':float(np.sqrt(np.mean((z-(x==0).mean(0))**2))),'standardized_mean_rmse':rms(np.divide(mu-x.mean(0),scale,out=np.zeros(width),where=scale>0),scale>0),'standardized_variance_rmse':rms(v,diag&mask),'standardized_offdiag_covariance_rmse':rms(v,~diag&mask),'unscorable_covariance_entries':int((~mask).sum())}
def compare(got,exp):
 assert got.keys()==exp.keys()
 for k in got:
  if exp[k] is None:assert got[k] is None
  else:np.testing.assert_allclose(got[k],exp[k],atol=1e-11,rtol=1e-11,err_msg=k)
metric_count=0;state_count=0;pair_count=0
for filename in es['outputs']:
 r=json.loads((H/'fits'/filename).read_text());e=json.loads((H/'evaluation'/filename).read_text());d=r['dataset'];K=r['selected_K'];width=r['panel'];kernel=r['kernel'];A=model.ASSAYS
 assert (e['selected_K'],e['alignment_status'],e['assignments'])==(K,r['alignment_status'],len(r['assignments']))
 if K is None:continue
 for ai,a in enumerate(A):
  f=r['fits'][a][str(K)];p=f['restarts'][f['best_restart']];w=np.array(p['weights']);rates=np.array(p['rates']);x=public[d][a]['raw_counts'][:,:width];sc=r['assays'][a]['fit_scale'];rec=e['within'][a]
  for v in sorted(set([30,width])):
   compare(rec['independent'][str(v)],expected_metrics(x,w,rates,kernel,sc,v));metric_count+=1
   b=r['fits'][a]['1'];base=b['restarts'][b['best_restart']];compare(rec['K1_baseline'][str(v)],expected_metrics(x,np.array(base['weights']),np.array(base['rates']),kernel,sc,v));metric_count+=1
  for ci,c in enumerate(r.get('completions',[])):
   for v in sorted(set([30,width])):
    compare(rec['completion'][ci][str(v)],expected_metrics(x,np.array(c['weights']),rates[c['permutation'][ai]],kernel,sc,v));metric_count+=1
  if d!='scmtr_seq_s1':
   state=refs[d]['state_'+a];assert np.array_equal(refs[d]['ids_'+a],public[d][a]['cell_ids'])
   if kernel=='poisson': lp=x@np.log(rates).T-rates.sum(1)-gammaln(x+1).sum(1)[:,None]
   else:lp=x@(np.log(rates)-np.log(2+rates)).T+(2*np.log(2/(2+rates))).sum(1)+np.log1p(x).sum(1)[:,None]
   lp+=np.log(w); q=np.exp(lp-logsumexp(lp,axis=1)[:,None]); corr=e['state'][a]['responsibility_state_correlations']
   for i in range(K):
    for j in range(state.shape[1]):
     expected=None if np.std(q[:,i])==0 or np.std(state[:,j])==0 else np.corrcoef(q[:,i],state[:,j])[0,1]
     if expected is None:assert corr[i][j] is None
     else:np.testing.assert_allclose(corr[i][j],expected,atol=1e-10)
     state_count+=1
 if d=='scmtr_seq_s1':
  ref=refs[d];joined=np.column_stack([public[d][a]['raw_counts'][ref['paired_indices_'+a]] for a in A]);assert np.array_equal(joined,ref['raw_counts'])
  for ai,bi in itertools.combinations(range(3),2):
   a,b=A[ai],A[bi];x=joined[:,ai*300:ai*300+width];y=joined[:,bi*300:bi*300+width];obs=(x-x.mean(0)).T@(y-y.mean(0))/len(x);scale=np.outer(r['assays'][a]['fit_scale'],r['assays'][b]['fit_scale']);valid=scale>0;rec=e['paired'][a+'__'+b];assert rec['valid_pairs']==valid.sum() and rec['unscorable_pairs']==(~valid).sum()
   for v in sorted(set([30,width])):
    mask=valid[:v,:v];out=rec['views'][str(v)];score=float(np.sqrt(np.mean((obs[:v,:v][mask]/scale[:v,:v][mask])**2))) if mask.any() else None
    if score is None:assert out['zero_baseline_rmse'] is None
    else:np.testing.assert_allclose(out['zero_baseline_rmse'],score,atol=1e-11,rtol=1e-11)
    assert out['candidate_rmse']==[] and out['entrywise_range'] is None
    pair_count+=1
print(json.dumps({'fit_and_evaluation_hashes':24,'within_metric_records':metric_count,'state_correlations':state_count,'paired_zero_baseline_records':pair_count,'candidate_predictions':0},sort_keys=True))
