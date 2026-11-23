import sys,json,hashlib,pathlib,collections
import numpy as np
from scipy.special import xlogy,gammaln,logsumexp
H=pathlib.Path(sys.argv[1]);repo=H.parents[4];data=repo/'project_resources/research/count_extensions_2026_11_01/data_v1';C01=H.parent/'C01';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((data/'manifest.json').read_text());seal=json.loads((H/'evaluation_seal.json').read_text());assert seal['C02_fit_seal_sha256']==sha(H/'fit_seal.json') and seal['C01_fit_seal_sha256']==sha(C01/'fit_seal.json') and seal['paired_or_state_reference_read'] is False
for f,h in seal['outputs'].items():assert sha(H/'evaluation'/f)==h
for f,h in seal['test_inputs'].items():assert sha(repo/f)==h
assert len(seal['outputs'])==36 and len(seal['test_inputs'])==9
z=np.array([[0,0],[0,1],[1,0],[1,1]]);num=collections.Counter();comparisons=collections.Counter()
def metrics(x,p,r,scale):
 width=x.shape[1];mu=p@r;center=r-mu;cov=(center*p[:,None]).T@center+np.diag(mu);zero=p@np.exp(-r);obs=np.cov(x,rowvar=False,bias=True);scale=np.asarray(scale[:width]);mask=(scale>0)[:,None]&(scale>0)[None,:];di=np.eye(width,dtype=bool);den=np.outer(scale,scale)
 diff=np.divide(cov-obs,den,out=np.zeros_like(cov),where=mask);delta=np.divide(mu-x.mean(0),scale,out=np.zeros(width),where=scale>0);rms=lambda v,m:float(np.sqrt(np.mean(v[m]**2))) if m.any() else None
 lp=np.full(len(p),-np.inf);lp[p>0]=np.log(p[p>0]);v=xlogy(x[:,None,:],r[None,:,:])-r[None,:,:]-gammaln(x[:,None,:]+1);logp=logsumexp(v.sum(2)+lp,axis=1);assert not np.isnan(logp).any() and not np.isposinf(logp).any();negative=int(np.isneginf(logp).sum())
 return {'features':width,'rows':len(x),'log_score':None if negative else float(logp.mean()),'log_score_status':'negative_infinity' if negative else 'finite','zero_probability_rows':negative,'zero_frequency_rmse':float(np.sqrt(np.mean((zero-(x==0).mean(0))**2))),'standardized_mean_rmse':rms(delta,scale>0),'standardized_variance_rmse':rms(diff,di&mask),'standardized_offdiag_covariance_rmse':rms(diff,~di&mask),'zero_scale_features':int((scale==0).sum()),'unscorable_covariance_entries':int((~mask).sum())}
def cmp(a,b):
 assert a.keys()==b.keys(),(a.keys(),b.keys())
 for k,v in a.items():
  if v is None or isinstance(v,str):assert v==b[k],(k,v,b[k])
  else:np.testing.assert_allclose(v,b[k],rtol=1e-10,atol=1e-10,err_msg=k)
for fn in seal['outputs']:
 e=json.loads((H/'evaluation'/fn).read_text());folder=H/'fits'/fn[:-5];app=json.loads((folder/'application.json').read_text());exe=json.loads((folder/'execution.json').read_text());assert e['execution']==exe;d,a,width,control=app['dataset'],app['assay'],app['panel'],app['control'];spec=m['datasets'][d]['assays'][a]
 with np.load(repo/spec['cohorts']['test']['path'],allow_pickle=False) as f:x=f['raw_counts'].astype(float);ids=f['feature_ids'];assert ids.tolist()==spec['selected_feature_ids']
 x=x[:,:width]
 if app['status'] in ('finite_converged','finite_nonconverged'):
  B=np.array(app['B']);p=np.array(app['state_masses']);r=B[:,0]+z@B[:,1:].T
  for v in sorted(set([30,width])):
   actual=e['C02'][str(v)];cmp(metrics(x[:,:v],p,r[:,:v],app['fit_scales']),actual);num['C02_metric_views']+=1;num['C02_'+actual['log_score_status']]+=1;num['C02_zero_probability_rows']+=actual['zero_probability_rows']
 else:assert e['C02'] is None;num['failed_apps_retained']+=1
 cpath=C01/'fits'/f'{d}__{width}__poisson__{control}.json';c=json.loads(cpath.read_text())
 for k in [1,4]:
  entry=c['fits'][a][str(k)];idx=entry['best_restart']
  if idx is None:assert e['C01_comparison'][str(k)]=={'status':'unavailable'};num['C01_unavailable']+=1;continue
  mod=entry['restarts'][idx];p=np.array(mod['weights']);r=np.array(mod['rates']);scale=c['assays'][a]['fit_scale']
  for v in sorted(set([30,width])):
   cmp(metrics(x[:,:v],p,r[:,:v],scale),e['C01_comparison'][str(k)][str(v)]);num['C01_metric_views']+=1
 if e['C02'] is not None:
  v=str(width);v2=e['C02'][v]['log_score'];one=e['C01_comparison']['1'][v]['log_score'];four=e['C01_comparison'].get('4',{}).get(v,{}).get('log_score')
  for name,other in [('K1',one),('K4',four)]:
   if other is None:comparisons[name+'_unavailable']+=1
   elif v2 is None:comparisons[name+'_C02_negative_infinity']+=1
   elif v2>other+1e-8:comparisons[name+'_C02_greater']+=1
   elif v2<other-1e-8:comparisons[name+'_C02_less']+=1
   else:comparisons[name+'_equal']+=1
print(json.dumps({'evaluation_hash':sha(H/'evaluation_seal.json'),'counts':num,'comparisons_panel_size':comparisons},sort_keys=True))
