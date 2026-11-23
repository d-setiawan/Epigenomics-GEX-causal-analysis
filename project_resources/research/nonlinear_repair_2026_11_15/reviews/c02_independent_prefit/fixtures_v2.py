import sys,json,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1]);import worker
from core import split
from dcrl.latent_law import SparseLatentLaw
from evaluate import metrics
ids=np.array(['a','b','c','d','e','f','g','h']);fit,cal=split(ids);features=np.array([f'f{i}' for i in range(300)]);x=np.zeros((8,300));x[cal[0],0]=1
with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
 base=Path(tmp); folder=base/'public'/'synthetic'/'train';folder.mkdir(parents=True); out=base/'out';out.mkdir();np.savez(folder/'RNA.npz',raw_counts=x,cell_ids=ids,feature_ids=features,library_depth=x.sum(1)+1)
 (base/'manifest.json').write_text(json.dumps({'datasets':{'synthetic':{'assays':{'RNA':{'selected_feature_ids':features.tolist()}}}}}))
 worker.DATA=base
 def fake_fit(X,cfg,start_specs,**kw):
  B=np.zeros((30,3));law=SparseLatentLaw(2,{0:1.});seed=start_specs[0]['seed']
  return {'status':'ok','raw_B':B,'B':B,'p':law,'gamma':None,'selected_start':f'spectral_{seed}','penalized_objective':float(seed),'orientation':{},'diagnostics':{'stopping_reason':'tolerance','optimizer_failures':0}}
 worker.fit_observed=fake_fit;worker.observed_loglikelihood=lambda *a,**kw:0.
 worker.run('synthetic','RNA',30,'original',out)
 r=json.loads((out/'application.json').read_text());assert r['status']=='finite_converged' and r['selected_start']=='spectral_2';assert r['calibration_score']=={'log_score':None,'log_score_status':'negative_infinity','rows':2,'zero_probability_rows':1};assert r['parameter_valid'];assert r['positive_state_count']==1
 print('PASS synthetic stubbed worker full calibration path: selected model retained; one zero-probability row, strict JSON')
 score=np.array([-np.inf,0.]);p=np.array([1.,0,0,0]);B=np.zeros((30,3));from core import moments
 mean,cov,zero=moments(B,p);report=metrics(x[cal,:30],mean,cov,zero,score,np.ones(30),30);assert report['log_score'] is None and report['zero_probability_rows']==1 and report['rows']==2;json.dumps(report,allow_nan=False);print('PASS full evaluator metrics path preserves both rows and -infinity status')
