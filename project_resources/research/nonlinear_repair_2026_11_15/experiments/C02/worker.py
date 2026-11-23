"""One three-start application, exclusively public training data."""
import json,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp,xlogy
from core import split,control,seed,q_certificate,exact_logprob,select_start,score_summary

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
DATA=REPO/'project_resources/research/count_extensions_2026_11_01/data_v1'
UPSTREAM=HERE.parents[1]/'sources/dcrl_root/src'
sys.path.insert(0,str(UPSTREAM))
from dcrl.config import ExperimentConfig
from dcrl.fitting import fit_observed,observed_loglikelihood
from dcrl.latent_law import SparseLatentLaw

def clean(v):
    if isinstance(v,SparseLatentLaw):return {'K':v.K,'masses':{str(k):float(x) for k,x in v.masses.items()},'uniform_mass':v.uniform_mass}
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v

def write(p,v):
    with p.open('x') as f:json.dump(clean(v),f,sort_keys=True,allow_nan=False)

def run(dataset,assay,width,control_name,partial):
    manifest=json.loads((DATA/'manifest.json').read_text());spec=manifest['datasets'][dataset]['assays'][assay]
    path=DATA/'public'/dataset/'train'/f'{assay}.npz'
    with np.load(path,allow_pickle=False) as f:
        x=f['raw_counts'].astype(float);ids=f['cell_ids'];features=f['feature_ids'];depth=f['library_depth'].astype(float)
    if x.ndim!=2 or x.shape[1]!=300 or len(x)!=len(ids) or len(set(ids.tolist()))!=len(ids) or features.tolist()!=spec['selected_feature_ids'] or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any() or depth.shape!=(len(x),):raise ValueError('invalid training schema')
    outside=depth-x.sum(1)
    if not np.isfinite(depth).all() or (outside<=0).any():raise ValueError('invalid depth metadata')
    fit,cal=split(ids)
    if control_name=='column_null':x=control(x,[fit,cal],seed(1000,dataset,assay))
    x=x[:,:int(width)];cfg=ExperimentConfig(N=len(fit),J=int(width),K=2,distribution='Poisson',algorithm='PSAEM',n_jobs=1).finalize()
    expected={'N':len(fit),'J':int(width),'K':2,'distribution':'Poisson','algorithm':'PSAEM','profile':'legacy','max_iter':20,'tol':.05,'C':1,'epsilon':1e-50,'kappa':1e-200,'pen':len(fit)**.25,'tau':2*len(fit)**(-.275),'inner_max_iter':None,'gain_numerator':None,'gain_offset':10.,'coefficient_bound_scale':1.}
    for name,value in expected.items():
        actual=getattr(cfg,name)
        if isinstance(value,float):
            if not np.isclose(actual,value,rtol=1e-12,atol=0):raise ValueError(f'unexpected finalized config {name}')
        elif actual!=value:raise ValueError(f'unexpected finalized config {name}')
    records=[]
    for s in range(3):
        write(partial/f'start_{s}.running.json',{'state':'started','seed':s})
        started=time.monotonic()
        try:
            np.random.seed(s)
            result=fit_observed(x[fit],cfg,start_specs=[{'kind':'spectral','seed':s,'id':f'spectral_{s}'}],known_Q=None,orientation='column_sum',backend='standard')
            result['application_start_elapsed_seconds']=time.monotonic()-started
            result['start_seed']=s
            if result.get('status')=='ok':
                rb=np.asarray(result['raw_B']); op=np.asarray(result['B']); law=result['p']; pp=np.array([law.probability(i) for i in range(4)]); zz=np.array([[0,0],[0,1],[1,0],[1,1]])
                rr=op[:,0]+zz@op[:,1:].T
                result['independent_validity']={'finite_raw_parameters':bool(np.isfinite(rb).all()),'raw_within_bounds':bool(np.all(rb>=-1e-12) and np.all(rb<=np.array([2,3,3])+1e-12)),'finite_oriented_parameters':bool(np.isfinite(op).all()),'finite_probabilities':bool(np.isfinite(pp).all()),'probability_normalized':bool((pp>=0).all() and np.isclose(pp.sum(),1)),'finite_nonnegative_rates':bool(np.isfinite(rr).all() and (rr>=0).all())} 
        except Exception as e:
            result={'status':'failed','start_seed':s,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'application_start_elapsed_seconds':time.monotonic()-started}
        write(partial/f'start_{s}.json',result);records.append(result)
    best=select_start(records)
    output={'dataset':dataset,'assay':assay,'panel':int(width),'control':control_name,'config':asdict(cfg),'effective_defaults':{'inner_max_iter':cfg.max_iter if cfg.inner_max_iter is None else cfg.inner_max_iter,'gain_numerator':.5 if cfg.gain_numerator is None else cfg.gain_numerator,'min_outer_iter':0 if cfg.min_outer_iter is None else cfg.min_outer_iter},'starts':[f'start_{s}.json' for s in range(3)],'fit_rows':len(fit),'cal_rows':len(cal),'feature_ids':features[:int(width)].tolist(),'fit_scales':np.std(x[fit],axis=0).tolist(),'status':'failed','selection':'largest attained penalized objective; first tie','null_depth_note':'original metadata; broken count-depth relationship' if control_name=='column_null' else 'original metadata'}
    if best is not None:
        B=np.asarray(best['B']); law=best['p'];p=np.array([law.probability(i) for i in range(4)]);raw=np.asarray(best['raw_B']); rates=B[:,0][None,:]+np.array([[0,0],[0,1],[1,0],[1,1]])@B[:,1:].T
        valid=(B.shape==(int(width),3) and np.isfinite(B).all() and np.isfinite(p).all() and (p>=0).all() and np.isclose(p.sum(),1) and (rates>=0).all() and np.all(raw>=-1e-12) and np.all(raw<=np.array([2,3,3])+1e-12))
        output.update(selected_start=best['selected_start'],B=B,state_masses=p,uniform_mass=law.uniform_mass,stored_masses=law.masses,parameter_valid=bool(valid),orientation=best['orientation'])
        if valid:
            fl=exact_logprob(x[fit],B,p);cl=exact_logprob(x[cal],B,p)
            official=observed_loglikelihood(x[fit],B,law,best['gamma'],'Poisson')
            if not np.isfinite(fl).all() or np.isnan(cl).any() or np.isposinf(cl).any() or not np.isclose(fl.sum(),official,rtol=1e-9,atol=1e-6):output.update(status='invalid_scoring',error='invalid training/calibration score or official scorer mismatch')
            else:
                z=np.array([[0,0],[0,1],[1,0],[1,1]],float);lj=np.sum(xlogy(x[fit,None,:],rates[None,:,:])-rates[None,:,:]-gammaln(x[fit,None,:]+1),axis=2)
                lp=np.full(4,-np.inf);lp[p>0]=np.log(p[p>0]);resp=np.exp(lj+lp-logsumexp(lj+lp,axis=1)[:,None]);post=resp@z
                def corr(v,w):return None if np.std(v)==0 or np.std(w)==0 else float(np.corrcoef(v,w)[0,1])
                dg=best['diagnostics'];nonconverged=dg['stopping_reason']!='tolerance' or dg['optimizer_failures']>0
                output.update(status='finite_nonconverged' if nonconverged else 'finite_converged',stopping_reason=dg['stopping_reason'],optimizer_failures=dg['optimizer_failures'],fit_average_loglik=float(fl.mean()),calibration_score=score_summary(cl),positive_state_count=int((p>0).sum()),uniform_fallback=bool(law.uniform_mass==1 and not law.masses),q_certificate=q_certificate(B[:,1:]!=0),Q=(B[:,1:]!=0).astype(int),raw_bound_saturation={'intercept_lower':int(np.isclose(raw[:,0],0,atol=1e-7).sum()),'intercept_upper':int(np.isclose(raw[:,0],2,atol=1e-7).sum()),'slope_lower':int(np.isclose(raw[:,1:],0,atol=1e-7).sum()),'slope_upper':int(np.isclose(raw[:,1:],3,atol=1e-7).sum())},fraction_fit_feature_means_exceed_8=float(np.mean(x[fit].mean(0)>8)),responsibility_depth_correlation=[corr(post[:,k],depth[fit]) for k in range(2)],responsibility_outside300_correlation=[corr(post[:,k],outside[fit]) for k in range(2)])
        else:output['status']='invalid_parameters'
    write(partial/'application.json',output)
if __name__=='__main__':
    d,a,w,c,directory=sys.argv[1:];run(d,a,int(w),c,Path(directory))
