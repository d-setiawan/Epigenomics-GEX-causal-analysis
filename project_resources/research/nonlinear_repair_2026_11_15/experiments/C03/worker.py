"""One C03 application; source-fit and calibration inputs only."""
import json,sys,time,traceback,warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from adapter import simplex_qp,scipy_compatibility,stable_inverse
from common import seed,split,control,score_rows,score_summary,active_status
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[4];STUDY=HERE.parents[1]
DATA=REPO/'project_resources/research/count_extensions_2026_11_01/data_v1';sys.path.insert(0,str(STUDY/'sources/zk_meitd_root'))
from solver.MEITD import MEITD

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def write(p,obj):
    with p.open('x') as f:json.dump(clean(obj),f,sort_keys=True,allow_nan=False)

def fit_start(x,idx,active,rs):
    start=time.monotonic();model=None
    try:
        train=x[idx][:,active];labels=KMeans(n_clusters=4,n_init=1,random_state=rs).fit_predict(np.log1p(train));counts=np.bincount(labels,minlength=4)
        if np.any(counts==0):raise ValueError('empty initial cluster')
        initial=np.stack([train[labels==k].mean(0) for k in range(4)],axis=1)
        if len(np.unique(initial,axis=1).T)!=4:raise ValueError('fewer than four distinct initial means')
        model=MEITD(d=4)
        with warnings.catch_warnings(record=True) as caught,scipy_compatibility():
            warnings.simplefilter('always')
            direct,w,conv=model.solve_mean_weight(train.T,initial,np.full(4,.25),100,Mtol=1e-4,wtol=1e-4,ftol=None,translate_init='data',regularize_data=True,update_weights=True,warmup=0,qp_solver=simplex_qp,AA_depth=0,monit=0)
            final_cost=float(model.eval_cost());raw,discrepancy=stable_inverse(model,train.T)
        if not np.isfinite(final_cost) or not np.isfinite(w).all() or np.any(w<.025-1e-8) or not np.isclose(w.sum(),1,atol=1e-8,rtol=0):raise ValueError('invalid final objective or weights')
        rates=np.tile(x[idx].mean(0),(4,1));rates[:,active]=raw.T
        if not np.isfinite(rates).all() or np.any(rates<0):raise ValueError('invalid restored rates')
        return {'status':'finite_converged' if conv else 'finite_nonconverged','seed':rs,'final_objective':final_cost,'weights':w,'rates':rates,'initial_weights':np.full(4,.25),'initial_rates_active':initial,'source_tau':model.tau,'costs':model.costs,'Mstepsize':model.Mstepsize,'wstepsize_relative_trace':model.wstepsize,'niter':model.niter,'n_Mstep':model.n_Mstep,'n_wstep':model.n_wstep,'Mconv':bool(model.Mconv),'wconv':bool(model.wconv),'n_cost':model.n_cost,'n_deriv':model.n_deriv,'box_lower':model.Mlb,'box_upper':model.Mub,'inverse_discrepancy_max':discrepancy,'weight_floor_hits':int(np.isclose(w,.025,atol=1e-8,rtol=0).sum()),'warnings':[f'{type(a.message).__name__}: {a.message}' for a in caught],'elapsed_seconds':time.monotonic()-start}
    except Exception as e:
        return {'status':'failed','seed':rs,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'iterations':None if model is None else model.niter,'costs':[] if model is None else model.costs,'Mstepsize':[] if model is None else model.Mstepsize,'wstepsize_relative_trace':[] if model is None else model.wstepsize}

def run(dataset,assay,width,ctl,folder):
    spec=json.loads((DATA/'manifest.json').read_text())['datasets'][dataset]['assays'][assay]
    with np.load(DATA/'public'/dataset/'train'/f'{assay}.npz',allow_pickle=False) as z:x=z['raw_counts'].astype(float);ids=z['cell_ids'];features=z['feature_ids'];depth=z['library_depth'].astype(float)
    if x.ndim!=2 or x.shape[1]!=300 or len(x)!=len(ids) or features.tolist()!=spec['selected_feature_ids'] or not np.isfinite(x).all() or np.any(x<0) or np.any(x!=np.floor(x)) or depth.shape!=(len(x),):raise ValueError('invalid inputs')
    outside=depth-x.sum(1)
    if not np.isfinite(depth).all() or np.any(outside<=0):raise ValueError('invalid depth')
    fit,cal=split(ids)
    if ctl=='column_null':x=control(x,[fit,cal],seed(1000,dataset,assay))
    x=x[:,:width];sd=np.std(x[fit],axis=0,ddof=1);active=np.flatnonzero(sd>0);inactive=np.flatnonzero(sd==0)
    out={'dataset':dataset,'assay':assay,'panel':width,'control':ctl,'feature_ids':features[:width],'fit_rows':len(fit),'cal_rows':len(cal),'fit_scales':np.std(x[fit],axis=0),'active_indices':active,'inactive_indices':inactive,'inactive_rates':x[fit][:,inactive].mean(0),'source_objective_constant':'C(V)=sum_{k=1}^4 k!*tau_k*sum_{i1<...<ik} (mean_t product_l V[i_l,t])^2; numerical value not computed','status':'failed','metadata_note':'original depths; count-depth relation broken by null' if ctl=='column_null' else 'original metadata'}
    gate=active_status(len(active))
    if gate!='eligible':out['status']=gate;write(folder/'application.json',out);return
    records=[]
    for rs in range(3):
        write(folder/f'start_{rs}.running.json',{'seed':rs,'state':'started'})
        result=fit_start(x,fit,active,rs);write(folder/f'start_{rs}.json',result);records.append(result)
    valid=[r for r in records if r['status'] in ('finite_converged','finite_nonconverged')]
    if valid:
        best=min(valid,key=lambda r:r['final_objective']);w=best['weights'];rates=best['rates'];fl,lj=score_rows(x[fit],w,rates);cl,_=score_rows(x[cal],w,rates)
        out.update(status=best['status'],selected_seed=best['seed'],weights=w,rates=rates,final_objective=best['final_objective'],source_tau=best['source_tau'],fit_score=score_summary(fl),calibration_score=score_summary(cl),start_objective_spread=float(max(r['final_objective'] for r in valid)-min(r['final_objective'] for r in valid)),start_disagreements=[])
        for r in valid:
            dist=np.linalg.norm(np.sqrt(rates)[:,None,:]-np.sqrt(r['rates'])[None,:,:],axis=2);i,j=linear_sum_assignment(dist**2);out['start_disagreements'].append({'seed':r['seed'],'assignment':j,'sqrt_rate_distances':dist[i,j]})
        total=rates.sum(1)
        out['component_total_rate']=total
        out['normalized_profiles']=[None if z==0 else (rates[i]/z).tolist() for i,z in enumerate(total)]
        out['zero_total_profile_count']=int((total==0).sum())
        supported=np.isfinite(fl);out['responsibility_undefined_rows']=int((~supported).sum())
        out['responsibility_diagnostics_scope']='supported source-fit rows only; conditional subset, not all cells'
        if supported.any():
            resp=np.exp(lj[supported]-logsumexp(lj[supported],axis=1)[:,None]);di=depth[fit][supported];oi=outside[fit][supported]
            def corr(a,b):return None if len(a)<2 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
            out['responsibility_depth_correlation']=[corr(resp[:,i],di) for i in range(4)]
            out['responsibility_outside300_correlation']=[corr(resp[:,i],oi) for i in range(4)]
        else:
            out['responsibility_depth_correlation']=[None]*4;out['responsibility_outside300_correlation']=[None]*4
    write(folder/'application.json',out)
if __name__=='__main__':
    d,a,w,c,p=sys.argv[1:];run(d,a,int(w),c,Path(p))
