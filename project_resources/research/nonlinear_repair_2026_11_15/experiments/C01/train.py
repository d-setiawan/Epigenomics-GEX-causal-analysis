"""C01 sealed training only. No public test or evaluator path is opened."""
import concurrent.futures
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
import numpy as np
import scipy, sklearn
from model import ASSAYS,split,validate,permute,key_seed,em,logprob,log_components,valid,stability,compatible,completion
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
DATA=REPO/'project_resources/research/count_extensions_2026_11_01/data_v1'
DATASETS=('scmtr_seq_s1','linear_chain_0','constrained_collider_0')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    return v
def write(path,obj):
    with path.open('x') as f: json.dump(clean(obj),f,indent=2,sort_keys=True,allow_nan=False)
def protocol():
    m=json.loads((DATA/'manifest.json').read_text()); inputs={}
    for d,a in itertools.product(DATASETS,ASSAYS):
        p=DATA/'public'/d/'train'/f'{a}.npz'; h=sha(p)
        if h!=m['datasets'][d]['assays'][a]['cohorts']['train']['sha256']:raise ValueError('manifest hash mismatch')
        inputs[str(p.relative_to(REPO))]=h
    return {'plan_sha256':sha(HERE.parents[1]/'plans/C01_COUNT_STATE_VALIDATION.md'), 'sources':{name:sha(HERE/name) for name in ('model.py','train.py','evaluate.py','test_model.py')},'manifest_sha256':sha(DATA/'manifest.json'),'inputs':inputs,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'settings':{'panels':[30,300],'K':[1,2,3,4],'kernels':['poisson','nb2'],'controls':['original','column_null'],'restarts':[0,1,2],'bootstraps':32,'max_iterations':300,'tolerance':1e-7,'monotonicity_tolerance':1e-8,'rate_floor':1e-8,'minimum_bootstraps':26,'weight_uncertainty_max':.10,'bootstrap_quantile':.9,'permutation_seed':1000,'split':'SHA256 count-dica-v1: ID, first floor(.75*n)','nb_shape':2,'processes':2,'blas_threads':1}}
def read_training(dataset,assay,control,manifest):
    p=DATA/'public'/dataset/'train'/f'{assay}.npz'; spec=manifest['datasets'][dataset]['assays'][assay]
    with np.load(p,allow_pickle=False) as f:
        ids=f['cell_ids']; features=f['feature_ids']; x=validate(f['raw_counts'],ids,features,spec['selected_feature_ids']); depth=f['library_depth'].astype(float); outside=depth-x.sum(1)
        if depth.shape!=(len(x),) or not np.isfinite(depth).all() or np.any(outside<=0): raise ValueError('invalid depth metadata')
    fit,cal=split(ids)
    if control=='column_null': x=permute(x,[fit,cal],key_seed(1000,dataset,assay))
    return x,fit,cal,features,depth,outside

def choose(score):
    eligible=[k for k,v in score.items() if v is not None]
    if not eligible:return None
    best=max(score[k] for k in eligible)
    return min(k for k in eligible if score[k]>=best-1e-8)

def task(arg):
    dataset,width,kernel,control=arg; manifest=json.loads((DATA/'manifest.json').read_text()); fits={}; data={}
    for assay in ASSAYS:
        x,fi,ca,features,depth,outside=read_training(dataset,assay,control,manifest); x=x[:,:width]; data[assay]=(x,fi,ca,features[:width],depth,outside); fits[assay]={}
        for k in range(1,5):
            runs=[em(x[fi],k,kernel,s) for s in (0,1,2)]
            ok=[(i,m) for i,m in enumerate(runs) if m['converged']]
            best=max(ok,key=lambda z:z[1]['trace'][-1])[0] if ok else None
            score=float(logprob(x[ca],runs[best]['weights'],runs[best]['rates'],kernel).mean()) if best is not None else None
            fits[assay][k]={'restarts':runs,'best_restart':best,'calibration_score':score}
    scores={k:sum(fits[a][k]['calibration_score'] for a in ASSAYS) if all(fits[a][k]['calibration_score'] is not None for a in ASSAYS) else None for k in range(1,5)}
    selected=choose(scores); preferred={a:choose({k:fits[a][k]['calibration_score'] for k in range(1,5)}) for a in ASSAYS}
    out={'dataset':dataset,'panel':width,'kernel':kernel,'control':control,'fits':fits,'scores':scores,'selected_K':selected,'preferred_K':preferred,'assays':{a:{'fit_rows':len(data[a][1]),'calibration_rows':len(data[a][2]),'feature_ids':data[a][3].tolist(),'fit_scale':np.std(data[a][0][data[a][1]],axis=0).tolist()} for a in ASSAYS},'alignment_status':'no_common_fit','assignments':[]}
    if selected is None:return out
    points=[fits[a][selected]['restarts'][fits[a][selected]['best_restart']] for a in ASSAYS]
    out['depth_diagnostics']={}
    for a,point in zip(ASSAYS,points):
        x,fi,_,_,depth,outside=data[a]; lj=log_components(x[fi],point['rates'],kernel)+np.log(point['weights']); resp=np.exp(lj-logsumexp(lj,axis=1)[:,None])
        def corr(v,z):
            return None if np.std(v)==0 or np.std(z)==0 else float(np.corrcoef(v,z)[0,1])
        total=point['rates'].sum(1)
        out['depth_diagnostics'][a]={'total_rate':total.tolist(),'normalized_profiles':(point['rates']/total[:,None]).tolist(),'responsibility_depth_correlation':[corr(resp[:,i],depth[fi]) for i in range(selected)],'responsibility_outside300_correlation':[corr(resp[:,i],outside[fi]) for i in range(selected)],'column_null_metadata':'original depth retained; count-depth relation broken' if control=='column_null' else 'original metadata'}
    if selected==1:out['alignment_status']='K1_baseline'; return out
    bs=[]; states=[]
    for a,point in zip(ASSAYS,points):
        x,fi,_,_,_,_=data[a]; boots=[]
        for rep in range(32):
            rng=np.random.default_rng(key_seed('bootstrap',dataset,width,kernel,control,a,rep)); sample=x[rng.choice(fi,len(fi),replace=True)]
            candidates=[em(sample,selected,kernel,0,initial=(point['weights'],point['rates'])),em(sample,selected,kernel,key_seed('restart',dataset,width,kernel,control,a,rep))]
            good=[(j,m) for j,m in enumerate(candidates) if m['converged']]
            best=max(good,key=lambda z:z[1]['trace'][-1])[0] if good else None
            boots.append({'candidates':candidates,'best':best,'model':candidates[best] if best is not None else {'converged':False,'error':'no converged bootstrap candidate'}})
        models=[b['model'] for b in boots]; state=stability(point,models,len(fi)); bs.append(boots); states.append(state)
    perms=compatible(points,states); out.update(bootstrap=dict(zip(ASSAYS,bs)),stability=dict(zip(ASSAYS,states)),assignments=perms,alignment_status=('unique' if len(perms)==1 and all(st['disjoint_weights'] for st in states) else 'ambiguous' if len(perms)>1 else 'near_tie_abstain' if perms else 'abstain'))
    out['completions']=[]
    ranges={}
    for perm in perms:
        weights,rates,covs=completion(points,perm)
        proj={a:float(logprob(data[a][0][data[a][2]],weights,rates[i],kernel).mean()) for i,a in enumerate(ASSAYS)}
        out['completions'].append({'weights':weights,'permutation':perm,'projected_calibration_scores':proj})
        for (i,j),v in covs.items():
            name=f'{ASSAYS[i]}__{ASSAYS[j]}'
            if name not in ranges: ranges[name]={'lower':v.copy(),'upper':v.copy()}
            else: ranges[name]['lower']=np.minimum(ranges[name]['lower'],v); ranges[name]['upper']=np.maximum(ranges[name]['upper'],v)
    out['cross_covariance_ranges']=ranges
    return out

def main():
    p=HERE/'protocol.json'; seal=HERE/'protocol_seal.json'
    if p.exists() or seal.exists() or (HERE/'fits').exists():raise FileExistsError('refuse overwriting study')
    prot=protocol(); write(p,prot); write(seal,{'protocol_sha256':sha(p),'review_gate':'scientific source review required before execution','evaluator_read':False})
    raise SystemExit('Protocol sealed. Run train.py --run only after scientific source approval.')

def run():
    prot=json.loads((HERE/'protocol.json').read_text()); ps=json.loads((HERE/'protocol_seal.json').read_text())
    if sha(HERE/'protocol.json')!=ps['protocol_sha256'] or protocol()!=prot:raise ValueError('protocol/source/input/environment changed')
    if not (HERE/'source_approval.json').is_file():raise ValueError('scientific approval receipt missing')
    approval=json.loads((HERE/'source_approval.json').read_text())
    if approval.get('protocol_sha256')!=ps['protocol_sha256'] or approval.get('approved') is not True:raise ValueError('approval mismatch')
    dest=HERE/'fits'; dest.mkdir(exist_ok=False)
    grid=list(itertools.product(DATASETS,(30,300),('poisson','nb2'),('original','column_null')))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(task,grid)):
            name='__'.join(map(str,grid[i]))+'.json'; write(dest/name,result); print(name,flush=True)
    files={p.name:sha(p) for p in sorted(dest.glob('*.json'))}
    if len(files)!=len(grid):raise ValueError('incomplete grid')
    write(HERE/'fit_seal.json',{'protocol_sha256':ps['protocol_sha256'],'source_approval_sha256':sha(HERE/'source_approval.json'),'artifacts':files,'evaluator_read':False})
if __name__=='__main__': run() if sys.argv[1:]==['--run'] else main()
