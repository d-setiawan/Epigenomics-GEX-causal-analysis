"""C01 evaluation: verify the complete sealed fits before opening test/reference arrays."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model import ASSAYS,logprob,log_components,completion
from train import HERE,DATA,DATASETS,sha,write

def verify():
    seal=json.loads((HERE/'fit_seal.json').read_text()); protocol=json.loads((HERE/'protocol.json').read_text())
    if seal['evaluator_read'] is not False or sha(HERE/'protocol.json')!=seal['protocol_sha256']:raise ValueError('invalid fit seal')
    if len(seal['artifacts'])!=24:raise ValueError('incomplete fit set')
    if sha(DATA/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('changed manifest')
    for name,h in protocol['sources'].items():
        if sha(HERE/name)!=h:raise ValueError('changed source')
    for p,h in protocol['inputs'].items():
        if sha(HERE.parents[4]/p)!=h:raise ValueError('changed training input')
    for p,h in seal['artifacts'].items():
        if sha(HERE/'fits'/p)!=h:raise ValueError('changed fit')
    if sha(HERE/'source_approval.json')!=seal['source_approval_sha256']:raise ValueError('changed source approval')
    return seal,json.loads((DATA/'manifest.json').read_text())

def moments(weights,rates,kernel):
    mean=weights@rates; centered=rates-mean; cov=centered.T@(weights[:,None]*centered)
    var=rates if kernel=='poisson' else rates+rates*rates/2
    cov+=np.diag(weights@var)
    zero=weights@(np.exp(-rates) if kernel=='poisson' else (2/(2+rates))**2)
    return mean,cov,zero

def metrics(x,w,r,kernel,scale,width):
    x=x[:,:width]; r=r[:,:width]; scale=np.asarray(scale[:width]); mu,cov,zero=moments(w,r,kernel); observed=np.cov(x,rowvar=False,bias=True)
    ok=scale>0; pair=ok[:,None]&ok[None,:]; diag=np.eye(width,dtype=bool); den=np.outer(scale,scale)
    def rmse(v,mask):return float(np.sqrt(np.mean(v[mask]**2))) if mask.any() else None
    diff=np.divide(cov-observed,den,out=np.zeros_like(cov),where=pair)
    return {'rows':len(x),'features':width,'zero_scale_features':int((~ok).sum()),'log_score':float(logprob(x,w,r,kernel).mean()),'zero_frequency_rmse':float(np.sqrt(np.mean((zero-(x==0).mean(0))**2))),'standardized_mean_rmse':rmse(np.divide(mu-x.mean(0),scale,out=np.zeros(width),where=ok),ok),'standardized_variance_rmse':rmse(diff,diag&pair),'standardized_offdiag_covariance_rmse':rmse(diff,~diag&pair),'unscorable_covariance_entries':int((~pair).sum())}

def main():
    seal,manifest=verify(); outdir=HERE/'evaluation'; outdir.mkdir(exist_ok=False)
    public={}; references={}; hashes={}
    for d in DATASETS:
        public[d]={}
        for a in ASSAYS:
            spec=manifest['datasets'][d]['assays'][a]; item=spec['cohorts']['test']; path=HERE.parents[4]/item['path']
            if sha(path)!=item['sha256']:raise ValueError('test hash changed')
            with np.load(path,allow_pickle=False) as f: public[d][a]={k:f[k] for k in ('raw_counts','feature_ids','cell_ids')}
            z=public[d][a]; x=z['raw_counts']
            if x.ndim!=2 or x.shape[1]!=300 or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any() or z['feature_ids'].tolist()!=spec['selected_feature_ids']:raise ValueError('test schema mismatch')
            hashes[item['path']]=item['sha256']
        item=manifest['evaluator'][d]; path=HERE.parents[4]/item['path']
        if sha(path)!=item['sha256']:raise ValueError('reference hash changed')
        with np.load(path,allow_pickle=False) as f: references[d]={k:f[k] for k in f.files}
        hashes[item['path']]=item['sha256']
    ref=references['scmtr_seq_s1']; joined=np.column_stack([public['scmtr_seq_s1'][a]['raw_counts'][ref[f'paired_indices_{a}']] for a in ASSAYS])
    if joined.shape!=(595,900) or not np.array_equal(joined,ref['raw_counts']) or not np.array_equal(ref['column_assays'],np.repeat(ASSAYS,300)) or not np.array_equal(ref['column_ids'],np.concatenate([public['scmtr_seq_s1'][a]['feature_ids'] for a in ASSAYS])):raise ValueError('paired join changed')
    for filename in seal['artifacts']:
        result=json.loads((HERE/'fits'/filename).read_text()); d=result['dataset']; k=result['selected_K']; width=result['panel']; kernel=result['kernel']; control=result['control']
        output={'dataset':d,'panel':width,'kernel':kernel,'control':control,'selected_K':k,'alignment_status':result['alignment_status'],'assignments':len(result['assignments']),'within':{},'paired':{},'state':{},'paired_caveat':'selected populations differ and CI completion is an assumption'}
        if k is not None:
            for a in ASSAYS:
                z=public[d][a]; x=z['raw_counts']; fit=result['fits'][a][str(k)]; point=fit['restarts'][fit['best_restart']]; w=np.array(point['weights']); rates=np.array(point['rates']); scale=result['assays'][a]['fit_scale']
                views=sorted(set([30,width])); baseline=result['fits'][a]['1']; ib=baseline['best_restart']
                base=None if ib is None else baseline['restarts'][ib]
                record={'independent':{str(v):metrics(x[:,:width],w,rates,kernel,scale,v) for v in views},'K1_baseline':None if base is None else {str(v):metrics(x[:,:width],np.asarray(base['weights']),np.asarray(base['rates']),kernel,scale,v) for v in views},'completion':[]}
                for c in result.get('completions',[]):
                    i=ASSAYS.index(a); cw=np.asarray(c['weights']); cr=np.asarray(point['rates'])[c['permutation'][i]]; record['completion'].append({str(v):metrics(x[:,:width],cw,cr,kernel,scale,v) for v in views})
                output['within'][a]=record
                if d!='scmtr_seq_s1':
                    ref=references[d]
                    if not np.array_equal(ref[f'ids_{a}'],z['cell_ids']):raise ValueError('state row mismatch')
                    state=ref[f'state_{a}']; lj=log_components(x[:,:width],rates,kernel)+np.log(w); resp=np.exp(lj-logsumexp(lj,axis=1)[:,None])
                    cor=[]
                    for i in range(k):
                        cor.append([None if np.std(resp[:,i])==0 or np.std(state[:,j])==0 else float(np.corrcoef(resp[:,i],state[:,j])[0,1]) for j in range(state.shape[1])])
                    output['state'][a]={'responsibility_state_correlations':cor,'note':'public discovery/test rows matched by ID; descriptive, no truth-driven selection or alignment'}
            if d=='scmtr_seq_s1':
                for ai,bi in itertools.combinations(range(3),2):
                    a,b=ASSAYS[ai],ASSAYS[bi]; x=joined[:,ai*300:ai*300+width].astype(float); y=joined[:,bi*300:bi*300+width].astype(float); obs=(x-x.mean(0)).T@(y-y.mean(0))/len(x)
                    sa=np.asarray(result['assays'][a]['fit_scale']); sb=np.asarray(result['assays'][b]['fit_scale']); scale=np.outer(sa,sb); ok=scale>0
                    rec={'status':result['alignment_status'],'valid_pairs':int(ok.sum()),'unscorable_pairs':int((~ok).sum()),'views':{}}
                    for v in sorted(set([30,width])):
                        mask=ok[:v,:v]; denom=scale[:v,:v]; truth=obs[:v,:v]
                        def score(pred):return float(np.sqrt(np.mean(((pred-truth)[mask]/denom[mask])**2))) if mask.any() else None
                        ranges=result.get('cross_covariance_ranges',{}).get(f'{a}__{b}')
                        if ranges:
                            low=np.asarray(ranges['lower'])[:v,:v]; high=np.asarray(ranges['upper'])[:v,:v]
                            diagnostics={'entries':int(mask.sum()),'empirical_inside_count':int(((truth>=low)&(truth<=high)&mask).sum()),'standardized_mean_width':float(np.mean((high-low)[mask]/denom[mask])) if mask.any() else None,'interpretation':'entrywise candidate range; descriptive, not confidence coverage'}
                        else: diagnostics=None
                        rec['views'][str(v)]={'zero_baseline_rmse':score(np.zeros((v,v))),'candidate_rmse':[score(completion([{'weights':np.asarray(result['fits'][aa][str(k)]['restarts'][result['fits'][aa][str(k)]['best_restart']]['weights']),'rates':np.asarray(result['fits'][aa][str(k)]['restarts'][result['fits'][aa][str(k)]['best_restart']]['rates'])} for aa in ASSAYS],c['permutation'])[2][ai,bi][:v,:v]) for c in result.get('completions',[])],'entrywise_range':diagnostics}
                    output['paired'][f'{a}__{b}']=rec
        write(outdir/filename,output)
    write(HERE/'evaluation_seal.json',{'fit_seal_sha256':sha(HERE/'fit_seal.json'),'inputs':hashes,'outputs':{p.name:sha(p) for p in sorted(outdir.glob('*.json'))},'interpretation':'heldout exploratory evaluation; no calibration or model selection','paired_triplet_bootstrap':'omitted, optional under protocol'})
if __name__=='__main__':main()
