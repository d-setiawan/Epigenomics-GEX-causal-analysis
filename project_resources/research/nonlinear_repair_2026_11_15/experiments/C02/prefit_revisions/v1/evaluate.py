"""Score only unchanged public test arrays after complete C02 and C01 fit seals."""
import json
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from core import exact_logprob,moments
from run import HERE,REPO,DATA,sha,write,tree,protocol
C01=HERE.parent/'C01'
def c01log(x,w,r):return logsumexp(x@np.log(r).T-r.sum(1)-gammaln(x+1).sum(1)[:,None]+np.log(w),axis=1)
def metrics(x,mean,cov,zero,score,scale,width):
    x=x[:,:width];scale=np.asarray(scale[:width]);ok=scale>0;pair=ok[:,None]&ok[None,:];diag=np.eye(width,dtype=bool);den=np.outer(scale,scale);obs=np.cov(x,rowvar=False,bias=True)
    def rms(v,mask):return float(np.sqrt(np.mean(v[mask]**2))) if mask.any() else None
    diff=np.divide(cov[:width,:width]-obs,den,out=np.zeros_like(obs),where=pair)
    dm=np.divide(mean[:width]-x.mean(0),scale,out=np.zeros(width),where=ok)
    return {'features':width,'rows':len(x),'log_score':float(score),'zero_frequency_rmse':float(np.sqrt(np.mean((zero[:width]-(x==0).mean(0))**2))),'standardized_mean_rmse':rms(dm,ok),'standardized_variance_rmse':rms(diff,diag&pair),'standardized_offdiag_covariance_rmse':rms(diff,~diag&pair),'zero_scale_features':int((~ok).sum()),'unscorable_covariance_entries':int((~pair).sum())}
def main():
    seal=json.loads((HERE/'fit_seal.json').read_text());prot=json.loads((HERE/'protocol.json').read_text()); cseal=json.loads((C01/'fit_seal.json').read_text())
    if seal['evaluator_read'] is not False or seal['applications']!=36 or sha(HERE/'protocol.json')!=seal['protocol_sha256'] or tree(HERE/'fits')!=seal['artifacts']:raise ValueError('invalid C02 seal')
    if cseal.get('evaluator_read') is not False:raise ValueError('invalid C01 fit seal')
    for n,h in cseal['artifacts'].items():
        if sha(C01/'fits'/n)!=h:raise ValueError('C01 fit changed')
    for n,h in prot['source_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('C02 source changed')
    approval=json.loads((HERE/'source_approval.json').read_text())
    if sha(HERE/'source_approval.json')!=seal['source_approval_sha256'] or approval.get('approved') is not True or approval.get('protocol_sha256')!=seal['protocol_sha256']:raise ValueError('approval changed')
    if protocol()!=prot:raise ValueError('training input, upstream, dependency, config, or runtime changed')
    m=json.loads((DATA/'manifest.json').read_text());out=HERE/'evaluation';out.mkdir(exist_ok=False);input_hashes={}
    for folder in sorted((HERE/'fits').iterdir()):
        e=json.loads((folder/'execution.json').read_text());output={'application':folder.name,'execution':e,'C02':None,'C01_comparison':{},'interpretation':'exploratory test comparison; no state or causal target'}
        parts=folder.name.split('__');dataset,assay,width,control=parts[0],parts[1],int(parts[2]),parts[3]
        record=m['datasets'][dataset]['assays'][assay]['cohorts']['test'];path=REPO/record['path']
        if sha(path)!=record['sha256']:raise ValueError('test hash mismatch')
        with np.load(path,allow_pickle=False) as f:x=f['raw_counts'].astype(float);ids=f['feature_ids']
        if x.ndim!=2 or x.shape[1]!=300 or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any() or ids.tolist()!=m['datasets'][dataset]['assays'][assay]['selected_feature_ids']:raise ValueError('test schema mismatch')
        input_hashes[record['path']]=record['sha256'];x=x[:,:width]
        ap=folder/'application.json'
        if e['complete_three_start_application'] and ap.exists():
            fit=json.loads(ap.read_text())
            if fit['status'] in ('finite_converged','finite_nonconverged'):
                B=np.asarray(fit['B']);p=np.asarray(fit['state_masses']);mean,cov,zero=moments(B,p);scale=fit['fit_scales']
                output['C02']={str(v):metrics(x,mean,cov,zero,exact_logprob(x[:,:v],B[:v],p).mean(),scale,v) for v in sorted(set((30,width)))}
        cpath=C01/'fits'/f'{dataset}__{width}__poisson__{control}.json'
        if cpath.is_file():
            c=json.loads(cpath.read_text());scale=c['assays'][assay]['fit_scale']
            for k in (1,4):
                entry=c['fits'][assay][str(k)];idx=entry['best_restart']
                if idx is None:output['C01_comparison'][str(k)]={'status':'unavailable'};continue
                mod=entry['restarts'][idx];w=np.asarray(mod['weights']);r=np.asarray(mod['rates']);mu=w@r;dr=r-mu;cov=dr.T@(w[:,None]*dr)+np.diag(mu);zero=w@np.exp(-r)
                output['C01_comparison'][str(k)]={str(v):metrics(x,mu,cov,zero,c01log(x[:,:v],w,r[:,:v]).mean(),scale,v) for v in sorted(set((30,width)))}
        write(out/(folder.name+'.json'),output)
    if len(list(out.glob('*.json')))!=36:raise ValueError('missing evaluation outputs')
    write(HERE/'evaluation_seal.json',{'C02_fit_seal_sha256':sha(HERE/'fit_seal.json'),'C01_fit_seal_sha256':sha(C01/'fit_seal.json'),'test_inputs':input_hashes,'outputs':tree(out),'paired_or_state_reference_read':False})
if __name__=='__main__':main()
