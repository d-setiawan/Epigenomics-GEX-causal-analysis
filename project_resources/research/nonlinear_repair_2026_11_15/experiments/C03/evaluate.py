"""Evaluate sealed C03 count predictions on unchanged public test matrices."""
import json
import numpy as np
from pathlib import Path
from common import score_rows,score_summary,moments
from run import HERE,REPO,DATA,sha,write,tree,protocol
C01=HERE.parent/'C01';C02=HERE.parent/'C02'
def metrics(x,w,r,scales,width):
    x=x[:,:width];r=r[:,:width];scales=np.asarray(scales[:width]);valid=scales>0;pair=valid[:,None]&valid[None,:];eye=np.eye(width,dtype=bool);den=np.outer(scales,scales);mean,cov,zero=moments(w,r);obs=np.cov(x,rowvar=False,bias=True)
    def rms(v,m):return float(np.sqrt(np.mean(v[m]**2))) if m.any() else None
    delta=np.divide(cov-obs,den,out=np.zeros_like(cov),where=pair);dm=np.divide(mean-x.mean(0),scales,out=np.zeros(width),where=valid);score,_=score_rows(x,w,r)
    return {'features':width,**score_summary(score),'zero_frequency_rmse':float(np.sqrt(np.mean((zero-(x==0).mean(0))**2))),'standardized_mean_rmse':rms(dm,valid),'standardized_variance_rmse':rms(delta,eye&pair),'standardized_offdiag_covariance_rmse':rms(delta,~eye&pair),'zero_scale_features':int((~valid).sum()),'unscorable_covariance_entries':int((~pair).sum())}
def checked_seal(study):
    seal=json.loads((study/'fit_seal.json').read_text())
    if seal['evaluator_read'] is not False:raise ValueError('fit seal permits evaluator access')
    if tree(study/'fits')!=seal['artifacts']:raise ValueError('fit artifact hash mismatch')
    return seal

def main():
    seal=checked_seal(HERE);prot=json.loads((HERE/'protocol.json').read_text());approval=json.loads((HERE/'source_approval.json').read_text())
    if seal['applications']!=36 or seal['protocol_sha256']!=sha(HERE/'protocol.json') or protocol()!=prot or approval.get('approved') is not True or approval.get('protocol_sha256')!=seal['protocol_sha256'] or seal['source_approval_sha256']!=sha(HERE/'source_approval.json'):raise ValueError('C03 seal, source or approval mismatch')
    c1=checked_seal(C01);c2=checked_seal(C02) if (C02/'fit_seal.json').exists() else None
    m=json.loads((DATA/'manifest.json').read_text());out=HERE/'evaluation';out.mkdir(exist_ok=False);input_hashes={}
    for folder in sorted((HERE/'fits').iterdir()):
        e=json.loads((folder/'execution.json').read_text());d,a,width,ctl=folder.name.split('__');width=int(width);rec=m['datasets'][d]['assays'][a]['cohorts']['test'];path=REPO/rec['path']
        if sha(path)!=rec['sha256']:raise ValueError('test hash mismatch')
        with np.load(path,allow_pickle=False) as f:x=f['raw_counts'].astype(float);ids=f['feature_ids']
        if x.ndim!=2 or x.shape[1]!=300 or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any() or ids.tolist()!=m['datasets'][d]['assays'][a]['selected_feature_ids']:raise ValueError('invalid test schema')
        input_hashes[rec['path']]=rec['sha256'];x=x[:,:width];views=sorted(set((30,width)));result={'execution':e,'C03':None,'C01':{},'C02':None,'interpretation':'exploratory prediction comparison; no shared-state or causal claim'}
        ap=folder/'application.json'
        if e['complete_application'] and ap.exists():
            f=json.loads(ap.read_text())
            if f['status'] in ('finite_converged','finite_nonconverged'):
                w=np.asarray(f['weights']);r=np.asarray(f['rates']);result['C03']={str(v):metrics(x,w,r,f['fit_scales'],v) for v in views}
        cpath=C01/'fits'/f'{d}__{width}__poisson__{ctl}.json'
        if cpath.exists():
            f=json.loads(cpath.read_text())
            for k in (1,4):
                q=f['fits'][a][str(k)];i=q['best_restart']
                if i is None:result['C01'][str(k)]={'status':'unavailable'};continue
                mod=q['restarts'][i];result['C01'][str(k)]={str(v):metrics(x,np.asarray(mod['weights']),np.asarray(mod['rates']),f['assays'][a]['fit_scale'],v) for v in views}
        if c2 is not None:
            p=C02/'fits'/folder.name/'application.json';ep=C02/'fits'/folder.name/'execution.json'
            if p.exists() and ep.exists() and json.loads(ep.read_text())['complete_three_start_application']:
                f=json.loads(p.read_text())
                if f['status'] in ('finite_converged','finite_nonconverged'):
                    B=np.asarray(f['B']);w=np.asarray(f['state_masses']);z=np.array([[0,0],[0,1],[1,0],[1,1]]);r=B[:,0]+z@B[:,1:].T;positive=w>0
                    result['C02']={str(v):metrics(x,w[positive],r[positive],f['fit_scales'],v) for v in views}
        write(out/(folder.name+'.json'),result)
    if len(list(out.glob('*.json')))!=36:raise ValueError('missing evaluation outputs')
    write(HERE/'evaluation_seal.json',{'C03_fit_seal_sha256':sha(HERE/'fit_seal.json'),'C01_fit_seal_sha256':sha(C01/'fit_seal.json'),'C02_fit_seal_sha256':sha(C02/'fit_seal.json') if c2 else None,'test_inputs':input_hashes,'outputs':tree(out),'paired_or_state_reference_read':False})
if __name__=='__main__':main()
