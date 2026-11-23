"""Evaluate sealed predictions against the heldout biological paired counts."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
DATA = REPO / 'project_resources/research/count_extensions_2026_11_01/data_v1'
ASSAYS = ('RNA','H3K27ac','H3K4me3')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    seal = json.loads((HERE/'prediction_seal.json').read_text())
    if seal['evaluator_read'] is not False or seal['prediction_rows'] != 8100: raise ValueError('invalid seal')
    if sha(HERE/'protocol.json') != seal['protocol_sha256'] or sha(HERE/'predictions.csv') != seal['predictions_sha256'] or sha(DATA/'manifest.json') != seal['manifest_sha256']: raise ValueError('changed seal')
    for name,digest in seal['source_sha256'].items():
        if sha(HERE/name) != digest: raise ValueError('changed source')
    for name,digest in seal['inputs'].items():
        if sha(REPO/name) != digest: raise ValueError('changed input')
    with (HERE/'predictions.csv').open(newline='') as f: predictions=list(csv.DictReader(f))
    if len(predictions) != 8100: raise ValueError('incomplete predictions')
    # The paired reference is opened only after all the above checks.
    manifest=json.loads((DATA/'manifest.json').read_text())
    refpath=DATA/'evaluator/scmtr_seq_s1/paired_reference.npz'
    if sha(refpath) != manifest['evaluator']['scmtr_seq_s1']['sha256']: raise ValueError('reference hash mismatch')
    with np.load(refpath,allow_pickle=False) as f:
        x=f['raw_counts']; ids=f['column_ids']; assays=f['column_assays']
        if x.shape != (595,900) or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any(): raise ValueError('reference shape/count error')
        if not np.array_equal(assays,np.repeat(ASSAYS,300)): raise ValueError('assay order mismatch')
        parts={a:(x[:,k*300:k*300+30]>0) for k,a in enumerate(ASSAYS)}
        feature_ids={a:ids[k*300:k*300+30] for k,a in enumerate(ASSAYS)}
    rows=[]
    for rec in predictions:
        if rec['dataset'] != 'scmtr_seq_s1': continue
        a,b=rec['assay_a'],rec['assay_b']; i,j=int(rec['index_a']),int(rec['index_b'])
        if feature_ids[a][i] != rec['feature_a'] or feature_ids[b][j] != rec['feature_b']: raise ValueError('reference feature mismatch')
        left,right=parts[a][:,i],parts[b][:,j]
        joint=float(np.mean(left & right)); p=float(left.mean()); q=float(right.mean())
        lo,hi=float(rec['lower']),float(rec['upper'])
        elo,ehi=max(0.,p+q-1),min(p,q)
        if not elo-1e-12 <= joint <= ehi+1e-12: raise ValueError('paired marginal consistency failed')
        gap=max(lo-joint,0.,joint-hi)
        rows.append([a,b,i,j,rec['feature_a'],rec['feature_b'],joint,p,q,lo,hi,float(rec['width']),gap,elo,ehi])
    if len(rows)!=2700: raise ValueError('incomplete evaluation')
    with (HERE/'evaluation.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['assay_a','assay_b','index_a','index_b','feature_a','feature_b','paired_joint','paired_p','paired_q','train_lower','train_upper','train_width','outside_distance','paired_marginal_lower_tautology','paired_marginal_upper_tautology']); w.writerows(rows)
    distances=np.array([row[12] for row in rows]); widths=np.array([row[11] for row in rows])
    results={'paired_rows':595,'reference_shape':[595,900],'anchor_columns_per_assay':30,'pairs':len(rows),'inside':int((distances<=1e-12).sum()),'outside':int((distances>1e-12).sum()),'outside_max_distance':float(distances.max()),'width_min_median_max':[float(widths.min()),float(np.median(widths)),float(widths.max())],'reference_sha256':sha(refpath),'prediction_seal_sha256':sha(HERE/'prediction_seal.json'),'evaluation_sha256':sha(HERE/'evaluation.csv'),'same_holdout_marginal_bounds_check':'all pass; tautological, evaluator-only','interpretation':'Empirical training-marginal coupling intervals; biological assay-specific QC pools differ from the paired all-QC population. Not confidence intervals or causal claims.'}
    with (HERE/'evaluation_summary.json').open('x') as f: json.dump(results,f,indent=2,sort_keys=True)

if __name__=='__main__': main()
