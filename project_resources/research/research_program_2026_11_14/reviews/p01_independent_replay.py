import csv, json, hashlib, itertools, pathlib, subprocess, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
EXP=ROOT/'experiments/P01'
DATA=REPO/'project_resources/research/count_extensions_2026_11_01/data_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadjson(p): return json.loads(p.read_text())
p=loadjson(EXP/'protocol.json'); seal=loadjson(EXP/'prediction_seal.json'); manifest=loadjson(DATA/'manifest.json')
assert sha(EXP/'protocol.json')==seal['protocol_sha256']
assert sha(EXP/'predictions.csv')==seal['predictions_sha256']
assert sha(DATA/'manifest.json')==p['manifest_sha256']==seal['manifest_sha256']
assert sha(REPO/p['plan'])==p['plan_sha256']
for name,h in p['source_sha256'].items(): assert sha(EXP/name)==h==seal['source_sha256'][name]
for name,h in p['inputs'].items(): assert sha(REPO/name)==h==seal['inputs'][name]
assert seal['evaluator_read'] is False
assays=['RNA','H3K27ac','H3K4me3']; datasets=['scmtr_seq_s1','linear_chain_0','constrained_collider_0']
values={}; shapes={}
for dataset,assay in itertools.product(datasets,assays):
 path=DATA/'public'/dataset/'train'/f'{assay}.npz'
 with np.load(path,allow_pickle=False) as f: x=f['raw_counts']; ids=f['feature_ids']
 assert x.ndim==2 and x.shape[1]==len(ids)==300 and x.shape[0]>0
 assert np.isfinite(x).all() and np.all(x>=0) and np.all(x==np.floor(x))
 assert ids.tolist()==manifest['datasets'][dataset]['assays'][assay]['selected_feature_ids']
 v=(x[:,:30]>0).mean(axis=0); assert np.array_equal(v,(x[::-1,:30]>0).mean(axis=0))
 values[dataset,assay]=(v,ids[:30]); shapes[dataset+'/'+assay]=[x.shape[0],30]
assert shapes==seal['train_shapes_first30']
with (EXP/'predictions.csv').open(newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==8100
seen=set(); err=0.
for r in rows:
 d,a,b=r['dataset'],r['assay_a'],r['assay_b']; i,j=int(r['index_a']),int(r['index_b']); key=(d,a,b,i,j)
 assert d in datasets and (a,b) in list(itertools.combinations(assays,2)) and 0<=i<30 and 0<=j<30 and key not in seen; seen.add(key)
 av,aid=values[d,a]; bv,bid=values[d,b]; assert r['feature_a']==aid[i] and r['feature_b']==bid[j]
 x,y=float(av[i]),float(bv[j]); lo=max(0.,x+y-1.); hi=min(x,y); z=x*y
 expected={'p':x,'q':y,'lower':lo,'upper':hi,'independent':z,'cov_lower':lo-z,'cov_upper':hi-z,'width':hi-lo}
 for name,v in expected.items(): err=max(err,abs(float(r[name])-v)); assert abs(float(r[name])-v)<1e-14
 for t in (lo,hi): assert min(t,x-t,y-t,1-x-y+t)>=-1e-14
 assert lo-1e-14<=z<=hi+1e-14
assert len(seen)==8100
before={name:sha(EXP/name) for name in ('predictions.csv','prediction_seal.json','evaluation.csv','evaluation_summary.json')}
for name in ('fit.py','evaluate.py'):
 t=subprocess.run([sys.executable,name],cwd=EXP,capture_output=True,text=True)
 assert t.returncode!=0 and 'FileExistsError' in t.stderr, (name,t.stderr)
assert before=={name:sha(EXP/name) for name in before}
# Read evaluation only after independent prediction checks above.
summary=loadjson(EXP/'evaluation_summary.json'); refpath=DATA/'evaluator/scmtr_seq_s1/paired_reference.npz'
assert sha(refpath)==manifest['evaluator']['scmtr_seq_s1']['sha256']==summary['reference_sha256']
with np.load(refpath,allow_pickle=False) as f: x=f['raw_counts']; ids=f['column_ids']; aa=f['column_assays']
assert x.shape==(595,900) and np.array_equal(aa,np.repeat(assays,300)) and np.isfinite(x).all() and np.all(x>=0) and np.all(x==np.floor(x))
with (EXP/'evaluation.csv').open(newline='') as f: ev=list(csv.DictReader(f))
assert len(ev)==2700
pred={(r['assay_a'],r['assay_b'],int(r['index_a']),int(r['index_b'])):r for r in rows if r['dataset']=='scmtr_seq_s1'}
seen=set(); above=below=0; maxgap=0.; eerr=0.
for r in ev:
 a,b=r['assay_a'],r['assay_b']; i,j=int(r['index_a']),int(r['index_b']); k=(a,b,i,j); assert k in pred and k not in seen; seen.add(k)
 ia=assays.index(a)*300+i; ib=assays.index(b)*300+j; t=pred[k]; assert ids[ia]==r['feature_a']==t['feature_a'] and ids[ib]==r['feature_b']==t['feature_b']
 aa1=x[:,ia]>0; bb1=x[:,ib]>0; joint=float((aa1&bb1).mean()); u=float(aa1.mean()); v=float(bb1.mean()); lo=float(t['lower']); hi=float(t['upper']); gap=max(lo-joint,0.,joint-hi)
 d={'paired_joint':joint,'paired_p':u,'paired_q':v,'train_lower':lo,'train_upper':hi,'train_width':float(t['width']),'outside_distance':gap,'paired_marginal_lower_tautology':max(0.,u+v-1.),'paired_marginal_upper_tautology':min(u,v)}
 for col,val in d.items(): eerr=max(eerr,abs(float(r[col])-val)); assert abs(float(r[col])-val)<1e-14
 assert d['paired_marginal_lower_tautology']-1e-12<=joint<=d['paired_marginal_upper_tautology']+1e-12
 above+=joint>hi+1e-12; below+=joint<lo-1e-12; maxgap=max(maxgap,gap)
assert len(seen)==2700 and above+below==summary['outside'] and summary['inside']==2700-summary['outside'] and abs(maxgap-summary['outside_max_distance'])<1e-14
assert sha(EXP/'evaluation.csv')==summary['evaluation_sha256'] and sha(EXP/'prediction_seal.json')==summary['prediction_seal_sha256']
report={'predictions_checked':len(rows),'evaluation_checked':len(ev),'max_prediction_error':err,'max_evaluation_error':eerr,'above':above,'below':below,'max_gap':maxgap,'input_hashes_verified':len(p['inputs']),'train_shapes_first30':shapes,'overwrite_refusal_verified':True,'no_files_overwritten':True,'reference_sha256':sha(refpath),'evaluation_sha256':sha(EXP/'evaluation.csv'),'protocol_sha256':sha(EXP/'protocol.json'),'prediction_seal_sha256':sha(EXP/'prediction_seal.json'),'prediction_sha256':sha(EXP/'predictions.csv'),'source_hashes':p['source_sha256']}
print(json.dumps(report,indent=2))
