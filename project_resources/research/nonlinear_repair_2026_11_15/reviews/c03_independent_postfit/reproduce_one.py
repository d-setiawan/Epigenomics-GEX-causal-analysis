import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1]);import worker
H=Path(sys.argv[1]);folder=H/'fits'/'scmtr_seq_s1__RNA__30__original';stored=json.loads((folder/'start_0.json').read_text());file=worker.DATA/'public/scmtr_seq_s1/train/RNA.npz'
with np.load(file,allow_pickle=False) as a:x=a['raw_counts'].astype(float)[:,:30];ids=a['cell_ids']
order=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()));idx=order[:int(.75*len(ids))];active=np.flatnonzero(x[idx].std(0,ddof=1)>0);r=worker.fit_start(x,idx,active,0)
assert r['status']==stored['status'];np.testing.assert_allclose(r['final_objective'],stored['final_objective'],rtol=1e-9,atol=1e-9);np.testing.assert_allclose(r['weights'],stored['weights'],rtol=1e-8,atol=1e-8);np.testing.assert_allclose(r['rates'],stored['rates'],rtol=1e-8,atol=1e-8)
print('PASS representative official optimizer seed replay: S1 RNA 30 original seed 0; status',r['status'],'iterations',r['niter'])
