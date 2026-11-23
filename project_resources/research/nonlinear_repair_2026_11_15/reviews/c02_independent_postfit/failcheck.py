import json,hashlib,pathlib,sys,numpy as np
from scipy.special import xlogy,gammaln,logsumexp
H=pathlib.Path(sys.argv[1]);data=H.parents[4]/'project_resources/research/count_extensions_2026_11_01/data_v1';Z=np.array([[0,0],[0,1],[1,0],[1,1]])
for folder in H.joinpath('fits').iterdir():
 app=json.loads((folder/'application.json').read_text());d,a,width,control=app['dataset'],app['assay'],app['panel'],app['control']
 with np.load(data/'public'/d/'train'/f'{a}.npz',allow_pickle=False) as f:X=f['raw_counts'].astype(float);ids=f['cell_ids']
 order=np.array(sorted(range(len(ids)),key=lambda i:hashlib.sha256(('count-dica-v1:'+str(ids[i])).encode()).hexdigest()));n=int(.75*len(ids));fi,ca=order[:n],order[n:]
 if control=='column_null':
  seed=int.from_bytes(hashlib.sha256(f'1000|{d}|{a}'.encode()).digest()[:4],'big');rng=np.random.default_rng(seed)
  for group in [fi,ca]:
   for j in range(300):X[group,j]=X[rng.permutation(group),j]
 X=X[fi,:width]
 for s in range(3):
  r=json.loads((folder/f'start_{s}.json').read_text())
  if r['status']=='ok':continue
  t=r['starts'][0];B=np.array(t['B']);p=t['p'];w=np.array([p['masses'].get(str(i),0)+p['uniform_mass']/4 for i in range(4)]);rate=B[:,0]+Z@B[:,1:].T;v=xlogy(X[:,None,:],rate[None,:,:])-rate[None,:,:]-gammaln(X[:,None,:]+1); lp=np.full(4,-np.inf);lp[w>0]=np.log(w[w>0]);logp=logsumexp(v.sum(2)+lp,axis=1)
  print(folder.name,s,'zero_prob_rows',np.isneginf(logp).sum(),'nan_rows',np.isnan(logp).sum(),'negative_rates',int((rate<0).sum()),'zero_rates',int((rate==0).sum()))
