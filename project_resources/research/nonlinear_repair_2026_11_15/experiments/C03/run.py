"""Seal and execute 36 C03 applications without accessing evaluation inputs."""
import concurrent.futures,hashlib,itertools,json,os,platform,signal,subprocess,sys,time
from importlib import metadata
from pathlib import Path
from threadpoolctl import threadpool_info
import numpy as np,scipy
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[4];STUDY=HERE.parents[1]
DATA=REPO/'project_resources/research/count_extensions_2026_11_01/data_v1';SOURCE=STUDY/'sources/zk_meitd_root'
ASSAYS=('RNA','H3K27ac','H3K4me3');DATASETS=('scmtr_seq_s1','linear_chain_0','constrained_collider_0')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
def tree(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
def protocol():
    m=json.loads((DATA/'manifest.json').read_text());inputs={}
    for d,a in itertools.product(DATASETS,ASSAYS):
        p=DATA/'public'/d/'train'/f'{a}.npz';h=sha(p)
        if h!=m['datasets'][d]['assays'][a]['cohorts']['train']['sha256']:raise ValueError('training hash mismatch')
        inputs[str(p.relative_to(REPO))]=h
    return {'plan_sha256':sha(STUDY/'plans/C03_MEITD_VALIDATION.md'),'manifest_sha256':sha(DATA/'manifest.json'),'inputs':inputs,'source_sha256':{p:sha(HERE/p) for p in ('adapter.py','common.py','worker.py','run.py','evaluate.py','test_adapter.py')},'upstream_source':tree(SOURCE/'solver'),'upstream_commit':'3cad5393b0dbbd98e53a139bd6ad046d685fac2b','interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),'versions':{n:metadata.version(n) for n in ('numpy','scipy','scikit-learn','threadpoolctl')},'threadpools':threadpool_info(),'environment_scope':'Base distributions recorded by version and loaded BLAS by path/version; not a byte-level full environment seal.','settings':{'r':4,'starts':[0,1,2],'initializer':'KMeans log1p, n_init=1; raw cluster mean, uniform weights','orders':[1,2,3,4],'actual_tau':'(n-k)!/(n-1)! = n*(n-k)!/n! [product reciprocal from j=1 to k-1]','solver':'MEITD source baseline; normalized tau common n factor','maxiter':100,'Mtol':1e-4,'wtol':1e-4,'ftol':None,'warmup':0,'AA_depth':0,'regularize_data':True,'translate_init':'data','lower_bound':.025,'qp_kkt_tolerance':1e-8,'standardized_box_tolerance':1e-10,'active_feature_rule':'fit sample variance ddof1 exactly >0','n_active_lt_9':'theorem dimension abstention','n_active_9':'source design unsupported','controls':['original','column_null'],'null_seed':'sha256(1000|dataset|assay), full300 separately within split','split':'sha256(count-dica-v1:ID), first floor(.75*n)','panels':[30,300],'processes':2,'blas_threads':1,'application_timeout_seconds':600,'source_random_seed':12345}}
def app(arg):
    d,a,width,ctl=arg;name='__'.join(map(str,arg));folder=HERE/'fits'/name;folder.mkdir(exist_ok=False);env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic()
    with (folder/'process.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),d,a,str(width),ctl,str(folder)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=proc.wait(timeout=600);timeout=False
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait();timeout=True
    states=[]
    for s in range(3):
        states.append('completed' if (folder/f'start_{s}.json').exists() else ('time_limited' if timeout else 'interrupted') if (folder/f'start_{s}.running.json').exists() else 'unattempted')
    rec=[json.loads((folder/f'start_{s}.json').read_text()) for s in range(3) if states[s]=='completed'];valid=[r for r in rec if r['status'] in ('finite_converged','finite_nonconverged')];best=min(valid,key=lambda r:r['final_objective']) if valid else None
    result=json.loads((folder/'application.json').read_text()) if (folder/'application.json').exists() else None
    early=result is not None and result['status'] in ('theorem_dimension_abstention','source_design_unsupported_n9')
    complete=code==0 and result is not None and (early or states==['completed']*3)
    write(folder/'execution.json',{'application':name,'elapsed_seconds':time.monotonic()-started,'exit_code':code,'timed_out':timeout,'start_states':states,'best_completed_seed':None if best is None else best['seed'],'best_completed_objective':None if best is None else best['final_objective'],'complete_application':complete,'status':'complete' if complete else 'resource_limited' if timeout else 'process_failed'})
    return name
def main():
    if sys.argv[1:]!=['--run']:
        if (HERE/'protocol.json').exists() or (HERE/'fits').exists():raise FileExistsError('study already frozen')
        p=protocol();write(HERE/'protocol.json',p);write(HERE/'protocol_seal.json',{'protocol_sha256':sha(HERE/'protocol.json'),'evaluator_read':False});print('Protocol sealed; independent review and source approval required');return
    p=json.loads((HERE/'protocol.json').read_text());s=json.loads((HERE/'protocol_seal.json').read_text());a=json.loads((HERE/'source_approval.json').read_text())
    if sha(HERE/'protocol.json')!=s['protocol_sha256'] or p!=protocol() or a.get('approved') is not True or a.get('protocol_sha256')!=s['protocol_sha256']:raise ValueError('protocol or review approval mismatch')
    (HERE/'fits').mkdir(exist_ok=False);grid=list(itertools.product(DATASETS,ASSAYS,(30,300),('original','column_null')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for name in pool.map(app,grid):print(name,flush=True)
    if len(list((HERE/'fits').glob('*/execution.json')))!=36:raise ValueError('missing application records')
    write(HERE/'fit_seal.json',{'protocol_sha256':s['protocol_sha256'],'source_approval_sha256':sha(HERE/'source_approval.json'),'artifacts':tree(HERE/'fits'),'applications':36,'evaluator_read':False})
if __name__=='__main__':main()
