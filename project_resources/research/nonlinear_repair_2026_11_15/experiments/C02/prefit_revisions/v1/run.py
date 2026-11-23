"""Seal and execute the C02 observed-input-only grid; 600-second application cap."""
import concurrent.futures,hashlib,itertools,json,os,platform,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,scipy
from core import select_start
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
STUDY=HERE.parents[1]
DATA=REPO/'project_resources/research/count_extensions_2026_11_01/data_v1'
UPSTREAM=STUDY/'sources/dcrl_root'
DEPS=Path('/tmp/epigenomics-dcrl-deps-20261115')
ASSAYS=('RNA','H3K27ac','H3K4me3')
DATASETS=('scmtr_seq_s1','linear_chain_0','constrained_collider_0')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
def tree(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
def protocol():
    m=json.loads((DATA/'manifest.json').read_text());inputs={}
    for d,a in itertools.product(DATASETS,ASSAYS):
        p=DATA/'public'/d/'train'/f'{a}.npz';h=sha(p)
        if h!=m['datasets'][d]['assays'][a]['cohorts']['train']['sha256']:raise ValueError('training input mismatch')
        inputs[str(p.relative_to(REPO))]=h
    receipt=STUDY/'reviews/dcrl_runtime_import_audit.json'
    if not DEPS.is_dir() or not receipt.is_file():raise ValueError('isolated runtime audit unavailable')
    return {'plan_sha256':sha(STUDY/'plans/C02_DCRL_VALIDATION.md'),'manifest_sha256':sha(DATA/'manifest.json'),'inputs':inputs,'upstream_commit':'2f768e048cb49b2af674fe31f76e55374362c00e','upstream_source':tree(UPSTREAM/'src'),'isolated_dependencies':tree(DEPS),'runtime_receipt_sha256':sha(receipt),'source_sha256':{n:sha(HERE/n) for n in ('core.py','worker.py','run.py','evaluate.py','test_core.py')},'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'config':{'K':2,'starts':[{'kind':'spectral','seed':i} for i in range(3)],'orientation':'column_sum','backend':'standard','known_Q':None,'algorithm':'PSAEM','distribution':'Poisson','profile':'legacy','max_iter':20,'tol':.05,'C':1,'epsilon':1e-50,'kappa':1e-200,'penalty':'N**0.25','tau':'2*N**(-0.275)','gain':'.5/(t+10)','inner_max_iter':20,'coefficient_bound_scale':1,'intercept_bounds':[0,2],'slope_bounds':[0,3],'processes':2,'blas_threads':1,'application_timeout_seconds':600,'panels':[30,300],'controls':['original','column_null'],'null_seed':'sha256(1000|dataset|assay), full300 within split','split':'sha256(count-dica-v1:ID), first floor(.75*n)'}}
def app(arg):
    dataset,assay,width,control=arg;name='__'.join(map(str,arg));folder=HERE/'fits'/name;folder.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(DEPS)+os.pathsep+str(UPSTREAM/'src'))
    start=time.monotonic()
    with (folder/'process.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),dataset,assay,str(width),control,str(folder)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=proc.wait(timeout=600);timed_out=False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);code=proc.wait();timed_out=True
    states=[]
    for s in range(3):
        states.append('completed' if (folder/f'start_{s}.json').exists() else ('time_limited' if timed_out else 'interrupted') if (folder/f'start_{s}.running.json').exists() else 'unattempted')
    complete=(folder/'application.json').exists() and states==['completed']*3 and code==0
    completed=[json.loads((folder/f'start_{s}.json').read_text()) for s in range(3) if states[s]=='completed']
    candidate=select_start(completed)
    write(folder/'execution.json',{'best_completed_start':None if candidate is None else candidate['selected_start'],'best_completed_penalized_objective':None if candidate is None else candidate['penalized_objective'],'application':name,'elapsed_seconds':time.monotonic()-start,'exit_code':code,'timed_out':timed_out,'start_states':states,'complete_three_start_application':complete,'status':'complete' if complete else 'resource_limited' if timed_out else 'process_failed'})
    return name

def main():
    if sys.argv[1:]!=['--run']:
        if (HERE/'protocol.json').exists() or (HERE/'fits').exists():raise FileExistsError('frozen study exists')
        p=protocol();write(HERE/'protocol.json',p);write(HERE/'protocol_seal.json',{'protocol_sha256':sha(HERE/'protocol.json'),'evaluator_read':False})
        print('Protocol sealed; scientific source approval required before --run');return
    p=json.loads((HERE/'protocol.json').read_text());s=json.loads((HERE/'protocol_seal.json').read_text());approval=json.loads((HERE/'source_approval.json').read_text())
    if s['protocol_sha256']!=sha(HERE/'protocol.json') or p!=protocol() or approval.get('approved') is not True or approval.get('protocol_sha256')!=s['protocol_sha256']:raise ValueError('protocol or scientific approval mismatch')
    dest=HERE/'fits';dest.mkdir(exist_ok=False);grid=list(itertools.product(DATASETS,ASSAYS,(30,300),('original','column_null')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for name in pool.map(app,grid):print(name,flush=True)
    files=tree(dest)
    if len(list(dest.glob('*/execution.json')))!=36:raise ValueError('incomplete execution records')
    write(HERE/'fit_seal.json',{'protocol_sha256':s['protocol_sha256'],'source_approval_sha256':sha(HERE/'source_approval.json'),'artifacts':files,'applications':36,'evaluator_read':False})
if __name__=='__main__':main()
