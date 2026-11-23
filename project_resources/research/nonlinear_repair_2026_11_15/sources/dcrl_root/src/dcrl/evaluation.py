import numpy as np
from collections.abc import Mapping
from causallearn.search.ScoreBased.GES import ges

from .latent_law import SparseLatentLaw


def stitch_full_A(ZZ,ZX,n_x):
    n_z=ZZ.shape[0]
    assert ZX.shape==(n_z,n_x)
    A=np.block([
        [ZZ,ZX],
        [np.zeros((n_x,n_z),dtype=int),np.zeros((n_x,n_x),dtype=int)],
    ])
    np.fill_diagonal(A,0)
    return A


def shd_cpdag(M1,M2):
    n=M1.shape[0]

    def code(M,i,j):
        a,b=M[i,j],M[j,i]
        if a==0 and b==0:
            return 0
        if a==-1 and b==-1:
            return 2
        if b==1 and a==-1:
            return 1
        if a==1 and b==-1:
            return -1
        return 2

    shd=0
    for i in range(n):
        for j in range(i+1,n):
            c1=code(M1,i,j)
            c2=code(M2,i,j)
            if c1!=c2:
                shd+=1
    return shd


def sample_latent_states_from_p(p_hat,K,sample_size):
    if isinstance(p_hat,SparseLatentLaw):
        if p_hat.K!=K:
            raise ValueError('K does not match the latent law')
        return p_hat.sample(sample_size)
    if isinstance(p_hat,Mapping):
        return SparseLatentLaw.from_counts(K,p_hat).sample(sample_size)

    # Backward compatibility for already-dense input: decode sampled states
    # only, rather than constructing the full 2**K by K configuration array.
    sam=np.zeros((sample_size,K))
    qwe=np.asarray(p_hat).reshape(-1)
    if qwe.size!=1<<K:
        raise ValueError('Dense probability vector must have length 2**K')
    counts=np.random.multinomial(sample_size,qwe)
    n=0
    for a in np.flatnonzero(counts):
        cnt=int(counts[a])
        state=int(a)
        sam[n:n+cnt]=[(state>>shift)&1 for shift in range(K-1,-1,-1)]
        n+=cnt
    return sam


def recover_full_graph_from_estimates(p_hat,B_hat,K,J,sample_size,*,
                                      method="resampling",ess=None,
                                      structure_prior=None,seed=None,
                                      max_parents=None):
    """Stitch the latent CPDAG and loading support using optional graph tuning.

    ``sample_size`` is the effective count size in plugin mode. With no
    options and K>=3, the historical sampling/GES path is unchanged.
    """
    if method=="resampling" and ess is None and structure_prior is None and seed is None and max_parents is None and K>=3:
        # Preserve the historical default path and random-number consumption.
        sam=sample_latent_states_from_p(p_hat,K,sample_size)
        record=ges(sam,score_func="local_score_BDeu")
        cpdag_ZZ_est=record["G"].graph
    else:
        from .graph_scoring import learn_latent_graph
        record=learn_latent_graph(p_hat,K,sample_size,method=method,
                                    ess=ess,structure_prior=structure_prior,
                                    seed=seed,max_parents=max_parents)
        cpdag_ZZ_est=record["graph"]
    ZX_est=(B_hat[:,1:]!=0).astype(int).T
    return stitch_full_A(cpdag_ZZ_est,ZX_est,n_x=J)


def compute_shd_triplet(p_hat,B_hat,K,J,Q_N,truth_graph,Q_truth,*,
                        method="resampling",ess=None,structure_prior=None,
                        seed=None,max_parents=None):
    """SHD at effective sizes N, 2N and 3N.

    An explicit seed restarts the local sampler at that seed for each size.
    With seed=None, the original sequential global RNG consumption remains.
    """
    A_truth=stitch_full_A(truth_graph.graph,Q_truth.T,n_x=J)
    options=dict(method=method,ess=ess,structure_prior=structure_prior,
                   seed=seed,max_parents=max_parents)
    A_est_1=recover_full_graph_from_estimates(p_hat,B_hat,K,J,Q_N,**options)
    A_est_2=recover_full_graph_from_estimates(p_hat,B_hat,K,J,2*Q_N,**options)
    A_est_3=recover_full_graph_from_estimates(p_hat,B_hat,K,J,3*Q_N,**options)
    shd_val=shd_cpdag(A_truth,A_est_1)
    shd2_val=shd_cpdag(A_truth,A_est_2)
    shd3_val=shd_cpdag(A_truth,A_est_3)
    return shd_val,shd2_val,shd3_val
