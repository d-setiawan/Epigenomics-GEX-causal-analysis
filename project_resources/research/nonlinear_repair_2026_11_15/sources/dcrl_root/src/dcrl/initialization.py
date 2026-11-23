from collections import Counter

import numpy as np
from scipy.linalg import inv
from scipy.special import logit
from factor_analyzer import Rotator

from .utils import thres


def _empirical_latent_probabilities(A,K,*,sparse):
    """Count observed states without constructing the binary state space.

    Integer keys use the same most-significant-bit-first encoding as PSAEM.
    Python integers also avoid overflow when the initializer is used alone
    with more coordinates than fit in a machine word.
    """
    counts=Counter()
    for row in A:
        state=0
        for bit in row:
            state=(state<<1)|int(bit)
        counts[state]+=1

    probabilities={state: count/len(A) for state,count in counts.items()}
    if sparse:
        return probabilities

    # Compatibility for callers that explicitly need a dense probability law.
    dense=np.zeros((1<<K,1),dtype=float)
    for state,probability in probabilities.items():
        dense[state,0]=probability
    return dense


def initialize_parameters(X,J,K,distribution,epsilon,*,sparse=False,cutoff_c=2.5):
    """Spectral initialization with an optional sparse initial latent law.

    ``sparse=True`` returns a dictionary of encountered encoded states and
    probabilities, with missing states having probability zero. It avoids
    every allocation proportional to 2**K. The default preserves the dense
    return format used by external callers. B, gamma and A are identical in
    both modes; PSAEM uses the sparse mode.
    """
    if not np.isfinite(cutoff_c) or cutoff_c<=0:
        raise ValueError("cutoff_c must be finite and positive")
    gamma_in=np.zeros(J)

    if distribution=="Poisson":
        U,S,Vt=np.linalg.svd(np.log(X+1),full_matrices=False)
        m=max(K+1,np.sum(S>1.01*np.sqrt(X.shape[0])))
        X_top_m=U[:,:m]@np.diag(S[:m])@Vt[:m,:]
        X_top_m=np.maximum(X_top_m,epsilon)
        X_inv=np.exp(X_top_m)-1
    elif distribution=="Lognormal":
        X_inv=np.log(X)
    elif distribution=="Bernoulli":
        U,S,Vt=np.linalg.svd(X,full_matrices=False)
        m=max(K+1,np.sum(S>1.01*np.sqrt(X.shape[0])))
        X_top_m=U[:,:m]@np.diag(S[:m])@Vt[:m,:]
        X_top_m=np.maximum(X_top_m,epsilon)
        X_top_m=np.minimum(X_top_m,1-epsilon)
        X_inv=logit(X_top_m)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    X_inv_adj=X_inv-np.mean(X_inv,axis=0)
    _,_,V_adj=np.linalg.svd(X_inv_adj,full_matrices=False)
    V_adj=V_adj.T

    rotator=Rotator(method="varimax")
    R_V=rotator.fit_transform(V_adj[:,:K])

    threshold=1/(cutoff_c*np.sqrt(J))
    B_est=thres(R_V[:,:K],threshold)

    mean_per_column=B_est.mean(axis=0)
    sign_flip=2*(mean_per_column>0)-1
    B_est=B_est*sign_flip

    G_est=(B_est!=0).astype(int)
    col_perm=np.zeros(K,dtype=int)
    remaining_cols=list(range(K))
    for k in range(K):
        tmp=np.argmax(np.sum(G_est[[k,K+k,2*K+k],:][:,remaining_cols],axis=0))
        col_perm[k]=remaining_cols[tmp]
        del remaining_cols[tmp]

    B_est=B_est[:,col_perm]
    G_est=(B_est!=0).astype(float)
    A_est=X_inv_adj@B_est@inv(B_est.T@B_est)
    A_est=(A_est>0).astype(float)

    A_centered=A_est-np.ones((X.shape[0],1))*np.mean(A_est,axis=0)
    B_re_est=(inv(A_centered.T@A_centered)@A_centered.T@X_inv_adj).T
    B_re_est=thres(B_re_est*G_est,0)

    if distribution=="Poisson":
        b=np.mean(X_inv,axis=0)-B_re_est@np.mean(A_est,axis=0)
    else:
        b=np.mean(X_inv_adj,axis=0)-B_re_est@np.mean(A_est,axis=0)

    B_ini=np.column_stack((b,B_re_est))

    nu_in=_empirical_latent_probabilities(A_est,K,sparse=sparse)

    A_long=np.hstack((np.ones((X.shape[0],1)),A_est))
    Tr=X_inv-A_long@B_ini.T
    for j in range(J):
        gamma_in[j]=np.sum(Tr[:,j]**2)/X.shape[0]

    return nu_in,B_ini,gamma_in.reshape(-1,1),A_est
