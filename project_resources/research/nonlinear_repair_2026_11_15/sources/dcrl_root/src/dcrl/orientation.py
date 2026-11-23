"""Distribution-preserving binary-coordinate orientation after fitting.

Accepts sparse or dense latent laws and preserves the input law type and shape.
"""

import numpy as np

from .latent_law import SparseLatentLaw


def orient_columns(B,p,gamma=None,*,zero_tolerance=0.0):
    """Orient negative-sum columns by replacing A_k with 1 - A_k.

    This is a reparameterization, not a constrained refit. The transformed
    intercept and slopes need not lie in the optimizer's original box. No
    column permutation is performed. Zero-sum columns have no chosen sign.
    Dense laws retain their original shape; sparse laws remain sparse.
    """
    B=np.asarray(B,dtype=float)
    if B.ndim!=2 or not np.all(np.isfinite(B)):
        raise ValueError("B must be a finite item-by-(K+1) matrix")
    K=B.shape[1]-1
    if isinstance(p,SparseLatentLaw):
        if p.K!=K:
            raise ValueError("p.K must match the number of slopes in B")
        dense=None
    else:
        dense=np.asarray(p,dtype=float)
        if (dense.ndim not in (1,2) or dense.size!=(1<<K) or
                not np.isfinite(dense).all() or (dense<0).any() or
                not np.isclose(dense.sum(),1.0)):
            raise ValueError("Dense p must be a normalized 2**K-state probability array")
    if not np.isfinite(zero_tolerance) or zero_tolerance<0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    sums=B[:,1:].sum(axis=0)
    flips=np.flatnonzero(sums<-zero_tolerance)
    unresolved=np.flatnonzero(np.abs(sums)<=zero_tolerance)
    oriented=B.copy()
    oriented[:,0]+=B[:,flips+1].sum(axis=1)
    oriented[:,flips+1]*=-1
    mask=sum(1<<(K-1-int(k)) for k in flips)
    if dense is None:
        law=SparseLatentLaw(K,{state^mask: mass for state,mass in p.masses.items()},p.uniform_mass)
    else:
        law=dense.reshape(-1)[np.arange(1<<K)^mask].reshape(dense.shape)
    return {
        "B": oriented,"p": law,
        "gamma": None if gamma is None else np.asarray(gamma).copy(),
        "flipped_columns": flips.tolist(),"unresolved_columns": unresolved.tolist(),
        "zero_sum_columns": unresolved.tolist(),"xor_mask_hex": format(mask,"x"),
        "raw_column_sums": sums.tolist(),"zero_tolerance": float(zero_tolerance),
        "box_preservation_guaranteed": False,
    }
