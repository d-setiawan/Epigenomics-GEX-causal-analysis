"""Count-source estimation from Poisson-corrected factorial cumulants.

Working model: Y_ij | t_i, lambda_i ~ independent Poisson(t_i*lambda_ij),
lambda_i = mu + B*eps_i, with independent unit-variance source coordinates.
Signed B requires source support that keeps every rate nonnegative. An observed
library total is not automatically an exogenous exposure, and feature-specific
NB overdispersion is not removed by these Poisson corrections.

The DICA cumulant construction follows Podosinnikova, Bach and Lacoste-Julien,
NeurIPS 2015, "Rethinking LDA: Moment Matching for Discrete ICA". Exposure
adjustment and source fingerprint/calibration helpers are local extensions.
All moments are finite-sample plug-in moments (ddof=0), not unbiased k-statistics.
No row matching, source count selection, labels or native Sturma code is used.
"""
from __future__ import annotations

import numpy as np

MAX_CONDITION = 1e8
N_DIRECTIONS = 8


def _inputs(counts, exposure):
    y = np.asarray(counts, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 1:
        raise ValueError("counts must be a nonempty n-by-p matrix with n>=2")
    if not np.isfinite(y).all() or (y < 0).any() or not np.equal(y, np.floor(y)).all():
        raise ValueError("counts must be finite nonnegative integers")
    t = np.asarray(exposure, dtype=float)
    if t.ndim == 0:
        t = np.full(y.shape[0], float(t))
    if t.shape != (y.shape[0],) or not np.isfinite(t).all() or (t <= 0).any():
        raise ValueError("exposure must be finite and strictly positive for every row")
    return y, t


def _cumulants(y, t, correction):
    n, p = y.shape
    u = y / t[:, None]
    mean = u.mean(axis=0)
    scale = u.std(axis=0, ddof=0)
    poisson_noise = np.mean(y / t[:, None]**2, axis=0)
    noise = poisson_noise if correction else np.zeros(p)
    second = u.T @ u / n - np.diag(noise)
    third = np.einsum("ni,nj,nk->ijk", u, u, u, optimize=True) / n
    if correction:
        repeated_pair = np.einsum("ni,nj,n->ij", u, u, 1/t, optimize=True) / n
        repeated_triple = np.mean(y / t[:, None]**3, axis=0)
        for j in range(p):
            third[j, j, :] -= repeated_pair[j, :]
            third[j, :, j] -= repeated_pair[j, :]
            third[:, j, j] -= repeated_pair[j, :]
            third[j, j, j] += 2*repeated_triple[j]
    covariance = second - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    third -= np.einsum("i,jk->ijk", mean, second)
    third -= np.einsum("j,ik->ijk", mean, second)
    third -= np.einsum("k,ij->ijk", mean, second)
    third += 2*np.einsum("i,j,k->ijk", mean, mean, mean)
    # Symmetrization removes only floating-point contraction asymmetry.
    third = sum(third.transpose(order) for order in
                [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]) / 6
    return mean, scale, covariance, third, poisson_noise, noise


def projected_factorial_moments(counts, exposure, unmixing, mean=None, correction=True):
    """Estimate moments for fixed source directions on any permitted cohort.

    `mean` is a p-vector on the count/exposure scale, usually the fitted training
    mean. Contributions are n-by-k-by-4 moments about that fixed center. Returned
    variance/skewness/excess_kurtosis are subsequently centered using this
    cohort's estimated first moment. Directions and centering are never refitted.
    Fingerprints are finite moment summaries, not a full source-law identifier.
    """
    y, t = _inputs(counts, exposure)
    a = np.asarray(unmixing, dtype=float)
    if a.ndim != 2 or a.shape[1] != y.shape[1] or not np.isfinite(a).all():
        raise ValueError("unmixing must be a finite k-by-p matrix")
    center = np.zeros(y.shape[1]) if mean is None else np.asarray(mean, dtype=float)
    if center.shape != (y.shape[1],) or not np.isfinite(center).all():
        raise ValueError("mean must be a finite p-vector on the rate scale")
    z = (y / t[:, None] - center) @ a.T
    contributions = np.empty((len(y), len(a), 4))
    contributions[:, :, 0] = z
    if correction:
        q2 = (y @ (a**2).T) / t[:, None]**2
        q3 = (y @ (a**3).T) / t[:, None]**3
        q4 = (y @ (a**4).T) / t[:, None]**4
        contributions[:, :, 1] = z**2-q2
        contributions[:, :, 2] = z**3-3*z*q2+2*q3
        contributions[:, :, 3] = z**4-6*z**2*q2+3*q2**2+8*z*q3-6*q4
    else:
        for order in [2,3,4]:
            contributions[:, :, order-1] = z**order
    moments = contributions.mean(axis=0)
    m1, m2, m3, m4 = moments.T
    variance = m2-m1**2
    c3 = m3-3*m1*m2+2*m1**3
    c4 = m4-4*m1*m3-3*m2**2+12*m1**2*m2-6*m1**4
    valid = np.isfinite(moments).all(axis=1) & (variance > 0)
    skewness = np.full(len(a), np.nan)
    kurtosis = np.full(len(a), np.nan)
    skewness[valid] = c3[valid] / variance[valid]**1.5
    kurtosis[valid] = c4[valid] / variance[valid]**2
    return {"moment_contributions": contributions, "moments": moments,
            "mean": m1, "variance": variance, "skewness": skewness,
            "excess_kurtosis": kurtosis, "valid_variance": valid}


def _spectral_sources(covariance, third, scale, k, seed):
    """Whiten the retained corrected covariance; diagonalize a cubic tensor."""
    standard = covariance / np.outer(scale, scale)
    values, vectors = np.linalg.eigh((standard + standard.T)/2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    info = {"eigenvalues": values,
            "negative_eigenvalue_count": int(np.sum(values < 0)),
            "condition_number": None}
    if values[0] <= 0 or values[k-1] <= 0:
        return {**info, "status": "abstained", "reason": "requested covariance modes are not positive"}
    condition = values[0]/values[k-1]
    info["condition_number"] = float(condition)
    if not np.isfinite(condition) or condition > MAX_CONDITION:
        return {**info, "status": "abstained", "reason": "retained covariance condition exceeds 1e8"}
    v, eigen = vectors[:, :k], values[:k]
    whitening = (v/np.sqrt(eigen)).T / scale[None, :]
    white_third = np.einsum("ai,bj,ck,ijk->abc", whitening, whitening, whitening, third, optimize=True)
    norm = np.linalg.norm(white_third)
    if not np.isfinite(norm):
        return {**info, "status": "abstained", "reason": "nonfinite whitened third cumulant"}
    if k > 1 and norm <= 1e-12:
        return {**info, "status": "abstained", "reason": "third cumulant has no numerical orientation information"}
    directions = np.random.default_rng(seed).normal(size=(N_DIRECTIONS, k))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    best = None
    for index, direction in enumerate(directions):
        contraction = np.einsum("ijk,k->ij", white_third, direction)
        eig, q = np.linalg.eigh((contraction + contraction.T)/2)
        if k > 1 and np.min(np.diff(eig)) <= 1e-12*max(1., np.max(np.abs(eig))):
            continue
        rotated = np.einsum("ia,jb,kc,ijk->abc", q, q, q, white_third, optimize=True)
        diag = np.array([rotated[j,j,j] for j in range(k)])
        diagonal = np.zeros_like(rotated)
        for j in range(k): diagonal[j,j,j] = diag[j]
        residual = np.linalg.norm(rotated-diagonal)/(norm if norm else 1.)
        if best is None or residual < best[0]:
            best = (float(residual), index, q)
    if best is None:
        return {**info, "status": "abstained", "reason": "all fixed tensor contractions have degenerate spectra"}
    _, selected, q = best
    mixing = (scale[:, None]*v*np.sqrt(eigen)) @ q
    unmixing = q.T @ whitening
    # Fix arbitrary sign without using labels, paired information or source truth.
    signs = np.sign(mixing[np.argmax(np.abs(mixing), axis=0), np.arange(k)])
    signs[signs == 0] = 1
    mixing *= signs
    unmixing *= signs[:, None]
    q *= signs
    source_third = np.einsum("ia,jb,kc,ijk->abc", q, q, q, white_third, optimize=True)
    diag = np.array([source_third[j,j,j] for j in range(k)])
    reconstruction = np.einsum("ir,jr,kr,r->ijk", mixing, mixing, mixing, diag, optimize=True)
    offdiagonal = source_third.copy()
    for j in range(k): offdiagonal[j,j,j] = 0
    third_norm = np.linalg.norm(third)
    return {**info, "status": "ok", "reason": None, "mixing": mixing,
            "unmixing": unmixing, "selected_contraction": selected,
            "third_tensor_reconstruction": reconstruction,
            "third_offdiagonal_residual": float(np.linalg.norm(offdiagonal)/(norm if norm else 1.)),
            "third_reconstruction_residual": float(np.linalg.norm(third-reconstruction)/(third_norm if third_norm else 1.))}


def fit_counts(counts, exposure, k, correction=True, seed=0):
    """Fit k externally selected sources; never select features or force sharing.

    Negative discarded sample covariance eigenmodes are reported, not ridged.
    Invalid requested modes, condition>1e8 or unavailable cubic orientation cause
    abstention. A valid k=0 request returns empty matrices and status='ok'.
    """
    y, t = _inputs(counts, exposure)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0 or k > y.shape[1]:
        raise ValueError("k must be an integer between zero and the feature count")
    p = y.shape[1]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean, scale, covariance, third, poisson_noise, noise = _cumulants(y, t, correction)
        outer_scale = np.outer(scale, scale)
        standard = np.divide(covariance, outer_scale, out=np.zeros_like(covariance), where=outer_scale > 0)
    result = {"status": "abstained", "reason": None, "k_requested": int(k), "k_fitted": 0,
              "correction": bool(correction), "mean": mean, "scale": scale,
              "covariance": covariance, "standardized_covariance": standard,
              "poisson_noise_diagonal": poisson_noise, "noise_diagonal": noise,
              "eigenvalues": np.array([]), "condition_number": None,
              "mixing": np.empty((p,0)), "unmixing": np.empty((0,p)),
              "source_skewness": np.array([]), "source_excess_kurtosis": np.array([]),
              "source_variance": np.array([]), "source_moments": None,
              "third_tensor": third, "third_tensor_reconstruction": None,
              "third_offdiagonal_residual": None, "third_reconstruction_residual": None,
              "moment_convention": "finite-sample plug-in; ddof=0"}
    if not all(np.isfinite(x).all() for x in [mean, scale, covariance, third, poisson_noise]):
        result["reason"] = "nonfinite estimated moments"
        return result
    if k == 0:
        result.update(status="ok", reason="zero sources requested")
        result["source_moments"] = projected_factorial_moments(y, t, result["unmixing"], mean, correction)
        result["eigenvalues"] = np.linalg.eigvalsh(standard)[::-1]
        return result
    if np.any(scale <= 0):
        result["reason"] = "constant observed normalized-count feature; no feature dropped"
        return result
    try:
        decomposition = _spectral_sources(covariance, third, scale, k, seed)
    except np.linalg.LinAlgError as error:
        result["reason"] = "linear algebra failure: " + str(error)
        return result
    result.update(decomposition)
    if result["status"] != "ok": return result
    moments = projected_factorial_moments(y, t, result["unmixing"], mean, correction)
    if not moments["valid_variance"].all():
        result.update(status="abstained", reason="invalid corrected source variance",
                      mixing=np.empty((p,0)), unmixing=np.empty((0,p)))
        return result
    result.update(k_fitted=int(k), source_moments=moments,
                  source_skewness=moments["skewness"],
                  source_excess_kurtosis=moments["excess_kurtosis"],
                  source_variance=moments["variance"])
    return result


def align_mixing(reference, candidate):
    """Align bootstrap columns by absolute cosine; not a cross-assay match test."""
    from scipy.optimize import linear_sum_assignment
    ref, cand = np.asarray(reference, float), np.asarray(candidate, float)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape[0] != cand.shape[0]:
        raise ValueError("mixing matrices must share their feature rows")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("mixing matrices must be finite")
    norms = np.outer(np.linalg.norm(ref,axis=0), np.linalg.norm(cand,axis=0))
    cosine = np.divide(ref.T@cand, norms, out=np.zeros_like(norms), where=norms>0)
    rows, cols = linear_sum_assignment(-np.abs(cosine))
    signs = np.where(cosine[rows,cols] < 0, -1., 1.)
    return {"reference_indices": rows, "candidate_indices": cols, "signs": signs,
            "absolute_cosine": np.abs(cosine[rows,cols]), "aligned_mixing": cand[:,cols]*signs}
