"""Count-kernel corrections and low-dimensional source orientation.

Observation mean is exposure * Lambda in every sensitivity arm. Conditional
active counts have NB shape theta and mean exposure*Lambda/retention, with an
independent Bernoulli(retention) mask per feature. Poisson means theta=inf.
Factorial order r is divided by prod(1+l/theta) and multiplied by q**(r-1).

The source model is Lambda=mu+B eps with independent unit-variance eps and
nonnegative Lambda throughout its support. Signed B needs compatible support.
All estimates are ddof=0 plug-in moments. No label, source-rank selection,
cross-assay matching, or full feature-cubic tensor is used by this module.

DICA cumulants: Podosinnikova, Bach & Lacoste-Julien, NeurIPS 2015.
Generalized-covariance motivation: same authors, ICML 2016, Beyond CCA.
Exposure/NB/ZI corrections, bounded observed-sample PGF processing and tensor
shrinkage here are local extensions, not inherited causal-identification claims.
"""
from __future__ import annotations

import numpy as np
from scipy.special import hyp1f1, logsumexp

N_DIRECTIONS = 8
MAX_CONDITION = 1e8
SHRINK_GROUPS = 5
SHRINK_GROUP_SEED = 7349
CF_ABSOLUTE_CONDITIONING_TOLERANCE = 1e-8


def _inputs(counts, exposure, theta, retention):
    y = np.asarray(counts, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 1:
        raise ValueError("counts must be n-by-p with n>=2 and p>=1")
    if not np.isfinite(y).all() or (y < 0).any() or not np.equal(y, np.floor(y)).all():
        raise ValueError("counts must be finite nonnegative integers")
    t = np.asarray(exposure, dtype=float)
    if t.ndim == 0:
        t = np.broadcast_to(t, y.shape)
    elif t.shape == (len(y),):
        t = np.broadcast_to(t[:, None], y.shape)
    elif t.shape == (len(y), 1):
        t = np.broadcast_to(t, y.shape)
    if t.shape != y.shape or not np.isfinite(t).all() or (t <= 0).any():
        raise ValueError("exposure must be positive scalar, n-vector, or n-by-p array")
    theta, retention = float(theta), float(retention)
    if np.isnan(theta) or theta <= 0 or theta == -np.inf:
        raise ValueError("theta must be positive or inf")
    if not np.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    return y, t, theta, retention


def _pseudo(y, t, theta, retention, order, correction):
    """Per-feature cumulant polynomials of unbiased factorial raw moments."""
    r = y / t
    out = [r]
    if not correction:
        return out + [np.zeros_like(r) for _ in range(order - 1)]
    if np.isinf(theta) and retention == 1:
        if order >= 2:
            out.append(-y / t**2)
        if order >= 3:
            out.append(2*y / t**3)
        if order >= 4:
            out.append(-6*y / t**4)
        return out
    invtheta = 0. if np.isinf(theta) else 1/theta
    f2 = retention*y*(y-1) / ((1+invtheta)*t**2)
    if order >= 2:
        out.append(f2-r**2)
    if order >= 3:
        f3 = retention**2*y*(y-1)*(y-2) / ((1+invtheta)*(1+2*invtheta)*t**3)
        out.append(f3-3*f2*r+2*r**3)
    if order >= 4:
        f4 = retention**3*y*(y-1)*(y-2)*(y-3) / ((1+invtheta)*(1+2*invtheta)*(1+3*invtheta)*t**4)
        out.append(f4-4*f3*r-3*f2**2+12*f2*r**2-6*r**4)
    return out


def _statistics(y, t, theta, retention, correction):
    r, h2 = _pseudo(y, t, theta, retention, 2, correction)
    mean = r.mean(axis=0)
    constant_mask = np.ptp(r,axis=0) == 0
    # Exact constant ratios can otherwise acquire tiny residuals when summing
    # a noninteger value. Preserve the observed SD=0 and exact zero crossrow.
    mean[constant_mask] = r[0,constant_mask]
    centered = r-mean
    centered[:,constant_mask] = 0
    scale = np.sqrt(np.mean(centered**2, axis=0))
    working_scale = np.where(scale > 0,scale,1.)
    noise = -h2.mean(axis=0)
    observed = centered.T @ centered / len(r)
    covariance = observed-np.diag(noise)
    standard = covariance/np.outer(working_scale,working_scale)
    return dict(mean=mean, scale=scale, covariance=(covariance+covariance.T)/2,
                working_scale=working_scale, constant_mask=constant_mask,
                constant_feature_count=int(constant_mask.sum()),
                standardized_covariance=(standard+standard.T)/2,
                noise_diagonal=noise, poisson_noise_diagonal=np.mean(y/t**2, axis=0),
                observed_covariance=observed, theta=theta, retention=retention,
                correction=bool(correction), moment_convention="finite-sample plug-in; ddof=0")


def count_covariance(counts, exposure, theta=np.inf, correction=True, retention=1.):
    """Raw-count covariance correction for an externally chosen rank procedure."""
    y, t, theta, retention = _inputs(counts, exposure, theta, retention)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _statistics(y, t, theta, retention, correction)


def projected_moments(counts, exposure, unmixing, mean=None, theta=np.inf,
                      correction=True, retention=1.):
    """First four corrected scalar moments, for fixed source directions.

    Contributions are about the supplied fixed center. Returned variance,
    skewness and excess kurtosis subsequently use this cohort's first moment.
    These finite summaries are not complete source-distribution identifiers.
    """
    y, t, theta, retention = _inputs(counts, exposure, theta, retention)
    a = np.asarray(unmixing, dtype=float)
    if a.ndim != 2 or a.shape[1] != y.shape[1] or not np.isfinite(a).all():
        raise ValueError("unmixing must be finite k-by-p")
    mu = np.zeros(y.shape[1]) if mean is None else np.asarray(mean, dtype=float)
    if mu.shape != (y.shape[1],) or not np.isfinite(mu).all():
        raise ValueError("mean must be a finite p-vector")
    r, h2, h3, h4 = _pseudo(y, t, theta, retention, 4, correction)
    z = (r-mu) @ a.T
    u2, u3, u4 = h2 @ (a**2).T, h3 @ (a**3).T, h4 @ (a**4).T
    contributions = np.stack([z, z**2+u2, z**3+3*z*u2+u3,
                              z**4+6*z*z*u2+3*u2*u2+4*z*u3+u4], axis=2)
    moments = contributions.mean(axis=0)
    m1, m2, m3, m4 = moments.T
    variance = m2-m1*m1
    third = m3-3*m1*m2+2*m1**3
    fourth_cumulant = m4-4*m1*m3-3*m2*m2+12*m1*m1*m2-6*m1**4
    valid = np.isfinite(moments).all(axis=1) & (variance > 0)
    skewness, kurtosis = np.full(len(a), np.nan), np.full(len(a), np.nan)
    skewness[valid] = third[valid]/variance[valid]**1.5
    kurtosis[valid] = fourth_cumulant[valid]/variance[valid]**2
    return dict(moment_contributions=contributions, moments=moments, mean=m1,
                variance=variance, skewness=skewness, excess_kurtosis=kurtosis,
                valid_variance=valid)


def _cf_log_weights(y, t, w, mean, frequency, theta, retention):
    """Count-kernel inverse CF, with a conservative numerical diagnostic."""
    phase = -1j*frequency*(w @ mean)
    arg = 1j*frequency*w[None,:]*retention/t
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if np.isinf(theta):
            log_h = y*np.log1p(arg)
            if not np.isfinite(log_h).all():
                return None, np.inf, "nonfinite Poisson kernel logarithm"
            if retention == 1:
                return log_h.sum(axis=1)+phase, 0., ""
            if np.any(log_h.real > 300):
                return None, np.inf, "Poisson kernel exceeds log-weight limit"
            kernel = 1+np.expm1(log_h)/retention
            conditioning_proxy = 0.
        else:
            h = hyp1f1(-y, theta, -theta*arg)
            if not np.isfinite(h).all():
                return None, np.inf, "nonfinite hypergeometric kernel"
            kernel = (h+retention-1)/retention
            # Sum of absolute coefficients of the finite polynomial. This is
            # a conditioning diagnostic, not a SciPy forward-error guarantee.
            term_sum = hyp1f1(-y, theta, -np.abs(theta*arg))
            if not np.isfinite(term_sum).all() or (term_sum < 0).any():
                return None, np.inf, "nonfinite hypergeometric conditioning diagnostic"
            delta = 64*np.finfo(float).eps*np.maximum(1.,y)*(term_sum+1-retention)/retention
            magnitude = np.abs(kernel)
            log_product = np.log(magnitude).sum(axis=1)
            log_inflated = np.log(magnitude+delta).sum(axis=1)
            gap = np.maximum(log_inflated-log_product, 0.)
            log_error = log_inflated+np.log(-np.expm1(-gap))
            log_mean_error = logsumexp(log_error)-np.log(len(y))
            conditioning_proxy = float(np.exp(log_mean_error)) if log_mean_error < 700 else np.inf
            if not np.isfinite(conditioning_proxy) or conditioning_proxy > CF_ABSOLUTE_CONDITIONING_TOLERANCE:
                return None, conditioning_proxy, "absolute product conditioning proxy exceeds 1e-8"
        if not np.isfinite(kernel).all():
            return None, np.inf, "nonfinite transformed count kernel"
        # Exact zero factors give zero cell weights and remain valid. Their
        # logarithms have real part -inf; no CF value is clipped or projected.
        logs = np.log(np.abs(kernel)).sum(axis=1)+1j*np.angle(kernel).sum(axis=1)+phase
        real_valid = np.isfinite(logs.real) | np.isneginf(logs.real)
        if not real_valid.all() or not np.isfinite(logs.imag).all():
            return None, conditioning_proxy, "nonfinite product logarithm"
        return logs, conditioning_proxy, ""


def generalized_cf(counts, exposure, unmixing, mean, theta=np.inf, retention=1.,
                    grid=(.125,.25,.5)):
    """Secondary count-aware source CF with uncertainty and numerical gates.

    Finite theta uses the terminating hypergeometric count-kernel inverse.
    Conditional expectation equals the rate CF when the relevant expectations
    exist. Bounded empirical SE does not prove population integrability.
    The conditioning proxy is heuristic, not a certified floating-point bound.
    """
    y,t,theta,retention = _inputs(counts,exposure,theta,retention)
    a,mu = np.asarray(unmixing,float),np.asarray(mean,float)
    frequencies = np.asarray(grid,float)
    if a.ndim != 2 or a.shape[1] != y.shape[1] or not np.isfinite(a).all():
        raise ValueError("unmixing must be finite k-by-p")
    if mu.shape != (y.shape[1],) or not np.isfinite(mu).all():
        raise ValueError("mean must be a finite p-vector")
    if frequencies.ndim != 1 or len(frequencies) == 0 or not np.isfinite(frequencies).all() or (frequencies <= 0).any():
        raise ValueError("grid must contain finite positive frequencies")
    shape = (len(a),len(frequencies))
    value = np.full(shape,complex(np.nan,np.nan))
    se,max_log,proxy = np.full(shape,np.inf),np.full(shape,np.inf),np.full(shape,np.inf)
    numerical = np.zeros(shape,bool)
    reasons = np.full(shape,"",dtype="<U80")
    for i,w in enumerate(a):
        for j,frequency in enumerate(frequencies):
            logs,proxy[i,j],reason = _cf_log_weights(y,t,w,mu,frequency,theta,retention)
            reasons[i,j] = reason
            if logs is None:
                continue
            max_log[i,j] = float(np.max(logs.real))
            if max_log[i,j] > 300:
                reasons[i,j] = "product exceeds log-weight limit"
                continue
            weights = np.exp(logs)
            if not np.isfinite(weights).all():
                reasons[i,j] = "nonfinite product weight"
                continue
            value[i,j] = weights.mean()
            se[i,j] = np.sqrt(np.mean(np.abs(weights-value[i,j])**2)/(len(y)-1))
            numerical[i,j] = np.isfinite(se[i,j])
    valid = (numerical.all(axis=1) & np.isfinite(value).all(axis=1)
             & (se <= .1).all(axis=1) & (np.abs(value) <= 1.1).all(axis=1)
             & (max_log <= 300).all(axis=1))
    return dict(value=value,se=se,max_logweight=max_log,valid=valid,numerical_valid=numerical,
                mean_absolute_conditioning_proxy=proxy,reason=reasons)


def _symmetrize(tensor):
    return sum(tensor.transpose(p) for p in
               [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)])/6


def _third_projected(r, h2, h3, whitening, mean, rows=None):
    if rows is not None:
        r, h2, h3 = r[rows], h2[rows], h3[rows]
    z = (r-mean) @ whitening.T
    tensor = np.einsum("na,nb,nc->abc", z, z, z, optimize=True)/len(r)
    second_cross = h2.T @ z / len(r)
    correction = np.einsum("pa,pb,pc->abc", whitening.T, whitening.T, second_cross, optimize=True)
    tensor += correction + correction.transpose(0,2,1) + correction.transpose(2,0,1)
    tensor += np.einsum("pa,pb,pc,p->abc", whitening.T, whitening.T, whitening.T, h3.mean(axis=0), optimize=True)
    return _symmetrize(tensor)


def _shrink_tensor(r, h2, h3, whitening, mean, tensor):
    groups = np.array_split(np.random.default_rng(SHRINK_GROUP_SEED).permutation(len(r)), min(SHRINK_GROUPS, len(r)))
    group_tensors = np.array([_third_projected(r, h2, h3, whitening, mean, rows) for rows in groups])
    sizes = np.array([len(rows) for rows in groups])
    variance = np.einsum("g,gabc->abc", sizes, (group_tensors-tensor)**2)/((len(groups)-1)*len(r))
    denominator = tensor**2+variance
    shrunk = np.divide(tensor**3, denominator, out=np.zeros_like(tensor), where=denominator > 0)
    return _symmetrize(shrunk), dict(shrink_variance_proxy=variance, shrink_group_tensors=group_tensors,
                                     shrink_group_sizes=sizes, shrink_group_seed=SHRINK_GROUP_SEED)


def _rotate_tensor(tensor, q):
    return np.einsum("ia,jb,kc,ijk->abc", q, q, q, tensor, optimize=True)


def _tensor_residual(tensor, q):
    rotated = _rotate_tensor(tensor, q)
    off = rotated.copy()
    for j in range(len(q)):
        off[j,j,j] = 0
    norm = np.linalg.norm(tensor)
    return float(np.linalg.norm(off)/(norm if norm else 1.))


def _orient_tensor(tensor, seed):
    k, norm = len(tensor), np.linalg.norm(tensor)
    if not np.isfinite(norm) or (k > 1 and norm <= 1e-12):
        return None, dict(reason="third cumulant has no finite numerical orientation information")
    directions = np.random.default_rng(seed).normal(size=(N_DIRECTIONS, k))
    directions /= np.linalg.norm(directions, axis=1)[:,None]
    best = None
    for index, direction in enumerate(directions):
        contracted = np.einsum("abc,c->ab", tensor, direction)
        values, q = np.linalg.eigh((contracted+contracted.T)/2)
        if k > 1 and np.min(np.diff(values)) <= 1e-12*max(1., np.max(np.abs(values))):
            continue
        score = _tensor_residual(tensor, q)
        if best is None or score < best[0]:
            best = score, index, q
    if best is None:
        return None, dict(reason="all fixed tensor contractions have degenerate spectra")
    return best[2], dict(selected_contraction=best[1], orientation_fit_residual=best[0])


def _pgf_matrices(y, t, mean, whitening, seed):
    """Whitened Hessians of log latent Laplace transform via Poisson PGF.

    Observed training weights are in (0,1]. The cap on observed exposure minima
    is not a statement about an unknown population exposure lower bound.
    Diagonal tilted variances are not comparable cross-assay fingerprints.
    """
    k, p = whitening.shape
    directions = np.abs(np.random.default_rng(seed).normal(size=(N_DIRECTIONS, p)))+0.05
    matrices, points, effective, log_ranges, min_bases = [], [], [], [], []
    for d in directions:
        factor = min(1./float(d @ mean), 0.5*float(np.min(t.min(axis=0)/d)))
        s = -factor*d
        log_weight = np.sum(y*np.log1p(s[None,:]/t), axis=1)
        min_bases.append(float(np.min(1+s[None,:]/t)))
        weight = np.exp(log_weight-np.max(log_weight))
        weight /= weight.sum()
        tilted = (y/(t+s)) @ whitening.T
        tilted_mean = weight @ tilted
        diagonal_correction = weight @ (y/(t+s)**2)
        matrix = tilted.T @ (weight[:,None]*tilted)-np.outer(tilted_mean, tilted_mean)
        matrix -= (whitening*diagonal_correction[None,:]) @ whitening.T
        matrices.append((matrix+matrix.T)/2)
        points.append(s)
        effective.append(1./np.sum(weight**2))
        log_ranges.append([float(log_weight.min()), float(log_weight.max())])
    return np.asarray(matrices), dict(pgf_processing_points=np.asarray(points), pgf_effective_cells=np.asarray(effective),
                                      pgf_log_weight_ranges=np.asarray(log_ranges), pgf_min_observed_base=min(min_bases))


def _orient_matrices(matrices):
    k = matrices.shape[1]
    centered = matrices-np.trace(matrices,axis1=1,axis2=2)[:,None,None]*np.eye(k)[None,:,:]/k
    norm = np.linalg.norm(centered)
    if k == 1:
        return np.eye(1), dict(selected_contraction=0, orientation_fit_residual=0., pgf_anisotropic_norm=float(norm))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None, dict(reason="generalized covariance has no finite anisotropic orientation information", pgf_anisotropic_norm=float(norm))
    best = None
    for index, matrix in enumerate(centered):
        values, q = np.linalg.eigh(matrix)
        if np.min(np.diff(values)) <= 1e-12*max(1., np.max(np.abs(values))):
            continue
        rotated = np.einsum("ia,gij,jb->gab", q, centered, q, optimize=True)
        off = rotated.copy()
        for j in range(k):
            off[:,j,j] = 0
        score = float(np.linalg.norm(off)/norm)
        if best is None or score < best[0]:
            best = score, index, q
    if best is None:
        return None, dict(reason="all generalized covariance spectra are degenerate", pgf_anisotropic_norm=float(norm))
    return best[2], dict(selected_contraction=best[1], orientation_fit_residual=best[0], pgf_anisotropic_norm=float(norm))


def fit_counts(counts, exposure, k, theta=np.inf, orientation="c3", correction=True,
               seed=0, retention=1.):
    """Fit externally selected k<=p source directions using all feature columns.

    Negative discarded covariance modes are reported. Invalid requested modes,
    excessive condition or unavailable orientation information cause abstention.
    No positive-definite ridge or source-count substitution is applied.
    c3_shrink's eligibility residual uses the original, unshrunk C3 tensor.
    """
    y, t, theta, retention = _inputs(counts, exposure, theta, retention)
    p = y.shape[1]
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k <= p:
        raise ValueError("k must be an integer between zero and p")
    if orientation not in {"c3", "c3_shrink", "pgf"}:
        raise ValueError("orientation must be c3, c3_shrink or pgf")
    if orientation == "pgf" and (not np.isinf(theta) or retention != 1 or not correction):
        raise ValueError("Poisson PGF orientation requires theta=inf, retention=1, correction=True")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        stats = _statistics(y, t, theta, retention, correction)
    result = dict(stats, status="abstained", reason=None, k_requested=int(k), k_fitted=0,
                  orientation=orientation, mixing=np.empty((p,0)), unmixing=np.empty((0,p)),
                  eigenvalues=np.array([]), condition_number=None, source_moments=None,
                  source_skewness=np.array([]), source_excess_kurtosis=np.array([]), source_variance=np.array([]),
                  orientation_residual=None, orientation_fit_residual=None, third_offdiagonal_residual=None,
                  white_third_tensor=None, orientation_target_tensor=None)
    if not all(np.isfinite(stats[key]).all() for key in ["mean", "scale", "covariance", "standardized_covariance", "noise_diagonal"]):
        result["reason"] = "nonfinite estimated count moments"
        return result
    try:
        values, vectors = np.linalg.eigh(stats["standardized_covariance"])
    except np.linalg.LinAlgError as error:
        result["reason"] = "covariance eigendecomposition failed: " + str(error)
        return result
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:,order]
    result.update(eigenvalues=values, negative_eigenvalue_count=int(np.sum(values < 0)))
    if k == 0:
        result.update(status="ok", reason="zero sources requested")
        result["source_moments"] = projected_moments(y, t, result["unmixing"], stats["mean"], theta, correction, retention)
        return result
    if values[k-1] <= 0 or values[0] <= 0:
        result["reason"] = "requested covariance modes are not positive"
        return result
    condition = float(values[0]/values[k-1])
    result["condition_number"] = condition
    if not np.isfinite(condition) or condition > MAX_CONDITION:
        result["reason"] = "retained covariance condition exceeds 1e8"
        return result
    v, eigen = vectors[:,:k], values[:k]
    whitening = (v/np.sqrt(eigen)).T/stats["working_scale"][None,:]
    result["whitening"] = whitening
    try:
        r, h2, h3 = _pseudo(y, t, theta, retention, 3, correction)
        tensor = _third_projected(r, h2, h3, whitening, stats["mean"])
        result["white_third_tensor"] = tensor
        if orientation == "pgf":
            matrices, metadata = _pgf_matrices(y, t, stats["mean"], whitening, seed)
            result.update(metadata, generalized_covariance_matrices=matrices)
            q, info = _orient_matrices(matrices)
        else:
            target = tensor
            if orientation == "c3_shrink":
                target, metadata = _shrink_tensor(r, h2, h3, whitening, stats["mean"], tensor)
                result.update(metadata)
            result["orientation_target_tensor"] = target
            q, info = _orient_tensor(target, seed)
        result.update(info)
        if q is None:
            return result
        mixing = (stats["working_scale"][:,None]*v*np.sqrt(eigen)) @ q
        unmixing = q.T @ whitening
        signs = np.sign(mixing[np.argmax(np.abs(mixing),axis=0),np.arange(k)])
        signs[signs == 0] = 1
        mixing *= signs
        unmixing *= signs[:,None]
        q *= signs
        raw_residual = _tensor_residual(tensor, q)
        moments = projected_moments(y, t, unmixing, stats["mean"], theta, correction, retention)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
        return result
    if not moments["valid_variance"].all() or not np.isfinite(mixing).all() or not np.isfinite(unmixing).all():
        result["reason"] = "invalid estimated source moments or directions"
        return result
    result.update(status="ok", reason=None, k_fitted=int(k), mixing=mixing, unmixing=unmixing,
                  source_moments=moments, source_skewness=moments["skewness"], source_excess_kurtosis=moments["excess_kurtosis"],
                  source_variance=moments["variance"], third_offdiagonal_residual=raw_residual,
                  orientation_residual=result["orientation_fit_residual"] if orientation == "pgf" else raw_residual)
    return result
