"""
Cross-mark structure as a covariance-completion problem.

THE PROBLEM, STATED EXACTLY
---------------------------
Six histone marks are six separate assays on separate cells.  A metacell holds
some cells from every assay, and variable j is averaged over that metacell's
channel-ch(j) cells only.  Write

    X[c, j] = mean over cells i in M(c, ch(j)) of V[i, j]
            = mu[c, j] + xi[c, j]

where mu[c, j] = E[V_j | metacell c] and xi is the sampling deviation of the
mean.  Two cells sets are either identical (same channel) or disjoint (cross
channel), so

    Cov(xi_j, xi_k)  =  Sigma_within[j, k] / n(c, ch)   if ch(j) == ch(k)
                     =  0                               otherwise.

Therefore the covariance of the metacell matrix that the pipeline actually
feeds to FCI/BOSS is

    Cov(X_j, X_k) = Sigma_between[j, k] + 1{same channel} * E_c[ Sw[j,k]/n(c) ]

**Same-channel and cross-channel entries are not on the same scale.**  Every
same-channel entry carries an extra positive term that no cross-channel entry
carries.  That asymmetry is a property of the assay design, not of chromatin,
and it is sufficient on its own to make same-mark edges easier to find than
cross-mark edges.

WHY NAIVE DECONVOLUTION DOES NOT FIX IT
---------------------------------------
The obvious repair -- subtract the extra term from the same-channel blocks and
run the search on Sigma_between -- fails, and the reason is worth stating.
mu[c] = E[V | c] is a deterministic function of the latent state, so
Sigma_between has rank <= d (the state dimension).  Subtracting the within term
removes precisely the noise floor that was keeping the aggregated covariance
invertible, and drives the matrix toward singularity.  Partial correlations at
depth then fail.  Cross-mark structure cannot be recovered by rescaling the
between-cluster signal, because the between-cluster signal is a d-factor model
and a p-variable DAG is not identified from it.

THE REFRAME
-----------
Decompose the *cell-level* covariance, which is the object the causal claim is
actually about:

    Sigma_total = Sigma_between + Sigma_within

    same-channel (j, k):  BOTH terms observed.  Sigma_within[j,k] is estimable
                          directly, from thousands of cells, with the metacell
                          (hence the latent state) held fixed.
    cross-channel (j, k): ONLY Sigma_between observed.  Sigma_within[j,k] is
                          structurally unobservable -- it would require the two
                          marks measured on the SAME cell, which the assay
                          never does.

So the cell-level covariance is **partially observed**, with a known and
precisely delimited missing block, and cross-mark identifiability is exactly
the question of whether that block can be completed.

Note what the same-channel case buys, independently of any completion:
Sigma_within conditions on the metacell, i.e. on the latent cell state.  It is
the covariance with the confounder held fixed, estimated at cell resolution
(N ~ thousands) rather than metacell resolution (n ~ 10^2).  That is both a
deconfounding step and a large power gain, and the current pipeline discards it
entirely by averaging.

THE COMPLETION, AND HOW IT CAN BE FALSIFIED
-------------------------------------------
Assume within-state and between-state covariation are driven by the same
mechanisms up to a scale factor:

    Sigma_within[j, k] ~ c * Sigma_between[j, k]

This is an assumption, but it is not an article of faith: on same-channel pairs
BOTH sides are observed, so c can be fit on the same-channel entries and the
fit can be checked -- including leave-one-channel-out, which asks whether a c
fit on five marks predicts the sixth.  `fit_proportionality` reports that R^2.

  - Fit good  -> extrapolate to the cross-channel entries, complete
                 Sigma_total, and the cell-level PAG is identified under the
                 stated assumption.
  - Fit poor  -> the completion is unjustified and we say so.  That is a real
                 answer to "is cross-mark structure identifiable here", not a
                 failure to produce one.

The completed matrix is fed to any LV search through a Tetrad CovarianceMatrix
(`IndTestFisherZ` and `SemBicScore` both take `ICovarianceMatrix`), so this
changes the input the search sees rather than the search itself.

Pure numpy; no Tetrad import at module level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------

@dataclass
class Components:
    """The observable pieces of the cell-level covariance decomposition."""
    between: np.ndarray        # (p, p) deconvolved between-cluster covariance
    within: np.ndarray         # (p, p) pooled within-cluster; NaN cross-channel
    observed: np.ndarray       # (p, p) bool: True where `within` is estimable
    raw_aggregate: np.ndarray  # (p, p) covariance of the metacell mean matrix
    noise_term: np.ndarray     # (p, p) the E_c[Sw/n] correction that was removed
    n_clusters: int
    n_cells_per_channel: np.ndarray
    within_dof: int


def pooled_within_covariance(V, labels, channel_cells, ch, min_cells=2):
    """
    Pooled within-cluster covariance of the cell-level variables.

    Entry (j, k) is estimable only when ch(j) == ch(k): the two variables must
    have been measured on the same cells for a within-cluster cross-product to
    exist at all.  Cross-channel entries are returned as NaN and flagged False
    in `observed` -- they are the missing block, not a value to impute here.

    Conditioning on the cluster is what makes this a deconfounded quantity: the
    latent state is (approximately) held fixed, so what is left is cell-level
    covariation at fixed state.
    """
    V = np.asarray(V, dtype=float)
    p = V.shape[1]
    Sw = np.full((p, p), np.nan)
    observed = np.zeros((p, p), dtype=bool)

    uniq = np.unique(labels)
    n_channels = len(channel_cells)

    for a in range(n_channels):
        vars_a = np.flatnonzero(ch == a)
        if vars_a.size == 0:
            continue
        cells_a = np.asarray(channel_cells[a])
        lab_a = labels[cells_a]

        acc = np.zeros((vars_a.size, vars_a.size))
        dof = 0
        for c in uniq:
            m = cells_a[lab_a == c]
            if m.size < min_cells:
                continue
            blk = V[np.ix_(m, vars_a)]
            blk = blk - blk.mean(axis=0, keepdims=True)
            acc += blk.T @ blk
            dof += m.size - 1
        if dof > 0:
            Sw[np.ix_(vars_a, vars_a)] = acc / dof
            observed[np.ix_(vars_a, vars_a)] = True

    return Sw, observed, dof


def decompose(V, labels, channel_cells, ch, X=None, min_cells=2):
    """
    Split the metacell covariance into the between component and the
    same-channel sampling-noise term that inflates it.

    Returns a `Components` carrying every observable piece, so that downstream
    code never has to re-derive which entries are real and which are missing.
    """
    V = np.asarray(V, dtype=float)
    p = V.shape[1]
    ch = np.asarray(ch)

    if X is None:
        X, _ = aggregate(V, labels, channel_cells, ch)

    raw = np.cov(X, rowvar=False)
    Sw, observed, dof = pooled_within_covariance(V, labels, channel_cells, ch,
                                                 min_cells=min_cells)

    # E_c[1 / n(c, a)] per channel: the factor multiplying Sw in the same-channel
    # inflation.  Averaged over the clusters that actually contributed rows.
    uniq = np.unique(labels)
    mean_inv_n = np.zeros(len(channel_cells))
    n_per_ch = np.zeros(len(channel_cells))
    for a, cells_a in enumerate(channel_cells):
        cells_a = np.asarray(cells_a)
        lab_a = labels[cells_a]
        sizes = np.array([np.sum(lab_a == c) for c in uniq], dtype=float)
        sizes = sizes[sizes > 0]
        mean_inv_n[a] = np.mean(1.0 / sizes) if sizes.size else np.nan
        n_per_ch[a] = sizes.mean() if sizes.size else np.nan

    noise = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            if ch[j] == ch[k] and observed[j, k]:
                noise[j, k] = Sw[j, k] * mean_inv_n[ch[j]]

    between = raw - noise
    return Components(between=between, within=Sw, observed=observed,
                      raw_aggregate=raw, noise_term=noise,
                      n_clusters=X.shape[0], n_cells_per_channel=n_per_ch,
                      within_dof=dof)


def aggregate(V, labels, channel_cells, ch, require_complete=True):
    """
    The pipeline's own aggregation: variable j averaged over its channel's
    cells within each cluster.  A cluster contributes a row only if it holds at
    least one cell from every channel (the completeness rule, itself an
    uncorrected selection).
    """
    V = np.asarray(V, dtype=float)
    p = V.shape[1]
    ch = np.asarray(ch)
    channel_cells = [np.asarray(c) for c in channel_cells]
    labs = [labels[c] for c in channel_cells]

    rows, kept, dropped = [], [], 0
    for c in np.unique(labels):
        members = [channel_cells[a][labs[a] == c] for a in range(len(channel_cells))]
        if require_complete and any(m.size == 0 for m in members):
            dropped += 1
            continue
        row = np.empty(p)
        ok = True
        for j in range(p):
            m = members[ch[j]]
            if m.size == 0:
                ok = False
                break
            row[j] = V[m, j].mean()
        if not ok:
            dropped += 1
            continue
        rows.append(row)
        kept.append(c)

    return np.asarray(rows), dict(n_kept=len(rows), n_dropped=dropped,
                                  kept_labels=np.asarray(kept))


# --------------------------------------------------------------------------
# the structural assumption, and its falsification test
# --------------------------------------------------------------------------

@dataclass
class ProportionalityFit:
    scale: float
    r2_in_sample: float
    r2_loco: float          # leave-one-channel-out: c fit on other marks
    per_channel_scale: dict
    n_pairs: int

    def credible(self, r2_min=0.5):
        return bool(np.isfinite(self.r2_loco) and self.r2_loco >= r2_min)


def fit_proportionality(comp: Components, ch, exclude_diagonal=True):
    """
    Fit Sigma_within ~ c * Sigma_between on the entries where BOTH are
    observed (i.e. the same-channel off-diagonals), and report how well that
    single scale generalises.

    `r2_loco` is the number that matters.  For each channel it refits c on the
    OTHER channels' entries and predicts the held-out channel's.  A c that only
    fits the marks it was trained on is not evidence that it transfers to
    cross-mark entries, which is what the completion actually needs.

    A negative r2_loco means the fitted scale predicts held-out entries worse
    than their own mean does -- proportionality is refuted.
    """
    ch = np.asarray(ch)
    p = comp.between.shape[0]

    iu = np.triu_indices(p, k=1 if exclude_diagonal else 0)
    same = ch[iu[0]] == ch[iu[1]]
    obs = comp.observed[iu] & same & np.isfinite(comp.within[iu])

    b = comp.between[iu][obs]
    w = comp.within[iu][obs]
    chan = ch[iu[0]][obs]

    if b.size < 3 or not np.any(np.abs(b) > 0):
        return ProportionalityFit(np.nan, np.nan, np.nan, {}, int(b.size))

    fit = lambda bb, ww: float(bb @ ww / (bb @ bb)) if bb @ bb > 0 else np.nan
    r2 = lambda y, yhat: (1.0 - np.sum((y - yhat) ** 2) /
                          np.sum((y - y.mean()) ** 2)
                          if np.sum((y - y.mean()) ** 2) > 0 else np.nan)

    c_all = fit(b, w)
    r2_in = r2(w, c_all * b)

    per_channel, preds, truths = {}, [], []
    for a in np.unique(chan):
        tr, te = chan != a, chan == a
        if te.sum() < 2 or tr.sum() < 2:
            continue
        per_channel[int(a)] = fit(b[te], w[te])
        c_tr = fit(b[tr], w[tr])
        if np.isfinite(c_tr):
            preds.append(c_tr * b[te])
            truths.append(w[te])

    r2_loco = (r2(np.concatenate(truths), np.concatenate(preds))
               if preds else np.nan)

    return ProportionalityFit(scale=c_all, r2_in_sample=float(r2_in),
                              r2_loco=float(r2_loco),
                              per_channel_scale=per_channel, n_pairs=int(b.size))


def complete_covariance(comp: Components, ch, mode="proportional",
                        fit: ProportionalityFit | None = None, ridge=0.0):
    """
    Assemble an estimate of the cell-level covariance Sigma_total.

    mode:
      'proportional'  fill the missing cross-channel within-entries with
                      c * Sigma_between, c from `fit_proportionality`.  The
                      completion the identifiability claim rests on.
      'zero'          fill them with 0: assume no within-state cross-mark
                      covariation.  The conservative null -- whatever
                      cross-mark structure survives here needs no assumption
                      beyond the decomposition itself.
      'between-only'  return Sigma_between.  Included to demonstrate the
                      rank collapse rather than to be used.
      'raw'           the covariance the current pipeline actually uses, with
                      the same-channel scale asymmetry left in place.

    Always PSD-projected; the projection is reported so a completion that had
    to be dragged a long way back can be distrusted.
    """
    p = comp.between.shape[0]
    ch = np.asarray(ch)

    if mode == "raw":
        Sig = comp.raw_aggregate.copy()
    elif mode == "between-only":
        Sig = comp.between.copy()
    else:
        W = np.where(comp.observed & np.isfinite(comp.within), comp.within, 0.0)
        if mode == "proportional":
            if fit is None or not np.isfinite(fit.scale):
                raise ValueError("proportional completion needs a finite fit")
            cross = (ch[:, None] != ch[None, :])
            W = np.where(cross, fit.scale * comp.between, W)
        elif mode != "zero":
            raise ValueError(f"unknown mode {mode!r}")
        Sig = comp.between + W

    Sig = 0.5 * (Sig + Sig.T)
    if ridge > 0:
        Sig = Sig + ridge * np.eye(p) * np.trace(Sig) / p
    Sig_psd, shift = project_psd(Sig)
    return Sig_psd, dict(mode=mode, psd_shift=shift,
                         eff_rank=effective_rank(Sig_psd),
                         cond=condition_number(Sig_psd))


# --------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------

def project_psd(Sig, floor_ratio=1e-6):
    """
    Nearest PSD matrix in the eigenvalue sense, with eigenvalues floored at a
    small fraction of the largest.  Returns the relative Frobenius distance
    moved, which is the honest measure of how much the completion had to be
    repaired to be a covariance at all.
    """
    Sig = 0.5 * (Sig + Sig.T)
    w, Q = np.linalg.eigh(Sig)
    if w.max() <= 0:
        return np.eye(Sig.shape[0]), float("inf")
    floor = floor_ratio * w.max()
    if w.min() >= floor:
        return Sig, 0.0
    w_fix = np.clip(w, floor, None)
    Sig_fix = (Q * w_fix) @ Q.T
    denom = np.linalg.norm(Sig, "fro")
    shift = float(np.linalg.norm(Sig_fix - Sig, "fro") / denom) if denom > 0 else float("inf")
    return 0.5 * (Sig_fix + Sig_fix.T), shift


def effective_rank(Sig):
    """Participation ratio of the eigenvalues: how many directions carry signal.

    This is the diagnostic that decides whether cross-mark DAG structure is a
    meaningful target at all.  If the between-cluster covariance has effective
    rank ~ d, cross-mark relations are factor loadings on a d-dimensional state
    manifold and no p-variable DAG over them is identified.
    """
    w = np.linalg.eigvalsh(0.5 * (Sig + Sig.T))
    w = np.clip(w, 1e-12, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def condition_number(Sig):
    w = np.linalg.eigvalsh(0.5 * (Sig + Sig.T))
    w = np.clip(w, 1e-12, None)
    return float(w.max() / w.min())


def to_correlation(Sig):
    d = np.sqrt(np.clip(np.diag(Sig), 1e-12, None))
    return Sig / np.outer(d, d)


# --------------------------------------------------------------------------
# Tetrad bridge
# --------------------------------------------------------------------------

def to_tetrad_covariance(Sig, names, sample_size):
    """
    Wrap a numpy covariance as a Tetrad CovarianceMatrix.

    `IndTestFisherZ(ICovarianceMatrix, alpha)` and `SemBicScore(ICovarianceMatrix)`
    both exist, so every FCI-family search and every BOSS-family score in the
    jar accepts this in place of a pipelines/empirical/dataSet.  The modification therefore lives
    entirely in the input, and the search is stock Tetrad.

    sample_size governs every p-value and every BIC penalty.  Pass the number
    of metacells, not the number of cells, unless you intend to claim the
    cross-channel entries were estimated at cell resolution -- they were not.
    """
    import jpype
    import edu.cmu.tetrad.data as td
    from java.util import ArrayList

    nodes = ArrayList()
    for n in names:
        nodes.add(td.ContinuousVariable(str(n)))
    arr = jpype.JArray(jpype.JDouble, 2)(np.asarray(Sig, dtype=float).tolist())
    mat = jpype.JClass("edu.cmu.tetrad.util.Matrix")(arr)
    return td.CovarianceMatrix(nodes, mat, int(sample_size))
