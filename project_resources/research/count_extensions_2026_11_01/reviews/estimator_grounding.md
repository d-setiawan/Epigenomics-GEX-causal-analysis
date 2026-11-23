# Grounding and scope of the count-estimator extensions

The computational foundation is [Podosinnikova, Bach and Lacoste-Julien, *Rethinking LDA: Moment Matching for Discrete ICA*, NeurIPS 2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/9be40cee5b0eee1462c82c6964087ff9-Paper.pdf), Sections 2–4. It constructs corrected cumulants for conditionally Poisson observations of independent latent intensities and estimates loading directions through diagonalization. Our affine rate model, exposure adjustment, negative-binomial and dropout sensitivities, projected implementation, and uncertainty shrinkage extend that construction locally. We do not claim that the paper proves this biological model or identifies its cross-assay graph.

[Podosinnikova, Bach and Lacoste-Julien, *Beyond CCA: Moment Matching for Multi-View Models*, ICML 2016](https://proceedings.mlr.press/v48/podosinnikova16.html) supplies the generalized-covariance motivation: exponential tilting gives a family of matrices that share independent-source loading directions. That paper's cross-view estimator uses joint observations. We use only a within-assay, conditional-Poisson generating-function identity; its multi-view identification theorem is not transferred to unpaired counts. Primary-paper captures and the earlier bounded search are preserved in `../../count_mdcrl_2026_11_01/reviews/count_options_*.json` and its search log.

## Observation kernels and common rate units

For feature $j$, the observed rate is $\Lambda_j$, not the unobserved rate before dropout. Given $\Lambda,t$, independently across features, let an active count have negative-binomial shape $\theta$ and mean $t_j\Lambda_j/q$, and retain it with probability $q$. Poisson is $\theta=\infty$; no dropout is $q=1$. Thus every arm has $\mathbb E[Y_j/t_j\mid\Lambda,t]=\Lambda_j$.

For $c_r(\theta)=\prod_{\ell=0}^{r-1}(1+\ell/\theta)$, corrected per-feature raw moments are

$$
f_{rj}=\frac{(Y_j)_r q^{r-1}}{t_j^r c_r(\theta)},
\qquad \mathbb E[f_{rj}\mid\Lambda,t]=\Lambda_j^r.
$$

The fixed $\theta,q$ settings are sensitivity assumptions, not inferred biological truths. Independent feature kernels are necessary: a shared dropout mask or correlated technical noise would change the cross-feature formulas. The mean and ordinary observed-rate standard deviation are unchanged across kernel sensitivities.

Define per-cell polynomials $h_2=f_2-f_1^2$, $h_3=f_3-3f_2f_1+2f_1^3$, and $h_4=f_4-4f_3f_1-3f_2^2+12f_2f_1^2-6f_1^4$. For a fixed projection $w$, $z=w^T(f_1-\mu)$ and $H_r=\sum_jw_j^rh_{rj}$ give the four corrected projected moments

$$
z,\quad z^2+H_2,\quad z^3+3zH_2+H_3,\quad
z^4+6z^2H_2+3H_2^2+4zH_3+H_4.
$$

These identities were checked by exact sums under independent Poisson/NB/dropout kernels, including signed projections. Ratios of estimated central moments are finite-sample plug-in estimates, not unbiased fingerprints.

## Source orientation extensions

1. **Projected C3:** covariance correction precedes whitening. Third-order corrections are contracted directly into the retained $k$-dimensional subspace. Only a $k^3$ tensor is formed, with $k$ supplied externally; there is no $p^3$ feature tensor. The Poisson limit reproduces the original full-tensor equations on fixed fixtures.
2. **Entrywise C3 shrinkage:** five deterministic row groups use the identical full-fit mean and whitening. For group tensors $T_g$ of size $n_g$, the variance proxy is $V=\sum_g n_g(T_g-T)^2/[(G-1)n]$. The orientation target is $T^3/(T^2+V)$ entrywise, with $0/0=0$. This proxy ignores the internally estimated mean/whitening; the outer bootstrap must refit both. Shrinkage depends on the whitening eigenbasis and is not generally rotation invariant. The eligibility residual is evaluated on the original unshrunk C3, so suppression cannot certify its own fit.
3. **Poisson PGF orientation:** for nonpositive processing vector $s$, use $g(Y)=\prod_j(1+s_j/t_j)^{Y_j}$. Under the Poisson kernel its conditional expectation is $\exp(s^T\Lambda)$. The Hessian of the log expectation equals the covariance of rates under exponential tilting. For independent affine sources it has the same loading directions with changed diagonal source variances. The implementation computes projected Hessians directly and removes scalar identity components when measuring joint-diagonalization residuals. It is restricted to $\theta=\infty,q=1$.

PGF processing is capped so every observed training base lies in $[0.5,1]$. A cap using the observed minimum exposure does not establish a lower bound on the full population exposure support. Unit exposure has the straightforward bounded-kernel guarantee. Different assays induce different source tilts; PGF diagonal values are therefore never used as cross-assay fingerprints.

All variants may abstain for invalid requested covariance modes, excessive condition, or insufficient orientation information. Negative modes are reported without a positive-definite ridge. Candidate sharing is decided separately from these within-assay estimators. Finite-moment agreement, tensor fit, or a stable covariance does not by itself identify shared causal variables. In particular, the fitted covariance $BB^T$ is invariant under orthogonal source rotations, so equal covariance scores across seeds do not demonstrate stable source recovery.

The preflight revision retains exact constant-rate columns instead of aborting a whole fit when a sparse bootstrap loses their detections. Their observed standard deviation remains zero; an internal working scale of one avoids division by zero. Exact centering makes their covariance cross-rows zero. Their corrected diagonal is nonpositive under every declared kernel, so they do not enter positive covariance eigenmodes. This is a coordinate preconditioner, not a positive-definite ridge or feature deletion. It changes completion behavior relative to the earlier blanket-abstention rule. Tests verify positive-mode, C3-source and unit-variance parity after appending zero and positive constant columns. Deleting a positive constant need not leave finite-sample PGF orientation unchanged, because all retained means determine processing-point strength.

## Secondary generalized characteristic-function check

For a fixed source direction, set $v_j=u w_j$. The local inverse-kernel identity is

$$
g_y(v)=1+\frac{{}_1F_1(-y;\theta;-iv\theta q/t)-1}{q},
$$

with Poisson limit $g_y(v)=1+[(1+ivq/t)^y-1]/q$. Under the declared conditional count kernel, the expectation of the feature product, with phase factor $\exp(-iu w^T\mu)$, gives the projected observed-rate characteristic function whenever the necessary expectations are integrable. These are local algebraic extensions, verified by exact conditional probability sums and a separate theory review; they are not a claim from the cited integration papers.

The helper reports the complex estimate, empirical standard error, maximum log-weight, and numerical diagnostics. It does not clip estimates to the unit disk. Finite-theta evaluation additionally checks a conservative product-conditioning proxy based on the absolute coefficients of the terminating hypergeometric polynomial. The mean absolute proxy must be at most $10^{-8}$. This is a heuristic diagnostic, not a proven error bound for SciPy. Exact zero cell products remain valid; overflow, nonfinite values, and excessive conditioning cause abstention. The Poisson branches use stable `log1p`/`expm1` formulas without the polynomial diagnostic.

A small empirical standard error does not prove population integrability or exclude rare high-weight cells. The protocol uses this finite-grid check only for a stronger secondary label among moment-screen candidates. It does not convert the primary finite-moment screen into a complete source-law test or supply nominal post-selection coverage.
