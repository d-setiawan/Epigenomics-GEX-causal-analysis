# Count-aware source estimation: hypothesis and scope

This experiment asks whether explicitly accounting for count sampling improves the source-estimation stage of unpaired causal representation learning. It reuses the existing RNA, H3K27ac and H3K4me3 panels. It introduces no paired training observations, labels, new modalities or benchmark worlds. The [protocol](protocol.json) fixes the comparison before fitting.

## What changes in the model

The working model separates a biological rate from its measurement:

$$
Z=(I-A)^{-1}\varepsilon,
\qquad
\lambda^{(a)}=\mu^{(a)}+G^{(a)}Z
=\mu^{(a)}+B^{(a)}\varepsilon,
$$

$$
Y^{(a)}_{ij}\mid\lambda^{(a)}_i,t^{(a)}_i
\sim\operatorname{Poisson}\!\left(t^{(a)}_i\lambda^{(a)}_{ij}\right).
$$

Here, $A$ describes a possible latent causal system, $B^{(a)}$ mixes independent source disturbances into assay rates, and $t^{(a)}_i$ is the exposure used by the observation model. Counts within an assay are conditionally independent given their rates and exposure. Sources may be shared or assay-specific, but the fitted model does not impose a common source distribution across assays.

Every rate must be nonnegative on the entire source support. An arbitrary continuous linear model does not satisfy this condition. Bounded sources with a suitable intercept permit some signed loadings; nonnegative sources and loadings provide another restricted construction. A Gaussian linear predictor passed through an exponential link is a different model: its rates no longer obey this affine ICA factorization. This pilot does not claim that either construction is the correct histone biology.

[Podosinnikova, Bach and Lacoste-Julien (2015)](https://proceedings.neurips.cc/paper_files/paper/2015/hash/9be40cee5b0eee1462c82c6964087ff9-Abstract.html) derive corrected cumulants for discrete ICA and estimate mixing through tensor diagonalization. We implement that measurement-aware principle, with local exposure adjustment, uncertainty checks and an unpaired fingerprint screen. This is a change to the estimator and its observation model; no PCA coordinates or variance-ranked replacement panel are supplied as input.

## Removing the Poisson contribution

Write $R=Y/t$. Under the stated measurement model,

$$
\mathbb E(R\mid\lambda,t)=\lambda,
\qquad
\operatorname{Cov}(R)
=\operatorname{Cov}(\lambda)
+\operatorname{diag}\!\left[\mathbb E\!\left(\frac{\lambda}{t}\right)\right].
$$

Thus the empirical covariance can be corrected by subtracting

$$
\operatorname{diag}\!\left[\frac1n\sum_i\frac{Y_i}{t_i^2}\right].
$$

Exposure may correlate with the biological rate without invalidating this identity, **provided the conditional measurement model itself holds**. An observed library total need not satisfy that requirement. Conditioning on a molecule total can induce multinomial dependence, and overlapping counting windows can share fragments.

The primary arm sets $t_i=1$: it removes the modeled Poisson noise, while retaining depth-related variation in the rate. The sensitivity arm uses the allowed library total minus the selected panel total, normalized by its source-fit median. That quantity is a proxy; it is never labeled known technical depth. It is computed from original counts before any control permutation and then held fixed.

For unit exposure, the third corrected tensor has entries

$$
T_{ijk}=\operatorname{cum}(Y_i,Y_j,Y_k)
-\delta_{ij}\operatorname{Cov}(Y_i,Y_k)
-\delta_{ik}\operatorname{Cov}(Y_i,Y_j)
-\delta_{jk}\operatorname{Cov}(Y_i,Y_j)
+2\delta_{ij}\delta_{jk}\mathbb E(Y_i).
$$

The implementation handles variable exposure through factorial raw moments before centering. In the population affine source model, corrected covariance and third cumulants factor through the same mixing columns. Whitening and diagonalizing cubic contractions can recover those columns when the required rank and third-order information are present. Multiple zero-skew sources can leave an unoriented subspace; non-Gaussianity alone does not guarantee useful third cumulants.

All finite-sample estimates here are plug-in estimates. No negative covariance eigenvalues are silently ridged away; discarded modes and failures are reported. The dimension gate compares corrected eigenvalues with training feature-permutation nulls. It detects coherent cross-feature signal above preserved marginal variation. It is not a posterior on the total number of biological or private sources.

## Comparing inferred source laws without comparing noisy count scores

For a fixed projection $w$, let $p_r=\sum_j w_j^rY_j$. The first four factorial projections are

$$
F_1=\frac{p_1}{t},\qquad
F_2=\frac{p_1^2-p_2}{t^2},\qquad
F_3=\frac{p_1^3-3p_1p_2+2p_3}{t^3},
$$

$$
F_4=\frac{p_1^4-6p_1^2p_2+3p_2^2+8p_1p_3-6p_4}{t^4}.
$$

Under the Poisson kernel, their conditional expectations are powers of $w\lambda$. These allow rate-source skewness and kurtosis to be estimated without pretending that a projected count vector is a noiseless disturbance. The test suite checks the identities against exact discrete probability sums.

A second probe estimates a characteristic function at a fixed small frequency grid. For fixed $w$, center $\mu$ and frequency $u$,

$$
Q_u(Y,t)=e^{-iu w\mu}\prod_j
\left(1+\frac{iu w_j}{t}\right)^{Y_j},
\qquad
\mathbb E(Q_u\mid\lambda,t)=e^{iu w(\lambda-\mu)}.
$$

This identity also exposes the risk:

$$
\mathbb E\left(|Q_u|^2\mid\lambda,t\right)
=\exp\!\left(\frac{u^2}{t}\sum_j\lambda_jw_j^2\right).
$$

Sparse weak signals can require large demixing weights, producing enormous variance. Some rate distributions do not even provide the integrability needed for an unconditional deconvolution expectation. Numerical, empirical-variance and magnitude gates therefore reject unstable fingerprints; passing them is not a proof of tail integrability.

Directions are learned on 75% of each existing training assay. The remaining 25% estimates fingerprints. Bootstrap resampling repeats both stages independently within assays, aligns columns within the same assay, and carries their signs into the fingerprints. Rank remains fixed in these resamples, so the reported uncertainty is conditional on the selected dimension. The earlier feature panel was selected using the original full training pool; this split does not undo that selection.

The comparison requires precise agreement within prespecified tolerances, a unique sign and mutually unique candidate sources. Broad uncertainty is not counted as evidence of equality. Even a successful comparison identifies only agreement of the tested moments and finite characteristic-function coordinates. Different source laws can pass such a finite screen.

## Controlled comparison and scientific interpretation

The uncorrected ablation uses the same source-fitting cells, features, selected dimension, seeds, fingerprint calibration cells and decision rules. It uses ordinary moments and empirical characteristic functions. This distinguishes the effect of explicit count-noise correction from imposing a low dimension. The earlier native Sturma result is contextual, because it used the entire training pool and a different dimension rule.

After fitting is sealed, a separate evaluator checks heldout moments. Primary cross-assay errors use the same fixed training scale for the prediction and reference. The model's within-assay prediction adds the Poisson variance back only in the corrected arm. For the uncorrected arm, ordinary covariance already contains observation noise; adding it again would be an error.

[Sturma et al. (2023)](https://arxiv.org/html/2302.00993v2) establish positive identification results under restricted linear source and measurement assumptions, with additional conditions for the latent graph. Count correction and finite fingerprint agreement do not establish those conditions. Native graph extraction is therefore an exploratory downstream calculation only when enough all-assay candidates survive; its anonymous latent coefficients are not peak–gene edges.

Feature-specific negative-binomial variation is a material unresolved alternative. For conditional variance $t\lambda+(t\lambda)^2/\theta$, Poisson correction leaves an additional diagonal rate variance $\mathbb E(\lambda^2/\theta)$. Marginal overdispersion alone cannot identify that contribution as technical rather than biological. A failure of this pilot means that the specified Poisson correction and source model did not suffice on these panels; it does not prove that every possible count model fails.
