# Independent count-source theory and implementation review

Reviewed on 2026-11-01. The source and protocol review preceded scientific fits; the independent algebra receipt was completed while the frozen fits were running. No project counts, paired references, state labels, or fitted scientific outcomes were used for this review. No learner or protocol files were edited by this reviewer.

**Decision:** the frozen pilot is a defensible test of a limited algorithmic change: replacing ordinary count moments with estimates of latent-intensity moments under a conditional Poisson measurement model. The reviewed code implements that change correctly. It is not a theorem that these panels identify biological sources, shared source identities, or a causal graph. A failed gate, zero source count, or zero matches must remain a substantive outcome.

Reviewed versions are sealed in [count_algebra_audit.json](count_algebra_audit.json). They are protocol `fb50a65d1dc8bfccd43889494686e50ad98407f2a898b1df906452c58c6fc31c`, learner `43d341a81b4160edf95af33124c51c71afb6b0f800f14086dba0c2d88d8d3fff`, and runner `bc6a411efb67c3e3667a6be257e9d116c164c7695eac010a20722edf11f37504`.

## Published inspiration and the actual extension

[Podosinnikova, Bach and Lacoste-Julien, NeurIPS 2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/9be40cee5b0eee1462c82c6964087ff9-Paper.pdf), model (4) and equations (10)–(13), derive discrete ICA using conditionally independent Poisson observations of linear combinations of independent source intensities. Their corrected second and third cumulants retain the source mixing structure. The paper supplies the within-assay statistical starting point. The current exposure adjustment, internal calibration split, count-deconvolved fingerprints, permutation dimension screen, and uncertainty-based matching gates are local extensions, not results established by that paper.

This is a clearer first ablation than a flexible count VAE because the changed estimand and removed noise term are explicit. A VAE with flexible mixing, an imposed common source prior, and separate unpaired assays would need its own identification argument: good likelihood alone does not establish shared source identity. This comparison does not rule out identifiable count generative models under suitable restrictions.

[Beyond CCA, ICML 2016](https://proceedings.mlr.press/v48/podosinnikova16.html) extends moment methods to multiview models. It is relevant mathematical context, but this pilot does not claim a paired multiview moment theorem for unpaired observations. The cross-assay step instead attempts distribution-based source matching. The earlier Sturma restrictions are still material, especially distinct source distributions, a suitable shared/private structure, and additional assumptions for the latent graph.

## Measurement identities checked independently

The working model is

$$
Y_j\mid\lambda,t\ \text{independently}\sim\operatorname{Poisson}(t\lambda_j),
\qquad \lambda=\mu+B\varepsilon,
$$

where the coordinates of $\varepsilon$ are independent and have unit variance. Let $R_j=Y_j/t$ and $Q_j=Y_j/t^2$. Then

$$
C_2=\operatorname{Cov}(R)-\operatorname{diag}\{\mathbb E(Y/t^2)\}
=\operatorname{Cov}(\lambda).
$$

The corrected third cumulant is

$$
\begin{aligned}
C_{3,ijk}={}&\kappa(R_i,R_j,R_k)
-\delta_{jk}\operatorname{Cov}(R_i,Q_j)
-\delta_{ik}\operatorname{Cov}(R_j,Q_i)\\
&-\delta_{ij}\operatorname{Cov}(R_k,Q_i)
+2\delta_{ij}\delta_{jk}\mathbb E(Y_i/t^3).
\end{aligned}
$$

This equals the third cumulant of the latent intensities. Conditional independence and the exposure-scaled Poisson kernel are the requirements for these identities; independence of $t$ from $\lambda$ is not additionally necessary. The code constructs equivalent raw factorial moments and then centers them. The sample estimates use plug-in means with `ddof=0`, so finite-sample unbiasedness is not claimed.

For a fixed projection $w$, define $u_r=\sum_jY_j(w_j/t)^r$. The first four projected factorial polynomials are

$$
u_1,\quad u_1^2-u_2,\quad u_1^3-3u_1u_2+2u_3,
\quad u_1^4-6u_1^2u_2+3u_2^2+8u_1u_3-6u_4.
$$

Their conditional expectations are $(w^\top\lambda)^r$. The corresponding centered identities used in the implementation also hold for signed $w$. Independent symbolic checks covered all repeated-index cases in $C_2$ and $C_3$, all four projected orders, and the alternative central-cumulant formula. An exact bounded-source tensor fixture verified unit modeled variance, mixing/unmixing units, and recovery up to sign/permutation. These are algebra fixtures, not new benchmark worlds or evidence of finite-sample recovery.

## What the characteristic function adds and what it cannot certify

For a direction and center fixed using the source-fit split,

$$
Z_\omega=\exp(-i\omega w^\top\mu)
\prod_j(1+i\omega w_j/t)^{Y_j}
$$

has conditional expectation $\exp\{i\omega w^\top(\lambda-\mu)\}$. Its conditional absolute second moment is

$$
\mathbb E(|Z_\omega|^2\mid\lambda,t)
=\exp\left\{\frac{\omega^2}{t}\sum_j\lambda_jw_j^2\right\}.
$$

Thus this probe is not merely a KS comparison of noisy count projections: it targets the latent-intensity law under the measurement model. It can also have extremely high variance. Unconditional validity requires integrability; an empirical standard-error gate and bootstrap cannot prove integrability or detect every rare tail. The code abstains for extreme weights instead of clipping them. Clipping would change the target.

The implementation uses the declared source-fit center and variance scale for the characteristic function; calibration skewness and kurtosis are recentered and rescaled on calibration moments. These are explicitly different finite-sample summaries of the fitted source coordinate. Sign reversal correctly conjugates the characteristic function and reverses skewness while preserving kurtosis.

Three characteristic-function frequencies and two moments do not identify a full probability law. Practical equivalence of this finite vector, even with a narrow bootstrap band, establishes only compatibility with these summaries. Distinct distributions can agree on them, and equal marginal source distributions alone do not prove that two independent biological mechanisms are one shared source. The unique-sign/unique-source gates reduce ambiguity without resolving that logical limitation.

## Count noise does not imply universal nonidentifiability

**Proposition (known fixed exposure).** Let $t_0>0$ be known and suppose the coordinates of $Y\mid\lambda$ are independent $\operatorname{Poisson}(t_0\lambda_j)$ variables for a random vector $\lambda\in[0,\infty)^p$. The full population law of $Y$ uniquely determines the law of $\lambda$. No finite-support or finite-moment assumption on that intensity law is needed for this proposition.

To see this, its multivariate probability generating function satisfies

$$
G_Y(z)=\mathbb E\prod_jz_j^{Y_j}
=\mathbb E\exp\{-t_0\lambda^\top(1-z)\}
=\mathcal L_\lambda\{t_0(1-z)\}.
$$

For $z\in(0,1)^p$, the count law therefore gives the intensity law's Laplace transform on the open box $(0,t_0)^p$. The transform is finite and analytic throughout the positive orthant. Two candidate intensity laws inducing the same count law have transforms equal on this box; analytic continuation followed by uniqueness of the multivariate Laplace transform makes their laws equal. This is a population statement about a known observation kernel, not a stable finite-sample inversion algorithm. The standard scalar generating-function relation is also the basis of [Meintanis's mixed-Poisson specification test](https://benthamopenarchives.com/abstract.php?ArticleCode=TOSPJ-3-27); the displayed multivariate argument is the reviewer's direct derivation of this standard principle, not a claimed new published theorem. Unit exposure is the special case $t_0=1$. No assertion about arbitrary unknown or count-derived exposure follows.

Each within-assay intensity law still leaves cross-assay coupling unspecified. If the centered intensity blocks themselves satisfy the exact Sturma linear full-rank observation and distinguishable independent-source assumptions, one may first identify those block laws by the proposition and then apply the population source-matching argument. The additional shared/private separation, partial-pure-child and rank-faithfulness conditions remain required for its shared latent graph theorem. See [the independently reviewed Sturma conditions](../../sturma_pilot_2026_11_01/reviews/assumption_evaluation_review.md). This is a conditional composition of identification arguments; the current estimator neither reconstructs complete intensity laws nor verifies those conditions. If a joint count law is also the target, its cross-assay measurement kernel must be specified too.

Consequently, a negative result from this finite-moment estimator cannot prove that all count-aware unpaired models fail. Conversely, population identification under those restrictions does not certify that these sparse, noisy panels support the inversion or satisfy the restrictions.

## Preconditions not established by this pilot

- **Valid rates and affine structure.** Signed loadings require a support restriction that keeps every component of $\mu+B\varepsilon$ nonnegative. A finite intercept cannot make an arbitrary unbounded two-sided disturbance lawful. Nonlinear rates and feature-specific measurement overdispersion are outside the fitted model.
- **Meaning of noise.** A gamma-Poisson mixture can express biological heterogeneity or technical overdispersion. Subtracting the Poisson diagonal does not decide which remaining variation is biological. Independent NB measurement noise leaves further factorial variance.
- **Exposure.** Unit exposure removes only Poisson shot noise and leaves depth in the signal. A count-derived leave-panel-out total is a proxy, not automatically a valid exposure in the required conditional kernel. Promoter overlap can retain technical dependence after subtracting the selected panel. The sensitivity analysis cannot establish the exposure assumption.
- **Third-order orientation.** Several sources with zero third cumulants leave an unoriented subspace. Nonsymmetry alone does not guarantee nonzero skewness. A diagonal third tensor and stable loadings also do not establish full statistical independence.
- **Dimension.** Column permutations preserve marginal overdispersion. The selected rank is a screen for coherent cross-feature signal, not an estimate of every private or single-feature source. A cap is explicitly reported. Bootstrap uncertainty is conditional on this selected dimension and the exposure normalization.
- **Cross-assay coupling and graph.** Candidate-shared covariance assumes that accepted source matches correspond to shared random sources. The native graph step additionally needs latent structural and measurement assumptions, including appropriate pure children and rank conditions. The exploratory coefficients are anonymous; they are not observed peak–gene causal effects. The native numerical threshold remains scale dependent.

## Implementation and comparison audit

The runner reads only frozen training files, hashes them, defines source-fit/calibration rows from opaque IDs, and computes exposures before any permutations. It uses the same selected rank, splits, seeds, and gates for corrected and uncorrected arms. Ordinary moments receive no second Poisson noise addition. The corrected rank screen now uses the explicit plus-one Monte Carlo rule with conservative ties; with 49 nulls, two null maxima at least as large as the statistic already give $p=3/50>0.05$. Depth-stratified permutations remain an approximate diagnostic, not a calibrated biological error guarantee.

Bootstrap columns are aligned in fixed source-fit standardized feature units. The sign is carried into the candidate unmixing matrix before calibration fingerprints are recomputed. Independent assay seeds make the paired bootstrap index a product-bootstrap replicate, not a cell match. Cross-assay sign bookkeeping and all-assay clique consistency are correct. Every source and candidate may abstain.

The principal controlled comparison is count correction versus the dimension-matched ordinary-moment estimator. The earlier 30-rank native Sturma fit is contextual because it used a different training split. Held-out paired moments can test a predicted covariance on the real panel; they are not causal truth. Simulated noisy-score/state association can test state recovery; the focal DAG is not a valid target for the anonymous latent coefficients.

The executable review is [verify_count_algebra.py](verify_count_algebra.py), with nine passing check groups in [count_algebra_audit.json](count_algebra_audit.json). It reads no project data and performs no model-data fits. Source hashes identify exactly the reviewed version. This approval covers the bounded implementation and comparison, not biological model validity or a reliable recovery claim.

## Untested follow-up: bounded generating-function derivatives

This is an algorithmic proposal only. It was not fitted, evaluated, or added to the frozen protocol. A possible follow-up replaces unweighted high-order moments and potentially explosive complex weights with generalized covariances from bounded count generating functions. [Podosinnikova, Bach and Lacoste-Julien, ICML 2016](https://proceedings.mlr.press/v48/podosinnikova16.html), provide the published generalized-covariance/joint-diagonalization inspiration; the use below would need its own detailed derivation and evaluation.

Restrict first to known constant exposure $t_0$, including the unit-exposure primary arm. For $s_j\in(-t_0,0)$, put $z_j=1+s_j/t_0\in(0,1)$. Then

$$
L(s)=\mathbb E e^{s^\top\lambda}
=\mathbb E\prod_jz_j^{Y_j}.
$$

The empirical product weight is at most one. Its first and second derivatives use ordinary falling-factorial terms, such as $Y_jz_j^{Y_j-1}/t_0$ and $Y_j(Y_j-1)z_j^{Y_j-2}/t_0^2$. For fixed interior $z$, these polynomial-times-geometric terms are bounded on integer counts. This avoids the exponential absolute-weight growth of the deconvolved characteristic-function probe. It does not guarantee a well-conditioned estimate: very small $L(s)$ can still make ratios and log derivatives unstable.

Under the lawful affine independent-source rate model, the population Hessian satisfies

$$
\nabla_s^2\log L(s)
=B\operatorname{diag}\{\kappa''_{\varepsilon_k}(b_k^\top s)\}B^\top,
$$

where $b_k$ is column $k$ of $B$ and $\kappa_{\varepsilon_k}$ is that source's log moment-generating function wherever it exists. Several prespecified tilts could therefore replace a single third-order tensor in a simultaneous diagonalization estimator. The source-rank and distinct tilted-variance conditions would require checking; repeated or constant variance profiles can remain ambiguous. Plug-in log-Hessians are not finite-sample unbiased merely because the component derivatives are unbiased.

This does **not** directly solve unpaired source matching. Different assay loadings turn the same observed tilt into different source tilts $b_k^\top s$. Comparing their diagonal variance values or tilted laws across assays would consequently compare different distributions. A valid source-law alignment or transformation argument is still needed before treating such agreement as sharing. Unknown or count-derived exposures also need a separate extension. Any follow-up should remain on the existing fixed panels, preserve the present results, and predeclare its new estimand, stability gates and controlled comparison before fitting.
