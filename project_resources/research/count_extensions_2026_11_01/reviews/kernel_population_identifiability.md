# What a known count kernel preserves at the population level

This is an independent mathematical derivation, not a new published theorem or a guarantee for the finite estimators in this sweep. It concerns **one assay at a time**, known constant positive exposures, and known kernel parameters. It does not identify unknown technical noise or the coupling between unpaired assays. No data, numerical experiment, or scientific setting was changed for this note.

Let the finite-dimensional intensity vector $\Lambda\in[0,\infty)^p$ be finite almost surely. Fix known constants $t_j>0$, $q_j\in(0,1]$ and $\theta_j\in(0,\infty]$. Conditional on $\Lambda$, use independent feature kernels

$$
N_j\mid\Lambda\sim\operatorname{NB}(t_j\Lambda_j/q_j,\theta_j),
\qquad D_j\sim\operatorname{Bernoulli}(q_j),
\qquad Y_j=D_jN_j,
$$

where the $D_j$ are mutually independent and independent of $(\Lambda,N)$. The NB arguments are mean and shape, with $\theta_j=\infty$ denoting Poisson. Under these restrictions, the complete population distribution of $Y$ uniquely determines the complete within-assay distribution of $\Lambda$. No finite-moment assumption is needed for the argument below.

## 1. Remove the known independent zero masking

Write $G_Y(z)=\mathbb E\prod_j z_j^{Y_j}$ and define the coordinate evaluation operator $E_jf(z)=f(z_1,\ldots,z_{j-1},1,z_{j+1},\ldots,z_p)$. Independent masking gives

$$
G_Y=\prod_{j=1}^p\{q_jI+(1-q_j)E_j\}G_N.
$$

Because $E_j^2=E_j$, each factor has inverse

$$
\{q_jI+(1-q_j)E_j\}^{-1}
=q_j^{-1}\{I-(1-q_j)E_j\}.
$$

The coordinate operators commute. Thus known $G_Y$ uniquely determines $G_N$ on the probability-generating-function domain. This signed operator inversion is a population statement; it is not proposed as a stable high-dimensional empirical estimator. Small retention and large dimension can make inversion badly conditioned.

## 2. Identify the gamma-multiplied intensity

The NB kernel has the Poisson mixture representation

$$
G_j\sim\operatorname{Gamma}(\theta_j,\text{rate}=\theta_j),
\qquad T_j=(t_j/q_j)\Lambda_jG_j,
\qquad N_j\mid T\sim\operatorname{Poisson}(T_j),
$$

with $G$ independent of $\Lambda$ and independent across coordinates. For the Poisson limit take $G_j=1$. Integrating the gamma density gives the specified NB conditional law.

Consequently

$$
G_N(z)=\mathbb E e^{-\sum_j(1-z_j)T_j}=\mathcal L_T(1-z).
$$

The count law determines the Laplace transform of $T$ on an open positive box. Analytic uniqueness, applied successively to each coordinate, extends this equality to the positive orthant; uniqueness of the multivariate Laplace transform determines the law of $T$. This is the same restricted Poisson-mixture step used in the [earlier population argument](../../count_mdcrl_2026_11_01/reviews/theory_implementation_review.md).

## 3. Undo known multiplicative gamma noise

First suppose all coordinates of $\Lambda$ are positive. Taking logarithms gives

$$
\log T_j=\log(t_j/q_j)+\log\Lambda_j+\log G_j.
$$

For finite $\theta_j$, direct integration of the gamma density yields

$$
\varphi_{\log G_j}(u)
=\mathbb E G_j^{iu}
=\theta_j^{-iu}\frac{\Gamma(\theta_j+iu)}{\Gamma(\theta_j)}.
$$

This quantity never vanishes for real $u$: the gamma function has no zeros and has no poles when its argument has positive real part. The integral and absence of zeros are documented in [NIST DLMF §5.2(i)](https://dlmf.nist.gov/5.2#i). Therefore the known characteristic function of $\log T$ can be divided by the known shift and product of these nonzero noise characteristic functions to determine that of $\log\Lambda$. Characteristic-function uniqueness identifies its law. For Poisson coordinates the noise factor equals one.

Zeros do not invalidate the result. Since $G_j>0$ almost surely, $T_j=0$ exactly when $\Lambda_j=0$. The identified law of $T$ therefore determines each zero-pattern probability. On every pattern of positive probability, condition on that pattern and perform the logarithmic deconvolution only on its positive coordinates. Conditioning on the intensity zero pattern leaves the independent gamma variables unchanged. Combining the identified conditional laws recovers the entire law of $\Lambda$, including its boundary mass.

## What this does and does not establish

Known Poisson, NB and independent excess-zero kernels need not erase the population information in a complete within-assay intensity law. This does not make recovery statistically easy: transform inversion can be severely ill-conditioned, and finite moments or three CF frequencies contain much less information than the entire count distribution. Empirical uncertainty gates do not prove population integrability of the inverse-kernel estimators.

The argument requires the true shape, retention, exposure and independence assumptions. The fixed sensitivity values in this study are not evidence that those parameters are known. Unknown dispersion or masking can be confounded with the latent intensity distribution. Random, endogenous library-depth proxies are outside this constant-exposure argument.

Even exact recovery of every assay's intensity law leaves its cross-assay coupling unspecified. Independent-source matching and causal graph recovery require additional substantive restrictions, such as the appropriate Sturma model conditions; the present finite fingerprints do not establish them. Thus failure of the finite count estimators cannot by itself prove a population count-law impossibility, and success at within-assay deconvolution would not by itself prove cross-assay causal recovery.
