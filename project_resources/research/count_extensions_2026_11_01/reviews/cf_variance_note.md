# Conditional variance of the generalized count CF probe

This is an independent mathematical check for the fixed NB/ZI sensitivity kernel, not a fitted result or an additional selection rule. The formulas below were checked by deterministic count-probability sums in [extension_algebra_audit.json](extension_algebra_audit.json). No unknown cell-level intensity is estimated or supplied to the scientific learner by this calculation.

Write $\Lambda\geq0$ for the target observed mean rate and $t>0$ for a fixed exposure. Conditional on retention, the active count has NB shape $\theta$ and mean $t\Lambda/q$. An independent Bernoulli($q$) mask produces the observed $Y$. For a real coordinate frequency $v$, define

$$
g_Y(v)=1+\frac{{}_1F_1(-Y;\theta;-iv\theta q/t)-1}{q}.
$$

Because the series terminates at integer $Y$ and the corrected factorial moments equal $\Lambda^r$, its conditional mean is $e^{iv\Lambda}$. For each fixed finite $\Lambda$, absolute summation of the factorial-series terms has finite expectation bounded by $e^{|v|\Lambda}$. Integrating over an arbitrary intensity distribution is a different question.

An exact conditional second moment is

$$
\mathbb E\{|g_Y(v)|^2\mid\Lambda,t\}
=\frac{F_\theta(v,\Lambda,t,q)+(1-q)\{1-2\cos(v\Lambda)\}}{q},
$$

where

$$
F_\theta(v,\Lambda,t,q)
={}_0F_1\left(;\theta;v^2\left\{q\theta\Lambda/t+\Lambda^2\right\}\right).
$$

One derivation writes the terminating hypergeometric function as a generalized Laguerre polynomial and sums its squared modulus under the active NB probability mass. The [Hille–Hardy identity, DLMF 18.18.27](https://dlmf.nist.gov/18.18.E27), followed by the relation between the modified Bessel function and ${}_0F_1$, gives $F_\theta$ for the active count. Expanding the squared modulus after the affine retention correction and adding the masked-zero mass gives the displayed second moment. The complex-conjugate arguments follow by analytic continuation of the polynomial identity.

In the Poisson limit,

$$
F_\infty(v,\Lambda,t,q)=\exp(v^2q\Lambda/t).
$$

At $q=1$ this recovers the earlier Poisson expression $\exp(v^2\Lambda/t)$. At $v=0$ both formulas give one. Their small-frequency expansion is

$$
1+v^2\left[\frac{\Lambda}{t}
+\Lambda^2\left\{\frac{1+1/\theta}{q}-1\right\}\right]+O(v^4),
$$

which agrees with the conditional variance of $Y/t$ under this mean-preserving kernel.

For conditionally independent features, the second moment of the product probe is the product of these scalar second moments at $v_j=u w_j$. The fixed centering phase has unit modulus and changes neither second moment nor conditional variance. The conditional complex variance is therefore that product minus one.

This explains why a valid inverse-kernel identity need not yield a useful estimator: weak latent variation can require large demixing weights, several feature contributions multiply, and small dispersion or retention can increase noise. An empirical finite-sample variance gate cannot establish integrability over an unobserved heavy-tailed rate law. The sweep uses the declared observed numerical and uncertainty checks; it does not substitute these intensity-dependent population formulas for observed evidence.
