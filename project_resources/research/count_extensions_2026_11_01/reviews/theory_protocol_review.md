# Independent review of the count-model extension sweep

Review started 2026-11-01 before scientific fitting. This is a targeted mathematical and experimental-design review of the extension families identified for the existing project panels. It is not an exhaustive review of every possible integration algorithm. No new datasets, worlds, labels, paired training observations or core implementation edits were made by this reviewer.

The bounded sweep should separate six questions: conditional count overdispersion, detection loss, depth response, higher-moment stabilization, generalized-covariance orientation, and panel size. Each changes a specific modeling or estimation assumption. All outcomes and failures must be retained, rather than selecting a preferred model on the repeatedly inspected test data.

## Observation kernels and their valid corrections

Use the same target intensity $\Lambda_j$ in every kernel: the conditional observed mean is $t_j\Lambda_j$. For fixed sensitivity parameters $\theta>0$ and $q\in(0,1]$, a precise mean-preserving kernel is

$$
D_j\sim\operatorname{Bernoulli}(q),\qquad
Y_j=D_j N_j,\qquad
N_j\mid\Lambda,t\sim\operatorname{NB}\left(\text{mean}=t_j\Lambda_j/q,\ \text{shape}=\theta\right),
$$

with conditionally independent feature kernels. The limits $\theta=\infty$ and $q=1$ give Poisson measurement. The target is the mean rate after averaging over the stated detection process, not an independently recovered pre-dropout intensity.

Let $(Y)_r=Y(Y-1)\cdots(Y-r+1)$ and

$$
c_r(\theta)=\prod_{\ell=0}^{r-1}(1+\ell/\theta).
$$

Then for $r\geq1$,

$$
\mathbb E[(Y_j)_r\mid\Lambda,t]
=(t_j\Lambda_j)^r c_r(\theta)q^{1-r},
\qquad
f_{rj}=\frac{(Y_j)_r q^{r-1}}{t_j^r c_r(\theta)}
$$

has conditional expectation $\Lambda_j^r$. Repeated indices need their own factorial order and correction; one scalar applied to an entire tensor would be wrong. Distinct feature indices factor conditionally. The formulas extend to prespecified feature-specific shapes or retentions, although the initial sweep need not fit that larger model.

For a compact implementation define per-cell polynomials

$$
\begin{aligned}
h_1&=f_1,\quad h_2=f_2-f_1^2,\\
h_3&=f_3-3f_2f_1+2f_1^3,\\
h_4&=f_4-4f_3f_1-3f_2^2+12f_2f_1^2-6f_1^4.
\end{aligned}
$$

For fixed direction $w$ and center $\mu$, let $z=\sum_jw_j f_{1j}-w^\top\mu$ and $H_r=\sum_jw_j^r h_{rj}$. Corrected projected raw moments about that center are

$$
z,\quad z^2+H_2,\quad z^3+3zH_2+H_3,
\quad z^4+6z^2H_2+3H_2^2+4zH_3+H_4.
$$

These are Bell-polynomial identities, not an assumption that noisy source projections are observed disturbances. Centered skewness and kurtosis remain finite-sample plug-in ratios. The feature covariance correction is $\operatorname{Cov}(f_1)+\operatorname{diag}\mathbb E(h_2)$.

Neither $\theta$ nor $q$ is identified as technical noise merely because a marginal count distribution is overdispersed or sparse. Biological rate heterogeneity and biological zeros can generate those same observations. Fixed sensitivity values are defensible; selecting a technical dispersion or dropout rate from the same marginal counts requires additional assumptions. [Sarkar and Stephens, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8370014/) distinguish measurement and expression models and explain why an NB observation distribution need not imply NB measurement noise. Their UMI-RNA arguments are not independent validation of a histone fragment measurement kernel.

## Depth elasticity and fair panel comparisons

Featurewise fitted elasticity changes exposure to an array $t_{ij}$, for example $t_{ij}=d_i^{\beta_j}$. Every factorial denominator and PGF calculation must then use that feature's exposure. Learning a slope against depth does not establish a technical effect when depth and biological state are associated. Fit slopes on the source-fit rows only, freeze them for calibration/test, and label the resulting biological interpretation as a sensitivity analysis.

[Hafemeister and Satija, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6927181/) provide the relevant regularized count-regression inspiration. That normalization paper does not prove cross-assay causal identification or guarantee that regressing out total count preserves this pilot's source model. An estimated slope is part of the fitted procedure; its uncertainty must be propagated by refitting inside the bootstrap or explicitly declared conditional on the fixed slope.

The 300-feature branch should retain the exact old 30 and append training-only selected features. Report both full-panel metrics and the original 30-feature block, because changing the number and prevalence of evaluated pairs changes aggregate error. The data agent confirmed nested panels, explicit original-30 indices and the same leave-all-300-out depth proxy at both widths. That proxy differs from the older study's leave-30-out quantity, so the new study's baseline—not the older score—is the controlled exposure comparison. More features can supply signal or simply more sampling noise; numerical rank alone is not evidence of more biological causes.

## Higher-moment shrinkage that changes the estimator

Multiplying the whole third tensor by a positive scalar does not change its contraction eigenvectors. That would not test a useful source-orientation extension. Entrywise uncertainty shrinkage in the small whitened tensor can change orientation while avoiding a full $300^3$ tensor.

For fixed fitted whitening $W$ and center, each cell contributes its corrected cubic projected polynomial. Split these contributions into $G=5$ fixed groups, retaining the full-fit center and whitening in every group. If group $g$ has size $n_g$ and mean tensor $T_g$, and $T=\sum_g n_gT_g/n$, use

$$
\widehat V=\frac{\sum_g n_g(T_g-T)^2}{(G-1)n},\qquad
\widetilde T=\frac{T^3}{T^2+\widehat V},
$$

entrywise, with zero at $0/0$ and symmetric tensor entries restored. The unequal-group-size formula is appropriate for the variance of a mean when the transform is fixed. Here the transform is estimated, so it is only an internal variance proxy. The outer bootstrap must recompute the transform, group variance and shrinkage.

This is a local biased finite-sample regularizer, not a published causal-identification result. It depends on the chosen whitened coordinate basis and is not invariant to arbitrary rotations of that basis. Its source gate should use the original unshrunk tensor residual; a separate shrunk-target fit residual can be reported. Otherwise the regularizer could certify the structure it imposed. Under suitable finite moments and consistent estimation, the variance proxy decreases with sample size, but this does not guarantee finite-sample improvement.

## Bounded-PGF orientation

For the Poisson kernel only, take processing vector $s\leq0$ with $1+s_j/t_{ij}>0$ and define

$$
g_i=\prod_j(1+s_j/t_{ij})^{Y_{ij}},\qquad
v_{ij}=Y_{ij}/(t_{ij}+s_j).
$$

The Hessian of the log intensity transform is estimated by

$$
\widehat H(s)=\widehat{\mathbb E}_g
\left[vv^\top-\operatorname{diag}\{Y/(t+s)^2\}\right]
-\widehat{\mathbb E}_g(v)\widehat{\mathbb E}_g(v)^\top.
$$

Under a lawful affine independent-source intensity model,

$$
H(s)=B\operatorname{diag}\{\kappa''_{\varepsilon_k}(b_k^\top s)\}B^\top.
$$

Several processing points can therefore supply an orientation criterion that does not require nonzero third cumulants. [Podosinnikova, Bach and Lacoste-Julien, 2016](https://proceedings.mlr.press/v48/podosinnikova16.html) motivate generalized covariances and joint diagonalization. Their multiview result is not automatically an unpaired causal theorem for this local estimator.

Subtracting each whitened matrix's scalar identity component before computing its off-diagonal fit residual avoids a large isotropic component making a poor orientation look good. Estimated matrices can still be noisy or indefinite. The proposed observed base bound of at least 0.5 controls observed weights; it does not prove a lower bound on unknown population exposures. Unit exposure has the clean population support condition. Low effective sample size or a nearly zero empirical transform can still be unstable.

The Poisson PGF identity is not an NB or ZI deconvolution identity, so those combinations are excluded. Also, different assays apply different source tilts $B^\top s$. Tilted diagonal variances are not directly comparable source-law fingerprints across assays. Use the fitted source coordinates with a separately justified calibration statistic.

## Matching, optional characteristic-function confirmation, and errors

Use the same two-moment practical-equivalence screen across all extension families, with the fixed skewness/kurtosis margins, bootstrap precision, loading stability, unique sign and unique source rules. Call outputs **two-moment source candidates**. Two distinct distributions can share these moments; a match is not proof of an equal source law or a shared biological cause. Returning ambiguity sets is appropriate when candidates are not unique.

A common strict secondary characteristic-function stage is mathematically available for the stated NB/ZI kernels. For a fixed coordinate frequency $v$ and integer count $y$,

$$
g_y(v)=1+\frac{{}_1F_1(-y;\theta;-iv\theta q/t)-1}{q}
$$

when $\theta<\infty$, and

$$
g_y(v)=1+\frac{(1+ivq/t)^y-1}{q}
$$

in the Poisson limit. The confluent hypergeometric series terminates because $-y$ is a nonpositive integer. Its order-$r$ coefficient is exactly $(y)_r(iv/t)^r q^{r-1}/\{r!c_r(\theta)\}$. Taking conditional expectations gives $\exp(iv\Lambda)$. Conditional absolute term summation is bounded by an exponential in the fixed finite rate; unconditional integrability over the latent rate distribution remains a separate requirement. Products over conditionally independent features and the fixed centering factor yield a projected intensity characteristic function.

Installed SciPy's complex `hyp1f1` calculation was checked against deterministic conditional count-probability sums for nine finite-$\theta$/retention fixtures; mean errors were below $6\times10^{-16}$. The [official SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html) describes its confluent-hypergeometric function. Finite test fixtures do not certify accuracy for every large count or weight. Cancellation, overflow, heavy-tail variance and an insufficient bootstrap sample remain material risks. Do not clip weights and silently retain the same deconvolution claim.

The protocol may compute this stricter stage only for prespecified first-stage candidates to save time. Then mark other characteristic-function comparisons as not assessed; do not convert selection into a nominal simultaneous confidence or false-discovery claim. Report first-stage candidates and strict confirmations separately. A finite CF grid still does not identify a complete source law.

Independent-column controls test spurious coherent structure in the within-assay estimator. When these controls select rank zero, they do not exercise or calibrate false source sharing. Shuffling independent assay rows preserves the marginal laws intentionally used by source matching and is not a valid false-sharing null. No existing control supplies biological FDR for the two-moment or finite-CF screen.

The previously inspected paired and state references remain evaluation-only and must be read after the complete fit block is sealed. Compare against the zero cross-covariance and diagonal within-assay baselines; report all seeds, settings, failures and masks. Exact feature edges and directions remain unsupported unless a separate justified target and model assumption establishes them. A broad sweep can diagnose which assumptions matter on these reused panels; it cannot become independent validation by having many arms.

## Literature and review provenance

This is a targeted source review, not a systematic search. OpenAlex discovery metadata and exact queries are saved in [literature_search_log.json](literature_search_log.json), with two 100-record response files. An initial browser API request was inaccessible, after which direct public API retrieval succeeded; no project counts or credentials were placed in query URLs. Primary proceedings, publisher/PMC papers and official mathematical/software documentation were used to verify the relevant claims. The earlier [DICA review](../../count_mdcrl_2026_11_01/reviews/theory_implementation_review.md) contains the Poisson cumulant derivations and the restricted population-identification proposition.

The module `extension_estimators.py` was read before scientific fitting. Its NB/ZI Bell polynomials, corrected projected cubic tensor, group-size-weighted shrinkage, PGF Hessian, observation-scale mixing/unmixing units, and use of the unshrunk C3 eligibility residual agree with the formulas above. Final protocol, runner and sealed-result approval remain separate audit steps.

**Final prefit review:** the 25-variant, 225-arm protocol and actual runner/evaluator were independently reviewed and approved for this bounded exploratory sweep. The global depth elasticity is frozen across source bootstraps, so its uncertainty remains explicitly conditional. The exact CF conditioning heuristic, minimum-over-processing-points PGF effective-sample gate, raw-moment candidate versus unique-match distinction, previous imported protocol provenance, and failed partial-artifact preservation are declared. See the hash-bound [prefit receipt](prefit_theory_review.json), [independent count-kernel checks](extension_algebra_audit.json), and [literal-resample Gram check](gram_algebra_audit.json). This approval does not imply successful recovery; scientific outcome interpretation awaits the sealed fit and evaluation blocks.

Before any scientific fit, a constant-column completion correction was reviewed and added uniformly. An exactly constant rate column retains its observed standard deviation of zero, is centered exactly to zero, and receives an internal working scale of one for numerical whitening and alignment. Its corrected covariance block has zero off-diagonals and a nonpositive diagonal, so it contributes no positive covariance mode. Independent checks across all 12 count kernels and the ordinary ablation verify the remaining positive subspace, zero loadings on appended constant columns, and unit demixing. This avoids aborting an otherwise estimable sparse bootstrap sample. It is a declared completion-rule change relative to the preceding pilot; no feature is removed and no rank or source-eligibility threshold is relaxed. Constant masks/counts and the working scales are saved. Finite-sample PGF processing scales can depend on all feature means, so this argument does not assert that PGF orientations equal a calculation that deleted the constant columns. Rejected CF bootstrap values and their diagnostic arrays were also added for auditing, without changing selection. The earlier prefit receipt is preserved beside the final receipt.
