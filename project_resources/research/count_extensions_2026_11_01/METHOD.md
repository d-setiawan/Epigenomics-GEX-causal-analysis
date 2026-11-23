# Count-source extensions on the existing observations

This study tests the six model or estimator changes identified after the
[Poisson pilot](../count_mdcrl_2026_11_01/README.md). The
[frozen protocol](protocol.json) specifies 25 variants, three existing datasets,
two source-estimation seeds and one independent-column control per variant and
dataset: 225 arms. This is a finite exploratory comparison, not an exhaustive
search or fresh locked validation.

## What is observed and what is being estimated

Training uses disjoint cells measured for RNA, H3K27ac or H3K4me3. Each assay is
fitted independently. No cell pairing, state labels, causal graph, experiment
day or evaluation outcome enters fitting, rank selection or source matching.
The two simulated worlds are the existing `linear_chain_0` and
`constrained_collider_0`; the real cohort is the existing scMTR-seq S1 split.
No new worlds or biological observations are generated.

The first target is a within-assay intensity representation

$$
\Lambda^{(a)}=\mu^{(a)}+B^{(a)}S^{(a)},
\qquad \operatorname{Cov}(S^{(a)})=I,
$$

with independent source coordinates and nonnegative intensities on their
support. The fitted columns of $B^{(a)}$ are anonymous source directions. They
are neither genomic feature edges nor a causal graph. Finite distribution
fingerprints then test whether selected source coordinates from different
assays are consistent with the same standardized law, up to sign.

[Sturma et al.](https://arxiv.org/abs/2302.00993) give identification results
under restricted unpaired linear latent models. We do not assume that our
counts satisfy all their conditions. The within-assay count correction is
inspired by [discrete ICA](https://proceedings.neurips.cc/paper_files/paper/2015/file/9be40cee5b0eee1462c82c6964087ff9-Paper.pdf).
The generalized-covariance branch is inspired by
[Podosinnikova et al.](https://proceedings.mlr.press/v48/podosinnikova16.html);
their paired multiview theorem does not become an unpaired theorem here.
The [independent grounding review](reviews/estimator_grounding.md) distinguishes
published constructions from the local extensions.

## Nested panels and depth

The 300-feature panel preserves the earlier 30-feature panel byte for byte.
Additional features pass the same training-only detection and nonconstant
filter and are ordered by a hash of feature identity. There is no variance
ranking, HVG selection or PCA input. Covariance eigendecomposition is an
internal source-model calculation. All original cells and split identities
are retained; the [extraction audit](reviews/data_extraction_audit.json)
reconstructs the panels from the existing source files.

For both panel widths, define the observed depth proxy

$$
d_i=\text{library depth}_i-\sum_{j\in\text{300-feature panel}}Y_{ij}.
$$

The source-fit/calibration partition is the earlier deterministic 75%/25%
hash partition. Let $d_0$ be the source-fit median of $d_i$. The three exposure
treatments are $t_i=1$, $t_i=d_i/d_0$, and $t_i=(d_i/d_0)^\beta$.
For the third, fit one assay-level slope $\beta\in[0,2]$ from the source-fit
300-feature total using a profiled Poisson quasi-likelihood. With
$z_i=\log(d_i/d_0)$ and $y_i=\sum_jY_{ij}$, minimize

$$
L(\beta)=\left(\sum_i y_i\right)\log\sum_i e^{\beta z_i}
-\beta\sum_i y_i z_i.
$$

The fitted slope and normalizer remain fixed across variants, panel widths,
controls and bootstraps. This is conditional nuisance adjustment; a depth
slope can include biology. The proxy also includes overlapping genomic
features and is not known technical exposure. It differs from the previous
pilot's outside-30 proxy, so controlled depth comparisons use the new block.

## A common target under three count kernels

For sensitivity parameters $\theta>0$ and $q\in(0,1]$, use

$$
D_j\sim\operatorname{Bernoulli}(q),\qquad
N_j\mid\Lambda,t\sim\operatorname{NB}(t\Lambda_j/q,\theta),\qquad
Y_j=D_jN_j.
$$

The negative-binomial arguments are mean and shape. Conditional on intensity
and exposure, $D_j$ and $N_j$ are independent, and feature kernels are
independent across features. The Poisson limit has $\theta=\infty$; $q=1$
removes the excess-zero component. Every sensitivity keeps
$\mathbb E[Y_j/t\mid\Lambda,t]=\Lambda_j$. Neither shape nor retention is
inferred as technical truth from marginal overdispersion or zeros.

For falling factorial $(Y)_r$ and
$c_r(\theta)=\prod_{\ell=0}^{r-1}(1+\ell/\theta)$,

$$
f_{rj}=\frac{(Y_j)_r q^{r-1}}{t^r c_r(\theta)},
\qquad \mathbb E[f_{rj}\mid\Lambda,t]=\Lambda_j^r.
$$

Repeated feature indices use their appropriate factorial order. Write
$r_j=Y_j/t$ and

$$
\begin{aligned}
h_2&=f_2-r^2,\\
h_3&=f_3-3f_2r+2r^3,\\
h_4&=f_4-4f_3r-3f_2^2+12f_2r^2-6r^4.
\end{aligned}
$$

The intensity covariance estimate is
$\widehat{\operatorname{Cov}}(r)+\operatorname{diag}\overline{h_2}$.
For a fixed projection $w$, let $z=w^\top(r-\mu)$ and
$H_s=\sum_j w_j^s h_{sj}$. Corrected projected moments about that center are

$$
z,\quad z^2+H_2,\quad z^3+3zH_2+H_3,\quad
z^4+6z^2H_2+3H_2^2+4zH_3+H_4.
$$

These provide skewness and excess-kurtosis estimates on calibration cells.
The resulting ratios are finite-sample estimates, not unbiased source laws.
The ordinary ablation uses observed moments without a noise correction and
shares the corresponding Poisson-selected dimension.

## Rank, orientation and stability

Rank selection compares positive eigenvalues of the standardized corrected
covariance against the largest eigenvalue from each of 49 independent-column
permutations. The retained empirical tail probability is at most 0.05. Rank
is capped at six; exceeding the cap prevents matching. Rank zero is allowed.
The permutations preserve eight ranked exposure strata for depth-adjusted
arms and one stratum for unit exposure. This is a diagnostic null for coherent
within-assay structure, not a biological false-sharing FDR calibration.

Three orientations are compared:

1. **Corrected third moments:** contract directly into the retained source
   subspace and compare eight seeded tensor contractions. No feature-cubic
   tensor is allocated.
2. **Uncertainty shrinkage:** use five fixed row groups and the same fitted
   mean and whitening to estimate entrywise tensor variance
   $V=\sum_gn_g(T_g-T)^2/[(G-1)n]$. Fit the orientation to
   $\widetilde T=T^3/(T^2+V)$, with $0/0=0$. Eligibility still uses the
   unshrunk tensor residual. This local regularizer depends on the whitening
   basis and is not rotation invariant.
3. **Bounded Poisson generating functions:** for eight nonpositive processing
   vectors $s$, use $g_i=\prod_j(1+s_j/t_i)^{Y_{ij}}$. Under the Poisson kernel,
   $\mathbb E[g_i\mid\Lambda_i,t_i]=e^{s^\top\Lambda_i}$. The log-transform
   Hessians share independent-source loading directions. Observed bases are
   at least 0.5; this branch uses unit exposure, no excess zeros and no NB
   correction. The worst effective sample size across processing points must
   be at least $\max(50,0.2n_{\mathrm{fit}})$. Assay-specific tilted variances
   are never used to match sources.

All orientations have a residual gate of 0.35. Thirty-two independent
within-assay source-fit/calibration bootstrap resamples refit the complete
source estimator at fixed rank, exposure parameters and noise kernel. Signed
loading alignment uses the point fit's working feature scale. A candidate needs at
least 26 valid moment replicates, median loading cosine at least 0.9 and
10th-percentile cosine at least 0.7.

An exact constant-rate feature keeps observed standard deviation zero but
uses an internal working scale of one. Exact centering makes its covariance
cross-row zero, and its corrected diagonal is zero or negative. This avoids
aborting an entire sparse bootstrap while preserving the other positive
modes; it adds no ridge and drops no feature. Insufficient positive modes
still cause abstention. This preflight completion-rule correction applies to
every new arm and differs from the preceding pilot. Actual zero observed
scales remain unscorable in fixed-scale evaluation.

## Two matching stages

The first fingerprint is skewness and excess kurtosis, with practical margins
$(0.25,0.5)$. For each source pair and sign, normalize the fingerprint
difference by those margins. Accept finite-moment equivalence only if the
maximum absolute point difference plus the 90th-percentile bootstrap band
radius is at most one. At least 26 jointly valid bootstrap replicates and both
source eligibility gates are required. Record all equivalent signed pairs,
then separately record pairs with mutually unique source and sign matches.

Only endpoints of equivalent signed pairs receive the secondary CF check.
Ambiguous pairs are included. At frequencies $u\in\{0.125,0.25,0.5\}$, the
per-feature inverse-kernel factor for a direction $w$ is

$$
g_y=1+\frac{{}_1F_1(-y;\theta;-iuw\theta q/t)-1}{q},
$$

or $1+[(1+iuwq/t)^y-1]/q$ in the Poisson limit. Multiply the feature factors
and apply phase $\exp(-iu w^\top\mu)$. Under the declared conditional kernel
and integrability assumptions, its expectation is the intensity projection's
characteristic function. Ordinary arms use the empirical projected-rate CF.

CF acceptance requires finite numerical diagnostics, maximum log-weight at
most 300, empirical complex standard error at most 0.1 and magnitude at most
1.1 at every frequency. Finite-shape kernels also require the declared
polynomial-conditioning proxy at most $10^{-8}$. This proxy is a heuristic,
not a certified floating-point error bound. No weights are clipped. Raw
rejected estimates and their bootstrap diagnostics are saved for audit.

The combined eight-coordinate fingerprint contains the two moments, three
real CF parts and three imaginary CF parts. CF margins are 0.1. Apply the same
band rule and then mutual uniqueness. CF can resolve an ambiguous moment
match, so the confirmed unique count need not be below the moment unique
count. Unassessed CFs are labeled unassessed, never confirmed.

Neither two moments nor a finite CF grid proves equality of full source laws
or identity of biological mechanisms. These selected checks do not establish
nominal post-selection coverage or biological FDR.

## Evaluation and interpretation

All 225 arm records and hashes must be sealed before the separate evaluator
loads reference values. Real evaluation uses the same 595 paired test cells
reserved previously. Their pairing is absent from training. Simulated state
references remain evaluator-only and belong to existing public-discovery
splits; the separate focal DAG is not used as background-source truth.

For a uniquely matched source pair, the candidate cross-covariance is the
signed outer product of its two loading vectors, summed over matches. Report
moment-only and CF-confirmed predictions separately, including zero when no
candidate survives. Compare both with heldout paired covariance and the zero
baseline using fixed source-fit observed-rate standard deviations. Pairwise
blocks need not assemble into a positive-semidefinite joint covariance:
they are candidate moment predictions, not a recovered joint distribution.

Report full-panel errors and the unchanged 30-feature anchor block separately.
Only identical exposure treatments have the same target and scale. For
within-assay prediction, restore the assumed observation-noise diagonal only
for corrected models and compare full/off-diagonal errors with a diagonal
training baseline. Equal covariance scores across orientations can occur
because $BB^\top$ is invariant to orthogonal source rotations.

Two hundred real-test bootstrap resamples preserve complete paired triplets.
An independently checked row-Gram identity calculates errors without
allocating a large covariance cube. These intervals condition on the fitted
model and observed evaluation cohort. Simulation readouts show correlations
between noisy rate projections and existing state coordinates for all sources
and both matched subsets; these are not causal-edge scores.

The independent [known-kernel derivation](reviews/kernel_population_identifiability.md)
shows that known constant exposure, shape and retention can preserve the full
within-assay intensity law at the population level. It does not cover unknown
technical noise, endogenous depth proxies or the missing cross-assay coupling.
Finite-estimator failure therefore cannot establish a universal count-data
impossibility.

Only two or more sign-consistent, all-assay CF-confirmed candidates permit
the unchanged native Sturma graph extraction. Any resulting coefficients are
exploratory anonymous latent coefficients. All failed arms, warnings,
bootstrap failures and abstentions remain in the report. No model or seed is
selected and no threshold is widened using evaluation outcomes.
