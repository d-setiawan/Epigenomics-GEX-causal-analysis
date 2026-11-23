# Exploratory count-native source search without parameter boxes

Prepared before this experiment's fitting, after the sealed matched-model and
labeling experiments. This is a separately recorded exploratory estimator test
using the same data. Earlier results remain intact. The user authorized removing
the identified optimization restrictions; no new simulated or measured data are
needed for this experiment.

## Question and scientific target

The earlier estimators recovered the one-source worlds but reached suboptimal
stationary points in the higher-dimensional worlds. Their true generating
parameters achieve better training objectives. Allowing both binary-label
orientations did not resolve that failure. This experiment asks whether removing
finite parameter boxes, changing coordinates and initializing from count-native
source separation improves recovery.

The model remains strict E2: one common population of independent Bernoulli
sources, full-column-rank assay maps, log-linear rates, and Poisson observations
independent across features and assays conditional on the common source. The
true source dimension is supplied. No assay-private factors, free assay-specific
intercepts or extra measurements are introduced. The targets remain shared-state
log-rate transport and the model's cross-assay count law, not genomic causal
directions.

## Exact change of coordinates

For binary state vector $B$ with independent coordinates of probabilities $p$,
write the log-rate vector of assay $c$ as

$$
\eta_c=A_c(B+h).
$$

The common vector $h$, all entries of $A_c$, and the probability logits are
unbounded real optimizer coordinates. Probabilities are logistic transforms;
there are no finite boxes on logits, loadings or shifts. For interior $p$, this
is exactly the earlier strict model, using coordinatewise multiplication and
division in

$$
M_c=A_c\operatorname{diag}\!\left(\sqrt{p(1-p)}\right),\qquad
\mu=\frac{h+p}{\sqrt{p(1-p)}}.
$$

Conversely, $A_c=M_c\operatorname{diag}(1/\sqrt{p(1-p)})$ and
$h=\sqrt{p(1-p)}\,\mu-p$. The probabilities affect mixture weights, while $A_c$
and $h$ determine rate atoms. This avoids changing the rate atoms whenever a
source probability changes.

Canonicalization is global. For a coordinate with $p_j>1/2$, replace $p_j$ by
$1-p_j$, $h_j$ by $-h_j-1$, and column $j$ of every $A_c$ by its negative.
Complementing that bit preserves all marginal and joint laws. Sorting source
probabilities permutes the same coordinates of $h$ and every assay loading
matrix. Preserve raw and canonical optimizer representations. Exactly symmetric
or numerically degenerate probabilities are retained as explicit diagnostic
failures when they cannot be represented in the accepted canonical source class.

## Two frozen estimator modes

Both modes use the same exact marginal count likelihood, averaged per cell and
feature within each assay and then averaged equally across assays.

**`plain_unbounded`.** Convert each of the four legacy seeded initializations
into $(p,h,A_c)$ without changing its physical state weights or rate atoms.
Refine the exact likelihood without parameter boxes or total iteration/evaluation
budgets. This arm measures the effect of the new coordinates and domain with the
original initialization strategy.

**`source_search`.** For each of four outer seeds, carry out the following
count-native initialization and refinement using fitting counts only.

1. In each assay, fit four unrestricted product-Poisson mixtures with
   $K=2^d$ components from independent Poisson k-means++-style count-space
   initializations. Choose raw count rows by their minimum Poisson deviance
   from already chosen centers. Initial centers are
   `0.8 * selected_row + 0.2 * max(feature_mean, 0.1)`, with any positivity
   floor restricted to initialization, and initial mixture weights are equal.
   Use every feature. No PCA, variance ranking, truth-derived labels, true source
   probabilities or paired cells enter these mixture fits. Preserve every
   mixture outcome and its convergence/failure diagnostics.
2. For each finite, gradient-converged mixture, assign its $K$ atoms to binary
   cube vertices. Nonconverged mixture fits remain diagnostic records and do
   not seed joint refinement.
   Enumerate all assignment orbits modulo global bit complements and coordinate
   permutations. There are 1, 3 and 840 orbits for dimensions 1, 2 and 3,
   respectively, covering all 2, 24 and 40,320 assignments. One component can
   be fixed at vertex zero by bit complementation; sorting the component
   indices assigned to its $d$ neighbors removes the $d!$ coordinate
   permutations. This quotient covers every bijection and does not use truth.
3. Fit affine geometry to the assigned log-rate atoms, infer Bernoulli
   probabilities from the assigned atom weights, and score the cube fit using
   the product-weight KL divergence plus the weight-averaged Poisson KL
   divergence of reconstructed component rates. Divide by assay feature count
   to retain the same per-feature scale. A free affine intercept may be used
   only inside this initialization fit; the joint model must use $A_ch$ with
   one common $h$ and no independent assay intercept. Retain the best orbit
   for each mixture with its score and assignment diagnostics.
4. Canonicalize and sort the inferred source probabilities to align the
   independent mixture initializers across assays. Combine the four retained
   mixture candidates per assay: at most $4^3=64$ combinations. Initialize a
   common probability vector from the assay estimates and a common $h$ by
   feature-balanced stacked least squares. Evaluate every finite combination
   on the public fitting likelihood.
5. Refine the four best finite combinations under the exact common-source
   likelihood. Retain all refined outcomes. The best finite converged inner
   refinement by training likelihood is the outer seed's candidate; a finite
   nonconverged diagnostic may be retained if none converged. Calibration and
   assessment values never select the inner candidate.

The source-search mode contains at most 432 marginal mixture runs and 144 joint
refinements across the nine worlds and four outer seeds. The plain mode adds
36 joint refinements. There are 72 outer records, including explicit failures;
this count must not conceal the additional source-search computation.

## Convergence and finite-search limits

Use an ordinary unbounded gradient infinity-norm tolerance of `1e-6` for joint
refinement. There is no projected finite-box criterion. Solver chunks may have
iteration or evaluation sizes, but exhausting a chunk resumes optimization
automatically without a total iteration/evaluation cap. Use 600-iteration and
15,000-evaluation chunks, objective tolerance zero from the first chunk, and
the 40-step per-line-search safeguard. A nonfinite trial proposal returns an
infinite objective so the line search can backtrack; it is not clipped into a
different model. An accepted nonfinite state or numerical stagnation is an
explicit failure. Relative objective stagnation alone does not qualify as
gradient convergence. The implementation is
[count_transport_source_search.py](../../count_transport_source_search.py).

For product-Poisson mixture EM, check the exact marginal objective gradient in
log-rate and unconstrained mixture-logit coordinates, using the same per-cell,
per-feature normalization. A gradient infinity norm at most `1e-6` qualifies as
convergence. There is no total EM iteration cap. Dead components, zero rates
that prevent finite log-rate representation, nonfinite values, material
likelihood decreases and numerical stagnation above tolerance remain explicit
failures. Initial positive smoothing is initialization only; do not silently
clip fitted rates or weights to obtain a successful outcome.

This is a finite search over independent mixture starts, cube assignments and
joint refinements. Exhaustive cube labeling is conditional on the fitted atoms;
it is not exhaustive optimization over all atom locations or parameters.
Floating-point and line-search failures remain possible. Neither unbounded
parameters, continued iterations nor local stationarity certify global
optimization. Report all remaining limits without describing this as literally
unlimited computation.

## Data, selection and information boundary

Reuse all nine original matched worlds at the 12,000-cell condition, their
2,000 calibration cells per assay, and the original RNA, H3K27ac and H3K4me3
feature panels. No new worlds, sample sizes or modalities are generated.
Run outer seeds 0, 1, 2 and 3 for both modes in every world. Preserve failures,
all inner/search diagnostics and every outer candidate.

Select each world/mode's outer candidate by the highest equal-assay,
per-feature calibration likelihood among finite gradient-converged fits.
Break exact ties by seed order. If none converged, retain the best finite
diagnostic; if none is finite, retain an explicit null selection and include
the world in failure denominators. Rank, source separation, true-parameter
likelihood and recovery metrics do not select the candidate.

The numerical acceptance thresholds remain a minimum canonical $M_c$ loading singular
value ratio of `1e-3`, minimum canonical source-probability gap of `0.03`, finite
parameters and the declared gradient convergence criterion. A single source
has no pairwise probability gap to fail. These are numerical/model-class gates,
not proof that the fitted law is correct.

Pin finalized source files, tests, this plan and all input references before
fitting. The learner may open only fitting and calibration arrays and the
supplied dimension. No generating parameters, hidden source values, paired
assessment counts, previous fitted models or oracle initialization enter the
learner. Seal every fit and selection before evaluation opens private values.

## Frozen evaluation and interpretation

Reuse the existing 4,000 paired assessment cells and exact generating parameters
only after the fit seal. Their previous use is explicit: these comparisons and
intervals are exploratory, not fresh confirmatory tests. The earlier matched
and labeling experiments remain fixed comparators.

Apply the unchanged exact population covariance, restricted log-rate transport,
mean/support transport, conditional predictive score and oracle conditional-gap
metrics. Secondary parameter comparisons use one global source permutation,
never separate truth-assisted assay alignment. Evaluate the true generating
parameters on the same fitting counts only after sealing; a positive fitted
minus true training objective is a known-feasible-point suboptimality witness.
It never stops fitting, chooses restarts or changes the estimator.

An ordered prediction pair passes only when the model is numerically accepted,
exact covariance relative error is at most `0.10`, restricted transport relative
error is at most `0.10`, the nominal 95% lower bound for gain over the fitted
target marginal is positive, and the oracle conditional-score gap is at most
`0.02` nats per target feature. A world passes only if all six ordered pairs
pass. The six per-cell gains represent three symmetric associations, not six
independent relationships or causal arrows.

Use the same 1,000 paired-row bootstrap procedure for selected outer candidates.
Intervals condition on fixed fits and are descriptive, with no familywise or
fresh-data claim. Report continuous metrics, successful and unsuccessful worlds,
all starts, numerical failures and optimization witnesses. Three mechanisms per
dimension do not establish general reliability.

Success would show practical recovery within this assumed, known-dimension
source family on the reused exact-model data. Failure despite a better feasible
point would retain a demonstrated optimization limitation. Neither outcome is a
new proof of the population corollary or a test of biological causal directions.
