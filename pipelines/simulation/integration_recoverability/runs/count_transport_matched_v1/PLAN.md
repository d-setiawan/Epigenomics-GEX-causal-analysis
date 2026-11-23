# Exact-assumption E2 simulation and convergence experiment

Prepared 2026-11-15 before generation, fitting or assessment access. The user
explicitly requested simulations without model mismatch and fitting without
total optimization-budget limits. This authorizes these new targeted simulated data;
it does not authorize or require additional biological measurements.

## Exact positive model

Use the strict model `log R_c = M_c (Z + mu)` and independent Poisson measurement
noise for every feature and assay conditional on the common source. Sources are
independent standardized Bernoulli variables with probabilities from the prefix
of `(0.12, 0.27, 0.40)`. Their source mean is the corresponding prefix of
`(0.6, -0.35, 0.25)`. These coordinates are non-Gaussian, have unit variance and
distinct asymmetric laws, excluding nontrivial signed-permutation symmetry.

Dimensions are 1, 2 and 3. RNA, H3K27ac and H3K4me3 have 30, 24 and 18 features.
For each dimension, generate three independently seeded mechanism/sampling
replicates. Each assay loading matrix is the orthogonal factor of a seeded
Gaussian matrix, multiplied by `0.45 sqrt(feature_count)`. All loadings have
full column rank and good conditioning. Check that all generating parameters
lie strictly inside the estimator's unchanged parameter domain. There are no
private factors, extra depth effects, Gaussian latent coordinates, zero
inflation, normalization, count clipping or negative-binomial substitutions.

The same source law and measurement mechanism apply to fitting, calibration
and assessment. Training and calibration assays receive independent source
draws and distinct row identifiers. Assessment uses one shared source draw per
cell and independent Poisson noise for each assay. The generator does not call
the estimator's decoder, reducing the chance of a common implementation bug.

## Data and fitting boundary

Generate 12,000 fitting cells and 2,000 independent calibration cells per assay
in each of nine worlds. Fit both the first 3,000 and all 12,000 fitting cells;
these nested sizes share the same held-out calibration and assessment. Generate
4,000 paired assessment cells per world. Seal data and truth hashes before fitting.

The learner receives raw counts and opaque feature/row IDs, plus the source
dimension explicitly supplied as a design condition. This primary positive
control tests parameter and coupling estimation with the correct model order;
it does not test discovery of an unknown dimension. Source probabilities,
means, loadings, source draws and cell pairing remain unavailable to fitting,
initialization and model selection. Do not match features by their loading or
use biological guidance. Every generated feature is retained, without PCA.

## Optimization comparison

Use four fixed initializations (seeds 0, 1, 2, 3), retaining the previous strict
initialization and exact likelihood. For each start, save the previous
600-iteration checkpoint, then continue from its exact parameters until the
box-projected gradient norm is at most `1e-6`. The initial checkpoint retains
the previous relative-objective tolerance; continuation sets that tolerance to
zero and uses the gradient criterion.

SciPy calls are divided into 600-iteration/15,000-evaluation chunks solely for
checkpointing. Reaching either chunk budget always resumes optimization. There
is **no total iteration or function-evaluation budget**. Curvature history resets
between chunks, while parameters are preserved. Actual solver failure or
floating-point stagnation is recorded as failure, not accepted convergence.
No guarantee of finite convergence or a global optimum is claimed. Four starts
are a finite optimization search. Numerical parameter bounds remain unchanged;
the generating parameters are strictly inside them. This removes computation
total iteration/evaluation cutoffs, not every mathematical constraint defining
the fitted model. The numerical solver retains 40 trial steps per line search;
exhaustion is recorded as a solver failure, never treated as successful recovery.
Reported evaluation totals count optimizer calls, excluding some objective
prechecks performed around each chunk.

Total: nine worlds x two sample sizes x four starts = 72 optimization streams,
each producing capped and continuation outcomes (144 fit artifacts). Two CPU
workers with one numerical-library thread each keep memory use modest.

For each world/sample-size/mode, select the best equal-assay per-feature
calibration likelihood among finite gradient-converged fits. Exact ties prefer
the lower seed. If none converge, retain the best finite diagnostic; an
all-failed group stops before assessment. Do not filter selections using the
source-separation or rank gates. Do not refit on calibration cells. Retain every
outcome and evaluate all four starts, never selecting from assessment results.

Numerical acceptance requires convergence, finite rates/objective, each loading
smallest/largest singular-value ratio at least 0.001, and minimum source
probability gap at least 0.03 for dimensions above one. Record boundary hits.
Acceptance is a numerical screen and does not certify a global solution.

## Recovery targets and frozen evaluation

After all fits and selections are sealed, open the true parameter files and
paired assessment counts. Verify source/data/artifact hashes and identifiers.

Primary truth comparisons are exact population cross-assay covariance and the
linear log-rate transport restricted to the true source subspace. For source
assay a and target b, evaluate

`||Mhat_b pinv(Mhat_a) M_a - M_b||_F / ||M_b||_F`.

Also report mean log-rate transport error and weighted true-support transport
error because the source mean is nonzero. Do not score arbitrary off-subspace
matrix extensions as identified quantities. These operator comparisons require
no truth-based alignment. Any secondary source-parameter comparison may use
only one global source permutation shared by all assays, not separate alignments.

On assessment counts, compare conditional prediction with the same fitted
target marginal, with the true independent target marginal, and with the oracle
true conditional distribution. Keep oracle gaps untruncated: finite-sample
noise can produce small negative estimates. Predictive directions do not denote
causal directions. Per-cell gains against fitted marginals are symmetric within
a pair; six ordered readouts contain three distinct association statistics.

As an evaluator-only optimization check, compute marginal training likelihood
at the true generating parameters on the same fitting cells. A fitted normalized
negative log likelihood larger than this feasible value by more than `1e-8`
demonstrates suboptimality to a known feasible point, even if the projected
gradient is small. A lower value does not certify a global optimum. Never use
this comparison for initialization, continuation, model selection or stopping.

Use 1,000 common paired-row bootstrap replicates per world for selected-fit
95% intervals and the paired capped-to-convergence differences. These intervals
are nominal and exploratory, conditional on the fits; they do not cover model
or parameter fitting uncertainty and do not provide familywise coverage over
the grid. Restart comparisons are descriptive point scores/ranges.

A pair meets the predeclared practical recovery criterion if the selected fit
passes numerical acceptance, exact relative covariance error and restricted
transport error are each at most 0.10, predictive-gain nominal95 lower bound
exceeds zero, and oracle conditional log-score gap is at most 0.02 nats per target
feature. These are engineering tolerances, not consequences of the theorem.
Report every component and failures, plus counts across the three replicates;
do not turn three successful replicates into a general reliability guarantee.

## Symmetry countercontrol

Construct an exact analytic dimension-two countercontrol with equal probabilities
`(0.27, 0.27)` and equal means `(0.6, 0.6)`. Swap source columns in only one assay.
Verify that its weighted marginal rate atoms, hence its unpaired Poisson law,
are unchanged, while cross-assay covariance changes. This deliberately violates
the no-symmetry premise and is labeled separately from the matched positive
simulations. It requires no extra fitted dataset.

## Interpretation and reproducibility

Success would demonstrate finite-sample recovery of the corollary's restricted
transport/count-joint target in a correctly specified, favorable source family.
It would not prove the corollary, recover a genomic causal DAG, or establish that
real biological data meet these assumptions. Failure requires separating a local
optimum or numerical failure from finite-sample estimation and theoretical issues.

Run `count_transport_matched.py freeze`, `count_transport_matched.py generate`,
`count_transport_matched.py fit`, then `count_transport_matched_evaluate.py`, in
the existing `scrna_r` environment. Every stage creates outputs exclusively.
Keep the earlier mismatched-world experiment and its frozen sources unchanged.
