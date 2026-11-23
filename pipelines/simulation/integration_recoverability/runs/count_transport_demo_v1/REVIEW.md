# Independent count-transport review

Reviewed 2026-11-15 by the design-review agent before assessment evaluation,
then updated after the sealed evaluation. The implementation and execution
design passed review. **The empirical demonstration did not recover useful
cross-assay count dependence on these panels.** No assessment count values,
latent truth arrays, or fitted outcome scores were inspected during the
pre-evaluation review. The results section below uses the subsequent sealed
scores and cached generator metadata; it introduces no refitting or selection.

The reviewed [protocol](protocol.json) has SHA256
`98c44abae5663cd2bc3bac6ecf92a00354cf3029b303de131392e21b3c1859ed`.
Its source hashes identify the code and tests below; this review is a subsequent
audit artifact and does not change the frozen protocol or its sources.

## Reviewed implementation

- [Model](../../count_transport_model.py), [runner](../../count_transport_run.py),
  [evaluator](../../count_transport_evaluate.py), their three corresponding
  [test files](../../tests/), and the [frozen plan](PLAN.md).
- Historical C01 validation, identifier split and within-partition permutation
  helpers; cached-panel extraction and evaluation contracts; the existing
  simulator and count-calibration code; E/E2 assumptions and the tall-matrix
  argument.

The product-Bernoulli weights, standardized independent sources, common learned
source probabilities, shared strict-model source mean, exact marginal Poisson
likelihood and analytic gradients are consistent. The affine sensitivity model
uses centered sources and separate intercepts. Neither implementation rematches
sources independently by assay or uses truth to align coordinates.

Independent central-difference checks for both variants at dimensions 1, 2 and 3,
including probabilities near the permitted range endpoints and nonzero source
means, agreed with the analytic gradient to a maximum absolute error below
`4.2e-11`. These were deterministic numerical fixtures, not generated benchmark
worlds. All **34 model, runner and evaluator tests passed** in the independent
preflight run. An initialization issue was corrected before freezing: strict
starts now alternate source polarity while preserving initial mean log rates.

The conditional prediction formulas implement the source posterior followed by
the target Poisson mixture. Cross-assay covariance correctly contains covariance
of the rates without adding independent observation noise; count correlations
include Poisson variance in each marginal denominator. Paired observed count
covariance is an empirical reference, not an oracle rate covariance.

## Information boundary and selection

The fit runner reads only the allowed public training counts, feature IDs and
cell IDs. The freeze stage records assessment paths and manifest hashes without
opening assessment arrays. Pairing, latent state, causal truth, guidance and
library-depth adjustments do not enter fitting. Historical feature eligibility
preceded the current split, so calibration is conditional on that existing panel.

The original identifier split is preserved. Independent column permutations
are confined to fitting and calibration partitions and reused across candidate
models; they preserve each partition's feature marginals and fitting scales.

Selection uses calibration likelihood among converged finite fits, with the
declared dimension/seed tie rule. Rank and probability-separation gates remain
separate from selection. The best finite nonconverged fallback is diagnostic
only; an entirely failed group stops before assessment. All 96 fit outcomes are
saved before evaluator access. Code, tests, helpers, manifests, feature identities,
the plan and inputs are pinned. Assessment checks verify those pins, feature
order, unique and disjoint training/evaluation IDs, and identical paired row order.

## Claim boundaries

The existing worlds have private noise, a Gaussian state coordinate, nonlinear
measurement transformations and depth/count effects outside the fitted model.
This experiment tests useful recovery under misspecification; success would not
verify the E/E2 assumptions, and failure would not refute the population result.
These reused background panels exclude the planted motif, so they cannot test
its causal adjacency or direction recovery.

Distinct asymmetric source laws and full loading rank are required model
conditions. The declared probability-gap and singular-value-ratio thresholds
are numerical screens, not statistical identification guarantees. Optimizer
convergence, boundary hits and restart variability must remain visible.

For a fixed joint model, the per-cell conditional log-score gain over its target
marginal is identical in the two prediction orders: six ordered scores represent
three unique assay-pair associations per world. Mean-prediction errors and
per-feature normalizations can differ by prediction order. None identifies a
causal direction.

Paired-row bootstrap intervals condition on fixed fitted models and omit fitting
and model uncertainty. The family-adjusted percentile tails with 1,000 replicates
are approximate and exploratory. Fixed source-shuffle comparisons introduce
permutation dependence between rows; their interval summaries are descriptive.
Restart ranges are also descriptive and must not be treated as confidence sets.

## Results review

All 96 fit artifacts were retained, with no execution exceptions. The protocol
and source pins, every fit artifact hash, and the evaluation/fit seal links were
independently verified. The [evaluation](evaluation.json) hash is
`e523d8575be4a2e360fd17d0538a3fa7b05a1d143898601b5f43501b53c00364`.
Its audit records 2,000 paired assessment cells per world, verified feature order
and no training/assessment overlap. The completed
[independent numerical replay](NUMERICAL_REVIEW.json) passed 136 hash comparisons,
recomputed all 288 assay calibration scores, verified eight selections and
checked 48 ordered selected-model prediction readouts. Maximum per-cell numerical
discrepancy was below `7.4e-13`. The replay did not repeat bootstrap quantiles or
unselected-restart assessment calculations. The present review independently
checks scientific interpretation, selection diagnostics and generator compatibility.

The table reports per-cell log-score gains over the same fitted target marginal.
The three columns are unique assay-pair association scores, in natural-log units.
The covariance ratio compares the model's standardized RMSE with the zero
cross-covariance baseline; a ratio below one would be better.

| World and model | Selected dimension | Numerical gates | RNA–H3K27ac gain | RNA–H3K4me3 gain | H3K27ac–H3K4me3 gain | Covariance RMSE ratio range |
|---|---:|---|---:|---:|---:|---:|
| Linear, strict | 1 | Passed | -0.792 | -0.563 | -0.213 | 1.25–2.02 |
| Collider, strict | 2 | Source-gap failure | -1.118 | -1.342 | -0.542 | 3.12–4.08 |
| Linear, affine sensitivity | 3 | Passed | -1.529 | -2.081 | -0.749 | 3.06–3.77 |
| Collider, affine sensitivity | 3 | Source-gap failure | -1.824 | -1.949 | -0.659 | 2.43–3.57 |

All 12 unique original-data pair scores have family-adjusted interval upper
endpoints below zero. Conditional-mean prediction also has higher MSE than the
fitted target mean in every ordered comparison. Every original-data restart at
the selected dimensions has negative mean pair gains; the negative outcome is
not confined to the chosen seed. The original selected fits show no positive
adjusted gain over the source-shuffle comparison. Null-trained selected fits
also have negative gains against their target marginals, so the controls provide
no rescue of the recovery claim. Their numerical acceptance alone illustrates
why rank, convergence and source separation are not recovery evidence.

Optimization limits matter. Only 21/48 strict fits converged: all 16 dimension-3
fits and 11/16 dimension-2 fits reached the 600-iteration cap. The best finite
dimension-3 calibration scores exceeded the selected strict fits in both worlds,
but those candidates were excluded by the frozen convergence requirement.
Consequently this experiment does not establish the best achievable result for
a better optimized strict estimator. All 48 affine fits converged, so the affine
negative results cannot be attributed solely to that iteration cap.

The selected collider strict and affine fits have probability gaps of 0.00647
and 0.01797, respectively, below the 0.03 screen. The two selected linear fits
pass all numerical gates but still predict the joint distribution poorly.
All selected original fits also report at least one parameter-bound hit. These
diagnostics remain part of the outcome; no threshold or iteration limit was
changed after inspection.

## Generator and cache compatibility after evaluation

The current [simulator](../../simulator.py) and
[count sampler](../../calibration.py) match the source hashes in the
[frozen world-bank manifest](../worlds/extended_bank_manifest.json).
Each world's saved `frozen_parameters.json` also matches its manifest hash.
The following checks found no designed training-to-assessment mapping change:

- `write_world` freezes one parameter object before any cohort and reuses its
  feature IDs, loadings, nonlinear forms, count parameters and exposure
  distributions for every cohort. The same informative-regime multipliers apply
  to training and assessment.
- `_cohort_population` draws one state array for each assessment population.
  `_assay_matrix` uses every row of that same population for all three assessment
  assays. Training instead uses disjoint assay assignments from the same state
  distribution. The state scaling uses fixed population moments, without
  cohort-specific normalization.
- Feature identifiers and source-column indices are pinned by the historical
  panel and verified by the evaluator. The identical paired assessment IDs
  therefore refer to the same generated biological cells and state, not to an
  arbitrary cross-assay match.

The relevant mismatch is the fitted model class, rather than a documented
cohort mapping shift. Cached metadata show these selected feature classes:

| World | Assay | Shared-state features | Assay-private features | Null features |
|---|---|---:|---:|---:|
| Linear | RNA | 19 | 6 | 5 |
| Linear | H3K27ac | 9 | 10 | 5 |
| Linear | H3K4me3 | 11 | 3 | 4 |
| Collider | RNA | 16 | 7 | 7 |
| Collider | H3K27ac | 8 | 5 | 11 |
| Collider | H3K4me3 | 10 | 3 | 5 |

Private features depend on an assay-specific Gaussian factor. Every background
feature additionally receives its own Gaussian activity noise with standard
deviation 0.5. Shared features mix linear and tanh state responses; counts use
an exposure-scaled softplus mean. Exposures are drawn separately by assay from
the same frozen assay-specific distributions in each cohort. Many selected
features also have finite negative-binomial dispersion; the remaining features
use the Poisson fallback. The true state includes a symmetric Gaussian coordinate,
which conflicts with the fitted asymmetric, symmetry-free source law.

An all-shared, low-dimensional Poisson source model can therefore attribute
assay-private or measurement variation to a source that it forcibly shares with
other assays. That is a plausible explanation for inflated or incorrect
cross-assay predictions, not a mechanism isolated by this experiment. The
private factors and observation model are stable across cohorts and should not
be described as distribution shifts.

The justified conclusion is a failed practical demonstration for this finite
source estimator, frozen optimization budget and these reused background panels.
It is not a falsification of E/E2, a proof that every count-aware estimator must
fail, or a causal-graph recovery result. No new worlds, extra measurements,
truth-informed fitting, model changes or assessment-driven reselection were
introduced during this review.
