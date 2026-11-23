# Count-transport demonstration on cached simulations

Experiment completed 2026-11-15. **This first estimator did not recover useful
cross-assay count dependence.** It improved within-assay calibration likelihood,
but its cross-assay predictions were worse than predictions using
only the fitted target-assay distribution. This is an empirical negative result
for a finite-sample estimator under model mismatch, not a counterexample to E/E2.

## What was implemented

The [E/E2 corollary](../../../project_resources/research/nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md)
identifies a restricted shared-state transport at the population level. To try
to estimate it, the new [model](count_transport_model.py) represents shared state
as one to three independent, asymmetric Bernoulli sources, each standardized
to unit variance. It fits their probabilities and assay-specific loadings
directly to raw counts using exact Poisson-mixture likelihoods. There is no PCA
or variance-based feature ranking.

The primary model uses

$$
\log R_c=M_c(Z+\mu),\qquad
Y_c\mid Z\sim\bigotimes_j\operatorname{Poisson}(R_{c,j}).
$$

The source law is shared, and every assay has its own loading matrix. A second,
affine sensitivity model uses $\log R_c=b_c+M_cZ$. Its identification requires
the stronger centered-source asymmetry assumption explained in the
[frozen plan](runs/count_transport_demo_v1/PLAN.md). Neither model estimates a
causal DAG merely by estimating this transport.

## Data and information boundary

Only existing informative-count inputs from `linear_chain_0` and
`constrained_collider_0` were used. No worlds, biological data or benchmark counts
were generated. The existing fixed background panel supplied 30 RNA, 24 H3K27ac
and 18 H3K4me3 features. These panels exclude the planted causal motif features.

Each assay contributed 3,000 unpaired fitting cells and 1,000 calibration cells,
using the existing identifier-hash split. Training and selection used only
counts and opaque row/feature identities. Paired cells, latent states, source
classes, guidance, causal labels and library-depth offsets were not supplied.

The existing 2,000 paired assessment cells in each world were opened only after
all models and selections were sealed. Pairing was used solely to evaluate
predictions and observed cross-assay covariance. Input hashes, feature order,
paired row order and disjoint training/assessment identities were checked.
These are reused development worlds, not fresh external validation. Historical
feature eligibility also preceded the current fitting/calibration split.

The search contained 96 fits: two worlds, original versus column-permuted
training controls, strict versus affine models, three source dimensions and four
initializations. Selection used calibration likelihood among converged fits,
without consulting assessment values or filtering for convenient source gaps.
All outcomes were preserved. Fitting took about **78 seconds** with two CPU
workers and one numerical-library thread per worker.

## Primary result

The table reports the strict model chosen in each world. Predictive gain is the
mean log probability of the target counts conditional on the source counts,
minus their log probability under the same fitted target marginal. Positive
values would favor cross-assay prediction; all observed values are negative.

The covariance-error ratio compares predicted cross-assay covariance RMSE with
the error from predicting zero cross-assay covariance. Both errors are scaled
by fitting-cell feature standard deviations. A ratio below one would favor the
estimated transport; all observed ratios exceed one.

| Cached world | Assay pair | Predictive gain, nats/cell | Approximate adjusted interval | Covariance-error ratio |
| --- | --- | ---: | --- | ---: |
| Linear chain | RNA / H3K27ac | -0.792 | [-0.942, -0.659] | 2.02 |
| Linear chain | RNA / H3K4me3 | -0.563 | [-0.659, -0.479] | 1.56 |
| Linear chain | H3K27ac / H3K4me3 | -0.213 | [-0.264, -0.166] | 1.25 |
| Constrained collider | RNA / H3K27ac | -1.118 | [-1.325, -0.929] | 4.08 |
| Constrained collider | RNA / H3K4me3 | -1.342 | [-1.527, -1.171] | 3.34 |
| Constrained collider | H3K27ac / H3K4me3 | -0.542 | [-0.640, -0.449] | 3.12 |

The selected linear-world model has one source and passes the numerical screens.
The selected collider model has two sources and **fails the source-separation
screen**: its probability gap is 0.00647, below the declared 0.03 threshold.
Its metrics are retained as diagnostic failures, not accepted recovery.

The affine sensitivity model also failed on all six distinct world/pair cases.
Its predictive gains range from -0.659 to -2.081 nats/cell, and its covariance
errors are 2.43 to 3.77 times the zero-covariance baseline. Its selected collider
fit also fails source separation. Mean-prediction MSE worsened in all 24 ordered
original-data cases across both variants and worlds.

This result is not caused solely by choosing an unfortunate initialization.
Every evaluated restart at the selected dimensions had negative mean
cross-assay predictive gain. Column-permuted training controls and source-row
shuffle checks also produced negative gains. All 32 evaluated fits, including
controls and failed numerical screens, are retained in the report.

## What the negative result means

Within-assay calibration improved over an independent-Poisson baseline in every
assay for each selected model. Nevertheless, the proposed joint distributions
performed worse than their own independent marginals. Better marginal fit did
not establish correct cross-assay coupling.

The cached generator falls outside the fitted E/E2 submodel. It has continuous
state, a Gaussian state coordinate, shared and assay-private background factors,
feature-specific noise, nonlinear count links, variable assay depth and
overdispersed count mechanisms. The training and assessment cohorts do reuse
the same feature layout and measurement parameters; the failure is not explained
by an intended change in feature meanings between cohorts.

One plausible failure mechanism is that a model treating every source as shared
turns assay-private variation into false cross-assay dependence. This experiment
does not isolate that mechanism from source-law mismatch, count-kernel mismatch
or optimization limitations. Of 96 fits, 69 converged and 55 passed all numerical
screens. Twenty-seven hit the 600-iteration limit; every strict three-source fit
did so. Better-calibration nonconverged candidates were excluded by the frozen
rule. No optimizer extension or assessment-informed reselection was performed.

We therefore have an implemented, testable count-native transport estimator and
a reproducible negative demonstration. We do **not** have a successful empirical
demonstration of the corollary, reliable causal recovery, or evidence that every
algorithm must fail on these simulations. The next methodological question is
whether shared variation can be separated from assay-private variation under
assumptions that are both identifiable from unpaired data and appropriate for
these counts.

## Uncertainty and independent validation

Per-cell predictive gains are symmetric under reversing a pair:

$$
\log p(y_b\mid y_a)-\log p(y_b)
=\log p(y_a,y_b)-\log p(y_a)-\log p(y_b).
$$

Thus six ordered readouts per world are three unique association statistics,
not evidence about causal direction. Per-target-feature normalization and mean
prediction errors can differ by direction; those are available in the CSV.

The intervals use 1,000 paired-row bootstrap replicates and condition on fixed
fitted models. They exclude parameter and model uncertainty. The adjusted tails
are approximate and exploratory, with about two bootstrap replicates per tail;
symmetric directions are conservatively counted twice. Fixed source-shuffle
intervals are descriptive because the permutation links observations. Covariance
ratios are point diagnostics, and their reference is finite-sample observed count
covariance, not oracle latent-rate covariance.

All **34 focused tests passed**. Independent preflight review checked the model,
training/evaluation separation, selection and metric formulas; extra numerical
gradient checks agreed within $4.2\times10^{-11}$. Independent post-result
replay verified 136 hashes, recomputed all 288 assay calibration scores and all
eight model selections, and independently reproduced all 48 selected directional
evaluations. The largest per-cell log-likelihood discrepancy was
$7.39\times10^{-13}$; no numerical issues were found. Bootstrap quantiles and
unselected restart metrics were not independently replayed. Tiny deterministic
algebra/count fixtures in software tests are separate from the reused scientific
benchmarks.

## Artifacts and reproduction

- [Frozen plan](runs/count_transport_demo_v1/PLAN.md) and [protocol with source/input hashes](runs/count_transport_demo_v1/protocol.json).
- [All 96 fits and frozen selections](runs/count_transport_demo_v1/fit_seal.json).
- [Full evaluation](runs/count_transport_demo_v1/evaluation.json), [evaluation seal](runs/count_transport_demo_v1/evaluation_seal.json) and [selected metrics CSV](runs/count_transport_demo_v1/selected_metrics.csv).
- [Scientific and implementation review](runs/count_transport_demo_v1/REVIEW.md) and [independent numerical replay](runs/count_transport_demo_v1/NUMERICAL_REVIEW.json).
- [Model](count_transport_model.py), [fit runner](count_transport_run.py), [evaluator](count_transport_evaluate.py) and [tests](tests/).

Use `/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python` and the commands in the
frozen plan. Existing sealed outputs cannot be overwritten. Reproduction needs
the already-cached inputs, which were not copied or regenerated here.
