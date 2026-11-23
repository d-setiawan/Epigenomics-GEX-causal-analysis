# Exploratory removal of the Bernoulli labeling restriction

Prepared 2026-11-15 after the sealed exact-model experiment, before any fit in
this follow-up. This is one targeted optimization diagnostic, using the same
simulated counts. It is explicitly exploratory: these worlds and their paired
assessment results have already been inspected. Preserve the complete primary
experiment, including its negative higher-dimensional results.

## Motivation and unchanged scientific target

The primary experiment recovered the one-source model. Its higher-dimensional
fits reached constrained stationarity but had worse training likelihood than
the feasible generating parameters. Several fits stopped at the imposed upper
probability bound near 0.4473. In one two-source world, the learned log-rate
transport was accurate while the source frequencies and covariance were wrong.
This motivates allowing either label orientation for each Bernoulli coordinate.
No true parameters, source draws or paired rows enter the repaired estimator.

The model remains independent Bernoulli sources with common unknown laws,
strict log-linear rates, full-column-rank assay maps and conditionally independent
Poisson counts. The target is shared-state transport and the count joint, not
genomic causal directions. The true source dimension remains supplied.

## Optimization parameterization

Use `p = sigmoid(raw_p)` with `raw_p` in `[-5,5]`, permitting probabilities on
either side of one half. Keep the previous loading and source-mean bounds, exact
likelihood, four starting seeds and convergence-driven continuation. Translate
the initial probability parameter so each start represents exactly the same
physical distribution as the legacy initialization. Do not initialize from truth.

The canonical probability range now extends down to about 0.0067, whereas the
earlier range began near 0.0527. Thus the computational domain broadens in
addition to removing the labeling restriction. The generating probabilities
remain inside both ranges. Do not attribute every change solely to orientation.

After fitting, for each coordinate with `p_j > 0.5`, replace its probability by
`1-p_j`, negate its source mean and negate its loading column in EVERY assay.
Complementing the corresponding latent bit permutes the rate/weight atoms and
preserves the entire joint count law. Apply the same transformation globally;
never relabel separate assays independently. Preserve the raw optimizer state
and the canonical state for auditing. An exactly symmetric fitted coordinate
is a retained diagnostic failure, not silently clipped into acceptance.

Remove total iteration/evaluation budgets as in the primary continuation arm.
600-iteration/15,000-evaluation chunks automatically resume; projected gradient
tolerance is `1e-6` and continuation objective tolerance is zero. Line-search and
floating-point failures remain explicit. Finite multistart optimization still
does not certify a global optimum.

## Frozen data and selection

Reuse all nine primary worlds (`d1_rep0` through `d3_rep2`) at the 12,000 fitting
cell condition only, with their original 2,000 calibration cells per assay.
Run seeds 0,1,2,3 for every world: 36 new fits. Keep every outcome, including
failure and invalid numerical screens. If a group has no finite model, report
that status explicitly. No further feature, source-family, sample-size or
initialization search is part of this follow-up.

Select the highest equal-assay/per-feature calibration likelihood among finite
gradient-converged fits, otherwise retain the best finite diagnostic. Break
exact ties by seed order. Do not filter selection on source gap or rank; do not
select on assessment or oracle likelihood. Numerical acceptance uses the same
rank ratio (0.001), source-gap (0.03) and gradient thresholds as the primary
experiment. No fitting or stopping decision uses the true feasible-point check.

## Evaluation and interpretation

Seal sources, input references and every fit/selection before evaluation. Reuse
the primary 4,000 paired assessment cells and exact generating parameters solely
for this evaluation, acknowledging their previous use. Apply the unchanged
covariance, restricted transport, mean transport, conditional predictive and
oracle-gap metrics. Keep one global source permutation for secondary parameter
comparisons. Report all starts and the post-fit true-parameter training-likelihood
comparison to diagnose remaining suboptimal stationary points.

The same descriptive recovery criteria apply: numerical acceptance; exact
relative covariance and restricted-transport errors at most 0.10; positive
nominal95 lower predictive-gain bound; oracle conditional gap at most 0.02 nats
per target feature. Use the same 1,000 paired-row bootstrap procedure for selected
fits; intervals condition on the fit, are exploratory, and are not familywise
guarantees. The primary comparison uses all nine 12,000-cell settings and reports
both successes and failures. Better results would support an optimization-domain
explanation, not provide a new proof or establish biological applicability.

No oracle warm-start, new measured information or additional simulated datasets
are used. The original exact-model result remains the primary experiment;
this follow-up is a separately labeled estimator modification.
