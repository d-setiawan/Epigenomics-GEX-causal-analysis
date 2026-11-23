# Independent labeling follow-up review

Preflight and sealed scientific results reviewed 2026-11-15. The optimizer,
runner, evaluator and [plan](PLAN.md) have no outstanding scientific
implementation blockers. The labeling change preserves the one-source success
but does not recover complete two- or three-source cross-assay structure. This
follow-up was motivated by the already evaluated primary experiment and is
exploratory, not a fresh test.

## Optimizer and model identity

Reviewed [label optimizer](../../count_transport_label_optimizer.py) and its
tests. Its exact marginal likelihood and analytic derivatives use the correct
logistic derivative, `dp/dlogit(p) = p(1-p)`. Independent reconstruction with
automatic differentiation at fixed one-, two- and three-source fixtures,
including probabilities above one half, agreed with the gradient within
`8.4e-17`. These checks used fixed numerical fixtures, not scientific fits.

The four initializations preserve the original physical probability law exactly:
only the coordinate encoding of each starting probability changes. The means,
loadings and starting count distributions remain unchanged.

Canonicalization negates the appropriate logit, common source mean and loading
column in every assay together. The resulting global complement of latent bits
permutes weights and rate atoms consistently across assays, preserving the full
joint law as well as each marginal. It is not separate truth-based matching of
individual assays. The symmetric parameter bounds also preserve the norm of
the box-gradient mapping under this reflection.

Raw optimizer state and canonical model fields have explicit separate records
and parameter schemas. Acceptance uses canonical probabilities. Exactly
symmetric fitted probabilities are retained as diagnostic failures, without
clipping them into the accepted source class or crashing later evaluation.

The new canonical domain extends below the old probability floor and above its
old upper bound toward one half. Consequently this is both a change of
parameterization and a broader feasible domain. An improvement would not isolate
binary-label traversal as its sole cause. True generating parameters remain
inside both domains.

## Information boundary, selection and failure handling

Reviewed [runner/evaluator](../../count_transport_label_run.py) and its tests.
Fitting reuses only the public 12,000-cell fitting and 2,000-cell calibration
arrays, plus the supplied dimension. It generates no new observations and reads
no true parameter or paired assessment values. The initialization does not use
the primary fitted results or true parameters.

All 36 starts are retained. Selection uses convergence and calibration likelihood
without filtering by rank, source gaps or recovery scores. The evaluator replays
this selection rule against the newly sealed records. Entirely failed groups
retain explicit null selections and remain in aggregate denominators; no
substitute selected model is invented.

New source and fit seals precede evaluator access to the existing true parameters
and paired assessment. Population, predictive, oracle-objective and restricted
transport formulas are unchanged. Source alignment remains one global
permutation and is evaluator-only. The count-joint and transport targets do not
establish genomic causal direction.

Continuation removes total iteration/evaluation budgets, while finite starts,
parameter bounds, line-search rules and numerical-failure conditions remain.
Local stationarity does not certify a global optimum. Reused-assessment
intervals are descriptive and cannot become confirmatory evidence through a
new fit seal.

## Verification and results status

All 87 tests in the count-transport suite passed independently, including the
11 new optimizer tests and eight new runner tests. The final runner tests cover
complete selection-seal replay, rejection of a tampered selection, preservation
of failed starts, and the public/private information boundary. Scientific
generation, fitting and evaluation were not run by this reviewer during
preflight.

## Sealed scientific results

Reviewed the [fit seal](fit_seal.json), [evaluation seal](evaluation_seal.json),
[evaluation](evaluation.json), [comparison table](comparison.tsv) and
[aggregate table](aggregate.tsv), with the primary 12,000-cell convergence
results as the fixed comparison. The evaluation output hash is
`305d876c258e8a3d4ae9840ce4de55a773b8dcf41a31debb6df1a195fa6f7075`.

| Shared sources | Selected models passing numerical gates | Worlds passing all pair criteria | Higher training objective than the feasible truth, all starts |
| --- | ---: | ---: | ---: |
| 1 | 3/3 | 3/3 | 0/12 |
| 2 | 3/3 | 0/3 | 12/12 |
| 3 | 3/3 | 0/3 | 12/12 |

All 36 starts reached the declared projected-gradient criterion. Across all
starts, 32/36 passed numerical and source-separation checks; all nine selected
models passed. These improvements in acceptance do not establish recovery.
For every higher-dimensional start, the true generating parameters achieve a
strictly smaller training objective on the same public fitting counts. The
selected positive gaps range from 0.00487 to 0.11525 under the declared
equal-assay, per-feature normalization. Thus the higher-dimensional failures
retain a concrete optimization defect even after removing total iteration
limits and allowing both binary label orientations. Gradient convergence
is not a global-optimum certificate.

The one-source results closely reproduce the primary experiment: all three
worlds pass, with worst-pair covariance errors of 1.80–4.45%, restricted
transport errors of 1.98–5.03%, and worst oracle conditional-score gaps no
greater than 0.000131 nats per target feature. These are the same three worlds
and assessments, so this is a consistency check of the modified estimator,
not an additional independent replication.

Two partial outcomes matter. In `d3_rep1`, the RNA–H3K4me3 association passes in
both prediction directions: covariance relative error is 5.58%, restricted
transport errors are 3.39% and 3.59%, and both descriptive gain intervals are
positive. This is one recovered association among three, not two independent
recoveries or a recovered causal direction. The other assay relationships
prevent complete-world recovery.

In `d2_rep0`, the selected calibration score improves from -1.37803 to
-1.36498 and the training gap decreases from 0.01794 to 0.00487. Its worst
covariance error improves from 1.897 to 0.743, while its previously accurate
restricted transport degrades from 0.0270 to 1.438 and the worst oracle
conditional-score gap grows from 0.000288 to 0.480 nats per target feature.
The new fit therefore should not be described as an across-the-board
improvement. Better approximate marginal likelihood did not imply better
coupling for this selected solution.

The probability-domain caveat was active: the selected `d3_rep2` canonical
probability 0.46291 exceeds the primary domain's upper bound, although it
remains asymmetric. No selected probability fell below the primary lower
bound. This confirms that the experiment does not isolate binary-label
traversal from the changed feasible domain.

## Independent result checks and conclusion

Without importing the population evaluator, this reviewer independently
reconstructed all selected Bernoulli states, weights and rate atoms from
probabilities, source means and loading matrices. Recomputed exact population
cross-covariances and restricted transport errors for all 54 ordered
source/target cases agreed with the stored results: maximum absolute metric
discrepancy was `8.9e-16` for covariance and zero for transport.

The model author's separate [numerical replay](NUMERICAL_REVIEW.json) is complete
and passes. Its independently implemented likelihood, moment calculations and
automatic differentiation checked 128 hashes, all 36 physical initializations
and global canonicalizations, 108 assay calibration scores, all 36 training
objectives, nine selections, 54 selected prediction directions and 216,000
paired-row predictions. Selected-fit raw and projected gradients agreed within
`1.8e-15`; full joint-law canonicalization within `2.2e-14`; predictive values
within `5.3e-13`. It did not repeat optimization, bootstrap quantiles or
unselected per-direction assessments. Thus this is independent numerical
implementation plus a separate scientific/code review, not a claim that every
audit was performed by someone who had no role in implementation.

The justified positive result remains finite-sample recovery of the model's
cross-assay joint distribution summaries and log-rate transport in these
one-source worlds. Higher-dimensional complete recovery remains unsolved by
the tested estimator; one additional association is recovered in one
three-source world. The feasible-point witnesses rule out claiming that the
higher-dimensional failure is caused solely by nonidentifiability or sampling
error. They do not prove that optimization is the only remaining difficulty,
that a global solution would meet every finite-sample threshold, or that an
alternative initializer would succeed.

The exact-model data, truth boundary and mathematical transport target remain
valid. These results neither refute the population corollary nor establish
biological causal edges, robustness to misspecification or general reliability.
The primary experiment remains intact; no result was used to reselect the
sealed fits or modify this follow-up's criteria.
