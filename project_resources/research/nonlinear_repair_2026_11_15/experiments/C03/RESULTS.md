# C03: author moment estimator on existing count panels

2026-11-15. Completed fitting, frozen evaluation and independent numerical replay. [Plan](../../plans/C03_MEITD_VALIDATION.md), [prefit review](../../reviews/c03_independent_prefit/REVIEW.md), [postfit review](../../reviews/c03_independent_postfit/REVIEW.md).

This is the Zhang–Kileel baseline off-diagonal moment estimator applied to existing biological and cached simulation count panels. It is not the full ALS++ procedure or a newly generated theorem-satisfying benchmark. Four components, three starts, orders one through four, solver settings, source bounds, controls and scoring were fixed before evaluation. Fitting used no paired, state or causal labels. The independently checked simplex quadratic-program adapter preserves the intended constrained update; it does not introduce a negative-rate projection or a predictive-score selection rule.

All 36 applications and 108 starts completed without timeouts. Of the starts, 34 met convergence criteria and 74 did not; the lowest final moment objective selected 13 converged and 23 nonconverged fits. All six original collider applications, six collider null applications and six chain null applications selected nonconverged fits. Six selected models assigned zero probability to at least one training row; eight did so on calibration. Such fits remain valid moment-estimation outcomes with negative-infinite predictive log scores. They were neither dropped nor repaired with a probability floor.

Independent replay verified all 108 final moment objectives, all 36 selections, active-feature reconstruction, bounds and weight normalization, training/calibration likelihoods, all 54 C03 heldout metric views, 108 C01 comparator views and 46 available C02 views. Six C02 applications were unavailable because that experiment failed to fit them. Replay does not certify global optimization or rerun every stochastic trajectory.

Of 54 C03 evaluation views, 42 have finite predictive log scores and 12 have negative-infinite scores. There are 164 impossible-row occurrences summed over these views; overlapping views mean this is not a count of unique cells. On the 36 full-panel application views:

| Comparator | C03 better | C03 worse with finite score | C03 negative-infinite | Comparator unavailable |
|---|---:|---:|---:|---:|
| C01 K=1 | 14 | 11 | 11 | 0 |
| C01 K=4 | 2 | 23 | 11 | 0 |
| C02 | 10 | 12 | 8 | 6 |

These are descriptive comparisons of different estimators and model restrictions, not a universal ranking. An off-diagonal moment criterion need not optimize full predictive likelihood. Nonconvergence, finite-sample moments, sparse histone measurements, source-imposed rate bounds, model misspecification and count noise remain possible contributors. The experiment does not isolate them completely. It supplies no cross-assay alignment certificate, shared-state coupling or causal graph.

The initial direct evaluator stopped at its protocol guard before reading test arrays: scoring imports loaded additional one-thread MKL/OpenMP libraries absent from the controller snapshot. A separate [runtime admission](runtime_admission.json) and [independent amendment review](../../reviews/c03_independent_postfit/RUNTIME_AMENDMENT_REVIEW.md) retained the original six numerical sources, protocol and fit artifacts. The exact launcher, three runtime-library hashes and full actual protocol are sealed; only the exact documented threadpool-list difference was admitted. This provenance correction did not change predictions or scoring.

Fit seal SHA256: `16b043cc977279f2dda3b000d4da523f568497d44e0f9b1c4d6b0899e4d7c5f9`.
Evaluation seal SHA256: `e743a6ccb8174e724cc613c1c7f729507884dace0f732ddcc5149b41d638f43e`.
Runtime sidecar seal SHA256: `5d6da9f68e625a29d199aac999e1dbba187b15e4fef867fba8911e4ac2464f7b`.
