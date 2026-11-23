# C03: author-source moment estimator for finite count mixtures

Astra scientific design approved 2026-11-15, before implementation/fitting. Exact source and numerical adapter review, protocol freeze and separate execution approval remain required. This extends C01 with a genuinely different estimator for its finite conditional-independence model; it does not change C01 or C02 outcomes. It is a sequential exploratory existing-input application, not a reproduction of the paper's reported simulation performance.

## Population and implementation scope

Zhang–Kileel (arXiv:2210.14386), Theorem2.8, identifies generic finite-class weights and conditional means from cross-coordinate moments under conditional feature independence. Orders2 and3 with at least9 active features satisfy its dimension inequality for r=4. Positive odd-order information removes the all-even sign ambiguity. Theorem3.9 is local convergence, not a global optimizer guarantee. Product-Poisson feature prediction is an additional specified emission model, justified as the same count subclass tested by C01; the general moment theorem does not itself identify a Poisson kernel or cross-assay coupling.

Use pinned author source commit3cad5393b0dbbd98e53a139bd6ad046d685fac2b under sources/zk_meitd_root. The tested configuration is the repository's baseline alternating update with normalized orders1–4, warmup0 and Anderson depth0. It intentionally does not claim to reproduce full ALS++ warmup/acceleration experiments. Source row clipping occurs throughout the run and differs from the paper's warmup-only description; finite-source behavior is the operative implementation. Preserve author code unchanged except through a separately hashed, reviewed compatibility adapter.

## Existing inputs and fixed configuration

All nine immutable train matrices, both first30/full300 panels and original/column-null controls:36 applications, each with initialization seeds0,1,2. Reuse the exact C01 opaque-ID source-fit/calibration partition and full300 within-partition null permutations before prefix slicing. Rank is fixed r=4; no rank/penalty/iteration search. Train only on source-fit rows. Columns with exactly zero source-fit sample variance are excluded from the solver because author standardization divides by standard deviation; retain their original positions and assign their source-fit mean as a common rate in all four components when restoring the full panel. No variance threshold or heldout values enter this rule. Fewer than9 active features yields an explicit theorem-dimension abstention, not a smaller-rank rescue. Original full-training feature-selection caveat remains.

For each seed use KMeans(n_clusters=4,n_init=1,random_state=seed) on log1p source-fit counts to initialize four raw count-mean vectors; initialize weights uniformly at1/4. Empty clusters or fewer than four distinct vectors fail the start. Pass active raw data transposed, translate_init='data' and regularize_data=True so the author uses source-fit mean and sample standard deviation (ddof1). Use MEITD(d=4) normalized tau_k=(J_active-k)!/J_active!, maxiter100, Mtol=1e-4, wtol=1e-4, ftol=None, warmup0, AA_depth0, update_weights=True, default row-block schedule and default weight lower bound0.1/r=.025. Source convergence actually tests relative mean change and absolute weight change; report this, and preserve its internal seed12345 for row permutations. Source ftol does not activate its fconv flag, so ftol=None avoids pretending objective-based stopping.

Preserve all starts, warnings, iteration/step traces and true final parameters. Recompute the final moment objective after the last weight update: nonconverged source trace's last entry may precede that update. Choose the finite parameter-valid start with the smallest final source objective, first-seed tie; convergence is a reported property, not a hidden post-hoc selector. Record the objective's omitted data-only constant if applicable; compare only starts on the same input/transform. No selection by predictive score, latent labels, pairing, support checks or depth.

## Reviewed numerical adapter and failure handling

The author requires a caller-supplied weight QP. For r=4 use exact enumeration of all15 nonempty free sets of the lower-bounded simplex, solving each equality-constrained quadratic, checking primal and KKT tolerances, and selecting the minimal feasible objective. Respect .025 bounds and sum1. Validate independently against a separate convex solver on deterministic positive-definite fixtures and against KKT conditions. If no numerically valid candidate exists, raise and fail the start; do not enter the source's questionable None-return fallback, add jitter, or modify the lower bound. Specify numerical tolerances in source before its seal.

A thin compatibility shim may translate removed SciPy sym_pos=True to assume_a='pos', preserving the same positive-definite linear solve; all other arguments pass through. Test equivalence against independently solved systems and restore global modules/random state after each isolated fit. Do not alter mathematical updates, clipping, acceleration, stopping or source weights. Singular systems, nonfinite calculations and invalid parameters are explicit failures with retained diagnostics. Do not clip final inverse-transformed negative rates or renormalize invalid weights after fitting. To avoid cancellation at the exact zero boundary, the adapter may use the algebraically equivalent stable inverse raw_min+(M_standardized-M_lower)/gamma after verifying the unaccelerated source box bounds; compare it against get_soln within a recorded floating tolerance. This restores mathematically nonnegative source-box values without projecting an out-of-box solution. Any substantive negative or out-of-box value fails. Validate weights finite, positive and normalized, and rates finite/nonnegative; record bounds and source clipping as part of the algorithm.

Cap the three-start application at600 seconds, at most two subprocess applications and one BLAS thread each. Checkpoint each completed start, preserve interrupted/unattempted statuses and best completed candidate separately; no incomplete application becomes a successful full-grid fit. Fixed budgets are resource rules, not evidence of statistical failure. No post-outcome rescue initialization or changed numerical regularization.

## Predictions, diagnostics and evaluation

For each selected valid fit restore all original feature positions and specify product-Poisson mixture predictions. Compute exact count log scores with zero-rate conventions; negative infinity is a valid zero-support prediction, recorded as null plus status and impossible-row count while retaining all rows. It is not NaN, a reason to clip rates, or a reason to drop the selected moment fit. Training/calibration log scores are diagnostic only and never choose starts. Retain all parameter-valid source solutions, including source-nonconverged fits, with distinct flags.

Report weights, rate profiles, inactive features, weight-floor hits, start objective spreads, and label-aligned start disagreement (Hungarian on square-root full rate vectors) where valid. These starts diagnose local optimizer sensitivity, not bootstrap uncertainty. Report responsibility/depth and outside300-proxy associations descriptively, with original metadata retained for the null control. No biological state label or causal reference is used. There is no sample-bootstrap or cross-assay matching gate in C03, so it cannot establish common class proportions or claim recovered shared states; it tests the alternative estimator and count-model fit.

Seal all operative sources, upstream source/adapters, environment versions/BLAS, configs, nine training hashes and split/control rules before fitting. Seal all36 application outcomes and prediction parameters before test access. Evaluate unchanged public test counts in all arms: log score, zero frequencies, means, diagonal/offdiagonal covariance errors, using fixed source-fit scales, explicit zero-scale unscorable entries, full-panel and common30 views. Compare frozen C01 Poisson K=1 and K=4 predictions and, where available, C02 fixed binaryK2 predictions on the identical views. Methods differ in optimization, regularization and model restrictions; comparisons are descriptive and exploratory. Do not use paired S1, simulator state or DAG truth. Independent prefit mathematical/numerical review and full postfit metric replay are required.

Success means a faithful documented execution and honest assessment, not necessarily improved prediction. An estimator failure, moment misspecification or poor heldout score does not disprove the population theorem. A good score does not prove genericity, conditional independence, same-population weights, latent biology or cross-assay causal effects.

### Approved prefreeze source-compatibility clarifications

The pinned source calls the removed `numpy.int` alias. A temporary, restored
`numpy.int = int` alias reproduces the historical alias without changing the
solver's mathematical update; it is used only during an isolated fit. The
source's `norm_tau_weights()` starts at 1, so the actual order-k coefficient
is `J_active * (J_active-k)! / J_active!`. Use and record those exact weights.
This is the same positive factor for all orders, preserving minimizers and
updates; raw source objectives are compared only within the same application.

The generic theorem's preregistered dimension gate remains J_active ≥ 9.
With the pinned source's default one-row blocks, its own rank bound rejects
J_active=9 for r=4; that case is a separately recorded source-design
unsupported failure, not a revised theorem gate, smaller-rank run or block-size
rescue. J_active ≥ 10 is necessary for this source schedule.

The stored source objective omits the data-only constant
`C(V)=sum_{k=1}^4 k! tau_k sum_{i1<...<ik} (mean_t product_l V[i_l,t])^2`
for the standardized fit matrix V. Record this exact definition; its numeric
value is not computed at full scale. It is the same across starts for an
application and does not change their ranking. The source objective may be
negative and is not compared across applications. A small deterministic
fixture verifies full moment loss = source objective + C(V).

Moment-objective selection permits valid count laws that assign zero
probability to some source-fit rows. Keep all rows in score summaries; report
the impossible-row count and negative-infinity status. Responsibilities are
undefined for those rows, so report their number and compute depth associations
only on the supported subset, explicitly labeling this conditional sample.
Zero-total component rate profiles are reported as undefined rather than
normalized or clipped.
