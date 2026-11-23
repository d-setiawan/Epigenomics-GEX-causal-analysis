# C02: faithful observed-count DCRL application

Astra approval of scientific design, 2026-11-15, before C02 implementation or fitting. Exact wrapper/source/dependency review and protocol seal are still required before execution. Basis: newly retrieved DCRL Theorem 1 and official observed-data code, plus C01 training diagnostics. This is a distinct count-native sparse latent measurement model, not a relaxation of C01 acceptance gates. C01 evaluation summaries are available; C02 is therefore explicitly a sequential exploratory application, not an untouched confirmatory test.

## Target and theorem boundary

Use Zhang, Wang and Gu (2026), Discrete Causal Representation Learning, arXiv:2603.25017, official source commit 2f768e048cb49b2af674fe31f76e55374362c00e. The theorem identifies measurement structure and binary latent joint distribution up to coordinate permutation under genericity, positive configuration probabilities, conditional item independence, monotonic response, sparse measurement design and latent Markov/faithfulness assumptions. The latent DAG conclusion is only Markov equivalence. We assess count prediction and the estimated measurement design within each assay; no latent DAG or cross-assay joint will be inferred or evaluated. The specified current inputs do not verify the true design or faithful latent mechanism.

Use the official fit_observed observed-data API, not a simulation constructor or truth-evaluating estimate method. No generators, new worlds, biological acquisition, pairing, state labels, causal labels or day labels enter fitting. Source-only generator definitions may be imported only if required by upstream imports; no generation function is called.

## Complete existing-input grid

The same nine immutable train matrices as C01: three assays in scmtr_seq_s1, linear_chain_0 and constrained_collider_0. Use both first30 and full300 fixed feature panels, retaining their stored order. Use the same SHA256 opaque-ID 75/25 source-fit/calibration split and the same independent-column null (seed1000 by dataset/assay, permuted separately within fit and calibration, all300 before prefix slicing). No test count is permuted. This gives36 assay/panel/control applications, each with starts0,1,2. Binary latent dimension is fixed K=2 (four configurations); no K or penalty search. K=1 would be outside the cited theorem and is supplied instead by C01's already frozen Poisson baseline. C01 Poisson K=4 is a separate unrestricted finite-mixture comparator; no C02 selection uses its heldout outcomes.

## Faithful code settings and resource rule

Use Poisson PSAEM, known_Q=None, standard backend and column_sum orientation. Preserve author defaults:20 outer iterations, tol=.05, C=1, epsilon1e-50, kappa1e-200, penalty N^(1/4), tau2N^(-.275), gain.5/(t+10), inner SLSQP cap20, coefficient_bound_scale1. Intercepts are bounded[0,2] and slopes[0,3]. These are code defaults, not a verified reproduction of every paper experiment. Record exact finalized config, all starts, initialization and optimizer diagnostics; author training penalized likelihood chooses the retained start. Freeze NumPy seeds separately and use isolated processes, at most two workers and one BLAS thread each. Cap each complete three-start application at600 seconds, recording a resource-limited failure rather than modifying the model. Report elapsed time and all failures; no optimizer, bound or seed rescue selected from outcomes.

Do not normalize counts, add depth offsets, clip invalid fitted rates, change the penalty or widen bounds. A zero rate is mathematically allowed and must use exact Poisson conventions. Nonfinite probabilities/rates/objectives and out-of-bound parameters are explicit invalid outputs. Finite results with unsuccessful optimizer/stopping diagnostics remain reportable as such; they do not become certified converged estimates. If upstream code has a blocking execution bug, stop and document a separately versioned correction before any rerun.

## Predictions and diagnostics

Predict full count vectors using the fitted SparseLatentLaw and additive rates beta0+sum beta_k z_k, enumerating the four binary states with xlogy/gammaln/logsumexp or the verified official exact scorer. Retain absent-state zero masses and any fallback/uniform-law diagnostic; never claim fitted positivity proves true positivity. Store fitted parameters and all state masses, start objectives, source/calibration likelihoods and stopping diagnostics before evaluator access. Compare seeds descriptively, with no post-hoc selection based on support or stability.

Check the fitted Q=(slope !=0) design exactly: existence of two disjoint K-row blocks with unit diagonals (off-diagonals arbitrary), plus unused rows collectively nonzero in every column, and pairwise column noncontainment. For K=2 implement an exact finite search or equivalent matching/count certificate and test it with deterministic matrices. A pass concerns only estimated Q; it is not a certificate of the true measurement model or causal identifiability. Report positive fitted state count, rate-bound saturation, fraction of fit-feature means exceeding the maximum attainable mean8, and fit posterior E[Z|Y] associations with permitted library depth/outside300 proxy. These cannot tune, exclude or relabel fits. Fixed row order and no row-order sensitivity study are explicit limitations.

## Seal and evaluation

Seal source, official source hashes, isolated dependencies, finalized configuration, training inputs and split/control rules before fitting. Seal all36 fit outputs (including failures), parameters and predictions before evaluation. Use the unchanged original public test counts for predictive log score, zero frequencies, means, diagonal and off-diagonal covariance errors, with fixed source-fit count scales and zero-scale entries explicitly unscorable. Report full panel and common30 views. Report the same quantities for frozen C01 Poisson K=1 and K=4 fits when available; the methods have different regularization/optimization and these are descriptive comparisons, not matched complexity or significance tests.

No paired biological reference, state labels or causal truth is needed for C02. C01 already reports separate evaluator-only state/paired endpoints; do not expand those to manufacture a new causal accuracy claim. All model-choice rules precede C02 test access, but the overall research sequence is exploratory. Independent source review before execution and full numerical/result review after execution are required. An unfavourable result or design failure is an admissible completed application; it does not refute the population theorem.

### Execution and timeout clarification

The wrapper calls the official observed-data API separately for each of the
three prespecified spectral starts inside one 600-second application budget,
checkpointing every completed start. It selects the greatest attained author
penalized objective, using the earliest start on exact ties, reproducing the
upstream selector. If the hard application limit interrupts a start, its
marker and the completed checkpoints are preserved; that start is reported
as time-limited, later starts as unattempted, and the application is incomplete.
No partial application is represented as a successful three-start result.

### Prefit v2: legitimate zero-probability observations

The prefit v1 files and hashes are preserved in
`experiments/C02/prefit_revisions/v1/`; no scientific fit occurred. A valid
fitted law may assign probability zero to a calibration or test count vector.
Retain every row and exact zero rates: encode a negative-infinite mean log
score as JSON null, status `negative_infinity`, and the number of zero-
probability rows. Such a calibration result is a valid predictive failure,
not a reason to discard the fitted law or change the author-selected start.
NaN or positive infinity is an invalid numerical result; no clipping or
row filtering is permitted. The training objective still must be finite.
