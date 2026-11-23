# C01 count-state validation

Status: training, sealed evaluation and independent numerical/result review complete. This is a newly executed count-model application on the existing inputs, not a replay of historical CSVs and not an exact theorem-generating simulation.

## Frozen training outcome

All 24 dataset/panel/kernel/control arms completed. No arm produced an accepted cross-assay state assignment: 21 selected K>1 arms abstained, and three real-data column-null arms selected the K=1 baseline. These outcomes do not falsify Poisson mixing-law injectivity. They show that the prescribed finite mixture estimator and finite-sample correspondence gates did not establish the sufficient shared-state model on these inputs.

All 16 cached-simulator arms selected K=4, the top of the prespecified grid. That boundary selection is a warning against treating K as an estimated biological state count; no post-outcome grid expansion is included in C01. In all eight S1 arms both histone assays independently preferred K=1, while RNA preferred K=4 in six arms and K=2 in the two 300-feature column-null arms. A common-K sum of calibration scores can therefore be driven by a different assay than the most weakly resolved one.

The independent reviewer reproduced 839 converged primary restarts out of 864; 25 reached the iteration limit. Of 4,032 bootstrap candidates, 3,883 converged, 101 reached the limit and 48 encountered dead components. Nine of 2,016 bootstrap replicates had no converged candidate. Local optimizer variability is material: 155 of 282 model cells with multiple converged restarts had average log-likelihood spreads exceeding 1e-5, with maximum spread about 3.803. The frozen protocol retains these failures and selects likelihood before stability gating.

Only 17 of 63 selected-K assay blocks passed the stability gate. Failures overlap: 15 had fewer than 26 valid bootstraps, 11 failed point membership, and among calculated diagnostics 31 failed displacement and 16 failed weight uncertainty. Stability and common-weight compatibility are separate: in the original 300-feature Poisson linear-chain arm all three assays passed stability, yet no weight-compatible joint assignment existed. The largest H3K27ac interval was approximately [0.557,0.628], above every RNA interval (largest upper endpoint approximately 0.462).

Observed depth can drive these effective-intensity components. For example, the original S1 300-feature Poisson RNA fit has a responsibility/depth correlation near +0.779. This descriptive association does not distinguish cell size from technical capture or establish a biological state. The model has unit exposure, and original panel selection also used the full training pool before our fit/calibration partition.

## Interpretation boundary

With zero compatible assignments, C01 has no conditional cross-assay covariance candidate to validate; its empty candidate set is abstention, not a discovered independent joint. The evaluator still assesses all frozen within-assay predictive models, K=1 controls, original test counts for both training controls, and the separate zero-covariance baseline. Its output cannot retrospectively select thresholds, state numbers, source panels or stopping rules. Assay-local QC populations differ from the paired evaluation population, and even an accepted common-state match would require an extra across-assay conditional-independence assumption for joint completion.

## Heldout counts and independent review

The independent reviewer replayed all 216 within-assay metric records, 384 simulator state correlations and 36 paired zero-baseline views, as well as frozen source/input/result bindings. This is independent numerical verification; only representative optimizer and bootstrap trajectories were rerun, not every fit. See [independent review](../../reviews/c01_independent_review/REVIEW.md).

For the full fitted-panel views, the selected mixture beat K=1 in test log score in 6/12 S1 original assay comparisons and lost in 6/12. S1 null fits had one improvement, nine ties and two losses. All 24 cached-simulator original assay comparisons improved log score and off-diagonal covariance RMSE; null fits improved log score in 23/24 but off-diagonal covariance RMSE in only 9/24. These correlated descriptive comparisons distinguish marginal predictive improvement from covariance fit; they are not significance tests or proof of latent state recovery.

No joint candidate exists, so no cross-assay covariance recovery or causal graph was validated. A zero-covariance baseline can be scored on the paired reference, but scoring it does not convert abstention into an independence result. The count experiment validated the implementation and tested a particular sufficient-model route, with mostly negative alignment applicability and mixed biological predictive results. The mathematical population theorem remains conditional on its stronger assumptions.
