# Review of the frozen count application

Reviewed on 2026-11-01 by `scmtr_data_pilot`. Scope: data boundaries, source/target joins, heldout encoding, moment readouts and interpretation. No model was fitted by this review.

The source reviewed is `apply_native.py`, SHA256 `62df9501d73619653604dbe682b8c9e9a290fc8d73b752deabc1d316c6de81ed`. The complete 18-record `application_v1/summary.json` has SHA256 `ac6577a5abe0bd4cdda140203dcbcc3e2e546e2612dcb126c22150a5829c32e3`.

**Scoped approval:** the input separation and reported moment calculations are valid for the declared exploratory application. No blocking implementation error was found in this review. The native algorithm did not recover a shared representation in this frozen 30-feature setup, so these runs provide no positive causal-recovery result.

The fitting path passes only the three training count matrices to the estimator. Test values are loaded for later encoding, but do not enter source estimation, matching, graph extraction, feature selection or centering. Pair and state reference files are opened only in evaluation. The independent-column control permutes training columns after preprocessing and preserves each feature's marginal values. Its explicitly stated purpose is a marginal-shape diagnostic; it is not a calibrated null experiment for biological edges.

The 30-feature extraction was verified against original H5AD counts by a separate replay implementation, and another agent independently checked the consumer schemas and evaluator joins. See [source replay](data_extraction_audit.json), [consumer audit](data_consumer_audit.json), and the [data contract](data_contract_review.md). The canonical training assignments remain disjoint, and real same-cell identities remain evaluator-only.

The independent [readout replay](audit_application_readouts.py) does not import the fitting runner. It reconstructed heldout sources using least squares, reconstructed the full model covariance from `joint_mixing @ joint_mixing.T`, and compared its cross blocks and marginal variances with the saved predictions. The real reference correlations were recomputed from the independently verified paired matrices using `numpy.corrcoef`; RMSE and zero-baseline differences were replayed. Simulation state IDs, empty source matrices and undefined state-recovery summaries were also checked. All 18 readouts passed. [Audit output](application_readout_audit.json), SHA256 `b83695fd0f8f43a30b7f8895540c5fe3d81f1b1e922b28e4aa198b5be3d18dca`.

All 18 runs estimated within-domain rank 30 and matched zero sources across all three assays. Graph status is `no_shared_sources`, not an empty recovered causal truth graph. The 12 original-input fits had no captured convergence warning. The normalized scMTR independent-column control produced one `FastICA did not converge` warning; it is retained and should not be described as converged.

For the real 595-cell paired reference, every pair has 900 valid feature comparisons. Both seeds and the column-permutation control produce zero predicted cross-assay correlation. The following errors equal their respective zero-correlation baselines exactly:

| Assay pair | Raw-count correlation RMSE | Normalized-count correlation RMSE | Improvement over matching zero baseline |
|---|---:|---:|---:|
| RNA–H3K27ac | 0.065149 | 0.037493 | 0 |
| RNA–H3K4me3 | 0.064725 | 0.036626 | 0 |
| H3K27ac–H3K4me3 | 0.052964 | 0.042467 | 0 |

The normalization branches have different reference correlations. Their absolute errors must not be compared as evidence that the algorithm improved: normalization reduces the reference's departure from zero, while the model predicts zero in both branches. No confidence interval or reliability rate is implied by these point summaries.

The simulation outputs also contain zero shared sources. Source-by-state arrays therefore have shape 0 × 2, and maximum source correlation per state is absent rather than set to a success-valued zero. This is the correct empty-output convention. No selected-background joint count truth was read from private cohorts, and no latent-DAG or focal-feature edge score was substituted for the descriptive state target.

The data limitations remain material. Most real histone rows are zero across the selected 30 features. The simulated background observations have nonlinear components and independent measurement noise; one state coordinate is Gaussian and the state graph has no edge. These are confirmed mismatches with the intended theorem setting, not a proof of the precise mechanism responsible for every failed match. The result means that this pinned native procedure found no consistent shared-source triplets in this frozen application. It does not mean that shared biology is absent, that every feature subset would behave identically, or that the paper's restricted positive theorem is false.

No thresholds, source counts, features, seeds, or native heuristics were changed in response to these readouts. Further adaptations should be recorded as a new algorithm/protocol and should retain this result as the original native application.
