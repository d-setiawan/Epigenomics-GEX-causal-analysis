# Independent count-application code review

Reviewer: `causal_reliability_pilot`  
Date: 2026-11-01  
Reviewed runner: [apply_native.py](../apply_native.py), SHA256 `62df9501d73619653604dbe682b8c9e9a290fc8d73b752deabc1d316c6de81ed`  
Protocol SHA256: `4991e33d3866b001c8f1d1ac9f7a9ec92bc7fd5dfdc1dcba56d335c1d7550fff`

The corrected implementation is consistent with the declared descriptive count pilot. No remaining blocking algebra or learner/evaluator boundary issue was found in this static review. This is not validation of the Sturma identification assumptions for the count data.

An initial blocking error omitted the transpose in the heldout encoder. With observations `X = sources @ mixing.T + mean`, the correct operation is `(X - mean) @ pinv(mixing).T`. The rectangular source-recovery test and independent review exposed the error. The corrected implementation now passes all three focused tests, independently rerun here. The early attempted application, its failed encoder-parity checks, and the protocol-text erratum are preserved in the [correction receipt](../implementation_checks/encoder_correction.json). Native algorithm settings were not changed.

The reviewed code:

- Uses only training matrices for ICA, source matching, and native graph extraction. Public heldout arrays are loaded for schema checks but do not enter fitting; evaluator state/pair references are opened during scoring.
- Encodes heldout observations with training column means and training mixing matrices, and checks reconstruction of the native standardized training sources.
- Forms cross-assay covariance from matched shared mixing columns with the recorded signs. The correlation denominator uses all within-assay source columns, including private variation. This distinction is covered by a focused test.
- Applies independent column permutations after preprocessing, preserving each transformed feature's marginal values. It fixes permutation and ICA seeds separately.
- Leaves undefined correlations unavailable and reports valid pair counts. The real-data comparison concerns moments of observed counts; it is not a causal graph score.
- Reports the anonymous native coefficient matrix separately from matched disturbance coordinates. The instrumented subclass records graph-stage outputs without changing native calculations.
- Preserves every assigned seed/control and captures native graph failures instead of changing the graph heuristic. Two ICA seeds assess computational sensitivity rather than sampling reliability.

The [independent input-consumer audit](data_consumer_audit.json) verified all 18 public data exports and three evaluator exports against the frozen data manifest. In particular, the real paired indices reconstruct the saved 595-by-90 reference exactly for both transforms, and simulated state rows match heldout IDs. This is a consumer-contract check; the separate preparation audit owns replay against the original source files and canonical split identities.

The runner records actual input hashes, but does not itself reject a public file whose hash differs from the frozen manifest. The independent audit found no such discrepancy. Future reuse should preserve the frozen inputs or add an explicit preflight assertion before starting a new run.

No fitted model was rerun by this reviewer. The mathematical checks concern the new encoding and moment calculations; they do not establish that matched factors correspond to biological causes.
