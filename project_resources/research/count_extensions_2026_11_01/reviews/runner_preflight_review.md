# Extension runner preflight review

Approved for the frozen bounded sweep; no launch-blocking issue found. The exact runner, protocol, and estimator hashes are bound in [runner_preflight_review.json](runner_preflight_review.json).

The independent runner/estimator test rerun passed 16 tests. Separate fixed mocked arms exercised zero-source and unassessed-CF exports, safe NPZ/JSON serialization, explicit failures, and hashed partial artifacts. The root separately reports all29 preflight tests passed; those are not presented as independently rerun here.

The runner keeps source fitting and fingerprint calibration separate and accesses no heldout reference data. Kernel sensitivities preserve observed mean-rate units. Fixed exposure nuisance parameters and all300-column controls are shared across widths. All bootstrap randomness is local and reproducible; the staged CF check replays calibration indices with saved directions instead of refitting sources. Ambiguous moment candidates remain eligible for the stricter check, and unassessed sources retain that status.

The constant-column preconditioner is used consistently by rank selection and loading alignment while preserving observed zero variances for evaluation. PGF point and bootstrap effective-cell gates are auditable from saved values. Rejected raw CF values and all numerical gate diagnostics are retained. Two fork workers share immutable prepared arrays; the main allocations are feature covariance matrices and low-dimensional tensors.

The independent fit-decision replay is prepared and will require the final seal before reading outputs. This preflight approval concerns execution and auditability; it does not certify the count model, complete source-law equality, or causal identification.
