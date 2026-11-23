# Exploratory Bernoulli-label optimization follow-up

See the [combined results report](../../COUNT_TRANSPORT_MATCHED.md). This run
reuses the nine already-assessed exact-model worlds at 12,000 fitting cells per
assay. It changes the probability parameterization/domain, preserves the physical
initial models, and applies one global source-label canonicalization after fitting.

All 36 starts converged numerically. Calibration-selected full recovery is 3/3
for one source, 0/3 for two and 0/3 for three. Every higher-dimensional fit remains
suboptimal relative to the feasible true parameters. This is exploratory
estimator evidence, not a theorem counterexample or causal-direction result.

The [plan](PLAN.md), [protocol](protocol.json), [fit seal](fit_seal.json),
[evaluation](evaluation.json), [evaluation seal](evaluation_seal.json),
[scientific review](REVIEW.md) and [numerical audit](NUMERICAL_REVIEW.json)
preserve the full record. No new counts or biological measurements were used
in this follow-up, and the primary experiment remains unchanged.
