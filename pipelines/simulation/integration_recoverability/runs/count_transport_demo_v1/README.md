# Count transport on two cached integration worlds

The November 15 demonstration fitted two count-source model specifications on existing
simulation worlds whose mechanisms differ from the fitted model. The learner saw unpaired
counts. The evaluator opened paired cells after fit selection. Across the 96 fits, 32 starts at
the selected dimensions were assessed. Both specifications beat an independent-Poisson baseline
on within-assay calibration likelihood. Their paired count predictions were worse than using the
target assay marginal and their covariance estimates had larger errors than predicting zero
covariance. This test illustrates how well an estimator can fit the separate assays while
missing their joint behavior. [Results and comparator](../../COUNT_TRANSPORT_DEMONSTRATION.md)

The plan, selection and evaluation receipts, and independent reviews travel with the Git
checkout. The per-fit payloads and full evaluation belong to the original run cache. The later
exact-model experiment supplied the source dimension and tested the model under its own
assumptions. [Plan](PLAN.md) · [Frozen protocol](protocol.json) · [Fit seal](fit_seal.json) ·
[Evaluation seal](evaluation_seal.json) · [Numerical review](NUMERICAL_REVIEW.json) ·
[Scientific review](REVIEW.md) ·
[Exact-model follow-up](../count_transport_matched_v1/README.md)
