# Applying native Sturma MDCRL to existing counts

The native estimator found **zero shared sources in every frozen three-assay run** on the real scMTR cohort and the two cached simulation worlds. No latent graph was generated. The exploratory real-data pairwise follow-up also found zero sources for every assay pair, both under its native pair-specific KS threshold and the original global threshold. These are failures to recover shared structure in this small application, not evidence that shared biology is absent.

## Frozen primary application

The [application protocol](application_protocol.json) fixes raw counts and log1p counts per 10,000 allowed-feature depth, ICA seeds 0/1, native alpha 0.05/gamma 0.2, and the original numerical-rank rule. Each assay contributes 30 features selected deterministically by feature-ID hash after own-training detection/nonconstant QC. There is no input PCA, variance ranking, fixed latent cap, or tuning on paired outcomes. FastICA itself performs whitening; this is intrinsic to the method. [Data contract](reviews/data_contract_review.md).

There are 12 original fits and six independent-column permutation controls. The controls preserve each transformed feature marginal and destroy within-assay dependence; they are diagnostic controls, not biological FDR estimates. All 18 fits completed. None of the 12 original fits emitted an ICA convergence warning; the real normalized permutation control did, and it remains included. All estimated per-assay ranks were 30.

| Existing dataset | Training cells per assay | Evaluation cells per assay | Raw shared sources, seeds 0/1 | Normalized shared sources, seeds 0/1 | Independent-column controls |
|---|---|---:|---|---|---|
| scMTR-seq S1 | RNA 602; H3K27ac 615; H3K4me3 614 | 595 | 0 / 0 | 0 / 0 | 0 in both representations |
| Cached linear_chain_0, informative | 4,000 each | 2,000 | 0 / 0 | 0 / 0 | 0 in both representations |
| Cached constrained_collider_0, informative | 4,000 each | 2,000 | 0 / 0 | 0 / 0 | 0 in both representations |

The cached worlds were neither regenerated nor altered. Their background model contains nonlinear responses, Gaussian components and count measurement noise; the focal chain/collider graph is not the shared latent-state graph targeted by Sturma. No private locked simulation cohort was opened. With no matched components, source/state correlations are unavailable rather than evidence of zero biological state information.

## Heldout real-data joint moments

Only evaluation uses the existing 595 paired cells. For each pair, the model-implied cross-covariance is the product of the corresponding shared mixing blocks; full shared-plus-private mixing variance normalizes it into correlation. Since the shared blocks are empty, every predicted cross-correlation is zero. The heldout pair rows and state information never enter fitting, feature selection or parameter choice.

| Input representation | Assay pair | Valid feature pairs | Correlation RMSE | Zero-correlation baseline RMSE | Improvement |
|---|---|---:|---:|---:|---:|
| raw_counts | RNA / H3K27ac | 900 / 900 | 0.06515 | 0.06515 | 0.00000 |
| raw_counts | RNA / H3K4me3 | 900 / 900 | 0.06472 | 0.06472 | 0.00000 |
| raw_counts | H3K27ac / H3K4me3 | 900 / 900 | 0.05296 | 0.05296 | 0.00000 |
| log1p_cp10k | RNA / H3K27ac | 900 / 900 | 0.03749 | 0.03749 | 0.00000 |
| log1p_cp10k | RNA / H3K4me3 | 900 / 900 | 0.03663 | 0.03663 | 0.00000 |
| log1p_cp10k | H3K27ac / H3K4me3 | 900 / 900 | 0.04247 | 0.04247 | 0.00000 |

Both ICA seeds give the same zero predictions. The smaller normalized-data RMSE is **not an improvement in recovery**: normalization changes the reference correlations, and each result exactly equals its own zero baseline. Paired observational covariance is not intervention-validated causal truth.

## Exploratory pairwise follow-up

After the real three-assay runs yielded no common sources, the [separate pairwise protocol](pairwise_protocol.json) froze all three assay pairs for all six real representation/seed/control arms. It reuses the already fitted ICA mixing matrices and training means. No ICA was refitted and no feature or hyperparameter was selected from the follow-up results.

All 18 pairwise comparisons completed with zero shared sources and no graph. Their predicted correlations again exactly equal the zero baseline. Native two-domain matching changes adjusted KS alpha from approximately 9.26e-6 to 2.78e-5 because fewer comparisons are made. The additional count under the original global alpha is also zero in all 18 comparisons. Thus simply dropping the all-three consistency requirement did not rescue matching on this panel. [All pairwise records](pairwise_v1/summary.json).

## What failed and what remains open

The failure occurs at source matching, before native graph extraction. Changing the graph rank threshold alone cannot rescue these zero-source outcomes because that stage is never reached. This pilot does not isolate a single cause: the count observation model, latent-dimension estimation, source-distribution estimation, population differences, and selected features can all matter.

The sample and measurement contrasts are substantial. The published model has five sources per domain and noiseless linear measurements; native numerical rank recovers five. Our 30-column count matrices have rank 30. In the real training panel, 341/615 H3K27ac cells and 317/614 H3K4me3 cells are zero across all 30 selected features; the selected columns have only 10–25 and 10–29 detections respectively. Hash selection avoids variance-based selection but does not guarantee biologically informative or partial-pure-child measurements. The 30-feature result therefore cannot establish that the full dataset or all adaptations are unusable.

The next algorithmic question is a count-aware observation model and a justified way to estimate biological source dimension before matching distributions. A latent cap, different threshold, new matching statistic or a denser panel would be a separately declared adaptation, not an unchanged replication. No such change was tuned in this study. Additional data are not presumed available.

## Engineering and independent verification

A preflight rectangular-matrix test and independent review caught an omitted transpose in the new heldout encoder. The premature earlier application attempt failed its native-source parity check; its artifacts remain in [implementation_checks](implementation_checks/encoder_correction.json) and are excluded from scientific results. The correction changed the adapter algebra and failure-stage reporting, not the upstream method or scientific settings. The frozen protocol text retained the original shorthand; the receipt explicitly corrects the formula to `(test_X - train_mean) @ pinv(train_mixing).T`. All three focused algebra tests pass, and every final fit verifies training-source parity.

Independent reviewers replayed the selected raw matrices and evaluator joins from original sources, then reconstructed heldout encodings and joint-moment metrics from saved matrices without fitting the learner. See the [source-data audit](reviews/data_extraction_audit.json), [application readout audit](reviews/application_readout_audit.json), [code review](reviews/application_code_review.md) and [data/readout interpretation](reviews/application_data_and_readout_review.md). The [primary summary](application_v1/summary.json) retains all warnings, statuses, source hashes and metrics. Two optimizer seeds measure computational sensitivity, not biological replication or statistical reliability.
