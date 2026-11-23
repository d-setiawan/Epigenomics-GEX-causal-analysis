# Independent held-out readout review

The sealed 36-arm block completed, and the separate evaluator validated and scored every arm without changing the protocol, features, ranks, models, or thresholds. Nine deterministic evaluator tests passed. All 73 fit artifacts were verified before held-out arrays were opened, and their hashes were checked again after scoring. The full receipt is [data_and_evaluation_audit.json](data_and_evaluation_audit.json); the reproducible outputs are in [../evaluation_v1](../evaluation_v1).

## Main finding

There were **zero accepted pairwise source matches and zero all-assay matches in all 36 arms**. Every real cross-assay covariance prediction therefore used the empty-candidate zero default. Improvement over the zero baseline is exactly zero for all 36 real pair/arm comparisons, with all 900 feature pairs retained in each primary score. The paired-cell bootstrap improvement intervals are also exactly zero because the two predictions are identical; this is a mathematical consequence of the empty candidate set, not evidence that the true cross-assay covariance is zero.

With unit exposure, the real original arms selected ranks RNA 1, H3K27ac 2, and H3K4me3 1. With the leave-panel-out exposure proxy they selected zero in each assay. This sensitivity is consistent with substantial depth-associated structure, but does not prove that every removed component was technical. The proxy itself is endogenous, and the experiment did not observe true capture efficiencies.

## What improved within assays

For the real original unit-exposure arm, the source-fit-standardized full covariance errors were:

| Assay | Poisson corrected | Same-rank uncorrected | Diagonal training covariance |
|---|---:|---:|---:|
| RNA | 0.14397 | 0.17187 | 0.14473 |
| H3K27ac | 0.11288 | 0.18677 | 0.10537 |
| H3K4me3 | 0.10794 | 0.21077 | 0.10526 |

The corrected model beat the same-rank uncorrected model on full covariance error in all 36 original assay/condition/seed readouts. It beat the diagonal training baseline in only 10 of those 36. For off-diagonal errors, correction beat the matched-rank comparator in 22 readouts and tied it in 14; against the diagonal baseline it won 18, tied 14, and lost 4. These readouts reuse assays and seeds and are not independent biological replications.

Restoring a measurement-noise diagonal explains part of the full covariance gain, especially when selected rank is zero. The uncorrected low-rank comparator then predicts zero within-assay variance, whereas the corrected arm still includes Poisson variance. Accordingly, the full covariance advantage alone does not establish better biological sources. The separate off-diagonal readout and diagonal baseline are necessary context.

## Cross-assay and state interpretation

For the real original unit-exposure arm, the zero/default predictor has primary RMSE 0.06157 for RNA–H3K27ac, 0.06200 for RNA–H3K4me3, and 0.05187 for H3K27ac–H3K4me3. These are **training-standardized rate covariance errors**, not Pearson correlation errors. Different exposure sensitivities change the observed target scale and should not be ranked merely by comparing their numerical RMSEs.

Some simulated individual noisy projected scores correlate with a saved state coordinate at approximately 0.61. No score survived as a shared source candidate, so those within-assay associations do not demonstrate cross-assay state alignment or latent graph recovery. The saved state coordinates are also not the separate focal-feature graph's causal nodes.

Twelve real pair/arm secondary Pearson comparisons are unscorable because a rank-zero uncorrected model has zero predicted marginal variance. They remain explicitly undefined, while the fixed-scale primary comparisons retain all feature pairs. This avoids turning a missing variance model into a perfect correlation score.

## Validity boundary

The fitting source was inspected to confirm that paired references and simulated states are not used for fitting or decisions. The evaluator independently replayed the SHA split, original-depth exposure normalization, exposure-preserving control permutations, training means and noise terms, source-product cross predictions, all 595 paired joins, and all simulated public-discovery state joins. No private cohorts, new worlds, new biological measurements, or extra training assays were accessed.

These are repeatedly inspected development panels and one internally held-out scMTR-seq sublibrary, with inherited six-target source QC and promoter-only histone features. The paired real reference measures observed dependence rather than causal ground truth. The supported result is that this bounded Poisson correction helps some within-assay moment estimates but **has not recovered cross-assay structure under the frozen acceptance rules**.
