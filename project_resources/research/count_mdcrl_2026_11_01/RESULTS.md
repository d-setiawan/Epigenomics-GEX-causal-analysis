# Count-aware source pilot: results

The Poisson-corrected estimator improved a narrower task—predicting within-assay covariance—against its same-rank uncorrected ablation. It did **not** produce accepted cross-assay sources, improve cross-assay prediction over the zero-covariance default, or reach latent graph extraction. This does not establish that all count-aware models must fail.

## Executed comparison

The [protocol](protocol.json) was fixed before fitting. All inputs came from the [prior frozen panel manifest](../sturma_pilot_2026_11_01/data_pilot_v1/manifest.json). Each assay retained exactly the same 30 features. A deterministic internal 75/25 training split separated source/rank fitting from source-fingerprint calibration. The original public test/reference cohorts were used only after all fits were sealed.

| Item | Executed |
|---|---:|
| Existing datasets | scMTR-seq S1, `linear_chain_0`, `constrained_collider_0` |
| Assays | RNA, H3K27ac, H3K4me3 |
| Original-data arms | 24 |
| Feature-permutation controls | 12 |
| Completed arms | 36/36 |
| Top-level fit failures / captured warnings | 0 / 0 |
| Accepted cross-assay source pairs | 0 in every arm |
| Accepted three-assay sources | 0 in every arm |
| Native graph extractions attempted | 0 |
| Saved fit computation | 303.30 seconds, one CPU thread |

Each original-data condition used source seeds 0 and 1. Controls used seed 0. There were 49 training permutation nulls per dimension screen and 32 cell bootstraps for each nonempty fitted assay model. Four bootstrap fits abstained because a resample made an observed feature constant; the records retain all four failures. Other invalid fingerprint replicates are retained as missing and counted by the uncertainty gates. Nothing was retried with looser settings.

This is a development comparison on previously inspected panels, not a new locked validation block. Seeds and reused worlds are not independent biological replications. No new worlds, datasets, modalities or feature panels were created.

## Dimension and source matching

The dimension gate tests coherent covariance above a null that preserves feature marginals. Its output is not the total number of biological or private factors. The same selected dimension was supplied to both estimators.

| Dataset | Exposure treatment | RNA | H3K27ac | H3K4me3 |
|---|---|---:|---:|---:|
| scMTR-seq S1 | Unit exposure | 1 | 2 | 1 |
| scMTR-seq S1 | Outside-panel depth proxy | 0 | 0 | 0 |
| Linear-chain cached world | Unit exposure | 2 | 2 | 3 |
| Linear-chain cached world | Outside-panel depth proxy | 0 | 0 | 1 |
| Collider cached world | Unit exposure | 2 | 2 | 3 |
| Collider cached world | Outside-panel depth proxy | 0 | 0 | 1 |

All permutation controls selected zero dimensions in all assays. No dimension cap bound. The previous native application returned numerical covariance rank 30 in every assay; these are different dimension criteria, so their difference alone is not evidence of recovered biology.

Twenty of the 36 arms had at least one within-assay source pass the declared source gates. **No source pair passed the practical equivalence, uncertainty and uniqueness checks.** Having an eligible individual source is therefore different from finding a supported shared source.

The real histone panels were especially weak for source calibration. In the reserved calibration rows, each H3K27ac feature was detected in only 1–8 cells and each H3K4me3 feature in 1–13 cells. More than half the calibration rows contained no counts across the entire selected histone panel. With unit exposure, corrected histone fingerprints and/or loadings failed the stability criteria; the uncorrected counterparts also failed source gates. The two simulations sometimes supplied eligible individual sources in more than one assay, but their estimated fingerprints did not agree precisely enough to pass the equivalence screen.

Depth adjustment removed almost all coherent dimensions. This is sensitivity to the chosen count-derived denominator. It does not establish that removed variation was purely technical: the library total can also carry biological variation, and dividing by it changes the model.

## Within-assay covariance improved, with a stronger baseline caveat

The primary covariance scale is fixed by source-fit observed-rate standard deviations. Lower RMSE is better. For real S1 with unit exposure:

| Assay | Ordinary moments, same rank | Poisson-corrected | Diagonal training covariance |
|---|---:|---:|---:|
| RNA | 0.17187 | **0.14397** | 0.14473 |
| H3K27ac | 0.18677 | 0.11288 | **0.10537** |
| H3K4me3 | 0.21077 | 0.10794 | **0.10526** |

The corrected model adds its estimated Poisson measurement variance back when predicting observed covariance. The uncorrected arm already fits observed variation, so no extra measurement term is added to it. The diagonal baseline retains each training feature's observed variance and predicts zero off-diagonal covariance.

Across the 18 distinct dataset × exposure × assay conditions, correction reduced full covariance error relative to the matched-rank ablation in all 18. It reduced off-diagonal error in 11 conditions and tied in the seven zero-rank conditions. Against the stronger diagonal baseline, it improved full covariance error in only 5/18 and off-diagonal error in 9/18.

For real unit-exposure histones, even corrected off-diagonal errors remained worse than the diagonal baseline: H3K27ac 0.06191 versus 0.04840; H3K4me3 0.06128 versus 0.05118. The gain over the ablation therefore does not establish a useful recovered histone dependence model.

The covariance product $BB^\mathsf T$ is invariant to the source rotation. Accordingly, these covariance results are identical across the two tensor seeds up to floating-point precision. They assess covariance estimation, not identification of individual independent sources.

## Cross-assay moment recovery did not improve

No source pair was accepted, so every predicted cross-assay covariance block is the empty-candidate default of zero. This is an abstention policy, not an inference of biological independence.

For real S1, the heldout training-standardized cross-covariance RMSEs were:

| Assay pair | Unit exposure | Outside-panel depth proxy |
|---|---:|---:|
| RNA–H3K27ac | 0.06157 | 0.04473 |
| RNA–H3K4me3 | 0.06200 | 0.03823 |
| H3K27ac–H3K4me3 | 0.05187 | 0.06342 |

Both estimators exactly equaled the zero predictor on all 900 feature pairs per pairwise comparison. Their improvement over zero and its paired-cell bootstrap interval were identically zero. The exposure columns concern different transformed references and scales; comparing their absolute errors does not show an improvement in recovery.

Secondary Pearson readouts include the model's observation variance. The zero-rank uncorrected models have undefined model correlation denominators, which remain explicitly unscorable; they are not reported as perfect predictions. This does not affect the primary fixed-scale covariance scores.

## Some within-assay state information survived

On the cached simulations, individual noisy source projections retained some association with the evaluator-only state. The largest absolute state correlation across the prespecified exposures, seeds, assays and retained sources was:

| Cached world | Ordinary moments, same rank | Poisson-corrected |
|---|---:|---:|
| Linear chain | 0.59082 | 0.59342 |
| Collider | 0.60630 | 0.60716 |

Both maxima came from the H3K4me3 depth-proxy arm. These are descriptive maxima, with selection over several coordinates; no significance or causal claim is attached. Full state-by-source matrices, including weaker associations, are saved with the [evaluation](evaluation_v1/summary.json).

No matched source was available for a cross-assay state or graph claim. The cached worlds' two background-state coordinates do not constitute the separate focal chain/collider graph. We did not score anonymous latent matrices against that unrelated truth.

## What this establishes and what remains open

The bounded hypothesis that Poisson noise correction alone would enable reliable source matching is **not supported on these panels**. The narrower benefit to within-assay moment estimation is supported relative to the matched-rank ablation, with the baseline qualifications above.

The failure has several visible locations: sparse histone calibration, unstable sources, lack of sufficiently precise distributional agreement, sensitivity to depth treatment, and possible mismatch between the affine Poisson source model and nonlinear/overdispersed data. This experiment does not identify one of them as the sole cause.

There is also no population-level impossibility conclusion from count noise alone. With a known independent Poisson observation kernel and fixed positive exposure, a domain's full count law can determine its nonnegative intensity law through the probability-generating/Laplace transform relationship. The [independent theory review](reviews/theory_implementation_review.md) gives the argument and explains the further conditions needed to compose it with Sturma's result. Our finite moments and finite fingerprint grid do not implement that population inversion or certify those conditions.

A subsequent model comparison should address measurement overdispersion and how depth relates to biological state before interpreting a larger latent dimension as progress. It could also test a larger prespecified panel drawn from the existing allowed features to assess sparse-feature power. Neither a negative-binomial source model nor that panel expansion was fitted here; frozen thresholds were not relaxed to force cross-assay edges.

## Verification and artifacts

- [Preflight](preflight.json): nine estimator/runner tests passed before the first scientific fit.
- [Independent algebra](reviews/count_algebra_audit.json): factorial moments, tensor units, the deconvolution identity, empirical standard error and rank-test boundary.
- [Independent fit replay](reviews/audit_count_fits.json): all 36 dimension, uncertainty, matching, sign and covariance decisions reproduced from sealed artifacts.
- [Evaluator validation](reviews/evaluator_validation.json): heldout joins, training-only scale, joint-cell bootstrap, zero-rank behavior and noise accounting.
- [Independent heldout metric replay](reviews/count_metric_audit.json) and [final interpretation review](reviews/result_interpretation_review.md).
- [Fit seal](runs_v1/provenance.json), [evaluation seal](evaluation_v1/provenance.json), and [final validation](validation.json).
