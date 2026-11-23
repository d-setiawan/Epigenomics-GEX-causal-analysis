# Independent interpretation of the count-source pilot

Reviewed on 2026-11-01 after all 36 fits and their heldout evaluation were sealed. No thresholds, features, exposure rules, source fits, or evaluation settings were changed. The independent replay imports neither learner nor evaluator. It reconstructs all 108 within-assay metric records, 36 real assay-pair records, 7,200 saved paired-bootstrap outcomes, and 72 simulated state-correlation matrices from the frozen public test/reference arrays. All checks pass; the maximum numerical difference is $1.42\times10^{-14}$. See [count_metric_audit.json](count_metric_audit.json) and [replay_count_metrics.py](replay_count_metrics.py).

**The results support improved within-assay covariance estimation relative to the specified low-rank ablation. They do not support successful cross-assay source matching or causal recovery.** These are different parts of the original hypothesis and should receive separate conclusions.

## Cross-assay result

All 36 arms completed, and every arm returned zero pair matches, zero all-assay matches, and no latent graph extraction. Every real primary covariance improvement over the zero predictor is exactly zero, including every paired-bootstrap replicate. This happens because the method accepted no shared sources and consequently used its empty-candidate zero covariance. It is not evidence that the true cross-assay relationships are absent.

Original-data ranks were the same for both methods and both source seeds, as designed. In assay order RNA, H3K27ac, H3K4me3:

| Panel | Unit exposure | Leave-panel-out proxy |
|---|---:|---:|
| scMTR-seq S1 | 1, 2, 1 | 0, 0, 0 |
| Cached linear-chain world | 2, 2, 3 | 0, 0, 1 |
| Cached constrained-collider world | 2, 2, 3 | 0, 0, 1 |

Every independent-column control selected rank zero in all assays. This is a useful bounded diagnostic of the coherent-dependence screen, not a false-discovery estimate or a proof that the selected original modes are biological. The large exposure sensitivity also cannot distinguish technical depth from biological variation removed by the proxy denominator.

On real unit-exposure data, only the RNA source passed the source gates in either method. The corrected histone estimates failed bootstrap-validity and/or loading-stability gates. For corrected seed 0, the two H3K27ac sources had only 22 valid bootstrap fingerprints each and loading 10th-percentile cosines 0.068 and 0.333; the H3K4me3 source had 23 valid fingerprints and cosine 0.470. The frozen requirements were at least 26 valid fingerprints and a loading 10th percentile at least 0.7. The ordinary-moment histone sources also failed loading stability. These were explicit abstentions from source matching, not downstream graph failures.

The simulated unit-exposure arms had some stable sources, but their finite fingerprints did not meet cross-assay equivalence. Even the smallest accepted-band score among source-gate-passing comparisons was above 14 in the corrected collider arms and above 40 in the corrected chain arms, against a frozen acceptance boundary of one. This is not a near-threshold success hidden by one rounding choice. No relaxation is warranted from this pilot.

## Within-assay improvement and its limits

There are 18 distinct original-data covariance comparison settings: three panels, two exposure choices, and three assays. The two source seeds do not provide independent covariance replications: $BB^\top$ is invariant to the orthogonal tensor rotation, and their covariance scores are identical to numerical precision. Counting each setting once avoids overstating replication.

| Comparison across those 18 settings | Corrected better | Tied | Corrected worse |
|---|---:|---:|---:|
| Full covariance versus same-rank ordinary moments | 18 | 0 | 0 |
| Off-diagonal covariance versus same-rank ordinary moments | 11 | 7 | 0 |
| Full covariance versus diagonal training covariance | 5 | 0 | 13 |
| Off-diagonal covariance versus zero off-diagonal baseline | 9 | 7 | 2 |

These are descriptive point comparisons within this fixed development study, not 18 independent biological tests. Each comparison uses the same training-standardized metric within a setting. Error magnitudes from different exposure definitions should not be compared as if they had one common reference scale.

For the primary real unit-exposure setting, the full covariance RMSEs are:

| Assay | Poisson corrected | Same-rank ordinary | Diagonal training baseline |
|---|---:|---:|---:|
| RNA | 0.14397 | 0.17187 | 0.14473 |
| H3K27ac | 0.11288 | 0.18677 | 0.10537 |
| H3K4me3 | 0.10794 | 0.21077 | 0.10526 |

Thus the real histone improvements over the ordinary low-rank arm do not beat the simpler diagonal baseline. RNA has a small full-matrix gain and a clearer off-diagonal gain. In rank-zero arms, adding the modeled Poisson diagonal can improve full-matrix error without learning any latent source. Off-diagonal comparisons show that some gains do extend beyond merely restoring a diagonal, but a better covariance matrix still does not validate ICA source independence or causal structure.

The simulated H3K4me3 scores retain moderate associations with existing state coordinates: corrected maxima reach approximately 0.54–0.57 at unit exposure and 0.59–0.61 for the proxy-exposure source. Other assay/state associations are weaker or mixed. These are noisy projection correlations, with selection over available factors; they do not map an anonymous factor to a causal node. Pair-matched and all-assay-matched state readouts remain empty.

## Scope of the conclusion

The implementation has moved beyond ordinary ICA on counts: it explicitly removes Poisson factorial-cumulant terms, estimates a smaller dimension, and evaluates latent-intensity distribution summaries with uncertainty. The mathematical checks validate those calculations under their model. The heldout result supports the noise-aware covariance part of that change on these panels. The evidence does not establish that this affine independent-source model describes their biology or that its sources can be shared across assays.

The [population proposition](theory_implementation_review.md#count-noise-does-not-imply-universal-nonidentifiability) remains relevant: a known independent Poisson kernel can preserve identification of each complete intensity law, and further Sturma assumptions can identify shared latent structure. These finite-moment results neither verify those assumptions nor refute that restricted possibility. Sparse high-order estimation, exposure misspecification, nonlinear rates, additional count noise, indistinguishable sources, and an inappropriate independent-source model remain distinct explanations; this one pilot cannot isolate them.

The correct result statement is therefore: **count correction improved selected within-assay moment predictions, while reliable shared-source and cross-assay causal recovery remains unachieved on the current fixed panels.**
