# Results of the 225-arm count-source comparison

The frozen comparison completed all **225 arms** without execution failures,
warnings or source-bootstrap fitting failures. It found **zero equivalent signed
cross-assay moment pairs**, not merely zero unique matches. Consequently no
source reached CF confirmation and no causal graph was extracted. Some noise
models improved heldout within-assay covariance prediction, and expanded
depth-adjusted simulated panels represented state more strongly. Neither
result established shared cross-assay sources.

The original 300-feature unit-exposure fits all reached the computational
rank ceiling. Their matching outcome is therefore cap-limited. The separate
[selected-dimension follow-up](../count_rank20_2026_11_01/README.md) addresses
that limitation and preserves this original block unchanged.

## Coverage and data

The six families were conditional negative-binomial noise, excess-zero
measurement noise, depth elasticity, uncertainty shrinkage of third moments,
bounded Poisson generating-function orientation, and a nested expansion from
30 to 300 existing features. Selected interactions and an ordinary-moment
ablation give 25 variants. Each ran on three datasets with two source seeds
and one independent-column control: 150 original-data and 75 control arms.
This is finite prespecified coverage, not every possible model.

| Dataset | Training cells per assay | Test cells per assay | Source-fit/calibration rows |
| --- | --- | --- | --- |
| scMTR-seq S1 | RNA 602; H3K27ac 615; H3K4me3 614 | 595 paired cells | Deterministic 75%/25% split within assay |
| Existing linear-chain world | 4,000 | 2,000 | 3,000/1,000 |
| Existing constrained-collider world | 4,000 | 2,000 | 3,000/1,000 |

Training remained unpaired, with no state or causal labels. The same cells and
cached worlds were reused. The original 30-feature arrays, IDs and depths are
exact prefixes of the expanded panels; the extraction audit independently
replayed every selected count from the existing source files. No PCA input or
variance-ranked feature selection was introduced.

In the real training pool, cells with zero panel counts changed from 14 to 0
for RNA, 341 to 18 for H3K27ac and 317 to 6 for H3K4me3. This improves panel
coverage, but individual retained histone features remain sparse. The expanded
panels contain two overlapping H3K27ac promoter pairs and one H3K4me3 pair;
independent feature measurement noise is not established for them.

## Source eligibility and where matching stopped

| Original-data block | Arms with at least one eligible source | Eligible source instances | Nonzero assay fits | Cap-binding assay fits |
| --- | ---: | ---: | ---: | ---: |
| Real scMTR-seq, 50 arms | 6 | 6 | 102 | 36 |
| Linear-chain world, 50 arms | 34 | 97 | 118 | 36 |
| Constrained-collider world, 50 arms | 36 | 126 | 118 | 36 |

These are repeated source instances across variants and seeds, not distinct
biological causes or independent replicates. All six real-data eligible
instances were RNA sources. Neither histone assay had an eligible real source
in this block.

Across all 675 assay fits, including controls, model fitting returned `ok`;
rank-zero fits are included in that count. There were 234 eligible source
instances across 81 arms, of which five eligible instances came from controls.
All cross-assay moment-equivalence sets were empty, so uniqueness was not the
bottleneck. CF was **not assessed** in every arm. The exact conditional-kernel
CF fixtures passed preflight, but this scientific block did not exercise CF
confirmation on candidate sources; it is not evidence that CF confirmation
itself failed.

The [gate audit](reviews/gate_summary.json) examined 1,166 fitted source
directions. Among overlapping failure reasons, 677 failed median loading
stability, 705 failed lower-decile loading stability, 267 failed orientation
fit, 148 lacked 26 valid moment replicates, and 61 had invalid or nonpositive
calibration variance. Successful numerical fitting therefore did not imply
reliable source estimation.

Even among the 344 signed original-data comparisons with eligible endpoints,
the smallest normalized point-distance plus bootstrap-band score was 11.895,
against a required value at most one. Its point distance was 2.144 and its
uncertainty radius 9.751. This was a collider-world PGF comparison; the closest
linear-world score was 28.363. There was no eligible real-data cross-assay
comparison. The block did not narrowly miss the frozen equivalence threshold.

Eight of the 75 column-control arms selected a nonzero rank, all in the
constrained-collider world and only in one assay per arm. No control accepted
a shared pair. These permutations do not supply a biological false-sharing
FDR estimate, especially when other assays have no eligible control source.

## The rank ceiling is a real limitation of this block

All 36 original unit-exposure 300-feature arms had every assay capped: 108
assay fits, with uncapped empirical ranks from 7 to 19. The real Poisson panel
selected 11 RNA, 12 H3K27ac and 12 H3K4me3 modes, but fitted only six per assay.
These empirical ranks measure coherent variation under the specified null;
they are not known biological dimensions.

Of the 648 fitted directions in these capped arms, 36 passed every other
eligibility gate and were excluded solely by the cap: 29 in the linear-chain
world and seven in the collider world. The remaining 612 directions also
failed another gate. No capped real-data direction passed all other gates.
This describes the truncated fits; a refit at the full selected dimension can
change every source and its uncertainty. It neither guarantees full-rank
success nor licenses treating cap-limited abstention as full-panel failure.

The subsequent [54-arm follow-up](../count_rank20_2026_11_01/RESULTS.md) is now
complete. It fitted the full selected dimensions without any cap exclusions
and still accepted no moment pair. Its source-stability failures and heldout
comparisons are reported separately; this original block remains unchanged.

## Within-assay count modeling did improve some predictions

Errors below are heldout covariance RMSE after dividing each covariance entry
by the corresponding product of fixed source-fit observed-rate standard
deviations. Lower is better. The table uses the same
real 30-feature panel and unit exposure. Covariance predictions are invariant
to the fitted orthogonal source orientation at a fixed covariance subspace;
two source seeds are not counted as independent covariance experiments.

| Model | RNA | H3K27ac | H3K4me3 |
| --- | ---: | ---: | ---: |
| Diagonal training baseline | 0.14473 | 0.10537 | 0.10526 |
| Ordinary moments, Poisson-matched rank | 0.16776 | 0.18677 | 0.21077 |
| Poisson correction | 0.14166 | 0.11288 | 0.10794 |
| NB shape 10 | 0.13272 | 0.11316 | 0.10998 |
| NB shape 2 | 0.11143 | 0.11406 | 0.11036 |
| NB shape 0.5 | 0.10661 | 0.11111 | 0.11105 |
| Excess-zero retention 0.75 | 0.11989 | 0.11372 | 0.10831 |
| Excess-zero retention 0.5 | 0.10583 | 0.11777 | 0.11068 |
| NB shape 2 plus retention 0.5 | 0.10661 | 0.11566 | 0.11105 |

The RNA improvement is substantial in this covariance task. It does not
establish the technical interpretation of NB dispersion or dropout. In this
table, every histone model remains worse than the diagonal baseline on full
covariance error. Different noise settings can also induce different selected
ranks, so these are changes to the complete estimator, not isolated tests of
one correction at fixed dimension.

Across the nine unique dataset-by-assay unit-exposure 30-feature settings,
NB shape 0.5 improved full covariance RMSE over Poisson in eight and worsened
one; off-diagonal error improved in six and worsened three. NB shape 2
improved both measures in seven and worsened two. These descriptive counts do
not establish significance or select a winning model across reused data.

PGF orientation and third-moment shrinkage retain the same Poisson covariance
subspace, so their covariance errors tie Poisson up to numerical precision.
Their tests concern source orientation and stability. Neither produced a
cross-assay moment pair.

Panel size must be compared on the unchanged 30-feature anchor, not by mixing
30-by-30 and 300-by-300 error averages. For the Poisson unit-exposure fits,
expansion improved full covariance error on eight of nine anchor settings;
real H3K27ac worsened from 0.11288 to 0.12061. These are the capped original
fits. Separate full-panel and anchor readouts for every arm remain in the
[evaluation summary](evaluation_v1/summary.json).

## Cross-assay prediction and simulated state

Every cross-assay prediction was the empty-candidate zero default at both
matching levels. Accordingly, every real paired-test improvement over the
zero predictor, including every bootstrap improvement, was exactly zero.
This is not a finding of biological independence. For reference, the
unit-exposure 30-feature zero-predictor errors were 0.06157 for RNA–H3K27ac,
0.06200 for RNA–H3K4me3, and 0.05187 for the two histone assays. The same 595
paired test cells and 200 complete-triplet resamples were used throughout.

The largest observed absolute source-to-state correlations across the
prespecified original-data arms were 0.92588 in the linear-chain world
(`nb2_300_elastic`, seed 0, H3K4me3) and 0.88295 in the collider world
(`poisson300_offset`, seed 0, H3K4me3). These are descriptive maxima across
many candidates, not unbiased performance estimates or cross-assay recovery.
In both cases the other two assays had rank zero, leaving no shared source to
match. All matched-source state readouts were therefore unavailable.

Depth adjustment eliminated all real-data coherent modes in both panel
widths. A fitted depth slope can remove biological variation as well as
technical variation; this outcome does not prove the removed structure was
technical. Comparisons across exposure treatments have different rate targets
and fixed scales and are not a common RMSE competition.

## Validation and preserved artifacts

- All 29 preflight tests passed; separate reviewers checked the observation
  algebra, runner, panel extraction and evaluator.
- [Decision replay](reviews/audit_extension_fits.json) reproduced all 225
  ranks, eligibility decisions, candidate sets, ambiguity sets and predictions.
- [Independent metric replay](reviews/extension_metric_audit.json) checked all
  225 arms, 1,350 within-assay views, 900 real cross-assay views and 2,250 state
  summaries, plus literal paired bootstrap checks and all 200 conditional
  error identities. Maximum numerical difference was $1.44\times10^{-15}$.
- [Fit provenance](runs_v1/provenance.json) and
  [evaluation provenance](evaluation_v1/provenance.json) seal the inputs,
  scientific sources and all arm artifacts. Fit wall time was 686.76 seconds
  with two CPU workers; no package or GPU changes were required.

The [interpretation review](reviews/result_interpretation_review.md) is
independent of the implementation. The [method](METHOD.md) and
[population note](reviews/kernel_population_identifiability.md) explain why
these finite negative results neither identify causal structure nor prove
that every restricted unpaired count model is impossible. Nonlinear source
models, feature-specific noise parameters and a comprehensive likelihood
model are outside this finite comparison.
