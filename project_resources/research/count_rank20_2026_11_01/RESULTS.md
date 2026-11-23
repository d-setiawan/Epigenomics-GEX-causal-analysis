# Results after removing the six-component ceiling

All **54 follow-up arms** completed without execution failures, warnings or
source-bootstrap fitting failures. Every nonzero fitted rank equaled the
uncapped training-selected rank, between 7 and 19; no fit reached the new
ceiling of 20. There were **zero equivalent signed cross-assay moment pairs**,
so CF confirmation was not assessed and no graph was extracted.

This resolves the computational-ceiling limitation of the
[original 300-feature unit-exposure branch](../count_extensions_2026_11_01/RESULTS.md).
It does not establish that every larger-panel or nonlinear source model must
fail. The full selected-dimension estimators still failed the unchanged source
stability and cross-assay fingerprint requirements.

## What was held fixed

Six unit-exposure 300-feature variants were repeated across three existing
datasets, with two source seeds and one column-permutation control: 36
original-data and 18 control arms. The runner, estimator and evaluator were
byte-identical to the original reviewed files. The only scientific setting
changed was the computational rank ceiling and its associated cap-binding
eligibility restriction.

The [independent fit audit](reviews/audit_rank20_fits.json) confirmed exact
preservation for all 162 assay records of the 49 null maxima, eigenvalues,
empirical p-values, uncapped ranks, input hashes, feature and split IDs,
exposure parameters, bootstrap seeds, and point count-moment quantities.
No original model or artifact was overwritten. No cells, assays, features or
worlds were added.

This is a training-motivated adaptation after the original fit/evaluation
seals, explicitly declared in [protocol.json](protocol.json). The datasets and
test cohorts have been inspected previously; this is exploratory sensitivity
analysis, not independent validation.

## Removing the ceiling did not stabilize the sources

| Original-data block | Arms | Arms with eligible sources | Eligible source instances |
| --- | ---: | ---: | ---: |
| Real scMTR-seq | 12 | 0 | 0 |
| Linear-chain world | 12 | 10 | 21 |
| Constrained-collider world | 12 | 2 | 4 |

The full-dimension fits contained 1,152 source directions. Twenty-five passed
all source gates; 1,127 failed at least one remaining gate. The
[gate summary](reviews/gate_summary.json) records overlapping failures:

| Gate | Failing source instances |
| --- | ---: |
| Median loading stability | 1,012 |
| Lower-decile loading stability | 1,097 |
| Orientation residual | 704 |
| At least 26 valid moment bootstraps | 206 |
| Positive finite calibration variance | 104 |
| Computational ceiling | 0 |

All 422 real-data source instances failed lower-decile loading stability.
Numerical completion therefore did not imply reproducible biological source
directions. The counts include repeated variants/seeds and overlapping gates;
they are not independent biological units.

Fourteen signed source comparisons had eligible endpoints. The closest
point-distance plus bootstrap-band score was 15.8213, against acceptance at
most one: point distance 4.3483 and uncertainty radius 11.4729. The closest
collider-world score was 23.8789. There were no eligible real-data comparisons.
The failure was not a narrow miss of the fixed equivalence margin.

All 18 controls had rank zero in every assay. Their empty shared-source sets
do not calibrate false sharing between independently eligible source laws.
All CF assessments remained explicitly unassessed because the first moment
screen returned no candidates.

## More dimensions did not consistently improve heldout covariance

The table compares each full-dimension estimator with its corresponding
cap-six version on nine unique dataset-by-assay settings. Cells and feature
scales are identical. Each entry is **improved / worsened**; there were no
ties at tolerance $10^{-12}$. Source seeds are not treated as independent
covariance replicates.

| Model | Full-panel full covariance | Full-panel off-diagonal | Original-30 full covariance | Original-30 off-diagonal |
| --- | ---: | ---: | ---: | ---: |
| Poisson / PGF / shrinkage, each | 2 / 7 | 1 / 8 | 7 / 2 | 1 / 8 |
| Ordinary moments | 3 / 6 | 0 / 9 | 9 / 0 | 0 / 9 |
| NB shape 2 | 1 / 8 | 1 / 8 | 5 / 4 | 0 / 9 |
| Excess-zero retention 0.5 | 1 / 8 | 2 / 7 | 3 / 6 | 0 / 9 |

Poisson, PGF and shrinkage have the same covariance prediction at a fixed
selected subspace because source rotation leaves $BB^\top$ unchanged. Their
different source orientations remain separate matching experiments.
These are descriptive comparisons, not independent significance tests.

For real Poisson fits, full covariance error on the unchanged 30-feature
anchor changed from 0.13412 to 0.12914 for RNA, from 0.12061 to 0.12467 for
H3K27ac, and from 0.10786 to 0.11182 for H3K4me3. Lower is better. Off-diagonal
prediction generally worsened with more components. Extra coherent training
modes therefore did not provide a consistent heldout improvement.

## Cross-assay and state readouts

Every cross-assay prediction remained the empty-candidate zero default. Both
matching levels exactly tied the zero baseline on the paired real test cells
and all 200 complete-cell bootstrap resamples. This is abstention, not a
finding of independence or recovered causal structure.

The largest absolute all-source state correlations in this follow-up were
0.74858 in the linear-chain world (`ordinary300_unit`, seed 1, H3K4me3) and
0.73193 in the collider world (`pgf300_unit`, seed 1, RNA). These are descriptive
maxima among many source projections. They do not identify shared sources or
causal directions; matched-source state readouts remain unavailable. The
larger maxima in some depth-adjusted original arms concern different targets
and settings and are not this follow-up's benchmark baseline.

## Independent checks and conclusion

The high-rank preflight verified exact tensor/source algebra at 20 dimensions
and correct abstention when the twentieth corrected mode is invalid. Its
[receipt](reviews/high_rank_preflight.json) records those deterministic checks;
they are not new benchmark worlds.

The [metric replay](reviews/extension_metric_audit.json) independently verified
all 54 arms, 324 within-assay views, 216 real cross-assay views, 540 state
summaries, 648 literal bootstrap comparisons and 43,200 conditional error
identities. The maximum numerical discrepancy was $1.17\times10^{-15}$.
There were no unscorable views. Scientific inputs and outputs are sealed in
[fit provenance](runs_v1/provenance.json) and
[evaluation provenance](evaluation_v1/provenance.json). Fit wall time was
313.58 seconds on two CPU workers.

Combined with the original block, the investigation completed 279 runs across
six identified extension families and this dimension sensitivity. Some count
models improved a narrower within-assay task, but none accepted a cross-assay
source pair under the declared rules. The remaining failures concern source
reproducibility and fingerprint agreement after computational truncation was
removed. Full nonlinear intensity models, learned feature-specific technical
noise and every possible alternative estimator remain untested.
