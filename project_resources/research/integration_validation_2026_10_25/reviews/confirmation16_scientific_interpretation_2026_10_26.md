# Scientific interpretation of the 16 completed confirmations

All 16 planned completed discovery-cohort reports are bound to their prior independent numerical receipts: all 48 pairwise state transport rows have been independently reproduced. The [completed-report synthesis](confirmation16_scientific_interpretation_2026_10_26.json) records every report SHA256, frozen plan/selection binding, pool group, diagnostic value and independent-review reference. This is a scientific report synthesis, not a replacement for the separate operational completion/integrity audit. The frozen settings remain GLUE `align_high` and scMRDR `baseline`.

The table summarizes both seeds in each world/regime/method group. “Weakest transfer” is the minimum of six cross-assay **mean** state R² values under the fixed source StandardScaler+Ridge probe; it is distinct from matching error.

| World | Counts | Method | Worst gap, seed 0 | Worst gap, seed 1 | Weakest transfer R², seed 0 | Weakest transfer R², seed 1 |
|---|---|---|---:|---:|---:|---:|
| constrained chain | realistic | GLUE | 0.34076 | 0.32776 | 0.66265 | 0.66162 |
| constrained chain | realistic | scMRDR | 0.13432 | 0.13262 | 0.77259 | 0.78014 |
| constrained chain | informative | GLUE | 0.12474 | 0.15011 | 0.86014 | 0.89365 |
| constrained chain | informative | scMRDR | 0.03664 | 0.04416 | -3.51271 | -12.63994 |
| unrestricted chain | realistic | GLUE | 0.34787 | 0.32757 | 0.63786 | 0.73755 |
| unrestricted chain | realistic | scMRDR | 0.12464 | **1.44463** | 0.79252 | -1.87561 |
| unrestricted chain | informative | GLUE | 0.14380 | 0.14657 | 0.90840 | 0.78897 |
| unrestricted chain | informative | scMRDR | 0.02690 | 0.02757 | 0.17300 | -1.95975 |

Matching and decoder behavior give different conclusions. scMRDR has lower oracle-relative matching gap than GLUE in seven of eight matched conditions, but unrestricted-realistic seed 1 is a substantial exception: both RNA-to-mark costs exceed the exact same-pool uniform-random-permutation expectation, and the worst gap is `1.4446291191821`. This adverse result is preserved. A normalized gap above one is mathematically valid; no clipping, exclusion, retry or selection change follows from it.

GLUE's stored cross-assay mean and coordinate R² values are positive in all eight confirmation conditions. scMRDR has negative cross-assay mean R² in four of eight conditions. Three are informative-count conditions with low state gaps (`0.02757`–`0.04416`): both constrained-chain seeds and unrestricted-chain seed 1. The other informative scMRDR seed has positive but weak RNA-to-H3K4me3 mean R² (`0.17300`). Thus low forced one-to-one state-matching error does not guarantee successful transfer of the fixed RNA-trained linear decoder. Negative R² refers to performance relative to the target-state mean under this declared probe; it does not establish that the embedding contains no state information or explain the training mechanism.

Directed neighborhood coverage supplies another property. Informative scMRDR conditions have minimum k15 target coverage approximately `0.282`–`0.371`, while their state gaps remain small. Conversely, unrestricted-realistic scMRDR seed 1 has minimum coverage `0.77249` yet poor state matching. Global one-to-one matching forces target use, while directed local neighbors and a fixed source decoder need not agree with that assignment. The [analytic illustration](scmrdr_constrained_chain_0_informative_seed0_diagnostics_crosscheck.md) proves that such disagreement can occur, without diagnosing these actual fits or identifying state ordering from unpaired marginals.

All four method/seed reports within each world/count-regime group share identical selected-ID hashes, cell counts and training-derived state scales. That supports fair conditional comparisons within each group. Informative conditions have lower state gaps than their corresponding realistic conditions throughout this matrix, but count regimes have different eligibility pools and training scales. These are conditional descriptions, not a causal attribution solely to count quality.

The synthesis uses completed report JSONs for all diagnostic summaries. Independent public-array reproduction of cross-probe/hubness values exists for eight of the 16 reports, flagged individually in the JSON; it is not claimed for every GLUE decoder. All primary state-cost rows have independent public-state reproduction. Supplemental native-float32 versus float64 probe differences remain recorded in the individual receipts and do not replace the frozen native metrics.

The experimental units are the two existing confirmation worlds. Seeds and count regimes are repeated conditions, not 16 independent biological-world replicates; this small cached bank does not support a generic integration-validity or population-generalization claim. No new setting was selected from confirmation. State transport, fixed decoder transfer and coverage do not identify cross-assay residual coupling or establish conditional independence or causal graph preservation. There was no private-biology access, model/checkpoint loading, GPU use, new fitting, causal evaluation or production edit in this completed-report synthesis. Baseline execution was not started by this review.
