# GLUE informative constrained-chain confirmation: seeds 0 and 1

Date: 2026-10-25. Passed the bounded independent public-state check of the two completed `constrained_chain_0` / informative / discovery-cohort GLUE `align_high` confirmation reports. All six squared-state transport rows and both maxima reproduce, with maximum absolute discrepancy `2.220446049250313e-15`. The [numerical receipt](scglue_constrained_chain_0_informative_confirmation_public_state_crosscheck.json) contains the full calculations, ID hashes, scaling, and metadata bindings.

| Seed | Assay pair | Matched squared cost | Oracle squared cost | Random expected squared cost | Normalized gap |
|---|---|---:|---:|---:|---:|
| 0 | RNA--H3K27ac | 0.431028897653467 | 0.031759375915185 | 3.870154758153636 | 0.104019904667934 |
| 0 | RNA--H3K4me3 | 0.436233793046022 | 0.044480688321602 | 3.872169607725827 | 0.102347163777718 |
| 0 | H3K27ac--H3K4me3 | 0.506125219270830 | 0.032614757346268 | 3.828595786743402 | **0.124739944235118** |
| 1 | RNA--H3K27ac | 0.490694925790525 | 0.031759375915185 | 3.870154758153636 | 0.119564428406461 |
| 1 | RNA--H3K4me3 | 0.451264817452772 | 0.044480688321602 | 3.872169607725827 | 0.106274082794190 |
| 1 | H3K27ac--H3K4me3 | 0.602427220575216 | 0.032614757346268 | 3.828595786743402 | **0.150109407506561** |

The independently recomputed maxima are `0.1247399442351175` and `0.15010940750656063`, agreeing with saved `0.12473994423511747` and `0.15010940750656052`. H3K27ac--H3K4me3 is the worst pair in both seeds. These are descriptive confirmation outcomes and did not change selection.

Outcome-free pool reconstruction uses only the completed selected fit's public IDs, both cached baseline-reference ID sets, public assay membership, and the declared pool seed 2718. It reproduces 1,942 training and 955 discovery cells per assay, with identical selected-ID hashes across both model seeds and disjoint training/discovery IDs. The common pooled-training state center is `[-0.0027020121334687883, 0.015548173476410607]`; scale is `[0.9941825329310068, 0.9981614176535301]`. Both reports use this same scale. Discovery state labels do not enter pool selection or scaling.

Saved matched IDs are valid permutations of these discovery pools with the declared RNA anchor order. Matched costs were recomputed directly in the training-scaled state metric. Oracle costs use a direct squared-distance matrix and Hungarian optimization, separately for each assay pair. Exact random-permutation expectations from state moments were also checked against the full squared-cost matrix mean. Gaps use `(matched - oracle) / (random - oracle)`; one-Lipschitz squared-error bounds equal the absolute matched costs. Pairwise oracle floors are not claimed to form a jointly attainable three-assay optimum.

Both reports have `role=confirmation`, `cohort=discovery`, `mode=selected`, `reference_only=false`, and only `scglue/align_high`. Configuration, completed-fit/export metadata, public file hashes, source/protocol pins, and frozen-plan membership passed the recorded checks. The frozen selection and plan hashes were unchanged at the start and end of the audit:

- Selection: `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`.
- Confirmation plan: `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.
- Seed 0 report: `dcf057a969bb9aa1b53035cfc0625e5beb3150544a339a05abf7c33df32f4c3c`.
- Seed 1 report: `94f935fd6f6754fa93cac591f1e1de90bb388ca68f3b1fdd530592f5bc7d43ee`.

Only `cell_ids`, `assigned_assay`, and `state` were decompressed from public training/discovery truth archives; only `cell_ids` were decompressed from public embedding archives. Numeric embeddings, hidden activities/residuals, private/probe/assessment cohorts, checkpoints, other confirmation reports, and active/partial fits were not accessed. There was no fitting, CUDA use, tuning, causal evaluation, or production edit. Native checkpoint verification is separate; this check reconciled digest references in metadata. Embedding-space assignment optimality and informative-condition secondary diagnostics were not recomputed.

These two seeds on one existing world/regime are repeated conditions. Their finite-pool state transport scores do not identify residual coupling, supply independent biological-world replication, establish population generalization or empirical biological correctness, or certify causal structure.
