# Second-world scMRDR public-state crosscheck

Date: 2026-10-25. Independent bounded check of the four-setting scMRDR report for `constrained_collider_0`, informative counts, validation seed 0. All 12 matched/oracle/random squared-cost rows and normalized gaps, four worst-pair summaries, public-ID pools, training scale, and checked configuration/source/export metadata bindings passed. Maximum arithmetic discrepancy was `8.881784197001252e-16`.

The checked report SHA256 is `b17f7a22cfb035fbd9d06116390121c60b1a7b2a332791412ee218e4dc0ce4b5`. The [numerical receipt](scmrdr_all_constrained_collider_0_public_state_crosscheck.json) records all pairwise costs, ID hashes, source/config hashes, and binding checks.

| scMRDR setting | First-world worst gap | Second-world worst gap | Equal-weight two-world mean |
|---|---:|---:|---:|
| baseline | 0.02824184032340051 | 0.03009361263100502 | 0.029167726477202765 |
| beta_low | 0.026568623533311585 | 0.038084162781526744 | 0.03232639315741916 |
| gamma_low | 0.026776625406681315 | 0.035672061123316896 | 0.031224343264999108 |
| adversarial_low | 0.027500429746778444 | 0.03628842203671056 | 0.0318944258917445 |

Each mean averages the two worlds' maxima over their three assay-pair gaps. It does not pool cells or add seed/world replication. The first-world report was reconciled to its prior independent receipt. All four supplied two-world means agree to floating-point roundoff. Baseline has the smallest descriptive mean within these four scMRDR settings; this check does not freeze or select settings, and no settings-freeze file was written. The complete both-method decision remains outside this audit.

The reconstructed common pools contain 1,934 training cells and 491 validation cells per assay, with disjoint train/validation IDs. The common training-derived state center is `[0.03450580511029551, -0.008007561567630953]`, and scale is `[1.012353982937625, 1.0043266803257538]`. Saved assignments are valid permutations of the same public validation pools with the declared RNA anchor order. Squared-cost Hungarian optima were recomputed directly; random expected costs were recomputed exactly from state moments.

Only `cell_ids` were decompressed from public embedding archives, and only `cell_ids`, `assigned_assay`, and `state` from the public training/validation truth archives. Numeric embeddings, hidden activities/residuals, private/probe/assessment cohorts, checkpoints, causal scores, and partial GLUE fits were not accessed. There was no integration/probe fitting, CUDA use, or causal evaluation. Checkpoint digest references were reconciled between JSON metadata; checkpoint binaries were not rehashed. Latent-space assignment optimality was not recomputed. These finite-pool state checks do not identify residual coupling or certify biological correctness on empirical data.
