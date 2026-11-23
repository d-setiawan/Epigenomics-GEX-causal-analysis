# scMRDR realistic constrained-chain confirmation: seed 1

Date: 2026-10-25. Passed the independent public-state arithmetic check of the completed `constrained_chain_0` / realistic / discovery-cohort scMRDR `baseline` report, seed 1. The [numerical receipt](scmrdr_constrained_chain_0_realistic_seed1_public_state_crosscheck.json) records direct costs, ID hashes, scaling, frozen metadata bindings, and the four-report pool comparison.

| Assay pair | Matched squared cost | Oracle squared cost | Random expected squared cost | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 0.543704057224665 | 0.043393842019570 | 3.815879339558831 | **0.132620845204426** |
| RNA--H3K4me3 | 0.297182171615013 | 0.051552743071172 | 3.836235325930271 | 0.064900932420674 |
| H3K27ac--H3K4me3 | 0.525785998500669 | 0.040859027182645 | 3.808866570767298 | 0.128695860002630 |

The independently recomputed maximum is `0.13262084520442566`, versus saved `0.13262084520442569`. Maximum absolute discrepancy among all checked transport values is `1.3322676295501878e-15`. The one-Lipschitz squared-error bounds equal the absolute matched squared costs.

The public pools were reconstructed independently from this completed fit's IDs, both cached baseline-reference ID sets, and public assay membership, including the declared seed-2718 equalization. They contain 1,614 training and 766 discovery cells per assay, with disjoint training/discovery IDs. The pooled-training state center is `[0.015363853270592968, 0.013271219663996041]` and scale is `[1.0036465736682494, 1.0026490522725928]`. The saved assignment is a valid permutation of these discovery pools with the declared RNA anchor order. Direct squared-distance Hungarian optima and exact random expectations from state moments reproduce the saved oracle and null costs; the null is additionally checked against the full squared-distance matrix mean.

All four completed realistic constrained-chain reports have exactly the same six selected-ID hashes, selected counts, state center, and training scale:

| Method / selected setting | Seed | Training cells per assay | Discovery cells per assay | Same six selected-ID hashes |
|---|---:|---:|---:|---|
| GLUE / align_high | 0 | 1,614 | 766 | Yes |
| GLUE / align_high | 1 | 1,614 | 766 | Yes |
| scMRDR / baseline | 0 | 1,614 | 766 | Yes |
| scMRDR / baseline | 1 | 1,614 | 766 | Yes |

This comparison reads the four completed reports and checks each against the independently reconstructed seed-1 pools. Prior seed-0 secondary diagnostics were not repeated. The shared pools make these finite-pool comparisons consistent across methods/seeds; they do not add independent biological-world replication.

The seed-1 report has `role=confirmation`, `cohort=discovery`, `mode=selected`, `reference_only=false`, and only `scmrdr/baseline`. Its configuration, source/protocol metadata, completed-fit/export seals, public file hashes, and frozen-plan membership pass the recorded checks. Report SHA256 is `d74b99376c7de2640ba3b8f9aa0edead745f916baaf217f0490dd5decd9742f8`. The selection hash remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`, and confirmation-plan hash remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

Only `cell_ids`, `assigned_assay`, and `state` were decompressed from public training/discovery truth archives, and only `cell_ids` from the permitted public embedding archives. Numeric embeddings, hidden biology, private/probe/assessment cohorts, checkpoint binaries, informative confirmation reports, and active/partial fits were not accessed. No fitting, CUDA use, tuning, causal scoring, or production edit occurred. Checkpoint digest references were reconciled in metadata; native checkpoint verification remains separate. Latent-space assignment optimality and seed-1 secondary diagnostics were not recomputed. State agreement does not identify residual coupling, population generalization, empirical biological correctness, or causal graphs.
