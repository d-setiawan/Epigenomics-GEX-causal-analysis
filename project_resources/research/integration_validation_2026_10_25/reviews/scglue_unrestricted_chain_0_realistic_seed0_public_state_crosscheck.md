# GLUE unrestricted-chain realistic confirmation: completed seed 0

Date: 2026-10-25. Passed the bounded independent public-state check of the sealed `unrestricted_chain_0` / realistic / discovery-cohort GLUE `align_high` confirmation, seed 0. The [numerical receipt](scglue_unrestricted_chain_0_realistic_seed0_public_state_crosscheck.json) records all three recomputed transport rows, public-ID hashes, scaling, and provenance checks. Maximum absolute discrepancy from saved transport values is `1.7763568394002505e-15`.

| Assay pair | Matched squared cost | Squared-state oracle | Random expected squared cost | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 1.238743921173069 | 0.038197658559855 | 4.003147287673899 | 0.302789789256786 |
| RNA--H3K4me3 | 0.845066748512881 | 0.035534760737365 | 4.148637543770377 | 0.196817835701778 |
| H3K27ac--H3K4me3 | 1.473181136584295 | 0.040841262974941 | 4.158295496193707 | **0.347870259747767** |

The maximum reproduces as `0.3478702597477667`, versus saved `0.34787025974776664`, on H3K27ac--H3K4me3. The one-Lipschitz squared-error bounds equal the absolute matched squared costs.

Outcome-free pool reconstruction from the completed selected-fit IDs, both cached baseline-reference ID sets, public assay membership, and the declared pool seed 2718 reproduces 1,567 training and 756 discovery cells per assay. Training/discovery IDs are disjoint, and each selected-ID hash matches the report. The pooled-training state center is `[-0.013880246927312139, 0.0003396350377540681]`, scale `[0.9913896719160854, 0.9969791425859608]`. Discovery labels are not used for pool construction or scaling. The saved assignment is a permutation of these discovery pools with the declared RNA anchor order. Direct squared-distance Hungarian oracles and exact random expectations from state moments reproduce; the random expectations also agree with full squared-cost matrix means.

The report has `role=confirmation`, `cohort=discovery`, `mode=selected`, `reference_only=false`, and only `scglue/align_high`. Exact candidate configuration, completed-fit/export metadata, public file hashes, source/protocol pins, and frozen-plan membership pass. Report SHA256 is `326754a1f67c66f3d89183d8afc17358dc4a8ee1b5fb9b36e9a6a53d19e398d1`. The frozen selection hash remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`, and confirmation-plan hash remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

Only `cell_ids`, `assigned_assay`, and `state` were decompressed from this world's public training/discovery truth archives; only `cell_ids` were decompressed from the permitted completed-fit and cached-reference public embedding archives. Numeric embeddings, hidden biology, private/probe/assessment cohorts, checkpoint binaries, other confirmation reports, and partial seed 1 were not accessed. There was no fitting, CUDA use, tuning, causal scoring, production edit, archive/move, or recovery operation. Checkpoint digest references were reconciled in metadata; native checkpoint verification is separate. Embedding-space assignment optimality and secondary diagnostics were not repeated.

This confirms the completed report's arithmetic and bounded metadata bindings. It does not establish whole-matrix completion, certify operational recovery, identify residual coupling, or imply population, empirical biological, or causal validity.
