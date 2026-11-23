# Four-setting scMRDR public-state crosscheck

_Independent numerical check · 2026-10-25 · linear_chain_0, informative counts, validation cohort, seed 0_

The [four-setting report](../../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261025_v1/reports/linear_chain_0_informative_val_seed0_scmrdr_all_8366327f.json) passes the independent crosscheck. All 12 pairwise state costs, exact squared-cost oracle floors, random expected costs, normalized gaps, one-Lipschitz bounds, and four worst-pair summaries reproduce. The largest arithmetic discrepancy is `8.881784197001252e-16`.

Normalized gaps are shown in the protocol's setting order, without selecting a setting:

| Setting | RNA–H3K27ac | RNA–H3K4me3 | H3K27ac–H3K4me3 | Worst pair |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0282418403 | 0.0217706661 | 0.0141642455 | 0.0282418403 |
| beta_low | 0.0265686235 | 0.0213694063 | 0.0152856919 | 0.0265686235 |
| gamma_low | 0.0267766254 | 0.0224871490 | 0.0133795950 | 0.0267766254 |
| adversarial_low | 0.0275004297 | 0.0219793341 | 0.0147602482 | 0.0275004297 |

The calculation reconstructed the common training and validation ID intersections from all four settings and both cached baseline references. Outcome-independent equalization reproduces all six saved ID hashes: 1,936 training and 487 validation cells per assay. Training and validation IDs are disjoint. Every saved assignment is a permutation of the same validation pools, with the declared RNA anchor order. The pooled training state center and standard deviations match the report within `1e-14`; the same state scale applies to every setting and assay pair.

Squared state distances were calculated directly from state differences, and an independent minimum-cost assignment supplied each oracle. Random expectation was calculated from source/target state moments. This avoids substituting the native matcher's unsquared Euclidean objective for the squared-cost oracle. Absolute costs and all hash checks are retained in the [numerical receipt](scmrdr_all_linear_chain_0_public_state_crosscheck.json).

The report/source/protocol bindings and all four configurations agree with their declared defaults plus one-setting overrides. Model manifests, study receipts, export seals, world/bank references, and public train/validation export digests also agree. Checkpoint digest references were compared between metadata; checkpoint binaries were not opened or rehashed. The state-cost calculation did not import the production evaluator or metric module.

Only `cell_ids` were decompressed from public embedding NPZs. Only `cell_ids`, `assigned_assay`, and `state` were decompressed from the public training/validation truth archives. Numeric embeddings, hidden activities/residuals, private/probe/assessment cohorts, causal scores, and the other selection world were not accessed. No integration or probe fitting, CUDA operation, latent-space reassignment, or setting selection occurred.

These are verified finite-pool state-transport diagnostics for one world and one seed. They are not the two-world selection result, a reliability comparison, or certification of residual coupling or real-data biological correctness.
