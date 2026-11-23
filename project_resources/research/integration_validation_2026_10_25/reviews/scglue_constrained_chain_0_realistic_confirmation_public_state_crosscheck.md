# GLUE confirmation state arithmetic: constrained chain, realistic counts

Date: 2026-10-25. Independent numerical check of only the two completed `constrained_chain_0` / realistic / public discovery-cohort reports for selected GLUE `align_high`, seeds 0 and 1. Both passed. All six pairwise squared-state transport rows and both maxima agree with the saved reports; the largest absolute arithmetic discrepancy is `1.3322676295501878e-15`. The [numerical receipt](scglue_constrained_chain_0_realistic_confirmation_public_state_crosscheck.json) records all calculations, public-ID hashes, metadata bindings, and checked file hashes.

| Seed | Assay pair | Matched squared cost | Oracle squared cost | Random expected squared cost | Normalized gap |
|---|---|---:|---:|---:|---:|
| 0 | RNA--H3K27ac | 1.124005966963974 | 0.043393842019570 | 3.815879339558831 | 0.286445667093823 |
| 0 | RNA--H3K4me3 | 0.610779150602715 | 0.051552743071172 | 3.836235325930271 | 0.147760451580349 |
| 0 | H3K27ac--H3K4me3 | 1.324845339411813 | 0.040859027182645 | 3.808866570767298 | **0.340760016368667** |
| 1 | RNA--H3K27ac | 1.132674716654888 | 0.043393842019570 | 3.815879339558831 | 0.288743555235889 |
| 1 | RNA--H3K4me3 | 0.637604194029704 | 0.051552743071172 | 3.836235325930271 | 0.154848243710786 |
| 1 | H3K27ac--H3K4me3 | 1.275848350674335 | 0.040859027182645 | 3.808866570767298 | **0.327756595284519** |

The worst-pair gap is `0.34076001636866726` for seed 0 and `0.327756595284519` for seed 1; H3K27ac--H3K4me3 is the maximum for both. These are confirmation descriptions and were not used to revise selection.

The exact outcome-free pool construction was reproduced from selected-fit IDs, both cached baseline-reference ID sets, and public assay membership, using the declared pool seed 2718. It yields 1,614 training and 766 discovery cells per assay for both model seeds. RNA eligibility determines these sizes; the mark pools are downsampled to match. The two seeds have identical selected ID hashes and disjoint training/discovery IDs. The common pooled-training state center is `[0.015363853270592968, 0.013271219663996041]`; its scale is `[1.0036465736682494, 1.0026490522725928]`. No discovery labels enter this scale. Saved assignments are valid permutations of the discovery pools with the declared RNA anchor order.

For each assay pair, matched costs were recomputed from saved ID assignments in this same training-scaled state metric. Oracle costs use a direct squared-distance matrix and Hungarian assignment. The exact random-permutation expectation was computed from state moments and crosschecked against the full squared-distance matrix mean. Normalized gaps use `(matched - oracle) / (random - oracle)`. The one-Lipschitz squared-error bounds equal the reported absolute matched costs. The pairwise oracle floors are not asserted to be a jointly attainable three-assay oracle.

Both reports have `role=confirmation`, `cohort=discovery`, `mode=selected`, and `reference_only=false`, and contain only `scglue/align_high`. Their configurations, world/protocol/source metadata, export seals, public file hashes, and membership in the frozen confirmation plan passed the checks recorded in the receipt. The frozen selection and plan hashes were verified unchanged at the start and end of the check:

- `selected_settings.json`: `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`.
- `confirmation_execution_plan.json`: `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.
- Seed 0 report: `5a3f92ecd62d6771c17f7970820d29d15cb63e5becd35984fdad904c872975a8`.
- Seed 1 report: `17c4d6c6387f65abddd16c87723bf23d7dc490140ae020a0662473b50db036c8`.

Only `cell_ids`, `assigned_assay`, and `state` were decompressed from this world's public training/discovery truth archives, and only `cell_ids` from the permitted public embedding archives. Numeric embeddings, hidden activities/residuals, private/probe/assessment cohorts, checkpoints, other confirmation reports, and partial fits were not accessed. There was no fitting, CUDA operation, causal evaluation, tuning, or production edit. Native checkpoint/export verification is a separate audit; this check reconciled checkpoint digest references in metadata without opening checkpoint binaries. Source-state probes, neighbor retention, hubness, shared variance, and latent-space matching optimality were not independently reproduced.

Two model seeds on one existing world and one count regime are repeated conditions. These finite-pool state transport calculations do not establish independent biological-world replication, population generalization, empirical biological correctness, cross-assay residual coupling, or causal graph recovery.
