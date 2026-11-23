# Eight-setting first-world state-cost crosscheck

_Independent numerical review · 2026-10-25 · public train/validation IDs and state only_

The [full first-world report](../../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261025_v1/reports/linear_chain_0_informative_val_seed0_scglue-scmrdr_all_8366327f.json), SHA-256 `24e89e3cde4dbeb3355baffd86417d57b8d10ff2eb01469ab1a3571361a2622c`, passes this bounded arithmetic and binding check. All 24 pairwise costs, exact squared-cost oracle floors, random expectations, normalized gaps, one-Lipschitz bounds, and eight worst-pair summaries independently reproduce. The maximum absolute discrepancy is `8.881784197001252e-16`.

The table follows the predeclared method/setting order; it is not a selection result.

| Method | Setting | Recomputed worst-pair gap |
| --- | --- | ---: |
| scGLUE | baseline, freshly fitted | 0.1461251857 |
| scGLUE | align_low | 0.1923659666 |
| scGLUE | align_high | 0.1240159932 |
| scGLUE | latent_small | 0.1067538512 |
| scMRDR | baseline, freshly fitted | 0.0282418403 |
| scMRDR | beta_low | 0.0265686235 |
| scMRDR | gamma_low | 0.0267766254 |
| scMRDR | adversarial_low | 0.0275004297 |

The fresh scGLUE baseline is distinct from the [cached baseline reference](../../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261025_v1/reports/linear_chain_0_informative_val_seed0_scglue-scmrdr_baseline_5e025d63.json), whose worst gap is `0.14200130451844178`. The cached value is a reference diagnostic, not the current all-setting baseline value. This review does not assign a cause to the difference between fitted artifacts.

All four scMRDR result objects and saved assignments are exactly equal to the [earlier scMRDR-only all-setting report](../../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261025_v1/reports/linear_chain_0_informative_val_seed0_scmrdr_all_8366327f.json). Its six selected-pool hashes, pool sizes, common-before-equalization counts, and training state center/scale reproduce exactly in the full report. Every setting uses the same 1,936 training and 487 validation cells per assay. Training and validation identities are disjoint, and every saved assignment is a valid permutation of the common validation pools with the declared RNA anchor order.

The independent calculation constructs squared distances directly from state differences and solves their assignment optimum, rather than substituting the latent matcher's unsquared-distance objective. Random expected costs are calculated from state moments. The common pooled training center and population standard deviations match within `1e-14`. All eight configurations agree with declared defaults and candidate overrides; report source/protocol hashes, model manifests, study receipts, export seals, world/bank references, and public train/validation export digests agree. The [numerical receipt](all_settings_linear_chain_0_public_state_crosscheck.json) records all 24 rows, hashes, checks, and limits.

Only public embedding `cell_ids`, and public truth `cell_ids`, `assigned_assay`, and `state` for training/validation, were decompressed. No numeric embedding matrices, hidden activities/residuals, private/probe/assessment cohorts, causal scores, checkpoints, or other-world scientific outcomes were accessed. No integration/probe fit, CUDA operation, or setting selection occurred. Saved assignments were scored and checked as permutations; latent-space matching itself was not rerun.

These are descriptive state-transport results for one world, one count regime, and seed 0. The second selection world is still required by the frozen rule. Neither this arithmetic pass nor the displayed differences certify residual coupling, population reliability, or real-data integration correctness.
