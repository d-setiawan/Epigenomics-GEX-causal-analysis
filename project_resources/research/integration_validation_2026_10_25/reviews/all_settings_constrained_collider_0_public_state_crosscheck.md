# Final development selection: independent numerical and provenance review

Date: 2026-10-25. Passed the bounded extension to the full `constrained_collider_0` informative-count validation report and the frozen two-world selection. The [numerical receipt](all_settings_constrained_collider_0_public_state_crosscheck.json) records 24 independently recomputed second-world pairwise transport rows, public-ID pool hashes, training scale, configuration/source/export bindings, all eight selection scores, and the exact confirmation schedule.

All 12 GLUE rows and all 12 scMRDR rows agree with the report, with maximum absolute arithmetic discrepancy `8.881784197001252e-16`. The four scMRDR result, assignment, and evidence objects exactly equal the previously reviewed method-only report; its common selected pools, state scale, and numerical receipt also agree exactly. The full second-world report uses 1,934 training and 491 validation cells per assay. The common training center is `[0.03450580511029551, -0.008007561567630953]` and scale is `[1.012353982937625, 1.0043266803257538]`. Assignments are permutations of these validation pools; squared-cost oracle optima and exact random-permutation expectations were recomputed directly.

| Method | Setting | First-world worst gap | Second-world worst gap | Two-world mean |
|---|---|---:|---:|---:|
| GLUE | baseline | 0.146125185660415 | 0.180440405135486 | 0.163282795397950 |
| GLUE | align_low | 0.192365966564603 | 0.231387967981496 | 0.211876967273050 |
| GLUE | **align_high** | 0.124015993151834 | 0.133285989904691 | **0.128650991528262** |
| GLUE | latent_small | 0.106753851218553 | 0.216033340817203 | 0.161393596017878 |
| scMRDR | **baseline** | 0.028241840323401 | 0.030093612631005 | **0.029167726477203** |
| scMRDR | beta_low | 0.026568623533312 | 0.038084162781527 | 0.032326393157419 |
| scMRDR | gamma_low | 0.026776625406681 | 0.035672061123317 | 0.031224343264999 |
| scMRDR | adversarial_low | 0.027500429746778 | 0.036288422036711 | 0.031894425891744 |

The unchanged rule takes the maximum over three pairwise normalized state gaps in each world, averages those two maxima equally, and minimizes within each method, using candidate names to break ties. Independently recomputed means differ from frozen JSON values by at most `2.7755575615628914e-17`. They select GLUE `align_high` (`0.1286509915282622`) and scMRDR `baseline` (`0.029167726477202765`). First-world arithmetic was linked to the prior independent eight-setting receipt and its exact report hash; only the second-world costs were recomputed again in this extension. The GLUE baseline is the fresh selection fit; the earlier cached baseline is reference-only. GLUE `latent_small` has the best first-world score, but its first-world advantage does not carry over to the second selection world.

The frozen evidence list contains exactly the two declared all-setting public validation reports, with complete candidate sets for both methods. The confirmation plan exactly matches the ordered Cartesian product of two declared worlds (`constrained_chain_0`, `unrestricted_chain_0`), two count regimes, two methods, and seeds 0/1: 16 unique jobs. Every GLUE job uses `align_high`, every scMRDR job uses `baseline`, and the plan binds the exact frozen selection hash. Selection and confirmation worlds are disjoint. This verifies the schedule, not its execution or results.

Checked artifact SHA256 values:

- Full second-world report: `ea8339e9cb728d08176b282f3d92b4187701f76d8abd126d24cfa33bf85d02ff`.
- Frozen `selected_settings.json`: `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`.
- `confirmation_execution_plan.json`: `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.
- Protocol: `df8e8424f02ea950062c1aad2076f89e9694861d63fe898722bfc3545adbfdf1`.

Only public train/validation `cell_ids`, `assigned_assay`, and `state` were decompressed from truth archives, and only IDs from public embedding archives. Source/config and completed-fit metadata were read and their recorded bindings checked. Numeric embeddings, hidden activities/residuals, private/probe/assessment cohorts, checkpoints, confirmation state/results, and partial fits were not accessed. No integration/probe fitting, CUDA operation, causal evaluation, selection/plan mutation, or production edit occurred. Checkpoint digest references were checked between metadata without reading checkpoint binaries; latent-space assignment optimality was not recomputed.

The conclusion is relative performance among the declared settings on this small existing development bank. It does not establish fresh-world reliability, empirical biological correctness, residual coupling, conditional independence, or causal graph recovery. Seeds and count regimes are repeated conditions; they do not supply additional biological-world replication.
