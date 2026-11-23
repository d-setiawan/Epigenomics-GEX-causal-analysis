# Old frozen baseline: constrained-chain draw 1, realistic counts

Independent JSON/graph-metadata review of the completed `realistic/primary/seed0` stage. Metadata bindings and reported score arithmetic pass; adverse scientific outcomes are retained. The [numerical/provenance receipt](constrained_chain_1_realistic_frozen_baseline_json_audit.json) records 54 exact metadata/source hashes, all 16 evaluation-stage bindings, full graph metrics, controls, saved probe summaries and scope limits. Report SHA256 is `5a01502fbd83520cb2e0c496caec655b1d808c271072d79e47e4192fb8ed5d8b`; input fingerprint is `5ec327bbab4507f0c045b2da9eeafc9aedba1a60394f3f7d7aa3b02434ea3c7b`.

This is the original frozen assessment suite, not the newly selected integration-validation settings. The frozen stage argv and source/assessment-lock bindings agree. GLUE uses `lam_align=0.05`, latent dimension 30 and maximum 120 epochs; scMRDR uses beta 5, gamma 10, adversarial weight 10, shared/private dimensions 20/20 and maximum 200 epochs. Both seeds are 0. Current model/export metadata match the report's recorded provenance. Checkpoint and array digest references agree between metadata; their binary contents were not opened or rehashed.

| Method | Layer | Cross adjacency TP/FP/FN | Global endpoint precision/recall | Biological directed stress TP/FP/FN |
|---|---|---|---|---|
| GLUE | activities | **4/1/3** | 0.5714 / 0.5000 | 0/1/2 |
| GLUE | counts | **2/1/10** | 0.3333 / 0.0476 | 0/0/2 |
| scMRDR | activities | **5/1/2** | 0.3333 / 0.2500 | 0/2/2 |
| scMRDR | counts | **4/2/8** | 0.6667 / 0.1905 | 0/0/2 |

All global/per-assay adjacency scores, exact TAIL/ARROW/CIRCLE endpoint scores, sentinel rates and directed/pair stress metrics were recomputed from saved graphs. The biological chain is H3K4me3→H3K27ac→RNA. Its two direct edges define a separately labeled stress test. The nine-activity source projection instead has **8 adjacencies, 7 cross-assay**, with circle endpoints; the count measurement PAG has **21 adjacencies, 12 cross-assay**, including latent assay-exposure effects. Matching source DAG arrows, matching projection endpoints and matching the count PAG are different tasks.

Report status is `completed`, with no failed integrations or diagnostics. All learned global graphs are nonempty. In particular, the random activity control has zero cross-assay edges but three within-assay edges; it is not an empty or failed graph. Main activity graphs have no sentinel-incident edges. Count graphs have one GLUE and two scMRDR sentinel-incident edges: incident rates are 1/21 and 2/21, while false-positive rates against the count target are 1/15 and 2/15. Sentinel counts can have genuine within-assay exposure-induced adjacencies, unlike independent sentinel activities.

The common discovery pools contain **792 cells per assay**, with unique, hash-verified JSON ID lists and valid assignment permutations. The paired-anchor reference uses those same 792 RNA anchor cells' complete same-cell truth panels; matched panels instead combine assigned-assay observations from different cells. Independent paired and state-observed activity controls use **2,000 assessment cells**.

| Reference/control | Activity cross TP/FP/FN | Count cross TP/FP/FN |
|---|---|---|
| Independent paired assessment, n=2,000 | 6/0/1 | 6/9/6 |
| Paired anchor, n=792 | 5/0/2 | 1/5/11 |
| True-state matching, n=792 | 7/1/0 | 2/1/10 |
| Random matching, n=792 | 0/0/7 | 2/2/10 |

The causally sufficient state-observed activity PC control returns 8/0/2 adjacencies against its 10-edge source oracle, with endpoint precision/recall 0.75/0.60 and zero correctly returned biological core directions. It is distinct from the nine-node marginal activity projection. Paired count controls already show substantial errors before learned matching, so count failures cannot be attributed solely to integration. True-state matching remains an imposed different-cell coupling and does not restore the original residual joint.

Saved raw-state mean Euclidean distances, whose accounting was checked without opening arrays, are:

| Assignment | RNA–H3K27ac | RNA–H3K4me3 | H3K27ac–H3K4me3 |
|---|---:|---:|---:|
| GLUE | 0.87126 | 0.70479 | 0.97116 |
| scMRDR | 0.65763 | 0.48834 | 0.66620 |
| True-state reference | 0.16398 | 0.18756 | 0.20541 |
| Random reference | 1.75939 | 1.76246 | 1.88543 |

Embedding-space cost sums equal 792 times their saved means; true-state RNA-to-mark means also equal the corresponding saved raw-state distances. These original Euclidean diagnostics are not the newer normalized squared-state validation gaps. Actual distances, matching optimality and readout values were not recomputed from representations or truth arrays. Raw embedding cost magnitudes across methods have different coordinate units.

Pure graph traversal verifies all **1,333** saved exact-oracle separation labels against hash-bound biological/measurement DAG metadata. All **1,562** saved statistical CI log entries have valid status/p-value ranges and deterministic query seeds, with no recorded warnings. No statistical CI test or PC/FCI run was executed. Saved target graph generation/orientation was not rerun; graph scores are checked against the saved targets. Matched-panel source-target agreement and nominal p-values do not constitute calibrated tests of the original paired joint. All 36 saved count-calibration comparison inequalities also reproduce.

Saved retention probes are successful computations: 120 GLUE rows, 240 scMRDR rows and 120 preprocessing rows. GLUE RNA retains 816 calibration/1,623 assessment cells; marks retain 1,000/2,000. scMRDR retains 1,000/2,000 for every assay. Shared ridge state R² ranges are approximately 0.723–0.933 for GLUE and 0.808–0.943 for scMRDR. These probes fit each source assay on private calibration cells and score same-cell state/activity targets on independent assessment cells. They are not the new study's fixed RNA-decoder transfer across assay coordinate systems and do not validate matching. Conditional residual targets use 128-repetition ancestral Monte Carlo means; restricted ridge/tree incremental gains are not exact CI tests. Their saved MSE-gain algebra passes, but probe fits and private target values were not reproduced.

Only this realistic stage's report, input manifest, stage JSONs, completed model/export metadata, frozen source/configuration metadata and truth-graph metadata were read. Five readout NPZ files were inventoried by filename/size only; no readout, cohort, representation or checkpoint array was opened. No fitting, CI rerun, tuning, causal rerun, model loading, GPU use or production edit occurred. Active informative outputs and the new integration-validation selection were untouched. These old scientific results do not select new settings or certify empirical causal recovery.
