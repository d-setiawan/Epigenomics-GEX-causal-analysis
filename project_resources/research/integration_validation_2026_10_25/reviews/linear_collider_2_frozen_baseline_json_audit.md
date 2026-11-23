# Frozen baseline audit: linear_collider_2

_Independent JSON-only review · 2026-10-25 · primary seed 0, both count regimes_

Both legacy reports pass this scoped arithmetic and metadata audit. They remain outcomes of the old frozen assessment suite. They do not select candidates or alter the running integration-validation study.

The audit independently canonicalized the saved graph edges and recomputed adjacency TP/FP/FN, precision, recall, F1, false-positive rates, exact endpoint agreement, and biological directed-edge stress scores. It compared activity graphs with the saved activity-projection oracle and count graphs with the saved count-measurement PAG. Hash-verified `truth/manifest.json` supplied the biological DAG edge list for the separately labeled direct-edge stress test; no hidden truth/cohort arrays were read. The two targets contain 10 and 12 cross-assay adjacencies, respectively, rather than just the two planted direct cross-assay edges.

Each table cell reports **TP / FP / FN**. Directed columns are the biological direct-edge stress test, not a required identifiable target or evidence that the arrows can be learned uniquely.

| Method / regime | Activity cross adjacency | Count cross adjacency | Directed activity stress | Directed count stress |
| --- | ---: | ---: | ---: | ---: |
| scGLUE / realistic | 3 / 0 / 7 | 4 / 5 / 8 | 1 / 0 / 1 | 0 / 0 / 2 |
| scMRDR / realistic | 4 / 1 / 6 | 2 / 3 / 10 | 0 / 0 / 2 | 0 / 0 / 2 |
| scGLUE / informative | 4 / 0 / 6 | 2 / 0 / 10 | 0 / 3 / 2 | 0 / 0 / 2 |
| scMRDR / informative | 6 / 1 / 4 | 1 / 0 / 11 | 0 / 5 / 2 | 0 / 0 / 2 |

Exact cross-assay endpoint marks also reproduce. For activity/count graphs, the numbers of correct marks over returned marks are scGLUE realistic `4/6` and `8/18`, scMRDR realistic `8/10` and `2/10`, scGLUE informative `2/8` and `3/4`, and scMRDR informative `0/14` and `0/2`. Endpoint recall additionally penalizes missing oracle adjacencies; its denominators are 20 activity marks and 24 count marks. All original whole-panel endpoint scores agree with the independent calculation. No graph in these reports is a failed computation or an empty whole-panel graph. Zero returned directed edges instead gives undefined directed precision and zero directed recall against the two-edge target.

Both reports are `completed`, with passed declared count calibration and empty integration/diagnostic failure lists. Their report hashes, input fingerprints, input-manifest copies, world/truth-manifest bindings, 17 frozen source pins, model and export JSON manifests, checkpoint-digest references, and discovery-export digest references agree. Three successful stage receipts bind world generation and the two regime pipelines; their saved artifact hashes still match the report and model/export metadata. Checks also link scGLUE preprocessing/training/export metadata and scMRDR training/validation input hashes back to the frozen world metadata. The common retained pools contain 775 cells per assay in realistic counts and 960 in informative counts. Actual binary matrices and checkpoints were not rehashed or opened.

The paired controls limit attribution. Independent paired activities recover 7/10 cross adjacencies and both planted directed edges in each regime. Independent paired counts recover only 1/12 cross adjacencies in realistic counts and return zero cross-assay adjacencies in informative counts, despite a nonempty whole-panel graph. The informative matched count graphs' precision of one therefore describes one or two returned edges with low recall; it is not reliable network recovery. Better activity adjacency agreement in scMRDR's informative arm coexists with five incorrect directed stress edges and zero exact cross-endpoint agreement.

This supports a narrow conclusion: these particular fitted integrations and fixed pseudo-pairing/readout do not establish reliable cross-assay count-graph recovery in this world. It does not isolate integration as the sole cause, establish unique direction recovery, or prove that all possible readouts fail. The regimes are repeated conditions of one world with different retained pools, not independent biological replicates. The new study's settings and rule stay frozen.

The machine-readable [audit receipt](linear_collider_2_frozen_baseline_json_audit.json) includes all recomputed method/control scores and checked metadata hashes. The [method × regime score table](linear_collider_2_method_regime_scores.csv) retains the layer-specific numerators and endpoint counts. Audited reports: [realistic](../../../../pipelines/simulation/integration_recoverability/runs/extended/linear_collider_2/realistic/primary/evaluation_scglue_scmrdr_seeds0/report.json) and [informative](../../../../pipelines/simulation/integration_recoverability/runs/extended/linear_collider_2/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json).

No scientific tests, fits, CUDA calls, checkpoint loads, or NPZ/H5AD reads were performed. Oracle generation and the unrecorded interval before the first export receipts were not retrospectively revalidated.
