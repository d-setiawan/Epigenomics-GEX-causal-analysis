# Representative confirmation diagnostics: constrained chain, realistic counts, seed 0

Date: 2026-10-25. Passed the independent public-data check of the completed scMRDR `baseline` confirmation and representative end-to-end diagnostics for it and completed GLUE `align_high`. Scope is only `constrained_chain_0`, realistic counts, public training/discovery cohorts, seed 0. No seed 1 output was accessed. The [numerical receipt](constrained_chain_0_realistic_seed0_representative_diagnostics_crosscheck.json) contains the recomputed values, formulas, tolerances, public ID hashes, and source/config/report bindings.

The scMRDR state transport rows reproduce as follows. The maximum discrepancy from saved transport values across both methods is `1.3322676295501878e-15`.

| Assay pair | Matched squared cost | Oracle squared cost | Random expected squared cost | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 0.550127421892745 | 0.043393842019570 | 3.815879339558831 | **0.134323532907869** |
| RNA--H3K4me3 | 0.326500154515340 | 0.051552743071172 | 3.836235325930271 | 0.072647416374998 |
| H3K27ac--H3K4me3 | 0.538520185862681 | 0.040859027182645 | 3.808866570767298 | 0.132075414638526 |

The scMRDR worst-pair gap is `0.13432353290786944`, agreeing with the saved `0.1343235329078695`. GLUE seed 0 is independently rechecked at `0.34076001636866726`. Both use identical public pools: 1,614 training and 766 discovery cells per assay, constructed from selected-fit eligibility, both cached baseline-reference ID sets, and public assay membership with seed 2718. Train/discovery IDs are disjoint. The reproduced common training center is `[0.015363853270592968, 0.013271219663996041]`; scale is `[1.0036465736682494, 1.0026490522725928]`. State oracle assignments minimize squared distance in this metric; the exact random expectation is also checked against the full squared-cost matrix mean.

For each method, raw shared embedding rows were joined to the common pool by public IDs. Independently constructed unsquared Euclidean distance matrices and Hungarian assignments reproduce both RNA-to-mark target-ID permutations exactly, including saved assignment cost sums. This confirms the frozen embedding-space matching objective separately from the squared-state oracle objective.

All nine fixed source-to-target probe scores per method reproduce at absolute tolerance `1e-12`. Rows are source training assays; columns below identify target discovery assays. Each source has its own training-fitted feature StandardScaler and Ridge with `alpha=1`; both are transferred unchanged to every target.

| Source → target | GLUE mean state R² | scMRDR mean state R² |
|---|---:|---:|
| RNA → RNA | 0.923068830379 | 0.921458142168 |
| RNA → H3K27ac | 0.662651851246 | 0.788659594978 |
| RNA → H3K4me3 | 0.778613100771 | 0.907256537896 |
| H3K27ac → RNA | 0.892654267072 | 0.816903477871 |
| H3K27ac → H3K27ac | 0.719418174868 | 0.819771127147 |
| H3K27ac → H3K4me3 | 0.780890890214 | 0.878449062975 |
| H3K4me3 → RNA | 0.901039181094 | 0.907309222975 |
| H3K4me3 → H3K27ac | 0.693784953643 | 0.772585226166 |
| H3K4me3 → H3K4me3 | 0.834924307089 | 0.943185164946 |

Production probe targets are raw state coordinates. Per-coordinate R² is invariant to a common nonzero affine scaling of the state target when the linear Ridge fit includes a free intercept. A separate float64 calculation solves the Ridge normal equations on the same training-scaled states and independently standardized source features, then predicts each target without refitting. Its maximum coordinate-R² differences from the native float32-library path are `1.407051166912865e-6` for GLUE and `1.9827327657484517e-7` for scMRDR. No target state variance is used for fitting or scaling; target means enter only the declared R² denominator.

The inexpensive remaining diagnostics were also independently reproduced:

- All 12 directed cross-assay hubness rows: direct stable 15-neighbor rankings, target indegrees including zeros, Gini from pairwise absolute indegree differences, coverage, and largest indegree divided by all neighbor edges. Values match at absolute tolerance `1e-15`.
- All six within-assay mean kNN Jaccards: direct public discovery PCA/LSI distances and shared embedding distances, with self excluded by row identity and Boolean neighbor-set intersections/unions. Values match at absolute tolerance `1e-15`.
- All shared coordinate variances: centered population second moments in float64 agree with native float32 reports within relative error `9.98e-7`; the native dtype variance values reproduce exactly. These coordinate summaries depend on the embedding basis.

| Assay | GLUE mean kNN Jaccard | scMRDR mean kNN Jaccard |
|---|---:|---:|
| RNA | 0.160851623536 | 0.078044040305 |
| H3K27ac | 0.123717048978 | 0.038229089405 |
| H3K4me3 | 0.174657869200 | 0.076526485281 |

The audit imports no production metric, matching, pool, or probe functions. It uses NumPy/SciPy, direct mathematical definitions, the declared sklearn estimators for exact probe reproduction, and a separate normal-equation probe check. Everything ran with one BLAS thread on CPU. The only refits were the six small, fixed source Ridge probes requested for verification; no integration/model training or tuning occurred.

Both reports carry `role=confirmation`, `cohort=discovery`, `mode=selected`, and only their frozen selected candidate. Their configurations, public files, metadata seals, protocol/source hashes, and frozen plan membership passed the bounded checks. The frozen selection SHA256 remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`, and confirmation-plan SHA256 remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`. The scMRDR report SHA256 is `4775a41e7eab9f7a44cea02eb17a311781ba5eb060f78274a594242a3cd296ad`; the GLUE report remains `5a3f92ecd62d6771c17f7970820d29d15cb63e5becd35984fdad904c872975a8`.

Access was limited to public training/discovery IDs and states, completed selected-fit shared embeddings, cached reference IDs, the discovery reference files' observation IDs and 50-column PCA/LSI arrays, and provenance metadata. H5AD count matrices, hidden activities/residuals, private/probe/assessment cohorts, checkpoint binaries, active fits, and causal scores were not opened. Native checkpoint/export validation is a separate audit. No selection, source, plan, or production output changed. These results describe two methods on one world/regime/seed; they do not identify cross-assay residual coupling, establish independent biological-world replication or population generalization, or certify a causal graph.
