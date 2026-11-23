# GLUE unrestricted-chain informative confirmation: seeds 0 and 1

Both completed selected `scglue/align_high` public discovery reports reproduce independently. All six transport rows and both maxima agree within `2.6645352591003757e-15`. The [combined numerical receipt](scglue_unrestricted_chain_0_informative_confirmation_public_state_crosscheck.json) binds the two per-seed numerical receipts and records their exact common pools and scale.

| Seed | Assay pair | Matched squared cost | Squared-state oracle | Exact random expectation | Normalized gap |
|---|---|---:|---:|---:|---:|
| 0 | RNA--H3K27ac | 0.528458074664627 | 0.028324344196245 | 4.056767918791008 | 0.124150610827084 |
| 0 | RNA--H3K4me3 | 0.521286715866378 | 0.028429742252926 | 4.126614923531927 | 0.120262250682298 |
| 0 | H3K27ac--H3K4me3 | 0.626685330285153 | 0.034776821543999 | 4.150873184992671 | **0.143803365246100** |
| 1 | RNA--H3K27ac | 0.525877687715806 | 0.028324344196245 | 4.056767918791008 | 0.123510068915291 |
| 1 | RNA--H3K4me3 | 0.568538568949498 | 0.028429742252926 | 4.126614923531927 | 0.131792196498063 |
| 1 | H3K27ac--H3K4me3 | 0.638061192564608 | 0.034776821543999 | 4.150873184992672 | **0.146567115478110** |

Recomputed maxima are `0.1438033652461001` and `0.14656711547810972`, matching saved `0.14380336524610007` and `0.1465671154781096`. The worst pair is H3K27ac--H3K4me3 in both seeds. One-Lipschitz squared-error bounds equal the absolute matched squared costs.

Outcome-free reconstruction reproduces 1,931 training and 957 discovery cells per assay, using selected-fit IDs, both cached baseline-reference ID sets, public assay membership and pool seed 2718. All six selected-ID hashes, counts, training center and scale are exactly equal across seeds. Training/discovery IDs are disjoint. The pooled-training center is `[-0.0011441230541105328, 0.0004934294164940117]`; scale is `[0.9908636842821651, 0.9944648203855664]`. Discovery labels do not determine eligibility, equalization or scale.

Saved assignment IDs are valid permutations of those discovery pools with the declared RNA anchor order. Their state costs were recomputed directly in the common training-scaled metric. State oracle costs use squared-distance Hungarian optimization, and exact random expectations from state moments are independently checked against full squared-distance matrix means. Pairwise oracle floors are not asserted to form a jointly attainable three-assay optimum.

Seed 1's final report was absent at the initial check and appeared before this review closed. Its final report plus complete model, export and study seals were verified before any seed-1 public arrays were read. Both reports have `role=confirmation`, `cohort=discovery`, `mode=selected`, `reference_only=false`, and only `scglue/align_high`. Exact configurations, completed-fit/export metadata, public file hashes, source/protocol pins and frozen-plan membership pass.

- Seed 0 report SHA256: `738ab4e2254b7b616e02d8e8b77c917aaebad6af0d2bed6919dda0fd5e8c0a22`.
- Seed 1 report SHA256: `74b96ff8ff6e4230bf09bd90197733331065b8c9603ba8615a05c989378ee4a6`.
- Frozen selection SHA256: `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`.
- Frozen confirmation-plan SHA256: `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

Only public training/discovery `cell_ids`, `assigned_assay` and `state` were decompressed from truth archives; only `cell_ids` from permitted public embedding archives. Numeric embeddings, hidden biology, private/probe/assessment cohorts, checkpoints, partial fit artifacts and active informative scMRDR fits were not accessed. No fitting, CUDA use, tuning, causal scoring or production/source/selection/plan edit occurred. Checkpoint digest references were checked between metadata without reading binaries; native checkpoint verification is separate. Embedding-space assignment optimization and secondary diagnostics were not repeated.

These two seeds are repeated conditions on one existing world/regime. The review establishes finite-pool numerical reproduction and bounded frozen-artifact bindings, without certifying residual coupling, population generalization, empirical biological correctness or causal structure.
