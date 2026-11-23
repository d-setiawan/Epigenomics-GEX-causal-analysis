# Final scMRDR confirmation: unrestricted-chain informative seed 1

The completed selected `scmrdr/baseline` public discovery report reproduces independently for all three state-cost rows, both embedding assignment permutations, six cross-assay fixed state probes and six directed hubness rows. The [numerical receipt](scmrdr_unrestricted_chain_0_informative_seed1_diagnostics_crosscheck.json) records calculations, public-ID hashes, frozen bindings and the four-report informative pool comparison.

| Assay pair | Matched squared state cost | Squared-state oracle | Exact random expectation | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 0.139374300538374 | 0.028324344196245 | 4.056767918791008 | **0.027566466871340** |
| RNA--H3K4me3 | 0.120383995869594 | 0.028429742252926 | 4.126614923531927 | 0.022437798574044 |
| H3K27ac--H3K4me3 | 0.082918602753952 | 0.034776821543999 | 4.150873184992671 | 0.011695980112967 |

The independently recomputed maximum is `0.027566466871340022`, versus saved `0.027566466871340012`; transport values agree within `1.7763568394002505e-15`. State oracles optimize direct squared distances, and exact random expectations from state moments also agree with full squared-cost matrix means. One-Lipschitz squared-error bounds equal absolute matched squared costs. Independent Euclidean embedding-space Hungarian assignments reproduce both saved RNA-to-mark target-ID permutations exactly.

Outcome-free reconstruction reproduces 1,931 training and 957 discovery cells per assay, with disjoint cohort IDs, from completed-fit IDs, both cached baseline-reference ID sets, public assay membership and seed-2718 equalization. All four informative unrestricted-chain reports—both selected methods, seeds 0 and 1—have exactly the same six selected-ID hashes, counts, training center and scale. The center is `[-0.0011441230541105328, 0.0004934294164940117]`, scale `[0.9908636842821651, 0.9944648203855664]`. Discovery state labels do not enter pools or scaling.

| Source → target | Native mean state R² | Target k15 coverage | Target indegree Gini |
|---|---:|---:|---:|
| RNA → H3K27ac | -1.929274587142653 | 0.319749216300940 | 0.847151004150248 |
| RNA → H3K4me3 | -1.959746330376534 | 0.377220480668757 | 0.826652719680501 |
| H3K27ac → RNA | 0.616267125553286 | 0.460815047021944 | 0.766274644255403 |
| H3K27ac → H3K4me3 | 0.975921732291170 | 0.994775339602926 | 0.273822868180235 |
| H3K4me3 → RNA | 0.431677921886468 | 0.475444096133751 | 0.744157315598241 |
| H3K4me3 → H3K27ac | 0.968283678620286 | 1.000000000000000 | 0.213205597574854 |

All six native mean/per-coordinate state-probe scores reproduce at `1e-12`. Each source fits its own feature StandardScaler and fixed Ridge `alpha=1` on public training data and transfers unchanged to the other discovery assays. Both coordinates are negative in both RNA-to-mark directions. All hubness quantities reproduce at `1e-15` from direct stable neighbor ranks and pairwise-difference target-indegree Gini, including zero-indegree targets and the all-neighbor-edge denominator for largest indegree fractions. Minimum coverage is `0.31974921630094044`.

The supplemental float64 normal-equation path differs from native float32 by at most `0.00046310883033617856` in coordinate R², exceeding the separately recorded `5e-5` comparison tolerance. Both RNA-to-mark coordinate scores remain negative in float64. Raw and common-training-affine-scaled state targets give equivalent R² within float64 at `1e-10`. Native report reproduction and supplemental precision sensitivity are distinct results; neither the frozen probe nor selection metric changed.

Report SHA256 is `8e7397603abd015e691d32c1e27e93851f964429f46841f351ed5403ed2d3bd1`. Selected-only/discovery labels, exact frozen baseline configuration, completed-fit/export/study metadata, public file hashes, source/protocol pins and frozen-plan membership pass. Report versions match the auditor environment. Recorded runtime overrides match the scMRDR launch setup. Frozen selection SHA256 remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`; plan SHA256 remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

No production pool, metric, matching or probe function was imported. Public train/discovery IDs, assay identities and states, this completed fit's shared embeddings, cached reference IDs and completed-report metadata were the only data used. There was no hidden biology/private cohort/checkpoint/H5AD array access, CUDA use, integration training, new dataset, tuning, retry or production edit. Only three fixed small diagnostic Ridge probes were refitted on one BLAS thread; native checkpoint verification is separate. Neighbor retention and coordinate variances were not repeated.

This condition again has low oracle-relative state-matching error together with failed fixed RNA-decoder transfer and concentrated local neighbor use. Those diagnostics test different properties. Their coexistence does not diagnose the training mechanism, identify residual coupling, establish population generalization or empirical biological correctness, or certify causal validity.
