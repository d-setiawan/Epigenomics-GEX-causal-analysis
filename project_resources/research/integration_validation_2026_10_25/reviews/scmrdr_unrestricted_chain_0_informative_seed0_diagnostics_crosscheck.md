# Unrestricted-chain informative scMRDR seed 0: reproduced diagnostics

The completed selected `scmrdr/baseline` public discovery report reproduces independently for all three state-cost rows, embedding assignments, six cross-assay fixed state probes and six directed hubness rows. The [numerical receipt](scmrdr_unrestricted_chain_0_informative_seed0_diagnostics_crosscheck.json) records calculations, tolerances, public-ID hashes and provenance. The state gap is low, and **none of the checked cross-assay mean or coordinate R² values is negative**. Weaker RNA-to-mark transfer and concentrated local neighbor use nevertheless coexist with this low gap.

| Assay pair | Matched squared state cost | Squared-state oracle | Exact random expectation | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 0.136699799403764 | 0.028324344196245 | 4.056767918791008 | **0.026902562540775** |
| RNA--H3K4me3 | 0.120939303122825 | 0.028429742252926 | 4.126614923531927 | 0.022573299345402 |
| H3K27ac--H3K4me3 | 0.086441291165111 | 0.034776821543999 | 4.150873184992671 | 0.012551812459955 |

The independently recomputed maximum is `0.026902562540774975`, versus saved `0.026902562540774964`. Transport values agree within `1.7763568394002505e-15`. Direct squared-distance Hungarian optimization supplies the state oracle; the exact random expectation from state moments also matches the full squared-cost matrix mean. One-Lipschitz squared-error bounds equal the absolute matched squared costs.

Outcome-free reconstruction yields 1,931 training and 957 discovery cells per assay, using public selected-fit IDs, both cached baseline-reference ID sets, public assay membership and pool seed 2718. Training/discovery IDs are disjoint. All six selected-ID hashes, counts and state scaling exactly match both completed informative GLUE seeds. The pooled-training center is `[-0.0011441230541105328, 0.0004934294164940117]`; scale is `[0.9908636842821651, 0.9944648203855664]`. Discovery state labels do not enter eligibility or scaling. Direct Euclidean embedding-space distances and Hungarian optimization reproduce both saved RNA-to-mark target-ID permutations exactly, with public rows aligned by IDs. This is separate from squared-state oracle optimization.

| Source → target | Native mean state R² | Target k15 coverage | Target indegree Gini |
|---|---:|---:|---:|
| RNA → H3K27ac | 0.432280472381195 | 0.309299895506792 | 0.850872287171066 |
| RNA → H3K4me3 | 0.173004714690565 | 0.354231974921630 | 0.845225941539853 |
| H3K27ac → RNA | 0.758014308327392 | 0.521421107628004 | 0.726664621205752 |
| H3K27ac → H3K4me3 | 0.982782097984153 | 0.991640543364681 | 0.285066351913179 |
| H3K4me3 → RNA | 0.888923496154721 | 0.543364681295716 | 0.703932489598904 |
| H3K4me3 → H3K27ac | 0.970034991989505 | 1.000000000000000 | 0.223142315672853 |

Each source's public training data fits its own feature StandardScaler and fixed Ridge `alpha=1`, transferred unchanged to both other discovery assays. Native per-coordinate and mean R² reproduce at `1e-12`. The weakest mean is RNA→H3K4me3, `0.17300471469056522`; its coordinate scores are `[0.19309790341005495, 0.1529115259710755]`. The minimum k15 coverage, `0.30929989550679204`, is RNA→H3K27ac. Direct stable neighbor rankings and independently calculated pairwise-difference indegree Gini, coverage and largest indegree fractions all reproduce at `1e-15`. No new probe or coverage acceptance threshold was introduced.

The supplemental float64 normal-equation path differs from native float32 by at most `0.00019688571408083` in coordinate R² (RNA→H3K4me3), exceeding the recorded `5e-5` supplemental comparison tolerance. It remains explicitly separate from exact native-report reproduction. All float64 coordinate scores are also positive; the weakest is approximately `0.15271464`. Raw and common-training-affine-scaled state targets give equivalent R² within float64 at `1e-10`. No declared metric was replaced.

The report SHA256 is `eb799ee8a88433df0ecc62941a7669eea4fbdb2e09dd14e0d7b99d3e452dd721`. Selected-only/discovery labels, exact frozen baseline configuration, source/protocol metadata, completed-fit/export/study bindings, public file hashes and frozen-plan membership pass. Report versions match the auditor environment. Recorded runtime overrides match the prior scMRDR launch setup: launch blocking `1`, the declared CUDA-runtime preload, and unset `CUDA_VISIBLE_DEVICES`/`LD_LIBRARY_PATH`. Frozen selection SHA256 remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`; confirmation-plan SHA256 remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

No production pool, metric, matching or probe function was imported. Access was restricted to public train/discovery IDs, assay identities and states, this completed fit's shared embeddings, cached reference IDs, and the three completed reports' provenance/pool metadata. Active scMRDR seed 1 was not read. There was no hidden biology/private cohort/checkpoint/H5AD array access, CUDA use, integration training, new dataset, tuning, retry or production edit. Only three fixed small diagnostic Ridge probes were refitted, on one BLAS thread. Native checkpoint verification is separate; neighbor retention and coordinate variances were not repeated.

This single condition has no negative-R² cross-decoder failure, while RNA-source transfer remains weaker than the mark-source transfers despite near-oracle-relative state transport. Those measurements do not identify their training mechanism, residual coupling, population generalization, empirical biological correctness or causal validity.
