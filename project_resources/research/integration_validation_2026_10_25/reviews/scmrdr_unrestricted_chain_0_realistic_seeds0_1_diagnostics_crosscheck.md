# Unrestricted-chain realistic scMRDR confirmations: reproduced seed-1 failure

Independent public-array review of completed selected `scmrdr/baseline` confirmations, seeds 0 and 1. Numerical reproduction passes; the seed-1 scientific outcome remains poor. Its worst gap is `1.4446291191821001`, and both RNA-to-mark matched state costs exceed the exact same-pool uniform-random-permutation expectation. This result is preserved. No metric, setting, fit, or production output was changed.

The [numerical receipt](scmrdr_unrestricted_chain_0_realistic_seeds0_1_diagnostics_crosscheck.json) records all calculations and provenance checks. All six pairwise state rows reproduce within `1.7763568394002505e-15`.

| Seed | Assay pair | Matched squared cost | Squared-state oracle | Exact random expectation | Normalized gap |
|---|---|---:|---:|---:|---:|
| 0 | RNA--H3K27ac | 0.514147141710237 | 0.038197658559855 | 4.003147287673899 | 0.120039225632415 |
| 0 | RNA--H3K4me3 | 0.272820212876328 | 0.035534760737365 | 4.148637543770377 | 0.057690134347673 |
| 0 | H3K27ac--H3K4me3 | 0.554037673907186 | 0.040841262974941 | 4.158295496193707 | **0.124639250824425** |
| 1 | RNA--H3K27ac | 5.597106214368602 | 0.038197658559855 | 4.003147287673899 | 1.402012402626882 |
| 1 | RNA--H3K4me3 | 5.977442811295790 | 0.035534760737365 | 4.148637543770377 | **1.444629119182100** |
| 1 | H3K27ac--H3K4me3 | 2.272223059526467 | 0.040841262974941 | 4.158295496193707 | 0.541932385926527 |

The recomputed maxima are `0.12463925082442502` and `1.4446291191821001`, matching saved `0.124639250824425` and `1.4446291191821`. The gap `(matched-oracle)/(random-oracle)` is not a probability or a score capped at one. All reference denominators here exceed 3.96; there is no empty or near-zero normalization. With a positive denominator, a gap above one means exactly that the matched cost exceeds the random expectation. It does not mean that every possible random permutation performs better.

Both methods' four realistic unrestricted-chain reports have exactly the same six selected-ID hashes, pool counts, common training state center and scale. Each has 1,567 training and 756 discovery cells per assay, with disjoint training/discovery IDs. The center is `[-0.013880246927312139, 0.0003396350377540681]`, scale `[0.9913896719160854, 0.9969791425859608]`. Each scMRDR pool was reconstructed independently from the completed fit's IDs, both cached baseline-reference ID sets, public assay membership and seed-2718 equalization. Discovery state labels do not determine pools or scaling.

For both seeds, direct Euclidean distances and Hungarian optimization reproduce the saved embedding-space RNA-to-mark target-ID permutations exactly, including assignment objectives. Direct squared state distances produce the independent state oracles. Exact random expectations from state moments also match full squared-distance matrix means. Thus the poor score is attached to the actual frozen embedding assignment, rather than a wrong public-ID join, different comparison pool, or confused Euclidean-versus-squared oracle.

All 12 cross-assay native probe rows (six per seed, 24 coordinate R² values) reproduce at absolute tolerance `1e-12`. Each source independently fits its declared feature StandardScaler and fixed Ridge `alpha=1` on public source training data, then predicts the other assays unchanged.

| Source → target | Seed 0 mean state R² | Seed 1 mean state R² |
|---|---:|---:|
| RNA → H3K27ac | 0.792524282347459 | -1.292582782618814 |
| RNA → H3K4me3 | 0.918960567478226 | -1.875607514563793 |
| H3K27ac → RNA | 0.881417105304569 | -1.840537089800925 |
| H3K27ac → H3K4me3 | 0.921270327366343 | 0.503432316503134 |
| H3K4me3 → RNA | 0.925121462752206 | -1.747031586207513 |
| H3K4me3 → H3K27ac | 0.798084040158088 | 0.329219154187475 |

The separate float64 normal-equation probe check differs from native float32 coordinate R² by at most `2.538713543276394e-6` in seed 0 and `8.411349458725681e-6` in seed 1, within the recorded `5e-5` supplemental tolerance. Native-path reproduction and normal-equation agreement remain separately reported. Affine state-unit scaling is checked within float64; no target refitting occurs.

All 12 directed hubness rows reproduce at `1e-15`, including Gini from pairwise absolute target-indegree differences, zero-indegree targets, coverage and largest indegree divided by all neighbor edges. Minimum k15 target coverage is `0.9695767195767195` in seed 0 and `0.7724867724867724` in seed 1. These coverage values do not certify a correct shared-state correspondence.

Two exploratory decompositions describe the poor run without replacing the declared metric. In seed 1's RNA--H3K4me3 matching, coordinate-wise squared state costs are `[2.181573216502821, 3.7958695947929657]`, versus random components `[2.1166363361654725, 2.032001207604904]`. Matched same-coordinate correlations are approximately `[-0.03070, -0.86988]`; for RNA--H3K27ac they are `[-0.01605, -0.79882]`. These are observations about the resulting public-state assignments, not an explanation of the training mechanism. The receipt's oracle coordinate components decompose the joint squared-cost optimum; they are not separately optimized one-dimensional oracles.

Independently computed discovery shared-covariance spectra also show strong variance concentration in seed 1:

| Assay | Seed 0 top covariance-eigenvalue fraction | Seed 1 top fraction | Seed 1 covariance participation ratio |
|---|---:|---:|---:|
| RNA | 0.51619 | 0.98344 | 1.03386 |
| H3K27ac | 0.52521 | 0.97671 | 1.04802 |
| H3K4me3 | 0.53690 | 0.98029 | 1.04042 |

Training embeddings show the same descriptive concentration (seed-1 top fractions 0.97780–0.98460). This supports the near-one-dimensional description in these embedding units, while it does not by itself establish the cause of the state mismatch. Spectrum and coordinate summaries are explicitly exploratory and were not used to change selection, retry the fit, or rescore the declared outcome.

Both reports have selected-only/discovery labels and complete public-export seals. Exact configurations match the frozen baseline defaults with their respective seeds; source/protocol, model/export/study hashes and frozen-plan membership pass. Report versions match the auditor environment. Both studies record the same runtime overrides: launch blocking `1`, the previously declared CUDA-runtime preload, and unset `CUDA_VISIBLE_DEVICES` and `LD_LIBRARY_PATH`. No mismatch was found in this checked metadata. Report SHA256 values are `35dec379cd621cfd3eaaf83adeedbd3c1bd4536bf73179f312303348f4b11578` (seed 0) and `a3a96ca984040efbf0f388a5a7efc0e1826ff29e037ce6b1892c7068379679fe` (seed 1). Frozen selection SHA256 remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`; confirmation-plan SHA256 remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

No production metric, matching, pool or probe function was imported. Only public train/discovery IDs, assay identities, states and these completed fits' shared embeddings were decompressed; cached references supplied IDs only. Four completed realistic reports supplied pool-comparison metadata. There was no active informative-fit access, hidden biology/private cohort/checkpoint/H5AD array access, CUDA call, integration training, new dataset, tuning, retry, archive action, or production edit. Only six small fixed source Ridge probes were refitted; native checkpoint/export verification remains a separate audit. Neighbor retention and full coordinate-variance reports were not repeated.

The narrow conclusion is a reproduced adverse integration-validation outcome for scMRDR baseline seed 1 on this existing world/regime, alongside materially better seed 0. It is not a statement about every scMRDR run, a mechanistic diagnosis, fresh-world replication, residual coupling, or causal validity.
