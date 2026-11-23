# Informative constrained-chain scMRDR seed 0: state, transfer, and hubness

Date: 2026-10-25. The completed selected `scmrdr/baseline` public discovery report reproduces independently: all three state transport rows, all six cross-assay native StandardScaler+Ridge scores, and all six directed hubness rows. The [numerical receipt](scmrdr_constrained_chain_0_informative_seed0_diagnostics_crosscheck.json) retains required report reproduction separately from a supplemental float64 probe calculation. No production metric, matching, pool, or probe function was imported.

| Assay pair | Matched squared state cost | Squared-state oracle | Random expected cost | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 0.172395464222623 | 0.031759375915185 | 3.870154758153636 | **0.036639291761919** |
| RNA--H3K4me3 | 0.167284471840020 | 0.044480688321602 | 3.872169607725827 | 0.032083010428531 |
| H3K27ac--H3K4me3 | 0.089041920258863 | 0.032614757346268 | 3.828595786743402 | 0.014864974949982 |

The recomputed maximum is `0.036639291761919233`; all transport values agree with the report within `1.7763568394002505e-15`. Outcome-free reconstruction reproduces 1,942 training and 955 discovery IDs per assay, disjoint across cohorts. The common training center is `[-0.0027020121334687883, 0.015548173476410607]`, scale `[0.9941825329310068, 0.9981614176535301]`. Direct squared-distance Hungarian state oracles and exact random expectations reproduce. An additional direct Euclidean embedding-space Hungarian calculation reproduces both saved RNA-to-mark ID permutations exactly.

| Source → target | Native mean state R² | Target k15 coverage | Target indegree Gini |
|---|---:|---:|---:|
| RNA → H3K27ac | -3.408289577636174 | 0.473298429319372 | 0.761911570406513 |
| RNA → H3K4me3 | -3.512707260372561 | 0.370680628272251 | 0.828262529353179 |
| H3K27ac → RNA | 0.857950950374814 | 0.396858638743455 | 0.823197902104292 |
| H3K27ac → H3K4me3 | 0.986766382077447 | 0.997905759162304 | 0.223036576117248 |
| H3K4me3 → RNA | 0.805563867085758 | 0.400000000000000 | 0.811022504865547 |
| H3K4me3 → H3K27ac | 0.967028531753481 | 1.000000000000000 | 0.182740458503513 |

The two negative RNA-transfer means reproduce exactly through an independently written native-library path: each source's public training shared coordinates fit one feature StandardScaler and fixed Ridge `alpha=1`, then transfer unchanged to the other assays. Coordinate R² values and means agree at `1e-12`. Hubness uses directly constructed distances and stable k15 rankings before assignment; Gini is independently calculated from pairwise absolute indegree differences, including zero-indegree targets. All hubness quantities, including largest indegree fractions, agree at `1e-15`.

A supplemental float64 normal-equation calculation is **not numerically identical** to the native float32 path. Its largest coordinate-R² discrepancy is `0.001670942677289844`, for RNA→H3K27ac; RNA→H3K4me3 differs by up to `0.001646840162969454`. These exceed the earlier representative check's `5e-5` comparison tolerance and remain explicitly flagged in the receipt. Both negative conclusions persist: the float64 RNA-transfer means are approximately -3.40872 and -3.51295. Raw-state and common-training-affine-scaled-state targets give the same R² within the float64 calculation at `1e-10`. This distinguishes dtype/solver-path sensitivity from a saved-report mismatch or an effect of state-unit scaling. The supplemental calculation does not replace the declared native probe or change selection.

The report SHA256 is `0a45a08e0771659384ea1cd6f2f2d45f73db0a9423e8590528c251be25660ee0`. Selected-only/discovery labels, configuration, source/protocol metadata, export seals, public file hashes, and membership in the unchanged frozen confirmation plan pass the bounded checks. Frozen selection SHA256 remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`; plan SHA256 remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

Access was confined to public training/discovery IDs, assay identities and states, this completed fit's shared embeddings, cached reference IDs, and provenance metadata. There was no hidden biology/private cohort/checkpoint/H5AD array access, integration training, CUDA use, tuning, or production edit. Only three fixed diagnostic Ridge probes were refitted. Other fits, including seed 1, were not read during this seed-0 check. Neighbor retention and coordinate variance were not repeated.

The proposed analytic illustration is valid with the following precise scope; no data were generated and no model was fitted for it. Let scalar state pools have uniform marginals on `[-1,1]`, source embedding `(s,0)` and target embedding `(3t,c)` with `c>0`. For `C(s,t)=sqrt((s-3t)^2+c^2)`,

\[
\frac{\partial^2 C}{\partial s\partial t}
=\frac{-3c^2}{((s-3t)^2+c^2)^{3/2}}<0.
\]

Integrating this derivative over any ordered rectangle gives a strict uncrossing inequality. Thus equal-size finite pools with distinct source states and distinct target states have the unique sorted/rank Euclidean assignment. Sorted matching also minimizes squared scalar-state cost, so `C_theta=C_star`; its normalized gap is zero when the random-minus-oracle denominator is positive. For `n=1` or degenerate pools the gap is unavailable. Finite oracle cost itself need not be zero; ties can remove uniqueness, and `c=0` removes strict convexity.

The ideal source decoder `d(z)=z1` predicts `3T` on the target. Its **population** target R² is `1-E[(3T-T)^2]/Var(T)=-3`. For a finite target sample with nonzero variance, its R² is `1-4 sum(t_i^2)/sum((t_i-mean(t))^2)`, which equals -3 only when the sample mean is zero and otherwise is smaller. A finite standardized Ridge includes shrinkage and an estimated intercept, so its score is not exactly this ideal-decoder value.

Directed target nearest-neighbor lookup from source queries minimizes `|s-3t|`; the fixed vertical offset does not change neighbor ranks. As equal source/target pool sizes grow with fixed k and dense uniform support, selected target states are confined asymptotically to `[-1/3,1/3]`, up to vanishing boundary effects. Coverage has asymptotic upper bound one third; some interior targets can remain unused, so equality is not asserted. Finite pools, gaps, and tie rules require qualification. A global one-to-one assignment instead uses every target by construction.

This is an existence example showing that oracle-relative state matching, fixed source-decoder transfer, and directed neighbor coverage can differ. It does not establish the cause of the actual 20-dimensional scMRDR behavior, identify state ordering from unpaired marginals, certify residual coupling, or justify changing the frozen metrics or selection rule. The empirical results remain one existing world/regime/seed, with no causal-validity claim.
