# Informative constrained-chain scMRDR seed 1: numerical confirmation

Date: 2026-10-25. The completed selected `scmrdr/baseline` public discovery report reproduces independently for all three state transport rows, six cross-assay fixed native Ridge scores, and six directed k15 hubness rows. The [numerical receipt](scmrdr_constrained_chain_0_informative_seed1_diagnostics_crosscheck.json) records the required native-path reproduction separately from the supplemental float64 normal-equation calculation and the four-report pool comparison.

| Assay pair | Matched squared state cost | Squared-state oracle | Random expected cost | Normalized gap |
|---|---:|---:|---:|---:|
| RNA--H3K27ac | 0.201261589839530 | 0.031759375915185 | 3.870154758153636 | **0.044159654502683** |
| RNA--H3K4me3 | 0.201411625002742 | 0.044480688321602 | 3.872169607725827 | 0.040998874251663 |
| H3K27ac--H3K4me3 | 0.089930907975868 | 0.032614757346268 | 3.828595786743402 | 0.015099166772892 |

The independently recomputed maximum is `0.044159654502683304`, versus saved `0.04415965450268331`; transport values agree within `1.7763568394002505e-15`. Pool reconstruction uses public selected-fit and cached-reference IDs, public assay membership, and seed-2718 equalization. It yields 1,942 training and 955 discovery cells per assay, disjoint across cohorts. All four completed informative constrained-chain reports—both selected methods, seeds 0 and 1—have exactly the same six selected-ID hashes, selected counts, training center, and scale. The center is `[-0.0027020121334687883, 0.015548173476410607]`, scale `[0.9941825329310068, 0.9981614176535301]`.

Squared-state Hungarian oracles and exact random expectations were recomputed directly. Saved state-scored assignments are valid public-ID permutations, and an independent Euclidean embedding-space Hungarian calculation reproduces both RNA-to-mark ID permutations exactly. These are separate assignment objectives.

| Source → target | Native mean state R² | Target k15 coverage | Target indegree Gini |
|---|---:|---:|---:|
| RNA → H3K27ac | -10.167532541702673 | 0.522513089005236 | 0.731892071672012 |
| RNA → H3K4me3 | -12.639935812975404 | 0.360209424083770 | 0.852960536535000 |
| H3K27ac → RNA | 0.806655079468557 | 0.281675392670157 | 0.888076532989776 |
| H3K27ac → H3K4me3 | 0.977964029597181 | 0.996858638743455 | 0.232149922790859 |
| H3K4me3 → RNA | 0.568833123078103 | 0.281675392670157 | 0.877000520819057 |
| H3K4me3 → H3K27ac | 0.965495000492107 | 1.000000000000000 | 0.188767193881747 |

All native per-coordinate and mean R² values reproduce at absolute tolerance `1e-12`. Each source feature StandardScaler and fixed Ridge `alpha=1` uses only that assay's public training rows and is transferred unchanged to both other discovery assays. Both coordinates are negative in both RNA-to-mark directions. Directed hubness was independently calculated from direct distance rankings and target indegrees; the pairwise-difference Gini, coverage and largest indegree fractions match at `1e-15`. The minimum coverage, `0.28167539267015707`, occurs in both mark-to-RNA directions.

The supplemental float64 normal-equation solution is not identical to the native float32 probe. Its largest coordinate-R² discrepancy is `0.0010114992759788777` (RNA→H3K4me3); the RNA→H3K27ac discrepancy is `0.0008432597431973221`. Both exceed `5e-5` and remain explicitly flagged. All four RNA-to-mark coordinate R² values remain negative in float64. Within float64, raw and common-training-affine-scaled state targets give equivalent R² at `1e-10`. Required report reproduction and this supplemental precision discrepancy are distinct conclusions; no declared probe or selection metric was replaced.

Report SHA256 is `d5f03e9ed1d148fe5669c2246e242faeea8e9d325ae8765c3835dc06c596f36f`. Selected-only/discovery labels, exact candidate configuration, frozen protocol/source metadata, export seals, public file hashes, and frozen-plan membership pass. Selection SHA256 remains `9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`; plan SHA256 remains `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.

This check imports no production pool, metric, matching, or probe function. It uses direct NumPy/SciPy calculations, independently written native-library fixed probes, and the separate float64 equations, with one BLAS thread. Access was limited to public train/discovery IDs, assay membership and states, this completed fit's shared embeddings, cached reference IDs, provenance metadata, and the four completed informative reports. No active fit, hidden biology, private cohort, checkpoint binary, H5AD array, CUDA operation, integration training, tuning, causal score, or production edit was involved. Only three fixed diagnostic Ridge probes were refitted; neighbor retention and coordinate variance were not repeated.

The [seed-0 review](scmrdr_constrained_chain_0_informative_seed0_diagnostics_crosscheck.md) supplies the purely analytic illustration of why oracle-relative state matching, fixed source transfer and local coverage can differ. It is an existence example, not a diagnosis of these embeddings. These measurements describe repeated conditions on one existing world and do not establish residual coupling, independent biological-world replication, population generalization, empirical biological correctness, or causal validity.
