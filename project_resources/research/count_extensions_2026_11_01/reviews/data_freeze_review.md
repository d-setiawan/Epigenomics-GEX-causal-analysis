# Cached nested-panel data review

The expansion uses the existing scMTR-seq S1, linear-chain, and constrained-collider cohorts, with the same RNA, H3K27ac, and H3K4me3 assays. No cells, experiments, biological measurements, simulated worlds, or private probe/assessment cohorts were added. The original extraction and its artifacts remain unchanged.

The [extraction protocol](../data_extraction_protocol.json) was frozen before writing expanded arrays (SHA256 `8bed467b52175781c96cdae9318fc6e535ab4d62ee4ee3d5ec3f74914ce1cdf0`). It preserves the own-assay training detection floor of 10 cells, nonconstant raw counts, original target exclusions, and ascending SHA256(feature ID) order. The selected panel is the first 300 eligible features; the first 30 must reproduce the immutable earlier panel. No variance ranking, PCA, shared/private/null role, test outcome, pairing, state label, or expression-correlation ranking enters selection.

## Frozen artifacts

The new [data manifest](../data_v1/manifest.json) has SHA256 `30749e63fc85d55edf5092efd5cb014c3fbb4947b0c2dce6ec3fa3e0f882ce2d`. Its 18 public NPZ files retain the same five fields: `raw_counts`, `log1p_cp10k`, `library_depth`, `cell_ids`, and `feature_ids`. Each has 300 features. All five original-prefix fields passed exact dtype, shape, and C-order byte comparison against the old 30-feature data. Every original opaque cell ID and full allowed-feature depth is unchanged.

Real training rows remain RNA 602, H3K27ac 615, and H3K4me3 614; real test rows remain 595 per assay. Simulated training/discovery rows remain 4,000/2,000 per assay. The real evaluator reference is 595 by 900, built from the expanded public test arrays using only the old held-out pair indices. Canonical training pair maps and day labels were not opened. Simulated state-reference files were copied byte-for-byte without loading their state values. No held-out covariance, correlation, or model-performance readout was computed during extraction.

The [data-view contract](../data_view_contract.json) explicitly lists the two nested column sets and original-30 anchor positions. For a clean width comparison, the new depth sensitivity uses original full depth minus the sum of **all 300** selected features for both widths. It must be normalized on source-fit rows only, and must remain fixed under count permutations. This differs from the previous leave-30-out sensitivity. Control permutations must also be nested: shuffle all 300 and then slice, or use feature-specific seeds independent of panel width.

## Training coverage and count-model caveats

| Dataset | Assay | All-selected-zero training cells: 30 to 300 | Detection range per selected 300-feature column |
|---|---|---:|---:|
| scMTR-seq S1 | RNA | 14 to 0 | 10–491 |
| scMTR-seq S1 | H3K27ac | 341 to 18 | 10–35 |
| scMTR-seq S1 | H3K4me3 | 317 to 6 | 10–33 |
| Linear chain | RNA | 19 to 0 | 10–3,986 |
| Linear chain | H3K27ac | 619 to 1 | 103–979 |
| Linear chain | H3K4me3 | 24 to 0 | 505–2,252 |
| Constrained collider | RNA | 29 to 0 | 10–4,000 |
| Constrained collider | H3K27ac | 537 to 1 | 125–993 |
| Constrained collider | H3K4me3 | 15 to 0 | 573–1,974 |

All training leave-300-out depths are positive. More columns improve per-cell coverage without changing the sparse per-feature histone detection distribution. The fixed histone panel includes two overlapping H3K27ac promoter pairs and one H3K4me3 pair. They were reported, not removed or replaced. Overlapping windows may count the same fragment, so independent featurewise Poisson or NB measurement errors are not automatically valid. The remaining inherited caveats also persist: six-target source QC, one mixed-day technical sublibrary, endogenous promoter-depth totals, and promoter-only feature scope.

## Feasibility and evaluation boundary

Extraction took 19.17 seconds with peak resident memory 465.63 MiB; compressed public/evaluator artifacts total 11.05 MiB before the manifest. The largest input CSR occupies about 123 MiB in memory. The helper loads one original CSR at a time and densifies only the selected columns. A 4,000 by 300 float64 matrix needs 9.6 MB; a 300-cubed float64 tensor needs 216 MB before work buffers and is unnecessary for the planned projected-moment methods.

Whole-panel covariance averages for 30 and 300 features describe different targets. The independent evaluator must retain a common original-30 block, with the same exposures and source-fit scaling, to test whether the additional features improve inference on fixed targets. Full-panel scores are supplementary descriptions. Paired cells and simulated states remain evaluator-only, and outcome readouts wait until all scientific fits and their hashes are sealed. These previously inspected development cohorts cannot become fresh confirmatory validation through feature expansion.
