# Cached-data contract for the Sturma application pilot

Prepared on 2026-11-01 by `scmtr_data_pilot`. No model fitting, biological data acquisition, new simulated worlds, or locked-cohort evaluation was performed. Existing source data were read without modification.

The three requested datasets have been exported under `../data_pilot_v1/`. They support an exploratory application of the algorithm to the permitted counts. They are not an in-class replication of the paper's causal-identification theorem. The simulation permits a descriptive shared-state check; the real heldout cells permit an empirical joint-moment check. Neither supplies a known nontrivial shared-latent DAG for this pilot.

## Frozen extraction

The [protocol](../data_extraction_protocol.json) was written before selecting the 30 columns. SHA256: `7e8e158a0e64913efb6078b3ec51f281e48a2b1de7598cd74c7df01a5cd4e9fe`. The [helper](../prepare_cached_data_pilot.py) refuses to overwrite an existing extraction and verifies source hashes before reading counts. The completed [manifest](../data_pilot_v1/manifest.json) has SHA256 `d56c1cbec8aa4bd899c712b5bb7e0ee2e1b3e0e58d42955b4b79e85758f1d0b8`.

For each dataset and assay, features must be detected in at least 10 of that assay's training cells and have nonconstant training counts. Eligible feature IDs are sorted by SHA256 of the feature ID; the first 30 are retained. There is no variance ranking, PCA, use of test outcomes, or selection by latent state, biological graph, coefficient, or cross-assay association. The helper fails if fewer than 30 qualify. No threshold was relaxed.

Real inputs already exclude the six promoter-overlap/alias features from the existing nine-node evaluation panel. Simulation inputs exclude exactly the nine previously declared scored targets, three per assay. This fixed holdout mask is the only evaluator-derived feature information used in extraction. The simulation's public `var` has no background-role field, so no hidden shared/private/null classification was used to enrich the selected features.

Two deterministic input branches are saved: unmodified integer counts and `log1p(1e4 * selected_counts / full_allowed_feature_depth)`. Depth sums every allowed original feature before selecting 30, after the declared target exclusions. It never uses another assay's counts or excluded focal counts. No source row is dropped. All full-depth values are positive in these exports. Selected zero-count rows remain zero rows in both branches.

## Exact locations and shapes

Paths below are relative to the repository root. For each assay `RNA`, `H3K27ac`, `H3K4me3`, the public files are:

`project_resources/research/sturma_pilot_2026_11_01/data_pilot_v1/public/<dataset>/{train,test}/<assay>.npz`

Every public archive contains exactly `raw_counts`, `log1p_cp10k`, `library_depth`, `cell_ids`, and `feature_ids`. Counts and normalized values have shape cells × 30. Cell IDs remain opaque and assay-specific; no canonical pairing, RT tag, differentiation day, state or true graph enters these archives.

| Dataset | Source count matrices | Training rows RNA/ac/me3 | Test rows per assay | Full allowed features RNA/ac/me3 |
|---|---|---|---:|---|
| `scmtr_seq_s1` | `pipelines/empirical/data/scmtr_seq/cohort_s1/model_inputs/{train,test}/<assay>.h5ad` | 602 / 615 / 614 | 595 | 23,383 / 26,569 / 22,481 |
| `linear_chain_0` | `pipelines/simulation/integration_recoverability/runs/worlds/linear_chain_0/observed/informative/{train,discovery}/<assay>.h5ad` | 4,000 / 4,000 / 4,000 | 2,000 | 1,997 / 14,997 / 14,997 |
| `constrained_collider_0` | `pipelines/simulation/integration_recoverability/runs/worlds/constrained_collider_0/observed/informative/{train,discovery}/<assay>.h5ad` | 4,000 / 4,000 / 4,000 | 2,000 | 1,997 / 14,997 / 14,997 |

The simulation `test` export is the existing **public discovery** cohort; validation, private probe and private assessment cohorts are unused. Simulation train/discovery assay populations are genuinely disjoint, rather than shuffled views of common cells. Old scGLUE preprocessing selections and compressed embeddings are not used. The helper verifies disjoint source IDs across every train/test/assay group.

The manifest records per-domain source paths and hashes, selected feature IDs and source-column indices, ordered row/feature hashes, archive hashes, eligible-feature totals, count sparsity, and centered numerical-rank diagnostics. Exact saved-array roundtrips are asserted. Each assay's test columns are fixed from its training selection, with exact source feature-order equality required.

## Evaluator-only inputs

For scMTR, `data_pilot_v1/evaluator/scmtr_seq_s1/paired_reference.npz` contains the selected raw and normalized same-cell reference matrices, each 595 × 90, in assay-major RNA/ac/me3 order. `paired_indices_<assay>` indexes that assay's public test rows into this common order. `column_ids`, `column_assays` and canonical IDs are evaluator-only. The source join is the existing `pipelines/empirical/data/scmtr_seq/cohort_s1/evaluator/heldout_pairs.tsv`. The existing training maps verify that no canonical cell overlaps between training assays or between training and test. Original RT/day labels remain in the source evaluator directory and are not exported into the new model inputs.

This is an internal complete-case holdout from a single pooled-day S1 sublibrary, not an independent biological replicate. The published source had already selected cells using total counts across six histone targets; that inherited selection cannot be undone with the three permitted assays. Subsequent training assignment and QC use the original frozen seed-2718 split and own-assay QC. The paired reference is an observed dependence reference, not causal truth. The promoter-only source also excludes distal regulatory elements. Overlapping promoter annotations can share fragments; selection does not silently replace features to improve this property.

For each simulation world, `data_pilot_v1/evaluator/<world>/state_reference.npz` contains `state_<assay>` arrays of shape 2,000 × 2 and matching `ids_<assay>`. These are joined from the existing `truth/discovery.npz` and verified against assigned assay IDs. They must be read only after fitting for descriptive state associations. The public discovery truth file contains paired counts only for the original nine focal/decoy/sentinel targets, not the arbitrarily selected background columns. The repository also has full paired matrices in locked private cohorts; those were not read or used here. No paired background counts were regenerated.

## Suitability and limits

Every selected training and test matrix has 30 distinct count columns and centered numerical rank 30 in both raw and normalized branches. Rank was logged as a diagnostic, not used to alter selection. Full count covariance rank does not establish that there are 30 identifiable biological sources. The algorithm's native rank/ICA logic and any internal whitening should be reported explicitly by the fitting runner; these exports contain raw feature columns without upstream PCA.

The real histone subset is particularly sparse:

| Assay | Training detections per selected feature | Training rows zero across all 30 | Test rows zero across all 30 |
|---|---:|---:|---:|
| RNA | 10–441 | 14 / 602 | 18 / 595 |
| H3K27ac | 10–25 | 341 / 615 | 362 / 595 |
| H3K4me3 | 10–29 | 317 / 614 | 325 / 595 |

Thus a source-distribution match could reflect similar sparse count shapes or depth effects. Success or failure of this small frozen subset should not be presented as a decisive assay-wide biological result. No additional feature replacement or sparsity threshold was chosen after inspecting these diagnostics.

The cached extended simulations also fail important assumptions of the theorem. In `simulator.py:134`, state is an asymmetric standardized Gaussian mixture `S0` and an independent standard Gaussian `S1`; their true state graph has no edge. Background generation at `simulator.py:349` uses a mixture of linear and tanh state responses, shared/private/null classes, assay-specific Gaussian private factors and independent feature noise. Count generation adds exposure-dependent negative-binomial measurement noise. The world name `linear_chain_0` describes its focal structural mechanism, not a noiseless linear observation model for all background features. Focal causal edges are a separate target from shared-latent DAG recovery.

These facts justify descriptive state association and heldout joint-moment tests, while preventing an in-class theorem claim. A real count application's fitted latent edges are hypotheses under unverified restrictions; genuine same-cell evaluator observations can assess joint moments but cannot supply causal direction labels.

## Reproduction and verification scope

The extraction command was:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
  project_resources/research/sturma_pilot_2026_11_01/prepare_cached_data_pilot.py
```

It completed successfully. Source checksums, count type/nonnegativity, exact feature order, archive roundtrips, finite normalization, disjoint public IDs, canonical real training/test separation and bijective heldout pairing were checked during extraction. No heldout covariance, causal discovery, learned source matching or new model result was computed by this data-preparation task.

A separate read-only [replay implementation](audit_data_extraction.py), which does not import the helper, subsequently verified all 18 public archives. It rederived feature eligibility using column moments, used a different full-depth summation, and rebuilt the real pairing from the independent per-assay canonical maps. Simulation state joins were replayed through sorted-ID lookup. All checks passed; see [audit result](data_extraction_audit.json), SHA256 `113663f6ec58b256c9430996c01f63e2d74c6b531df8acd98a730a1a4a7e39ba`. This is independent code-path verification by the same data-preparation reviewer, not a claim of independent biological replication.
