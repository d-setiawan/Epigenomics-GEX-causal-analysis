# Count-aware unpaired source pilot

Completed 2026-11-01. Explicit Poisson correction improved heldout within-assay covariance estimates relative to the same-rank ordinary-moment estimator. **It did not recover supported cross-assay source matches or a causal graph on the tested panels.** The implementation and negative results are preserved for subsequent model work.

## Read the study

- [Results and interpretation](RESULTS.md): all runs, baselines, failure points and what the hypothesis test establishes.
- [Model and derivations](METHOD.md): count sampling, source estimation, sparse-count uncertainty and identification limits.
- [Frozen protocol](protocol.json): training/calibration splits, dimension rule, bootstrap, matching and evaluation criteria.
- [Fit summary](runs_v1/summary.json) and [independent evaluation](evaluation_v1/summary.json): complete machine-readable outcomes.
- [Validation receipt](validation.json): tests, independent audits, source identities and checked links.

## What changed

The earlier [native Sturma pilot](../sturma_pilot_2026_11_01/README.md) applied continuous ICA directly to count representations. This extension uses a conditional Poisson observation model and factorial cumulants to estimate variation in underlying rates. It changes the source estimator and the source-comparison statistic; it does not merely tune native thresholds.

The core [count_sources.py](count_sources.py) estimates mixing from corrected covariance and third cumulants, informed by [discrete ICA](https://proceedings.neurips.cc/paper_files/paper/2015/hash/9be40cee5b0eee1462c82c6964087ff9-Abstract.html). The [training runner](apply_counts.py) selects coherent dimensions using feature permutations, calibrates source fingerprints on a separate part of the training pool, and checks bootstrap stability. It compares corrected moments and finite characteristic-function coordinates across assays without imposing identical source priors or running KS on noisy count projections.

All 30 frozen features enter each assay. Internal whitening and spectral dimension selection remain part of the estimator; no PCA input matrix or new variance-ranked feature panel is constructed. A matched-dimension version with the count correction switched off provides the controlled comparison.

The [separate evaluator](evaluate_count_sources.py) opens heldout references only after every fit is sealed. It checks moments on a common scale, includes measurement variance appropriately, and distinguishes noisy within-assay state scores from matched sources and causal structure.

## Scope and execution

The 36 arms reuse scMTR-seq S1 and two cached simulation worlds, RNA/H3K27ac/H3K4me3, two exposure treatments, two estimators, two original-data seeds and one permutation-control seed. There are 24 original-data arms and 12 controls. No new biological data, project worlds, features or modalities were added. The fit stage used approximately five minutes of CPU computation in the existing `scrna_r` environment; no packages or GPU configuration changed.

The frozen panels comprise background features. The simulation's focal graph describes a different feature set, and the held-out paired cells provide empirical moment comparisons. Git retains the panel manifest, fitting decisions and evaluation summaries. Replaying the fits requires the cached or reconstructed count arrays and the pinned Sturma upstream clone. [Panel manifest](../sturma_pilot_2026_11_01/data_pilot_v1/manifest.json) · [Restore instructions](../../../TRANSFER.md#other-simulation-tracks-and-historical-runs).

With the upstream clone in place, run the fixture tests from this directory in the active `scrna_r` environment. The two original audit scripts listed below write receipts. Run them in a replay copy with the fitted arrays available. [Environment guide](../../../envs/README.md).

```bash
python -m pytest -q test_count_sources.py test_apply_counts.py test_evaluate_count_sources.py
python reviews/verify_count_algebra.py
python reviews/audit_count_fits.py
```

`apply_counts.py` and `evaluate_count_sources.py` refuse to overwrite their output directories. Preserve these results; use a separately versioned study/protocol for another scientific comparison. The [fit provenance](runs_v1/provenance.json) and [evaluation provenance](evaluation_v1/provenance.json) bind inputs, sources and artifacts to this run.

## Independent reviews

- [Theory and implementation](reviews/theory_implementation_review.md), with [independent algebra checks](reviews/count_algebra_audit.json).
- [Algorithm choices and literature](reviews/algorithm_options.md) and [runner review](reviews/runner_review.md).
- [Data and evaluation contract](reviews/data_protocol_review.md), [training-split diagnostics](reviews/training_split_diagnostics.json), and [evaluator validation](reviews/evaluator_validation.json) and [review of the held-out readout](reviews/evaluation_readout_review.md).
- [Replay of all 36 fitted decision paths](reviews/audit_count_fits.json).
- [Heldout metric replay](reviews/count_metric_audit.json), [data and evaluator audit](reviews/data_and_evaluation_audit.json), and [final interpretation review](reviews/result_interpretation_review.md).

The Poisson kernel, affine independent-source rate model and exposure assumptions are unverified on these panels. A finite fingerprint compares selected features of a source distribution. Translating a source match into a latent causal graph requires further model assumptions. The subsequent 225-arm extension and 54-arm dimension follow-up also returned zero source pairs passing the moment test. [Derivations](METHOD.md) · [Extension results](../count_extensions_2026_11_01/RESULTS.md) · [Dimension follow-up](../count_rank20_2026_11_01/RESULTS.md) · [Observation limit](../LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution).
