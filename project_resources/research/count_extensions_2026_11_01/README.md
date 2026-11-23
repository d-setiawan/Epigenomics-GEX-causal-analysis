# Count-source extension comparison

This study tests six identified extensions of the previous
[count-aware source pilot](../count_mdcrl_2026_11_01/README.md) on the same
RNA, H3K27ac and H3K4me3 observations. It covers 25 prespecified variants across
the real scMTR-seq split and two cached simulations, with two source seeds and
an independent-column control: 225 arms. No new biological data or benchmark
worlds are generated.

Both this block and the separate
[54-arm selected-dimension follow-up](../count_rank20_2026_11_01/README.md)
are complete and independently audited. Both returned zero cross-assay source
pairs passing the moment test, so no pair reached characteristic-function
confirmation. Count-model changes improved some within-assay predictions.
The later [source-search positive control](../../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md)
concerns shared-state transport in worlds generated from a specified Poisson
model. Source matching on these original count panels remains unresolved.

The families are negative-binomial measurement noise, excess-zero measurement
noise, depth elasticity, uncertainty shrinkage of third moments, bounded
generating-function orientation, and expansion from 30 to 300 existing
features. The full frozen grid is in [protocol.json](protocol.json).

Read [METHOD.md](METHOD.md) for the assumptions and equations,
[RESULTS.md](RESULTS.md) for outcomes, and the
[independent theory review](reviews/theory_protocol_review.md) for limitations.
Finite source fingerprints and heldout covariance prediction are narrower
targets than biological causal structure.

## Reproducibility

- The [preflight receipt](preflight.json) records source hashes, the [runner review](reviews/runner_preflight_review.md) and the [data-freeze checks](reviews/data_freeze_review.md).
- [Data manifest](data_v1/manifest.json): immutable nested panels and reference lineage.
- [Extraction audit](reviews/data_extraction_audit.json): source replay and exact original-panel preservation.
- [Fit records](runs_v1/summary.json) and [fit seal](runs_v1/provenance.json): every arm, including failures and abstentions.
- The [evaluator](evaluate_extensions.py) verifies the fit seal before opening reference values. Read the [evaluation protocol review](reviews/evaluation_protocol_review.md) and [readout review](reviews/evaluation_readout_review.md) for the scoring rules. The [characteristic-function variance note](reviews/cf_variance_note.md) explains uncertainty in the fingerprint comparison.
- [Estimator grounding](reviews/estimator_grounding.md) and [primary references](reviews/primary_references.md): published inspiration and local changes.

The source-fit/calibration split, features, noise parameters, thresholds,
matching rules and all 225 arms were specified before scientific fits. Earlier
studies and panels remain unchanged. The same heldout cohort has been inspected
in prior work, so these are exploratory development results.

Git includes the manifest, protocols and decision records. Fit replay also
needs the cached or reconstructed panel arrays named in the manifest.
The numerical fixtures and fit runner import the Sturma pilot's separate
upstream learner. Restore the pinned clone and activate a compatible
`scrna_r` environment before running commands from this study directory.
[Upstream restore instructions](../../../TRANSFER.md#other-simulation-tracks-and-historical-runs) · [Environment guide](../../../envs/README.md).

Run the numerical and decision checks with

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  test_extension_estimators.py test_apply_extensions.py test_evaluate_extensions.py
```

The historical fit and evaluation ran in this order, using two CPU workers and one BLAS thread per worker.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u apply_extensions.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u evaluate_extensions.py
```

Both scripts refuse to overwrite their output directories. Preserve the
sealed artifacts; rerunning scientific fits requires a separate study copy
with its own provenance. The numerical fixtures exercise algebra and boundary
behavior and are not new benchmark worlds.
