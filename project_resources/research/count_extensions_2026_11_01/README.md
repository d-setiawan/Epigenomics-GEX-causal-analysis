# Count-source extension comparison

This study tests six identified extensions of the previous
[count-aware source pilot](../count_mdcrl_2026_11_01/README.md) on the same
RNA, H3K27ac and H3K4me3 observations. It covers 25 prespecified variants across
the real scMTR-seq split and two cached simulations, with two source seeds and
an independent-column control: 225 arms. No new biological data or benchmark
worlds are generated.

Both this block and the separate
[54-arm selected-dimension follow-up](../count_rank20_2026_11_01/README.md)
are complete and independently audited. Neither accepted a cross-assay moment
pair, so CF confirmation was not reached. Count-model changes improved some
within-assay predictions; causal structure recovery remains unsupported. The
two result reports preserve the narrower gains and failed recovery stages.

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

- [Preflight receipt](preflight.json): tested source hashes and independent reviews.
- [Data manifest](data_v1/manifest.json): immutable nested panels and reference lineage.
- [Extraction audit](reviews/data_extraction_audit.json): source replay and exact original-panel preservation.
- [Fit records](runs_v1/summary.json) and [fit seal](runs_v1/provenance.json): every arm, including failures and abstentions.
- [Evaluator](evaluate_extensions.py): separate, verifies the complete fit seal before reading reference values.
- [Estimator grounding](reviews/estimator_grounding.md) and [primary references](reviews/primary_references.md): published inspiration and local changes.

The source-fit/calibration split, features, noise parameters, thresholds,
matching rules and all 225 arms were specified before scientific fits. Earlier
studies and panels remain unchanged. The same heldout cohort has been inspected
in prior work, so these are exploratory development results.

The installed `scrna_r` environment supplies NumPy, SciPy, pytest and the
existing source-model dependencies. CPU execution uses two workers and one
BLAS thread per worker; it does not require a GPU or new packages.

From this directory, the preflight command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -m pytest -q \
  test_extension_estimators.py test_apply_extensions.py test_evaluate_extensions.py
```

The recorded execution order is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -u apply_extensions.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -u evaluate_extensions.py
```

Both scripts refuse to overwrite their output directories. Preserve the
sealed artifacts; rerunning scientific fits requires a separate study copy
with its own provenance. The numerical fixtures exercise algebra and boundary
behavior and are not new benchmark worlds.
