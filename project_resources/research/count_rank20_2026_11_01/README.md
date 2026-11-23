# Selected-dimension follow-up for the 300-feature panels

Completed and independently audited: all 54 arms ran, with no accepted
cross-assay moment pair. Removing the computational ceiling did not resolve
source instability or fingerprint disagreement. [Results](RESULTS.md) · [Independent review of fit decisions](reviews/fit_decision_review.md)

This is a separate exploratory follow-up to the
[225-arm count-extension study](../count_extensions_2026_11_01/README.md).
Every original unit-exposure 300-feature assay fit in that study selected
more than six coherent covariance modes: the uncapped ranks ranged from
7 to 19. The original computational ceiling therefore prevented those arms
from testing source matching at their selected dimensions.

The [new protocol](protocol.json) raises the ceiling to 20 for the complete
affected family: Poisson, ordinary moments, NB shape 2, excess-zero retention
0.5, PGF orientation and third-moment shrinkage, all at 300 features and unit
exposure. The same three datasets, two source seeds and one column-permutation
control give 54 arms. These empirical dimensions are not known biological
dimensions.

The runner, estimator and evaluator are byte-identical copies of the reviewed
original files. The same training counts, null RNG keys, bootstrap streams,
stability requirements, orientation residual, moment margins and CF gates
apply. The ceiling and its cap-binding eligibility restriction change; a
full-rank fit may still be unstable or fail another gate. No new data or
benchmark worlds are generated.

This follow-up was designed after the original fit and evaluation seals.
Its stated trigger is the saved training-rank evidence, and all affected
variants are retained. It is an adaptive development sensitivity, not fresh
validation. Earlier protocols and artifacts remain unchanged.

The [original method](../count_extensions_2026_11_01/METHOD.md) supplies all
equations; its cap-six rule is replaced by the cap-twenty rule only here.
[RESULTS.md](RESULTS.md) owns the follow-up outcomes. The later exact-model
experiment supplied the latent dimension (one, two or three) and estimated
transport in all nine reused worlds. Its known source count and generative
assumptions differ from estimating dimension and matching sources on these
300-feature panels. [Exact-model experiment](../../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md).

## Records and execution

- [Preflight](preflight.json) and [independent theory review](reviews/prefit_theory_review.json).
- [Fit summary](runs_v1/summary.json) and [fit seal](runs_v1/provenance.json).
- [Evaluation summary](evaluation_v1/summary.json) and [evaluation seal](evaluation_v1/provenance.json).
- [Original data manifest](../count_extensions_2026_11_01/data_v1/manifest.json).

Git preserves the shared data manifest and fit and evaluation records. Replay
requires the cached or reconstructed 300-feature panel arrays and the pinned
Sturma learner. Restore them in the same relative study layout, activate the
compatible `scrna_r` environment, and run from a separate study copy with fresh
output directories. [Source restoration](../../../TRANSFER.md#other-simulation-tracks-and-historical-runs) · [Environment guide](../../../envs/README.md).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u apply_extensions.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u evaluate_extensions.py
```

The scripts refuse output overwrite. Two CPU workers and one BLAS thread per
worker are used. Larger tensors and full bootstrap refitting can take longer
than the first block. Failures and abstentions are recorded without reducing
the selected dimension or changing thresholds after inspecting outcomes.
