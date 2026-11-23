# Fixed-pool state readout

The separate [state scorer](../../../pipelines/empirical/integration/methods/scconfluence/state_readout.py)
reuses the unchanged `count_input_experiment.evaluate_views` implementation. It
compares each new scConfluence seed with the existing compressed and full-count
scGLUE fits. It performs no model fitting or causal discovery.

The earlier completed count-input reports define the exact ordered training and
validation pools: 1,936/487 cells per assay for `linear_chain_0`, and 1,934/491
for `constrained_collider_0`. Every supplied embedding is first restricted to
these pools. Missing cells cause an error; pools cannot shrink. The result must
return exactly those same IDs, and baseline transport gaps must reproduce the
earlier reports within absolute tolerance `1e-8`.

The primary endpoint is the worst normalized squared-state transport gap among
the three assay pairs; lower is better. Reports also retain all pairwise costs,
state-prediction transfer from each training assay without target refitting,
hubness and latent variance. PCA/LSI neighborhood retention is included only as
fidelity to the prior representation, not as biological validity. Full-pool
one-to-one matching is an evaluation diagnostic separate from the model's
partial-transport training objective.

Only public validation state labels are used for scoring, after the learned
matches are fixed. Training state labels fit evaluator-only probes. Neither
enters the integration model. State alignment does not establish a direct
causal graph or recover an unobserved residual coupling.

The [frozen protocol](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_scconfluence_state_v1/protocol.json)
binds the root execution plan, previous reports and runtime amendment, original
scientific sources, and scorer implementation. New model and export seals are
verified using the causal agent's existing loader, plus exact configuration,
source-input and training-selection checks against the execution plan. Reports
retain all model/export/protocol/source hashes. They cannot overwrite an
existing scored result.

The focused pool-integrity test passed (one test, 0.54 seconds). It checks exact
row ordering, rejection of missing cells and rejection of changed pools. The
[runner review receipt](runner_independent_review.json) records an independent
check of the six-fit schedule, fixed configuration, source/input hashes, serial
execution, cohort coverage and failure preservation. The causal agent's
[independent scorer review](state_readout_independent_review.json) also passed:
it traced pool integrity, artifact checks and the separation of matching from
evaluator-only state labels. All four planned state reports have now completed.

## Results

Worst-pair normalized state-transport gap is shown below. Zero is the optimal
finite-pool state assignment; one is the expected random-assignment cost, after
the same normalization. Lower is better.

| Existing simulation | scGLUE compressed, seed 0 | scGLUE full counts, seed 0 | scConfluence seed 0 | scConfluence seed 1 |
|---|---:|---:|---:|---:|
| Linear chain | 0.1476 | 0.1365 | 0.3747 | 1.0191 |
| Constrained collider | 0.1312 | 0.1228 | 0.1011 | 1.0213 |

This fixed count-input adaptation is not reliable across seeds in this pilot.
Seed 0 improves the collider's worst-pair matching score, and improves
RNA–H3K4me3 matching on both worlds. Seed 1 is near the random reference on both
worlds. Reporting the better seed alone would hide the main weakness.

The transfer probes expose a separate limitation. In collider seed 0, where
matching improves, H3K27ac-source linear state prediction has R² of −0.4705 in
RNA and −0.6929 in H3K4me3. The other cross-assay directions range from 0.3242 to
0.7196, versus 0.9106–0.9740 for full-count scGLUE. These are fixed linear Ridge
probes; poor R² does not prove that all nonlinear state information is absent.
Absolute latent variance is also not comparable across differently scaled
architectures.

The [four-report review](state_review_summary.json) binds results, execution
logs and audit receipts. All pools were preserved exactly, and prior baseline
transport scores reproduced. The independent
[NumPy Ridge replay](recompute_state_transfer.py) checks all 27 method/direction
readouts in each report, including source-only normalization, intercept and
alpha conventions. The saved GLUE loader preserves float32, whereas the
scConfluence verifier returns float64; the audit records those conventions and
precision-specific tolerances. The root agent independently reconstructed
assignments and state costs from exported arrays using NumPy/SciPy.

These results concern the declared full-pool diagnostic and this no-PCA/NB
adaptation. They do not establish a general failure of published scConfluence,
or successful causal recovery in the improved matching case. Causal tests and
real scMTR-seq readouts are reported separately.

Run the scorer in the original `scrna_r` environment, with the same process-local
runtime and one-thread BLAS settings as the other evaluators:

```bash
env -u CUDA_VISIBLE_DEVICES -u LD_LIBRARY_PATH \
  LD_PRELOAD=/tmp/integration_followup_runtime_20261031/libcudart.so.13.0.96 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ir_mpl_cache \
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
  pipelines/empirical/integration/methods/scconfluence/state_readout.py \
  score --world linear_chain_0 --seed 0
```

The same fixed command applies to the other world and seed. There are two
previously inspected development worlds, not independent confirmation data;
scGLUE has one seed per arm and scConfluence has two. No thresholds or model
settings are selected from these readouts.
