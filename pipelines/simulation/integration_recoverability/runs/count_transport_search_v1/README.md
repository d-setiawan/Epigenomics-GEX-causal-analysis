# Completed unbounded source-search comparison

This run reuses all nine exact-model worlds from `count_transport_matched_v1`.
No new counts or biological measurements were generated. See the
[full result report](../../COUNT_TRANSPORT_SOURCE_SEARCH.md) for interpretation.

| Sources | Plain unbounded, old physical starts | Count-informed search and calibration selection |
| --- | ---: | ---: |
| 1 | 3/3 | 3/3 |
| 2 | 2/3 | 3/3 |
| 3 | 0/3 | 3/3 |

The successful procedure includes four outer starts and unpaired calibration
selection. Individual starts remain unreliable; five of the twelve informed
three-source starts meet the continuous-error and numerical criteria. Only
selected fits have the bootstrap intervals required for a formal recovery pass.
All datasets and assessment cohorts were previously inspected, so this is
exploratory evidence in a known-dimension, exactly specified source model.
It does not establish causal directions or global optimization.

## Execution

Commands were run from the repository root in the existing `scrna_r` environment,
with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`:

```bash
python pipelines/simulation/integration_recoverability/count_transport_search_run.py freeze
python pipelines/simulation/integration_recoverability/count_transport_search_run.py fit
python pipelines/simulation/integration_recoverability/count_transport_search_run.py evaluate
python pipelines/simulation/integration_recoverability/runs/count_transport_search_v1/numerical_audit.py
```

These completed stages refuse to overwrite their sealed outputs. A new
reproduction needs a fresh run directory containing a copy of `PLAN.md`; pass
that directory with `--run-dir` to each stage and the numerical audit. This
reuses the same original counts and remains a repeated, exploratory experiment.

The run retains 72 outer models, 180 joint refinements and 432 marginal mixture
attempts, including failures. Freeze-to-fit-seal elapsed time was about 50.6
minutes on two CPU workers. The longest joint refinement used 36,134 iterations;
neither joint fitting nor EM had a cumulative iteration cap. Numerical safeguards
and finite start inventories remain explicit in the [plan](PLAN.md).

All 112 scoped count-transport tests passed. The independent numerical audit
passed, replaying 72 training objectives, 18 selections, 108 selected ordered-pair
evaluations and 216 bootstrap intervals. Its limits are listed in the audit JSON.

## Artifacts

- [Protocol](protocol.json) and [fit/selection seal](fit_seal.json).
- [Full evaluation](evaluation.json), [selected comparison](comparison.tsv),
  [aggregate table](aggregate.tsv) and [evaluation seal](evaluation_seal.json).
- [Execution counts](execution_summary.json), [resource snapshot](resources.json)
  and [prior source pins](baseline_source_pins.json).
- [Scientific review](REVIEW.md), [numerical audit](NUMERICAL_REVIEW.json)
  and [independent audit code](numerical_audit.py).

