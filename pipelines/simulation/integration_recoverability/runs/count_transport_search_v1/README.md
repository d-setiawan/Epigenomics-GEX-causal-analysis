# Count-informed source search on nine exact-model worlds

In the November 15 experiment each assay contributed 12,000 unpaired fitting cells per reused
world. RNA, H3K27ac and H3K4me3 shared the stipulated independent Bernoulli sources, the fitting
model matched the count generator, and the true source dimension was supplied. The learner
selected one of four outer starts per world using unpaired calibration data. A world passed when
the selected fit met all criteria across every ordered assay pair, including shared-state
transport, count covariance and conditional prediction.
[Experiment and criteria](../../COUNT_TRANSPORT_SOURCE_SEARCH.md) · [Frozen plan](PLAN.md)

| Sources | Earlier labeling study | Unbounded control with original starts | Count-informed search |
| --- | ---: | ---: | ---: |
| 1 | 3/3 | 3/3 | 3/3 |
| 2 | 0/3 | 2/3 | 3/3 |
| 3 | 0/3 | 0/3 | 3/3 |
| **All worlds** | **3/9** | **5/9** | **9/9** |

Only 5/12 individual three-source informed starts passed the continuous-error and numerical
checks. Bootstrap intervals for the full recovery criterion were computed for the selected fits.
With assessment on previously inspected worlds and source dimension supplied, the result
supports recovery under this observation model. Causal directions, unknown source dimension and
generalization to new biological populations remain open. [Selected fits](comparison.tsv) ·
[Aggregate results](aggregate.tsv) ·
[Labeling study](../count_transport_labels_v1/aggregate.tsv)

The historical two-worker run took about 50.6 minutes from freeze to fit seal. It recorded 72
outer models, 180 joint refinements and 432 marginal mixture attempts. Joint fitting and EM
continued to their stopping criteria without a cumulative iteration cap. The longest joint
refinement used 36,134 iterations. An independent audit replayed 72 training objectives, 18
selections, 108 ordered-pair evaluations and 216 bootstrap intervals.
[Execution summary](execution_summary.json) · [Audit](NUMERICAL_REVIEW.json) ·
[Audit code](numerical_audit.py) · [Scientific review](REVIEW.md)

## Reproduction

A Git checkout carries the frozen protocol, seals, tabulated scores and reviews. The complete
original evaluation and per-fit payloads reside in the run cache. To repeat the comparison from
its deterministic inputs, use a fresh run ID in the CPU environment and run this command from
the repository root. It creates both the matched baseline and the subsequent search under
`runs/replay-search/` in this track.
[Environment and replay instructions](../../../../../TRANSFER.md) · [Protocol](protocol.json) ·
[Original source pins](baseline_source_pins.json) · [Fit seal](fit_seal.json) ·
[Evaluation seal](evaluation_seal.json)

```bash
python pipelines/simulation/integration_recoverability/replay_count_transport.py replay-search --stage search
```

On November 22 a clean checkout passed all 112 scoped CPU tests, reproduced the full matched
baseline, froze the source search against that replay, and completed one public-count
source-search fit. The 9/9 result above comes from the original full source-search run.
[Handoff verification](../../../../../reproduce/VERIFICATION.md) ·
[Original checks](FINAL_CHECKS.json)
