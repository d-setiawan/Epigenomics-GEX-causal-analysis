# Frozen-suite synthesis protocol

_Pre-assessment reporting specification, 2026-10-04; no assessment outcomes inspected._

---

## Scope and freeze

This protocol specifies reporting for the existing frozen integration-
recoverability schedule. It does not alter simulations, integration fits,
matching, causal discovery, scores, or the assessment gate. Development
results were already available when this reporting protocol was written.
It is therefore a pre-assessment analysis commitment, not a claim that the
entire analysis was registered before development.

The companion [summary helper](scripts/summarize_frozen_suite.py) reads saved
JSON artifacts only. It imports no scientific pipeline modules and never
generates data, fits a model or probe, runs a conditional-independence test,
constructs a new graph, or rescores edges. The original reports and metadata
remain unchanged. Every output is a new snapshot in a previously absent
directory.

Before assessment release, the `freeze` command binds the helper, this
protocol, NumPy version, schedule, scientific source universe, bank,
calibration receipt, exact world IDs and split/stratum memberships, primary
report paths, and metric schema. It refuses to freeze if the assessment lock
or any declared assessment artifact already exists. Review must precede the
production freeze. Subsequent changes require an explicitly new protocol;
the helper refuses a changed identity under the existing pin.

## Units and populations

The causal assessment population is exactly 18 worlds: families `linear`,
`constrained`, and `unrestricted`, each crossed with `chain` and `collider`,
with independent draws 1, 2, and 3. The world ID is
`{family}_{motif}_{draw}`. The three `shared_state_1` through
`shared_state_3` worlds form a separate control population for unsupported
direct regulation, not adjacency-null controls: shared state can produce
real assay associations.

Development consists of the six causal draw-0 worlds, `shared_state_0`, and
the two exact ambiguity counterparts `ambiguity_rho_0.0` and
`ambiguity_rho_0.7`. The simple three-seed study remains a separate first-pass
diagnostic. The ambiguity counterparts are one observational-ambiguity
control, not two independent biological replicates. They are not used in
world-bootstrap summaries.

The two count regimes are repeated conditions within a world, not additional
worlds. Integration seeds, edges, cells, and matching tuples are not
independent biological replicates. Primary method comparisons use seed 0,
the `primary` arm, and the original RCIT configuration only. Their paths are
fixed as:

```text
runs/extended/{world}/{realistic|informative}/primary/
  evaluation_scglue_scmrdr_seeds0/report.json
```

All scheduled stages are inventoried. Other seeds, ablations, secondary KCI,
Fisher-Z, guidance-only priors, and twin controls retain separate labels and
never replace or duplicate a primary result. The primary source retains
RCIT alpha 0.01, CI seed 0, matching seed 3141, pool seed 2718, and its full
saved RCIT configuration. Predeclared secondary probe skips are respected.

Two separately frozen, development-only anchor/readout diagnostics sit
outside the main schedule. Their exact directories are
`runs/secondary/{linear_chain_0,constrained_chain_0}/informative/anchor_readout_v1/`.
The synthesis pin binds each existing `freeze.json` hash and the expected
`secondary_report.json` and `secondary_artifact_manifest.json` paths. Frozen
inputs alone are `pending`, not completed scientific evidence. Completed
reports must match the input hash, artifact seal, four predeclared condition
IDs, and an immutable completion receipt binding the report and seal hashes.
Original raw payloads, failures, attempts, and links are inventoried
separately. They never enter primary aggregates or add independent worlds.
Only these two explicit supplementary paths are selected; no result glob
is used to discover favorable diagnostics.

## Individual outcomes and estimands

Each metric row identifies split, world, family/motif/draw, regime, arm,
method/seed, report/hash, common-pool hash, stage, target, assay scope, metric,
value or null, saved numerator/denominator when available, and status/reason.
The fixed metric schema in the pin is authoritative.

| Layer | Reported outcomes | Interpretation |
| --- | --- | --- |
| Activities | Adjacency precision, recall, F1, false-positive rate and counts; endpoint scores | Agreement with the activity latent-projection oracle |
| Original counts | The same fields, including all/cross/within-assay and individual assay-pair scopes | Agreement with the count-measurement PAG, not the direct biological DAG |
| Biological-edge stress test | Exact cross-assay pair and directed-edge precision/recall/counts; saved exact-set flags | Identifiability stress, not an expected recovery-success criterion |
| State matching | Pairwise mean absolute and Euclidean state error; saved per-dimension errors | Matching-state diagnostic, not proof of preserved residual dependence |
| Private probes | All saved ridge and shallow-tree scores, same/cross-assay, shared/private representations, state-only and incremental scores, population sizes | Fixed readout performance; no probe selection or refitting |
| Native retention | Observed/common/eligible cell counts and saved feature/dimension metadata | Different pools and feature retention are reported, not silently equated |
| Sentinel behavior | Saved sentinel incidence and target-relative false-positive rates with their available counts | Exposure-induced count adjacencies must not be mislabeled biological sentinel recovery |
| Hyperedges/ambiguity sets | Producer availability and reason | N/A when the required graph object or candidate-family contract is absent |

Both paired references, random matching, true-state matching, the
state-supplied activity control, and graph oracles remain separate outcomes.
The independent paired reference and common-pool anchor reference may have
different populations and sample sizes. This helper does not invent
selection-adjusted truth or new scores. Assay-participation claims are not
inferred when no hyperedge object is emitted.

Probe metadata, conditional-mean Monte Carlo uncertainty, count-calibration
flags, integration provenance, original limitations, graph-oracle payloads,
exact-set stress outputs, and producer failure lists are retained or linked
with the original report hash. Missing feature-retention metadata does not
imply omission, and retaining a focal input does not prove retention of its
information in an embedding.

## Missingness and failures

A successfully computed empty graph has undefined precision when its
prediction count is zero; recall is zero only when a nonempty truth target
exists. A numerical or other graph-execution failure has unavailable graph
metrics, not an empty graph. Undefined, missing, pending, blocked, failed,
invalid, and not-applicable outcomes remain distinct from numeric zero.
Saved graph scores with a non-null precision or recall at a recorded zero
denominator are quarantined as invalid rather than reinterpreted.

Generic graph false-positive and some sentinel denominators are not stored
by the evaluator. They remain `not_recorded`; no denominator is reconstructed
from a rounded rate, candidate count, or another target. Endpoint scores and
stress-test flags retain their own meaning rather than being merged with
adjacency scores.

Component rows count each report/method/discovery layer, oracle,
state-matching diagnostic, and probe collection once. Counts are grouped by
split, analysis role, regime, method, and component, with explicit status
counts and row denominators. They are descriptive execution inventories,
not independent Bernoulli trials. Metric rows and graph edges are never the
denominator of a failure frequency. The producer status
`completed_with_failures` is retained even when its only cause is the
predeclared `planned_skip` of secondary probes.

Immutable stage-attempt receipts are retained, including original exit
codes, logs/receipt hashes, notes, and artifact hashes. A nonzero-exit stage
without its complete report is an operational failure, not pending science
or a zero score. Its unavailable metric/component rows are labeled
`blocked_operational_failure`; this does not claim every diagnostic was
attempted or that every CI calculation failed. A later complete result does not erase a recorded earlier
attempt failure. Missing receipts do not establish that no historical or
manual attempt occurred. Executor-session validation refusals before any
scientific command are a separate `operator_session` category and are not
counted as failed scientific stages. Missing reports without a receipt
remain pending/partial; the helper does not guess their cause from log text.

## Paired method comparisons

The signed contrast is always

$$
\Delta_{w,r,t}=M_{\mathrm{scMRDR},w,r,t}-M_{\mathrm{scGLUE},w,r,t}.
$$

It uses the same world, regime, target, scope, metric, and combined-report
common pool. Positive error differences therefore favor scGLUE; signs are
not flipped to manufacture a composite winner. There is no aggregate
"best method" score and no performance-dependent selection of a CI test,
seed, world, regime, or probe.

The helper reports every individual paired difference, then stratum and
population summaries. A difference is unavailable if either method's value
is undefined or failed. Complete-bank means require all expected values.
The additional available-pair arithmetic mean is explicitly labeled a
**varying-world-set descriptive mean**, includes exact valid/unavailable
world IDs and counts, and does not silently reweight the surviving strata.
It is conditional on availability and cannot stand in for a complete-bank
estimate. Stratum medians and ranges are descriptive. Controls are never
folded into causal-world averages.

No probe contrasts are aggregated across assays, capacities, or unequal
native probe populations. Probe rows are retained individually. No missing
precision is imputed as zero or one. Metric-specific available sets can
differ and are exposed rather than equated.

## World-level uncertainty

Only fully defined paired metrics across **all 18 causal assessment worlds**
receive a 95% pointwise percentile interval. The bootstrap draws three
worlds with replacement within each of the six family/motif strata, for
10,000 replicates, using `numpy.random.PCG64` seed 61709. Each replicate
averages the 18 sampled paired differences; equal stratum weighting follows
from the balanced design. NumPy's `linear` quantile rule is used at 0.025
and 0.975.

One `(10000, 6, 3)` index array and its exact world-ID order are saved and
reused across every method comparison, metric, target, and regime. There is
no resampling of cells, edges, seeds, or count regimes. An interval is not
computed when even one expected paired value is unavailable. No inferential
interval is reported for development, an individual three-world stratum,
the three controls, or the twin pair. With only three draws per stratum,
these are approximate, pointwise, limited-bank uncertainty summaries, not
simultaneous coverage or a reliability guarantee. No p-values or
post-development reliable-success thresholds are introduced.

## Evidence flags and limitations

Flags identify recorded execution failures, failed declared count-calibration
checks, nonpositive residual-probe test $R^2$, and the mismatch between a
source-population oracle and potentially QC-selected readouts. They are
nonexclusive descriptions. They do not identify a unique failing mechanism,
prove information loss, or apportion discrepancies among measurement,
selection, compression, model objective, matching, probe power, Monte Carlo
target uncertainty, and CI calibration. Negative probe $R^2$ means this
readout did not outperform the assessment-mean baseline, not negative
information. Random-control source-graph overlap is not causal recovery.

## Assessment access and provenance

Without `--include-assessment`, no phase-F scientific artifact is opened;
its status is `blocked_assessment_lock`. With the flag, the helper first
validates the existing assessment lock against the frozen source/bank/
calibration identities and every locked development report hash. It does
not create or release the lock. No real assessment summary is authorized
merely by implementing or testing this helper.

When the lock exists but the flag is absent, all byte reads of opaque
mixed-directory attempt receipts and operator-session journals are
suppressed. The summary explicitly records that accounting scope. Before
the lock, an unexpected assessment-stage receipt or session is rejected,
not incorporated. The bank's already frozen specification metadata is
used only to enumerate expected worlds, not to inspect assessment data.

The helper verifies report input fingerprints and configuration, scheduled
world paths, world/truth receipts, and integration-manifest identity. It
hashes every inspected report/artifact and compares any immutable attempt
receipt. It does not rehash large H5AD/checkpoint/embedding matrices or
rerun upstream validation; those remain covered by existing pipeline
receipts. This limitation is explicit in the snapshot's integrity scope.

The outputs are `synthesis.json`, long-form `metrics.csv`,
`paired_differences.csv`, `stage_status.csv`, `component_status.csv`, and
`bootstrap_world_indices.npz`, optionally a short linked `summary.md`.
`output_manifest.json` hashes each new file. No existing output directory
is overwritten.
