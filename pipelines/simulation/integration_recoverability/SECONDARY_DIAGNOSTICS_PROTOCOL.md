# Secondary matching diagnostics and completion audit

_Development-only protocol addendum, 2026-10-04; specified before these secondary runs._

---

## Purpose and status

The [extended plan](../../../project_resources/research/integration_recoverability_extended_plan.md)
calls for alternative-anchor diagnostics and repeated readout randomness.
The frozen primary evaluator uses RNA as its anchor. Its suite repeats
integration seeds but does not separately vary the readout seed. This addendum
closes those execution gaps without changing the primary evaluator, existing
reports, fitted models, simulated populations, or assessment inputs.

The [secondary helper](scripts/secondary_matching_diagnostics.py) has passed
independent source review and 23 lightweight tests with mocked CI calls. A
source and input receipt must be frozen before execution. Track subsequent
execution in the [implementation record](IMPLEMENTATION_STATUS.md); this
document is a protocol, not evidence that the diagnostics have run.
Development results were already
available when it was written; these checks are therefore labeled secondary,
not an independently preregistered validation study.

## Fixed comparison

Use only `linear_chain_0` and `constrained_chain_0`, both in the informative
count regime, with the existing primary seed-0 scGLUE and paper-consistent
scMRDR fits. These are the same two development worlds already selected for
the secondary CI comparison. No assessment world is eligible.

Retain the exact common cell pools and candidate order of each primary
seed-0 report. Do not select new cells, fit an embedding, generate a world,
change a likelihood, or use hidden information to choose a matching.

| Diagnostic | Anchor | CI seed | Random-reference seed |
| --- | --- | ---: | ---: |
| Cached primary context | RNA | 0 | 3141 |
| Alternative anchor | H3K27ac | 0 | 3141 |
| Alternative anchor | H3K4me3 | 0 | 3141 |
| Readout repeat 1 | RNA | 1 | 3142 |
| Readout repeat 2 | RNA | 2 | 3143 |

All new graph tests use the frozen primary RCIT configuration, including
`alpha=0.01`, its explicit approximation, and its random-feature settings.
Only the declared test seed changes in the readout repeats. Learned and
true-state assignments are deterministic given the fixed pools and
representations; those assignments must remain identical across the RNA
readout repeats. The random control changes its permutation as declared.
This is a combined random-control/test sensitivity check, not an isolation of
the permutation effect from the test-seed effect.

Changing the anchor also changes row order. The installed RCIT/RIT backend
estimates kernel bandwidths from the first 500 rows (or all rows when fewer
are present), so an anchor difference can include test-bandwidth sensitivity
even when an assay-pair assignment is only inverted. Preserve the declared
ordering and label this as full matching/readout/test sensitivity, not a
pure effect of biological coupling. Do not reorder rows to repair a numerical
failure; the [sparse-CI audit](SPARSE_CI_AUDIT.md) documents the same backend
limitation in the simple study.

## Controls and recovery targets

For each secondary condition, retain both methods, random matching, true-state
matching, and a same-cell paired reference. Each anchor uses its own hidden
same-cell reference: histone-anchored comparisons must not retain the RNA
anchor's paired panel. Hidden activities and state enter evaluation and
oracle controls only, never the learned matching.

Reuse the existing minimum-total-cost, one-to-one Euclidean matching kernel.
Read out original named-feature counts and activities, with no averaging.
Check that each feature's marginal values and detection count remain identical
across the no-reuse matchings: the fixed pools are only being permuted. With a
unique assignment optimum, swapping the anchor between a given assay pair
preserves that pair's assignment up to inversion, but can change its coupling
to the third assay and therefore its conditional tests. Ties can break the
inversion equivalence. These are linked comparisons, not independent trials.
Retain the existing separate activity-projection and count-measurement PAG
targets, assay-pair breakdowns, sentinel metrics, endpoint scores, and labeled
exact-pair/direction stress tests. A changed anchor does not turn an imposed
coupling into paired observations.

Show each anchor's own paired reference next to its matched results. Different
anchor reference panels contain different cells, so their differences also
include finite-reference variation. Two extra CI seeds and one random
permutation per condition are sensitivity examples, not an estimate of a
stable variance or false-adjacency rate. Retain per-query logs because an
adaptive graph search can propagate one CI decision into several edge changes.

The primary independent paired-assessment, state-supplied control, and
retention probes remain contextual primary results. Do not rerun them and
present them as independent secondary evidence. If a secondary test seed does
not have an independent paired-assessment result, mark that comparison as
not run rather than relabeling a seed-0 result.

Show every declared condition. Do not select the best anchor, readout seed,
graph, or reference as the benchmark result. No new success threshold is
introduced by this addendum. These repeated conditions are not additional
independent biological worlds.

## Provenance and failure policy

Place the operator under `scripts/`, outside the frozen primary source
universe, and pin its own source, protocol, and imported scientific sources.
Bind each result to the primary report, its configuration and fingerprint,
world and truth manifests, observation validation, model/export manifests,
embeddings, original readouts, and fixed common-cell IDs. Verify those inputs
before calculating or reusing any stage.

The secondary receipt also pins external test dependencies, including the
HBE approximation implementation. Pins first recorded by this helper attest
to the secondary environment at freeze time, not retrospectively to an
unrecorded dependency's bytes during an earlier primary evaluation.

Use separate output directories and immutable per-stage receipts. An existing
cache with changed inputs must be refused. Numerical failures remain explicit
failures with their errors; they are never empty graphs. No fallback test,
row reordering to repair a failure, model refit, or replacement world is
permitted. Any explicit continuation of an interrupted secondary calculation
must verify the same inputs and retain the interruption record.

Execution remains serial with the primary suite: at most one GPU or
large-memory stage at a time. An independent reviewer must check the helper,
lightweight tests, and final evidence. A supplemental review receipt must be
included in the phase-E assessment-release review; the old primary schedule
must not be silently rewritten to claim these checks were already included.

## Full-scope completion requirements

The completion audit found that finishing the simple study and two extended
smoke worlds is not completion of the authorized extended benchmark. The
frozen [suite schedule](runs/suite/schedule_v1.json) contains 170 stages and
178 unique model fits, including the simple study, development controls,
exact-fit ambiguity-twin reuse, manual evidence review, and the 21-world
assessment block.

Remaining deliverables include all declared development evidence, these
secondary diagnostics, a reviewed assessment lock, every assessment outcome,
and a world-level synthesis. The synthesis must separate primary from
secondary results, development from assessment, graphical targets, methods,
count regimes, calibration failures, and numerical failures. Undefined
precision is not zero. Cells, edges, integration seeds, and count regimes
must not be counted as independent worlds.

The serial operator stops on an operational failure; it cannot invent a
successful calibration or model manifest to satisfy the schedule. If such a
failure prevents ordinary downstream stages, preserve the failed artifact
and attempt log. Explicit terminal-failure accounting must be reviewed in a
versioned addendum before assessment release. It must neither replace the
world nor erase the missing outcome. That accounting mechanism is not yet
implemented by this document.

## Review record

- A read-only completion audit identified the missing anchor execution,
  separate readout repeats, final world-level synthesis, and terminal-failure
  handling gap before assessment release.
- The primary scientific source hashes and `schedule_v1.json` remain unchanged.
- Independent secondary-helper review passed at source SHA-256
  `9f61a2a15dc0f86fc4aa81e4c746b439baaa76911d57177a0d6494d8bb5b6429`;
  the main agent independently reran all 23 lightweight tests successfully.
- At protocol approval, no secondary CI calculation had run. Input-freeze and
  execution receipts are separate artifacts; final synthesis review remains
  a separate completion requirement.
