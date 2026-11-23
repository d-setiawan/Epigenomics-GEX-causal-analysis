# Integration recoverability: implementation record

_Updated 2026-11-01. The original extended benchmark remains incomplete._

---

## Scope and completion rule

The current work is the [unpaired-integration proof](../../../project_resources/research/theory/07_unpaired_integration_nonidentifiability.md),
with independent reviews and exact algebra checks linked in its
[research record](../../../project_resources/research/integration_identifiability_2026_11_01/README.md).
This is a theory pivot, with no new fits or generated datasets. It does not
complete or restart the unfinished frozen benchmark.

The latest experimental work is the [scConfluence/scMTR-seq pilot](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md),
following the completed [count-input comparison](COUNT_INPUT_FOLLOWUP.md) and
[integration-validation supplement](INTEGRATION_VALIDATION.md). The new pilot
adds causal reliability readouts on two cached worlds and a separately authorized
real-data holdout. Each study has its own protocol and outputs, preserving the
original frozen scientific sources and baseline work.

The earlier goal authorized both the [simple study](README.md) and the
[extended design](../../../project_resources/research/integration_recoverability_extended_plan.md),
using scGLUE and a paper-faithful scMRDR implementation. The earlier
documentation-only and deferred labels describe the planning stage, not the
current authorization. Completion requires executed simulations, validated
model adapters, and results with explicit recovery targets and failures.

Only the original RNA, H3K27ac and H3K4me3 summaries calibrate the cached
simulations. The November 1 authorization adds a separate scMTR-seq measurement
arm; it does not recalibrate or replace those worlds. Pairing and day labels
remain evaluator-only, and the learner uses the same three modalities. Existing
CA-BOSS results and the unrelated repository migration remain untouched.
Frozen worlds must be reused rather than replaced after failure.

## Latest experimental follow-up: 2026-11-01

The [fixed pilot record](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)
owns the six-fit execution plan, isolated runtime, corrected native scConfluence
objective and independent reviews. Both cached worlds have completed fixed
paired, random, true-state and existing scGLUE causal controls. The real cohort
has completed paired/random controls, independently replayed from original
source-cell counts. All six GPU fits and their held-out exports completed, as did
all primary and separately declared exploratory-null readouts. Independent
reviews reproduce state costs/transfer, empirical pair retrieval/day-confusion
scores and every empirical control/null graph. The
[causal report](runs/integration_validation/20261101_causal_reliability_v1/RESULTS.md)
records weak recovery and a demonstrated sparse-null CI calibration problem;
the [state report](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/state_readout.md)
records integration seed instability. These new fits are separate from the
original suite's 178-fit denominator. No calibrated replacement CI method or
reliable causal model is claimed.

## Earlier follow-up: 2026-10-31

- Four fresh scGLUE fits are complete: PCA/LSI and full-feature inputs on cached
  informative `linear_chain_0` and `constrained_collider_0`, seed 0,
  `lam_align=0.2`. No worlds, measured datasets or fitting inputs were added.
- Full-feature input modestly improves state matching in both worlds. The
  [complete results](COUNT_INPUT_FOLLOWUP.md#observed-results) retain all decoder
  directions, geometry changes and limitations; no new causal graph was run.
- The GPU and CUDA libraries are present. Fits ran successfully with the verified
  process-local CUDA 13.0.96 override, preserving the shared environment.
- All four fits exited successfully. The original scoring calls rejected
  scGLUE's native unset→`0` GPU-mask mutation, so their controller exit 1 is
  retained. A declared analysis-only correction produced both final reports
  without refitting or altering the original protocol/receipts.
- The 35 focused tests pass. Independent reviews reproduce both worlds' state
  costs and decoder scores exactly and verify configurations, pools and exports.
  The four-fit follow-up is complete; it does not establish reliable causal
  recovery or change the original suite's 118/178 completed-fit count.
- Published alternative methods and existing-data viability were reviewed, with
  a verified public-dataset shortlist. No alternative package was installed and
  no new count matrices or sequencing datasets were downloaded.

## Earlier execution snapshot: 2026-10-26

- The original simple/extended suite has 118 completed model manifests, including
  all 94 simple/development fits and 24 assessment fits. It is not complete.
- The previous phase-F execution stopped at the saved count-calibration failure
  in `linear_collider_3`. It is not still running. Both regimes of five earlier
  assessment worlds completed. The three chain worlds and both collider worlds
  now have independent scoped reviews; the
  [collider follow-up](RESULTS.md#linear-colliders-1-and-2-follow-up-review-on-2026-10-25)
  preserves the failed count readout separately from completed graphs.
- Continued generation of the predeclared `constrained_chain_1` world passed
  calibration. Its realistic scGLUE stage failed at native CUDA discovery; its
  informative stage then failed during the first training iteration despite a
  process-only runtime override. Both attempts and partial outputs are retained.
- The new integration-only protocol, fit bridge, metrics and CLI are implemented.
  The [combined final-source check](runs/integration_validation/20261025_v1/focused_tests_20261026_final_sources.json)
  passes 75 tests and 25 subtests, covering 53 validation tests and 22 baseline
  retry-helper tests. All 17 frozen scientific source hashes still match.
- The cached informative linear-chain reference audit is complete and independently
  reproduced. It supports state matching while exposing a probe-sensitive
  cross-assay decoder failure; it is not evidence of recovered causal structure.
- All sixteen new selection fits on `linear_chain_0` and
  `constrained_collider_0` are complete. The independently checked
  [two-world rule](INTEGRATION_VALIDATION.md#frozen-two-world-selection) selects
  scGLUE `lam_align=0.2` and scMRDR's recorded baseline. The selected-settings
  file and exact sixteen-job confirmation plan stayed frozen; all confirmation
  fits completed on two other cached development worlds. These new fits
  are separate from the baseline suite's 178-fit denominator.
- Both count regimes of `constrained_chain_0` now have all four selected-setting
  fits and scoped independent native/numerical reviews. The
  [confirmation results](INTEGRATION_VALIDATION.md#informative-constrained-chain-confirmation)
  retain the disagreement between scMRDR's low global-matching state error
  and negative RNA-to-mark decoder scores under informative counts.
  Realistic `unrestricted_chain_0` also has four completed fits and native-fit
  reviews. Its scMRDR seeds give gaps 0.1246 and 1.4446; independent numerical
  review reproduces the latter's state-alignment failure. Both informative
  scGLUE seeds on this world have also completed with independent reviews,
  giving gaps 0.1438 and 0.1466. Informative scMRDR seed 0 has also completed
  with gap 0.0269; its native and numerical reviews passed. The final scMRDR
  seed completed with gap 0.0276 and passed native/numerical review. All sixteen
  confirmation fits are complete and independently reviewed. The
  [final matrix audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/final_matrix_integrity_interpretation_2026_10_26.md)
  passes scoped checks of all 32 fits, 288 public exports and frozen bindings.
  All 48 primary state-cost rows have independent reproduction; independent
  secondary probe/hubness array checks cover eight of sixteen reports.
- The first matrix controller terminated with exit 143 during unrestricted-chain
  realistic scGLUE seed 1, after nine confirmation fits completed. The
  [reviewed recovery](INTEGRATION_VALIDATION.md#controller-interruption-and-preserved-retry)
  preserved the sole partial fit and resumed the unchanged controller.
  All 25 completed fits were reused; existing reports and selection passed
  exact-equality checks. The same-seed retry completed and passed scoped native
  and numerical reviews; the resumed matrix exited with code 0 after all 32
  planned fits. The signal's
  cause remains unknown. The post-reentry review records that pre-action
  process scans were sandbox-scoped; subsequent execution guards use host
  process inspection.

The verified process-only CUDA library supports complete fits for both methods;
the new scGLUE fit also used synchronous CUDA reporting. The cause of the
earlier baseline failure remains unresolved. No shared
environment or frozen model implementation was changed. The supplement records
runtime provenance and the limitations of validating integration without paired
data. Later sections retain dated historical snapshots.

An independent [baseline retry preparation](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_retry_preparation_readonly_2026_10_25.md)
verified that neither failed `constrained_chain_1` fit has a recoverable
checkpoint. It records a bounded operational retry with unchanged settings,
archived partial outputs and retained failed receipts. The
[two-stage retry helper](scripts/retry_constrained_chain_baseline.py) is now
implemented with a read-only default preview. Its
[independent review](../../../project_resources/research/integration_validation_2026_10_25/reviews/retry_constrained_chain_baseline_independent_review_2026_10_25.json)
found no blocker, and a separate
[root test run](../../../project_resources/research/integration_validation_2026_10_25/reviews/retry_constrained_chain_baseline_root_tests_2026_10_26.json)
passed all 22 focused checks. The real preview verifies the two original
commands and partial inventories. After matrix completion, independent review
and host guards, the helper
[started](runs/integration_validation/20261025_v1/baseline_retry_observed_start_20261026.json)
with the unchanged original commands. Both old partial outputs are archived
with their failed evidence retained. Independent archive/start and completed
realistic scGLUE audits passed 125 and 181 scoped checks, respectively.
Both realistic-count models and recovery evaluation have completed. Its
stage receipt records exit code 0 and complete scheduled artifacts; native,
stage-binding and scientific report reviews passed. Both informative-count
models and their evaluation have also completed. All four native-fit reviews
passed. The [final stage/helper review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_baseline_retry_final_ledger_and_informative_stage_binding_audit_2026_10_26.json)
passes 330 scoped checks, and the [informative scientific review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_informative_frozen_baseline_json_audit.md)
preserves the weak recovery outcomes. The helper's [terminal result](runs/suite/constrained_chain_1_retry_20261025_v1/result.json)
records `completed_two_retries`, with both child stages exiting 0. Both
evaluation reports have status `completed`, with no failed integrations or
diagnostics. The [terminal host observation](runs/integration_validation/20261025_v1/baseline_retry_terminal_host_observation_20261026.json)
found no remaining study workers or held suite/GPU training locks. No broader
assessment execution was started.
The unrelated stopped scGPT process was left untouched. Successful retries
do not establish the cause of the earlier CUDA failures. The
[current-objective completion review](../../../project_resources/research/integration_validation_2026_10_25/reviews/objective_completion_final_audit_2026_10_26.md)
accounts for the validation matrix, verified research, conditional theory,
two-stage continuation and their limits. The separate original benchmark's
remaining assessment and final synthesis are still unfinished.

## Frozen-suite snapshot on 2026-10-04

| Component | Current evidence |
| --- | --- |
| Count calibration and simulator | Implemented; 19 component tests passed |
| Matching, discovery, and scoring | Implemented; 33 component tests passed in `scrna_r` |
| Independent checks | Simulator/evaluator cross-review, 12-primary-report audit, 18-report chain-control audit, and production scMRDR fidelity review completed; findings and scope recorded |
| Production simple population | Generated once; integrity, focal, and panel checks passed |
| scGLUE frozen preprocessing and adapter | All simple/development fits, including seed/guidance controls, complete; original seed-0 post-training log-reader failure and audited recovery preserved |
| Paper-faithful scMRDR | All simple/development fits and controls complete; native checkpoints and configuration receipts retained |
| Extended 30-specification bank | Specifications and mechanisms frozen; all nine development specifications validated; independently verified assessment release passed and serial phase F is running |
| Recovery results | All six causal development worlds complete under both count regimes, plus simple three-seed study; five of 24 learned primary seed-0 count readouts failed; both conditions of all three linear-chain assessment worlds completed and independently reviewed |
| Shared-state-only control | All primary seeds, input/guidance ablations, and prior baselines in both regimes reviewed; real state-induced associations but false activity ancestry claims even in paired references; no consistent ablation improvement |
| Secondary CI diagnostics | Linear KCI/Fisher-Z and constrained KCI diagnostics complete and independently compared; no primary result replaced |
| Serial suite operator | Implemented outside frozen scientific sources; 19 safety tests and independent review passed; phase D completed all 86 executed commands with process exit zero and preserved typed scientific failures |
| Completion-audit additions | Both frozen alternative-anchor/readout runs completed all 80 graphs without failures; independent artifact and fresh development-only synthesis reviews passed |

At 23:16 UTC on 2026-10-04, 106 of 178 planned model manifests were complete:
all 94 simple/development fits and twelve assessment fits. The remaining 72 fits
are in the now-running assessment block. A completed fit manifest is not a
completed discovery report or a benchmark reliability estimate.
All nine extended
development specifications have been generated, and all six causal development
worlds have completed both count regimes with both seed-0 methods. The serial
operator has completed phase D, including constrained KCI. Both supplementary
anchor/readout diagnostics and a fresh development-only synthesis have run;
their independent reviews passed. Phase E created the exact assessment lock
at 21:39 UTC, and independent post-execution verification passed. The serial
phase-F operator started at 21:45 UTC, initially generating the already-frozen
`linear_chain_1` specification. All three linear-chain assessment worlds are
now evaluated under both regimes; `linear_collider_1` is in progress.
Earlier fit counts below are
dated snapshots of this same continuing execution.

A fresh read-only audit verified 12 causal development primary reports against
their saved source/world identities and recomputed graph adjacency counts.
Nineteen learned count graphs completed, five failed, and all 24 activity
graphs completed. No completed learned count graph returns a fully directed
cross-assay edge. The [development results table](RESULTS.md) preserves empty
graphs, pre-CI constant-column failures, and RCIT numerical failures separately.
An independent reviewer checked all 12 reports and the new table, passing
134 identity and score checks with no substantive correction. This review
covers the causal primary reports. A separate read-only review passed for
18 chain-world seed/guidance evaluations and 18 prior-only baselines, including
fit/export bindings, source pins, identical within-world/regime pools, and exact
seed-0 reproduction. Its 54 learned count readouts include 15 failures and
eight returned graphs with no cross-assay predictions. The complete table and
review digest are in [RESULTS.md](RESULTS.md). Supplementary/synthesis reviews
passed; the full-suite assessment and final world-level synthesis remain pending.

The completed shared-state-only primary control has also been independently
reviewed. Its six realistic primary learned activity graphs return unsupported
cross-assay arrows, but the genuinely paired activity reference also returns
four false arrows. This does not isolate an integration-specific orientation
error. Returned count graphs have no fully directed cross-assay arrows;
their exact directed-set agreement with an empty direct target is not recovery.
The [shared-state results](RESULTS.md#shared-state-only-control-completed-primary-readouts)
separate these targets and preserve the scMRDR seed-2 count numerical failure.
Both realistic ablations and their three prior baselines passed a further
176 bounded JSON checks. They preserve the same 809-cell pools; removing focal
inputs or shuffling guidance does not consistently eliminate false activity
ancestry. The informative shared-state integrations and their reviews have also
completed.
All informative primary seeds also passed a further 316-check bounded audit.
Both seed-2 count readouts have no cross-assay predictions and zero recall
against the state-induced association target; agreement with the empty direct
target is not causal recovery. Their activity readouts still assert one or two
false cross-assay directions, so the earlier absence of such arrows is not
stable across seeds.
The informative background-only report passed 52 further bounded checks:
scGLUE returns no cross-assay count edges and one false cross-assay activity direction;
scMRDR returns one of 12 count associations and no fully directed activity
edges. Removing focal inputs has not established reliable recovery.
The informative shuffled-guidance report and three prior baselines passed
76 further bounded checks. Shuffled scGLUE returns no cross-assay count edges
but one false cross-assay activity direction. All three prior baselines abstain because
the scored features are excluded, not because direct null structure was learned.

The combined benchmark and paper-model core suite passed 231 tests after
installing the missing log reader. These include actual wrapper fits, strict
cache receipts, typed assessment evidence, ambiguity-twin checks, prior-only
controls, and dependency-backed history/export tests. They are software checks,
not independent biological replicates or evidence of causal recoverability.

The [results report](RESULTS.md) includes both methods on common eligible
pools. All six simple-study integrations succeeded; the seed-2 scGLUE count
readout failed its RCIT numerical computation with "Eigenvalues did not
converge." It is recorded as failed, not as an empty graph. All six learned
activity matchings returned a triangle rather than the stipulated chain.
The combined evidence is
`runs/simple/realistic/evaluation_suite_all_seeds/report.json`.
Both phase-C model-execution gates passed. These gates establish
that the native adapters run, not that their graphs are reliable. A separate
[sparse-CI audit](SPARSE_CI_AUDIT.md) documents a reproduced false association
and why nominal test significance must not be treated as calibrated error
control in these sparse data. The code knowledge graph was updated with AST
extraction.

The [input-retention audit](INPUT_RETENTION_AUDIT.md) checks the two informative
extended worlds. All nine scored features reach each model's selected inputs;
none of their scMRDR gene mappings pools multiple original columns. The fixed
same-assay residual probes already fail to outperform a constant on native
PCA/LSI coordinates. This rules out simple feature omission as the explanation
in these runs, but does not isolate measurement noise, compression, learning
objectives, or probe limitations.

The secondary CI stages reuse fixed fits and assignments. Their predeclared
`--skip-probes` setting produces a partial-evaluation status with
`probes/not_run` and `preprocessing_probes/not_run`; these omissions are not
model or graph-computation failures. Fisher-Z remains a linear-control
sensitivity diagnostic: mixture-distributed state and nonlinear sparse count
observations do not become jointly Gaussian just because activity mechanisms
are linear. The 21-world assessment block is now running under the reviewed
development evidence lock; its full results remain pending.
The completed linear KCI diagnostic has no graph-computation failures. Its
independent paired counts recover three of 12 cross-assay measurement-PAG
adjacencies; scGLUE and scMRDR matching recover one and two, respectively,
without false count adjacencies in this particular comparison. These few
calls do not establish reliable recovery or calibrated error control. See
the secondary-test comparison in [RESULTS.md](RESULTS.md).

Constrained KCI also completed all 13 discovery graphs; its report's failure
status is solely the two planned probe skips. scMRDR's two count-level
cross-assay pairs exactly match the nonempty biological direct-pair target,
but their directions are unresolved and count-PAG recall is only 2/12.
This predeclared secondary stress-test outcome is retained without changing
the primary RCIT backend or claiming reliable causal recovery.

## Preliminary result: simple scMRDR, seed 0 only

The paired activity reference recovers the two-edge chain skeleton exactly.
Both scMRDR matching and true-state matching produce a triangle among the three
core activities, adding the non-edge between H3K27ac and RNA. This is consistent
with preservation of state-dependent means without preservation of the original
same-cell residual coupling; it is not a complete comparison of integration
methods.

Sparse counts are a separate bottleneck. Among 2,000 paired-reference cells,
the core RNA, H3K27ac, and H3K4me3 variables are detected in 67, 12, and 55 cells;
their cross-assay co-detections number only 1, 3, and 1. Paired counts recover
none of the three cross-assay adjacencies in the count-measurement PAG and call
two false cross-assay edges. scMRDR-matched counts call four cross-assay edges:
one true and three false (precision 0.25, recall 0.33). Random matching also
calls four cross-assay edges, warning that nominal RCIT alpha is not calibrated
false-adjacency control in this sparse setting.

These count metrics concern the measurement PAG, including technical exposure
causes, not the two-edge activity DAG. Successful execution and passed marginal
count calibration do not establish statistical power or reliable recovery.
Evidence is saved in
`runs/simple/realistic/evaluation_preliminary_scmrdr_seed0/report.json` and
`diagnostic_summary.json`. This historical single-fit subsection is superseded
by the combined three-seed report described above and in [RESULTS.md](RESULTS.md).

## Calibration and provenance

Empirical profiles are cached in
`runs/calibration/empirical_count_profiles.json`, derived from the existing
`scglue_joint_v1` preprocessing and reconstruction panels. The calibration hash is
`8fefacaf48416454daa905bafbcb3b737e80e3f99bf36816ed480133bf41ae57`.
The deterministically selected focal profiles have detection rates near
2.46% for RNA, 0.52% for H3K27ac, and 1.76% for H3K4me3. Passing focal moment
checks does not substitute for the generated-panel realism audit. The cached
simple population has median detected-feature counts 3,242 / 110 / 360
(RNA / H3K27ac / H3K4me3), compared with empirical targets approximately
3,165 / 94 / 363. All were within the predeclared 35% panel-quantile tolerance;
the H3K27ac deviation is about 17%, not an exact match. No world was regenerated.
Its manifest hash is
`9cbccc70061c7f3daaa93a3291ef06367166c2218a49a4ab005c9d9f19ef5559`.

Native scGLUE preprocessing retains all 2,000 cells per assay. RNA contributes
3,000 selected genes; each mark has 15,000 reconstruction bins, with 29,982
and 30,000 nonzero encoder bins respectively. Neither RNA scored feature
survived HVG selection. Histone candidates have mixed encoder/reconstruction
retention; no scored feature was force-selected. The preprocessor verifies
the native held-out LSI extension numerically and saves every fitted transform.

The installed integration environment currently contains scGLUE 0.2.3,
Scanpy 1.10.4, AnnData 0.9.2, PyTorch 2.10.0, scikit-learn 1.8.0, and SciPy
1.14.1. The missing `scikit-misc==0.5.2` dependency was installed without
changing other packages, preserving the selected Seurat-v3 HVG procedure.
There is an 8 GB NVIDIA GPU and approximately 11.7 GiB total system RAM.
GPU access requires execution outside the filesystem sandbox.
Training will be serialized; these observations do not promise a runtime.
The focused [environment record](../../../envs/requirements-integration-recoverability-2026-10-04.txt)
also includes the subsequently installed TensorBoard reader. Native scGLUE
completed both phases with finite histories through epoch 38, restoring native
checkpoint 37 in each phase, before the missing-reader exception. Recovery
must use the saved native final snapshot with exact checkpoint/feature/source
verification; it must not retrain or choose a new checkpoint criterion.

## Method changes that require explicit labels

The saved plans originally specified the local scMRDR-inspired model. The
current goal explicitly replaces that comparator with the paper-faithful
model. Its input must use an annotation-aligned gene universe with availability
masks; unrelated assay-specific bin encoders are not the paper's architecture.
The audit distinguishes equations in the supplied paper from the released
authors' code, pinned at commit
`7280e64ae19f7391c6f5adf00d9d353f79e96603`.[^1]

Production fidelity review passed for all 12 causal primary fits and 20
completed nonshared control/seed fits. It checked recorded configuration,
source identities, complete finite training histories, best-checkpoint
selection, input isolation, and separate shared/private exports. Its
JSON/source/log scope cross-checks recorded binary identities without claiming
an independent binary rehash. The [adapter README](../../empirical/integration/methods/scmrdr/paper_faithful/README.md)
records this evidence and the distinction between paper consistency and
reproduction of the paper's original experiments. Five primary fits reached
the epoch cap; that is not evidence of convergence or causal recovery.

The causal test is explicitly pinned to RCIT (not its RCoT option), with the
`hbe` approximation, 100 conditioning and 5 feature random Fourier components,
and per-query seeds. The explicit approximation avoids the installed backend's
implicit approximation fallback. No Fisher-Z fallback or expression-sink
prior is permitted. Counts receive log1p once; activities remain unlogged.

The frozen primary scGLUE arm already implements focal-guidance masking:
every scored feature is excluded from cross-feature guidance, while remaining
eligible for native feature selection and count reconstruction. Required
self-loops are retained. The `background_only` arm instead removes scored
features before normalization, selection, dimensional reduction, reconstruction,
guidance, and gene scoring. Shuffled guidance permutes the remaining background
links. The suite records this distinction in `protocol.guidance_masking`; the
proposed focal-guidance-mask check does not require an additional fit under
the frozen protocol. A guidance-only baseline can correctly abstain on the
scored panel because its scored cross-feature links are already held out.

## Remaining work and interpretation

The adapters and end-to-end contracts are implemented and tested. The simple
three-seed first pass and all declared development controls have run. The
reviewed development evidence lock is saved and the assessment block is
running. Full assessment execution, final synthesis, and independent close-out
remain required. Preserve sparse-feature, training, and statistical failures
in results.
Count-proxy graphs are scored against measurement truth, not automatically
against the activity DAG. State alignment is not evidence of recovered
cell-specific residual coupling. Unsupported hyperedge or ambiguity-set outputs
remain not applicable.

Before assessment release, review all 106 simple/C/D prerequisite stages and
the exact 82 development evidence files, plus both sealed supplementary
diagnostics and a fresh development-only synthesis. A separate immutable
review receipt will bind those hashes and review scopes. The existing phase-E
command must keep its exact 82-file list; its successful, operator-reported
attempt will link the supplemental receipt by absolute path and SHA-256 in
the note. Independent procedure review found this compatible with the frozen
schema, but the note is documentary: the old gate does not enforce supplemental
review. The main agent must verify the linked evidence, actual exit/log,
resulting lock, and recorded attempt before explicitly launching phase F.
No terminal calibration/training failure can be waived by a generic review note.

The immutable [compiled development review](runs/suite/reviews/development_release_20261004T2135Z.json)
binds the exact report/artifact maps, supplementary completion chains, fresh
synthesis, source pins, and review scopes. Its SHA-256 is
`7558a0afcdad73947232fad7accedbfbac4e1eb056854b20dcacdd5f374a5d25`.
The [independent compiled-receipt check](runs/suite/reviews/compiled_receipt_check_20261004T2135Z.json)
passed and binds two wording clarifications: paired anchors change the finite
cell panel, not the underlying biological population; supplementary conditions
were frozen after primary outcomes were known. Its SHA-256 is
`08771a78aa86caca961db4fb3b8a0caf30cd748d1c4cced009d7c0f8da28bae9`.

The exact 89-token phase-E command executed once under the existing global
lock, after a locked preflight rechecked the pinned evidence. The actual child
and wrapper both exited zero at 21:39 UTC. The
[handoff journal](runs/suite/manual_release/20261004T213929Z/result.json)
links the preserved transcript, actual status markers, and append-only attempt
receipt. The lock SHA-256 is
`2f69a3f1f2c4ab0da9d116a3901120f0cfa32e8a7133fcec452690049358d0cf`.
This is configuration/evidence freezing, not evidence of reliable recovery.
The [independent post-E check](runs/suite/manual_release/20261004T213929Z/post_E_verification.json)
passed: exact lock contents, 106 terminal prerequisites, 286 artifact hashes,
actual execution markers/terminal status, and every handoff link reconciled.
The legacy note linkage remains documentary, not an automated supplemental gate.
No competing benchmark work may run alongside phase F.

The explicitly launched [phase-F operator session](runs/suite/executions/1791150340834241831_ca6b4f5ef2e24e6f901f128936044e35/session.json)
uses the unchanged schedule and source hashes. It serializes the 21 frozen
assessment specifications and 84 model fits, preserves failures, and does not
automatically retry or replace worlds. The active goal is not complete until
that block, final synthesis, and independent result reviews finish.

The first locked assessment condition, `linear_chain_1` / realistic counts,
completed with both fits successful and the sole scMRDR count numerical failure
preserved. The independent metadata/source audit passed. Its 812-cell matched
readouts yield scGLUE count cross TP/FP 0/3; scMRDR counts are unavailable.
Both learned activity readouts recover four of seven projected cross-assay
adjacencies, but neither recovers a biological direct pair and both assert
unsupported ancestry. The [partial assessment table](RESULTS.md#first-locked-assessment-condition-linear-chain-1-realistic-counts)
includes all references and failure semantics. This is not a reliability
estimate. The informative condition also completed and passed independent
metadata/source review: scGLUE count cross TP/FP 1/2, scMRDR 0/0 with undefined
cross precision, and activity TP/FP 4/1 versus 5/0. All 13 graphs completed;
no count graph has a fully directed cross-assay arrow. scMRDR activities recover
one biological adjacency without a fully directed cross-assay arrow; this is
not exact direct-structure recovery. The two conditions share one world and
are not independent replications. The operator then moved to
`linear_chain_2`; [the informative table](RESULTS.md#same-locked-world-informative-counts)
records all references and limits.

The realistic `linear_chain_2` evaluation also completed and passed independent
source/metadata/score review. Its 771-cell common panel gives scMRDR count cross
TP/FP 1/0; scGLUE, random, and independent paired count readouts failed
numerically and remain unavailable. Activities give scGLUE TP/FP 5/0 and
scMRDR 6/0 against the projection target. scMRDR retains both biological chain
adjacencies with four additional cross-assay pairs and no fully directed
cross-assay arrows; its count true positive is not a biological direct pair.
The [second realistic table](RESULTS.md#second-locked-assessment-world-linear-chain-2-realistic-counts)
preserves all references, failure semantics, and review limits. The informative
condition also completed and passed independent review: scGLUE count cross
TP/FP 1/0, scMRDR 2/0, and both activity readouts 6/0. Both learned activity
graphs retain the two chain adjacencies plus four shared-state cross pairs.
scMRDR counts recover one biological adjacency, but no compared count/activity
graph returns a fully directed cross-assay arrow or an exact biological pair
set. The [informative table](RESULTS.md#linear-chain-2-informative-counts) records
all references and the differing panels. The operator is now processing
`linear_chain_3`; no settings were changed.

The realistic `linear_chain_3` condition completed and passed independent
read-only review. Its 791-cell matched panel gives scMRDR count cross TP/FP
2/1; scGLUE counts fail numerically and paired-anchor counts fail before CI
because `H3K27ac_sentinel` is constant. Learned activities give scGLUE TP/FP
3/0 and scMRDR 4/0, but neither retains either biological chain adjacency.
Both have zero fully directed cross-assay arrows. Separately, paired activities
with true state supplied to PC recover both correctly directed core edges,
with overall TP/FP/FN 9/0/1 against their ten-edge oracle. This assisted control
is not state matching or evidence of integration recovery. The
[third realistic table](RESULTS.md#third-locked-assessment-world-linear-chain-3-realistic-counts)
preserves the two different failures, all references, and review scope.

The informative `linear_chain_3` condition also completed and passed independent
read-only review. All 13 graphs completed. Both learned count graphs have
empty cross-assay predictions and undefined cross precision, but retain the
within-assay `H3K4me3_core–H3K4me3_decoy` edge. Both learned activity graphs give
cross TP/FP 5/0; scGLUE retains neither biological adjacency and scMRDR retains
only the H3K4me3–H3K27ac core pair. Neither learned graph returns a fully
directed cross-assay arrow. True-state matching reverses a core edge, while
the separate paired-state PC diagnostic again correctly directs both core
edges. The [informative table](RESULTS.md#linear-chain-3-informative-counts)
retains the distinct targets, oracle scopes, and changed panels. All six
linear-chain assessment conditions are now reviewed; the operator is processing
`linear_collider_1` under unchanged settings.

A separate completion-requirements review found no missing normal-path
experiment beyond the remaining assessment block. Final close-out must account
for all 170 scheduled stages and 178 unique fits, including 21 assessment
specifications, 42 pipelines, and 84 assessment fits. Both supplementary
diagnostics remain separately inventoried. The assessment-inclusive synthesis
and independent scientific interpretation are additional deliverables, not
operator stages. A complete synthesis output manifest certifies an export,
not full benchmark completion. Failure accounting and all declared diagnostics
must be checked against the planned denominators before the goal is complete.

An independent paper-fidelity check passed for the first three completed
assessment scMRDR fits: both `linear_chain_1` regimes and realistic
`linear_chain_2`. Seven implementation/source identities, all fit configurations,
538 saved epochs, 15 cohort receipts, and 27 public export identity entries
agree with the approved development implementation. Best/final epochs are
194/200, 172/182, and 146/156 respectively. The first stopped at the epoch cap;
the others stopped after ten non-improving validation epochs. The capped fit
is not evidence of convergence. The review checked source, metadata, and logs,
not checkpoint/H5AD/NPZ binary contents or exact paper-benchmark reproduction.
The recorded exclusion of one H3K27ac assessment cell in realistic
`linear_chain_2` is post-fit QC, not a change to training inputs.

The prerequisite audit passed within its JSON/source/log scope: 106 terminal
dependencies, 82 typed evidence files, 286 declared artifacts, and 94 completed
model/public manifests (51 scGLUE, 43 scMRDR). The 810 public path-to-digest
entries comprise six simple fits with three train exports each and 88 extended
fits with nine train/validation/discovery exports each; they are not unique
digest values. The audit reconciled 86 operator launches and 564 lower-level
command/result/log pairs. Twelve pre-existing executable outcomes have no
suite-level attempt receipt, and eight unreceipted stages are declarative N/A.
It did not independently rehash original binary datasets or checkpoints.

The simple scMRDR seed-0 public receipt retains its historical writer
`ad37e95fd7d7dd5f4a54fc13f8b8c5714dc6e6df3ee54f2df4cf0e19cdeb1c9e`,
with its documented first post-export verification and earlier unrecorded
interval. Model/core/export bindings agree; the historical receipt was not
rewritten to imply stronger provenance. Fifty repeated count-readout failure
occurrences remain null-edge failures (24 eigendecomposition, 26 constant
columns); six planned probe-skip entries remain in three secondary-CI reports.
These occurrences are not independent trials.

At 20:31 UTC, the exact phase-E definition was rechecked: 89 argument tokens,
106 prerequisites, and 82 reports. A nonblocking `/usr/bin/flock` probe against
`runs/suite/executor.lock` returned its declared conflict code 73 while the
development operator was live. This verifies manual-command interoperability
with the same advisory lock, not release readiness. The lock file was not
removed and no assessment command was executed.

The concrete manual handoff received a further independent safety review.
An execution log preserved from tool output must be labeled an
operator-preserved transcript, not independently established byte-exact
stdout/stderr. Preserve every output chunk, tool/session identifiers, terminal
exit result, and any truncation or interruption limitation; never reconstruct
missing text or infer a missing exit code. Separate lock/preflight refusals
(`phase_e_launched=false`, no child exit code) from actual child execution.
A refusal belongs in the operational journal, not a failed scientific-stage
receipt; the conflict code alone cannot prove whether a child ran.

If the command lock is released before transcript persistence and attempt
recording, no assessment or competing benchmark work may start during that
gap. Final verification remains mandatory. An interruption after creating the
assessment lock leaves an incomplete handoff: reconcile that existing lock and
execution evidence rather than rerunning phase E to manufacture a missing log.
These safeguards do not make the legacy supplemental linkage an automated gate.

The 09:00 UTC execution snapshot counts 12 published model fits out of 178 scheduled,
with three of the 30 extended specifications generated. The remaining 166 fits
are development controls and assessment, not a requirement fulfilled by the
smoke runs. The [secondary diagnostic addendum](SECONDARY_DIAGNOSTICS_PROTOCOL.md)
specifies the previously unwired alternative-anchor/readout checks without
changing primary results. Final reporting must include paired method
differences across worlds, explicit unavailable outcomes, and justified
world-level uncertainty. Any terminal calibration/training failure needs
audited failure accounting before assessment release; a missing model manifest
must not be replaced by a successful-looking stub.

The first realistic extended evaluation uses 788 common cells per assay.
Both integration fits succeeded, but the common H3K27ac sentinel column is
constant. The frozen discovery precheck therefore rejects all four unpaired
count readouts (random, true-state, scGLUE, and scMRDR) before any CI query.
These are unavailable graph results, not empty graphs, and are distinct from
the earlier simple-study RCIT eigensolver failure. Paired-reference count
graphs and all activity readouts completed. Subsequent development worlds and
controls use the unchanged frozen features, thresholds, and count calibration.

## Execution entry points

Run from the repository root in the `scrna_r` environment. The runner records
per-stage command logs, serializes GPU stages, verifies frozen-bank hashes,
refuses changed/incomplete caches, and requires an assessment lock before
assessment-world training.

```bash
python pipelines/simulation/integration_recoverability/run_benchmark.py generate \
  --world-id linear_chain_0
python pipelines/simulation/integration_recoverability/run_benchmark.py run-world \
  --world-id simple_saturating_chain --regime realistic --seeds 0 1 2
```

Generation is a separate explicit action; `run-world` does not silently create
replacement data. A failed count-calibration gate stops ordinary runs. Any
explicit diagnostic continuation carries the failed calibration in its scope
record and evaluator report. Ambiguity twins require shared frozen fits, not
two separately trained models. Their dedicated execution path has now run on
the production pair under both count regimes; the saved comparison reports
are complete and independent production review passed. The reviewer verified
public byte identity, checkpoint/embedding reuse, matching and graph identity,
and differing hidden discovery-cohort residual truth. The original realistic
private paired-count failure remains recorded. The comparison demonstrates
observational ambiguity, and supplies no calibrated ambiguity-set output.

`suite_plan.py` freezes and inspects the full execution schedule without
launching work. `runs/suite/schedule_v1.json` records 170 stages and 178 unique
fits, with phase-C reuse and exact-twin reuse; all 82 development evidence
artifacts must be pinned before assessment. Partial model directories require
explicit recovery instead of an automatic refit. The assessment lock and final
independent handoff check are saved; serial assessment execution is underway.

The reviewed [serial operator](scripts/execute_frozen_suite.py) executes the
existing schedule without changing scientific settings. It defaults to a
read-only preview, requires explicit phases, preserves logs and attempt
receipts, and stops on incomplete caches or operational failure. Phase E is
always manual; phase F still requires the reviewed assessment lock. Existing
manual jobs must finish before an operator execution begins.

```bash
python pipelines/simulation/integration_recoverability/scripts/execute_frozen_suite.py \
  --plan pipelines/simulation/integration_recoverability/runs/suite/schedule_v1.json \
  --phase simple C D
```

Adding `--execute` launches the exact eligible commands serially. The lock and
process check prevent common accidental overlaps; they do not make it safe to
start an independent benchmark process during execution. The operator's own
source hash is recorded separately from the already-frozen scientific sources.

The first production operator launch stopped before any scientific command
because an unrelated same-user `sd-pam` process protects its working directory.
The reviewed process-check correction avoids requesting that directory for
irrelevant commands; unreadable relative Python-script candidates still fail
closed. The zero-launch session remains in the journal. The corrected operator
(`6bde64d8d361eea62fe9f51fa5a8197a1c87661dca61f8981d24b116c4f7ce9f`)
started phase D with `linear_chain_0`, realistic counts, both seed-0 methods.
No frozen scientific source or schedule was changed for this correction.

The separately reviewed secondary helper is now pinned, with two immutable
input receipts: [linear](runs/secondary/linear_chain_0/informative/anchor_readout_v1/freeze.json)
and [constrained nonlinear](runs/secondary/constrained_chain_0/informative/anchor_readout_v1/freeze.json).
Their SHA-256 values are respectively
`eea38d04952dc17c24f67fe237315de62e0e5f0c2f7ff053928751913f6812b3`
and `c61f5014de094628e08d169a3ce1b800b1a19eb376bccc5952a733b13e53eba2`.
These receipts are input audits; both runs now have sealed reports and completion
receipts. Their eight conditions and 80 graphs passed 2,626 independent checks,
including all 284 newly sealed file hashes and the small readout arrays. The
[results table](RESULTS.md#secondary-anchor-and-readout-seed-checks-both-informative-worlds)
retains every condition separately. No model was refit. Do not edit the frozen secondary helper,
protocol, or operator without an explicit new version and input receipt.

The [world-level synthesis protocol](SUITE_SYNTHESIS_PROTOCOL.md) was
independently reviewed and frozen before assessment access. Its production
[pin](runs/suite/synthesis_protocol_v1.json) has logical hash
`c0f09e20436ebed54b53b61cc034a26e2536c8094f8e640d4acd8b0dabba7927`
and file SHA-256
`ff9049e7da5c769d2ae15eccc38f6b794b51055d6ecb9940829aed0eacb73043`.
The operator, secondary helper, and synthesis helper passed 62 lightweight
tests together (19 + 23 + 20). This is additional software validation, not
62 scientific trials. The first [development-only snapshot](runs/suite/synthesis/development_20261004T0900Z/summary.md)
inventories all 170 scheduled stages, keeps both supplementary input freezes
pending until execution, and does not inspect assessment artifacts. It is a
dated partial snapshot, not the final benchmark result.

An independent production-snapshot audit verified all 170 stage statuses,
all seven output-file hashes, and 38 development-artifact hashes. The four
realistic count-readout failures remain unavailable in metric rows and method
contrasts; zero-denominator precision remains undefined. Assessment rows are
blocked placeholders, not inspected outcomes. The AST code graph was refreshed
after the final helper changes.

Reporting is read-only with respect to scientific artifacts, but is not
memory-light: this snapshot's JSON is 178.4 MiB, mostly repeated long-form
metric rows and pending placeholders. A static size projection suggests
roughly 0.6–0.8 GiB for the full suite, with unmeasured peak RAM potentially
several times larger during serialization. Subsequent full synthesis runs
must use an idle scientific execution slot, check available memory, and
measure peak resident memory. This is a resource precaution, not a change to
the frozen reporting specification.

[^1]: scMRDR authors' source. https://github.com/sjl-sjtu/scMRDR/tree/7280e64ae19f7391c6f5adf00d9d353f79e96603
