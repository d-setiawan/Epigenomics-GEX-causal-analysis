# Integration recoverability: observed results

_Original integration results through November 1, 2026. The simple three-seed first pass completed with one CI failure. The extended benchmark remains incomplete._

For the subsequent raw-count experiments, see the [existing-world demonstration](COUNT_TRANSPORT_DEMONSTRATION.md) and [exact-model source search](COUNT_TRANSPORT_SOURCE_SEARCH.md). Those experiments measured transport and prediction under a specific count model. [Clean-checkout replay scope](../../../reproduce/VERIFICATION.md)

The results tables are part of Git. Detailed readouts, logs and fit receipts linked under `runs/` in this historical report require the original run cache. [Input and replay guide](../../../TRANSFER.md)

## scConfluence and scMTR-seq follow-up

The [2026-11-01 pilot](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)
adds causal reliability checks on the saved full-count scGLUE fits, a fixed
scConfluence comparison and a separate scMTR-seq paired-reference evaluation.
Its [simulation controls](runs/integration_validation/20261101_causal_reliability_v1/BASELINE_RESULTS.md)
and [empirical controls](runs/integration_validation/20261101_scmtr_reliability_v1/BASELINE_RESULTS.md)
are complete. All six fits and the [final causal readouts](runs/integration_validation/20261101_causal_reliability_v1/RESULTS.md)
are complete and independently reviewed. The pilot exposes seed instability
and sparse-count CI miscalibration, without establishing reliable causal
recovery. The earlier count-input and integration-only reports below retain
their original scope and dates.

## Full-feature versus PCA/LSI integration inputs

The [four-fit follow-up](COUNT_INPUT_FOLLOWUP.md#observed-results) is complete on
two cached informative-count development worlds, seed 0. It changes scGLUE's
encoder input pipeline while retaining the same cells, features, count decoder,
guidance and hidden/training settings. Native normalization and input-layer size
also change, so this is not an isolated PCA-rank test.

| World | Worst state gap: PCA/LSI → full features | Minimum cross-assay decoder $R^2$: PCA/LSI → full features |
| --- | ---: | ---: |
| Linear chain | 0.1476 → 0.1365 | 0.8168 → 0.9271 |
| Constrained collider | 0.1312 → 0.1228 | 0.8914 → 0.9106 |

Lower state gap is better: zero is the finite-pool oracle and one is expected
random matching. These are not adjacency-recovery percentages. All six pairwise
state costs improve, but some individual decoder transfers worsen and retention
of the original PCA/LSI neighborhoods decreases. Independent reviews reproduce
both reports exactly. No new causal discovery was run on these fits, and this
small comparison does not establish reliability across seeds, worlds or real data.

## Integration-only validation, before new causal tests

The [new validation supplement](INTEGRATION_VALIDATION.md) evaluates state
alignment independently of causal discovery. On cached informative
`linear_chain_0` references, the worst-pair normalized squared-state transport
gap is 0.1420 for scGLUE and 0.0282 for scMRDR, using the same 487 validation
cells per assay. Zero reaches the finite-pool state optimum; one is the
expected random-matching cost. These are state-correspondence diagnostics,
not causal-recovery percentages.

An independent numerical audit reproduced all costs, IDs and declared decoder
scores. scMRDR's negative standardized RNA-to-mark decoder scores are sensitive
to the readout: exploratory raw-coordinate Ridge and nearest-neighbor transfer
remain positive. A single decoder therefore cannot certify or refute all
aspects of integration. The declared primary selection rule is unchanged.

All sixteen fresh selection fits are complete. The
[full first-world table](INTEGRATION_VALIDATION.md#first-selection-world-all-eight-settings)
uses common cells for all settings. scGLUE's fresh baseline has gap 0.1461;
`latent_dim=10` gives 0.1068. scMRDR's fresh baseline has gap 0.0282;
`beta=2` gives 0.0266 but worsens the weakest fixed-decoder transfer score.
The second world changes those rankings. The predeclared
[two-world rule](INTEGRATION_VALIDATION.md#frozen-two-world-selection) selects
scGLUE `lam_align=0.2` (mean gap 0.1287) and scMRDR's recorded baseline (0.0292).
Independent reviews reproduced all state costs and the selection. The sixteen
confirmation fits have completed with these frozen settings on two other cached
development worlds. These selection results do not establish reliability or
joint-distribution recovery. No new setting has been evaluated with causal
discovery. Existing causal results below retain their original settings and
interpretation.

The [first complete confirmation condition](INTEGRATION_VALIDATION.md#first-confirmation-condition-constrained-chain-realistic-counts),
realistic `constrained_chain_0`, uses identical pools for both methods and
seeds. scGLUE's state gaps are 0.328–0.341; scMRDR's are 0.133–0.134.
Both have positive cross-assay fixed-decoder scores here. scGLUE retains more
input neighborhoods, while scMRDR has less concentrated cross-assay neighbor
searches. The supplement reports absolute transport costs and all four runs
separately. This one condition does not establish biological generalization
or residual-joint recovery.

All four [informative constrained-chain confirmations](INTEGRATION_VALIDATION.md#informative-constrained-chain-confirmation)
also completed and passed independent reviews. scGLUE's gaps are 0.1247 and
0.1501, with weakest cross-assay decoder $R^2$ of 0.8601 and 0.8937. scMRDR's
gaps are lower, 0.0366 and 0.0442, but its weakest decoder scores are -3.5127
and -12.6399. Both methods use the same 955 discovery cells per assay, compared
with 766 in the realistic condition. The
[analytic explanation](INTEGRATION_VALIDATION.md#global-matching-and-direct-coordinate-use)
shows why global matching and direct coordinate-based diagnostics can disagree;
it does not diagnose the actual model.

The four [realistic unrestricted-chain fits](INTEGRATION_VALIDATION.md#realistic-unrestricted-chain-confirmation)
are also complete. scGLUE's gaps are 0.3479 and 0.3276. scMRDR's seed-0 gap is
0.1246, but seed 1 reaches 1.4446, worse than expected random matching, with
negative decoder scores in all four RNA/mark directions. Independent native
and numerical reviews passed; the scored pools are identical, and assignments,
state costs, decoder scores and hubness reproduce. This seed failure remains
in the results.

The four [informative unrestricted-chain fits](INTEGRATION_VALIDATION.md#informative-unrestricted-chain-confirmation)
have also completed. scGLUE gaps are 0.1438 and 0.1466; scMRDR gaps are 0.0269
and 0.0276. scMRDR's minimum cross-assay decoder scores are 0.1730 and -1.9597,
so similar matching performance coexists with different coordinate-transfer
outcomes. All 32 planned study fits and sixteen confirmation reports are
complete and have passed scoped independent reviews. The
[descriptive comparison](INTEGRATION_VALIDATION.md#descriptive-comparison-across-confirmation-conditions)
reports every confirmation condition without treating optimization seeds as
independent biological replications.

The original baseline's bounded `constrained_chain_1` continuation completed
with its unchanged settings. All four models and both evaluation reports have
completed, with both stages and the helper exiting 0. The realistic report's
[reviewed recovery results](#constrained-nonlinear-chain-1-realistic-baseline-continuation)
and [informative results](#constrained-nonlinear-chain-1-informative-baseline-continuation)
remain limited. Both scientific reviews and the final artifact-binding review
passed. Completed execution does not establish reliable causal recovery.

## Extended development: all six causal worlds, primary seed 0

The six causal development worlds have completed both integration methods
under realistic and informative counts. This covers the linear, constrained
nonlinear, and unrestricted nonlinear families, each with a chain and a
collider. Each family/motif has only one development draw; the two count
regimes reuse its biology. Assessment-world results are reported separately below.

The table counts cross-assay adjacency overlaps (TP) and extra calls (FP)
against the 12-edge count-measurement PAG in each world. A completed `0 / 0`
means an empty cross-assay graph with undefined precision and zero recall.
`Failed` means the graph could not be computed and its scores are unavailable.

| Development world | Counts | Common cells per assay | scGLUE TP / FP | scMRDR TP / FP |
| --- | --- | ---: | ---: | ---: |
| Linear chain | Realistic | 788 | Failed | Failed |
| Linear chain | Informative | 969 | 1 / 0 | 0 / 0 |
| Linear collider | Realistic | 769 | 2 / 4 | 1 / 3 |
| Linear collider | Informative | 967 | 1 / 0 | 0 / 0 |
| Constrained chain | Realistic | 766 | 2 / 2 | 2 / 1 |
| Constrained chain | Informative | 955 | 1 / 0 | 1 / 0 |
| Constrained collider | Realistic | 789 | 2 / 3 | Failed |
| Constrained collider | Informative | 967 | 0 / 0 | 0 / 0 |
| Unrestricted chain | Realistic | 756 | Failed | Failed |
| Unrestricted chain | Informative | 957 | 1 / 0 | 1 / 0 |
| Unrestricted collider | Realistic | 785 | 2 / 0 | 2 / 0 |
| Unrestricted collider | Informative | 963 | 1 / 0 | 1 / 0 |

Five of the 12 learned realistic-count readouts failed: two from a constant
H3K27ac sentinel in the linear chain, and three RCIT eigensolver failures in
the constrained collider and unrestricted chain. All 12 informative-count
readouts completed. The 19 completed count graphs return no fully directed
cross-assay count edges. Their adjacency recall is at most 2/12 (0.17), and
the informative regime returns at most one of 12 adjacencies per graph.
These development observations do not establish reliable count recovery.

All 24 learned activity graphs completed. The activity-projection targets
have eight adjacencies in chain worlds and 12 in collider worlds, so their
scores must use their respective targets. Activity diagnostics use simulator
truth and assess a different layer from the observed-count result above.

On 2026-10-04 at 18:15 UTC, a read-only audit independently recomputed all
saved learned adjacency TP/FP/FN counts, checked the evaluator and world
manifest identities, and counted directed edges from graph endpoints. It
verified 12 primary reports, 19 completed and five failed count graphs, and
24 activity graphs. No models or CI calculations were rerun. Reports are
under `runs/extended/{family}_{motif}_0/{regime}/primary/` with filename
`evaluation_scglue_scmrdr_seeds0/report.json`. The chain-world seed and guidance
controls are reviewed below; shared-state controls and supplementary diagnostics
have also completed, with their reviews recorded in the implementation status.

## Development chain worlds: seeds and guidance controls

All declared seed and guidance arms have completed for the linear, constrained,
and unrestricted chain development worlds under both count regimes. An
independent read-only review checked 18 evaluation reports and 18 guidance-only
baselines, their recorded fit/export identities, and all 17 frozen scientific
source hashes. The all-seed reports reproduce the original seed-0 assignments,
graphs, and probes exactly. A separate recomputation confirmed the table and
the aggregate count outcomes below without rerunning models or CI tests.

The primary arm retains focal input features but excludes all nine focal
features from cross-feature guidance. Background-only removes those inputs
before preprocessing for both methods. Shuffled guidance changes only the
remaining background guidance for scGLUE; scMRDR does not consume that graph,
so its shuffled-guidance fit is explicitly not applicable. These are comparisons
of frozen arms, not algorithm changes selected after viewing their outcomes.

The table reports cross-assay count-measurement-PAG recall against 12 true
adjacencies. Primary ranges include all three integration seeds; each ablation
uses seed 0. A failed graph has unavailable recall, not zero recall.

| Chain world | Counts | Common cells per assay | Primary scGLUE | Primary scMRDR | Background scGLUE / scMRDR | Shuffled scGLUE |
| --- | --- | ---: | --- | --- | --- | --- |
| Linear | Realistic | 788 | 3/3 failed | 3/3 failed | Failed / Failed | Failed |
| Linear | Informative | 969 | 0–8.3% | 0–8.3% | 8.3% / 0% | 8.3% |
| Constrained | Realistic | 766 | 8.3–16.7% | 0–16.7% | 8.3% / 8.3% | 8.3% |
| Constrained | Informative | 955 | 0–8.3% | 0–8.3% | 8.3% / 8.3% | 0% |
| Unrestricted | Realistic | 756 | 3/3 failed | 16.7%; 2/3 failed | Failed / 16.7% | 8.3% |
| Unrestricted | Informative | 957 | 8.3–16.7% | 8.3–16.7% | 16.7% / 8.3% | 0% |

Among 54 learned count readouts, 15 failed: nine constant-sentinel prechecks
and six eigenvalue-convergence failures. All preserve unavailable scores and
`edges: null`. Of the 39 returned graphs, eight contain no cross-assay edges
and retain undefined precision. No returned graph recovers the exact biological
cross-assay pair or directed set in the identifiability stress test. These
results do not show a consistent ablation improvement or reliable count-level
structure recovery.

Common-pool IDs are identical across seeds and arms within each world/regime;
these particular ablation differences are therefore not due to changing pools.
Pools differ between regimes, and the count-QC/source-oracle limitation below
still applies. Paired-anchor count recall is also only 0–16.7%, so poor learned
recovery cannot be attributed solely to integration. Seeds, regimes, and repeated
controls are not independent biological replicates.

Informative shared-embedding state-probe $R^2$ is approximately 0.910–0.995
across these arms. Every declared shared-embedding-only cross-assay residual
probe has negative $R^2$, although some incremental MSE gains over the state
baseline are positive. These diagnostic, uncorrected comparisons neither prove
that residual information is absent nor show that it is reliably recovered.
scGLUE RNA probe cohorts retain fewer cells than the other views and scMRDR;
they belong to development worlds, not the locked assessment-world bank.
Activity readouts use hidden simulator values and a different graphical target;
their higher recovery does not establish recovery from observed counts.

Evidence: `runs/extended/{linear,constrained,unrestricted}_chain_0/{regime}/`
under `primary/evaluation_suite_all_seeds/`,
`background_only/evaluation_scglue_scmrdr_seeds0/`, and
`shuffled_guidance/evaluation_scglue_seeds0/`. Each arm also has
`guidance_only_baseline.json`. The digest of their 36-file SHA-256 map is
`b7d0cc5467a2c1bb3f63d79be2abc156a1916d0eb311a507401c36c0b64e2112`.
Map keys are relative to `pipelines/simulation/integration_recoverability`
and start with `runs/extended/`. Serialization is
`json.dumps(hashes, sort_keys=True, ensure_ascii=True, separators=(", ", ": "))`
encoded as UTF-8 without a final newline (5,856 bytes). Both reviewers and the
main agent independently reproduced this digest. The second reviewer signed
off on the scientific values, fit/export bindings, probe claims, and masking
interpretation; different recorded CI runtimes are not new scientific outputs.

## Shared-state-only control: completed primary readouts

The `shared_state_0` biological DAG has state-to-feature arrows but no causal
paths between measured features. Shared state nevertheless induces 12 genuine
cross-assay adjacencies in each activity/count oracle PAG. This is a negative
control for interfeature causal ancestry, not an adjacency-null population.

Both primary arms have completed all three seeds. Both regimes' guidance/input
ablations and prior-only baselines have also completed and been reviewed.
The supplementary development diagnostics have now run; their final review
and synthesis are described in the execution record. An initial independent
read-only review checked four completed reports, eight distinct fit identities,
source/export bindings, fixed pools, and seed reproduction, passing 409 bounded
metadata/consistency checks. A further 316-check bounded audit covered the new
informative seeds-1–2 and all-seed reports, including reproduction of seed 0.
These overlapping metadata checks are not independent biological replications.
The main agent separately counted directed arrows from the raw graph endpoints
and reconciled them with the saved scores.

TP/FP below refers to cross-assay adjacency against the appropriate oracle's
12-edge target. Directed arrows are fully directed cross-assay PAG edges;
every such arrow is unsupported by this world's biological ancestry.

| Counts regime | Method / seed | Activity TP / FP | Activity arrows | Count TP / FP | Count arrows |
| --- | --- | ---: | ---: | ---: | ---: |
| Realistic | scGLUE 0 | 4 / 0 | 1 | 0 / 2 | 0 |
| Realistic | scGLUE 1 | 5 / 0 | 3 | 1 / 3 | 0 |
| Realistic | scGLUE 2 | 3 / 0 | 1 | 1 / 3 | 0 |
| Realistic | scMRDR 0 | 5 / 0 | 3 | 0 / 3 | 0 |
| Realistic | scMRDR 1 | 4 / 0 | 2 | 3 / 1 | 0 |
| Realistic | scMRDR 2 | 5 / 0 | 1 | Failed | Unavailable |
| Informative | scGLUE 0 | 4 / 0 | 0 | 1 / 0 | 0 |
| Informative | scGLUE 1 | 4 / 0 | 0 | 1 / 0 | 0 |
| Informative | scGLUE 2 | 5 / 0 | 1 | 0 / 0 | 0 |
| Informative | scMRDR 0 | 6 / 0 | 0 | 1 / 0 | 0 |
| Informative | scMRDR 1 | 5 / 0 | 0 | 1 / 1 | 0 |
| Informative | scMRDR 2 | 8 / 0 | 2 | 0 / 0 | 0 |

All six realistic activity outputs contain `H3K4me3_core → RNA_core`, despite
the absence of that biological ancestry. A directed PAG edge is not generally
a direct molecular interaction; in this control even its ancestry interpretation
is false. Correct adjacency alone therefore does not establish causal recovery.
Activities are hidden diagnostic values, not additional inputs to integration.

The failed realistic scMRDR seed-2 count readout preserves `edges: null`,
unavailable scores, and `Eigenvalues did not converge`. Every returned count
graph has an empty fully directed cross-assay set. Its `exact_directed_set: true` is empty-set
agreement with an empty biological target, not recovered causal directions.
The informative activity readouts without directed arrows have the same caveat.
Both informative seed-2 count graphs have no cross-assay adjacencies but retain
one within-assay `H3K4me3_core o-o H3K4me3_decoy` edge. Their
`exact_pair_set: true` agrees with the empty biological-direct cross-assay target
while their cross-assay PAG recall is zero and precision is undefined. All
other returned learned primary readouts have nonempty cross-assay adjacency
sets and hence `exact_pair_set: false` against that direct target. Recall against
a zero-edge direct target is undefined, not zero.

The apparent absence of false informative activity directions at seeds 0–1
does not survive seed 2. scGLUE returns `RNA_decoy → H3K4me3_decoy`; scMRDR
returns `H3K4me3_decoy → H3K27ac_decoy` and
`RNA_core → H3K4me3_decoy`. All of these adjacencies are state-induced oracle
associations, but none of these ancestry claims is biologically supported.

The paired controls matter for attribution. The independent paired activity
reference returns four unsupported cross-assay arrows in both regimes. It
reuses the same 2,000-cell activity cohort and the identical graph, so these
are not two independent confirmations. The paired-anchor activity reference
returns four such arrows with realistic counts and zero with informative
counts; true-state matching returns one and zero, respectively. Random matching
returns none. All of these reference computations completed, and all reference
count graphs have no fully directed cross-assay arrows. An independent check
verified the reference counts and raw endpoints.

False activity orientations thus occur even in genuinely paired reference
data and cannot be attributed solely to integration. These reports do not
separate finite-sample CI errors, orientation propagation, or other discovery
limitations. The realistic primary comparisons all use the same 809 common
cells per assay; all informative primary seeds use the same 970. Cross-regime
differences also change the evaluated population. These are development
diagnostics, not locked assessment
results or independent biological replications.

Evidence: [realistic all-seed report](runs/extended/shared_state_0/realistic/primary/evaluation_suite_all_seeds/report.json),
SHA-256 `69aabd0d35ffa2ea6fa12c887769d3d77df353c22b4aa3bf3f7fffc1381dc6af`;
[informative seed-0 report](runs/extended/shared_state_0/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `803cc323f14e37f5a815eb21bf032072df07876c545dda072aa0d626010d0b61`.
The [informative seeds-1–2 report](runs/extended/shared_state_0/informative/primary/evaluation_scglue_scmrdr_seeds1-2/report.json)
has SHA-256 `7c78b1032d6f31b0c3f0f4cbef012cfb241fb4795169c87a99ca42947aa5e63d`;
the [informative all-seed report](runs/extended/shared_state_0/informative/primary/evaluation_suite_all_seeds/report.json)
has SHA-256 `cec77c58a61f34905ade366a83890a6514359362b1a71c38e0b724cc56bac738`.
The realistic seed-0 and seeds-1–2 reports remain separately retained and agree
with the all-seed scientific outputs. A second reviewer independently checked
the earlier eight-row table, hashes, reference arrows, zero-target interpretation,
and repeated cohort identity; the four additional informative seed rows are
covered by the further audit above. The second reviewer also checked the four
new rows, report hashes, and seed-2 endpoint/empty-target interpretations;
whole-graph recall is not being confused with cross-assay recall. Wording
distinguishes learned from reference graphs and empty direct targets from
state-association recovery.

### Realistic shared-state input and guidance ablations

The two additional evaluation reports and all three guidance-only baselines
passed a separate 176-check bounded JSON audit. All three new learned count
and activity readouts completed, preserving the earlier primary seed-2 count
failure. They use exactly the same 809 cells per assay and RNA-anchor order as
the primary arm, but their partner assignments change substantially.

| Arm / method, seed 0 | Activity TP / FP | False cross-assay activity arrows | Count TP / FP | Cross-assay count arrows |
| --- | ---: | ---: | ---: | ---: |
| Background-only / scGLUE | 3 / 0 | 0 | 4 / 2 | 0 |
| Background-only / scMRDR | 4 / 0 | 3 | 1 / 4 | 0 |
| Shuffled guidance / scGLUE | 3 / 0 | 1 | 3 / 4 | 0 |

Removing focal inputs or shuffling background guidance does not consistently
eliminate false activity ancestry. Background-only scGLUE's absence of arrows
is agreement with an empty ancestry target, not discovery of causal directions.
Its extra count-adjacency overlap concerns state-induced association, not direct
regulation. All eight returned realistic learned count graphs have no fully
directed cross-assay arrows, and every returned learned graph has a nonempty
cross-assay adjacency set, so none exactly matches the empty biological-direct
pair target. This one development control does not establish an ablation winner.

Recorded preprocessing metadata confirms the frozen definitions. Background-only
scGLUE removes three focal features per assay; the scMRDR gene union falls from
2,009 to 2,000. Shuffled scGLUE retains the primary training-input hashes and
feature counts. All three prior baselines correctly abstain with zero eligible
scored pairs: the nine focal features are masked from cross-feature guidance.
Primary/shuffled retain their vertices; background-only excludes them. This is
imposed masking, not successful null recovery. Reference graphs/scores are
identical across arms and must not be counted as independent replications.

Evidence: [background-only report](runs/extended/shared_state_0/realistic/background_only/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `af16663695eab6cfc0a1babc97deb1e0f42f40d0b3366cd50c6e10460a457328`;
[shuffled-guidance report](runs/extended/shared_state_0/realistic/shuffled_guidance/evaluation_scglue_seeds0/report.json),
SHA-256 `a729e95f36627bdfd019b4112460d17018098397a6b9493b971d974a8038f7e0`.
The three `guidance_only_baseline.json` files remain within their respective arm directories.

### Informative shared-state input and guidance ablations

The background-only evaluation has completed for both methods, with no graph
failures, and passed 52 bounded consistency checks. The shuffled-guidance
evaluation and three prior baselines passed another 76 bounded checks. The
970-cell pools and RNA-anchor order exactly match primary seed 0; partner
assignments differ.

| Arm / method, seed 0 | Activity TP / FP | False cross-assay activity arrows | Count TP / FP | Cross-assay count arrows |
| --- | ---: | ---: | ---: | ---: |
| Background-only / scGLUE | 4 / 0 | 1 | 0 / 0 | 0 |
| Background-only / scMRDR | 7 / 0 | 0 | 1 / 0 | 0 |
| Shuffled guidance / scGLUE | 3 / 0 | 1 | 0 / 0 | 0 |

Background-only scGLUE's activity graph asserts `H3K4me3_core → RNA_core`
despite no biological interfeature ancestry. Its count graph has no cross-assay edges but retains
one within-assay edge: cross-assay recall is zero, precision is undefined, and
exact biological-direct set agreement is vacuous. scMRDR's count recall is
one of 12 target associations; its empty activity-direction set is not recovered
causal direction. The counts target concerns shared-state association, not a
direct edge. These outcomes do not establish a superior input arm.

The arrow columns count only fully directed cross-assay edges. Background-only
scGLUE also returns the unsupported within-assay activity arrow
`RNA_decoy → RNA_core`; shuffled scGLUE returns
`H3K27ac_decoy → H3K27ac_core`. These are not included in the cross-assay counts.

Shuffled-guidance scGLUE asserts `H3K4me3_core → H3K27ac_core` in activities,
also unsupported. Its count graph retains only the within-assay
`H3K4me3_core o-o H3K4me3_decoy` edge. All nine informative learned count
readouts have completed without fully directed cross-assay arrows. Four have
no cross-assay adjacencies: both primary seed-2 readouts and the background-only
and shuffled-guidance scGLUE readouts. Their exact biological-direct pair-set
agreement does not recover any of the 12 shared-state associations.

Recorded background-only feature counts decrease by three per assay for scGLUE
and from 2,009 to 2,000 in the scMRDR gene union. Shuffled scGLUE preserves
primary training/discovery input hashes and feature counts, while its guidance
graph hash changes. Source/configuration identities and recorded export
bindings agree with the frozen inputs. All three informative prior baselines
correctly abstain: zero eligible scored pairs, with nine masked vertices in
primary/shuffled and no scored vertices in background-only. Imposed exclusions
are not successful null recovery. Reference outputs duplicate primary controls
and are not new replications.

Evidence: [informative background-only report](runs/extended/shared_state_0/informative/background_only/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `0675cf42563dbca80faa219520316a46bbcdea9141cc0895e353ba704c61ab67`;
[informative shuffled-guidance report](runs/extended/shared_state_0/informative/shuffled_guidance/evaluation_scglue_seeds0/report.json),
SHA-256 `64139a3c790c9f14e2baf4aac18d51afcef677c29022dcfc5e9268b21487bd2f`.
Each arm retains its `guidance_only_baseline.json`.
An independent second review checked the ablation table, hashes, nine completed
informative count readouts, four empty cross-assay sets, and prior/reference
interpretations; arrow labels were clarified to exclude within-assay arrows.

## Observational ambiguity control

The exact ambiguity counterparts have completed both count regimes. Their
public unpaired observations are identical, while their hidden joint truth
differs by the direct H3K27ac-core to RNA-core edge. An independent audit
verified all 18 public H5AD pairs, reused model checkpoints, public embeddings,
common pools, assignments, original readouts, and 16 returned activity/count
graphs across scGLUE, scMRDR, random, and true-state matching. All of those
learner-visible inputs and outputs agree across the counterparts.

The complete hidden discovery cohort changes 6,000 activity entries while
each assay's own observed activities and counts remain identical. Residual
correlation after the saved conditional-mean adjustment is descriptively
-0.027 in the zero-coupling counterpart and 0.692 in the other. These values
describe the stipulated finite simulation; they are not inferential intervals.

This demonstrates the intended ambiguity: identical available inputs can
correspond to different hidden residual relationships. Reusing the same fits
makes identical graph outputs an expected consistency check. It does not
constitute an independent recovery trial. A graph scoring better against one
counterpart, or an empty directed graph exactly matching a zero-edge truth,
does not demonstrate discrimination between the two structures.

Both informative evaluations completed without failed diagnostics. Both
realistic evaluations retain an RCIT eigensolver failure in the private
paired-anchor count control. Their twin comparison still says `completed`
because its four assignment-based comparisons succeeded; the original failed
control remains visible. Neither method emits a justified ambiguity set, so
coverage and average set size remain unavailable.

Evidence: [realistic comparison](runs/extended/ambiguity_rho_0.7/realistic/primary/twin_comparison.json),
SHA-256 `0928c93bcc46b7f435e4137d190834cf621275c7f4fd26062158491a0e16d706`;
[informative comparison](runs/extended/ambiguity_rho_0.7/informative/primary/twin_comparison.json),
SHA-256 `9f5c9d36d3742ded504bd516f9e9920dd5a9d7bed5bb35c7cb38358d0fc007cd`.

## Simple simulation: first pass across integration seeds 0, 1, and 2

All six integration fits completed on the same frozen simple population. The
all-seed evaluation uses the same 2,000 common-eligible cells per assay and is
`completed_with_failures`: scGLUE seed 2 has a numerical failure in count
discovery. These are integration-seed repeats of one biological world, not
three independent biological replications.

Every matched activity graph is the same undirected core triangle: both true
chain adjacencies plus the extra H3K27ac–RNA adjacency, giving precision 0.67
and recall 1.00. The paired-activity reference recovers the two-edge skeleton
exactly; true-state matching also produces the triangle. No integration seed
returns a fully directed activity edge. Thus the repeated result is stable
over these seeds but is not exact structure recovery.

Count results below use the count-measurement PAG's three cross-assay target
adjacencies, not just the two direct biological edges. The failed row is kept
visible and is not included as an empty graph or zero-recall observation.

| Method / seed | Source overlaps / cross-assay calls | Extra cross-assay calls | Precision / recall |
| --- | ---: | ---: | ---: |
| scGLUE / 0 | 2 / 7 | 5 | 0.29 / 0.67 |
| scGLUE / 1 | 2 / 7 | 5 | 0.29 / 0.67 |
| scGLUE / 2 | CI failure | Unavailable | Unavailable |
| scMRDR / 0 | 1 / 4 | 3 | 0.25 / 0.33 |
| scMRDR / 1 | 1 / 7 | 6 | 0.14 / 0.33 |
| scMRDR / 2 | 1 / 4 | 3 | 0.25 / 0.33 |

Equal scores do not imply identical returned pairs. None of the five completed
count graphs returns a fully directed edge. The paired-count reference still
returns two cross-assay calls with no source-target overlap; random matching
returns four calls with one coincidental overlap. As detailed below, these
controls prevent a calibrated causal-recovery claim or an integration-only
explanation for the count errors.

State-matching errors vary modestly across seeds. The entries are minimum to
maximum mean absolute state differences over the three integration seeds;
they are descriptive ranges, not confidence intervals.

| Method | RNA–H3K27ac | RNA–H3K4me3 | H3K27ac–H3K4me3 |
| --- | ---: | ---: | ---: |
| scGLUE | 0.508 to 0.528 | 0.475 to 0.520 | 0.581 to 0.623 |
| scMRDR | 0.392 to 0.402 | 0.264 to 0.285 | 0.409 to 0.422 |

Both improve on the fixed random control's 1.107–1.118 range, while neither
reaches the true-state reference's 0.046–0.065 range. Better state alignment
does not, in these runs, preserve the same-cell activity skeleton.

### Numerical count failure: scGLUE seed 2

The saved failed query is marginal `RNA_core` versus `H3K27ac_sentinel`, with
query seed 1466141879 and `LinAlgError: Eigenvalues did not converge`. The
[saved readout](runs/simple/realistic/evaluation_suite_all_seeds/readouts/scglue_seed2_acb634b6.npz)
has 67 RNA-core and eight H3K27ac-sentinel detections among 2,000 rows; the
sentinel is zero in every one of the first 500 rows.

Data/source inspection identifies a concrete sparse-bandwidth failure path:
the installed marginal RIT backend takes the median of nonzero pairwise
distances in that 500-row prefix. The sentinel has no such distances, producing
a nonfinite bandwidth. Its fallback handles zero or missing bandwidths but
not NaN, which propagates into the covariance eigendecomposition. This is a
numerical test failure, not evidence of independence, a scGLUE training failure,
or an estimated empty graph. No test was rerun or repaired for this diagnosis;
the frozen failure and unavailable count-graph metrics are preserved.

Evidence: [all-seed common-pool report](runs/simple/realistic/evaluation_suite_all_seeds/report.json),
SHA-256 `6bb9157c98d3e5d4b3fb43f7bfe2fd13a763eea3cd97e2619a776f9b625e4d90`.
The historical seed-0 detail follows. Across-seed agreement does not resolve
the sparse-test calibration problems or establish biological reliability.

## Simple simulation: historical integration-seed-0 detail

Both native pipelines completed on the same frozen population, retaining all
2,000 observed cells per assay. No aggregation is used. Matching is the fixed
RNA-anchored, one-to-one assignment in shared embedding space; discovery uses
the original named variables, not embedding coordinates. Hidden activities and
true-state matching are evaluator-only diagnostics, not extra learner inputs.

This is one transductive development world and one seed per method. It cannot
establish reliability, method superiority, or unique causal identification.
No settings were changed to improve these results.

### State alignment improves, but the activity chain becomes a triangle

Lower state-matching error means that assigned cells have more similar true
biological states. The errors below use the simulator's standardized state.

| Assignment | RNA–H3K27ac error | RNA–H3K4me3 error | H3K27ac–H3K4me3 error |
| --- | ---: | ---: | ---: |
| Random | 1.111 | 1.107 | 1.118 |
| scGLUE | 0.517 | 0.510 | 0.595 |
| Paper-consistent scMRDR | 0.398 | 0.273 | 0.409 |
| True-state reference | 0.046 | 0.053 | 0.065 |

The stipulated activity chain has two edges, with no direct H3K27ac–RNA edge.
The paired-activity reference recovers its skeleton exactly. Both integration
methods, and even true-state matching, return all three core adjacencies.

| Activity input | True edges recovered | Extra edges | Adjacency precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| True paired cells | 2/2 | 0 | 1.00 | 1.00 |
| scGLUE-matched cells | 2/2 | 1 | 0.67 | 1.00 |
| scMRDR-matched cells | 2/2 | 1 | 0.67 | 1.00 |
| True-state-matched cells | 2/2 | 1 | 0.67 | 1.00 |

The paired chain's diagnostic conditional-independence test has $p=0.322$.
This contrast supports the distinction between recovering state-dependent
associations and preserving the original same-cell dependence. It does not
establish that no other permitted inference procedure could do better.
The paired chain itself is not uniquely oriented by ordinary observational
conditional independences, so exact direction recovery is a stress test.

### Sparse counts already limit recovery before integration

Among the 2,000 true-paired reference cells, core detections are 67 for RNA,
12 for H3K27ac, and 55 for H3K4me3. Core co-detections are only 1 for RNA–H3K27ac,
3 for RNA–H3K4me3, and 1 for H3K27ac–H3K4me3. Marginal count-calibration checks
passed, but that does not imply adequate graph-discovery power.

Count results are scored against the count-measurement PAG, not the biological
activity DAG. That target has three core cross-assay adjacencies because the
activities are hidden causes of noisy measurements; its within-assay targets
also include common-exposure associations.

| Count input | Cross-assay calls | Source-target overlaps | Extra cross-assay calls | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| True paired cells | 2 | 0 | 2 | 0.00 | 0.00 |
| scGLUE-matched cells | 7 | 2 | 5 | 0.29 | 0.67 |
| scMRDR-matched cells | 4 | 1 | 3 | 0.25 | 0.33 |
| True-state-matched cells | 4 | 0 | 4 | 0.00 | 0.00 |
| Random matching control | 4 | 1 | 3 | 0.25 | 0.33 |

For random matching, all four cross-assay calls are negative-control calls;
one happens to overlap a pair in the source graph. That overlap is not recovery
of a joint distribution. More generally, matched matrices impose a coupling;
agreement with the source graph is a diagnostic, not proof of direct regulation.

These outcomes do not support reliable count-level structure recovery. The
paired-count and random-matching results warn against attributing all errors
to integration. Sparse measurement, finite-sample CI calibration, and the
matching procedure require separate examination. Nominal RCIT alpha 0.01 is
not an established false-adjacency guarantee here.

A read-only statistical audit reproduced one saved random-matching call
exactly: H3K27ac core versus H3K4me3 core had 17 and 18 detections among 2,000
rows, no co-detections, correlation approximately $-0.0088$, and an RCIT-backend
marginal-test $p$-value of $0.0029$. No co-detection is unsurprising at that
frequency (the independent expected number is only 0.153). The audit found the
requested backend/configuration, not an accidental test substitution. This is
evidence of a sparse finite-sample null-approximation problem, not a new
biological relationship. The frozen primary results remain unchanged and must
not be presented as calibrated significance or a method ranking.

Neither method returns a fully directed biological edge in this count readout.
Hyperedge and general ambiguity-set metrics are N/A because these methods do
not emit the required objects. Sentinel counts are scored with their exposure
causes included; an exposure-induced within-assay edge is not falsely called a
biological sentinel effect. The guidance-only control is also N/A for scored
pair recovery: all scored cross-feature guidance links were deliberately masked.

### Execution and provenance

The primary evidence is
[the combined seed-0 report](runs/simple/realistic/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `0a93070c05bc59e5dc0ef29c318a8c57fa52fbb0dd16ed29ea101d75aeb63580`.
Its common-pool, assignment, graph, CI-query, and input/source receipts are saved
alongside it. The earlier scMRDR-only exploratory report is retained separately.

scMRDR completed 173 epochs and restored checkpoint 163. Native scGLUE completed
38 epochs in each of its two phases, restoring checkpoint 37 in each phase.
Its first adapter export failed because the TensorBoard reader was missing,
after training had already finished. Following dependency installation, the
saved native final snapshot was exported without retraining, with exact
checkpoint equality and zero embedding round-trip error. Both the original
failure and [recovery provenance](runs/simple/realistic/scglue_seed0/recovery_provenance.json)
are retained. This was an infrastructure failure, not a changed model fit.

The combined benchmark and paper-model test suite passes 231 tests. Independent
agents reviewed simulation truth boundaries, native preprocessing, scMRDR
objective fidelity, checkpoint recovery, and scoring. Software validity does
not substitute for biological realism or statistical power.

## Extended linear chain: informative counts, seed 0

Both methods completed the first extended world, including held-out encoding
and all retention probes, with no execution failures. Unlike the simple study,
this world has two state dimensions and nine candidates (core, decoy, and
sentinel per assay). Its activity-projection and count-measurement targets
therefore differ from the simple chain's targets.

This predeclared diagnostic regime multiplies RNA exposure by 5 and chromatin
exposure by 20 without changing the biology or recalibrating intercepts. It is
not claimed to reproduce realistic measurement depth. Discovery uses 969
common-eligible cells per assay. Independent paired assessment has 2,000 cells;
the methods' private probe populations follow their recorded native filtering.

| Count input | Returned cross-assay edges | Overlaps with 12 source-PAG cross-assay edges | Extra edges | Recall |
| --- | ---: | ---: | ---: | ---: |
| Independent paired assessment (2,000 cells) | 3 | 3 | 0 | 0.25 |
| Paired anchor reference (969 cells) | 2 | 2 | 0 | 0.17 |
| scGLUE matching | 1 | 1 | 0 | 0.08 |
| scMRDR matching | 0 | 0 | 0 | 0.00 |
| True-state matching | 2 | 2 | 0 | 0.17 |
| Random matching | 0 | 0 | 0 | 0.00 |

An empty graph has undefined precision, not perfect precision. The lack of
false random cross-assay calls in this particular informative world is not a
general null-calibration guarantee. Even the independent paired-count reference
has low recall, so this is not a clean integration-only test.

Activity-projection adjacency precision/recall are 1.00/0.75 for the independent
paired reference, 0.71/0.63 for scGLUE matching, and 0.86/0.75 for scMRDR matching.
These measure agreement with the latent projection, not recovery of only the
direct biological chain or uniquely determined directions.
The separate paired-activity control with true state supplied detects both
core chain adjacencies. It recovers seven of ten source-oracle adjacencies,
missing three state-to-feature links, and does not recover the two core
directions. Thus even this activity control is not a perfect full-graph oracle
at the chosen finite sample size and test settings.

### Information retention on held-out synthetic cells

The table reports ranges across all six source-assay/state-dimension or
source-assay/cross-assay-core combinations, using shared embeddings. Both
predeclared probe capacities are shown, without selecting the better one.
Residual targets subtract conditional means estimated with 128 independent
simulator draws at each state; saved Monte Carlo standard errors quantify
additional target noise. Small residual-score differences should not be
interpreted as exact differences in retained residual information.

| Method / probe | State $R^2$ range | Cross-assay core-residual $R^2$ range |
| --- | ---: | ---: |
| scGLUE / ridge | 0.965 to 0.992 | -0.082 to -0.035 |
| scGLUE / shallow boosted tree | 0.917 to 0.962 | -0.053 to -0.029 |
| scMRDR / ridge | 0.947 to 0.991 | -0.050 to -0.018 |
| scMRDR / shallow boosted tree | 0.930 to 0.970 | -0.050 to -0.020 |

Negative test $R^2$ means worse squared-error prediction than a constant equal
to the assessment target's mean; it is not a negative amount of information.
Adding scMRDR's private coordinates also leaves all six cross-assay core-residual
scores negative for both probes. Importantly, same-assay core-residual scores
from shared embeddings are negative as well. Thus this result cannot attribute
the entire deficit to missing cross-assay cell pairing: sparse measurements,
representation capacity, and the restricted probes can also limit detection
of residual information. The subsequent input-retention audit below rules out
omission of scored inputs in these two extended runs.

The diagnostic supports strong shared-state retention with no demonstrated
residual prediction by these fixed probes. It does not prove that every possible
readout would fail. The supervised probes use evaluator-only labels and are
not deployable unpaired causal-discovery procedures. Small differences between
methods should not be ranked as general performance: their probe populations
follow native filtering, and this is one world and one integration seed.

Evidence: [extended linear report](runs/extended/linear_chain_0/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `96c4b34e68fc5782d5bd06fef8b46d953ce45bca6c3a1dae98a5ebaa5d1e8340`.
Native scGLUE ran 36 epochs per phase (about 277 seconds); scMRDR ran 182 epochs,
restoring checkpoint 172 (about 223 seconds). Full evaluation took about 304
seconds. These are measured development runtimes, not throughput guarantees.

### Secondary CI sensitivity check: linear informative world

The predeclared KCI and Fisher-Z checks are complete. They reuse the exact
RCIT integration manifests, common cell pools, matching assignments, and
truth receipts: 969 matched cells per assay and 2,000 independent paired
assessment cells. Both secondary reports say `completed_with_failures`
only because embedding and preprocessing probes were deliberately skipped.
Their graph calculations, state-supplied controls, and graph oracles completed
without numerical failures; integration was not refitted.

The following are cross-assay adjacency precision / recall against the
12-edge **count-measurement PAG**, not the biological two-edge chain.
N/A precision means zero returned cross-assay adjacencies.

| Count input | Primary RCIT | Secondary KCI | Diagnostic Fisher-Z |
| --- | ---: | ---: | ---: |
| Independent paired assessment | 1.00 / 0.25 | 1.00 / 0.25 | 1.00 / 0.50 |
| Paired anchor reference | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.33 |
| scGLUE matching | 1.00 / 0.08 | 1.00 / 0.08 | 0.75 / 0.25 |
| scMRDR matching | N/A / 0.00 | 1.00 / 0.17 | 1.00 / 0.33 |
| True-state matching | 1.00 / 0.17 | N/A / 0.00 | 1.00 / 0.33 |
| Random matching | N/A / 0.00 | N/A / 0.00 | N/A / 0.00 |

For the eight-edge **activity-projection target**, KCI gives scGLUE
precision/recall 0.75/0.75 and scMRDR 0.86/0.75, compared with RCIT
0.71/0.63 and 0.86/0.75. Fisher-Z gives 0.75/0.75 for both. Neither
method returns a fully directed cross-assay count edge under any of these
three tests. Activity directions vary across tests and remain an
identifiability stress test, not demonstrated direct biological regulation.

The differences show dependence on the CI test even with fixed integration
and matching. KCI is not a calibration guarantee for these counts. Fisher-Z
is only an assumption-sensitive diagnostic: linear structural functions do
not make the variables jointly Gaussian when biological state is a mixture,
and the observation model adds nonlinear sparse counts. Neither secondary
test replaces RCIT or resolves the [sparse-CI concerns](SPARSE_CI_AUDIT.md).
These are one-world, one-seed development comparisons, not a selection rule
for a preferred test or evidence of general recovery reliability.

Evidence: [KCI report](runs/extended/linear_chain_0/informative/primary/evaluation_suite_secondary_kci/report.json),
SHA-256 `81095278a56e4c445ce9cfaa48602065f10ae805e4d63fc6ec4988f21b9607af`;
[Fisher-Z report](runs/extended/linear_chain_0/informative/primary/evaluation_suite_linear_fisherz_diagnostic/report.json),
SHA-256 `e4d2d2f640a4bf718b71e8ade82393a460b98de5d4463a92d1e1715ad1992a32`.

## Extended linear chain: realistic counts, seed 0

Both integrations completed, and the declared count-calibration checks passed.
However, the frozen common pool has no detected H3K27ac sentinel counts.
The count-discovery precheck rejects all four unpaired readouts with
`Constant columns: ['H3K27ac_sentinel']`, before any CI query. Their graphs and
scores are unavailable, not empty graphs or zero-recovery results. This is
distinct from the earlier simple-study RCIT eigensolver failure. No scored
variable was removed from discovery and no alternative test substituted after
seeing the failure.

The common pool contains 788 cells per assay: scGLUE retained 788 of the first
1,000 RNA discovery cells, while scMRDR retained all 1,000. Both retained the
first 1,000 chromatin cells before equalization. The independent paired
reference has 2,000 cells. The paired anchor uses the 788 RNA-anchor cell IDs
but their actual same-cell measurements in every assay; those measurements
include four H3K27ac-sentinel detections. Its successful computation therefore
does not contradict failure on the different unpaired chromatin cells.

The count target has 12 cross-assay adjacencies (21 overall); the activity
projection has eight adjacencies overall. These are different estimands.

| Readout | Cross-assay count TP / FP / FN | Count precision / recall | Whole-panel activity precision / recall |
| --- | ---: | ---: | ---: |
| Independent paired reference | 4 / 2 / 8 | 0.67 / 0.33 | 1.00 / 0.75 |
| Paired anchor reference | 1 / 1 / 11 | 0.50 / 0.08 | 1.00 / 0.63 |
| scGLUE matching | Failed before CI | Unavailable | 1.00 / 0.75 |
| scMRDR matching | Failed before CI | Unavailable | 0.83 / 0.63 |
| True-state matching | Failed before CI | Unavailable | 1.00 / 0.75 |
| Random matching | Failed before CI | Unavailable | 0.33 / 0.13 |

scGLUE returns six true activity-projection adjacencies and no extras;
scMRDR returns five true adjacencies and one extra. Neither integrated method
returns a fully directed activity edge. Both successful paired-count controls
also return no fully directed cross-assay edge; failed count graphs have no
direction score. The state-supplied activity control recovers seven of ten
source adjacencies, including the two core-chain adjacencies but neither core
direction. These diagnostics do not establish direct biological regulation.

All activity calculations, graph oracles, and probe collections completed;
none of the probe collections was deliberately skipped. All nine scored
features survive both methods' selected integration inputs. Shared-embedding
state-prediction $R^2$ ranges from 0.63 to 0.93 across the two fixed probes and
assays, while every cross-assay core-residual $R^2$ is negative. Same-assay
residual scores are also negative, including scMRDR shared-plus-private and
native PCA/LSI probes. Negative residual $R^2$ is not proof of absent
information; a small positive incremental gain over a state-only comparator
occurs in one scGLUE tree readout. Probe populations differ under native
filtering, and conditional-mean targets carry Monte Carlo uncertainty.

This is one development world and one seed. QC-selected readouts are compared
with source-population graph targets, and even the paired-count controls are
imperfect. Neither integrated method has an available count-recovery result
in this run, despite successful fitting and improved state alignment over
random matching. The informative regime above remains a separate diagnostic,
not a replacement for this failed realistic-count comparison.

Evidence: [realistic linear report](runs/extended/linear_chain_0/realistic/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `4e1dd99e176eae69ced9b80896a43afee4fbabbe68e84471a97e0d7c9339e9cb`.

## Extended constrained nonlinear chain: informative counts, seed 0

Both methods completed this second development world and all diagnostic stages
without execution failures. The declared count-calibration checks passed. This
is still one world and one integration seed per method, not a reliability
estimate. The linear and constrained worlds have different simulation seeds
and coefficient draws, so their comparison does not isolate nonlinearity alone.

As in the linear world, the activity latent projection has eight adjacencies,
while the count-measurement PAG has 21, including 12 cross-assay adjacencies.
Neither target is simply the two-edge direct biological chain.

scGLUE retained 955 of the first 1,000 RNA discovery cells, versus 969 in the
linear world. Both methods retained all first 1,000 chromatin cells, and scMRDR
retained all first 1,000 RNA cells. Common-pool matching therefore uses 955 cells
per assay; the independent paired assessment still uses 2,000. The frozen
scGLUE reconstruction panels contain 1,949 RNA features and 15,000 per mark,
versus 1,944 RNA features in the linear world. scMRDR uses 2,009 gene-space
features in both. All nine original scored variables remain in the discovery
readout regardless of their integration-feature selection.

### Source-graph agreement remains limited

The table scores cross-assay count adjacencies against the 12-edge source-PAG
target. N/A precision means that no cross-assay edge was returned.

| Count input | Nonlinear source overlaps / calls | Nonlinear precision / recall | Linear precision / recall |
| --- | ---: | ---: | ---: |
| Independent paired assessment (2,000 cells) | 1 / 1 | 1.00 / 0.08 | 1.00 / 0.25 |
| Paired anchor reference (955 nonlinear; 969 linear) | 0 / 0 | N/A / 0.00 | 1.00 / 0.17 |
| scGLUE matching | 1 / 1 | 1.00 / 0.08 | 1.00 / 0.08 |
| scMRDR matching | 1 / 1 | 1.00 / 0.08 | N/A / 0.00 |
| True-state matching | 0 / 0 | N/A / 0.00 | 1.00 / 0.17 |
| Random matching | 0 / 0 | N/A / 0.00 | N/A / 0.00 |

Each method misses 11 of 12 source cross-assay count adjacencies. A precision of
1.00 based on one call does not demonstrate reliable recovery, and neither
method returns a fully directed edge in the biological-direction stress test.
Even the independent paired counts have low recall. Together with the
[sparse-CI audit](SPARSE_CI_AUDIT.md), this prevents interpreting these results
as an isolated test of integration quality or calibrated causal significance.

Against the eight-edge activity-projection target, both paired references have
precision/recall 1.00/0.75. scGLUE has 0.71/0.63, unchanged from the linear world;
scMRDR has 0.75/0.75, versus 0.86/0.75 in the linear world. Its eight returned
activity adjacencies include two extras and omit two source adjacencies, so
matching the target's edge count is not exact recovery. True-state matching
has 0.63/0.63, versus 0.86/0.75 in the linear world. These are adjacency scores,
not recovered direct mechanisms or uniquely identified directions.

### State retention does not demonstrate residual retention

Using the same fixed probes and range definitions as above:

| Method / probe | State $R^2$ range | Cross-assay core-residual $R^2$ range |
| --- | ---: | ---: |
| scGLUE / ridge | 0.963 to 0.993 | -0.037 to -0.014 |
| scGLUE / shallow boosted tree | 0.931 to 0.980 | -0.041 to -0.008 |
| scMRDR / ridge | 0.948 to 0.994 | -0.038 to -0.017 |
| scMRDR / shallow boosted tree | 0.933 to 0.977 | -0.038 to -0.018 |

As in the linear world, all shared-embedding same-assay core-residual scores
are also negative. Adding scMRDR's private coordinates does not produce
positive cross-assay core-residual scores with either probe. This supports
strong shared-state prediction but no demonstrated core-residual prediction
by these readouts; it does not prove that the embeddings contain no residual
information or that unpaired sampling is the only limitation.

The [input-retention audit](INPUT_RETENTION_AUDIT.md) confirms that all nine
scored variables survive both methods' input selection in both extended
worlds. Each focal scMRDR gene mapping is singleton, so background gene-sum
dilution is not the explanation either. The saved native PCA/LSI controls
already have negative same-assay residual scores before scGLUE fitting. These
checks do not apportion the remaining deficit among noisy measurement,
compression, learned objectives, probe limitations, and target uncertainty.

Probe populations differ from the matching pool and between methods. scGLUE's
RNA probes use 968 calibration and 1,935 assessment cells; its mark probes and
all scMRDR probes use 1,000 and 2,000, respectively. Residuals again use
128-draw conditional-mean estimates, not exact conditional means. Median
assessment Monte Carlo standard errors for RNA, H3K27ac, and H3K4me3 core
targets are approximately 0.043, 0.088, and 0.063 in activity units. Thus small
score differences do not establish method rankings. Hyperedge and general
ambiguity-set metrics remain N/A because the required objects are not emitted.

Evidence: [constrained nonlinear report](runs/extended/constrained_chain_0/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `1bff56125eb007e6d9e448b230a8cc96f9f80789566cffcbcb3d82d63e88a227`.
The separate world seeds and mechanisms are recorded in the
[constrained frozen parameters](runs/worlds/constrained_chain_0/truth/frozen_parameters.json)
and [linear frozen parameters](runs/worlds/linear_chain_0/truth/frozen_parameters.json).

### Secondary KCI sensitivity check: constrained informative world

The predeclared KCI diagnostic is complete and independently reviewed. It
reuses the exact primary fits, ordered 955-cell pools, assignments, and original
variables; independent paired assessment still uses 2,000 cells. Its
`completed_with_failures` status records only the planned `probes/not_run` and
`preprocessing_probes/not_run` entries. All 13 discovery graphs completed;
all 1,200 logged CI queries have finite valid p-values and no logged warnings.
Runtime was 2,450.499 seconds. These checks do not establish calibrated error
control.

The following scores are cross-assay adjacency counts. Their targets are
12 count-PAG adjacencies and seven activity-projection adjacencies, not the
full-panel targets of 21 and eight.

| Readout | Count TP / FP | Count recall | Activity TP / FP | Activity recall |
| --- | ---: | ---: | ---: | ---: |
| Independent paired | 1 / 0 | 1 / 12 | 6 / 0 | 6 / 7 |
| Paired anchor | 1 / 0 | 1 / 12 | 6 / 0 | 6 / 7 |
| scGLUE | 1 / 0 | 1 / 12 | 6 / 1 | 6 / 7 |
| scMRDR | 2 / 0 | 2 / 12 | 5 / 1 | 5 / 7 |
| True-state matching | 0 / 0 | 0 / 12 | 5 / 2 | 5 / 7 |
| Random matching | 0 / 0 | 0 / 12 | 0 / 0 | 0 / 7 |

Cross-assay precision is undefined for the empty predictions; it is not zero.
All count graphs have zero fully directed cross-assay arrows. scMRDR alone
has `exact_pair_set: true`: its predicted cross-assay pairs are
`H3K27ac_core–H3K4me3_core` and `H3K4me3_core–RNA_core`, exactly the two
biological direct pairs. Unlike the shared-state-only empty-target agreement,
this is a nonempty pair-set match. Its `exact_directed_set` is false. This is
one secondary identifiability-stress outcome, not proof of identified direct
mechanisms or reliable recovery, and it does not replace the primary RCIT
analysis.

KCI scGLUE activities return `H3K27ac_core → RNA_core` and
`H3K4me3_core → RNA_core`. Both assert correct ancestry in the stipulated
`Ac → Me3 → RNA` chain, but `Ac → RNA` is not a direct biological edge.
Changing the test backend changes graph results on fixed integrations: compared
with RCIT, paired-anchor count TP increases from zero to one and scMRDR from
one to two; activity TP changes scGLUE from five to six and scMRDR from six
to five, with one cross-assay FP each. This is backend sensitivity, not an
improvement to integration training.

State-supplied PC on 2,000 genuinely paired activities returns 7 TP, 1 FP, and
3 FN against its ten-edge oracle (precision 0.875; recall 0.70). RCIT returns
6 TP, 0 FP, and 4 FN. Both retain the two core-chain adjacencies without fully
orienting them. Directly supplying evaluator-only state is not an available
input to the learned integration methods.

Evidence: [constrained KCI report](runs/extended/constrained_chain_0/informative/primary/evaluation_suite_secondary_kci/report.json),
SHA-256 `badb7d8455cf748a2d2fbf049176e8f2b8ef35029cb18d03cab7e9d266ffd388`.
Review cross-checked source, fit/export metadata, truth bindings, pools,
assignments, metrics, and raw graph endpoints without reopening original
binary datasets or rerunning CI.

### Secondary anchor and readout-seed checks: both informative worlds

Both supplementary runs completed and passed independent production review.
Their conditions were frozen before supplementary execution, after primary
outcomes were known; they are not independently preregistered validation.
They reuse the original integrations and common discovery pools: 969
cells per assay in the linear world and 955 in the constrained nonlinear world.
No additional world or model fit was created. Each condition evaluates five
readouts on counts and activities, giving 80 completed graphs across the eight
conditions. All 6,662 logged CI queries returned valid results without logged
warnings; this is not evidence of calibrated CI error control.

The table reports cross-assay count-PAG TP / FP against 12 target adjacencies.
CI seeds 1 and 2 are the RNA-anchor readout repeats; their random-reference
permutation seeds are 3142 and 3143. Histone-anchor conditions use CI seed 0 and
random-reference seed 3141.

| World | Anchor | CI seed | scGLUE | scMRDR | Paired anchor | True-state | Random |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear chain | H3K27ac | 0 | 1 / 0 | 1 / 0 | 3 / 0 | 1 / 0 | 0 / 1 |
| Linear chain | H3K4me3 | 0 | 1 / 0 | 1 / 0 | 1 / 1 | 2 / 0 | 1 / 1 |
| Linear chain | RNA | 1 | 0 / 0 | 1 / 0 | 2 / 0 | 0 / 0 | 0 / 1 |
| Linear chain | RNA | 2 | 0 / 0 | 0 / 0 | 2 / 0 | 1 / 0 | 2 / 0 |
| Constrained chain | H3K27ac | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 | 1 / 0 |
| Constrained chain | H3K4me3 | 0 | 0 / 0 | 0 / 0 | 1 / 0 | 1 / 0 | 0 / 0 |
| Constrained chain | RNA | 1 | 1 / 0 | 1 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| Constrained chain | RNA | 2 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

All 40 count graphs have zero fully directed cross-assay arrows; none has an
exact match to the two-edge biological direct-pair or directed-edge set. An
empty prediction has undefined precision, not zero precision or a discovery
failure. Learned activity readouts recover four to six of seven projected
cross-assay adjacencies, with zero or one FP; some assert incorrect ancestry.
These activity results do not establish reliable direct causal recovery.

Changing the anchor preserves the selected cells and their within-assay values,
marginals, and detection counts in each unpaired readout. It changes the
three-assay coupling and row order; RCIT's first-500-row bandwidth calculation
can also be sensitive to this order. Paired-anchor references use the actual
cells belonging to the newly selected anchor, so their finite cell panel
changes; the underlying biological population is unchanged.
The RNA-anchor repeats keep learned, true-state, and paired-anchor input arrays
identical and change the CI randomization. Their random controls change both
the permutation and CI seed. Cached independent-paired and state-supplied
contexts are not rerun at CI seeds 1 and 2. None of these conditions is an
independent biological replication or a replacement for primary RCIT seed 0.

Review verified 2,626 assertions, all 284 newly sealed file hashes, 120 stage
receipts and their independent completion links, and the small readout arrays.
Original input/checkpoint identities were checked through their bindings;
this audit did not independently reread the original H5ADs or checkpoints.

Evidence: [linear supplementary report](runs/secondary/linear_chain_0/informative/anchor_readout_v1/secondary_report.json),
SHA-256 `905f8698e309d3f1526db1faa4f1a196dc646bcb168a6d7606ca0298a4b1de6a`;
[constrained supplementary report](runs/secondary/constrained_chain_0/informative/anchor_readout_v1/secondary_report.json),
SHA-256 `aaedfa7bccd8700d732bd6f85a5980d75b6bc8e0d3ebf97a30d5f357f9e24f8e`.

### Reviewed development synthesis

The [21:09 UTC development snapshot](runs/suite/synthesis/development_20261004T2109Z/summary.md)
inventories all 170 stages: 77 OK, 21 complete-with-failures, eight N/A, one
pending manual lock, and 63 blocked assessment stages. All 106 development
prerequisites are terminal. This inventory differs from prerequisite statuses
(53 complete, 21 complete-with-failures, 32 N/A) because declarative N/A stages
and file-backed N/A outcomes have different inventory labels.

Independent review verified the eight-file output set, all seven declared
output hashes, 17 scientific source pins, and 286 unique non-assessment artifact
hashes. It confirmed primary RCIT seed-0 selection, retention of both sealed
supplementary sets, and separate failed, undefined, skipped, and N/A values.
Assessment fields remain blocked and unpopulated. All 1,744 summary intervals
are null: this snapshot makes no development-world bootstrap inference. The
measured peak synthesis RSS was 3,281,688 KiB (about 3.13 GiB), recorded in the
[preserved resource profile](runs/suite/synthesis/development_20261004T2109Z_rss_profile.log).
Output-manifest SHA-256:
`79e4d8ce5606523131f0adaafa413f607dfe2785bcb706899df55b1dd7870a7a`.

### First locked assessment condition: linear chain 1, realistic counts

The first assessment pipeline completed under the unchanged lock and primary
RCIT seed-0 configuration. Independent metadata/source review passed. Both
integration fits succeeded; the report is `completed_with_failures` because
the scMRDR count graph failed. Matched readouts use the same 812 cells per
assay; independent genuinely paired references use 2,000 cells.

This table reports cross-assay TP / FP / FN against 12 count-measurement-PAG
and seven activity-projection adjacencies. These are association/projection
targets, not a count of recovered biological direct edges.

| Readout | Count TP / FP / FN | Activity TP / FP / FN |
| --- | ---: | ---: |
| Independent paired | 5 / 5 / 7 | 6 / 0 / 1 |
| Paired anchor | 1 / 1 / 11 | 5 / 0 / 2 |
| scGLUE | 0 / 3 / 12 | 4 / 1 / 3 |
| scMRDR | Unavailable | 4 / 0 / 3 |
| True-state matching | 0 / 1 / 12 | 4 / 1 / 3 |
| Random matching | 1 / 2 / 11 | 0 / 1 / 7 |

The sole failure is `LinAlgError: Eigenvalues did not converge` in scMRDR's
fourth logged count query, marginal `RNA_core–H3K27ac_decoy`, CI seed
2508518555. Its edges remain null and its graph scores unavailable, not an
empty returned graph or zero scores. Metadata alone does not establish the
numerical mechanism. The other 12 discovery graphs and all probes completed.
Count calibration passed its declared checks; this does not validate CI error
calibration.

All returned count graphs have zero fully directed cross-assay arrows.
scGLUE's three false cross-assay count edges connect `H3K4me3_sentinel` to
the three RNA variables. Both learned activity graphs assert unsupported
`H3K27ac_decoy → RNA_decoy` ancestry; scMRDR also asserts
`H3K4me3_decoy → RNA_decoy`. Neither learned activity graph recovers either
biological direct pair. All scored, returned exact biological-pair/direction
flags are false; the failed scMRDR count flags remain unavailable. Random
count overlap is not causal recovery.

State-supplied PC on paired activities gives 7 TP, 0 FP, and 3 FN against its
ten-edge oracle, with both core-chain adjacencies unoriented. Paired-count
false edges show that limitations cannot be attributed solely to integration.
This is one assessment condition, not an independent method ranking or a
world-level reliability estimate. The informative condition has also completed
as reported below; the remaining frozen block continues without changing settings.

Evidence: [first realistic assessment report](runs/extended/linear_chain_1/realistic/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `998b65a585e919e753b9319fa7a41693fe646b181841b8a4069502f400ebd8d6`.
Review checked lock/source/configuration, native train-input and checkpoint/
export metadata, world/truth bindings, common-pool uniqueness, raw endpoints,
and score arithmetic. It did not reopen original binaries or rerun science.

### Same locked world: informative counts

The informative condition also completed and passed independent metadata/source
review. All 13 discovery graphs and probes completed; all 1,267 logged CI
queries returned finite valid p-values without logged warnings. This does not
establish calibrated error control. Matched readouts use 963 cells per assay;
independent paired references again use 2,000. Targets remain 12 count-PAG and
seven activity-projection cross-assay adjacencies.

| Readout | Count TP / FP / FN | Activity TP / FP / FN |
| --- | ---: | ---: |
| Independent paired | 1 / 0 / 11 | 6 / 0 / 1 |
| Paired anchor | 1 / 0 / 11 | 6 / 0 / 1 |
| scGLUE | 1 / 2 / 11 | 4 / 1 / 3 |
| scMRDR | 0 / 0 / 12 | 5 / 0 / 2 |
| True-state matching | 0 / 0 / 12 | 5 / 0 / 2 |
| Random matching | 0 / 0 / 12 | 0 / 0 / 7 |

scMRDR counts retain the within-assay edge
`H3K4me3_core o-o H3K4me3_decoy`. Their empty cross-assay prediction has
undefined precision and zero cross-assay recall, not a discovery failure or
an entirely empty graph. No count graph returns a fully directed cross-assay
arrow; all count exact biological-pair/direction flags are false.

scGLUE activities still assert unsupported
`H3K27ac_decoy → RNA_decoy` ancestry. scMRDR activities have no fully directed
cross-assay arrows. They recover the biological adjacency
`H3K4me3_core–H3K27ac_core`, but miss the other biological direct pair,
`H3K27ac_core–RNA_core`. Finding one adjacency does not identify its mechanism
or direction, and neither learned activity graph exactly recovers the direct
pair set.

Both conditions share identical truth receipts, independent paired activity
results, and state-supplied results. The selected matched panels differ: RNA
overlap is 812 cells, H3K27ac 785, and H3K4me3 781. These are two conditions of
one world, not independent biological replications or an isolated comparison
of count depth with everything else fixed. All nine scored features survive
both methods' input processing; scMRDR mappings are singleton.

Evidence: [informative assessment report](runs/extended/linear_chain_1/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `362490d6efc1008b241bc2a8efba0738af9a4e743502141c12320eaab1be9c97`.
Review verified frozen settings, calibration gates, source/world/truth bindings,
fit/export/input-retention metadata, raw endpoints, and score arithmetic without
binary reopening, scientific reruns, or setting changes. The operator then moved
to `linear_chain_2`; this partial assessment is not a full-suite conclusion.

### Second locked assessment world: linear chain 2, realistic counts

This condition completed with both integration fits successful and three
count-discovery failures preserved. Independent source/metadata/score review
passed. The common matched panel contains 771 cells per assay; the independent
paired reference contains 2,000. Targets again have 12 count-PAG and seven
activity-projection cross-assay adjacencies.

| Readout | Count TP / FP / FN | Activity TP / FP / FN |
| --- | ---: | ---: |
| Independent paired | Unavailable | 6 / 0 / 1 |
| Paired anchor | 2 / 3 / 10 | 6 / 0 / 1 |
| scGLUE | Unavailable | 5 / 0 / 2 |
| scMRDR | 1 / 0 / 11 | 6 / 0 / 1 |
| True-state matching | 0 / 1 / 12 | 6 / 1 / 1 |
| Random matching | Unavailable | 0 / 0 / 7 |

The three count failures occur in independent paired, scGLUE, and random
readouts. Each fails its third marginal query, `RNA_core–H3K27ac_core`, with
`LinAlgError: Eigenvalues did not converge`. Their edges and graph metrics
remain unavailable, not empty graphs, zero scores, or planned skips. The
independent paired failure means the problem cannot be attributed solely to
integration. Ten graphs and all probes completed. Logs contain 1,064 successful
CI queries and three failed queries, with no logged warnings; successful
execution does not establish CI calibration.

This world's biological chain is
`H3K27ac_core → H3K4me3_core → RNA_core`. scMRDR activities retain both
biological adjacencies, but also four additional cross-assay pairs; their
exact biological-pair set is incorrect and they have no fully directed
cross-assay arrows. scGLUE activities retain only the first biological
adjacency and assert unsupported `RNA_decoy → H3K27ac_decoy` ancestry.
The scMRDR count true positive is `H3K4me3_decoy–RNA_core`, not a biological
direct pair. No returned count graph has a fully directed cross-assay arrow
or an exact biological-pair/direction match. Failed count graphs have
unavailable stress-test scores.

All nine scored features remain represented; scMRDR mappings are singleton.
The review checked scheduled command/attempt/log bindings, five sealed JSON
artifacts, all 17 frozen scientific source pins, configurations, truth/input
metadata, native fit/export receipts, pool uniqueness/order, raw endpoints,
and score arithmetic. Original binary inputs and checkpoints were not
reopened. The informative condition has also completed and passed independent
review, as reported below. These results do not support a full-suite
reliability estimate or an assessment-driven setting change.

Evidence: [second realistic assessment report](runs/extended/linear_chain_2/realistic/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `9d88e6a9b992a70471b0667f86a507aabd76f1e0c338b0d0a30a78bed56dfd05`.

### Linear chain 2: informative counts

The informative condition completed all 13 graphs and probes, with 1,205
valid logged CI queries and no failures or warnings. Calibration passed its
declared checks; neither this nor successful execution establishes calibrated
CI error control. The independently reviewed matched panel has 958 cells per
assay; independent paired references use 2,000.

| Readout | Count TP / FP / FN | Activity TP / FP / FN |
| --- | ---: | ---: |
| Independent paired | 2 / 1 / 10 | 6 / 0 / 1 |
| Paired anchor | 1 / 0 / 11 | 6 / 0 / 1 |
| scGLUE | 1 / 0 / 11 | 6 / 0 / 1 |
| scMRDR | 2 / 0 / 10 | 6 / 0 / 1 |
| True-state matching | 1 / 0 / 11 | 6 / 2 / 1 |
| Random matching | 0 / 0 / 12 | 0 / 0 / 7 |

Targets remain 12 count-PAG and seven activity-projection cross-assay
adjacencies. Random matching's empty cross-assay predictions have undefined
precision and zero recall; its graphs retain within-assay edges and are not
failed discoveries.

Both learned activity graphs retain the two biological chain adjacencies,
plus four additional shared-state cross-assay pairs. scGLUE's cross-assay count edge is
`H3K27ac_decoy–H3K4me3_decoy`, not a biological direct pair. scMRDR counts retain
that pair and `H3K4me3_core–RNA_core`, but miss `H3K27ac_core–H3K4me3_core`.
No compared count or activity graph returns a fully directed cross-assay
arrow. All exact biological-pair/direction flags remain false.

Truth and independent paired/state activity results match the realistic
condition, but the matched panels differ: overlap is 771 RNA cells, 738
H3K27ac cells, and 734 H3K4me3 cells. The regimes are conditions of one world,
not independent replications or a count-depth-only comparison. Independent
review verified command/attempt/log/artifact bindings, all 17 frozen source
pins, configuration, truth/input metadata, native fit/export receipts,
retention of all nine scored features, pool uniqueness/no reuse, raw endpoints,
and score arithmetic. It did not reopen original binaries or rerun science.

Evidence: [second informative assessment report](runs/extended/linear_chain_2/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `a02c1ce90f6b2407165fec2c59af3c64e7ca2bbfb2ec28c2051eb2b4c331d33d`.
Both regimes of the first two assessment worlds are now reviewed; the unchanged
operator has started `linear_chain_3`. The remaining assessment block and
final world-level synthesis are still required.

### Third locked assessment world: linear chain 3, realistic counts

Both integration fits completed; independent read-only review passed for the
evaluation, which preserves two count-discovery failures. The common matched
panel contains 791 cells per assay. Targets have 12 count-PAG and seven
activity-projection cross-assay adjacencies; the independent paired reference
uses 2,000 cells.

| Readout | Count TP / FP / FN | Activity TP / FP / FN |
| --- | ---: | ---: |
| Independent paired | 6 / 7 / 6 | 7 / 0 / 0 |
| Paired anchor | Unavailable | 6 / 0 / 1 |
| scGLUE | Unavailable | 3 / 0 / 4 |
| scMRDR | 2 / 1 / 10 | 4 / 0 / 3 |
| True-state matching | 1 / 4 / 11 | 4 / 1 / 3 |
| Random matching | 2 / 5 / 10 | 0 / 0 / 7 |

Paired-anchor counts stop before CI testing because `H3K27ac_sentinel` is
constant. scGLUE counts fail their fourth marginal query,
`RNA_core–H3K27ac_decoy`, with `LinAlgError: Eigenvalues did not converge`,
CI seed 2508518555. Both retain null edges and unavailable graph metrics;
neither is an empty graph. Eleven graphs and all probes completed, with
1,654 successful CI queries, one failed query, and no logged warnings.
Declared count-calibration checks passed; this does not establish CI calibration.

This world's biological chain is
`H3K4me3_core → H3K27ac_core → RNA_core`. Neither learned activity graph
retains either biological direct adjacency, despite its projected-adjacency
true positives. scMRDR's two count true positives are not biological direct
pairs; its false count adjacency involves a sentinel. All returned count
graphs and both learned activity graphs have zero fully directed cross-assay
arrows. All six-readout exact biological-pair/direction flags are false where
available. Random count overlap is not causal recovery; random activities
retain three within-assay edges but have an empty cross-assay set with
undefined precision.

True-state matching activities assert unsupported
`H3K4me3_decoy → H3K27ac_decoy` and `RNA_decoy → H3K27ac_decoy` ancestry.
This is different from the paired-activity diagnostic with true state supplied
to PC: that assisted control recovers both correctly directed core edges,
with overall TP/FP/FN 9/0/1 against its ten-edge oracle. The latter uses genuine
cell pairing and includes true state as variables in PC. It is an assisted
diagnostic on hidden activities, not a result of matching disjoint cells or
proof of recovery from observed counts.

Review verified attempt/command/log bindings, five sealed JSON artifacts,
17 frozen source pins, configuration, world/truth metadata, fit/export
receipts, retention of all nine scored features, ordered unique common pools,
raw endpoints, and score arithmetic. Original binaries were not reopened;
no science was rerun and no informative-condition artifact was inspected.

Evidence: [third realistic assessment report](runs/extended/linear_chain_3/realistic/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `9fd98f7f170560aa9ce7a0a9b6a456543f1bced4cac089efe063808910eeec12`.
The informative condition also completed and passed independent review, as
reported below. The remaining frozen block continues without a configuration
change. Final synthesis and independent interpretation remain pending.

### Linear chain 3: informative counts

All 13 graphs and probes completed, with 1,387 valid logged CI queries and no
failures or warnings. Declared calibration checks passed. These checks do not
establish calibrated CI error control. The common matched panel contains 962
cells per assay; the independent paired reference uses 2,000. Cross-assay
targets remain 12 count-PAG and seven activity-projection adjacencies.

| Readout | Count TP / FP / FN | Activity TP / FP / FN |
| --- | ---: | ---: |
| Independent paired | 1 / 1 / 11 | 7 / 0 / 0 |
| Paired anchor | 0 / 0 / 12 | 7 / 0 / 0 |
| scGLUE | 0 / 0 / 12 | 5 / 0 / 2 |
| scMRDR | 0 / 0 / 12 | 5 / 0 / 2 |
| True-state matching | 0 / 1 / 12 | 5 / 1 / 2 |
| Random matching | 1 / 0 / 11 | 0 / 0 / 7 |

Paired-anchor and both learned count graphs contain only the within-assay
edge `H3K4me3_core o-o H3K4me3_decoy`. Their empty cross-assay predictions
have undefined precision and zero recall; they are completed graphs, not
failures or entirely empty graphs.

scGLUE activities retain neither biological chain adjacency. scMRDR activities
retain only `H3K4me3_core–H3K27ac_core`. Both learned activity graphs and all
returned count graphs have zero fully directed cross-assay arrows. True-state
matching activities assert four unsupported directions, including the reversed
core edge `H3K27ac_core → H3K4me3_core`. All six-readout exact biological-pair/
direction flags are false. Random counts happen to include
`H3K27ac_core–RNA_core`; this overlap under random coupling is not evidence of
structure preservation.

The separate PC diagnostic again correctly directs both core edges, with
overall TP/FP/FN 9/0/1. It uses genuinely paired hidden activities and includes
true state as tested variables, unlike the disjoint-cell matching readouts.
Realistic/informative truth and independent paired/state activity results
agree. Their selected matched panels differ: overlap is 791 RNA cells, 760
H3K27ac cells, and 761 H3K4me3 cells. These regimes are not independent
replications or a count-depth-only comparison.

Independent review verified attempt/command/log/artifact bindings, 17 frozen
source pins, configuration, world/truth metadata, native fit/export receipts,
all nine retained features, ordered unique common pools, raw endpoints, and
score arithmetic. It did not reopen original binaries or rerun science.

Evidence: [third informative assessment report](runs/extended/linear_chain_3/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json),
SHA-256 `9c974b5ce40f57dff97fe80cdfaf1395c38beffe533b4830c6b96cc7702614ce`.
Both regimes of all three linear-chain assessment worlds are now reviewed.
Both completed linear-collider assessment worlds have also received the
bounded follow-up reviews below. The remaining assessment conditions and
final synthesis are incomplete.

### Linear colliders 1 and 2: follow-up review on 2026-10-25

These are saved results from the frozen baseline suite, not new causal tests
of the integration-validation settings. Activity scores use the projected
activity graph's ten cross-assay adjacencies; count scores use the measurement
PAG's twelve. Neither target is simply the two planted direct cross-assay edges.
Each entry reports true positives / false positives / false negatives.

| World | Count regime | Method | Activity TP / FP / FN | Count TP / FP / FN |
| --- | --- | --- | ---: | ---: |
| `linear_collider_1` | realistic | scGLUE | 5 / 0 / 5 | Failed; unavailable |
| `linear_collider_1` | realistic | scMRDR | 6 / 0 / 4 | 2 / 4 / 10 |
| `linear_collider_1` | informative | scGLUE | 5 / 0 / 5 | 0 / 0 / 12 |
| `linear_collider_1` | informative | scMRDR | 6 / 0 / 4 | 1 / 0 / 11 |
| `linear_collider_2` | realistic | scGLUE | 3 / 0 / 7 | 4 / 5 / 8 |
| `linear_collider_2` | realistic | scMRDR | 4 / 1 / 6 | 2 / 3 / 10 |
| `linear_collider_2` | informative | scGLUE | 4 / 0 / 6 | 2 / 0 / 10 |
| `linear_collider_2` | informative | scMRDR | 6 / 1 / 4 | 1 / 0 / 11 |

The realistic scGLUE count readout in `linear_collider_1` failed with
`LinAlgError: Eigenvalues did not converge`. Its graph and graph metrics are
unavailable. The informative scGLUE count readout in that world completed
with within-assay edges and no cross-assay edges: its cross-assay precision
is undefined and recall is zero. These are different outcomes.

All seven completed learned count graphs return zero fully directed
cross-assay edges. Against the two biological direct arrows, their separately
labeled direction stress scores are therefore 0 / 0 / 2, not successful
direction recovery. In `linear_collider_2`, informative activity graphs assert
three incorrect biological directions for scGLUE and five for scMRDR, with
zero correct directions. Adjacency agreement does not imply correct ancestry.

The paired controls also struggle with counts. In `linear_collider_2`, the
independent paired informative count graph returns no cross-assay adjacencies
despite a nonempty whole-panel graph; its paired activity graph recovers seven
of ten cross-assay adjacencies and both planted direct arrows. These results
do not isolate integration as the sole cause of count-graph failure. Regimes
reuse a world and have different retained cell pools, so they are not
independent biological replications or a count-depth-only comparison.

The [first collider review](../../../project_resources/research/integration_validation_2026_10_25/reviews/frozen_baseline_linear_collider_1_followup_audit.json)
passed 151 checks, including public export/model/world/source bindings and
recomputed scores. Its additional private-input receipt checks used metadata
only; private binary arrays were not opened or rehashed. The
[second collider review](../../../project_resources/research/integration_validation_2026_10_25/reviews/linear_collider_2_frozen_baseline_json_audit.md)
recomputed graph scores and checked JSON artifact and stage bindings; it did
not reopen binary matrices or checkpoints. Neither review reran science or
revalidated oracle generation. These baseline outcomes are excluded from
selection in the new integration-only study.

### Constrained nonlinear chain 1: realistic baseline continuation

The original `primary/seed0` baseline stage completed on 2026-10-26 after the
preserved operational retry. It uses scGLUE `lam_align=0.05` and scMRDR's
recorded 5/10/10 configuration. These are the frozen assessment settings;
new integration-validation settings were not used for this causal evaluation.
The [stage report](runs/extended/constrained_chain_1/realistic/primary/evaluation_scglue_scmrdr_seeds0/report.json)
has status `completed`, with no failed integrations or diagnostics.

All matching comparisons use 792 cells per assay. The activity target is the
nine-variable latent projection with eight adjacencies, seven cross-assay.
The count measurement target has 21 adjacencies, twelve cross-assay, including
assay-exposure effects. The table scores adjacency against the corresponding
layer's target; it does not score every edge against the two-edge biological
chain.

| Readout | Cells | Activity cross TP/FP/FN | Count cross TP/FP/FN |
| --- | ---: | --- | --- |
| scGLUE | 792 | 4/1/3 | 2/1/10 |
| scMRDR | 792 | 5/1/2 | 4/2/8 |
| Paired anchor | 792 | 5/0/2 | 1/5/11 |
| True-state matching | 792 | 7/1/0 | 2/1/10 |
| Random matching | 792 | 0/0/7 | 2/2/10 |
| Independent paired assessment | 2,000 | 6/0/1 | 6/9/6 |

Count adjacency recovery remains weak even in the paired controls; those
errors cannot be attributed solely to integration. The independent paired
control also has a larger sample, so its difference from the anchor control
does not isolate pairing quality. True-state matching still combines different
cells and does not recover their residual joint distribution. The random
activity graph has within-assay edges despite having no cross-assay edges;
it is not an empty or failed graph.

The biological chain is H3K4me3→H3K27ac→RNA. Neither learned integration
recovers a correct directed biological edge in the separately labeled
identifiability stress test. Activity directed TP/FP/FN is 0/1/2 for scGLUE and
0/2/2 for scMRDR; both count readouts give 0/0/2. These results are distinct
from recovery of the graphical projections and are not an expected-success
criterion for arbitrary unpaired data. Main activity graphs have no sentinel
edges. Count graphs have one scGLUE and two scMRDR sentinel edges, with false
positive rates against the count target of 1/15 and 2/15. This target allows
genuine within-assay exposure-induced sentinel adjacencies.

The [independent scientific review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_realistic_frozen_baseline_json_audit.md)
reproduces graph/endpoint/sentinel score arithmetic, checks all sixteen saved
evaluation-stage bindings and verifies 1,333 exact-oracle separation labels
by graph traversal. It does not rerun statistical CI tests, probe fits or raw
array calculations. The [stage-binding review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_realistic_baseline_stage_completion_binding_audit_2026_10_26.json)
and separate native-fit audits bind the completed outputs and retained failed
attempts. Saved retention probes use same-cell hidden targets on independent
assessment cells; they are not cross-assay fixed-decoder transfer tests.
Nominal CI p-values after matching remain diagnostic. This one cached world
does not establish reliable causal recovery.

### Constrained nonlinear chain 1: informative baseline continuation

The [informative report](runs/extended/constrained_chain_1/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json)
also completed with no failed integrations or diagnostics, using the same
original settings and biological world. Its matched panels contain 971 cells
per assay. The activity and count graph targets are unchanged from the
realistic condition above.

| Readout | Cells | Activity cross TP/FP/FN | Count cross TP/FP/FN |
| --- | ---: | --- | --- |
| scGLUE | 971 | 5/1/2 | 0/0/12 |
| scMRDR | 971 | 6/1/1 | 0/0/12 |
| Paired anchor | 971 | 6/0/1 | 1/0/11 |
| True-state matching | 971 | 5/1/2 | 0/0/12 |
| Random matching | 971 | 0/1/7 | 0/0/12 |
| Independent paired assessment | 2,000 | 6/0/1 | 1/0/11 |

Both learned count graphs contain two within-assay edges despite returning
zero cross-assay edges. Their cross-assay precision is undefined and recall
is zero; they are completed graphs, not execution failures. Paired count
controls recover only one of twelve target cross-assay adjacencies, again
showing limitations beyond integration. Each learned count graph has one
sentinel-incident edge permitted by the measurement target and no sentinel
false positive against that target.

Neither learned integration returns a correct directed biological edge.
Activity directed stress TP/FP/FN is 0/2/2 for scGLUE and 0/0/2 for scMRDR;
both count readouts give 0/0/2. scGLUE's H3K4me3→RNA activity arrow follows
an ancestral path but is false against the direct-edge target. The two edge
meanings must remain distinct.

Activity adjacency overlap increases relative to realistic counts, but count
overlap decreases to zero and scGLUE's exact activity endpoint agreement also
declines. The regimes use different retained pools, so this is neither an
isolated depth comparison nor an independent biological replication. High
same-cell state-retention probe scores do not validate the matched residual
joint distribution.

The [independent scientific review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_informative_frozen_baseline_json_audit.md)
checks graph/endpoint/sentinel arithmetic, all sixteen evaluation-stage
bindings, 1,333 graph-only oracle labels and 1,114 saved statistical CI entries.
It does not reopen arrays or checkpoints, refit probes, or rerun CI tests.
These are original-baseline outcomes; they did not select the new integration
settings or establish reliable cross-assay causal recovery.

## Extended study and remaining work

The 30 specifications and mechanisms are frozen. All nine development
specifications have been generated once and validated. The six causal worlds
have completed both realistic and informative seed-0 evaluations. The three
chain-world seed/guidance control blocks and exact ambiguity comparison are
also complete and reviewed. These remain development observations.

The simple three-seed first pass is complete, including its preserved CI
failure. Shared-state controls and both supplementary diagnostic runs have
also completed, and their independent supplementary and synthesis reviews passed.
The reviewed development evidence lock and independent handoff check are saved;
the frozen assessment operator subsequently stopped at a count-calibration
failure in `linear_collider_3`. Continued `constrained_chain_1` attempts retained
their native CUDA failures and partial outputs. The bounded retry has now
completed both regimes, bringing the original suite to 118 of 178 model fits.
See the [October 26 execution snapshot](IMPLEMENTATION_STATUS.md#earlier-execution-snapshot-2026-10-26)
for the completed, separate integration-validation matrix. Remaining work includes
unfinished assessment conditions, final synthesis, and their independent
reviews. The frozen schedule lists 178
unique fits in total; seeds, regimes, and ambiguity counterparts are not
independent biological trials.
All GPU work is serialized. No unfavorable world is replaced, and no
assessment-world results have been used for configuration selection.

An additional extended-study limitation is count-based QC selection. For
example, scGLUE retained 969 of the first 1,000 RNA discovery cells in the
informative linear world. Common-pool scoring compares methods on the same
eligible cells, but does not eliminate selection-induced dependencies. The
saved oracle describes the original source measurement DAG/PAG, not an explicit
selection-augmented graph conditioned on QC. Its agreement scores therefore
combine measurement, selection, testing, and integration/readout effects.
The simple seed-0 population retained all 2,000 cells per assay, so this specific
QC discrepancy does not explain the simple experiment's false calls.
