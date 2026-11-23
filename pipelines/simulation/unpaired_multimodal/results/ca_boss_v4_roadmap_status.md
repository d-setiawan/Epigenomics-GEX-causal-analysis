# CA-BOSS v4 roadmap status

_Existing-world-only checkpoint · dual-null stability bridge through v4.59; controlled core-estimator diagnostics through v4.64_

## Current result

### Project progress summary (2026-09-06)

This is the starting point for the **unpaired-multimodal simulation track**.
It records completed work through v4.64, not a new benchmark run. The
recommended cross-assay checkpoint remains **v4.59**; v4.32 remains the
assay-local production implementation. Later version numbers denote research
experiments, not automatically better or promoted models.

The result so far is conservative recovery of shared-state relationships:
77.35% hyperedge recall and 65.05% exact-pair recall, with no observed false
reported hypotheses in the current cached campaign. This is not complete
recovery or evidence of a directed cross-assay causal graph. The
[metric table](#cross-assay-benchmark-metrics) gives every denominator, and
[reading cross-assay accuracy](#reading-cross-assay-accuracy) explains the
difference between exact pairs and ambiguity sets.

| Stage completed | What changed | What the work established |
| --- | --- | --- |
| Frozen CA-BOSS v3 baseline | Channel-aware search separated same-assay from disjoint-assay evidence | Improved attainable-motif recovery over production FCI, but exposed dense graphs and fragile sparse H3K4me3 detection |
| Explicit state and sparse-count modeling | Held-out background state, state/depth-adjusted focal models, partially pooled Bernoulli/beta-binomial evidence, and empirical nulls | Replaced the loose Fisher rescue and separated residual direct candidates from state association |
| Assay-local layers through v4.32 | State membership, observed collider structure, loading-group uncertainty, and set-valued outputs | Strong cached within-assay results, without converting state-equivalent alternatives into direct edges |
| Cross-assay bridge through v4.59 | Background component distribution matching, global permutation synchronization, feature-fold agreement, and null-calibrated stability | The current scored shared-state checkpoint, with exact claims and unresolved alternatives reported separately |
| Core-estimator diagnostics v4.60–v4.64 | Depth-conditioned geometry, characteristic functions, representation refits, soft reporter evidence, and joint reporter regression | Identified instability and tested specific mechanisms; none has replaced v4.59 |

The detailed progression and counts remain in
[earlier layers and the version-by-version results](#earlier-production-layers-v422v432).
The original requirements are assessed in the
[roadmap completion audit](#roadmap-completion-audit), including the
still-unimplemented probabilistic cell-to-metacell membership component.
The [experiment log](channel_aware_boss_v4_experiment_log.md) preserves dated
successful, failed, and superseded trials, source hashes, tests, and artifact
locations; its early fresh-world protocols are historical, not the current
permission to generate more data.

The current data boundary is unchanged: reuse existing worlds, with six
chromatin assays plus RNA; infer state from observed background features;
use observed focal measurements and depth; do not add paired cells,
interventions, external biological information, simulator state, or known
cross-assay correspondences to discovery. Other-block simulation truth is
used to identify calibration nulls, and target truth is used for scoring
after prediction freeze. Thus "observed-only discovery" does not mean that
the whole benchmark is label-free.

Independent reviews in the recorded work checked both implementation and
scientific interpretation. They include source/input/output binding,
separate scoring replay, matched-subset controls, missing-fit denominators,
and distinctions between working uncertainty scores and calibrated
posteriors. This documentation update does not rerun those reviews or tests.
The latest completed controlled comparison is v4.64; its unchanged primary
pair set is a negative result, not an improvement. The next recorded research
direction changes the background representation to a count likelihood; no
recovery gain from that prototype is established here.

For the broader project's real-data integration, partition-instability, and
biological-validation history, start with the
[central README](../../../../README.md) and [master roadmap](../../../../project_resources/research/ROADMAP.md).
Those results and external validation resources are a separate track, not
extra inputs to this simulation algorithm. The consolidated claim boundaries
are in [LIMITATIONS L16](../../../../project_resources/research/LIMITATIONS.md#l16-ca-boss-cached-shared-state-recovery-is-not-directed-causal-certification).

### Checkpoint algorithm and evidence chain

The production assay-local layers remain at v4.32. The recommended cached
cross-assay checkpoint is now CA-BOSS v4.59. V4.44–v4.47 remain its
conservative base: latent components come only from held-out background
features; reporter permutations are synchronized globally in the full fit and
two disjoint feature folds; reporter registration remains separate from
assay-local direct structure; and target-block pair scores must exceed the
largest eligible other-block null score. V4.49–v4.53 add sparse unresolved-
assay proposals, whole-world stability selection, a directional empirical-
copula score, and strict mutual-nearest-assay competition. Overlapping selected
pairs remain one connected, unexpanded ambiguity hyperedge.

V4.54 adds distribution shape to that bridge. For each assay-local rank-three
ICA component in the full fit and both disjoint background-feature folds, it
compares 99 empirical quantiles with a sign-invariant two-Wasserstein cost.
Each assay-pair cost matrix is median normalized, and the existing exact
multi-assay permutation solver produces global component co-assignments and
pair-exclusion margins. Four fixed signals—statistical support, W2 global
co-assignment count, maximum normalized W2 reporter cost, and mean W2
exclusion margin—enter the same directional empirical-copula score.

Every proposal must be the strict unique mutual nearest neighbour of both
endpoint assays in all four leave-one-calibration-block folds. Its minimum
endpoint-competition margin must strictly exceed every analogous other-block
null margin, and the full rule must recur in at least 80 of 100 whole-world
bootstrap refits. Features are quantized to 12 decimals before empirical
ranking so theoretically tied ranks do not depend on TSV serialization. The
rule assumes neither a perfect matching nor a fixed relationship count and
constructs no cross-assay cell coupling.

V4.55 retains those nominal W2 proposals but calibrates stability against the
observed failure rate rather than applying 0.80 everywhere. For each sealed
target block, null pairs in each of the four non-target blocks are rescored in
folds whose labels exclude both that candidate block and the target. A target
proposal must recur in strictly more than 50 of 100 refits, strictly exceed the
largest null count, and reach at least 0.975 working Beta(1,1) probability of a
larger selection probability. Bootstrap replicates are dependent, so the Beta
quantity is an uncertainty-aware working score, not a calibrated posterior.
Connected additions are still merged into unexpanded ambiguity hyperedges.

V4.56 tested joint sliced-distribution geometry on two random 40-world panels.
The raw and copula forms each admitted false pairs, while their intersection
added nothing in the remaining 23 worlds. V4.57 independently froze the raw
joint evidence, but its global empirical-null stability rule selected nothing:
the worst null recurred in 96--100 of 100 refits in every target block. Both
variants are rejected diagnostics.

V4.58 retains the raw joint proposals only when both endpoints are strict local
mutual nearest neighbours in the full component-score fit and both disjoint
held-out-background-feature folds. Null calibration is restricted to pairs in
that same observable three-of-three context stratum. A proposal must recur in
more than 50 of 100 refits, strictly beat the stratum-specific worst null, and
pass the same 0.975 working-Beta dominance guard. It is unioned with target-safe
v4.55 and overlapping pairs remain one unexpanded ambiguity hyperedge. The
rule assumes neither a perfect matching nor a fixed relationship count.

V4.59 adds an exactly-two-context child stage. It requires a proposal to
strictly beat both the same-stratum and unstratified global null counts, in
addition to the existing strict majority and working-Beta dominance checks.
Only the 100/100 H3K4me1–H3K4me3 proposal in world 49011 passes, exceeding
null counts 88 and 96. Independent review verified the calibrations and full
103-world scoring; a separate replay reproduced all bound score outputs.

The design draws on Sturma, Squires, Drton, and Uhler's [unpaired multi-domain
causal representation learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html),
Liu et al.'s [MMD-MA](https://doi.org/10.4230/LIPIcs.WABI.2019.10), Demetci et
al.'s [SCOT](https://doi.org/10.1089/cmb.2021.0446), Pachauri et al.'s
[permutation synchronization](https://proceedings.neurips.cc/paper_files/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html),
Tang and Jebara's [multiway matching](https://proceedings.mlr.press/v54/tang17a.html),
Yi and Liu's [sliced Wasserstein geometry](https://proceedings.mlr.press/v189/yi23a.html),
Meinshausen and Bühlmann's [stability selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x),
Li et al.'s [empirical-copula tail scoring](https://doi.org/10.1109/ICDM50108.2020.00135),
and Haghverdi et al.'s [mutual-nearest-neighbour matching](https://doi.org/10.1038/nbt.4091).
These are methodological adaptations, not direct applications or claims under
a cited identifiability theorem. No publication supplies data, a cell/feature
correspondence, a biological prior, or a simulator label.

### Cross-assay benchmark metrics

The v4.59 campaign reuses 103 disjoint cached worlds across five previously
selected panels and creates no new world or dataset. Target-block truth is
opened only after the evidence and calibration chain is hash verified.
Other-block simulation truth identifies empirical-null pairs only; positive
labels do not fit the score. Because all five blocks and the W1/W2, joint-
geometry, and context-stratum diagnostics informed development, this remains retrospective
cross-fit evidence rather than fresh locked certification. The hash-verified
micro-aggregate is:

| Requested metric family | V4.59 cached dual-null campaign |
| --- | ---: |
| Shared-state hyperedges | 239 TP, 0 FP, 70 FN; precision 1.000, recall 0.7735 |
| Exact assay participation | 402/402 correct of 618 truth participations; precision 1.000, recall 0.6505 |
| Sentinel false-positive rate | 0/1,236 sentinel-null feature opportunities |
| Equivalence-class coverage | 239/309 = 0.7735 at coverage precision 1.000 |
| Ambiguity size | 309 candidate feature-pair alternatives over 239 hypotheses; mean 1.293, maximum 6 |
| Exact pair stress test | 201/201 exact claims correct of 309 truth pairs; precision 1.000, recall 0.6505 |
| Direction stress test | 0 claims; direction is undefined for these shared-state truths |

The exact pair and direction family is labeled
`identifiability_stress_test_not_expected_success_criterion` and excluded from
the reliability gate. Thirty-eight set-valued participation hypotheses remain
point-estimate abstentions, not expanded predictions. Thus 1.000 is
hyperedge-level precision; it does not assert that every candidate alternative
inside an ambiguity set is true.

### Reading cross-assay accuracy

Of the 309 truth relationships, **201 have an unambiguous exact-pair match**,
**38 more are covered by a set-valued hypothesis**, and **70 are missed**.
The first two categories sum to the 239 matched hyperedges. The 38 ambiguous
hypotheses contain 108 pair alternatives; together with the 201 exact claims,
these account for the 309 alternatives used to calculate ambiguity size.
The equality between the number of alternatives and the number of truths is
incidental: it is not complete recovery. Alternatives are not extra edges.

Precision answers "how many reported hypotheses matched truth?" Recall
answers "how many truth relationships were recovered?" Here, 100% observed
hyperedge precision means each selected hypothesis matches a distinct truth
under the set-matching scorer, not that every alternative inside it is true.
The exact-pair score credits only unambiguous matches, which are still
shared-state pairings rather than demonstrated direct causal edges.
Direction recovery is undefined for this truth family; zero direction claims
must not be presented as either 0% or 100% direction accuracy.

The exact assay-participation denominator counts membership in individual
relationships. It is different from the earlier broad question of whether
an assay participates in any shared biological state. Similarly, the
within-assay results below use a larger cached cohort; they are not additional
cross-assay successes and must not be pooled into the cross-assay denominator.
The zero sentinel rate applies to this campaign's sentinel-feature
opportunities, not every historical null experiment.

The authoritative machine source is the
[v4.59 checkpoint record](ca_boss_v4_bootstrap_dual_null_disagreement_rescue_v459_summary.json).
These are retrospective development results: reused worlds informed model
choices, and calibration-world bootstrap does not by itself propagate target
representation uncertainty. No new locked or real-data certification is
claimed; see [LIMITATIONS L16](../../../../project_resources/research/LIMITATIONS.md#l16-ca-boss-cached-shared-state-recovery-is-not-directed-causal-certification).

V4.58 adds 25 true and zero false hypotheses over v4.55. The target-block
same-stratum worst-null counts are 0, 1, 78, 0, and 0 of 100 refits. The
three-of-three stratum was selected after truth-separated inspection of the
cached proposals: it retained 26/27 true proposals and excluded all three
false proposals, so this is development evidence rather than independent
confirmation. The compact machine record is
[`ca_boss_v4_bootstrap_joint_context_stratified_stability_rescue_v458_summary.json`](ca_boss_v4_bootstrap_joint_context_stratified_stability_rescue_v458_summary.json).
The current v4.59 child adds one further true hypothesis with zero false
additions; its [machine record](ca_boss_v4_bootstrap_dual_null_disagreement_rescue_v459_summary.json)
binds the new source, calibration, predictions, and score artifacts. Seventy
truth hyperedges remain missed, so the full recovery goal is incomplete.

### Current model experiments and independent review

V4.60's observed-depth-conditioned geometry produced only false additions on
a random 25-world panel and was rejected. V4.61's marginal characteristic-
function summary combined with joint sliced-W2 produced five stable true
additions and one false addition across 65 distinct cached worlds. It remains
unpromoted. The raw and standardized characteristic-function views coincide
to numerical precision because the ICA input is already standardized. The
five true additions are new beyond v4.58, but the proposed three-context
global guard was recognized after their outcomes were inspected.

Independent statistical and published-method reviews identified a different
uncertainty source: calibration-world bootstrapping holds target ICA and its
reporter assignments fixed. V4.62 therefore implements an
[ICASSO-inspired repeated-fit experiment](https://doi.org/10.1016/j.neuroimage.2004.03.027).
It compares the cached representation, changed ICA initialization, resampled
cells, and both changes. Every mode uses identical evaluation rows within an
assay/replicate, including a cached-reference control for subset noise.
Normalization and ICA use only in-bag backgrounds for resampling modes;
reporter assignments and joint sliced-W2 synchronization are recomputed on
the evaluation subset. Feature contexts share same-assay rows, while assays
are resampled independently.

The first random ten-world panel with eight replicates selected 17, 19, 11,
and 14 true assay pairs, respectively, under the same majority/other-block
null diagnostic rule, with zero false proposals. All of these were already
covered by the v4.59 checkpoint: this is evidence about representation
uncertainty, not a recovery gain. Independent review reaggregated all 14,400
replicate/context pair rows and verified sources, inputs, outcomes, and failure
denominators. The experiment retains its sensitivity-only interpretation:
cached references and reporter feature identities already used the complete
data, and eight replicates do not establish precise stability probabilities.
The measured comparison and further replication are recorded in the
[experiment log](channel_aware_boss_v4_experiment_log.md).

The subsequent soft reporter model (v4.63) selected 21 true and one false
diagnostic pair on ten cached worlds, with two new truths and one new false
pair relative to complete v4.59 outputs. Both new truths also occur with
cached representations. V4.64 then replaces marginal correlation attribution
with joint-regression unique-effect energy, keeping the exact ICA fits,
evaluation rows, global alignment and selection rule fixed. It selects the
same primary pairs: 21 true and one false. Its marginal control exactly
reproduces v4.63. All successful joint designs are identified and have
condition number at most 1.353, so severe predictor collinearity does not
explain the current error. Independent review verified 33 source hashes and
the input/output/truth hashes; the paired regression suite passes 25 tests.

Neither candidate is promoted. Per the user's research direction, the next
prototype changes the background representation objective to a conditional
count likelihood while retaining the independent-component coordinate
criterion. Its proposed advantage is reducing measurement distortion in
source-distribution matching, not adding another post hoc rescue gate.
Count-likelihood improvement and cross-assay recovery both remain unproven.
No new worlds, paired cells or additional biological information are used.

### Earlier production layers (v4.22–v4.32)

CA-BOSS v4.32 remains the production assay-local layered implementation. It preserves the
v4.22 within-assay direct selector, v4.23 state-equivalence tier, v4.25
empirical-null tier, and v4.26 familywise multievidence tier. V4.27 adds a
sentinel-certified topology-consensus route only for a still-unresolved,
formally eligible feature that has both positive cross-fitted marginal and
partial state fit and is an endpoint of a separately typed within-assay direct
candidate. Direct topology alone is explicitly insufficient for membership.
V4.28 leaves that state set unchanged and guards assay-local loading groups
with an all-members-state route. V4.29 adds one bounded alternative for a
group with at least one v4.27 state anchor: it must join a selected cross-assay
axis under leave-one-group-out group evidence, and that axis must contain at
least two groups and two modalities. This certifies only the assay-local
loading group; it neither changes state membership nor asserts an exact
cross-assay feature pair. A proposal failing both routes remains typed
`unresolved`. V4.30 then requires the all-members-state route to retain a
minimum member-centroid cosine of 0.90 and downgrades any retained route that
overlaps a complete direct collider child or contains two parents of the same
child. This is an uncertainty downgrade, not evidence of biological absence;
it changes neither v4.27 feature membership nor the direct candidates.
V4.31 then adds a fair-comparison observability prerequisite for a retained
state-route group: its exact original structure-anchor record must contain the
selected two-parent collider, and the unselected competing parent must pass
the already frozen v4.27 formal/state/residual observability conjunction. The
independently estimated cross-assay-axis route bypasses this guard. A failure
is downgraded to `unresolved`; it is not interpreted as biological absence.
No numeric threshold is fitted, and neither state membership nor the direct
graph changes.

V4.32 leaves every v4.31 selection unchanged but removes an unjustified point
claim from the state route. Each retained state-route result is now one typed
assay-local structure-equivalence hypothesis over the selected or competing
parent and the noncollider sibling or collider child. Those four pairs are
joint alternatives and are never emitted as four edges. The independently
supported cross-assay-axis route remains one exact assay-local group. This
representation separates recoverable shared-state structure from exact pair
or direction claims that disjoint cells cannot support.

The nested v4.27 state audit covers all 152 eligible cached worlds. The v4.29,
v4.30, v4.31, and v4.32 loading audits cover the 151 worlds with complete cached
group-axis and structure-anchor artifacts, and the v4.32 source-current production replays cover 60 evaluations
of 44 unique cached worlds. No world, dataset, modality, paired observation,
intervention, external variable, or missing artifact was created. Both axis
tables are bound through the producer, scorer, cached replay, formal runner
registration, global pretruth barrier, and per-world candidate manifest. The
v4.30 structure/uncertainty rule and v4.31 exact competing-parent rule are also
source- and manifest-bound; the v4.32 fixed structure-equivalence rule and
no-expansion semantics are bound through the same executable closure. This
remains retrospective development evidence: the fixed rule and upstream
selectors were informed by the cached pool, so none of these audits is a fresh locked
claim.

| Layer | Complete-cache result | Interpretation |
| --- | ---: | --- |
| Within-assay direct candidates | 1,824 TP, 0 FP, 0 FN, 0 reversal | 152/152 worlds exact |
| Direct pairwise density | 12.0/world | Down from 22.725 biological edges/world in v3; the shared-state object is no longer expanded into marginal pairwise edges |
| Shared-state assay participation | 1,064/1,064 calls | All six chromatin assays plus RNA in 152/152 worlds |
| Strict shared-state feature membership | 2,990 TP, 0 FP, 810 FN | Precision 1.000, recall 0.7868; deliberately incomplete |
| Strict plus state-equivalence support | 3,246 TP, 0 FP, 554 FN | Precision 1.000, recall 0.8542; no pairwise expansion |
| Production-path v4.24 random replay | 60/60 reliable instances | 52 unique cached worlds; direct precision/recall 1.000; state-supported precision 1.000 and mean recall 0.866 |
| Nested v4.25 empirical-null support | 3,401 TP, 0 FP, 399 FN | Seven of seven held source blocks had zero false additions; precision 1.000, recall 0.8950 |
| Nested-audit v4.25 random panels | 60 world evaluations | Five block-balanced panels, zero false additions, mean recall 0.8953 (range 0.8767–0.9300) |
| Production-wired v4.25 random replay | 60/60 reliable evaluations | 49 unique cached worlds; mean recall 0.8973 versus 0.8620 before calibration; precision 1.000; zero sentinel, pairwise, or direct additions |
| Nested v4.26 multievidence support | 3,490 TP, 0 FP, 310 FN | Precision 1.000, recall 0.9184; 89 incremental true members beyond v4.25 |
| Production-wired v4.26 random replay | 57/60 full-layer reliable evaluations | 48 unique worlds; precision 1.000, mean recall 0.9107; the same cached world `41019` appeared three times and had v4.26 state recall 0.76 |
| Nested v4.27 topology consensus | 3,637 TP, 0 FP, 163 FN | 147/147 true additions; precision 1.000, recall 0.9571, minimum world recall 0.84; 152/152 worlds meet state precision 1 and recall ≥0.80 |
| Production-wired v4.27 random replay | 60/60 state-layer reliable; 59/60 whole-layer reliable | 1,429/1,500 state members, precision 1.000, mean recall 0.9527, minimum 0.84; 65/65 true additions and zero sentinel, pairwise, or direct additions. The sole whole-layer failure is the pre-existing loading-group result in world `41019` |
| Production-wired v4.28 loading guard | 60/60 whole-layer reliable | State results unchanged; 329 proposed loading groups became 293 retained and 36 typed abstentions. Mean loading-pair precision 0.9972, mean recall 0.8111, minimum world precision 0.8333, and zero loading sentinels |
| Nested v4.29 axis-consensus loading audit | 756 TP, 10 FP, 150 FN across 151 cached worlds | Adds 20 TP, 0 FP, and 0 sentinels beyond v4.28; precision 0.9869 and recall 0.8344. All seven source blocks have zero false axis-route additions |
| Production-wired v4.29 random replay | 60/60 whole-layer reliable | Same panel seeds as v4.28: 293 state-route plus 7 axis-route groups; mean loading precision 0.9972 and recall 0.8306; zero loading sentinels |
| Nested v4.30 structure/uncertainty audit | 756 TP, 5 FP, 150 FN across 151 cached worlds | Removes five inherited false groups and zero true groups; precision 0.9934, recall 0.8344, zero sentinels, and 151/151 worlds pass the loading reliability gate |
| Production-wired v4.30 random replay | 60/60 whole-layer reliable | 44 unique cached worlds; exact-panel v4.29 baseline 302 TP/5 FP becomes 302 TP/2 FP. Micro precision 0.9934, recall 0.8389, direct precision/recall 1.000, zero sentinels |
| Nested v4.31 competing-parent audit | 754 TP, 3 FP, 152 FN across 151 cached worlds | Exact original structure competition removes two false and two true groups from v4.30; precision 0.9960, recall 0.8322, zero sentinels, and 151/151 worlds pass the loading reliability gate |
| Production-wired v4.31 paired replay | 60/60 whole-layer reliable | Same 44 unique worlds and panel rows as v4.30; 302 TP/2 FP becomes 302 TP/0 FP at unchanged 58 FN. Micro precision 1.000, recall 0.8389, direct precision/recall 1.000, zero sentinels |
| Nested v4.32 set-valued audit | 757/757 hypotheses cover distinct truth groups across 151 cached worlds | The unchanged 757 v4.31 selections become 737 structure-equivalence hyperedges and 20 exact axis groups; coverage precision 1.000, recall 0.8355, 149 uncovered truth groups, and zero sentinels. The 2,968 alternatives are not edges and were not expanded |
| Production-wired v4.32 paired replay | 60/60 whole-layer reliable | Exact same seeds, rows, and 44 unique worlds as v4.31; all 302 hypotheses cover distinct truth groups at precision 1.000 and recall 0.8389. They comprise 296 structure-equivalence hyperedges and six exact axis groups; 1,190 alternatives remain unexpanded |
| V4.33 assay-local reporter baseline | 888 TP, 0 FP of 912 truth groups across 152 cached worlds | Precision 1.000, recall 0.9737; establishes that the remaining difficulty is cross-assay registration rather than within-assay reporter recovery |
| V4.41 component-bootstrap audit | 112 TP, 0 FP of 309 truth hyperedges across 103 cached worlds | Full plus two feature-fold reporter assignments require Bonferroni-calibrated within-assay cell-bootstrap support; precision 1.000, recall 0.3625 |
| V4.42 alternative-set component audit | 135 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Precision 1.000, recall 0.4369; 23 additional one-context-disagreement hypotheses are retained only as explicit ambiguity sets |
| V4.43 extended ambiguity-pool audit | 141 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Precision 1.000, recall 0.4563; locally unresolved assays can contribute only all-context-bootstrap-supported nonpoint alternatives |
| V4.44 global-bijection completion audit | 154 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Precision 1.000, recall 0.4984; 13 additional exact participation hypotheses pass strict-majority, all-context bootstrap, unique global assignment, and all-assigned-pair KS-fit guards |
| V4.45 exact multi-assay synchronization audit | 156 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Precision 1.000, recall 0.5049; a context-wide permutation-consistency solution adds two hypotheses after v4.44 abstains |
| V4.46 held-out synchronized-majority audit | 167 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Precision 1.000, recall 0.5405; ordinary 2/3 support must be accompanied by reporter coassignment in at least one held-out global synchronization |
| V4.47 cross-fit empirical-null synchronization audit | 185 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Hyperedge precision 1.000, recall 0.5987; the edge-specific threshold is the maximum eligible null margin in other seed blocks, with one admitted null alternative contained inside an unexpanded ambiguity hyperedge |
| V4.48 nested multivariate cross-fit audit | 190 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Exploratory full-fit result, not carried into the stable checkpoint: all five additions had whole-world bootstrap frequencies below 0.80 |
| V4.49 sparse-reporter proposal audit | 193 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Three proposals pass dual nested null envelopes and unique-reporter/partner guards; they remain proposals until world-bootstrap calibration |
| V4.50 interaction-rescue audit | 196 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Exploratory and not promoted: its three additions had bootstrap frequencies 0.12, 0.02, and 0.00 |
| V4.51 whole-world stability audit | 187 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Historical stable base: v4.47 plus two v4.49 proposals stable in 1.00 and 0.95 of 100 refits; the 0.46 H3K4me3 proposal is rejected |
| V4.52 empirical-copula stability audit | 190 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Three parameter-free directional-rank proposals survive at 0.99, 1.00, and 0.98; no null reaches the 0.80 gate |
| V4.53 mutual-nearest-margin stability audit | 196 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Historical stable base: six endpoint-reciprocal additions over v4.52 survive at 0.82–1.00; the only nominal null reaches 0.48 and is rejected |
| V4.54 W2 distribution-geometry stability audit | 209 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Historical checkpoint: 13 globally synchronized W2 additions survive deterministic 12-decimal rank quantization and the 0.80 bootstrap guard; seven additional nominal truths remain abstentions |
| V4.55 empirical-null-calibrated W2 stability audit | 213 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Historical checkpoint: four additional W2 proposals beat their target-separated worst-null counts and working-Beta guard |
| V4.56 joint sliced-geometry diagnostic | Not promoted | Raw and copula routes each admit false pairs on one of two random 40-world panels; intersection confirmation adds nothing in the remaining 23 worlds |
| V4.57 global joint-geometry null-stability diagnostic | 0 additions | Not promoted: a global worst null recurs in 96--100 of 100 refits in every target block |
| V4.58 local-context-stratified stability audit | 238 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Historical checkpoint: 25 additional joint-geometry proposals pass three-of-three local-context agreement and same-stratum empirical-null stability; the stratum was chosen during cached-world development |
| V4.59 dual-null disagreement child | 239 TP, 0 FP of 309 truth hyperedges across the same 103 worlds | Current retrospective checkpoint: one two-context proposal exceeds both null ceilings; independently reviewed and scoring-replayed |
| V4.60 observed-depth geometry | Not promoted | Random 25-world diagnostic produces only false additions |
| V4.61 ECF/joint geometry | Not promoted | Five stable true and one false addition over two disjoint panels; proposed context guard remains post hoc |
| V4.62 representation-refit diagnostic | Not promoted | Matched-subset controls over 18 unique worlds distinguish initialization effects from cell/refit variation; one new candidate needs full benchmark calibration |
| V4.63 soft reporter evidence | Not promoted | Primary ten-world diagnostic selects 21 TP/1 FP; two true and one false pair are new beyond v4.59. The candidate universe contains 29 of the panel's 30 truths |
| V4.64 joint unique-effect regression | Not promoted; no primary gain | Same ten worlds and 16 replicates as v4.63; primary selection remains exactly 21 TP/1 FP with shared representations, evaluation rows, and alignment. Independent paired checks and 25 targeted tests are recorded in the experiment log |
| Sentinel pseudo-assay nulls | 6/720 false calls | 5/120 worlds, familywise rate 0.0417 versus the nominal 0.10 target |

The direct layer is 47.2% smaller than the mean v3 biological-edge count. That
comparison is intentionally conservative: v4 represents cross-assay shared
state as one hypothesis-only n-ary set rather than converting it to pairwise
edges.

## Roadmap completion audit

| Roadmap item | Status | Authoritative evidence |
| --- | --- | --- |
| Record CA-BOSS v3 | Complete | Frozen v3 GAPCA result: 37/40 successful worlds, 22.725 mean biological edges, matched-skeleton F1 0.544; H3K4me3 direct recovered in 38/40 |
| Explicit latent-state model | Implemented | State coordinates use held-out background features only; focal feature models are cross-fitted within assay; shared state and residual direct structure are emitted separately |
| Hierarchical sparse-count replacement | Implemented | State/depth-conditioned cross-fitted Bernoulli evidence, hierarchical beta-binomial pooling, posterior/uncertainty gates, and empirical-null collider evidence replace the loose Fisher `p <= 0.25` rescue |
| Empirical-null stability calibration | Implemented through cross-fit bridge v4.59 | Existing production null controls remain active. Three-context proposals use same-stratum null stability; v4.59 two-context proposals must additionally exceed the global null ceiling. Both retain the working-Beta guard. The result remains retrospective |
| Explicit multi-layer graph | Implemented through cross-fit bridge v4.59 | Direct candidates, shared-state and loading equivalence, ambiguity hyperedges, prior-constrained peak–gene, and unresolved outputs remain separate; alternatives cannot be expanded as edges or directions |
| Integration robustness | Mostly implemented | Existing production views retain cross-fitted focal models, multiple aggregation resolutions/seeds, and leave-one-group-out axes. The component bridge adds full plus two disjoint held-out background-feature-fold alignments and cell-bootstrap stability in every context. No exact cell or feature registration is assumed. Probabilistic cell-to-metacell membership remains unimplemented |
| Cross-assay benchmark metrics | Implemented through v4.59 | Hash-verified world/panel/campaign scores report all four requested metric families. V4.59 verifies the full parent chain and freezes all target predictions before scoring; independent review and replay reproduce the totals |
| Randomized simulation factors | Covered by existing cache | All 152 worlds have unique focal and null layouts and unique state-loading layouts; coefficients and detection rates vary; the balanced 8-cell fractional factorial contains 19 worlds per cell |
| Six chromatin modalities | Complete in existing cache | Every world contains H3K27ac, H3K27me3, H3K4me1, H3K4me2, H3K4me3, H3K9me3, plus RNA |
| Lower detection, population and sample-size stress | Complete in existing cache | Detection probability 0.0270–0.13; assay cells 780–3,600; population fractions 0.45–0.75; 76 worlds each activate detection, population-divergence, and background-observability stress |
| Shuffled guidance | Not needed by current candidate | The deployed shared-state layer uses assay-local held-out background embeddings and no cross-assay guidance graph. The earlier GAPCA/scGLUE guidance diagnostic is not part of v4.22 |
| Fresh locked block after changes | Not opened | Explicitly outside the current existing-world-only instruction. Full-cache success is not a fresh generalization claim |

## What is reliable, and what is not

The existing cache supports reliable recovery of these separately typed,
estimable objects:

1. within-assay direct collider arms among variables measured in the same
   cells; and
2. an assay-level, shared-latent-state participation set across disjoint
   modalities; and
3. within-assay equivalence sets for strong state reporters that are not
   uniquely distinguishable after redundancy handling; and
4. empirical-null and familywise-multievidence state-member tiers whose
   parameters were held outside the predicted source block; and
5. a v4.27 topology-consensus state tier requiring agreement between positive
   shared-state evidence and a separately typed residual-structure endpoint;
   and
6. v4.29 assay-local loading support certified either by the v4.28
   all-members-state route or by leave-one-group-out cross-assay axis
   consensus with at least one v4.27 state anchor; and
7. v4.30 precision-first loading support after a frozen coherence floor and an
   explicit unresolved label for groups entangled with complete direct
   collider structure; and
8. v4.31 loading support after an exact, frozen comparison between the
   selected and competing collider parents, with an unresolved label whenever
   the state route did not observe both sides fairly; and
9. v4.32 assay-local loading structure as either one exact axis-route group or
   one unexpanded two-by-two state-route equivalence hypothesis; and
10. v4.59 cross-context component hypotheses under conservative v4.44–v4.55
    routes or globally synchronized joint distribution-geometry proposals that
    win reciprocal endpoint competition in the full fit and both disjoint
    feature folds, then beat a same-stratum empirical null under whole-world
    bootstrap refits, with conflicts represented as one explicit alternative
    set. The two-context child additionally requires both stratum and global
    null stability ceilings.

The strict feature annotation is precision-first and incomplete. It must not
be expanded into exact peak-to-peak or peak-to-gene matches across assays. The
v4.24 equivalence tier has recall 0.8542, v4.25 raises this to 0.8950, and
v4.26 raises it to 0.9184. The nested v4.27 conjunction adds 147 true reporters
and zero null reporters across seven held source blocks, producing 3,637 TP,
0 FP, and 163 FN (recall 0.9571). Every cached world has precision 1.000 and
recall at least 0.84. Five production-path panels independently exercise the
final wiring: all 60 state layers pass, with mean recall 0.9527 and zero false
members. V4.28 converts the older `41019` loading-group sentinel failure into
an explicit abstention. V4.29 then restores only groups supported by an
independently estimated leave-one-group-out cross-assay axis. Across the same
five production panels, 60/60 complete layer checks pass. Of 329 proposals,
293 use the all-members-state route, seven use the axis route, and 29 abstain.
Mean loading-pair precision remains 0.9972 while recall rises from 0.8111 to
0.8306, with no loading sentinels. Across the 151-world audit the axis route
adds 20 true and zero false groups; ten false groups inherited from the
upstream local proposals remain. V4.30 removes five of those inherited false
groups without losing a cached true group: two are downgraded for direct-layer
overlap and three for sub-threshold loading coherence. Its aggregate precision
is 0.9934 at unchanged 0.8344 recall. The five source-current production panels
likewise improve their exact-panel v4.29 baseline from 302 TP/5 FP to 302 TP/2
FP. This is a precision-first group-level improvement rather than a claim of
complete assay-local pairing. V4.31 removes two false and two true groups from
the full cache, giving 754 TP/3 FP/152 FN (precision 0.9960, recall 0.8322).
On the exact same five production panels, however, both v4.30 false groups are
downgraded and no true group is lost: 302 TP/0 FP/58 FN, precision 1.000 and
recall 0.8389. The three full-cache false groups that remain cannot be
separated from true groups by the role-blind cached scores without a worse
true-group tradeoff; they remain explicit shared-state/parent-equivalence
ambiguities rather than being thresholded away post hoc.

V4.32 does not attempt another cutoff. It preserves all 757 v4.31 selections
and instead scores the structural uncertainty those observations actually
support. Across the full cache, 737 state-route equivalence hyperedges and 20
exact axis groups cover 757 distinct truth groups, for coverage precision
1.000 and recall 0.8355 with 149 uncovered truth groups and zero sentinels.
The nominal v4.31 point-pair diagnostic remains 754 TP/3 FP/152 FN, making it
explicit that the improvement comes from replacing three over-specific point
claims with calibrated alternatives, not from adding evidence. The same five
production panels contain 296 equivalence hyperedges and six exact groups;
all 302 hypotheses cover distinct truth groups and all 60 complete checks pass.
Neither the 2,968 full-cache alternatives nor the 1,190 panel alternatives are
expanded into edges.

For the cross-assay bridge, v4.59 raises the stable 103-world shared-state
coverage from v4.44's 154/309, v4.53's 196/309, v4.54's 209/309, and v4.55's
213/309 and v4.58's 238/309 to 239/309 while retaining
zero false hyperedges and zero sentinel features. The gain comes from matching
the joint geometry of assay-local component distributions, globally
synchronizing those component permutations, requiring local mutual-nearest
agreement in all three feature contexts, calibrating within that context
stratum across seed blocks, and retaining additions only after whole-world
empirical-null-calibrated stability selection. It does not come from a perfect-matching assumption, new
modality, paired cell, cell-level transport coupling, simulator state, or
external prior. The remaining 70 truth hyperedges are abstentions. Cached
precision and calibration-world resampling results do not establish
general reliability: v4.62 demonstrates sensitivity when the latent
representation itself is refitted. Cross-assay recovery remains incomplete.

This is evidence for recovery of a cross-assay shared-latent-state
participation structure, represented as typed sets/hyperedges. It is not
evidence for exact peak-to-peak or peak-to-gene correspondence. Cell-specific
residual effects transmitted between disjoint modalities remain unidentified
from the agreed variables, so v4.59 does not label them as direct cross-assay
causal effects.

## Reproducible artifacts

- Frozen v3 baseline: [`channel_aware_boss_validation_summary.json`](channel_aware_boss_validation_summary.json)
- Prior 142-world checkpoint: [`ca_boss_v4_cached_random_v422_summary.json`](ca_boss_v4_cached_random_v422_summary.json)
- Complete-cache checkpoint: [`ca_boss_v4_cached_full_pool_v422_summary.json`](ca_boss_v4_cached_full_pool_v422_summary.json)
- Historical v4.23 checkpoint: [`ca_boss_v4_cached_full_pool_v423_summary.json`](ca_boss_v4_cached_full_pool_v423_summary.json)
- Production-wired v4.24 random replay: [`ca_boss_v4_cached_random_v424_summary.json`](ca_boss_v4_cached_random_v424_summary.json)
- Nested v4.25 empirical-null checkpoint: [`ca_boss_v4_cached_nested_empirical_null_v425_summary.json`](ca_boss_v4_cached_nested_empirical_null_v425_summary.json)
- Production-wired v4.25 random replay: [`ca_boss_v4_cached_random_v425_summary.json`](ca_boss_v4_cached_random_v425_summary.json)
- Nested v4.27 and production replay checkpoint: [`ca_boss_v4_cached_random_v427_summary.json`](ca_boss_v4_cached_random_v427_summary.json)
- Production-wired v4.28 loading-guard replay: [`ca_boss_v4_cached_random_v428_summary.json`](ca_boss_v4_cached_random_v428_summary.json)
- V4.29 axis-consensus checkpoint: [`ca_boss_v4_cached_random_v429_summary.json`](ca_boss_v4_cached_random_v429_summary.json)
- V4.30 structure/uncertainty checkpoint: [`ca_boss_v4_cached_random_v430_summary.json`](ca_boss_v4_cached_random_v430_summary.json)
- V4.31 competing-parent checkpoint: [`ca_boss_v4_cached_random_v431_summary.json`](ca_boss_v4_cached_random_v431_summary.json)
- Current dual-null disagreement checkpoint: [`ca_boss_v4_bootstrap_dual_null_disagreement_rescue_v459_summary.json`](ca_boss_v4_bootstrap_dual_null_disagreement_rescue_v459_summary.json)
- Historical joint context-stratified checkpoint: [`ca_boss_v4_bootstrap_joint_context_stratified_stability_rescue_v458_summary.json`](ca_boss_v4_bootstrap_joint_context_stratified_stability_rescue_v458_summary.json)
- Historical empirical-null-calibrated W2 checkpoint: [`ca_boss_v4_bootstrap_null_calibrated_stability_rescue_v455_summary.json`](ca_boss_v4_bootstrap_null_calibrated_stability_rescue_v455_summary.json)
- V4.54 bootstrap-stable W2 checkpoint: [`ca_boss_v4_bootstrap_wasserstein_geometry_rescue_v454_summary.json`](ca_boss_v4_bootstrap_wasserstein_geometry_rescue_v454_summary.json)
- V4.53 mutual-nearest-margin checkpoint: [`ca_boss_v4_bootstrap_mutual_nearest_margin_rescue_v453_summary.json`](ca_boss_v4_bootstrap_mutual_nearest_margin_rescue_v453_summary.json)
- V4.52 empirical-copula checkpoint: [`ca_boss_v4_bootstrap_empirical_copula_pair_rescue_v452_summary.json`](ca_boss_v4_bootstrap_empirical_copula_pair_rescue_v452_summary.json)
- V4.51 sparse-completion stability checkpoint: [`ca_boss_v4_bootstrap_stable_sparse_completion_v451_summary.json`](ca_boss_v4_bootstrap_stable_sparse_completion_v451_summary.json)
- Non-promoted v4.50 exploratory record: [`ca_boss_v4_interaction_reporter_rescue_v450_summary.json`](ca_boss_v4_interaction_reporter_rescue_v450_summary.json)
- V4.49 sparse-reporter proposal record: [`ca_boss_v4_sparse_reporter_completion_v449_summary.json`](ca_boss_v4_sparse_reporter_completion_v449_summary.json)
- Historical v4.48 cross-assay checkpoint: [`ca_boss_v4_crossfit_multivariate_v448_summary.json`](ca_boss_v4_crossfit_multivariate_v448_summary.json)
- Historical v4.47 cross-assay checkpoint: [`ca_boss_v4_crossfit_empirical_null_v447_summary.json`](ca_boss_v4_crossfit_empirical_null_v447_summary.json)
- Historical v4.44 cross-assay checkpoint: [`ca_boss_v4_component_conditioned_v444_summary.json`](ca_boss_v4_component_conditioned_v444_summary.json)
- Historical v4.42 cross-assay checkpoint: [`ca_boss_v4_component_conditioned_v442_summary.json`](ca_boss_v4_component_conditioned_v442_summary.json)
- V4.47 relaxed reporter evidence: [`../scripts/relaxed_reporter_synchronization_evidence_v4.py`](../scripts/relaxed_reporter_synchronization_evidence_v4.py)
- V4.47 cross-fit empirical-null calibrator: [`../scripts/crossfit_empirical_null_reporter_synchronization_v4.py`](../scripts/crossfit_empirical_null_reporter_synchronization_v4.py)
- V4.47 truth-separated scorer: [`../scripts/score_crossfit_empirical_null_reporter_synchronization_v4.py`](../scripts/score_crossfit_empirical_null_reporter_synchronization_v4.py)
- V4.48 nested multivariate calibrator: [`../scripts/crossfit_multivariate_reporter_rescue_v4.py`](../scripts/crossfit_multivariate_reporter_rescue_v4.py)
- V4.48 truth-separated scorer: [`../scripts/score_crossfit_multivariate_reporter_rescue_v4.py`](../scripts/score_crossfit_multivariate_reporter_rescue_v4.py)
- V4.49 sparse-reporter evidence: [`../scripts/sparse_reporter_completion_evidence_v4.py`](../scripts/sparse_reporter_completion_evidence_v4.py)
- V4.49 sparse-reporter calibrator/scorer: [`../scripts/crossfit_sparse_reporter_completion_v4.py`](../scripts/crossfit_sparse_reporter_completion_v4.py), [`../scripts/score_crossfit_sparse_reporter_completion_v4.py`](../scripts/score_crossfit_sparse_reporter_completion_v4.py)
- V4.50 exploratory interaction calibrator/scorer: [`../scripts/crossfit_interaction_reporter_rescue_v4.py`](../scripts/crossfit_interaction_reporter_rescue_v4.py), [`../scripts/score_crossfit_interaction_reporter_rescue_v4.py`](../scripts/score_crossfit_interaction_reporter_rescue_v4.py)
- V4.51 whole-world stability calibrator/scorer: [`../scripts/bootstrap_stable_sparse_completion_v4.py`](../scripts/bootstrap_stable_sparse_completion_v4.py), [`../scripts/score_bootstrap_stable_sparse_completion_v4.py`](../scripts/score_bootstrap_stable_sparse_completion_v4.py)
- V4.52 empirical-copula calibrator/scorer: [`../scripts/bootstrap_empirical_copula_pair_rescue_v4.py`](../scripts/bootstrap_empirical_copula_pair_rescue_v4.py), [`../scripts/score_bootstrap_empirical_copula_pair_rescue_v4.py`](../scripts/score_bootstrap_empirical_copula_pair_rescue_v4.py)
- V4.53 mutual-nearest-margin calibrator/scorer: [`../scripts/bootstrap_mutual_nearest_margin_rescue_v4.py`](../scripts/bootstrap_mutual_nearest_margin_rescue_v4.py), [`../scripts/score_bootstrap_mutual_nearest_margin_rescue_v4.py`](../scripts/score_bootstrap_mutual_nearest_margin_rescue_v4.py)
- V4.54 W1/W2 diagnostic and truth-blind evidence freezer: [`../scripts/diagnose_quantile_geometry_rescue_v4.py`](../scripts/diagnose_quantile_geometry_rescue_v4.py), [`../scripts/quantile_geometry_synchronization_evidence_v4.py`](../scripts/quantile_geometry_synchronization_evidence_v4.py)
- V4.54 W2 calibrator/scorer: [`../scripts/bootstrap_wasserstein_geometry_rescue_v4.py`](../scripts/bootstrap_wasserstein_geometry_rescue_v4.py), [`../scripts/score_bootstrap_wasserstein_geometry_rescue_v4.py`](../scripts/score_bootstrap_wasserstein_geometry_rescue_v4.py)
- V4.32 assay-local structure-equivalence checkpoint: [`ca_boss_v4_cached_random_v432_summary.json`](ca_boss_v4_cached_random_v432_summary.json)
- Two-phase audit utility: [`../scripts/audit_cached_layered_contract_v4.py`](../scripts/audit_cached_layered_contract_v4.py)
- Fast production-layer replay utility: [`../scripts/replay_cached_layered_state_v4.py`](../scripts/replay_cached_layered_state_v4.py)
- State-equivalence module: [`../scripts/state_equivalence_membership_v4.py`](../scripts/state_equivalence_membership_v4.py)
- Empirical-null calibration module: [`../scripts/empirical_null_state_calibration_v4.py`](../scripts/empirical_null_state_calibration_v4.py)
- Nested calibration audit: [`../scripts/audit_nested_empirical_null_state_calibration_v4.py`](../scripts/audit_nested_empirical_null_state_calibration_v4.py)
- Familywise multievidence module: [`../scripts/multievidence_empirical_null_state_calibration_v4.py`](../scripts/multievidence_empirical_null_state_calibration_v4.py)
- Topology-consensus module: [`../scripts/topology_consensus_state_calibration_v4.py`](../scripts/topology_consensus_state_calibration_v4.py)
- Loading-group state guard: [`../scripts/loading_group_state_guard_v4.py`](../scripts/loading_group_state_guard_v4.py)
- Loading-group structure/uncertainty guard: [`../scripts/loading_group_structure_uncertainty_guard_v4.py`](../scripts/loading_group_structure_uncertainty_guard_v4.py)
- Loading-group competing-parent guard: [`../scripts/loading_group_competing_parent_observability_guard_v4.py`](../scripts/loading_group_competing_parent_observability_guard_v4.py)
- Loading-group structure-equivalence representation: [`../scripts/loading_group_structure_equivalence_v4.py`](../scripts/loading_group_structure_equivalence_v4.py)
- Nested topology audit: [`../scripts/audit_nested_topology_consensus_state_v4.py`](../scripts/audit_nested_topology_consensus_state_v4.py)
- Nested loading-axis audit: [`../scripts/audit_nested_loading_group_axis_consensus_v4.py`](../scripts/audit_nested_loading_group_axis_consensus_v4.py)
- Nested loading structure/uncertainty audit: [`../scripts/audit_nested_loading_group_structure_uncertainty_v4.py`](../scripts/audit_nested_loading_group_structure_uncertainty_v4.py)
- Nested competing-parent audit: [`../scripts/audit_nested_loading_group_competing_parent_observability_v4.py`](../scripts/audit_nested_loading_group_competing_parent_observability_v4.py)
- Nested structure-equivalence audit: [`../scripts/audit_nested_loading_group_structure_equivalence_v4.py`](../scripts/audit_nested_loading_group_structure_equivalence_v4.py)
- Historical soft-alignment diagnostic: [`soft_distributional_alignment_train_seed30001.md`](soft_distributional_alignment_train_seed30001.md)
- Historical loading-group diagnostic: [`structure_anchored_loading_groups_old_train.md`](structure_anchored_loading_groups_old_train.md)
- Full derived audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_layered_contract_v423_full_pool/phase_b_summary.json`
- Full nested v4.25 audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_empirical_null_v425_full_pool/phase_b_summary.json`
- Full nested v4.27 audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_topology_consensus_v427_full_pool/phase_b_summary.json`
- Full nested v4.29 loading-axis audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_loading_axis_consensus_v429_full_pool_final/phase_b_summary.json`
- Full nested v4.30 loading structure/uncertainty audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_loading_structure_uncertainty_v430_full_pool/phase_b_summary.json`
- Full nested v4.31 competing-parent audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_loading_parent_observability_v431_full_pool_final/phase_b_summary.json`
- Full nested v4.32 structure-equivalence audit: `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_loading_structure_equivalence_v432_full_pool_final/phase_b_summary.json`

The v4.23 full-cache audit first loaded and hash-closed all observable predictions and
membership decisions, then opened existing scorer files. Its Phase-A manifest
SHA-256 is
`e3442601aa5c9b93c5cd80fd54f0bda7ab6165af5ed0a8426258f3987edea148`;
its Phase-B summary SHA-256 is
`48b6759995c9f6289711fd3c9f4f76e28969fb829ded2f8f2f6845afd0749119`.

The nested v4.25 audit froze the observable table and five random panels before
opening labels. Its Phase-A manifest SHA-256 is
`81ce1f1fe184f8a93009a0276f7d0c3da8b6bed74854c42d87ae1570609bfeaa`;
its Phase-B summary SHA-256 is
`abb342ebe3c1f798711376c2335fcfd4eb49c50425aa81bb342600bb24582e1c`
with payload SHA-256
`6c5d1a0d928c283d27b08e8d9d86e9d2719ddb67249b494cc4c953968d305487`.
The future-application calibrator SHA-256 is
`adc8de2aae1b698c4d46899b7f5ca8137fd881bbd829e54b405aa064bc40cd87`;
its training rows are explicitly excluded from a validation claim.

The nested v4.27 Phase-A manifest SHA-256 is
`8e1be654894693f6834a913235ecc7dc6b75dc8075964a5354b4f7bfc729f1dc`;
the Phase-B summary file SHA-256 is
`e083cfefe9a851e41934db7bca7719c835ad4eef55aa08859ed4dbdb8c5e8276`;
its payload SHA-256 is
`98d592bf69079e79c3b81953eb2fef327b45169063bf2548fd110d2589a4eb36`;
and the future-application calibrator SHA-256 is
`ad3e0ef8ffaef95ff9dbde75716a5ade187d6cedb860c5f2f634ab433be86efb`.
Its training rows are explicitly excluded from a fresh-validation claim.

The final v4.29 loading-axis Phase-A manifest SHA-256 is
`708494d7615b668d2d2e2b8e539cc637499b731f026aa0163a60cb5d5482a42d`;
the Phase-B summary file SHA-256 is
`64f88ee1b24b37140fe5ab7cc4a339a4e5939e206b2b16e5d214881cf0b25c57`;
and its payload SHA-256 is
`c9e692a22ba70a2393ecaea111ec2340d989ccd54cac66fe92e5ee05caa00a6a`.
The audit creates no future-application numeric calibrator: v4.29 is a fixed
logical route over already frozen state, group, and axis selections.

The v4.30 structure/uncertainty Phase-A manifest SHA-256 is
`21880912a8a5da756cf80e4157fedf46a141d51843a644a3531063f1d1f6df10`;
the fixed-guard artifact SHA-256 is
`8ef500ce8659dc330861ce276359887cd958c77b099defd1f19219ec50021320`;
the Phase-B summary file SHA-256 is
`2b852d6d17cdc24831c41269b4ac58d3e7964828ee70f378331db79ade3cc47c`;
and its payload SHA-256 is
`041e8798eac308e2cbe1810b5c66f3b814def4451edd4278f256de162c275b13`.
The threshold was developed on this reused cache. The leave-one-source-block-
out result measures internal stability but does not convert it into a fresh
locked claim.

The valid v4.31 competing-parent Phase-A manifest SHA-256 is
`8795188fabd775823a1b06c1213711b63ddeba08341d76f75e12632ec55760cf`;
the fixed-guard artifact SHA-256 is
`4fdc0ef80d6e196a336a19b86855d1de2917366e7a55408b755a75ff4f2f3f62`;
the Phase-B summary file SHA-256 is
`00e7b595707e5d2e0f0cc46414f15963a3dbecb06f1e274473c16b46ed77aac7`;
and its payload SHA-256 is
`4555040ab703927d67c02f105229681befe3b53f978b10dc9f672a16078c1f96`.
The audit fitted no numeric threshold. It is nevertheless retrospective
because the rule was assessed on the reused cache; the leave-one-source-block
audit does not turn it into fresh locked validation. The similarly named
provisional directory without the `_final` suffix reconstructed competition
from a later repaired graph and is not a valid v4.31 artifact.

The valid v4.32 structure-equivalence Phase-A manifest SHA-256 is
`873a74f19f784e3879cfc414756ed860898917990f9e976cd963569ead6cb81c`;
the fixed-rule artifact SHA-256 is
`1f628ee77c90813f7b159404cc1e876a7185b8cfb9bd830838db326c9ef26c04`;
the Phase-B summary file SHA-256 is
`6afa6e78f3b4f96692c69ae2ede6bdf7e7b0ead1bb87282ea981afe92276b96d`;
and its payload SHA-256 is
`d23c7e4a7a289c63a77f5ebe4b8350c37f5b4052c319238ae6c8853e2f628302`.
The production checkpoint summary payload SHA-256 is
`de0471f2590ba8844cc4d9268916bb75c104b4d4b9930fe31415b58ccacdab7c`.
The fixed transformation is truth-free at application and fits no numeric
threshold, but its evaluation is still retrospective. Only the audit directory
ending in `_full_pool_final` is authoritative; the similarly named directory
without `_final` is provisional and must not be cited.
