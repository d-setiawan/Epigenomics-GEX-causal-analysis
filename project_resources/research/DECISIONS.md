# Why the pipeline looks like this

↑ [Central README](../../README.md) · [MISSION](MISSION.md) (what we're trying to do) ·
[ROADMAP](ROADMAP.md) (what's next) · [LIMITATIONS](LIMITATIONS.md) (what it can't do)

The numbered sections record choices made for the original empirical pipeline. Later experiments exposed two distinct questions. Which relationships do the marginal assay counts determine at the population level? Can an estimator recover them at the available sample size?

## Why we tested shared-state transport

A peak and a transcript from independent cell samples can covary across matched states even when their within-state, same-cell dependence is unknown. Running a graph learner on a chosen coupling gives an answer for those constructed rows. This motivated testing integration against known couplings and using ambiguous worlds as controls. [Aggregation argument](theory/06_latent_space_aggregation.md) · [Unpaired counterexamples](theory/07_unpaired_integration_nonidentifiability.md) · [Integration results](../../pipelines/simulation/integration_recoverability/INTEGRATION_VALIDATION.md)

The E/E2 model restricts the possible couplings. Distinct source distributions can align log intensities across assays, and the known count channel ties the observations to their latent rate laws. We tested a known-dimension binary-source subclass to distinguish failure to fit the likelihood from failure of population identification. On the same worlds, a count-informed search with several starts selected by unpaired calibration recovered more often than the unbounded fit initialized from the old starts. These choices changed the estimator’s ability to fit the model. The targets were transport, covariance and prediction. [Corollary](nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md) · [Exact-model comparison](../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_MATCHED.md) · [Source search and failures by start](../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md)

The November 22 transfer prioritized a clean CPU replay, checksum-verified inputs and portable launch commands. The distinction between a completed scientific run and the portion independently repeated at handoff is recorded in the [transfer verification](../../reproduce/VERIFICATION.md).

**Contents**
[1. Local per-gene graphs](#1-why-local-per-gene-graphs-and-not-one-big-network) ·
[2. scGLUE](#2-why-scglue-for-integration) ·
[3. Pseudo-pairing](#3-why-we-pair-cells-at-all) ·
[4. Metacells](#4-why-metacells-replaced-knn-averaging-the-noise-domination-story) ·
[5. Stability selection](#5-why-bootstrap-stability-selection-became-the-headline) ·
[6. FCI/PAG](#6-why-fcipag-and-not-pc-or-a-dag-learner) ·
[7. GFCI](#7-why-gfci-was-added-and-why-fci-still-carries-direction) ·
[8. Confounders + orientation](#8-why-library-depth-as-root-nodes-plus-a-mechanistic-prior) ·
[9. Donor & cell-cycle](#9-why-donor-and-cell-cycle-are-not-in-the-model) ·
[10. Three validation axes](#10-why-three-validation-axes-and-why-target-gene-aware) ·
[11. Dropping ePerturbDB](#11-why-eperturbdb-was-dropped) ·
[12. Distance-matched null](#12-why-the-elementelement-null-matches-on-distance) ·
[13. Co-accessibility contrast](#13-why-we-benchmark-against-co-accessibility) ·
[14. Leave-one-locus-out probe](#14-why-the-global-activity-probe-leaves-the-locus-out) ·
[15. Gene panel](#15-why-these-15-genes) ·
[16. Monocyte](#16-why-monocyte-and-not-t-cell)

---

## 1. Why local per-gene graphs, and not one big network

**The question.** Genome-wide regulatory-network inference is the obvious framing, so why do we learn
a separate small graph around each target gene?

**Why local.** Constraint-based causal discovery scales badly and needs statistical power per variable.
A genome-wide graph over tens of thousands of variables with ~10² samples is hopeless — you'd be
testing far more edges than the data can support, and the false-positive floor (≈ α × number of pairs)
would swamp everything. A per-gene graph over ~40 nearby peaks + confounders + expression is a problem
the data can actually speak to.

**What we give up.** No trans-regulation, no cross-locus structure. We say this explicitly — the claim
is structure in each gene’s local regulatory neighborhood ([LIMITATIONS L6](LIMITATIONS.md#l6-observational-pag-not-dag)).

**How.** A 300 kb window around each gene's TSS (interpretable range for enhancer–promoter contacts).
1 Mb was tried and produced graphs that were too dense to interpret.

---

## 2. Why scGLUE for integration

In the original PBMC pipeline, RNA and histone marks were measured in separate cells. scGLUE and the scMRDR-style implementation offered candidate common embeddings. An early comparison measured how often a cell's nearest neighbors belonged to its own assay.

| Run | same-modality neighbour fraction | silhouette by modality |
| --- | --- | --- |
| **scGLUE `joint_v2`** | **0.2548** | −0.0122 |
| scMRDR_gaussian | 0.5693 | −0.0984 |

The lower same-assay neighbor fraction prompted the historical choice of scGLUE for constructing rows. A later check using held-out proteins found both concordant and reversed biological contrasts between assay pairs. The saved scMRDR comparator also had limitations. Mixing is therefore a diagnostic of assay separation. Biological correspondence and causal recovery require their own tests. [Integration guide](../../pipelines/empirical/integration/README.md) · [Biological alignment baseline](../../pipelines/empirical/integration/results/biological_alignment_2026-09-16.md)

---

## 3. Why we pair cells at all

**The problem.** Causal discovery needs rows where chromatin and expression are observed *together*.
We don't have that — no cell has both.

**What we do.** Use the shared scGLUE latent space to match RNA profiles with chromatin profiles that
occupy the same region of that space, producing synthetic joint rows ("pseudo-pairs").

Matching supplies a matrix for the empirical analysis and imposes a coupling between assays. Where biological state is aligned, its means can covary across assays. The coupling of cell-specific deviations remains unobserved. Chromatin hashtags record technical groups, and the separate RNA experiment has no verified common donor map. [Sample provenance](../../pipelines/empirical/data/results/data_sources.md) · [Pseudo-pairing limits](LIMITATIONS.md#l2-pseudo-pairs-are-not-within-individual)

---

## 4. Why metacells replaced kNN-averaging (the noise-domination story)

*This is the decision worth understanding in full; it's the one that changed the most downstream.*

**The original approach.** Each RNA cell was kept as an anchor and its chromatin signal was the average
of its *k* = 10 nearest chromatin neighbours. Every RNA cell became one row → ~1,900 rows. Lots of data,
apparently.

**The problem we suspected.** Neighbour-averaging with overlap **reuses cells**. If cell A and cell B
are near each other, their 10 neighbours overlap heavily, so their rows are not independent. Every
statistical test in PC/FCI/DAGMA is an explicit function of *n* — feed it 1,900 rows that carry the
information of maybe 150, and every p-value is overconfident. Two failure modes at once: (a) inflated
*n* → type-I error blow-up → denser, falser graphs; (b) averaging shrinks within-feature variance →
kills the very signal the independence tests look for.

**How we confirmed it — the null-calibration test.** Suspicion isn't evidence, so we built a direct
measurement (`pipelines/empirical/evaluation/scripts/null_calibration.py`): **permute every feature column independently**
— destroying all real structure while preserving each feature's marginal distribution — then re-run the
identical FCI pipeline and count edges. Any edges found are by construction false positives. Compare to
the real run:

| gene | real edges | null edges |
| --- | --- | --- |
| CD14 | 56 | 51.7 |
| CSF1R | 10 | 9.2 |
| IL1B | 6 | 12.4 |

**The real graphs sat inside the noise band.** IL1B actually found *fewer* edges than noise. Edge counts
were tracking α × number-of-pairs, not biology. Everything built on those graphs was uninterpretable.

**Options considered.**
1. *Tighten α.* Helps the floor but doesn't fix dependent rows — you'd be correcting a symptom.
2. *SEACells* (the standard metacell tool). **Blocked:** installing it force-upgrades numpy 1.26→2.2,
   which breaks scGLUE and causal-learn in the same environment. Not worth fracturing the env.
3. *Non-overlapping metacells via native scanpy Leiden clustering* over the scGLUE joint latent.
4. *Do nothing and report honestly.* Rejected — the graphs would have no signal to report.

**What we chose: option 3.** Cluster cells in the joint latent into disjoint groups, aggregate each
group into one row. **Each cell is used exactly once**, so `n_rows ≈ n_effective` — the sample size is
honest. Result: 133 metacells for monocyte.

**Why this was the right shape of fix.** The problem was *dependence between rows*, so the fix had to be
structural (make rows independent), not statistical (adjust a threshold). Native Leiden also avoided the
dependency conflict without sacrificing the property we needed (disjointness).

**Did it work?** Yes, measurably. At α = 0.01 the ratio of null-to-expected edges went from **3.0× to
1.0×** — the tests became calibrated. Metacell + α=0.01 was the only single-run configuration whose real
graph beat the null's 95th percentile (16 edges vs 14).

**The cost, stated honestly.** *n* drops from ~1,900 to 133. That is a real loss of power, and it's why
several genes yield no edges at all. We took an honest small *n* over a dishonest large one — if 133 is
too few for FCI, that is itself a reportable result rather than something to paper over.

`pipelines/empirical/analysis/scripts/metacell_assign.py`, `build_metacell_matching_dir.py` ·
[aggregation_bakeoff.md](../../pipelines/simulation/aggregation_identifiability/results/aggregation_bakeoff.md),
[null_calibration_findings.md](../../pipelines/empirical/evaluation/results/null_calibration_findings.md)

---

## 5. Why bootstrap stability selection became the headline

**The problem.** Even with calibrated tests, a *single* FCI run is one draw. With n = 133, some edges
appear because of which rows you happened to have.

**What we do.** Re-run discovery on many resamples (n_boot = 50, 70% subsample) and report each edge's
**recurrence frequency**. Edges appearing in ≥50% of resamples are "stable".

The historical fixed-partition comparison found 12–57 stable edges on the analyzed data and zero after the tested permutations, with a maximum null edge frequency of 0.12–0.22. Recurrence under resampling was therefore useful for screening in that partition. Changing the partition tested a separate source of variation and changed many edge identities. [Partition comparison](../../pipelines/empirical/evaluation/results/partition_stability_findings.md)

**The limitation.** It's row-subsampling, not a block bootstrap. Donor blocks would be the standard
upgrade, and they are **impossible here** — there are no donor labels to block on
([L1](LIMITATIONS.md#l1-donor-is-not-an-available-variable), [L4](LIMITATIONS.md#l4-stability-is-row-subsampling-not-block-bootstrap)).

`pipelines/empirical/analysis/scripts/bootstrap_stability.py`

---

## 6. Why FCI/PAG, and not PC or a DAG learner

**The core issue: causal insufficiency.** PC and DAG learners (NOTEARS, DAGMA, GES) assume **causal
sufficiency** — that every common cause of any two measured variables is itself measured. For a gene's
regulatory neighbourhood that is plainly false: transcription factors, signalling state, and local
chromatin context all drive multiple peaks at once and none of them are in our matrix.

**What happens if you assume it anyway.** The algorithm must explain shared variation with the variables
it has, so it invents direct peak→peak edges that are really "both driven by something we didn't
measure". You get a confident, wrong graph.

**How to interpret the output.** FCI returns a partial ancestral graph (PAG). Under its assumptions, fixed endpoints are shared across compatible maximal ancestral graphs. An arrowhead at a node rules out ancestry of the other endpoint under the usual no-selection interpretation. A circle leaves the endpoint unresolved. Two arrowheads can be compatible with latent confounding. An adjacency may represent an inducing path through unmeasured variables. A fitted endpoint on constructed metacells cannot by itself establish direct regulation or name a hidden cause. [FCI interpretation](identifiability/fci_identifiability_summary.md) · [Project limits](LIMITATIONS.md#l6-observational-pag-not-dag)

DAGMA served as a sensitivity analysis. The results for a single algorithm are especially fragile when changing the metacell partition also changes the identities of selected edges. [Partition evidence](../../pipelines/empirical/evaluation/results/partition_stability_findings.md)

---

## 7. Why GFCI was added, and why FCI still carries direction

**The problem.** Plain FCI starts from a fully connected graph and prunes, which tends to leave graphs
too dense at our sample size.

**Why GFCI.** GFCI builds the skeleton with a *score-based* search (FGES) first, then applies FCI-style
orientation. On metacell CD14, GFCI returned one edge where FCI returned 28. GFCI, FCI and RCD picked the same peak–CD14 adjacency with target-aware eQTL support. Agreement across methods describes sensitivity to the learner. Accuracy on the biological graph is unknown.

**The catch we hit.** Tetrad's `Gfci` does **not** propagate background-knowledge tiers to the PAG
endpoints (causal-learn's FCI does). So GFCI runs confounder-conditioned but leaves cross-tier edges
undetermined.

**The resolution.** GFCI contributes an adjacency comparison. FCI contributes endpoint estimates under the specified orientation constraints. Both are reported
([L10](LIMITATIONS.md#l10-gfci-orients-the-skeleton-only)).

*Debugging note worth keeping:* Tetrad's `Knowledge` object silently binds to nothing when variable
names contain `:` or `-` — which every genomic coordinate does (`chr5:140638860-140639949`). It fails
without erroring. Fixed by renaming variables to safe tokens before the search and mapping back after.

`pipelines/empirical/analysis/scripts/run_gfci_tetrad.py`

---

## 8. Why library depth as root nodes, plus a mechanistic prior

RNA and chromatin library depths can affect observed counts and create associations. The empirical graph includes depth variables as roots. A separate ordering assumption permits chromatin-to-expression edges and forbids the reverse direction in this analysis. The tier encodes a biological hypothesis. Different cells in the two assays provide no temporal evidence for that direction.

In the historical 15-gene panel the prior oriented all 19 selected peak–gene edges. Depth adjustment also changed which edges were selected at PTPRJ. External peak–gene evidence assesses support for a pair. It does not independently test the direction assigned by the tier. [Implementation and measured outcomes](../../pipelines/empirical/evaluation/results/expanded_panel_confounder_results.md) · [PAG limitations](LIMITATIONS.md#l6-observational-pag-not-dag)

---

## 9. Why donor and cell-cycle are *not* in the model

No verified donor identifier connects RNA and chromatin in the PBMC data. The chromatin HTO labels describe technical groups. Mixing of those labels inside metacells cannot establish that aggregation removed donor or batch effects. The CD14 monocyte analysis also found little cycling signal in the available cell-cycle scores. These findings explain which covariates the historical fit used. Correlations can remain after depth adjustment, and their biological or technical cause can be unidentified. [Provenance correction](../../pipelines/empirical/data/results/data_sources.md#provenance-correction-2026-09-10) · [Donor limitations](LIMITATIONS.md#l1-donor-is-not-an-available-variable)

---

## 10. Why three validation axes, and why target-gene-aware

**The problem.** Discovery must not be graded by the priors that produced it. Validation has to come
from evidence the pipeline never saw.

**Why three, and why these three.** Each has a different failure mode, so agreement across them is
meaningful:

| axis | source | what it would miss alone |
| --- | --- | --- |
| prediction | ENCODE-rE2G | it's a model — inherits its own training biases |
| physical | BLUEPRINT PCHi-C | contact ≠ regulation; sparse coverage |
| genetics | eQTL Catalogue | needs a common variant in the right place |

The historical tables call an edge “validated” when at least one axis supports that specific pair. Agreement on two axes adds corroboration while the causal direction remains open.

**Why "target-gene-aware" is the crucial qualifier.** Early on, an edge counted as supported if the peak
overlapped *any* annotated regulatory element. rE2G immediately showed why that's too weak: our
then-flagship "stable CD14 edge" peak is the **TMCO6 promoter** — a real regulatory element, for a
different gene. Under the loose rule it validated. Under the strict rule
(`region_interval_with_matching_target_gene`: the element must be supported as a regulator *of this
specific gene*) it does not.

That one catch set the standing rule: **trust no promoter→expression edge without target-gene evidence.**

**A consequence we accept.** peak↔peak edges have no target gene, so this rule cannot validate them —
they're reported as `no_target_validator`, never "validated". That gap is exactly what §12–14 were built
to fill.

`pipelines/simulation/unpaired_multimodal/scripts/build_{re2g,pchic,eqtl}_gene_link_support.py`

---

## 11. Why ePerturbDB was dropped

Perturbation data is the strongest possible evidence — it's interventional. So it was the first source
added.

**Why it went.** It returned **zero** hits for CSF1R, CD14, and IL1B. So did the ENCODE-rE2G CRISPR
benchmark and IGVF. The relevant experiments simply have not been done in myeloid cells for our targets.

**The decision.** Stop investing in perturbation databases; keep a standing query in case myeloid
FlowFISH data appears later. A source that returns nothing for every target isn't a validation axis, and
continuing to maintain it would have cost time for no coverage.

---

## 12. Why the element↔element null matches on distance

**The problem.** We want to ask "do bidirected pairs share property X more than chance?" — but there is
**no accepted null for element↔element claims** in the field. We had to build one, which meant choosing
what "chance" means.

**The trap.** Genomic distance is the dominant confounder of everything in this domain. Nearby peaks
share TF motifs, chromatin domains, contacts, and accessibility *by proximity alone*. And bidirected
edges cluster at short range. So comparing them to a random pair guarantees a false positive on
essentially any signature.

**What we do.** Bin all pairs by genomic distance and permute the edge label **within** bins (2,000
permutations). This asks the sharper question: *among pairs the same distance apart, do bidirected ones
differ?*

**Evidence the null does its job.** The PCHi-C axis looked mildly positive on raw rates (7.1% vs 6.6%).
After distance matching the effect vanished (p = 1.00) — the raw signal was pure distance artifact. The
null caught a false positive we would otherwise have reported.

The later cross-mark analysis matched on both distance and graph degree and used other observed edges as a comparison group. Partition-stable edges were enriched for contacts to a shared promoter under those controls. [Distance and degree controls](../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md) · [Scope of the null](LIMITATIONS.md#l7-elementelement-validation-has-no-accepted-null)

---

## 13. Why we benchmark against co-accessibility

The empirical comparison asked whether absolute Spearman correlation distinguished pairs assigned different PAG endpoints. The average scores for bidirected and other adjacent pairs were similar (0.238 and 0.255). This describes two fitted edge classes and one correlation baseline. The true mechanism for each biological pair is unknown. A direct comparison with a full distance-penalized co-accessibility method would address a different question. [Comparison and validation results](../../pipelines/empirical/evaluation/results/peak_peak_validation_results.md) · [Limitations](LIMITATIONS.md#l8-the-co-accessibility-contrast-uses-a-spearman-baseline)

---

## 14. Why the global-activity probe leaves the locus out

A summary calculated from the tested peaks can induce dependence when used as a covariate. Earlier self-derived principal components inflated the graph’s edge counts. The revised probe used the mean standardized signal from peaks at other panel loci. Under that proxy, the reported bidirected count changed from 126 to 129, and the number of externally supported peak–gene pairs stayed at 11. The check tests this particular proxy. An unmeasured global or local biological driver remains possible. [Probe and evaluation](../../pipelines/empirical/evaluation/results/peak_peak_validation_results.md)

---

## 15. Why these 15 genes

**The problem.** The original three targets (CD14, CSF1R, IL1B) had thin validation coverage — CD14 had
1 eQTL dataset, CSF1R 2, IL1B **0**. If the reference data can't speak to a gene, discovery around it
can't be checked either way.

**What we did.** Ranked all genes by *validation coverage* (genome-wide rE2G distal links × eQTL credible
sets across 11 monocyte datasets) crossed with expression in our own RNA. 331 genes are both
validation-rich and expressed ≥15%. We took ~15 of the best that also had usable nearby peaks.

**Why this isn't cherry-picking.** We select on *whether a gene is checkable*, never on whether it gives
the answer we want. A validation-rich gene that yields nothing is a reportable negative — and several
do: 4 of 15 produce no peak→gene edge at all, and PTPRJ produces 2 directed edges with 0 validated.

`pipelines/empirical/analysis/configs/gene_panels/` · [target_gene_selection.md](../../pipelines/empirical/analysis/results/target_gene_selection.md)

---

## 16. Why monocyte, and not T cell

**Context.** The data is PBMC — T cells are actually the *largest* population (20.6k vs 13.4k monocytes).
Monocyte was chosen initially for the best scGLUE integration confidence (0.96), not because it was the
only option.

**So we tested the alternative properly.** Built the entire T-cell validation suite (rE2G 2.3M links,
18 resting eQTL datasets, PCHi-C), did T-cell-specific gene selection (12 genes from 286 candidates), and
ran the identical pipeline at n = 245 metacells.

**The result, which was not what we expected.** Despite ~1.8× more cells and a ~3× richer reference
suite, T cell yielded **fewer** validated stable edges: 4 vs 7 (0.36 vs 0.47 per gene).

Monocytes remained the primary empirical panel. The T-cell comparison also changed the population, assayed genes and reference coverage, so it does not isolate a sample-size or effect-size explanation. It provides a cross-cell-type sensitivity check. Treating cell types as valid invariance environments would require additional assumptions about which mechanisms are shared. [Cell-type comparison](../../pipelines/empirical/evaluation/results/celltype_fair_comparison.md) · [Sampling limits](LIMITATIONS.md#l1-donor-is-not-an-available-variable)

---

## The through-line

Nearly every decision above came from the same loop: **build a way to measure whether the current thing
is working → discover it isn't → change the design, not the threshold.**

The null-calibration test (§4) is the clearest case — it turned "the graphs look plausible" into "the
graphs are noise", which forced the aggregation rebuild, which made stability selection meaningful,
which made validation worth doing. The distance-matched null (§12) did the same job for
element↔element claims, and the collider mistake (§14) is a reminder that a reasonable-looking control
can make things actively worse.

If you add a stage, add its diagnostic first.
