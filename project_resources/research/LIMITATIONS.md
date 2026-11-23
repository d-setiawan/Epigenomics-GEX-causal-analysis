# Limitations & honesty ledger

**Owner doc:** this file is the *single home* for every known caveat. Other project_resources/research must **link here**
rather than restate a limitation, so a caveat can never be fixed in one place and left stale in
three. ↑ [Central README](../../README.md) · [MISSION](MISSION.md) · [ROADMAP](ROADMAP.md)

Each entry: what the limit is → what it blocks → what we do instead.

---

## L1. Donor is not an available variable

**Corrected 2026-09-10.** There is no verified donor correspondence across the
RNA and chromatin inputs. The chromatin `donor_id` field is the dominant HTO
class, not a validated donor ID: the source protocol split a PBMC sample into
hashtagged tubes. Technical tag groups are available; independent biological
donor labels and a shared cross-assay donor map are not established. See the
[source correction](../../pipelines/empirical/data/results/data_sources.md#provenance-correction-2026-09-10).

**What it blocks.**

- **A justified donor-level block bootstrap cannot be performed with the current
  metadata.** HTO blocking would test technical-group sensitivity, not donor replication.
- **"Invariance across donors" is not an available route to causal claims.** Earlier drafts of
  MISSION §4 listed donor-invariance as a pillar of the method; it has been replaced.
- `--confounders depth,donor` exists in the CLI but is inert (it is honored only if a `donor__*`
  column exists, and none does). Treat it as unavailable.

**What we do instead.**

- **Cross-cell-type comparison** is an available sensitivity axis, not automatically
  a valid ICP environment: biological mechanisms may differ between cell types.
  The T-cell suite is already built
  ([celltype_fair_comparison.md](../../pipelines/empirical/evaluation/results/celltype_fair_comparison.md)); the ICP run
  itself is still open (ROADMAP §0.5).
- Retain library depth as an observed technical covariate and explicitly report
  unmeasured batch, donor and state effects. Hashtag mixing across metacells does
  **not** show that donor/batch effects are removed or that depth is sufficient.

**Consequence for the science.** The previous claim that aggregation neutralized
donor/batch and therefore made residual bidirected edges biological is withdrawn.
Those edges remain compatible with biological or technical unmeasured causes;
the graph alone does not identify which.

## L2. Pseudo-pairs are not within-individual

**Corrected 2026-09-10.** RNA and chromatin cells are matched through the scGLUE
latent, not by verified donor or cell identity. Pseudo-pairs are **not established
within-individual pairs**; neither shared nor distinct donor identities can be
deduced from the alignment. Their intended correspondence is cell state, itself
under review ([L17](#l17-biological-integration-is-not-established-by-modality-mixing)).

**What it blocks.** Alignment alone cannot reconstruct matched-donor genotype
covariation or cell-specific cross-assay residuals. No donor-level eQTL inference
is justified from these pseudo-pairs. Validated state alignment could support
state-associated comparisons, not establish direct regulation.

> This is consistent with an observation already logged independently: eQTL support localises to the
> CD14 promoter but **never changes a peak→gene edge verdict** — the fine-mapped variants sit in the
> promoter, not in the distal peaks discovery connects to expression
> ([PIPELINE.md](../../pipelines/empirical/analysis/PIPELINE.md)). Worth stating explicitly in the write-up as a
> scope limit on the genetics axis rather than a null result.

## L3. Small n, and n_eff is not reported

≈133 metacells per gene. Honest but small — hence stability selection and transforms. The
`n_eff` diagnostic (participation ratio / reuse proxy) is still unimplemented, so for any legacy
kNN-averaged result the effective sample size is well below the row count and fisherz p-values are
optimistic. The current metacell datasets are non-overlapping, so `n_rows ≈ n_eff` there.

## L4. Stability is row-subsampling, not block bootstrap

n_boot 50, threshold 0.5. Per L1, donor blocks require a verified biological donor
map that is currently absent. Cell-state/cluster blocks or HTO technical-group
sensitivity checks address different dependence structures and are not donor replication.

## L5. The genetics axis is cell-type aligned but not always state aligned

The only monocyte eQTL credible sets for CD14/CSF1R come from **TLR-stimulated** monocytes
(Quach_2016 Pam3CSK4/R848), not resting. IL1B has no monocyte credible set anywhere; naive-monocyte
datasets (BLUEPRINT/Quach/Schmiedel/Randolph) have zero credible sets for those targets.

## L6. Observational, PAG not DAG

Without monocyte CRISPR, every causal claim is conditional on assumptions (Markov, faithfulness, the
mechanistic orientation constraints) plus corroboration by independent evidence. Causal insufficiency
is permanent and is *handled* — via PAG bidirected edges — not assumed away. Output is
hypothesis-generating: a prioritized set of element↔element hypotheses for CRISPRi follow-up.

## L7. Element↔element validation has no accepted null

There is no community-standard null for element↔element claims, which is why we built the
distance-matched permutation framework. Current null matches on **genomic distance only**;
degree-matching is not yet implemented. See
[peak_peak_validation_results.md](../../pipelines/empirical/evaluation/results/peak_peak_validation_results.md).

## L8. The co-accessibility contrast uses a |Spearman| baseline

Not a faithful Cicero. Raw co-accessibility does not prune indirect paths through *observed* peaks,
so a distance-penalized graphical-lasso would remove many of the non-adjacent false positives. The
**method-fair** part of the contrast is the *latent* class: graphical-lasso removes indirect paths
through observed variables but cannot remove a shared latent driver. That is the theoretical
guarantee behind the 53% number.

## L9. Panel scale is a pilot

15 genes. Every enrichment/contrast number is pilot-scale until the panel is scaled out.

## L10. GFCI orients the skeleton only

Tetrad's `Gfci` does not propagate the `Knowledge` tiers to PAG endpoints (causal-learn's FCI does),
so GFCI contributes multi-algorithm *adjacency* while **directed claims come from FCI**.

## L11. Edge identities are partition-dependent

Discovery runs on metacells, and the Leiden partition is one arbitrary draw. Re-clustering the same
cells with a different random seed — same algorithm, same parameters, same n=133 — **loses 9 of 11
validated peak→gene edges**. Across three partitions, **zero** validated edges survive all three; two
CD14 edges survive two of three, and those are the project's flagship edges.

**What it blocks.** Any claim of the form "peak X regulates gene Y" from a single partition. The
aggregate counts (19 directed, 11 validated) *do* reproduce, so distribution-level claims are on firmer
ground than individual-edge claims.

**Why the current stability tier does not cover this.** Bootstrap stability selection resamples rows
*within* a fixed partition, so it structurally cannot observe between-partition variance. The "5
stable+validated" tier inherits the blind spot.

**What we do instead.** Nothing yet — this is unresolved and is the current top priority. The proposed
fix is an outer bootstrap over partitions — **now built and run at K=10**
([partition_bootstrap_results.md](../../pipelines/empirical/evaluation/results/partition_bootstrap_results.md)): no peak→gene
edge is recovered in all 10 partitions, 68% of edges appear in a single partition, and the most robust
validated edge (rE2G-supported CD14 H3K4me3) is found in 6/10. Report edges by recovery frequency, not
from one partition. **Mechanism (2026-07-22 simulation):** aggregation induces collinearity between causal and confounded elements that collapses the causation-vs-confounding distinction — identifiable at the cell level, lost after aggregation ([simulation_identifiability_results.md](../../pipelines/simulation/aggregation_identifiability/results/simulation_identifiability_results.md)). Background: [partition_stability_findings.md](../../pipelines/empirical/evaluation/results/partition_stability_findings.md).

## L12. Per-mark technical depth is not conditioned on

We enter two observed technical confounders: `conf__rna_depth` and `conf__chrom_depth`. The second is
`log1p(mean library size across the six histone-mark channels)` — **a single averaged variable**. Depth
variation *specific to one assay channel* is therefore uncontrolled, even though the per-mark library
sizes (`libsize__<MARK>`) are present in every matrix.

**Why this now matters.** Bidirected `<->` pairs are strongly enriched for being measured on the **same
histone mark** (43.7% vs 29.2% distance-matched, p = 0.0005; robust to degree matching). Two peaks from
the same assay share that assay's per-metacell technical fluctuation, which would induce covariation
with no directional structure — exactly what FCI reports as a latent common cause. The enrichment is
specific to the bidirected class (the `o-o` class shows +0.037, p = 0.16), which is what a shared
*unmodelled confounder* predicts.

**What it puts at risk.** Some unknown fraction of the 126 bidirected edges — the project's novel object.

**Update 2026-07-21 — RESOLVED, biological.** Two technical explanations were tested and both
refuted: (1) per-mark sequencing depth — the enrichment survives six per-mark depth roots (p=0.0025);
(2) shared cell substrate — a split-half test (quantify the two peaks from *disjoint* cells within each
metacell, 20-split averaged) shows the same-mark bidirected correlation **fully survives** breaking the
shared cells (104% of the excess preserved). So the same-mark enrichment is genuine metacell-level
(cell-state) covariation, not a depth or substrate artifact — the project's positive element↔element
signature. This limitation is downgraded to a recommendation: **prefer `--confounders depth_permark`**
(per-mark roots) as the more correct conditioning, though it does not change the same-mark conclusion.
Full evidence chain: [same_mark_finding.md](../../pipelines/empirical/evaluation/results/same_mark_finding.md). Remaining
caveats there (n=55, single cell type, correlational).

**Correction 2026-07-22 (partition bootstrap).** The "resolved as biological" verdict above was from a
3-seed pilot and was **partition-lucky**. Across 10 partitions the same-mark enrichment is significant
in only **4/10** (mean +0.046, negative in 3); the two strong partitions include the canonical seed 0.
It is not a technical artifact where it appears, but it is **partition-fragile and cannot be claimed as
an established signature**. See [partition_bootstrap_results.md](../../pipelines/empirical/evaluation/results/partition_bootstrap_results.md).

**Further correction 2026-09-10.** Passing the tested depth and split-half checks
does not rule out all technical artifacts or establish biological causation.
The earlier phrase “not a technical artifact” is too broad. Biological alignment
also requires review independently of those checks; see [L17](#l17-biological-integration-is-not-established-by-modality-mixing).

## L13. Peaks are histone-mark peaks, not ATAC; validation is enhancer/ATAC-derived

The covariates are **per-histone-mark CUT&Tag peaks** (H3K27ac, H3K4me1/2/3 = active; H3K27me3, H3K9me3
= repressive), not ATAC open-chromatin peaks. Consequences:

- A "peak" is region **+ chromatin state**, so one genomic locus can appear as up to 6 separate
  variables (one per mark). Same-locus different-mark covariation is biologically expected (active
  elements carry H3K27ac + H3K4me1 together) — relevant to the same-mark analysis
  ([same_mark_finding.md](../../pipelines/empirical/evaluation/results/same_mark_finding.md)).
- The three validation axes (ENCODE-rE2G, BLUEPRINT PCHi-C, eQTL) are **enhancer / ATAC / genetics**
  derived. Validating a histone-mark peak against them is really "does this interval correspond to a
  known regulatory element of this gene" — coherent for **active** marks (which do mark enhancers), a
  **category error** for repressive marks (H3K27me3/H3K9me3 are not enhancers, so an enhancer-gene
  database cannot meaningfully confirm them).

**Empirically this is mostly self-correcting:** across the 10-partition run, 91/94 peak→gene edges land
on active marks and 55 of 56 validated edges are on active marks; only **1** validated edge is on a
repressive mark (H3K27me3) — a genuine but negligible category error. The mechanistic prior + FCI route
almost all expression-linked edges to active marks, which is biologically right. Still: repressive-mark
peaks should be excluded from (or flagged in) peak→gene validation, and the write-up must say "histone
CUT&Tag peaks," never "ATAC/accessibility."

## L14. Same-mark and cross-mark covariances are not on the same scale

A metacell holds cells from every assay, and variable *j* is averaged over that metacell's
**mark-ch(j) cells only**. Two variables of the same mark are averaged over the *same* cells; two
variables of different marks over *disjoint* cells. So the sampling noise of the metacell mean
contributes to the covariance of a same-mark pair and contributes **nothing** to a cross-mark pair,
while still inflating both variances:

```
Cov(X_j, X_k) = Sigma_between[j,k] + 1{same mark} * E_c[ Sigma_within[j,k] / n(c) ]
```

Every same-mark entry of the covariance fed to FCI/BOSS therefore carries a positive term that no
cross-mark entry carries, and every cross-mark **correlation** is attenuated by the inflated
denominator with no compensating numerator. This is an artifact of the assay design, not a property of
chromatin, and it is sufficient on its own to make same-mark edges easier to find than cross-mark ones.

In simulation the effect is direct and large: at matched cells-per-variable, removing cell *sharing*
drops same-mark adjacency recall from 0.803 to 0.054, and **thinning** cells per metacell *raises*
same-mark recall (0.532 → 0.803) because a smaller n means a larger noise term. Any comparison of
same-mark against cross-mark findings must account for this.
See [crossmark_identifiability.md](../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md).

## L15. Most panel variables carry no between-state signal

Splitting each variable's metacell-level variance into real between-state signal and the sampling
noise of averaging ~11 cells per mark per metacell, the median variable's signal share across the panel
(442 variables, 15 genes) is **−0.017**: its entire between-metacell variance is accounted for by
sampling noise. Only ~19% exceed a share of 0.10 and ~12% exceed 0.25.

This is a precondition, not a modelling preference: a variable with no between-state signal has no
structure for any search to recover, and it still occupies every conditioning set.

**The underlying cause is sparsity.** The peak x cell matrices are **0.5-1.5% dense** (verified from
the matrix headers). A peak is detected in ~1% of cells and a metacell holds ~11 cells per mark, so a
given (peak, metacell) has ~0.1 expected detections — most of the metacell matrix is zero because
nothing was observed, not because absence was measured. Pooling adjacent peaks raises counts and
detection ~7x without moving signal share at all.

**Scope limit on this caveat.** The estimate is ~0 under a Gaussian share, a count-native Poisson
dispersion, and a distribution-free test alike, and no non-Gaussian view rescues a variable the
Gaussian one discards (0 of 442). But the count *test* at these depths rejects 0.000 under its own
null and never reaches 80% power even for a 20-fold swing, so **"no between-state signal" must not be
stated as established absence** — the defensible claim is that this design cannot resolve it. Marginal
per-variable power is also much weaker than pairwise-association power, which is why externally
validated cross-mark adjacencies coexist with marginally invisible variables. Do not use this caveat
to rule out algorithm, CI-test, or model-class changes; test those at the edge level.

**Signal share is itself partition-dependent** — for CD14, 3–9 variables clear 0.10 on a single
partition but only 2 clear it on the cross-partition median. The instability documented in
[L11](#l11-edge-identities-are-partition-dependent) reaches one level below the edges, into which
variables qualify as signal-carrying at all.

**Screening on signal share does not fix it.** At matched panel size, screened panels produced fewer
edges than random subsets and **zero** partition-stable edges.
See [crossmark_identifiability.md](../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md).

## L16. CA-BOSS cached shared-state recovery is not directed causal certification

_Added 2026-09-06 for the separate unpaired-multimodal simulation track._

The [current CA-BOSS checkpoint](../../pipelines/simulation/unpaired_multimodal/results/ca_boss_v4_roadmap_status.md#cross-assay-benchmark-metrics)
measures shared-state hypotheses, assay participation, and unambiguous
state-associated pairs. It does not establish a directed cross-assay causal
graph. An ambiguity hypothesis is scored as one set that can match a truth;
its alternative pairs are not individually validated edges. The exact-pair
stress test credits unambiguous shared-state matches, not direct regulatory
effects. Direction is undefined for the benchmark's shared-state truths.

The campaign uses cached worlds that informed method development. Freezing
predictions before target scoring and excluding the target block from null
calibration protect that scoring step, but do not undo adaptive reuse of the
world pool. Observed perfect precision is therefore not a guarantee of
unseen-world or real-data reliability. Discovery uses observed variables;
other-block simulation truth still identifies empirical calibration nulls.
There is no claim that the entire evaluation is label-free.

Calibration-world bootstrap and target-cell/representation refitting test
different uncertainty sources. Bootstrap repetitions are dependent, and the
checkpoint's working Beta comparison is not a calibrated causal posterior.
Later representation and reporter-estimator diagnostics are not gains until
they improve the scored checkpoint under the same reporting contract.

Within-assay, broad state-membership, and cross-assay-pair results use different
estimands and sometimes different world cohorts. Their denominators and
sentinel tests must remain separate. Recasting direct edges as shared-state
sets also changes the reported graph object; lower pairwise density is not
by itself proof of improved causal recovery. Existing-world reuse is the
current constraint; these caveats do not authorize new datasets, paired cells,
interventions, or external biological inputs. Exact results and recorded
negative experiments remain in the
[roadmap status](../../pipelines/simulation/unpaired_multimodal/results/ca_boss_v4_roadmap_status.md) and
[experiment log](../../pipelines/simulation/unpaired_multimodal/results/channel_aware_boss_v4_experiment_log.md).

## L17. Biological integration is not established by modality mixing

_Added 2026-09-10 for the real-data biological review; separate from CA-BOSS._

**The evidence.** The existing scGLUE validation reports modality mixing and
RNA labels transferred through the same embedding. Transfer confidence measures
neighbor agreement, not independently measured biological accuracy. The saved
run did not compute guidance consistency; even that check would assess agreement
with a modeling prior, not independent cell-state truth. Existing ADT-marker
checks show label/marker disagreements requiring investigation; their measured
results are in the [biological review](../../pipelines/empirical/data/results/biological_foundation_review_2026-09-10.md).

**What it blocks.** Current monocyte pseudo-pairs and downstream real-data graphs
must not be treated as biologically validated merely because modalities mix or
RNA-label transfer is confident. Existing protein measurements provide an
orthogonal assay check, **not definitive labels**: antibody background, specificity,
sparsity, doublets and barcode/feature provenance can also cause disagreements.
Protein-assisted integration would consume that validation channel and must be
reported as a distinct input contract, not a histone–RNA-only success.

**Preprocessing and guidance.** A common narrow-peak caller and abundance-ranked
bin selection may underrepresent broad repressive-mark domains. Retained-bin
counts are not FRiP or original fragment depth; empty retained count rows do not
imply empty original cells. Broad-domain, promoter and enhancer marks need
appropriate, separate quality checks. Gene-distance windows and positive/negative
mark signs are priors, not measured regulatory effects: H3K4me1 need not indicate
an active enhancer, and repressive marks are not universal negative-expression
measurements. This also qualifies the simplified active/repressive labels in L13.
Overlapping-bin quantification approximates a peak interval and requires a
locus-specific aliasing/coverage audit before interpreting exact peak identities.

**What we do next.** Reconcile sample/barcode/feature provenance, check RNA QC and
annotations, and evaluate biological conservation using held-out ADTs/features
alongside mixing. Preserve unmatched populations and distinguish internal
guidance checks from independent evidence. The initial numerical audit is
descriptive, not a full structural/checksum validation of every input; raw RNA
QC and retained-cell labels use different cell sets. No causal biological
certification, automatic relabeling or dataset replacement follows from the
review. Controlled retraining was separately authorized on September 16.

**Investigation update 2026-09-11.** The
[existing-data follow-up](../../pipelines/empirical/data/results/biological_investigation_2026-09-11.md)
passes local ADT/HTO/count lineage and saved coarse-label replay checks. This rules
out the tested local corruption/reordering explanations, not all implementation
errors or errors in original experimental feature identities. It does not certify
fine labels or biological correspondence. RNA marker checks reuse the expression
used for annotation and are not independent label validation.

Multi-protein contrasts were fixed for the diagnostic, not preregistered or validated
classifiers. Upper-ADT-quartile selection changes cell composition and is neither
depth matching nor background adjustment; zero median isotype counts do not eliminate
antibody-specific background. Within-assay neighbor conservation is descriptive and
unadjusted for depth, not a method ranking or a causal test.

Exact stage-subset checks establish mark-dependent **count truncation**, not loss of
biologically informative counts or the cause of misalignment. The saved LSI remains
based on its larger preprocessing feature set even when training reconstructs fewer
features. Fixed-bin integration does not consume peak calls: peak-definition defects
cannot directly explain the separate fixed-bin feature-selection loss. Original
RNA annotation ambiguity and mitochondrial flags require a separate QC decision;
they do not justify automatic relabeling or exclusion.

**Execution update 2026-09-16.** The
[alignment baseline](../../pipelines/empirical/integration/results/biological_alignment_2026-09-16.md)
adds development-only protein prediction with conditional shuffles, nuisance
adjustment sensitivity and independent numerical replication. Previously examined
data cannot become a pristine prospective holdout merely by assigning a new split;
the outcome reservation starts now. Existing embeddings remain transductive.
Conditional-shuffle ranges are descriptive, not calibrated confidence intervals
or significance tests, and adjustment may remove genuine biology. Matched-method
intersections exclude cells, particularly H3K9me3, so full-cohort coverage is
reported separately. Histone-to-histone protein agreement does not independently
validate RNA anchoring, recover missing same-cell residuals or establish causality.

## L18. Latent-space aggregation does not reconstruct the missing joint distribution

_Added 2026-09-16 for the aggregation-first research track._

**The limit.** A shared latent space can align cells by estimated biological
state, but different assays still measure different cells. State-matched
pseudobulks can estimate covariance among assay-specific conditional means. They
do not observe the within-state, same-cell residual covariance needed to recover
the unrestricted cell-level joint distribution. The derivation and model
counterexample are in
[the latent-space aggregation note](theory/06_latent_space_aggregation.md).

**What it blocks.** Embedding proximity, modality mixing, one-to-one matching,
neighbor averaging, Leiden metacells, or optimal-transport coupling cannot alone
certify that a cross-assay adjacency is a direct cell-level regulatory edge. A
downstream graph can reflect shared state, the integration guidance prior,
matching choices, complete-group selection, cell reuse, unequal aggregate sizes,
or the aggregation operator. Gaussian observational recovery generally targets
an equivalence class rather than every direction.

**What we do instead.** Separate alignment, covariance and graph targets in the
benchmark. Use common-cell aggregation as the positive control; compare arbitrary,
oracle-state and learned-state matching on the same cell draw; hold focal variables
out of integration; and include shuffled-guidance and row-permutation nulls. Any
covariance completion must state its assumption about the unobserved residual
cross-assay block and report sensitivity to alternative admissible completions.
Exact cross-assay direction remains an identifiability stress test.

**Integration-simulation interpretation, 2026-09-23.** Learned embedding
coordinates can mix biological variables; a graph over those coordinates is
not the original region–RNA graph. Graph recovery must name its original-feature
reconstruction or other readout. Hidden paired synthetic cells may score frozen
representations, but a probe trained on their activities or cross-assay targets
is an evaluator diagnostic, not an unpaired discovery procedure. Predicting
state or state-dependent means does not by itself certify causal edges; a
learned state coordinate is not automatically an upstream biological cause.

A biological activity SEM followed by noisy count measurement also has distinct
activity-level and measurement-level graphical targets. In particular,
conditioning on a noisy count proxy for a mediator need not block the original
biological path. Paired-count performance is therefore a required reference
when interpreting integration losses, not a general promise that the activity
DAG is recoverable by ordinary CI tests on counts. The
[integration simulation proposal](../../pipelines/simulation/integration_recoverability/design.md)
specifies the reference comparisons. Even when a state-conditioned product model
identifies a joint observational distribution, identifying a causal graph
requires further assumptions; conditional independence is not an
assumption-free proof that no biological direct effect exists.

**Formal boundary, 2026-11-01.** The
[unpaired-integration proof](theory/07_unpaired_integration_nonidentifiability.md)
uses faithful models with identical full observed assay laws but different
direct edges, effects and state-conditioned CIs. Under independent disjoint
sampling, every integration/readout has the same output law in the two models;
general exact recovery is therefore impossible over a class containing both.
This is not an assertion that every particular dataset or every restricted
nonlinear model is nonidentified. The note states positive exclusion,
conditional-copula and unlinked-regression cases. It also distinguishes direct
biological edges from observed-variable PAG adjacencies: the state-loaded
fork/triangle pair alone does not change a two-node PAG after hiding state.

The exact product-law proof does not apply to two shuffled views of the same
paired cohort. The scMTR-seq holdout has that design, even though its training
assay pools are disjoint. Local assay QC can be included in a local observation
kernel; inherited filtering using jointly measured targets requires a separate
selected-population model. None of the pilot's failures is used as proof of
nonidentifiability. A graph learned on constructed pseudo-pairs still needs an
explicit biological joint-completion assumption before ordinary causal-discovery
guarantees can be transferred to the original variables.

**Positive-model boundary, 2026-11-01.** The
[literature review](integration_literature_scope_2026_11_01/REVIEW.md)
identifies a published theorem for recovering a shared latent causal graph from
unpaired domains under restrictive linear measurement, source-distribution and
graph conditions. Thus extra data are not the only possible route to
identification. Those assumptions have not been established for our sparse
counts, and a latent graph is not automatically an original peak–gene graph.
The general proof applies to classes containing its countermodels, not to every
restricted model compatible with unpaired observations.

The subsequent [native Sturma pilot](sturma_pilot_2026_11_01/README.md) separates
published-model replication from direct count application. Its small
training-selected feature panels do not guarantee informative partial pure
children, and full numerical count rank need not equal biological dimension.
Failure to match estimated sources does not establish absence of shared biology;
paired moment agreement would still not establish direct causal effects.

The [count-aware follow-up](count_mdcrl_2026_11_01/README.md) explicitly models
Poisson sampling and checks source uncertainty. Its affine rate model, conditional
noise assumptions and observed-depth proxy remain unverified for these data.
The dimension screen detects coherent covariance, not every latent/private
factor; finite moment/characteristic-function agreement is not full source-law
identification. Empty candidate sets produce a zero-covariance default, not an
inference of independence. A known Poisson observation kernel can preserve
population identification of each intensity law under stated conditions, so
finite-sample failure does not prove a universal count-data impossibility.

The [broader extension comparison](count_extensions_2026_11_01/README.md)
also treats NB shape and excess-zero retention as fixed sensitivities, not
identified technical noise. The larger-panel branch requires the separate
[selected-dimension follow-up](count_rank20_2026_11_01/README.md) because its
initial computational ceiling barred matching. Coherent empirical rank is
not known biological dimension. Neither source stability, finite fingerprint
agreement nor improved within-assay covariance establishes cross-assay causal
structure. Unassessed CF confirmation is distinct from a failed CF test.
Pairwise candidate covariance blocks need not form a coherent joint law.

The [known-kernel derivation](count_extensions_2026_11_01/reviews/kernel_population_identifiability.md)
extends the within-assay population argument to known NB shape and independent
zero masking under constant known exposure. Unknown noise parameters and
endogenous depth proxies are outside that claim. Repeated use of the same
cohorts, fixed-rank bootstrap uncertainty and exploratory comparisons remain
limitations; independent arithmetic review does not create fresh validation.

## L19. scMTR-seq pilot agreement is not causal ground truth

_Added 2026-11-01 for the separate scConfluence/scMTR-seq pilot._

The [S1 cohort](../../pipelines/empirical/data/scmtr_seq/README.md) supplies real
same-cell measurements for evaluation, while fitting uses disjoint cells and
only the three permitted assays. A paired-count FCI graph is an uncertain
observational reference, not an intervention-validated causal graph. Empty
paired and predicted cross-assay graphs cannot demonstrate recovery; reference
coverage with a zero denominator is unavailable.

This is one pooled-day technical sublibrary with an internal cell holdout,
not independent biological replication. The released source cells inherit a
six-target DNA-depth filter. RNA dominant-tag selection is reconstructed from
the released count matrix rather than unavailable read-summary counts; barcode
collisions and same-day doublets remain unresolved. Promoter windows exclude
distal regulation, and overlapping windows can count a fragment for more than
one gene. The frozen nonoverlapping evaluation panel is sparse.

The pilot's FCI input is log-transformed observed counts without explicit
depth adjustment. Measurement noise, count ties, depth variation and the chosen
coupling can affect its conditional-independence results. Neither nominal CI
p-values after matching nor three cell-subsampling repetitions establish false
discovery control. Stable edges under a random coupling remain possible.
The separate post-primary product-coupling audit in the
[study record](scconfluence_scmtr_pilot_2026_11_01/README.md#exploratory-check-of-the-causal-readout)
demonstrates a sparse-null calibration problem for this panel and fixed readout.
Its conditional permutation frequencies are not biological false-discovery
rates, and its posthoc detection-overlap calculation does not replace a full
count-aware conditional-independence procedure.
Published scConfluence partial transport is used during training; a downstream
full one-to-one matching diagnostic evaluates a distinct, forced coupling.

Exact cell retrieval, day-label transfer, state alignment and graph agreement
are separate diagnostics. Optimization seeds and subsamples are not biological
replicates. The [study record](scconfluence_scmtr_pilot_2026_11_01/README.md)
preserves the input contract, objective corrections, outcomes and independent
reviews. These limits qualify the pilot rather than establishing that the
whole dataset or every alternative readout is unusable.
