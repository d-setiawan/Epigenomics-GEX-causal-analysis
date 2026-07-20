# Why the pipeline looks like this

↑ [Central README](../README.md) · [MISSION](MISSION.md) (what we're trying to do) ·
[ROADMAP](ROADMAP.md) (what's next) · [LIMITATIONS](LIMITATIONS.md) (what it can't do)

**Read this if you're picking the project back up.** ROADMAP records *what* was decided in one line
each; this file records **why**, and **how we found out** — the diagnostic that forced each choice,
the alternatives we rejected, and where the code lives. Most decisions here were not made up front;
they were made because something broke and we went looking.

Every decision follows the same shape: **a problem we could measure → options → what we picked and
why**. Where a number appears, it comes from the results doc linked at the end of that section.

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
is *local* causal architecture, per gene ([LIMITATIONS L6](LIMITATIONS.md#l6-observational-pag-not-dag)).

**How.** A 300 kb window around each gene's TSS (interpretable range for enhancer–promoter contacts).
1 Mb was tried and produced graphs that were too dense to interpret.

---

## 2. Why scGLUE for integration

**The problem.** Chromatin and RNA are measured in *different cells*. Nothing can be correlated until
the two modalities live in a shared space.

**Options.** scGLUE (graph-linked unified embedding) vs. a scMRDR-style model (the "Jianle" track).

**What decided it.** A direct measurement of how well each mixes the modalities — the fraction of each
cell's nearest neighbours that come from its *own* modality. Lower is better (cells sit next to cells
from the other modality rather than clustering by assay):

| Run | same-modality neighbour fraction | silhouette by modality |
| --- | --- | --- |
| **scGLUE `joint_v2`** | **0.2548** | −0.0122 |
| scMRDR_gaussian | 0.5693 | −0.0984 |

scMRDR left cells largely clustered by assay (0.57 — barely better than not integrating), which would
make any downstream pairing meaningless. scGLUE won on the metric that matters for our use.

**Kept, not deleted.** The scMRDR track stays as a comparison so the choice remains checkable.

📁 `Integration/scripts/scglue/` · [Integration README](../Integration/README.md)

---

## 3. Why we pair cells at all

**The problem.** Causal discovery needs rows where chromatin and expression are observed *together*.
We don't have that — no cell has both.

**What we do.** Use the shared scGLUE latent space to match RNA profiles with chromatin profiles that
occupy the same region of that space, producing synthetic joint rows ("pseudo-pairs").

**What this buys and costs.** It buys a joint matrix. It costs within-individual coupling: the pairing
aligns on *cell state*, not genotype or donor
([LIMITATIONS L2](LIMITATIONS.md#l2-pseudo-pairs-are-not-within-individual)) — the chromatin data is
from GSE195725's hashed donors and the RNA is a separate PBMC experiment
([data_sources.md](../Data/results/data_sources.md)). So the covariation we can recover is
state-driven, not genotype-driven.

---

## 4. Why metacells replaced kNN-averaging — the noise-domination story

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
measurement (`Evaluation/scripts/null_calibration.py`): **permute every feature column independently**
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

📁 `CausalDiscovery/scripts/metacell_assign.py`, `build_metacell_matching_dir.py` ·
📊 [aggregation_bakeoff.md](../Evaluation/results/aggregation_bakeoff.md),
[null_calibration_findings.md](../Evaluation/results/null_calibration_findings.md)

---

## 5. Why bootstrap stability selection became the headline

**The problem.** Even with calibrated tests, a *single* FCI run is one draw. With n = 133, some edges
appear because of which rows you happened to have.

**What we do.** Re-run discovery on many resamples (n_boot = 50, 70% subsample) and report each edge's
**recurrence frequency**. Edges appearing in ≥50% of resamples are "stable".

**Why it's the primary signal/noise separator.** Run the same procedure on permuted (null) data:
real data gives 12–57 stable edges; the null gives **0 stable edges**, with maximum single-edge
frequency 0.12–0.22. Real structure recurs; noise flickers. That separation is much cleaner than any
single-run threshold, which is why the pipeline's top tier is "stable **and** validated" rather than
just "found".

**The limitation.** It's row-subsampling, not a block bootstrap. Donor blocks would be the standard
upgrade, and they are **impossible here** — there are no donor labels to block on
([L1](LIMITATIONS.md#l1-donor-is-not-an-available-variable), [L4](LIMITATIONS.md#l4-stability-is-row-subsampling-not-block-bootstrap)).

📁 `CausalDiscovery/scripts/bootstrap_stability.py`

---

## 6. Why FCI/PAG, and not PC or a DAG learner

**The core issue: causal insufficiency.** PC and DAG learners (NOTEARS, DAGMA, GES) assume **causal
sufficiency** — that every common cause of any two measured variables is itself measured. For a gene's
regulatory neighbourhood that is plainly false: transcription factors, signalling state, and local
chromatin context all drive multiple peaks at once and none of them are in our matrix.

**What happens if you assume it anyway.** The algorithm must explain shared variation with the variables
it has, so it invents direct peak→peak edges that are really "both driven by something we didn't
measure". You get a confident, wrong graph.

**What FCI does instead.** FCI outputs a **PAG**, whose edge marks encode what the data can and cannot
determine:

| mark | meaning |
| --- | --- |
| `A --> B` | A causes B (no latent confounder on this pair) |
| `A <-> B` | **a latent common cause** drives both — *not* a direct link |
| `A o-o B` | related, direction undetermined |

The `<->` edge is the honest representation of "something unmeasured explains this". **This is the
project's novel object** — not a nuisance to be cleaned up (see [MISSION](MISSION.md) §2).

**What we give up.** A PAG is weaker than a DAG: some edges stay undetermined. We accept that, because
the alternative is a stronger claim that isn't true. DAGMA is kept only as a sensitivity check.

---

## 7. Why GFCI was added, and why FCI still carries direction

**The problem.** Plain FCI starts from a fully connected graph and prunes, which tends to leave graphs
too dense at our sample size.

**Why GFCI.** GFCI builds the skeleton with a *score-based* search (FGES) first, then applies FCI-style
orientation. The score-based skeleton is sparser and more accurate — on metacell CD14, GFCI returned
1 edge where FCI returned 28. Independent agreement between different algorithm families is also
evidence in itself: GFCI, FCI, and RCD independently picked the same eQTL-validated peak→CD14 edge.

**The catch we hit.** Tetrad's `Gfci` does **not** propagate background-knowledge tiers to the PAG
endpoints (causal-learn's FCI does). So GFCI runs confounder-conditioned but leaves cross-tier edges
undetermined.

**The resolution.** GFCI contributes **multi-algorithm adjacency**; **directed claims come from FCI**.
Both are reported, and the limitation is written down rather than hidden
([L10](LIMITATIONS.md#l10-gfci-orients-the-skeleton-only)).

*Debugging note worth keeping:* Tetrad's `Knowledge` object silently binds to nothing when variable
names contain `:` or `-` — which every genomic coordinate does (`chr5:140638860-140639949`). It fails
without erroring. Fixed by renaming variables to safe tokens before the search and mapping back after.

📁 `CausalDiscovery/scripts/run_gfci_tetrad.py`

---

## 8. Why library depth as root nodes, plus a mechanistic prior

**The problem.** Two of them, actually.

*(a) A technical confounder.* Sequencing depth drives both peak signal and expression. Two variables
that both scale with depth look dependent even when nothing biological connects them.

*(b) Unresolved orientation.* FCI returned mostly ambiguous marks on peak→gene edges. Observational data
alone often cannot orient an edge — but we have real mechanistic knowledge the algorithm doesn't.

**What we do.**
1. Add `conf__rna_depth` and `conf__chrom_depth` as **root nodes** — variables allowed to cause others
   but not be caused. Now depth-driven covariation is attributed to depth instead of fabricating a
   chromatin→gene link.
2. Add a **tiered background prior**: confounders → peaks → expression. Chromatin state is upstream of
   transcription; expression cannot cause a histone mark measured in a different cell. This forbids
   edge directions we know are impossible.

**Why this is legitimate and not circularity.** The prior encodes *mechanism* (chromatin precedes
transcription), not *the answer* (which peak regulates which gene). Validation evidence is kept strictly
independent of it — no database informs the graph.

**Did it work?** Both aims. Orientation: **19/19 peak→gene edges directed** across the panel, up from a
mix of ambiguous marks. Confounding: PTPRJ dropped from 4 peak→gene edges to 2 — depth-driven
covariation reattributed to the roots. Importantly, **no validated edge was lost**.

📁 `run_pc_causallearn.py --confounders depth` ·
📊 [expanded_panel_confounder_results.md](../Evaluation/results/expanded_panel_confounder_results.md)

---

## 9. Why donor and cell-cycle are *not* in the model

Both were investigated as candidate confounders and both were rejected — for different reasons.

**Donor: not available.** Not a modelling choice — a data fact. The RNA cells were never hashtagged, and
the chromatin HTO donors belong to a different experiment than the RNA
([data_sources.md](../Data/results/data_sources.md)). No donor structure is shared across modalities, so
there is nothing to condition on. Separately, even on the chromatin side metacells average donor out:
donor purity per metacell ≈ 0.27 against a random-mix floor of 0.23, and donor is statistically
independent of metacell (χ², p = 0.43). **Donor confounding is handled by aggregation, not conditioning.**

**Cell-cycle: irrelevant here.** Monocytes are ~1–2% cycling; S/G2M scores are near zero and don't
correlate with the CD14 peaks. Adding it would cost a variable and buy nothing.

**Why this matters for the main claim.** If donor and batch are already neutralised by aggregation, then
the peak↔peak `<->` edges that survive depth-conditioning are **not technical artifacts** — which is
what licenses reading them as biological. That argument is load-bearing for
[MISSION](MISSION.md) §2, and it's why we tested it four more ways (§12–14).

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

An edge counts as validated on ≥1 axis; ≥2 agreeing is the gold standard.

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

📁 `Evaluation/scripts/build_{re2g,pchic,eqtl}_gene_link_support.py`

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

**Known gap.** Matching is on distance only; **degree-matching is not implemented**
([L7](LIMITATIONS.md#l7-elementelement-validation-has-no-accepted-null)).

📁 `Evaluation/scripts/peak_peak_validator.py`

---

## 13. Why we benchmark against co-accessibility

**The problem.** "Our method separates direct from latent-confounded element pairs" is only interesting
against the thing people currently use. Co-accessibility (Cicero and successors) is that thing.

**Why it's the right comparison.** Those papers **explicitly concede** they cannot distinguish a direct
element–element link from co-dependence via a shared hidden driver. That's precisely the gap our method
addresses, so the comparison is on their stated terms, not a strawman.

**What we found.** On the same data, co-accessibility scores bidirected (latent) pairs at 0.238 and other
adjacent pairs at 0.255 — statistically indistinguishable. It cannot separate them. **53% of the PAG's
element↔element adjacencies are bidirected**, so co-accessibility would report all 126 latent pairs as
regulatory links.

**Where we're careful.** Our baseline is |Spearman|, not a faithful Cicero — a distance-penalized
graphical lasso would remove many of the indirect pairs. So we restrict the headline claim to what is
theoretically guaranteed: **graphical lasso removes indirect paths through *observed* variables but
cannot remove a shared *latent* driver.** The latent class is the unassailable part
([L8](LIMITATIONS.md#l8-the-co-accessibility-contrast-uses-a-spearman-baseline)).

📁 `Evaluation/scripts/coaccess_contrast.py`

---

## 14. Why the global-activity probe leaves the locus out

**The question.** Having ruled out shared TF and shared domain, the remaining candidate was a global
chromatin-activity/cell-state latent. Test: condition on it and see whether the `<->` edges disappear.

**The trap — and why the obvious version backfires.** The intuitive proxy for "how active is this locus"
is the mean of the locus's own peaks. **That is a collider.** It's a *descendant* of the peaks, not a
cause. Conditioning on a common effect *induces* dependence between its parents — so this doesn't remove
edges, it manufactures them. We learned this the hard way: an earlier prototype regressed out
self-derived principal components and edge counts *inflated* (8→18, 6→10, 9→17, 1→9).

**The fix.** Build the proxy from peaks **outside** the gene's own locus — the mean z-scored accessibility
across all *other* panel genes' peaks. That's a plausible common cause of the locus and, by construction,
not a descendant of it. Collider-safe.

**Result.** Bidirected edges 126 → 129 (unchanged), validated peak→gene 11 → 11. Genome-wide activity is
**not** the driver.

📁 `CausalDiscovery/scripts/build_global_activity.py`

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

📁 `CausalDiscovery/configs/gene_panels/` · 📊 [target_gene_selection.md](../CausalDiscovery/results/target_gene_selection.md)

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

**What we learned.** More data and more reference material did not produce more validated causal edges.
**Edge recovery is limited by per-gene effect size**, not by sample size. That's a useful negative — it
means "just add cells" is not the lever, and it's part of why *scale* is now about more **genes**, not
more cells per gene.

**Decision.** Monocyte stays primary; the T-cell suite is retained for the cross-cell-type invariance
(ICP) test, which is now the project's invariance axis.

📊 [celltype_fair_comparison.md](../Evaluation/results/celltype_fair_comparison.md)

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
