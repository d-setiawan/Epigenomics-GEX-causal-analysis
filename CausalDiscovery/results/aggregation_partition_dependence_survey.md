<!-- PROVENANCE: compiled 2026-07-21 by an automated literature-search agent (web + paper-lookup),
then reviewed and placed here. DOIs/arXiv IDs were agent-verified; items flagged [unverified] and the
pre-DOI monographs (Openshaw 1984, Theil 1954, etc.) MUST be re-checked before formal citation. Treat
this as a research input that informs strategy, not a settled result. -->

> **↑ [ROADMAP](../../docs/ROADMAP.md) · [partition_stability_findings.md](../../Evaluation/results/partition_stability_findings.md) (our empirical result this survey contextualizes) · [L11](../../docs/LIMITATIONS.md#l11-edge-identities-are-partition-dependent)**
>
> **Why this exists.** We found that re-clustering the cells with a different seed changes which
> validated edges FCI recovers (9/11 flip), though aggregate counts are stable. This survey checked
> whether that is a known problem, how other fields handle aggregation-dependent results, and whether
> causal discovery can avoid aggregation. **Bottom line: the specific failure is largely
> uncharacterized in the literature, it is a textbook Modifiable Areal Unit Problem, and bootstrapping
> over metacell *partitions* appears genuinely undone — a defensible novel contribution.**

# Aggregation-Partition Dependence in Single-Cell Causal Discovery: A Cited Survey

*Prepared for a methods paper on FCI/PAG causal discovery over chromatin-region covariates + gene expression, where ~single cells are aggregated into ~133 metacells (disjoint Leiden groups over a latent embedding), and re-clustering with a different seed changes which validated edges are recovered (9/11 do not survive), even though aggregate counts are stable.*

*Compiled 2026-07-21. Every non-obvious claim carries an inline citation with DOI / arXiv ID / stable identifier. Items marked **[unverified]** could not be pinned to a stable identifier and are flagged as such.*

---

## Executive summary

- **Partition-dependence is real and, in a narrow QC sense, recognized — but the specific failure you observed (edge set flips under re-seeding while marginal/aggregate counts stay stable) is essentially uncharacterized in the metacell literature.** The one paper that squarely studies "the partition changes the answer," mcRigor (Liu & Li 2025, *Nat Commun* 16:8602, doi:10.1038/s41467-025-63626-5), attributes spurious downstream findings to *heterogeneous ("dubious") metacells*, not to *seed/zoning variability among equally-valid partitions*, and explicitly does **not** analyze random-seed stability.

- **The metacell field's own review concedes seed sensitivity but claims global robustness** — "methods reliant on stochastic processes may lead to varying metacell outcomes based on different initializations or random seeds," yet "the global representation … is in general robust to small variations in metacell definitions" (Bilous et al. 2024, *Mol Syst Biol* 20, doi:10.1038/s44320-024-00045-6). Your result is the important counter-case: *fine-grained conditional-independence structure is not a "global representation" and is not protected by that robustness.*

- **Most metacell methods produce disjoint groups whose boundaries are stochastic** (MetaCell, Metacell-2, SuperCell, MetaQ), so re-seeding genuinely redraws the sample rows FCI sees; kNN-pooling/neighbor-averaging instead produces *overlapping* pooled profiles, and pseudobulk uses *externally defined* (cell-type/replicate) groups — a spectrum from "arbitrary data-driven zonation" to "substantively defined zonation."

- **This is a textbook instance of three deep cross-field problems.** It is the **Modifiable Areal Unit Problem** (MAUP: scale effect + zoning effect; Openshaw 1984; Fotheringham & Wong 1991, doi:10.1068/a231025), the **ecological-inference / ecological-fallacy** problem (aggregate correlations ≠ individual correlations; Robinson 1950, doi:10.2307/2087176; King 1997), and **aggregation bias** in econometrics (Theil 1954; Grunfeld & Griliches 1960, doi:10.2307/1926089). Metacell partition choice = the zoning effect. Each field's headline lesson is: *report sensitivity across zonations; do not treat one aggregation as ground truth.*

- **Aggregation is not strictly necessary.** A now-credible line of work runs constraint- or score-based causal discovery **directly on sparse, zero-inflated single-cell counts**: test-wise deletion under the Causal Dropout Model (Dai et al. 2024, ICLR oral, arXiv:2403.15500), zero-inflated score-based DAG learning (Choi & Ni 2023, JMLR 24:200; Sato et al. 2025 ZICO, arXiv:2512.16233), and Hurdle directed graphical models (Yu, Drton & Shojaie 2023, *CLeaR* / PMLR 213:27–67). These sidestep the partition-choice problem entirely, at the cost of parametric/mechanistic assumptions.

- **But constraint-based discovery needs effective sample size that grows with conditioning-set size**, which is exactly why you aggregated. CI-test power collapses as conditioning sets grow (Kocaoglu 2023, NeurIPS; Kalisch & Bühlmann 2007, JMLR 8:613–636), and for discrete tests the rule of thumb needs ≳5–10 samples per contingency-table cell — roughly exponential in conditioning-set size. With ~133 rows and ~41 variables, FCI is already sample-starved, which amplifies partition variance.

- **The standard fix from statistics is subsampling/ensembling with edge-frequency reporting** — stability selection (Meinshausen & Bühlmann 2010, doi:10.1111/j.1467-9868.2010.00740.x), bootstrap confidence on structure features (Friedman, Goldszmidt & Wyner 1999, UAI, arXiv:1301.6695), and consensus clustering (Kiselev et al. 2017 SC3, doi:10.1038/nmeth.4236; von Luxburg 2010 clustering stability, doi:10.1561/2200000008). The natural adaptation here is to **bootstrap over the partition itself** — resample metacell assignments across K seeds/resolutions and report each edge's recovery frequency.

- **This partition-bootstrap appears to be genuinely undone in the single-cell causal-discovery setting** (see Gaps). Existing bootstrap-causal-discovery resamples *rows*; existing metacell QC filters *bad metacells within one partition*; consensus clustering ensembles *cluster labels*, not *downstream causal edges*. Reporting edge frequency over an ensemble of metacell partitions is a defensible, novel-in-context contribution — and reframes your "9/11 edges don't survive" from a bug into a headline stability result.

- **Bottom line for positioning.** Do not claim edges from a single partition. Either (a) move to a partition-free count-based discovery method, or (b) keep aggregation but report **edge-recovery frequency across an ensemble of K partitions** (seeds × resolutions), keep only edges above a stability threshold, and show sensitivity to graining level. Frame (b) as importing MAUP/ecological-inference/stability-selection rigor into single-cell causal discovery.

---

## Q1. Catalog of single-cell aggregation methods used before downstream statistical / graphical / causal inference

For each method: how groups are formed, whether they are **disjoint**, and stated stability/reproducibility properties. A useful unifying reference is the review by **Bilous, Hérault, Gabriel, Teleman & Gfeller (2024), "Building and analyzing metacells in single-cell genomics data," *Molecular Systems Biology* 20, doi:10.1038/s44320-024-00045-6** (bioRxiv 2024.02.04.578815), which taxonomizes these tools and is quoted on stability in Q2.

| Method | Grouping mechanism | Disjoint? | Stated stability/reproducibility properties |
|---|---|---|---|
| **MetaCell** (Baran et al. 2019, *Genome Biology* 20:206, doi:10.1186/s13059-019-1812-2) | Partitions a kNN similarity graph into small dense subgraphs via a resampling/graph-partition (bootstrap-and-cover of the kNN graph); each metacell is a "group of profiles that could have been resampled from the same cell." | **Yes** — disjoint, homogeneous, compact. | Uses bootstrap resampling of the kNN graph to stabilize the partition (co-occurrence over resamples); homogeneity is a stated *goal* but there is no formal seed-invariance guarantee. mcRigor (Q2) later ranks it among the more reliable methods. |
| **Metacell-2 / MC2** (Ben-Kiki, Bercovich, Lifshitz & Tanay 2022, *Genome Biology* 23:100, doi:10.1186/s13059-022-02667-1) | Recursive **divide-and-conquer** graph partition of the kNN graph; scalable rewrite of MetaCell with outlier detection. | **Yes** — disjoint. | Emphasizes scalability and outlier/rare-type detection; no explicit partition-stability guarantee. mcRigor finds it and SuperCell somewhat less reliable than MetaCell/SEACells. |
| **SEACells** (Persad et al. 2023, *Nature Biotechnology* 41:1746–1757, doi:10.1038/s41587-023-01716-9) | **Archetypal analysis** on a low-dimensional embedding / kernel: learns archetypes and a *soft* cell-to-archetype assignment, then hard-assigns each cell to its dominant archetype. Works for RNA and ATAC. | Soft assignment is **overlapping**; the delivered metacells are **disjoint** after hard assignment. | Claims comprehensive, compact, well-separated metacells; benchmarked as top-tier by mcRigor. Archetype initialization (e.g. via k-means++/waypoint) is stochastic, so archetype identity can vary across seeds. |
| **SuperCell** (Bilous et al. 2022, *BMC Bioinformatics* 23:336, doi:10.1186/s12859-022-04861-1) | Builds a kNN graph on PCA space, then **walktrap community merging** of highly similar cells down to a user-set **graining level γ** (= n_cells / n_metacells); averages expression within group. | **Yes** — disjoint. | Benchmarked to "preserve or improve" downstream tasks (clustering, DE, gene correlation, RNA velocity, integration). γ is an explicit knob; results are γ-dependent. mcRigor flags it as producing more heterogeneous metacells at high γ. |
| **MetaQ** (Li et al. 2025, *Nature Communications* 16:1205, doi:10.1038/s41467-025-56424-6) | **Deep vector quantization**: an autoencoder learns cell embeddings and a learnable discrete codebook; each cell is assigned to its nearest codebook entry (each entry = one metacell reconstructing its cells). Linear time, constant memory. | **Yes** — disjoint (nearest-codebook assignment). | Strong scaling; matches/surpasses prior methods on DE, trajectory, integration. Being autoencoder-based it is initialization/seed-dependent; no seed-stability guarantee stated. |
| **Simple kNN-pooling / neighbor-averaging** (used by e.g. Cicero: Pliner et al. 2018, *Molecular Cell* 71:858–871, doi:10.1016/j.molcel.2018.06.044; also MAGIC-style smoothing) | For each cell (or each of many random anchor cells) average its k nearest neighbors to form a smoothed profile; Cicero aggregates similar cells into overlapping "windows" before computing co-accessibility. | **No** — pooled profiles **overlap** (a cell contributes to many pools); rows are correlated by construction. | Reduces sparsity but *induces spurious correlation* between overlapping pools — a known hazard for correlation/network inference; not a set of independent samples. |
| **Pseudobulk** (aggregate by cell type × biological replicate; Squair et al. 2021, *Nature Communications* 12:5692, doi:10.1038/s41467-021-25960-2) | Sum/mean of counts within *externally defined* groups (cell type/cluster × sample/replicate). | **Yes** — disjoint, but boundaries are **substantively defined**, not data-driven at fine grain. | The most reproducible aggregation *because groups are defined by known biology/replicate*. Squair et al. show that **aggregating random pseudo-replicates inflates false discoveries and biases toward highly expressed genes**, whereas replicate-respecting pseudobulk controls FDR — a direct warning that *how you form the groups* changes the statistical answer. |

**Cross-cutting observations.** (i) All the data-driven metacell methods (MetaCell, MC2, SuperCell, MetaQ, and SEACells' archetype step) rely on stochastic optimization or graph partitioning, so **the boundary between disjoint groups is a random variable across seeds**. (ii) kNN-pooling trades disjointness for smoothness and thereby *manufactures* correlation — the worst case for CI-based discovery. (iii) Pseudobulk is the most reproducible precisely because it *does not* choose fine-grained groups from the data; its groups come from external labels. Your pipeline sits at the least-reproducible end: fine-grained, data-driven, stochastic, disjoint.

---

## Q2. How does the choice of aggregation/partition affect downstream inference?

**Is "results depend on how you draw the groups" recognized?** Partially, and only recently — and mostly as a *quality* problem (bad metacells) rather than a *variance-over-equally-valid-partitions* problem (your case).

**mcRigor — the metacell-QC / rigor paper (Liu & Li 2025, *Nature Communications* 16:8602, doi:10.1038/s41467-025-63626-5; bioRxiv 2024.10.30.621093).** This is the most on-point work. Key verified claims:
- It states plainly that partition choice is consequential: *"These methods can produce different metacell partitions, which are also influenced by the hyperparameters they employ."*
- It ties partition quality to *false downstream discoveries*: *"Including dubious metacells in correlation estimation often leads to spurious co-expression findings,"* and, for a COVID-19 B-cell example, *"This artifact was absent at single-cell resolution and was induced solely by the presence of dubious metacells."* Removing dubious metacells changed gene-module enrichment p-values by many orders of magnitude (e.g. from a non-significant 0.55 to 3.7e-31 for an adaptive-immune module) and changed enhancer–gene association scores (GATA2 top-peak association 0.6904 → 0.7606; 5536 → 5551 significant gene–peak pairs).
- Its statistic is **mcDiv**, a Frobenius-norm deviation of the within-metacell feature-correlation matrix from identity, with a **double-permutation** null (within-cell permutation preserving library size, then within-feature permutation); metacells above the 95%-quantile null threshold are flagged "dubious."
- **Crucially for your question: mcRigor does *not* analyze stability across random seeds or across equally-valid partitions.** Its only stochastic element is deliberate permutation. So the specific phenomenon "re-seeding flips the edge set" is *outside* what mcRigor characterizes — it addresses "is this metacell internally homogeneous?", not "would a different valid partition give different edges?"

**The metacell review's robustness claim — and why it doesn't cover you.** Bilous et al. (2024, *Mol Syst Biol* 20, doi:10.1038/s44320-024-00045-6) acknowledge seed sensitivity but argue it washes out globally: the main tools "internally fix random seeds," yet "methods reliant on stochastic processes may lead to varying metacell outcomes based on different initializations or random seeds," while "the global representation of single-cell data is in general robust to small variations in metacell definitions." The load-bearing gap: **a PAG edge from a fine-grained CI test is not a "global representation."** Global structure (UMAP shape, major clusters, mean expression) can be seed-robust while the fine partial-correlation structure that FCI keys on is not — which is exactly your observation that aggregate counts are stable but 9/11 validated edges are not.

**Aggregation choice specifically changes *network/graph* inference.** Zhang & Zhu (2025), "Library size-stabilized metacells construction enhances co-expression network analysis in single-cell data," *PLOS Computational Biology*, doi:10.1371/journal.pcbi.1013697, show that how metacells are constructed (and library-size heterogeneity within them) materially changes the recovered co-expression network, and propose a construction that stabilizes it — direct evidence that the partition drives edge recovery in graphical models, not just marginal statistics.

**Robustness/stability analyses and bootstrapping *over* partitions.** Beyond mcRigor's per-metacell QC, I found **no** metacell paper that (a) runs a downstream graphical/causal model across an ensemble of independently-seeded partitions and reports the distribution of recovered edges, or (b) bootstraps *the partition* to attach stability/confidence to downstream edges. Consensus clustering (SC3; Kiselev et al. 2017, doi:10.1038/nmeth.4236) and the clustering-stability literature (von Luxburg 2010, doi:10.1561/2200000008; Ben-Hur, Elisseeff & Guyon 2002, PSB) ensemble *cluster labels* for a stable clustering, but stop at the clustering — they do not propagate partition uncertainty into a downstream causal graph. This is the crux of the Gaps section.

---

## Q3. Cross-field analogs — "a statistical result that changes with how you aggregate units"

This is one of the oldest recognized problems in quantitative science. Three literatures characterize it, and all converge on the same mitigation logic. **Metacell partition choice is the direct analog of the "zoning effect" in each.**

### 3.1 Modifiable Areal Unit Problem (MAUP) — geography / spatial statistics
The MAUP is the sensitivity of any statistic computed on areal (zonal) data to the *arbitrary* choice of spatial units, decomposed into two effects:
- **Scale effect** — results change as the same data are aggregated to coarser/finer units. *(Analog: your graining level / number of metacells, γ.)*
- **Zoning (aggregation) effect** — results change when units of the *same* number/scale are drawn with *different boundaries*. *(Analog: re-seeding Leiden gives a different set of ~133 metacells at the same resolution.)*

Foundational statements: Openshaw & Taylor (1979), "A million or so correlation coefficients: three experiments on the modifiable areal unit problem" (in *Statistical Applications in the Spatial Sciences*, Pion, pp. 127–144; no DOI — pre-DOI chapter), which coined "MAUP" and showed correlations swing wildly (even changing sign) across zonations; and Openshaw (1984), *The Modifiable Areal Unit Problem*, CATMOG 38 (Geo Books; ISSN 0306-6142; monograph, no DOI). Fotheringham & Wong (1991), "The modifiable areal unit problem in multivariate statistical analysis," *Environment and Planning A* 23(7):1025–1044, doi:10.1068/a231025, show the effect is **larger and less predictable in multivariate/regression settings than in bivariate correlation** — directly relevant, because FCI runs many multivariate partial-correlation tests. A recent retrospective, Goodchild (2022), "The Openshaw effect," *Int. J. Geographical Information Science* 36(9), doi:10.1080/13658816.2022.2102637, frames MAUP as a fundamental property of areal data, not a nuisance.

**Mitigations:** explicit **sensitivity analysis across many alternative zonations and scales** (Swift, Liu & Uber 2014, "MAUP sensitivity analysis of ecological bias in health studies," *GeoJournal* 79:137–153, doi:10.1007/s10708-013-9504-z); **multilevel/hierarchical modeling** that separates individual- from area-level variance; designing zones around the process rather than convenience; and reporting results at multiple scales with zonation treated as a source of uncertainty rather than fixed.

### 3.2 Ecological inference / ecological fallacy — sociology, epidemiology, political science
Robinson (1950), "Ecological correlations and the behavior of individuals," *American Sociological Review* 15(3):351–357, doi:10.2307/2087176, established that correlations/regression slopes computed on aggregated groups can differ sharply — even in sign — from the corresponding individual-level relationships; assuming otherwise is the **ecological fallacy** (the mirror error, inferring group facts from individuals, is the atomistic fallacy). The problem is fundamentally information loss: within-group behavior is unidentified. *(Analog: an edge that appears among 133 metacell means need not reflect a within-cell mechanism; the aggregation can create or destroy the dependence FCI tests.)*

**Mitigations:** deterministic **bounds** on the unobserved individual quantities from accounting identities (Duncan & Davis 1953, "An alternative to ecological correlation," *ASR* 18(6):665–666, doi:10.2307/2088122); **King's EI model** fusing bounds with a hierarchical statistical model (King 1997, *A Solution to the Ecological Inference Problem*, Princeton Univ. Press; reissue doi:10.1515/9781400849208); and the R×C / Bayesian generalizations in King, Rosen & Tanner (2004), *Ecological Inference: New Methodological Strategies*, Cambridge Univ. Press, doi:10.1017/cbo9780511510595. The consensus caution: EI estimates rest on **untestable homogeneity assumptions**, so bounds and sensitivity checks remain essential — a direct warning for reading metacell-level edges as cell-level mechanism.

### 3.3 Aggregation bias — econometrics
Theil (1954), *Linear Aggregation of Economic Relations* (North-Holland; monograph, no DOI), is the canonical result: a regression on aggregated (macro) data generally does **not** recover the average of the underlying micro-relations — aggregate coefficients are weighted mixtures of heterogeneous micro-coefficients, so they **change with the aggregation scheme**; exact ("perfect") aggregation holds only under restrictive homogeneity conditions. A parallel literature treats **grouping as a deliberate tool**: Wald (1940), "The fitting of straight lines if both variables are subject to error," *Annals of Mathematical Statistics* 11(3):284–300, doi:10.1214/aoms/1177731868, and Durbin (1954), "Errors in variables," *Rev. Int. Statistical Institute* 22(1/3):23–32, doi:10.2307/1401917, show grouping/ranking observations and regressing on group means acts as an **instrumental variable** that *reduces* attenuation from measurement (errors-in-variables) bias; Cramer (1964), "Efficient grouping, regression and correlation in Engel curve analysis," *JASA* 59(305):233–250, doi:10.1080/01621459.1964.10480714, makes explicit how grouping alters coefficients and efficiency.

**The key trade-off statement:** Grunfeld & Griliches (1960), "Is aggregation necessarily bad?", *Review of Economics and Statistics* 42(1):1–13, doi:10.2307/1926089 — aggregation can *lower* some biases (measurement error, misspecification) even as it *introduces* aggregation bias, so it is "not necessarily bad." **This is exactly your situation:** aggregating sparse cells into metacells reduces dropout/measurement noise (why you did it) but introduces aggregation/zoning bias (why edges flip). The econometric prescription — prefer micro-data when heterogeneity is present, test homogeneity, use grouping/IV *explicitly and validly* when measurement error dominates, and **report sensitivity to the level of aggregation** — maps cleanly onto "either run count-based discovery on single cells, or report edge stability across graining levels and partitions."

**The unifying analogy.** Metacell construction = choosing an areal partition of cell-space. Graining level γ = the MAUP scale effect. Re-seeding at fixed γ = the MAUP zoning effect. Reading a metacell-level PAG edge as a cell-level mechanism = the ecological fallacy. The variance you see across seeds = aggregation bias made visible. No field treats a single arbitrary partition as authoritative; all recommend sensitivity analysis across partitions/scales and, where possible, working closer to the micro-unit.

---

## Q4. Can constraint-/score-based causal discovery run WITHOUT aggregation — directly on sparse single cells?

**Yes — this is now a small but credible literature.** It removes the partition-choice problem entirely, trading it for parametric/mechanistic assumptions about the count/dropout process.

**Dropout-aware constraint-based discovery (the strongest fit for FCI).** Dai, Ng, Luo, Spirtes, Stojanov & Zhang (2024), "Gene Regulatory Network Inference in the Presence of Dropouts: a Causal View," **ICLR 2024 (oral)**, arXiv:2403.15500 (code: github.com/MarkDana/scRNA-Causal-Dropout). They define the **Causal Dropout Model** (a graphical model of the zero-generating mechanism, mixing biological non-expression and technical dropout) and prove that after **test-wise deletion** — dropping, *for each CI test*, the samples that are zero in the *conditioning* variables — the CI relations are asymptotically identical to those in dropout-free data. This makes PC/FCI-family and score-based learners correct **on raw single cells**, with no imputation and no aggregation; existing statistical dropout corrections are shown to be special cases. A same-group follow-up handling selection bias and latent confounders (Luo et al. 2025, arXiv:2501.10124) is on arXiv (venue **[unverified]**). *Caveat for your pipeline: test-wise deletion shrinks effective sample size per test — it spends the very resource FCI is short on — so on 40+ covariates it may be as sample-starved as your metacell run.*

**Score-based discovery modeling counts directly (no CI tests, no aggregation).**
- **ZiG-DAGs** — Choi & Ni (2023), "Model-Based Causal Discovery for Zero-Inflated Count Data," *JMLR* 24, art. 200. Score-based DAG search over zero-inflated **generalized-hypergeometric** conditionals (encompassing Poisson, negative-binomial, hyper-Poisson, beta-NB); operates on **raw single-cell counts**, and proves DAG **identifiability without a faithfulness assumption** — attractive when faithfulness is dubious in sparse data. Validated on a TGF-β network from CRC scRNA-seq (~40% zeros).
- **ZICO** — Sato, Scutari, Kawano, Yamaguchi & Imoto (2025), "DAG Learning from Zero-Inflated Count Data Using Continuous Optimization," arXiv:2512.16233. A **NOTEARS/DAGMA-style** continuous-optimization learner where each node is a zero-inflated GLM under a differentiable acyclicity constraint — directly relevant given your DAGMA line, and it targets counts without metacells.
- **Hurdle directed graphical models** — Yu, Drton & Shojaie (2023), "Directed Graphical Models and Causal Discovery for Zero-Inflated Data," ***CLeaR 2023*, PMLR 213:27–67** (also PMC11257027). Hurdle conditionals (point mass at 0 + conditional Gaussian with parent indicators); exact DAG identifiable under a "strong Hurdle polynomial" condition where Gaussian DAGs are not. *Note: this is a CLeaR paper — same venue as your target — and models continuous-expression-with-excess-zeros rather than discrete counts.*

**Framed-causal GRN methods (weaker guarantees, need ordering/time).** SCODE (Matsumoto et al. 2017, *Bioinformatics*, doi:10.1093/bioinformatics/btx194) infers ODE-based GRNs from **pseudotime-ordered** cells; SINCERITIES (Papili Gao et al. 2018, *Bioinformatics* 34:258–266, doi:10.1093/bioinformatics/btx575) uses a Granger-causal notion over **time-stamped** data. Both need temporal/ordering information and give GRN rather than PAG-level guarantees. General context: Glymour, Zhang & Spirtes (2019), "Review of causal discovery methods based on graphical models," *Frontiers in Genetics* 10:524, doi:10.3389/fgene.2019.00524.

**The single-cell-specific constraint-based platform.** CausalCell — Wen et al. (2023), "Applying causal discovery to single-cell analyses using CausalCell," *eLife*, doi:10.7554/eLife.81464 (PMID 37129360) — benchmarks 9 feature-selection × 9 CI-test combinations on scRNA-seq and concludes the **constraint-based PC algorithm with kernel-based CI tests works best in most cases**. It runs on normalized single-cell expression *within a cell type* (not pseudobulk) but does **no explicit dropout modeling**. On sample size it reports 200/400/600/800 cells: more cells → more inferred edges, but **high-significance edges are stable** and **no minimum-cell-count rule is given** — a useful, if soft, data point that stability improves with n.

**How much effective sample size does PC/FCI actually need?** The literature is consistent that constraint-based discovery is sample-hungry in a way that worsens with conditioning-set size:
- Kalisch & Bühlmann (2007), "Estimating high-dimensional directed acyclic graphs with the PC-algorithm," *JMLR* 8:613–636 (JMLR, no DOI — cite by volume/pages), prove uniform consistency of PC (Fisher-z partial-correlation test) with the number of variables growing as p = O(n^a) for any a>0 — **but only under sparsity**, i.e. neighborhoods (hence max conditioning-set size / degree) of lower order than n. Practically: reliability degrades as conditioning sets grow relative to n.
- Kocaoglu (2023), "Characterization and Learning of Causal Graphs with Small Conditioning Sets," **NeurIPS 2023**, states plainly that **CI tests rapidly lose statistical power as the conditioning set grows**, and proposes k-PC (capping conditioning-set size) for robustness in small samples.
- For discrete G²/χ² CI tests, the standard rule of thumb is to trust the test only when the average count **per contingency-table cell** is ≳5 (often 10); otherwise declare insufficient data and default to independence — making required sample size grow **roughly exponentially in conditioning-set size** (a widely-used convention from the Spirtes–Glymour–Scheines / Tetrad tradition, *Causation, Prediction, and Search*; **[unverified]** as a single pinned theorem — it is convention, not one citable result).

**Implication.** With ~133 rows and ~41 variables, FCI is *already* at the edge of reliability, and any CI test conditioning on more than a handful of covariates is underpowered. This is (a) why aggregation was attractive, and (b) why the recovered edges are variance-dominated and flip under re-partitioning. Count-based score methods (ZiG-DAGs, ZICO, Hurdle) that use all cells sidestep both the CI-power ceiling *and* the partition-choice problem — the most promising alternative for you, though at the price of stronger distributional assumptions and (for FCI's latent-confounder handling) losing the PAG/latent-aware guarantee unless a latent-aware score variant is used.

---

## Q5. Practical recommendations — making an aggregation-based pipeline robust to partition choice

Synthesized from Q1–Q4. Ordered from "keep aggregation but make it honest" to "remove the dependence."

1. **Never report edges from a single partition. Bootstrap over the partition and report edge-recovery frequency.** Generate an ensemble of K partitions (K ≥ 50–100) by varying the Leiden seed (zoning effect) and, ideally, also the resolution/target metacell count (scale effect); run FCI on each; report, per edge, the fraction of partitions in which it (and its orientation/PAG mark) is recovered. Keep only edges above a stability threshold. This is the direct import of **stability selection** (Meinshausen & Bühlmann 2010, doi:10.1111/j.1467-9868.2010.00740.x) and **bootstrap confidence on structure features** (Friedman, Goldszmidt & Wyner 1999, UAI, arXiv:1301.6695) into the metacell setting. Your "9/11 validated edges don't survive re-seeding" then becomes a *reported stability statistic*, and the 2/11 that do survive become your defensible findings.

2. **Do a MAUP-style sensitivity analysis over graining level.** Sweep the number of metacells (e.g. 80–300) and plot edge recovery vs. γ. Edges stable across both seed *and* scale are the credible ones; edges that appear only in a narrow γ window are scale artifacts (the MAUP scale effect; Fotheringham & Wong 1991, doi:10.1068/a231025; Swift et al. 2014, doi:10.1007/s10708-013-9504-z).

3. **Use consensus partitioning to reduce (not hide) seed variance.** Consensus/ensemble clustering yields a more reproducible base partition (SC3; Kiselev et al. 2017, doi:10.1038/nmeth.4236; clustering-stability theory: von Luxburg 2010, doi:10.1561/2200000008; Ben-Hur et al. 2002, PSB). But consensus gives *one* stable partition — it does **not** substitute for reporting downstream edge stability (recommendation 1); use both.

4. **Filter dubious metacells before discovery.** Run mcRigor (Liu & Li 2025, doi:10.1038/s41467-025-63626-5) on each partition and drop heterogeneous metacells; this removes a known source of *spurious* edges. Complement with a construction that stabilizes the variance/library-size structure CI tests rely on (Zhang & Zhu 2025, doi:10.1371/journal.pcbi.1013697). mcRigor also gives a principled way to choose method/hyperparameters per dataset.

5. **Choose aggregation to preserve the covariance structure FCI keys on.** CI tests read partial correlations, so the aggregation should preserve the *within-population variance and covariance*, not just the mean (which is what "aggregate counts are stable" reports). Avoid overlapping kNN-pooling for CI-based discovery — it manufactures correlation between rows (Cicero-style pooling, Pliner et al. 2018, doi:10.1016/j.molcel.2018.06.044, is designed for co-accessibility, not for independent-sample CI tests). Prefer disjoint metacells, and verify that per-metacell variance is not collapsed to noise.

6. **Respect biological replicate structure where it exists (pseudobulk lesson).** If your ~cells span donors/samples, forming groups that cross replicate boundaries inflates false positives (Squair et al. 2021, doi:10.1038/s41467-021-25960-2). Blocking metacells within replicate (or including replicate as a context variable) guards against this.

7. **Cross-validate against a partition-free method.** Run a count-based / dropout-aware discovery method on the raw cells — ZiG-DAGs (Choi & Ni 2023, JMLR 24:200), ZICO (Sato et al. 2025, arXiv:2512.16233), Hurdle DAGs (Yu, Drton & Shojaie 2023, CLeaR/PMLR 213:27–67), or FCI with test-wise deletion (Dai et al. 2024, arXiv:2403.15500) — and keep edges that agree with your partition-ensemble consensus. Agreement across a fundamentally different pipeline (no aggregation) is the strongest evidence an edge is real, and directly answers a reviewer asking "is this an aggregation artifact?"

8. **Report honestly and frame the aggregation trade-off.** State the Grunfeld–Griliches (1960, doi:10.2307/1926089) trade-off explicitly: aggregation buys you sample-level signal-to-noise at the cost of aggregation/zoning bias; you manage the latter by reporting stability across partitions and scales rather than pretending a single partition is ground truth.

---

## Key references

**Aggregation methods (Q1)**
- Baran Y, et al. (2019). MetaCell: analysis of single-cell RNA-seq data using K-nn graph partitions. *Genome Biology* 20:206. doi:10.1186/s13059-019-1812-2.
- Ben-Kiki O, Bercovich A, Lifshitz A, Tanay A (2022). Metacell-2: a divide-and-conquer metacell algorithm for scalable scRNA-seq analysis. *Genome Biology* 23:100. doi:10.1186/s13059-022-02667-1.
- Persad S, et al. (2023). SEACells infers transcriptional and epigenomic cellular states from single-cell genomics data. *Nature Biotechnology* 41:1746–1757. doi:10.1038/s41587-023-01716-9.
- Bilous M, et al. (2022). Metacells untangle large and complex single-cell transcriptome networks (SuperCell). *BMC Bioinformatics* 23:336. doi:10.1186/s12859-022-04861-1.
- Li Y, et al. (2025). MetaQ: fast, scalable and accurate metacell inference via single-cell quantization. *Nature Communications* 16:1205. doi:10.1038/s41467-025-56424-6.
- Pliner HA, et al. (2018). Cicero predicts cis-regulatory DNA interactions from single-cell chromatin accessibility data. *Molecular Cell* 71:858–871. doi:10.1016/j.molcel.2018.06.044.
- Squair JW, et al. (2021). Confronting false discoveries in single-cell differential expression (pseudobulk). *Nature Communications* 12:5692. doi:10.1038/s41467-021-25960-2.
- Bilous M, Hérault L, Gabriel A, Teleman M, Gfeller D (2024). Building and analyzing metacells in single-cell genomics data. *Molecular Systems Biology* 20. doi:10.1038/s44320-024-00045-6.

**Partition-dependence / metacell QC (Q2)**
- Liu P, Li JJ (2025). mcRigor: a statistical method to enhance the rigor of metacell partitioning in single-cell data analysis. *Nature Communications* 16:8602. doi:10.1038/s41467-025-63626-5.
- Zhang T, Zhu H (2025). Library size-stabilized metacells construction enhances co-expression network analysis in single-cell data. *PLOS Computational Biology*. doi:10.1371/journal.pcbi.1013697.

**Cross-field analogs (Q3)**
- Openshaw S, Taylor PJ (1979). A million or so correlation coefficients: three experiments on the modifiable areal unit problem. In *Statistical Applications in the Spatial Sciences*, Pion, pp. 127–144. (no DOI)
- Openshaw S (1984). *The Modifiable Areal Unit Problem*. CATMOG 38, Geo Books. (monograph, no DOI)
- Fotheringham AS, Wong DWS (1991). The modifiable areal unit problem in multivariate statistical analysis. *Environment and Planning A* 23(7):1025–1044. doi:10.1068/a231025.
- Goodchild MF (2022). The Openshaw effect. *Int. J. Geographical Information Science* 36(9). doi:10.1080/13658816.2022.2102637.
- Swift A, Liu L, Uber J (2014). MAUP sensitivity analysis of ecological bias in health studies. *GeoJournal* 79:137–153. doi:10.1007/s10708-013-9504-z.
- Robinson WS (1950). Ecological correlations and the behavior of individuals. *American Sociological Review* 15(3):351–357. doi:10.2307/2087176.
- Duncan OD, Davis B (1953). An alternative to ecological correlation. *American Sociological Review* 18(6):665–666. doi:10.2307/2088122.
- King G (1997). *A Solution to the Ecological Inference Problem*. Princeton Univ. Press. (reissue doi:10.1515/9781400849208)
- King G, Rosen O, Tanner MA, eds. (2004). *Ecological Inference: New Methodological Strategies*. Cambridge Univ. Press. doi:10.1017/cbo9780511510595.
- Theil H (1954). *Linear Aggregation of Economic Relations*. North-Holland. (monograph, no DOI)
- Wald A (1940). The fitting of straight lines if both variables are subject to error. *Annals of Mathematical Statistics* 11(3):284–300. doi:10.1214/aoms/1177731868.
- Durbin J (1954). Errors in variables. *Rev. Int. Statistical Institute* 22(1/3):23–32. doi:10.2307/1401917.
- Cramer JS (1964). Efficient grouping, regression and correlation in Engel curve analysis. *JASA* 59(305):233–250. doi:10.1080/01621459.1964.10480714.
- Grunfeld Y, Griliches Z (1960). Is aggregation necessarily bad? *Review of Economics and Statistics* 42(1):1–13. doi:10.2307/1926089.

**Causal discovery without aggregation / on counts (Q4)**
- Dai H, Ng I, Luo G, Spirtes P, Stojanov P, Zhang K (2024). Gene regulatory network inference in the presence of dropouts: a causal view. *ICLR 2024 (oral)*. arXiv:2403.15500.
- Choi J, Ni Y (2023). Model-based causal discovery for zero-inflated count data. *JMLR* 24, art. 200.
- Sato R, Scutari M, Kawano S, Yamaguchi R, Imoto S (2025). DAG learning from zero-inflated count data using continuous optimization (ZICO). arXiv:2512.16233.
- Yu S, Drton M, Shojaie A (2023). Directed graphical models and causal discovery for zero-inflated data. *CLeaR 2023*, PMLR 213:27–67.
- Wen Y, et al. (2023). Applying causal discovery to single-cell analyses using CausalCell. *eLife*. doi:10.7554/eLife.81464 (PMID 37129360).
- Matsumoto H, et al. (2017). SCODE: ... reconstruction of GRNs. *Bioinformatics*. doi:10.1093/bioinformatics/btx194.
- Papili Gao N, et al. (2018). SINCERITIES: inferring GRNs from time-stamped scRNA-seq. *Bioinformatics* 34:258–266. doi:10.1093/bioinformatics/btx575.
- Glymour C, Zhang K, Spirtes P (2019). Review of causal discovery methods based on graphical models. *Frontiers in Genetics* 10:524. doi:10.3389/fgene.2019.00524.
- Kalisch M, Bühlmann P (2007). Estimating high-dimensional directed acyclic graphs with the PC-algorithm. *JMLR* 8:613–636. (no DOI)
- Kocaoglu M (2023). Characterization and learning of causal graphs with small conditioning sets. *NeurIPS 2023*.

**Mitigations / stability (Q5)**
- Meinshausen N, Bühlmann P (2010). Stability selection. *JRSS-B* 72(4):417–473. doi:10.1111/j.1467-9868.2010.00740.x.
- Friedman N, Goldszmidt M, Wyner A (1999). Data analysis with Bayesian networks: a bootstrap approach. *UAI 1999*. arXiv:1301.6695.
- Kiselev VY, et al. (2017). SC3: consensus clustering of single-cell RNA-seq data. *Nature Methods* 14:483–486. doi:10.1038/nmeth.4236.
- von Luxburg U (2010). Clustering stability: an overview. *Foundations and Trends in Machine Learning* 2(3). doi:10.1561/2200000008.
- Ben-Hur A, Elisseeff A, Guyon I (2002). A stability based method for discovering structure in clustered data. *Pacific Symposium on Biocomputing* 7:6–17.

---

## Gaps / what nobody seems to have done

Stated plainly, because these matter for how the paper is positioned:

1. **Bootstrapping/ensembling over metacell *partitions*, with downstream edge-frequency reporting, appears to be genuinely undone.** I found no paper that runs a graphical or causal model across an ensemble of independently-seeded (or independently-resolved) metacell partitions and reports the *distribution* of recovered edges as a stability statistic. The three adjacent literatures each stop short: mcRigor (Liu & Li 2025) filters *bad metacells within one partition*; consensus clustering (SC3; von Luxburg) ensembles *cluster labels* into one partition and stops before any downstream model; and bootstrap/stability methods in causal discovery (Friedman et al. 1999; Meinshausen & Bühlmann 2010; DAGBag/RFCI-bootstrap-type work **[unverified specifics]**) resample the *sample rows*, not the *aggregation that generated the rows*. **Partition-bootstrap for single-cell causal discovery is a defensible, novel-in-context contribution** — this is the strongest positioning claim your data support.

2. **No characterization of *seed/zoning* variance (as opposed to *quality*) in metacell-based inference.** The field's robustness claim (Bilous et al. 2024) is about *global* representations; nobody has shown — or refuted — that fine-grained CI/PAG structure is seed-stable. Your 9/11 finding is, as far as I can tell, a novel empirical demonstration that it is *not*, at least for FCI over chromatin covariates. That is publishable as a cautionary result in its own right.

3. **No metacell method advertises or guarantees seed-invariance of the partition** at fine grain; MetaCell's kNN-graph resampling is the closest thing to a stability mechanism, but it targets metacell *cohesion*, not invariance of the *set* of metacells.

4. **The explicit MAUP / ecological-inference framing has not, to my knowledge, been applied to metacell aggregation.** Naming metacell construction as an instance of the MAUP (scale + zoning effects) and the ecological fallacy is, in the single-cell methods literature, apparently new — a clean conceptual contribution that also imports 50+ years of mitigation practice.

**Honest caveats about this survey.** (i) Absence of evidence isn't proof: "partition-bootstrap for causal edges" may exist in a preprint or applied paper I did not surface — I searched metacell, causal-discovery, consensus-clustering, and stability-selection literatures but cannot exclude an obscure instance; frame the novelty claim as "to our knowledge." (ii) Two mitigation pointers for *data-row* bootstrap in causal discovery (DAGBag; RFCI+bootstrap calibration; a 2025 resampling simulation study) surfaced in search but I did **not** pin their DOIs — they are marked **[unverified]** and should be verified before citing. (iii) Three foundational Q3 items (Openshaw & Taylor 1979, Openshaw 1984, Theil 1954) are pre-DOI monographs/chapters; their existence, year, venue, and pagination are confirmed via archives and review records, but no stable DOI exists. (iv) The discrete-CI "≥5–10 per cell" heuristic is a widely-used convention, not a single citable theorem.
