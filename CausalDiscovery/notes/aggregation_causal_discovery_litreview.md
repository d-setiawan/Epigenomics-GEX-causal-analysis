# How aggregation affects downstream causal discovery — a literature review

**Date:** 2026-07-09 · Grounding: OpenAlex retrieval (polite pool), access 2026-07-09.
**Purpose:** justify the project's Q2 decision (replace overlapping kNN-averaging with
non-overlapping metacells) and the observed bake-off results, against the published
literature. Cross-refs: [../evaluation/aggregation_bakeoff.md](../evaluation/aggregation_bakeoff.md),
[../evaluation/null_calibration_findings.md](../evaluation/null_calibration_findings.md),
[../../ROADMAP.md](../../ROADMAP.md).

> TL;DR — Three literatures converge on the same conclusion we reached empirically.
> (1) **Averaging/smoothing overlapping neighborhoods manufactures correlations that were
> not in the data** and inflates type-I error, because the resulting rows are not
> independent. (2) **Non-overlapping metacells** (each cell used once) are the accepted fix
> — they restore approximate independence and denoise without the smoothing artifact, *if*
> the metacells are homogeneous. (3) In the *causal* setting specifically, **aggregation is
> not statistically neutral**: the level of coarse-graining can change which causal
> structure is even identifiable, so metacell graining is a modeling choice, not just
> denoising. Stability selection is the recommended guard for graph estimation at small n.

---

## 1. The aggregation landscape in single-cell

Single-cell data are sparse and noisy, so almost every pipeline aggregates cells before
downstream modeling. The methods differ in one property that turns out to be decisive for
statistics: **whether a cell can belong to more than one aggregate.**

- **Non-overlapping (disjoint) metacells.** **MetaCell** (Baran et al. 2019, *Genome Biol*)
  introduced partitioning scRNA-seq into "disjoint and homogenous groups of profiles that
  could have been resampled from the same cell" — the resampling framing is exactly the
  independence argument. **SuperCell** (Bilous et al. 2022, *BMC Bioinformatics*) coarse-grains
  similar cells and shows downstream results (clustering, DE, **gene–gene correlation**, GRN)
  are "preserved and often improved." **SEACells** (Persad et al. 2023, *Nat Biotechnol*) uses
  archetypal analysis to find granular metacells across RNA **and ATAC**, overcoming sparsity
  while retaining heterogeneity clustering hides. **MetaQ** (Li et al. 2025, *Nat Commun*)
  scales this via vector quantization. A 2024 review (Bilous et al., *Mol Syst Biol*) surveys
  the family.
- **Overlapping smoothing / imputation / kNN-averaging.** Each cell contributes to many
  neighbors' values (imputation methods like MAGIC/DCA; kNN-pooling; **our current
  RNA-anchor kNN-averaging**). This is the branch the statistics literature warns about.

**Takeaway for us.** Our current aggregation (each chromatin cell reused 11–17× across
anchors — see ROADMAP §Q2) is squarely in the *overlapping* category; the accepted
alternative is a *disjoint* metacell partition (MetaCell/SEACells), which is exactly what we
built (native scanpy Leiden partition, one cell per metacell).

## 2. Why overlapping averaging breaks the statistics (the mechanism we hit)

- **Smoothing manufactures correlations.** Andrews & Hemberg (2019, *F1000Research*), *"False
  signals induced by single-cell imputation"*, show that imputation/smoothing methods that
  borrow information across similar cells **generate strong gene–gene correlations that are
  not present in the unsmoothed data**, producing false network signal. Overlapping
  kNN-averaging is the same operation. This is the direct mechanism behind our
  null-calibration result: the smoothed matrix supports edges that are artifacts of the
  averaging, not the biology.
- **Non-independence inflates type-I error (pseudoreplication).** Squair et al. (2021, *Nat
  Commun*, ~1000 cites) and Zimmerman et al. (2021, *Nat Commun*) show that treating cells
  from a shared source as independent replicates ("pseudoreplication") yields **"biased
  inference, highly inflated type-1 error rates, and reduced reproducibility."** The accepted
  remedy is aggregation to genuinely independent units (**pseudobulk**), confirmed by muscat
  (Crowell et al. 2020, *Nat Commun*) and Murphy & Skene (2022, *Nat Commun*). Overlapping
  averaging is *worse* than naive pseudoreplication because it also adds the smoothing
  artifact of §2.1.
- **Variance shrinkage + heavier tails.** Averaging k neighbors shrinks within-feature
  variance and, because rows share cells, correlates the rows. For a CI test whose p-value is
  a function of the sample covariance and a nominal n, this reads as *more* information than
  exists → anti-conservative tails. (Classic dependent-data / effective-sample-size result;
  the single-cell instantiations above.) **This is precisely our α=0.01 finding: the
  overlapping baseline produced 3.0× the nominal false edges, while the disjoint metacell
  produced 1.0× (calibrated).**

## 3. Aggregation is not neutral for *causal* identifiability

Beyond variance/inference, the causal-discovery literature has a distinct and stronger
warning: coarse-graining can change *which causal graph is recoverable at all*.

- **Temporal aggregation destroys/rewrites causal structure.** Gong, Zhang & Schölkopf
  (2017, *"Causal Discovery from Temporally Aggregated Time Series"*, UAI) prove that when the
  observed series is an aggregate of a finer causal process, the causal structure is
  recoverable **only under extra conditions (non-Gaussian, independent noise)** — otherwise
  aggregation confounds it. The analogy for us: metacells aggregate over the cell population;
  the aggregation level is a first-class modeling assumption, and Gaussianity of the aggregate
  is not automatically benign (supports our rank/nonparanormal transform choice).
- **The right macro-variables must be *constructed*.** Chalupka, Perona & Eberhardt (2014,
  *Visual Causal Feature Learning*) and the **Causal Coarsening Theorem** formalize when a
  coarse-grained (macro) variable preserves the causal relationships of its micro-constituents.
  Metacells are macro-variables over cells; there exists a "correct" granularity for causal
  consistency, motivating principled graining (and QC — see §5) rather than an arbitrary k.
- **Aggregation can reverse associations (Simpson / ecological fallacy).** Pearl (2013,
  *Understanding Simpson's Paradox*) and Kievit et al. (2013, *Front Psychol*) show a
  population-level association can invert within subgroups; Loney & Nagelkerke (2014, *Emerg
  Themes Epidemiol*) formalize the ecological fallacy — aggregate-level associations need not
  equal individual-level causal effects. **Consequence for us:** edges discovered on
  metacell-aggregated data are *metacell/population-level* causal claims; over-coarse or
  heterogeneous metacells risk aggregation artifacts. This is why we keep the metacell
  granularity fine (~133 metacells of ~90 cells) and validate edges externally.

## 4. The downstream engine's own sensitivity to n and dependence

Even with clean rows, constraint-based causal discovery is fragile at small n / many
variables — which frames the trade-off aggregation forces (fewer, cleaner rows).

- **CI testing is fundamentally hard.** Shah & Peters (2020, *Ann Stat*) prove no CI test can
  be uniformly valid for continuous conditioning sets — every test trades calibration for
  power. Fast approximations (RCIT/RCoT, Strobl et al. 2019, *J Causal Inference*; CMIknn,
  Runge 2017) exist precisely because exact non-parametric tests scale badly with n. This
  justifies our retirement of KCI for rank-fisherz + RCoT.
- **Small-n, high-p is ill-posed.** Schäfer & Strimmer (2004, *Bioinformatics*) show network
  inference with n ≪ p is ill-posed and needs shrinkage; neighborhood selection with the
  nonparanormal (huge package) is standard. Our honest metacell n (~133) is small, so we lean
  on the transform + stability rather than raw p-values.
- **Reviews.** Glymour, Zhang & Spirtes (2019, *Front Genet*) and Spirtes & Zhang (2016,
  *Applied Informatics*) survey the constraint/score families and their assumptions.

## 5. Guards the literature recommends — and that we adopted

- **Check metacell homogeneity.** mcRigor (Liu & Li 2025, *Nat Commun*) shows that
  **over-aggregation of heterogeneous cells biases downstream analysis and yields "spurious
  discoveries,"** and provides a statistical homogeneity test + graining selection. This is
  the direct QC for §3's coarsening risk — a planned add-on for our metacell step.
- **Stability selection.** Subsample-and-recur variable/edge selection (Meinshausen &
  Bühlmann; implemented for graphs in the huge package) controls false discoveries in
  high-dimensional graph estimation without relying on any single fit's p-values. **This is
  exactly our `bootstrap_stability.py` result**: real stable edges (recurring ≥0.5) vs a
  permuted null that never exceeds 0.12–0.22 — the clean signal/noise separation.
- **Aggregate to independent units for inference** (Squair, Zimmerman, muscat): our
  non-overlapping metacells are the causal-discovery analogue of pseudobulk-for-DE.

## 6. Unpaired multi-omics matching (our design specifics)

Our RNA and CUT&Tag cells are *different* cells in a shared latent (scGLUE; Cao & Gao 2022,
*Nat Biotechnol*). The metacell partition over the joint latent co-aggregates both modalities
per region of latent space — a simple, robust alternative to explicit optimal-transport
matching (uniPort, Cao et al. 2022, *Nat Commun*; neural OT, Bunne et al. 2023, *Nat
Methods*). OT metacell-to-metacell matching remains a documented refinement (ROADMAP Phase C).

---

## 7. Synthesis — what the literature says about *our* pipeline

| Our observation | Literature explanation | Citation anchor |
| --- | --- | --- |
| Overlapping kNN-avg → dense, noise-level graphs | smoothing manufactures gene–gene correlations | Andrews & Hemberg 2019 |
| Real edge count ≈ α×pairs; α=0.01 baseline 3.0× miscalibrated | non-independent rows inflate type-I error | Squair 2021; Zimmerman 2021 |
| Disjoint metacell → calibration 3.0×→1.0× | disjoint aggregates ≈ independent units | Baran 2019 (MetaCell); Murphy & Skene 2022 |
| Metacell graining is a real choice, not just denoising | aggregation can change causal identifiability | Gong 2017; Chalupka 2014 |
| Interpret edges at the metacell/population level | Simpson / ecological fallacy | Pearl 2013; Loney 2014 |
| Stability selection separates signal from noise | subsampling-based FDR control for graphs | Meinshausen–Bühlmann / huge |
| Small honest n (~133) is acceptable, needs care | small-n network inference is ill-posed → shrink/transform | Schäfer & Strimmer 2004; Shah & Peters 2020 |
| Metacell homogeneity must be checked | over-aggregation → spurious discoveries | Liu & Li 2025 (mcRigor) |

**Bottom line.** The literature independently supports every step of our Q2 conclusion:
overlapping averaging is a known false-signal generator; non-overlapping metacells are the
accepted remedy; the aggregation *level* is causally consequential (so QC it with mcRigor and
interpret edges at population level); and stability selection is the right guard at the small
honest n that disjoint aggregation produces.

---

## References (retrieved via OpenAlex, 2026-07-09)

**Metacells / aggregation methods**
- Baran et al. 2019. MetaCell: analysis of scRNA-seq using K-nn graph partitions. *Genome Biol.* doi:10.1186/s13059-019-1812-2
- Bilous et al. 2022. Metacells untangle large and complex single-cell transcriptome networks (SuperCell). *BMC Bioinformatics.* doi:10.1186/s12859-022-04861-1
- Persad et al. 2023. SEACells infers transcriptional and epigenomic cellular states. *Nat Biotechnol.* doi:10.1038/s41587-023-01716-9
- Bilous et al. 2024. Building and analyzing metacells in single-cell genomics data. *Mol Syst Biol.* doi:10.1038/s44320-024-00045-6
- Li et al. 2025. MetaQ: metacell inference via single-cell quantization. *Nat Commun.* doi:10.1038/s41467-025-56424-6
- Liu & Li 2025. mcRigor: enhancing the rigor of metacell partitioning. *Nat Commun.* doi:10.1038/s41467-025-63626-5

**Smoothing / pseudoreplication / inference validity**
- Andrews & Hemberg 2019. False signals induced by single-cell imputation. *F1000Research.* doi:10.12688/f1000research.16613.2
- Squair et al. 2021. Confronting false discoveries in single-cell differential expression. *Nat Commun.* doi:10.1038/s41467-021-25960-2
- Zimmerman et al. 2021. A practical solution to pseudoreplication bias in single-cell studies. *Nat Commun.* doi:10.1038/s41467-021-21038-1
- Murphy & Skene 2022. A balanced measure shows superior performance of pseudobulk methods. *Nat Commun.* doi:10.1038/s41467-022-35519-4
- Crowell et al. 2020. muscat: multi-sample multi-condition scRNA-seq DS analysis. *Nat Commun.* doi:10.1038/s41467-020-19894-4

**Aggregation & causal structure**
- Gong, Zhang & Schölkopf 2017. Causal Discovery from Temporally Aggregated Time Series. *UAI 2017.* (Proc. of UAI; no DOI)
- Chalupka, Perona & Eberhardt 2014. Visual Causal Feature Learning (Causal Coarsening Theorem). *arXiv.* doi:10.48550/arXiv.1412.2309
- Pearl 2013. Understanding Simpson's Paradox. doi:10.2139/ssrn.2343788
- Kievit et al. 2013. Simpson's paradox in psychological science. *Front Psychol.* doi:10.3389/fpsyg.2013.00513
- Loney & Nagelkerke 2014. The individualistic fallacy, ecological studies and instrumental variables. *Emerg Themes Epidemiol.* doi:10.1186/1742-7622-11-18

**Causal discovery machinery / CI testing**
- Shah & Peters 2020. The hardness of conditional independence testing and the generalised covariance measure. *Ann Stat.* doi:10.1214/19-aos1857
- Strobl, Shyam & Zhang 2019. Approximate Kernel-Based CI Tests (RCIT/RCoT). *J Causal Inference.* doi:10.1515/jci-2018-0017
- Runge 2017. CI testing based on a nearest-neighbor estimator of CMI (CMIknn). *arXiv:1709.01447* (AISTATS 2018)
- Glymour, Zhang & Spirtes 2019. Review of Causal Discovery Methods Based on Graphical Models. *Front Genet.* doi:10.3389/fgene.2019.00524
- Spirtes & Zhang 2016. Causal discovery and inference: concepts and recent methodological advances. *Applied Informatics.* doi:10.1186/s40535-016-0018-x
- Schäfer & Strimmer 2004. An empirical Bayes approach to inferring large-scale gene association networks. *Bioinformatics.* doi:10.1093/bioinformatics/bti062

**Integration / context**
- Cao & Gao 2022. Multi-omics single-cell integration with graph-linked embedding (GLUE). *Nat Biotechnol.* doi:10.1038/s41587-022-01284-4
- Cao et al. 2022. uniPort: unified single-cell integration with optimal transport. *Nat Commun.* doi:10.1038/s41467-022-35094-8
- Aibar et al. 2017. SCENIC: single-cell regulatory network inference and clustering. *Nat Methods.* doi:10.1038/nmeth.4463

*Note on provenance:* citations were retrieved by OpenAlex full-text search (polite pool) on
2026-07-09; DOIs are as returned by OpenAlex. The Meinshausen–Bühlmann stability-selection
method is referenced as implemented in the `huge` R package (retrieved); the original 2010
JRSS-B paper was not pulled in this session. Verify DOIs before formal citation.
