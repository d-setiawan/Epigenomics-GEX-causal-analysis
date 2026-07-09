# Target-gene selection by validation-suite coverage (monocyte)

**Date:** 2026-07-09. **Question:** are there genes with much richer monocyte peak→gene
validation than our current panel (CD14/CSF1R/IL1B), so discovered edges can actually be
validated? **Answer: yes — hundreds.** Ranked table:
[`../evaluation/candidate_genes_validation_rich.tsv`](../evaluation/candidate_genes_validation_rich.tsv).

## How genes were ranked (all monocyte-specific)

Three signals, all from databases we already query for monocyte:

1. **rE2G distal links (prediction axis)** — genome-wide ENCODE-rE2G CD14+ monocyte
   enhancer→gene predictions (667k links). Count of NON-promoter (distal) links per gene =
   how many enhancer→gene predictions a discovered peak→gene edge could match.
2. **eQTL datasets (genetics axis)** — number of the **11 monocyte eQTL-Catalogue `ge`
   datasets** in which the gene has a SuSiE credible set. `n_datasets` (not variant count) is
   the robust metric: appearing in many independent monocyte studies = strong, reproducible
   genetic support. (Variant counts inflate in LD blocks — e.g. HLA.)
3. **Expression in OUR RNA** — mean count + % of cells expressing, from
   `rna_with_glue.h5ad`. A gene is only usable if it's actually expressed in our monocytes.

Feasible + validation-rich filter (`expr ≥15% of cells` AND `eQTL ≥5 datasets` AND
`rE2G distal ≥30`) → **331 genes**, vs. our current panel which is weak on genetics.

## Our current panel is validation-poor on the genetics axis

| gene | rE2G distal | eQTL datasets (of 11) | eQTL cs-variants | % cells expr |
| --- | --- | --- | --- | --- |
| CD14 | 48 | **1** | 17 | 19% |
| CSF1R | 152 | **2** | 2 | 18% |
| IL1B | 53 | **0** | 0 | 9% |

IL1B has **zero** monocyte eQTL anywhere; CD14/CSF1R appear in only 1–2 datasets. They were
chosen for biology, not validation coverage.

## Recommended targets — validation-rich AND well-expressed in our monocytes

Top of the 331 (myeloid-relevant, high expression, strong on both axes):

| gene | rE2G distal | eQTL datasets | % cells expr | note |
| --- | --- | --- | --- | --- |
| **LSP1** | 55 | 9 | **84%** | myeloid cytoskeleton; best expression + strong validation |
| **PTPRJ** (CD148) | **247** | 8 | 51% | very enhancer-rich locus + good eQTL |
| **CTSC** | 35 | 9 | 56% | cathepsin C, myeloid protease |
| **CD52** | 86 | 8 | 83% | high expression + broad eQTL |
| **CST3** | 84 | 8 | 28% | cystatin C, very highly expressed (mean 20) |
| **PLCL1** | 43 | 9 | 38% | 9 datasets |
| **CD151** | 47 | 9 | 33% | 9 datasets |
| **RAPGEF1** | 243 | 8 | 51% | enhancer-rich + expressed |
| **BIN1** | 102 | 8 | 49% | myeloid; well-studied eQTL |
| **HLA-DQB1** | 39 | 10 | 36% | MHC-II; huge eQTL but LD-complex (localization caveat) |
| **IL4R** | 98 | 8 | 27% | immune receptor |
| **CTSH** | 92 | 8 | 29% | cathepsin H |

For contrast, **highly expressed but validation-poor** genes (the "IL1B trap" — expressed but
0 eQTL datasets): S100A9, VCAN, SPI1, CEBPB, FCGR3A. Marker status ≠ validatable.
Also **validation-rich but too rare in our data** (drop): MARCO (3% cells — macrophage, not
classical monocyte), C3AR1 (6%), TMEM176A/B (5–6%).

## Caveats / remaining feasibility gate

- **Nearby CUT&Tag peaks not yet verified.** The final gate is that the gene's locus has
  peaks within our window in OUR data. High rE2G-distal counts imply enhancer-rich loci (peaks
  likely), but this must be confirmed by running the `dataset` stage per candidate (which also
  needs each gene's TSS/strand — obtainable from Ensembl or the rE2G promoter links).
- **PCHi-C (physical axis) not ranked genome-wide** — only rE2G + eQTL here (PCHi-C raw matrix
  wasn't cached). Adding it would re-download the Javierre matrix and count monocyte `Mon≥5`
  bait-gene contacts genome-wide; would further sharpen the shortlist.
- **peak-peak validation still unbuilt** (Phase B: ChIP-Atlas/Hi-C) — applies to all genes.
- eQTL variant counts inflate in LD blocks; trust `n_datasets`.

## Recommendation

Expand the panel from 3 to ~10–15 genes drawn from the top of
`candidate_genes_validation_rich.tsv` (keep CD14/CSF1R for continuity + biology). Verify nearby
peaks by running the `dataset` stage on the shortlist, then run the metacell + GFCI + stability
+ 3-axis pipeline. This directly fixes the "edges we can't validate" problem: on these genes,
most discovered peak→gene edges will have a real chance of prediction- and genetics-axis support.

---

## Should we use other cell types? — pros / cons

Our **experimental data is monocyte** (scRNA + scCUT&Tag). We build causal graphs from those
cells; we cannot switch the *discovery* cell type without new experiments. So "other cell
types" realistically means one of three moves:

### A. Stay strict monocyte (current) + pick validation-rich genes
- **Pros:** cell-type-aligned validation is honest and defensible; an edge validated in
  monocyte is real for our system; no cross-cell-type confounding; the 331-gene pool shows
  monocyte data is *not* actually limiting once we pick the right genes.
- **Cons:** some individual genes (IL1B) have thin monocyte validation; monocyte databases are
  smaller than whole-blood.
- **Verdict: default.** The gene-selection above resolves most of the "not enough data" worry.

### B. Broaden validation to related myeloid / blood (labeled, supporting tier)
Add macrophage/DC enhancer maps, PBMC/whole-blood eQTL (eQTLGen ~cis for most genes; GTEx whole
blood), monocyte-adjacent Hi-C.
- **Pros:** dramatically more data — whole-blood eQTL covers nearly every gene; would raise
  validation coverage for almost any target, including IL1B.
- **Cons:** **cell-type/state mismatch** — a whole-blood eQTL may be lymphocyte-driven; a
  macrophage enhancer may be inactive in classical monocytes. This is the same relaxation we
  already flag for the TLR-stimulated-monocyte eQTL. It weakens the "monocyte-specific causal"
  claim if used as primary evidence.
- **Verdict: acceptable only as a clearly-labeled SUPPORTING tier** (like the stimulated-eQTL
  handling), never as primary monocyte validation. Cheap to add; must be flagged.

### C. Switch the whole study to a CRISPR-validated ENCODE line (K562 / GM12878)
- **Pros:** the **gold standard** — ENCODE CRISPRi-FlowFISH gives thousands of
  *experimentally-validated causal* enhancer→gene links in K562. That perturbation axis (the
  only truly causal validation) was a **dead end for monocytes**; in K562 it exists at scale.
- **Cons:** we have **no scRNA/scCUT&Tag in K562/GM12878** — this requires entirely new data.
  K562 is an aneuploid cancer line (CML), not primary immune cells — it discards the monocyte
  biology that motivates the project and is less physiologically meaningful.
- **Verdict: not feasible now** (no matching epigenome data) and off-mission. Worth remembering
  only if the goal ever shifts to benchmarking the *method* against CRISPR ground truth, in
  which case K562 public multiome + CRISPR-FlowFISH would be the benchmark of choice.

**Bottom line:** stay monocyte for discovery, upgrade the gene panel to validation-rich targets
(A), optionally add a labeled related-blood supporting tier (B). Only consider a K562-type
switch (C) if the objective becomes method-benchmarking against CRISPR truth, and only with new
matched data.
