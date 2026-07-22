# Limitations & honesty ledger

**Owner doc:** this file is the *single home* for every known caveat. Other docs must **link here**
rather than restate a limitation, so a caveat can never be fixed in one place and left stale in
three. ↑ [Central README](../README.md) · [MISSION](MISSION.md) · [ROADMAP](ROADMAP.md)

Each entry: what the limit is → what it blocks → what we do instead.

---

## L1. Donor is not an available variable

**The fact.** The dataset was derived from separate donors, but donor is **not a usable variable in
either modality**: the RNA cells were never hashtagged (no donor labels at all), and the
HTO-multiplexed donor labels on the chromatin side do not correspond to donors on the RNA side.
There is therefore no donor structure shared across modalities.

**What it blocks.**
- **Donor-level block-bootstrap is impossible** — there are no donor labels to block on. Stability
  selection is row-subsampling bootstrap and will stay that way. This is *not* a pending task.
- **"Invariance across donors" is not an available route to causal claims.** Earlier drafts of
  MISSION §4 listed donor-invariance as a pillar of the method; it has been replaced.
- `--confounders depth,donor` exists in the CLI but is inert (it is honored only if a `donor__*`
  column exists, and none does). Treat it as unavailable.

**What we do instead.**
- **Cross-cell-type invariance (ICP)** is the available invariance axis — edges that hold in both
  monocyte and T-cell are the stronger claim. The T-cell suite is already built
  ([celltype_fair_comparison.md](../Evaluation/results/celltype_fair_comparison.md)); the ICP run
  itself is still open (ROADMAP §0.5).
- Donor confounding is handled by **aggregation, not conditioning**: metacells average donor out by
  construction (purity ≈0.27 ≈ the random-mix floor; χ² donor×metacell p=0.43), so there is no
  cross-metacell donor variance left to confound edges. Library depth is the right and sufficient
  observed confounder set.

**Consequence for the science.** Because metacells already neutralize donor/batch, residual peak↔peak
`<->` edges are *not* donor artifacts — which is what licenses reading them as biological shared
drivers (MISSION §2).

## L2. Pseudo-pairs are not within-individual

RNA and chromatin cells are matched through the scGLUE latent, not by donor identity. Since the two
modalities do not share donors (L1), a paired row combines expression and chromatin **from different
individuals**, aligned on cell state rather than genotype.

**What it blocks.** Genotype-driven (eQTL-like) chromatin↔expression covariation cannot be recovered
in principle — the chromatin donor's genotype is not the RNA donor's genotype. What the pairing can
carry is **cell-state-driven** covariation, which is what the latent aligns on.

> This is consistent with an observation already logged independently: eQTL support localises to the
> CD14 promoter but **never changes a peak→gene edge verdict** — the fine-mapped variants sit in the
> promoter, not in the distal peaks discovery connects to expression
> ([PIPELINE.md](../CausalDiscovery/PIPELINE.md)). Worth stating explicitly in the write-up as a
> scope limit on the genetics axis rather than a null result.

## L3. Small n, and n_eff is not reported

≈133 metacells per gene. Honest but small — hence stability selection and transforms. The
`n_eff` diagnostic (participation ratio / reuse proxy) is still unimplemented, so for any legacy
kNN-averaged result the effective sample size is well below the row count and fisherz p-values are
optimistic. The current metacell datasets are non-overlapping, so `n_rows ≈ n_eff` there.

## L4. Stability is row-subsampling, not block bootstrap

n_boot 50, threshold 0.5. Per L1 this cannot be upgraded to donor blocks. If a stronger dependence
structure needs accounting for, the option is a cell-state/cluster block, not a donor block.

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
[peak_peak_validation_results.md](../Evaluation/results/peak_peak_validation_results.md).

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
([partition_bootstrap_results.md](../Evaluation/results/partition_bootstrap_results.md)): no peak→gene
edge is recovered in all 10 partitions, 68% of edges appear in a single partition, and the most robust
validated edge (rE2G-supported CD14 H3K4me3) is found in 6/10. Report edges by recovery frequency, not
from one partition. Background: [partition_stability_findings.md](../Evaluation/results/partition_stability_findings.md).

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
Full evidence chain: [same_mark_finding.md](../Evaluation/results/same_mark_finding.md). Remaining
caveats there (n=55, single cell type, correlational).

**Correction 2026-07-22 (partition bootstrap).** The "resolved as biological" verdict above was from a
3-seed pilot and was **partition-lucky**. Across 10 partitions the same-mark enrichment is significant
in only **4/10** (mean +0.046, negative in 3); the two strong partitions include the canonical seed 0.
It is not a technical artifact where it appears, but it is **partition-fragile and cannot be claimed as
an established signature**. See [partition_bootstrap_results.md](../Evaluation/results/partition_bootstrap_results.md).

## L13. Peaks are histone-mark peaks, not ATAC; validation is enhancer/ATAC-derived

The covariates are **per-histone-mark CUT&Tag peaks** (H3K27ac, H3K4me1/2/3 = active; H3K27me3, H3K9me3
= repressive), not ATAC open-chromatin peaks. Consequences:

- A "peak" is region **+ chromatin state**, so one genomic locus can appear as up to 6 separate
  variables (one per mark). Same-locus different-mark covariation is biologically expected (active
  elements carry H3K27ac + H3K4me1 together) — relevant to the same-mark analysis
  ([same_mark_finding.md](../Evaluation/results/same_mark_finding.md)).
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
