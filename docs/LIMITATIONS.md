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
