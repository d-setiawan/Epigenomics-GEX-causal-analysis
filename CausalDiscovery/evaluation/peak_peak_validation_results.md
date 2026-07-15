# Peak↔peak (element↔element) validation — first axis (PCHi-C physical hub)

**Date:** 2026-07-15. Script: `../scripts/peak_peak_validator.py`. Panel:
`outputs/pipeline_confounder_panel` (15 genes, confounder-conditioned FCI). This is the keystone for
the MISSION novelty — turning bidirected `<->` edges from an unvalidatable object into a falsifiable,
distance-matched test. First signature axis implemented; three more queued.

## The test

- **Positive set:** the **126 bidirected `<->` peak↔peak edges** from the confounder-conditioned PAG
  (the "shared latent common cause" edges). Extraction verified against the panel totals (126 of 239
  peak↔peak edges are bidirected). No peak↔peak edge is *directed* — the orientation prior directs
  only peak→gene, so peak↔peak is bidirected or partial (o-o), as expected.
- **Null:** distance- (and, next, degree-) matched **non-adjacent** peak pairs (6,803 pairs). Genomic
  distance is THE confound — near peaks share everything by chance — so the label is permuted WITHIN
  distance bins (n=2000 permutations).
- **Signature (this axis):** `pchic_hub` — two peaks loop to a shared PCHi-C **bait** (same promoter
  hub / physical domain). Offline, and independent of both discovery (PCHi-C is not used to build
  peak↔peak edges) and the peak→gene validation (target-gene-aware, a different use).

## Result — a clean NEGATIVE (physical hub does not explain bidirected edges)

| edge class | n pairs | raw hub rate | distance-matched diff vs null | perm p |
| --- | --- | --- | --- | --- |
| **bidirected** | 126 | 0.071 | **+0.008** | **1.00** |
| other_adjacent (o-o) | 113 | 0.097 | +0.019 | 1.00 |
| non_adjacent (null) | 6,803 | 0.066 | — | — |

Bidirected edges are **not** enriched for shared PCHi-C hubs once distance is controlled. The raw rate
(7.1% vs 6.6%) looked marginally positive, but that is a **distance artifact** — bidirected edges
cluster at short range where hub co-membership is naturally higher; matching removes it. (The test
correctly catches and strips this confound — evidence the null is doing its job, not a bug.)

## Reading — this sharpens the hypothesis, it does not sink it

1. **Shared *physical domain* is the wrong mechanism (or invisible here).** The MISSION hypothesis is
   that a `<->` edge reflects a shared **latent TRANS regulator (a TF)** — not necessarily a shared 3D
   promoter hub. PCHi-C tests the *domain* mechanism; this null pushes the weight onto the
   **shared-TF** axis (motif / ChIP), which is next.
2. **Power caveat (important):** PCHi-C hub co-membership is sparse — only ~6.6% of all pairs share a
   hub, so only ~9 bidirected pairs are hub-positive. This axis is **underpowered**; the negative is
   "no evidence," not "evidence of no effect." The shared-TF axes should have far denser coverage.
3. **The framework is the deliverable.** `peak_peak_validator.py` now provides the positive-set
   extraction, the distance-matched permutation null, and a plug-in signature slot. Adding an axis =
   adding one 0/1 (or continuous) per-pair column.

## Axis 2 — shared-TF motif (2026-07-15): NEGATIVE, and saturated

`scan_peak_motifs.py` scanned all 416 panel peaks against 150 JASPAR CORE-vertebrate motifs (44,547
hits; UCSC sequence + JASPAR PWMs, cached). Signature = shared TF-motif set per pair. Same
distance-matched permutation test, threshold ≥0.95 relative score.

| signature | bidirected | null | matched diff | p |
| --- | --- | --- | --- | --- |
| share ≥1 motif | 0.900 | 0.892 | +0.009 | 1.00 |
| motif-set Jaccard | 0.196 | 0.211 | **−0.011** | 1.00 |

**Bidirected edges do NOT share more TF motifs than distance-matched non-edges** — if anything very
slightly fewer. **But motif presence is saturated** (median ~14–37 TFs/peak; 90% of *all* pairs share
≥1 motif), so this axis has almost no discriminating power. Motif *presence* ≠ TF *binding*: a monocyte
master TF (PU.1/CEBP) motif is in nearly every myeloid enhancer, so "shared motif" cannot localize a
shared *active* regulator. **The decisive shared-TF test is ChIP-Atlas co-binding (actual occupancy),
not motif presence** — next.

## Co-accessibility contrast (the core novelty comparison) — STRONG POSITIVE

`coaccess_contrast.py` computes Cicero-style co-accessibility (|Spearman| of peak accessibility across
metacells) on the **same** data, and compares it to what the PAG resolves.

| PAG edge class | mean co-access \|ρ\| | n |
| --- | --- | --- |
| bidirected `<->` | **0.238** | 126 |
| other adjacent (o-o) | **0.255** | 113 |
| non-adjacent (pruned) | 0.064 | 6,803 |

- **Co-accessibility conflates.** It gives bidirected (0.238) and other adjacent (0.255) essentially the
  **same** score — it cannot separate the latent-confounded `<->` edges from other adjacencies.
- **Cicero's-eye view** (top-10% co-access = 705 "links"): **67% are pairs the PAG prunes as
  non-adjacent**, 17% bidirected (latent), 15% other. Most of what co-accessibility calls "links" are
  the indirect/confounded relationships our conditioning removes.
- **Headline:** **53% of the PAG's element↔element adjacencies are bidirected (latent common cause)** —
  co-accessibility reports these as regulatory links; only our latent-variable method flags them as
  confounded.

**Robustness note (be precise in the paper):** raw co-accessibility does not prune indirect paths
through *observed* peaks; Cicero's graphical-lasso would remove many of the 67% non-adjacent
false-positives (as our conditioning does). So the *method-fair, unassailable* part of the contrast is
the **latent** class: graphical-lasso removes indirect paths through **observed** variables but
**cannot** remove a shared **latent** driver — precisely the bidirected class. That is the theoretical
guarantee behind the 53% number. (A distance-penalized-glasso faithful-Cicero variant is the refinement
to add.)

## Interpretation after two biological axes — honest status

The **methodological** novelty is supported: co-accessibility demonstrably conflates the direct vs
latent-confounded distinction our PAG makes (strong contrast above). But the **biological identity** of
the latent driver behind the `<->` edges is **not yet established**: neither shared physical domain
(PCHi-C) nor shared TF-motif presence explains them above a distance-matched null. Three live
possibilities, in order of what to test next:
1. **Shared *TF occupancy* (not mere motif presence)** — ChIP-Atlas monocyte TF ChIP is the decisive
   test; motif presence was too saturated to decide. **Do this next.**
2. **A shared *global chromatin-activity* latent** (general "openness"/state co-varying many enhancers)
   rather than a specific TF or domain. If ChIP-Atlas is also null, this is the likely story — and it
   reshapes the claim: the `<->` edges are real shared-latent-cause structure, but the cause is global
   state, which is either an honest biological finding (enhancers co-vary by state, not pairwise
   regulation) or a residual confound to condition on.
3. **Small-n PAG instability** — partially covered by the existing stability selection; revisit if 1–2
   are null.

## Queued / next
- **chip_shared (ChIP-Atlas co-binding)** — the decisive shared-TF test (download monocyte TF ChIP).
- **Faithful-Cicero** distance-penalized graphical-lasso co-accessibility (refinement of the contrast).
- **Degree-matching** added to the null (currently distance-only).
- **Global-activity latent** probe (per-metacell mean accessibility as a candidate shared driver).
