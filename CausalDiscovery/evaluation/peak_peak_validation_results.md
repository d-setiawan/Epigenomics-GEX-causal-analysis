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

## Queued signature axes (framework ready)

- **motif_shared** — shared JASPAR TF motif per peak pair (the direct "shared latent TF" test).
  Needs the online JASPAR + UCSC-sequence scan (already coded in `build_node_support_table.py`,
  `fetch_jaspar_records`); run once to attach TF-motif sets per peak, then Jaccard/shared-count.
- **chip_shared** — co-binding by the same TF in monocyte ChIP (ChIP-Atlas / ENCODE); stronger than
  motif presence. Needs a download.
- **coaccess (Cicero contrast)** — run Cicero on the same metacell data; show co-accessibility
  collapses the direct-vs-latent-confounded distinction our PAG makes. The core novelty comparison;
  R/monocle3 build.

Add degree-matching to the null (currently distance-only) when coverage allows.
