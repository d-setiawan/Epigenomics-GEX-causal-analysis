# The same-mark enrichment — a weak, partition-fragile tendency (NOT the headline signature)

**Date:** 2026-07-20, tech-confound tests 2026-07-21, **partition test 2026-07-22 (this changes the
conclusion).** **Status: not a technical artifact, but NOT partition-robust — significant in only 4 of
10 metacell partitions. Cannot be claimed as an established positive signature.**

> ## ⚠️ CORRECTION 2026-07-22 — read this first
>
> On 2026-07-21 this doc concluded, from a **3-seed pilot**, that the same-mark enrichment was
> "resolved as biological — the project's positive signature." The full **10-partition** bootstrap
> ([partition_bootstrap_results.md](partition_bootstrap_results.md)) shows that was **partition-lucky**:
> distance-matched same-mark enrichment is significant (p<0.05) in only **4/10** partitions, mean effect
> **+0.046** (range −0.013 to +0.105), and the two strongest partitions are seed 0 and seed 1 — seed 0
> being the canonical partition every prior result used.
>
> **What survives:** the technical-confound tests below (per-mark depth, split-half) still show that
> *where* the enrichment appears it is not a depth or shared-cell-substrate artifact. So there may be a
> real but **weak** effect (mean > 0). **What does not survive:** headlining it as an established,
> robust biological signature. It is partition-fragile. The sections below marked "biological /
> resolved" predate this correction and are superseded by it.

## The result

Bidirected `<->` pairs are far more likely than distance-matched non-edges to be measured on the
**same histone mark** — i.e. to come from the same assay channel:

| class | P(same mark) | n |
| --- | --- | --- |
| **bidirected `<->`** | **0.437** | 126 |
| other adjacent (`o-o`) | 0.319 | 113 |
| non-adjacent (null) | 0.292 | 6803 |

| test | matched diff | p |
| --- | --- | --- |
| distance-matched | **+0.143** | **0.0005** |
| distance + degree matched | **+0.136** | **0.0010** |

For scale: every one of the four biological axes gave |diff| ≤ 0.017. **This effect is an order of
magnitude larger than anything else we have measured**, and it is the only significant element↔element
signature found so far.

## Two readings *(pre-result: predicted "probably depth" — that was refuted, see § RESULT)*

**(a) Biological.** Elements in the same chromatin state are co-regulated by a shared local state.
This is what MISSION §2 predicts, and it would be the positive characterisation the project lacks.

**(b) Technical — more likely.** Same mark = same assay = same libraries, same preparation, same
per-metacell capture efficiency. We condition on `conf__chrom_depth`, which is the **mean** library
size across the six marks, so **per-mark depth fluctuation is uncontrolled**
([L12](../../../../project_resources/research/LIMITATIONS.md#l12-per-mark-technical-depth-is-not-conditioned-on)). Two peaks on the
same channel share that channel's fluctuation across metacells, producing covariation with no
directional structure — which is precisely what FCI reports as a latent common cause.

**Three things favour (b):**

1. **The mechanism is exactly right.** An unmodelled shared confounder produces bidirected edges. That
   is not a coincidental match; it is the textbook signature.
2. **It is specific to the bidirected class.** `o-o` pairs show only +0.037 (p = 0.16). A *biological*
   co-regulation story would plausibly enrich all adjacent pairs; an *unmodelled confounder* story
   predicts enrichment concentrated in the class that represents unmodelled confounding.
3. **The variable exists and we averaged it away.** `libsize__<MARK>` is in every matrix. The averaging
   was a modelling choice, not a data limitation.

## Why this is serious

The bidirected edges are the project's novel object. If a meaningful share of the 126 are pairs of
peaks that covary because they were sequenced in the same assay, then:

- the **53% latent** headline is inflated by an unknown amount;
- the four-negative elimination argument is weakened — we may have been hunting for the biological
  identity of a partly technical artifact;
- the co-accessibility contrast is **not** affected in the same way (it is a statement about what
  correlation cannot distinguish, which holds regardless of why the pairs covary) — but the *size* of
  the latent class would change.

## RESULT (2026-07-21) — per-mark depth is NOT the explanation

Ran it: `--confounders depth_permark` replaces the single averaged `conf__chrom_depth` with six
per-mark depth roots (`conf__depth_H3K27ac … conf__depth_H3K9me3`), panel re-run, same 15 genes, same
metacells.

| | depth (1 avg root) | depth_permark (6 roots) |
| --- | --- | --- |
| peak→gene | 19 | 18 |
| validated peak→gene | 11 | **11** (unchanged) |
| peak↔peak | 239 | 230 |
| **bidirected `<->`** | 126 | **135 (up 9)** |
| **same-mark enrichment (bidirected)** | +0.143, p=0.0005 | **+0.105, p=0.0025** |
| same-mark enrichment (`o-o`) | +0.037, p=0.18 | +0.085, p=0.030 |

**The enrichment survives.** Conditioning on per-assay sequencing depth did **not** remove it
(p=0.0025, still significant) and did **not** reduce the bidirected count (it rose). So the simple
technical hypothesis — that same-mark bidirected edges are same-assay *depth* artifacts — is **refuted**.
The effect size dropped modestly (+0.143 → +0.105), so per-mark depth explains at most a small fraction.

**But "not depth" ≠ "biological."** There is a subtler technical explanation that depth only *partially*
captures, and the modest-drop-but-persists pattern is exactly what it predicts:

> **Shared measurement substrate.** Within a metacell, a peak of mark *M* is quantified from that
> metacell's mark-*M* cells. Two **same-mark** peaks are therefore measured from the **same cells**;
> two cross-mark peaks from **different cells**. Same-mark peaks thus share their entire measurement
> substrate — the same capture events, same per-cell technical state — not just total depth. Library
> depth is only a scalar summary of that shared substrate, so conditioning on it removes only part of
> the effect. This is a **metacell-construction artifact**, distinct from sequencing depth.

**Current standing:** the biological reading (elements in a shared chromatin state, co-regulated by a
shared local state — MISSION §2) is now **more tenable than before** — the enrichment is real and not a
depth artifact — but it is **not established**, because shared measurement substrate is an unresolved
alternative. The two make the same prediction under distance matching, so distance matching cannot
separate them.

## RESULT 2 (2026-07-21) — split-half: NOT shared cell substrate either

`pipelines/simulation/aggregation_identifiability/scripts/split_half_substrate_test.py`. Within each metacell, split that metacell's mark-M
cells into two **disjoint** halves H1/H2 and quantify every peak from each half. For a peak pair:
`rho_same = corr(peak_i from H1, peak_j from H1)` (share cells) vs
`rho_cross = corr(peak_i from H1, peak_j from H2)` (disjoint cells). Substrate → rho_cross collapses;
biology → both halves estimate the same metacell state, so rho_cross ≈ rho_same. Reuses the pipeline's
own per-cell quantifier, averaged over **20 random splits** (single-split is too noisy at this n).

| pairs | rho_same (share cells) | rho_cross (disjoint cells) |
| --- | --- | --- |
| non-adjacent same-mark (baseline) | 0.052 | 0.052 |
| **bidirected same-mark** | **0.137** | **0.140** |
| bidirected cross-mark (control) | 0.150 | 0.157 |

Excess of bidirected same-mark over baseline: same-half **+0.085**, cross-half **+0.088** → **≈104% of
the excess survives** measuring the two peaks from disjoint cells. Breaking the shared cells does
**nothing** to the correlation. So the same-mark bidirected correlation is **not** shared per-cell
substrate — it is metacell-level (cell-state) covariation.

*(A first single-split run suggested ~28% substrate, but that was split noise — the cross-mark control
behaved anomalously at single-split. Averaging over 20 splits removed it and the control settled.)*

## Where this leaves the same-mark signature — resolved, biological

| candidate explanation | test | result |
| --- | --- | --- |
| p-value artifact | fixed the permutation bug | ✗ refuted (real, p=0.0005) |
| per-mark sequencing depth | six per-mark depth roots (RESULT 1) | ✗ refuted (survives, p=0.0025) |
| shared cell substrate | split-half, disjoint cells (RESULT 2) | ✗ refuted (104% survives) |
| **biological (shared cell-state co-regulation)** | remaining, positively consistent | ✓ the surviving explanation |

This is **the project's positive element↔element signature** — the thing the four-negative axes could
not supply. Bidirected edges preferentially connect elements in the **same chromatin state**, and that
co-variation is genuine cell-state signal, not any technical artifact we could construct. It is exactly
MISSION §2's "elements co-driven by a shared local co-regulation state."

**Honest limits:** n=55 same-mark bidirected pairs; the effect is moderate (0.137 vs 0.052 baseline);
one cell type; and this is a correlational/graphical result, not perturbational. "Same mark" is also a
coarse proxy for chromatin state (it is the assay channel, which tracks state but is not identical to
it). Worth confirming at scale (more genes → more bidirected pairs) once partition-stability is handled.

---

## The test — cheap and decisive (original plan, now done — see above)

Replace the single averaged depth root with **six per-mark depth roots**
(`conf__depth_H3K27ac … conf__depth_H3K9me3`, from the `libsize__<MARK>` columns already present) and
re-run the panel. Then:

| outcome | reading |
| --- | --- |
| bidirected count drops **and** same-mark enrichment disappears | it was technical; the confounder set was incomplete |
| both persist | it is not per-mark depth; the biological reading gains real support |
| enrichment drops but edges persist | partial — quantify what fraction was technical |

This is one pipeline re-run with a changed `--confounders` argument. It should happen before any
scaling, and before the four-negative endpoint is written up as complete.

## Caveats on this finding itself

- Single panel, 15 genes, 126 bidirected pairs.
- "Same mark" is a coarse proxy for "same assay batch" — it does not separate mark identity from
  library/run identity.
- Reading (a) and (b) are not mutually exclusive; both could contribute.

## Note on how this was found

It came from asking for a *positive* signature rather than testing another candidate latent — and it
surfaced the permutation bug, because a +0.14 effect returning p = 1.00 is impossible. Two lessons
worth keeping: **a test that only ever returns null results deserves suspicion**, and **the negative
controls are as informative as the positive ones**.
