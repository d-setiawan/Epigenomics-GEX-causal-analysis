# The same-mark enrichment — a positive signature that is probably technical

**Date:** 2026-07-20. **Status: unresolved, and it is the highest-priority thing to test.**
Found while looking for a *positive* element↔element signature, immediately after fixing a bug that
had invalidated every previous p-value (see the correction banner in
[peak_peak_validation_results.md](peak_peak_validation_results.md)).

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

## Two readings, and the unflattering one is more likely

**(a) Biological.** Elements in the same chromatin state are co-regulated by a shared local state.
This is what MISSION §2 predicts, and it would be the positive characterisation the project lacks.

**(b) Technical — more likely.** Same mark = same assay = same libraries, same preparation, same
per-metacell capture efficiency. We condition on `conf__chrom_depth`, which is the **mean** library
size across the six marks, so **per-mark depth fluctuation is uncontrolled**
([L12](../../docs/LIMITATIONS.md#l12-per-mark-technical-depth-is-not-conditioned-on)). Two peaks on the
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

## The test — cheap and decisive

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
