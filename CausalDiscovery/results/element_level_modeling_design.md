# Element-level causal modeling — EXPLORATORY design note

**Status: EXPLORATORY / not adopted.** A modeling-design probe, not a production decision.
Prototype: `../scripts/prototype_element_aggregation.py`. Date: 2026-07-10.

## Why (the problem)

At the peak level, ~92% of discovered edges are peak↔peak, and empirically they are **not
regulation**: across the monocyte panel's 243 peak↔peak edges, **49% are bidirected (`<->` =
latent common cause), only 1/243 is a directed arrow**, and 72% join elements >50 kb apart. The
FCI PAG is correctly reporting that peaks co-vary because of **shared latent causes** (TF
programs, chromatin-domain/cell-state activity), not because one enhancer regulates another. The
cause is the variable choice: each `(region × mark)` is treated as an independent causal variable,
though many peaks are (a) redundant readouts of one element's activity, or (b) co-driven by
shared latent regulators.

## What the prototype does

1. **Merge** co-located mark-peaks (any mark, intervals within 2 kb) into one **regulatory element**.
2. **Summarize** each element's activity as **PC1** of its z-scored member-peak signals (the shared
   axis the marks co-measure); report PC1 explained-variance as a homogeneity check.
3. **Label** the element by chromatin state from its marks (active_enhancer, promoter, poised…).
4. **Rerun** GFCI + FCI on `{element activities + gene expression}` and compare to the peak graph.

## Results (4 genes, monocyte metacell, n=133)

| gene | peaks → elements | peak FCI edges (`<->`) | **element** FCI edges (`<->`) | gene-adjacent element(s) |
| --- | --- | --- | --- | --- |
| CD14 | 43 → 17 | 28 (15) | **8 (1)** | **promoter** + 2 active enhancers (one rE2G-validated) |
| CD151 | 52 → 24 | 49 (34) | **9 (0)** | **promoter** |
| IL4R | 36 → 15 | 24 (13) | **6 (1)** | poised enhancer chr16:27242827 (**PCHi-C-validated**) |
| PTPRJ | 27 → 12 | 12 (5) | **1 (0)** | none |

**Consistent findings:**
- **The confounding explosion collapses.** Bidirected (`<->`) edges go 15→1, 34→0, 13→1, 5→0.
  Total edges drop 3–12×. The unvalidatable peak↔peak majority was largely latent confounding,
  and reducing to element nodes absorbs most of it.
- **The surviving structure is interpretable.** Genes connect to their **own promoter** (CD14,
  CD151) and to **externally-validated enhancers** (CD14 rE2G element E09; IL4R PCHi-C element
  E05). The chromatin-state labels make each node self-describing.
- **Noise genes go quiet.** PTPRJ (4 peak→gene edges at peak level, 0 validated) has **no
  gene-adjacent element** — the element model is more conservative/honest about weak loci.

## Limitations (why this is exploratory, not adopted)

1. **PC1 collapse is lossy.** Multi-mark elements have **low PC1 explained variance (~0.36–0.55)**
   — co-located marks are only semi-redundant, so "element activity = PC1" discards real
   mark-specific information. (This is itself informative: same-locus marks are not simple copies.)
2. **Some validated edges drop.** CD14's eQTL-validated peak (chr5:140638860) is not gene-adjacent
   at the element level; aggregation can lose real signal along with the confounding.
3. **Co-location merging only fixes the ~9% same-element redundancy.** The 72% *distal*
   shared-latent-cause edges are reduced mostly as a side effect of fewer variables, not addressed
   at the source.
4. **GFCI under-powers at element level** (0–1 edges): with ~15–24 nodes at n=133, the score-based
   skeleton is too conservative; FCI is more informative here.

## Interpretation & recommended direction

Element aggregation **confirms the diagnosis** — most peak↔peak edges are confounding, and
collapsing variables removes them while leaving an interpretable promoter/enhancer → gene graph.
But PC1-collapse is too crude and lossy to adopt as-is. The cleaner successor, suggested by the
low PC1 variance and the distal-confounding majority, is a **latent-factor / measurement model**:

- Treat the marks as **multiple indicators** of a small number of **latent regulatory factors**
  (element activity, TF-program activity, chromatin-domain/cell-state), rather than collapsing to
  one PC. Infer the factors and ask **factor → gene** — using the latent-variable tools already
  installed (**GIN** in causal-learn; **RCD/CAM-UV**), which are built for exactly this.
- Add the **global cell-state confounders** (donor, cell-cycle) as explicit nodes to regress out —
  this targets the distal co-variation directly (ROADMAP Q3).

**Verdict:** element-level modeling is a promising interpretability upgrade and a strong argument
that the peak↔peak edges are confounding, not regulation — but the right production form is an
explicit latent-factor model, not PC1 aggregation. Keep exploring; do not wire into the pipeline yet.

---

## Latent-factor variant (the next step) — and what it revealed

Prototype: `../scripts/prototype_latent_factor.py`. Treat the mark-peaks as **indicators of a few
latent regulatory factors**: FactorAnalysis(K) on the peaks (K = #eigenvalues>1, Kaiser, capped 8),
factor scores as latent variables, then FCI/GFCI on `{factors + gene}`. Interpret each factor by
its top-loading peaks; report the gene-expression variance the factors explain (linear R²).

| gene | K | **gene R² by factors** | factor→gene edges | factor↔factor edges |
| --- | --- | --- | --- | --- |
| CD14 | 8 | **0.06** | 1 (F0 loads the rE2G+eQTL-validated peaks → gene) | 0 |
| CD151 | 8 | **0.06** | **0** | 2 |
| IL4R | 8 | **0.04** | **0** (despite factors loading heavily on PCHi-C/rE2G peaks) | 5–8 |
| PTPRJ | 7 | **0.16** | 1 (F5 → gene) | 5 |

**The key finding — unsupervised factors capture the confounders, not the gene-driver.**
The latent factors explain **~4–16% of gene-expression variance** (essentially chance for 7–8
predictors at n=133). Factor analysis is unsupervised w.r.t. the gene, so it extracts the
**dominant peak↔peak co-variation — which is the shared confounding** (cell-state / global activity
spanning the whole locus; the factors' top peaks are 100 kb+ apart), *not* the gene-specific
signal. Consequences:
- **factor→gene edges mostly vanish** (0–1 per gene) even when a factor clearly loads on
  externally-validated enhancers (IL4R F1 loads PCHi-C peaks at 0.74 and still has no gene edge).
- The gene-relevant regulatory signal is a **small residual**, drowned out by the confounding modes.
- GIN (data-driven latent discovery) independently finds only **1–2 latents** — agreeing the
  dominant latent structure is low-dimensional and shared.

## Three approaches compared

| approach | #variables | confounding removed | **gene signal retained** | interpretable |
| --- | --- | --- | --- | --- |
| peak-level (current) | ~40 | no (92% peak↔peak, 49% `<->`) | yes (validated edges found) | low |
| **element (PC1)** | ~15–24 | mostly | **yes** (promoter + validated enh → gene) | **high** |
| latent-factor (FA) | ~7–8 | yes | **no** (R²≈0.05; factor→gene ≈ 0) | medium (factors span locus) |

## Synthesis / recommended production direction

Unsupervised global factorization **over-collapses**: it removes the peak↔peak edges by turning the
confounders into variables, but in doing so it discards the gene-driving signal (the confounders
dominate the variance). **Element-level modeling is the better abstraction** — it stays local and
interpretable and keeps the promoter/validated-enhancer → gene edges — but PC1-collapse is lossy.

The exploration triangulates to a clear recipe, which is really ROADMAP Q3 sharpened:
1. **element-level nodes** (merge co-located marks; keep local, interpretable units), and
2. **regress out the top global factors as explicit nuisance** (they are the cell-state/donor
   confounders — the very modes FA surfaces), then
3. run causal discovery on the residualized element nodes.

That combines the confounding removal of the factor model with the signal retention of the element
model. **Neither PC1-element nor unsupervised-FA is adopted; this is the design target to build next.**
(Cosmetic prototype note: the factor `state` label is crude — it flags any H3K4me3 as "promoter";
factors are actually locus-spanning mixes.)

---

## Fused variant (element nodes + regress-out global nuisance) — the closing test

Prototype: `../scripts/prototype_fused_element_nuisance.py`. Build element nodes, then regress the
**top-k global element-PCs** (the dominant shared modes) out of every element **and** the gene, then
FCI/GFCI on the residuals. Nuisance PCs confirmed **gene-irrelevant** (|corr(PC, gene)| ≤ 0.16 for
all genes), so removing them *should* be safe. The test: do validated gene→element edges survive,
and does element↔element confounding drop?

| gene | element FCI: edges (→gene) | **fused FCI: edges (→gene)** | validated edge survives? |
| --- | --- | --- | --- |
| CD14 | 8 (3) | **18 (1)** | yes (E09 rE2G) — but `<->` 1→12 |
| IL4R | 6 (1) | **10 (1)** | yes (E05 PCHi-C) |
| CD151 | 9 (1) | **17 (0)** | **no** (E19 promoter lost) |
| PTPRJ | 1 (0) | **9 (0)** | n/a |

**The naive fused approach BACKFIRES.** Regressing out data-driven PCs **inflated** the edge count
in every gene (8→18, 6→10, 9→17, 1→9) and multiplied bidirected edges (CD14 `<->` 1→12) — the
opposite of the goal. **Mechanism:** residualizing a set of variables on principal components
*estimated from those same variables* forces the residuals to be orthogonal to the removed
directions, which **induces spurious partial correlations** among them (a conditioning/collider
artifact). Validated-edge survival was also only 2/4. So data-driven nuisance regression is not a
valid confounder control here.

Two things it *did* teach us: (1) validated gene→element edges are **partially robust** — CD14 and
IL4R kept theirs even under an artifact-inducing transform; (2) it pinpoints the correct method.

## CLEAR DIRECTION (conclusion of the modeling exploration)

Across peak → element → factor → fused, the evidence converges on a specific, principled design:

1. **Adopt element-level nodes** as the modeling unit. Merge co-located mark-peaks into regulatory
   elements (interpretable chromatin-state labels; genes connect to their promoter + validated
   enhancers). This is the only variant that both suppresses the peak↔peak confounding *and* keeps
   the validated gene→element signal. *Open refinement:* PC1-collapse is lossy (marks are
   semi-independent, explained var ~0.4) — replace it with a small per-element measurement model or
   keep the 1–2 most informative marks per element.
2. **Do NOT use data-driven nuisance regression / self-derived PCs.** Proven here to induce spurious
   edges. Likewise, PCs cannot be added as conditioning *nodes* (they are linear combos of the
   elements → singular).
3. **Control confounders with MEASURED covariates as conditioning nodes** — the principled form of
   ROADMAP Q3. Compute per-metacell **donor id, cell-cycle score, and library size / global
   chromatin activity** from the RNA/chromatin, add them to the discovery variable set, and let FCI
   condition on them. These are external to the elements (not collinear), so conditioning is valid
   and removes the global cell-state confounding without the residualization artifact.
4. **Use FCI, not GFCI, at this scale** — the score-based skeleton under-powers at ~15–24 nodes,
   n=133 (GFCI repeatedly returned 0 edges).

**Net:** element-level nodes + measured-covariate conditioning (donor/cell-cycle/library-size),
discovered with FCI, is the design target. The PC1-element and both latent-factor variants
(unsupervised FA, self-PC nuisance regression) are **rejected as production forms** but were
necessary to locate this conclusion. Status remains exploratory — the measured-covariate step is
the next build, and requires threading those covariates through the metacell + dataset stages.
