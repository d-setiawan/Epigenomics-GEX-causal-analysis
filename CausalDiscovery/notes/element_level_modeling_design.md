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
