# Monocyte vs T-cell — FAIR validated-edge comparison

**Date:** 2026-07-10. This completes the cell-type question: each cell type is run on **its OWN
validation-rich, cell-type-matched gene panel**, with **its own cell-type-specific validation
suite** (rE2G + eQTL + PCHi-C), through the identical metacell + GFCI + FCI + stability + 3-axis
pipeline. This removes the monocyte-bias confound of the earlier power probe
([celltype_power_comparison.md](celltype_power_comparison.md)).

## Setup — genuinely matched

| | monocyte | T cell |
| --- | --- | --- |
| metacells (n) | 133 | **245** (~1.8×) |
| genes in panel | 15 | 12 (11 ran; HLA-C excluded — MHC, hyphen-naming) |
| gene selection | monocyte rE2G + eQTL × monocyte expr | T-cell rE2G + eQTL × T-cell expr |
| rE2G validation | CD14+ monocyte (667k links) | **naive CD4+CD8 (2.3M links)** |
| eQTL validation | 11 monocyte datasets | **18 resting T-cell datasets** |
| PCHi-C validation | BLUEPRINT Mon | naive CD4+CD8 (nCD4∪nCD8) |

T cell has BOTH advantages one would expect to help: **more cells (245 vs 133)** and a **richer
validation suite** (3.4× more rE2G links, 18 vs 11 eQTL datasets).

## Result — the advantages did NOT translate into more validated edges

| metric | monocyte | T cell |
| --- | --- | --- |
| **top-tier edges** (bootstrap-stable AND target-aware validated) | **7** | 4 |
| genes carrying a top-tier edge | 4 | 4 |
| top-tier edges per gene run | 7/15 = 0.47 | 4/11 = 0.36 |
| validated (stability-relaxed) peak→gene edges | 12 (6 genes) | 7 (7 genes) |

**T-cell top-tier edges (n=245):**

| gene | peak | axis | stability |
| --- | --- | --- | --- |
| UCP2 | `chr11:73982710-73983376__H3K27ac` | prediction (rE2G) | 0.70 |
| B4GALT1 | `chr9:33159003-33159460__H3K27ac` | **physical + prediction** (PCHi-C+rE2G) | 0.63 |
| CCSER2 | `chr10:84139311-84139511__H3K4me1` | physical (PCHi-C) | 0.60 |
| CTSW | `chr11:66001553-66002880__H3K4me3` | physical (PCHi-C) | 0.53 |

## Interpretation

- **Despite ~1.8× more cells and a 3× richer validation suite, T-cell did not beat monocyte on
  validated stable-edge yield (4 vs 7 top-tier; 0.36 vs 0.47 per gene).** More `n` and more
  reference data did **not** produce more validated causal edges. This is consistent with the
  power probe: causal-edge recovery is limited by **per-gene regulatory effect size**, not by
  cell count or database size.
- **Different axis profile (a real, interesting difference):** T-cell top-tier edges are
  **physical-axis-dominated** (3/4 PCHi-C, one dual-axis), whereas monocyte's were
  prediction/genetics-dominated (rE2G + eQTL). This likely reflects both cell-type regulatory
  architecture and the relative density of each validation database per cell type.
- **Numbers are small** (4 vs 7 across ~12–15 genes) — the robust qualitative claim is
  "comparable, monocyte marginally ahead," not a precise ratio.

## Conclusion / recommendation

**Keep monocyte as the primary analysis.** The fair, T-cell-specific, well-powered comparison
shows T cell's larger `n` and richer databases do **not** yield more validated causal edges —
monocyte is marginally ahead and has the best scGLUE integration confidence (0.96). The T-cell
suite is fully built and reusable (`pipelines/empirical/data/raw/external/*_tcell/`, `tcell_validation_peak_genes.tsv`),
so T cell remains available as a secondary system or for cell-type-contrast analyses (e.g. the
physical-axis-heavy T-cell edges), but there is no power-based reason to switch the primary
analysis to it.

## Caveats
- HLA-C excluded (MHC complexity + a gene-name hyphen/underscore dir mismatch); not re-run.
- Stability is row-subsampling (not donor-block) in both arms — matched, so the comparison holds.
- Small edge counts; treat as qualitative.
