# Expanded-panel pipeline results (15 genes, metacell + GFCI + stability + 3-axis)

> **⚠️ Superseded for the default pipeline** by
> [expanded_panel_confounder_results.md](expanded_panel_confounder_results.md) (2026-07-14), which
> re-runs the panel with confounder root-nodes + mechanistic orientation ON (`--confounders depth`)
> — the current default. That run directs **19/19** peak→gene edges and keeps 11 validated. This
> page is retained for the pre-confounder `gfci,fci` baseline and the GFCI comparison.

**Date:** 2026-07-09. Dataset: `monocyte_cuttag_peak_genes_300kb_metacell_expanded` (133 metacells,
window 300 kb). Pipeline: `--methods gfci,fci --stability-method fci --n-boot 30`. Validation suite:
rE2G (prediction, genome-wide) + PCHi-C (physical, 15-gene) + eQTL (genetics, 15-gene, mostly naive).
All 15 genes ran (2 needed the collinear-column fix, commit e3e07e3).

## Headline — TOP-TIER edges (bootstrap-stable AND target-aware validated)

**7 top-tier peak→gene edges across 4 genes, spanning all 3 validation axes** (vs. 2 edges on CD14
alone in the original 3-gene panel):

| gene | peak | axis | source | algos | stability |
| --- | --- | --- | --- | --- | --- |
| **LSP1** | `chr11:1844174-1844753__H3K4me2` | prediction | rE2G | fci | 0.77 |
| **LSP1** | `chr11:1850243-1850903__H3K27ac` | prediction | rE2G | fci | 0.57 |
| **IL4R** | `chr16:27401872-27402382__H3K4me3` | physical | PCHi-C | fci | 0.73 |
| **IL4R** | `chr16:27242827-27243157__H3K4me2` | physical | PCHi-C | fci | 0.53 |
| **CD14** | `chr5:140638860-140639949__H3K4me2` | genetics | eQTL | fci+gfci | 0.70 |
| **CD14** | `chr5:140591420-140591815__H3K4me3` | prediction | rE2G | fci | 0.70 |
| **PLCL1** | `chr2:197786015-197786538__H3K4me2` | prediction | rE2G | fci | 0.63 |

All three axes now yield validated stable edges: **prediction** (LSP1, PLCL1, CD14), **physical**
(IL4R), **genetics** (CD14). The CD14 genetics edge is also multi-algorithm (fci∩gfci).

## Per-gene peak→gene summary

| gene | peak→gene edges | validated | top-tier (stable+validated) |
| --- | --- | --- | --- |
| LSP1 | 2 | 2 | **2** |
| IL4R | 3 | 3 | **2** |
| CD14 | 4 | 4 | **2** |
| PLCL1 | 2 | 1 | **1** |
| CD52 | 1 | 1 | 0 (not stable) |
| CD151 | 2 | 1 | 0 |
| PTPRJ | 4 | 0 | 0 |
| RAPGEF1 | 1 | 0 | 0 |
| BIN1 | 1 | 0 | 0 |
| CSF1R | 1 | 0 | 0 |
| IL1B | 1 | 0 | 0 |
| CTSC / CST3 / CTSH / LGALS2 | 0 | 0 | 0 (gene node not adjacent to a peak in the PAG) |

## Reading

- **Expanding to validation-rich genes worked**: 4 genes now carry top-tier edges (vs 1 before), and
  the physical + genetics axes (not just prediction) produce validated stable edges.
- **Not every validation-rich gene yields a validated edge** — richness of the *databases* for a gene
  doesn't guarantee the *discovered* peak→gene edge coincides with a validated element (e.g. PTPRJ found
  4 peak→gene edges but none overlapped a validated peak; several genes had no peak→gene edge in the PAG
  at all). This is expected and honest: validation is a filter, not a guarantee.
- **LSP1 and IL4R are the new stars** — LSP1 (2 rE2G-validated stable edges, 84% expression) and IL4R
  (2 PCHi-C-validated stable edges). Both are stronger validated-edge yields than the original panel.

## Caveats
- Stability is row-subsampling bootstrap (not yet donor-block).
- eQTL for the new genes is mostly naive/resting monocyte (better state-match than the original panel's
  stimulated-only hits); still a minority from stimulated conditions.
- peak-peak validation still unbuilt (Phase B).
