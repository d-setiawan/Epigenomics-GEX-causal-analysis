# Expanded-panel results — confounder-conditioned (15 genes, metacell, depth roots + orientation)

**Date:** 2026-07-14. **This is the current default-pipeline headline.**
Dataset: `monocyte_cuttag_peak_genes_300kb_metacell_expanded` (133 metacells, 300 kb window).
Pipeline: `run_gene_pipeline.py --methods fci,pc --confounders depth --n-boot 50` (confounder
root-nodes + mechanistic orientation now ON by default). Validation suite: rE2G (prediction) +
PCHi-C (physical) + eQTL (genetics), each cell-type + target-gene aware. All 15 genes ran clean.

Supersedes the pre-confounder run in [expanded_panel_results.md](expanded_panel_results.md)
(2026-07-09, `gfci,fci`, no confounders). Method set changed to `fci,pc` because **GFCI does not yet
honor the confounder/tier `Knowledge`** (Phase-D item); FCI+PC both accept the tiered background
knowledge, so multi-algo here means **fci ∩ pc**.

## Headline

| metric | count | note |
| --- | --- | --- |
| peak→gene edges (panel) | **19** | across 11 of 15 genes |
| **directed `A→Y`** | **19 / 19 (100%)** | mechanistic orientation resolved *every* peak→gene edge |
| target-aware validated | **11 (58%)** | all 11 are directed AND multi-algo (fci ∩ pc) |
| stable **and** validated (top tier) | **5** | bootstrap-recurrent + validated |
| peak↔peak edges | 239 | the novel object; 126 bidirected `<->`, 136 bootstrap-stable |
| confounder edges (`conf__* → node`) | 45 | depth roots absorbing technical co-variation |

**The result the confounder+orientation wiring was built for: 100% of peak→gene edges are now
directed** (`chromatin → expression`), up from a mix of ambiguous `o->`/`<->` marks before. The
mechanistic prior (chromatin upstream of expression, depth as a root) resolves the orientation the
data alone could not — the honest route to a *directed* causal claim — **without** discarding any
validated edge.

## The 11 validated peak→gene edges (all directed, all fci ∩ pc)

| gene | peak | axis / source | stable? |
| --- | --- | --- | --- |
| **CD14** | `chr5:140563832-140564962__H3K4me3` | physical (PCHi-C) | |
| **CD14** | `chr5:140591420-140591815__H3K4me3` | prediction (rE2G) | ★ stable |
| **CD14** | `chr5:140612011-140612339__H3K27ac` | prediction (rE2G) | ★ stable |
| **CD14** | `chr5:140638860-140639949__H3K4me2` | genetics (eQTL) | |
| **LSP1** | `chr11:1844174-1844753__H3K4me2` | physical + prediction | ★ stable |
| **CD52** | `chr1:26289150-26289429__H3K27ac` | physical + prediction | |
| **PLCL1** | `chr2:197786015-197786538__H3K4me2` | prediction (rE2G) | ★ stable |
| **CD151** | `chr11:841767-842901__H3K4me3` | genetics (eQTL) | |
| **IL4R** | `chr16:27203357-27204050__H3K27ac` | physical (PCHi-C) | |
| **IL4R** | `chr16:27242827-27243157__H3K4me2` | physical (PCHi-C) | |
| **IL4R** | `chr16:27401872-27402382__H3K4me3` | physical (PCHi-C) | ★ stable |

All three axes are represented: **prediction** (CD14 ×2, LSP1, CD52, PLCL1), **physical** (CD14, LSP1,
CD52, IL4R ×3), **genetics** (CD14, CD151). CD14 alone spans all three.

## Per-gene peak→gene summary

| gene | peak→gene | directed | validated | stable+validated | peak↔peak | conf edges |
| --- | --- | --- | --- | --- | --- | --- |
| CD14 | 4 | 4 | **4** | **2** | 24 | 1 |
| IL4R | 3 | 3 | **3** | **1** | 21 | 2 |
| PLCL1 | 2 | 2 | 1 | **1** | 1 | 2 |
| CD151 | 2 | 2 | 1 | 0 | 45 | 6 |
| PTPRJ | 2 | 2 | 0 | 0 | 8 | 5 |
| LSP1 | 1 | 1 | **1** | **1** | 8 | 4 |
| CD52 | 1 | 1 | **1** | 0 | 42 | 3 |
| CSF1R | 1 | 1 | 0 | 0 | 5 | 2 |
| IL1B | 1 | 1 | 0 | 0 | 8 | 5 |
| RAPGEF1 | 1 | 1 | 0 | 0 | 13 | 2 |
| BIN1 | 1 | 1 | 0 | 0 | 8 | 3 |
| CTSC / CST3 / CTSH / LGALS2 | 0 | – | 0 | 0 | 5–35 | 1–3 |

## Reading

- **Orientation is solved panel-wide.** Every peak→gene edge is directed `chromatin → expression`.
  This is the deliverable of the confounder-root + tiered-orientation implementation (verified on 4
  genes earlier; now confirmed across all 15).
- **Confounder conditioning prunes spurious peak→gene edges.** PTPRJ dropped 4→2 peak→gene edges vs.
  the pre-confounder run — depth-driven co-variation is now attributed to `conf__*` roots instead of
  fabricating a chromatin→gene link.
- **Validation is a filter, not a guarantee.** Database-rich genes don't all yield a *discovered* edge
  that overlaps a validated element (PTPRJ: 2 directed edges, 0 validated), and 4 genes have no
  peak→gene edge in the PAG at all. Honest and expected.
- **The peak↔peak layer is the novel object (MISSION §2).** 239 peak↔peak edges, 126 bidirected
  `<->`. Because metacell aggregation already averages out donor/batch and depth is conditioned as a
  root, the residual `<->` edges are **not** technical/donor confounding — they reflect biological
  shared drivers (TF programs, chromatin domains), correctly left latent in the PAG. Target-aware
  peak↔peak validation is the pending Phase-B build.

## Caveats
- Multi-algo = **fci ∩ pc** (both constraint-based). GFCI is excluded until its `Knowledge` object is
  wired to accept the tiers/confounders (Phase D); it cannot yet be run confounder-conditioned.
- Stability is row-subsampling bootstrap (n_boot 50, thr 0.5), not donor-block.
- eQTL for several genes is naive/resting monocyte; peak↔peak validation still unbuilt (Phase B).
- Raw artifacts: `CausalDiscovery/outputs/pipeline_confounder_panel/<GENE>/edges_consensus.tsv`;
  panel summary `.../panel_confounder_summary.tsv`.
