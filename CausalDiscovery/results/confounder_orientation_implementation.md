# Confounder roots + mechanistic orientation — implementation & result

**Status: implemented & verified (4 genes); ready to wire into the pipeline.**
Script: `../scripts/prototype_confounder_tiers.py`. Date: 2026-07-10.

This is the first implementation step from [MISSION.md](../../docs/MISSION.md) and the original design
(`notes/local_multiome_notears_plan.md`): observed confounders as **roots**, chromatin **upstream**
of expression, gene does **not** cause chromatin — all encoded as one **tiered background-knowledge**
object and given to FCI.

## Method

```
tier 0 = confounders   ->   tier 1 = peaks   ->   tier 2 = gene expression
```

Tiers forbid backward edges (peak→confounder, gene→peak, gene→confounder) and, by adding the
confounders as **nodes**, let FCI **condition** on them. `causal-learn` `BackgroundKnowledge`
(`add_node_to_tier`, `forbid_within_tier`) + `fci(background_knowledge=…)`; the tier API is present
in both the `scrna_r` (0.1.4.5) and `causaldisco` (0.1.4.7) envs, so productionizing is de-risked.

**Confounders available now:** per-metacell **library sizes** already in every matrix —
`conf__rna_depth = log(libsize__rna)`, `conf__chrom_depth = log(mean libsize__<mark>)`. These are
the dominant *technical* confounders (depth drives both peak signal and expression). **Not yet
available:** donor (metacell obs is empty — needs threading from raw cell metadata), cell-cycle
(computable from metacell RNA — optional next).

## Result (metacell, n=133, 43–52 peaks)

| gene | peak→gene **directed** (baseline → tiers) | validated peak→gene (all now directed) | `<->` (baseline → tiers) | peak↔peak kept |
| --- | --- | --- | --- | --- |
| CD14 | 1/4 → **4/4** | 4 (rE2G, eQTL, PCHi-C, rE2G) | 15 → 12 | 24 |
| IL4R | 0/3 → **3/3** | 3 (PCHi-C ×3) | 15 → 11 | 21 |
| CD151 | 0/2 → **2/2** | 1 (eQTL) | 41 → 34 | 45 |
| PTPRJ | 0/2 → **2/2** | 0 | 8 → 3 | 8 |

**What it achieves (consistent across all genes):**
1. **Orientation solved.** Every peak→gene edge becomes a directed `A→Y` (was mostly ambiguous
   `<->`/`o->`). The mechanistic prior (chromatin upstream of expression) resolves the direction the
   data alone could not — the honest route to a *directed* causal claim.
2. **Validated edges survive AND become directed** — CD14 (4), IL4R (3), CD151 (1) are all directed
   peak→gene after tiers. Independent-axis validation stays honest (the tiers use mechanism, not the
   validated edges).
3. **Confounders act as roots** — `conf__chrom_depth → peak` edges appear (correctly oriented);
   depth-driven peak co-variation is attributed to the confounder.
4. **peak↔peak edges preserved** — the deliverable is kept; bidirected `<->` count drops (confounding
   partly resolved).

## Donor & cell-cycle investigated (2026-07-10) — library depth is the right & sufficient set

A parallel investigation checked whether **donor** or **cell-cycle** should be added:
- **Donor: NO.** 5 donors exist (HTO-multiplexed: HTO5/7/9/10/13) but the label is on **chromatin
  cells only** (RNA was never hashtagged) and, decisively, **metacells average donor out** — donor
  purity per metacell ≈ 0.27 (random-mix floor 0.23), donor is statistically independent of metacell
  (χ² p = 0.43). There is no cross-metacell donor variance to confound the edges; metacell
  aggregation already neutralizes donor/batch by construction.
- **Cell-cycle: NO (marginal).** Monocytes here are ~1–2% cycling (S/G2M scores near zero;
  proliferation markers in ~1–2% of cells). S-score has a weak tie to CD14 *expression* (r = −0.24)
  but **no correlation with the CD14 peaks** (max |r| ≈ 0.24, none > 0.3), so it can't act as a
  confounder for the epigenomic edges.

**Consequence:** the **per-metacell library depths are the right and sufficient observed-confounder
set** for this metacell-level model — the confounder implementation is effectively complete. And a
deeper point: since donor/cell-cycle are already averaged out, the peak↔peak `<->` edges that remain
after depth-conditioning are **not** technical/donor/cell-cycle confounding — they reflect
**biological shared drivers** (TF programs, chromatin domains), i.e. the novel object (MISSION §2),
correctly represented as latent via the PAG bidirected edges (and pursued observationally via the
element/latent-factor direction, not by adding more observed confounders).

## GFCI knowledge wiring (2026-07-15) — done, with two Tetrad gotchas

GFCI (`run_gfci_tetrad.py`) now accepts `--confounders` + `--background-mode` and is threaded through
`run_gene_pipeline.py gfci_cmd`, so `--methods gfci,fci --confounders depth` runs GFCI
confounder-conditioned. Verified on CD14: GFCI now agrees with FCI on the genetics edge
(`chr5:140638860-140639949__H3K4me2 → CD14`), promoting it to **VALIDATED_MULTI_ALGO (fci|gfci)**.
Two findings worth recording:

1. **Tetrad `Knowledge` silently no-ops on node names containing `:` / `-`.** Our peak names
   (`chr5:140638860-140639949__H3K4me3`) fail Tetrad's variable-name spec matching, so
   `set_forbidden`/`add_to_tier` register but bind to nothing — the search runs as if there were no
   knowledge. **Fix:** rename every variable to a knowledge-safe token (`V0`, `V1`, …) before the
   search, then translate node names back in the output graph string. With safe names the knowledge
   binds (`isForbidden == True`) and the FGES skeleton becomes confounder-aware. Any earlier GFCI
   knowledge attempt would have failed silently — check this first if GFCI ever "ignores" knowledge.
2. **GFCI's `Gfci` uses knowledge for the skeleton but not for PAG endpoint orientation** (this Tetrad
   build): cross-tier edges stay `o-o` even though the forbidden directions are registered. Tetrad's
   constraint-based FCI *does* orient them (`o->`). So the **directed** chromatin→expression claims
   come from the causal-learn FCI primary; GFCI contributes a confounder-aware, knowledge-constrained
   *adjacency* set as a score-based sensitivity check (validation keys on adjacency, so GFCI edges
   still validate and can be multi-algo).

## Remaining

- The tiers **orient** the skeleton but don't prune it; the residual peak↔peak is latent-biological
  (above), so it is a target for the element-level / latent-factor modeling, not more observed
  confounders.

## Productionization plan (next build)

1. Add `--confounders auto` (+ explicit column names) and `--tiers chromatin_expr` to
   `run_pc_causallearn.py`: build `conf__*` from the matrix libsize columns, construct the tiered
   `BackgroundKnowledge`, pass to `fci`/`pc`.
2. **Support/consensus must treat `conf__*` nodes as confounders, not peaks** — exclude them from
   peak→gene / peak↔peak validation and edge accounting.
3. Thread through `causal_cli graph` + `run_gene_pipeline.py`; add a per-metacell **cell-cycle**
   covariate (scanpy `score_genes_cell_cycle` on the metacell RNA) and, later, **donor** (majority
   donor per metacell from raw cell metadata).
