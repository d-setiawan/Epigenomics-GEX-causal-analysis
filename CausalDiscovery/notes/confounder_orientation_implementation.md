# Confounder roots + mechanistic orientation — implementation & result

**Status: implemented & verified (4 genes); ready to wire into the pipeline.**
Script: `../scripts/prototype_confounder_tiers.py`. Date: 2026-07-10.

This is the first implementation step from [MISSION.md](../../MISSION.md) and the original design
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

## Limits / next

- The **2 depth confounders capture technical depth, not biological cell-state** — so peak↔peak
  edges are only partly de-confounded (`<->` drops but many remain). Adding **cell-cycle score** and
  **donor** (once threaded) should remove more of the shared-cell-state peak↔peak confounding,
  leaving the credible element↔element structure (the novel object per MISSION §2).
- The tiers **orient** the skeleton but don't prune it; stronger de-confounding needs the extra
  confounders above.
- GFCI knowledge (py-tetrad `Knowledge`) not yet wired — FCI carries the signal at this scale.

## Productionization plan (next build)

1. Add `--confounders auto` (+ explicit column names) and `--tiers chromatin_expr` to
   `run_pc_causallearn.py`: build `conf__*` from the matrix libsize columns, construct the tiered
   `BackgroundKnowledge`, pass to `fci`/`pc`.
2. **Support/consensus must treat `conf__*` nodes as confounders, not peaks** — exclude them from
   peak→gene / peak↔peak validation and edge accounting.
3. Thread through `causal_cli graph` + `run_gene_pipeline.py`; add a per-metacell **cell-cycle**
   covariate (scanpy `score_genes_cell_cycle` on the metacell RNA) and, later, **donor** (majority
   donor per metacell from raw cell metadata).
