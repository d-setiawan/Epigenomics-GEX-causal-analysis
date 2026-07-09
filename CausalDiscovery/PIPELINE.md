# End-to-End Causal Pipeline (test a gene → validate its edges)

This is the runnable spine that ties the whole project together: **give it a gene,
it runs several causal-discovery algorithms and tells you which of their edges the
evaluation suite can actually validate.** It is the concrete answer to the project
question and the thing to grow as Phases C/D land.

Driver: [`scripts/run_gene_pipeline.py`](scripts/run_gene_pipeline.py) ·
Panel roll-up: [`scripts/summarize_pipeline_panel.py`](scripts/summarize_pipeline_panel.py) ·
Method decisions & phase plan: [`../ROADMAP.md`](../ROADMAP.md).

---

## What it does (one command)

```bash
python CausalDiscovery/scripts/run_gene_pipeline.py --gene CD14 --methods fci,pc,dagma
# then, across genes:
python CausalDiscovery/scripts/summarize_pipeline_panel.py --genes CD14,CSF1R,IL1B
```

Stages (each shells out to the tested `causal_cli.py` sub-commands — no stage is
re-implemented):

1. **Discovery** — for each algorithm, `causal_cli graph` on the gene's nearby-peak
   matrix → a graph (PAG for FCI, CPDAG for PC, weighted DAG for DAGMA).
2. **Validation** — `causal_cli support` on every graph with the target-gene-aware
   evaluation suite (ENCODE-rE2G + BLUEPRINT PCHi-C; eQTL slots in next).
3. **Consensus** — collapse edges *across algorithms* and assign an **axis-aware
   verdict** (below). Outputs per gene: `edges_consensus.tsv`,
   `edges_per_method.tsv`, `pipeline_report.md`, `pipeline_manifest.json`.

Outputs land in `CausalDiscovery/outputs/pipeline/<GENE>/`.

## The axis-aware validation rule (the important part)

The support engine emits a `validation_rule` per edge. **Not all "validated" edges
mean the same thing** — the pipeline separates them into tiers so we never over-claim:

| edge kind | rule used | what "validated" means | counts as a real causal edge? |
| --- | --- | --- | --- |
| **peak → gene** | `region_interval_with_matching_target_gene` | the peak is a supported regulatory element **of that specific gene** | **YES** — this is the headline |
| peak – peak | `region_interval_on_either_endpoint` | both intervals appear in *some* database (gene-agnostic) | **NO** — node-existence only |

Verdicts on peak→gene edges: `VALIDATED_MULTI_ALGO` (target-aware support **and**
found by ≥2 algorithms) › `VALIDATED` › `recurrent_unvalidated` (≥2 algorithms,
no support) › `unvalidated`. Axes map: `re2g→prediction`, `pchic→physical`,
`eqtl→genetics`; "≥2 axes agree" is the Q1 gold standard.

> ⚠️ peak-peak edges are the majority of what discovery returns, but the suite has
> **no target-aware peak-peak validator yet** (Phase B: ChIP-Atlas co-binding /
> Hi-C loops). The pipeline reports them as `no_target_validator`, not "validated."

## Demonstrated run (300 kb kNN-avg baseline, fisherz + rank_gaussian, α=0.05, depth 2)

| gene | adjacencies | peak→gene | ≥2-algo | target-aware validated |
| --- | --- | --- | --- | --- |
| **CD14** | 56 | 2 | 2 | **1** |
| CSF1R | 10 | 0 | 0 | 0 |
| IL1B | 6 | 1 | 1 | 0 |

The one validated causal edge:
**`chr5:140612012-140612249__H3K4me2 → CD14`** — found by *both* FCI and PC, and
ENCODE-rE2G predicts that element regulates CD14 (prediction axis). The other
peak→gene edges are reproducible across algorithms but have no target-aware support,
so they stay `recurrent_unvalidated` — exactly the honesty the TMCO6 catch demanded.
DAGMA (Gaussian) returned 0 edges on all three (consistent with the audit; it is a
sensitivity check only).

---

## Alternative paths (decision points to revisit)

Each row is a place the pipeline commits to one branch. Current choice is what the
driver defaults to; the alternative is what to swap in and re-run to compare.

| # | Stage | Current default | Alternative paths to try | Where |
| --- | --- | --- | --- | --- |
| 1 | **Aggregation** | kNN-avg (k=10) 300 kb baseline | SEACells metacells (non-overlapping) · metacell↔metacell OT · dist-weighted kNN · 1:1 pairs. Report n_eff + null calibration. | ROADMAP Phase C |
| 2 | **Algorithm set** | `fci, pc` (+`dagma` sensitivity) | **GFCI** (py-tetrad, PAG — the Q3 primary, not yet wired) · RFCI · DirectLiNGAM/RCD/CAM-UV · fGES | `--methods`; ROADMAP Phase D |
| 3 | **CI test** | `fisherz` on `rank_gaussian` | **RCoT** (nonlinearity guard) · KCI (retired — slow/brittle) · CMIknn audit | `--indep-test` |
| 4 | **Window** | 300 kb (interpretable) | 1 Mb (more distal candidates, denser/noisier) | `--dataset-root` |
| 5 | **Background knowledge** | `tiered_distal_promoter_expr` | `minimal_expr_sink` · `none` (see how much orientation is assumption-driven) | `--background-mode` |
| 6 | **Validation axes** | rE2G + PCHi-C (2 of 3) | add **eQTL Catalogue** (genetics) → true 3-axis "≥2 agree"; add peak-peak (ChIP-Atlas/Hi-C) & peak-node (BLUEPRINT ChIP) validators | `--eqtl-tsv`; ROADMAP Phase B |
| 7 | **Edge validation rule** | target-aware for peak→gene; peak-peak = unvalidated | once peak-peak validators exist, promote peak-peak to a target-aware rule | `build_node_support_table.py::records_for_edge` |
| 8 | **Post-selection screen** | off (full peak set) | elastic-net screen with `--graph-split-fraction` (disjoint screen/graph rows) to cut leakage | `causal_cli elastic-net-screen` |
| 9 | **Stability** | single run per algorithm | donor-level block-bootstrap edge-mark frequencies (Q3 headline output) | ROADMAP Phase D |
| 10 | **Node transform** | expr kept as `_log1p` column; peaks continuous | VST the gene count; drop count-family DAGMA (done) | ROADMAP Phase D |

## Known limitations / honesty ledger

- **n_eff not yet reported.** kNN-avg reuses cells → the effective sample size is
  well below the row count; fisherz p-values are optimistic until Phase C lands.
- **Only 2 of 3 peak→gene axes** are wired (prediction + physical). A single-axis
  `VALIDATED` edge is weaker than a ≥2-axis one — the report shows the axis count.
- **peak-peak edges dominate but can't be validated yet** — treat them as hypotheses.
- **Bug fixed en route (2026-07-08):** the support engine derived the target gene
  from the expression node name and kept the quant suffix (`expr__CD14_log1p → gene
  "CD14_log1p"`), which silently failed every `same_gene` match. Fixed in
  `parse_graph_node` (strip `_log1p`/`_raw_counts`/…). This is what turned the CD14
  flagship edge from a false negative into a true `VALIDATED_MULTI_ALGO`.
