# End-to-End Causal Pipeline (test a gene → validate its edges)

↑ [Central README](../README.md) · [MISSION](../docs/MISSION.md) (why) · [ROADMAP](../docs/ROADMAP.md) (plan) · [CausalDiscovery/README.md](README.md) (stage detail)

This is the runnable spine that ties the whole project together: **give it a gene,
it runs several causal-discovery algorithms and tells you which of their edges the
evaluation suite can actually validate.** It is the concrete answer to the project
question and the thing to grow as Phases C/D land.

Driver: [`scripts/run_gene_pipeline.py`](scripts/run_gene_pipeline.py) ·
Panel roll-up: [`scripts/summarize_pipeline_panel.py`](scripts/summarize_pipeline_panel.py) ·
Method decisions & phase plan: [`../ROADMAP.md`](../docs/ROADMAP.md).

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
   evaluation suite, now 3-axis: **ENCODE-rE2G** (prediction) + **BLUEPRINT PCHi-C**
   (physical) + **eQTL Catalogue** SuSiE credible sets (genetics).
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

> peak-peak edges are the majority of what discovery returns. The *support suite* still
> reports them as `no_target_validator` (node-existence only) — a target-aware peak-peak
> validator does not live here. Element↔element claims are instead tested by the dedicated
> framework in [`Evaluation/`](../Evaluation/README.md): a distance-matched permutation
> null across five axes.

## Current results

**This doc carries no result numbers** — they go stale. The results this driver produces
are owned by:

- **[expanded_panel_confounder_results.md](../Evaluation/results/expanded_panel_confounder_results.md)**
  — current default-pipeline headline (15-gene panel, confounder-conditioned).
- **[peak_peak_validation_results.md](../Evaluation/results/peak_peak_validation_results.md)**
  — element↔element axes and the co-accessibility contrast.
- **[run_audit_report.md](../Evaluation/results/run_audit_report.md)** — saved-run audit, rigor tiers.

---

## Alternative paths (decision points to revisit)

Each row is a place the pipeline commits to one branch. Current choice is what the
driver defaults to; the alternative is what to swap in and re-run to compare.

| # | Stage | Current default | Alternative paths to try | Where |
| --- | --- | --- | --- | --- |
| 1 | **Aggregation** | kNN-avg (k=10) 300 kb baseline | SEACells metacells (non-overlapping) · metacell↔metacell OT · dist-weighted kNN · 1:1 pairs. Report n_eff + null calibration. | ROADMAP Phase C |
| 2 | **Algorithm set** | `fci, pc` (+`dagma` sensitivity) | **GFCI is wired** (`--methods gfci`, py-tetrad/JDK21) but orients the skeleton only, so directed claims come from FCI ([L10](../docs/LIMITATIONS.md#l10-gfci-orients-the-skeleton-only)) · RFCI · DirectLiNGAM/RCD/CAM-UV · fGES | `--methods`; ROADMAP Phase D |
| 3 | **CI test** | `fisherz` on `rank_gaussian` | **RCoT** (nonlinearity guard) · KCI (retired — slow/brittle) · CMIknn audit | `--indep-test` |
| 4 | **Window** | 300 kb (interpretable) | 1 Mb (more distal candidates, denser/noisier) | `--dataset-root` |
| 5 | **Background knowledge** | `tiered_distal_promoter_expr` | `minimal_expr_sink` · `none` (see how much orientation is assumption-driven) | `--background-mode` |
| 6 | **Validation axes** | rE2G + PCHi-C + **eQTL** (all 3 peak→gene axes wired) | add peak-peak (ChIP-Atlas/Hi-C) & peak-node (BLUEPRINT ChIP) validators; state-match the eQTL (resting vs TLR-stimulated) | `--eqtl-tsv`; ROADMAP Phase B |
| 7 | **Edge validation rule** | target-aware for peak→gene; peak-peak = unvalidated | once peak-peak validators exist, promote peak-peak to a target-aware rule | `build_node_support_table.py::records_for_edge` |
| 8 | **Post-selection screen** | off (full peak set) | elastic-net screen with `--graph-split-fraction` (disjoint screen/graph rows) to cut leakage | `causal_cli elastic-net-screen` |
| 9 | **Stability** | bootstrap stability selection (row-subsampling) is the report headline | cell-state/cluster block bootstrap — *donor blocks are unavailable* ([L1](../docs/LIMITATIONS.md#l1-donor-is-not-an-available-variable)) | ROADMAP Phase D |
| 10 | **Node transform** | expr kept as `_log1p` column; peaks continuous | VST the gene count; drop count-family DAGMA (done) | ROADMAP Phase D |

## Known limitations

Caveats are **not** duplicated here — they live in
**[docs/LIMITATIONS.md](../docs/LIMITATIONS.md)**. The ones that bite this driver hardest:
[L3 small n / n_eff](../docs/LIMITATIONS.md#l3-small-n-and-n_eff-is-not-reported) ·
[L4 stability is row-subsampling](../docs/LIMITATIONS.md#l4-stability-is-row-subsampling-not-block-bootstrap) ·
[L5 eQTL state mismatch](../docs/LIMITATIONS.md#l5-the-genetics-axis-is-cell-type-aligned-but-not-always-state-aligned) ·
[L10 GFCI orients skeleton only](../docs/LIMITATIONS.md#l10-gfci-orients-the-skeleton-only).

**Bug fixed en route (2026-07-08)** — kept because it explains a verdict change: the support engine derived the target gene
  from the expression node name and kept the quant suffix (`expr__CD14_log1p → gene
  "CD14_log1p"`), which silently failed every `same_gene` match. Fixed in
  `parse_graph_node` (strip `_log1p`/`_raw_counts`/…). This is what turned the CD14
  flagship edge from a false negative into a true `VALIDATED_MULTI_ALGO`.
