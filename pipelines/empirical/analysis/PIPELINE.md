# Per-gene discovery and external evidence in PBMC monocytes

↑ [Central README](../../../README.md) · [MISSION](../../../project_resources/research/MISSION.md) (why) · [ROADMAP](../../../project_resources/research/ROADMAP.md) (plan) · [Analysis/README.md](README.md) (stage detail)

For a target gene, this driver estimates a graph on a chosen nearby-peak matrix,
looks for external annotations of the proposed relationships and reports
which algorithms select the same pair. The RNA and histone assays sampled
different cells. An external annotation can support a locus hypothesis, while
the direction of the underlying same-cell relationship remains unresolved.
[Per-gene driver](scripts/run_gene_pipeline.py) · [Panel summary](scripts/summarize_pipeline_panel.py) · [Aggregation units](README.md#observational-units-and-inputs)

---

## Run the driver

Use a generated locus matrix and the empirical environment. Specify
`--dataset-root` to choose the aggregation. The three external evidence tables
are used when available locally or explicitly supplied with `--re2g-tsv`,
`--pchic-tsv` and `--eqtl-tsv`. [Data and evidence sources](../data/README.md)

```bash
python pipelines/empirical/analysis/scripts/run_gene_pipeline.py --gene CD14 --methods fci,pc,dagma
# After running the driver for CD14, CSF1R and IL1B:
python pipelines/empirical/analysis/scripts/summarize_pipeline_panel.py --genes CD14,CSF1R,IL1B
```

1. **Discovery.** The driver calls `causal_cli graph` for FCI, PC or DAGMA on the selected matrix. Optional GFCI has a separate py-tetrad runner.
2. **Annotation.** For each graph, `causal_cli support` queries the configured evidence tables. ENCODE-rE2G supplies regulatory predictions, BLUEPRINT PCHi-C supplies physical contacts, and eQTL Catalogue credible sets link genetic variation to expression.
3. **Consensus.** The driver groups unordered pairs across methods and records their individual orientations and evidence sources. Per-gene outputs include `edges_consensus.tsv`, `edges_per_method.tsv`, `pipeline_report.md` and `pipeline_manifest.json`.

Outputs land in `pipelines/empirical/analysis/outputs/pipeline/<GENE>/`.

## Reading the support verdicts

| Candidate | Rule in the output | What a match establishes |
| --- | --- | --- |
| Peak–gene | `region_interval_with_matching_target_gene` | An external record overlaps the peak interval and names the same gene. The rule checks the interval and gene irrespective of the fitted arrow. |
| Peak–peak | `region_interval_on_either_endpoint` | At least one of the two intervals has an annotation. The driver assigns `no_target_validator` to the pair. |

For peak–gene candidates, `VALIDATED_MULTI_ALGO` means a target-specific match from a recognized source (rE2G, PCHi-C or eQTL) and selection by at least two discovery methods. `VALIDATED` means such a match with one method. Without a recognized match, the driver emits `recurrent_unvalidated` for multiple methods and `unvalidated` for one. It reports the number of evidence types separately in `n_axes`. These labels leave the effect of manipulating a histone mark on expression untested. [Support rules](scripts/commands/build_node_support_table.py) · [Verdict implementation](scripts/run_gene_pipeline.py)

The dedicated peak–peak analysis compares evidence on the relationship between intervals with distance-matched pairs. The partition study asks whether the same supported peak–gene pairs recur when the cells are grouped differently. [Peak–peak evaluation](../evaluation/results/peak_peak_validation_results.md) · [Partition stability](../evaluation/results/partition_stability_findings.md)

## Recorded results

- **[expanded_panel_confounder_results.md](../evaluation/results/expanded_panel_confounder_results.md)**
  — results for the 15-gene panel conditioned on the specified confounders.
- **[peak_peak_validation_results.md](../evaluation/results/peak_peak_validation_results.md)**
  — element↔element axes and the co-accessibility contrast.
- **[run_audit_report.md](../evaluation/results/run_audit_report.md)** — saved-run audit, rigor tiers.

---

## Alternative paths (decision points to revisit)

Each row is a place the pipeline commits to one branch. Current choice is what the
driver defaults to; the alternative is what to swap in and re-run to compare.

| # | Stage | Current default | Alternative paths to try | Where |
| --- | --- | --- | --- | --- |
| 1 | **Aggregation** | Generated `monocyte_cuttag_peak_genes_300kb` matrix by default. Its matching table determines the unit. | Compare one-to-one pairs, RNA-anchor averages and nonoverlapping metacells. | `--dataset-root` |
| 2 | **Algorithm set** | `fci,pc`, with DAGMA available for sensitivity analysis | The GFCI runner supplies a skeleton comparison. FCI endpoint marks depend on the observations and orientation constraints. [GFCI limitation](../../../project_resources/research/LIMITATIONS.md#l10-gfci-orients-the-skeleton-only) | `--methods` |
| 3 | **CI test** | `fisherz` on `rank_gaussian` | **RCoT** (nonlinearity guard) · KCI (retired — slow/brittle) · CMIknn audit | `--indep-test` |
| 4 | **Window** | 300 kb (interpretable) | 1 Mb (more distal candidates, denser/noisier) | `--dataset-root` |
| 5 | **Background knowledge** | `tiered_distal_promoter_expr` | `minimal_expr_sink` · `none` (see how much orientation is assumption-driven) | `--background-mode` |
| 6 | **Evidence axes** | rE2G, PCHi-C and eQTL tables when available | Check whether the eQTL context matches the cell state under study. | `--eqtl-tsv` |
| 7 | **Evidence rules** | Target-specific for peak–gene, interval context for peak–peak | Use the separate [peak–peak evaluation](../evaluation/results/peak_peak_validation_results.md) for pair-specific evidence. | `build_node_support_table.py::records_for_edge` |
| 8 | **Post-selection screen** | off (full peak set) | elastic-net screen with `--graph-split-fraction` (disjoint screen/graph rows) to cut leakage | `causal_cli elastic-net-screen` |
| 9 | **Stability** | bootstrap stability selection (row-subsampling) is the report headline | cell-state/cluster block bootstrap — *donor blocks are unavailable* ([L1](../../../project_resources/research/LIMITATIONS.md#l1-donor-is-not-an-available-variable)) | ROADMAP Phase D |
| 10 | **Node transform** | expr kept as `_log1p` column; peaks continuous | VST the gene count; drop count-family DAGMA (done) | ROADMAP Phase D |

## Known limitations

Caveats are **not** duplicated here — they live in
**[project_resources/research/LIMITATIONS.md](../../../project_resources/research/LIMITATIONS.md)**. The ones that bite this driver hardest:
[L3 small n / n_eff](../../../project_resources/research/LIMITATIONS.md#l3-small-n-and-n_eff-is-not-reported) ·
[L4 stability is row-subsampling](../../../project_resources/research/LIMITATIONS.md#l4-stability-is-row-subsampling-not-block-bootstrap) ·
[L5 eQTL state mismatch](../../../project_resources/research/LIMITATIONS.md#l5-the-genetics-axis-is-cell-type-aligned-but-not-always-state-aligned) ·
[L10 GFCI orients skeleton only](../../../project_resources/research/LIMITATIONS.md#l10-gfci-orients-the-skeleton-only).

In the July 8 historical run, the support engine initially read the expression
node `expr__CD14_log1p` as the gene name `CD14_log1p`, so it missed a matching
external record. After `parse_graph_node` stripped the quantification suffix,
the CD14 candidate received `VALIDATED_MULTI_ALGO` under the rule above.
