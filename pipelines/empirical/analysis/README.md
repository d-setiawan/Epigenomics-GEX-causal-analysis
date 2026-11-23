# Analysis — causal discovery

↑ [Central README](../../../README.md) · [MISSION](../../../project_resources/research/MISSION.md) (why) · [ROADMAP](../../../project_resources/research/ROADMAP.md) (plan) · [PIPELINE.md](PIPELINE.md) (end-to-end per-gene driver) · [Evaluation/](../evaluation/README.md) (validation)

This stage estimates locus graphs from PBMC monocyte RNA and six separately measured CUT&Tag marks. The historical `scGLUE joint_v2` embedding supports several choices of observational unit. Among 15 genes, the 133- and 240-metacell partitions each produced 11 externally supported peak–gene edges, with no shared edge identities. Treat individual edges as candidates whose stability depends on the aggregation. [Partition stability findings](../evaluation/results/partition_stability_findings.md)

The examples below use generated embeddings, matching tables and matrix roots. The November transfer verified the source counts and preprocessing. Replaying later stages requires a compatible fitted embedding or the original cache. [Restore inputs and train](../../../TRANSFER.md) · [Verification record](../../../reproduce/VERIFICATION.md)

## Layout

```
pipelines/empirical/analysis/
├── scripts/     code
│   ├── causal_cli.py          the single CLI: match | dataset | graph | support
│   ├── run_gene_pipeline.py   end-to-end per-gene driver (see PIPELINE.md)
│   ├── commands/              the implementations the CLI dispatches to
│   ├── prototypes/            exploratory modeling probes, not part of the pipeline
│   └── legacy/                superseded scripts, kept for reference
├── configs/     gene panels and per-locus region definitions
├── outputs/     datasets, graphs, pipeline runs — gitignored
└── results/     modeling-design notes, literature reviews, competitive landscape
```

Validation of what this section produces lives in [`Evaluation/`](../evaluation/README.md).

### results/ — design notes and reviews

- [element_level_modeling_design.md](results/element_level_modeling_design.md) — the element-level modeling exploration (what was tried, what was rejected and why).
- [confounder_orientation_implementation.md](results/confounder_orientation_implementation.md) — confounder roots + mechanistic orientation, as implemented.
- [competitive_landscape_novelty.md](results/competitive_landscape_novelty.md) — ranked competitors + reviewer-objection counters.
- [target_gene_selection.md](results/target_gene_selection.md) — validation-rich target ranking + cell-type pros/cons.
- [aggregation_causal_discovery_litreview.md](results/aggregation_causal_discovery_litreview.md) — why aggregation choice matters for CI-test-based discovery.
- **[aggregation_partition_dependence_survey.md](results/aggregation_partition_dependence_survey.md)** — cited survey: partition-dependence is a Modifiable Areal Unit Problem; partition-bootstrap for single-cell causal discovery appears novel.
- **[professor_suggestions_and_boss_pivot.md](results/professor_suggestions_and_boss_pivot.md)** — score-based search (BOSS) + Markov-checker verification; the biology-first pivot.
- [dagma_mixed_family_implementation.md](results/dagma_mixed_family_implementation.md) — DAGMA count-family notes (kept as sensitivity only).
- [monocyte_candidate_loci.md](results/monocyte_candidate_loci.md) · [csf1r_locus.md](results/csf1r_locus.md) — locus-level background.

Scripts reference: [scripts/README.md](scripts/README.md) · superseded code: [scripts/legacy/README.md](scripts/legacy/README.md).

## Start Here

Use the [per-gene driver](scripts/run_gene_pipeline.py) to run several discovery methods on the same locus matrix and annotate the resulting edges. It accepts an explicit `--dataset-root`. The [pipeline guide](PIPELINE.md) covers the driver, including the separate py-tetrad GFCI route. The [stage CLI](scripts/causal_cli.py) builds a matching table, exports matrices and runs individual PC, FCI and DAGMA jobs.

The script tree is organized as:

- `scripts/causal_cli.py`: user-facing CLI.
- `scripts/commands/`: implementation modules called by the CLI.
- `scripts/legacy/`: older scripts retained for reference only.
- `scripts/run_dagma_log1p_gaussian.sh`: convenience DAGMA script for Gaussian/Gaussian runs.
- `scripts/run_dagma_rawcounts_family_presets.sh`: convenience DAGMA script for raw-count mixed-family presets.

## Observational units and inputs

| Unit used by a graph | Construction | Interpretation |
| --- | --- | --- |
| One-to-one pseudo-pair | Match each RNA anchor to a neighbor in each mark, without replacement within each mark. | Each row joins measurements from different cells at similar estimated states. |
| RNA-anchored neighbor average | Average the nearest cells in each mark for each RNA anchor. | Nearby RNA anchors can reuse chromatin cells. |
| Metacell | Assign cells to nonoverlapping groups using the embedding and aggregate each available assay within the group. | Each row estimates group means. Changing the partition can change which edges appear. |

The embedding defines candidate neighbors. The nearby-peak exporter quantifies mark-specific intervals from the clean fixed-bin signal. The graph includes the target expression and candidate peaks. Literature regions and other sources used for support enter after discovery. See the [aggregation analysis](../../../project_resources/research/aggregation_restart_plan.md) for the distinction between covariance of state means and same-cell cross-assay covariance.

## External Support Layer

The graph plotting layer can cross-reference saved edges against external
regulatory evidence and write support-aware plots:

- `eperturbdb`: local ePerturbDB Database TSV rows, using hg38 enhancer
  coordinates and `Gene (literature)` as target-gene evidence.
- `eperturbdb_enhancer_gene`: local enhancer-gene BED interval annotation.
- `eperturbdb_tf_overlap`: local enhancer-TF overlap BED context.
- `encode_pbmc`: online ENCODE Portal GRCh38 peak files for PBMC/monocyte
  biosample terms.
- `encode_screen`: online ENCODE SCREEN cCRE Registry intervals.
- `jaspar`: online JASPAR motif matrices plus UCSC hg38 sequence retrieval.

The original July evidence audit found no ePerturbDB literature-target rows for CSF1R, CD14 or IL1B, and several interval/motif sources were cell-type agnostic. For target-specific peak–gene evidence, the per-gene driver also accepts monocyte rE2G predictions, PCHi-C contacts and eQTL credible sets. The support tables used must be present or supplied when running the driver. [Original audit](../evaluation/results/run_audit_report.md) · [Evidence sources](../evaluation/results/external_sources.md) · [Driver](scripts/run_gene_pipeline.py)

Support outputs are written into each graph directory:

```text
node_support.tsv
node_support_matches.tsv
edge_support.tsv
support_report.md
```

The support engine requires both a peak interval and its external target gene for a peak–expression match. For peak–peak candidates it annotates intervals or motifs at the endpoints. The [peak–peak evaluation](../evaluation/results/peak_peak_validation_results.md) separately tests evidence for the relationship between those intervals.

Use local ePerturbDB downloads plus the monocyte regulatory online preset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py plot \
  --graph-dir "${GRAPH_DIR:?set GRAPH_DIR to a saved graph directory}" \
  --online-support \
  --support-preset monocyte_regulatory \
  --eperturbdb-dir pipelines/empirical/data/raw/external/eperturbdb \
  --layout local \
  --max-edges -1
```

For local ePerturbDB only:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py plot \
  --graph-dir "${GRAPH_DIR:?set GRAPH_DIR to a saved graph directory}" \
  --online-support \
  --support-preset local_eperturbdb
```

Plot colors use orange for ePerturbDB, purple for ENCODE, green for JASPAR, red
for multi-source support, and faint gray for unvalidated edges.

## Historical inputs and generated paths

Pairing root:

```text
pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor
```

The one-to-one pairing root has `1561` RNA-anchored pairs per CUT&Tag mark.

RNA-anchor neighbor-averaged pairing root:

```text
pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10
```

The neighbor-averaged pairing keeps all monocyte RNA anchors and averages each CUT&Tag mark over the $10$ nearest chromatin neighbors in scGLUE space. The historical averaged matrices had 1,887 RNA-anchor rows.

Gene panel:

```text
pipelines/empirical/analysis/configs/gene_panels/monocyte_cuttag_peak_genes.tsv
```

The original nearby-peak panel contains three genes.

- `CSF1R`
- `CD14`
- `IL1B`

### Dataset Roots

```text
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

The `300kb` root uses `log1p_norm` peak quantification. The `300kb_rawcounts` root uses raw count peak quantification. These are the original one-to-one pseudo-pair nearby-peak roots.

Current RNA-anchor neighbor-averaged dataset roots:

```text
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts
```

These are the broad-window roots used for the historical comparison. They use mark-specific CUT&Tag peaks within TSS $\pm 1$ Mb, quantified from the raw clean fixed-bin matrices. The first root stores normalized/log-transformed peak values, while the second stores raw peak counts.

Nonzero-filtered versions:

```text
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz005
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz005
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz010
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz010
```

These keep only nearby peak columns with:

$$
p_\mathrm{nonzero} > t
$$

where the current thresholds are $t \in \{0.01, 0.05, 0.10\}$. Retained peak counts are:

| Dataset suffix | `CSF1R` | `CD14` | `IL1B` |
| --- | ---: | ---: | ---: |
| `nz001` | `60 / 69` | `99 / 113` | `74 / 82` |
| `nz005` | `41 / 69` | `73 / 113` | `49 / 82` |
| `nz010` | `26 / 69` | `40 / 113` | `28 / 82` |

Elastic-net screened version:

```text
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_elasticnet_top30_crossfit
```

This root keeps the non-peak sample, expression, and library-size columns from the unfiltered `average_k10` dataset, then keeps up to `30` nearby CUT&Tag peaks per gene selected by cross-fit elastic-net stability. Reports are written under:

```text
pipelines/empirical/analysis/outputs/methods/elastic_net_screen/joint_v2_monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10/top30_crossfit/
```

All nearby-peak matrices include:

- `expr__GENE_log1p`
- `expr__GENE_raw_counts`
- nearby peak columns
- library-size columns used by count-family DAGMA runs

Alternative RNA-anchor neighbor-averaged dataset roots:

```text
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts
```

These use all monocyte RNA anchors and average each CUT&Tag mark over the $10$ nearest chromatin neighbors in scGLUE space.

### Dataset Semantics

Use the dataset roots this way:

- `log1p_norm` roots are preferred for Gaussian-style graph discovery, `PC + fisherz`, and elastic-net screening.
- `rawcounts` roots are preferred for count-family DAGMA, especially Bernoulli peak or NB2 gene-expression presets.
- `nz001`, `nz005`, and `nz010` roots are structural-filtered roots. They remove sparse peak columns, but they do not use expression association, curated-region overlap, or motif evidence.
- Curated promoter/enhancer regions are evaluation annotations only. They are not graph variables in the current nearby-peak workflow.

Raw counts are not directly comparable across cells because library sizes differ. The raw-count roots therefore keep `libsize__...` columns so count-family models can use offsets. The normalized roots are generated from the same raw fixed-bin signal using library-size normalization and log transformation.

### Screening Plan

The expression-informed screen uses elastic-net stability selection. The screen fits each gene separately:

$$
\mathrm{expr}_{g,\log 1p} \sim \text{nearby CUT\&Tag peaks}
$$

on the unfiltered `log1p_norm` `average_k10` matrix. Elastic net is preferred over pure lasso because nearby peaks are correlated by genomic distance and chromatin state; pure lasso can pick one arbitrary proxy from a correlated group, while elastic net can retain groups more smoothly.

The screen estimates selection frequency across repeated subsamples:

$$
\pi_j = \Pr(\hat{\beta}_j \ne 0)
$$

and keeps the top `30` peaks per gene by selection frequency, with coefficient magnitude and genomic coordinate/name as deterministic tie-breakers. This should be treated as candidate reduction, not causal evidence.

Important inference caveat:

If the same samples are used to select peaks and then run `PC`, `FCI`, or `DAGMA`, the downstream conditional-independence tests or graph scores are no longer clean post-selection inference. The graph learner is no longer operating on a pre-specified variable set. Instead, it sees a selected set:

$$
S = f(X, y)
$$

where $S$ depends on the same data later used for graph discovery. This can make downstream edges look more stable or more expression-linked than they would under a fresh independent sample.

Practical mitigations:

- treat elastic net as an exploratory candidate-reduction screen
- report the screening rule and retained peak counts
- run graph discovery across several screening thresholds
- where possible, split or cross-fit the data so one split screens peaks and another split learns the graph
- always compare screened graphs against unfiltered, `nz005`, `nz010`, and curated-region overlap annotations

Run the elastic-net screen with:

```bash
PYTHON_BIN="$(which python)" bash pipelines/empirical/analysis/scripts/run_elasticnet_screen_average_k10.sh
```

#### Row holdout with `--graph-split-fraction`

By default the screen keeps every row in the screened matrix, so peak selection and
graph discovery share samples (exploratory only; the existing `..._elasticnet_top30_crossfit`
runs use this design). The following command reserves graph rows before peak selection.

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py elastic-net-screen \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10 \
  --top-k 30 \
  --graph-split-fraction 0.3
```

With `--graph-split-fraction 0.3`, the screen uses 70% of the rows and the graph uses the other 30%. The seed (`--split-random-state`), per-gene assignments (`*_elasticnet_sample_split.tsv`) and `sample_split` in each `run_summary.json` record that partition. The derived root gains a `_graphsplit<pct>` suffix. In an RNA-anchored neighbor average, two different rows can contain measurements from the same chromatin cell. A holdout of rows alone therefore leaves uncertainty about the calibration of subsequent tests.

### Interpreting the screening comparisons

The `nz001`, `nz005` and `nz010` roots test sensitivity to sparse-peak filtering. The unfiltered and elastic-net roots test sensitivity to expression-informed selection. Results about a particular peak also depend on which cells enter each aggregate. [Metacell partition comparison](../evaluation/results/partition_stability_findings.md) · [Empirical null calibration](../evaluation/results/null_calibration_findings.md)

## Output Layout

Dataset matrices stay under:

```text
pipelines/empirical/analysis/outputs/datasets/<run-id>/<dataset-name>/<gene>/nearby_peaks/
```

Graph results go under:

```text
pipelines/empirical/analysis/outputs/methods/<method>/<dataset>/<gene>/<result-run>/
```

For DAGMA, the family configuration adds one extra layer:

```text
pipelines/empirical/analysis/outputs/methods/dagma/<dataset>/<family-config>/<gene>/<result-run>/
```

This keeps outputs readable as:

$$
\text{method} \rightarrow \text{dataset} \rightarrow \text{gene} \rightarrow \text{result}
$$

and, for DAGMA:

$$
\text{dagma} \rightarrow \text{dataset} \rightarrow \text{family config} \rightarrow \text{gene} \rightarrow \text{result}
$$

## Graph Methods

The active graph runner supports:

- `PC + fisherz` and `PC + KCI`
- `FCI + fisherz` and `FCI + KCI`
- repo-local mixed-family `DAGMA`

The default biological background mode is `tiered_distal_promoter_expr`, which prevents gene-to-peak edges while allowing peak-to-peak and peak-to-gene edges. A peak-to-expression orientation may therefore reflect an imposed restriction.

The DAGMA implementation lives in:

- `scripts/commands/dagma_mixed_family.py`
- `scripts/commands/run_pc_causallearn.py`
- `results/dagma_mixed_family_implementation.md`

DAGMA family presets:

- `gaussian_gaussian`: Gaussian peaks and Gaussian gene expression on the `log1p_norm` root.
- `bernoulli_peak_nb_gene`: Bernoulli peaks with logit link and NB2 gene counts with log link on the `raw_counts` root.
- `gaussian_nb`: Gaussian peaks with NB2 gene counts.
- `nb_nb`: NB2 peaks with NB2 gene counts.

The `--dagma-loss-type` CLI argument remains for compatibility, but the important model choice is now `--dagma-family-config`.

## Core Commands

Build monocyte one-to-one matches:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py match --run-id joint_v2
```

Build RNA-anchor $k$-neighbor averaging tables:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate `300kb` nearby CUT&Tag peak datasets with log-normalized peak quantification:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --gene-panel pipelines/empirical/analysis/configs/gene_panels/monocyte_cuttag_peak_genes.tsv \
  --window-bp 300000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb
```

Generate the raw-count version:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --gene-panel pipelines/empirical/analysis/configs/gene_panels/monocyte_cuttag_peak_genes.tsv \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

Generate an RNA-anchor neighbor-averaged dataset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the current 1 Mb RNA-anchor neighbor-averaged datasets:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 1000000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10

python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 1000000 \
  --quant-mode raw_counts \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts
```

Create $p_\mathrm{nonzero} > 0.01$ filtered copies:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py filter-dataset \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001 \
  --min-nonzero-fraction 0.01

python3 pipelines/empirical/analysis/scripts/causal_cli.py filter-dataset \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001 \
  --min-nonzero-fraction 0.01
```

Run one `PC + KCI` graph:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method pc \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one `FCI + KCI` graph:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method fci \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one Gaussian DAGMA graph:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config gaussian_gaussian \
  --background-mode tiered_distal_promoter_expr
```

Run one raw-count mixed-family DAGMA graph:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config bernoulli_peak_nb_gene \
  --transform none \
  --background-mode tiered_distal_promoter_expr
```

## Sweeps

Run a PC/FCI depth sweep:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py sweep \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb \
  --methods pc,fci \
  --depths 1,2,3,4,5 \
  --background-mode tiered_distal_promoter_expr
```

Run Gaussian DAGMA sweep on log-normalized peaks:

```bash
bash pipelines/empirical/analysis/scripts/run_dagma_log1p_gaussian.sh
```

Run raw-count DAGMA family presets:

```bash
bash pipelines/empirical/analysis/scripts/run_dagma_rawcounts_family_presets.sh
```

Or call the DAGMA sweep directly:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dagma-sweep \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts \
  --family-configs bernoulli_peak_nb_gene,nb_nb \
  --lambda1-values 0.03,0.01 \
  --w-threshold-values 0.3,0.1 \
  --T-values 5 \
  --background-mode tiered_distal_promoter_expr
```

`DAGMA` is intentionally excluded from the PC/FCI `sweep` because it does not use `max_depth`.

## Plotting And Evaluation

Graph runs write plots by default. Re-plot a saved graph with:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py plot \
  --graph-dir pipelines/empirical/analysis/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --layout local
```

The [evaluation guide](../evaluation/README.md), [graph evaluation protocol](../evaluation/results/graph_evaluation_pipeline.md) and [literature review](../evaluation/results/literature_review_graph_evaluation.md) distinguish target-specific evidence, interval overlap and partition stability.

Saved graph runs can be compared with:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py \
  evaluate \
  --methods pc,fci \
  --datasets joint_v2_monocyte_cuttag_peak_genes_300kb \
  --genes cd14 \
  --out-dir pipelines/empirical/evaluation/outputs/graph_stability/cd14_pc_fci_300kb
```

This writes edge-frequency, pairwise Jaccard, orientation-consistency, and curated-region-support summaries from existing graph output directories.
