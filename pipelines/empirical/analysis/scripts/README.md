# Scripts

The [per-gene driver](run_gene_pipeline.py) takes a locus matrix from a chosen dataset root, runs discovery and joins external support. The [stage CLI](causal_cli.py) exposes the individual operations. The examples below use generated PBMC `joint_v2` artifacts. A fresh checkout must first restore counts and train or supply an integration run. [Pipeline guide](../PIPELINE.md) · [Transfer guide](../../../../TRANSFER.md)

## What To Run

- `causal_cli.py match`: build one-to-one `scGLUE` pseudo-pairs.
- `causal_cli.py match-average`: build RNA-anchor $k$-neighbor tables for averaged chromatin modality profiles.
- `causal_cli.py dataset`: generate nearby CUT&Tag peak matrices for the active gene panel.
- `causal_cli.py filter-dataset`: copy an existing nearby-peak dataset root while keeping only peaks that pass a structural filter such as $p_\mathrm{nonzero} > 0.01$.
- `causal_cli.py elastic-net-screen`: create an expression-associated elastic-net screened dataset root.
- `causal_cli.py graph`: run one `PC`, `FCI`, or `DAGMA` graph job and write a plot.
- `causal_cli.py graph --online-support --fetch-encode-pbmc`: run graph discovery, fetch external evidence, and render the final plot with externally supported edges highlighted.
- `causal_cli.py graph --online-support --support-preset monocyte_regulatory`: run graph discovery with the default monocyte regulatory evidence bundle.
- `causal_cli.py plot --online-support --support-preset local_eperturbdb`: rebuild local ePerturbDB support from `pipelines/empirical/data/raw/external/eperturbdb` and highlight edges with matching external evidence.
- `causal_cli.py plot --online-support --support-preset monocyte_regulatory --eperturbdb-dir pipelines/empirical/data/raw/external/eperturbdb`: combine local ePerturbDB with online ENCODE/JASPAR support for an existing graph.
- `causal_cli.py sweep`: run many `PC`/`FCI` graph jobs across genes, methods, and depths.
- `causal_cli.py dagma-sweep`: run many DAGMA jobs across genes, family configs, and regularization settings.
- `causal_cli.py organize-outputs`: migrate older graph directories into the method-first output tree.
- `causal_cli.py plot`: re-plot a saved graph directory.
- `causal_cli.py plot --online-support --fetch-encode-pbmc`: fetch ENCODE Portal PBMC/monocyte peak support, write support tables, and render externally supported edges highlighted.
- `causal_cli.py plot --online-support --fetch-encode-screen`: fetch ENCODE SCREEN support, write support tables, and render externally supported edges highlighted.
- `causal_cli.py plot --online-support --fetch-jaspar`: fetch JASPAR motif support, write support tables, and render externally supported edges highlighted.
- `causal_cli.py support`: add external-support annotations to a saved graph directory from local ePerturbDB files, online ENCODE sources, JASPAR, or template TSVs.
- `causal_cli.py evaluate`: compare saved graph runs for edge stability, orientation consistency, and curated-region support.

Convenience shell scripts:

- `run_dagma_log1p_gaussian.sh`: Gaussian/Gaussian DAGMA on the log-normalized nearby-peak dataset.
- `run_dagma_rawcounts_family_presets.sh`: raw-count DAGMA presets such as Bernoulli peaks with NB2 gene counts.
- `run_1mb_average_pc_fci_grid.sh`: PC/FCI grid over depths, alphas, and `fisherz`/`kci` for the 1 Mb RNA-anchor averaged dataset.
- `run_elasticnet_screen_average_k10.sh`: cross-fit elastic-net screen for the unfiltered 1 Mb RNA-anchor averaged dataset.
- `plot_existing_graphs.sh`: re-plot completed graph directories found under `pipelines/empirical/analysis/outputs/methods`.

## Directory Layout

- [run_gene_pipeline.py](run_gene_pipeline.py) runs a multi-method locus analysis. Its optional GFCI path needs py-tetrad and JDK 21.
- [causal_cli.py](causal_cli.py) runs individual stages.
- [metacell_assign.py](metacell_assign.py) and [build_metacell_matching_dir.py](build_metacell_matching_dir.py) prepare nonoverlapping group assignments for aggregation.
- `commands/`: implementation modules used by the CLI.
- `legacy/`: older scripts retained for reference.

The active output layout is:

```text
pipelines/empirical/analysis/outputs/methods/<method>/<dataset>/<gene-or-locus>/<result-run>
```

For DAGMA:

```text
pipelines/empirical/analysis/outputs/methods/dagma/<dataset>/<family-config>/<gene-or-locus>/<result-run>
```

## Common Commands

Build matches:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py match --run-id joint_v2
```

Build RNA-anchor averaged-neighbor mappings:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate `300kb` log-normalized nearby-peak datasets:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --window-bp 300000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb
```

Generate raw-count nearby-peak datasets:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

Rebuild biological support and plot one saved graph:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py plot \
  --graph-dir "${GRAPH_DIR:?set GRAPH_DIR to a saved graph directory}" \
  --online-support \
  --support-preset monocyte_regulatory \
  --eperturbdb-dir pipelines/empirical/data/raw/external/eperturbdb \
  --layout local \
  --max-edges -1
```

This writes `node_support.tsv`, `node_support_matches.tsv`,
`edge_support.tsv`, and `support_report.md` into `<graph_dir>` before plotting.

For peak–expression support, the external source must connect the peak interval to the same target gene. The per-gene driver accepts rE2G, PCHi-C and eQTL tables for this purpose. For peak–peak candidates, the plot annotates source overlap at either endpoint. Evidence for the relationship between the two endpoints has a separate distance-matched evaluation. [Evidence sources](../../evaluation/results/external_sources.md) · [Peak–peak results](../../evaluation/results/peak_peak_validation_results.md)

Generate the RNA-anchor averaged-neighbor dataset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the current 1 Mb RNA-anchor averaged-neighbor datasets:

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

Create $p_\mathrm{nonzero} > 0.01$ filtered copies of those 1 Mb roots:

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

Create an elastic-net screened copy of the unfiltered 1 Mb averaged root:

```bash
PYTHON_BIN="$(which python)" bash pipelines/empirical/analysis/scripts/run_elasticnet_screen_average_k10.sh
```

For a quick single-gene smoke test:

```bash
PYTHON_BIN="$(which python)" \
GENES=CD14 \
N_REPEATS=1 \
N_ALPHAS=8 \
OUT_ROOT=/tmp/elasticnet_cd14_smoke_dataset \
REPORT_ROOT=/tmp/elasticnet_cd14_smoke_reports \
bash pipelines/empirical/analysis/scripts/run_elasticnet_screen_average_k10.sh
```

Run one PC job:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method pc \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one FCI job:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method fci \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one DAGMA job:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config gaussian_gaussian \
  --background-mode tiered_distal_promoter_expr
```

Run PC/FCI sweep:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py sweep \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb \
  --methods pc,fci \
  --depths 1,2,3,4,5
```

Run the 1 Mb RNA-anchor averaged PC/FCI grid:

```bash
bash pipelines/empirical/analysis/scripts/run_1mb_average_pc_fci_grid.sh
```

Re-plot completed graph runs:

```bash
FILTER=fci_fisherz MAX_EDGES=-1 bash pipelines/empirical/analysis/scripts/plot_existing_graphs.sh
```

Run DAGMA sweeps:

```bash
bash pipelines/empirical/analysis/scripts/run_dagma_log1p_gaussian.sh
bash pipelines/empirical/analysis/scripts/run_dagma_rawcounts_family_presets.sh
```

Graph results include a coordinate-aware local plot when peak coordinates are available. Use `--no-plot` to skip plotting or `causal_cli.py plot --layout spring` for the older force-directed style.

Evaluate saved graph runs:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py \
  evaluate \
  --methods pc,fci \
  --datasets joint_v2_monocyte_cuttag_peak_genes_300kb \
  --genes cd14 \
  --out-dir pipelines/empirical/evaluation/outputs/graph_stability/cd14_pc_fci_300kb
```

The evaluator reads existing `run_summary.json`, `*_edges.tsv`, and nearby-peak metadata files. It writes run-level graph metrics, normalized edge tables, pairwise Jaccard comparisons, edge-frequency summaries, and a short Markdown report.

## Screening Notes

The elastic-net/lasso layer is not a causal-discovery method by itself. It is an association screen used to reduce the number of nearby peaks before running `PC`, `FCI`, or `DAGMA`.

Current version:

- starts from the unfiltered `log1p_norm` `average_k10` dataset
- predict `expr__GENE_log1p` from nearby CUT&Tag peak columns
- prefer elastic net over pure lasso because nearby peaks are correlated
- use repeated cross-fit splits and keep the top `30` peaks per gene by selection frequency
- select elastic-net hyperparameters by average rank across held-out RMSE and $R^2$
- compare screened graphs against unfiltered, `nz005`, and `nz010` graphs

Elastic-net selection uses the expression outcome. Cross-fit scores summarize the screen. The optional `--graph-split-fraction` holds out rows for graph learning. RNA-anchored neighbor averages may still share the same underlying chromatin cells across row splits. [Screening and holdout details](../README.md#row-holdout-with---graph-split-fraction)
