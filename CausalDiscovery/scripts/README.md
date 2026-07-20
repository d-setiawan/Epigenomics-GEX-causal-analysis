# Scripts

Use `causal_cli.py` for the active workflow.

## What To Run

- `causal_cli.py match`: build one-to-one `scGLUE` pseudo-pairs.
- `causal_cli.py match-average`: build RNA-anchor $k$-neighbor tables for averaged chromatin modality profiles.
- `causal_cli.py dataset`: generate nearby CUT&Tag peak matrices for the active gene panel.
- `causal_cli.py filter-dataset`: copy an existing nearby-peak dataset root while keeping only peaks that pass a structural filter such as $p_\mathrm{nonzero} > 0.01$.
- `causal_cli.py elastic-net-screen`: create an expression-associated elastic-net screened dataset root.
- `causal_cli.py graph`: run one `PC`, `FCI`, or `DAGMA` graph job and write a plot.
- `causal_cli.py graph --online-support --fetch-encode-pbmc`: run graph discovery, fetch external evidence, and render the final plot with validated edges highlighted.
- `causal_cli.py graph --online-support --support-preset monocyte_regulatory`: run graph discovery with the default monocyte regulatory evidence bundle.
- `causal_cli.py plot --online-support --support-preset local_eperturbdb`: rebuild local ePerturbDB support from `Data/raw/external/eperturbdb` and render validated/support-highlighted graph edges.
- `causal_cli.py plot --online-support --support-preset monocyte_regulatory --eperturbdb-dir Data/raw/external/eperturbdb`: combine local ePerturbDB with online ENCODE/JASPAR support for an existing graph.
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
- `plot_existing_graphs.sh`: re-plot completed graph directories found under `CausalDiscovery/outputs/methods`.

## Directory Layout

- `causal_cli.py`: the only script most users should call directly.
- `commands/`: implementation modules used by the CLI.
- `legacy/`: older scripts retained for reference.

The active output layout is:

```text
CausalDiscovery/outputs/methods/<method>/<dataset>/<gene-or-locus>/<result-run>
```

For DAGMA:

```text
CausalDiscovery/outputs/methods/dagma/<dataset>/<family-config>/<gene-or-locus>/<result-run>
```

## Common Commands

Build matches:

```bash
python3 CausalDiscovery/scripts/causal_cli.py match --run-id joint_v2
```

Build RNA-anchor averaged-neighbor mappings:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate `300kb` log-normalized nearby-peak datasets:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --window-bp 300000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb
```

Generate raw-count nearby-peak datasets:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

Rebuild biological support and plot one saved graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir <graph_dir> \
  --online-support \
  --support-preset monocyte_regulatory \
  --eperturbdb-dir Data/raw/external/eperturbdb \
  --layout local \
  --max-edges -1
```

This writes `node_support.tsv`, `node_support_matches.tsv`,
`edge_support.tsv`, and `support_report.md` into `<graph_dir>` before plotting.

Edge validation rules:

- Peak-to-expression: requires peak interval support plus a matching external
  target gene, mainly from ePerturbDB.
- Peak-to-peak: accepts interval or motif support from ENCODE, JASPAR, or
  ePerturbDB TF-overlap records.
- Unvalidated edges remain in the plot but are drawn faint gray.

Generate the RNA-anchor averaged-neighbor dataset:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the current 1 Mb RNA-anchor averaged-neighbor datasets:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 1000000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10

python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 1000000 \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts
```

Create $p_\mathrm{nonzero} > 0.01$ filtered copies of those 1 Mb roots:

```bash
python3 CausalDiscovery/scripts/causal_cli.py filter-dataset \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001 \
  --min-nonzero-fraction 0.01

python3 CausalDiscovery/scripts/causal_cli.py filter-dataset \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001 \
  --min-nonzero-fraction 0.01
```

Create an elastic-net screened copy of the unfiltered 1 Mb averaged root:

```bash
PYTHON_BIN="$(which python)" bash CausalDiscovery/scripts/run_elasticnet_screen_average_k10.sh
```

For a quick single-gene smoke test:

```bash
PYTHON_BIN="$(which python)" \
GENES=CD14 \
N_REPEATS=1 \
N_ALPHAS=8 \
OUT_ROOT=/tmp/elasticnet_cd14_smoke_dataset \
REPORT_ROOT=/tmp/elasticnet_cd14_smoke_reports \
bash CausalDiscovery/scripts/run_elasticnet_screen_average_k10.sh
```

Run one PC job:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method pc \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one FCI job:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method fci \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one DAGMA job:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config gaussian_gaussian \
  --background-mode tiered_distal_promoter_expr
```

Run PC/FCI sweep:

```bash
python3 CausalDiscovery/scripts/causal_cli.py sweep \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb \
  --methods pc,fci \
  --depths 1,2,3,4,5
```

Run the 1 Mb RNA-anchor averaged PC/FCI grid:

```bash
bash CausalDiscovery/scripts/run_1mb_average_pc_fci_grid.sh
```

Re-plot completed graph runs:

```bash
FILTER=fci_fisherz MAX_EDGES=-1 bash CausalDiscovery/scripts/plot_existing_graphs.sh
```

Run DAGMA sweeps:

```bash
bash CausalDiscovery/scripts/run_dagma_log1p_gaussian.sh
bash CausalDiscovery/scripts/run_dagma_rawcounts_family_presets.sh
```

Graph results include a coordinate-aware local plot when peak coordinates are available. Use `--no-plot` to skip plotting or `causal_cli.py plot --layout spring` for the older force-directed style.

Evaluate saved graph runs:

```bash
python3 CausalDiscovery/scripts/causal_cli.py \
  --python-bin /home/dgsetiawan/anaconda3/bin/python \
  evaluate \
  --methods pc,fci \
  --datasets joint_v2_monocyte_cuttag_peak_genes_300kb \
  --genes cd14 \
  --out-dir Evaluation/outputs/graph_stability/cd14_pc_fci_300kb
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

Do not interpret an elastic-net-selected peak as causal by itself. Selection uses the expression outcome, so downstream graph inference should be treated as post-selection and exploratory unless a held-out or cross-fit graph-learning step is used.
