# CausalDiscovery

This directory contains the active causal-discovery workflow built on top of `joint_v2` `scGLUE` integration.

## Start here

Use the single entrypoint:

- [causal_cli.py](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py)

The script directory is now organized like this:

- [causal_cli.py](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py)
  - The only script most people should run directly.
- [commands/](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/commands)
  - Implementation scripts used by the CLI.
- [legacy/](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/legacy)
  - Older scripts kept for reference only.
- [scripts/README.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/README.md)
  - Short “what to run” guide.

## Current workflow

The active pipeline is:

$$
\text{scGLUE latent space for pairing}
\rightarrow
\text{raw clean chromatin bins for quantification}
\rightarrow
\text{nearby CUT\&Tag peak dataset generation}
\rightarrow
\text{locus-level causal graph discovery}
$$

Concretely:

1. Restrict to one cell type, currently `monocyte`.
2. Build RNA-anchored one-to-one pseudo-pairs in `scGLUE` space.
3. Generate nearby CUT&Tag peak matrices per gene.
4. Keep literature-backed curated regions only as overlap annotations and evaluation references.
5. Run `PC + KCI`, `FCI + KCI`, or `DAGMA` on the resulting nearby peak matrices.

## Active data and outputs

Current active pairing:

- [harmonized_coarse__monocyte__rna_anchor](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor)

Current active nearby-peak datasets:

- [monocyte_cuttag_peak_genes](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes)
- [monocyte_cuttag_peak_genes_300kb](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb)

Current literature-aligned configs:

- [csf1r_e1e5_regions.tsv](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/configs/loci/csf1r_e1e5_regions.tsv)
- [cd14_regions678.tsv](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/configs/loci/cd14_regions678.tsv)
- [monocyte_cuttag_peak_genes.tsv](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv)

## Default graph-learning setup

The active graph runner lives in [commands/run_pc_causallearn.py](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/commands/run_pc_causallearn.py), but you should usually call it through [causal_cli.py](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py).

The current default workflow uses:

- `PC + KCI`
- `FCI + KCI`
- `DAGMA` for continuous optimization-based DAG search
- tiered background knowledge by default in the CLI
- optional rank-Gaussian transform
- optional local `--max-depth` cap

For `DAGMA`, the new family presets are:

- `gaussian_gaussian`
  - Gaussian peaks and Gaussian gene expression on the `log1p_norm` nearby-peak root
- `bernoulli_peak_nb_gene`
  - Bernoulli peaks with logit link and NB2 gene counts with log link on the `raw_counts` root
- `gaussian_nb`
  - Gaussian peaks with NB2 gene counts
- `nb_nb`
  - NB2 peaks with NB2 gene counts

The mixed-family implementation is now repo-local rather than delegated to the upstream global-loss DAGMA package. The current implementation note is in [dagma_mixed_family_implementation.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/notes/dagma_mixed_family_implementation.md).

## Recommended commands

Build the one-to-one monocyte matches:

```bash
python3 CausalDiscovery/scripts/causal_cli.py match --run-id joint_v2
```

Generate nearby CUT&Tag peak datasets:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --gene-panel CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv
```

Generate raw-count nearby peak datasets into a separate dataset root:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --gene-panel CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

Install `DAGMA` into the active environment before using `--method dagma`:

```bash
pip install "git+https://github.com/kevinsbello/dagma.git@088616885d71b56c0573cd4902c1fcbac02e649f"
```

Run one depth-limited `PC + KCI` job:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method pc \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2
```

This now writes both the graph tables and a plot in the run directory by default. Use `--no-plot` to skip the figure, or `--plot-layout spring` if you want the older force-directed layout instead of the nearby-peak local-coordinate layout.

Run one depth-limited `FCI + KCI` job:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method fci \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2
```

Run one `DAGMA` job on the same nearby-peak matrix:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config gaussian_gaussian \
  --background-mode tiered_distal_promoter_expr
```

Run one mixed-family `DAGMA` job with Bernoulli peaks and NB2 gene behavior:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config bernoulli_peak_nb_gene \
  --dagma-loss-type l2 \
  --transform none \
  --background-mode tiered_distal_promoter_expr
```

Run a full nearby-peak sweep across genes, methods, and depths for an existing dataset root:

```bash
python3 CausalDiscovery/scripts/causal_cli.py sweep \
  --window-bp 300000 \
  --depths 1,2,3,4,5
```

Run the same sweep on a raw-count dataset root that was generated earlier:

```bash
python3 CausalDiscovery/scripts/causal_cli.py sweep \
  --depths 1,2,3,4,5 \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

`DAGMA` is intentionally excluded from the depth-based `sweep`, because that command is designed around `max_depth`. Use `dagma-sweep` instead:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dagma-sweep \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb \
  --family-configs gaussian_gaussian,bernoulli_peak_nb_gene,gaussian_nb,nb_nb \
  --lambda1-values 0.03,0.01 \
  --w-threshold-values 0.3,0.1 \
  --T-values 5
```

Plot a saved graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --layout local
```

## Evaluation

Evaluation and validation ideas live in:

- [evaluation.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/evaluation.md)
- [evaluation/README.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/evaluation/README.md)
