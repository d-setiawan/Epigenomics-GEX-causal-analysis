# Scripts

Use [causal_cli.py](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py) for the active workflow.

## What to use

- `causal_cli.py match`
  - Build one-to-one `scGLUE` pseudo-pairs.
- `causal_cli.py dataset`
  - Bulk-generate nearby CUT&Tag peak matrices for a gene panel.
- `causal_cli.py graph`
  - Run one `PC`, `FCI`, or `DAGMA` job on a matrix and write a graph plot by default.
- `causal_cli.py sweep`
  - Reuse an existing dataset root and run many `PC`/`FCI` graph jobs across genes, methods, and depths, with per-run plots by default.
- `causal_cli.py dagma-sweep`
  - Reuse an existing dataset root and run many `DAGMA` graph jobs across genes and DAGMA family or regularization settings.
- `causal_cli.py plot`
  - Re-plot a saved graph directory with the local-coordinate or spring layout.
- `causal_cli.py support`
  - Add external-support annotations for a saved `PC` graph directory.

## Directory layout

- [commands/](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/commands)
  - Implementation scripts used by the CLI.
- [legacy/](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/legacy)
  - Older scripts kept for reference only.

## Examples

```bash
python3 CausalDiscovery/scripts/causal_cli.py match --run-id joint_v2
python3 CausalDiscovery/scripts/causal_cli.py dataset --run-id joint_v2 --window-bp 300000
python3 CausalDiscovery/scripts/causal_cli.py graph --method pc --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv --max-depth 2
python3 CausalDiscovery/scripts/causal_cli.py graph --method dagma --dagma-family-config gaussian_gaussian --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv
python3 CausalDiscovery/scripts/causal_cli.py sweep --window-bp 300000 --depths 1,2 --alpha 0.1
python3 CausalDiscovery/scripts/causal_cli.py dagma-sweep --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb --family-configs gaussian_gaussian,bernoulli_peak_nb_gene
```

If you want to keep multiple quantification modes side by side, give `dataset` separate `--out-root` paths and point `sweep` at the one you want with `--dataset-root`.

Nearby-peak graphs use a coordinate-aware local layout automatically when possible, so the gene stays at the bottom and CUT&Tag peaks are arranged by genomic position across the locus. Use `--no-plot` to skip figure generation, or `causal_cli.py plot --layout spring` if you want the older force-directed view.

For `DAGMA`, the repo-local mixed-family runner now supports exact node-family scoring for the currently exposed families:

- `gaussian_gaussian`
- `bernoulli_peak_nb_gene`
- `gaussian_nb`
- `nb_nb`

The implementation note is in [dagma_mixed_family_implementation.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/notes/dagma_mixed_family_implementation.md).

`DAGMA` is not part of the depth-based `sweep`; use `dagma-sweep` for batch DAGMA runs instead.
