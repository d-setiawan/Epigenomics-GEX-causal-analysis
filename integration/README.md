# Integration Module

This directory is the repo-native home for all downstream integration work.

Primary active method track:
1. `scGLUE` joint integration (RNA + all histone marks in one model)

Secondary/legacy track:
1. Pilot per-mark scripts (`*_pilot*`) for debugging and backward compatibility
2. `jianle` method stubs

## Layout
- `scripts/setup_integration_workspace.py`: shared workspace setup.
- `scripts/check_scglue_env.sh`: environment smoke-check.
- `scripts/scglue/`: ordered scGLUE joint wrappers.
- `methods/scglue/scripts/`: scGLUE Python implementations.
- `manifests/`: generated input manifests.
- `workspace/`: local symlinked input view (generated, gitignored).
- `outputs/`: all run outputs (generated, gitignored).
- `logs/`: runtime logs (generated, gitignored).

## Joint scGLUE (recommended)

From repo root:

```bash
python3 integration/scripts/setup_integration_workspace.py
bash integration/scripts/check_scglue_env.sh
```

Run the full joint pipeline:

```bash
bash integration/scripts/scglue/run_joint_pipeline.sh \
  joint_v1 \
  /absolute/path/to/gencode.v49.primary_assembly.annotation.gtf.gz \
  --graph-arg --backend --graph-arg custom \
  --graph-arg --window-bp --graph-arg 50000 \
  --graph-arg --gene-region --graph-arg promoter \
  --graph-arg --promoter-len --graph-arg 1000 \
  --graph-arg --max-chrom-features-per-mark --graph-arg 15000 \
  --graph-arg --max-edges-per-bin --graph-arg 30 \
  --graph-arg --no-export-graphml \
  --train-arg --max-epochs --train-arg 120 \
  --train-arg --compute-consistency
```

Output root:
- `integration/outputs/scglue/joint/<RUN_ID>/`

Expected stage outputs:
1. `preprocess/`
- `rna_preprocessed.h5ad`
- `chrom_<MARK>_preprocessed.h5ad`
- `chrom_preprocessed_manifest.tsv`
- `joint_preprocess_summary.json`
2. `graph/`
- `guidance_edges.tsv`
- `guidance_nodes.tsv`
- `guidance_summary.json`
3. `train/`
- `scglue_joint_model.dill`
- `all_cells_glue_embeddings.tsv`
- `modality_outputs.tsv`
- `feature_glue_embeddings.tsv`
- `train_summary.json`
4. `train/validation/`
- `cells_umap_clusters.tsv`
- `validation_metrics.json`
- `umap_by_modality.png`
- `umap_by_leiden.png`

## Causal discovery export

Build PC-ready inputs for target genes:

```bash
bash integration/scripts/scglue/run_export_causal_inputs_joint.sh \
  joint_v1 \
  MYC,TP53,GATA1
```

Outputs:
- `integration/outputs/scglue/joint/<RUN_ID>/train/causal_inputs/<GENE>_pc_input.tsv`
- `integration/outputs/scglue/joint/<RUN_ID>/train/causal_inputs/target_gene_linked_features.tsv`
- `integration/outputs/scglue/joint/<RUN_ID>/train/causal_inputs/causal_export_summary.json`
