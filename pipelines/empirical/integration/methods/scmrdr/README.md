# Jianle / scMRDR Method Track

The PBMC comparison used the repository's Jianle approximation. The later integration recoverability benchmark used a separate adapter derived from the published model. Choose the implementation that matches the result being reproduced.

| Implementation | Data and evidence |
| --- | --- |
| Historical shell pipeline in this directory | The saved `scmrdr_scMRDR_gaussian` PBMC run was compared with scGLUE `joint_v2`. Analysis of the protein panel found discrepancies in cross-mark neighborhoods in both methods. [Alignment report](../../results/biological_alignment_2026-09-16.md) |
| [Paper-consistent adapter](paper_faithful/README.md) | The simulation benchmark used this implementation and its frozen shared-coordinate export. The [model and objective review](../../../../../project_resources/research/integration_identifiability_2026_11_01/reviews/modern_scmrdr.md) records the inspected source and adapter-guide hashes. See the [execution record](../../../../simulation/integration_recoverability/IMPLEMENTATION_STATUS.md) for the scientific readouts and outstanding assessment. |

To run the paper adapter's focused checks in an activated empirical environment from the repository root, use `python -m unittest discover -s pipelines/empirical/integration/methods/scmrdr/paper_faithful/tests -p 'test_*.py'`. See the [environment guide](../../../../../envs/README.md) for dependencies.

## What This Track Implements

The Jianle / scMRDR path includes:

- Joint preprocessing across RNA and all CUT&Tag marks.
- scMRDR-style training with reconstruction, latent alignment, and geometry-preservation objectives.
- Shared-gene feature-universe construction from histone peak matrices.
- Optional feature-availability masks for shared-gene training.
- Optional decoder covariates when measured. The empirical sources have no verified donor matching across RNA and chromatin.
- Validation outputs modeled after the scGLUE validation reports.

This track does not currently provide the active graph-discovery export.

## Scripts

Python implementations:

- [preprocess_joint_jianle.py](preprocess_joint_jianle.py)
- [train_joint_jianle.py](train_joint_jianle.py)
- [validate_joint_jianle.py](validate_joint_jianle.py)
- [build_joint_gene_feature_universe.py](build_joint_gene_feature_universe.py)

Shell wrappers from repo root:

- `pipelines/empirical/integration/methods/scmrdr/run_preprocess_joint.sh`
- `pipelines/empirical/integration/methods/scmrdr/run_train_joint.sh`
- `pipelines/empirical/integration/methods/scmrdr/run_validate_joint.sh`
- `pipelines/empirical/integration/methods/scmrdr/run_joint_pipeline.sh`
- `pipelines/empirical/integration/methods/scmrdr/run_build_gene_feature_universe.sh`

## Running The Comparison Workflow

Restore and preprocess the PBMC counts following the [transfer guide](../../../../../TRANSFER.md). The command below regenerates the input manifest and workspace symlinks after a move.

```bash
python3 pipelines/empirical/integration/scripts/setup_integration_workspace.py
```

For the historical approximation, use a fresh run ID for either the stage-by-stage example or the combined wrapper.

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_preprocess_joint.sh pbmc_scmrdr_replay
bash pipelines/empirical/integration/methods/scmrdr/run_train_joint.sh pbmc_scmrdr_replay --max-epochs 200
bash pipelines/empirical/integration/methods/scmrdr/run_validate_joint.sh pbmc_scmrdr_replay --transfer-labels
```

Or run the wrapper:

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_joint_pipeline.sh pbmc_scmrdr_replay \
  --train-arg --max-epochs --train-arg 200 \
  --train-arg --learning-rate --train-arg 1e-3 \
  --train-arg --batch-size --train-arg 256
```

## Shared Gene Feature Universe

The gene-feature helper converts histone peak matrices into gene-level feature matrices. This is useful for Jianle experiments that require a common RNA-like feature axis across modalities.

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_build_gene_feature_universe.sh \
  gene_space_v1 \
  pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz \
  --gene-universe-mode union_linked \
  --gene-region promoter \
  --promoter-len 2000 \
  --weight-mode binary
```

Expected outputs:

- `pipelines/empirical/integration/runs/scmrdr_gene_features_<RUN_ID>/gene_universe.tsv`
- `pipelines/empirical/integration/runs/scmrdr_gene_features_<RUN_ID>/joint_gene_feature_manifest.tsv`
- `pipelines/empirical/integration/runs/scmrdr_gene_features_<RUN_ID>/gene_feature_summary.json`
- `pipelines/empirical/integration/runs/scmrdr_gene_features_<RUN_ID>/matrices/<MARK>_gene_scores.mtx`
- `pipelines/empirical/integration/runs/scmrdr_gene_features_<RUN_ID>/matrices/<MARK>_gene_scores_availability.tsv`

In shared-gene mode, the model can use `feature_available` masks so structurally missing genes are not treated as observed zeros.

## Output Layout

Root:

```text
pipelines/empirical/integration/runs/scmrdr_<RUN_ID>/
```

Important outputs:

- `preprocess/rna_preprocessed.h5ad`
- `preprocess/chrom_<MARK>_preprocessed.h5ad`
- `preprocess/chrom_preprocessed_manifest.tsv`
- `train/jianle_joint_model.pt`
- `train/all_cells_jianle_embeddings.tsv`
- `train/modality_outputs.tsv`
- `train/train_history.tsv`
- `train/train_summary.json`
- `train/validation/validation_metrics.json`
- `train/validation/umap_by_modality.png`
- `train/validation/cluster_by_modality_fraction.tsv`
- `train/validation/joint_harmonized_label_transfer.tsv` when `--transfer-labels` is used

## Interpretation

The PBMC metrics quantify compartment mixing and protein-score agreement. Direct cross-assay edge identification requires an additional link between these neighborhoods and the missing same-cell measurements. The project tests that distinction in the [integration benchmark](../../../../simulation/integration_recoverability/README.md).
