# Jianle / scMRDR Method Track

This folder contains the repo-native Jianle / scMRDR-style comparison implementation for unpaired multi-omic integration.

## Role In The Project

This method is not the active causal-discovery base. It is the comparison track used to sanity-check the scGLUE integration decision.

The important current run is:

```text
Integration/outputs/jianle/joint/scMRDR_gaussian
```

Compared with `scGLUE` `joint_v2`, this run showed weaker cross-modality mixing by same-modality neighbor fraction, so downstream monocyte pseudo-pairing currently uses scGLUE instead.

## What This Track Implements

The Jianle / scMRDR path includes:

- Joint preprocessing across RNA and all CUT&Tag marks.
- scMRDR-style training with reconstruction, latent alignment, and geometry-preservation objectives.
- Shared-gene feature-universe construction from histone peak matrices.
- Optional feature-availability masks for shared-gene training.
- Optional decoder conditioning on covariates such as `batch_id` and `donor_id`.
- Validation outputs modeled after the scGLUE validation reports.

This track does not currently provide the active graph-discovery export.

## Scripts

Python implementations:

- `scripts/preprocess_joint_jianle.py`
- `scripts/train_joint_jianle.py`
- `scripts/validate_joint_jianle.py`
- `scripts/build_joint_gene_feature_universe.py`

Shell wrappers from repo root:

- `Integration/scripts/jianle/run_preprocess_joint.sh`
- `Integration/scripts/jianle/run_train_joint.sh`
- `Integration/scripts/jianle/run_validate_joint.sh`
- `Integration/scripts/jianle/run_joint_pipeline.sh`
- `Integration/scripts/jianle/run_build_gene_feature_universe.sh`

## Running The Comparison Workflow

Set up the integration workspace first:

```bash
python3 Integration/scripts/setup_integration_workspace.py
```

Run the staged comparison workflow:

```bash
bash Integration/scripts/jianle/run_preprocess_joint.sh scMRDR_gaussian
bash Integration/scripts/jianle/run_train_joint.sh scMRDR_gaussian --max-epochs 200
bash Integration/scripts/jianle/run_validate_joint.sh scMRDR_gaussian --transfer-labels
```

Or run the wrapper:

```bash
bash Integration/scripts/jianle/run_joint_pipeline.sh scMRDR_gaussian \
  --train-arg --max-epochs --train-arg 200 \
  --train-arg --learning-rate --train-arg 1e-3 \
  --train-arg --batch-size --train-arg 256
```

## Shared Gene Feature Universe

The gene-feature helper converts histone peak matrices into gene-level feature matrices. This is useful for Jianle experiments that require a common RNA-like feature axis across modalities.

```bash
bash Integration/scripts/jianle/run_build_gene_feature_universe.sh \
  gene_space_v1 \
  /absolute/path/to/gencode.v49.primary_assembly.annotation.gtf.gz \
  --gene-universe-mode union_linked \
  --gene-region promoter \
  --promoter-len 2000 \
  --weight-mode binary
```

Expected outputs:

- `Integration/outputs/jianle/gene_features/<RUN_ID>/gene_universe.tsv`
- `Integration/outputs/jianle/gene_features/<RUN_ID>/joint_gene_feature_manifest.tsv`
- `Integration/outputs/jianle/gene_features/<RUN_ID>/gene_feature_summary.json`
- `Integration/outputs/jianle/gene_features/<RUN_ID>/matrices/<MARK>_gene_scores.mtx`
- `Integration/outputs/jianle/gene_features/<RUN_ID>/matrices/<MARK>_gene_scores_availability.tsv`

In shared-gene mode, the model can use `feature_available` masks so structurally missing genes are not treated as observed zeros.

## Output Layout

Root:

```text
Integration/outputs/jianle/joint/<RUN_ID>/
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

Use this track to ask whether a non-scGLUE integration strategy gives similar biological neighborhoods. For the current project state, scGLUE is preferred because it mixes modalities more effectively and supports the monocyte pseudo-pairing workflow used by causal discovery.
