# Jianle Method Track

This folder hosts the repo-native implementation for Jianle's integration method
track with a parity-upgraded scMRDR-style objective for unpaired multi-omics.

## Scope

Implemented here:
1. Joint preprocessing across RNA + all histone marks
2. Joint model training with scMRDR-style terms:
   reconstruction via ZINB + beta-VAE KL + adversarial alignment + isometric preserve loss
3. Joint validation (UMAP + cluster composition + mixing metrics)

Not implemented yet:
1. Full paper-faithful masked reconstruction over a single explicit cross-modality feature universe
   (current implementation reconstructs measured feature spaces per modality)
2. Causal-export wrapper specific to Jianle outputs

## Scripts

Python scripts:
1. `integration/methods/jianle/scripts/preprocess_joint_jianle.py`
2. `integration/methods/jianle/scripts/train_joint_jianle.py`
3. `integration/methods/jianle/scripts/validate_joint_jianle.py`

Wrappers:
1. `integration/scripts/jianle/run_preprocess_joint.sh`
2. `integration/scripts/jianle/run_train_joint.sh`
3. `integration/scripts/jianle/run_validate_joint.sh`
4. `integration/scripts/jianle/run_joint_pipeline.sh`

## Usage

From repo root:

```bash
python3 integration/scripts/setup_integration_workspace.py
```

### Full joint pipeline

```bash
bash integration/scripts/jianle/run_joint_pipeline.sh jianle_v1 \
  --train-arg --max-epochs --train-arg 200 \
  --train-arg --learning-rate --train-arg 1e-3 \
  --train-arg --batch-size --train-arg 128 \
  --train-arg --beta --train-arg 2.0 \
  --train-arg --lam-alignment --train-arg 1.0 \
  --train-arg --lam-preserve --train-arg 1.0
```

### Step-by-step

```bash
bash integration/scripts/jianle/run_preprocess_joint.sh jianle_v1
bash integration/scripts/jianle/run_train_joint.sh jianle_v1 --max-epochs 200
bash integration/scripts/jianle/run_validate_joint.sh jianle_v1
```

## Output layout

Root:
- `integration/outputs/jianle/joint/<RUN_ID>/`

Expected outputs:
1. `preprocess/`
- `rna_preprocessed.h5ad`
- `chrom_<MARK>_preprocessed.h5ad`
- `chrom_preprocessed_manifest.tsv`
- `joint_preprocess_summary.json`

2. `train/`
- `jianle_joint_model.pt`
- `all_cells_jianle_embeddings.tsv`
- `modality_outputs.tsv`
- `train_history.tsv`
- `train_summary.json`

3. `train/validation/`
- `cells_umap_clusters.tsv`
- `cluster_sizes.tsv`
- `cluster_by_modality.tsv`
- `validation_metrics.json`
- `umap_by_modality.png`
- `umap_by_leiden.png`

## Metrics interpretation

`validation_metrics.json` includes:
1. `silhouette_by_modality` (closer to 0 is usually better mixing)
2. `mean_same_modality_neighbor_fraction` (lower usually means stronger cross-modality integration)

Use these metrics with the UMAP plots and `cluster_by_modality.tsv` for sanity checks.
