# Jianle Method Track

This folder hosts the repo-native implementation for Jianle's integration method
track with a parity-upgraded scMRDR-style objective for unpaired multi-omics.

## Scope

Implemented here:
1. Joint preprocessing across RNA + all histone marks
2. Joint model training with scMRDR-style terms:
   reconstruction via ZINB + beta-VAE KL + adversarial alignment + isometric preserve loss
3. Joint validation (UMAP + cluster composition + mixing metrics + RNA annotation overlays + optional label transfer)
4. Gene-centric shared-feature-universe bootstrap from histone peak matrices to gene-level features
5. Shared-gene preprocess and training mode with per-modality feature-availability masks
6. Decoder conditioning on explicit covariates such as `batch_id` and `donor_id`
7. scGLUE-style validation outputs including modality facets, cluster-by-modality fractions, RNA label overlays, and optional harmonized label transfer
8. Shared-gene HVG-union reduction with automatic Gaussian reconstruction on z-scored log-normalized gene features

Not implemented yet:
1. Paper-faithful training protocol and benchmark suite
2. Causal-export wrapper specific to Jianle outputs

## Scripts

Python scripts:
1. `integration/methods/jianle/scripts/preprocess_joint_jianle.py`
2. `integration/methods/jianle/scripts/train_joint_jianle.py`
3. `integration/methods/jianle/scripts/validate_joint_jianle.py`
4. `integration/methods/jianle/scripts/build_joint_gene_feature_universe.py`

Wrappers:
1. `integration/scripts/jianle/run_preprocess_joint.sh`
2. `integration/scripts/jianle/run_train_joint.sh`
3. `integration/scripts/jianle/run_validate_joint.sh`
4. `integration/scripts/jianle/run_joint_pipeline.sh`
5. `integration/scripts/jianle/run_build_gene_feature_universe.sh`

## Usage

From repo root:

```bash
python3 integration/scripts/setup_integration_workspace.py
```

### Build a shared gene feature universe from histone peaks

This builds a shared feature universe that the joint Jianle preprocess can now
consume directly.

```bash
bash integration/scripts/jianle/run_build_gene_feature_universe.sh \
  gene_space_v1 \
  /absolute/path/to/gencode.v49.primary_assembly.annotation.gtf.gz \
  --gene-universe-mode union_linked \
  --gene-region promoter \
  --promoter-len 2000 \
  --weight-mode binary
```

Default output root:

- `integration/outputs/jianle/gene_features/<RUN_ID>/`

Expected outputs:
1. `gene_universe.tsv`
2. `joint_gene_feature_manifest.tsv`
3. `gene_feature_summary.json`
4. `matrices/<MARK>_gene_scores.mtx`
5. `matrices/<MARK>_gene_scores_barcodes.tsv`
6. `matrices/<MARK>_gene_scores_features.tsv`
7. `matrices/<MARK>_gene_scores_availability.tsv`

Notes:
1. This step aggregates histone peak counts into gene-level scores using a GTF.
2. `--gene-universe-mode rna` reproduces the older RNA-anchored universe.
3. `--gene-universe-mode union_linked` keeps RNA genes plus any extra genes linked in at least one selected histone mark.
4. Each mark now also exports a `feature_available` table so training can mask structurally unavailable genes.
5. The joint builder now also preserves `source_prefix` so shared-gene preprocessing can keep a chromatin-side `batch_id`.

### Run Jianle in shared-gene mode

```bash
bash integration/scripts/jianle/run_preprocess_joint.sh jianle_gene_union \
  --gene-feature-manifest integration/outputs/jianle/gene_features/gene_space_v1/joint_gene_feature_manifest.tsv \
  --shared-hvg-top-genes 3000

bash integration/scripts/jianle/run_train_joint.sh jianle_gene_union \
  --covariate-cols donor_id \
  --max-epochs 200
bash integration/scripts/jianle/run_validate_joint.sh jianle_gene_union \
  --transfer-labels
```

Or via the full pipeline:

```bash
bash integration/scripts/jianle/run_joint_pipeline.sh jianle_gene_union \
  --preprocess-arg --gene-feature-manifest \
  --preprocess-arg integration/outputs/jianle/gene_features/gene_space_v1/joint_gene_feature_manifest.tsv \
  --preprocess-arg --shared-hvg-top-genes --preprocess-arg 3000 \
  --train-arg --covariate-cols --train-arg donor_id \
  --train-arg --max-epochs --train-arg 200
```

Covariate notes:
1. Training now one-hot encodes the requested `obs` columns and concatenates them into the decoder input.
2. The default is `--covariate-cols batch_id,donor_id`.
3. If a covariate is missing for a modality, it is encoded with a `__missing__` category rather than dropped.
4. In the current workspace, chromatin has `donor_id` from HTO demultiplexing, while RNA does not, so RNA cells are conditioned on the missing-donor category unless richer RNA metadata are added later.

Shared-gene feature notes:
1. `--shared-hvg-top-genes` now selects per-modality HVGs and reduces the shared-gene feature space to their union.
2. The preprocess stores modality-specific `model_mean` and `model_std` values computed from log-normalized shared-gene counts.
3. `train_joint_jianle.py` now supports `--reconstruction-distribution auto|zinb|gaussian`.
4. In shared-gene mode with stored normalization stats, `--reconstruction-distribution auto` resolves to Gaussian reconstruction with masked MSE.

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

Shared-gene mode uses the same commands, but `run_preprocess_joint.sh` should be
given `--gene-feature-manifest <joint_gene_feature_manifest.tsv>`.

Validation notes:
1. `run_validate_joint.sh` now accepts the same RNA annotation overlay arguments as the scGLUE validator.
2. By default it looks for `integration/workspace/data/rna/cell_type_annotation.tar.gz`.
3. Add `--transfer-labels` to write harmonized RNA-to-non-RNA label transfer outputs in Jianle space.

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
- `cluster_by_modality_fraction.tsv`
- `validation_metrics.json`
- `umap_by_modality.png`
- `umap_by_modality_facets.png`
- `umap_by_leiden.png`
- `cluster_by_modality_heatmap.png`
- `rna_umap_<LABEL>.tsv` when RNA annotation tar is available
- `umap_rna_<LABEL>.png` when RNA annotation tar is available
- `joint_harmonized_label_transfer.tsv` when `--transfer-labels` is used
- `umap_joint_harmonized_coarse.png` when `--transfer-labels` is used
- `umap_joint_harmonized_fine.png` when `--transfer-labels` is used

## Metrics interpretation

`validation_metrics.json` includes:
1. `silhouette_by_modality` (closer to 0 is usually better mixing)
2. `mean_same_modality_neighbor_fraction` (lower usually means stronger cross-modality integration)
3. `annotation_overlay` summary for RNA annotations, harmonization, and optional label transfer

Use these metrics with the UMAP plots, `cluster_by_modality.tsv`, and `cluster_by_modality_fraction.tsv` for sanity checks.
