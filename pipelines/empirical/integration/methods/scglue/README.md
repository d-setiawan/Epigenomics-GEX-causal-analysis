# scGLUE Method Track

This implementation produced the historical PBMC embedding `joint_v2` from the 10x RNA and six CUT&Tag datasets. The November handoff reproduced preprocessing and checked the rebuilt workspace. Fitted model weights and cell embeddings are generated locally. [Restore commands](../../../../../TRANSFER.md) · [Verification](../../../../../reproduce/VERIFICATION.md)

## Role In The Project

scGLUE gives us a shared latent space across RNA and six single-cell CUT&Tag marks:

- `H3K27ac`
- `H3K27me3`
- `H3K4me1`
- `H3K4me2`
- `H3K4me3`
- `H3K9me3`

The [analysis stage](../../../analysis/README.md) used the coordinates for one-to-one pseudo-pairs, RNA-anchored averages and metacells. The graph matrices derive their measurements from RNA counts and clean CUT&Tag bins over mark-specific peaks.

## PBMC workflow

Python stages:

1. [preprocess_joint_scglue.py](preprocess_joint_scglue.py)
2. [build_guidance_graph_joint_scglue.py](build_guidance_graph_joint_scglue.py)
3. [train_joint_scglue.py](train_joint_scglue.py)
4. [validate_joint_scglue.py](validate_joint_scglue.py)
5. [export_causal_inputs_joint_scglue.py](export_causal_inputs_joint_scglue.py)

Run from the repository root after restoring and preprocessing PBMC counts. Choose a fresh run ID when repeating the fit. The following staged and combined commands are alternatives. The historical `joint_v2` manifest recorded `compute_consistency: false`. The examples explicitly request this optional diagnostic. [Handoff instructions](../../../../../TRANSFER.md)

```bash
bash pipelines/empirical/integration/methods/scglue/run_preprocess_joint.sh pbmc_scglue_replay
bash pipelines/empirical/integration/methods/scglue/run_build_graph_joint.sh pbmc_scglue_replay pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz
bash pipelines/empirical/integration/methods/scglue/run_train_joint.sh pbmc_scglue_replay --max-epochs 120 --compute-consistency
bash pipelines/empirical/integration/methods/scglue/run_validate_joint.sh pbmc_scglue_replay --cell-type-level harmonized_coarse --transfer-labels
```

The all-in-one wrapper is:

```bash
bash pipelines/empirical/integration/methods/scglue/run_joint_pipeline.sh \
  pbmc_scglue_replay \
  pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz \
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

## Outputs

Output root:

```text
pipelines/empirical/integration/runs/scglue_<RUN_ID>/
```

Important stage outputs:

- `preprocess/rna_preprocessed.h5ad`: RNA AnnData after scGLUE preprocessing.
- `preprocess/chrom_<MARK>_preprocessed.h5ad`: one CUT&Tag AnnData per mark.
- `preprocess/chrom_preprocessed_manifest.tsv`: manifest of processed chromatin AnnData files.
- `graph/guidance_edges.tsv`: RNA-to-chromatin guidance edges.
- `graph/guidance_nodes.tsv`: graph node metadata.
- `train/scglue_joint_model.dill`: trained scGLUE model.
- `train/all_cells_glue_embeddings.tsv`: shared latent cell embeddings.
- `train/modality_outputs.tsv`: modality/cell metadata aligned to embeddings.
- `train/feature_glue_embeddings.tsv`: feature embeddings.
- `train/validation/validation_metrics.json`: quantitative integration diagnostics.
- `train/validation/umap_by_modality.png`: UMAP colored by modality.
- `train/validation/joint_harmonized_label_transfer.tsv`: optional transferred labels from RNA to chromatin cells.

In `joint_v2`, within-mark neighbors conserved protein signals. Cross-mark results depended on the two marks compared. For example, matched H3K27me3 to H3K4me1 neighbors gave an adjusted T–myeloid correlation of −0.564. A lower same-modality neighbor fraction alone cannot establish that states align. [Alignment report](../../results/biological_alignment_2026-09-16.md)

## Guidance Graph Backends

`build_guidance_graph_joint_scglue.py` supports:

- `--backend custom`: recommended for full multi-mark runs; streams graph construction and avoids holding the full graph in memory.
- `--backend scglue`: uses `scglue.genomics.rna_anchored_prior_graph` directly; useful as a reference but can be more memory intensive.

The guidance graph uses genomic proximity between RNA genes and chromatin bins/features. It is an integration prior, not the same object as the causal-discovery graph.

## Label Harmonization

The RNA label mapping is [tracked](../../manifests/rna_label_harmonization.tsv). To inspect a regenerated template, write to a new output path.

```bash
bash pipelines/empirical/integration/methods/scglue/run_build_label_harmonization_template.sh --out-tsv /tmp/pbmc_harmonization_review.tsv
```

Then validate with transferred labels:

```bash
NUMBA_CACHE_DIR=/tmp/numba_scglue_validation \
bash pipelines/empirical/integration/methods/scglue/run_validate_joint.sh pbmc_scglue_replay \
  --cell-type-level harmonized_coarse \
  --transfer-labels
```

The current Analysis workflow uses the harmonized coarse `monocyte` label for pseudo-pairing.

## Causal Export Note

`export_causal_inputs_joint_scglue.py` is retained for older scGLUE-feature-based exports:

```bash
bash pipelines/empirical/integration/methods/scglue/run_export_causal_inputs_joint.sh pbmc_scglue_replay CSF1R,CD14,IL1B
```

The nearby-peak workflow reads the selected pairing or metacell table and quantifies the peak intervals from counts. [Analysis inputs and units](../../../analysis/README.md#observational-units-and-inputs)

## Legacy Pilot Scripts

These are retained for debugging and backward compatibility:

- [preprocess_pilot_scglue.py](preprocess_pilot_scglue.py)
- [build_guidance_graph_pilot_scglue.py](build_guidance_graph_pilot_scglue.py)
- [train_pilot_scglue.py](train_pilot_scglue.py)
