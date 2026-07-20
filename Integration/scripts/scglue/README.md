# scGLUE Method Track

This folder contains the scGLUE-specific Python implementations. The active integration run for the project is `joint_v2`.

## Role In The Project

scGLUE gives us a shared latent space across RNA and six single-cell CUT&Tag marks:

- `H3K27ac`
- `H3K27me3`
- `H3K4me1`
- `H3K4me2`
- `H3K4me3`
- `H3K9me3`

The latent space is used for RNA-anchored one-to-one pseudo-pairing in `CausalDiscovery`. The graph matrices themselves are generated later from raw clean CUT&Tag bin signal over mark-specific CUT&Tag peaks.

## Active Workflow

Python stages:

1. `scripts/preprocess_joint_scglue.py`
2. `scripts/build_guidance_graph_joint_scglue.py`
3. `scripts/train_joint_scglue.py`
4. `scripts/validate_joint_scglue.py`
5. `scripts/export_causal_inputs_joint_scglue.py`

Shell wrappers from repo root:

```bash
bash Integration/scripts/scglue/run_preprocess_joint.sh joint_v2
bash Integration/scripts/scglue/run_build_graph_joint.sh joint_v2 /absolute/path/to/gencode.v49.primary_assembly.annotation.gtf.gz
bash Integration/scripts/scglue/run_train_joint.sh joint_v2 --max-epochs 120 --compute-consistency
bash Integration/scripts/scglue/run_validate_joint.sh joint_v2 --cell-type-level harmonized_coarse --transfer-labels
```

The all-in-one wrapper is:

```bash
bash Integration/scripts/scglue/run_joint_pipeline.sh \
  joint_v2 \
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

## Outputs

Output root:

```text
Integration/outputs/scglue/joint/<RUN_ID>/
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

For `joint_v2`, the validation metrics currently support using scGLUE over the scMRDR comparison run for downstream pairing.

## Guidance Graph Backends

`build_guidance_graph_joint_scglue.py` supports:

- `--backend custom`: recommended for full multi-mark runs; streams graph construction and avoids holding the full graph in memory.
- `--backend scglue`: uses `scglue.genomics.rna_anchored_prior_graph` directly; useful as a reference but can be more memory intensive.

The guidance graph uses genomic proximity between RNA genes and chromatin bins/features. It is an integration prior, not the same object as the causal-discovery graph.

## Label Harmonization

Build a label harmonization template:

```bash
bash Integration/scripts/scglue/run_build_label_harmonization_template.sh --force
```

Then validate with transferred labels:

```bash
NUMBA_CACHE_DIR=/tmp/numba_scglue_validation \
bash Integration/scripts/scglue/run_validate_joint.sh joint_v2 \
  --cell-type-level harmonized_coarse \
  --transfer-labels
```

The current CausalDiscovery workflow uses the harmonized coarse `monocyte` label for pseudo-pairing.

## Causal Export Note

`export_causal_inputs_joint_scglue.py` is retained for older scGLUE-feature-based exports:

```bash
bash Integration/scripts/scglue/run_export_causal_inputs_joint.sh joint_v2 CSF1R,CD14,IL1B
```

The active graph workflow no longer starts from this export. It starts from `CausalDiscovery/scripts/causal_cli.py match` and then generates nearby CUT&Tag peak datasets from the matched cells.

## Legacy Pilot Scripts

These are retained for debugging and backward compatibility:

- `scripts/preprocess_pilot_scglue.py`
- `scripts/build_guidance_graph_pilot_scglue.py`
- `scripts/train_pilot_scglue.py`
