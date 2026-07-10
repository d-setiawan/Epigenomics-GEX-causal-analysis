# Integration

↑ [Central README](../README.md) · [MISSION](../MISSION.md) (why) · [ROADMAP](../ROADMAP.md) (plan)

This directory contains the multi-omic integration layer used before causal discovery. The current project decision is:

- `scGLUE` `joint_v2` is the active integration used for downstream pairing and graph discovery.
- `Jianle` / `scMRDR_gaussian` is kept as a comparison track.
- Older pilot scripts are useful for debugging only.

## What Integration Does

The integration stage puts RNA cells and single-cell CUT&Tag cells into a shared latent space. We do not make the modalities biologically identical; we use the latent space to find plausible cross-modality neighbors, then the causal-discovery code builds RNA-anchored pseudo-pairs.

The active downstream pipeline uses:

$$
\text{RNA and CUT\&Tag preprocessing}
\rightarrow
\text{scGLUE joint latent space}
\rightarrow
\text{monocyte one-to-one pseudo-pairs}
\rightarrow
\text{nearby CUT\&Tag peak matrices}
$$

The graph-discovery matrices are generated under `CausalDiscovery/`, not directly from the older integration causal-export helper.

## Layout

- `scripts/setup_integration_workspace.py`: creates the local integration workspace and symlinks expected inputs.
- `scripts/check_scglue_env.sh`: checks that the scGLUE environment can import the required packages.
- `scripts/scglue/`: shell wrappers for the active scGLUE joint workflow.
- `scripts/jianle/`: shell wrappers for the Jianle / scMRDR comparison workflow.
- `methods/scglue/scripts/`: Python implementations for scGLUE preprocessing, guidance graph construction, training, validation, and legacy causal export.
- `methods/jianle/scripts/`: Python implementations for Jianle preprocessing, training, validation, and gene-feature construction.
- `manifests/scglue_input_manifest.tsv`: source-of-truth manifest for the six CUT&Tag marks, including fixed-bin matrices, clean-cell metadata, `bins_bed`, and `peaks_bed`.
- `workspace/`: generated local input view, ignored by git.
- `outputs/`: generated integration outputs, ignored by git.
- `logs/`: runtime logs, ignored by git.

## Active scGLUE Run

The active integration run is:

- `outputs/scglue/joint/joint_v2`

The important outputs are:

- `train/all_cells_glue_embeddings.tsv`: shared cell embeddings for RNA and all CUT&Tag marks.
- `train/modality_outputs.tsv`: modality labels and metadata aligned to the embeddings.
- `train/feature_glue_embeddings.tsv`: feature embeddings.
- `train/validation/validation_metrics.json`: modality-mixing metrics.
- `train/validation/umap_by_modality.png`: modality-level UMAP sanity check.
- `train/validation/joint_harmonized_label_transfer.tsv`: harmonized RNA-to-chromatin label transfer when validation is run with `--transfer-labels`.

Current validation snapshot:

| Run | Mean Same-Modality Neighbor Fraction | Silhouette By Modality | Interpretation |
| --- | ---: | ---: | --- |
| `scGLUE` `joint_v2` | `0.2548` | `-0.0122` | Better cross-modality mixing; active choice |
| `scMRDR_gaussian` | `0.5693` | `-0.0984` | Useful comparison, but less mixed |

Lower same-modality neighbor fraction means cells are more often close to cells from other modalities. Silhouette values near zero are usually better for modality mixing.

## Running scGLUE

From the repository root:

```bash
python3 integration/scripts/setup_integration_workspace.py
bash integration/scripts/check_scglue_env.sh
```

The full joint pipeline wrapper is:

```bash
bash integration/scripts/scglue/run_joint_pipeline.sh \
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

Use the stage wrappers if you need to rerun only one step:

```bash
bash integration/scripts/scglue/run_preprocess_joint.sh joint_v2
bash integration/scripts/scglue/run_build_graph_joint.sh joint_v2 /absolute/path/to/gencode.gtf.gz
bash integration/scripts/scglue/run_train_joint.sh joint_v2 --max-epochs 120 --compute-consistency
bash integration/scripts/scglue/run_validate_joint.sh joint_v2 --cell-type-level harmonized_coarse --transfer-labels
```

The custom guidance-graph backend is preferred for full multi-mark runs because it streams edge construction instead of holding a large global graph in memory.

## Jianle / scMRDR Comparison Track

The important comparison run is:

- `outputs/jianle/joint/scMRDR_gaussian`

This track converts or aligns modalities into a shared feature representation and trains a scMRDR-style model with reconstruction, alignment, and geometry-preservation terms. It is useful for checking whether scGLUE is actually giving better modality mixing, but it is not the current causal-discovery base.

Common commands:

```bash
bash integration/scripts/jianle/run_preprocess_joint.sh scMRDR_gaussian
bash integration/scripts/jianle/run_train_joint.sh scMRDR_gaussian --max-epochs 200
bash integration/scripts/jianle/run_validate_joint.sh scMRDR_gaussian --transfer-labels
```

The gene-feature universe helper can build shared gene-space matrices for Jianle experiments:

```bash
bash integration/scripts/jianle/run_build_gene_feature_universe.sh \
  gene_space_v1 \
  /absolute/path/to/gencode.v49.primary_assembly.annotation.gtf.gz \
  --gene-universe-mode union_linked \
  --gene-region promoter \
  --promoter-len 2000 \
  --weight-mode binary
```

## Hand-Off To Causal Discovery

Causal discovery now starts from `scGLUE` `joint_v2` embeddings and matched sample tables in:

- `CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor`

The downstream datasets use CUT&Tag peak definitions and quantify those peaks from the raw clean fixed-bin matrices. That means `peaks_bed` defines candidate regulatory intervals, while `bins_bed` and the bin matrices provide the actual per-cell signal used to quantify each peak.

The older scGLUE causal export wrapper still exists:

```bash
bash integration/scripts/scglue/run_export_causal_inputs_joint.sh joint_v2 CSF1R,CD14,IL1B
```

But this is not the active monocyte nearby-peak graph workflow anymore.

## Alternative Next Directions

Two mentor-discussed directions are now explicitly tracked here.

### Causal Dropout

Causal Dropout, discussed with Haoyue, is a possible future graph-learning or graph-regularization direction. It is not implemented in this repository yet. The high-level idea is to use dropout-style perturbations during model fitting or graph evaluation to ask whether inferred edges remain stable when parts of the representation are masked.

This would be conceptually different from the current PC/FCI/DAGMA runs:

$$
\text{current graph methods}
\rightarrow
\text{fit graph on a fixed matrix}
$$

whereas Causal Dropout would add an explicit robustness or perturbation layer:

$$
\text{matrix}
\rightarrow
\text{dropout perturbations}
\rightarrow
\text{edge stability / causal robustness}
$$

This is documented as a candidate direction only; no dataset or graph command currently uses it.

### RNA-Anchor Neighbor Averaging

The current main pipeline uses RNA-anchored one-to-one pseudo-pairs without replacement. A new alternative mapping strategy now keeps RNA cells as anchors and averages each CUT&Tag modality over the $k$ nearest chromatin neighbors in scGLUE space.

Conceptually:

$$
\text{one RNA anchor}
\rightarrow
k \text{ nearest cells per CUT\&Tag mark}
\rightarrow
\text{average mark-specific peak signal}
\rightarrow
\text{one graph row}
$$

This avoids forcing every chromatin cell to be used at most once. It also allows all monocyte RNA anchors to remain in the dataset.

The current generated $k=10$, uniform-weight version is:

- `CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10`
- `CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10`
- `CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts`

The matching table contains `1887` RNA anchor rows, compared with `1561` rows in the current one-to-one pseudo-pair table.

Commands:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate the log-normalized alternative dataset:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the raw-count alternative dataset:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts
```

These roots can be passed to the same downstream `graph`, `sweep`, and `dagma-sweep` commands through `--dataset-root`.
