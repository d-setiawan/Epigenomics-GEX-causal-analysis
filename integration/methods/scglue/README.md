# scGLUE Method Track

This folder contains scGLUE-specific Python implementations.

## Active joint workflow (recommended)

Joint scripts:
1. `preprocess_joint_scglue.py`
2. `build_guidance_graph_joint_scglue.py`
3. `train_joint_scglue.py`
4. `validate_joint_scglue.py`
5. `export_causal_inputs_joint_scglue.py`

Wrapper entrypoint:
1. `integration/scripts/scglue/run_joint_pipeline.sh`
2. Optional stage wrappers under `integration/scripts/scglue/`

Expected joint output root:
1. `integration/outputs/scglue/joint/<RUN_ID>/`

Stages:
1. `preprocess/`: one RNA AnnData + one AnnData per mark (prefixed feature namespace).
2. `graph/`: one combined guidance graph spanning RNA + all mark features.
3. `train/`: one joint scGLUE model and shared latent embeddings for all modalities.
4. `train/validation/`: clustering/UMAP diagnostics and mixing metrics.
5. `train/causal_inputs/`: PC-ready per-target-gene tables.

Example:

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

## Joint graph backends

`build_guidance_graph_joint_scglue.py` supports:
1. `--backend custom` (default): streaming TSV writer, per-mark build, no in-memory global `networkx` graph.
2. `--backend scglue`: uses `scglue.genomics.rna_anchored_prior_graph` directly (higher memory).

`custom` backend is recommended for full multi-mark runs because it is more memory stable while still building a full RNA + chromatin guidance graph.

Export causal inputs (example target genes):

```bash
bash integration/scripts/scglue/run_export_causal_inputs_joint.sh \
  joint_v1 \
  MYC,TP53,GATA1
```

## Legacy pilot scripts

Pilot scripts are still present for debugging/backward compatibility:
1. `preprocess_pilot_scglue.py`
2. `build_guidance_graph_pilot_scglue.py`
3. `train_pilot_scglue.py`
