# Integration

↑ [Central README](../../../README.md) · [Empirical pipeline](../README.md) · [MISSION](../../../project_resources/research/MISSION.md) (why) · [ROADMAP](../../../project_resources/research/ROADMAP.md) (plan)

This directory contains the multi-omic integration layer used before causal discovery.
The existing real-data setup is:

- `scGLUE` `joint_v2` is the historical baseline used for downstream pairing and graph discovery; its biological alignment is under review.
- `Jianle` / `scMRDR_gaussian` is kept as a comparison track.
- Older pilot scripts are useful for debugging only.

For isolated run directories and configuration templates, see the
[run registry](runs/README.md) and [run configuration guide](configs/runs/README.md).

The separate [scConfluence count-input pilot](methods/scconfluence/README.md)
uses the newly authorized [scMTR-seq cohort](../data/scmtr_seq/README.md) and two
cached simulations. The [study record](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)
tracks fixed fits and independent causal reliability checks. Pairing and day
labels remain evaluator-only; see [L19](../../../project_resources/research/LIMITATIONS.md#l19-scmtr-seq-pilot-agreement-is-not-causal-ground-truth).

The subsequent [count-source extension comparison](../../../project_resources/research/count_extensions_2026_11_01/README.md)
and [selected-dimension follow-up](../../../project_resources/research/count_rank20_2026_11_01/README.md)
test observation models and source estimation directly on the existing assay
counts. Their independently checked readouts distinguish within-assay
covariance, cross-assay source consistency and the separate causal target.

## Biological integration review — 2026-09-10

The biology track audits the existing data and integration before accepting
further biological interpretation. It is separate from the
[aggregation agent's work](../../../project_resources/research/aggregation_restart_plan.md), and CA-BOSS remains paused.
Measured results and the method shortlist are in the
[biological foundation review](../data/results/biological_foundation_review_2026-09-10.md);
status is in [ROADMAP](../../../project_resources/research/ROADMAP.md#two-goals-2026-09-10).
Use [L17](../../../project_resources/research/LIMITATIONS.md#l17-biological-integration-is-not-established-by-modality-mixing)
for the interpretation boundary on mixing, transferred labels, and protein-marker checks.

**Follow-up 2026-09-11:** the
[existing-data investigation](../data/results/biological_investigation_2026-09-11.md)
checks local count/identity lineage and saved label replay, and separates
within-assay conservation from cross-assay label correspondence. It also distinguishes
the LSI feature set, reconstruction feature set and independent peak-calling branch.
No integration or label files were changed. Diagnostic commands live in the
[Data README](../data/README.md#existing-data-investigation-2026-09-11).

## Controlled validation execution — 2026-09-16

The user authorized execution using existing data. Three saved-embedding
evaluations are complete: matched scGLUE/scMRDR and full-cohort scGLUE.
Within-assay protein conservation and cross-assay agreement are now measured
separately, with depth/hashtag sensitivity, conditional shuffles and reserved
assessment outcomes. Independent review reproduced key effects from raw ADTs.
Results and commands are in the
[alignment baseline and rerun protocol](results/biological_alignment_2026-09-16.md).

Seven isolated H5AD inputs preserve counts and PCA/LSI while avoiding the dense
RNA `X`; their `X` explicitly contains raw counts. Historical inputs/results
are unchanged. CUDA was verified with a real GPU tensor calculation outside the
sandbox. Twenty-three tests pass. No new integration has yet been trained or
promoted; current-environment controls precede feature/guidance experiments.

## Simulation design before causal aggregation

The [integration recoverability protocol](../../simulation/integration_recoverability/README.md)
defines a count-based comparison before aggregation. Both its simple and
extended designs are now in implementation, using native scGLUE and the new
paper-consistent scMRDR adapter. See the
[execution record](../../simulation/integration_recoverability/IMPLEMENTATION_STATUS.md)
for current scope, completed checks, and pending scientific results.

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

The graph-discovery matrices are generated under `pipelines/empirical/analysis/`, not directly from the older integration causal-export helper.

Integration and aggregation solve different problems. The embedding supplies an
estimated state geometry; downstream matching or metacell averaging supplies a
coupling between cells that were never jointly measured. The coupling does not by
itself reconstruct same-cell covariance or preserve causal conditional
independences. See the
[latent-space aggregation derivation](../../../project_resources/research/theory/06_latent_space_aggregation.md)
and the owned claim boundary in
[L18](../../../project_resources/research/LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution).

The [modern-objective analysis](../../../project_resources/research/theory/08_modern_integration_objectives.md)
audits the scGLUE, paper-consistent scMRDR and scConfluence inputs/losses and
proves why these objectives cannot distinguish certain causal systems under
strictly disjoint sampling. It separates the published objectives from local
adaptations and preserves their useful alignment targets.

The [literature review](../../../project_resources/research/integration_literature_scope_2026_11_01/REVIEW.md)
compares these methods with paired/bridge approaches and restricted models that
do have unpaired identification results. It separates latent-graph recovery
from original peak–gene causal effects.

The [Sturma native-method pilot](../../../project_resources/research/sturma_pilot_2026_11_01/README.md)
implements that positive-model follow-up: published-model replication and direct
count application using the existing permitted assays, with heldout joint-moment
evaluation and independent reviews.
The subsequent [count-aware source extension](../../../project_resources/research/count_mdcrl_2026_11_01/README.md)
changes the count observation model and source estimator, with training-only
calibration and independent heldout evaluation. It remains a research pilot;
improved within-assay covariance fitting did not yield supported cross-assay
source matches or a causal graph.

## Layout

```text
integration/
├── methods/
│   ├── scglue/       scGLUE preprocessing, graph, training, validation, and export
│   ├── scmrdr/       scMRDR comparison workflow (originally grouped as jianle)
│   └── scconfluence/ pinned count-input pilot and transparent native loss corrections
├── scripts/          shared setup and environment checks
├── configs/          environment and run configuration templates
├── manifests/        tracked input manifests generated from the data stage
├── runs/             one isolated directory per integration run
├── comparisons/      cross-run evaluation artifacts
├── workspace/        generated symlink view of data/; rebuild, never edit by hand
├── logs/             historical runner logs
├── results/          method notes and interpretation
└── tests/            integration-specific checks
```

The existing runs were moved into stable, separate folders such as
`runs/scglue_joint_v2/` and `runs/scmrdr_scMRDR_gaussian/`. Keep new runs in a
new directory rather than writing into an existing run. The small
`comparisons/` directory holds results that combine several runs.

- `manifests/scglue_input_manifest.tsv`: source-of-truth manifest for the six CUT&Tag marks, including fixed-bin matrices, clean-cell metadata, `bins_bed`, and `peaks_bed`.
- `manifests/step1_data_check.tsv`: per-input existence check emitted alongside the manifest.

Both manifests are **generated** — rebuild them and the workspace symlinks with:

```bash
python pipelines/empirical/integration/scripts/setup_integration_workspace.py
```

Run this after anything moves under `pipelines/empirical/data/`; workspace symlinks use absolute paths.

### results/ — method notes

- [biological_alignment_2026-09-16.md](results/biological_alignment_2026-09-16.md) — current biological baseline, validation contract and controlled rerun sequence.
- [scGLUE_start_guide.md](results/scGLUE_start_guide.md) — practical scGLUE setup and run guide.
- [GLUE_notes.md](results/GLUE_notes.md) — working notes on GLUE behaviour.
- [scglue_custom_joint_backend_native_semantics.md](results/scglue_custom_joint_backend_native_semantics.md) — semantics of the custom streaming guidance-graph backend.
- [jianle_scMRDR_method_and_repo_comparison.md](results/jianle_scMRDR_method_and_repo_comparison.md) — the scMRDR comparison track, method + repo review.

Per-method detail: [scGLUE](methods/scglue/README.md) · [scMRDR](methods/scmrdr/README.md).

## Existing scGLUE Run

The reviewed baseline run is [`runs/scglue_joint_v2/`](runs/scglue_joint_v2/).

Its important outputs include:

- `train/all_cells_glue_embeddings.tsv`: shared cell embeddings for RNA and all CUT&Tag marks.
- `train/modality_outputs.tsv`: modality labels and metadata aligned to the embeddings.
- `train/feature_glue_embeddings.tsv`: feature embeddings.
- `train/validation/validation_metrics.json`: modality-mixing metrics.
- `train/validation/umap_by_modality.png`: modality-level UMAP sanity check.
- `train/validation/joint_harmonized_label_transfer.tsv`: harmonized RNA-to-chromatin label transfer when validation is run with `--transfer-labels`.

Historical modality-mixing snapshot (not biological-label validation):

| Run | Mean Same-Modality Neighbor Fraction | Silhouette By Modality | Interpretation |
| --- | ---: | ---: | --- |
| `scGLUE` `joint_v2` | `0.2548` | `-0.0122` | More mixed by neighbor fraction; baseline under biological review |
| `scMRDR_gaussian` | `0.5693` | `-0.0984` | Useful comparison, but less mixed |

Lower same-modality neighbor fraction means cells are more often close to cells from other modalities.
These metrics address modality separation, not correct cell-state correspondence; see
[L17](../../../project_resources/research/LIMITATIONS.md#l17-biological-integration-is-not-established-by-modality-mixing).

## Running scGLUE

Historical workflow commands, from the repository root. Controlled reruns are
now authorized under the September 16 protocol, but no new training has yet
completed. **Do not execute these examples over historical `joint_v2` outputs:**
the old wrappers can overwrite files. Use fresh, isolated run directories.
The example below requests `--compute-consistency`, but the saved
`joint_v2/train/train_summary.json` records `compute_consistency: false` and no
consistency TSV; it must not be cited as a completed check.

```bash
python3 pipelines/empirical/integration/scripts/setup_integration_workspace.py
bash pipelines/empirical/integration/scripts/check_scglue_env.sh
```

The full joint pipeline wrapper is:

```bash
bash pipelines/empirical/integration/methods/scglue/run_joint_pipeline.sh \
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
bash pipelines/empirical/integration/methods/scglue/run_preprocess_joint.sh joint_v2
bash pipelines/empirical/integration/methods/scglue/run_build_graph_joint.sh joint_v2 /absolute/path/to/gencode.gtf.gz
bash pipelines/empirical/integration/methods/scglue/run_train_joint.sh joint_v2 --max-epochs 120 --compute-consistency
bash pipelines/empirical/integration/methods/scglue/run_validate_joint.sh joint_v2 --cell-type-level harmonized_coarse --transfer-labels
```

The custom guidance-graph backend is preferred for full multi-mark runs because it streams edge construction instead of holding a large global graph in memory.

## Jianle / scMRDR Comparison Track

The important comparison run is:

- `runs/scmrdr_scMRDR_gaussian/`

This track converts or aligns modalities into a shared feature representation and trains a scMRDR-style model with reconstruction, alignment, and geometry-preservation terms. It is useful for checking whether scGLUE is actually giving better modality mixing, but it is not the current causal-discovery base.

Common commands:

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_preprocess_joint.sh scMRDR_gaussian
bash pipelines/empirical/integration/methods/scmrdr/run_train_joint.sh scMRDR_gaussian --max-epochs 200
bash pipelines/empirical/integration/methods/scmrdr/run_validate_joint.sh scMRDR_gaussian --transfer-labels
```

The gene-feature universe helper can build shared gene-space matrices for Jianle experiments:

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_build_gene_feature_universe.sh \
  gene_space_v1 \
  /absolute/path/to/gencode.v49.primary_assembly.annotation.gtf.gz \
  --gene-universe-mode union_linked \
  --gene-region promoter \
  --promoter-len 2000 \
  --weight-mode binary
```

## Hand-Off To Causal Discovery

Causal discovery now starts from `scGLUE` `joint_v2` embeddings and matched sample tables in:

- `pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor`

The downstream datasets use CUT&Tag peak definitions and quantify those peaks from the raw clean fixed-bin matrices. That means `peaks_bed` defines candidate regulatory intervals, while `bins_bed` and the bin matrices provide the actual per-cell signal used to quantify each peak.

The older scGLUE causal export wrapper still exists:

```bash
bash pipelines/empirical/integration/methods/scglue/run_export_causal_inputs_joint.sh joint_v2 CSF1R,CD14,IL1B
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

- `pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10`
- `pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10`
- `pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts`

The matching table contains `1887` RNA anchor rows, compared with `1561` rows in the current one-to-one pseudo-pair table.

Commands:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate the log-normalized alternative dataset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the raw-count alternative dataset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts
```

These roots can be passed to the same downstream `graph`, `sweep`, and `dagma-sweep` commands through `--dataset-root`.
