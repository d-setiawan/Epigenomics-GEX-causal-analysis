# Integration

↑ [Central README](../../../README.md) · [Empirical pipeline](../README.md) · [MISSION](../../../project_resources/research/MISSION.md) (why) · [ROADMAP](../../../project_resources/research/ROADMAP.md) (plan)

This directory fits shared coordinates across RNA and chromatin assays and evaluates the neighborhoods they imply. Two empirical studies supply different checks.

- **PBMC monocytes.** The historical `scGLUE joint_v2` embedding and the repo-native `scMRDR_gaussian` comparator used separate RNA and CUT&Tag cells with no verified matched donors. Protein measurements held outside fitting show that some cross-mark neighborhoods reversed the T–myeloid contrast, even when neighborhoods within each mark conserved it. [Input provenance](../data/README.md) · [Alignment report](results/biological_alignment_2026-09-16.md)
- **scMTR-seq S1.** The [scConfluence pilot](methods/scconfluence/README.md) used disjoint learner cells and reserved the original pairing for evaluation. Its sparse 595-cell paired panel produced no positive cross-assay graph reference. Two cached synthetic worlds tested the same adapter against known truth with two seeds apiece. [Cohort](../data/scmtr_seq/README.md) · [Results and independent checks](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)

The [transfer guide](../../../TRANSFER.md) covers source restoration, preprocessing and a new integration run. The November replay verified all 24 compared PBMC preprocessing products, 27 workspace checks and six scMTR learner H5ADs. Full GPU model training remains outside that replay. [Verification](../../../reproduce/VERIFICATION.md) · [Run registry](runs/README.md) · [Run configurations](configs/runs/README.md)

## Biological integration review — 2026-09-10

The biology track audits the existing data and integration before accepting
further biological interpretation. It is separate from the
[aggregation study](../../../project_resources/research/aggregation_restart_plan.md), and CA-BOSS remains paused.
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

Three saved-embedding evaluations used the matched scGLUE and repo-native scMRDR cohort and the full scGLUE cohort.
Within-assay protein conservation and cross-assay agreement are now measured
separately, with depth/hashtag sensitivity, conditional shuffles and reserved
assessment outcomes. Independent review reproduced key effects from raw ADTs.
Results and commands are in the
[alignment baseline and rerun protocol](results/biological_alignment_2026-09-16.md).

Seven isolated PBMC H5AD inputs preserved counts and PCA/LSI, with raw counts stored in `X`. The September report records 23 tests and a CUDA tensor check in its original environment. It assessed existing models.

## Simulation before causal aggregation

The [integration recoverability benchmark](../../simulation/integration_recoverability/README.md) compares native scGLUE with a paper-consistent scMRDR adapter. Development fits and the count-input supplement yielded results, while the original extended assessment remains incomplete. The [execution record](../../simulation/integration_recoverability/IMPLEMENTATION_STATUS.md) identifies which arms ran.

The [exact-model source search](../../simulation/integration_recoverability/runs/count_transport_search_v1/README.md) is a separate restricted-model experiment. Its simulations test when information in assay marginals can identify finite latent components. Applying its conclusions to these PBMC and scMTR measurements would require establishing the same assumptions for the empirical data.

## What Integration Does

The embedding estimates shared biological state. An RNA anchor, its matched chromatin neighbor and a metacell each define a different way to build a row for graph discovery. Even perfect state alignment can reveal how two assays' conditional means vary together while leaving the dependence between their residuals within each state unknown. For the derivation and explicit counterexamples, see [latent-space aggregation](../../../project_resources/research/theory/06_latent_space_aggregation.md) and [the unpaired integration result](../../../project_resources/research/theory/07_unpaired_integration_nonidentifiability.md).

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

The reviewed PBMC baseline was stored locally at `runs/scglue_joint_v2/`. The tracked [alignment report](results/biological_alignment_2026-09-16.md) records the cohort and protein-marker comparisons.

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

Use a fresh run ID in the restored PBMC workspace. These wrappers can overwrite stage outputs. The examples below request the optional `--compute-consistency` diagnostic. The saved `joint_v2` training summary recorded this option as false. [Restore and preprocess](../../../TRANSFER.md)

```bash
python3 pipelines/empirical/integration/scripts/setup_integration_workspace.py
bash pipelines/empirical/integration/scripts/check_scglue_env.sh
```

The full joint pipeline wrapper is:

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

Use the stage wrappers if you need to rerun only one step:

```bash
bash pipelines/empirical/integration/methods/scglue/run_preprocess_joint.sh pbmc_scglue_replay
bash pipelines/empirical/integration/methods/scglue/run_build_graph_joint.sh pbmc_scglue_replay pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz
bash pipelines/empirical/integration/methods/scglue/run_train_joint.sh pbmc_scglue_replay --max-epochs 120 --compute-consistency
bash pipelines/empirical/integration/methods/scglue/run_validate_joint.sh pbmc_scglue_replay --cell-type-level harmonized_coarse --transfer-labels
```

The custom guidance-graph backend is preferred for full multi-mark runs because it streams edge construction instead of holding a large global graph in memory.

## Jianle / scMRDR Comparison Track

The important comparison run is:

- `runs/scmrdr_scMRDR_gaussian/`

The historical shell wrappers train a repo-native approximation with reconstruction, alignment and geometry-preservation terms. The separate [paper-consistent simulation adapter](methods/scmrdr/paper_faithful/README.md) implements the audited model.

Common commands:

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_preprocess_joint.sh pbmc_scmrdr_replay
bash pipelines/empirical/integration/methods/scmrdr/run_train_joint.sh pbmc_scmrdr_replay --max-epochs 200
bash pipelines/empirical/integration/methods/scmrdr/run_validate_joint.sh pbmc_scmrdr_replay --transfer-labels
```

The gene-feature universe helper can build shared gene-space matrices for Jianle experiments:

```bash
bash pipelines/empirical/integration/methods/scmrdr/run_build_gene_feature_universe.sh \
  gene_space_v1 \
  pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz \
  --gene-universe-mode union_linked \
  --gene-region promoter \
  --promoter-len 2000 \
  --weight-mode binary
```

## Hand-Off To Causal Discovery

The [analysis stage](../analysis/README.md) defines the observation unit before graph discovery. Its one-to-one comparator used `outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor` under the analysis directory. Other analyses used RNA-anchored averages or nonoverlapping metacells. The `peaks_bed` intervals define candidate peaks, and the clean fixed-bin matrices provide the signal within them. The older `run_export_causal_inputs_joint.sh` exports features directly from scGLUE and is retained for historical comparisons.

## Historical design alternatives

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

The one-to-one comparator selected chromatin cells without replacement. The neighbor-average comparator keeps each RNA cell as an anchor and averages its nearest chromatin neighbors separately for each mark. Different anchors can reuse chromatin cells.

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

The historical $k=10$ uniform-weight example used the following generated paths.

- `pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10`
- `pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10`
- `pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts`

The historical matching tables contained 1,887 averaged RNA anchor rows and 1,561 one-to-one rows. The commands below assume that the `joint_v2` model and matching artifacts are available locally.

Commands:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate the log-normalized alternative dataset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the raw-count alternative dataset:

```bash
python3 pipelines/empirical/analysis/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts
```

Use `--dataset-root` with `sweep` or `dagma-sweep`. For an individual `graph` call, pass the locus matrix with `--matrix-tsv`. [Command examples](../analysis/scripts/README.md)
