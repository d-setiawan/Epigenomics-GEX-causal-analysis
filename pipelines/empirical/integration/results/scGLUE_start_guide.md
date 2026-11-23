# scGLUE Start Guide (Detailed)

This guide is a practical roadmap for integrating your scRNA-seq data with your histone-modification chromatin matrices using scGLUE.

---

## 0) Goal and philosophy

Primary goal:
1. Build a shared cross-modality cell embedding (RNA + chromatin marks).
2. Preserve biological structure while reducing modality-specific separation.
3. Keep the pipeline consistent so you can scale from pilot to all histone marks.

Recommended strategy:
1. Start with a small pilot (one RNA + one histone mark).
2. Validate preprocessing and graph construction.
3. Scale to all marks only after metrics look good.

---

## 1) Data you should have prepared

### 1.1 scRNA-seq input
Use:
1. 10x Gene Expression HDF5 (`.h5`) file as primary input.
2. Optional barcodes/annotations if you need extra metadata.

### 1.2 Histone input
Use your pipeline outputs (per mark/sample):
1. `*_bin_chromatin_clean.mtx`
2. `*_bin_chromatin_clean_barcodes.tsv`
3. `*_bin_chromatin_clean_features.tsv`
4. `*_clean_cells.tsv` (metadata/QC covariates)

Why bins first:
1. Consistent feature space across samples/marks.
2. Easier initial integration than separate peak sets.

### 1.3 Consistency constraints (do not skip)
Ensure all histone datasets use:
1. Same genome build (e.g., hg38).
2. Same bin size.
3. Same chrom sizes file.
4. Same QC logic and threshold philosophy.

---

## 2) Project structure recommendation

Create a clean structure for integration work (outside Git repo if preferred):

```text
project_resources/research/
pipelines/empirical/integration/
  data/
    rna/
    chromatin/
  scripts/
  outputs/
    pilot/
    full_run/
  logs/
```

Suggested naming:
1. Keep each modality as an AnnData object with clear `.obs` columns:
   - `modality`, `sample`, `batch`, `donor`, `mark`
2. Keep one manifest CSV/TSV listing all inputs.

---

## 3) Environment setup

Use a dedicated environment to avoid version conflicts.

Example (adjust for your package manager):
1. Python >= 3.10
2. `scglue`
3. `scanpy`
4. `anndata`
5. `scipy`, `numpy`, `pandas`
6. plotting stack (`matplotlib`, `seaborn`)

Checklist:
1. `import scglue` works.
2. `import scanpy as sc` works.
3. Sparse matrix loading works for `.mtx`.

---

## 4) Pilot phase (one mark first)

Use one RNA dataset + one histone mark (for example H3K4me1).

### 4.1 Load RNA into AnnData
Typical approach:
1. `sc.read_10x_h5(...)`
2. Make gene names unique.
3. Add metadata columns (`modality='rna'`, sample, donor, etc.).

### 4.2 Load chromatin bins into AnnData
From `.mtx` + barcodes + features:
1. Read sparse matrix.
2. Set obs names from barcodes TSV.
3. Set var names from features TSV.
4. Add metadata (`modality='chromatin'`, `mark='H3K4me1'`, etc.).

### 4.3 Join clean-cell metadata
Merge `*_clean_cells.tsv` into histone `.obs` by barcode.

### 4.4 Current pilot config snapshot (H3K4me1)
Based on your current successful pilot preprocess + graph build:

Preprocess:
1. RNA: `n_cells=9462`, `n_genes=25215`, `n_hvg=2000`, `n_pcs=30`
2. Chromatin (H3K4me1): `n_cells=5683`, `n_features=30000`, `n_lsi=30`
3. Chromatin feature cap at preprocess: `chrom_top_features=30000`

Guidance graph:
1. Backend: `scglue` (native)
2. Gene region: `promoter`
3. Promoter length: `1000 bp`
4. Window size: `50000 bp`
5. Decay scale: `75000 bp`
6. Sign: `+1` (activating mark assumption)
7. Alpha: `1.0`
8. Max chrom features in graph: `15000`
9. Export GraphML: `false`

Guidance graph result:
1. `n_nodes_total=39724`
2. `n_edges_total=115938`
3. `n_prior_edges=76214`
4. `n_connected_genes=8154`
5. `n_connected_bins=12675`

---

## 5) Modality-specific preprocessing

Do not force identical preprocessing across modalities; use appropriate pipelines.

### 5.1 RNA preprocessing
Typical baseline:
1. Filter low-quality cells/genes (light-touch if already filtered).
2. Normalize per cell.
3. Log transform.
4. Select HVGs.
5. Scale (as needed).
6. PCA + neighbor graph (for diagnostics).

### 5.2 Chromatin-bin preprocessing
Typical baseline:
1. Keep sparse counts.
2. TF-IDF transform.
3. LSI (latent semantic indexing).
4. Neighbor graph (for diagnostics).

Rationale:
1. RNA and chromatin count properties differ.
2. Modality-aware preprocessing improves representation quality before GLUE alignment.

---

## 6) Guidance graph construction

This is the key biological prior.

### 6.1 Minimum viable graph (start here)
1. Map chromatin features (bins or peaks) to genes by genomic distance.
2. Use a max distance window (e.g., 150 kb).
3. Add distance-decayed edge weights (e.g., power-law style).

### 6.2 Edge sign recommendations
1. Accessibility / activating marks (e.g., H3K4me1, H3K27ac): mostly positive edges.
2. Repressive marks (e.g., H3K27me3, H3K9me3): consider negative edges.

### 6.3 Graph quality checks
1. Fraction of chromatin features connected to at least one gene.
2. Degree distribution not too extreme.
3. Reasonable gene coverage.

---

## 7) Train scGLUE

### 7.1 Training objective components
1. Data reconstruction terms (per modality).
2. Graph reconstruction term.
3. Adversarial modality-alignment term.

### 7.2 Practical training workflow
1. Configure datasets and graph.
2. Fit scGLUE model.
3. Monitor losses and convergence behavior.
4. Export:
   - integrated cell embeddings,
   - feature embeddings,
   - model checkpoints.

### 7.3 Early warning signs
1. Loss unstable/divergent.
2. Extreme modality separation in embedding.
3. One modality collapsing to tiny variance.

If seen:
1. Revisit preprocessing.
2. Revisit graph density/signs.
3. Tune alignment/graph weights and learning rates.

---

## 8) Evaluate without paired ground truth

You can evaluate robustly even without one-to-one matched cells.

### 8.1 Omics mixing
1. SAS
2. Omics-layer ASW
3. Graph connectivity

### 8.2 Biology conservation
If labels or pseudo-labels are available:
1. MAP
2. Cell-type ASW
3. Neighbor consistency

### 8.3 GLUE-specific consistency
1. Integration consistency score against guidance graph.

### 8.4 Visual diagnostics
1. UMAP of integrated embedding colored by:
   - modality
   - sample/batch
   - known biology labels
2. Desired pattern: biology clusters mixed by modality.

---

## 9) Scale from pilot to all histone marks

After pilot success, expand in a controlled way.

### 9.1 Add marks incrementally
1. Add one mark at a time.
2. Re-check metrics each addition.

### 9.2 Keep integration assumptions fixed
1. Same bin size and genome build.
2. Same graph-construction rule family.
3. Same preprocessing policy by modality.

### 9.3 Record run metadata
For each run save:
1. data manifest
2. software versions
3. key hyperparameters
4. metric table
5. plots

---

## 10) Suggested deliverables per run

For each pilot/full run, save:
1. `integrated_cells.csv` or AnnData with embeddings
2. `feature_embeddings.csv`
3. `metrics_summary.tsv`
4. `umap_by_modality.png`
5. `umap_by_celltype.png` (or cluster)
6. `run_config.yaml`
7. `training_log.txt`

---

## 11) Common pitfalls and fixes

1. Problem: strong modality split after training
- Check graph edges are not too sparse/incorrect.
- Increase alignment influence carefully.

2. Problem: biology looks overmixed (cell types collapse)
- Alignment may be too strong.
- Re-check preprocessing and latent dimensionality.

3. Problem: one mark behaves very differently
- Check mark-specific biology/sign assumptions in graph.
- Verify QC consistency and count depth distribution.

4. Problem: memory/runtime issues
- Reduce feature space for pilot.
- Use chunked preprocessing and sparse-friendly operations.

---

## 12) Your immediate next actions

1. Pick one RNA + one histone sample for pilot.
2. Build a small script to load both into AnnData.
3. Implement a first distance-based guidance graph.
4. Run one scGLUE training pass.
5. Generate mixing + biology plots and metrics.
6. Decide tuning changes before scaling.

---

## 13) Notes for your current pipeline

Given your existing CUT&Tag workflow outputs, the natural starting point is:
1. Use `*_bin_chromatin_clean.*` for integration.
2. Keep `*_clean_cells.tsv` as metadata covariates.
3. Use peak matrices later for interpretability/regulatory follow-up.

This minimizes moving parts while you establish a stable scGLUE workflow.
