# Epigenomics-GEX-causal-analysis

## scCUT&Tag QC + clean-matrix pipeline

This repository provides a preprocessing pipeline to:
1. Demultiplex pooled cells using HTO.
2. Filter cells by ADT/HTO depth and scCUT&Tag fragment depth.
3. Build clean barcode whitelist + clean metadata.
4. Build a clean cell-by-chromatin sparse matrix for downstream integration and causal analysis.

## End-to-end command

```bash
./scripts/run_cuttag_qc_pipeline.sh \
  <adt.tsv> <hto.tsv> <fragments.tsv.gz> <peaks.bed> <chrom.sizes> \
  <out_dir> <sample_prefix> \
  <min_adt> <min_hto> <min_cuttag_fragments>
```

Example:

```bash
./scripts/run_cuttag_qc_pipeline.sh \
  Data/H3K4me1/H3K4me1_rep1_ADT_counts.tsv \
  Data/H3K4me1/H3K4me1_rep1_HTO_counts.tsv \
  /path/to/fragments.tsv.gz \
  /path/to/peaks.bed \
  /path/to/chrom.sizes \
  Data/H3K4me1/outputs H3K4me1_rep1 10 10 100
```

Note:
- `Rscript` and Seurat are required for step 1 (`01_demux_adt_hto.R`).
- `snapatac2` is required for steps 5 and 7.

## End-to-end command (peaks + fixed bins)

Use this when you want both feature spaces from one run:

```bash
./scripts/run_cuttag_qc_dual_features.sh \
  <adt.tsv> <hto.tsv> <fragments.tsv.gz> <peaks.bed> <chrom.sizes> \
  <out_dir> <sample_prefix> \
  <min_adt> <min_hto> <min_cuttag_fragments> [bin_size]
```

Example:

```bash
./scripts/run_cuttag_qc_dual_features.sh \
  Data/H3K4me1/H3K4me1_rep1_ADT_counts.tsv \
  Data/H3K4me1/H3K4me1_rep1_HTO_counts.tsv \
  /path/to/fragments.tsv.gz \
  /path/to/peaks.bed \
  /path/to/chrom.sizes \
  Data/H3K4me1/outputs H3K4me1_rep1 10 10 100 5000
```

## Chunked execution

Use these stage scripts if you want to stop after clean-cell QC, call peaks separately, then run downstream matrices.

### Stage 1: Pre-peak QC and clean-cell selection

```bash
./scripts/run_cuttag_stage1_prepeak.sh \
  <adt.tsv> <hto.tsv> <fragments.tsv.gz> \
  <out_dir> <sample_prefix> \
  <min_adt> <min_hto> [min_cuttag_fragments]
```

This gives you `*_clean_barcodes.tsv` and `*_clean_cells.tsv`.

### Stage 2: Peak calling from clean fragments (MACS3)

```bash
./scripts/run_cuttag_stage2_call_peaks.sh \
  <fragments.tsv.gz> <clean_barcodes.tsv> \
  <out_dir> <sample_prefix> <genome_size> [macs_q]
```

This filters fragments to clean barcodes, runs `macs3 callpeak`, and writes `*_peaks.bed`.

### Stage 3: Downstream matrix generation (SnapATAC2)

```bash
./scripts/run_cuttag_stage3_downstream.sh \
  <fragments.tsv.gz> <peaks.bed> <chrom.sizes> \
  <clean_barcodes.tsv> <out_prefix> [bin_size]
```

This creates both peak-based and fixed-bin chromatin matrices.

## Script-by-script description

### `scripts/run_cuttag_qc_pipeline.sh`
Orchestrates all steps in order:
1. `01_demux_adt_hto.R`
2. `02_count_fragments.py`
3. `03_build_clean_cell_metadata.py`
4. `04_filter_matrix_by_barcodes.py` (ADT)
5. `04_filter_matrix_by_barcodes.py` (HTO)
6. `05_build_chromatin_matrix.py`

### `scripts/run_cuttag_qc_dual_features.sh`
Orchestrates:
1. Entire `run_cuttag_qc_pipeline.sh` (QC + peak matrix).
2. `07_build_fixed_bin_matrix.py` (fixed-bin matrix from the same clean cells).

### `scripts/run_cuttag_stage1_prepeak.sh`
Purpose:
- Runs QC/demux/filtering only (steps 1-4).
- Produces clean barcodes and clean-cell metadata before peak calling.

### `scripts/run_cuttag_stage2_call_peaks.sh`
Purpose:
- Filters fragments to clean barcodes.
- Calls peaks with MACS3.
- Produces pipeline-ready `*_peaks.bed`.

### `scripts/run_cuttag_stage3_downstream.sh`
Purpose:
- Runs SnapATAC2 matrix generation from known clean barcodes + known peaks.
- Produces both peak and bin matrices.

### `scripts/01_demux_adt_hto.R`
Purpose:
- Reads ADT and HTO count matrices.
- Harmonizes barcode suffixes (`-1`, `.1`) and intersects barcodes.
- Applies minimum per-cell count thresholds (`min_adt`, `min_hto`).
- Runs HTO CLR normalization + `HTODemux` in Seurat.

Main outputs:
- `*_hto_adt_metadata.tsv`
- `*_singlet_barcodes.tsv`

### `scripts/02_count_fragments.py`
Purpose:
- Reads `fragments.tsv` or `fragments.tsv.gz`.
- Counts fragment lines per barcode.

Main output:
- `*_fragment_counts.tsv`

### `scripts/03_build_clean_cell_metadata.py`
Purpose:
- Merges HTO/ADT metadata with fragment counts.
- Keeps only cells that pass final clean-cell criteria:
- `hto_classification == "Singlet"`
- `donor_id` present (unless `--allow-missing-donor`)
- `n_cuttag_fragments >= --min-fragments`

Main outputs:
- `*_clean_cells.tsv`
- `*_clean_barcodes.tsv`

### `scripts/04_filter_matrix_by_barcodes.py`
Purpose:
- Subsets any feature-by-cell TSV matrix to a clean barcode whitelist.
- Used in pipeline for ADT and HTO filtered matrices.

Main outputs:
- `*_adt_clean_matrix.tsv`
- `*_hto_clean_matrix.tsv`

### `scripts/05_build_chromatin_matrix.py`
Purpose:
- Imports fragments in SnapATAC2 using clean barcode whitelist and chromosome sizes.
- Builds cell-by-peak matrix via SnapATAC2 `make_peak_matrix`.
- Exports the result to MatrixMarket + barcode/feature TSV files.

Main outputs:
- `*_chromatin_clean.mtx`
- `*_chromatin_clean_barcodes.tsv`
- `*_chromatin_clean_features.tsv`

### `scripts/06_generate_toy_cuttag_data.py`
Purpose:
- Generates a small synthetic dataset for testing pipeline logic.
- Also writes mock `toy_hto_adt_metadata.tsv` so downstream steps can be tested without R/Seurat.

### `scripts/07_build_fixed_bin_matrix.py`
Purpose:
- Imports fragments in SnapATAC2 using clean barcode whitelist and chromosome sizes.
- Builds fixed-bin (tile) matrix via SnapATAC2 `add_tile_matrix`.
- Supports `--bin-size` (default `5000`).

Main outputs:
- `*_bin_chromatin_clean.mtx`
- `*_bin_chromatin_clean_barcodes.tsv`
- `*_bin_chromatin_clean_features.tsv`
- `*_bins.bed`

### `scripts/08_filter_fragments_by_barcodes.py`
Purpose:
- Filters `fragments.tsv(.gz)` to a clean barcode whitelist.
- Used by stage 2 before MACS3 peak calling.

Main output:
- filtered fragments TSV (typically then gzipped and used by MACS3).

## Output files: rows, columns, and meaning

Below, `<prefix>` means `<out_dir>/<sample_prefix>`.

### `<prefix>_hto_adt_metadata.tsv`
Row meaning:
- One row per cell barcode after ADT/HTO intersection + minimum count filters.

Columns:
- `barcode`: normalized cell barcode ID.
- `adt_total`: total ADT counts for that cell.
- `hto_total`: total HTO counts for that cell.
- `hto_classification`: Seurat HTO call (`Singlet`, `Doublet`, `Negative`).
- `donor_id`: top HTO identity (`HTO_maxID`).
- `second_id`: second-best HTO identity (`HTO_secondID`).
- `hto_margin`: confidence margin between top and second HTO.
- `doublet_flag`: `True/False` for `Doublet`.
- `negative_flag`: `True/False` for `Negative`.
- `singlet_flag`: `True/False` for `Singlet`.

### `<prefix>_singlet_barcodes.tsv`
Row meaning:
- One row per HTO-singlet barcode from step 1.

Columns:
- `barcode`: singlet barcode.

### `<prefix>_fragment_counts.tsv`
Row meaning:
- One row per barcode observed in fragments file.

Columns:
- `barcode`: cell barcode from fragments.
- `n_cuttag_fragments`: number of fragment records for that barcode.

### `<prefix>_clean_cells.tsv`
Row meaning:
- One row per final clean cell passing all filters.

Columns:
- Includes all columns from `*_hto_adt_metadata.tsv`.
- Adds `n_cuttag_fragments` from fragment counting.

This is the main cell-level metadata table for downstream analysis.

### `<prefix>_clean_barcodes.tsv`
Row meaning:
- One row per final clean barcode.

Columns:
- `barcode`: final whitelist used to subset matrices.

### `<prefix>_adt_clean_matrix.tsv`
Row meaning:
- One row per ADT feature (antibody tag).

Column meaning:
- Column 1: feature name.
- Columns 2..N: clean cell barcodes.
- Each value: ADT count for feature x cell.

Matrix orientation:
- `features x cells`

### `<prefix>_hto_clean_matrix.tsv`
Row meaning:
- One row per HTO feature.

Column meaning:
- Column 1: HTO feature name.
- Columns 2..N: clean cell barcodes.
- Each value: HTO count for feature x cell.

Matrix orientation:
- `features x cells`

### `<prefix>_chromatin_clean.mtx`
Format:
- MatrixMarket sparse coordinate format.

Matrix orientation:
- `cells x chromatin_features`

Row meaning:
- Row index = cell (barcode mapping in `*_chromatin_clean_barcodes.tsv`).

Column meaning:
- Column index = chromatin feature/peak (mapping in `*_chromatin_clean_features.tsv`).

Entry meaning:
- Value at `(cell_i, feature_j)` = SnapATAC2 peak-count matrix entry for cell `i` and peak `j`.

### `<prefix>_chromatin_clean_barcodes.tsv`
Row meaning:
- One row per matrix row in `*_chromatin_clean.mtx` (same order).

Columns:
- `barcode`: clean cell barcode.

### `<prefix>_chromatin_clean_features.tsv`
Row meaning:
- One row per matrix column in `*_chromatin_clean.mtx` (same order).

Columns:
- `feature`: peak name from BED column 4, or fallback `chr:start-end`.

### `<prefix>_bin_chromatin_clean.mtx`
Format:
- MatrixMarket sparse coordinate format.

Matrix orientation:
- `cells x fixed_bins`

Row meaning:
- Row index = cell (barcode mapping in `*_bin_chromatin_clean_barcodes.tsv`).

Column meaning:
- Column index = fixed genomic bin (mapping in `*_bin_chromatin_clean_features.tsv`).

Entry meaning:
- Value at `(cell_i, bin_j)` = SnapATAC2 tile-count matrix entry for cell `i` and bin `j`.

### `<prefix>_bin_chromatin_clean_barcodes.tsv`
Row meaning:
- One row per matrix row in `*_bin_chromatin_clean.mtx` (same order).

Columns:
- `barcode`: clean cell barcode.

### `<prefix>_bin_chromatin_clean_features.tsv`
Row meaning:
- One row per matrix column in `*_bin_chromatin_clean.mtx` (same order).

Columns:
- `feature`: fixed-bin name formatted as `chr:start-end`.

### `<prefix>_bins.bed`
Row meaning:
- One row per fixed genomic bin used to build the bin matrix.

Columns:
- `chrom`: chromosome.
- `start`: 0-based bin start.
- `end`: bin end (exclusive).
- `name`: bin ID (`chr:start-end`).

## Recommended downstream usage

For integration and causal graph discovery, use:
1. `*_chromatin_clean.mtx` + `*_chromatin_clean_barcodes.tsv` + `*_chromatin_clean_features.tsv` as the primary epigenomic matrix.
2. `*_clean_cells.tsv` as cell-level covariates (donor/sample/QC).
3. Optional `*_adt_clean_matrix.tsv` as additional modality/covariates.

If you generate both peak and bin matrices, recommended default:
1. Integration:
- Start with `*_bin_chromatin_clean.mtx` for robust cross-sample alignment (consistent feature space).
2. Causal graph discovery:
- Start with `*_chromatin_clean.mtx` (peaks) for better biological interpretability and reduced dimensionality.
3. Practical strategy:
- Run integration on bins, then transfer cluster/cell-state labels and run causal analysis on peaks within states.

## Integration module (scGLUE + Jianle)

The repository now includes a dedicated integration module:

- `integration/methods/scglue/` for scGLUE runs.
- `integration/methods/jianle/` for Jianle-method runs.
- `integration/scripts/` for shared setup and environment checks.

Build/update the local integration workspace and manifests:

```bash
python3 integration/scripts/setup_integration_workspace.py
```

Check scGLUE environment:

```bash
bash integration/scripts/check_scglue_env.sh
```

Shared input manifest for both methods:

- `integration/manifests/scglue_input_manifest.tsv`

Generated runtime directories are gitignored:

- `integration/workspace/`
- `integration/outputs/`
- `integration/logs/`

## Toy test (small dataset)

Generate a tiny synthetic dataset:

```bash
python3 ./scripts/06_generate_toy_cuttag_data.py ./sandbox_outputs/toy_validation/toy_data
```

Run the full pipeline on toy data (requires `Rscript` + Seurat):

```bash
./scripts/run_cuttag_qc_pipeline.sh \
  ./sandbox_outputs/toy_validation/toy_data/toy_ADT_counts.tsv \
  ./sandbox_outputs/toy_validation/toy_data/toy_HTO_counts.tsv \
  ./sandbox_outputs/toy_validation/toy_data/toy_fragments.tsv.gz \
  ./sandbox_outputs/toy_validation/toy_data/toy_peaks.bed \
  ./sandbox_outputs/toy_validation/toy_data/toy_chrom.sizes \
  ./sandbox_outputs/toy_validation/toy_out toy 10 10 100
```

Test downstream steps without R/Seurat (using mock metadata):

```bash
python3 ./scripts/02_count_fragments.py \
  ./sandbox_outputs/toy_validation/toy_data/toy_fragments.tsv.gz \
  ./sandbox_outputs/toy_validation/toy_out_fragment_counts.tsv

python3 ./scripts/03_build_clean_cell_metadata.py \
  ./sandbox_outputs/toy_validation/toy_data/toy_hto_adt_metadata.tsv \
  ./sandbox_outputs/toy_validation/toy_out_fragment_counts.tsv \
  ./sandbox_outputs/toy_validation/toy_out/toy --min-fragments 100

python3 ./scripts/04_filter_matrix_by_barcodes.py \
  ./sandbox_outputs/toy_validation/toy_data/toy_ADT_counts.tsv \
  ./sandbox_outputs/toy_validation/toy_out/toy_clean_barcodes.tsv \
  ./sandbox_outputs/toy_validation/toy_out/toy_adt_clean_matrix.tsv

python3 ./scripts/05_build_chromatin_matrix.py \
  ./sandbox_outputs/toy_validation/toy_data/toy_fragments.tsv.gz \
  ./sandbox_outputs/toy_validation/toy_data/toy_peaks.bed \
  ./sandbox_outputs/toy_validation/toy_data/toy_chrom.sizes \
  ./sandbox_outputs/toy_validation/toy_out/toy_clean_barcodes.tsv \
  ./sandbox_outputs/toy_validation/toy_out/toy
```

To test fixed-bin matrix on toy data, run step 7:

```bash
python3 ./scripts/07_build_fixed_bin_matrix.py \
  ./sandbox_outputs/toy_validation/toy_data/toy_fragments.tsv.gz \
  ./sandbox_outputs/toy_validation/toy_data/toy_chrom.sizes \
  ./sandbox_outputs/toy_validation/toy_out/toy_clean_barcodes.tsv \
  ./sandbox_outputs/toy_validation/toy_out/toy --bin-size 1000
```
