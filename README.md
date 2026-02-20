# Epigenomics-GEX-causal-analysis

## scCUT&Tag QC + clean-matrix pipeline

Run the end-to-end preprocessing pipeline:

```bash
./scripts/run_cuttag_qc_pipeline.sh \
  <adt.tsv> <hto.tsv> <fragments.tsv.gz> <peaks.bed> \
  <out_dir> <sample_prefix> \
  <min_adt> <min_hto> <min_cuttag_fragments>
```

Example:

```bash
./scripts/run_cuttag_qc_pipeline.sh \
  Data/H3K4me1/outputs/ADT.tsv \
  Data/H3K4me1/outputs/HTO.tsv \
  Data/H3K4me1/outputs/fragments.tsv.gz \
  Data/H3K4me1/outputs/peaks.bed \
  out H3K4me1 10 10 100
```

Key outputs:
- `*_hto_adt_metadata.tsv`: HTO demux metadata after ADT/HTO minimum-count filtering.
- `*_fragment_counts.tsv`: per-cell scCUT&Tag fragment depth.
- `*_clean_cells.tsv`: final clean-cell table (Singlet + donor assigned + min fragments).
- `*_clean_barcodes.tsv`: final barcode whitelist for matrix subsetting.
- `*_adt_clean_matrix.tsv`, `*_hto_clean_matrix.tsv`: integration-ready filtered matrices.
- `*_chromatin_clean.mtx`: sparse MatrixMarket file with rows=cells and cols=chromatin features.
- `*_chromatin_clean_barcodes.tsv`: row annotations for `*_chromatin_clean.mtx`.
- `*_chromatin_clean_features.tsv`: column annotations for `*_chromatin_clean.mtx`.

## Toy test (small dataset)

Generate a tiny synthetic dataset:

```bash
python3 ./scripts/06_generate_toy_cuttag_data.py ./sandbox_outputs/toy_validation/toy_data
```

Run the full pipeline on toy data (requires `Rscript` + `Seurat`):

```bash
./scripts/run_cuttag_qc_pipeline.sh \
  ./sandbox_outputs/toy_validation/toy_data/toy_ADT_counts.tsv \
  ./sandbox_outputs/toy_validation/toy_data/toy_HTO_counts.tsv \
  ./sandbox_outputs/toy_validation/toy_data/toy_fragments.tsv.gz \
  ./sandbox_outputs/toy_validation/toy_data/toy_peaks.bed \
  ./sandbox_outputs/toy_validation/toy_out toy 10 10 100
```

If you want to test without R/Seurat, run downstream steps using generated mock metadata:

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
  ./sandbox_outputs/toy_validation/toy_out/toy_clean_barcodes.tsv \
  ./sandbox_outputs/toy_validation/toy_out/toy
```
