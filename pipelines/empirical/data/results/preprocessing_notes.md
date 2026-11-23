# Raw CUT&Tag preprocessing

↑ [Data README](../README.md) · [Central README](../../../../README.md) · [MISSION](../../../../project_resources/research/MISSION.md) (why) · [ROADMAP](../../../../project_resources/research/ROADMAP.md) (plan)

Turns raw single-cell CUT&Tag (`pipelines/empirical/data/raw/<MARK>/`) into the per-mark **peaks + matrices**
(`pipelines/empirical/data/outputs/<MARK>/`) that feed [integration](../../integration/README.md). One pass per histone mark:
`H3K27ac`, `H3K27me3`, `H3K4me1`, `H3K4me2`, `H3K4me3`, `H3K9me3`.

## Staged pipeline

1. Demultiplex ADT/HTO counts — [`01_demux_adt_hto.R`](../scripts/01_demux_adt_hto.R)
2. Count fragments per barcode — [`02_count_fragments.py`](../scripts/02_count_fragments.py)
3. Build clean cell metadata + barcode whitelists — [`03_build_clean_cell_metadata.py`](../scripts/03_build_clean_cell_metadata.py)
4. Filter fragments and call mark-specific peaks with MACS3 — [`run_cuttag_stage2_call_peaks.sh`](../scripts/run_cuttag_stage2_call_peaks.sh)
5. Build peak (cell × peak) matrices — [`05_build_chromatin_matrix.py`](../scripts/05_build_chromatin_matrix.py)
6. Build fixed-bin (cell × bin) matrices — [`07_build_fixed_bin_matrix.py`](../scripts/07_build_fixed_bin_matrix.py)

## Key outputs (per mark)

| File | Contents |
| --- | --- |
| `*_clean_cells.tsv` | QC-passed cell metadata |
| `*_clean_barcodes.tsv` | barcode whitelist used downstream |
| `*_peaks.bed` | mark-specific MACS3 peak intervals |
| `*_chromatin_clean.mtx` | cell × peak count matrix |
| `*_bin_chromatin_clean.mtx` | cell × fixed-bin count matrix |
| `*_bins.bed` | fixed-bin genomic intervals |

The active integration manifest
[`Integration/manifests/scglue_input_manifest.tsv`](../../integration/manifests/scglue_input_manifest.tsv)
points to the fixed-bin matrices, clean-cell metadata, bins BED, and peaks BED for all six marks.

Env: **`scrna_r`** (snapatac2 for the fixed-bin / peak matrices) — see [envs/](../../../../envs/README.md).
