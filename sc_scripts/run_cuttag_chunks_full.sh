#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
  cat <<'USAGE' >&2
Usage:
  run_cuttag_chunks_full.sh \
    <sample_prefix> <out_dir> \
    <adt.tsv> <hto.tsv> <fragments.tsv.gz> \
    <chrom.sizes> <genome_size> \
    <min_adt> <min_hto> [min_cuttag_fragments] [macs_q] [bin_size]

Example:
  ./scripts/run_cuttag_chunks_full.sh \
    H3K4me1_rep1 Data/H3K4me1/outputs \
    Data/H3K4me1/H3K4me1_rep1_ADT_counts.tsv \
    Data/H3K4me1/H3K4me1_rep1_HTO_counts.tsv \
    /abs/path/fragments.tsv.gz \
    /abs/path/hg38.chrom.sizes hs \
    10 10 100 0.01 5000
USAGE
  exit 1
fi

SAMPLE_PREFIX="$1"
OUT_DIR="$2"
ADT_TSV="$3"
HTO_TSV="$4"
FRAGMENTS_TSV_GZ="$5"
CHROM_SIZES="$6"
GENOME_SIZE="$7"
MIN_ADT="$8"
MIN_HTO="$9"
MIN_CUTTAG_FRAGMENTS="${10:-100}"
MACS_Q="${11:-0.01}"
BIN_SIZE="${12:-5000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

echo "[1/3] Stage 1: pre-peak QC + clean barcodes"
"$SCRIPT_DIR/run_cuttag_stage1_prepeak.sh" \
  "$ADT_TSV" "$HTO_TSV" "$FRAGMENTS_TSV_GZ" \
  "$OUT_DIR" "$SAMPLE_PREFIX" "$MIN_ADT" "$MIN_HTO" "$MIN_CUTTAG_FRAGMENTS"

CLEAN_BARCODES_TSV="${OUT_DIR}/${SAMPLE_PREFIX}_clean_barcodes.tsv"
if [[ ! -s "$CLEAN_BARCODES_TSV" ]] || [[ "$(wc -l < "$CLEAN_BARCODES_TSV")" -le 1 ]]; then
  echo "Error: $CLEAN_BARCODES_TSV has no usable barcodes. Stopping before peak calling." >&2
  exit 1
fi

echo "[2/3] Stage 2: peak calling (MACS3)"
"$SCRIPT_DIR/run_cuttag_stage2_call_peaks.sh" \
  "$FRAGMENTS_TSV_GZ" "$CLEAN_BARCODES_TSV" \
  "$OUT_DIR" "$SAMPLE_PREFIX" "$GENOME_SIZE" "$MACS_Q"

PEAKS_BED="${OUT_DIR}/${SAMPLE_PREFIX}_peaks.bed"
if [[ ! -s "$PEAKS_BED" ]]; then
  echo "Error: peak BED was not created: $PEAKS_BED" >&2
  exit 1
fi

echo "[3/3] Stage 3: downstream matrices (SnapATAC2)"
"$SCRIPT_DIR/run_cuttag_stage3_downstream.sh" \
  "$FRAGMENTS_TSV_GZ" "$PEAKS_BED" "$CHROM_SIZES" \
  "$CLEAN_BARCODES_TSV" "${OUT_DIR}/${SAMPLE_PREFIX}" "$BIN_SIZE"

echo "Done. Key outputs:"
echo "  ${OUT_DIR}/${SAMPLE_PREFIX}_clean_cells.tsv"
echo "  ${OUT_DIR}/${SAMPLE_PREFIX}_clean_barcodes.tsv"
echo "  ${OUT_DIR}/${SAMPLE_PREFIX}_peaks.bed"
echo "  ${OUT_DIR}/${SAMPLE_PREFIX}_chromatin_clean.mtx"
echo "  ${OUT_DIR}/${SAMPLE_PREFIX}_bin_chromatin_clean.mtx"
