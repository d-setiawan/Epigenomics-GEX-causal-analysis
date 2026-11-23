#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat <<'USAGE' >&2
Usage:
  run_cuttag_stage3_downstream.sh \
    <fragments.tsv.gz> <peaks.bed> <chrom.sizes> \
    <clean_barcodes.tsv> <out_prefix> [bin_size]

Outputs:
  <out_prefix>_chromatin_clean.mtx
  <out_prefix>_chromatin_clean_barcodes.tsv
  <out_prefix>_chromatin_clean_features.tsv
  <out_prefix>_bin_chromatin_clean.mtx
  <out_prefix>_bin_chromatin_clean_barcodes.tsv
  <out_prefix>_bin_chromatin_clean_features.tsv
  <out_prefix>_bins.bed
USAGE
  exit 1
fi

FRAGMENTS_TSV_GZ="$1"
PEAKS_BED="$2"
CHROM_SIZES="$3"
CLEAN_BARCODES_TSV="$4"
OUT_PREFIX="$5"
BIN_SIZE="${6:-5000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/05_build_chromatin_matrix.py" \
  "$FRAGMENTS_TSV_GZ" "$PEAKS_BED" "$CHROM_SIZES" "$CLEAN_BARCODES_TSV" "$OUT_PREFIX"

python3 "$SCRIPT_DIR/07_build_fixed_bin_matrix.py" \
  "$FRAGMENTS_TSV_GZ" "$CHROM_SIZES" "$CLEAN_BARCODES_TSV" "$OUT_PREFIX" \
  --bin-size "$BIN_SIZE"

echo "Stage 3 complete"
echo "Key outputs:"
echo "  ${OUT_PREFIX}_chromatin_clean.mtx"
echo "  ${OUT_PREFIX}_bin_chromatin_clean.mtx"
