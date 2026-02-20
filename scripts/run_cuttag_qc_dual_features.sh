#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
  cat <<'USAGE' >&2
Usage:
  run_cuttag_qc_dual_features.sh \
    <adt.tsv> <hto.tsv> <fragments.tsv.gz> <peaks.bed> <chrom.sizes> \
    <out_dir> <sample_prefix> \
    <min_adt> <min_hto> <min_cuttag_fragments> [bin_size]

This runs the standard pipeline (including peak matrix) and additionally builds
fixed-bin chromatin matrix outputs.
USAGE
  exit 1
fi

ADT_TSV="$1"
HTO_TSV="$2"
FRAGMENTS_TSV_GZ="$3"
PEAKS_BED="$4"
CHROM_SIZES="$5"
OUT_DIR="$6"
SAMPLE_PREFIX="$7"
MIN_ADT="$8"
MIN_HTO="$9"
MIN_CUTTAG_FRAGMENTS="${10}"
BIN_SIZE="${11:-5000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_PREFIX="$OUT_DIR/$SAMPLE_PREFIX"
CLEAN_BARCODES_TSV="${OUT_PREFIX}_clean_barcodes.tsv"

"$SCRIPT_DIR/run_cuttag_qc_pipeline.sh" \
  "$ADT_TSV" "$HTO_TSV" "$FRAGMENTS_TSV_GZ" "$PEAKS_BED" "$CHROM_SIZES" \
  "$OUT_DIR" "$SAMPLE_PREFIX" "$MIN_ADT" "$MIN_HTO" "$MIN_CUTTAG_FRAGMENTS"

python3 "$SCRIPT_DIR/07_build_fixed_bin_matrix.py" \
  "$FRAGMENTS_TSV_GZ" "$CHROM_SIZES" "$CLEAN_BARCODES_TSV" "$OUT_PREFIX" \
  --bin-size "$BIN_SIZE"

echo "Dual-feature pipeline complete for sample: $SAMPLE_PREFIX"
echo "Additional fixed-bin outputs:"
echo "  ${OUT_PREFIX}_bin_chromatin_clean.mtx"
echo "  ${OUT_PREFIX}_bin_chromatin_clean_barcodes.tsv"
echo "  ${OUT_PREFIX}_bin_chromatin_clean_features.tsv"
echo "  ${OUT_PREFIX}_bins.bed"
