#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat <<'USAGE' >&2
Usage:
  run_cuttag_stage2_call_peaks.sh \
    <fragments.tsv.gz> <clean_barcodes.tsv> \
    <out_dir> <sample_prefix> <genome_size> [macs_q]

Example genome_size:
  hs (human), mm (mouse), or effective genome size integer.

Outputs:
  <out_dir>/<sample_prefix>_clean_fragments.tsv.gz
  <out_dir>/<sample_prefix>_peaks.narrowPeak
  <out_dir>/<sample_prefix>_peaks.bed
USAGE
  exit 1
fi

FRAGMENTS_TSV_GZ="$1"
CLEAN_BARCODES_TSV="$2"
OUT_DIR="$3"
SAMPLE_PREFIX="$4"
GENOME_SIZE="$5"
MACS_Q="${6:-0.01}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

if ! command -v macs3 >/dev/null 2>&1; then
  echo "Error: macs3 not found. Install MACS3 to call peaks." >&2
  exit 1
fi

CLEAN_FRAG_TSV="${OUT_DIR}/${SAMPLE_PREFIX}_clean_fragments.tsv"
CLEAN_FRAG_GZ="${CLEAN_FRAG_TSV}.gz"
PEAK_PREFIX="${SAMPLE_PREFIX}_peaks"

python3 "$SCRIPT_DIR/08_filter_fragments_by_barcodes.py" \
  "$FRAGMENTS_TSV_GZ" "$CLEAN_BARCODES_TSV" "$CLEAN_FRAG_TSV"

gzip -f "$CLEAN_FRAG_TSV"

macs3 callpeak \
  -t "$CLEAN_FRAG_GZ" \
  -f BED \
  -g "$GENOME_SIZE" \
  -n "$PEAK_PREFIX" \
  --outdir "$OUT_DIR" \
  --nomodel --shift -100 --extsize 200 \
  -q "$MACS_Q"

awk 'BEGIN{OFS="\t"} NR>=1{print $1,$2,$3,$4}' \
  "${OUT_DIR}/${PEAK_PREFIX}_peaks.narrowPeak" > "${OUT_DIR}/${SAMPLE_PREFIX}_peaks.bed"

echo "Stage 2 complete: $SAMPLE_PREFIX"
echo "Key outputs:"
echo "  ${CLEAN_FRAG_GZ}"
echo "  ${OUT_DIR}/${PEAK_PREFIX}_peaks.narrowPeak"
echo "  ${OUT_DIR}/${SAMPLE_PREFIX}_peaks.bed"
