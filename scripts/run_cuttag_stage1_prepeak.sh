#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  cat <<'USAGE' >&2
Usage:
  run_cuttag_stage1_prepeak.sh \
    <adt.tsv> <hto.tsv> <fragments.tsv.gz> \
    <out_dir> <sample_prefix> \
    <min_adt> <min_hto> [min_cuttag_fragments]

Outputs:
  <out_dir>/<sample_prefix>_hto_adt_metadata.tsv
  <out_dir>/<sample_prefix>_singlet_barcodes.tsv
  <out_dir>/<sample_prefix>_fragment_counts.tsv
  <out_dir>/<sample_prefix>_clean_cells.tsv
  <out_dir>/<sample_prefix>_clean_barcodes.tsv
  <out_dir>/<sample_prefix>_adt_clean_matrix.tsv
  <out_dir>/<sample_prefix>_hto_clean_matrix.tsv
USAGE
  exit 1
fi

ADT_TSV="$1"
HTO_TSV="$2"
FRAGMENTS_TSV_GZ="$3"
OUT_DIR="$4"
SAMPLE_PREFIX="$5"
MIN_ADT="$6"
MIN_HTO="$7"
MIN_CUTTAG_FRAGMENTS="${8:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

OUT_PREFIX="$OUT_DIR/$SAMPLE_PREFIX"
META_TSV="${OUT_PREFIX}_hto_adt_metadata.tsv"
FRAG_COUNTS_TSV="${OUT_PREFIX}_fragment_counts.tsv"
CLEAN_BARCODES_TSV="${OUT_PREFIX}_clean_barcodes.tsv"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Error: Rscript not found. Install R + Seurat to run HTO demux." >&2
  exit 1
fi

Rscript "$SCRIPT_DIR/01_demux_adt_hto.R" \
  "$ADT_TSV" "$HTO_TSV" "$OUT_PREFIX" "$MIN_ADT" "$MIN_HTO"

python3 "$SCRIPT_DIR/02_count_fragments.py" \
  "$FRAGMENTS_TSV_GZ" "$FRAG_COUNTS_TSV"

python3 "$SCRIPT_DIR/03_build_clean_cell_metadata.py" \
  "$META_TSV" "$FRAG_COUNTS_TSV" "$OUT_PREFIX" \
  --min-fragments "$MIN_CUTTAG_FRAGMENTS"

python3 "$SCRIPT_DIR/04_filter_matrix_by_barcodes.py" \
  "$ADT_TSV" "$CLEAN_BARCODES_TSV" "${OUT_PREFIX}_adt_clean_matrix.tsv"

python3 "$SCRIPT_DIR/04_filter_matrix_by_barcodes.py" \
  "$HTO_TSV" "$CLEAN_BARCODES_TSV" "${OUT_PREFIX}_hto_clean_matrix.tsv"

echo "Stage 1 complete: $SAMPLE_PREFIX"
echo "Key outputs:"
echo "  ${OUT_PREFIX}_clean_cells.tsv"
echo "  ${OUT_PREFIX}_clean_barcodes.tsv"
