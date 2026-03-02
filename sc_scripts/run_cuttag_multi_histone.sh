#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  cat <<'USAGE' >&2
Usage:
  run_cuttag_multi_histone.sh \
    <data_root> <chrom.sizes> <genome_size> \
    <min_adt> <min_hto> <min_cuttag_fragments> \
    [bin_size] [macs_q] [mode]

Behavior:
  - Discovers samples from *_ADT_counts.tsv or *_ADT_counts.tsv.gz under <data_root>/*/
  - For each sample, expects matching HTO + fragments files in the same folder.
  - Runs both peak and fixed-bin outputs for each sample.

Modes:
  auto        If <out_dir>/<sample>_peaks.bed exists, reuse it; otherwise call peaks.
  reuse-peaks Require existing peaks.bed; fail if missing.
  call-peaks  Always call peaks with MACS3 (ignores any existing peaks.bed).

Defaults:
  bin_size = 5000
  macs_q   = 0.01
  mode     = auto

Optional:
  Set DRY_RUN=1 to print commands without executing them.

Example:
  ./sc_scripts/run_cuttag_multi_histone.sh \
    Data Data/chromsizes/hg38.chrom.sizes hs \
    10 10 100 5000 0.01 auto
USAGE
  exit 1
fi

DATA_ROOT="$1"
CHROM_SIZES="$2"
GENOME_SIZE="$3"
MIN_ADT="$4"
MIN_HTO="$5"
MIN_CUTTAG_FRAGMENTS="$6"
BIN_SIZE="${7:-5000}"
MACS_Q="${8:-0.01}"
MODE="${9:-auto}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Error: data_root does not exist: $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$CHROM_SIZES" ]]; then
  echo "Error: chrom sizes file not found: $CHROM_SIZES" >&2
  exit 1
fi

case "$MODE" in
  auto|reuse-peaks|call-peaks) ;;
  *)
    echo "Error: invalid mode '$MODE'. Use auto|reuse-peaks|call-peaks." >&2
    exit 1
    ;;
esac

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

pick_existing_file() {
  local path
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

processed=0
skipped=0

mapfile -t ADT_FILES < <(
  find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type f \
    \( -name "*_ADT_counts.tsv" -o -name "*_ADT_counts.tsv.gz" \) | sort
)

if [[ "${#ADT_FILES[@]}" -eq 0 ]]; then
  echo "No ADT files found under $DATA_ROOT (expected *_ADT_counts.tsv[.gz])." >&2
  exit 1
fi

for adt_path in "${ADT_FILES[@]}"; do
  sample_dir="$(dirname "$adt_path")"
  adt_file="$(basename "$adt_path")"

  case "$adt_file" in
    *_ADT_counts.tsv.gz) sample_prefix="${adt_file%_ADT_counts.tsv.gz}" ;;
    *_ADT_counts.tsv) sample_prefix="${adt_file%_ADT_counts.tsv}" ;;
    *)
      echo "Skipping unrecognized ADT filename: $adt_path" >&2
      skipped=$((skipped + 1))
      continue
      ;;
  esac

  hto_path="$(pick_existing_file \
    "$sample_dir/${sample_prefix}_HTO_counts.tsv" \
    "$sample_dir/${sample_prefix}_HTO_counts.tsv.gz" || true)"
  if [[ -z "${hto_path:-}" ]]; then
    echo "Skipping $sample_prefix: missing HTO file in $sample_dir" >&2
    skipped=$((skipped + 1))
    continue
  fi

  fragments_path="$(pick_existing_file \
    "$sample_dir/${sample_prefix}_histone_fragments.tsv.gz" \
    "$sample_dir/${sample_prefix}_histone_fragments.tsv" \
    "$sample_dir/${sample_prefix}_fragments.tsv.gz" \
    "$sample_dir/${sample_prefix}_fragments.tsv" || true)"
  if [[ -z "${fragments_path:-}" ]]; then
    echo "Skipping $sample_prefix: missing fragments file in $sample_dir" >&2
    skipped=$((skipped + 1))
    continue
  fi

  out_dir="$sample_dir/outputs"
  mkdir -p "$out_dir"

  peaks_bed="$(pick_existing_file \
    "$out_dir/${sample_prefix}_peaks.bed" \
    "$sample_dir/${sample_prefix}_peaks.bed" || true)"

  echo "================================================================="
  echo "Sample: $sample_prefix"
  echo "  ADT:       $adt_path"
  echo "  HTO:       $hto_path"
  echo "  Fragments: $fragments_path"
  echo "  Out dir:   $out_dir"

  if [[ "$MODE" == "call-peaks" ]]; then
    echo "  Action: call peaks + build peak/bin matrices"
    run_cmd "$SCRIPT_DIR/run_cuttag_chunks_full.sh" \
      "$sample_prefix" "$out_dir" \
      "$adt_path" "$hto_path" "$fragments_path" \
      "$CHROM_SIZES" "$GENOME_SIZE" \
      "$MIN_ADT" "$MIN_HTO" "$MIN_CUTTAG_FRAGMENTS" \
      "$MACS_Q" "$BIN_SIZE"
  elif [[ "$MODE" == "reuse-peaks" ]]; then
    if [[ -z "${peaks_bed:-}" ]]; then
      echo "Error: mode=reuse-peaks but no peaks file found for $sample_prefix." >&2
      echo "Expected: $out_dir/${sample_prefix}_peaks.bed or $sample_dir/${sample_prefix}_peaks.bed" >&2
      exit 1
    fi
    echo "  Action: reuse peaks at $peaks_bed + build peak/bin matrices"
    run_cmd "$SCRIPT_DIR/run_cuttag_qc_dual_features.sh" \
      "$adt_path" "$hto_path" "$fragments_path" "$peaks_bed" "$CHROM_SIZES" \
      "$out_dir" "$sample_prefix" \
      "$MIN_ADT" "$MIN_HTO" "$MIN_CUTTAG_FRAGMENTS" "$BIN_SIZE"
  else
    if [[ -n "${peaks_bed:-}" ]]; then
      echo "  Action: reuse peaks at $peaks_bed + build peak/bin matrices"
      run_cmd "$SCRIPT_DIR/run_cuttag_qc_dual_features.sh" \
        "$adt_path" "$hto_path" "$fragments_path" "$peaks_bed" "$CHROM_SIZES" \
        "$out_dir" "$sample_prefix" \
        "$MIN_ADT" "$MIN_HTO" "$MIN_CUTTAG_FRAGMENTS" "$BIN_SIZE"
    else
      echo "  Action: no peaks found; call peaks + build peak/bin matrices"
      run_cmd "$SCRIPT_DIR/run_cuttag_chunks_full.sh" \
        "$sample_prefix" "$out_dir" \
        "$adt_path" "$hto_path" "$fragments_path" \
        "$CHROM_SIZES" "$GENOME_SIZE" \
        "$MIN_ADT" "$MIN_HTO" "$MIN_CUTTAG_FRAGMENTS" \
        "$MACS_Q" "$BIN_SIZE"
    fi
  fi

  processed=$((processed + 1))
done

echo "================================================================="
echo "Completed multi-histone run"
echo "  Processed samples: $processed"
echo "  Skipped samples:   $skipped"
