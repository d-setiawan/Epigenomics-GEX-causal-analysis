#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis}"
PYTHON_BIN="${PYTHON_BIN:-/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python}"
RUN_ID="${RUN_ID:-joint_v2}"
WINDOW_BP="${WINDOW_BP:-300000}"
ALPHA="${ALPHA:-0.05}"
INDEP_TEST="${INDEP_TEST:-kci}"
BACKGROUND_MODE="${BACKGROUND_MODE:-tiered_distal_promoter_expr}"
QUANT_MODE="${QUANT_MODE:-log1p_norm}"
GENE_PANEL="${GENE_PANEL:-$REPO_ROOT/CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/CausalDiscovery/outputs/datasets/$RUN_ID/monocyte_cuttag_peak_genes_300kb}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_scrna_r}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"

export MPLCONFIGDIR
export XDG_CACHE_HOME

echo "[dataset] generating nearby-peak datasets for window ${WINDOW_BP} bp with quant_mode=${QUANT_MODE}"
"$PYTHON_BIN" "$REPO_ROOT/CausalDiscovery/scripts/generate_monocyte_cuttag_peak_datasets.py" \
  --run-id "$RUN_ID" \
  --gene-panel "$GENE_PANEL" \
  --window-bp "$WINDOW_BP" \
  --quant-mode "$QUANT_MODE" \
  --out-root "$OUT_ROOT"

declare -a GENE_DIRS=("csf1r" "cd14" "il1b")
declare -a METHODS=("pc" "fci")
declare -a DEPTHS=("1" "2" "3" "4" "5")

for gene_dir in "${GENE_DIRS[@]}"; do
  matrix_tsv="$OUT_ROOT/$gene_dir/nearby_peaks/$(tr '[:lower:]' '[:upper:]' <<< "${gene_dir:0:1}")${gene_dir:1}_nearby_peak_matrix.tsv"
  case "$gene_dir" in
    csf1r) matrix_tsv="$OUT_ROOT/$gene_dir/nearby_peaks/CSF1R_nearby_peak_matrix.tsv" ;;
    cd14) matrix_tsv="$OUT_ROOT/$gene_dir/nearby_peaks/CD14_nearby_peak_matrix.tsv" ;;
    il1b) matrix_tsv="$OUT_ROOT/$gene_dir/nearby_peaks/IL1B_nearby_peak_matrix.tsv" ;;
  esac

  for method in "${METHODS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
      echo "[run] gene=${gene_dir} method=${method} depth=${depth} background=${BACKGROUND_MODE}"
      "$PYTHON_BIN" "$REPO_ROOT/CausalDiscovery/scripts/run_pc_causallearn.py" \
        --method "$method" \
        --matrix-tsv "$matrix_tsv" \
        --indep-test "$INDEP_TEST" \
        --alpha "$ALPHA" \
        --background-mode "$BACKGROUND_MODE" \
        --max-depth "$depth"
    done
  done
done
