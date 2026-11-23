#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAUSAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${CAUSAL_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"

# Comma-separated filtered dataset suffixes to run.
# Current filtered roots are:
#   monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz005
#   monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz010
DATASET_SUFFIXES="${DATASET_SUFFIXES:-nz005,nz010}"

# Match the current PC/FCI grid shape, but use Gaussian/Fisher-Z only.
METHODS="${METHODS:-pc,fci}"
DEPTHS="${DEPTHS:-1,2,3}"
ALPHAS="${ALPHAS:-0.01,0.05,0.1}"
INDEP_TESTS="${INDEP_TESTS:-fisherz}"

BACKGROUND_MODE="${BACKGROUND_MODE:-tiered_distal_promoter_expr}"
TRANSFORM="${TRANSFORM:-none}"
PLOT="${PLOT:-true}"
PLOT_MAX_EDGES="${PLOT_MAX_EDGES:--1}"
PLOT_LAYOUT="${PLOT_LAYOUT:-local}"

# Keep FCI's expensive search bounded consistently with each depth.
FCI_POSSIBLE_DSEP_DEPTH="${FCI_POSSIBLE_DSEP_DEPTH:--2}"
FCI_MAX_PATH_LENGTH="${FCI_MAX_PATH_LENGTH:--2}"

GENE_PANEL="${GENE_PANEL:-${CAUSAL_DIR}/configs/gene_panels/monocyte_cuttag_peak_genes.tsv}"
LOG_ROOT="${LOG_ROOT:-${CAUSAL_DIR}/outputs/run_logs/pc_fci_fisherz_filtered_average_k10}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

IFS=',' read -r -a DATASET_SUFFIX_ARRAY <<< "${DATASET_SUFFIXES}"

echo "[run_filtered_average_pc_fci_fisherz_grid] project_root=${PROJECT_ROOT}"
echo "[run_filtered_average_pc_fci_fisherz_grid] dataset_suffixes=${DATASET_SUFFIXES}"
echo "[run_filtered_average_pc_fci_fisherz_grid] methods=${METHODS}"
echo "[run_filtered_average_pc_fci_fisherz_grid] depths=${DEPTHS}"
echo "[run_filtered_average_pc_fci_fisherz_grid] alphas=${ALPHAS}"
echo "[run_filtered_average_pc_fci_fisherz_grid] indep_tests=${INDEP_TESTS}"
echo "[run_filtered_average_pc_fci_fisherz_grid] log_root=${LOG_ROOT}"

for suffix in "${DATASET_SUFFIX_ARRAY[@]}"; do
  dataset_root="${CAUSAL_DIR}/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_${suffix}"
  if [[ ! -d "${dataset_root}" ]]; then
    echo "[run_filtered_average_pc_fci_fisherz_grid] missing dataset root: ${dataset_root}" >&2
    exit 1
  fi

  log_file="${LOG_ROOT}/${suffix}_pc_fci_fisherz.log"
  echo "[run_filtered_average_pc_fci_fisherz_grid] starting suffix=${suffix}"
  echo "[run_filtered_average_pc_fci_fisherz_grid] dataset_root=${dataset_root}"
  echo "[run_filtered_average_pc_fci_fisherz_grid] log_file=${log_file}"

  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_ROOT="${dataset_root}" \
  GENE_PANEL="${GENE_PANEL}" \
  METHODS="${METHODS}" \
  DEPTHS="${DEPTHS}" \
  INDEP_TESTS="${INDEP_TESTS}" \
  ALPHAS="${ALPHAS}" \
  BACKGROUND_MODE="${BACKGROUND_MODE}" \
  TRANSFORM="${TRANSFORM}" \
  PLOT="${PLOT}" \
  PLOT_MAX_EDGES="${PLOT_MAX_EDGES}" \
  PLOT_LAYOUT="${PLOT_LAYOUT}" \
  FCI_POSSIBLE_DSEP_DEPTH="${FCI_POSSIBLE_DSEP_DEPTH}" \
  FCI_MAX_PATH_LENGTH="${FCI_MAX_PATH_LENGTH}" \
  bash "${SCRIPT_DIR}/run_1mb_average_pc_fci_grid.sh" 2>&1 | tee "${log_file}"
done

echo "[run_filtered_average_pc_fci_fisherz_grid] finished"
