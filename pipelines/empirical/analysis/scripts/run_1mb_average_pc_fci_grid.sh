#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAUSAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${CAUSAL_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
CLI="${CLI:-${SCRIPT_DIR}/causal_cli.py}"
DATASET_ROOT="${DATASET_ROOT:-${CAUSAL_DIR}/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10}"
GENE_PANEL="${GENE_PANEL:-${CAUSAL_DIR}/configs/gene_panels/monocyte_cuttag_peak_genes.tsv}"

METHODS="${METHODS:-pc,fci}"
DEPTHS="${DEPTHS:-1,2,3}"
INDEP_TESTS="${INDEP_TESTS:-fisherz,kci}"
ALPHAS="${ALPHAS:-0.01,0.05,0.1}"
BACKGROUND_MODE="${BACKGROUND_MODE:-tiered_distal_promoter_expr}"
TRANSFORM="${TRANSFORM:-none}"
PLOT_LAYOUT="${PLOT_LAYOUT:-local}"
PLOT_MAX_EDGES="${PLOT_MAX_EDGES:--1}"
PLOT="${PLOT:-true}"
FCI_POSSIBLE_DSEP_DEPTH="${FCI_POSSIBLE_DSEP_DEPTH:--2}"
FCI_MAX_PATH_LENGTH="${FCI_MAX_PATH_LENGTH:--2}"
FASTKCI_K="${FASTKCI_K:-3}"
FASTKCI_J="${FASTKCI_J:-4}"
FASTKCI_ALPHA_PARAM="${FASTKCI_ALPHA_PARAM:-500}"
FASTKCI_FALLBACK="${FASTKCI_FALLBACK:-rcit}"
FASTKCI_USE_GP="${FASTKCI_USE_GP:-false}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

IFS=',' read -r -a INDEP_TEST_ARRAY <<< "${INDEP_TESTS}"
IFS=',' read -r -a ALPHA_ARRAY <<< "${ALPHAS}"

cd "${PROJECT_ROOT}"

echo "[run_1mb_average_pc_fci_grid] dataset_root=${DATASET_ROOT}"
echo "[run_1mb_average_pc_fci_grid] methods=${METHODS}"
echo "[run_1mb_average_pc_fci_grid] depths=${DEPTHS}"
echo "[run_1mb_average_pc_fci_grid] indep_tests=${INDEP_TESTS}"
echo "[run_1mb_average_pc_fci_grid] alphas=${ALPHAS}"
echo "[run_1mb_average_pc_fci_grid] plot=${PLOT}"
echo "[run_1mb_average_pc_fci_grid] plot_max_edges=${PLOT_MAX_EDGES} (-1 means unlimited)"
echo "[run_1mb_average_pc_fci_grid] fci_possible_dsep_depth=${FCI_POSSIBLE_DSEP_DEPTH} (-2 means reuse each depth)"
echo "[run_1mb_average_pc_fci_grid] fci_max_path_length=${FCI_MAX_PATH_LENGTH} (-2 means reuse each depth)"
echo "[run_1mb_average_pc_fci_grid] fastkci K=${FASTKCI_K} J=${FASTKCI_J} alpha_param=${FASTKCI_ALPHA_PARAM} fallback=${FASTKCI_FALLBACK} use_gp=${FASTKCI_USE_GP}"

PLOT_FLAG="--plot"
if [[ "${PLOT}" == "false" || "${PLOT}" == "0" || "${PLOT}" == "no" ]]; then
  PLOT_FLAG="--no-plot"
fi
FASTKCI_GP_FLAG=""
if [[ "${FASTKCI_USE_GP}" == "true" || "${FASTKCI_USE_GP}" == "1" || "${FASTKCI_USE_GP}" == "yes" ]]; then
  FASTKCI_GP_FLAG="--fastkci-use-gp"
fi

for indep_test in "${INDEP_TEST_ARRAY[@]}"; do
  for alpha in "${ALPHA_ARRAY[@]}"; do
    echo "[run_1mb_average_pc_fci_grid] starting indep_test=${indep_test} alpha=${alpha}"
    "${PYTHON_BIN}" "${CLI}" sweep \
      --dataset-root "${DATASET_ROOT}" \
      --gene-panel "${GENE_PANEL}" \
      --window-bp 1000000 \
      --methods "${METHODS}" \
      --depths "${DEPTHS}" \
      --alpha "${alpha}" \
      --indep-test "${indep_test}" \
      --fastkci-k "${FASTKCI_K}" \
      --fastkci-j "${FASTKCI_J}" \
      --fastkci-alpha-param "${FASTKCI_ALPHA_PARAM}" \
      --fastkci-fallback "${FASTKCI_FALLBACK}" \
      ${FASTKCI_GP_FLAG} \
      --background-mode "${BACKGROUND_MODE}" \
      --fci-possible-dsep-depth "${FCI_POSSIBLE_DSEP_DEPTH}" \
      --fci-max-path-length "${FCI_MAX_PATH_LENGTH}" \
      --transform "${TRANSFORM}" \
      --plot-layout "${PLOT_LAYOUT}" \
      --plot-max-edges "${PLOT_MAX_EDGES}" \
      "${PLOT_FLAG}"
  done
done

echo "[run_1mb_average_pc_fci_grid] finished"
