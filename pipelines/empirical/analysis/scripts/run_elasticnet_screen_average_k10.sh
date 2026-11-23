#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAUSAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${CAUSAL_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
CLI="${CLI:-${SCRIPT_DIR}/causal_cli.py}"

DATASET_ROOT="${DATASET_ROOT:-${CAUSAL_DIR}/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10}"
OUT_ROOT="${OUT_ROOT:-${CAUSAL_DIR}/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_elasticnet_top30_crossfit}"
REPORT_ROOT="${REPORT_ROOT:-}"
GENE_PANEL="${GENE_PANEL:-${CAUSAL_DIR}/configs/gene_panels/monocyte_cuttag_peak_genes.tsv}"

TOP_K="${TOP_K:-30}"
N_SPLITS="${N_SPLITS:-5}"
N_REPEATS="${N_REPEATS:-20}"
RANDOM_STATE="${RANDOM_STATE:-13}"
L1_RATIOS="${L1_RATIOS:-0.1,0.25,0.5,0.75,0.9,0.95,1.0}"
ALPHA_MIN_EXP="${ALPHA_MIN_EXP:--4}"
ALPHA_MAX_EXP="${ALPHA_MAX_EXP:-1}"
N_ALPHAS="${N_ALPHAS:-60}"
MAX_ITER="${MAX_ITER:-20000}"
TOL="${TOL:-1e-4}"
COEF_TOL="${COEF_TOL:-1e-10}"
GENES="${GENES:-}"

cd "${PROJECT_ROOT}"

echo "[run_elasticnet_screen_average_k10] dataset_root=${DATASET_ROOT}"
echo "[run_elasticnet_screen_average_k10] out_root=${OUT_ROOT}"
if [[ -n "${REPORT_ROOT}" ]]; then
  echo "[run_elasticnet_screen_average_k10] report_root=${REPORT_ROOT}"
fi
echo "[run_elasticnet_screen_average_k10] top_k=${TOP_K}"
echo "[run_elasticnet_screen_average_k10] n_splits=${N_SPLITS}"
echo "[run_elasticnet_screen_average_k10] n_repeats=${N_REPEATS}"
echo "[run_elasticnet_screen_average_k10] n_alphas=${N_ALPHAS}"
echo "[run_elasticnet_screen_average_k10] l1_ratios=${L1_RATIOS}"
if [[ -n "${GENES}" ]]; then
  echo "[run_elasticnet_screen_average_k10] genes=${GENES}"
fi

GENES_ARGS=()
if [[ -n "${GENES}" ]]; then
  GENES_ARGS=(--genes "${GENES}")
fi

REPORT_ARGS=()
if [[ -n "${REPORT_ROOT}" ]]; then
  REPORT_ARGS=(--report-root "${REPORT_ROOT}")
fi

"${PYTHON_BIN}" "${CLI}" elastic-net-screen \
  --dataset-root "${DATASET_ROOT}" \
  --out-root "${OUT_ROOT}" \
  "${REPORT_ARGS[@]}" \
  --gene-panel "${GENE_PANEL}" \
  --top-k "${TOP_K}" \
  --n-splits "${N_SPLITS}" \
  --n-repeats "${N_REPEATS}" \
  --random-state "${RANDOM_STATE}" \
  --l1-ratios "${L1_RATIOS}" \
  --alpha-min-exp "${ALPHA_MIN_EXP}" \
  --alpha-max-exp "${ALPHA_MAX_EXP}" \
  --n-alphas "${N_ALPHAS}" \
  --max-iter "${MAX_ITER}" \
  --tol "${TOL}" \
  --coef-tol "${COEF_TOL}" \
  "${GENES_ARGS[@]}"

echo "[run_elasticnet_screen_average_k10] finished"
