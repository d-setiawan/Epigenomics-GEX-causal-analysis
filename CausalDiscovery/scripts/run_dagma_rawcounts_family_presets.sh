#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAUSAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python}"
CLI="${CLI:-${SCRIPT_DIR}/causal_cli.py}"
DATASET_ROOT="${DATASET_ROOT:-${CAUSAL_DIR}/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts}"
GENE_PANEL="${GENE_PANEL:-${CAUSAL_DIR}/configs/gene_panels/monocyte_cuttag_peak_genes.tsv}"

LAMBDA1_VALUES="${LAMBDA1_VALUES:-0.03,0.01,0.003}"
W_THRESHOLD_VALUES="${W_THRESHOLD_VALUES:-0.3,0.1,0.05}"
T_VALUES="${T_VALUES:-5,7}"
BACKGROUND_MODE="${BACKGROUND_MODE:-tiered_distal_promoter_expr}"
PLOT_LAYOUT="${PLOT_LAYOUT:-local}"
PEAK_BINARY_THRESHOLD="${PEAK_BINARY_THRESHOLD:-0.0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# Keep transform at 'none' here because the Bernoulli peak preset binarizes
# the peak columns itself and rejects additional transforms.
"${PYTHON_BIN}" "${CLI}" \
  dagma-sweep \
  --dataset-root "${DATASET_ROOT}" \
  --gene-panel "${GENE_PANEL}" \
  --family-configs bernoulli_peak_nb_gene,gaussian_nb,nb_nb \
  --dagma-loss-type l2 \
  --lambda1-values "${LAMBDA1_VALUES}" \
  --w-threshold-values "${W_THRESHOLD_VALUES}" \
  --T-values "${T_VALUES}" \
  --background-mode "${BACKGROUND_MODE}" \
  --transform none \
  --dagma-peak-binary-threshold "${PEAK_BINARY_THRESHOLD}" \
  --plot-layout "${PLOT_LAYOUT}"
