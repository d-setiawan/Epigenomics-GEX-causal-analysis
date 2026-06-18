#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAUSAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${CAUSAL_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
CLI="${CLI:-${SCRIPT_DIR}/causal_cli.py}"
ROOT="${ROOT:-${CAUSAL_DIR}/outputs/methods}"
FILTER="${FILTER:-}"
LAYOUT="${LAYOUT:-local}"
MAX_EDGES="${MAX_EDGES:--1}"
PEAKS_PER_ROW="${PEAKS_PER_ROW:-18}"
DRY_RUN="${DRY_RUN:-false}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "${PROJECT_ROOT}"

echo "[plot_existing_graphs] root=${ROOT}"
echo "[plot_existing_graphs] filter=${FILTER:-<none>}"
echo "[plot_existing_graphs] layout=${LAYOUT}"
echo "[plot_existing_graphs] max_edges=${MAX_EDGES} (-1 means unlimited)"
echo "[plot_existing_graphs] dry_run=${DRY_RUN}"

count=0
while IFS= read -r summary_path; do
  graph_dir=$(dirname "${summary_path}")
  if [[ -n "${FILTER}" && "${graph_dir}" != *"${FILTER}"* ]]; then
    continue
  fi

  count=$((count + 1))
  echo "[plot_existing_graphs] ${count}: ${graph_dir}"
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" || "${DRY_RUN}" == "yes" ]]; then
    continue
  fi

  "${PYTHON_BIN}" "${CLI}" plot \
    --graph-dir "${graph_dir}" \
    --layout "${LAYOUT}" \
    --max-edges "${MAX_EDGES}" \
    --peaks-per-row "${PEAKS_PER_ROW}"
done < <(find "${ROOT}" -name run_summary.json | sort)

echo "[plot_existing_graphs] processed ${count} graph directories"
