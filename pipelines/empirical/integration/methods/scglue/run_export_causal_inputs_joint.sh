#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash pipelines/empirical/integration/methods/scglue/run_export_causal_inputs_joint.sh <RUN_ID> <GENES_CSV> [extra args...]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <RUN_ID> <GENES_CSV> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
GENES="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"

TRAIN_DIR="pipelines/empirical/integration/runs/scglue_${RUN_ID}/train"
GRAPH_DIR="pipelines/empirical/integration/runs/scglue_${RUN_ID}/graph"

python3 pipelines/empirical/integration/methods/scglue/export_causal_inputs_joint_scglue.py \
  --train-dir "$TRAIN_DIR" \
  --graph-dir "$GRAPH_DIR" \
  --genes "$GENES" \
  "$@"
