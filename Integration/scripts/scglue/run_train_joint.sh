#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Integration/scripts/scglue/run_train_joint.sh <RUN_ID> [extra args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ID> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PREPROCESS_DIR="Integration/outputs/scglue/joint/${RUN_ID}/preprocess"
GRAPH_DIR="Integration/outputs/scglue/joint/${RUN_ID}/graph"
OUT_DIR="Integration/outputs/scglue/joint/${RUN_ID}/train"

python3 Integration/scripts/scglue/train_joint_scglue.py \
  --preprocess-dir "$PREPROCESS_DIR" \
  --graph-dir "$GRAPH_DIR" \
  --out-dir "$OUT_DIR" \
  --no-cpu-only \
  "$@"
