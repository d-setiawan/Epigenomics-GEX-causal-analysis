#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash integration/scripts/jianle/run_train_joint.sh <RUN_ID> [extra args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ID> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PREPROCESS_DIR="integration/outputs/jianle/joint/${RUN_ID}/preprocess"
OUT_DIR="integration/outputs/jianle/joint/${RUN_ID}/train"

python3 integration/methods/jianle/scripts/train_joint_jianle.py \
  --preprocess-dir "$PREPROCESS_DIR" \
  --out-dir "$OUT_DIR" \
  --no-cpu-only \
  "$@"
