#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Integration/scripts/jianle/run_validate_joint.sh <RUN_ID> [extra args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ID> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

TRAIN_DIR="Integration/outputs/jianle/joint/${RUN_ID}/train"

python3 Integration/scripts/jianle/validate_joint_jianle.py \
  --train-dir "$TRAIN_DIR" \
  "$@"
