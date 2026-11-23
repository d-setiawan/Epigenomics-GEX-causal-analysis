#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash pipelines/empirical/integration/methods/scmrdr/run_validate_joint.sh <RUN_ID> [extra args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ID> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"

TRAIN_DIR="pipelines/empirical/integration/runs/scmrdr_${RUN_ID}/train"

python3 pipelines/empirical/integration/methods/scmrdr/validate_joint_jianle.py \
  --train-dir "$TRAIN_DIR" \
  "$@"
