#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash pipelines/empirical/integration/methods/scglue/run_preprocess_joint.sh <RUN_ID> [extra args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ID> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"

python3 pipelines/empirical/integration/scripts/setup_integration_workspace.py

python3 pipelines/empirical/integration/methods/scglue/preprocess_joint_scglue.py \
  --run-id "$RUN_ID" \
  "$@"
