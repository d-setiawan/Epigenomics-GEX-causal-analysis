#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Integration/scripts/scglue/run_preprocess_joint.sh <RUN_ID> [extra args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ID> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

python3 Integration/scripts/setup_integration_workspace.py

python3 Integration/scripts/scglue/preprocess_joint_scglue.py \
  --run-id "$RUN_ID" \
  "$@"
