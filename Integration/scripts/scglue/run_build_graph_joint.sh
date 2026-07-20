#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Integration/scripts/scglue/run_build_graph_joint.sh <RUN_ID> <GTF> [extra args...]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <RUN_ID> <GTF> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
GTF="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PREPROCESS_DIR="Integration/outputs/scglue/joint/${RUN_ID}/preprocess"

python3 Integration/scripts/scglue/build_guidance_graph_joint_scglue.py \
  --preprocess-dir "$PREPROCESS_DIR" \
  --gtf "$GTF" \
  "$@"
