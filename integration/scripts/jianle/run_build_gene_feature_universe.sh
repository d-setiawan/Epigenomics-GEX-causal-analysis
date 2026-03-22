#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash integration/scripts/jianle/run_build_gene_feature_universe.sh <RUN_ID> <GTF> [extra args...]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <RUN_ID> <GTF> [extra args...]" >&2
  exit 1
fi

RUN_ID="$1"
GTF="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

python3 integration/scripts/setup_integration_workspace.py

python3 integration/methods/jianle/scripts/build_joint_gene_feature_universe.py \
  --run-id "$RUN_ID" \
  --gtf "$GTF" \
  "$@"
