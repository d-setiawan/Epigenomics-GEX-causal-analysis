#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash integration/scripts/run_preprocess_jianle_pilot.sh <MARK> [extra args...]
# Example:
#   bash integration/scripts/run_preprocess_jianle_pilot.sh H3K4me1 --chrom-top-features 30000

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MARK> [extra args...]" >&2
  exit 1
fi

MARK="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 integration/scripts/setup_integration_workspace.py

python3 integration/methods/jianle/scripts/preprocess_pilot_jianle.py \
  --mark "$MARK" \
  "$@"
