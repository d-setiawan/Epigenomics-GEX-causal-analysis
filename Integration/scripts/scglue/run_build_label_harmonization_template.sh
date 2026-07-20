#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Integration/scripts/scglue/run_build_label_harmonization_template.sh [extra args...]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

python3 Integration/scripts/scglue/build_label_harmonization_template.py "$@"
