#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash pipelines/empirical/integration/methods/scglue/run_build_label_harmonization_template.sh [extra args...]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"

python3 pipelines/empirical/integration/methods/scglue/build_label_harmonization_template.py "$@"
