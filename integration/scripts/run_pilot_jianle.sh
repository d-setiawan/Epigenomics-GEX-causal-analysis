#!/usr/bin/env bash
set -euo pipefail

# Placeholder launcher for pilot Jianle-method integration.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 integration/scripts/setup_integration_workspace.py

echo
echo "Pilot workspace ready for Jianle method."
echo "Use integration/manifests/scglue_input_manifest.tsv as shared input manifest."
echo "Implement method script under integration/methods/jianle/."
