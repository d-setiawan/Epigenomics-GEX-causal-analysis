#!/usr/bin/env bash
set -euo pipefail

# Placeholder launcher for pilot scGLUE integration.
# It currently validates workspace + prints next command stubs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 integration/scripts/setup_integration_workspace.py

echo
echo "Pilot workspace ready."
echo "Use one mark from: integration/manifests/scglue_input_manifest.tsv"
echo "Then run your pilot training script from integration/methods/scglue/."
