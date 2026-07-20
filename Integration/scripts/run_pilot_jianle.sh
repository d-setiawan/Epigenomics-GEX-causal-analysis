#!/usr/bin/env bash
set -euo pipefail

# Launcher note for Jianle method track.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 Integration/scripts/setup_integration_workspace.py

echo
echo "Workspace ready for Jianle method."
echo "Preferred joint pipeline:"
echo "  bash Integration/scripts/jianle/run_joint_pipeline.sh <RUN_ID> [args]"
echo "Pilot single-mark preprocess is still available via:"
echo "  bash Integration/scripts/run_preprocess_jianle_pilot.sh <MARK> [args]"
