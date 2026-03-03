# Integration Module

This directory is the repo-native home for all downstream integration work.

It is designed for two methods in parallel:
1. `scGLUE`
2. `jianle` (your second integration method from `papers/IntegrationJianLe.pdf`)

## Layout

- `scripts/`: shared setup and validation scripts.
- `manifests/`: tabular input manifests generated from `Data/`.
- `configs/`: method configs and environment templates.
- `methods/scglue/`: scGLUE-specific code and notes.
- `methods/jianle/`: Jianle-method-specific code and notes.
- `workspace/`: local symlinked input view (generated, gitignored).
- `outputs/`: integration outputs (generated, gitignored).
- `logs/`: run logs (generated, gitignored).

## Quick Start

From repo root:

```bash
python3 integration/scripts/setup_integration_workspace.py
bash integration/scripts/check_scglue_env.sh
```

After setup, use:
- `integration/manifests/scglue_input_manifest.tsv`
- `integration/manifests/step1_data_check.tsv`

These are generated with repo-relative paths so scripts are portable across machines as long as the same repo layout is kept.
