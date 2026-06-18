# Legacy Scripts

These scripts are not part of the active workflow anymore. They are kept for reference while the project continues to consolidate around:

```text
CausalDiscovery/scripts/causal_cli.py
```

## What Replaced Them

- `build_scglue_metacells.py`: superseded by `causal_cli.py match`, which builds RNA-anchored one-to-one pseudo-pairs from `scGLUE` `joint_v2`.
- `export_locus_matrix_scglue.py`: superseded by `causal_cli.py dataset`, which generates nearby CUT&Tag peak matrices.
- `export_locus_panel_scglue_matches.py`: superseded by panel-based dataset generation through `causal_cli.py dataset --gene-panel ...`.
- `run_300kb_nearby_peak_sweep.sh`: superseded by `causal_cli.py sweep` for PC/FCI and `causal_cli.py dagma-sweep` for DAGMA.

If you are starting fresh, do not run scripts from this directory. Use the CLI and the shell helpers one level up.
