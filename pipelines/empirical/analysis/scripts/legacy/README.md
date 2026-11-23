# Legacy analysis scripts

These scripts preserve earlier implementations of the empirical PBMC analysis. Choose the entry point according to the unit you want to analyze.

| Earlier script | Current entry point |
| --- | --- |
| `build_scglue_metacells.py` | [Metacell assignment](../metacell_assign.py) and [metacell matching tables](../build_metacell_matching_dir.py) support pooled groups. The separate `causal_cli.py match` and `match-average` commands construct RNA-anchored comparisons. |
| `export_locus_matrix_scglue.py` | `causal_cli.py dataset` builds nearby CUT&Tag peak matrices from a matching directory. |
| `export_locus_panel_scglue_matches.py` | `causal_cli.py dataset --gene-panel ...` applies the exporter to a panel. |
| `run_300kb_nearby_peak_sweep.sh` | `causal_cli.py sweep` runs PC/FCI and `dagma-sweep` runs DAGMA. |

The [script guide](../README.md) documents the commands. The [partition comparison](../../../evaluation/results/partition_stability_findings.md) shows why group membership matters for edge recovery.
