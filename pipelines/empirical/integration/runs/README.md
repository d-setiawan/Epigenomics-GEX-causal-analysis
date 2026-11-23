# Integration runs

Each run writes to a unique directory here. The run ID includes the method and
the run name, for example `scglue_joint_v2` or `scmrdr_scMRDR_gaussian`. Existing
artifacts were moved from the old method-specific output tree without changing
their contents.

For new experiments, keep the resolved run configuration under
[`../configs/runs/`](../configs/runs/), add its input-manifest reference and
environment to the run record, and use a new run ID. Do not write into an
existing run directory. Cross-run comparisons belong in
[`../comparisons/`](../comparisons/).

Run artifacts are gitignored; keep small configs and comparison summaries
tracked with the integration stage.
