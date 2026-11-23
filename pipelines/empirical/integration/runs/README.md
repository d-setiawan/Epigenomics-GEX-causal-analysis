# Integration runs

Each full-data integration writes a local directory named for the method and run ID. Historical PBMC runs include `scglue_joint_v2` and `scmrdr_scMRDR_gaussian`. Their saved-embedding comparison and protein-marker checks are available in the [alignment report](../results/biological_alignment_2026-09-16.md).

Fitted weights, embeddings and large run directories are excluded from Git. To train from the restored PBMC inputs, use a fresh run ID because the legacy wrappers can overwrite stage outputs. Keep the resolved invocation in the [run configurations](../configs/runs/README.md). Cross-run artifacts go under `integration/comparisons/`, and portable summaries go under [integration results](../results/). [Restore and run](../../../../TRANSFER.md)
