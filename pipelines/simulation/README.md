# Simulation pipelines

These studies ask which relationships between RNA and chromatin can be recovered when each assay
observes different cells. Each track changes a different part of the observation process.

| Track | What the experiment tests | Recorded status |
| --- | --- | --- |
| [Integration recoverability](integration_recoverability/README.md) | Can an estimator infer one assay’s state-dependent signal from another assay? Can an embedding and cell-matching procedure reproduce paired-reference dependences? | The count-source estimator met its transport, covariance and prediction criteria on nine reused, exactly specified worlds with known source dimension. The original scGLUE/scMRDR benchmark remains incomplete. [Count-source experiment](integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md) · [Integration benchmark status](integration_recoverability/IMPLEMENTATION_STATUS.md) |
| [Aggregation identifiability](aggregation_identifiability/README.md) | Which relationships survive averaging the same cells, independent cells, or cells grouped by state? | Earlier metacell controls and cross-mark experiments are recorded. The proposed two-assay comparison at a fixed cell budget remains a research question. [Restart plan](../../project_resources/research/aggregation_restart_plan.md) |
| [Unpaired multimodal benchmark](unpaired_multimodal/README.md) | Can CA-BOSS recover which assays and features participate in shared state, and resolve specific pairs? | Parked after validation on the existing world bank. The checkpoint report defines the shared-state and exact-pair scores. [CA-BOSS results](unpaired_multimodal/results/ca_boss_v4_roadmap_status.md) |

For the exact-model study, start with the CPU replay setup and transfer checks. Git carries the
plans, software, selected results and receipts. A replay recreates the corresponding count
matrices and fitted outputs under a new run ID. [Reproduction guide](../../TRANSFER.md) ·
[Clean-checkout verification](../../reproduce/VERIFICATION.md)
