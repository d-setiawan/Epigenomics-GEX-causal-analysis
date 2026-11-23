# Unpaired multimodal benchmark

CA-BOSS is the parked shared-state benchmark for disjoint assays. Version 4.59 is its audited
cross-assay checkpoint and version 4.32 supplies the assay-local layers. Versions 4.60–4.64
investigated failure modes. The checkpoint reports shared-state hyperedge precision and recall,
assay participation and sentinel errors, and coverage and size of ambiguity sets. Exact pair and
direction recovery are scored as identifiability stress tests.
[Checkpoint and metric definitions](results/ca_boss_v4_roadmap_status.md) ·
[Experiment log](results/channel_aware_boss_v4_experiment_log.md)

The formal analyses use the existing world bank. In the original workspace,
`worlds/formal_worlds/` and `runs/formal_runs/` contain generated inputs and outputs. The Git
handoff provides the world IDs, generator settings and historical manifest hashes. On seed
49011, clean-checkout verification reproduced all 12 scorer payloads and 32 of 40 discovery
payloads byte for byte. The differences involve alignment metadata or derived embedding and
centroid values. [World inventory](../../../reproduce/formal_worlds.tsv) ·
[Generator](scripts/multimodal_simulation_v4.py) · [Replay instructions](../../../TRANSFER.md) ·
[Verification record](../../../reproduce/VERIFICATION.md)

The current count-source and aggregation studies use different observation models and scoring
criteria. Their results have separate denominators. [Simulation tracks](../README.md)
