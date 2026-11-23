# Pipelines

Start with the row that matches the question you want to answer. The empirical PBMC analysis, the scMTR-seq evaluation and the count-transport simulation have different observation units and different standards of evidence.

| Question | Entry point |
| --- | --- |
| Where did the cells come from, and which identifiers are shared? | [Empirical data and provenance](empirical/data/README.md) |
| How were unpaired PBMC cells aligned, aggregated and analyzed at a locus? | [Empirical workflow](empirical/README.md) · [Integration runs](empirical/integration/README.md) · [Per-gene analysis](empirical/analysis/README.md) |
| What biological evidence supports a selected edge? | [Empirical evaluation](empirical/evaluation/README.md) |
| What does changing an aggregation operator or partition do? | [Aggregation and identifiability](simulation/aggregation_identifiability/README.md) |
| What did CA-BOSS recover across the cached multimodal worlds? | [Parked multimodal benchmark and scoring](simulation/unpaired_multimodal/README.md) |
| Can integration recover known structure, or a raw-count model recover shared-state transport? | [Integration-recoverability benchmarks](simulation/integration_recoverability/README.md) |
| What does disjoint fitting recover on the separate scMTR-seq S1 cohort? | [S1 data and split](empirical/data/scmtr_seq/README.md) · [Pilot results](../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md) |

Use the [transfer guide](../TRANSFER.md) for an empty checkout and the [verification record](../reproduce/VERIFICATION.md) for the stages exercised during the handoff. The [mission](../project_resources/research/MISSION.md) explains what each workflow can establish.
