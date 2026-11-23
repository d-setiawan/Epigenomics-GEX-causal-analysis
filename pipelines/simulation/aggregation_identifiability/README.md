# Aggregation identifiability

Which features of a cell-level graph survive the construction of pseudobulk rows? The answer
depends on how the cells in one assay relate to those in the other. In a linear model, averaging
jointly observed variables over the same cells with fixed weights preserves the structural
equations. Independently sampled assay groups have no paired residuals. Grouping them by a
shared biological state can retain the covariance of state-dependent means. The proposed
two-assay study compares these regimes at a fixed total cell budget.
[Aggregation derivation](../../../project_resources/research/theory/06_latent_space_aggregation.md)
· [Proposed study](../../../project_resources/research/aggregation_restart_plan.md)

| Existing evidence | What the comparison measures |
| --- | --- |
| [Paired-cell simulation](results/simulation_identifiability_results.md) | The control clusters jointly measured cells on their true latent coordinates. Its `distinguish` score counts two planted peak–gene adjacencies and the exclusion of two state-confounded decoys. The tier prior supplies peak-to-gene orientation. Detection of the two additional null peaks is reported separately. [Scoring code](scripts/simulate_aggregation_identifiability.py) |
| [Metacells and RNA-anchor neighbors](results/aggregation_bakeoff.md) | An empirical CD14 comparison measured stability and null calibration under overlapping neighbor averages and disjoint metacells. |
| [Cross-mark study](results/crossmark_identifiability.md) | The count and depth experiments describe signal within versus across marks. The planted-graph partition study found enrichment for shared drivers among stable pairs. Use the dated corrections and sections 14–17 when interpreting the earlier count results. |

The paired simulation uses the analysis environment with `causal-learn`. From the repository
root this command reproduces its default seeded sweep and writes a separate summary.
[Environment guide](../../../envs/README.md)

```bash
python pipelines/simulation/aggregation_identifiability/scripts/simulate_aggregation_identifiability.py \
  --out-json /tmp/aggregation-paired-summary.json
```

The historical multichannel runs also require the external `paircheck` modules and, for Tetrad
discovery, its runtime and JAR. Their code, assumptions and results are documented separately.
The integration study measures recoverability before aggregation.
[Multichannel adapter](scripts/paircheck_multichannel.py) ·
[Simulation design and dependencies](../../../project_resources/research/theory/04_simulation.md)
· [Integration study](../integration_recoverability/README.md)
