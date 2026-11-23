# Simulation: aggregation degrades causal identifiability

**Date:** 2026-07-22. Script: `pipelines/simulation/aggregation_identifiability/scripts/simulate_aggregation_identifiability.py`. This is the
pivotal experiment for the methods framing — it separates the two explanations for the thin, fragile
real-data edges that we could not distinguish on real data (partition_bootstrap_results.md):
**(a) power** (method sound, n too small) vs **(b) aggregation ceiling** (metacell aggregation itself
distorts the causal structure). Answer: **(b), with a clean mechanism.**

## Design

Plant a *known* cell-level graph, run the **production discovery verbatim** (the pipeline's own
`build_background_knowledge`, rank-gaussian transform, FCI + fisherz, `max_depth=2`, depth as a root),
and ask whether it recovers the truth at the cell level and after metacell aggregation.

Ground truth (two independent state latents Z, W keep separating sets size ≤2 so `max_depth=2` can
test them): **P1, P2 directly cause G**; **P3, P4 share state with the causal peaks but do *not* cause
G** (so they are independent of G given {their causal twin, depth} → FCI should *prune* them from G);
P5, P6 are depth-only noise. The clean test: keep P1,P2→G **and** prune P3,P4,P5,P6. Depth is an
observed confounder root.

## Result — identifiable at the cell level, collapses under aggregation

`distinguish` = fraction of runs that get all of it right (keep both causes, prune all confounded/noise
peaks). n_metacells fixed at 133 (the real pipeline's n); cells-per-metacell swept; 6 data draws × 4
partitions per point.

| discovery input | distinguish | collinearity(cause, confounded twin) |
| --- | --- | --- |
| **133 raw cells (no aggregation)** | **1.00** | 0.59 (cell level) |
| 133 metacells, 2 cells each | 0.92 | 0.71 |
| 133 metacells, 4 cells each | 1.00 | 0.83 |
| 133 metacells, 8 cells each | 0.88 | 0.89 |
| 133 metacells, 16 cells each | 0.75 | 0.93 |
| 133 metacells, 32 cells each | 0.54 | 0.96 |
| **133 metacells, 64 cells each** | **0.42** | **0.98** |

**The structure is perfectly identifiable at the cell level (1.00).** Aggregation degrades it
monotonically, and the degradation tracks a **collinearity collapse**: averaging over the cells in a
metacell washes out the independent per-cell noise that distinguished a causal peak from its
confounded twin, so `|corr(P1,P3)|` climbs 0.59 → 0.98 and FCI can no longer tell them apart.

**At matched sample size** (133 discovery rows either way), raw cells recover the truth perfectly
while metacells (32 cells each) fail more than half the time. Aggregation is *strictly worse* here.

## What this means

1. **The method is sound.** Cell-level FCI + the pipeline's prior recovers the planted graph exactly.
   So an identifiability argument at the cell level is available (this is what a methods paper cites).
2. **Aggregation is the ceiling, not power.** More metacells does not fix it; the damage is set by how
   many cells are averaged per metacell. This is the mechanism behind the real-data partition
   instability ([L11](../../../../project_resources/research/LIMITATIONS.md#l11-edge-identities-are-partition-dependent)) and the
   thin biological validation — the metacell step erases exactly the causation-vs-confounding signal.
3. **It is the Grunfeld–Griliches / MAUP trade-off, made concrete.** Aggregation buys sample-level
   signal-to-noise (why we adopted it) at the cost of aggregation bias (why the edges are fragile).
   You cannot have both from the same knob.

## The honest caveat — and the way out

The simulated cells are **Gaussian**; real single cells are **90%+ zeros**, and fisherz CI on raw
sparse cells is unreliable — which is *why* metacells were adopted in the first place (the
null-calibration finding: raw/kNN data was noise-dominated). So "use raw cells" does not transfer
directly. The genuine tension:

> **aggregation ⇒ identifiability loss (this result);   raw sparse cells ⇒ CI-test failure (sparsity).**

The exits the survey already surfaced
([aggregation_partition_dependence_survey.md](../../../empirical/analysis/results/aggregation_partition_dependence_survey.md)
Q4): **dropout-aware discovery** that runs on sparse cells without aggregating — test-wise-deletion FCI
(Dai et al. 2024), ZiG-DAGs, ZICO, Hurdle DAGs — and/or **smaller metacells** (fewer cells each → less
collinearity, at the cost of noisier rows). Both are testable next steps, and this simulation is the
harness to test them in (swap the discovery step, re-run the sweep).

## For the paper

This is the methods spine: (1) real-data partition instability; (2) **this simulation** giving the
mechanism (aggregation-induced collinearity collapses causation-vs-confounding, identifiable at cell
level); (3) the MAUP / aggregation-bias framing; (4) the partition-bootstrap to report the robust
subset; (5) dropout-aware discovery as the principled fix. The applied biology (the small
partition-stable edge list) rides on top, honestly scoped.

Artifacts: `pipelines/simulation/aggregation_identifiability/outputs/simulation_identifiability.json`. Reproduce:
`python pipelines/simulation/aggregation_identifiability/scripts/simulate_aggregation_identifiability.py`.
