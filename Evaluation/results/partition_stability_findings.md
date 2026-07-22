# Partition stability: validated edge *identities* do not reproduce

**Date:** 2026-07-20, pilot 2026-07-21. **Status: confirmed; partition-bootstrap machinery built and piloted (see below). Read before citing any specific edge.**
Scripts: `Evaluation/scripts/compare_panels.py`, `CausalDiscovery/scripts/metacell_resolution_sweep.py`.

## The question

Discovery runs on **metacells** — cells aggregated into disjoint groups by Leiden clustering over the
scGLUE latent. That partition is one arbitrary draw from many possible partitions. Bootstrap stability
selection resamples **rows within a fixed partition**, so by construction it cannot see variance caused
by the partition itself. Nobody had checked whether the partition matters.

It matters more than anything else we have measured.

## Design

Three partitions of the same 13,410 monocyte cells, everything else held fixed — same gene panel, same
300 kb windows, same peak universe, same α, same CI test (rank-fisherz), same confounder roots and
orientation prior, same validation suite:

| arm | partition | usable n |
| --- | --- | --- |
| `pipeline_confounder_panel` | Leiden seed 0, target 150 | 133 |
| `pipeline_seed1` | Leiden **seed 1**, target 150 | 133 |
| `pipeline_res400_fisherz` | Leiden seed 0, target 400 | 240 |

Two controls ran first, and both passed — which is what makes the result attributable:

- **Determinism.** Re-running CD14 on the same dataset with the same settings reproduces the edge set
  exactly (4/4 edges, identical verdicts). The pipeline is not stochastic.
- **Peak universe.** Both arms select the identical 43 candidate peaks for CD14 (43 shared, 0 unique to
  either). The arms are not looking at different variables.

## Result 1 — changing the aggregation level: counts stable, identities not

Panel-wide, 15 genes, n=133 vs n=240:

| | n=133 | n=240 | shared |
| --- | --- | --- | --- |
| peak→gene edges | 19 | 19 | **2** |
| directed | 19 | 19 | — |
| **validated** | **11** | **11** | **0** |

The aggregate counts are strikingly reproducible. **Not one of the eleven validated edges is the same
edge.** IL4R's three validated edges vanish; CST3 and CTSC gain edges they did not have.

## Result 2 — same n, different seed: the partition alone does it

Six genes (those carrying validated edges), n=133 in both arms, only the Leiden random seed changed:

| gene | seed 0 validated | seed 1 validated | shared |
| --- | --- | --- | --- |
| CD14 | 4 | 2 | **2** |
| IL4R | 3 | 1 | 0 |
| LSP1 | 1 | 1 | 0 |
| PLCL1 | 1 | 0 | 0 |
| CD52 | 1 | 0 | 0 |
| CD151 | 1 | 0 | 0 |
| **total** | **11** | **4** | **2** |

**Re-clustering with a different random seed — same algorithm, same parameters, same n — loses 9 of 11
validated edges.** So this is not about the 133→240 aggregation level. It is the partition itself.

## Result 3 — what survives all three partitions

| edges surviving | count |
| --- | --- |
| all 3 partitions | **0** |
| 2 of 3 partitions | **2** (both CD14) |

The two survivors are exactly the project's most-cited edges:

- `chr5:140591420-140591815__H3K4me3 → CD14` — prediction axis (rE2G)
- `chr5:140638860-140639949__H3K4me2 → CD14` — genetics axis (eQTL); the flagship fci∩gfci edge

**The headline results are the robust ones; the long tail is not.** That is reassuring for the flagship
claims and damaging for the panel-level count.

## What this means

1. **"11 validated edges" is not a reproducible claim.** The number reproduces; the edges do not. Any
   statement of the form "peak X regulates gene Y" from a single partition needs a partition-stability
   qualifier that we cannot currently supply.
2. **Bootstrap stability selection is measuring the wrong variance component.** It resamples rows
   inside one partition. The dominant variance is *between* partitions, which it structurally cannot
   observe. The current "5 stable+validated" tier inherits this blind spot.
3. **Panel-level statistics may still be sound.** Counts (19 directed, 11 validated) and the
   element↔element proportions (53% latent) were stable across arms. Claims about *distributions*
   are better supported than claims about *individual edges* — but this needs testing directly, not
   assuming.

## Caveats on this finding itself

- **Scale:** 3 partitions, and the seed control covers 6 genes. Enough to show the effect is large; not
  enough to estimate it precisely.
- **`--no-stability`:** all three arms were run without bootstrap stability selection, for speed. This
  compares *single* runs, the least stable output the pipeline produces. Stability selection may filter
  to a partition-robust subset — **untested, and the obvious next check.**
- The effect size (9/11 lost) is far larger than run-to-run noise, and determinism was verified, so it
  is unlikely to be an artifact.

## Literature context (2026-07-21)

A cited survey —
[aggregation_partition_dependence_survey.md](../../CausalDiscovery/results/aggregation_partition_dependence_survey.md)
— found this failure mode is **largely uncharacterized**: it is a textbook Modifiable Areal Unit Problem
(the "zoning effect"), and **bootstrapping over metacell partitions appears genuinely undone**, so the
fix below is both a rigor step and a defensible methods contribution.

## Proposed fix — partition-level (outer) bootstrap

Replace, or wrap, the current within-partition bootstrap:

```
for seed in 1..K:            # OUTER: re-cluster the cells into a new partition
    build metacells(seed) -> matching dir -> dataset
    run discovery
report edge frequency ACROSS partitions   # this is the honest stability
```

An edge's stability becomes the fraction of *partitions* in which it is found, which is the quantity
the claims actually depend on. Cost is K full pipeline builds per panel, so K is a real budget
decision — but K=5 on the current 15-gene panel is affordable and would settle it.

This is arguably a methodological contribution rather than only a fix: metacell aggregation is
near-universal in single-cell causal discovery, and if partition choice dominates edge recovery, that
is worth reporting on its own.

## Partition-bootstrap PILOT (2026-07-21) — machinery works; instability confirmed with n controlled

`CausalDiscovery/scripts/partition_bootstrap.py` builds K partitions (Leiden seeds) end-to-end
(metacells → matching → dataset → panel, resumable) and reports each edge's recovery frequency across
partitions. Pilot: **K=3 seeds × 6 validated-edge genes** (CD14, IL4R, LSP1, PLCL1, CD52, CD151),
`--confounders depth_permark`.

**n is controlled** — the three partitions have 133 / 132 / 130 complete metacells, so anything below
is the *zoning effect*, not a sample-size effect.

| seed (partition) | peak→gene | validated | peak↔peak | bidirected | same-mark rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 13 | 11 | 133 | 83 | 0.361 |
| 1 | 4 | 3 | 144 | 86 | 0.360 |
| 2 | 7 | 3 | 145 | 87 | 0.333 |

Two findings, pointing in opposite directions — which is the whole story:

1. **Individual edges are severely partition-unstable.** Same data, same n, only the Leiden seed
   differs → peak→gene yield swings **13 / 4 / 7**. Across the 3 partitions, **ZERO peak→gene edges are
   recovered in all three**; only 3 edges appear in 2 of 3; 18 appear in just one. Even the flagship
   CD14 edges do not survive all three. `CORE (all partitions) = 0`.
2. **The distributional biology IS partition-stable.** The same-mark bidirected rate is **0.36 / 0.36 /
   0.33** across the three partitions, and bidirected counts are tight (83 / 86 / 87). The positive
   signature ([same_mark_finding.md](same_mark_finding.md)) reproduces across partitions even though the
   individual edges do not — direct confirmation of "distributions reproduce, identities don't."

**Sobering corollary:** seed 0 — the partition every prior result used — is **edge-rich** (13) relative
to re-seeds (4, 7). The project's "11 validated edges" headline came from an unusually productive
partition. Single-partition edge counts are not trustworthy; the frequency table is.

**Design lesson for the full run:** `CORE = found in 100% of partitions` is too strict (0 at K=3, worse
at higher K). Report the frequency **distribution** and pick a **threshold** (stability-selection style,
e.g. recovered in ≥50–80% of partitions), not all-or-nothing. The full run (K≈10 × 15 genes) is needed
for frequency resolution.

## Consequences for the plan

**Scaling should wait.** Scaling multiplies whichever edges a single partition happens to produce; it
does not address, and would obscure, the reproducibility problem. Sequence: partition-level stability
first, then scale with the stable definition in place.

See [ROADMAP](../../docs/ROADMAP.md) §0.5 and
[LIMITATIONS L11](../../docs/LIMITATIONS.md#l11-edge-identities-are-partition-dependent).
