# Partition bootstrap — definitive result (K=10 × 15 genes)

**Date:** 2026-07-22. Script: `pipelines/empirical/analysis/scripts/partition_bootstrap.py`. Ten Leiden partitions
(seeds 0–9) of the same 13,410 monocyte cells, `--confounders depth_permark`, everything else fixed.
This is the honest denominator the project's edge claims needed, and it revises two earlier conclusions.
Background: [partition_stability_findings.md](partition_stability_findings.md) (why),
[aggregation_partition_dependence_survey.md](../../analysis/results/aggregation_partition_dependence_survey.md) (novelty).

## Per-partition summary

n is controlled: every partition has 130–133 complete metacells.

| seed | peak→gene | validated | peak↔peak | bidirected | same-mark diff (dist-matched) | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 11 | 230 | 135 | +0.105 | 0.002 |
| 1 | 8 | 5 | 242 | 127 | +0.105 | 0.007 |
| 2 | 11 | 4 | 248 | 131 | +0.006 | 0.418 |
| 3 | 7 | 5 | 231 | 116 | +0.086 | 0.019 |
| 4 | 11 | 6 | 236 | 118 | −0.013 | 0.623 |
| 5 | 9 | 6 | 251 | 139 | +0.069 | 0.045 |
| 6 | 3 | 2 | 229 | 115 | +0.060 | 0.090 |
| 7 | 9 | 5 | 245 | 120 | −0.003 | 0.514 |
| 8 | 6 | 5 | 230 | 112 | −0.009 | 0.598 |
| 9 | 12 | 7 | 245 | 129 | +0.056 | 0.103 |

## What is stable, what is not

**STABLE — aggregate counts.** peak↔peak (229–251) and bidirected (112–139) are tight across
partitions. The *amount* of latent element↔element structure — "~half of adjacencies are bidirected"
— is partition-robust. Distribution-level claims survive.

**NOT STABLE — individual peak→gene edges.** Same data, same n, seed only → peak→gene yield swings
**3 to 18** (validated 2 to 11). Of 59 distinct peak→gene edges in the union:

| recovered in | # edges | # validated |
| --- | --- | --- |
| ≥60% of partitions | 2 | 1 |
| ≥40% | 3 | 2 |
| ≥30% | 9 | 6 |
| ≥20% | 19 | 12 |
| exactly 1/10 | **40 (68%)** | — |
| **all 10 (CORE)** | **0** | **0** |

**No edge is recovered in all 10 partitions.** 68% of the union appears in a single partition. The
project's "11 validated edges" headline came from **seed 0, which is an outlier** (18 edges vs a median
of 9). Single-partition edge lists are not trustworthy; the frequency table is.

### The defensible stable-edge list (≥30% of partitions, validated)

| freq | edge | axis |
| --- | --- | --- |
| **6/10** | `chr5:140591420-140591815__H3K4me3 → CD14` | prediction (rE2G) |
| 4/10 | `chr2:197786015-197786538__H3K4me2 → PLCL1` | prediction (rE2G) |
| 3/10 | `chr5:140638860-140639949__H3K4me2 → CD14` | genetics (eQTL) — the old "flagship" |
| 3/10 | `chr2:113127422-113127624__H3K4me2 → IL1B` | |
| 3/10 | `chr11:1850726-1850993__H3K4me1 → LSP1` | |
| 3/10 | `chr20:23361457-23362256__H3K27ac → CST3` | |

The single most robust validated edge is the **rE2G-supported CD14 promoter-proximal H3K4me3 edge**
(6/10, validated in all 6). Notably the old flagship eQTL edge is only 3/10 — less robust than assumed.

## ⚠️ Correction — the same-mark enrichment is partition-fragile

**This revises [same_mark_finding.md](same_mark_finding.md) and [L12](../../../../project_resources/research/LIMITATIONS.md#l12-per-mark-technical-depth-is-not-conditioned-on).** On 2026-07-21 we concluded, from a 3-seed pilot (seeds
0/1/2), that the same-mark enrichment was "resolved as biological — the positive E↔E signature." The
full 10-partition test shows that conclusion was **partition-lucky**:

- Distance-matched same-mark enrichment is **significant (p<0.05) in only 4 of 10 partitions**, p<0.10 in 5.
- Mean effect **+0.046** (sd 0.044), range **−0.013 to +0.105** — *negative* in 3 partitions.
- The two strongest partitions are **seed 0 and seed 1** (+0.105) — seed 0 is the canonical partition
  every prior result used. The pilot's tight 0.36/0.36/0.33 was those two strong seeds plus a
  borderline third.

**What still holds:** the split-half test showed that *where* the enrichment appears it is not a
technical (shared-cell substrate) artifact, and per-mark depth does not explain it. So there may be a
real but **weak** same-mark tendency (mean +0.046 > 0). **What does not hold:** calling it an
established, robust positive biological signature. It is partition-fragile and cannot headline.

## Bottom line

The partition-bootstrap did its job: it converted "we have 11 validated edges and a positive same-mark
signature" into the honest picture —

1. **Robust:** aggregate latent-structure counts; the co-accessibility methodological contrast; and a
   *small* stable-edge list (2 edges ≥50%, 9 ≥30%), headed by the rE2G-validated CD14 edge.
2. **Partition-fragile:** the specific validated-edge list (seed-0 was lucky) and the same-mark
   enrichment (significant in 4/10).
3. **The strongest, most novel result is the instability itself** — a Modifiable Areal Unit Problem in
   single-cell causal discovery, with the partition-bootstrap as the quantifying method (survey:
   apparently undone before).

Artifacts: `pipelines/empirical/analysis/outputs/partition_bootstrap/edge_frequency.tsv`,
`partition_bootstrap_summary.json`.
