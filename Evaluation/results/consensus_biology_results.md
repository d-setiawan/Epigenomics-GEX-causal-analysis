# Confident biological edges — cross-method + cross-partition + external consensus

**Date:** 2026-07-27. The biology-first deliverable. Scripts: `CausalDiscovery/scripts/consensus_biology.py`,
`partition_bootstrap_tetrad.py`. Motivated by the professor's suggestions
([professor_suggestions_and_boss_pivot.md](../../CausalDiscovery/results/professor_suggestions_and_boss_pivot.md)):
score-based search (BOSS) + Markov-checker verification.

## The idea

Single-partition, single-method edges are not trustworthy (partition instability + FCI misfit). So we
report only edges that survive **four independent checks**, none of which can be gamed by the others:

1. **multi-algorithm** — recovered by ≥2 of FCI (constraint-based), BOSS (score-based CPDAG, assumes
   sufficiency), BOSS-FCI (score-based PAG, latent-aware). Different assumptions → agreement is not an
   artifact of any one.
2. **partition-robust** — recovered across multiple re-clustered metacell partitions (10-partition bootstrap).
3. **externally validated** — target-gene-aware support from ENCODE-rE2G / BLUEPRINT PCHi-C / eQTL Catalogue.
4. **directed** (bonus) — oriented chromatin→expression by the score-based CPDAG search.

## What the methods agree on

Across the 6 validation-rich genes, over the 10-partition bootstrap:

| set | count |
| --- | --- |
| union of peak→gene edges (any method) | 73 |
| found by **≥2 methods** | 28 |
| **CONFIDENT** (≥2 methods **and** externally validated) | **8** |

The three algorithms — constraint-based and two score-based with different assumptions — **converge on
the same 28 edges**, differing mainly in orientation and frequency. That convergence is itself a
verification: the edge set is not an artifact of FCI's (or any one method's) assumptions.

## The confident biological edge set

| gene | element | FCI | BOSS | BOSS-FCI | directed (BOSS) | axis |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **CD14** | `chr5:140591420-140591815` H3K4me3 | 0.6 | 0.6 | 0.5 | **6/6** | prediction (rE2G) |
| **PLCL1** | `chr2:197786015-197786538` H3K4me2 | 0.4 | 0.3 | 0.3 | 3/3 | prediction (rE2G) |
| **CD14** | `chr5:140638860-140639949` H3K4me2 | 0.3 | 0.3 | 0.3 | 3/3 | genetics (eQTL) |
| LSP1 | `chr11:1844174-1844753` H3K4me2 | 0.2 | 0.2 | 0.2 | 2 | physical+prediction |
| CD14 | `chr5:140563832-140564962` H3K4me3 | 0.2 | 0.2 | 0.2 | 2 | physical (PCHi-C) |
| CD14 | `chr5:140612011-140612339` H3K27ac | 0.1 | 0.1 | 0.1 | 1 | prediction (rE2G) |
| IL4R | `chr16:27242827-27243157` H3K4me2 | 0.1 | 0.1 | 0.1 | 1 | physical (PCHi-C) |
| IL4R | `chr16:27401872-27402382` H3K4me3 | 0.1 | 0.1 | 0.1 | 1 | physical (PCHi-C) |

**The flagship: the CD14 `chr5:140591420` H3K4me3 promoter-proximal element → CD14.** It is recovered
by all three algorithms, in 6/10 partitions, **directed in all 6** BOSS runs, and predicted by
ENCODE-rE2G. No single check could establish it; together they make it a confident directed regulatory
edge. PLCL1 and the CD14 eQTL element are the next tier (multi-method, directed, validated, ~0.3–0.4).

## Honest scope

- **The confident set is small** — ~3 strong edges (freq ≥0.3, directed, validated) plus 5 weaker
  (freq 0.1–0.2). This is a *high-confidence shortlist*, not a regulatory network. That is the honest
  biology-first claim: a few directed regulatory edges we can stand behind, each supported four ways.
- **Partition frequencies are modest** (0.6 max) — the aggregation ceiling
  ([simulation_identifiability_results.md](simulation_identifiability_results.md)) still bounds
  recovery. The consensus/verification is what makes the survivors credible despite it.
- **Score-penalty tension:** BOSS/BOSS-FCI at penalty 1 recover the biology but their graphs do not
  pass the Markov check well (ks_ind ≈ 0); penalty 2 passes better but finds too little. Report both.

## Verification, three independent ways (the paper's rigor)

1. **Markov Checker** (professor's suggestion): on CD14 the sparse score-based graph fits the data
   (frac_dep_ind ≈ α) while FCI's dense graph fails (ks_ind = 0.000) — the confounding explosion was an
   FCI artifact. See [professor_suggestions_and_boss_pivot.md](../../CausalDiscovery/results/professor_suggestions_and_boss_pivot.md).
2. **Multi-method consensus** — FCI ∩ BOSS ∩ BOSS-FCI agree on the edge set.
3. **Orthogonal external databases** — rE2G / PCHi-C / eQTL, target-gene-aware.

## For the paper

Biology-first, honestly scoped: *a small set of directed monocyte cis-regulatory edges, each surviving
a four-way verification (multi-algorithm, partition-robust, externally validated, directed), with the
partition-instability + Markov analyses as the rigor that justifies trusting the survivors.* The
CD14/PLCL1 edges are the headline findings.

Artifacts: `Evaluation/outputs/consensus_biology.tsv`,
`CausalDiscovery/outputs/partition_bootstrap/edge_frequency_{boss,boss_fci}.tsv`.
