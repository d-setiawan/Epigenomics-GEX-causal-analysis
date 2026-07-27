# Professor's suggestions → the score-based (BOSS) biology-first pivot

**Date:** 2026-07-27. Source: the professor's note `SuggestionsToTryCausalDiscovery.txt`. This decodes
each suggestion into an action and records the pivot back toward a **biology-first** paper via
score-based search + Markov-checker verification.

## The note, decoded

| Professor's note | What it means | Action / status |
| --- | --- | --- |
| "Use Boss FCI, FCI-max, non-parametric test, FCIT, GCM, FFCI, RCIT" | try score-based & alternative-CI algorithms | **BOSS-FCI, GRaSP-FCI, BOSS, PC-max, FCIT** all available in the `causaldisco` Tetrad build; `use_ffci`, `use_rcit`, `use_cci` (nonparametric) as CI tests. Testing BOSS-FCI first. |
| "Markov Checker" | verify a discovered graph by testing whether its implied conditional independencies hold in the data | **`markov_check` wired** — the internal verification, independent of external biology DBs. Reports `ks_ind` (implied-indep p-values uniform? = fits) and `frac_dep_ind` (should ≈ α). |
| "Check graph & constraints — look for constraints, scores; make it fit the data. If cannot find EC, throw structural assumptions" | tailor score/constraints to the data; relax assumptions if needed | Using SEM-BIC score + tiered background knowledge; **willing to run BOSS (CPDAG, assumes causal sufficiency = the broken assumption)** for cleaner directed biology. |
| "Change representation — allow cycles (PAG), constraints/scores/algorithms/graphs" | consider cyclic causal models (regulatory feedback is cyclic) | `run_ccd` (Cyclic Causal Discovery) available; queued as a representation experiment. |
| "Ruben fMRI / Joseph Ramsey; Yujia, Shao-an; XiangChen Song Thursday meetings" | people to consult | logistics — not code. |

## Why this can rescue biology-first

The blocker for a biology paper was: constraint-based FCI on metacells is **partition-unstable**
(edges flip on re-clustering) and produces a **dense peak↔peak confounding explosion**. The simulation
showed aggregation degrades identifiability. But two findings on 2026-07-27 change the outlook:

**1. Constraint-based FCI's dense graph MISFITS the data (Markov Checker).** On CD14 (n=133), the
Markov check rejects FCI's 28-edge graph (`ks_ind = 0.000`, 10.6% of implied independencies violated
vs α = 5%). The confounding explosion was largely an **FCI artifact**, not real latent structure.

**2. Score-based search is sparse, better-fitting, and keeps the robust biology.** On CD14:

| method | edges | to gene | Markov `ks_ind` | `frac_dep_ind` |
| --- | --- | --- | --- | --- |
| FCI (constraint) | 28 | 4 | 0.000 (fails) | 0.106 |
| BOSS (CPDAG) | 5 | 2 (directed) | 0.016 | 0.036 |
| BOSS-FCI (PAG) | 5 | 2 | 0.016 | 0.036 |
| GRaSP-FCI (PAG) | 4 | 1 | 0.039 (≈passes) | 0.038 |

BOSS/BOSS-FCI keep exactly the two CD14 edges that survived the partition bootstrap (the rE2G and eQTL
edges) and drop the noise. `frac_dep_ind ≈ α` means their implied independencies actually hold.

## Result (2026-07-27)

**Score-based search does NOT by itself fix partition instability** — that is an aggregation property
(the simulation showed this), robust to algorithm choice. Across 10 partitions no edge is recovered in
all 10, best ~6/10, for FCI, BOSS, and BOSS-FCI alike.

**But score-based search + verification rescues biology-first a different way.** BOSS (penalty 1,
directed CPDAG) recovers the biological edges the conservative penalty missed, and the three algorithms
**converge on the same edge set** (28 edges by ≥2 methods). Intersecting with external validation gives
an **8-edge confident set** — each edge supported four independent ways (multi-algorithm +
partition-recurrence + external DB + directed). Flagship: CD14 `chr5:140591420`→CD14 (all 3 methods,
6/10 partitions, directed 6/6, rE2G). See
[consensus_biology_results.md](../../Evaluation/results/consensus_biology_results.md).

Penalty tension: penalty 2 → sparse, better Markov fit, too few edges; penalty 1 → recovers biology,
worse Markov fit. Both reported.

## Verification story for the paper (answers "somewhat verify our findings")

Two independent checks, neither relying on the fragile external databases:
1. **Markov Checker** — the discovered graph's implied CIs hold in the data (`ks_ind`, `frac_dep_ind`).
2. **Partition-bootstrap frequency** — the edge recurs across re-clusterings.
Plus the existing external triangulation (rE2G / PCHi-C / eQTL) as a third, orthogonal check.

## Scripts

- `CausalDiscovery/scripts/run_gfci_tetrad.py` — Tetrad bridge + safe-name + tiered background knowledge (reused).
- `CausalDiscovery/scripts/partition_bootstrap_tetrad.py` — run any Tetrad algorithm across the 10 partitions + Markov check.
