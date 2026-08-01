# Is the cross-assay pairing manufacturing our bidirected edges?

**Date:** 2026-07-31, extended 2026-08-01. **Status: the composition test we designed turned out to have
no power (§3 correction). What the simulation *did* settle is an identifiability asymmetry —
same-channel structure survives the construction, cross-channel does not (§6) — plus an algorithm
result that reframes the project's headline (§7).**

> **Self-correction, same day.** The first version of this doc claimed the artifact was "refuted"
> (OR = 1.87, p = 7.3e-04). That statistic was computed on `peak_peak_pairs.tsv`, which comes from the
> **canonical seed-0 partition — one of the two most favourable of ten**. Recomputed across all 10
> partitions the effect is **mean +0.047 (sd 0.048), significant in 2/10**. Same failure mode, and
> nearly the same number, as the [K=10 correction to the same-mark
> finding](partition_bootstrap_results.md) (+0.046). The corrected claim is below and is much weaker.

Context: an outside methods advisor built a simulation harness (`paircheck`) testing whether the
"embed both assays → cluster → average → run FCI/GFCI" pipeline recovers cell-level causal structure
or something else. This doc records what that harness implies for us, the gap it does not cover, and
the real-data test that gap made possible.

---

## 1. The advisor's core claim: `G_matched`

Under matched (cross-assay) pairing, the covariate block and the outcome are averaged over **disjoint
cells**, so their residual terms are independent. Every covariate→outcome path running through the
error terms is **severed**; only covariation through the latent cell state `C` survives. His derived
target for matched data is therefore *the cell DAG with all covariate→outcome edges deleted*.

Sharp version, in his words: **matched data should recover `G_matched`, not `G_cell` — so every
recovered edge into the outcome is an artifact of the construction** (relative to `G_cell`).

His simulation confirms the prediction: `direct_outcome_rec` falls from ~0.44–0.48 (true pairing) to
~0.22–0.32 (matched) while the *count* of outcome-adjacent edges barely moves. The surviving edges are
enriched for the confounding-induced ones — i.e. the bidirected edges we treat as our object.

**The null arm is clean**: permuted pairing gives `direct_outcome_rec = 0.000` and essentially no
outcome-adjacent edges. **The apparatus does not fabricate cross-channel structure from nothing.** What
it changes is the *interpretation* of the structure it does find.

## 2. The gap: his generator has two channels, we have seven

His generator observes `(S, V₁…V_{p−1})` on one set of cells and `(S, V_p)` on a disjoint set. **All
covariates live in a single channel and share cells** — so every covariate–covariate pair in his
simulation is effectively `true`-paired, and the harness cannot exhibit a cross-channel
covariate–covariate pair at all.

Our data is not like that. Each histone mark is a **separate assay on separate cells**
([data_sources.md](../../Data/results/data_sources.md): scCUT&Tag-pro, one mark per experiment). Within
a metacell:

| pair type | assays | cells | analogue in his harness |
| --- | --- | --- | --- |
| **same-mark** peak↔peak | one assay | **shared** — both peaks quantified from the same mark-M cells | `pairing = true` |
| **cross-mark** peak↔peak | two assays | **disjoint** | `pairing = matched` |
| peak→gene | two assays | **disjoint** | `pairing = matched` |

**70.5% of our candidate peak pairs are cross-mark**, i.e. unpaired. So the majority of the
element↔element layer — our novel object — is subject to exactly the severing `G_matched` describes,
and his harness has no arm that reaches it.

## 3. The test this makes possible, and the result

The gap yields a sharp prediction. If unpairing *manufactures* bidirected marks — direct edges severed,
only the latent-state path surviving, which FCI must render `<->` — then **cross-mark pairs should be
more bidirected than same-mark pairs**.

Measured across **all 10 partitions** of the partition bootstrap (same-mark rate among bidirected
peak↔peak edges, against the non-adjacent baseline within each partition):

| seed | n bidirected | bidirected same-mark | baseline | diff | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 135 | 0.400 | 0.292 | **+0.108** | 0.010 |
| 1 | 127 | 0.402 | 0.294 | **+0.107** | 0.011 |
| 2 | 131 | 0.305 | 0.295 | +0.011 | 0.77 |
| 3 | 116 | 0.379 | 0.292 | +0.087 | 0.050 |
| 4 | 118 | 0.280 | 0.295 | −0.015 | 0.76 |
| 5 | 139 | 0.360 | 0.293 | +0.067 | 0.091 |
| 6 | 115 | 0.357 | 0.293 | +0.063 | 0.15 |
| 7 | 120 | 0.292 | 0.294 | −0.003 | 1.00 |
| 8 | 112 | 0.286 | 0.295 | −0.010 | 0.92 |
| 9 | 129 | 0.349 | 0.292 | +0.057 | 0.17 |

**Mean +0.047 (sd 0.048); positive in 7/10; significant in 2/10.** Seeds 0 and 1 are the two strongest
— and seed 0 is the canonical partition every earlier result was computed on.

**What this does and does not support.** The artifact hypothesis predicts a *cross*-mark enrichment
(negative diff). We never observe a meaningful negative — the worst partition is −0.015, and the
distribution is centred positive.

> ### ⚠️ Second correction (2026-08-01): this test has no power, in either direction
>
> The multi-channel simulation (§6) was built to check whether the artifact actually *leaves* a
> channel-composition signature. **It does not.** In the bidirected-rich regime, the fraction of
> bidirected edges that are same-channel is essentially identical under `true` and `matched` pairing
> (continuum 0.142 → 0.154; discrete 0.149 → 0.117; baseline 0.105). The construction does not
> differentially move bidirected edges between same- and cross-channel pairs.
>
> So the same-/cross-mark composition test **cannot distinguish artifact from biology at all** — the
> premise behind it (that unpairing would push bidirected edges onto cross-channel pairs) is false.
> This section is retained as a record of a test that turned out to be uninformative; it should not be
> cited as evidence in either direction. The live question moved to §6 and §7.

*(Pooling the ten partitions to gain power is exactly the move the advisor warned against — they
re-cluster the same cells, so they are not independent replicates and combining their p-values
manufactures evidence that was never collected.)*

### The one control that does hold up

The same-mark bidirected correlation **survives measuring the two peaks from disjoint cells** within a
mark — 104% of the excess persists over 20 random splits
([same_mark_finding.md](same_mark_finding.md)). *Where* the tendency appears, it is not shared per-cell
substrate. That control is about mechanism, not effect size, so partition-fragility of the effect does
not undo it.

| candidate artifact | test | verdict |
| --- | --- | --- |
| shared per-cell substrate | split-half, disjoint cells within a mark | refuted (104% survives) |
| cross-assay unpairing | same- vs cross-mark composition, 10 partitions | **no evidence for it** (mean +0.047 the *other* way, but weak: 2/10 significant) |

## 4. What this does *not* clear — two live threats

**(a) Legal PAG.** His README reports that `guaranteePag` defaults to **false** on `Fci`, `Gfci` and
`Bfci`, and that turning it on **roughly halves the bidirected fraction** at essentially no cost in
adjacency recall — because a meaningful share of bidirected edges under the non-CP variants **are not
part of any legal PAG**. Our entire headline is an edge *mark*, so this is a direct threat to the 53%
figure and is immediately testable.

**(b) Continuum geometry.** He finds edge-identity stability tracks whether the state space has real
boundaries: mean pairwise Jaccard across re-clustering seeds 0.75–0.91 (discrete) vs 0.56–0.73
(continuum), with count CV ≤ 0.16 in both. **Our observed stability is worse than his worst continuum
arm** (peak→gene edge-set Jaccard 0.25; 0 of 11 validated edges shared across partitions —
[partition_stability_findings.md](partition_stability_findings.md)). That is consistent with monocyte
cell state being a **continuum with no natural cluster boundaries**, which would make any hard
partition arbitrary — a mechanism for L11 rather than a fix for it.

**(c) An uncorrected selection we had not flagged.** He notes clusters missing a channel are dropped
and that this "preferentially removes sparse regions of state space — and is not corrected for
anywhere." That is exactly our completeness rule (all 6 marks + RNA). We drop 15/148 clusters at
n=133, but **192/432 (44%)** at the higher resolution. Uncorrected, and larger than we had appreciated.

## 5. Harness status and what to build

`paircheck` runs against our stack. One compatibility shim was needed: our jar (classes dated
2026-06-12) names BFCI `BossFci` and its superclass `StarFci`, where his newer beta uses `Bfci` /
`StarFciCheckPag`. **`setGuaranteePag` is present on our `BossFci`**, so `bfci-cp` is fully supported
here. His arm-vs-search result reproduces on our runs (`direct_outcome_rec` sd ratio 11.7×: the unit
construction dominates the choice of search by an order of magnitude).

**The extension we need — a multi-channel generator.** Split the covariates across `M` channels so that
cross-channel covariate–covariate pairs exist, then score the bidirected rate separately for
same-channel and cross-channel pairs under each pairing arm. That is the simulation counterpart of §3,
and it is the only way to check the §3 result against a known ground truth rather than by prediction
sign alone.

Secondary arms worth adding: a **two-latent** generator (a global state plus a locus-local state) to
test whether the pipeline can distinguish them — that is MISSION §2's actual claim, and elimination on
real data cannot settle it.

---

## 6. The multi-channel simulation (2026-08-01) — what it settled

Built `paircheck/multichannel.py` to close the gap in §2: covariates spread across **M channels**, the
outcome in its own, cells split M+1 ways disjointly, and a cluster contributes a row only if it holds a
cell from **every** channel (our completeness rule). The derived truth generalizes the advisor's
`G_matched`: **cross-channel direct edges are severed** (disjoint cells ⇒ independent residuals),
same-channel ones retained.

Dense regime (p=19, 6 covariate channels, `p_state_parent`=0.7, K=133, 4 data × 3 cluster seeds):

| geometry | pairing | `dir_same` | `dir_cross` |
| --- | --- | ---: | ---: |
| discrete | true | 0.213 | **0.337** |
| discrete | **matched** | 0.435 | **0.188** (−44%) |
| discrete | permuted | 0.889 | **0.002** |
| continuum | true | 0.361 | **0.241** |
| continuum | **matched** | 0.472 | **0.104** (−57%) |
| continuum | permuted | 0.833 | **0.007** |

**The identifiability result.** Matched (cross-assay) pairing destroys 44–57% of *cross-channel* direct
edges while leaving *same-channel* direct edges intact. The permuted arm drives cross-channel recall to
≈0 while same-channel stays high — the control behaviour that makes the effect attributable to the
pairing rather than to aggregation generally.

*(`dir_same` rising from `true` to `matched` is a density artifact, not a gain: severing cross-channel
edges thins the graph, so the surviving same-channel edges face less competition from alternative
conditioning sets. Read the `true`→`matched` contrast within a column, not across columns.)*

**Translated to our data:** same-mark peak↔peak structure is the part of the element↔element layer this
construction can recover. Cross-mark pairs — **70.5% of our candidate pairs** — largely cannot be, and
peak→gene edges sit in the same severed regime.

## 7. The algorithm discrepancy — and a reframing

Running the panel across the Tetrad family (new `tetrad-current.jar`, 2026-07-31 build, via
`run_gfci_tetrad.py --algo`):

| algo | edges | **bidirected** | `o-o` | oriented | peak→gene w/ arrowhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| GFCI | 65 | **0** | 49 | 16 | 0 |
| BFCI | 71 | **0** | 45 | 26 | 2 |
| BFCI-CP | 71 | **0** | 45 | 26 | 2 |
| causal-learn FCI *(our pipeline)* | ~239 peak↔peak | **126 (53%)** | 113 | 0 | — |

Two things follow.

**(a) BFCI substantially repairs [L10](../../docs/LIMITATIONS.md#l10-gfci-orients-the-skeleton-only).**
GFCI left cross-tier edges at `o-o`; BFCI orients 26 vs 16 and produces peak→gene edges *with
arrowheads*. BFCI-CP is byte-identical to BFCI here — there were no bidirected edges to legalize, so the
"legalizing halves the bidirected fraction" threat did not fire in this family.

**(b) Only FCI produces bidirected edges at all.** Every score-based search returns zero on the same
data. In the advisor's simulation the same pattern holds with ground truth available: BFCI `bi_frac`
≈0.03, LV-Heuristic exactly 0.000, and FCI's bidirected edges are largely **false positives**.

**The reframing this invites.** The value of a latent-variable search may be the structure it recovers
*despite* latents, not the latents it claims to *identify*. LV-Heuristic — the PAG of the BOSS DAG, pure
scoring, sufficiency assumed, no bidirected edges by construction — roughly **doubles** adjacency and
direct-outcome recall against every other search in every arm, at a real but modest precision cost
(≈0.78 vs ≈0.93). Critically, **the pairing contrast survives the search change**: permuted
`direct_outcome_rec` 0.023 vs matched 0.501.

If that holds, the 53%-bidirected headline is a property of FCI's behaviour rather than a finding about
chromatin, and the defensible contribution shifts to: **recover structure that survives the
construction, and restrict claims to the same-channel pairs where it provably does.**
