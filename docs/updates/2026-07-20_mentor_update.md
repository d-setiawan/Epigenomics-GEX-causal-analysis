# Project update — causal discovery with latent confounding on single-cell data

**Date:** 2026-07-20 · **Since we last spoke:** the evaluation layer was just starting.
*(Revised 2026-07-20 for a methods audience, and to incorporate a reproducibility result found after
the first draft — see §4.)*

**In one paragraph.** We run constraint-based causal discovery (FCI → PAG) on a small per-target
problem: ~43 covariates, 2 observed confounders, 1 outcome, n ≈ 133. The covariates are measurements of
regulatory activity at genomic locations near the target; the outcome is the target's expression level.
Causal sufficiency is certainly violated — unmeasured regulatory proteins drive many covariates at
once — so we use FCI rather than a DAG learner, and the **bidirected edges (explicit latent common
causes) are the object of interest**, not a nuisance. Two results below: the method claim is solid, and
a reproducibility problem in the sample construction is the current blocker.

---

## 1. The setup, without the biology

| in the paper's language | what it is statistically |
| --- | --- |
| target gene | the outcome variable *Y* |
| nearby chromatin "peaks" | ~43 covariates *X₁…X₄₃*, each measuring regulatory activity at one genomic location |
| 6 histone marks | 6 measurement channels; a location can appear once per channel |
| sequencing depth | 2 observed technical confounders, entered as **root nodes** |
| enhancer → gene regulation | a directed edge *Xᵢ → Y* |
| "element↔element architecture" | the edge structure **among the covariates** — the novel part |

One extra wrinkle worth knowing: **the covariates and the outcome are measured in different cells.**
Two separate assays. We align them in a shared latent space (a VAE-style embedding), cluster cells into
disjoint groups, and aggregate each group into one row. That construction step is where §4's problem
lives.

**Mechanistic prior.** We add a tiered constraint: confounders → covariates → outcome. Chromatin state
precedes transcription, and the outcome cannot cause a covariate measured in a different cell. This
forbids orientations we know are impossible; it does *not* encode which covariate matters.

---

## 2. Where the evaluation went since last time

| then | now |
| --- | --- |
| one perturbation database as evidence | **dropped** — zero coverage for our targets |
| an edge "supported" if its location overlapped anything annotated | **support must name the same outcome variable** |
| no evaluation at all of the covariate-covariate edges | **a full framework + 5 tests** |

The tightening was forced by a concrete failure: our then-flagship edge pointed at a regulatory element
that is real — but belongs to a *different* gene. Under the loose rule it validated. The rule is now
"the reference must link this element **to this specific outcome**," which that edge fails.

---

## 3. Result A — directed edges *Xᵢ → Y*

19 such edges across 15 targets. **All 19 are oriented** (the mechanistic prior resolves what the data
alone leaves ambiguous). 11 of 19 have external support from independent reference databases.

Three independent reference sources, each with a different failure mode, so agreement is meaningful:
a predictive model, a physical-contact assay, and a genetic-association catalogue. None of them is
used anywhere in fitting the graph.

**But see §4 before trusting the list of 11.**

---

## 4. ⚠️ The reproducibility problem (found today)

Recall that rows are *constructed*: cells are clustered into disjoint groups, and each group becomes
one sample. The clustering has a random seed.

I re-ran everything with **a different seed** — same algorithm, same parameters, same n = 133, same
covariates:

| | seed 0 | seed 1 | overlap |
| --- | --- | --- | --- |
| supported *Xᵢ → Y* edges | 11 | 4 | **2** |

**Nine of eleven supported edges do not survive re-clustering.** Across three different partitions,
**zero** edges appear in all three.

Two controls first, which is what makes this attributable:
- **Determinism** — same data, same settings reproduces the edge set exactly. Not algorithmic noise.
- **Same covariates** — both arms select the identical 43 covariates. Not a variable-selection artifact.

So it is the partition itself.

**Why our stability check missed it.** We do bootstrap stability selection — but it resamples **rows
within a fixed partition**. Between-partition variance is invisible to it by construction. That is a
real gap, and I think a general one: aggregating cells into constructed samples is near-universal in
this literature, and I have not seen anyone bootstrap *over the aggregation*.

**What survives is reassuring.** The 2 edges that persist across partitions are exactly the two most
externally-corroborated ones. The headline results are the robust ones; the tail is not.

**Also worth noting:** aggregate counts were extremely stable (19 edges / 11 supported reproduced
exactly at a different aggregation level). **Distribution-level claims held; individual-edge claims did
not.** That distinction matters for what we can write.

**Proposed fix:** an outer bootstrap over partitions — re-cluster K times, report edge frequency across
partitions. Redefines "stable edge" as the quantity the claims actually rest on.

---

## 5. Result B — the latent edges, and what the numbers mean

*This is the part worth reading closely; it's the contribution.*

FCI found **239 edges among covariates**. Their marks:

| mark | reading | count |
| --- | --- | --- |
| `<->` | **an unmeasured variable drives both** | **126 (53%)** |
| `o-o` | related, orientation undetermined | 113 |
| `-->` | one causes the other | 0 |

So **over half of the covariate-covariate structure is explicitly flagged as confounded** — and none is
inferred as direct causation.

### 5a. Trying to identify the latent — four negatives

A bidirected edge asserts a latent exists. The obvious next question is *what is it?* We had four
substantive candidates and tested each against a **distance-matched permutation null**: compare
bidirected pairs to non-adjacent pairs **at the same genomic separation**, permuting labels within
distance strata (2,000 permutations).

Distance matching is essential — nearby locations share everything by proximity, so an unmatched test
returns a false positive on essentially any signature. It immediately caught one: the contact-domain
signature looked mildly positive raw (7.1% vs 6.6%) and went to zero once matched.

| candidate latent | matched effect | p |
| --- | --- | --- |
| shared physical contact domain | +0.008 | 1.00 |
| shared binding motif | −0.017 | 1.00 |
| shared measured protein occupancy | +0.000 | 1.00 |
| genome-wide activity state | edges unchanged (126 → 129) | — |

All four null. Also robust to matching on **graph degree** as well as distance (hub covariates are both
more connected and more signature-rich — that confound is controlled too).

**Why this is an argument rather than a failure.** Had we found "protein P binds both," the honest move
would be to *measure P and put it in the model as an observed variable* — and the edge would stop being
latent. Instead the coupling survives every observable proxy we can construct. That supports the claim
that the confounder is **real but genuinely unmeasurable at this resolution**, which is precisely the
situation a bidirected edge is designed to represent.

*Honest caveat:* one axis (motif) was a weak test — 90% of *all* pairs share a motif, so it had no
discriminating power. The occupancy axis had full dynamic range and is the trustworthy null.

### 5b. The positive result — what the incumbent method cannot do

The standard approach in this field is thresholded correlation between locations. Run it on the same
data:

| pair type | mean \|Spearman ρ\| |
| --- | --- |
| bidirected (confounded) | **0.238** |
| other adjacent | **0.255** |
| non-adjacent (pruned) | 0.064 |

**0.238 vs 0.255 — indistinguishable.** Correlation cannot separate the confounded pairs from the
possibly-direct ones. It sees one population; we resolve two.

Taking the top decile of correlations as "links" (705 of them): **67% are pairs our method prunes
entirely**, 17% are the confounded class, 15% genuinely adjacent. So two-thirds of what correlation
calls a link is indirect, and another sixth is confounded.

**The theoretically clean part:** a graphical-lasso variant would remove the indirect ones — but it
**cannot** remove a shared *latent* driver. That's a guarantee, not an empirical claim, and it's why we
restrict the headline to the latent class.

---

## 6. What's novel, and what isn't

**Not novel — worth conceding up front:** the PAG edge taxonomy (that's just FCI); "bidirected = latent
common cause" on genomic data (MIIC, 2017 — but gene-level only); the predictive task of linking
regulatory elements to genes (largely solved); causal discovery on single-cell expression (done, but
assuming causal sufficiency); correlation-based coupling between elements (the incumbent).

**Novel — the combination:**
1. **Latent-variable causal discovery at the regulatory-element level** on this data type. Both
   literatures exist; they have not been joined here.
2. **Operationalizing a confound the incumbent literature explicitly concedes it cannot resolve** — and
   quantifying it (53%).
3. **A validation framework where no accepted null exists** — the distance-matched (now also
   degree-matched) test is itself a contribution.
4. **The elimination result:** the latent is real, local, and unmeasurable.
5. Possibly now: **partition-level instability of aggregation-based causal discovery** (§4), which may
   be a general methods finding rather than only our bug.

---

## 7. Two things I'd value your judgement on

1. **Does §4 kill the edge-level claims, or reframe them?** My read: distribution-level results survive
   (they reproduced exactly), individual-edge claims need the outer bootstrap before they can be
   stated. But you have better calibration than I do on how a reviewer treats "our headline number is
   reproducible but its constituents aren't."

2. **Is the four-negative result (§5a) publishable as a result?** It's an elimination argument, not a
   positive finding. It makes the case that latent-variable modelling is *required* here, but a
   reviewer could read it as "we looked and found nothing." That judgement decides whether we scale up
   next or first hunt for a positive signature.

**Current plan:** hold scaling until the partition-stability fix is in — scaling a non-reproducible
edge set would multiply the problem rather than resolve it.

---

*Detail:* `partition_stability_findings.md` · `peak_peak_validation_results.md` ·
`expanded_panel_confounder_results.md` · `LIMITATIONS.md`
