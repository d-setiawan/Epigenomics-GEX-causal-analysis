# How we build the samples we run causal discovery on

*A self-contained note for a methods reader. The biology is abstracted away; everything below is
stated as a statistics/graphical-modelling problem. Written to ask: **does the way we construct our
samples make sense, and is the non-i.i.d. framing of Tillman (2009) the right lens for its main
failure?***

Reference throughout: R. E. Tillman, *Structure learning with independent non-identically distributed
data*, ICML 2009.

---

## 1. The estimand

For each target we run **constraint-based and score-based-hybrid causal discovery** — FCI, and GFCI (a
score-based FGES skeleton search followed by FCI-style orientation), both → **PAG** — on a small problem:

- **outcome** *Y* — a scalar (the target's expression level),
- **covariates** *X₁…Xₚ*, `p` on the order of tens (~13–53; 43 in our running example) — each a
  measurement of regulatory activity at a genomic location near the target,
- **2 observed confounders** entered as root nodes (technical: sequencing depth),
- sample size **n ≈ 133**.

Causal sufficiency is certainly violated — unobserved regulators drive many covariates at once — so we
use FCI rather than a DAG learner, and the **bidirected (latent-common-cause) edges among the
covariates are the object of interest**, not a nuisance. About **53%** of the covariate–covariate
adjacencies come back bidirected.

## 2. The wrinkle that forces aggregation: the data is unpaired

The covariates and the outcome are measured in **different cells**, by two separate single-cell assays.
**No unit is jointly observed.** Of ~13,400 cells (one cell type), ~14% carry the outcome *Y*; the rest
carry covariate measurements, spread across **6 measurement channels** (a location can be measured once
per channel). So we never observe the joint (*X*, *Y*) at the level of the measured units — we have to
**construct** joint samples before any structure learning can happen.

This is the crux, and it's what removes the easy options. In Tillman's terms (§1 of his paper): we
cannot "use one large i.i.d. sample," because none exists; and we cannot "pool/concatenate the raw
data," because the raw units don't contain (*X*, *Y*) jointly at all.

## 3. What we actually do — the aggregation

We construct samples by **disjoint aggregation in a shared latent space**:

1. **Embed** every cell from both assays into one common latent space via a VAE-type model, so that
   cells in similar biological states sit near each other regardless of which assay measured them. (The
   embedding is learned; it is the cross-assay bridge.)
2. **Partition** that latent space into **disjoint clusters** (Leiden community detection). One draw.
3. **Keep** clusters that contain ≥1 cell from **every** channel and the outcome ("complete"), and
   **average** each such cluster into a single row. Every cell is used **at most once**.

The result is **n ≈ 133 constructed rows**, each a vector (*X*, *Y*), which we then treat as an i.i.d.
sample and hand to FCI.

## 4. Why we chose this construction

Three requirements drove it, and each maps to one property of the scheme:

| requirement | why | what delivers it |
| --- | --- | --- |
| **rows must be ~independent** | every FCI CI test is a function of `n`; overlapping neighbourhoods reuse cells → dependent rows → inflated `n` → miscalibrated tests (we measured this: with overlap, real edge counts sat *inside* a permutation-null band) | **disjoint** clusters — each cell used once, so `n_rows ≈ n_effective` |
| **variables must be continuous & low-noise** | per-cell signal is near-binary and sparse; Gaussian/rank CI tests need continuous, stable values | **averaging** within a cluster |
| **must bridge the unpaired gap** | (*X*, *Y*) is never jointly measured | **co-clustering both assays**: a cluster's *X* comes from its covariate-cells, its *Y* from its outcome-cells, under the assumption they share cell-state |

We deliberately accept a **small honest `n`** (≈133) over a large dishonest one. If 133 is too few for
FCI, we consider that a reportable result rather than something to inflate.

## 5. The problem we want your read on

**The partition in step 2 is one arbitrary draw, and the choice of draw dominates the edge-level
results.** Re-running everything with a different Leiden seed — *same data, same n = 133, same
covariate set, same α, same CI test, same priors*, only the partition reseeded:

| | seed A | seed B | shared |
| --- | --- | --- | --- |
| supported *Xᵢ → Y* edges | 11 | 4 | **2** |

**Nine of eleven supported edges do not survive re-clustering.** Across three partitions, **zero**
supported edges appear in all three; the two that survive two-of-three are our most externally
corroborated ones.

Two controls make this attributable to the partition and nothing else:
- **Determinism** — same data + same settings reproduces the edge set exactly (so it is not
  algorithmic randomness in FCI),
- **Identical covariates** — every partition selects the *same* covariate set (so it is not a
  variable-selection artifact).

And the key structural observation: **aggregate/distributional quantities are stable across partitions**
while **edge identities are not**. Counts (edges found, fraction supported) and the 53%-bidirected
proportion reproduce; the *identities* of individual edges move almost completely. A within-partition
bootstrap — which is the stability check we currently run — **cannot see this**, because it resamples
rows inside one fixed partition.

## 6. Why we think this is your paper's setting

Framing each partition as one of Tillman's **"independent non-identically distributed samples"** fits
unusually well:

- **Same variables** across partitions (verified: identical covariate set), **plausibly same causal
  structure**, **different induced distribution** — because each aggregation reshapes the joint. That
  is his precondition, almost verbatim.
- Our current practice — pick one partition — is exactly his rejected **"analyse a single sample":**
  low-power, and here also unstable.
- His rejected **"pool the data"** isn't even available to us (unpaired units), which removes the
  Yule–Simpson failure mode he warns about and leaves his prescription as the natural remaining option:
  **run the CI tests within each partition and combine them** (he recommends Fisher's method) rather
  than committing to any single partition.
- Most tellingly, **our empirical fingerprint — stable distribution, unstable edges — is the signature
  of "shared structure, non-identical samples,"** which is precisely the regime his method is built
  for. That is what makes us think the framing is not a stretch.

## 7. What we'd value your thoughts on

1. **Is the framing right?** Should we treat "which metacell partition" as a nuisance to *marginalise
   over* via your combined-CI-test construction, rather than a hyperparameter to *select*? Our current
   plan is heuristic — count the fraction of partitions an edge appears in — and your method looks like
   the principled replacement (one graph, sound type-I-error interpretation). One wrinkle: part of our
   skeleton search is **score-based** (GFCI's FGES phase), so the CI-test combination would cover our
   constraint-based runs and the orientation step but not the score-based skeleton directly — does that
   part need its own score-level aggregation?
2. **Dependence between samples — and why the clean fix may not be available to us.** Combining tests
   across partitions is only valid if the per-partition p-values are independent, and ours are not:
   every partition re-clusters the *same* cells. The clean fix would be to build each partition from a
   **disjoint subset of cells**, so the samples genuinely are independent — but that looks infeasible at
   our scale. `n` is already marginal (~133 on all the data), the outcome modality is only ~14% of the
   cells, and splitting into `K` folds leaves each fold with **at most ~`n/K` usable rows** (fewer in
   practice — a usable metacell needs a cell from every channel, which gets harder as clusters shrink).
   Even a handful of folds drops each fold to ~40–60 rows, below what FCI's conditional-independence
   tests support at ~tens of covariates. That seems to force us back onto **overlapping, dependent**
   partitions. How would you handle the dependence there — or is the independent-sample route simply not
   viable at this `n`, and we need a different construction?
3. **The shared-structure assumption.** It is most defensible across partitions at the *same*
   resolution/`n`; across different aggregation levels the effective variables may shift. Would you
   restrict the combination to same-resolution partitions, or is there a principled way to combine
   across resolutions too?
4. **Is there a better construction entirely** for the unpaired setting that avoids the arbitrary hard
   partition — soft/overlapping assignment, optimal-transport matching instead of clustering, or a
   measurement-error/latent-variable model that works on the cells directly without aggregating? Or is
   "construct many partitions and combine soundly" simply the right move?

---

*Full internal write-up with all numbers and controls: pipelines/empirical/evaluation/results/partition_stability_findings.md.*
