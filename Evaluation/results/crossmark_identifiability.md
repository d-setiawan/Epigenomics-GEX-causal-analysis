# Cross-mark structure: what blocks it, and what unblocks it

**Status: in progress (2026-08-02).** Sections 1–4 are complete and reproducible.
Sections 5–6 are being filled as runs land; numbers there are marked PROVISIONAL
until the full grid finishes.

Scope: whether cross-mark (peak↔peak across assays, and peak→gene) structure can
be claimed either with identifiability or with validation, and what has to change
in the LV search for that to be possible.

Caveats belong in [docs/LIMITATIONS.md](../../docs/LIMITATIONS.md), not here.

---

## 1. The mechanism: cross-mark correlations are attenuated by construction

A metacell holds cells from every assay, and variable *j* is the mean of that
metacell's **mark-ch(j) cells only**. Writing `X[c,j] = mu[c,j] + xi[c,j]` with
`mu` the metacell's population mean and `xi` the sampling deviation:

```
Cov(xi_j, xi_k)  =  Sigma_within[j,k] / n(c)   when ch(j) == ch(k)   (same cells)
                 =  0                          otherwise             (disjoint cells)
```

so the covariance the pipeline feeds the search is

```
Cov(X_j, X_k) = Sigma_between[j,k] + 1{same mark} * E_c[ Sigma_within[j,k] / n(c) ]
```

**Same-mark and cross-mark entries are not on the same scale.** Every same-mark
entry carries an extra positive term that no cross-mark entry carries. The
asymmetry is a property of the assay design, not of chromatin.

The effect on a cross-mark pair is one-sided and therefore worse than it looks.
The sampling noise adds nothing to the numerator of a cross-mark correlation, but
it still inflates **both variances in the denominator**:

```
rho_obs = rho_between * sqrt( (s2b_j/(s2b_j + s2w_j/n_j)) * (s2b_k/(s2b_k + s2w_k/n_k)) )
```

This is classical attenuation by measurement error, driven entirely by how many
cells happened to land in each metacell.

Removing it requires only `Sigma_within`, which is estimable for every variable
from its own mark's cells. **There is no missing block and no completion
assumption** — a point that took one wrong turn to establish (§3).

## 2. How much noise there actually is: the signal-share audit

Split each variable's metacell-level variance into real between-state signal and
the noise of averaging ~10 cells:

```
signal share  =  1  -  E_c[ Var(V_j | c) / n_c ]  /  Var_c( metacell means )
```

Estimator note, and it matters: the noise term must be estimated **per cluster**
as `E_c[S_c/n_c]`. The pooled form `Sw_pooled * E[1/n]` is degrees-of-freedom
weighted toward large metacells while `E[1/n]` is dominated by small ones, and
small Leiden clusters are internally tighter — so the product overestimates. It
returned noise shares above 1, an impossible decomposition, which is how the bias
was caught. The per-cluster estimator returns a median noise share of exactly
**1.000** (mean 0.983), i.e. correctly calibrated against a true share of zero.

Panel-wide (`Evaluation/scripts/crossmark_snr_panel.py`, 442 variables, 15 genes,
133 complete metacells, median ~11 cells per mark per metacell):

| quantity | value |
|---|---|
| median signal share | **−0.017** |
| share > 0 | 33.0% |
| share > 0.10 | 18.6% |
| share > 0.25 | 11.8% |
| share > 0.50 | 7.5% |

By mark, the fraction above 0.10 runs 12.5% (H3K27me3) to 23.1% (H3K4me3); no
mark is materially better than the others.

**Roughly 85% of panel variables carry no between-state signal that this design
can distinguish from sampling noise.** The 7.5% above 0.50 is far more than
noise-around-zero would produce, so a real high-signal minority does exist.

This is a precondition check, not a modelling choice: a variable with no
between-state signal has no structure for any search to recover, and it still
occupies every conditioning set, spending power that the signal-carrying
variables need.

## 3. What does *not* work, and why (recorded so it is not retried)

**Naive full deconvolution.** Subtracting the whole noise term to recover
`Sigma_between` fails in the simulator because `E[V|C]` is a deterministic
function of the latent state, so `Sigma_between` has rank ≤ d and the subtraction
removes exactly the floor that kept the aggregated covariance invertible.

**But this does not transfer to the real data**, and that distinction is the
reason cross-mark is not dead:

| | effective rank | condition |
|---|---|---|
| simulator (d=3), between | ~2.1 of 13 | — |
| real CD14, raw metacell cov | **22.6 of 43** | 97 |
| real CD14, de-attenuated | **20.5 of 43** | 1e4 (capped) |

The simulator clusters on an anchor that is a function of `C` alone, forcing
rank d. Real metacells are Leiden clusters on a ~50-dim scGLUE latent learned
from the data, so the between-metacell covariance is **not** low-rank. The
rank-collapse objection is an artefact of the simulator's `d=3`, not a property
of the assay.

Because the correction cannot be taken at full strength, the amount applied is
reported rather than assumed: **λ\*** is the largest fraction of the noise term
that can be subtracted while keeping the condition number under a ceiling. For
CD14, λ\* = **0.177** — only ~18% of the full de-attenuation is admissible.

**Proportionality completion (`Sigma_within ≈ c · Sigma_between`) is not needed.**
It was designed for a missing-block framing that turned out not to apply: the
cross-mark *off-diagonals* are already unbiased, and only the *diagonal* is
inflated. The falsification machinery (leave-one-mark-out R²) is retained in
`crossmark_covariance.py` for the same-mark blocks but is not on the critical
path.

## 4. What the real data looks like before any repair

CD14 panel, 43 variables, 133 metacells, production transform (`rank_gaussian`),
marginal Fisher-z. Median pair distance is matched (same-mark 93.6 kb vs
cross-mark 90.4 kb), so genomic distance does not confound the split.

| α | same-mark significant | cross-mark significant | null |
|---|---|---|---|
| 0.05 | 6.8% | 4.0% | 5.0% |
| 0.01 | 4.8% | 1.8% | 1.0% |
| 0.001 | 2.0% | 0.9% | 0.1% |

**Cross-mark association is at or below the null rate.** Same-mark is barely
above it. This is the covariance-level statement of the noise-domination finding,
and it explains the partition fragility recorded in
[partition_stability_findings.md](partition_stability_findings.md) without
appealing to anything about the search.

After de-attenuation at λ\* = 0.177, mean |r| rises 1.22× (same-mark) and 1.21×
(cross-mark), and cross-mark significance at α=0.05 moves 4.0% → 5.5%. The gain
is real but small, because λ\* is small.

## 5. Simulation: the same-mark advantage IS the shared cells

`Evaluation/scripts/crossmark_experiment.py`, multi-channel generator, 60 runs
(2 geometries × 3 data seeds × 2 cluster seeds), FCI at depth 2 matching
production, scored against the **cell-level** DAG and split by channel type.

### 5a. The mechanism arms

`thinned` and `disjoint` give each variable the same number of cells; they differ
only in whether variables in the same mark draw the *same* cells.

| arm | cells/var | rows | **adj_same** | adj_cross |
|---|---|---|---|---|
| shared | 6.28 | 123.4 | 0.532 | 0.119 |
| thinned | 2.12 | 123.4 | **0.803** | 0.102 |
| disjoint | 2.37 | 84.3 | **0.054** | 0.075 |

Two things follow.

**Removing cell sharing collapses same-mark recall to the cross-mark level**
(0.803 → 0.054, a 15× drop) while leaving cross-mark essentially unchanged. The
same-mark advantage is the shared-cell sampling term, not structure.

**Thinning cells *increases* same-mark recall** (0.532 → 0.803). Fewer cells per
metacell means a larger 1/n, hence a larger noise term added to same-mark
covariances, hence more same-mark association. No biological signal behaves that
way; this is the artefact observed directly.

Caveat on the contrast: `disjoint` retains fewer metacell rows (84 vs 123),
because splitting cells per variable pushes more clusters below the completeness
rule. The row loss is real but far too small to explain a 15× drop, and `thinned`
holds rows fixed at 123 while moving in the opposite direction.

### 5b. The estimator arms

| estimator | adj_same | **adj_cross** | **ori_cross** | adj_prec | n_bi | eff rank |
|---|---|---|---|---|---|---|
| raw | 0.532 | 0.119 | 0.032 | 0.896 | 2.1 | 3.2 |
| **between-only** | 0.165 | **0.176** | **0.042** | **0.672** | 3.8 | 2.0 |
| zero | 0.944 | 0.063 | 0.008 | 0.964 | 1.8 | 6.9 |

De-attenuation raises cross-mark adjacency by 48% (0.119 → 0.176) and cross-mark
orientation by 31% (0.032 → 0.042), and costs precision (0.896 → 0.672). The gain
is real but modest and not free.

The `zero` estimator moves cross-mark the *wrong* way (0.063), as the attenuation
algebra in §1 predicts: setting the missing within-state block to zero shrinks
cross-mark correlations relative to same-mark ones.

An earlier single run showed 0.000 → 0.211; that was one favourable draw and the
12-run average above supersedes it.

### 5c. Proportionality completion is refuted

Leave-one-mark-out R² for `Sigma_within ≈ c · Sigma_between`, fit on same-mark
entries where both are observed:

| geometry | r2_in_sample | **r2_loco** |
|---|---|---|
| continuum | 0.371 | **0.013** |
| discrete | 0.341 | **0.211** |

A scale fitted on five marks does not predict the sixth. The completion is not
licensed, confirming the decision in §3 to take it off the critical path.

## 6. Signal-share screening FAILS (negative result)

The obvious use of §2 is to drop the noise variables before searching. It does
not work, and the prediction that it would was wrong.

`Evaluation/scripts/crossmark_screening_stability.py`: screen on cross-partition
median signal share, run FCI on 11 independent metacell partitions, count edges
recovered in ≥6. Every screened arm is matched by **random** arms of identical
size, because a smaller panel is more stable for free — screening is only
supported if it beats random at matched size.

| arm | vars | edges/partition | stable (≥6/11) | cross-mark stable |
|---|---|---|---|---|
| full | 38.7 | 25.0 | **9.0** | **5.3** |
| random (matched size) | 7.0 | 0.73 | 0.33 | 0.13 |
| **screened** | 6.5 | 0.23 | **0.0** | **0.0** |

Screening produces *fewer* edges than a random subset of the same size and **zero**
partition-stable edges. Selecting variables on their marginal between-state
variance selects away the pairs: a variable can carry state signal and still share
none of it with any other variable in the panel.

Two further observations, both of which matter more than the screen itself:

**The full panel does yield partition-stable edges** — about 9 per gene, of which
~5 are cross-mark. Whatever is wrong with this pipeline, "no edge ever repeats
across partitions" is not it under this criterion.

**Signal share is itself partition-dependent.** Requiring share > 0.10 on the
cross-partition *median* is far stricter than on any single partition:

| gene | vars | share>0.10 per single partition | on the median |
|---|---|---|---|
| CD14 | 43 | 3–9 | **2** |
| CD151 | 52 | 6–10 | **4** |
| PTPRJ | 27 | 8–11 | **8** |
| IL4R | 36 | 6–10 | **7** |

The partition instability documented for edges reaches one level down, into which
variables even qualify as signal-carrying.

## 7. De-attenuation on real data is equivalent to loosening alpha (negative result)

§5b showed de-attenuation helps in simulation. It does not survive a proper
control on real data.

`Evaluation/scripts/crossmark_deattenuated_stability.py`: FCI on 11 independent
metacell partitions, edges recovered in ≥6, identical variables and settings
across arms. The control arm re-runs the **raw** covariance with alpha loosened
until it emits the same number of edges per partition — because de-attenuation
raises correlations, it returns more edges, and more edges give more
partition-stable edges for free.

Partition-stable **cross-mark** edges:

| gene | λ* | raw (α=.05) | de-attenuated | **raw, edge-matched** |
|---|---|---|---|---|
| CD14 | 0.211 | 1 | 6 | **5** |
| CD151 | 0.149 | 7 | 10 | **10** |
| CD52 | 0.144 | 8 | 8 | **8** |
| IL4R | 0.222 | 6 | 6 | **6** |
| PTPRJ | 0.463 | 2 | 5 | **5** |

**Once edge count is matched, de-attenuation adds nothing** — 35 vs 34 stable
cross-mark edges across the five genes, and `frac_stable` is within noise in every
gene. The apparent gain over raw-at-α=0.05 is entirely explained by the fact that
raising correlations lowers the effective threshold.

This does not contradict §5b so much as reinterpret it: the simulation arm also
returned more edges (14.7 vs 12.0) at *lower* precision (0.672 vs 0.896), which is
the same signature. The simulation comparison lacked an edge-matched control and
should be read as suggestive only.

**Conclusion for the algorithm question.** Changing what the LV search consumes —
de-attenuated covariance, between-only covariance, completed covariance, or a
screened variable set — does not produce cross-mark identifiability here. The
binding constraint is §2: the metacell matrix carries almost no between-state
signal to begin with, and no transformation of a covariance creates information
that the aggregation destroyed. The lever that would matter is upstream (cells per
metacell, quantification, or coarser aggregation), not the search.

## 8. IN PROGRESS — are the surviving cross-mark edges real?

The raw arm does yield partition-stable cross-mark edges (1–8 per gene). Whether
those are signal is a separate question from whether de-attenuation helps, and it
needs a null.

Running: a **permutation arm** that reassigns metacell labels at random within each
mark after the first. Each mark keeps its own metacell size distribution, every
variable keeps its marginal, and the whole aggregation apparatus is preserved —
only the correspondence *between* marks is destroyed. Cross-mark edges that remain
partition-stable under this are produced by the aggregation itself, not by shared
cell state.

---

## Reproduce

```bash
# signal-share audit across the panel
python Evaluation/scripts/crossmark_snr_panel.py --out snr_panel.tsv

# per-gene decomposition, lambda*, and the de-attenuated significance table
python Evaluation/scripts/real_data_deconvolution.py \
    --matrix CausalDiscovery/outputs/datasets/joint_v2/\
monocyte_cuttag_peak_genes_300kb_metacell_expanded/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv

# simulation: estimators x searches, scored against the cell-level DAG
PAIRCHECK_DIR=<dir containing paircheck/> \
python Evaluation/scripts/crossmark_experiment.py --depth 2

# screening vs random at matched size, across 11 partitions
python Evaluation/scripts/crossmark_screening_stability.py --threshold 0.10
```

`real_data_deconvolution.py` reconciles its reconstruction against the shipped
metacell matrix before reporting anything (median r = 0.974 for CD14) and aborts
the comparison if that fails.
