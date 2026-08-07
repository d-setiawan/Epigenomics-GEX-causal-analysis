# Cross-mark structure: what blocks it, and what unblocks it

**Status: 2026-08-02.** Complete. Two same-day corrections are recorded inline:
§9 (superseded by §10) and **§14, which withdraws a claim made in §13**. §11
reports the full 15-gene panel and supersedes an earlier underpowered six-gene
pass recorded in the same section.

**Bottom line.** Cross-mark structure here is *not* identifiable. Nothing that
changes what the LV search consumes recovers it — screening, completion, and
de-attenuation all fail under proper controls (§6, §5c, §7) — and coarsening the
metacells, which the signal-share curve suggested, makes it strictly worse (§10).
The direct statement: **no cross-mark edge is partition-stable as a *directed*
edge at any resolution tested**, including the current one.

Cross-mark **adjacencies**, by contrast, hold up under two independent nulls.
41 partition-stable cross-mark edges over the 15-gene panel survive a
matched-power permutation null (§8), and they are enriched for **PCHi-C shared
promoter hubs** against a distance- *and* degree-matched null — 17.1% vs 6.3% for
non-edges (p = 0.002) and vs 7.5% for edges found but not partition-stable
(p = 0.004) (§11). That second contrast is the informative one: stability selects biology,
not merely strong correlations.

So the supported claim is an **adjacency** claim at cell-state level: these
element pairs sit in shared physical domains. Not orientation, and not a
cell-level direct effect.

The root cause sits upstream of any search: same-mark and cross-mark covariance
entries are not on the same scale by construction (§1), and the peak × cell
matrices are 0.5–1.5% dense, so a given (peak, metacell) has ~0.1 expected
detections (§13). The median panel variable's estimated between-state signal
share is ~0 under both a Gaussian and a count-native estimator, and no
non-Gaussian view rescues a variable the Gaussian one discards. **But see §14:**
the count *test* is far too underpowered at these depths to license "there is no
signal" — only "this design cannot resolve it".

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

## 8. The surviving cross-mark edges ARE real (positive result)

De-attenuation adds nothing (§7), but the raw arm does yield partition-stable
cross-mark edges. Whether *those* are signal is a separate question, and it needs
a null.

The **permutation arm** reassigns metacell labels at random within each mark after
the first. Each mark keeps its own metacell size distribution, every variable
keeps its marginal, and the entire aggregation apparatus is preserved — only the
correspondence *between* marks is destroyed. Cross-mark edges that stay
partition-stable under this would have been produced by the apparatus itself.

Two properties make this a usable null, and both hold empirically:
edges per partition are unchanged (so it is not a power comparison), and
**same-mark** stable edges survive (permutation preserves within-mark
correspondence, so they should).

Partition-stable cross-mark edges, real data vs null, α=0.05, FCI depth 2:

| gene | edges/partition (real / null) | **cross-mark stable: real** | **null** |
|---|---|---|---|
| CD14 | 29.8 / 30.3 | 1 | **0** |
| CD151 | 41.2 / 41.1 | 7 | **0** |
| CD52 | 42.7 / 42.3 | 8 | **0** |
| IL4R | 24.8 / 21.4 | 6 | **0** |
| PTPRJ | 7.5 / 8.1 | 2 | **0** |
| **total** | matched | **24** | **0** |

LGALS2 completed later and adds 7, for **31** over six genes; §11 uses all six.

**24 partition-stable cross-mark edges across five genes, against zero under a
matched-power null.** Edge counts per partition agree closely in every gene, so
this is not a power difference. CD151, CD52 and IL4R carry the result; CD14 (1)
and PTPRJ (2) are individually uninformative.

So cross-mark structure here is **detectable and reproducible even though it is
not identifiable** by any of the covariance repairs in §5–§7. Those are different
questions and they have different answers.

What these edges support is an **adjacency** claim — "these two marks covary
across cell states beyond what the pairing apparatus manufactures" — at the level
of metacell (cell-state) variation. They do **not** support an orientation, and
they do not license a cell-level direct effect: matched pairing severs
cross-channel direct edges, and the same-mark/cross-mark scale asymmetry
([LIMITATIONS L14](../../docs/LIMITATIONS.md#l14-same-mark-and-cross-mark-covariances-are-not-on-the-same-scale))
is still present in the covariance these were found in.

LGALS2 was still running when this was written and is not included.

## 9. Metacell resolution: signal share improves, edge recovery does not

> **Correction 2026-08-02 (same day).** This section originally presented ~35
> metacells as the operating point to move to, on the strength of the signal-share
> curve below, and left "whether 35 rows can support FCI" as the open question.
> **That question has since been answered, and the answer is no** — see §10. The
> signal-share curve is correct and reproducible; the recommendation drawn from it
> was wrong. Signal share per variable is not the objective. Read §9 as a
> diagnostic of *why* the current design is noisy, and §10 for what happens when
> you act on it.

If the binding constraint is signal per variable (§2) and the noise term is
`Sigma_within / n`, then coarser metacells buy signal directly and cost rows.
That is a tradeoff with a computable optimum, and the current setting is at the
wrong end of it.

`Evaluation/scripts/crossmark_resolution_tradeoff.py` agglomerates the *existing*
metacells by their mean profiles (Ward), which holds the embedding, the cells and
the completeness rule fixed so that only granularity varies. Pooled over CD14,
CD151, IL4R:

| metacells | rows | cells/mark | median signal share | **usable (>0.25)** | usable (>0.50) |
|---|---|---|---|---|---|
| **133 (current)** | 133 | 10.0 | **−0.012** | **10.4%** | 3.7% |
| 100 | 100 | 11.7 | −0.001 | 12.4% | 4.5% |
| 80 | 80 | 12.8 | +0.015 | 18.6% | 4.5% |
| 60 | 60 | 13.8 | +0.046 | 24.7% | 8.5% |
| 45 | 45 | 19.3 | +0.091 | 39.1% | 13.3% |
| **35** | 35 | 22.0 | **+0.209** | **49.2%** | 22.8% |
| 25 | 25 | 28.0 | +0.159 | 43.9% | 30.9% |
| 18 | 18 | 23.0 | +0.104 | 41.3% | 25.6% |

Between 133 and 35 metacells the median variable goes from carrying **no**
between-state signal to carrying +0.209, and the fraction of usable variables
rises **4.7×** (10.4% → 49.2%). The price is 133 rows → 35.

The curve flattens below ~35 and the median share stops improving, because merging
past that point starts destroying between-state variance as well as noise. So 25–45
is the region worth testing, not the extreme.

This is a lower bound: merging existing metacells is a coarse proxy for
re-clustering at a lower resolution, which has the same cells available and can
place boundaries better.

Whether these rows can still support FCI is the question §10 answers.

## 10. Coarsening destroys edge recovery, and nothing orients at any resolution

Acting on §9 — running the full stability harness with each partition's metacells
agglomerated to a target count, permutation null included, everything else
identical (`--merge-to`). Three genes, 11 partitions, edges stable in ≥6.

Partition-stable **cross-mark** edges (raw arm):

| metacells | CD151 | CD52 | IL4R | **total** | null | λ\* | **directed stable** |
|---|---|---|---|---|---|---|---|
| **133 (current)** | 7 | 8 | 6 | **21** | 0 | 0.14–0.22 | **0** |
| 60 | 2 | 1 | 2 | **5** | 0 | 0.01–0.11 | **0** |
| 45 | 2 | 0 | 1 | **3** | 0 | 0.00–0.05 | **0** |
| 25 | 0 | 0 | 0 | **0** | 0 | 0.000 | **0** |

(CD151 alone at 35 metacells: 1. The decline is monotonic in every gene
separately, not only in the total.)

**Coarsening monotonically destroys cross-mark edge recovery**, in all three genes
independently: 21 stable cross-mark edges at 133 metacells, 5 at 60, 3 at 45, none
at 25. 133 metacells is the best of the tested points, not the worst. Signal share per variable rises as
metacells coarsen (§9) but the row count falls faster, and partition stability —
the thing an edge claim actually rests on — collapses with it. By 25 metacells
nothing repeats across partitions at all.

The unmerged row is a re-run through the same code path as the merged ones and
reproduces the §7/§8 numbers exactly (raw 9 stable / 7 cross, de-attenuated 13/10,
edge-matched 13/10, permuted 1/0), so the comparison is like-for-like.

λ\* falls to 0.000 by 45 metacells: with that few rows the covariance is too
ill-conditioned to subtract any of the noise term, so de-attenuation is not merely
unhelpful there but inadmissible.

**And the identifiability answer, directly.** *No* cross-mark edge is
partition-stable **as a directed edge** in any gene at any resolution, including
the current one — 0 out of 0 across all twelve gene x resolution cells. Adjacencies repeat; orientations never do. That is the cleanest available
statement that the cross-mark claim this design supports is adjacency and not
direction, and it does not depend on any of the covariance arguments in §1–§7.

So §9's signal-share optimum is a real property of the data and the wrong
objective to optimise. The two curves point in opposite directions, and edge
stability is the one that matters.

## 11. Reproducible is not the same as biological (negative, underpowered)

§8 shows the stable cross-mark edges are not an artefact of the aggregation
apparatus. That is an *internal* statement. The biological question is separate,
and this asks it with the repo's existing element<->element machinery so the null
is the one already agreed on: **PCHi-C shared hub** (both peaks loop to a common
promoter bait, i.e. one physical domain), scored against a **distance-matched
permutation** null. PCHi-C never enters the search, so it is independent of
discovery.

The positive class is the *partition-stable cross-mark* set rather than the `<->`
mark class the validator normally scores, because orientation was never
partition-stable (§10) — adjacency is what survived.

**Full panel (15 genes, the pre-specified unit of analysis).** 41 stable
cross-mark pairs, 7042 candidate pairs:

| reference | stable hub rate | reference rate | distance-matched | **+ degree-matched** |
|---|---|---|---|---|
| non_adjacent (n=5566) | **17.1%** | 6.3% | +0.108, p=0.0079 | **+0.116, p=0.0022** |
| unstable_adjacent (n=1435) | **17.1%** | 7.5% | +0.097, p=0.0204 | **+0.109, p=0.0036** |

The degree-matched columns are the ones to quote. PAG degree is a live confound
here and cuts the same way as distance: a peak adjacent to many others is both
more likely to produce a partition-stable edge *and* more likely to overlap a
PCHi-C fragment, since both track how active and well-covered the region is. The
right-hand columns bin jointly on distance x degree, so the label is permuted only
among pairs matched on both. **The enrichment survives and in fact strengthens** —
degree was slightly masking it, not manufacturing it.

**Both are significant under both matchings, including the demanding one.** The second row compares
stable cross-mark edges against edges that were *found but did not survive
re-partitioning* — same search, same data, same distance bins, differing only in
whether the edge repeated across metacell boundaries. Stable edges carry 2.3x the
shared-hub rate of unstable ones after distance matching. **Partition stability is
selecting biology, not merely strong correlations.**

*Disclosure of sequence.* A first pass on six genes gave 31 positives at a 9.7%
hub rate, p = 0.102 and 0.205 — reported here because it was run first, not
because it is the analysis. Extending to all 15 genes is a move to the complete
panel, not a search over subsets, and it is the result to quote. Note that both
the count and the rate changed (31 -> 41 positives, 9.7% -> 17.1%), so this was
not purely a power gain: the nine added genes carry a higher hub rate among their
stable cross-mark edges. With per-gene yields of 0-8 that spread is expected, but
it means the six-gene pass should be read as underpowered rather than as an
independent replication.

**What this licenses.** Two independent nulls now support the cross-mark
adjacencies: the permutation null (§8) says they are not manufactured by the
pairing apparatus, and the distance-matched PCHi-C test says they are enriched for
shared promoter hubs relative both to non-edges and to unstable edges. Together
that is an evidenced claim that partition-stable cross-mark adjacencies pick out
element pairs sitting in shared physical domains.

**What it still does not license.** Orientation — no cross-mark edge is
partition-stable as a directed edge at any resolution (§10). Nor a cell-level
direct effect: matched pairing severs cross-channel direct edges, and the claim
lives at cell-state level. "Shares a physical domain" is not "causes".

## 12. The edge list

The p-values in §11 say the list is enriched; the list itself is the finding.
Written to
[`data/crossmark_stable_edges.tsv`](data/crossmark_stable_edges.tsv) — 41 edges
over 11 genes, annotated with distance, PAG degree, PCHi-C hub status, and how
the production single-partition run classified the same pair.

**These are distal pairs, not same-locus co-occurrence.** Median separation is
**125 kb** and 39/41 exceed 10 kb, so this is not the "one element, several
marks" effect flagged in
[LIMITATIONS L13](../../docs/LIMITATIONS.md#l13-peaks-are-histone-mark-peaks-not-atac-validation-is-enhanceratac-derived).
Of the 7 hub edges, 6 exceed 10 kb (median 62 kb); the single 102 bp pair is the
one genuinely same-locus case.

**All 7 hub edges pair two ACTIVE marks** — H3K27ac×H3K4me2 (4), H3K27ac×H3K4me3
(2), H3K4me2×H3K4me3 (1) — against 32/41 active-active among the stable set as a
whole. Active regulatory elements sharing a promoter hub is the biologically
expected pattern, and no repressive mark appears among the hub edges, which is
the right outcome given that an enhancer-contact database cannot meaningfully
confirm H3K27me3/H3K9me3.

**16 of the 41 were never found by the production pipeline** (classified
`non_adjacent` in its consensus), against 17 `bidirected` and 8 `other_adjacent`.
Running the search across metacell partitions and keeping what repeats therefore
recovers edges a single-partition run misses — and those recovered edges are part
of the set carrying the hub enrichment.

Hub edges concentrate in IL1B (3), IL4R (3) and RAPGEF1 (1).

## 13. Is the noise-domination finding an artefact of the Gaussian framing?

§2's signal share is a **variance** decomposition on log1p(CPM) values, so it
presupposes that variance is the right measure of information. For sparse counts
that is a real assumption: under a Poisson-like model Var ≈ mean, so counting
noise mechanically consumes the variance of exactly the low-count features a
count-native model might still learn from. A second-moment statistic cannot by
itself rule out signal living in the count distribution.

So the same question was asked four ways
(`Evaluation/scripts/crossmark_count_signal.py`), three of them not variance
decompositions — a count-native Poisson dispersion on **pooled** counts, a
Poisson deviance test, and a distribution-free Kruskal-Wallis:

| measure | result |
|---|---|
| median gaussian_share (current) | −0.017 |
| median poisson_share (count-native) | −0.141 |
| gaussian_share > 0.25 | 11.8% |
| poisson_share > 0.25 | 6.8% |
| **variables the count test rescues** | **0 of 442** |

The count-native view is if anything harsher, and **not one variable that the
Gaussian view calls dead is revived by it**. Rank correlation between the two
shares is only +0.073, so they are not simply the same statistic rewritten —
they disagree at the variable level while agreeing on the aggregate picture.

**The reason is sparsity, and it is severe.** The peak × cell matrices are
**0.5–1.5% dense**, verified directly from the matrix headers and independent of
any extraction code here. A peak is detected in roughly 1% of cells and a
metacell holds ~11 cells per mark, so the expected number of detections for a
given (peak, metacell) is ~0.1. Most of the metacell matrix is zero because
nothing was ever observed there, not because a state was measured as absent.

**Pooling peaks into coarser variables does not rescue it**
(`crossmark_variable_granularity.py`, adjacent peaks of the same mark merged):

| level | n vars | median counts | detection | median span | gaussian_share |
|---|---|---|---|---|---|
| peak (current) | 442 | 11 | 0.6% | 0.6 kb | −0.017 |
| merge2 | 236 | 28 | 1.5% | 27 kb | −0.017 |
| merge4 | 135 | 56 | 2.9% | 90 kb | −0.008 |
| merge8 | 83 | 79 | 4.4% | 224 kb | −0.008 |
| whole mark | 64 | 74 | 3.8% | 306 kb | −0.015 |

Counts and detection rise ~7×; signal share does not move. The extra reads
pooling brings in are not carrying between-state information.

## 14. CORRECTION — the count test is underpowered, so its zeros prove nothing

§13 originally also reported that **0.0%** of variables showed rate variation
beyond counting noise, at every granularity, and treated that as evidence of
absence. **That claim is withdrawn.** A test that cannot fire returns 0% whether
or not signal exists, and this one cannot fire.

`crossmark_power_check.py` simulates from each variable's **own** observed
per-metacell depths and totals, so the null is the actual design:

| total counts | rejection at α=.05 under H0 | calibration | MDE (80% power) |
|---|---|---|---|
| **11 (median peak)** | **0.000** | conservative | **never reached, even σ=3** |
| 79 (merge8) | 0.004 | conservative | σ = 2.0 |
| 500 | 0.181 | anti-conservative | σ = 0.5 |
| 5000 | 0.052 | calibrated | σ = 0.25 |

At the observed depth the Poisson deviance test rejects **zero** times under its
own null and never reaches 80% power even for a 20-fold between-metacell swing.
It would take roughly **450× more counts per variable** for the test to be both
calibrated and sensitive to a realistic state effect. The 500-count row is also
unusable in the other direction — 18% false positives.

What survives §13 after this correction:

- **Stands:** the count-native *effect-size* estimates (poisson_share) and the
  Gaussian share agree, and no non-Gaussian view rescues a variable the Gaussian
  view discards (0 of 442). The sparsity figures are direct measurements.
- **Withdrawn:** any use of "0.0% significant" as evidence that between-state
  variation is absent. The honest statement is that this design cannot resolve
  it, which is a statement about the assay, not about chromatin.

**This does not touch the empirical findings.** §8's permutation null, §10's
resolution curve, §11's PCHi-C enrichment and §12's edge list are direct
measurements on pipeline output, not inferences from the count model. They stand
unchanged.

It also means the marginal audits must **not** be used to pre-judge whether a
different transform or model class would help — marginal per-variable power is
far weaker than pairwise-association power across 133 metacells, which is exactly
how 41 externally-validated cross-mark adjacencies coexist with variables that
are marginally indistinguishable from noise. That question is asked at the edge
level instead (`crossmark_modelclass.py`).

---

## Reproduce

```bash
# is the noise finding an artefact of the Gaussian framing? (S13)
python Evaluation/scripts/crossmark_count_signal.py
python Evaluation/scripts/crossmark_variable_granularity.py

# can the count test even detect signal at these depths? (S14 -- run this
# before quoting any zero from the one above)
python Evaluation/scripts/crossmark_power_check.py --totals 11 79 500 5000

# metacell resolution tradeoff
python Evaluation/scripts/crossmark_resolution_tradeoff.py

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

# de-attenuation vs an edge-matched raw arm, plus the permutation null
python Evaluation/scripts/crossmark_deattenuated_stability.py --permute

# the same, at coarser metacell resolutions (S10)
python Evaluation/scripts/crossmark_deattenuated_stability.py --permute --merge-to 45

# do the stable cross-mark edges carry a PCHi-C hub signature? (S11, S12)
python Evaluation/scripts/crossmark_external_validation.py \
    --degree-bins 3 --n-bins 6 \
    --edges-out Evaluation/results/data/crossmark_stable_edges.tsv
```

`real_data_deconvolution.py` reconciles its reconstruction against the shipped
metacell matrix before reporting anything (median r = 0.974 for CD14) and aborts
the comparison if that fails.
