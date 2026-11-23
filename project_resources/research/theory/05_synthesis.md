# 05 — Synthesis: four repairs, and what survived

[04](04_simulation.md) confirmed the model. This is what happened when the repairs it suggested were
taken to the real data, why each failed, and what the final claim actually rests on.

---

## 0. The real data, measured

| quantity | value | meaning |
|---|---|---|
| median cells per mark per metacell | $\approx 11$ (min 1) | the $n_c$ in $\Sigma^{W}/n_c$ |
| median reliability $\lambda$ | $-0.017$ | the typical variable carries **no** between-state signal |
| $\lambda > 0.25$ | 11.8% of 442 variables | the usable minority |
| eff-rank of raw $\hat\Sigma^{X}$ (CD14) | 22.6 of 43 | **not** low-rank — the sim's bound does not transfer |
| max admissible correction $\lambda^{*}$ (CD14) | 0.177 | only ~18% of the noise term can be subtracted |

And the covariance-level consequence, on CD14 at $n = 133$ with matched median pair distance
(same-mark 93.6 kb vs cross-mark 90.4 kb, so distance is not confounding the split):

| $\alpha$ | same-mark significant | cross-mark significant | null |
|---|---|---|---|
| 0.05 | 6.8% | **4.0%** | 5.0% |
| 0.01 | 4.8% | **1.8%** | 1.0% |

**Cross-mark marginal association sits at or below the null rate.** This is
$\rho^{\mathrm{obs}} = \rho^{B}\sqrt{\lambda_j\lambda_k}$ with $\lambda \approx 0$, read directly off
the data. Everything that follows is an attempt to get around it.

## 1. Repair 1 — screen out the noise variables. **Failed.**

If 85% of variables have $\lambda \approx 0$, drop them. The screen uses only marginal variances, never
a pairwise association, so it cannot bias the CI tests toward finding edges — a correlation-based
screen would.

The control that makes it a test: a screened panel is smaller, and smaller graphs are more
partition-stable *for free*. So every screened arm is matched by **random** arms of identical size.

| arm | variables | edges/partition | stable ($\geq 6/11$) | cross-mark stable |
|---|---|---|---|---|
| full | 38.7 | 25.0 | **9.0** | **5.3** |
| random, matched size | 7.0 | 0.73 | 0.33 | 0.13 |
| **screened** | 6.5 | 0.23 | **0.0** | **0.0** |

Screening produced *fewer* edges than a random subset of the same size and **zero** stable edges.

**Why.** $\lambda_j$ is a **marginal** property. A variable can carry strong between-state signal and
share none of it with any other variable in the panel: high $\lambda_j$ says $\sigma^{B}_j{}^{2} > 0$,
it says nothing about $\Sigma^{B}_{jk}$. Selecting on the marginal selects away the *pairs*, which
are the object of interest. A screen on $\rho^{B}$ would target the right quantity — but it would
also be selection on the test statistic, and would invalidate the subsequent tests.

A second finding, at one level below the edges: $\lambda_j$ is **itself partition-dependent**. For
CD14, 3–9 variables clear $\lambda > 0.10$ on any single partition but only **2** clear it on the
cross-partition median. The instability documented for edges reaches down into which variables even
qualify as signal-carrying.

## 2. Repair 2. De-attenuated covariance equivalent to raising $\alpha$

Subtract the noise term at the largest admissible strength: $\hat\Sigma(\lambda) = \hat\Sigma^{X} -
\lambda\hat N$, with $\lambda^{*}$ the largest value keeping the condition number bounded.

### Why this could not have worked, in one derivation

De-attenuation inflates correlations. Suppose, as an idealisation, that it multiplies every
correlation by a common $\kappa > 1$. An edge survives the Fisher-$z$ screen at level $\alpha$ iff

$$\sqrt{n - |S| - 3}\;\tanh^{-1}\!\left(\kappa\hat\rho\right) \;>\; z_{1-\alpha/2}
\quad\Longleftrightarrow\quad
\sqrt{n - |S| - 3}\;\tanh^{-1}\!\left(\hat\rho\right) \;>\; \frac{\tanh^{-1}(\hat\rho)}{\tanh^{-1}(\kappa\hat\rho)}\, z_{1-\alpha/2}.$$

Since $\tanh^{-1}$ is increasing and $\kappa > 1$, the bracketed ratio is $< 1$. So the corrected test
at level $\alpha$ is the **raw** test at some level

$$\alpha' \;>\; \alpha, \qquad z_{1-\alpha'/2} \;=\; \frac{\tanh^{-1}(\hat\rho)}{\tanh^{-1}(\kappa\hat\rho)}\, z_{1-\alpha/2}.$$

**A uniform inflation of correlations is a threshold relaxation, not new information.** The real
correction is per-pair rather than uniform ($\kappa_{jk} = 1/\sqrt{\lambda_j\lambda_k}$), so this is
an idealisation — but it predicts that an **edge-count-matched raw arm** should reproduce the result.

### The control, and the outcome

Loosen $\alpha$ on the raw covariance until it emits the same edges per partition, then compare
partition-stable cross-mark edges:

| gene | $\lambda^{*}$ | raw ($\alpha = .05$) | de-attenuated | **raw, edge-matched** |
|---|---|---|---|---|
| CD14 | 0.211 | 1 | 6 | **5** |
| CD151 | 0.149 | 7 | 10 | **10** |
| CD52 | 0.144 | 8 | 8 | **8** |
| IL4R | 0.222 | 6 | 6 | **6** |
| PTPRJ | 0.463 | 2 | 5 | **5** |
| **total** | | 24 | **35** | **34** |

35 versus 34. The apparent gain over raw-at-$\alpha=0.05$ is **entirely** the looser effective
threshold, exactly as the derivation predicts. The same signature was visible in simulation and
missed for want of this control: `between-only` also returned more edges (14.7 vs 12.0) at lower
precision (0.672 vs 0.896).

## 3. Repair 3 — complete the missing block. **Refuted.**

Covered in [04 §6](04_simulation.md#6-test-4-the-proportionality-completion-refuted):
$R^{2}_{\mathrm{LOCO}} = 0.013$. A scale fitted on five marks does not predict the sixth.

## 4. Repair 4 — coarsen the metacells. **Monotonically worse.**

$\lambda$ depends on $n_c$, which we control: merging metacells raises cells-per-metacell and lowers
the row count. There is a genuine trade-off here, and it has an optimum — which is why this looked
like the most promising repair of the four.

### The trade-off, solved for a single test

Write $N_a$ for the cells in channel $a$ and $n$ for the number of metacells, so $n_c \approx N_a/n$.
Then

$$\lambda(n) \;=\; \frac{\sigma^{B2}}{\sigma^{B2} + \sigma^{W2}\,n/N_a}, \qquad
\rho^{\mathrm{obs}} \;=\; \rho^{B}\lambda(n).$$

For small $\rho$, the Fisher-$z$ statistic of a single marginal test scales as

$$T(n) \;\propto\; \sqrt{n}\,\rho^{B}\lambda(n) \;=\; \rho^{B}\,\frac{\sqrt{n}}{1 + \theta n},
\qquad \theta := \frac{\sigma^{W2}}{\sigma^{B2}N_a}.$$

Maximising: $\dfrac{d}{dn}\dfrac{\sqrt n}{1+\theta n} = 0 \Rightarrow 1 + \theta n = 2\theta n
\Rightarrow$

$$\boxed{\;n^{*} = \frac{1}{\theta} = \frac{\sigma^{B2}N_a}{\sigma^{W2}}, \qquad n_c^{*} = \frac{\sigma^{W2}}{\sigma^{B2}}, \qquad \lambda(n^{*}) = \tfrac12 .\;}$$

**The single-test optimum sits at reliability exactly $1/2$** — equal between- and within-state
contributions. Since the measured $\lambda \approx 0$ at $n_c \approx 11$ implies
$\sigma^{W2} \gg 11\,\sigma^{B2}$, we have $n^{*} \ll 133$: the marginal calculation says **coarsen**.

And the signal-share measurements agree with it. Pooled over CD14, CD151, IL4R:

| metacells | cells/mark | median $\lambda$ | $\lambda > 0.25$ |
|---|---|---|---|
| **133 (production)** | 10.0 | $-0.012$ | 10.4% |
| 60 | 13.8 | $+0.046$ | 24.7% |
| **35** | 22.0 | $\mathbf{+0.209}$ | **49.2%** |
| 25 | 28.0 | $+0.159$ | 43.9% |

### What actually happened

| metacells | CD151 | CD52 | IL4R | **total stable cross-mark** | $\lambda^{*}$ | **directed stable** |
|---|---|---|---|---|---|---|
| **133** | 7 | 8 | 6 | **21** | 0.14–0.22 | **0** |
| 60 | 2 | 1 | 2 | **5** | 0.01–0.11 | **0** |
| 45 | 2 | 0 | 1 | **3** | 0.00–0.05 | **0** |
| 25 | 0 | 0 | 0 | **0** | 0.000 | **0** |

**Monotonically worse in every gene separately.** Reliability per variable rises exactly as the
theory says, and edge recovery collapses anyway.

### Why the two curves disagree — the honest reading

$n^{*}$ optimises **one marginal test in isolation**. The search runs a cascade of
$O\!\left(p^{2}\binom{p}{\mathrm{depth}}\right)$ *dependent* conditional tests, and needs the whole
cascade to be stable, not one statistic to be maximal. Two things degrade faster than $\lambda$
improves:

1. **Conditioning consumes degrees of freedom.** At depth 2 the statistic carries $n - 5$ df. Going
   $133 \to 25$ takes that from 128 to 20; the variance of $\hat\rho_{jk\cdot S}$ rises as $1/(n-|S|-3)$,
   and *every* test in the cascade degrades together.
2. **The test count is fixed while $n$ falls.** The number of pairs and conditioning sets does not
   shrink with the row count, so the false-positive edges per partition stay roughly constant while
   the true ones get noisier. Recovered graphs become dominated by test noise — and test noise does
   not replicate across partitions any better than sampling noise does.

$\lambda^{*}$ reaching $0.000$ by 45 metacells is the same story from the conditioning side: with
that few rows the covariance is too ill-conditioned to subtract anything at all.

**The lesson is specific and worth keeping.** Optimising per-variable reliability is *not* the same
as optimising graph recovery, and on this data they point in opposite directions. Any future
aggregation-tuning proposal should be evaluated on edge stability directly, never on a signal-share
proxy.

## 5. The turn — stability as the estimator, not the problem

Four repairs failed because they all tried to extract more from a fixed covariance. The move that
worked inverts the framing: stop treating partition instability as a defect to correct, and use it as
the **selection rule** — which is precisely what
[partition_stability_findings.md](../../../pipelines/empirical/evaluation/results/partition_stability_findings.md) proposed
in July ("report the frequency distribution and pick a threshold, e.g. $\geq 50$–$80\%$").

$$\hat{E}_{\mathrm{stable}} \;=\; \left\{ (j,k) \;:\; \sum_{t=1}^{T}\mathbf{1}\!\left[(j,k) \in \hat G_t\right] \;\geq\; K \right\},
\qquad T = 11,\; K = 6 .$$

This is stability selection over the *aggregation*, not over rows — the variance component the
within-partition bootstrap structurally cannot see. On the 15-gene panel it retains **41 cross-mark
adjacencies**.

## 6. The two nulls

### Why not a binomial

[01 §6](01_aggregation_algebra.md#6-partition-instability-is-a-corollary-not-a-separate-problem) gives
$P(\geq K \text{ of } T) \approx 3.5\times 10^{-7}$ per pair under independence — which would make 41
edges overwhelming. **It is inadmissible**: the 11 partitions re-cluster the *same* cells and are
strongly dependent. An edge driven by any persistent structure, real or artefactual, recurs.

### Null A — permutation over channel correspondence

Reassign metacell labels at random **within each mark** after the first. Each mark keeps its own
metacell size distribution, every variable keeps its marginal, and the whole aggregation apparatus is
preserved; only the correspondence *between* marks is destroyed. Cross-mark edges surviving this
would have been manufactured by the apparatus.

| gene | edges/partition (real / null) | cross-mark stable: real | null |
|---|---|---|---|
| CD14 | 29.8 / 30.3 | 1 | **0** |
| CD151 | 41.2 / 41.1 | 7 | **0** |
| CD52 | 42.7 / 42.3 | 8 | **0** |
| IL4R | 24.8 / 21.4 | 6 | **0** |
| PTPRJ | 7.5 / 8.1 | 2 | **0** |

Two properties make this a usable null and both hold: edge counts per partition are matched (so it is
not a power comparison), and **same-mark** stable edges survive the permutation, as they must, since
within-mark correspondence is untouched.

### Null B — the biological signature

Adjacency being non-artefactual is still an internal statement. The biological test uses PCHi-C
**shared promoter hubs**: two peaks share a hub iff their bait-fragment sets intersect. PCHi-C never
enters the search, so it is independent of discovery.

Genomic distance is the dominant confound, and PAG degree is a second one that cuts the same way — a
well-covered peak is both more likely to yield a stable edge *and* more likely to overlap a PCHi-C
fragment. So the label is permuted only within joint distance $\times$ degree bins:

$$\hat\delta = \sum_{b} w_b\left(\bar s^{\,\mathrm{pos}}_b - \bar s^{\,\mathrm{neg}}_b\right),
\qquad
p = \frac{1}{B}\sum_{m=1}^{B}\mathbf{1}\!\left[\hat\delta^{(m)} \geq \hat\delta\right],$$

with $s_{jk} \in \{0,1\}$ the shared-hub indicator and $\hat\delta^{(m)}$ from labels permuted within
bins ($B = 20{,}000$).

| reference | stable hub rate | reference rate | distance-matched | **+ degree-matched** |
|---|---|---|---|---|
| non-adjacent ($n = 5566$) | **17.1%** | 6.3% | $+0.108$, $p = 0.0079$ | $\mathbf{+0.116},\ p = \mathbf{0.0022}$ |
| unstable-adjacent ($n = 1435$) | **17.1%** | 7.5% | $+0.097$, $p = 0.0204$ | $\mathbf{+0.109},\ p = \mathbf{0.0036}$ |

The **second row** carries the weight. It compares stable cross-mark edges against edges the *same
search on the same data* found but that did **not** survive re-partitioning — same distance bins, same
degree bins, differing only in whether the edge recurred. Stability selection is therefore selecting
**biology**, not merely strong correlations. And the enrichment *strengthens* under the stricter
matching, so degree was masking it rather than manufacturing it.

## 7. What the 41 edges look like

- **Distal, not same-locus.** Median separation **125 kb**; 39 of 41 exceed 10 kb. Of the 7 hub edges,
  6 exceed 10 kb (median 62 kb). This is not the "one element, several marks" co-occurrence flagged in
  [LIMITATIONS L13](../LIMITATIONS.md).
- **Biologically coherent.** All 7 hub edges pair two **active** marks — H3K27ac$\times$H3K4me2 (4),
  H3K27ac$\times$H3K4me3 (2), H3K4me2$\times$H3K4me3 (1) — against 32/41 active-active overall. No
  repressive mark appears among them, which is correct: an enhancer-contact database cannot confirm
  H3K27me3 or H3K9me3.
- **16 of 41 were missed by production.** Classified `non_adjacent` in the single-partition consensus
  (against 17 `bidirected`, 8 `other_adjacent`). Searching across partitions and keeping what recurs
  finds edges a single run does not — and those recovered edges are part of the set carrying the
  enrichment.

Edge list: [`Evaluation/results/data/crossmark_stable_edges.tsv`](../../../pipelines/empirical/evaluation/results/data/crossmark_stable_edges.tsv).

## 8. The claim, stated exactly

> Partition-stable cross-mark adjacencies identify **distal active-element pairs sharing a physical
> domain**, at cell-state level.

**Supported by:** two independent nulls — not manufactured by the pairing apparatus (Null A), and
enriched for shared promoter hubs beyond distance and degree (Null B).

**Not supported:**

- **Orientation.** [02 §2](02_identifiability.md#2-cross-mark-pairs-lose-theirs) forbids it, and the
  measurement confirms it: zero partition-stable directed cross-mark edges across all twelve
  gene $\times$ resolution cells.
- **A cell-level direct effect.** The surviving dependence runs through $\mu_c$, a cell-state-level
  quantity.
- **Causation.** "Shares a physical domain" is not "causes".

## 9. What would move this

Everything in [01](01_aggregation_algebra.md) says the binding constraint is $\lambda$, and $\lambda$
is set by $\sigma^{W2}/(\sigma^{B2}n_c)$. Only two terms are actionable, and neither is a search
change:

1. **Raise $n_c$ without lowering $n$** — impossible by re-partitioning ($n_c \approx N_a/n$), so it
   requires **more cells per assay**. This is the clean fix and it is an experimental one.
2. **Lower $\sigma^{W2}/\sigma^{B2}$** — aggregate *features* rather than cells. Quantifying at
   region level rather than per-peak pools counts across correlated bins, raising per-variable counts
   without touching the metacell partition. Untested, and the cheapest remaining option.

A third, narrower option sharpens the biology rather than the identifiability: the **motif** and
**ChIP** axes of the peak↔peak validator (currently stubs) would test whether these pairs share
*drivers* as well as domains — the mechanism the shared-hub result implies but does not demonstrate.

---

Back to [the index](README.md).
