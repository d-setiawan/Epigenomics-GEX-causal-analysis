# 04 — The simulation

Every prediction in [01](01_aggregation_algebra.md) and [02](02_identifiability.md) is falsifiable.
This is where each one is given a controlled test, on data where the true graph is known.

Harness: the advisor's `paircheck` generator plus a multi-channel extension
(`Evaluation/scripts/paircheck_multichannel.py`, `crossmark_experiment.py`). The advisor's four
modules are deliberately **not** committed — see the project memory note on `dylan.zip`.

---

## 1. The generator

Draw $N$ cells with latent state $C_i \in \mathbb{R}^{d}$ under one of two geometries — `discrete`
($K$ well-separated centres, mimicking cell types) or `continuum` ($C \sim \mathcal{N}(0, I_d)$,
mimicking a differentiation axis). Draw a random DAG on $p$ variables with expected degree
$\bar{k}$, coefficients $A_{ij} \sim \pm\,\mathcal{U}(0.3, 0.9)$, and a latent loading mask with
$\Pr[\Gamma_{mj} \neq 0] = 0.3$. Generate

$$V = (I - A^{\top})^{-1}\left(\Gamma^{\top} C + e\right), \qquad e \sim \mathcal{N}(0, I).$$

An **anchor block** $S = W_s^{\top} C + \eta$ plays the role of the scGLUE embedding: it is the only
thing the clustering is allowed to see, and $\eta$ controls how well it recovers $C$.

Covariates are then assigned to $M$ channels, cells are split **disjointly** across channels, and
metacells are formed by $k$-means on $S$. Production parameters for the runs reported here:

$$p = 16,\quad d = 3,\quad M = 5\ (+1\ \text{outcome channel}),\quad N = 800/\text{channel},\quad n = 133,\quad \bar k = 3,$$

with FCI at `depth = 2` to match production, over $2\ \text{geometries} \times 3\ \text{data seeds}
\times 2\ \text{cluster seeds} = 12$ replicates per configuration (60 runs total).

## 2. The truth graphs

Scoring uses three graphs, following [02](02_identifiability.md):

$$G_{\mathrm{cell}} \;\supset\; G_{\mathrm{matched}} \;\supset\; G_{\mathrm{null}}$$

- $G_{\mathrm{cell}}$: the cell-level DAG, $C$ latent. The ceiling a jointly-observed assay would give.
- $G_{\mathrm{matched}}$: same, with **all cross-channel direct edges deleted** — the formal content
  of [02 §2](02_identifiability.md). The graph matched data can actually attain.
- $G_{\mathrm{null}}$: additionally isolates the outcome.

Scoring is split by channel type, and separately for **adjacency** and **orientation**, because
[02](02_identifiability.md) predicts these come apart. An edge counts as *oriented* only if the
estimate carries an arrowhead at the child and no arrowhead at the parent — i.e. it is compatible
with $\mathrm{pa} \to \mathrm{ch}$ and is not $\leftrightarrow$.

## 3. Test 1 — is the same-mark advantage the shared cells?

The prediction from [01 §7](01_aggregation_algebra.md#7-what-this-predicts-before-any-simulation): the
same-mark advantage comes from the $\mathbf{1}[\mathrm{ch}(j)=\mathrm{ch}(k)]$ term, so removing cell
*sharing* should abolish it.

The confound to defeat is that giving each variable its own cells also gives it *fewer* cells. So
three arms, two of which are matched on cells-per-variable:

| arm | construction | cells/variable |
|---|---|---|
| `shared` | all variables in a channel use all that channel's cells in the metacell | $n_{c,a}$ |
| `thinned` | all variables use the **same** random subset of size $n_{c,a}/m_a$ | $n_{c,a}/m_a$ |
| `disjoint` | each variable gets its **own** disjoint subset of size $n_{c,a}/m_a$ | $n_{c,a}/m_a$ |

`thinned` vs `disjoint` is the controlled contrast: identical cells per variable, differing only in
whether same-mark variables share them.

### Result

| arm | cells/var | rows | **same-mark adjacency** | cross-mark adjacency | precision |
|---|---|---|---|---|---|
| `shared` | 6.28 | 123.4 | 0.532 | 0.119 | 0.896 |
| `thinned` | 2.12 | 123.4 | **0.803** | 0.102 | 0.870 |
| `disjoint` | 2.37 | 84.3 | **0.054** | 0.075 | 0.798 |

**Two confirmations, and the second is the sharper one.**

**(a) Removing sharing collapses same-mark to cross-mark level.** At matched cells-per-variable,
$0.803 \to 0.054$, a 15-fold drop, landing at the cross-mark value ($0.075$). Once same-mark
variables stop sharing cells they behave exactly like cross-mark variables — which is what the
covariance formula says they should, since the extra term vanishes.

**(b) Thinning cells *raises* same-mark recall**, $0.532 \to 0.803$, with rows held fixed at 123.
This is the signature prediction: the noise term is $\mathbb{E}_c[\Sigma^W_{jk}/n_c]$, so shrinking
$n_c$ **grows** it, inflating same-mark covariances and manufacturing same-mark edges. **No
biological signal can be strengthened by collecting fewer cells.** Observing the inversion is the
cleanest available proof that the same-mark advantage is an artefact of the design.

*Caveat, stated because it is real:* `disjoint` retains fewer rows (84 vs 123), since splitting cells
per variable pushes more metacells below the completeness rule. The row loss is genuine but far too
small to produce a 15-fold drop, and `thinned` holds rows fixed at 123 while moving in the *opposite*
direction.

## 4. Test 2 — does repairing the covariance help?

Three estimators, all fed to the same stock FCI via a Tetrad `CovarianceMatrix`
([03 §1](03_algorithms.md#1-the-shared-substrate)):

$$\hat\Sigma^{\mathrm{raw}} = \widehat{\operatorname{Cov}}(X), \qquad
\hat\Sigma^{\mathrm{between}} = \hat\Sigma^{\mathrm{raw}} - \hat N, \qquad
\hat\Sigma^{\mathrm{zero}} = \hat\Sigma^{\mathrm{between}} + \hat\Sigma^{W}\!\big|_{\text{cross-block} := 0},$$

with $\hat N_{jk} = \mathbf{1}[\mathrm{ch}(j)=\mathrm{ch}(k)]\cdot \mathbb{E}_c[\hat\Sigma^{W}_{jk}/n_c]$.

| estimator | same-mark adj | **cross-mark adj** | cross-mark **oriented** | precision | edges | $n_{\leftrightarrow}$ | eff-rank |
|---|---|---|---|---|---|---|---|
| `raw` | 0.532 | 0.119 | 0.032 | 0.896 | 12.0 | 2.08 | 3.23 |
| `between-only` | 0.165 | **0.176** | **0.042** | 0.672 | 14.7 | 3.83 | **2.01** |
| `zero` | 0.944 | **0.063** | 0.008 | 0.964 | 8.5 | 1.75 | 6.87 |

**`between-only` behaves as predicted, in both directions.** Cross-mark adjacency rises 48%
($0.119 \to 0.176$) and cross-mark orientation 31% ($0.032 \to 0.042$), while same-mark falls
($0.532 \to 0.165$) — the correction removes the same-mark advantage the shared-cell term was
manufacturing and converts part of it into cross-mark recoverability. It also costs precision
($0.896 \to 0.672$) and drives the effective rank to $2.01 \approx d$, exactly the bound in
[02 §4](02_identifiability.md#4-the-rank-bound-on-sigmab-and-why-it-does-not-transfer-to-real-data).

**`zero` moves cross-mark the *wrong* way**, $0.119 \to 0.063$, which is prediction (3) of
[01 §7](01_aggregation_algebra.md#7-what-this-predicts-before-any-simulation) confirmed. Setting the
missing within-state cross-block to zero adds $\Sigma^{W}_{jj}$ to every **diagonal** while adding
nothing to cross-mark **off-diagonals**, so by the attenuation formula cross-mark correlations shrink.
It is the attenuation mechanism inflicted deliberately, and it behaves exactly as the algebra says.

> **This result did not replicate on real data.** §4's gain is measured against `raw` at fixed
> $\alpha$, and `between-only` also returns *more* edges (14.7 vs 12.0) at *lower* precision — the
> signature of an effectively looser threshold rather than a better estimate. The simulation
> comparison lacks an edge-count-matched control; the real-data test has one, and once edge count is
> matched the advantage vanishes entirely. See
> [05 §2](05_synthesis.md#2-repair-2-de-attenuated-covariance-equivalent-to-raising-alpha).

## 5. Test 3 — the pairing arms

Running the LV-Heuristic across the three pairing regimes of
[02 §3](02_identifiability.md#3-the-simulations-three-arms-and-what-each-isolates), scoring adjacency
recall of cell-level direct edges:

| | `true` | `matched` | `permuted` |
|---|---|---|---|
| same-channel | 0.800 | 0.833 | 0.867 |
| cross-channel | 0.789 | **0.325** | **0.005** |

Same-channel recall is **flat** — the pairing construction does not touch it. Cross-channel recall
falls by 59% under matched pairing and to essentially zero when correspondence is destroyed. This is
[02](02_identifiability.md) §1 and §2 side by side in one table, and it is the cleanest statement of
the identifiability asymmetry available.

## 6. Test 4 — the proportionality completion, refuted

[01 §4](01_aggregation_algebra.md#4-the-central-result) leaves the cross-mark block of $\Sigma^{W}$
structurally unobservable. One route to completing it assumes within- and between-state covariation
share a common mechanism up to scale:

$$\Sigma^{W}_{jk} \;\approx\; c \cdot \Sigma^{B}_{jk}.$$

The virtue of this assumption is that it is **falsifiable on the same-mark entries**, where both
sides are observed. Fit $c$ on same-mark pairs and check whether it transfers, leaving one mark out:

$$R^{2}_{\mathrm{LOCO}} \;=\; 1 - \frac{\sum_{a}\sum_{(j,k) \in a}\left(\Sigma^{W}_{jk} - \hat c_{-a}\,\Sigma^{B}_{jk}\right)^{2}}{\sum \left(\Sigma^{W}_{jk} - \overline{\Sigma^{W}}\right)^{2}}$$

| geometry | $R^{2}$ in-sample | $R^{2}_{\mathrm{LOCO}}$ |
|---|---|---|
| continuum | 0.371 | **0.013** |
| discrete | 0.341 | **0.211** |

A scale fitted on five marks does not predict the sixth. **The completion is not licensed**, and the
machinery is retained in `crossmark_covariance.py` only as a diagnostic. This is the intended use of
a falsifiable structural assumption: it was proposed, tested, and rejected before anything was built
on it.

## 7. Predictions and outcomes

| # | prediction (source) | outcome |
|---|---|---|
| 1 | removing cell sharing collapses same-mark recall to cross-mark level ([01 §7](01_aggregation_algebra.md#7-what-this-predicts-before-any-simulation)) | ✅ $0.803 \to 0.054$ at matched cells/variable |
| 2 | *fewer* cells per metacell **raises** same-mark recall ([01 §7](01_aggregation_algebra.md#7-what-this-predicts-before-any-simulation)) | ✅ $0.532 \to 0.803$ with rows fixed |
| 3 | zeroing the cross-block **worsens** cross-mark recovery ([01 §7](01_aggregation_algebra.md#7-what-this-predicts-before-any-simulation)) | ✅ $0.119 \to 0.063$ |
| 4 | same-channel recall is invariant to the pairing arm ([02 §1](02_identifiability.md#1-same-mark-pairs-keep-their-structure)) | ✅ 0.800 / 0.833 / 0.867 |
| 5 | cross-channel recall collapses under matched pairing ([02 §2](02_identifiability.md#2-cross-mark-pairs-lose-theirs)) | ✅ 0.789 → 0.325 → 0.005 |
| 6 | $\operatorname{rank}(\Sigma^{B}) \le d$ when clustering depends only on $C$ ([02 §4](02_identifiability.md#4-the-rank-bound-on-sigmab-and-why-it-does-not-transfer-to-real-data)) | ✅ eff-rank $2.01$ at $d = 3$ |

**Six predictions, six confirmations.** The covariance model in [01](01_aggregation_algebra.md) is
not a story told after the fact; it forecast each of these before the arms were run, and predictions
2 and 3 run in directions nobody would choose to fabricate — *fewer* cells helping, and a
"conservative" estimator hurting.

Separately, one **structural assumption was proposed and refuted**: proportionality completion, at
$R^{2}_{\mathrm{LOCO}} = 0.013$ ([§6](#6-test-4-the-proportionality-completion-refuted)). That is
deliberately *not* listed as a seventh confirmation. The model did not predict it; it was an extra
assumption, adopted because it was falsifiable, and then falsified. Counting a refutation as a
success would be reading the scorecard backwards.

---

Next: [05 — Synthesis](05_synthesis.md) — what happened when these repairs were taken to the real
data, and what survived.
