# 01 — The aggregation algebra

What the search actually receives, and why that determines everything downstream.

---

## 1. The cell-level model

Cells $i = 1,\dots,N$ each carry a latent state $C_i \in \mathbb{R}^{d}$ (cell type, cell-cycle
phase, activation program — whatever the transcriptional manifold encodes). Over $p$ measured
variables $V_{i\cdot} \in \mathbb{R}^{p}$ we assume a linear structural equation model

$$V_{ij} \;=\; \sum_{k \neq j} A_{kj}\,V_{ik} \;+\; \sum_{m=1}^{d} \Gamma_{mj}\, C_{im} \;+\; e_{ij},
\qquad e_{ij} \perp C_i,$$

or in matrix form, with $A$ strictly triangular in some causal order,

$$V \;=\; A^{\top} V + \Gamma^{\top} C + e
\qquad\Longrightarrow\qquad
V \;=\; (I - A^{\top})^{-1}\!\left(\Gamma^{\top} C + e\right).$$

$A$ encodes the **cell-level DAG** we want. $C$ is unobserved, so $\Gamma$ generates confounding —
this is why the target is a PAG (via FCI) and not a DAG. See
[identifiability/](../identifiability) for what a PAG does and does not assert.

## 2. The assay structure — the part that makes this problem unusual

There are $M$ measurement channels: six histone marks plus RNA, $M = 7$. Let $\mathrm{ch}(j) \in
\{1,\dots,M\}$ be the channel of variable $j$, and $\mathcal{A}_a \subset \{1,\dots,N\}$ the cells
assayed in channel $a$. The critical structural fact:

$$\mathcal{A}_a \cap \mathcal{A}_b = \varnothing \quad \text{for } a \neq b.$$

**No cell is measured in two channels.** scCUT&Tag destroys the cell; a cell profiled for H3K27ac is
not available for H3K4me3 or for RNA. Consequently the joint cell-level distribution of
$(V_{\cdot j}, V_{\cdot k})$ for $\mathrm{ch}(j) \neq \mathrm{ch}(k)$ is **never observed** — not
noisily, not partially. It is structurally absent.

This is the fact the entire analysis turns on, and it is easy to lose sight of because the analysis
matrix looks like an ordinary $n \times p$ table.

## 3. Metacell aggregation

Cells are partitioned into $n$ metacells by Leiden clustering over the scGLUE latent embedding. For
metacell $c$ and channel $a$, write $M_{c,a} = \mathcal{A}_a \cap \mathrm{cluster}_c$ and $n_{c,a} =
|M_{c,a}|$. The analysis matrix is

$$X_{cj} \;=\; \frac{1}{n_{c,\mathrm{ch}(j)}} \sum_{i \in M_{c,\mathrm{ch}(j)}} V_{ij}.$$

A metacell contributes a row only if $n_{c,a} \geq 1$ for **every** channel (the completeness rule —
itself an uncorrected selection, see [LIMITATIONS](../LIMITATIONS.md)). In the production build this
leaves $n = 133$ complete metacells out of 148, with a median of about **11 cells per mark per
metacell** and a minimum of 1.

Decompose each entry into the population metacell mean and the sampling deviation of the average:

$$X_{cj} \;=\; \underbrace{\mu_{cj}}_{\mathbb{E}[V_j \mid c]} \;+\; \underbrace{\xi_{cj}}_{\text{sampling deviation}},
\qquad \mathbb{E}[\xi_{cj} \mid c] = 0 .$$

## 4. The central result

Let $\Sigma^{W}_{jk} = \operatorname{Cov}(V_j, V_k \mid c)$ be the within-metacell (state-conditional)
covariance and $\Sigma^{B}_{jk} = \operatorname{Cov}_c(\mu_{cj}, \mu_{ck})$ the between-metacell
covariance. Conditioning on the metacell, the sampling deviations satisfy

$$\operatorname{Cov}(\xi_{cj}, \xi_{ck} \mid c) \;=\;
\begin{cases}
\dfrac{\Sigma^{W}_{jk}}{n_{c,a}}, & \mathrm{ch}(j) = \mathrm{ch}(k) = a \quad \text{(the same cells)}\\[2.2ex]
0, & \mathrm{ch}(j) \neq \mathrm{ch}(k) \quad \text{(disjoint cells)} .
\end{cases}$$

The second line is the whole point: averages over disjoint cell sets have **independent** sampling
error, conditional on the metacell. Taking expectations over metacells,

$$\boxed{\;\operatorname{Cov}(X_j, X_k) \;=\; \Sigma^{B}_{jk} \;+\; \mathbf{1}\!\left[\mathrm{ch}(j) = \mathrm{ch}(k)\right] \cdot \mathbb{E}_c\!\left[\frac{\Sigma^{W}_{jk}}{n_{c,\mathrm{ch}(j)}}\right].\;}$$

**Same-mark and cross-mark entries of the covariance fed to FCI/BOSS are not on the same scale.**
Every same-mark entry carries an extra term that no cross-mark entry carries. This is a property of
the assay design; chromatin has nothing to do with it. Recorded as
[LIMITATIONS L14](../LIMITATIONS.md#l14-same-mark-and-cross-mark-covariances-are-not-on-the-same-scale).

## 5. Attenuation, and the reliability $\lambda_j$

The CI test sees correlations, not covariances. For a **cross-mark** pair the noise contributes
nothing to the numerator but inflates both variances in the denominator:

$$\rho^{\mathrm{obs}}_{jk}
= \frac{\Sigma^{B}_{jk}}{\sqrt{\left(\sigma^{B}_j{}^{2} + \bar\sigma^{W}_j{}^{2}\right)\left(\sigma^{B}_k{}^{2} + \bar\sigma^{W}_k{}^{2}\right)}}
= \rho^{B}_{jk}\,\sqrt{\lambda_j \lambda_k},$$

where $\bar\sigma^{W}_j{}^{2} := \mathbb{E}_c[\Sigma^{W}_{jj}/n_{c}]$ and

$$\boxed{\;\lambda_j \;:=\; \frac{\sigma^{B}_j{}^{2}}{\sigma^{B}_j{}^{2} + \bar\sigma^{W}_j{}^{2}}
\;=\; 1 - \frac{\mathbb{E}_c\!\left[\Sigma^{W}_{jj}/n_c\right]}{\operatorname{Var}(X_j)} \;\in\; [0,1].\;}$$

This is exactly **Spearman's attenuation by measurement error**, and $\lambda_j$ is the classical
*reliability* of variable $j$. It is what the code calls the **signal share**.

For a **same-mark** pair no such clean factorisation exists, because the numerator gains
$\mathbb{E}_c[\Sigma^{W}_{jk}/n_c]$ too. If within-metacell and between-metacell covariation have the
same sign — which for co-regulated elements they generally do — the same-mark correlation is
*inflated* rather than attenuated. The two mark classes are biased in **opposite directions**.

### 5a. Measuring $\lambda$ — and getting the estimator right

$\lambda_j$ is estimable, because $\Sigma^{W}$ is estimable from the single cells we still have. The
estimator matters:

$$\widehat{\bar\sigma^{W}_j{}^{2}}
\;=\; \underbrace{\frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \frac{S_{c,jj}}{n_{c}}}_{\textbf{per-cluster (unbiased)}}
\qquad\text{not}\qquad
\underbrace{\widehat{\Sigma^{W}_{jj}}\cdot \mathbb{E}_c\!\left[\tfrac{1}{n_c}\right]}_{\textbf{pooled (biased here)}} .$$

The pooled form is degrees-of-freedom weighted, so large metacells dominate $\widehat{\Sigma^{W}}$,
while $\mathbb{E}[1/n_c]$ is dominated by small ones. Small Leiden clusters are internally tighter,
so the product **overestimates**. Diagnostically it returns noise shares above $1$ — i.e.
$\lambda_j < 0$ by more than sampling error allows — which is an impossible variance decomposition
and is how the bias was caught. The per-cluster estimator returns a median noise share of exactly
$1.000$ (mean $0.983$), i.e. correctly calibrated against a true $\lambda = 0$.

### 5b. What $\lambda$ actually is on this data

Across 442 variables over 15 genes ([owning
doc](../../Evaluation/results/crossmark_identifiability.md)):

| statistic | value |
|---|---|
| median $\lambda$ | $-0.017$ |
| $\lambda > 0$ | 33.0% |
| $\lambda > 0.10$ | 18.6% |
| $\lambda > 0.25$ | 11.8% |
| $\lambda > 0.50$ | 7.5% |

**The median variable has $\lambda \approx 0$**: its entire between-metacell variance is the sampling
noise of averaging ~11 cells. Roughly 85% of the panel carries no between-state signal this design
can distinguish from noise. That $7.5\%$ exceed $\lambda = 0.5$ is far more than noise-around-zero
would give, so a real high-signal minority does exist — but it is a minority.

Since $\rho^{\mathrm{obs}} = \rho^{B}\sqrt{\lambda_j\lambda_k}$, a cross-mark pair of two median
variables has $\rho^{\mathrm{obs}} \approx 0$ **whatever $\rho^{B}$ is**. The information is not
merely noisy; for most pairs it is gone.

## 6. Partition instability is a corollary, not a separate problem

This is the connection back to the result that started the investigation
([partition_stability_findings.md](../../Evaluation/results/partition_stability_findings.md): 9 of 11
validated edges lost on a seed change, 0 surviving all three partitions).

When $\lambda_j \approx 0$,

$$X_{cj} = \mu_{cj} + \xi_{cj} \;\approx\; \xi_{cj},$$

so the analysis matrix is *approximately pure sampling noise*. Re-partitioning changes
$\{M_{c,a}\}$, which redraws every $\xi_{cj}$. The estimated correlation matrix is then close to a
fresh draw from a null distribution on each partition, and the recovered edges are whichever pairs
land in the upper tail of that draw. Tail membership does not persist across independent draws.

Quantitatively, if a null pair enters the graph with per-partition probability $q$ and partitions
were independent, the chance of appearing in at least $K$ of $T$ is

$$P\left(\text{recovered} \geq K \text{ of } T\right) = \sum_{t=K}^{T}\binom{T}{t} q^{t} (1-q)^{T-t}.$$

For CD151 ($q \approx 41/1326 \approx 0.031$, $T = 11$, $K = 6$) this is $\approx 3.5\times10^{-7}$
per pair, so $\approx 5\times10^{-4}$ stable edges expected across all 1326 pairs.

> **This calculation is illustrative only and must not be used as a test.** The 11 partitions are
> re-clusterings of the *same* cells, so they are strongly dependent and the independence assumption
> fails badly. That dependence is precisely why the real analysis uses a **permutation null**
> instead — see [05 — Synthesis](05_synthesis.md#6-the-two-nulls).

The practical consequence, and the reason coarsening does not help: partition stability is a
function of $\lambda$, and $\lambda$ is set upstream by how many cells land in each metacell per
mark. No property of the search touches it.

## 7. What this predicts, before any simulation

Three falsifiable consequences, all tested in [04 — The simulation](04_simulation.md):

1. **Removing cell *sharing* should collapse same-mark recall to the cross-mark level**, because it
   zeroes the $\mathbf{1}[\mathrm{ch}(j)=\mathrm{ch}(k)]$ term. Give each same-mark variable its own
   disjoint cells and the two classes should become statistically identical.
2. **Reducing cells per metacell should *raise* same-mark recall**, because
   $\mathbb{E}[\Sigma^W_{jk}/n_c]$ grows as $n_c$ shrinks. No biological signal can behave this way,
   so observing it is a clean signature of the artefact.
3. **Adding $\Sigma^{W}$ to the diagonal while leaving cross-mark off-diagonals at zero should
   *worsen* cross-mark recovery**, by the attenuation formula applied deliberately.

All three are confirmed. §7's second prediction in particular is the sharpest available evidence
that the same-mark advantage is not biology.

---

Next: [02 — Identifiability under matched pairing](02_identifiability.md), which asks what the
*target graph* becomes once variables are averaged over disjoint cells.
