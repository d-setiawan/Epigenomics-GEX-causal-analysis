# 02 — Identifiability under matched pairing

_Scope correction, 2026-11-01: the algebra below concerns the stated aggregation
design. It does not force universal PAG endpoints or rule out stable wrong
arrows. See [07](07_unpaired_integration_nonidentifiability.md) for the precise
disjoint-sampling nonidentifiability proof and its positive exceptions._

[01](01_aggregation_algebra.md) showed what happens to the *covariance*. This asks the prior
question: once variables are averaged over disjoint cells, **what graph is the search even trying to
recover?** The constructed distribution need not have the cell-level DAG, even
at infinite sample size.

---

## 1. Common-cell averaging preserves a linear structural equation

Take two variables $u, w$ in the **same** channel, with a cell-level edge $u \to w$:

$$w_i \;=\; \beta\, u_i \;+\; \gamma^{\top} C_i \;+\; \epsilon_i .$$

Both are averaged over the *same* cell set $M_{c,a}$. Averaging is linear, so the structural equation
passes through unchanged:

$$\bar w_c \;=\; \beta\, \bar u_c \;+\; \gamma^{\top} \bar C_c \;+\; \bar\epsilon_c .$$

$\bar w_c$ is a function of the realised $\bar u_c$, not merely its expectation.
This algebra preserves the coefficient when all terms use the same cells and
weights. For equal-size, non-overlapping random groups formed independently of
the values, with common fixed weights, a jointly observed Gaussian SEM's whole
covariance scales together and its CIs survive. It does not by
itself identify an orientation, guarantee that all parents were observed, or
justify applying the same guarantee to data-dependent grouping or a nonlinear
SEM. See [06](06_latent_space_aggregation.md) for these conditions.

## 2. Cross-mark pairs lose theirs

Now let $u$ be in channel $1$ and $v$ in channel $2$, again with a cell-level edge $u \to v$:

$$v_i = \beta\, u_i + \gamma^{\top} C_i + \epsilon_i .$$

But $\bar u_c$ averages over $M_{c,1}$ and $\bar v_c$ over $M_{c,2}$, and these are **disjoint**. The
$u$-values that actually generated the observed $v$'s — namely $\{u_i : i \in M_{c,2}\}$ — were never
measured. Writing $\mu^{u}_c = \mathbb{E}[u \mid c]$ and $\mu^{v}_c = \mathbb{E}[v \mid c]$,

$$\bar u_c = \mu^{u}_c + \xi^{u}_c, \qquad
\bar v_c = \mu^{v}_c + \xi^{v}_c, \qquad
\xi^{u}_c \perp\!\!\!\perp \xi^{v}_c \mid c,$$

Assume $\mathbb E[\epsilon\mid c]=0$ and take conditional expectations:

$$\mu^{v}_c \;=\; \beta\, \mu^{u}_c \;+\; \gamma^{\top} \kappa_c, \qquad \kappa_c := \mathbb{E}[C \mid c].$$

Under independent sampling within fixed state strata, dependence between
$\bar u_c$ and $\bar v_c$ runs through the stratum's mean vector. A component
of this shared-state dependence can be represented by

$$\mu^{u}_c \;\longrightarrow\; \bar u_c, \qquad \mu^{u}_c \;\longrightarrow\; \mu^{v}_c \;\longrightarrow\; \bar v_c .$$

This construction has a latent common cause for the two aggregate measurements.
Its particular latent projection can contain a bidirected edge. That does not
make two arrowheads identifiable in a PAG: other observationally equivalent
graphs may yield circles, and conditioning on other variables can change the
adjacency. Nor does every biological edge have to survive as an aggregate
adjacency. A graph over the constructed measurements is a different target
from the original cell-level graph.

### 1+2 together: the identifiability asymmetry

$$\begin{array}{lll}
\textbf{same mark} & \text{shared cells} & \Rightarrow\ \text{linear equation can survive} \\
\textbf{cross mark} & \text{disjoint cells} & \Rightarrow\ \text{same-cell residual coupling is unobserved}
\end{array}$$

The harness's `build_truth_dag_mc(drop_cross_channel=True)` stipulates a
matched-pairing reference graph by deleting cross-channel direct edges and
retaining its latents. That is a simulator convention, not a universal theorem
for every integration or aggregation operator.

### Historical empirical observation, not a theorem's prediction

Nonidentifiability does not imply that a procedure cannot produce a stable
directed edge. It can repeatedly choose the same wrong model. The following
absence of stable directions is an empirical result, not a logical consequence
of the information limit.

Measured across 11 partitions and four aggregation levels (133 / 60 / 45 / 25 metacells) on three
genes — twelve gene $\times$ resolution cells in total — the number of partition-stable **directed**
cross-mark edges is:

$$0 \quad \text{in every cell.}$$

Adjacencies recur (41 of them, panel-wide); directed outputs did not recur in
this audit. That observation neither proves nonidentifiability nor establishes
the surviving adjacencies as direct effects. ([Owning
doc](../../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md).)

## 3. The simulation's three arms, and what each isolates

The harness realises three pairing regimes with the following stipulated reference targets:

| arm | construction | target graph |
|---|---|---|
| `true` | every variable averaged over the **same** cells | $G_{\mathrm{cell}}$ — the jointly-observed ceiling the assays cannot supply |
| `matched` | variable $j$ from its own channel's cells | $G_{\mathrm{matched}}$ — cross-channel direct edges severed |
| `permuted` | as matched, with channel cluster labels permuted | $G_{\mathrm{null}}$ — cross-channel correspondence destroyed |

Running the LV-Heuristic across them and scoring **adjacency** recall of cell-level direct edges,
split by channel type:

| | `true` | `matched` | `permuted` |
|---|---|---|---|
| same-channel recall | 0.800 | 0.833 | **0.867** |
| cross-channel recall | 0.789 | 0.325 | **0.005** |

Same-channel recall is similar across the three arms. Cross-channel recall
falls from 0.789 to 0.325 when pairing goes from joint to matched, and to 0.005
when correspondence is destroyed. The matched result is compatible with
shared-state association; this comparison alone does not establish the source
of each recovered adjacency or an identifiable orientation.

## 4. The rank bound on $\Sigma^{B}$ and why it does not transfer to real data

The obvious repair for [01](01_aggregation_algebra.md)'s scale asymmetry is to subtract the noise
term and work with $\Sigma^{B}$ directly. There is a structural obstruction, and it is worth knowing
precisely when it bites.

From the model, the metacell means are

$$\mu_c \;=\; (I - A^{\top})^{-1}\left(\Gamma^{\top}\kappa_c + \mathbb{E}[e \mid c]\right).$$

If the clustering is a function of $C$ **alone**, then $e \perp c$, so $\mathbb{E}[e \mid c] = 0$ and

$$\mu_c = (I - A^{\top})^{-1}\Gamma^{\top}\kappa_c
\qquad\Longrightarrow\qquad
\operatorname{rank}\!\left(\Sigma^{B}\right) \le d .$$

$\Sigma^{B}$ is then a $d$-factor covariance. When $p>d$, it is singular,
so ordinary nonsingular Gaussian-CI arguments cannot be applied to it. The
rank bound alone is not a theorem excluding every restricted DAG model.
Subtracting the noise term removes the floor that kept the aggregate covariance
invertible.

Measuring the **effective rank** (participation ratio of the eigenvalues),

$$\text{eff-rank}(\Sigma) \;=\; \frac{\left(\sum_i w_i\right)^{2}}{\sum_i w_i^{2}},
\qquad w_i = \mathrm{eig}(\Sigma),$$

| | eff-rank | of |
|---|---|---|
| simulator ($d = 3$), $\Sigma^{B}$ | **2.01** | 16 |
| simulator ($d = 3$), raw $\Sigma^{X}$ | 3.23 | 16 |
| real CD14, raw $\Sigma^{X}$ | **22.59** | 43 |
| real CD14, de-attenuated | 20.54 | 43 |

The simulator clusters on an anchor that is a deterministic function of $C$, so it sits exactly at
the bound: $2.01 \approx d = 3$. **Real metacells do not.** They are Leiden clusters on a ~50-dimensional
scGLUE latent *learned from the data itself*, so $\mathbb{E}[e \mid c]$ need not be zero and the rank bound
does not apply — the measured effective rank is 22.6 of 43.

This distinction matters, and getting it wrong in either direction is easy:

- **The rank objection is a simulator artefact.** It does not license "cross-mark structure is
  low-rank in the real data".
- **But the repair still fails on real data**, for the different reason in
  [01 §5b](01_aggregation_algebra.md#5b-what-lambda-actually-is-on-this-data): $\lambda \approx 0$
  means there is little $\Sigma^{B}$ to recover. The maximum admissible correction (the largest
  $\lambda^{*}$ keeping the corrected matrix well-conditioned) is only $0.177$ for CD14, and falls
  to $0.000$ by 45 metacells.

Two different obstructions, one per regime. Conflating them produces the wrong diagnosis.

## 5. What claim survives

With adequate null calibration, cross-mark structure can support an association
claim at the level of cell-state variation:

> These two elements covary across cell states beyond what the pairing apparatus manufactures.

It does **not** support:

- an **original biological orientation** without a valid joint-completion and causal identification argument;
- a **cell-level direct effect** — the surviving dependence runs through $\mu_c$, a cell-state-level
  quantity, not through the cell-level structural equation;
- a **causal** reading of the PCHi-C enrichment — "shares a physical domain" is not "causes".

---

Next: [03 — What each algorithm actually does](03_algorithms.md), including why the choice of search
turned out to matter less than the choice of *implementation*.
