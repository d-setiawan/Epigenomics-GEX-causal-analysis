# 02 — Identifiability under matched pairing

[01](01_aggregation_algebra.md) showed what happens to the *covariance*. This asks the prior
question: once variables are averaged over disjoint cells, **what graph is the search even trying to
recover?** The answer is not the cell-level DAG, and the difference is not a matter of power.

---

## 1. Same-mark pairs keep their structure

Take two variables $u, w$ in the **same** channel, with a cell-level edge $u \to w$:

$$w_i \;=\; \beta\, u_i \;+\; \gamma^{\top} C_i \;+\; \epsilon_i .$$

Both are averaged over the *same* cell set $M_{c,a}$. Averaging is linear, so the structural equation
passes through unchanged:

$$\bar w_c \;=\; \beta\, \bar u_c \;+\; \gamma^{\top} \bar C_c \;+\; \bar\epsilon_c .$$

$\bar w_c$ is a direct function of $\bar u_c$ — the *realised* average, not its expectation. The
functional dependence is preserved exactly, and the correct mark for the aggregated pair remains
$\bar u \to \bar w$. Aggregation costs power here, nothing else.

## 2. Cross-mark pairs lose theirs

Now let $u$ be in channel $1$ and $v$ in channel $2$, again with a cell-level edge $u \to v$:

$$v_i = \beta\, u_i + \gamma^{\top} C_i + \epsilon_i .$$

But $\bar u_c$ averages over $M_{c,1}$ and $\bar v_c$ over $M_{c,2}$, and these are **disjoint**. The
$u$-values that actually generated the observed $v$'s — namely $\{u_i : i \in M_{c,2}\}$ — were never
measured. Writing $\mu^{u}_c = \mathbb{E}[u \mid c]$ and $\mu^{v}_c = \mathbb{E}[v \mid c]$,

$$\bar u_c = \mu^{u}_c + \xi^{u}_c, \qquad
\bar v_c = \mu^{v}_c + \xi^{v}_c, \qquad
\xi^{u}_c \perp\!\!\!\perp \xi^{v}_c \mid c,$$

and taking conditional expectations through the structural equation,

$$\mu^{v}_c \;=\; \beta\, \mu^{u}_c \;+\; \gamma^{\top} \kappa_c, \qquad \kappa_c := \mathbb{E}[C \mid c].$$

So the dependence between $\bar u_c$ and $\bar v_c$ runs **entirely through the latent metacell mean
$\mu^{u}_c$**, which is not observed — we see only $\bar u_c$, a noisy proxy for it.

$$\mu^{u}_c \;\longrightarrow\; \bar u_c, \qquad \mu^{u}_c \;\longrightarrow\; \mu^{v}_c \;\longrightarrow\; \bar v_c .$$

That is the definition of a **latent common cause**. In PAG terms the correct mark is

$$\bar u \;\leftrightarrow\; \bar v, \qquad \textbf{never} \qquad \bar u \to \bar v,$$

*even though the cell-level truth is a direct causal edge $u \to v$.* The adjacency survives; the
orientation is destroyed, and it is destroyed by the measurement design rather than by noise.

### 1+2 together: the identifiability asymmetry

$$\begin{array}{lll}
\textbf{same mark} & \text{shared cells} & \Rightarrow\ \text{direct edge survives as } \to \\
\textbf{cross mark} & \text{disjoint cells} & \Rightarrow\ \text{direct edge degrades to } \leftrightarrow
\end{array}$$

This is the formal content of the harness's `build_truth_dag_mc(drop_cross_channel=True)`: the
matched-pairing truth graph $G_{\mathrm{matched}}$ is the cell-level DAG with **all cross-channel
direct edges deleted**, latents retained.

### The prediction this makes, and its confirmation

If cross-mark orientation is unavailable *by construction*, then no amount of data, no choice of
search, and no aggregation resolution should ever produce a **partition-stable directed cross-mark
edge**. That is a strong, falsifiable claim.

Measured across 11 partitions and four aggregation levels (133 / 60 / 45 / 25 metacells) on three
genes — twelve gene $\times$ resolution cells in total — the number of partition-stable **directed**
cross-mark edges is:

$$0 \quad \text{in every cell.}$$

Adjacencies recur (41 of them, panel-wide); orientations never do. The theory and the measurement
agree exactly. ([Owning
doc](../../Evaluation/results/crossmark_identifiability.md).)

## 3. The simulation's three arms, and what each isolates

The harness realises three pairing regimes, which are exactly the three graphs above:

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

Read this as a decomposition. Same-channel recall is **flat across all three arms** — the pairing
construction does not touch it, exactly as §1 says. Cross-channel recall falls from 0.789 to 0.325
when pairing goes from joint to matched, and to 0.005 when correspondence is destroyed. The residual
$0.325$ under matched pairing is the part of the cell-level edge that still shows up as an
*adjacency* through the shared metacell mean, consistent with §2: the edge is visible, its direction
is not.

## 4. The rank bound on $\Sigma^{B}$ — and why it does not transfer to real data

The obvious repair for [01](01_aggregation_algebra.md)'s scale asymmetry is to subtract the noise
term and work with $\Sigma^{B}$ directly. There is a structural obstruction, and it is worth knowing
precisely when it bites.

From the model, the metacell means are

$$\mu_c \;=\; (I - A^{\top})^{-1}\left(\Gamma^{\top}\kappa_c + \mathbb{E}[e \mid c]\right).$$

If the clustering is a function of $C$ **alone**, then $e \perp c$, so $\mathbb{E}[e \mid c] = 0$ and

$$\mu_c = (I - A^{\top})^{-1}\Gamma^{\top}\kappa_c
\qquad\Longrightarrow\qquad
\operatorname{rank}\!\left(\Sigma^{B}\right) \le d .$$

$\Sigma^{B}$ is then a $d$-factor model, and **no $p$-variable DAG is identified from it** when
$p \gg d$. Subtracting the noise term removes precisely the floor that was keeping the aggregated
covariance invertible.

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
scGLUE latent *learned from the data itself*, so $\mathbb{E}[e \mid c] \neq 0$ and the rank bound
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

Cross-mark structure supports an **adjacency** claim at the level of cell-state variation:

> These two elements covary across cell states beyond what the pairing apparatus manufactures.

It does **not** support:

- an **orientation** — §2 forbids it, and §2's prediction is confirmed at every resolution;
- a **cell-level direct effect** — the surviving dependence runs through $\mu_c$, a cell-state-level
  quantity, not through the cell-level structural equation;
- a **causal** reading of the PCHi-C enrichment — "shares a physical domain" is not "causes".

---

Next: [03 — What each algorithm actually does](03_algorithms.md), including why the choice of search
turned out to matter less than the choice of *implementation*.
