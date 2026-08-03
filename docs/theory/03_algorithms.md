# 03 — What each algorithm actually does

Six searches appear in this project. They differ along three axes — **what they assume**, **what
they optimise**, and **what object they output** — and those differences explain most of the
disagreement between runs. But the largest single disagreement we measured was not between
algorithms at all; it was between two implementations of the *same* algorithm. That is §6.

---

## 1. The shared substrate

Under a linear-Gaussian SEM every search below is a function of the covariance matrix alone. This is
worth stating explicitly because it is what made the whole covariance-repair line of work possible:
if the estimator only ever touches $\hat\Sigma$ and $n$, then handing it a *constructed* $\hat\Sigma$
substitutes cleanly for handing it data.

**The conditional-independence test — Fisher's $z$.** For the partial correlation $\rho_{jk\cdot S}$
of $X_j, X_k$ given a set $S$,

$$z_{jk\cdot S} \;=\; \frac{1}{2}\sqrt{\,n - |S| - 3\,}\;\ln\!\frac{1 + \hat\rho_{jk\cdot S}}{1 - \hat\rho_{jk\cdot S}}
\;\overset{H_0}{\sim}\; \mathcal{N}(0,1),$$

and $X_j \perp\!\!\!\perp X_k \mid S$ is declared when $|z| < z_{1-\alpha/2}$. The partial correlation
comes from the inverse of the sub-covariance, so the test reads $\hat\Sigma$ and nothing else.

**The rank-Gaussian (van der Waerden) transform**, which production applies before testing:

$$\tilde X_{cj} \;=\; \Phi^{-1}\!\left(\frac{r_{cj} - \tfrac12}{n}\right), \qquad r_{cj} = \operatorname{rank}\left(X_{cj}\right).$$

This makes Fisher's $z$ valid under a **nonparanormal** (Gaussian-copula) model rather than requiring
marginal Gaussianity — the right move for log-normalised count data, which is heavily skewed. It does
not fix nonlinearity in the dependence itself, only in the marginals.

**The score — SEM-BIC.** For a DAG $G$ with $k$ free parameters,

$$\mathrm{BIC}(G) \;=\; -2\,\ell\!\left(\hat\theta_G\right) \;+\; c\,k\log n,$$

with $c$ the *penalty discount* (production uses $c = 2$). For a Gaussian SEM the log-likelihood is
again a function of $\hat\Sigma$ only:

$$\ell \;=\; -\frac{n}{2}\sum_{j} \log \hat\sigma^{2}_{j \mid \mathrm{pa}(j)} + \text{const},$$

so both the test and the score are covariance functionals. Larger $c$ means a sparser graph.

**Why this matters here.** `IndTestFisherZ` and `SemBicScore` in Tetrad both accept an
`ICovarianceMatrix`, so a de-attenuated or completed $\hat\Sigma$ can be fed to *any* of the searches
below without modifying the search. Every "algorithm modification" tested in this project is an
intervention on $\hat\Sigma$, with the search left stock. See
[05 — Synthesis](05_synthesis.md).

## 2. The output objects

| object | produced by | asserts |
|---|---|---|
| **CPDAG** | PC, FGES, BOSS | Markov equivalence class of DAGs, **assuming causal sufficiency** |
| **PAG** | FCI, GFCI, BFCI | equivalence class of MAGs; **allows latent confounders and selection** |

PAG edge marks, per endpoint:

| mark | at endpoint $B$ of $A \mathbin{*\!\!-\!\!*} B$ | reading |
|---|---|---|
| arrowhead `>` | $B$ is **not** an ancestor of $A$ | |
| tail `—` | $A$ **is** an ancestor of $B$ | |
| circle `o` | undetermined within the class | |

giving the four edge types:

$$A \to B \;(\text{$A$ causes $B$}), \qquad
A \leftrightarrow B \;(\text{latent common cause}), \qquad
A \mathbin{\circ\!\!\to} B, \qquad
A \mathbin{\circ\!\!-\!\!\circ} B .$$

$A \leftrightarrow B$ is the mark this project once treated as its novel object; [02
§2](02_identifiability.md) explains why it is also the *correct* mark for a cross-mark pair under
matched pairing, and [§6](#6-the-implementation-gap) explains why its measured frequency is not a
property of the data.

## 3. Constraint-based: PC and FCI

**PC** — start from the complete graph; delete $j - k$ whenever some $S \subseteq \mathrm{adj}(j)
\setminus \{k\}$ with $|S| \le \text{depth}$ gives $j \perp k \mid S$; record the separating sets;
orient unshielded triples $j - m - k$ with $m \notin \mathbf{S}_{jk}$ as colliders $j \to m \leftarrow
k$; close under Meek's rules. Output: CPDAG. **Assumes causal sufficiency** — with latent confounders
present it is misspecified, which is why it is not the primary estimator here.

**FCI** — same adjacency phase, then two things PC does not do:

1. a **possible-d-sep** phase that removes further edges using conditioning sets drawn from
   `Possible-D-SEP`, sets PC never considers;
2. the complete FCI orientation rule set $\mathcal{R}_1 \ldots \mathcal{R}_{10}$, which propagates
   arrowheads and tails through discriminating paths.

Output: PAG. **Does not assume sufficiency** — this is the reason FCI is the project's primary
estimator ([identifiability/](../identifiability)).

The cost is combinatorial. The adjacency search is $O\!\left(p^{2}\binom{p}{\text{depth}}\right)$ in
tests, which is why `depth` is bounded (production: $\text{depth} = 2$). Leaving it unbounded was a
real incident in this project: an unbounded-depth FCI on ~46 variables ran **4 hours per gene**
against 6.7 seconds at depth 2 — a factor of ~2000 — and, being a different estimator, would not
have been comparable to production even had it finished.

## 4. Score-based: BOSS

**BOSS** (Best Order Score Search) optimises over **permutations** rather than over graphs. For an
order $\pi$, define the DAG $G_\pi$ by giving each variable the parent set obtained by a grow–shrink
sweep over its $\pi$-predecessors; then

$$\hat\pi = \arg\max_{\pi} \; \mathrm{score}\!\left(G_{\pi}\right),$$

searched by repeatedly relocating single variables to their best position. The permutation space is
smoother than the graph space, which is why BOSS tends to beat greedy graph search (FGES) at the same
score.

BOSS returns a DAG (or its CPDAG). It **assumes causal sufficiency**.

## 5. Hybrids: GFCI, BFCI, and the LV-Heuristic

| search | skeleton from | orientation from | sufficiency | can emit $\leftrightarrow$? |
|---|---|---|---|---|
| **GFCI** | FGES (score) | FCI rules + CI tests | not assumed | yes |
| **BFCI** | BOSS (score) | FCI rules + CI tests | not assumed | yes |
| **LV-Heuristic** | BOSS (score) | `dagToPag` on the BOSS DAG | **assumed** | **no — never** |

The score-based skeleton is the point of the hybrids: at $n = 133$ with $p \approx 43$, a BIC penalty
prunes edges that a CI test at $\alpha = 0.05$ would keep, so GFCI and BFCI return far sparser graphs
than FCI on identical data ([§6](#6-the-implementation-gap): 65–71 edges against 845).

### The LV-Heuristic is different in kind

LVH runs BOSS to a DAG and then applies the DAG-to-PAG transform. Since a DAG contains no latent
variables, its PAG image contains **no bidirected edges by construction**:

$$G_{\text{BOSS}} \text{ is a DAG} \;\Longrightarrow\; \mathrm{dagToPag}\!\left(G_{\text{BOSS}}\right) \text{ has } n_{\leftrightarrow} = 0 .$$

This is not a tuning artefact; it is definitional. And it creates a sharp tension with
[02](02_identifiability.md): under matched pairing the **correct** mark for a cross-mark pair is
$\leftrightarrow$, so LVH is *wrong by construction* on exactly the pairs of interest.

Yet scored on **adjacency** against $G_{\mathrm{matched}}$ it is the strongest of the three:

| search | same-channel recall | cross-channel recall | $n_{\leftrightarrow}$ | adjacency precision |
|---|---|---|---|---|
| FCI | 0.444 | 0.162 | 3.4 | 0.926 |
| BFCI-CP | 0.456 | 0.129 | 0.1 | 0.923 |
| **LV-Heuristic** | **0.833** | **0.325** | **0.0** | **0.925** |

$1.88\times$ and $2.01\times$ the recall of FCI at **indistinguishable precision**. The resolution of
the apparent paradox: adjacency and orientation are separate questions. LVH finds the *edges* better
because a score-based order search is better powered at $n = 133$ than a CI-test cascade; it then
labels them with marks it cannot justify. Using LVH for adjacency and declining to read its
orientations is coherent — and is what the final analysis does.

> A note on scoring against the right truth. Reported precision for LVH depends on which truth graph
> is used: against $G_{\mathrm{cell}}$ it carries a visible penalty, but against
> $G_{\mathrm{matched}}$ — the graph matched data can actually attain ([02
> §2](02_identifiability.md)) — the penalty disappears (0.925 vs FCI's 0.926). Scoring aggregated
> data against the cell-level graph charges the algorithm for damage the aggregation did.

## 6. The implementation gap

The largest disagreement measured in this project is not between algorithms. On **identical inputs** —
same 46 columns, 133 rows, `rank_gaussian` transform, depth 2, same confounder roots and background
knowledge — only the implementation and $\alpha$ varying:

| configuration | edges | $n_{\leftrightarrow}$ | $\leftrightarrow$ fraction | oriented | $\mathbin{\circ\!\!-\!\!\circ}$ |
|---|---|---|---|---|---|
| Tetrad FCI, $\alpha = 0.01$ | 705 | 6 | **0.9%** | 599 | 100 |
| Tetrad FCI-CP, $\alpha = 0.01$ | 705 | 7 | **1.0%** | 603 | 95 |
| Tetrad FCI, $\alpha = 0.05$ | 845 | 78 | **9.2%** | 681 | 86 |
| Tetrad GFCI | 65 | 0 | 0.0% | 16 | 49 |
| Tetrad BFCI / BFCI-CP | 71 | 0 | 0.0% | 26 | 45 |
| **causal-learn FCI, $\alpha = 0.05$** (production) | 258 | 126 | **48.8%** | 19 | 113 |

At **matched $\alpha$**, two implementations of FCI differ **5.3-fold** in bidirected fraction —
9.2% against 48.8%. Tetrad finds more than three times as many edges (845 vs 258) and orients
**36 times** as many (681 vs 19), while producing far fewer bidirected marks.

Two candidate explanations, one of which is ruled out:

- **Legal-PAG enforcement — ruled out.** Forcing `guaranteePag` moves Tetrad FCI from 6 to 7
  bidirected edges out of 705. It is not the mechanism.
- **The orientation cascade.** The striking column is `oriented`: 681 vs 19. An implementation that
  propagates few arrowheads leaves many pairs at $\mathbin{\circ\!\!-\!\!\circ}$ or resolves them to
  $\leftrightarrow$; one that propagates aggressively converts them to $\to$. The bidirected
  *fraction* is therefore largely a readout of how much orientation propagation happens, not of how
  much latent confounding is present.

**Consequence for the project's headline.** "53% of element↔element structure is explicitly flagged
as confounded" does not survive an implementation change. It is a property of causal-learn's FCI on
this data, not a property of monocyte chromatin. It is recorded as superseded in
[MISSION §2](../MISSION.md) and the numbers are owned by
[pairing_artifact_findings.md](../../Evaluation/results/pairing_artifact_findings.md).

## 7. Summary table

| | assumes sufficiency | optimises | output | emits $\leftrightarrow$ | role here |
|---|---|---|---|---|---|
| PC | yes | CI tests | CPDAG | no | not used — misspecified |
| **FCI** | **no** | CI tests | PAG | yes | primary estimator |
| GFCI | no | BIC skeleton + CI | PAG | yes | sparse cross-check |
| BOSS | yes | BIC over orders | CPDAG/DAG | no | inside BFCI / LVH |
| BFCI | no | BIC + CI | PAG | yes | sparse cross-check |
| **LV-Heuristic** | **yes** | BIC over orders | PAG (no $\leftrightarrow$) | **no** | best adjacency recall; orientations not interpretable |

---

Next: [04 — The simulation](04_simulation.md), where each prediction from
[01](01_aggregation_algebra.md) and [02](02_identifiability.md) is given a controlled test.
