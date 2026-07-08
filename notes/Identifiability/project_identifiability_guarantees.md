# Draft Identifiability Guarantees for the Epigenomics-GEX Causal Workflow

This note drafts identifiability assumptions and proof-sketch-style guarantees for the active monocyte causal-discovery workflow:

$$
\text{scGLUE joint latent space}
\to
\text{RNA-anchored CUT\&Tag pseudo-pairs}
\to
\text{nearby peak matrices}
\to
\text{PC, FCI, and DAGMA graph discovery}.
$$

The purpose is not to claim that the current pipeline fully identifies biological causality. The purpose is to state, in the style of causal-discovery papers, what assumptions would be needed for each kind of guarantee.

## Source-Paper Style Being Borrowed

The writing pattern follows four papers by Kun Zhang, Peter Spirtes, and collaborators:

- Zhang and Hyvarinen, "On the Identifiability of the Post-Nonlinear Causal Model": define a restricted model class, state what is identifiable, and explicitly describe non-identifiable exceptions. https://arxiv.org/abs/1205.2599
- Zhang, Gong, Ramsey, Batmanghelich, Spirtes, and Glymour, "Causal Discovery in the Presence of Measurement Error: Identifiability Conditions": separate the unobserved error-free causal variables from observed noisy measurements, then state what can be recovered from observed statistics. https://arxiv.org/abs/1706.03768
- Spirtes and Zhang, "A Uniformly Consistent Estimator of Causal Effects under the $k$-Triangle-Faithfulness Assumption": distinguish population-level identifiability from finite-sample uniform consistency, and state assumptions such as Causal Markov, faithfulness, and effect-size separation. https://arxiv.org/abs/1502.00829
- Ali, Richardson, and Spirtes, "Markov Equivalence for Ancestral Graphs": when latent variables or selection variables may exist, state recovery in terms of equivalence classes of ancestral graphs rather than a single fully oriented DAG. https://arxiv.org/abs/0908.3605

## Core Identifiability Template

Let $\mathcal{M}$ be the model class, $O(M)$ be the observable information produced by a model $M \in \mathcal{M}$, and $T(M)$ be the causal target.

The target $T$ is identifiable from $O$ over $\mathcal{M}$ if, for any $M_1, M_2 \in \mathcal{M}$,

$$
O(M_1) = O(M_2) \implies T(M_1) = T(M_2).
$$

For this project, the key question is:

> If two possible biological/data-generating processes produce the same integrated pseudo-paired nearby-peak matrix distribution, do they agree on the causal graph features we report?

The honest answer depends on which target we claim.

## Project-Specific Observable Objects

For a target gene $g \in \{\text{CSF1R}, \text{CD14}, \text{IL1B}\}$, define one graph-learning row as:

$$
W_{ig}
=
\left(
Y_{ig},
\widehat{C}_{ig1},
\ldots,
\widehat{C}_{igp_g}
\right),
$$

where:

- $Y_{ig}$ is RNA expression for gene $g$ in RNA anchor cell $i$.
- $\widehat{C}_{igj}$ is the pseudo-paired CUT&Tag signal for nearby peak variable $j$ around gene $g$.
- $p_g$ is the number of nearby peak variables retained for gene $g$.

For the RNA-anchor neighbor-average workflow with $k=10$:

$$
\widehat{C}_{igj}^{(m)}
=
\frac{1}{k}
\sum_{\ell \in N_k(i,m)}
C_{\ell gj}^{(m)},
$$

where $m$ is a histone mark and $N_k(i,m)$ is the set of $k$ nearest CUT&Tag cells of mark $m$ to RNA anchor $i$ in the scGLUE latent space.

The observable distribution for graph learning is therefore:

$$
P(W_g)
$$

or the corresponding empirical matrix with rows indexed by RNA anchors.

## Current Sample Sizes and Feature Counts

The active monocyte label filter selects $13{,}410$ cells across RNA and CUT&Tag modalities:

| Modality | Selected monocyte cells |
| --- | ---: |
| RNA | $1{,}887$ |
| H3K27ac | $1{,}577$ |
| H3K27me3 | $2{,}321$ |
| H3K4me1 | $1{,}561$ |
| H3K4me2 | $2{,}099$ |
| H3K4me3 | $2{,}390$ |
| H3K9me3 | $1{,}575$ |

The one-to-one RNA-anchor pseudo-pairing has:

$$
n = 1{,}561
$$

rows, limited by the smallest CUT&Tag mark after filtering.

The active RNA-anchor $k$-nearest-neighbor average workflow has:

$$
n = 1{,}887, \qquad k = 10
$$

RNA-anchor rows. Neighbor assignments are reused across RNA anchors, so the formal sample size for graph estimation should be treated as an effective sample size $n_{\mathrm{eff}} \le n$ unless row dependence is shown to be negligible.

For the current $1$ Mb nearby-peak roots, the unfiltered peak counts are:

| Gene | Nearby peak variables |
| --- | ---: |
| CSF1R | $69$ |
| CD14 | $113$ |
| IL1B | $82$ |

After nonzero filtering:

| Filter | CSF1R | CD14 | IL1B |
| --- | ---: | ---: | ---: |
| $p_{\mathrm{nonzero}} > 0.01$ | $60$ | $99$ | $74$ |
| $p_{\mathrm{nonzero}} > 0.05$ | $41$ | $73$ | $49$ |
| $p_{\mathrm{nonzero}} > 0.10$ | $26$ | $40$ | $28$ |

The elastic-net cross-fit screen keeps up to $30$ nearby peaks per gene and should be treated as candidate reduction, not causal evidence.

## Target Hierarchy for This Project

The desired target is the full local graph for each monocyte locus:

$$
G_g
\text{ over }
\{Y_g\}
\cup
\{\text{nearby mark-specific CUT\&Tag peak variables around } g\}.
$$

Because causal sufficiency is not a credible default assumption for this project, the primary identifiable target should be stated as an equivalence-class object rather than a fully resolved DAG.

The target hierarchy is:

1. **Primary target:** the local PAG/MAG equivalence class over observed expression and nearby CUT&Tag peak variables, allowing latent common causes.
2. **Aspirational target:** the full local graph if the PAG is sufficiently resolved or if additional orientation information justifies resolving uncertain edges.
3. **Backup target:** stable local edge orientations and adjacencies that persist across reasonable settings, such as feature filters, CI tests, conditioning depths, graph methods, and null checks.

Literature-supported promoters/enhancers should be used as an evaluation layer, not as an identifiability assumption. Literature support can make an edge biologically plausible, but it does not by itself prove the edge is identifiable from the observed distribution.

## Integration and Pseudo-Pairing Assumptions

These assumptions are needed before any causal graph guarantee can apply to the constructed matrix.

### I1. Shared Monocyte Target Population

All RNA and CUT&Tag cells retained by the harmonized coarse monocyte filter are samples from the same target biological population, possibly observed through different modalities.

Equivalently, there is a latent monocyte state $S$ such that RNA and each CUT&Tag mark measure different noisy views of the same state distribution:

$$
S \sim P_{\mathrm{mono}}.
$$

Why this matters: if RNA monocytes and CUT&Tag monocytes come from different biological populations, nearest-neighbor pairing in latent space has no causal interpretation.

### I2. Correct Enough Label Harmonization

The transferred/harmonized monocyte labels select the intended cell type with low error. Mislabeling is either negligible or independent of the graph variables after conditioning on observed covariates.

Why this matters: label error can create mixture distributions, and causal graphs for mixtures need not match the graph for any one cell type.

### I3. Cross-Modality Support Overlap

For every RNA anchor state $S_i$ used downstream, each CUT&Tag mark has nearby cells in the same region of latent biological state space.

In support language:

$$
\mathrm{supp}(S_{\mathrm{RNA}})
\subseteq
\mathrm{supp}(S_{\mathrm{CUT\&Tag},m})
$$

for each mark $m$, at least over the monocyte states used for graph learning.

Why this matters: without overlap, pseudo-pairing extrapolates instead of matching.

### I4. Latent-Space Neighborhood Consistency

Distances in the scGLUE latent space preserve biologically meaningful local neighborhoods. If two cells are close in latent space, their latent monocyte states are close enough that CUT&Tag measurements from one can approximate the other's chromatin state.

One formal version is:

$$
\|Z_a - Z_b\| \to 0
\implies
d_S(S_a, S_b) \to 0,
$$

where $Z$ is the scGLUE embedding and $d_S$ is a distance over latent biological states.

Why this matters: all downstream pseudo-pairing depends on nearest-neighbor distances being meaningful.

### I5. Neighbor-Averaged Measurement Error Is Controlled

For each peak variable $j$ and mark $m$, the neighbor average $\widehat{C}_{igj}^{(m)}$ is a controlled-error proxy for the latent chromatin value at RNA anchor state $S_i$:

$$
\widehat{C}_{igj}^{(m)}
=
C_{gj}^{(m)}(S_i) + \eta_{igj}^{(m)}.
$$

The error $\eta_{igj}^{(m)}$ is mean-zero or asymptotically negligible, and its dependence across rows is either negligible or handled through an effective sample-size argument.

For a formal nonparametric asymptotic guarantee, one would usually require:

$$
k_n \to \infty,
\qquad
\frac{k_n}{n_m} \to 0,
\qquad
n_m \to \infty
$$

for each mark-specific chromatin sample size $n_m$.

The current implementation uses fixed $k=10$, so the practical claim is weaker: neighbor averaging reduces noise and preserves all $1{,}887$ RNA anchors, but by itself it is not a full asymptotic proof that chromatin states are exactly recovered.

### I6. Batch, Donor, and Technical Effects Are Not Acting as Unmodeled Common Causes

Any donor, batch, chemistry, depth, or run effects that influence both pseudo-paired chromatin variables and expression are either adjusted, included as covariates, or negligible within the monocyte subset.

Why this matters: hidden common causes can create edges that are not direct regulatory relationships.

### I7. Feature Quantification Is Stable

Peak variables are constructed consistently from raw clean fixed-bin CUT&Tag signal. Library-size effects are either normalized away for Gaussian-style methods or included as offsets/covariates for count-family methods.

Why this matters: if peak scores are mostly library-size artifacts, graph edges can identify technical variation rather than regulatory structure.

## Sample-Size and Estimation Assumptions

Sample size does not create identifiability by itself. Identifiability is a population-level property. Sample size matters for estimating the population objects reliably.

For this project, a finite-sample consistency statement should assume:

### S1. Independent or Weakly Dependent Rows

The rows $W_{ig}$ are independent samples from $P(W_g)$, or the dependence induced by reused CUT&Tag neighbors is weak enough that the graph learner behaves as if it had:

$$
n_{\mathrm{eff}} \to \infty.
$$

A useful conservative statement is:

$$
n_{\mathrm{eff}} \le 1{,}887.
$$

### S2. Sparse Local Graphs

The number of direct neighbors of each graph node is small relative to $n_{\mathrm{eff}}$.

For PC/FCI with maximum conditioning depth $q$, the largest conditioning sets tested should satisfy:

$$
q \ll n_{\mathrm{eff}}.
$$

In practice, this supports the current plan of using low depths first, especially when $p_g$ is large.

### S3. Detectable Signal Strength

Relevant conditional dependencies are bounded away from zero, and irrelevant dependencies are close enough to zero that the test can separate them.

For partial-correlation-style tests, a generic version is:

$$
\min_{\rho_{ab \mid S} \neq 0}
|\rho_{ab \mid S}|
\ge
\lambda_n,
$$

where $\lambda_n$ must be large enough relative to sampling noise.

This is the practical role of strong faithfulness or weaker variants such as $k$-Triangle-Faithfulness.

### S4. Distributional Fit of the Test or Score

The chosen statistical test or graph score is appropriate for the variables:

- `log1p_norm` roots support Gaussian-style procedures such as Fisher-$z$ PC as an approximation.
- KCI/fastKCI support nonlinear conditional-dependence testing, but need enough samples and are numerically harder.
- raw-count roots require count-aware modeling, usually with library-size offsets.

### S5. Screening Is Fixed or Cross-Fit

If peaks are selected using expression association, the selection rule is either treated as fixed before graph learning or performed in a split/cross-fit way.

Otherwise the graph learner receives:

$$
S = f(X, y),
$$

where the variable set $S$ depends on the same data used for graph discovery. That weakens ordinary conditional-independence guarantees.

## Causal Graph Assumptions

These assumptions apply after the pseudo-paired matrix $W_g$ is constructed.

### C1. Local Graph Target

The target is not the full genome-wide regulatory graph. The desired target is a local graph over:

$$
\{Y_g\}
\cup
\{\text{nearby mark-specific CUT\&Tag peak variables around } g\}.
$$

However, because omitted regulators, transcription factors, signaling variables, genotype, donor effects, and cell-state variables may act as hidden common causes, the primary population target is the observed-variable equivalence class represented by a PAG or MAG.

### C2. Acyclicity

For DAG-based methods, the local causal structure among included variables is acyclic.

This does not mean biology has no feedback. It means the graph is interpreted as a time-collapsed or equilibrium approximation where the relevant causal ordering can be represented without directed cycles.

### C3. Causal Markov Condition

The distribution of the observed graph variables is Markov with respect to the relevant causal graph. Informally, once a node's direct causes are known, it is independent of non-effects that are not otherwise connected through those causes.

### C4. Faithfulness or Near-Faithfulness

The conditional independences in the distribution are exactly, or approximately, the ones implied by the graph. In particular, path cancellations do not make causally connected variables appear independent.

This is one of the strongest assumptions and should be stated explicitly.

### C5. No Causal Sufficiency as a Default Assumption

The main guarantee should not assume causal sufficiency. In this project, unobserved common causes are plausible because gene expression and chromatin marks may share unmeasured drivers such as transcription factors, genotype, donor state, cell-cycle state, signaling environment, batch, and other latent biological state variables.

Therefore, the default theoretical target is a PAG/MAG equivalence class over observed variables. PC and DAGMA can still be used as sensitivity analyses or stronger-assumption models, but their fully directed DAG outputs should not be the main identifiability claim unless additional assumptions are explicitly added.

### C6. Correct Model Family for Score-Based Methods

For mixed-family DAGMA, the nodewise likelihood family is correctly specified or close enough:

- continuous/log-normalized variables are treated as Gaussian-like,
- sparse binary-like peak presence may be treated as Bernoulli-like,
- count expression or count peaks may be modeled with Poisson/NB-like likelihoods,
- library-size offsets correctly account for exposure.

### C7. Positivity and Nondegenerate Variables

Each retained graph variable has enough variation. Structural nonzero filters help approximate this condition by removing near-constant peaks.

### C8. Correct Background Knowledge

If tiered prior knowledge constrains edge directions, those constraints are correct. Incorrect background knowledge can force wrong orientations even when the observational signal is otherwise adequate.

## Conservative Guarantees We Can State

### Guarantee 1: Pseudo-Paired Matrix as a Valid Observational Object

**Claim.** Under assumptions I1-I7 and S1, the RNA-anchor neighbor-average workflow produces an observational matrix whose rows approximate samples from a well-defined monocyte-level distribution $P(W_g)$ for each target gene $g$.

More formally, for each gene $g$:

$$
\widehat{P}_n(W_g)
\to
P(W_g)
$$

as the RNA and CUT&Tag sample sizes grow, provided latent-space matching error and measurement error vanish or remain controlled.

**Proof sketch.** The monocyte filter restricts all modalities to the same target population. Cross-modality support ensures each RNA anchor has comparable CUT&Tag cells. Latent-space neighborhood consistency ensures nearest neighbors approximate the same latent biological state. Neighbor averaging gives a controlled-error proxy for mark-specific peak signal. Therefore each constructed row is an approximate draw from the intended joint distribution over expression and nearby chromatin features.

**Important limitation.** This does not identify causal edges. It only justifies treating the pseudo-paired matrix as the observational input for causal discovery.

### Guarantee 2: FCI Identifies a PAG When Hidden Confounding Is Allowed

**Claim.** For a fixed gene $g$, assume C1-C4 and allow latent common causes among the observed variables. With exact conditional-independence information from $P(W_g)$, FCI identifies the PAG representing the Markov equivalence class of maximal ancestral graphs over the observed variables.

Symbolically:

$$
\mathcal{I}(P(W_g))
\implies
[G_g]_{\mathrm{PAG}}.
$$

**Proof sketch.** With latent variables, several underlying DAGs can induce the same observed conditional independences. Ancestral graphs summarize the observed-variable implications of those latent-variable DAGs. Under Markov and faithfulness assumptions, FCI uses conditional independences to recover the invariant adjacencies, arrowheads, and uncertainty marks shared by the compatible ancestral graphs. The output is not a single biological DAG; it is an equivalence-class summary.

**Interpretation.** This is the main identifiability guarantee for the project because it does not assume causal sufficiency.

## Concrete FCI Identifiability Draft

This is the most concrete version of the FCI argument for the current project.

### Step 1. Fix the Locus and Observed Node Set

Fix one gene:

$$
g \in \{\mathrm{CSF1R}, \mathrm{CD14}, \mathrm{IL1B}\}.
$$

After choosing a dataset root and a feature filter, define the observed graph node set:

$$
V_g
=
\{Y_g\}
\cup
P_g,
$$

where:

- $Y_g$ is the selected expression column, usually `expr__GENE_log1p` for FCI/PC.
- $P_g$ is the retained set of nearby mark-specific CUT&Tag peak variables.

For the current `run_pc_causallearn.py` auto-column logic, the FCI node set includes expression log1p columns and histone peak columns, but excludes `libsize__...` columns. Thus library-size columns are not graph nodes in the default FCI target. They are preprocessing/normalization information for normalized roots, or offsets for count-family DAGMA runs.

The concrete observable object is:

$$
W_g = (W_v : v \in V_g),
\qquad
P(W_g).
$$

### Step 2. Define the Latent-Variable Model Class

For FCI, the model class should allow hidden common causes:

$$
\mathcal{M}^{\mathrm{FCI}}_g
=
\{M:
M \text{ has an underlying acyclic causal graph over } V_g \cup H_g
\text{ and induces } P(W_g)\},
$$

where $H_g$ contains unobserved variables such as:

- transcription factors,
- genotype,
- donor state,
- batch or chemistry effects,
- cell-cycle or activation state,
- unmeasured signaling variables,
- unmeasured chromatin or expression regulators.

The underlying graph over $V_g \cup H_g$ is not assumed causally sufficient over $V_g$.

### Step 3. State the Project-Specific FCI Assumptions

**A1. Constructed observational distribution.** The scGLUE RNA-anchor neighbor-average procedure produces rows that approximate samples from a well-defined selected-monocyte distribution:

$$
W_{ig} \sim P(W_g \mid \mathrm{monocyte}, \mathrm{QC}).
$$

This uses the integration assumptions: shared monocyte population, cross-modality support overlap, latent-space neighborhood consistency, and controlled pseudo-pairing error.

**A2. Fixed graph variable set.** The set $V_g$ is treated as fixed before running FCI. If nonzero filters or elastic-net screens are used, the target becomes the graph over the retained variables. If expression-informed screening uses the same data as graph discovery, the resulting graph should be treated as exploratory unless screening is split or cross-fit.

**A3. Underlying local causal DAG with latent variables.** There exists an acyclic causal graph $D_g$ over observed variables $V_g$ and latent variables $H_g$ whose marginal distribution over $V_g$ is the distribution FCI sees.

**A4. Selected-population Markov condition.** The selected-monocyte distribution is Markov with respect to the latent-variable graph after marginalizing $H_g$ and conditioning on the selection event used to define the analysis population.

Equivalently, the observed distribution is Markov with respect to a maximal ancestral graph:

$$
P(W_g)
\text{ is Markov to }
M_g,
$$

where $M_g$ is the MAG induced over $V_g$.

**A5. Faithfulness.** The conditional independences in $P(W_g)$ are exactly the conditional independences entailed by $M_g$:

$$
X \perp Y \mid Z \text{ in } P(W_g)
\iff
X \perp_m Y \mid Z \text{ in } M_g.
$$

Here $\perp_m$ denotes $m$-separation in the MAG.

**A6. Correct conditional-independence oracle.** At the population level, FCI is given the correct answer to every conditional-independence query among variables in $V_g$:

$$
X \perp Y \mid Z
\quad
\text{for } X,Y \in V_g,\ Z \subseteq V_g \setminus \{X,Y\}.
$$

In finite samples, this becomes a consistency and signal-strength assumption for the chosen CI test, such as Fisher-$z$, KCI, fastKCI, or RCIT.

**A7. Selection is part of the target.** Because the analysis filters to high-confidence monocytes and QC-passed cells, the target is the local graph for the selected monocyte population. If the filtering event creates collider bias, FCI identifies the equivalence class for the selected distribution, not necessarily the graph for all immune cells or all monocytes before filtering.

**A8. Population FCI is run without search truncation.** The clean identifiability theorem assumes the population FCI procedure has access to all conditional-independence decisions it needs, including the relevant Possible-D-Sep search and orientation rules. If the implementation caps conditioning depth, `fci_possible_dsep_depth`, or `fci_max_path_length` for runtime, the output should be treated as a bounded-FCI approximation or robustness result rather than the exact population FCI guarantee.

### Step 4. State the Identifiable Target

The target is the PAG equivalence class:

$$
[M_g]_{\mathrm{PAG}}.
$$

This class represents all MAGs over $V_g$ that entail the same conditional independences as $P(W_g)$.

The full local graph is identifiable only for features invariant in this class:

$$
T_g
=
\mathrm{Inv}([M_g]_{\mathrm{PAG}}).
$$

Concrete examples of identifiable features are:

- an adjacency that appears in every compatible MAG,
- an arrowhead that appears in every compatible MAG,
- an absent edge implied by a separating set,
- a definite non-ancestor relation represented by invariant PAG marks.

Unresolved PAG marks are not identified from the observed distribution alone.

### Step 5. Proof Sketch

**Claim.** Under A1-A8, the FCI population output identifies the PAG equivalence class $[M_g]_{\mathrm{PAG}}$ for the local observed node set $V_g$.

**Proof sketch.**

1. By A1 and A2, the graph learner has a well-defined observed distribution $P(W_g)$ over a fixed observed node set $V_g$.
2. By A3, this distribution arises from an underlying acyclic causal graph over observed and latent variables.
3. Marginalizing latent variables, and conditioning on the selected monocyte population if needed, induces a MAG $M_g$ over the observed variables.
4. By A4 and A5, the conditional independences in $P(W_g)$ are exactly the $m$-separations in $M_g$.
5. By A6, population FCI receives the correct conditional-independence oracle.
6. By A8, FCI is not artificially truncated in a way that removes required separating sets or orientation paths.
7. The soundness and completeness result for FCI then implies that FCI returns the PAG representing the Markov equivalence class of $M_g$.
8. Therefore any edge feature shared by all MAGs in the PAG equivalence class is identifiable from $P(W_g)$.

In the identifiability template:

$$
P_{M_1}(W_g) = P_{M_2}(W_g)
\implies
[M_1]_{\mathrm{PAG}} = [M_2]_{\mathrm{PAG}}.
$$

For the invariant target:

$$
P_{M_1}(W_g) = P_{M_2}(W_g)
\implies
\mathrm{Inv}([M_1]_{\mathrm{PAG}})
=
\mathrm{Inv}([M_2]_{\mathrm{PAG}}).
$$

### Step 6. Finite-Sample Version

The population proof assumes an exact CI oracle. With the current finite matrices, add these estimation assumptions:

1. The effective sample size satisfies $n_{\mathrm{eff}} \to \infty$; practically, $n_{\mathrm{eff}} \le 1{,}887$ for the current neighbor-average rows.
2. The local graph is sparse enough that the relevant conditioning sets are not too large.
3. Nonzero conditional dependencies are strong enough to be detected at the available sample size.
4. The chosen CI test is appropriate for the transformed data.
5. Reused CUT&Tag neighbors do not create dependence strong enough to invalidate the test, or this is handled by sensitivity analysis.
6. Any bounded FCI settings are reported as computational approximations, and exact FCI claims are reserved for runs whose search settings do not rule out required separating sets or orientation paths.

Then the estimated PAG converges to the population PAG:

$$
\widehat{[M_g]}_{\mathrm{PAG}}
\to
[M_g]_{\mathrm{PAG}}.
$$

This is an estimation consistency statement, not a new identifiability statement.

### Step 7. What the Evaluation Layer Adds

The evaluation layer can support interpretation of the FCI output, but it does not replace the assumptions above.

Use evaluation to ask:

- Does an edge involve a peak overlapping a curated promoter/enhancer?
- Does the edge survive reasonable graph settings?
- Does it weaken under pairing-null or region-shift null tests?
- Does it recur across `nz001`, `nz005`, `nz010`, unfiltered, and screened roots?

These checks help prioritize graph features:

$$
\text{identified or stable graph feature}
\quad
+\quad
\text{literature support}
\quad
\Rightarrow
\quad
\text{higher-priority biological candidate}.
$$

They do not imply:

$$
\text{literature support}
\Rightarrow
\text{identifiability}.
$$

### Guarantee 3: Full Local Graph Recovery Requires Additional Orientation Information

**Claim.** The full local graph $G_g$ is identifiable from $P(W_g)$ only for the edge marks and orientations that are invariant across the PAG/MAG equivalence class, unless additional valid information is supplied.

If the PAG has no unresolved marks for the local graph, then the full observed-variable local graph is identified. Otherwise, only the invariant features are identified:

$$
\mathcal{I}(P(W_g))
\implies
\mathrm{Inv}([G_g]_{\mathrm{PAG}}),
$$

where $\mathrm{Inv}([G_g]_{\mathrm{PAG}})$ denotes the adjacencies, arrowheads, tails, and absence claims that are shared by every graph in the equivalence class.

Additional orientation information could include valid interventions, time ordering, credible biological direction constraints, or background knowledge whose correctness is explicitly assumed.

**Proof sketch.** The PAG represents all observed-variable ancestral graphs compatible with the conditional independences. If every graph in the class agrees on an adjacency or orientation, that feature is identifiable. If two compatible graphs disagree, then the feature is not identifiable from the observed distribution alone. Additional correct orientation information can shrink the equivalence class and may identify more of the local graph.

### Guarantee 4: Stable Edge Orientations Are an Empirical Backup Target

**Claim.** When the full local graph is not identifiable, the project can still report a stable-edge target:

$$
T_{\mathrm{stable}}(g)
=
\{e:
e \text{ appears with the same orientation across accepted settings}\}.
$$

Accepted settings may include different nonzero filters, conditioning depths, CI tests, graph methods, bootstrap samples, pairing-distance filters, and null controls.

**Proof sketch.** Stability is not an identifiability theorem by itself. It is an empirical robustness target. If an edge orientation is invariant across perturbations of reasonable analysis choices and is also compatible with the PAG orientation marks, then it is stronger evidence than a one-run edge. Literature support can further validate biological plausibility.

**Important limitation.** Literature support and empirical stability do not replace identifiability assumptions. They support interpretation and prioritization of edges for follow-up.

### Guarantee 5: PC Identifies a CPDAG Only Under the Stronger Causal-Sufficiency Sensitivity Model

**Claim.** If, as a sensitivity analysis, we add causal sufficiency for the included variables and assume exact knowledge of all conditional independences in $P(W_g)$, then PC identifies the Markov equivalence class of the observed-variable DAG.

Symbolically:

$$
\mathcal{I}(P(W_g))
\implies
[G_g]_{\mathrm{CPDAG}}.
$$

**Proof sketch.** By the Causal Markov and faithfulness assumptions, conditional independences in $P(W_g)$ correspond to graphical separations in the true DAG. The PC skeleton step removes edges exactly when a separating set exists. Orientation rules then orient unshielded colliders and compelled directions. The result is the CPDAG, representing the DAGs that are Markov equivalent to the true graph.

**Interpretation.** This should be clearly labeled as a stronger-assumption sensitivity model, not the primary guarantee.

### Guarantee 6: DAGMA Can Identify a Sparse DAG Only Under Stronger Model Assumptions

**Claim.** For a fixed gene $g$, suppose the observed pseudo-paired variables are generated by an acyclic structural equation model in the DAGMA model class, with independent nodewise noise, correct or approximately correct likelihood families, correct offsets for count variables, sparsity, and a unique population optimum. Then the population DAGMA objective identifies the observed-variable DAG.

Symbolically, if:

$$
W_j = f_j(W_{\mathrm{pa}(j)}) + \varepsilon_j,
\qquad
\varepsilon_j \perp \varepsilon_k \text{ for } j \neq k,
$$

and the acyclicity and family assumptions hold, then the true parent sets are identifiable as the unique minimizer of the population score.

**Proof sketch.** The correct structural family implies that each node's conditional distribution is best explained by its true parents. Independent residuals and acyclicity rule out cyclic or wrong-parent explanations. Sparsity and uniqueness prevent multiple graphs from achieving the same population score. Therefore the population optimum recovers the observed-variable DAG.

**Important limitation.** This is stronger and less assumption-light than the PC/FCI equivalence-class guarantees. It should be presented as model-based recovery, not as a generic causal guarantee.

### Guarantee 7: Structural Nonzero Filtering Preserves the Target Only If True Regulators Are Not Filtered Out

**Claim.** Nonzero filtering preserves the local causal target if every true direct regulatory peak variable has:

$$
p_{\mathrm{nonzero}} > t
$$

for the chosen threshold $t$.

**Proof sketch.** The graph learner can only recover edges among variables included in the matrix. If a true direct regulator is removed by the filter, the target changes from the original local graph to the induced graph over the retained variables. Therefore filtering is compatible with the original target only under a retention assumption.

**Practical interpretation.** Report graphs separately for unfiltered, `nz001`, `nz005`, `nz010`, and elastic-net-screened roots. Treat agreement across roots as robustness evidence, not proof of identifiability.

## Recommended Wording for the Current Project

The safest statement is:

```text
Under shared-population, cross-modality-overlap, and latent-neighborhood
consistency assumptions, the scGLUE RNA-anchor neighbor-average workflow defines
an approximate observational distribution over gene expression and nearby
CUT&Tag peak variables for each target monocyte locus.

Conditional on this constructed distribution, and under the Causal Markov and
faithfulness assumptions, FCI identifies a PAG/MAG equivalence class over the
observed local variables while allowing latent common causes. This is the primary
identifiable object for the current project.

The current data should therefore be interpreted as supporting equivalence-class
and robustness claims over local observed variables. The desired full local graph
can be reported only to the extent that its adjacencies and orientations are
invariant in the identified equivalence class or justified by additional valid
orientation information.
```

A stronger statement, only if the mentor is comfortable with more assumptions:

```text
If the pseudo-paired matrix is a consistent proxy for matched RNA/chromatin
states, if the local ancestral graph is sparse and faithful, and if
conditional-independence tests are consistent at the effective sample size of the
pseudo-paired rows, then the estimated local PAG converges to the population PAG
as sample size increases.
```

For DAGMA:

```text
For mixed-family DAGMA, graph recovery further requires correct nodewise family
specification, correct library-size offsets, independent structural noise,
acyclicity, sparsity, and uniqueness of the population optimum. Under those
stronger assumptions, the population score identifies the observed-variable
local DAG. This should be treated as a stronger-assumption sensitivity analysis
because the primary project guarantee does not assume causal sufficiency.
```

For the backup stable-edge target:

```text
When full local graph recovery is not justified, we report stable oriented edge
features that persist across feature filters, graph-learning settings, and null
controls. Literature-supported promoters/enhancers are used as biological
validation of these candidate edges, not as assumptions that make the graph
identifiable.
```

## What We Should Not Claim Yet

Do not claim the current observational/pseudo-paired workflow identifies:

- the full genome-wide biological causal graph of monocyte regulation,
- all edge directions among nearby peaks and expression,
- causal effects of chromatin perturbations on expression,
- absence of hidden confounding,
- causal sufficiency among the local observed variables,
- true matched multi-omic profiles at the single-cell level,
- validity of graph p-values after same-sample elastic-net screening,
- causal interpretation of edges that disappear under null pairing or region-shift tests,
- identifiability of an edge merely because it overlaps a literature-supported enhancer or promoter.

## Questions to Resolve Before Finalizing the Guarantees

1. Which PAG features should count as enough support to report a full local graph edge: definite adjacencies only, definite arrowheads, or only fully oriented directed edges?
2. Should stable edge orientation require agreement across PC, FCI, and DAGMA, or only across settings within the same method family?
3. How should literature support be weighted: validation-only, prioritization score, or a background-knowledge constraint for orientation?
4. Should donor, batch, genotype, cell-cycle, or library-size variables be explicit graph covariates, adjustment variables, or only QC checks?
5. Should pseudo-pairing be treated as fixed preprocessing, or should matching uncertainty be part of the formal model?
6. For neighbor averaging, do we want a formal asymptotic statement with $k_n \to \infty$ and $k_n/n_m \to 0$, or a practical fixed-$k=10$ bounded-error statement?
7. Should screened datasets be described as exploratory only, or do we want a strict split-sample/cross-fit theorem for screening followed by graph learning?

## Short Proof-Sketch Template for the Paper

```text
Proof sketch.
First, under the integration assumptions, the RNA-anchor neighbor-average
procedure yields an approximate sample from a well-defined joint distribution
over expression and nearby CUT&Tag peak variables for each monocyte locus.

Second, under the causal Markov and faithfulness assumptions, conditional
independence relations in this distribution correspond to graphical separations
in the underlying local ancestral graph over observed variables, allowing latent
common causes.

Third, FCI recovers the PAG representing the compatible local ancestral graphs.
Thus the primary identifiable object is the equivalence class, not necessarily a
single fully oriented DAG.

Finally, the full local graph is reported only where adjacencies and orientations
are invariant across the equivalence class or supported by additional valid
orientation information. Where this is not possible, stable edge orientations
across analysis settings become the backup reporting target. Finite-sample
recovery requires consistent tests, enough effective samples relative to the
number of variables and conditioning depth, sparse local graphs, and detectable
nonzero dependencies.
```
