# Why modern integration objectives do not generally identify causal structure

_2026-11-01. Method-specific companion to the unpaired-integration proof._

## What we can prove

For the strictly disjoint observational setting in this project, **scGLUE,
scMRDR, and scConfluence cannot guarantee recovery of the original cross-assay
causal graph over a model class that permits the counterexamples in
[Note 07](07_unpaired_integration_nonidentifiability.md)**. Their training
objectives cannot distinguish those causal systems, even at the population
level and even with perfect optimization.

This applies to our use of their outputs for causal discovery. It does not
assert that the authors promised general causal identification, that their
alignment tasks are impossible, or that they fail on every dataset. We must
separate three statements:

1. A method can align biological states or predict features successfully.
2. Its objective can select a particular cross-assay coupling.
3. That coupling and its causal graph are identified as the true biological ones.

The first two do not imply the third. Below, the method audit explains why the
general proof applies to each algorithm; an exact transport calculation then
shows how an objective can prefer the wrong coupling.

## 1. Recall the indistinguishable causal systems

With independent standard normal $S,U,V$, let

$$
X=S+U,\qquad
Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}V,
\qquad 0\leq\rho<1.
$$

Assay A measures $(S,X)$ and assay B $(S,Y)$ in independent disjoint cells.
Even if the true state $S$ is given to integration,

$$
X\mid S=s\sim\mathcal N(s,1),\qquad
Y\mid S=s\sim\mathcal N(s,1)
$$

for every $\rho$. At $\rho=0$ there is no direct $X\to Y$ edge; at
$\rho=1/2$ there is one. The conditional covariance and intervention coefficient
are $\rho$. Both models are faithful to their respective DAGs.

Note 07 proves equality of the entire observed-data laws, including appropriate
assay-local background/depth/count observations, and discusses the limits of
extending this construction. It is not enough to match two focal marginals
while allowing another observed feature to reveal which model generated them.
All genuinely available information, including fixed annotations $G$, must be
the same in both models.

## 2. The objective itself contains no distinguishing evidence

Write $P_m$ for the full observed law of an assay block, including its allowed
covariates. Fix preprocessing rules, masks, annotation $G$, architecture and
hyperparameters. A population integration objective has the form

$$
\mathcal L(\theta,\phi;P_1,\ldots,P_M,G),
$$

where $\theta$ comprises encoders/decoders and $\phi$ can include adversarial
parameters. Within-assay expectations, cross-assay distribution distances,
and transport plans constructed from the observed laws are all functions of
these arguments. A minibatch objective can instead be defined by its expectation
over the independent assay samples and algorithmic randomness; the conclusion
is the same.

If two causal models have the same $\{P_m\}$ and $G$, then for **every** fixed
parameter choice,

$$
\mathcal L_0(\theta,\phi)=\mathcal L_1(\theta,\phi).
$$

Therefore the entire objective is identical across the two causal models.
Where minima or equilibria exist, the sets of optima or equilibria are identical
as well. This does **not** require multiple optima within a single objective:
a unique optimum can select the same coupling in both causal worlds and be
wrong in one of them.

For actual finite training, Note 07 gives the stronger algorithm-independent
statement: the complete datasets, hence all training outputs, have the same
law. Early stopping, seeds, larger networks and hyperparameter selection using
only allowed unpaired observations do not escape it. Two independently drawn
finite datasets can of course yield different realized losses; those sampling
differences are not evidence about the missing causal edge.

An unrestricted downstream graph learner cannot repair this. It receives
encodings or pseudo-pairs derived from the same inputs. A forced binary edge
decision has worst-case error at least $1/2$ across the two countermodels, at
every sample size. Abstaining or reporting both possibilities remains valid.

## 3. Where the limitation enters each method

The equations below organize the active losses schematically; implementation
reductions and weights are documented in the linked source audits.

### scGLUE: reconstruct assays and guidance, align cell distributions

The unsupervised generator objective combines modality-specific data
reconstruction/KL losses, guidance-graph reconstruction/KL losses, and an
adversarial modality term:

$$
\mathcal L_{\rm gen}
=\sum_m w_m\mathcal L_{{\rm data},m}
+\lambda_G\mathcal L_G
-\lambda_A\mathcal L_{\rm modality\ CE}.
$$

The discriminator minimizes modality cross-entropy. Cell and feature embeddings
connect the reconstruction components. This description follows the
[official trainer source](https://scglue.readthedocs.io/en/latest/_modules/scglue/models/scglue.html).
The local audit checks scGLUE 0.2.3; these online docs describe 0.4.1. Our
adapter supplies neither paired-cell UIDs nor cell-type supervision. The
[source audit](../integration_identifiability_2026_11_01/reviews/modern_scglue_scconfluence.md)
records the local implementation and input flags.

Each observed assay law and the supplied guidance graph are unchanged in the
countermodels, so every component has the same population value at fixed
parameters. Guidance can favor a feature relationship, but proximity or a
candidate association does not itself specify its cell-level residual coupling
or intervention effect. If guidance actually encodes a true causal edge or a
valid identifying constraint, that is additional assumed information and must
be included in the theorem's inputs.

There is also a narrower, direct orientation limitation: scGLUE's documented
feature regulatory score is a cosine similarity, hence

$$
\operatorname{score}(i,j)=\operatorname{score}(j,i).
$$

That score alone cannot distinguish $i\to j$ from $j\to i$. A direction
assigned using feature roles or a supplied graph is coming from those inputs,
not the symmetry of the score. This does not prohibit additional downstream
identifying assumptions. See the
[official regulatory-inference definition](https://scglue.readthedocs.io/en/latest/reginf.html).

### scMRDR: shared/specific representation and within-assay geometry

The paper-consistent implementation combines masked ZINB reconstruction, KL
terms for shared and modality-specific latent variables, adversarial modality
alignment, and within-assay full-versus-shared distance preservation:

$$
\mathcal L_{\rm gen}
=\mathcal L_{\rm masked\ recon}+\beta\mathcal L_{\rm KL}
-\lambda_A\mathcal L_{\rm modality\ CE}
+\lambda_I\mathcal L_{\rm isometry}.
$$

It uses gene-aligned feature inputs and availability masks. The active model
has no graph encoder or cycle-consistency term. See
[Sun et al., Methods 3.2–3.5](https://arxiv.org/html/2510.24987v2) and the
[paper-consistent adapter](../../../pipelines/empirical/integration/methods/scmrdr/paper_faithful/README.md).

Our inference is that reconstruction and isometry constrain observed
within-assay information, while modality confusion constrains latent marginal
distributions. None measures which residual in one assay accompanied a residual
in the same cell's unobserved modality. Calling a latent component “shared” or
“specific” does not establish that its coordinates are biological causal
variables. Identical full observed laws and annotations give the same loss
for every fixed parameter choice, including the masks and geometry terms.

### scConfluence: learn embeddings around a constructed transport plan

scConfluence combines assay-specific autoencoder losses with an inverse-OT
loss based on a plan derived from connected features, plus an unbalanced
Sinkhorn divergence between latent distributions:

$$
\mathcal L
=\sum_m w_m\mathcal L_{{\rm AE},m}
+\lambda_T\sum_{m<n}\langle\pi_{mn}(C),c_Z\rangle
+\lambda_R\sum_{m<n}\operatorname{Sinkhorn}(P_{Z_m},P_{Z_n}).
$$

The input cost $C$ is computed using the observed feature conversions and
permitted feature connections. The plan attracts matched latent points; the
regularizer encourages distributional overlap. See the
[primary paper, Methods](https://www.nature.com/articles/s41467-024-51382-x) and
[pinned native-source audit](../integration_identifiability_2026_11_01/reviews/modern_scglue_scconfluence.md).
The paper's IOT formula uses squared Euclidean latent distance; the pinned
native IOT code uses unsquared Euclidean distance, which our pilot retains.
The information-limit argument applies to either version. The balanced
squared-cost example below is not an exact formula for either full objective.

The causal inference step is ours: a plan computed from the observed assay
blocks is a data-derived choice, not an additional observation of missing
cell pairs. Equal complete block laws imply equal laws of $C$, the fitted plan
and the final outputs. Partial transport can accommodate unmatched populations
without identifying the residual coupling of the cells it does align.

Our local count-input pilot corrects two native loss-handling errors and adapts
the reconstruction inputs. Those changes matter for empirical fidelity, but
neither add information that distinguishes the countermodels. The proof
does not rely on the old implementation bugs or on a failed fit.

## 4. An exact example: an alignment objective prefers the wrong coupling

This calculation is a deliberately simple **balanced, squared-cost transport
problem**, not the exact scConfluence objective. It demonstrates how selecting
an optimal coupling can differ from identifying the true coupling.

Work at a fixed state and standardize the residuals:

$$
R_X=X-S,\qquad R_Y=Y-S.
$$

Both observed residual marginals are standard normal. Their true joint
correlation is $\rho$, which is unobserved. Let $Q$ be a candidate coupling
with those two marginals, and let $r=\mathbb E_Q[R_XR_Y]$. Squared matching
cost is

$$
\mathbb E_Q[(R_X-R_Y)^2]=2-2r.
$$

Minimizing this cost selects $r=1$, realized by $R_Y=R_X$. This can happen
even when the real residuals are independent. The marginals match perfectly;
the chosen correspondence creates a residual relationship absent from the
true population. The unregularized optimizer is singular, so it is not itself
an ordinary positive-density CI example. Entropy regularization supplies one.

### Entropy regularization chooses the strength of that relationship

For a positive coefficient $\lambda$, consider

$$
J_\lambda(Q)
=\mathbb E_Q[(R_X-R_Y)^2]
+\lambda\operatorname{KL}(Q\,\|\,\varphi\otimes\varphi),
$$

where $\varphi$ is the standard normal law. We use natural-log KL and the full
squared cost as written; rescaling either changes the numerical value of
$\lambda$. For a jointly Gaussian candidate
$G_r$ with correlation $r\in(-1,1)$,

$$
J_\lambda(G_r)=2-2r-\frac{\lambda}{2}\log(1-r^2).
$$

This is not merely optimization within a guessed Gaussian family. Any
finite-KL coupling $Q$ with correlation $r$ satisfies

$$
\operatorname{KL}(Q\,\|\,\varphi\otimes\varphi)
=\operatorname{KL}(Q\,\|\,G_r)-\frac12\log(1-r^2).
$$

Finite KL excludes $|r|=1$, whose couplings are singular. The cost depends only
on $r$, so $G_r$ minimizes the objective among all couplings with that
correlation. The remaining scalar objective has

$$
J'_\lambda(r)=-2+\frac{\lambda r}{1-r^2},\qquad
J''_\lambda(r)=\lambda\frac{1+r^2}{(1-r^2)^2}>0.
$$

It diverges as $r$ approaches either endpoint and therefore has the unique
global minimizer

$$
r_\lambda=\frac{\sqrt{\lambda^2+16}-\lambda}{4}.
$$

| Transport choice | Selected residual correlation |
| --- | --- |
| No entropy penalty | $r=1$ |
| $\lambda=3$ | $r=1/2$ |
| $\lambda\to\infty$ | $r\to0$ |

In particular, $\lambda=3$ reconstructs the joint law of the $\rho=1/2$
countermodel and gets the $\rho=0$ model wrong. The observed inputs are
identical, so this parameter value is selected equally in both worlds. For
every finite positive $\lambda$, $r_\lambda>0$: conditional independence in
the true $\rho=0$ model becomes conditional dependence in the optimal
completed distribution. This is a population error, not a power issue.

If a cost or regularization parameter is independently known to encode the
correct coupling, recovery may be possible. The information-limit theorem says
that such correctness cannot be established from these identical marginals
alone. It does not say that a correctly specified transport model can never
recover a joint law.

## 5. Stronger consistency losses do not automatically solve identification

A generic cycle-consistent mapping illustrates the same issue. At a fixed
state $s$, consider

$$
F_+(x)=x,\qquad F_-(x)=2s-x.
$$

Each sends $\mathcal N(s,1)$ to $\mathcal N(s,1)$, each preserves pairwise
distances within that state, and each is its own inverse:

$$
F_+^{-1}(F_+(x))=x,\qquad F_-^{-1}(F_-(x))=x.
$$

Thus perfect cycle consistency, marginal alignment and that within-state
geometry criterion permit opposite residual correspondences. Fixed valid
asymmetric guidance may break this particular symmetry; that is an additional
constraint, not a contradiction. This is an illustrative construction, not a
claim that all these losses are present in scGLUE, scMRDR or scConfluence, or
that both maps minimize their full objectives.

Similarly, preserving all information within each assay does not observe its
missing cross-assay partners. The main proof does not depend on PCA, a
bottleneck, Gaussian decoder misspecification, or discarding raw counts.

## 6. What a downstream causal graph would mean

After fitting, the pipeline uses matching, neighbor averages or imputation to
construct a joint distribution $Q$. A correctly calibrated CI test on that
distribution answers questions about $Q$; it answers questions about the
original biological distribution $P$ only when the needed correspondence is
justified. In the regularized-transport example, even an exact conditional
independence oracle for $Q$ gives the wrong answer for the $\rho=0$ biology.

That does not make every PAG feature wrong or forbid useful state associations.
As Note 07 explains, a direct biological edge, a state-conditioned CI, and an
observed-variable PAG adjacency are distinct targets. The two-node marginal
PAG need not change in the state-loaded example, although the biological direct
edge and CI given state do change. Causal orientation can additionally remain
ambiguous even after the true joint law is supplied.

The resulting method-specific claim is:

> Under independent disjoint observational sampling, and without restrictions
> that distinguish the admissible biological couplings, the audited integration
> objectives and any downstream processing of their outputs cannot guarantee
> recovery of the original cross-assay direct causal structure. Alignment,
> reconstruction, geometry preservation and a unique optimum do not supply the
> missing identifying restriction.

This claim does not apply unchanged to paired or partially paired supervision,
shuffled views of the same cells, an independently informative reference, or a
correctly specified restricted causal model. The existing scMTR paired-reference
holdout has already been separated from the strictly disjoint experiment in
[Note 07](07_unpaired_integration_nonidentifiability.md).

The [research record](../integration_identifiability_2026_11_01/README.md)
links independent method and mathematics reviews, including the exact OT
derivation. No new worlds or model fits are used in this argument.
