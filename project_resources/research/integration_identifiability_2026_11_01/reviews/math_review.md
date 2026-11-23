# Independent mathematical review: unpaired integration and causal identification

Date: 2026-11-01. This is an algebraic review, with no new simulations, fits,
datasets or benchmark outcomes.

## Verdict and necessary scope

The proposed example proves a useful impossibility result: over a model class
containing the faithful fork and triangle below, genuinely disjoint assay
samples cannot identify a direct cross-assay edge, its causal effect, or the
conditional independence of the two variables given their true state. Giving
the learner the exact state does not remove this ambiguity.

The result is about uniform identification over the stated model class. It
does not say that every cross-assay relation, every restricted causal model,
every dataset, or every measured-variable PAG is unrecoverable.

Two qualifications are essential:

1. The first construction distinguishes joint PAGs over \(S,X,Y\), but does
   not distinguish the two-node PAG over \(X,Y\) after state is omitted. A
   separate construction below establishes broad measured-PAG
   nonidentifiability.
2. Genuinely independent assay samples are different from two independently
   shuffled views of the same finite cells. The exact finite-sample equality
   and one-half lower bound do not automatically apply to the latter.

## 1. Faithful fork and triangle with the same assay laws

Let \(S,U,V\) be mutually independent standard Gaussian variables. For
\(0\leq \rho<1\), define

$$
X=S+U,\qquad
Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}\,V.
$$

Substituting the first equation gives

$$
Y=S+\rho U+\sqrt{1-\rho^2}\,V.
$$

At \(\rho=0\), the minimal causal DAG is \(S\to X,\ S\to Y\). At
\(0<\rho<1\), it additionally contains \(X\to Y\), while the coefficient of
\(S\to Y\) remains strictly positive. All exogenous noise variances are
positive. The excluded endpoint \(\rho=1\) would make \(X=Y\).

The covariance matrix in order \(S,X,Y\) is

$$
\Sigma_\rho=
\begin{pmatrix}
1&1&1\\
1&2&1+\rho\\
1&1+\rho&2
\end{pmatrix},
\qquad \det(\Sigma_\rho)=1-\rho^2>0.
$$

The equations are acyclic with independent exogenous errors, so they satisfy
the causal Markov property. Faithfulness can be checked exhaustively in this
three-variable Gaussian model:

| Pair | Marginal covariance | Covariance given the third variable |
| --- | ---: | ---: |
| \(S,X\) | \(1\) | \((1-\rho)/2\) |
| \(S,Y\) | \(1\) | \((1-\rho)/2\) |
| \(X,Y\) | \(1+\rho\) | \(\rho\) |

For \(\rho=0\), the only nontrivial conditional independence is
\(X\perp Y\mid S\), exactly as the fork entails. For \(0<\rho<1\), there are
no nontrivial conditional independences, exactly as the complete triangle
entails. Pairwise checks cover all possible independences in this
three-variable Gaussian distribution.

This is not a cancellation or vanishing-signal example. At \(\rho=1/2\), the
triangle's nonzero partial correlations are at least \(1/\sqrt{7}\), while
the fork's nonzero partial correlations are at least \(1/2\). Comparing these
two fixed models already proves impossibility with dependencies bounded away
from zero. The two paths from \(S\) to \(Y\) in the triangle have positive
coefficients \(1-\rho\) and \(\rho\); keeping their sum fixed across models is
not destructive path cancellation within a model.

Nevertheless, for every parameter value,

$$
X\mid S=s\sim N(s,1),\qquad Y\mid S=s\sim N(s,1).
$$

Thus the two observable assay marginals \(P(S,X)\) and \(P(S,Y)\) are exactly
the same across the family.

### Finite, faithful state-background extension

Append any finite number of variables

$$
G_j=a_jS+\epsilon_j,\qquad a_j>0,\qquad
\epsilon_j\sim N(0,\sigma_j^2),\quad \sigma_j^2>0,
$$

with all new errors independent. Allocate these variables to any number of
additional assays or to background features within either focal assay.
Every within-assay law is still unchanged across \(\rho\), including the
joint law of all features measured together in that assay.

The enlarged Gaussian DAG, a state-parent star with an optional \(X\to Y\)
edge, is also faithful. Conditional on any subset of other background
variables, let \(t=\operatorname{Var}(S\mid G_{\mathrm{subset}})>0\).
The relevant covariances become

$$
\operatorname{Cov}(X,Y\mid G_{\mathrm{subset}})=t+\rho,
$$

$$
\operatorname{Cov}(S,X\mid Y,G_{\mathrm{subset}})
=\operatorname{Cov}(S,Y\mid X,G_{\mathrm{subset}})
=\frac{t(1-\rho)}{t+1}>0,
$$

and \(\operatorname{Cov}(X,Y\mid S,G_{\mathrm{subset}})=\rho\).
Background-to-core covariances are nonzero multiples of these expressions
when \(S\) is not conditioned on; they vanish when conditioning on \(S\)
blocks their only paths. For two unconditioned background variables,
their conditional covariance is
\(a_i a_j\operatorname{Var}(S\mid\text{conditioned variables})>0\)
unless \(S\) itself is conditioned on. These exhaust the remaining pairwise
Gaussian independence checks.

This extension establishes faithfulness for a finite informative background
panel. It does not assert a panel-size-independent lower bound on every
partial correlation.

## 2. Equality of the entire observed experiment

Let assay A observe \(S,X\) plus a within-assay block \(W_A\); assay B observes
\(S,Y\) plus \(W_B\). More generally, append fixed stochastic kernels

$$
K_A(d w_A\mid s,x),\qquad K_B(d w_B\mid s,y).
$$

They may include same-assay measurements, noisy residual predictors, library
depth, metadata, or subsequent measurement/quantification. The kernels and
their interpretation must be identical across the compared models.
Since \(P(S,X)\) and \(P(S,Y)\) are unchanged, so are the complete assay
laws \(P_A\) and \(P_B\).

For independently sampled cells in the two assays,

$$
D_A\sim P_A^{\otimes n_A},\qquad
D_B\sim P_B^{\otimes n_B},\qquad D_A\perp D_B,
$$

so

$$
\mathcal L_\rho(D_A,D_B)
=P_A^{\otimes n_A}\otimes P_B^{\otimes n_B}
$$

is exactly independent of \(\rho\), at every finite sample size. Adding
independent samples from further assays with the same fixed state-based
observation laws preserves this equality.

The arbitrary-kernel statement proves equality of observable laws. It does
not claim that arbitrary deterministic, nonlinear or discrete measurement
kernels preserve faithfulness of an enlarged graph. The explicit Gaussian
background construction above supplies a faithful full-variable extension
when that property is required.

The comparison also holds with identical fixed guidance, annotations and
prior constraints that admit both models. A prior that certifies the disputed
edge, a known shared residual bridge, joint-dependent selection information,
or another constraint excluding one model changes the admissible model class.
Such information cannot be silently treated as irrelevant.

## 3. Targets that actually differ

The full joint distributions differ:

$$
\operatorname{Cov}_\rho(X,Y)=1+\rho,\qquad
\operatorname{Cov}_\rho(X,Y\mid S)=\rho.
$$

The causal effect also differs. Replacing the equation for \(X\) by
\(X=x\) gives

$$
E_\rho[Y\mid\operatorname{do}(X=x),S=s]=(1-\rho)s+\rho x,
$$

and therefore

$$
\frac{\partial}{\partial x}
E_\rho[Y\mid\operatorname{do}(X=x)]=\rho.
$$

In contrast, the identifiable state-mean relationships in this family are
the same:

$$
E[X\mid S]=E[Y\mid S]=S,\qquad
\operatorname{Cov}_S(E[X\mid S],E[Y\mid S])=1.
$$

Perfect state alignment recovers this between-state term. It does not
determine the remaining \(\rho\), or establish that the state-mean association
is a direct regulatory effect.

## 4. Any integration pipeline inherits the ambiguity

Represent the entire algorithm as a Markov kernel \(Q(d o\mid D_A,D_B,G)\),
where \(G\) denotes permitted common side information. This includes learned
embeddings, data-dependent feature selection, matching, transport, imputation,
aggregation, bootstrap resampling, neural networks, CI testing, graph search
and randomized initialization. The kernel may depend arbitrarily on the
observed data; it may not receive the unknown true model through extra
side information.

The output law is

$$
\mathcal L_\rho(O)=
\int Q(d o\mid d_A,d_B,G)\,
P_A^{\otimes n_A}(d d_A)P_B^{\otimes n_B}(d d_B),
$$

which is identical under every compared model. More computation, averaging
over all states, or a unique optimum of an integration objective does not
change that equality.

For a binary decision \(\widehat e\) about whether the direct edge \(X\to Y\)
exists, compare \(\rho=0\) and a fixed \(r\in(0,1)\), such as \(r=1/2\).
Let \(q=P(\widehat e=1)\) under their common data law. Then

$$
P_0(\widehat e\ne0)=q,\qquad
P_r(\widehat e\ne1)=1-q,
$$

so

$$
\max\{P_0(\text{edge error}),P_r(\text{edge error})\}\geq\frac12.
$$

The equal-prior average error is exactly one-half. This is a finite-sample
and asymptotic impossibility result, not a finite-power argument.
It does not require a particular CI test to be miscalibrated.

For any causal-effect estimator \(\widehat\tau\), the triangle inequality
similarly gives

$$
\max\{E_0|\widehat\tau|,\ E_r|\widehat\tau-r|\}\geq r/2.
$$

An algorithm can abstain or return a valid ambiguity set. That is different
from uniformly recovering the disputed edge. In this explicit family the
observed data leave every \(\rho\in[0,1)\) admissible.

## 5. PAG claims require a fixed, explicit node set

A PAG describes a Markov equivalence class, not a unique underlying biological
DAG. Ancestral-graph equivalence is defined through the conditional
independences represented after marginalization/selection; different graphs
can encode the same independences.
[Ali, Richardson and Spirtes (2009)](https://arxiv.org/abs/0908.3605)
provide the formal ancestral-graph equivalence framework.

For the first construction on \(\{S,X,Y\}\), the fork and triangle have
different CI structures and different adjacencies. In particular,
\(X\perp Y\mid S\) is true only in the fork. Thus the corresponding faithful
joint PAGs on this node set cannot be selected from the separate assay laws.

However, if the target contains only \(\{X,Y\}\), both models make those
variables dependent. With unrestricted latent confounding and no orientation
prior, the two-node PAG is \(X\mathbin{\circ\!-\!\circ}Y\) in both cases.
The first construction therefore does **not** prove nonidentification of
that particular two-node PAG. It still proves nonidentification of the causal
effect, biological direct edge and numerical joint covariance.

### Separate measured-PAG counterexample

Keep all shared-state background variables, but take focal variables

$$
X=U,\qquad Y=\rho X+\sqrt{1-\rho^2}\,V,
$$

independent of \(S\) except for the common availability of the separate
state-background component. The observed assay laws remain unchanged across
\(\rho\). At \(\rho=0\), \(X\perp Y\), so their joint two-node PAG is empty.
At \(\rho>0\), they are dependent, so the joint two-node PAG contains an
adjacency. Both focal models are nondegenerate and faithful. Appending the
independent shared-state component preserves faithfulness.

This proves that a broad shared-state multiview model class need not identify
even a measured-variable adjacency. It permits some focal variables to have
state-independent variation. It does not establish the same claim for a
narrower class requiring every focal variable to have a nonzero state loading.

Consequently “we only need a PAG” does not generally eliminate the missing
joint-distribution problem. Conversely, one should not assert that every
specific PAG feature is ambiguous. The node set and admissible model class
determine the target.

## 6. Nonlinearity and count observations

### Smooth invertible nonlinear transformations

Replace the core variables by
\(\widetilde X=g_X(X)\), \(\widetilde Y=g_Y(Y)\), and optionally
\(\widetilde S=g_S(S)\), using the same componentwise smooth, strictly
increasing bijections in every model. Logistic transformations are one
example, with inverse defined on the open interval \((0,1)\).

Each transformed variable generates the same sigma-algebra as its original.
Therefore all CI statements, the core faithfulness distinction, and
observational-law equality are preserved. The transformed structural equation

$$
\widetilde Y=
g_Y\!\left(
(1-\rho)g_S^{-1}(\widetilde S)
\rho g_X^{-1}(\widetilde X)
\sqrt{1-\rho^2}\,V
\right)
$$

is nonlinear. At \(\rho=0\) it does not depend on \(\widetilde X\); at
\(\rho>0\) it does. For transforms with positive derivatives, the response
to \(\widetilde X\) is strictly positive at fixed state/noise.
Nonlinearity alone therefore does not restore identification.

This does not claim that every identifiable nonlinear additive-noise model
class contains both transformed models. A particular structural restriction
can exclude the counterexample and must be assessed separately.
Noninvertible coarsening preserves equality of observable laws but need not
preserve graph edges or faithfulness.

### Identical count channels

Applying the same per-assay Poisson, negative-binomial or other observation
kernel to identical assay laws leaves those laws identical. This immediately
preserves nonidentification of the latent biological edge/effect. It does not
automatically preserve a particular graph over measured count variables:
a constant-output kernel, for example, can erase every dependency.

A concrete informative count example uses independent measurement noise:

$$
C_X\mid X,d_X\sim\operatorname{Poisson}(d_Xe^X),\qquad
C_Y\mid Y,d_Y\sim\operatorname{Poisson}(d_Ye^Y),
\qquad d_X,d_Y>0.
$$

For fixed oracle state and depths, the separate count laws are unchanged
across \(\rho\), while Gaussian exponential moments give

$$
\operatorname{Cov}(C_X,C_Y\mid S=s,d_X,d_Y)
=d_Xd_Ye^{2s+1}(e^\rho-1).
$$

At \(\rho=0\), the counts are conditionally independent given state and
depths. At \(\rho>0\), their strictly positive conditional covariance proves
dependence. Thus the missing conditional joint remains consequential after
real-valued activities are replaced by noisy counts.

For the separate state-independent focal construction, the same channels
give

$$
\operatorname{Cov}(C_X,C_Y)=d_Xd_Ye(e^\rho-1).
$$

At \(\rho=0\), the two counts are independent; at \(\rho>0\), they are
dependent. This supplies the analogous empty-versus-adjacent two-count PAG
example without requiring an invertible or noise-free measurement.
The arrows between count measurements are not automatically the biological
activity arrows: those are different target graphs.

## 7. Direction and latent-common-cause alternatives

For fixed \(r\in(0,1)\), the reverse-direction model

$$
Y=S+V',\qquad
X=(1-r)S+rY+\sqrt{1-r^2}\,U'
$$

has the same full Gaussian joint law of \(S,X,Y\) as the forward triangle.
It has the opposite direct edge and zero effect of intervening on \(X\) on
\(Y\). This is ordinary observational causal ambiguity even with paired
measurements; it should not be attributed solely to unpairing.

Likewise, with mutually independent standard Gaussians \(S,L,E_X,E_Y\),

$$
X=S+\sqrt r\,L+\sqrt{1-r}\,E_X,\qquad
Y=S+\sqrt r\,L+\sqrt{1-r}\,E_Y
$$

has the same full joint law but no direct \(X-Y\) edge. It instead has an
additional latent residual common cause \(L\). This illustrates the separate
reason that identifying a joint distribution need not identify a direct
regulatory DAG when hidden confounding is allowed.

The fork-versus-triangle comparison remains the clean primary argument:
unpairing hides a CI distinction that a genuinely paired joint distribution
would reveal.

## 8. Same-cell shuffled views are outside the product-law proof

Suppose one draws \(n\) complete cells from the first model and independently
shuffles their two assay lists. Removing row identities leaves both sample
means unchanged. Consequently

$$
\operatorname{Cov}(\overline X,\overline Y)=\frac{1+\rho}{n}.
$$

If exact state values are also retained,

$$
\operatorname{Cov}(\overline{X-S},\overline{Y-S})=\frac{\rho}{n}.
$$

These are properties of the distribution over repeated whole datasets.
They show that the two unordered multisets need not have the same finite-sample
joint law across models. They do not, by themselves, show consistent estimation
from one shuffled cohort. With exact continuous state recorded in both
same-cell views, matching equal state values would additionally identify the
pairing almost surely.

The scMTR paired holdout consists of the same cells with assay-specific row
ordering and hidden identities. Its input design is therefore not the
independent-product experiment used in the lower bound. Its disjoint training
views and its paired evaluation reference must be discussed separately.
Empirical retrieval failures and sparse-CI calibration failures are evidence
about that pilot, not a proof of this theorem.

## 9. What remains learnable and what additional assumptions change

Given correctly aligned oracle state, each assay's full conditional
distribution, its within-assay relationships, and the covariance of conditional
means across state are observable targets. Their statistical identification
does not itself give them a causal interpretation.

One may further assume

$$
X_A\perp X_B\mid S.
$$

Then the joint completion

$$
P(d x_A,d x_B)=\int P(d x_A\mid s)P(d x_B\mid s)P(d s)
$$

is identified. This removes the disputed residual coupling by assumption;
it does not discover that coupling from disjoint cells.

A specified invariant equation with an exclusion restriction can also be
identifiable. For example, if one assumes \(Y=\beta X+\epsilon\),
\(E[\epsilon\mid S]=0\), no separate direct state effect, and sufficient
variation in \(E[X\mid S]\), then
\(E[Y\mid S]=\beta E[X\mid S]\) can identify \(\beta\).
Parent-set knowledge, exclusion, invariance and rank are doing the work.
The unrestricted fork/triangle family does not satisfy that exclusion.

The appropriate general output is the set of causal targets compatible with
the observed assay laws and explicit restrictions. There are two layers:
alternative cross-assay joint completions, and ordinary causal equivalence
within each completed joint. A single ordinary PAG summarizes the second
layer for a specified joint CI structure. It does not generally summarize a
union of completions with different adjacency skeletons. Circle endpoints
express orientation ambiguity, not uncertainty about whether an edge exists.

## 10. Corrections recommended in current notes

- [02 — Identifiability](../../theory/02_identifiability.md) should not claim
  that disjoint aggregation universally converts every direct cross-assay
  edge into a bidirected PAG edge. Neither adjacency retention nor a
  particular endpoint pattern follows from the impossibility theorem.
  Stable directed outputs can be stable artifacts; impossibility does not
  predict that algorithms will never return them.
- [Draft workflow guarantees](../../identifiability/project_identifiability_guarantees.md),
  especially Guarantee 1, need a joint-completion assumption between
  same-state neighborhood consistency and recovery of the intended biological
  joint. A proxy for a state mean is not automatically a proxy for an
  individual cell's unobserved residual value.
- [FCI identifiability summary](../../identifiability/fci_identifiability_summary.md)
  can retain its conditional population theorem when given the correct
  joint distribution and suitable Markov/faithfulness/CI/search assumptions.
  That theorem does not establish that integration supplied that joint.
- [Latent-space aggregation](../../theory/06_latent_space_aggregation.md)
  already makes the main mean-versus-residual distinction correctly. Its
  restricted positive cases should remain alongside the negative theorem.

The defensible conclusion is failure of general cross-assay causal
identification under the stated information and model class, with identifiable
shared-state targets and restricted positive cases kept explicit.

## 11. Review of the actual main draft

Reviewed [07 — Unpaired integration nonidentifiability](../../theory/07_unpaired_integration_nonidentifiability.md)
in full. The reviewed snapshot has SHA256
d782e2781f6eb382ded0b87918ab2232515ae63c5b73fb9bc8da067ff143905c.
No blocking mathematical error was found.

The draft correctly states the independent disjoint observation experiment,
the worst-case rather than universal per-dataset one-half error bound,
fixed common guidance, the distinction between law-preserving channels and
faithfulness-preserving transformations, and the positive-depth Poisson
formula. It also correctly separates the node-set-dependent PAG claim,
ordinary reverse-direction equivalence, same-cell shuffled multisets, and the
restricted linear exclusion/rank result.

Two optional precision improvements were sent to the author:

1. Label the conditional-product matching distribution as an ideal population
   construction, so it is not mistaken for an exact finite-sample description
   of every nearest-neighbor or transport method.
2. Link the background discussion to Section 1 of this review when a finite
   fully faithful multifeature extension is desired. Arbitrary observation
   kernels preserve marginal-law equality but need not preserve full-graph
   faithfulness; the draft already makes that caveat correctly.

### Final draft approval

The current main note, SHA256
847fe532640c82a336b6517ccd8dfe82da7f69411b3f56798973cf97073f313b,
is approved within its explicit model and sampling scope. The exact
indistinguishability argument is the proof; algebra checks and empirical
benchmarks serve different purposes.

The revised older aggregation note and the revised I5, I8, Guarantee 1,
Guarantee 2 and A1 sections of the workflow guarantee note address the earlier
logical overclaims: state means need not reproduce same-cell residuals,
conditional FCI guarantees require a valid joint target, and nonidentifiability
does not force particular PAG endpoints or prevent stable wrong outputs.

Two small precision suggestions accompany this approval: specify equal-size
non-overlapping groups with common fixed weights in the older Gaussian
averaging statement, and use exact faithfulness in an exact population-FCI
guarantee. These do not alter the main impossibility theorem.
