# Independent mathematical review of integration objectives

Date: 2026-11-01. Scope: mathematical derivation and claim checking only. No
new simulations, model fits, worlds, or datasets were used.

## Review verdict

The proposed objective-level extension is valid under the same information
restriction as the [main impossibility proof](../../theory/07_unpaired_integration_nonidentifiability.md):
genuinely disjoint independent assay samples, identical complete observable
assay laws, and the same permitted guidance or other fixed information.

The Gaussian optimal-transport calculation is correct, including the stated
regularization value. It can be proved globally over all couplings, not just
by finding a stationary Gaussian candidate. The sign-map example is also a
valid illustration for a marginal-alignment plus cycle-consistency objective.
Neither example should be described as an exact derivation of the full
scGLUE or scConfluence training objective.

Three distinctions are necessary:

1. Equality of an objective across biological models is different from
   nonuniqueness of its optimizer. A unique optimizer can select a coupling
   unsupported by the observations.
2. Equal output laws do not require two independent training runs to return
   identical realized weights or graphs.
3. A coupling selected by an integration loss is not thereby identified as
   the biological joint distribution or its causal structure.

## 1. Objective equality follows from equality of the allowed observations

Let the population information available to integration be

$$
\mathcal O(M)=\bigl(P_{A_1,M},\ldots,P_{A_k,M},G\bigr),
$$

where each $P_{A_j,M}$ is the law of the complete observable block in assay
$j$, and $G$ contains all other permitted information. In particular, $G$
must not silently contain true same-cell pairs, a true residual coupling,
or a causal answer that differs between the countermodels.

Consider a fixed parameter space $\Theta$, fixed hyperparameters $\eta$, and
any population loss that is a functional of the permitted information:

$$
L_M(\theta)=\mathcal F\bigl(\theta;\mathcal O(M),\eta\bigr).
$$

If $\mathcal O(M_0)=\mathcal O(M_1)$, then

$$
L_{M_0}(\theta)=L_{M_1}(\theta)\qquad\text{for every }\theta\in\Theta.
$$

Consequently their infima and their argmin sets agree. This statement also
covers the case in which a minimum is not attained: the common argmin set
can be empty. Geometry, feature costs, latent embeddings, and auxiliary
parameters learned solely from the same permitted observations can be
included in $\theta$ or in the functional $\mathcal F$; their presence does
not change the implication.

A fixed tie-breaking rule, optimization procedure, or randomized selection
using only these observations also has the same output law in the two
models. For empirical losses, this is equality **in law** across independent
datasets. Conditional on the same observed dataset and the same random
seed, the allowed procedure has the same inputs and therefore the same
output, assuming a fixed implementation. Distinct randomly drawn datasets
or seeds need not produce equal realized losses or outputs.

The same reasoning covers data-dependent hyperparameter selection if it
uses only allowed observations. Selection informed by paired validation
measurements, ground-truth structure, interventions, or other information
not included in the equal observed experiment requires a new analysis.

This is an information argument, not a special property of gradient
descent, variational autoencoders, or transport. It does not require
multiple minimizers, imperfect optimization, a local minimum, or weak
signals. In particular, a strongly preferred or uniquely selected joint
completion can be the same in two biological models whose true joints
differ. It cannot equal both.

Under genuinely independent disjoint sampling, the finite-sample input
laws are also identical. Thus the binary edge decision in the main proof
still has errors $q$ and $1-q$ under the two models, and worst-case error
at least $1/2$. Objective equality is an interpretation of that theorem,
not a separate reason to extend its lower bound to shuffled views of the
same finite cells.

## 2. An exact Gaussian optimal-transport example

Suppose state is known exactly and has already been removed. The two
observed residual laws are

$$
P_U=P_V=\mathcal N(0,1).
$$

For example, these are the laws of $U=X-S$ and $V=Y-S$ in the main proof.
The notation $V$ here denotes the observable assay-B residual, not its
independent structural noise term. Its true correlation with $U$ could be
$\rho=0$ or $\rho=1/2$; neither value changes the two observed residual
marginals.

Let $\Pi(P_U,P_V)$ be all couplings with these marginals. For any such
coupling $Q$, write

$$
r=\mathbb E_Q[UV],\qquad -1\leq r\leq 1.
$$

Since the marginal means are zero and both variances equal one,

$$
\mathbb E_Q[(U-V)^2]=2-2r.
$$

### Unregularized squared-distance transport

The minimum cost is zero, achieved exactly when $U=V$ almost surely. This
is the unique diagonal coupling with standard normal marginals, and it
has $r=1$. Its joint distribution is singular. By comparison, the true
Gaussian couplings with $\rho=0$ and $\rho=1/2$ have costs $2$ and $1$.

Thus minimizing the distance between residuals selects perfect positive
residual association even when the true same-cell residual association is
absent. This is a property of the stated cost and constraints, not an
assertion that every integration method minimizes precisely this cost.

### Entropic regularization: proof over all couplings

Fix $\lambda>0$ and define

$$
J_\lambda(Q)=\mathbb E_Q[(U-V)^2]
 +\lambda\,D_{\mathrm{KL}}\bigl(Q\,\|\,P_U\otimes P_V\bigr).
$$

This convention uses the full squared distance, not half of it, and the
KL divergence uses natural logarithms. The numerical relation between
$\lambda$ and $r$ changes if either convention changes.

If the KL divergence is finite, $Q$ has a density relative to the
independent product distribution. Such a coupling has $|r|<1$: equality
would imply $U=V$ or $U=-V$ almost surely, a set of product measure zero.
For $r\in(-1,1)$, let

$$
G_r=\mathcal N\!\left(0,
\begin{pmatrix}1&r\\r&1\end{pmatrix}\right).
$$

The Gaussian $G_r$ has the same marginal laws and covariance as $Q$. The
log density ratio of $G_r$ to the independent product is a quadratic:

$$
\log\frac{g_r(u,v)}{p_U(u)p_V(v)}
=-\frac12\log(1-r^2)
 +\frac{2ruv-r^2(u^2+v^2)}{2(1-r^2)}.
$$

Its expectation under $Q$ is $-\tfrac12\log(1-r^2)$, because
$\mathbb E_Q[UV]=r$ and $\mathbb E_Q[U^2]=\mathbb E_Q[V^2]=1$.
All these quadratic expectations are finite. Therefore

$$
D_{\mathrm{KL}}(Q\|P_U\otimes P_V)
=D_{\mathrm{KL}}(Q\|G_r)-\frac12\log(1-r^2)
\geq -\frac12\log(1-r^2),
$$

with equality if and only if $Q=G_r$. This exact KL decomposition supplies
the Gaussian maximum-entropy argument without assuming that every
candidate differential entropy is initially well defined.

The global minimization consequently reduces to

$$
f_\lambda(r)=2-2r-\frac{\lambda}{2}\log(1-r^2),
\qquad -1<r<1.
$$

Its first two derivatives are

$$
f_\lambda'(r)=-2+\frac{\lambda r}{1-r^2},
\qquad
f_\lambda''(r)=\lambda\frac{1+r^2}{(1-r^2)^2}>0.
$$

The function diverges to $+\infty$ at both endpoints. It is strictly
convex and therefore has exactly one global minimum. Solving
$f_\lambda'(r)=0$ gives

$$
\boxed{
r_\lambda=\frac{\sqrt{\lambda^2+16}-\lambda}{4}
=\frac{4}{\sqrt{\lambda^2+16}+\lambda}.
}
$$

The unique global minimizing coupling is $Q^*_lambda=G_{r_\lambda}$.
Couplings of infinite KL have infinite objective for $\lambda>0$, so they
cannot improve this finite optimum.

Useful checks are

$$
\lambda=3\ \Longrightarrow\ r_\lambda=\frac12,
\qquad
\lim_{\lambda\downarrow0}r_\lambda=1,
\qquad
\lim_{\lambda\to\infty}r_\lambda=0.
$$

Every finite positive $\lambda$ gives $0<r_\lambda<1$. Exact zero residual
correlation is reached only in the infinite-regularization limit for this
objective. More generally, selecting a desired $r\in(0,1)$ requires

$$
\lambda=\frac{2(1-r^2)}{r}.
$$

At $\lambda=3$, the selected Gaussian joint coincides with the true
$\rho=1/2$ joint, but is wrong when the truth is $\rho=0$. The observable
information supplied to the loss is identical in these two cases. The
coincidence in one model cannot validate the chosen regularization in the
other, and no unpaired-data criterion can distinguish those two truths.

The selected $r_\lambda$ is an association in a completed distribution.
It is not by itself an identified intervention effect or edge direction.
Even access to a complete Gaussian joint does not generally orient all
causal edges without additional assumptions.

### Conditioning on state

In the main counterfamily, the conditional residual marginals are standard
normal for every state $s$. If the illustrative loss is minimized separately
over conditional residual couplings $Q_s$ with the same cost and positive
regularization, the pointwise solution is $Q_s=G_{r_\lambda}$ for almost
every $s$. Equivalently, minimizing its expectation over $P_S$ gives this
same solution almost everywhere. The construction therefore persists
despite exact state alignment.

This calculation is deliberately conditional. A method that may transport
between different states, imposes another cost, uses partial or unbalanced
transport, changes marginal masses, or includes other terms has a
different optimization problem. The displayed formula should not be
assigned to those methods without a derivation of their actual objective.

## 3. Exact cycle consistency need not choose the true pairing

Consider only standard normal residual marginals and two deterministic
maps, one in each direction. Both choices

$$
F_+(u)=u,\quad H_+(v)=v,
\qquad\text{and}\qquad
F_-(u)=-u,\quad H_-(v)=-v
$$

push the source standard normal law to the target standard normal law.
Both also obey

$$
H_\pm\circ F_\pm=\mathrm{id},\qquad
F_\pm\circ H_\pm=\mathrm{id}.
$$

Thus an objective consisting only of exact marginal-distribution alignment
and cycle-reconstruction errors assigns them the same optimal value. Yet
the induced couplings $(U,F_+(U))$ and $(U,F_-(U))$ have correlations $+1$
and $-1$ respectively.

The same example can preserve state explicitly: for a value $x$ at state
$s$, map it to $s+(x-s)$ or $s-(x-s)$, leave $s$ unchanged, and use the same
map for the inverse. Both respect the conditional $\mathcal N(s,1)$ law.
These deterministic completions are singular; they are illustrations of
loss ambiguity and are not claimed to belong to the nonsingular causal
counterfamily or the finite-KL transport solution above.

An extra term can break the sign symmetry. Squared residual distance, for
example, favors the positive map. Breaking the symmetry establishes which
map that added assumption prefers; it does not establish which map or
stochastic coupling generated the biology.

## 4. Limits on method-specific symmetry claims

The objective-invariance argument in Section 1 applies to any fixed
data-only integration pipeline satisfying its information assumptions.
The exact OT and cycle-map formulas apply only to their explicitly
defined illustrative losses.

For a reconstruction-only encoder/decoder pair, an invertible change of
latent coordinates can be canceled by its inverse in the decoder. This
fact alone does not show invariance of the full training objective.
Prior penalties, variational families, feature-guidance terms, distance
costs, and restrictions on decoders must each be checked.

In particular, arbitrary rotations need not preserve a diagonal-Gaussian
variational family. Independent coordinate changes in different assays
can also change cross-assay penalties, even where a common transformation
would preserve them. Any claim of an exact symmetry of scGLUE,
scConfluence, or another named implementation needs verification of all
its terms and its parameterization. The basic information theorem does
not depend on such an exact symmetry.

## Checks completed

The review independently derives the KL decomposition, the unique global
minimizer, its endpoint limits, the $\lambda=3$ example, and both
cycle-map identities. It also checks the information assumptions required
to connect population-objective equality to finite-sample output laws.
No empirical result was used as a premise of these proofs.

This review initially covers the proposed mathematical extension; review
of the actual methods-focused draft will be recorded below when that
draft is available.

## Actual draft review

Reviewed [Note 08](../../theory/08_modern_integration_objectives.md), snapshot
SHA256 `df524dfc350f59124b52fb4546d4f5890dc6494dc56388ace5e91dac850deed2`.
There is no blocking mathematical error in this snapshot.

The pointwise objective argument correctly separates identical losses
across causal worlds from nonunique optimization within a world. The
finite-sample passage retains equality in law and the independent-disjoint
sampling restriction. The exact KL decomposition justifies reducing the
transport optimization over all couplings to Gaussian candidates, and the
derivatives, solution, numerical example, and consequences for conditional
independence are correct. The conditional-state reflection preserves the
stated within-state distances and laws; the draft does not mistakenly
claim preservation of arbitrary geometry between different states.

Two small additions would make the transport proof more explicit: finite
KL excludes correlation endpoints $\pm1$, and the reduced objective tends
to $+\infty$ at both endpoints. Stating natural-log KL and the full
squared-cost convention also fixes the meaning of the numerical choice
$\lambda=3$. These details are derived above and do not alter the result.

The draft correctly labels the transport and reflection examples as
illustrative, qualifies the measured-PAG target, and does not use benchmark
failure as evidence for the theorem. The method-specific source formulas
and citations are assigned to the separate source reviewers; this
mathematical review does not claim independent verification of every
implementation reduction or published loss term.
