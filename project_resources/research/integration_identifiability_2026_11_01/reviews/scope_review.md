# Skeptical review of the unpaired-integration impossibility claim

**Review conclusion:** the proposed construction supports nonidentifiability of
specified residual conditional independences, direct biological edges and
effects in a broad model class. It does not support a claim that all structure,
every joint distribution under every assumption, or every particular dataset
is unrecoverable. No new measurements, simulations, model fits or modifications
to frozen experiments were used for this review.

## State the observation experiment and target first

Let an assay observe an entire vector, not just one selected feature. A useful
population-level observation object is

$$
\mathcal O(P)=\left(A,\{P(S,O_a,D_a):a=1,\ldots,M\}\right),
$$

where $A$ is fixed permitted annotation/guidance, $O_a$ contains all features
measured together in assay $a$, and $D_a$ denotes permitted depth or technical
metadata. Giving the learner the exact common state $S$ is an oracle
strengthening of the actual learning problem. It removes state-estimation and
coordinate-alignment errors from the argument.

The sampling claim should assume independent, disjoint cells drawn from a
common population within each world, with assay assignment independent of
their unobserved cross-assay residuals. Two candidate worlds can have different
full joint distributions while each separately satisfies this same-population
assumption. Assuming that a common biological distribution exists does not
specify its missing cross-assay coupling.

For that experiment, equality of every complete assay-block law makes the
finite-sample learner-data law identical in both worlds. An integration,
matching, aggregation or discovery procedure that uses only these observations,
the same annotation and independent algorithmic randomness then has the same
output law. Learned embeddings and synthetic pairings cannot distinguish the
two worlds. If the targets differ, one procedure cannot be consistently correct
at both worlds; a uniform guarantee over a class containing them is impossible.

This statement must include any other world-dependent information actually
available to the estimator. A known quantitative mechanism, a genuinely
informative common residual proxy, or feedback from paired evaluation is not
automatically covered by calling it an integration prior.

## What the proposed example establishes

For independent standard Gaussian $S,U,V$, let

$$
X=S+U,\qquad
Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}\,V,
\qquad 0\leq\rho<1.
$$

Both separately observable conditionals remain

$$
X\mid S=s\sim\mathcal N(s,1),\qquad
Y\mid S=s\sim\mathcal N(s,1),
$$

while the missing conditional covariance is

$$
\operatorname{Cov}(X,Y\mid S)=\rho.
$$

The conditional mean under intervention is

$$
\mathbb E[Y\mid\operatorname{do}(X=x),S=s]
=(1-\rho)s+\rho x.
$$

Thus residual independence, direct-edge presence and the causal coefficient
differ between $\rho=0$ and a fixed interior value such as $\rho=1/2$.
The direct and indirect state contributions are both positive for an interior
value; the equality of the separately observable means is not cancellation
creating an extra conditional independence in the triangle DAG.

The shared-state quantity

$$
\operatorname{Cov}\{\mathbb E[X\mid S],\mathbb E[Y\mid S]\}
=\operatorname{Var}(S)=1
$$

is nevertheless identified when the common state is given. The example therefore
supports separating state-associated structure from unidentified residual
structure.

**Graph-target qualification:** after hiding $S$ and retaining only $X,Y$,
both worlds have dependent measured variables. This example alone does not
give different two-node measured-variable PAGs. Use it for the stated direct
biological edge/effect or state-augmented CI target. A separate example with
$X=U$ and $Y=\rho U+\sqrt{1-\rho^2}V$ gives empty versus adjacent measured
two-variable graphs while state-dependent background blocks remain fixed.
That secondary example belongs to a class allowing focal features without a
state loading. Neither example identifies a unique causal direction merely
from a joint observational distribution.

## Compatibility with the RNA/ac/me3 information contract

**Complete within-assay blocks.** Assign $X$ to one mark, $Y$ to RNA and append
a third-assay variable $Z=S+W$, with an independent positive-variance Gaussian
innovation. Add any finite number of informative state backgrounds
$G_j=a_jS+\epsilon_j$, with positive coefficients and independent
positive-variance noise, distributed across the assays. The law of each entire
assay block is unchanged across $\rho$. Thus a large number of background
features or additional modalities does not invalidate this counterexample.
This is an existence argument; some other background designs can supply
identifying restrictions.

There is also a route to an explicitly faithful finite extension: conditioning
on a subset of background children leaves a positive state variance $t$.
Then $\operatorname{Cov}(X,Y\mid G)=t+\rho$, while
$\operatorname{Cov}(S,X\mid Y,G)=t(1-\rho)/(t+1)>0$ and the analogous
$S,Y$ covariance is positive. Remaining background-pair covariances are their
loading product times a positive posterior state variance unless $S$ is
conditioned upon. These facts allow the finite star/fork versus star/triangle
construction to be checked without asserting that arbitrary appended
measurement kernels preserve faithfulness.

More generally, a fixed assay-local kernel
$K_a(B_a,D_a\mid S,X_a)$ preserves its complete observable block law when the
law of $(S,X_a)$ is fixed. Such kernels can include within-assay dependence,
local count generation and derived library summaries. Faithfulness of a
particular augmented causal graph requires a separate check; equality of
observable laws does not.

**Fixed annotations.** Gene identities, genomic positions, promoter definitions
and a list of permissible peak–gene links can be held identical. A possible
positive regulatory link is compatible with a shared-state association without
a direct effect. If a prior instead fixes the causal edge, coefficient, residual
copula or exclusion restriction as truth, it narrows the model class and may
exclude one world. That is an assumption supplying identification, not evidence
that the counterexample distinguishes itself in the assay data. If an edge is
assumed present, two distinct interior $\rho$ values still demonstrate effect
ambiguity, but they do not demonstrate ambiguity of that edge's presence.

**State sufficiency.** Exact access to a specified common state is not the same
as assuming that it contains every cell-specific innovation. If “perfect state”
is defined to determine all assay values, the stochastic-residual example is
outside that restricted class. An encoder's existence or modality mixing does
not establish that stronger sufficient-state assumption.

**Depth and QC.** Identical assay-local observable laws remain identical after
the same local QC rule, provided retained counts and other disclosed selection
metadata are included. Joint source filtering on several modalities requires
care: changing a cross-assay coupling can change a jointly selected population.
The theorem should specify its source/post-selection population and should not
claim to reproduce every detail of scMTR's inherited six-target DNA filter.
It is not a theorem fitted to the exact S1 empirical distribution.

## Counts and nonlinear mechanisms

Using the same stochastic measurement kernel in each world preserves equality
of the learner-visible assay laws. For an explicit count example, fix
positive exposures $L_X,L_Y$ and generate independent Poisson measurements with
means $L_Xe^X$ and $L_Ye^Y$ given the biological activities. Their separate
conditional count distributions are identical across $\rho$, but

$$
\operatorname{Cov}(C_X,C_Y\mid S=s,L_X,L_Y)
=L_XL_Y e^{2s+1}(e^\rho-1).
$$

Consequently, the hidden count joint can differ too. The same conditional-mean
calculation applies to independent negative-binomial measurement channels with
fixed dispersion. This does not assert that every measurement kernel preserves
every distinction, or that intervening on a read count is the same intervention
as changing the underlying biological activity.

A linear example is sufficient to refute a universal guarantee in any model
class containing it. Componentwise invertible nonlinear transformations also
preserve the indistinguishability and conditional-independence argument while
producing nonlinear structural equations. They do not establish impossibility
for every separately specified nonlinear additive-noise or known-noise class.

## Independent unpaired cells versus anonymized same-cell samples

This distinction is necessary for applying the proof to the project.

- The disjoint scMTR training pools emulate the independent unpaired setting.
- The three heldout scMTR views contain the **same 595 cells**, independently
  reordered and given opaque IDs. Removing IDs does not turn their sampling
  law into three independent iid samples. Unordered samples arising from the
  same underlying pairs can retain joint finite-sample information.
- Canonical pairing, day labels and paired panel values are evaluator-only in
  the frozen pipeline. They validate performance under that experimental
  protocol; they cannot be silently added to a theorem claiming that the
  estimator sees only independent unpaired observations.
- A statement about a learner trained on disjoint cells is different from a
  statement about a transductive estimator also permitted to exploit all the
  unlinked same-cell test arrays. The latter needs its own observation model.
- Tuning a learner using paired evaluation outcomes also changes its effective
  information. Keep any theorem's observation contract explicit.

These distinctions do not undermine the independent-disjoint-sampling
counterexample. They prevent extending its exact finite-sample equality claim
to a different experiment without proof.

## Restricted positive cases that the note should retain

| Restriction | What it can identify | Qualification |
| --- | --- | --- |
| Exact common state and a specified conditional copula, including conditional independence of assay blocks | The full statistical joint from the separate conditional laws | The missing copula is supplied as a modeling restriction; generic causal orientation still needs its own assumptions. |
| Deterministic assay functions of a sufficient common state | Their same-state joint, up to the observed state's support | Measurement-noise coupling must also be specified when counts remain stochastic; deterministic relations can violate ordinary CI/faithfulness regularity. |
| Known parent/exclusion structure, an invariant structural equation and sufficient variation of parent conditional means | Structural coefficients in the restricted equation | A direct unmodeled effect of state violates the exclusion. Full-rank variation is essential. |
| A fixed-direction monotone regression function with known independent noise and identifiable deconvolution | The regression mechanism on the covariate support | This is a narrower class, not unrestricted causal discovery across arbitrary modality vectors. |

For the third row, suppose the causal equation is

$$
Y=a+\beta^\top X+\epsilon,\qquad
\mathbb E[\epsilon\mid S]=0,
$$

with no additional direct state term and the same coefficients over the state
support. Separate assays can identify the conditional-mean functions
$m_X(s)$ and $m_Y(s)$, hence

$$
m_Y(s)=a+\beta^\top m_X(s).
$$

If the centered parent-mean functions have full-rank covariance under the
common state distribution, this equation identifies $\beta$. The negative
example deliberately allows a direct state coefficient to vary; it does not
satisfy this exclusion restriction. Causal interpretation relies on the
structural/exclusion assumptions, not on conditional-mean regression alone.

For the last row, an elementary sufficient case takes scalar continuous $X$,
an increasing $m$, and independent known noise with a nonvanishing
characteristic function. Deconvolution identifies the distribution of $m(X)$;
monotonicity then identifies $m$ through quantiles, up to the usual support and
almost-everywhere qualifications. Published unlinked-regression work establishes
estimation results under monotonicity and known-noise assumptions.
[Balabdaoui, Doss and Durot, 2021](https://www.jmlr.org/papers/v22/20-689.html).
Multivariate positive results also exist under cyclical monotonicity and the
paper's specified noise conditions; ordinary optimal-transport alignment does
not establish that those identifying conditions hold biologically.
[Slawski and Sen, 2024](https://jmlr.org/papers/v25/22-0058.html).

Known structural noise in these examples is stronger than knowing that an
assay's observation channel is Poisson or negative binomial. An assumed
direction or mechanism is also different from discovering an arbitrary DAG.

## Claims to avoid

- “No cross-assay structure is recoverable.” Shared-state associations and
  permitted within-assay relationships can be identifiable.
- “This proves the graph is unidentifiable for every possible dataset.” A
  finite pair of indistinguishable worlds rules out a guarantee over a class
  containing them; it does not prove ambiguity at every observable law.
- “All nonlinear causal models are covered.” State the model class, including
  whether linear submodels, unknown structural noise and direct state paths
  are allowed.
- “Every CI test must fail after integration.” Some targets are within-assay
  or identified under restrictions. The missing-joint argument concerns
  unidentified cross-assay queries; the observed sparse-count test calibration
  failure is a separate finite-data/algorithmic issue.
- “The 48/50 artificial-null result proves impossibility.” It diagnoses one
  fixed readout on one panel. The mathematical nonidentifiability proof does
  not depend on that empirical result.
- “Recovering the joint would automatically recover causal directions.”
  Ordinary observational causal equivalence can remain even with a joint.
- “More modalities or better integration can never help.” They can improve
  state estimation or provide identifying restrictions in a narrower model.
  They do not supply a uniform guarantee over the stated counterexample class.

Recommended main-note conclusion: **With independent unpaired assay samples
and the stated broad structural model class, even exact knowledge of the common
state and every assay's conditional distribution does not identify certain
cross-assay residual relationships, direct edges or effects. Integration can
recover identifiable shared-state information, but selecting a missing
cross-assay coupling requires additional model restrictions. This conclusion
is a limit on a general guarantee, not a claim that every useful or restricted
causal target is impossible.**

## Review of the actual main draft

Reviewed [07_unpaired_integration_nonidentifiability.md](../../theory/07_unpaired_integration_nonidentifiability.md)
after the main proof and exceptions were written. **Scoped approval: no
blocking mathematical or scope error found in this draft.**

The draft correctly states the independent-disjoint sampling experiment,
gives the complete-state-law counterexample and two-model error bound, and
separates direct biological targets from measured PAG adjacency. It explicitly
qualifies nonlinear/count extensions, distinguishes shuffled same-cell
scMTR holdouts from the theorem's data experiment, retains valid restricted
positive cases, and separates empirical calibration failures from the proof.
The existing two-variable PAG and full-joint direction qualifications address
the most important potential overclaims.

One useful optional clarification in Section 4 or 8 is that assay-local QC is
covered by identical local kernels, whereas a jointly defined source filter
such as the inherited six-target scMTR threshold is not automatically covered
by that argument. The draft already states that the actual scMTR experiment is
not its strict-disjoint observation model and links the source-data record, so
this omission does not invalidate its stated theorem. No model fit or source
data were used to assess the theorem.
