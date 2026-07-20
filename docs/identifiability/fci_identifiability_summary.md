# FCI Identifiability Summary for PI Discussion

## Setup

For each target gene

$$
g \in \{\mathrm{CSF1R}, \mathrm{CD14}, \mathrm{IL1B}\},
$$

we build a local graph over gene expression and nearby CUT&Tag peak variables.

Let

$$
V_g = \{Y_g\} \cup P_g,
$$

where $Y_g$ is the gene expression node and $P_g$ is the retained set of nearby mark-specific peak nodes. The observed data object is

$$
W_g = (W_v : v \in V_g),
$$

with rows coming from the scGLUE RNA-anchor pseudo-pairing / neighbor-averaging workflow.

In the default FCI setup, `libsize__...` columns are not graph nodes. They are part of preprocessing or count-model offsets, not the FCI target.

## Model Class

We do not assume causal sufficiency. Instead, we use a latent-variable local causal model:

$$
\mathcal{M}^{\mathrm{FCI}}_g
=
\{M:
M \text{ has an acyclic causal graph over } V_g \cup H_g
\text{ and induces } P(W_g)\}.
$$

Here $H_g$ represents unobserved common causes, such as transcription factors, genotype, donor state, cell-cycle state, batch, or unmeasured regulatory state.

## Assumptions

1. The pseudo-paired rows approximate a real selected-monocyte distribution:

$$
W_{ig} \sim P(W_g \mid \mathrm{monocyte}, \mathrm{QC}).
$$

2. The graph node set $V_g$ is fixed before FCI is run. If we filter peaks, the target becomes the graph over the retained peaks.

3. There is an underlying acyclic causal graph over $V_g \cup H_g$.

4. After marginalizing hidden variables, the observed distribution is Markov and faithful to an induced MAG $M_g$ over $V_g$:

$$
X \perp Y \mid Z \text{ in } P(W_g)
\iff
X \perp_m Y \mid Z \text{ in } M_g.
$$

5. Population FCI gets the correct conditional-independence information. In finite samples, this means the CI test has to be reliable enough for the available effective sample size.

6. The clean FCI theorem assumes the full population FCI search. If we cap conditioning depth or Possible-D-Sep for runtime, that result is a bounded approximation, not the exact theorem.

## Identifiability Claim

Under these assumptions, FCI identifies the PAG equivalence class for the local observed variables:

$$
P(W_g)
\implies
[M_g]_{\mathrm{PAG}}.
$$

Equivalently, if two models produce the same observed distribution, they agree on the PAG:

$$
P_{M_1}(W_g) = P_{M_2}(W_g)
\implies
[M_1]_{\mathrm{PAG}} = [M_2]_{\mathrm{PAG}}.
$$

The fully resolved local graph is identifiable only for features that are invariant across the PAG:

$$
T_g = \mathrm{Inv}([M_g]_{\mathrm{PAG}}).
$$

So the honest target is not "we identify every edge direction." It is:

> We identify the local PAG, and we report edge directions only where the PAG or additional valid evidence supports them.

## Proof Sketch

1. The integration step gives us an approximate observed distribution $P(W_g)$ over a fixed local node set $V_g$.
2. We allow hidden common causes by assuming an underlying DAG over $V_g \cup H_g$.
3. Marginalizing $H_g$ induces a MAG $M_g$ over the observed variables.
4. Markov plus faithfulness lets conditional independences in $P(W_g)$ correspond to $m$-separations in $M_g$.
5. Population FCI is sound and complete for recovering the PAG of $M_g$ from those independences.
6. Therefore the PAG and its invariant features are identifiable from $P(W_g)$.

## Practical Interpretation

The main result we can defend is:

$$
\widehat{[M_g]}_{\mathrm{PAG}}
\approx
[M_g]_{\mathrm{PAG}},
$$

provided the effective sample size is large enough, the graph is sparse enough, and the CI tests are stable. For the current neighbor-averaged matrices, the nominal row count is $n = 1{,}887$, but the effective sample size may be smaller because CUT&Tag neighbors can be reused.

Literature-supported promoters/enhancers help validate and prioritize edges. They do not make an edge identifiable by themselves.

The backup reporting target is stable oriented edges across reasonable settings:

$$
T_{\mathrm{stable}}(g)
=
\{e:
e \text{ appears with the same orientation across accepted runs}\}.
$$

That is a robustness claim, not a pure identifiability theorem.
