# Testing causal information after integration

_Discussion draft · 2026-09-23 · nonlinear simulation options and a proposed first experiment_

---

## The question to test

Before comparing aggregation methods, test whether the integration preserves
information about a known region–RNA network. The first experiment should use
RNA and one of the existing CUT&Tag marks, with a small nonlinear biological
network and realistic count measurements. Additional marks can follow once the
two-assay result is understood.

There are two questions to measure separately:

1. **Information retention:** after fitting integration to unpaired cells, can
   its representations still describe the original biological variables and
   their dependencies on an independent evaluation set?
2. **Unpaired graph recovery:** can a specified procedure using only permitted
   unpaired inputs infer region–RNA edges and their identifiable orientations?

The first is a useful prerequisite for the second. The simulation should expose
where information is lost: measurement, preprocessing, compression, alignment,
or the eventual construction of a joint feature matrix. The interpretation
rules are owned by [LIMITATIONS L18](../../../project_resources/research/LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution).

Current scope and execution status are in the
[September 23 roadmap entry](../../../project_resources/research/ROADMAP.md#integration-recoverability-design-2026-09-23).
The calculations below are proposed models, not measured benchmark results.

## Confirmed py-tetrad options

The Python convenience functions expose only part of Tetrad. The installed Java
library exposes additional simulators through JPype. The table combines helper
source inspection with successful class loading and constructor/description
checks for the continuous and mixed candidates in the `causaldisco` environment
on September 23. No datasets were generated during this inventory.[^tetrad]

| Option | What it provides | Proposed role |
| --- | --- | --- |
| `simulateSemSimulation` / `SemSimulation` | Linear structural equations | Simple positive control |
| `GeneralSemSimulation` + `GeneralizedSemPm` | Explicit equations and error expressions for named nodes | Preferred starting point for interpretable nonlinear motifs |
| `AdditiveNoiseSimulation` | Nonlinear neural functions of parents, with additive noise | Generic nonlinear comparison[^additive] |
| `simulateGeneralNoise` / `GeneralNoiseSimulation` | Neural structural mechanisms with a more general noise construction | Harder nonlinear comparison[^general] |
| `GpSemSimulation` | Nonlinear Gaussian-process-style SEM | Small secondary comparison; source allocates cell-by-cell kernel matrices[^gp] |
| `PostnonlinearSem` / `NonlinearFunctionsOfLinear` | Further nonlinear SEM/transformation families | Later robustness checks, after auditing their exact mechanisms |
| `simulateDiscrete` / `BayesNetSimulation` | Categorical Bayesian networks | Discrete controls; categories are not sequencing counts |
| `simulateLeeHastie` / `LeeHastieSimulation` | Mixed continuous/categorical data | Mixed-data controls; not an RNA/CUT&Tag count model |

For the first experiment, prefer explicit saturating equations over a random
large neural network: an edge should have an understandable effect, and its
strength should be easy to vary. Tetrad's generalized model supports custom
nonlinear functions and noise distributions.[^generalized]

There is no ready-made RNA plus histone CUT&Tag observation model among these
reviewed convenience functions. A useful division of work is Tetrad for the
structural graph and biological variables, followed by an explicit count
sampler. Alternatively, implement the same small equations directly in Python
and use Tetrad for graph conversion and discovery. The equations, random streams,
and measurement assumptions should be identical in either implementation.

### What is available locally

The installed package reports `py-tetrad 0.1.2`; its bundled JAR reports
`7.6.11-SNAPSHOT`. The separate checkout at
`/home/dgsetiawan/MachineLearning/CLeaR/py-tetrad` is at commit
`b967024e14e4918d282130b65ee39e6936f6a9e5`. Its JAR differs from the installed
one, so the checkout and environment must not be treated as interchangeable.

The newer checkout additionally contains `simulateDesignedExperiment` and
`simulateObservationalStudy` wrappers. They are absent from the installed
Python helper inspected here and are unnecessary for the first experiment.
The installed `GeneralSemSimulation` accepts a `GeneralizedSemPm`, a
`GeneralizedSemIm`, or a graph generator. `GeneralizedSemPm.setNodeExpression`
and its error/parameter expression setters are available.

Record the chosen JAR, not just the Python version:

```text
Installed JAR SHA-256:
e2073ae80f6bb2fe33e531153575427e3570f2695dd71cbe558d4c086d8adc9e

Separate checkout JAR SHA-256:
ed4b4e3da97c3bd0b1d1685d992b2116c65f10dd5800f273a16a24d90bff3d36
```

## Simulate biology first, then assay counts

Start by drawing a complete synthetic cell, including its unobserved biological
state, region activities, and RNA abundance. Generate each downstream biological
variable from its parents in a known acyclic graph. Only then sample the assay
counts from those activities. A fragment count and an RNA read count are noisy
observations of this biology.

Use a small set of motifs:

| Motif | What it tests |
| --- | --- |
| Region 1 → region 2 → RNA | Direct versus mediated regulation |
| State → region and state → RNA | A common driver without a direct region–RNA edge |
| Region 1 → RNA ← region 2 | A collider and multi-parent dependence |
| Independent biological region | False-edge control, with technical depth modeled separately |

These can be separate small configurations rather than one complicated network.
Use smooth saturation first, then add repression and one cooperative interaction.
A sigmoid or Hill response models a plateau; an interaction models two regions
whose combined effect differs from the sum of their effects. These are chosen
simulation mechanisms, not assertions that every histone mark directly drives
transcription. Retain a linear version with comparable signal as a diagnostic.

### Count distributions

Use a **negative binomial** measurement model for RNA initially, with adjustable
mean, dispersion, and cell-specific sequencing depth. For CUT&Tag, start with
sparse **Poisson counts**, then use a negative binomial alternative if extra
variation is needed to match the existing data. Choose parameters from existing
assay summaries: detection rates, nonzero count magnitudes, depth variation, and
mean–variance relationships. Do not import a new modality to calibrate them.

Keep integer fragment counts as the primary input. A separate detection-only
arm can threshold these counts; a Bernoulli observation is a detection indicator,
not a general raw-count model. Add explicit zero inflation only if the simpler
count model fails relevant checks. Freeze parameters within a simulation world
before drawing training and assessment cohorts.

Add background genes and regions that report biological state, along with
assay-specific and null features. This gives integration enough context to work
without supplying the focal causal graph. Vary how well the available background
features measure state. Also vary biological noise independently of sequencing
noise; the two have different consequences.

<details>
<summary><strong>Optional equations for a first nonlinear world</strong></summary>

Let $S_i\sim N(0,I_2)$ be a two-dimensional biological state and let the
$\epsilon$ terms be independent standard-normal biological innovations,
independent of $S_i$. One small model on a signed activity scale is

$$
\begin{aligned}
A_{i1} &= a^\top S_i + \sigma_1\epsilon_{i1},\\
A_{i2} &= b\tanh(A_{i1}) + \sigma_2\epsilon_{i2},\\
G_i &= c\tanh(A_{i2}) + d^\top S_i + \sigma_G\epsilon_{iG},\\
D_i &= q^\top S_i + \sigma_D\epsilon_{iD},\\
N_i &= \sigma_N\epsilon_{iN}.
\end{aligned}
$$

$A_1,A_2$ form the regulatory chain, $G$ is RNA activity, $D$ is a
state-associated region with no direct edge to $G$, and $N$ is a biological
null. The term $\tanh$ makes the response saturate. The direct region effects
also transmit cell-specific biological variation when the $\sigma$ values are
nonzero. Keep them nonzero for the ordinary causal-discovery controls;
deterministic limits require separate treatment.

For an RNA feature $j$, define

$$
\mu_{ij}^{R}
= L_i^{R}\operatorname{softplus}(\alpha_j+\kappa_j G_{ij}),
\qquad
C_{ij}^{R}\mid G_{ij},L_i^R
\sim \operatorname{NB}(\mu_{ij}^{R},\theta_j),
$$

where $\operatorname{softplus}(x)=\log(1+e^x)$, $L_i^R>0$ is an exposure factor,
and this negative binomial parameterization has variance

$$
\operatorname{Var}(C_{ij}^{R}\mid G_{ij},L_i^R)
=\mu_{ij}^{R}+(\mu_{ij}^{R})^2/\theta_j.
$$

For a chromatin feature $k$, use

$$
\lambda_{ik}^{C}
=L_i^{C}\operatorname{softplus}(\gamma_k+\eta_k A_{ik}),
\qquad
C_{ik}^{C}\mid A_{ik},L_i^C
\sim\operatorname{Poisson}(\lambda_{ik}^{C}).
$$

An optional detection observation is

$$
B_{ik}=\mathbf{1}\{C_{ik}^{C}>0\},
\qquad
\Pr(B_{ik}=1\mid A_{ik},L_i^C)=1-e^{-\lambda_{ik}^{C}}.
$$

Draw technical exposures independently across assays and independently of biology
in the first control. Later introduce state-associated or correlated assay
depth as separate controls. $L$ is a generative exposure, not necessarily
the realized count total; discovery uses the same observable depth proxy as the
real workflow, with true exposure reserved for an oracle diagnostic.

For a cooperative mechanism on positive activities, a candidate response is
$h(A_1)h(A_2)$, where $h(a)=a^n/(K^n+a^n)$ for $a\geq0$. This introduces a
specific interaction rather than making every equation arbitrarily complex.

An alternative **conditional count DAG** samples each count directly from a
Poisson or NB conditional distribution whose mean is a nonlinear function of
parent counts. Under the stipulated independent-noise model, this gives a DAG
over the count variables themselves. It is a useful algorithmic control but
has different biological semantics from activity followed by measurement.

</details>

---

## Integration inputs must match the actual workflows

The [scGLUE workflow](../../empirical/integration/methods/scglue/README.md) uses RNA
PCA and chromatin LSI representations together with count reconstruction and
a feature guidance graph. Reproduce those preprocessing stages, including
feature selection. The upstream training tutorial documents these inputs.[^glue]

Generate synthetic genomic locations and proximity-based candidate links without
copying the planted causal edge list. Keep noncausal nearby candidates and score
supplied versus withheld causal links separately. Include a baseline using only
the guidance graph and a shuffled-guidance control. A background-only integration
arm should hold focal features out of PCA/LSI, reconstruction, guidance, and any
gene-score construction. Compare it with focal-inclusive integration as a
separately labeled input contract.

For [the local scMRDR-style workflow](../../empirical/integration/methods/scmrdr/README.md),
name the exact implementation, feature mode, reconstruction likelihood, and
shared/private embedding output. The repository supports native assay features
and an optional shared gene-space path; the default native-feature path does
not become the authors' implementation merely by sharing the method name.
The detailed differences are recorded in the
[method comparison](../../empirical/integration/results/jianle_scMRDR_method_and_repo_comparison.md).
The authors' package is a separate possible arm; its example uses RNA and
chromatin gene activity in an aligned feature universe with optional masks.[^mrdr]

For gene-space input, derive region-to-gene scores from the allowed synthetic
annotation/proximity rules. Keep original region counts for evaluation, and
measure the information lost during that conversion before attributing it to
integration. Keep private embeddings as a diagnostic when the shared embedding
discards information.

## Evaluate integration before averaging cells

Draw independent training cohorts from one fixed biological model. RNA training
cells expose only RNA; chromatin training cells expose only chromatin. Their
cell IDs are disjoint. Freeze the graph, mechanisms, feature annotations, and
count parameters across cohorts; never regenerate a different mechanism for
each assay. Split cohorts before fitting feature selection or preprocessing.

Reserve a separate complete synthetic evaluation cohort. Encode its two views
independently with the frozen models. The evaluator retains the hidden pair IDs,
activities, and states; the integration method receives none of those labels.
This is a simulation scoring device, not a request for paired experimental data.
Exact partner retrieval and biological-state correspondence should be scored
separately, since several cells may be indistinguishable at the available depth.

| Evaluation stage | Reference comparison | Purpose |
| --- | --- | --- |
| Complete biological activities | Known activity graph | Check the learner under its stated causal assumptions |
| Complete paired counts | Same underlying cells | Measure the effect of count observation |
| PCA/LSI or gene-space features | Paired counts | Locate preprocessing losses |
| Frozen integration embeddings | Same raw-feature diagnostic | Measure state and feature information retained |
| Original-feature reconstructions on hidden paired cells | Paired-count and activity references | Diagnose information after encoding/decoding |
| One fixed unpaired reconstruction or matching procedure | The preceding references | Test the deployable integration-plus-readout pipeline |

The first five stages use no pseudobulk averaging. The final stage is a later,
explicit readout choice, held fixed across integration methods before exploring
aggregation alternatives. Native scGLUE feature-link scores may be included as
an undirected association baseline; its regulatory-inference tutorial describes
those scores, which are a different output from a cell-level causal graph.[^reginf]

For the activity reference, distinguish an oracle analysis that receives true
$S$ from an analysis with $S$ hidden. The latter targets the appropriate latent
projection/equivalence class over observed biological variables, rather than the
complete DAG containing $S$. These controls are evaluator-only references.

Stage differences describe performance of the specified learner or readout.
A failed probe or graph search does not establish that all methods would fail
or that information has been irreversibly lost. Integration may improve
performance through denoising; paired raw-count performance is a reference,
not a universal upper bound.

Use original region and gene identities for graph scoring. For information
diagnostics, fit equally flexible probes on a separate evaluator-only calibration
split and assess them on a disjoint split. Probes can predict biological
activities, state, or target features from each representation. Their access to
simulation labels is explicitly an oracle diagnostic. A prediction trained with
hidden cross-assay targets is not an unpaired discovery method.

Report state reconstruction and feature prediction, including prediction of
deviations from the true state-dependent means. Separately report graph adjacency
precision/recall, null false-positive rate, justified CPDAG/PAG endpoints, and
exact directed edges as a stress test. Evaluate shared-state hyperedges only if
the method returns that object; do not substitute them for direct edges.

## Controls, interpretation, and a small pilot

Predeclare a short set of conditional-independence questions from the generating
model, alongside full graph recovery. For example, the activity chain should
satisfy its stipulated mediator separation when all required conditioning
variables are observed. In the proposed model, the oracle separation is
$A_1\perp G\mid(A_2,S)$; conditioning on $A_2$ alone leaves the shared-state
path open. Use nonlinear tests on the nonlinear activity control;
calibrate their false positives and power on the count regime before relying on
their results. A linear Fisher-Z test alone does not assess arbitrary nonlinear
dependence.[^tests]

Derive any count-level graphical reference from the full measurement model.
Keep activity-graph scores and observed-count graph scores distinct. Follow the
measurement and embedding rules in
[L18](../../../project_resources/research/LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution)
when interpreting those comparisons.

Start with a high-signal nonlinear configuration and its linear counterpart.
Then change one factor at a time: chromatin detection, biological residual
variation, alignment guidance, or population composition. Include state-only
common-cause pairs, biological nulls, random cross-assay matching, and an analytic
ambiguity control whose state-conditioned assay distributions stay fixed while
its unseen residual coupling changes. The ambiguity control evaluates false
certainty rather than expecting universal edge recovery. Oracle-state matching
is a state-alignment reference; it is not an oracle for the residual joint law.

An illustrative development pilot is two assays, 4–8 focal biological variables,
500–1,000 background features per assay, and 2,000–5,000 training cells per assay,
plus an independent evaluation cohort. Begin with sufficient detection to make
the positive control informative, then impose the existing data's sparsity.
These sizes are proposals, not validated minimums or a locked configuration.

Start with scGLUE, the explicitly labeled local scMRDR-style model, and an
unintegrated preprocessing baseline. Reuse identical frozen counts across
methods, with a few integration seeds. Cache inputs and embeddings. Separate
integration-seed variability from variability across biological mechanisms;
multiple integration seeds in one world do not establish reliable recovery
across worlds. Reuse the frozen development panel for fast iterations, and
reserve an assessment configuration before reporting reliability.

The first useful result is a stage-by-stage diagnosis: whether structure is
detectable before measurement, after counts, after preprocessing, and after
integration. It need not be a leaderboard of large networks.

## Existing components and published alternatives

The [earlier pilot](../unpaired_multimodal/scripts/multimodal_simulation_pilot.py)
already implements RNA negative-binomial sampling, sparse chromatin detection,
PCA/LSI preparation, and scGLUE fitting. The
[count simulator](../aggregation_identifiability/scripts/crossmark_count_simulator.py)
provides detection/depth calibration. Reuse these as inspected components, not
as an unchanged benchmark: the earlier RNA sampler couples feature means through
a row-wise softmax, and its chromatin routine forces a background detection in
all-zero rows. Explicitly model compositional sampling as its own arm, and report
all-zero filtering rather than silently inserting observations in the new pilot.

**scMultiSim** is a relevant later realism comparison: it simulates linked RNA
and chromatin accessibility with a supplied regulatory network and technical
variation.[^multisim] Its accessibility model is not already a histone CUT&Tag
model. It can inform our design without introducing ATAC, spatial information,
RNA velocity, or additional experimental information as inference inputs.
For the first experiment, explicit Tetrad equations keep the graph and failure
mechanisms easier to inspect.

Independent methodological review informed the separation of biological and
measurement truth, hidden paired evaluation, original-feature graph scoring,
and the guidance controls. No performance claim follows from this design note.

## Sources

[^tetrad]: CMU Philosophy. py-tetrad source and Python/Java interface. https://github.com/cmu-phil/py-tetrad . Local Python helpers, JAR contents, and Java class metadata inspected 2026-09-23; runtime inventory is described above.
[^additive]: CMU Philosophy. `AdditiveNoiseSimulation` source. https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/algcomparison/simulation/AdditiveNoiseSimulation.java
[^general]: CMU Philosophy. `GeneralNoiseSimulation` source. https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/algcomparison/simulation/GeneralNoiseSimulation.java
[^gp]: CMU Philosophy. `GpSemSimulation` source. https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/algcomparison/simulation/GpSemSimulation.java
[^generalized]: Tetrad manual. Generalized parametric models. https://tetrad-manual.readthedocs.io/en/latest/tetrad-interface/detail-callouts/detail-generalized-parametric-model.html
[^glue]: scGLUE documentation. Stage 2: Model training. https://scglue.readthedocs.io/en/latest/training.html
[^mrdr]: Sun et al. scMRDR official implementation and input examples. https://github.com/sjl-sjtu/scMRDR
[^reginf]: scGLUE documentation. Stage 3: Regulatory inference. https://scglue.readthedocs.io/en/latest/reginf.html
[^tests]: Tetrad manual. Tests and scores by type. https://tetrad-manual.readthedocs.io/en/latest/tests-and-scores-catalog.html
[^multisim]: Zhang Lab. scMultiSim documentation. https://zhanglabgt.github.io/scMultiSim/
