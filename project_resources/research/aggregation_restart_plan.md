# Aggregation-first causal discovery: proposed restart

_Started 2026-09-10 · updated 2026-09-23 · discussion draft, not a locked benchmark protocol_

The immediate precursor is now an
[integration recoverability simulation](../../pipelines/simulation/integration_recoverability/design.md),
following the [September 23 priority change](ROADMAP.md#integration-recoverability-design-2026-09-23).
It tests what scGLUE and the scMRDR-style workflow preserve before comparing
cell-aggregation operators. The stages below remain the subsequent aggregation plan.

The mathematical foundation is now permanent in
[latent-space aggregation across unpaired assays](theory/06_latent_space_aggregation.md).
That note distinguishes state alignment, covariance completion, equivalence-class
recovery and exact direction recovery. The operational claim boundary is owned by
[LIMITATIONS L18](LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution).

## Research tracks

Track A preserves the CA-BOSS disjoint-assay latent-state research, including
code, cached simulations, source-bound outputs, failed candidates and reviews.
Its strongest previously audited checkpoint remains v4.59; shared-state
recovery is not equivalent to recovery of direct cell-level regulation. This
track is parked in favor of the aggregation question. No files are relocated,
deleted or represented as a new immutable archive by this decision.

Track B asks: under what sampling, grouping and measurement assumptions does
aggregation preserve causal structure, and which aggregation/learning methods
best retain that structure at a fixed data budget? Start with two assays and a
small explicit SEM, then reproduce the actual integration and aggregation
operators. Do not extend the six-mark simulator as the starting point.

This turn performs planning and read-only review only apart from documentation.
It does not generate simulations or launch benchmarks. The former existing-
world-only constraint applies to Track A; the user now proposes a new simple
simulation for Track B, whose configuration remains to be agreed.

## What the existing work provides

The [advisor simulation overview](theory/04_simulation.md) describes a
two-assay `paircheck` generator: a linear SEM, shared noisy anchor variables,
and true/matched/permuted pairing arms. The original four modules and
`dylan.zip` were not found in the inspected workspace. Attribution to Joe
and the intended version need confirmation before claiming reproduction.
The [multichannel extension](../../pipelines/simulation/aggregation_identifiability/scripts/paircheck_multichannel.py)
is not self-contained because its base package imports are missing.

The [older small simulation](../../pipelines/simulation/aggregation_identifiability/scripts/simulate_aggregation_identifiability.py)
averages jointly observed variables within k-means clusters of exact latent
coordinates. It is a useful paired-oracle control, not the main unpaired
experiment. Its fixed-row experiment does not establish an infinite-sample
identifiability ceiling. Its `distinguish` metric checks two causes and two
state-confounded decoys, but not both additional null peaks despite historical
prose; orientations use a prescribed peak-to-gene tier. These claims and
definitions must not become new benchmark assumptions.

The [actual integration workflow](../../pipelines/empirical/integration/README.md) uses scGLUE
embeddings. Existing aggregation alternatives include one-to-one RNA anchoring,
overlapping k-nearest-neighbor chromatin averaging, and
[Leiden metacells](../../pipelines/empirical/analysis/scripts/metacell_assign.py).
The [metacell adapter](../../pipelines/empirical/analysis/scripts/build_metacell_matching_dir.py)
sums raw RNA counts but averages chromatin values. Downstream normalization
must be traced before asserting equivalence to a uniform mean operator.

## The estimand comes before the aggregation method

Distinguish three targets:

| Target | Meaning | Benchmark role |
| --- | --- | --- |
| Cell-level graph | Direct mechanisms among variables in the same cell | Known simulation truth; attainable paired control and unpaired stress test |
| Aggregate-level graph | Mechanisms/CI structure of the constructed group observations | Derived separately for each observation regime |
| Shared-state association | Covariation of conditional means across matched states | Descriptive target; not automatically direct regulation |

For an i.i.d. homogeneous linear SEM `X_i = B X_i + e_i`, averaging the same
cells with the same externally fixed weights across variables gives
`Xbar = B Xbar + ebar`. With equal-size random disjoint bags of size m,
`Cov(Xbar) = Cov(X)/m`; Gaussian population partial correlations are unchanged.
At a fixed total-cell budget, however, discovery has fewer independent rows.
These are analytical controls, not a claim about state-dependent clustering.

If assays use independently sampled cells from one fixed joint population,
their arbitrarily paired aggregate means have zero cross-assay covariance,
even when the same-cell variables have a direct causal relationship.
Matching groups through a common state H can instead retain
`Cov_H(E[X|H], E[Y|H])`; it does not generally retain
`E_H[Cov(X,Y|H)]`. Shared donor/sample effects introduce a different hierarchical
regime, only applicable if those units exist in the real data.

Do not obtain aggregate truth by simply deleting cross-assay edges from the
cell DAG. Derive the observable distribution and its CI structure, retaining
reduced-form state effects. Report equivalence classes; Gaussian observational
data do not generally identify every direction.

The current latent-space logic adds a second distinction. Integration estimates
a state coordinate or coupling; aggregation uses that estimate to construct rows.
Neither operation observes the missing within-state cross-assay residual covariance.
Every proposed method must therefore declare whether it targets conditional-mean
association, completes a covariance matrix under an explicit residual assumption,
or claims a graph under a stated model class. See the full derivation and benchmark
contract in [the latent-space note](theory/06_latent_space_aggregation.md).

## Proposed staged simulation

### Stage 1: analytical sampling and averaging controls

Use four or five observed causal variables, two assays, a few hand-checkable
motifs (chain, fork, collider and null), and linear Gaussian errors. Keep DAG
coefficients fixed within each paired comparison. Generate one joint cell
population per draw, then expose different permitted views to each method.
Hidden same-cell values are scorer-only in unpaired arms.

1. Jointly observed raw cells: checks causal learner and truth scoring.
2. Jointly observed equal-size random bags: isolates averaging alone.
3. Unpaired random bags from the same distribution: missing-covariance control.
4. Unpaired groups formed with an observed noisy anchor, following the advisor
   design: primary simple matching regime.
5. Exact-state anchor: diagnostic ceiling only, never a deployable method.
6. Row-permuted completed pseudobulks in one assay: destroys cross-assay
   correspondence while preserving all within-assay aggregate distributions.

The paired arm is a simulation control, not permission to acquire paired
experimental data. Begin without latent confounding, then add one common
state and a state-associated noncausal decoy. Separately vary signal carried
by group means versus cell-specific residuals; do not make all true edges
strongly state-visible by construction.

### Stage 2: compare aggregation operators

Hold the observed anchor/embedding and causal learner fixed. Compare equal-
mass hard groups, the existing Leiden grouping, existing one-to-one matching,
and k-neighbor weighted averaging. Random bagging remains a negative or
analytical control, not an expected solution. Start with a few group sizes;
defer robust summaries and large method grids until a failure mechanism is clear.

Keep every cell-to-group weight matrix, cell count, effective count, reuse
count, and discarded-group reason. Prefer common weights within each assay
across its causal variables. Where possible compare identical retained groups
as a diagnostic, alongside each method's full operational yield.

### Stage 3: realistic observation and integration

Add counts, library depth and sparse detection one at a time, preserving the
same underlying biological draw across methods. Explicitly distinguish
mean-of-log, log-of-mean, and sum-plus-offset analyses. These are different
estimands/operators, not interchangeable implementations.

Use held-out background features to fit the same scGLUE integration pathway
used by the real analysis, including its normalization and guidance policy.
Freeze each embedding for the within-run aggregation comparison; test
integration-seed variability separately. Background-only versus focal-feature-
inclusive integration is a leakage sensitivity comparison, not a license to
use planted graph edges as guidance. A toy noisy shared anchor remains an
analytical approximation, not a claim of reproducing scGLUE.

## Learning methods worth formulating

Start with one fixed CI-based learner: PC for the causally sufficient linear
control, FCI for the explicitly latent-confounded extension. Do not impose
chromatin-to-RNA directions in the primary benchmark. A score-based method
can serve as a secondary check after aggregation effects are understood.

The first substantive algorithm candidate is an aggregation-aware score or
CI test. Model each pseudobulk as an uncertain estimate of a group mean rather
than an independent noise-free cell. Use measured weights and same-assay
within-group covariance to quantify heteroskedastic sampling noise; account
for overlap between rows. Cross-assay cell-specific covariance is an unknown,
not an entry to set to zero or fill from the between-group correlation.

A second candidate learns over several nonoverlapping partitions with
uncertainty calibrated by resampling the original independent units. Partition
agreement is a robustness measure, not proof of causality. Neither candidate
has a recovery guarantee yet; the controlled simulation must establish the
conditions where it improves error rates and graph recovery.

For the nonlinear extension, a substantive alternative is moment-aware
aggregation. If `Y = a X^2 + error`, the exact same-cell aggregate equation is
`mean(Y) = a * (mean(X)^2 + mean((X - mean(X))^2)) + mean(error)`.
Keeping the within-group variance restores a term discarded by mean-only
aggregation. Compare mean-only with mean-plus-variance summaries before
introducing generic learned aggregators. This gives vector-valued aggregate
nodes and requires an appropriate structural/CI learner. Interactions among
parents in different assays require unobserved mixed moments, so this is not
a general repair for disjoint-cell information loss.

## Evaluation and pitfalls

Use paired method comparisons on the same generated cells. Vary total cells
at fixed bag size and bag size at fixed total cells in separate contrasts;
also compare matched aggregate-row counts, explicitly disclosing their higher
cell budget. Repeated partitions of one draw are not independent Monte Carlo
replicates. Resample donors/samples if present; otherwise respect original
cell reuse and draw-level dependence.

Primary outputs: within- and cross-assay adjacency precision/recall, structural
Hamming distance under an appropriate CPDAG/PAG metric, false discovery and
false orientation rates, and interval coverage. Also report pairwise/partial
covariance distortion, planted CI preservation, retained independent units,
effective sample sizes, row overlap, completeness loss, and computation time.
Compare precision-recall curves or recall at matched false-positive rates;
more selected edges alone is not an improvement.

Specific pitfalls are outcome-dependent grouping, collider/selection bias from
complete-metacell filtering, unequal cell-type mixtures, within-assay versus
cross-assay noise asymmetry, overlap/pseudoreplication, count normalization,
nonlinear mechanisms that do not commute with averaging, Gaussianization of
non-Gaussian errors, rank-deficient state means, and reusing benchmark draws
for repeated model selection. Good embedding mixing is not a causal-recovery
metric. A finite-sample failure does not establish mathematical impossibility.

The first deliverable should be a small control table reproducing analytical
predictions, followed by a fixed-data aggregation comparison, not a new large
simulation platform. Cell-level and aggregate-level truth scores must remain
separate throughout.

## Literature anchors and unresolved choices

Fan, Gong and Zhang (ICML 2024) distinguish functional and CI consistency under
aggregation; their setting is temporally aggregated i.i.d. observations, not
unpaired assay matching.[^1] Luo, Sun and Zhang (2026 preprint) study bulk gene
expression aggregation and emphasize linear/affine preservation conditions
within their formal setup; this is not a theorem solving the unpaired case.[^2]
The GLUE paper motivates reproducing the actual guidance-linked integration
step rather than substituting an oracle latent coordinate.[^3]

Before implementation, confirm which advisor/Joe simulation archive to use,
which biological sample identifiers genuinely overlap between assays, and
whether the first primary target is preserving the original cell-level CI
structure or learning a separately defined sample/group-level graph. The
proposed benchmark reports both, without treating them as interchangeable.

[^1]: Fan, S., Gong, M. and Zhang, K. (2024). On the Recoverability of Causal Relations from Temporally Aggregated I.I.D. Data. https://proceedings.mlr.press/v235/fan24a.html
[^2]: Luo, G., Sun, B. and Zhang, K. (2026). On the Recoverability of Causal Relations from Bulk Gene Expression Data. Preprint, revised June 8. https://arxiv.org/abs/2606.00568
[^3]: Cao, Z.-J. and Gao, G. (2022). Multi-omics single-cell data integration and regulatory inference with graph-linked embedding. https://www.nature.com/articles/s41587-022-01284-4
