# Independent exact-model experiment review

Preflight review performed 2026-11-15 before scientific generation or fitting,
then updated after the sealed evaluation. No implementation blocker was found.
The experiment demonstrates useful one-source recovery in this exact model
family; the two- and three-source results expose suboptimal fitted solutions.
The following scientific review does not change the frozen experiment.

The reviewed sources are the [generator and runner](../../count_transport_matched.py),
[convergence wrapper](../../count_transport_converged.py),
[matched evaluator](../../count_transport_matched_evaluate.py), their tests,
the unchanged finite-source model and earlier metric/selection helpers, and the
[experiment plan](PLAN.md). Only this review file was edited by this reviewer.

## Exact-assumption contract

The generator independently constructs standardized product-Bernoulli sources,
common source means, assay-specific full-rank loadings and independent Poisson
observations with strictly log-linear rates. It does not call the estimator's
decoder. Finite discrete sources satisfy the required independent, non-Gaussian,
unit-variance conditions; their distinct probabilities below one half exclude
nonidentity signed-permutation symmetries. Nonzero common source means do not
invalidate these checks.

Independent checks of all nine planned mechanisms verified exact source moments,
orthogonal loading columns and strict parameter-bound inclusion. The minimum
loading singular-value ratio was `0.9999999999999996`; rates ranged from
`0.00262248` to `172.63329`, finite and positive without clipping. These checks
inspect predetermined mechanism parameters, not fitted recovery outcomes.

There are no added depth variables, private factors, feature-noise layers,
normalization, zero inflation or negative-binomial substitutions. The same
parameters define all cohorts. Training and calibration use distinct random
streams by assay and cohort; assessment uses one shared source draw per cell
and separate observation-noise streams for each assay.

The supplied latent dimension is an explicit positive-control condition. The
learner does not read true probabilities, means, loadings, latent draws or paired
assessment counts. The 3,000-cell fit is a prefix of the 12,000-cell fit; both use
the same independently generated calibration and assessment cohorts. Source
dimension and feature counts are not learned in this experiment.

## Optimization and selection

The wrapper preserves each capped checkpoint and continues its parameters,
without a total iteration or function-evaluation limit. Per-call chunk budgets
resume automatically. The initial loose objective tolerance is disabled for
continuation; accepted convergence requires the declared box-gradient mapping
tolerance. Floating-point stagnation, nonmonotonicity and solver failure are
explicit failures, not successful termination. L-BFGS curvature resets between
calls, and the unchanged line-search limit remains a numerical failure condition.

This is constrained numerical stationarity with four starts, not certified
global optimization. Calibration selects among converged finite candidates;
rank and probability-gap acceptance gates do not influence selection. Fallbacks
and failures remain visible. No truth-informed initialization, continuation or
reselection is implemented.

The evaluator-only feasible-point diagnostic compares the stored fitting
objective with the generating parameters' objective on the identical fitting
prefix, using the same equal-assay/per-feature normalization. A gap above `1e-8`
is a witness that a feasible point has lower training loss. A nonpositive gap
does not establish a global optimum. The comparison occurs after fitting and
does not alter any selected model.

## Evaluation and interpretation

Population covariance error and transport action on the true source subspace
are evaluated directly; neither requires source alignment. Mean transport and
weighted-support errors account for the nonzero source mean. Secondary parameter
diagnostics use one global source permutation across every assay. They cannot
repair an incorrect relative alignment between assays.

Predictive scores distinguish the fitted target marginal, true independent
target marginal and true conditional distribution. Oracle score gaps remain
signed. Source-to-target and target-to-source gains against fitted marginals
are the same association statistic, not recovered causal directions.

Selected-fit intervals and capped-versus-continuation comparisons use common
paired bootstrap rows. They are nominal intervals conditional on fixed models;
they omit fitting uncertainty and familywise adjustment. The declared 10% error
and 0.02-nat-per-feature tolerances are performance criteria, not consequences
of E/E2. Three mechanism/sample replicates cannot support a broad reliability
guarantee. This experiment targets count-law transport, not genomic causal DAGs.

The symmetry control correctly sets both source probabilities and means equal
before swapping columns in one assay. Its weighted marginal rate atoms agree
up to `2.14e-14`, while relative cross-covariance changes by approximately
`0.8851`. The unchanged marginal count law follows from that exact atom
permutation; this is an analytic countercontrol, not an empirical fitted result.

## Verification status

The final complete new-plus-legacy suite passed **68 tests** in the independent
run after addition of the feasible-point diagnostic. It covers the
likelihood/gradient algebra, convergence continuation, population model contract,
symmetry control, evaluator formulas, seals and objective comparison. No
scientific datasets were generated, fitted or scored by this reviewer during
preflight.

## Sealed empirical results

The protocol/source pins, all 144 fit artifact hashes, and evaluation seal links
were independently checked. The evaluated output hash is
`be887bfcd1740a649487493ed74c4c0e75b138d4431c54a4cb7def9329035636`.
The [complete evaluation](evaluation.json), [selected-fit table](comparison.tsv)
and [aggregate table](aggregate.tsv) preserve every declared outcome. The completed
[independent numerical replay](NUMERICAL_REVIEW.json) passed 274 hash comparisons,
recomputed 432 calibration scores and 216 selected ordered prediction readouts,
and verified the population covariance/transport and feasible-objective checks.
Independent automatic differentiation at the 36 selected points agreed with
stored raw and projected gradients within `1.2e-15`. Bootstrap intervals,
optimization trajectories and unselected-restart assessment metrics were not
replayed. This section independently reviews scientific interpretation.

All 72 continued optimization streams reached the declared gradient tolerance.
The selected continuation outcomes were:

| Sources | Fitting cells per assay | Replicates passing every recovery criterion | Numerically accepted selected fits |
|---|---:|---:|---:|
| 1 | 3,000 | 2/3 | 3/3 |
| 1 | 12,000 | 3/3 | 3/3 |
| 2 | 3,000 | 0/3 | 3/3 |
| 2 | 12,000 | 0/3 | 3/3 |
| 3 | 3,000 | 0/3 | 2/3 |
| 3 | 12,000 | 0/3 | 0/3 |

With one source and 12,000 fitting cells, each replicate recovered all three
assay-pair relationships under the declared criteria. Across those replicates,
the worst-pair covariance errors were 1.80%–4.48%, restricted transport errors
were 1.98%–5.02%, and worst oracle conditional-score gaps were at most 0.000131
nats per target feature. The one-source case that missed the criteria at 3,000
cells had a worst covariance error of 10.263%, slightly above the declared 10%
threshold. These are successful finite-sample illustrations in a favorable,
known-dimension family, not a general reliability guarantee.

The higher-dimensional results require two qualifications. First, every selected
two- or three-source fit, and every other evaluated start at those dimensions,
has a higher training objective than the feasible generating parameters. The
selected objective gaps range from approximately 0.0114 to 0.1153 normalized
NLL units. Thus suboptimality is established directly, despite numerical
stationarity; an infinite sequence of further iterations at the same stationary
point is not a global-optimization method.

Second, `d2_rep0` partially succeeds and must not be described as a complete
transport failure. Its restricted transport error is 4.46% at 3,000 cells and
2.70% at 12,000 cells, and its worst oracle conditional-score gap is below
0.000568 nats per feature. However, population covariance error is approximately
195% and 190%, respectively, so neither fit recovers the full population law
under the criteria. Correct action on log-rate supports and good conditional
prediction can coexist with incorrect source-state frequencies.

Read-only parameter inspection supports that explanation for this partial case.
At 12,000 cells the fitted loading matrix in each assay is approximately the
true loading matrix multiplied by `[[0, -1.52], [1, 0]]`, with only 1.8%–2.8%
relative residual outside the true column space. The coherent permutation and
reflection preserve relative transport, while the reflected source probability
hits the fitted upper boundary near 0.4473. The equivalent reflected true source
would instead have probability near 0.88. This identifies a specific binary-label
orientation barrier worth investigating in a separately frozen algorithm change;
it is not evidence that higher-dimensional recovery is impossible.

## What continuation established

Every selected capped checkpoint stopped on the relative-objective tolerance
before reaching 600 iterations. Continued optimization mostly tightened
stationarity without materially changing these selected predictions. The
one-source capped fits already had nearly identical covariance and transport
accuracy; their recovery flags were false because they failed the stricter
gradient-acceptance requirement. Therefore the positive one-source result should
not be attributed to escaping a 600-iteration limit, and the two modes should
not be presented as a large accuracy improvement from extra iterations.

The four starts are a finite initialization search, even though the continuation
has no total iteration or evaluation budget. True parameters were not used to
initialize, optimize or select any fit. A true-parameter warm start would be an
oracle optimization diagnostic, not a further demonstration of learning from
unpaired data; it was not needed to establish the feasible-objective failures.

The supported conclusion is that the corollary's count-joint and transport
targets can be recovered empirically in the one-source positive control, while
this optimizer/parameterization has not achieved reliable multidimensional
recovery. The theorem and the correctly specified generator are compatible with
these outcomes. No planted biological causal DAG, unknown-dimension discovery,
global optimization or realistic-data guarantee follows.
