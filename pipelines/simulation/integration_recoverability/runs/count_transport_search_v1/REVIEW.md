# Independent source-search review

Preflight and sealed scientific results reviews are complete. The finalized [plan](PLAN.md),
[source-search model](../../count_transport_source_search.py),
[runner/evaluator](../../count_transport_search_run.py) and tests have no
outstanding scientific implementation blockers. The complete four-seed
source-search procedure with calibration selection meets the frozen recovery
criteria in all nine reused worlds. Individual starts remain unreliable.
No scientific data generation, benchmark fitting or assessment scoring was
performed by this reviewer during preflight.

## Model identity and optimization

The new coordinates are exactly equivalent to the strict Bernoulli E2 model for
interior source probabilities: $M_c=A_c\operatorname{diag}(\sqrt{p(1-p)})$ and
$\mu=(h+p)/\sqrt{p(1-p)}$. The common offset $h$ supplies the log-rate intercept
through $A_ch$; there are no independent assay intercepts. Probability logits,
offsets and loading entries have no finite optimizer boxes. The plain arm
converts the legacy initializers without changing their physical laws.

The exact marginal objective and its derivatives were independently
reconstructed with float64 automatic differentiation on fixed one-, two- and
three-source fixtures, including probabilities above one half. Objective
differences were zero and maximum gradient difference was `3.7e-17`.
Independent full-law checks of global bit complementation and source sorting
agreed in log weights and joint log-rate atoms within `4.5e-16`.

The complement operation correctly uses $h_j\mapsto-h_j-1$ and negates the
same $A_c$ column in every assay. Conversion to standardized source parameters
preserves the model and supports the unchanged evaluator. No separate
truth-assisted assay rematching is used.

Joint refinement has no cumulative iteration/evaluation budget, objective
tolerance is zero from the first chunk, and convergence requires the actual
unbounded gradient infinity norm to be at most `1e-6`. A nonfinite trial is
rejected for line-search backtracking, and the final returned point is evaluated
again with the real objective before convergence is certified. Thus placeholder
gradients from rejected trials cannot become false convergence. Numerical
failures, line-search safeguards and finite floating-point representation remain
explicit limits.

## Count-based initialization and discrete search

Poisson mixture EM uses raw counts from every feature, not selected principal
components. Initial smoothing applies only to seeded rates. Exact updates do
not clip fitted weights or rates; zero/dead components and nonfinite states are
explicit failures. EM checks its normalized marginal gradient, has no total
iteration cap and verifies monotonicity before declaring success. Only finite,
gradient-converged mixtures enter cube assignment.

The cube quotient is valid. Bit complementation fixes component zero at vertex
zero; sorting the indices at its coordinate neighbors removes coordinate
permutations. Independently normalizing all 2, 24 and 40,320 assignments produced
exactly the implementation's 1, 3 and 840 representatives. The complete-state
Poisson and product-weight KL score is correctly normalized. Its affine
least-squares geometry is an initialization heuristic, not a global likelihood
solution for each assignment.

Assay-specific affine intercepts exist only inside that initializer. The joint
candidate projects them to one common $h$ with feature-balanced least squares.
Source probabilities are inferred from public mixture weights, canonicalized
and sorted; no generating probabilities or source labels are supplied. Up to
64 assay-candidate combinations are scored by training likelihood, four are
refined, and the inner winner uses training likelihood only. Every mixture,
assignment score, candidate failure and refinement record is retained.

The search is exhaustive only over labelings of fitted atoms. Marginal mixture
starts and joint refinements are finite. This is a stronger estimator test,
not a global-optimization certificate or a guarantee of finite-sample recovery.

## Information boundary, selection and interpretation

The runner reuses the nine existing 12,000-cell worlds, original feature panels
and calibration counts. It creates no new scientific dataset. Fitting receives
only public counts and the supplied dimension. An explicit payload guard
prevents an inner fit from overwriting outer world, mode or seed identity.
Scoring failures preserve the returned model and inner diagnostics.

All 72 outer records and their nested searches remain available. Outer
selection is calibration-only within each world/mode and is replayed at seal
verification. Numerical rank and source-gap screens do not select the winner.
Groups with no finite model retain null selections and stay in denominators.
The declared maximum workload is 432 marginal mixture runs and 180 joint
refinements; reporting only the 72 outer records would understate the search.

The evaluator reuses the previously reviewed population, predictive and
feasible-point metrics after the new fit seal. Paired assessment and generating
parameters do not enter initialization, fitting, stopping or selection. These
worlds have already been assessed, so subsequent results and bootstrap intervals
are exploratory. Known source dimension and the finite independent-source family
remain substantive assumptions. Successful transport would not by itself
establish biological causal arrows or applicability to real disjoint assays.

## Verification status

All 112 count-transport tests passed independently in the finalized suite,
including 18 new model tests and seven new runner tests. They cover physical
initializer equivalence, analytic derivatives, exhaustive cube coverage,
canonicalization, unbounded continuation, rejection of invalid trial points,
EM boundaries, failed-fit retention and separate mode selection seals.

## Sealed scientific findings

Reviewed the [fit seal](fit_seal.json), [evaluation seal](evaluation_seal.json),
[evaluation](evaluation.json), [comparison table](comparison.tsv),
[aggregate table](aggregate.tsv), and nested fitting diagnostics. The evaluation
output hash is
`0d8cc7c9fc2e5ee4723c3b77c67201f40752e20adbe103fdd4faf1341c180f39`.
The earlier 12,000-cell matched and labeling runs are fixed comparisons.

| Shared sources | Original bounded model | Two-sided labels | Plain unbounded | Count-based source search |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3/3 | 3/3 | 3/3 | 3/3 |
| 2 | 0/3 | 0/3 | 2/3 | 3/3 |
| 3 | 0/3 | 0/3 | 0/3 | 3/3 |

These are formal complete-world recovery counts for the calibration-selected
models only. Each selected model has the prescribed bootstrap intervals. The
six ordered predictions per world represent three associations; they do not
increase the number of independent worlds or establish causal directions.
Unselected starts lack these intervals and must not be assigned formal recovery
counts from their continuous metrics alone.

For the nine selected source-search models, each world's worst-pair covariance
relative error is 1.73–4.46%, and its worst restricted transport relative error
is 1.97–5.02%. The largest oracle conditional-score gap is 0.000202 nats per
target feature. Every predictive-gain interval has a positive lower endpoint;
the smallest per-cell lower endpoint is about 0.272 nats. All selected models
pass numerical rank and source-separation gates. These results are comfortably
inside the frozen thresholds and constitute a positive empirical demonstration
of cross-assay transport and count relationships in this assumed model family.

All nine selected source-search training objectives are smaller than the
objective at the true generating parameters on those same fitting counts.
The earlier positive feasible-point witnesses therefore disappear for the
selected procedure. A lower objective at a fitted parameter is expected to be
possible in finite samples; it does not certify a global optimum. Four selected
plain models still have positive feasible-point witnesses: `d2_rep1` and all
three-source worlds. Thus removing boxes and changing coordinates improves the
plain results but does not, by itself, solve the tested higher-dimensional
optimization problem.

## Restart instability and computation

The successful object is the full source-search and calibration-selection
procedure, not an arbitrary single start. Among its 36 outer candidates,
33 pass numerical gates and nine still have positive feasible-point training
gaps: two in dimension two and seven in dimension three. Some poor candidates
have very large covariance errors despite gradient convergence. In `d3_rep0`,
only one of the four source-search outer candidates avoids that objective
witness; calibration selects it. The plain mode has 22 such witnesses among
its 36 outer candidates. These are optimization diagnostics, not formal
unselected recovery counts.

All 72 outer candidates are finite and gradient-converged; 65 pass the
numerical/model-class gates. Of 432 attempted marginal mixture fits, 415
converge and 17 fail explicitly because of zero/dead/nonfinite mixture
parameters or rate updates. All those failures remain recorded. Of 180 joint
refinements, 179 converge and one ends in floating-point stagnation. The longest
joint refinement takes 36,134 iterations and belongs to source-search
`d3_rep1`, outer seed 1. The reported fitting wall time is approximately
50.6 minutes. This experiment removed cumulative budgets and finite boxes,
but its successful search requires materially more computation than the
earlier similar-start fits.

The comparisons identify effects of estimator procedures, not isolated causal
effects of individual software changes. Plain optimization changes coordinates
and parameter bounds together. Source search additionally changes initialization
and computational effort. The data do not distinguish which individual
subcomponent was necessary or sufficient.

## Independent checks and scientific conclusion

Using no imported population evaluator, this reviewer reconstructed selected
source laws and rate atoms from probabilities, source means and loadings, and
recomputed covariance and restricted transport errors for all 108 selected
ordered cases. Maximum absolute discrepancies in the reported relative errors
were `1.1e-14` for covariance and `6.7e-16` for transport.

The separate [numerical replay](NUMERICAL_REVIEW.json) is complete and passes.
The runner/evaluator author independently implemented SciPy likelihoods,
NumPy moments and transport, and Torch gradients without importing the fitted
objective or evaluator. The replay verifies 144 hashes, all 72 outer training
objectives, all 18 calibration selections, 108 selected ordered predictions
and 216 main bootstrap intervals. Maximum full-gradient discrepancy is
`2.9e-15` and main interval discrepancy is `1.6e-14`. This reviewer also
inspected the audit's raw/canonical bit mapping and unbounded convergence gate.
The audit does not rerun fitting, reconstruct every initializer, independently
score every inner refinement or replay other descriptive bootstrap intervals;
inner selection is checked against retained training objectives. Scientific
review and numerical replay therefore have explicit, complementary scopes.

The empirical answer is now positive for the tested finite-source instance of
E2: the frozen count-based source-search procedure recovers the specified
cross-assay relationships in all nine reused exact-model worlds. This is
stronger evidence than the earlier one-source-only success and shows that the
previous higher-dimensional failure was not an inherent barrier in these data.
It does not establish success for every source-search start, all distributions
covered by the population theorem, arbitrary nonlinear integrations, unknown
source dimension or real biological assays. These are only three mechanisms
per dimension, selected from a previously assessed benchmark, so general
reliability and fresh-test performance remain unestablished.

The observations remain disjoint during fitting. The success depends on the
assumed common population, independent Bernoulli sources, known dimension,
full-rank log-linear maps and conditional Poisson independence. It concerns
transport and the declared count-law recovery metrics, not deterministic
matching of noisy cells or genomic intervention directions. No frozen fitting
rule, candidate selection or evaluation threshold was changed after results.
