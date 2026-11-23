# What the existing unpaired assays identify

Astra synthesis, 2026-11-14. Scientific review and requirement audit accepted;
see [the full audit](FINAL_REQUIREMENT_AUDIT.md) for evidence and remaining
scientific limits. Mechanical documentation sealing is recorded separately.

The available evidence supports **conditional positive identification,
class-specific nonidentification and useful partial identification**, rather
than universal success or universal impossibility. The current assays have
not established an identified original genomic cross-assay causal graph.
That empirical and assumption-level conclusion must not be confused with the
mathematical fact that some restricted unpaired latent models are identifiable.

## Observation boundary

The [contract](theory/OBSERVATION_TARGET_CONTRACT.md) distinguishes the historical
RNA/six-histone monocyte datasets from the S1 RNA/H3K27ac/H3K4me3 training
cohorts. Historical donors are not verified as matched. S1 training pools
are disjoint but have different local QC; its paired holdout uses a different,
all-assay-QC selection event. Feature annotations and fixed guidance are
observations/priors, not experimentally established causal effects. Library
depth is an observed proxy whose relation to latent technical exposure must
be modeled. Existing ADTs, simulation state/graph truth, paired cell links and
other reserved outcomes retain their evaluator-only roles.

The [study inventory](reviews/OBSERVATION_INVENTORY.md) also records historical
experiments with different contracts. In particular, CA-BOSS used other-block
simulation truth to define calibration nulls; it is not evidence of an entirely
label-free learner under the present contract. Paired aggregation oracle arms
remain upper-bound controls. Neither was rerun or silently adopted as a new
permitted fitting procedure here.

Independent disjoint samples and shuffled views of one finite paired cohort
are different statistical experiments. The latter can retain dependence in
whole-sample summaries. Neither an exact product-law impossibility theorem
nor a method designed for a shuffled common cohort should be transferred to
the other design without a separate argument. Inherited joint selection also
prevents blindly applying a preselection counterexample to the selected law.

## Population results by target and model class

| Model class and target | Established result | Boundary |
|---|---|---|
| Unrestricted cross-assay coupling, specified selected population | Identical complete assay marginal laws can arise from different joint laws. The finite rational core supplies explicit countermodels including identical fixed guidance and metadata. | This is failure of identification over classes containing the pair, not a claim that every particular data law or every restricted model is ambiguous. |
| Direct functional parent and intervention risk difference in explicit finite SCMs | Equation replacement and finite pushforwards give models with identical observations but different parent/effect targets. A strictly positive joint counterpart also differs. | The Lean model is small and explicit; general noisy genomic measurement/selection correspondence is not kernel-certified. |
| Exact recovery by an algorithm receiving population observations | No deterministic function of the shared observation recovers the differing target uniformly; a forced randomized binary decision has worst-case error at least one half on the pair. | The Lean randomized result is a function of population observations. The general finite-sample stochastic-kernel argument is in prose/prior theory, not silently included in that formal theorem. |
| Binary same-cell detection joint from common-population marginals | The Fréchet interval is sharp; every feasible point is attained by a normalized nonnegative table. Independence is always a compatible completion. | Pairwise intervals need not be jointly attainable for all feature pairs at once. No causal direction or activity effect follows. |
| Binary joint under unconditional cross-assay independence | Marginals uniquely determine the full table; existence and uniqueness are proved in Lean. | Independence is an additional cross-assay restriction, not testable from the two marginals alone. It is not claimed for actual data. |
| Finite shared-stratum conditional-independence completion | A mathematical product formula identifies the joint, with zero-mass strata handled explicitly. | This broader theorem is a prose argument, not the formalized binary theorem. True shared cell strata are not supplied to our learner. |
| Sturma restricted unpaired linear latent model | Published assumptions identify the joint; stronger pure-child/rank-faithfulness conditions identify shared latent causal structure with specified representation symmetries. | Linear measurement, distinguishable source laws and structural assumptions are not established for the sparse counts; anonymous latent nodes are not named regions/genes. |
| Known Poisson/NB/independent-mask observation kernels | The archived transform argument preserves each assay's full latent-intensity law under known constant exposure and known kernel parameters. | Unknown technical noise, endogenous proxies and arbitrary cross-assay coupling remain outside the claim; population injection need not be statistically stable. |
| Finite latent mixtures with separately identifiable within-assay mixtures | Distinct common class weights can align labels; an additional cross-assay conditional-independence premise completes the joint. Equal weights admit an explicit label-swap counterexample. | The within-assay primary theorem differs between discrete and continuous models; selection can change weights; finite-class biology is not established. |
| Auxiliary/multiview/anchor/monotone/transport restrictions | Exact positive results recover their stated representations, response functions, class parameters or chosen couplings. | Required paired views, observed informative auxiliaries, structural noise, valid anchors, common measurement maps or true metric/cost restrictions are absent or unverified. They are not interchangeable causal-graph theorems. |

The [primary theorem ledger](literature/THEOREM_LEDGER.md) and
[Astra decisions](plans/ASTRA_APPROACH_DECISIONS.md) give result-specific
conditions, symmetries, applicability and reproduction dispositions. In
particular, Timilsina's unpaired shared-component target differs from Sturma's
joint/graph targets; its paired-sample theorem cannot be fitted under this
contract. The recent manifold result is provisional and concerns alignment
modulo metric-measure symmetries, not causal directions.

The subsequent OpenAlex pass also surfaced Shrestha–Fu's claimed nonlinear
content/style recovery with unknown dimensions bounded in advance. Its
separate [P11 disposition](plans/ASTRA_APPROACH_DECISIONS.md)
retains this positive result while noting its common measurement coordinates,
smooth bijection and exact sparsity/population restrictions. Those assumptions
are not supplied by heterogeneous, stochastic count assays; a block-transform
equivalence is not a genomic causal graph. Independent review then found a
literal premise obstruction: the all-measurable non-cylinder domain-variability
condition is contradicted by removing a common null point under continuous
domain laws. The 2026 differential-independence extension repeats this
condition. The [explicit argument](theory/DOMAIN_VARIABILITY_OBSTRUCTION.md)
has independent mathematical checks, but is not part of the Lean formalization.
We therefore retain these as claimed conditional results with an unresolved
premise-consistency issue, not demonstrated nonvacuous positive classes. A
potential intended interpretation modulo null sets requires clarification and
a repaired proof; it is not silently substituted. The finding does not refute
the implemented GANs' empirical behavior or all nonlinear identification.

The subsequent [official Timilsina code audit](literature/TIMILSINA_CODE_AUDIT.md)
also finds paired training anchors in the shipped single-cell command and a
static evaluation-indexing concern. It does not allege that labels necessarily
enter gradients or that the population theorem is false. This reinforces the
need to inspect the actual training/evaluation path before treating an unpaired
method title as a theorem-faithful, permitted-data reproduction. No upstream
experiment or bundled example dataset was executed or analyzed.

## What the computations establish

The existing [native Sturma replication](../sturma_pilot_2026_11_01/PUBLISHED_REPLICATION.md)
is retained as a reduced finite-sample reproduction under its published
continuous linear generator. It is not biological validation. The subsequent
[count pilot](../count_mdcrl_2026_11_01/README.md),
[count extensions](../count_extensions_2026_11_01/RESULTS.md) and
[selected-dimension follow-up](../count_rank20_2026_11_01/RESULTS.md) are
distinct adaptations using existing inputs. The completed extension blocks
accepted no cross-assay source pair under their fixed rules. Within-assay
covariance improvements, failed source stability and finite-fingerprint
disagreement are reported separately. Unreached characteristic-function
confirmation and graph extraction are unassessed, not failed graph tests.
The [current integrity audit](reviews/EXISTING_EVIDENCE.md) checks stored
artifacts/review receipts; it is not a new rerun of those scientific experiments.

The [prior generator/model audit](../sturma_pilot_2026_11_01/reviews/assumption_evaluation_review.md)
specifically found a Gaussian state factor, nonlinear background responses,
private/feature noise and count measurement in the cached genomics worlds.
Their `linear_chain` label describes a focal feature mechanism; it does not
make the complete background generator a native Sturma model. Count-kernel
correction alone does not repair all of these structural mismatches.

The new [P01 computation](experiments/P01/RESULTS.md) reports all prespecified
binary detection-joint intervals from the frozen public training panels,
including both cached simulation worlds. It selects no coupling and fits no
causal graph. Every covariance interval contains zero by theorem, which is
not evidence that the biological variables are independent. The paired
biological readout is evaluated only after complete predictions are sealed.
Some empirical holdout joints exceed training-marginal plug-in bounds;
different sample marginals and QC populations can cause this. It does not
refute the sharp population theorem. Bounds recomputed from holdout marginals
are labeled a tautological evaluator check, never a learner success.

These plug-in intervals couple the supplied empirical assay-specific marginal
laws. They are neither population confidence intervals nor certified identified
sets for the all-assay-QC holdout population. No nominal post-selection coverage
is asserted. Simulated latent states were not scored as if they were observed
detection joints. No new benchmark world or biological dataset was created.

## Five questions kept separate

1. **Full-law identification:** proved or disproved only inside an explicit
   model class with the actual observation map.
2. **Objective identification:** a unique optimizer can choose one completion
   without showing that the biological joint is that completion.
3. **Estimator consistency:** requires a theorem about the specified estimator
   under its assumptions; the local count fingerprints and P01 plug-ins do not
   inherit every published consistency result automatically.
4. **Finite-sample recovery:** native replications and reused-data applications
   measure different targets and retain failures/abstentions. A simulation
   success is not an identification proof; a failed fit is not a countermodel.
5. **Biological credibility:** known kernels, source laws, common populations,
   anchors, exclusion restrictions and shared metrics require substantive
   justification. Good modality mixing or numerical fit cannot certify them.

The observed failures do not uniquely separate count-layer misspecification,
limited power, estimation instability, unpaired information loss or incorrect
latent structure. Existing controls narrow some explanations but leave others
alive. No threshold is widened until an edge appears, and no latent graph is
scored against unrelated feature-level truth.

## Search scope and remaining scientific questions

The dated [search protocol](literature/SEARCH_PROTOCOL.md),
[query/access log](literature/SEARCH_LOG.jsonl), primary locators and independent
review cover all eight requested theorem families, extending the earlier
review instead of rescreening it under a false new denominator. They record
actual backward/forward paths, inaccessible primary sources and a bounded
stopping rule. This is not an all-paper exhaustive search or proof of global
saturation. Inaccessible historical count proofs are not assigned invented
theorem numbers; our transform derivation is labeled separately. Provisional
sources and unaudited code remain explicitly marked.

The principal unresolved scientific dependencies are: which positive latent
measurement model is defensible for the selected count data; whether technical
noise/exposure can be identified from permitted observations; whether distinct
source or class laws genuinely match across differently selected pools; what
equivalence class of biological variables a recovered representation denotes;
and whether an assumed coupling/metric/anchor restriction is scientifically
justified. Further empirical fits should follow a concrete theorem-compatible
observation model and an Astra-approved frozen plan, rather than an arbitrary
combination of existing software or forbidden training supervision.

The Lean [inventory and reproduction commands](lean/README.md) and
[independent correspondence review](reviews/LEAN_INDEPENDENT_REVIEW.md) state
exactly what is kernel checked. Formal correctness of a finite model does not
certify the biology. The [cycle ledger](RESEARCH_CYCLES.md) records what each
approach contributed and what a justified next step would require.
