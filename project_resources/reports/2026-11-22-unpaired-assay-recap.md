# What we recovered from unpaired assays

Research recap, 22 November 2026. The latest experiment in this recap was run on 15 November.

**We recovered how a shared latent state links assay intensities in all nine simulations generated under our count model.** The learner saw disjoint sets of RNA, H3K27ac, and H3K4me3 cells. In each simulation, the selected model passed the criteria for cross-assay covariance, transport of log intensities, and prediction of one assay from another. Across all nine worlds, the worst covariance error was 4.46% and the worst transport error was 5.02%. The experiments supply a positive example for the corollary. Whether a histone peak directly regulates an RNA feature in the biological data remains open. [Final experiment][search] · [Per-world results][scores]

The November 22 handoff reproduced the complete earlier matched comparison from a clean checkout and ran one fit from the follow-up source search. The nine-world source-search result above comes from the original November 15 experiment and its independent numerical audit. [Transfer verification](../../reproduce/VERIFICATION.md)

## Why we started with aggregation

In the monocyte analysis we integrated RNA and histone assays with scGLUE, formed matched rows or metacells, and ran causal discovery on nearby peaks and genes. The historical data did not establish shared donor identities that could define matched donor pseudobulks. At CD14 we compared a 1,561-row RNA-anchored matrix that reused chromatin cells with 133 non-overlapping metacells. The latter improved the permutation-null calibration. Across a subsequent 15-gene panel, 41 cross-mark adjacencies were stable across partitions and enriched for shared promoter contacts. No cross-mark direction was stable across partitions at the resolutions tested. [Aggregation comparison][aggtest] · [Cross-mark analysis][crossmark] · [Sampling contract][contract]

The sampling distinction explains why these observations can coexist. In a linear structural model, averaging **the same cells with the same fixed weights** preserves the structural equations. For an equal-weight average of $m$ independent cells, the covariance matrix becomes $\Sigma/m$. If RNA and chromatin instead come from independent cells from one homogeneous population, arbitrarily pairing their averages has zero population cross-covariance. Even with a perfectly matched state $H$, the same-cell covariance has two terms.

$$
\operatorname{Cov}(X,Y)
= \operatorname{Cov}\!\bigl(\mathbb E[X\mid H],\mathbb E[Y\mid H]\bigr)
+ \mathbb E\!\bigl[\operatorname{Cov}(X,Y\mid H)\bigr].
$$

The first term is an **assay–assay relationship across states**. If the average RNA and histone signals rise in the same states, their state means covary. The second term asks whether a cell with unusually high histone signal for its state also has unusually high RNA for that state. Disjoint cells reveal each assay's within-state variation separately. Different ways of coupling those deviations give the same observations and different answers to a cross-assay conditional-independence test. Averaging across more states leaves that ambiguity in place. This led us to test integration before asking which aggregation operator best preserves structure. [Aggregation derivation][aggregation] · [Population counterexamples][nonid] · [Two-track decision][tracks]

## What we tried before the count-transport corollary

**CA-BOSS and the archived multimodal benchmark.** We replaced a permissive sparse H3K4me3 rescue with state- and depth-adjusted evidence, partial pooling, empirical nulls, and separate within-assay and shared-state outputs. The v4.59 checkpoint used six chromatin assays plus RNA across 103 cached worlds. It reported 239 correct shared-state hyperedges, zero false reports and 70 misses, giving precision 1.00 and recall 77.35%. It identified 402/618 assay participations, with 0/1,236 sentinel false positives. Equivalence sets covered 239/309 truths with 1.293 alternatives on average. All 201 exact pair claims were correct, giving 65.05% recall. This truth family has no defined direction. Other simulation blocks supplied calibration nulls. Later variants through v4.64 did not displace this checkpoint. [CA-BOSS results][caboss]

**Integration on simulated and biological counts.** A 32-fit comparison tested scGLUE and scMRDR against state and matching controls. Four further scGLUE fits compared PCA/LSI inputs with full-feature encoders on the same two worlds. State matching improved modestly with full features, while direct structure recovery remained inconsistent. We also tested scConfluence in six fits spanning two simulations and a scMTR-seq pilot. One simulated fit recovered a count adjacency between two shared-state decoys. The biological holdout showed some histone–histone correspondence and weak RNA–histone correspondence. An artificial-null control exposed miscalibration in the sparse-count conditional-independence test. The broader original integration benchmark remained incomplete at its recorded checkpoint (118/178 model manifests). [Benchmark status][integrstatus] · [Integration study][integration] · [Count-input comparison][countinput] · [scMTR-seq pilot][scmtr]

**Sturma et al. and count-aware source matching.** Using the authors' continuous linear simulator and implementation, we recovered the shared dimension in 43/60 fits and met the declared latent-coefficient criterion in 21/60. The native method failed to recover shared sources on the existing small count panels. A 36-arm Poisson-correction pilot improved some within-assay covariance estimates. A further 225 arms tested count noise, excess zeros, depth, higher moments, source orientation, and panel size. A 54-arm follow-up raised the dimension ceiling. None of these count studies accepted a cross-assay source pair. [Sturma replication][sturma] · [Count pilot][pilot] · [Extensions][extensions] · [Dimension follow-up][rank]

**Identification from marginals.** We derived examples in which identical observed assay distributions permit different same-cell joints, direct parents, and intervention effects. We also computed the sharp range of binary joint detections compatible with given marginals. Every such population range permits independence. On scMTR-seq, 25 of 2,700 paired empirical detection joints lay above the upper bounds calculated from the training marginals. Sampling variation and the different training and holdout quality filters limit that comparison. A literature review traced which integration and latent-variable results can narrow the possibilities through assumptions about sources, measurement, or auxiliary observations. The nonlinear recheck found multiple distinct results and documented problems with an unrestricted domain-variability premise. [Identification synthesis][idsynth] · [Detection bounds][p01] · [Integration literature][lit] · [Nonlinear recheck][recheck]

## The corollary and its unequal-width extension

**What would let two unpaired assays agree on the same hidden source?** In this model each source has a distributional signature that survives measurement and distinguishes it from the others. Both assays can use that signature even though they contain different cells and different features. The common source law is a substantive assumption about the populations retained after quality control.

Let $S\in\mathbb R^d$ be the shared state. For assay $c$, let $M_c$ map the state to log intensities $L_c$, let $R_c$ be the vector of positive intensities, and let $Y_c$ be the observed counts. The measurement model is

$$
L_c=M_cS,\qquad R_c=\exp(L_c),\qquad
Y_{cj}\mid S\sim\operatorname{Poisson}(R_{cj}).
$$

The exponential acts coordinate by coordinate. The assumptions are independent, non-Gaussian, unit-variance source coordinates with one common law across assays. No nonidentity permutation or sign change, including combinations of both, may preserve that law. Every $M_c$ has full column rank. Counts within an assay are conditionally independent given the state. The model allows a nonzero source mean. Assay-private latent factors and freely varying assay intercepts fall outside this statement. [Square corollary E][E] · [Unequal-width corollary E2][E2]

1. **The counts identify each assay's population log-intensity distribution.** Under the product-Poisson model, the generating function of the full within-assay count vector gives the multivariate Laplace transform of its rate distribution on an open box. That transform uniquely determines the rate law. Taking the known logarithm gives the law of $L_c$. This uses the distribution of counts across cells, including within-assay feature dependence. Taking the logarithm of one observed count cannot recover its cell's intensity. [Count-channel proof review][Ereview]
2. **E identifies transport for square maps.** If $M_a$ and $M_b$ are square and invertible, the unique invertible linear map carrying the law of $L_a$ to the law of $L_b$ is $T_{ba}=M_bM_a^{-1}$. Any second candidate would induce a linear transformation preserving the source distribution. Independent component analysis restricts that ambiguity to signs and permutations under the stated normalization, and the source asymmetry rules out every nonidentity choice. A common relabeling of source coordinates across all assays cancels in the transport. On the intensity scale, transport is $r\mapsto\exp(T_{ba}\log r)$. A simple permitted example is $R_b=R_a^2$. [Square result and example][E]
3. **E2 allows different feature counts.** Assay $c$ may have $p_c\ge d$ features. Since $\operatorname{Cov}(S)=I_d$, the log-intensity covariance is $M_cM_c^\top$. Its rank identifies $d$ at the population level and its range identifies $V_c=\operatorname{col}(M_c)$. Expressing each assay in a basis of $V_c$ reduces it to the square case. The resulting transport is the unique linear map $U_{ba}:V_a\to V_b$ satisfying $U_{ba}(M_as)=M_bs$. One representative is $M_bM_a^\dagger$ restricted to $V_a$, where $\dagger$ denotes the pseudoinverse. Data place no constraint on how the matrix acts outside $V_a$. RNA and histone features can differ in both number and meaning. [Extension and independent proof review][E2review]

**Completing a same-cell joint law takes an additional step.** The structural model says that the intensities of all assays *in one hypothetical cell* are functions of the same realization of $S$. This couples their rate vectors. To obtain the count joint, we additionally assume independent measurement noise **across** assays conditional on that state. Then we can average the product of the assay-specific Poisson probabilities over the common state law. If residual measurement noise may be shared across assays, its coupling remains free. Neither construction gives the intervention effect of changing a peak on a gene. [Joint-law argument][Ereview]

**Where the result comes from.** Our novelty audit judged E/E2 an application-specific corollary with low theoretical novelty. Foster et al.'s CoMP work supplies the unpaired linear alignment argument. Lopez et al. and Chaussard et al. give prior count-observation identification results. Restricting an unequal-width problem to its column spaces is standard linear algebra, and Sturma et al. already treat heterogeneous dimensions. The project contribution here is the assembled statement, its assumptions for disjoint assays, and a count estimator tested against explicit controls. The E/E2 proofs received independent mathematical review. The project's Lean proofs cover narrower finite counterexamples and two-atom algebra. [Novelty audit][novelty] · [Foster et al.](https://proceedings.mlr.press/v162/foster22a.html) · [Lopez et al.](https://proceedings.mlr.press/v236/lopez24a.html) · [Chaussard et al.](https://proceedings.mlr.press/v306/chaussard26a.html) · [Formal proof scope][lean]

## What we tested directly against E/E2

**First attempt on existing, biologically motivated simulations.** We fitted the strict model and an affine sensitivity variant to background counts from the cached linear-chain and constrained-collider worlds. Each assay supplied 3,000 unpaired fitting and 1,000 calibration cells. Each world supplied 2,000 paired cells for evaluation. The 96 fits included column-permutation controls, three candidate dimensions, and four starts. The selected strict fits improved within-assay calibration, yet conditional prediction was worse than using the fitted target marginal for all six world/assay-pair combinations. Covariance errors were **1.25–4.08 times** the error from predicting zero covariance. The affine variant also failed. These generators included continuous state, private factors, varying depth and overdispersion. This experiment used the earlier estimator on background features outside the planted causal motifs. [Cached-world demonstration][demo]

**Positive control with the generator inside the fitted model.** We generated nine worlds for the original matched experiment and reused the same nine for both follow-ups. There were three independently seeded loading configurations per source dimension $d=1,2,3$. Each cell drew independent binary sources, standardized them to unit variance, added a common mean, and passed them through assay-specific linear log-rate maps and independent Poisson sampling. The source frequencies were the first $d$ entries of $(0.12,0.27,0.40)$. RNA, H3K27ac, and H3K4me3 had 30, 24, and 18 features. The unequal widths directly exercise E2. The learner received the correct $d$ and raw counts. It learned source frequencies and loadings without PCA, guidance, source labels, or paired training cells. [Matched design][matched]

At full size, each assay supplied **12,000 disjoint fitting cells and 2,000 calibration cells**. Each world had **4,000 paired evaluation cells**. The benchmark selected the converged outer start with the best unpaired calibration likelihood before opening evaluation values for that run. A world passed when its fit met the numerical checks and **all six ordered predictions** met the declared tolerances. Relative cross-covariance error and relative restricted log-rate transport error each had to be at most 10%. Predictive gain over the fitted target marginal needed a positive nominal 95% lower bound. The mean gap between the oracle and fitted conditional log scores had to be at most 0.02 nats per target feature. The six predictions cover three assay pairs. Their ordering specifies which assay is used for prediction. [Frozen plan][plan] · [Scoring and results][search]

| Estimator on the same nine worlds | 1 source | 2 sources | 3 sources | All worlds |
|---|---:|---:|---:|---:|
| Original model, continued to numerical stationarity | 3/3 | 0/3 | 0/3 | 3/9 |
| Allowed both binary source orientations and a wider range of frequencies | 3/3 | 0/3 | 0/3 | 3/9 |
| Removed finite parameter bounds and changed coordinates, retained old initializations | 3/3 | 2/3 | 0/3 | 5/9 |
| Same unbounded model, initialized by count-mixture fitting and source-assignment search | 3/3 | 3/3 | 3/3 | **9/9** |

The original experiment also tested a 3,000-cell prefix. Two of three one-source worlds passed at that size and none of the higher-dimensional worlds passed. Removing total iteration and evaluation caps brought all original starts to numerical stationarity. All 48 continued two- and three-source starts across the two sample sizes nevertheless had worse training likelihood than the feasible generating parameters. One two-source fit achieved **2.7% transport error** while its **covariance error was about 190%**. The fit approximately preserved the map between assays while misestimating how often source states occurred. Allowing both binary orientations improved one-source stability but left all 24 higher-dimensional starts in that follow-up worse than the generating parameters. [Matched and label-orientation experiments][matched]

**What changed in the successful estimator.** With binary state vector $B$, we wrote $\log R_c=A_c(B+h)$ so the source frequencies determine state probabilities and $A_c,h$ determine rate values. For each assay the informed initializer fitted mixtures of $2^d$ product-Poisson components, tried the possible assignments of components to binary states, and matched source axes using their inferred frequencies. For three sources, 840 distinct label classes cover all $8!=40{,}320$ assignments of the eight estimated components. We refined candidate joint fits on training counts and selected among four outer starts using unpaired calibration counts. This comparison changed both initialization and search effort. [Estimator and search protocol][plan]

**Accuracy and reliability.** Among the nine selected informed fits, each world's worst-pair covariance error ranged from **1.73% to 4.46%** and its worst transport error from **1.97% to 5.02%**. The largest gap to oracle conditional prediction was **0.0002011 nats per target feature**. Across individual informed starts, 12/12 one-source, 10/12 two-source, and **5/12 three-source** fits met the numerical and continuous-error criteria. The full pass decision, including the bootstrap interval, applies to the selected fits. In total, 415/432 marginal mixture attempts and 179/180 joint refinements converged. One refinement took 36,134 iterations. All 112 scoped tests passed. Independent replay checked 72 outer objectives, 18 selections across the two final methods, 108 ordered-pair evaluations and 216 bootstrap intervals. [Results][search] · [Scientific review][sreview] · [Numerical audit][audit]

**A control where recovery really is ambiguous.** Giving two sources equal probabilities and equal means makes them exchangeable. Swapping their loading columns in just one assay leaves every unpaired count law unchanged, yet changes the true cross-assay covariance by **88.5%** in the saved analytic example. This violates the corollary's asymmetry assumption. In the positive-control worlds the sources had distinct signatures, and a better estimator recovered relationships that the earlier optimizer missed. [Analytic symmetry control][swap]

## Related tests and what remains open

Three companion experiments used different models and objectives from E/E2. **C01** aligned finite shared classes by their population frequencies. Its 24 Poisson and fixed-shape negative-binomial model/control arms accepted no shared-state candidate. **C02** tested a restricted latent-class estimator in 36 applications. On comparable full panels, 29/30 finite fits scored worse on held-out counts than a four-component C01 model. **C03** tested an estimator based on off-diagonal moments. Across 36 full-panel comparisons with C01, two improved, 23 had worse finite predictive scores and 11 had negative-infinite scores. [C01 results][c01] · [C02 results][c02] · [C03 results][c03]

The theory work also considered restricted content/style encoders, known inverse links, and auxiliary-variable models. For the count channel, a **known** negative-binomial shape still permits identification of the intensity law. With unknown shape, a constant mean of one at shape one and a uniformly distributed mean on $(0,2)$ at shape two give identical count laws. In a separate nonlinear ICA review, we found a counterexample to a signed-affine identification claim and proved a restricted within-assay result for an exact independent standard Laplace prior. Separate assay fits still retain separate source-label ambiguities in that model. The matched 9/9 experiment used **Poisson** counts and binary sources. [Theory synthesis][repair] · [Known-shape proof][nb] · [Unknown-shape counterexample][unknownnb]

The evidence at this pause supports three conclusions.

- **Population mathematics.** Under E2's assumptions, complete unpaired count laws determine the common dimension, the assay log-intensity subspaces and their transport. The shared-cell construction and cross-assay noise assumption also determine a model-implied count joint.
- **Finite samples.** On nine previously examined worlds generated under one binary-source subclass, the four-start procedure met the prespecified covariance, transport and prediction tolerances. Other distributions, unknown dimension, fresh worlds and distances measuring the entire joint distribution remain untested by that experiment. Calibration selected a successful fit in every world here. Global optimality and a general success rate remain open.
- **Biological interpretation.** Distinct common sources, comparable post-QC populations, full-rank log-linear measurement, and conditionally independent assay noise remain unverified for the biological panels. The successful estimator has yet to demonstrate the same performance on those panels or on the earlier biologically motivated worlds. Direct peak–gene regulation, causal directions and cell-specific residual relationships remain unresolved. [Experimental limits][search] · [Biological observation contract][contract]

[aggtest]: ../../pipelines/simulation/aggregation_identifiability/results/aggregation_bakeoff.md
[crossmark]: ../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md
[contract]: ../research/research_program_2026_11_14/theory/OBSERVATION_TARGET_CONTRACT.md
[aggregation]: ../research/theory/06_latent_space_aggregation.md
[nonid]: ../research/theory/07_unpaired_integration_nonidentifiability.md
[tracks]: ../research/aggregation_restart_plan.md
[caboss]: ../../pipelines/simulation/unpaired_multimodal/results/ca_boss_v4_roadmap_status.md
[integration]: ../../pipelines/simulation/integration_recoverability/INTEGRATION_VALIDATION.md
[countinput]: ../../pipelines/simulation/integration_recoverability/COUNT_INPUT_FOLLOWUP.md
[scmtr]: ../research/scconfluence_scmtr_pilot_2026_11_01/README.md
[sturma]: ../research/sturma_pilot_2026_11_01/README.md
[pilot]: ../research/count_mdcrl_2026_11_01/RESULTS.md
[extensions]: ../research/count_extensions_2026_11_01/README.md
[rank]: ../research/count_rank20_2026_11_01/README.md
[idsynth]: ../research/research_program_2026_11_14/SYNTHESIS.md
[p01]: ../research/research_program_2026_11_14/experiments/P01/RESULTS.md
[lit]: ../research/integration_literature_scope_2026_11_01/REVIEW.md
[recheck]: ../research/nonlinear_recheck_2026_11_14/ASTRA_REASSESSMENT.md
[E]: ../research/nonlinear_repair_2026_11_15/theory/COMP_COUNT_SHARED_TRANSPORT.md
[E2]: ../research/nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md
[Ereview]: ../research/nonlinear_repair_2026_11_15/reviews/COMP_COUNT_TRANSPORT_INDEPENDENT_REVIEW.md
[E2review]: ../research/nonlinear_repair_2026_11_15/reviews/COMP_TALL_INDEPENDENT_REVIEW.md
[novelty]: ../research/count_transport_novelty_2026_11_15/README.md
[lean]: ../research/nonlinear_repair_2026_11_15/lean/README.md
[demo]: ../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_DEMONSTRATION.md
[matched]: ../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_MATCHED.md
[search]: ../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md
[plan]: ../../pipelines/simulation/integration_recoverability/runs/count_transport_search_v1/PLAN.md
[scores]: ../../pipelines/simulation/integration_recoverability/runs/count_transport_search_v1/comparison.tsv
[sreview]: ../../pipelines/simulation/integration_recoverability/runs/count_transport_search_v1/REVIEW.md
[audit]: ../../pipelines/simulation/integration_recoverability/runs/count_transport_search_v1/NUMERICAL_REVIEW.json
[swap]: ../../pipelines/simulation/integration_recoverability/runs/count_transport_matched_v1/symmetry_control.json
[repair]: ../research/nonlinear_repair_2026_11_15/SYNTHESIS.md
[nb]: ../research/nonlinear_repair_2026_11_15/theory/FIXED_SHAPE_NB_BRIDGE.md
[unknownnb]: ../research/nonlinear_repair_2026_11_15/theory/UNKNOWN_SHAPE_NB_COUNTERMODEL.md

[integrstatus]: ../../pipelines/simulation/integration_recoverability/IMPLEMENTATION_STATUS.md
[c01]: ../research/nonlinear_repair_2026_11_15/experiments/C01/RESULTS.md
[c02]: ../research/nonlinear_repair_2026_11_15/experiments/C02/RESULTS.md
[c03]: ../research/nonlinear_repair_2026_11_15/experiments/C03/RESULTS.md
