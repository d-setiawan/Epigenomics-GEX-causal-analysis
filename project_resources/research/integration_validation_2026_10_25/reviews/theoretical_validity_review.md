# What a pre-causal integration check can establish

_Independent theoretical review · 2026-10-25 · proposal, with no new fits or data analysis_

Use the cached development worlds to validate a common biological state chart and the state alignment of the actual cross-assay matching operator. Passing this check would support a specified shared-state use of an embedding. It would not certify the missing same-cell joint distribution, residual coupling, or a causal graph. Keep this exploratory track outside the frozen recoverability suite and leave the old serial runs intact.

**The identification problem.** Distributional alignment, exact reconstruction, cycle consistency, and preservation of all within-assay distances can hold under an incorrect cross-assay correspondence. For example, let the true state be spherical Gaussian, with noiseless states encoded as `S` in RNA and `Q S` in chromatin for a nonidentity orthogonal matrix `Q`. Each assay retains every distance and has the same latent marginal; inverse decoders can reconstruct both assays. Cross-assay nearest neighbors can nevertheless match different biological states. This is a counterexample to a general guarantee, not a claim that every dataset has this symmetry. The corresponding cycle-loss ambiguity under measure-preserving automorphisms is formalized by Moriakov et al., Proposition 2.1.[^1]

Non-Gaussian state alone does not remove the issue: for a continuous one-dimensional state, `F_S^{-1}(1-F_S(S))` also preserves its distribution while reversing its order. A known increasing measurement relationship would rule out that reversal, but the increasing relationship is an extra assumption. Likewise, biological anchors can break some symmetries; the anchor information, its resolution, and its independence from training must be stated.

scGLUE's integration consistency compares aligned metacell correlations with the guidance graph used to fit the model. It is a useful internal incompatibility diagnostic, not independent biological ground truth.[^2] scMRDR's isometric and adversarial terms encourage structure preservation and alignment; the paper acknowledges nonuniqueness of disentangled subspaces.[^3] Neither observation supplies a general biological-pairing theorem for the present RNA/H3K27ac/H3K4me3 data.

**One practical validation approach: held-out common-state transfer, checked against the matching operator.** The following is a proposed protocol, not a measured result.

1. Use only the existing development-world `train` and `val` cohorts. Freeze candidate configurations, feature/guidance contracts, eligible cell IDs, preprocessing, probe capacity, random seeds, and reporting before comparing new results. Fit preprocessing and encoders on training counts. Use a fixed inner training split for early stopping; reserve the existing validation cohort for candidate selection and a stable, disjoint confirmation subset. If any of those validation cells already influenced a particular fitted model, label that comparison accordingly or fit a fresh baseline under the same contract. The check cannot retroactively create a pristine holdout.
2. After an encoder is frozen, fit one simple state decoder `d_a` using source-assay training embeddings and evaluator-only synthetic state. Transfer that exact decoder to each target assay's validation embeddings, without target refitting, a target-specific normalization, an affine realignment, or target state labels. Run every ordered assay direction and every state dimension. Use fixed ridge and a fixed small nonlinear probe as complementary readouts; fit any probe scaler on the source training assay only. Also fit target-own probes as information-retention references, never as replacements for source-to-target transfer.
3. Separately form the declared one-to-one cross-assay assignment in validation embeddings, using identical truth-independent pools for all candidates and controls. Measure hidden state error and preservation of cached state-dependent activity means under that actual assignment. This prevents a usable state decoder from concealing a poor Euclidean matching geometry. Do not run causal discovery as part of candidate selection.
4. Compare against random one-to-one matching and the minimum-squared-state-distance assignment on those same pools. Keep depth-only or nuisance-only prediction, source-state-label shuffles, and within-assay state retention as diagnostic controls. Source-label shuffles are descriptive controls unless exchangeability under the proposed null is justified. Preserve the exact public-equivalence ambiguity twins as the fixed negative control for claims about residual coupling. Do not count them as two independent worlds.

Read actual cohort sizes from frozen metadata and the public cell-ID audit rather than treating the simulator's size parameters as combined assay totals. The verified legacy `linear_chain_0` baseline has 4,000 observed training cells and 500 validation cells **per assay**; its exported common pools contain 1,936 training and 487 validation cells per assay. Details are in the [public baseline crosscheck](baseline_public_train_val_crosscheck.md). Keep both assessment **worlds** and private assessment **cohorts** out of hyperparameter selection. The private probe/assessment cohorts are already assigned to the frozen benchmark; this proposal does not repurpose them.

For a source decoder, the held-out endpoint is

$$
R_{a\to b}=\mathbb E_b\left[\left\|D^{-1}\{d_a(Z_b)-S_b\}\right\|^2\right],
$$

where `D` contains state scales estimated from training truth only. Report the full matrix of risks, target-own risks, and per-state-coordinate errors. A source decoder that works only within its own assay shows state retention without establishing a common chart. Exact zero population risk for one common decoder across all assays establishes that the measured `S` can be recovered in the same chart. Low finite-sample risk supports that narrower claim at the measured resolution; it leaves unused latent dimensions and unmeasured biology unconstrained. A restricted probe can also miss recoverable information, so its failure is a failure of the declared readout rather than an information-theoretic impossibility proof.

For the chosen matching plan `pi`, define the state energy

$$
E_S(\pi)=\sum_{i,j}\pi_{ij}
 \left\|D^{-1}(S_i^{(a)}-S_j^{(b)})\right\|^2,
\qquad \sum_{i,j}\pi_{ij}=1.
$$

In the population, this nonnegative quantity is zero **if and only if** the coupling matches the same measured state almost surely. This is an immediate consequence of a nonnegative squared norm, not a new identification theorem. Finite independent assay samples need not contain exactly equal states: the true-state assignment supplies the attainable floor under the selected sample, metric, and no-reuse constraints. Show absolute error as well as excess error above that floor; improvement over random alone is insufficient for an acceptance claim.

There is also a deterministic finite-pool certificate for transport of any `L`-Lipschitz state function `g`. With standardized states `s_i,t_j`, nonnegative weights `a_i` summing to one, and a nonnegative row-stochastic kernel `W`, Jensen's inequality followed by the Lipschitz inequality gives

$$
\sum_i a_i\left\|g(s_i)-\sum_j W_{ij}g(t_j)\right\|^2
\leq\sum_{i,j}a_iW_{ij}\|g(s_i)-g(t_j)\|^2
\leq L^2\sum_{i,j}a_iW_{ij}\|s_i-t_j\|^2.
$$

No sampling independence, equal state marginals, or population limit is required for this inequality. `L` must refer to the same state metric used in the cost. For the proposed same-pool assignment comparison, use `a_i=1/n` and permutation `W` (or a doubly stochastic plan). Row stochasticity alone permits target-cell reuse, invalidating a comparison to the no-reuse assignment floor. The bound concerns deterministic state functions and true conditional means, not residual innovations or noisy counts. Unknown empirical state and an unsupported Lipschitz constant prevent empirical certification.

Let `C_theta` be that squared state cost for the actual embedding assignment and `C_*` its minimum over the same permitted permutations. For uniform random assignments the expected cost is available exactly:

$$
C_{\rm random}=n^{-2}\sum_{i,j}\|s_i-t_j\|^2
=n^{-1}\sum_i\|s_i\|^2+n^{-1}\sum_j\|t_j\|^2
-2\bar s^\top\bar t.
$$

Report `C_theta`, `C_*`, `C_random`, and `(C_theta-C_*)/(C_random-C_*)`. The ratio can exceed one. A zero or numerically negligible denominator is an unavailable, uninformative reference, not a perfect score. Empty pools are unavailable; a one-cell pool has no normalized comparison. Negative excess cost or `C_random<C_*` beyond a declared rounding tolerance indicates an inconsistent oracle, metric, weights, or pool. A high `C_*` remains a limitation even at a normalized gap of zero. Report each assay pair separately; the sum of independent pairwise oracle minima is not generally a jointly attainable three-assay optimum.

The frozen `match_one_to_one` minimizes **unsquared** Euclidean distances. Preserve it for candidate assignments, but compute the new oracle with squared distances; its old true-state assignment is not necessarily the squared-cost minimum. A concrete two-cell check uses source states `(0,0),(3,0)` and target states `(0,0),(-7/6,5*sqrt(11)/6)`. Their Euclidean cost matrix is `[[0,3],[3,5]]`: minimizing unsquared distance chooses a squared total of 25, while the squared-distance optimum is 18. This small distinction is necessary for the advertised nonnegative gap and lower bound.

The operator interpretation makes the biological target explicit. Push `pi` forward to a joint measure on source state `S` and target state `T`, and let

$$
(K_\pi h)(s)=\mathbb E_\pi[h(T)\mid S=s].
$$

For any square-integrable function `h`, measure

$$
E_h(\pi)=\mathbb E_\pi[(h(S)-h(T))^2].
$$

If source and target state marginals are equal, direct expansion gives

$$
E_h(\pi)=2\langle h,(I-K_\pi)h\rangle_{L^2(P_S)}.
$$

Thus the test checks whether the operator preserves specific biological state functions. Include all standardized state coordinates, which together separate states, and report cached candidate conditional-mean activities `mu_j(s)=E[V_j|S=s]` as biological functions. Constant functions, such as an independent sentinel's conditional mean, cannot establish correspondence. A finite set of noninjective biological means can miss state permutations, and a low error of an averaged/barycentric prediction can conceal incorrect donors; retain the pairwise energy above. If a function is `L`-Lipschitz in standardized state, direct substitution gives `E_h <= L^2 E_S`, linking a specified state tolerance to its mean-signal tolerance. The converse requires a stated lower-separation condition and must not be assumed for arbitrary focal means.

The cached conditional means are Monte Carlo approximations with saved standard errors, not exact zero-innovation trajectories. Carry that uncertainty into interpretation; do not turn their approximation error into an embedding failure. No new simulations are needed for these state and mean-function checks. The current [evaluate_probes](../../../../pipelines/simulation/integration_recoverability/evaluate.py) fits and tests a probe within one source embedding assay; its `cross_assay` label refers to a hidden target variable from another assay. It is not the fixed-decoder transfer experiment described here.

Choose acceptable state and mean-function tolerances before reading the confirmation result. There is no universal theoretical scGLUE consistency threshold, transfer `R^2`, or mixing score that certifies biology for these assays. Report results by assay, world, state direction, count regime, and retained population; preserve failures. Shared population support, stable measurement mechanisms across cohorts, complete measured state for the intended claim, and meaningful state scales are explicit assumptions. A common global chart change is harmless; an independent chart change in each assay is the ambiguity the transfer check must expose.

Held-out per-cell prediction errors support uncertainty conditional on a fixed model and a fixed simulation world. They are not independent biological-world replicates. One-to-one assignment creates dependence between matched rows, so a bootstrap of already matched rows is not automatically valid for the matching endpoint. Any resampling inference for that endpoint needs to resample the input pools and recompute assignment, with its target stated. The proposed first report can keep matching endpoints descriptive.

**What remains unidentified after a perfect pass.** Even if the real biological state were observed without error, the assay-specific laws `P_m(X_m|S)` would not generally determine their joint conditional copula. Write

$$
X_m=\mu_m(S)+\epsilon_m,\qquad
\mathbb E[\epsilon_m\mid S]=0.
$$

The data can then fix the state-dependent means and the separate conditional residual distributions while leaving

$$
\operatorname{Cov}(X_a,X_b)=
\operatorname{Cov}(\mu_a(S),\mu_b(S))+
\mathbb E[\operatorname{Cov}(\epsilon_a,\epsilon_b\mid S)]
$$

undetermined through its second term. Assuming conditional independence supplies one joint completion; it does not estimate that missing term. Zero conditional covariance is itself weaker than conditional independence outside special model families. For example, with fixed Gaussian residual marginals, different admissible cross-residual correlations give the same unpaired observations. Statistical matching literature therefore uses identified sets rather than unique estimates when the observation pattern leaves covariance parameters free.[^4]

The existing [ambiguity twins](../../../../pipelines/simulation/integration_recoverability/twin_controls.py) provide an especially strong local control: public training/validation/discovery files are checked for byte equality, and different hidden joint structures use the exact same public embedding. A pipeline may pass shared-state checks in both twins; it must not claim to distinguish their residual couplings from those public inputs. Different evaluator-only paired prediction scores do not overturn public-input equivalence.

For any later residual association claim, define the admissible set of joint distributions satisfying the observed assay marginals, permitted anchors, and explicit model restrictions. Report the range of the target over that set. A few alternative state-constrained permutations of existing cells provide useful coupling sensitivity examples; they are not a sharp identified set or a confidence interval unless the full defined optimization problem has been solved. With three assays, pairwise couplings also need a compatible joint completion. No state score, graph score, or stability across optimization seeds eliminates this issue.

On the real unpaired data, validation can support correspondence only at the resolution of independent, existing anchors or labels. Labels transferred through the integration under review cannot independently certify it; annotation-derived guidance and gene scores carry the same prior assumptions into training and evaluation. Passing synthetic checks establishes performance on the tested worlds, not a theorem transporting their state model to PBMC biology. The defensible output is a model qualified for specified shared-state analyses, with residual causal claims still conditional on additional assumptions.

[^1]: Moriakov, N., Adler, J., and Teuwen, J. (2020). *Kernel of CycleGAN as a Principle homogeneous space*. ICLR 2020, Proposition 2.1. Primary full text: https://arxiv.org/html/2001.09061v1. Accessed 2026-10-25.
[^2]: Cao, Z.-J., and Gao, G. (2022). *Multi-omics single-cell data integration and regulatory inference with graph-linked embedding*. Nature Biotechnology 40, 1458–1466. https://doi.org/10.1038/s41587-022-01284-4. The authors' score definition is also documented at https://scglue.readthedocs.io/en/stable/api/scglue.models.dx.integration_consistency.html. Accessed 2026-10-25.
[^3]: Sun, J., et al. (2025). *scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration*. Sections 2–3. Primary full text: https://arxiv.org/html/2510.24987v1. Accessed 2026-10-25.
[^4]: Ahfock, D., Pyne, S., Lee, S. X., and McLachlan, G. J. (2016). *Partial identification in the statistical matching problem*. Computational Statistics & Data Analysis 104, 79–90. https://doi.org/10.1016/j.csda.2016.06.005. Primary manuscript: https://pmc.ncbi.nlm.nih.gov/articles/PMC5423529/. Accessed 2026-10-25. Its covariance-completion treatment assumes a multivariate-normal family; the conditional-copula argument above is stated separately and follows directly from fixing only conditional marginals.

No code, frozen sources, fitted models, large binary files, or running simulations were changed or loaded for this review. Model fits, numerical validation, and real-data biological certification were not performed.
