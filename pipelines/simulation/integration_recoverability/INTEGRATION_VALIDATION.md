# Validate integration before causal recovery

_Development supplement started 2026-10-25; validation matrix completed 2026-10-26. Uses cached worlds and the existing three assays._

The separate [2026-10-31 count-input follow-up](COUNT_INPUT_FOLLOWUP.md) has
completed four GPU fits comparing fresh PCA/LSI and full-feature scGLUE encoders
on the same two cached selection worlds. It modestly improves state matching
in both worlds and includes alternative-method and data-viability reviews.
Its protocol, runtime correction, results and independent reviews remain
separate from this completed 32-fit study.

---

## Question and information boundary

Before testing a new integration setting with causal discovery, check whether its
shared representation and the matching procedure align the simulated biological
state. This separates a failure of integration or matching from a later failure
of causal recovery. State alignment alone cannot establish the missing
cross-assay residual relationship.

The executable [protocol](configs/integration_validation_v1.json) permits only
existing development-world RNA, H3K27ac and H3K4me3 counts. Encoders never receive
simulator state, focal innovations, hidden paired measurements, or planted edges.
Evaluator-only state is joined after truth-independent cell-pool selection and
embedding-only matching. Private probe/assessment cohorts and assessment worlds
are excluded from this supplement. No validation worlds are generated.

The earlier [frozen causal benchmark](README.md) remains a separate baseline
study. Its sources, worlds, assessment lock, failures and completed outputs are
retained. Results from that study do not select the new settings.

The 32-fit validation matrix and the prescribed two-stage baseline continuation
are complete. The [completion review](../../../project_resources/research/integration_validation_2026_10_25/reviews/objective_completion_final_audit_2026_10_26.md)
reconciles these outputs with the research, conditional theoretical guarantees
and independent reviews. The broader frozen benchmark remains incomplete.

## Small hyperparameter screen

Each alternative changes one parameter around the recorded benchmark configuration.

| Method | Setting | Changed value |
| --- | --- | ---: |
| scGLUE | `baseline` | Recorded benchmark baseline |
| scGLUE | `align_low` | `lam_align=0.01` |
| scGLUE | `align_high` | `lam_align=0.2` |
| scGLUE | `latent_small` | `latent_dim=10` |
| scMRDR | `baseline` | Recorded benchmark baseline |
| scMRDR | `beta_low` | `beta=2` |
| scMRDR | `gamma_low` | `gamma=5` |
| scMRDR | `adversarial_low` | `lambda_adv=5` |

These values come from released GLUE sensitivity settings and the scMRDR
two-/three-assay reference configurations; they are a small screen, not an
exhaustive optimization.[^1][^2] Other model, preprocessing and optimization
settings remain explicit in the protocol. Batch size stays fixed because
scMRDR's geometry-loss reduction depends on it.

| Stage | Cached worlds | Counts | Integration seeds | New fits |
| --- | --- | --- | --- | ---: |
| Selection | `linear_chain_0`, `constrained_collider_0` | Informative | 0 | 16 |
| Confirmation | `constrained_chain_0`, `unrestricted_chain_0` | Both regimes | 0, 1 | 16 |

Selection refits recorded baselines alongside alternatives under the same runtime.
Old cached baselines are reference diagnostics only. Confirmation fits only the
selected setting for each method, after `selected_settings.json` is written.
Worlds are the biological simulation units; regimes and optimization seeds are
repeated conditions. Confirmation reuses development worlds and is not a fresh
locked-world assessment.

scMRDR early-stops on public `val` counts and ranks candidates using those
cells' evaluator-only states. scGLUE early-stops on an internal 10% training
split and ranks using public `val` states. This is exploratory within-method
selection, not identically held-out cross-method evaluation. Confirmation
scores use the separate public `discovery` cohort. Full outcomes, including
failures, must be accounted for before a setting is selected.

## Measurements and selection rule

The primary score evaluates the actual frozen one-to-one Euclidean matching
operator. Mark–mark correspondence is induced by the two RNA-anchored matches,
not optimized separately. For each assay pair, let $s_i,t_j$ be true state
coordinates scaled by
one common, pooled training-state standard deviation. Let $\pi_\theta$ be the
permutation induced by matching the learned embeddings. Compute

$$
C_\theta=\frac1n\sum_i\lVert s_i-t_{\pi_\theta(i)}\rVert^2,
\qquad
C_*=\min_{\pi}\frac1n\sum_i\lVert s_i-t_{\pi(i)}\rVert^2,
$$

$$
C_{\mathrm{random}}=\frac1{n^2}\sum_{i,j}\lVert s_i-t_j\rVert^2,
\qquad
G=\frac{C_\theta-C_*}{C_{\mathrm{random}}-C_*}.
$$

$G=0$ reaches the finite-pool state optimum; $G=1$ equals the expected cost of
uniform random matching. This is not the fraction of correctly paired cells.
Values above one are allowed. An uninformative denominator produces an
unavailable score. Absolute costs remain necessary: a small gap does not make
a large oracle floor acceptable. The squared-cost oracle is computed directly;
the old unsquared Euclidean true-state matcher is not substituted for it.
Three pairwise oracle minima need not be attainable by one joint matching.

For each setting and world, take the worst of the three pairwise $G$ values.
Within each method, minimize its arithmetic mean across the two selection
worlds; ties use the setting name. Loss values from differently weighted
objectives are not compared for selection. The complete all-setting reports
are the selection inputs; interim reports cannot choose a winner.

Report these secondary checks separately:

| Diagnostic | What it checks | Limitation |
| --- | --- | --- |
| Fixed source-to-target state decoder | One source-fitted `StandardScaler` plus `Ridge(alpha=1)` transfers unchanged to each assay | Probe regularization and extrapolation can affect the result |
| Within-assay neighbor Jaccard | Retention of frozen RNA PCA / histone LSI neighborhoods | Input geometry includes technical variation |
| Cross-assay neighbor indegree and coverage | Hubs or collapsed neighbor searches before assignment | Low hubness does not prove correspondence |
| Shared-coordinate variance | Obvious collapse or strongly unequal coordinate scales | Variance alone is not biological validity |
| Fit/configuration/export receipts | Native objective, retained cells, checkpoint and input identities | Software validity is distinct from biological validity |

Measure hubness before one-to-one assignment, which otherwise makes incoming
degree one by construction. This concern is supported by a recent RNA/histone
integration benchmark.[^3] Do not apply paired-barcode metrics to these disjoint
cohorts. Guidance agreement and modality mixing are useful internal checks,
but neither independently identifies the biological correspondence.[^1][^3]

Pools intersect eligible IDs across the compared candidates and both cached
baseline references before equalization. An interim report can still lose
additional cells when another candidate is added; compare selected-ID hashes
before interpreting scores across separate reports. No state labels or graph
outcomes determine pool selection.

## Global matching and direct coordinate use

The primary score evaluates the encoder together with the fixed global matching
procedure. A source-trained decoder and a nearest-neighbor lookup query the
exported coordinates directly. These operations can give different answers.

For an analytic illustration, take independent scalar state pools with
$S,T\sim\operatorname{Uniform}[-1,1]$ and embeddings

$$
z_A(S)=(S,0),\qquad z_B(T)=(3T,c),\qquad c>0.
$$

Their Euclidean matching cost is $d(s,t)=\sqrt{(s-3t)^2+c^2}$, with

$$
\frac{\partial^2 d}{\partial s\,\partial t}
=-\frac{3c^2}{((s-3t)^2+c^2)^{3/2}}<0.
$$

Swapping crossing matches therefore lowers the cost. For equal finite pools
with distinct states, the unique optimum matches state ranks. Rank matching
also minimizes squared scalar-state cost, so $G=0$ whenever its denominator is
positive. The finite-pool oracle cost can still be nonzero.

An ideal source decoder $d_{\mathrm{state}}(z)=z_1$ is exact on assay A but
predicts $3T$ on assay B. Its population target score is

$$
R^2=1-\frac{\mathbb E[(3T-T)^2]}{\operatorname{Var}(T)}=-3.
$$

The exact value describes this ideal decoder; finite fitted Ridge scores also
depend on shrinkage, the estimated intercept and sample moments. Fixed-$k$
nearest-neighbor lookup from A can reach only the central third of B's state
support asymptotically, up to vanishing boundary effects. Coverage is at most
one third in that limit and can be lower. Global one-to-one matching uses every
target by construction.

The [independent derivation](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_constrained_chain_0_informative_seed0_diagnostics_crosscheck.md)
checks the finite-pool, population and sampling qualifications. This example
shows that accurate state matching can coexist with poor direct transfer and
concentrated neighbor use. It does not establish the cause of the actual
scMRDR results or identify biological state ordering from unpaired marginals.
The measured matching score therefore supports a claim about the specified
matching procedure; the other diagnostics retain their separate interpretation.

## A limited theoretical guarantee

For nonnegative weights $a_i$ summing to one, a nonnegative row-stochastic
matching kernel $W$, and an $L$-Lipschitz state function $g$, Jensen's inequality
and the Lipschitz bound give

$$
\sum_i a_i\left\lVert g(s_i)-\sum_jW_{ij}g(t_j)\right\rVert^2
\le L^2\sum_{i,j}a_iW_{ij}\lVert s_i-t_j\rVert^2.
$$

For uniform one-to-one matching, the right side is $L^2C_\theta$. Thus measured
state-matching error bounds the transport error of a specified smooth
state-dependent mean, with respect to the stated state metric. No sampling
independence or equal state marginal is needed for this finite-pool inequality.
Comparisons to the permutation oracle require the same no-reuse constraints.

There is also a bound on shared-state association. For scalar state functions
$f,g$, define finite-pool weighted covariance by
$\operatorname{Cov}_a(u,v)=\sum_i a_i(u_i-\bar u_a)(v_i-\bar v_a)$ and let
$\sigma_a(f)$ be the weighted standard deviation of $f(s_i)$. If $g$ is
$L_g$-Lipschitz in the same state metric and the weighted moments are finite,

$$
\left|\operatorname{Cov}_a\!\left(f(s),g(t_\pi)\right)
-\operatorname{Cov}_a\!\left(f(s),g(s)\right)\right|
\le \sigma_a(f)L_g\sqrt{\sum_i a_i\lVert s_i-t_{\pi(i)}\rVert^2}.
$$

To see this, write $\delta_i=g(t_{\pi(i)})-g(s_i)$. The covariance difference
is $\sum_i a_i(f(s_i)-\bar f_a)\delta_i$; Cauchy–Schwarz and the Lipschitz
bound give the result. Uniform weights make the square-root term
$\sqrt{C_\theta}$. No smoothness assumption on $f$ is needed. If $f,g$ are
conditional means, their specification and the constant must be justified;
approximation error is additional uncertainty. This controls distortion of
that finite-pool shared-state covariance, not residual covariance or population
generalization. Correlation stability also needs variance control; exact
conditional independences and graph preservation do not follow from the bound.

This does not bound cell-specific innovations, recover their conditional
copula, or preserve cell-level conditional independences. Even exact state
alignment leaves the cross-assay residual joint law unspecified. Distribution
matching, reconstruction and cycle consistency can also admit incorrect
biological correspondences.[^4] See the [independent theory review](../../../project_resources/research/integration_validation_2026_10_25/reviews/theoretical_validity_review.md)
for assumptions, counterexamples and the derivation. On empirical data, true
state and a valid Lipschitz constant are generally unknown; these simulation
checks are not a universal certificate of biological truth.

The [existing ambiguity control](RESULTS.md#observational-ambiguity-control)
provides a concrete finite-data check. A [fresh checksum audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/existing_ambiguity_control_public_check.json)
verified all 18 public count-file pairs and both previously reviewed comparison
reports. The available input files remain identical across counterparts with
different saved hidden-joint structures. A deterministic pipeline using those
inputs cannot discriminate these saved counterparts. Reused fits make this a
consistency check; the population identification argument is given in the
theory review above.

## Initial results and execution status

The cached `linear_chain_0` informative-count baseline audit uses 487 validation
cells and 1,936 training cells per assay, with identical pools for both methods.

| Cached reference | RNA–H3K27ac $G$ | RNA–H3K4me3 $G$ | Mark–mark $G$ | Worst pair |
| --- | ---: | ---: | ---: | ---: |
| scGLUE | 0.1093 | 0.1030 | 0.1420 | 0.1420 |
| scMRDR | 0.0282 | 0.0218 | 0.0142 | 0.0282 |

An [independent numerical crosscheck](../../../project_resources/research/integration_validation_2026_10_25/reviews/baseline_public_train_val_crosscheck.md)
reproduced all state costs, ID hashes and 18 declared decoder scores. Sorted-ID
and shuffled-match controls have gaps near one. The scMRDR standardized RNA
decoder has negative cross-assay $R^2$, while exploratory raw-coordinate Ridge
and nearest-neighbor probes transfer well. This is a probe-sensitive discrepancy,
not sufficient evidence that the entire integration failed. The declared score
and selection rule are unchanged.

### First selection world: all eight settings

The [complete first-world comparison](runs/integration_validation/20261025_v1/reports/linear_chain_0_informative_val_seed0_scglue-scmrdr_all_8366327f.json)
uses fresh fits on `linear_chain_0`, informative counts, seed 0, with 1,936
common training cells and 487 common validation cells per assay. The decoder
column is the minimum of the six off-diagonal source-to-target mean state
$R^2$ scores. It is a secondary diagnostic, not the setting-selection criterion.

| Method | Setting | Worst pair $G$; lower is better | Minimum transferred $R^2$ |
| --- | --- | ---: | ---: |
| scGLUE | `baseline` | 0.1461 | 0.8246 |
| scGLUE | `align_low` | 0.1924 | 0.5681 |
| scGLUE | `align_high` | 0.1240 | 0.8792 |
| scGLUE | `latent_small` | 0.1068 | 0.8467 |
| scMRDR | `baseline` | 0.0282 | -0.9759 |
| scMRDR | `beta_low` | 0.0266 | -1.2100 |
| scMRDR | `gamma_low` | 0.0268 | 0.3442 |
| scMRDR | `adversarial_low` | 0.0275 | 0.0975 |

These are descriptive results from one selection world, not selected settings
or a reliability estimate. The fresh scGLUE baseline differs from the cached
reference above; selection uses the fresh result. A smaller scGLUE latent
dimension gives the lowest matching gap here. scMRDR's beta-low setting gives
a small matching improvement alongside a worse fixed-decoder score. The
earlier baseline diagnostic establishes probe sensitivity, but that explanation
has not been separately tested for beta-low. Neither the low gaps nor a
positive decoder score proves correct residual coupling or causal structure.

The [independent all-setting numerical check](../../../project_resources/research/integration_validation_2026_10_25/reviews/all_settings_linear_chain_0_public_state_crosscheck.md)
reproduces all 24 state costs, squared-cost oracle floors, random reference
costs and gaps, plus eight worst-pair summaries, from saved assignments and
public train/validation states. Its maximum numerical discrepancy is
$8.88\times10^{-16}$. Native fit reviews cover the
[first scMRDR fit](../../../project_resources/research/integration_validation_2026_10_25/reviews/first_scmrdr_adversarial_low_independent_audit.json),
[first scGLUE fit](../../../project_resources/research/integration_validation_2026_10_25/reviews/first_scglue_align_high_independent_audit.json),
[gamma-low and fresh scGLUE baseline](../../../project_resources/research/integration_validation_2026_10_25/reviews/gamma_low_and_fresh_scglue_baseline_independent_audit.json),
[fresh scMRDR baseline and align-low](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_baseline_and_align_low_independent_audit.json),
and [beta-low and latent-small](../../../project_resources/research/integration_validation_2026_10_25/reviews/beta_low_and_latent_small_first_world_final_batch_audit.json).
All eight fits and the full both-method comparison passed their scoped reviews.

### Frozen two-world selection

All sixteen selection fits are complete. The immutable
[selected settings](runs/integration_validation/20261025_v1/selected_settings.json)
choose scGLUE `align_high` (`lam_align=0.2`) and the recorded scMRDR `baseline`
under the predeclared mean worst-pair-gap rule. Each column below uses that
world's common cell pools. These are development-selection scores, not
confirmation results or a binary verdict that an integration is correct.

| Method | Setting | Linear chain $G$ | Constrained collider $G$ | Mean worst-pair $G$ |
| --- | --- | ---: | ---: | ---: |
| scGLUE | `baseline` | 0.1461 | 0.1804 | 0.1633 |
| scGLUE | `align_low` | 0.1924 | 0.2314 | 0.2119 |
| scGLUE | `align_high` | 0.1240 | 0.1333 | 0.1287 |
| scGLUE | `latent_small` | 0.1068 | 0.2160 | 0.1614 |
| scMRDR | `baseline` | 0.0282 | 0.0301 | 0.0292 |
| scMRDR | `beta_low` | 0.0266 | 0.0381 | 0.0323 |
| scMRDR | `gamma_low` | 0.0268 | 0.0357 | 0.0312 |
| scMRDR | `adversarial_low` | 0.0275 | 0.0363 | 0.0319 |

The smaller scGLUE latent space was best on the first world but worse than the
baseline on the second. All three scMRDR alternatives slightly improved the
first-world gap and worsened the second-world gap. The two-world rule therefore
retains scMRDR's baseline; it does not require an alternative to win. These
observations do not estimate performance across a broad population of worlds.

The secondary decoder check still distinguishes the methods. For the selected
scGLUE setting, the weakest cross-assay fixed-decoder $R^2$ is 0.8792 on the
linear chain and 0.8557 on the constrained collider. For the selected scMRDR
baseline, it is -0.9759 and -0.9656, respectively. The primary state-matching
rule therefore selects a setting that still has a failed decoder-transfer
diagnostic. The earlier exploratory readout analysis explains sensitivity on
the first cached world only; it has not established the same explanation for
the second world. Selection is not a claim that every integration diagnostic
passes.

The [independent second-world and selection review](../../../project_resources/research/integration_validation_2026_10_25/reviews/all_settings_constrained_collider_0_public_state_crosscheck.md)
recomputes state costs, verifies exact agreement with the prior scMRDR-only
report, confirms all eight two-world means and the selections, and checks the
exact sixteen-job confirmation schedule. Second-world native-fit reviews cover
[align-low and beta-low](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_collider_align_low_and_beta_low_independent_batch_audit.json)
and [align-high and adversarial-low](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_collider_align_high_and_adversarial_low_independent_batch_audit.json),
[fresh scGLUE baseline and gamma-low](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_collider_baseline_and_gamma_low_independent_batch_audit.json),
and [fresh scMRDR baseline and latent-small](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_collider_baseline_and_latent_small_final_selection_batch_audit.json).
The second-world pools contain 1,934 training and 491 validation cells per assay.

The selected-settings file has SHA-256
`9c8bf9ccfc9d7e1d7d7cfa15953ef5359bc5aa3c394d5c7af7c606b00593b09b`.
The [confirmation plan](runs/integration_validation/20261025_v1/confirmation_execution_plan.json)
has SHA-256 `6de5226a2ee40bcf09098ac5a1720717243ef325543d1247520f2bfbb39d04ee`.
Confirmation has started with these fixed settings on the two other cached
development worlds, both count regimes and seeds 0 and 1. It remains incomplete.
The matrix continues serially;
the recorded [one-off driver](runs/integration_validation/20261025_v1/matrix_orchestration.py.txt)
reuses complete verified fits and stops on partial outputs. Its [observed launch receipt](runs/integration_validation/20261025_v1/matrix_execution_observed_start.json)
pins the driver, plan and runtime command for independent final verification.

Before confirmation, a separate [dependency preflight](../../../project_resources/research/integration_validation_2026_10_25/reviews/confirmation_cached_reference_preflight.json)
passed 701 checks across all 16 cached reference runs and eight train/discovery
geometry sets. Configurations, checkpoint/export bindings, public IDs and
PCA/LSI dependencies agree. This checks that confirmation can read its required
inputs; it does not open state labels or evaluate confirmation performance.

The combined [final-source software check](runs/integration_validation/20261025_v1/focused_tests_20261026_final_sources.json)
passes 75 tests and 25 subtests: the 53 validation tests plus 22 tests of the
separate baseline retry helper. Frozen scientific source pins remain unchanged.
These are software checks, not biological replications; model manifests are
not completed causal reports.

### First confirmation condition: constrained chain, realistic counts

The four fits for this condition are complete. All use identical selected IDs:
1,614 training and 766 discovery cells per assay. The table reports the pair
with the largest normalized gap for each fit, including its absolute costs.
Only the selected setting for each method is fitted here; these runs do not
measure the benefit of changing settings.

| Method | Seed | Worst pair | $G$ | $C_\theta$ | $C_*$ | $C_{\mathrm{random}}$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| scGLUE | [0](runs/integration_validation/20261025_v1/reports/constrained_chain_0_realistic_discovery_seed0_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.3408 | 1.3248 | 0.0409 | 3.8089 |
| scGLUE | [1](runs/integration_validation/20261025_v1/reports/constrained_chain_0_realistic_discovery_seed1_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.3278 | 1.2758 | 0.0409 | 3.8089 |
| scMRDR | [0](runs/integration_validation/20261025_v1/reports/constrained_chain_0_realistic_discovery_seed0_scmrdr_selected_8366327f.json) | RNA–H3K27ac | 0.1343 | 0.5501 | 0.0434 | 3.8159 |
| scMRDR | [1](runs/integration_validation/20261025_v1/reports/constrained_chain_0_realistic_discovery_seed1_scmrdr_selected_8366327f.json) | RNA–H3K27ac | 0.1326 | 0.5437 | 0.0434 | 3.8159 |

The secondary diagnostics measure different properties. Decoder minima are
over six cross-assay directions, using each direction's mean $R^2$ across the
two state coordinates. Neighbor-Jaccard ranges span the three assays.
Coverage and largest hub share summarize the six directed neighbor searches
before assignment, using 15 neighbors. Coverage is the fraction of target cells
reached at least once; hub share is the fraction of all neighbor slots assigned
to the most-used target. Both are fractions, not percentages.

| Method | Seed | Minimum cross-assay $R^2$ | Neighbor Jaccard range | Minimum coverage | Maximum hub share |
| --- | ---: | ---: | --- | ---: | ---: |
| scGLUE | 0 | 0.6627 | 0.1237–0.1747 | 0.6632 | 0.0328 |
| scGLUE | 1 | 0.6616 | 0.1317–0.1786 | 0.6658 | 0.0209 |
| scMRDR | 0 | 0.7726 | 0.0382–0.0780 | 0.9648 | 0.0064 |
| scMRDR | 1 | 0.7801 | 0.0379–0.0809 | 0.9674 | 0.0070 |

scMRDR has lower state-matching error, stronger weakest-direction decoder
transfer and less concentrated neighbor searches here. scGLUE retains more
of the input PCA/LSI neighborhoods. Input geometry includes technical variation,
so retention is not itself a biological-accuracy score. Unlike the informative
selection conditions, all cross-assay decoder scores are positive for scMRDR
in this condition. Neither this observation nor low normalized gaps establish
residual coupling or causal recovery. Two seeds on one world are repeated
optimization runs, not independent biological replications.

The [two-GLUE native audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_0_realistic_scglue_confirmation_seeds0_1_independent_audit.json)
checks configurations, sources, checkpoints and all public exports. The
[GLUE numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scglue_constrained_chain_0_realistic_confirmation_public_state_crosscheck.md)
reproduces both seeds' state costs. A separate
[representative seed-0 review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_0_realistic_seed0_representative_diagnostics_crosscheck.md)
reproduces both methods' embedding assignments, all decoder scores, neighbor
retention, hubness and coordinate variances, and scMRDR's state costs.
The [scMRDR seed-1 numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_constrained_chain_0_realistic_seed1_public_state_crosscheck.md)
reproduces its state costs and confirms identical selected IDs and state scales
across all four reports. The
[two-scMRDR native audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_0_realistic_scmrdr_confirmation_seeds0_1_independent_audit.json)
verifies both completed fits and all eighteen public exports. Both reached the
declared 200-epoch cap, retaining best epochs 194 and 195; neither triggered
early stopping. The other confirmation conditions are reported below; settings
stayed frozen throughout confirmation.

### Informative constrained chain: confirmation

All four selected-setting fits are complete. They use identical pools of
1,942 training and 955 discovery cells per assay. As above, the transport table
reports the pair with the largest normalized gap, with its absolute costs.

| Method | Seed | Worst pair | $G$ | $C_\theta$ | $C_*$ | $C_{\mathrm{random}}$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| scGLUE | [0](runs/integration_validation/20261025_v1/reports/constrained_chain_0_informative_discovery_seed0_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.1247 | 0.5061 | 0.0326 | 3.8286 |
| scGLUE | [1](runs/integration_validation/20261025_v1/reports/constrained_chain_0_informative_discovery_seed1_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.1501 | 0.6024 | 0.0326 | 3.8286 |
| scMRDR | [0](runs/integration_validation/20261025_v1/reports/constrained_chain_0_informative_discovery_seed0_scmrdr_selected_8366327f.json) | RNA–H3K27ac | 0.0366 | 0.1724 | 0.0318 | 3.8702 |
| scMRDR | [1](runs/integration_validation/20261025_v1/reports/constrained_chain_0_informative_discovery_seed1_scmrdr_selected_8366327f.json) | RNA–H3K27ac | 0.0442 | 0.2013 | 0.0318 | 3.8702 |

The secondary summaries use the same definitions as the realistic condition.

| Method | Seed | Minimum cross-assay $R^2$ | Neighbor Jaccard range | Minimum coverage | Maximum hub share |
| --- | ---: | ---: | --- | ---: | ---: |
| scGLUE | 0 | 0.8601 | 0.1706–0.2289 | 0.8115 | 0.0135 |
| scGLUE | 1 | 0.8937 | 0.1545–0.2259 | 0.7644 | 0.0138 |
| scMRDR | 0 | -3.5127 | 0.1007–0.1433 | 0.3707 | 0.0211 |
| scMRDR | 1 | -12.6399 | 0.0994–0.1652 | 0.2817 | 0.0305 |

scMRDR's lower matching error coexists with failed RNA-trained decoder
transfer to both marks in both seeds. Its four other transfer directions have
positive mean scores. In seed 0, the negative RNA-transfer means are driven by
the second state coordinate; in seed 1, both state coordinates have negative
scores. The minimum coverage is RNA→H3K4me3 for seed 0 and both mark→RNA
searches for seed 1. This describes concentrated donor use before assignment,
not the fraction of correct matches. The analytic illustration above explains
why such diagnostics can disagree without establishing this fit's mechanism.
Earlier exploratory raw-Ridge and nearest-neighbor decoder successes on the
cached linear-chain world have not been established for this condition.
The [independent interpretation review](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_informative_transport_vs_coordinates_interpretation.md)
also distinguishes assignment quality, source-decoder calibration and donor use.

Independent native-fit reviews cover
[both scGLUE seeds](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_0_informative_scglue_confirmation_seeds0_1_independent_audit.json)
and scMRDR
[seed 0](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_0_informative_scmrdr_confirmation_seed0_independent_audit.json)
and [seed 1](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_0_informative_scmrdr_confirmation_seed1_independent_audit.json).
The [scGLUE numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scglue_constrained_chain_0_informative_confirmation_public_state_crosscheck.md)
reproduces both seeds' state costs. The scMRDR
[seed-0](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_constrained_chain_0_informative_seed0_diagnostics_crosscheck.md)
and [seed-1 numerical reviews](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_constrained_chain_0_informative_seed1_diagnostics_crosscheck.md)
also reproduce all six cross-assay probe and hubness rows. A supplemental
float64 solver check differs from native float32 coordinate $R^2$ by up to
0.00167 and 0.00101, respectively; the strong negative conclusions persist.
Those checks do not replace the frozen metric. The four reports' pool IDs and
training scales match exactly.

Regimes differ in both count measurements and retained cells, so their score
difference does not isolate a count-depth effect on identical scored cells.
These are eight of sixteen confirmation fits on one biological world.
Settings remain frozen, with no new causal tests of these settings.

### Realistic unrestricted chain: confirmation

All four selected-setting fits are complete and use the same 1,567 training
and 756 discovery cells per assay. This condition exposes a large scMRDR seed
difference. Seed 1's two RNA-to-mark assignment costs exceed their respective
expected random-matching costs. Its worst normalized gap is 1.4446; the metric
is not capped at one, and this result is retained without retuning or retry.

| Method | Seed | Worst pair | $G$ | $C_\theta$ | $C_*$ | $C_{\mathrm{random}}$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| scGLUE | [0](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_realistic_discovery_seed0_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.3479 | 1.4732 | 0.0408 | 4.1583 |
| scGLUE | [1](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_realistic_discovery_seed1_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.3276 | 1.3896 | 0.0408 | 4.1583 |
| scMRDR | [0](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_realistic_discovery_seed0_scmrdr_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.1246 | 0.5540 | 0.0408 | 4.1583 |
| scMRDR | [1](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_realistic_discovery_seed1_scmrdr_selected_8366327f.json) | RNA–H3K4me3 | 1.4446 | 5.9774 | 0.0355 | 4.1486 |

The secondary summaries use the same definitions as the preceding conditions.

| Method | Seed | Minimum cross-assay $R^2$ | Neighbor Jaccard range | Minimum coverage | Maximum hub share |
| --- | ---: | ---: | --- | ---: | ---: |
| scGLUE | 0 | 0.6379 | 0.1450–0.1786 | 0.7526 | 0.0227 |
| scGLUE | 1 | 0.7375 | 0.1306–0.1618 | 0.5661 | 0.0239 |
| scMRDR | 0 | 0.7925 | 0.0463–0.0884 | 0.9696 | 0.0077 |
| scMRDR | 1 | -1.8756 | 0.0270–0.0620 | 0.7725 | 0.0086 |

Unlike the informative constrained-chain discrepancy, scMRDR seed 1 fails
both the state-matching diagnostic and all four RNA/mark decoder-transfer
directions. Its mark-to-mark decoder scores remain positive. Positive
within-assay prediction and broad neighbor coverage therefore do not establish
correct cross-assay correspondence. The two seeds are repeated fits to one
world, not an estimate of a population failure probability.

The [scMRDR native audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/unrestricted_chain_0_realistic_scmrdr_confirmation_seeds0_1_independent_audit.json)
passes 528 checks, including input, gene-map, configuration, source, runtime,
checkpoint and export bindings. All four reports have identical scored pools
and state scaling. Both scMRDR runs have 200 finite epochs; their best epochs
are 194 and 190. Seed 1 reaches its patience limit exactly at the epoch cap.
On the scored public embeddings, its leading covariance axis accounts for
97.7–98.3% of variance across assays, compared with 51.6–53.7% for seed 0.
This is evidence of learned representation concentration and seed sensitivity;
it does not identify the complete optimization mechanism or prove which state
information is absent. No audited software or input mismatch explains it.
The [independent numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_unrestricted_chain_0_realistic_seeds0_1_diagnostics_crosscheck.md)
reproduces all six pairwise state-cost rows, the exact embedding assignments,
all twelve cross-assay decoder rows and all twelve hubness rows across the two
seeds. The normalization denominators exceed 3.96, so a small denominator
does not explain the failure. An exploratory decomposition finds negative
matched correlations in both state coordinates for seed 1's RNA-to-mark
assignments; it describes the mismatch without establishing its training cause.

The [scGLUE native audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/unrestricted_chain_0_realistic_scglue_confirmation_seeds0_1_independent_audit.json)
and independent state-cost checks for
[seed 0](../../../project_resources/research/integration_validation_2026_10_25/reviews/scglue_unrestricted_chain_0_realistic_seed0_public_state_crosscheck.md)
and [seed 1](../../../project_resources/research/integration_validation_2026_10_25/reviews/scglue_unrestricted_chain_0_realistic_seed1_public_state_crosscheck.md)
passed. The latter is the preserved operational retry described below.
Informative unrestricted-chain confirmation is reported below. These results
do not justify calling either method universally
valid or treating its matched rows as recovered cell-level joint observations.

### Informative unrestricted chain: confirmation

All four fits are complete, using identical pools of 1,931 training and 957
discovery cells per assay. The tables retain both matching and coordinate-use
diagnostics, with the same definitions as the preceding conditions.

| Method | Seed | Worst pair | $G$ | $C_\theta$ | $C_*$ | $C_{\mathrm{random}}$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| scGLUE | [0](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_informative_discovery_seed0_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.1438 | 0.6267 | 0.0348 | 4.1509 |
| scGLUE | [1](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_informative_discovery_seed1_scglue_selected_8366327f.json) | H3K27ac–H3K4me3 | 0.1466 | 0.6381 | 0.0348 | 4.1509 |
| scMRDR | [0](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_informative_discovery_seed0_scmrdr_selected_8366327f.json) | RNA–H3K27ac | 0.0269 | 0.1367 | 0.0283 | 4.0568 |
| scMRDR | [1](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_informative_discovery_seed1_scmrdr_selected_8366327f.json) | RNA–H3K27ac | 0.0276 | 0.1394 | 0.0283 | 4.0568 |

| Method | Seed | Minimum cross-assay $R^2$ | Neighbor Jaccard range | Minimum coverage | Maximum hub share |
| --- | ---: | ---: | --- | ---: | ---: |
| scGLUE | 0 | 0.9084 | 0.1782–0.2259 | 0.8495 | 0.0109 |
| scGLUE | 1 | 0.7890 | 0.1748–0.2345 | 0.8150 | 0.0116 |
| scMRDR | 0 | 0.1730 | 0.0970–0.1857 | 0.3093 | 0.0180 |
| scMRDR | 1 | -1.9597 | 0.0994–0.1874 | 0.3197 | 0.0162 |

Both scMRDR seeds have low matching gaps. Seed 0 has positive but weak
RNA-to-H3K4me3 decoder transfer; seed 1 has negative RNA-to-both-mark decoder
scores, while its four other cross-assay directions remain positive. Thus the
matching result alone does not establish interchangeable latent coordinates.
The count regimes also retain different cells, so their comparison does not
isolate count depth on identical scored pools.

Independent scGLUE native audits cover
[seed 0](../../../project_resources/research/integration_validation_2026_10_25/reviews/unrestricted_chain_0_informative_scglue_confirmation_seed0_independent_audit.json)
and [seed 1](../../../project_resources/research/integration_validation_2026_10_25/reviews/unrestricted_chain_0_informative_scglue_confirmation_seed1_independent_audit.json).
The [combined numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scglue_unrestricted_chain_0_informative_confirmation_public_state_crosscheck.md)
reproduces all six pairwise state-cost rows within $2.7\times10^{-15}$ and
checks the common pools and scaling.

The [scMRDR seed-0 report](runs/integration_validation/20261025_v1/reports/unrestricted_chain_0_informative_discovery_seed0_scmrdr_selected_8366327f.json)
is also complete, with worst gap 0.0269 and minimum cross-assay decoder $R^2$
of 0.1730. Its [native audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/unrestricted_chain_0_informative_scmrdr_confirmation_seed0_independent_audit.json)
passes 290 checks; it reached the 200-epoch budget with best epoch 199, without
early stopping. The [numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_unrestricted_chain_0_informative_seed0_diagnostics_crosscheck.md)
reproduces state costs, assignments, all six cross-assay probe rows and all six
hubness rows. All cross-assay decoder scores are positive here, although the
RNA-to-H3K4me3 score is weak. Its pools and scaling match both scGLUE reports.
The [final-seed native audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/unrestricted_chain_0_informative_scmrdr_confirmation_seed1_independent_audit.json)
passes 376 checks and maps all 32 fits to native-review receipts. This seed
reaches patience 10 exactly at epoch 200, with best epoch 190. Its
[numerical review](../../../project_resources/research/integration_validation_2026_10_25/reviews/scmrdr_unrestricted_chain_0_informative_seed1_diagnostics_crosscheck.md)
reproduces costs, assignments, decoder scores and hubness, including the
negative RNA-to-mark results. A supplemental float64 calculation differs from
native float32 coordinate $R^2$ by up to 0.000463, beyond its separate comparison
tolerance; the negative conclusions persist. It does not replace the frozen
metric. All four reports' pools and scaling match. The controller exited
successfully with all 32 planned fits and sixteen confirmation reports complete.
The [final integrity review](../../../project_resources/research/integration_validation_2026_10_25/reviews/final_matrix_integrity_interpretation_2026_10_26.md)
passes its scoped checks of all 32 native fit chains, 288 public exports and
the frozen execution/report bindings. All 48 confirmation state-cost rows have
independent reproduction; independent public-array checks of secondary probe
and hubness values cover eight of sixteen reports. The remaining secondary
values come from the sealed reports and tested production metric.

### Descriptive comparison across confirmation conditions

These summaries span eight fits per method: two cached biological worlds,
two measurement regimes and two optimization seeds. They are descriptive
counts and ranges, not eight independent biological replications or estimated
failure probabilities. No numerical acceptance threshold was preregistered.

| Selected method | Worst-pair $G$ range | Minimum cross-assay $R^2$ range | Fits with $G>1$ | Fits with negative minimum cross-assay $R^2$ |
| --- | --- | --- | ---: | ---: |
| scGLUE, `lam_align=0.2` | 0.1247–0.3479 | 0.6379–0.9084 | 0/8 | 0/8 |
| scMRDR, recorded baseline | 0.0269–1.4446 | -12.6399–0.7925 | 1/8 | 4/8 |

scGLUE is more consistent in these tested matching and decoder readouts.
scMRDR often approaches the state-assignment optimum more closely, but its
realistic unrestricted-chain seed failure prevents a reliability claim across
these conditions. Good state matching and poor decoder transfer can also
coexist. Neither method is certified as recovering the missing cross-assay
residual joint distribution. Confirmation fits only the selected settings,
so it does not measure their improvement over newly fitted confirmation-world
baselines. No new-setting causal graph was used to reach these conclusions.
The [independent scientific synthesis](../../../project_resources/research/integration_validation_2026_10_25/reviews/confirmation16_scientific_interpretation_2026_10_26.md)
binds all sixteen reports to their numerical reviews and preserves the same
limits on interpretation.

### Controller interruption and preserved retry

At 01:58 UTC on 2026-10-26, the original matrix session returned exit code 143
during fine-tuning of realistic `unrestricted_chain_0` scGLUE `align_high`,
seed 1. Initial process scans returned no matches, but a later review established
that these scans and the archive guard ran in a sandbox process namespace.
They did not independently prove host-process absence before the archive.
The signal's cause is unknown; this is not a completed scientific result. The
[observed interruption](runs/integration_validation/20261025_v1/matrix_interruption_observed_20261026_015827.json)
records nine completed confirmation reports and unchanged frozen artifacts.

The independent [partial-fit inventory](../../../project_resources/research/integration_validation_2026_10_25/reviews/matrix_interruption_unrestricted_chain_0_realistic_scglue_seed1_recovery_preparation_2026_10_26.md)
found seven files totaling 42,317,600 bytes, with no completed fine-tune snapshot
or export seals. Completed-export recovery was inapplicable. A separate
[reentry review](../../../project_resources/research/integration_validation_2026_10_25/reviews/matrix_reentry_readonly_review_2026_10_26.md)
verified all 25 completed fit chains and 225 public exports.

At 02:19 UTC, only the interrupted fit directory was moved to its recorded
[archive](runs/integration_validation/20261025_v1/interruption_archives/20261026_015827/archive_mapping.json);
all seven files passed post-move size/hash verification. The unchanged original
controller then resumed with the same command, runtime and settings. The
[new observed-start receipt](runs/integration_validation/20261025_v1/matrix_resume_observed_start_20261026_after_sigterm.json)
records its live execution after all existing audit reports, selection and
confirmation plan passed exact-equality reuse gates. The retry's input file
and input fingerprint exactly match the archived original.

The [post-reentry audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/matrix_archive_reentry_unrestricted_chain_0_realistic_scglue_seed1_independent_audit_2026_10_26.json)
verifies the archive, retry inputs and unchanged frozen bindings. Host process
checks at 02:33 UTC found only the resumed controller and its current scMRDR
child, with no old/orphan candidates. This later observation does not prove
host absence at the earlier archive time; that audit limitation is retained.
Subsequent process guards must run in the host namespace.

The original launch receipt and failed attempt's evidence remain intact.
The [completed resumed computation](runs/integration_validation/20261025_v1/matrix_execution_complete.json)
reports 25 reused and seven newly executed fits, covering the original 32-fit
plan; the resumed controller exited with code 0. These are counts for the
resumed invocation. Final review must account for both attempts.

The [host process observation](runs/integration_validation/20261025_v1/matrix_host_quiescence_20261026.json)
passes the reviewed study-process guards and records an unrelated stopped
scGPT process that remained listed by the GPU monitor. That process was left
untouched. This establishes observed study quiescence, not an empty GPU context
list or guaranteed future exclusivity. The final independent matrix review
found no material blocker to the bounded baseline continuation below.

### Separate baseline continuation

The baseline suite has 118 completed model manifests. Its old assessment operator stopped
at a retained calibration failure in `linear_collider_3`. During continuation,
`constrained_chain_1` passed its declared count checks, but its realistic and
informative scGLUE stages encountered native CUDA failures. Those failed
attempts and their original partial outputs remain preserved. A process-local CUDA 13.0.96 preload
supports complete fits for both methods; the new scGLUE run also used
`CUDA_LAUNCH_BLOCKING=1`. Successful fits, including the original-setting retries, do not
establish the cause of the earlier baseline failure. The [runtime record](../../../project_resources/research/integration_validation_2026_10_25/reviews/cuda_runtime_isolation_2026_10_25.json)
preserves hashes and commands. Shared environments have not been modified.
The [bounded baseline retry helper](scripts/retry_constrained_chain_baseline.py)
has passed independent review and 22 focused software tests. Its real default
preview passed. After the completed matrix, independent review and host guards,
the helper [started on 2026-10-26](runs/integration_validation/20261025_v1/baseline_retry_observed_start_20261026.json).
Its [one-use journal](runs/suite/constrained_chain_1_retry_20261025_v1/session.json)
binds the unchanged original stage commands. Both old partial directories were
archived with inventory verification; failed receipts and logs remain intact.
Both realistic-count models and their original recovery evaluation completed.
The [realistic-stage outcome](runs/suite/constrained_chain_1_retry_20261025_v1/attempt_realistic/result.json)
records exit code 0 and complete scheduled artifacts. The report status is
`completed`, with no failed integrations or diagnostics; this is execution
success, not a claim of correct causal recovery. Its
[reviewed recovery results](RESULTS.md#constrained-nonlinear-chain-1-realistic-baseline-continuation)
remain limited. Both informative-count models and their recovery evaluation
have also completed. The [informative-stage outcome](runs/suite/constrained_chain_1_retry_20261025_v1/attempt_informative/result.json)
records exit code 0 and complete scheduled artifacts; its evaluation status
is also `completed`. The
[archive/start audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_baseline_retry_archive_start_independent_audit_2026_10_26.json)
passes 125 checks, including all four archived files and original launch
bindings. The completed
[scGLUE audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_realistic_scglue_original_baseline_completed_fit_audit_2026_10_26.json)
passes 181 checks of the original configuration, checkpoint and exports.
Its alignment weight remains the original 0.05. All four native-fit reviews
have passed. The [helper's terminal result](runs/suite/constrained_chain_1_retry_20261025_v1/result.json)
is `completed_two_retries`, and its execution session exited with code 0.
The [terminal host observation](runs/integration_validation/20261025_v1/baseline_retry_terminal_host_observation_20261026.json)
checks the preserved input bindings and finds no remaining study worker or
held suite/GPU training lock. It leaves the unrelated stopped process untouched
and does not assert an empty global GPU process list. The
[final helper/stage audit](../../../project_resources/research/integration_validation_2026_10_25/reviews/constrained_chain_1_baseline_retry_final_ledger_and_informative_stage_binding_audit_2026_10_26.json)
passes 330 scoped checks, including all ten scheduled artifacts and four
native-audit chains. The [informative recovery results](RESULTS.md#constrained-nonlinear-chain-1-informative-baseline-continuation)
also have an independent scientific review. Both learned informative count
graphs return zero cross-assay edges, while paired controls recover only one
of twelve target cross-assay adjacencies. Neither regime yields a correct
directed biological edge from either learned integration in the separately
labeled identifiability stress test.
These completed retries do not establish the cause of the earlier CUDA
failures or complete the original 178-fit benchmark. The one-use helper must
not be restarted; the old failed receipts remain part of the record.

## Running and inspecting the supplement

Run from the repository root with the existing `scrna_r` environment. Training
uses the suite lock and GPU lock, refuses existing candidate outputs, and
writes to `runs/integration_validation/20261025_v1/`. The process-only runtime
override is recorded per fit; its library currently lives under `/tmp` and
must be revalidated if missing or changed.

```bash
python pipelines/simulation/integration_recoverability/scripts/validate_integration.py audit \
  --world linear_chain_0 --regime informative --candidates baseline

python pipelines/simulation/integration_recoverability/scripts/validate_integration.py audit \
  --world linear_chain_0 --regime informative \
  --method scmrdr --candidates adversarial_low

# Requires every declared selection fit; freezes settings before confirmation.
python pipelines/simulation/integration_recoverability/scripts/validate_integration.py select
```

The fit bridge reuses native adapters and frozen transformations. Reports bind
inputs, checkpoints, configurations, export hashes and analysis sources. See
the [evidence review](../../../project_resources/research/integration_validation_2026_10_25/reviews/primary_evidence_and_endpoints.md)
for verified sources and additional diagnostics that are proposed rather than
implemented in this first screen. No candidate in this supplement has entered
new causal discovery.

[^1]: Cao and Gao (2022), GLUE. https://doi.org/10.1038/s41587-022-01284-4. Released sensitivity grid: https://github.com/gao-lab/GLUE/blob/5a30146c6148a52210f547c9e9c330ecfbdf1090/evaluation/config/config.yaml
[^2]: Sun et al., scMRDR, NeurIPS 2025, Appendix A.1. https://arxiv.org/html/2510.24987v2
[^3]: Naqing, Yuan and Duren (2026), *Benchmarking component choices for unpaired single cell RNA and epigenomic integration*. https://pmc.ncbi.nlm.nih.gov/articles/PMC13192178/
[^4]: Moriakov, Adler and Teuwen (2020), *Kernel of CycleGAN as a Principle homogeneous space*. https://arxiv.org/html/2001.09061v1. For statistical matching under an explicitly assumed Gaussian family, see Ahfock et al. (2016): https://pmc.ncbi.nlm.nih.gov/articles/PMC5423529/.
