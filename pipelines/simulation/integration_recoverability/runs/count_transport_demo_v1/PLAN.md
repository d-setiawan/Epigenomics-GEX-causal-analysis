# Count-transport demonstration: frozen plan

Prepared 2026-11-15, before fitting or reading paired assessment count arrays.

## Question and scope

Can an estimator built around E/E2 recover useful cross-assay count dependence
from unpaired training cells in two existing simulation worlds? This is a
finite-sample, misspecified-model demonstration, not another proof of E/E2 or
a test of causal DAG recovery. No new worlds or measurements are generated.

The worlds are `linear_chain_0` and `constrained_collider_0`, informative count
regime. The fixed background feature prefixes contain 30 RNA, 24 H3K27ac and
18 H3K4me3 features, using the historical ID-ordered panel. Planted causal
features were excluded by that panel. No new feature ranking is performed.
Historical feature eligibility used the entire original training pool, so
calibration is independent of fitting conditional on that preselected panel,
not independent of every historical preprocessing decision. These are reused
development worlds, not newly locked external validation data.

## Estimator

Enumerate independent standardized Bernoulli sources, with learned probabilities
strictly below one half and common across assays. Fit exact product-Poisson
marginal likelihoods, summing over all source states. Shared probabilities
connect assays; no paired rows, latent labels, guidance or causal truth enter
fitting. All assays use separate loading matrices, without PCA.

Primary strict model: `log R_c = M_c (Z + mu)`. This is a parametric submodel
of E2 when source probabilities are distinct and every loading is full rank.
Sensitivity model: `log R_c = b_c + M_c Z`, with centered independent asymmetric
sources. Its affine identification needs the explicit stronger centered-law
assumption; it is not automatically the same theorem as E2.

Fit dimensions 1, 2 and 3, with starts 0, 1, 2 and 3. Strict initializations
alternate the first source's polarity by seed parity, preserving initial mean
log rates. Bounds and optimizer settings are recorded in the frozen model
source: L-BFGS-B, at most 600 iterations, fixed convergence tolerances. Each
assay contributes its mean per-cell, per-feature log likelihood equally.

The existing source law includes a Gaussian coordinate, private noise and
nonlinear observation transformations; the realistic biological mechanism does
not satisfy this finite Bernoulli model exactly. Library depth is not supplied
as a known exogenous exposure. A failed fit does not refute E/E2, and a good
fit does not verify its assumptions or make them biologically true.

## Selection, controls and checks

Reuse the historical identifier-hash 75%/25% split: 3,000 fitting and 1,000
calibration cells per assay. Select the highest equally weighted calibration
likelihood among converged finite candidates. Exact ties prefer lower dimension,
then lower seed. If no fit converges, retain the best finite diagnostic and
exclude it from a recovery claim. If all candidates fail, stop before assessment.
Do not refit using calibration cells. Do not select using rank or source-gap gates.

Acceptance requires finite rates/likelihood, optimizer convergence, loading
smallest/largest singular-value ratio at least 0.001 in every assay, and minimum
source-probability gap at least 0.03 when dimension exceeds one. Report boundary
hits and calibration improvement over independent Poisson as diagnostics.
These are numerical screens, not tests of the actual biological assumptions.

Repeat fitting after independent within-column permutations, separately inside
the fitting and calibration partitions. These destroy within-assay dependence
while preserving feature marginals. The same permuted counts are reused across
all candidate dimensions and seeds. Assess these null-trained fits on the same
original paired assessment cohort, reporting them as negative controls.
Total: 2 worlds x 2 controls x 2 variants x 3 dimensions x 4 starts = 96 fits.

## Sealed evaluation

Pin all code, tests, plan, feature identities, input manifests and data hashes.
Save all 96 outcomes, calibration selections and four starts at each selected
dimension before opening paired assessment counts. The evaluator verifies
hashes and unique IDs, training/assessment disjointness and exact paired assay
row order. Use the existing 2,000 paired assessment cells per world only here.

For all six ordered assay pairs, report conditional predictive log-likelihood
gain over the identical fitted target marginal, both per cell and per target
feature. Also report conditional-mean MSE versus the fitted target mean,
cross-assay covariance RMSE standardized by fitting-cell feature standard
deviations, a zero cross-covariance baseline and count-correlation cosine.
The per-cell log-score gain is symmetric under reversing an assay pair: it
equals joint log likelihood minus the two marginal log likelihoods. Thus six
ordered gains contain three unique association statistics, not six independent
results or evidence of causal direction. Per-target-feature normalization and
conditional mean errors can differ by prediction direction.
Compare true source rows against a fixed source-row shuffle. The observed paired
count covariance is a finite-sample reference, not an oracle latent-rate covariance.

Use 1,000 paired-row bootstrap replicates, common across assay directions within
a world. Report 95% and Bonferroni family-adjusted intervals across the 12
primary directions. These intervals condition on the frozen fit; they exclude
parameter and model uncertainty. At this bootstrap size the adjusted tail
quantiles are approximate. Shuffled-score intervals are descriptive because a
fixed permutation links rows. Affine results, controls and restart ranges are
descriptive sensitivity analyses. No new selection uses assessment values.

Keep all four starts at each selected dimension visible. A convincing pilot
recovery claim needs positive adjusted predictive-gain intervals, covariance
error below the zero-covariance baseline, rejection of source-shuffled gains,
valid fits and consistent results across restarts. Report failures and partial
successes separately. No result on these two background panels establishes
reliable genomic causal adjacency or direction recovery.

## Reproduction

Run `count_transport_run.py freeze`, then `count_transport_run.py fit`, then
`count_transport_evaluate.py`, using the existing `scrna_r` environment with one
BLAS thread per worker. The fit runner uses two CPU workers. Outputs are created
exclusively and cannot overwrite sealed prior results. No GPU is required.
