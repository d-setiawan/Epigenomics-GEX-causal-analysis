# C01: count-native finite shared-state model validation

Astra approval, 2026-11-15, before implementation or new evaluator access.
This is a new count-model experiment, not P01 marginal bounds or a replay of
old fit hashes. The independently reviewed population result is Option B in
[repair proposals](../theory/REPAIR_PROPOSALS.md), qualified by
[independent proof review](../reviews/REPAIR_PROOF_REVIEW.md).

## Scientific question and model

Can independently fitted count-mixture distributions explain unseen counts
and supply stable, unambiguous shared-state correspondence using only mixture
weights in the permitted unpaired observations?

For assay a and finite state k, use positive effective rates lambda[a,k,j]
and positive weights pi[a,k]. Fit each assay separately. The Poisson primary
has independent feature counts conditional on state. A fixed NB shape 2
sensitivity uses the same mean parameterization and conditional feature
independence. No depth offset is imposed: effective count intensity is the
target, not biological activity or identified technical exposure.

Full multivariate Poisson laws identify finite mixing measures with distinct
rate vectors. Distinct common state weights then align states across assays.
Known-shape NB injection is composed with the same finite-atom argument, using
the archived known-kernel derivation; shape 2 is a declared model sensitivity,
not an estimate of true technical noise. The implementation and independent
review must check this kernel algebra. No common weights are imposed during
fitting, so matching them is not a circular goodness-of-fit check.

This is a finite shared-state alternative, not a claim that continuous
content/style has become identified without restrictions. The separate
continuous mean-contrast and nonlinear auxiliary-count repairs are mathematical
results with different assumptions; C01 does not pretend to validate those
assumptions. Actual cached worlds have continuous/noisy mechanisms, and S1
local QC can change state weights. Those are explicit misspecification and
population-transport questions, not hidden premises.

## Frozen inputs and model grid

- Reuse `count_extensions_2026_11_01/data_v1/public/{dataset}/train` for
  `scmtr_seq_s1`, `linear_chain_0`, `constrained_collider_0`.
- Assays: RNA, H3K27ac, H3K4me3. Two nested panels: the frozen first 30 and
  all 300 features; original feature IDs and hashes required.
- Keep the previous SHA256(`count-dica-v1:` + opaque ID) sorted 75% source-fit,
  25% calibration partition. Historical panel selection used the full training
  pool, so this split is not a fresh post-selection validation cohort.
- Kernels: Poisson and NB with fixed shape 2. K in {1,2,3,4}. Fit each assay
  independently with three restarts using seeds 0,1,2. Preserve every restart,
  failed/nonconverged fit and diagnostic; no evaluator-based model choice.
- Controls: original counts and independently column-permuted counts. Permute
  the full 300 columns separately within source-fit/calibration rows using
  seed 1000 and deterministic dataset/assay keys, then take the prefix. This
  preserves marginal counts across panel widths and breaks within-cell
  dependence. It is a diagnostic negative control, not a biological FDR null.

## Estimator and validation checks

Implement the ordinary finite-mixture likelihood with log-sum-exp EM. Use
source-fit log1p-count K-means (one initialization per declared seed) only to
initialize component means/weights; no PCA, state labels, pairing or outcomes.
The count means are constrained to at least 1e-8 to make predictive likelihood
defined on unseen nonzero counts. The M step uses responsibility-weighted raw
count means (also the optimum for fixed NB shape) and mixture proportions.
Record the floor and any dead component; do not silently prune/reduce K.

Maximum 300 EM iterations, average log-likelihood relative improvement tolerance
1e-7, and a monotonicity tolerance of 1e-8 times max(1,abs(previous objective)).
Select the converged restart with highest source-fit likelihood. An invalid
fit is a failure record, not a dropped row. K=1 supplies an explicit product-
count baseline. Numerical bugs may be fixed before the final source seal;
settings cannot change after evaluator inspection.

For each dataset/panel/kernel/control, select a common K only from the converged
independent assay fits by maximizing the sum of the three calibration average
per-cell log scores; ties within 1e-8 use the smaller K. All K/restart results
remain reported, and the independent per-assay preferred K is also recorded
as a model-compatibility diagnostic. Do not force a match by excluding a poor
assay. A common K is a tested modeling restriction, not proof of biology.

Before scientific fitting, unit/self-checks must verify: closed-form K=1 MLE;
NB/Poisson log mass against the installed numerical library; responsibility
normalization and monotone updates; count/feature/schema rejection; stable
zero handling; weight and component permutation invariance; and exact finite
mixture moment/completion algebra on a labeled deterministic three-state
fixture. No random benchmark world is generated for these checks.

## Stability and alignment without paired information

For the selected K>1, independently bootstrap source-fit rows within each
assay 32 times. Refit the complete mixture at fixed K/kernel/panel from the
point solution and one fresh deterministic seeded initialization, retaining
the better converged likelihood. This is conditional uncertainty at fixed
selected K/panel and a limited optimizer search, not a global MLE guarantee.

Align bootstrap components to their own assay's point solution by Hungarian
assignment on squared Euclidean distances between square-root rate vectors.
Keep the full fitted parameters/assignment and objective diagnostics. At least
26 valid replicates are required. Each point component needs effective source-
fit membership at least max(20,0.02*n_fit). A bootstrap is valid only if it
converged, has finite parameters and retains that same minimum membership.

Let d_k be the nearest distinct point component's distance in square-root rate
space. Require d_k>0 and the 90th percentile bootstrap displacement/d_k below
1/2 for every component. This checks local component stability; it does not
prove a biological state. Let epsilon_a be the 90th percentile over valid
bootstraps of max_k abs(pi_boot[k]-pi_point[k]). Require epsilon_a<=0.10 to
enumerate compatible assignments. Within-assay weight intervals pi[k] +/-
epsilon_a must additionally be disjoint before a unique state match is
accepted. Ties and near ties remain explicit ambiguity or abstention rather
than arbitrary rank sorting.

Across assays, two components are weight-compatible if their intervals
intersect. Enumerate all permutations for K<=4, and all consistent three-assay
assignments relative to RNA, checking all assay pairs. Save every compatible
assignment. This is a finite-data compatibility diagnostic, not an equivalence
test or proof that true weights are equal. No assignment means abstention;
multiple assignments mean explicit ambiguity; one means a unique model-based
candidate only when all within-assay intervals are separated. A singleton
with near ties is labeled near_tie_abstain. K=1 remains a baseline and is not a discovered shared state.

## Joint completion is an additional assumption

Component correspondence alone does not recover a same-cell joint. For an
explicit prediction arm, additionally assume conditional independence across
assays given shared state. For each compatible global assignment, average the
three matched fitted weight vectors equally and normalize; this supplies one
coherent joint mixture with the original fitted component rates. Report the
effect of this projection on within-assay calibration scores. Imposing this
completion is not evidence that its assumption holds.

Unique assignments give a conditional cross-count covariance prediction.
Ambiguous assignments retain all candidates and entrywise ranges; those ranges
are not themselves one joint law or calibrated confidence intervals. Empty
assignment sets remain abstentions, not zero-covariance discoveries. The
independent/product zero-covariance baseline is reported separately.

## Observable depth and biological interpretation diagnostics

Unit exposure does not make latent components biological states. Report their
total fitted rate, normalized rate profiles, and responsibility association
with the permitted observed library-depth field and outside300 count proxy.
Use original metadata for original and column-null fits, and label the null's
broken count-depth relation. These are descriptive nuisance diagnostics only:
they cannot choose K, match components, alter stopping or remove an unfavorable
result. Rate vectors differing mostly by amplitude may reflect technical depth
or cell size; a correlation with depth does not by itself distinguish those
possibilities. No hidden true exposure or biological state enters this check.

## Seal and separate evaluator

Freeze protocol, source, environment, all input hashes, model-selection and
alignment rules before fitting. Freeze all fitted parameters, selected K,
bootstrap diagnostics and complete joint predictions before evaluator access.
The evaluator verifies seals, and never feeds a result back into fitting.

1. **Within-assay counts:** existing public test arrays only after freeze;
   test counts remain original for both original and column-null training arms;
   proper predictive log score, zero frequencies, mean, diagonal and off-
   diagonal covariance errors. Use fixed source-fit observed-count scales;
   zero-scale coordinates are explicitly unscorable. Report full 300 panels
   and common first30 anchors separately. Distinguish independent mixture fit
   from common-weight completion. Predictive success does not prove assumptions.
2. **Biological paired reference:** existing S1 same-cell reference only after
   freeze. Verify assay-major feature ordering; compare conditional predicted
   cross-covariance and the zero baseline on full panels and common30. Report
   abstention/ambiguity coverage before errors. Optional intervals use 200
   complete-triplet row bootstraps with seed3000, conditioning on frozen fits;
   differences in QC populations and repeated holdout use remain explicit.
3. **Cached simulated state references:** evaluator-only descriptive association
   of frozen responsibilities with available continuous state coordinates,
   for every component and control. Do not score them against discrete class
   labels or the separate focal DAG, and do not choose a permutation from
   truth. If target-aligned truth is absent, report unavailable rather than
   inventing a graph score.

## Interpretation, acceptance and continuation

Acceptance requires independent implementation/algebra review and complete
numerical replay of the relevant outcomes. Improvements in count prediction,
stable components, weight compatibility and successful conditional joint
prediction are separate results. Lack of matching cannot prove absence of
shared biology; good matching cannot verify common selected populations or
conditional independence. Poisson versus NB and original versus column-null
comparisons help diagnose measurement/model problems without resolving every
alternative explanation.

If models fail, report where (within-assay predictive fit, optimization,
stability, common-weight compatibility, conditional joint completion). The
next scientific step must respond to that evidence and the continuing theorem
review; no threshold widening, test-driven seed selection or new worlds.
This experiment is one part of the stronger goal, not automatic completion
of the literature, continuous-theorem repair or final independent audit.
