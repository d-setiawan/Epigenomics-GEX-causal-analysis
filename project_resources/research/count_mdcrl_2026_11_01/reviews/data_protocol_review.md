# Independent data and evaluation review

Reviewed on 2026-11-01 by the data/evaluation agent before count-source fitting or reading its outcomes. The experiment is a development pilot on the existing fixed panels, not a new locked validation block. This review covers the frozen `../protocol.json`; it does not certify the source estimator or the Sturma identification assumptions.

## Data boundary and controlled comparison

The public inputs remain the 30 raw count columns, their feature IDs, opaque cell IDs, and allowed-feature library totals from `../../sturma_pilot_2026_11_01/data_pilot_v1`. No feature reselection, PCA, variance ranking, additional biological measurements, paired training cells, state labels, or new simulated worlds are introduced. Feature selection used the earlier full training pools; the additional 75/25 source-fit/calibration split is therefore a calibration split within an already selected training panel, not an independent feature-selection validation set.

The fixed training populations contain 602 RNA, 615 H3K27ac, and 614 H3K4me3 cells for scMTR-seq S1, and 4,000 cells per assay in each cached simulation. The 595 complete-case real test cells are paired only in evaluator files. The two simulations have 2,000 public-discovery cells per assay and separately stored two-dimensional state references; they have no permitted paired background-feature reference for these selected columns. The evaluator must not substitute focal-feature graph truth for a background latent graph.

The dimension-matched uncorrected estimator is the primary controlled comparison. Both arms use the same fit/calibration split, features, selected dimension, seeds, and candidate-matching gates. Native Sturma results are contextual because that earlier run used all training rows. A capped dimension must remain visibly capped; zero rank and abstention must remain possible. Bootstrap source stability and finite-grid fingerprint equivalence are descriptive screens, not posterior probabilities, false-discovery guarantees, or proof of full source-law equality.

## What Poisson correction assumes

For an externally meaningful positive exposure \(d_i\), the working measurement model is

$$
C_{ij}\mid\lambda_i,d_i\sim\operatorname{Poisson}(d_i\lambda_{ij}),
\qquad R_{ij}=C_{ij}/d_i.
$$

Conditional independence of features gives

$$
\operatorname{Cov}(R)
=\operatorname{Cov}(\lambda)
+\operatorname{diag}\!\left\{\mathbb E(\lambda/d)\right\}.
$$

The training average of \(C_{ij}/d_i^2\) estimates the second term under that model. Removing this diagonal is a measurement-model hypothesis, not evidence that every remaining mode is biological. An NB measurement term would generally leave additional variance proportional to squared rates; its magnitude cannot simply be inferred as technical noise from marginal overdispersion, which also contains biology. The distinction between measurement and expression models is established for RNA sequencing, but does not independently validate Poisson noise for these promoter fragment counts. [Sarkar and Stephens, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8370014/).

For two assays A and B, conditional independence of their measurement errors given both rates and exposures implies

$$
\operatorname{Cov}(R^A,R^B)=\operatorname{Cov}(\lambda^A,\lambda^B).
$$

This justifies comparing a hypothesized shared-rate covariance with paired rate covariance under the working assumptions. It does not identify the missing cross-assay coupling from disjoint training cells. Correlated assay noise, an endogenous denominator, selection, or a wrong rate model can violate the equality.

## Exposure and sparsity caveats

The primary exposure is one. Its retained signal can therefore include cell size, depth, and state-dependent detection. The sensitivity exposure is the original allowed-feature total minus the selected 30-feature sum, divided by its source-fit median. It is an observed proxy, not known technical exposure. It is computed before permutations and held fixed. A total conditional on observed molecules can instead induce multinomial dependence; dividing by a count-derived total is not guaranteed to preserve independent Poisson errors. [Townes et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6927135/).

The selected panel contributes a median of zero to both real histone library totals, and a median 0.188% to the real RNA total. All training leave-panel-out depths are positive. Selected promoter windows do not overlap one another within an assay, but other allowed promoter windows can overlap them. The full promoter total may count one fragment in more than one window, so subtracting the panel does not create an external independent depth measurement. The eight exposure-stratum permutations only approximate exchangeability and do not calibrate biological false discovery.

The real histone panels remain severely sparse: 341/615 H3K27ac and 317/614 H3K4me3 training cells have zero counts in all 30 selected columns. Each selected histone feature was detected in only 10–29 training cells. The 75/25 split and higher moment/factorial-CF estimation consequently require explicit variance, stability, and numerical gates. No failed gate justifies replacing a feature or choosing a seed retrospectively.

## Frozen evaluation contract

1. Fit artifacts and hashes are sealed before the separate evaluator opens held-out counts, pairs, or states. The evaluator verifies the data manifest, all public input hashes, feature ordering, fit/calibration row IDs and indices, and every run artifact hash. It writes a new `evaluation_v1` directory and never edits a fit artifact.
2. Within each assay, compare held-out **observed-rate** mean and covariance with frozen training predictions. The corrected prediction is the fitted signal covariance plus its training Poisson noise diagonal; the uncorrected prediction already contains observed variation and must not receive the diagonal again. Compare against the diagonal training observed covariance, with full-matrix and off-diagonal errors kept separate. This is held-out moment prediction, not a likelihood or posterior predictive check from a fully specified count distribution.
3. For real paired data, compare predicted and held-out rate cross-covariances after dividing both by the same source-fit observed-rate SD outer product. This quantity is training-standardized covariance, not Pearson correlation. Retain all 900 feature pairs per assay pair. Report improvement over a zero cross-covariance predictor, including negative improvement. A scale that is undefined causes an explicit unscorable status, not outcome-dependent feature removal.
4. The secondary Pearson comparison uses the predicted observed marginal variances, including measurement noise only in the corrected arm. Record how many empirical/model feature pairs have positive finite variances. A zero-variance secondary result is undefined, not perfect recovery. The secondary mask cannot change the primary score.
5. Pair bootstrap uncertainty resamples all 595 complete triplets jointly for 200 replicates with seed 3000 while keeping the model, exposure normalization, and source-fit SDs fixed. It describes paired-cell sampling variation conditional on the fitted model. It is not uncertainty over biological replicates or a correction for repeated model development.
6. In the simulations, apply the frozen unmixing matrix to held-out rate deviations. Compare every available noisy score with the existing two state coordinates, and separately summarize scores retained in candidate cross-assay matches. Preserve full correlation matrices and factor counts alongside maxima. A score is an observed noisy projection, not an inferred posterior biological state. With zero sources the state summary is unavailable, not success.
7. Paired exposure relationships, if reported, are evaluator-only nuisance diagnostics. They cannot be relabeled as a learned cross-assay baseline. No paired reference, state association, or test moment may change rank, matching, thresholds, preprocessing, or seed selection.

## Interpretation limits retained from the source cohort

The scMTR-seq source was previously selected on DNA coverage over six targets; this inherited selection cannot be undone by the current three-assay learner. The cohort is one mixed-day technical sublibrary with an internal complete-case cell holdout, and the histone features are promoter-only. Paired observed covariance is not causal ground truth. The saved holdout and cached development worlds have also been inspected in earlier iterations. Results may support or reject the bounded Poisson-correction hypothesis on these panels; they cannot establish general causal recovery, identify direct peak–gene effects, or serve as fresh confirmatory replication.

Subject to the declared gates and independent implementation checks, the frozen protocol is appropriate for this bounded comparison. The principal unresolved scientific risks are endogenous depth, sparse high-order moment estimation, misspecified conditional count noise, the affine independent-source rate assumption, and uniqueness of distribution-based source matching.
