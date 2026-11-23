# Independent evaluation design review

The scientific protocol fixes 25 variants across six named extension families, three existing datasets, two original source seeds, and one independent-column control seed: 225 arms. This is a bounded prespecified comparison, not every mathematically possible extension. No held-out performance is computed until all 225 fit records and source/data/artifact hashes are sealed.

The separate evaluator imports no estimator or fitting runner. It independently verifies source-fit/calibration indices, common original-300 depth, nested full-300 permutations, training means and observed-rate scales, and Poisson/NB/retention noise restoration. An independent convex derivative solution checks the frozen one-dimensional depth elasticity; the exported fitted coefficient is used for every prediction. This check does not choose a different coefficient or refit a source model. The historical evaluator contributes only pinned serialization, metric, correlation, and split utilities.

Both moment-only and finite-grid-CF-confirmed candidate predictions are reported separately. Neither establishes equality of full biological source distributions. The primary moment-only rule also differs from the earlier pilot's combined CF requirement and must not be presented as a strict recovery-rate improvement under unchanged criteria. Unassessed CF endpoints are not CF agreement. Absent source matches produce an explicitly labeled default zero predictor, not evidence of independence.

Within-assay prediction compares observed-rate covariance with the model's signal covariance plus its declared measurement-noise diagonal. The ordinary ablation receives no additional noise term. The evaluator independently reproduces that diagonal from the known-kernel sensitivity parameters:

$$
\widehat N_j
=\frac1n\sum_i\left[
\frac{Y_{ij}^2}{t_i^2}
-\frac{qY_{ij}(Y_{ij}-1)}{(1+1/\theta)t_i^2}
\right].
$$

For \(q=1,\theta=\infty\), this is the Poisson noise term. Sensitivity choices for \(q\) and \(\theta\) are not estimates of a uniquely identified technical-noise process. Full covariance and off-diagonal errors are kept separate, with the diagonal source-fit observed covariance as a simple baseline.

Every real pair score compares predicted and empirical covariance on the same rate scale, divided by fixed source-fit observed-rate SDs. All feature pairs are retained: 900 for the original-30 block and 90,000 for a full 300-feature pair. A zero or nonfinite fixed scale makes the relevant score undefined rather than deleting features. The original-30 block is the controlled width comparison; whole-panel averages describe different feature sets. The evaluator additionally verifies that original-30 scales agree across width, noise, and orientation variants with the same data/depth/control.

The 200 paired bootstrap replicates use one resampled set of 595 complete triplets at a time. To avoid storing covariance cubes, the evaluator uses exact row-Gram identities. For centered, fixed-scale matrices \(X,Y\) and row multiplicities \(w\),

$$
C_w=X^T\left[\operatorname{diag}(w/n)-ww^T/n^2\right]Y.
$$

Writing \(G_X=XX^T\) and \(G_Y=YY^T\),

$$
\|C_w\|_F^2
=\frac{w^T(G_X\circ G_Y)w}{n^2}
-\frac{2\sum_i w_i(G_Xw)_i(G_Yw)_i}{n^3}
+\frac{(w^TG_Xw)(w^TG_Yw)}{n^4}.
$$

The prediction inner product is evaluated with the same weighted centering, then combined with the fixed prediction norm to obtain RMSE. Tests compare this calculation against explicit complete-pair resamples at 3, 30, and 300 columns. The intervals describe cell resampling conditional on the selected model and existing test population; they are not biological replication, multiplicity-adjusted inference, or posterior uncertainty.

Simulated state evaluation uses frozen noisy projections of original test rates. All-source, moment-matched, and CF-confirmed subsets are reported with full correlation matrices and source counts; maxima have factor-selection multiplicity. These scores are not posterior biological states, original-feature graph truth, or causal directions. No private paired background truth is introduced.

An independent reviewer checked the formulas and data boundaries without finding a blocking error. The reviewer highlighted that pairwise unique matches can be inconsistent across three assays. Their blocks therefore need not assemble into a positive-semidefinite joint intensity covariance. Reports describe pairwise moment predictions; a coherent joint distribution is not established. All-assay graph extraction remains limited to the separately recorded sign-consistent cliques.

The existing source-selection, sparse detection, overlapping-promoter, endogenous-depth, single-sublibrary, and repeated-development-use limitations remain. Added features and additional kernels can test hypotheses within this data contract; they do not turn the real paired reference into causal truth.
