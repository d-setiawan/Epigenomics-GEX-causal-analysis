# Independent review of the Sturma pilot

Reviewer: `causal_reliability_pilot`  
Date: 2026-11-01  
Scope: read-only paper/source inspection; no fits or new worlds on this reviewer's side.

The authors supply working implementations of both source matching and shared latent graph estimation. This is a practical algorithm to try, but applying it to our existing counts is a model-mismatch pilot rather than an in-class validation of its identification theorem. The native simulation replication and the existing-data application need separate conclusions.

## Source and implementation provenance

The [official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html) link the [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf) and a [supplementary code archive](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Supplemental-Conference.zip). I independently downloaded that archive for inspection; SHA256: `571d62925b1181fb5d1b57c7ed24a771e9db806e40c26c0b1a90231741fdd73c`.

The local [upstream source](../upstream/README.md) is from [NilsSturma/multi-domain-crl](https://github.com/NilsSturma/multi-domain-crl), commit `578b5900590164b2ed1750fb6a7f98ae7153e220`. Its learner, scoring, experiment runner, generator, utilities and inspected run configuration are byte-identical to the official archive. The inspected dependency difference is SciPy `1.8.0` in GitHub versus `1.7.3` in the archive. Both pin scikit-learn `1.2.1` and NumPy `1.21.5`; our runtime differs and must be recorded.

Core learner SHA256: `4cf4627befcc2ac3c11dc16a0929bd88a442fc1fb0d259e23bbe794cb5ab5868`. Core scoring SHA256: `6181c44e3b8d478244981ef30a907c811be361f81cb6f3734ee230e565206aa0`.

## What is actually implemented

`LinearMDCRL.fit()` in [learner.py](../upstream/src/learner.py) takes separate cells-by-features matrices. It fits FastICA within each assay, standardizes recovered components, matches component distributions across assays with sign-aware KS comparisons, assembles a joint mixing matrix, and calls graph recovery when at least one shared component was selected. Its normal fit path does not invoke the separate integer-programming code or require Gurobi.

The fit does **not require the true shared or private latent dimensions as inputs**. However, `get_sources()` uses the numerical rank of each empirical covariance matrix, with NumPy's default tolerance, as the ICA dimension. It does not implement the paper's stated `gamma`-threshold rank selection there. The noiseless low-rank simulation makes this distinction much less consequential than noisy counts. Any latent cap or alternate rank rule is an adaptation.

Matching uses mutual row/column maxima of KS p-values above the adjusted threshold, followed by cross-assay consistency. Despite its function name, `maximum_matching()` is not a general maximum-cardinality matching solver. Graph recovery selects approximately proportional observed rows, heuristically orders the resulting matrix, normalizes it, and computes `A = I - inverse(B_star)`. These are executable finite-sample procedures, rather than a graph theorem with no implementation.

The matched ICA coordinates represent **exogenous source disturbances**. They are not automatically the causally related latent-state coordinates. Graph recovery locally changes column order and signs without persisting those maps. A heldout encoder must save the training transformations, and any correspondence between exported coordinates and graph nodes must be made explicit. The native learner does not retain ICA estimator objects/training means or expose a new-cohort `transform()` method.

## Identification and existing-data scope

The theorem assumes a linear latent SEM, linear full-column-rank observations, independent distinguishable nonsymmetric standardized disturbances, and separated shared/private blocks for graph recovery. Two partial pure observed children per shared latent and rank faithfulness are additional graph conditions. The observed joint law is identified exactly; latent labels and coefficient signs have the stated ambiguities. See Definition 2.1, conditions C1–C4 and Theorems 3.1/4.4. [Official paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf)

The data reviewer independently checked our two cached worlds: their two state factors have no causal edge between them, one is Gaussian, and backgrounds include nonlinear state responses, Gaussian private/feature noise and negative-binomial measurement. The `linear_chain` name describes a focal mechanism; it does not make the complete observation model a Sturma model. Comparing the inferred latent graph to that focal feature DAG would compare different targets.

For actual counts, centering does not turn a nonlinear noisy measurement process into a noiseless linear mixture. Log transforms, depth normalization and feature selection also need their own stated interpretation. A count panel can have full numerical rank while its biological state is low-dimensional. Low-rank truncation can help estimation but changes the native procedure; internal ICA whitening should not be described as avoiding all spectral processing. A hash-selected background panel avoids outcome-based feature selection, but does not guarantee the pure-child condition.

## Evaluation issues to preserve in the report

| Issue | Verified behavior and implication |
|---|---|
| Conditional graph scoring | [experiments.py](../upstream/experiments/experiments.py) records graph error only when the estimated shared dimension equals truth. Report that denominator and all dimension failures alongside the conditional score. |
| Continuous coefficients versus recovered edges | Native output is a coefficient matrix. Approximate ordering/inversion need not produce exact triangularity, exact zeros, or a DAG. No adjacency threshold or precision/recall evaluation is supplied by the main runner. |
| Mixing-error normalization | [utils.py](../upstream/experiments/utils.py), lines70–71, divides the Frobenius error by the number of compared entries. The paper's displayed score uses its square root. Preserve the native score and label any paper-formula correction separately. |
| Permutation scoring | Native scorers select a permutation using absolute entries before optimizing signs. This is a heuristic for the stated signed-permutation objective. A tiny exhaustive check can assess parity without replacing the native replication score. |
| KS interpretation | The accepted event is failure to reject equality. Lowering the threshold protects true matches from false rejection; it is not a calibrated false-sharing FDR. Estimated ICA sources also differ from directly observed IID sources. |
| Empty and failed outcomes | Zero shared factors, ICA convergence warnings, singular inversions, invalid child selections and graph exceptions must remain visible. None is an exact-DAG recovery success. |
| Row-shuffle control | Independent row permutations preserve the unpaired block distributions this method intentionally uses. They are not a meaningful source-matching null. |

Two further static source hazards were sent to the implementation lead. `get_duplicates()` does not check every overlap between candidate row pairs; for example, two pairs sharing only their second endpoint can pass its check. Candidate exhaustion can subsequently produce an out-of-bounds index. Also, `score_rows()` uses `1/(sigma_min + 0.1)`, so its comparison with `1/gamma` differs from a literal singular-value threshold of `gamma`. These findings support explicit failure recording; this review does not authorize silently modifying the native arm.

## Recommended interpretation of this pilot

The reduced native replication should report shared-dimension recovery, unconditional completion/failure counts, and conditional coefficient errors with their exact normalization and alignment conventions. Seed both the generator and ICA; retain warnings and exceptions. The paper's study is a small synthetic proof of concept, with graph error conditional on the shared count being correct. It is not a published validation of direct peak–gene edges from raw count assays. [Paper, §5 and Appendix D](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf)

The existing-data application can report reproducibility, heldout reconstruction or state association, and dependence agreement with an available paired evaluator. These outcomes do not supply latent-DAG ground truth for the real data. Keep truth, states and pair identifiers in evaluation; freeze transformations and thresholds before outcome inspection. A failure outside the assumed class does not refute the theorem, and a stable fitted graph does not verify its assumptions.

No upstream source, existing cohort or benchmark setting was changed by this review.
