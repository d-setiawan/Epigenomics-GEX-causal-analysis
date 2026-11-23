# Reduced replication of the published Sturma model

All 60 planned fits completed using the unchanged upstream generator, learner and scorers. The method recovered the correct shared dimension in 43/60 fits. The predeclared coefficient criterion was met in 21/60 fits. These are results on the publication's restricted continuous linear model, not on biological counts or integrated assays, and they do not establish reliable original-feature causal recovery.

The [protocol](published_replication_protocol.json) was frozen before fitting, with SHA256 `13629b82a4056685cc8cb0ed065a06da9f83ffb4b35642c301defbdd7b5a7a84`. The [runner](replicate_published_model.py) imports upstream commit `578b5900590164b2ed1750fb6a7f98ae7153e220` directly. No upstream source or environment was changed.

## Design and result

The shared latent dimension is three, with two private latent variables per domain. We used two or three domains, a total of 30 observed features, the original nine error distributions, graph density 0.75, mixing density 0.9, the original two-partial-pure-child construction, KS matching with alpha 0.05 and gamma 0.2. Ten fixed seeds 0–9 were run for each sample-size/domain setting. NumPy and Python RNGs were explicitly reset before generation and again before ICA. Seeds repeat across settings, so the six settings are not independent samples of models.

The unconditional coefficient criterion is: correct shared dimension, a finite graph coefficient estimate, and native graph error divided by three at most 0.1. Missing or failed graph estimates count as failure. This descriptive criterion was specified before fitting; it is **not exact DAG recovery**.

| Domains | Cells per domain | Correct shared dimension | Coefficient criterion | Graph median error, conditional on correct dimension | Conditional graph sample size |
|---|---:|---:|---:|---:|---:|
| 2 | 1,000 | 5/10 | 0/10 | 0.5431 | 5 |
| 2 | 10,000 | 10/10 | 6/10 | 0.0549 | 10 |
| 2 | 25,000 | 10/10 | 6/10 | 0.0277 | 10 |
| 3 | 1,000 | 1/10 | 0/10 | 0.5225 | 1 |
| 3 | 10,000 | 8/10 | 3/10 | 0.2789 | 8 |
| 3 | 25,000 | 9/10 | 6/10 | 0.0213 | 9 |

There were 15 fits with too few shared factors and two with no shared factors. None overestimated the dimension. No exception or timeout occurred. Three fits emitted ICA convergence warnings; none met the coefficient criterion. Covariance numerical rank was five within every domain. Total recorded per-run elapsed time was 55.14 seconds using sequential CPU execution and one BLAS thread.

More cells generally improved the mixing estimates and conditional median coefficient errors, but some correct-dimension runs still had very large graph coefficient errors. A single accurate median does not establish reliable recovery across models; the unconditional counts expose that difference. Ten draws per setting are a small descriptive replication, and the singleton conditional graph result at three domains/1,000 cells is especially weak evidence.

## Comparison with upstream saved results

Only the public JSON [saved statistics](upstream/experiments/experiment_l=3/results_gamma=0.2/stats.json) were read. No pickle was loaded. The original scripts use 1,000 models per setting at five sample sizes; this study uses ten at three sizes. Original random draws and the historical dependency environment were not replayed.

| Domains | Cells per domain | Our average shared dimension | Original average shared dimension | Our native mixing median | Original native mixing median | Our conditional graph median | Original conditional graph median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1,000 | 2.5 | 2.692 | 0.01697 | 0.03012 | 0.5431 | 0.6356 |
| 2 | 10,000 | 3.0 | 2.974 | 0.00326 | 0.00363 | 0.0549 | 0.3919 |
| 2 | 25,000 | 3.0 | 2.983 | 0.00195 | 0.00226 | 0.0277 | 0.0583 |
| 3 | 1,000 | 1.3 | 1.400 | 0.01963 | 0.02064 | 0.5225 | 0.5724 |
| 3 | 10,000 | 2.8 | 2.832 | 0.00444 | 0.00436 | 0.2789 | 0.4223 |
| 3 | 25,000 | 2.9 | 2.958 | 0.00257 | 0.00269 | 0.0213 | 0.0952 |

The saved JSON calls its error summaries “mean,” but the inspected upstream plotting helper computes medians. Original conditional graph denominators and the new unconditional criterion are not available from these aggregate files, so no original success rate is invented. Lower errors in this small subset are not evidence that we improved the algorithm.

## Scoring and implementation boundaries

For mixing error, let beta be the smaller of true and estimated shared dimensions times 30. The original plotting helper divides the native raw error by beta. The paper denominator is its square root. The [summary](published_replication_summary.json) reports both, distinctly labeled, and includes medians and quartiles with their available sample counts. Both use the native alignment numerator; the latter changes only the denominator and is not a replacement exact optimizer for the paper's permutation/sign objective. Zero recovered shared factors yield unavailable mixing scores, never perfect scores.

The native graph metric compares coefficient matrices up to its implemented sign/permutation alignment. That implementation selects a permutation using absolute entries before adjusting signs, rather than exhaustively minimizing the final signed objective. Its estimated coefficient matrix is not constrained to be a thresholded DAG. No causal adjacency or direction accuracy is claimed here.

The theory review identified native duplicate-endpoint and candidate-exhaustion edge cases in pure-child selection. They were not repaired for this arm. Any triggered exception would have been retained in the unconditional denominator; none occurred in these 60 fits. Gamma does not set ICA dimension: upstream uses covariance numerical rank for that step.

The current environment uses Python 3.11.14, NumPy 2.2.6, SciPy 1.14.1, scikit-learn 1.8.0, causaldag 0.1a163 and NetworkX 3.6.1. This differs from several upstream historical dependency pins and limits exact numerical reproduction.

## Validation and artifacts

Two [self-checks](test_published_replication.py) passed before fitting: graph recovery from a noiseless true mixing matrix with signed/permuted columns, including an independent exhaustive alignment check for that oracle case; and zero-shared unavailability with the two normalization formulas. This validates the wrapper and a controlled graph-helper case, not every native heuristic or every theorem assumption.

An independent agent reconstructed the native scores from all 60 saved matrix files without importing or fitting the learner/generator, and verified seals, summary quantiles and denominators. Maximum disagreement was 2.22e-16 for raw mixing error and zero for graph error; all status and criterion counts agreed. The [audit receipt](reviews/native_replication_audit.json) records this check. Agreement verifies the reported implementation metrics, not causal identification outside the published model class.

Per-run JSON files retain status, warnings, timing, data hashes and all available scores. Small NPZ files retain true/estimated mixing and graph matrices, with checksums; synthetic cell matrices were discarded. The [manifest](published_replication_manifest.json) binds source, protocol, results and comparison inputs. [Resource receipt](replication_resources.json), [self-check receipt](published_replication_self_check.json), and [execution log](published_replication_run.log) are retained.

To verify the frozen inputs and summarize existing results from this directory:

```bash
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python replicate_published_model.py summarize
```

The runner's `run` command checks the frozen source/environment and skips already sealed results. No existing biological benchmark world, modality, paired information or GPU fit was added for this published-model control.

