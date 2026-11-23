# Count-source runner review

Reviewed `apply_counts.py`, `protocol.json`, the count-source API, focused tests, and the reused native graph assembly. Source hashes and the exact check scope are in [runner_review.json](runner_review.json). The reviewer implemented `count_sources.py` but did not implement the runner; the separate theory agent audits the estimator.

**Decision: approved for the frozen bounded pilot; no blocking issue found.** All nine source/runner tests passed independently. Additional fixed discrete fixtures exercised zero-source fitting, estimator abstention, empty pair matching, strict JSON, and NPZ loading without pickle or object arrays. No scientific count panel was fitted and no evaluator reference was read during this review.

The runner selects rank and sources on its deterministic 75% training partition and estimates fingerprints on the remaining 25%. Bootstrap resampling keeps those partitions separate. It computes outside-panel exposure before permutations and holds it fixed. Controls preserve integer count entries within exposure strata and partition. Corrected and uncorrected arms share selected rank, source seeds, and bootstrap resampling seeds.

Within-assay bootstrap alignment carries column permutations and signs into the demixing matrices. Cross-assay matches instead use calibrated moment/characteristic-function comparisons; they require source gates, narrow uncertainty bands, a unique sign, and mutual uniqueness. Three-assay candidates require all three accepted pairs and consistent signs. No matching is forced. Zero-source and abstained model outputs are handled without dereferencing unavailable arrays.

Interpretation remains bounded: the rank screen is an approximate marginal-preserving permutation diagnostic; finite fingerprint equivalence does not identify a full source law; and the depth proxy does not establish exogenous technical exposure. A zero predicted cross-covariance when no match is accepted is an abstention baseline, not evidence of biological independence. Native graph coefficients remain exploratory. The separate evaluator must guard zero/constant prediction variances and use frozen exposure normalization.
