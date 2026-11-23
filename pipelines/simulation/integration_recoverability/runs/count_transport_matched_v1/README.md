# Exact-model count-transport baseline

This November 15 experiment generated nine worlds with one, two or three independent Bernoulli
sources, three mechanisms per dimension, and count measurements from RNA, H3K27ac and H3K4me3.
Each fit received the true dimension and separate cells from each assay. Four starts at each of
two fitting sizes produced 72 optimization streams and 144 capped or continued fit artifacts. At
12,000 fitting cells per assay, the calibration-selected continued fit passed in 3/3 one-source
worlds, 0/3 two-source worlds and 0/3 three-source worlds. The higher-dimensional selected fits
ended at worse training objectives than the feasible generating parameters.
[Results and figure](../../COUNT_TRANSPORT_MATCHED.md) · [Per-world results](comparison.tsv) ·
[Aggregate results](aggregate.tsv)

The repository includes the generation, fit and evaluation seals. Raw generated counts, fitted
parameters and full evaluations are run outputs. The November 22 clean-checkout replay matched
all 46 generator artifact hashes and repeated the full matched baseline with the same aggregate
recovery counts. [Frozen plan](PLAN.md) · [Generation seal](generation_seal.json) ·
[Fit seal](fit_seal.json) · [Evaluation seal](evaluation_seal.json) ·
[Scientific review](REVIEW.md) · [Numerical audit](NUMERICAL_REVIEW.json) ·
[Verification record](../../../../../reproduce/VERIFICATION.md)

To replay this baseline, activate the CPU environment and run the following from the repository
root. Choose a new run ID for each invocation. The example writes to
`runs/replay-matched/matched/` under this track.
[Environment and replay instructions](../../../../../TRANSFER.md)

```bash
python pipelines/simulation/integration_recoverability/replay_count_transport.py replay-matched --stage matched
```

The same nine worlds were reused in two later experiments.
[Label parameterization](../count_transport_labels_v1/README.md) ·
[Count-informed search](../count_transport_search_v1/README.md)
