# Exploratory Bernoulli-label optimization

The November 15 labeling experiment reused the same nine assessed exact-model worlds, 12,000
fitting cells per assay and four physical initial models per world. It expanded the parameter
range, allowed both Bernoulli label orientations, and applied one global relabeling after
fitting. All 36 starts met the numerical convergence criterion. The unpaired
calibration-selected models met full recovery in 3/3 one-source worlds, 0/3 two-source worlds
and 0/3 three-source worlds. In each of the six higher-dimensional groups, the generating
parameters attained a better training objective than the selected fit.
[Combined results report](../../COUNT_TRANSPORT_MATCHED.md) ·
[Per-world results](comparison.tsv) · [Aggregate results](aggregate.tsv)

The Git checkout includes the plan, protocol, fit and evaluation receipts, and independent
reviews. The original run cache contains the fitted models and full evaluation. The later
count-informed search reused these worlds as well. [Plan](PLAN.md) · [Protocol](protocol.json) ·
[Fit seal](fit_seal.json) · [Evaluation seal](evaluation_seal.json) ·
[Scientific review](REVIEW.md) · [Numerical audit](NUMERICAL_REVIEW.json) ·
[Source-search results](../count_transport_search_v1/README.md)
