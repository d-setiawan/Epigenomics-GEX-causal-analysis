# Count-native source search without parameter bounds

Experiment date: 2026-11-15. This is an exploratory comparison on the nine
existing exact-model worlds. No new datasets or biological information are
introduced. The [earlier experiments](COUNT_TRANSPORT_MATCHED.md) remain intact.

**Completed result:** count-informed source search followed by calibration-based
selection meets the frozen recovery criteria for shared-state transport and
cross-assay count relationships in **all nine worlds**. This includes all three
two-source and all three three-source worlds.
The unbounded version using the old physical starts recovers five of nine;
the preceding labeling experiment recovered three of nine. These results
demonstrate recovery under the specified model, not general causal discovery.

## Recovery results

Each entry counts complete worlds passing every declared assay-pair criterion
at 12,000 unpaired fitting cells per assay. Only calibration-selected models
receive the bootstrap intervals needed for the full recovery decision.

| Sources | Previous labeling experiment | New coordinates and no boxes, old starts | Count-informed search and selection |
| --- | ---: | ---: | ---: |
| 1 | 3/3 | 3/3 | 3/3 |
| 2 | 0/3 | 2/3 | 3/3 |
| 3 | 0/3 | 0/3 | 3/3 |
| Total | 3/9 | 5/9 | 9/9 |

The following errors are each selected informed model's worst ordered assay-pair
error. They compare against exact population quantities, not a noisy empirical
covariance from the assessment cohort.

| World | Worst covariance error | Worst restricted transport error | Full recovery |
| --- | ---: | ---: | --- |
| `d1_rep0` | 1.99% | 1.98% | Pass |
| `d1_rep1` | 4.46% | 4.22% | Pass |
| `d1_rep2` | 1.80% | 5.02% | Pass |
| `d2_rep0` | 3.14% | 2.50% | Pass |
| `d2_rep1` | 1.84% | 2.61% | Pass |
| `d2_rep2` | 2.48% | 2.82% | Pass |
| `d3_rep0` | 1.83% | 1.97% | Pass |
| `d3_rep1` | 1.73% | 2.81% | Pass |
| `d3_rep2` | 2.64% | 3.06% | Pass |

All 54 ordered predictions pass; these represent 27 assay-pair associations
across nine worlds. The largest oracle conditional-score gap is **0.0002011
nats per target feature**, compared with the declared tolerance of 0.02.
The smallest nominal 95% lower predictive-gain bound is **0.00908 nats per
target feature**. Source probabilities were estimated from unpaired counts;
they were not supplied to the learner.

Every selected informed model has a better training objective than the known
feasible generating parameters. The previous higher-dimensional suboptimality
witness therefore disappears for these selected fits. This comparison still
does not certify a global optimum.

## What this says about the earlier failures

Adding sources did not make recovery impossible in these worlds. The stronger
fitting procedure recovers the same targets from the same observations. The
earlier failures were limitations of those estimators under these conditions,
not evidence against the population identification claim.

Removing the parameter boxes and changing coordinates alone was insufficient:
the plain arm still fails one two-source world and every three-source world.
Initializing from count-based source separation and using multiple starts
provides the additional improvement in this comparison. The experiment does
not isolate which individual element of that combined change is responsible.

Individual optimization runs remain unreliable. Across the 12 informed outer
starts at each dimension, 12/12 one-source, 10/12 two-source and **5/12
three-source** fits meet the covariance, transport, oracle-gap and numerical
criteria. These are point-estimate diagnostics, not full bootstrap-based
recovery decisions for every start. Nine of the 36 informed outer fits remain
provably worse than the feasible generating point. Some incorrect fits have
extremely large covariance errors. Thus the successful procedure includes
four outer starts and selection by unpaired calibration likelihood; it is not
a claim that one arbitrary initialization succeeds.

## What changed

The earlier estimators already had no cumulative iteration or evaluation cap.
They retained finite parameter boxes and four similar initializations. The new
experiment removes the boxes and compares two fitting strategies:

- **Plain unbounded:** use the same physical starting distributions as before,
  expressed in new coordinates, and optimize without finite parameter bounds.
- **Source search:** estimate finite Poisson mixtures from each assay's raw
  counts, enumerate their possible source labelings, and use these estimates
  to initialize the same joint model. Refine several candidates and preserve
  unsuccessful attempts alongside successful ones.

Both strategies use exactly the same marginal count likelihood. There is no
PCA, feature selection, paired-cell training, true-parameter initialization or
guidance graph. Each strategy has four outer seeds per world; selection uses
the existing unpaired calibration counts.

## Why the new coordinates may help

Let the independent binary sources be $B_j\sim\operatorname{Bernoulli}(p_j)$.
The previous strict model was

$$
\log R_c=M_c\left(\frac{B-p}{\sqrt{p(1-p)}}+\mu\right).
$$

The new coordinates express the same model as

$$
\log R_c=A_c(B+h),\qquad
M_c=A_c\operatorname{diag}\!\left(\sqrt{p(1-p)}\right),\qquad
\mu=\frac{h+p}{\sqrt{p(1-p)}}.
$$

Here $p$ changes the probabilities of the source states, while $A_c$ and $h$
determine their log-rate values. Updating a source frequency no longer forces
an accompanying change in its standardized values. The shift $h$ is shared
across assays; the final model has no free assay-specific intercept.

For three sources, there are eight binary states. The initializer considers
all 40,320 assignments of estimated mixture components to these states,
represented by 840 classes after removing equivalent bit complements and
source permutations. Estimated asymmetric source frequencies align the axes
across assays. This enumeration is exhaustive only for the already estimated
mixture atoms, not for all possible continuous parameters.

## Data and recovery criteria

The same nine worlds cover one, two and three sources, with three mechanisms
per dimension. Each assay supplies 12,000 unpaired fitting cells and 2,000
calibration cells. RNA, H3K27ac and H3K4me3 have 30, 24 and 18 abstract features.
The correct source dimension is supplied. A separate, previously assessed
cohort of 4,000 paired cells is used only after the new fit seal.

A calibration-selected world passes only when all six ordered assay predictions
meet the existing criteria: numerical acceptance, population cross-covariance
error at most 10%, restricted log-rate transport error at most 10%, a positive
nominal 95% lower predictive-gain bound, and an oracle conditional-score gap
at most 0.02 nats per target feature. The six prediction directions represent
three associations; they are not causal arrows.

The source-search arm attempts four marginal mixture starts per assay and
outer seed, compares up to 64 combinations across assays, and refines the best
four. Across the experiment this allows 432 marginal mixture fits and 180 joint
refinements, including the plain arm. The 72 outer records preserve inner
outcomes and failures. Joint refinement and mixture EM have no cumulative
iteration cap. Numerical stopping, floating-point limits, a 40-trial line-search
safeguard and finite start inventories remain explicit.

## Interpretation limits

The plain arm changes coordinates and parameter bounds together. The informed
arm additionally changes initialization and search effort. This experiment
cannot isolate the contribution of each individual change.

The assessment worlds have already been inspected, so the results are
exploratory. Bootstrap intervals condition on a fixed fitted model and do not
measure variability from fitting or provide simultaneous coverage. A small
gradient does not certify global optimization. A fitted training objective
worse than the true generating parameters proves suboptimality; a better
objective does not prove optimality.

The targets are shared-state transport and the count joint under the specified
model. Passing the covariance, transport and predictive-score tolerances does
not establish exact finite-sample equality or bound every full-joint distance.
These favorable simulations assume independent observation noise,
independent binary sources, full-rank assay maps and the correct source count.
They do not establish biological realism, general nonlinear integration,
unknown-dimension recovery or cross-assay causal directions.

## Computation and independent validation

The full run attempted **432 marginal Poisson mixtures** and **180 joint
refinements**, retained in 72 outer artifacts. Of the mixture attempts, 415
converged and 17 ended with explicit zero/dead-component or rate failures.
Of the joint refinements, 179 met the gradient criterion and one stopped on
floating-point stagnation. All 72 retained outer models were gradient-converged;
65 passed the additional numerical/model-class checks.

The longest EM fit used 6,417 iterations; the longest joint refinement used
**36,134 iterations**. The slowest outer run took 31.4 minutes and still failed
source separation. The elapsed time from protocol freeze to fit seal was about
**50.6 minutes** with two CPU workers and one BLAS thread each. This is an
execution record, not an isolated hardware timing benchmark. Long runtime
and local convergence are not substitutes for a good fitted law.

All **112 count-transport tests pass**. An independent reviewer verified the
coordinate transformation, gradients, global relabeling, and coverage of all
40,320 three-source assignments. A separate numerical audit recomputed all 72
outer training objectives, all 18 calibration selections, 108 selected ordered
prediction evaluations and 216 bootstrap intervals. It found no discrepancies
beyond numerical tolerance. Independent PyTorch gradients agree with the
implemented gradients within $2.88\times10^{-15}$. Previous frozen experiment
sources and artifacts were preserved.

## Records

- [Frozen plan](runs/count_transport_search_v1/PLAN.md) and
  [protocol](runs/count_transport_search_v1/protocol.json).
- [Fit and selection seal](runs/count_transport_search_v1/fit_seal.json),
  [full evaluation](runs/count_transport_search_v1/evaluation.json),
  [comparison table](runs/count_transport_search_v1/comparison.tsv),
  [aggregates](runs/count_transport_search_v1/aggregate.tsv), and
  [evaluation seal](runs/count_transport_search_v1/evaluation_seal.json).
- [Scientific review](runs/count_transport_search_v1/REVIEW.md) and
  [independent numerical review](runs/count_transport_search_v1/NUMERICAL_REVIEW.json).
- [Execution counts](runs/count_transport_search_v1/execution_summary.json),
  [reproduction notes](runs/count_transport_search_v1/README.md), and
  [independent numerical audit code](runs/count_transport_search_v1/numerical_audit.py).
- [Optimizer](count_transport_source_search.py),
  [experiment driver](count_transport_search_run.py), and
  [optimizer tests](tests/test_count_transport_source_search.py).

