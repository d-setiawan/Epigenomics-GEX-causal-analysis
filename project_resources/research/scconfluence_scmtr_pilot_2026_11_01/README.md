# scConfluence and scMTR-seq pilot

_Completed 2026-11-01. All six fixed GPU fits, scientific readouts and independent reviews completed; reliable causal recovery was not established._

## Results and next step

The [causal reliability report](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_causal_reliability_v1/RESULTS.md)
contains all synthetic count/activity comparisons, direction stress tests,
subsamples and empirical outcomes. The [state report](state_readout.md) records
the separate integration diagnostics. scConfluence improved collider matching
in one seed, but the second seed gave approximately random matching in both
worlds. The only full-count cross-assay adjacency recovered by a new simulated
fit connected two shared-state decoys; it was not a direct biological edge.

The real-data pilot showed partial histone–histone pair-rank correspondence,
weak RNA–histone correspondence and inconsistent differentiation-day transfer.
Its sparse paired-count panel provided no positive cross-assay graph reference.
The independently reproduced [artificial-null check](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_scmtr_product_null_audit_v1/RESULTS.md)
exposed a concrete sparse-count calibration problem in the current RIT/RCIT–FCI
readout. Empty graphs and repeated edges therefore cannot substantiate causal
recovery here. These findings concern this fixed count-input adaptation and
small promoter-only cohort, not every published configuration or all scMTR data.

The next algorithmic priority is to develop and validate a count-CI calibration
that handles sparse detection and ties, using these cached count arrays and
negative controls before more integration fitting. Merely switching the
installed RIT permutation flag is insufficient: the
[source review](reviews/installed_rit_review_addendum.json) records its strict-tail
and missing plus-one behavior. No CI method, threshold or fitted model was
changed after observing the present results.

The [execution receipts](execution_results.json), [state review](state_review_summary.json)
and [final integrity review](final_integrity_review.json) bind the completed
models, exports, readouts and independent audits. The original frozen simulation
suite remains a separate, incomplete study.

The user authorized testing a different integration algorithm and scMTR-seq,
with independent agents testing causal-discovery reliability. The learner still
receives only RNA, H3K27ac, H3K4me3 and permitted gene/region annotation. Genuine
cell pairing and day labels are reserved for evaluation. No simulated worlds
are generated or replaced.

## Fixed algorithm comparison

Use scConfluence commit `14d81e2ca679e161ec5c6806103a66c14fcdcd9a`, with an explicit
correction of its pair-loss accumulation bug. The correction must preserve all
three pair contributions and their gradients, independently of pair ordering.
Original source and license are preserved beside the local adapter. This is a
count-input adaptation of the published architecture, not a replication of its
PC-based high-dimensional experiments.

The chosen pilot configuration is latent dimension 30, hidden width 128, total
batch size 128, maximum 120 epochs, early-stopping patience 8 and an internal
validation fraction of 0.2. Learning rate is 0.003; native encoder/decoder depths
are 3/2, with zero dropout and no batch normalization. Native transport settings are mass 0.5, reach 0.3,
blur 0.01, IOT weight 0.01 and Sinkhorn weight 0.1. Negative-binomial decoders
receive integer counts; encoders receive library-normalized `log1p` features,
without PCA or variance ranking. Training-derived feature scaling and existing
gene/region links define cross-assay costs. No state, pairing, day or causal
truth informs model selection or tuning.

The initial synthetic comparison reuses informative `linear_chain_0` and
`constrained_collider_0`, seeds 0 and 1. Source matrices remain the original
observed files; training cells and feature IDs are restricted to those already
retained in the frozen scGLUE preprocessing. Reference PCA/LSI values are not
used. Existing full-count and compressed scGLUE fits provide the baselines.
New exports must cover the already frozen discovery pools; poor coverage cannot
be hidden by shrinking those pools.

## Reliability checks

On simulation, assess the fixed nine-node panels against the known activity
graph and measurement-count PAG. The primary recovery endpoint uses counts;
latent activity is a diagnostic control. Exact biological directions remain an
identifiability stress test. Retain paired, random-assignment and true-state
controls, as well as three independently subsampled 80% assay pools with
rematching. These few subsamples give coarse conditional stability, not
confidence intervals, false-discovery control or independent biological
replication. Model seeds are repeated fits of each world.

The existing RCIT/FCI settings remain fixed: alpha 0.01, HBE approximation,
100/5 random features, canonical-query seed 0 and minimum 20 cells. Matching
uses pool seed 2718; the random control uses 3141; subsample seeds are
7000–7002. Only public discovery cohorts are used. Protected assessment cohorts
are excluded. Numerical failures remain explicit outcomes.

## scMTR-seq intake and hidden pairing

The first real-data pilot uses the human sublibrary pair DNA `GSM8529544` and
RNA `GSM8529552` from `GSE277778`. Necessary processed files total about 160 MB.
The published DNA-filtered source population contains 2,444 canonical cells.
RNA tags encode days 0–3; they are not a generic suffix to discard. The authors'
dominant-tag logic and exact sublibrary linkage are checked before joining.

Freeze a seeded canonical-ID split with seed 2718: reserve 600 source cells for
evaluation, then assign remaining cells evenly to disjoint RNA/ac/me3 training
pools before applying QC using only each cell's observed assay. RNA QC is at
least 1,000 counts and 500 detected genes, a dominant valid RT-tag fraction above
0.5 and mitochondrial fraction at most 5%. Histone QC requires at least 250
fragments in the assigned mark. Complete-case QC is permitted only for the
paired evaluation target, whose population must be reported explicitly.

Use existing GRCh38 GENCODE annotation for fixed ±2 kb gene-TSS promoter counts.
This pilot studies promoter-associated structure; distal regulation is outside
its feature representation. Select the nine-node evaluation panel as three
non-overlapping genes using only unpaired training detection: at least 10% in
RNA, 5% in each mark, and 20 detected cells per assay. Order eligible genes by
SHA256 of `20261101_causal_panel_v1|` plus the unversioned Ensembl ID. Insufficient
eligible genes is a reported failure, not a reason to lower thresholds after
seeing results. Exclude panel genes, aliases and overlapping promoter features
from integration encoders and cost construction.

Learner-facing IDs are opaque and assay-specific. Canonical pairing and day
metadata are separate evaluator artifacts. The source already imposed a
six-target DNA-depth filter; that inherited selection cannot be undone using
only these processed three-assay files. One mixed-day sublibrary supports an
internal cell holdout, not independent biological replication. A graph from its
paired counts is a noisy observed-dependence reference, not known causal truth.

## Runtime and review roles

The workspace-local environment uses the [minimal pinned overlay](../../../envs/requirements-scconfluence-pilot-2026-11-01.txt)
on the existing `scrna_r` Python. PyTorch 2.10.0, Lightning 2.6.1 and TensorBoard
2.20.0 are inherited; NumPy 1.26.4, AnnData 0.10.9, MuData 0.3.1, POT 0.9.6.post1
and GeomLoss 0.2.6 are local. Python 3.11 is an explicit deviation from upstream's
metadata restriction to Python 3.10. Optional preprocessing packages are not
required by the inspected native runtime modules. Basic CPU transport and
gradient checks passed, followed by native CPU/GPU model checks and all six
scientific GPU fits.

The shared environment is unchanged. Unrelated inherited pgmpy/SnapATAC
dependency conflicts mean this is a training overlay, not a general replacement
environment; causal scoring continues in the original environment. Resource
inspection found 5.51 GiB available RAM and an 8 GiB RTX 3070 Ti Laptop GPU.
GPU fits ran serially using the previously verified process-local CUDA
13.0.96 runtime. Resource and package acquisition receipts are saved here.

The data agent owns source/barcode/QC preparation; the adapter agent owns the
native objective correction and its tests; the causal agent independently
evaluates recovery and stability. The primary agent coordinates execution,
resource use, cross-checks and reporting. The data agent independently replayed
all empirical controls and null graphs, both real matching evaluations and the
fixed day decoders' reported confusion matrices. State assignments/costs and
Ridge transfer calculations were independently reconstructed for every simulated
fit. Validation of execution and calculations does not establish causal recovery.

The [serial runner](run_fixed_pilot.py) and [frozen execution plan](execution_plan.json)
record all six fits before execution, their source/input hashes and configuration.
It preserves each step's log and exit status and refuses existing output paths.
The [GPU API check](gpu_api_check.json) passed native fitting, exact checkpoint
roundtrip and held-out export. The [input preflight](input_preflight.json) covers
all three dataset inputs. No cells were dropped in that preflight.

The fixed [simulation controls](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_causal_reliability_v1/BASELINE_RESULTS.md)
and [empirical controls](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_scmtr_reliability_v1/BASELINE_RESULTS.md)
are already complete. They separate paired-count limitations from integration
performance. [L19](../LIMITATIONS.md#l19-scmtr-seq-pilot-agreement-is-not-causal-ground-truth)
records the empirical interpretation boundary.

## Exploratory check of the causal readout

After the primary random control produced cross-assay edges, a separate
[product-coupling null protocol](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_scmtr_product_null_audit_v1/protocol.json)
fixed seeds 10000–10049 before executing them. It permutes the existing full
595-cell assay blocks independently, preserves all within-assay measurements,
and retains the same panel, RCIT/FCI settings and failure handling. This is a
post-primary diagnostic, not a new simulation world or a tuning stage.

The [null summary](../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261101_scmtr_product_null_audit_v1/summary.json)
records every graph, the cross-edge-count distribution and edge recurrence.
An [independent replay](reviews/empirical_product_null_audit.json) rebuilt all
50 assignments and count arrays from the original source-cell matrices and
reproduced every graph edge, endpoint and summary value. These artificial-null
frequencies are not biological false-discovery rates. No threshold or model
setting was changed from these outcomes.
