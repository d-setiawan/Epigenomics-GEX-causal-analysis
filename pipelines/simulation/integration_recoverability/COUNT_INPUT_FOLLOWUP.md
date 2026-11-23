# Count-input integration comparison

_Completed 2026-10-31 local time / 2026-11-01 UTC. Uses existing development worlds and the same three assays._

**Full-feature scGLUE input modestly improves state matching in both tested
worlds, with better worst-direction decoder transfer and neighbor coverage.**
This supports further testing without PCA/LSI input compression; it does not
establish reliable causal recovery. All four GPU fits and both independent
numerical reviews are complete.

## Question and scope

Does giving scGLUE its full retained feature matrix, instead of PCA/LSI coordinates,
improve biological-state correspondence? This follows the user's concern that
variance-based compression might remove useful structure before integration.
The downstream causal tests already operate on feature counts or activities;
the change here concerns the integration encoder.

The original [32-fit validation study](INTEGRATION_VALIDATION.md) is complete and
unchanged. This is a separate four-fit exploratory comparison, using cached
`linear_chain_0` and `constrained_collider_0`, informative counts, seed 0, and
the previously selected `lam_align=0.2`. No worlds or measured datasets are
generated or replaced. RNA, H3K27ac and H3K4me3 remain the model inputs, with
their existing permitted guidance. No hidden state or cell pairing enters fitting.

## Controlled comparison

| Arm | Encoder input | Reconstruction |
| --- | --- | --- |
| `compressed` | Existing RNA PCA and histone LSI coordinates | Original negative-binomial counts |
| `native_counts` | All retained count features through native library normalization and `log1p`; `use_rep=None` | Same negative-binomial counts |

Both arms are fitted afresh with identical cells, feature IDs/order, guidance,
latent dimension, hidden architecture, batch size and native training phases.
Input-dependent encoder widths and parameter counts necessarily differ. The
120-epoch limit applies separately to pretraining and fine-tuning; native early
stopping and balancing remain active. The runner checks the actual configured
encoder and first-layer widths and rejects zero-library cells rather than
silently selecting a different pool. All original frozen scientific source
files remain unchanged.

This is an input-pipeline comparison, not an isolated PCA-rank experiment:
native count mode also changes normalization. It removes fixed PCA/LSI
compression, but retains training QC, the existing feature universe, guidance,
and a learned low-dimensional bottleneck. The RNA variance-selection cap is
not binding in these two worlds. It does not test removal of all empirical
feature-selection rules.

The [runner](scripts/count_input_experiment.py) reuses native fitting, checkpoint
roundtrip checks, public-cohort export and the existing validation metrics.
The immutable [protocol](runs/integration_validation/20261031_count_input_v1/protocol.json)
pins sources, configuration, runtime and analysis scope. Fits execute serially,
compressed then native counts for each world. The [execution ledger](runs/integration_validation/20261031_count_input_v1/execution.jsonl)
retains exit codes and log locations; existing attempts cannot be overwritten.

## Measurements

The primary endpoint is the worst of the three assay-pair normalized squared-state
transport gaps, using the same one-to-one RNA-anchored matching and common
validation pools in both arms. Zero reaches the finite-pool state optimum;
one is the expected random-matching cost. It is not a percentage of correctly
paired cells. Absolute oracle and achieved costs are saved alongside the gap.

Secondary endpoints are fixed source-standardized state-decoder transfer,
within-assay neighborhood retention, neighbor hubness before assignment and
coordinate variance. State labels enter evaluation after pool selection and
embedding-only matching. PCA/LSI neighborhood retention measures fidelity to the
old representation, not a neutral biological ground truth.

Only public train/validation/discovery embeddings are exported; this comparison
scores validation cells. There is no new causal graph test or private assessment
release. Two previously inspected development worlds and one seed cannot
establish general reliability. All outcomes, including failures, are retained.

## CUDA verification

The driver, GPU and CUDA libraries are present. Host inspection identifies an
RTX 3070 Ti Laptop GPU with 8 GiB VRAM. The user's `nvcc` output reports compiler
11.5, while the active Python environment reports PyTorch 2.10.0 built for CUDA
13.0. These are different components; the compiler version alone does not
describe the library loaded by PyTorch.

In a fresh host process, the default environment segfaulted during
`torch.cuda.is_available()` (exit 139). A second process using the official CUDA
13.0.96 runtime through `LD_PRELOAD` completed a small GPU forward/backward
calculation and verified the loaded library. This localizes the practical
workaround; it does not establish the underlying cause of the default crash.

The fits use `/tmp/integration_followup_runtime_20261031/libcudart.so.13.0.96`,
SHA256 `96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0`,
with `CUDA_LAUNCH_BLOCKING=1` and one-thread BLAS/OpenMP caps. The library was
extracted from [NVIDIA's official redistribution archive](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.0.96-archive.tar.xz).
The [live process-map observation](runs/integration_validation/20261031_count_input_v1/runtime_observation.json)
confirms the first fit loaded that library. No driver, shared
environment or package installation was changed. A temporary library path must
be restored and hash-verified if it disappears before reproduction.

## Review and execution status

Before fitting, an independent reviewer ran 24 focused CPU tests covering mode
selection/restoration, zero-library rejection, public-only fitting, failure
preservation, artifact/source/runtime binding, and all three state-pair costs
under nonidentity permutations. All passed. The 17 original scientific source
pins still matched. These tests mocked native fitting; actual GPU model fits
and their native encoder/export checks provide the separate runtime validation.

All four fits completed, including native pretraining, balancing, fine-tuning,
checkpoint reload and public-cohort export. Recorded checkpoint roundtrip errors
are zero for all assays. Compressed encoders used 50 inputs per assay. Native
count encoders used 1,944 RNA features in the linear world, 1,922 in the collider
world, and 15,000 features for each histone assay. Fit elapsed times were 7.60 and
9.66 minutes for the linear-world compressed/count arms, and 9.03 and 11.52
minutes for the collider arms. These are execution records, not controlled
runtime benchmarks.

The first completed native fit exposed a bookkeeping issue in the original
scorer: scGLUE's `autodevice()` changes `CUDA_VISIBLE_DEVICES` from unset to `0`.
The original guard expected the initial environment value, although the receipt
correctly records its value after GPU selection. An independently reviewed
[analysis-only amendment](runs/integration_validation/20261031_count_input_v1/runtime_scoring_amendment.json)
was declared before state scores. The [corrected scorer](scripts/score_count_input_followup.py)
accepts only this documented transition, with the installed native source and
first observation pinned; all other original checks remain. It writes separate
`reports_runtime_corrected` outputs. Training, the original protocol and source,
fit receipts, and original evaluator failures are preserved. No model is refitted
for this correction. The combined focused suite has 35 passing tests.

The original controller exited with code 1 because both original scoring calls
rejected that environment mismatch; all four fitting calls exited with code 0.
Both corrected scoring calls completed. The original errors remain in the
execution ledger and logs. They are not failed integrations or hidden retries.

Independent reviews of the [linear world](runs/integration_validation/20261031_count_input_v1/reviews/linear_chain_0_independent_numeric_artifact_review.md)
and [collider world](runs/integration_validation/20261031_count_input_v1/reviews/constrained_collider_0_independent_numeric_artifact_review.md)
reproduced every pair cost/gap and source-transfer score exactly from saved
arrays, independently rebuilt pools and checked native configurations,
checkpoints, exports and source/protocol bindings. They did not deserialize
models or run additional GPU training. The [interpretation review](../../../project_resources/research/integration_followup_2026_10_31/count_input_methods_independent_review.md)
also checked the information boundary and alternative-method/data claims; its
input-layer wording correction is incorporated above.

## Observed results

Both arms use identical pools: 1,936 training and 487 validation cells per assay
in `linear_chain_0`, and 1,934 training and 491 validation cells per assay in
`constrained_collider_0`. All values below concern informative counts and seed 0.

The decoder summary is the minimum mean state $R^2$ over all six directed
cross-assay transfers. Neighbor coverage is the fraction of target cells
receiving at least one 15-nearest-neighbor link before assignment. Higher
coverage is not higher matching accuracy. Lower indegree Gini means less
concentration of those links. Each worst-direction summary may concern a
different assay pair.

| World | Encoder | Worst gap | Minimum cross-assay $R^2$ | Minimum neighbor coverage | Maximum indegree Gini |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear chain | PCA/LSI | 0.1476 | 0.8168 | 0.7803 | 0.6891 |
| Linear chain | Full features | 0.1365 | 0.9271 | 0.9528 | 0.4308 |
| Constrained collider | PCA/LSI | 0.1312 | 0.8914 | 0.8473 | 0.6254 |
| Constrained collider | Full features | 0.1228 | 0.9106 | 0.9491 | 0.4334 |

The descriptive mean worst-pair gap decreases from 0.13943 to 0.12961, about
7.0% relative to the fresh compressed baseline. This is not a seven-percentage-point
gain in pairing accuracy or evidence of statistical significance. There are
only two previously inspected development worlds and one optimization seed.
No threshold, guidance, feature set or training parameter was selected from
these new outcomes.

Absolute squared-state costs retain the finite-pool oracle floor. In the next
table, P denotes PCA/LSI input and F denotes full-feature input; the state
coordinates use the common pooled training-state scale. The histone–histone
pair remains the hardest pair under the declared gap in both worlds.

| World | Pair | Achieved cost P | Achieved cost F | Oracle cost | Expected random cost | Gap P | Gap F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | H3K27ac–H3K4me3 | 0.6402 | 0.5977 | 0.0786 | 3.8829 | 0.1476 | 0.1365 |
| Linear | RNA–H3K27ac | 0.4852 | 0.3978 | 0.0618 | 4.0605 | 0.1059 | 0.0840 |
| Linear | RNA–H3K4me3 | 0.4140 | 0.3966 | 0.0699 | 3.9379 | 0.0889 | 0.0844 |
| Collider | H3K27ac–H3K4me3 | 0.5806 | 0.5470 | 0.0588 | 4.0354 | 0.1312 | 0.1228 |
| Collider | RNA–H3K27ac | 0.5019 | 0.4722 | 0.1042 | 3.8998 | 0.1048 | 0.0970 |
| Collider | RNA–H3K4me3 | 0.5053 | 0.4496 | 0.1141 | 3.8950 | 0.1035 | 0.0887 |

Matching improves for all six world/pair combinations. Decoder transfer is
mixed by direction, despite improvement in the weakest direction of each world:

| Source→target | Linear P | Linear F | Collider P | Collider F |
| --- | ---: | ---: | ---: | ---: |
| H3K27ac→H3K4me3 | 0.9797 | 0.9754 | 0.9562 | 0.9740 |
| H3K27ac→RNA | 0.9533 | 0.9395 | 0.9543 | 0.9292 |
| H3K4me3→H3K27ac | 0.8168 | 0.9353 | 0.9124 | 0.9240 |
| H3K4me3→RNA | 0.9450 | 0.9452 | 0.9460 | 0.9106 |
| RNA→H3K27ac | 0.8929 | 0.9271 | 0.8914 | 0.9525 |
| RNA→H3K4me3 | 0.9481 | 0.9656 | 0.9420 | 0.9536 |

Retention of the original PCA/LSI neighborhoods decreases in every assay:

| World | Encoder | RNA Jaccard | H3K27ac Jaccard | H3K4me3 Jaccard |
| --- | --- | ---: | ---: | ---: |
| Linear | PCA/LSI | 0.2598 | 0.2499 | 0.3110 |
| Linear | Full features | 0.2477 | 0.2091 | 0.2786 |
| Collider | PCA/LSI | 0.2613 | 0.2497 | 0.3294 |
| Collider | Full features | 0.2469 | 0.2140 | 0.2628 |

That reference favors preservation of the old geometry; its decline does not
by itself establish biological information loss. Conversely, better matching,
decoder and coverage summaries do not certify the unobserved residual joint
distribution or causal conditional independences.

Full precision, all decoder directions, hubness and variance values, selected
IDs, assignments and provenance are retained in the [linear report](runs/integration_validation/20261031_count_input_v1/reports_runtime_corrected/linear_chain_0.json)
and [collider report](runs/integration_validation/20261031_count_input_v1/reports_runtime_corrected/constrained_collider_0.json).
The practical conclusion is to retain the full-feature arm as a promising
comparator. More seeds or other cached worlds are needed before replacing the
default on reliability grounds. No downstream causal graph was computed for
these four fits.

## Alternative methods and empirical data

The [method review](../../../project_resources/research/integration_followup_2026_10_31/alternative_methods.md)
compares scConfluence, uniPort, UINMF and Pamona. The most relevant prospective
alternative retains separate count autoencoders and aligns them using relaxed
optimal transport with existing feature links. scConfluence provides that
architecture, but a reviewed upstream loss-accumulation issue must be corrected
before a three-assay pilot. Its published high-dimensional experiments used
PC inputs; a no-PCA histone-count version would be an explicit adaptation.
None of these methods is established here as recovering causal structure.

The [data review](../../../project_resources/research/integration_followup_2026_10_31/data_viability.md)
finds evidence supporting continued state-alignment work with the current data.
H3K27ac and H3K4me3 retain a positive broad protein-defined state correspondence
in historical scGLUE embeddings. Fine locus coverage is sparse, and the separate
RNA and chromatin collections have no verified donor match. This supports a
bounded investigation, not a claim that the RNA anchor or within-monocyte
coupling has been validated.

The [dataset shortlist](../../../project_resources/research/integration_followup_2026_10_31/dataset_shortlist.md)
identifies scMTR-seq as a possible future external test with all three assays
measured in the same cells, plus pairwise Paired-Tag and genuinely unpaired
mouse-brain alternatives. Pairing could be withheld from a learner and retained
only for evaluation. These change biological context and are optional future
validation resources, not inputs to this experiment. No count matrices or raw
sequencing files were downloaded. Observed pairing would validate dependence,
not supply a known causal graph.
