# Paper-consistent scMRDR adapter

_Implementation audit, 2026-10-03; production validation updated 2026-10-04. This is separate from the older local `train_joint_jianle.py` approximation._

---

## Scope and status

This adapter uses the authors' actual neural architecture and loss implementations, pinned to scMRDR commit `7280e64ae19f7391c6f5adf00d9d353f79e96603`. Its training loop fixes two implementation errors in the upstream trainer and follows explicit paper-consistent configuration choices. It provides deterministic posterior-mean encoding for new cells and complete checkpoints.[^1][^2]

The target is **paper-consistent scMRDR applied to the agreed histone/RNA simulation**, not a numerical reproduction of the published RNA/ATAC/protein experiment. Histone gene scores and the benchmark's feature-selection budget are applications of the model, not preprocessing details claimed by the paper. Successful integration is not evidence that the unknown unpaired cross-assay joint distribution, or its causal graph, has been identified.

The standalone unit suite passes 11 tests, including a two-epoch CPU fit. That is implementation validation, not a structure-recovery result. Scientific benchmark results belong in `pipelines/simulation/integration_recoverability/runs/` and must report each fit's actual configuration, retention, stopping epoch, and failures.

## Inputs and API

All assays must use one explicitly gene-aligned feature index. Region-to-gene sums must come from allowed annotation, not planted causal edges, same-cell IDs, or latent state. A cell with zero counts for a measured gene is different from an assay that never measures that gene. Only the latter has availability zero. Supply nonnegative integer raw counts; the native encoder applies `log1p` internally and the native decoder uses the observed count row sum as library size.[^2]

```python
from paper_faithful import SCmRDRConfig, ScMRDRAdapter, fit

model = fit(
    counts_by_modality,          # dict[str, scipy.sparse.csr_matrix[N_m, G]]
    availability_by_modality,    # dict[str, binary numpy.ndarray[G]]
    feature_names,              # unique ordered list[str], shared by every assay
    out_dir="run/model",
    config=SCmRDRConfig(seed=0, device="cuda"),
    validation_counts_by_modality=validation_counts,  # optional separate cohort
)
encoded = model.encode(new_counts, "RNA", feature_names)
shared = encoded["shared"]      # N x 20 posterior means
specific = encoded["specific"]  # N x 20 posterior means
full = encoded["full"]          # N x 40, diagnostic only

restored = ScMRDRAdapter.load("run/model/checkpoint.pt", device="cuda")
```

Add the containing `methods/scmrdr/` directory to `sys.path`, or import the package through the repository's benchmark wrapper. The adapter does not depend on AnnData, Scanpy, POT, simulation truth, cell labels, or the old local model.

An explicit validation cohort is recommended for the extended benchmark. With no explicit cohort, training makes a seeded modality-stratified 10% optimization holdout. Feature preprocessing must already be fitted only on the permitted training data. In the simple transductive benchmark, an internal optimization holdout does not turn the experiment into an independent held-out evaluation.

`encode` validates the frozen feature names and order and never updates BatchNorm statistics, preprocessing, or model parameters. It returns posterior means directly; it does not draw latent samples, borrow another cell's private latent vector, or predict cross-assay residuals. Covariates, if genuinely available, can be supplied as fixed finite arrays; simulated hidden states and true exposures are not covariates. This benchmark uses no extra covariates.

## Architecture and objective audit

The reference is the supplied paper, arXiv `2510.24987v2`, especially Eqs. 5–13 and Appendix A.1, together with the pinned authors' code.[^1][^2]

| Component | Adapter behavior | Fidelity basis |
| --- | --- | --- |
| Shared input | `log1p(raw aligned counts)` | Native `EmbeddingNet.forward` |
| Shared encoder | Unified dense/BatchNorm/LeakyReLU/dropout network | Native `Encoder` |
| Specific encoder | Second unified encoder; counts plus modality one-hot | Native `EmbeddingNet` |
| Decoder | One unified decoder; shared plus specific latent and actual covariates | Paper Eq. 5; native `Decoder` |
| Shared prior | Standard Gaussian | Paper Eq. 6; native `klLoss` |
| Specific prior | Modality-conditioned scalar mean/log-variance broadcast across dimensions | Native prior network; isotropic paper prior |
| Count likelihood | ZINB; library-scaled softmax mean, zero inflation | Paper Eq. 5; native ZINB implementation |
| Missingness | Feature-availability mask, scaled by total/available feature count | Paper Eq. 13; native `ZINBLoss` |
| KL reduction | Sum latent dimensions, then mean cells | Native KL functions |
| Alignment | Discriminator minimizes CE; VAE minimizes negative CE | Paper Eqs. 8–10 |
| Geometry | Within-modality full/shared posterior-mean distance discrepancy | Paper Eq. 11; native `isometric_loss` |
| Output | Deterministic shared and specific posterior means | Native inference return values |

The native geometry reduction is the sum over within-modality pairs divided by minibatch size, not a mean over all pair entries. Both full and shared means receive gradients; the full representation is not detached. Validation preserves this minibatch estimator and uses fixed batch composition. It is not an exact all-validation-pairs geometry calculation.[^2]

The original local implementation used modality-specific input/output adapters, a different encoder normalization, and different loss reductions. Keeping the names of the loss terms was insufficient to make it the same model. This directory does not import or silently fall back to that implementation.

## Explicit paper and source discrepancies

The paper and the public experiment script do not agree on every hyperparameter. These differences are exposed rather than hidden.[^1][^3]

| Item | Paper or upstream discrepancy | Default chosen here |
| --- | --- | --- |
| Dispersion | Paper feature-wise; upstream default feature-by-modality | `gene`, explicitly selected from the native decoder |
| Latent sizes | Appendix: 20 shared/20 specific; BMMC script: 30/30 | 20/20 |
| Batch size | Appendix: 128; BMMC script: 64 | 128 |
| Learning rate | Appendix: cosine schedule; BMMC script: constant | Adam `1e-3`, cosine per optimizer step |
| Three-assay loss weights | Appendix and BMMC: beta 5, gamma 10, alignment 10 | 5/10/10 |
| Width and dropout | Not specified in supplied methods; BMMC: 128/128 and 0.2 | 128/128 and 0.2 |
| Training duration | Appendix and source: at most 200 epochs | 200, validation early stopping |
| Patience | README example 10; BMMC script 25 | 10, explicit configuration |
| Feature budget | No exact common budget stated in supplied methods | Benchmark wrapper records its own choice |

The upstream prior network includes dropout and emits one mean and one log-variance per modality input; these source architecture details are preserved. Gene-cell and gene-modality dispersion remain explicit options but are not the paper-equation default. The benchmark must not select one based on assessment-world graph recovery.

## Corrected training behavior

The following changes are deliberate implementation corrections, not new alignment methods:

1. **Isolate adversarial updates.** Upstream leaves discriminator gradients from the VAE's negative-CE phase attached to discriminator parameters, then adds the next discriminator phase's gradients before stepping. The adapter clears discriminator gradients before its update and freezes discriminator parameters during the VAE update. Gradients still pass through the discriminator to the encoder. This implements the intended alternating minimization/maximization.
2. **Use all validation minibatches.** Upstream `validate_model` accumulates losses but returns the last batch's `loss`. The adapter returns the cell-weighted mean across every batch. Fixed Monte Carlo draws make successive validations comparable without consuming training RNG state.
3. **Restore the best checkpoint.** Upstream early stopping does not save or restore its best state. The adapter saves the best and last states and restores the best before encoding.
4. **Record reproducibility inputs.** Checkpoints include features, masks, modality order, configuration, native source commit/hashes, weights and BatchNorm buffers. Reports include count digests, software versions, actual stopping epoch, and zero-library counts.

Gradient norm clipping at 1, shuffled mixed-modality minibatches, and dropping the final incomplete training minibatch follow the upstream training convention. The model is not pretrained or given biological labels. CPU/GPU device choice is explicit; unavailable CUDA raises an error instead of silently changing hardware.

Masked likelihood terms have zero direct derivative with respect to the unavailable feature's independent likelihood inputs. A decoder softmax still couples all output logits through its normalizer, as in the authors' implementation. Therefore the mask must not be described as guaranteeing that every parameter associated with an unavailable output has zero gradient.

## Tests and provenance

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -m unittest discover \
  -s pipelines/empirical/integration/methods/scmrdr/paper_faithful/tests -v
```

The suite checks analytical masked-ZINB agreement, mask scaling, direct unavailable-feature gradients, KL dimensional reduction, within-assay geometry, zero-count numerical behavior, the unified encoder dimensions, paper dispersion shape, adversarial gradient isolation, complete validation aggregation, checkpoint equality, feature-order rejection, frozen held-out encoding, raw-count contracts, and refusal to overwrite a completed fit. These are numerical/software checks, not a proof of causal identifiability.

`upstream/LICENSE` is byte-identical to the pinned source. `upstream/loss.py` differs only in end-of-file blank lines. `upstream/model.py` differs only by an attribution header, end-of-file whitespace normalization, and removal of unused NumPy, matplotlib, SciPy, and POT imports. No computational source code is changed. Original hashes are in [upstream_provenance.json](upstream_provenance.json). The adapter and its upstream-derived package are distributed under the [upstream GPL-3.0 license](upstream/LICENSE); this does not assert a license for unrelated repository files.

## Production benchmark validation

On 2026-10-04, a production review checked all 12 primary fits for the six
causal development worlds under both count regimes, plus 20 completed
nonshared control or seed fits. The review reconciled JSON manifests, frozen
source bytes, command receipts, and training logs. Every audited fit used the
configuration described above, with finite component losses, the declared
learning-rate schedule, and consistent validation and stopping records.
Saved best epochs agree with the first minimum validation loss. Five of the
12 primary fits reached the 200-epoch cap; reaching the cap does not establish
convergence.

Each primary fit used 6,000 observed training cells and 1,500 separate
validation cells. Training drops its shuffled 112-cell final remainder per
epoch, while validation includes all cells with cell-weighted aggregation.
Feature selection used training counts only. All nine scored variables survive
as singleton gene mappings in the 2,009-gene union, and no cells were dropped
by these scMRDR fits. The six reviewed background-only controls exclude those
nine scored variables before processing.

Exports retain distinct `shared` and `private` arrays. Frozen benchmark
matching uses only `shared`; shared-plus-private coordinates are used only by
the declared diagnostic probes. Fitting does not consume hidden biological
state, true cell pairing, or diagnostic-cohort labels. The review excluded
active shared-state controls and the locked assessment-world bank.

The production review rehashed source, JSON, and log files and cross-checked
recorded binary identities across receipts. It did not reopen or independently
rehash H5AD, NPZ, or checkpoint contents. This verification scope supplements
the existing pipeline's input/checkpoint validation. The frozen adapter hash
is `0b2857139120573013d7e6bd30398523d2a6680bca8b7f162c2453a9c828874a`;
the benchmark wrapper hash is
`4ebfbb7fe86f264698af0cb53ad9a229e7587bfbbd7c1e4075ba340566828d23`.

Histone-to-gene sums, the benchmark feature budget, and external simulation
validation remain application choices. Validation uses one fixed Monte Carlo
draw and minibatch geometry. Checkpoints permit exact inference restoration,
but do not contain optimizer and scheduler states for exact training resumption.
These results support consistency with the paper model and its intended
training procedure; causal recovery is assessed separately in the integration
recoverability benchmark.

## References

[^1]: Sun et al. (2026 version of NeurIPS 2025 paper). *scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration*. https://arxiv.org/abs/2510.24987v2
[^2]: Sun et al. *scMRDR source code*, commit `7280e64ae19f7391c6f5adf00d9d353f79e96603`. https://github.com/sjl-sjtu/scMRDR/tree/7280e64ae19f7391c6f5adf00d9d353f79e96603/src/scMRDR
[^3]: Sun et al. *BMMC integration experiment script*, same commit. https://github.com/sjl-sjtu/scMRDR/blob/7280e64ae19f7391c6f5adf00d9d353f79e96603/experiments/BMMC_codes/main.py
