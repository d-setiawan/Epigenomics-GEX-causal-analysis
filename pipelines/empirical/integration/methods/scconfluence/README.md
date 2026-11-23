# scConfluence count-input pilot

This isolated adapter runs the pinned upstream scConfluence architecture with
RNA, H3K27ac and H3K4me3 count inputs. Its published high-dimensional experiments used compressed inputs. The project adaptation receives all retained features and uses negative-binomial reconstruction for each assay.

The fixed pilot completed six GPU fits across two cached synthetic worlds and the scMTR-seq S1 cohort, each with two seeds. Matching varied by seed. The one recovered full-count synthetic cross-assay adjacency joined shared-state decoys. In S1, the paired 595-cell panel yielded no positive cross-assay adjacency under the specified graph readout. [Study and reliability results](../../../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)

The upstream source is preserved under [upstream/](upstream/) at commit
`14d81e2ca679e161ec5c6806103a66c14fcdcd9a`. [UPSTREAM.json](upstream/UPSTREAM.json)
records file hashes and provenance, and the original GPL-3.0 license is retained.
The adapter validates those bytes before importing the package. No installed
scConfluence package or shared conda environment is modified.

## Corrections to the native implementation

[loss_patch.py](loss_patch.py) contains two narrow corrections:

1. The upstream pair loop overwrites and self-doubles its accumulators, retaining
   only the last eligible pair. The corrected loop sums every eligible pair once
   and applies the native IOT/Sinkhorn coefficients once. Reconstruction terms,
   partial-OT solver, Sinkhorn regularizer and their numerical definitions are
   unchanged. A one-pair objective therefore also loses the erroneous factor of
   two.
2. A mixed minibatch can contain no observations from one assay. Native
   reconstruction and norm logging then take an empty mean and produce NaN.
   The local override skips only absent-assay reconstruction/log terms; native
   pair losses already skip missing pairs. No observed cells are discarded or
   padded, and contributions from nonempty assays retain their native weights.

The second failure was reproduced with the real native forward pass before
correction. The regression test checks finite corrected losses and gradients.
The three-pair test independently checks contributions and gradients under every
pair ordering and verifies the one-pair weighting.

## Inputs and fitting

[pilot_adapter.py](pilot_adapter.py) accepts a directory containing three
`<assay>.h5ad` files. `X` must contain finite nonnegative integer counts;
`var.gene_id` supplies permitted gene/region correspondences. Only matrix values,
feature names, gene annotations and observation IDs are used. Other `obs`,
`obsm`, `uns` and annotation columns are not copied into learner objects.
Assay observation IDs must be disjoint; the caller is responsible for ensuring
these IDs represent genuinely disjoint underlying cells rather than renamed
paired observations.

For the cached simulation comparison, `selection_paths` points to the three
existing scGLUE prepared training H5ADs. The adapter reads only their HDF5 index
datasets, selecting exactly the same raw count rows and features, in reference
order. Input provenance still names and hashes the original observed count
files. Reference file, selected-cell and selected-feature hashes are recorded
separately. It does not read reference embeddings. No independent 2,000-row cap
is applied after this selection.

For scMTR-seq, the caller supplies already split and masked training views,
including any predeclared outcome-panel exclusions, with `selection_paths=None`.
All training-positive features are retained. An all-zero library is excluded
with its ID recorded. Held-out views use the frozen training feature indices
and annotations, with no new feature selection or fitted normalization.

The encoder receives `log1p(10,000 × count / retained observed library)` and
the decoder reconstructs the original counts. No PCA or HVG ranking is used.
For alignment costs, the adapter reuses the repository's `annotation_matrix`
sum mapping. It takes all common annotated genes, normalizes their sums per
cell, removes genes constant in any training assay, and uses per-assay training
feature scaling followed by Pearson correlation distance. This follows the
official tutorial's cost construction while removing its HVG ranking. Zero
guide libraries, too few variable genes and undefined correlation rows fail
explicitly; no invented cost or hidden state fills them in.

All three pair-cost matrices are constructed from training cells. The native
model splits those supplied training cells internally for early stopping.
Preprocessing thus uses the whole supplied training cohort, including its
internal validation fraction; public validation/discovery/test cohorts are
never used for fitting costs, feature selection, scaling or model parameters.

The fixed pilot configuration is seed 0 or 1, latent dimension 30, hidden width
128, batch size 128, at most 120 epochs, patience 8, internal validation fraction
0.2 and learning rate 0.003. Native alignment defaults remain mass 0.5, reach
0.3, blur 0.01 and IOT/Sinkhorn weights 0.01/0.1. Native encoder/decoder depths
are 3/2, dropout is zero, and each assay's NB reconstruction has weight one and
averages over features. These are one predeclared setting, not tuned results.

Native `.fit()` uses AdamW, early stopping and best-checkpoint restoration.
The adapter saves a plain tensor checkpoint, reloads it independently and checks
training posterior-mean embeddings before completing the manifest. Held-out
exports use the frozen native encoder posterior mean directly; they do not
replace the model's training dataset, refit the model or compute held-out costs.
Training cell IDs cannot be reused as a held-out cohort.

## Output contract

- `model_manifest.json`: complete status, `method="scconfluence"`, full
  configuration, original training input paths/hashes, selection references,
  preprocessing/checkpoint/source hashes, native input dimensions, versions and
  checkpoint roundtrip errors.
- `preprocessing.json`: raw feature/annotation order, selected indices,
  training cell IDs and guide construction. `checkpoint.pt` contains only
  tensors, configuration and the preprocessing hash.
- `<cohort>/<assay>.npz`: exactly `cell_ids` and `shared`; raw learner-visible IDs
  are preserved, with no state or paired-cell metadata.
- `<cohort>/encoding_manifest.json`: input and checkpoint/preprocessing hashes,
  plus `exports[assay]` with path, SHA256, cell count, latent dimension,
  dropped IDs and `cell_ids_sha256=SHA256('\n'.join(ids).encode())`.

Existing fit/cohort output directories cannot be overwritten. Failed fits keep
`failure.json` and their partial outputs. The enclosing study runner must bind
these artifacts to its predeclared protocol and verify common evaluation pools.

## Runtime and checks

The [recorded pilot requirements](../../../../../envs/requirements-scconfluence-pilot-2026-11-01.txt) document Python packages for the historical fit. The pilot used Python 3.11, NumPy 1.26.4, AnnData
0.10.9, MuData 0.3.1, POT 0.9.6.post1, GeomLoss 0.2.6, PyTorch 2.10.0 and
PyTorch Lightning 2.6.1. Python 3.11 is an explicit deviation from the pinned
package's Python `<3.11` metadata, whose comment attributes that bound to an old
scikit-misc installation issue. This adapter does not import scikit-misc,
scanpy, muon or louvain. Compatibility is tested rather than inferred from the
metadata comment. Upstream pyproject version 0.1.1 differs from its exported
`__version__` 0.1.0; the commit and source hashes define the implementation.

The focused suite exercises loss sums/gradients/order, absent-assay minibatches,
exact reference selection, train-only transformations, invalid counts,
degenerate guide rejection, a real two-epoch native CPU fit with mixed
minibatches, checkpoint reload, held-out export and tamper/no-overwrite guards.
These tiny fixtures are software checks, not new scientific simulation worlds.
The [study runner](../../../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/run_fixed_pilot.py) ran the serialized GPU jobs and scoring. The November transfer rebuilt the cohort and learner H5ADs. Full GPU training was outside that replay. [Transfer verification](../../../../../reproduce/VERIFICATION.md)

Source methods and interpretation are discussed in the [prior review](../../../../../project_resources/research/integration_followup_2026_10_31/alternative_methods.md),
the [paper](https://doi.org/10.1038/s41467-024-51382-x), and the
[official tutorial](https://scconfluence.readthedocs.io/en/latest/tutorials/RNA_ATAC_pbmc_tutorial.html).
Half-mass minibatch transport is a training device. Forced full-pool matching of
exported coordinates is a separate diagnostic, and neither it nor a reconstructed
joint distribution establishes causal direction or the unobserved residual
coupling of disjoint cells.
