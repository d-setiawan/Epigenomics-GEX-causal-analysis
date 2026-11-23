# Soft distributional alignment: TRAIN seed 30001

Status: exploratory CA-BOSS v4 integration candidate; not opened on DEV or LOCKED.

## Frozen discovery evidence

The estimator was frozen before its first scorer read. It consumed only the
seven `background_matrix__*.npz` files in TRAIN world 30001. It did not read
focal tables, supplied embeddings, manifests, scorer/truth files, paired cells,
feature correspondences, or external data.

- Algorithm: uncertainty-weighted entropic optimal transport with multi-start
  orthogonal Procrustes refinement.
- Feature folds: two, reconciled row-wise only within the same assay.
- Population robustness: observed microstate masses shrunk 75% toward uniform.
- Source SHA-256:
  `0be8913212a47d7bb10de888f43f2d0b06cf38fc0bcaa50fd84b5fea8c09db65`
- Frozen embedding SHA-256:
  `0bf454fbffad6457c9c36802f7a90114c882508dd90b66e0c7db42d90d86671b`
- Runtime: 89.7 seconds on the seven-assay TRAIN world.

The complete information-boundary declaration, configuration, and file hashes
are in
`pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/soft_alignment_seed30001_frozen/frozen_evidence_manifest.json`.

## Post-freeze TRAIN evaluation

The scorer was opened only after the evidence hashes above had been written.
The hard baseline is the independently recomputed two-fold Hungarian/Procrustes
embedding used by the existing v4 benchmark.

| Metric | Hard baseline | Soft distributional |
|---|---:|---:|
| Mean non-RNA R2 under an RNA-fitted map to latent state | -0.001 | 0.572 |
| Mean non-RNA within-assay latent-state ceiling R2 | 0.753 | 0.881 |
| Mean absolute true-axis loading cosine over all assay pairs | 0.575 | 0.722 |
| Within-assay shared-state supports selected | 1/6 | 3/6 |
| Cross-assay shared-state supports selected | 0/3 | 1/3 |
| Slow peak-gene loading support selected | 0/1 | 1/1 |
| Sentinel-null supports selected | 0/15 | 0/15 |

Per-assay RNA-map transfer R2 for the soft estimator was 0.820 (H3K27ac),
0.283 (H3K27me3), 0.863 (H3K4me1), 0.870 (H3K4me2), 0.866 (H3K4me3),
and -0.271 (H3K9me3). Thus four of six chromatin assays were strongly aligned,
one was partial, and one retained the wrong orthogonal solution.

Before truth was opened, the transport diagnostics identified H3K9me3 as
ambiguous (near-optimal rotation dispersion 0.613) and also flagged H3K4me3
(0.391). H3K27me3 was not flagged despite its weak transfer R2, so the current
observable confidence measure is useful but incomplete.

## Interpretation

This is a material improvement over hard microstate matching and directly
improves the downstream loading-group estimand without introducing cell
pairing. It is not yet reliable enough to replace the integration layer without
an abstention or consensus mechanism. A post-freeze TRAIN diagnostic suggests
that aligning ambiguous assays against several independently confident assay
anchors and synchronizing the rotations can resolve H3K9me3, but that extension
must be frozen and evaluated on new TRAIN worlds before any DEV use.

Validation completed with 14 passing tests across hard alignment, soft
alignment, and cell-level loading groups. DEV and LOCKED were not accessed.

## Confidence-gated rotation synchronization

The reference-only estimator above was extended with all-pairs orthogonal
rotation synchronization. Pair fits are gated using only observed transport
separation and near-optimal rotation dispersion, checked for multi-assay cycle
support, and synchronized with RNA fixed as the gauge. Assays outside the
reference-connected confidence core abstain and do not enter the consensus.

The finalized implementation was frozen again before any further scorer read:

- Final frozen directory:
  `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/soft_alignment_sync_seed30001_frozen_v2`
- Source SHA-256:
  `de5f69d54de92a90ba9d98b9bd65037f5d2d1ca703de64a1fd0cde6f6d2b86ad`
- Frozen embedding SHA-256:
  `5c80d013d13016ff0b2765e4ab42258ff641bb18d1b4db29f777a07f78f338ab`
- Runtime: 294.9 seconds for 21 pair fits across seven assays.
- Observable confidence graph: seven accepted assays, 16 accepted pair
  rotations, and weighted mean synchronization residual 0.0252.

The earlier post-freeze TRAIN diagnostic transfers to the finalized artifact:
its coordinates differ from the diagnostic freeze only at nondeterministic
linear-algebra roundoff below `9e-15`. No DEV or LOCKED artifact was opened.

| Metric | Reference-only soft | Synchronized soft |
|---|---:|---:|
| Mean non-RNA R2 under an RNA-fitted map to latent state | 0.572 | 0.870 |
| Mean non-RNA within-assay latent-state ceiling R2 | 0.881 | 0.881 |
| Mean absolute true-axis loading cosine over all assay pairs | 0.722 | 0.996 |
| Within-assay shared-state supports selected | 3/6 | 2/6 |
| Cross-assay shared-state supports selected | 1/3 | 3/3 |
| Slow peak-gene loading support selected | 1/1 | 0/1 |
| Sentinel-null supports selected | 0/15 | 0/15 |

The last four rows use the generic global mutual-neighbor projection from the
standalone loading-group module. The synchronized coordinates make all three
cross-assay truth classes geometrically strong (best absolute loading cosines
0.966, 0.972, and 0.941) with no selected sentinel. The decrease in
within-assay and slow selections is competition for the same global neighbor
budget, not loss of state signal: all six within-assay truth classes have best
absolute loading cosine at least 0.923, and the slow peak-gene class is 0.924.
This supports the CA-BOSS v4 design choice to apply separate within-assay,
cross-assay, and peak-gene neighbor budgets.

The focused alignment suite now contains five tests, including an exact
non-contamination test: adding an observed isotropic count assay makes it
abstain, leaves all four accepted assay embeddings bit-for-bit unchanged, and
adds the configured uncertainty penalty. The combined hard alignment, soft
alignment, and loading-group suite completed with 15 passing tests.
