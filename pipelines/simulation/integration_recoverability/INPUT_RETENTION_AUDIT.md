# Input-retention audit

_Extended informative-count development worlds, integration seed 0; 2026-10-04._

## Verdict and scope

**Focal-feature omission and focal dilution by gene sums do not explain the negative same-assay residual probes in these two runs.** All nine scored variables survive scGLUE's encoder-feature and reconstruction-feature selection. All nine also survive scMRDR's assay-wise HVG selection and gene union, each under a singleton, assay-specific annotation. This establishes input availability, not preservation of information in the learned coordinates.

This read-only audit covers `linear_chain_0` and `constrained_chain_0`, `informative/primary`, seed 0. It compares saved manifests with raw/prepared AnnData metadata and the existing probe reports. Training used the first 2,000 cached cells per assay before native QC, not all 4,000 cells in each raw training file. No datasets, fits, probes, settings, thresholds, source files, or tests were changed or rerun. The only new artifact is this note.

The two run roots are [linear](runs/extended/linear_chain_0/informative/primary/) and [constrained nonlinear](runs/extended/constrained_chain_0/informative/primary/). Paths below are relative to those roots unless stated otherwise. The worlds have different simulation seeds and mechanism draws; this is not a controlled contrast isolating nonlinearity.

## scGLUE: selection retains every scored variable

Each assay has three scored variables: `core`, `decoy`, and `sentinel`. Thus 3/3 includes the core, not just background controls.

| World | Assay | Raw features | Normalization features | Encoder features | Reconstruction features | Scored retained, encoder / reconstruction | Training cells after QC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | RNA | 2,000 | 1,944 | 1,944 | 1,944 | 3/3 / 3/3 | 1,936 |
| Linear | H3K27ac | 15,000 | 15,000 | 15,000 | 15,000 | 3/3 / 3/3 | 2,000 |
| Linear | H3K4me3 | 15,000 | 15,000 | 15,000 | 15,000 | 3/3 / 3/3 | 2,000 |
| Constrained | RNA | 2,000 | 1,949 | 1,949 | 1,949 | 3/3 / 3/3 | 1,942 |
| Constrained | H3K27ac | 15,000 | 15,000 | 15,000 | 15,000 | 3/3 / 3/3 | 2,000 |
| Constrained | H3K4me3 | 15,000 | 15,000 | 15,000 | 15,000 | 3/3 / 3/3 | 2,000 |

The RNA feature losses occur at the training detection/QC stage; the 3,000-HVG budget exceeds the retained panel. Every chromatin column has positive training counts, and the 30,000-feature encoder cap and 15,000-feature reconstruction cap retain the entire 15,000-column panel. These statements come from `scglue_preprocessing/manifest.json` and each referenced `transforms/{assay}.json`, specifically `normalization_feature_ids`, `encoder_feature_ids`, `reconstruction_feature_ids`, and `audit`.

The encoder does **not** receive these thousands of columns as separate neural inputs: RNA is compressed into 50 PCA coordinates and each mark into 50 LSI coordinates. The fitted scGLUE encoder then produces 30 shared coordinates. The retained original counts separately supply the NB reconstruction objective. [scglue_adapter.py](scglue_adapter.py) explicitly uses the saved representation and `layers["counts"]`; all prepared reconstruction columns are marked `highly_variable=True`.

For all nine scored columns in both worlds, the prepared reconstruction counts were checked to equal the original raw counts exactly on retained training cell IDs. Saved `probe` and `assessment` cohort manifests reference the same fitted preprocessing-manifest hash and retain the same feature order. Presence in a PCA/LSI input panel does not imply an individual variable's residual variation survives that compression.

All nine scored IDs are excluded from cross-feature guidance by the recorded `guidance_excluded_feature_ids` configuration, but not from the primary count input. This is a deliberately masked prior, not an omitted measurement.

## scMRDR: background gene sums are not focal gene sums

[gene_features.py](gene_features.py) sums raw columns sharing `gene_id`, selects up to 2,000 HVGs per assay using training cells, and freezes their union. Both runs have **2,009 union features**. Availability masks represent measured annotations, not whether a count happened to be zero.

| World | Assay | Original columns used | Measured annotation genes | Assay-selected HVGs | Measured genes retained in union | Genes pooling multiple original columns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear | RNA | 2,000 | 2,000 | 1,992 | 2,000 | 0 |
| Linear | H3K27ac | 15,000 | 2,002 | 2,000 | 2,002 | 1,989 |
| Linear | H3K4me3 | 15,000 | 2,002 | 2,000 | 2,002 | 1,989 |
| Constrained | RNA | 2,000 | 2,000 | 1,994 | 2,000 | 0 |
| Constrained | H3K27ac | 15,000 | 2,000 | 2,000 | 2,000 | 1,989 |
| Constrained | H3K4me3 | 15,000 | 2,002 | 2,000 | 2,002 | 1,995 |

The lower RNA HVG counts reflect the positive-training-count restriction before selection. Other assays' selections restore the remaining measured RNA annotation genes to the union. However, **every scored gene is already in its own assay's HVG set**; none depends on such restoration.

For all 18 scored feature instances across the two worlds, the audit checked `gene_transform/manifest.json` fields `assays.{assay}.input_features`, `input_gene_annotations`, `used_feature_indices`, `selected_hvg_genes`, `availability`, and the global `feature_names`. Every focal annotation is `g_unlinked_*`, has exactly one contributing original column, and is structurally available only in its own assay. No scored focal column is mixed with a background bin or another scored column.

The exact core mappings are:

| World | Core | Original feature ID | Singleton annotation gene |
| --- | --- | --- | --- |
| Linear | RNA | `f_5c945b1aaa9adec344b5` | `g_unlinked_49854300c7bb2b5ca36b` |
| Linear | H3K27ac | `f_60971c0b5dfbbafd05a8` | `g_unlinked_9733910be467a78bff00` |
| Linear | H3K4me3 | `f_a8d56f81ccf9ce1967fb` | `g_unlinked_4af94cf0eee91e09abfb` |
| Constrained | RNA | `f_ddacecaf6d9fb04731e4` | `g_unlinked_084e4dd19f0ea7eefeca` |
| Constrained | H3K27ac | `f_2cd05f9e4dbaea44ae87` | `g_unlinked_ec8c02f4a19fbce519c9` |
| Constrained | H3K4me3 | `f_2c5cb93aaf0d7245999c` | `g_unlinked_e8b41d7341caeee09023` |

Consequently, each focal gene-space count equals the original focal count before the native encoder's `log1p` transform. This is different from the many-to-one representation of most background chromatin features. scMRDR then compresses the input into 20 shared and 20 private coordinates; singleton input retention does not establish residual retention in either representation.

## What the saved residual probes show

Targets are hidden activities minus simulation Monte Carlo estimates of their state-conditional means, not the observed counts themselves. Probes were fitted on private calibration cells and evaluated on separate private assessment cells, without refitting integration. The fixed probe families are ridge and shallow boosted trees. The following are saved `embedding_only.r2` scores for **same-assay core-residual targets**, rounded to six decimals.

| World | Representation | Probe | RNA core | H3K27ac core | H3K4me3 core |
| --- | --- | --- | ---: | ---: | ---: |
| Linear | Native PCA/LSI | Ridge | -0.063073 | -0.063665 | -0.070499 |
| Linear | Native PCA/LSI | Tree | -0.026886 | -0.032789 | -0.027389 |
| Linear | scGLUE shared | Ridge | -0.025197 | -0.039843 | -0.069530 |
| Linear | scGLUE shared | Tree | -0.026339 | -0.040998 | -0.048085 |
| Linear | scMRDR shared | Ridge | -0.026848 | -0.048199 | -0.052345 |
| Linear | scMRDR shared | Tree | -0.015889 | -0.036986 | -0.026125 |
| Linear | scMRDR shared + private | Ridge | -0.028015 | -0.048876 | -0.048425 |
| Linear | scMRDR shared + private | Tree | -0.020065 | -0.039645 | -0.026962 |
| Constrained | Native PCA/LSI | Ridge | -0.052754 | -0.063811 | -0.064195 |
| Constrained | Native PCA/LSI | Tree | -0.031092 | -0.038533 | -0.030498 |
| Constrained | scGLUE shared | Ridge | -0.019335 | -0.027604 | -0.038605 |
| Constrained | scGLUE shared | Tree | -0.023658 | -0.022911 | -0.043817 |
| Constrained | scMRDR shared | Ridge | -0.016361 | -0.021371 | -0.030708 |
| Constrained | scMRDR shared | Tree | -0.021727 | -0.042701 | -0.032817 |
| Constrained | scMRDR shared + private | Ridge | -0.019014 | -0.021143 | -0.038607 |
| Constrained | scMRDR shared + private | Tree | -0.021597 | -0.042044 | -0.030072 |

Including decoys and sentinels, all 18 native-preprocessing rows, all 18 scGLUE rows, and all 36 scMRDR rows have negative same-assay residual R² in **each** world. Source fields are `preprocessing_probes.representations.rows` and `probes.{method}.rows`, filtered by `relationship="same_assay"` and `target` beginning with `residual::`. This statement is about embedding-only R², not a claim that every incremental state-plus-embedding MSE gain is negative.

Native preprocessing and scGLUE use RNA calibration/assessment populations of 970/1,930 in the linear world and 968/1,935 in the constrained world. Their mark probes, and all scMRDR probes, use 1,000/2,000. Thus the method comparisons do not all concern identical retained RNA populations.

Although labeled informative, the actual focal measurements are still frequently zero. The first 2,000 raw training cells have core detections of RNA/H3K27ac/H3K4me3 = **181/189/379** in the linear world and **164/150/531** in the constrained world. These are descriptive input checks, not a power analysis or proof that counts cannot carry residual information.

Negative test R² means worse squared-error prediction than the constant equal to the assessment target mean. It does not prove zero mutual information, conditional independence, or irrecoverability by every estimator. The native PCA/LSI control is already negative before scGLUE fitting, so attributing all failure to integration would be unwarranted. This audit cannot apportion failure among noisy count observation, lossy native compression, learned representation objectives, finite probe samples, restricted probe capacity, and Monte Carlo target noise. In particular, same-assay probes are not intrinsically blocked by absent cross-assay pairing.

The bounded conclusion is that the existing pipeline has **not demonstrated residual recovery despite retaining the focal inputs**. Feature-selection omission and focal gene-sum dilution are ruled out as explanations in these two runs; the remaining mechanisms are not isolated. QC-selection and source-graph caveats in [SPARSE_CI_AUDIT.md](SPARSE_CI_AUDIT.md) still apply.

## Verification and provenance

Raw AnnData feature and annotation order matched the gene manifests. The raw training-file hashes agreed across scGLUE preprocessing and scMRDR receipts. Referenced transform JSONs, transform parameter files, prepared training H5AD files, and training representation NPZs matched their recorded hashes. The report's integration-manifest hashes and scMRDR's linked gene-manifest hash also matched. No model was loaded for fitting or encoding during this audit.

| World | Artifact | SHA-256 |
| --- | --- | --- |
| Linear | [Combined report](runs/extended/linear_chain_0/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json) | `96c4b34e68fc5782d5bd06fef8b46d953ce45bca6c3a1dae98a5ebaa5d1e8340` |
| Linear | [scGLUE preprocessing](runs/extended/linear_chain_0/informative/primary/scglue_preprocessing/manifest.json) | `dfb97922a30c2c706932985adcdb8a8fd64ba184cd654b6b197124e141bee9de` |
| Linear | [Gene mapping](runs/extended/linear_chain_0/informative/primary/gene_transform/manifest.json) | `c55186303c42973c7282c9050e921bb198eb9db675e575c55f60efb0c2cc4a2e` |
| Constrained | [Combined report](runs/extended/constrained_chain_0/informative/primary/evaluation_scglue_scmrdr_seeds0/report.json) | `1bff56125eb007e6d9e448b230a8cc96f9f80789566cffcbcb3d82d63e88a227` |
| Constrained | [scGLUE preprocessing](runs/extended/constrained_chain_0/informative/primary/scglue_preprocessing/manifest.json) | `46914e44f7cf662810b1eef7c270e75ab70a5f99eb3f9a69627577eddc582d89` |
| Constrained | [Gene mapping](runs/extended/constrained_chain_0/informative/primary/gene_transform/manifest.json) | `3c69dc2e8ae96c46a659a35bac121ef45d6a2c67ef1477caa2d7c63186dd0983` |

Inspected source hashes:

```text
preprocessing.py
8f430c656e885b02fbb9fa00fc605092cf12745b141f44a17dfacbaf931b6acb
gene_features.py
57b008b292d08605fb1898d9f6f0e3b867bca29eb51868be3fc3b58e78b3bfbb
scglue_adapter.py
e021d91f0f8ea93b90f9192efd7715ae68fdfe86e0f4643a420c5516fa18db5f
evaluate.py
f2c8536be44fe07081ed582cc1e79deb55fd1d962319e9968931799ee13eb171
```
