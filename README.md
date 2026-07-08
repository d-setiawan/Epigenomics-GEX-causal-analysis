# Epigenomics-GEX Causal Analysis

This repository connects single-cell gene expression with single-cell CUT&Tag histone-mark data for monocyte-focused causal graph discovery.

The current active workflow is:

$$
\text{CUT\&Tag preprocessing}
\rightarrow
\text{scGLUE joint integration}
\rightarrow
\text{monocyte pseudo-pairing}
\rightarrow
\text{nearby peak causal graph discovery}
$$

The project has three main areas:

- [sc_scripts](sc_scripts): raw CUT&Tag preprocessing, QC, peak calling, and matrix generation.
- [integration](integration): cross-modality integration, with `scGLUE joint_v2` as the active downstream base.
- [CausalDiscovery](CausalDiscovery): monocyte pseudo-pairing, nearby CUT&Tag peak matrix generation, and graph discovery with `PC`, `FCI`, and repo-local mixed-family `DAGMA`.

## Current Status

The active integration is `scGLUE joint_v2`. It is preferred over the current `scMRDR_gaussian` comparison run because it has better modality mixing and keeps a region-aware representation that fits the causal-discovery workflow.

The active graph-discovery setup uses:

- cell type: `monocyte`
- genes: `CSF1R`, `CD14`, `IL1B`
- pairing: RNA-anchored one-to-one pseudo-pairs in `scGLUE` latent space
- candidate regions: all mark-specific CUT&Tag peaks within $300{,}000$ bp of the TSS
- graph inputs: nearby peak matrices in both `log1p_norm` and `raw_counts` forms
- graph methods: `PC + KCI`, `FCI + KCI`, and mixed-family `DAGMA`

The current biological-support layer cross-references saved graph edges against
external regulatory evidence before plotting:

- local ePerturbDB downloads in `Data/external/eperturbdb`
- ENCODE Portal PBMC/monocyte GRCh38 peak files
- ENCODE SCREEN cCRE annotations
- JASPAR motif support using UCSC hg38 sequence lookup

Validated edges are highlighted in the final SVG/PNG graph plots. Unvalidated
edges are kept but drawn faintly, so graph discovery remains exploratory while
external evidence is interpreted as a post-hoc support layer.

## Preprocessing

The raw CUT&Tag preprocessing scripts live in [sc_scripts](sc_scripts).

The staged pipeline is:

1. Demultiplex ADT/HTO counts with [01_demux_adt_hto.R](sc_scripts/01_demux_adt_hto.R).
2. Count fragments per barcode with [02_count_fragments.py](sc_scripts/02_count_fragments.py).
3. Build clean cell metadata and clean barcode whitelists with [03_build_clean_cell_metadata.py](sc_scripts/03_build_clean_cell_metadata.py).
4. Filter fragments and call mark-specific peaks with MACS3 using [run_cuttag_stage2_call_peaks.sh](sc_scripts/run_cuttag_stage2_call_peaks.sh).
5. Build peak matrices with [05_build_chromatin_matrix.py](sc_scripts/05_build_chromatin_matrix.py).
6. Build fixed-bin matrices with [07_build_fixed_bin_matrix.py](sc_scripts/07_build_fixed_bin_matrix.py).

Important outputs per histone mark are:

- `*_clean_cells.tsv`: QC-passed cell metadata.
- `*_clean_barcodes.tsv`: barcode whitelist used downstream.
- `*_peaks.bed`: mark-specific MACS3 CUT&Tag peak intervals.
- `*_chromatin_clean.mtx`: cell-by-peak count matrix.
- `*_bin_chromatin_clean.mtx`: cell-by-fixed-bin count matrix.
- `*_bins.bed`: fixed-bin genomic intervals.

The active integration manifest is [integration/manifests/scglue_input_manifest.tsv](integration/manifests/scglue_input_manifest.tsv). It points to the fixed-bin matrices, clean-cell metadata, bins BED files, and peaks BED files for:

- `H3K27ac`
- `H3K27me3`
- `H3K4me1`
- `H3K4me2`
- `H3K4me3`
- `H3K9me3`

## Integration

Integration code lives in [integration](integration).

The two method tracks are:

- [integration/methods/scglue](integration/methods/scglue): active recommended track.
- [integration/methods/jianle](integration/methods/jianle): scMRDR-inspired comparison track.

The active run is:

- [integration/outputs/scglue/joint/joint_v2](integration/outputs/scglue/joint/joint_v2)

`joint_v2` writes shared latent embeddings for RNA and all histone-mark modalities. These embeddings are used for pseudo-pairing, not as the final source of local chromatin values.

The comparison run is:

- [integration/outputs/jianle/joint/scMRDR_gaussian](integration/outputs/jianle/joint/scMRDR_gaussian)

Current validation favors `scGLUE joint_v2` over `scMRDR_gaussian` for this project:

- `scGLUE joint_v2` mean same-modality neighbor fraction: about `0.255`
- `scMRDR_gaussian` mean same-modality neighbor fraction: about `0.569`
- `scGLUE joint_v2` modality silhouette is closer to $0$

## Graph Discovery

Causal graph discovery lives in [CausalDiscovery](CausalDiscovery).

Use the single CLI entrypoint:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --help
```

The active graph workflow is:

1. Use `scGLUE joint_v2` latent space.
2. Restrict to monocytes.
3. Build RNA-anchored one-to-one pseudo-pairs or RNA-anchor $k$-nearest-neighbor averaged chromatin profiles.
4. Generate nearby peak matrices for `CSF1R`, `CD14`, and `IL1B`.
5. Optionally apply structural peak filtering, currently $p_\mathrm{nonzero} > 0.01$.
6. Run `PC`, `FCI`, or `DAGMA`.
7. Build external support tables from ePerturbDB, ENCODE, and/or JASPAR.
8. Re-render graph plots with externally validated edges highlighted.

Active pairing root:

- [CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor](CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor)

Active dataset roots:

- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb): `log1p_norm`
- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts): `raw_counts`

Alternative RNA-anchor neighbor-averaged roots:

- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10): `log1p_norm`
- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts): `raw_counts`

Current broad-window RNA-anchor averaged roots:

- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10): `log1p_norm`
- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts): `raw_counts`
- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001): `log1p_norm`, filtered to $p_\mathrm{nonzero} > 0.01$
- [CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001](CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001): `raw_counts`, filtered to $p_\mathrm{nonzero} > 0.01$

The older `300kb` one-to-one and `300kb` RNA-anchor averaged roots contain the smaller nearby peak sets:

- `CSF1R`: $19$ nearby peak variables
- `CD14`: $43$ nearby peak variables
- `IL1B`: $24$ nearby peak variables

The current 1 Mb RNA-anchor averaged roots contain broader candidate sets before filtering:

- `CSF1R`: $69$ peaks, filtered to $60$
- `CD14`: $113$ peaks, filtered to $99$
- `IL1B`: $82$ peaks, filtered to $74$

Graph outputs live under:

```text
CausalDiscovery/outputs/methods/<method>/<dataset>/<gene>/<result-run>
```

For `DAGMA`, the output tree includes one extra family layer:

```text
CausalDiscovery/outputs/methods/dagma/<dataset>/<family-config>/<gene>/<result-run>
```

Each externally supported graph directory can contain:

- `node_support.tsv`: node-level external evidence summary.
- `node_support_matches.tsv`: record-level evidence matches.
- `edge_support.tsv`: edge-level validation result and highlight class.
- `support_report.md`: source counts, match counts, and interpretation notes.

Peak-to-expression edge validation is target-gene-aware: the peak must overlap
an external interval record whose target gene matches the expression endpoint.
Interval-only sources such as ENCODE peak files or JASPAR motifs support peak
nodes and peak-peak edges, but they do not by themselves validate a peak-to-gene
edge.

## Common Commands

Set up integration workspace:

```bash
python3 integration/scripts/setup_integration_workspace.py
```

Build monocyte pseudo-pairs:

```bash
python3 CausalDiscovery/scripts/causal_cli.py match --run-id joint_v2
```

Generate $300$ kb nearby peak matrices:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --window-bp 300000
```

Generate raw-count matrices:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

Run a `PC + KCI` sweep:

```bash
python3 CausalDiscovery/scripts/causal_cli.py sweep \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb \
  --methods pc \
  --depths 1,2 \
  --alpha 0.1
```

Run Gaussian-family DAGMA sweeps:

```bash
bash CausalDiscovery/scripts/run_dagma_log1p_gaussian.sh
```

Run raw-count DAGMA sweeps:

```bash
bash CausalDiscovery/scripts/run_dagma_rawcounts_family_presets.sh
```

## Notes

The older curated locus-matrix workflow is preserved under `CausalDiscovery/outputs/locus_matrices_matched` and `CausalDiscovery/scripts/legacy`, but the active discovery workflow now uses nearby CUT&Tag peaks with curated literature regions kept as evaluation annotations.

Elastic-net stability screening on the `log1p_norm` nearby-peak matrices is the candidate-reduction layer. It is association-based screening before causal discovery, not causal evidence by itself. To keep peak selection and graph discovery from sharing samples, use `elastic-net-screen --graph-split-fraction` (see [CausalDiscovery/README.md](CausalDiscovery/README.md) → Screening Plan); without it, selection and graph learning use the same rows and downstream tests are optimistically biased.

## Documentation Map

- [notes/](notes) — research/planning notes, identifiability derivations (`notes/Identifiability/`), the Spring 2026 final report, and the integration summary. These are now tracked in this repo.
- [CausalDiscovery/evaluation.md](CausalDiscovery/evaluation.md) and [CausalDiscovery/evaluation/](CausalDiscovery/evaluation) — evaluation framing, external-source docs, and the saved-run audit.
- [CausalDiscovery/evaluation/run_audit_report.md](CausalDiscovery/evaluation/run_audit_report.md) — audit of the saved graph runs (trust tiers, post-selection leakage, count-family DAGMA gaps, cell-type coverage of the external evidence).
- [integration/README.md](integration/README.md), [CausalDiscovery/README.md](CausalDiscovery/README.md), [sc_scripts/](sc_scripts) — per-stage READMEs and preprocessing scripts.
