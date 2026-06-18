# External Regulatory Evidence Sources

This file tracks the external evidence sources used by the biological-support
layer for local peak-gene graph evaluation.

## Current Supported Sources

### ENCODE Portal PBMC Peak Files

Purpose: cell-context regulatory support from released ENCODE peak files.

The support builder can query the ENCODE Portal for released human GRCh38 BED
peak files, download the selected files, and retain only peak rows overlapping
the graph node intervals. The default biosample term is:

```text
peripheral blood mononuclear cell
```

For the current monocyte-focused analyses, the most useful override is:

```text
CD14-positive monocyte
```

The source is written as `encode_pbmc` in `node_support.tsv`,
`node_support_matches.tsv`, and `edge_support.tsv`.

### ENCODE SCREEN cCRE Registry

Purpose: broad regulatory-plausibility support for graph-adjacent peak nodes.

Default online source:

```text
https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.bed
```

Fallback:

```text
https://storage.googleapis.com/gcp.wenglab.org/Registry-V3/GRCh38-cCREs.bed
```

The support builder streams the BED file, keeps only cCREs overlapping graph
node intervals, and writes matching rows into `node_support_matches.tsv`.

Current interpretation:

- `PLS`: promoter-like signature.
- `pELS`: proximal enhancer-like signature.
- `dELS`: distal enhancer-like signature.
- `CTCF-only` / `CTCF-bound`: architectural or boundary-style support.
- `DNase-H3K4me3`: accessible promoter-associated regulatory signature.

This registry is human GRCh38 and genome-wide. It is not PBMC-specific by
itself, so it should be treated as regulatory plausibility rather than
cell-type-specific proof.

### ePerturbDB Evidence

Purpose: stronger perturbation-style support when graph-adjacent peaks overlap
experimentally perturbed enhancers or regulatory elements linked to a target
gene.

The repository supports two ePerturbDB input modes.

Downloaded database mode uses the files placed in:

```text
Data/external/eperturbdb/
```

The local parser recognizes:

- `Database*.tsv*`: main database export. Uses `Chr`, `Start_38`, `End_38`,
  `Enhancer-id`, and `Gene (literature)` for enhancer-to-target-gene support.
- `enhancer_gene*.bed`: enhancer interval/nearest-gene annotation.
- `TFoverlapEnh*.bed`: enhancer interval/TF overlap context.

Run it with:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir <graph_dir> \
  --support-preset local_eperturbdb
```

Plot with ePerturbDB support highlighted:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir <graph_dir> \
  --online-support \
  --support-preset local_eperturbdb
```

Template mode supports any online or local TSV with these columns:

```text
chrom	start	end	gene	support_label	external_id	biosample	notes
```

Use `--eperturbdb-url` for a remote TSV or `--eperturbdb-tsv` for a local TSV.
The parser requires at least `chrom`, `start`, and `end`; `gene` enables exact
matching to expression nodes, and the remaining columns make plots and reports
interpretable.

### JASPAR Motif Evidence

Purpose: motif-level regulatory plausibility support for peak nodes, especially
for peak-peak edges or TF-centered follow-up.

The support builder can fetch JASPAR CORE motifs online, retrieve peak interval
sequences from the UCSC Genome Browser API, and scan each peak with a simple PWM
score. The source is written as `jaspar`.

Common focused usage scans one or a few motifs:

```text
--fetch-jaspar --jaspar-matrix-id MA0139.1
```

Unfocused scans are bounded by `--max-jaspar-motifs`, `--jaspar-min-score`, and
`--jaspar-max-peak-bp` to avoid slow broad sequence scans.

## Presets

The CLI provides support presets for common source bundles:

- `pbmc_encode`: ENCODE SCREEN plus ENCODE Portal `peripheral blood mononuclear cell` peaks.
- `monocyte_encode`: ENCODE SCREEN plus ENCODE Portal `CD14-positive monocyte` `DNase-seq` peaks.
- `monocyte_regulatory`: `monocyte_encode` plus JASPAR CTCF (`MA0139.1`) motif support.
- `local_eperturbdb`: downloaded ePerturbDB files from `Data/external/eperturbdb`.

Use a preset with `graph`, `plot`, or `support`:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir <graph_dir> \
  --online-support \
  --support-preset monocyte_regulatory
```

## Commands

Build support tables for one saved graph directory using ENCODE SCREEN:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/fci/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/fci_kci_alpha_0_05_depth_5_bg_tiered_distal_promoter_expr \
  --fetch-encode-screen
```

Build support tables using ENCODE Portal PBMC or monocyte peak files:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/fci/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/fci_kci_alpha_0_05_depth_5_bg_tiered_distal_promoter_expr \
  --fetch-encode-pbmc \
  --encode-biosample-term "CD14-positive monocyte" \
  --encode-assay-term DNase-seq \
  --max-encode-files 10
```

Build support using JASPAR motif evidence:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/fci/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/fci_kci_alpha_0_05_depth_5_bg_tiered_distal_promoter_expr \
  --fetch-jaspar \
  --jaspar-matrix-id MA0139.1 \
  --jaspar-min-score 0.75
```

This writes:

```text
node_support.tsv
node_support_matches.tsv
edge_support.tsv
support_report.md
```

`support_report.md` records requested sources, source URLs or access modes,
record counts, match counts, and validated-edge counts. Use it as the first
audit artifact when interpreting highlighted SVG edges.

Build support and plot in one command:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/fci/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/fci_kci_alpha_0_05_depth_5_bg_tiered_distal_promoter_expr \
  --online-support \
  --fetch-encode-pbmc \
  --encode-biosample-term "CD14-positive monocyte" \
  --encode-assay-term DNase-seq \
  --layout local
```

Add ePerturbDB-style online evidence:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/fci/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/fci_kci_alpha_0_05_depth_5_bg_tiered_distal_promoter_expr \
  --online-support \
  --fetch-encode-screen \
  --eperturbdb-url https://example.org/cd14_eperturbdb_support.tsv \
  --layout local
```

Additional template-compatible online sources can be added with:

```text
--external-support-url SOURCE_NAME=https://example.org/source.tsv
```

## Plot Semantics

Validated edges are colored by support source:

- orange: ePerturbDB only.
- purple: ENCODE support only (`encode_screen`, `encode_pbmc`, or another ENCODE-prefixed source).
- green: JASPAR motif support only.
- red: multiple source families.
- faint gray: no external support found.

The edge is considered validated using edge-aware support rules:

- Peak-expression edges: the peak endpoint must overlap interval evidence and
  the external target gene must match the expression endpoint. This is the main
  rule for ePerturbDB-style peak-to-gene evidence; interval-only sources remain
  node/context support but do not validate these edges.
- Peak-peak edges: interval evidence on either peak is treated as regulatory
  plausibility, not causal proof. JASPAR support belongs in this category.
- Expression-expression edges: gene-symbol evidence on either endpoint is
  recorded, but these edges are not the main target for the current local
  peak-gene graph workflow.

The `edge_support.tsv` file records the rule used in `validation_rule`.

## Next Source Adapters To Consider

- A dedicated ePerturbDB API/parser if a stable public endpoint is available.
- JASPAR motif scans for TF motif support in peak nodes.
- ENTEx/BLUEPRINT PBMC- or monocyte-specific tracks if they can be mapped
cleanly to the same interval-support template.
