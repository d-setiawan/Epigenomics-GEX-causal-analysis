# External Regulatory Evidence Sources

This file tracks the external evidence sources used by the biological-support
layer for local peak-gene graph evaluation.

## Cell-Type Alignment And Coverage (2026-07 check)

Our data is CD14-positive / classical **monocyte** (RNA harmonized label `monocyte`).
An audit of how the support layer aligns to that cell type found:

- **Cell type is not used as a match/filter anywhere except the ENCODE Portal API query.**
  In `build_node_support_table.py` a record matches a node on exactly two axes: genomic
  interval overlap and (for expression nodes) target-gene-name equality. The tissue/cell-type
  field is read from ePerturbDB but only written into the descriptive `biosample` string — it
  never rejects a record. So `encode_screen`, `jaspar`, and all `eperturbdb` sources can
  "support" a monocyte edge with breast/skin/hESC/K562 evidence. Treat their matches as
  cell-type-agnostic plausibility, not monocyte validation.
- **ENCODE Portal is the only cell-type-gated source** (server-side `biosample_ontology.term_name`).
  ENCODE has usable monocyte data: `CD14-positive monocyte` returns **159** GRCh38 peak-bed
  files, `peripheral blood mononuclear cell` returns 25. So for cell-type-aligned interval
  support, prefer the `monocyte_encode` / `monocyte_regulatory` presets over `pbmc_encode`
  and over the code default (which is PBMC).
- **ePerturbDB cannot validate our peak-to-gene edges.** Its content is overwhelmingly
  non-immune (Skin ~50k, Bone Marrow ~11k, embryonic ~10k, Breast ~6k rows; ~2 immune rows).
  Crucially, `CSF1R` and `CD14` have **0** rows in the `Gene (literature)` column and `IL1B`
  has **0** (its 96 hits are only `nearest_gene_name`, which the validation does not use).
  Since ePerturbDB is the only source carrying target-gene semantics, **no current source can
  validate a peak-to-gene edge (e.g. a promoter-peak -> expression edge) for our genes.**
- **Net:** the suite can give cell-type-aligned **peak-node / peak-peak interval** support
  (ENCODE-monocyte), plus cell-type-agnostic context (SCREEN, JASPAR). It **cannot** give
  cell-type-aligned **peak-to-gene edge** validation for CSF1R/CD14/IL1B with the sources on
  hand. See `run_audit_report.md` for the full breakdown and options.

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
Data/raw/external/eperturbdb/
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

### ENCODE-rE2G Monocyte Gene Links (target-gene-aware, cell-type-aligned)

Purpose: the one source that can **validate peak-to-gene edges** for our monocyte panel.
Unlike ENCODE peaks / SCREEN / JASPAR (interval/motif-only) and ePerturbDB (no monocyte
coverage), ENCODE-rE2G predictions are native GRCh38, CD14-positive-monocyte-specific, and
carry an explicit `TargetGene` per regulatory element.

Build the support table (fetches all CD14+ monocyte donor datasets from ENCODE, merges the
`thresholded element gene links`, and writes the support-template TSV):

```bash
python3 Evaluation/scripts/build_re2g_gene_link_support.py \
  --biosample-term "CD14-positive monocyte" \
  --out-tsv Data/raw/external/re2g/re2g_cd14_monocyte_gene_links.tsv
```

Then feed it to the support layer as the first-class **`re2g`** source (target-gene-aware; the
template `gene` column enables exact matching to expression nodes):

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir <graph_dir> \
  --re2g-tsv Data/raw/external/re2g/re2g_cd14_monocyte_gene_links.tsv
```

Matches are labeled `source=re2g` with `highlight_class=re2g_only` (plot color blue, `#2E86DE`).
A peak node overlapping an rE2G element whose `TargetGene` matches the expression node validates
that peak-to-gene edge. Worked example (CD14 300 kb graph): the stable
`chr5:140638835-140639790__H3K4me3 -> expr__CD14` edge stays **unvalidated** because rE2G links
that interval (which overlaps the TMCO6 TSS) to **TMCO6**, not CD14 — i.e. the source correctly
flags a likely neighbor-promoter artifact. The true CD14 promoter element is
`chr5:140633141-140633951` (rE2G score 0.9999).

### BLUEPRINT Monocyte PCHi-C Gene Links (physical peak->gene axis)

Purpose: the **physical** peak->gene axis (promoter-capture Hi-C), complementing rE2G (prediction)
and eQTL (genetics). BLUEPRINT promoter-capture Hi-C (Javierre et al., Cell 2016) links a gene
promoter (bait) to distal fragments (other-end) with a per-cell-type CHiCAGO score; the monocyte
column is `Mon` (>= 5 is significant). So a peak overlapping a fragment that physically contacts a
target gene's promoter in monocytes is physical evidence for a peak->gene edge.

Build the support table (downloads the OSF peak matrix, filters to the panel genes with `Mon >= 5`,
and lifts hg19 -> hg38 via `scripts/liftover.py`'s chain):

```bash
python3 Evaluation/scripts/build_pchic_gene_link_support.py \
  --genes CSF1R,CD14,IL1B --cell-type-col Mon --min-score 5 \
  --out-tsv Data/raw/external/pchic/pchic_mon_gene_links.tsv
```

Feed it as the local named source `pchic` (plot color pink, `#E84393`):

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir <graph_dir> \
  --external-support-tsv pchic=Data/raw/external/pchic/pchic_mon_gene_links.tsv \
  --re2g-tsv Data/raw/external/re2g/re2g_cd14_monocyte_gene_links.tsv
```

Worked example (CD14 300 kb graph): PCHi-C matched CD14 CUT&Tag peaks that physically contact the
CD14 promoter in monocytes (e.g. `chr5:140547304-140547778__H3K27ac`), and **15 nodes gained
`multi_source` support** — corroborated by *both* rE2G and PCHi-C (the "two axes agree" case). The
`chr5:140638835-140639790__H3K4me3` stable-edge peak is NOT among them (it is the TMCO6 promoter,
not a CD14 contact), consistent with rE2G.

### eQTL Catalogue Monocyte eQTL (genetics peak->gene axis)

The GENETICS axis: eQTL Catalogue fine-maps each gene's cis-eQTL to a SuSiE 95% credible set of
variants. A credible-set variant that falls INSIDE a CUT&Tag peak, for a credible set targeting the
edge's gene, is independent genetic evidence the peak carries regulatory sequence for that gene.

```bash
python3 Evaluation/scripts/build_eqtl_gene_link_support.py \
  --out-tsv Data/raw/external/eqtl/eqtl_mon_gene_links.tsv   # all monocyte ge datasets, full credible sets
```

It reads eQTL Catalogue's `tabix_ftp_paths.tsv`, keeps monocyte `ge` datasets, downloads their
`*.credible_sets.tsv.gz` (cached under `Data/raw/external/eqtl/`, GRCh38-native — no liftOver), filters to
the target Ensembl genes, and emits one `[pos,pos+1]` interval per credible-set variant. Feed it as the
local named source `eqtl` (plot color teal, `#16A085`):

```bash
causal_cli.py support --graph-dir <dir> \
  --external-support-tsv eqtl=Data/raw/external/eqtl/eqtl_mon_gene_links.tsv
```

Result for the panel: **19 variant-links — CD14=17, CSF1R=2, IL1B=0.** ⚠️ **Cell-state caveat:** the
only monocyte credible sets for CD14/CSF1R are from **TLR-stimulated** monocytes (Quach_2016
Pam3CSK4_6h / R848_6h), not resting — cell-type aligned but not state aligned. Naive-monocyte datasets
(BLUEPRINT QTD000021, Quach naive, Schmiedel, Randolph) have **zero** credible sets for the three
targets; IL1B has none in any monocyte dataset. `--min-pip` tightens to well-localised variants; the
CSF1R credible set is a single variant at PIP≈0.99, CD14's is 16 variants at the promoter (PIP≈0.08–0.16).

### Generic local named sources

Any support-template TSV can be added as a first-class named source with
`--external-support-tsv SOURCE=PATH` (the local analog of `--external-support-url SOURCE=URL`).
This is how new Phase-B validators (pchic, and later eqtl / blueprint_chip / chipatlas) plug in
without per-source CLI plumbing. `pchic` gets its own source family and color; add colors for new
families in `plot_pc_graph.py` `support_style()` and `source_family()`.

## Presets

The CLI provides support presets for common source bundles:

- `pbmc_encode`: ENCODE SCREEN plus ENCODE Portal `peripheral blood mononuclear cell` peaks.
- `monocyte_encode`: ENCODE SCREEN plus ENCODE Portal `CD14-positive monocyte` `DNase-seq` peaks.
- `monocyte_regulatory`: `monocyte_encode` plus JASPAR CTCF (`MA0139.1`) motif support.
- `local_eperturbdb`: downloaded ePerturbDB files from `Data/raw/external/eperturbdb`.

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
- blue: ENCODE-rE2G only (`re2g` — the target-gene-aware monocyte gene-link source).
- pink: BLUEPRINT monocyte PCHi-C only (`pchic` — physical promoter contact).
- teal: eQTL Catalogue monocyte eQTL only (`eqtl` — genetics; credible-set variant in peak).
- red: multiple source families (**this is the "two-or-more axes agree" high-confidence case**).
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
