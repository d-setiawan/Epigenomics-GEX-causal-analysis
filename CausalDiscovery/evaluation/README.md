# Evaluation

This directory is the working area for external validation of the causal-discovery pipeline.

## Purpose

The graph-learning outputs live under `CausalDiscovery/outputs/...`.

This directory is for the external evidence that helps us evaluate those graphs, especially:

- `ePerturbDB`
- `ENCODE SCREEN`
- derived node-support tables used to highlight graph nodes

## Recommended layout

- `templates/`
  - column templates for external evidence tables
- `external/`
  - locus-focused TSV extracts prepared from `ePerturbDB` and `ENCODE SCREEN`
- `reports/`
  - optional benchmark summaries, overlap reports, and notes

You can create `external/` and `reports/` as needed; they are not required for the plotting workflow.

## External evidence table format

Use the template in:

- `templates/external_support_regions_template.tsv`

Expected columns:

- `chrom`
- `start`
- `end`
- `gene`
- `support_label`
- `external_id`
- `biosample`
- `notes`

The support-builder script only requires `chrom`, `start`, and `end`, but adding the other columns makes the output much more interpretable.

## Workflow

### 1. Prepare locus-focused external TSVs

Create two TSVs for the locus you want to evaluate:

- one from `ePerturbDB`
- one from `ENCODE SCREEN`

Best practice:

- keep the tables locus-focused instead of genome-wide
- include only the target gene and nearby candidate regulatory elements you want to compare against
- for `ENCODE SCREEN`, prefer cCREs or cCRE-target gene links from the most relevant immune/monocyte-like biosamples

### 2. Build node-level support annotations

Example:

```bash
python3 CausalDiscovery/scripts/build_node_support_table.py \
  --pc-dir CausalDiscovery/outputs/locus_matrices_matched/joint_v2/cd14_regions678_rawbins/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --locus-config CausalDiscovery/configs/loci/cd14_regions678.tsv \
  --eperturbdb-tsv CausalDiscovery/evaluation/external/cd14_eperturbdb.tsv \
  --encode-screen-tsv CausalDiscovery/evaluation/external/cd14_encode_screen.tsv
```

This writes:

- `node_support.tsv`
- `node_support_matches.tsv`

directly into the graph output directory.

### 3. Plot the graph

The plotter now auto-loads `node_support.tsv` when it exists:

```bash
python3 CausalDiscovery/scripts/plot_pc_graph.py \
  --pc-dir CausalDiscovery/outputs/locus_matrices_matched/joint_v2/cd14_regions678_rawbins/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr
```

Highlight scheme:

- orange outline = `ePerturbDB`
- purple outline = `ENCODE SCREEN`
- red outline = both sources

## Matching logic

The current support-builder uses:

- expression nodes:
  - matched by exact gene symbol
- coordinate region nodes such as `chr5:140612011-140612339__H3K27ac`:
  - matched by genomic interval overlap
- named region nodes such as `promoter_primary_tss__H3K4me3`:
  - matched by interval overlap after resolving coordinates from `--locus-config`

## How to use the sources well

### `ePerturbDB`

Use it as the stronger causal-support layer.

Good uses:

- flag region nodes whose intervals overlap experimentally perturbed enhancers
- flag the expression node if the target gene appears in those perturbation records
- prioritize loci where the graph recovers edges adjacent to perturbed regions

### `ENCODE SCREEN`

Use it as a broader regulatory-support layer.

Good uses:

- check whether recovered nodes overlap cCREs
- check whether distal nodes align with enhancer-like cCREs
- check whether promoter nodes align with promoter-like signatures or cCRE-target gene links

### Combining them

The most convincing nodes are those supported by both:

\[
\text{graph support} \cap \text{ePerturbDB} \cap \text{ENCODE SCREEN}
\]

In practice, that means:

- `ePerturbDB` helps with causal credibility
- `ENCODE SCREEN` helps with regulatory plausibility and broader annotation coverage

Together they give a good first-pass external benchmark even before new perturbation experiments.
