# CausalDiscovery

This directory is the repo-native home for causal graph discovery work built on top of the joint `scGLUE` integration outputs.

## Goal

Use the joint `scGLUE` embedding to pseudo-pair cells across modalities inside one cell type, then export gene-centric locus tables that can be used as inputs for causal discovery methods such as the PC algorithm.

## Recommended layout

- `configs/`: run settings for causal-discovery experiments
- `notes/`: design notes, gene panels, method comparisons
- `scripts/`: causal-discovery runners and export utilities
- `outputs/`: generated tables, graphs, and run summaries

## Current starting point

The strongest current `scGLUE` run is:

- `integration/outputs/scglue/joint/joint_v2/`

The most important files for causal discovery are:

- `integration/outputs/scglue/joint/joint_v2/train/all_cells_glue_embeddings.tsv`
- `integration/outputs/scglue/joint/joint_v2/train/validation/joint_harmonized_label_transfer.tsv`
- `integration/outputs/scglue/joint/joint_v2/train/modalities/rna_with_glue.h5ad`
- `integration/manifests/scglue_input_manifest.tsv`

The key distinction in the current workflow is:

- `scGLUE` latent coordinates are used for cross-modality pairing
- raw clean chromatin bin matrices are used for locus-region scoring

This avoids being limited to the reduced chromatin feature subset retained during `scGLUE` preprocessing.

## Current cell type

The working cell type is `monocyte` at the harmonized coarse-label level.

Approximate `joint_v2` monocyte counts across modalities:

- `rna`: `1887`
- `chrom_H3K27ac`: `1577`
- `chrom_H3K27me3`: `2321`
- `chrom_H3K4me1`: `1561`
- `chrom_H3K4me2`: `2099`
- `chrom_H3K4me3`: `2390`
- `chrom_H3K9me3`: `1575`

## Current pairing strategy

The current working approach is **RNA-anchored one-to-one pseudo-pairing** inside one coarse cell type.

The `scGLUE` embedding places all modalities in a shared latent space
\(z \in \mathbb{R}^{30}\).
We then:

1. Restrict to one coarse label, currently `monocyte`.
2. Use RNA cells as anchors.
3. Rank RNA anchors by how well they are supported across all chromatin marks in GLUE space.
4. For each histone modality separately, solve a one-to-one assignment from the chosen RNA anchors to cells of that mark.

This gives a pseudo-paired table with:

\[
\text{row} = \text{RNA anchor} + \text{one matched cell from each chromatin modality}
\]

Important caveats:

- these are **pseudo-paired** samples, not true multiome cells
- no chromatin cell is reused within a given modality
- the validity of the rows depends on the quality of the `scGLUE` geometry

## What goes into the PC matrix?

Keep the graph small and interpretable.

For a target gene \(g\), prefer columns such as:

- \(Y_g\): normalized expression of \(g\)
- \(H_{g,\mathrm{mark}}^{\mathrm{promoter}}\): promoter-linked histone signal for each mark
- \(H_{g,\mathrm{mark}}^{\mathrm{distal}}\): distal or enhancer-linked histone summary for each mark when a named region is available

Do **not** use all chromatin bins directly as PC variables in the first pass.
Use a small number of literature-backed named regions per locus, then aggregate the raw clean bins that overlap each curated region into one region-mark score.

Also do **not** treat the `GLUE_*` dimensions themselves as causal nodes.
Use them for alignment and pairing, not as graph variables.

## Current human monocyte loci

Current active human monocyte validation loci:

- `CSF1R`
- `CD14`
- `IL1B`
- `CCR2`

Current literature-aligned raw-bin configs:

- `CausalDiscovery/configs/loci/csf1r_e1e5_regions.tsv`
- `CausalDiscovery/configs/loci/cd14_regions678.tsv`

Exploratory but currently secondary:

- `CEBPA`

## Current scripts

- `CausalDiscovery/scripts/build_scglue_one_to_one_matches.py`
- `CausalDiscovery/scripts/export_locus_matrix_scglue_matches.py`
- `CausalDiscovery/scripts/export_locus_panel_scglue_matches.py`
- `CausalDiscovery/scripts/run_pc_causallearn.py`
- `CausalDiscovery/scripts/plot_pc_graph.py`

## Baseline workflow

1. Filter `joint_v2` to `harmonized_coarse == monocyte`.
2. Build RNA-anchored one-to-one pseudo-pairs in `scGLUE` space.
3. Export one locus matrix per target gene from the paired cells, scoring curated regions from the raw clean chromatin bins.
4. Run baseline PC on the raw matched locus matrix.
5. Compare the learned graph against known promoter/enhancer biology before adding transforms or background knowledge.

## Example commands

Build the one-to-one monocyte matches:

```bash
python3 CausalDiscovery/scripts/build_scglue_one_to_one_matches.py \
  --run-id joint_v2 \
  --cell-type monocyte \
  --label-column harmonized_coarse
```

Export a small human monocyte panel in one pass:

```bash
python3 CausalDiscovery/scripts/export_locus_panel_scglue_matches.py \
  --run-id joint_v2 \
  --locus-config CausalDiscovery/configs/loci/cd14_regions.tsv \
  --locus-config CausalDiscovery/configs/loci/il1b_regions.tsv \
  --locus-config CausalDiscovery/configs/loci/ccr2_regions.tsv
```

Run baseline PC on one locus:

```bash
python3 CausalDiscovery/scripts/run_pc_causallearn.py \
  --matrix-tsv CausalDiscovery/outputs/locus_matrices_matched/joint_v2/cd14_pilot_rawbins/CD14_matched_locus_matrix.tsv \
  --max-depth 2
```

Optional background-knowledge modes currently supported:

- `--background-mode minimal_expr_sink`
- `--background-mode tiered_distal_promoter_expr`

Optional depth cap:

- `--max-depth <d>`
- use `--max-depth -1` for the default uncapped search

Plot the learned graph:

```bash
python3 CausalDiscovery/scripts/plot_pc_graph.py \
  --pc-dir CausalDiscovery/outputs/locus_matrices_matched/joint_v2/cd14_pilot_rawbins/pc_kci_alpha_0_05
```

## Output naming

To distinguish the old retained-feature analyses from the current hybrid workflow:

- directories ending in `_gluebins` use the retained chromatin features stored in the `scGLUE` modality `h5ad` files
- directories ending in `_rawbins` use the same one-to-one `scGLUE` pairing but recompute region scores from the raw clean chromatin bin matrices
- `csf1r_e1e5_rawbins` and `cd14_regions678_rawbins` are the current literature-aligned raw-bin runs
- PC run directories include `_depth_<d>` when a depth cap is applied

## Current interpretation stance

The current baseline intentionally uses:

- raw matched locus matrices
- default `KCI` PC
- no background-knowledge constraints
- no extra transforms beyond the exporter normalization

This is a useful first sanity check, but not the final causal-discovery setup.
If a locus looks promising under this weakly constrained baseline, it can then be revisited with stronger preprocessing or biological priors.
