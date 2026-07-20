# CausalDiscovery

↑ [Central README](../README.md) · [MISSION](../docs/MISSION.md) (why) · [ROADMAP](../docs/ROADMAP.md) (plan) · [PIPELINE.md](PIPELINE.md) (end-to-end per-gene driver) · [Evaluation/](../Evaluation/README.md) (validation)

This directory contains the active causal-discovery workflow built from `scGLUE` `joint_v2` monocyte pseudo-pairs.

## Layout

```
CausalDiscovery/
├── scripts/     code
│   ├── causal_cli.py          the single CLI: match | dataset | graph | support
│   ├── run_gene_pipeline.py   end-to-end per-gene driver (see PIPELINE.md)
│   ├── commands/              the implementations the CLI dispatches to
│   ├── prototypes/            exploratory modeling probes, not part of the pipeline
│   └── legacy/                superseded scripts, kept for reference
├── configs/     gene panels and per-locus region definitions
├── outputs/     datasets, graphs, pipeline runs — gitignored
└── results/     modeling-design notes, literature reviews, competitive landscape
```

Validation of what this section produces lives in [`Evaluation/`](../Evaluation/README.md).

### results/ — design notes and reviews

- [element_level_modeling_design.md](results/element_level_modeling_design.md) — the element-level modeling exploration (what was tried, what was rejected and why).
- [confounder_orientation_implementation.md](results/confounder_orientation_implementation.md) — confounder roots + mechanistic orientation, as implemented.
- [competitive_landscape_novelty.md](results/competitive_landscape_novelty.md) — ranked competitors + reviewer-objection counters.
- [target_gene_selection.md](results/target_gene_selection.md) — validation-rich target ranking + cell-type pros/cons.
- [aggregation_causal_discovery_litreview.md](results/aggregation_causal_discovery_litreview.md) — why aggregation choice matters for CI-test-based discovery.
- [dagma_mixed_family_implementation.md](results/dagma_mixed_family_implementation.md) — DAGMA count-family notes (kept as sensitivity only).
- [monocyte_candidate_loci.md](results/monocyte_candidate_loci.md) · [csf1r_locus.md](results/csf1r_locus.md) — locus-level background.

Scripts reference: [scripts/README.md](scripts/README.md) · superseded code: [scripts/legacy/README.md](scripts/legacy/README.md).

## Start Here

Use the single CLI entrypoint for active work:

- `scripts/causal_cli.py`

The script tree is organized as:

- `scripts/causal_cli.py`: user-facing CLI.
- `scripts/commands/`: implementation modules called by the CLI.
- `scripts/legacy/`: older scripts retained for reference only.
- `scripts/run_dagma_log1p_gaussian.sh`: convenience DAGMA script for Gaussian/Gaussian runs.
- `scripts/run_dagma_rawcounts_family_presets.sh`: convenience DAGMA script for raw-count mixed-family presets.

## Current Workflow

The active pipeline is:

$$
\text{scGLUE joint latent space}
\rightarrow
\text{monocyte RNA-anchored pseudo-pairs}
\rightarrow
\text{nearby CUT\&Tag peak matrices}
\rightarrow
\text{locus-level graph discovery}
\rightarrow
\text{external biological support}
$$

The important design choices are:

- Pairing is done once in the `scGLUE` latent space.
- The current cell type is `monocyte`.
- Each row is one RNA-anchored pseudo-paired sample.
- Candidate graph nodes are mark-specific CUT&Tag peaks near each target gene.
- Peak signal is quantified from raw clean fixed-bin matrices.
- Curated literature regions are retained as evaluation annotations, not mixed into the graph matrices.
- External support is computed after graph discovery; it does not constrain PC,
  FCI, or DAGMA graph exploration.

## External Support Layer

The graph plotting layer can cross-reference saved edges against external
regulatory evidence and write support-aware plots:

- `eperturbdb`: local ePerturbDB Database TSV rows, using hg38 enhancer
  coordinates and `Gene (literature)` as target-gene evidence.
- `eperturbdb_enhancer_gene`: local enhancer-gene BED interval annotation.
- `eperturbdb_tf_overlap`: local enhancer-TF overlap BED context.
- `encode_pbmc`: online ENCODE Portal GRCh38 peak files for PBMC/monocyte
  biosample terms.
- `encode_screen`: online ENCODE SCREEN cCRE Registry intervals.
- `jaspar`: online JASPAR motif matrices plus UCSC hg38 sequence retrieval.

> **Cell-type alignment caveat (2026-07).** Only the ENCODE Portal source is cell-type-gated
> (use `monocyte_encode` / `monocyte_regulatory`, which query `CD14-positive monocyte`; ENCODE
> has 159 such GRCh38 peak beds). `encode_screen`, `jaspar`, and all `eperturbdb` sources apply
> **no** cell-type filter, so they can credit a monocyte edge with off-target evidence. ePerturbDB
> also has ~zero immune content and **zero `Gene (literature)` rows for CSF1R/CD14/IL1B**, so it
> cannot validate peak-to-gene edges for our genes. Net: the suite gives cell-type-aligned
> peak-node / peak-peak interval support but **cannot** validate peak-to-gene edges for our
> panel. Details and options: [Evaluation/results/run_audit_report.md](../Evaluation/results/run_audit_report.md)
> and [Evaluation/results/external_sources.md](../Evaluation/results/external_sources.md).

Support outputs are written into each graph directory:

```text
node_support.tsv
node_support_matches.tsv
edge_support.tsv
support_report.md
```

Peak-to-expression edge validation requires a peak interval match with a
matching external target gene. This prevents interval-only annotations from
overstating peak-to-gene support. Peak-to-peak edges can be supported by
interval or motif evidence from ENCODE, JASPAR, or ePerturbDB TF overlap.

Use local ePerturbDB downloads plus the monocyte regulatory online preset:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir <graph_dir> \
  --online-support \
  --support-preset monocyte_regulatory \
  --eperturbdb-dir Data/raw/external/eperturbdb \
  --layout local \
  --max-edges -1
```

For local ePerturbDB only:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir <graph_dir> \
  --online-support \
  --support-preset local_eperturbdb
```

Plot colors use orange for ePerturbDB, purple for ENCODE, green for JASPAR, red
for multi-source support, and faint gray for unvalidated edges.

## Active Inputs

Pairing root:

```text
CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor
```

The one-to-one pairing root has `1561` RNA-anchored pairs per CUT&Tag mark.

RNA-anchor neighbor-averaged pairing root:

```text
CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10
```

The neighbor-averaged pairing keeps all monocyte RNA anchors and averages each CUT&Tag mark over the $10$ nearest chromatin neighbors in scGLUE space. The current generated averaged matrices have `1887` RNA-anchor rows.

Gene panel:

```text
CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv
```

Active genes:

- `CSF1R`
- `CD14`
- `IL1B`

### Dataset Roots

```text
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

The `300kb` root uses `log1p_norm` peak quantification. The `300kb_rawcounts` root uses raw count peak quantification. These are the original one-to-one pseudo-pair nearby-peak roots.

Current RNA-anchor neighbor-averaged dataset roots:

```text
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts
```

These are the current broad-window roots for graph testing. They use mark-specific CUT&Tag peaks within TSS $\pm 1$ Mb, quantified from the raw clean fixed-bin matrices. The first root stores normalized/log-transformed peak values, while the second stores raw peak counts.

Nonzero-filtered versions:

```text
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz005
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz005
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz010
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz010
```

These keep only nearby peak columns with:

$$
p_\mathrm{nonzero} > t
$$

where the current thresholds are $t \in \{0.01, 0.05, 0.10\}$. Retained peak counts are:

| Dataset suffix | `CSF1R` | `CD14` | `IL1B` |
| --- | ---: | ---: | ---: |
| `nz001` | `60 / 69` | `99 / 113` | `74 / 82` |
| `nz005` | `41 / 69` | `73 / 113` | `49 / 82` |
| `nz010` | `26 / 69` | `40 / 113` | `28 / 82` |

Elastic-net screened version:

```text
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_elasticnet_top30_crossfit
```

This root keeps the non-peak sample, expression, and library-size columns from the unfiltered `average_k10` dataset, then keeps up to `30` nearby CUT&Tag peaks per gene selected by cross-fit elastic-net stability. Reports are written under:

```text
CausalDiscovery/outputs/methods/elastic_net_screen/joint_v2_monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10/top30_crossfit/
```

All nearby-peak matrices include:

- `expr__GENE_log1p`
- `expr__GENE_raw_counts`
- nearby peak columns
- library-size columns used by count-family DAGMA runs

Alternative RNA-anchor neighbor-averaged dataset roots:

```text
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10_rawcounts
```

These use all monocyte RNA anchors and average each CUT&Tag mark over the $10$ nearest chromatin neighbors in scGLUE space.

### Dataset Semantics

Use the dataset roots this way:

- `log1p_norm` roots are preferred for Gaussian-style graph discovery, `PC + fisherz`, and elastic-net screening.
- `rawcounts` roots are preferred for count-family DAGMA, especially Bernoulli peak or NB2 gene-expression presets.
- `nz001`, `nz005`, and `nz010` roots are structural-filtered roots. They remove sparse peak columns, but they do not use expression association, curated-region overlap, or motif evidence.
- Curated promoter/enhancer regions are evaluation annotations only. They are not graph variables in the current nearby-peak workflow.

Raw counts are not directly comparable across cells because library sizes differ. The raw-count roots therefore keep `libsize__...` columns so count-family models can use offsets. The normalized roots are generated from the same raw fixed-bin signal using library-size normalization and log transformation.

### Screening Plan

The active expression-informed peak-culling layer is association screening with elastic net stability selection. The screen fits each gene separately:

$$
\mathrm{expr}_{g,\log 1p} \sim \text{nearby CUT\&Tag peaks}
$$

on the unfiltered `log1p_norm` `average_k10` matrix. Elastic net is preferred over pure lasso because nearby peaks are correlated by genomic distance and chromatin state; pure lasso can pick one arbitrary proxy from a correlated group, while elastic net can retain groups more smoothly.

The planned screen should estimate selection frequency across repeated subsamples:

$$
\pi_j = \Pr(\hat{\beta}_j \ne 0)
$$

and keeps the top `30` peaks per gene by selection frequency, with coefficient magnitude and genomic coordinate/name as deterministic tie-breakers. This should be treated as candidate reduction, not causal evidence.

Important inference caveat:

If the same samples are used to select peaks and then run `PC`, `FCI`, or `DAGMA`, the downstream conditional-independence tests or graph scores are no longer clean post-selection inference. The graph learner is no longer operating on a pre-specified variable set. Instead, it sees a selected set:

$$
S = f(X, y)
$$

where $S$ depends on the same data later used for graph discovery. This can make downstream edges look more stable or more expression-linked than they would under a fresh independent sample.

Practical mitigations:

- treat elastic net as an exploratory candidate-reduction screen
- report the screening rule and retained peak counts
- run graph discovery across several screening thresholds
- where possible, split or cross-fit the data so one split screens peaks and another split learns the graph
- always compare screened graphs against unfiltered, `nz005`, `nz010`, and curated-region overlap annotations

Run the elastic-net screen with:

```bash
PYTHON_BIN="$(which python)" bash CausalDiscovery/scripts/run_elasticnet_screen_average_k10.sh
```

#### Clean post-selection: `--graph-split-fraction`

By default the screen keeps every row in the screened matrix, so peak selection and
graph discovery share samples (exploratory only; the existing `..._elasticnet_top30_crossfit`
runs are this kind). To get clean post-selection inference, hold out a disjoint graph split:

```bash
python3 CausalDiscovery/scripts/causal_cli.py elastic-net-screen \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10 \
  --top-k 30 \
  --graph-split-fraction 0.3
```

Peaks are selected on the `screen` split (here `1 - 0.3` of rows); the screened matrix
written for downstream `graph`/`sweep`/`dagma-sweep` contains **only** the disjoint `graph`
split rows. Selection and graph learning therefore never share samples, so the downstream
conditional-independence tests / DAGMA scores are not biased by the selection step. The split
is deterministic (`--split-random-state`), recorded per gene in
`*_elasticnet_sample_split.tsv`, and summarized under `sample_split` in each `run_summary.json`.
Output roots and report dirs gain a `_graphsplit<pct>` suffix so split runs never clobber the
shared-sample ones.

### Current Next Steps

The immediate goal is to reduce overly dense `PC` and `FCI` graphs using both structural filters and expression-informed screening. The current staged plan is:

- first compare structural nonzero filters `nz001`, `nz005`, and `nz010`
- run `PC` and `FCI` with `fisherz` on the normalized roots before using slower kernel tests
- use smaller significance thresholds such as $\alpha \in \{0.001, 0.005, 0.01\}$ when graphs remain dense
- increase conditioning depth only after node counts are lower, starting with depths $1$, $2$, and $3$
- reserve `KCI` or `fastkci` for smaller candidate sets because kernel conditional-independence tests are computationally expensive and numerically brittle on large peak matrices
- compare elastic-net screened graphs against the structural-filter baselines

If elastic-net screening is used, prefer split or cross-fit screening so that peak selection and graph discovery are not performed on the exact same sample split.

## Output Layout

Dataset matrices stay under:

```text
CausalDiscovery/outputs/datasets/<run-id>/<dataset-name>/<gene>/nearby_peaks/
```

Graph results go under:

```text
CausalDiscovery/outputs/methods/<method>/<dataset>/<gene>/<result-run>/
```

For DAGMA, the family configuration adds one extra layer:

```text
CausalDiscovery/outputs/methods/dagma/<dataset>/<family-config>/<gene>/<result-run>/
```

This keeps outputs readable as:

$$
\text{method} \rightarrow \text{dataset} \rightarrow \text{gene} \rightarrow \text{result}
$$

and, for DAGMA:

$$
\text{dagma} \rightarrow \text{dataset} \rightarrow \text{family config} \rightarrow \text{gene} \rightarrow \text{result}
$$

## Graph Methods

The active graph runner supports:

- `PC + fisherz` and `PC + KCI`
- `FCI + fisherz` and `FCI + KCI`
- repo-local mixed-family `DAGMA`

The default biological background mode is `tiered_distal_promoter_expr`, which prevents gene-to-peak edges while allowing peak-to-peak and peak-to-gene edges. This is important because expression should behave as a sink in these local epigenomic graphs.

The DAGMA implementation lives in:

- `scripts/commands/dagma_mixed_family.py`
- `scripts/commands/run_pc_causallearn.py`
- `results/dagma_mixed_family_implementation.md`

DAGMA family presets:

- `gaussian_gaussian`: Gaussian peaks and Gaussian gene expression on the `log1p_norm` root.
- `bernoulli_peak_nb_gene`: Bernoulli peaks with logit link and NB2 gene counts with log link on the `raw_counts` root.
- `gaussian_nb`: Gaussian peaks with NB2 gene counts.
- `nb_nb`: NB2 peaks with NB2 gene counts.

The `--dagma-loss-type` CLI argument remains for compatibility, but the important model choice is now `--dagma-family-config`.

## Core Commands

Build monocyte one-to-one matches:

```bash
python3 CausalDiscovery/scripts/causal_cli.py match --run-id joint_v2
```

Build RNA-anchor $k$-neighbor averaging tables:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python match-average \
  --run-id joint_v2 \
  --n-neighbors 10 \
  --neighbor-weight-mode uniform
```

Generate `300kb` nearby CUT&Tag peak datasets with log-normalized peak quantification:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --gene-panel CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv \
  --window-bp 300000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb
```

Generate the raw-count version:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --gene-panel CausalDiscovery/configs/gene_panels/monocyte_cuttag_peak_genes.tsv \
  --window-bp 300000 \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts
```

Generate an RNA-anchor neighbor-averaged dataset:

```bash
python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/bin/python dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 300000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rna_anchor_average_k10
```

Generate the current 1 Mb RNA-anchor neighbor-averaged datasets:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 1000000 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10

python3 CausalDiscovery/scripts/causal_cli.py dataset \
  --run-id joint_v2 \
  --matching-dir CausalDiscovery/outputs/scglue_pairings/joint_v2/harmonized_coarse__monocyte__rna_anchor_neighbor_average_k10 \
  --window-bp 1000000 \
  --quant-mode raw_counts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts
```

Create $p_\mathrm{nonzero} > 0.01$ filtered copies:

```bash
python3 CausalDiscovery/scripts/causal_cli.py filter-dataset \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10 \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_nz001 \
  --min-nonzero-fraction 0.01

python3 CausalDiscovery/scripts/causal_cli.py filter-dataset \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts \
  --out-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_rawcounts_nz001 \
  --min-nonzero-fraction 0.01
```

Run one `PC + KCI` graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method pc \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one `FCI + KCI` graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method fci \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --max-depth 2 \
  --background-mode tiered_distal_promoter_expr
```

Run one Gaussian DAGMA graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config gaussian_gaussian \
  --background-mode tiered_distal_promoter_expr
```

Run one raw-count mixed-family DAGMA graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --method dagma \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --dagma-family-config bernoulli_peak_nb_gene \
  --transform none \
  --background-mode tiered_distal_promoter_expr
```

## Sweeps

Run a PC/FCI depth sweep:

```bash
python3 CausalDiscovery/scripts/causal_cli.py sweep \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb \
  --methods pc,fci \
  --depths 1,2,3,4,5 \
  --background-mode tiered_distal_promoter_expr
```

Run Gaussian DAGMA sweep on log-normalized peaks:

```bash
bash CausalDiscovery/scripts/run_dagma_log1p_gaussian.sh
```

Run raw-count DAGMA family presets:

```bash
bash CausalDiscovery/scripts/run_dagma_rawcounts_family_presets.sh
```

Or call the DAGMA sweep directly:

```bash
python3 CausalDiscovery/scripts/causal_cli.py dagma-sweep \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts \
  --family-configs bernoulli_peak_nb_gene,nb_nb \
  --lambda1-values 0.03,0.01 \
  --w-threshold-values 0.3,0.1 \
  --T-values 5 \
  --background-mode tiered_distal_promoter_expr
```

`DAGMA` is intentionally excluded from the PC/FCI `sweep` because it does not use `max_depth`.

## Plotting And Evaluation

Graph runs write plots by default. Re-plot a saved graph with:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --layout local
```

Evaluation notes live in:

- `evaluation.md`
- `../Evaluation/README.md`
- `../Evaluation/results/graph_evaluation_pipeline.md`
- `../Evaluation/results/literature_review_graph_evaluation.md`

The most important current evaluation question is whether uncurated nearby peak graphs recover literature-backed promoter or enhancer regions through overlap annotations, rather than forcing those curated regions into the graph as variables.

Saved graph runs can be compared with:

```bash
python3 CausalDiscovery/scripts/causal_cli.py \
  --python-bin /home/dgsetiawan/anaconda3/bin/python \
  evaluate \
  --methods pc,fci \
  --datasets joint_v2_monocyte_cuttag_peak_genes_300kb \
  --genes cd14 \
  --out-dir Evaluation/outputs/graph_stability/cd14_pc_fci_300kb
```

This writes edge-frequency, pairwise Jaccard, orientation-consistency, and curated-region-support summaries from existing graph output directories.
