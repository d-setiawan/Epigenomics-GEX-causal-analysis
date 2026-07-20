# Mixed-Family DAGMA Implementation

## Scope

This note documents the repo-local mixed-family DAGMA path for the active nearby-peak workflow in `CausalDiscovery`.

Current scope:

- nearby CUT&Tag peak matrices only
- monocyte one-to-one `scGLUE` pseudo-pairs
- one shared family for all CUT&Tag peak nodes
- one shared family for the RNA expression node

This refactor leaves the `PC` and `FCI` workflows unchanged.

## Data Requirements

Each nearby-peak matrix now exports both RNA representations:

- `expr__GENE_log1p`
- `expr__GENE_raw_counts`

and the library-size columns needed for count-family offsets:

- `libsize__rna`
- `libsize__H3K27ac`
- `libsize__H3K27me3`
- `libsize__H3K4me1`
- `libsize__H3K4me2`
- `libsize__H3K4me3`
- `libsize__H3K9me3`

Peak quantification still depends on dataset root:

- `quant_mode=log1p_norm`
  - nearby peaks are normalized and transformed with $\log(1 + x)$
- `quant_mode=raw_counts`
  - nearby peaks are raw summed counts over bins overlapping the CUT&Tag peak

## Family Model

The repo-local implementation replaces the old single global loss with a nodewise score:

$$
Q(W; X) = \sum_{j=1}^{d} Q_j\!\left(W_{\cdot j}; X, \mathrm{family}_j\right)
$$

The DAGMA acyclicity term is unchanged. The family only depends on the child node, not on the edge type.

Implemented node families:

- Gaussian with identity link
- Bernoulli with logit link
- Negative Binomial with log link

The current negative-binomial implementation uses **NB2**:

$$
\operatorname{Var}(Y) = \mu + \mu^2 / \theta
$$

## Supported Family Configurations

The public CLI exposes family configurations rather than column-by-column family assignment:

- `gaussian_gaussian`
- `bernoulli_peak_nb_gene`
- `gaussian_nb`
- `nb_nb`

These map to:

- `gaussian_gaussian`
  - peaks: Gaussian
  - gene: Gaussian
  - peak dataset root: `log1p_norm`
  - RNA column: `expr__GENE_log1p`
- `bernoulli_peak_nb_gene`
  - peaks: Bernoulli
  - gene: NB2
  - peak dataset root: `raw_counts`
  - RNA column: `expr__GENE_raw_counts`
- `gaussian_nb`
  - peaks: Gaussian
  - gene: NB2
  - peak dataset root: `log1p_norm`
  - RNA column: `expr__GENE_raw_counts`
- `nb_nb`
  - peaks: NB2
  - gene: NB2
  - peak dataset root: `raw_counts`
  - RNA column: `expr__GENE_raw_counts`

## Preprocessing Rules

### Gaussian nodes

- optional `rank_gaussian` transform
- optional standardization

### Bernoulli peak nodes

Peak columns are binarized in the runner using:

$$
x > 0 \mapsto 1,\qquad x = 0 \mapsto 0
$$

The threshold is configurable with `--dagma-peak-binary-threshold`.

### NB2 nodes

- raw counts are used directly
- a log-link mean is fit with an intercept
- offsets are supported through the exported library-size columns

Default offsets:

- RNA nodes use `libsize__rna`
- CUT&Tag peak nodes use `libsize__MARK`

## NB1 vs NB2 Decision

The first implementation target was chosen by analyzing raw matched monocyte RNA counts for `CSF1R`, `CD14`, and `IL1B`.

Outputs:

- `CausalDiscovery/outputs/analysis/joint_v2/matched_rna_count_families/matched_rna_family_summary.tsv`
- `CausalDiscovery/outputs/analysis/joint_v2/matched_rna_count_families/matched_rna_family_report.md`

That comparison used an intercept-only log-link mean with RNA library-size offset and compared:

- Poisson
- NB1 with $\\operatorname{Var}(Y) = \\mu (1 + \\alpha)$
- NB2 with $\\operatorname{Var}(Y) = \\mu + \\mu^2 / \\theta$

Recommendation:

- `CSF1R`: NB2 best by AIC
- `CD14`: NB2 best by AIC
- `IL1B`: NB2 best by AIC

So the current mixed-family DAGMA implementation uses NB2 as the count family default.

## Exactness

This mixed-family path is intended to be an exact implementation of the families above **within the current repo-local optimizer**:

- Gaussian nodes use a Gaussian score
- Bernoulli nodes use a Bernoulli logit score
- NB nodes use an NB2 log-link score

The remaining approximation is practical rather than mathematical:

- the optimizer is still a lightweight repo-local DAGMA-style implementation
- we currently use one family for all peaks and one family for the gene node

## CLI Usage

Single run:

```bash
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
  /home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py \
  graph \
  --method dagma \
  --matrix-tsv /home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --background-mode tiered_distal_promoter_expr \
  --dagma-family-config gaussian_gaussian
```

Raw-count mixed-family run:

```bash
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
  /home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py \
  graph \
  --method dagma \
  --matrix-tsv /home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --background-mode tiered_distal_promoter_expr \
  --dagma-family-config bernoulli_peak_nb_gene
```

Batch sweep:

```bash
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
  /home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/scripts/causal_cli.py \
  dagma-sweep \
  --dataset-root /home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts \
  --family-configs bernoulli_peak_nb_gene,gaussian_nb,nb_nb \
  --lambda1-values 0.03,0.01 \
  --w-threshold-values 0.3,0.1 \
  --T-values 5
```

## Validation Path

Use the staged validation path before broad sweeps:

1. `gaussian_gaussian` on the `log1p_norm` dataset root
2. `bernoulli_peak_nb_gene` on the `raw_counts` dataset root
3. `gaussian_nb`
4. `nb_nb`

The first mixed-family smoke test should stay narrow, ideally one gene and then one small curated-overlap subset, before moving to full-window batch sweeps.
